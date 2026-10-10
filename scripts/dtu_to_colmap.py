"""Convert a DTU SampleSet scan into a COLMAP sparse model for upstream gsplat.

Why: upstream gsplat ``examples/simple_trainer.py`` (the real default trainer)
reads COLMAP ``sparse/0`` and initialises Gaussians from SfM points
(``init_type="sfm"``). DTU ships calibrated 3x4 projection matrices, so we keep
the official poses/intrinsics fixed and only triangulate points with COLMAP
(SIFT + exhaustive matching + ``triangulate_points``). No pose is re-estimated.

Coordinates stay in raw DTU world units (millimetres), the same frame as the
official STL reference scan. gsplat normalises the world internally; the
evaluation step recovers that transform from the gsplat Parser.

Output layout (what gsplat's Parser expects):
    <out>/images/rect_XXX.png     downscaled RGB
    <out>/sparse/0/{cameras,images,points3D}.bin
    <out>/dtu_colmap_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

FRAME_RE = re.compile(r"rect_(\d{3})_(\d+)_r5000\.png$")


def decompose_projection(P: np.ndarray):
    """RQ-decompose P = K [R | t]; returns K (3x3), R (3x3), t (3,) world->camera."""
    P = np.asarray(P, dtype=np.float64)
    flip = np.flipud(np.eye(3))
    q, r = np.linalg.qr((flip @ P[:, :3]).T)
    K = flip @ r.T @ flip
    R = flip @ q.T
    D = np.diag(np.where(np.sign(np.diag(K)) == 0, 1, np.sign(np.diag(K))))
    K, R = K @ D, D @ R
    if np.linalg.det(R) < 0:
        K, R = -K, -R
    K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "invalid rotation"
    # reprojection self-check: K[R|t] must equal P up to scale
    P2 = K @ np.hstack([R, t[:, None]])
    s = P[2, 3] / P2[2, 3] if abs(P2[2, 3]) > 1e-12 else 1.0
    assert np.allclose(P2 * s, P, rtol=1e-5, atol=1e-6 * np.abs(P).max()), "K[R|t] != P"
    return K, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scan", type=int, default=6)
    ap.add_argument("--light", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    img_root = args.data_root / "Rectified" / f"scan{args.scan}"
    cal_root = args.data_root / "Calibration" / "cal18"
    paths = sorted(img_root.glob(f"rect_*_{args.light}_r5000.png"))
    if not paths:
        raise FileNotFoundError(img_root)
    out = args.output
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    sparse = out / "sparse" / "0"
    sparse.mkdir(parents=True)

    views = []
    for p in paths:
        vid = int(FRAME_RE.search(p.name).group(1))
        K, R, t = decompose_projection(np.loadtxt(cal_root / f"pos_{vid:03d}.txt"))
        im = Image.open(p).convert("RGB")
        w, h = round(im.width * args.scale), round(im.height * args.scale)
        im = im.resize((w, h), Image.Resampling.LANCZOS)
        name = f"rect_{vid:03d}.png"
        im.save(out / "images" / name)
        # pixel-centre convention: scaling a K whose principal point is in
        # pixel units by s keeps the same convention used by DTU/COLMAP.
        Ks = K.copy()
        Ks[:2] *= args.scale
        views.append(dict(name=name, view_id=vid, K=Ks, R=R, t=t, w=w, h=h))

    # 1) SIFT features + exhaustive matching into a COLMAP database
    db_path = out / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    pycolmap.extract_features(db_path, out / "images", camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              reader_options=reader)
    pycolmap.match_exhaustive(db_path)

    # 2) Overwrite database cameras with the calibrated intrinsics
    by_name = {v["name"]: v for v in views}
    db = pycolmap.Database.open(db_path)
    db_images = db.read_all_images()
    db_cams = {c.camera_id: c for c in db.read_all_cameras()}
    for img in db_images:
        v = by_name[img.name]
        cam = db_cams[img.camera_id]
        cam.model = "PINHOLE"
        cam.width, cam.height = v["w"], v["h"]
        cam.params = [v["K"][0, 0], v["K"][1, 1], v["K"][0, 2], v["K"][1, 2]]
        db.update_camera(cam)
    db.close()

    # 3) Reconstruction with fixed known poses, then triangulate
    rec = pycolmap.Reconstruction()
    for img in db_images:
        v = by_name[img.name]
        cam = db_cams[img.camera_id]
        rec.add_camera_with_trivial_rig(cam)
        image = pycolmap.Image(name=img.name, camera_id=img.camera_id, image_id=img.image_id)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(v["R"]), v["t"].reshape(3, 1))
        rec.add_image_with_trivial_frame(image, pose)
    rec = pycolmap.triangulate_points(rec, db_path, out / "images", sparse, clear_points=True,
                                      refine_intrinsics=False)
    rec.write(sparse)

    # sanity: poses unchanged, reprojection error small
    max_dc = 0.0
    for img in rec.images.values():
        v = by_name[img.name]
        c_ref = -v["R"].T @ v["t"]
        max_dc = max(max_dc, float(np.linalg.norm(img.projection_center() - c_ref)))
    pts = np.array([p.xyz for p in rec.points3D.values()])
    errs = np.array([p.error for p in rec.points3D.values()])
    manifest = {
        "source": "DTU MVS SampleSet",
        "scan": args.scan, "light": args.light, "image_scale": args.scale,
        "units": "DTU world coordinates (mm), same frame as STL reference",
        "num_images": rec.num_reg_images(), "num_points3D": int(len(pts)),
        "mean_reproj_error_px": float(errs.mean()) if len(errs) else None,
        "max_camera_center_drift_mm": max_dc,
        "points_bbox_mm": [pts.min(0).tolist(), pts.max(0).tolist()] if len(pts) else None,
        "views": [{"name": v["name"], "view_id": v["view_id"]} for v in views],
    }
    (out / "dtu_colmap_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in manifest if k != "views"}, indent=2))


if __name__ == "__main__":
    main()
