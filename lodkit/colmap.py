"""Minimal COLMAP binary reader + the world normalisation used by gsplat's trainer.

Why not pycolmap: two incompatible packages are both named ``pycolmap``
(the official COLMAP bindings and the rmbrualla fork gsplat's examples use);
installing one silently shadows the other. This reader has no such conflict.

Equivalence contract: a teacher trained with gsplat ``simple_trainer.py``
(``normalize_world_space=True``) lives in the normalised frame produced below.
``Scene`` must reproduce gsplat ``examples/datasets/colmap.py::Parser`` exactly:
images sorted by name, test split ``i % test_every == 0``, same transform.
The normalisation functions are adapted from gsplat (Apache-2.0,
nerfstudio-project/gsplat, examples/datasets/normalize.py).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, mid, w, h = _read(f, "iiQQ")
            name, k = CAMERA_MODELS[mid]
            cams[cid] = dict(model=name, width=w, height=h, params=np.array(_read(f, "d" * k)))
    return cams


def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = _read(f, "Q")
            f.seek(npts * 24, 1)  # (x, y, point3D_id) per keypoint, unused
            imgs[iid] = dict(name=name.decode(), q=np.array([qw, qx, qy, qz]), t=np.array([tx, ty, tz]), camera_id=cid)
    return imgs


def read_points3d_bin(path):
    xyz = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            _id, x, y, z, _r, _g, _b, _err, tl = _read(f, "QdddBBBdQ")
            f.seek(tl * 8, 1)
            xyz.append((x, y, z))
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


# ---- gsplat normalisation (adapted, Apache-2.0) --------------------------------
def similarity_from_cameras(c2w):
    t, R = c2w[:, :3, 3], c2w[:, :3, :3]
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    up_cam = np.array([0.0, -1.0, 0.0])
    c = (up_cam * world_up).sum()
    cross = np.cross(world_up, up_cam)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    R_align = np.eye(3) + skew + (skew @ skew) / (1 + c) if c > -1 else np.diag([-1.0, 1.0, 1.0])
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
    translate = -np.median(nearest, axis=0)
    T = np.eye(4)
    T[:3, 3], T[:3, :3] = translate, R_align
    T[:3, :] *= 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return T


def align_principal_axes(pts):
    centroid = np.median(pts, axis=0)
    evals, evecs = np.linalg.eigh(np.cov(pts - centroid, rowvar=False))
    evecs = evecs[:, evals.argsort()[::-1]]
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = evecs.T
    T[:3, 3] = -evecs.T @ centroid
    return T


def transform_points(M, pts):
    return pts @ M[:3, :3].T + M[:3, 3]


def transform_cameras(M, c2w):
    c2w = np.einsum("nij, ki -> nkj", c2w, M)
    s = np.linalg.norm(c2w[:, 0, :3], axis=1)
    c2w[:, :3, :3] = c2w[:, :3, :3] / s[:, None, None]
    return c2w


@dataclass
class View:
    index: int
    name: str
    image_path: str
    c2w: np.ndarray       # 4x4, normalised frame, OpenCV convention
    K: np.ndarray         # 3x3
    width: int
    height: int


class Scene:
    """COLMAP scene in gsplat's normalised world frame."""

    def __init__(self, data_dir: str, test_every: int = 8, normalize: bool = True):
        sp = os.path.join(data_dir, "sparse", "0")
        if not os.path.isdir(sp):
            sp = os.path.join(data_dir, "sparse")
        cams = read_cameras_bin(os.path.join(sp, "cameras.bin"))
        imgs = read_images_bin(os.path.join(sp, "images.bin"))
        pts = read_points3d_bin(os.path.join(sp, "points3D.bin"))
        order = sorted(imgs.values(), key=lambda im: im["name"])
        w2c = np.stack([np.block([[qvec2rotmat(im["q"]), im["t"][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
                        for im in order])
        c2w = np.linalg.inv(w2c)
        T = np.eye(4)
        if normalize:
            T1 = similarity_from_cameras(c2w)
            c2w, pts = transform_cameras(T1, c2w), transform_points(T1, pts)
            T2 = align_principal_axes(pts)
            c2w, pts = transform_cameras(T2, c2w), transform_points(T2, pts)
            T = T2 @ T1
            if np.median(pts[:, 2]) > np.mean(pts[:, 2]):
                T3 = np.diag([1.0, -1.0, -1.0, 1.0])
                c2w, pts = transform_cameras(T3, c2w), transform_points(T3, pts)
                T = T3 @ T
        self.transform = T            # raw COLMAP world -> normalised
        self.points = pts
        self.test_every = test_every
        self.views = []
        for i, im in enumerate(order):
            cam = cams[im["camera_id"]]
            p = cam["params"]
            if cam["model"] == "PINHOLE":
                fx, fy, cx, cy = p
            elif cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = p[0]
                cx, cy = p[1], p[2]
                if cam["model"] != "SIMPLE_PINHOLE" and np.any(np.abs(p[3:]) > 0):
                    raise ValueError("distorted cameras are not supported; undistort with COLMAP first")
            else:
                raise ValueError(f"camera model {cam['model']} not supported; undistort to PINHOLE first")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            self.views.append(View(i, im["name"], os.path.join(data_dir, "images", im["name"]),
                                   c2w[i], K, int(cam["width"]), int(cam["height"])))

    @property
    def train_views(self):
        return [v for v in self.views if v.index % self.test_every != 0]

    @property
    def test_views(self):
        return [v for v in self.views if v.index % self.test_every == 0]
