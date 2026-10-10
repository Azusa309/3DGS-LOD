"""Check that the DTU STL reference and the calibrated cameras share one frame.

Projects a random subset of STL points with the official 3x4 projection
matrices and reports, per view, the fraction of projected points that land
inside the image foreground mask (non-white pixels). A correct frame gives a
high in-mask ratio on every view; a wrong frame gives near-random ratios.
This is a sanity check of the coordinate chain, not a geometry metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--scan", type=int, default=6)
    ap.add_argument("--light", type=int, default=3)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    stl = args.data_root / "Points" / "stl" / f"stl{args.scan:03d}_total.ply"
    v = PlyData.read(str(stl))["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    rng = np.random.default_rng(0)
    pts = pts[rng.choice(len(pts), size=min(args.samples, len(pts)), replace=False)]
    homo = np.hstack([pts, np.ones((len(pts), 1))])

    img_root = args.data_root / "Rectified" / f"scan{args.scan}"
    cal_root = args.data_root / "Calibration" / "cal18"
    rows = []
    for p in sorted(img_root.glob(f"rect_*_{args.light}_r5000.png")):
        vid = int(p.name.split("_")[1])
        P = np.loadtxt(cal_root / f"pos_{vid:03d}.txt")
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        fg = np.linalg.norm(rgb - 1.0, axis=-1) > 0.15
        h, w = fg.shape
        uvw = homo @ P.T
        front = uvw[:, 2] > 0
        uv = uvw[front, :2] / uvw[front, 2:3]
        u, vv = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inside = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        in_mask = fg[vv[inside], u[inside]].mean() if inside.any() else 0.0
        rows.append({"view_id": vid, "in_image": float(inside.mean()), "in_mask": float(in_mask),
                     "fg_fraction": float(fg.mean())})
    in_mask = np.array([r["in_mask"] for r in rows])
    fg_frac = np.array([r["fg_fraction"] for r in rows])
    report = {
        "stl": str(stl), "num_points_sampled": int(len(pts)), "num_views": len(rows),
        "in_mask_min": float(in_mask.min()), "in_mask_median": float(np.median(in_mask)),
        "random_baseline_median": float(np.median(fg_frac)),
        "views": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "views"}, indent=2))


if __name__ == "__main__":
    main()
