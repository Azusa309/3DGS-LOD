"""Export real camera poses for the demo (no GPU needed).

Reads the COLMAP scene the teacher was trained on, in gsplat's normalised
frame (via lodkit.colmap.Scene, verified identical to gsplat's Parser), and
writes demo/cameras.json with the held-out views:
    c2w (4x4, OpenCV: x right, y down, z forward), fov_y (deg), width, height, image name.
The page converts OpenCV -> PlayCanvas camera axes with diag(1, -1, -1).

Usage: python demo/export_cameras.py <colmap_dir> demo/cameras.json
"""
import json
import math
import sys

import numpy as np

from lodkit.colmap import Scene

data_dir, out = sys.argv[1], sys.argv[2]
scene = Scene(data_dir, test_every=8)
pts = scene.points
cams = []
for v in scene.test_views:
    fov_y = 2 * math.degrees(math.atan(v.height / (2 * v.K[1, 1])))
    cams.append({"name": v.name, "c2w": v.c2w.tolist(), "fov_y": fov_y,
                 "width": v.width, "height": v.height})
# robust object centre for the orbit pivot: median of SfM points
center = np.median(pts, axis=0).tolist()
with open(out, "w") as f:
    json.dump({"frame": "gsplat normalised, OpenCV camera axes", "center": center, "views": cams}, f, indent=1)
print(f"wrote {len(cams)} views, center={np.round(center, 3).tolist()}")
