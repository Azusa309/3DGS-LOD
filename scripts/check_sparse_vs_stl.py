"""3D frame check: distance from COLMAP-triangulated points (fixed DTU poses) to the STL reference.

If poses and STL share the DTU world frame (mm), most sparse points lie within a few mm of the STL.
Random points in the same bbox give the null baseline.
"""
import json, sys
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

pts = np.load(sys.argv[1])
v = PlyData.read(sys.argv[2])["vertex"]
stl = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
tree = cKDTree(stl)
d, _ = tree.query(pts)
lo, hi = stl.min(0), stl.max(0)
inside = np.all((pts >= lo) & (pts <= hi), 1)
rnd = np.random.default_rng(0).uniform(lo, hi, size=(20000, 3))
dr, _ = tree.query(rnd)
rep = {
    "num_sparse": int(len(pts)), "num_sparse_in_stl_bbox": int(inside.sum()),
    "stl_bbox_mm": [lo.tolist(), hi.tolist()],
    "dist_in_bbox_mm": {q: float(np.percentile(d[inside], p)) for q, p in [("p50", 50), ("p90", 90), ("p95", 95)]},
    "frac_in_bbox_within_2mm": float((d[inside] < 2).mean()),
    "random_baseline_p50_mm": float(np.median(dr)),
}
json.dump(rep, open(sys.argv[3], "w"), indent=2); print(json.dumps(rep, indent=2))
