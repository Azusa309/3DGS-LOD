"""Check lodkit.colmap.Scene reproduces gsplat examples' Parser exactly.

A teacher trained by gsplat simple_trainer lives in Parser's normalised frame,
so lodkit must match it: image order, c2w, K, image size, world transform and
the held-out split. Run with gsplat's examples on sys.path.
Usage: check_scene_equivalence.py <colmap_dir> <gsplat_examples_dir> <out.json>
"""
import json
import sys

import numpy as np

data_dir, examples, out = sys.argv[1:4]
sys.path.insert(0, examples)
from datasets.colmap import Parser  # noqa: E402
from lodkit.colmap import Scene  # noqa: E402

ref = Parser(data_dir=data_dir, factor=1, normalize=True, test_every=8)
sc = Scene(data_dir, test_every=8, normalize=True)

names_ok = [v.name for v in sc.views] == list(ref.image_names)
c2w_err = max(float(np.abs(v.c2w - ref.camtoworlds[i]).max()) for i, v in enumerate(sc.views))
K_err = max(float(np.abs(v.K - ref.Ks_dict[ref.camera_ids[i]]).max()) for i, v in enumerate(sc.views))
size_ok = all((v.width, v.height) == tuple(ref.imsize_dict[ref.camera_ids[i]]) for i, v in enumerate(sc.views))
T_err = float(np.abs(sc.transform - ref.transform).max())
n = len(ref.image_names)
split_ok = [v.index for v in sc.test_views] == [i for i in range(n) if i % 8 == 0]
paths_ok = all(v.image_path == ref.image_paths[i] for i, v in enumerate(sc.views))

rep = {"data_dir": data_dir, "num_views": n, "names_order_equal": names_ok,
       "max_abs_c2w_diff": c2w_err, "max_abs_K_diff": K_err, "image_size_equal": size_ok,
       "max_abs_transform_diff": T_err, "test_split_equal": split_ok, "image_paths_equal": paths_ok}
rep["pass"] = bool(names_ok and size_ok and split_ok and paths_ok
                   and c2w_err < 1e-6 and K_err < 1e-9 and T_err < 1e-6)
json.dump(rep, open(out, "w"), indent=2)
print(json.dumps(rep, indent=2))
sys.exit(0 if rep["pass"] else 1)
