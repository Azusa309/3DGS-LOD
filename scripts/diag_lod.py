"""Diagnostic (hypothesis-level, not a result): is the LOD1 PSNR drop caused by
(a) splat-transform decimation itself, or (b) the Streamed SOG encode/decode?
Also prints PLY header comments (model tags) and depth-error medians.
Renders one held-out view of each asset with the same gsplat call as eval_lod.py.
"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/root/work/geolod/scripts")
sys.path.insert(0, "/root/work/geolod/gsplat-upstream/examples")
from eval_lod import load_splat_ply, render  # noqa: E402
from datasets.colmap import Parser  # noqa: E402
import imageio.v2 as imageio

assets = dict(a.split("=", 1) for a in sys.argv[1:])
for name, path in assets.items():
    with open(path, "rb") as f:
        head = f.read(4096).split(b"end_header")[0].decode(errors="ignore")
    print(name, "header comments:", [l for l in head.splitlines() if l.startswith(("comment", "obj_info"))])

parser = Parser(data_dir="/root/datasets/dtu-scan6/colmap", factor=1, normalize=True, test_every=8)
res = {}
for vid in (0, 8, 16):
    c2w = torch.from_numpy(parser.camtoworlds[vid]).float().cuda()
    K = torch.from_numpy(parser.Ks_dict[parser.camera_ids[vid]]).float().cuda()
    W, H = parser.imsize_dict[parser.camera_ids[vid]]
    gt = torch.from_numpy(imageio.imread(parser.image_paths[vid])[..., :3]).float().cuda() / 255
    for name, path in assets.items():
        p, deg, n = load_splat_ply(path)
        rgb, dep, alp = render(p, deg, torch.linalg.inv(c2w), K, W, H)
        psnr = float(-10 * torch.log10(((rgb - gt) ** 2).mean()))
        res.setdefault(name, []).append(round(psnr, 2))
        del p
        torch.cuda.empty_cache()
print(json.dumps(res))
