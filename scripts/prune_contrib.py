"""Control baseline: prune by occlusion-aware contribution (sum of T*alpha over
training views), no merging, no fine-tuning.

Purpose: tell apart "this teacher is fragile to removing Gaussians" from
"splat-transform's merge decimation loses quality". Training views only
(i % test_every != 0), so held-out evaluation views are never used.

Contribution trick: render a 1-channel feature that is zero for every Gaussian
with requires_grad. Alpha compositing is linear in features, so
d(sum of image)/d f_i = sum over pixels of T_i * alpha_i, exactly, including
occlusion, without storing a dense pixel x Gaussian table.

Usage: prune_contrib.py --data-dir COLMAP --teacher teacher.ply --out-dir DIR \
           --keep 0.5,0.25,0.1
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from plyfile import PlyData, PlyElement

sys.path.insert(0, "/root/work/geolod/scripts")
sys.path.insert(0, "/root/work/geolod/gsplat-upstream/examples")
from eval_lod import load_splat_ply  # noqa: E402
from datasets.colmap import Parser  # noqa: E402
from gsplat import rasterization  # noqa: E402


def contributions(p, parser, test_every):
    n = p["means"].shape[0]
    score = torch.zeros(n, device="cuda", dtype=torch.float64)
    views = [i for i in range(len(parser.image_names)) if i % test_every != 0]
    for i in views:
        c2w = torch.from_numpy(parser.camtoworlds[i]).float().cuda()
        K = torch.from_numpy(parser.Ks_dict[parser.camera_ids[i]]).float().cuda()
        W, H = parser.imsize_dict[parser.camera_ids[i]]
        f = torch.zeros(n, 1, device="cuda", requires_grad=True)
        img, _, _ = rasterization(p["means"], p["quats"], p["scales"], p["opacities"], f,
                                  torch.linalg.inv(c2w)[None], K[None], W, H)
        (g,) = torch.autograd.grad(img.sum(), f)
        score += g[:, 0].double()
    return score, views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep", default="0.5,0.25,0.1")
    ap.add_argument("--test-every", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    parser = Parser(data_dir=a.data_dir, factor=1, normalize=True, test_every=a.test_every)
    t0 = time.time()
    p, _, n = load_splat_ply(a.teacher)
    score, views = contributions(p, parser, a.test_every)
    t_score = time.time() - t0
    order = torch.argsort(score, descending=True).cpu().numpy()
    v = PlyData.read(a.teacher)["vertex"]
    info = {"teacher": a.teacher, "num_gaussians": n, "train_views": len(views),
            "score_seconds": t_score, "zero_contribution_frac": float((score == 0).float().mean()),
            "levels": []}
    for li, k in enumerate([float(x) for x in a.keep.split(",")], start=1):
        m = int(round(n * k))
        keep = np.sort(order[:m])
        out = os.path.join(a.out_dir, f"lod{li}.ply")
        PlyData([PlyElement.describe(v.data[keep], "vertex")]).write(out)
        info["levels"].append({"lod": li, "keep": k, "count": m,
                               "retained_score_frac": float(score[torch.from_numpy(keep).cuda()].sum() / score.sum())})
    json.dump(info, open(os.path.join(a.out_dir, "prune_info.json"), "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
