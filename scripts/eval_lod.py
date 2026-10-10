"""Evaluate Gaussian splat assets (teacher / LOD levels) on DTU: appearance + geometry.

Appearance (held-out views, gsplat Parser split test_every=8, black background,
same convention as upstream simple_trainer eval):
    PSNR / SSIM / LPIPS(alex) on the full image, and PSNR inside the reference
    object mask (projected STL).
Geometry (all views, the asset's surface as rendered):
    * depth error (mm): rendered expected depth vs. z-buffered STL depth,
      inside the STL mask where rendered alpha > 0.5; reported overall and in
      an edge band (STL silhouette + depth discontinuities, dilated 3 px)
    * DTU-style point cloud accuracy / completeness (mm, truncated at 20 mm,
      ObsMask-filtered) from back-projected rendered depth.
The STL is DTU's structured-light reference scan, not CAD ground truth.

Usage: eval_lod.py --data-dir COLMAP_DIR --stl STL.ply --obsmask ObsMask.mat \
          --out result.json NAME=asset.ply [NAME=asset.ply ...]
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scipy.io import loadmat
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/work/geolod/gsplat-upstream/examples")
from datasets.colmap import Parser  # noqa: E402
from gsplat import rasterization  # noqa: E402
from torchmetrics.image import StructuralSimilarityIndexMeasure  # noqa: E402
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: E402

DEV = "cuda"


def load_splat_ply(path):
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    g = lambda n: np.asarray(v[n], dtype=np.float32)
    n = len(v.data)
    means = np.stack([g("x"), g("y"), g("z")], 1)
    quats = np.stack([g(f"rot_{i}") for i in range(4)], 1)
    scales = np.stack([g(f"scale_{i}") for i in range(3)], 1)
    op = g("opacity")
    dc = np.stack([g(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]
    rest = sorted([k for k in names if k.startswith("f_rest_")], key=lambda s: int(s[7:]))
    if rest:  # channel-major [N, 3, K] as in the original 3DGS PLY
        r = np.stack([g(k) for k in rest], 1).reshape(n, 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, r], 1)
    else:
        sh = dc
    deg = int(round(math.sqrt(sh.shape[1]))) - 1
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
    return dict(means=t(means), quats=F.normalize(t(quats), dim=-1), scales=torch.exp(t(scales)),
                opacities=torch.sigmoid(t(op)), colors=t(sh)), deg, n


@torch.no_grad()
def render(p, deg, viewmat, K, W, H):
    out, alpha, _ = rasterization(p["means"], p["quats"], p["scales"], p["opacities"], p["colors"],
                                  viewmat[None], K[None], W, H, sh_degree=deg, render_mode="RGB+ED")
    return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], alpha[0, ..., 0]


def minpool(x, k):
    return -F.max_pool2d(-x[None, None], k, 1, k // 2)[0, 0]


def maxpool(x, k):
    return F.max_pool2d(x[None, None], k, 1, k // 2)[0, 0]


@torch.no_grad()
def stl_depth(P, viewmat, K, W, H, edge_thresh):
    """Z-buffer the STL points; 3x3 min filter closes point-splat holes."""
    pc = P @ viewmat[:3, :3].T + viewmat[:3, 3]
    pc = pc[pc[:, 2] > 1e-6]
    uv = pc @ K.T
    u = torch.floor(uv[:, 0] / uv[:, 2]).long()
    v = torch.floor(uv[:, 1] / uv[:, 2]).long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    BIG = 1e9
    zb = torch.full((H * W,), BIG, device=DEV)
    zb.scatter_reduce_(0, v[ok] * W + u[ok], pc[ok, 2], reduce="amin")
    z = minpool(zb.view(H, W), 3)
    mask = z < BIG / 2
    zm = torch.where(mask, z, torch.zeros_like(z))
    rng = maxpool(zm, 3) - minpool(torch.where(mask, z, torch.full_like(z, BIG)), 3)
    boundary = mask & (minpool(mask.float(), 3) < 0.5)
    edges = (mask & (rng > edge_thresh)) | boundary
    band = (maxpool(edges.float(), 7) > 0.5) & mask
    return torch.where(mask, z, torch.zeros_like(z)), mask, band


def stats(x):
    if x.size == 0:
        return None
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "frac_gt_1mm": float((x > 1.0).mean()), "n": int(x.size)}


def obs_filter(pts_mm, obs):
    m, bb, res = obs["ObsMask"], obs["BB"], float(obs["Res"].squeeze())
    idx = np.round((pts_mm - bb[0]) / res).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(m.shape)), 1)
    keep = np.zeros(len(pts_mm), bool)
    keep[ok] = m[idx[ok, 0], idx[ok, 1], idx[ok, 2]].astype(bool)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stl", required=True)
    ap.add_argument("--obsmask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("assets", nargs="+")
    a = ap.parse_args()

    parser = Parser(data_dir=a.data_dir, factor=1, normalize=True, test_every=a.test_every)
    T = parser.transform.astype(np.float64)                 # raw mm -> normalised
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    mm_per_unit = 1.0 / s
    Tinv = np.linalg.inv(T)
    v = PlyData.read(a.stl)["vertex"]
    stl_mm = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    stl_n = torch.from_numpy((stl_mm @ T[:3, :3].T + T[:3, 3]).astype(np.float32)).to(DEV)
    obs = loadmat(a.obsmask)
    tree_stl = cKDTree(stl_mm)
    rng = np.random.default_rng(0)
    stl_obs = stl_mm[obs_filter(stl_mm, obs)]
    stl_obs = stl_obs[rng.choice(len(stl_obs), min(300_000, len(stl_obs)), replace=False)]

    n_img = len(parser.image_names)
    val_ids = [i for i in range(n_img) if i % a.test_every == 0]
    cams = []
    for i in range(n_img):
        c2w = torch.from_numpy(parser.camtoworlds[i]).float().to(DEV)
        K = torch.from_numpy(parser.Ks_dict[parser.camera_ids[i]]).float().to(DEV)
        W, H = parser.imsize_dict[parser.camera_ids[i]]
        vm = torch.linalg.inv(c2w)
        z, m, band = stl_depth(stl_n, vm, K, W, H, edge_thresh=2.0 * s)
        cams.append(dict(i=i, vm=vm, c2w=c2w, K=K, W=W, H=H, z=z, m=m, band=band))
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEV)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(DEV)
    gts = {i: torch.from_numpy(imageio.imread(parser.image_paths[i])[..., :3]).float().to(DEV) / 255
           for i in val_ids}

    meta = {"data_dir": a.data_dir, "stl": a.stl, "mm_per_unit": mm_per_unit,
            "val_ids": val_ids, "num_views": n_img,
            "stl_mask_coverage_mean": float(np.mean([c["m"].float().mean().item() for c in cams])),
            "edge_band_frac_of_mask": float(np.mean([(c["band"].sum() / c["m"].sum()).item() for c in cams]))}
    results = {"meta": meta, "assets": {}}
    for spec in a.assets:
        name, path = spec.split("=", 1)
        t0 = time.time()
        p, deg, n = load_splat_ply(path)
        app = {"psnr": [], "psnr_mask": [], "ssim": [], "lpips": []}
        derr, derr_band, cloud = [], [], []
        for c in cams:
            rgb, dep, alp = render(p, deg, c["vm"], c["K"], c["W"], c["H"])
            if c["i"] in gts:
                gt = gts[c["i"]]
                mse = ((rgb - gt) ** 2).mean()
                mm = c["m"].float()[..., None]
                mse_m = (((rgb - gt) ** 2) * mm).sum() / (mm.sum() * 3)
                app["psnr"].append(float(-10 * torch.log10(mse)))
                app["psnr_mask"].append(float(-10 * torch.log10(mse_m)))
                x, y = rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
                app["ssim"].append(float(ssim(x, y)))
                app["lpips"].append(float(lpips(x, y)))
            valid = c["m"] & (alp > 0.5)
            e = ((dep - c["z"]).abs() * mm_per_unit)
            derr.append(e[valid].cpu().numpy())
            derr_band.append(e[valid & c["band"]].cpu().numpy())
            # back-project rendered depth inside the (dilated) reference mask
            sel = (maxpool(c["m"].float(), 11) > 0.5) & (alp > 0.5)
            vv, uu = torch.nonzero(sel, as_tuple=True)
            zz = dep[vv, uu]
            K = c["K"]
            xc = torch.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * zz, (vv + 0.5 - K[1, 2]) / K[1, 1] * zz, zz], 1)
            xw = xc @ c["c2w"][:3, :3].T + c["c2w"][:3, 3]
            cloud.append(xw.cpu().numpy().astype(np.float64))
        cloud = np.concatenate(cloud)
        cloud = cloud[rng.choice(len(cloud), min(1_000_000, len(cloud)), replace=False)]
        cloud_mm = cloud @ Tinv[:3, :3].T + Tinv[:3, 3]
        cloud_mm = cloud_mm[obs_filter(cloud_mm, obs)]
        acc, _ = tree_stl.query(cloud_mm, distance_upper_bound=20.0)
        acc = np.minimum(acc, 20.0)
        comp, _ = cKDTree(cloud_mm).query(stl_obs, distance_upper_bound=20.0)
        comp = np.minimum(comp, 20.0)
        de, db = np.concatenate(derr), np.concatenate(derr_band)
        res = {
            "path": path, "num_gaussians": n, "file_bytes": os.path.getsize(path), "sh_degree": deg,
            "appearance": {k: float(np.mean(vs)) for k, vs in app.items()},
            "appearance_per_view": app,
            "depth_err_mm": stats(de), "depth_err_edge_mm": stats(db),
            "dtu_accuracy_mm": float(acc.mean()), "dtu_completeness_mm": float(comp.mean()),
            "dtu_overall_mm": float((acc.mean() + comp.mean()) / 2),
            "accuracy_p95_mm": float(np.percentile(acc, 95)),
            "eval_seconds": time.time() - t0,
        }
        results["assets"][name] = res
        print(name, json.dumps({k: res[k] for k in ("num_gaussians", "appearance", "dtu_overall_mm")}),
              "depth_p95", res["depth_err_mm"]["p95"], "edge_p95", res["depth_err_edge_mm"]["p95"], flush=True)
        del p
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
