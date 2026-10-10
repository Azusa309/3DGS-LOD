"""M1: contribution pruning + geometry-gated merge + fixed-geometry attribute recovery.

Pipeline for one budget (fraction `keep` of the teacher):
  1. score every Gaussian by occlusion-aware contribution sum(T*alpha) over
     TRAINING views (held-out views i % test_every == 0 are never used);
  2. keep the top-k; every removed Gaussian is assigned to its nearest kept
     Gaussian and merged into it ONLY IF
        - normals agree: |n_r . n_k| > cos_gate (normal = shortest axis), and
        - it is close: dist < dist_gate * max_scale(kept);
     otherwise it is dropped. Merge = contribution-weighted moment matching
     of mean/covariance and SH; the kept Gaussian's opacity is kept. This
     forbids merges across thin walls / folds (the failure mode of
     appearance-only merge costs);
  3. optional: fixed-geometry recovery -- optimise only SH and opacity on
     training views (L1 + 0.2 D-SSIM, gsplat default weights). Means, scales
     and rotations are frozen, so no new geometry is invented.

Any input PLY can be refined alone (--refine-only) so the same recovery can be
applied to the official baseline for a fair comparison.

Usage:
  merge_lod.py --data-dir COLMAP --teacher T.ply --keep 0.1 --out out.ply [--no-merge] [--refine 1000]
  merge_lod.py --data-dir COLMAP --refine-only in.ply --out out.ply --refine 1000
"""
import argparse, json, math, os, sys, time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/work/geolod/scripts")
sys.path.insert(0, "/root/work/geolod/gsplat-upstream/examples")
from eval_lod import load_splat_ply  # noqa: E402
from prune_contrib import contributions  # noqa: E402
from datasets.colmap import Parser  # noqa: E402
from gsplat import rasterization  # noqa: E402
from fused_ssim import fused_ssim  # noqa: E402


def quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def rot_to_quat(R):
    """Rotation matrix -> unit quaternion (w,x,y,z), Bar-Itzhack method.

    The quaternion is the eigenvector of the largest eigenvalue of a symmetric
    4x4 matrix built from R; robust for every rotation (no trace branches).
    """
    m = R.double()
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    K = torch.stack([
        m00 - m11 - m22, m10 + m01, m20 + m02, m21 - m12,
        m10 + m01, m11 - m00 - m22, m21 + m12, m02 - m20,
        m20 + m02, m21 + m12, m22 - m00 - m11, m10 - m01,
        m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22], -1).view(-1, 4, 4) / 3.0
    # CPU float64: batched CUDA eigh asks cuSOLVER for a huge workspace on
    # millions of tiny matrices (51 GB OOM observed); CPU LAPACK is fast here.
    _, vecs = torch.linalg.eigh(K.cpu())
    xyzw = vecs[:, :, -1].to(R.device)
    q = torch.stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]], -1)
    return F.normalize(q, dim=-1).to(R.dtype)


def write_ply(path, p, sh_rest_k):
    """Write activated params back to the standard 3DGS PLY convention."""
    n = p["means"].shape[0]
    sh = p["colors"].detach().cpu().numpy()                       # [N, K, 3]
    fields = {"x": p["means"][:, 0], "y": p["means"][:, 1], "z": p["means"][:, 2]}
    # No nx/ny/nz: gsplat's export omits them, and splat-transform refuses to
    # stack LOD levels whose PLY layers differ ("available-layer mismatch").
    cols = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    cols += [(f"f_dc_{i}", "f4") for i in range(3)]
    cols += [(f"f_rest_{i}", "f4") for i in range(3 * sh_rest_k)]
    cols += [("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)]
    arr = np.zeros(n, dtype=cols)
    for k, t in fields.items():
        arr[k] = t.detach().cpu().numpy()
    for i in range(3):
        arr[f"f_dc_{i}"] = sh[:, 0, i]
    if sh_rest_k:
        rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)       # channel-major
        for i in range(rest.shape[1]):
            arr[f"f_rest_{i}"] = rest[:, i]
    op = p["opacities"].detach().clamp(1e-6, 1 - 1e-6)
    arr["opacity"] = torch.logit(op).cpu().numpy()
    ls = torch.log(p["scales"].detach().clamp_min(1e-12)).cpu().numpy()
    for i in range(3):
        arr[f"scale_{i}"] = ls[:, i]
    q = F.normalize(p["quats"].detach(), dim=-1).cpu().numpy()
    for i in range(4):
        arr[f"rot_{i}"] = q[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def subset(p, idx):
    return {k: v[idx].contiguous() for k, v in p.items()}


@torch.no_grad()
def gated_merge(p, score, keep_idx, cos_gate, dist_gate):
    n = p["means"].shape[0]
    kept = torch.zeros(n, dtype=torch.bool, device="cuda")
    kept[keep_idx] = True
    rem = torch.nonzero(~kept).squeeze(1)
    R = quat_to_rot(p["quats"])
    s = p["scales"]
    normal = torch.gather(R, 2, s.argmin(1)[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    km = p["means"][keep_idx].cpu().numpy()
    _, nn = cKDTree(km).query(p["means"][rem].cpu().numpy(), k=1, workers=-1)
    tgt = keep_idx[torch.from_numpy(nn).cuda()]
    d = (p["means"][rem] - p["means"][tgt]).norm(dim=-1)
    cosv = (normal[rem] * normal[tgt]).sum(-1).abs()
    ok = (cosv > cos_gate) & (d < dist_gate * s[tgt].max(-1).values) & (score[rem] > 0)
    src, dst = rem[ok], tgt[ok]
    # moment matching, contribution-weighted (kept Gaussian weighted by its own score)
    w_all = score.float().clamp_min(1e-12)
    W = torch.zeros(n, device="cuda").index_add_(0, dst, w_all[src])
    W[keep_idx] += w_all[keep_idx]
    mu = p["means"] * w_all[:, None]
    MU = torch.zeros_like(p["means"]).index_add_(0, dst, mu[src])
    MU[keep_idx] += mu[keep_idx]
    MU = MU / W[:, None].clamp_min(1e-20)
    cov = R @ torch.diag_embed(s ** 2) @ R.transpose(1, 2)
    def second(ids, base_mu):
        dm = p["means"][ids] - base_mu
        return (cov[ids] + dm[:, :, None] * dm[:, None, :]) * w_all[ids, None, None]
    C = torch.zeros_like(cov).index_add_(0, dst, second(src, MU[dst]))
    C[keep_idx] += second(keep_idx, MU[keep_idx])
    C = C / W[:, None, None].clamp_min(1e-20)
    sh = p["colors"] * w_all[:, None, None]
    SH = torch.zeros_like(p["colors"]).index_add_(0, dst, sh[src])
    SH[keep_idx] += sh[keep_idx]
    SH = SH / W[:, None, None].clamp_min(1e-20)
    k = keep_idx
    # small 3x3 eigh on CPU: batched CUDA eigh allocates a huge cuSOLVER workspace
    evals, evecs = torch.linalg.eigh(C[k].double().cpu())
    evals, evecs = evals.cuda(), evecs.cuda()
    evecs = evecs * torch.sign(torch.det(evecs))[:, None, None]
    out = {"means": MU[k], "scales": evals.clamp_min(1e-16).sqrt().float(),
           "quats": rot_to_quat(evecs.float()), "opacities": p["opacities"][k], "colors": SH[k]}
    merged_into = torch.zeros(n, device="cuda").index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    return out, {"num_removed": int(rem.numel()), "num_merged": int(src.numel()),
                 "frac_removed_merged": float(src.numel() / max(rem.numel(), 1)),
                 "kept_receiving_merges": int((merged_into[k] > 0).sum())}


def refine(p, deg, parser, test_every, steps, seed=0):
    """Optimise SH + opacity only; geometry frozen."""
    torch.manual_seed(seed)
    views = [i for i in range(len(parser.image_names)) if i % test_every != 0]
    cams = []
    for i in views:
        c2w = torch.from_numpy(parser.camtoworlds[i]).float().cuda()
        K = torch.from_numpy(parser.Ks_dict[parser.camera_ids[i]]).float().cuda()
        W, H = parser.imsize_dict[parser.camera_ids[i]]
        gt = torch.from_numpy(imageio.imread(parser.image_paths[i])[..., :3]).float().cuda() / 255
        cams.append((torch.linalg.inv(c2w), K, W, H, gt))
    dc = p["colors"][:, :1].clone().requires_grad_(True)
    rest = p["colors"][:, 1:].clone().requires_grad_(True)
    lo = torch.logit(p["opacities"].clamp(1e-6, 1 - 1e-6)).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [dc], "lr": 2.5e-3}, {"params": [rest], "lr": 2.5e-3 / 20},
                            {"params": [lo], "lr": 5e-2}], eps=1e-15)
    g = torch.Generator().manual_seed(seed)
    for it in range(steps):
        vm, K, W, H, gt = cams[int(torch.randint(len(cams), (1,), generator=g))]
        img, _, _ = rasterization(p["means"], p["quats"], p["scales"], torch.sigmoid(lo),
                                  torch.cat([dc, rest], 1), vm[None], K[None], W, H, sh_degree=deg)
        x = img[0].clamp(0, 1)
        l1 = (x - gt).abs().mean()
        ssim = fused_ssim(x.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], padding="valid")
        loss = 0.8 * l1 + 0.2 * (1 - ssim)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    out = dict(p)
    out["colors"] = torch.cat([dc, rest], 1).detach()
    out["opacities"] = torch.sigmoid(lo).detach()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--teacher")
    ap.add_argument("--refine-only")
    ap.add_argument("--keep", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--cos-gate", type=float, default=0.8)
    ap.add_argument("--dist-gate", type=float, default=3.0)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--test-every", type=int, default=8)
    a = ap.parse_args()
    parser = Parser(data_dir=a.data_dir, factor=1, normalize=True, test_every=a.test_every)
    t0 = time.time()
    info = {"args": vars(a)}
    if a.refine_only:
        p, deg, _ = load_splat_ply(a.refine_only)
    else:
        p, deg, n = load_splat_ply(a.teacher)
        score, _ = contributions(p, parser, a.test_every)
        m = int(round(n * a.keep))
        keep_idx = torch.sort(torch.argsort(score, descending=True)[:m]).values
        if a.no_merge:
            p = subset(p, keep_idx)
        else:
            p, minfo = gated_merge(p, score, keep_idx, a.cos_gate, a.dist_gate)
            info["merge"] = minfo
    t1 = time.time()
    if a.refine:
        p = refine(p, deg, parser, a.test_every, a.refine)
    info.update(num_gaussians=int(p["means"].shape[0]), build_seconds=t1 - t0, refine_seconds=time.time() - t1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_ply(a.out, p, (deg + 1) ** 2 - 1)
    json.dump(info, open(a.out + ".json", "w"), indent=2)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
