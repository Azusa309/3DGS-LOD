"""Geometry-gated merge (device-agnostic, no renderer needed).

Every removed Gaussian is assigned to its nearest kept Gaussian and merged into
it only if the normals agree (|n_r . n_k| > cos_gate, normal = shortest axis)
and it is close (dist < dist_gate * max_scale(kept)); otherwise it is dropped.
Merge = contribution-weighted moment matching of mean, covariance and SH; the
kept Gaussian's opacity is unchanged. This forbids merges across thin walls.
"""
from __future__ import annotations

import torch
from scipy.spatial import cKDTree

from .quat import quat_to_rot, rot_to_quat


@torch.no_grad()
def gated_merge(p: dict, score: torch.Tensor, keep_idx: torch.Tensor, cos_gate: float = 0.8,
                dist_gate: float = 3.0):
    dev = p["means"].device
    n = p["means"].shape[0]
    kept = torch.zeros(n, dtype=torch.bool, device=dev)
    kept[keep_idx] = True
    rem = torch.nonzero(~kept).squeeze(1)
    R = quat_to_rot(p["quats"])
    s = p["scales"]
    normal = torch.gather(R, 2, s.argmin(1)[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    _, nn = cKDTree(p["means"][keep_idx].cpu().numpy()).query(p["means"][rem].cpu().numpy(), k=1, workers=-1)
    tgt = keep_idx[torch.from_numpy(nn).to(dev)]
    d = (p["means"][rem] - p["means"][tgt]).norm(dim=-1)
    cosv = (normal[rem] * normal[tgt]).sum(-1).abs()
    ok = (cosv > cos_gate) & (d < dist_gate * s[tgt].max(-1).values) & (score[rem] > 0)
    src, dst = rem[ok], tgt[ok]

    w = score.float().clamp_min(1e-12)
    W = torch.zeros(n, device=dev).index_add_(0, dst, w[src])
    W[keep_idx] += w[keep_idx]
    mu = p["means"] * w[:, None]
    MU = torch.zeros_like(p["means"]).index_add_(0, dst, mu[src])
    MU[keep_idx] += mu[keep_idx]
    MU = MU / W[:, None].clamp_min(1e-20)
    cov = R @ torch.diag_embed(s ** 2) @ R.transpose(1, 2)

    def second(ids, base):
        dm = p["means"][ids] - base
        return (cov[ids] + dm[:, :, None] * dm[:, None, :]) * w[ids, None, None]

    C = torch.zeros_like(cov).index_add_(0, dst, second(src, MU[dst]))
    C[keep_idx] += second(keep_idx, MU[keep_idx])
    C = C / W[:, None, None].clamp_min(1e-20)
    SH = torch.zeros_like(p["colors"]).index_add_(0, dst, (p["colors"] * w[:, None, None])[src])
    SH[keep_idx] += p["colors"][keep_idx] * w[keep_idx, None, None]
    SH = SH / W[:, None, None].clamp_min(1e-20)

    k = keep_idx
    # CPU float64: batched CUDA eigh requests a huge cuSOLVER workspace on
    # millions of 3x3 matrices (51 GB OOM observed on an RTX 4090).
    evals, evecs = torch.linalg.eigh(C[k].double().cpu())
    evecs = evecs * torch.sign(torch.det(evecs))[:, None, None]
    out = {"means": MU[k], "scales": evals.clamp_min(1e-16).sqrt().float().to(dev),
           "quats": rot_to_quat(evecs.float().to(dev)), "opacities": p["opacities"][k], "colors": SH[k]}
    recv = torch.zeros(n, device=dev).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    info = {"num_removed": int(rem.numel()), "num_merged": int(src.numel()),
            "frac_removed_merged": float(src.numel() / max(rem.numel(), 1)),
            "kept_receiving_merges": int((recv[k] > 0).sum())}
    return out, info
