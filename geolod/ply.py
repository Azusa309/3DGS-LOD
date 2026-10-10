"""Standard 3DGS PLY I/O (original INRIA convention).

Stored (raw) -> activated:
    scale_i  : log scale            -> exp
    opacity  : logit                -> sigmoid
    rot_i    : quaternion (w,x,y,z) -> normalised
    f_dc_i, f_rest_j : SH, f_rest channel-major [3, K-1]
In memory every splat set is a dict of activated tensors:
    means [N,3], quats [N,4], scales [N,3], opacities [N], colors [N,K,3]
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def read_ply(path: str, device: str = "cpu"):
    """Return (splats, sh_degree, count)."""
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    g = lambda n: np.asarray(v[n], dtype=np.float32)
    n = len(v.data)
    means = np.stack([g("x"), g("y"), g("z")], 1)
    quats = np.stack([g(f"rot_{i}") for i in range(4)], 1)
    scales = np.stack([g(f"scale_{i}") for i in range(3)], 1)
    op = g("opacity")
    dc = np.stack([g(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda s: int(s[7:]))
    if rest:
        r = np.stack([g(k) for k in rest], 1).reshape(n, 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, r], 1)
    else:
        sh = dc
    deg = int(round(math.sqrt(sh.shape[1]))) - 1
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    splats = dict(means=t(means), quats=F.normalize(t(quats), dim=-1), scales=torch.exp(t(scales)),
                  opacities=torch.sigmoid(t(op)), colors=t(sh))
    return splats, deg, n


def write_ply(path: str, p: dict) -> None:
    """Write activated splats back to the standard PLY layout.

    No nx/ny/nz fields: splat-transform treats them as an extra layer and then
    refuses to stack LOD levels that lack them (observed with 3.10.0).
    """
    n = p["means"].shape[0]
    sh = p["colors"].detach().cpu().numpy()
    k_rest = sh.shape[1] - 1
    cols = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    cols += [(f"f_dc_{i}", "f4") for i in range(3)]
    cols += [(f"f_rest_{i}", "f4") for i in range(3 * k_rest)]
    cols += [("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)]
    arr = np.zeros(n, dtype=cols)
    m = p["means"].detach().cpu().numpy()
    arr["x"], arr["y"], arr["z"] = m[:, 0], m[:, 1], m[:, 2]
    for i in range(3):
        arr[f"f_dc_{i}"] = sh[:, 0, i]
    if k_rest:
        rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
        for i in range(rest.shape[1]):
            arr[f"f_rest_{i}"] = rest[:, i]
    arr["opacity"] = torch.logit(p["opacities"].detach().clamp(1e-6, 1 - 1e-6)).cpu().numpy()
    ls = torch.log(p["scales"].detach().clamp_min(1e-12)).cpu().numpy()
    for i in range(3):
        arr[f"scale_{i}"] = ls[:, i]
    q = F.normalize(p["quats"].detach(), dim=-1).cpu().numpy()
    for i in range(4):
        arr[f"rot_{i}"] = q[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def subset(p: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx].contiguous() for k, v in p.items()}
