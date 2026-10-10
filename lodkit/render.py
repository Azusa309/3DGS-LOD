"""gsplat-backed steps: contribution scores and fixed-geometry recovery.

Both need a CUDA GPU (gsplat rasterization is CUDA-only). gsplat is imported
lazily so the rest of the package (PLY I/O, merge, COLMAP reader) works and is
testable on CPU-only machines.
"""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import torch


def _raster():
    try:
        from gsplat import rasterization
    except ImportError as e:  # pragma: no cover
        raise ImportError("gsplat is required for this step: pip install '3dgs-lod[gpu]' "
                          "(needs a CUDA GPU)") from e
    return rasterization


def _cam(view, dev):
    vm = torch.linalg.inv(torch.from_numpy(view.c2w).float().to(dev))
    K = torch.from_numpy(view.K).float().to(dev)
    return vm, K


def contributions(p: dict, views, dev: str = "cuda") -> torch.Tensor:
    """Occlusion-aware contribution sum_pixels(T_i * alpha_i) over `views`.

    Alpha compositing is linear in features, so the gradient of the summed
    image w.r.t. an all-zero 1-channel feature equals each Gaussian's exact
    T*alpha mass, including occlusion, without a dense pixel x Gaussian table.
    """
    rasterization = _raster()
    n = p["means"].shape[0]
    score = torch.zeros(n, device=dev, dtype=torch.float64)
    for v in views:
        vm, K = _cam(v, dev)
        f = torch.zeros(n, 1, device=dev, requires_grad=True)
        img, _, _ = rasterization(p["means"], p["quats"], p["scales"], p["opacities"], f,
                                  vm[None], K[None], v.width, v.height)
        (g,) = torch.autograd.grad(img.sum(), f)
        score += g[:, 0].double()
    return score


def recover(p: dict, sh_degree: int, views, steps: int = 3000, seed: int = 0, dev: str = "cuda") -> dict:
    """Optimise SH + opacity only on `views`; means, scales, rotations frozen.

    Loss and learning rates follow gsplat simple_trainer defaults
    (0.8 L1 + 0.2 D-SSIM; sh0 2.5e-3, shN 2.5e-3/20, opacity 5e-2).
    """
    rasterization = _raster()
    try:
        from fused_ssim import fused_ssim
        ssim_fn = lambda x, y: fused_ssim(x, y, padding="valid")
    except ImportError:  # slower fallback with the same definition
        from torchmetrics.functional.image import structural_similarity_index_measure as _s
        ssim_fn = lambda x, y: _s(x, y, data_range=1.0)
    torch.manual_seed(seed)
    cams = []
    for v in views:
        vm, K = _cam(v, dev)
        gt = torch.from_numpy(np.asarray(imageio.imread(v.image_path))[..., :3]).float().to(dev) / 255
        cams.append((vm, K, v.width, v.height, gt))
    dc = p["colors"][:, :1].clone().requires_grad_(True)
    rest = p["colors"][:, 1:].clone().requires_grad_(True)
    lo = torch.logit(p["opacities"].clamp(1e-6, 1 - 1e-6)).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [dc], "lr": 2.5e-3}, {"params": [rest], "lr": 2.5e-3 / 20},
                            {"params": [lo], "lr": 5e-2}], eps=1e-15)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        vm, K, W, H, gt = cams[int(torch.randint(len(cams), (1,), generator=g))]
        img, _, _ = rasterization(p["means"], p["quats"], p["scales"], torch.sigmoid(lo),
                                  torch.cat([dc, rest], 1), vm[None], K[None], W, H, sh_degree=sh_degree)
        x = img[0].clamp(0, 1)
        loss = 0.8 * (x - gt).abs().mean() + 0.2 * (1 - ssim_fn(x.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    out = dict(p)
    out["colors"] = torch.cat([dc, rest], 1).detach()
    out["opacities"] = torch.sigmoid(lo).detach()
    return out
