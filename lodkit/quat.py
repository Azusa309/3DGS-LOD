"""Quaternion <-> rotation matrix, quaternions in (w, x, y, z)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Bar-Itzhack method: eigenvector of the largest eigenvalue of a symmetric 4x4.

    Robust for every rotation (no trace branches). The eigendecomposition runs
    on CPU in float64: batched CUDA eigh over millions of 4x4 matrices requested
    a ~51 GB cuSOLVER workspace in our runs.
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
    _, vecs = torch.linalg.eigh(K.cpu())
    xyzw = vecs[:, :, -1].to(R.device)
    q = torch.stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]], -1)
    return F.normalize(q, dim=-1).to(R.dtype)
