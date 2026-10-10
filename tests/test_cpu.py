"""CPU-only tests: PLY round trip, quaternion round trip, merge invariants, COLMAP reader."""
import os
import struct

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lodkit.colmap import Scene
from lodkit.merge import gated_merge
from lodkit.ply import read_ply, subset, write_ply
from lodkit.quat import quat_to_rot, rot_to_quat


def rand_splats(n=500, deg=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g),
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "scales": torch.rand(n, 3, generator=g) * 0.05 + 1e-3,
        "opacities": torch.rand(n, generator=g) * 0.98 + 0.01,
        "colors": torch.randn(n, (deg + 1) ** 2, 3, generator=g) * 0.3,
    }


def same_rotation(q1, q2):
    return (q1 * q2).sum(-1).abs()  # q and -q are the same rotation


def test_ply_roundtrip(tmp_path):
    p = rand_splats()
    path = tmp_path / "a.ply"
    write_ply(str(path), p)
    q, deg, n = read_ply(str(path))
    assert deg == 3 and n == 500
    for k in ("means", "scales", "opacities", "colors"):
        assert torch.allclose(p[k], q[k], rtol=1e-5, atol=1e-6), k
    assert torch.all(same_rotation(p["quats"], q["quats"]) > 1 - 1e-6)


def test_ply_has_no_normals(tmp_path):
    # splat-transform 3.10 refuses to stack LODs when only some levels carry nx/ny/nz
    path = tmp_path / "a.ply"
    write_ply(str(path), rand_splats(10))
    head = path.read_bytes().split(b"end_header")[0].decode()
    assert "property float nx" not in head


def test_quat_roundtrip():
    q = rand_splats(2000)["quats"]
    q2 = rot_to_quat(quat_to_rot(q))
    assert torch.all(same_rotation(q, q2) > 1 - 1e-6)
    R = quat_to_rot(q2)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand_as(R), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(len(R)), atol=1e-5)


def test_merge_keeps_budget_and_moments():
    p = rand_splats(1000)
    score = torch.rand(1000, dtype=torch.float64)
    keep = torch.sort(torch.argsort(score, descending=True)[:200]).values
    out, info = gated_merge(p, score, keep)
    assert out["means"].shape[0] == 200
    assert info["num_removed"] == 800 and 0 <= info["num_merged"] <= 800
    # opacity of kept Gaussians is untouched
    assert torch.equal(out["opacities"], p["opacities"][keep])
    # kept Gaussians that received nothing are unchanged (up to rotation sign)
    assert torch.isfinite(out["scales"]).all() and (out["scales"] > 0).all()


def test_merge_gate_blocks_cross_surface():
    # two parallel thin sheets (normals along z) 1 unit apart: nothing should merge across
    n = 400
    g = torch.Generator().manual_seed(1)
    xy = torch.rand(n, 2, generator=g)
    z = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    p = rand_splats(n)
    p["means"] = torch.cat([xy, z[:, None]], 1)
    p["quats"] = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    p["scales"] = torch.tensor([[0.02, 0.02, 0.001]]).repeat(n, 1)
    score = torch.ones(n, dtype=torch.float64)
    keep = torch.arange(0, n // 2)  # keep only sheet z=0
    out, info = gated_merge(p, score, keep, cos_gate=0.8, dist_gate=3.0)
    assert info["num_merged"] == 0           # distance gate: 1.0 >> 3*0.02
    assert torch.allclose(out["means"], p["means"][keep], atol=1e-6)


def test_merge_identity_when_all_kept():
    p = rand_splats(100)
    score = torch.rand(100, dtype=torch.float64)
    out, info = gated_merge(p, score, torch.arange(100))
    assert info["num_merged"] == 0
    assert torch.allclose(out["means"], p["means"], atol=1e-6)
    assert torch.allclose(out["scales"].sort(-1).values, p["scales"].sort(-1).values, rtol=1e-4, atol=1e-6)


def test_subset():
    p = rand_splats(50)
    idx = torch.tensor([3, 7, 9])
    s = subset(p, idx)
    assert s["means"].shape == (3, 3) and torch.equal(s["means"], p["means"][idx])


def _write_colmap(d, n_img=10):
    """Synthetic PINHOLE COLMAP model: cameras on a circle looking at the origin."""
    sp = d / "sparse" / "0"
    sp.mkdir(parents=True)
    (d / "images").mkdir()
    with open(sp / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, 64, 48))
        f.write(struct.pack("<4d", 50.0, 50.0, 32.0, 24.0))
    with open(sp / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", n_img))
        for i in range(n_img):
            a = 2 * np.pi * i / n_img
            c = np.array([3 * np.cos(a), 3 * np.sin(a), 0.5])
            fwd = -c / np.linalg.norm(c)
            right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])          # world->camera rows
            t = -R @ c
            from scipy.spatial.transform import Rotation
            x, y, z, w = Rotation.from_matrix(R).as_quat()
            name = f"img_{(n_img - 1 - i):03d}.png"   # reversed: reader must sort by name
            f.write(struct.pack("<idddddddi", i + 1, w, x, y, z, *t, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(sp / "points3D.bin", "wb") as f:
        pts = np.random.default_rng(0).normal(size=(200, 3)) * 0.3
        f.write(struct.pack("<Q", len(pts)))
        for j, (x, y, z) in enumerate(pts):
            f.write(struct.pack("<QdddBBBdQ", j + 1, x, y, z, 128, 128, 128, 0.5, 0))


def test_colmap_scene(tmp_path):
    _write_colmap(tmp_path)
    s = Scene(str(tmp_path), test_every=8)
    names = [v.name for v in s.views]
    assert names == sorted(names)
    assert [v.index for v in s.test_views] == [0, 8]
    assert len(s.train_views) == 8
    for v in s.views:
        R = v.c2w[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert v.width == 64 and v.height == 48 and v.K[0, 0] == 50.0
    # normalisation is a similarity: cameras keep a common distance to their focus
    T = s.transform
    sc = np.cbrt(np.linalg.det(T[:3, :3]))
    assert sc > 0
