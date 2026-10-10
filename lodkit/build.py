"""lodkit build: teacher PLY + COLMAP training views -> standard Streamed SOG."""
from __future__ import annotations

import json
import os
import time

import torch

from . import sog
from .colmap import Scene
from .merge import gated_merge
from .ply import read_ply, subset, write_ply
from .render import contributions, recover


def build(data_dir: str, teacher: str, out: str, levels=(0.5, 0.25, 0.1), recover_steps: int = 3000,
          merge: bool = True, cos_gate: float = 0.8, dist_gate: float = 3.0, test_every: int = 8,
          bundle: bool = True, device: str = "cuda") -> dict:
    os.makedirs(out, exist_ok=True)
    scene = Scene(data_dir, test_every=test_every)
    train = scene.train_views
    t0 = time.time()
    p, deg, n = read_ply(teacher, device)
    score = contributions(p, train, device)
    order = torch.argsort(score, descending=True)
    report = {"teacher": teacher, "num_gaussians": n, "sh_degree": deg, "train_views": len(train),
              "held_out_views": len(scene.test_views), "score_seconds": time.time() - t0,
              "config": dict(levels=list(levels), recover_steps=recover_steps, merge=merge,
                             cos_gate=cos_gate, dist_gate=dist_gate, test_every=test_every),
              "levels": []}
    paths = [teacher]
    for li, keep in enumerate(levels, start=1):
        t = time.time()
        m = int(round(n * keep))
        keep_idx = torch.sort(order[:m]).values
        q, minfo = (gated_merge(p, score, keep_idx, cos_gate, dist_gate) if merge
                    else (subset(p, keep_idx), None))
        if recover_steps:
            q = recover(q, deg, train, recover_steps, dev=device)
        path = os.path.join(out, f"lod{li}.ply")
        write_ply(path, q)
        report["levels"].append({"lod": li, "keep": keep, "count": m, "merge": minfo,
                                 "seconds": time.time() - t, "file_bytes": os.path.getsize(path)})
        paths.append(path)
        del q
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if bundle:
        t = time.time()
        sdir = os.path.join(out, "streamed")
        sog.bundle_streamed(paths, sdir)
        report["bundle_seconds"] = time.time() - t
        report["streamed_bytes"] = sum(os.path.getsize(os.path.join(r, f))
                                       for r, _, fs in os.walk(sdir) for f in fs)
    report["total_seconds"] = time.time() - t0
    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
