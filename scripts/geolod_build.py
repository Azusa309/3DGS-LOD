"""geolod build: teacher PLY + training views -> standard Streamed SOG.

One command for the method that held up in the scan6 ablation:
  per level: occlusion-aware contribution pruning (T*alpha, training views only)
             -> geometry-gated merge (normal + distance gate)
             -> fixed-geometry recovery (SH + opacity only, geometry frozen)
  then bundle level 0 (teacher) + levels into Streamed SOG with the official
  splat-transform 3.10.0, so the unmodified PlayCanvas viewer loads it.

Without training views, use splat-transform's own -d / --decimate-adaptive:
contribution scores need cameras.

Usage:
  geolod_build.py --data-dir COLMAP_DIR --teacher teacher.ply --out OUT_DIR \
      [--levels 0.5,0.25,0.1] [--refine 3000] [--no-merge]
Output:
  OUT_DIR/lod{1..N}.ply, OUT_DIR/streamed/lod-meta.json, OUT_DIR/build_report.json
"""
import argparse, json, os, subprocess, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/work/geolod/gsplat-upstream/examples")
from eval_lod import load_splat_ply  # noqa: E402
from prune_contrib import contributions  # noqa: E402
from merge_lod import gated_merge, refine, subset, write_ply  # noqa: E402
from datasets.colmap import Parser  # noqa: E402

ST = ["npx", "--prefix", "/root/work/geolod/tools", "splat-transform"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="0.5,0.25,0.1")
    ap.add_argument("--refine", type=int, default=3000)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--cos-gate", type=float, default=0.8)
    ap.add_argument("--dist-gate", type=float, default=3.0)
    ap.add_argument("--test-every", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parser = Parser(data_dir=a.data_dir, factor=1, normalize=True, test_every=a.test_every)
    t0 = time.time()
    teacher, deg, n = load_splat_ply(a.teacher)
    score, views = contributions(teacher, parser, a.test_every)
    report = {"teacher": a.teacher, "num_gaussians": n, "train_views": len(views),
              "score_seconds": time.time() - t0, "levels": []}
    order = torch.argsort(score, descending=True)
    bundle = [a.teacher, "-l", "0"]
    for li, keep in enumerate([float(x) for x in a.levels.split(",")], start=1):
        t = time.time()
        m = int(round(n * keep))
        keep_idx = torch.sort(order[:m]).values
        if a.no_merge:
            p, minfo = subset(teacher, keep_idx), None
        else:
            p, minfo = gated_merge(teacher, score, keep_idx, a.cos_gate, a.dist_gate)
        if a.refine:
            p = refine(p, deg, parser, a.test_every, a.refine)
        path = os.path.join(a.out, f"lod{li}.ply")
        write_ply(path, p, (deg + 1) ** 2 - 1)
        report["levels"].append({"lod": li, "keep": keep, "count": m, "merge": minfo,
                                 "seconds": time.time() - t, "file_bytes": os.path.getsize(path)})
        bundle += [path, "-l", str(li)]
        del p
        torch.cuda.empty_cache()
    t = time.time()
    meta = os.path.join(a.out, "streamed", "lod-meta.json")
    subprocess.run(ST + ["-w"] + bundle + [meta, "--filter-nan"], check=True)
    report["bundle_seconds"] = time.time() - t
    report["streamed_bytes"] = sum(os.path.getsize(os.path.join(r, f))
                                   for r, _, fs in os.walk(os.path.dirname(meta)) for f in fs)
    report["total_seconds"] = time.time() - t0
    json.dump(report, open(os.path.join(a.out, "build_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
