"""Aggregate per-scan ablation results into one table (JSON + Markdown).

Reads outputs/scan{S}/m1/k{50,25,10}/eval.json and the teacher eval
(outputs/scan6/eval_baselines.json for scan6, outputs/scan{S}/eval_teacher.json otherwise).
Metric columns: mask PSNR (dB), SSIM, LPIPS, DTU overall (mm, mean of accuracy and completeness).
"""
import json, os, sys

G = "/root/work/geolod/outputs"
SCANS = [int(s) for s in (sys.argv[1:] or ["6", "1"])]
LEVELS = [("k50", "50%"), ("k25", "25%"), ("k10", "10%")]
METHODS = ["official", "adaptive", "prune", "merge", "official_ref", "adaptive_ref", "prune_ref", "merge_ref"]


def row(r):
    a = r["appearance"]
    return {"n": r["num_gaussians"], "psnr_mask": a["psnr_mask"], "psnr": a["psnr"], "ssim": a["ssim"],
            "lpips": a["lpips"], "dtu_mm": r["dtu_overall_mm"], "acc_mm": r["dtu_accuracy_mm"],
            "comp_mm": r["dtu_completeness_mm"]}


def load(p):
    return json.load(open(p))["assets"] if os.path.exists(p) else {}


out = {}
for s in SCANS:
    d = {}
    t = load(f"{G}/scan{s}/eval_teacher.json") or load(f"{G}/scan{s}/eval_baselines.json")
    if "teacher" in t:
        d["teacher"] = row(t["teacher"])
    for tag, _ in LEVELS:
        e = load(f"{G}/scan{s}/m1/{tag}/eval.json")
        if s == 6 and "adaptive" not in e:   # scan6 ran adaptive in the baseline eval only
            b = load(f"{G}/scan6/eval_baselines.json")
            key = {"k50": "ada_lod1", "k25": "ada_lod2", "k10": "ada_lod3"}[tag]
            if key in b:
                e = dict(e, adaptive=b[key])
        d[tag] = {m: row(e[m]) for m in METHODS if m in e}
    out[f"scan{s}"] = d

json.dump(out, open(f"{G}/summary.json", "w"), indent=2)

lines = []
for s, d in out.items():
    tch = d.get("teacher")
    lines.append(f"### DTU {s}" + (f" (teacher {tch['n']:,} Gaussians: {tch['psnr_mask']:.2f} dB, "
                                   f"{tch['dtu_mm']:.2f} mm)" if tch else ""))
    lines.append("")
    lines.append("| budget | method | mask PSNR | SSIM | LPIPS | DTU overall (mm) |")
    lines.append("|---|---|---|---|---|---|")
    for tag, lab in LEVELS:
        for m in METHODS:
            r = d.get(tag, {}).get(m)
            if r:
                lines.append(f"| {lab} | {m} | {r['psnr_mask']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} | {r['dtu_mm']:.3f} |")
    lines.append("")

# head-to-head after equal recovery: merge_ref vs official_ref
lines.append("### merge_ref - official_ref (equal recovery)")
lines.append("")
lines.append("| scan | budget | d mask PSNR (dB) | d DTU (mm) |")
lines.append("|---|---|---|---|")
for s, d in out.items():
    for tag, lab in LEVELS:
        a, b = d.get(tag, {}).get("merge_ref"), d.get(tag, {}).get("official_ref")
        if a and b:
            lines.append(f"| {s} | {lab} | {a['psnr_mask'] - b['psnr_mask']:+.2f} | {a['dtu_mm'] - b['dtu_mm']:+.3f} |")
open(f"{G}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
