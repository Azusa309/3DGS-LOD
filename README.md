# 3DGS-LOD

English | [中文](README.zh-CN.md)

**Better level-of-detail for web 3D Gaussian Splatting: same visual quality with half the Gaussians.**
Outputs standard **Streamed SOG** (`lod-meta.json`), so the stock PlayCanvas viewer loads it unchanged.

`lodkit build` takes a trained 3DGS PLY plus the COLMAP views it was trained on and writes coarser LOD levels that look much better than PlayCanvas `splat-transform`'s built-in decimation at the same Gaussian count.

<p align="center">
  <img src="docs/demo_10pct.gif" width="560" alt="Split-screen orbit of DTU scan6 at 10% of the Gaussians: 3DGS-LOD on the left is sharp, splat-transform -d on the right is blurry">
  <br>
  <sub>DTU scan6, both sides keep 10% of 1.48 M Gaussians, rendered in the browser (PlayCanvas) from the shipped Streamed SOG.
  Left: 3DGS-LOD. Right: <code>splat-transform -d</code>. Recorded with <a href="demo/make_gif.py"><code>demo/make_gif.py</code></a>.</sub>
</p>

> Scope: measured on 2 DTU objects (scan1, scan6) at 50% / 25% / 10% budgets on one RTX 4090.
> Browser FPS / first-frame time and phone measurements are **not done yet**.

## Results

All numbers are measured on the asset **after Streamed SOG encode → decode**, i.e. exactly what the browser loads.
7 held-out views (every 8th image, never used for building). Format: mask PSNR (dB) / SSIM / LPIPS / DTU overall (mm, vs. DTU structured-light reference scan).

| Object | Budget | `splat-transform -d` (default) | `-d` + same recovery as ours | **3DGS-LOD** |
|---|---|---|---|---|
| scan6 | 100% (teacher) | 26.47 / 0.884 / 0.085 / 1.95 | | |
| scan6 | 50% | 19.99 / 0.746 / 0.195 / 2.32 | 26.29 / 0.868 / 0.096 / 1.982 | **26.44 / 0.879 / 0.084 / 1.950** |
| scan6 | 25% | 18.68 / 0.700 / 0.248 / 2.49 | 25.94 / 0.854 / 0.118 / 2.017 | **26.34 / 0.873 / 0.090 / 1.982** |
| scan6 | 10% | 17.94 / 0.645 / 0.317 / 2.62 | **24.92** / 0.818 / 0.167 / **2.045** | 24.72 / **0.830 / 0.137** / 2.087 |
| scan1 | 100% (teacher) | 26.92 / 0.871 / 0.110 / 1.96 | | |
| scan1 | 50% | 20.25 / 0.742 / 0.235 / 2.35 | 26.38 / 0.845 / 0.151 / 1.974 | **26.58 / 0.859 / 0.117 / 1.949** |
| scan1 | 25% | 18.73 / 0.704 / 0.303 / 2.49 | 25.93 / 0.829 / 0.192 / 1.959 | **26.60 / 0.862 / 0.119 / 1.959** |
| scan1 | 10% | 18.08 / 0.667 / 0.383 / 2.60 | 25.19 / 0.783 / 0.251 / **2.015** | **25.27 / 0.816 / 0.167** / 2.028 |

What this shows:

- **vs. the default pipeline:** +6.3 to +7.9 dB mask PSNR, 50–64% lower LPIPS at the same Gaussian count. Our 10% level beats the default 50% level by 4.7–5.0 dB.
- **vs. the default pipeline given the same recovery step:** lower LPIPS in 6/6 cases (−12% to −38%), higher SSIM in 6/6, higher PSNR in 5/6 (scan6 10% is 0.20 dB lower). Our 25% level matches or beats this baseline's 50% level on every metric, on both objects — this is the "half the Gaussians" claim.
- **Geometry is a tie**, not a win: differences vs. the reference scan stay within ±0.05 mm. We do not claim better geometry.

Raw evidence: [`evidence/`](evidence/) (per-run `eval.json`, build reports, [`summary.md`](evidence/summary.md)).

## Why the default decimation loses quality

Diagnosis on the real delivery path (details in the scripts and evidence):

1. Halving the Gaussians with `splat-transform -d` drops ~6.5 dB, but DTU geometry error only rises 1.95 → 2.32 mm.
2. The loss is **not** SOG encoding: the decimated PLY and its SOG-decoded version differ by < 0.1 dB.
3. It is in **colour and opacity**: merged coarse levels have far lower opacity (mean logit 1.01 → −0.36).

So the fix targets appearance, not geometry.

## Method

```text
teacher PLY + training views (cameras + images)
  1. contribution score  Σ T·α over training views (one backward pass, exact incl. occlusion)
  2. per level: keep the top-k by contribution
  3. geometry-gated merge: each removed Gaussian merges into its nearest kept one only if
     normals agree (|cos| > 0.8) and it is close (< 3× scale); contribution-weighted moment matching
  4. fixed-geometry recovery: freeze position/scale/rotation, optimise only SH + opacity
     (3000 steps, L1 + 0.2 D-SSIM, training views only)
  5. bundle with the official splat-transform 3.10.0 → standard Streamed SOG
```

Recovery cannot invent geometry (positions are frozen); held-out views are never used.

## Usage

Requirements: CUDA GPU, Python ≥ 3.10, Node.js ≥ 22 (for `npx @playcanvas/splat-transform@3.10.0`).

```bash
pip install -e '.[gpu]'          # torch + gsplat 1.5.3
lodkit build --data path/to/colmap --teacher point_cloud.ply --out out/
# -> out/lod1..3.ply, out/streamed/lod-meta.json, out/build_report.json
lodkit decode out/streamed/lod-meta.json --level 2 --out lod2.ply
```

- `--data` must be the COLMAP folder (`images/`, `sparse/0/`) the teacher was trained on, with gsplat's default world normalisation. `lodkit` reproduces gsplat's `Parser` frame exactly (verified: poses differ by < 5e-8, identical intrinsics and split; [`evidence/*/scene_equivalence`](evidence/)).
- Undistorted pinhole cameras only.
- `--levels 0.5,0.25,0.1`, `--recover-steps 3000` (0 disables), `--no-merge`, `--test-every 8`.

Cost: 1.48 M Gaussians, three levels, **173 s** on an RTX 4090 (scoring 9 s, levels 42/24/13 s, bundling 85 s). Re-running the packaged CLI reproduces the evidence within 0.02 dB ([`evidence/scan6/lodkit_repro`](evidence/scan6/lodkit_repro)).

## Evaluation protocol

- Data: DTU SampleSet scan1/scan6, light 3, 800×600. Official DTU projection matrices kept fixed; COLMAP only triangulates init points (reprojection ≈ 0.5 px). Frame check: triangulated points are 0.37 mm (median) from the STL; random points 81 mm.
- Teacher: gsplat v1.5.3 upstream `simple_trainer.py default`, 30k steps.
- Geometry: rendered depth back-projected to DTU millimetres, accuracy/completeness inside ObsMask (20 mm cap), averaged as "DTU overall". The reference is a structured-light scan, not CAD ground truth.
- Fairness: the baseline gets the identical recovery step and is SOG-quantised exactly once, like ours.

## Limitations

- Needs training views and cameras; `splat-transform` needs only a PLY. Without views, use `splat-transform -d`.
- Only 2 DTU objects (turntable objects, 1.45–1.48 M Gaussians); no Mip-NeRF 360 scenes yet.
- No browser FPS, first-frame or memory measurements yet; equal Gaussian count implies similar rendering cost, but this is not measured.
- Not compared against LightGaussian, AtlasLC or CVT-GS.
- Recovery steps (3000) not tuned.

## Repository layout

- `lodkit/` — the installable package (`build`, `decode`, PLY/COLMAP I/O, merge, recovery)
- `scripts/` — the experiment scripts that produced `evidence/` (baselines, ablations, evaluation)
- `evidence/` — all result JSONs
- `tests/` — CPU unit tests (`pytest`)

## Acknowledgements

Built on [PlayCanvas splat-transform](https://github.com/playcanvas/splat-transform) (MIT), [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0; world normalisation adapted) and the [DTU MVS dataset](https://roboimagedata.compute.dtu.dk/?page_id=36).

## License

MIT
