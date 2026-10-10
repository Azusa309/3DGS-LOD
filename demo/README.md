# Browser demo

Split-screen A/B viewer (PlayCanvas, WebGPU with WebGL2 fallback) for the LOD levels, opened at a real held-out DTU camera. Also used to record the README GIF and the browser benchmark.

The `.sog` files are not in git (≈ 72 MB). Rebuild them as follows.

## 1. Build the Streamed SOGs (GPU machine)

```bash
lodkit build --data <scan6_colmap> --teacher <teacher.ply> --out out/ours      # -> out/ours/streamed/lod-meta.json
bash scripts/build_baselines.sh <teacher.ply> <out_dir>                  # splat-transform -d -> <out_dir>/uniform/streamed
bash scripts/fair_official.sh <scan_id> <colmap_dir> <stl> <obsmask>     # -d + the same recovery step
```

(The two scripts were written for the remote GPU box; edit the `/root/...` paths at the top for your machine.)

Copy the three `streamed/` folders to `demo/assets/{lodkit_repro,baselines/uniform,fair_official_raw}/streamed/`.

## 2. Export each fixed level to a single `.sog` (any machine, Node ≥ 22)

```bash
cd demo && mkdir -p sog
for m in lodkit_repro fair_official_raw baselines/uniform; do
  tag=$(basename "$m"); [ "$tag" = uniform ] && tag=default
  for L in 1 2 3; do
    npx --yes @playcanvas/splat-transform@3.10.0 -w "assets/$m/streamed/lod-meta.json" -L "$L" "sog/${tag}_lod${L}.sog"
  done
done
```

`cameras.json` (held-out cameras in gsplat's normalised frame) is committed; regenerate with `python export_cameras.py <scan6_colmap> cameras.json`.

## 3. Run

```bash
python -m http.server 8765        # from demo/
# open http://localhost:8765/index.html?a=ours_lod3&b=default_lod3
```

URL parameters: `a`, `b` (asset keys: `ours|recov|default` + `_lod1..3` = 50/25/10%), `mode=split|a|b`, `api=webgl2`, `view` (held-out camera index), `yaw`, `pitch`, `dist`, `spin=0`, `clean=1` (hide UI), `bench=1`.

## Scripts (need `pip install playwright`, server running)

- `python make_gif.py ours_lod3 default_lod3 ../docs/demo_10pct.gif` — deterministic orbit GIF.
- `BENCH_ROUNDS=3 python bench.py > ../evidence/browser/<name>.jsonl` — one asset at a time, full window, vsync off; fps = frames / wall time.
- `python smoke.py` — load test + screenshot.

WebGL2 fallback (`api=webgl2`) is checked on the same M1: renders correctly, ours 10% at 158 fps, but first frame ≈ 3.5 s vs ≈ 0.25 s on WebGPU (one run, not investigated).

Known gap: the principal-point offset is not modelled (only the vertical FOV), so framing differs slightly from the real photo.
