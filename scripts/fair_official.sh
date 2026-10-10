#!/usr/bin/env bash
# Fair comparison: official uniform LOD + the same fixed-geometry recovery,
# then bundled into Streamed SOG and decoded, exactly like geolod build output.
# Usage: fair_official.sh <scan_id> <colmap_dir> <stl> <obsmask>
set -euo pipefail
SCAN="$1"; COL="$2"; STL="$3"; OBS="$4"
G=/root/work/geolod; cd "$G"
export PATH=/usr/local/miniconda3/envs/py312/bin:/root/opt/node22/bin:$PATH
export PYTHONPATH=/root/work/objectgs-edge/.runtime/site-packages
export TORCH_EXTENSIONS_DIR=/root/work/objectgs-edge/.runtime/torch_extensions
V=$G/venv/bin/python
ST="npx --prefix $G/tools splat-transform"
O=$G/outputs/scan$SCAN
# Input = splat-transform's raw decimated PLY (baselines/uniform/lodN.ply), NOT the
# SOG-decoded level, so the official path is SOG-quantised exactly once, like ours.
F=$O/fair_official_raw; mkdir -p "$F"
T=$O/teacher/ply/point_cloud_29999.ply
ARGS=("$T" -l 0)
i=1
for tag in k50 k25 k10; do
  # re-run recovery with the fixed writer (no nx/ny/nz) so splat-transform can stack levels
  [ -f "$F/lod$i.ply" ] || $V scripts/merge_lod.py --data-dir "$COL" --refine-only "$O/baselines/uniform/lod$i.ply" \
      --refine 3000 --out "$F/lod$i.ply"
  ARGS+=("$F/lod$i.ply" -l "$i"); i=$((i+1))
done
[ -f "$F/streamed/lod-meta.json" ] || $ST -w "${ARGS[@]}" "$F/streamed/lod-meta.json" --filter-nan
bash scripts/decode_streamed.sh "$F/streamed" "$F/decoded" 4
$V scripts/eval_lod.py --data-dir "$COL" --stl "$STL" --obsmask "$OBS" --out "$F/eval.json" \
   f_lod1=$F/decoded/lod1.ply f_lod2=$F/decoded/lod2.ply f_lod3=$F/decoded/lod3.ply
du -sb "$F/streamed"
echo FAIR_DONE
