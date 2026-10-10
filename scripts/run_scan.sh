#!/usr/bin/env bash
# Full per-scan pipeline, same protocol as scan6.
# Usage: run_scan.sh <scan_id> <data_root>
#   data_root contains Rectified/, Calibration/, Points/stl/, ObsMask/
set -euo pipefail
SCAN="$1"; ROOT="$2"
G=/root/work/geolod
cd "$G"
export PATH=/usr/local/miniconda3/envs/py312/bin:/root/opt/node22/bin:$PATH
export PYTHONPATH=/root/work/objectgs-edge/.runtime/site-packages
export TORCH_EXTENSIONS_DIR=/root/work/objectgs-edge/.runtime/torch_extensions
export TORCH_CUDA_ARCH_LIST=8.9
P=/usr/local/miniconda3/envs/py312/bin/python
V=$G/venv/bin/python
S=$(printf "%03d" "$SCAN")
COL=$ROOT/colmap
STL=$ROOT/Points/stl/stl${S}_total.ply
OBS=$ROOT/ObsMask/ObsMask${SCAN}_10.mat
O=$G/outputs/scan$SCAN
mkdir -p "$O"
ST="npx --prefix $G/tools splat-transform"

echo "== [1] COLMAP (fixed DTU poses)"
[ -f "$COL/sparse/0/points3D.bin" ] || $P scripts/dtu_to_colmap.py --data-root "$ROOT" --output "$COL" --scan "$SCAN" --light 3 --scale 0.5

echo "== [2] teacher (gsplat v1.5.3 simple_trainer default)"
TPLY=$O/teacher/ply/point_cloud_29999.ply
if [ ! -f "$TPLY" ]; then
  (cd gsplat-upstream/examples && $V simple_trainer.py default --data_dir "$COL" --data_factor 1 \
     --result_dir "$O/teacher" --disable_viewer --save_ply > "$O/teacher_train.log" 2>&1)
fi
cat "$O/teacher/stats/val_step29999.json"; echo

echo "== [3] official baselines (-d, --decimate-adaptive) + Streamed SOG"
[ -f "$O/baselines/adaptive/streamed/lod-meta.json" ] || bash scripts/build_baselines.sh "$TPLY" "$O/baselines" > "$O/baselines_build.log" 2>&1

echo "== [4] decode Streamed SOG levels"
for m in uniform adaptive; do
  [ -f "$O/decoded/$m/lod3.ply" ] || bash scripts/decode_streamed.sh "$O/baselines/$m/streamed" "$O/decoded/$m" 4
done

echo "== [5] ablation at 50/25/10%"
for spec in "0.5 k50 1" "0.25 k25 2" "0.1 k10 3"; do
  set -- $spec; KEEP=$1; TAG=$2; L=$3
  M=$O/m1/$TAG; mkdir -p "$M"
  OFF=$O/decoded/uniform/lod$L.ply; ADA=$O/decoded/adaptive/lod$L.ply
  [ -f "$M/prune.ply" ]        || $V scripts/merge_lod.py --data-dir "$COL" --teacher "$TPLY" --keep $KEEP --no-merge --out "$M/prune.ply"
  [ -f "$M/merge.ply" ]        || $V scripts/merge_lod.py --data-dir "$COL" --teacher "$TPLY" --keep $KEEP --out "$M/merge.ply"
  [ -f "$M/prune_ref.ply" ]    || $V scripts/merge_lod.py --data-dir "$COL" --refine-only "$M/prune.ply" --refine 3000 --out "$M/prune_ref.ply"
  [ -f "$M/merge_ref.ply" ]    || $V scripts/merge_lod.py --data-dir "$COL" --refine-only "$M/merge.ply" --refine 3000 --out "$M/merge_ref.ply"
  [ -f "$M/official_ref.ply" ] || $V scripts/merge_lod.py --data-dir "$COL" --refine-only "$OFF" --refine 3000 --out "$M/official_ref.ply"
  [ -f "$M/adaptive_ref.ply" ] || $V scripts/merge_lod.py --data-dir "$COL" --refine-only "$ADA" --refine 3000 --out "$M/adaptive_ref.ply"
  [ -f "$M/eval.json" ] || $V scripts/eval_lod.py --data-dir "$COL" --stl "$STL" --obsmask "$OBS" --out "$M/eval.json" \
     official=$OFF adaptive=$ADA official_ref=$M/official_ref.ply adaptive_ref=$M/adaptive_ref.ply \
     prune=$M/prune.ply merge=$M/merge.ply prune_ref=$M/prune_ref.ply merge_ref=$M/merge_ref.ply \
     > "$M/eval.log" 2>&1
done

echo "== [6] teacher eval"
[ -f "$O/eval_teacher.json" ] || $V scripts/eval_lod.py --data-dir "$COL" --stl "$STL" --obsmask "$OBS" \
   --out "$O/eval_teacher.json" teacher=$TPLY > "$O/eval_teacher.log" 2>&1
echo SCAN_DONE
