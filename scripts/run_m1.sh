#!/usr/bin/env bash
# M1 ablation at 10% budget (same teacher, same eval). Every asset goes through
# Streamed SOG encode/decode before evaluation, like the official baselines.
set -euo pipefail
cd /root/work/geolod
export PATH=/usr/local/miniconda3/envs/py312/bin:/root/opt/node22/bin:$PATH
export PYTHONPATH=/root/work/objectgs-edge/.runtime/site-packages TORCH_EXTENSIONS_DIR=/root/work/geolod/.runtime/torch_extensions
V=venv/bin/python; D=/root/datasets/dtu-scan6/colmap; T=outputs/scan6/teacher/ply/point_cloud_29999.ply
O=outputs/scan6/m1; mkdir -p $O
ST="npx --prefix /root/work/geolod/tools splat-transform"
STEPS=${STEPS:-2000}
$V scripts/merge_lod.py --data-dir $D --teacher $T --keep 0.1 --no-merge --out $O/prune.ply
$V scripts/merge_lod.py --data-dir $D --teacher $T --keep 0.1 --out $O/merge.ply
$V scripts/merge_lod.py --data-dir $D --teacher $T --keep 0.1 --no-merge --refine $STEPS --out $O/prune_ref.ply
$V scripts/merge_lod.py --data-dir $D --teacher $T --keep 0.1 --refine $STEPS --out $O/merge_ref.ply
$V scripts/merge_lod.py --data-dir $D --refine-only outputs/scan6/baselines/uniform/lod3.ply --refine $STEPS --out $O/uni_ref.ply
$V scripts/merge_lod.py --data-dir $D --refine-only outputs/scan6/baselines/adaptive/lod3.ply --refine $STEPS --out $O/ada_ref.ply
ARGS=()
for m in prune merge prune_ref merge_ref uni_ref ada_ref; do
  # single-level SOG round trip = what the browser decodes for this level
  $ST -w $O/$m.ply $O/sog_$m/meta.json >/dev/null 2>&1
  $ST -w $O/sog_$m/meta.json $O/dec_$m.ply >/dev/null 2>&1
  ARGS+=("$m=$O/dec_$m.ply")
done
$V scripts/eval_lod.py --data-dir $D --stl /root/datasets/dtu-scan6/Points/stl/stl006_total.ply \
  --obsmask /root/datasets/dtu-scan6/ObsMask/ObsMask6_10.mat --out $O/eval_m1.json "${ARGS[@]}"
echo M1_DONE
