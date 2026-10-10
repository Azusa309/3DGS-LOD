#!/usr/bin/env bash
# M1 ablation at one budget. Usage: run_m1_ablation.sh <keep> <tag> <official_lod.ply>
set -euo pipefail
cd /root/work/geolod
export PATH=/usr/local/miniconda3/envs/py312/bin:$PATH
export PYTHONPATH=/root/work/objectgs-edge/.runtime/site-packages
export TORCH_EXTENSIONS_DIR=/root/work/objectgs-edge/.runtime/torch_extensions
V=venv/bin/python
D=/root/datasets/dtu-scan6/colmap
T=outputs/scan6/teacher/ply/point_cloud_29999.ply
KEEP="$1"; TAG="$2"; OFF="$3"; STEPS=3000
O=outputs/scan6/m1/$TAG; mkdir -p "$O"
$V scripts/merge_lod.py --data-dir $D --teacher $T --keep $KEEP --no-merge --out $O/prune.ply
$V scripts/merge_lod.py --data-dir $D --teacher $T --keep $KEEP --out $O/merge.ply
$V scripts/merge_lod.py --data-dir $D --refine-only $O/prune.ply --refine $STEPS --out $O/prune_ref.ply
$V scripts/merge_lod.py --data-dir $D --refine-only $O/merge.ply --refine $STEPS --out $O/merge_ref.ply
$V scripts/merge_lod.py --data-dir $D --refine-only "$OFF" --refine $STEPS --out $O/official_ref.ply
$V scripts/eval_lod.py --data-dir $D --stl /root/datasets/dtu-scan6/Points/stl/stl006_total.ply \
  --obsmask /root/datasets/dtu-scan6/ObsMask/ObsMask6_10.mat --out $O/eval.json \
  official=$OFF official_ref=$O/official_ref.ply prune=$O/prune.ply merge=$O/merge.ply \
  prune_ref=$O/prune_ref.ply merge_ref=$O/merge_ref.ply
echo M1_DONE
