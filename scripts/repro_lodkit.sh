#!/usr/bin/env bash
# Reproduce evidence/scan6/eval_geolod_build.json with the packaged lodkit CLI.
set -euo pipefail
G=/root/work/geolod; cd "$G"
export PATH=/usr/local/miniconda3/envs/py312/bin:/root/opt/node22/bin:$PATH
export PYTHONPATH=$G/pkg:/root/work/objectgs-edge/.runtime/site-packages
export TORCH_EXTENSIONS_DIR=/root/work/objectgs-edge/.runtime/torch_extensions
export LODKIT_SPLAT_TRANSFORM="npx --prefix $G/tools splat-transform"
V=$G/venv/bin/python
O=$G/outputs/scan6/lodkit_repro
rm -rf "$O"; mkdir -p "$O"
$V -m lodkit.cli build --data /root/datasets/dtu-scan6/colmap \
   --teacher outputs/scan6/teacher/ply/point_cloud_29999.ply --out "$O"
bash scripts/decode_streamed.sh "$O/streamed" "$O/decoded" 4
$V scripts/eval_lod.py --data-dir /root/datasets/dtu-scan6/colmap \
   --stl /root/datasets/dtu-scan6/Points/stl/stl006_total.ply \
   --obsmask /root/datasets/dtu-scan6/ObsMask/ObsMask6_10.mat --out "$O/eval.json" \
   r_lod1=$O/decoded/lod1.ply r_lod2=$O/decoded/lod2.ply r_lod3=$O/decoded/lod3.ply
echo REPRO_DONE
