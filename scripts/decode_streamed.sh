#!/usr/bin/env bash
# Decode each LOD level of a Streamed SOG back to PLY: this is exactly the data
# the browser viewer loads (SOG quantisation included).
# Usage: decode_streamed.sh <streamed_dir> <out_dir> <num_levels>
set -euo pipefail
IN="$1"; OUT="$2"; N="$3"
export PATH=/root/opt/node22/bin:$PATH
ST="npx --prefix /root/work/geolod/tools splat-transform"
mkdir -p "$OUT"
for ((L=0; L<N; L++)); do
  $ST -w "$IN/lod-meta.json" -L "$L" "$OUT/lod$L.ply" 2>&1 | tail -1
done
