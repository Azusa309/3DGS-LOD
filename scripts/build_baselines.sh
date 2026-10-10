#!/usr/bin/env bash
# Official splat-transform 3.10.0 LOD baselines, following the documented
# two-step workflow ("Generating Streamed SOG"):
#   1) decimate the full-detail source into lower levels (.ply)
#   2) bundle source + levels into Streamed SOG (lod-meta.json)
# Two baselines: uniform (-d) and adaptive (--decimate-adaptive).
# Usage: build_baselines.sh <teacher.ply> <out_dir>
set -euo pipefail
SRC="$1"; OUT="$2"
export PATH=/root/opt/node22/bin:$PATH
ST="npx --prefix /root/work/geolod/tools splat-transform"
LEVELS=(50 25 10)   # percentages from the official docs example

timed() {  # timed <label> <cmd...>
  local label="$1"; shift
  local t0; t0=$(date +%s.%N)
  "$@" 2>&1 | tail -2
  local t1; t1=$(date +%s.%N)
  echo "TIMING $label wall=$(echo "$t1 - $t0" | bc)s"
}

for MODE in uniform adaptive; do
  D="$OUT/$MODE"; mkdir -p "$D"
  FLAG=$([ "$MODE" = uniform ] && echo "-d" || echo "--decimate-adaptive")
  ARGS=("$SRC" -l 0)
  i=1
  for P in "${LEVELS[@]}"; do
    timed "$MODE lod$i ${P}%" $ST -w "$SRC" $FLAG "${P}%" "$D/lod$i.ply"
    ARGS+=("$D/lod$i.ply" -l "$i"); i=$((i+1))
  done
  timed "$MODE bundle" $ST -w "${ARGS[@]}" "$D/streamed/lod-meta.json" --filter-nan
  du -sb "$D"/lod*.ply "$D/streamed" | tee "$D/sizes.txt"
done
echo BUILD_DONE
