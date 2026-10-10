"""Command line: `lodkit build` and `lodkit decode`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lodkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="teacher PLY + COLMAP views -> Streamed SOG (lod-meta.json)")
    b.add_argument("--data", required=True, help="COLMAP dir the teacher was trained on (images/, sparse/0/)")
    b.add_argument("--teacher", required=True, help="trained 3DGS PLY (gsplat/INRIA layout)")
    b.add_argument("--out", required=True)
    b.add_argument("--levels", default="0.5,0.25,0.1", help="kept fraction per coarser level")
    b.add_argument("--recover-steps", type=int, default=3000, help="0 disables fixed-geometry recovery")
    b.add_argument("--no-merge", action="store_true", help="prune only, no gated merge")
    b.add_argument("--cos-gate", type=float, default=0.8)
    b.add_argument("--dist-gate", type=float, default=3.0)
    b.add_argument("--test-every", type=int, default=8,
                   help="views i %% N == 0 are held out (never used); match your trainer")
    b.add_argument("--no-bundle", action="store_true", help="write lod*.ply only")
    b.add_argument("--device", default="cuda")

    d = sub.add_parser("decode", help="decode one LOD of a Streamed SOG to PLY")
    d.add_argument("lod_meta")
    d.add_argument("--level", type=int, required=True)
    d.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "build":
        from .build import build
        r = build(a.data, a.teacher, a.out, tuple(float(x) for x in a.levels.split(",")),
                  a.recover_steps, not a.no_merge, a.cos_gate, a.dist_gate, a.test_every,
                  not a.no_bundle, a.device)
        print(json.dumps(r, indent=2))
    elif a.cmd == "decode":
        from .sog import decode_level
        decode_level(a.lod_meta, a.level, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
