"""Thin wrapper around the official PlayCanvas splat-transform CLI (MIT).

Pinned to the version every result in evidence/ was produced with. Override
with GEOLOD_SPLAT_TRANSFORM (e.g. a local install path) if needed.
Requires Node.js >= 22 (splat-transform 3.10 'engines' field).
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess

SPLAT_TRANSFORM_VERSION = "3.10.0"


def _cmd() -> list[str]:
    custom = os.environ.get("GEOLOD_SPLAT_TRANSFORM")
    if custom:
        return shlex.split(custom)
    if shutil.which("npx") is None:
        raise RuntimeError("npx not found: install Node.js >= 22 (https://nodejs.org)")
    return ["npx", "--yes", f"@playcanvas/splat-transform@{SPLAT_TRANSFORM_VERSION}"]


def run(args: list[str]) -> None:
    subprocess.run(_cmd() + args, check=True)


def bundle_streamed(level_plys: list[str], out_dir: str) -> str:
    """level_plys[0] is full detail (LOD 0). Writes out_dir/lod-meta.json."""
    args = ["-w"]
    for i, p in enumerate(level_plys):
        args += [p, "-l", str(i)]
    meta = os.path.join(out_dir, "lod-meta.json")
    run(args + [meta, "--filter-nan"])
    return meta


def decode_level(lod_meta: str, level: int, out_ply: str) -> None:
    """Decode one LOD of a Streamed SOG back to PLY (what the browser loads)."""
    run(["-w", lod_meta, "-L", str(level), out_ply])


def official_decimate(src_ply: str, percent: float, out_ply: str, adaptive: bool = False) -> None:
    run(["-w", src_ply, "--decimate-adaptive" if adaptive else "-d", f"{percent:g}%", out_ply])
