"""Record a split-screen A/B orbit GIF of demo/index.html (for the README).

Usage (from demo/, with a static server on :8765):
    python make_gif.py [a] [b] [out.gif]
    e.g. python make_gif.py ours_lod3 default_lod3 ../docs/demo.gif
The camera is driven deterministically via window.__setOrbit, so every frame
is fully rendered (no dropped frames, unlike screen recording).
"""
import glob
import io
import math
import os
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

CHROME = os.environ.get("DEMO_CHROME") or next(iter(sorted(glob.glob(os.path.expanduser(
    "~/Library/Caches/ms-playwright/chromium-*/chrome-mac-arm64/Google Chrome for Testing.app/"
    "Contents/MacOS/Google Chrome for Testing")))), None)

a = sys.argv[1] if len(sys.argv) > 1 else "ours_lod3"
b = sys.argv[2] if len(sys.argv) > 2 else "default_lod3"
out = sys.argv[3] if len(sys.argv) > 3 else "demo.gif"
N, YAW, PITCH = int(os.environ.get("GIF_FRAMES", 40)), 35.0, 8.0   # frames, yaw / pitch amplitude (deg)
W, H = 1200, 600                                                   # capture size
OUT_W = int(os.environ.get("GIF_WIDTH", 640))                      # GIF width
COLORS = int(os.environ.get("GIF_COLORS", 128))
DITHER = Image.FLOYDSTEINBERG if os.environ.get("GIF_DITHER") == "1" else Image.NONE

frames = []
with sync_playwright() as p:
    br = p.chromium.launch(executable_path=CHROME, headless=True,
                           args=["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"])
    page = br.new_page(viewport={"width": W, "height": H})
    page.goto(f"http://localhost:8765/index.html?a={a}&b={b}&mode=split&spin=0&clean=1")
    page.wait_for_function("window.__ready && window.__ready()", timeout=180_000)
    for i in range(N):
        t = 2 * math.pi * i / N             # one smooth closed loop
        page.evaluate(f"window.__setOrbit({YAW * math.sin(t)}, {PITCH * math.sin(2 * t)}, 1)")
        # wait two rAFs so the new camera pose is rendered before capturing
        page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
        frames.append(img.resize((OUT_W, OUT_W * H // W), Image.LANCZOS))
        print(f"\rframe {i + 1}/{N}", end="", flush=True)
    br.close()

# Shared palette (from a few frames) avoids per-frame colour flicker.
sample = Image.new("RGB", (OUT_W, frames[0].height * 3))
for k, f in enumerate(frames[:: N // 3][:3]):
    sample.paste(f, (0, k * f.height))
pal = sample.quantize(colors=COLORS, method=Image.MEDIANCUT)
q = [f.quantize(palette=pal, dither=DITHER) for f in frames]
# 40 frames x 80 ms = 3.2 s loop, same motion as before (sin path is resolution-independent)
q[0].save(out, save_all=True, append_images=q[1:], duration=80, loop=0, optimize=True)
print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {N} frames)")
