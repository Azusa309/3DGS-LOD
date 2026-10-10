"""Browser benchmark for demo/index.html: one asset at a time, full window.

Usage (from demo/, with a static server on :8765):
    python bench.py [asset ...]          default: all 9 assets
    BENCH_ROUNDS=3 python bench.py       interleaved rounds (asset order rotates each round)
    BENCH_HEADED=1 python bench.py       visible window instead of headless
Prints one JSON line per run: throughput fps (frames / wall-clock over 10 s of rotation),
p95 frame time, first-frame time (load + decode + first render), graphics API and GPU string.
vsync and the frame-rate limit are disabled, so fps is uncapped.
"""
import glob
import json
import os
import sys

from playwright.sync_api import sync_playwright

CHROME = os.environ.get("DEMO_CHROME") or next(iter(sorted(glob.glob(os.path.expanduser(
    "~/Library/Caches/ms-playwright/chromium-*/chrome-mac-arm64/Google Chrome for Testing.app/"
    "Contents/MacOS/Google Chrome for Testing")))), None)
ALL = ["teacher"] + [f"{m}_lod{l}" for l in (1, 2, 3) for m in ("ours", "recov", "default")]
assets = sys.argv[1:] or ALL
W, H = int(os.environ.get("BENCH_W", 1280)), int(os.environ.get("BENCH_H", 720))
API = os.environ.get("BENCH_API", "")          # "webgl2" to force WebGL2
ROUNDS = int(os.environ.get("BENCH_ROUNDS", 1))

with sync_playwright() as p:
    br = p.chromium.launch(executable_path=CHROME, headless=os.environ.get("BENCH_HEADED") != "1",
                           args=["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist",
                                 "--disable-frame-rate-limit", "--disable-gpu-vsync"])
    for rnd in range(ROUNDS):
        k = rnd % len(assets)
        for key in assets[k:] + assets[:k]:      # rotate order to spread thermal / cache effects
            page = br.new_page(viewport={"width": W, "height": H})
            api = f"&api={API}" if API else ""
            page.goto(f"http://localhost:8765/index.html?a={key}&b={key}&mode=a&bench=1{api}")
            page.wait_for_function("window.__bench", timeout=180_000)
            r = page.evaluate("window.__bench")
            gpu = page.evaluate("""() => { const c = document.createElement('canvas').getContext('webgl2');
                const e = c && c.getExtension('WEBGL_debug_renderer_info');
                return e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")
            print(json.dumps({"round": rnd + 1, "asset": key, "api": r["api"], "gpu": gpu,
                              "size": [r["width"], r["height"]], "fps": round(r["fps"], 1),
                              "p95_ms": round(r["p95_ms"], 2),
                              "first_frame_ms": round(r["first_frame_ms"]["A"]), "frames": r["frames"]}), flush=True)
            page.close()
    br.close()
