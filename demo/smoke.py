"""Headless smoke test for demo/index.html: loads, renders, saves a screenshot.

Usage (from demo/, with a static server on :8765):
    python smoke.py [query]      e.g. python smoke.py "mode=split&spin=0"
Not a benchmark: headless Chromium may fall back to software rendering.
"""
import glob
import json
import os
import sys

from playwright.sync_api import sync_playwright

# Use an already-cached Chromium if Playwright's pinned build is not installed.
CHROME = os.environ.get("DEMO_CHROME") or next(iter(sorted(glob.glob(os.path.expanduser(
    "~/Library/Caches/ms-playwright/chromium-*/chrome-mac-arm64/Google Chrome for Testing.app/"
    "Contents/MacOS/Google Chrome for Testing")))), None)

query = sys.argv[1] if len(sys.argv) > 1 else "mode=split&spin=0"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/demo_smoke.png"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, headless=True,
                          args=["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"])
    page = b.new_page(viewport={"width": 800, "height": 600})
    logs = []
    page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    page.goto(f"http://localhost:8765/index.html?{query}")
    try:
        page.wait_for_function("window.__ready && window.__ready()", timeout=120_000)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        logs.append(f"timeout: {e}")
    page.wait_for_timeout(1500)
    page.screenshot(path=out)
    info = {
        "ready": ok,
        "tagA": page.text_content("#tagA"),
        "tagB": page.text_content("#tagB"),
        "stats": page.text_content("#stats"),
        "screenshot": out,
        "logs": logs[-15:],
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    b.close()
