"""Watch DigitalOcean for NVIDIA GPU capacity (21 Oct 2026; Chris: "the next big training runs can run on DO. Once we're
ready just set up a watcher to find the cards when they are available").

Polls GET /v2/sizes every INTERVAL seconds and reports when a wanted NVIDIA size has capacity in some region. It never
creates a droplet: launching stays with the guarded launcher and Chris's go. On a hit it logs, pushes a phone
notification (tools/notify.py, public ntfy: no secrets in the text) and, with --once, exits so a launcher can follow.

    python -X utf8 tools/chitonga/train/do_gpu_watch.py [--want h100,h200,l40s,6000ada,4000ada] [--interval 300] [--once]
Log: internal/chitonga_train/runs/do_gpu_watch.log
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
LOG = os.path.join(REPO, "internal", "chitonga_train", "runs", "do_gpu_watch.log")
NVIDIA = {"h100": "gpu-h100", "h200": "gpu-h200", "l40s": "gpu-l40s", "6000ada": "gpu-6000ada", "4000ada": "gpu-4000ada"}


def token():
    for l in open(os.path.join(REPO, ".env.local"), encoding="utf-8"):
        m = re.match(r"\s*DO_API_TOKEN\s*=\s*(.*)", l)
        if m:
            return m.group(1).strip().strip('"')
    raise SystemExit("DO_API_TOKEN missing in .env.local")


def available(want):
    req = urllib.request.Request("https://api.digitalocean.com/v2/sizes?per_page=200",
                                 headers={"Authorization": "Bearer " + token()})
    sizes = json.load(urllib.request.urlopen(req, timeout=60))["sizes"]
    prefixes = [NVIDIA[w] for w in want]
    return [(z["slug"], z["price_hourly"], z["regions"]) for z in sizes
            if z["available"] and z["regions"] and any(z["slug"].startswith(p) for p in prefixes)
            and "spot" not in z["slug"]]


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def notify(text):
    try:
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import notify as N
        N.send(text, title="DO GPU")
    except Exception as e:  # noqa: BLE001
        log(f"notify failed: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", default="h100,h200,l40s,6000ada,4000ada")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--max-hours", type=float, default=48)
    a = ap.parse_args()
    want = [w.strip() for w in a.want.split(",") if w.strip() in NVIDIA]
    t0, seen = time.time(), set()
    log(f"watch start: {want}, every {a.interval}s")
    while time.time() - t0 < a.max_hours * 3600:
        try:
            hits = available(want)
        except Exception as e:  # noqa: BLE001
            log(f"check failed: {e!r}")
            time.sleep(a.interval)
            continue
        new = {h[0] for h in hits} - seen
        if hits:
            log("available: " + "; ".join(f"{s} ${p}/h {r}" for s, p, r in hits))
        if new:
            notify("DigitalOcean GPU free: " + ", ".join(sorted(new)))
            if a.once:
                return
        seen = {h[0] for h in hits}
        time.sleep(a.interval)
    log("watch ended (max hours)")


if __name__ == "__main__":
    main()
