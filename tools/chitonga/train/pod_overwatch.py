"""Independent pod overwatch (21 Oct 2026; Chris: "use the pod but check on it, if I hate one thing it's wasting money").

Runs beside any launcher's own watchdog. Every INTERVAL seconds: lists our pods and the account's spend per hour, logs
both, and TERMINATES every chitonga-* pod (runpod.py down) once any pod has been seen running longer than CAP_MIN
minutes, or when spend continues while no pod of ours is listed. Ends by itself after the last pod has been gone for
QUIET_MIN minutes (or after MAX_H hours).

    python -X utf8 tools/chitonga/train/pod_overwatch.py [cap_min=160] [max_h=8]
Log: internal/chitonga_train/runs/pod_overwatch.log
"""
import datetime as dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.join(HERE, "runpod.py")
LOG = os.path.normpath(os.path.join(HERE, "..", "..", "..", "internal", "chitonga_train", "runs", "pod_overwatch.log"))
INTERVAL, QUIET_MIN = 180, 30
CAP_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 160
MAX_H = float(sys.argv[2]) if len(sys.argv) > 2 else 8


def rp(cmd):
    r = subprocess.run([sys.executable, "-X", "utf8", RP, cmd], capture_output=True, text=True, timeout=120)
    return (r.stdout + r.stderr).strip()


def do_droplets(destroy=False):
    """our DigitalOcean droplets named chitonga-*: [(id, name, size, created)], destroying them when asked"""
    import json
    import re
    import urllib.request
    env = {}
    for l in open(os.path.join(HERE, "..", "..", "..", ".env.local"), encoding="utf-8"):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)", l)
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"')
    h = {"Authorization": "Bearer " + env["DO_API_TOKEN"]}
    req = urllib.request.Request("https://api.digitalocean.com/v2/droplets?per_page=100", headers=h)
    ds = json.load(urllib.request.urlopen(req, timeout=60))["droplets"]
    ours = [(d["id"], d["name"], d["size_slug"], d["created_at"]) for d in ds if d["name"].startswith("chitonga")]
    if destroy:
        for d in ours:
            urllib.request.urlopen(urllib.request.Request(f"https://api.digitalocean.com/v2/droplets/{d[0]}",
                                                          headers=h, method="DELETE"), timeout=60)
    return ours


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    t0 = time.time()
    first_seen, last_seen, ever = {}, None, False
    log(f"overwatch start: cap {CAP_MIN} min, max {MAX_H} h" + (" (RunPod only)" if os.environ.get("OVERWATCH_RUNPOD_ONLY") == "1" else ""))
    while time.time() - t0 < MAX_H * 3600:
        try:
            st, me = rp("status"), rp("me")
        except Exception as e:  # noqa: BLE001
            log(f"check failed: {e!r}")
            time.sleep(INTERVAL)
            continue
        pods = [l for l in st.splitlines() if "chitonga" in l.lower()]
        try:  # DigitalOcean droplets count too (21 Oct: generation may run on the DO credit)
            # OVERWATCH_RUNPOD_ONLY=1 (24 Oct): a RunPod job's overwatch leaves DO alone, so it can never destroy another
            # job's droplet (the stage 19 droplet ran beside the translator pod, with its own cap and KEEP rule)
            for d in ([] if os.environ.get("OVERWATCH_RUNPOD_ONLY") == "1" else do_droplets()):
                age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(d[3].replace("Z", "+00:00")))
                pods.append(f"do:{d[0]} {d[1]} {d[2]} age {age.total_seconds() / 60:.0f} min")
                if age.total_seconds() / 60 >= CAP_MIN:
                    log(f"DO CAP REACHED for {d[1]}: destroying")
                    try:
                        do_droplets(destroy=True)
                    except Exception as e:  # noqa: BLE001  (21 Oct: a token without delete rights)
                        log(f"DO DESTROY FAILED: {e!r}")
                        subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "..", "..", "notify.py"),
                                        f"DO droplet {d[1]} could NOT be destroyed: delete it by hand", "--title",
                                        "Tonga pod", "--high"], capture_output=True, timeout=60)
        except Exception as e:  # noqa: BLE001
            log(f"DO check failed: {e!r}")
        spend = me.split("spend/hr $")[-1].split("|")[0].strip() if "spend/hr" in me else "?"
        now = time.time()
        for p in pods:
            first_seen.setdefault(p.split()[0], now)
        if pods:
            ever, last_seen = True, now
        oldest = max((now - t for k, t in first_seen.items() if any(k in p for p in pods)), default=0) / 60
        log(f"pods {len(pods)} | spend/hr ${spend} | oldest seen {oldest:.0f} min | " + " || ".join(pods)[:300])
        if pods and oldest >= CAP_MIN:
            log(f"CAP REACHED ({oldest:.0f} >= {CAP_MIN} min): terminating every chitonga pod")
            log(rp("down")[:400])
        try:
            if not pods and float(spend) > 0:
                log(f"WARNING: spend ${spend}/hr with no chitonga pod listed: check the account")
        except ValueError:
            pass
        if ever and not pods and last_seen and now - last_seen > QUIET_MIN * 60:
            log("all pods gone for 30 min: overwatch ends")
            return
        time.sleep(INTERVAL)
    log("overwatch max time reached")


if __name__ == "__main__":
    main()
