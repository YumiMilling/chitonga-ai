"""Start, inspect and DESTROY one DigitalOcean GPU droplet for a Chitonga job (19 Oct 2026, the Jesuit; Chris: put the
$55 prepaid DO credit to work). The DO twin of runpod.py, with the same command names.

    python tools/chitonga/train/do_gpu.py sizes                       GPU sizes, price, regions with capacity
    python tools/chitonga/train/do_gpu.py up [size[@region] ...]      create one droplet (first size/region that works);
                                                                      prints JSON {id, name, size, region, price_hourly}
    python tools/chitonga/train/do_gpu.py status                      our droplets (tag chitonga), state and SSH address
    python tools/chitonga/train/do_gpu.py down [name]                 DESTROY (never power off: an off droplet still
                                                                      bills) the named droplet, or every chitonga droplet
Env: DO_NAME (default chitonga-gpu), DO_IMAGE (default gpu-h100x1-base, DO's "NVIDIA AI/ML Ready Image": driver + CUDA;
AMD sizes get gpu-amd-base). Token DO_API_TOKEN in zamai-pro/.env.local; SSH key "keyzamai" (id 59775433, the same key
as ~/.ssh/zamai_sandbox). Log in as root.
Account limits seen 19 Oct: droplet limit 10; CPU droplets up to 16 GB RAM only; GPU sizes from $0.76/h (RTX 4000 Ada
20 GB) to H200 $4.47/h; the sizes listing shows capacity per region (often empty: capacity comes and goes).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TOK = [l.split("=", 1)[1].strip() for l in open(os.path.join(ROOT, ".env.local"), encoding="utf-8") if l.startswith("DO_API_TOKEN=")][0]
KEY_ID = 59775433
TAG = "chitonga"
API = "https://api.digitalocean.com/v2/"


def call(method, path, body=None):
    r = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Authorization": "Bearer " + TOK, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            t = f.read()
            return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"DO API {method} {path}: HTTP {e.code} {e.read().decode()[:400]}")


def droplets():
    return call("GET", f"droplets?tag_name={TAG}&per_page=100")["droplets"]


def ip(d):
    v4 = [n["ip_address"] for n in d["networks"]["v4"] if n["type"] == "public"]
    return v4[0] if v4 else None


cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
if cmd == "sizes":
    for s in call("GET", "sizes?per_page=200")["sizes"]:
        if s["slug"].startswith("gpu-"):
            print(f"{s['slug']:28} {s['memory'] // 1024:5} GB RAM {s['vcpus']:4} vCPU {s['disk']:5} GB disk ${s['price_hourly']:.2f}/h regions {s['regions']}")
elif cmd == "up":
    name = os.environ.get("DO_NAME", "chitonga-gpu")
    if [d for d in droplets() if d["name"] == name]:
        raise SystemExit(f"a droplet named {name} already exists: run status / down first")
    sizes = {s["slug"]: s for s in call("GET", "sizes?per_page=200")["sizes"]}
    want = sys.argv[2:] or ["gpu-h200x1-141gb@nyc2"]
    errors = []
    for w in want:
        slug, _, region = w.partition("@")
        regions = [region] if region else (sizes.get(slug, {}).get("regions") or ["nyc2", "tor1", "atl1", "nyc1", "nyc3", "sfo3", "ams3", "fra1", "lon1"])
        image = os.environ.get("DO_IMAGE") or ("gpu-amd-base" if "-mi" in slug else "gpu-h100x1-base")
        for rg in regions:
            body = {"name": name, "region": rg, "size": slug, "image": image, "ssh_keys": [KEY_ID], "tags": [TAG], "monitoring": False}
            try:
                d = call("POST", "droplets", body)["droplet"]
            except SystemExit as e:
                errors.append(f"{slug}@{rg}: {str(e)[:220]}")
                continue
            print(json.dumps({"id": d["id"], "name": d["name"], "size": slug, "region": rg, "image": image,
                              "price_hourly": sizes.get(slug, {}).get("price_hourly")}))
            raise SystemExit(0)
    print("\n".join(errors))
    raise SystemExit("no droplet could be created")
elif cmd == "status":
    for d in droplets():
        a = ip(d)
        print(d["id"], d["name"], d["status"], d["size_slug"], d["region"]["slug"], f"${d['size']['price_hourly']}/h",
              "|", f"ssh root@{a} -p 22" if a and d["status"] == "active" else "no ssh yet")
elif cmd == "down":
    for d in droplets():
        if (d["name"] == sys.argv[2]) if len(sys.argv) > 2 else d["name"].startswith("chitonga-"):
            call("DELETE", f"droplets/{d['id']}")
            print("destroyed", d["id"], d["name"])
    for _ in range(12):
        left = [d for d in droplets() if ((d["name"] == sys.argv[2]) if len(sys.argv) > 2 else d["name"].startswith("chitonga-"))]
        if not left:
            break
        time.sleep(5)
    print("remaining droplets:", len(droplets()))
else:
    raise SystemExit(__doc__)
