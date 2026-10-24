"""Start, inspect and terminate one RunPod GPU pod for a Chitonga training run (10 Oct 2026).

    python tools/chitonga/train/runpod.py me                      balance and whether our SSH key is registered
    python tools/chitonga/train/runpod.py key                     register ~/.ssh/zamai_sandbox.pub on the account
    python tools/chitonga/train/runpod.py up [gpu type id]        deploy one on-demand pod (default RTX 4090, then A40)
    python tools/chitonga/train/runpod.py status                  our pods, their state and SSH address
    python tools/chitonga/train/runpod.py down                    TERMINATE every pod named chitonga-* (not stop: stop bills disk)
Key: RUNPOD_API_KEY in zamai-pro/.env.local. The account has auto-pay off and a $10 balance, so nothing can cost more.
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
KEY = [l.split("=", 1)[1].strip() for l in open(os.path.join(ROOT, ".env.local"), encoding="utf-8") if l.startswith("RUNPOD_API_KEY=")][0]
IMAGE = os.environ.get("CHITONGA_IMAGE", "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04")  # torch 2.8 preinstalled (13 Oct): no pip at boot


def gql(q, v=None):
    r = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q, "variables": v or {}}).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY, "User-Agent": "zamai"})
    out = json.load(urllib.request.urlopen(r))
    if out.get("errors"):
        raise SystemExit(json.dumps(out["errors"])[:600])
    return out["data"]


def pods():
    return gql("query { myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } } }")["myself"]["pods"]


cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
if cmd == "me":
    m = gql("query { myself { clientBalance currentSpendPerHr pubKey } }")["myself"]
    print("balance $", m["clientBalance"], "| spend/hr $", m["currentSpendPerHr"], "| our key registered:", "zamai-sandbox" in (m["pubKey"] or ""))
elif cmd == "key":
    pub = open(os.path.expanduser("~/.ssh/zamai_sandbox.pub"), encoding="utf-8").read().strip()
    have = gql("query { myself { pubKey } }")["myself"]["pubKey"] or ""
    if pub.split()[1] in have:
        print("already registered")
    else:
        gql("mutation($k:String!){ updateUserSettings(input:{pubKey:$k}){ id } }", {"k": (have.strip() + "\n" + pub).strip()})
        print("registered")
elif cmd == "up":
    NAME = os.environ.get("CHITONGA_NAME", "chitonga-gpu")      # 14 Oct: a second pod in parallel needs its own name
    if [p for p in pods() if p["name"] == NAME]:
        raise SystemExit(f"a pod named {NAME} already exists: run status / down first")
    for gpu in ([sys.argv[2]] if len(sys.argv) > 2 else ["NVIDIA GeForce RTX 4090", "NVIDIA A40"]):
        clouds = ("COMMUNITY", "SECURE") if os.environ.get("CHITONGA_CLOUD", "COMMUNITY") == "COMMUNITY" else ("SECURE", "COMMUNITY")
        if os.environ.get("CHITONGA_CLOUD_ONLY"):          # 23 Oct (translator pod): no fallback to the dearer cloud
            clouds = clouds[:1]
        for cloud in clouds:   # Community ~1/3 cheaper (12 Oct)
            try:
                p = gql("""mutation($i:PodFindAndDeployOnDemandInput){ podFindAndDeployOnDemand(input:$i){ id name costPerHr machine { gpuDisplayName } } }""",
                        {"i": {"name": NAME, "gpuTypeId": gpu, "gpuCount": 1, "cloudType": cloud, "imageName": IMAGE,
                               "containerDiskInGb": int(os.environ.get("CHITONGA_DISK", "30")), "volumeInGb": 0, "ports": os.environ.get("CHITONGA_PORTS", "22/tcp"), "startSsh": True, "supportPublicIp": True}})
                print(json.dumps(p["podFindAndDeployOnDemand"]))
                raise SystemExit(0)
            except SystemExit as e:
                if e.code == 0:
                    raise
                print(f"{gpu} / {cloud}: {str(e)[:200]}")
    raise SystemExit("no pod could be deployed")
elif cmd == "status":
    for p in pods():
        ssh = [f"ssh root@{x['ip']} -p {x['publicPort']}" for x in ((p.get("runtime") or {}).get("ports") or []) if x["privatePort"] == 22 and x["isIpPublic"]]
        print(p["id"], p["name"], p["desiredStatus"], (p.get("machine") or {}).get("gpuDisplayName"), "$/hr", p["costPerHr"], "|", ssh[0] if ssh else "no ssh yet")
elif cmd == "info":       # info <name>: uptime and every port mapping (23 Oct: a 3090 never showed a public ssh port)
    for p in pods():
        if p["name"] == sys.argv[2]:
            rt = p.get("runtime") or {}
            print(p["id"], p["desiredStatus"], "uptime", rt.get("uptimeInSeconds"), json.dumps(rt.get("ports")))
elif cmd == "down":       # down [name]: only that pod; without a name every chitonga-* pod
    for p in pods():
        if (p["name"] == sys.argv[2]) if len(sys.argv) > 2 else p["name"].startswith("chitonga-"):
            gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": p["id"]})
            print("terminated", p["id"], p["name"])
    print("remaining pods:", len(pods()))
