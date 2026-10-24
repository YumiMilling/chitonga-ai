"""Put one pod run FIRST in internal/chitonga_train/runs/ACTIVE_RUN.json, the registry of the zamai.pro/beelink Training
panel (tools/beelink/training.py), keeping every other entry as it was (24 Oct 2026, the Jesuit; the reader bake-off's
translator pod). Called by launch_translator_pod.sh the moment the machine exists.

    python -X utf8 tools/chitonga/train/register_run.py <id> <pod_id> <card> <usd_hr> <cap_min> <launch.log> <host_file> <purpose>

host_file: the file the launcher writes with 'user@ip port' (outside the repo); the address itself never goes in here.
"""
import datetime as dt
import json
import os
import re
import sys

REG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                    "internal", "chitonga_train", "runs", "ACTIVE_RUN.json"))


def main(rid, pid, card, price, cap, log, hf, purpose):
    j = json.load(open(REG, encoding="utf-8"))
    j["runs"] = [r for r in j["runs"] if r.get("id") != rid]
    e = {"id": rid, "title": "Translator pod (served a050) for " + purpose, "provider": "runpod",
         "machine": "chitonga-translator", "machine_id": pid, "card": card, "usd_hr": float(price),
         "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "estimate_min": 120,
         "cap_min": int(cap), "log": log.replace("\\", "/"), "remote_log": "/root/run.log",
         "done_marker": "SERVER STOPPED", "fail_marker": "TPOD_FAILED", "tail_min": 10, "plan": []}
    if hf:   # the launcher is bash (~); the panel is Windows Python (~)
        hf = re.sub(r"^/([a-zA-Z])/", lambda m: m.group(1).upper() + ":/", hf)
        e["ssh"] = {"host_file": hf, "key": "~/.ssh/zamai_sandbox"}
    j["runs"].insert(0, e)
    tmp = REG + ".tmp"
    open(tmp, "w", encoding="utf-8", newline="\n").write(json.dumps(j, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, REG)
    print("REGISTRY", rid, "first of", len(j["runs"]), [r["id"] for r in j["runs"]])


if __name__ == "__main__":
    main(*sys.argv[1:9])
