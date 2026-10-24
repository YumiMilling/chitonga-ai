"""End to end over the education plug-in's frozen test set (23 Oct 2026, the Jesuit): every request as the first message
of a new teacher, through the whole engine (gates, intake, reader, confirmation, delivery), rules-only reader by
default. Measures: coverage (requests that reach an accepted target: unit, item, route, intent or honest outcome),
how many went through a confirmation (and whether 'Yes' then delivers), how many the rules could not place (these
would go to the model reader), and time per turn. K0, no model, nothing sent.

    python -X utf8 tools/chitonga/harness/plugins/education/run_testset.py [--out DIR]
Writes internal/chitonga_train/harness/runs/<stamp>_rules/{results.jsonl, summary.json, sheet.md}.
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HARNESS)

from core import io as IO  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.log import TurnLog  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402

REPO = os.path.normpath(os.path.join(HARNESS, "..", "..", ".."))


def reached(P, rec):
    """the targets this turn reached: {'item:..', 'unit:..', 'route:..', 'intent:..', 'outcome:..'}"""
    got = set()
    r = rec.get("reading") or {}
    for p in rec["path"]:
        if p.startswith("deliver:"):
            it = p.split(":", 1)[1].rsplit(":", 1)[0]
            got.add(it if it.count(":") else "item:" + it)
            u = P.unit_of.get(it)
            if u:
                got.add("unit:" + u)
        elif p.startswith("confirm:") and p not in ("confirm:yes", "confirm:no"):
            it = p.split(":", 1)[1]
            got.add(it)
            if P.unit_of.get(it):
                got.add("unit:" + P.unit_of[it])
        elif p.startswith("route:"):
            got.add(p.split(":", 1)[1])
        elif p.startswith("no_content:unit:"):
            got.add("unit:" + p.split(":")[-1])
        elif p.startswith("health_no_card"):
            got.add("outcome:health_no_card")
        elif p.startswith("observation:"):
            got.add("intent:struggle")
        elif p in ("redirect", "encourage", "safeguarding"):
            got.add("outcome:" + p)
        elif p == "needs_model":
            body = " ".join(x.get("body", "") for x in rec["replies"])
            got.add("outcome:health_no_card" if "health material" in body else "outcome:no_material")
    # the item each lesson delivery starts with also counts for its fixed id form ('lesson:..')
    got |= {g.replace("item:", "") for g in got if g.startswith("item:")}
    if r.get("unit"):
        got.add("unit:" + r["unit"])
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P, hooks = education.load()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = a.out or os.path.join(REPO, "internal", "chitonga_train", "harness", "runs", f"{stamp}_rules")
    os.makedirs(out, exist_ok=True)
    rows = []
    t_all = time.time()
    for i, t in enumerate(P.testset):
        mem = SQLiteMemory(":memory:")
        e = Engine(P, mem, hooks=hooks, log=TurnLog(os.path.join(out, "turns.jsonl")))
        n = f"2609990{i:05d}"
        t0 = time.time()
        replies = e.handle(n, {"text": t["text"]})
        dt = time.time() - t0
        rec = e.last
        got = reached(P, rec)
        confirm = any(p.startswith("confirm:") for p in rec["path"])
        after_yes = None
        if confirm:
            r2 = e.handle(n, {"choice": "c:yes"})
            after_yes = [p for p in e.last["path"] if p.startswith("deliver:") or p.startswith("health")]
        ok = bool(set(t["accept"]) & got)
        rows.append({"id": t["id"], "text": t["text"], "lang": t["lang"], "accept": t["accept"], "reached": sorted(got),
                     "ok": ok, "confirm": confirm, "after_yes": after_yes, "needs_model": "needs_model" in rec["path"],
                     "reading": {k: (rec["reading"] or {}).get(k) for k in ("intent", "unit", "item", "confidence",
                                                                          "source", "why")},
                     "path": rec["path"], "origins": rec["origins"], "flags": rec["flags"], "s": round(dt, 3),
                     "reply": IO.as_text(replies[0])[:300] if replies else ""})
    with open(os.path.join(out, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    times = [r["s"] for r in rows]
    show = [r for r in rows if r["id"].startswith("SC")]
    mine = [r for r in rows if r["id"].startswith("J")]
    summ = {"plugin": P.name, "version": P.manifest["version"], "reader": "rules", "n": len(rows),
            "ok": sum(r["ok"] for r in rows), "ok_showcase": f"{sum(r['ok'] for r in show)}/{len(show)}",
            "ok_jesuit": f"{sum(r['ok'] for r in mine)}/{len(mine)}",
            "via_confirm": sum(r["confirm"] for r in rows),
            "confirm_then_delivered": sum(bool(r["after_yes"]) for r in rows if r["confirm"]),
            "needs_model": sum(r["needs_model"] for r in rows),
            "needs_model_ids": [r["id"] for r in rows if r["needs_model"]],
            "failed_ids": [r["id"] for r in rows if not r["ok"]],
            "s_median": round(statistics.median(times), 3), "s_max": round(max(times), 3),
            "s_total": round(time.time() - t_all, 1), "out": out}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    with open(os.path.join(out, "sheet.md"), "w", encoding="utf-8") as f:
        f.write(f"# Harness test set, rules-only reader ({stamp})\n\n{json.dumps(summ)}\n\n"
                "| id | request | ok | confirm | reached | reading |\n|---|---|---|---|---|---|\n")
        for r in rows:
            rd = r["reading"]
            f.write(f"| {r['id']} | {r['text']} | {'yes' if r['ok'] else 'NO'} | {'yes' if r['confirm'] else ''} | "
                    f"{', '.join(r['reached'][:4])} | {rd.get('intent')} {rd.get('confidence')} {rd.get('source')} |\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
