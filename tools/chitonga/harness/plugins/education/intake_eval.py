"""Does the rules-first intake catch a Tonga request before any model? (24 Oct 2026, the Jesuit; K0, no model.)

Measures the education plug-in's deterministic intake on four sets, so a change to the keyword table can be judged
BEFORE and AFTER on the same numbers instead of on a feeling:

  A  plugin      the plug-in's own frozen test set (testset.jsonl, 34): coverage, as run_testset.py reports it.
  B  chat_v13    the chat round's held-out requests (llm/chat_v10_tests/routing_v13_test.jsonl, 61), rules only.
                 Their gold is written for the chat reader, so it is mapped to harness targets here (GOLD_MAP):
                 lesson/card ids as they are, word -> route:language, marking -> route:marking, health_no_card /
                 redirect / social / safeguarding / no_material -> the harness outcome. 'advice' has no harness
                 target at all (the harness answers class struggles through strategies), so those rows are counted
                 apart and never as a failure of the intake.
  C  real        T01's 8 real requests (desk/real_requests_2026-10-23/T01_answers_v2.md). Most are out of
                 scope on purpose (TB, a map of Zambia, the smallest country, the skeleton). The number that matters
                 is 'caught': a request routed to a lesson, card or unit. It must NOT rise: a confident wrong lesson
                 is worse than the honest "I have no checked material".
  D  toi_probe   20 Tonga requests built ONLY by combining teacher-written phrases and words (the file names the
                 source of every piece; derived, never speaker-checked, attest-flagged). They test the matcher's
                 reach on the teachers' own vocabulary, not a teacher's real phrasing.

    python -X utf8 tools/chitonga/harness/plugins/education/intake_eval.py --tag before
    python -X utf8 tools/chitonga/harness/plugins/education/intake_eval.py --tag after  --compare <before.json>

Writes internal/chitonga_train/harness/runs/<stamp>_intake_<tag>/{results.jsonl, summary.json, sheet.md} and prints
the summary. Rules-only reader (no translator, no Gemma), each request as the first message of a new teacher.
"""
import argparse
import datetime
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HARNESS)
sys.path.insert(0, HERE)

from core import io as IO  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.log import TurnLog  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402
from run_testset import reached  # noqa: E402

REPO = os.path.normpath(os.path.join(HARNESS, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
ROUTING = os.path.join(CT, "llm", "chat_v10_tests", "routing_v13_test.jsonl")
REAL = os.path.join(CT, "desk", "real_requests_2026-10-23", "T01_answers_v2.md")
PROBE = os.path.join(CT, "harness", "intake_probe_toi_2026-10-24.jsonl")

# a target that means the rules placed the request somewhere (as against an honest "no material" / a plain menu)
PLACED = ("lesson:", "card:", "unit:", "route:", "intent:struggle")
HONEST = ("outcome:no_material", "outcome:unclear", "outcome:health_no_card")


def gold_targets(gold):
    """the chat-reader gold of routing_v13 as harness targets ([] = this row has no harness equivalent)"""
    out, pick = [], gold.get("pick") or {}
    for intent in gold.get("intent") or []:
        if intent == "lesson":
            out += ["lesson:" + x.split(":", 1)[1] for x in pick.get("lesson", [])]
        elif intent == "card":
            out += ["card:" + x for x in pick.get("card", [])]
        elif intent == "say":
            out += ["lesson:" + x.split(":", 1)[1].split("#")[0] for x in pick.get("say", [])]
        elif intent == "word":
            out.append("route:language")
        elif intent == "marking":
            out.append("route:marking")
        elif intent == "health_no_card":
            out.append("outcome:health_no_card")
        elif intent == "redirect":
            out.append("outcome:redirect")
        elif intent == "social":
            out.append("outcome:encourage")
        elif intent == "safeguarding":
            out.append("outcome:safeguarding")
        elif intent == "no_material":
            out += ["outcome:no_material", "outcome:unclear"]
    return list(dict.fromkeys(out))


# Accept changes on the held-out real requests, each with the old value and the reason. A change here follows a POLICY
# ruling, never a result (the held-out rule): the row's text is checked so a renumbered file cannot shift it.
C_ACCEPT_CHANGES = {
    "R1": {"text": "How would I define Tuberculosis",
           "add": ["card:c06"],
           "why": "2026-10-24 Chris: 'Nothing is held back, just label it for review' (DECISIONS 2026-10-24). The "
                  "labelled TB card c06 (served with 'Needs review: Health content ...') is a correct answer under "
                  "that ruling; the honest no-material outcomes stay accepted. Changed because of the ruling, not "
                  "because of a result."},
}


def load_sets(P):
    """[(set name, [{id, text, lang, accept, note}])]"""
    A = [{"id": t["id"], "text": t["text"], "lang": t["lang"], "accept": t["accept"], "note": t.get("origin", "")}
         for t in P.testset]
    B = []
    for ln in open(ROUTING, encoding="utf-8"):
        r = json.loads(ln)
        B.append({"id": r["id"], "text": r["request"], "lang": r["lang"], "accept": gold_targets(r["gold"]),
                  "note": "advice only" if not gold_targets(r["gold"]) else r.get("source", "")[:60],
                  "gold_intent": r["gold"]["intent"]})
    C = []
    with open(REAL, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^(\d)\.\s+(.+?)\s*$", line)
            if m and len(C) < 8:
                C.append({"id": f"R{m.group(1)}", "text": m.group(2), "lang": "mixed", "accept": list(HONEST),
                          "note": "held-out real request; honest outcome expected"})
                ch = C_ACCEPT_CHANGES.get(C[-1]["id"])
                if ch:
                    if ch["text"] != C[-1]["text"]:
                        raise SystemExit(f"C_ACCEPT_CHANGES {C[-1]['id']}: the text is now {C[-1]['text']!r}")
                    C[-1].update(accept_before_2026_10_24=list(C[-1]["accept"]), accept=C[-1]["accept"] + ch["add"],
                                 accept_change_why=ch["why"])
    D = []
    if os.path.isfile(PROBE):
        for ln in open(PROBE, encoding="utf-8"):
            r = json.loads(ln)
            D.append({"id": r["id"], "text": r["text"], "lang": r["lang"], "accept": r["accept"],
                      "note": r["built_from"]})
    return [("A_plugin_testset", A), ("B_chat_v13", B), ("C_real_requests", C), ("D_toi_probe", D)]


def run_one(P, hooks, out_dir, text):
    mem = SQLiteMemory(":memory:")
    e = Engine(P, mem, hooks=hooks, log=TurnLog(os.path.join(out_dir, "turns.jsonl")))
    n = "26099%08d" % (abs(hash(text)) % 10 ** 8)
    t0 = time.time()
    replies = e.handle(n, {"text": text})
    dt = time.time() - t0
    rec = e.last
    got = reached(P, rec)
    confirm = any(p.startswith("confirm:") for p in rec["path"])
    after_yes = None
    if confirm:
        e.handle(n, {"choice": "c:yes"})
        after_yes = [p for p in e.last["path"] if p.startswith("deliver:") or p.startswith("health")]
    return got, confirm, after_yes, rec, dt, replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="a previous summary.json: prints the difference per set")
    a = ap.parse_args()
    P, hooks = education.load()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = a.out or os.path.join(CT, "harness", "runs", f"{stamp}_intake_{a.tag}")
    os.makedirs(out, exist_ok=True)
    rows, summary = [], {"plugin_version": P.fingerprint, "tag": a.tag, "sets": {}}
    for name, items in load_sets(P):
        times = []
        for it in items:
            got, confirm, after_yes, rec, dt, replies = run_one(P, hooks, out, it["text"])
            times.append(dt)
            placed = sorted(g for g in got if g.startswith(PLACED))
            hit = bool(set(it["accept"]) & got) if it["accept"] else None
            rows.append({"set": name, "id": it["id"], "text": it["text"], "lang": it["lang"], "accept": it["accept"],
                         "reached": sorted(got), "placed": placed, "ok": hit, "confirm": confirm,
                         "after_yes": after_yes, "needs_model": "needs_model" in rec["path"],
                         "reading": {k: (rec["reading"] or {}).get(k) for k in ("intent", "unit", "item",
                                                                               "confidence", "source", "why")},
                         "path": rec["path"], "flags": rec["flags"], "note": it.get("note", ""),
                         **({"accept_before_2026_10_24": it["accept_before_2026_10_24"],
                             "accept_change_why": it["accept_change_why"]} if it.get("accept_before_2026_10_24") else {}),
                         "reply": IO.as_text(replies[0])[:200] if replies else ""})
        r = [x for x in rows if x["set"] == name]
        scored = [x for x in r if x["ok"] is not None and x["accept"]]
        summary["sets"][name] = {
            "n": len(r), "scored": len(scored), "hit": sum(1 for x in scored if x["ok"]),
            "placed": sum(1 for x in r if x["placed"]),
            "wrong_catch": sum(1 for x in scored if x["placed"] and not x["ok"]),
            "honest": sum(1 for x in r if any(g in HONEST for g in x["reached"])),
            "via_confirm": sum(1 for x in r if x["confirm"]),
            "needs_model": sum(1 for x in r if x["needs_model"]),
            "miss_ids": [x["id"] for x in scored if not x["ok"]],
            "wrong_catch_ids": [x["id"] for x in scored if x["placed"] and not x["ok"]],
            "s_median": round(statistics.median(times), 3) if times else None}
    with open(os.path.join(out, "results.jsonl"), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out, "sheet.md"), "w", encoding="utf-8") as f:
        f.write(f"# Intake evaluation ({a.tag}, {stamp}, plug-in {P.fingerprint})\n\n")
        f.write("| set | n | scored | hit | placed | wrong catch | honest | via confirm |\n|---|---|---|---|---|---|---|---|\n")
        for k, v in summary["sets"].items():
            f.write(f"| {k} | {v['n']} | {v['scored']} | {v['hit']} | {v['placed']} | {v['wrong_catch']} | "
                    f"{v['honest']} | {v['via_confirm']} |\n")
        f.write("\n| set | id | request | ok | reached | reading |\n|---|---|---|---|---|---|\n")
        for x in rows:
            rd = x["reading"]
            f.write(f"| {x['set'][:3]} | {x['id']} | {x['text'][:60]} | {'yes' if x['ok'] else ('NO' if x['ok'] is False else '-')} "
                    f"| {', '.join(x['reached'][:3])} | {rd.get('source')} {rd.get('confidence')} {str(rd.get('why'))[:60]} |\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if a.compare and os.path.isfile(a.compare):
        with open(a.compare, encoding="utf-8") as f:
            old = json.load(f)
        print("\nBEFORE -> AFTER")
        for k, v in summary["sets"].items():
            o = old["sets"].get(k, {})
            print(f"  {k:18s} hit {o.get('hit')} -> {v['hit']} (of {v['scored']}); placed {o.get('placed')} -> "
                  f"{v['placed']}; wrong catch {o.get('wrong_catch')} -> {v['wrong_catch']}; honest "
                  f"{o.get('honest')} -> {v['honest']}")
    print("out:", out)


if __name__ == "__main__":
    main()
