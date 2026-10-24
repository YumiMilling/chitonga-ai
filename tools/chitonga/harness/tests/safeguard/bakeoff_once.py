"""The reader bake-off's test rows through the FIXED safeguarding gate, ONCE (24 Oct 2026, the Jesuit).

The bake-off (internal/chitonga_train/runs/2026-10-24_reader_bakeoff/) found the gate missing seven kinds of disclosure.
The gate was then rebuilt by category and tuned on tests/safeguard/dev_v1.jsonl only. This script is the single
look at the bake-off rows afterwards: every row of testset_v2.jsonl (sha checked by the bake-off's own loader) through
the LIVE harness (RB_LIVE_HARNESS=1: the fixed engine and gate) with the bake-off's frozen plug-in data, for the rules
arm (no model) and for every model arm from its frozen replies (no model runs). Compared with the arm records the
bake-off wrote: safeguarding rows missed before / after, rows newly sent to safeguarding that are NOT safeguarding rows
(false positives on real requests), and correct rows before / after.

After it has run, the folder carries SAFEGUARD_ROWS_SEEN.md: these rows are no longer held out for safeguarding.
Refuses to run twice (the result file exists) unless --again is given, and then says so in the result.

    python -X utf8 tools/chitonga/harness/tests/safeguard/bakeoff_once.py
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

os.environ["RB_LIVE_HARNESS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
LLM = os.path.join(REPO, "tools", "chitonga", "llm")
sys.path.insert(0, LLM)
import reader_bakeoff as RB  # noqa: E402

RUN = os.path.join(REPO, "internal", "chitonga_train", "runs", "2026-10-24_reader_bakeoff")
OUT = os.path.join(RUN, "safeguard_fix_2026-10-24.json")

# speed only: turn() builds a new Engine per row and every Engine rebuilt the retrieval index (~1.6 s a turn here); the
# index is read-only, so one per plug-in object is reused. Behaviour unchanged.
import core.engine as _CE  # noqa: E402
_IX, _IX0 = {}, _CE.Index
_CE.Index = lambda p: _IX.setdefault(id(p), _IX0(p))
# and the rules reader compiles ~2,300 patterns per read, more than re's 512-entry cache holds: raise the cache
import re as _re  # noqa: E402
_re._MAXCACHE = 100000
FULL_ARMS = None               # None = every arm on all 1,164 rows (with the two caches a turn takes ~0.08 s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--again", action="store_true")
    ap.add_argument("--arms", default="all", help="'all', 'R' or a comma list of reply tags")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.again:
        sys.exit(f"already run once: {OUT}")
    ns = argparse.Namespace(testset=os.path.join(RUN, "testset_v2_en_noisy.jsonl"), work=RUN)   # what RULE v4 scored, N 1164
    rows, tsha = RB.load_testset(ns)
    P, hooks = RB.education.load(data_dir=os.path.join(RUN, "frozen", "plugin_education"))
    assert os.path.dirname(os.path.dirname(RB.core.__file__)).endswith(os.path.join("tools", "chitonga", "harness")), RB.core.__file__
    table, _ = RB.load_frozen(ns)
    fm = json.load(open(os.path.join(RUN, "freeze_manifest.json"), encoding="utf-8"))
    rdir = os.path.join(RUN, "replies")
    tags = sorted(f[:-6] for f in os.listdir(rdir) if f.endswith(".jsonl"))
    want = ["R"] + tags if a.arms == "all" else a.arms.split(",")
    sg_ids = {r["id"] for r in rows if "outcome:safeguarding" in r["accept"]}
    result = {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "again": bool(a.again), "testset_sha": tsha,
              "n_rows": len(rows), "n_safeguarding_rows": len(sg_ids), "plugin": P.fingerprint,
              "plugin_at_freeze": fm.get("plugin"), "harness": "live (RB_LIVE_HARNESS=1)",
              "rows_per_arm": "all 1164 for every arm",
              "note": "two earlier starts, no result seen: (1) stopped after a few minutes for speed (index rebuilt and "
                      "regexes recompiled per turn), no arm had finished, nothing printed or written; (2) 21:19 SAST, "
                      "pointed at the WRONG file (testset_v2.jsonl, 997 rows, not the 1,164 rows RULE v4 scored) and "
                      "crashed in its own bookkeeping after the rules arm, before printing or writing anything",
              "arms": {}}
    for tag in want:
        before_p = os.path.join(RUN, "arms", tag + ".jsonl")
        if not os.path.exists(before_p):
            continue
        before = {r["id"]: r for r in RB.jl(before_p)}
        if tag == "R":
            reader = RB.RulesReader()
            missing = None
        else:
            rep = RB.load_replies(os.path.join(rdir, tag + ".jsonl"))
            missing = {"n": 0}

            def post(body, k, rep=rep, missing=missing):
                r = rep.get(k)
                if r is None or r.get("err") or r.get("raw") is None:
                    missing["n"] += 1
                    raise RuntimeError("no reply for this body")
                return r["raw"]
            reader = RB.BodyReader(post, RB.FrozenBackends(table, None), "model:" + tag,
                                   fm["translator"]["renderer"] or "a050")
        t0 = time.time()
        res = {}
        arm_rows = rows if FULL_ARMS is None or tag in FULL_ARMS else [r for r in rows if r["id"] in sg_ids]
        if len(arm_rows) < len(rows):
            keep = {x["id"] for x in arm_rows}
            before = {i: r for i, r in before.items() if i in keep}
        for i, it in enumerate(arm_rows):
            try:
                res[it["id"]] = RB.turn(P, hooks, reader, it, i)
            except Exception as e:  # noqa: BLE001
                res[it["id"]] = {"id": it["id"], "error": repr(e)[:200], "path": [], "ok": False,
                                 "safeguarding_missed": it["id"] in sg_ids}
        sent = {i for i, r in res.items() if "safeguarding" in r.get("path", [])}
        sent_before = {i for i, r in before.items() if "safeguarding" in r.get("path", [])}
        arm = {
            "safeguarding_missed_before": sorted(i for i in sg_ids if before.get(i, {}).get("safeguarding_missed")),
            "safeguarding_missed_after": sorted(i for i in sg_ids if res[i].get("safeguarding_missed")),
            "false_positive_before": sorted(sent_before - sg_ids),
            "false_positive_after": sorted(sent - sg_ids),
            "newly_sent_not_safeguarding": sorted((sent - sent_before) - sg_ids),
            "ok_before": sum(1 for r in before.values() if r.get("ok")), "ok_after": sum(1 for r in res.values() if r.get("ok")),
            "ok_lost": sorted(i for i in before if before[i].get("ok") and not res.get(i, {}).get("ok")),
            "ok_gained": sorted(i for i in before if not before[i].get("ok") and res.get(i, {}).get("ok")),
            "errors": sorted(i for i, r in res.items() if r.get("error")),
            "replies_missing": missing["n"] if missing else None, "s": round(time.time() - t0, 1)}
        result["arms"][tag] = arm
        print(f"{tag}: sg missed {len(arm['safeguarding_missed_before'])} -> {len(arm['safeguarding_missed_after'])}; "
              f"FP {len(arm['false_positive_before'])} -> {len(arm['false_positive_after'])}; ok {arm['ok_before']} -> "
              f"{arm['ok_after']} (lost {len(arm['ok_lost'])}, gained {len(arm['ok_gained'])}); replies missing "
              f"{arm['replies_missing']}; {arm['s']} s", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        with open(os.path.join(RUN, "arms_after_safeguard_fix_" + tag + ".jsonl"), "w", encoding="utf-8") as f:
            for i in sorted(res):
                r = res[i]
                f.write(json.dumps({k: r.get(k) for k in ("id", "slice", "ok", "safeguarding_item", "safeguarding_missed",
                                                           "path", "error")}, ensure_ascii=False) + "\n")
    print("->", OUT)


if __name__ == "__main__":
    main()
