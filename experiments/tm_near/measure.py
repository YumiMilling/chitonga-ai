"""Before / after for translator optimisation step 1 (24 Oct 2026, the Jesuit; Chris approved, K0): the lesson Tonga in
the translation memory and near memory rows as rerank candidates (translator_service step 6d). Beelink CPU, no pod,
nothing sent. Run it through run_measure.py (the Beelink lock, name "tm-near").

ARMS (all with the service's default readers and seeds, memory ON: the serving configuration)
  before   h2_memory.LESSONS = False, cfg tm_near False: the memory and the decode as before step 1 (the one thing of
           step 1 that has no switch, a guarded WHOLE input giving its pieces no memory, is counted separately:
           'pieces of a guarded input with an exact memory hit')
  loose    LESSONS = True, tm_near on, tm_near_strict False (differs-false only): translated for real, keep=True
  strict   the DEFAULT configuration: re-decided from the loose run with the non-strict tm-near candidates removed
  lessons  LESSONS = True, tm_near off: re-decided from the loose run with every tm-near candidate removed
Re-deciding is exact: decide() is pure, and every tm-near candidate is checked ALONE and apart from the model's pool
(translator_service.check), so removing one changes no other candidate's facts.

SETS (read-only)
  p1      the 174 P1 items (114 en>toi, 60 toi>en) of the stage 16 pre-screen: the zero-collateral gate of the glossary
          and forbid-rule work (LOGBOOK 24 Oct 01:45-03:00: 'P1 174: 0 lines changed')
  maths   the 16 maths pre-test sentences (glossary run's loader)
  probes  the 13 confusable probes of gate 4b
  dlines  100 English classroom lines whose Tonga is still D (derived, needs confirmation) in the lesson boxes, drawn
          with seed 2410: the lines the assistant serves where no teacher has written the Tonga yet
  NOT the stage 18 frozen 44 nor its dev slice: they stay for the kill test (and the memory guard blocks them anyway).

    python -X utf8 internal/chitonga_train/runs/2026-10-24_tm_near/measure.py [p1|maths|probes|dlines|all]
Out: <set>.json in this folder and a printed summary (raw numbers, no verdict words).
"""
import copy
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "llm"))
CT = os.path.join(REPO, "internal", "chitonga_train")
GL = os.path.join(CT, "runs", "2026-10-24_glossary_lever")


def items(which):
    if which in ("p1", "maths", "probes"):
        import importlib.util            # the glossary run's loaders (the same 174 / 16 / 13 items); its module is
        sp = importlib.util.spec_from_file_location("glossary_measure", os.path.join(GL, "measure.py"))  # also 'measure'
        GLM = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(GLM)
        return GLM.items(which)
    if which == "dlines":
        import h2_memory as M
        seen, pool = set(), []
        for f in M._box_files():
            sec = ""
            for l in open(f, encoding="utf-8"):
                if l.startswith("## "):
                    sec = l.strip()
                    continue
                if not l.startswith("| ") or not sec.startswith("## Classroom"):
                    continue
                c = [x.strip() for x in l.strip().strip("|").split("|")]
                if len(c) >= 4 and c[3].startswith("D") and c[1] and c[1] != "English" and M.norm(c[1]) not in seen:
                    seen.add(M.norm(c[1]))
                    rel = os.path.relpath(f, CT).replace("\\", "/")
                    pool.append({"id": f"{rel}#{c[0]}", "en": c[1], "dir": "en>toi", "ref": None, "d_toi": c[2]})
        random.Random(2410).shuffle(pool)
        return pool[:100]
    raise SystemExit(f"unknown set {which}")


def redecide(r, cfg, drop):
    """the output text of a keep=True translation with the candidates `drop(c)` removed, decided again"""
    import translator_service as TS
    if not r.get("plan"):
        return r["text"], []
    outs, chosen_tm = [], []
    for q in r["pieces"]:
        if q.get("stage") == "memory" or not q.get("candidates"):
            outs.append(q.get("chosen", ""))
            continue
        cs = [c for c in copy.deepcopy(q["candidates"]) if not drop(c)]
        j, _ = TS.decide(cs, r["direction"], cfg)
        outs.append(TS.render(q, cs[j]["text"]))
        if cs[j].get("tm"):
            chosen_tm.append({"piece": q["src"], **cs[j]["tm"], "chosen": outs[-1]})
    return TS.assemble(r["plan"], outs), chosen_tm


def pieces_of(r):
    return [q.get("chosen", "") for q in r.get("pieces", [])] if r.get("plan") else None


def run(which):
    import h2_memory as M
    import translator_service as TS
    its = items(which)
    t0 = time.time()
    # ---- before: lessons off, tm-near off
    M.LESSONS = False
    M.memory()
    s0 = TS.Service({"tm_near": False})
    before = []
    for n, it in enumerate(its, 1):
        before.append(s0.translate(it["en"], it["dir"], keep=True))
        if n % 20 == 0:
            print(f"  [{which} before] {n}/{len(its)} {round(time.time() - t0)}s", flush=True)
    # what the 'blocked whole input' rule (no switch) takes away: pieces of a guarded input that the memory would serve
    blocked_pieces = 0
    for it, b in zip(its, before):
        if b.get("memory_blocked"):
            for q in b.get("pieces", []):
                if s0.memory(q["src"], it["dir"]):          # memory() itself refuses a guarded piece
                    blocked_pieces += 1
    # ---- loose: lessons on, tm-near on, strict off (the superset of tm-near candidates)
    M.LESSONS = True
    M.memory()
    s1 = TS.Service({"tm_near_strict": False})
    s1._r = s0._r                                         # the same loaded models
    rows = []
    for n, (it, b) in enumerate(zip(its, before), 1):
        r = s1.translate(it["en"], it["dir"], keep=True)
        strict_txt, strict_tm = redecide(r, TS.CFG, lambda c: bool(c.get("tm")) and not c["tm"].get("strict"))
        less_txt, _ = redecide(r, TS.CFG, lambda c: bool(c.get("tm")))
        offered = [c for q in r.get("pieces", []) for c in (q.get("candidates") or []) if c.get("tm")]
        rows.append({
            "id": it["id"], "dir": it["dir"], "en": it["en"], "ref": it.get("ref"),
            "before": b["text"], "before_stage": b["stage"], "before_pieces": pieces_of(b),
            "strict": strict_txt, "loose": r["text"], "lessons": less_txt, "after_stage": r["stage"],
            "after_pieces": pieces_of(r), "memory_blocked": r.get("memory_blocked"),
            "tm_offered": len(offered), "tm_offered_strict": sum(1 for c in offered if c["tm"].get("strict")),
            "tm_offered_list": [{"text": c["text"], **c["tm"], "facts": {k: c.get(k) for k in
                                ("copy", "lost", "hard", "meaning", "gmiss", "why")}} for c in offered],
            "tm_chosen_loose": r.get("tm_near", []), "tm_chosen_strict": strict_tm,
            "memory_before": b.get("memory", {}).get("src") if b["stage"] == "memory" else
            [q.get("why") for q in b.get("pieces", []) if q.get("stage") == "memory"],
            "memory_after": r.get("memory", {}).get("src") if r["stage"] == "memory" else
            [q.get("why") for q in r.get("pieces", []) if q.get("stage") == "memory"],
            "alarms_before": b["alarms"], "alarms_after": r["alarms"],
            **{k: v for k, v in it.items() if k in ("expect", "forbid", "a050_23oct", "d_toi")}})
        if n % 20 == 0:
            print(f"  [{which} after] {n}/{len(its)} {round(time.time() - t0)}s", flush=True)
    json.dump({"set": which, "blocked_input_pieces_lost": blocked_pieces, "rows": rows,
               "seconds": round(time.time() - t0)},
              open(os.path.join(HERE, f"{which}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report(which, rows, blocked_pieces)
    return rows


def _pc(a, b):
    """(pieces compared, pieces changed) between two piece lists; a whole-line memory answer counts as one piece"""
    if a is None or b is None or len(a) != len(b):
        return 1, 0
    return len(a), sum(1 for x, y in zip(a, b) if (x or "").strip() != (y or "").strip())


def report(which, rows, blocked_pieces=0):
    print(f"\n== {which}: {len(rows)} lines")
    for arm in ("lessons", "strict", "loose"):
        ch = [x for x in rows if x[arm].strip() != x["before"].strip()]
        print(f"   {arm:8s} vs before: lines changed {len(ch)} of {len(rows)}")
    npieces = sum(len(x["after_pieces"]) if x["after_pieces"] is not None else 1 for x in rows)
    pc_loose = sum(_pc(x["before_pieces"], x["after_pieces"])[1] for x in rows)
    print(f"   pieces (loose run): {npieces}; changed vs before (piece by piece where the layout is the same): {pc_loose}")
    print(f"   whole-line memory answers: before {sum(x['before_stage'] == 'memory' for x in rows)}, "
          f"after {sum(x['after_stage'] == 'memory' for x in rows)}; guarded whole inputs {sum(bool(x['memory_blocked']) for x in rows)}"
          f" (pieces of them the memory would have served: {blocked_pieces})")
    print(f"   tm-near offered: {sum(x['tm_offered'] for x in rows)} candidates on {sum(1 for x in rows if x['tm_offered'])} "
          f"lines ({sum(x['tm_offered_strict'] for x in rows)} strict); chosen: strict {sum(len(x['tm_chosen_strict']) for x in rows)}, "
          f"loose {sum(len(x['tm_chosen_loose']) for x in rows)}")
    for x in rows:
        if x["loose"].strip() != x["before"].strip() or x["strict"].strip() != x["before"].strip():
            print(f"   - {x['id']} [{x['dir']}]")
            print(f"       SRC:     {x['en'][:110]}")
            print(f"       before:  {x['before'][:110]}")
            if x["lessons"] != x["before"]:
                print(f"       lessons: {x['lessons'][:110]}   (memory: {str(x['memory_after'])[:90]})")
            if x["strict"] != x["lessons"]:
                print(f"       strict:  {x['strict'][:110]}")
            if x["loose"] != x["strict"]:
                print(f"       loose:   {x['loose'][:110]}   {[ (c['near'][:50], c['ratio']) for c in x['tm_chosen_loose']]}")
            if x.get("ref"):
                print(f"       ref:     {str(x['ref'])[:110]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "probes"
    for w in (["probes", "maths", "p1", "dlines"] if which == "all" else [which]):
        run(w)
