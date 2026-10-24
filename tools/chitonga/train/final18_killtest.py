"""The final round's KILL TEST, pre-registered in runs/2026-10-23_final_translator_data/PACK.md section 6 (23 Oct 2026,
the Jesuit), written before any stage 18 model exists. Runs on the Beelink after launch_final18.sh has confirmed the
machine gone. K0.

    python -X utf8 tools/chitonga/train/final18_killtest.py add <tsv>     # teacher lines that arrived AFTER the build ->
                                                                           # the frozen teacher test (en, toi, teacher, src)
    python -X utf8 tools/chitonga/train/final18_killtest.py freeze        # >= 24 items, none in the stage 18 data -> sha
    python -X utf8 tools/chitonga/train/final18_killtest.py plan          # the judging steps, in order
    python -X utf8 tools/chitonga/train/final18_killtest.py decide <results.json>   # applies the rule, prints the verdict
    python -X utf8 tools/chitonga/train/final18_killtest.py selftest      # the rule on made-up numbers (no model)

WHY a separate frozen teacher test: the 57 teacher sentences of 21-23 Oct are TRAINED in this round (test-then-train,
DECISIONS 22 Oct), so "the new teacher items better" can only be measured on teacher lines the model has not seen.
24 Oct (Chris: "Prepare"): the large evening round of 23-24 Oct arrived; stage18_teacher_2324.py drew a stratified
44-line held-out set from it (teacher-written / better lines only, both teachers, maths / English / science / harness;
seed 1824) BEFORE the rebuild, the build guards it, and the rest trains. Frozen with a .sha256 so an edit after the
models exist stops the scoring. The plateau plan's 40 fresh sentences (K80) are no longer needed for this gate.

Added gates 24 Oct (PACK.md 6, dated addendum): the confusable PROBES (confusable_probes_final18.jsonl: fresh English
lines whose Tonga must carry the right word and never the dangerous one: nkolo / goyooyo, masamu / masaamu, mweelwe /
nombolo, kkolona / kkolela; deterministic, no judge) and the RAINFALL categories (the 15 ZMD lines of the frozen
diagnosis, en>toi: right category words no fewer than a050's, no category reversed).

results.json (one block per seed, filled from the blind labels; counts are against a050 as served):
  {"seeds": {"1": {"neg_critical_sweep": 7, "neg_critical_regression": 0, "p1": [better, worse],
                   "docs300": [b, w], "docs164": [b, w], "teacher": [b, w], "confusable_worse": 0,
                   "probe_fail": 0, "rainfall": [right_B18m, right_a050, reversed_B18m],
                   "speakers": [b, w]}, "2": {...}},
   "a050_neg_critical_sweep": 8}
"""
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "llm")]
CT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "internal", "chitonga_train"))
RUN = os.path.join(CT, "runs", "2026-10-23_final_translator_data")
TEST = os.path.join(RUN, "teacher_test_final18.jsonl")
MIN_TEACHER = 24


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def add(tsv):
    """append teacher lines (columns en, toi, teacher, src) to the frozen-test file; refused once frozen"""
    if os.path.exists(TEST + ".sha256"):
        raise SystemExit("the teacher test is frozen: new lines go to the next round's test")
    have = {(r["en"], r["toi"]) for r in jl(TEST)}
    n = 0
    with open(TEST, "a", encoding="utf-8", newline="\n") as f:
        for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            if (r["en"], r["toi"]) in have:
                continue
            f.write(json.dumps({"id": f"f18t:{len(have) + n}", "en": r["en"], "toi": r["toi"], "teacher": r["teacher"],
                                "src": r["src"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"added {n}; {len(have) + n} in the teacher test")


def freeze():
    """refuse under MIN_TEACHER items or if any item sits in the stage 18 data; then write the .sha256"""
    import mix_groups as MG
    rows = jl(TEST)
    if len(rows) < MIN_TEACHER:
        raise SystemExit(f"{len(rows)} items < {MIN_TEACHER}: send the plateau plan's 40 fresh sentences (K80) first")
    key = lambda s: MG._n(s)
    tr = set()
    for f in ("train.jsonl", "dev.jsonl"):
        for r in jl(os.path.join(RUN, f)):
            tr.add(key(r.get("en")))
            tr.add(key(r.get("toi")))
    seen = [r["id"] for r in rows if key(r["en"]) in tr or key(r["toi"]) in tr]
    if seen:
        raise SystemExit(f"teacher test items inside the stage 18 data (not unseen): {seen[:5]}")
    h = hashlib.sha256(open(TEST, "rb").read()).hexdigest()
    open(TEST + ".sha256", "w").write(h + "\n")
    print(f"FROZEN {len(rows)} items, sha256 {h[:16]}")


PLAN = """Judging steps (PACK.md 6; systems: a050 served = control, B18m s1, B18m s2; raw B18 and A16 reported only):
 1. download the SIX merge int8 repos PRIVATE_HF_ORG/chitonga-nllb-f18-B18m{03,05,07}-s{1,2}-ct2-int8 (and
    -B18-s{1,2} for the paper) to the Beelink (beelink_lock.py acquire final18 first); run each through
    translator_service (term guard on, memory off). The alpha is a dial: gates 1-3 and 4b are read for every alpha,
    the FROZEN 44 only for the alpha that survives them (one look, on the winner).
 2. NEGATION (gate 1): negation_sweep.py gen/flags on the 150 sweep lines for each B18m; every flag read by hand;
    critical = reversed negation or dropped prohibition / obligation / warning (ruling A + C); plus the regression lines
    (cholera posters 0 1 7 8 13 14, ce2t:06, ce2t:17, before-eating line, zema 0047).
 3. P1 174 + DOCUMENTS (gates 2, 3): merge_judge.py machinery with systems a050 / B18m s1 / B18m s2 (letters seeded,
    labels carried by (item, text)); the 300 AB_PREREG documents ranked; the 164 never-judged held-out document lines
    A / B against a050.
 4. TEACHER TEST (gate 4): frozen 44 lines (teacher_test_final18.jsonl); both directions; Claude blind A / B vs a050
    as the FINDER (letters seeded, key opened after every label), the teacher's line as the reference, slash forms all
    acceptable; better : worse >= 1.5 : 1. The blind judge's labels are later compared with the speakers' (gate 5).
 4b. PROBES (gate 4b, deterministic): confusable_probes_final18.jsonl through each system; a forbidden word in the
    output = fail; expected word missing = reported. RAINFALL: the 15 ZMD lines of the frozen diagnosis, en>toi, each
    output read for its category (normal / above / below); right no fewer than a050's, no reversal.
 5. SPEAKERS (gate 5): 24 changed lines of the teacher test + documents, blind, two speakers (K96, Chris's go).
 6. decide results.json -> verdict; LOGBOOK + DECISIONS the same turn."""


MIN_DECIDED = 30       # under this many decided items the teacher gate is REPORTED, not decided (review F8)


def ratio_ok(bw, r):
    b, w = bw
    return b >= r * w and b > 0


def no_worse(bw):
    """a no-worse gate with a margin (review F9): a set fails only when 'worse' beats 'better' by more than the noise
    of the comparison, margin = max(2, 10% of the decided items). Two document sets x two seeds agreeing in sign by
    chance is ~1 in 4, so a bare 'better >= worse' on each threw away a real teacher gain most of the time."""
    b, w = bw
    return w <= b + max(2, round(0.1 * (b + w)))


def decide(res):
    """the pre-registered rule: ALL gates on BOTH seeds; seeds pointing different ways on docs or teacher = no decision"""
    a050 = res["a050_neg_critical_sweep"]
    out, fails, dirs = {}, [], {}
    for sd, g in sorted(res["seeds"].items()):
        f = []
        if g["neg_critical_sweep"] > a050:
            f.append(f"negation sweep {g['neg_critical_sweep']} > a050 {a050}")
        if g["neg_critical_regression"] > 0:
            f.append(f"negation on regression lines {g['neg_critical_regression']}")
        if not no_worse(g["p1"]):
            f.append(f"P1 {g['p1'][0]}:{g['p1'][1]} (worse beyond the margin)")
        for k in ("docs300", "docs164"):
            if not no_worse(g[k]):
                f.append(f"{k} {g[k][0]}:{g[k][1]} (worse beyond the margin)")
        if sum(g["teacher"]) < MIN_DECIDED:
            f.append(f"teacher test only {sum(g['teacher'])} decided items (< {MIN_DECIDED}): reported, not decided")
        elif not ratio_ok(g["teacher"], 1.5):
            f.append(f"teacher test {g['teacher'][0]}:{g['teacher'][1]} < 1.5:1")
        if g["confusable_worse"] > 0:
            f.append(f"confusable probes worse {g['confusable_worse']}")
        if g.get("probe_fail", 0) > 0:
            f.append(f"confusable PROBES carry the dangerous word {g['probe_fail']} times")
        if "rainfall" in g:
            rb, ra, rev = g["rainfall"]
            if rb < ra:
                f.append(f"rainfall categories right {rb} < a050 {ra}")
            if rev > 0:
                f.append(f"rainfall category reversed {rev} times")
        if "speakers" in g and not ratio_ok(g["speakers"], 1.5):
            f.append(f"speakers {g['speakers'][0]}:{g['speakers'][1]} < 1.5:1")
        out[sd] = f or ["pass"]
        fails += f
        # the seeds must agree on the CLAIM (the teacher gate), not on document noise (review F9)
        dirs[sd] = ((g["teacher"][0] > g["teacher"][1]) - (g["teacher"][0] < g["teacher"][1]),)
    if len(set(dirs.values())) > 1:
        return "NO DECISION (the seeds point different ways on the teacher gate): a050 stays", out
    if fails and all(f.startswith("teacher test only") for f in fails):
        return "REPORTED, NOT DECIDED (fewer than 30 decided teacher items): a050 stays; the gate is reported, not decided", out
    if fails:
        return "FAIL: a050 stays, the data is kept, the paper reports it", out
    if not all("speakers" in g for g in res["seeds"].values()):
        return "PASS ON [L]; speakers (gate 5) still to judge before install", out
    return ("PASS: install this alpha's B18m behind translator_service (CHITONGA_TRANSLATOR), a050 as fallback; "
            "the alpha is the one these numbers were computed for", out)


ALPHAS = ("03", "05", "07")


def pick_alpha(results_by_alpha):
    """{alpha: results.json dict} -> (alpha or None, verdicts). Gates first: an alpha is eligible only if decide() PASSES
    on it (both seeds; negation A + C, P1 / documents no worse within the margin, no forbidden probe word, rainfall,
    the frozen 44 >= 1.5 : 1 with >= 30 decided). Among eligible alphas the HIGHEST teacher-test ratio (better : worse,
    both seeds pooled) wins; a tie within 0.1 of the best goes to 0.5. None eligible -> a050 stays."""
    verdicts, elig = {}, {}
    for a_, res in results_by_alpha.items():
        v, o = decide(res)
        verdicts[a_] = v
        if v.startswith("PASS"):
            b = sum(g["teacher"][0] for g in res["seeds"].values())
            w = sum(g["teacher"][1] for g in res["seeds"].values())
            elig[a_] = b / max(1, w)
    if not elig:
        return None, verdicts
    best = max(elig.values())
    tied = [a_ for a_, r in elig.items() if best - r <= 0.1]
    return ("05" if "05" in tied else max(tied, key=lambda a_: elig[a_])), verdicts


def selftest():
    good = {"neg_critical_sweep": 7, "neg_critical_regression": 0, "p1": [20, 15], "docs300": [60, 50],
            "docs164": [30, 25], "teacher": [24, 12], "confusable_worse": 0, "probe_fail": 0, "rainfall": [3, 1, 0],
            "speakers": [10, 5]}      # 36 decided of the 88 judgements: above the MIN_DECIDED floor
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": good, "2": good}})
    assert v.startswith("PASS:"), v
    bad = dict(good, neg_critical_regression=1)
    v, o = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": good, "2": bad}})
    assert v.startswith("FAIL"), v
    flip = dict(good, teacher=[12, 24])
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": good, "2": flip}})
    assert v.startswith("NO DECISION"), v
    noisy = dict(good, docs164=[30, 31])          # F9: one document set one item down is noise, not a failure
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": noisy, "2": noisy}})
    assert v.startswith("PASS"), v
    thin2 = dict(good, teacher=[8, 4])            # F8: too few decided items = reported, not decided (a050 stays)
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": thin2, "2": thin2}})
    assert v.startswith("REPORTED, NOT DECIDED"), v
    thin = dict(good, teacher=[20, 16])      # 36 decided but only 1.25 : 1
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": thin, "2": thin}})
    assert v.startswith("FAIL"), v
    nosp = {k: v for k, v in good.items() if k != "speakers"}
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": nosp, "2": nosp}})
    assert v.startswith("PASS ON [L]"), v
    rain = dict(good, rainfall=[1, 1, 1])
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": good, "2": rain}})
    assert v.startswith("FAIL") and "reversed" in json.dumps(_), v
    probe = dict(good, probe_fail=2)
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": probe, "2": good}})
    assert v.startswith("FAIL"), v
    # the alpha pick: 0.3 fails P1, 0.5 and 0.7 pass with ratios 2.0 and 2.6 -> 0.7; within 0.1 -> 0.5; none pass -> None
    def res(teacher, p1=(20, 15)):
        g = dict(good, teacher=teacher, p1=list(p1))
        return {"a050_neg_critical_sweep": 8, "seeds": {"1": g, "2": g}}
    a_, _ = pick_alpha({"03": res([24, 12], p1=(10, 25)), "05": res([24, 12]), "07": res([26, 10])})
    assert a_ == "07", a_
    a_, _ = pick_alpha({"03": res([24, 12]), "05": res([24, 12]), "07": res([25, 12])})
    assert a_ == "05", a_
    a_, _ = pick_alpha({"03": res([24, 12], p1=(10, 25)), "05": res([24, 12], p1=(10, 25))})
    assert a_ is None, a_
    v, _ = decide({"a050_neg_critical_sweep": 8, "seeds": {"1": dict(good, teacher=[8, 4]), "2": dict(good, teacher=[8, 4])}})
    assert v.startswith("REPORTED, NOT DECIDED"), v
    print("SELFTEST OK (13 cases)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "add":
        add(sys.argv[2])
    elif cmd == "freeze":
        freeze()
    elif cmd == "plan":
        print(PLAN)
    elif cmd == "decide":
        v, o = decide(json.load(open(sys.argv[2], encoding="utf-8")))
        print(json.dumps(o, indent=1))
        print("VERDICT:", v)
    elif cmd == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
