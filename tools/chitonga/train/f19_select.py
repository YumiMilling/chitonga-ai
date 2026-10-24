"""The PRE-REGISTERED rule that picks the stage 19 <cls> dose from the sweep (the Jesuit, 24 Oct 2026, written before any
stage 19 output exists). It runs on the GPU machine between the sweep and the full runs, so nobody can choose the dose
after reading outputs. Built on f18_select.py (gates PAIRED against a050, ties to the lower dose); what is new:

  - the classroom loss is read WITH the tag (teach_both_cls): that is the path the service will use for lessons;
  - forgetting is read untagged AND with <gen> (the old readers get untagged input; the new drafter will pass <gen>);
  - memorisation of the synthetic group is the GENERALISATION GAP (trained rows minus the held-out synthetic slice, both
    with <cls>), not an absolute verbatim share: a deterministic number frame SHOULD be reproduced if the pattern was
    learned (stage 18's S18c failed the absolute gate at 42% without the gate being able to say which it was);
  - probes are read untagged, with <cls> and with <gen>.

Candidates: S19a (<cls> 20% of draws), S19b (25%), S19c (30%), SWEEP_STEPS updates each from a050, seed 1.
  1. ELIGIBLE (all):
     (a) FORGETTING: dev_both <= a050's dev_both + 0.02 AND dev_both_gen <= a050's dev_both + 0.02
     (b) MEMORISATION, teacher rows: verbatim t_teach (written / better, decoded with <cls>) <= a050's untagged share + 25 pp
     (c) MEMORISATION, synthetic: syn_gap <= 0.25
     (d) PROBES: no forbidden word, untagged / <cls> / <gen>, on any probe where a050 (untagged) produced none
     (e) the target is never tagged: tag_in_output == 0
  2. WINNER: the lowest teach_both_cls among the eligible; TIE = within 0.01 of the best; the lowest dose among the tied.
  3. Nothing eligible: S19a, marked ALARM.
  4. ALARMS (decide nothing): the winner's teach_both_cls not below a050's teach_both; the TAG NOT WORKING (the winner's
     teach_both_cls not below its own untagged teach_both); its sel above a050's.

    python f19_select.py <dir with f19_a050.json, f19_S19a.json ...> S19a,S19b,S19c       prints 'WINNER <name>' last
    python f19_select.py selftest
"""
import json
import os
import sys

MEM_PP, GAP, TIE, FORGET = 0.25, 0.25, 0.01, 0.02
ORDER = ["S19a", "S19b", "S19c"]


def share(x, key):
    v = (x.get("verbatim") or {}).get(key)
    return v["exact"] / max(1, v["n"]) if v else 0.0


def probe_new(x, base):
    """a forbidden word the dose makes where a050 makes none UNDER THE SAME INPUT MODE (none / <cls> / <gen>): paired, as
    stage 18's delta-review fix (a confusion a050 already makes is not held against a dose)"""
    out = []
    for tn, ps in (x.get("probes") or {}).items():
        b = {p["id"]: bool(p.get("forbidden")) for p in (base.get("probes") or {}).get(tn, [])}
        out += [f"{p['id']}({tn})" for p in ps if p.get("forbidden") and not b.get(p["id"], False)]
    return out


def ref_dev_gen(base):
    """the <gen> forgetting reference: a050 under the same tagged input or untagged, whichever is higher (a050 never saw a
    tag, so its tagged reading may be worse; a dose IDENTICAL to a050 must be eligible - the stage 18 delta-review rule)"""
    return max(base["dev_both"], base.get("dev_both_gen", base["dev_both"]))


def decide(base, c):
    elig, why = [], {}
    for n, x in c.items():
        bad = []
        if x.get("dev_both", 9e9) > base["dev_both"] + FORGET:
            bad.append(f"forgetting: dev_both {x.get('dev_both')} > a050 {base['dev_both']} + {FORGET}")
        if x.get("dev_both_gen", 9e9) > ref_dev_gen(base) + FORGET:
            bad.append(f"forgetting: dev_both_gen {x.get('dev_both_gen')} > a050 {ref_dev_gen(base)} + {FORGET}")
        mref = max(share(base, "t_teach_none"), share(base, "t_teach_cls"))
        if share(x, "t_teach_cls") > mref + MEM_PP:
            bad.append(f"memorisation: t_teach {share(x, 't_teach_cls'):.0%} > a050 {mref:.0%} + 25 pp")
        if x.get("syn_gap", 0) > GAP:
            bad.append(f"memorisation: synthetic gap {x['syn_gap']:.0%} > {GAP:.0%}")
        new = probe_new(x, base)
        if new:
            bad.append(f"probes: forbidden word a050 does not make on {new}")
        if x.get("tag_in_output", 0) > base.get("tag_in_output", 0):     # paired: a050 may echo a tag it never saw
            bad.append(f"tag in output x{x['tag_in_output']} (a050 x{base.get('tag_in_output', 0)})")
        why[n] = bad or ["eligible"]
        if not bad:
            elig.append(n)
    alarm = []
    if elig:
        best = min(elig, key=lambda n: c[n]["teach_both_cls"])
        tied = [n for n in elig if c[n]["teach_both_cls"] - c[best]["teach_both_cls"] <= TIE]
        w = min(tied, key=lambda n: ORDER.index(n) if n in ORDER else 9)
        rule = f"lowest <cls> teacher-dev loss among eligible {sorted(elig)} = {best}; tied within {TIE}: {sorted(tied)}; lowest dose wins"
    else:
        w = "S19a" if "S19a" in c else sorted(c)[0]
        rule = "no candidate eligible: the smallest dose, marked ALARM"
        alarm.append("ALARM no eligible candidate")
    x = c[w]
    if x["teach_both_cls"] >= base["teach_both"]:
        alarm.append("ALARM the winner's <cls> teacher-dev loss is not below a050's")
    if x["teach_both_cls"] >= x["teach_both"]:
        alarm.append("ALARM TAG NOT WORKING: the winner's <cls> teacher-dev loss is not below its own untagged loss")
    if x.get("sel_score", 0) > base.get("sel_score", 9e9):
        alarm.append("ALARM the winner's SEL score is above a050's")
    return w, why, alarm, rule


def main(d, names):
    base = json.load(open(os.path.join(d, "f19_a050.json"), encoding="utf-8"))
    c = {n: json.load(open(os.path.join(d, f"f19_{n}.json"), encoding="utf-8")) for n in names
         if os.path.exists(os.path.join(d, f"f19_{n}.json"))}
    if not c:
        print("NO CANDIDATES")
        return 1
    print(f"a050: dev {base['dev_both']} teach {base['teach_both']} sel {base.get('sel_score')} verbatim t_teach "
          f"{share(base, 't_teach_none'):.0%} probe_fail {base.get('probe_fail')}")
    w, why, alarm, rule = decide(base, c)
    for n, x in c.items():
        print(f"{n}: dev {x['dev_both']} / <gen> {x.get('dev_both_gen')} teach none {x['teach_both']} <cls> "
              f"{x['teach_both_cls']} <gen> {x.get('teach_both_gen')} syn_gap {x.get('syn_gap')} -> {'; '.join(why[n])}")
    for l in alarm:
        print(l)
    print(f"RULE: {rule}")
    print(f"WINNER {w}")
    return 0


def _s(dev, devg, teach, tcls, vt=0.1, gap=0.1, bad=(), tio=0):
    return {"dev_both": dev, "dev_both_gen": devg, "teach_both": teach, "teach_both_cls": tcls, "sel_score": 1.0,
            "syn_gap": gap, "tag_in_output": tio,
            "verbatim": {"t_teach_cls": {"n": 40, "exact": round(vt * 40)}, "t_teach_none": {"n": 40, "exact": round(vt * 40)}},
            "probes": {tn: [{"id": f"cp:{i:02d}", "forbidden": ["x"] if f"cp:{i:02d}" in bad else []} for i in range(1, 14)]
                       for tn in ("none", "cls", "gen")}}


def selftest():
    base = _s(0.912, 0.912, 1.85, 1.85, vt=0.0, bad=("cp:01",))
    c = {"S19a": _s(0.915, 0.916, 1.80, 1.70, bad=("cp:01",)), "S19b": _s(0.918, 0.919, 1.79, 1.66, bad=("cp:01",)),
         "S19c": _s(0.925, 0.926, 1.78, 1.62, bad=("cp:01",))}
    w, why, alarm, _ = decide(base, c)
    assert all(v == ["eligible"] for v in why.values()) and w == "S19c", (why, w)
    c2 = dict(c, S19c=_s(0.925, 0.940, 1.78, 1.55, bad=("cp:01",)))            # forgets on the <gen> path only
    w, why, _, _ = decide(base, c2)
    assert why["S19c"][0].startswith("forgetting: dev_both_gen") and w == "S19b", (why, w)
    c3 = dict(c, S19c=_s(0.925, 0.926, 1.78, 1.55, gap=0.40, bad=("cp:01",)))  # memorised the synthetic rows
    w, why, _, _ = decide(base, c3)
    assert "synthetic gap" in why["S19c"][0] and w == "S19b", (why, w)
    c4 = dict(c, S19c=_s(0.925, 0.926, 1.78, 1.55, bad=("cp:01", "cp:05")))   # a new confusion
    w, why, _, _ = decide(base, c4)
    assert why["S19c"][0].startswith("probes") and w == "S19b", (why, w)
    c5 = {"S19a": _s(0.915, 0.916, 1.80, 1.659, bad=("cp:01",)), "S19b": _s(0.918, 0.919, 1.79, 1.657, bad=("cp:01",)),
          "S19c": _s(0.925, 0.926, 1.78, 1.650, bad=("cp:01",))}               # all within 0.01 of the best -> S19a
    w, _, _, _ = decide(base, c5)
    assert w == "S19a", w
    c6 = dict(c, S19c=_s(0.925, 0.926, 1.78, 1.55, tio=2, bad=("cp:01",)))     # a tag leaked into an output
    w, why, _, _ = decide(base, c6)
    assert "tag in output" in why["S19c"][0] and w == "S19b", (why, w)
    c7 = {"S19a": _s(0.915, 0.916, 1.70, 1.72, bad=("cp:01",))}                # the tag hurts: alarm, not a gate
    w, _, alarm, _ = decide(base, c7)
    assert any("TAG NOT WORKING" in a_ for a_ in alarm), alarm
    # IDENTITY (the stage 18 delta-review rule): a dose whose scores equal a050's must be eligible, even where a050's own
    # tagged readings are worse than its untagged ones (it never saw a tag) and it echoes a tag once
    a = _s(0.912, 0.990, 1.85, 1.95, vt=0.05, bad=("cp:01",), tio=1)
    a["probes"]["cls"][3]["forbidden"] = ["x"]                                 # a050 makes cp:04 only under <cls>
    w, why, alarm, rule = decide(a, {"S19a": json.loads(json.dumps(a))})
    assert why["S19a"] == ["eligible"] and w == "S19a", (why, rule)
    print("SELFTEST OK (8 cases: all eligible, <gen> forgetting, synthetic gap, new probe, tie to the lower dose, "
          "tag leak, tag-not-working alarm, a dose identical to a050 is eligible)")


def real_a050():
    """the REAL a050 numbers from the stage 18 machine (logs/final18_20261024_052012/run_final18.log, F18SEL init_model):
    dev 0.91212, teacher dev 1.85078, SEL 2.66273, one probe fail (cp:01). The tagged fields are not measured yet (f19_sel
    has never run on the real a050): set here to a050's untagged values and to a worse tagged reading (+0.08), the
    pessimistic case. Candidates = a050 itself, and stage 18's measured S18a / S18b moved onto the stage 19 names with the
    tag giving nothing (teach_cls = teach) - the selector must still declare a winner, and the identity must be eligible."""
    base = _s(0.91212, 0.99212, 1.85078, 1.93078, vt=0.0, bad=("cp:01",))
    base["sel_score"] = 2.66273
    c = {"S19a": json.loads(json.dumps(base)),
         "S19b": _s(0.90973, 0.91100, 1.77892, 1.77892, vt=0.0, bad=("cp:01",)),
         "S19c": _s(0.90825, 0.90900, 1.75746, 1.75746, vt=0.02, bad=("cp:01",))}
    for n in ("S19b", "S19c"):
        c[n]["sel_score"] = 2.52
    w, why, alarm, rule = decide(base, c)
    for n in c:
        print(f"  {n}: {'; '.join(why[n])}")
    for l in alarm:
        print("  " + l)
    print(f"  RULE: {rule}")
    print(f"  WINNER {w}")
    assert why["S19a"] == ["eligible"], why
    return w


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "real_a050":
        print("REAL a050 CHECK:", real_a050())
    else:
        sys.exit(main(sys.argv[1], sys.argv[2].split(",")))
