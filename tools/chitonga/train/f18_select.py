"""The PRE-REGISTERED rule that picks the stage 18 mix from the sweep (the Jesuit, 24 Oct 2026, written before any
sweep output exists; DECISIONS 24 Oct MIX SWEEP; the architect's STRATEGY_TEACHER_DATA_2026-10-24.md A5: "the sweep is
a reject, not a rank ... ties go to the LOWER teacher weight"). It runs on the GPU machine between the sweep and the
two full runs, so nobody (me included) can choose the dose after reading the outputs.

Revised 24 Oct 03:30 after the delta review (STAGE18_DELTA_REVIEW_2026-10-24.md, section 3, flaw A): the memorisation
and probe gates were ABSOLUTE, and a050 fails them by construction (81 of the 136 t_teach rows are a050's own output
confirmed, so a050 reproduces ~half of a t_teach sample verbatim; the probes were built from a050's own confusions).
Every dose would have been ineligible and the sweep a no-op. The gates are now PAIRED against a050 on the same
material: f18_sel.py scores a050 first on the same seeded sample and the same probes.

Candidates: S18a (half dose), S18b (the planned dose), S18c (double), each trained SWEEP_STEPS updates from a050 with
seed 1 and scored by f18_sel.py.

  1. ELIGIBLE, all three must hold, each relative to a050's own score on the same slice:
     (a) FORGETTING: dev_both <= a050's dev_both + 0.02 (the stage 16 rule; absolute margin on a token-mean NLL).
     (b) MEMORISATION: verbatim reproduction of the sampled t_teach rows (written / better rows only, f18_sel.py) and of
         the sampled synth24 rows no more than a050's share + 25 percentage points, per group.
     (c) PROBES: no forbidden word on any probe where a050 itself produced none (a confusion a050 already makes is not
         held against the dose; a NEW one is).
  2. WINNER: among the eligible, the LOWEST teacher-dev loss (teach_both).
     TIE: every eligible dose whose teach_both is within 0.01 of the BEST eligible is tied with it, and the LOWEST dose
     among the tied wins (compared with the best, not chained). A dose earns its weight only by a margin.
  3. No candidate eligible: S18a, marked ALARM (the smallest dose that was built; the kill test downstream decides).
  4. ALARMS (decide nothing, printed for the record): the winner's teach_both not below a050's; its SEL score above
     a050's; its verbatim share over a050's + 15 points in any group; a050 itself failing a probe.

    python f18_select.py <dir with f18_a050.json, f18_S18a.json ...> S18a,S18b,S18c     prints 'WINNER <name>' last
    python f18_select.py selftest
"""
import json
import os
import sys

MEM_PP, TIE, FORGET, ALARM_PP = 0.25, 0.01, 0.02, 0.15
ORDER = ["S18a", "S18b", "S18c"]


def verb(x, g):
    v = (x.get("verbatim") or {}).get(g)
    return (v["exact"] / max(1, v["n"])) if v else 0.0


def probe_new(x, base):
    """probes where the dose produced a forbidden word and a050 produced none on the same probe"""
    b = {p["id"]: bool(p.get("forbidden")) for p in base.get("probes", [])}
    return [p["id"] for p in x.get("probes", []) if p.get("forbidden") and not b.get(p["id"], False)]


def decide(base, c):
    """(winner, why per dose, alarms, rule) from the a050 scores and {dose: scores}"""
    elig, why = [], {}
    for n, x in c.items():
        bad = []
        if x["dev_both"] > base["dev_both"] + FORGET:
            bad.append(f"forgetting: dev {x['dev_both']} > a050 {base['dev_both']} + {FORGET}")
        for g in ("t_teach", "synth24"):
            if verb(x, g) > verb(base, g) + MEM_PP:
                bad.append(f"memorisation: verbatim {g} {verb(x, g):.0%} > a050 {verb(base, g):.0%} + {MEM_PP:.0%}")
        new = probe_new(x, base)
        if new:
            bad.append(f"probes: forbidden word a050 does not make on {new}")
        why[n] = bad or ["eligible"]
        if not bad:
            elig.append(n)
    alarm = []
    if elig:
        best = min(elig, key=lambda n: c[n].get("teach_both", 9e9))
        tied = [n for n in elig if c[n].get("teach_both", 9e9) - c[best].get("teach_both", 9e9) <= TIE]
        w = min(tied, key=lambda n: ORDER.index(n) if n in ORDER else 9)
        rule = f"lowest teacher-dev loss among eligible {sorted(elig)} = {best}; tied within {TIE}: {sorted(tied)}; " \
               f"the lowest dose among the tied wins"
    else:
        w = "S18a" if "S18a" in c else sorted(c)[0]
        rule = "no candidate eligible: the smallest dose, marked ALARM"
        alarm.append("ALARM no eligible candidate")
    x = c[w]
    if x.get("teach_both") is not None and base.get("teach_both") is not None and x["teach_both"] >= base["teach_both"]:
        alarm.append("ALARM the winner's teacher-dev loss is not below a050's")
    if x["sel_score"] > base["sel_score"]:
        alarm.append("ALARM the winner's SEL score is above a050's (the held-out documents got worse)")
    for g in ("t_teach", "synth24"):
        if verb(x, g) > verb(base, g) + ALARM_PP:
            alarm.append(f"ALARM verbatim {g} {verb(x, g):.0%} is over a050's {verb(base, g):.0%} + {ALARM_PP:.0%}")
    if any(p.get("forbidden") for p in base.get("probes", [])):
        alarm.append(f"ALARM a050 itself produces a forbidden word on {[p['id'] for p in base['probes'] if p.get('forbidden')]}")
    return w, why, alarm, rule


def main(d, names):
    base = json.load(open(os.path.join(d, "f18_a050.json"), encoding="utf-8"))
    c = {}
    for n in names:
        p = os.path.join(d, f"f18_{n}.json")
        if os.path.exists(p):
            c[n] = json.load(open(p, encoding="utf-8"))
        else:
            print(f"MISSING {p}: {n} not a candidate")
    if not c:
        print("NO CANDIDATES")
        return 1
    print(f"a050: dev {base['dev_both']} teach {base.get('teach_both')} sel {base['sel_score']} verbatim t_teach "
          f"{verb(base, 't_teach'):.0%} synth24 {verb(base, 'synth24'):.0%} probe_fail {base.get('probe_fail')}")
    w, why, alarm, rule = decide(base, c)
    for n, x in c.items():
        print(f"{n}: dev {x['dev_both']} (a050 + {round(x['dev_both'] - base['dev_both'], 4)}) teach "
              f"{x.get('teach_both')} sel {x['sel_score']} verbatim t_teach {verb(x, 't_teach'):.0%} synth24 "
              f"{verb(x, 'synth24'):.0%} new probe fails {probe_new(x, base)} -> {'; '.join(why[n])}")
    for l in alarm:
        print(l)
    print(f"RULE: {rule}")
    print(f"WINNER {w}")
    return 0


def _scores(dev, teach, vt, vs, probes_bad=(), sel=1.0):
    return {"dev_both": dev, "teach_both": teach, "sel_score": sel,
            "verbatim": {"t_teach": {"n": 40, "exact": round(vt * 40)}, "synth24": {"n": 40, "exact": round(vs * 40)}},
            "probes": [{"id": f"cp:{i:02d}", "forbidden": (["x"] if f"cp:{i:02d}" in probes_bad else [])} for i in range(1, 14)]}


def selftest():
    # the delta review's case: a050 reproduces 45% of t_teach and fails one probe; every dose a little better on the
    # teacher dev and within the forgetting margin -> with ABSOLUTE gates all were ineligible; PAIRED they are eligible
    base = _scores(1.000, 0.600, 0.45, 0.05, probes_bad=("cp:04",))
    c = {"S18a": _scores(1.005, 0.590, 0.50, 0.10, probes_bad=("cp:04",)),
         "S18b": _scores(1.010, 0.575, 0.55, 0.15, probes_bad=("cp:04",)),
         "S18c": _scores(1.015, 0.560, 0.60, 0.25, probes_bad=("cp:04",))}
    w, why, alarm, rule = decide(base, c)
    assert all(v == ["eligible"] for v in why.values()), why
    assert w == "S18c", (w, rule)                       # lowest teacher-dev loss, no tie (0.015 and 0.030 apart)
    # a truly forgetting dose is rejected; a memorising dose is rejected; a NEW probe confusion is rejected
    c2 = dict(c, S18c=_scores(1.030, 0.500, 0.60, 0.25, probes_bad=("cp:04",)))
    w, why, _, _ = decide(base, c2)
    assert why["S18c"][0].startswith("forgetting") and w == "S18b", (why, w)
    c3 = dict(c, S18c=_scores(1.015, 0.500, 0.75, 0.25, probes_bad=("cp:04",)))
    w, why, _, _ = decide(base, c3)
    assert why["S18c"][0].startswith("memorisation") and w == "S18b", (why, w)
    c4 = dict(c, S18c=_scores(1.015, 0.500, 0.60, 0.25, probes_bad=("cp:04", "cp:09")))
    w, why, _, _ = decide(base, c4)
    assert why["S18c"][0].startswith("probes") and "cp:09" in why["S18c"][0] and w == "S18b", (why, w)
    # the tie: b and c within 0.01 of the best (c) -> the lower dose b wins; a is 0.017 behind c and NOT tied (no chain)
    c5 = {"S18a": _scores(1.005, 0.520, 0.50, 0.10, probes_bad=("cp:04",)),
          "S18b": _scores(1.010, 0.512, 0.55, 0.15, probes_bad=("cp:04",)),
          "S18c": _scores(1.015, 0.503, 0.60, 0.25, probes_bad=("cp:04",))}
    w, why, _, rule = decide(base, c5)
    assert w == "S18b", (w, rule)
    # nothing eligible -> S18a with the alarm
    c6 = {n: _scores(1.050, 0.5, 0.5, 0.1, probes_bad=("cp:04",)) for n in ORDER}
    w, why, alarm, _ = decide(base, c6)
    assert w == "S18a" and any("no eligible" in a for a in alarm), (w, alarm)
    print("SELFTEST OK (6 cases: paired gates, forgetting, memorisation, new probe, tie to the lower dose, fallback)")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    else:
        sys.exit(main(sys.argv[1], sys.argv[2].split(",")))
