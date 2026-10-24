"""Negation sweep through the SERVED path (22 Oct 2026, the Jesuit; Chris: "Negation sweep go", K0 on the Beelink).

Why: the a050 confirmation found one restriction reversed on a fresh document (zema livingstone 0047, "restrict
employment of non-skilled labour to local communities"), and a070 makes the same reversal through the service. One item
is not a rate. This decodes EVERY test line we hold through translator_service (term guard on, as served) for R (s13
soup), wise-a050 and wise-a070, runs the automatic negation check on each output against its source, and lists every
flag for a hand read (blind to system). Dropped prohibitions / obligations / warnings count (Chris, ruling A+C).

Sets (1,909 decodes per system, `items()`): the 174 P1 pre-screen + 48 El Nino items (merge prescreen/items.jsonl),
all 687 stage 16 held-out document items (they contain the judged 150 and the fresh 150), the 500-line diagnosis set
in both directions (en as typed > Tonga; the human Tonga > English).

How the service is run for a merge: translator_service.SOUP is swapped for the merge (the drafter: incumbent, n-best,
term guard, loop guard, names route); the other readers (seeds s13_s1 / s13_s2 in the pool, the back-translation checker
HARNESS_READER = R) stay R's, as in stage15_climate_challenge.gen. Memory off (a memory hit is the same line for every
system and could serve a book row; the question is the model).

FAST PATH (exact, validated): decide() switches away from the incumbent (the soup's beam-4 top) only when the
incumbent's decisive facts (copy / loop, notation lost, hard checks; attest and dict_missing reranking are OFF in CFG)
are non-zero: the consensus margin is 9 (off) and a candidate can only win by being strictly cleaner. So the sweep first
decodes the incumbent alone and checks it; ANY piece with a non-zero decisive tuple, or a toi>en piece with a school-
spelling reading (MBR may lead with it), sends the whole item through the full service. `validate` proves identity on a
sample (full vs fast) before the sweep is trusted.

    python -X utf8 tools/chitonga/train/negation_sweep.py validate --model s13-soup --n 40
    python -X utf8 tools/chitonga/train/negation_sweep.py gen --model wise-a050          # resumable
    python -X utf8 tools/chitonga/train/negation_sweep.py flags                           # auto check + blind read sheet
    (the reader writes read.jsonl {"iid", "L": "ok|reversal|drop|false_alarm", "note"} per letter)
    python -X utf8 tools/chitonga/train/negation_sweep.py score
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "tools", "chitonga", "llm")]
CT = os.path.join(REPO, "internal", "chitonga_train")
OUT = os.path.join(CT, "runs", "2026-10-22_merge", "negation_sweep")
MODELS = os.environ.get("CHITONGA_MODELS", r"models")   # env: a pod run (22 Oct)
SYSTEMS = ["s13-soup", "wise-a050", "wise-a070"]
FOCUS = "zema:3rd_draft_sessd_eis_livingstone_bernt_checked_sw:0047"

# prohibitions / obligations / warnings (Chris 22 Oct, A+C; merge_judge.OBL_EN plus "restrict ... to", limit, except)
OBL_EN = re.compile(r"\b(must|shall|should|have to|has to|need to|needs to|required|requires?|prohibit\w*|forbid\w*|"
                    r"warn\w*|avoid|ensure|do not|don't|never|may not|not allowed|illegal|only|restrict\w*|limit\w*|"
                    r"unless|except|danger\w*|careful|beware)\b", re.I)
OBL_TOI = re.compile(r"(eelede|elede|yandika|leelede|uta|muta|kota|mutaci|taku|kutaa|kakwiila|kucenjel|cenjel|"
                     r"zumizy|zuminzy|alikke|buyo|biyo|kuleka|kwaleka|kulesy|lesyegw|kasya|kasyigw|ntenda|bbaleka)", re.I)


# the auto CHECK uses narrower Tonga markers than the phase split above (which must stay fixed once the run started):
# OBL_TOI's ta-/uta/kota substrings match most long Tonga lines, so an obligation drop could never be flagged. Negatives
# are found by negation_test.toi_neg; these are the obligation / prohibition / warning / 'only' words themselves.
OBL_TOI_CHECK = re.compile(r"(eelede|elede|yandika|kasya|kasyigw|kasizy|zumizy|zuminzy|cenjel|alikke|buyo(?![a-z])|biyo(?![a-z])|"
                           r"lesya|lesyegw|kulesy|ntenda|bbaleka|mulawo|tazumizyigwi)", re.I)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def items():
    rows = []
    for x in jl(os.path.join(CT, "runs", "2026-10-22_merge", "prescreen", "items.jsonl")):
        rows.append({"iid": x["iid"], "set": "elnino" if x.get("set") != "p1" else "p1", "dir": x["dir"],
                     "src": x["src"], "ref": x.get("ref")})
    j150 = {x["iid"] for x in jl(os.path.join(CT, "runs", "2026-10-21_stage16", "doctest", "test_doc_sample150.jsonl"))}
    f150 = {x["iid"] for x in jl(os.path.join(CT, "runs", "2026-10-22_merge", "confirm_doc_fresh150.jsonl"))}
    for x in jl(os.path.join(CT, "runs", "2026-10-21_stage16", "doctest", "test_doc_items.jsonl")):
        rows.append({"iid": x["iid"], "set": "docs", "sub": "judged150" if x["iid"] in j150 else "fresh150"
                     if x["iid"] in f150 else "docs_rest", "dir": x["dir"], "src": x["src"], "ref": x.get("ref")})
    for x in jl(os.path.join(CT, "llm", "panel", "diagnosis_500_2026-10-21", "set.jsonl")):
        rows.append({"iid": x["id"] + ":en>toi", "set": "diag500", "dir": "en>toi", "src": x["en"], "ref": x["ref_toi"],
                     "input_kind": x["input_kind"]})
        rows.append({"iid": x["id"] + ":toi>en", "set": "diag500", "dir": "toi>en", "src": x["ref_toi"], "ref": x["en_clean"]})
    assert len({r["iid"] for r in rows}) == len(rows) == 222 + 687 + 1000, len(rows)
    return rows


def src_hits(r):
    """what the SOURCE carries: neg (a negation) and obl (a prohibition / obligation / warning)"""
    import negation_test as N
    s = r["src"]
    if r["dir"] == "en>toi":
        return bool(N.EN_NEG.search(s)), bool(OBL_EN.search(s))
    return bool(N.toi_neg(s)), bool(OBL_TOI.search(s))


def priority(rows):
    """the 0047 focus item first, then sources with a negation / obligation (the drop candidates), then the rest"""
    def k(r):
        neg, obl = src_hits(r)
        return (0 if FOCUS in r["iid"] else 1 if (neg or obl) else 2)
    return sorted(rows, key=k)


# ------------------------------------------------------------------------------------------------ the service
def make_service(model):
    import translator_service as TS
    TS.SOUP = os.path.join(MODELS, "chitonga-nllb-" + model)
    full = TS.Service({"memory": False})
    fast = _fast_class()({"memory": False})
    fast._r = full._r                                 # one set of loaded readers
    return TS, full, fast


def _fast_class():
    import translator_service as TS

    class Fast(TS.Service):
        """PIECE-LEVEL fast path (v2, 22 Oct: v1 sent a whole paragraph to the full service when one piece was flagged;
        documents cost ~100 s each). candidates(): the incumbent (the soup's beam-4 top over the WHOLE batch of pieces,
        exactly as Service.candidates builds cands[0]) is checked alone; a piece whose decisive facts are non-zero, or a
        toi>en piece with a school-spelling reading, gets the full candidate pool (Service.candidates on those pieces);
        clean pieces keep the incumbent alone, which decide() would keep anyway. Batch composition can move an int8
        decode, so the full pool's own cands[0] is compared with the whole-batch incumbent and a mismatch is logged
        (self.mismatch) and the whole-batch incumbent is kept first, as the service would have it."""
        def candidates(self, prep, direction):
            rel = self.nbest("soup", [q["text"] for q in prep], direction, 1, beam=4)
            one = [[{"src": "soup:b4", "text": rel[i][0][0], "score": rel[i][0][1], "also": []}] for i in range(len(prep))]
            self.check(prep, one, direction)
            bad = [i for i, q in enumerate(prep) if any(TS.decisive(one[i][0], self.cfg))
                   or (direction == "toi>en" and q.get("norm"))]
            self.n_full = getattr(self, "n_full", 0) + len(bad)
            self.n_piece = getattr(self, "n_piece", 0) + len(prep)
            if bad:
                more = super().candidates([prep[i] for i in bad], direction)
                for i, cs in zip(bad, more):
                    if cs[0]["text"].strip() != one[i][0]["text"].strip():
                        self.mismatch = getattr(self, "mismatch", []) + [(prep[i]["text"], cs[0]["text"], one[i][0]["text"])]
                        cs = [c for c in cs if c["text"].strip() != one[i][0]["text"].strip()]
                        cs.insert(0, dict(one[i][0], also=["subset:" + cs[0]["src"]] if cs else []))
                    one[i] = cs
            return one
    return Fast


def translate(TS, full, fast, text, direction):
    """(text, path, reason, alarms): the piece-level fast path (exact by construction except the logged batch
    mismatch; validated against the full service)"""
    n0, m0 = getattr(fast, "n_full", 0), len(getattr(fast, "mismatch", []))
    o = fast.translate(text, direction)
    nf, mm = getattr(fast, "n_full", 0) - n0, len(getattr(fast, "mismatch", [])) - m0
    return o["text"], ("fast" if nf == 0 else f"pieces_full:{nf}") + (f" MISMATCH:{mm}" if mm else ""), None, o.get("alarms", [])


def validate(a):
    """full vs fast on a seeded sample of every set and both directions: the outputs must be identical"""
    TS, full, fast = make_service(a.model)
    rows = items()
    rnd = random.Random(2210)
    sample = [r for r in rows if FOCUS in r["iid"]]
    by = {}
    for r in rows:
        by.setdefault((r["set"], r["dir"]), []).append(r)
    per = max(1, a.n // len(by))
    for k in sorted(by):
        sample += rnd.sample(by[k], min(per, len(by[k])))
    res, same = [], 0
    for r in sample:
        t0 = time.time()
        f = full.translate(r["src"], r["dir"])["text"]
        t1 = time.time()
        g, path, why, _ = translate(TS, full, fast, r["src"], r["dir"])
        t2 = time.time()
        same += f == g
        res.append({"iid": r["iid"], "dir": r["dir"], "full": f, "fast": g, "path": path, "why": why, "same": f == g,
                    "t_full": round(t1 - t0, 1), "t_fast": round(t2 - t1, 1)})
        print(r["iid"][-40:], path, "SAME" if f == g else "DIFF", f"{t1 - t0:.0f}s / {t2 - t1:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"model": a.model, "n": len(res), "identical": same, "rows": res},
              open(os.path.join(OUT, f"validate_{a.model}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"identical {same}/{len(res)}; time full {sum(x['t_full'] for x in res):.0f}s fast {sum(x['t_fast'] for x in res):.0f}s")


def gen(a):
    TS, full, fast = make_service(a.model)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"gen_{a.model}.jsonl")
    done = {x["iid"] for x in jl(p)}
    rows = [r for r in priority(items()) if r["iid"] not in done]
    if a.only:                                       # phase 1 'hit' (source negation / obligation), phase 2 'rest'
        rows = [r for r in rows if any(src_hits(r)) == (a.only == "hit")]
    if a.limit:
        rows = rows[:a.limit]
    t0 = time.time()
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            t = time.time()
            out, path, why, alarms = translate(TS, full, fast, r["src"], r["dir"])
            f.write(json.dumps({**r, "model": a.model, "out": out, "path": path, "why": why,
                                "alarms": [x for x in alarms if "negation" in x or "term_guard" in x],
                                "t": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(a.model, len(done) + i + 1, "/", len(done) + len(rows), f"{time.time() - t0:.0f}s", path, flush=True)
    print("done", a.model, f"{time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ the check
def out_hits(r, out):
    import negation_test as N
    if r["dir"] == "en>toi":
        return bool(N.toi_neg(out)), bool(OBL_TOI_CHECK.search(out))
    return bool(N.EN_NEG.search(out)), bool(OBL_EN.search(out))


def src_hits_check(r):
    import negation_test as N
    s = r["src"]
    if r["dir"] == "en>toi":
        return bool(N.EN_NEG.search(s)), bool(OBL_EN.search(s))
    return bool(N.toi_neg(s)), bool(OBL_TOI_CHECK.search(s))


def auto_flags(r, out):
    """the automatic check. neg_dropped: the source has a negation, the output none. obl_dropped: the source carries a
    prohibition / obligation / warning and the output neither a negation nor an obligation marker. neg_added: the
    output has a negation the source lacks (a possible reversal)."""
    sn, so = src_hits_check(r)
    on, oo = out_hits(r, out)
    fl = []
    if sn and not on:
        fl.append("neg_dropped")
    if so and not (on or oo):
        fl.append("obl_dropped")
    if on and not sn:
        fl.append("neg_added")
    return fl


def flags(a):
    gens = {m: {x["iid"]: x for x in jl(os.path.join(OUT, f"gen_{m}.jsonl"))} for m in SYSTEMS}
    common = sorted(set.intersection(*(set(g) for g in gens.values())))
    assert len(common) == 1909 or a.partial, f"only {len(common)} of 1909 items decoded by all three"
    rnd = random.Random(22104)
    key, lines, auto = {}, [], {}
    for iid in common:
        r = gens[SYSTEMS[0]][iid]
        fl = {m: auto_flags(r, gens[m][iid]["out"]) for m in SYSTEMS}
        auto[iid] = fl
        differ = len({gens[m][iid]["out"] for m in SYSTEMS}) > 1
        if not any(fl.values()) and not (any(src_hits_check(r)) and differ):
            continue
        texts = sorted({gens[m][iid]["out"] for m in SYSTEMS})   # every distinct text, so one read serves all three
        rnd.shuffle(texts)
        key[iid] = {"ABC"[i]: [m for m in SYSTEMS if gens[m][iid]["out"] == t] for i, t in enumerate(texts)}
        lines.append(f"## {iid} [{r['dir']}] flags: {sorted({f for m in SYSTEMS for f in fl[m]})}\n   src: {r['src']}\n"
                     f"   ref: {r.get('ref')}")
        lines += [f"   {'ABC'[i]}: {t}" for i, t in enumerate(texts)]
    open(os.path.join(OUT, "read_sheet.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(key, open(os.path.join(OUT, "READ_KEY_do_not_open.json"), "w", encoding="utf-8"), indent=0)
    json.dump(auto, open(os.path.join(OUT, "auto_flags.json"), "w", encoding="utf-8"), indent=0)
    print(len(common), "items in all three;", len(key), "with a flag;", sum(len(v) for v in key.values()), "texts to read")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def score(a):
    """opens the key AFTER every flagged item is read. Per system over the common items: real reversals and drops
    (the error count), unclear (a possible reversal the reader could not settle; reported, counted only in the strict
    rate), false alarms (auto-flagged, read fine), misses (a real error the auto check did not flag). Items not on the
    sheet (no flag, the three texts identical) count as fine for all three. Paired: exact McNemar on the discordant
    items, merge vs R."""
    key = json.load(open(os.path.join(OUT, "READ_KEY_do_not_open.json"), encoding="utf-8"))
    auto = json.load(open(os.path.join(OUT, "auto_flags.json"), encoding="utf-8"))
    read = {x["iid"]: x for x in jl(os.path.join(OUT, "read.jsonl"))}
    missing = [i for i in key if i not in read or any(L not in read[i] for L in key[i])]
    assert not missing, f"{len(missing)} flagged items unread, e.g. {missing[:3]}"
    n = len(auto)
    verdict = {m: {i: "ok" for i in auto} for m in SYSTEMS}
    for iid, lk in key.items():
        for L, ms in lk.items():
            for m in ms:
                verdict[m][iid] = read[iid][L]
    res = {"n_items": n, "systems": {}, "paired": {}, "real": {}, "unclear": {}, "notes": {}}
    for m in SYSTEMS:
        v = verdict[m]
        err = sorted(i for i, x in v.items() if x in ("reversal", "drop"))
        unc = sorted(i for i, x in v.items() if x == "unclear")
        res["systems"][m] = {
            "reversal": sum(1 for x in v.values() if x == "reversal"), "drop": sum(1 for x in v.values() if x == "drop"),
            "unclear": len(unc), "errors": len(err), "rate": round(len(err) / n, 4), "wilson95": wilson(len(err), n),
            "strict_rate_incl_unclear": round((len(err) + len(unc)) / n, 4), "wilson95_strict": wilson(len(err) + len(unc), n),
            "auto_flags": sum(1 for i in auto if auto[i][m]),
            "false_alarms": sum(1 for i in auto if auto[i][m] and v[i] == "ok"),
            "missed_by_auto": sum(1 for i in err if not auto[i][m])}
        res["real"][m] = {i: v[i] for i in err}
        res["unclear"][m] = unc
    R_ = "s13-soup"
    rset = set(res["real"][R_])
    for m in SYSTEMS[1:]:
        ms = set(res["real"][m])
        b, c = len(ms - rset), len(rset - ms)
        k, t = min(b, c), b + c
        p = min(1.0, 2 * sum(math.comb(t, i) for i in range(k + 1)) / 2 ** t) if t else 1.0
        res["paired"][m] = {"merge_errs_R_not": sorted(ms - rset), "R_errs_merge_not": sorted(rset - ms),
                            "both": sorted(ms & rset), "b": b, "c": c, "mcnemar_exact_p": round(p, 4)}
    res["notes"] = {i: read[i].get("note") for m in SYSTEMS for i in list(res["real"][m]) + res["unclear"][m]}
    json.dump(res, open(os.path.join(OUT, "result.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(res["systems"], indent=1))
    print(json.dumps({m: {k: v for k, v in res["paired"][m].items()} for m in SYSTEMS[1:]}, indent=1))
    print(json.dumps(res["unclear"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("validate", "gen", "flags", "score"))
    ap.add_argument("--model", default="s13-soup")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--only", choices=("hit", "rest"), default=None)
    a = ap.parse_args()
    {"validate": validate, "gen": gen, "flags": flags, "score": score}[a.cmd](a)
