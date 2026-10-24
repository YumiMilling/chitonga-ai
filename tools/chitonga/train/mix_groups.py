"""The translator's training groups and the rules every builder's --mix must obey (21 Oct 2026, audit decisions steps 2
and 3; the Jesuit). Call `check_build(run_dir, mix)` at the end of every build_stage*.py, BEFORE anything goes to a pod.

WHY (step 2, the forgotten Hopgood pairs). train_nllb.py draws only the groups NAMED in --mix ("groups not named are
not used"). The 1,302 Hopgood 1992 pairs (origin hopgood:*, group 'hopgood') have sat in every train.jsonl since stage 5
but were named only in the stage 5 and 6 experiment arms (H1-H3, X3; runs/2026-10-11_stage5/RESULTS.md). Those arms
were not kept: on the translators' test (single-reference chrF, since distrusted) H1 scored 33.2 vs 34.2 without, inside
the seed spread, while the Hopgood test rose +7 to +9. Stage 7's baseline B0 re-used the kept model's recipe ("mix of
N"), which never had the group; every later mix (stages 8-14) descends from it, and build_stage13/14 compute the
mix from a REST dict that has no 'hopgood' key. Nothing ever failed: an unnamed group is silently skipped. So the pairs
were not ruled out after stage 6, they were simply never named again. DECISIONS 21 Oct (AUDIT DECISIONS, (3)) puts
them back into stage 15. The fix is structural: every group present must be NAMED (weight 0 allowed and visible), and
the groups in MUST_DRAW must carry a weight above 0.

WHY (step 3, Bible). Rulings 18 Oct (later) and 20 Oct, restated 21 Oct ("literal Bible pairs OUT"): the Bible stays
MONOLINGUAL (Tonga in the copy pool only). Stage 13 dropped the WEB-English verse pairs ('liseli:bible' -> copy);
stage 14 put the Jesuit's literal pilot back as 'synthetic:bible_lit' (694 pairs at 2%, build_stage14.py --bible).
No builder may now emit a Bible pair: by group ('bible', 'bible_lit'), by origin / id (anything with 'bible'), or by
content (a pair whose Tonga is a Bible verse of 5+ words, read from the copy pool's Bible lines and bible_literal/).

The same rules are enforced on the pod by train_nllb.py (self-contained copy). After the Codex guard review (21 Oct,
findings 1, 8, 11, 12, 16): weights finite and >= 0, hopgood present AND drawn, Bible wrappers and verse numbers
stripped before the content check, explicit exceptions (python -O safe); check_build() also runs the held-out guard
over train / mono / tasks / dev and writes preflight.json, which the pod requires before step 1.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "internal", "chitonga_train"))

MUST_DRAW = ("hopgood",)          # DECISIONS 21 Oct (3): the Hopgood pairs are in the stage 15 bundle
BIBLE_GROUPS = ("bible", "bible_lit")
BIBLE_MIN_WORDS = 3               # Codex guard review 8: 5 left short verses out; 1-2 word verses ('Jesu wakalila.') stay
                                  # out ON PURPOSE: they equal ordinary speech and would strip it from every build
BIBLE_INVENTORY_MIN = 25000       # the canonical verse list must be there (stage 14 copy pool: 30,541 lines)
S14 = os.path.join(CT, "runs", "2026-10-18_stage14")


class MixError(RuntimeError):
    """a mix or preflight rule failed. An exception, not an assert: python -O keeps it (Codex guard review 16)"""


def group(o):
    """identical to train_nllb.group (unit_guard.py asserts it): which pool a line belongs to, by its origin label"""
    if o in ("liseli:storybook", "storybooks_zambia", "asb:storybook"):
        return "story"
    if o == "jctr:homily":
        return "prose"
    if o.startswith("hopgood:"):
        return "hopgood"
    if o == "liseli:bible":
        return "bible"
    if o.startswith("synthetic:"):
        return o.split(":", 1)[1]          # stage 7: syn_tab, syn_sub, syn_frm, syn_vrb (one group per method)
    return "own"


def parse_mix(mix):
    """{group: weight}; every weight finite and >= 0, the total > 0 (Codex 11: 'story:-1' or NaN passed and silently
    starved the other groups in random.choices)"""
    import math
    if isinstance(mix, dict):
        out = {k: float(v) for k, v in mix.items()}
    else:
        out = {}
        for kv in mix.split(","):
            g, w = kv.split(":")
            out[g.strip()] = float(w)
    bad = {g: w for g, w in out.items() if not math.isfinite(w) or w < 0}
    if bad:
        raise MixError(f"mix weights must be finite and >= 0: {bad}")
    if not sum(out.values()) > 0:
        raise MixError("mix weights sum to 0")
    return out


def _n(s):
    return " ".join(re.findall(r"[\w']+", (s or "").lower()))


_WRAP = re.compile(r"^\s*(?:\[[^\]]{0,20}\]|<[^>]{0,20}>)\s*")
_VNUM = re.compile(r"^\s*\d{1,3}(?:[:.]\d{1,3})*[a-z]?[.)]?\s+")


def verse_key(s):
    """a Bible-verse key: task wrappers ('[BT] ', '<copy> ') and verse numbers ('12 ', '3:16 ') stripped, then the
    word key (Codex 8: '[BT] '+V and a numbered verse passed)"""
    s = s or ""
    for _ in range(3):
        t = _VNUM.sub("", _WRAP.sub("", s))
        if t == s:
            break
        s = t
    return _n(s)


def bible_tonga(run_dir=None, mono=None):
    """the Bible verse inventory (3+ words): the CANONICAL stage 14 copy pool's 'liseli:bible' lines (so a build that
    dropped them cannot empty the inventory), the given run / mono, and every bible_literal/*.jsonl toi"""
    out = set()
    rows = list(mono or [])
    for d in (run_dir, S14):
        if d and os.path.exists(os.path.join(d, "mono.jsonl")):
            rows += [json.loads(l) for l in open(os.path.join(d, "mono.jsonl"), encoding="utf-8") if l.strip()]
    for r in rows:
        if str(r.get("origin", "")).startswith("liseli:bible"):
            k = verse_key(r.get("toi"))
            if len(k.split()) >= BIBLE_MIN_WORDS:
                out.add(k)
    for f in glob.glob(os.path.join(CT, "bible_literal", "*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                k = verse_key(json.loads(l).get("toi"))
                if len(k.split()) >= BIBLE_MIN_WORDS:
                    out.add(k)
    if len(out) < BIBLE_INVENTORY_MIN:
        raise MixError(f"Bible verse inventory too small ({len(out)} < {BIBLE_INVENTORY_MIN}): the content check would be blind")
    return out


def _toi_of(r):
    if "toi" in r:
        return r.get("toi")
    if r.get("tgt_lang") == "toi":
        return r.get("tgt")
    if r.get("src_lang") == "toi":
        return r.get("src")
    return None


def is_bible_pair(r, verses):
    """a training pair (train.jsonl row, or a task with an English side) that is Bible text with English"""
    o, i = str(r.get("origin", "")), str(r.get("id", ""))
    if "bible" in o.lower() or "bible" in i.lower():
        return "origin/id"
    if r.get("pool") in BIBLE_GROUPS:
        return "pool"
    if verses and verse_key(_toi_of(r)) in verses:
        return "content"
    return None


def check(train, mix, tasks=(), mono=None, verses=None, must_draw=MUST_DRAW, concat=False):
    """the rules, on rows in memory. Returns a report dict; raises MixError on any breach.
    train: train.jsonl rows; tasks: tasks.jsonl rows (pool, src_lang, tgt_lang); mono: the copy pool (Tonga only)."""
    errs = []
    w = parse_mix(mix)
    pools = {}
    for x in train:
        pools.setdefault(group(x["origin"]), []).append(x)
    for x in tasks:
        pools.setdefault(x["pool"], []).append(x)
    if mono:
        pools["copy"] = list(mono)
    unnamed = {g: len(v) for g, v in pools.items() if v and g not in w}
    if unnamed:
        errs.append(f"groups present but not named in the mix (name them; weight 0 leaves one out ON PURPOSE): {unnamed}")
    for g in w:
        if g == "concat" and concat:
            continue
        if not pools.get(g):
            errs.append(f"mix names group {g!r} with no lines")
    for g in must_draw:
        if not pools.get(g):
            errs.append(f"group {g!r} must be in the build (Codex 12: a build without it passed): DECISIONS 21 Oct (3)")
        elif w.get(g, 0) <= 0:
            errs.append(f"group {g!r} must be drawn (weight > 0): DECISIONS 21 Oct (3)")
    for g in BIBLE_GROUPS:
        if pools.get(g):
            errs.append(f"Bible group {g!r} holds {len(pools[g])} English pairs: the Bible is monolingual (rulings 18/20/21 Oct)")
    if verses is None:
        verses = bible_tonga(mono=mono or [])
    bib = {}
    for x in list(train) + [t for t in tasks if "eng" in (t.get("src_lang"), t.get("tgt_lang"))]:
        why = is_bible_pair(x, verses)
        if why:
            bib.setdefault(why, []).append(x.get("id"))
    if bib:
        errs.append("Bible English pairs in the build: " + str({k: (len(v), v[:3]) for k, v in bib.items()}))
    rep = {"groups": {g: len(v) for g, v in pools.items()}, "mix": w, "unnamed": unnamed,
           "zero_weight": sorted(g for g, v in w.items() if v == 0), "bible_pairs": {k: len(v) for k, v in bib.items()},
           "bible_verses_checked": len(verses)}
    if errs:
        raise MixError("mix check failed:\n  " + "\n  ".join(errs))
    return rep


PREFLIGHT = "preflight.json"
PREFLIGHT_FILES = ("train.jsonl", "mono.jsonl", "tasks.jsonl", "dev.jsonl", "test.en.txt", "test_hopgood.en.txt")


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def guard_rows(run_dir, G):
    """every row of train / mono / tasks / dev through the held-out guard (Codex 1: dev selects the checkpoint, so it
    must be as clean as train). Returns {file: [(id, reason)]}"""
    hits = {}
    for f in ("train.jsonl", "mono.jsonl", "tasks.jsonl", "dev.jsonl"):
        p = os.path.join(run_dir, f)
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if "toi" in r or "en" in r:
                toi, en = r.get("toi"), r.get("en")
            else:
                src = _WRAP.sub("", r.get("src") or "")
                toi = r.get("tgt") if r.get("tgt_lang") == "toi" else src if r.get("src_lang") == "toi" else None
                en = r.get("tgt") if r.get("tgt_lang") == "eng" else src if r.get("src_lang") == "eng" else None
            k = G.which(toi, en)
            if k:
                hits.setdefault(f, []).append((r.get("id"), k))
    return hits


def check_build(run_dir, mix, must_draw=MUST_DRAW, concat=False, guard_run=S14):
    """check a built run folder against its mix string AND the held-out guard, then record the PREFLIGHT the pod
    requires (train_nllb.py refuses to train without it, Codex 1): the sha256 of every data file read here, the
    guard's inventory sha, and every mix checked on exactly these files. Calling it again with another mix on
    unchanged files adds that mix; changed files start a new record."""
    import time
    rd = lambda n: [json.loads(l) for l in open(os.path.join(run_dir, n), encoding="utf-8") if l.strip()] \
        if os.path.exists(os.path.join(run_dir, n)) else []
    mono = rd("mono.jsonl")
    rep = check(rd("train.jsonl"), mix, rd("tasks.jsonl"), mono, bible_tonga(mono=mono), must_draw, concat)
    import heldout_guard as HG
    G = HG.load(guard_run)
    hits = guard_rows(run_dir, G)
    if hits:
        raise MixError("held-out guard hits in the build: " + str({f: (len(v), v[:3]) for f, v in hits.items()}))
    files = {f: _sha(os.path.join(run_dir, f)) for f in PREFLIGHT_FILES if os.path.exists(os.path.join(run_dir, f))}
    for need in ("train.jsonl", "dev.jsonl", "test.en.txt"):
        if need not in files:
            raise MixError(f"preflight: {need} missing in {run_dir}")
    p = os.path.join(run_dir, PREFLIGHT)
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    this = ",".join(f"{k}:{v}" for k, v in parse_mix(mix).items())
    mixes = (old["mixes"] if old and old.get("files") == files else []) + [this]
    rec = {"made": time.strftime("%Y-%m-%d %H:%M"), "files": files, "guard_inventory_sha": G.inventory_sha(),
           "guard_sets": len(G.inventory), "mixes": sorted(set(mixes)), "by": "mix_groups.check_build"}
    json.dump(rec, open(p, "w", encoding="utf-8"), indent=1)
    rep["preflight"] = {"file": p, "mixes": rec["mixes"], "guard_inventory_sha": rec["guard_inventory_sha"]}
    return rep


def hopgood_counts(run_dir):
    """the Hopgood group in a run folder, by origin (drill / exercise ...), for the stage 15 builder"""
    import collections
    c = collections.Counter()
    for l in open(os.path.join(run_dir, "train.jsonl"), encoding="utf-8"):
        if l.strip():
            o = json.loads(l)["origin"]
            if group(o) == "hopgood":
                c[o] += 1
    return dict(c)


if __name__ == "__main__":
    import sys
    run, mix = sys.argv[1], sys.argv[2]
    print(json.dumps(check_build(run, mix), indent=1))
