"""Held-out check for the speaker test pool (24 Oct 2026, the Jesuit).

Why: a sentence the translator trained on, or the memory serves, measures recall, not translation; a sentence that is in
the app content or another test set would leak between uses. Every pool sentence is checked, normalised, against:
  stage 13 / 16 / 18 / 19 data (train, tasks, dev, sel), the translation memory (h2_memory, as served), the app content
  (assistant_library cards, core fixed lines, plugin_education, lesson boxes), the diagnosis_500 set, the stage 19
  kill-test sets (items, random classroom 40, probes, teacher dev / held-out), the stage 18 frozen teacher test + dev,
  and the reader bake-off test set v2 (a test set: reported only).
A HIT is: exact (normalised), CONTAINED (the whole sentence inside a longer line, word-bounded), or NEAR (difflib ratio
>= 0.85 on the normalised text, or content-word Jaccard >= 0.8 with the same length +-2 words).
Streaming over every string in every file (all fields: Tonga strings simply never match English).
    python -X utf8 heldout_check.py pool.jsonl heldout_report.json
"""
import difflib
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
RUNS = os.path.join(CT, "runs")
LIB = os.path.join(CT, "assistant_library")
STOP = set("a an the to of and in on at is are be it you your we our i my me for with this that these those do not "
           "can will please now up down out its his her he she they them us".split())


def norm(s):
    s = (s or "").replace("’", "'").replace("‘", "'").lower()
    return " ".join(re.findall(r"[a-z0-9']+", s))


def sents(s):
    for part in re.split(r"(?<=[.!?])\s+|\n+|\s\|\s", s):
        n = norm(part)
        if n:
            yield n


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def iter_file(p):
    if p.endswith(".jsonl"):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                yield from strings(json.loads(line))
            except ValueError:
                yield line
    elif p.endswith(".json"):
        try:
            yield from strings(json.load(open(p, encoding="utf-8")))
        except ValueError:
            yield open(p, encoding="utf-8", errors="replace").read()
    else:
        yield open(p, encoding="utf-8", errors="replace").read()


def sources():
    S = {}
    for st, d in (("stage13", "2026-10-17_stage13"), ("stage16", "2026-10-21_stage16"),
                  ("stage18", "2026-10-23_final_translator_data"), ("stage19", "2026-10-24_stage19_data")):
        S[st] = [os.path.join(RUNS, d, f) for f in ("train.jsonl", "tasks.jsonl", "dev.jsonl", "sel.jsonl",
                                                  "test.en.txt", "test_hopgood.en.txt")
                 if os.path.exists(os.path.join(RUNS, d, f))]
    S["stage18_frozen"] = [os.path.join(RUNS, "2026-10-23_final_translator_data", f)
                           for f in ("teacher_test_final18.jsonl", "teacher_dev_final18.jsonl", "confusable_probes_final18.jsonl")]
    k19 = os.path.join(RUNS, "2026-10-24_stage19_data")
    S["stage19_killtest"] = [os.path.join(k19, "killtest", "items.jsonl"), os.path.join(k19, "random_classroom_40.jsonl"),
                             os.path.join(k19, "confusable_probes_final18.jsonl"), os.path.join(k19, "teacher_dev_stage19.jsonl"),
                             os.path.join(k19, "synthetic_s19_heldout.jsonl")]
    S["diagnosis_500"] = [os.path.join(CT, "llm", "panel", "diagnosis_500_2026-10-21", "set.jsonl")]
    app = glob.glob(os.path.join(LIB, "cards_*.jsonl")) + [os.path.join(LIB, "core_fixed_lines.json")]
    app += glob.glob(os.path.join(LIB, "plugin_education", "*.json*"))
    for sub in ("grade1", "grade1_new", "grade23", "grade1_chitonga", "grade23_chitonga", "cards_v2_src", "cards_v2_g4_src"):
        app += glob.glob(os.path.join(LIB, sub, "**", "*.md"), recursive=True)
        app += glob.glob(os.path.join(LIB, sub, "**", "*.json*"), recursive=True)
    S["app_content"] = sorted(set(app))
    S["reader_bakeoff_testset_v2 (test set, report only)"] = [os.path.join(RUNS, "2026-10-24_reader_bakeoff", "testset_v2.jsonl")]
    return {k: [p for p in v if os.path.exists(p)] for k, v in S.items()}


def main():
    pool = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    C = [(r["id"], norm(r["en"])) for r in pool]
    ctoks = {cid: set(n.split()) - STOP or set(n.split()) for cid, n in C}
    inv = {}
    for cid, t in ctoks.items():
        for w in t:
            inv.setdefault(w, []).append(cid)
    cn = dict(C)
    hits = {cid: [] for cid, _ in C}
    files_read = {}

    def check(src, s):
        toks = set(s.split())
        cnt = Counter()
        for w in toks:
            for cid in inv.get(w, ()):
                cnt[cid] += 1
        for cid, k in cnt.items():
            need = len(ctoks[cid])
            if k < max(1, 0.6 * need):
                continue
            c = cn[cid]
            kind = None
            if s == c:
                kind = "exact"
            elif len(c.split()) >= 3 and (" " + c + " ") in (" " + s + " "):
                kind = "contained"
            else:
                ls, lc = len(s.split()), len(c.split())
                if abs(ls - lc) <= 4:
                    r = difflib.SequenceMatcher(None, s, c).ratio()
                    st = toks - STOP or toks
                    jac = len(st & ctoks[cid]) / max(1, len(st | ctoks[cid]))
                    if r >= 0.85 or (jac >= 0.8 and abs(ls - lc) <= 2):
                        kind = f"near r={r:.2f} j={jac:.2f}"
            if kind and len(hits[cid]) < 6:
                hits[cid].append({"src": src, "kind": kind, "line": s[:160]})

    for src, files in sources().items():
        n = 0
        for p in files:
            for s0 in iter_file(p):
                for s in sents(s0):
                    n += 1
                    check(src, s)
        files_read[src] = {"files": len(files), "sentences": n}
        print(src, files_read[src], flush=True)
    # the translation memory as served (h2_memory, lessons on): every English key
    sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "llm"))
    import h2_memory as M
    m = M.memory()
    n = 0
    for k, rows in m["by"].items():
        for r in rows[:1]:
            for s in sents(r.get("en", "")):
                n += 1
                check("translation_memory", s)
    files_read["translation_memory"] = {"keys": len(m["by"]), "sentences": n, "stamp": m.get("stamp")}
    out = {"files_read": files_read, "hits": {k: v for k, v in hits.items() if v},
           "n_pool": len(C), "n_hit": sum(1 for v in hits.values() if v)}
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("pool", len(C), "hit", out["n_hit"])


if __name__ == "__main__":
    main()
