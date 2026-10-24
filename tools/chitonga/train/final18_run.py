"""The EXECUTOR of the final round's kill test (24 Oct 2026, the Jesuit). The rule lives in final18_killtest.py and was
fixed in PACK.md sections 6-8 before any stage 18 model existed; this file only produces the numbers the rule reads.
Written after the models existed on HF, before any candidate was decoded. K0: Beelink CPU under beelink_lock
'f18-killtest'; the blind judge runs on the subscription (claude -p in an empty folder, as the 22 Oct weakness map did).

    python -X utf8 tools/chitonga/train/final18_run.py regress <model>    # 11 regression lines, served path
    python -X utf8 tools/chitonga/train/final18_run.py sweep <model>      # the 150 negation-sweep lines, served, resumable
    python -X utf8 tools/chitonga/train/final18_run.py teach <model>      # the frozen 44, both directions: served + raw
    python -X utf8 tools/chitonga/train/final18_run.py raw <model>        # P1 174 + El Nino 48 + all 687 document items, raw
    python -X utf8 tools/chitonga/train/final18_run.py blind              # packs + keys (p1, docs, teach, teachraw, neg sheet)
    python -X utf8 tools/chitonga/train/final18_run.py judge <pack> [--par 3]
    python -X utf8 tools/chitonga/train/final18_run.py score              # opens the keys ONLY when every label is in

WHY these shapes (each is a precedent of the record, not a new idea):
- SERVED path = negation_sweep.make_service: translator_service with its SOUP swapped for the candidate; the n-best's
  other readers (s13 seeds) and the back-translation checker stay as installed; memory OFF (a memory hit is the same
  line for every system); term guard and the 24 Oct glossary lever ON, as a teacher would meet them. The piece-level
  fast path is the one validated exact on 22 Oct (full vs fast 42/42 + 36/36). One model per PROCESS, because the swap
  is a module global.
- The 150 sweep lines are exactly the 150 common lines of 22 Oct (a050's gen file): the pre-registered bar "a050's 8"
  was counted on them. a050's own texts (served CPU re-check where it exists, else the GPU decode of the same weights)
  sit on the read sheet as unmarked letters so the hand read is blind to system; a050 is NOT recounted: 8 is the bar.
- RAW decodes for P1 and the documents use h2_reader.Reader._run (beam 4, max 256) in batches of 16 in file order per
  direction over ALL items, the batching of stage15_diag_judge.gen / merge_judge.gen, so wise-a050's 22 Oct outputs can
  be reused unchanged (int8 CPU decoding is deterministic for the same batch; proved on 32 items before reuse; taking
  a line out of a batch can move its neighbours, 22 Oct). Labels for texts already read blind on 15/16/22 Oct are
  CARRIED by (item, text) through merge_judge.earlier_labels, the same chain as every earlier judge.
- The documents are two sets: the 300 AB_PREREG items (stage 16 sample 150 + the fresh 150) and the never-judged pool,
  173 lines of LOGBOOK 22 Oct (reproduced here from the files: lines none of whose directions were ever judged) minus
  the 9 lines the 22 Oct writing check used = 164 lines (both directions where the item exists).
- The frozen 44 (sha asserted on every read) are judged A/B against a050 SERVED, the teacher's line as the reference
  (PACK 7); the RAW A/B is reported beside it (the weights' own change, before the serve-time levers).
- Blinding: letters shuffled per item with a seeded generator; keys written at `blind`, read only by `score`, and only
  when every judged file is complete; the judged files are hashed into result.json.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "tools", "chitonga", "llm")]
CT = os.path.join(REPO, "internal", "chitonga_train")
RUN = os.path.join(CT, "runs", "2026-10-23_final_translator_data")
KT = os.path.join(RUN, "killtest")
M22 = os.path.join(CT, "runs", "2026-10-22_merge")
DOC16 = os.path.join(CT, "runs", "2026-10-21_stage16", "doctest")
WCHK = os.path.join(CT, "llm", "panel", "writing_check_2026-10-22_a050_vs_claude")
MODELS = os.environ.get("CHITONGA_MODELS", r"models")
A050 = "wise-a050"
SYSTEMS = ["f18-B18m03-s1", "f18-B18m03-s2", "f18-B18m05-s1", "f18-B18m05-s2", "f18-B18m07-s1", "f18-B18m07-s2"]
RAWB18 = ["f18-B18-s1", "f18-B18-s2"]           # reported for the paper; no gate reads them
ALPHA = {"03": ("f18-B18m03-s1", "f18-B18m03-s2"), "05": ("f18-B18m05-s1", "f18-B18m05-s2"),
         "07": ("f18-B18m07-s1", "f18-B18m07-s2")}
A050_BAR = 8                                     # PACK 6 gate 1: a050's critical errors on the 150 (served CPU, 22 Oct)
LET = "ABCDEFGHIJ"
LANG = {"en>toi": ("eng_Latn", "toi_Latn"), "toi>en": ("toi_Latn", "eng_Latn")}
RANK = {"right": 2, "partly": 1, "wrong": 0}
POSTER = "e2t:bk:poster_moh_who_cholera_pairs:"
REG_IIDS = [POSTER + str(i) for i in (0, 1, 7, 8, 10, 13, 14)] + ["ce2t:06", "ce2t:17"]
FOCUS = "test:zema:3rd_draft_sessd_eis_livingstone_bernt_checked_sw:zema:3rd_draft_sessd_eis_livingstone_bernt_checked_sw:0047:en>toi"
HARNESS_LINE = {"iid": "harness:before_eating:en>toi", "set": "harness", "dir": "en>toi",
                "src": "Wash your hands with soap before eating.", "ref": None}
# the frozen 44's confusable items (PACK 6 gate 4: "no item is worse" on the probes inside it): matched on these words
CONFUSABLE = re.compile(r"\b(maths?|mathematics|number|amount|throat|breast|nails?|cholera|corona|muscle|sound|line)\b|"
                        r"masaamu|masamu|nombolo|mweelwe|nkolo|goyooyo|\bmala\b|kkolela|kkolona|milambi", re.I)
JUDGE_MODEL = "claude-opus-5-5"                  # the weakness map's judge (22 Oct); fresh session per batch
BATCH = {"p1": 40, "docs": 15, "teach": 40, "teachraw": 40}
ASK = ("Read prompt.md in this folder and do exactly what it says. items.jsonl holds {n} items; judge every item. "
       "Write the answer to judged.jsonl in this folder (JSON Lines, one line per item, in the order of items.jsonl). "
       "When the file is complete, reply with the single line: DONE <number of lines written>.")
LOCK = threading.Lock()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def wl(p, rows, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def frozen44():
    p = os.path.join(RUN, "teacher_test_final18.jsonl")
    want = open(p + ".sha256", encoding="utf-8").read().split()[0]
    assert sha(p) == want, "the frozen teacher test has changed: refusing"
    return jl(p)


def sweep_iids():
    return [x["iid"] for x in jl(os.path.join(M22, "negation_sweep", "gen_wise-a050.jsonl"))]


def a050_sweep_texts():
    """a050's text per sweep line: the served CPU re-check (22 Oct, 16 items) where it exists, else the GPU decode"""
    t = {x["iid"]: x["out"] for x in jl(os.path.join(M22, "negation_sweep", "gen_wise-a050.jsonl"))}
    t.update({x["iid"]: x["out"] for x in jl(os.path.join(M22, "negation_sweep", "cpu_check_wise-a050.jsonl"))})
    return t


def doc_sets():
    """(the 300 AB_PREREG iids, the 164-line pool's iids): reproduces LOGBOOK 22 Oct's 173 lines, then takes out the 9
    writing-check lines by their English"""
    items = jl(os.path.join(DOC16, "test_doc_items.jsonl"))
    line = lambda iid: iid.rsplit(":", 1)[0]
    s300 = {x["iid"] for x in jl(os.path.join(DOC16, "test_doc_sample150.jsonl"))} | \
           {x["iid"] for x in jl(os.path.join(M22, "confirm_doc_fresh150.jsonl"))}
    assert len(s300) == 300, len(s300)
    judged = {line(i) for i in s300}
    judged |= {line(x["iid"]) for x in jl(os.path.join(DOC16, "doc_judged.jsonl"))}
    judged |= {line(x["iid"]) for x in jl(os.path.join(M22, "docs", "doc_ranked.jsonl"))}
    lines = {}
    for x in items:
        lines.setdefault(line(x["iid"]), []).append(x)
    pool = [l for l in lines if l not in judged]
    assert len(pool) == 173 and sum(len(lines[l]) for l in pool) == 239, (len(pool), "LOGBOOK 22 Oct says 173 / 239")
    norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().lower()
    used_en = {norm(x["en"]) for x in jl(os.path.join(WCHK, "lines_en.jsonl")) if x["src"].startswith("stage 16 held-out document")}
    assert len(used_en) == 9, len(used_en)
    out, hit = set(), set()
    for l in pool:
        ens = {norm(x["src"] if x["dir"] == "en>toi" else x["ref"]) for x in lines[l]}
        if ens & used_en:
            hit |= ens & used_en
            continue
        out |= {x["iid"] for x in lines[l]}
    assert len(hit) == 9 and len(pool) - 9 == 164, (len(hit), "the 9 writing-check lines must all be found")
    return s300, out


# --------------------------------------------------------------------------------------------- served decodes
def served(model):
    import negation_sweep as N
    TS, full, fast = N.make_service(model)
    return N, TS, full, fast


def regress(a):
    import negation_sweep as N
    rows = [r for r in N.items() if r["iid"] in REG_IIDS or r["iid"] == FOCUS] + [HARNESS_LINE]
    assert len(rows) == 11, len(rows)
    N, TS, full, fast = served(a.model)
    out = []
    for r in rows:
        t = time.time()
        o, path, _, alarms = N.translate(TS, full, fast, r["src"], r["dir"])
        out.append({**r, "model": a.model, "out": o, "path": path, "alarms": alarms, "t": round(time.time() - t, 1)})
        print(r["iid"][-45:], "|", o[:110], flush=True)
    wl(os.path.join(KT, f"regress_{a.model}.jsonl"), out)


def sweep(a):
    import negation_sweep as N
    want = sweep_iids()
    rows = {r["iid"]: r for r in N.items() if r["iid"] in set(want)}
    p = os.path.join(KT, f"sweep_{a.model}.jsonl")
    done = {x["iid"] for x in jl(p)}
    todo = [rows[i] for i in want if i not in done]
    print(a.model, "sweep:", len(done), "done,", len(todo), "to go", flush=True)
    if not todo:
        return
    N, TS, full, fast = served(a.model)
    t0 = time.time()
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(todo):
            t = time.time()
            o, path, _, alarms = N.translate(TS, full, fast, r["src"], r["dir"])
            f.write(json.dumps({**r, "model": a.model, "out": o, "path": path,
                                "alarms": [x for x in alarms if "negation" in x or "term_guard" in x],
                                "t": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(a.model, len(done) + i + 1, "/", len(want), f"{time.time() - t0:.0f}s", path, flush=True)
    print("done", a.model, f"{time.time() - t0:.0f}s", flush=True)


def teach(a):
    rows = frozen44()
    N, TS, full, fast = served(a.model)
    out = []
    t0 = time.time()
    for r in rows:
        for d, src in (("en>toi", r["en"]), ("toi>en", r["toi"])):
            o, path, _, alarms = N.translate(TS, full, fast, src, d)
            out.append({"iid": r["id"], "dir": d, "teacher": r["teacher"], "src": src,
                        "ref": r["toi"] if d == "en>toi" else r["en"], "model": a.model, "out": o, "path": path,
                        "alarms": alarms})
        print(r["id"], "|", out[-2]["out"][:70], "||", out[-1]["out"][:60], flush=True)
    wl(os.path.join(KT, f"teach_{a.model}.jsonl"), out)
    print("served 88 in", f"{time.time() - t0:.0f}s", flush=True)
    import h2_reader as R
    rd = R.Reader(os.path.join(MODELS, "chitonga-nllb-" + a.model))
    raw = []
    for d, key in (("en>toi", "en"), ("toi>en", "toi")):
        for i in range(0, len(rows), 16):
            b = rows[i:i + 16]
            for r, o in zip(b, rd._run([x[key] for x in b], *LANG[d], 256)):
                raw.append({"iid": r["id"], "dir": d, "teacher": r["teacher"], "src": r[key],
                            "ref": r["toi"] if d == "en>toi" else r["en"], "model": a.model, "out": o})
    wl(os.path.join(KT, f"teachraw_{a.model}.jsonl"), raw)
    print("raw 88 done", flush=True)


# --------------------------------------------------------------------------------------------- raw decodes
def p1_items():
    return jl(os.path.join(M22, "prescreen", "items.jsonl"))


def doc_items():
    return jl(os.path.join(DOC16, "test_doc_items.jsonl"))


def raw(a):
    """P1 174 + El Nino 48 and all 687 document items, batches of 16 in file order per direction (the 22 Oct batching).
    wise-a050: the stored 22 Oct outputs are reused once a 32-item re-decode (the first batch per direction and set)
    is identical; otherwise it is decoded in full like a candidate."""
    import h2_reader as R
    p1, docs = p1_items(), doc_items()
    stored = None
    if a.model == A050:
        g1 = {x["iid"]: x["out"][A050] for x in jl(os.path.join(M22, "prescreen", "gen.jsonl"))}
        g2 = {x["iid"]: x["out"][A050] for x in jl(os.path.join(M22, "docs", "gen_docs.jsonl"))}
        stored = {**g1, **g2}
    rd = R.Reader(os.path.join(MODELS, "chitonga-nllb-" + a.model))
    out, same, checked = {}, 0, 0
    t0 = time.time()
    for name, items in (("p1", p1), ("docs", docs)):
        for d in ("en>toi", "toi>en"):
            sub = [x for x in items if x["dir"] == d]
            for i in range(0, len(sub), 16):
                b = sub[i:i + 16]
                if stored is not None and i > 0:
                    for x in b:
                        out[x["iid"]] = stored[x["iid"]]
                    continue
                for x, o in zip(b, rd._run([x["src"] for x in b], *LANG[d], 256)):
                    out[x["iid"]] = o
                    if stored is not None:
                        checked += 1
                        same += o == stored[x["iid"]]
                print(a.model, name, d, i + len(b), "/", len(sub), f"{time.time() - t0:.0f}s", flush=True)
        if stored is not None:
            print(f"a050 identity check so far: {same}/{checked}", flush=True)
    if stored is not None:
        if same != checked:
            print(f"a050 re-decode differs from the 22 Oct outputs on {checked - same} of {checked}: decoding a050 in full", flush=True)
            a.model = A050 + "_fresh"
            out = {}
            for name, items in (("p1", p1), ("docs", docs)):
                for d in ("en>toi", "toi>en"):
                    sub = [x for x in items if x["dir"] == d]
                    for i in range(0, len(sub), 16):
                        b = sub[i:i + 16]
                        for x, o in zip(b, rd._run([x["src"] for x in b], *LANG[d], 256)):
                            out[x["iid"]] = o
            a.model = A050
        json.dump({"checked": checked, "identical": same}, open(os.path.join(KT, "a050_identity_check.json"), "w"))
    rows = [{"iid": x["iid"], "set": "p1" if x.get("set") == "p1" else "climate", "dir": x["dir"], "src": x["src"],
             "ref": x.get("ref"), "model": a.model, "out": out[x["iid"]]} for x in p1]
    rows += [{"iid": x["iid"], "set": "docs", "doc": x["doc"], "dir": x["dir"], "src": x["src"], "ref": x.get("ref"),
              "model": a.model, "out": out[x["iid"]]} for x in docs]
    wl(os.path.join(KT, f"raw_{a.model}.jsonl"), rows)
    print("raw done", a.model, len(rows), f"{time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------------------------- blind packs
def have(prefix, systems):
    return [m for m in systems if os.path.exists(os.path.join(KT, f"{prefix}_{m}.jsonl"))]


def rnd_for(iid):
    return random.Random(int(hashlib.sha256(("f18:" + iid).encode()).hexdigest()[:12], 16) + 1824)


def lexicon_copy():
    """the lexicon as the diagnosis packs give it to the judge: speaker names redacted, nothing trimmed"""
    src = os.path.join(REPO, "tools", "chitonga", "grammar2", "lexicon.tsv")
    # PUBLIC COPY: the working copy lists the speakers' names here; the list is read from the environment instead
    names = re.compile(r"(?i)\b(fr\.?\s*)?(" + (os.environ.get("SPEAKER_NAMES_RX") or "(?!x)x") + r")\b")
    lex = names.sub("speaker", open(src, encoding="utf-8").read())
    p = os.path.join(KT, "packs", "lexicon.tsv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8", newline="\n").write(lex)
    return p


def letters_for(iid, texts_by_sys, rnd):
    texts = sorted(set(texts_by_sys.values()))
    rnd.shuffle(texts)
    return {LET[i]: t for i, t in enumerate(texts)}, {LET[i]: [m for m, t in texts_by_sys.items() if t == texts[i]]
                                                     for i in range(len(texts))}


def blind(a):
    import merge_judge as MJ   # the label carry chain (15 / 16 / 22 Oct)
    os.makedirs(os.path.join(KT, "packs"), exist_ok=True)
    lexicon_copy()
    systems = have("raw", [A050] + SYSTEMS + RAWB18)
    assert A050 in systems, "raw wise-a050 first"
    # P1 + El Nino: labels per distinct text, earlier labels carried
    raws = {m: {x["iid"]: x for x in jl(os.path.join(KT, f"raw_{m}.jsonl"))} for m in systems}
    old = MJ.earlier_labels()
    key, items_, pack, carried_n, todo_n = {}, [], [], 0, 0
    for x in p1_items():
        iid = x["iid"]
        by = {m: raws[m][iid]["out"] for m in systems}
        lt, k = letters_for(iid, by, rnd_for(iid))
        key[iid] = k
        pre = {L: old[(iid, t)] for L, t in lt.items() if (iid, t) in old}
        carried_n += len(pre)
        items_.append({"iid": iid, "set": "p1" if x.get("set") == "p1" else "climate", "dir": x["dir"], "src": x["src"],
                       "ref": x.get("ref"), **lt, "carried": pre})
        need = {L: t for L, t in lt.items() if L not in pre}
        if need:
            todo_n += len(need)
            pack.append({"item": iid, "set": items_[-1]["set"], "dir": x["dir"], "src": x["src"], "ref": x.get("ref"),
                         "candidates": need})
    wl(os.path.join(KT, "p1_items.jsonl"), items_)
    json.dump(key, open(os.path.join(KT, "KEY_p1_do_not_open.json"), "w", encoding="utf-8"), indent=0)
    write_pack("p1", pack)
    print(f"p1: {len(items_)} items, {carried_n} labels carried, {todo_n} texts to label in {len(pack)} items", flush=True)
    # documents: ranking of the distinct texts per item (300 + 164-line pool)
    s300, s164 = doc_sets()
    key, items_, pack, one = {}, [], [], 0
    for x in doc_items():
        iid = x["iid"]
        if iid not in s300 and iid not in s164:
            continue
        by = {m: raws[m][iid]["out"] for m in systems}
        lt, k = letters_for(iid, by, rnd_for(iid))
        key[iid] = k
        items_.append({"iid": iid, "set": "docs300" if iid in s300 else "docs164", "doc": x["doc"], "dir": x["dir"],
                       "src": x["src"], "ref": x.get("ref"), **lt})
        if len(lt) == 1:
            one += 1
            continue
        pack.append({"item": iid, "doc": x["doc"].split(":")[1], "dir": x["dir"], "src": x["src"], "ref": x.get("ref"),
                     "candidates": lt})
    wl(os.path.join(KT, "docs_items.jsonl"), items_)
    json.dump(key, open(os.path.join(KT, "KEY_docs_do_not_open.json"), "w", encoding="utf-8"), indent=0)
    write_pack("docs", pack)
    print(f"docs: {len(items_)} items ({sum(1 for i in items_ if i['set'] == 'docs300')} + "
          f"{sum(1 for i in items_ if i['set'] == 'docs164')}), {one} with one text for all, {len(pack)} to rank", flush=True)
    # the frozen 44: A/B per candidate vs a050, served and raw
    for prefix in ("teach", "teachraw"):
        sys_ = have(prefix, [A050] + SYSTEMS + RAWB18)
        if A050 not in sys_ or len(sys_) < 2:
            print(prefix, "skipped (need a050 and one candidate)", flush=True)
            continue
        outs = {m: {(x["iid"], x["dir"]): x for x in jl(os.path.join(KT, f"{prefix}_{m}.jsonl"))} for m in sys_}
        key, pack, n_id = {}, [], 0
        for (iid, d), base in sorted(outs[A050].items()):
            a0 = base["out"]
            seen = {}
            for m in sys_[1:]:
                t = outs[m][(iid, d)]["out"]
                if t == a0:
                    key.setdefault(f"{iid}:{d}:id", {"identical": True, "systems": []})["systems"].append(m)
                    n_id += 1
                    continue
                if t in seen:
                    key[seen[t]]["systems"].append(m)
                    continue
                item = f"{iid}:{d}:{len(seen)}"
                seen[t] = item
                flip = rnd_for(item).random() < 0.5
                key[item] = {"A": "cand" if flip else A050, "B": A050 if flip else "cand", "systems": [m], "cand_text": t}
                pack.append({"item": item, "dir": d, "source": base["src"], "teacher_reference": base["ref"],
                             "A": t if flip else a0, "B": a0 if flip else t})
        json.dump(key, open(os.path.join(KT, f"KEY_{prefix}_do_not_open.json"), "w", encoding="utf-8"), indent=0)
        write_pack(prefix, pack)
        print(f"{prefix}: systems {sys_[1:]}, {n_id} identical to a050, {len(pack)} pairs to judge", flush=True)
    neg_sheet()


def write_pack(name, pack):
    d = os.path.join(KT, "packs", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "batches"))
    wl(os.path.join(d, "items.jsonl"), pack)
    n = BATCH[name]
    for i in range(0, len(pack), n):
        wl(os.path.join(d, "batches", f"b{i // n:02d}", "items.jsonl"), pack[i:i + n])


def neg_sheet():
    """the negation read sheet: every sweep line where any system is auto-flagged, or where the source carries a
    negation / obligation and the systems' texts differ; letters over the distinct texts incl. a050's (unmarked)"""
    import negation_sweep as N
    sys_ = have("sweep", SYSTEMS)
    if not sys_:
        return
    want = sweep_iids()
    rows = {r["iid"]: r for r in N.items() if r["iid"] in set(want)}
    gens = {m: {x["iid"]: x for x in jl(os.path.join(KT, f"sweep_{m}.jsonl"))} for m in sys_}
    common = [i for i in want if all(i in gens[m] for m in sys_)]
    a0 = a050_sweep_texts()
    key, lines, auto = {}, [], {}
    for iid in common:
        r = rows[iid]
        texts_by = {m: gens[m][iid]["out"] for m in sys_}
        texts_by[A050] = a0[iid]
        fl = {m: N.auto_flags(r, t) for m, t in texts_by.items()}
        auto[iid] = fl
        differ = len(set(texts_by.values())) > 1
        if not any(fl.values()) and not (any(N.src_hits_check(r)) and differ):
            continue
        lt, k = letters_for(iid, texts_by, rnd_for("neg:" + iid))
        key[iid] = k
        lines.append(f"## {iid} [{r['dir']}] flags: {sorted({f for m in texts_by for f in fl[m]})}\n   src: {r['src']}\n"
                     f"   ref: {r.get('ref')}")
        lines += [f"   {L}: {t}" for L, t in lt.items()]
    open(os.path.join(KT, "neg_read_sheet.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(key, open(os.path.join(KT, "KEY_neg_do_not_open.json"), "w", encoding="utf-8"), indent=0)
    json.dump(auto, open(os.path.join(KT, "neg_auto_flags.json"), "w", encoding="utf-8"), indent=0)
    print(f"negation sheet: {len(common)} lines common to {sys_}; {len(key)} on the sheet; "
          f"{sum(len(v) for v in key.values())} texts to read -> neg_read_sheet.txt (write neg_read.jsonl)", flush=True)
    # the regression lines: the same sheet shape (a050 decoded today beside the candidates)
    rs = have("regress", [A050] + SYSTEMS + RAWB18)
    if rs:
        regs = {m: {x["iid"]: x for x in jl(os.path.join(KT, f"regress_{m}.jsonl"))} for m in rs}
        key, lines = {}, []
        for iid in list(regs[rs[0]]):
            texts_by = {m: regs[m][iid]["out"] for m in rs if iid in regs[m]}
            lt, k = letters_for(iid, texts_by, rnd_for("reg:" + iid))
            key[iid] = k
            r = regs[rs[0]][iid]
            lines.append(f"## {iid} [{r['dir']}]\n   src: {r['src']}\n   ref: {r.get('ref')}")
            lines += [f"   {L}: {t}" for L, t in lt.items()]
        open(os.path.join(KT, "reg_read_sheet.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        json.dump(key, open(os.path.join(KT, "KEY_reg_do_not_open.json"), "w", encoding="utf-8"), indent=0)
        print(f"regression sheet: {len(key)} lines, systems {rs} -> reg_read_sheet.txt (write reg_read.jsonl)", flush=True)


# --------------------------------------------------------------------------------------------- the judge
def valid(name, it, ans):
    a = ans.get(it["item"])
    if not a:
        return False
    if name == "p1":
        lab = a.get("labels", {})
        return set(it["candidates"]) <= set(lab) and all(lab[L].get("label") in RANK for L in it["candidates"])
    if name == "docs":
        s = str(a.get("rank", "")).replace(" ", "")
        got = [L for grp in s.split(">") for L in grp.split("=") if L]
        return sorted(got) == sorted(it["candidates"]) and len(got) == len(set(got))
    return a.get("v") in ("A", "B", "tie")


def run_batch(name, b, items, dest, done, scratch):
    todo = [it for it in items if it["item"] not in done]
    for attempt in (1, 2):
        if not todo:
            return b, "complete"
        wd = os.path.join(scratch, f"{name}_{b}_t{attempt}_{int(time.time())}")
        os.makedirs(wd)
        wl(os.path.join(wd, "items.jsonl"), todo)
        shutil.copy2(os.path.join(KT, "prompts", f"{'teach' if name.startswith('teach') else name}.md"), os.path.join(wd, "prompt.md"))
        shutil.copy2(os.path.join(KT, "packs", "lexicon.tsv"), wd)
        t0 = time.time()
        r = subprocess.run(["claude", "-p", ASK.format(n=len(todo)), "--model", JUDGE_MODEL, "--permission-mode", "acceptEdits",
                            "--disallowedTools", "Bash,WebFetch,WebSearch,Task", "--output-format", "json"],
                           cwd=wd, capture_output=True, text=True, encoding="utf-8", timeout=5400, shell=False)
        meta = {"pack": name, "batch": b, "attempt": attempt, "n": len(todo), "secs": round(time.time() - t0), "rc": r.returncode}
        try:
            d = json.loads(r.stdout)
            meta.update({k: d.get(k) for k in ("is_error", "num_turns", "total_cost_usd")}, models=list(d.get("modelUsage", {})),
                        result=str(d.get("result"))[:200])
        except ValueError:
            meta["stdout_tail"] = (r.stdout + r.stderr)[-600:]
        ans = {}
        jp = os.path.join(wd, "judged.jsonl")
        if os.path.exists(jp):
            os.makedirs(os.path.join(KT, "judged", "raw"), exist_ok=True)
            shutil.copy2(jp, os.path.join(KT, "judged", "raw", f"{name}_{b}_t{attempt}.jsonl"))
            for line in open(jp, encoding="utf-8"):
                line = line.strip().strip("`")
                if line.startswith("{"):
                    try:
                        x = json.loads(line)
                        if "item" in x:
                            ans[x["item"]] = x
                    except ValueError:
                        pass
        got = [it for it in todo if valid(name, it, ans)]
        meta["complete"] = len(got)
        with LOCK:
            with open(dest, "a", encoding="utf-8", newline="\n") as f:
                for it in got:
                    f.write(json.dumps({**ans[it["item"]], "judge": JUDGE_MODEL + " (claude -p)", "batch": b,
                                        "attempt": attempt}, ensure_ascii=False) + "\n")
                    done.add(it["item"])
            with open(os.path.join(KT, "judged", "calls.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(name, b, f"attempt {attempt}: {len(got)}/{len(todo)} in {meta['secs']}s", flush=True)
        todo = [it for it in todo if it["item"] not in done]
    return b, "complete" if not todo else f"MISSING {len(todo)}"


def judge(a):
    name = a.pack
    scratch = os.environ.get("JUDGE_SCRATCH", os.path.join(os.environ.get("TEMP", "/tmp"), "claude_judge_f18"))
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(os.path.join(KT, "judged"), exist_ok=True)
    dest = os.path.join(KT, "judged", f"{name}.jsonl")
    done = {x["item"] for x in jl(dest)}
    bdir = os.path.join(KT, "packs", name, "batches")
    jobs = []
    for b in sorted(os.listdir(bdir)):
        if a.only and b not in a.only.split(","):
            continue
        items = jl(os.path.join(bdir, b, "items.jsonl"))
        if all(it["item"] in done for it in items):
            continue
        jobs.append((b, items))
    print(name, len(jobs), "batches to run,", len(done), "items already judged", flush=True)
    with cf.ThreadPoolExecutor(a.par) as ex:
        for b, st in ex.map(lambda j: run_batch(name, j[0], j[1], dest, done, scratch), jobs):
            print("BATCH", b, st, flush=True)
    print("ALL", name, len(done), flush=True)


# --------------------------------------------------------------------------------------------- score
def parse_rank(s):
    lv = {}
    for i, grp in enumerate(str(s).replace(" ", "").split(">")):
        for L in grp.split("="):
            if L:
                lv[L] = i
    return lv


def score(a):
    import final18_killtest as K
    res = {"files": {}, "systems": {}, "per_alpha": {}}
    systems = have("raw", SYSTEMS + RAWB18)
    # ---- P1 labels
    key = json.load(open(os.path.join(KT, "KEY_p1_do_not_open.json"), encoding="utf-8"))
    items = {x["iid"]: x for x in jl(os.path.join(KT, "p1_items.jsonl"))}
    jud = {x["item"]: x for x in jl(os.path.join(KT, "judged", "p1.jsonl"))}
    lab, missing = {}, []
    for iid, k in key.items():
        got = dict(items[iid]["carried"])
        got.update({L: v["label"] for L, v in jud.get(iid, {}).get("labels", {}).items()})
        if any(L not in got for L in k):
            missing.append(iid)
            continue
        lab[iid] = {m: RANK[got[L]] for L, mm in k.items() for m in mm}
    if missing and not a.force:
        raise SystemExit(f"P1: {len(missing)} items without a complete label set, e.g. {missing[:3]}; judge first")
    res["files"]["p1_judged_sha256"] = sha(os.path.join(KT, "judged", "p1.jsonl"))[:16]
    res["p1_missing"] = len(missing)
    # ---- documents ranking
    dkey = json.load(open(os.path.join(KT, "KEY_docs_do_not_open.json"), encoding="utf-8"))
    ditems = {x["iid"]: x for x in jl(os.path.join(KT, "docs_items.jsonl"))}
    djud = {x["item"]: x for x in jl(os.path.join(KT, "judged", "docs.jsonl"))}
    dmiss = [i for i, k in dkey.items() if len(k) > 1 and i not in djud]
    if dmiss and not a.force:
        raise SystemExit(f"docs: {len(dmiss)} items unranked, e.g. {dmiss[:3]}; judge first")
    res["files"]["docs_judged_sha256"] = sha(os.path.join(KT, "judged", "docs.jsonl"))[:16]
    res["docs_missing"] = len(dmiss)
    # ---- teacher test A/B (served = the gate; raw reported)
    tk, tj = {}, {}
    for prefix in ("teach", "teachraw"):
        p = os.path.join(KT, f"KEY_{prefix}_do_not_open.json")
        if not os.path.exists(p):
            continue
        tk[prefix] = json.load(open(p, encoding="utf-8"))
        tj[prefix] = {x["item"]: x for x in jl(os.path.join(KT, "judged", f"{prefix}.jsonl"))}
        tm = [i for i, k in tk[prefix].items() if not k.get("identical") and i not in tj[prefix]]
        if tm and prefix == "teach" and not a.force:
            raise SystemExit(f"teach: {len(tm)} pairs unjudged, e.g. {tm[:3]}; judge first")
        res["files"][f"{prefix}_judged_sha256"] = sha(os.path.join(KT, "judged", f"{prefix}.jsonl"))[:16] if tj[prefix] else None
        res[f"{prefix}_missing"] = len(tm)
    # ---- negation hand read
    nkey = json.load(open(os.path.join(KT, "KEY_neg_do_not_open.json"), encoding="utf-8"))
    nread = {x["iid"]: x for x in jl(os.path.join(KT, "neg_read.jsonl"))}
    nmiss = [i for i in nkey if i not in nread or any(L not in nread[i] for L in nkey[i])]
    if nmiss and not a.force:
        raise SystemExit(f"negation: {len(nmiss)} sheet items unread, e.g. {nmiss[:3]}")
    res["files"]["neg_read_sha256"] = sha(os.path.join(KT, "neg_read.jsonl"))[:16] if nread else None
    rkey = json.load(open(os.path.join(KT, "KEY_reg_do_not_open.json"), encoding="utf-8")) if os.path.exists(os.path.join(KT, "KEY_reg_do_not_open.json")) else {}
    rread = {x["iid"]: x for x in jl(os.path.join(KT, "reg_read.jsonl"))}
    rmiss = [i for i in rkey if i not in rread or any(L not in rread[i] for L in rkey[i])]
    if rmiss and not a.force:
        raise SystemExit(f"regression: {len(rmiss)} lines unread, e.g. {rmiss[:3]}")
    rain = {(x["model"], x["id"]): x for x in jl(os.path.join(KT, "rain_read.jsonl"))}
    # ---- per system
    s300, s164 = doc_sets()
    for m in systems:
        S = {}
        # P1 / climate
        for st in ("p1", "climate"):
            b = w = 0
            for iid, l in lab.items():
                if items[iid]["set"] != st or m not in l:
                    continue
                b += l[m] > l[A050]
                w += l[m] < l[A050]
            S[st] = [b, w]
        # docs
        for st, ids in (("docs300", s300), ("docs164", s164)):
            b = w = t = idn = 0
            for iid in ids:
                k = dkey.get(iid)
                if not k:
                    continue
                lm = next((L for L, mm in k.items() if m in mm), None)
                lr = next((L for L, mm in k.items() if A050 in mm), None)
                if lm is None:
                    continue
                if lm == lr:
                    idn += 1
                    continue
                if iid not in djud:
                    continue
                lv = parse_rank(djud[iid]["rank"])
                if lv[lm] < lv[lr]:
                    b += 1
                elif lv[lm] > lv[lr]:
                    w += 1
                else:
                    t += 1
            S[st] = [b, w, t, idn]
        # teacher test
        for prefix in tk:
            b = w = t = idn = cw = 0
            worse_items, better_items = [], []
            tt = {x["iid"] + ":" + x["dir"]: x for x in jl(os.path.join(KT, f"{prefix}_{A050}.jsonl"))}
            for item, k in tk[prefix].items():
                if m not in k.get("systems", []):
                    continue
                if k.get("identical"):
                    idn += 1
                    continue
                j = tj[prefix].get(item)
                if not j:
                    continue
                v = j["v"]
                base = item.rsplit(":", 1)[0]
                conf = bool(CONFUSABLE.search((tt.get(base, {}).get("src") or "") + " " + (tt.get(base, {}).get("ref") or "")))
                if v == "tie":
                    t += 1
                elif k[v] == "cand":
                    b += 1
                    better_items.append(base)
                else:
                    w += 1
                    worse_items.append(base)
                    cw += conf
            S[prefix] = {"better": b, "worse": w, "tie": t, "identical": idn, "confusable_worse": cw,
                         "better_items": better_items, "worse_items": worse_items}
        # negation
        errs = []
        for iid, k in nkey.items():
            L = next((L for L, mm in k.items() if m in mm), None)
            if L and nread.get(iid, {}).get(L) in ("reversal", "drop"):
                errs.append([iid, nread[iid][L]])
        swept = len(jl(os.path.join(KT, f"sweep_{m}.jsonl")))
        S["neg"] = {"swept": swept, "critical": len(errs), "items": errs}
        regs = []
        for iid, k in rkey.items():
            L = next((L for L, mm in k.items() if m in mm), None)
            if L and rread.get(iid, {}).get(L) in ("reversal", "drop"):
                regs.append([iid, rread[iid][L]])
        S["regression_critical"] = regs
        # probes + rainfall
        pr = jl(os.path.join(RUN, "probes", f"{m}.jsonl"))
        S["probe_fail"] = [x["id"] for x in pr if x.get("forbidden_hit")]
        S["probe_expected_missing"] = [x["id"] for x in pr if x.get("set") != "zmd_rainfall" and not x.get("expected_hit")]
        rr = [x for x in pr if x.get("set") == "zmd_rainfall"]
        S["rain"] = {"right": sum(1 for x in rr if rain.get((m, x["id"]), {}).get("right")),
                     "reversed": sum(1 for x in rr if rain.get((m, x["id"]), {}).get("reversed")),
                     "auto_flags": sum(1 for x in rr if x.get("flag")), "read": sum(1 for x in rr if (m, x["id"]) in rain)}
        res["systems"][m] = S
    pa = jl(os.path.join(RUN, "probes", f"{A050}.jsonl"))
    res["a050"] = {"probe_fail": [x["id"] for x in pa if x.get("forbidden_hit")],
                   "rain_right": sum(1 for x in pa if x.get("set") == "zmd_rainfall" and rain.get((A050, x["id"]), {}).get("right")),
                   "neg_bar": A050_BAR}
    # ---- the rule, per alpha
    by_alpha = {}
    for al, (s1, s2) in ALPHA.items():
        if s1 not in res["systems"] or s2 not in res["systems"]:
            continue
        seeds = {}
        for sd, m in (("1", s1), ("2", s2)):
            S = res["systems"][m]
            T = S.get("teach", {"better": 0, "worse": 0, "confusable_worse": 0})
            seeds[sd] = {"neg_critical_sweep": S["neg"]["critical"] if S["neg"]["swept"] >= 150 else 999,
                         "neg_critical_regression": len(S["regression_critical"]),
                         "p1": S["p1"], "docs300": S["docs300"][:2], "docs164": S["docs164"][:2],
                         "teacher": [T["better"], T["worse"]], "confusable_worse": T["confusable_worse"],
                         "probe_fail": len(S["probe_fail"]),
                         "rainfall": [S["rain"]["right"], res["a050"]["rain_right"], S["rain"]["reversed"]]}
        by_alpha[al] = {"a050_neg_critical_sweep": A050_BAR, "seeds": seeds}
        v, o = K.decide(by_alpha[al])
        res["per_alpha"][al] = {"verdict": v, "gates": o, "input": by_alpha[al]}
    pick, verdicts = K.pick_alpha(by_alpha) if by_alpha else (None, {})
    res["PICK"] = pick
    res["pick_verdicts"] = verdicts
    res["forced"] = bool(a.force)
    json.dump(res, open(os.path.join(KT, "result.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for al, r in res["per_alpha"].items():
        print(al, r["verdict"])
        for sd, g in r["gates"].items():
            print("  seed", sd, g)
    print("PICK", pick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("regress", "sweep", "teach", "raw", "blind", "judge", "score"))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--pack")
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true", help="score with missing labels (a progress look, never the verdict)")
    a = ap.parse_args()
    os.makedirs(KT, exist_ok=True)
    if a.cmd == "judge" and not a.pack:
        a.pack = a.model
    {"regress": regress, "sweep": sweep, "teach": teach, "raw": raw, "blind": blind, "judge": judge, "score": score}[a.cmd](a)
