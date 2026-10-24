"""The EXECUTOR of the stage 19 kill test (24 Oct 2026, the Jesuit). The rule is RUN_PACKAGE.md section 4 of
runs/2026-10-24_stage19_data (written before any stage 19 model existed); how it is executed, and every change of method
made before any output, is in killtest/METHOD_NOTES.md (written 15:40, before a single candidate line was decoded).
This file produces the numbers; final18_killtest.decide() applies the rule.

    python -X utf8 tools/chitonga/train/final19_run.py build                 # killtest/items.jsonl + pins.json (local)
    python -X utf8 tools/chitonga/train/final19_run.py selftest              # the probe matcher, the word guard, the plan
    python -X utf8 tools/chitonga/train/final19_run.py decode <system>       # one system, every set, raw + served (pod)
    python -X utf8 tools/chitonga/train/final19_run.py podall --par 3        # every system, N processes (pod)
    python -X utf8 tools/chitonga/train/final19_run.py consistency           # a050 on the pod vs the stored Beelink outputs
    python -X utf8 tools/chitonga/train/final19_run.py blind | judge <pack> | score     (local, after the decodes)

WHY these shapes (each a precedent of the record):
- One system per PROCESS: translator_service.SOUP is a module global (negation_sweep.make_service). The raw decode reuses
  the service's own soup reader (one copy of the weights per process).
- RAW = CTranslate2 beam 4, max 256, batches of 16 in file order per direction (final18_run.raw / teach), so a050's
  stored raw outputs are comparable line by line. The tag is prefixed as the trainer and f19_sel.py do: "<cls> " + text.
- SERVED = translator_service with serving_tags.patch (the pod's copy only): register -> the soup reader only; memory off
  (so tm_near off); term guard and glossary_prefer on; the piece-level fast path of negation_sweep (exact, 22 Oct).
- Every output row: {set, iid, dir, src, ref, system, path, mode, out, alarms?, route?}; files dec/<system>/<set>.<path>.<mode>.jsonl,
  resumable (served rows are appended and flushed one by one).
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
RUN = os.path.join(CT, "runs", "2026-10-24_stage19_data")
RUN18 = os.path.join(CT, "runs", "2026-10-23_final_translator_data")
KT18 = os.path.join(RUN18, "killtest")
KT = os.path.join(RUN, "killtest")
DEC = os.path.join(KT, "dec")
MODELS = os.environ.get("CHITONGA_MODELS", r"models")
A050 = "wise-a050"
# system name -> the model folder (= the HF repo name; the int8 twin is <folder>-ct2-int8)
SYSTEMS = {A050: "chitonga-nllb-wise-a050",
           "B13-s1": "chitonga-nllb-s19-B13-s1", "B13-s2": "chitonga-nllb-s19-B13-s2",
           "P1": "chitonga-nllb33-s19-P1", "B33-s1": "chitonga-nllb33-s19-B33-s1", "B33-s2": "chitonga-nllb33-s19-B33-s2"}
CANDS = ["B13-s1", "B13-s2", "B33-s1", "B33-s2", "P1"]
ARMS = {"B13": ("B13-s1", "B13-s2"), "B33": ("B33-s1", "B33-s2")}
POD_ORDER = ["P1", "B33-s1", "B33-s2", A050, "B13-s1", "B13-s2"]      # the slow 3.3B first
TAGS = {"none": None, "cls": "<cls>", "gen": "<gen>"}
LANG = {"en>toi": ("eng_Latn", "toi_Latn"), "toi>en": ("toi_Latn", "eng_Latn")}
# set -> (candidates' modes, paths); a050 is decoded untagged only. Order = the pod's order (gates first, documents last)
PLAN = [("regress", ("none", "cls"), ("served", "raw")),
        ("probes", ("none", "cls", "gen"), ("served", "raw")),
        ("rain", ("gen", "none"), ("served", "raw")),
        ("teach", ("cls", "none", "gen"), ("served", "raw")),
        ("rand40", ("cls", "none", "gen"), ("served", "raw")),
        ("dev", ("none", "cls", "gen"), ("raw",)),
        ("sweep", ("none", "gen"), ("served", "raw")),
        ("p1", ("none", "gen"), ("served", "raw")),
        ("docs", ("none", "gen"), ("raw", "served"))]
POSTER = "e2t:bk:poster_moh_who_cholera_pairs:"
REG_IIDS = [POSTER + str(i) for i in (0, 1, 7, 8, 10, 13, 14)] + ["ce2t:06", "ce2t:17"]
FOCUS = "test:zema:3rd_draft_sessd_eis_livingstone_bernt_checked_sw:zema:3rd_draft_sessd_eis_livingstone_bernt_checked_sw:0047:en>toi"
HARNESS_LINE = {"iid": "harness:before_eating:en>toi", "dir": "en>toi", "src": "Wash your hands with soap before eating.", "ref": None}
ZMD = ["d500:115", "d500:118", "d500:119", "d500:120", "d500:121", "d500:122", "d500:123", "d500:124", "d500:125",
       "d500:127", "d500:128", "d500:139", "d500:140", "d500:141", "d500:142"]
# 24 Oct 16:20, before any document output existed (METHOD_NOTES, dated addendum): the served path is CPU-bound on the
# pod (~10-15 s a document line, one core per process), so served documents are decoded in the mode the service will
# pass on the general page (<gen>, RUN_PACKAGE 2 point 9) and not untagged; raw documents keep both modes
SERVED_MODES = {"docs": ("gen",)}
RAND40_SHA = "a37a7af316f8e7abefdf83ffe93b27f632f67bc4ce378383c6e44415f2d800e1"
# lesson 3 (stage 18 KILLTEST_RESULT 5): a forbidden word also counts after a nominal / locative / infinitive prefix
PREFIX = r"(?:a|ka|ko|ku|mu|i|li|ma|zi|ba|ci)?"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def jl_resume(p):
    """the rows of a decode file that parse; a half-written last line (a killed process) is cut off the file"""
    if not os.path.exists(p):
        return []
    good, raw = [], open(p, encoding="utf-8").read().splitlines()
    for l in raw:
        try:
            good.append(json.loads(l))
        except ValueError:
            break
    if len(good) != len([l for l in raw if l.strip()]):
        wl(p, good)
    return good


def wl(p, rows, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def word_re(w):
    """the matcher of lessons 3 and 4: the word alone or after one prefix, never inside a longer word"""
    return re.compile("(?<![a-z'])" + PREFIX + re.escape(w.strip().lower()) + "(?![a-z])", re.I)


def forbidden_hits(text, forbid):
    low = (text or "").lower()
    return [w for w in forbid if word_re(w).search(low)]


# --------------------------------------------------------------------------------------------- build (local)
def build(a):
    """killtest/items.jsonl: every line any set decodes, built from the frozen files (shas asserted)"""
    import final18_run as F
    import negation_sweep as N
    items = []
    nrows = {r["iid"]: r for r in N.items()}
    for iid in REG_IIDS + [FOCUS]:
        r = nrows[iid]
        items.append({"set": "regress", "iid": iid, "dir": r["dir"], "src": r["src"], "ref": r.get("ref")})
    items.append({"set": "regress", **HARNESS_LINE})
    for iid in F.sweep_iids():
        r = nrows[iid]
        items.append({"set": "sweep", "iid": iid, "dir": r["dir"], "src": r["src"], "ref": r.get("ref")})
    for x in F.p1_items():
        items.append({"set": "p1", "iid": x["iid"], "sub": "p1" if x.get("set") == "p1" else "climate", "dir": x["dir"],
                      "src": x["src"], "ref": x.get("ref")})
    s300, s164 = F.doc_sets()
    for x in F.doc_items():
        items.append({"set": "docs", "iid": x["iid"], "doc": x["doc"], "dir": x["dir"], "src": x["src"], "ref": x.get("ref"),
                      "sub": "docs300" if x["iid"] in s300 else "docs164" if x["iid"] in s164 else None})
    for r in F.frozen44():
        for d, src, ref in (("en>toi", r["en"], r["toi"]), ("toi>en", r["toi"], r["en"])):
            items.append({"set": "teach", "iid": f"{r['id']}:{d}", "line": r["id"], "dir": d, "src": src, "ref": ref,
                          "teacher": r["teacher"], "alternatives": r.get("alternatives")})
    for p in jl(os.path.join(RUN, "confusable_probes_final18.jsonl")):
        items.append({"set": "probes", "iid": p["id"], "dir": "en>toi", "src": p["en"], "ref": None, "forbid": p["forbid"],
                      "expect": p["expect"]})
    diag = {x["id"]: x for x in jl(os.path.join(CT, "llm", "panel", "diagnosis_500_2026-10-21", "set.jsonl"))}
    for i in ZMD:
        items.append({"set": "rain", "iid": i, "dir": "en>toi", "src": diag[i]["en"], "ref": diag[i]["ref_toi"]})
    p40 = os.path.join(RUN, "random_classroom_40.jsonl")
    assert sha(p40) == RAND40_SHA, "random_classroom_40.jsonl changed: refusing"
    for x in jl(p40):
        items.append({"set": "rand40", "iid": x["id"], "dir": "en>toi", "src": x["en"], "ref": None, "subject": x["subject"]})
    for x in jl(os.path.join(RUN, "teacher_dev_stage19.jsonl")):
        for d, src, ref in (("en>toi", x["en"], x["toi"]), ("toi>en", x["toi"], x["en"])):
            items.append({"set": "dev", "iid": f"{x['id']}:{d}", "dir": d, "src": src, "ref": ref})
    cnt = {}
    for it in items:
        cnt[it["set"]] = cnt.get(it["set"], 0) + 1
    want = {"regress": 11, "sweep": 150, "p1": 222, "docs": 687, "teach": 88, "probes": 13, "rain": 15, "rand40": 40, "dev": 72}
    assert cnt == want, cnt
    assert len({(i["set"], i["iid"]) for i in items}) == len(items)
    p = os.path.join(KT, "items.jsonl")
    wl(p, items)
    print("items", cnt, "sha256", sha(p)[:16])


# --------------------------------------------------------------------------------------------- decode (pod)
def service(system):
    import negation_sweep as N
    import translator_service as TS
    assert hasattr(TS, "SERVE_TAGS"), "translator_service.py is not the tag-patched copy (serving_tags.patch)"
    TS.SOUP = os.path.join(MODELS, SYSTEMS[system])
    full = TS.Service({"memory": False})
    fast = N._fast_class()({"memory": False})
    fast._r = full._r
    return TS, full, fast


def serve_config(TS, full):
    files = {}
    for mod in ("translator_service", "term_guard", "h3_select", "h2_reader", "negation_sweep", "h2_text", "names"):
        try:
            m = sys.modules.get(mod) or __import__(mod)
            files[mod + ".py"] = sha(m.__file__)
        except Exception as e:  # noqa: BLE001
            files[mod + ".py"] = f"unavailable: {e!r}"[:80]
    try:
        import term_guard as TG
        files["glossary_terms.json"] = sha(TG.GLOSSARY)
    except Exception as e:  # noqa: BLE001
        files["glossary_terms.json"] = f"unavailable: {e!r}"[:80]
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in full.cfg.items() if isinstance(v, (str, int, float, bool, tuple, list, type(None)))}
    return {"files": files, "cfg": cfg, "tm_near_effective": bool(cfg.get("tm_near") and cfg.get("memory")),
            "readers": list(full.cfg.get("readers", [])), "CT2_DEVICE": os.environ.get("CT2_DEVICE", "cpu"),
            "CT2_COMPUTE": os.environ.get("CT2_COMPUTE", "int8")}


def file_lock(p):
    """an exclusive lock on <file>.lock held for the life of the process (Linux pods); None elsewhere. Lets an extra
    process be started for a queued job (pods 4/5, 17:00) without two writers on one file."""
    try:
        import fcntl
    except ImportError:
        return None
    f = open(p + ".lock", "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    _LOCKS.append(f)
    return f


_LOCKS = []


def raw_run(r, texts, d, tag):
    tok = r.tok
    src, tgt = LANG[d]
    tok.src_lang = src
    pre = (tag + " ") if tag else ""
    res = r.tr.translate_batch([tok.convert_ids_to_tokens(tok.encode(pre + t)) for t in texts],
                               target_prefix=[[tgt]] * len(texts), beam_size=4, max_decoding_length=256)
    return [tok.decode(tok.convert_tokens_to_ids(x.hypotheses[0][1:]), skip_special_tokens=True) for x in res]


def parse_spec(text):
    """which (set, path, sub) a process decodes; several pods share the sets (24 Oct 16:30: the served path is CPU-bound,
    one core per process, so the work is spread over pods by SET; every pod decodes a050 AND every candidate for its
    sets, so each comparison stays on one machine). Tokens: 'set' | 'set:path' | 'set:path:sub', comma or space
    separated; '' = everything. -> None or {set: [(path|None, sub|None)]}"""
    toks = [t for t in re.split(r"[,\s]+", text or "") if t]
    if not toks:
        return None
    out = {}
    for t in toks:
        parts = t.split(":")
        sub = parts[2] if len(parts) > 2 else None
        if sub and len(parts) > 3:                       # 'docs:served:docs300:0/2' = every 2nd row from row 0
            sub = sub + "@" + parts[3]
        out.setdefault(parts[0], []).append((parts[1] if len(parts) > 1 else None, sub))
    return out


def modes_for(system, modes):
    return ("none",) if system == A050 else modes


def decode(a):
    system = a.model
    assert system in SYSTEMS, system
    items = jl(os.path.join(KT, "items.jsonl"))
    by = {}
    for it in items:
        by.setdefault(it["set"], []).append(it)
    TS, full, fast = service(system)
    out_dir = os.path.join(DEC, system)
    os.makedirs(out_dir, exist_ok=True)
    rd = full.reader("soup")
    assert os.path.normcase(os.path.abspath(rd.dir)) == os.path.normcase(os.path.abspath(TS.SOUP)), (rd.dir, TS.SOUP)
    cfg = serve_config(TS, full)
    cfg["system"], cfg["soup"] = system, TS.SOUP
    json.dump(cfg, open(os.path.join(out_dir, "serve_config.json"), "w", encoding="utf-8"), indent=1)
    t_all = time.time()
    sj = os.path.join(KT, "SKIP_JOBS")
    if os.path.exists(sj) and f"{system}@{a.sets}" in open(sj).read().split():
        # 24 Oct 17:10: this queued job's rows are decoded by shard processes started beside podall; the job holds its
        # slot (no model loaded) until every shard waiter / shard decode on this pod has ended, so the pod is not
        # packed and stopped while a shard still runs
        print(f"{system}@{a.sets}: in SKIP_JOBS (decoded by shard processes); waiting for them to end", flush=True)
        while True:
            r = subprocess.run(["pgrep", "-f", r"wait_shard\.sh|--sets [a-z0-9:,]*:[0-9]+/[0-9]+"], capture_output=True, text=True)
            if not r.stdout.strip():
                break
            time.sleep(30)
        print(f"{system}@{a.sets}: shards ended", flush=True)
        return
    spec = parse_spec(a.sets or (open(os.path.join(KT, "POD_SETS")).read() if os.path.exists(os.path.join(KT, "POD_SETS")) else ""))
    print(f"{system}: sets {spec or 'all'}", flush=True)
    for st, modes, paths in PLAN:
        rows = by[st]
        if spec is not None and st not in spec:
            continue
        for path in paths:
            if spec is not None and not any(pp in (None, path) for pp, _ in spec[st]):
                continue
            subs = {sb for pp, sb in (spec or {}).get(st, []) if pp == path and sb}
            for mode in modes_for(system, SERVED_MODES.get(st, modes) if path == "served" else modes):
                # a sub-slice (e.g. docs300) gets its own file: two processes never append to one file; merge() joins them
                p = os.path.join(out_dir, f"{st}.{path}.{mode}" + (("." + "+".join(sorted(re.sub(r"[^A-Za-z0-9]+", "_", x) for x in subs)))
                                                                    if subs else "") + ".jsonl")
                tag = TAGS[mode]
                t0 = time.time()
                if path == "raw":
                    if os.path.exists(p) and len(jl_resume(p)) == len(rows):
                        continue
                    out = {}
                    for d in ("en>toi", "toi>en"):
                        sub = [x for x in rows if x["dir"] == d]
                        for i in range(0, len(sub), 16):
                            b = sub[i:i + 16]
                            for x, o in zip(b, raw_run(rd, [x["src"] for x in b], d, tag)):
                                out[x["iid"]] = o
                    wl(p, [{"set": st, "iid": x["iid"], "dir": x["dir"], "system": system, "path": "raw", "mode": mode,
                            "out": out[x["iid"]]} for x in rows])
                    print(f"{system} {st}.raw.{mode}: {len(rows)} in {time.time() - t0:.0f}s", flush=True)
                    continue
                todo_rows = [x for x in rows if not (st == "docs" and not x.get("sub"))]   # served docs: the judged 524
                if subs:
                    keep = []
                    for sb in subs:
                        name, _, shard = sb.partition("@")
                        rows_sb = list(todo_rows) if name == "all" else [x for x in todo_rows if x.get("sub") == name]
                        if shard:
                            k, n = (int(v) for v in shard.split("/"))
                            rows_sb = rows_sb[k::n]
                        keep += rows_sb
                    todo_rows = keep
                lk = file_lock(p)        # 24 Oct 17:00: a second process for the same file waits, then resumes
                done = {x["iid"] for x in jl_resume(p)}
                todo = [x for x in todo_rows if x["iid"] not in done]
                if not todo:
                    continue
                with open(p, "a", encoding="utf-8", newline="\n") as f:
                    for i, x in enumerate(todo):
                        t = time.time()
                        n0, m0 = getattr(fast, "n_full", 0), len(getattr(fast, "mismatch", []))
                        o = fast.translate(x["src"], x["dir"], register=tag)
                        nf, mm = getattr(fast, "n_full", 0) - n0, len(getattr(fast, "mismatch", [])) - m0
                        route = ("fast" if nf == 0 else f"pieces_full:{nf}") + (f" MISMATCH:{mm}" if mm else "")
                        f.write(json.dumps({"set": st, "iid": x["iid"], "dir": x["dir"], "system": system, "path": "served",
                                            "mode": mode, "out": o["text"], "alarms": o.get("alarms", []), "route": route,
                                            "t": round(time.time() - t, 2)}, ensure_ascii=False) + "\n")
                        f.flush()
                        if i % 50 == 0:
                            print(f"{system} {st}.served.{mode}: {len(done) + i + 1}/{len(todo_rows)} {time.time() - t0:.0f}s", flush=True)
                print(f"{system} {st}.served.{mode}: done {len(todo)} in {time.time() - t0:.0f}s", flush=True)
    print(f"DECODE DONE {system} {time.time() - t_all:.0f}s", flush=True)


def podall(a):
    """every system in its own process, a.par at a time; the pod's run script waits for PODALL DONE / PODALL FAILED"""
    os.makedirs(os.path.join(DEC, "logs"), exist_ok=True)
    if a.jobs:
        jobs = [(j.split("@")[0], j.split("@")[1] if "@" in j else "") for j in a.jobs.split(";") if j.strip()]
    else:
        jobs = [(s, a.sets) for s in POD_ORDER if not a.only or s in a.only.split(",")]
    lock = threading.Lock()
    status = {}

    def one(job):
        s, spec = job
        lp = os.path.join(DEC, "logs", f"{s}{('_' + re.sub(r'[^A-Za-z0-9]+', '_', spec)) if spec else ''}.log")
        for attempt in (1, 2):
            with open(lp, "a", encoding="utf-8") as lf:
                r = subprocess.run([sys.executable, "-X", "utf8", __file__, "decode", s] + (["--sets", spec] if spec else []),
                                   stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode == 0:
                break
            with lock:
                print(f"{s}: attempt {attempt} rc={r.returncode}", flush=True)
        with lock:
            status[f"{s}@{spec}"] = r.returncode
            print(f"JOB {s}@{spec} rc={r.returncode} {time.strftime('%H:%M:%S')}", flush=True)

    with cf.ThreadPoolExecutor(a.par) as ex:
        list(ex.map(one, jobs))
    bad = [s for s, rc in status.items() if rc]
    print("PODALL FAILED " + ",".join(bad) if bad else "PODALL DONE", flush=True)


# --------------------------------------------------------------------------------------------- merge (local)
def merge(a):
    """the pods' results into killtest/dec: dec_pod1 (cheap sets), dec_pod2 (sweep, P1), dec_pod3 (documents). A file
    may come from ONE pod only (each set's comparisons stay on one machine), except the documents' served sub-slices,
    which are joined (docs300 + docs164, both decoded for every system on the same pod)."""
    pods = sorted([d for d in os.listdir(KT) if d.startswith("dec_pod")] +
                  [os.path.join(d, "dec") for d in os.listdir(KT) if d.startswith("pod") and os.path.isdir(os.path.join(KT, d, "dec"))])
    seen, slices = {}, {}
    shutil.rmtree(DEC, ignore_errors=True)          # dec/ is derived: rebuilt from the pods' folders every time
    for pd in pods:
        for s in os.listdir(os.path.join(KT, pd)):
            src = os.path.join(KT, pd, s)
            if s == "logs" or not os.path.isdir(src):
                continue
            os.makedirs(os.path.join(DEC, s), exist_ok=True)
            for f in os.listdir(src):
                if not f.endswith(".jsonl"):
                    if f == "serve_config.json":
                        shutil.copy2(os.path.join(src, f), os.path.join(DEC, s, "serve_config." + re.sub(r"[^A-Za-z0-9]+", "_", pd) + ".json"))
                    continue
                parts = f[:-6].split(".")
                target = ".".join(parts[:3]) + ".jsonl"
                k = (s, target)
                if len(parts) == 3:
                    assert k not in seen, f"{s}/{target} from two pods: {seen.get(k)} and {pd}"
                    seen[k] = pd
                    shutil.copy2(os.path.join(src, f), os.path.join(DEC, s, target))
                else:
                    # a sub-slice (docs300 on pod 4, docs164 on pod 5): every system's copy of ONE slice must come from
                    # one pod (checked below); different slices may come from different pods
                    sub = re.sub(r"_[0-9]+_[0-9]+$", "", parts[3])
                    seen.setdefault((s, target, sub), pd)
                    slices.setdefault(sub, set()).add(pd)
                    with open(os.path.join(DEC, s, target), "a", encoding="utf-8", newline=chr(10)) as out:
                        out.write(open(os.path.join(src, f), encoding="utf-8").read())
    bad = {k: v for k, v in slices.items() if len(v) > 1}
    assert not bad, f"a sub-slice decoded on two pods: {bad}"
    for s in os.listdir(DEC):
        if os.path.isdir(os.path.join(DEC, s)) and s != "logs":
            print(s, len(os.listdir(os.path.join(DEC, s))), "files")
    print("sub-slices:", {k: sorted(v) for k, v in slices.items()})
    json.dump({"/".join(k): v for k, v in sorted(seen.items())}, open(os.path.join(KT, "dec_provenance.json"), "w"), indent=0)


# --------------------------------------------------------------------------------------------- consistency (local)
def consistency(a):
    """a050 decoded on the pod against every stored Beelink a050 output of the stage 18 kill test: a REPORT, overrides
    nothing (METHOD_NOTES 2.3)"""
    import final18_run as F
    pod = lambda st, path: {x["iid"]: x["out"] for x in jl(os.path.join(DEC, A050, f"{st}.{path}.none.jsonl"))}
    pairs = []   # (what, iid, beelink, pod)
    raw18 = {x["iid"]: x["out"] for x in jl(os.path.join(KT18, "raw_wise-a050.jsonl"))}
    for st in ("p1", "docs"):
        pr = pod(st, "raw")
        pairs += [(f"{st} raw", i, raw18[i], pr[i]) for i in pr if i in raw18]
    for pref, path in (("teach", "served"), ("teachraw", "raw")):
        stored = {x["iid"] + ":" + x["dir"]: x["out"] for x in jl(os.path.join(KT18, f"{pref}_{A050}.jsonl"))}
        pr = pod("teach", path)
        pairs += [(f"teach {path}", i, stored[i], pr[i]) for i in pr if i in stored]
    stored = {x["iid"]: x["out"] for x in jl(os.path.join(KT18, f"regress_{A050}.jsonl"))}
    pr = pod("regress", "served")
    pairs += [("regress served", i, stored[i], pr[i]) for i in pr if i in stored]
    probes18 = jl(os.path.join(RUN18, "probes", f"{A050}.jsonl"))
    ps, prw, rs = pod("probes", "served"), pod("probes", "raw"), pod("rain", "served")
    for x in probes18:
        if x.get("set") == "zmd_rainfall":
            if x["id"] in rs:
                pairs.append(("rain served", x["id"], x["served"], rs[x["id"]]))
        else:
            pairs.append(("probes served", x["id"], x["served"], ps[x["id"]]))
            if x.get("raw") is not None:
                pairs.append(("probes raw", x["id"], x["raw"], prw[x["id"]]))
    sw = F.a050_sweep_texts()
    pr = pod("sweep", "served")
    pairs += [("sweep served (22 Oct)", i, sw[i], pr[i]) for i in pr if i in sw]
    by, diffs = {}, []
    for what, i, b, p in pairs:
        s = by.setdefault(what, [0, 0])
        s[0] += 1
        s[1] += b == p
        if b != p:
            diffs.append({"what": what, "iid": i, "beelink": b, "pod": p})
    tot = sum(v[0] for v in by.values())
    same = sum(v[1] for v in by.values())
    rep = {"total": tot, "identical": same, "share": round(same / max(1, tot), 4), "by_set": by, "differences": diffs}
    json.dump(rep, open(os.path.join(KT, "a050_consistency.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, (n, s) in by.items():
        print(f"{k:24} {s}/{n} identical")
    print(f"ALL {same}/{tot} identical ({100 * same / max(1, tot):.1f} %); {len(diffs)} differences -> a050_consistency.json")


# --------------------------------------------------------------------------------------------- word guard (local)
def leak_words():
    return [w["toi"] for w in json.load(open(os.path.join(RUN, "leak_scan_19.json"), encoding="utf-8"))["word_level"]]


def word_guard(a=None):
    """lesson 4: the 19 teacher words at word level in every test line; and the stage 19 t_word rows against the test"""
    items = jl(os.path.join(KT, "items.jsonl"))
    words = leak_words()
    res = {"words": words, "test_lines_with_word": {}, "train_bare_rows_in_test": []}
    for it in items:
        text = " ".join(str(it.get(k) or "") for k in ("src", "ref"))
        hit = [w for w in words if word_re(w).search(text.lower())]
        if hit:
            res["test_lines_with_word"].setdefault(it["set"], {})[it["iid"]] = hit
    # the stage 19 training rows that are bare words / short runs (<= 3 words): any of them inside a test line?
    short = set()
    for r in jl(os.path.join(RUN, "train.jsonl")):
        g = str(r.get("group") or r.get("origin") or "")
        t = (r.get("toi") or "").strip()
        if t and len(t.split()) <= 3 and "t_word" in g:          # the teacher-word groups (bare words, red team B7)
            short.add((g, t.lower()))
    test_text = {}
    for it in items:
        if it["set"] in ("teach", "rand40", "probes", "dev"):
            test_text[(it["set"], it["iid"])] = " ".join(str(it.get(k) or "") for k in ("src", "ref")).lower()
    hits = []
    for g, t in short:
        if len(t) < 4:
            continue
        rx = word_re(t)
        for k, txt in test_text.items():
            if rx.search(txt):
                hits.append({"group": g, "toi": t, "set": k[0], "iid": k[1]})
    res["train_bare_rows_in_test"] = hits
    res["n_short_rows_scanned"] = len(short)
    json.dump(res, open(os.path.join(KT, "word_guard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in res["test_lines_with_word"].items()}, "| short train rows inside test lines:", len(hits),
          f"(of {len(short)} short rows)")
    return res


def selftest(a):
    assert forbidden_hits("Bikka janza lyako ankolo waambe", ["nkolo"]) == ["nkolo"]          # stage 18 cp:02 raw
    assert forbidden_hits("Mwana ulacisa nkolo", ["nkolo"]) == ["nkolo"]
    assert forbidden_hits("Mwana ulacisa mukanwa", ["nkolo"]) == []
    assert forbidden_hits("ciinkolo", ["nkolo"]) == []                                         # inside a longer word
    assert forbidden_hits("Muciindi ca coronavirus", ["corona"]) == []
    assert forbidden_hits("ikkolona", ["kkolona"]) == ["kkolona"]
    assert word_re("dulowing'a").search("kodulowing'a mukuzungulukide")
    assert [p[0] for p in PLAN] == ["regress", "probes", "rain", "teach", "rand40", "dev", "sweep", "p1", "docs"]
    assert modes_for(A050, ("cls", "none")) == ("none",)
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--sets", default="")
    ap.add_argument("--jobs", default="")
    ap.add_argument("--pack")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fn = {"build": build, "decode": decode, "podall": podall, "consistency": consistency, "selftest": selftest, "merge": merge,
          "word_guard": word_guard}
    try:
        import final19_judge as J      # blind / judge / score live beside this file (written after the pod started)
        fn.update({"blind": J.blind, "judge": J.judge, "score": J.score, "report": getattr(J, "report", None)})
    except (ImportError, AttributeError):   # 24 Oct 18:16: a pod with an older final19_judge.py must still decode
        pass
    if a.cmd == "judge" and not a.pack:
        a.pack = a.model
    fn[a.cmd](a)
