"""Blind packs, the judge and the score of the stage 19 kill test (24 Oct 2026, the Jesuit). Called through
final19_run.py (blind / judge <pack> / score). Method: killtest/METHOD_NOTES.md (written before any output).

Packs (letters shuffled per item by a seeded generator; keys written here, opened only by score()):
  p1                 every distinct text of a P1 / El Nino item (all systems, paths, modes, a050 included) labelled
                     right / partly / wrong in ONE item (stage 18 prompt p1.md); no carried labels (lesson 2)
  docs_<path>_<mode> per slice: a050 (untagged, same path) + the candidates' distinct texts, ranked (docs.md)
  teach_<path>       the frozen 44, candidate <cls> vs a050, A/B with the teacher's line (teach.md)
  rand_<path>        the random 40, candidate <cls> vs a050, A/B, no reference (rand.md); served = the gate
  abl                the tag ablation, RAW, same model <cls> vs untagged: the 44, the random 40, the dev slice (abl.md)
  neg                the negation sheet (sweep 150 + regression 11, raw and served, every mode): ok / reversal / drop /
                     unclear / false_alarm per distinct text (neg.md); every reversal / drop / unclear is then read by
                     hand (neg_confirm.jsonl) and the hand reading decides
Hand sheets: rain_sheet.txt (rainfall categories, read by hand into rain_read.jsonl).
"""
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time

import final19_run as F

KT, DEC, A050, CANDS = F.KT, F.DEC, F.A050, F.CANDS
SYSALL = [A050] + CANDS
LET = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["A" + chr(c) for c in range(ord("A"), ord("Z") + 1)]
JUDGE_MODEL = "claude-opus-5-5"
BATCH = {"p1": 10, "docs": 25, "teach": 40, "rand": 40, "abl": 40, "neg": 12}
PROMPT = {"p1": "p1.md", "docs": "docs.md", "teach": "teach.md", "rand": "rand.md", "abl": "abl.md", "neg": "neg.md"}
ASK = ("Read prompt.md in this folder and do exactly what it says. items.jsonl holds {n} items; judge every item. "
       "Write the answer to judged.jsonl in this folder (JSON Lines, one line per item, in the order of items.jsonl). "
       "When the file is complete, reply with the single line: DONE <number of lines written>.")
RANK = {"right": 2, "partly": 1, "wrong": 0}
LOCK = threading.Lock()
jl, wl, sha = F.jl, F.wl, F.sha


def kind(pack):
    return pack.split("_")[0]


def items():
    return {(x["set"], x["iid"]): x for x in jl(os.path.join(KT, "items.jsonl"))}


def outs(system, st, path, mode):
    return {x["iid"]: x["out"] for x in jl(os.path.join(DEC, system, f"{st}.{path}.{mode}.jsonl"))}


def rnd_for(s):
    return random.Random(int(hashlib.sha256(("f19:" + s).encode()).hexdigest()[:12], 16) + 1924)


def lettered(key_id, texts_by):
    """texts_by {label: text} -> ({letter: text}, {letter: [labels]}), distinct texts, seeded order"""
    texts = sorted(set(texts_by.values()))
    rnd_for(key_id).shuffle(texts)
    return ({LET[i]: t for i, t in enumerate(texts)},
            {LET[i]: sorted(k for k, v in texts_by.items() if v == texts[i]) for i in range(len(texts))})


def write_pack(name, pack, key):
    d = os.path.join(KT, "packs", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "batches"))
    wl(os.path.join(d, "items.jsonl"), pack)
    n = BATCH[kind(name)]
    for i in range(0, len(pack), n):
        wl(os.path.join(d, "batches", f"b{i // n:02d}", "items.jsonl"), pack[i:i + n])
    json.dump(key, open(os.path.join(KT, f"KEY_{name}_do_not_open.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    print(f"{name}: {len(pack)} items to judge", flush=True)


def modes(system, ms):
    return F.modes_for(system, ms)


# --------------------------------------------------------------------------------------------- blind
def blind(a):
    import final18_run as F18
    import negation_sweep as N
    IT = items()
    want = set((a.pack or "p1,docs,teach,rand,abl,neg,rain").split(","))
    os.makedirs(os.path.join(KT, "packs"), exist_ok=True)
    lex = F18.lexicon_copy()                                     # writes the stage 18 killtest's packs/lexicon.tsv
    shutil.copy2(lex, os.path.join(KT, "packs", "lexicon.tsv"))
    plan = {st: (ms, paths) for st, ms, paths in F.PLAN}
    # ---- P1: every distinct text of every slice in one item
    pack, key = [], {}
    if "p1" not in want:
        IT_p1 = {}
    else:
        IT_p1 = IT
    ms, paths = plan["p1"]
    O = {(s, p, m): outs(s, "p1", p, m) for s in SYSALL for p in paths for m in modes(s, ms)} if IT_p1 else {}
    for (st, iid), x in IT_p1.items():
        if st != "p1":
            continue
        by = {f"{s}|{p}|{m}": O[(s, p, m)][iid] for (s, p, m) in O}
        lt, k = lettered("p1:" + iid, by)
        key[iid] = k
        pack.append({"item": iid, "set": x["sub"], "dir": x["dir"], "src": x["src"], "ref": x.get("ref"), "candidates": lt})
    if IT_p1:
        write_pack("p1", pack, key)
    # ---- documents: one ranking pack per slice
    ms, paths = plan["docs"]
    for p in (paths if "docs" in want else ()):
        for m in (F.SERVED_MODES.get("docs", ms) if p == "served" else ms):
            pack, key, one = [], {}, 0
            O = {s: outs(s, "docs", p, "none" if s == A050 else m) for s in SYSALL}
            for (st, iid), x in IT.items():
                if st != "docs" or not x.get("sub"):
                    continue
                by = {s: O[s][iid] for s in SYSALL}
                lt, k = lettered(f"docs:{p}:{m}:{iid}", by)
                key[iid] = k
                if len(lt) == 1:
                    one += 1
                    continue
                pack.append({"item": iid, "doc": x["doc"].split(":")[1], "dir": x["dir"], "src": x["src"], "ref": x.get("ref"),
                             "candidates": lt})
            write_pack(f"docs_{p}_{m}", pack, key)
    # ---- A/B packs: teach_<path>, rand_<path> (candidate <cls> vs a050), abl (same model, raw, <cls> vs none)
    def ab(name, rows, pairs, ref_field):
        pack, key, n_id = [], {}, 0
        for (iid, x) in rows:
            seen = {}
            for tag, ta, tb in pairs(iid):                 # tag = (system, what); ta = the "cand" text, tb = the base text
                if ta == tb:
                    key.setdefault(f"{iid}:{tag[1]}:id", {"identical": True, "systems": []})["systems"].append(tag[0])
                    n_id += 1
                    continue
                pk = (ta, tb)
                if pk in seen:
                    key[seen[pk]]["systems"].append(tag[0])
                    continue
                item = f"{iid}:{tag[1]}:{len(seen)}"
                seen[pk] = item
                flip = rnd_for(name + item).random() < 0.5
                key[item] = {"A": "cand" if flip else "base", "B": "base" if flip else "cand", "systems": [tag[0]], "what": tag[1]}
                e = {"item": item, "dir": x["dir"], "source": x["src"], "A": ta if flip else tb, "B": tb if flip else ta}
                if ref_field:
                    e["teacher_reference"] = x.get("ref")
                pack.append(e)
        write_pack(name, pack, key)
        print(f"   {name}: {n_id} identical", flush=True)
    for p in (("served", "raw") if "teach" in want else ()):
        base = outs(A050, "teach", p, "none")
        C = {s: outs(s, "teach", p, "cls") for s in CANDS}
        rows = [(iid, x) for (st, iid), x in IT.items() if st == "teach"]
        ab(f"teach_{p}", rows, lambda iid: [((s, "cls"), C[s][iid], base[iid]) for s in CANDS], True)
    for p in (("served", "raw") if "rand" in want else ()):
        base = outs(A050, "rand40", p, "none")
        C = {s: outs(s, "rand40", p, "cls") for s in CANDS}
        rows = [(iid, x) for (st, iid), x in IT.items() if st == "rand40"]
        ab(f"rand_{p}", rows, lambda iid: [((s, "cls"), C[s][iid], base[iid]) for s in CANDS], False)
    rows, pairsd = [], {}
    for st in (("teach", "rand40", "dev") if "abl" in want else ()):
        Cc = {s: outs(s, st, "raw", "cls") for s in CANDS}
        Cn = {s: outs(s, st, "raw", "none") for s in CANDS}
        for (s2, iid), x in IT.items():
            if s2 == st:
                rows.append((f"{st}|{iid}", x))
                pairsd[f"{st}|{iid}"] = [((s, "cls_vs_none"), Cc[s][iid], Cn[s][iid]) for s in CANDS]
    if "abl" in want:
        ab("abl", rows, lambda k: pairsd[k], True)
    # ---- negation sheet: sweep 150 + regression 11, every path and mode, every distinct text
    nrows = {r["iid"]: r for r in N.items()}
    pack, key, auto = [], {}, {}
    for st in (("regress", "sweep") if "neg" in want else ()):
        ms, paths = plan[st]
        O = {(s, p, m): outs(s, st, p, m) for s in SYSALL for p in paths for m in modes(s, ms)}
        for (s2, iid), x in IT.items():
            if s2 != st:
                continue
            r = nrows.get(iid) or {"iid": iid, "dir": x["dir"], "src": x["src"], "ref": x.get("ref")}
            by = {f"{s}|{p}|{m}": O[(s, p, m)][iid] for (s, p, m) in O}
            fl = {k: N.auto_flags(r, t) for k, t in by.items()}
            auto[iid] = fl
            differ = len(set(by.values())) > 1
            if st == "sweep" and not any(fl.values()) and not (any(N.src_hits_check(r)) and differ):
                continue
            # 24 Oct 18:50: two regression lines (poster 13, zema 0047) are ALSO sweep lines; the first pack used the bare
            # iid as the item id, so the two copies collided. Item ids are now '<set>|<iid>'; letters keep their seed
            # ('neg:' + iid) so every earlier judgement of a non-colliding item still fits; the regression copy of a
            # colliding line gets its own seed and both copies are judged again
            dup = ("regress", iid) in IT and ("sweep", iid) in IT
            lt, k = lettered(("negreg:" if (dup and st == "regress") else "neg:") + iid, by)
            key[f"{st}|{iid}"] = k
            pack.append({"item": f"{st}|{iid}", "dir": x["dir"], "src": x["src"], "ref": x.get("ref"),
                         "flags": sorted({f for v in fl.values() for f in v}), "candidates": lt})
    if "neg" in want:
        write_pack("neg", pack, key)
        json.dump(auto, open(os.path.join(KT, "neg_auto_flags.json"), "w", encoding="utf-8"), indent=0)
        print(f"   neg: {sum(len(p['candidates']) for p in pack)} texts on the sheet", flush=True)
    if "rain" not in want:
        return
    # ---- rainfall hand sheet
    ms, paths = plan["rain"]
    O = {(s, p, m): outs(s, "rain", p, m) for s in SYSALL for p in paths for m in modes(s, ms)}
    lines, key = [], {}
    for (st, iid), x in IT.items():
        if st != "rain":
            continue
        by = {f"{s}|{p}|{m}": O[(s, p, m)][iid] for (s, p, m) in O}
        lt, k = lettered("rain:" + iid, by)
        key[iid] = k
        lines.append(f"## {iid}\n   EN : {x['src']}\n   ZMD: {x['ref']}")
        lines += [f"   {L}: {t}" for L, t in lt.items()]
    open(os.path.join(KT, "rain_sheet.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(key, open(os.path.join(KT, "KEY_rain_do_not_open.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    print(f"rain sheet: {len(key)} lines, {sum(len(v) for v in key.values())} texts -> rain_sheet.txt", flush=True)


# --------------------------------------------------------------------------------------------- judge
def valid(name, it, ans):
    a = ans.get(it["item"])
    if not a:
        return False
    k = kind(name)
    if k == "p1":
        lab = a.get("labels", {})
        return set(it["candidates"]) <= set(lab) and all((lab[L] or {}).get("label") in RANK for L in it["candidates"])
    if k == "neg":
        lab = a.get("labels", {})
        return set(it["candidates"]) <= set(lab) and all((lab[L] or {}).get("label") in
                                                         ("ok", "reversal", "drop", "unclear", "false_alarm") for L in it["candidates"])
    if k == "docs":
        s = str(a.get("rank", "")).replace(" ", "")
        got = [L for grp in s.split(">") for L in grp.split("=") if L]
        return sorted(got) == sorted(it["candidates"]) and len(got) == len(set(got))
    return a.get("v") in ("A", "B", "tie")


def run_batch(name, b, todo, dest, done, scratch):
    for attempt in (1, 2, 3):
        todo = [it for it in todo if it["item"] not in done]
        if not todo:
            return b, "complete"
        wd = os.path.join(scratch, f"{name}_{b}_t{attempt}_{int(time.time())}")
        os.makedirs(wd)
        wl(os.path.join(wd, "items.jsonl"), todo)
        shutil.copy2(os.path.join(KT, "prompts", PROMPT[kind(name)]), os.path.join(wd, "prompt.md"))
        shutil.copy2(os.path.join(KT, "packs", "lexicon.tsv"), wd)
        t0 = time.time()
        try:
            r = subprocess.run(["claude", "-p", ASK.format(n=len(todo)), "--model", JUDGE_MODEL, "--permission-mode",
                                "acceptEdits", "--disallowedTools", "Bash,WebFetch,WebSearch,Task", "--output-format", "json"],
                               cwd=wd, capture_output=True, text=True, encoding="utf-8", timeout=5400, shell=False)
            rc, so, se = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            rc, so, se = -9, "", "timeout"
        meta = {"pack": name, "batch": b, "attempt": attempt, "n": len(todo), "secs": round(time.time() - t0), "rc": rc}
        try:
            d = json.loads(so)
            meta.update({k: d.get(k) for k in ("is_error", "num_turns", "total_cost_usd")}, models=list(d.get("modelUsage", {})),
                        result=str(d.get("result"))[:200])
        except ValueError:
            meta["stdout_tail"] = (so + se)[-600:]
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
    names = a.pack.split(",")
    scratch = os.environ.get("JUDGE_SCRATCH", os.path.join(os.environ.get("TEMP", "/tmp"), "claude_judge_f19"))
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(os.path.join(KT, "judged"), exist_ok=True)
    jobs = []
    for name in names:
        dest = os.path.join(KT, "judged", f"{name}.jsonl")
        done = {x["item"] for x in jl(dest)}
        bdir = os.path.join(KT, "packs", name, "batches")
        for b in sorted(os.listdir(bdir)):
            its = jl(os.path.join(bdir, b, "items.jsonl"))
            if all(it["item"] in done for it in its):
                continue
            jobs.append((name, b, its, dest, done))
        print(name, "batches to run:", sum(1 for j in jobs if j[0] == name), "| already judged:", len(done), flush=True)
    with cf.ThreadPoolExecutor(a.par) as ex:
        for b, st in ex.map(lambda j: run_batch(j[0], j[1], j[2], j[3], j[4], scratch), jobs):
            print("BATCH", b, st, flush=True)
    print("JUDGE DONE", names, flush=True)


# --------------------------------------------------------------------------------------------- score
def parse_rank(s):
    lv = {}
    for i, grp in enumerate(str(s).replace(" ", "").split(">")):
        for L in grp.split("="):
            if L:
                lv[L] = i
    return lv


def need(cond, msg, force):
    if not cond and not force:
        raise SystemExit(msg)


def load_key(name):
    return json.load(open(os.path.join(KT, f"KEY_{name}_do_not_open.json"), encoding="utf-8"))


def judged(name):
    return {x["item"]: x for x in jl(os.path.join(KT, "judged", f"{name}.jsonl"))}


def ab_count(name, force, sel=None):
    """A/B pack -> {system: {better, worse, tie, identical, better_items, worse_items}} (cand vs base)"""
    key, jud = load_key(name), judged(name)
    miss = [i for i, k in key.items() if not k.get("identical") and i not in jud]
    need(not miss, f"{name}: {len(miss)} pairs unjudged, e.g. {miss[:3]}", force)
    res = {}
    for item, k in key.items():
        base = item.rsplit(":", 2)[0]
        if sel and not sel(base):
            continue
        for s in k["systems"]:
            S = res.setdefault(s, {"better": 0, "worse": 0, "tie": 0, "identical": 0, "better_items": [], "worse_items": []})
            if k.get("identical"):
                S["identical"] += 1
                continue
            j = jud.get(item)
            if not j:
                continue
            if j["v"] == "tie":
                S["tie"] += 1
            elif k[j["v"]] == "cand":
                S["better"] += 1
                S["better_items"].append(base)
            else:
                S["worse"] += 1
                S["worse_items"].append(base)
    return res


def score(a):
    import final18_killtest as K
    import final18_run as F18
    IT = items()
    force = a.force
    res = {"files": {}, "systems": {s: {} for s in SYSALL}, "verdicts": {}}
    for f in sorted(os.listdir(os.path.join(KT, "judged"))):
        if f.endswith(".jsonl") and f != "calls.jsonl":
            res["files"][f] = sha(os.path.join(KT, "judged", f))[:16]
    plan = {st: (ms, paths) for st, ms, paths in F.PLAN}
    labels = lambda s, st: [f"{s}|{p}|{m}" for p in plan[st][1] for m in modes(s, plan[st][0])]
    # ---- negation: judge as finder, my hand read decides every critical / unclear mark
    nkey, njud = load_key("neg"), judged("neg")
    conf = {(x["iid"], x["L"]): x for x in jl(os.path.join(KT, "neg_confirm.jsonl"))}
    need(all(i in njud for i in nkey), "neg: sheet items unjudged", force)
    final, unconf = {}, []
    for iid, k in nkey.items():          # iid here = the pack item id '<set>|<iid>'
        for L in k:
            jlab = ((njud.get(iid) or {}).get("labels", {}).get(L) or {}).get("label")
            if jlab in ("reversal", "drop", "unclear"):
                if (iid, L) in conf:
                    final[(iid, L)] = conf[(iid, L)]["label"]
                else:
                    unconf.append((iid, L))
                    final[(iid, L)] = jlab
            else:
                final[(iid, L)] = conf[(iid, L)]["label"] if (iid, L) in conf else (jlab or "ok")
    need(not unconf, f"neg: {len(unconf)} judge-critical texts not yet read by hand, e.g. {unconf[:3]}", force)
    res["neg_unconfirmed"] = len(unconf)
    crit = {}                           # (label, set) -> [iids critical]
    for item, k in nkey.items():
        st, iid = item.split("|", 1)
        for L, labs in k.items():
            if final[(item, L)] in ("reversal", "drop"):
                for lab in labs:
                    crit.setdefault((lab, st), []).append(iid)
    for s in SYSALL:
        res["systems"][s]["neg_sweep"] = {lab.split("|", 1)[1]: len(crit.get((lab, "sweep"), [])) for lab in labels(s, "sweep")}
        res["systems"][s]["neg_sweep_items"] = {lab.split("|", 1)[1]: sorted(crit.get((lab, "sweep"), [])) for lab in labels(s, "sweep")}
        res["systems"][s]["neg_regress"] = {lab.split("|", 1)[1]: sorted(crit.get((lab, "regress"), [])) for lab in labels(s, "regress")}
    # ---- P1 absolute labels (every slice in one item)
    pkey, pjud = load_key("p1"), judged("p1")
    need(all(i in pjud for i in pkey), "p1: items unjudged", force)
    lab = {}
    for iid, k in pkey.items():
        got = (pjud.get(iid) or {}).get("labels", {})
        if not all(L in got for L in k):
            continue
        lab[iid] = {x: RANK[got[L]["label"]] for L, xs in k.items() for x in xs}
    for s in CANDS:
        P = {}
        for p in plan["p1"][1]:
            for m in plan["p1"][0]:
                for sub in ("p1", "climate"):
                    b = w = 0
                    for iid, l in lab.items():
                        if IT[("p1", iid)]["sub"] != sub:
                            continue
                        c, r = l[f"{s}|{p}|{m}"], l[f"{A050}|{p}|none"]
                        b += c > r
                        w += c < r
                    P[f"{sub}|{p}|{m}"] = [b, w]
            b = w = 0
            for iid, l in lab.items():
                if IT[("p1", iid)]["sub"] == "p1":
                    b += l[f"{s}|{p}|gen"] > l[f"{s}|{p}|none"]
                    w += l[f"{s}|{p}|gen"] < l[f"{s}|{p}|none"]
            P[f"gen_vs_none|{p}"] = [b, w]
        res["systems"][s]["p1"] = P
    for s in SYSALL:
        res["systems"][s]["p1_mean"] = {l2.split("|", 1)[1]: round(sum(l[l2] for iid, l in lab.items() if IT[("p1", iid)]["sub"] == "p1")
                                                                  / max(1, sum(1 for iid in lab if IT[("p1", iid)]["sub"] == "p1")), 3)
                                        for l2 in labels(s, "p1")}
    # ---- documents per slice
    for p in plan["docs"][1]:
        for m in (F.SERVED_MODES.get("docs", plan["docs"][0]) if p == "served" else plan["docs"][0]):
            name = f"docs_{p}_{m}"
            dkey, djud = load_key(name), judged(name)
            miss = [i for i, k in dkey.items() if len(k) > 1 and i not in djud]
            need(not miss, f"{name}: {len(miss)} unranked", force)
            for s in CANDS:
                for sub in ("docs300", "docs164"):
                    b = w = t = idn = 0
                    for iid, k in dkey.items():
                        if IT[("docs", iid)]["sub"] != sub:
                            continue
                        lm = next(L for L, xs in k.items() if s in xs)
                        lr = next(L for L, xs in k.items() if A050 in xs)
                        if lm == lr:
                            idn += 1
                            continue
                        if iid not in djud:
                            continue
                        lv = parse_rank(djud[iid]["rank"])
                        b += lv[lm] < lv[lr]
                        w += lv[lm] > lv[lr]
                        t += lv[lm] == lv[lr]
                    res["systems"][s].setdefault("docs", {})[f"{sub}|{p}|{m}"] = [b, w, t, idn]
    # ---- the 44 (gate 3), the random 40 (gate 6), the ablation (gate 7)
    wg = json.load(open(os.path.join(KT, "word_guard.json"), encoding="utf-8"))
    leak44 = set(wg["test_lines_with_word"].get("teach", {}))
    t01 = {x["iid"] for x in IT.values() if x["set"] == "teach" and x.get("teacher") == "T01"}
    conf_items = {x["iid"] for x in IT.values() if x["set"] == "teach" and F18.CONFUSABLE.search((x["src"] or "") + " " + (x["ref"] or ""))}
    for p in ("served", "raw"):
        for sub, sel in (("all", None), ("T01", lambda b: b in t01), ("T09", lambda b: b not in t01),
                         ("no_word_leak", lambda b: b not in leak44), ("word_leak", lambda b: b in leak44),
                         ("en>toi", lambda b: b.endswith("en>toi")), ("toi>en", lambda b: b.endswith("toi>en"))):
            for s, S in ab_count(f"teach_{p}", force, sel).items():
                if sub == "all":
                    S["confusable_worse"] = sum(1 for b in S["worse_items"] if b in conf_items)
                res["systems"][s].setdefault("teach", {})[f"{p}|{sub}"] = S
        for s, S in ab_count(f"rand_{p}", force).items():
            res["systems"][s].setdefault("rand40", {})[p] = S
    for st in ("teach", "rand40", "dev"):
        for s, S in ab_count("abl", force, lambda b, st=st: b.startswith(st + "|")).items():
            res["systems"][s].setdefault("ablation_raw_cls_vs_none", {})[st] = {k: S[k] for k in ("better", "worse", "tie", "identical")}
    # ---- probes (deterministic, the prefixed matcher; paired with a050 on the same probe and path)
    probes = {x["iid"]: x for x in IT.values() if x["set"] == "probes"}
    ms, paths = plan["probes"]
    hits = {}
    for s in SYSALL:
        for p in paths:
            for m in modes(s, ms):
                O = outs(s, "probes", p, m)
                for iid, x in probes.items():
                    h = F.forbidden_hits(O[iid], x["forbid"])
                    if h:
                        hits[(s, p, m, iid)] = h
    for s in SYSALL:
        res["systems"][s]["probe_hits"] = {f"{p}|{m}|{iid}": h for (s2, p, m, iid), h in hits.items() if s2 == s}
    # ---- rainfall (hand read; paired)
    rkey = load_key("rain")
    rread = {(x["iid"], x["L"]): x["v"] for x in jl(os.path.join(KT, "rain_read.jsonl"))}
    need(all((i, L) in rread for i, k in rkey.items() for L in k), "rain: texts unread", force)
    rv = {}
    for iid, k in rkey.items():
        for L, labs in k.items():
            for l2 in labs:
                rv[(l2, iid)] = rread.get((iid, L))
    for s in SYSALL:
        R = {}
        for l2 in labels(s, "rain"):
            R[l2.split("|", 1)[1]] = {"right": sum(1 for iid in rkey if rv.get((l2, iid)) == "r"),
                                      "reversed": sorted(iid for iid in rkey if rv.get((l2, iid)) == "R")}
        res["systems"][s]["rain"] = R
    # ---- the gates, per (path, gate-2 mode), through final18_killtest.decide() (pre-registered, unchanged)
    A = res["systems"][A050]

    def gate_input(s, p, m):
        S = res["systems"][s]
        reg_bad = set()
        for mm in modes(s, plan["regress"][0]):
            reg_bad |= set(S["neg_regress"][f"{p}|{mm}"]) - set(A["neg_regress"][f"{p}|none"])
        pf = 0
        for mm in modes(s, plan["probes"][0]):
            for iid in probes:
                pf += len(set(hits.get((s, p, mm, iid), [])) - set(hits.get((A050, p, "none", iid), [])))
        r_c, r_a = S["rain"][f"{p}|gen"], A["rain"][f"{p}|none"]
        T = S["teach"][f"{p}|all"]
        dm = m if p == "raw" else F.SERVED_MODES.get("docs", (m,))[0]     # served documents exist in <gen> only
        return {"neg_critical_sweep": S["neg_sweep"][f"{p}|{m}"], "neg_critical_regression": len(reg_bad),
                "regression_lines_worse": sorted(reg_bad),
                "p1": S["p1"][f"p1|{p}|{m}"], "docs300": S["docs"][f"docs300|{p}|{dm}"][:2],
                "docs164": S["docs"][f"docs164|{p}|{dm}"][:2], "teacher": [T["better"], T["worse"]],
                "confusable_worse": T.get("confusable_worse", 0), "probe_fail": pf,
                "rainfall": [r_c["right"], r_a["right"], len(set(r_c["reversed"]) - set(r_a["reversed"]))]}

    def rand_ok(s):
        R = res["systems"][s]["rand40"]["served"]
        return R["better"] >= R["worse"], R["better"] >= 1.5 * R["worse"] and R["better"] + R["worse"] >= 25 and R["better"] > 0

    units = {"B13": F.ARMS["B13"], "B33": F.ARMS["B33"], "P1": ("P1",)}
    for u, systems in units.items():
        V = {}
        for p in ("served", "raw"):
            for m in ("none", "gen"):
                r = {"a050_neg_critical_sweep": A["neg_sweep"][f"{p}|none"],
                     "seeds": {str(i + 1): gate_input(s, p, m) for i, s in enumerate(systems)}}
                v, o = K.decide(r)
                V[f"{p}|{m}"] = {"verdict": v, "gates": o, "input": r}
        g6 = {s: rand_ok(s) for s in systems}
        allpass = all(x["verdict"].startswith("PASS") for x in V.values()) and all(ok for ok, _ in g6.values())
        res["verdicts"][u] = {"by_path_mode": V, "gate6_no_worse": {s: g[0] for s, g in g6.items()},
                              "gate6_claim": {s: g[1] for s, g in g6.items()},
                              "VERDICT": "PASS ON [L] (speakers, gate 8, still to judge before any install)" if allpass else "FAIL"}
    res["forced"] = bool(force)
    json.dump(res, open(os.path.join(KT, "result.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for u, r in res["verdicts"].items():
        print(u, r["VERDICT"])
        for pm, x in r["by_path_mode"].items():
            print("  ", pm, x["verdict"][:60], json.dumps(x["gates"], ensure_ascii=False)[:500])
        print("   gate 6 (random 40, served, no worse):", r["gate6_no_worse"], "claim:", r["gate6_claim"])


# --------------------------------------------------------------------------------------------- report tables
def short(iid):
    if "poster" in iid:
        return "poster" + iid.rsplit(":", 1)[1]
    if "0047" in iid:
        return "zema0047"
    if "before_eating" in iid:
        return "before-eating"
    return iid


def report(a):
    """markdown tables from killtest/result.json for KILLTEST_RESULT.md (numbers only; the words are written by hand)"""
    R = json.load(open(os.path.join(KT, "result.json"), encoding="utf-8"))
    S = R["systems"]
    out = []
    bw = lambda x: f"{x[0]}:{x[1]}"
    for p in ("served", "raw"):
        out.append(f"\n### {p.upper()}\n")
        out.append("| gate | " + " | ".join(CANDS) + " | a050 |")
        out.append("|---|" + "---|" * (len(CANDS) + 1))
        A = S[A050]
        for m in ("none", "gen"):
            out.append(f"| 1 sweep critical ({m}) | " + " | ".join(str(S[s]["neg_sweep"][f"{p}|{m}"]) for s in CANDS) + f" | {A['neg_sweep'][f'{p}|none']} |")
        for m in ("none", "cls"):
            out.append(f"| 1 regression lines critical ({m}) | " + " | ".join(",".join(map(short, S[s]["neg_regress"][f"{p}|{m}"])) or "0" for s in CANDS)
                       + " | " + (",".join(map(short, A["neg_regress"][f"{p}|none"])) or "0") + " |")
        for m in ("none", "gen"):
            out.append(f"| 2 P1 174 b:w ({m}) | " + " | ".join(bw(S[s]["p1"][f"p1|{p}|{m}"]) for s in CANDS) + " | |")
            out.append(f"| (El Nino 48, {m}) | " + " | ".join(bw(S[s]["p1"][f"climate|{p}|{m}"]) for s in CANDS) + " | |")
        for m in (("none", "gen") if p == "raw" else ("gen",)):
            for sub in ("docs300", "docs164"):
                out.append(f"| 2 {sub} b:w:tie:id ({m}) | " + " | ".join(":".join(map(str, S[s]["docs"][f"{sub}|{p}|{m}"])) for s in CANDS) + " | |")
        Z = {"better": 0, "worse": 0, "tie": 0, "identical": 0}
        for sub in ("all", "T01", "T09", "no_word_leak", "word_leak", "en>toi", "toi>en"):
            out.append(f"| 3 frozen 44 <cls> b:w (tie, id) {sub} | " + " | ".join(
                "{better}:{worse} ({tie}, {identical})".format(**S[s]["teach"].get(f"{p}|{sub}", Z)) for s in CANDS) + " | |")
        out.append("| 3 confusable worse | " + " | ".join(str(S[s]["teach"][f"{p}|all"].get("confusable_worse", 0)) for s in CANDS) + " | |")
        for m in ("none", "cls", "gen"):
            out.append(f"| 4 probe forbidden hits ({m}) | " + " | ".join(
                ", ".join(f"{k.split('|')[2]}:{'/'.join(v)}" for k, v in S[s]["probe_hits"].items() if k.startswith(f"{p}|{m}|")) or "0" for s in CANDS)
                + " | " + (", ".join(f"{k.split('|')[2]}:{'/'.join(v)}" for k, v in A["probe_hits"].items() if k.startswith(f"{p}|none|")) or "0") + " |")
        for m in ("gen", "none"):
            out.append(f"| 5 rainfall right / reversed ({m}) | " + " | ".join(
                f"{S[s]['rain'][f'{p}|{m}']['right']} / {len(S[s]['rain'][f'{p}|{m}']['reversed'])}" for s in CANDS)
                + f" | {A['rain'][f'{p}|none']['right']} / {len(A['rain'][f'{p}|none']['reversed'])} |")
        out.append("| 6 random 40 <cls> b:w (tie, id) | " + " | ".join(
            f"{S[s]['rand40'][p]['better']}:{S[s]['rand40'][p]['worse']} ({S[s]['rand40'][p]['tie']}, {S[s]['rand40'][p]['identical']})" for s in CANDS) + " | |")
    out.append("\n### Tag ablation (RAW, <cls> vs untagged of the same model, b:w (tie, identical))\n")
    out.append("| set | " + " | ".join(CANDS) + " |")
    out.append("|---|" + "---|" * len(CANDS))
    for st in ("teach", "rand40", "dev"):
        out.append(f"| {st} | " + " | ".join(
            "{better}:{worse} ({tie}, {identical})".format(**S[s]["ablation_raw_cls_vs_none"][st]) for s in CANDS) + " |")
    for p in ("served", "raw"):
        out.append(f"| P1 174 <gen> vs untagged ({p}) | " + " | ".join(bw(S[s]["p1"][f"gen_vs_none|{p}"]) for s in CANDS) + " |")
    out.append("\n### Verdicts\n")
    for u, v in R["verdicts"].items():
        out.append(f"- **{u}: {v['VERDICT']}**")
        for pm, x in v["by_path_mode"].items():
            out.append(f"  - {pm}: {x['verdict'][:70]} | " + "; ".join(f"seed {k}: {', '.join(g)}" for k, g in x["gates"].items()))
        out.append(f"  - gate 6 no worse: {v['gate6_no_worse']}; claim 1.5:1: {v['gate6_claim']}")
    txt = "\n".join(out)
    open(os.path.join(KT, "tables.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)
