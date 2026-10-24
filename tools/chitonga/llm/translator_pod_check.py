"""The 20-line identity check for the translator pod (23 Oct 2026, the Jesuit; Chris: "If the beelink is the bottleneck,
we need to get cheap pods to open up the bottleneck").

Why: six streams share one served translator (translator_service, chitonga-nllb-wise-a050 + term guard + P1). The pod
must serve the SAME thing. Known from the record (LOGBOOK 22 Oct, negation sweep and weakness map): CT2 int8 on a GPU is
not byte-identical to CT2 int8 on the Beelink CPU (56 of 103 and 77 of 153 lines identical): same weights, other
kernels. This check measures that for the service as called, and the pod's throughput.

    python -X utf8 translator_pod_check.py local      Beelink CPU, under beelink_lock; also writes the list of repo
                                                        files the service opened (what the pod tar must carry)
    python -X utf8 translator_pod_check.py remote [threads]   through translate_remote (the pod), N requests in parallel
    python -X utf8 translator_pod_check.py compare    local vs remote, line by line
Lines: diagnosis_500 set, n % 50 == 7 (en>toi, English as typed) and n % 50 == 32 (toi>en, the human Tonga); default
call shape translate(text, direction) (memory on, as the streams call it). Out: runs/2026-10-23_translator_pod/.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
# TPOD_CHECK_DIR (24 Oct, the reader bake-off pod): the run folder of THIS pod, relative to the repo root; the reference
# check_local.json must be made with the same code the pod serves (the served config changed on 24 Oct)
OUT = os.path.join(REPO, os.environ.get("TPOD_CHECK_DIR") or os.path.join("internal", "chitonga_train", "runs",
                                                                         "2026-10-23_translator_pod"))
SET = os.path.join(REPO, "internal", "chitonga_train", "llm", "panel", "diagnosis_500_2026-10-21", "set.jsonl")
sys.path.insert(0, HERE)


def lines():
    rows = [json.loads(l) for l in open(SET, encoding="utf-8") if l.strip()]
    return ([(r["id"], "en>toi", r["en"]) for r in rows if r["n"] % 50 == 7]
            + [(r["id"], "toi>en", r["ref_toi"]) for r in rows if r["n"] % 50 == 32])


def local():
    opened = set()

    def hook(ev, args):
        if ev in ("open", "sqlite3.connect") and args and isinstance(args[0], str):
            p = os.path.normcase(os.path.abspath(args[0]))
            if p.startswith(os.path.normcase(REPO)):
                opened.add(os.path.relpath(p, os.path.normcase(REPO)).replace("\\", "/"))
    sys.addaudithook(hook)
    import beelink_lock
    import translator_service as TS
    os.makedirs(OUT, exist_ok=True)
    with beelink_lock.held("translator-pod-check"):
        s = TS.service()
        res, t0 = [], time.time()
        for i, d, text in lines():
            t = time.time()
            r = s.translate(text, d)
            res.append({"id": i, "dir": d, "src": text, "out": r["text"], "stage": r.get("stage"), "alarms": r.get("alarms"),
                        "t": round(time.time() - t, 1)})
            print(i, d, res[-1]["t"], "s", flush=True)
        wall = time.time() - t0
    json.dump({"device": "beelink-cpu", "wall_s": round(wall, 1), "rows": res}, open(os.path.join(OUT, "check_local.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)
    mods = sorted({os.path.relpath(os.path.normcase(m.__file__), os.path.normcase(REPO)).replace("\\", "/")
                   for m in list(sys.modules.values()) if getattr(m, "__file__", None)
                   and os.path.normcase(m.__file__).startswith(os.path.normcase(REPO))})
    open(os.path.join(OUT, "deps.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(sorted(opened | set(mods))) + "\n")
    print(f"local: {len(res)} lines in {wall:.0f} s = {len(res) / wall * 60:.1f} lines/min; deps {len(opened | set(mods))}")


def remote(threads=4):
    import translate_remote as TR
    from concurrent.futures import ThreadPoolExecutor
    L = lines()
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        # keep=True (24 Oct): the call shape of reader_bakeoff.py freeze, and the pieces' candidate pools travel back, so a
        # line that differs from the Beelink can be explained (a near tie flipped by GPU kernels, or a lever that fired)
        outs = list(ex.map(lambda x: (x, time.time(), TR.translate(x[2], x[1], keep=True, allow_local=False)), L))
    wall = time.time() - t0
    res = [{"id": i, "dir": d, "src": text, "out": r["text"], "stage": r.get("stage"), "alarms": r.get("alarms"),
            "t": r.get("t"), "served_by": r.get("served_by"), "tm_near": r.get("tm_near"),
            "term_guard": bool(r.get("term_guard")), "pieces": r.get("pieces")} for (i, d, text), _, r in outs]
    json.dump({"device": "pod", "threads": threads, "wall_s": round(wall, 1), "rows": res},
              open(os.path.join(OUT, f"check_remote_t{threads}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"remote: {len(res)} lines, {threads} in parallel, {wall:.0f} s = {len(res) / wall * 60:.1f} lines/min")


def compare(name="check_remote_t4.json"):
    a = {(r["id"], r["dir"]): r for r in json.load(open(os.path.join(OUT, "check_local.json"), encoding="utf-8"))["rows"]}
    b = {(r["id"], r["dir"]): r for r in json.load(open(os.path.join(OUT, name), encoding="utf-8"))["rows"]}
    same = 0
    for k in a:
        if a[k]["out"] == b[k]["out"]:
            same += 1
        else:
            print(f"DIFF {k[0]} {k[1]}\n  src   {a[k]['src'][:160]}\n  cpu   {a[k]['out'][:160]}\n  pod   {b[k]['out'][:160]}")
    print(f"identical {same} / {len(a)}")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "local":
        local()
    elif c == "remote":
        remote(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        compare(*(sys.argv[2:3]))
