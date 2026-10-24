"""Decode the 500-line diagnosis set through the SERVED translator (22 Oct 2026, the Jesuit; Chris: "We want to find the
weaknesses of the translation model, which can probably also help with the chat model" ... "If the beelink is busy, use a
pod").

Why a second decoder: the stage 15b map (diagnosis_500_2026-10-21) judged RAW decodes (h2_reader, CT2 int8, beam 4, one
input per line). What users get is translator_service: sentence split, n-best pool, term guard (en>toi), loop guard,
names route, back-translation checker = R. This runs exactly that, with chitonga-nllb-wise-a050 (the installed model,
translator_service.SOUP default since 22 Oct), memory off, through negation_sweep's piece-level fast path (validated
identical to the full service: 42/42 R, 10/10 a050, 36/36 piece-level v2; LOGBOOK 22 Oct negation sweep).

History: started on the Beelink CPU at 16:15 (~22 s / line); at 16:42 another agent's job (gen_a050.py, 5.4 GB) took the
CPU and memory and the rate fell to ~1 line / min, so the run moved to ONE RunPod pod (Chris's rule for a busy Beelink).
The 103 en>toi lines decoded on the CPU are kept (outputs/a050_served_cpu_partial.jsonl) as the GPU-vs-CPU check.

Sources as in decode.py: en>toi = the English AS TYPED (bad-input lines included), toi>en = the human Tonga.
Jobs = every (direction, line); shard k of n takes every n-th job (so the shards mix both directions and lengths evenly).
Output (resumable): <DIAG>/outputs/<label>.shard<k>of<n>.jsonl {"id","dir","out","path","alarms","t"}; merge with
    python -X utf8 decode_served.py merge --label a050_srv_gpu --n 3   (writes <label>.jsonl and <label>__toi2en.jsonl)
    python -X utf8 decode_served.py run --label a050_srv_gpu --shard 0 --n 3 [--dirs "en>toi,toi>en"] [--limit 2]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "tools", "chitonga", "train"), os.path.join(REPO, "tools", "chitonga", "llm")]
DIAG = os.path.join(REPO, "internal", "chitonga_train", "llm", "panel", "diagnosis_500_2026-10-21")
MODEL = "wise-a050"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def run(a):
    import negation_sweep as NS
    rows = jl(os.path.join(DIAG, "set.jsonl"))
    jobs = [(d, r) for d in a.dirs.split(",") for r in rows]
    jobs = [j for k, j in enumerate(jobs) if k % a.n == a.shard]
    p = os.path.join(DIAG, "outputs", f"{a.label}.shard{a.shard}of{a.n}.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    done = {(x["dir"], x["id"]) for x in jl(p)}
    jobs = [(d, r) for d, r in jobs if (d, r["id"]) not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    TS, full, fast = NS.make_service(MODEL)
    assert os.path.basename(TS.SOUP) == "chitonga-nllb-" + MODEL
    t0 = time.time()
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        for k, (d, r) in enumerate(jobs):
            t = time.time()
            out, path, _, alarms = NS.translate(TS, full, fast, r["en"] if d == "en>toi" else r["ref_toi"], d)
            f.write(json.dumps({"id": r["id"], "dir": d, "out": out, "system": a.label, "model": "chitonga-nllb-" + MODEL,
                                "device": os.environ.get("CT2_DEVICE", "cpu"), "path": path, "alarms": alarms,
                                "t": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 20 == 0:
                print(a.label, a.shard, len(done) + k + 1, f"{time.time() - t0:.0f}s", path, flush=True)
    print("done", a.label, a.shard, f"{time.time() - t0:.0f}s", "mismatch", len(getattr(fast, "mismatch", [])), flush=True)


def merge(a):
    got = {}
    for k in range(a.n):
        for x in jl(os.path.join(DIAG, "outputs", f"{a.label}.shard{k}of{a.n}.jsonl")):
            got[(x["dir"], x["id"])] = x
    ids = [r["id"] for r in jl(os.path.join(DIAG, "set.jsonl"))]
    for d, suffix in (("en>toi", ""), ("toi>en", "__toi2en")):
        miss = [i for i in ids if (d, i) not in got]
        if miss:
            print(d, "MISSING", len(miss), miss[:5])
            continue
        with open(os.path.join(DIAG, "outputs", a.label + suffix + ".jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for i in ids:
                f.write(json.dumps(got[(d, i)], ensure_ascii=False) + "\n")
        print(d, "merged", len(ids))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "merge"))
    ap.add_argument("--label", default="a050_srv_gpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--dirs", default="en>toi,toi>en")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": run, "merge": merge}[a.cmd](a)
