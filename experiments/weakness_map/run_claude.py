"""Claude as the blind judge of a diagnosis pack, batch by batch (22 Oct 2026, the Jesuit; coordinator for Chris: "Judge
blind with CLAUDE (not Codex: weekly limit reached) ... headless Claude Code CLI calls (`claude -p`) from an empty folder,
in batches of 50").

Why this shape: the pack (make_pack.py) is the same format and prompt as the 21 Oct stage 15b map, so the labels are
comparable. Each batch is copied into a NEW EMPTY folder outside the repo (items, the batch's engine readings, the full
lexicon, prompt.md); the CLI runs there with edits auto-accepted inside that folder only, Bash / web / sub-agents off. It
never sees the key (KEY_<pack>_do_not_open.json lives in the diagnosis folder; only score.py opens it). Resumable: an item
already in judged/<pack>__claude.jsonl is skipped; missing items are re-sent once in a fresh folder.
Claude's labels are evaluations, never training data.
    python -X utf8 run_claude.py <pack> [--par 4] [--only b00,b01]
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
DIAG = os.path.join(REPO, "internal", "chitonga_train", "llm", "panel", "diagnosis_500_2026-10-21")
SCRATCH = os.environ.get("JUDGE_SCRATCH", os.path.join(os.environ.get("TEMP", "/tmp"), "claude_judge_diag500"))
RAW = os.path.join(HERE, "claude_raw")
MODEL = "claude-opus-5-5"
ASK = ("Read prompt.md in this folder and do exactly what it says. items.jsonl holds {n} items; judge every candidate "
       "letter of every item. Write the answer to judged.jsonl in this folder (JSON Lines, one line per item, in the "
       "order of items.jsonl). When the file is complete, reply with the single line: DONE <number of lines written>.")
LOCK = threading.Lock()


def jl(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip().strip("`")
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def ok(it, ans):
    lab = ans.get(it["item"], {}).get("labels", {})
    return set(it["candidates"]) <= set(lab) and all(lab[L].get("label") in ("right", "partly", "wrong")
                                                     for L in it["candidates"])


def run_batch(pack, b, items, dest, done):
    pdir = os.path.join(DIAG, "pack", pack)
    todo = [it for it in items if it["item"] not in done]
    for attempt in (1, 2):
        if not todo:
            return b, "complete"
        wd = os.path.join(SCRATCH, f"{pack}_{b}_t{attempt}_{int(time.time())}")
        os.makedirs(wd)
        with open(os.path.join(wd, "items.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for it in todo:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        shutil.copy2(os.path.join(pdir, "batches", b, "engine_readings.txt"), wd)
        shutil.copy2(os.path.join(pdir, "lexicon.tsv"), wd)
        shutil.copy2(os.path.join(pdir, "prompt.md"), wd)
        t0 = time.time()
        r = subprocess.run(["claude", "-p", ASK.format(n=len(todo)), "--model", MODEL, "--permission-mode", "acceptEdits",
                            "--disallowedTools", "Bash,WebFetch,WebSearch,Task", "--output-format", "json"],
                           cwd=wd, capture_output=True, text=True, encoding="utf-8", timeout=5400, shell=False)
        meta = {"pack": pack, "batch": b, "attempt": attempt, "n": len(todo), "secs": round(time.time() - t0),
                "rc": r.returncode}
        try:
            d = json.loads(r.stdout)
            meta.update({k: d.get(k) for k in ("is_error", "num_turns", "total_cost_usd", "result")},
                        models=list(d.get("modelUsage", {})))
        except ValueError:
            meta["stdout_tail"] = (r.stdout + r.stderr)[-800:]
        ans = {}
        jp = os.path.join(wd, "judged.jsonl")
        if os.path.exists(jp):
            shutil.copy2(jp, os.path.join(RAW, f"{pack}_{b}_t{attempt}.jsonl"))
            ans = {x["item"]: x for x in jl(jp) if "item" in x}
        got = [it for it in todo if ok(it, ans)]
        meta["complete"] = len(got)
        with LOCK:
            with open(dest, "a", encoding="utf-8", newline="\n") as f:
                for it in got:
                    f.write(json.dumps({"item": it["item"], "labels": ans[it["item"]]["labels"], "judge": MODEL + " (claude -p)",
                                        "batch": b, "attempt": attempt}, ensure_ascii=False) + "\n")
                    done.add(it["item"])
            with open(os.path.join(RAW, "calls.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(pack, b, f"attempt {attempt}: {len(got)}/{len(todo)} in {meta['secs']}s", flush=True)
        todo = [it for it in todo if it["item"] not in done]
    return b, "complete" if not todo else f"MISSING {len(todo)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(DIAG, "judged", f"{a.pack}__claude.jsonl")
    done = {x["item"] for x in jl(dest)} if os.path.exists(dest) else set()
    bdir = os.path.join(DIAG, "pack", a.pack, "batches")
    jobs = []
    for b in sorted(os.listdir(bdir)):
        if a.only and b not in a.only.split(","):
            continue
        items = jl(os.path.join(bdir, b, "items.jsonl"))
        if all(it["item"] in done for it in items):
            continue
        jobs.append((b, items))
    with cf.ThreadPoolExecutor(a.par) as ex:
        for b, st in ex.map(lambda j: run_batch(a.pack, j[0], j[1], dest, done), jobs):
            print("BATCH", b, st, flush=True)
    print("ALL", a.pack, len(done), flush=True)


if __name__ == "__main__":
    main()
