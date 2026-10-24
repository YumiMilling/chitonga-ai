"""Gate 0 for the translator pod when the Beelink cannot hold a full local server (24 Oct 2026, the Jesuit; the reader
bake-off pod: the tm-near measurement held the Beelink lock and RAM was at 92-96 %, so gates 1-3 of
TRANSLATOR_POD_PLAN.md, which load the four CT2 models, could not run).

What it still proves, at idle priority and without any model: run FROM AN UNPACKED TAR (an empty folder), every module of
the served path imports, the held-out guard loads (the 23 Oct GuardError), every file the memory and its guards list for
themselves exists inside the tar, the teacher glossary and the term-guard rules compile, the memory's rows build, and
the service's forbid / held / blocked lists build. And it prints the SERVICE CONFIG HASH: sha256 of the resolved
translator_service.CFG plus the sha256 of every .py and data file of the tar (sorted), so the pod's log and the
Beelink's can be compared by one number.

    python -X utf8 -I <unpacked>/zamai-pro/tools/chitonga/train/translator_pod_gate0.py      (from the unpacked tar)
    python -X utf8 tools/chitonga/train/translator_pod_gate0.py --hash-only                   (the repo: hash only)
What it does NOT prove: that the models load and translate (the pod server's own memory-on / memory-off probes, and
the 20-line identity check, cover that).
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "tools", "chitonga", "llm"), HERE]


def low_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception as e:  # noqa: BLE001
        print("priority not lowered:", repr(e)[:80])


def cfg_hash():
    import translator_service as TS
    cfg = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in TS.CFG.items()}, sort_keys=True, default=str)
    return hashlib.sha256(cfg.encode("utf-8")).hexdigest()[:16], TS.CFG


def code_hash(names):
    """sha256 over (path, sha256 of the file) for the given repo-relative paths, sorted"""
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode("utf-8") + b"\0" + hashlib.sha256(open(os.path.join(REPO, n), "rb").read()).hexdigest().encode() + b"\n")
    return h.hexdigest()[:16]


def main():
    low_priority()
    t0 = time.time()
    ch, cfg = cfg_hash()
    print("CFG", json.dumps({k: v for k, v in cfg.items() if k != "set_by"}, default=str))
    print("SERVICE CFG HASH", ch)
    if "--manifest" in sys.argv:          # the tar's file list (from tar -tzf), to hash the same files in the repo
        names = [l.strip()[len("zamai-pro/"):] for l in open(sys.argv[sys.argv.index("--manifest") + 1], encoding="utf-8")
                 if l.strip().startswith("zamai-pro/") and not l.strip().endswith("/")]
        print("TAR CODE+DATA HASH", code_hash(names), f"({len(names)} files)")
    if "--hash-only" in sys.argv:
        return
    import h2_memory as M
    G = M._guard()
    print("guard loaded:", type(G).__name__)
    srcs = M.sources()
    miss = [f for f in srcs if not os.path.isfile(f)]
    print("memory sources:", len(srcs), "missing", len(miss))
    assert not miss, miss[:5]
    for f in M.FROZEN18:
        assert os.path.isfile(f), f"frozen set missing: {f}"
    import term_guard as TG
    gl = TG.glossary_terms()
    print("glossary terms:", len(gl), "| term-guard rules:", len(TG.RULES))
    rows = M._rows()
    print("memory rows:", len(rows[0]), "| LESSONS", M.LESSONS)
    held = M.serve_held()
    print("serve_held built:", type(held).__name__)
    import translator_service as TS
    s = TS.Service()
    print("forbidden:", len(s.forbidden()), "| held:", len(s.held()))
    print("blocked('Wash your hands with soap.'):", s.blocked("Wash your hands with soap."))
    print(f"GATE0 PASS {time.time() - t0:.0f}s from {REPO}")


if __name__ == "__main__":
    main()
