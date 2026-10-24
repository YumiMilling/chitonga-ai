"""Run measure.py under the Beelink lock as 'tm-near' (24 Oct 2026, the Jesuit; translator optimisation step 1).

ONE heavy Beelink job at a time (tools/chitonga/llm/beelink_lock.py): this waits for whoever holds the lock (today the
stage 18 kill test, then the chat v14 kill test is queued), takes it, runs the sets, and releases it even if a set fails.

    python -X utf8 internal/chitonga_train/runs/2026-10-24_tm_near/run_measure.py [sets ...]
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "llm"))
sys.path.insert(0, HERE)
import beelink_lock as L  # noqa: E402

SETS = sys.argv[1:] or ["probes", "maths", "p1", "dlines"]

print(f"waiting for the Beelink lock ... ({time.strftime('%H:%M:%S')})", flush=True)
with L.held("tm-near"):
    print(f"lock held as 'tm-near' {time.strftime('%H:%M:%S')}", flush=True)
    import measure
    for w in SETS:
        t0 = time.time()
        try:
            measure.run(w)
        except Exception:                                    # noqa: BLE001  one set must not cost the others
            traceback.print_exc()
        print(f"[{w}] {round(time.time() - t0)}s", flush=True)
print(f"lock released {time.strftime('%H:%M:%S')}", flush=True)
