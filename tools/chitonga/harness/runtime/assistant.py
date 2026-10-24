"""The process inside the "Zamai assistant" console window (24 Oct 2026, the Jesuit).

    python -X utf8 tools/chitonga/harness/runtime/assistant.py [--poll 15]

Normally started by launch_assistant.cmd -> start_assistant.ps1, never by hand from a VS Code terminal (a child of
VS Code can die with it; the launcher goes through its own console). What it does, in order:
  1. writes its start line to pilot_logs/runtime.log (the launcher and the watchdog wait for proof of life);
  2. takes the OS lock pilot_logs/assistant.lock, or refuses: one copy only, ever;
  3. if a shadow loop started some other way is alive (shadow.pid, e.g. `run.py shadow-start`), TAKES IT OVER:
     writes SHADOW_EXIT, waits up to 120 s for it to finish the message in hand and exit, kills it only after that.
     Two pollers must never run at once: both would read the same mark and process every message twice;
  4. removes a stale SHADOW_EXIT (else the new loop would stop at its first poll), writes shadow.pid = its own pid;
  5. runs adapter/run.py `shadow --poll N [--send-tester]` IN THIS PROCESS (runpy), so the lock holder, shadow.pid and the heartbeat
     writer are one process; stdout and stderr go to the console AND to pilot_logs/assistant_console.log (the widget
     reads its last error there).
THE MODE (changed 24 Oct 2026 07:30, Chris via the main session: "just do it live, whats your problme"; PILOT_GO.json
scope T00 tester only). The runtime starts exactly what the main session started by hand:
    ZAMAI_ASSISTANT_LIVE=1  run.py shadow --poll 15 --send-tester
but ONLY while PILOT_GO.json holds Chris's words and no PILOT_STOP exists (config.Config.go / stopped); otherwise
plain `shadow --poll 15` (nothing sent). Consequence, said plainly: under the runtime the env key follows the go file,
so the two keys that remain are Chris's go file and PILOT_STOP. run.py still re-checks all three before EVERY send,
and --send-tester only ever sends to numbers flagged tester (T00, Chris); teachers stay in the outbox until their
consent is recorded. Removing PILOT_GO.json (or creating PILOT_STOP) and pressing STOP then START drops it to
shadow; PILOT_STOP alone already stops every send at once. The chosen mode goes to pilot_logs/runtime_mode.json for
the widget. There is no teacher-live path here on purpose (`run.py live` is never started by the runtime).
"""
import argparse
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402


class Tee:
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        try:
            self.stream.write(s)
        except (UnicodeEncodeError, OSError, ValueError, AttributeError):
            pass
        self.f.write(s)
        return len(s)

    def flush(self):
        for x in (self.stream, self.f):
            try:
                x.flush()
            except (OSError, ValueError, AttributeError):
                pass

    def isatty(self):
        return False


def takeover(p, pid, wait_s=120, alive=C.pid_alive, kill=C.kill_pid):
    open(p.shadow_exit, "w").write("takeover by the runtime")
    C.log_line(p.runtime_log, f"taking over from pid {pid}: SHADOW_EXIT written")
    print(f"{C.stamp()} a shadow loop is already running (pid {pid}, not the runtime): asking it to exit ...",
          flush=True)
    for _ in range(wait_s):
        if not alive(pid):
            C.log_line(p.runtime_log, f"pid {pid} exited cleanly")
            return "clean"
        time.sleep(1)
    kill(pid)
    C.log_line(p.runtime_log, f"pid {pid} did not exit in {wait_s} s: killed")
    return "killed"


def choose_mode(cfg):
    """('live_tester', reason) when Chris's go file is present and no PILOT_STOP, else ('shadow', reason)"""
    if cfg.stopped():
        return "shadow", "PILOT_STOP present"
    g = cfg.go()
    if not g:
        return "shadow", "no PILOT_GO.json with Chris's words"
    return "live_tester", f"PILOT_GO.json {g.get('date')}: {g.get('chris_words')!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=15)
    a = ap.parse_args()
    p = C.Paths()
    me = os.getpid()
    C.log_line(p.runtime_log, f"assistant.py start pid {me}")
    lock = C.Lock(p.lock)
    if not lock.acquire():
        C.log_line(p.runtime_log, f"pid {me} REFUSED: another runtime holds {p.lock}")
        print("REFUSED: the Zamai assistant is already running (lock held). This window closes.", flush=True)
        sys.exit(3)
    try:
        old = C.read_pid(p)
        if old and old != me and C.pid_alive(old):
            how = takeover(p, old)
            print(f"{C.stamp()} took over from pid {old} ({how})", flush=True)
        for f in (p.shadow_exit,):
            if os.path.exists(f):
                os.remove(f)
        open(p.pid, "w").write(str(me))
        sys.stdout = Tee(sys.__stdout__, p.console_log)
        sys.stderr = Tee(sys.__stderr__, p.console_log)
        mode, why = choose_mode(p.cfg)
        sys.argv = [C.RUN_PY, "shadow", "--poll", str(a.poll)]
        if mode == "live_tester":
            os.environ["ZAMAI_ASSISTANT_LIVE"] = "1"
            sys.argv.append("--send-tester")
        else:
            os.environ.pop("ZAMAI_ASSISTANT_LIVE", None)
        json.dump({"mode": mode, "why": why, "pid": me, "at": C.stamp(), "argv": sys.argv[1:]},
                  open(p.mode_file, "w", encoding="utf-8"), ensure_ascii=False)
        label = "LIVE (tester only)" if mode == "live_tester" else "shadow (nothing sent)"
        print(f"{C.stamp()} Zamai assistant runtime pid {me}: {label} [{why}], polling every {a.poll} s. "
              f"Close this window = the watchdog brings it back within 10 min; STOP in the widget keeps it down.",
              flush=True)
        C.log_line(p.runtime_log, f"pid {me} running run.py {' '.join(sys.argv[1:])} ({label})")
        try:
            runpy.run_path(C.RUN_PY, run_name="__main__")
            C.log_line(p.runtime_log, f"pid {me} exited (shadow loop returned)")
        except KeyboardInterrupt:
            C.log_line(p.runtime_log, f"pid {me} exited (Ctrl+C)")
        except SystemExit as e:
            C.log_line(p.runtime_log, f"pid {me} exited (SystemExit {e.code})")
            raise
        except BaseException as e:  # noqa: BLE001
            C.log_line(p.runtime_log, f"pid {me} CRASHED: {type(e).__name__}: {e}")
            raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
