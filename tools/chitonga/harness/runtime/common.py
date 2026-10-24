"""The always-on runtime of the Zamai teacher assistant: shared paths, the single-instance lock, liveness, the
watchdog decision and the status record (24 Oct 2026, the Jesuit).

WHY THIS EXISTS. Chris, 24 Oct 2026: "for the assistant we need to build something like the ERP that runs
outside VS Code in a separate window." Until today the shadow loop (adapter/run.py shadow) was a background process
an agent started; nothing restarted it after a crash or a reboot, and nothing showed whether it was alive. This
package copies an earlier responder's proven setup (not in this repo: launch_responder.cmd ->
restart_responder.ps1, watchdog_hidden.pyw every 10 min via pythonw, the line lock) for the assistant:

  launch_assistant.cmd -> start_assistant.ps1 -> a console window "Zamai assistant" running assistant.py,
  which holds the lock and runs `run.py shadow --poll 15 [--send-tester]` in-process (so the lock holder, shadow.pid and the
  heartbeat writer are one and the same process);
  watchdog.pyw (scheduled task, every 10 min) relaunches through explorer.exe when the heartbeat is stale or the
  process is dead, never while RUNTIME_STOP exists;
  widget.py, a local page on 127.0.0.1 with the status and a STOP / START button.

WHAT IT SENDS. The runtime only ever starts `run.py shadow`: replies to teachers go to pilot_logs/outbox.jsonl. Since
24 Oct 07:30 (Chris: "just do it live"; PILOT_GO.json scope T00) it adds --send-tester while the go file holds his
words and no PILOT_STOP exists (assistant.choose_mode): real replies to the tester number only. The per-send
three-key gate in config.py is untouched and still checked before every send.

TWO STOP FILES, ON PURPOSE:
  PILOT_STOP   (the adapter's kill switch, Chris's): the loop keeps RUNNING and logging but answers nothing and
               routes every message to the desk. The runtime never writes or removes it.
  RUNTIME_STOP (this package's): the process is stopped and the watchdog does NOT bring it back. Written by the
               widget's STOP button (or by hand), removed by its START button.

THE JOB-OBJECT LESSON (measured 4 Sep 2026 on the earlier responder): a process a scheduled task spawns directly is killed when the task
ends. Every relaunch therefore goes `explorer.exe launch_assistant.cmd`: the desktop shell runs it in its own tree.
"""
import datetime
import json
import os
import re
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
sys.path.insert(0, HARNESS)

from adapter.config import REPO, Config  # noqa: E402

RUN_PY = os.path.join(HARNESS, "adapter", "run.py")
LAUNCH_CMD = os.path.join(HERE, "launch_assistant.cmd")
ZAMBIA = datetime.timezone(datetime.timedelta(hours=2))      # CAT, no daylight saving
STALE_S = 180          # heartbeat older than this = hung (the loop beats every 15 s)
GRACE_S = 300          # a process younger than this is still starting (engine load) and is not judged hung
LOOP_LIMIT = 3         # relaunches within LOOP_WINDOW_S without one fresh heartbeat = crash loop: stop trying
LOOP_WINDOW_S = 3600


class Paths:
    """every file the runtime reads or writes, derived from the adapter's Config (so a test points all at a tmp dir)"""

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        d = self.cfg.log_dir
        self.log_dir = d
        self.lock = os.path.join(d, "assistant.lock")
        self.pid = self.cfg.shadow_pid
        self.heartbeat = os.path.join(d, "heartbeat")
        self.shadow_exit = self.cfg.shadow_exit
        self.runtime_log = os.path.join(d, "runtime.log")
        self.console_log = os.path.join(d, "assistant_console.log")
        self.legacy_console_log = os.path.join(d, "shadow_console.log")
        self.watchdog_log = os.path.join(d, "watchdog.log")
        self.watchdog_state = os.path.join(d, "watchdog_state.json")
        self.mode_file = os.path.join(d, "runtime_mode.json")
        self.stop = os.path.join(self.cfg.base, "RUNTIME_STOP")
        self.pilot_stop = self.cfg.stop_path


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def stamp():
    return now_utc().isoformat(timespec="seconds")


def log_line(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{stamp()}  {text}\n")


# ---------------------------------------------------------------- processes
def pid_alive(pid):
    """True when `pid` is a running PYTHON process (the image-name check guards against a recycled pid)"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    h = k.OpenProcess(0x1000, False, pid)              # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        if not k.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
            return False
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return "python" in os.path.basename(buf.value).lower()
        return True
    finally:
        k.CloseHandle(h)


def kill_pid(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        os.kill(int(pid), 9)


def read_pid(p):
    try:
        return int(open(p.pid).read().strip())
    except (OSError, ValueError):
        return None


def age_s(path, now=None):
    """seconds since the ISO time written in `path` (the heartbeat), None when absent or unreadable"""
    try:
        t = datetime.datetime.fromisoformat(open(path).read().strip())
    except (OSError, ValueError):
        return None
    return ((now or now_utc()) - t).total_seconds()


def mtime_age_s(path):
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return None


# ---------------------------------------------------------------- the lock
class Lock:
    """An OS file lock on pilot_logs/assistant.lock, held for the life of the process. The OS releases it when the
    process dies, however it dies, so a crash never leaves a stale lock (unlike a pid file)."""

    def __init__(self, path):
        self.path, self.f = path, None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        f = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self.f = f
        return True

    def release(self):
        if self.f:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.f.close()
            self.f = None


def lock_held(p):
    """True when a runtime process holds the lock (probe: take it and give it back at once)"""
    lk = Lock(p.lock)
    if lk.acquire():
        lk.release()
        return False
    return True


def state(p, alive=pid_alive):
    """'stop' (RUNTIME_STOP present), 'runtime' (our process holds the lock), 'legacy' (a shadow loop started some
    other way is alive: the runtime takes it over), or 'free'"""
    if os.path.exists(p.stop):
        return "stop"
    if lock_held(p):
        return "runtime"
    pid = read_pid(p)
    if pid and alive(pid):
        return "legacy"
    return "free"


# ---------------------------------------------------------------- launching
def launch_via_explorer(p, reason, wait_s=20):
    """Hand launch_assistant.cmd to explorer.exe (escapes the task's job object) and PROVE it started: the .ps1's
    first act is a line in runtime.log. Returns True only when that line appeared."""
    log_line(p.runtime_log, f"launch requested: {reason}")
    before = os.path.getsize(p.runtime_log)
    if os.name == "nt":
        subprocess.Popen(["explorer.exe", LAUNCH_CMD], close_fds=True)
    else:
        subprocess.Popen(["sh", "-c", f"python3 {os.path.join(HERE, 'assistant.py')} &"])
    for _ in range(int(wait_s * 2)):
        time.sleep(0.5)
        if os.path.exists(p.runtime_log) and os.path.getsize(p.runtime_log) > before:
            return True
    return False


# ---------------------------------------------------------------- the watchdog decision
def _load_state(p):
    try:
        return json.load(open(p.watchdog_state, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def watchdog_check(p, alive=pid_alive, launch=launch_via_explorer, kill=kill_pid, now=None):
    """One watchdog run. Returns (verdict, note) and logs one line to watchdog.log.
    ok        heartbeat fresh, process alive: nothing to do
    stopped   RUNTIME_STOP present: never relaunch
    starting  alive, no fresh beat yet, but younger than GRACE_S (engine loading)
    relaunch  dead, or hung (alive + stale beat: killed first), launch proved by the start line
    failed    the launch never wrote its start line
    loop      LOOP_LIMIT relaunches in the last hour without one fresh heartbeat: give up until a beat or a START"""
    now = now or now_utc()
    st = _load_state(p)
    beat = age_s(p.heartbeat, now)
    fresh = beat is not None and beat <= STALE_S

    def done(verdict, note):
        log_line(p.watchdog_log, f"{verdict}: {note}")
        return verdict, note

    if fresh:
        if st.get("relaunches"):
            st["relaunches"] = []
            json.dump(st, open(p.watchdog_state, "w", encoding="utf-8"))
    if os.path.exists(p.stop):
        return done("stopped", f"RUNTIME_STOP present ({p.stop}); not relaunching")
    pid = read_pid(p)
    is_alive = bool(pid and alive(pid)) or lock_held(p)
    if is_alive and fresh:
        return done("ok", f"pid {pid}, heartbeat {int(beat)} s old")
    if is_alive:
        young = mtime_age_s(p.pid)
        if young is not None and young < GRACE_S:
            return done("starting", f"pid {pid} started {int(young)} s ago, no fresh heartbeat yet")
        if pid:
            kill(pid)
        why = f"hung: pid {pid} alive, heartbeat {'absent' if beat is None else f'{int(beat)} s old'}; killed"
    else:
        why = f"dead: pid {pid or '-'} not running, heartbeat {'absent' if beat is None else f'{int(beat)} s old'}"
    t = now.timestamp()
    recent = [x for x in st.get("relaunches", []) if t - x < LOOP_WINDOW_S]
    if len(recent) >= LOOP_LIMIT:
        return done("loop", f"{why}; {len(recent)} relaunches in the last hour without a fresh heartbeat: NOT "
                            f"relaunching (look at {p.console_log}; START in the widget resets this)")
    recent.append(t)
    st["relaunches"] = recent
    os.makedirs(p.log_dir, exist_ok=True)
    json.dump(st, open(p.watchdog_state, "w", encoding="utf-8"))
    if launch(p, f"watchdog: {why}"):
        return done("relaunch", f"{why}; relaunched through explorer.exe")
    return done("failed", f"{why}; the launch never wrote its start line to {p.runtime_log}")


def reset_loop_guard(p):
    try:
        os.remove(p.watchdog_state)
    except OSError:
        pass


# ---------------------------------------------------------------- stop / start (the widget's buttons)
def request_stop(p, who="widget", wait_s=90, alive=pid_alive, kill=kill_pid):
    """write RUNTIME_STOP (the watchdog leaves it down) and SHADOW_EXIT (the loop exits at its next poll, after the
    message in hand); if it is still alive after wait_s, kill it. Returns a note."""
    os.makedirs(os.path.dirname(p.stop), exist_ok=True)
    with open(p.stop, "a", encoding="utf-8") as f:
        f.write(f"{stamp()} stopped by {who}\n")
    log_line(p.runtime_log, f"STOP by {who}")
    pid = read_pid(p)
    if not (pid and alive(pid)):
        return "stop file written; no process was running"
    open(p.shadow_exit, "w").write("stop")
    for _ in range(int(wait_s)):
        time.sleep(1)
        if not alive(pid):
            log_line(p.runtime_log, f"pid {pid} exited cleanly")
            return f"pid {pid} stopped cleanly"
    kill(pid)
    for f in (p.shadow_exit, p.pid):
        try:
            os.remove(f)
        except OSError:
            pass
    log_line(p.runtime_log, f"pid {pid} did not exit in {wait_s} s: killed")
    return f"pid {pid} killed after {wait_s} s"


def request_start(p, who="widget", launch=launch_via_explorer):
    if os.path.exists(p.stop):
        os.remove(p.stop)
    reset_loop_guard(p)
    log_line(p.runtime_log, f"START by {who} (RUNTIME_STOP removed)")
    s = state(p)
    if s == "runtime":
        return "already running"
    return "launched" if launch(p, f"START by {who}") else "launch did not write its start line"


# ---------------------------------------------------------------- the status record
def _jsonl(path):
    try:
        return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    except (OSError, ValueError):
        return []


def _tail(path, n=400):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64000))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


ERR = re.compile(r"poll error|Traceback|Error|REFUSED|exception", re.I)


def mode_label(p, running, owner):
    """what the running process does with replies: from runtime_mode.json (written by assistant.py at start), with the
    go file and PILOT_STOP read NOW (a send needs both at the moment of sending, whatever the process started as)"""
    cfg = p.cfg
    try:
        m = json.load(open(p.mode_file, encoding="utf-8"))
    except (OSError, ValueError):
        m = {}
    if not running:
        return "not running"
    if owner != "runtime" or m.get("pid") != read_pid(p):
        return "unknown (process not started by the runtime)"
    if m.get("mode") == "live_tester":
        if cfg.stopped():
            return "LIVE (tester only) but PILOT_STOP is ON: nothing is sent"
        if not cfg.go():
            return "LIVE (tester only) but PILOT_GO.json is gone: nothing is sent"
        return "LIVE (tester only): real replies to T00; teachers to the outbox"
    return "shadow (nothing sent; replies to the outbox)"


def status(p, alive=pid_alive):
    cfg = p.cfg
    pid = read_pid(p)
    running = bool(pid and alive(pid))
    owner = "runtime" if lock_held(p) else ("other (not the runtime)" if running else "-")
    beat = age_s(p.heartbeat)
    today = datetime.datetime.now(ZAMBIA).date().isoformat()
    turns = _jsonl(os.path.join(p.log_dir, f"{today}.jsonl"))
    labels = _jsonl(cfg.labels_path)
    outbox = _jsonl(cfg.outbox_path)
    numbers = []
    allow = cfg.allowlist()
    modes = {}
    if os.path.exists(cfg.adapter_db):
        try:
            con = sqlite3.connect(f"file:{cfg.adapter_db}?mode=ro", uri=True, timeout=2)
            modes = {r[0]: r[1:] for r in con.execute("select number, mode, since, last_in_at from modes")}
            con.close()
        except sqlite3.Error:
            pass
    for n, e in allow.items():
        m = modes.get(n, ("desk", None, None))
        numbers.append({"code": e.get("code"), "name": e.get("name"), "tester": bool(e.get("tester")),
                        "mode": m[0], "last_in_at": m[2],
                        "turns_today": sum(1 for r in turns if r.get("number") == n),
                        "engine_today": sum(1 for r in turns if r.get("number") == n and r.get("route") == "engine")})
    errs = [x for x in _tail(p.console_log) + _tail(p.legacy_console_log) if ERR.search(x)]
    sends_err = [r for r in turns for s in (r.get("sends") or []) if isinstance(s, dict) and s.get("error")]
    last_error = errs[-1] if errs else (json.dumps(sends_err[-1]["sends"])[:200] if sends_err else None)
    wd = _tail(p.watchdog_log, 1)
    return {
        "at": stamp(),
        "running": running and beat is not None and beat <= STALE_S,
        "process": {"pid": pid, "alive": running, "owner": owner},
        "heartbeat_age_s": None if beat is None else int(beat),
        "mode": mode_label(p, running, owner),
        "runtime_stop": os.path.exists(p.stop),
        "pilot_stop": cfg.stopped(),
        "numbers": numbers,
        "turns_today": len(turns),
        "engine_turns_today": sum(1 for r in turns if r.get("route") == "engine"),
        "labels": len(labels),
        "labels_today": sum(1 for x in labels if str(x.get("at", "")).startswith(today)),
        "outbox": len(outbox),
        "last_error": last_error,
        "watchdog_last": wd[-1] if wd else None,
        "day": today,
    }


def main():
    """`python common.py check` -> exit 0 free, 1 legacy (takeover), 3 runtime already running, 4 RUNTIME_STOP;
    `python common.py status` -> the status record as JSON"""
    p = Paths()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "check":
        s = state(p)
        print(s)
        sys.exit({"free": 0, "legacy": 1, "runtime": 3, "stop": 4}[s])
    print(json.dumps(status(p), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
