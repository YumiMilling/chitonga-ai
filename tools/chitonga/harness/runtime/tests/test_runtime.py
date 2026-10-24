"""Tests for the assistant runtime (24 Oct 2026): the single-instance lock, the watchdog's decisions (stale heartbeat
-> kill + relaunch, dead -> relaunch, STOP respected, starting grace, crash-loop guard), STOP / START, the status
record and the widget's token check. Every path points at a temp directory; nothing is launched or killed for real.

    python -X utf8 -m unittest discover -s tools/chitonga/harness/runtime/tests -v
"""
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.dirname(HERE)
sys.path.insert(0, RUNTIME)

import common as C  # noqa: E402
from adapter.config import Config  # noqa: E402


def mk():
    d = tempfile.mkdtemp(prefix="rt_")
    json.dump({"teachers": [{"number": "260900000001", "code": "T00", "name": "Tester", "tester": True}]},
              open(os.path.join(d, "allow.json"), "w"))
    p = C.Paths(Config(base=d, allowlist=os.path.join(d, "allow.json"), env={}))
    os.makedirs(p.log_dir, exist_ok=True)
    return p


def beat(p, age):
    t = C.now_utc() - datetime.timedelta(seconds=age)
    open(p.heartbeat, "w").write(t.isoformat(timespec="seconds"))


def pidfile(p, pid, age):
    open(p.pid, "w").write(str(pid))
    t = time.time() - age
    os.utime(p.pid, (t, t))


class Rec:
    def __init__(self, ok=True):
        self.calls, self.ok = [], ok

    def __call__(self, *a):
        self.calls.append(a)
        return self.ok


class TestLock(unittest.TestCase):
    def test_second_holder_refused_and_released_on_death(self):
        p = mk()
        code = ("import sys,time; sys.path.insert(0, sys.argv[1]); import common as C; "
                "l=C.Lock(sys.argv[2]); print(l.acquire(), flush=True); time.sleep(30)")
        child = subprocess.Popen([sys.executable, "-c", code, RUNTIME, p.lock], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "True")
            self.assertFalse(C.Lock(p.lock).acquire())            # a second copy is refused
            self.assertTrue(C.lock_held(p))
            self.assertEqual(C.state(p, alive=lambda pid: False), "runtime")
        finally:
            child.kill()
            child.wait()
        time.sleep(0.3)
        self.assertFalse(C.lock_held(p))                          # the OS released it when the holder died
        lk = C.Lock(p.lock)
        self.assertTrue(lk.acquire())
        lk.release()

    def test_state_stop_legacy_free(self):
        p = mk()
        self.assertEqual(C.state(p, alive=lambda pid: False), "free")
        pidfile(p, 4242, 10)
        self.assertEqual(C.state(p, alive=lambda pid: pid == 4242), "legacy")
        open(p.stop, "w").write("x")
        self.assertEqual(C.state(p, alive=lambda pid: True), "stop")


class TestWatchdog(unittest.TestCase):
    def test_fresh_heartbeat_nothing(self):
        p = mk()
        pidfile(p, 11, 3600)
        beat(p, 20)
        launch, kill = Rec(), Rec()
        v, _ = C.watchdog_check(p, alive=lambda pid: True, launch=launch, kill=kill)
        self.assertEqual(v, "ok")
        self.assertEqual((launch.calls, kill.calls), ([], []))

    def test_stale_heartbeat_kills_and_relaunches(self):
        p = mk()
        pidfile(p, 11, 3600)
        beat(p, 600)
        launch, kill = Rec(), Rec()
        v, note = C.watchdog_check(p, alive=lambda pid: True, launch=launch, kill=kill)
        self.assertEqual(v, "relaunch")
        self.assertEqual(kill.calls, [(11,)])
        self.assertEqual(len(launch.calls), 1)
        self.assertIn("hung", note)

    def test_dead_relaunches(self):
        p = mk()
        pidfile(p, 11, 3600)
        beat(p, 60)                                               # fresh beat but the process died since
        launch, kill = Rec(), Rec()
        v, _ = C.watchdog_check(p, alive=lambda pid: False, launch=launch, kill=kill)
        self.assertEqual(v, "relaunch")
        self.assertEqual(kill.calls, [])

    def test_stop_respected(self):
        p = mk()
        open(p.stop, "w").write("x")
        launch, kill = Rec(), Rec()
        v, _ = C.watchdog_check(p, alive=lambda pid: False, launch=launch, kill=kill)
        self.assertEqual(v, "stopped")
        self.assertEqual((launch.calls, kill.calls), ([], []))
        self.assertIn("stopped", open(p.watchdog_log).read())

    def test_starting_grace(self):
        p = mk()
        pidfile(p, 11, 30)                                        # 30 s old, no heartbeat yet
        launch, kill = Rec(), Rec()
        v, _ = C.watchdog_check(p, alive=lambda pid: True, launch=launch, kill=kill)
        self.assertEqual(v, "starting")
        self.assertEqual((launch.calls, kill.calls), ([], []))

    def test_crash_loop_gives_up_and_beat_resets(self):
        p = mk()
        launch = Rec()
        verdicts = [C.watchdog_check(p, alive=lambda pid: False, launch=launch, kill=Rec())[0] for _ in range(4)]
        self.assertEqual(verdicts, ["relaunch"] * 3 + ["loop"])
        self.assertEqual(len(launch.calls), 3)
        beat(p, 10)                                               # a fresh beat clears the guard
        C.watchdog_check(p, alive=lambda pid: False, launch=launch, kill=Rec())
        self.assertEqual(len(launch.calls), 4)

    def test_failed_launch_reported(self):
        p = mk()
        v, _ = C.watchdog_check(p, alive=lambda pid: False, launch=Rec(ok=False), kill=Rec())
        self.assertEqual(v, "failed")


class TestStopStart(unittest.TestCase):
    def test_stop_writes_file_and_exit_then_start_clears(self):
        p = mk()
        pidfile(p, 11, 100)
        alive = {"v": True}

        def fake_alive(pid):
            if os.path.exists(p.shadow_exit):                     # the loop sees SHADOW_EXIT and exits
                os.remove(p.shadow_exit)
                alive["v"] = False
            return alive["v"]
        note = C.request_stop(p, "test", wait_s=5, alive=fake_alive, kill=Rec())
        self.assertIn("cleanly", note)
        self.assertTrue(os.path.exists(p.stop))
        self.assertEqual(C.watchdog_check(p, alive=lambda pid: False, launch=Rec(), kill=Rec())[0], "stopped")
        launch = Rec()
        self.assertEqual(C.request_start(p, "test", launch=launch), "launched")
        self.assertFalse(os.path.exists(p.stop))
        self.assertEqual(len(launch.calls), 1)

    def test_stop_kills_after_wait(self):
        p = mk()
        pidfile(p, 11, 100)
        kill = Rec()
        note = C.request_stop(p, "test", wait_s=1, alive=lambda pid: True, kill=kill)
        self.assertIn("killed", note)
        self.assertEqual(kill.calls, [(11,)])
        self.assertFalse(os.path.exists(p.shadow_exit))          # no stale SHADOW_EXIT for the next start


class TestStatus(unittest.TestCase):
    def test_status_empty_and_with_data(self):
        p = mk()
        st = C.status(p, alive=lambda pid: False)
        self.assertFalse(st["running"])
        self.assertEqual(st["outbox"], 0)
        self.assertEqual(st["mode"], "not running")
        day = st["day"]
        with open(os.path.join(p.log_dir, f"{day}.jsonl"), "w") as f:
            f.write(json.dumps({"number": "260900000001", "route": "engine"}) + "\n")
            f.write(json.dumps({"number": "260900000001", "route": "desk"}) + "\n")
        open(p.cfg.outbox_path, "w").write('{"to": "x"}\n')
        open(p.console_log, "w").write("ok\n2026-10-24T05:00:00+00:00 poll error: timeout\nfine\n")
        pidfile(p, 11, 100)
        beat(p, 5)
        st = C.status(p, alive=lambda pid: True)
        self.assertTrue(st["running"])
        self.assertEqual((st["turns_today"], st["engine_turns_today"], st["outbox"]), (2, 1, 1))
        self.assertEqual(st["numbers"][0]["turns_today"], 2)
        self.assertIn("poll error", st["last_error"])

    def test_mode_label_live_tester_and_gate_now(self):
        import assistant as A
        p = mk()
        self.assertEqual(A.choose_mode(p.cfg)[0], "shadow")                 # no go file
        json.dump({"chris_words": "just do it live", "date": "2026-10-24"}, open(p.cfg.go_path, "w"))
        self.assertEqual(A.choose_mode(p.cfg)[0], "live_tester")
        lk = C.Lock(p.lock)
        self.assertTrue(lk.acquire())                                         # this process plays the runtime
        try:
            pidfile(p, os.getpid(), 10)
            beat(p, 5)
            json.dump({"mode": "live_tester", "pid": os.getpid()}, open(p.mode_file, "w"))
            # lock_held probes with a second handle: on Windows the same process is refused too
            st = C.status(p, alive=lambda pid: True)
            self.assertTrue(st["mode"].startswith("LIVE (tester only)"), st["mode"])
            open(p.cfg.stop_path, "w").write("x")                           # PILOT_STOP: nothing is sent
            self.assertIn("PILOT_STOP is ON", C.status(p, alive=lambda pid: True)["mode"])
            self.assertEqual(A.choose_mode(p.cfg)[0], "shadow")
            json.dump({"mode": "shadow", "pid": os.getpid()}, open(p.mode_file, "w"))
            self.assertTrue(C.status(p, alive=lambda pid: True)["mode"].startswith("shadow"))
        finally:
            lk.release()
        self.assertTrue(C.status(p, alive=lambda pid: True)["mode"].startswith("unknown"))


class TestWidget(unittest.TestCase):
    def test_page_renders_and_post_needs_token(self):
        import widget as W
        p = mk()
        html = W.page(C.status(p, alive=lambda pid: False))
        self.assertIn(W.TOKEN, html)
        self.assertIn("NOT RUNNING", html)
        self.assertNotIn("#c00", html.lower())


if __name__ == "__main__":
    unittest.main()
