"""Tests of the pilot WhatsApp adapter (24 Oct 2026, the Jesuit). No network: the sender is mocked, every path is a
temp directory. Run: python -X utf8 tools/chitonga/harness/adapter/tests/test_adapter.py
"""
import datetime
import glob
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HARNESS)

from core import io as IO  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.fixed import FixedLines  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from core.plugin import Plugin  # noqa: E402

from adapter import inbound as IN  # noqa: E402
from adapter import payload as PL  # noqa: E402
from adapter.adapter import PilotAdapter, engine_clock  # noqa: E402
from adapter.config import Config  # noqa: E402
from adapter.simulate import CODES, Sim, run_script  # noqa: E402
from adapter.store import AdapterStore  # noqa: E402
from adapter.transport import CloudSender  # noqa: E402

T01, T09, STRANGER = CODES["T01"], CODES["T09"], "260970000099"


def tmp(test):
    d = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, d, ignore_errors=True)
    return d


class TestPayload(unittest.TestCase):
    def test_buttons_interactive(self):
        p, text, opts, flags = PL.build(T01, IO.buttons("Continue?", [("c:continue", "Continue"), ("c:else", "Else")]),
                                        "interactive")
        self.assertEqual(p["type"], "interactive")
        self.assertEqual(p["interactive"]["type"], "button")
        self.assertEqual(p["interactive"]["action"]["buttons"][0], {"type": "reply",
                                                                   "reply": {"id": "c:continue", "title": "Continue"}})
        self.assertEqual(opts, [("c:continue", "Continue"), ("c:else", "Else")])
        self.assertEqual(flags, [])

    def test_list_interactive(self):
        rows = [(f"unit:{i}", f"Unit {i}", "ready") for i in range(5)]
        p, _, opts, _ = PL.build(T01, IO.listing("Which unit?", rows), "interactive")
        sec = p["interactive"]["action"]["sections"]
        self.assertEqual(p["interactive"]["type"], "list")
        self.assertEqual(len(sec[0]["rows"]), 5)
        self.assertLessEqual(len(p["interactive"]["action"]["button"]), 20)
        self.assertEqual(len(opts), 5)

    def test_numbered_mode(self):
        p, text, opts, _ = PL.build(T01, IO.buttons("Continue?", [("c:continue", "Continue")]), "numbered")
        self.assertEqual(p["type"], "text")
        self.assertIn("1. Continue", text)
        self.assertIn(PL.NUMBER_HINT, text)
        self.assertEqual(opts, [("c:continue", "Continue")])

    def test_invalid_reply_falls_back_to_text_never_dropped(self):
        bad = {"type": "buttons", "body": "x", "buttons": [{"id": "a", "title": "A title much longer than twenty"}]}
        p, text, _, flags = PL.build(T01, bad, "interactive")
        self.assertEqual(p["type"], "text")
        self.assertTrue(any(f.startswith("fallback") for f in flags))
        self.assertIn("A title much longer", text)

    def test_dashes_and_control_bytes(self):
        p, text, _, flags = PL.build(T01, IO.text("Grade 1 — Maths 1–9"), "interactive")
        self.assertNotIn("—", text)
        self.assertNotIn("–", text)
        self.assertIn("1-9", text)
        _, _, _, flags = PL.build(T01, IO.text("bad" + chr(8) + "byte"), "interactive")
        self.assertIn("control byte", flags)


class TestInbound(unittest.TestCase):
    def setUp(self):
        self.store = AdapterStore(":memory:")
        self.store.add_menu(T01, "t1", "w1", [("c:continue", "Continue"), ("c:else", "Something else")])
        self.store.add_menu(T01, "t2", "w2", [("c:yes", "Yes"), ("c:no", "No"), ("c:menu", "Menu")])

    def test_choice_id_wins(self):
        m, how = IN.parse({"msg_type": "interactive", "body": "No", "choice_id": "c:no"}, self.store, T01)
        self.assertEqual((m["choice"], how), ("c:no", "choice_id"))

    def test_title_maps_to_the_newest_menu(self):
        self.store.add_menu(T01, "t3", "w3", [("c:continue", "Continue"), ("c:mistakes", "Mistakes")])
        m, how = IN.parse({"msg_type": "interactive", "body": "Continue"}, self.store, T01)
        self.assertEqual((m["choice"], how), ("c:continue", "title_map"))
        m, how = IN.parse({"msg_type": "interactive", "body": "Something else"}, self.store, T01)
        self.assertEqual(m["choice"], "c:else")         # older menu, still found

    def test_typed_number_maps_only_to_the_last_menu(self):
        m, how = IN.parse({"msg_type": "text", "body": "2"}, self.store, T01)
        self.assertEqual((m["choice"], how), ("c:no", "number_map"))
        m, how = IN.parse({"msg_type": "text", "body": "7"}, self.store, T01)
        self.assertEqual((m, how), ({"text": "7"}, "text"))
        m, _ = IN.parse({"msg_type": "text", "body": "2"}, self.store, T09)   # no menu sent to this number
        self.assertEqual(m, {"text": "2"})

    def test_bare_interactive_media_and_silent(self):
        self.assertEqual(IN.parse({"msg_type": "interactive", "body": "[interactive]"}, self.store, T01)[1],
                         "bare_interactive")
        self.assertEqual(IN.parse({"msg_type": "image", "body": "[image]"}, self.store, T01)[0], {"media": "image"})
        for t in ("reaction", "edit"):
            self.assertEqual(IN.parse({"msg_type": t, "body": "x"}, self.store, T01)[0], {"silent": t})


class TestSwitches(unittest.TestCase):
    def cfg(self, env):
        d = tmp(self)
        c = Config(base=d, env=env)
        return c

    def test_live_needs_env_go_and_no_stop(self):
        c = self.cfg({"ZAMAI_ASSISTANT_LIVE": "1"})
        self.assertFalse(c.live())                                   # no go file
        json.dump({"chris_words": "go", "date": "2026-10-25"}, open(c.go_path, "w"))
        self.assertTrue(c.live())
        c.stop("test")
        self.assertFalse(c.live())                                   # the stop wins
        c2 = self.cfg({})
        json.dump({"chris_words": "go", "date": "2026-10-25"}, open(c2.go_path, "w"))
        self.assertFalse(c2.live())                                  # no env flag
        self.assertTrue(self.cfg({"ZAMAI_ASSISTANT_STOP": "1"}).stopped())

    def test_go_file_needs_words_and_date(self):
        c = self.cfg({"ZAMAI_ASSISTANT_LIVE": "1"})
        json.dump({"date": "2026-10-25"}, open(c.go_path, "w"))
        self.assertFalse(c.live())

    def test_allowlist_refuses_without_consent_in_live(self):
        c = Config()                                                 # the real file: consent not yet asked
        self.assertEqual(set(c.allowlist()), {T01, T09, "260970000100"})
        self.assertEqual(set(c.allowlist(live=True)), {"260970000100"})   # only the tester
        self.assertEqual(c.testers(), {"260970000100"})

    def test_outbox_never_sends_to_a_teacher_and_tester_only_when_live(self):
        from adapter.transport import OutboxSender
        d = tmp(self)
        allow = os.path.join(d, "allow.json")
        json.dump({"teachers": [{"number": "260970000100", "code": "T00", "tester": True},
                                {"number": T01, "code": "T01", "consent": {"words": "yes"}}]}, open(allow, "w"))
        c = Config(base=d, allowlist=allow, env={"ZAMAI_ASSISTANT_LIVE": "1"})

        class Real:
            sent = []

            def send(self, payload, text, t):
                self.sent.append(payload["to"])
                return "wamid.real", None
        real = Real()
        ob = OutboxSender(c.outbox_path, c, real=real)
        self.assertTrue(ob.send({"to": "260970000100"}, "x", "text")[0].startswith("outbox:"))   # not live: outbox
        json.dump({"chris_words": "go", "date": "2026-10-25"}, open(c.go_path, "w"))
        self.assertEqual(ob.send({"to": "260970000100"}, "x", "text")[0], "wamid.real")           # tester, live
        self.assertTrue(ob.send({"to": T01}, "x", "text")[0].startswith("outbox:"))               # teacher: never
        self.assertEqual(real.sent, ["260970000100"])
        self.assertEqual(len(open(c.outbox_path, encoding="utf-8").read().splitlines()), 2)

    def test_cloud_sender_refuses_when_not_live(self):
        c = self.cfg({})
        s = object.__new__(CloudSender)                              # no network: skip the credential read
        s.config = c
        wa, err = s.send({"to": T01}, "x", "text")
        self.assertIsNone(wa)
        self.assertIn("refused", err)

    def test_pilot_refuses_the_tonga_interface(self):
        from plugins import education
        P, H = education.load()
        e = Engine(P, SQLiteMemory(":memory:"), hooks=H, tonga_ui=True)
        with self.assertRaises(ValueError):
            PilotAdapter(e, Config(base=tmp(self)), AdapterStore(":memory:"), None, lambda: None)


class TestAdapter(unittest.TestCase):
    def sim(self, **kw):
        return Sim(tmp(self), **kw)

    def start(self, s, who="T01", lang="English"):
        s.step({"from": who, "text": "Assistant"})
        s.step({"from": who, "tap": "Skip"})
        s.step({"from": who, "tap": lang})
        s.step({"from": who, "tap": "Grade 1"})

    def test_stranger_never_reaches_the_engine(self):
        s = self.sim()
        r = s.step({"from": STRANGER, "text": "Assistant"})
        self.assertEqual(r["route"], "ignored")
        self.assertEqual(r["_sent"], [])
        self.assertIsNone(s.engine.mem.get_teacher(STRANGER))
        log = open(glob.glob(os.path.join(s.cfg.log_dir, "*.jsonl"))[0], encoding="utf-8").read()
        self.assertNotIn(STRANGER, log)
        self.assertNotIn("Assistant", log)

    def test_desk_mode_by_default(self):
        s = self.sim()
        r = s.step({"from": "T01", "text": "Hello, here are my answers"})
        self.assertEqual((r["route"], r["_sent"]), ("desk", []))
        self.assertIsNone(s.engine.mem.get_teacher(T01))

    def test_greeting_discloses_ai_in_english(self):
        s = self.sim()
        r = s.step({"from": "T01", "text": "Assistant"})
        self.assertIn("an AI", r["_sent"][0]["text"])

    def test_english_interface_tonga_content(self):
        s = self.sim()
        self.start(s, "T09", "Tonga")
        r = s.step({"from": "T09", "text": "I want to teach addition"})
        r = s.step({"from": "T09", "tap": "Continue"})
        sent = " / ".join(x["text"] for x in r["_sent"])
        self.assertIn("Say it in Tonga", sent)                       # Tonga is the content
        titles = [t for _, t in s.ad.store.last_menu(T09)["options"]]
        from plugins import education
        P, _ = education.load()
        # the lesson's buttons first (English); since 24 Oct the turn's one flywheel question may follow, its buttons
        # stored in the same menu (FLYWHEEL.md), also English
        self.assertEqual(titles[:3], [P.button_titles[k] for k in ("continue", "mistakes", "something_else")])
        self.assertTrue(set(titles[3:]) <= {"Tonga OK", "Fix a line", "Stop asking", "Right as is", "Write it",
                                            "Not now"}, titles)

    def test_safeguarding_first_then_pupil_guard(self):
        s = self.sim()
        self.start(s)
        r = s.step({"from": "T01", "text": "A girl in grade 3 told me her stepfather hurts her at night."})
        self.assertEqual(r["route"], "engine")
        self.assertIn("116", r["_sent"][0]["text"])
        self.assertIn("safeguarding", r["turn"]["path"])
        r = s.step({"from": "T01", "text": "Mutinta scored 3 out of 10 in the test"})
        self.assertEqual(r["how"], "pupil_withheld")
        for f in glob.glob(os.path.join(s.cfg.base, "**", "*"), recursive=True):
            if os.path.isfile(f):
                blob = open(f, "rb").read()
                self.assertNotIn(b"stepfather", blob, f)
                self.assertNotIn(b"Mutinta", blob, f)

    def test_kill_switch_stops_before_the_engine_and_between_sends(self):
        s = self.sim()
        self.start(s)
        s.cfg.stop("test")
        r = s.step({"from": "T01", "text": "I want to teach addition"})
        self.assertEqual((r["route"], r["_sent"]), ("desk", []))
        os.remove(s.cfg.stop_path)
        # stop created DURING a turn: the first reply goes, the rest do not
        orig = s.sender.send

        def send_then_stop(payload, text, t):
            out = orig(payload, text, t)
            s.cfg.stop("mid-turn")
            return out
        s.sender.send = send_then_stop
        r = s.step({"from": "T01", "text": "I want to teach addition"})
        self.assertEqual(len(r["_sent"]), 1)
        self.assertEqual(r["sends"][-1]["why"], "stop switch")

    def test_window_held_does_not_touch_the_engine(self):
        s = self.sim()
        self.start(s)
        before = s.engine.mem.get_state(T01)
        r = s.step({"from": "T01", "text": "subtraction lesson", "age_h": 25})
        self.assertEqual((r["route"], r["_sent"]), ("held", []))
        self.assertEqual(s.engine.mem.get_state(T01), before)
        r = s.step({"from": "T01", "text": "Stop"})                   # the held row did not age the session
        self.assertEqual(r["route"], "assistant_stop")

    def test_resume_next_day(self):
        s = self.sim()
        self.start(s)
        s.step({"from": "T01", "text": "I want to teach addition"})
        s.clock.advance(hours=20)
        r = s.step({"from": "T01", "text": "Continue"})
        self.assertEqual(r["route"], "desk")                         # 2 h of silence: back to the desk
        r = s.step({"from": "T01", "text": "Assistant"})
        self.assertIn("Last time", "\n".join(x["text"] for x in r["_sent"]))

    def test_send_failures_pause_the_teacher(self):
        s = self.sim()
        self.start(s)
        s.sender.fail = 3
        for _ in range(3):
            s.step({"from": "T01", "text": "I want to teach addition"})
        self.assertEqual(s.store.mode(T01)["mode"], "paused")
        self.assertTrue(any("paused" in x for x in s.notes))
        r = s.step({"from": "T01", "text": "Assistant"})
        self.assertEqual((r["route"], r["_sent"]), ("desk", []))      # a person must clear a pause

    def test_rate_limits(self):
        s = self.sim()
        self.start(s)
        s.cfg.max_out_per_in = 1
        r = s.step({"from": "T01", "text": "I want to teach addition"})
        self.assertEqual(len(r["_sent"]), 1)
        self.assertIn("over 1 replies", r["sends"][-1]["why"])
        s.cfg.max_in_per_hour = 0
        r = s.step({"from": "T01", "text": "I want to teach addition"})
        self.assertEqual((r["route"], r["why"]), ("held", "inbound rate limit"))

    def test_desk_message_since_start_returns_to_desk(self):
        s = self.sim()
        self.start(s)
        s.ad.desk_sent = lambda n, since, until: True
        r = s.step({"from": "T01", "text": "Kamubala"})
        self.assertEqual(r["route"], "desk")
        self.assertEqual(s.store.mode(T01)["mode"], "desk")

    def test_held_out_first_n_requests(self):
        s = self.sim()
        s.cfg.held_out_first = 2
        self.start(s)
        flags = [s.step({"from": "T01", "text": t}).get("held_out")
                 for t in ("I want to teach addition", "counting lesson", "subtraction lesson")]
        self.assertEqual(flags, [True, True, False])

    def test_labels_never_hold_a_phone_number(self):
        s = self.sim()
        self.start(s, "T09", "Tonga")
        s.step({"from": "T09", "text": "Pandulula nsaizi"})
        s.step({"from": "T09", "tap": "No"})
        lab = open(s.cfg.labels_path, encoding="utf-8").read()
        self.assertIn('"teacher": "T09"', lab)
        self.assertNotIn(T09, lab)


class TestScripts(unittest.TestCase):
    def test_every_script_passes(self):
        for p in sorted(glob.glob(os.path.join(HARNESS, "adapter", "sim", "*.json"))):
            sc = json.load(open(p, encoding="utf-8"))
            fails, _, _ = run_script(sc, tmp(self), quiet=True)
            self.assertEqual(fails, [], os.path.basename(p))


class TestEngineEnglishInterface(unittest.TestCase):
    """24 Oct (Chris): the interface is English by default; the Tonga fixed-line table stays, switched off"""

    def table(self):
        d = tmp(self)
        from plugins import education
        P, H = education.load()
        path = os.path.join(d, "fixed.json")
        json.dump({"lines": {}, "buttons": {"lesson_help": {"en": P.button_titles["lesson_help"], "toi": "TONGA-LH",
                                                            "origin": ["T01"], "status": "served"}}}, open(path, "w"))
        return Plugin(P.dir, hooks=H, fixed=FixedLines(path)), H

    def talk(self, tonga_ui):
        p2, H = self.table()
        e = Engine(p2, SQLiteMemory(":memory:"), hooks=H, tonga_ui=tonga_ui)
        for m in ({"text": ""}, {"choice": "c:skip_name"}, {"choice": "lang:toi"}):
            e.handle(T01, m)
        return e.handle(T01, {"choice": "newgrade:1"})

    def test_default_english(self):
        titles = [b["title"] for r in self.talk(False) for b in r.get("buttons") or []]
        self.assertNotIn("TONGA-LH", titles)

    def test_switch_on_serves_tonga(self):
        titles = [b["title"] for r in self.talk(True) for b in r.get("buttons") or []]
        self.assertIn("TONGA-LH", titles)


if __name__ == "__main__":
    unittest.main(verbosity=1)
