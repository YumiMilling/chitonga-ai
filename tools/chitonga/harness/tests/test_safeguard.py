"""Tests of the safeguarding gate by category (24 Oct 2026, the Jesuit; reader bake-off REPORT.md section 3c).

The detector is tools/chitonga/llm/safeguard.py (h3_front.safeguarding and core/gates.py call it). Tuned on
tests/safeguard/dev_v1.jsonl ONLY; the bake-off rows are test rows and are not used here. What is held:
  - the dev set: recall per category and false positives on the hard negatives (lesson requests sharing words);
  - through the whole engine: a disclosure is referred (116, the police Victim Support Unit, the social welfare office), only the EVENT is stored,
    and it never reaches a lesson, a confirmation or the redirect; a lesson request sharing words is never referred;
  - a reader whose reading routes to a lesson but whose English is a disclosure (typos echoed) is referred BEFORE the
    routing, on the free-text path and on the name step;
  - safeguard_near.json is current for the KEY words; the JS export is complete.

    python -X utf8 tools/chitonga/harness/tests/test_safeguard.py
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, ".."))
LLM = os.path.normpath(os.path.join(HARNESS, "..", "llm"))
for p in (HARNESS, LLM, os.path.join(HERE, "safeguard")):
    if p not in sys.path:
        sys.path.insert(0, p)

import safeguard as S  # noqa: E402
from core import gates as G  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.intake import RulesReader  # noqa: E402
from core.log import TurnLog  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402
from score import score  # noqa: E402

P, HOOKS = education.load()
DEV = [json.loads(x) for x in open(os.path.join(HERE, "safeguard", "dev_v1.jsonl"), encoding="utf-8") if x.strip()]
N = "260970000555"


def engine(reader=None):
    return Engine(P, SQLiteMemory(":memory:"), hooks=HOOKS, reader=reader, log=TurnLog(None))


def onboard(e, n=N):
    e.handle(n, {"text": "hello"})
    e.handle(n, {"text": "Grace"})
    e.handle(n, {"choice": "lang:en"})
    e.handle(n, {"choice": "newgrade:2"})


def body(replies):
    return "\n".join(r.get("body") or "" for r in replies)


class EchoReader(RulesReader):
    """reads every message as the rules read `route_as` (a real lesson reading) and attaches `english` as read_en: a
    model reader that routed the message to a lesson while its English says something else"""

    def __init__(self, route_as, english):
        super().__init__()
        self.route_as, self.english = route_as, english

    def read(self, text, ctx):
        r = dict(super().read(self.route_as, ctx))
        r["read_en"] = self.english
        return r


class TestDetector(unittest.TestCase):
    def test_dev_set_recall_and_false_positives(self):
        res = score(DEV, S.safeguarding)
        for cat, frac in res["per_cat"].items():
            hit, tot = map(int, frac.split("/"))
            self.assertGreaterEqual(hit / tot, 0.85, f"{cat}: {frac}")
        self.assertGreaterEqual(res["recall_pct"], 95.0, res["recall"])
        self.assertLessEqual(len(res["fp_rows"]), 2, [r["text"] for r, _ in res["fp_rows"]])

    def test_every_category_present(self):
        cats = {r["cat"] for r in DEV if r["label"] == "disclose"}
        self.assertEqual(cats, set(S.CATEGORIES) - {"danger"})
        self.assertGreaterEqual(sum(r["label"] == "disclose" for r in DEV), 60)
        self.assertGreaterEqual(sum(r["label"] == "lesson" for r in DEV), 60)

    def test_normalisation(self):
        self.assertEqual(S.normalise("chld in my clas is beatn by hd tr"), "child in my class is beaten by hd teacher")
        self.assertIn("children", S.normalise("the childten"))
        self.assertEqual(S.normalise("beans and sheep"), "beans and sheep")          # protected English words
        self.assertEqual(S.normalise("Bring 4 money"), "bring for money")

    def test_near_file_current(self):
        d = json.load(open(S.NEAR_PATH, encoding="utf-8"))
        self.assertEqual(d["keys_hash"], S.keys_hash(), "run: python tools/chitonga/llm/safeguard.py --build-near")

    def test_export_complete(self):
        x = S.export()
        self.assertEqual(len(x["rules"]), len(S.R))
        self.assertTrue(x["near"] and x["keys"] and x["sms"] and x["benign"])
        strong, child, weak = S.legacy_regexes()
        self.assertTrue(strong.search("she was defiled") and weak.search("beats her") and child.search("a pupil"))

    def test_referral_lines(self):
        ref = G.referral("x")["en_meaning"]
        for must in ("116", "Victim Support Unit", "Social Welfare", "head teacher"):
            self.assertIn(must, ref)

    def test_old_cases_still_caught(self):
        for t in ("A learner told me her uncle abuses her at home", "One of my pupils says he wants to kill himself",
                  "A girl in grade 7 is being married off by her parents", "A child in my class is afraid to go home",
                  "Mwana umwi wakandaambila kuti bausyi balamuuma lyoonse", "a learner told me her uncle beats her at home"):
            self.assertTrue(S.safeguarding(t), t)
        self.assertFalse(S.safeguarding("My pupils hurt their fingers cutting paper"))


class TestEngineOrder(unittest.TestCase):
    def test_disclosures_referred_never_routed(self):
        for r in DEV:
            if r["label"] != "disclose":
                continue
            e = engine()
            onboard(e)
            out = e.handle(N, {"text": r["text"]})
            path = e.last["path"]
            self.assertIn("safeguarding", path, r["text"])
            self.assertFalse([p for p in path if p.startswith(("deliver:", "confirm:", "read:"))] or "redirect" in path,
                             (r["text"], path))
            self.assertIn("116", body(out))
            self.assertEqual([x["kind"] for x in e.mem.events(N)], ["safeguarding_referral"])

    def test_lesson_requests_not_referred(self):
        for r in DEV:
            if r["label"] != "lesson":
                continue
            e = engine()
            onboard(e)
            e.handle(N, {"text": r["text"]})
            self.assertNotIn("safeguarding", e.last["path"], r["text"])

    def test_reader_english_checked_before_routing(self):
        # the reading routes to the money lesson; its English (typos echoed) is a disclosure: referred, not delivered
        e = engine(EchoReader("money lesson grade 2", "the deputy beatn the lerners wit a hosepipe til they bled"))
        onboard(e)
        out = e.handle(N, {"text": "(a Tonga message)"})
        path = e.last["path"]
        self.assertIn("gate:reading", path)
        self.assertFalse([p for p in path if p.startswith(("deliver:", "confirm:"))], path)
        self.assertIn("116", body(out))

    def test_reader_english_checked_on_the_name_step(self):
        e = engine(EchoReader("zzzz qqqq", "a man takes naked pics of the gals"))
        e.handle(N, {"text": "hello"})
        out = e.handle(N, {"text": "Mmm kkk"})          # at the name question: never stored as a name
        self.assertIn("safeguarding", e.last["path"])
        self.assertIn("116", body(out))
        self.assertNotEqual((e.mem.get_teacher(N) or {}).get("name"), "Mmm kkk")


if __name__ == "__main__":
    unittest.main(verbosity=1)
