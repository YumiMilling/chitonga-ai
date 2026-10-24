"""Tests for the timetable, school calendar, lesson notes, planner, quick add and the weekly digest (24 Oct 2026, the
Jesuit).

    python -X utf8 tools/chitonga/harness/tests/test_timetable.py

Temporary memory, keys, sync store and logs only; no model and no translator are loaded (the digest tests use stubs of
the same interface; the real run is plugins/education/notes_digest.py on the Beelink under the heavy-job lock).
"""
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, ".."))
WEB = os.path.join(HARNESS, "web")
for p in (HARNESS, WEB):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.memory import SQLiteMemory  # noqa: E402
from core.school_calendar import SchoolCalendar  # noqa: E402
from core.timetable import TimetableStore  # noqa: E402
from plugins.education import notes_digest as ND  # noqa: E402
from plugins.education.planner import Planner, probable_names  # noqa: E402

N1 = "260970000071"
N2 = "260970000072"
T0 = "2026-10-24T08:00:00Z"
_P = None


def plugin():
    global _P
    if _P is None:
        from plugins import education
        _P = education.load()
    return _P


def at(day, hm="09:00"):
    return dt.datetime.fromisoformat(f"{day}T{hm}")


class Base(unittest.TestCase):
    def setUp(self):
        self.P, _ = plugin()
        self.mem = SQLiteMemory(":memory:")
        self.tt = TimetableStore(self.mem)
        self.cal = SchoolCalendar.for_plugin(self.P)
        for n in (N1, N2):
            self.mem.get_or_create(n)
        self.c2 = self.mem.add_class(N1, "2C", [2])
        self.c3 = self.mem.add_class(N1, "Grade 3 Blue", [3])
        self.other = self.mem.add_class(N2, "2A", [2])
        self.now = at("2026-10-26")                       # a Monday in term 3
        self.pl = Planner(self.P, self.mem, self.tt, self.cal, clock=lambda: self.now)

    def period(self, pid, day, start, cls, subject, n=N1, **kw):
        return self.tt.upsert_period(n, dict(id=pid, day=day, start=start, end=None, class_id=cls["id"],
                                             subject=subject, updated_at=T0, **kw))

    def std_week(self):
        self.period("p-mon-maths", 1, "07:30", self.c2, "maths")
        self.period("p-mon-engl3", 1, "08:10", self.c3, "english")
        self.period("p-tue-maths", 2, "07:30", self.c2, "maths")
        self.period("p-wed-maths", 3, "07:30", self.c2, "maths")


# ------------------------------------------------------------------------------------------------ calendar
class CalendarTest(Base):
    def test_term3_2026_week_and_flags(self):
        d = self.cal.day("2026-10-26")
        self.assertEqual((d["term"], d["week"]), (3, 8))
        self.assertTrue(d["school_day"])
        self.assertIn("term_end_to_confirm", d["flags"])        # no official closing date: blank, flagged
        self.assertIn("term_start_reported", d["flags"])        # term 3 start is a reported date, not a circular

    def test_term1_official_start(self):
        t = self.cal.terms(2026)[0]
        self.assertEqual((t["start"], t["start_status"]), ("2026-01-12", "official"))
        self.assertIsNone(t["end"])
        self.assertEqual(t["end_status"], "to_confirm")
        self.assertEqual(self.cal.day("2026-01-09")["break"], "between_terms")

    def test_holidays_2026(self):
        self.assertFalse(self.cal.day("2026-12-29")["school_day"])      # Prayer and Christian Heritage, SI 73/2026
        self.assertFalse(self.cal.day("2026-08-13")["school_day"])      # election day
        d = self.cal.day("2026-10-19")                                    # NOT observed in 2026 (SI 78/2015 revoked)
        self.assertTrue(d["school_day"])
        self.assertTrue(any(h["kind"] == "not_holiday" for h in self.cal.holidays_on("2026-10-18")))
        m = self.cal.day("2026-03-09")                                    # s.3 day's Monday: to confirm, not skipped
        self.assertTrue(m["school_day"])
        self.assertEqual(m["holidays"][0]["status"], "to_confirm")

    def test_2027_computed_rules(self):
        names = {h["date"]: h for h in self.cal.data["years"]["2027"]["holidays"]}
        self.assertEqual(names["2027-03-26"]["name"], "Good Friday")
        self.assertEqual(names["2027-07-05"]["name"], "Heroes Day")
        self.assertEqual(names["2027-07-06"]["name"], "Unity Day")
        self.assertEqual(names["2027-10-25"]["kind"], "substitute")     # Independence Day on a Sunday: Monday off
        self.assertTrue(all(h["status"] == "computed" for h in names.values()))
        d = self.cal.day("2027-02-10")                                    # no term dates for 2027 yet
        self.assertTrue(d["school_day"])
        self.assertIn("term_dates_to_confirm", d["flags"])
        self.assertFalse(self.cal.day("2027-10-25")["school_day"])

    def test_teacher_override_and_midterm(self):
        self.tt.set_term(N1, 2026, 3, T0, end="2026-12-04", midterm_start="2026-10-29", midterm_end="2026-10-30")
        ov = self.tt.term_overrides(N1)
        self.assertEqual(self.cal.day("2026-12-07", ov)["break"], "between_terms")
        self.assertEqual(self.cal.day("2026-10-29", ov)["break"], "midterm")
        self.assertNotIn("term_end_to_confirm", self.cal.day("2026-11-02", ov)["flags"])
        t3 = self.cal.terms(2026, ov)[2]
        self.assertEqual((t3["end"], t3["end_by"], t3["end_status"]), ("2026-12-04", "teacher", "teacher"))
        self.tt.reset_term(N1, 2026, 3)
        self.assertIsNone(self.cal.terms(2026, self.tt.term_overrides(N1))[2]["end"])
        with self.assertRaises(ValueError):
            self.tt.set_term(N1, 2026, 3, T0, start="2026-12-01", end="2026-11-01")


# ------------------------------------------------------------------------------------------------ store
class StoreTest(Base):
    def test_period_validation(self):
        bad = [dict(day=8), dict(start="7:30"), dict(start="09:00", end="08:00"), dict(subject="history"),
               dict(subject="other")]
        for b in bad:
            with self.assertRaises(ValueError, msg=b):
                self.tt.upsert_period(N1, dict(dict(id="p-bad-0001", day=1, class_id=self.c2["id"],
                                                    subject="maths", updated_at=T0), **b))
        with self.assertRaises(ValueError):                     # another teacher's class
            self.tt.upsert_period(N1, dict(id="p-x-000001", day=1, class_id=self.other["id"], subject="maths"))
        p = self.tt.upsert_period(N1, dict(id="p-oth-0001", day=5, subject="other", label="Assembly",
                                           updated_at=T0))
        self.assertEqual(p["label"], "Assembly")

    def test_last_writer_wins_and_tombstone(self):
        self.period("p-lww-0001", 1, "07:30", self.c2, "maths")
        r = self.tt.upsert_period(N1, dict(id="p-lww-0001", day=2, class_id=self.c2["id"], subject="maths",
                                           updated_at="2026-10-23T08:00:00Z"))
        self.assertTrue(r["stale"])
        self.assertEqual(self.tt.period("p-lww-0001")["day"], 1)
        self.tt.delete_period(N1, "p-lww-0001", "2026-10-24T09:00:00Z")
        r = self.tt.upsert_period(N1, dict(id="p-lww-0001", day=3, class_id=self.c2["id"], subject="maths",
                                           updated_at="2026-10-24T08:30:00Z"))
        self.assertTrue(r.get("stale"))
        self.assertEqual(self.tt.periods(N1), [])
        with self.assertRaises(ValueError):
            self.tt.upsert_period(N2, dict(id="p-lww-0001", day=1, subject="maths",
                                           updated_at="2026-10-25T08:00:00Z"))

    def test_copy_day_is_repeatable(self):
        self.std_week()
        ids = ["c-tue-0001", "c-tue-0002"]
        self.tt.copy_day(N1, 1, 4, ids, True, "2026-10-24T09:00:00Z")
        self.tt.copy_day(N1, 1, 4, ids, True, "2026-10-24T09:00:00Z")        # a retried copy: the same two rows
        thu = [p for p in self.tt.periods(N1) if p["day"] == 4]
        self.assertEqual(sorted(p["id"] for p in thu), ids)
        self.assertEqual([p["subject"] for p in thu], ["maths", "english"])

    def test_note_partial_update_and_translation_reset(self):
        n = self.tt.upsert_note(N1, dict(id="n-0000001", class_id=self.c2["id"], slot_date="2026-10-26",
                                         text="Counting went well", tags=["worked"], updated_at=T0))
        self.assertEqual(n["tags"], ["worked"])
        self.tt.set_translation("n-0000001", "x", "toi", "translator")
        n = self.tt.upsert_note(N1, dict(id="n-0000001", item="lesson:g2.2.1-counting",
                                         updated_at="2026-10-24T08:01:00Z"))
        self.assertEqual((n["text"], n["tags"], n["text_en"]), ("Counting went well", ["worked"], "x"))
        n = self.tt.upsert_note(N1, dict(id="n-0000001", text="Counting went badly", updated_at="2026-10-24T08:02:00Z"))
        self.assertNotIn("text_en", n)
        with self.assertRaises(ValueError):
            self.tt.upsert_note(N1, dict(id="n-0000002", tags=["great"]))
        with self.assertRaises(ValueError):
            self.tt.upsert_note(N2, dict(id="n-0000001", text="mine now", updated_at="2026-10-25T00:00:00Z"))


# ------------------------------------------------------------------------------------------------ planner
class PlannerTest(Base):
    def test_slots_skip_holidays_and_weekends(self):
        self.std_week()
        w = self.pl.slots(N1, "2026-12-28", "2027-01-03", at("2026-12-28"))
        days = {d["date"]: d for d in w["days"]}
        self.assertTrue(days["2026-12-28"]["slots"])                     # Monday: lessons (term end unknown)
        self.assertEqual(days["2026-12-29"]["slots"], [])                  # holiday: none
        self.assertFalse(days["2026-12-29"]["school_day"])
        self.assertEqual(days["2027-01-01"]["slots"], [])                  # New Year

    def test_projection_resume_and_funda(self):
        self.std_week()
        w = self.pl.week(N1)
        maths = [s for d in w["days"] for s in d["slots"] if s["subject"] == "maths"]
        seq = self.pl.sequence(2, "maths")
        self.assertEqual([s["suggestion"]["pick"]["id"] for s in maths], seq[:3])
        self.assertEqual(maths[0]["suggestion"]["pick"]["source"], "resume")
        fu = maths[0]["suggestion"]["by_grade"][0]["funda"]
        self.assertIn("Funda estimate", fu["label"])
        self.assertEqual((fu["term"], fu["week"]), (3, 8))
        self.assertTrue(all(2 in (self.P.items[i["id"]].get("grades") or [2]) for i in fu["items"]))

    def test_taught_moves_the_chain(self):
        self.std_week()
        seq = self.pl.sequence(2, "maths")
        r = self.pl.taught(N1, self.c2["id"], seq[0], "2026-10-26", "p-mon-maths", "n-taught-01", T0)
        self.assertEqual(r["note"]["item"], seq[0])
        self.assertTrue(r["note"]["taught"])
        w = self.pl.week(N1)
        maths = [s for d in w["days"] for s in d["slots"] if s["subject"] == "maths"]
        self.assertEqual(maths[0]["taught"]["id"], seq[0])
        self.assertEqual(maths[1]["suggestion"]["pick"]["id"], seq[1])
        # home (the existing app) agrees on where the class is
        p = self.mem.position(self.c2["id"], "education", "maths")
        self.assertTrue(p["item"] in (seq[0], seq[1]))

    def test_taught_within_and_at_end_of_unit(self):
        seq = self.pl.sequence(2, "maths")
        same = next(i for i in range(len(seq) - 1) if self.P.unit_of[seq[i]] == self.P.unit_of[seq[i + 1]])
        r = self.pl.taught(N1, self.c2["id"], seq[same])
        self.assertEqual((r["position"]["item"], r["position"]["finished"]), (seq[same + 1], False))
        last = next(i for i in range(len(seq) - 1) if self.P.unit_of[seq[i]] != self.P.unit_of[seq[i + 1]])
        r = self.pl.taught(N1, self.c2["id"], seq[last])
        self.assertTrue(r["position"]["finished"])
        with self.assertRaises(ValueError):
            self.pl.taught(N1, self.other["id"], seq[0])

    def test_planned_item_wins(self):
        self.std_week()
        self.tt.upsert_note(N1, dict(id="n-plan-0001", class_id=self.c2["id"], slot_date="2026-10-27",
                                     period_id="p-tue-maths", planned_item="lesson:g2.2.2-addition", updated_at=T0))
        w = self.pl.week(N1)
        tue = next(s for d in w["days"] if d["date"] == "2026-10-27" for s in d["slots"])
        self.assertEqual(tue["suggestion"]["pick"], dict(self.pl.item_ref("lesson:g2.2.2-addition"),
                                                         source="planned"))

    def test_past_slots_have_no_suggestion(self):
        self.std_week()
        w = self.pl.slots(N1, "2026-10-19", "2026-10-23", self.now)
        self.assertTrue(all("suggestion" not in s for d in w["days"] for s in d["slots"]))

    def test_quick_add(self):
        self.std_week()
        r = self.pl.parse_quick(N1, "2C counting didn't work, needs bundles again", at("2026-10-26", "10:00"))
        self.assertEqual(r["class"]["label"], "2C")
        self.assertEqual(r["tags"], ["didnt", "repeat"])
        self.assertIn("counting", r["item"]["id"])
        self.assertEqual(r["slot"]["period_id"], "p-mon-maths")
        r = self.pl.parse_quick(N1, "Grade 3: Mutinta needs help with reading", at("2026-10-26", "10:00"))
        self.assertEqual((r["class"]["label"], r["attention"]), ("Grade 3 Blue", "learner"))
        self.assertIn("attention", r["tags"])
        r = self.pl.parse_quick(N1, "Most of the class needs more practice with sounds", at("2026-10-27", "06:00"))
        self.assertIsNone(r["class"])                                      # two classes, no period on yet
        r = self.pl.parse_quick(N1, "Most of the class needs more practice", at("2026-10-26", "08:20"))
        self.assertEqual(r["class"]["label"], "Grade 3 Blue")              # the period on now
        self.assertTrue(self.pl.parse_quick(N1, "a learner told me her uncle beats her", self.now)["safeguarding"])

    def test_quick_add_model_fills_only_from_lists(self):
        seen = {}

        def model(text, classes, cands):
            seen.update(classes=[c["id"] for c in classes], cands=cands)
            return {"class_id": 999, "item": "lesson:not-listed", "tags": ["idea", "bogus"]}
        r = self.pl.parse_quick(N1, "something about the morning", at("2026-10-27", "06:00"), model=model)
        self.assertIsNone(r["class"])
        self.assertIsNone(r["item"])
        self.assertEqual(r["tags"], ["idea"])
        self.assertEqual(r["source"], "rules+model")

    def test_suggested_tray(self):
        self.std_week()
        seq = self.pl.sequence(2, "maths")
        self.tt.upsert_note(N1, dict(id="n-rep-00001", class_id=self.c2["id"], slot_date="2026-10-22",
                                     item=seq[0], tags=["repeat", "didnt"],
                                     text="Learners mixed up the numbers when counting, too fast", updated_at=T0))
        tray = self.pl.suggested(N1, self.now)
        kinds = [t["kind"] for t in tray]
        self.assertIn("next", kinds)
        self.assertIn("repeat", kinds)
        self.assertIn("tip", kinds)
        tip = next(t for t in tray if t["kind"] == "tip")
        self.assertIn(tip["match"]["kind"], ("strategy", "tip"))
        self.assertEqual(tip["because"]["notes"], ["n-rep-00001"])
        self.pl.taught(N1, self.c2["id"], seq[0], "2026-10-26", "p-mon-maths", "n-again-001", T0)
        self.assertNotIn("repeat", [t["kind"] for t in self.pl.suggested(N1, self.now)])

    def test_names(self):
        self.assertTrue({"Mutinta", "Gift", "Chipo"} <= set(probable_names("Mutinta needs help; Gift and Chipo were "
                                                                           "absent")))
        self.assertEqual(probable_names("Counting went well. Reading groups were noisy."), [])


# ------------------------------------------------------------------------------------------------ digest
class StubModel:
    name = "stub"

    def __init__(self, reply, pick=None):
        self.reply = reply
        self.pick = pick
        self.calls = []

    def chat(self, messages, schema, max_tokens=900):
        self.calls.append((messages, schema))
        if "pick" in schema["properties"]:
            ids = schema["properties"]["pick"]["anyOf"][0]["enum"]
            return {"pick": self.pick if self.pick in ids else (self.pick if self.pick else ids[0])}, "{}"
        return self.reply, json.dumps(self.reply)


class DigestTest(Base):
    def notes(self):
        self.tt.upsert_note(N1, dict(id="n-d-000001", class_id=self.c2["id"], slot_date="2026-10-26",
                                     item="lesson:g2.2.1-counting", tags=["didnt"],
                                     text="Counting past 20 did not work, they lost track", updated_at=T0))
        self.tt.upsert_note(N1, dict(id="n-d-000002", class_id=self.c2["id"], slot_date="2026-10-27",
                                     tags=["worked"], text="Bottle tops for counting worked well", updated_at=T0))
        self.tt.upsert_note(N1, dict(id="n-d-000003", class_id=self.c2["id"], slot_date="2026-10-28",
                                     tags=["attention"], attention="learner",
                                     text="Mutinta cannot yet count to ten", updated_at=T0))
        self.tt.upsert_note(N1, dict(id="n-d-000004", class_id=self.c2["id"], slot_date="2026-10-28",
                                     tags=["idea"], text="Try groups of five next time", updated_at=T0))

    def test_checks_drop_unsupported_items(self):
        self.notes()
        reply = {"worked": [{"text": "Bottle tops helped with counting", "notes": ["N2"]},
                            {"text": "Counting past 20 went well", "notes": ["N1"]}],      # N1 says it did NOT work
                 "didnt": [{"text": "Counting past 20 did not work", "notes": ["N1"]}],
                 "ideas": [{"text": "Use a song about the weather", "notes": ["N4"]}],    # no shared word
                 "attention": [{"text": "Mutinta needs help counting to ten", "notes": ["N3"]}],
                 "repeat": [{"text": "Repeat counting", "notes": ["N9"]}],                # unknown id
                 "themes": []}
        job = ND.DigestJob(self.P, self.mem, self.tt, model=StubModel(reply), translator=None)
        d = job.digest(N1, self.c2["id"], "2026-10-26")
        kept = {c: [it["text"] for it in d["sections"][c]] for c in ND.CATS}
        self.assertEqual(kept["worked"], ["Bottle tops helped with counting"])
        self.assertEqual(kept["didnt"], ["Counting past 20 did not work"])
        self.assertEqual(kept["ideas"], [])
        self.assertEqual(kept["repeat"], [])
        self.assertEqual(len(d["dropped"]), 3)
        self.assertEqual([x["id"] for x in d["not_summarised"]], ["n-d-000004"])
        self.assertTrue(d["ai"])
        for m in d["matches"]:
            self.assertIn(m["match"]["id"], m["candidates"])
        self.assertEqual(self.tt.digest(N1, self.c2["id"], "2026-10-26")["week"], "2026-10-26")

    def test_malformed_reply_is_skipped_not_crashed(self):
        # reader bake-off, 24 Oct 2026: a server without schema enforcement returned string items and DigestJob raised.
        # Now the model digest is skipped with a log line and the teacher's tagged notes are shown; run() never stops.
        self.notes()
        for reply in ({"worked": ["Bottle tops helped"], "didnt": "Counting past 20"}, ["not", "an", "object"],
                      {"worked": [{"text": "Bottle tops", "notes": [{"id": "N2"}]}]}):
            log = []
            job = ND.DigestJob(self.P, self.mem, self.tt, model=StubModel(reply), translator=None,
                               log=lambda *x: log.append(" ".join(map(str, x))))
            d = job.digest(N1, self.c2["id"], "2026-10-26")
            self.assertFalse(d["ai"])
            self.assertEqual(d.get("skipped"), "model reply not in the schema's shape")
            self.assertTrue(any("model digest skipped" in x for x in log), log)
            self.assertTrue(any(d["sections"][c] for c in ND.CATS))     # the tagged notes are still shown
        log = []
        job = ND.DigestJob(self.P, self.mem, self.tt, model=None, log=lambda *x: log.append(" ".join(map(str, x))))
        job.digest = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        self.assertEqual(job.run(week="2026-10-26"), [])
        self.assertTrue(any("digest skipped for teacher" in x for x in log), log)

    def test_model_pick_cannot_invent(self):
        self.notes()
        reply = {c: [] for c in ND.CATS}
        reply["didnt"] = [{"text": "Counting past 20 did not work", "notes": ["N1"]}]
        job = ND.DigestJob(self.P, self.mem, self.tt, model=StubModel(reply, pick="strategy:made-up"))
        d = job.digest(N1, self.c2["id"], "2026-10-26")
        for m in d["matches"]:
            self.assertIn(m["match"]["id"], m["candidates"])

    def test_no_model_groups_tags(self):
        self.notes()
        d = ND.DigestJob(self.P, self.mem, self.tt, model=None).digest(N1, self.c2["id"], "2026-10-27")
        self.assertFalse(d["ai"])
        self.assertEqual(d["sections"]["ideas"][0]["notes"], ["n-d-000004"])
        self.assertTrue(all(not it["ai"] for c in ND.CATS for it in d["sections"][c]))

    def test_translator_cached_on_note(self):
        class Tr:
            def note_english(self, text):
                return ("Slowly. One stone, one number.", "toi", "translator") if text.startswith("Asyoonto") \
                    else (text, "en", None)
        self.tt.upsert_note(N1, dict(id="n-toi-00001", class_id=self.c2["id"], slot_date="2026-10-26", tags=["worked"],
                                     text="Mubukulo Leza wakalenga julu anyika yoonse.", updated_at=T0))
        reply = {c: [] for c in ND.CATS}
        reply["worked"] = [{"text": "Counting slowly, one stone per number, worked", "notes": ["N1"]}]
        sm = StubModel(reply)
        d = ND.DigestJob(self.P, self.mem, self.tt, model=sm, translator=Tr()).digest(N1, self.c2["id"], "2026-10-26")
        self.assertEqual(d["translated"], 1)
        self.assertIn("One stone, one number", sm.calls[0][0][1]["content"])
        self.assertEqual(len(d["sections"]["worked"]), 1)
        self.assertEqual(self.tt.note("n-toi-00001")["text"], "Mubukulo Leza wakalenga julu anyika yoonse.")

    def test_aggregate_strips_names_and_learner_notes(self):
        self.notes()
        self.tt.upsert_note(N1, dict(id="n-d-000005", class_id=self.c2["id"], slot_date="2026-10-28",
                                     item="lesson:g2.2.1-counting", tags=["didnt"],
                                     text="Chipo and Gift could not follow the counting game", updated_at=T0))
        self.tt.upsert_note(N2, dict(id="n-d-000006", class_id=self.other["id"], slot_date="2026-10-28",
                                     item="lesson:g2.2.1-counting", tags=["didnt", "attention"], attention="learner",
                                     text="Mwiinga did not work at all", updated_at=T0))
        r = ND.DigestJob(self.P, self.mem, self.tt).aggregate("2026-10-20", "2026-10-31", path=None)
        blob = json.dumps(r)
        for nm in ("Mutinta", "Chipo", "Gift", "Mwiinga", N1, N2):
            self.assertNotIn(nm, blob)
        les = next(x for x in r["lessons"] if x["item"] == "lesson:g2.2.1-counting")
        self.assertEqual((les["n_notes"], les["n_teachers"]), (2, 1))
        self.assertIn("[learner]", blob)


class SupportMatchTest(Base):
    """24 Oct run 2 of the 20-note test: own lesson first, no other subject, no lone-word lures, subject words count"""
    def test_rules(self):
        s = self.pl.support
        self.assertEqual(s.match("The water experiment did not work because we had no clear bottles",
                                 subject="science"), [])                         # 'clear' alone: no 'Clear means safe'
        own = s.match("Counting past 20 did not work, they lost track", lesson="lesson:g2.2.1-counting",
                      subject="maths")
        self.assertTrue(own and all(c.get("own_lesson") for c in own[:3]))
        self.assertEqual({c["lesson"] for c in own[:3]}, {"lesson:g2.2.1-counting"})
        noisy = s.match("Reading groups were noisy; the back group did not read at all.", subject="english")
        self.assertIn("noise-07", [c["id"] for c in noisy])                      # noisy -> noise
        for c in s.match("Reading aloud: many learners read word by word without expression.", subject="english"):
            self.assertTrue(c["kind"] == "tip" or c.get("subject") == "english")
            self.assertTrue(c["kind"] == "tip" or c["title_hit"])
        self.assertNotIn("g3.E.3.3-punctuation-intro#m2", [c["id"] for c in s.match(
            "Reading aloud: many learners read word by word without expression.", subject="english")])

    def test_subject_words_in_the_check(self):
        n = [{"text": "Counting past 20 did not work", "tags": '["didnt"]', "subject": "maths",
              "item": "lesson:g2.2.1-counting"}]
        ok, _ = ND.supported("themes", {"text": "Some concepts in maths required more time"}, n, self.P)
        self.assertTrue(ok)
        ok, why = ND.supported("themes", {"text": "Weather songs were popular"}, n, self.P)
        self.assertFalse(ok)


class NightlyTest(unittest.TestCase):
    """the scheduled run: skips when the Beelink lock is held, runs once per night, releases the lock"""
    def test_skip_then_run_once(self):
        sys.path.insert(0, os.path.normpath(os.path.join(HARNESS, "..", "llm")))
        import beelink_lock as BL
        tmp = tempfile.mkdtemp()
        old = (BL.LOCK, ND.NIGHT_MARK, ND.NIGHT_LOG, ND.BACKLOG)
        try:
            BL.LOCK = os.path.join(tmp, ".lock")
            ND.NIGHT_MARK, ND.NIGHT_LOG = os.path.join(tmp, "mark.json"), os.path.join(tmp, "night.log")
            ND.BACKLOG = os.path.join(tmp, "backlog.json")
            db = os.path.join(tmp, "m.sqlite")
            with open(BL.LOCK, "w") as f:
                json.dump({"by": "f18-killtest", "at": __import__("time").time()}, f)
            args = ["--nightly", "--db", db, "--no-model", "--no-translator"]
            self.assertEqual(ND.main(args), "skipped")
            self.assertIn("SKIPPED, the Beelink lock is held by f18-killtest", open(ND.NIGHT_LOG).read())
            self.assertTrue(os.path.exists(BL.LOCK))                         # someone else's lock is left alone
            os.remove(BL.LOCK)
            r = ND.main(args)
            self.assertIsInstance(r, dict)
            self.assertFalse(os.path.exists(BL.LOCK))                         # released after the run
            self.assertEqual(ND.main(args), "done before")                   # the 02:40 retry does nothing
            with open(BL.LOCK, "w") as f:                                    # a stale lock (> 4 h) is taken over
                json.dump({"by": "dead-job", "at": 0}, f)
            self.assertTrue(ND.try_lock("notes-digest"))
        finally:
            BL.LOCK, ND.NIGHT_MARK, ND.NIGHT_LOG, ND.BACKLOG = old
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ web API + HTTP
class WebTest(unittest.TestCase):
    def setUp(self):
        import api as A
        self.A = A
        self.tmp = tempfile.mkdtemp()
        P, hooks = plugin()
        self.keys = A.KeyStore(os.path.join(self.tmp, "keys.json"))
        self.key = self.keys.create("T71", N1, "Test", tester=False)
        self.key2 = self.keys.create("T72", N2, "Other", tester=False)
        clock = lambda: dt.datetime(2026, 10, 26, 7, 0, tzinfo=dt.timezone.utc)   # 09:00 in Lusaka  # noqa: E731
        self.api = A.WebAPI(plugin=P, hooks=hooks, memory=SQLiteMemory(os.path.join(self.tmp, "m.sqlite")),
                            keys=self.keys, logs=os.path.join(self.tmp, "logs"),
                            sync_db=os.path.join(self.tmp, "s.sqlite"), clock=clock)
        self.who = self.api.teacher(self.key)
        self.who2 = self.api.teacher(self.key2)
        self.api.onboard(self.who, {"name": "Test", "classes": [{"label": "2C", "grades": [2]}]})
        self.api.onboard(self.who2, {"name": "Other", "classes": [{"label": "2A", "grades": [2]}]})
        self.cid = self.api.me(self.who)["classes"][0]["id"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def sync(self, who, ops):
        return self.api.sync_batch(who, ops)

    def test_sync_ops_idempotent(self):
        ops = [{"client_id": "op-000000001", "type": "period", "at": T0,
                "period": {"id": "p-mon-000001", "day": 1, "start": "07:30", "end": "08:10", "class_id": self.cid,
                           "subject": "maths", "updated_at": T0}},
               {"client_id": "op-000000002", "type": "note", "at": T0,
                "note": {"id": "n-web-000001", "class_id": self.cid, "slot_date": "2026-10-26",
                         "period_id": "p-mon-000001", "tags": ["didnt"], "text": "Counting did not work",
                         "updated_at": T0}}]
        r1 = self.sync(self.who, ops)
        self.assertTrue(all(x["ok"] for x in r1), r1)
        r2 = self.sync(self.who, ops)
        self.assertTrue(all(x.get("duplicate") for x in r2))
        self.assertEqual(len(self.api.notes(self.who, "2026-10-01", "2026-10-31")["notes"]), 1)
        r = self.sync(self.who2, [{"client_id": "op-000000003", "type": "note", "note": {
            "id": "n-web-000001", "text": "x", "updated_at": "2026-10-25T00:00:00Z"}}])
        self.assertFalse(r[0]["ok"])

    def test_note_safeguarding_withholds_text(self):
        r = self.sync(self.who, [{"client_id": "op-sg-000001", "type": "note", "note": {
            "id": "n-sg-000001", "class_id": self.cid, "text": "a girl told me her uncle beats her",
            "updated_at": T0}}])
        res = r[0]["result"]
        self.assertIn("116", res["safeguarding"]["reply"])
        self.assertIsNone(res["note"]["text"])
        self.assertEqual(res["note"]["withheld"], "safeguarding")
        self.assertIn("safeguarding_referral", [e["kind"] for e in self.api.mem.events(N1)])
        self.assertNotIn("uncle", json.dumps(self.api.tt.note("n-sg-000001")))

    def test_taught_and_today(self):
        self.sync(self.who, [{"client_id": "op-p-000001", "type": "period", "period": {
            "id": "p-mon-000001", "day": 1, "start": "07:30", "class_id": self.cid, "subject": "maths",
            "updated_at": T0}}])
        t = self.api.today(self.who)
        self.assertEqual(t["date"], "2026-10-26")
        pick = t["slots"][0]["suggestion"]["pick"]
        r = self.sync(self.who, [{"client_id": "op-t-000001", "type": "taught", "class_id": self.cid,
                                  "item": pick["id"], "slot_date": "2026-10-26", "period_id": "p-mon-000001",
                                  "note_id": "n-t-0000001", "updated_at": T0}])
        self.assertTrue(r[0]["ok"], r)
        self.assertEqual(self.api.today(self.who)["slots"][0]["taught"]["id"], pick["id"])
        home = self.api.home(self.who)
        self.assertTrue(home["classes"][0]["next"])

    def test_digest_view_resolves_notes(self):
        self.sync(self.who, [{"client_id": "op-n-000001", "type": "note", "note": {
            "id": "n-v-0000001", "class_id": self.cid, "slot_date": "2026-10-26", "tags": ["idea"],
            "text": "Try groups of five", "updated_at": T0}}])
        ND.DigestJob(self.api.P, self.api.mem, self.api.tt).digest(N1, self.cid, "2026-10-26")
        v = self.api.digest(self.who)["digests"][0]
        self.assertNotIn("raw", v)
        self.assertEqual(v["notes_index"]["n-v-0000001"]["text"], "Try groups of five")
        self.assertEqual(self.api.digest(self.who2)["digests"], [])

    def test_http(self):
        import server as SV
        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer(("127.0.0.1", 0), SV.make_handler(self.api, {"127.0.0.1", "localhost"}))
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def call(path, key=None, body=None):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"X-Key": key or "", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=20) as f:
                    return f.status, json.loads(f.read())
            except urllib.error.HTTPError as e:
                return e.code, None
        try:
            for path in ("/api/timetable", "/api/calendar?from=2026-12-20&to=2027-01-10", "/api/today",
                         "/api/week?start=2026-10-26", "/api/suggested", "/api/notes", "/api/digest"):
                code, body = call(path, self.key)
                self.assertEqual(code, 200, path)
            self.assertEqual(call("/api/today")[0], 401)
            self.assertEqual(call("/api/today?date=26-10-2026", self.key)[0], 400)
            self.assertEqual(call("/api/calendar?from=2026-01-01&to=2026-12-31", self.key)[0], 400)
            code, body = call("/api/quickadd", self.key, {"text": "2C sets went well"})
            self.assertEqual(code, 200)
            self.assertEqual(body["parse"]["tags"], ["worked"])
            code, body = call("/api/timetable", self.key)
            self.assertEqual(body["calendar"]["terms"][0]["start"], "2026-01-12")
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=1)
