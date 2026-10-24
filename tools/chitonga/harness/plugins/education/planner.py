"""The teacher's week: timetable slots with a suggested lesson, Today, the Suggested tray, quick-add notes and "Taught
it" (24 Oct 2026, the Jesuit).

Chris, 24 Oct 2026: the timetable and lesson notes ("they can click on lessons in a calendar to keep notes"); then
"smart, easy, not overloaded ... Microsoft To Do" (web/DESIGN_DIRECTION.md, last section): a Today screen like My Day,
a Suggested tray (the lightbulb), one quick-add line, "Taught it" as one tap. This module is the logic behind those; it
writes only through core/timetable.py and core/memory.py (positions), and never writes content.

A SLOT'S SUGGESTION (a suggestion the teacher can change or ignore, never a plan imposed):
  resume   where this class left off in this subject (memory positions, shared with WhatsApp and "Taught it"),
           projected forward: the k-th upcoming slot of a subject gets the k-th lesson after the resume point, in the
           menu's order. One lesson per period is itself an assumption; "Taught it" moves the chain on.
  funda    the syllabus sub-topics Funda's pacing puts in this term and week, and the served lessons linked to them,
           ALWAYS labelled "Funda estimate" (the CDC book gives no weeks; plugins/education/syllabus.py).
  planned  the teacher's own pick for that slot (a note's planned_item) wins over both.
  pick     planned, else resume, else the first Funda lesson; `source` says which.
Past slots carry what the teacher recorded (taught lesson, tags), no suggestion.

QUICK ADD ("2C counting didn't work, needs bundles again"): rules first (class by label / grade / the period on now,
tags by phrase, lesson by the lexical index over the class's grades, else the slot's lesson), then optionally the
standard model reading English (translator first for Tonga) picking ONLY among the listed classes, lessons and tags.
The parse is returned for the teacher to see and correct; nothing is saved by parsing. The note text is kept verbatim.

SUPPORT MATCHING (the tray's tip for last week's "didn't work", and the digest's matches): lexical retrieval over the
plug-in's strategies (common mistakes + fixes, per lesson) and teaching tips; a candidate list, never a new strategy.
"""
import datetime as dt
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, "..", ".."))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from core import gates as G  # noqa: E402
from core.intake import grades_in  # noqa: E402
from core.memory import norm_number  # noqa: E402
from core.retrieval import Index, L  # noqa: E402
from core.school_calendar import SchoolCalendar, d_, monday  # noqa: E402
from core.timetable import TAGS  # noqa: E402

AREA_OF = {"english": "english", "chitonga": "chitonga", "maths": "maths_science", "science": "maths_science",
           "cts": "cts"}
FUNDA_LABEL = "Funda estimate (the Ministry's syllabus gives no weeks)"
SUGGESTION_NOTE = "A suggestion from your timetable, where you left off and Funda's pacing. Change it or ignore it."
MAX_RANGE_DAYS = 120
RESUME_LABEL = {"continue": "where you left off", "next": "next in the menu", "start": "first lesson in the menu"}

SUBJECT_RX = [
    ("maths", r"maths?|mathematics|numeracy|counting|count|numbers?|addition|adding|subtract\w*|place value|shapes?|"
              r"division|multipl\w*|sets?|fractions?|measur\w*|money|time"),
    ("science", r"science|plants?|animals?|body|water|weather|hygiene|materials?|soil|food|senses"),
    ("english", r"english|phonics|reading|read|letters?|sounds?|spelling|writing|handwriting|sight words?|story|"
                r"stories|vocabulary|grammar|literacy"),
    ("chitonga", r"chitonga|citonga|tonga"),
    ("cts", r"cts|creative|art|music|drawing|craft|dance|physical education|pe"),
]
SUBJECT_RX = [(s, re.compile(r"(?i)(?<![a-z])(?:" + rx + r")(?![a-z])")) for s, rx in SUBJECT_RX]
TAG_RX = {
    "didnt": re.compile(r"(?i)didn'?t\s+work|did\s+not\s+work|not\s+work(ing|ed)?|didn'?t\s+go\s+well|did\s+not\s+go\s+"
                        r"well|failed|flopped|too\s+(hard|difficult|fast|long)|confus\w*|lost\s+(them|the\s+class)|"
                        r"could(n'?t|\s+not)\s+(follow|get|understand|do)|struggl\w*|chaos|chaotic|noisy|bored|"
                        r"went\s+badly|was\s+a\s+mess"),
    "worked": re.compile(r"(?i)(?<![a-z'])worked|went\s+well|good\s+lesson|enjoyed|loved\s+it|loved|great|success\w*|"
                         r"understood|got\s+it|they\s+liked|liked\s+it|well\s+done|easy\s+for\s+them"),
    "idea": re.compile(r"(?i)(?<![a-z])(idea|next\s+time|try|could\s+use|maybe|what\s+if|should|plan\s+to|i\s+will|"
                       r"bring|use\s+(bottle|stones|sticks|bundles|cards))(?![a-z])"),
    "repeat": re.compile(r"(?i)(?<![a-z])(again|repeat\w*|re-?teach|revis\w*|redo|go\s+over|one\s+more\s+time|"
                        r"more\s+practi[cs]e|practi[cs]e\s+more)(?![a-z])"),
    "attention": re.compile(r"(?i)needs?\s+(extra\s+)?(help|attention|support)|attention|falling\s+behind|behind|"
                            r"absent|special\s+(needs|attention)|can'?t\s+(yet\s+)?(read|write|count)|cannot\s+"
                            r"(yet\s+)?(read|write|count)|struggles?\s+with|watch\s+\w+|sit\s+\w+\s+(at\s+the\s+)?front"),
}
CLASS_WORDS = re.compile(r"(?i)\b(class|they|them|most|many|some|several|few|half|all|the\s+learners|the\s+pupils|"
                         r"the\s+children|everyone|nobody)\b")
NAME_RX = re.compile(r"(?<![A-Za-z])([A-Z][a-z]{2,})(?![A-Za-z])")
NOT_NAMES = {"Grade", "English", "Tonga", "Chitonga", "Maths", "Math", "Science", "Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday", "Term", "Week", "Zambia", "Lusaka", "Choma", "Monze", "Kalomo",
             "Southern", "Province", "I", "CTS", "Funda", "Today", "Tomorrow", "Yesterday", "The", "They", "Most",
             "Some", "Many", "Next", "Bundles", "Phonics", "Reading", "Class", "Group", "Ok", "OK"}


def iso(d):
    return d.isoformat() if d else None


class SupportIndex:
    """lexical candidates from strategies (per-lesson mistakes, their fix) and teaching tips (title, when, steps): IDF
    over the library's content words, a bonus for a strategy of the note's own lesson. Returns candidates only."""

    def __init__(self, P, tips=None):
        self.P = P
        self.docs = []
        for lesson, s in (P.strategies or {}).items():
            for m in s.get("mistakes", []):
                txt = " ".join([m.get("title") or "", m.get("text") or "", m.get("fix") or ""])
                unit = P.units.get(P.unit_of.get(lesson) or "") or {}
                self.docs.append({"kind": "strategy", "id": m["id"], "lesson": lesson, "title": m.get("title"),
                                  "fix": m.get("fix"), "subject": unit.get("subject") or
                                  (P.items.get(lesson) or {}).get("subject"), "words": set(L.content_words(txt)),
                                  "title_words": set(L.content_words(m.get("title") or ""))})
        for t in tips if tips is not None else load_tips(P):
            txt = " ".join([t.get("title") or "", t.get("when") or "", t.get("topic") or ""] + list(t.get("steps") or []))
            self.docs.append({"kind": "tip", "id": t["id"], "topic": t.get("topic"), "title": t.get("title"),
                              "when": t.get("when"), "words": set(L.content_words(txt)),
                              "title_words": set(L.content_words(t.get("title") or ""))})
        self.weak = set(L.content_words(self.WEAK_WORDS))
        self.df = {}
        for d in self.docs:
            for w in d["words"]:
                self.df[w] = self.df.get(w, 0) + 1

    def idf(self, w):
        return math.log((1 + len(self.docs)) / (1 + self.df.get(w, 0))) + 1.0

    def expand(self, ws):
        """a few derived forms the stemmer does not join, kept only when the form is in the index's vocabulary:
        noisy -> noise, slowly -> slow, carrying -> carry (the 20-note test: 'noisy' never met the noise tips)"""
        out = list(ws)
        for w in ws:
            forms = []
            if w.endswith("y") and len(w) > 4:
                forms += [w[:-1] + "e", w[:-1]]
            if w.endswith("ly") and len(w) > 5:
                forms.append(w[:-2])
            if w.endswith("ing") and len(w) > 6:
                forms += [w[:-3], w[:-3] + "e"]
            for f in forms:
                if f in self.df and f not in out:
                    out.append(f)
        return out

    def search(self, text, lesson=None, k=5, min_score=1.5):
        q = self.expand(list(dict.fromkeys(L.content_words(text))))
        if not q:
            return []
        out = []
        for d in self.docs:
            hits = [w for w in q if w in d["words"]]
            if not hits:
                continue
            strong = [w for w in hits if w not in self.weak]
            sc = sum(self.idf(w) for w in hits) + 0.5 * sum(1 for w in hits if w in d["title_words"])
            if d["kind"] == "strategy" and lesson and d["lesson"] == lesson:
                sc += 2.0
            if sc >= min_score:
                out.append((round(sc, 2), len(strong), any(w in d["title_words"] for w in strong), d))
        out.sort(key=lambda r: (-r[0], r[3]["kind"], r[3]["id"]))
        return [dict({k2: v for k2, v in d.items() if k2 not in ("words", "title_words")}, score=s, hits=h,
                     title_hit=th) for s, h, th, d in out[:k]]

    # 24 Oct 2026 (the 20-note digest test: 2 of 3 matches were off, one lured by the lone word "clear"):
    # candidates prefer the note's OWN lesson, a strategy from another subject is never offered, and a candidate from
    # elsewhere needs at least MIN_HITS distinct shared content words.
    MIN_HITS = 2
    # words that say nothing about WHICH problem (counted in the score, not as a shared word)
    WEAK_WORDS = ("work worked working again after start started starting time times class lesson lessons learner "
                  "learners pupil pupils child children day days week today need needs help did not too most all many "
                  "some first next back still well good")

    def match(self, text, lesson=None, subject=None, k=6):
        """the candidate list for one problem: the own lesson's common mistakes first (all of them, best scored first,
        at most 3), then other strategies of the SAME subject and teaching tips sharing >= MIN_HITS content words"""
        scored = {c["id"]: c for c in self.search(text, lesson=lesson, k=40, min_score=0.0)}
        own = []
        if lesson:
            for d in self.docs:
                if d["kind"] == "strategy" and d["lesson"] == lesson:
                    c = scored.get(d["id"]) or dict({k2: v for k2, v in d.items() if k2 not in ("words", "title_words")},
                                                    score=0.0, hits=0)
                    own.append(dict(c, own_lesson=True))
            own.sort(key=lambda c: (-c["score"], c["id"]))
            own = own[:3]
        rest = []
        for c in scored.values():
            if any(o["id"] == c["id"] for o in own) or c["hits"] < self.MIN_HITS:
                continue
            if c["kind"] == "strategy" and subject and c.get("subject") and c["subject"] != subject:
                continue
            if c["kind"] == "strategy" and not c.get("title_hit"):
                continue                 # another lesson's mistake must share a word with its TITLE (the problem's
                                         # name): 'read word by word' is not 'Missing list commas' (20-note test, run 2)
            rest.append(c)
        rest.sort(key=lambda c: (-c["score"], c["kind"], c["id"]))
        return (own + rest)[:k]

    def best(self, text, lesson=None, subject=None, min_score=1.5):
        """the no-model choice: an own-lesson mistake sharing a content word, else the top other candidate (it already
        shares >= MIN_HITS words); None when nothing qualifies"""
        for c in self.match(text, lesson, subject):
            if c.get("own_lesson") and c["hits"] >= 1 and c["score"] >= min_score:
                return c
            if not c.get("own_lesson") and c["score"] >= min_score:
                return c
        return None


def load_tips(P):
    p = os.path.join(getattr(P, "dir", "") or "", "teaching_tips.jsonl")
    out = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    return out


class Planner:
    def __init__(self, P, mem, tt, calendar=None, syllabus=None, clock=None, tips=None):
        self.P = P
        self.mem = mem
        self.tt = tt
        self.cal = calendar or SchoolCalendar.for_plugin(P)
        if syllabus is None:
            from plugins.education import syllabus as SY  # noqa: PLC0415
            syllabus = SY.for_plugin(P)
        self.syl = syllabus
        self.clock = clock or (lambda: dt.datetime.now())
        self.index = Index(P)
        self.support = SupportIndex(P, tips)
        self._seq = {}

    # ------------------------------------------------------------------------------------------------ sequences
    def sequence(self, grade, subject):
        """the servable lessons / cards / pointers of a grade's subject, in menu order (the resume chain)"""
        key = (int(grade), subject)
        if key in self._seq:
            return self._seq[key]
        seq = []
        for s in self.P.subjects(int(grade)) or []:
            if s["id"] != subject:
                continue
            for u in s["units"]:
                for i in u["items"]:
                    it = self.P.items.get(i)
                    if not it or i in seq:
                        continue
                    if it["kind"] == "pointer" or (it["kind"] in ("lesson", "card") and
                                                   G.item_allowed(self.P.rules, it)[0] == "ok"):
                        seq.append(i)
        self._seq[key] = seq
        return seq

    def item_ref(self, iid, source=None):
        it = self.P.items.get(iid)
        if not it:
            return None
        r = {"id": iid, "title": it["title"], "kind": it["kind"], "unit": self.P.unit_of.get(iid)}
        if source:
            r["source"] = source
        return r

    def resume_index(self, cls, subject, grade):
        """(index into the sequence, state, position) for this class / subject / grade"""
        seq = self.sequence(grade, subject)
        p = self.mem.position(cls["id"], self.P.name, subject) if subject else None
        if not p or not p.get("item"):
            return 0, "start", p
        unit = self.P.units.get(p.get("unit") or "") or {}
        if unit and unit.get("grade") not in (None, int(grade)):
            return 0, "start", p
        if p["item"] in seq:
            i = seq.index(p["item"])
            return (i + 1, "next", p) if p.get("finished") else (i, "continue", p)
        return 0, "start", p

    def funda(self, grade, subject, term, week):
        """Funda's pacing for this grade, subject, term and week: the sub-topics and the served lessons they link"""
        area = AREA_OF.get(subject or "")
        if not area or not term or not week or int(grade) not in self.syl.grades():
            return None
        w = self.syl.week(int(grade), int(term), int(week), area)
        if not w["rows"]:
            return None
        subs, items = [], []
        for r in w["rows"]:
            subs.append({"title": r["sub_topic"]["title"], "topic": r["topic"]["title"], "area": r["area"],
                         "weeks": f"{r['start']}-{r['end']}" if r.get("start") else None})
            for c in r["competences"]:
                for ln in c.get("links") or []:
                    it = self.P.items.get(ln["id"])
                    if not ln.get("served") or not it or ln["id"] in items:
                        continue
                    unit = self.P.units.get(self.P.unit_of.get(ln["id"]) or "") or {}
                    if subject in ("maths", "science") and (unit.get("subject") or it.get("subject")) not in (subject,):
                        continue
                    if int(grade) not in (it.get("grades") or [unit.get("grade")]):
                        continue                              # a link to another grade's lesson (cards span grades)
                    items.append(ln["id"])
        return {"label": FUNDA_LABEL, "term": int(term), "week": int(week), "matched": w["matched"],
                "sub_topics": subs[:4], "items": [self.item_ref(i) for i in items[:4]]}

    # ------------------------------------------------------------------------------------------------ slots
    def _classes(self, teacher):
        return {c["id"]: c for c in self.mem.classes(teacher)}

    def slots(self, teacher, start, end, now=None):
        """{"days": [day info + slots], "today"}: every school day in [start, end] with the teacher's periods, each
        slot's note(s) and its suggestion (see the module docstring)"""
        now = now or self.clock()
        today = now.date() if isinstance(now, dt.datetime) else d_(now)
        start, end = d_(start), d_(end)
        if end < start:
            raise ValueError("end before start")
        if (end - start).days > MAX_RANGE_DAYS:
            raise ValueError(f"at most {MAX_RANGE_DAYS} days")
        ov = self.tt.term_overrides(teacher)
        periods = self.tt.periods(teacher)
        classes = self._classes(teacher)
        gen_start = min(start, today)
        gen_end = max(end, today)
        days = self.cal.days(gen_start, gen_end, ov)
        notes = {}
        for n in self.tt.notes(teacher, since=iso(gen_start), until=iso(gen_end)):
            if n.get("slot_date") and n.get("period_id"):
                notes.setdefault((n["slot_date"], n["period_id"]), []).append(n)
        chains = {}                                     # (class, subject, grade) -> next sequence offset
        out = []
        for day in days:
            day["slots"] = []
            if not day["school_day"]:
                if start <= d_(day["date"]) <= end:
                    out.append(day)
                continue
            for p in sorted([p for p in periods if p["day"] == day["weekday"]],
                            key=lambda p: (p["start"] or "99:99", p["id"])):
                cls = classes.get(p["class_id"])
                slot = {"slot_id": f"{day['date']}:{p['id']}", "date": day["date"], "period": p,
                        "class": {"id": cls["id"], "label": cls["label"], "grades": cls["grades"]} if cls else None,
                        "subject": p["subject"], "label": p.get("label")}
                ns = notes.get((day["date"], p["id"]), [])
                rec = self._recorded(ns)
                if rec:
                    slot["notes"] = rec["notes"]
                    if rec.get("taught"):
                        slot["taught"] = rec["taught"]
                d = d_(day["date"])
                if d >= today and cls and p["subject"] and not (rec and rec.get("taught")):
                    # a taught slot shows what was taught, not a suggestion (the chain moved when it was tapped)
                    slot["suggestion"] = self._suggest(cls, p["subject"], day, chains, rec)
                day["slots"].append(slot)
            if start <= d_(day["date"]) <= end:
                out.append(day)
        return {"from": iso(start), "to": iso(end), "today": iso(today), "days": out,
                "note": SUGGESTION_NOTE}

    def _recorded(self, ns):
        if not ns:
            return None
        views = [self.tt.note_view(n) for n in ns]
        taught = next((v for v in reversed(views) if v["taught"] and v.get("item")), None)
        r = {"notes": views}
        if taught:
            r["taught"] = self.item_ref(taught["item"]) or {"id": taught["item"], "title": taught.get("taught_text")}
        planned = next((v.get("planned_item") for v in reversed(views) if v.get("planned_item")), None)
        if planned:
            r["planned"] = planned
        return r

    def _suggest(self, cls, subject, day, chains, rec):
        out = {"by_grade": [], "note": SUGGESTION_NOTE}
        for g in cls["grades"]:
            seq = self.sequence(g, subject)
            key = (cls["id"], subject, g)
            if key not in chains:
                i, state, _ = self.resume_index(cls, subject, g)
                chains[key] = [i, state]
            i, state = chains[key]
            resume = None
            planned = self.item_ref(rec["planned"], "planned") if rec and rec.get("planned") else None
            if i < len(seq):
                resume = dict(self.item_ref(seq[i]), state=state, label=RESUME_LABEL.get(state, "next in the menu"))
                if not planned:
                    chains[key] = [i + 1, "next"]             # a slot the teacher planned does not use up the chain
            fu = self.funda(g, subject, day.get("term"), day.get("week"))
            pick = planned or (dict(resume, source="resume") if resume else None)
            if not pick and fu and fu["items"]:
                pick = dict(fu["items"][0], source="funda")
            out["by_grade"].append({"grade": g, "pick": pick, "resume": resume, "funda": fu, "planned": planned,
                                    "end_of_menu": i >= len(seq) and bool(seq)})
        first = out["by_grade"][0] if out["by_grade"] else {}
        out["pick"] = first.get("pick")
        return out

    # ------------------------------------------------------------------------------------------------ today / week
    def week_start(self, d):
        return monday(d_(d))

    def today(self, teacher, now=None):
        now = now or self.clock()
        d = now.date() if isinstance(now, dt.datetime) else d_(now)
        s = self.slots(teacher, d, d, now)
        day = s["days"][0] if s["days"] else self.cal.day(d, self.tt.term_overrides(teacher))
        nxt = None
        if not day.get("slots"):                       # nothing today: the next school day with periods (<= 14 days)
            later = self.slots(teacher, d + dt.timedelta(days=1), d + dt.timedelta(days=14), now)
            nxt = next((x for x in later["days"] if x.get("slots")), None)
        return {"date": iso(d), "day": {k: v for k, v in day.items() if k != "slots"}, "slots": day.get("slots", []),
                "next_day": nxt, "suggested": self.suggested(teacher, now),
                "has_timetable": bool(self.tt.periods(teacher)), "note": SUGGESTION_NOTE}

    def week(self, teacher, start=None, now=None):
        now = now or self.clock()
        d = d_(start) if start else (now.date() if isinstance(now, dt.datetime) else d_(now))
        m = monday(d)
        return self.slots(teacher, m, m + dt.timedelta(days=6), now)

    # ------------------------------------------------------------------------------------------------ suggested tray
    def suggested(self, teacher, now=None, limit_tips=2, limit_repeat=3):
        """the lightbulb: next lesson per class, a tip or fix for last week's "didn't work", lessons marked repeat"""
        now = now or self.clock()
        d = now.date() if isinstance(now, dt.datetime) else d_(now)
        out = []
        classes = self.mem.classes(teacher)
        # 1. next lesson per class: the class's next upcoming slot (today .. 14 days), else its first subject's resume
        upcoming = self.slots(teacher, d, d + dt.timedelta(days=14), now)
        seen = set()
        for day in upcoming["days"]:
            for s in day.get("slots", []):
                c = s.get("class")
                sg = s.get("suggestion") or {}
                if not c or c["id"] in seen or not sg.get("pick") or s.get("taught"):
                    continue
                seen.add(c["id"])
                out.append({"kind": "next", "class": c, "subject": s["subject"], "item": sg["pick"],
                            "when": {"date": day["date"], "slot_id": s["slot_id"], "start": s["period"].get("start")},
                            "why": "your next " + (s["subject"] or "") + " period with " + c["label"]})
        for c in classes:
            if c["id"] in seen:
                continue
            for g in c["grades"]:
                for s in self.P.subjects(g) or []:
                    seq = self.sequence(g, s["id"])
                    i, state, p = self.resume_index(c, s["id"], g)
                    if p and i < len(seq):
                        out.append({"kind": "next", "class": {"id": c["id"], "label": c["label"],
                                                              "grades": c["grades"]},
                                    "subject": s["id"], "item": self.item_ref(seq[i], "resume"), "when": None,
                                    "why": "where " + c["label"] + " left off in " + s["label"]})
                        seen.add(c["id"])
                        break
                if c["id"] in seen:
                    break
        # 2. tips for last week's "didn't work" (the digest's matches when one exists, else live retrieval)
        since = iso(d - dt.timedelta(days=7))
        tips = []
        for dg in self.tt.digests(teacher):
            if dg.get("week", "") >= iso(monday(d) - dt.timedelta(days=7)):
                for m in dg.get("matches", []):
                    tips.append({"kind": "tip", "match": m["match"], "because": {"notes": m.get("notes", []),
                                                                                 "text": m.get("for")},
                                 "class": dg.get("class"), "from": "weekly digest"})
        if not tips:
            for n in self.tt.notes(teacher, since=since, until=iso(d)):
                tags = json.loads(n.get("tags") or "[]")
                if "didnt" not in tags or not (n.get("text_en") or n.get("text")):
                    continue
                b = self.support.best(n.get("text_en") or n.get("text"), lesson=n.get("item"), subject=n.get("subject"))
                if b:
                    tips.append({"kind": "tip", "match": b, "because": {"notes": [n["id"]],
                                                                              "text": (n.get("text") or "")[:120]},
                                 "class": n.get("class_id"), "from": "your note"})
        seen_t = set()
        for t in tips:
            k = (t["match"]["kind"], t["match"]["id"])
            if k in seen_t:
                continue
            seen_t.add(k)
            out.append(t)
            if len(seen_t) >= limit_tips:
                break
        # 3. lessons marked repeat (last 21 days) and not taught again since
        rep = 0
        notes = self.tt.notes(teacher, since=iso(d - dt.timedelta(days=21)), until=iso(d))
        for n in reversed(notes):
            tags = json.loads(n.get("tags") or "[]")
            if "repeat" not in tags or not n.get("item"):
                continue
            when = n.get("slot_date") or n["created_at"][:10]
            again = any(m.get("taught") and m.get("item") == n["item"] and m.get("class_id") == n.get("class_id")
                        and (m.get("slot_date") or m["created_at"][:10]) > when for m in notes)
            if again:
                continue
            out.append({"kind": "repeat", "class": n.get("class_id"), "item": self.item_ref(n["item"], "repeat"),
                        "note_id": n["id"], "why": "you marked it to repeat on " + when})
            rep += 1
            if rep >= limit_repeat:
                break
        return out

    # ------------------------------------------------------------------------------------------------ taught it
    def taught(self, teacher, class_id, item, slot_date=None, period_id=None, note_id=None, updated_at=None):
        """one tap: the class's resume point moves to the next lesson in the menu (or the unit is finished), and the
        slot's note records the lesson taught. Returns the new position and what is next."""
        if item not in self.P.items:
            raise ValueError("no such item")
        cls = self.mem.get_class(int(class_id)) if str(class_id or "").isdigit() else None
        if not cls or cls["teacher"] != norm_number(teacher):
            raise ValueError("not your class")
        unit = self.P.unit_of.get(item)
        subject = self.P.units[unit]["subject"] if unit in self.P.units else self.P.items[item].get("subject")
        grade = (self.P.units.get(unit) or {}).get("grade") or (self.P.items[item].get("grades") or [None])[0]
        seq = self.sequence(grade, subject) if grade else []
        nxt = seq[seq.index(item) + 1] if item in seq and seq.index(item) + 1 < len(seq) else None
        if nxt and self.P.unit_of.get(nxt) == unit:
            pos = self.mem.set_position(cls["id"], self.P.name, subject, unit=unit, item=nxt, part=0, parts=0,
                                        finished=False)
        else:
            pos = self.mem.set_position(cls["id"], self.P.name, subject, unit=unit, item=item, finished=True)
        self.mem.log_request(teacher, "lesson", item, outcome="taught")
        note = None
        if note_id:
            note = self.tt.upsert_note(teacher, {"id": note_id, "class_id": cls["id"], "slot_date": slot_date,
                                                 "period_id": period_id, "subject": subject, "item": item,
                                                 "taught": True, "updated_at": updated_at})
        return {"position": pos, "next": self.item_ref(nxt) if nxt else None, "note": note,
                "next_unit": None if nxt else ({"id": self.P.next_unit(unit)["id"], "label": self.P.next_unit(unit)
                                                ["label"]} if unit and self.P.next_unit(unit) else None)}

    # ------------------------------------------------------------------------------------------------ quick add
    def parse_quick(self, teacher, text, now=None, model=None, translate=None):
        """one line -> {class, subject, item, tags, attention, slot, confidence, source, alternatives, text}; rules
        first, `model(text_en, classes, candidates) -> {class_id, item, tags}` only for what the rules left open"""
        now = now or self.clock()
        t = re.sub(r"\s+", " ", str(text or "")).strip()[:1000]
        if not t:
            raise ValueError("empty")
        if G.safeguarding(t):
            return {"safeguarding": True, "text": None}
        classes = self.mem.classes(teacher)
        text_en, lang = t, "en"
        if translate is not None:
            try:
                tr = translate(t)
                if tr and tr.get("lang") == "toi":
                    text_en, lang = tr["text"] or t, "toi"
            except Exception:  # noqa: BLE001
                pass
        why = []
        # class
        cls, how = self._parse_class(t, classes, now)
        if cls:
            why.append(f"class by {how}")
        # tags
        tags = self._parse_tags(text_en)
        attention = None
        if "attention" in tags:
            attention = "learner" if self._names(t) and not CLASS_WORDS.search(t) else "class"
        # subject
        subject = next((s for s, rx in SUBJECT_RX if rx.search(text_en)), None)
        # slot: today's period for this class (now or the latest one started), else None
        slot = self._slot_for(teacher, cls, subject, now)
        if slot and not subject:
            subject = slot["subject"]
        # lesson
        grades = cls["grades"] if cls else sorted({g for c in classes for g in c["grades"]}) or None
        item, item_how, cands = None, None, []
        query = self._topic_words(text_en, cls)
        for _, _, _, iid in self.index.search(query, k=6, grades=grades, min_cover=0.25):
            it = self.P.items[iid]
            if grades and not set(it.get("grades") or []) & set(grades):
                continue
            cands.append(iid)
        if subject and cands:
            same = [i for i in cands if (self.P.units.get(self.P.unit_of.get(i) or "") or {}).get("subject",
                                                                                            self.P.items[i].get("subject")) == subject]
            cands = same + [i for i in cands if i not in same]
        if cands:
            item, item_how = cands[0], "words"
        elif slot and slot.get("item"):
            item, item_how = slot["item"]["id"], "the period's lesson"
        if item:
            why.append(f"lesson by {item_how}")
        conf = 0.0
        conf += 0.4 if cls and how in ("label", "only class", "grade") else 0.25 if cls else 0.0
        conf += 0.3 if tags else 0.0
        conf += 0.3 if item_how == "words" else 0.2 if item else 0.0
        source = "rules"
        open_parts = [k for k, v in (("class", cls), ("tags", tags), ("item", item)) if not v]
        if model is not None and open_parts and (len(classes) > 1 or "class" not in open_parts):
            slot_items = [slot["item"]["id"]] if slot and slot.get("item") else []
            m_cands = list(dict.fromkeys(cands + slot_items))[:8]
            try:
                m = model(text_en, classes, m_cands) or {}
            except Exception as e:  # noqa: BLE001
                m, why = {}, why + [f"model failed: {str(e)[:80]}"]
            if m:
                source = "rules+model"
                if not cls and m.get("class_id") in {c["id"] for c in classes}:
                    cls = next(c for c in classes if c["id"] == m["class_id"])
                    why.append("class by the model")
                if not tags and m.get("tags"):
                    tags = [x for x in m["tags"] if x in TAGS]
                    why.append("tags by the model")
                if not item and m.get("item") in m_cands:
                    item = m["item"]
                    why.append("lesson by the model")
        return {"text": t, "text_en": text_en if lang == "toi" else None, "lang": lang,
                "class": {"id": cls["id"], "label": cls["label"], "grades": cls["grades"]} if cls else None,
                "subject": subject, "item": self.item_ref(item) if item else None, "tags": tags,
                "attention": attention, "slot": {k: slot[k] for k in ("slot_id", "date", "period_id")} if slot
                else None, "confidence": round(min(conf, 1.0), 2), "source": source, "why": "; ".join(why),
                "alternatives": {"classes": [{"id": c["id"], "label": c["label"]} for c in classes],
                                 "items": [self.item_ref(i) for i in cands[:5]]}}

    def _parse_class(self, t, classes, now):
        if not classes:
            return None, None
        low = " " + re.sub(r"[^\w\s/]", " ", t.lower()) + " "
        hits = []
        for c in classes:
            lab = re.sub(r"[^\w\s/]", " ", c["label"].lower()).strip()
            toks = [x for x in re.split(r"[\s/]+", lab) if x and x not in ("grade", "gr", "class", "the")]
            if lab and (" " + lab + " ") in low:
                hits.append((3, c))
            elif toks and all((" " + x + " ") in low for x in toks) and any(not x.isdigit() for x in toks):
                hits.append((2, c))
        if hits:
            best = max(h[0] for h in hits)
            top = [c for s, c in hits if s == best]
            if len(top) == 1:
                return top[0], "label"
        g = grades_in(t)
        m = re.search(r"(?i)(?<![a-z0-9])([1-7])\s?[a-z](?![a-z])", t)       # '2C', '3 B'
        if not g and m:
            g = [int(m.group(1))]
        if g:
            top = [c for c in classes if set(c["grades"]) & set(g)]
            if len(top) == 1:
                return top[0], "grade"
        if len(classes) == 1:
            return classes[0], "only class"
        p = self._period_now(classes[0]["teacher"], now)
        if p and p.get("class_id"):
            c = next((c for c in classes if c["id"] == p["class_id"]), None)
            if c:
                return c, "the period on now"
        return None, None

    def _period_now(self, teacher, now):
        if not isinstance(now, dt.datetime):
            return None
        hm = now.strftime("%H:%M")
        ps = [p for p in self.tt.periods(teacher) if p["day"] == now.isoweekday() and p.get("start")]
        started = [p for p in ps if p["start"] <= hm]
        return max(started, key=lambda p: p["start"]) if started else None

    def _slot_for(self, teacher, cls, subject, now):
        d = now.date() if isinstance(now, dt.datetime) else d_(now)
        try:
            day = self.slots(teacher, d, d, now)["days"]
        except ValueError:
            return None
        slots = [s for x in day for s in x.get("slots", [])]
        if cls:
            slots = [s for s in slots if s.get("class") and s["class"]["id"] == cls["id"]]
        if subject:
            slots = [s for s in slots if s["subject"] == subject] or ([] if cls else slots)
        if not slots:
            return None
        hm = now.strftime("%H:%M") if isinstance(now, dt.datetime) else "23:59"
        started = [s for s in slots if (s["period"].get("start") or "00:00") <= hm]
        s = (started or slots)[-1]
        item = (s.get("taught") or (s.get("suggestion") or {}).get("pick"))
        return {"slot_id": s["slot_id"], "date": s["date"], "period_id": s["period"]["id"], "subject": s["subject"],
                "item": item}

    @staticmethod
    def _topic_words(t, cls):
        """the note minus its tag phrases and the class label: what is left names the lesson ('counting')"""
        q = t
        for rx in TAG_RX.values():
            q = rx.sub(" ", q)
        if cls:
            for tok in re.split(r"[\s/]+", cls["label"]):
                if tok:
                    q = re.sub(r"(?i)(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])", " ", q)
        q = re.sub(r"(?i)(?<![a-z])(grade|class|lesson|today|the|learners|pupils|children|they|it)(?![a-z])", " ", q)
        return re.sub(r"\s+", " ", q).strip() or t

    @staticmethod
    def _parse_tags(t):
        tags = []
        rest = t
        m = TAG_RX["didnt"].search(t)
        if m:
            tags.append("didnt")
            rest = TAG_RX["didnt"].sub(" ", t)
        if TAG_RX["worked"].search(rest):
            tags.append("worked")
        for k in ("idea", "repeat", "attention"):
            if TAG_RX[k].search(t):
                tags.append(k)
        return [x for x in TAGS if x in tags]

    @staticmethod
    def _names(t):
        """capitalised words that are neither English words nor known labels: probable learner names"""
        return probable_names(t)


def probable_names(t):
    """probable learner names in a note, erring towards catching (in a cross-teacher file a false redaction costs
    nothing; a leaked name costs a child's privacy): the house name finder (h2_text.en_names: capitals that are not
    ordinary English words), plus Zambian given names that are English words (Gift, Precious: h2_text's list), plus a
    capitalised word of Bantu shape (ends in a vowel) that the English dictionary does not hold, wherever it stands
    ('Mutinta needs help': en_names skips it, the memory vocabulary has 'mutinta' in lower case)."""
    t = t or ""
    out = []
    try:
        import h2_text as T  # noqa: PLC0415
        out += T.en_names(t)
        given, ordinary = T.EN_WORD_GIVEN_NAMES, T._en_dict_ordinary
    except Exception:  # noqa: BLE001
        given, ordinary = set(), (lambda w: False)
    for m in NAME_RX.finditer(t):
        w = m.group(1)
        if w in NOT_NAMES:
            continue
        if w in given or (w[-1] in "aeiou" and not ordinary(w)):
            out.append(w)
    return list(dict.fromkeys(out))
