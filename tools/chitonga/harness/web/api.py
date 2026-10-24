"""The teacher web app's API over the harness (24 Oct 2026, the Jesuit).

Chris, 24 Oct: "the issue is speed but also just that it's fiddly, I think a page would be better, quicker, and
information can be presented better, eg it can also display pictures." DECISIONS 24 Oct: the harness stays the
engine; a mobile-first PWA becomes the main front-end; local first, nothing published.

WHAT THIS IS. A thin layer that READS the harness and writes only through it: the education Plugin (items, menu,
strategies, glossary, rules), the teacher memory (core.memory.SQLiteMemory, the SAME file as the WhatsApp adapter,
pilot_memory.sqlite, so a teacher's classes and syllabus positions are shared between the page and WhatsApp), the
rules reader (core.intake.RulesReader, as the live adapter uses), the gates (safeguarding first, scope, the status
rules for health items, holds), the struggle matcher (Engine._match_mistake, called, not copied) and the flywheel's
ONE writer (adapter.flywheel_log.FlywheelWriter, pointed at web_logs/ so the records have the adapter's exact shape).
Nothing in core/ or adapter/ is edited.

WHY BUNDLES. A tap must answer in < 300 ms and the page must work offline, so the content of a grade (compact lesson
views, Tonga boxes, key words, diagrams) is ONE keyed, versioned, gzipped response (bundle) that the service worker
caches; a tap inside the app is then in memory. The server is needed only for search, struggles and sync.

SYNC WITHOUT LOSS OR DUPLICATES. Every write (flywheel answer, written line or word, struggle, progress, follow-up,
lesson opened) carries a client id made on the phone when the teacher tapped. The server stores the id with its
result in web_sync.sqlite; the same id again returns the stored result and does nothing (a retried offline queue
never pays twice, never files a struggle twice).

NEVER: a Tonga word that no source holds (diagram labels and key words show "Tonga word needed" instead and become
a K1 flywheel item); pupil-level text stored (pupil_level() as the engine); a safeguarding text stored (the event
only); teacher Tonga in a public file (everything here is behind the teacher key).
"""
import datetime
import gzip
import hashlib
import html
import inspect
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(HARNESS, "..", "..", ".."))
for p in (HARNESS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import gates as G  # noqa: E402
from core import render as R  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.intake import RulesReader, validate_reading  # noqa: E402
from core.memory import SQLiteMemory, norm_number, pupil_level  # noqa: E402
from core.retrieval import Index  # noqa: E402
from core import search_index as SI  # noqa: E402  (24 Oct 2026: the teacher search; site/assistant/search.js)

import diagrams as D  # noqa: E402
import guide as GD  # noqa: E402
from timetable_api import TimetableMixin  # noqa: E402  (24 Oct 2026: timetable, notes, digest; TIMETABLE_API.md)

DATA = os.path.join(REPO, "internal", "chitonga_train", "harness")
KEYS_PATH = os.path.join(DATA, "web_keys.json")
WEB_LOGS = os.path.join(DATA, "web_logs")
PAY = {"tonga_ok": 0, "fix": 2, "right": 1, "write": 2, "word": 1}    # the desk's K1 word / confirm, K2 line
DAILY_PAID_CAP = 20
HELD_OUT_FIRST = 40          # as the adapter: a teacher's first 40 free-text requests are test, never train
# Chris, 24 Oct: "we just continue with the teachers as is, they seem to like WhatsApp": the paid contribution prompts
# stay on the WhatsApp desk. op_flywheel is kept for later and refused while this is False; the page offers only the
# unpaid "report a problem with this line" (op_report).
FLYWHEEL_IN_APP = os.environ.get("ZAMAI_WEB_FLYWHEEL") == "1"
ORIGIN_NAME = {"T": "teacher", "B": "book", "L": "lexicon", "D": "needs a teacher's check"}


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _badge(origins):
    """['T09'] -> 'T'; ['L','B'] -> 'B' (book beats lexicon); ['D'] -> 'D'"""
    o = [str(x) for x in (origins or [])]
    if any(x.startswith("T") for x in o):
        return "T"
    if "D" in o:
        return "D"
    if "B" in o:
        return "B"
    if "L" in o:
        return "L"
    return o[0] if o else ""


def _compact(x):
    """drop None / empty values (the Grade 1 printed Chitonga lessons make the bundle; every byte is data cost)"""
    if isinstance(x, dict):
        return {k: _compact(v) for k, v in x.items() if v is not None and v != [] and v != "" and v is not False
                or k in ("held", "needed", "allowed")}
    if isinstance(x, list):
        return [_compact(v) for v in x]
    return x


class KeyStore:
    """web_keys.json: {"keys": {key: {code, number, name, role, tester, created}}}. Keys are random 24-char tokens;
    the file sits under internal/ and is gitignored."""

    def __init__(self, path=KEYS_PATH):
        self.path = path

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f).get("keys", {})
        except (OSError, ValueError):
            return {}

    def get(self, key):
        if not key or len(key) < 16:
            return None
        return self.load().get(key)

    def create(self, code, number, name, role=None, tester=False):
        import secrets  # noqa: PLC0415
        keys = self.load()
        k = secrets.token_urlsafe(18)
        keys[k] = {"code": code, "number": norm_number(number), "name": name, "role": role, "tester": bool(tester),
                   "created": utcnow().isoformat(timespec="seconds")}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"note": "teacher web app keys (24 Oct 2026); private; never commit, never publish",
                       "keys": keys}, f, ensure_ascii=False, indent=1)
        return k


class SyncStore:
    """client ids already applied, with their stored result (idempotent sync)"""

    def __init__(self, path):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        self.db.execute("create table if not exists applied (client_id text primary key, teacher text, type text, "
                        "result text, at text)")

    def get(self, cid):
        with self.lock:
            r = self.db.execute("select result from applied where client_id=?", (cid,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, cid, teacher, typ, result):
        with self.lock:
            self.db.execute("insert or ignore into applied values (?,?,?,?,?)",
                            (cid, teacher, typ, json.dumps(result, ensure_ascii=False, default=str),
                             utcnow().isoformat(timespec="seconds")))


class WebAPI(TimetableMixin):
    def __init__(self, plugin=None, hooks=None, memory=None, keys=None, logs=WEB_LOGS, sync_db=None, clock=None,
                 allowlist=None):
        if plugin is None:
            from plugins import education  # noqa: PLC0415
            plugin, hooks = education.load()
        self.P = plugin
        self.hooks = hooks or getattr(plugin, "hooks", None)
        if memory is None:
            from adapter.config import Config  # noqa: PLC0415
            memory = SQLiteMemory(Config().memory_db)
        self.mem = memory
        self.keys = keys or KeyStore()
        self.logs = logs
        self.sync = SyncStore(sync_db or os.path.join(DATA, "web_sync.sqlite"))
        self.clock = clock or utcnow
        self.reader = RulesReader()
        self.index = Index(plugin)
        # the engine object is used for its struggle matcher only (no turn is run through it from here)
        self.eng = Engine(plugin, memory, hooks=self.hooks)
        self.diagrams = D.load_all()
        self.resolver = D.Resolver(plugin)
        self.diagram_of = D.for_lessons(self.diagrams)
        self.allow = allowlist
        self._bundles = {}
        self._wlock = threading.Lock()
        self.tips = self._load_tips()
        # 24 Oct 2026: the teacher search (core/search_index.py): the index the phone fetches (chunk part 'search') and
        # the matcher behind /api/search, the same as site/assistant/search.js on the phone and in the Netlify function
        self.search_ix, self.searcher = self._build_search()
        # 24 Oct 2026: the Grade 1-3 syllabus (plugins/education/syllabus.py over the plug-in's 'syllabus' items); empty
        # for a plug-in without them. SYLLABUS_API.md is the UI's contract.
        from plugins.education import syllabus as SY  # noqa: PLC0415
        self.syl = SY.for_plugin(plugin)
        # 24 Oct 2026: timetable, calendar, lesson notes, digest (TIMETABLE_API.md is the UI's contract). The tables sit
        # in the same memory file (core/timetable.py); the calendar is the plug-in's preloaded school_calendar.json.
        from core.school_calendar import SchoolCalendar  # noqa: PLC0415
        from core.timetable import TimetableStore  # noqa: PLC0415
        from plugins.education.planner import Planner  # noqa: PLC0415
        self.tt = TimetableStore(self.mem)
        self.cal = SchoolCalendar.for_plugin(plugin)
        self.planner = Planner(plugin, self.mem, self.tt, self.cal, self.syl, clock=self.local_now, tips=self.tips)
        self._digest_jobs = {}

    def local_now(self):
        """Lusaka wall-clock time (CAT, UTC+2, no daylight saving), naive: periods are written in school time"""
        return (self.clock() + datetime.timedelta(hours=2)).replace(tzinfo=None)

    # ------------------------------------------------------------------------------------------------ identity
    def teacher(self, key):
        """the key's entry with the normalised number, or None"""
        e = self.keys.get(key)
        if not e or not e.get("number"):
            return None
        return dict(e, number=norm_number(e["number"]))

    def fingerprint(self):
        h = hashlib.sha1((self.P.fingerprint + "|" + "|".join(
            f"{k}:{len(json.dumps(v))}" for k, v in sorted(self.diagrams.items())) + "|tips:" +
            hashlib.sha1(json.dumps(self.tips, sort_keys=True).encode()).hexdigest() + "|view:" + self.VIEW +
            "|search:" + self._search_hash()).encode()).hexdigest()[:10]
        return h

    def _search_hash(self):
        """the search index's own hash (concept terms, lexicon, matcher version, the items): a change moves the
        fingerprint, so the phone fetches the new index (?v=)"""
        ix = getattr(self, "search_ix", None)
        return ix["fp"] if ix else "-"

    def _build_search(self):
        d = getattr(self.P, "dir", "") or ""
        try:
            with open(os.path.join(d, "search_concepts.json"), encoding="utf-8") as f:
                concepts = json.load(f).get("items") or {}
        except (OSError, ValueError):
            concepts = {}
        ix = SI.build(self.P, self.tips, concepts, SI.lexicon(os.path.join(d, "search_lexicon.json")))
        # the safeguarding gate travels with the index, so the phone can check it OFFLINE before it shows any hit
        # (h3_front.safeguarding: a strong word, or a child word with a weak word; the referral in English)
        ix["safe"] = {"strong": SI.js_rx(G.F.SAFE_STRONG), "child": SI.js_rx(G.F.SAFE_CHILD),
                      "weak": SI.js_rx(G.F.SAFE_WEAK), "referral": G.referral("web")["en_meaning"],
                      # 24 Oct 2026: the full detector (safeguard.py: categories, normalisation, benign phrases) for
                      # search.js's port; the three regexes above stay for an older phone bundle
                      "rules": G.F._SG.export()}
        ix["fp"] = hashlib.sha1(json.dumps(ix, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
        return ix, SI.Searcher(ix)

    # the shape of the lesson views; bump it when lesson_view adds a field (phones cache bundles by fingerprint)
    # 2: printed-guide lines carry src and page (24 Oct 2026)
    VIEW = "5"   # 5: the live export cleans what a teacher reads (public_text.py, persona review M3); 3: the grade in parts (index, item, subject; 24 Oct 2026); 4: key-word counts in the index

    TIP_FIELDS = ("id", "topic", "title", "when", "steps", "why", "watch_out", "source", "origin", "tonga_needed")
    TIP_TOPICS = ("curriculum", "groups", "noise", "large-class", "questioning", "checking", "routines", "materials",
                  "multilingual")

    def _load_tips(self):
        """Teaching tips (Chris, 24 Oct 2026: they replace the "What does your class find hard?" form in the app):
        general pedagogy for the new curriculum, groups, noise, large classes, checking, materials, many languages.
        English only; origin B (a printed source) or F (Funda material, AI-drafted); tonga_needed lists the classroom
        words a teacher still has to give us. Read from the plug-in's teaching_tips.jsonl; a row with a field or topic
        outside the schema is left out (and said so on stderr), never half-shown."""
        p = os.path.join(getattr(self.P, "dir", "") or "", "teaching_tips.jsonl")
        out = []
        if not os.path.exists(p):
            return out
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    print(f"teaching_tips.jsonl:{n}: not JSON, left out", file=sys.stderr)
                    continue
                bad = (set(r) - set(self.TIP_FIELDS) or r.get("topic") not in self.TIP_TOPICS or
                       r.get("origin") not in ("B", "F") or not r.get("title") or not isinstance(r.get("steps"), list))
                if bad:
                    print(f"teaching_tips.jsonl:{n}: outside the schema, left out", file=sys.stderr)
                    continue
                out.append({k: r[k] for k in self.TIP_FIELDS if r.get(k)})
        return out

    def _writer(self, who):
        from adapter.config import Config  # noqa: PLC0415
        from adapter.flywheel_log import FlywheelWriter  # noqa: PLC0415
        cfg = Config(base=self.logs)
        cfg.flywheel_path = os.path.join(self.logs, "flywheel_web.jsonl")
        entry = {"code": who.get("code"), "name": who.get("name"), "role": who.get("role"),
                 "tester": who.get("tester"), "consent": who.get("consent")}
        w = FlywheelWriter(cfg, {who["number"]: entry}, "web")
        w.ledger = os.path.join(self.logs, "pay_ledger_web.tsv")
        return w

    # ------------------------------------------------------------------------------------------------ me / home
    def me(self, who):
        n = who["number"]
        prof = self.mem.get_teacher(n) or {}
        classes = [self._class_view(c) for c in self.mem.classes(n)]
        return {"teacher": {"name": prof.get("name") or "", "school": prof.get("school") or "",
                            "language_pref": prof.get("language_pref") or "both", "code": who.get("code"),
                            "tester": bool(who.get("tester")), "last_class_id": prof.get("last_class_id")},
                "classes": classes, "onboarded": bool(prof.get("name") and classes),
                "grades_available": self.P.grades(), "fingerprint": self.fingerprint(),
                "origins": {"T": "Teacher: written or confirmed by a Tonga teacher",
                            "B": "Book: printed school book, verbatim", "L": "Lexicon: word list / dictionary",
                            "D": "Draft: built from teacher input, needs a teacher's check"},
                "flags": {"flywheel": FLYWHEEL_IN_APP},
                "pay": {"right": PAY["right"], "fix": PAY["fix"], "write": PAY["write"], "word": PAY["word"],
                        "cap": DAILY_PAID_CAP, "tester": bool(who.get("tester"))}}

    def _class_view(self, c):
        return {"id": c["id"], "label": c["label"], "grades": c["grades"], "size": c.get("size")}

    def onboard(self, who, body):
        n = who["number"]
        self.mem.get_or_create(n)
        name = re.sub(r"\s+", " ", str(body.get("name") or "")).strip()[:60]
        fields = {}
        if name:
            fields["name"] = name
        if body.get("school"):
            fields["school"] = str(body["school"]).strip()[:60]
        if body.get("language_pref") in ("en", "toi", "both"):
            fields["language_pref"] = body["language_pref"]
        if fields:
            self.mem.update_teacher(n, **fields)
        have = {c["label"].lower() for c in self.mem.classes(n)}
        for c in body.get("classes") or []:
            grades = sorted({int(g) for g in c.get("grades") or [] if str(g).isdigit() and int(g) in self.P.grades()})
            label = re.sub(r"\s+", " ", str(c.get("label") or "")).strip()[:40] or (
                "Grade " + "/".join(map(str, grades)) if grades else "")
            if grades and label.lower() not in have:
                self.mem.add_class(n, label, grades, size=int(c["size"]) if str(c.get("size") or "").isdigit() else None)
                have.add(label.lower())
        return self.me(who)

    def _next_for(self, cls):
        """per subject of the class's grade(s): where the class is (position) and the lesson to open next"""
        out = []
        pos = {(p["subject"]): p for p in self.mem.positions(cls["id"]) if p["plugin"] == self.P.name}
        for g in cls["grades"]:
            for s in self.P.subjects(g):
                ready = [u for u in s["units"] if u["items"]]
                if not ready:
                    continue
                p = pos.get(s["id"])
                unit, state = None, "start"
                if p and p.get("unit") in self.P.units and self.P.units[p["unit"]].get("grade") == g:
                    if p["finished"]:
                        nu = self.P.next_unit(p["unit"])
                        unit, state = (nu["id"], "next") if nu else (p["unit"], "done")
                    else:
                        unit, state = p["unit"], "continue"
                if not unit:
                    unit = ready[0]["id"]
                u = self.P.units[unit]
                item = (p or {}).get("item") if state == "continue" else None
                item = item if item in u["items"] else self._first_servable(u["items"])
                out.append({"grade": g, "subject": s["id"], "subject_label": s["label"], "unit": unit,
                            "unit_label": u["label"], "item": item,
                            "title": self.P.items[item]["title"] if item else u["label"], "state": state,
                            "updated_at": (p or {}).get("updated_at")})
        return out

    def _first_servable(self, items):
        for i in items:
            it = self.P.items.get(i)
            if it and G.item_allowed(self.P.rules, it)[0] in ("ok",) or (it and it["kind"] == "pointer"):
                return i
        return items[0] if items else None

    def home(self, who):
        n = who["number"]
        prof = self.mem.get_teacher(n) or {}
        classes = self.mem.classes(n)
        rows = []
        for c in classes:
            rows.append({"class": self._class_view(c), "next": self._next_for(c)})
        resume = None
        latest = []
        for c in classes:
            for p in self.mem.positions(c["id"]):
                if p["plugin"] == self.P.name and p.get("item") in self.P.items and not p["finished"]:
                    latest.append((p.get("updated_at") or "", c, p))
        if latest:
            _, c, p = max(latest, key=lambda r: r[0])
            resume = {"class": self._class_view(c), "item": p["item"], "title": self.P.items[p["item"]]["title"],
                      "part": p.get("part") or 0, "unit": p.get("unit"), "updated_at": p.get("updated_at")}
        fu = self.mem.pending_followup(n)
        follow = None
        if fu:
            follow = {"suggestion_id": fu["id"], "what": fu.get("what"), "class_label": fu.get("class_label"),
                      "observation": fu.get("observation"), "suggested_at": fu.get("suggested_at")}
        return {"name": prof.get("name") or "", "resume": resume, "classes": rows, "followup": follow}

    # ------------------------------------------------------------------------------------------------ content
    def bundle(self, grade):
        """everything a teacher of this grade can open, compact (cached per plug-in fingerprint)"""
        grade = int(grade)
        key = (grade, self.fingerprint())
        if key in self._bundles:
            return self._bundles[key]
        gm = self.P.grade_menu(grade)
        if not gm:
            return None
        subjects, lessons, dia = [], {}, set()
        for s in gm["subjects"]:
            units = []
            for u in s["units"]:
                units.append({"id": u["id"], "label": u["label"], "ref": u.get("ref"), "items": u["items"],
                              "note": u.get("note") or "", "ready": bool(u["items"])})
                if u.get("sample"):    # 24 Oct 2026: a sample lesson's unit says so, with its curriculum
                    units[-1].update(sample=True, curriculum=u.get("curriculum"), level=u.get("level"),
                                     status=u.get("status"))
                if any((self.P.items.get(i) or {}).get("needs_review") for i in u["items"]):
                    units[-1]["needs_review"] = True          # a unit holding a labelled item says so
                for i in u["items"]:
                    if i not in lessons and i in self.P.items:
                        lessons[i] = self.lesson_view(i, unit=u["id"])
                        dia.update(self.diagram_of.get(i, []))
            subjects.append({"id": s["id"], "label": s["label"], "units": units})
        out = {"grade": grade, "fingerprint": self.fingerprint(), "subjects": subjects, "lessons": lessons,
               "diagrams": {d: self.diagram_view(d) for d in sorted(dia)}, "terms": self.terms(),
               "tips": self.tips, "syllabus": self.syllabus_bundle(grade)}
        if gm.get("samples_only"):     # 24 Oct 2026: a grade with sample lessons only (SAMPLES_API.md)
            out["samples_only"] = True
            out["grade_label"] = gm.get("label")
            out["grade_note"] = gm.get("note")
        raw = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._bundles[key] = (out, raw, gzip.compress(raw, 6))
        return self._bundles[key]

    # ------------------------------------------------------------------------------------------------ split (24 Oct 2026)
    # Chris: "the app stutters when switching from Grade 1 to Grade 2 or 3". A grade bundle was 0.6-2.4 MB of JSON,
    # fetched and parsed whole on every grade switch. The phone now asks for parts (GET /api/chunk):
    #   index    the menu: subjects and units, one small stub per lesson (title, kind, labels), the method terms
    #   item     one lesson's full view with the drawings it uses (on open)
    #   subject  every full view of one subject of the grade (prefetched at idle for offline use)
    #   syllabus the grade's syllabus;  tips  the teaching tips (the same for every grade)
    # The same split serves the Netlify export (files) and this server (memory), so both answer identically.
    STUB_KEYS = ("title", "kind", "subject", "unit", "scope", "teaches", "toi_primary", "needs_review", "review_tag",
                 "sample", "curriculum", "level", "withheld")

    @staticmethod
    def split_bundle(b):
        lessons = b["lessons"]
        stubs = {i: {k: L[k] for k in WebAPI.STUB_KEYS if L.get(k) not in (None, "", False)} for i, L in lessons.items()}
        for i, L in lessons.items():                  # key words: how many, how many with an attested Tonga word
            kws = L.get("key_words") or []
            if kws:
                stubs[i]["kw"] = [len(kws), sum(1 for k in kws if not k.get("needed") and k.get("toi"))]
        index = {"grade": b["grade"], "fingerprint": b["fingerprint"], "subjects": b["subjects"], "lessons": stubs,
                 "terms": b.get("terms"), "has_syllabus": bool(b.get("syllabus") and b["syllabus"].get("areas")),
                 # the grade's drawings, so the phone can keep them for offline use without the lesson views
                 "svgs": sorted({d["svg_url"] for d in (b.get("diagrams") or {}).values() if d.get("svg_url")})}
        for k in ("samples_only", "grade_label", "grade_note"):
            if b.get(k):
                index[k] = b[k]
        dia = b.get("diagrams") or {}

        def with_dia(ids):
            ds = sorted({d for i in ids for d in (lessons[i].get("diagrams") or [])})
            return {d: dia[d] for d in ds if d in dia}
        items = {i: {"lesson": L, "diagrams": with_dia([i])} for i, L in lessons.items()}
        subjects = {}
        for i, L in lessons.items():
            subjects.setdefault(L.get("subject") or "other", []).append(i)
        subj = {sid: {"lessons": {i: lessons[i] for i in ids}, "diagrams": with_dia(ids)} for sid, ids in subjects.items()}
        return {"index": index, "items": items, "subjects": subj, "syllabus": b.get("syllabus"), "tips": b.get("tips") or []}

    def chunk(self, grade, part, id=None, subject=None):
        """one part of a grade's bundle (see split_bundle); None when there is no such part. part 'search': the search
        index (the same for every grade; core/search_index.py)"""
        if part == "search":
            return self.search_ix
        b = self.bundle(grade)
        if not b:
            return None
        key = (int(grade), self.fingerprint(), "split")
        if key not in self._bundles:
            self._bundles[key] = self.split_bundle(b[0])
        sp = self._bundles[key]
        if part == "index":
            return sp["index"]
        if part == "item":
            return sp["items"].get(id)
        if part == "subject":
            return sp["subjects"].get(subject)
        if part == "syllabus":
            return sp["syllabus"]
        if part == "tips":
            return {"tips": sp["tips"]}
        return None

    # ------------------------------------------------------------------------------------------------ syllabus
    # 24 Oct 2026 (the Jesuit). The grade's syllabus travels IN the bundle (so browse, the week view and coverage work
    # offline and on the Netlify demo, which serves the bundle files as exported, with no Python behind it); the GET
    # endpoints below answer the same questions from the server. Fields: SYLLABUS_API.md.
    def syllabus_bundle(self, grade):
        """compact: areas > topics > sub-topics (Funda's pacing) > competences (CDC text, links, coverage)"""
        tree = self.syl.tree(grade)
        if not tree["areas"]:
            return None
        for a in tree["areas"]:
            for t in a["topics"]:
                for s in t["sub_topics"]:
                    s["competences"] = [self._syl_compact(c) for c in s["competences"]]
        cov = self.syl.coverage(grade)
        tree["coverage"] = cov["coverage"].get(str(int(grade)), {})
        tree["coverage_note"] = cov["note"]
        return tree            # not _compact(): it would drop covered=False, the one flag the backlog view needs

    @staticmethod
    def _syl_compact(c):
        return {k: v for k, v in WebAPI._syl_full(c).items() if v is not None}

    @staticmethod
    def _syl_full(c):
        return {"id": c["id"], "code": c["code"], "cdc_code": c.get("cdc_code") if c.get("cdc_code") != c["code"]
                else None, "n": c.get("n"), "title": c["title"], "lang": c.get("lang"),
                "activities": c.get("learning_activities") or [], "standard": c.get("expected_standard"),
                "links": [{k: ln.get(k) for k in ("id", "kind", "title", "match", "served", "needs_review")
                           if k != "needs_review" or ln.get(k)} for ln in c["links"]],
                "covered": c["covered"], "covered_by": c.get("covered_by"),
                "auto_extracted": c.get("auto_extracted"), "prov": c.get("prov"),
                "foundational_weight": (c.get("pacing") or {}).get("foundational_weight"),
                # 24 Oct 2026, Grade 4: the Funda drafts say so on every competence; partial material travels apart
                "draft": c.get("draft_label"),
                "related": [{k: r.get(k) for k in ("id", "kind", "title", "why", "served")}
                            for r in c.get("related") or []] or None}

    def syllabus(self, grade, area=None, topic=None, sub=None):
        if not str(grade or "").isdigit() or int(grade) not in self.syl.grades():
            raise ValueError("grade")
        return self.syl.tree(int(grade), area or None, topic if str(topic or "").isdigit() else None,
                             sub if str(sub or "").isdigit() else None)

    def syllabus_item(self, iid):
        it = self.syl.by_id.get(iid or "")
        if not it:
            raise ValueError("no such competence")
        return self.syl.view(it)

    def syllabus_week(self, grade, term, week=None, area=None):
        if not str(grade or "").isdigit() or int(grade) not in self.syl.grades():
            raise ValueError("grade")
        if str(term or "") not in ("1", "2", "3"):
            raise ValueError("term")
        if week not in (None, "") and not (str(week).isdigit() and 1 <= int(week) <= 14):
            raise ValueError("week")
        return self.syl.week(int(grade), int(term), week, area or None)

    def syllabus_coverage(self, grade=None, area=None):
        if grade not in (None, "") and (not str(grade).isdigit() or int(grade) not in self.syl.grades()):
            raise ValueError("grade")
        return self.syl.coverage(int(grade) if grade not in (None, "") else None, area or None)

    # the written-methods prototype's vocabulary (Tonga captions are WORDS from attested sources, never a sentence
    # we built: a step caption in Tonga mode shows the numbers and these terms; full Tonga step sentences need a teacher)
    TERMS = ("divide", "multiply", "subtract", "add", "carry", "borrow", "bring down", "left over", "column", "ones",
             "tens", "hundreds", "digit", "answer", "step")

    def terms(self):
        return {k: self.resolver.word([k]) for k in self.TERMS}

    def diagram_view(self, did):
        return self.resolver.resolve(self.diagrams[did])

    def lesson_view(self, iid, unit=None):
        it = self.P.items[iid]
        verdict, why = G.item_allowed(self.P.rules, it)
        v = {"id": iid, "kind": it["kind"], "title": it["title"], "subject": it.get("subject"),
             "grades": it.get("grades") or [], "unit": unit or self.P.unit_of.get(iid), "status": it.get("status"),
             "allowed": verdict}
        if it.get("needs_review"):     # Chris 24 Oct: "Nothing is held back, just label it for review"
            v.update(needs_review=True, review_tag=it.get("review_tag"), review_note=it.get("review_note"),
                     review_reason=it.get("review_reason"))
        if it["kind"] == "sample":     # the label travels even when the content is withheld (a health sample)
            v.update(sample=True, label=it.get("label"), curriculum=it.get("curriculum"), level=it.get("level"))
        if verdict == "hold":
            v["withheld"] = "This topic is held for a child-protection review; it is not served yet."
            return v
        if verdict.endswith("_no_card"):
            v["withheld"] = ("Health topics are served only from a card a health worker has checked; this one is not "
                             "checked yet.")
            return v
        if it["kind"] == "pointer":
            v["where"] = it.get("where") or []
            v["missing"] = it.get("missing")
            return v
        if it["kind"] == "route":
            return v
        if it["kind"] == "sample":
            return _compact(self._sample_view(it, v))
        if it["kind"] == "card":
            p = it.get("parts") or {}
            order = [("answer", "In short"), ("idea", "The idea"), ("local_example", "A local example"),
                     ("how_to_teach", "How to teach it"), ("check", "Check"), ("mistakes", "Common mistakes")]
            v["guide"] = {"type": "sections", "label": "English card drafted by Claude, not yet reviewed",
                          "sections": [[t, p[k]] for k, t in order if p.get(k)]
                          + [[k.replace("_", " ").capitalize(), p[k]] for k in p if k not in dict(order) and
                             isinstance(p[k], str)]}
            v["check_questions"] = it.get("check_questions") or []
            # 24 Oct 2026 (cards_v2): the card's classroom key words, English only; the Tonga is never ours to write, so
            # every one is shown as needed (the lesson page's key-word block: "Tonga word needed")
            if it.get("key_words_en"):
                v["key_words"] = [{"en": w, "needed": True, "tonga_needed": True}      # no Tonga key at all (as a
                                  for w in it["key_words_en"]]                         # lesson's needed words)
            return v
        # lesson
        v["scope"] = it.get("scope")
        v["guide"] = GD.guide_for(it, REPO)
        og = GD.older_for(it, REPO)        # 24 Oct 2026 (persona review M2): a guide pitched above this lesson, apart
        if og:
            v["older_guide"] = og
        v["toi_primary"] = bool(it.get("toi_primary"))
        box = []
        for ln in it.get("lines", []):
            if not ln.get("served"):
                continue
            if it.get("toi_primary"):
                if not ln.get("toi"):
                    continue
                # src: the book's code (G, G2, M2, M3, E2, E3, LR, NA; build_chitonga_g1.py / _g23.py) and its page, so
                # the page can say which printed book the Tonga is copied from (Chris, 24 Oct 2026)
                box.append({"n": ln.get("n"), "en": ln.get("en"), "toi": ln["toi"], "badge": "B",
                            "kind": ln.get("kind") if ln.get("kind") != "text" else None,
                            "src": (ln.get("src") or "").split(" ")[0] or None, "page": ln.get("page")})
                continue
            ok = R.line_ok(ln, self.P.glossary)[0]
            box.append({"n": ln.get("n"), "en": ln.get("en"), "toi": ln.get("toi") if ok else None,
                        "origin": list(ln.get("origin") or []), "badge": _badge(ln.get("origin")),
                        "origin_text": ln.get("origin_text"), "held": not ok,
                        "note": ln.get("note") if ok and _badge(ln.get("origin")) == "T" else None})
        v["box"] = box
        kws = []
        for k in it.get("key_words", []):
            missing = not k.get("toi") or k["toi"].strip().startswith("(")
            kws.append({"en": k["en"], "toi": None if missing else k["toi"], "badge": _badge(k.get("origin")),
                        "needed": missing, "note": k.get("note") or ""})
        v["key_words"] = kws
        st = self.P.strategies.get(iid)
        if st:
            v["strategy"] = self._strategy_view(st)
        v["diagrams"] = self.diagram_of.get(iid, [])
        return _compact(v)

    # 24 Oct 2026 (sample lessons, plugins/education/build_samples.py; fields: SAMPLES_API.md). The same compact view
    # shape as a lesson (guide, key_words, strategy) so the lesson page can open one, plus what makes it a sample: the
    # label, the curriculum placement (2023 first; 2013 marked old), the syllabus refs and the guide's own grade band.
    def _sample_view(self, it, v):
        g = it.get("guide") or {}
        v.update({"sample": True, "label": it.get("label"), "origin": "F", "band": it.get("band"),
                  "guide_code": it.get("guide_code"), "part": it.get("part"), "teaches": it.get("teaches"),
                  "what_it_is": it.get("what_it_is"), "curriculum": it.get("curriculum"), "level": it.get("level"),
                  "curriculum_label": ("2023 curriculum" if it.get("curriculum") == "2023" else
                                       "old curriculum (2013)") if it.get("curriculum") else None,
                  "placements": [self._placement_view(p) for p in sorted(
                      it.get("placements") or [], key=lambda p: (p.get("curriculum") != "2023", p.get("grade") or 0))],
                  "syllabus_ref": [self._ref_view(r) for r in it.get("syllabus_ref") or []],
                  "related_refs": [self._ref_view(r) for r in it.get("related_refs") or []],
                  "placement_note": it.get("placement_note"), "prefix": it.get("prefix"),
                  "set_intro": it.get("set_intro"), "role": it.get("role"),
                  # Chris 24 Oct: the three health samples are served with a badge and one line (SAMPLES_API.md)
                  "needs_review": bool(it.get("needs_review")), "review_tag": it.get("review_tag"),
                  "review_note": it.get("review_note")})
        html_body = g.get("html") or (GD.sanitise_funda(os.path.join(REPO, g["funda"])) if g.get("funda") else None)
        v["guide"] = ({"type": "html", "html": html_body, "label": it.get("label")} if html_body else {"type": "none"})
        kws = []
        for k in it.get("key_terms") or []:
            need = bool(k.get("tonga_needed") or not k.get("toi"))
            kws.append({"en": k["en"], "toi": None if need else k["toi"], "badge": None if need else k.get("badge"),
                        "needed": need, "tonga_needed": need, "source": None if need else k.get("source"),
                        "note": k.get("note") or ""})
        v["key_words"] = kws
        v["tonga_needed"] = [k["en"] for k in kws if k["needed"]]
        if it.get("strategy"):
            v["strategy"] = self._strategy_view(it["strategy"])
        return v

    @staticmethod
    def _ref_view(r):
        return {k: r.get(k) for k in ("curriculum", "level", "code", "topic", "sub_topic", "match", "status")
                if r.get(k) is not None}

    def _placement_view(self, p):
        out = {"grade": p.get("grade"), "curriculum": p.get("curriculum"), "level": p.get("level"),
               "label": ("2023 curriculum" if p.get("curriculum") == "2023" else "old curriculum (2013)"),
               "match": p.get("match"), "basis": p.get("basis"), "refs": [self._ref_view(r) for r in p.get("refs") or []]}
        if p.get("in_2023"):
            out["in_2023"] = {"level": p["in_2023"].get("level"), "note": p["in_2023"].get("note"),
                              "refs": [self._ref_view(r) for r in p["in_2023"].get("refs") or []]}
        return out

    def _strategy_view(self, st):
        def tg(t):
            return {"toi": t["toi"], "badge": _badge(t.get("origin"))} if R._tonga_ok(self.P, st, t) else None
        return {"mistakes": [{"id": m["id"], "title": m["title"], "text": m.get("text"), "fix": m.get("fix"),
                              "fix_tonga": tg(m.get("fix_tonga")), "origin": m.get("origin")}
                             for m in st.get("mistakes", [])],
                "moves": [{"when": m.get("when"), "say": m.get("say"),
                           "say_tonga": tg(m.get("say_tonga")) if m.get("say_tonga") else None}
                          for m in st.get("moves", [])],
                "checkpoint": st.get("checkpoint"),
                "intervention": [s["step"] for s in st.get("intervention", [])],
                "escalate": [s["step"] for s in st.get("escalate", [])],
                "challenge": [s["step"] for s in st.get("challenge", [])],
                "label": st.get("label") or "Funda guide and mastery loop, not yet reviewed by a teacher"}

    # ------------------------------------------------------------------------------------------------ search
    def search(self, who, text):
        n = who["number"]
        t = (text or "").strip()[:300]
        if not t:
            return {"results": []}
        sg = G.safeguarding(t)
        if sg:
            self.mem.event(n, "safeguarding_referral", note="web search")
            ref = G.referral(sg)
            return {"safeguarding": {"reply": ref.get("reply") or ref.get("en_meaning")}, "results": []}
        dn = G.deny(t)
        if dn:
            rd = G.redirect(dn)
            return {"redirect": {"reply": rd.get("reply") or rd.get("en_meaning") or
                                 "That is not part of my training. Would you like a lesson, a class struggle or a "
                                 "Tonga word?"}, "results": []}
        cls = self._current_class(n)
        ctx = {"plugin": self.P, "index": self.index, "number": n, "recall": None,
               "grades": cls["grades"] if cls else [1]}
        r = self.reader.read(t, ctx)
        if validate_reading(r, plugin=self.P):
            r = {"intent": "unknown", "confidence": 0, "candidates": [], "source": "invalid"}
        out = {"reading": {k: r.get(k) for k in ("intent", "confidence", "source", "why", "unit", "item")},
               "results": []}
        # 24 Oct 2026: the hits come from the search index (core/search_index.py), the same matcher as the phone
        # (site/assistant/search.js) and the Netlify function, so local and live agree; the reader above still reads
        # the intent (struggle, a route such as the Tonga word question). strong False = nothing called that: the
        # hits are related topics. The old keys (results, samples, syllabus) are filled from the same hits.
        sr = self.searcher.search(t, grades=ctx["grades"], k=10)
        out.update(hits=sr["hits"], strong=sr["strong"], grade_named=sr["grade"], corrected=sr["corrected"],
                   headline="" if sr["strong"] else (f"Nothing called '{t[:40]}' yet." +
                                                      (" Related:" if sr["hits"] else "")))
        ids = [h["id"] for h in sr["hits"] if h["type"] in ("lesson", "card", "sample")]
        hit_grade = {h["id"]: h["grade"] for h in sr["hits"]}
        seen = set()
        samples = []
        for i in ids:
            if i in seen or i not in self.P.items:
                continue
            seen.add(i)
            it = self.P.items[i]
            if it["kind"] == "sample":
                # 24 Oct 2026: sample lessons in their OWN key, never in "results" (the current page opens a result
                # from the class's grade bundle; a Grade 5 sample is not in a Grade 2 bundle). Order as the index
                # gave them: for a Grade 1-3 class after all its material, for a Grade 4-7 class its own grade first.
                if len(samples) < 3:
                    u = self.P.units.get(self.P.unit_of.get(i))
                    samples.append({"id": i, "title": it["title"], "kind": "sample", "subject": it.get("subject"),
                                    "grades": it.get("grades") or [], "grade": hit_grade.get(i) or (u["grade"] if u else None),
                                    "unit": u["id"] if u else None, "curriculum": it.get("curriculum"),
                                    "level": it.get("level"), "label": "Sample lesson",
                                    "needs_review": bool(it.get("needs_review")),
                                    "allowed": G.item_allowed(self.P.rules, it)[0]})
                continue
            u = self.P.units.get(self.P.unit_of.get(i))
            out["results"].append({"id": i, "title": it["title"], "kind": it["kind"],
                                   "grade": hit_grade.get(i) or (it.get("grades") or [u["grade"] if u else None])[0],
                                   "subject": it.get("subject"), "unit": u["id"] if u else None,
                                   "unit_label": u["label"] if u else None,
                                   "in_menu": bool(u), "allowed": G.item_allowed(self.P.rules, it)[0],
                                   "needs_review": bool(it.get("needs_review"))})
            if len(out["results"]) >= 8:
                break
        if samples:
            out["samples"] = samples
        if r.get("intent") == "struggle":
            out["struggle"] = {"text": t, "unit": r.get("unit")}
        # 24 Oct 2026: competences of the syllabus, in their OWN key and after the material (a lesson or card that
        # answers the request is the answer; the competence is reference). Kept out of "results" so the current page,
        # which opens every result as a lesson, is not handed an item it cannot open (SYLLABUS_API.md).
        syl = [self.syl.by_id[h["id"]] for h in sr["hits"] if h["type"] == "syllabus" and h["id"] in self.syl.by_id][:3]
        if syl:
            out["syllabus"] = [self._syl_compact(self.syl.view(it)) | {"grade": it["grade"], "area": it["area"],
                                                                         "sub_topic": it["sub_topic"]["title"]}
                               for it in syl]
        if r.get("item") and str(r["item"]).startswith("route:") and self.hooks:
            try:
                if "ctx" in inspect.signature(self.hooks.run).parameters:
                    res = self.hooks.run(r["item"], t, "both", ctx={"plugin": self.P, "grades": ctx["grades"],
                                                                     "now": self.clock()})
                else:
                    res = self.hooks.run(r["item"], t, "both")
            except Exception as e:  # noqa: BLE001
                res, out["route_error"] = None, str(e)[:200]
            if res:
                out["route"] = {"kind": r["item"].split(":")[1], "body": res["body"], "origins": res.get("origins")}
        word = self._word_question(t)
        if not word and sr.get("word"):
            # 24 Oct 2026: every Tonga-word question gets a word card ('leaf in Tonga', 'how do you say cow', 'Tonga
            # word for X'; core.search_index.parse_word): the attested word, else needed (the page: "Add it")
            word = self.word_card(sr["word"]["en"])
        if word:
            out["word"] = word
        held = self._log_request(who, t, r, bool(sr["strong"] or out.get("route") or out.get("word")))
        out["held_out"] = held
        return out

    WORD_Q = re.compile(r"(?i)^\s*(?:how\s+do\s+(?:you|i|we)\s+say|what\s+is|what's|tonga\s+(?:word\s+)?for|"
                        r"the\s+word\s+for)\s+(?:the\s+word\s+)?['\"]?([a-z][a-z \-]{1,30}?)['\"]?"
                        r"(?:\s+in\s+(?:chi)?tonga)?\s*\??\s*$")

    def _word_question(self, t):
        """'how do you say heart in Tonga' -> the attested word (diagrams.Resolver order) or 'needed'"""
        m = self.WORD_Q.match(t)
        if not m or not re.search(r"(?i)tonga|word", t):
            return None
        en = m.group(1).strip().lower()
        w = self.resolver.word([en])
        return dict(w, en=en)

    def word_card(self, en):
        """the Resolver's answer for an English word, trying the singular too ('cows' -> 'cow'); needed when none"""
        en = en.strip().lower()
        keys = [en] + ([en[:-1]] if en.endswith("s") and len(en) > 3 else [])
        return dict(self.resolver.word(keys), en=en)

    def _current_class(self, n):
        prof = self.mem.get_teacher(n) or {}
        c = self.mem.get_class(prof["last_class_id"]) if prof.get("last_class_id") else None
        if c and c["teacher"] == n:
            return c
        cls = self.mem.classes(n)
        return cls[0] if cls else None

    def _log_request(self, who, text, reading, answered):
        n = who["number"]
        self.mem.log_request(n, reading.get("intent") or "unknown", reading.get("item"),
                             outcome="web_search" if answered else "no_material", text=text)
        path = os.path.join(self.logs, "web_requests.jsonl")
        with self._wlock:
            k = 0
            try:
                with open(path, encoding="utf-8") as f:
                    k = sum(1 for x in f if f'"teacher": "{who.get("code")}"' in x)
            except OSError:
                pass
            rec = {"at": self.clock().isoformat(timespec="seconds"), "teacher": who.get("code"),
                   "text": None if pupil_level(text) else text, "reading": {x: reading.get(x) for x in (
                       "intent", "item", "unit", "confidence", "source")}, "answered": answered,
                   "held_out": k < HELD_OUT_FIRST}
            os.makedirs(self.logs, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not answered:
            self._writer(who).capture(n, self.clock().isoformat(timespec="seconds"),
                                      {"path": ["no_material"], "flywheel": []}, {"text": None if pupil_level(text)
                                                                                     else text}, {})
        return k < HELD_OUT_FIRST

    # ------------------------------------------------------------------------------------------------ writes
    def sync_batch(self, who, ops):
        """[{client_id, type, at, ...}] -> [{client_id, ok, result | error, duplicate}] (idempotent per client id)"""
        out = []
        for op in ops[:200]:
            cid = str(op.get("client_id") or "")
            if not re.fullmatch(r"[A-Za-z0-9_\-]{8,64}", cid):
                out.append({"client_id": cid, "ok": False, "error": "bad client id"})
                continue
            prev = self.sync.get(cid)
            if prev is not None:
                out.append({"client_id": cid, "ok": True, "duplicate": True, "result": prev})
                continue
            fn = getattr(self, "op_" + str(op.get("type")), None)
            if not fn:
                out.append({"client_id": cid, "ok": False, "error": f"unknown type {op.get('type')}"})
                continue
            try:
                with self.mem.transaction():
                    res = fn(who, op)
            except ValueError as e:
                out.append({"client_id": cid, "ok": False, "error": str(e)[:200]})
                self.sync.put(cid, who.get("code"), op.get("type"), {"error": str(e)[:200]})
                continue
            self.sync.put(cid, who.get("code"), op.get("type"), res)
            out.append({"client_id": cid, "ok": True, "result": res})
        return out

    def _own_class(self, who, cid):
        c = self.mem.get_class(int(cid)) if str(cid or "").isdigit() else None
        if not c or c["teacher"] != who["number"]:
            raise ValueError("not your class")
        return c

    def op_open(self, who, op):
        """a lesson opened: the class's position moves there (shared with WhatsApp), the request is logged"""
        iid = op.get("item")
        if iid not in self.P.items:
            raise ValueError("no such item")
        c = self._own_class(who, op.get("class_id"))
        unit = self.P.unit_of.get(iid)
        subject = self.P.units[unit]["subject"] if unit in self.P.units else self.P.items[iid].get("subject")
        cur = self.mem.position(c["id"], self.P.name, subject) or {}
        part = int(op.get("part") or 0)
        if cur.get("item") == iid and cur.get("finished"):
            return {"position": cur}
        pos = self.mem.set_position(c["id"], self.P.name, subject, unit=unit, item=iid, part=part,
                                    parts=int(op.get("parts") or 0), finished=False)
        self.mem.update_teacher(who["number"], last_class_id=c["id"])
        self.mem.log_request(who["number"], "lesson", iid, outcome="web_open")
        if unit in self.P.units and not self.P.units[unit]["items"]:
            self._writer(who).capture(who["number"], self._at(op), {"path": [f"no_content:unit:{unit}"]}, {}, {})
        return {"position": pos}

    def op_progress(self, who, op):
        """'taught': the lesson is finished for that class; home then offers the next unit"""
        iid = op.get("item")
        if iid not in self.P.items:
            raise ValueError("no such item")
        c = self._own_class(who, op.get("class_id"))
        unit = self.P.unit_of.get(iid)
        subject = self.P.units[unit]["subject"] if unit in self.P.units else self.P.items[iid].get("subject")
        pos = self.mem.set_position(c["id"], self.P.name, subject, unit=unit, item=iid,
                                    finished=bool(op.get("finished", True)))
        nu = self.P.next_unit(unit) if unit else None
        return {"position": pos, "next_unit": {"id": nu["id"], "label": nu["label"]} if nu else None}

    def op_struggle(self, who, op):
        text = re.sub(r"\s+", " ", str(op.get("text") or "")).strip()[:300]
        if not text:
            raise ValueError("empty")
        if G.safeguarding(text):
            self.mem.event(who["number"], "safeguarding_referral", note="web struggle form")
            ref = G.referral("web")
            return {"safeguarding": {"reply": ref.get("reply") or ref.get("en_meaning")}}
        why = pupil_level(text)
        if why:
            return {"refused": "This looks like it is about one learner (" + why + "). For the class as a whole I "
                               "can help; names and marks are not stored."}
        c = self._own_class(who, op.get("class_id"))
        unit = op.get("unit") if op.get("unit") in self.P.units else None
        if not unit:
            r = self.reader.read(text, {"plugin": self.P, "index": self.index, "number": who["number"],
                                        "recall": None, "grades": c["grades"]})
            unit = r.get("unit") if r.get("unit") in self.P.units else None
        if not unit:
            pos = [p for p in self.mem.positions(c["id"]) if p["plugin"] == self.P.name]
            unit = pos[0]["unit"] if pos else None
        s, m, score = self.eng._match_mistake(text, unit)
        subject = self.P.units[unit]["subject"] if unit in self.P.units else None
        oid = self.mem.add_observation(c["id"], self.P.name, text, unit=unit, subject=subject,
                                       mistake=m["id"] if m else None)
        if not s:
            return {"matched": False, "observation_id": oid, "class_label": c["label"]}
        what = m["title"] if m else self.P.items[s["lesson"]]["title"]
        sid = self.mem.add_suggestion(c["id"], self.P.name, oid, s["lesson"], "intervention",
                                      m["id"] if m else None, what=what)
        sv = self._strategy_view(s)
        mv = next((x for x in sv["mistakes"] if m and x["id"] == m["id"]), None)
        return {"matched": True, "observation_id": oid, "suggestion_id": sid, "class_label": c["label"],
                "lesson": {"id": s["lesson"], "title": self.P.items[s["lesson"]]["title"]},
                "mistake": mv, "steps": sv["intervention"], "escalate": sv["escalate"],
                "challenge": sv["challenge"], "scope": s.get("scope"), "label": sv["label"],
                "drafted_by_claude": bool(m and m.get("origin") == "C")}

    def op_followup(self, who, op):
        sid = int(op.get("suggestion_id") or 0)
        s = self.mem.suggestion(sid)
        if not s:
            raise ValueError("no such suggestion")
        self._own_class(who, s["class_id"])
        if op.get("outcome") not in ("helped", "partly", "no"):
            raise ValueError("outcome")
        self.mem.mark_followup_asked(sid)
        self.mem.record_outcome(sid, op["outcome"])
        return {"recorded": op["outcome"]}

    def _paid_today(self, who):
        path = os.path.join(self.logs, "flywheel_web.jsonl")
        today = self.clock().date().isoformat()
        k = 0
        try:
            with open(path, encoding="utf-8") as f:
                for x in f:
                    r = json.loads(x)
                    if (r.get("teacher") == who.get("code") and r.get("ev") == "answer" and r.get("pay")
                            and str(r.get("at", "")).startswith(today)):
                        k += 1
        except OSError:
            pass
        return k

    def _at(self, op):
        a = str(op.get("at") or "")
        return a if re.fullmatch(r"\d{4}-\d\d-\d\dT[\d:.]+Z?([+-]\d\d:\d\d)?", a) else \
            self.clock().isoformat(timespec="seconds")

    def op_flywheel(self, who, op):
        """a tap or a written line from the lesson page / a diagram / a unit without material.
        kind: tonga_check (answer ok | fix), derived_confirm (right | write), write_line (write), unit_line (write),
        word_needed (word: a key word or diagram label with no attested Tonga, K1).
        The teacher's text is kept VERBATIM beside the served line (never overwriting it)."""
        if not FLYWHEEL_IN_APP:
            raise ValueError("contributions are collected on the WhatsApp desk (flywheel in the app is off)")
        kind, answer = op.get("kind"), op.get("answer")
        allowed = {"tonga_check": ("ok", "fix"), "derived_confirm": ("right", "write"), "write_line": ("write",),
                   "unit_line": ("write",), "word_needed": ("word",)}
        if kind not in allowed or answer not in allowed[kind]:
            raise ValueError("kind / answer")
        text = op.get("teacher_text")
        text = text.strip()[:400] if isinstance(text, str) else None
        if answer in ("fix", "write", "word") and not text:
            raise ValueError("the Tonga is empty")
        if text and G.safeguarding(text):
            self.mem.event(who["number"], "safeguarding_referral", note="web flywheel text")
            ref = G.referral("web")
            return {"safeguarding": {"reply": ref.get("reply") or ref.get("en_meaning")}}
        item = op.get("item")
        if item is not None and item not in self.P.items and not str(item).startswith("diagram:"):
            raise ValueError("no such item")
        served = self._served_line(item, op.get("n")) if item in self.P.items else None
        if kind == "word_needed":
            served = {"en": str(op.get("en") or "")[:80], "diagram": op.get("diagram"), "label": op.get("label")}
        elif kind == "unit_line":
            served = {"unit": op.get("unit")}
            if text and "=" in text:
                toi, en = text.split("=", 1)
                served.update(teacher_toi=toi.strip(), teacher_en=en.strip() or None)
        pay = PAY.get({"ok": "tonga_ok", "fix": "fix", "right": "right", "write": "write", "word": "word"}[answer], 0)
        capped = False
        if pay and self._paid_today(who) >= DAILY_PAID_CAP:
            pay, capped = 0, True
        ev = {"ev": "answer", "at": self._at(op), "pid": op["client_id"][:12], "kind": kind, "answer": answer,
              "item": item, "part": None, "unit": op.get("unit") or (self.P.unit_of.get(item) if item else None),
              "served": served, "lines": op.get("lines") if answer == "ok" else None, "teacher_text": text,
              "pay": pay, "control": None, "channel": "web", "client_id": op["client_id"],
              "over_cap": capped or None}
        recs = self._writer(who).capture(who["number"], self.clock().isoformat(timespec="seconds"),
                                         {"flywheel": [ev], "path": []}, {}, {"request_id": op["client_id"]})
        return {"recorded": recs[0]["id"] if recs else None, "pay": pay, "over_cap": capped,
                "tester": bool(who.get("tester"))}

    def _served_line(self, item, n):
        it = self.P.items.get(item) or {}
        ln = next((x for x in it.get("lines", []) if str(x.get("n")) == str(n)), None)
        if not ln:
            return None
        return {k: ln.get(k) for k in ("n", "en", "toi", "origin") if ln.get(k) is not None}

    def op_report(self, who, op):
        """'report a problem with this line' (unpaid, optional): the note VERBATIM beside the served line, in
        web_logs/reports.jsonl, for the main session to read with the desk material. Never overwrites a line."""
        item = op.get("item")
        if item not in self.P.items:
            raise ValueError("no such item")
        text = str(op.get("text") or "").strip()[:400]
        if not text:
            raise ValueError("empty")
        if G.safeguarding(text):
            self.mem.event(who["number"], "safeguarding_referral", note="web report text")
            ref = G.referral("web")
            return {"safeguarding": {"reply": ref.get("reply") or ref.get("en_meaning")}}
        rec = {"id": uuid.uuid4().hex[:12], "at": self._at(op), "teacher": who.get("code"),
               "tester": bool(who.get("tester")), "item": item, "served": self._served_line(item, op.get("n")),
               "teacher_text": None if pupil_level(text) else text, "client_id": op["client_id"], "channel": "web"}
        os.makedirs(self.logs, exist_ok=True)
        with self._wlock, open(os.path.join(self.logs, "reports.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"recorded": rec["id"]}

    def op_contribute(self, who, op):
        """24 Oct 2026: a gap filled in the app (a Tonga word where one is needed; a few English lines for a competence
        with no lesson). Unpaid, never served directly: web_logs/contributions.jsonl, for review on the desk (a second
        teacher checks it; pay follows the desk rules there). Verbatim; credit = may her initials be shown once approved."""
        kind = str(op.get("kind") or "")
        # search_gap (24 Oct 2026): "Tell us what you were looking for" under a search that found nothing called that
        # (en = what she typed, text = what she was looking for); flywheel_export.py counts them in the content backlog
        if kind not in ("word", "competence_lesson", "search_gap"):
            raise ValueError("kind")
        text = str(op.get("text") or "").strip()[:120 if kind == "word" else 400 if kind == "search_gap" else 1200]
        if not text:
            raise ValueError("empty")
        note = str(op.get("note") or "").strip()[:400] or None
        if G.safeguarding(text + " " + (note or "")):
            self.mem.event(who["number"], "safeguarding_referral", note="web contribution")
            ref = G.referral("web")
            return {"safeguarding": {"reply": ref.get("reply") or ref.get("en_meaning")}}
        rec = {"id": uuid.uuid4().hex[:12], "at": self._at(op), "teacher": who.get("code"), "tester": bool(who.get("tester")),
               "kind": kind, "en": (str(op.get("en") or "")[:120] or None), "item": op.get("item"), "label": op.get("label"),
               "competence": op.get("competence"), "text": None if pupil_level(text) else text,
               "note": None if (note and pupil_level(note)) else note, "lang": "toi" if kind == "word" else "en",
               "credit": op.get("credit") is not False, "status": "to_review", "client_id": op["client_id"], "channel": "web"}
        os.makedirs(self.logs, exist_ok=True)
        with self._wlock, open(os.path.join(self.logs, "contributions.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"recorded": rec["id"]}

    def op_class(self, who, op):
        """add a class later (the 'Me' page)"""
        me = self.onboard(who, {"classes": [{"label": op.get("label"), "grades": op.get("grades"),
                                             "size": op.get("size")}]})
        return {"classes": me["classes"]}
