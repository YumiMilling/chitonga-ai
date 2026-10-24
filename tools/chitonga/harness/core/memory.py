"""Teacher memory, shared by every plug-in (23 Oct 2026, the Jesuit; Chris: "the system needs to remember the teacher,
just as an ERP remembers different customers"; "Not starting from 0 every time"; the classes and the strategy
follow-up, same day).

PATTERN, learnt from an earlier WhatsApp business responder by the same author (not in this repo): one row per counterparty
keyed by the number normalised to '260' + the last 9 digits, a short running note, `updated_at` / `updated_by`; the
responder reads it before answering and writes it after. Here the same idea with the fields a teacher assistant needs.
No data of that responder was copied.

WHAT IS KEPT (data minimisation: only what the assistant needs to serve the teacher):
  teachers        number, name, school, district, language preference (en / toi / both), consent note, last class
  classes         1:n per teacher (two shifts, subject teachers, multigrade): label as the teacher names it, grade(s),
                  subjects, size if given, conditions (textbooks, chalkboard, paper, electricity), readable in the
                  shared TEACHER_PROFILE.md v1.1 shape (class_profile)
  positions       per class, plug-in and subject: unit, item, part, parts, finished (where they are in the syllabus)
  observations    CLASS-LEVEL only ("most struggle with carrying"), dated, with the unit and the matched mistake
  suggestions     what the assistant suggested for an observation, when the follow-up was asked, the outcome
  requests        recent requests: intent, item served, outcome; the text only when it carries no pupil data
  open_items      open questions / confirmations for the teacher
  events          e.g. 'safeguarding_referral' (the EVENT only: never the disclosure, never a profile fact)
  state           the conversation state (JSON), so an interrupted path resumes where it stopped
NEVER KEPT: pupil names, marks or cases. `pupil_level()` refuses an observation that names or scores one learner; the
engine then stores nothing of it. A safeguarding message is never stored as text anywhere (the log withholds it too).

TRANSACTIONS (24 Oct, review item 8): `transaction()` wraps a turn (BEGIN ... COMMIT; nested calls are SAVEPOINTs), so
a crash mid-turn leaves no observation without its suggestion, and a Supabase store can map a turn to one RPC.

STORAGE: SQLite under internal/ (never site/), behind `MemoryStore`, so the same calls can move to Supabase tables
(zamai_*) later without touching the engine.
"""
import contextlib
import datetime
import json
import os
import re
import sqlite3
import threading

DEFAULT_DB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                                           "internal", "chitonga_train", "harness", "teacher_memory.sqlite"))


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def norm_number(n):
    """'+260 97 000 0001' / '0970000001' / '260970000001' -> '260970000001' (fictional numbers)"""
    d = re.sub(r"\D", "", str(n or ""))
    return "260" + d[-9:] if len(d) >= 9 else d


# ------------------------------------------------------------------------------------------------ pupil-level guard
GROUP = re.compile(r"(?i)\b(most|many|some|several|few|half|a\s+third|third|quarter|all|none|nobody|everyone|the\s+class|"
                   r"my\s+class|the\s+learners|my\s+learners|the\s+pupils|my\s+pupils|the\s+children|they|them|"
                   r"learners|pupils|children|students|\d+\s+(of|out\s+of)\s+(my\s+|the\s+)?\d*\s*(learners|pupils|"
                   r"children|them)|bana|basicikolo)\b")
ONE = re.compile(r"(?i)\b(one\s+(pupil|learner|child|boy|girl|student)|a\s+(pupil|learner|child|boy|girl|student)\s+"
                 r"(named|called)|(named|called)\s+[A-Z]|my\s+(pupil|learner|student)\s+[A-Z]|\b(he|she|his|her)\b)")
MARKS = re.compile(r"(?i)\b(scored|got\s+\d+\s*(/|out\s+of|marks|%)|\d+\s*%|\d+\s*/\s*\d+\s+marks?|marks?\s+of\s+\d+)")
NOT_NAMES = {"Grade", "English", "Tonga", "Chitonga", "Maths", "Math", "Science", "Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Term", "Day", "Most", "Many", "Some", "My", "The", "They", "I", "Half", "All", "A",
             "Few", "Zambia", "Lusaka", "Choma", "Monze", "Kalomo", "Southern", "Province", "Ok", "OK", "Yes", "No"}
NAME_VERB = re.compile(r"\b([A-Z][a-z]{2,})\s+(can't|cannot|can|could|couldn't|struggles|struggled|failed|fails|"
                       r"does|did|doesn't|didn't|is|was|got|scored|keeps|writes|reads|counts)\b")


def pupil_level(text):
    """the reason the text is about one learner (and must not be stored), or '' when it is about the class"""
    t = text or ""
    if MARKS.search(t):
        return "marks or a score"
    m = NAME_VERB.search(t)
    if m and m.group(1) not in NOT_NAMES:
        return f"a learner's name ({m.group(1)})"
    if ONE.search(t) and not GROUP.search(t):
        return "one learner"
    return ""


# ------------------------------------------------------------------------------------------------ the interface
class MemoryStore:
    """the calls the engine makes; SQLiteMemory implements them, a Supabase store would implement the same"""
    def get_teacher(self, number): raise NotImplementedError
    def get_or_create(self, number): raise NotImplementedError
    def update_teacher(self, number, **fields): raise NotImplementedError
    def classes(self, number): raise NotImplementedError
    def add_class(self, number, label, grades, subjects=(), size=None, kind="class", plugin="education"):
        raise NotImplementedError
    def get_class(self, class_id): raise NotImplementedError
    def position(self, class_id, plugin, subject): raise NotImplementedError
    def positions(self, class_id): raise NotImplementedError
    def set_position(self, class_id, plugin, subject, **fields): raise NotImplementedError
    def add_observation(self, class_id, plugin, text, unit=None, subject=None, mistake=None): raise NotImplementedError
    def observations(self, class_id): raise NotImplementedError
    def add_suggestion(self, class_id, plugin, observation_id, item, kind, mistake=None, what=""): raise NotImplementedError
    def pending_followup(self, number, before=None): raise NotImplementedError
    def mark_followup_asked(self, suggestion_id): raise NotImplementedError
    def record_outcome(self, suggestion_id, outcome): raise NotImplementedError
    def strategy_evidence(self, plugin): raise NotImplementedError
    def log_request(self, number, intent, item=None, outcome=None, text=None): raise NotImplementedError
    def recent_requests(self, number, limit=10): raise NotImplementedError
    def recall(self, number, text): raise NotImplementedError
    def add_open_item(self, number, kind, text): raise NotImplementedError
    def open_items(self, number): raise NotImplementedError
    def event(self, number, kind, note=""): raise NotImplementedError
    def events(self, number): raise NotImplementedError
    def get_state(self, number): raise NotImplementedError
    def set_state(self, number, state): raise NotImplementedError

    def transaction(self):
        """one unit of work (a turn); a store without transactions may keep this no-op"""
        return contextlib.nullcontext()


SCHEMA = """
create table if not exists teachers (number text primary key, name text, school text, district text,
  language_pref text check (language_pref in ('en','toi','both') or language_pref is null), consent_note text,
  last_class_id integer, created_at text, updated_at text, updated_by text default 'harness');
create table if not exists classes (id integer primary key autoincrement, teacher text not null, label text not null,
  grades text not null, subjects text, size integer, conditions text, created_at text, updated_at text,
  kind text default 'class', plugin text default 'education');
create table if not exists positions (class_id integer not null, plugin text not null, subject text not null,
  unit text, item text, part integer default 0, parts integer default 0, finished integer default 0, updated_at text,
  primary key (class_id, plugin, subject));
create table if not exists observations (id integer primary key autoincrement, class_id integer not null, plugin text,
  subject text, unit text, text text not null, mistake text, created_at text);
create table if not exists suggestions (id integer primary key autoincrement, class_id integer not null, plugin text,
  observation_id integer, item text, kind text, mistake text, what text, suggested_at text, followup_asked_at text,
  outcome text check (outcome in ('helped','partly','no') or outcome is null), outcome_at text);
create table if not exists requests (id integer primary key autoincrement, teacher text not null, at text, intent text,
  item text, outcome text, text text);
create table if not exists open_items (id integer primary key autoincrement, teacher text not null, kind text, text text,
  created_at text, closed_at text);
create table if not exists events (id integer primary key autoincrement, teacher text not null, kind text, note text,
  at text);
create table if not exists state (teacher text primary key, json text, updated_at text);
"""

# columns added after the first release, per table: (column, declaration); _migrate adds what an older file lacks
MIGRATIONS = {"classes": [("kind", "text default 'class'"), ("plugin", "text default 'education'")]}


def _norm_text(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", (t or "").lower())).strip()


class SQLiteMemory(MemoryStore):
    def __init__(self, path=None):
        self.path = path or DEFAULT_DB
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # autocommit mode: every statement commits alone unless a transaction() is open (then BEGIN ... COMMIT)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()       # held for a whole transaction: one turn at a time on this connection
        self._depth = 0
        with self.lock:
            self._migrate()
            self.db.executescript(SCHEMA)

    def _migrate(self):
        """add the columns a file made by an older version lacks (classes.kind / classes.plugin, 24 Oct); old rows get
        the defaults (kind 'class', plugin 'education': every class so far is the education plug-in's)"""
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in self.db.execute(f"pragma table_info({table})").fetchall()}
            if not have:
                continue                                  # a new file: SCHEMA creates the table complete
            for col, decl in cols:
                if col not in have:
                    self.db.execute(f"alter table {table} add column {col} {decl}")

    @contextlib.contextmanager
    def transaction(self):
        """BEGIN ... COMMIT around a turn; nested = SAVEPOINT (the engine's inner savepoint rolls back a failed
        _turn and still commits the state and the teacher row). Holds the store lock throughout."""
        with self.lock:
            depth = self._depth
            self.db.execute("begin" if depth == 0 else f"savepoint sp{depth}")
            self._depth += 1
            try:
                yield self
            except BaseException:
                self._depth -= 1
                if depth == 0:
                    self.db.execute("rollback")
                else:
                    self.db.execute(f"rollback to sp{depth}")
                    self.db.execute(f"release sp{depth}")
                raise
            self._depth -= 1
            self.db.execute("commit" if depth == 0 else f"release sp{depth}")

    def _x(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args)

    def _one(self, sql, args=()):
        with self.lock:                     # fetch under the lock too: a shared connection's cursor is not thread-safe
            r = self._x(sql, args).fetchone()
        return dict(r) if r else None

    def _all(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self._x(sql, args).fetchall()]

    # teachers
    def get_teacher(self, number):
        return self._one("select * from teachers where number=?", (norm_number(number),))

    def get_or_create(self, number):
        """(profile, created: bool)"""
        n = norm_number(number)
        t = self.get_teacher(n)
        if t:
            return t, False
        self._x("insert into teachers (number, created_at, updated_at) values (?,?,?)", (n, now(), now()))
        return self.get_teacher(n), True

    def update_teacher(self, number, **fields):
        ok = {"name", "school", "district", "language_pref", "consent_note", "last_class_id"}
        bad = set(fields) - ok
        if bad:
            raise ValueError(f"not a teacher field: {sorted(bad)}")
        if not fields:
            return self.get_teacher(number)
        sets = ", ".join(f"{k}=?" for k in fields)
        self._x(f"update teachers set {sets}, updated_at=? where number=?",
                (*fields.values(), now(), norm_number(number)))
        return self.get_teacher(number)

    # classes
    def _cls(self, r):
        if r:
            r["grades"] = json.loads(r["grades"] or "[]")
            r["subjects"] = json.loads(r["subjects"] or "[]")
        return r

    def classes(self, number):
        return [self._cls(r) for r in self._all("select * from classes where teacher=? order by id", (norm_number(number),))]

    def add_class(self, number, label, grades, subjects=(), size=None, kind="class", plugin="education"):
        """kind: what the unit of memory is ('class' for education; a farmer group or a clinic cohort later);
        plugin: whose it is"""
        cur = self._x("insert into classes (teacher, label, grades, subjects, size, created_at, updated_at, kind, plugin) "
                      "values (?,?,?,?,?,?,?,?,?)", (norm_number(number), label,
                                                     json.dumps(sorted(set(int(g) for g in grades))),
                                                     json.dumps(list(subjects)), size, now(), now(), kind, plugin))
        return self.get_class(cur.lastrowid)

    def get_class(self, class_id):
        return self._cls(self._one("select * from classes where id=?", (class_id,)))

    def class_profile(self, class_id):
        """the class in the shared TEACHER_PROFILE.md v1.1 shape (h3_session.parse_profile: grades, learners,
        textbooks, chalkboard, paper, electricity, multigrade, subjects), so the h3 harness and this one read one
        record. Only class-level facts; never a learner."""
        c = self.get_class(class_id)
        if not c:
            return None
        d = dict(json.loads(c.get("conditions") or "{}"), grades=c["grades"], subjects=c["subjects"],
                 multigrade=len(c["grades"]) > 1)
        if c.get("size"):
            d["learners"] = c["size"]
        try:
            import h3_session  # noqa: PLC0415
            return h3_session.parse_profile(d)
        except Exception:  # noqa: BLE001
            return d

    def set_class_conditions(self, class_id, **cond):
        """class-level conditions from TEACHER_PROFILE.md §1 (textbooks, chalkboard, paper, electricity); size =
        learners"""
        ok = {"textbooks", "chalkboard", "paper", "electricity", "learners"}
        bad = set(cond) - ok
        if bad:
            raise ValueError(f"not a class condition: {sorted(bad)}")
        c = self.get_class(class_id)
        cur = json.loads(c.get("conditions") or "{}")
        size = cond.pop("learners", None)
        cur.update(cond)
        self._x("update classes set conditions=?, size=coalesce(?, size), updated_at=? where id=?",
                (json.dumps(cur), size, now(), class_id))
        return self.class_profile(class_id)

    # positions
    def position(self, class_id, plugin, subject):
        r = self._one("select * from positions where class_id=? and plugin=? and subject=?", (class_id, plugin, subject))
        if r:
            r["finished"] = bool(r["finished"])
        return r

    def positions(self, class_id):
        return self._all("select * from positions where class_id=? order by updated_at desc", (class_id,))

    def set_position(self, class_id, plugin, subject, **fields):
        cur = self.position(class_id, plugin, subject) or {"unit": None, "item": None, "part": 0, "parts": 0,
                                                            "finished": False}
        cur.update(fields)
        self._x("insert or replace into positions (class_id, plugin, subject, unit, item, part, parts, finished, "
                "updated_at) values (?,?,?,?,?,?,?,?,?)", (class_id, plugin, subject, cur["unit"], cur["item"],
                                                          cur["part"], cur["parts"], int(bool(cur["finished"])), now()))
        return self.position(class_id, plugin, subject)

    # observations and suggestions
    def add_observation(self, class_id, plugin, text, unit=None, subject=None, mistake=None):
        why = pupil_level(text)
        if why:
            raise ValueError(f"pupil-level data is not stored: {why}")
        cur = self._x("insert into observations (class_id, plugin, subject, unit, text, mistake, created_at) "
                      "values (?,?,?,?,?,?,?)", (class_id, plugin, subject, unit, (text or "")[:200], mistake, now()))
        return cur.lastrowid

    def observations(self, class_id):
        return self._all("select * from observations where class_id=? order by id", (class_id,))

    def add_suggestion(self, class_id, plugin, observation_id, item, kind, mistake=None, what=""):
        cur = self._x("insert into suggestions (class_id, plugin, observation_id, item, kind, mistake, what, "
                      "suggested_at) values (?,?,?,?,?,?,?,?)",
                      (class_id, plugin, observation_id, item, kind, mistake, what[:120], now()))
        return cur.lastrowid

    def pending_followup(self, number, before=None):
        """the oldest suggestion for this teacher whose follow-up was never asked (made before `before`, an ISO time)"""
        q = ("select s.*, o.text as observation, c.label as class_label from suggestions s join classes c on "
             "c.id=s.class_id left join observations o on o.id=s.observation_id where c.teacher=? and "
             "s.followup_asked_at is null and s.outcome is null and s.kind='intervention'")
        args = [norm_number(number)]
        if before:
            q += " and s.suggested_at < ?"
            args.append(before)
        return self._one(q + " order by s.id limit 1", tuple(args))

    def mark_followup_asked(self, suggestion_id):
        self._x("update suggestions set followup_asked_at=? where id=?", (now(), suggestion_id))

    def record_outcome(self, suggestion_id, outcome):
        if outcome not in ("helped", "partly", "no"):
            raise ValueError(outcome)
        self._x("update suggestions set outcome=?, outcome_at=? where id=?", (outcome, now(), suggestion_id))

    def suggestion(self, suggestion_id):
        return self._one("select * from suggestions where id=?", (suggestion_id,))

    def strategy_evidence(self, plugin):
        """aggregate, anonymous: per (item, mistake, kind) the outcome counts across all teachers; no numbers, no names"""
        rows = self._all("select item, mistake, kind, outcome, count(*) n from suggestions where plugin=? "
                         "group by item, mistake, kind, outcome", (plugin,))
        out = {}
        for r in rows:
            k = (r["item"], r["mistake"], r["kind"])
            out.setdefault(k, {"helped": 0, "partly": 0, "no": 0, "unknown": 0})
            out[k][r["outcome"] or "unknown"] += r["n"]
        return [{"item": k[0], "mistake": k[1], "kind": k[2], **v} for k, v in sorted(out.items(), key=str)]

    # requests, open items, events
    def log_request(self, number, intent, item=None, outcome=None, text=None):
        if text and pupil_level(text):
            text = None                                   # never store a pupil's name or mark, even in passing
        self._x("insert into requests (teacher, at, intent, item, outcome, text) values (?,?,?,?,?,?)",
                (norm_number(number), now(), intent, item, outcome, _norm_text(text)[:160] if text else None))

    def recent_requests(self, number, limit=10):
        return self._all("select * from requests where teacher=? order by id desc limit ?", (norm_number(number), limit))

    def recall(self, number, text):
        """the item served the last time this teacher sent the same words (memory of seen requests), or None. A later
        "Not this one" on that item (outcome 'not_this', 24 Oct) cancels it: recall must not repeat a wrong reading."""
        r = self._one("select item, outcome from requests where teacher=? and text=? and item is not null and outcome in "
                      "('served','not_this') order by id desc limit 1", (norm_number(number), _norm_text(text)[:160]))
        return r["item"] if r and r["outcome"] == "served" else None

    def add_open_item(self, number, kind, text):
        return self._x("insert into open_items (teacher, kind, text, created_at) values (?,?,?,?)",
                       (norm_number(number), kind, text[:200], now())).lastrowid

    def open_items(self, number):
        return self._all("select * from open_items where teacher=? and closed_at is null order by id",
                         (norm_number(number),))

    def event(self, number, kind, note=""):
        self._x("insert into events (teacher, kind, note, at) values (?,?,?,?)", (norm_number(number), kind, note[:120], now()))

    def events(self, number):
        return self._all("select * from events where teacher=? order by id", (norm_number(number),))

    # conversation state
    def get_state(self, number):
        r = self._one("select json from state where teacher=?", (norm_number(number),))
        return json.loads(r["json"]) if r else None

    def set_state(self, number, state):
        self._x("insert or replace into state (teacher, json, updated_at) values (?,?,?)",
                (norm_number(number), json.dumps(state, ensure_ascii=False), now()))
