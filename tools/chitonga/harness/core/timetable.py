"""Timetable, term dates, lesson notes and weekly digests, in the teacher memory file (24 Oct 2026, the Jesuit).

Chris, 24 Oct 2026: "teachers would appreciate if they could enter their Stundenplan and they can click on lessons in a
calendar to keep notes for example, the notes could then be analyzed by the AI (no personal notes, just about what
worked, didn't work, ideas, kids that need special attention etc)".

WHY A SEPARATE STORE ON THE SAME FILE. The tables live in the SAME SQLite file as core/memory.py (one teacher record,
one backup, one future move to Supabase), but behind their own class, so MemoryStore's interface (which the engine and
the WhatsApp adapter use) does not grow. TimetableStore borrows the SQLiteMemory's connection, lock and transaction().

OFFLINE-FIRST WRITES. The phone makes the ids (periods and notes: client ids, 8-64 chars [A-Za-z0-9_-]) and stamps
every write with its own `updated_at` (ISO, UTC). An upsert is LAST WRITER WINS on that stamp: an older write than the
stored one is ignored and the stored row returned with `stale: true`; the same write twice is a no-op. So a retried
queue, two phones, or a sync after days offline converge without duplicates. Deletes are tombstones (deleted=1, with
their stamp), so a late upsert from an older edit cannot resurrect a deleted row.

LEARNERS IN NOTES (Chris's rule for notes, 24 Oct): a note may name a learner who needs attention. It is stored as the
teacher wrote it, ONLY in that teacher's rows; nothing here copies it anywhere. The cross-teacher aggregate
(plugins/education/notes_digest.py) strips names and never reads 'attention' notes. Two guards still apply to the text,
as everywhere in the harness: a safeguarding disclosure is never stored as text (the caller records the referral event
and passes text=None, withheld='safeguarding'); marks are not refused here (a teacher's own note), but never leave the
teacher's data.
"""
import datetime as dt
import json
import re

from .memory import norm_number

TAGS = ("worked", "didnt", "idea", "attention", "repeat")
SUBJECTS = ("english", "chitonga", "maths", "science", "cts", "other")
CID = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")
TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
DATE = re.compile(r"^\d{4}-\d\d-\d\d$")

SCHEMA = """
create table if not exists tt_periods (id text primary key, teacher text not null, day integer not null,
  start text, end text, class_id integer, subject text, label text, room text, deleted integer default 0,
  updated_at text, server_at text);
create index if not exists tt_periods_teacher on tt_periods (teacher);
create table if not exists tt_terms (teacher text not null, year integer not null, term integer not null, start text,
  end text, midterm_start text, midterm_end text, updated_at text, primary key (teacher, year, term));
create table if not exists tt_notes (id text primary key, teacher text not null, class_id integer, slot_date text,
  period_id text, subject text, tags text, attention text, text text, item text, taught integer default 0,
  taught_text text, planned_item text, withheld text, created_at text, updated_at text, server_at text,
  deleted integer default 0, text_en text, text_lang text, text_en_by text, text_en_at text);
create index if not exists tt_notes_teacher on tt_notes (teacher, slot_date);
create table if not exists tt_digests (teacher text not null, class_id integer not null, week text not null,
  json text, built_at text, primary key (teacher, class_id, week));
"""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _stamp(x):
    """a client stamp, normalised so that string order is time order ('...Z' and '+00:00' both to '+00:00')"""
    s = str(x or "").strip()
    if not s:
        return utc_now()
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError("updated_at is not an ISO time") from e
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds")


class TimetableStore:
    def __init__(self, mem):
        self.mem = mem
        self.db = mem.db
        self.lock = mem.lock
        with self.lock:
            self.db.executescript(SCHEMA)

    def transaction(self):
        return self.mem.transaction()

    def _all(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def _one(self, sql, args=()):
        with self.lock:
            r = self.db.execute(sql, args).fetchone()
        return dict(r) if r else None

    def _x(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args)

    def _own_class(self, teacher, class_id):
        if class_id in (None, ""):
            return None
        c = self.mem.get_class(int(class_id)) if str(class_id).isdigit() else None
        if not c or c["teacher"] != norm_number(teacher):
            raise ValueError("not your class")
        return c

    # ------------------------------------------------------------------------------------------------ periods
    @staticmethod
    def _period_view(r):
        return {k: r[k] for k in ("id", "day", "start", "end", "class_id", "subject", "label", "room", "updated_at")}

    def periods(self, teacher, include_deleted=False):
        q = "select * from tt_periods where teacher=?" + ("" if include_deleted else " and deleted=0")
        rows = self._all(q + " order by day, coalesce(start, '99:99'), id", (norm_number(teacher),))
        return [self._period_view(r) | ({"deleted": True} if r["deleted"] else {}) for r in rows]

    def period(self, pid):
        return self._one("select * from tt_periods where id=?", (pid,))

    def upsert_period(self, teacher, p):
        """p: {id, day 1-7, start 'HH:MM'|None, end, class_id, subject, label, room, updated_at}"""
        n = norm_number(teacher)
        pid = str(p.get("id") or "")
        if not CID.match(pid):
            raise ValueError("bad period id")
        day = int(p.get("day") or 0)
        if not 1 <= day <= 7:
            raise ValueError("day must be 1 (Monday) to 7")
        start, end = p.get("start") or None, p.get("end") or None
        for t in (start, end):
            if t is not None and not TIME.match(str(t)):
                raise ValueError("time must be HH:MM")
        if start and end and end <= start:
            raise ValueError("end before start")
        subject = p.get("subject") or None
        if subject is not None and subject not in SUBJECTS:
            raise ValueError("subject")
        c = self._own_class(teacher, p.get("class_id"))
        label = re.sub(r"\s+", " ", str(p.get("label") or "")).strip()[:40] or None
        if subject == "other" and not label:
            raise ValueError("an 'other' period needs a label")
        room = re.sub(r"\s+", " ", str(p.get("room") or "")).strip()[:30] or None
        st = _stamp(p.get("updated_at"))
        cur = self.period(pid)
        if cur and cur["teacher"] != n:
            raise ValueError("not your period")
        if cur and (cur["updated_at"] or "") > st:
            return dict(self._period_view(cur), stale=True)
        if cur and cur["deleted"] and (cur["updated_at"] or "") >= st:
            return {"id": pid, "deleted": True, "stale": True}
        self._x("insert or replace into tt_periods (id, teacher, day, start, end, class_id, subject, label, room, "
                "deleted, updated_at, server_at) values (?,?,?,?,?,?,?,?,?,0,?,?)",
                (pid, n, day, start, end, c["id"] if c else None, subject, label, room, st, utc_now()))
        return self._period_view(self.period(pid))

    def delete_period(self, teacher, pid, updated_at=None):
        cur = self.period(pid)
        if not cur or cur["teacher"] != norm_number(teacher):
            raise ValueError("no such period")
        st = _stamp(updated_at)
        if (cur["updated_at"] or "") > st:
            return {"id": pid, "stale": True}
        self._x("update tt_periods set deleted=1, updated_at=?, server_at=? where id=?", (st, utc_now(), pid))
        return {"id": pid, "deleted": True}

    def copy_day(self, teacher, from_day, to_day, new_ids, replace=True, updated_at=None):
        """copy one weekday's periods to another; new_ids: the client's ids for the copies, in the source's order (so a
        retried copy makes the same rows, not new ones). replace: tombstone the target day's periods first."""
        src = [p for p in self.periods(teacher) if p["day"] == int(from_day)]
        if len(new_ids or []) < len(src):
            raise ValueError(f"need {len(src)} new ids")
        out = []
        if replace:
            for p in self.periods(teacher):
                if p["day"] == int(to_day) and p["id"] not in new_ids:
                    self.delete_period(teacher, p["id"], updated_at)
        for p, nid in zip(src, new_ids):
            out.append(self.upsert_period(teacher, dict(p, id=nid, day=int(to_day), updated_at=updated_at)))
        return out

    # ------------------------------------------------------------------------------------------------ term dates
    def term_overrides(self, teacher):
        rows = self._all("select * from tt_terms where teacher=?", (norm_number(teacher),))
        return {(r["year"], r["term"]): {k: r[k] for k in ("start", "end", "midterm_start", "midterm_end")}
                for r in rows}

    def set_term(self, teacher, year, term, updated_at=None, **fields):
        """the teacher's own date(s) for one term; a None / '' field falls back to the preloaded calendar"""
        if int(term) not in (1, 2, 3) or not 2020 <= int(year) <= 2100:
            raise ValueError("year / term")
        clean = {}
        for k in ("start", "end", "midterm_start", "midterm_end"):
            v = fields.get(k)
            if v not in (None, "") and not DATE.match(str(v)):
                raise ValueError(f"{k} must be YYYY-MM-DD")
            clean[k] = v or None
        if clean["start"] and clean["end"] and clean["end"] < clean["start"]:
            raise ValueError("term ends before it starts")
        st = _stamp(updated_at)
        cur = self._one("select * from tt_terms where teacher=? and year=? and term=?",
                        (norm_number(teacher), int(year), int(term)))
        if cur and (cur["updated_at"] or "") > st:
            return dict(cur, stale=True)
        self._x("insert or replace into tt_terms values (?,?,?,?,?,?,?,?)",
                (norm_number(teacher), int(year), int(term), clean["start"], clean["end"], clean["midterm_start"],
                 clean["midterm_end"], st))
        return self._one("select * from tt_terms where teacher=? and year=? and term=?",
                         (norm_number(teacher), int(year), int(term)))

    def reset_term(self, teacher, year, term):
        self._x("delete from tt_terms where teacher=? and year=? and term=?", (norm_number(teacher), int(year), int(term)))
        return {"year": int(year), "term": int(term), "reset": True}

    # ------------------------------------------------------------------------------------------------ notes
    @staticmethod
    def note_view(r, private=True):
        v = {k: r.get(k) for k in ("id", "class_id", "slot_date", "period_id", "subject", "attention", "item",
                                   "taught_text", "planned_item", "withheld", "created_at", "updated_at")}
        v["tags"] = json.loads(r.get("tags") or "[]")
        v["taught"] = bool(r.get("taught"))
        v["text"] = r.get("text") if private else None
        if r.get("text_en"):
            v["text_en"] = r["text_en"]
            v["text_en_by"] = r.get("text_en_by")
        if r.get("deleted"):
            v["deleted"] = True
        return v

    def note(self, nid):
        return self._one("select * from tt_notes where id=?", (nid,))

    def upsert_note(self, teacher, n):
        """n: {id, class_id, slot_date, period_id, subject, tags[], attention 'class'|'learner'|None, text, item,
        taught (bool), taught_text, planned_item, withheld, updated_at}. Fields absent from n keep their stored value
        (a quick tag after "Taught it" does not wipe the lesson)."""
        tn = norm_number(teacher)
        nid = str(n.get("id") or "")
        if not CID.match(nid):
            raise ValueError("bad note id")
        cur = self.note(nid)
        if cur and cur["teacher"] != tn:
            raise ValueError("not your note")
        st = _stamp(n.get("updated_at"))
        if cur and (cur["updated_at"] or "") > st:
            return dict(self.note_view(cur), stale=True)
        if cur and cur["deleted"] and (cur["updated_at"] or "") >= st:
            return {"id": nid, "deleted": True, "stale": True}
        row = dict(cur or {"id": nid, "teacher": tn, "tags": "[]", "taught": 0, "created_at": st})
        if "class_id" in n:
            c = self._own_class(teacher, n.get("class_id"))
            row["class_id"] = c["id"] if c else None
        if "slot_date" in n:
            sd = n.get("slot_date") or None
            if sd is not None and not DATE.match(str(sd)):
                raise ValueError("slot_date must be YYYY-MM-DD")
            row["slot_date"] = sd
        if "period_id" in n:
            pid = n.get("period_id") or None
            if pid is not None:
                p = self.period(pid)
                if not p or p["teacher"] != tn:
                    raise ValueError("no such period")
            row["period_id"] = pid
        if "subject" in n:
            if n.get("subject") not in (None, "") + SUBJECTS:
                raise ValueError("subject")
            row["subject"] = n.get("subject") or None
        if "tags" in n:
            tags = [t for t in dict.fromkeys(n.get("tags") or [])]
            bad = [t for t in tags if t not in TAGS]
            if bad:
                raise ValueError(f"unknown tag {bad[0]}")
            row["tags"] = json.dumps(tags)
        if "attention" in n:
            if n.get("attention") not in (None, "", "class", "learner"):
                raise ValueError("attention is 'class' or 'learner'")
            row["attention"] = n.get("attention") or None
        if "text" in n:
            t = n.get("text")
            new = str(t).strip()[:1000] if t else None
            if new != row.get("text"):
                row["text_en"] = row["text_lang"] = row["text_en_by"] = row["text_en_at"] = None   # re-translate
            row["text"] = new
        for k, lim in (("item", 120), ("taught_text", 200), ("planned_item", 120), ("withheld", 40)):
            if k in n:
                row[k] = (str(n[k]).strip()[:lim] or None) if n.get(k) else None
        if "taught" in n:
            row["taught"] = 1 if n.get("taught") else 0
        row.update(updated_at=st, server_at=utc_now(), deleted=0)
        cols = ("id", "teacher", "class_id", "slot_date", "period_id", "subject", "tags", "attention", "text", "item",
                "taught", "taught_text", "planned_item", "withheld", "created_at", "updated_at", "server_at", "deleted",
                "text_en", "text_lang", "text_en_by", "text_en_at")
        self._x(f"insert or replace into tt_notes ({', '.join(cols)}) values ({', '.join('?' * len(cols))})",
                tuple(row.get(c) for c in cols))
        return self.note_view(self.note(nid))

    def delete_note(self, teacher, nid, updated_at=None):
        cur = self.note(nid)
        if not cur or cur["teacher"] != norm_number(teacher):
            raise ValueError("no such note")
        st = _stamp(updated_at)
        if (cur["updated_at"] or "") > st:
            return {"id": nid, "stale": True}
        self._x("update tt_notes set deleted=1, text=null, text_en=null, updated_at=?, server_at=? where id=?",
                (st, utc_now(), nid))
        return {"id": nid, "deleted": True}

    def notes(self, teacher, class_id=None, since=None, until=None, include_deleted=False):
        """a teacher's notes; since / until on slot_date (a note without a slot date counts by created_at's day)"""
        q = "select * from tt_notes where teacher=?"
        a = [norm_number(teacher)]
        if not include_deleted:
            q += " and deleted=0"
        if class_id is not None:
            q += " and class_id=?"
            a.append(int(class_id))
        day = "coalesce(slot_date, substr(created_at, 1, 10))"
        if since:
            q += f" and {day} >= ?"
            a.append(str(since))
        if until:
            q += f" and {day} <= ?"
            a.append(str(until))
        return self._all(q + f" order by {day}, created_at, id", tuple(a))

    def all_notes(self, since=None, until=None):
        """every teacher's notes in a window (the nightly job and the aggregate; never served to a teacher)"""
        day = "coalesce(slot_date, substr(created_at, 1, 10))"
        q, a = "select * from tt_notes where deleted=0", []
        if since:
            q += f" and {day} >= ?"
            a.append(str(since))
        if until:
            q += f" and {day} <= ?"
            a.append(str(until))
        return self._all(q + f" order by teacher, {day}, id", tuple(a))

    def set_translation(self, nid, text_en, lang, by):
        self._x("update tt_notes set text_en=?, text_lang=?, text_en_by=?, text_en_at=? where id=?",
                (text_en, lang, by, utc_now(), nid))

    def notes_for_slot(self, teacher, slot_date, period_id):
        return self._all("select * from tt_notes where teacher=? and slot_date=? and period_id=? and deleted=0 "
                         "order by updated_at", (norm_number(teacher), slot_date, period_id))

    # ------------------------------------------------------------------------------------------------ digests
    def put_digest(self, teacher, class_id, week, digest):
        self._x("insert or replace into tt_digests values (?,?,?,?,?)",
                (norm_number(teacher), int(class_id), str(week), json.dumps(digest, ensure_ascii=False), utc_now()))

    def digest(self, teacher, class_id, week=None):
        q = "select * from tt_digests where teacher=? and class_id=?"
        a = [norm_number(teacher), int(class_id)]
        if week:
            q += " and week=?"
            a.append(str(week))
        r = self._one(q + " order by week desc limit 1", tuple(a))
        return json.loads(r["json"]) if r else None

    def digests(self, teacher, week=None):
        q = "select * from tt_digests where teacher=?"
        a = [norm_number(teacher)]
        if week:
            q += " and week=?"
            a.append(str(week))
        return [json.loads(r["json"]) for r in self._all(q + " order by week desc, class_id", tuple(a))]
