"""The teacher app's timetable, calendar, lesson notes and weekly digest endpoints (24 Oct 2026, the Jesuit).

Chris, 24 Oct 2026: "teachers would appreciate if they could enter their Stundenplan and they can click on lessons in a
calendar to keep notes ... the notes could then be analyzed by the AI"; then "smart ... easy ... not overload me with
options ... Microsoft Todo" (DESIGN_DIRECTION.md, last section): Today, a Suggested tray, quick add, "Taught it", no
settings screen, term dates and holidays preloaded.

A MIXIN of api.WebAPI (kept in its own file so the lesson / search code in api.py and this code can change apart).
It uses the WebAPI's memory, plug-in, gates, sync store and clock, and these attributes set in WebAPI.__init__:
self.tt (core/timetable.TimetableStore), self.cal (core/school_calendar.SchoolCalendar), self.planner
(plugins/education/planner.Planner), self._digest_jobs, self.local_now().

Reads are GET endpoints (server.py); writes are sync ops through WebAPI.sync_batch (idempotent per op client id;
periods and notes are also last-writer-wins on their own updated_at). The contract: web/TIMETABLE_API.md.
"""
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, ".."))

from core import gates as G  # noqa: E402


class TimetableMixin:
    @staticmethod
    def _date(x, default=None):
        if x in (None, ""):
            return default
        if not re.fullmatch(r"\d{4}-\d\d-\d\d", str(x)):
            raise ValueError("date must be YYYY-MM-DD")
        return datetime.date.fromisoformat(str(x))

    # ------------------------------------------------------------------------------------------------ reads
    def timetable(self, who, year=None):
        """the editor sheet: periods, classes, the year's terms (preloaded or the teacher's) and holidays"""
        from core.timetable import SUBJECTS, TAGS  # noqa: PLC0415
        n = who["number"]
        y = int(year) if str(year or "").isdigit() else self.local_now().year
        return {"periods": self.tt.periods(n), "classes": [self._class_view(c) for c in self.mem.classes(n)],
                "calendar": self.cal.year_view(y, self.tt.term_overrides(n)), "subjects": list(SUBJECTS),
                "tags": list(TAGS), "weekdays": sorted(self.cal.weekdays)}

    def calendar(self, who, frm=None, to=None):
        today = self.local_now().date()
        a = self._date(frm, today.replace(day=1))
        b = self._date(to, a + datetime.timedelta(days=41))
        if b < a or (b - a).days > 120:
            raise ValueError("range: at most 120 days, from <= to")
        return {"from": a.isoformat(), "to": b.isoformat(),
                "days": self.cal.days(a, b, self.tt.term_overrides(who["number"])),
                "status_meaning": self.cal.data.get("status_meaning", {})}

    def today(self, who, date=None):
        now = self.local_now()
        if date:
            d = self._date(date)
            if d != now.date():
                now = datetime.datetime.combine(d, datetime.time(6, 0))
        return self.planner.today(who["number"], now)

    def week(self, who, start=None):
        return self.planner.week(who["number"], self._date(start), self.local_now())

    def suggested(self, who, date=None):
        now = self.local_now()
        if date:
            now = datetime.datetime.combine(self._date(date), now.time())
        return {"items": self.planner.suggested(who["number"], now)}

    def notes(self, who, frm=None, to=None, class_id=None):
        today = self.local_now().date()
        a = self._date(frm, today - datetime.timedelta(days=30))
        b = self._date(to, today)
        cid = int(class_id) if str(class_id or "").isdigit() else None
        if cid is not None:
            self._own_class(who, cid)
        return {"notes": [self.tt.note_view(r) for r in self.tt.notes(who["number"], cid, a.isoformat(),
                                                                      b.isoformat())]}

    def quickadd(self, who, body):
        """parse one line, save nothing: the UI shows the parse, the teacher corrects it, then sends op 'note'.
        body.model true: the standard model may fill what the rules left open, only if a model server already runs
        (a web request never starts one)."""
        model = None
        if body.get("model") and os.environ.get("ZAMAI_WEB_MODEL", "1") == "1":
            from plugins.education import notes_digest as ND  # noqa: PLC0415
            g = ND.Gemma(timeout=30)
            if g.up():
                model = ND.quick_model(g)
        r = self.planner.parse_quick(who["number"], str(body.get("text") or ""), self.local_now(), model=model)
        if r.get("safeguarding"):
            self.mem.event(who["number"], "safeguarding_referral", note="web quick add")
            ref = G.referral("web")
            return {"safeguarding": {"reply": ref.get("reply") or ref.get("en_meaning")}}
        return {"parse": r}

    def digest(self, who, class_id=None, week=None):
        n = who["number"]
        wk = self.planner.week_start(self._date(week)).isoformat() if week else None
        out = []
        for c in self.mem.classes(n):
            if class_id not in (None, "") and str(c["id"]) != str(class_id):
                continue
            d = self.tt.digest(n, c["id"], wk)
            if d:
                out.append(self._digest_view(n, d))
        job = dict(self._digest_jobs.get(n) or {})
        if job.get("proc") is not None:
            job["running"] = job["proc"].poll() is None
            job["exit"] = job["proc"].poll()
        job.pop("proc", None)
        return {"digests": out, "job": job or None}

    def _digest_view(self, n, d):
        """the teacher's view: no raw model text; each line's notes resolved (date, slot, text) so it links back"""
        v = {k: d.get(k) for k in ("class", "week", "week_end", "n_notes", "built_at", "sections", "matches",
                                   "not_summarised", "label", "ai", "model")}
        v["dropped"] = len(d.get("dropped") or [])
        ids = {i for c in (d.get("sections") or {}).values() for it in c for i in it["notes"]}
        ids |= {x["id"] for x in d.get("not_summarised") or []}
        idx = {}
        for i in sorted(ids):
            r = self.tt.note(i)
            if r and r["teacher"] == n and not r["deleted"]:
                idx[i] = {"date": r.get("slot_date") or r["created_at"][:10],
                          "slot_id": f"{r['slot_date']}:{r['period_id']}" if r.get("slot_date") and r.get(
                              "period_id") else None,
                          "text": (r.get("text") or "")[:160], "tags": json.loads(r.get("tags") or "[]")}
        v["notes_index"] = idx
        for m in v.get("matches") or []:
            mt = m.get("match") or {}
            if mt.get("kind") == "strategy":
                mt["lesson_title"] = (self.P.items.get(mt.get("lesson")) or {}).get("title")
        return v

    def digest_run(self, who, body):
        """start the digest job for this teacher in its own process (it takes the Beelink heavy-job lock and may wait
        for it); poll GET /api/digest"""
        import subprocess  # noqa: PLC0415
        n = who["number"]
        job = self._digest_jobs.get(n)
        if job and job.get("proc") is not None and job["proc"].poll() is None:
            return {"started": False, "running": True, "since": job["started"]}
        args = [sys.executable, "-X", "utf8", os.path.join(HARNESS, "plugins", "education", "notes_digest.py"),
                "--db", self.mem.path, "--teacher", n]
        if str(body.get("class_id") or "").isdigit():
            self._own_class(who, body["class_id"])
            args += ["--class", str(body["class_id"])]
        if body.get("week"):
            args += ["--week", self._date(body["week"]).isoformat()]
        if body.get("no_model"):
            args.append("--no-model")
        os.makedirs(self.logs, exist_ok=True)
        log = open(os.path.join(self.logs, "digest_job.log"), "a", encoding="utf-8")
        p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._digest_jobs[n] = {"proc": p, "started": self.clock().isoformat(timespec="seconds"), "pid": p.pid}
        return {"started": True, "pid": p.pid}

    # ------------------------------------------------------------------------------------------------ sync ops
    def op_period(self, who, op):
        p = dict(op.get("period") or {})
        p.setdefault("updated_at", op.get("at"))
        return {"period": self.tt.upsert_period(who["number"], p)}

    def op_period_delete(self, who, op):
        return self.tt.delete_period(who["number"], str(op.get("id") or ""), op.get("updated_at") or op.get("at"))

    def op_copy_day(self, who, op):
        return {"periods": self.tt.copy_day(who["number"], int(op.get("from_day") or 0), int(op.get("to_day") or 0),
                                            list(op.get("new_ids") or []), bool(op.get("replace", True)),
                                            op.get("updated_at") or op.get("at"))}

    def op_term(self, who, op):
        if op.get("reset"):
            return self.tt.reset_term(who["number"], op.get("year"), op.get("term"))
        return {"term": self.tt.set_term(who["number"], op.get("year"), op.get("term"),
                                         op.get("updated_at") or op.get("at"),
                                         **{k: op.get(k) for k in ("start", "end", "midterm_start", "midterm_end")})}

    def op_note(self, who, op):
        """a note upsert (fields absent keep their value). Safeguarding: the text is NOT stored (withheld), the
        referral event is recorded, the referral returned. A learner's name is kept as written: the teacher's own
        data, never in the cross-teacher backlog (notes_digest.aggregate strips names, skips learner notes)."""
        n = dict(op.get("note") or {})
        n.setdefault("updated_at", op.get("at"))
        out = {}
        if n.get("text") and G.safeguarding(str(n["text"])):
            self.mem.event(who["number"], "safeguarding_referral", note="web lesson note")
            ref = G.referral("web")
            n["text"], n["withheld"] = None, "safeguarding"
            out["safeguarding"] = {"reply": ref.get("reply") or ref.get("en_meaning")}
        for k in ("item", "planned_item"):
            if n.get(k) and n[k] not in self.P.items:
                raise ValueError("no such item")
        out["note"] = self.tt.upsert_note(who["number"], n)
        return out

    def op_note_delete(self, who, op):
        return self.tt.delete_note(who["number"], str(op.get("id") or ""), op.get("updated_at") or op.get("at"))

    def op_taught(self, who, op):
        """'Taught it': one tap; the class's resume point moves on, the slot's note records the lesson"""
        return self.planner.taught(who["number"], op.get("class_id"), op.get("item"), op.get("slot_date"),
                                   op.get("period_id"), op.get("note_id"), op.get("updated_at") or op.get("at"))
