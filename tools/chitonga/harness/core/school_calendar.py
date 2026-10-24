"""The school calendar at run time: which term a date is in, its week, holidays and breaks (24 Oct 2026, the Jesuit).

Reads a plug-in's school_calendar.json (education: plugins/education/build_calendar.py, every entry with status and
source) and a teacher's own term dates (core/timetable.py, tt_terms), which win field by field. Generic: nothing here
is education-specific except that a 'term' is the unit; another plug-in may ship a calendar of seasons the same way.

Honesty rules the views keep:
  - a blank official date stays blank ('to_confirm'); it is never filled with a guess. When a term's END is unknown, a
    day after its start counts as in term but carries `term_end_to_confirm` (the UI asks the teacher for the date);
  - a holiday row with skip_lessons false (an unclear Sunday substitute, a revoked holiday) is shown, not skipped;
  - status and source travel with every term field and holiday, so the page can say "official", "reported",
    "computed, check the gazette" or "your date".
"""
import datetime as dt
import json
import os

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def d_(x):
    """'2026-10-24' | date | None -> date | None"""
    if x is None or x == "":
        return None
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x)[:10])


def monday(d):
    return d - dt.timedelta(days=d.weekday())


class SchoolCalendar:
    FIELDS = ("start", "end", "midterm_start", "midterm_end")

    def __init__(self, data):
        self.data = data or {"years": {}}
        self.weekdays = set(self.data.get("weekdays") or [1, 2, 3, 4, 5])      # ISO weekdays with lessons
        self._hol = {}
        for y, v in self.data.get("years", {}).items():
            for h in v.get("holidays", []):
                self._hol.setdefault(h["date"], []).append(h)

    @classmethod
    def load(cls, path):
        try:
            with open(path, encoding="utf-8") as f:
                return cls(json.load(f))
        except (OSError, ValueError):
            return cls(None)

    @classmethod
    def for_plugin(cls, P):
        return cls.load(os.path.join(getattr(P, "dir", "") or "", "school_calendar.json"))

    # ------------------------------------------------------------------------------------------------ terms
    def terms(self, year, overrides=None):
        """[{term, start, end, midterm_start, midterm_end, <field>_status, <field>_source, note}] for the year, the
        teacher's own dates (overrides: {(year, term): {field: 'YYYY-MM-DD' | None}}) winning per field"""
        ov = overrides or {}
        base = {t["term"]: t for t in self.data.get("years", {}).get(str(year), {}).get("terms", [])}
        out = []
        for n in (1, 2, 3):
            t = dict(base.get(n) or {"term": n, "start": None, "end": None, "midterm_start": None,
                                     "midterm_end": None, "start_status": "to_confirm"})
            mine = ov.get((int(year), n)) or {}
            for f in self.FIELDS:
                sk = (f if f in ("start", "end") else "midterm") + "_status"
                if mine.get(f):
                    t[f] = mine[f]
                    t[f + "_by"] = "teacher"
                    t[sk] = "teacher"
                else:
                    t.setdefault(f, None)
                    t[f + "_by"] = "preloaded" if t.get(f) else None
                    if not t.get(sk):
                        t[sk] = "official" if t.get(f) else "to_confirm"
            t["term"] = n
            t["year"] = int(year)
            out.append(t)
        return out

    def term_of(self, d, overrides=None):
        """(term dict, week of term) for a date in term, else (None, None). A term with no known end runs until the
        next term's start (flagged by the caller via `end`)."""
        d = d_(d)
        ts = self.terms(d.year, overrides)
        for i, t in enumerate(ts):
            s, e = d_(t["start"]), d_(t["end"])
            if not s or d < s:
                continue
            if e and d > e:
                continue
            if not e:
                nxt = next((d_(x["start"]) for x in ts[i + 1:] if x["start"]), None)
                if nxt and d >= nxt:
                    continue
            return t, self.week_of(d, t)
        return None, None

    def week_of(self, d, t):
        """teaching week of the term (1 = the week of the opening day); weeks wholly inside a known mid-term break are
        not counted"""
        d, s = d_(d), d_(t["start"])
        w = (monday(d) - monday(s)).days // 7 + 1
        ms, me = d_(t.get("midterm_start")), d_(t.get("midterm_end"))
        if ms and me and d > me:
            broken = sum(1 for k in range((monday(me) - monday(ms)).days // 7 + 1)
                         if monday(ms) + dt.timedelta(days=7 * k) >= ms and
                         monday(ms) + dt.timedelta(days=7 * k + 4) <= me)
            w -= broken
        return max(1, w)

    # ------------------------------------------------------------------------------------------------ days
    def holidays_on(self, d):
        return list(self._hol.get(d_(d).isoformat(), []))

    def day(self, d, overrides=None):
        """everything the calendar knows about one date"""
        d = d_(d)
        t, w = self.term_of(d, overrides)
        hols = self.holidays_on(d)
        skip_h = [h for h in hols if h.get("skip_lessons")]
        info = {"date": d.isoformat(), "weekday": d.isoweekday(), "weekday_name": WEEKDAY_NAMES[d.weekday()],
                "term": t["term"] if t else None, "week": w, "holidays": hols, "flags": []}
        brk = None
        if t:
            ms, me = d_(t.get("midterm_start")), d_(t.get("midterm_end"))
            if ms and me and ms <= d <= me:
                brk = "midterm"
            if not t.get("end"):
                info["flags"].append("term_end_to_confirm")
            if t.get("start_status") == "reported":
                info["flags"].append("term_start_reported")
        else:
            brk = "between_terms" if self.terms(d.year, overrides) and any(
                x["start"] for x in self.terms(d.year, overrides)) else "no_calendar"
        info["break"] = brk
        weekday_ok = not skip_h and d.isoweekday() in self.weekdays
        if brk == "no_calendar":
            # no term dates known for this year at all (2027 until the Ministry publishes): every weekday that is not a
            # holiday is offered, flagged, rather than an empty calendar; the teacher's own dates switch this off
            brk = info["break"] = None
            info["flags"].append("term_dates_to_confirm")
            info["school_day"] = weekday_ok
            return info
        info["school_day"] = bool(t) and not brk and weekday_ok
        return info

    def days(self, start, end, overrides=None):
        start, end = d_(start), d_(end)
        out, d = [], start
        while d <= end:
            out.append(self.day(d, overrides))
            d += dt.timedelta(days=1)
        return out

    def year_view(self, year, overrides=None):
        """terms (effective) + holidays + the status legend, for the timetable sheet and the calendar header"""
        y = self.data.get("years", {}).get(str(year), {})
        return {"year": int(year), "terms": self.terms(year, overrides), "holidays": y.get("holidays", []),
                "checked": y.get("checked"), "status_meaning": self.data.get("status_meaning", {}),
                "authority": self.data.get("authority")}
