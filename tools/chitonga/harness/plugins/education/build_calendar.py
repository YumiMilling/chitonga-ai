"""The Zambian school calendar for the timetable (24 Oct 2026, the Jesuit). Writes the education plug-in's
school_calendar.json: term dates and public holidays per year, every entry with its basis, its status and where it was
checked, so the calendar can show holidays and breaks and the slot generator can skip them.

Chris, 24 Oct 2026: "Term dates are standardized in Zambia, you can preload them, including public holidays." So the
teacher never types term dates; a teacher's own dates still override (core/timetable.py, table tt_terms).

STATUS of an entry (shown to the teacher, never hidden):
  official      an official text says it (the Act, a statutory instrument, a Ministry or Cabinet Office notice)
  reported      a government body's date as reported by the press (a Ministry spokesperson, RTSA), no circular seen
  computed      derived by the Act's own rules (fixed dates, first Monday, the Sunday rule, Easter), not yet gazetted
  to_confirm    not found in any official source: left BLANK (never guessed); the teacher can fill it in

WHAT WAS CHECKED (24 Oct 2026; the rules here encode only what these sources say):
  - Public Holidays Act, Cap. 272 (as amended by Act No. 23 of 1987), Parliament copy: s.2(1) the Schedule's days are
    public holidays; s.2(2) "Whenever any of the days mentioned in the Schedule falls on a Sunday, the following Monday
    shall be observed as a public holiday"; s.3 the Minister may declare more days by statutory notice. Schedule: New
    Year's Day, Youth Day (12 March), Good Friday, Holy Saturday, Labour Day (1 May), African Freedom Day (25 May), Heroes
    Day (first Monday in July), Unity Day (Tuesday following Heroes Day), Farmers Day (first Monday in August),
    Independence Day (24 October), Christmas Day.
  - Days added under s.3 (not in the Schedule): International Women's Day (8 March), Kenneth Kaunda Day (28 April),
    Easter Monday, and the National Day of Prayer (named in the National Assembly committee report on the Public
    Holidays (Amendment) Bill, N.A.B. No. 76 of 2026). That report says the Sunday rule's application to s.3 days is
    UNCLEAR, and the Bill (to REMOVE the Sunday rule) was reported in the 5th session of the 13th Assembly; whether it
    passed is not known here. So: the Sunday rule is applied to Schedule days as the Act says (status computed), and
    a Sunday-substitute Monday for an s.3 day is 'to_confirm'.
  - National Day of Prayer: SI No. 78 of 2015 (18 October) REVOKED in Oct 2026; SI No. 73 of 2026 declares 29 December
    the "National Day of Prayer and Christian Heritage" from 2026; "18th October, 2026 will not be observed as a public
    holiday" (Secretary to the Cabinet, Zambia Daily Mail 10 Oct 2026 p1; church mother bodies, 9 Oct 2026).
  - General election day, Thursday 13 August 2026: public holiday under Article 56(2) of the Constitution, Cabinet
    Office Circular No. 12 of 2026 (cabinet.gov.zm). No inauguration holiday was found.
  - Term 1 2026 opened Monday 12 January 2026 (Ministry of Education notice, PS Education Services, edu.gov.zm/?p=9803;
    "zero week" 5-9 January for teachers, ZANIS via Lusaka Times 2 Jan 2026). Term 2 re-opened Monday 11 May 2026 (RTSA
    statement, Zambia Monitor 9 May 2026). Term 3 re-opened Monday 7 September 2026 (Ministry Head of Communication
    Kunda Mando, Lusaka Times 7 Sep 2026). No closing date and no mid-term break was found in an official source: blank.
    Private-school calendars (Scribd) and foreign "Zambia calendar" sites conflict and were not used.
  - 2027: no Ministry calendar found; holidays computed by the rules above, terms blank.

Rebuild: python -X utf8 tools/chitonga/harness/plugins/education/build_calendar.py   (add a year: YEARS below)
"""
import datetime as dt
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
OUT = os.path.join(REPO, "internal", "chitonga_train", "assistant_library", "plugin_education", "school_calendar.json")
CHECKED = "2026-10-24"
YEARS = (2026, 2027)

SRC = {
    "act": "https://www.parliament.gov.zm/sites/default/files/documents/acts/Public%20Holidays%20Act.pdf",
    "bill76": ("https://www.parliament.gov.zm/sites/default/files/documents/committee_reports/REPORT%20%20OF%20THE%20"
               "COMMITTEE%20ON%20GOVERNANCE.%20HOUSING%20AND%20CHIEFS%27%20AFFAIRS%20ON%20THE%20PUBLIC%20HOLIDAYS%20%28"
               "AMENDMENT%29%20BILL%2C%20N.A.B.%20NO.%2076%20OF%202026%20FOR%20THE%205TH%20SESSION%20OF%20THE%2013TH%20"
               "%20NATIONAL%20ASSEMBLY.pdf"),
    "prayer": "Zambia Daily Mail e-paper 10 Oct 2026 p1 'Prayer Day SI revoked' (SI 78/2015 revoked, SI 73/2026); "
              "9 Oct 2026 (church mother bodies)",
    "election": "https://www.cabinet.gov.zm/?p=11553",
    "kk_day": "Zambia Daily Mail e-paper 28 Apr 2026 ('As we mark Kaunda Day today'); added 2021 (statutory notice)",
    "t1": "https://www.edu.gov.zm/?p=9803",
    "t1_zero": "https://www.lusakatimes.com/2026/01/02/ministry-of-education-set-for-reopening-of-schools/",
    "t2": "https://www.zambiamonitor.com/rtsa-calls-for-caution-as-schools-reopen-for-term-two/",
    "t3": "https://www.lusakatimes.com/2026/09/07/schools-ready-for-new-term-mando/",
}


def easter(y):
    """Gregorian Easter Sunday (Anonymous / Meeus-Jones-Butcher algorithm)"""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(y, month, day)


def first_monday(y, month):
    d = dt.date(y, month, 1)
    return d + dt.timedelta(days=(7 - d.weekday()) % 7)


def holidays(y):
    """[(date, name, basis, kind)] kind: schedule (Act s.2) | s3 (added by statutory notice) | one_off"""
    e = easter(y)
    hm = first_monday(y, 7)
    out = [
        (dt.date(y, 1, 1), "New Year's Day", "Act Cap. 272, Schedule (a)", "schedule"),
        (dt.date(y, 3, 8), "International Women's Day", "declared under s.3 (statutory notice)", "s3"),
        (dt.date(y, 3, 12), "Youth Day", "Act Cap. 272, Schedule (b)", "schedule"),
        (e - dt.timedelta(days=2), "Good Friday", "Act Cap. 272, Schedule (c)", "schedule"),
        (e - dt.timedelta(days=1), "Holy Saturday", "Act Cap. 272, Schedule (d)", "schedule"),
        (e + dt.timedelta(days=1), "Easter Monday", "declared under s.3 (statutory notice)", "s3"),
        (dt.date(y, 4, 28), "Kenneth Kaunda Day", "declared under s.3 (statutory notice, 2021)", "s3"),
        (dt.date(y, 5, 1), "Labour Day", "Act Cap. 272, Schedule (e)", "schedule"),
        (dt.date(y, 5, 25), "African Freedom Day", "Act Cap. 272, Schedule (f)", "schedule"),
        (hm, "Heroes Day", "Act Cap. 272, Schedule (g): first Monday in July", "schedule"),
        (hm + dt.timedelta(days=1), "Unity Day", "Act Cap. 272, Schedule (h): Tuesday following Heroes Day", "schedule"),
        (first_monday(y, 8), "Farmers Day", "Act Cap. 272, Schedule (i): first Monday in August", "schedule"),
        (dt.date(y, 10, 24), "Independence Day", "Act Cap. 272, Schedule (j)", "schedule"),
        (dt.date(y, 12, 25), "Christmas Day", "Act Cap. 272, Schedule (k)", "schedule"),
    ]
    if y >= 2026:   # SI 73 of 2026 (from 2026); SI 78 of 2015 (18 October) revoked
        out.append((dt.date(y, 12, 29), "National Day of Prayer and Christian Heritage", "SI No. 73 of 2026 (s.3)",
                    "s3"))
    if y == 2026:
        out.append((dt.date(2026, 8, 13), "General election day", "Constitution Art. 56(2); Cabinet Office Circular "
                                                                    "No. 12 of 2026", "one_off"))
    return sorted(out)


def holiday_rows(y):
    """the year's rows with status and source; Sunday substitutes as their own rows"""
    rows = []
    official_2026 = {"General election day": ("official", SRC["election"]),
                     "National Day of Prayer and Christian Heritage": ("official", SRC["prayer"]),
                     "Kenneth Kaunda Day": ("reported", SRC["kk_day"])}
    for d, name, basis, kind in holidays(y):
        if y == 2026 and name in official_2026:
            status, src = official_2026[name]
        elif kind == "schedule":
            status, src = "computed", SRC["act"]
        else:
            status, src = "computed", SRC["bill76"]
        rows.append({"date": d.isoformat(), "name": name, "basis": basis, "status": status, "source": src,
                     "checked": CHECKED, "kind": kind})
        if d.weekday() == 6 and name != "Holy Saturday":
            sub = d + dt.timedelta(days=1)
            if kind == "schedule":
                rows.append({"date": sub.isoformat(), "name": f"{name} (observed: Sunday rule)",
                             "basis": "Act Cap. 272 s.2(2): a Schedule day on a Sunday is observed the following "
                                      "Monday", "status": "computed", "source": SRC["act"], "checked": CHECKED,
                             "kind": "substitute", "for": d.isoformat()})
            else:
                rows.append({"date": sub.isoformat(), "name": f"{name} (Monday off?)",
                             "basis": "s.3 day on a Sunday: whether the Sunday rule applies is unclear (committee "
                                      "report on N.A.B. No. 76 of 2026); check the government's notice",
                             "status": "to_confirm", "source": SRC["bill76"], "checked": CHECKED,
                             "kind": "substitute", "for": d.isoformat(), "skip_lessons": False})
    if y == 2026:
        rows.append({"date": "2026-10-18", "name": "National Day of Prayer (NOT a holiday in 2026)",
                     "basis": "SI No. 78 of 2015 revoked; merged into 29 December (SI No. 73 of 2026)",
                     "status": "official", "source": SRC["prayer"], "checked": CHECKED, "kind": "not_holiday",
                     "skip_lessons": False})
    for r in rows:
        r.setdefault("skip_lessons", r["kind"] != "not_holiday")
    return sorted(rows, key=lambda r: r["date"])


def terms(y):
    blank = {"end": None, "end_status": "to_confirm", "midterm_start": None, "midterm_end": None,
             "midterm_status": "to_confirm"}
    if y == 2026:
        return [
            dict(term=1, start="2026-01-12", start_status="official", start_source=SRC["t1"],
                 note="Teachers' zero week 5-9 Jan 2026 (" + SRC["t1_zero"] + ")", **blank),
            dict(term=2, start="2026-05-11", start_status="reported", start_source=SRC["t2"],
                 note="Date from the Road Transport and Safety Agency's statement; Ministry circular not seen", **blank),
            dict(term=3, start="2026-09-07", start_status="reported", start_source=SRC["t3"],
                 note="Ministry Head of Communication, as reported", **blank),
        ]
    return [dict(term=t, start=None, start_status="to_confirm", start_source=None,
                 note="No Ministry calendar found on " + CHECKED, **blank) for t in (1, 2, 3)]


def build():
    cal = {"version": CHECKED, "country": "ZM",
           "authority": "The Minister of Education sets the national school calendar under the Education Act; all "
                        "schools except international / authorised ones follow it (edu.gov.zm/?p=9803).",
           "status_meaning": {"official": "an official text says it", "reported": "a government body's date as "
                              "reported by the press; no circular seen", "computed": "derived by the Act's rules, "
                              "check the gazette", "to_confirm": "not found officially: blank, the teacher may fill "
                              "it in"},
           "weekdays": [1, 2, 3, 4, 5],
           "years": {}}
    for y in YEARS:
        cal["years"][str(y)] = {"terms": terms(y), "holidays": holiday_rows(y), "checked": CHECKED}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=1)
    return cal


if __name__ == "__main__":
    c = build()
    for y, v in c["years"].items():
        print(y, "terms:", [(t["term"], t["start"], t["start_status"]) for t in v["terms"]])
        for h in v["holidays"]:
            print("  ", h["date"], dt.date.fromisoformat(h["date"]).strftime("%a"), h["status"].ljust(10), h["name"])
    print("->", OUT)
