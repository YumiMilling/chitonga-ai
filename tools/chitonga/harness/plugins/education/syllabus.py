"""The syllabus at run time: browse, the week view, coverage, search, and the WhatsApp answer (24 Oct 2026, the Jesuit).
Reads the plug-in's 'syllabus' items (build_syllabus.py); writes nothing. Used by hooks.py (route:syllabus) and by
web/api.py (the /api/syllabus endpoints and the grade bundle), so the page and WhatsApp say the same thing.

Three rules every view keeps:
  - Pacing (term, week, weeks needed, weight) is FUNDA'S ESTIMATE, labelled so in every reply that shows it. The
    Ministry's book gives no weeks for these topics.
  - A syllabus line is the CDC's text (or, for Chitonga, the printed module's Tonga, auto-extracted). We never write
    Tonga here; the Chitonga lines are shown as the module prints them, marked as auto-extracted.
  - 'Our material' lists what covers a competence and how it was found (by code, or a title lead); an unchecked health
    card or a held card is shown as not served, never as cover.

"This term" is read from the month (Zambian public school terms: January-April Term 1, May-August Term 2, September-
December Term 3; a school's own dates may differ, and the reply says which term it took).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, "..", ".."))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from core.intake import grades_in  # noqa: E402
from core.retrieval import Index  # noqa: E402

PACING_LABEL = "Funda's pacing estimate, not the Ministry's (the syllabus gives no weeks)"
AREA_WORDS = [  # request words -> learning area (English mine; Tonga subject names the teachers use, from SUBJECT_WORDS)
    ("maths_science", r"maths?|mathematics|numeracy|science|integrated science|sayaasi"),
    ("english", r"english|literacy|phonics|cikuwa|cingisi"),
    ("chitonga", r"chitonga|citonga|zambian language|tonga lesson|tonga reading"),
    ("cts", r"cts|creative|technology studies|expressive arts|art|music|physical education|home economics|"
            r"needlework|crafts?"),
]
AREA_RX = [(a, re.compile(r"(?i)(?<![a-z])(?:" + rx + r")(?![a-z])")) for a, rx in AREA_WORDS]
TERM_RX = re.compile(r"(?i)\bterm\s*([123])\b|\b(first|second|third)\s+term\b|\b(this|next|current)\s+term\b")
WEEK_RX = re.compile(r"(?i)\bweek\s*(\d{1,2})\b")
META = set("""syllabus syllabi curriculum competence competences competency competencies specific learning activity
activities expected standard standards scheme work what should shall must will does do learn learners learner pupils
class teach cover covered study grade grades term week this next current first second third planned plan around about
give show list tell need want the for in of on a an and to my our we they is are please which subject subjects topic
topics sub say says tell mean means need help me us there any have has get find look""".split())


PROV_G13 = ("CDC 2024 Lower Primary Syllabi (English, Mathematics and Science, CTS) as transcribed by Funda and checked "
            "against the book; Chitonga: the CDC 2025 Grade 1 modules, auto-extracted")
PROV_G4 = ("Grade 4 (2023 curriculum): Expressive Arts and Technology Studies from the CDC syllabi (2024), as transcribed "
           "by Funda and checked against the books; Home Economics and Hospitality from the CDC 2024 syllabus as "
           "transcribed by Funda, not checked here; English, Mathematics, Science and Social Studies: Funda draft, not "
           "a published CDC syllabus (the CDC has not published them)")
FOOTER = {4: "[Syllabus: Grade 4 2023: CDC Expressive Arts, Technology, HEH; English, Maths, Science, Social Studies: "
             "Funda draft, not a published CDC syllabus]"}
# Grade 4 learning areas are separate subjects (maths and science are no longer one area)
AREA_WORDS_G4 = [
    ("maths", r"maths?|mathematics|numeracy"),
    ("science", r"science|integrated science|sayaasi"),
    ("english", r"english|literacy|cikuwa|cingisi"),
    ("social", r"social studies"),
    ("arts", r"expressive arts|art|music|physical education"),
    ("home_economics", r"home economics|hospitality"),
    ("technology", r"technology studies|technology|ict"),
]
AREA_RX_G4 = [(a, re.compile(r"(?i)(?<![a-z])(?:" + rx + r")(?![a-z])")) for a, rx in AREA_WORDS_G4]


def provenance(grade):
    return PROV_G4 if int(grade) >= 4 else PROV_G13


def term_now(now):
    return 1 if now.month <= 4 else 2 if now.month <= 8 else 3


class Syllabus:
    def __init__(self, plugin):
        self.P = plugin
        self.items = [it for it in plugin.items.values() if it["kind"] == "syllabus"]
        self.by_id = {it["id"]: it for it in self.items}
        self.index = Index(plugin, kinds=("syllabus",))
        self.areas = {}
        for it in self.items:
            self.areas.setdefault(it["area"], it["area_label"])

    # ------------------------------------------------------------------------------------------------ views
    def view(self, it, short=False):
        v = {"id": it["id"], "code": it["code"], "cdc_code": it.get("cdc_code"), "n": it.get("n"),
             "title": it["title"], "grade": it["grade"], "area": it["area"], "lang": it.get("lang"),
             "topic": it["topic"], "sub_topic": it["sub_topic"], "covered": it["covered"],
             "covered_by": it.get("covered_by"),
             "links": [{k: ln.get(k) for k in ("id", "kind", "title", "match", "served", "strategy", "span")}
                       for ln in it.get("links") or []]}
        if it.get("auto_extracted"):
            v["auto_extracted"] = True
        if it.get("draft_label"):          # Grade 4 Funda drafts: said on every competence (24 Oct 2026)
            v["draft_label"] = it["draft_label"]
        if it.get("related"):
            v["related"] = it["related"]
        if short:
            return v
        v.update(learning_activities=it.get("learning_activities") or [],
                 expected_standard=it.get("expected_standard"), pacing=it.get("pacing"), prov=it.get("prov"),
                 origin=it.get("origin"), status=it.get("status"), module=it.get("module"))
        return v

    def grades(self):
        return sorted({it["grade"] for it in self.items})

    def tree(self, grade, area=None, topic=None, sub=None):
        """grade > area > topic > sub-topic > competence (full views at the leaves). Optional filters narrow it."""
        out = []
        for a in self.areas:
            if area and a != area:
                continue
            its = [it for it in self.items if it["grade"] == int(grade) and it["area"] == a]
            if not its:
                continue
            label = its[0]["area_label"]      # per grade: 'english' is "English (Literacy and Language)" in G1-3
            topics = {}
            for it in its:
                if topic is not None and it["topic"]["n"] != int(topic):
                    continue
                if sub is not None and it["sub_topic"]["n"] != int(sub):
                    continue
                t = topics.setdefault(it["topic"]["n"], {"n": it["topic"]["n"], "title": it["topic"]["title"],
                                                         "sub_topics": {}})
                s = t["sub_topics"].setdefault(it["sub_topic"]["n"], {
                    "n": it["sub_topic"]["n"], "title": it["sub_topic"]["title"],
                    "pacing": {k: it["pacing"].get(k) for k in ("term", "expected_week", "weeks_needed", "weight")},
                    "competences": []})
                s["competences"].append(self.view(it))
            out.append({"area": a, "label": label, "competences": len(its),
                        "covered": sum(1 for it in its if it["covered"]),
                        "topics": [dict(t, sub_topics=list(t["sub_topics"].values()))
                                   for _, t in sorted(topics.items())]})
        return {"grade": int(grade), "areas": out, "pacing_note": PACING_LABEL, "provenance": provenance(grade)}

    def week(self, grade, term, week=None, area=None):
        """the sub-topics Funda's pacing puts in this term (and around this week: started on or before it and not yet
        finished, or, when nothing spans it, the nearest start). Chitonga has terms but no weeks (the module's own)."""
        grade, term = int(grade), int(term)
        week = int(week) if week not in (None, "") else None
        rows = {}
        for it in self.items:
            if it["grade"] != grade or (area and it["area"] != area) or it["pacing"].get("term") != term:
                continue
            p = it["pacing"]
            key = (it["area"], it["topic"]["n"], it["sub_topic"]["n"])
            r = rows.setdefault(key, {"area": it["area"], "label": it["area_label"], "topic": it["topic"],
                                      "sub_topic": it["sub_topic"], "start": p.get("expected_week"),
                                      "weeks": p.get("weeks_needed") or 1, "weight": p.get("weight"),
                                      "competences": []})
            r["competences"].append(self.view(it, short=True))
        rows = list(rows.values())
        for r in rows:
            r["end"] = (r["start"] + r["weeks"] - 1) if r["start"] else None
        picked, how = rows, "the whole term"
        if week is not None:
            now = [r for r in rows if r["start"] and r["start"] <= week <= r["end"]]
            how = f"week {week}"
            if not now:
                timed = [r for r in rows if r["start"]]
                if timed:
                    best = min(abs(r["start"] - week) for r in timed)
                    now = [r for r in timed if abs(r["start"] - week) == best]
                    how = f"nothing spans week {week}: the nearest start (week {now[0]['start']})"
            untimed = [r for r in rows if not r["start"]]
            picked = now + untimed
        picked.sort(key=lambda r: (r["area"], r["start"] or 99, r["topic"]["n"], r["sub_topic"]["n"]))
        return {"grade": grade, "term": term, "week": week, "area": area, "matched": how, "rows": picked,
                "pacing_note": PACING_LABEL,
                "chitonga_note": "Chitonga: the term is the module's own; the modules give no weeks, so all of the "
                                 "term's cycles are listed in module order. Term 1 module not published."}

    def coverage(self, grade=None, area=None):
        out = {}
        for it in self.items:
            if (grade and it["grade"] != int(grade)) or (area and it["area"] != area):
                continue
            c = out.setdefault(str(it["grade"]), {}).setdefault(it["area"], {
                "label": it["area_label"], "competences": 0, "covered": 0, "by_code": 0, "by_code_broad": 0,
                "by_title": 0, "uncovered": []})
            c["competences"] += 1
            if it["covered"]:
                c["covered"] += 1
                c["by_" + it["covered_by"]] += 1
            else:
                c["uncovered"].append({"id": it["id"], "code": it["code"], "title": it["title"],
                                       "sub_topic": it["sub_topic"]["title"], "topic": it["topic"]["title"],
                                       "term": it["pacing"].get("term"),
                                       "foundational_weight": it["pacing"].get("foundational_weight"),
                                       "not_served": [ln["id"] for ln in it.get("links") or [] if not ln["served"]]})
        for g in out.values():
            for c in g.values():
                c["uncovered"].sort(key=lambda r: (-(r["foundational_weight"] or 0), r["code"]))
        return {"coverage": out, "note": "covered = at least one card or lesson we can serve, linked by syllabus code "
                                         "(code_broad: only by an item that claims more than four competences) or by "
                                         "a title lead (unverified). Uncovered, most foundational first: the content "
                                         "backlog."}

    @staticmethod
    def topic_words(text):
        """the request minus the words that only say 'syllabus', a grade, a term, a week or a subject"""
        t = text or ""
        for _, rx in AREA_RX + AREA_RX_G4:
            t = rx.sub(" ", t)
        t = TERM_RX.sub(" ", WEEK_RX.sub(" ", t))
        return " ".join(w for w in re.findall(r"[A-Za-z']+", t) if w.lower() not in META)

    def search(self, text, grades=None, area=None, k=5, prefer=None):
        """syllabus items for the request's topic words. grades: a FILTER (a grade the teacher named); prefer: the
        class's grades, a preference only (the index's +0.2 grade bonus: a Grade 2 teacher asking about sorting still finds
        the Grade 1 competence when it is the better match)"""
        q = self.topic_words(text)
        if not q.strip():
            return []
        hits = []
        for _, cover, th, iid in self.index.search(q, k=40, grades=grades or prefer, min_cover=0.5):
            it = self.by_id[iid]
            if area and it["area"] != area:
                continue
            if grades and it["grade"] not in grades:
                continue
            hits.append(it)
        return hits[:k]

    # ------------------------------------------------------------------------------------------------ reading
    def read(self, text, grades=None, now=None):
        """{grade, grade_named, area, term, week, term_from} from a request (grade: named, else the class's single
        grade)"""
        have = set(self.grades())
        gs = [g for g in grades_in(text) if g in have]
        grade = gs[0] if gs else (grades[0] if grades and len(grades) == 1 and grades[0] in have else None)
        named = bool(gs)
        rxs = AREA_RX_G4 if grade and grade >= 4 else AREA_RX
        area = next((a for a, rx in rxs if rx.search(text or "")), None)
        term, term_from = None, None
        m = TERM_RX.search(text or "")
        if m:
            if m.group(1):
                term, term_from = int(m.group(1)), "named"
            elif m.group(2):
                term, term_from = {"first": 1, "second": 2, "third": 3}[m.group(2).lower()], "named"
            elif now is not None:
                t = term_now(now)
                if m.group(3).lower() == "next":
                    t = t % 3 + 1
                term, term_from = t, f"{m.group(3).lower()} term = Term {t} by the month"
        w = WEEK_RX.search(text or "")
        return {"grade": grade, "grade_named": named, "area": area, "term": term,
                "week": int(w.group(1)) if w else None, "term_from": term_from}

    # ------------------------------------------------------------------------------------------------ WhatsApp
    def _material(self, it):
        served = [ln for ln in it.get("links") or [] if ln["served"]]
        held = [ln for ln in it.get("links") or [] if not ln["served"]]
        if served:
            s = "; ".join(f"{'Lesson' if ln['kind'] == 'lesson' else 'Card'} '{ln['title']}'"
                          + (" (title match, unverified)" if ln["match"] == "title" else "") for ln in served[:3])
            return "Our material: " + s
        if held:
            return "Our material: none we can serve yet (a card is waiting for a check)"
        return "Our material: none yet"

    def _code(self, it):
        if it["area"] == "chitonga":
            return f"Module Term {it['module']['term']}, cycle {it['module']['cycle']}"
        return it["code"] + (f" (printed {it['cdc_code']})" if it.get("cdc_code") and it["cdc_code"] != it["code"]
                             else "")

    def _pacing(self, it):
        p = it.get("pacing") or {}
        if it["area"] == "chitonga":
            return f"Term {p.get('term')} (module)"
        if not p.get("term"):
            return ""
        w = p.get("expected_week")
        n = p.get("weeks_needed") or 1
        return f"Term {p['term']}" + (f", week {w}" + (f"-{w + n - 1}" if n > 1 else "") if w else "")

    def answer(self, text, grades=None, now=None, limit=1400):
        """{'body', 'origins', 'flags'} for the WhatsApp route; English frame, CDC / module text as printed"""
        r = self.read(text, grades, now)
        origins = ["CDC"]
        hits = self.search(text, grades=[r["grade"]] if r["grade_named"] else None, area=r["area"], k=3,
                           prefer=[r["grade"]] if r["grade"] and not r["grade_named"] else None)
        lines = []
        if hits:
            for it in hits:
                if it.get("auto_extracted"):
                    origins.append("B")
                acts = it.get("learning_activities") or []
                lines += [f"Grade {it['grade']} {it['area_label']}, {self._code(it)}",
                          f"*{it['title']}*" + (" [Tonga as printed in the module, auto-extracted]"
                                                if it.get("auto_extracted") else ""),
                          f"Sub-topic: {it['sub_topic']['title']}",
                          "Activities: " + "; ".join(acts[:3]) + (f" (+{len(acts) - 3} more)" if len(acts) > 3 else ""),
                          self._material(it)]
                pc = self._pacing(it)
                if pc:
                    lines.append(f"When: {pc} [{PACING_LABEL}]")
                lines.append("")
            body = "\n".join(lines).strip()
        elif r["grade"] and r["area"]:
            term = r["term"] or (term_now(now) if now else None)
            if term is None:
                term = 1
            wk = self.week(r["grade"], term, r["week"], r["area"])
            head = (f"Grade {r['grade']} {self.areas.get(r['area'], r['area'])}, Term {term}"
                    + (f", around week {r['week']}" if r["week"] else "")
                    + (f" ({r['term_from']})" if r["term_from"] and r["term_from"] != "named" else
                       " (this term by the month)" if not r["term"] else "")
                    + f".\n[{PACING_LABEL}]")
            lines = [head]
            if not wk["rows"]:
                lines.append("Nothing in the syllabus we hold for this." + (
                    " The Chitonga Term 1 module is not published, and we hold the Grade 1 modules only."
                    if r["area"] == "chitonga" else ""))
            for row in wk["rows"]:
                when = (f"Wk {row['start']}" + (f"-{row['end']}" if row["end"] and row["end"] != row["start"] else "")
                        if row["start"] else "Module")
                comps = "; ".join(f"{c['code']} {c['title']}" + ("" if c["covered"] else " [no material yet]")
                                  for c in row["competences"])
                lines.append(f"- {when}: {row['sub_topic']['title']}: {comps}")
            body = "\n".join(lines)
        elif r["grade"]:
            cov = self.coverage(r["grade"])["coverage"].get(str(r["grade"]), {})
            body = (f"Grade {r['grade']}: which subject? " + "; ".join(
                f"{c['label']} ({c['competences']} competences)" for c in cov.values())
                    + ". For example: 'Grade {g} maths syllabus this term', or name a topic.".format(g=r["grade"]))
        else:
            body = ("Tell me the grade and subject (for example 'Grade 2 maths syllabus this term', 'Grade 1 English "
                    "week 3 of term 2'), or a topic ('competence for sorting things').")
        if len(body) > limit:
            body = body[:limit].rsplit("\n", 1)[0] + "\n(more: name a sub-topic or a week)"
        body += "\n" + FOOTER.get(r["grade"] or 0, "[Syllabus: CDC 2024 Lower Primary; Chitonga: CDC 2025 modules]")
        return {"body": body, "origins": list(dict.fromkeys(origins)), "flags": []}


_CACHE = {}


def for_plugin(P):
    """one Syllabus per loaded plug-in object (built on first use)"""
    k = (id(P), P.fingerprint)
    if k not in _CACHE:
        _CACHE[k] = Syllabus(P)
    return _CACHE[k]
