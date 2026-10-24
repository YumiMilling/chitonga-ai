"""Check the Funda curriculum JSON against the CDC 2024 Lower Primary Syllabi text (24 Oct 2026, the Jesuit).

Why. The syllabus items of the education plug-in (build_syllabus.py) carry the Funda JSON's competence titles, learning
activities and expected standards as "CDC verbatim". Funda's README says its provenance varies (MoE / CDC extracts,
ai_draft, human), so the claim is checked here, field by field, against the decoded CDC book (Ministry of Education,
CDC 2024, ISBN 978-9982-00-943-6), not taken on trust. Every English-language competence is compared (titles,
activities, expected standards), and the topic / sub-topic titles; the result is written into the plug-in's
syllabus provenance and printed.

How. The decoded text keeps the table's reading order per row: the code (1.9.1.1), the competence title, the bullets
(activities), the expected standard. A competence's BLOCK is everything from its code to the next code of the same
grade, with page furniture removed (page markers, the running footer, the repeated table header). Both sides are
normalised to [a-z0-9] after the PDF's ligature glyphs are mapped (൵ ff, ¿ fi, À fl, ৽ / ൶ ffi, ৼ ff), so a check
is "the Funda text, normalised, is in the CDC block" (title: at its start). Symbols (∈, ∉, ∅, <, >) decode to junk
in the PDF and drop out on both sides.

Not checked here: the Chitonga file (g1-zamlang-tonga.json is auto-extracted from the 2025 Chitonga modules, a
different book), and the Funda fields that the CDC book does not carry at all (term_number, expected_week,
weeks_needed, weight, foundational_weight, why_it_matters, checkpoint_method, lessons): the book has no pacing for
these learning areas (it gives 5 hours / 10 periods a week per area and nothing per topic; the only weekly schedules
are for the Zambian-language sounds), so those fields are Funda's own.

    python -X utf8 tools/chitonga/harness/plugins/education/syllabus_check.py [--show N]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CURR = os.path.join(REPO, "internal", "funda", "content", "curriculum", "2023", "lower-primary")
CDC_TXT = os.path.normpath(os.path.join(
    REPO, "..", "zamai-corpus-raw", "chitonga_mono", "moe_modules_2026-10-19",
    "LOWER-PRIMARY-SYLLABI-GRADE-1-3-FINAL-CAMERA-REDY-1", "LOWER-PRIMARY-SYLLABI-GRADE-1-3-FINAL-CAMERA-REDY-1.decoded.txt"))
CDC_BOOK = "CDC 2024 Lower Primary Education Syllabi, Grades 1-3 (Ministry of Education; ISBN 978-9982-00-943-6)"

# the four learning areas' section headers in the book, in order; Zambian Languages (section B) is not in Funda's files
SECTIONS = [("litlan", r"^\s*GRADE ([123]):? ENGLISH LANGUAGE\s*$"),
            ("zamlang", r"^\s*GRADE ([123]):? ZAMBIAN LANGUAGES\s*$"),
            ("mathsci", r"^\s*GRADE ([123]):? MATHEMATICS AND SCIENCE\s*$"),
            ("cts", r"^\s*GRADE ([123]):? CREATIVE AND TECHNOLOGY STUDIES\s*$")]
END_RX = re.compile(r"^=====PAGE 166\b|^\s*APPENDI|Weekly Schedule", re.I)   # the appendix after CTS Grade 3
LIG = {"൵": "ff", "ৼ": "ff", "¿": "fi", "À": "fl", "৽": "ffi", "൶": "ffi"}
FURNITURE = re.compile(r"^\s*(=====PAGE \d+|Lower Primary Syllabi Grade 1-3|\d{1,3}|TOPIC|SUB-TOPIC|SPECIFIC|"
                       r"COMPETENCES|SPECIFIC COMPETENCES|LEARNING ACTIVITIES|EXPECTED|STANDARD|EXPECTED STANDARDS?|"
                       r"LISTENING AND SPEAKING|READING|WRITING|GRADE [123]:?.*)\s*$")
FILES = {"litlan": "LITLAN", "mathsci": "MATHSCI", "cts": "CTS"}


def norm(s):
    s = (s or "").lower()
    for k, v in LIG.items():
        s = s.replace(k, v)
    return re.sub(r"[^a-z0-9]", "", s)


def _clean(line):
    return line.replace("\u0087", " ").replace("\uf0b7", " ").strip().lstrip("=").strip()


def load_cdc(path=CDC_TXT):
    """{(area, grade): {"comp": {code: block}, "sub": {code: text}, "topic": {code: text}}}; codes as printed"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    starts = []
    for i, ln in enumerate(lines):
        if "...." in ln:
            continue                      # the table of contents
        for area, rx in SECTIONS:
            m = re.match(rx, ln)
            if m and not any(s[1] == area and s[2] == int(m.group(1)) for s in starts):
                # the header is printed at the foot of the section's first page in the decoded order: the section
                # starts at that page's marker
                pg = next((x for x in range(i, -1, -1) if lines[x].startswith("=====PAGE")), i)
                starts.append((pg, area, int(m.group(1))))
    # one range per learning AREA (its three grades together): the grade footers do not always sit on a grade's first
    # page in the decoded order (Grade 3 Maths and Science 3.1.1.5-3.1.1.7 decode before the Grade 3 footer), so a
    # code's grade is its own first digit, never the page it was found on
    first = {}
    for pg, area, g in sorted(starts):
        first.setdefault(area, pg)
    order = sorted((pg, area) for area, pg in first.items())
    out = {}
    for k, (i, area) in enumerate(order):
        j = order[k + 1][0] if k + 1 < len(order) else next(
            (x for x in range(i + 1, len(lines)) if END_RX.search(lines[x])), len(lines))
        # (?!\d), not a word boundary: the book often runs the title into the code ("2.12.2.1Perform tumbling")
        comp_rx = re.compile(r"^([123])\.(\d+)\.(\d+)\.(\d+)(?:\.\d+)?(?!\d)\.?\s*=?\s*(.*)$")
        sub_rx = re.compile(r"^([123])\.(\d+)\.(\d+)(?![.\d]*\d)\.?\s*=?\s*(.*)$")
        top_rx = re.compile(r"^([123])\.(\d+)(?![\d]|\.\d)\.?\s*=?\s*([A-Z][A-Z ,'&/-]*.*)$")
        d = {g: {"sub": {}, "topic": {}} for g in (1, 2, 3)}
        entries = {g: [] for g in (1, 2, 3)}    # (code, [lines]) in book order: some codes are printed twice
        cur = None
        # a sub-topic and its first competence can share one decoded line ("1.1.4 =Classroom Rules =1.1.4.1 Use
        # appropriate"): split before every code with at least three parts that is not at the start of the line
        split_rx = re.compile(r"(?<=[\s=])(?=[123]\.\d+\.\d+(?:\.\d+)?(?!\d))")
        pieces = []
        for ln in lines[i + 1:j]:
            if FURNITURE.match(ln.strip()):
                continue
            pieces += split_rx.split(ln)
        for ln in pieces:
            t = _clean(ln)
            if not t or FURNITURE.match(t):
                continue
            m = comp_rx.match(t)
            if m:
                g = int(m.group(1))
                entries[g].append((f"{g}.{m.group(2)}.{m.group(3)}.{m.group(4)}", [m.group(5)]))
                cur = ("entry", g, len(entries[g]) - 1)
                continue
            m = sub_rx.match(t)
            if m:
                g = int(m.group(1))
                cur = ("sub", g, f"{g}.{m.group(2)}.{m.group(3)}")
                d[g]["sub"].setdefault(cur[2], []).append(m.group(4))
                continue
            m = top_rx.match(t)
            if m and m.group(3).strip():
                g = int(m.group(1))
                cur = ("topic", g, f"{g}.{m.group(2)}")
                d[g]["topic"].setdefault(cur[2], []).append(m.group(3))
                continue
            if cur and cur[0] == "entry":
                entries[cur[1]][cur[2]][1].append(t)
            elif cur:
                d[cur[1]][cur[0]][cur[2]].append(t)
        for g in (1, 2, 3):
            res = {kind: {c: " ".join(v) for c, v in d[g][kind].items()} for kind in ("sub", "topic")}
            res["entries"] = [(c, " ".join(v)) for c, v in entries[g]]
            res["comp"] = {}
            for c, v in res["entries"]:
                res["comp"].setdefault(c, v)
            out[(area, g)] = res
    return out


def _starts(block, text):
    return bool(text) and norm(block).startswith(norm(text))


def _inside(block, text):
    return bool(text) and norm(text) in norm(block)


def check(cdc=None, curr=CURR):
    """[{area, grade, code, cdc_code, title_ok, activities (found, total), standard_ok, sub_ok, topic_ok, ...}] for every
    English-language competence of the Funda files. cdc_code is the code printed in the book when Funda renumbered
    (g3-cts topics 15-19: the book's own numbering skips; matched by competence title within the grade)."""
    cdc = cdc or load_cdc()
    rows = []
    for area, code in FILES.items():
        for g in (1, 2, 3):
            path = os.path.join(curr, f"g{g}-{area}.json")
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            book = cdc.get((area, g)) or {"comp": {}, "sub": {}, "topic": {}}
            for t in d["topics"]:
                for s in t["sub_topics"]:
                    for c in s["specific_competences"]:
                        fc = f"{g}.{t['number']}.{s['number']}.{c['number']}"
                        cc = fc
                        ents = book.get("entries", [])
                        same = [b for k, b in ents if k == fc and _starts(b, c["title"])]
                        block = same[0] if same else book["comp"].get(fc, "")
                        if not same:
                            alt = [(k, b) for k, b in ents if _starts(b, c["title"])]
                            if len({k for k, _ in alt}) == 1:
                                cc, block = alt[0]
                        acts = c.get("learning_activities") or []
                        found = [a for a in acts if _inside(block, a)]
                        sub_code = cc.rsplit(".", 1)[0]
                        rows.append({
                            "area": area, "grade": g, "code": fc, "cdc_code": cc, "found": bool(block),
                            "title_ok": _starts(block, c["title"]),
                            "activities": [len(found), len(acts)],
                            "activities_missing": [a for a in acts if a not in found],
                            "standard_ok": _inside(block, c.get("expected_standard")),
                            "sub_ok": _starts(book["sub"].get(sub_code, ""), s["title"]),
                            "topic_ok": _inside(book["topic"].get(sub_code.rsplit(".", 1)[0], ""), t["title"]),
                            "title": c["title"], "block": block[:600]})
    return rows


def summary(rows):
    out = {}
    for r in rows:
        k = f"g{r['grade']}-{r['area']}"
        s = out.setdefault(k, {"competences": 0, "found": 0, "title_ok": 0, "activities_found": 0, "activities": 0,
                               "standard_ok": 0, "sub_ok": 0, "topic_ok": 0, "renumbered": 0})
        s["competences"] += 1
        s["found"] += r["found"]
        s["title_ok"] += r["title_ok"]
        s["activities_found"] += r["activities"][0]
        s["activities"] += r["activities"][1]
        s["standard_ok"] += r["standard_ok"]
        s["sub_ok"] += r["sub_ok"]
        s["topic_ok"] += r["topic_ok"]
        s["renumbered"] += r["code"] != r["cdc_code"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0, help="print N rows that fail a check")
    ap.add_argument("--code", help="print one competence (e.g. litlan:2.9.1.1) beside its CDC block")
    a = ap.parse_args()
    if not os.path.isfile(CDC_TXT):
        sys.exit(f"CDC text not found: {CDC_TXT}")
    rows = check()
    if a.code:
        area, code = a.code.split(":")
        for r in rows:
            if r["area"] == area and r["code"] == code:
                print(json.dumps(r, ensure_ascii=False, indent=1))
        return
    print(json.dumps(summary(rows), indent=1))
    bad = [r for r in rows if not (r["title_ok"] and r["standard_ok"] and r["activities"][0] == r["activities"][1])]
    print(f"{len(rows)} competences, {len(bad)} with a field not found verbatim")
    for r in bad[:a.show]:
        print(json.dumps({k: r[k] for k in ("area", "code", "cdc_code", "title_ok", "activities", "standard_ok",
                                            "activities_missing", "title")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
