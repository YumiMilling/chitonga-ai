"""Check Funda's Grade 4 transcriptions of the CDC UPPER-PRIMARY syllabi against the books (24 Oct 2026, the Jesuit).

Why. build_syllabus.py serves Funda's Grade 4 JSON as syllabus items. Three of the seven Grade 4 files say they come
from CDC syllabi (Expressive Arts, finalised 31 Jan 2025; Technology Studies 2024; Home Economics and Hospitality
2024), but Funda marks the Expressive Arts and Technology competences `authored_by: ai_draft`, and a first look showed
Funda's activities expanded from the book ("Identifying correct attire and tools." in the book, "... for Expressive
Arts activities." in Funda) and a renumbered code (the book prints 4.1.2.2 "Compose health related songs", Funda
4.1.2.1). So, as for Grades 1-3 (syllabus_check.py), every field is checked against the book's text, not taken on
trust, and the result travels in each item's `prov`.

Books (downloaded 24 Oct 2026 from the Ministry's Directorate of Curriculum Development list, edu.gov.zm/?page_id=1142,
into zamai-corpus-raw/moe_upper_primary_2026-10-24/; text by pdftotext). The February 2025 uploads are used: the April
2025 re-uploads of the same books carry a shifted font encoding (text unreadable without decoding).
  Expressive Arts  EXPRESSIVE-ARTS_UPPER-PRIMARY-FINALISED-31-01-25.pdf
  Technology       Technology-Studies-Syllabus.pdf (ISBN 978-9982-54-005-8, the one Funda cites)
  HEH              NOT available: not on the Ministry's list, not on this machine (Funda's source-pdfs/heh-g4-6.pdf was
                   not copied into internal/funda). HEH stays 'unchecked'; its printed codes are taken from Funda's own
                   note ("CDC numbers 4.8 twice (Introduction to Travel and Sewing)"), see build_syllabus.py.

How. The Grade 4 section of each book (from the first 4.x.x.x code to the first 5.1.1.1) is normalised to [a-z0-9];
a Funda field is 'verbatim' when its normalised text is inside the section. The pdftotext reading order splits table
columns, so a field is checked against the whole Grade 4 section, not a row: a check of presence, not of position.
The printed code of a competence is the 4.x.x.x printed right before its title.

    python -X utf8 tools/chitonga/harness/plugins/education/syllabus_check_up.py [--show]
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CURR = os.path.join(REPO, "internal", "funda", "content", "curriculum", "2023", "upper-primary")
BOOKDIR = os.path.normpath(os.path.join(REPO, "..", "zamai-corpus-raw", "moe_upper_primary_2026-10-24"))
BOOKS = {
    "expressive": {"txt": os.path.join(BOOKDIR, "EXPRESSIVE-ARTS_UPPER-PRIMARY-FINALISED-31-01-25.txt"),
                   "book": "CDC Expressive Arts Syllabus, Grade 4-6 (Ministry of Education, CDC; finalised 31 Jan 2025)",
                   "url": "https://www.edu.gov.zm/wp-content/uploads/2025/02/EXPRESSIVE-ARTS_UPPER-PRIMARY-FINALISED-"
                          "31-01-25.pdf"},
    "tech": {"txt": os.path.join(BOOKDIR, "Technology-Studies-Syllabus.txt"),
             "book": "CDC Technology Studies Syllabus, Grade 4-6 (Ministry of Education, CDC 2024; ISBN 978-9982-54-005-8)",
             "url": "https://www.edu.gov.zm/wp-content/uploads/2025/02/Technology-Studies-Syllabus.pdf"},
}
CODE_RX = re.compile(r"(?<![\d.])(4\.\d{1,2}\.\d{1,2}\.\d{1,2})(?![\d])\s*([A-Za-z][^\n]*)?")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def section(path):
    """the Grade 4 matrix text of a book: from a little before the first 4.1.1.1 to the first 5.1.1.1"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    start = next(i for i, ln in enumerate(lines) if re.search(r"(?<![\d.])4\.1\.1\.1(?!\d)", ln) and "...." not in ln)
    end = next((i for i in range(start, len(lines)) if re.search(r"(?<![\d.])5\.1\.1\.1(?!\d)", lines[i])), len(lines))
    return "\n".join(lines[max(0, start - 12):end])


def printed_codes(text):
    """[(code, normalised text after it)] in book order"""
    out = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        for m in CODE_RX.finditer(ln):
            rest = m.group(2) or ""
            if not norm(rest) and i + 1 < len(lines):
                rest = lines[i + 1]
            out.append((m.group(1), norm(rest)))
    return out


def check():
    """rows {area, code, cdc_code, title_ok, standard_ok, activities: (ok, n), activities_missing, sub_ok, topic_ok,
    book, url} for every Grade 4 competence of the books on this machine"""
    rows = []
    for stem, b in BOOKS.items():
        if not os.path.isfile(b["txt"]):
            continue
        text = section(b["txt"])
        S = norm(text)
        codes = printed_codes(text)
        with open(os.path.join(CURR, f"g4-{stem}.json"), encoding="utf-8") as f:
            d = json.load(f)
        for t in d["topics"]:
            for s in t["sub_topics"]:
                for c in s["specific_competences"]:
                    code = f"4.{t['number']}.{s['number']}.{c['number']}"
                    nt = norm(c["title"])
                    acts = [a.strip() for a in c.get("learning_activities") or []]
                    miss = [a for a in acts if norm(a) not in S]
                    k = nt[:24]
                    cdc = next((cd for cd, rest in codes if rest and (rest.startswith(k) or k.startswith(rest[:24]))),
                               None)
                    rows.append({"area": stem, "code": code, "cdc_code": cdc or code, "cdc_code_found": bool(cdc),
                                 "title_ok": nt in S, "standard_ok": norm(c.get("expected_standard")) in S,
                                 "activities": (len(acts) - len(miss), len(acts)), "activities_missing": miss,
                                 "sub_ok": norm(s["title"]) in S, "topic_ok": norm(t["title"]) in S,
                                 "book": b["book"], "url": b["url"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    rows = check()
    for stem in BOOKS:
        r = [x for x in rows if x["area"] == stem]
        if not r:
            print(stem, ": book text not on this machine")
            continue
        print(f"{stem}: {len(r)} competences; titles verbatim {sum(x['title_ok'] for x in r)}; standards verbatim "
              f"{sum(x['standard_ok'] for x in r)}; activities verbatim {sum(x['activities'][0] for x in r)}/"
              f"{sum(x['activities'][1] for x in r)}; sub-topics {sum(x['sub_ok'] for x in r)}; printed code differs "
              f"{sum(x['cdc_code'] != x['code'] for x in r)}; printed code not found {sum(not x['cdc_code_found'] for x in r)}")
        if a.show:
            for x in r:
                if not (x["title_ok"] and x["standard_ok"]) or x["activities_missing"] or x["cdc_code"] != x["code"]:
                    print("  ", x["code"], "printed", x["cdc_code"], "title", x["title_ok"], "std", x["standard_ok"],
                          "acts", x["activities"], x["activities_missing"][:2])


if __name__ == "__main__":
    main()
