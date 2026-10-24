"""What a teacher may read: internal notes and codes taken out of the served lesson data (24 Oct 2026, the Jesuit).

Persona review M3 (REVIEW_PERSONA_2026-10-24.md): reviewer codes ("T01 23 Oct (capital T added)"), desk notes ("his
version; ours was: ..."), build provenance ("Claude-written English (the Jesuit, 23 Oct)", "[C: AI draft (Claude) ...]",
"Funda's Grade 4 card", internal file paths as a word's source) reached teacher screens. They stay in the library and
on the desk; the export to the live app runs every served part through clean() so none of it ships, and leak_report()
counts what is left (export_netlify.py fails the build if anything is).

The rules, by key (the same strings the app's render layer cleans for copies saved before this export):
- note on a Tonga line or key word: dropped (all of them are desk notes: checks, versions, reviewer codes);
- origin_text: dropped (the badge letter carries the origin); a list of reviewer codes in origin: 'teacher';
- toi: "(T01)", "(T01 23 Oct: ...)" removed from the Tonga ("word (T01) / word (T09)" -> "word / word");
- source: an internal path dropped; "; English: ... (Claude) ... the Jesuit ..." clauses dropped, the book kept;
- label / review_note / unit note: provenance rewritten in plain words ("AI draft, needs review");
- print (a printed-guide reference "M2 p25-26 (...)"): the book's name instead of its code.
Nothing here touches a Tonga word itself: only the bracketed reviewer codes after it.
"""
import re

BOOKS = {"G": "Chitonga Giledi 1 teacher's guide (CDC, 2013)", "G2": "Chitonga Giledi 2 teacher's guide (CDC, 2014)",
         "M2": "Chitonga Grade 1 teaching module, Term 2 (CDC, 2025)",
         "M3": "Chitonga Grade 1 teaching module, Term 3 (CDC, 2025)",
         "E2": "ECE Chitonga module, Term 2 (CDC, 2025)", "E3": "ECE Chitonga module, Term 3 (CDC, 2025)",
         "LR": "Let's Read Grade 2 readers (Ministry of Education)", "NA": "Nondo Abayuni, a Grade 3 reader"}

TCODE = re.compile(r"\bT\d{2}\b")
DATE = re.compile(r"\b\d{1,2} Oct(?:ober)?(?: 20\d\d)?\b")
AI_LINE = "Written with AI from the CDC syllabus. Not yet checked by a teacher."

# what must never reach a teacher (leak_report)
LEAKS = {"claude": re.compile(r"Claude"), "jesuit": re.compile(r"Jesuit"), "reviewer code": TCODE,
         "desk note": re.compile(r"his version|ours was:|CHECK\b|\bconfirm which\b"),
         "C bracket": re.compile(r"\[C: "), "menu talk": re.compile(r"outside the Grade|Funda's Grade \d (?:card|draft)"),
         "internal path": re.compile(r"internal/|\.md\b"), "printed code": re.compile(r"^(?:M[23]|E[23]|G2?|LR|NA) p\d")}
# the keys whose strings a teacher reads (leak_report looks only at these; ids, regexes and search terms are not shown)
SHOWN = {"label", "note", "toi", "source", "review_note", "review_tag", "print", "title", "scope", "why", "what", "en",
         "text", "fix", "say", "when", "task", "how", "mastered", "developing", "next", "teach"}


def _label(s):
    t = s
    if re.search(r"Claude|Jesuit|\[C: ", t):
        if re.search(r"Funda teacher guide|Sample lesson", t):
            t = re.sub(r"\(English[,;] AI draft[^)]*\)", "(English, AI draft, needs review)", t)
        elif re.search(r"CDC syllabus", t):
            m = re.search(r"unit ([\d.]+(?: \([^)]*\))?)", t)
            t = "AI draft from the CDC syllabus" + (f", unit {m.group(1)}" if m else "") + ", needs review"
        else:
            t = "AI draft, needs review"
    t = re.sub(r"\(English; AI draft, not yet reviewed by a Zambian teacher\)", "(English, AI draft, needs review)", t)
    t = re.sub(r"\[C: ([^\]]*)\]", r"\1", t)
    t = t.replace("(Claude)", "").replace("Claude", "AI")
    t = re.sub(r"\s*\(?the Jesuit[^)]*\)?", "", t)
    return re.sub(r"\s{2,}", " ", t).strip(" ,;")


def _sentences(s):
    """a scope line or note without the editor's sentences (printed-module page notes, "Not in the 2013 guide", "Funda's
    Grade 3 card", file names)"""
    keep = [p for p in re.split(r"(?<=\.)\s+", s)
            if not re.search(r"\b(?:M[23]|E[23]|G2?) p\d|Not in the 2013 guide|Funda's Grade \d|\.md\b", p)]
    return " ".join(keep).strip()


def _unit_note(s):
    s = _sentences(s)
    if not s:
        return None
    if re.match(r"Cards \(English, drafted by Claude", s):
        return "Cards in English: AI draft, needs review."
    m = re.match(r"Funda draft, not a published CDC syllabus \(Funda's Grade (\d) draft\):\s*(.*)", s)
    if m:
        return "Draft unit, not from a published CDC syllabus: " + m.group(2)
    if re.match(r"(?:M[23]|E[23]|G2?) (?:week|p)\d|.*\bno lesson of its own\b", s):
        return None          # an editor's note on the printed modules
    return _label(s)


def _source(s):
    if TCODE.search(s) or s.lower().startswith("teacher"):
        return "A Tonga teacher"     # desk teachers are not named in the app (in-app contributors choose initials)
    s = re.sub(r"\s*\([A-Z_]+\.md[^)]*\)", "", s)
    s = re.sub(r"\s*\b[A-Z_]+\.md,?", "", s)
    if s.startswith("internal/") or re.search(r"[\\/][\w.-]+\.(?:md|json|jsonl|txt)\b", s):
        return None
    parts = [p.strip() for p in re.split(r";\s*|\s\+\s", s) if p.strip()]
    keep = [p for p in parts if not re.search(r"English:|Claude|Jesuit|\bc=0?\.\d|translation pass", p)]
    keep = [DATE.sub("", p).strip(" ,") for p in keep]
    return "; ".join(p for p in keep if p) or None


def _toi(s):
    t = re.sub(r"\s*\(T\d{2}(?:[ ,][^)]*)?\)", "", s)
    # spelling notes for the desk ("(double aa; masamu = trees)"); a bracket with "=" in a lesson line can be maths
    # ("(3 + 4 = 7)"), so only the "double ..." form goes
    t = re.sub(r"\s*\(double [a-z]+[^)]*\)", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _print(s):
    out = []
    for p in re.split(r";\s*", s):
        m = re.match(r"(M[23]|E[23]|G2?|LR|NA) p(\d+)(?:-(\d+))?\s*(.*)", p)
        if not m:
            out.append(p)
            continue
        pages = m.group(2) + (f"-{m.group(3)}" if m.group(3) and m.group(3) != m.group(2) else "")
        out.append(f"{BOOKS[m.group(1)]}, p. {pages}" + (f" {m.group(4)}" if m.group(4) else ""))
    return "; ".join(out)


def clean(x, key="", parent=""):
    """the served copy of x, with the rules above applied by key; returns a new object"""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("origin_text",):
                continue
            if k == "note" and parent in ("box", "key_words", "labels"):
                continue
            if k == "origin" and isinstance(v, list):
                out[k] = ["teacher" if isinstance(o, str) and TCODE.fullmatch(o) else o for o in v]
                continue
            c = clean(v, k, key if isinstance(x, dict) and key else parent)
            if c is None and k in ("source", "note"):
                continue
            out[k] = c
        return out
    if isinstance(x, list):
        return [clean(v, key, key) for v in x]
    if not isinstance(x, str):
        return x
    if key == "source":
        return _source(x)
    if key == "toi":
        return _toi(x)
    if key == "print":
        return _print(x)
    if key == "note":
        return _unit_note(x)
    if key in ("scope", "what"):   # "The place-value guide is Funda's Grade 3 card"; "M3 p46 prints gw only ..."
        return _sentences(x)
    if key == "label" and re.match(r"T[1-3]: ", x):     # a unit label "T3: gw, pw" is Term 3
        return "Term " + x[1] + ": " + x[4:]
    if key in ("label", "review_note", "review_tag", "why"):
        return _label(x)
    return x


def leak_report(x, key="", path="", out=None):
    """{leak kind: [path, ...]} over the strings a teacher reads"""
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            leak_report(v, k, f"{path}.{k}", out)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            leak_report(v, key, f"{path}[{i}]", out)
    elif isinstance(x, str) and key in SHOWN:
        for name, rx in LEAKS.items():
            if rx.search(x):
                out.setdefault(name, []).append(f"{path}: {x[:90]}")
    return out
