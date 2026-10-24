"""Harness v2, translation memory (17 Oct 2026; TODO_ENGINE_HARNESS E4).

Before anything is generated, the English is looked up among REAL pairs: Tonga a person wrote. An exact match
(after normalising case, punctuation, spaces, curly quotes) returns that Tonga with its source; a near match is
returned as a reference for the drafter and the teacher, never as the answer (a near match can differ by exactly
the word that matters: 'I am hungry' / 'I am not hungry' are 0.9 alike by letters; so a near match whose negative or
numbers differ from the query is marked 'differs').

Sources, read at run time (they grow every day):
  books     internal/chitonga_train/vocab/new_books_2026-10-16/*pairs*.jsonl, confidence high or teacher only;
            skipped: *.fidelity*, *SUPERSEDED*, the machine file (its English is the stage 11 translator's), every row
            with held_out true or exam true. The English of a book pair is the Jesuit's context-aware guess (en_by);
            the Tonga is the book's. The source says so.
  teachers  internal/chitonga_train/desk/teacher_pairs_*.jsonl (heldout false), teacher_corrections_v0*.jsonl,
            desk/pairs.jsonl (the teacher's 'chosen')
  Thompson  zamai-corpus-raw/chitonga_mono/phrasebook_thompson/thompson_train.tsv (the training part only; the
            held-out Thompson lines live in llm/eval_v07 and are not read)
  lessons   (24 Oct 2026, translator optimisation step 1; Chris approved, K0) the teacher-checked Tonga of the teacher
            assistant, kind 'teacher', src = file + row; off with CHITONGA_TM_LESSONS=0 (or LESSONS = False + reload()):
            - the classroom-line rows of every "Say it in Tonga" box (assistant_library/grade1, grade1_new,
              grade23/boxes, place_value_try_2026-10-22; sections 'Classroom lines' and 'Added ...') whose Origin cell
              starts with a teacher code (T01 wrote it, 'T09 confirmed 24 Oct', 'T09 24 Oct (his version; ...)').
              The desk answers of desk/batches_2026-10-24*/ reach the boxes through desk/apply_batch.py + keys.json,
              so the box IS the applied answer; keys.json only adds the desk item to src. NOT taken: Origin 'D ...'
              (derived from teacher input, needs confirmation, even when the note names the teacher whose word it
              used), B / L rows (book Tonga under our English), a 'frame' row, '(dropped ...)', slot lines ({x}).
            - assistant_library/core_fixed_lines.json lines and buttons with status 'served' (teacher-written or
              teacher-confirmed, a teacher code in origin) and no {slot}; a 'variant_T09' beside a line is T09's own.
            - the English > Tonga sentences written in a desk answers file of batches_2026-10-24* ('7a. <English> >
              <Tonga>', 3+ English words; today T09's three rainfall lines, T09_03_answers.md).
            A cell is taken as the teacher wrote it; one slash group ('tupanga/tubamba') gives one row per
            alternative and '(class: X)' a row of its own; any other bracket or a second slash group: the row is
            skipped (a note or an option, not a line he wrote whole; skipped counts in memory()['lessons']).
CONFIDENCE TALLY (20 Oct 2026, Chris: "each teacher input counts at once"; FLYWHEEL_DESIGN.md section 000; replaces
the second-source gate written earlier the same day). Every exact row carries:
  confidence   the number of independent sources that give THIS written Tonga for this English: each teacher counts
               once, a book line once, Thompson once. A single teacher pair has confidence 1 and is usable at once.
  contradicts  other Tonga for the same English from other sources: stored beside it with its own count, never
               overwritten (variants and disagreements both stay visible; a person or more answers decide)
Serving (harness3.translate stage 0) picks the HIGHEST confidence and logs every count; ties keep the old order
(teacher, Thompson, book). Spelling variants count as different Tonga here (no folding).
The memory re-reads its sources when any of them changes (mtime check on every lookup), so the resident serve picks up
new teacher pairs without a restart; reload() forces it.

Leak guard: any Tonga or English that also occurs in a held-out row (any book file, held_out / exam) is dropped from
the memory, so a held-out line can never come back through a duplicate in a training file.
24 Oct 2026: the lesson boxes HOLD the stage 18 frozen teacher test (its 44 lines were drawn from them,
stage18_teacher_2324.py) and its dev slice, so every row whose English or Tonga equals OR is near (GUARD_NEAR /
GUARD_WORDS, wider than the serving near) any English, Tonga or alternative of those two files is dropped at build
(memory()['dropped18']). serve_held() is the same index plus the harness's held-out requests (the first 40 real
requests per teacher, held_out true in harness/pilot_logs), read at serve time with its own stamp so a new WhatsApp
message does not rebuild the memory; translator_service checks inputs and served rows against it.

NEAR (24 Oct 2026): near(text, direction) gives near rows in BOTH directions (English key for en>toi, Tonga key for
toi>en; lookup() is unchanged), each with `differs` (negation or numbers differ: English by h2_checks, Tonga by
h3_detect.toi_neg3 and h2_text.numbers_toi) and `strict` (the words that differ are only Tonga-invariant ones:
English a / an / the / please; Tonga sena / hena / na / ino / pe, or a typing variant: case, hyphen, ŋ / ng', ch / c;
doubled letters are NEVER folded, tonga_structure 2.8). translator_service uses them as rerank candidates only.

    python -X utf8 tools/chitonga/llm/h2_memory.py "English sentence"
"""
import csv
import difflib
import glob
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
VOC = os.path.join(CT, "vocab", "new_books_2026-10-16")
THOMPSON = os.path.join(REPO, "..", "zamai-corpus-raw", "chitonga_mono", "phrasebook_thompson", "thompson_train.tsv")
sys.path.insert(0, HERE)
NEAR_MIN = float(os.environ.get("HARNESS_TM_NEAR", "0.85"))
NEAR_WORDS = float(os.environ.get("HARNESS_TM_NEAR_WORDS", "0.75"))
# the held-out guard's near is WIDER than the serving near: anything a near lookup could reach from a held line is
# inside it (24 Oct 2026)
GUARD_NEAR, GUARD_WORDS = 0.80, 0.60
LIB = os.path.join(CT, "assistant_library")
BOX_DIRS = ("grade1", "grade1_new", os.path.join("grade23", "boxes"), "place_value_try_2026-10-22")
FIXED = os.path.join(LIB, "core_fixed_lines.json")
DESK = os.path.join(CT, "desk")
FINAL18 = os.path.join(CT, "runs", "2026-10-23_final_translator_data")
FROZEN18 = [os.path.join(FINAL18, f) for f in ("teacher_test_final18.jsonl", "teacher_dev_final18.jsonl")]
HARNESS_DATA = os.path.join(CT, "harness")
LESSONS = os.environ.get("CHITONGA_TM_LESSONS", "1") != "0"
TEACHER_CODE = re.compile(r"^T\d\d(?!\d)")
_mem = {}


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\[[^\]]*\]", " ", s)               # [is coming], [?...?]: the guesser's brackets
    s = re.sub(r"^\s*(\d+[.)]|[a-z][.)])\s+", "", s)  # numbered exercise lines
    # Codex refix2 P1-1, translator memory (20 Oct evening): the letters-only key made "Calculate 1+2." = "Calculate
    # 1-2." and "Write 3/4." = "Write 3.4.", so a maths line could get stored Tonga with the wrong numbers. Kept now:
    # + × ÷ = / % *, a minus before a digit, and a . , or / between digits; other punctuation still folds away
    toks = re.findall(r"(?<![\w.,/])-?\d+(?:[.,/]\d+)*|-(?=\s*\d)|[+×÷=%/*]|[\w']+", s.lower())
    return " ".join(toks)


# 21 Oct 2026 (audit decisions, step 1): labels a book or a teacher puts in front of the same item. Raw text, before
# norm(), stripped repeatedly: 'Question 3:', 'Exercise 2', 'Q1.', 'Q1:Where' (Codex guard review 15: no space after
# the delimiter), '1.2', '(iv)', 'ii.', '(b)', 'B)', '3 ' before a CAPITAL or a blank, bullets. Unicode is NFKC-folded
# first (full-width 'Ｑ１：'). English label words only: no Tonga label word is assumed (never invent a word).
_LABEL_RAW = re.compile(r"^\s*(?:(?:question|exercise|activity|example|q|ex)\s*\d+[a-z]?\s*(?:[.):\-–]\s*|\s+)"
                        r"|\(\s*(?:\d{1,2}|[a-hA-H]|[ivxIVX]{1,4})\s*\)\s*"
                        r"|(?:\d{1,2}(?:\.\d{1,2})?|[a-hA-H]|[ivxIVX]{1,4})\s*[.):]\s+"
                        r"|\d{1,2}\.\d{1,2}\s+"
                        r"|(?-i:\d{1,2}\s+(?=[A-ZŊ_]))"
                        r"|[-–•*]\s+)", re.I)
# a mark annotation at the end, in brackets only: '(2)', '[8]', '(2 marks)', '(maaka 2)', or 'maaka 2' / '2 marks'
# written out. A bare number at the end is a QUANTITY and stays (Codex 14: 'Give me 2.' must not equal 'Give me 3.')
_MARK_RAW = re.compile(r"\s*(?:[(\[]\s*(?:maaka\s+)?\d{1,3}\s*(?:marks?)?\s*[)\]]|maaka\s+\d{1,3}|\d{1,3}\s+marks?)\s*[.]?\s*$",
                       re.I)


def held_norm(s):
    """the held-out identity of a line: norm() plus labels, option letters (A-E), blanks and mark annotations folded.
    Why (P1 self-check, 20 Oct 2026): the books repeat exam items under another label (Grade 8 p87 'a) Having sex with
    two people only.' is the held-out Grade 10 'B Having sex ...'); norm() keeps 'b ', so the duplicate passed the leak
    guard. 21 Oct (audit step 1, then Codex's guard review 14-15): labels in front (repeatedly, after NFKC), a blank of
    any length ('____' = '________'), and bracketed marks at the end. Numbers that are part of the sentence stay
    ('Give me 2.' != 'Give me 3.', 'Calculate 1+2.' != 'Calculate 1+7.'). Used ONLY as a leak-guard key (heldout_guard,
    this memory's guard, P1's held()); never a lookup key that serves text."""
    s = unicodedata.normalize("NFKC", s or "")
    for _ in range(4):
        t = _LABEL_RAW.sub("", s, count=1)
        if t == s:
            break
        s = t
    for _ in range(2):
        t = _MARK_RAW.sub("", s)
        if t == s:
            break
        s = t
    s = norm(re.sub(r"_{2,}", " _ ", s))
    s = re.sub(r"_+", "_", s)
    for _ in range(3):
        t = re.sub(r"^[a-e]\s+(?=\S)", "", s)          # an option letter left after punctuation folded ('B Having')
        if t == s:
            break
        s = t
    return s


def _raw_rows():
    """every memory row BEFORE any guard, plus the book held-out keys (Codex guard review 7: purity checks need the
    unfiltered sources; a guarded view hides what the memory once held)"""
    held_toi, held_en, out = set(), set(), []
    for f in sorted(glob.glob(os.path.join(VOC, "*pairs*.jsonl"))):
        b = os.path.basename(f)
        if "fidelity" in b or "SUPERSEDED" in b:
            continue
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            toi, en = r.get("toi") or "", r.get("en_guess") or ""
            if r.get("held_out") or r.get("exam"):
                held_toi.add(held_norm(toi))
                held_en.add(held_norm(en))
                continue
            if "machine" in b or r.get("confidence") not in ("high", "teacher"):
                continue
            if "_" in toi.replace("_", "") or "____" in toi or "[?" in toi or r.get("has_unclear_mark"):
                continue
            out.append({"en": en, "toi": toi, "src": f"book {r.get('book')} ({b}, page {r.get('page')}; English: "
                        f"{r.get('en_by', '?')[:40]}, confidence {r.get('confidence')})", "kind": "book"})
    for f in sorted(glob.glob(os.path.join(CT, "desk", "teacher_pairs_*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if not r.get("heldout") and r.get("en") and r.get("toi"):
                    out.append({"en": r["en"], "toi": r["toi"], "kind": "teacher", "who": r.get("teacher") or "?",
                                "src": f"teacher {r.get('teacher')} {r.get('date')} ({os.path.basename(f)})"})
    for f in sorted(glob.glob(os.path.join(CT, "desk", "teacher_corrections_v0*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("en") and r.get("toi"):
                    out.append({"en": r["en"], "toi": r["toi"], "kind": "teacher", "who": r.get("by") or os.path.basename(f),
                                "src": f"teacher correction ({os.path.basename(f)}, {r.get('src', '')})"})
    p = os.path.join(CT, "desk", "pairs.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("en") and r.get("chosen"):
                    out.append({"en": r["en"], "toi": r["chosen"], "kind": "teacher", "who": r.get("by") or "?", "src": f"teacher {r.get('by')} {r.get('date')} (desk/pairs.jsonl)"})
    if os.path.exists(THOMPSON):
        for r in csv.DictReader(open(THOMPSON, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            if r.get("en") and r.get("toi"):
                out.append({"en": r["en"], "toi": r["toi"], "kind": "thompson",
                            "src": f"Thompson 1979 p{r.get('page')} ({r.get('section')})"})
    _raw_rows.lessons = None
    if LESSONS:
        lr, _raw_rows.lessons = lesson_rows()
        out += lr
    held_toi.discard("")
    held_en.discard("")
    return out, held_toi, held_en


# ------------------------------------------------------------------------------------------------ lessons (24 Oct)
# a row whose ORIGIN cell says it is not settled stays out of a memory that serves verbatim (24 Oct): a form marked
# 'to check', a slip we removed (no longer the speaker's line), a spelling marked 'as given'. Stage 18 trains these at
# half weight (t24s_flag); a translation memory has no half weight. A CHECK in the Notes column is OUR doubt from
# before the speaker answered: the speaker's confirmation stands, the line goes in.
FLAGGED = re.compile(r"(?i)\bcheck\b|\bslip\b|\bloan\b|read as|as given")
# desk answer lines held by the desk's own note, quoted
HOLD = {"T09_03_answers.md #7b": "a place name the desk note asks a second speaker to confirm (23 Oct)"}
_SLASH = re.compile(r"([^\s/.,?!;:]+(?:/[^\s/.,?!;:]+)+)([.,?!;:]*)(?=\s|$)")


def cell_variants(toi):
    """([lines], why_skipped) for one teacher cell. A line is served verbatim, so only what the teacher wrote WHOLE:
    '(class: X)' / '(plural: X)' is a second line (T01 writes the class form beside the one-learner form); ONE slash
    group gives one line per alternative ('tupanga/tubamba': both his); a second slash group (a combination he never
    wrote as one sentence), any other bracket ('(least also: ...)', '(pe)', '(Amuswiilila.)': a note or an option)
    or a stray '/' skips the cell. Same reading as stage18_teacher_2324.variants for the cases it keeps."""
    toi = (toi or "").strip()
    extra = []
    m = re.search(r"\s*\((?:class|plural):\s*([^)]+)\)", toi)
    if m:
        extra.append(m.group(1).strip())
        toi = (toi[:m.start()] + toi[m.end():]).strip()
    if re.search(r"[()\[\]{}]", toi + " ".join(extra)):
        return [], "bracket (a note or an option)"
    groups = _SLASH.findall(toi)
    if len(groups) > 1:
        return [], "two slash groups"
    lines = [toi]
    if groups:
        g = groups[0][0]
        lines = [toi.replace(g, alt, 1) for alt in g.split("/") if alt]
        if toi[:1].isupper():          # 'Kolomba/kobuzya kotana ...': the second alternative starts the sentence too
            lines = [x[:1].upper() + x[1:] for x in lines]
    if any("/" in x for x in lines + extra):
        return [], "slash outside a word"
    return [re.sub(r"\s+", " ", x).strip() for x in lines + extra if x.strip()], None


def _desk_keys():
    """'lesson#row' -> 'batches_2026-10-24b/T09_13 #2': the desk message behind a box row (provenance only)"""
    m = {}
    for kf in sorted(glob.glob(os.path.join(DESK, "batches_2026-10-24*", "keys.json"))):
        try:
            for batch, b in json.load(open(kf, encoding="utf-8")).items():
                for it in b.get("items", []):
                    if "#" in str(it.get("key", "")):
                        m[it["key"]] = f"{os.path.basename(os.path.dirname(kf))}/{batch} #{it.get('n')}"
        except (OSError, ValueError, AttributeError):
            continue
    return m


def _box_files():
    return [f for d in BOX_DIRS for f in sorted(glob.glob(os.path.join(LIB, d, "*SAY_IT_IN_TONGA.md")))]


def _answer_files():
    return sorted(glob.glob(os.path.join(DESK, "batches_2026-10-24*", "*answers*.md")))


def lesson_rows():
    """(rows, report): the teacher-checked lesson Tonga (module docstring, 'lessons'). report counts what was taken
    and what was skipped, per reason, so the LOGBOOK numbers come from the code"""
    from collections import Counter
    rep, rows = Counter(), []
    keys = _desk_keys()

    def add(en, cell, who, src, srcset):
        lines, why = cell_variants(cell)
        if why:
            rep[f"skipped: {why}"] += 1
            return
        for j, toi in enumerate(lines, 1):
            rows.append({"en": en.strip(), "toi": toi, "kind": "teacher", "who": who, "srcset": srcset,
                         "src": src + (f" [line {j} of {len(lines)}]" if len(lines) > 1 else "")})
        rep[f"taken: {srcset}"] += 1
    for f in _box_files():
        rel = os.path.relpath(f, CT).replace("\\", "/")
        lesson = os.path.basename(f).split("_SAY_IT")[0]
        lkey = lesson.split("-")[0] if os.path.basename(os.path.dirname(f)) == "grade1" else lesson
        sec = ""
        for l in open(f, encoding="utf-8"):
            if l.startswith("## "):
                sec = l.strip()
                continue
            if not l.startswith("| ") or not (sec.startswith("## Classroom lines") or sec.startswith("## Added")):
                continue
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            if len(c) < 4 or c[0] == "#" or not c[0] or set(c[0]) <= set("-: "):
                continue
            origin = c[3]
            if not TEACHER_CODE.match(origin):
                rep["not a teacher row (D / B / L / none)"] += 1
                continue
            en, toi = c[1], c[2]
            if FLAGGED.search(origin):
                rep["held: flagged in its Origin cell (check / slip / loan / as given)"] += 1
                continue
            if "frame" in origin.lower() or not en or not toi or toi.startswith("(dropped") \
                    or "{" in en + toi or "[" in en:
                rep["skipped: frame / dropped / slot / empty"] += 1
                continue
            desk = keys.get(f"{lkey}#{c[0]}") or keys.get(f"{lesson}#{c[0]}") or keys.get(f"{lesson.split('-')[0]}#{c[0]}")
            add(en, toi, origin[:3], f"teacher {origin[:3]} lesson box {rel}#{c[0]} ({origin[:60]}"
                + (f"; desk {desk}" if desk else "") + ")", "lessons")
    try:
        fx = json.load(open(FIXED, encoding="utf-8"))
    except (OSError, ValueError):
        fx = {}
    for part in ("lines", "buttons"):
        for k, v in (fx.get(part) or {}).items():
            codes = [o for o in v.get("origin", []) if TEACHER_CODE.match(str(o))]
            en, toi = v.get("en") or "", v.get("toi") or ""
            base = f"core_fixed_lines.json {part}:{k}"
            if v.get("status") == "served" and part == "buttons" and len(en.split()) < 2:
                # a one-word button is a UI label whose Tonga agrees with what it labels (the word for 'Both'
                # agrees with the two languages, class 8/10); served as a translation of 'Both.' anywhere it would be wrong for
                # people (bobilo). Two-word buttons ('No, thank you', 'Lesson help') go in.
                rep["held: one-word button (a UI label)"] += 1
            elif v.get("status") == "served" and codes and en and toi and "{" not in en + toi:
                add(en, toi, "+".join(codes), f"teacher {'+'.join(codes)} {base} (status served)", "fixed")
            elif v.get("status") == "served":
                rep["skipped: fixed line slot / no Tonga"] += 1
            else:
                rep[f"fixed line not served ({v.get('status')})"] += 1
            if v.get("variant_T09") and en and "{" not in en + v["variant_T09"]:
                add(en, v["variant_T09"], "T09", f"teacher T09 {base} variant_T09", "fixed")
    for f in _answer_files():
        b = os.path.basename(f)
        who = "T01" if b.startswith("T01") else "T09" if b.startswith("T09") else None   # private file names by code
        if not who:
            continue
        for l in open(f, encoding="utf-8"):
            m = re.match(r"^\s*(\d+[a-z]?)\.\s+(.+?)\s+>\s+(.+?)\s*$", l)
            if m and len(m.group(2).split()) >= 3 and len(m.group(3).split()) >= 2:
                if f"{b} #{m.group(1)}" in HOLD:
                    rep["held: " + HOLD[f"{b} #{m.group(1)}"][:60]] += 1
                    continue
                add(m.group(2), m.group(3), who,
                    f"teacher {who} desk/{os.path.basename(os.path.dirname(f))}/{b} #{m.group(1)}", "desk_answers")
    return rows, dict(rep)


# ------------------------------------------------------------------------------------------------ held-out (24 Oct)
def _near_ok(a, b, rmin, wmin, share=0.6):
    """the near test of lookup() on two norm() keys: shared words, length, letter ratio AND word ratio"""
    ta, tb = a.split(), b.split()
    if a == b:
        return bool(a), 1.0
    if not ta or not tb:
        return False, 0.0
    if len(set(ta) & set(tb)) < max(1, share * len(ta)) or abs(len(a) - len(b)) > 0.3 * max(len(a), 1) + 3:
        return False, 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    rw = difflib.SequenceMatcher(None, ta, tb).ratio()
    return (r >= rmin and rw >= wmin), round(r, 3)


class HeldIndex:
    """held-out lines for the serve-time guard: hit(s) -> the set's name when s equals (held_norm) or is near (GUARD_NEAR
    / GUARD_WORDS) a held line, else None. Lines of 1-2 words match exactly only (a near test on them is noise)."""
    def __init__(self, lines):
        self.keys, self.idx = {}, {}
        for text, name in lines:
            k = held_norm(text)
            if not k:
                continue
            self.keys.setdefault(k, name)
            for w in set(k.split()):
                self.idx.setdefault(w, set()).add(k)

    def __len__(self):
        return len(self.keys)

    def hit(self, s):
        k = held_norm(s or "")
        if not k:
            return None
        if k in self.keys:
            return self.keys[k]
        toks = k.split()
        if len(toks) < 3:
            return None
        cand = set()
        for w in set(toks):
            cand |= self.idx.get(w, set())
        for kk in cand:
            if len(kk.split()) >= 3 and (_near_ok(k, kk, GUARD_NEAR, GUARD_WORDS, 0.5)[0]
                                         or _near_ok(kk, k, GUARD_NEAR, GUARD_WORDS, 0.5)[0]):
                return self.keys[kk] + " (near)"
        return None


def frozen18_lines():
    """[(text, set)]: English, Tonga and every alternative of the stage 18 frozen teacher test (44) and its dev slice.
    A missing file is an ERROR, not an empty guard (heldout_guard's rule, Codex guard review finding 3)."""
    out = []
    for p in FROZEN18:
        if not os.path.exists(p):
            raise RuntimeError(f"frozen set missing: {p}; the memory would serve its lines")
        name = os.path.basename(p).replace(".jsonl", "")
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                for s in [r.get("en"), r.get("toi")] + list(r.get("alternatives") or []):
                    if isinstance(s, str) and s.strip():
                        out.append((s, name))
    return out


def harness_files(base=None):
    """the harness logs that can carry a held-out request (top level of pilot_logs only: the shadow / simulation folders
    below it are scripted, not teachers)"""
    d = os.path.join(base or HARNESS_DATA, "pilot_logs")
    return sorted(glob.glob(os.path.join(d, "*.jsonl")))


def harness_held_lines(base=None):
    """[(text, set)]: the request text of every harness record with held_out true (the first 40 real requests per
    teacher, adapter config held_out_first; labels.py / flywheel_log.py): day logs inbound.text, labels 'request',
    flywheel gap 'text'. Pupil-level text is already null there and stays out."""
    out = []
    for p in harness_files(base):
        try:
            fh = open(p, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for l in fh:
                if '"held_out": true' not in l:
                    continue
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r.get("held_out") is not True:
                    continue
                for s in ((r.get("inbound") or {}).get("text") if isinstance(r.get("inbound"), dict) else None,
                          r.get("request"), r.get("text")):
                    if isinstance(s, str) and s.strip():
                        out.append((s, "harness held-out request"))
    return out


_SH = {}


def serve_held(base=None):
    """HeldIndex of the stage 18 frozen test + dev and the harness's held-out requests, rebuilt when one of their
    files changes (stat only; its own stamp, so the translation memory is not rebuilt for a WhatsApp message)"""
    st = _stamp_of(FROZEN18 + harness_files(base)) + str(base)
    if _SH.get("stamp") != st:
        _SH.clear()
        _SH.update(stamp=st, idx=HeldIndex(frozen18_lines() + harness_held_lines(base)))
    return _SH["idx"]


def _rows():
    out, held_toi, held_en = _raw_rows()
    out = [r for r in out if held_norm(r["toi"]) not in held_toi and held_norm(r["en"]) not in held_en]
    # 21 Oct 2026 (audit step 1, guard sweep): the book held-out rows above were the memory's only guard, so it still
    # served 101 rows that equal another frozen set (Thompson held-out 36, children + radio 30, the new-form test 29,
    # heldout_books 4, 6-gram 2) and 13 relabelled repeats. Now every row also passes the training builds' guard
    # (train/heldout_guard.py on the stage 14 run: every frozen test, the CDC held-out modules, the anchor files)
    G = _guard()
    out = [r for r in out if not G.hit(r["toi"], r["en"])]
    # 24 Oct 2026: the stage 18 frozen teacher test + dev slice, exact OR near, either side (the boxes hold them)
    F = HeldIndex(frozen18_lines())
    keep = [r for r in out if not (F.hit(r["en"]) or F.hit(r["toi"]))]
    _rows.dropped18 = len(out) - len(keep)
    return keep, len(held_toi)


_G = {}
GUARD_RUN = os.path.join(CT, "runs", "2026-10-18_stage14")


def _hg():
    sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "train"))
    import heldout_guard as HG
    return HG


def _guard():
    """the training builds' guard, rebuilt whenever one of ITS files changes (Codex guard review 2: a cached guard
    survived reload(), so an anchor frozen after start-up was not guarded); its files are part of stamp() too"""
    st = _stamp_of(_hg().dependencies(GUARD_RUN))
    if _G.get("stamp") != st:
        _G.clear()
        _G.update(stamp=st, G=_hg().load(GUARD_RUN))
    return _G["G"]


def sources():
    """every file the memory reads (for the change check and the snapshot stamp)"""
    fs = [f for f in glob.glob(os.path.join(VOC, "*pairs*.jsonl"))]
    fs += glob.glob(os.path.join(CT, "desk", "teacher_pairs_*.jsonl"))
    fs += glob.glob(os.path.join(CT, "desk", "teacher_corrections_v0*.jsonl"))
    fs += [os.path.join(CT, "desk", "pairs.jsonl"), THOMPSON]
    fs += _hg().dependencies(GUARD_RUN)            # the guard's frozen sets and anchors (Codex guard review 2)
    fs += FROZEN18 + [p + ".sha256" for p in FROZEN18]        # 24 Oct: the stage 18 frozen test + dev (guard)
    if LESSONS:                                    # 24 Oct: the lesson boxes, the fixed lines, the desk answers
        fs += _box_files() + [FIXED] + _answer_files() + glob.glob(os.path.join(DESK, "batches_2026-10-24*", "keys.json"))
    return sorted(set(f for f in fs if os.path.exists(f)))


def stamp():
    """PER SOURCE identity, nanosecond mtime and size (Codex build review, finding 15: a count + max mtime in seconds +
    total size missed same-length edits and changes to an older file); hashed into one short stamp. The lessons switch
    is part of it (24 Oct), so flipping LESSONS rebuilds the snapshot on the next lookup."""
    return _stamp_of(sources()) + ("" if LESSONS else "-nolessons")


def _stamp_of(files):
    import hashlib
    parts = []
    for f in sorted(set(files)):
        try:
            st = os.stat(f)
            parts.append(f"{f}|{st.st_dev}:{st.st_ino}|{st.st_mtime_ns}|{st.st_size}")
        except OSError:
            parts.append(f"{f}|gone")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _build():
    """a NEW snapshot, built apart from the live one; raises on a malformed source line"""
    rows, nheld = _rows()
    by, idx = {}, {}
    by_toi, idx_toi = {}, {}            # 24 Oct: the Tonga key, for near() toi>en
    from collections import Counter
    kinds = Counter()
    for r in rows:
        kinds[r.get("srcset") or r["kind"]] += 1
        k = norm(r["en"])
        if not k:
            continue
        by.setdefault(k, []).append(r)
        kt = norm(r["toi"])
        if kt:
            by_toi.setdefault(kt, []).append(r)
    for k in by:
        for w in set(k.split()):
            idx.setdefault(w, set()).add(k)
    for k in by_toi:
        for w in set(k.split()):
            idx_toi.setdefault(w, set()).add(k)
    return dict(by=by, idx=idx, by_toi=by_toi, idx_toi=idx_toi, n=len(rows), held=nheld, kinds=dict(kinds),
                dropped18=getattr(_rows, "dropped18", None), lessons=getattr(_raw_rows, "lessons", None))


KEEP_LAST_GOOD = True          # off only in h3_tests/mutation_check.py


def reload():
    """force a rebuild; the last good snapshot AND its stamp stay if the rebuild fails (R6)"""
    if _mem:
        _mem["forced"] = True
    _G.clear()
    m = memory()
    return {"pairs": m["n"], "stamp": m.get("stamp"), "reload_error": m.get("reload_error")}


def memory():
    """the snapshot, rebuilt when a source changed (finding 12): the replacement is built and validated apart, the
    sources must be unchanged from before the read to after it, and on any failure the LAST GOOD snapshot stays in
    service (its reload_error says why); with no good snapshot yet the error is raised"""
    st = stamp()
    if _mem and _mem.get("stamp") == st and not _mem.get("forced"):
        return _mem
    try:
        new = _build()
        if stamp() != st:
            raise RuntimeError("a memory source changed during the read")
    except Exception as e:  # noqa: BLE001
        if KEEP_LAST_GOOD and _mem and _mem.get("by") is not None:
            _mem["reload_error"] = repr(e)[:200]          # the last good snapshot keeps its own stamp
            _mem["forced"] = False
            return _mem
        raise
    _mem.clear()
    _mem.update(new, stamp=st, reload_error=None, forced=False)
    return _mem


def lookup(en, n_near=3):
    """{'exact': [rows], 'near': [(ratio, row, differs)]}; rows carry en, toi, src, kind"""
    import h2_checks as C
    m = memory()
    k = norm(en)
    exact = m["by"].get(k, [])
    toks = k.split()
    cand = {}
    for w in toks:
        for kk in m["idx"].get(w, ()):
            cand[kk] = cand.get(kk, 0) + 1
    near = []
    for kk, shared in cand.items():
        if kk == k or shared < max(1, 0.6 * len(toks)) or abs(len(kk) - len(k)) > 0.3 * max(len(k), 1) + 3:
            continue
        r = difflib.SequenceMatcher(None, k, kk).ratio()
        rw = difflib.SequenceMatcher(None, toks, kk.split()).ratio()   # by words too: 'angry' / 'hungry' share letters
        if r >= NEAR_MIN and rw >= NEAR_WORDS:
            row = m["by"][kk][0]
            differs = C.en_neg(en) != C.en_neg(row["en"]) or \
                set(C.T.numbers_en(en)) != set(C.T.numbers_en(row["en"]))
            near.append((round(r, 3), row, differs))
    near.sort(key=lambda x: -x[0])
    # teacher and Thompson before books (a speaker's own line before a guessed English)
    order = {"teacher": 0, "thompson": 1, "book": 2}
    srcs = {}
    for r in exact:                                   # independent sources per written Tonga (normalised, no folding)
        srcs.setdefault(norm(r["toi"]), set()).add(("teacher:" + str(r.get("who"))) if r["kind"] == "teacher" else r["kind"])
    uniq = {}
    for r in sorted(exact, key=lambda r: order.get(r["kind"], 3)):
        if r["toi"].strip() in uniq:
            continue                                  # the same Tonga from several pages: once, first source
        r = dict(r)
        r["confidence"] = len(srcs.get(norm(r["toi"]), ()))
        uniq[r["toi"].strip()] = r
    rows = list(uniq.values())
    for r in rows:
        r["contradicts"] = [{"toi": x["toi"], "confidence": x["confidence"]} for x in rows if norm(x["toi"]) != norm(r["toi"])]
    rows.sort(key=lambda r: -r["confidence"])         # stable: ties keep teacher, Thompson, book
    return {"exact": rows, "near": near[:n_near]}


# ------------------------------------------------------------------------------------------------ near, both ways
EN_INVARIANT = {"a", "an", "the", "please"}          # differences the Tonga does not carry (24 Oct)
TOI_INVARIANT = {"sena", "hena", "na", "ino", "pe"}  # question markers and the negative's 'pe': same polarity checked


def _toi_fold(w):
    """a typing variant, not a different word: case, hyphen, apostrophe, ŋ / ng' / ng', ch / c. Doubled letters and long
    vowels are NOT folded: kkala 'sit' / kala 'nail', boola 'come' / bola 'rot' (tonga_structure 2.8)"""
    w = (w or "").lower().replace("’", "'").replace("ŋ", "ng'").replace("-", "")
    return w.replace("ch", "c")


def strict_near(a, b, direction):
    """True when the words that differ between two lines (source side) are only ones the target does not carry:
    English articles / please (en>toi), Tonga question markers / pe or a typing variant (toi>en). Why: a near match
    can differ by exactly the word that matters ('Put 4 stones here' / 'Put 4 sticks here' is 0.9 alike by letters),
    the readers' cosine cannot tell stones from sticks, and a near line that wins a check-driven switch SILENCES the
    incumbent's alarm, so a wrong content word would reach a class unseen (24 Oct 2026)"""
    from collections import Counter
    if direction == "en>toi":
        ta, tb = Counter(norm(a).split()), Counter(norm(b).split())
        return all(w in EN_INVARIANT for w in (ta - tb) + (tb - ta))
    fa = [_toi_fold(w) for w in norm(a).split()]
    fb = [_toi_fold(w) for w in norm(b).split()]
    if "".join(fa) == "".join(fb):
        return True
    ta, tb = Counter(fa), Counter(fb)
    return all(w in TOI_INVARIANT for w in (ta - tb) + (tb - ta))


def _toi_neg(s):
    try:
        import h3_detect as D
        return D.toi_neg3(s)
    except Exception:  # noqa: BLE001  (h3_detect needs the grammar engine; h2_checks alone is the older reading)
        import h2_checks as C
        return C.toi_neg(s)


def near(text, direction="en>toi", n=12):
    """[{ratio, rword, row, differs, strict, target}] near rows of `text` (en>toi: by English; toi>en: by Tonga),
    best first; every row of a near key (teacher, Thompson, book order), the exact key excluded. Candidates for a
    person or for translator_service's rerank, NEVER an answer by themselves (module docstring, 'NEAR')."""
    import h2_checks as C
    m = memory()
    k = norm(text)
    if not k:
        return []
    by, idx = (m["by"], m["idx"]) if direction == "en>toi" else (m["by_toi"], m["idx_toi"])
    toks = k.split()
    cand = {}
    for w in toks:
        for kk in idx.get(w, ()):
            cand[kk] = cand.get(kk, 0) + 1
    order = {"teacher": 0, "thompson": 1, "book": 2}
    if direction == "en>toi":
        neg0, num0 = C.en_neg(text), set(C.T.numbers_en(text))
    else:
        neg0, num0 = _toi_neg(text), set(C.T.numbers_toi(text))
    out = []
    for kk, shared in cand.items():
        if kk == k or shared < max(1, 0.6 * len(toks)) or abs(len(kk) - len(k)) > 0.3 * max(len(k), 1) + 3:
            continue
        r = difflib.SequenceMatcher(None, k, kk).ratio()
        rw = difflib.SequenceMatcher(None, toks, kk.split()).ratio()
        if r < NEAR_MIN or rw < NEAR_WORDS:
            continue
        for row in sorted(by[kk], key=lambda x: order.get(x.get("kind"), 3)):
            if direction == "en>toi":
                key_side, target = row["en"], row["toi"]
                differs = C.en_neg(key_side) != neg0 or set(C.T.numbers_en(key_side)) != num0
            else:
                key_side, target = row["toi"], row["en"]
                differs = _toi_neg(key_side) != neg0 or set(C.T.numbers_toi(key_side)) != num0
            out.append({"ratio": round(r, 3), "rword": round(rw, 3), "row": row, "differs": differs,
                        "strict": strict_near(text, key_side, direction), "target": target})
    out.sort(key=lambda x: (-x["ratio"], -x["rword"], order.get(x["row"].get("kind"), 3)))
    return out[:n]


if __name__ == "__main__":
    m = memory()
    print(f"memory: {m['n']} pairs, {len(m['by'])} distinct English; {m['held']} held-out Tonga lines guarded; "
          f"by source {m.get('kinds')}; stage 18 test/dev rows dropped {m.get('dropped18')}")
    print(json.dumps(lookup(sys.argv[1]), ensure_ascii=False, indent=1))
