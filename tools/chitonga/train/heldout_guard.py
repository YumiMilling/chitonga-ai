"""One held-out guard for every training build (18 Oct 2026). Before stage 14, the paper session asked that every
frozen test set stay out of training (Monday's native-speaker test is drawn only from them). A check found 81 rows
leaking into stage 14, mostly teachers' translations of held-out items: the old guards compared the TONGA only, but a
teacher's Tonga often differs from the reference while the ENGLISH is the same sentence, which contaminates the
English > Tonga test. This guard matches on either side.

Sets: the translators' frozen test (150, English), the Hopgood held-outs (181, English), every book pair marked
held_out / exam (book exams, readers' last lessons, the health set: WHO/UNICEF poster, Grade 10 PDF 49-50), the
children's books + radio set, the Thompson and teacher held-outs, heldout_books.txt (Grade 8 guard etc.), every
*_heldout.toi.txt, and (19 Oct 2026) the 40-item new-form test llm/chat_v10_tests/new_forms_40.jsonl (hash-checked), and (19 Oct 2026)
the three conversation tests llm/chat_v10_tests/{multiturn,comprehension,everyday}_test.jsonl (hash-checked; each item's
`guard` list), the education test edu_test.jsonl in the same folder (same glob), and the two CDC Literature-in-Chitonga
Form 1 modules, terms 2 and 3 (MOE_HELD_OUT: in no training set, 19 Oct 2026). A row goes if its Tonga or English equals a held-out line (normalised), or if its Tonga has 8+ words
and shares a 6-word run with held-out Tonga.
21 Oct 2026 (audit decisions step 1): also if either side equals a held-out line under h2_memory.held_norm (labels,
numbering, option letters, blanks and end marks folded; 2+ tokens left): the books repeat exam items under other labels.
which() names such a hit 'relabelled: <set>'. And every translator anchor file (candidates included) is a held-out set.
21 Oct 2026 (later): every text file under zamai-corpus-raw/chitonga_mono/_test_only/ is a held-out set
('_test_only: <path>'); the Cimbonimboni test excerpt had been in the copy pool since stage 13 (load_test_only).
21 Oct 2026 (night): the 500-line translator diagnosis set and its document sections (load_diag500), frozen.
22 Oct 2026 (the Jesuit, writing check llm/panel/writing_check_2026-10-22_a050_vs_claude/): NOT UNSEEN, whatever this
guard says today. (1) thompson_heldout.tsv: 33 of its 77 rows were in earlier training (30 in stages 11 / 12 / 12b /
13, so in R = s13 soup and every merge built on it; 10 in stage 14; 7 even in stages 16 / 17 through a bracketed
alternative such as 'Kamubuka (Amubuke.)': the guard keys the whole bracketed string, not each alternative). List:
internal/chitonga_train/runs/2026-10-22_stage17_data/thompson_heldout_seen_in_training.json. (2) The JCTR document
jctr:adventiv05.htm of the stage 16 held-out documents (_test_only/stage16_heldout_docs): all 57 of its lines checked
are in stage 4 train, 14 in stage 13 tasks, 35 in stage 15 (list: writing_check_2026-10-22_a050_vs_claude/
doc_lines_seen_in_training.json), so R and the merges saw them. Any test that needs lines unseen by R must drop both.
Sweep of the existing files: guard_sweep.py; tests: unit_guard.py, unit_guard_diag500.py.

    import heldout_guard as HG; G = HG.load(run_dir); keep = [x for x in rows if not G.hit(toi, en)]
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_stage8b import g6, norm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "internal", "chitonga_train"))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "zamai-corpus-raw", "chitonga_mono", "new_books_2026-10-16"))


sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm")))
from h2_memory import held_norm  # noqa: E402

RELABEL_MIN = 2   # tokens left after the labels go; shorter lines are matched by the exact norm only
SKEL_MIN_WORDS = 3  # Tonga lines this long or longer also get a typing-skeleton key


class GuardError(RuntimeError):
    """a guard invariant failed (missing or edited frozen set). An exception, not an assert: python -O keeps it"""


def rnorm(s):
    """the relabelled-repeat key (21 Oct 2026, audit decisions step 1): h2_memory.held_norm, i.e. labels, numbering,
    option letters, blanks and end marks folded. Why: the books repeat exam items under another label ('a) Having sex
    ...' in Grade 8 = the held-out Grade 10 'B Having sex ...'); the exact norm() kept the label, so 8 of the 114
    diagnosis lines and 1 anchor candidate were in the stage 13 training data (P1 self-check, 20 Oct). Returns '' for
    a line too short to key safely (a 1-token key would strip 'Yes' or 'Mulimo' from every build)."""
    k = held_norm(s or "")
    return k if len(k.split()) >= RELABEL_MIN else ""


def skeleton(s):
    """the typing-skeleton key of a TONGA line (21 Oct 2026, Codex guard review finding 4): every typing difference the
    anchor's bad-input stratum makes, and the ZV speakers' spellings of spelling.tsv (REGISTER ZV), folded at once:
    case, punctuation and spaces (a prefix written apart), ng' / ŋ, ch for c, a doubled letter written single (long
    vowels, bb / kk). 'Bika buku lyako mu cikolo lelo.' = 'Bikka bbuku lyako mucikolo lelo.'. Only lines of 3+ words
    are keyed (a short line's skeleton would collide with ordinary words). A guard key only, never a lookup key."""
    import unicodedata
    if len((s or "").split()) < SKEL_MIN_WORDS:
        return ""
    t = unicodedata.normalize("NFKC", s).lower().replace("ŋ", "ng").replace("ng'", "ng").replace("ng’", "ng")
    t = re.sub(r"[^a-z0-9]", "", t).replace("ch", "c")     # digits kept: Mutwe 4 is not Mutwe 1
    t = re.sub(r"(.)\1+", r"\1", t)
    return t if len(t) >= 8 else ""                 # measured on the anchor: a 3-word gap line folds to 11


class Guard:
    """sets: exact norm() per held-out set; rsets: the same lines under rnorm() (relabelled repeats); ssets: Tonga lines
    under skeleton() (typing variants); g6: 6-word runs of Tonga. inventory: what load() read (name, path, items, sha)"""
    def __init__(self):
        self.sets, self.rsets, self.ssets, self.g6, self.inventory = {}, {}, {}, set(), []

    def add(self, name, s, grams=True):
        """grams=True marks a TONGA string: 6-word runs and the typing skeleton are kept for it"""
        if s and s.strip():
            self.sets.setdefault(name, set()).add(norm(s))
            k = rnorm(s)
            if k:
                self.rsets.setdefault(name, set()).add(k)
            if grams:
                self.g6 |= g6(s)
                sk = skeleton(s)
                if sk:
                    self.ssets.setdefault(name, set()).add(sk)

    def hit(self, toi=None, en=None):
        return self.which(toi, en) is not None

    def which(self, toi=None, en=None):
        """the first reason a row is held out: exact, then 'relabelled: <set>', then 'typing variant: <set>' (Tonga),
        then '6-gram'; None = clean"""
        r = self.reasons(toi, en, first=True)
        return r[0] if r else None

    def reasons(self, toi=None, en=None, first=False):
        """every reason (for the sweep: old predicates and new ones apart, Codex finding 20)"""
        out = []
        for side in (toi, en):
            if side:
                n = norm(side)
                for k, v in self.sets.items():
                    if n in v:
                        out.append(k)
                        if first:
                            return out
        for side in (toi, en):
            if side:
                r = rnorm(side)
                if r:
                    for k, v in self.rsets.items():
                        if r in v:
                            out.append("relabelled: " + k)
                            if first:
                                return out
        if toi:
            sk = skeleton(toi)
            if sk:
                for k, v in self.ssets.items():
                    if sk in v:
                        out.append("typing variant: " + k)
                        if first:
                            return out
        if toi and len(toi.split()) >= 8 and g6(toi) & self.g6:
            out.append("6-gram")
        return out

    def note(self, name, path, items):
        import hashlib
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if path and os.path.isfile(path) else None
        self.inventory.append({"set": name, "path": os.path.relpath(path, CT) if path else None, "items": items,
                               "sha16": sha})

    def inventory_sha(self):
        import hashlib
        return hashlib.sha256(json.dumps(sorted((x["set"], x["path"] or "", x["items"], x["sha16"] or "")
                                                for x in self.inventory)).encode()).hexdigest()[:16]


def _need(path, what):
    if not os.path.exists(path):
        raise GuardError(f"frozen set missing: {what} ({path}); a guard without it would let its items into training")
    return path


def _sidecar(path):
    """a .sha256 beside a frozen file must match it (Codex finding 5: an edited frozen file must stop every build)"""
    import hashlib
    sc = path + ".sha256"
    if os.path.exists(sc):
        want = open(sc, encoding="utf-8").read().split()[0]
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != want:
            raise GuardError(os.path.basename(path) + " changed after it was frozen (sha256 differs from its .sha256)")


def dependencies(run_dir):
    """every file load() reads (for the memory's change stamp, Codex finding 2, and for the inventory)"""
    fs = [os.path.join(run_dir, f) for f in RUN_FILES]
    fs += sorted(glob.glob(os.path.join(CT, "vocab", "new_books_2026-10-16", "*pairs*.jsonl")))
    ec = os.path.join(CT, "llm", "eval_children")
    fs += [os.path.join(ec, "items.tsv")] + [os.path.join(ec, f) for f in ("items.tsv.sha256", "corrections.tsv")
                                             if os.path.exists(os.path.join(ec, f))]
    fs += sorted(glob.glob(os.path.join(RAW, "*", "*_heldout.toi.txt")) + glob.glob(os.path.join(RAW, "*", "*", "*_heldout.toi.txt")))
    nf = os.path.join(CT, "llm", "chat_v10_tests", "new_forms_40.jsonl")
    fs += [nf, nf + ".sha256"]
    for tf in sorted(glob.glob(os.path.join(CT, "llm", "chat_v10_tests", "*_test.jsonl"))):
        fs += [tf] + ([tf + ".sha256"] if os.path.exists(tf + ".sha256") else [])
    fs += [os.path.join(MOE, stem, stem + ".pages.txt") for stem in MOE_HELD_OUT]
    for f in ANCHOR_FILES:
        p = os.path.join(CT, "llm", "panel", f)
        fs += [p] + ([p + ".sha256"] if os.path.exists(p + ".sha256") else [])
    if os.path.isdir(TEST_ONLY):
        fs += test_only_files()[0]
    for f in DIAG500_FILES:
        p = os.path.join(DIAG500, f)
        fs += [p, p + ".sha256"]
    return fs


# The INVENTORY (21 Oct 2026, Codex guard review finding 3): every frozen set the guard must hold. A missing file used
# to drop its items from protection in silence (a run folder without test_hopgood.en.txt; with every input absent the
# guard was empty). load() now refuses: GuardError names the missing set. Optional: anchor_translator.jsonl until it
# is frozen (once its .sha256 exists, the file is required and must match it).
RUN_FILES = ("test.en.txt", "test_hopgood.en.txt", "thompson_heldout.tsv", "teacher_heldout.tsv", "heldout_books.txt")
CONV_TESTS = ("comprehension_test.jsonl", "edu_test.jsonl", "everyday_test.jsonl", "multiturn_test.jsonl")
MIN_HELDOUT_TOI_FILES = 2
REQUIRED_ANCHORS = ("translator_anchor_candidates.jsonl", "translator_anchor_candidates_v2.jsonl")


def load(run_dir):
    G = Guard()
    for f, name in (("test.en.txt", "translators frozen test"), ("test_hopgood.en.txt", "Hopgood held-outs")):
        p = _need(os.path.join(run_dir, f), name)
        n = 0
        for l in open(p, encoding="utf-8"):
            G.add(name, l, grams=False)
            n += 1
        G.note(name, p, n)
    books = glob.glob(os.path.join(CT, "vocab", "new_books_2026-10-16", "*pairs*.jsonl"))
    nb = 0
    for f in books:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l) if l.strip() else {}
            if r.get("held_out") or r.get("exam"):
                G.add("book held-out", r.get("toi")); G.add("book held-out (en)", r.get("en_guess"), grams=False)
                nb += 1
    if not nb:
        raise GuardError("frozen set missing: book held-out rows (vocab/new_books_2026-10-16/*pairs*.jsonl)")
    G.note("book held-out", None, nb)
    p = _need(os.path.join(CT, "llm", "eval_children", "items.tsv"), "children+radio")
    _sidecar(p)
    # 22 Oct 2026: through children_items, so a dated correction (llm/eval_children/corrections.tsv, e.g. ntk03's
    # English) is guarded in BOTH forms: the frozen text and the corrected one
    import children_items as CI
    n = 0
    for _id, f, t in CI.variants():
        if f == "toi":
            G.add("children+radio", t)
            n += 1
        else:
            G.add("children+radio (en)", t, grams=False)
    G.note("children+radio", p, n)
    for f in ("thompson_heldout.tsv", "teacher_heldout.tsv"):
        p = _need(os.path.join(run_dir, f), f)
        n = 0
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            G.add(f, r.get("toi")); G.add(f + " (en)", r.get("en"), grams=False)
            n += 1
        G.note(f, p, n)
    p = _need(os.path.join(run_dir, "heldout_books.txt"), "heldout_books")
    paras = open(p, encoding="utf-8").read().split("\n\n")
    for para in paras:
        G.add("heldout_books", para)
    G.note("heldout_books", p, len(paras))
    ht = glob.glob(os.path.join(RAW, "*", "*_heldout.toi.txt")) + glob.glob(os.path.join(RAW, "*", "*", "*_heldout.toi.txt"))
    if len(ht) < MIN_HELDOUT_TOI_FILES:
        raise GuardError(f"frozen set missing: *_heldout.toi.txt under {RAW} ({len(ht)} found, {MIN_HELDOUT_TOI_FILES} expected)")
    for f in ht:
        paras = open(f, encoding="utf-8").read().split("\n\n")
        for para in paras:
            G.add("_heldout.toi.txt", para)
        G.note("_heldout.toi.txt", f, len(paras))
    # 19 Oct 2026: the 40-item new-form test (freeze_new_forms_40.py), frozen before the run 2 synthetic material; its
    # sha256 is checked so a silently edited test set stops every build instead of guarding the wrong lines
    nf = _need(os.path.join(CT, "llm", "chat_v10_tests", "new_forms_40.jsonl"), "new-form test")
    _need(nf + ".sha256", "new-form test sha256")
    _sidecar(nf)
    n = 0
    for l in open(nf, encoding="utf-8").read().splitlines():
        r = json.loads(l)
        G.add("new-form test (19 Oct)", r["en"], grams=False)
        for a in r["accept"]:
            G.add("new-form test (19 Oct)", a)
        n += 1
    G.note("new-form test (19 Oct)", nf, n)
    # 19 Oct 2026: the conversation test sets of chat run 2 (multiturn_test, comprehension_test, everyday_test, built
    # before training) and the education test. A record with a `guard` list (freeze_conv_tests.py) is guarded by that
    # list; any other record by every string of 2+ words in it (except bookkeeping fields); a .sha256 beside a file is
    # checked like new_forms_40's
    for c in CONV_TESTS:
        _need(os.path.join(CT, "llm", "chat_v10_tests", c), "conversation / education test " + c)
    for tf in sorted(glob.glob(os.path.join(CT, "llm", "chat_v10_tests", "*_test.jsonl"))):
        _sidecar(tf)
        name = "conversation test: " + os.path.basename(tf)[:-6]
        n = 0
        for l in open(tf, encoding="utf-8").read().splitlines():
            if l.strip():
                r = json.loads(l)
                n += 1
                if "guard" in r:
                    # freeze_conv_tests.py (the Jesuit, 19 Oct) gives each item its own list: Tonga of 4+ words (exact
                    # line + 6-word runs), English of 5+ words (exact line). Formula lines and Thompson's trained-page
                    # lines are left out ON PURPOSE: every 2+ word string would strip "Mwapona buti?", "Kamucaala kabotu."
                    # and "Twalumba kapati." from every later training build
                    for t in r["guard"]["toi"]:
                        G.add(name, t)
                    for t in r["guard"]["en"]:
                        G.add(name + " (en)", t, grams=False)
                else:
                    for t in _strings(r):
                        G.add(name, t)
        G.note(name, tf, n)
    # 19 Oct 2026 (the Jesuit, education test set): the two CDC Literature-in-Chitonga Form 1 modules (terms 2 and 3,
    # 2025) are in NO training set (absent from Liseli, which carries the other seven CDC Chitonga modules, and from every
    # build); they stay held out as test text. Every page line of 4+ words is guarded (exact + 6-word runs). The other
    # seven modules fed the NLLB copy task (build_stage2.py 'liseli:moe') and are NOT held out.
    for stem in MOE_HELD_OUT:
        p = _need(os.path.join(MOE, stem, stem + ".pages.txt"), "CDC module held out " + stem)
        n = 0
        for line in open(p, encoding="utf-8"):
            if len(line.split()) >= 4 and not line.startswith("=====PAGE"):
                G.add("CDC module held out: " + stem, line)
                n += 1
        G.note("CDC module held out: " + stem, p, n)
    # 21 Oct 2026 (audit decisions: "heldout_guard must carry the final anchor file before stage 15 is built"): every
    # translator anchor file, candidates included (an unchosen candidate costs one training line; a chosen one found in
    # training costs the anchor). Both sides; Tonga with 6-word runs and the typing skeleton; for a bad-input item also
    # the real line it was typed from (`source_real`, `reference`). The frozen anchor must match its .sha256.
    for f in ANCHOR_FILES:
        p = os.path.join(CT, "llm", "panel", f)
        if f in REQUIRED_ANCHORS:
            _need(p, "translator anchor " + f)
        elif not os.path.exists(p):
            if os.path.exists(p + ".sha256"):
                raise GuardError(f"frozen set missing: {f} (its .sha256 exists, so it was frozen)")
            continue
        _sidecar(p)
        n = 0
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if "dir" not in r:
                    continue
                n += 1
                toi_keys = ("source", "source_real") if r["dir"] == "toi>en" else ("reference",)
                for k in ("source", "source_real", "reference"):
                    if isinstance(r.get(k), str):
                        G.add("translator anchor: " + f, r[k], grams=k in toi_keys)
        G.note("translator anchor: " + f, p, n)
    # 21 Oct 2026: every file under chitonga_mono/_test_only (the Cimbonimboni leak; see load_test_only)
    load_test_only(G)
    # 21 Oct 2026: the 500-line translator diagnosis (see load_diag500)
    load_diag500(G)
    return G


ANCHOR_FILES = ("translator_anchor_candidates.jsonl", "translator_anchor_candidates_v2.jsonl", "anchor_translator.jsonl")


# 21 Oct 2026 (the Jesuit; coordinator for Chris): zamai-corpus-raw/chitonga_mono/_test_only/ is held out as a WHOLE
# folder. Why: the Cimbonimboni excerpt (Chris, 15 Oct: "TEST ONLY: poetry-understanding test; never training") sits
# in manifest.jsonl with use 'TEST ONLY', the builders since stage 13 took the manifest without reading `use`, and this
# guard never read _test_only, so all 72 copy-pool rows of the excerpt (mono:cimbonimboni_excerpt1.toi.txt:*) were in
# stage 13, 14, 15 and 15b. Rule: every text file under _test_only (recursive) is a held-out set, keyed like the other
# Tonga held-outs (exact norm, relabelled, typing skeleton, 6-word runs) on every paragraph, every line of 2+ words and
# every sentence of 2+ words inside a line (the builders cut long paragraphs into sentence pieces, which neither the
# exact line nor the 8-word 6-gram rule catches). One-word lines ('Kumunzi', a poem title and an ordinary word) are
# keyed only inside their paragraph. A page image or PDF needs a typed text in its folder or a parent folder below _test_only (a test file the guard cannot
# read is refused, not skipped); *.md files are the folder's notes, not test text. A new test file dropped into
# _test_only is guarded at the next load without any code change.
TEST_ONLY = os.path.normpath(os.path.join(os.path.dirname(RAW), "_test_only"))
TEST_ONLY_TEXT = (".txt", ".tsv", ".jsonl", ".json")
TEST_ONLY_IMAGE = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
TEST_ONLY_NOTES = (".md",)
MIN_TEST_ONLY_FILES = 1
_SENT = re.compile(r"(?<=[.!?])\s+")


def test_only_files(root=None):
    """(text files, image files, other files) under _test_only, sorted"""
    root = root or TEST_ONLY
    txt, img, other = [], [], []
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            e = os.path.splitext(f)[1].lower()
            if e in TEST_ONLY_TEXT:
                txt.append(p)
            elif e in TEST_ONLY_IMAGE:
                img.append(p)
            elif e not in TEST_ONLY_NOTES:
                other.append(p)
    return sorted(txt), sorted(img), sorted(other)


def test_only_strings(path):
    """every key string of one _test_only text file: paragraphs, lines of 2+ words, sentences of 2+ words"""
    raw = open(path, encoding="utf-8").read()
    if path.lower().endswith((".jsonl", ".json")):
        recs = [json.loads(l) for l in raw.splitlines() if l.strip()] if path.lower().endswith(".jsonl") else [json.loads(raw)]
        base = [s for r in recs for s in _strings(r)]
        paras = []
    else:
        base = [l for l in raw.splitlines() if l.strip()]
        paras = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
    out = list(paras)
    for l in base:
        if len(l.split()) >= 2:
            out.append(l)
        for s in _SENT.split(l.strip()):
            if len(s.split()) >= 2 and s.strip() != l.strip():
                out.append(s)
    return out


def load_test_only(G, root=None):
    """add every _test_only text file to Guard G as its own set '_test_only: <relative path>'; GuardError if the folder
    is missing, holds no text, or holds an image / PDF / unknown file with no typed text in the same folder"""
    root = root or TEST_ONLY
    if not os.path.isdir(root):
        raise GuardError(f"frozen set missing: the _test_only folder ({root})")
    txt, img, other = test_only_files(root)
    if len(txt) < MIN_TEST_ONLY_FILES:
        raise GuardError(f"frozen set missing: no text file under {root} ({len(img)} images / PDFs)")
    have = {os.path.normcase(os.path.dirname(p)) for p in txt}
    top = os.path.normcase(os.path.normpath(root))
    for p in img + other:
        d = os.path.normcase(os.path.dirname(p))
        while d not in have and d != top and os.path.dirname(d) != d:   # page images sit in a subfolder of the book
            d = os.path.dirname(d)
        if d not in have:
            raise GuardError(f"_test_only file the guard cannot read and with no typed text beside it: {p}")
    for p in txt:
        name = "_test_only: " + os.path.relpath(p, root).replace(os.sep, "/")
        keys = test_only_strings(p)
        for s in keys:
            G.add(name, s)
        G.note(name, p, len(keys))
    return txt


# 21 Oct 2026 (the Jesuit; Chris: "The translator is integral to our plans"): the 500-line translator diagnosis,
# llm/panel/diagnosis_500_2026-10-21/. Two files, both REQUIRED and frozen by their .sha256 (an edited or missing file
# stops every build): set.jsonl (the 500 lines: English as typed, the clean English, the human Tonga reference) and
# heldout_sections.jsonl (every row of the document sections the 500 were drawn from, chosen or not, so the sections
# leave training whole). Both sides of every row are keyed; the Tonga with 6-word runs and the typing skeleton. Why
# both files: a section's unchosen rows sit between test lines and share their sentences' neighbours; training on them
# would teach the context of the test. The bad-input English (typos, caps) is keyed too, though the clean line already
# catches its source row.
DIAG500 = os.path.join(CT, "llm", "panel", "diagnosis_500_2026-10-21")
DIAG500_FILES = ("set.jsonl", "heldout_sections.jsonl")


def load_diag500(G, root=None):
    root = root or DIAG500
    for f in DIAG500_FILES:
        p = _need(os.path.join(root, f), "translator diagnosis 500 " + f)
        _need(p + ".sha256", "translator diagnosis 500 " + f + " sha256")
        _sidecar(p)
        name = "diagnosis 500: " + f
        n = 0
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            n += 1
            for k in ("ref_toi", "toi"):
                if isinstance(r.get(k), str):
                    G.add(name, r[k])
            for k in ("en", "en_clean"):
                if isinstance(r.get(k), str):
                    G.add(name + " (en)", r[k], grams=False)
        G.note(name, p, n)


MOE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "zamai-corpus-raw", "chitonga_mono", "moe_modules_2026-10-19"))
MOE_HELD_OUT = ("LIT-IN-CHITONGA-TEACHING-MODULE-TERM-2", "LIT-IN-CHITONGA-TEACHING-MODULE-FORM-1-TERM-3")


def conv_tests_only():
    """a Guard holding only the chat_v10_tests/*_test.jsonl sets: to report which training rows touch THEM"""
    G = Guard()
    for tf in sorted(glob.glob(os.path.join(CT, "llm", "chat_v10_tests", "*_test.jsonl"))):
        name = "conversation test: " + os.path.basename(tf)[:-6]
        for l in open(tf, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if "guard" in r:
                    for t in r["guard"]["toi"]:
                        G.add(name, t)
                    for t in r["guard"]["en"]:
                        G.add(name + " (en)", t, grams=False)
                else:
                    for t in _strings(r):
                        G.add(name, t)
    return G


_SKIP_KEYS = {"id", "src", "source", "note", "notes", "kind", "cell", "scenario", "origin", "by", "date", "role", "type",
              "lang", "sha256", "skill", "set", "split", "teacher", "rubric", "criteria", "why"}


def _strings(x, key=None):
    """every string of 2+ words in a JSON record (lists and nested dicts included), bookkeeping keys skipped; a
    multi-line string also yields each of its lines"""
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in _SKIP_KEYS:
                yield from _strings(v, k)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v, key)
    elif isinstance(x, str) and len(x.split()) >= 2:
        yield x
        if "\n" in x:
            for line in x.split("\n"):
                if len(line.split()) >= 2:
                    yield line
