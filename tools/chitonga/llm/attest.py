"""Is every word real Tonga? The attestation check of the harness (15 Oct 2026; Chris on Bemba leaking in: "This is an
interesting challenge ... A blacklist the model can check against?").

An ALLOWLIST with the blacklist as evidence, not a blacklist alone (a blacklist only catches what someone already
listed; the model keeps producing new Bemba-shaped words):
  attested    the word is in lexicon.tsv (incl. plurals), or written by our speakers / in the Tonga corpus (2+ times),
              or a form engine 2 builds from a KNOWN Tonga stem (the old checker accepted any verb-SHAPED word, which
              is how 'Ndilatia' passed), or a name (capitalised, not sentence-initial)
  contact     not attested AND common in Bemba or Nyanja web text (FineWeb-2 bem_Latn / nya_Latn), as a whole word or
              by its last five letters (a stem proxy: ndilatia ~ ...latia): 'probably Bemba / Nyanja', teacher first
  unattested  not attested, no contact evidence: maybe a real Tonga word we have not seen: teacher
Teachers settle it once: 'Tonga' goes into the lexicon (the allowlist grows), 'not Tonga' into the blacklist
(internal/chitonga_train/vocab/not_tonga.tsv, always flagged). One answer settles every sentence with that word.
Counts are cached in internal/chitonga_train/vocab/attest_cache.json (built once, then milliseconds per word).

    python -X utf8 tools/chitonga/llm/attest.py build          build the cache (a few minutes: reads the web text once)
    python -X utf8 tools/chitonga/llm/attest.py "<Tonga line>"
"""
import collections
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RAW = os.path.normpath(os.path.join(REPO, "..", "zamai-corpus-raw"))
VOC = os.path.join(REPO, "internal", "chitonga_train", "vocab")
CACHE = os.path.join(VOC, "attest_cache.json")
NOT_TONGA = os.path.join(VOC, "not_tonga.tsv")
sys.path.insert(0, os.path.join(HERE, "..", "grammar2"))
WORD = re.compile(r"[a-zŋ]+(?:'[a-zŋ]+)*")   # 17 Oct: an apostrophe INSIDE a word (mung'anga) stays; quote marks around it do not ('mutwe')
_c = {}


def _parquet(pattern):
    import pyarrow.parquet as pq
    c = collections.Counter()
    for p in glob.glob(pattern):
        for t in pq.read_table(p).column("text").to_pylist():
            c.update(WORD.findall(t.lower()))
    return c


def build():
    import gen_everyday_v075 as ev
    # CLEAN Tonga only (15 Oct: web text carries Bemba / Zimbabwe forms: 'ndilatia' 3x, 'balasebenza' 2x there):
    # what our speakers wrote, Thompson, the typed books, the Bible and storybooks; no web text
    CT = os.path.join(REPO, "internal", "chitonga_train")
    toi = collections.Counter()
    for p in [os.path.join(CT, "runs", "2026-10-14_export", "train.jsonl"), os.path.join(CT, "desk", "teacher_pairs_2026-10-14.jsonl")] + \
            glob.glob(os.path.join(CT, "desk", "teacher_corrections_v0*.jsonl")):
        for l in open(p, encoding="utf-8"):
            toi.update(WORD.findall((json.loads(l).get("toi") or "").lower()))
    for l in open(os.path.join(CT, "runs", "2026-10-12_stage7", "train.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if (r.get("origin") or "") in ("liseli:bible", "liseli:storybook", "storybooks_zambia", "asb:storybook"):
            toi.update(WORD.findall((r.get("toi") or "").lower()))
    for f in glob.glob(os.path.join(RAW, "chitonga_mono", "phrasebook_thompson", "typed", "*.tsv")) + \
            glob.glob(os.path.join(RAW, "chitonga_mono", "museum_choma", "*", "typed", "*.txt")) + \
            glob.glob(os.path.join(RAW, "chitonga_mono", "library_choma", "*", "*.toi.txt")) + \
            glob.glob(os.path.join(RAW, "chitonga_mono", "edc_letsread", "*.txt")):
        toi.update(WORD.findall(open(f, encoding="utf-8").read().lower()))
    # R9 (18 Oct 2026, Chris: "i suggest yes"): every word of the school-book pairs counts, even once, because the books
    # are the norm; NEVER the held-out lines (exams, the health posters, the readers' last lessons)
    book = collections.Counter()
    for p in glob.glob(os.path.join(CT, "vocab", "new_books_2026-10-16", "*pairs*.jsonl")):
        if "fidelity" in p or "SUPERSEDED" in p or "machine_s11" in p:
            continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l) if l.strip() else {}
            if r.get("toi") and not r.get("held_out") and not r.get("exam"):
                book.update(WORD.findall(r["toi"].lower()))
    bem = _parquet(os.path.join(RAW, "web_contact", "data", "bem_Latn", "*", "*.parquet"))
    nya = _parquet(os.path.join(RAW, "web_contact", "data", "nya_Latn", "*", "*.parquet"))
    # which verb stems our speakers and the clean books actually use (engine reading of every clean word, once)
    import engine2
    used = collections.Counter()
    for w, k in toi.items():
        for p in {x[0] for x in engine2.read(w, known=ev.VERBS)}:
            used[p] += k

    def keep(c, n):
        return {w: k for w, k in c.items() if k >= n}

    def suffixes(c):
        s = collections.Counter()
        for w, k in c.items():
            if len(w) >= 6 and k >= 3:
                s[w[-5:]] += k
        return dict(s)
    data = {"toi": keep(toi, 2), "book": dict(book), "bem": keep(bem, 3), "nya": keep(nya, 3),
            "toi_suf": suffixes(toi), "bem_suf": suffixes(bem), "nya_suf": suffixes(nya),
            "suspects": sorted(r["word"].lower() for r in csv.DictReader(open(os.path.join(VOC, "contact_suspects.tsv"), encoding="utf-8"),
                                                                        delimiter="\t")
                               if float(r["score"] or 0) >= 50 and float(r["speakers"] or 0) == 0),
            "stems_used": dict(used),
            "lex": sorted(ev.NOUNS | ev.VERBS | {r["word"].lower() for r in ev._LEX}), "totals": {
                "toi": sum(toi.values()), "bem": sum(bem.values()), "nya": sum(nya.values())}}
    json.dump(data, open(CACHE, "w", encoding="utf-8"))
    print({k: len(v) if isinstance(v, (dict, list)) else v for k, v in data.items()})


def _load():
    if not _c:
        _c.update(json.load(open(CACHE, encoding="utf-8")))
        _c["sus"] = set(_c.get("suspects", []))
        _c["lexset"] = set(_c["lex"]) - _c["sus"]
        _c["bad"] = {r["word"].lower() for r in csv.DictReader(open(NOT_TONGA, encoding="utf-8"), delimiter="\t")} \
            if os.path.exists(NOT_TONGA) else set()
        import engine2
        import gen_everyday_v075 as ev
        _c["engine"], _c["verbs"] = engine2, ev.VERBS - _c["sus"]
        _sourced_verbs(_c)
    return _c


# --- 18 Oct 2026 (the Jesuit; Chris approved BIG-C review fixes 3 and 5: "3. Yes"): verb stems the lexicon lacks but real
# sources print. internal/chitonga_train/vocab/verb_roots_sourced.tsv: hand-checked roots (kotama, selemuka, bungana, yiisya
# ...), labelled spelling variants (ziba beside zyiba, tondeezya beside tondezya, yinda beside inda; never normalised), regular
# derivations of dictionary verbs whose derived form is itself printed (bonya > kulibonya, langana, katazigwa), and irregular
# perfects the books print (nyamuna > nyamwide). Every row names its source and line. Zambezi Voice is NOT a source here
# (verb_roots_zv_only.tsv waits for Chris). They join the known verbs AFTER the leak lists: a stem on the suspect or not-Tonga
# list is never added, and every not_tonga / contact check in word() still runs first. They count as used stems (printed).
SOURCED = os.path.join(VOC, "verb_roots_sourced.tsv")


def _sourced_verbs(c):
    c["perf_irr"] = {}
    if not os.path.exists(SOURCED):
        return
    rows = csv.DictReader((l for l in open(SOURCED, encoding="utf-8") if not l.startswith("#")), delimiter="\t")
    add = set()
    for r in rows:
        s = (r["root"] or "").strip().lower()
        if not s or s in c["sus"] or s in c["bad"]:
            continue
        if r["kind"] == "perfect":                 # suffix column = the bare printed perfect (nyamwide), root = its verb
            pf = (r["suffix"] or "").strip().lower()
            if len(pf) >= 4:
                c["perf_irr"][pf] = s
        elif r["kind"] in ("root", "derived", "variant"):     # 'held' rows (Bemba/Nyanja look-alikes: temwa, fwalwa) stay out
            add.add(s)
    c["verbs"] = c["verbs"] | add
    c["stems_used"] = dict(c["stems_used"])
    for s in add:
        c["stems_used"].setdefault(s, 1)


def _irregular_perfect(c, lw):
    """a printed irregular perfect the engine tables do not list yet (banyamwide): read it as the rule perfect of the same verb
    (banyamunide) and accept a perfect reading only. A reader rule until perfects.tsv carries the row."""
    eng = c["engine"]
    for pf, st in c.get("perf_irr", {}).items():
        if pf in lw:
            alt = lw.replace(pf, eng.perfect(st))
            if any("perf" in t[2] or t[2] == "plup" for t in eng.read(alt, known={st})):
                return f"irregular perfect '{pf}' of '{st}' (printed in the books; not yet in perfects.tsv)"
    return None


# --- 18 Oct 2026 (the Jesuit, BIG-C review, patch 1): words the noun tables, one prefix, or an object relative explain.
# Runs AFTER the not_tonga and contact checks, so it never passes a word already judged a leak. The object relative is
# a READER rule until the engine tables build it (the books print ncoba- 109 word types, nciba- 50, mboba- 101,
# nzyoba- 62, ncaa- 100; the engine builds none). On the BIG-C flags: 2,654 word types, ~9 in 10 right on form.
_POSS = ("zya", "lya", "lwa", "bwa", "kwa", "twa", "wa", "ca", "ba", "ka", "ya", "za", "aa", "a", "mu", "ku")
_REL = ("nzyo", "nzya", "ndyo", "ndya", "nco", "nca", "nci", "mbo", "mba", "mbwa", "nko", "nka", "mpo", "mpa",
        "ngo", "nga", "ngwa", "njo", "nja", "ndwa", "nto", "ntwa")
# English words that sit inside some toi fields of the export and the book pairs and so count as 'attested' ("the"
# 1,120x, "is" 424x); never Tonga unless the lexicon has them (18 Oct)
ENGLISH = set("the is this she and of to in it that with he was for on are as his they be at have from or by not what "
              "all were when your can said there an which do their if will each how up about out them then some her "
              "would make like him into has look two more go see could my than been who its now people down did get "
              "come made may nd also just very".split())


def _whole(c, x):
    return x in c["lexset"] or c["toi"].get(x, 0) >= 2 or c.get("book", {}).get(x, 0) >= 1


def _prefixed(c, lw):
    """why a word is Tonga by its parts, or None: the noun tables (locative, copula, augment), one possessive /
    locative / 'and' prefix before an attested noun or ku-infinitive (lyamisobano, amulombwana, zyakukazya), or an
    object relative before a finite form of a known verb (ncaalibonya, ncobasamide, mbobasobana)"""
    eng = c["engine"]
    if "nounset" not in c:
        import gen_everyday_v075 as ev
        c["nounset"] = {r["word"].lower() for r in ev._LEX if (r.get("pos") or "").startswith("n")}
    if eng._noun_reading(lw, strict=True):
        return "noun form (locative / copula / augment), read by the noun tables"
    for p in _POSS:
        x = lw[len(p):]
        if lw.startswith(p) and len(x) >= 3 and _whole(c, x):
            if eng._noun_reading(x, strict=True) or x in c["nounset"] or \
               (x.startswith(("ku", "kw")) and any(t[2] == "inf" for t in eng.read(x, known=c["verbs"]))):
                return f"'{p}-' + the attested word '{x}'"
    for m in _REL:
        rest = lw[len(m):]
        if lw.startswith(m) and len(rest) >= 3:
            for r in (rest, "a" + rest if m.endswith("a") else None):      # nca + a-jata (class 1 subject)
                if r and any(t[1] != "-" and t[2] not in ("inf", "imp") for t in eng.read(r, known=c["verbs"])):
                    return f"object relative '{m}-' + a form of a known verb (reader rule; not yet in the engine tables)"
    return None


# ================================================================================================================
# 19 Oct 2026 (the Jesuit; Chris: "We have done a lot of work on Tonga now, you should understand how to set this up
# properly that it doesn't fire unnecessarily"). The check fired on 38.5% of real book-dialogue lines and 68% of
# held-out real Tonga, mostly on long, correct forms (ncookabikkila, nkulizyalila, ulikabotu). The rebuilt word():
#   KNOWN    (a) attested ANYWHERE in our real Tonga text, once is enough, spelling variants as written (ng' / ŋ looked
#            up both ways, never folded otherwise); (b) in the lexicon; (c) analysed by what the engine and this file
#            already know: a verb complex whose stem is a sourced root (or a root + extension: applied -il-, causative
#            -isy-, passive -w- / -igw-, reciprocal -an-, stative -ik-, reflexive li-, an object marker), a proclitic
#            (copula, negative copula, possessive, locative ku- / mu- / a-, 'and' a-, vocative / 'when' no-, object
#            relative) before a known word, a noun prefix before a stem attested with other prefixes, reduplication, an
#            enclitic (-ko / -wo / -mo), two known words written together, and the older reader rules (_prefixed, the
#            printed irregular perfects). An English loan (loanwords.py adaptations, three consonants or more of an
#            English word our pairs use) is known too, tagged as a loan: loanwords are findings.
#   NEVER    names (capitalised inside a sentence, or a name our texts print capitalised), numbers, punctuation.
#   FOREIGN  ('contact') only when NOT attested in Tonga and positively Bemba / Nyanja: the whole word frequent in their
#            web text (Bemba >= 3, Nyanja >= 5: the corpora differ 45x in size), a non-Tonga spelling (dz, word-initial
#            uku-; related_languages.md 2.4), a Bemba augment (ici- / ifi- / umu-) on a word that is not Tonga without
#            it, or, when no analysis explains it, a reading on a stem only Bemba / Nyanja use, or an ending frequent
#            there and absent here.
#   ENGLISH  a common English word inside a Tonga line ('english'): not unknown (the model reads English), but still
#            returned by line() so the harness sees English in a Tonga DRAFT.
# Verdicts: 'ok' (attested / lexicon / name), 'inferred' (the analysis or a loan: generators that ask for '== "ok"'
# keep their stricter meaning), 'english', 'contact', 'not_tonga', 'unattested'.
# note() is the word-check note the harness gives the model and the training data carry: ONE implementation.
REAL = os.path.join(VOC, "attest_real.json")
# 19 Oct 2026 (Chris: "We should have Bemba and Nyanja words in our dictionary, marked as such ... especially younger
# people speak mixed Tonga"): mixed Tonga is speech. A word in grammar2/lexicon_contact.tsv (sourced meaning, tagged
# bem / nya / ila; build_contact_lexicon.py) is KNOWN, verdict 'mixed': note() names it with its Tonga equivalent, and
# line() still lists it (a Bemba word in a Tonga DRAFT is the model leaking). Tier A (a source says it is not Tonga)
# stays 'mixed' unless our Tonga prints it 20+ times; tier B (Hopgood's index) only when our Tonga never prints it.
CONTACT_LEX = os.path.join(HERE, "..", "grammar2", "lexicon_contact.tsv")
LANG_NAME = {"bem": "Bemba", "nya": "Nyanja", "ila": "Ila / Lenje"}
SENSE_TAGGED = {"kumana": "kumana 'finish' is Tonga (47x); 'meet' is Nyanja (F037, F080): flagged only with English 'meet'"}
PROCLITICS = sorted(set(_POSS) | set(_REL) | {
    "n", "m", "ng", "nj", "nz", "nd", "nc", "mb", "nk", "mp", "nt", "ny",          # copula (nkulizyalila, mbwaakali)
    "no", "na", "ni", "ne", "nee", "noo", "naa",                                     # vocative / 'when' / 'and'
    "ta", "to", "tee", "taa", "tii", "ti", "tu", "te",                               # negative copula (tobamakaintu)
    "a", "aa", "ka", "ku", "kw", "mu", "mw", "li", "lu", "ci", "zi", "bu", "ma", "mi", "ba", "i", "u", "o", "e",
    "si", "sa", "ha", "kuli", "muli", "ali", "kuti"}, key=len, reverse=True)
NOUN_PREF = sorted({"mu", "mw", "ba", "mi", "li", "ma", "ci", "zi", "zy", "in", "im", "iŋ", "lu", "tu", "ka", "bu",
                    "ku", "kw", "si", "sa", "ha", "i", "a"}, key=len, reverse=True)
# verb extensions (Hopgood L16-28, L46-58; the books): applied, causative, passive, reciprocal, stative, intensive,
# reversive, and the perfect; each with the final vowels -a / -e / -i (subjunctive, negative)
_CORE = ("ilil", "elel", "uluk", "olok", "ulul", "olol", "isisy", "ezyezy", "ikizy", "ekezy", "ilizy", "elezy",
         "izizy", "isy", "esy", "izy", "ezy", "igw", "egw", "ilw", "elw", "idw", "edw", "ikw", "ekw", "inw", "enw",
         "ik", "ek", "il", "el", "an", "w", "y")
EXT = tuple(sorted({c + v for c in _CORE for v in "aei"} | {"ide", "ede", "ene", "ine", "ile", "ele", "idwe", "edwe",
                                                           "ilwe", "elwe", "igwe", "egwe"}, key=len, reverse=True))
OC = ("ndi", "ku", "mu", "tu", "ba", "ci", "zi", "bu", "lu", "ka", "li", "a", "i", "n", "m")
STRONG_PRINT = re.compile(r"dz|^uku")         # 0 per 1,000 words in our Tonga (related_languages.md 2.4)
AUGMENT = re.compile(r"^(ici|ifi|umu)")        # the Bemba augment; Tonga writers use it now and then (icibaanka)


def build_real():
    """internal/chitonga_train/vocab/attest_real.json: every word of our real Tonga text (count >= 1), the names our
    texts print, the verb stems attested as infinitives, the English words of our pairs (for loans). Real text = the
    sources of build() at ANY count (the clean export, teacher pairs and corrections, the public-domain Bible and storybooks,
    Thompson, museum_choma, library_choma, edc_letsread, the school-book pairs), the 19 Oct export, and the real
    (non-synthetic, non-persona, non-Zambezi-Voice) rows of the v0.11 training file, which are guarded against every
    held-out set. Never the held-out lines, never web text."""
    CT = os.path.join(REPO, "internal", "chitonga_train")
    real, caps, en = collections.Counter(), collections.Counter(), collections.Counter()
    texts = []

    def add(t):
        if t:
            texts.append(t)
    for p in [os.path.join(CT, "runs", "2026-10-14_export", "train.jsonl"), os.path.join(CT, "runs", "2026-10-19_export", "train.jsonl"),
              os.path.join(CT, "desk", "teacher_pairs_2026-10-14.jsonl")] + glob.glob(os.path.join(CT, "desk", "teacher_corrections_v0*.jsonl")):
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                add(r.get("toi"))
                en.update(re.findall(r"[a-z]{4,}", (r.get("en") or "").lower()))
    for l in open(os.path.join(CT, "runs", "2026-10-12_stage7", "train.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if (r.get("origin") or "") in ("liseli:bible", "liseli:storybook", "storybooks_zambia", "asb:storybook"):
            add(r.get("toi"))
    for f in glob.glob(os.path.join(RAW, "chitonga_mono", "phrasebook_thompson", "typed", "*.tsv")) + \
            glob.glob(os.path.join(RAW, "chitonga_mono", "museum_choma", "*", "typed", "*.txt")) + \
            glob.glob(os.path.join(RAW, "chitonga_mono", "library_choma", "*", "*.toi.txt")) + \
            glob.glob(os.path.join(RAW, "chitonga_mono", "edc_letsread", "*.txt")):
        add(open(f, encoding="utf-8").read())
    for p in glob.glob(os.path.join(CT, "vocab", "new_books_2026-10-16", "*pairs*.jsonl")):
        if "fidelity" in p or "SUPERSEDED" in p or "machine_s11" in p:
            continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l) if l.strip() else {}
            if r.get("toi") and not r.get("held_out") and not r.get("exam"):
                add(r["toi"])
                en.update(re.findall(r"[a-z]{4,}", (r.get("en_guess") or "").lower()))
    seed = set(_load()["toi"]) | set(_load().get("book", {})) | _load()["lexset"]
    v11 = os.path.join(CT, "llm", "chat_v11", "O", "train.jsonl")
    for l in open(v11, encoding="utf-8"):
        e = json.loads(l)
        if e["src"].startswith(("synthetic:", "persona:", "bt:zv")):
            continue
        parts = [e["text"]] if e["type"] == "text" else [m["content"] for m in e["messages"] if m["role"] != "system"]
        for t in parts:
            for ln in re.split(r"\n+", t):
                ws = WORD.findall(ln.lower())
                if ws and sum(w in seed for w in ws) >= 0.6 * len(ws) and sum(w in ENGLISH for w in ws) <= 0.1 * len(ws):
                    add(ln)                      # a Tonga line (the English sides and prompts fail the share test)
                elif len(ws) >= 4 and sum(w in ENGLISH for w in ws) >= 0.25 * len(ws):
                    en.update(re.findall(r"[a-z']{3,}", ln.lower()))     # an English line: its words are English
    for t in texts:
        start = True
        for x in re.findall(r"[A-Za-zŋŊ]+(?:'[A-Za-zŋŊ]+)*|[.!?:\"“]", t):
            if not x[0].isalpha():
                start = True
                continue
            real[x.lower()] += 1
            if x[0].isupper() and not start:
                caps[x] += 1
            start = False
    names = sorted(x for x, k in caps.items() if real[x.lower()] <= 2 * k and x.lower() not in _load()["lexset"])
    # English as Zambians print it (News Diggers e-papers, 2017-2026): a word printed lower-case 10+ times is English;
    # names are capitalised and Bemba / Nyanja words are rare in print, so they stay out
    low = collections.Counter()
    for f in sorted(glob.glob(os.path.join(RAW, "nd_full", "*.txt"))):
        t = open(f, encoding="utf-8", errors="ignore").read()
        low.update(re.findall(r"(?<![A-Za-z\-])[a-z]{3,}(?![a-z\-])", t))
    en_pairs = {w: k for w, k in en.items() if k >= 3 and w not in real}      # loans: only what our pairs' English says
    en.update({w: k for w, k in low.items() if k >= 10})
    en_total = sum(low.values()) + 1
    inf = collections.Counter()
    for w, k in real.items():
        m = re.fullmatch(r"ku([^aeiou].{1,}a)|kw([aeiou].{1,}a)", w)
        if m:
            inf[m.group(1) or m.group(2)] += k
    data = {"real": dict(real), "names": names, "inf_stems": {s: k for s, k in inf.items() if len(s) >= 3},
            "english": {w: k for w, k in en.items() if k >= 2 and w not in real}, "english_total": en_total,
            "english_pairs": en_pairs, "lines": len(texts)}
    json.dump(data, open(REAL, "w", encoding="utf-8"), ensure_ascii=False)
    print({k: (len(v) if isinstance(v, (dict, list)) else v) for k, v in data.items()})


def _real():
    c = _load()
    if "real" not in c:
        d = json.load(open(REAL, encoding="utf-8")) if os.path.exists(REAL) else {"real": {}, "names": [], "inf_stems": {}, "english": {}}
        c["real"] = d["real"]
        c["names"] = set(d["names"])
        c["roots"] = (set(c["verbs"]) | set(d["inf_stems"]) | set(c["stems_used"])) - c["sus"] - c["bad"]
        c["english_vocab"] = d["english"]
        c["english_total"] = d.get("english_total", 3e7)
        c["english_pairs"] = d.get("english_pairs", {})
        stems = collections.Counter()
        for w in c["real"]:
            for p in NOUN_PREF:
                if w.startswith(p) and len(w) - len(p) >= 3:
                    stems[w[len(p):]] += 1
        c["noun_stems"] = stems
        c["memo"] = {}
        c["contact_lex"] = {}
        if os.path.exists(CONTACT_LEX):
            rows = [l for l in open(CONTACT_LEX, encoding="utf-8") if l.strip() and not l.startswith("#")]
            for r in csv.DictReader(rows, delimiter="	", quoting=csv.QUOTE_NONE):
                c["contact_lex"][r["word"]] = r
        c["contact_verbs"] = {w for w, r in c["contact_lex"].items() if r["pos"] == "v"}
    return c


def _variants(lw):
    """the spellings a word is looked up under (never folded in the text): ng' / ŋ, and sh beside sy (a Tonga area
    variant a speaker noted, finding F068)"""
    return list(dict.fromkeys([lw, lw.replace("ng'", "ŋ"), lw.replace("ŋ", "ng'"), lw.replace("’", "'"),
                               lw.replace("shy", "sy").replace("sh", "sy")]))


def _attested(c, lw):
    for v in _variants(lw):
        if c["real"].get(v, 0) >= 1 or v in c["lexset"] or c.get("book", {}).get(v, 0) >= 1 or c["toi"].get(v, 0) >= 1:
            return True
    return False


def _root_ok(c, stem):
    """the sourced root under a verb stem the engine proposed (a reflexive li- or an object marker before it, up to two
    extensions after it), or None. The engine inflects; nothing new is made: the root itself must be sourced"""
    roots = c["roots"]
    cands = {stem, stem.replace("shy", "sy").replace("sh", "sy")}     # sh beside sy (F068), lookup only
    for x in list(cands):
        if x.startswith("li") and len(x) > 4:
            cands.add(x[2:])
        if re.match(r"[bmlktc]w[aeiou]", x):                 # an object marker glided before a vowel: bw-iminina
            cands.add(x[2:])
    for x in list(cands):
        for o in OC:
            if x.startswith(o) and len(x) - len(o) >= 3:
                cands.add(x[len(o):])
    for _ in range(2):                    # up to two extensions: bik-kil-a, ga(y)-igw-a, zyal-il-a
        for x in list(cands):
            for e in EXT:
                if x.endswith(e) and len(x) - len(e) >= 2:
                    cands.add(x[:-len(e)] + "a")
                    cands.add(x[:-len(e)] + "ya")        # gaya > gaigwa: the y drops before the passive
    return next((x for x in sorted(cands, key=len, reverse=True) if x in roots and len(x) >= 3), None)


def _analyse(c, lw, depth=0):
    """why a word not attested as a whole is Tonga by its parts, or None (memoised)"""
    key = (lw, depth)
    if key in c["memo"]:
        return c["memo"][key]
    why = None
    if _attested(c, lw):
        why = "attested"
    elif len(lw) >= 3:
        eng = c["engine"]
        e = eng.enclitic(lw)
        if e and _attested(c, e[0]):
            why = f"'{e[0]}' + the enclitic -{e[1]}"
        h = len(lw) // 2
        if not why and len(lw) >= 6 and lw[:h] == lw[h:] and _attested(c, lw[:h]):
            why = f"reduplicated '{lw[:h]}'"
        if not why:
            for t in eng.read(lw):
                r = _root_ok(c, t[0])
                if r:
                    why = f"verb: the engine reads '{t[0]}' ({t[1]} {t[2]}) on the sourced root -{r}"
                    break
        if not why and lw[-1] in "aei":
            r = _root_ok(c, lw[:-1] + "a")
            if r:
                why = f"bare verb stem (command / perfect / passive) on the sourced root -{r}"
        if not why:                       # deverbal nouns: agent mu-vwozy-i, instrument ci-syomenzy-o (Hopgood L58)
            for p in ("mu", "mw", "ba", "si", "ci", "zi", "ka", "tu", "bu", "lu", "ma", "mi", "i"):
                x = lw[len(p):]
                if lw.startswith(p) and len(x) >= 4 and x[-1] in "io":
                    st = x[:-1] + "a"                 # the root itself (or one extension off), no object markers
                    r = st if st in c["roots"] else next((st[:-len(e)] + "a" for e in EXT if st.endswith(e) and
                                                          st[:-len(e)] + "a" in c["roots"] and len(st) - len(e) >= 3), None)
                    if r and len(r) >= 4:
                        why = f"noun from the sourced verb root -{r} ('{p}-' ... '-{x[-1]}')"
                        break
        if not why and depth < 2:
            for p in PROCLITICS:
                rest = lw[len(p):]
                if not lw.startswith(p) or len(rest) < 3:
                    continue
                for r in dict.fromkeys([rest, "a" + rest, "u" + rest, "i" + rest]):
                    sub = _analyse(c, r, depth + 1)
                    if sub and (len(r) >= 4 or sub == "attested"):
                        why = f"'{p}-' + '{r}' ({sub})"
                        break
                if why:
                    break
        if not why:
            for p in NOUN_PREF:
                x = lw[len(p):]
                if lw.startswith(p) and len(x) >= 4 and c["noun_stems"].get(x, 0) >= 2:
                    why = f"noun prefix '{p}-' + the stem '{x}' (attested with other prefixes)"
                    break
        if not why and depth == 0:        # two frequent words written together (ulikabotu = uli kabotu)
            for i in range(3, len(lw) - 2):
                a, b = lw[:i], lw[i:]
                if max(len(a), len(b)) >= 5 and c["real"].get(a, 0) >= 5 and c["real"].get(b, 0) >= 5:
                    why = f"two words written together: '{a}' + '{b}'"
                    break
        if not why and depth == 0:
            why = _prefixed(c, lw) or _irregular_perfect(c, lw)
    c["memo"][key] = why
    return why


def _english(c, lw):
    """English: in our pairs' English or the Zambian press, and more frequent there (per million words) than in the
    Bemba and Nyanja web text. The press prints a few Bemba / Nyanja words in lower case (amenshi 17x, moto 13x); their
    rate in Bemba / Nyanja text is far higher, so they stay Bemba / Nyanja"""
    k = c["english_vocab"].get(lw, 0)
    if k < 2 or not re.fullmatch(r"[a-z']+", lw):
        return False
    t = c["totals"]
    en_rate = k / c["english_total"]
    return en_rate >= c["bem"].get(lw, 0) / t["bem"] and en_rate >= c["nya"].get(lw, 0) / t["nya"]


def _loan(c, lw, strong=False):
    """the English word a loan in Tonga dress comes from (loanwords.py adaptations; a skeleton of three consonants or
    more of an English word our pairs use), or None"""
    if "loan_ix" not in c:
        sys.path.insert(0, os.path.join(HERE, ".."))
        import loanwords as LW
        ix = {}
        for x in c["english_pairs"]:
            for v in LW._en_variants(x):
                for f in LW._en_forms(v):
                    sk = LW._skel(LW._en_norm(f))
                    if len(sk) >= 3:
                        ix.setdefault(sk, x)
        c["loan_ix"], c["LW"] = ix, LW
    LW = c["LW"]
    if not lw.endswith(("a", "e", "i", "o", "u")):
        return None
    for cand in LW._toi_cands(lw):
        sk = LW._skel(LW.norm_toi(cand))
        if len(sk) >= (4 if strong else 3) and sk in c["loan_ix"]:
            return c["loan_ix"][sk]
    return None


def _mixed(c, lw):
    """the note for a word of the contact lexicon (a known Bemba / Nyanja / Ila word), or None"""
    r = c["contact_lex"].get(lw)
    if not r and len(lw) > 5:                      # Tonga markers on a Bemba / Nyanja verb root: balasebenza
        ps = [q for q in c["engine"].read(lw, known=c["contact_verbs"]) if q[0] in c["contact_verbs"]]
        if ps and c["real"].get(lw, 0) == 0 and not any(_root_ok(c, t[0]) for t in c["engine"].read(lw)):
            r = c["contact_lex"][ps[0][0]]
            eq = f"Tonga '-{r['toi_equivalent']}'" if r["toi_equivalent"] else f"'{r['meaning']}'"
            return f"{LANG_NAME.get(r['lang'], r['lang'])} word = {eq} (root '-{ps[0][0]}' with Tonga markers; {r['source'][:60]})"
    if not r:
        return None
    n = c["real"].get(lw, 0)
    if (r["tier"] == "A" and n >= 20) or (r["tier"] == "B" and n > 0):
        return None                                # printed in our Tonga: Tonga wins
    eq = f"Tonga '{r['toi_equivalent']}'" if r["toi_equivalent"] else f"'{r['meaning']}'"
    return f"{LANG_NAME.get(r['lang'], r['lang'])} word = {eq} ({r['source'][:80]})"


def word(w, sentence_initial=False):
    """(verdict, note): 'ok' | 'inferred' | 'mixed' | 'english' | 'contact' | 'not_tonga' | 'unattested' (see above)"""
    c = _real()
    lw = w.lower()
    if not re.search(r"[a-zŋ]", lw):
        return "ok", "number or punctuation"
    if lw in SENSE_TAGGED:
        return "ok", SENSE_TAGGED[lw]
    m = _mixed(c, lw)
    if m:
        return "mixed", m
    bad = c["bad"] - set(SENSE_TAGGED)
    if lw in bad or any(p[0] in bad for p in c["engine"].read(lw, known=bad)):
        return "not_tonga", "on the teachers' not-Tonga list (or built on a stem they rejected)"
    if lw in ENGLISH and not (c["real"].get(lw, 0) >= 3 and lw in c["lexset"]):
        return "english", "a common English word inside a Tonga line"
    if _attested(c, lw):
        if lw in c["sus"] and c["real"].get(lw, 0) == 0:
            return "contact", "in Hopgood's index but common in Bemba / Nyanja and never in our real Tonga"
        return "ok", "attested in our Tonga" if c["real"].get(lw, 0) else "dictionary word"
    if w[:1].isupper() and (not sentence_initial or w in c["names"]):
        return "ok", "name"
    if _english(c, lw):
        return "english", "an English word inside a Tonga line"
    for t in c["engine"].read(lw):                 # a Tonga verb form wins over a web-text look-alike (amutende = cut!)
        r = _root_ok(c, t[0])
        if r and len(lw) >= 5:
            return "inferred", f"verb: the engine reads '{t[0]}' ({t[1]} {t[2]}) on the sourced root -{r}"
    ps = [q for q in c["engine"].read(lw, known=c["contact_verbs"]) if q[0] in c["contact_verbs"]] if len(lw) > 5 else []
    if ps:                                         # Tonga markers on a Bemba / Nyanja verb root: balasebenza
        r = c["contact_lex"][ps[0][0]]
        eq = f" = Tonga '-{r['toi_equivalent']}'" if r["toi_equivalent"] else f" = '{r['meaning']}'"
        return "mixed", f"{LANG_NAME.get(r['lang'], r['lang'])} root '-{ps[0][0]}'{eq} with Tonga markers ({r['source'][:60]})"
    ln = _loan(c, lw, strong=True)
    if ln:                                         # four consonants of an English word: a loan, even if Nyanja has it too
        return "inferred", f"English loan (skeleton matches '{ln}'; a finding, not an unknown)"
    b, n = c["bem"].get(lw, 0), c["nya"].get(lw, 0)
    if b >= 3 or n >= 5:
        return "contact", f"in {'Bemba' if b * 45 >= n else 'Nyanja'} web text ({b} / {n}x), never in our Tonga"
    if STRONG_PRINT.search(lw):
        return "contact", "a Bemba / Nyanja spelling (dz, uku-), never in our Tonga"
    # the pre-19-Oct rules keep their 'ok' (generators that ask for '== "ok"' see what they saw before)
    ps = list(c["engine"].read(lw, known=c["verbs"]))
    if ps and any(c["stems_used"].get(q[0], 0) for q in ps):
        return "ok", "built from a known Tonga stem"
    why = _prefixed(c, lw) or _irregular_perfect(c, lw)
    if why:
        return "ok", why
    why = _analyse(c, lw[1:]) if AUGMENT.search(lw) else _analyse(c, lw)
    if why:
        return "inferred", (f"Bemba-style augment '{lw[0]}-' before a Tonga word: " if AUGMENT.search(lw) else "") + why
    if AUGMENT.search(lw):
        return "contact", "a Bemba-style augment (ici-, ifi-, umu-) on a word our Tonga does not have"
    ln = _loan(c, lw)
    if ln:
        return "inferred", f"English loan (skeleton matches '{ln}'; a finding, not an unknown)"
    ps = list(c["engine"].read(lw, known=c["sus"]))
    if ps:
        return "contact", f"built on '{ps[0][0]}', a stem common in Bemba / Nyanja that our texts never use"
    bs, ns, ts = c["bem_suf"].get(lw[-5:], 0), c["nya_suf"].get(lw[-5:], 0), c["toi_suf"].get(lw[-5:], 0)
    if len(lw) >= 6 and (bs + ns) >= 20 and ts == 0:
        return "contact", f"ending '-{lw[-5:]}' common in {'Bemba' if bs >= ns else 'Nyanja'}, never in our Tonga"
    return "unattested", "not in our texts or the dictionary, and no analysis explains it"


def line(text):
    """the words of a line that are not plainly Tonga: (verdict, word, note) for every verdict except 'ok' and
    'inferred'; 'english' is listed (English inside a Tonga draft) but is not an unknown word (see note())"""
    out = []
    start = True
    prev = ""
    for x in re.findall(r"[A-Za-zŋŊ]+(?:'[A-Za-zŋŊ]+)*|[.!?:\"“‘]|\d+", text or ""):
        if not x[0].isalpha():
            start = start or not x[0].isdigit()
            continue
        v, why = word(x, sentence_initial=start)
        if v not in ("ok", "inferred", "mixed") and prev in ("a", "A"):
            v2, why2 = word("a" + x.lower())       # the a- of a command written apart: 'A mutende' = amutende
            if v2 in ("ok", "inferred"):
                v, why = v2, f"with the a- written apart: {why2}"
        prev = x
        start = False
        if v not in ("ok", "inferred"):
            out.append((v, x, why))
    return out


def note(text):
    """the word-check note for a user turn, as the harness passes it to the model and the training data carry it:
    (note, the unknown words). note = "[unknown word: 'x']" when a word is in none of our sources (Tonga, Bemba,
    Nyanja, English); else one "[Nyanja word: 'madzi' = Tonga 'maanzi']" per known Bemba / Nyanja / Ila word (19 Oct:
    mixed Tonga is speech); else '[all words known]'. The return format is stable: (str, list of unknown words)."""
    items = line(text)
    bad = [w for v, w, _ in items if v in ("unattested", "contact", "not_tonga")]
    if bad:
        return f"[unknown word: '{bad[0]}']", bad
    mixed = []
    for v, w, why in items:
        if v == "mixed" and w.lower() not in [x[0] for x in mixed]:
            lang, eq = why.split(" word = ", 1) if " word = " in why else (why.split(" root ", 1)[0], "")
            eq = eq.split(" (")[0] if eq else re.sub(r"^.*?=\s*", "", why.split(" with Tonga")[0])
            mixed.append((w.lower(), lang, eq))
    if mixed:
        return " ".join(f"[{lang} word: '{w}' = {eq}]" for w, lang, eq in mixed), bad
    return "[all words known]", bad


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "build_real":
        build_real()
    else:
        for x in line(sys.argv[1]):
            print(x)
