"""P1: the decoding profile around the released translator (20 Oct 2026, the Jesuit; council/jesuit_translator_plan
L1 + architect_translator_plan P1; Chris: "we start with the levers that cost nothing").

WHAT IS SERVED. The released translator is the stage 13 soup (zamai-zambia/chitonga-nllb-s13, int8 CT2;
DECISIONS 18 Oct BASE MODEL; tools/chitonga/space/app.py MODEL_REPO default; h2_reader.READER default). Stage 14 is
NOT served (level with 13, LOGBOOK 18 Oct). P1 changes the decoding only, never the weights (one component class).

ONE ENTRY POINT: `service().translate(text, direction)` (direction 'en>toi', 'toi>en' or 'auto'). The harness
(harness3.Harness.p1 / teacher_turn span routing, L8) uses it; the translator page backend (space/app.py) can call the
same function later. Nothing is deployed from here.

The steps, in order, each for a measured reason:
  0 memory     an exact line in the translation memory (h2_memory: books, teachers, Thompson; held-out rows already
               dropped there) is served with its source; per line and per sentence piece. The anchor candidates and
               the chosen anchor are NEVER served from memory (ANCHOR_FILES; Chris's anchor rule).
               24 Oct 2026 (translator optimisation step 1, Chris approved, K0): the memory also holds the teachers'
               checked lesson Tonga (h2_memory 'lessons': box rows a teacher wrote or confirmed, the served fixed
               lines, the 24 Oct desk sentences; never a D line). The guard now also covers the stage 18 frozen
               teacher test + dev and the harness's held-out first 40 requests per teacher (h2_memory.serve_held,
               exact OR near), and when the WHOLE input is a guarded line none of its pieces is served from memory
               either (before, a piece of an anchor line could still be; memory_blocked in the output).
  1 layout     line breaks and table cells (' | ') are kept; an exercise label (1. a) (b) D ...) and a mark ((2), [8],
               (maaka 2) <> (2 marks)) are cut off, kept and put back; marks inside a line split it.
  2 caps       an ALL-CAPS cell is read in sentence case and given back in capitals (20 Oct: 'CHOLERA KILLS!' came
               back garbled, the same line in sentence case right).
  3 protect    morpheme notation (u–la–bona, –la–, -ide), quoted spans in the target language or single letters ("D")
               are replaced by stand-in names the translator copies (h2_text measured Sam / Kim 5 of 5) and restored.
               Blanks (____) are copied by the soup already (probe 20 Oct) and are checked, not replaced.
  4 phone form a lower-case start gets a capital and a missing end mark a full stop (ZV 18 Oct: 38% copied through
               as typed, 0.3% with this); the added full stop is taken off again in the output.
  4b spelling  Tonga input only (Chris's own test, 20 Oct): phone-typed spelling toward the school books
               (tonga_spelling.py: ch > c, ng' > the eng letter, a doubled vowel or b / k, w/u y/i, a split word joined,
               reduplication hyphenated, the lexicon's labelled variants; a word is changed only if the original is
               not school-attested and the result is). BOTH readings are translated; the reading whose top beam the
               whole pool agrees with more leads (MBR); both are logged; the stored text stays the teacher's.
  6b guard     a top beam that loops gets a no-repeat decode beside it (no_repeat_ngram_size 3, repetition penalty
               1.2); a looping candidate counts as a copy-through (the page's 'SOMETHING SOMETHING ...' on caps input).
  5 split      every sentence is translated on its own (NLLB drops second sentences: the pivot test 18 Oct), long
               sentences cut at commas above 25 words (h2_text.split), and rejoined.
  6 n-best     the soup's 5-best (beam 5) + the top beam of each seed (S13 s1, S13 s2): the 'other readers'.
  6c glossary  E>T only, 24 Oct 2026: when the English carries a term the teachers settled (term_guard.gloss_armed,
               internal/chitonga_train/llm/glossary_terms.json), that piece is decoded ONCE MORE on its own at
               num_hypotheses 8 (cfg gloss_nbest; armed pieces only, so an unarmed piece keeps the exact output it
               had before: an int8 batch's padding changes its neighbours, measured 22 Oct). The extra hypotheses
               join the pool as soup:g0..g7. Then (step 7) a missing glossary form is a DECISIVE check: among the
               candidates the highest-SCORING one that carries an accepted form wins, provided it fails no check the
               incumbent passes and costs no more than loss_guard in consensus. If no candidate carries it, the plain
               choice stands and the alarm 'glossary_missing: <term> > expected <forms>' is raised. The service never
               writes the word itself (Chris: no invented words). Off switch: cfg {"glossary_prefer": False} or the
               environment variable CHITONGA_GLOSSARY_PREFER=0 (both leave the old path byte for byte).
  6d tm-near   24 Oct 2026: near rows of the translation memory as CANDIDATES, never as answers (h2_memory.near, both
               directions: English key E>T, Tonga key T>E). Taken: speaker rows (teacher, Thompson) at any length, book
               rows only for a piece of 3+ words; `differs` false (negation / numbers agree); `strict` (cfg
               tm_near_strict, default on): the words that differ are only ones the target does not carry (English
               a / an / the / please; Tonga sena / hena / na / ino / pe or a typing variant; doubled letters never
               folded). Why strict: a near line that wins a check-driven switch SILENCES the incumbent's alarm, and
               the readers cannot tell 'Put 4 stones here' from 'Put 4 sticks here'. Never for a guarded input or
               row (anchor, held-out, stage 18 test, harness held-out requests), never one carrying a word the term
               guard forbids for this input, never for a piece with stand-ins. At most tm_near_k per piece, appended
               to the pool with src 'tm-near <ratio> <row src>'. Their facts are computed APART from the model's
               pool (E>T: their own reader check; T>E: cosine to the model's candidates), so the model's candidates
               keep exactly the facts they had and an output can change ONLY by a tm-near candidate being chosen in
               step 7 (checks + readers; no fiat). A chosen one is recorded: piece tm_near {ratio, src, strict},
               output tm_near [...]; with strict off a chosen loose line also raises the alarm tm_near_loose.
               Off: cfg {"tm_near": False} or CHITONGA_TM_NEAR=0; also off whenever the memory is off.
  7 rerank     the soup's top beam stays unless a check fires on it that another candidate clears, or the readers'
               consensus prefers another candidate by more than a margin. Checks: copy-through, protected notation
               lost, the hard checks of the harness (h3_select.issues: negation, numbers, names, glossary, English left
               in, judge / rules), attestation weight (E>T), dict_missing (L2, if its false alarms stay <= 20%).
               Consensus = MBR (Eikema & Aziz 2020, "Is MAP decoding all you need?"; Freitag et al. 2022, "High
               quality rather than high model probability: MBR decoding with neural metrics"): E>T the median of the
               three readers' back-translations against the English (bge-small cosine), ties broken by the mean
               agreement among the candidates; T>E each English candidate's mean cosine to the other candidates.
               Known risk (ARCHITECT_LEDGER): the readers are relatives of the drafter and reward Tonga that reads back
               easily; only teachers settle that, so P1 changes the incumbent only on evidence, and every change is
               logged with its reason.
  8 restore    stand-ins, caps, labels, marks, cells, lines.
Alarms on the output (for the gate): copy_through, notation_lost, the hard checks, dict_missing, not_tonga / contact
words. P1 never inserts a word of its own (Chris: no invented words): every output is a translator beam or a memory
line, plus the input's own labels, marks and protected spans.

Settings: CFG below, with `set_by` naming the lines that set them (the 181 diagnosis lines only; never the anchor).

    python -X utf8 tools/chitonga/llm/translator_service.py "<text>" [--dir en>toi|toi>en|auto] [--plain] [--json]
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
sys.path[:0] = [HERE, os.path.join(REPO, "tools", "chitonga", "grammar2"), os.path.join(REPO, "tools", "chitonga", "vectors")]

sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "space"))
import p1_space as PS  # noqa: E402  (public helpers: no private data; Codex review 21 Oct: one implementation for both)
import threading  # noqa: E402

MODELS = os.environ.get("CHITONGA_MODELS", r"models")   # env: a pod run (22 Oct)
# 22 Oct 2026 (Chris: "Ok, install 50%"): the teacher assistant's translator is the WiSE-FT blend 50% stage 13 soup /
# 50% stage 16 soup (runs/2026-10-22_merge/; negation sweep: no worse than R). Fallback: CHITONGA_TRANSLATOR=
# chitonga-nllb-s13-soup. The public HF Space is not changed by this.
SOUP = os.path.join(MODELS, os.environ.get("CHITONGA_TRANSLATOR", "chitonga-nllb-wise-a050"))
ANCHOR_FILES = [os.path.join(CT, "llm", "panel", "translator_anchor_candidates.jsonl"),
                os.path.join(CT, "llm", "panel", "translator_anchor_candidates_v2.jsonl"),   # 21 Oct, step 4
                os.path.join(CT, "llm", "panel", "anchor_translator.jsonl"),                 # the frozen pick (step 6)
                os.path.join(CT, "llm", "panel", "translator_anchor.jsonl"),
                os.path.join(CT, "llm", "panel", "anchor.jsonl")]
CFG = {
    "nbest": 5,
    "seeds": ("s13_s1", "s13_s2"),
    "readers": ("soup", "s13_s1", "s13_s2"),
    "margin_e2t": 9,             # consensus switch OFF (self-check 20 Oct: 2 better / 4 worse; 0.10 was the plan value)
    "margin_t2e": 9,             # consensus switch OFF, T>E likewise (MBR still orders the two spelling readings)
    "loss_guard": 0.10,          # a check-driven switch may not cost more than this in consensus
    # L2 false alarms on known-good lines 4.4% T>E / 1.3% E>T (lexicon 20e7b4c5): under the plan's 20%, so it MAY rerank;
    # it does not by default: in the self-check (round 5) the attestation + dict_missing reranks changed 6 more lines for
    # no verdict gained (14 better / 6 worse either way). Both stay ALARMS for the gate (coordinator, audit 20 Oct:
    # "ship P1 as input correction + hard checks WITHOUT the reranking unless your rerun shows a real gain")
    "rerank_dict_missing": False,
    "spelling": True,            # tonga_spelling: the school-spelling reading of phone-typed Tonga beside the original
    "rerank_attest": False,
    "split_words": 25,
    "memory": True,
    "term_guard": True,          # term_guard.py (22 Oct, council step 2): forbid kkolona for cholera, kusoomona for dizzy
    # 24 Oct 2026: the teachers' own classroom words (glossary_terms.json) decide between the n-best, E>T only.
    # Off: service({"glossary_prefer": False}) or CHITONGA_GLOSSARY_PREFER=0 (then this file behaves as on 23 Oct).
    "glossary_prefer": os.environ.get("CHITONGA_GLOSSARY_PREFER", "1") != "0",
    "gloss_nbest": 8,            # hypotheses for an ARMED piece only (unarmed pieces keep the 5 they had)
    # 24 Oct 2026 (step 6d): near memory rows as rerank candidates. Off: service({"tm_near": False}) or CHITONGA_TM_NEAR=0
    "tm_near": os.environ.get("CHITONGA_TM_NEAR", "1") != "0",
    "tm_near_k": 2,              # near rows per piece at most (distinct targets, best ratio first)
    "tm_near_strict": True,      # only Tonga-invariant word differences (see 6d); False = differs-false only, + alarm
    "set_by": "p1_selfcheck 20 Oct round 5, the 181 diagnosis lines (blind 60 x 2 directions + health 61 E>T), "
              "lexicon 20e7b4c53586af33; HARNESS_V3_BUILD.md P1",
}
LANG = {"en": "eng_Latn", "toi": "toi_Latn"}
STAND = ["Kim", "Sam", "Jordan", "Grace", "Robin", "Jamie", "Casey", "Morgan", "Riley", "Quinn"]

# ------------------------------------------------------------------------------------------------ text layout (pure)
ENUM = re.compile(r"^(\s*(?:\d{1,2}[.)]|\d{1,2}(?=\s+[A-Z])|[a-hA-H][.)]|\((?:[a-hA-H]|[ivx]{1,4}|\d{1,2})\)|"
                  r"[ivx]{1,4}[.)]|[•*·▪]|[-–](?=\s))\s+|\s*[A-E]\s+(?=[A-Z][a-z]))")
MARK_END = re.compile(r"(\s*(?:\(\s*(?:maaka\s*)?\d{1,3}(?:\s*marks?)?\s*\)|\[\s*\d{1,3}\s*\]))\s*$", re.I)
MARK_IN = re.compile(r"(\s\(\s*\d{1,2}\s*\)\s)")
CELL = re.compile(r"(\s+\|\s+)")
# a trailing cross-reference '(Figure 1).' is translated as its own piece (22 Oct 2026, weakness map item 3): inside the
# long sentence the model drops it (d500:119 lost 'Figure 1' once 'Figure' stopped counting as a name); alone it comes
# back right: '(Figure 1)' > '(Cifwanikiso 1)', the human Tonga has '(Citondezyo 1)'
REF_END = re.compile(r"\s*\((?:see\s+)?(?:Figure|Fig\.|Table|Annex|Appendix|Section|Chart|Map|Box|Plate)\s+\d+[a-z]?\)\s*[.;]?\s*$",
                     re.I)
NOTATION = re.compile(r"(?<![\w–-])(?:[–-]?[a-zŋ']{1,12}(?:[–-][a-zŋ']{1,12})*[–-]?)(?![\w–-])", re.I)
QUOTE = re.compile(r"(?<![\w])([\"“‘'])([^\"“”‘’'\n]{1,40})([\"”’'])(?![\w])")
BLANK = re.compile(r"_{3,}|\.{4,}|…{2,}")
ABBR = {"mr", "mrs", "ms", "dr", "st", "no", "fr", "sr", "e.g", "i.e", "vs", "etc", "p", "pp", "nb"}


def is_notation(tok):
    """morpheme notation, not an ordinary hyphenated word: an en dash anywhere, or hyphen-minus at an edge (u-, -bona)
    or twice (u-la-bona); reduplication (siyene-siyene, kati-kati) and English compounds (twenty-five) are words"""
    if not NOTATION.fullmatch(tok or "") or not re.search(r"[–-]", tok):
        return False
    segs = [s for s in re.split(r"[–-]", tok.lower()) if s]
    if "–" in tok:
        return True
    if tok.startswith("-") or tok.endswith("-"):
        return True
    return tok.count("-") >= 2 and len(set(segs)) > 1


def is_caps(s):
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) >= 0.8 * len(letters) and len(s.split()) >= 1 \
        and not re.fullmatch(r"\s*[A-Z]{2,5}\s*", s)          # an acronym alone (HIV, CTS) is not a caps heading


def sentence_case(s):
    s = s.lower()
    return re.sub(r"(^|[.!?]\s+)([a-zŋ])", lambda m: m.group(1) + m.group(2).upper(), s)


def sentences(s):
    """sentence pieces; no cut after an abbreviation or an exercise number"""
    out, cur = [], ""
    parts = re.split(r"(?<=[.!?])([\"”’)]?)\s+(?=[\"“‘(]?[A-Z0-9])", s)
    i = 0
    while i < len(parts):
        seg = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        i += 2
        cur = (cur + " " + seg).strip() if cur else seg
        last = re.findall(r"([A-Za-z.]+)[.!?][\"”’)]?$", cur)
        if last and (last[0].lower().rstrip(".") in ABBR or re.fullmatch(r"[A-Za-z]", last[0])):
            continue
        if re.fullmatch(r"\s*\d{1,2}[.)]?", cur):
            continue
        out.append(cur)
        cur = ""
    if cur:
        out.append(cur)
    return [p for p in out if p.strip()]


def mark_to(m, direction):
    """(maaka 2) <> (2 marks); bare (2) and [8] unchanged"""
    n = re.search(r"\d+", m).group(0)
    lead = re.match(r"\s*", m).group(0)
    if direction == "toi>en" and re.search(r"maaka", m, re.I):
        return f"{lead}({n} mark{'s' if n != '1' else ''})"
    if direction == "en>toi" and re.search(r"marks?", m, re.I):
        return f"{lead}(maaka {n})"
    return m


def layout(text, direction, split_words=25):
    """(plan, pieces): plan = list of literals (str) and piece indices (int); pieces = [{text, caps}] to translate"""
    import h2_text as T
    plan, pieces = [], []
    for li, line in enumerate(PS.lines_of(text)):
        if li:
            plan.append("\n")
        for cell in CELL.split(line):
            if CELL.fullmatch(cell):
                plan.append(cell)
                continue
            m = ENUM.match(cell)
            if m and m.end() < len(cell):
                plan.append(m.group(0))
                cell = cell[m.end():]
            post = ""
            mm = MARK_END.search(cell)
            if mm and mm.start() > 0:
                post = mark_to(mm.group(1), direction)
                cell = cell[:mm.start()]
            chunks = [cell]                           # marks inside a line stay in it (self-check 20 Oct)
            rf = REF_END.search(cell)
            ref_tail = ""
            if rf and rf.start() > 0 and re.search(r"[A-Za-z]", cell[:rf.start()]):
                m2 = re.match(r"(\s*\(.*\))(\s*[.;]?\s*)$", cell[rf.start():])
                chunks = [cell[:rf.start()], m2.group(1)]            # '(Figure 1)' its own piece (22 Oct, d500:119)
                ref_tail = m2.group(2)                               # the end mark after it stays a literal
            for ci, ch in enumerate(chunks):
                if MARK_IN.fullmatch(ch):
                    plan.append(ch)
                    continue
                lead = re.match(r"\s*", ch).group(0)
                trail = re.search(r"\s*$", ch).group(0)
                core = ch.strip()
                if lead:
                    plan.append(lead)
                if not core:
                    continue
                if not re.search(r"[A-Za-zŋŊ]", core) or all(is_notation(t) or BLANK.fullmatch(t) for t in core.split()):
                    plan.append(core)                     # numbers, notation, blanks alone: copied
                else:
                    caps = is_caps(core) and PS.is_caps(core)          # no Titlecase name in a caps line (#13)
                    prot, mp = protect(core, direction)                  # before any case change (#13)
                    body = PS.sentence_case(prot, keep=[k for k in mp if k != "__unprotected__"]) if caps else prot
                    sents = PS.sentences(body)                           # lower-case / Unicode initials (#14)
                    for si, snt in enumerate(sents):
                        for pj, p in enumerate(T.split(snt, split_words)):
                            if si or pj:
                                plan.append(" ")
                            plan.append(len(pieces))
                            sub = {ph: o for ph, o in mp.items() if ph != "__unprotected__" and PS.ph_count(p, ph)}
                            if mp.get("__unprotected__"):
                                sub["__unprotected__"] = mp["__unprotected__"]
                            pieces.append({"text": p, "caps": caps, "map": sub,
                                           "counts": {ph: PS.ph_count(p, ph) for ph in sub if ph != "__unprotected__"}})
                if trail:
                    plan.append(trail)
                if ref_tail and ci == 0 and pieces:
                    pieces[-1]["strip_stop"] = True    # the sentence's end mark comes after the reference
            if ref_tail:
                plan.append(ref_tail)
            if post:
                plan.append(post)
    return plan, pieces


def strong_norm(s):
    """h2_memory.norm plus option letters (A-E) and end marks stripped: the identity of a line whatever its label"""
    import h2_memory as M
    return M.held_norm(s)                 # 21 Oct: one definition (labels, numbering, blanks, marks) for every guard


def strip_label(src, out):
    """an exercise label the source line does not carry is taken off the output (memory rows carry their page's label);
    never one whose number or letter the source has (Codex #7: '12 Children.' keeps its 12)"""
    return PS.strip_label(src, out)


def render(q, text):
    """one piece's chosen candidate back into the input's shape: stand-ins restored, caps given back, the full stop
    that phone_form added taken off again"""
    txt = restore(text.upper() if q["caps"] else text, q["map"])
    txt = strip_label(q["src"], txt)                     # P1 never inserts a label the piece did not have
    if (q["stop_added"] or q.get("strip_stop")) and txt.endswith("."):
        txt = txt[:-1]
    return txt


def assemble(plan, outs):
    return "".join(x if isinstance(x, str) else outs[x] for x in plan)


# ------------------------------------------------------------------------------------------------ protect (pure)
def _en_vocab():
    import h2_text as T
    return T._en_vocab()


def _quoted_target(span, direction):
    """a quoted span is kept when it is a single letter / number, or already in the TARGET language (Tonga in an
    English instruction for E>T: 'the word "bana"'; English in a Tonga line for T>E: 'bbala "shirt"')"""
    ws = re.findall(r"[A-Za-zŋ']+", span)
    if len(span.split()) > 3:
        return False
    if re.fullmatch(r"\s*([A-Za-z]|\d+)\s*", span):
        return True
    if not ws:
        return False
    voc = _en_vocab()
    english = all(w.lower() in voc for w in ws)
    return (not english) if direction == "en>toi" else english


def protect(text, direction):
    """(text with stand-ins, mapping {stand-in: original}): notation tokens and target-language quoted spans"""
    pool = [s for s in STAND if s.lower() not in (text or "").lower()]
    mapping = {}

    def take(orig):
        for ph, o in mapping.items():
            if o == orig:
                return ph
        if not pool:
            mapping.setdefault("__unprotected__", []).append(orig)
            return None
        ph = pool.pop(0)
        mapping[ph] = orig
        return ph

    def qrep(m):
        if not _quoted_target(m.group(2), direction):
            return m.group(0)
        ph = take(m.group(0))
        return ph or m.group(0)
    text = QUOTE.sub(qrep, text)
    out = []
    for tok in re.split(r"(\s+)", text):
        core = tok.strip(",.;:!?()")
        if core and is_notation(core):
            ph = take(core)
            tok = tok.replace(core, ph) if ph else tok
        out.append(tok)
    unprot = mapping.pop("__unprotected__", [])
    if unprot:
        mapping["__unprotected__"] = unprot
    return "".join(out), mapping


def restore(text, mapping):
    """the ONE whole-token, case-insensitive, Unicode-aware matcher (p1_space) for restoring and checking (#11)"""
    return PS.restore(text, mapping)


def lost(text, q_or_map):
    """stand-ins whose COUNT in the output differs from the piece (#12); a bare mapping = presence (old callers)"""
    if isinstance(q_or_map, dict) and "counts" in q_or_map:
        return PS.lost(text, q_or_map)
    return [ph for ph in q_or_map if ph != "__unprotected__" and not PS.ph_count(text, ph)]


def end_kind(s):
    """'?' / '.' (a full stop or '!') / None (no end mark: a phone-typed line or a heading) at the end of a line,
    closing quotes and brackets aside"""
    m = re.search(r"([?.!])[\"'”’)\]]*\s*$", s or "")
    return None if not m else "?" if m.group(1) == "?" else "."


def same_kind(a, b):
    """24 Oct (memory): norm() folds '?', so 'Continue.' found the fixed line 'Continue?' > 'Zumanana?', a question
    for a command. A row must not be a question when the input is a statement or the other way round; a line with no
    end mark (phone typing) is not judged, so it keeps its memory hits"""
    ka, kb = end_kind(a), end_kind(b)
    return ka is None or kb is None or ka == kb


def phone_form(s):
    """(text, cap_added, stop_added): only a lower-case start of 2+ words is phone form (ZV 18 Oct); a capitalised
    heading without a full stop and a single word are left as typed (self-check 20 Oct)"""
    cap = stop = False
    if not s or not s[0].islower() or len(s.split()) < 2:
        return s, False, False
    if s and s[0].islower():
        s, cap = s[0].upper() + s[1:], True
    if s and s[-1] not in ".?!\"'”’:;)":
        s, stop = s + ".", True
    return s, cap, stop


# ------------------------------------------------------------------------------------------------ checks (pure)
WORDS = re.compile(r"[a-zŋ']+")


def is_copy(src, out):
    """the output is the input again (copy-through): >= 70% of the words both ways; lines of one word never count"""
    a = [w for w in WORDS.findall((src or "").lower()) if w not in {s.lower() for s in STAND}]
    b = [w for w in WORDS.findall((out or "").lower()) if w not in {s.lower() for s in STAND}]
    if len(a) < 2 or not b:
        return False
    sa, sb = set(a), set(b)
    return len(sa & sb) >= 0.7 * len(sa) and len(sa & sb) >= 0.7 * len(sb)


def has_loop(text, src=""):
    """a decoding loop: a token three times in a row, a token trigram three times, or the whole output said twice
    ('Kamulya! Kamulya!' for 'Eat!'); hyphenated reduplication (kati-kati) is one token and a repetition the source has
    is no loop (Codex #21; p1_space.has_loop, one implementation)"""
    return PS.has_loop(text, src)


def decisive(f, cfg=CFG):
    return (int(bool(f.get("copy"))), f.get("lost", 0), f.get("hard", 0),
            f.get("attest_w", 0) if cfg.get("rerank_attest", True) else 0,
            f.get("dm", 0) if cfg.get("rerank_dict_missing", True) else 0)


def decide(cands, direction, cfg=CFG):
    """(index chosen, reason). cands[0] is the incumbent (the soup's top beam). Pure: works on stored facts, so the
    tuning re-decides without re-translating."""
    if not cands:
        return None, "no candidate"
    inc = cands[0]
    di, mi = decisive(inc, cfg), inc.get("meaning") or 0.0
    margin = cfg["margin_e2t"] if direction == "en>toi" else cfg["margin_t2e"]
    order = sorted(range(len(cands)), key=lambda i: (decisive(cands[i], cfg), -(cands[i].get("meaning") or 0),
                                                       -(cands[i].get("agree") or 0), i))
    for i in order:
        if i == 0:
            break
        c = cands[i]
        if decisive(c, cfg) < di and (c.get("meaning") or 0) >= mi - cfg["loss_guard"]:
            return i, f"checks: incumbent {di} > {decisive(c, cfg)} ({', '.join(inc.get('why', [])[:3])})"
    # 24 Oct 2026: the teacher glossary, after the safety checks and before consensus. The incumbent is left alone
    # unless it misses a term another candidate carries; that candidate must fail no check the incumbent passes
    # (decisive no worse) and must not cost more than loss_guard in consensus. Among those, the HIGHEST-SCORING one
    # (the model's own score; None last), ties by pool order. Nothing is written: only a hypothesis is chosen.
    if cfg.get("glossary_prefer", True) and (inc.get("gmiss") or 0):
        gm = inc["gmiss"]
        have = [i for i in range(1, len(cands))
                if (cands[i].get("gmiss") or 0) < gm and decisive(cands[i], cfg) <= di
                and (cands[i].get("meaning") or 0) >= mi - cfg["loss_guard"]]
        if have:
            def _key(i):
                sc = cands[i].get("score")
                return ((cands[i].get("gmiss") or 0), -(sc if sc is not None else -9e9), i)
            b = min(have, key=_key)
            return b, f"glossary: {', '.join(inc.get('gloss_miss', [])[:3])} ({cands[b]['src']})"
    same = [i for i in range(1, len(cands)) if decisive(cands[i], cfg) == di]
    if same:
        b = max(same, key=lambda i: ((cands[i].get("meaning") or 0), cands[i].get("agree") or 0))
        if (cands[b].get("meaning") or 0) > mi + margin:
            return b, f"consensus: {cands[b].get('meaning')} > {mi} + {margin}"
    return 0, "incumbent"


# ------------------------------------------------------------------------------------------------ the service
class Service:
    def __init__(self, cfg=None):
        self.cfg = dict(CFG, **(cfg or {}))
        self._r = {}
        self._rev = None
        self._forbid = None
        self._tl = threading.local()     # the whole input of the current translate() call: the term guard's context
        self.guard_log = []              # every armed decode (term_guard report), newest last; bounded below

    # models ------------------------------------------------------------------
    def reader(self, name):
        import h2_reader as R
        import h3_select as S
        if name not in self._r:
            if name == "soup":
                r = S.reader("soup")
                if os.path.normcase(os.path.abspath(r.dir)) != os.path.normcase(SOUP):
                    r = R.Reader(SOUP)           # the served model, whatever HARNESS_READER says
                self._r[name] = r
            else:
                self._r[name] = S.reader(name)
        return self._r[name]

    def nbest(self, name, texts, direction, k, guard=False, beam=None, ctx=None):
        """ctx: the whole English input (the term guard's trigger); default the current translate() call's input"""
        r = self.reader(name)
        if ctx is None:
            ctx = getattr(self._tl, "ctx", None)
        with self._lock_for(name):
            return self._nbest(r, texts, direction, k, guard, beam, ctx, name)

    # readers that are actually on this machine -------------------------------
    def _reader_dir(self, name):
        import h3_select as S
        return SOUP if name == "soup" else S.EXTRA_READERS.get(name)

    def _have(self, name):
        """a reader needs BOTH folders: the tokenizer in <dir> (h2_reader: AutoTokenizer.from_pretrained(d)) and the
        int8 weights in <dir>-ct2-int8. Checking both matters: on 24 Oct 03:45 the two S13 seeds came back from HF as
        `-ct2-int8` only (model.bin, config.json, shared_vocabulary.json), with no sentencepiece.bpe.model, so the
        Reader would still have raised. The missing part is named in the message."""
        d = self._reader_dir(name)
        if not d:
            return False
        self._why_missing = (f"{d}: no tokenizer folder" if not os.path.isdir(d) else
                             f"{d}-ct2-int8: no int8 weights" if not os.path.isdir(d + "-ct2-int8") else "")
        return not self._why_missing

    def available(self):
        """(readers, seeds, missing): the pool this machine can actually load, cached.

        WHY (24 Oct 2026). The two S13 seed models were deleted in the night's clean-up; the service then raised
        OSError inside h2_reader on the first seed, and every caller (harness, desk, pipeline) swallows the
        exception, so translation went silently missing. A missing reader must COST A READER, not the translation:
        the pool shrinks to what is there (the soup alone if need be) and every answer carries the alarm
        `reader_missing: <names>`, so the harness routes it to a person instead of serving a quietly weaker line.
        The soup itself is never dropped: without it there is no translator, and that error must be loud."""
        if getattr(self, "_avail", None) is None:
            miss, rd, sd, why = [], [], [], []
            for kind, names in (("reader", self.cfg["readers"]), ("seed", self.cfg["seeds"])):
                for n in names:
                    if (n == "soup" and kind == "reader") or self._have(n):
                        (rd if kind == "reader" else sd).append(n)
                    elif n not in miss:
                        miss.append(n)
                        why.append(f"{n} ({getattr(self, '_why_missing', '') or 'not configured'})")
            if miss and not getattr(self, "_warned", False):
                self._warned = True
                print(f"[translator_service] reader_missing: {'; '.join(why)}; running with {rd or ['soup']}",
                      file=sys.stderr, flush=True)
            self._avail = (tuple(rd), tuple(sd), tuple(miss))
        return self._avail

    def _lock_for(self, name):
        """Codex #5: the readers' tokenizers are shared and src_lang is state: one lock per reader around the language
        switch, the encoding and the decode (the harness reader objects are shared with h3_select)"""
        with _LOCKS_GUARD:
            return _LOCKS.setdefault(name, threading.RLock())

    def _nbest(self, r, texts, direction, k, guard, beam, ctx=None, name=""):
        src, tgt = ("en", "toi") if direction == "en>toi" else ("toi", "en")
        tok = r.tok
        tok.src_lang = LANG[src]
        kw = dict(target_prefix=[[LANG[tgt]]] * len(texts), beam_size=beam or max(4, k), num_hypotheses=k,
                  max_decoding_length=256, return_scores=True,
                  **({"no_repeat_ngram_size": 3, "repetition_penalty": 1.2} if guard else {}))
        toks = [tok.convert_ids_to_tokens(tok.encode(t)) for t in texts]
        if self.cfg.get("term_guard", True):           # 23 Oct: toi>en rules too (term_guard.py "dir")
            import term_guard as TG
            # the trigger reads the whole input AND the piece (a stand-in route may pass a piece the input lacks)
            hyps, reps = TG.translate_batch(r.tr, tok, toks, [f"{ctx or ''} {t}" for t in texts], direction, **kw)
            for t, rep in zip(texts, reps):
                if rep:
                    self._gseq = getattr(self, "_gseq", 0) + 1
                    self.guard_log.append(dict(rep, reader=name, piece=t, seq=self._gseq, tid=threading.get_ident()))
            del self.guard_log[:-200]
        else:
            res = r.tr.translate_batch(toks, **kw)
            hyps = [list(zip(x.hypotheses, x.scores)) for x in res]
        return [[(tok.decode(tok.convert_tokens_to_ids(h[1:]), skip_special_tokens=True), round(s, 3))
                 for h, s in x] for x in hyps]

    # memory ------------------------------------------------------------------
    def forbidden(self):
        """normalised texts of the anchor candidates and the chosen anchor: never served from memory"""
        st = self._guard_stamp()
        if getattr(self, "_forbid_stamp", None) != st:     # Codex guard review 2: an anchor frozen after start-up
            self._forbid, self._held, self._forbid_stamp = None, None, st
        if self._forbid is None:
            import h2_memory as M
            f = set()
            for p in ANCHOR_FILES:
                if os.path.exists(p):
                    for l in open(p, encoding="utf-8"):
                        if l.strip():
                            r = json.loads(l)
                            for k in ("source", "source_real", "reference", "en", "toi", "request", "text"):
                                if isinstance(r.get(k), str) and r[k].strip():
                                    f.add(M.norm(r[k]))
            self._forbid = f
        return self._forbid

    def _guard_stamp(self):
        import glob
        import h2_memory as M
        return M._stamp_of(ANCHOR_FILES + [p + ".sha256" for p in ANCHOR_FILES]
                           + glob.glob(os.path.join(CT, "vocab", "new_books_2026-10-16", "*pairs*.jsonl")))

    def held(self):
        """held-out and exam rows of every book pairs file, both sides, in the STRONG norm (labels incl. option
        letters and marks stripped). Why: the books repeat exam items under another label (Grade 8 p87 'a) Having sex
        with two people only.' = the held-out Grade 10 'B Having sex ...'); h2_memory's guard keeps 'b ' and missed
        them, so after P1 cut the label the memory served a held-out line (self-check 20 Oct: 3 of 5 hits)"""
        self.forbidden()                                   # refreshes both caches when a guard file changed
        if getattr(self, "_held", None) is None:
            import glob
            h = set()
            for f in glob.glob(os.path.join(CT, "vocab", "new_books_2026-10-16", "*pairs*.jsonl")):
                b = os.path.basename(f)
                if "fidelity" in b or "SUPERSEDED" in b:
                    continue
                for l in open(f, encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l)
                        if r.get("held_out") or r.get("exam"):
                            h.add(strong_norm(r.get("toi") or ""))
                            h.add(strong_norm(r.get("en_guess") or ""))
            h.discard("")
            self._held = h | {strong_norm(x) for x in self.forbidden()}
        return self._held

    def blocked(self, text):
        """why `text` may never be served from memory nor be a memory row served (None = free): an anchor line
        (forbidden), a held-out book / guard line (held), or, 24 Oct 2026, a stage 18 frozen teacher test / dev line
        or a harness held-out request, exact or NEAR (h2_memory.serve_held, its own stamp: a new harness message
        does not rebuild the memory)"""
        import h2_memory as M
        k = M.norm(text)
        if not k:
            return "empty"
        if k in self.forbidden():
            return "anchor"
        if strong_norm(text) in self.held():
            return "held-out"
        return M.serve_held().hit(text)

    def memory(self, text, direction):
        import h2_memory as M
        if not self.cfg.get("memory", True):
            return None
        k = M.norm(text)
        if not k or self.blocked(text):
            return None
        nw = len(k.split())

        def usable(r, side):
            # a book row of 1-2 words is often a list item or a bare stem ('Eat!' > 'lya', Grade 6 word list): speakers'
            # rows (teachers, Thompson) at any length, book rows only for lines of 3+ words. 24 Oct: neither side of
            # the row may be a guarded line (the stage 18 test sits in the lesson boxes)
            other = "en" if side == "toi" else "toi"
            # 24 Oct: the row's source side and the input agree on question / statement (same_kind)
            return (r.get("kind") != "book" or nw >= 3) and not self.blocked(r[side]) and not self.blocked(r[other]) \
                and same_kind(r[other], text)
        if direction == "en>toi":
            rows = [r for r in M.lookup(text)["exact"] if usable(r, "toi")]
            return {"text": strip_label(text, rows[0]["toi"]), "src": rows[0]["src"],
                    "confidence": rows[0].get("confidence")} if rows else None
        m = M.memory()
        if self._rev is None or self._rev[0] != m.get("stamp"):
            rev = {}
            for rows in m["by"].values():
                for r in rows:
                    rev.setdefault(M.norm(r["toi"]), []).append(r)
            self._rev = (m.get("stamp"), rev)
        rows = [r for r in self._rev[1].get(k, []) if usable(r, "en")]
        if not rows:
            return None
        order = {"teacher": 0, "thompson": 1, "book": 2}
        rows.sort(key=lambda r: order.get(r.get("kind"), 3))
        return {"text": strip_label(text, rows[0]["en"]), "src": rows[0]["src"], "confidence": None}

    # the entry point ---------------------------------------------------------
    def translate(self, text, direction="auto", keep=False, use_memory=None):
        """{'text', 'direction', 'stage', 'alarms', 'pieces' (per piece: src, chosen, why, candidates if keep), 't'}"""
        t0 = time.time()
        if direction == "auto":
            direction = detect_direction(text)
        self._tl.ctx = text                      # the term guard reads the whole input (term_guard.py)
        nlog = getattr(self, "_gseq", 0)
        out = {"input": text, "direction": direction, "model": os.path.basename(SOUP), "stage": None, "alarms": [],
               "pieces": [], "cfg_set_by": self.cfg["set_by"]}
        mem_on = self.cfg.get("memory", True) if use_memory is None else use_memory
        # 24 Oct 2026: a guarded WHOLE input (anchor, held-out, stage 18 test, harness held-out request) gets no memory
        # for any of its pieces either, exact or near: a test line must be translated, never looked up piecewise
        if mem_on and self.blocked(text):
            out["memory_blocked"] = self.blocked(text)
            mem_on = False
        out["tm_near"] = []
        if mem_on:
            hit = self.memory(text, direction)
            if hit and PS.is_caps(text):                 # Codex #16: the input's shape, not the memory row's
                hit = dict(hit, text=hit["text"].upper())
            lab = ENUM.match(text or "")
            if hit and lab and not ENUM.match(hit["text"]):
                hit = dict(hit, text=lab.group(0) + hit["text"])
            if hit:
                out.update(text=hit["text"], stage="memory", memory=hit, t=round(time.time() - t0, 3))
                return out
        # a reader model that is not on this machine shrinks the pool and is SAID (never silent): the harness routes
        # on an alarm, so a degraded translation reaches a person instead of a teacher's class (24 Oct, the night's
        # clean-up removed the two S13 seeds and every caller swallowed the OSError)
        if self.available()[2]:
            out["alarms"].append("reader_missing: " + ", ".join(self.available()[2]))
        plan, pieces = layout(text, direction, self.cfg["split_words"])
        prep = []
        for p in pieces:
            pt, mp = p["text"], p["map"]
            pt, cap, stop = phone_form(pt)
            q = {"src": p["text"], "text": pt, "map": mp, "counts": p["counts"], "caps": p["caps"], "stop_added": stop,
                 "cap_added": cap, "strip_stop": p.get("strip_stop", False),
                 "_mem_off": not mem_on}          # step 6d: tm-near is memory (private key: never in the output)
            if direction == "toi>en" and self.cfg.get("spelling", True):
                import tonga_spelling as TSP
                nt, ch = TSP.normalise(pt)
                if ch:
                    q.update(norm=nt, norm_changes=ch)
            prep.append(q)
        outs = [None] * len(prep)
        todo = []
        for i, q in enumerate(prep):
            hit = self.memory(q["src"], direction) if mem_on else None
            if hit:
                outs[i] = render(dict(q, stop_added=False), hit["text"])      # caps and labels as typed (Codex #16)
                q.update(chosen=outs[i], why="memory: " + hit["src"][:80], stage="memory")
            else:
                todo.append(i)
        if todo:
            cands = self.candidates([prep[i] for i in todo], direction)
            self.check([prep[i] for i in todo], cands, direction)
            for i, cs in zip(todo, cands):
                # two readings of phone-typed Tonga: the one whose top beam the whole pool agrees with more leads (MBR)
                j = next((k for k, c in enumerate(cs) if c["src"] == "norm:soup:b4" or "norm:soup:b4" in c["also"]), None)
                if j and (cs[j].get("meaning") or 0) >= (cs[0].get("meaning") or 0):
                    cs.insert(0, cs.pop(j))
                    prep[i]["reading"] = "school spelling"
                elif prep[i].get("norm"):
                    prep[i]["reading"] = "as typed"
            for i, cs in zip(todo, cands):
                j, why = decide(cs, direction, self.cfg)
                q = prep[i]
                c = cs[j]
                txt = render(q, c["text"])
                outs[i] = txt
                q.update(chosen=txt, chosen_src=c["src"], switched=j != 0, why=why, stage="model",
                         alarms=c.get("why", []))
                if c.get("tm"):                  # step 6d: a memory line won the rerank (checks + readers), say so
                    q["tm_near"] = dict(c["tm"])
                    out["tm_near"].append({"piece": q["src"], "chosen": txt, **c["tm"]})
                    if not c["tm"].get("strict"):
                        out["alarms"].append(f"tm_near_loose: {c['tm']['ratio']} {c['tm']['src'][:60]}")
                elif any(str(a).startswith("tm-near") for a in c.get("also", [])):
                    q["tm_near_agrees"] = True   # the model's own beam IS the near memory line (information only)
                if keep:
                    q["candidates"] = cs
                for a in c.get("why", []):
                    if a.startswith("glossary_missing") and a in out["alarms"]:
                        continue                      # one alarm per term, however many pieces ask for it
                    out["alarms"].append(a)
        out["text"] = assemble(plan, outs)
        if keep:
            out["plan"] = plan
        out["stage"] = "model" if todo else "memory"
        # private keys (the compiled glossary terms) never leave the service; q["glossary"] carries the term names
        out["pieces"] = [{k: v for k, v in q.items() if (k != "map" or keep) and not k.startswith("_")} for q in prep]
        out["switched"] = sum(1 for q in prep if q.get("switched"))
        g = [x for x in list(self.guard_log) if x["seq"] > nlog and x["tid"] == threading.get_ident() and x["reader"] == "soup"]
        if g:
            out["term_guard"] = g
            for a in dict.fromkeys(f"term_guard: {', '.join(x['rules'])}" + (" unresolved" if x["unresolved"] else "")
                                   for x in g if x.get("blocked") or x["unresolved"]):
                out["alarms"].append(a)              # one alarm per rule (the soup decodes a piece twice: k-best, beam 4)
        out["t"] = round(time.time() - t0, 3)
        return out

    def glossary_for(self, prep, direction):
        """per piece, the teacher glossary terms THIS PIECE's English arms (E>T only). {} when the lever is off.

        Why the piece and not the whole input, unlike the term guard: a forbidden word is forbidden anywhere in the
        line, but a term's Tonga belongs in the piece that says it. Armed from the whole input, every piece of the
        long menu line f18t:2 demanded maths, English and science (24 Oct 03:15): three alarms per piece and a
        switch on a piece that never mentioned them."""
        if direction != "en>toi" or not self.cfg.get("glossary_prefer", True):
            return {}
        import term_guard as TG
        out = {}
        for i, q in enumerate(prep):
            ts = TG.gloss_armed(q["text"])
            if ts:
                out[i] = ts
                q["_gterms"] = ts
                q["glossary"] = [t["term"] for t in ts]
        return out

    def candidates(self, prep, direction):
        """per piece: [soup beam 0 (incumbent), soup beams 1..k-1, seed tops], deduplicated, sources kept.
        An ARMED piece (glossary_for) is decoded once more alone at gloss_nbest hypotheses; those join the pool as
        soup:g0..g7. Unarmed pieces are not touched, so their output cannot change (int8 batch padding, 22 Oct)."""
        import h2_text as T
        texts = [q["text"] for q in prep]
        k = self.cfg["nbest"]
        soup = self.nbest("soup", texts, direction, k)
        rel = self.nbest("soup", texts, direction, 1, beam=4)    # what the released page and plain decode give
        sd = self.available()[1]                 # only the seed models this machine actually has (24 Oct)
        seeds = {s: self.nbest(s, texts, direction, 1) for s in sd}
        # the glossary's extra decode comes AFTER the three calls above, in a batch of its own, so the calls that make
        # the incumbent and the old pool are the same calls, in the same order, as before 24 Oct. It is only run for a
        # piece whose own plain output MISSES a term: when the teacher's word is already there, nothing is decoded
        # again and nothing can change (this is also what keeps the lever cheap on the Beelink's CPU)
        gl = self.glossary_for(prep, direction)
        kg = max(k, int(self.cfg.get("gloss_nbest", 8)))
        gext = {}
        if gl:
            import term_guard as TG
            for i, ts in gl.items():
                if TG.gloss_missing(rel[i][0][0], ts) or TG.gloss_missing(soup[i][0][0], ts):
                    gext[i] = self.nbest("soup", [prep[i]["text"]], direction, kg, beam=kg)[0]
        ni = [i for i, q in enumerate(prep) if q.get("norm")]
        nsoup = self.nbest("soup", [prep[i]["norm"] for i in ni], direction, k) if ni else []
        nrel = self.nbest("soup", [prep[i]["norm"] for i in ni], direction, 1, beam=4) if ni else []
        nseeds = {s: self.nbest(s, [prep[i]["norm"] for i in ni], direction, 1) for s in sd} if ni else {}
        gi = [i for i in range(len(prep)) if has_loop(rel[i][0][0], texts[i]) or has_loop(soup[i][0][0], texts[i])]
        guard = self.nbest("soup", [texts[i] for i in gi], direction, 1, guard=True) if gi else []
        allc = []
        for i, q in enumerate(prep):
            cs, seen = [], {}
            items = [("soup:b4", rel[i][0][0], rel[i][0][1])] + [(f"soup:{b}", t, sc) for b, (t, sc) in enumerate(soup[i])]
            items += [(f"{s}:0", seeds[s][i][0][0], seeds[s][i][0][1]) for s in sd]
            if i in gext:              # the glossary's own wider decode of an armed piece (24 Oct)
                items += [(f"soup:g{b}", t, sc) for b, (t, sc) in enumerate(gext[i])]
            if i in ni:                 # the school-spelling reading of phone-typed Tonga (tonga_spelling)
                j = ni.index(i)
                items += [("norm:soup:b4", nrel[j][0][0], nrel[j][0][1])]
                items += [(f"norm:soup:{b}", t, sc) for b, (t, sc) in enumerate(nsoup[j])]
                items += [(f"norm:{s}:0", nseeds[s][j][0][0], nseeds[s][j][0][1]) for s in sd]
            if i in gi:                    # the top beam loops ('SOMETHING SOMETHING ...'): a no-repeat decode beside it
                items.append(("soup:guard", guard[gi.index(i)][0][0], guard[gi.index(i)][0][1]))
            # names: E>T a name the soup loses gets the stand-in route (h2_text.safe_en); T>E names.safe_translate
            if direction == "en>toi" and T.en_names(q["text"]):
                nm = T.en_names(q["text"])
                if any(n.lower() not in soup[i][0][0].lower() for n in nm):
                    t2, rep = T.safe_en(q["text"], lambda x: self.nbest("soup", [x], direction, 1)[0][0][0])
                    items.append(("soup:names", t2, None))
            elif direction == "toi>en":
                import names as N
                spans = [s for s in N.tag(q["text"]) if s["type"] in ("PERSON", "PLACE", "ORG")]
                if spans:
                    t2, rep = N.safe_translate(q["text"], lambda x: self.nbest("soup", [x], direction, 1)[0][0][0])
                    items.append(("soup:names", t2, None))
                    q["names"] = sorted({s["text"] for s in spans})
            for src, t, sc in items:
                key = t.strip()
                if key in seen:
                    cs[seen[key]]["also"].append(src)
                    continue
                seen[key] = len(cs)
                cs.append({"src": src, "text": t, "score": sc, "also": []})
            # step 6d (24 Oct): near memory rows, LAST in the pool (cs[0] stays the incumbent); a target the model
            # already wrote is only noted on that beam ('also'), never a second candidate
            for src, t, info in self.tm_near_for(q, direction):
                key = t.strip()
                if key in seen:
                    cs[seen[key]]["also"].append(src)
                    continue
                seen[key] = len(cs)
                cs.append({"src": src, "text": t, "score": None, "also": [], "tm": info})
            allc.append(cs)
        return allc

    def tm_near_for(self, q, direction):
        """[(src, target, info)] near memory rows for one piece (step 6d; module docstring). Empty when the lever or
        the memory is off, the piece carries stand-ins (a memory line cannot carry them: it would fail `lost`), or the
        piece is guarded. The source side AND the target of every row are guarded too, and a target carrying a word the
        term guard forbids for this input is dropped (the guard acts at decode; a memory line never passed through it)."""
        import h2_memory as M
        if not (self.cfg.get("tm_near", True) and self.cfg.get("memory", True)) or q.get("_mem_off"):
            return []
        if any(k != "__unprotected__" for k in (q.get("map") or {})):
            return []
        nw = len(M.norm(q["src"]).split())
        rules = []
        if self.cfg.get("term_guard", True):
            import term_guard as TG
            rules = TG.rules_for(f"{getattr(self._tl, 'ctx', None) or ''} {q['src']}", direction)
        strict = self.cfg.get("tm_near_strict", True)
        out, seen = [], set()
        for t in [q["src"]] + ([q["norm"]] if q.get("norm") else []):     # T>E: the school-spelling reading too
            if self.blocked(t):
                return []
            for h in M.near(t, direction):
                r = h["row"]
                if h["differs"] or r.get("kind") not in ("teacher", "thompson", "book"):
                    continue
                if r.get("kind") == "book" and nw < 3:
                    continue
                if strict and not h["strict"]:
                    continue
                if self.blocked(r["en"]) or self.blocked(r["toi"]):
                    continue
                if not same_kind(r["en"] if direction == "en>toi" else r["toi"], t):
                    continue                  # a question for a statement (norm folds '?'; as in memory())
                if rules:
                    import term_guard as TG
                    if TG.bad(h["target"], rules):
                        continue
                tgt = strip_label(q["src"], h["target"]).strip()      # a book row carries its page's label
                if not tgt or tgt in seen:
                    continue
                seen.add(tgt)
                info = {"ratio": h["ratio"], "rword": h["rword"], "strict": h["strict"], "kind": r.get("kind"),
                        "src": r.get("src", "")[:160], "near": r["en"] if direction == "en>toi" else r["toi"]}
                out.append((f"tm-near {h['ratio']} {r.get('src', '')[:70]}", tgt, info))
                if len(out) >= int(self.cfg.get("tm_near_k", 2)):
                    return out
        return out

    def check(self, prep, allc, direction):
        """fills every candidate's facts: copy, lost, hard, attest_w, dm, meaning, agree, why.
        24 Oct 2026 (step 6d): the tm-near candidates are checked APART from the model's pool, never inside it: the
        readers' int8 batches and the pool-level agreement both depend on what else is in the pool (22 Oct: padding
        changes neighbours), so a memory line in the same batch could move the facts of the model's candidates and
        change an output it does not win. Apart, the model's candidates get exactly the facts they had before. Each
        memory line is checked ALONE as well (one pool of one per layer), so its facts do not depend on which other
        memory lines were offered: the strict and loose gates can be compared by re-deciding one run (decide is pure)."""
        base = [[c for c in cs if not c.get("tm")] for cs in allc]
        self._check(prep, base, direction)
        tms = [[c for c in cs if c.get("tm")] for cs in allc]
        for k in range(max((len(x) for x in tms), default=0)):
            self._check(prep, [[cs[k]] if k < len(cs) else [] for cs in tms], direction, ref=base)

    def _check(self, prep, allc, direction, ref=None):
        """the facts of one pool per piece; ref = the model's pools when `allc` holds tm-near candidates (E>T: their
        own reader check, agreement left out; T>E: 'meaning' = mean cosine to the model's candidates)"""
        import h3_detect as D
        import term_guard as TG
        if direction == "en>toi":
            import h3_select as S
            for q, cs in zip(prep, allc):
                if not cs:
                    continue
                pool = [{"src": c["src"], "answer": c["text"], "thought": ""} for c in cs]
                facts = S.check_pool(pool, q["text"], "translate", readers=self.available()[0], certainty=False)
                for c, d in zip(cs, pool):
                    if not d.get("usable"):
                        c.update(copy=True, why=["empty"], meaning=0.0)
                        continue
                    # the stand-ins are ours, not the model's Tonga: never an attestation reason (Kim read as Nyanja)
                    st = {k.lower() for k in q["map"]}
                    d["checks"]["attest"] = [x for x in d["checks"].get("attest", []) if str(x[1]).lower() not in st]
                    iss = S.issues(d)
                    m = d.get("meaning") or {}
                    dm = D.dict_missing(q["text"], c["text"], "en>toi")
                    ls = lost(c["text"], q) + (["blank"] if len(BLANK.findall(q["text"])) > len(BLANK.findall(c["text"])) else [])
                    # the teacher glossary (24 Oct): a fact of its own, NEVER folded into `hard`, so a line with no
                    # armed term decides exactly as it did before. The alarm rides on the candidate, so it reaches
                    # out['alarms'] only if the chosen output is the one that misses the term.
                    gmiss = TG.gloss_missing(c["text"], q["_gterms"]) if q.get("_gterms") else []
                    c.update(copy=is_copy(q["text"], c["text"]) or has_loop(c["text"], q["text"]), lost=len(ls),
                             hard=len(iss["hard"]) + len(iss["block"]),
                             attest_w=iss["attest_w"], dm=len(dm), meaning=m.get("median"), agree=d.get("agreement"),
                             readers=m.get("readers"), gmiss=len(gmiss),
                             gloss_miss=[t["term"] for t in gmiss])
                    c["why"] = (["loop"] if has_loop(c["text"], q["text"]) else ["copy_through"] if c["copy"] else []) + \
                        [f"notation_lost: {x}" for x in ls] + iss["block"] + iss["hard"] + [a for a in iss["attest"] if re.search(r"not_tonga|contact|mixed", a)] + \
                        [f"dict_missing: {x[1]}" for x in dm] + TG.gloss_alarm(gmiss)
                    if ref is not None:
                        c["agree"] = None        # a memory line is checked alone: no agreement (a tie-break only)
                if ref is None:
                    q["spread"] = facts.get("spread")
            return
        import h2_checks as C
        import h3_select as S
        for pi, (q, cs) in enumerate(zip(prep, allc)):
            if not cs:
                continue
            V = S.embed([c["text"] for c in cs])
            n = len(cs)
            R = S.embed([c["text"] for c in ref[pi]]) if ref is not None and ref[pi] else None
            for i, c in enumerate(cs):
                if R is not None:                # a memory line against the model's candidates (step 6d)
                    util = float(sum(V[i] @ R[j] for j in range(len(R))) / len(R))
                else:
                    util = float(sum(V[i] @ V[j] for j in range(n) if j != i) / max(1, n - 1)) if n > 1 else 1.0
                neg = [x[0] for x in D.negation3(c["text"], q["text"]) if x[0] in ("neg_dropped", "neg_added")]
                num = [x[0] for x in C.numbers(c["text"], q["text"]) if x[0] in ("num_missing", "num_added")]
                nm = [n_ for n_ in q.get("names", []) if not re.search(r"(?<![A-Za-z])" + re.escape(n_) + r"(?![A-Za-z])", c["text"])]
                dm = D.dict_missing(q["text"], c["text"], "toi>en")
                ls = lost(c["text"], q) + (["blank"] if len(BLANK.findall(q["text"])) > len(BLANK.findall(c["text"])) else [])
                c.update(copy=is_copy(q.get("norm") or q["text"], c["text"]) or is_copy(q["text"], c["text"])
                         or has_loop(c["text"], q["text"]), lost=len(ls), hard=len(neg) + len(num) + len(nm), attest_w=0,
                         dm=len(dm), meaning=round(util, 3), agree=round(util, 3))
                c["why"] = (["loop"] if has_loop(c["text"], q["text"]) else ["copy_through"] if c["copy"] else []) + \
                    [f"notation_lost: {x}" for x in ls] + [f"negation: {x}" for x in neg] + [f"numbers: {x}" for x in num] + [f"name_missing: {x}" for x in nm] + \
                    [f"dict_missing: {x[1]}" for x in dm]

    def plain(self, text, direction):
        """R0 as the comparisons ran it: the soup, beam 4, the whole input, no normalisation"""
        r = self.reader("soup")
        src, tgt = ("en", "toi") if direction == "en>toi" else ("toi", "en")
        with self._lock_for("soup"):
            return r._run([text], LANG[src], LANG[tgt], 256)[0]


_LOCKS, _LOCKS_GUARD = {}, threading.Lock()


def detect_direction(text):
    """'toi>en' when the text reads as Tonga (most words known to attest and few English), else 'en>toi'"""
    import attest
    ws = re.findall(r"[A-Za-zŋŊ']+", text or "")
    if not ws:
        return "en>toi"
    eng = sum(1 for v, w, _ in attest.line(text) if v == "english")
    voc = _en_vocab()
    en_like = sum(1 for w in ws if w.lower() in voc)
    return "toi>en" if eng < 0.3 * len(ws) and en_like < 0.5 * len(ws) else "en>toi"


_svc = {}


def service(cfg=None):
    k = json.dumps(cfg or {}, sort_keys=True, default=str)
    if k not in _svc:
        _svc[k] = Service(cfg)
    return _svc[k]


def main():
    a = sys.argv[1:]
    d = a[a.index("--dir") + 1] if "--dir" in a else "auto"
    text = a[0]
    s = service()
    if "--plain" in a:
        print(s.plain(text, d if d != "auto" else detect_direction(text)))
        return
    r = s.translate(text, d, keep="--json" in a)
    if "--json" in a:
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    else:
        print(r["text"])
        print(f"[{r['direction']} {r['stage']} {r['t']}s switched {r.get('switched', 0)} alarms {r['alarms']}]")


if __name__ == "__main__":
    main()
