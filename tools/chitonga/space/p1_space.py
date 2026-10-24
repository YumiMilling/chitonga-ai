"""P1 for the public translator page (21 Oct 2026; v2 after the Codex review of the same day,
internal council/codex/2026-10-21_1038_review_p1.md). The decoding profile of tools/chitonga/llm/translator_service.py,
cut down to what needs NO private data, because this Space is public: no translation memory, no lexicon, no word lists
from the books, no held-out sets, no keys. Everything here is code and the model's own beams.

Per input sentence (the page rates sentence by sentence):
  layout      line breaks (CR, LF, CRLF) and table cells (' | ') kept; an exercise label (1. a) (b) D ...) and an end mark
              ((2), [8], (maaka 2) <> (2 marks)) cut off and put back; a label the model invents is taken off again,
              never a number the source has
  protect     morpheme notation (u–la–bona, –la–, -ide) and quoted single letters / numbers ("D") are replaced by stand-in
              names, BEFORE any case change; every occurrence counted; stand-ins never collide with a word of the input
              (case-folded); one whole-token matcher for checking and restoring
  caps        an ALL-CAPS line (no Titlecase word in it) is read in sentence case and given back in capitals; the
              protected spans come back as typed
  phone form  a lower-case start of 2+ words gets a capital and a full stop for reading (the stop is taken off again)
  spelling    Tonga input only: a second reading in the school-book spelling from a rules table (ch > c, ng' > ŋ,
              reduplication written apart > hyphenated), on lower-case words only (names untouched); names are checked
              against the ORIGINAL input
  split       sentences (abbreviations and labels respected), then commas above 25 words, then words above MAX_IN_TOKENS
              sentencepiece tokens: no piece is ever cut by the decoder's input limit
  candidates  full mode: the released decode (beam 4) as the incumbent + the beam-5 5-best + a no-repeat decode when a beam
              loops; fast mode (time budget): beam 4 only, with the same preprocessing, restoring and checks
  checks      empty output, output cut off at the decode limit, copy-through, a loop (hyphenated reduplication and
              repetition the source has are not loops), protected span lost or miscounted, numbers changed (counted
              expressions, signs and separators kept), English left in the Tonga. The incumbent is replaced by the
              candidate with the fewest failures (ties: the model's order) only if that candidate adds no new kind of
              failure; the chosen candidate's own alarms are reported. A lost name is an alarm only.
terms       v3 (24 Oct 2026): a forbid-only term check from PUBLIC sources (dictionaries, our own test logs; none of the
              rules mined from teachers' corrections, no teacher glossary): when the source triggers a rule and the
              output carries the forbidden form, that is a failure kind ("term") and a candidate without it wins if it
              adds no new kind of failure. It never writes a word.
time        v3: when the incumbent decode alone used so much time that the n-best would pass the deadline, the sentence
              goes to fast mode (incumbent + checks) instead of answering late (a050 measured 28 s on one long line).
Left out (they need private data, or were measured to add nothing): the translation memory, the dictionary check
(dict_missing), attestation, negation (its tables read the private lexicon; NEGATION IS NOT CHECKED HERE), the glossary,
the back-translation readers, the two seed models, the consensus switch.
"""
import collections
import re
import time

LANG = {"toi": "toi_Latn", "en": "eng_Latn"}
STAND = ["Kim", "Sam", "Jordan", "Grace", "Robin", "Jamie", "Casey", "Morgan", "Riley", "Quinn"]
MAX_IN_TOKENS = 200          # sentencepiece tokens per piece (CT2 would silently truncate at 1,024; NLLB degrades far earlier)
MAX_OUT_FACTOR, MAX_OUT_MIN, MAX_OUT_CAP = 3, 64, 512
ENUM = re.compile(r"^(\s*(?:\d{1,2}[.)]|\d{1,2}(?=\s+[A-Z])|[a-hA-H][.)]|\((?:[a-hA-H]|[ivx]{1,4}|\d{1,2})\)|"
                  r"[ivx]{1,4}[.)]|[•*·▪]|[-–](?=\s))\s+|\s*[A-E]\s+(?=[A-Z][a-z]))")
MARK_END = re.compile(r"(\s*(?:\(\s*(?:maaka\s*)?\d{1,3}(?:\s*marks?)?\s*\)|\[\s*\d{1,3}\s*\]))\s*$", re.I)
CELL = re.compile(r"(\s+\|\s+)")
NOTATION = re.compile(r"(?<![\w–-])(?:[–-]?[a-zŋ']{1,12}(?:[–-][a-zŋ']{1,12})*[–-]?)(?![\w–-])", re.I)
QUOTE = re.compile(r"(?<!\w)([\"“‘'])([^\"“”‘’'\n]{1,40})([\"”’'])(?!\w)")
BLANK = re.compile(r"_{3,}|\.{4,}|…{2,}")
TOKEN = re.compile(r"[^\W\d_]+(?:[-–'’][^\W\d_]+)*", re.U)     # a word; hyphenated reduplication is ONE token
NUMEXP = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,/:–-]\d+)*")
ABBR = {"mr", "mrs", "ms", "dr", "st", "no", "fr", "sr", "e.g", "i.e", "vs", "etc", "p", "pp", "nb", "prof", "rev"}
MAX_WORDS = 25
NBEST_COST, DEADLINE_MARGIN = 1.6, 3.0     # v3 time rule (see translate_many)
NOT_NAMES = {"I", "God", "Lord", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January",
             "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
             "December", "English", "Chitonga", "Tonga", "Mr", "Mrs", "Ms", "Dr", "Unit", "Grade", "Lesson", "Chapter",
             "Term", "OK", "AIDS", "HIV", "Doctor", "Teacher", "Mother", "Father", "Cholera"}
EN_LEAK = re.compile(r"(?i)(?<![^\W\d_])(i[’']m|it[’']s|don[’']t|can[’']t|i[’']ll|you[’']re|the|is|are|you|your|my|and|"
                     r"this|that|with|what|of|for|he|she|they|was|were|will|not|have|has)(?![^\W\d_])")


# v3, 24 Oct 2026: the public-safe subset of tools/chitonga/llm/term_guard.py (sources: dictionaries and our own test
# logs; the rules mined from teachers' corrections and the teacher glossary stay on the Beelink). "dir" = direction.
TERMS = [
    ("cholera_not_corona", "en>toi", r"cholera", r"corona|covid", r"k+o[lr]ona"),           # kkolela is the poster form
    ("dizzy_not_diarrhoea", "en>toi", r"(?<![A-Za-z])dizz(?:y|iness|ier|iest)(?![A-Za-z])",
     r"diarrh|dysenter|loose stool|runny stomach|running stomach", r"soomon|somoon"),      # Thompson 1979 p85
    ("nails_not_buttocks", "en>toi", r"(?<![A-Za-z])(?:finger|toe)?nails?(?![A-Za-z])",
     r"(?<![A-Za-z])(?:hammer|buttock|nail (?:it|them|the)\b)", r"(?<![a-z])(?:ma)?tako(?![a-z])"),
    ("kkolona_not_cholera", "toi>en", r"k+o[lr]ona", r"k+olela|kolera|cholera", r"cholera"),
    ("mala_not_buttocks", "toi>en", r"(?<![a-z])mala(?![a-z])", r"buttock|matako", r"buttock"),  # Thompson 1979 p82
]
TERMS = [(n, d, re.compile(w, re.I), re.compile(u, re.I), re.compile(f, re.I)) for n, d, w, u, f in TERMS]


def term_fails(src, out, direction):
    """the public term rules this source triggers whose forbidden form is in the output"""
    return [n for n, d, w, u, f in TERMS if d == direction and w.search(src or "") and not u.search(src or "")
            and f.search(out or "")]


def lines_of(text):
    """CR, LF and CRLF are all line breaks (a lone CR is never deleted)"""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


# ------------------------------------------------------------------------------------------------ the one matcher
def _ph_rx(ph):
    """the ONE whole-token, case-insensitive, Unicode-aware matcher of a stand-in (checking and restoring use it)"""
    return re.compile(r"(?<![^\W_])" + re.escape(ph) + r"(?:['’]s)?(?![^\W_])", re.I)


def ph_count(text, ph):
    return len(_ph_rx(ph).findall(text or ""))


# ------------------------------------------------------------------------------------------------ layout
def is_notation(tok):
    if not NOTATION.fullmatch(tok or "") or not re.search(r"[–-]", tok):
        return False
    if "–" in tok or tok.startswith("-") or tok.endswith("-"):
        return True
    segs = [s for s in tok.lower().split("-") if s]
    return tok.count("-") >= 2 and len(set(segs)) > 1


def is_caps(s):
    """an ALL-CAPS line: 80%+ of letters upper case, no Titlecase word (a name typed normally, 'PLEASE GREET John.'),
    not an acronym alone"""
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 4 or sum(c.isupper() for c in letters) < 0.8 * len(letters):
        return False
    if any(len(w) >= 2 and w[0].isupper() and w[1:].islower() for w in TOKEN.findall(s)):
        return False
    return not re.fullmatch(r"\s*[A-Z]{2,5}\s*", s)


def sentence_case(s, keep=()):
    s = re.sub(r"(^|[.!?]\s+)([^\W\d_])", lambda m: m.group(1) + m.group(2).upper(), s.lower())
    for ph in keep:                                  # the stand-ins stay names (Kim, not kim)
        s = _ph_rx(ph).sub(ph, s)
    return s


def sentences(s):
    """sentence pieces: a cut after . ! ? before a capital, a digit, a quote or a lower-case letter, never after an
    abbreviation, a single letter (initials, option letters) or an exercise number"""
    out, cur = [], ""
    parts = re.split(r"(?<=[.!?])([\"”’)]?)\s+(?=[\"“‘(]?[^\W_])", s)
    i = 0
    while i < len(parts):
        seg = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        i += 2
        cur = (cur + " " + seg).strip() if cur else seg
        last = re.findall(r"([^\W\d_.]+(?:\.[^\W\d_]+)*)[.!?][\"”’)]?$", cur)
        if last and (last[0].lower() in ABBR or len(last[0]) == 1):
            continue
        if re.fullmatch(r"\s*\d{1,2}[.)]?", cur):
            continue
        out.append(cur)
        cur = ""
    if cur:
        out.append(cur)
    return [p for p in out if p.strip()]


def clauses(text, max_words=MAX_WORDS):
    if len(text.split()) <= max_words:
        return [text]
    out, cur = [], ""
    for c in re.split(r"(?<=,)\s+", text):
        if cur and len(cur.split()) + len(c.split()) > max_words:
            out.append(cur)
            cur = c
        else:
            cur = (cur + " " + c).strip()
    if cur:
        out.append(cur)
    return out


def by_tokens(text, ntok, limit=MAX_IN_TOKENS):
    """a piece still above `limit` sentencepiece tokens is cut at word boundaries (ntok counts a string's tokens)"""
    if ntok(text) <= limit:
        return [text]
    out, cur = [], []
    for w in text.split():
        if cur and ntok(" ".join(cur + [w])) > limit:
            out.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def mark_to(m, direction):
    n = re.search(r"\d+", m).group(0)
    lead = re.match(r"\s*", m).group(0)
    if direction == "toi>en" and re.search(r"maaka", m, re.I):
        return f"{lead}({n} mark{'s' if n != '1' else ''})"
    if direction == "en>toi" and re.search(r"marks?", m, re.I):
        return f"{lead}(maaka {n})"
    return m


def layout(text, direction, ntok=None):
    """(plan, pieces): plan = literals and piece indices; pieces = [{text (protected), src (as typed), caps, map}].
    Protection happens before any case change, so a protected span comes back exactly as typed."""
    plan, pieces = [], []
    for li, line in enumerate(lines_of(text)):
        if li:
            plan.append("\n")
        for cell in CELL.split(line):
            if CELL.fullmatch(cell):
                plan.append(cell)
                continue
            if not cell.strip():
                plan.append(cell)                    # a whitespace-only cell, kept once
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
            lead = re.match(r"\s*", cell).group(0)
            trail = re.search(r"\s*$", cell).group(0)
            core = cell.strip()
            if lead:
                plan.append(lead)
            if not re.search(r"[^\W\d_]", core) or all(is_notation(t) or BLANK.fullmatch(t) for t in core.split()):
                plan.append(core)
            else:
                caps = is_caps(core)
                prot, mp = protect(core)
                body = sentence_case(prot, keep=mp) if caps else prot
                first = True
                for snt in sentences(body):
                    for p in clauses(snt):
                        for q in (by_tokens(p, ntok) if ntok else [p]):
                            if not first:
                                plan.append(" ")
                            first = False
                            plan.append(len(pieces))
                            sub = {ph: o for ph, o in mp.items() if ph_count(q, ph)}
                            pieces.append({"text": q, "caps": caps, "map": sub,
                                           "counts": {ph: ph_count(q, ph) for ph in sub}})
            if trail:
                plan.append(trail)
            if post:
                plan.append(post)
    return plan, pieces


def assemble(plan, outs):
    return "".join(x if isinstance(x, str) else outs[x] for x in plan)


# ------------------------------------------------------------------------------------------------ protect
def protect(text):
    """notation tokens and quoted single letters / numbers > stand-ins (one stand-in per distinct span, every
    occurrence counted); a stand-in whose name occurs in the text in any case is never used. (text, {stand-in: span})"""
    low = (text or "").lower()
    pool = [s for s in STAND if s.lower() not in low]
    mapping = {}

    def take(orig):
        for ph, o in mapping.items():
            if o == orig:
                return ph
        if not pool:
            mapping.setdefault("__unprotected__", [])
            mapping["__unprotected__"].append(orig)
            return None
        ph = pool.pop(0)
        mapping[ph] = orig
        return ph
    text = QUOTE.sub(lambda m: (take(m.group(0)) or m.group(0)) if re.fullmatch(r"\s*([^\W\d_]|\d+)\s*", m.group(2))
                     else m.group(0), text)
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
    for ph, orig in sorted(((p, o) for p, o in mapping.items() if p != "__unprotected__"), key=lambda x: -len(x[0])):
        text = _ph_rx(ph).sub(lambda _: orig, text)
    return text


def lost(text, q):
    """stand-ins whose COUNT in the output differs from the source piece"""
    return [ph for ph, n in q.get("counts", {}).items() if ph_count(text, ph) != n]


def phone_form(s):
    if not s or not s[0].islower() or len(s.split()) < 2:
        return s, False
    s = s[0].upper() + s[1:]
    if s[-1] not in ".?!\"'”’:;)":
        return s + ".", True
    return s, False


def strip_label(src, out):
    """a label the model invented is taken off; never one whose number or letter the source piece has (source
    '12 children' > '12 Children.' keeps its 12)"""
    if ENUM.match(src or ""):
        return out
    m = ENUM.match(out or "")
    while m and m.end() < len(out):
        lab = re.sub(r"[^\w]", "", m.group(0))
        if lab and re.search(r"(?<![\w])" + re.escape(lab) + r"(?![\w])", src or "", re.I):
            break
        out = out[m.end():]
        m = ENUM.match(out)
    return out


def render(q, text):
    txt = text.upper() if q["caps"] else text
    txt = restore(txt, q["map"])
    txt = strip_label(q["src"], txt)
    if q["stop_added"] and txt.endswith("."):
        txt = txt[:-1]
    return txt


# ------------------------------------------------------------------------------------------------ spelling (rules only)
SPELL_TOKEN = [(re.compile(r"ch"), "c", "ch > c"), (re.compile(r"ng['’]"), "ŋ", "ng' > ŋ")]


def spelling(text):
    """(text in the school spelling, [rules]): token rules on LOWER-CASE tokens only (a capitalised word may be a name:
    Chang'a stays Chang'a), reduplication only between two lower-case tokens"""
    used = []

    def tok(m):
        w = m.group(0)
        if not w[:1].islower():
            return w
        for rx, to, name in SPELL_TOKEN:
            w2 = rx.sub(to, w)
            if w2 != w:
                used.append(name)
                w = w2
        return w
    out = re.sub(r"[^\W\d_]+(?:['’][^\W\d_]+)*", tok, text)

    def red(m):
        a, b = m.group(1), m.group(2)
        if a[:1].islower() and b[:1].islower() and len(b) >= 3 and a.endswith(b) and a != b:
            used.append("reduplication hyphenated")
            return a + "-" + b
        return m.group(0)
    out = re.sub(r"(?<![^\W\d_-])([^\W\d_]+) ([^\W\d_]+)(?![^\W\d_-])", red, out)
    return out, sorted(set(used))


# ------------------------------------------------------------------------------------------------ checks
def is_copy(src, out, stand=()):
    st = {s.lower() for s in stand}
    a = [w.lower() for w in TOKEN.findall(src or "") if w.lower() not in st]
    b = [w.lower() for w in TOKEN.findall(out or "") if w.lower() not in st]
    if len(a) < 2 or not b:
        return False
    sa, sb = set(a), set(b)
    return len(sa & sb) >= 0.7 * len(sa) and len(sa & sb) >= 0.7 * len(sb)


def has_loop(text, src=""):
    """a decoding loop: a token three times in a row, a token trigram three times, or the whole output said twice;
    hyphenated reduplication (kati-kati) is one token, and a repetition the source itself has is not a loop"""
    ws = [w.lower() for w in TOKEN.findall(text or "")]
    sw = [w.lower() for w in TOKEN.findall(src or "")]

    def runs(x):
        return any(x[i] == x[i + 1] == x[i + 2] for i in range(len(x) - 2))

    def tri(x):
        c = collections.Counter(tuple(x[i:i + 3]) for i in range(len(x) - 2))
        return bool(c) and max(c.values()) >= 3

    def twice(x):
        h = len(x) // 2
        return bool(x) and len(x) % 2 == 0 and x[:h] == x[h:]
    return (runs(ws) and not runs(sw)) or (tri(ws) and not tri(sw)) or (twice(ws) and not twice(sw))


def names_of(text):
    out = []
    for m in TOKEN.finditer(text or ""):
        w = m.group(0)
        before = (text or "")[:m.start()].rstrip()
        initial = not before or before[-1] in ".!?:;\"“‘(—–|" or re.fullmatch(r".*\b\d+[.)]?", before or "")
        if w[:1].isupper() and not initial and len(w) >= 3 and w not in NOT_NAMES and not w.isupper() \
                and w not in STAND:
            out.append(w)
    return out


def numbers(text):
    return collections.Counter(x.replace("−", "-") for x in NUMEXP.findall(text or ""))


def hard(src_orig, src_read, out, direction, q, cut=False):
    """the checks that need no private data: [reason]. src_orig = the input as typed (names, numbers),
    src_read = what the model read (copy-through)"""
    if not re.search(r"[^\W_]", out or ""):
        return ["empty"]
    why = []
    if cut:
        why.append("cut_off")
    if has_loop(out, src_read):
        why.append("loop")
    elif is_copy(src_read, out, q["map"]):
        why.append("copy_through")
    why += [f"notation_lost: {q['map'][ph]}" for ph in lost(out, q)]
    if len(BLANK.findall(src_read)) != len(BLANK.findall(out)):
        why.append("blank_changed")
    ns, no = numbers(src_orig), numbers(out)
    if ns != no:
        why.append(f"numbers_changed: {dict(ns - no) or ''} {dict(no - ns) or ''}".strip())
    outtok = {w.lower() for w in TOKEN.findall(out)}
    for n in names_of(src_orig):
        if n.lower() not in outtok:
            why.append(f"name_lost: {n}")
    if direction == "en>toi":
        names = {n.lower() for n in names_of(src_orig)}
        eng = [x for x in EN_LEAK.findall(out) if x.lower() not in names]
        if eng:
            why.append("english_left: " + ", ".join(sorted({x.lower() for x in eng})))
    why += [f"term: {n}" for n in term_fails(src_orig, out, direction)]
    return why


TIER = {"empty": 0, "cut_off": 1, "loop": 2, "copy_through": 2}


def kinds(f):
    return {x.split(":")[0] for x in f if not x.startswith("name_lost")}


def _key(f):
    """the failures that switch, worst first: empty, cut off, loop / copy, a public term rule (v3), then preservation
    (notation, blank, numbers), then English left; a lost name never switches"""
    k = kinds(f)
    return (int("empty" in k), int("cut_off" in k), int(bool(k & {"loop", "copy_through"})), int("term" in k),
            sum(x in k for x in ("notation_lost", "blank_changed", "numbers_changed")), len(k))


def _agree(a, others):
    def g(s):
        s = " " + re.sub(r"\s+", " ", (s or "").lower()) + " "
        return {s[i:i + 3] for i in range(len(s) - 2)}
    ga = g(a)
    vals = [2 * len(ga & g(o)) / max(1, len(ga) + len(g(o))) for o in others]
    return sum(vals) / len(vals) if vals else 0.0


# ------------------------------------------------------------------------------------------------ the translator
class Busy(Exception):
    """the shared model was not free before the deadline: the caller answers 503, never a late or partial result"""


class P1:
    def __init__(self, tr, sp, lock=None, nbest=5):
        self.tr, self.sp, self.lock, self.nbest = tr, sp, lock, nbest

    def ntok(self, s):
        return len(self.sp.encode(s, out_type=str))

    def _decode(self, texts, src, tgt, beam, k, guard=False, deadline=None):
        """[[(text, score, cut)]] ; the lock is taken with a timeout and the deadline checked after it (Busy)"""
        if not texts:
            return []
        toks = [[LANG[src]] + self.sp.encode(t, out_type=str) + ["</s>"] for t in texts]
        lim = min(MAX_OUT_CAP, max(MAX_OUT_MIN, MAX_OUT_FACTOR * max(len(t) for t in toks)))
        kw = {"no_repeat_ngram_size": 3, "repetition_penalty": 1.2} if guard else {}
        if self.lock is not None:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not self.lock.acquire(timeout=wait if wait is not None else -1):
                raise Busy("model busy")
        try:
            if deadline is not None and time.monotonic() > deadline:
                raise Busy("deadline passed while waiting")
            res = self.tr.translate_batch(toks, target_prefix=[[LANG[tgt]]] * len(toks), beam_size=beam,
                                          num_hypotheses=k, max_decoding_length=lim, max_input_length=MAX_IN_TOKENS + 8,
                                          return_scores=True, **kw)
        finally:
            if self.lock is not None:
                self.lock.release()
        out = []
        for x in res:
            hyps = []
            for h, s in zip(x.hypotheses, x.scores):
                body = [w for w in h if w not in (LANG[tgt], "</s>")]
                hyps.append((self.sp.decode(body), round(s, 3), len(h) >= lim - 1))
            out.append(hyps)
        return out

    def prepare(self, text, src):
        direction = "toi>en" if src == "toi" else "en>toi"
        plan, pieces = layout(text, direction, ntok=self.ntok)
        prep = []
        for p in pieces:
            nt, used = spelling(p["text"]) if direction == "toi>en" else (p["text"], [])
            pt, stop = phone_form(p["text"])
            q = dict(p, src=p["text"], text=pt, stop_added=stop, orig=restore(p["text"], p["map"]))
            if used:
                q.update(norm=phone_form(nt)[0], rules=used)
            prep.append(q)
        return plan, prep

    def translate_many(self, texts, src, fast=False, deadline=None):
        """[{'out', 'score', 'switched', 'why', 'reading', 'mode'}] for several sentences, every one of them (fast: one
        batched beam-4 decode, the same preprocessing, restoring and checks)"""
        tgt = "en" if src == "toi" else "toi"
        direction = "toi>en" if src == "toi" else "en>toi"
        jobs = [self.prepare(t, src) for t in texts]
        allp = [q for _, prep in jobs for q in prep]
        texts_ = [q["text"] for q in allp]
        t_rel = time.monotonic()
        rel = self._decode(texts_, src, tgt, 4, 1, deadline=deadline)
        t_rel = time.monotonic() - t_rel
        # v3: the beam-5 5-best costs ~1.5x the incumbent; if that would pass the deadline (with the norm reading and a
        # guard decode still possible), answer in fast mode instead of late
        if not fast and deadline is not None and time.monotonic() + NBEST_COST * t_rel + DEADLINE_MARGIN > deadline:
            fast = True
        nb = [] if fast else self._decode(texts_, src, tgt, max(4, self.nbest), self.nbest, deadline=deadline)
        ni = [i for i, q in enumerate(allp) if q.get("norm")]
        nrel = self._decode([allp[i]["norm"] for i in ni], src, tgt, 4, 1, deadline=deadline) if ni else []
        nnb = [] if fast or not ni else self._decode([allp[i]["norm"] for i in ni], src, tgt, max(4, self.nbest),
                                                     self.nbest, deadline=deadline)
        gi = [i for i in range(len(allp)) if has_loop(rel[i][0][0], allp[i]["text"])
              or (nb and has_loop(nb[i][0][0], allp[i]["text"]))]
        gd = self._decode([texts_[i] for i in gi], src, tgt, 4, 1, guard=True, deadline=deadline) if gi else []
        results, k0 = [], 0
        for plan, prep in jobs:
            outs, scores, sw, why, reading = [], [], 0, [], None
            for q in prep:
                i = k0
                k0 += 1
                cands = [("b4",) + rel[i][0]] + ([] if fast else [(f"beam{b}",) + h for b, h in enumerate(nb[i])])
                if i in ni:
                    j = ni.index(i)
                    norm = [("norm:b4",) + nrel[j][0]] + ([] if fast else [(f"norm:beam{b}",) + h for b, h in enumerate(nnb[j])])
                    allt = [c[1] for c in cands + norm]
                    ua = _agree(allt[0], allt[1:])
                    ub = _agree(allt[len(cands)], allt[:len(cands)] + allt[len(cands) + 1:])
                    if ub >= ua:
                        cands, reading = norm + cands, "school spelling (" + ", ".join(q["rules"]) + ")"
                    else:
                        cands, reading = cands + norm, "as typed"
                if i in gi:
                    cands.append(("guard",) + gd[gi.index(i)][0])
                seen, uniq = set(), []
                for c in cands:
                    if c[1].strip() not in seen:
                        seen.add(c[1].strip())
                        uniq.append(c)
                facts = [hard(q["orig"], q.get("norm") if c[0].startswith("norm") else q["text"], c[1], direction, q,
                              cut=c[3]) for c in uniq]
                k = 0
                if _key(facts[0]) > (0, 0, 0, 0, 0, 0):
                    k0kinds = kinds(facts[0])
                    ok = [j for j in range(1, len(uniq)) if _key(facts[j]) < _key(facts[0])
                          and kinds(facts[j]) <= k0kinds]              # never a new kind of failure
                    if ok:
                        k = min(ok, key=lambda j: (_key(facts[j]), j))
                        sw += 1
                        why.append("replaced: " + ", ".join(facts[0]))
                why += [x for x in facts[k]]
                if q["map"].get("__unprotected__"):
                    why.append(f"unprotected (no stand-in left): {q['map']['__unprotected__']}")
                outs.append(render(q, uniq[k][1]) if uniq[k][1].strip() else "")
                scores.append(uniq[k][2])
            sc = [s for s in scores if s is not None]
            results.append({"out": assemble(plan, outs), "score": round(sum(sc) / len(sc), 3) if sc else None,
                            "switched": sw, "why": why, "reading": reading, "mode": "fast" if fast else "full",
                            "pieces": len(prep)})
        return results

    def translate(self, text, src, fast=False, deadline=None):
        return self.translate_many([text], src, fast=fast, deadline=deadline)[0]
