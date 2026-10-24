"""Harness v3, step 1: new detectors (19 Oct 2026, the Jesuit; HARNESS_V3_PLAN.md §1.1, §1.3, §2.2, §3).

Every detector returns a list of (key, detail, evidence) issues; empty = nothing found. They feed selection (h3_select)
and the repairer (h3_repair). A detector that fires is a QUESTION, not a verdict: the repairer acts only with positive
evidence (§2.1), and the never-list (labelled variants, attested perfects) is a unit test.

  ide_rule        a perfect-stem form that no source prints and engine2.build() refuses (F086: -ide is lexical, never
                  by rule). tabakokede flagged; Ulicisidwe (attested state perfect), nyamwide, bwene not.
  negfut_mixed    two future markers inside a ta- word (Tokooyoolima, Tatukooyoosyanga): -koo- + -yoo- / -yaku- / -zoo-.
                  Read by engine2 as neg_fut_koo with a stem that is itself a future marker + a known verb.
  askback         an ask-back draft (Nsemvwide / Tiindamvwa / Mwati nzi ...) while the shared word check knows every word
                  of the user's turn (h3_wordcheck): askback_unjustified (hard in selection)
  colour          the English names a colour, the Tonga has another book colour and not its own (bigc_score's closed set,
                  18 Oct; kept identical, checked by unit_detect)
  loop3           a word trigram three times or more (bigc_score._loop); judge's own 'repetition' stays in the v2 checks
  neg3            v2's negation check with the gap of 19 Oct closed: the negative participle / subjunctive
                  ka-SC-ta-...-i (Kamutapangi koonse 'please don't make noise') and every engine tense whose name
                  carries 'neg' count as a Tonga negative. v2's own check (h2_checks.negation) is NOT changed.
  lex_align       English content words with lexicon entries against the draft: does the draft contain a form of one of
                  their stems (engine2.read with the lexicon's verb stems, nouns by word / plural) or of a confirmed
                  synonym? [hypothesis] a reason to route only on short lines (< 4 words) with nothing aligned.
  yesno_polarity  [hypothesis, OFF by default] a positive Sena / Hena question answered Peepe / Pe + a positive verb.
"""
import collections
import csv
import functools
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
G2 = os.path.join(REPO, "tools", "chitonga", "grammar2")
sys.path[:0] = [HERE, G2]
WORD = re.compile(r"[A-Za-zŋŊ]+(?:'[A-Za-zŋŊ]+)*")
LEXICON = os.path.join(G2, "lexicon.tsv")
CACHE = os.path.join(REPO, "internal", "chitonga_train", "vocab", "attest_cache.json")


@functools.lru_cache(maxsize=1)
def E():
    import engine2
    return engine2


@functools.lru_cache(maxsize=1)
def known_verbs():
    """the verb stems the error map reads with (gen_everyday_v075.VERBS: lexicon + sourced) plus the lexicon's verbs"""
    import gen_everyday_v075 as ev
    v = set(ev.VERBS)
    for r in lex_rows():
        if (r.get("pos") or "").strip() == "v":
            v.add(re.sub(r"^ku", "", (r.get("word") or "").strip().lower()))
    v.discard("")
    return frozenset(v)


@functools.lru_cache(maxsize=1)
def printed():
    """{word: count} printed in clean Tonga (attest cache: speakers + clean books) or the school books"""
    c = json.load(open(CACHE, encoding="utf-8"))
    out = collections.Counter(c["toi"])
    for w, n in c["book"].items():
        out[w] += n
    return out


def _obj(r):
    """engine2.read gives (stem, person, tense) or (stem, person, tense, 'obj cN'); build() takes the object as 'cN'"""
    return r[3].replace("obj ", "") if len(r) > 3 and r[3] else None


@functools.lru_cache(maxsize=4096)
def readings(w):
    return tuple(E().read(w.lower(), known=known_verbs()))


# ------------------------------------------------------------------------------------------------ ide_rule
def ide_rule(toi):
    out = []
    eng = E()
    pr = printed()
    for w in WORD.findall(toi or ""):
        lw = w.lower()
        if pr.get(lw) or len(lw) < 5:
            continue
        rs = readings(lw)
        if not rs or not all(r[2] in eng.PERFECT_TENSES for r in rs):
            continue
        if any(lw in eng.build(r[0], r[1], r[2], _obj(r)) for r in rs):
            continue                                   # an attested / listed perfect: the engine builds it
        out.append(("ide_rule", lw, f"perfect-stem form, not printed, engine refuses it: {sorted({tuple(r[:3]) for r in rs})[:3]}"))
    return out


# ------------------------------------------------------------------------------------------------ negfut_mixed
FUT_MARK = re.compile(r"^(yoo|yaku|zoo|zyoo)(\w+)$")


def negfut_mixed(toi):
    out = []
    kv = known_verbs()
    for w in WORD.findall(toi or ""):
        lw = w.lower()
        if not re.match(r"^t[aio]", lw):
            continue
        for r in E().read(lw):                         # the guesser: the stem 'yoolima' is not a known verb
            s, p, t = r[:3]
            m = FUT_MARK.match(s)
            if t.startswith("neg_fut") and m and m.group(2) in kv:
                out.append(("negfut_mixed", lw, {"person": p, "stem": m.group(2), "markers": t + "+" + m.group(1)}))
                break
    return out


# ------------------------------------------------------------------------------------------------ ask-back
def askback(draft, wc):
    import h3_wordcheck as W
    if W.is_askback(draft) and not W.askback_justified(draft, wc):
        return [("askback_unjustified", draft[:40], f"the word check: {wc['note']}")]
    return []


# ------------------------------------------------------------------------------------------------ colour, loop3
# bigc_score's closed set (18 Oct, review section 5), kept identical (unit_detect compares the two)
COLOURS = {"red": ("subil", "subid"), "black": ("siya", "siide"), "white": ("tuba", "tubide"), "green": ("nyanzabil",),
           "blue": ("bbuluu", "buluu")}
_CONC = r"(?:ci|zi|bu|li|lu|ka|tu|mu|ba|mi|ma|i)?"
_POSS = r"(?:wa|ca|ya|za|lya|zya|bwa|lwa|kwa|twa|ka|ba)?"
COLOUR_RX = {c: re.compile("^" + (_POSS if c == "blue" else _CONC) + "(?:" + "|".join(st) + r")\w*$") for c, st in COLOURS.items()}


def colour(toi, en):
    enw = set(re.findall(r"[a-z]+", (en or "").lower()))
    toks = WORD.findall((toi or "").lower())
    has = {c for c, rx in COLOUR_RX.items() if any(rx.match(t) for t in toks)}
    for c in COLOURS:
        if c in enw and c not in has and has - enw:
            return [("colour", f"{c} vs {'/'.join(sorted(has - enw))}", "book colour words (bigc_score set)")]
    return []


def loop3(toi):
    w = (toi or "").lower().split()
    g = collections.Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2))
    if g and g.most_common(1)[0][1] >= 3:
        return [("loop3", " ".join(g.most_common(1)[0][0]), f"{g.most_common(1)[0][1]} times")]
    return []


# ------------------------------------------------------------------------------------------------ neg3
# negative participle / subjunctive forms engine2 reads (part_neg, neg_subj, imp_ko_neg, sub_neg_na ...): every tense
# whose name carries 'neg'. 'ka-SC-ta-...-i' is part_neg (Kamutapangi > panga 2p part_neg).
def _neg_reading(w):
    return any("neg" in r[2] for r in readings(w))


def toi_neg3(toi, lexical=True):
    import h2_checks as C
    if C.toi_neg(toi, lexical=lexical):
        return True
    nouns = C._lexicon_nouns()
    for w in re.findall(r"[a-zŋ']+", (toi or "").lower()):
        if w in nouns or len(w) < 5:
            continue
        if re.match(r"^(ka|ko|ke|nke)\w*t[aio]\w+i$", w) and _neg_reading(w):
            return True
    return False


def negation3(en, toi, back=None):
    """h2_checks.negation with toi_neg3 in place of toi_neg (the same keys: neg_dropped / neg_added / neg_roundtrip)"""
    import h2_checks as C
    out = []
    e, t, tg = C.en_neg(en), toi_neg3(toi), toi_neg3(toi, lexical=False)
    if e and not t:
        out.append(("neg_dropped", "the English has a negative, the Tonga none the engine reads", "neg3"))
    if tg and not e and not C.NEG_LIKE.search(en or ""):
        out.append(("neg_added", "the Tonga has a negative tense, the English none", "neg3"))
    if back is not None:
        b = C.en_neg(back) or (t and bool(C.NEG_LIKE.search(back)))
        if (tg and not b) or (b and not t and not C.NEG_LIKE.search(en or "")):
            out.append(("neg_roundtrip", f"Tonga {'negative' if t else 'positive'}, back-translation "
                                         f"{'negative' if b else 'positive'}: '{back[:80]}'", "neg3"))
    return out


# ------------------------------------------------------------------------------------------------ lex_align
EN_STOP = set("""a an the i you he she it we they me him her us them my your his its our their this that these those is
am are was were be been being have has had do does did not no yes to of in on at for with from by and or but if so as
there here what who whom whose which when where why how will would shall should can could may might must very too also
just only still now then today tomorrow yesterday please some any all one go going come coming get got make made say said
tell told let want like know see""".split())


def lex_rows():
    return list(csv.DictReader(open(LEXICON, encoding="utf-8"), delimiter="\t"))


@functools.lru_cache(maxsize=1)
def gloss_index():
    """English word > {(kind, tonga)}: kind 'v' (a verb stem) or 'n' (a noun word or plural form), from lexicon.tsv's
    meaning column (first gloss words before ; or ,) and confirmed synonyms (h2_synonyms)"""
    idx = collections.defaultdict(set)
    for r in lex_rows():
        pos = (r.get("pos") or "").strip()
        w = (r.get("word") or "").strip().lower()
        if not w or pos not in ("v", "n", "adj", "adv"):
            continue
        mean = (r.get("meaning") or "").lower()
        mean = re.sub(r"\(.*?\)|“|”|\"", " ", mean)
        for part in re.split(r"[;,/]", mean)[:4]:
            ws = [x for x in re.findall(r"[a-z]+", part) if x not in EN_STOP and len(x) > 2]
            for x in ws[:3]:
                if pos == "v":
                    idx[x].add(("v", re.sub(r"^ku", "", w)))
                else:
                    idx[x].add(("n", w))
                    for pl in re.split(r"[,/ ]+", (r.get("plural") or "").strip().lower()):
                        if pl:
                            idx[x].add(("n", pl))
    return idx


def _en_forms(x):
    """crude English lemma candidates: eats > eat, cooked > cook, going > go, children > child"""
    c = {x}
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("ren", ""),
                     ("ry", "er"), ("ty", "t"), ("y", ""), ("ness", "")):
        if x.endswith(suf) and len(x) - len(suf) >= 3:
            c.add(x[: len(x) - len(suf)] + rep)
    return c


def lex_align(en, toi):
    """{'aligned': [(en word, tonga)], 'missing': [en words with entries, none found], 'no_entry': [...]}"""
    import h2_synonyms as SY
    idx = gloss_index()
    toks = [w.lower() for w in WORD.findall(toi or "")]
    stems = set()
    for w in toks:
        for r in readings(w):
            stems.add(r[0])
    words = set(toks)
    syn = set()
    for w in toks:
        syn.update(SY.synonyms_of(w))
    out = {"aligned": [], "missing": [], "no_entry": []}
    for x in re.findall(r"[a-z]+", (en or "").lower()):
        if x in EN_STOP or len(x) < 3:
            continue
        ent = set()
        for f in _en_forms(x):
            ent |= idx.get(f, set())
        if not ent:
            out["no_entry"].append(x)
            continue
        hit = next(((k, t) for k, t in ent if (k == "v" and t in stems) or (k == "n" and (t in words or t in syn))
                    or (k == "v" and any(s == t for s in syn))), None)
        if hit:
            out["aligned"].append((x, hit[1]))
        else:
            out["missing"].append(x)
    return out


def lex_align_issue(en, toi):
    """[hypothesis] the short-line reason: < 4 English words, at least one content word has an entry, none aligned"""
    if len((en or "").split()) >= 4:
        return []
    a = lex_align(en, toi)
    if a["missing"] and not a["aligned"]:
        return [("lex_unaligned", ", ".join(a["missing"]), "no lexicon stem of the English content words in the draft")]
    return []


# ------------------------------------------------------------------------------------------------ yes/no (off)
YESNO_ON = os.environ.get("HARNESS3_YESNO", "0") == "1"


def yesno_polarity(user, reply):
    if not YESNO_ON:
        return []
    if not re.match(r"^\s*(sena|hena)\b", user or "", re.I) or toi_neg3(user, lexical=False):
        return []
    m = re.match(r"^\s*(peepe|pe)\b[,.!]?\s*(.*)$", reply or "", re.I)
    if m and m.group(2) and not toi_neg3(m.group(2)):
        return [("yesno_polarity", reply[:40], "'no' to a positive question, then a positive statement")]
    return []


# ------------------------------------------------------------------------------------------------ neighbour words
# A Bemba / Nyanja word in a DRAFT (the model leaking, or answering mixed Tonga in kind) is attest's own verdict
# 'mixed' (grammar2/lexicon_contact.tsv, 19 Oct): attest.line lists it, so it is an attestation reason in the gate and
# weighs 2 in selection (h3_select.ATTEST_W). Nothing here re-reads that lexicon or parses the note text.


# ------------------------------------------------------------------------------------------------ English left in
# replay 19 Oct: the soup's beam 'Peepe, I’m ucilindila.' was SENT: attest and judge split "I’m" at the curly
# apostrophe into one-letter tokens, which they skip. English function words and contractions never occur in Tonga.
EN_LEAK = re.compile(r"(?i)(?<![A-Za-zŋ'’])(i[’']m|it[’']s|don[’']t|can[’']t|i[’']ll|you[’']re|the|is|are|you|your|my|and|"
                     r"this|that|with|what|of|for|he|she|they|was|were|will|not|have|has)(?![A-Za-zŋ'’])")


def english_leak(toi):
    m = EN_LEAK.findall(toi or "")
    return [("english_leak", ", ".join(sorted(set(x.lower() for x in m))), "English words in the Tonga")] if m else []


# run 4 with the persona prompt (20 Oct): replies in a tool format ('{"type": "text", ...}', 'REPORT: ...', code fences)
FORMAT_JUNK = re.compile(r'^\s*[\[{]|"type"\s*:|\bREPORT\s*:|```|^\s*CONFIDENCE\s*:')


def format_junk(text):
    m = FORMAT_JUNK.search(text or "")
    return [("format_junk", m.group(0).strip()[:20], "a tool / JSON / report format, not a reply")] if m else []


# ------------------------------------------------------------------------------------------------ object pronoun
EN_OBJ = re.compile(r"(?i)\b(?:[a-z]+)\s+(me|you|him|her|us|them)\b")
EN_NOT_VERB_BEFORE = {"to", "for", "with", "at", "from", "about", "of", "than", "like", "do", "are", "will", "can",
                      "have", "did", "would", "could", "should", "thank", "and", "or", "if", "when", "see", "tell"}


def obj_preserve(en, toi):
    """[soft] the English has an object pronoun after a verb ('I sent you a message'), the Tonga has no verb with an
    object concord and no free pronoun. Replay 19 Oct, A125: the readers read Ndakutuma and Ndakatuma alike ('I sent a
    message'), so form agreement chose the draft that lost 'you'. A SIGNAL in selection (soft tier) until its false
    alarms on known-good lines are measured; never a reason to route."""
    def you_is_object(m):          # 'a game you know', 'books you want': 'you' is the subject of a relative clause
        if m.group(1).lower() != "you":
            return True
        nxt = re.match(r"\s*([A-Za-z]+)", (en or "")[m.end():])
        return not nxt or nxt.group(1).lower() in {"a", "an", "the", "on", "at", "in", "to", "for", "now", "today",
                                                    "tomorrow", "again", "yesterday", "soon", "later", "with", "please",
                                                    "some", "this", "that", "my", "our", "money", "back"}
    hits = [m for m in EN_OBJ.finditer(en or "") if m.group(0).split()[0].lower() not in EN_NOT_VERB_BEFORE
            and you_is_object(m)]
    if not hits:
        return []
    words = WORD.findall(toi or "")
    # a PERSON object for me / you / him / her / us (nda-ka-tuma also reads 'I sent it (class 13)': not 'you');
    # 'them' may be people or things
    pron = hits[0].group(1).lower()
    ok = (lambda o: True) if pron == "them" else (lambda o: o in ("1s", "2s", "1p", "2p", "c1", "c2"))
    if any(_obj(r) and ok(_obj(r)) for w in words for r in readings(w.lower())):
        return []
    if any(re.match(r"(?i)^(ndime|nduwe|ndinywe|ndiswe|nguwe|mbabo|ndiye|ngonya|yebo|ndiwe|mebo|ndiwe|swebo|ndwe|nywebo|mbaabo|mulindime|kulindime|kuli(ndi|nduwe|ndiswe|nywe))", w) for w in words):
        return []
    return [("obj_missing", hits[0].group(1), "the English has an object pronoun, no Tonga verb carries an object concord")]


# ------------------------------------------------------------------------------------------------ dict_missing (L2)
# L2 of the translator round (council/jesuit_translator_plan_2026-10-20.md, 20 Oct 2026). WHY: the largest failure mode
# of the translator in teacher material is the rare or domain word (15 of 32 non-right lines in the 20 Oct blind
# reading: civwuka, zipipila / mifwolo, gaila, minsale ...). The meaning column of lexicon.tsv is a word list, not a
# sense index (lex_align's 29% false alarms on short known-good lines, step 1), so this check uses ONLY rows that cannot
# be read two ways:
#   - the word has exactly one lexicon row (homographs out), pos n / v / adj / adv;
#   - its meaning has no doubt marker (?, unknown, another reading, cf., or, probably, perhaps, uncertain, brackets
#     that open the gloss) and no 'unclear' note; the row is not sourced from a held-out set (HELD-OUT, TEST SOURCE):
#     a check built from a test line must not be scored on that line;
#   - no nasal / ku- neighbour in the lexicon (sima 'ripe' beside nsima 'porridge', the 20 Oct error read the other way):
#     n+w, m+w, ny+w, w without its nasal, ku+w; and the hand list HOMOGRAPHS;
#   - E>T only: the English word has exactly one Tonga entry in the whole lexicon and is that entry's head gloss.
# Tonga > English: a source word with such a row whose gloss content words (lemma forms, a 5-letter prefix, or a
# confirmed Tonga synonym's gloss) are all absent from the English output. English > Tonga: an English content word
# whose one entry (word, plural, or a verb form read by engine 2 to its stem, or a confirmed synonym) is absent from
# the Tonga. It is an ALARM for the gate; it reranks only when its false alarms on right lines stay <= 20% (the plan's
# kill test; measured in HARNESS_V3_BUILD.md "P1").
HOMOGRAPHS = {"sima", "nsima", "kumana", "mana", "mena", "nsimo"}
FORM_POS = {"n.pl", "n.loc", "n.conn", "n.var", "n.cop", "adj.form", "num.form", "conj.form", "var"}
DOUBT = re.compile(r"\?|unknown|another reading|\bcf\.|\bor\b|probably|perhaps|uncertain|possibly|maybe|^\s*[\[(]",
                   re.I)
HELD_SRC = re.compile(r"HELD|TEST SOURCE|held[- ]out", re.I)
DM_STOP = EN_STOP | set("""thing things person people something someone way kind sort part make made do does be become
cause let give take put get one ones other another self also used use very much many more most less little big small
good bad new old""".split())


def _gloss_words(meaning):
    """content words of the FIRST sense of a lexicon meaning ('sugar-cane' > sugar, cane, sugarcane)"""
    m = re.sub(r"[“”\"]", " ", (meaning or "").lower())
    first = re.split(r"[;,/]", m)[0]
    first = re.sub(r"\bto\b", " ", first)
    ws = [x for x in re.findall(r"[a-z]+", first) if x not in DM_STOP and len(x) > 2]
    if "-" in first:
        ws += [re.sub(r"[^a-z]", "", t) for t in first.split() if "-" in t]
    return [w for w in dict.fromkeys(ws) if w]


@functools.lru_cache(maxsize=1)
def dm_index():
    """{'toi': {tonga form: (headword, pos, gloss words, meaning)}, 'verbs': {stem: ...}, 'en': {english word: entry}}"""
    rows = lex_rows()
    by = collections.defaultdict(list)
    for r in rows:
        w = (r.get("word") or "").strip().lower()
        if w:
            by[w].append(r)
    # 20 Oct 2026 (the Jesuit, lexicon_corpus.py): the lexicon now carries FORM rows of its words (n.pl bana > mwana,
    # n.loc kucisi > cisi, n.conn zyabantu ...). A form row is the same word in another shape, not a second word: it
    # must not make its base a 'neighbour' (kucisi would block cisi by the ku+w rule) nor stop a plural from inheriting
    # its singular's entry (bana from mwana, as before the form rows). Measured on the 25,171 known-good lines.
    form_only = {w for w, rs in by.items() if all((x.get("pos") or "").strip() in FORM_POS for x in rs)}
    allw = set(by) - form_only
    plurals = set()
    for r in rows:
        for pl in re.split(r"[,/ ]+", (r.get("plural") or "").strip().lower()):
            if pl:
                plurals.add(pl)

    def neighbour(w):
        alts = {"n" + w, "m" + w, "ny" + w, "ku" + w, "kw" + w}
        if re.match(r"^(n|m)(?=[^aeiou])", w):
            alts.add(w[1:])
        if w.startswith("ny"):
            alts.add(w[2:])
        if w.startswith(("ku", "kw")):
            alts.add(w[2:])
        return bool(alts & (allw | plurals))
    toi, verbs, en_map = {}, {}, collections.defaultdict(set)
    for r in rows:                                   # every entry glossing an English word, ambiguous ones included
        if (r.get("pos") or "").strip() in FORM_POS:
            continue                                 # a form row repeats its base's gloss (bana 'children' < mwana)
        w = (r.get("word") or "").strip().lower()
        for g in _gloss_words(r.get("meaning")):
            en_map[g].add(w)
    en = {}
    for w, rs in by.items():
        if len(rs) != 1 or w in HOMOGRAPHS or len(w) < 4 or " " in w:
            continue
        r = rs[0]
        pos = (r.get("pos") or "").strip()
        mean = (r.get("meaning") or "").strip()
        if pos not in ("n", "v", "adj", "adv") or not mean or DOUBT.search(mean) or (r.get("unclear") or "").strip():
            continue
        if HELD_SRC.search(r.get("source") or "") or neighbour(w) or w in plurals:
            continue
        gw = _gloss_words(mean)
        if not gw:
            continue
        ent = (w, pos, tuple(gw), mean)
        if pos == "v":
            verbs[re.sub(r"^ku", "", w)] = ent
        else:
            toi[w] = ent
            for pl in re.split(r"[,/ ]+", (r.get("plural") or "").strip().lower()):
                if pl and pl not in allw and pl != "of":
                    toi.setdefault(pl, ent)
        head = gw[0]
        if en_map.get(head) == {w}:
            en[head] = ent
    return {"toi": toi, "verbs": verbs, "en": en}


IRREG = {"began": "begin", "went": "go", "saw": "see", "seen": "see", "found": "find", "came": "come", "gave": "give",
         "given": "give", "took": "take", "taken": "take", "told": "tell", "said": "say", "made": "make", "ate": "eat",
         "eaten": "eat", "drank": "drink", "drove": "drive", "driven": "drive", "ran": "run", "sat": "sit", "stood": "stand",
         "wrote": "write", "written": "write", "knew": "know", "known": "know", "thought": "think", "brought": "bring",
         "bought": "buy", "caught": "catch", "taught": "teach", "fell": "fall", "fallen": "fall", "grew": "grow",
         "grown": "grow", "held": "hold", "kept": "keep", "left": "leave", "lost": "lose", "met": "meet", "paid": "pay",
         "sent": "send", "slept": "sleep", "spoke": "speak", "spoken": "speak", "threw": "throw", "thrown": "throw",
         "understood": "understand", "woke": "wake", "won": "win", "wore": "wear", "worn": "wear", "broke": "break",
         "broken": "break", "chose": "choose", "chosen": "choose", "drew": "draw", "flew": "fly", "forgot": "forget",
         "hid": "hide", "rode": "ride", "rose": "rise", "shook": "shake", "sang": "sing", "stole": "steal", "tore": "tear",
         "children": "child", "men": "man", "women": "woman", "feet": "foot", "teeth": "tooth", "mice": "mouse",
         "heard": "hear", "felt": "feel", "built": "build", "dug": "dig", "hung": "hang", "led": "lead", "lay": "lie",
         "lain": "lie", "fed": "feed", "fought": "fight", "sold": "sell", "sought": "seek", "swept": "sweep", "wept": "weep"}


def _forms(x):
    f = _en_forms(x) | {x}
    if x in IRREG:
        f.add(IRREG[x])
    return f


def _en_has(words_out, gloss):
    """is any gloss word in the English output (lemma forms, or a shared 5-letter prefix for words of 5+ letters)?"""
    for g in gloss:
        gf = _forms(g)
        for o in words_out:
            if o in gf or _forms(o) & gf:
                return True
            if len(g) >= 5 and len(o) >= 5 and o[:5] == g[:5]:
                return True
    return False


def _toi_entries(toi):
    """[(source word, entry)] for the Tonga words with an unambiguous row (nouns by form, verbs by an engine reading
    whose every reading has the same known stem)"""
    ix = dm_index()
    out = []
    for w in WORD.findall(toi or ""):
        lw = w.lower()
        if lw in ix["toi"]:
            out.append((lw, ix["toi"][lw]))
            continue
        rd = readings(lw)
        st = {r[0] for r in rd}
        if len(st) == 1 and next(iter(st)) in ix["verbs"]:
            out.append((lw, ix["verbs"][next(iter(st))]))
    return out


DM_CALIB = os.path.join(REPO, "internal", "chitonga_train", "vocab", "dict_missing_calib.json")
DM_FALSE_MAX = 0.20          # the plan's kill line, applied per word: a word that fires on > 20% of known-good lines is out


def lexicon_stamp():
    """the lexicon state a measurement used (coordinator, 20 Oct: the Harris merge will change lexicon.tsv)"""
    import hashlib
    b = open(LEXICON, "rb").read()
    return {"sha256": hashlib.sha256(b).hexdigest()[:16], "mtime": int(os.path.getmtime(LEXICON)), "rows": b.count(bytes([10])) - 1}


def _known_good_pairs():
    """non-held-out school-book pairs (book Tonga, the Jesuit's English, confidence high / teacher): training-side
    material only; held-out and exam rows (the diagnosis lines, the anchor) are never read"""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "internal", "chitonga_train", "vocab", "new_books_2026-10-16", "*pairs*.jsonl"))):
        b = os.path.basename(f)
        if "fidelity" in b or "SUPERSEDED" in b or "machine" in b:
            continue
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if (not r.get("held_out") and not r.get("exam") and r.get("confidence") in ("high", "teacher")
                        and r.get("toi") and r.get("en_guess")):
                    out.append((r["toi"], r["en_guess"]))
    return out


def dm_calibrate(write=True, pairs=None):
    """per-word reliability on the known-good pairs: {kind: {key: [lines where the word occurs, lines where it fired]}}.
    Rebuilt automatically when the lexicon changes (its sha is stored)."""
    import datetime as _dt
    dm_index.cache_clear()
    stats = {"toi>en": collections.defaultdict(lambda: [0, 0]), "en>toi": collections.defaultdict(lambda: [0, 0])}
    pairs = _known_good_pairs() if pairs is None else pairs
    ix = dm_index()
    for toi, en in pairs:
        for w, ent in dict.fromkeys(_toi_entries(toi)):
            stats["toi>en"][ent[0]][0] += 1
        for x in set(re.findall(r"[a-z]+", en.lower())):
            ent = next((ix["en"][f] for f in _en_forms(x) if f in ix["en"]), None)
            if ent:
                stats["en>toi"][f"{x} > {ent[0]}"][0] += 1
        for k, det, _ in _dm_raw(toi, en, "toi>en"):
            stats["toi>en"][det["key"]][1] += 1
        for k, det, _ in _dm_raw(en, toi, "en>toi"):
            stats["en>toi"][det["key"]][1] += 1
    c = {"lexicon": lexicon_stamp(), "built": _dt.datetime.now().isoformat(timespec="seconds"), "pairs": len(pairs),
         "false_max": DM_FALSE_MAX, "toi>en": dict(stats["toi>en"]), "en>toi": dict(stats["en>toi"])}
    if write:
        json.dump(c, open(DM_CALIB, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    _calib.clear()
    _calib.update(c)
    return c


_calib = {}


def dm_calib():
    if not _calib:
        try:
            c = json.load(open(DM_CALIB, encoding="utf-8"))
        except (OSError, ValueError):
            c = None
        if not c or c.get("lexicon", {}).get("sha256") != lexicon_stamp()["sha256"]:
            return dm_calibrate()
        _calib.update(c)
    return _calib


def dm_reliable(direction, key, pos):
    """a word fires only if, on the known-good lines where it occurs, it fired on <= DM_FALSE_MAX of them; with fewer
    than 2 occurrences there is no evidence: nouns / adjectives / adverbs fire (the rare-word mode is mostly nouns),
    verbs do not (a verb's English varies with the sentence)"""
    occ, fires = dm_calib().get(direction, {}).get(key, [0, 0])
    if occ < 2:
        return pos != "v"
    return fires / occ <= DM_FALSE_MAX


def dict_missing(src, out, direction, calibrated=True):
    """L2 alarm. direction 'toi>en' (src Tonga, out English) or 'en>toi' (src English, out Tonga). Returns
    [("dict_missing", "<word> (<gloss>)", evidence)]; calibrated=False gives every raw firing (for dm_calibrate)"""
    out_ = []
    for k, det, ev in _dm_raw(src, out, direction):
        if not calibrated or dm_reliable(direction, det["key"], det["pos"]):
            out_.append((k, det["text"], ev))
    return list(dict.fromkeys(out_))


def _dm_raw(src, out, direction):
    import h2_synonyms as SY
    if not (src or "").strip() or not (out or "").strip():
        return []
    ix = dm_index()
    iss = []
    if direction == "toi>en":
        ow = [x.lower() for x in re.findall(r"[A-Za-z]+", out)]
        for w, ent in _toi_entries(src):
            gloss = list(ent[2])
            for s in SY.synonyms_of(ent[0]):        # a confirmed Tonga synonym's gloss counts too
                e2 = ix["toi"].get(s) or ix["verbs"].get(re.sub(r"^ku", "", s))
                if e2:
                    gloss += list(e2[2])
            if not _en_has(ow, gloss):
                iss.append(("dict_missing", {"key": ent[0], "pos": ent[1], "text": f"{w} ({ent[3][:40]})"},
                            "unambiguous lexicon word, its gloss is not in the English"))
    else:
        toks = [w.lower() for w in WORD.findall(out)]
        words = set(toks)
        stems = {r[0] for w in toks for r in readings(w)}
        syn = set()
        for w in toks:
            syn.update(SY.synonyms_of(w))
        for x in re.findall(r"[a-z]+", src.lower()):
            if x in DM_STOP or len(x) < 3:
                continue
            ent = next((ix["en"][f] for f in _en_forms(x) if f in ix["en"]), None)
            if not ent:
                continue
            w, pos = ent[0], ent[1]
            forms = {w} | ({re.sub(r"^ku", "", w)} if pos == "v" else set())
            hit = (words & forms) or (pos == "v" and re.sub(r"^ku", "", w) in stems) or (forms & syn) \
                or any(t.endswith(w) and len(t) - len(w) <= 4 for t in toks if pos != "v")
            if not hit:
                iss.append(("dict_missing", {"key": f"{x} > {w}", "pos": pos, "text": f"{x} > {w}"},
                            "the English word's one lexicon entry is not in the Tonga"))
    seen, uniq = set(), []
    for k, d, e in iss:
        if d["key"] not in seen:
            seen.add(d["key"])
            uniq.append((k, d, e))
    return uniq


# ------------------------------------------------------------------------------------------------ all
def detect(draft, en=None, mode="translate", wc=None, user=None):
    """every v3 detector on one draft (answer only). Returns a list of [key, detail, evidence]"""
    iss = []
    iss += ide_rule(draft)
    iss += negfut_mixed(draft)
    iss += loop3(draft)
    iss += english_leak(draft)
    iss += format_junk(draft)
    if mode == "translate" and en:
        iss += obj_preserve(en, draft)
        iss += colour(draft, en)
        iss += negation3(en, draft)
        iss += lex_align_issue(en, draft)
    if mode == "converse" and wc is not None:
        iss += askback(draft, wc)
        iss += yesno_polarity(user, draft)
    return [list(i) for i in iss]
