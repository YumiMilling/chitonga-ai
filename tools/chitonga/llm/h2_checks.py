"""Harness v2, the outside checks (17 Oct 2026; TODO_ENGINE_HARNESS E2, E3, E5).

E2 replaces the letter-overlap meaning gate (chrF) by PRESERVATION checks: what must survive a translation whatever
the wording. Each check returns a list of issues (kind, detail); empty = passed. Pairs compared:
  en  > toi    the English source against the Tonga draft
  toi > back   the Tonga draft against its back-translation (the round trip), which is compared with the English
  negative   EN_NEG (negation_test.py) on the English side, toi_neg (engine-2 reading of negative tenses) on the Tonga
             side. Dropped AND added negatives are issues: either reverses the meaning ("does not kill" > "kills").
  numbers    digits, English number words, Tonga numerals (h2_text.numbers_*). 'one', ordinals and 'both' are soft
             (bumwi buzuba 'one day' is often not a counted 'one'): reported, never routed.
  names      English names (h2_text.en_names) must appear in the Tonga; Tonga names (vectors/names.py tag, the kept
             types) must appear in the back-translation.
  thought    bge-small similarity of the back-translation and the English (harness.thought): a SIGNAL only, never a gate
             on its own (E2; it is an NLLB relative reading NLLB output, and short lines are guessed well).
E3 drift: every output word against the FineWeb-2 Bemba / Nyanja counts and our clean Tonga counts (the attestation
   cache built by attest.py from contact_check.py's sources: internal/chitonga_train/vocab/attest_cache.json).
   Flag = frequent in Bemba or Nyanja web text, rare in what our speakers and the clean books write. A flag is a
   question for a teacher ('Tonga, or not?'), never a verdict: shared Bantu words (mwana, kuti) are kept out by the
   ratio, a real but rare Tonga word can still be flagged. Ila: no web text is held (FineWeb-2 has none we fetched);
   Ila / Lenje / Soli leakage is covered only by the teachers' not-Tonga list (attest.py).
E5 glossary: internal/chitonga_train/vocab/health_glossary.tsv, fixed health terms with their sources, checked in both
   directions: an English term needs its Tonga term in the draft; a Tonga term needs an accepted English reading in the
   back-translation; forbidden readings are issues (muŋanga read as 'witch': Chris, 17 Oct, muŋanga = healer).
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
VOC = os.path.join(REPO, "internal", "chitonga_train", "vocab")
GLOSSARY = os.path.join(VOC, "health_glossary.tsv")
CACHE = os.path.join(VOC, "attest_cache.json")
sys.path[:0] = [HERE, os.path.join(REPO, "tools", "chitonga", "grammar2"), os.path.join(REPO, "tools", "chitonga", "train"),
                os.path.join(REPO, "tools", "chitonga", "vectors")]
import h2_text as T  # noqa: E402

_neg = {}


def _negation():
    """negation_test imports panel_analyse, which imports harness and switches its self-check off (H.PERSONA = None);
    keep the caller's harness as it was"""
    if not _neg:
        H = sys.modules.get("harness")
        keep = H.PERSONA if H is not None else None
        import negation_test as NT
        if H is not None:
            H.PERSONA = keep
        _neg["en"], _neg["toi"] = NT.EN_NEG, NT.toi_neg
    return _neg["en"], _neg["toi"]


def en_neg(text):
    return bool(_negation()[0].search(text or ""))


# Tonga negatives negation_test.toi_neg misses, read in the E8 false alarms on held-out lines (18 Oct; fitted on that
# set, so its numbers after this change are optimistic). Grammatical: taku 'there is not', kwakanyina / kakunyina
# 'there was none' (ku- / kwa- / kwaka- + nyina; NOT banyina 'his mother'), naata- 'when he had not' (naatakajata),
# tiiba / teeba 'was not'. Lexical ('like' a negative: they satisfy an English negative, but are not an ADDED negative
# when the English has none: 'Cimana kale kubula mumuni' = 'It is already dark'): kubula 'lack, without', -aalila
# 'fail, be too hard for' (cakamwaalila 'he could not'), tee 'is it not' (question).
EXTRA_NEG = re.compile(r"^(taku|tiiba|teeba|(ka)?(ku|kwa|kwaka|kwaaka|kaku)nyina|naata\w{3,})$")
NEG_LEXICAL = re.compile(r"^(tee|\w*(ku|la|ka|a)bula|\w*aalila|\w*aalilwa)$")
_nouns = set()


def _lexicon_nouns():
    """nouns (and plurals) of lexicon.tsv, read at run time: a dictionary noun is never a negative verb (negation_test
    reads 'bataata' 'father' as ba-ta-..., E8 18 Oct)"""
    if not _nouns:
        import csv as _csv
        for r in _csv.DictReader(open(os.path.join(REPO, "tools", "chitonga", "grammar2", "lexicon.tsv"), encoding="utf-8"),
                                 delimiter="\t"):
            if (r.get("pos") or "").strip() == "n":
                _nouns.add((r.get("word") or "").strip().lower())
                _nouns.update(x for x in re.split(r"[,/ ]+", (r.get("plural") or "").strip().lower()) if x)
        _nouns.discard("")
    return _nouns


def toi_neg(text, lexical=True):
    """a negative in the Tonga: grammatical (engine reading via negation_test, plus EXTRA_NEG), and with lexical=True
    also the lexical negatives (NEG_LEXICAL)"""
    nouns = _lexicon_nouns()
    tn = _negation()[1]
    for w in re.findall(r"[a-zŋ']+", (text or "").lower()):
        if EXTRA_NEG.match(w):
            return True
        if NEG_LEXICAL.match(w):          # lexical: counts only when asked for, never through the engine reading
            if lexical:
                return True
            continue
        if w in nouns:
            continue
        if tn(w):
            return True
    return False


# ------------------------------------------------------------------------------------------------------- E2
# English that Tonga says with a negative tense: 'before eating' = kamutana kulya (while you have not yet eaten),
# 'until', 'unless', 'avoid', 'fail', 'refuse', 'stop', 'prevent', 'lack', 'without'; seen in the smoke test of 17 Oct
# (poster MOH/WHO: kamutaninga aakulya 'before eating'). The negative is then not ADDED, and not required either.
NEG_LIKE = re.compile(r"\b(before|until|till|unless|yet|avoid\w*|fail\w*|refus\w*|stop\w*|prevent\w*|lack\w*|"
                      r"without|except|apart from)\b", re.I)


def negation(en, toi, back=None):
    out = []
    e, t, tg = en_neg(en), toi_neg(toi), toi_neg(toi, lexical=False)
    if e and not t:
        out.append(("neg_dropped", "the English has a negative, the Tonga none the engine reads"))
    if tg and not e and not NEG_LIKE.search(en or ""):
        out.append(("neg_added", "the Tonga has a negative tense, the English none"))
    if back is not None:
        b = en_neg(back) or (t and bool(NEG_LIKE.search(back)))
        if (tg and not b) or (b and not t and not NEG_LIKE.search(en or "")):
            out.append(("neg_roundtrip", f"Tonga {'negative' if t else 'positive'}, back-translation "
                                         f"{'negative' if b else 'positive'}: '{back[:80]}'"))
    return out


def numbers(en, toi, back=None):
    out = []
    ne = T.numbers_en_composed(en)
    nt = T.numbers_toi(toi)
    hard = {v for v, k in ne.items() if k == "hard"}
    miss = sorted(v for v in hard if v not in nt)
    if miss:
        out.append(("num_missing", f"English {miss} not found in the Tonga (read {sorted(x for x in nt if x)})"))
    extra = sorted(v for v in nt if v and v != 1 and v not in ne)
    if extra and not ne:
        out.append(("num_added", f"Tonga numeral(s) {extra} and no number in the English"))
    soft = sorted(v for v, k in ne.items() if k == "soft" and v not in nt)
    if soft:
        out.append(("num_soft", f"soft English number(s) {soft} not counted in the Tonga (signal only)"))
    if back is not None:
        nb = T.numbers_en_composed(back)
        bm = sorted(v for v in hard if v not in nb)
        if bm:
            out.append(("num_roundtrip", f"English {bm} not in the back-translation '{back[:80]}'"))
    return out


def _tonga_names(toi):
    try:
        import names as N
        return sorted({s["text"] for s in N.tag(toi) if s["type"] in N.KEEP_TYPES})
    except Exception:  # noqa: BLE001
        return [w for i, w in enumerate(re.findall(r"[A-Za-zŋŊ']+", toi or "")) if i and w[:1].isupper()]


def _has(name, text):
    """case-insensitive, inside a word too: Tonga glues prefixes to names (kuLusaka, ba Mwiinga, kwaMweemba)"""
    return name.lower() in (text or "").lower()


def names(en, toi, back=None):
    out = []
    miss = [n for n in T.en_names(en) if not _has(n, toi)]
    if miss:
        out.append(("name_missing", f"English name(s) {miss} not in the Tonga"))
    if back is not None:
        tn = _tonga_names(toi)
        bm = [n for n in tn if not _has(n, back)]
        if bm:
            out.append(("name_roundtrip", f"Tonga name(s) {bm} not in the back-translation '{back[:80]}'"))
    return out


# ------------------------------------------------------------------------------------------------------- E5
_gl = []


def glossary_rows():
    if not _gl:
        for r in csv.DictReader(open(GLOSSARY, encoding="utf-8"), delimiter="\t"):
            _gl.append({k: (re.compile(v, re.I) if k.endswith("_rx") and v else v) for k, v in r.items()})
    return _gl


def glossary(en, toi, back=None):
    """both directions; en may be None (conversation: only the forbidden readings are checked on the round trip).
    Round trip: a term the ENGLISH has, whose Tonga is in the draft, must come back in one of the accepted readings
    of any glossary row whose Tonga matches the same word (musilisi is a healer, a doctor or a nurse; cibbadela a
    clinic or a hospital). A Tonga term the English does not have is not demanded back ('BULWAZI BWA CHOLERA' for
    'CHOLERA KILLS': the reader may drop 'disease of')."""
    out = []
    rows = glossary_rows()
    hits = {}
    for g in rows:
        m = g["toi_rx"].search(toi or "")
        if m:
            hits[g["term"]] = (g, m.span())
    for g in rows:
        if en and g["en_rx"].search(en):
            via = _synonym_hit(g, toi) if g["term"] not in hits else None
            if via:
                out.append(("gloss_synonym", f"{g['term']}: the draft says '{via[0]}', a confirmed synonym of "
                                             f"'{via[1]}' (synonyms_antonyms.tsv; not an error, E9)"))
            elif g["term"] not in hits:
                out.append(("gloss_missing", f"{g['term']}: the English says '{g['en_rx'].search(en).group(0)}', the "
                                             f"Tonga has none of /{g['toi_rx'].pattern}/ ({g['source'][:60]})"))
            if g["forbid_toi_rx"] and g["forbid_toi_rx"].search(toi or ""):
                out.append(("gloss_wrong", f"{g['term']}: the Tonga has '{g['forbid_toi_rx'].search(toi).group(0)}', "
                                           f"which is not this term"))
            elif back is not None and g["term"] in hits:
                span = hits[g["term"]][1]
                sib = [h for h, sp in hits.values() if sp[0] < span[1] and span[0] < sp[1]]
                if not any(h["back_ok_rx"].search(back) for h in sib):
                    out.append(("gloss_roundtrip", f"{g['term']}: the Tonga has '{toi[span[0]:span[1]]}', the "
                                                   f"back-translation has none of its readings"))
    if back is not None:
        for term, (g, span) in hits.items():
            if g["forbid_back_rx"] and g["forbid_back_rx"].search(back):
                out.append(("gloss_wrong", f"{term}: the Tonga term '{toi[span[0]:span[1]]}' was read as "
                                           f"'{g['forbid_back_rx'].search(back).group(0)}'"))
    return out


def _synonym_hit(g, toi):
    """E9: a draft word whose CONFIRMED synonym (book / teacher) matches the glossary term's Tonga"""
    import h2_synonyms as SY
    for w in re.findall(r"[a-zŋ']+", (toi or "").lower()):
        for s in SY.synonyms_of(w):
            if g["toi_rx"].search(s):
                return (w, s)
    return None


# ------------------------------------------------------------------------------------------------------- E3
DRIFT_RATIO = float(os.environ.get("HARNESS_DRIFT_RATIO", "50"))   # contact rate / Tonga rate (per million, +1 each)
DRIFT_MIN_PM = float(os.environ.get("HARNESS_DRIFT_MIN_PM", "20"))  # the word is at least this common in Bem / Nya
_dc = {}


def _drift_data():
    if not _dc:
        c = json.load(open(CACHE, encoding="utf-8"))
        _dc.update(toi=c["toi"], bem=c["bem"], nya=c["nya"], tot=c["totals"])
    return _dc


def drift(toi):
    """[(word, reason)] for words frequent in Bemba / Nyanja web text and rare in our clean Tonga"""
    d = _drift_data()
    out, seen = [], set()
    prev_end = 0
    for i, m in enumerate(re.finditer(r"[A-Za-zŋŊ']+", toi or "")):
        w, gap = m.group(0), (toi or "")[prev_end:m.start()]
        prev_end = m.end()
        initial = i == 0 or any(ch in ".!?:;\"“”‘(—–-" for ch in gap)
        lw = w.lower()
        if lw in seen or len(lw) < 3 or (w[:1].isupper() and not initial):   # a mid-sentence capital: a name
            continue
        seen.add(lw)
        b = 1e6 * d["bem"].get(lw, 0) / d["tot"]["bem"]
        n = 1e6 * d["nya"].get(lw, 0) / d["tot"]["nya"]
        t = 1e6 * d["toi"].get(lw, 0) / d["tot"]["toi"]
        c = max(b, n)
        if c >= DRIFT_MIN_PM and (c + 1) / (t + 1) >= DRIFT_RATIO:
            lang = "Bemba" if b >= n else "Nyanja"
            out.append((lw, f"{lang} web text {c:.0f} per million, our Tonga {t:.1f} per million "
                            f"({d['toi'].get(lw, 0)}x in {d['tot']['toi']:,} words)"))
    return out
