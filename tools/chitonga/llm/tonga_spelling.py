"""Phone-typed Tonga toward the school-book spelling, as an ALTERNATIVE reading for the translator (20 Oct 2026, the
Jesuit; Chris's own test the same day, relayed by the coordinator: "akucincha" (school spelling kucinca), "kuli bambila"
(kulibambila), "abuyumu yumu" (buyumu-yumu), dropped double vowels (makani / maakani), ng' for ŋ, single for double
consonants (b / bb, k / kk), w / u and y / i swaps).

Why the school books are the target: DECISIONS 17 Oct, "Books are the norm, speakers are variants": the written
standard is the norm for FORM, speaker forms are labelled spoken variants. So a variant is never an error and never
erased: the translator reads BOTH the original and the normalised form and the readers' consensus chooses
(translator_service); both are logged; what is stored is the teacher's original text.

A word is changed only when
  - the original is NOT school-attested (printed fewer than 2 times in the school books of the attestation cache and
    not a lexicon headword / plural), and
  - the candidate IS school-attested (printed 2+ times in the books, or a lexicon headword / plural);
  the candidate printed most often wins. Names (capitalised mid-sentence), numbers and English words are left alone.
The candidates, in this order:
  1 the lexicon's labelled variants: '[printed variant ... of X; X is the book's usual form]' and '[ZV speakers'
    spelling of X; X is the lexicon's form]' (DECISIONS 18 Oct: the 10 labelled ZV spelling rows);
  2 ch > c (kucincha > kucinca); ng' / ng’ > ŋ;
  3 one vowel doubled (makani-type: a short vowel where the books write a long one);
  4 b > bb, k > kk (one at a time);
  5 u > w and i > y before a vowel (muana > mwana), and back (w > u, y > i); u > w only before a, e, i (the
    spelling.tsv CHECK rule: Cw never before o / u); grammar2/spelling.tsv's sh / sy row is per word only (F068), so
    sh is never changed here;
  6 a split word joined: two tokens whose concatenation is school-attested while the second is not (kuli bambila >
    kulibambila);
  7 reduplication: 'X Y' where X ends with Y (buyumu yumu, abuyumu yumu) > 'X-Y', the books' hyphenated form;
  8 the a- written on a word (akucincha = a + kucincha): the rules on the rest, the a- kept.
Several rules on one word are tried one level deep (ch > c, then a doubled vowel), never more: a far-fetched form is
not a reading of what the teacher typed.
"""
import functools
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CACHE = os.path.join(REPO, "internal", "chitonga_train", "vocab", "attest_cache.json")
LEXICON = os.path.join(REPO, "tools", "chitonga", "grammar2", "lexicon.tsv")
VOW = "aeiou"
TOK = re.compile(r"[A-Za-zŋŊ]+(?:['’][A-Za-zŋŊ]+)*")
LABEL = re.compile(r"\[(?:printed variant|ZV speakers' spelling(?: variant)?)[^\]]*?\bof ([a-zŋ']+); \1 is the "
                   r"(?:book's usual|lexicon's|written) form\]")


@functools.lru_cache(maxsize=1)
def tables():
    import csv
    c = json.load(open(CACHE, encoding="utf-8"))
    lex, var = set(), {}
    for r in csv.DictReader(open(LEXICON, encoding="utf-8"), delimiter="\t"):
        w = (r.get("word") or "").strip().lower()
        m = LABEL.search(r.get("meaning") or "")
        # a verb headword is a bare STEM (bambila), not a written word; a labelled variant is not the school form; a
        # row imported from the corpus with no school-book print (lexicon_corpus_2026-10-20: "0 in the school books")
        # is a spoken form, not the books' spelling
        corpus_only = re.search(r"(?<![0-9])0 in the school books|lexicon_corpus", r.get("source") or "")
        if w and (r.get("pos") or "").strip() != "v" and not m and not corpus_only:
            lex.add(w)
        for pl in re.split(r"[,/ ]+", (r.get("plural") or "").strip().lower()):
            if pl and pl != "of":
                lex.add(pl)
        if m and w and " " not in w:
            var[w] = m.group(1)
    return {"book": c["book"], "lex": lex, "var": var}


def school(w):
    """printed count in the school books, or 2 for a lexicon form (None when neither)"""
    T = tables()
    n = T["book"].get(w, 0)
    if n >= 2:
        return n
    return 2 if w in T["lex"] else None


def _one(w):
    """single-step candidates for one lower-case word"""
    out = []
    if "ch" in w:
        out.append(w.replace("ch", "c"))
    if "ng'" in w or "ng’" in w:
        out.append(w.replace("ng'", "ŋ").replace("ng’", "ŋ"))
    for i, ch in enumerate(w):
        # never at the word's first letter (a- / aa- / u- / uu- are different prefixes: atakwe is not aatakwe)
        if ch in VOW and i > 0 and (i + 1 >= len(w) or w[i + 1] != ch) and w[i - 1] != ch:
            out.append(w[:i + 1] + ch + w[i + 1:])
        if ch in "bk" and (i + 1 < len(w) and w[i + 1] not in (ch, "w", "y")) and (i == 0 or w[i - 1] != ch):
            out.append(w[:i] + ch + w[i:])
        if i + 1 < len(w) and w[i + 1] in VOW:
            # grammar2/spelling.tsv CHECK Cw+o/u (Grade 1 teacher's guide, CDC 2013): a consonant + w takes a, e, i only
            if ch == "u" and i > 0 and w[i - 1] not in VOW and w[i + 1] in "aei":
                out.append(w[:i] + "w" + w[i + 1:])
            if ch == "i" and i > 0 and w[i - 1] not in VOW:
                out.append(w[:i] + "y" + w[i + 1:])
            if ch == "w" and i > 0 and w[i - 1] not in VOW:
                out.append(w[:i] + "u" + w[i + 1:])
            if ch == "y" and i > 0 and w[i - 1] not in VOW:
                out.append(w[:i] + "i" + w[i + 1:])
    return [x for x in dict.fromkeys(out) if x != w]


def word_variant(w, _a_done=False):
    """(school form, rule) or None. The a- rule is tried once, never recursively (Codex #18: 'a' x 1100 hit the
    recursion limit)"""
    lw = w.lower()
    if school(lw) is not None:
        return None
    T = tables()
    if lw in T["var"] and school(T["var"][lw]) is not None:
        return T["var"][lw], "lexicon variant"
    best = None
    lvl1 = _one(lw)
    for c in lvl1:
        n = school(c)
        if n is not None and (best is None or n > best[1]):
            best = (c, n, "one change")
    if best is None and len(lw) >= 5:
        for c in lvl1:
            if "ch" in lw and c == lw.replace("ch", "c") or "ng'" in lw:
                for c2 in _one(c):
                    n = school(c2)
                    if n is not None and (best is None or n > best[1]):
                        best = (c2, n, "two changes")
    if best is None and not _a_done and lw.startswith("a") and 5 < len(lw) <= 40:
        # the conjunctive / augment a- written on a word: akucincha = a + kucinca
        r = word_variant(lw[1:], _a_done=True)
        if r:
            return "a" + r[0], r[1] + " (after a-)"
    return (best[0], best[2]) if best else None


def _case(orig, new):
    return new[:1].upper() + new[1:] if orig[:1].isupper() else new


def _en_vocab():
    """English words seen in lower case in the English of our pairs (h2_text, the translation memory's English)"""
    try:
        import h2_text as T
        return T._en_vocab()
    except Exception:  # noqa: BLE001
        return set()


def normalise(text):
    """(normalised text, [changes]); changes = [{'from', 'to', 'rule'}]. The text is unchanged when nothing applies."""
    import attest
    toks = list(TOK.finditer(text or ""))
    if not toks:
        return text, []
    pieces, changes, last, i = [], [], 0, 0
    initial = True
    while i < len(toks):
        m = toks[i]
        w = m.group(0)
        lw = w.lower()
        pieces.append(text[last:m.start()])
        last = m.end()
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        gap = text[m.end():nxt.start()] if nxt else ""
        name = w[:1].isupper() and not initial

        def ok(tok, first):
            """a token the pair rules may touch: no name (capitalised, not sentence-initial), no English word (Codex
            review 21 Oct #17: 'na Chuma Uma' > 'Chuma-Uma', 'understand stand' > 'understand-stand')"""
            if len(tok) > 40 or (tok[:1].isupper() and not first):
                return False
            if school(tok.lower()) is None and tok.lower() in _en_vocab():
                return False                         # an English word of our pairs, not printed in the school books
            return attest.word(tok)[0] != "english"
        pair_ok = bool(nxt) and gap == " " and ok(w, initial) and ok(nxt.group(0), False)
        # 7 reduplication written apart: buyumu yumu > buyumu-yumu
        if pair_ok and len(nxt.group(0)) >= 3 and lw.endswith(nxt.group(0).lower()) and lw != nxt.group(0).lower():
            pieces.append(w + "-" + nxt.group(0))
            changes.append({"from": w + gap + nxt.group(0), "to": w + "-" + nxt.group(0), "rule": "reduplication"})
            last = nxt.end()
            i += 2
            initial = False
            continue
        # 6 a split word: kuli bambila > kulibambila
        if pair_ok and not name:
            j = lw + nxt.group(0).lower()
            if school(nxt.group(0).lower()) is None and (school(j) or 0) >= 2 and len(lw) <= 5:
                pieces.append(_case(w, j))
                changes.append({"from": w + gap + nxt.group(0), "to": _case(w, j), "rule": "split word joined"})
                last = nxt.end()
                i += 2
                initial = False
                continue
        v = None
        # a token longer than any Tonga word (40 letters) is left as typed (Codex #18: 'a' x 1100)
        if not name and len(w) <= 40 and not re.search(r"\d", w) and attest.word(w)[0] != "english":
            v = word_variant(w)
        if v:
            pieces.append(_case(w, v[0]))
            changes.append({"from": w, "to": _case(w, v[0]), "rule": v[1]})
        else:
            pieces.append(w)
        initial = bool(re.search(r"[.!?]\s*$", gap))
        i += 1
    pieces.append(text[last:])
    return "".join(pieces), changes


if __name__ == "__main__":
    import sys
    print(json.dumps(normalise(sys.argv[1]), ensure_ascii=False, indent=1))
