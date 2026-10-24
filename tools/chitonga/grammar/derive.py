"""Derived words from base verbs and adjectives, by the rules Hopgood states (2nd ed., ZEPH 1992) (11 Oct 2026; Chris:
"make all the derivatives and mark them as such"). Every generated word is SYNTHETIC (origin synthetic:derivation)
and a prediction until a speaker confirms it; derived NOUNS are lexical (not every verb has every noun, meanings
shift), so they are marked lower confidence than derived verb forms.

    python -X utf8 tools/chitonga/grammar/derive.py test         # rules against the 109 derived forms Hopgood's index lists
    python -X utf8 tools/chitonga/grammar/derive.py generate     # all derivatives of the candidate list

Output (generate): internal/chitonga_train/derived/derivations.jsonl, one line per derived word:
 {base, base_pos, base_meaning, derived, type, rule, source, confidence, attested, origin, review}
attested = "hopgood_index" (Hopgood lists it), "corpus" (the exact word occurs in our Chitonga text), or null.
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "internal", "chitonga_train", "derived")
RUN = os.path.join(REPO, "internal", "chitonga_train", "runs", "2026-10-11_stage5")
V = "aeiou"


def last_vowel(s):
    v = re.findall(r"[aeiou]", s)
    return v[-1] if v else "a"


def mono(stem):
    """one-syllable verb stems (lya, nywa, fwa, pa, ba, za, ya, twa, bba, mvwa)"""
    return len(re.findall(r"[aeiou]+", stem)) == 1


def applied(stem):
    """Applied ('for, to, on behalf of'): L21 p79 -ila after a, i, u and -ela after e, o; L22 p84 -ina/-ena after
    a final -ma/-na; L23 p89 one-syllable verbs -ida/-eda (lya > lida, nywa > nywida); L51 p216 -sya > -sizya;
    L52 p221 -zya > -zizya/-zezya (katazya > katazizya, bozya > bozezya)."""
    if mono(stem):
        # one-syllable verbs: a y-glide drops (lya > lida), other glides stay (nywa > nywida)
        return (stem[:-2] if stem[-2:-1] == "y" else stem[:-1]) + "ida"
    root = stem[:-1]
    # L60 p255: verbs in -ola, -ula (and -una) make the applied form in -weda/-wida (tulula > tulwida, yumuna >
    # yumwida, pasaula > pasawida)
    long = len(re.findall(r"[aeiou]+", stem)) >= 3  # only longer verbs: tola > tolela, ula > ulila stay regular
    if long and stem.endswith(("ula", "una")):
        return stem[:-3] + "wida"
    if long and stem.endswith("ola"):
        return stem[:-3] + "weda"
    if stem.endswith(("sya", "zya")):
        e = "e" if last_vowel(stem[:-3]) in "eo" else "i"
        return stem[:-3] + stem[-3:-2] + e + "zya"
    harm = "e" if last_vowel(root) in "eo" else "i"
    if stem.endswith(("ma", "na")):
        return root + harm + "na"
    return root + harm + "la"


def causative(stem):
    """Causative ('make, cause to'): L10 p37 insert y before the final a (lima > limya); L31 p121 and L66 p273
    -nga/-nda > -nzya (anga > anzya, binda > binzya); L35 p139 and L42 p174 -la > -zya (njila > njizya)."""
    if mono(stem):
        return stem[:-1] + "isya"            # L66 p277: wa > wisya, nywa > nywisya
    if stem.endswith(("nga", "nda")):
        return stem[:-3] + "nzya"
    if stem.endswith("ka"):
        return stem[:-2] + "sya"             # L44 p183 yaka > yasya; leka > lesya, seluka > selusya (index)
    if stem.endswith(("ya", "la")):
        return stem[:-2] + "zya"             # L66 p277 jaya > jazya; L35 p139, L42 p174 njila > njizya, ula > uzya
    if stem.endswith("wa"):
        return stem[:-2] + "sya"             # yoowa > yoosya (index) [pattern, not a stated rule]
    if stem.endswith("ina"):
        return stem[:-3] + "izya"            # zumina > zumizya (index) [pattern, not a stated rule]
    return stem[:-1] + "ya"


def passive(stem):
    """Passive: L20 p75 -wa (bona > bonwa); one-syllable verbs -igwa/-egwa (twigwa, bbigwa, ligwa, nywigwa; pa >
    pegwa as printed, p75); verbs in -sya/-zya > -sigwa/-zigwa, -segwa/-zegwa by the root vowel (kasya > kasigwa,
    lesya > lesegwa, penzya > penzegwa, langilizya > langilizigwa; notes)."""
    if stem in ("pa", "lya"):
        return {"pa": "pegwa", "lya": "ligwa"}[stem]          # as printed, p75
    if mono(stem):
        return stem[:-1] + ("egwa" if last_vowel(stem[:-1]) in "eo" else "igwa")
    if stem.endswith(("sya", "zya")) and len(stem) > 3:
        return stem[:-2] + ("egwa" if last_vowel(stem[:-3]) in "eo" else "igwa")
    return stem[:-1] + "wa"


def reciprocal(stem):
    """Reciprocal ('each other'): L51 p216 -ana (bona > bonana)."""
    if stem.endswith(("sya", "zya")):
        return stem + "nya"                  # enzya > enzyanya (index) [pattern, not a stated rule]
    return stem[:-1] + "ana"


def intensive(stem):
    """Intensive ('well, thoroughly'): L27 p104 -isya/-esya (bona > bonesya)."""
    return stem[:-1] + ("esya" if last_vowel(stem[:-1]) in "eo" else "isya")


def agent_si(stem):
    """Agent noun with si- before the infinitive: L33 p129 (sikuula 'buyer' from kuula); plural ba-."""
    return "si" + "ku" + stem, "basi" + "ku" + stem


def agent_mu(stem):
    """Agent noun mu- ... -i (mukambausi 'preacher', mulozi 'sorcerer', in Hopgood's index; a general Bantu
    pattern); plural ba-. Sound changes before -i are not modelled: a prediction to check."""
    root = stem[:-1]
    return "mu" + root + "i", "ba" + root + "i"


def instrument(stem):
    """Instrument or place noun ci- + applied form with final -o: L23 p89 (nywida > cinywido 'drinking vessel');
    plural zi-."""
    ap = applied(stem)
    return "ci" + ap[:-1] + "o", "zi" + ap[:-1] + "o"


def abstract(adj):
    """Abstract noun bu- + adjective stem (bukali 'anger, fierceness' from -kali; L1, index)."""
    return "bu" + adj


VERB_RULES = [  # type, function, source, confidence
    ("applied", applied, "L21 p79; L22 p84; L23 p89; L51 p216; L52 p221", "rule"),
    ("causative", causative, "L10 p37; L31 p121; L35 p139; L42 p174; L66 p273", "rule"),
    ("passive", passive, "L20 p75; L53 p226", "rule"),
    ("reciprocal", reciprocal, "L51 p216", "rule"),
    ("intensive", intensive, "L27 p104", "rule"),
]
NOUN_RULES = [
    ("agent_si", agent_si, "L33 p129", "lexical"),
    ("agent_mu", agent_mu, "index (mukambausi, mulozi); Bantu pattern", "lexical"),
    ("instrument_ci", instrument, "L23 p89", "lexical"),
]
TYPE_OF = {"app": "applied", "caus": "causative", "pass": "passive", "recip": "reciprocal", "intens": "intensive"}


def test():
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(OUT, "attested_derivations.tsv"), encoding="utf-8")][1:]
    fn = {t: f for t, f, _, _ in VERB_RULES}
    by = collections.defaultdict(lambda: [0, 0])
    misses = []
    for derived, typ, base, meaning in rows:
        t = TYPE_OF.get(typ)
        if not t or not base:
            continue
        base = base.strip("-")
        got = fn[t](base)
        by[t][1] += 1
        if got == derived:
            by[t][0] += 1
        else:
            misses.append(f"  {t:10s} {base:12s} Hopgood {derived:14s} rule {got}")
    tot = sum(n for _, n in by.values())
    ok = sum(a for a, _ in by.values())
    print(f"Hopgood's listed derivations reproduced: {ok}/{tot}")
    for t, (a, n) in sorted(by.items()):
        print(f"  {t:10s} {a}/{n}")
    print("\n".join(misses))


def generate():
    sys.path.insert(0, HERE)
    cands = [l.rstrip("\n").split("\t") for l in open(os.path.join(OUT, "candidates.tsv"), encoding="utf-8")][1:]
    listed = {r.split("\t")[0] for r in open(os.path.join(OUT, "attested_derivations.tsv"), encoding="utf-8")}
    words = collections.Counter()
    for name in ("train.jsonl", "mono.jsonl"):
        for l in open(os.path.join(RUN, name), encoding="utf-8"):
            words.update(w.lower() for w in re.findall(r"[a-zA-Zŋ']+", json.loads(l).get("toi") or ""))
    out = []

    def add(base, pos, meaning, derived, typ, src, conf, plural=None):
        att = "hopgood_index" if derived in listed else "corpus" if words.get(derived) else None
        rec = {"base": base, "base_pos": pos, "base_meaning": meaning, "derived": derived, "type": typ,
               "rule": typ, "source": src, "confidence": conf, "attested": att, "origin": "synthetic:derivation",
               "review": "unreviewed"}
        if plural:
            rec["plural"] = plural
        out.append(rec)

    for rank, word, pos, meaning, *_ in cands:
        if pos == "v" and word.endswith("a"):
            add(word, pos, meaning, "ku" + word, "infinitive_noun", "L6 p21", "rule")
            for typ, f, src, conf in VERB_RULES:
                add(word, pos, meaning, f(word), typ, src, conf)
            for typ, f, src, conf in NOUN_RULES:
                sg, pl = f(word)
                add(word, pos, meaning, sg, typ, src, conf, pl)
        elif pos == "adj":
            add(word, pos, meaning, abstract(word), "abstract_bu", "L1; index (bukali)", "lexical")
    with open(os.path.join(OUT, "derivations.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    c = collections.Counter(r["type"] for r in out)
    a = collections.Counter(r["type"] for r in out if r["attested"])
    print(f"{len(out)} derived words from {len(cands)} base words")
    for t, n in c.most_common():
        print(f"  {t:16s} {n:5d}   attested (index or corpus): {a.get(t, 0)}")


if __name__ == "__main__":
    {"test": test, "generate": generate}[sys.argv[1] if len(sys.argv) > 1 else "test"]()
