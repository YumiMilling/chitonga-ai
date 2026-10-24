"""Chitonga noun-side agreement from Hopgood's Table of Concords (Appendix B, 2nd ed., ZEPH 1992; transcribed in
vision_notes/appendix_b_concords.json) (11 Oct 2026). The table is the data; this file only reads it and applies the
joining rules Hopgood states. Main form = the first form in a cell; forms in brackets are dialect variants and are
kept as alternatives.

    python tools/chitonga/grammar/concords.py          # print the table as the engine reads it
"""
import json
import os
import re

TABLE = r"..\zamai-corpus-raw\chitonga_books\hopgood_1992\vision_notes\appendix_b_concords.json"
COLS = ["1s", "2s", "1p", "2p"] + [f"c{i}" for i in range(1, 19)]

_t = json.load(open(TABLE, encoding="utf-8"))
ROWS = {}
for r in _t["rows"]:
    label = ((r.get("group") or "") + " " + r["label"]).strip()
    ROWS[label] = dict(zip(COLS, r["cells"]))


def row(label_part):
    hits = [k for k in ROWS if label_part.lower() in k.lower()]
    if not hits:
        raise KeyError(label_part)
    return ROWS[hits[0]]


def forms(cell):
    """all forms in a cell, main first; dialect variants (in brackets) after; hyphens and '?' kept as printed"""
    cell = (cell or "").replace("�", "").strip()
    main, variants = [], []
    for part in re.split(r"\s*/\s*", cell):
        inside = re.findall(r"\(([^)]*)\)?", part)
        outside = re.sub(r"\([^)]*\)?", "", part)
        for x in re.split(r"[,;]\s*", outside):
            x = x.strip(" .")
            if x and x not in ("—", "-"):
                main.append(x)
        for grp in inside:
            variants += [x.strip(" .") for x in re.split(r"[,;]\s*", grp) if x.strip(" .")]
    return main + variants


def first(label_part, col):
    f = forms(row(label_part).get(col))
    return f[0] if f else None


def join(prefix, rest):
    """glue a concord to what follows, with the vowel coalescence Hopgood describes: a+i -> e (wa+isu = wesu,
    L5 p17), a+u -> o, a+a -> aa; a final hyphen marks the joining point"""
    p = prefix.strip("-").lower()
    r = rest.lstrip("-")
    if p.endswith("aa") and r[:1] == "i":
        return p[:-2] + "e" + r[1:]          # class 6 aa- + -isu = esu (as printed; L5)
    if p.endswith("a") and r[:1] == "i":
        return p[:-1] + "e" + r[1:]
    if p.endswith("a") and r[:1] == "u":
        return p[:-1] + "o" + r[1:]
    return p + r


def genitive(c):
    """the 'of' concord for a possessed noun of class c (Appendix B 'Possessive Concords: 1. of')"""
    return first("1. of", c)


def possessive(c, owner):
    """'my/thy/his/our/your/their' agreeing with a possessed noun of class c: genitive concord + possessive suffix
    (Appendix B 'Possessive Suffixes'; L4-L6 p14, p17, p21, p25, p28, p50). e.g. c2 + 1s = bangu, c7 + 1p = cesu.
    Returns every form the table allows (both suffixes where it prints two, e.g. 2p -inu and -nu), main form first."""
    out = []
    for g in forms(row("1. of").get(c))[:1]:
        for suf in forms(row("Possessive Suffixes").get(owner)):
            f = join(g, suf)
            if f not in out:
                out.append(f)
    return out


def demonstrative(c, n):
    """demonstrative type n (1 'this', 2 'that', 3 'yonder', 4 'this' -no forms) for class c (Appendix B)"""
    return forms(row(f"Demonstratives: {n}.").get(c) if n > 1 else row("Demonstratives: 1.").get(c))


def adjective(c, stem):
    """adjective agreeing with class c, from the printed -botu row: replace 'botu' by the stem (Appendix B).
    Classes 9 and 10 take a nasal that fuses with the stem (im-botu, im-bi; L7 p24, L22 p88 sound rules): before a
    nasal-initial stem there is no extra nasal, and class 10 then uses zi- (zinji 'many', as printed)."""
    if c in ("c9", "c10"):
        if stem[:1] in ("m", "n", "\u014b"):
            return ("zi" if c == "c10" else "i") + stem
        import tonga
        return "i" + tonga.nasal(stem)
    model = first("Adjective -botu", c)
    return model[:-4] + stem if model and model.endswith("botu") else None


def o_concord(c, stem):
    """quantifiers and numerals with the o- concord (Appendix B rows '-onse (all)', 'Numerals'): the prefix of the
    printed -onse form + the stem: boonse > b + otatwe = botatwe; zyoonse > zy + otatwe = zyotatwe (L11 p40)."""
    model = first("-onse (all)", c)
    if not model or not model.endswith("oonse"):
        return None
    if stem in ("onse", "oonse"):
        return model                     # the printed form itself
    return model[:-5] + (stem if stem.startswith("o") else "o" + stem)


EXTRA_PREFIXES = {"c6": ["ma"],  # the table heads class 6 'A-'; its nouns take ma- (maamba, matwi; L14 p50)
                  "c9": ["im", "in", "iŋ"], "c10": ["im", "in", "iŋ"]}  # iŋ- before vowels (L7 p24)
LOCATIVE = ("ku", "mu", "a")  # place prefixes in front of a noun (L10 p36); agreement follows the inner noun


def noun_classes(noun):
    """candidate classes for a noun from its prefix (Appendix B 'Noun Prefixes'); several when prefixes coincide"""
    n = noun.lower().replace("-", "")
    out = [(len(p), c) for c, ps in EXTRA_PREFIXES.items() for p in ps if n.startswith(p) and len(n) > len(p)]
    for c in COLS[4:]:
        for p in forms(row("Noun Prefixes").get(c)):
            for alt in re.split(r"[ ,]+", p.lower().replace("etc.", "")):
                alt = alt.strip("-. ")
                if alt and n.startswith(alt) and len(n) > len(alt):
                    out.append((len(alt), c))
    best = max((l for l, _ in out), default=0)
    found = [c for l, c in out if l == best]
    if not found and n.startswith("mw"):  # mu- becomes mw- before a vowel (L4 p13): mweeta = mu-eeta
        return noun_classes("mu" + n[2:])
    if not found and n[:1] in ("ŋ", "n", "m") and not n.startswith(("mu", "mi", "ma")):
        return noun_classes("i" + n)  # the initial i of im-/in-/iŋ- nouns drops after a particle (L7 p24): a-ŋanda
    return found


if __name__ == "__main__":
    for label in ROWS:
        print(label[:40].ljust(40), " | ".join((forms(ROWS[label].get(c)) or ["-"])[0] for c in COLS[4:13]))
