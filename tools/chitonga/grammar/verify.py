"""Test the Chitonga rules (tonga.py) against the forms Hopgood PRINTS (11 Oct 2026). The expected forms are read
from the transcribed notes (vision_notes/pNNN.json), never typed in here: this file only says which paradigm on
which page holds which verb, tense and persons.

    python tools/chitonga/grammar/verify.py            # summary + every mismatch
Comparison ignores case, the word-join hyphens, punctuation and brackets; a printed "(or X" alternative counts.
A rule is trusted for generation only when all its printed forms match.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grammar2"))
tonga = importlib.import_module(os.environ.get("CHITONGA_ENGINE", "tonga"))   # engine2: the table engine

NOTES = r"..\zamai-corpus-raw\chitonga_books\hopgood_1992\vision_notes"
ROWS6 = ["1s", "1p", "2s", "2p", "c1", "c2"]          # p11 layout: three lines of two forms, read across
GRID = [("1s", "1p"), ("2s", "2p"), ("c1", "c2")]     # singular | English | plural | English

# (page, title contains, verb, tense, layout); layout: "rows6", "grid", "grid_from:<row index of 1s/2s/c1>",
# or a list of (row, col, person[, verb]) cells
SPEC = [
    (11, "Paradigm of bona in the -la- tense", "bona", "pres", "rows6"),
    (11, "Paradigm of bona in the -a- tense", "bona", "a", "rows6"),
    (11, "Remote past with -ka-", "bona", "remote", [(1, 0, "1s")]),
    (7, "Forms of ya", "ya", "pres", [(0, 1, "1p"), (0, 2, "2p"), (0, 3, "c2"), (0, 4, "c1")]),
    (7, "Forms of ya", "ya", "a", [(1, 1, "1p"), (1, 2, "2p"), (1, 3, "c2"), (1, 4, "c1")]),
    (4, "Subject concord varying", "fwa", "a", [(1, 0, "c2"), (2, 0, "c13")]),
    (40, "Examples of ci-/zi- subject concords", None, None,
     [(0, 0, "c7", "cisa", "pres"), (1, 0, "c7", "fwa", "a"), (2, 0, "c8", "panda", "pres"), (3, 0, "c8", "sweeka", "a")]),
    (68, "Negative verb paradigm of bona", "bona", "neg", "grid"),
    (67, "Negative verb: coalescence", None, None, [(0, 1, "2s", "bona", "neg"), (1, 1, "c1", "bona", "neg"),
                                                    (2, 1, "c9", "konzya", "neg")]),
    (96, "Future tense with -zoo-", "tola", "fut_zoo", "grid"),
    (93, "Future tense with -yoo-", "lya", "fut_yoo", [(0, 0, "c2")]),
    (282, "Past continuous tense (b)", "beza", "past_cont", "grid"),
    (285, "Pluperfect (alternative form)", "mana", "plup", "grid"),
    (289, "Negative immediate future", "bona", "neg_fut", "grid"),
    (289, "Negative perfect", "bona", "neg_perf", "grid"),
    (289, "Negative pluperfect", "bona", "neg_plup", [(0, 0, "1s"), (0, 2, "1p")]),
    (290, "Negative pluperfect", "bona", "neg_plup", [(0, 0, "2s"), (0, 2, "2p"), (1, 0, "c1"), (1, 2, "c2")]),
    (290, "Negative future perfect", "bona", "neg_futperf", "grid"),
    (36, "Subjunctive subject concord a-", "leta", "subj", [(0, 0, "c1")]),
    (84, "First person singular of the subjunctive", None, "subj", "verbcol:0:1:1s"),
    (92, "First person singular subjunctive before vowels", None, None,
     [(0, 0, "1s", "amba", "subj"), (1, 0, "1s", "ona", "subj"), (2, 0, "1s", "ula", "subj"), (3, 0, "1s", "ita", "subj"),
      (4, 0, "1s", "embela", "subj")]),
    (88, "First person singular of the subjunctive: forms", None, None,
     [(0, 0, "1s", "leta", "subj"), (1, 0, "1s", "ya", "subj")]),
    (60, "Perfect form of the verb", None, "perf", "verbcol:0:1:"),
    (64, "Irregular perfect verb forms", None, "perf", "verbcol:0:2:"),
    (75, "Perfect verb forms met in Lesson 20", None, "perf", "verbcol:0:2:"),
    (134, "Irregular perfect forms", None, "perf", "verbcol:0:1:"),
    (144, "Perfect verb forms", None, "perf", "verbcol:0:1:"),
    (178, "Perfect forms of verbs in -ala", None, "perf", "verbcol:0:1:"),
    (216, "Perfect form", None, "perf", "verbcol:0:2:"),
]


def norm(s):
    s = (s or "").lower().replace("’", "").replace("'", "")
    s = re.sub(r"[-.,;:!?()\[\]\"“”]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def alternatives(cell):
    parts = re.split(r"\(or\s+|\bor\b", cell or "")
    return [norm(p) for p in parts if norm(p)]


def paradigm(page, title):
    d = json.load(open(os.path.join(NOTES, f"p{page:03d}.json"), encoding="utf-8"))
    for p in d.get("paradigms") or []:
        if title.lower() in (p.get("title") or "").lower():
            return p["rows"]
    raise LookupError(f"p{page}: no paradigm titled '{title}'")


def cases(entry):
    page, title, verb, tense, layout = entry
    rows = paradigm(page, title)
    out = []
    if layout == "rows6":
        for i, person in enumerate(ROWS6):
            out.append((rows[i][0], verb, tense, person))
    elif layout == "grid":
        for i, (a, b) in enumerate(GRID):
            out.append((rows[i][0], verb, tense, a))
            out.append((rows[i][2], verb, tense, b))
    elif isinstance(layout, str) and layout.startswith("verbcol:"):
        _, vc, fc, person = layout.split(":")
        for r in rows:
            out.append((r[int(fc)], norm(r[int(vc)]), tense, person or None))
    else:
        for c in layout:
            r, col, person = c[0], c[1], c[2]
            v = c[3] if len(c) > 3 else verb
            t = c[4] if len(c) > 4 else tense
            out.append((rows[r][col], v, t, person))
    return [(page, title) + x for x in out]


TAM = {"la": "pres", "ka": "remote", "zoo": "fut_zoo", "yoo": "fut_yoo", "koo": "neg_fut"}
SC_OF = {}
for _p, (_a, _b) in tonga.SC.items():
    SC_OF.setdefault(_a, _p)
    SC_OF.setdefault(_b + "+a", _p)


def drill_test():
    """Independent test: verb forms with Hopgood's joins in the drill sentences (never used to write the rules).
    Patterns: SC-la-stem (pres), SCa-ka-stem (remote), SC-zoo/yoo-stem, ta-SC-stem (neg, stem ends in -i)."""
    import glob
    seen, total, ok, misses = set(), 0, 0, []
    for f in glob.glob(os.path.join(os.path.dirname(NOTES), "vision", "p*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for it in d.get("drill") or []:
            for tok in re.findall(r"[A-Za-zŋ']+(?:-[A-Za-zŋ']+)+", it.get("toi") or ""):
                parts = tok.lower().split("-")
                case = None
                if len(parts) >= 3 and parts[1] in ("la", "zoo", "yoo") and parts[0] in SC_OF:
                    case = (SC_OF[parts[0]], "".join(parts[2:]), TAM[parts[1]])
                elif len(parts) >= 3 and parts[1] == "ka" and parts[0].endswith("a"):
                    p = next((k for k, (a, b) in tonga.SC.items() if b == parts[0]), None)
                    if p:
                        case = (p, "".join(parts[2:]), "remote")
                elif len(parts) >= 3 and parts[0] == "ta" and parts[1] in SC_OF and parts[-1].endswith("i"):
                    case = (SC_OF[parts[1]], "".join(parts[2:])[:-1] + "a", "neg")
                if not case or tok.lower() in seen:
                    continue
                seen.add(tok.lower())
                person, stem, tense = case
                got = tonga.conjugate(stem, person, tense)
                total += 1
                # Hopgood writes some negatives analysed (ta-a-siki) and states their written form is fused (L18 p67)
                fused = re.sub("^ta-a-", "ta", re.sub("^ta-u-", "to", re.sub("^ta-i-", "tii", tok.lower())))
                fused = fused.replace("-la-e", "-lee")   # L29 p116: ula-enda is written uleenda
                if any(norm(g) in (norm(tok), norm(fused)) for g in got):
                    ok += 1
                else:
                    misses.append(f"  {tok:22s} ({tense}, {person}, stem {stem})  rule gave {', '.join(got)}")
    print(f"\nINDEPENDENT test, verb forms in the drills: {total}, reproduced {ok} ({100 * ok / max(1, total):.0f}%)")
    for m in misses[:40]:
        print(m)


def possessive_test():
    """Independent test of the noun side: 'noun owner-possessive' phrases in the drills (Bana ba-ngu, Mi-nwe ya-ko).
    The class comes from the NOUN's prefix; the engine builds the possessive from the concord table; a hit when the
    form for any candidate class (prefixes that coincide, e.g. mu- for classes 1, 3, 18) equals the printed form."""
    import glob
    import concords
    owner = {"ngu": "1s", "ko": "2s", "kwe": "c1", "isu": "1p", "esu": "1p", "inu": "2p", "nu": "2p", "bo": "c2"}
    total = ok = 0
    misses, noclass = [], 0
    for f in glob.glob(os.path.join(os.path.dirname(NOTES), "vision", "p*.json")):
        for it in json.load(open(f, encoding="utf-8")).get("drill") or []:
            for m in re.finditer(r"(\S+)\s+([A-Za-z]+-(ngu|ko|kwe|isu|esu|inu|nu|bo))\b", it.get("toi") or ""):
                noun, poss, suf = m.group(1).strip(".,?!"), m.group(2), m.group(3)
                cands = concords.noun_classes(noun)
                n0 = noun.lower().replace("-", "")
                for loc in concords.LOCATIVE:  # ku-muunda, a-ŋanda: the possessive agrees with the inner noun
                    if n0.startswith(loc) and len(n0) > len(loc) + 2:
                        cands += [c for c in concords.noun_classes(n0[len(loc):]) if c not in cands]
                if not cands:
                    noclass += 1
                    continue
                got = {f for c in cands for f in concords.possessive(c, owner[suf])}
                total += 1
                if norm(poss) in {norm(g) for g in got if g}:
                    ok += 1
                else:
                    misses.append(f"  {noun} {poss}: classes {cands} -> {sorted(g for g in got if g)}")
    print(f"\nINDEPENDENT test, possessives in the drills: {total}, reproduced {ok} ({100 * ok / max(1, total):.0f}%); "
          f"{noclass} nouns whose class the prefix does not show")
    for m in misses[:30]:
        print(m)


def main():
    total = ok = 0
    by_tense = {}
    misses = []
    for entry in SPEC:
        for page, title, printed, verb, tense, person in cases(entry):
            if "[?]" in (printed or ""):
                continue
            got = [tonga.perfect(verb)] if tense == "perf" else tonga.conjugate(verb, person, tense)
            want = alternatives(printed)
            # the printed cell may hold a whole phrase: compare against its first word(s) of the same length
            hit = any(norm(g) == w or norm(g) in w.split() for g in got for w in want)
            total += 1
            ok += hit
            if tense == "perf" and verb in tonga.PERFECT_IRREGULAR:
                tense = "perf (lookup)"   # copied from these very pages: a match proves nothing
            b = by_tense.setdefault(tense, [0, 0])
            b[0] += hit
            b[1] += 1
            if not hit:
                misses.append(f"  p{page} {tense:11s} {verb:8s} {person or '':4s} printed '{printed}'  rule gave '{', '.join(got)}'")
    print(f"printed forms checked: {total}, reproduced: {ok} ({100 * ok / max(1, total):.0f}%)")
    for t, (a, n) in sorted(by_tense.items()):
        print(f"  {t:12s} {a}/{n}")
    if misses:
        print("mismatches:")
        print("\n".join(misses))
    drill_test()
    possessive_test()


if __name__ == "__main__":
    main()
