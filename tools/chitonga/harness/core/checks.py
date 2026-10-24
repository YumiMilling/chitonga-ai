"""Checks on Tonga that no teacher or book wrote (origin D), used at plug-in build time and by the renderer.

Alarms, never verdicts (memory: distrust automatic tests; teachers settle meaning). A line with a flag is served in
English with the flag in the log, never silently in Tonga.

  numbers   every number in the English must be in the Tonga, as digits or as a Tonga number word / stem (lexicon
            1-10: omwe, -bili/-bilo/-bile, -tatwe, -ne, -sanwe/-sanu, cisambomwi, ciloba, lusele, fwuka, kkumi; tens
            makumi; hundreds from the speakers' maths glossary, not in this copy). Stems, because concord changes the prefix
            (tobilo, zyobilo, bobilo ...).
  glossary  the plug-in glossary's forbids: a form a teacher ruled wrong for an English term (two cases from
            speaker corrections) or the term guard's (kkolona for cholera, soomona for dizzy).
  attest    (optional, a callable = attest.line) words whose verdict is unattested / not_tonga / contact / mixed /
            english (books are the norm for form; contact = leakage from Bemba / Nyanja: brief_parts/related_languages).
  negation  (optional, a callable = negation_sweep.auto_flags wrapper) neg_dropped / obl_dropped / neg_added.
Pure functions; attest and negation are passed in (tests run without the lexicon).
"""
import re

NUM_STEMS = {0: ["nooto", "kkoto"], 1: ["mwe"], 2: ["bili", "bilo", "bile"], 3: ["tatwe", "tatu"], 4: ["one", "ne"],
             5: ["sanwe", "sanu"], 6: ["sambomwi"], 7: ["ciloba", "loba"], 8: ["lusele", "sele"],
             9: ["fwuka", "fuka"], 10: ["kkumi", "kumi"]}
FAIL_VERDICTS = ("unattested", "not_tonga", "contact", "mixed", "english")


def number_ok(n, toi):
    low = (toi or "").lower()
    if re.search(r"(?<!\d)" + str(n) + r"(?!\d)", low):
        return True
    if n in NUM_STEMS:
        return any(s in low for s in NUM_STEMS[n])
    if n >= 100:
        return "mwaanda" in low or "myaanda" in low
    if n > 10:
        return "kumi" in low
    return False


def numbers(en, toi):
    return [f"numbers: {n} not in the Tonga" for n in map(int, re.findall(r"(?<![\d/])\d{1,4}(?![\d/])", en or ""))
            if not number_ok(n, toi)]


def glossary_flags(en, toi, glossary):
    out = []
    for g in glossary or []:
        for fb in g.get("forbid") or []:
            if re.search(fb["when"], en or "") and re.search(r"(?i)" + re.escape(fb["form"]), toi or ""):
                out.append(f"glossary: '{fb['form']}' for '{g['en']}' ({fb['why']})")
    return out


def check_line(en, toi, glossary=(), attest=None, negation=None):
    """[flags] for one English / Tonga pair; [] = no alarm rang (not a verdict that the line is right)"""
    if not toi:
        return ["no Tonga"]
    fl = numbers(en, toi) + glossary_flags(en, toi, glossary)
    if attest:
        # a quoted English word is the lesson's content ('the' in a sight-word line), not leakage
        bare = re.sub(r"'[^']{1,20}'|\"[^\"]{1,20}\"", " ", toi)
        for verdict, w, why in attest(bare):
            if verdict in FAIL_VERDICTS:
                fl.append(f"attest: {w} {verdict} ({why[:80]})")
    if negation:
        fl += [f"negation: {x}" for x in negation(en, toi)]
    return fl
