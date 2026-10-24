"""Chitonga verb and concord rules from Hopgood, A Practical Introduction to Chitonga (2nd ed., ZEPH 1992), as code
(11 Oct 2026). Every rule names the lesson/page it comes from; verify.py tests the rules against every paradigm the
book prints (vision_notes/*.json, Appendix B). A form produced here is a PREDICTION until a speaker confirms it.

Persons: 1s 2s 1p 2p, and noun classes c1 ... c18 (c1 = 'he/she', c2 = 'they' for people).
Tenses implemented (each with its Hopgood source):
  pres       -la- tense        Ndabona/Ndilabona, Tulabona, Ulabona ...           L3 p11
  a          -a- tense         Ndabona, Twabona, Wabona, Babona ...              L3 p11
  remote     -a- + -ka-        Ndakabona, Twakabona ...                          L3 p11, p92
  fut_zoo    -zoo- future      Nzootola, Tuzootola ...                           L25 p96
  fut_yoo    -yoo- future      Bayoolya ...                                      L24 p93
  neg        ta- ... -i        Nsiboni, Toboni, Taboni, Tatuboni ...             L18 p67-68
  neg_fut    -koo-             Nsikoobona, Tokoobona ...                         L69 p289
  perf       perfect stem      Tulisambide (with -li-), bwene ...               L16 p60, p64
  neg_perf   ta- + perfect     Nsibwene, Tobwene ...                             L69 p289
  neg_plup   tee- + -a- + ka + perfect   Teendakabwene ...                        L69 p289-290
  neg_futperf -koozoo- + perfect  Nsikoozoobwene ...                             L69 p290
  past_cont  ka- + concord     Nkembeza, Kobeza, Kabeza, Katubeza ...            L67 p282
  plup       ka- + concord + perfect   Nkemanide, Komanide ...                   L68 p285
  subj       -e subjunctive    Mpale, Ndete, alete, tulete ...                   L22-24 p84, p88, p92, p36
"""
import re

PERSONS = ["1s", "2s", "1p", "2p"] + [f"c{i}" for i in range(1, 19)]

# Subject concord and its form in the -a- tenses (L24 p92-93 summary; Appendix B row "Subjective Concord")
SC = {"1s": ("ndi", "nda"), "2s": ("u", "wa"), "1p": ("tu", "twa"), "2p": ("mu", "mwa"),
      "c1": ("u", "wa"), "c2": ("ba", "ba"), "c3": ("u", "wa"), "c4": ("i", "ya"), "c5": ("li", "lya"), "c6": ("a", "a"),
      "c7": ("ci", "ca"), "c8": ("zi", "zya"), "c9": ("i", "ya"), "c10": ("zi", "zya"), "c11": ("lu", "lwa"),
      "c12": ("tu", "twa"), "c13": ("ka", "ka"), "c14": ("bu", "bwa"), "c15": ("ku", "kwa"), "c16": ("a", "a"),
      "c17": ("ku", "kwa"), "c18": ("mu", "mwa")}
# Object concord (Appendix B row "Object Concord"; L25 p100 complete list)
OC = {"1s": "ndi", "2s": "ku", "1p": "tu", "2p": "mu", "c1": "mu", "c2": "ba", "c3": "u", "c4": "i", "c5": "li",
      "c6": "a", "c7": "ci", "c8": "zi", "c9": "i", "c10": "zi", "c11": "lu", "c12": "tu", "c13": "ka", "c14": "bu",
      "c15": "ku", "c16": "a", "c17": "ku", "c18": "mu"}

# Irregular perfect stems printed by Hopgood (L16 p64, L20 p75, L33 p134, L35 p144, L43 p178, L51 p216)
PERFECT_IRREGULAR = {"bona": "bwene", "kkala": "kkede", "ima": "imvwi", "lala": "lede", "jana": "jene",
                     "bbala": "bbede", "zyala": "zyede", "kotela": "kotede", "buzya": "buzizye", "zyiba": "zyi",
                     "jata": "jisi", "ba": "bede"}


def nasal(stem):
    """First-person-singular nasal concord before a stem (L22-24 p88, p92 sound rules):
    n+l=nd, n+y=nj; before w the concord is ng; before m, n, ny, ŋ it is assimilated (M-mame -> Mame); before e, i:
    nj; before a, o, u: ng; otherwise the homorganic nasal (m before p b bb f v; n before t d s z c j k g)."""
    s = stem
    if s.startswith("l"):
        return "nd" + s[1:]
    if s.startswith("y"):
        return "nj" + s[1:]
    if s.startswith("w"):
        return "ng" + s
    if re.match(r"^(m|n|ŋ)", s):
        return s
    if s[:1] in "ei":
        return "nj" + s
    if s[:1] in "aou":
        return "ng" + s
    if re.match(r"^(p|b|f|v)", s):
        return "m" + s
    return "n" + s


def final(stem, vowel):
    """replace the final -a of a verb stem (negative -i, subjunctive -e; L8 p29, L22 p84)"""
    return stem[:-1] + vowel if stem.endswith("a") else stem + vowel


def perfect(stem):
    """perfect stem: irregulars as printed; verbs in -ala -> -ede (L43 p178); otherwise -ide after a, i, u in the
    root and -ede after e, o (vowel harmony: samba>sambide, lemba>lembede, bola>bolede; L16 p60)"""
    if stem in PERFECT_IRREGULAR:
        return PERFECT_IRREGULAR[stem]
    if stem.endswith("aala") and len(stem) > 4:
        return stem[:-4] + "eede"            # syaala > syeede [pattern by analogy, not printed]
    if stem.endswith("ala") and len(stem) > 3:
        return stem[:-3] + "ede"             # kkala > kkede (L43 p178)
    long = len(re.findall("[aeiou]+", stem)) >= 3   # only longer verbs contract: cela > celede, mana > manide (p60, p285)
    if long and stem.endswith("ela"):
        return stem[:-3] + "ede"             # elela > elede (index)
    if long and stem.endswith("ana"):
        return stem[:-3] + "ene"             # kozyana > kozyene, tangana > tangene (index)
    root = stem[:-1]
    vowels = re.findall(r"[aeiou]", root)
    return root + ("ede" if vowels and vowels[-1] in "eo" else "ide")


def neg_prefix(p):
    """negative ta- + subject concord with Hopgood's coalescence (L18 p67-68): 1s Nsi-, 2s ta+u=to, 3s (c1) ta+a=ta,
    ta+i=tii (c4, c9: tiikonzyi), otherwise ta + concord (tatu-, tamu-, taba-, taci- ...)"""
    if p == "1s":
        return "nsi"
    if p in ("2s", "c3"):
        return "to"
    if p == "c1":
        return "ta"
    sc = SC[p][0]
    if sc == "i":
        return "tii"
    if sc == "a":
        return "ta"
    return "ta" + sc


def ka_prefix(p, stem_initial_form):
    """ka- + subject concord (past continuous, alternative pluperfect; L67 p281-282, L68 p285):
    1s Nke + nasal (Nkembeza, Nkemanide), 2s ka+u=ko, 3s ka+a=ka, otherwise ka + concord."""
    if p == "1s":
        return "nke" + nasal(stem_initial_form)
    if p in ("2s", "c3"):
        return "ko" + stem_initial_form
    if p == "c1":
        return "ka" + stem_initial_form
    sc = SC[p][0]
    return "ka" + ("" if sc == "a" else sc) + stem_initial_form


def conjugate(stem, person, tense):
    """the form of verb `stem` (infinitive without ku-, e.g. 'bona') for `person` in `tense`; returns a list of
    forms (alternatives first-listed is Hopgood's main form). Raises KeyError for an unknown tense."""
    sc, sca = SC[person]
    if tense == "pres":
        if person == "1s":
            return [nasal("la" + stem).replace("ndla", "nda", 1) if False else "nda" + stem, "ndila" + stem]  # n+la = nda
        return [sc + "la" + stem]
    if tense == "a":
        return [sca + stem]
    if tense == "remote":
        return [sca + "ka" + stem]
    if tense == "fut_zoo":
        return [nasal("zoo" + stem)] if person == "1s" else [sc + "zoo" + stem]
    if tense == "fut_yoo":
        return [nasal("yoo" + stem)] if person == "1s" else [sc + "yoo" + stem]
    if tense == "neg":
        if stem == "li":                     # 'be' keeps -li in the negative: tiili, tatuli, tamuli (L69 p289)
            return [neg_prefix(person) + "li"]
        return [neg_prefix(person) + final(stem, "i")]
    if tense == "neg_fut":
        return [neg_prefix(person) + "koo" + stem]
    if tense == "perf":
        return [perfect(stem)]
    if tense == "neg_perf":
        return [neg_prefix(person) + perfect(stem)]
    if tense == "neg_plup":
        return ["tee" + sca + "ka" + perfect(stem)]
    if tense == "neg_futperf":
        return [neg_prefix(person) + "koozoo" + perfect(stem)]
    if tense == "past_cont":
        return [ka_prefix(person, stem)]
    if tense == "plup":
        return [ka_prefix(person, perfect(stem))]
    if tense == "subj":
        if person == "1s":
            return [nasal(final(stem, "e"))]
        if person == "c1":
            return ["a" + final(stem, "e")]  # L10 p36, L24 p92: a- for class 1 in the subjunctive
        return [sc + final(stem, "e")]
    raise KeyError(tense)


if __name__ == "__main__":
    import sys
    stem = sys.argv[1] if len(sys.argv) > 1 else "bona"
    for t in ("pres", "a", "remote", "fut_zoo", "neg", "neg_fut", "neg_perf", "neg_plup", "past_cont", "subj"):
        print(t.ljust(12), ", ".join(conjugate(stem, p, t)[0] for p in ("1s", "2s", "c1", "1p", "2p", "c2")))
