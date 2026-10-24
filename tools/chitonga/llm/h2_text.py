"""Harness v2, text helpers (17 Oct 2026; TODO_ENGINE_HARNESS F2, F3, E2 numbers and names).

  split(text, max_words=25)   F2: long input is cut into sentences, then clauses, before it is translated, and the
                              pieces are rejoined after. Why: the stage 12b back-translation collapses above ~25 words
                              (bt_band_check, 17 Oct: long lines come back shortened, half the meaning gone), and the
                              chat model drifts on long prompts the same way.
  en_names(en)                capitalised English words that are names (not sentence-initial, not I / days / months /
                              languages / God); sentence-initial words only when the gazetteer knows them as a place.
                              22 Oct 2026: an English dictionary word (hunspell en_GB/en_US, cached in
                              internal/chitonga_train/vocab/en_dictionary_words.json) is ordinary too, so Title Case
                              headings are translated (weakness map item 3); given names that are English words
                              (EN_WORD_GIVEN_NAMES), dictionary proper names and the provinces (PROVINCE_WORDS) stay names.
  protect_en(en) / restore()  F3, English > Tonga: names are swapped for sex-neutral English stand-ins (Alex, Sam;
                              the style names.py measured as kept best by our translator, 16 Oct) and put back after.
  numbers_en(text)            digits and English number words -> values
  numbers_toi(text)           digits and Tonga numerals -> values. The numerals are read from lexicon.tsv AT RUN TIME
                              (pos num / enum, gloss starting with a number word; the lexicon is being extended by
                              other agents, never copy it) plus the concord-built forms of -mwe, -bili, -tatu, -ne,
                              -sanwe (o-stems after a concord: obile, otatwe, one, osanwe; zyobilo, yotatwe, bone ...).
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
LEXICON = os.path.join(REPO, "tools", "chitonga", "grammar2", "lexicon.tsv")
MAX_WORDS = int(os.environ.get("HARNESS_SPLIT_WORDS", "25"))

# ------------------------------------------------------------------------------------------------------- F2 split
SENT = re.compile(r"(?<=[.!?;:])\s+")
CLAUSE = re.compile(r"(?<=,)\s+")


def nwords(s):
    return len(s.split())


def split(text, max_words=MAX_WORDS):
    """[(piece, joiner)]: sentences first; a sentence still above max_words is cut at commas into runs that stay
    under the limit. A piece that cannot be cut (no punctuation) stays whole and is marked by its length; we never cut
    inside a clause (a cut between a verb and its object would change the meaning more than the collapse does)."""
    text = " ".join((text or "").split())
    if nwords(text) <= max_words:
        return [text]
    out = []
    for s in SENT.split(text):
        if nwords(s) <= max_words:
            out.append(s)
            continue
        cur = ""
        for c in CLAUSE.split(s):
            if cur and nwords(cur) + nwords(c) > max_words:
                out.append(cur)
                cur = c
            else:
                cur = (cur + " " + c).strip()
        if cur:
            out.append(cur)
    return [p for p in out if p.strip()]


def join(pieces):
    return " ".join(p.strip() for p in pieces if p and p.strip())


# ------------------------------------------------------------------------------------------------------- names
NOT_NAMES = {"I", "God", "Lord", "Jesus", "Christ", "Bible", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December", "English", "Chitonga", "Tonga", "Bemba", "Nyanja", "Lozi",
             "Ila", "Mr", "Mrs", "Ms", "Dr", "Fr", "Sr", "Unit", "Grade", "Lesson", "Chapter", "Term", "OK", "AIDS",
             "HIV", "Christmas", "Easter", "Doctor", "Teacher", "Mother", "Father", "Sister", "Brother", "Chief",
             "Headmaster", "Person", "Cholera", "Mass", "Satan", "Heaven"}
EN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


def en_names(en):
    """names in an English sentence: capitals not at a sentence start (after . ! ? : ; a quote or a dash a capital
    proves nothing), all-caps words of 2+ letters excluded unless they are mixed-case words (headings are all caps)"""
    out = []
    prev_end, prev = 0, None
    for m in EN_WORD.finditer(en or ""):
        w = m.group(0).rstrip("'’-")
        gap = (en or "")[prev_end:m.start()]
        initial = prev is None or any(ch in ".!?:;\"“”‘(—–" for ch in gap) or re.fullmatch(r"\d+[.)]?", prev or "")
        prev_end, prev = m.end(), w
        if not w[:1].isupper() or w in NOT_NAMES or (w.isupper() and len(w) > 1):
            continue
        if re.sub(r"['’]s$", "", w) in NOT_NAMES:
            continue
        # an ordinary English word written with a capital is not a name to copy, wherever it stands: 'He', 'That'
        # after an exercise letter, the story characters Hare and Buffalo (the Tonga says sulwe, munyati: a
        # character's name is translated, F046 is about personal names); unless the gazetteer knows it as a place
        # (E8, 18 Oct: the gazetteer has English words as village names, 'Little'; so an ordinary word is never a
        # name, gazetteer or not)
        if len(w) < 3 or (w.lower() in _en_vocab() and not _gaz_place(w, strong=True)):
            continue
        # 22 Oct 2026 (weakness map item 3, Chris: "fix it"): the memory vocabulary (~7,900 words) lacks most civic and
        # technical nouns, so headings went through the names route in English ('Rainfall Forecast', 'Policy
        # Framework', 'Right to Identity': served 43 untranslated vs raw 18 on the 500-line diagnosis). A word in the
        # English dictionary in lower case is ordinary too, unless the dictionary also lists it as a proper name
        # (Grace, Hope, Frank) or it is a Zambian given name that is an English word (Gift, Precious, Memory)
        # ('Northwestern Province' stays a name: a compass word before Province / District is a place; so is a
        # province name in a list, 'much of Northwestern and Eastern provinces', d500:119)
        if _en_dict_ordinary(w) and not _gaz_place(w, strong=True) and w not in PROVINCE_WORDS \
                and not re.match(r"\s+(?:Province|District)", (en or "")[m.end():]):
            continue
        # a sentence-initial word not seen in lower case is a name only with evidence: a gazetteer place, a name seen
        # mid-sentence in our English, or a Bantu shape (ends in a vowel: Mwiinga, Choobe); 'Entertainment' is none
        if initial and not (_gaz_place(w) or w in _en_caps() or w[-1:].lower() in "aeiou"):
            continue
        out.append(re.sub(r"['’]s$", "", w))
    return list(dict.fromkeys(out))


_vocab = set()


def _en_vocab():
    """English words seen in LOWER case in the English of our pairs (the translation memory's rows): a sentence-
    initial capital that is an ordinary English word is not a name ('Yesterday', 'Wash'); one never seen in lower
    case is ('Mwiinga went to Choma')"""
    if not _vocab:
        try:
            import h2_memory as M
            for rows in M.memory()["by"].values():
                for r in rows:
                    _vocab.update(re.findall(r"(?<![A-Za-z])[a-z][a-z'’\-]*", r["en"]))
        except Exception:  # noqa: BLE001
            _vocab.add("")
    return _vocab


EN_DICT_DIRS = [os.environ.get("EN_DICT_DIR", ""),
                r"C:\Program Files\ONLYOFFICE\DesktopEditors\dictionaries"]   # hunspell en_GB + en_US (LGPL wordlists)
EN_DICT_CACHE = os.path.join(REPO, "internal", "chitonga_train", "vocab", "en_dictionary_words.json")
# Zambian given names that are ordinary English words and that the hunspell lists do not carry as names: kept as names
# (a mid-sentence 'Gift' / 'Precious' is a person far more often than the noun). Chris's and the teachers' corrections
# extend this list; it only stops the dictionary rule, the memory-vocabulary rule above is unchanged.
EN_WORD_GIVEN_NAMES = {"Gift", "Precious", "Mercy", "Memory", "Innocent", "Patience", "Joy", "Given", "Chance",
                       "Clever", "Blessings", "Blessing", "Lovemore", "Golden", "Prudence", "Princess", "Loveness",
                       "Lucky", "Talent", "Beauty", "Knowledge", "Witness", "Rejoice", "Gracious", "Brilliant", "Trust",
                       "Goodwill", "Wisdom", "Pleasure", "Royal", "Promise", "Peace", "Comfort", "Favour", "Miracle",
                       "Destiny", "Honest", "Obvious", "Mighty", "Choice", "Learnmore", "Fortunate", "Moment"}
_dict = {}
# Zambia's provinces whose names are English words (the gazetteer spells North-Western; the ZMD bulletins write
# Northwestern): capitalised, they are places, never translated as compass directions
PROVINCE_WORDS = {"Northern", "Southern", "Eastern", "Western", "Central", "Northwestern", "North-Western",
                  "Copperbelt"}


def _en_dict():
    """{'lower': lower-case dictionary stems, 'proper': capitalised dictionary entries}; from the cache file, else built
    from the hunspell .dic files (affix flags dropped) and cached; empty when neither exists (a pod without the
    dictionary: en_names then behaves exactly as before 22 Oct)"""
    if _dict:
        return _dict
    import json
    lower, proper = set(), set()
    try:
        if os.path.exists(EN_DICT_CACHE):
            d = json.load(open(EN_DICT_CACHE, encoding="utf-8"))
            lower, proper = set(d["lower"]), set(d["proper"])
        else:
            for base in EN_DICT_DIRS:
                for loc in ("en_GB", "en_US"):
                    p = os.path.join(base, loc, loc + ".dic") if base else ""
                    if not p or not os.path.exists(p):
                        continue
                    for line in open(p, encoding="utf-8", errors="replace"):
                        w = line.strip().split("/")[0]
                        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w):
                            continue
                        (lower if w[0].islower() else proper).add(w if w[0].isupper() else w.lower())
            if lower:
                with open(EN_DICT_CACHE, "w", encoding="utf-8", newline="\n") as f:
                    json.dump({"source": "hunspell en_GB + en_US .dic (ONLYOFFICE dictionaries), affix flags dropped; "
                                         "built by h2_text._en_dict", "lower": sorted(lower), "proper": sorted(proper)}, f)
    except Exception:  # noqa: BLE001  (no dictionary: the old behaviour, the safe side for names)
        lower, proper = set(), set()
    _dict.update(lower=lower, proper=proper)
    return _dict


def _en_stems(w):
    """the word and its plain inflection stems (the .dic lists stems; the affix rules are not applied here)"""
    w = w.lower()
    out = [w]
    if w.endswith("ies") and len(w) > 4:
        out.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4:
        out.append(w[:-2])
    if w.endswith("s") and len(w) > 3:
        out.append(w[:-1])
    for suf in ("ed", "ing"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            out += [w[:-len(suf)], w[:-len(suf)] + "e"]
    if w.endswith("ly") and len(w) > 4:
        out.append(w[:-2])
    return out


def _en_dict_ordinary(w):
    """a capitalised word that is an ordinary English dictionary word: a lower-case entry (or its stem) and NOT also a
    proper-name entry (Grace, Hope, Mark, Frank, Will) and not a Zambian given name that is an English word"""
    d = _en_dict()
    if not d.get("lower") or w in EN_WORD_GIVEN_NAMES or w in d["proper"]:
        return False
    return any(s in d["lower"] for s in _en_stems(w))


_caps = set()


def _en_caps():
    """capitalised words seen mid-sentence in the English of our pairs (names the books use)"""
    if not _caps:
        try:
            import h2_memory as M
            for rows in M.memory()["by"].values():
                for r in rows:
                    _caps.update(re.findall(r"(?<=[a-z,] )[A-Z][a-z]+", r["en"]))
        except Exception:  # noqa: BLE001
            _caps.add("")
    return _caps


def _gaz_place(w, strong=False):
    """an exact gazetteer place (strong=True: a strong entry only: district, province, town, major river; GeoNames
    streams and villages are weak and include English words, 'Little')"""
    try:
        import sys
        sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "vectors"))
        import names as N
        ents, how = N.gaz_lookup(w)
        return bool(ents) and how == "exact" and (not strong or any(e.get("strength") == "strong" for e in ents))
    except Exception:  # noqa: BLE001  (no gazetteer: initial words are never names, the safe side)
        return False


# English > Tonga stand-ins, ordered by what the stage 12b translator copies unchanged (17 Oct 2026, 5 frames each:
# Sam, Kim, Jordan, Grace 5/5; Robin 4 (Robeni); Peter 3 (Pita); Mary 2 (Maliya); John 1 (Joni); Alex 0 (Aleki)).
# Tonga-shaped names (Mwiinga, Haakulikambila, Nabukowa, Choobe, Sikoswe, Mutinta) were copied 30 of 30 WITHOUT
# protection, so safe_en tries the plain translation first and protects only when a name is lost.
STAND_INS = ["Sam", "Kim", "Jordan", "Grace", "Robin", "Jamie", "Casey", "Morgan", "Riley", "Quinn", "Avery", "Taylor"]


def protect_en(en):
    """(text with names replaced by stand-ins, mapping {stand-in: name}); a stand-in already in the text is skipped"""
    mapping = {}
    pool = [s for s in STAND_INS if s not in (en or "")]
    text = en
    for nm in en_names(en):
        if not pool:
            break
        ph = pool.pop(0)
        mapping[ph] = nm
        text = re.sub(r"(?<![A-Za-z])" + re.escape(nm) + r"(?![a-z])", ph, text)
    return text, mapping


def safe_en(en, translate):
    """F3, English > Tonga with the names carried through: plain first; if a name is lost, stand-ins and restore;
    the version that loses fewest names wins. Returns (tonga, report)"""
    nm = en_names(en)
    plain = translate(en)
    miss = [n for n in nm if n.lower() not in plain.lower()]
    if not miss:
        return plain, {"method": "plain", "names": nm, "missing": []}
    t, mp = protect_en(en)
    prot = restore(translate(t), mp)
    miss2 = [n for n in nm if n.lower() not in prot.lower()]
    if len(miss2) < len(miss):
        return prot, {"method": "stand_in", "names": nm, "missing": miss2}
    return plain, {"method": "plain", "names": nm, "missing": miss}


# the Tonga spellings the translator gives some stand-ins (17 Oct: Robin > Robeni; 22 Oct, d500:119: Jordan > Joredani,
# read as the country / river): restored like the stand-in itself. Safe: a stand-in is only used when the input does not
# contain it, so these forms in the output can only be our placeholder
STAND_IN_FORMS = {"Jordan": ["Joredani", "Jorodani", "Jordani"], "Robin": ["Robeni", "Robini"], "Jamie": ["Jaime"]}


def restore(text, mapping):
    """put names back; tolerant of a Tonga prefix glued to the stand-in (kuAlex, ba Alex) and of 'Alexs'"""
    for ph, nm in list(mapping.items()):
        for form in STAND_IN_FORMS.get(ph, []):
            text = re.sub(r"(?<![A-Za-z])" + form + r"(?![a-z])", ph, text)
    for ph, nm in sorted(mapping.items(), key=lambda x: -len(x[0])):
        text = re.sub(re.escape(ph) + r"(?![a-z])", nm, text)
        text = re.sub(re.escape(ph.lower()) + r"(?![a-z])", nm, text)
    return text


# ------------------------------------------------------------------------------------------------------- numbers
EN_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
          "thousand": 1000, "million": 10 ** 6, "one": 1, "once": 1, "twice": 2, "first": 1, "second": 2, "third": 3,
          "fourth": 4, "fifth": 5, "half": 0.5, "double": 2, "both": 2}
# ordinal / softer English words: present in the English, the Tonga may say them with a construction we cannot read
SOFT_EN = {"first", "second", "third", "fourth", "fifth", "once", "twice", "half", "double", "both", "one"}
# Tonga numeral stems after a concord (Hopgood index p298-p317; Thompson pp7-36; teachers 14-15 Oct): the o-stem forms
# (otatwe, obile, osanwe, omwe, one 'four') with an optional class prefix before the o: zy-, by-, y-, ly-, b-, c-, t-,
# k-, l-, w-, m-, p-, ku-
CONC = r"(?:zy|zi|by|bi|y|ly|li|b|bo|c|ci|t|tu|k|ka|l|lu|w|m|mu|p|ku|ma)?"
TOI_STEM = [(re.compile(r"^" + CONC + r"o(?:bile|bilo|bili)$"), 2), (re.compile(r"^" + CONC + r"(?:otatwe|otatu)$"), 3),
            (re.compile(r"^" + CONC + r"o?sanwe$"), 5), (re.compile(r"^(?:zy|by|y|ly|b|c|t|k|l|w|m|p)one$"), 4),
            (re.compile(r"^" + CONC + r"omwe$"), 1), (re.compile(r"^(?:ma|li|i)?k?kumi$"), 10),
            (re.compile(r"^(?:mw|my)aanda$"), 100), (re.compile(r"^(?:c|zy)?[iu]?uulu$"), 1000),
            (re.compile(r"^(?:to|tu)(?:tatwe|sanwe|ne|bile)$"), None)]   # 'three times', counted as present
WORD_NUM = {"one": 4, "bobilo": 2, "babili": 2, "zibili": 2, "bili": 2, "tatu": 3, "musanu": 5, "sanu": 5, "lusele": 8,
            "ciloba": 7, "cisambomwe": 6, "kakole": 6, "fwuka": 9, "kkumi": 10, "ikkumi": 10, "makumi": 10}
_lex_nums = None


def _lexicon_numerals():
    """lexicon rows with pos num / enum whose gloss starts with a number word, read from lexicon.tsv each process
    (never cached in a file: the lexicon changes)"""
    global _lex_nums
    if _lex_nums is None:
        _lex_nums = {}
        try:
            for r in csv.DictReader(open(LEXICON, encoding="utf-8"), delimiter="\t"):
                if (r.get("pos") or "").strip() not in ("num", "enum"):
                    continue
                m = re.match(r"\s*(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\b",
                             (r.get("meaning") or "").lower())
                if m:
                    _lex_nums[(r.get("word") or "").strip().lower()] = EN_NUM.get(m.group(1), None) or \
                        {"hundred": 100, "thousand": 1000}.get(m.group(1))
        except OSError:
            pass
    return _lex_nums


def numbers_en(text):
    """{value: 'hard' | 'soft'}: digits and number words; 'one' / ordinals / 'both' are soft (the Tonga may carry
    them in a construction the reader cannot count: bumwi buzuba 'one day', boonse bobilo)"""
    out = {}
    for d in re.findall(r"\d+(?:[.,]\d+)?", text or ""):
        v = float(d.replace(",", "")) if "." in d or "," in d else int(d)
        out[v] = "hard"
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in EN_NUM:
            v = EN_NUM[w]
            out.setdefault(v, "soft" if w in SOFT_EN else "hard")
    return out


def numbers_toi(text):
    """{value}: digits and Tonga numerals (value None = a 'times' form, present but not counted)"""
    vals = set()
    for d in re.findall(r"\d+(?:[.,]\d+)?", text or ""):
        vals.add(float(d.replace(",", "")) if "." in d or "," in d else int(d))
    seq = []
    # days are numbered words (Bwabili Tuesday ... Bwasanu Friday): a name of a day, not a count (E8, 18 Oct)
    text = re.sub(r"\b[Bb]wa(bili|tatu|ne|sanu)\b", " ", text or "")
    for w in re.findall(r"[a-zŋ']+", (text or "").lower()):
        v = toi_numeral(w)
        seq.append((w, v))
        if v:
            vals.add(v)
    # tens: makumi obile 'twenty', makumi otatwe aa(o)sanwe 'thirty-five' (Hopgood p303 likumi / makumi)
    for i, (w, v) in enumerate(seq):
        if w == "makumi" and i + 1 < len(seq) and seq[i + 1][1] and seq[i + 1][1] < 10:
            t = 10 * seq[i + 1][1]
            vals.add(t)
            if i + 3 < len(seq) and seq[i + 2][0] in ("a", "aa", "ane", "alimwi") and seq[i + 3][1] and seq[i + 3][1] < 10:
                vals.add(t + seq[i + 3][1])
            elif i + 2 < len(seq) and seq[i + 2][0].startswith("a") and toi_numeral(seq[i + 2][0][1:].lstrip("a")):
                vals.add(t + toi_numeral(seq[i + 2][0][1:].lstrip("a")))
    return vals


# ordinals: genitive concord + stem (Hopgood p298 'bili ... joined to gen.conc. second'; tatu, ne, sanu likewise):
# cibeela cabili 'section two', giledi yasanu 'grade five', cibeela cane 'section four' (E8, 18 Oct)
ORDINAL = re.compile(r"^(?:ca|wa|ya|za|zya|la|lya|ka|lwa|bwa|kwa|ba|ma|twa|mwa|pa)(bili|tatu|ne|sanu|cisanu|sanwe)$")
ORD_VAL = {"bili": 2, "tatu": 3, "ne": 4, "sanu": 5, "cisanu": 5, "sanwe": 5}


def toi_numeral(w):
    m = ORDINAL.match(w)
    if m:
        return ORD_VAL[m.group(1)]
    if re.match(r"^(ku|mu|ka)(zy|c)?[iu]?uulu$", w):     # kuzyuulu 'to thousands'
        return 1000
    lex = _lexicon_numerals()
    if w in lex and lex[w]:
        return lex[w]
    if w in WORD_NUM:
        return WORD_NUM[w]
    for rx, v in TOI_STEM:
        if rx.match(w):
            return v
    return None


def numbers_en_composed(text):
    """numbers_en plus 'twenty-five' / 'twenty five' = 25"""
    out = numbers_en(text)
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
    for m in re.finditer(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s-]+(one|two|three|four|five|six|"
                         r"seven|eight|nine)\b", (text or "").lower()):
        out[tens[m.group(1)] + EN_NUM[m.group(2)]] = "hard"
        if m.group(2) != "one" and out.get(EN_NUM[m.group(2)]) == "hard":
            out.pop(EN_NUM[m.group(2)])   # the unit is inside the composed number, not a number of its own
    for k in ("sixty", "seventy", "eighty", "ninety"):
        if re.search(r"\b" + k + r"\b", (text or "").lower()):
            out.setdefault(tens[k], "hard")
    return out
