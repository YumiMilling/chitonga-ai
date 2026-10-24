"""The teacher search: an index built at export time and ONE matcher, the same in Python and in JavaScript (24 Oct 2026,
the Jesuit).

Chris, 24 Oct 2026: "the search sucks, I type in metabolism and I get nothing." The live demo searched by keyword in the
function (core.retrieval.Index's stems and IDF; a request had to share a word with an item's title), the phone's offline
search by substring over the lessons it had saved. A teacher typing messy English on a phone (PERSONA_TEACHER.md:
"pls", "gr 2", one bar of signal) got nothing for a misspelling, a synonym or a topic we hold under another name.

What this is:
  build(plugin, tips, concepts, fingerprint)  -> the index (a dict, written as search_index.json behind the key, never
                                                 under site/: titles and English terms only, no teacher Tonga)
  Searcher(index).search(query, grades)       -> ranked hits, the same answer site/assistant/search.js gives
                                                 (tests/search_parity.mjs checks every query of the test set)

The index, per servable item (an item the app can OPEN: a lesson, card or sample lesson in a grade's menu, a syllabus
competence, a teaching tip), holds four fields of stemmed terms, ranked in this order:
  T  the title (and a sample's short label, a competence's sub-topic)                       weight 10
  K  key words: the lesson's English key words, the unit label, the tip's topic             weight 6
  C  concept terms: what a teacher might type for it (search_concepts.json, written by     weight 5
     Claude from the item's own content, conservative: related, never loosely associated)
  B  the body's 24 most distinctive terms (tf-idf over the items)                           weight 2
plus adjacent-word pairs of T, K and C ('times tables' is a pair, so it is not 'telling the time').
Two shared tables (search_lexicon.json): SYN, equivalents a teacher uses for the same thing (pupils = learners,
borrowing = regrouping, times tables = multiplication), matched almost as exact; REL, broader related concepts for
words we hold nothing under (metabolism -> digestion, food groups, human body), matched weakly and SHOWN as related
("Nothing called 'metabolism' yet. Related: ..."). Concept terms written per syllabus sub-topic are kept on a
competence only where its own text touches them. The index also carries the normalisation tables and the
safeguarding regexes (web/api.py adds them), so the phone needs nothing else.

Matching: typo-tolerant (edit distance 1, or 2 for stems of 8+ letters, only the nearest words, candidates found by
shared trigrams; a looser match never weighs more than the exact word would), prefixes,
the SMS and Zambian spellings (pls, hw, wat, std 3, gr2), British and American spellings; a grade in the request
("gr2", "std 4", "grade two") is parsed out and becomes the grade preferred. Ranking: title > key word > concept term >
body, scaled by how much of the request the item covers; among the good matches (at least 40% of the best score) the
class's grade comes first, other grades after and marked; below them the class grade weighs 1.25 times. "strong" is
False when the best hit matched no title, key word or concept term: then nothing is called that, and the hits are
related topics.

Measured (tests/search_eval.py, 60 teacher queries written before the build, 20 held out): precision@3 0.336 -> 0.900
(held-out 0.408 -> 0.900), success@3 0.617 -> 1.000; search.js at 4x CPU throttle: mean 0.7 ms, max 6 ms a query.

Rules kept: deterministic (no model at query time), the same arithmetic in both languages (scores rounded to 4 decimals,
ties broken by the item's place in the index), nothing here decides what is SERVED: the app opens the item as before,
with its labels (needs review, sample, health gate).
"""
import json
import math
import os
import re
import unicodedata

VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------------ normalisation
# MIRRORED in site/assistant/search.js (the index carries these tables, so the phone uses exactly these)
STOP = set("""a an the of to in on at for from by with and or but if then so as is are was were be been being am do does
did doing have has had having it its this that these those there here what whats why how which where when who whom whose
me us you your we our they their i my he she his her them can could would should will shall may might must please some
any much more most very about into out up over than too also just only not no yes make makes made way ways
need needs want wants give show help find get got idea ideas thing things lot lots really good best
lesson lessons teach teaching taught""".split())
# SMS and Zambian English spellings -> the plain word (applied to the raw token, before the stop list)
SMS = {"pls": "please", "plz": "please", "plse": "please", "hw": "how", "hu": "how", "wat": "what", "wot": "what",
       "wht": "what", "u": "you", "ur": "your", "r": "are", "n": "and", "abt": "about", "bout": "about", "wid": "with",
       "wit": "with", "thru": "through", "b4": "before", "2day": "today", "tdy": "today", "tmrw": "tomorrow",
       "tmr": "tomorrow", "2moro": "tomorrow", "tomoro": "tomorrow", "lrnrs": "learners", "lnrs": "learners",
       "kidz": "kids", "nos": "numbers", "eng": "english", "sci": "science", "maths": "math", "mathematics": "math",
       "bcoz": "because", "coz": "because", "cuz": "because", "dnt": "dont", "cnt": "cant", "cant": "cannot",
       "gonna": "going", "wanna": "want", "teachin": "teaching", "readin": "reading", "writin": "writing",
       "spellin": "spelling", "countin": "counting", "ok": "okay", "thx": "thanks", "tnx": "thanks"}
# irregular plurals and forms the suffix rules miss
IRREG = {"children": "child", "kids": "child", "kid": "child", "pupils": "learner", "pupil": "learner",
         "learners": "learner", "men": "man", "women": "woman", "feet": "foot", "teeth": "tooth", "mice": "mouse",
         "geese": "goose", "leaves": "leaf", "halves": "half", "knives": "knife", "wolves": "wolf", "shelves": "shelf",
         "lives": "life", "loaves": "loaf", "people": "person", "oxen": "ox", "calves": "calf", "thieves": "thief",
         "bigger": "big", "biggest": "big", "smaller": "small", "smallest": "small", "longer": "long",
         "longest": "long", "shorter": "short", "shortest": "short", "taller": "tall", "tallest": "tall",
         "heavier": "heavy", "heaviest": "heavy", "lighter": "light", "lightest": "light", "more": "more",
         "fewer": "few", "fewest": "few", "greater": "great", "greatest": "great", "ate": "eat", "eaten": "eat",
         "grew": "grow", "grown": "grow", "drank": "drink", "drunk": "drink", "wrote": "write", "written": "write",
         "read": "read", "spoke": "speak", "spoken": "speak", "told": "tell", "taught": "teach", "bought": "buy",
         "sold": "sell", "thought": "think", "brought": "bring", "began": "begin", "begun": "begin", "sang": "sing",
         "sung": "sing", "swam": "swim", "ran": "run", "went": "go", "gone": "go", "data": "data", "news": "news",
         "series": "series", "species": "species", "bus": "bus", "gas": "gas", "glasses": "glass", "maths": "math"}
# British -> one spelling (both sides of the match go through this)
SPELL = {"colour": "color", "colours": "colors", "coloured": "colored", "colouring": "coloring", "favourite": "favorite",
         "neighbour": "neighbor", "neighbours": "neighbors", "behaviour": "behavior", "labour": "labor",
         "metre": "meter", "metres": "meters", "centimetre": "centimeter", "centimetres": "centimeters",
         "millimetre": "millimeter", "millimetres": "millimeters", "kilometre": "kilometer", "kilometres": "kilometers",
         "litre": "liter", "litres": "liters", "millilitre": "milliliter", "millilitres": "milliliters",
         "centre": "center", "centres": "centers", "organise": "organize", "organising": "organizing",
         "recognise": "recognize", "recognising": "recognizing", "practise": "practice", "practising": "practicing",
         "programme": "program", "analyse": "analyze", "grey": "gray", "honour": "honor", "flavour": "flavor",
         "harbour": "harbor", "rumour": "rumor", "humour": "humor", "odour": "odor", "storey": "story",
         "tyre": "tire", "tyres": "tires", "aeroplane": "airplane", "aeroplanes": "airplanes", "mum": "mother",
         "mom": "mother", "mummy": "mother", "dad": "father", "daddy": "father", "jewellery": "jewelry",
         "plough": "plow", "ploughing": "plowing", "defence": "defense", "offence": "offense", "licence": "license",
         "catalogue": "catalog", "dialogue": "dialog", "cheque": "check", "moustache": "mustache",
         "fibre": "fiber", "theatre": "theater", "manoeuvre": "maneuver", "paediatric": "pediatric",
         "anaemia": "anemia", "diarrhoea": "diarrhea", "faeces": "feces", "foetus": "fetus", "oesophagus": "esophagus"}
VOWELS = set("aeiouy")
COMB_LO, COMB_HI, RSQUO = chr(0x300), chr(0x36F), chr(0x2019)      # combining accents; the typographic apostrophe
TOKEN_SPLIT = re.compile("[^a-z0-9" + chr(0x14B) + "]+")          # letters, digits and Tonga's velar nasal


def fold(text):
    """accents off (papier mache, cafe), as JavaScript's normalize('NFKD') + dropping U+0300-U+036F does"""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text or "")) if not COMB_LO <= c <= COMB_HI)


def raw_tokens(text):
    """lower case, accents off, apostrophes dropped (don't -> dont), split on anything that is not a letter or digit
    (and Tonga's velar nasal)"""
    t = fold(text).lower().replace(RSQUO, "").replace("'", "")
    return [w for w in TOKEN_SPLIT.split(t) if w]


def norm(w):
    w = SMS.get(w, w)
    w = IRREG.get(w, w)
    return SPELL.get(w, w)


def _undouble(w):
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "lsz" and w[-1] not in VOWELS:
        return w[:-1]
    return w


def stem(w):
    """a light English stemmer (plural, -ing, -ed, final e). The same rules run on the phone (search.js); keep them
    identical. Words of 3 letters or less, numbers and the irregular table's answers are left as they are."""
    if len(w) <= 3 or w.isdigit() or w in KEEP:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("es") and len(w) > 4 and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
        w = w[:-2]
    elif w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("eed"):
        pass
    elif w.endswith("ing") and len(w) >= 6 and any(c in VOWELS for c in w[:-3]):
        w = _undouble(w[:-3])
    elif w.endswith("ed") and len(w) >= 5 and any(c in VOWELS for c in w[:-2]):
        w = _undouble(w[:-2])
    if w.endswith("e") and len(w) >= 4:
        w = w[:-1]
    return w


KEEP = {"news", "series", "species", "bus", "gas", "data", "glass", "read", "more", "times"}


def terms(text, keep_stop=False):
    """the stemmed content words of a text, in order (stop words dropped unless keep_stop)"""
    return [s for s, _ in terms_raw(text, keep_stop)]


def terms_raw(text, keep_stop=False):
    """[(stem, the normalised word before stemming)]: the typo matcher compares both ('cyle' stems to 'cyl')"""
    out = []
    for w in raw_tokens(text):
        w = norm(w)
        if not keep_stop and (w in STOP or len(w) < 2 and not w.isdigit()):
            continue
        s = stem(w)
        if not keep_stop and s in STOP_STEM:
            continue
        out.append((s, w))
    return out


STOP_STEM = {stem(w) for w in STOP if len(w) > 3}


def pairs(ts):
    return [ts[i] + "_" + ts[i + 1] for i in range(len(ts) - 1)]


# ------------------------------------------------------------------------------------------------ grades in a request
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "1st": 1, "2nd": 2, "3rd": 3,
           "4th": 4, "5th": 5, "6th": 6, "7th": 7}
GRADE_RX = re.compile(r"(?<![a-z0-9])(?:grades?|grd|gr|gd|g|std|standard|stds|class|form|giledi|year|yr)\s*\.?\s*"
                      r"([1-7]|one|two|three|four|five|six|seven)(?![a-z0-9])|(?<![a-z0-9])([1-7](?:st|nd|rd|th))\s+"
                      r"(?:grade|std|standard|class)(?![a-z0-9])")


def parse_grade(text):
    """(grade or None, the text without the grade words). 'gr2', 'gr 2', 'grade two', 'std 4', 'class 3', '2nd grade'.
    Only grades 1-7; 'class' needs a number right after it ('noisy class' stays a request about a class)."""
    low = str(text or "").lower()
    m = GRADE_RX.search(low)
    if not m:
        return None, low
    g = m.group(1) or m.group(2)
    g = int(g) if g.isdigit() else WORDNUM.get(g)
    return g, (low[:m.start()] + " " + low[m.end():]).strip()


# ------------------------------------------------------------------------------------------------ the Tonga-word question
# "leaf in Tonga", "how do you say cow", "Tonga word for heart", "what is leaf in chitonga", "cow in tonga pls": the
# request is for a WORD; the app answers with a word card (the checked Tonga and its badge, or "Tonga word needed" and
# "Add it") and searches the English word alone. Mirrored in search.js (WORD_* there); keep the two identical.
_W = r"[\"']?([a-z][a-z \-]{0,30}?)[\"']?"
_TONGA = r"(?:chi|ci)?tonga"
_TAIL = r"(?:\s+(?:in|into)\s+" + _TONGA + r")?(?:\s+(?:pls|please|plz))?"
WORD_ASK = re.compile(r"^(?:(?:pls|please|plz)\s+)?(?:how\s+(?:(?:do|would|can|should)\s+(?:you|i|we|one)|to)\s+"
                      r"(?:say|call|write)|(?:the\s+)?(?:" + _TONGA + r"\s+)?word\s+for|" + _TONGA + r"\s+for|translate|"
                      r"whats?(?:\s+is)?(?:\s+the)?(?:\s+" + _TONGA + r")?(?:\s+word)?(?:\s+for)?)\s+(?:the\s+word\s+)?" + _W +
                      _TAIL + r"\s*$")
WORD_IN = re.compile(r"^(?:(?:pls|please|plz)\s+)?" + _W + r"\s+(?:in|into)\s+" + _TONGA + r"(?:\s+(?:pls|please|plz))?\s*$")


def parse_word(text, with_form=False):
    """the English word of a Tonga-word question, or None. 'what is X' counts only with Tonga or 'word' in it (else
    it is a question about X); 'how do you say X' counts without (Tonga is the app's other language). with_form: also
    say whether it was an explicit ask ('how do you say X', 'word for X': search the word alone) or 'X in Tonga'
    (search the whole request too: 'teaching in Tonga' wants the many-languages tips as well as a word card)."""
    t = re.sub(r"[?!.]+\s*$", "", re.sub(r"\s+", " ", str(text or "").lower().replace(RSQUO, "").replace("'", ""))).strip()
    m = WORD_ASK.match(t)
    if m and t.startswith(("what", "pls what", "please what", "plz what")) and not re.search(r"tonga|word", t):
        m = None
    explicit = bool(m)
    m = m or WORD_IN.match(t)
    if not m:
        return (None, False) if with_form else None
    w = re.sub(r"^(?:a|an|the)\s+", "", m.group(1).strip())
    if not w or len(w.split()) > 3 or w in ("it", "this", "that", "word", "tonga", "chitonga"):
        return (None, False) if with_form else None
    return (w, explicit) if with_form else w


# ------------------------------------------------------------------------------------------------ the documents
FIELD_W = {"T": 10.0, "K": 6.0, "C": 5.0, "B": 2.0}
FIELDS = ("T", "K", "C", "B")
TYPE_W = {"lesson": 1.0, "card": 1.0, "sample": 1.0, "tip": 1.0, "syllabus": 0.8}
BODY_N = 24
TIP_TOPICS = {"curriculum": "New curriculum", "groups": "Group work", "noise": "Noisy class",
              "large-class": "Large classes", "questioning": "Questions", "checking": "Quick checks",
              "routines": "Routines", "materials": "Local materials", "multilingual": "Many languages"}
SUBJECT_LABEL = {"english": "English", "maths": "Maths", "science": "Science", "maths_science": "Maths and science",
                 "chitonga": "Chitonga", "cts": "Creative and technology studies", "health": "Health",
                 "social": "Social studies", "arts": "Expressive arts", "technology": "Technology studies",
                 "heh": "Home economics"}
# the fixed skill headings of the CDC Chitonga Grade 1 module (the competences are the module's Tonga, auto-extracted):
# an English gloss of the HEADING only, from words our sources attest (kubala read/count, kulemba write, tubala letters,
# mabala words, tuvwuumina sounds: the lesson key words, CDC 2013 guide; kuswiilila listen, kwaambaula speak). The
# competence's own Tonga is never translated here.
TOI_HEADINGS = [("kuswiilila", "listening speaking"), ("kwaambaula", "speaking talking"), ("kumvwisisya", "comprehension understanding"),
                ("kulemba", "writing"), ("bulembo", "handwriting writing"), ("kubala", "reading"),
                ("tuvwuumina", "letter sounds phonics"), ("tubala", "letters"), ("mabala", "words vocabulary"),
                ("cakupozya", "fluency fluent reading"), ("tubeela", "syllables")]


def _strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(s or ""))).strip()


def _kw_en(k):
    return re.sub(r"\(.*?\)", " ", k.get("en") or "")


def documents(P, tips, menu_grades=None):
    """[{id, type, title, grades (the grades whose app bundle opens it), subject, sub, fields: {T,K,B text}}] for every
    servable item, in a fixed order (lessons, cards, samples by menu; competences; tips). menu_grades: grade -> the
    plug-in's grade_menu (default: P.grade_menu for each grade)."""
    menus = menu_grades or {g: P.grade_menu(g) for g in P.grades()}
    where = {}          # item id -> (grades it opens in, unit label, subject)
    for g, gm in sorted(menus.items()):
        if not gm:
            continue
        for s in gm["subjects"]:
            for u in s["units"]:
                for i in u["items"]:
                    w = where.setdefault(i, {"grades": [], "units": [], "subject": s["id"]})
                    if g not in w["grades"]:
                        w["grades"].append(g)
                    if u["label"] not in w["units"]:
                        w["units"].append(u["label"])
    docs = []
    for iid in sorted(where, key=lambda i: (where[i]["grades"][0], i)):
        it = P.items.get(iid)
        if not it or it["kind"] not in ("lesson", "card", "sample") or it.get("role") == "set_intro":
            continue
        w = where[iid]
        title = it.get("title") or ""
        T, K, B = [title], list(w["units"]), []
        if it["kind"] == "lesson":
            K += [_kw_en(k) for k in it.get("key_words") or []]
            B += [it.get("scope") or ""] + [ln.get("en") or "" for ln in it.get("lines") or [] if ln.get("served")]
            st = P.strategies.get(iid) or {}
            B += [st.get("scope") or ""] + [m.get("title", "") + " " + m.get("text", "") for m in st.get("mistakes") or []]
            B += [m.get("title", "") for m in st.get("struggles") or []]
        elif it["kind"] == "card":
            p = it.get("parts") or {}
            K += list(it.get("key_words_en") or []) + list(it.get("keep_english") or [])
            B += [p.get(k) or "" for k in ("answer", "idea", "local_example", "how_to_teach", "check", "mistakes")
                  if isinstance(p.get(k), str)]
            B += [q.get("q", "") for q in it.get("check_questions") or []]
            if isinstance(it.get("syllabus_ref"), dict):
                B.append(it["syllabus_ref"].get("title") or "")
        else:
            T.append(it.get("label_short") or "")
            K += [_kw_en(k) for k in it.get("key_terms") or []]
            B += [it.get("teaches") or "", it.get("what_it_is") or "", _strip_html((it.get("guide") or {}).get("html"))]
            for r in it.get("syllabus_ref") or []:
                if isinstance(r, dict):
                    K += [r.get("sub_topic") or ""]
                    B += [r.get("topic") or ""] + list(r.get("competences") or [])
        docs.append({"id": iid, "type": it["kind"], "title": title, "grades": w["grades"], "subject": it.get("subject")
                     or w["subject"], "sub": w["units"][0] if w["units"] else "", "review": bool(it.get("needs_review")),
                     "fields": {"T": " ".join(T), "K": " ; ".join(K), "B": " ".join(B)}})
    syl = sorted((it for it in P.items.values() if it["kind"] == "syllabus"),
                 key=lambda it: (it["grade"], it.get("area") or "", it.get("code") or "", it["id"]))
    for it in syl:
        st, tp = (it.get("sub_topic") or {}).get("title", ""), (it.get("topic") or {}).get("title", "")
        K = [st, tp, it.get("area_label") or ""]
        if it.get("lang") == "toi":
            low = (st + " " + tp).lower()
            K += [en for toi, en in TOI_HEADINGS if toi in low] + ["chitonga", "tonga"]
        B = list(it.get("learning_activities") or []) + [it.get("expected_standard") or ""]
        docs.append({"id": it["id"], "type": "syllabus", "title": it["title"], "grades": [it["grade"]],
                     "subject": it.get("subject"), "sub": st, "code": it.get("code"), "lang": it.get("lang"),
                     "fields": {"T": it["title"] + " ; " + st, "K": " ; ".join(K), "B": " ".join(B)}})
    for t in tips:
        K = [TIP_TOPICS.get(t.get("topic"), t.get("topic") or ""), t.get("when") or ""]
        B = list(t.get("steps") or []) + [t.get("why") or "", t.get("watch_out") or ""]
        docs.append({"id": "tip:" + t["id"], "type": "tip", "title": t["title"], "grades": [], "subject": None,
                     "sub": TIP_TOPICS.get(t.get("topic"), t.get("topic") or ""),
                     "fields": {"T": t["title"], "K": " ; ".join(K), "B": " ".join(B)}})
    return docs


def concept_text(doc):
    """what Claude reads to write an item's concept terms (web/search_concepts.py): title, keys, a body extract"""
    f = doc["fields"]
    return {"id": doc["id"], "type": doc["type"], "grades": doc["grades"], "title": doc["title"], "keys": f["K"][:300],
            "body": f["B"][:500]}


# ------------------------------------------------------------------------------------------------ build
def _load(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def lexicon(path):
    """search_lexicon.json -> {"syn": [[phrase, ...] groups], "rel": {phrase: [related phrases]}}"""
    return _load(path, {"syn": [], "rel": {}})


def build(P, tips, concepts, lex, fingerprint="", menu_grades=None):
    """the index (a dict; JSON-ready, compact). concepts: {item id: [phrases]}; lex: lexicon()"""
    docs = documents(P, tips, menu_grades)
    vocab = {}

    def tid(t):
        if t not in vocab:
            vocab[t] = len(vocab)
        return vocab[t]
    # field term sets (title, key and concept phrases also give their adjacent pairs)
    sibs = {}
    for d in docs:
        if d["type"] == "syllabus":
            k = (d["grades"][0], d.get("subject"), d.get("sub"))
            sibs[k] = sibs.get(k, 0) + 1
    per = []
    for d in docs:
        f = {}
        for fld in ("T", "K"):
            ts = set()
            for part in re.split(r"\s*;\s*", d["fields"][fld]):
                tt = terms(part)
                ts.update(tt)
                ts.update(pairs(tt))
            f[fld] = ts
        cs = set()
        phrases = concepts.get(d["id"]) or []
        if d["type"] == "syllabus" and sibs.get((d["grades"][0], d.get("subject"), d.get("sub")), 0) > 1:
            # the concept terms were written per sub-topic; a competence keeps only those its OWN text touches ('use
            # whole numbers in subtraction' keeps subtraction and sums' siblings, not multiplication)
            own = set(terms(d["title"] + " " + d["fields"]["B"]))
            phrases = [ph for ph in phrases if own & set(terms(ph))]
        for ph in phrases:
            tt = terms(ph)
            cs.update(tt)
            cs.update(pairs(tt))
        f["C"] = cs - f["T"] - f["K"]
        f["K"] = f["K"] - f["T"]
        f["Bfull"] = terms(d["fields"]["B"])
        per.append(f)
    # body: the most distinctive single terms (tf-idf over the documents), not already in T/K/C
    df_b = {}
    for f in per:
        for t in set(f["Bfull"]):
            df_b[t] = df_b.get(t, 0) + 1
    n = len(docs)
    for f in per:
        tf = {}
        for t in f["Bfull"]:
            tf[t] = tf.get(t, 0) + 1
        have = f["T"] | f["K"] | f["C"]
        sc = sorted(((-(1 + math.log(c)) * math.log((n + 1) / (1 + df_b[t])), t) for t, c in tf.items()
                     if t not in have and len(t) > 2 and not t.isdigit()))
        f["B"] = {t for _, t in sc[:BODY_N]}
        del f["Bfull"]
    # document frequency over all fields -> idf (2 decimals, as an integer x100: the phone reads the same number)
    df = {}
    for f in per:
        for t in f["T"] | f["K"] | f["C"] | f["B"]:
            df[t] = df.get(t, 0) + 1
    order = sorted(df)                     # vocabulary sorted: the phone finds prefixes by binary search
    for t in order:
        tid(t)
    # lexicon: phrases -> stemmed term lists (single terms and pairs), kept only where they reach the vocabulary
    syn = []
    for g in lex.get("syn") or []:
        grp = sorted({" ".join(terms(p)) for p in g if terms(p)})
        if len(grp) >= 2:
            syn.append(grp)
    rel = {}
    for k, vs in sorted((lex.get("rel") or {}).items()):
        kk = " ".join(terms(k))
        tgt = []
        for v in vs:
            tv = terms(v)
            key = "_".join(tv)
            if key in df and key not in tgt:
                tgt.append(key)
        if kk and tgt:
            rel[kk] = tgt
    out_docs, post = [], [[] for _ in order]
    for di, (d, f) in enumerate(zip(docs, per)):
        flags = 1 if d.get("review") else 0
        flags |= 2 if d.get("lang") == "toi" else 0
        out_docs.append([d["id"], d["type"], d["title"], d["grades"], d.get("subject") or "", d.get("sub") or "", flags])
        for fi, fld in enumerate(FIELDS):
            for t in f[fld]:
                post[vocab[t]].append(di * 4 + fi)
    # postings, term by term: doc * 4 + field, ascending, as differences (smaller numbers: 56 KB gzipped, not 100)
    enc = []
    for pl in post:
        prev, row = 0, []
        for x in sorted(pl):
            row.append(x - prev)
            prev = x
        enc.append(row)
    return {"v": VERSION, "fp": fingerprint, "n": n, "fields": list(FIELDS), "fw": [FIELD_W[x] for x in FIELDS],
            "terms": order, "idf": [int(round(100 * (math.log((n + 1) / (df[t] + 1)) + 1.0))) for t in order],
            "post": enc, "docs": out_docs, "syn": syn, "rel": rel, "norm": norm_tables(),
            "doc_cols": ["id", "type", "title", "grades", "subject", "sub", "flags (1 needs review, 2 Tonga text)"],
            "note": "teacher search index (core/search_index.py; site/assistant/search.js). English terms and titles "
                    "only (the Chitonga competences' CDC titles as printed); behind the teacher key. Rebuilt by "
                    "export_netlify.py with every bundle."}


def js_rx(p):
    """a compiled Python regex -> {"src", "flags"} for JavaScript (as export_netlify.rx): a leading (?i) becomes the i
    flag; a construct the two engines read differently is refused, never silently changed"""
    src, flags = p.pattern, "i" if p.flags & re.IGNORECASE else ""
    while True:
        m = re.match(r"\(\?([aiLmsux]+)\)", src)
        if not m:
            break
        if set(m.group(1)) - {"i"}:
            raise ValueError(f"inline flags {m.group(1)} in {src[:60]}")
        src = src[m.end():]
    for bad in ("(?P<", "(?P=", r"\Z", r"\A", "(?x", "(?s", "(?m", "(?#"):
        if bad in src:
            raise ValueError(f"{bad} in {src[:80]}")
    return {"src": src, "flags": flags}


def norm_tables():
    """the normalisation tables, carried IN the index so the phone uses exactly these (search.js reads them)"""
    return {"stop": sorted(STOP), "stop_stem": sorted(STOP_STEM), "sms": SMS, "irreg": IRREG, "spell": SPELL,
            "keep": sorted(KEEP), "wordnum": WORDNUM}


# ------------------------------------------------------------------------------------------------ matching
def osa(a, b, cap):
    """optimal string alignment distance (Damerau: a swap of two neighbours counts 1); returns cap+1 once it exceeds
    cap. Integer arithmetic: the same in JavaScript."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            if v < best:
                best = v
        if best > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[lb]


def trigrams(t):
    s = "^" + t + "$"
    return {s[i:i + 3] for i in range(len(s) - 2)}


Q = {"exact": 1.0, "syn": 0.9, "prefix": 0.8, "fuzzy1": 0.75, "fuzzy2": 0.55, "rel": 0.4}
GOOD = 0.4          # a hit within 40% of the best is a "good match" (the class grade goes first among those)
IN_GRADE = 1.25     # below the good matches, an item for the class's grade weighs this much more
STRONG_MIN = 4.0    # the best hit needs this score from title / key / concept matches to count as "found"


def r4(x):
    return math.floor(x * 10000 + 0.5) / 10000


class Searcher:
    """the matcher over a built index (dict). search() -> {hits, strong, grade, query, corrected}"""

    def __init__(self, index):
        ix = index
        self.ix = ix
        self.terms = ix["terms"]
        self.tix = {t: i for i, t in enumerate(self.terms)}
        self.idf = [x / 100.0 for x in ix["idf"]]
        self.docs = ix["docs"]
        self.fw = ix["fw"]
        # postings: term -> [(doc, field)], field 0..3 = T K C B (each doc lists a term in at most one field); the
        # title terms of each doc, for the whole-title test
        self.post = {}
        self.tterms = [set() for _ in self.docs]
        for t, row in enumerate(ix["post"]):
            x, pl = 0, []
            for dx in row:
                x += dx
                pl.append((x >> 2, x & 3))
                if x & 3 == 0 and "_" not in self.terms[t]:
                    self.tterms[x >> 2].add(self.terms[t])
            self.post[t] = pl
        self.syn = {}
        for grp in ix.get("syn") or []:
            for p in grp:
                self.syn.setdefault(p, [])
                for o in grp:
                    if o != p and o not in self.syn[p]:
                        self.syn[p].append(o)
        self.rel = ix.get("rel") or {}
        self._tri = None

    # -- candidate terms for one query term
    def tri(self):
        if self._tri is None:
            self._tri = {}
            for i, t in enumerate(self.terms):
                if "_" in t or len(t) < 3:
                    continue
                for g in trigrams(t):
                    self._tri.setdefault(g, []).append(i)
        return self._tri

    def _prefix_range(self, p):
        lo, hi = 0, len(self.terms)
        while lo < hi:
            m = (lo + hi) // 2
            if self.terms[m] < p:
                lo = m + 1
            else:
                hi = m
        out = []
        i = lo
        while i < len(self.terms) and self.terms[i].startswith(p):
            out.append(i)
            i += 1
            if len(out) > 40:
                return []        # too common a beginning to mean anything
        return out

    def candidates(self, t, raw=None):
        """[(term index, quality, how)] for a query term, best first; exact, prefix, the term's own beginning, typos.
        raw: the word before stemming (a typo is measured against the stem AND against the indexed word + 'e', the
        letter the stemmer drops: 'cyle' -> 'cycle')"""
        raw = raw or t
        out = {}

        def add(i, q, how):
            if i not in out or out[i][0] < q:
                out[i] = (q, how)
        i = self.tix.get(t)
        if i is not None:
            add(i, Q["exact"], "exact")
        if len(t) >= 4 and not t.isdigit():
            p = t[:-1] if t.endswith("y") and len(t) >= 6 else t
            for j in self._prefix_range(p):
                tt = self.terms[j]
                if "_" not in tt and j != i and 2 * len(p) >= len(tt):
                    # the shorter the typed part against the word, the weaker (read -> reader 0.53, multipl ->
                    # multiplication 0.4 is cut: under half the word)
                    add(j, r4(Q["prefix"] * len(p) / len(tt)), "prefix")
            # the query word is longer than an indexed word that begins it (fractional -> fraction)
            for k in range(len(t) - 1, max(4, int(len(t) * 0.7 + 0.999)) - 1, -1):
                j = self.tix.get(t[:k])
                if j is not None and j != i:
                    add(j, r4(Q["prefix"] * k / len(t)), "prefix")
                    break
        if i is None and len(raw) >= 4 and not t.isdigit():
            cap = 2 if len(t) >= 8 else 1
            seen = {}
            tri = self.tri()
            for g in sorted(trigrams(t) | trigrams(raw)):
                for j in tri.get(g, ()):
                    seen[j] = seen.get(j, 0) + 1
            fz = []
            for j in sorted(seen):
                tt = self.terms[j]
                if abs(len(tt) - len(t)) > cap and abs(len(tt) + 1 - len(raw)) > cap:
                    continue
                d = min(osa(t, tt, cap), osa(raw, tt, cap), osa(raw, tt + "e", cap))
                if d <= cap:
                    fz.append((d, j))
            # the nearest words only: a word one letter away beats every word two letters away ('borowing' is
            # 'borrow', not 'row' or 'grow')
            dmin = min([d for d, _ in fz] or [0])
            for d, j in fz:
                if d == dmin:
                    add(j, Q["fuzzy1"] if d == 1 else Q["fuzzy2"], "fuzzy")
        return sorted(((q, j, how) for j, (q, how) in out.items()), key=lambda x: (-x[0], x[1]))

    # -- the query
    def parse(self, query):
        grade, rest = parse_grade(query)
        tr = terms_raw(rest)
        ts = [x for x, _ in tr]
        self._raw = {}
        for x, w in tr:
            self._raw.setdefault(x, w)
        seen, uniq = set(), []
        for t in ts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return grade, ts, uniq

    def search(self, query, grades=None, k=10):
        """hits best first. grades: the class's grades (a preference: the class grade first among the good matches);
        a grade named in the request ("gr2", "std 4") replaces them."""
        # a Tonga-word question: the word card is the answer (res["word"]); the search runs on the English word alone
        g0, rest = parse_grade(query)
        word, explicit = parse_word(rest, with_form=True)
        if word and not explicit:
            # 'X in Tonga': the English word alone when it finds something called that ('heart in chitonga' -> the
            # body lessons), else the whole request ('teaching in Tonga' -> the many-languages tips)
            r1 = self.search(word, grades=[g0] if g0 else grades, k=k)
            if r1["strong"]:
                r1["grade"] = g0
                r1["word"] = {"en": word}
                return r1
        grade, seq, toks = self.parse(word if explicit else query)
        grade = grade or g0
        want = [grade] if grade else list(grades or [])
        res = {"query": " ".join(toks), "grade": grade, "hits": [], "strong": False, "corrected": []}
        if word:
            res["word"] = {"en": word}
        if not toks:
            return res
        # units: each query term with its candidates (and its SYN equivalents); a two-word SYN phrase of the request
        # ('cannot read', 'times tables') is one more unit that covers both its words
        units = []
        for a, b in zip(seq, seq[1:]):
            ph = a + " " + b
            if ph in self.syn:
                cands = []
                for o in self.syn[ph]:
                    cands += self._phrase_cands(o, Q["syn"])
                j = self.tix.get(a + "_" + b)
                if j is not None:
                    cands.append((Q["exact"], j, "exact"))
                units.append({"t": ph, "c": cands, "cover": [a, b]})
        for t in toks:
            cands = self.candidates(t, self._raw.get(t))
            for o in self.syn.get(t, []):
                cands += self._phrase_cands(o, Q["syn"])
            fz = [self.terms[j] for q, j, how in cands if how == "fuzzy"]
            if fz and not any(how == "exact" for _, _, how in cands):
                res["corrected"].append([t, fz[0]])
            units.append({"t": t, "c": cands, "cover": [t]})
        # pairs of the query (adjacent content words) that the index knows: a phrase bonus
        qpairs = [j for j in (self.tix.get(p) for p in pairs(seq)) if j is not None]
        # related (REL) for the whole query, and for each term with no strong candidate
        rel_terms = []
        keys = [" ".join(seq)] + [a + " " + b for a, b in zip(seq, seq[1:])]
        keys += [u["t"] for u in units if not any(q >= Q["fuzzy2"] for q, _, _ in u["c"])]
        for key in keys:
            for tt in self.rel.get(key, []):
                j = self.tix.get(tt)
                if j is not None and j not in rel_terms:
                    rel_terms.append(j)
        n_units = len(toks)
        scores = {}
        # per document: best match per unit, its field, its quality
        for ui, u in enumerate(units):
            best = {}
            # a looser match never weighs more than the exact word would: read -> reader is capped at idf(read)
            cap = max([self.idf[j] for q, j, how in u["c"] if how == "exact"] or [99.0])
            for q, j, how in u["c"]:
                w = q * min(self.idf[j], cap)
                for di, fi in self.post.get(j, ()):
                    s = self.fw[fi] * w
                    if di not in best or best[di][0] < s:
                        best[di] = (s, fi, q)
            for di, (s, fi, q) in best.items():
                e = scores.setdefault(di, {"s": 0.0, "cov": set(), "fmin": 9, "rel": 0.0, "pair": 0.0})
                e["s"] += s
                e["cov"].update(u["cover"])
                if q >= Q["fuzzy2"] and fi < e["fmin"]:
                    e["fmin"] = fi
        for j in qpairs:
            for di, fi in self.post.get(j, ()):
                if fi < 3:
                    e = scores.setdefault(di, {"s": 0.0, "cov": set(), "fmin": 9, "rel": 0.0, "pair": 0.0})
                    e["pair"] = max(e["pair"], 0.5 * self.fw[fi] * self.idf[j])
        if rel_terms:
            # a related topic listed first counts most (the lexicon lists them closest first); an item that carries
            # several of them gathers them, with diminishing returns (1, 0.6, 0.4, 0.3)
            relhits = {}
            for rank, j in enumerate(rel_terms):
                ow = 1.0 / (1 + 0.15 * rank)
                for di, fi in self.post.get(j, ()):
                    if fi < 3:
                        relhits.setdefault(di, []).append(self.fw[fi] * Q["rel"] * self.idf[j] * ow)
            for di, xs in relhits.items():
                xs.sort(reverse=True)
                e = scores.setdefault(di, {"s": 0.0, "cov": set(), "fmin": 9, "rel": 0.0, "pair": 0.0})
                e["rel"] = sum(x * w for x, w in zip(xs, (1.0, 0.6, 0.4, 0.3)))
        hits = []
        qset = set(toks)
        for di, e in scores.items():
            d = self.docs[di]
            cov = len(e["cov"]) / n_units
            s = (e["s"] + e["pair"]) * (0.25 + 0.75 * cov * cov) + e["rel"] * 0.6
            # the whole title asked for: every title word in the request and every request word in the title
            tset = self.tterms[di]
            if tset and qset and tset <= qset | {"and"} and cov == 1:
                s *= 1.5
            s *= TYPE_W.get(d[1], 1.0)
            why = ["title", "keyword", "concept", "body"][e["fmin"]] if e["fmin"] < 4 else ("related" if e["rel"] else "body")
            if e["fmin"] < 9 and cov < 0.5 and e["rel"]:
                why = "related"
            hits.append([r4(s), di, why, cov])
        hits.sort(key=lambda h: (-h[0], h[1]))
        if not hits:
            return res
        top = hits[0][0]
        best_why = hits[0][2]

        def ingrade(di):
            g = self.docs[di][3]
            return not g or not want or any(x in want for x in g)
        strong = bool(top >= STRONG_MIN and best_why in ("title", "keyword", "concept"))
        # the class grade first among the GOOD matches: within GOOD of the best score AND covering as much of the
        # request as the best hit does (a grade-less tip that shares one word does not jump a full match); the same in
        # a list of related topics (nothing found); below the good matches the class grade weighs 1.25 times
        tcov = hits[0][3]
        good = [h for h in hits if h[0] >= GOOD * top and h[3] >= tcov]
        gset = {h[1] for h in good}
        rest = sorted((h for h in hits if h[1] not in gset),
                      key=lambda h: (-(r4(h[0] * (IN_GRADE if ingrade(h[1]) else 1.0))), h[1]))
        ordered = [h for h in good if ingrade(h[1])] + [h for h in good if not ingrade(h[1])] + rest
        res["strong"] = strong
        if word and explicit and not strong:
            return res          # 'how do you say cow': the word card is the answer; no list of loose relatives
        for s, di, why, cov in ordered[:k]:
            d = self.docs[di]
            g = d[3]
            res["hits"].append({"id": d[0], "type": d[1], "title": d[2], "grades": g,
                                "grade": next((x for x in g if x in want), g[0] if g else None) if want else (g[0] if g else None),
                                "subject": d[4], "sub": d[5], "review": bool(d[6] & 1), "toi": bool(d[6] & 2),
                                "other_grade": not ingrade(di), "why": why, "score": s})
        return res

    def _phrase_cands(self, phrase, q):
        """a SYN equivalent -> candidates. One word: the word. Several words ('slow learner', 'below grade level'):
        only their adjacent pairs that the index knows, never the single words ('level' alone is not 'below grade
        level')"""
        ts = phrase.split(" ")
        if len(ts) == 1:
            j = self.tix.get(ts[0])
            return [(q, j, "syn")] if j is not None else []
        out = []
        for a, b in zip(ts, ts[1:]):
            j = self.tix.get(a + "_" + b)
            if j is not None:
                out.append((q, j, "syn"))
        return out
