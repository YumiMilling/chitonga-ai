"""Child-safeguarding detector, by CATEGORY (24 Oct 2026, the Jesuit). Pure Python, no model, no network.

WHY THIS FILE. The reader bake-off (internal/chitonga_train/runs/2026-10-24_reader_bakeoff/REPORT.md section 3c) found
that the old gate (h3_front.SAFE_STRONG / SAFE_CHILD / SAFE_WEAK) missed seven kinds of disclosure in every arm,
rules-only included, and two of them then landed on lessons (Money, Forces). The causes, by category, not by sentence:
  - a short list of CLINICAL words (abuse, defile, rape, molest, sexual) and a few FIXED PHRASES ('married off', 'chased
    from home'): teachers describe what happened in plain words ('sleeps with men', 'marry her to', 'chased him',
    'naked pictures', 'not given food'), and any other inflection or word order passed;
  - whole categories had no pattern at all: images and online abuse, adult cover-up / pressure to keep a secret, child
    labour beyond the two words 'child labour', self-harm beyond 'kill himself';
  - exact spelling: 'childten', 'head tr', 'beatn', 'maried' passed; a reader that echoed the typos let them through;
  - and in the other direction, bare weak words with a child in the message ('hit', 'beating', 'burnt') sent lesson and
    advice requests to the referral ('hit the ball', 'discipline without beating', 'burnt the rubbish').

HOW IT WORKS
  1 normalise: lowercase, punctuation out, apostrophes dropped; SMS / teacher spellings mapped (tr -> teacher,
    gal -> girl, bcoz -> because, pics -> pictures, 4 money -> for money ...); a token that is not an English word
    (hunspell en_GB + en_US) and lies within one edit (two for 9+ letters, same first letter) of a KEY word is read as
    that word ('childten' -> children, 'bleding' -> bleeding). English words near a key word are protected
    (safeguard_near.json, built by this file: 'beans' never becomes 'beats', 'sheep' never 'sleep').
  2 benign phrases are blanked (beat the drum, hit the ball, heartbeat, discipline without beating, sex education, how
    to say bride price in Tonga ...): they never count for a rule.
  3 rules by category. Each rule is one or more regexes that must ALL match; 'ctx' rules also need a young person in
    the message (child / learner / pupil / girl / boy / daughter ..., 'grade 3', an age under 18, 'told me', or 'her
    stepfather' etc.). The categories: sexual abuse and exploitation (incl. transactional sex and grooming), child
    marriage / betrothal, physical abuse, neglect (incl. food deprivation), abandonment / chasing from home, images
    and online abuse, adult cover-up / pressure to keep a secret, child labour, self-harm; plus trafficking / danger.
  4 run on the light form (lowercase, punctuation out) AND the normalised form; the first hit wins.

The referral itself is not here (tools/chitonga/assistant/gate.REFERRAL: head / guidance teacher or focal person today,
Childline 116, the police Victim Support Unit, the District Social Welfare Office; Children's Code Act No. 12 of 2022). Tuned ONLY on
tools/chitonga/harness/tests/safeguard/dev_v1.jsonl; the bake-off rows were run once afterwards (their folder carries
SAFEGUARD_ROWS_SEEN.md: no longer held out). Every rule errs on the side of catching: a false referral costs a teacher
one extra line; a missed disclosure can cost a child.

    python -X utf8 tools/chitonga/llm/safeguard.py "text"        what it catches and why
    python -X utf8 tools/chitonga/llm/safeguard.py --build-near  rebuild safeguard_near.json (after KEYS change)
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEAR_PATH = os.path.join(HERE, "safeguard_near.json")
EN_DICT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "internal", "chitonga_train", "vocab",
                                        "en_dictionary_words.json"))

CATEGORIES = ("sexual", "marriage", "physical", "neglect", "abandonment", "images", "secrecy", "labour", "self_harm",
              "danger")

# ------------------------------------------------------------------------------------------------ 1 normalisation
# phrase rewrites on the lowercased text, before tokenising (order matters)
PHRASES = [
    (r"\bh\s*/\s*t\b|\bh\.\s*t\b", " head teacher "),
    (r"\bs\s*/\s*he\b", " she "),
    (r"\by\s*/\s*o\b", " years old "),
    (r"\bstep[\s-]+(father|mother|dad|daddy|mum|mom|mummy|son|daughter|brother|sister)\b", r" step\1 "),
    (r"\bhead[\s-]*tr\b", " head teacher "),
    (r"\b4\s+(?=(money|cash|sex|food|gifts?|marks|school|fees|k\d+|kwacha|airtime)\b)", " for "),
    (r"\b2\s+(?=(sleep|keep|tell|report|have|marry|work|go|eat|die|kill|stay|be|get|take|sell|meet|send|herd)\b)", " to "),
]
PHRASES_RX = [(re.compile(p), r) for p, r in PHRASES]

# token -> replacement (applied before the English-word check: these win even when the token is an English word)
SMS = {
    "tr": "teacher", "trs": "teachers", "tchr": "teacher", "tchrs": "teachers", "ht": "head teacher",
    "hm": "headmaster", "lrnr": "learner", "lrnrs": "learners", "leaner": "learner", "leaners": "learners",
    "pupl": "pupil", "pupls": "pupils", "puple": "pupil", "puples": "pupils", "chld": "child", "chldrn": "children",
    "childs": "children", "childrens": "children", "kid": "child", "kids": "children", "gal": "girl", "gals": "girls",
    "gurl": "girl", "gurls": "girls", "gril": "girl", "grils": "girls", "grl": "girl", "grls": "girls", "boi": "boy",
    "gr": "grade", "grd": "grade", "gd": "grade", "yrs": "years", "yr": "year", "yo": "years old",
    "mum": "mother", "mom": "mother", "mama": "mother", "mummy": "mother", "mommy": "mother", "dad": "father",
    "daddy": "father", "papa": "father", "farther": "father", "stepdad": "stepfather", "stepdaddy": "stepfather",
    "stepmum": "stepmother", "stepmom": "stepmother", "stepmummy": "stepmother", "aunty": "aunt", "auntie": "aunt",
    "granny": "grandmother", "grandma": "grandmother", "grandpa": "grandfather", "bf": "boyfriend",
    "guy": "man", "guys": "men", "ol": "old",
    "hv": "have", "hav": "have", "hs": "has", "dnt": "dont", "nt": "not", "nvr": "never", "wnt": "want",
    "tld": "told", "tel": "tell", "tels": "tells", "sez": "says", "sayz": "says", "sed": "said", "wit": "with",
    "wid": "with", "wth": "with", "wiv": "with", "u": "you", "ur": "your", "r": "are", "nd": "and",
    "bcoz": "because", "bcos": "because", "bcz": "because", "becoz": "because", "coz": "because", "cos": "because",
    "cuz": "because", "abt": "about", "nyt": "night", "nite": "night", "evry": "every", "evryday": "every day",
    "everyday": "every day", "smone": "someone", "sm1": "someone", "som1": "someone", "pls": "please",
    "plz": "please", "pliz": "please", "dat": "that", "dis": "this", "wen": "when", "hr": "her", "hiz": "his",
    "hym": "him", "thm": "them", "dem": "them", "dey": "they", "clas": "class", "nxt": "next", "mnth": "month",
    "bein": "being", "til": "till", "untill": "until", "gona": "going to", "gonna": "going to", "wanna": "want to",
    "pic": "picture", "pics": "pictures", "pix": "pictures", "picz": "pictures", "foto": "photo", "fotos": "photos",
    "vid": "video", "vids": "videos", "nudes": "nude pictures", "sx": "sex", "secks": "sex", "sexx": "sex",
    "seks": "sex", "raiped": "raped", "nakd": "naked", "privet": "private", "privates": "private parts",
    "beatin": "beating", "beatn": "beaten",
}

# KEY words: an unknown token one edit away (two for 9+ letters, same first letter) is read as the key word
KEYS = (
    "child", "children", "learner", "learners", "pupil", "pupils", "student", "students", "girl", "girls", "daughter",
    "daughters", "stepfather", "stepmother", "father", "mother", "uncle", "guardian", "guardians", "teacher",
    "teachers", "grandmother", "grandfather", "brother", "cousin", "neighbour", "boyfriend", "headmaster",
    "beat", "beats", "beaten", "beating", "bleed", "bleeds", "bleeding", "bruise", "bruises", "wounds", "whip",
    "whips", "whipped", "slapped", "slaps", "punched", "kicked", "burnt", "burned", "naked", "nude", "pictures",
    "picture", "photos", "photo", "videos", "video", "married", "marry", "marriage", "marrying", "betrothed",
    "lobola", "wife", "sexual", "sexually", "abuse", "abused", "abuses", "defiled", "defile", "defilement", "rape",
    "raped", "molested", "touches", "touched", "touching", "private", "secret", "quiet", "report", "police",
    "punishment", "punished", "punishes", "starve", "starved", "starves", "starving", "chased", "chases",
    "abandoned", "pregnant", "suicide", "poison", "labour", "forced", "forces", "forcing", "sleeps", "sleeping",
    "money", "threatened", "threatens", "scared", "afraid", "because", "without", "refuse", "refuses", "refused",
    "every", "night", "someone", "anyone", "himself", "herself", "themselves",
)


def _dl(a, b, cap):
    """optimal string alignment distance, early exit above cap"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
            best = min(best, cur[j])
        if best > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


def _cap(token):
    return 2 if len(token) >= 9 else 1


def _fuzzy_key(token, protect):
    """the KEY word this token is a misspelling of, or None"""
    if len(token) < 5 or token in protect or token in _KEYSET or not token.isalpha():
        return None
    cap = _cap(token)
    best, bd = None, cap + 1
    for k in KEYS:
        if k[0] != token[0] or abs(len(k) - len(token)) > cap:
            continue
        d = _dl(token, k, cap)
        if d < bd:
            best, bd = k, d
    return best if bd <= cap else None


_KEYSET = set(KEYS)
_NEAR = None


def keys_hash():
    return hashlib.sha1(("|".join(KEYS) + "#1").encode()).hexdigest()[:12]


def near_words():
    """English words (hunspell base forms + regular inflections) that lie within the fuzzy distance of a KEY word: they
    are never rewritten. Loaded from safeguard_near.json (built by --build-near)."""
    global _NEAR
    if _NEAR is None:
        try:
            d = json.load(open(NEAR_PATH, encoding="utf-8"))
            _NEAR = set(d["words"]) if d.get("keys_hash") == keys_hash() else None
        except (OSError, ValueError, KeyError):
            _NEAR = None
        if _NEAR is None:          # stale or missing: fall back to building from the dictionary, else no fuzzy at all
            try:
                _NEAR = set(build_near(write=False)["words"])
            except (OSError, ValueError, KeyError):
                _NEAR = {"__nofuzzy__"}
    return _NEAR


def build_near(write=True):
    words = json.load(open(EN_DICT, encoding="utf-8"))["lower"]
    cand = set()
    for w in words:
        if not w.isalpha():
            continue
        cand.add(w)
        cand.update((w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "er", w + "ers", w + "ly"))
        if w.endswith("e"):
            cand.update((w[:-1] + "ing", w[:-1] + "ed"))
        if w.endswith("y"):
            cand.update((w[:-1] + "ies", w[:-1] + "ied"))
        if len(w) >= 3 and w[-1] not in "aeiouwxy" and w[-2] in "aeiou" and w[-3] not in "aeiou":
            cand.update((w + w[-1] + "ed", w + w[-1] + "ing", w + w[-1] + "er"))
    near = set()
    for t in cand:
        if len(t) < 5 or t in _KEYSET:
            continue
        cap = _cap(t)
        for k in KEYS:
            if k[0] == t[0] and abs(len(k) - len(t)) <= cap and _dl(t, k, cap) <= cap:
                near.add(t)
                break
    near -= set(SMS)        # an SMS spelling always wins
    out = {"keys_hash": keys_hash(), "source": "hunspell en_GB + en_US (vocab/en_dictionary_words.json) + regular "
           "inflections; words within the fuzzy distance of a safeguard KEY word", "words": sorted(near)}
    if write:
        with open(NEAR_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=0)
    return out


_TOK = re.compile(r"[^\W_]+", re.UNICODE)


def light(text):
    """lowercase, apostrophes dropped, punctuation to spaces"""
    t = (text or "").lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    t = t.replace("'", "")
    return " ".join(_TOK.findall(t))


def normalise(text):
    """the reading the rules run on: light() + phrase rewrites + SMS map + fuzzy KEY words + letter runs cut to two"""
    t = (text or "").lower().replace("’", "'").replace("‘", "'").replace("`", "'").replace("'", "")
    for rx, rep in PHRASES_RX:
        t = rx.sub(rep, t)
    protect = near_words()
    out = []
    for tok in _TOK.findall(t):
        tok = re.sub(r"(.)\1{2,}", r"\1\1", tok)
        if tok in SMS:
            out.append(SMS[tok])
            continue
        k = None if "__nofuzzy__" in protect else _fuzzy_key(tok, protect)
        out.append(k or tok)
    return " ".join(out)


# ------------------------------------------------------------------------------------------------ 2 context, benign
CTX = re.compile(
    r"\b(child|children|childs|kid|kids|learner|learners|pupil|pupils|student|students|girl|girls|boy|boys|daughter|"
    r"daughters|son|sons|niece|nieces|nephew|nephews|orphan|orphans|baby|babies|toddler|toddlers|teen|teens|teenager|"
    r"teenagers|minor|minors|schoolgirl|schoolgirls|schoolboy|schoolboys|twins|mwana|bana|kasimbi|kamwana|musimbi)\b"
    r"|\bgrade\s*\d\b|\bgr\s*\d\b|\b(1[0-7]|[1-9])\s*(year|years|yr|yrs)\b|\baged?\s*(1[0-7]|[1-9])\b"
    r"|\b(she|he) is (only |just )?(1[0-7]|[1-9])\b"
    r"|\b(told|tells|telling|confided|disclosed|reported|complained|came)\s+(to\s+)?(me|us)\b"
    r"|\b(she|he|they)\s+(told|tells|said|says|wrote|writes|confided|disclosed|reported|complained|cried)\b"
    r"|\b(her|his|their)\s+(stepfather|stepmother|father|mother|uncle|aunt|grandmother|grandfather|guardian|guardians|"
    r"parents|brother|sister|cousin)\b")

BENIGN = [
    r"\bbeat(s|ing)? (the |a |our )?(drum|drums|time|rhythm|beat|record|eggs?|mixture|clock|heat)\b",
    r"\b(heart|drum) ?beats?\b", r"\bbeats? (per|of the)\b",
    r"\bhit(s|ting)? (the |a )?(ball|target|nail|drum|board|wall|mark|road|jackpot)\b",
    r"\bkick(s|ed|ing)? (the |a )?ball\b",
    r"\b(beat|beats|beaten|beating) (us|them|him|her|the other|another|our)( team| class| school)? (at|in) (the |a )?"
    r"(game|games|match|race|football|netball|chess|draughts|quiz|competition|sports?|running|spelling|maths|cards|tug of war)\b",
    r"\bbeaten (at|in) (the |a )?(game|match|race|football|netball|competition|quiz)\b",
    # the teacher's own advice question, never a report about someone else ('he never stops beating her' stays)
    r"\b(without|instead of|alternatives? to|avoid|avoiding|ban on|banned|illegal|other than) (using |to use )?"
    r"(the |any )?(corporal punishment|beating|beat|caning|cane|canes|hitting|hit|whipping|smacking|slapping)( them| him| her| learners| pupils| children)?",
    r"\b(i|we|teachers|you) (do not|dont|cannot|cant|should not|shouldnt|must not|may not|are not allowed to) "
    r"(beat|cane|hit|whip|smack|slap)\b",
    r"\bhow (do|can|should) (i|we) (stop|avoid) (beating|caning|hitting|whipping)\b",
    r"\b(past tense|tense|verb|verbs|spell\w*|word|words|meaning) (of |for )?(beat|hit|run|kick|burn)\b",
    r"\b(fire|matches|hot water|the sun|candle|stove|iron) (can |will |may )?burns?\b",
    r"\bburn(t|ed|s|ing)? (the )?(rubbish|grass|bush|paper|charcoal|wood|fields?|bricks?|calories|fuel)\b",
    r"\bsex (education|organs?|of (the|an?) \w+)\b|\b(male|female|opposite) sex\b",
    r"\b(how (do|to) (you |i |we )?say|word for|tonga word|in tonga|translate|translation of|vocabulary|meaning of|"
    r"lesson on|lesson about|customs?|traditions?|traditional)( \w+){0,4} (lobola|bride ?price|marriage|marriages|wedding|"
    r"weddings|wife|husband|married)\b",
    r"\b(good|bad|safe|unsafe|okay|ok) touch\b",
    r"\bfirst aid\b|\bhow to (clean|treat|dress) (a |the )?(wound|cut|burn)s?\b",
    r"\bmissing (\w+ )?(numbers?|words?|letters?|sounds?|values?|parts?|addends?|digits?|terms?|pieces?|shapes?|items?)\b",
    r"\bthreatened (species|animals?|plants?|habitats?)\b",
    r"\b(marriage|marriages|wedding|weddings|lobola|bride ?price) (customs?|traditions?|vocabulary|words?|songs?|"
    r"ceremon\w*|lesson|lessons|story|stories|in the past)\b",
    r"\b(types|kinds|forms) of marriages?\b",
    # ('marks' is left out on purpose: 'not to tell social welfare about the marks on the boy' is a cover-up)
    r"\b(not|never|dont|do not)( to)? (tell|say|show|give)( \w+){0,3} (the |their |your )?(answers?|results?|"
    r"scores?|password|surprise|ending|secret number)\b",
]
BENIGN_RX = [re.compile(p) for p in BENIGN]


def _blank(t):
    for rx in BENIGN_RX:
        t = rx.sub(" ~ ", t)
    return t


# ------------------------------------------------------------------------------------------------ 3 rules
# a young person as the object of the act; one adjective allowed after the determiner ('the small pupils')
PER = (r"(?:her|him|them|me|us|(?:the|a|my|our|this|that|one|his|her|their|some|other) (?:\w+ ){0,2}(?:child|children|girl|"
       r"girls|boy|boys|learner|learners|pupil|pupils|kid|kids|baby|babies|orphan|orphans|twins|daughter|son|niece|"
       r"nephew|sister|brother)|learners|pupils|children|girls|boys)")
ADULT = (r"(?:man|men|adult|adults|uncle|stepfather|stepmother|father|mother|teacher|teachers|driver|neighbour|guard|"
         r"watchman|boyfriend|pastor|relative|relatives|stranger|cousin|brother|headmaster|head teacher|deputy|someone|"
         r"somebody|aunt|grandmother|grandfather|guardian|guardians|parents?|husband|employer|police\w*)")
VIOL = (r"(?:beat|beats|beaten|beating|hit|hits|hitting|whip|whips|whipped|whipping|flog\w*|lash\w*|slap\w*|punch\w*|kick|kicks|"
        r"kicked|kicking|cane|canes|caned|caning|smack\w*|thrash\w*|batter\w*|strangl\w*|chok(?:e|es|ed|ing)|burn|burns|"
        r"burnt|burned|burning|scald\w*|stab\w*|push|pushes|pushed|assault\w*|tortur\w*|bit|bites|"
        r"hurt|hurts|hurting|injur\w*)")
NEG = r"(?:not|never|no|dont|do not|doesnt|does not|didnt|did not|wont|will not|cant|cannot|isnt|is not|wasnt|was not)"
FOOD = (r"(?:food|fed|feed|feeding|meals?|to eat|anything to eat|eating|eat|nshima|nsima|breakfast|supper|lunch|dinner|"
        r"nothing to eat)")
DUR = (r"(?:for days|days|\d+ days|two days|three days|four days|a day|whole day|the whole day|a week|weeks|the night|"
       r"whole night|the whole night|many days|a month|overnight)")
AGE = r"(?:\b(?:1[0-7]|[5-9])\s*(?:years?|yrs?)(?: old)?\b|\baged? (?:1[0-7]|[5-9])\b|\b(?:she|he) is (?:only |just )?(?:1[0-7]|[5-9])\b|\b(?:girl|boy|child|learner|pupil|daughter) of (?:1[0-7]|[5-9])\b)"
MARRY = r"(?:marri\w*|marry\w*|wedding|wife|bride|husband|lobola|bride ?price|dowry|kitchen party|betroth\w*)"
SCHOOL_OUT = (r"(?:stopped coming|stop coming|stopped school|left school|leave school|leaves school|withdrawn from school|"
              r"withdrew (?:\w+ )?from school|removed from school|taken out of school|pulled out of school|drop(?:ped|s)? out|"
              r"must leave school|no longer comes?)")
ONLINE = r"(?:facebook|whatsapp|tiktok|instagram|snapchat|online|internet|social media|phone|chat\w*|messag\w*|texts?|texting)"
IMG = r"(?:pictures?|photos?|photographs?|images?|videos?|films?|movies?|selfies?|clips?)"

R = []          # (category, [regex, ...], needs_ctx)


def rule(cat, *rxs, ctx=False):
    R.append((cat, [re.compile(x) for x in rxs], ctx))


# --- sexual abuse and exploitation (incl. transactional sex and grooming)
rule("sexual", r"\b(?:abus\w*|defil\w*|rap(?:e|ed|es|ing|ist|ists)|molest\w*|incest\w*|sexual\w*|indecen\w*|sodom\w*|"
               r"paedophil\w*|pedophil\w*|prostitut\w*|sex ?work\w*|transactional sex|sugar dadd(?:y|ies)|blessers?|"
               r"pregnan\w*|impregnat\w*)\b")
rule("sexual", r"\b(?:have|has|had|having|do|does|did|doing) sex\b|\bsex (?:with|for)\b|\bfor sex\b|\bforc\w* (?:\w+ ){0,3}(?:to )?(?:have )?sex\b")
rule("sexual", r"\bsex\b", ctx=True)
rule("sexual", r"\b(?:sleep|sleeps|sleeping|slept) (?:together )?with (?:a |an |the |some |many |different |older |old |big |"
               r"grown |married |rich )*(?:men|man|adults?|grown ups?|teachers?|drivers?|uncle|stepfather|boyfriends?|"
               r"soldiers?|traders?|truck drivers?)\b")
rule("sexual", r"\b(?:sleep|sleeps|sleeping|slept) together\b", r"\b(?:boyfriend|man|men|teacher|uncle|older|adult)\b", ctx=True)
rule("sexual", r"\b(?:sleep|sleeps|sleeping|slept) with (?:her|him)\b", ctx=True)
rule("sexual", r"\btouch\w* (?:her|his)(?: \w+)? (?:private|privates|breasts?|chest|bottom|buttocks?|thighs?|vagina|penis|genitals?|"
               r"between (?:her|his|their) legs|under (?:her|his|their) (?:dress|skirt|clothes))\b")
rule("sexual", r"\btouch(?:es|ed|ing)? (?:her|him)\b", ctx=True)
rule("sexual", r"\btouch\w* (?:\w+ ){0,3}(?:inappropriat\w*|badly|wrongly|in a bad way)\b")
rule("sexual", r"\b(?:remove|removes|removed|removing|take off|takes off|took off|pull\w* down) (?:her|his|their) "
               r"(?:pants|panties|knickers|underwear)\b")
rule("sexual", r"\b(?:undress\w*|(?:remove|removes|removed|removing|take off|takes off|took off|pull\w* down) (?:her|his|their) "
               r"(?:clothes|dress|skirt|trousers|shorts))\b", ctx=True)
rule("sexual", r"\b(?:lies|lie|lay|laid|lying) on top of (?:her|him|them)\b")
rule("sexual", r"\b(?:comes?|came|coming|gets?|got|getting|goes|went|creeps?|crept|enters?|entered) (?:in)?(?:to )?(?:her|his|their) "
               r"(?:bed|beds|blankets?|sleeping (?:place|room|mat))\b", r"\b" + ADULT + r"\b", ctx=True)
rule("sexual", r"\b(?:fondl\w*|grop\w*|caress\w*)\b", ctx=True)
# a young person handed to men ('sold his maize to the men at the market' has no person object)
rule("sexual", r"\b(?:sell|sells|selling|sold|offer\w*|send|sends|sent|sending|take|takes|took|taking|give|gives|gave) "
               r"(?:her|him|them|(?:his|her|their|the|a) (?:\w+ )?(?:daughter|girl|girls|child|children|niece|sister|son|boy|boys)) "
               r"to (?:the |some |many |older |old |rich |different |grown )*(?:men|truck drivers|soldiers)\b")
rule("sexual", r"\bto (?:a|an|the|that|this|some) (?:old |older |rich |grown )?man\b",
     r"\b(?:money|pay|pays|paid|paying|k\d+|kwacha|cash|gifts?)\b", ctx=True)
rule("sexual", r"\b(?:does|do|did|doing|done) (?:it|this|that|things|bad things|dirty things|sex things) (?:with|to|for) "
               r"(?:men|a man|older|old men|grown|boys|him|her|them)\b", ctx=True)
rule("sexual", r"\b(?:bad|dirty|nasty|evil|terrible) things (?:to|with) (?:her|him|them|me|the \w+)\b", ctx=True)
rule("sexual", r"\b(?:gives?|gave|giving|buys?|bought|buying|sends?|sent) (?:her|him|them|the (?:girl|boy|child|learner|pupil)) "
               r"(?:\w+ ){0,2}(?:money|cash|gifts?|presents?|phones?|airtime|sweets|clothes|lotion|k\d+)\b",
     r"\b(?:secret|not to tell|tell no ?one|come to his|go to his|his house|his room|at night|alone|touch\w*|sleep\w*|"
     r"boyfriend|love)\b")
rule("sexual", r"\b(?:ask\w*|call\w*|tell\w*|told|invit\w*|take\w*|took) (?:her|him|them|the (?:girl|boy|child)) (?:\w+ ){0,2}"
               r"(?:to come )?(?:to|into|behind|at) (?:his|the) (?:house|room|home|office|car|bush|classroom|toilets?|back)\b",
     r"\b(?:at night|alone|secret|money|sweets|gifts?|pants|touch\w*|after school|behind)\b", ctx=True)
rule("sexual", r"\bpropos\w* (?:love )?(?:to )?(?:her|him|a girl|the girl|girls|learners?|pupils?|a pupil|a learner)\b", ctx=True)
rule("sexual", r"\blove (?:letters?|messages?|notes?|affair|affairs|relationship)\b", ctx=True)
rule("sexual", r"\b(?:relationship|affair) with (?:a|an|her|his|the|their) (?:\w+ )?(?:teacher|man|older|adult|married|uncle)\b", ctx=True)
rule("sexual", r"\bboyfriend (?:who is |is |of )?(?:a |an )?(?:\d\d|old|older|grown|married|big|adult)\b", ctx=True)
rule("sexual", r"\b(?:[2-6]\d) ?(?:year|years|yr|yrs)?(?: old)? (?:man|men)\b", ctx=True)
rule("sexual", r"\brapped\b", ctx=True)

# --- child marriage or betrothal
rule("marriage", r"\b(?:early|child|forced|underage|under age) marriages?\b|\bchild brides?\b|\bbetroth\w*\b")
rule("marriage", r"\bmarr(?:y|ies|ied|ying)(?: (?:her|him|them|(?:the|his|her|their|a|my|our) (?:\w+ )?(?:daughter|girl|girls|"
                 r"child|niece|sister|learner|pupil)))? off\b")
rule("marriage", r"\b(?:give|gives|giving|gave|given|promis\w*|offer\w*|sold|sell\w*) (?:\w+ ){0,5}(?:in marriage|as (?:a |his |"
                 r"the )?(?:\w+ )?(?:wife|bride)|to (?:a |an |the )?(?:\w+ )?man as)\b")
rule("marriage", r"\b(?:to be|be|being|get|gets|getting|got|was|is|will be|has been|forced to|made to|is to be|going to be|"
                 r"going to get) married\b", ctx=True)
rule("marriage", r"\b(?:wants?|wanted|forc\w*|mak\w*|made|arrang\w*|plan\w*|told|tells|pressur\w*) (?:\w+ ){0,3}to (?:marry|wed)\b", ctx=True)
rule("marriage", r"\b(?:taken|take|took|taking|become|becomes|became|to be) (?:as )?(?:a |his |the )?(?:second |third |fourth |"
                 r"young |new )?wife\b", ctx=True)
rule("marriage", r"\b(?:is|was|been|being) promised to (?:a|an|the|his|her) \w+", ctx=True)
rule("marriage", r"\b(?:lobola|bride ?price|dowry|bride ?wealth)\b", ctx=True)
rule("marriage", r"\b(?:accept\w*|receiv\w*|took|taken|paid|pay\w*) (?:\w+ ){0,2}(?:cattle|cows|goats|money)\b",
     r"\b(?:daughter|wife|marry|marri\w*|bride)\b", ctx=True)
# marriage EVENT words with a stated age under 18 ('husband and wife' in a family lesson for 7 year olds is not one)
rule("marriage", r"\b(?:marri\w*|marry\w*|wedding|lobola|bride ?price|dowry|kitchen party|betroth\w*|(?:his|a|second|third|new|"
                 r"young) wife|as (?:a |his )?wife|to (?:a |an )?(?:old |older |rich )?husband)\b", AGE)
rule("marriage", r"\b(?:marri\w*|marry\w*|wife|bride|husband|lobola|kitchen party|wedding)\b", SCHOOL_OUT, ctx=True)
rule("marriage", r"\b(?:withdrawn|taken|removed|pulled|stopped) (?:\w+ ){0,3}(?:school|class) (?:\w+ ){0,2}(?:for|to get|to) "
                 r"(?:marriage|marry|married)\b")

# --- physical abuse
rule("physical", r"\b" + VIOL + r"\b(?: \w+){0,8} (?:until|till|untill|and|so|so hard|so badly|that) (?:s?he|they|it|the child|the girl|"
                 r"the boy|her|his|him|some|one) (?:\w+ ){0,3}(?:bleed\w*|bled|faint\w*|passed out|pass out|collaps\w*|lost consciousness|"
                 r"cannot walk|cant walk|could not walk|couldnt walk|swell\w*|was hospitali\w*|went to hospital)\b")
rule("physical", r"\b" + VIOL + r" " + PER + r"\b", ctx=True)
rule("physical", r"\b" + VIOL + r" " + PER + r" (?:\w+ ){0,4}with (?:a |an |the |his |her |electric |electrical |hot )?(?:pipe|hosepipe|"
                 r"hose|stick|sticks|belt|buckle|wire|cable|sjambok|whip|chain|rod|iron|plank|stone|stones|brick|fist|fists|"
                 r"shoe|shoes|boot|boots|knife|panga|axe|hammer|bottle|water|oil)\b")
rule("physical", r"\b" + VIOL + r" (?:his|her|their) (?:hands?|arms?|legs?|face|back|body|feet|foot|head|fingers?|skin|ears?|"
                 r"bottom|buttocks|mouth|eyes?|neck)\b", r"\b" + ADULT + r"\b", ctx=True)
rule("physical", r"\b(?:beaten|whipped|flogged|caned|slapped|punched|kicked|burnt|burned|battered|assaulted|tortured|hit|"
                 r"thrashed|stabbed|injured|hurt) (?:\w+ ){0,3}by (?:a |an |the |his |her |their |my |our )?(?:\w+ )?" + ADULT + r"\b")
rule("physical", r"\b(?:is|was|were|are|being|been|got|get|gets|getting) (?:\w+ ){0,2}(?:beaten|whipped|flogged|caned|slapped|"
                 r"punched|kicked|burnt|burned|battered|assaulted|tortured|thrashed|stabbed)\b", ctx=True)
rule("physical", r"\b(?:bruis\w*|wounds?|wounded|scars?|swollen|swelling|burn marks?|whip marks?|marks on (?:her|his|their) "
                 r"(?:body|back|legs?|arms?|face)|injur\w*|broken (?:arm|leg|hand|rib|ribs|nose|tooth|teeth|finger|bone))\b", ctx=True)
rule("physical", r"\b(?:corporal punishment|threat\w*|violen\w*)\b", ctx=True)
rule("physical", r"\b(?:put|poured|pours|threw|throws) (?:hot|boiling) (?:water|oil|porridge) (?:on|over) (?:her|him|them|the \w+)\b")
rule("physical", r"\b(?:ties|tied|tie|tying|chain\w*) (?:her|him|them|the \w+) (?:to|up)\b", ctx=True)
rule("physical", r"\bhurt(?:s|ing)? (?:her|him|them|me)\b|\b\w*m?uuma\b|\bkuuma\b|\bbalamuuma\b", ctx=True)

# --- neglect (incl. food deprivation)
rule("neglect", r"\bneglect\w*\b")
rule("neglect", r"\b(?:" + NEG + r"|refus\w*|den(?:y|ies|ied|ying)|deprive\w*|withh[oe]ld\w*|stop\w*|without|no)\b(?: \w+){0,4} " + FOOD + r"\b",
     r"\b(?:punish\w*|" + DUR[3:-1] + r"|when (?:s?he|they) (?:makes?|comes?|does|fails?|refuses?|is late|breaks?|loses?)|lock\w*)\b")
rule("neglect", r"\b(?:den(?:y|ies|ied|ying)|deprive\w*|withh[oe]ld\w*|refus\w*) (?:\w+ ){0,3}(?:food|meals?|to eat|eating)\b", ctx=True)
rule("neglect", r"\b(?:goes|go|went|gone|going|stays?|stayed|spends?|spent|sleeps?|slept|is) (?:for )?(?:days|\d+ days|two days|"
                r"three days|a week|weeks|the whole day|whole days|many days|long) without (?:food|eating|a meal|meals|anything to eat)\b")
rule("neglect", r"\bwithout (?:food|eating|a meal) for (?:days|\d+ days|two days|three days|a week|weeks|many days)\b")
rule("neglect", r"\bno food (?:\w+ ){0,4}(?:for|in) (?:days|\d+ days|two days|three days|a week|weeks|many days)\b", ctx=True)
rule("neglect", r"\b(?:given|gets|get|got|receives?) nothing to eat\b", ctx=True)
rule("neglect", r"\bstarv\w*\b", ctx=True)
rule("neglect", r"\bstarv(?:e|es|ed|ing) " + PER + r"\b")
rule("neglect", r"\block\w* " + PER + r" (?:\w+ )?(?:in|inside|out|outside|up)\b", ctx=True)
rule("neglect", r"\b(?:refus\w*|wont|will not|do not|dont|does not|doesnt|did not|didnt|never) (?:to )?(?:take|took|taken|bring|send) "
                + PER + r" to (?:the )?(?:clinic|hospital|health cent(?:re|er)|doctor)\b", ctx=True)
rule("neglect", r"\bleft (?:\w+ ){0,3}alone\b", r"\b(?:" + DUR[3:-1] + r"|no food|no one|nobody)\b", ctx=True)
rule("neglect", r"\b(?:not|never|no one|nobody) (?:\w+ ){0,2}(?:caring|cares|care|looking after|looks after)\b", ctx=True)

# --- abandonment, chased from home
rule("abandonment", r"\babandon\w* (?:\w+ ){0,2}(?:child|children|baby|babies|twins|kids|son|daughter|boy|girl|family|him|her|them)\b", ctx=True)
rule("abandonment", r"\babandoned (?:by|at|with)\b", ctx=True)
rule("abandonment", r"\bchas\w* (?:\w+ ){0,5}(?:from|out of|away from|out from) (?:the |their |his |her |our |my )?(?:home|house|homes|"
                    r"houses|village|compound|yard)\b", ctx=True)
rule("abandonment", r"\b(?:father|mother|stepfather|stepmother|parents?|uncle|aunt|grandmother|grandfather|guardians?|relatives?|"
                    r"family|brother|sister|husband)\b(?: \w+){0,3} (?:chas\w*|kick\w* (?:\w+ )?out|throw\w* (?:\w+ )?out|threw (?:\w+ )?out|"
                    r"thrown (?:\w+ )?out|drove (?:\w+ )?out|sent (?:\w+ )?away|evict\w*)\b", ctx=True)
rule("abandonment", r"\b(?:kicked|thrown|threw|throws|throw|kicks|chased|chases|driven|drove|pushed|sent) (?:\w+ ){0,2}out of "
                    r"(?:the |their |his |her )?(?:house|home)\b", ctx=True)
rule("abandonment", r"\b(?:kicked|thrown|threw|chased) (?:him|her|them) (?:out|away)\b", ctx=True)
rule("abandonment", r"\b(?:no ?where|nowhere|no place|no home|no house|no family) to (?:sleep|stay|go|live)\b", ctx=True)
rule("abandonment", r"\b(?:lives?|living|stays?|staying|sleeps?|slept|sleeping) (?:\w+ ){0,1}(?:at|in|on|under) (?:the |a |an )?(?:streets?|market|"
                    r"bus (?:stop|station|stage)|station|bush|veranda|verandah|old car|graveyard)\b", ctx=True)
rule("abandonment", r"\b(?:sleep\w*|slept) (?:in|at) (?:the |a |our )?(?:class ?room|school) (?:at night|overnight)\b")
rule("abandonment", r"\b(?:sleeps?|slept|sleeping) outside\b|\bstreet (?:child|children|kid|kids|boy|boys|girl|girls)\b", ctx=True)
rule("abandonment", r"\b(?:runs?|ran|running) away\b|\bmissing\b", ctx=True)
rule("abandonment", r"\b(?:afraid|scared|fear\w*|frightened|refuses?|refused|doesnt want|does not want|dont want) (?:\w+ )?to (?:go )?"
                    r"(?:back )?home\b", ctx=True)
rule("abandonment", r"\b(?:told|tells|tell) (?:him|her|them) not to come (?:back|home)\b", ctx=True)
rule("abandonment", r"\b(?:left|leave|leaves|leaving) (?:the |his |her |their |a )?(?:child|children|kids|baby|babies|twins) "
                    r"(?:\w+ ){0,2}(?:alone|with nobody|with no one|behind)\b", ctx=True)
rule("abandonment", r"\b(?:left|went|gone|moved)\b(?: \w+){0,5} (?:and )?(?:left|leaving) (?:him|her|them|the \w+) "
                    r"(?:alone|with nobody|with no one|behind|with (?:a |the )?neighbours?)\b", ctx=True)
rule("abandonment", r"\bnever (?:came|come|coming) back\b", ctx=True)
rule("abandonment", r"\b(?:left|leave|leaves|leaving) (?:her|him|them) (?:alone|with nobody|with no one|on (?:her|his|their) own)\b",
     ctx=True)     # not 'left them behind' (a race)

# --- images and online abuse
rule("images", r"\b(?:naked|nude|nudes|undressed|without clothes|no clothes|topless|private parts|indecent|sexy|porn\w*|"
               r"sex)(?: \w+){0,2} " + IMG + r"\b")
# 'dirty' and 'blue' only right before the picture word ('blue movie'), never 'dirty water pictures' or 'blue sky pictures'
rule("images", r"\b(?:dirty|blue|adult|sex) (?:pictures?|photos?|pics|videos?|films?|movies?|clips?)\b", ctx=True)
rule("images", IMG + r" (?:\w+ ){0,3}(?:naked|nude|without (?:her |his |their |any )?clothes|undressed|in (?:her|his|their) "
               r"underwear|of (?:her|his|their) (?:private parts|private|body|bodies|breasts|chest|privates))\b")
rule("images", r"\bporn\w*\b|\bsext\w*\b")
rule("images", r"\b(?:film\w*|record\w*|video\w*|photograph\w*|take|takes|took|taking|watch\w*) (?:\w+ ){0,4}(?:while|when) "
               r"(?:s?he|they) (?:were |was |are |is )?(?:bath\w*|washing (?:themselves|their bodies|her body|his body)|undress\w*|"
               r"changing (?:their |her |his )?clothes|in the (?:toilet|bathroom|shower)|naked|showering)\b")
rule("images", r"\b(?:film\w*|record\w*|photograph\w*) (?:\w+ ){0,2}(?:in the|at the) (?:toilet|toilets|bathroom|shower|showers)\b", ctx=True)
rule("images", r"\b(?:show\w*) (?:\w+ ){0,3}(?:(?:porn\w*|sex|naked|nude)(?: \w+)?|dirty|blue|adult) (?:videos?|films?|movies?|"
               r"pictures?|photos?|images?)\b")
rule("images", ONLINE, r"\b(?:meet|meeting) (?:her|him|them|up|in person)\b|\bto meet\b|\bwants? to meet\b", ctx=True)
rule("images", ONLINE, r"\b(?:send|sends|sent|ask\w*|wants?|beg\w*) (?:\w+ ){0,4}" + IMG + r" of (?:her|his|their) (?:body|bodies|self|"
               r"selves|private|privates)\b")
rule("images", r"\bthreat\w* (?:\w+ ){0,3}(?:post|share|send|show|upload|spread)\b", ctx=True)
rule("images", r"\b(?:post|share|send|upload|spread)\w* (?:\w+ ){0,3}(?:naked|nude)\b")

# --- adult cover-up, pressure to keep a secret
TELL_NOT = (r"(?:to )?(?:not|never|no|dont|do not|doesnt|wont)(?: to)? (?:tell|report|say|inform|involve|go to|talk to|speak to|"
            r"reveal|disclose|mention|call)(?: it| this| that)?(?: to)? (?:anyone|anybody|no ?one|nobody|the police|police|social "
            r"welfare|welfare|the head\w*|the school|the teacher|teachers|her mother|his mother|her father|his father|mother|father|"
            r"anything|what|about|the case|the matter|the man)\b")
rule("secrecy", r"\b(?:beg\w*|plead\w*|ask\w*|told|tells|telling|tell|warn\w*|pressur\w*|brib\w*|paid|pay|pays|paying|threat\w*|"
                r"forc\w*|want\w*|said|says|instruct\w*|urg\w*|insist\w*|so)\b(?: \w+){0,4} " + TELL_NOT, ctx=True)
# pressure not to report to the authorities or about a person: no child word needed ('the parents begged me not to
# report the man'); 'learners must not tell anyone the answers' has no pressure verb and no such object
rule("secrecy", r"\b(?:beg\w*|plead\w*|ask\w*|told|tells|warn\w*|pressur\w*|brib\w*|paid|pay|pays|paying|threat\w*|forc\w*|"
                r"instruct\w*|urg\w*|insist\w*)\b(?: \w+){0,4} (?:to )?(?:not|never|dont|do not)(?: to)? (?:report|tell|inform|go to|call|involve)"
                r"(?: it| this| that)?(?: to)? (?:the police|police|social welfare|welfare|the man|the case|what (?:he|she|they|happened))\b")
rule("secrecy", r"\b(?:its|it is|is|was) (?:a |our |their |his )?secret\b|\bkeep (?:it |this |that )?(?:a )?secret\b", ctx=True)
rule("secrecy", r"\b(?:keep|kept|keeping|stay|be) (?:quiet|silent) about\b", ctx=True)
rule("secrecy", r"\b(?:settle\w*|solv\w*|resolv\w*|sort\w*|handl\w*|finish\w*|deal\w*)\b(?: \w+){0,4} (?:at home|in the family|within "
                r"the family|privately|quietly|traditionally|the traditional way|between the families|out of court)\b",
     r"\b(?:case|matter|issue|police|report\w*|it)\b", ctx=True)
rule("secrecy", r"\b(?:withdr\w*|drop\w*|clos\w*) (?:the )?(?:case|charges?|complaint|report)\b", ctx=True)
rule("secrecy", r"\b(?:not|never|dont|do not|without|no need to) (?:to )?(?:go to|involve|call|inform|tell) (?:the )?police\b", ctx=True)
rule("secrecy", r"\b(?:paid|pay|pays|paying|compensat\w*|gave|given) (?:\w+ ){0,3}(?:a cow|cows|cattle|goats?|money|k\d+)\b",
     r"\b(?:case|police|quiet|secret|silent|withdr\w*|settle\w*|(?:not|dont|never) (?:to )?report\w*)\b", ctx=True)
rule("secrecy", r"\bwhat (?:the |her |his |their |my )?(?:\w+ )?(?:stepfather|stepmother|father|uncle|man|teacher|deputy|boyfriend|"
                r"brother|cousin|neighbour|he|they|someone|headmaster|guard) (?:did|does|has done|was doing|is doing|did to|does to)\b",
     r"\b(?:to (?:her|him|them|the \w+)|quiet|secret|report|tell|police|say)\b", ctx=True)

# --- child labour
rule("labour", r"\bchild labou?r\b|\btraffick\w*\b")
rule("labour", r"\b(?:works?|working|worked|employed|employs|employ|made to work|forced to work|sent to work)\b(?: \w+){0,4} "
               r"(?:as (?:a |an )?)?(?:maid|house ?(?:girl|boy|help|helper|worker|servant)|servant|herd ?boy|cattle herder|herder|"
               r"labourer|laborer|vendor|bar ?(?:maid|girl|boy|tender)|(?:in|at) (?:a |the )?(?:mines?|quarry|bar|shebeen|tavern|"
               r"brick\w*|tobacco|cotton|plantation|fields?|farms?|construction|market|bus station))\b", ctx=True)
rule("labour", r"\b(?:taken|took|take|withdrawn|withdrew|removed|pulled|stopped|stop|kept|keeps|keep)\b(?: \w+){0,3} (?:out of |from |"
               r"away from |at )?(?:school|class|home)\b(?: \w+){0,3} (?:to work|to herd|to sell|to look after|to care for|to help "
               r"(?:at|in|on)|for work|to fish|to mine|to do (?:house ?work|chores))\b", ctx=True)
rule("labour", r"\binstead of (?:going to |coming to |attending |being in )?(?:school|class)\b",
     r"\b(?:work\w*|sell\w*|herd\w*|maid|vend\w*|kept at home|hawk\w*|fish\w*|mine|mining|quarry|bar)\b", ctx=True)
rule("labour", r"\b(?:herd\w*|sell\w*|vend\w*|hawk\w*|work\w*|fish\w*|crush\w*|break\w*)\b(?: \w+){0,5} (?:all day|the whole day|every "
               r"day|until late|until night|till night|at night|every school day|during school|until midnight)\b", ctx=True)
rule("labour", r"\b(?:bar|tavern|shebeen|beer|kachasu|alcohol|quarry|mines?|mining)\b", r"\b(?:work\w*|sell\w*|employ\w*)\b", ctx=True)
rule("labour", r"\b(?:crush\w*|break\w*) stones\b", ctx=True)
rule("labour", r"\b(?:sends?|sent|makes?|made|forces?|forced) (?:him|her|them|the \w+) (?:\w+ )?to (?:herd|work|sell|beg|fish|mine)\b", ctx=True)
rule("labour", r"\bbeg\w* (?:on|in) the streets?\b", ctx=True)
rule("labour", r"\b(?:stopped coming|stop coming|does not come|doesnt come|never comes|misses school|absent)\b(?: \w+){0,6} "
               r"(?:herd\w*|work\w*|sell\w*|maid|farm\w*|field)\b", ctx=True)

# --- self-harm
rule("self_harm", r"\b(?:self ?harm\w*|suicid\w*|kulijaya|lijaya)\b")
# 'kill myself' is a learner's own words; 'cut / hurt yourself' is every scissors-safety lesson, so only him / her / them
rule("self_harm", r"\b(?:kill\w*|hang\w*|harm\w*|poison\w*) (?:him|her|them|my|your|our)sel(?:f|ves)\b|"
                  r"\b(?:cut\w*|hurt\w*|burn\w*) (?:him|her|them)sel(?:f|ves)\b")
rule("self_harm", r"\b(?:end|take|took|taking|ends|ended) (?:his|her|their|my) (?:own )?life\b")
rule("self_harm", r"\b(?:want|wants|wanted|wanting|wish|wishes|wished|plans?|planned|planning|tri(?:ed|es)|trying|threat\w*) to die\b")
rule("self_harm", r"\b(?:going|about) to die\b", ctx=True)    # 'the plant is going to die' has no child in it
rule("self_harm", r"\b(?:dont|doesnt|do not|does not|no longer|didnt|did not) (?:want|wish) to (?:live|be alive)\b|\bno (?:reason|point) "
                  r"(?:in |to )?(?:living|live)\b|\b(?:better off|be better) dead\b|\bwish(?:es|ed)? (?:s?he|they|i) (?:was|were) dead\b")
# a person's intent ('children will drink poison if bottles are not labelled' is a safety lesson: no pronoun subject)
rule("self_harm", r"\b(?:s?he|they|i)\s+(?:will|wants? to|wanted to|is going to|was going to|tried to|tries to|try to|threat\w* to|"
                  r"plans? to|planning to|said s?he will)\s+(?:drink|swallow|take|eat)\s+(?:\w+ ){0,2}(?:poison|rat poison|pesticide|"
                  r"insecticide)\b")
# household liquids and pills (a poison-safety lesson says 'Nchimunya wants to drink paraffin'): only with an intent to die
rule("self_harm", r"\b(?:drink|drank|swallow\w*|take|took|eat|ate)\s+(?:\w+ ){0,2}(?:bleach|jik|paraffin|acid|chemicals?|pills|tablets)\b",
     r"\b(?:to die|kill (?:him|her|them)sel(?:f|ves)|end (?:his|her|their) life|because of (?:problems|what))\b")
rule("self_harm", r"\b(?:drank|drunk|swallowed|took|ate) (?:\w+ ){0,2}(?:poison|pesticide|insecticide|bleach|jik|paraffin|acid|(?:many|"
                  r"lots of|all the|a lot of|too many) (?:pills|tablets))\b", ctx=True)
rule("self_harm", r"\b(?:cuts?|cutting|burn\w*|scratch\w*) (?:\w+ ){0,2}(?:his|her|their|my) (?:own )?(?:arms?|wrists?|thighs?|legs?|skin|body)\b",
     r"\b(?:on purpose|deliberately|himself|herself|themselves|razor|blade|matches|did it|does it|by (?:him|her)self)\b", ctx=True)
rule("self_harm", r"\bcut\w* (?:\w+ ){0,3}wrists?\b", ctx=True)
rule("self_harm", r"\b(?:goodbye|farewell|suicide) (?:letter|note|message)\b|\b(?:letter|note|message) (?:\w+ ){0,2}(?:saying )?goodbye\b", ctx=True)

# --- danger (general)
rule("danger", r"\bin danger\b|\b(?:not|isnt|is not) safe at home\b|\bunsafe at home\b")
rule("danger", r"\b(?:afraid|scared|frightened|terrified|fears?) of (?:her|his|their) (?:\w+ )?(?:father|mother|stepfather|stepmother|"
               r"uncle|parents|guardian|guardians|teacher|aunt|brother|grandfather)\b", ctx=True)


# parity probes for the JS ports (export_netlify parity_probes.json, web/tests/safeguard_parity.mjs): written by me as
# regression cases, one or two per category and the hard negatives; NOT bake-off rows
PROBES = [
    "a learner told me her uncle beats her", "the teacher hit a child", "A girl in grade 3 told me her stepfather hurts her at night.",
    "Mwana umwi wakandaambila kuti bausyi balamuuma lyoonse", "My pupils hurt their fingers cutting paper",
    "a pupil in gr 4 says her mums boyfrend sleeps with her", "our deputy beatn the lerners wit a hosepipe til they bled",
    "gal of 12 to be maried after harvest", "the man at the bar gives her k20 for sex", "learner sleps at the bus stop, uncle chased him",
    "her granny says i must keep quiet about what happened to her", "a teacher takes naked pics of girls on his fone",
    "the girl is given no food for days when she is late", "boy herds cattle every day instead of school",
    "she wrote that she wants to die", "money lesson grade 2", "forces and pushes for grade 3", "pictures of plants",
    "marriage customs in social studies", "food groups", "beat the drum", "how to discipline learners without beating them",
    "the children burnt the rubbish", "find the missing numbers", "the dog chased the cat out of the house",
    "she says her stepfather’s friend touches her privates", "we can’t beat the drum today, the learners are tired",
]


def detect(text):
    """None, or {'category', 'hit', 'why', 'form'} for the first rule that catches. Errs on the side of catching."""
    if not text or not str(text).strip():
        return None
    forms = []
    lt = light(text)
    nt = normalise(text)
    for name, t in (("light", lt), ("normalised", nt)):
        if name == "normalised" and t == lt:
            continue
        forms.append((name, t, _blank(t)))
    for name, t, b in forms:
        ctx = bool(CTX.search(b))
        for cat, rxs, need_ctx in R:
            if need_ctx and not ctx:
                continue
            m0 = None
            for rx in rxs:
                m = rx.search(b)
                if not m:
                    break
                m0 = m0 or m
            else:
                hit = m0.group(0).strip()
                why = f"{cat}: '{hit}'" + (" with a child in the message" if need_ctx else "")
                if name == "normalised":
                    why += " (spelling normalised)"
                return {"category": cat, "hit": hit, "why": why, "form": name}
    return None


def safeguarding(text):
    """'' or the reason (category and the words that caught it): the h3_front.safeguarding contract"""
    d = detect(text)
    return d["why"] if d else ""


# ------------------------------------------------------------------------------------------------ export (JS port)
def _js(rx):
    src = rx.pattern
    for bad in ("(?P<", "(?P=", r"\Z", r"\A", "(?x", "(?s", "(?m", "(?#", "(?i"):
        if bad in src:
            raise ValueError(f"{bad} in {src[:80]}")
    return {"src": src, "flags": "i"}


def export():
    """everything a JavaScript port needs to give the same answer (export_netlify / web api; site/assistant search.js and
    netlify/functions/assistant.mjs read it): the phrase rewrites, the SMS map, the KEY words, the protected English
    words near them, the context and benign regexes, and the rules in order"""
    return {"version": "2026-10-24.1", "keys_hash": keys_hash(),
            "phrases": [{"src": p, "rep": r.replace("\\1", "$1")} for p, r in PHRASES], "sms": SMS, "keys": list(KEYS),
            "near": sorted(near_words() - {"__nofuzzy__"}), "ctx": _js(CTX), "benign": [_js(x) for x in BENIGN_RX],
            "rules": [{"cat": c, "all": [_js(x) for x in rxs], "ctx": n} for c, rxs, n in R]}


def legacy_regexes():
    """(strong, child, weak) unions for an old three-regex consumer: strong = every single-regex rule that needs no
    context, child = CTX, weak = every single-regex context rule. Combination rules and normalisation are not in them."""
    strong = "|".join(f"(?:{rxs[0].pattern})" for _, rxs, n in R if len(rxs) == 1 and not n)
    weak = "|".join(f"(?:{rxs[0].pattern})" for _, rxs, n in R if len(rxs) == 1 and n)
    return re.compile(strong, re.I), re.compile(CTX.pattern, re.I), re.compile(weak, re.I)


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-near"]:
        o = build_near()
        print(f"{len(o['words'])} protected words -> {NEAR_PATH}")
    else:
        for a in sys.argv[1:]:
            print(repr(a), "->", repr(normalise(a)), "->", detect(a))
