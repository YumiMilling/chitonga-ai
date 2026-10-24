"""Messy teacher English for the reader bake-off: calibration on real Zambian messages, then a deterministic noise
generator (24 Oct 2026, the Jesuit).

Chris, 24 Oct 2026: "english only, broken english, typos etc." The English test must count messy teacher English, not
only clean prose. The coordinator added: calibrate on how our Zambian contacts actually type (the Zamai line), widen
the catalogue (broken sentences, wrong punctuation, autocorrect damage, sound-spelling, case, spacing, code-switching,
Zambian idioms, grade forms, several questions, greeting then request), and two typing PROFILES from Chris's own
words: "look at my own mess, how I type, which might be extreme, on the laptop; better when I use autocorrect on the
phone." Profile A (fast laptop typist, no autocorrect) is the STRONG level, profile B (phone with autocorrect) the MILD.

PRIVACY. `calibrate` reads zamai_messages READ-ONLY (the desk watcher's read path, responder_common.sb), inbound only,
only from the teachers and reviewers of the desk (by speaker code; never the tester's number, never business
contacts or any other WhatsApp table), only messages the served language
detector reads as English. It keeps COUNTS and pattern names only: no message, name or number is written anywhere
and nothing from a message is copied into the test set. The generator's word lists are generic (SMS forms, QWERTY
neighbours, a curated autocorrect map); a code-switch token is used only if it occurs in the real sample AND is a
greeting or an address form (no Tonga content words: Chris, 24 Oct: "this test shouldn't have anything to do with
real Tonga").

    python -X utf8 tools/chitonga/llm/english_noise.py calibrate       # -> noise/real_stats.json (counts only)
    python -X utf8 tools/chitonga/llm/english_noise.py measure         # typo / broken share in testset_v2_en
    python -X utf8 tools/chitonga/llm/english_noise.py build           # testset_v2_noisy.jsonl + the combined set
Limit, stated in every report: this approximates how teachers type to an assistant; the real check is the first real
teachers' held-out requests (the flywheel's first 40 per teacher).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
RUN = os.path.join(CT, "runs", "2026-10-24_reader_bakeoff")
OUT = os.path.join(RUN, "noise")
SEED = 20261024
N_NOISY = 300

# desk teachers and reviewers (internal/chitonga_train/desk/teachers.json, balances.tsv, the watcher's list): numbers
# are read from those files at run time, never written here
DESK_CODES = ("T01", "T09", "T12")
EXTRA_NAMES = ()   # PUBLIC COPY: the working copy lists further reviewers by name; read from private files only

SMS = {"please": ["pls", "plz", "pliz"], "you": ["u"], "your": ["ur"], "for": ["4"], "what": ["wat", "wht"],
       "how": ["hw"], "because": ["coz", "bcoz", "bcz"], "thanks": ["thnx", "thx", "tnx"], "thank you": ["thank u", "tnx"],
       "today": ["2day"], "tomorrow": ["tmrw", "2moro"], "before": ["b4"], "are": ["r"], "message": ["msg"],
       "good": ["gud"], "know": ["knw"], "with": ["wit", "wth"], "about": ["abt"], "teacher": ["tr"],
       "people": ["ppl"], "something": ["smthng"], "okay": ["ok", "k"], "and": ["n", "&"]}
SMS_TOKENS = {v for vs in SMS.values() for v in vs} | {"gr", "std", "pple", "nyt", "gd", "hlp", "lol", "asap", "bt", "ok"}
APOS = {"don't": "dont", "can't": "cant", "won't": "wont", "i'm": "im", "didn't": "didnt", "isn't": "isnt",
        "doesn't": "doesnt", "i've": "ive", "you're": "youre", "that's": "thats", "what's": "whats", "let's": "lets",
        "it's": "its", "teacher's": "teachers", "pupil's": "pupils", "learners'": "learners", "aren't": "arent",
        "couldn't": "couldnt", "wouldn't": "wouldnt", "there's": "theres", "i'll": "ill"}
NOAPOS = set(APOS.values()) - {"its", "ill", "teachers", "pupils", "learners", "lets"}
# autocorrect damage: a REAL word replaces the intended one (profile B); some change the meaning
AUTOCORRECT = {"borrowing": ["borrow in", "browsing"], "pupils": ["pupil's", "pupil"], "their": ["there"],
               "your": ["you're"], "of": ["off"], "then": ["than"], "form": ["from"], "lesson": ["lessen"],
               "sum": ["some"], "add": ["and"], "learners": ["leaners", "learner's"], "subtract": ["subtracts"],
               "teach": ["reach", "teach a"], "where": ["were"], "weather": ["whether"], "write": ["right"],
               "four": ["for"], "two": ["too"], "class": ["clash"], "fractions": ["fraction's"], "plants": ["planets"],
               "count": ["court"], "letters": ["litters"], "sounds": ["sound's"], "times": ["tines"],
               "shapes": ["shades"], "grade": ["grace", "grades"], "maths": ["math's", "moths"]}
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
SOUND = [("l", "r"), ("r", "l"), ("th", "t"), ("ch", "sh"), ("children", "chidren"), ("think", "tink"),
         ("thank", "tank"), ("v", "f")]
GREET_OPEN = ["Good morning sir,", "Good morning madam,", "Good afternoon sir,", "Hello sir,", "Morning,", "Good evening,"]
POLITE = ["kindly help me on", "kindly assist with", "please help me with", "I need help on", "let me ask about",
          "may you help me with"]
GRADE_FORMS = ["grade {n}", "gr {n}", "gr{n}", "G{n}", "std {n}", "grade {w}", "std {w}", "{o} grade"]
NUMW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th"}
GREET_WORDS = {"mwabuka", "mwabonwa", "mwalibizya", "mwami", "ba", "ma", "muli", "bwanji", "twalumba", "ndalumba",
               "mwalumba", "mulibuti", "mwabuka buti"}
ZAMBIAN = {"kindly": r"\bkindly\b", "let me ask": r"\blet me ask\b", "is it?": r"\bis it\s*\?",
           "failing to": r"\bfail(ing|s)? to\b", "help me on": r"\bhelp me on\b", "sir/madam": r"\b(sir|madam|mam)\b",
           "how to (no subject)": r"(^|[.?!]\s*)how to\b", "am (for I am)": r"(^|\s)am\s+(fine|good|well|ok|asking|a teacher|here)",
           "may you": r"\bmay you\b", "on how to": r"\bon how to\b"}


def wre(w):
    return re.compile(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", re.I)


# ------------------------------------------------------------------------------------------------ text features
TOK = re.compile(r"[A-Za-z][A-Za-z']*")


def _spell():
    from spellchecker import SpellChecker  # noqa: PLC0415
    return SpellChecker()


def _tonga_sets():
    d = json.load(open(os.path.join(CT, "vocab", "attest_cache.json"), encoding="utf-8"))
    toi = {w for w, n in d.get("toi", {}).items() if n >= 3}
    nya = {w for w, n in d.get("nya", {}).items() if n >= 50}
    return toi, nya


def features(text, sp, toi, nya):
    """one message -> counts of every catalogue pattern (no text kept)"""
    t = text or ""
    toks = TOK.findall(t)
    low = [w.lower() for w in toks]
    letters = [c for c in t if c.isalpha()]
    f = collections.Counter()
    f["words"] = len(toks)
    f["no_initial_capital"] = int(bool(letters) and letters[0].islower())
    f["all_lowercase"] = int(bool(letters) and not any(c.isupper() for c in letters))
    f["all_caps"] = int(len(letters) >= 10 and sum(c.isupper() for c in letters) / len(letters) > 0.7)
    f["no_final_punct"] = int(bool(t.strip()) and t.strip()[-1] not in ".?!")
    f["multi_question"] = int(t.count("?") >= 2 or len(re.findall(r"\b(how|what|which|when|why|can)\b[^.?!]*\?", t, re.I)) >= 2)
    f["repeated_qmark_or_dots"] = int(bool(re.search(r"\?\?|\.\.\.|!!", t)))
    f["space_before_punct"] = int(bool(re.search(r"\w\s+[,.?!]", t)))
    f["missing_space_after_punct"] = int(bool(re.search(r"[a-z][,.?!][a-zA-Z]", t)))
    f["double_space"] = int("  " in t.strip())
    f["doubled_word"] = int(any(low[i] == low[i + 1] and len(low[i]) > 1 for i in range(len(low) - 1)))
    f["greeting_then_newline"] = int("\n" in t.strip() and bool(re.match(r"\s*(good|hello|hi|morning|mwa|dear)\b", t, re.I)))
    f["greeting_opener"] = int(bool(re.match(r"\s*(good (morning|afternoon|evening)|hello|hi|morning|dear|greetings)\b", t, re.I)))
    f["comma_splice"] = int(t.count(",") >= 2 and "." not in t.strip()[:-1])
    f["missing_apostrophe"] = sum(1 for w in low if w in NOAPOS)
    f["sms_tokens"] = sum(1 for w in low if w in SMS_TOKENS)
    for w in low:
        if w in SMS_TOKENS:
            f["sms:" + w] += 1
    gr = re.findall(r"\b(grade|gr|std|standard|g)\s*\.?\s*(\d|one|two|three|four|five|six|seven)\b", t, re.I)
    for g, _ in gr:
        f["gradeform:" + g.lower()] += 1
    f["gradeform:ordinal"] += len(re.findall(r"\b\d(st|nd|rd|th) grade\b", t, re.I))
    cs = [w for w in low if (w in GREET_WORDS) or (w in toi and w not in sp and len(w) > 3) or (w in nya and w not in sp and w not in toi and len(w) > 3)]
    f["codeswitch_tokens"] = len(cs)
    f["codeswitch_greeting"] = sum(1 for w in low if w in GREET_WORDS)
    for name, rx in ZAMBIAN.items():
        if re.search(rx, t, re.I):
            f["zam:" + name] += 1
    # typos: not an English word, not Tonga / Nyanja, not an SMS form, not a capitalised name, not a no-apostrophe form
    unk = []
    for w, lw in zip(toks, low):
        if lw in SMS_TOKENS or lw in NOAPOS or lw in toi or lw in nya or lw in GREET_WORDS or len(lw) < 3:
            continue
        if w[0].isupper() and w != toks[0] and not w.isupper():
            continue
        if lw.replace("'", "") in sp or lw in sp:
            continue
        unk.append(lw)
    f["typos"] = len(unk)
    for lw in unk:
        c = sp.correction(lw) or ""
        if not c:
            continue
        if any(len(a) == len(b) == 1 and lw == c.replace(a, b, 1) for a, b in (("l", "r"), ("r", "l"))):
            f["sound:l_r"] += 1
        elif len(c) == len(lw) + 1 and c.startswith(lw):
            f["sound:dropped_final"] += 1
        elif lw == c.replace("th", "t", 1) and "th" in c:
            f["sound:th_t"] += 1
        elif sorted(lw) == sorted(c) and lw != c:
            f["typo:transposition"] += 1
        elif len(lw) == len(c) - 1:
            f["typo:dropped_letter"] += 1
        elif len(lw) == len(c) + 1:
            f["typo:extra_letter"] += 1
        elif len(lw) == len(c):
            f["typo:substitution"] += 1
    f["has_typo"] = int(bool(unk))
    f["fragment_question"] = int(t.strip().endswith("?") and len(toks) <= 5 and not re.search(
        r"\b(is|are|do|does|can|have|has|was|were|will|teach|explain|help|make|need|want|get|give)\b", t, re.I))
    return f


def summarise(feats):
    n = len(feats)
    words = [f["words"] for f in feats]
    tot_words = max(1, sum(words))
    S = {"messages": n, "words_total": sum(words),
         "length_words": {"mean": round(statistics.mean(words), 1) if words else None,
                          "median": statistics.median(words) if words else None,
                          "p90": sorted(words)[int(0.9 * (n - 1))] if words else None}}
    msg_flags = ["no_initial_capital", "all_lowercase", "all_caps", "no_final_punct", "multi_question",
                 "repeated_qmark_or_dots", "space_before_punct", "missing_space_after_punct", "double_space",
                 "doubled_word", "greeting_then_newline", "greeting_opener", "comma_splice", "has_typo",
                 "fragment_question"]
    S["share_of_messages"] = {k: round(sum(f[k] for f in feats) / max(1, n), 3) for k in msg_flags}
    S["share_of_messages"]["any_sms_token"] = round(sum(1 for f in feats if f["sms_tokens"]) / max(1, n), 3)
    S["share_of_messages"]["any_missing_apostrophe"] = round(sum(1 for f in feats if f["missing_apostrophe"]) / max(1, n), 3)
    S["share_of_messages"]["any_codeswitch"] = round(sum(1 for f in feats if f["codeswitch_tokens"]) / max(1, n), 3)
    S["share_of_messages"]["codeswitch_greeting"] = round(sum(1 for f in feats if f["codeswitch_greeting"]) / max(1, n), 3)
    S["per_word"] = {"typos": round(sum(f["typos"] for f in feats) / tot_words, 4),
                     "sms_tokens": round(sum(f["sms_tokens"] for f in feats) / tot_words, 4),
                     "missing_apostrophe": round(sum(f["missing_apostrophe"] for f in feats) / tot_words, 4)}
    agg = collections.Counter()
    for f in feats:
        for k, v in f.items():
            if ":" in k:
                agg[k] += v
    S["sms_forms"] = {k[4:]: v for k, v in agg.most_common() if k.startswith("sms:")}
    S["grade_forms"] = {k[10:]: v for k, v in agg.most_common() if k.startswith("gradeform:") and v}
    S["zambian_constructions_share"] = {k[4:]: round(v / max(1, n), 3) for k, v in agg.most_common() if k.startswith("zam:")}
    S["typo_kinds"] = {k: v for k, v in agg.most_common() if k.startswith(("typo:", "sound:"))}
    return S


# ------------------------------------------------------------------------------------------------ calibrate
def desk_numbers():
    """the teachers' and reviewers' numbers from the desk's own files (held in memory only)"""
    nums = set()
    T = json.load(open(os.path.join(CT, "desk", "teachers.json"), encoding="utf-8"))
    for code in DESK_CODES:
        n = str((T.get(code) or {}).get("whatsapp") or "").replace("+", "")
        if n:
            nums.add(n)
    watch = os.environ.get("DESK_WATCH_FILE", "")   # PUBLIC COPY: a private watcher file in the working copy
    src = open(watch, encoding="utf-8").read() if os.path.exists(watch) else ""
    for m in re.finditer(r'"(260\d{9})": "([^"]+)"', src):
        if any(x.split()[0] in m.group(2) for x in EXTRA_NAMES):
            nums.add(m.group(1))
    bal = open(os.path.join(CT, "desk", "balances.tsv"), encoding="utf-8").read()
    for line in bal.splitlines():
        if any(x and x in line for x in EXTRA_NAMES):
            for m in re.finditer(r"(?<!\d)(0?9[5-7]\d{7}|2609[5-7]\d{7})(?!\d)", line):
                v = m.group(1)
                nums.add(v if v.startswith("260") else "260" + v.lstrip("0"))
    return nums


def cmd_calibrate(a):
    sys.path.insert(0, r"<private-repo-not-included>")
    from responder_common import sb  # noqa: PLC0415  (read-only GET on zamai_messages)
    sys.path.insert(0, HERE)
    import translator_service as TS  # noqa: PLC0415
    nums = desk_numbers()
    rows, off = [], 0
    while True:
        r = sb(f"zamai_messages?direction=eq.in&select=counterparty,body&order=created_at.asc&limit=1000&offset={off}")
        rows += r
        if len(r) < 1000:
            break
        off += 1000
    sp, (toi, nya) = _spell(), _tonga_sets()
    feats, skipped, greet_seen = [], collections.Counter(), set()
    for r in rows:
        if r["counterparty"] not in nums:
            skipped["not a desk teacher / reviewer"] += 1
            continue
        b = (r.get("body") or "").strip()
        if not b or b.startswith("*") or len(TOK.findall(b)) < 3:
            skipped["empty / code / under 3 words"] += 1
            continue
        if TS.detect_direction(b) != "en>toi":
            skipped["read as Tonga"] += 1
            continue
        feats.append(features(b, sp, toi, nya))
        for g in GREET_WORDS:
            if re.search(r"(?<![a-z])" + re.escape(g) + r"(?![a-z])", b.lower()):
                greet_seen.add(g)
    S = summarise(feats)
    S["_greet_seen"] = sorted(greet_seen)     # generic greeting / address words only (no content, no names)
    S["senders"] = len(nums)
    S["skipped"] = dict(skipped)
    S["source"] = ("zamai_messages, inbound, desk teachers and reviewers only, read as English by the served detector; "
                   "counts only, no text kept (read " + __import__("datetime").datetime.now().isoformat(timespec="minutes") + ")")
    os.makedirs(OUT, exist_ok=True)
    json.dump(S, open(os.path.join(OUT, "real_stats.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print(json.dumps(S, indent=1, ensure_ascii=False)[:3000])


def cmd_measure(a):
    sp, (toi, nya) = _spell(), _tonga_sets()
    rows = [json.loads(l) for l in open(os.path.join(RUN, "testset_v2_en.jsonl"), encoding="utf-8") if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slice"]].append(features(r["text"], sp, toi, nya))
    out = {k: summarise(v) for k, v in by.items()}
    out["all"] = summarise([f for v in by.values() for f in v])
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "testset_en_stats.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    for k, s in out.items():
        m = s["share_of_messages"]
        print(f"{k}: {s['messages']} rows; has typo {m['has_typo']}; no initial capital {m['no_initial_capital']}; "
              f"no final punct {m['no_final_punct']}; sms {m['any_sms_token']}; typos/word {s['per_word']['typos']}")


# ------------------------------------------------------------------------------------------------ generator
def _adjacent(c):
    for row in QWERTY:
        i = row.find(c.lower())
        if i >= 0:
            opts = ([row[i - 1]] if i > 0 else []) + ([row[i + 1]] if i < len(row) - 1 else [])
            return opts
    return []


def typo_word(w, rng, kinds):
    """one raw typo in a word of 3+ letters: adjacent key, transposition, dropped, doubled / inserted, cross-syllable"""
    if len(w) < 3 or not w.isalpha():
        return w
    k = rng.choice(kinds)
    i = rng.randrange(1, len(w) - 1)
    if k == "adjacent":
        o = _adjacent(w[i])
        return w[:i] + rng.choice(o) + w[i + 1:] if o else w
    if k == "transpose":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if k == "drop":
        return w[:i] + w[i + 1:]
    if k == "double":
        return w[:i] + w[i] + w[i:]
    if k == "insert":
        o = _adjacent(w[i])
        return w[:i] + (rng.choice(o) if o else w[i]) + w[i:]
    if k == "swap_far" and len(w) >= 6:
        j = min(len(w) - 1, i + 2)
        lst = list(w)
        lst[i], lst[j] = lst[j], lst[i]
        return "".join(lst)
    return w


def grade_variant(text, rng, p):
    def rep(m):
        n = int(m.group(2))
        if rng.random() >= p:
            return m.group(0)
        f = rng.choice(GRADE_FORMS)
        return f.format(n=n, w=NUMW.get(n, n), o=ORD.get(n, n))
    return re.sub(r"\b(Grade|grade)\s+([1-7])\b", rep, text)


def noisy(text, level, rng, P):
    """level 'mild' (profile B, phone + autocorrect) or 'strong' (profile A, fast laptop typist, no autocorrect); P =
    the calibrated parameters. Returns (text, applied pattern names)."""
    t, applied = text, []
    words = t.split(" ")

    def on(name, p):
        if rng.random() < p:
            applied.append(name)
            return True
        return False
    # Zambian layer (calibrated rates; both levels)
    def sub(name, p, rx, repl, flags=0):   # a pattern counts as applied only when it changed the text
        nonlocal t
        if rng.random() < p:
            new = re.sub(rx, repl, t, count=1, flags=flags)
            if new != t:
                t = new
                applied.append(name)
    sub("zam:failing_to", P["zam_failing"], r"\b(cannot|can't|can not|do not know how to|don't know how to)\b", "are failing to")
    sub("zam:how_to", P["zam_how_to"], r"^(How (do|can|should) I|How do we|How can we)\s+", "how to ", re.I)
    sub("zam:drop_article", P["drop_article"], r"\b(the|a|an) (?=\w)", "")
    if on("zam:polite_opener", P["polite_opener"]):
        t = f"{rng.choice(GREET_OPEN)} {rng.choice(POLITE)} " + t[0].lower() + t[1:]
    elif on("greeting_then_newline", P["greeting_newline"]):
        t = rng.choice(GREET_OPEN).rstrip(",") + "\n" + t
    if P["codeswitch_tokens"] and on("codeswitch_greeting", P["codeswitch"]):
        t = rng.choice(P["codeswitch_tokens"]).capitalize() + " " + t
    t = grade_variant(t, rng, P["grade_form"])
    # SMS forms (per word)
    for full, forms in SMS.items():
        rx = wre(full)
        if rx.search(t) and rng.random() < P["sms_word"]:
            t = rx.sub(rng.choice(forms), t, count=1)
            applied.append("sms")
    # apostrophes dropped
    for full, bare in APOS.items():
        if wre(full).search(t) and rng.random() < P["no_apostrophe"]:
            t = wre(full).sub(bare, t)
            applied.append("missing_apostrophe")
    if level == "mild":       # profile B: autocorrect damage, few raw typos, capitals kept after full stops
        cands = [w for w in AUTOCORRECT if wre(w).search(t)]
        if cands and on("autocorrect", P["autocorrect"]):
            w = rng.choice(cands)
            t = wre(w).sub(rng.choice(AUTOCORRECT[w]), t, count=1)
        ws = t.split(" ")
        for i, w in enumerate(ws):
            if rng.random() < P["typo_word_mild"]:
                nw = typo_word(w, rng, ["adjacent", "drop"])
                if nw != w:
                    ws[i] = nw
                    applied.append("typo")
        t = " ".join(ws)
    else:                     # profile A: fast laptop typist, no autocorrect
        ws = t.split(" ")
        for i, w in enumerate(ws):
            if rng.random() < P["typo_word_strong"]:
                nw = typo_word(w, rng, ["adjacent", "transpose", "drop", "double", "insert", "swap_far"])
                if nw != w:
                    ws[i] = nw
                    applied.append("typo")
        t = " ".join(ws)
        for s_, d in SOUND:
            if s_ in t and rng.random() < P["sound"] / len(SOUND):
                t = t.replace(s_, d, 1)
                applied.append("sound_spelling")
        if on("all_lowercase", P["all_lower_strong"]):
            t = t.lower()
        if on("comma_for_fullstop", P["comma_splice"]):
            t = re.sub(r"\.\s+(?=\w)", ", ", t)
        if on("run_together", P["run_together"]):
            ws = t.split(" ")
            if len(ws) > 3:
                i = rng.randrange(len(ws) - 1)
                ws[i:i + 2] = [ws[i] + ws[i + 1]]
                t = " ".join(ws)
    # punctuation, case and spacing (calibrated; strong doubles each rate)
    k = 1.0 if level == "mild" else 2.0
    if on("no_initial_capital", min(1, P["no_initial_capital"] * k)) and t[:1].isupper():
        t = t[0].lower() + t[1:]
    if on("no_final_punct", min(1, P["no_final_punct"] * (1 if level == "mild" else 1.3))):
        t = t.rstrip(".?! ")
    elif on("repeated_qmark", P["repeated_q"] * k):
        t = t.rstrip(".?! ") + rng.choice(["???", "??", "..."])
    if on("space_before_punct", P["space_before_punct"] * k):
        t = re.sub(r"(\w)([,?])", r"\1 \2", t, count=1)
    if on("doubled_word", P["doubled_word"] * k):
        ws = t.split(" ")
        i = rng.randrange(len(ws))
        ws.insert(i, ws[i])
        t = " ".join(ws)
    if on("double_space", P["double_space"] * k):
        t = t.replace(" ", "  ", 1)
    if on("all_caps", P["all_caps"]):
        t = t.upper()
    return t, applied


def params(real):
    """generator parameters from the real sample; each with a floor so the pattern exists, the floors stated"""
    m, w = real["share_of_messages"], real["per_word"]
    z = real.get("zambian_constructions_share", {})
    fl = lambda v, floor: max(v, floor)   # noqa: E731
    P = {"no_initial_capital": m["no_initial_capital"], "no_final_punct": m["no_final_punct"],
         "repeated_q": fl(m["repeated_qmark_or_dots"], 0.03), "space_before_punct": fl(m["space_before_punct"], 0.03),
         "doubled_word": fl(m["doubled_word"], 0.02), "double_space": fl(m["double_space"], 0.03),
         "all_caps": fl(m["all_caps"], 0.01), "greeting_newline": fl(m["greeting_then_newline"], 0.03),
         "polite_opener": fl(m["greeting_opener"], 0.05), "comma_splice": fl(m["comma_splice"], 0.15),
         "sms_word": fl(min(1.0, w["sms_tokens"] * 8), 0.10), "no_apostrophe": fl(m["any_missing_apostrophe"] * 3, 0.3),
         "typo_word_mild": fl(w["typos"] * 0.5, 0.01), "typo_word_strong": 0.12,
         "autocorrect": 0.5, "sound": 0.25, "all_lower_strong": 0.8, "run_together": 0.25,
         "grade_form": 0.5, "drop_article": fl(z.get("how to (no subject)", 0) * 2, 0.15),
         "zam_failing": fl(z.get("failing to", 0) * 3, 0.3), "zam_how_to": fl(z.get("how to (no subject)", 0) * 3, 0.3),
         # measured (greetings like mwabuka / twalumba in ~4.5% of real messages) but NOT generated: a Tonga greeting would
         # route the row into the translator, against Chris's English-only ruling ("this test shouldn't have anything
         # to do with real Tonga"); reported, not used
         "codeswitch": 0.0,
         "codeswitch_tokens": [g for g in sorted(GREET_WORDS) if g in real.get("_greet_seen", [])]}
    return P


def cmd_build(a):
    real = json.load(open(os.path.join(OUT, "real_stats.json"), encoding="utf-8"))
    P = params(real)
    src = os.path.join(RUN, "testset_v2_en.jsonl")
    ssha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    assert ssha == open(src + ".sha256").read().split()[0], "testset_v2_en changed"
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "harness"))
    from plugins import education  # noqa: PLC0415
    PL, _ = education.load()

    def subj(r):
        it = PL.items.get(r.get("from_item") or "") or {}
        return it.get("subject") or (r.get("intent") or "-")
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["slice"], r.get("grade"), subj(r))].append(r)
    rng = random.Random(SEED)
    keys = sorted(strata, key=str)
    for k in keys:
        rng.shuffle(strata[k])
    pick, i = [], 0
    while len(pick) < N_NOISY:          # round-robin over strata: spread over slices, grades, subjects
        k = keys[i % len(keys)]
        if strata[k]:
            pick.append(strata[k].pop())
        i += 1
    rng.shuffle(pick)
    out, applied_all = [], collections.Counter()
    for j, r in enumerate(pick):
        level = "mild" if j % 2 == 0 else "strong"
        rr = random.Random(f"{SEED}-{r['id']}-{level}")
        txt, ap = noisy(r["text"], level, rr, P)
        for x in set(ap):
            applied_all[(level, x)] += 1
        n = {k: v for k, v in r.items() if k not in ("guard", "title_overlap", "draft_file")}
        n.update(id="NZ" + r["id"], slice="noisy", noise_level=level,
                 profile="B phone+autocorrect" if level == "mild" else "A fast laptop typist, no autocorrect",
                 source_id=r["id"], source_text=r["text"], text=txt, applied=sorted(set(ap)), synthetic=True,
                 use="TEST ONLY, synthetic noise on a testset_v2_en row (english_noise.py, seed 20261024); gold = the source row's",
                 guard={"toi": [], "en": [txt]})
        out.append(n)
    p = os.path.join(RUN, "testset_v2_noisy.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nsha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    open(p + ".sha256", "w", newline="\n").write(f"{nsha}  testset_v2_noisy.jsonl\n")
    comb = os.path.join(RUN, "testset_v2_en_noisy.jsonl")
    with open(comb, "w", encoding="utf-8", newline="\n") as f:
        for r in rows + out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    csha = hashlib.sha256(open(comb, "rb").read()).hexdigest()
    open(comb + ".sha256", "w", newline="\n").write(f"{csha}  testset_v2_en_noisy.jsonl\n")
    sp, (toi, nya) = _spell(), _tonga_sets()
    gen = {lv: summarise([features(r["text"], sp, toi, nya) for r in out if r["noise_level"] == lv])
           for lv in ("mild", "strong")}
    gen["pooled"] = summarise([features(r["text"], sp, toi, nya) for r in out])
    src_stats = summarise([features(r["source_text"], sp, toi, nya) for r in out])
    meta = {"rows_noisy": len(out), "rows_combined": len(rows) + len(out), "seed": SEED, "params": P,
            "sha256": {"testset_v2_en": ssha, "testset_v2_noisy": nsha, "testset_v2_en_noisy": csha},
            "applied_counts": {f"{lv}:{x}": n for (lv, x), n in sorted(applied_all.items())},
            "levels": dict(collections.Counter(r["noise_level"] for r in out)),
            "slices_of_sources": dict(collections.Counter(r["source_id"][:2] for r in out)),
            "real": {k: real[k] for k in ("messages", "length_words", "share_of_messages", "per_word", "sms_forms",
                                          "grade_forms", "zambian_constructions_share", "typo_kinds")},
            "generated": gen, "sources_clean": src_stats,
            "limit": "an approximation of how teachers type to an assistant; the real check is the first real teachers' "
                     "held-out requests (the flywheel's first 40 per teacher)"}
    json.dump(meta, open(os.path.join(OUT, "noisy_meta.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("rows_noisy", "rows_combined", "levels", "sha256")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "measure", "build"])
    A = ap.parse_args()
    {"calibrate": cmd_calibrate, "measure": cmd_measure, "build": cmd_build}[A.cmd](A)
