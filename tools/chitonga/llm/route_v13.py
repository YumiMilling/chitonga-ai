"""Chat round v0.13 (23 Oct 2026, the Jesuit): the READER of the teacher assistant, and nothing else.

Chris, 23 Oct, relayed: "isn't behaviour guided through the code surrounding the model?" Agreed. Behaviour lives in
CODE: the redirect, the encouragement, the referral and the no-material lines are fixed templates; the candidate cards,
lessons and classroom lines come from RETRIEVAL over the library; facts come only from cards. The MODEL's one job is
READING: a free-form teacher request (Tonga, mixed, English, phone spelling) -> one line of JSON:

    {"read": "<the request in plain English>", "intent": "<one of INTENTS>", "pick": "<option id | template | null>",
     "slots": {"grade": int, "count": int, "number": int, "out_of": int, "words": ["<Tonga word copied from the request>"]}}

The same prompt (SYSTEM + user_msg) is used for training rows, for the frozen test, for the two no-training baselines
and at serving time, so every arm is scored on identical inputs. Retrieval (shortlist) is deterministic and lexical;
Tonga words are expanded into English through the lexicon glosses (all senses: the model must choose, e.g. SC08 'mala'
= entrails AND, per the CDC module, nails). No model and no translator is needed to build a prompt.

    import route_v13 as R; C = R.Catalogue.load(); opts = C.shortlist(text); msgs = R.messages(text, opts)
    R.parse(reply) -> dict | None;  R.score(pred, gold) -> {'ok', 'intent_ok', 'pick_ok', 'slots_ok', ...}
"""
import csv
import functools
import glob
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
LIB = os.path.join(CT, "assistant_library")
ADV = os.path.join(CT, "advice", "cards")
LEX = os.path.join(REPO, "tools", "chitonga", "grammar2", "lexicon.tsv")

INTENTS = ("card", "lesson", "say", "advice", "marking", "word", "social", "safeguarding", "health_no_card",
           "no_material", "redirect")
MARKING = ("mark_scheme", "word_check", "word_parts")
WORDQ = ("senses", "parts", "months")
PICK_FROM_OPTIONS = ("card", "lesson", "say", "advice")

SYSTEM = (
    "You read requests sent to a teacher assistant for Zambian primary school teachers. Teachers write in Tonga "
    "(Chitonga), in English, or mixed, often with phone spelling. You do not answer the request: the assistant's code "
    "answers from checked material. Your only job is to read the request and reply with ONE line of JSON, nothing else:\n"
    '{"read": "<the request in plain English>", "intent": "<intent>", "pick": "<id or template or null>", '
    '"slots": {<only the slots the request gives>}}\n'
    "Intents:\n"
    "card = explain or teach a topic that one of the listed card options covers (pick its id)\n"
    "lesson = a Grade 1 lesson guide from the options (pick its id)\n"
    "say = how to say a classroom line in Tonga, one of the listed line options (pick its id)\n"
    "advice = a classroom situation: large class, no books, mixed levels, discipline, absence, hungry learners, "
    "parents, no electricity, a tired teacher, multigrade (pick the advice option)\n"
    "marking = pick mark_scheme (mark answers, marks out of N), word_check (check a pupil's Tonga writing or mixed "
    "words) or word_parts (can the learner break a word into parts)\n"
    "word = a question about Tonga words: pick senses (what a word means, how two words differ), parts (the parts of "
    "a word) or months (the months in Tonga)\n"
    "social = greeting, thanks, or a hard day (no pick)\n"
    "safeguarding = a child may be harmed or abused (no pick)\n"
    "health_no_card = a health question that no listed option covers (no pick)\n"
    "no_material = a school topic that no listed option covers (no pick)\n"
    "redirect = anything else: news, politics, money, sport, jokes, personal matters (no pick)\n"
    "Slots: grade, count (how many things are asked for), number (a number the lesson is about), out_of (marks), "
    "words (Tonga words the request asks about, copied exactly). Pick only from the options listed or the templates "
    "named above. Never write facts and never write Tonga of your own.")

_STOP = set("""a an the of to in on at for from by with and or but if then so as is are was were be been being am do
does did doing have has had having it its this that these those there here what whats why how which where when who whom
whose give list name explain describe define state tell me us you your we our they their i my he she his her them can
could would should will shall may might must please some any many much more most very about into out up down over under
than too also just only not no yes make makes made one two three four five lesson lessons teach teaching class grade
learners learner pupils pupil children child help need want tomorrow today topic""".split())


def stem(w):
    w = w.lower().replace("’", "'")
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    return w.replace("metre", "meter")


def words(text):
    return re.findall(r"[A-Za-zŋŊ']+", text or "")


def content(text):
    return [stem(w) for w in words(text) if w.lower() not in _STOP and len(w) > 1]


# ------------------------------------------------------------------------------------------------ lexicon glosses
@functools.lru_cache(maxsize=1)
def lexicon():
    out = {}
    with open(LEX, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            w = (r.get("word") or "").lower().strip()
            if not w or (r.get("unclear") or "").strip():
                continue
            m = re.sub(r"\s+", " ", r.get("meaning") or "")
            m = re.sub(r"(?i)hopgood index:?|index p\d+|\(.*?\)|cf\..*|plural of \S+", " ", m)
            out.setdefault(w, []).append(m)
            for pl in re.split(r"[,/;]", (r.get("plural") or "").lower()):
                if pl.strip():
                    out.setdefault(pl.strip(), []).append(m)
    return out


@functools.lru_cache(maxsize=1)
def en_vocab():
    p = os.path.join(CT, "vocab", "en_dictionary_words.json")
    try:
        d = json.load(open(p, encoding="utf-8"))
        return frozenset(x.lower() for x in (d if isinstance(d, list) else d.keys()))
    except Exception:  # noqa: BLE001
        return frozenset()


def gloss_expand(text, cap=4):
    """English content words for the Tonga words of a request: every lexicon sense (up to cap per word). English words
    pass through. Not a translation: a retrieval key only."""
    out, lx, ev = [], lexicon(), en_vocab()
    for w in words(text):
        lw = w.lower()
        if lw in ev and lw not in lx:
            continue
        for m in (lx.get(lw) or [])[:cap]:
            out += content(m)
    return out


# ------------------------------------------------------------------------------------------------ the catalogue
ROW = re.compile(r"^\|\s*(\d{1,2})\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|")


def _lesson_files():
    fs = sorted(glob.glob(os.path.join(LIB, "grade1", "*_SAY_IT_IN_TONGA.md")))
    fs.append(os.path.join(LIB, "place_value_try_2026-10-22", "PLACE_VALUE_SAY_IT_IN_TONGA.md"))
    return fs


class Catalogue:
    """every pickable item: {'id', 'kind' card|lesson|say|advice, 'label' (the option line), 'text' (retrieval text),
    'health' bool, 'grade' [ints], 'tonga' (say lines only), 'origin'}"""

    def __init__(self, items):
        self.items = items
        self.by_id = {x["id"]: x for x in items}
        self.df = {}
        for x in items:
            for w in set(content(x["text"])):
                self.df[w] = self.df.get(w, 0) + 1

    @classmethod
    def load(cls):
        items = []
        for f in ("cards_v0.jsonl", "cards_v1.jsonl"):
            for l in open(os.path.join(LIB, f), encoding="utf-8"):
                if not l.strip():
                    continue
                c = json.loads(l)
                g = c.get("grade") or []
                gs = "-".join(str(x) for x in (g[:1] + g[-1:] if len(g) > 1 else g))
                health = c.get("subject") == "Health"
                items.append({"id": c["id"], "kind": "card", "grade": g, "health": health, "subject": c.get("subject"),
                              "label": f"{c['id']}: {c['topic']} (card, Grade {gs}, {c.get('subject')})",
                              "text": " ".join([c["topic"]] * 3 + [c.get("answer", ""), c.get("idea", "")])})
        for f in _lesson_files():
            md = open(f, encoding="utf-8").read()
            head = md.splitlines()[0]
            code = "3.1-place-value-intro" if "PLACE_VALUE" in f else os.path.basename(f).replace("_SAY_IT_IN_TONGA.md", "")
            title = re.sub(r"^#\s*", "", head).split("(")[0].split(":")[0].strip()
            subj = {"E": "English", "S": "Science"}.get(code[0], "Maths")
            health = code.startswith("S.1.2")
            lid = "L:" + code
            kw = []
            for ln in md.splitlines():
                m = ROW.match(ln)
                if m:
                    en, toi, origin = m.group(2), m.group(3), m.group(4)
                    if toi.startswith("(dropped") or not toi.strip() or origin.strip() in ("-", ""):
                        continue
                    sid = f"S:{code}#{m.group(1)}"
                    if sid in {x["id"] for x in items}:
                        sid += "b"
                    items.append({"id": sid, "kind": "say", "grade": [1], "health": health, "subject": subj,
                                  "tonga": toi, "origin": origin, "en": en,
                                  "label": f'{sid}: say "{en}" (Grade 1 {title})', "text": en + " " + toi + " " + title})
                elif ln.startswith("|") and ln.count("|") >= 4 and not re.match(r"^\|\s*(English|#|---)", ln):
                    kw += [ln.split("|")[1].strip(), ln.split("|")[2].strip()]
            items.append({"id": lid, "kind": "lesson", "grade": [1], "health": health, "subject": subj,
                          "label": f"{lid}: {title} (Grade 1 lesson, {subj}, with Tonga lines)",
                          "text": " ".join([title] * 3 + kw + ["lesson guide plan grade 1"])})
        for f in sorted(glob.glob(os.path.join(ADV, "[0-9][0-9]_*.md"))):
            md = open(f, encoding="utf-8").read()
            num = os.path.basename(f)[:2]
            title = md.splitlines()[0].lstrip("# ").split("·")[-1].strip()
            sit = re.search(r"\*\*Situation:\*\*\s*(.+)", md)
            items.append({"id": "A:" + num, "kind": "advice", "grade": [], "health": False, "subject": "classroom",
                          "label": f"A:{num}: {title} (advice)", "text": " ".join([title] * 3 + [sit.group(1) if sit else ""])})
        return cls(items)

    def idf(self, w):
        return math.log((1 + len(self.items)) / (1 + self.df.get(w, 0))) + 1.0

    def shortlist(self, text, k=8, force=None, drop=None):
        """the k options for a request: lexical IDF cover of the request's content words + the lexicon glosses of its
        Tonga words; at most 4 say lines (they crowd out cards otherwise). force: ids to include (the test inserts the
        gold when retrieval misses it, flagged); drop: ids to remove (training rows for no_material)."""
        q = list(dict.fromkeys(content(text) + gloss_expand(text)))
        drop = set(drop or ())
        ranked = []
        if q:
            tot = sum(self.idf(w) for w in q)
            for x in self.items:
                if x["id"] in drop:
                    continue
                xs = set(content(x["text"]))
                s = sum(self.idf(w) for w in q if w in xs) / tot
                if s > 0:
                    ranked.append((s, x["id"]))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        out, says = [], 0
        for s, i in ranked:
            if len(out) >= k:
                break
            if self.by_id[i]["kind"] == "say":
                if says >= 4:
                    continue
                says += 1
            out.append(i)
        for f in force or ():
            if f not in out and f in self.by_id:
                out = out[: k - 1] + [f] if len(out) >= k else out + [f]
        return out


def user_msg(text, options, cat):
    lines = [f"Request: {text.strip()}", "", "Options:"]
    lines += [f"- {cat.by_id[i]['label']}" for i in options] or ["- (none)"]
    return "\n".join(lines)


def messages(text, options, cat, reply=None):
    ms = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(text, options, cat)}]
    if reply is not None:
        ms.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
    return ms


# ------------------------------------------------------------------------------------------------ parse and score
def parse(reply):
    """the first JSON object in the reply (code fences and thinking stripped), or None"""
    t = re.sub(r"(?s)<\|channel>thought.*?<channel\|>", "", reply or "")
    t = re.sub(r"```(json)?", "", t)
    i = t.find("{")
    while i >= 0:
        depth = 0
        for j in range(i, len(t)):
            depth += {"{": 1, "}": -1}.get(t[j], 0)
            if depth == 0:
                try:
                    d = json.loads(t[i:j + 1])
                    return d if isinstance(d, dict) else None
                except ValueError:
                    break
        i = t.find("{", i + 1)
    return None


def _pick(p):
    p = (p.get("pick") if isinstance(p, dict) else None)
    if p in (None, "", "null", "none", "None"):
        return None
    return str(p).strip()


def score(pred, gold, options=None):
    """gold: {'intent': [ok intents], 'pick': {intent: [ok picks]} , 'slots': {name: value}}.
    ok = intent in gold AND pick right for that intent (no pick expected -> pick must be empty). A pick from outside the
    options (or the fixed templates) is wrong. Slots are scored separately (only the slots the gold names)."""
    if not isinstance(pred, dict):
        return {"ok": False, "parsed": False, "intent_ok": False, "pick_ok": False, "slots_ok": None}
    it = str(pred.get("intent") or "").strip().lower()
    pk = _pick(pred)
    intent_ok = it in gold["intent"]
    want = (gold.get("pick") or {}).get(it)
    if want:
        pick_ok = pk in want and (it not in PICK_FROM_OPTIONS or options is None or pk in options)
    else:
        pick_ok = pk is None
    gs = gold.get("slots") or {}
    ps = pred.get("slots") if isinstance(pred.get("slots"), dict) else {}
    sl = None
    if gs:
        sl = True
        for k, v in gs.items():
            pv = ps.get(k)
            if k == "words":   # each gold word matched by a predicted word that contains it or is contained in it
                pw = [str(x).lower() for x in (pv or [])] if isinstance(pv, list) else [str(pv or "").lower()]
                sl &= bool(pw) and all(any(g.lower() in w or (w and w in g.lower()) for w in pw) for g in v)
            else:
                try:
                    sl &= int(pv) == int(v)
                except (TypeError, ValueError):
                    sl = False
    return {"ok": bool(intent_ok and pick_ok), "parsed": True, "intent_ok": intent_ok, "pick_ok": bool(pick_ok),
            "slots_ok": sl, "intent": it, "pick": pk}


def copied_words_ok(pred, request):
    """'words' must be copied from the request (no invented forms through the slot)"""
    ws = (pred or {}).get("slots", {}).get("words") if isinstance((pred or {}).get("slots"), dict) else None
    low = (request or "").lower()
    return all(str(w).lower() in low for w in (ws or []))


# ------------------------------------------------------------------------------------------------ code rules after reading
HEALTH = re.compile(r"(?i)(cholera|kkolela|kkolona|corona|covid|malaria|diarrh|\bhiv\b|\baids\b|tubercul|\btb\b|vaccin|"
                    r"medicine|fever|\bsick|illness|disease|bulwazi|infect|germs?\b|hygien|epilep|asthma|diabet|"
                    r"pregnan|anaemi|anemi|measles|ebola|mpox|rabies|snake|burns?\b|wound)")


def finalize(pred, request):
    """behaviour in code (Chris 23 Oct): a no-material reading of a health request gets the health line (it points to
    the clinic); Tonga words in the slot that are not in the request are dropped (never passed on as invented forms,
    but counted by the scorer before this step through copied_words_ok on the raw reply)."""
    if not isinstance(pred, dict):
        return pred
    p = dict(pred)
    if p.get("intent") == "no_material" and (HEALTH.search(request or "") or HEALTH.search(str(p.get("read") or ""))):
        p["intent"] = "health_no_card"
    return p
