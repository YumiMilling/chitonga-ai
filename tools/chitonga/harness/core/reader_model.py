"""The MODEL reader of the harness, and the one contract the chat round trains to (24 Oct 2026, the Jesuit).

Why this file exists. The 23 Oct chat round (chat_v13, RUN_PACKAGE.md) trained the model to its OWN structure
(route_v13: card / lesson / say / advice / marking / word / social / safeguarding ...), and the harness READER interface
(core/intake.py) has another one (lesson / struggle / language / marking / menu / unknown, item from the retrieved
candidates). An adapter would have had to guess between two vocabularies. So the round is re-aimed: the model's output
IS the harness reading, the model's candidates ARE the harness retrieval's list, and the same prompt, candidate list,
JSON schema and adapter are used (a) to build the training rows, (b) for the two no-training baselines and (c) at
serving time. Nothing the model says can reach a teacher except through intake.validate_reading and the engine's
gates: behaviour lives in code (DECISIONS 23 Oct "the model only READS").

THE MODEL'S OUTPUT (one line of JSON, constrained by `schema()` at serving time):
    {"read": "<the message in plain English>", "intent": lesson|struggle|language|marking|menu|unknown,
     "item": "<one of the listed candidate ids>" | null, "grade": int|null, "count": int|null, "subject": id|null}
`to_reading()` turns it into the harness reading: plugin, unit (the item's unit), subject, slots (grade, count, topic =
the English reading, language), confidence (a fixed MODEL_CONF: the model's own number is not calibrated, Xiong et al.
2023; the pilot confirms every model reading anyway), source "model:<name>", candidates, why, read_en (the engine
re-runs the safeguarding and scope gates on it: a Tonga message the English-only gates could not read).

CANDIDATES (`candidates()`): the rules reading's own list (the matched unit's items), the plug-in's Tonga vocabulary
hits (glossary forms, intake keywords, lesson key words: each maps to the lessons that carry it), the lexical index over
the English reading (translator output, if any) and over the lexicon glosses of the Tonga words (route_v13.gloss_expand:
every sense, a retrieval key, not a translation). At most K_CANDS, deterministic order. The model may only pick from
them; `language` and `marking` map to the route items in code; `unknown` means none fits.

Three readers share the class: ModelReader(mode="gemma") = the untouched -it reads the Tonga itself (baseline 1);
mode="translator+gemma" = the served translator toi>en first, then the untouched -it (baseline 2, the 23 Oct winner);
mode="trained" = a chat-round GGUF (the kill test's arm). All on llama.cpp (port 8091 by default), greedy, thinking off.
"""
import json
import os
import re
import sys

from .intake import CONFIRM_BELOW, Reader, RulesReader, validate_reading

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(TOOLS, "assistant"), os.path.join(TOOLS, "llm")):
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_INTENTS = ("lesson", "struggle", "language", "marking", "menu", "unknown")
ROUTE_OF = {"language": "route:language", "marking": "route:marking"}
K_CANDS = 8
MODEL_CONF = 0.7           # below CONFIRM_BELOW: a model reading is confirmed by the teacher (pilot rule, review item 2)
GEMMA_TIMEOUT = 40
CONTRACT_VERSION = "v14.1 (24 Oct 2026)"

SYSTEM = (
    "You read a teacher's message to a lesson assistant for Zambian primary school teachers (Grades 1 to 3: English, "
    "Chitonga, Maths, Science). Teachers write in Tonga (Chitonga), in English, or mixed, often with phone spelling. "
    "You do not answer the message: the assistant's code answers from checked material. Your only job is to read the "
    "message and reply with ONE line of JSON, nothing else:\n"
    '{"read": "<the message in plain English>", "intent": "<intent>", "item": "<id or null>", "grade": <int or null>, '
    '"count": <int or null>, "subject": "<english|chitonga|maths|science or null>"}\n'
    "Intents:\n"
    "lesson = the teacher wants material on a topic that one of the listed items covers (item = that id)\n"
    "struggle = the teacher describes what the class cannot do yet (item = the listed item for that topic, or null)\n"
    "language = a question about Tonga words: what a word means, how two words differ, the parts of a word, the "
    "months (item null)\n"
    "marking = marking answers or a test, marks out of N, checking a pupil's Tonga writing, whether a learner can "
    "split a word into parts (item null)\n"
    "menu = only a subject is named, no topic (subject set, item null)\n"
    "unknown = no listed item covers it, or it is not about teaching: news, money, politics, personal matters, a "
    "health or medical question with no listed item (item null)\n"
    "grade = a grade named in the message; count = how many things are asked for (3 reasons, 5 questions). Pick an item "
    "only from the list. Never write facts and never write Tonga of your own.")


def item_label(P, iid):
    it = P.items.get(iid) or {}
    g = it.get("grades") or []
    gl = ("Grade " + "-".join(str(x) for x in (g[:1] + g[-1:] if len(g) > 1 else g))) if g else "any grade"
    kind = it.get("kind", "item")
    subj = it.get("subject") or ""
    return f"{iid}: {it.get('title', '')} ({gl}, {subj} {kind})".replace("  ", " ")


def user_msg(text, cands, P, reading_en=None):
    lines = [f"Message: {text.strip()}"]
    if reading_en and reading_en.strip() and reading_en.strip().lower() != text.strip().lower():
        lines.append(f"English reading (by the translator, may be wrong): {reading_en.strip()}")
    lines += ["", "Items:"]
    lines += [f"- {item_label(P, c)}" for c in cands] or ["- (none)"]
    return "\n".join(lines)


def messages(text, cands, P, reply=None, reading_en=None):
    ms = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(text, cands, P, reading_en)}]
    if reply is not None:
        ms.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
    return ms


def target(read, intent, item=None, grade=None, count=None, subject=None):
    """the training target / the model's expected reply (field order fixed, nulls written: the schema wants them)"""
    assert intent in MODEL_INTENTS, intent
    return {"read": read, "intent": intent, "item": item, "grade": grade, "count": count, "subject": subject}


def schema(cands, subjects=("english", "chitonga", "maths", "science")):
    """llama.cpp json_schema: item only from the candidates (or null); intent from MODEL_INTENTS"""
    return {"type": "object",
            "properties": {"read": {"type": "string", "maxLength": 300},
                           "intent": {"type": "string", "enum": list(MODEL_INTENTS)},
                           "item": {"anyOf": [{"type": "string", "enum": list(cands)}, {"type": "null"}]} if cands
                           else {"type": "null"},
                           "grade": {"type": ["integer", "null"]}, "count": {"type": ["integer", "null"]},
                           "subject": {"anyOf": [{"type": "string", "enum": list(subjects)}, {"type": "null"}]}},
            "required": ["read", "intent", "item", "grade", "count", "subject"], "additionalProperties": False}


# ------------------------------------------------------------------------------------------------ candidates
_TOI_MAP = {}


def tonga_map(P):
    """Tonga form (lower) -> [item ids]: glossary forms -> their lesson; intake unit keywords -> the unit's items; lesson
    key words -> the lesson. Only the plug-in's own teacher / book / lexicon material (no invented words)."""
    if P.name in _TOI_MAP:
        return _TOI_MAP[P.name]
    m = {}

    def add(form, iid):
        f = (form or "").lower().strip()
        if len(f) >= 3 and iid in P.items:
            m.setdefault(f, [])
            if iid not in m[f]:
                m[f].append(iid)
    for g in P.glossary:
        for f in g.get("toi") or []:
            for alt in re.split(r"[;/]", f):
                add(alt, g.get("lesson"))
    for uid, kw in P.intake.get("units", {}).items():
        for f in kw.get("toi", []):
            for iid in P.units.get(uid, {}).get("items", []):
                add(f, iid)
    for iid, it in P.items.items():
        for k in it.get("key_words", []):
            if k.get("toi") and not k["toi"].startswith("("):
                for alt in re.split(r"[;/]", re.sub(r"\(.*?\)", "", k["toi"])):
                    add(alt, iid)
    _TOI_MAP[P.name] = m
    return m


def tonga_hits(text, P):
    """[(score, item id)] from the Tonga map: whole-word or whole-phrase matches, longer forms first"""
    low = " " + re.sub(r"[^\w\s'ŋŊ-]", " ", (text or "").lower()) + " "
    toks = set(re.findall(r"[a-zŋ']+", low))
    sc = {}
    for f, ids in tonga_map(P).items():
        if " " in f:
            hit = (" " + f + " ") in low
        else:
            hit = f in toks or (len(f) >= 5 and any(t.endswith(f) or t == f for t in toks))
        if hit:
            for iid in ids:
                sc[iid] = sc.get(iid, 0) + (2.0 if " " in f else 1.0)
    return sorted(((v, k) for k, v in sc.items()), key=lambda x: (-x[0], x[1]))


def gloss_query(text):
    """English retrieval key for the Tonga words of a message: every lexicon sense (route_v13.gloss_expand)"""
    try:
        import route_v13 as R  # noqa: PLC0415
        return " ".join(R.gloss_expand(text))
    except Exception:  # noqa: BLE001
        return ""


def candidates(text, P, index, base=None, reading_en=None, grades=None, k=K_CANDS):
    """the ordered candidate list the model may pick from (see the module docstring)"""
    out = []

    def push(ids):
        for i in ids:
            if i in P.items and i not in out and P.items[i]["kind"] in ("card", "lesson", "pointer"):
                out.append(i)
    if base:
        push(base.get("candidates") or [])
    push([i for _, i in tonga_hits(text, P)])
    # min_cover 0.25 / 0.2, not the index's 0.5 (measured 24 Oct on the 99 item-targeted held-out requests: an accepted
    # item among the 8 candidates 84 -> 87; the list is the model's to judge, the rules reader keeps its own 0.5)
    push([c[3] for c in index.search(text, k=k, grades=grades, min_cover=0.25)])
    if reading_en and reading_en.strip().lower() != text.strip().lower():
        push([c[3] for c in index.search(reading_en, k=k, grades=grades, min_cover=0.25)])
    gq = gloss_query(text)
    if gq:
        push([c[3] for c in index.search(gq, k=k, grades=grades, min_cover=0.2)])
    return out[:k]


# ------------------------------------------------------------------------------------------------ the adapter
def parse(reply):
    """the first JSON object in the reply (code fences and a thought channel stripped), or None"""
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


def to_reading(out, base, P, cands, source, reading_en=None):
    """model JSON -> the harness reading (intake.validate_reading decides whether it stands)"""
    r = dict(base)
    intent = str((out or {}).get("intent") or "unknown").strip().lower()
    item = (out or {}).get("item")
    item = None if item in (None, "", "null", "None") else str(item).strip()
    if intent in ROUTE_OF:
        item = ROUTE_OF[intent] if ROUTE_OF[intent] in P.items else None
        if item is None:
            intent = "unknown"
    elif intent not in MODEL_INTENTS:
        intent = "unknown"
    if intent in ("menu", "unknown"):
        item = None
    if intent == "lesson" and not item:
        intent = "unknown"     # review F3: 'lesson' with item null is schema-legal; the honest no-material line (with the
                               # menu) is the same answer for every reader, not the 'unclear' line
    dropped = ""
    if item and item not in P.items:
        dropped = f"; item {item} not in the plug-in, dropped"
        item = None
        if intent == "lesson":
            intent = "unknown"     # a lesson reading without a listed item is no reading (the schema forbids it at serving)
    unit = P.unit_of.get(item) if item else None
    it = P.items.get(item) or {}
    subject = (out or {}).get("subject")
    if subject not in P.intake.get("subjects", {}):
        subject = None
    if intent == "struggle" and item and not unit:
        unit = None
    read_en = str((out or {}).get("read") or "").strip()[:300]
    g = (out or {}).get("grade")
    c = (out or {}).get("count")
    slots = dict(base["slots"])
    slots["grade"] = int(g) if isinstance(g, int) and 1 <= g <= 7 else base["slots"].get("grade")
    slots["count"] = int(c) if isinstance(c, int) and 0 < c <= 50 else base["slots"].get("count")
    slots["topic"] = read_en or base["slots"].get("topic")
    r.update(intent=intent, item=item, unit=unit, subject=it.get("subject") or subject or base.get("subject"),
             slots=slots, confidence=MODEL_CONF, source=source,
             candidates=list(cands) + ([item] if item and item.startswith("route:") else []),
             why=f"read as: {read_en}" + (f" (translator: {reading_en})" if reading_en and reading_en != read_en else "") + dropped,
             read_en=read_en)
    if intent == "menu" and not r["subject"]:
        r["intent"] = "unknown"
    r["translated"] = bool(reading_en) and reading_en.strip().lower() != base.get("_text", "").strip().lower()
    r.pop("_text", None)
    return r


class ModelReader(Reader):
    """rules first (as the engine serves); the model only for free text the rules cannot place with confidence"""

    def __init__(self, mode="trained", backends=None, url=None, post=None, rules=None, name=None, translator="a050"):
        assert mode in ("gemma", "translator+gemma", "trained"), mode
        self.mode = mode
        self.name = name or ("model:" + mode)
        self.b = backends
        self.translator = translator
        self.url = (url or os.environ.get("ASSISTANT_GEMMA_URL", "http://127.0.0.1:8091")).rstrip("/")
        self.post = post or self._post
        self.rules = rules or RulesReader()
        self.last = None          # diagnostics of the last read: prompt, raw reply, candidates, reading_en

    def _post(self, body):
        import urllib.request  # noqa: PLC0415
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=GEMMA_TIMEOUT) as f:
            return json.loads(f.read())["choices"][0]["message"]["content"]

    def translate(self, text):
        if self.mode != "translator+gemma" or self.b is None:
            return None
        try:
            r = self.b.translate(text, "toi>en", model=self.translator)
            return (r.get("text") if isinstance(r, dict) else r) or None
        except Exception:  # noqa: BLE001
            return None

    def read(self, text, ctx):
        base = self.rules.read(text, ctx)
        if base["confidence"] >= CONFIRM_BELOW or base["intent"] in ("start_over", "menu", "continue", "set_name",
                                                                         "set_language", "set_school", "add_class",
                                                                         "language", "marking"):
            return base
        P = ctx["plugin"]
        reading_en = self.translate(text) if base["slots"]["language"] != "en" else None
        cands = candidates(text, P, ctx["index"], base=base, reading_en=reading_en, grades=ctx.get("grades"))
        msgs = messages(text, cands, P, reading_en=reading_en)
        self.last = {"candidates": cands, "reading_en": reading_en, "prompt": msgs, "reply": None}
        try:
            raw = self.post({"messages": msgs, "temperature": 0, "max_tokens": 160, "seed": 1,
                             "chat_template_kwargs": {"enable_thinking": False},
                             "response_format": {"type": "json_schema",
                                                 "json_schema": {"name": "reading", "schema": schema(cands)}}})
        except Exception as e:  # noqa: BLE001
            return dict(base, why=base["why"] + f"; model reader failed: {e}"[:200])
        self.last["reply"] = raw
        out = parse(raw)
        if out is None:
            return dict(base, why=base["why"] + "; model reply not JSON")
        r = to_reading(out, dict(base, _text=text), P, cands, self.name, reading_en)
        bad = validate_reading(r, plugin=P)
        if bad:
            return dict(base, why=base["why"] + "; model reading invalid: " + "; ".join(bad))
        return r
