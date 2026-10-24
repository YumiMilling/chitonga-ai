"""Intake: what does the teacher want? (23 Oct 2026, the Jesuit; DECISIONS 23 Oct "the model only READS").

Deterministic first: menu taps never come here (the engine routes them); then fixed phrasings (start over, menu,
language, name, school, add a class), the plug-in's keywords per unit (English mine; Tonga only words the plug-in's
teacher / book material carries, checked by the loader), the teacher's own earlier requests (memory recall), the
language and marking routes, and lexical retrieval. Then, for free text the rules cannot place, a READER.

THE READER INTERFACE. Every reader returns the same fixed structure (validate_reading enforces it):
    {"intent": one of INTENTS, "plugin": str, "item": id or None (MUST be one of the retrieved candidates or the unit's
     items), "unit": id or None, "subject": id or None,
     "slots": {"grade": int or None, "count": int or None, "topic": str or None, "language": "en" | "toi" | "mixed"},
     "confidence": 0..1, "source": str, "candidates": [ids], "why": str}
Backends: RulesReader (default, no model); TranslatorGemmaReader (translator toi>en, then the untouched Gemma on
llama.cpp 8091 with a JSON schema; not run by default); a trained reader later (the chat run plugs in here, same
structure). The engine asks for confirmation when confidence < CONFIRM_BELOW or when the reading came through a
translation (a reading that changes the answer must be visible to the teacher).

24 Oct 2026 (architecture review items 2, 7, 10): the engine validates EVERY reading (validate_reading, plus: the item
and unit exist in the plug-in), so the reader contract is enforced in one place, not only inside the model reader; a
MODEL reading always goes through a confirmation for the pilot (needs_confirm: verbalised LLM confidence is poorly
calibrated, Xiong et al. 2023, until measured on our held-out set). The plug-in's own intents and routes are no longer
here: RulesReader asks the plug-in's hooks (`hooks.detect(text, reading, plugin, units, strong, ...)` -> a reading
or None; education: the language and marking routes and the class-struggle phrasings) before the generic rules; a
plug-in names its extra intents in `hooks.INTENTS`. A route is any item whose id starts with 'route:'.
Gemma's HTTP timeout is 20 s (was 120 s: one hung llama.cpp held a teacher's queue for two minutes).
"""
import json
import os
import re
import sys

from .retrieval import off_grade as _off_grade

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(TOOLS, "assistant"), os.path.join(TOOLS, "llm")):
    if p not in sys.path:
        sys.path.insert(0, p)

INTENTS = ("lesson", "struggle", "route", "menu", "start_over", "continue", "set_language", "set_name", "set_school",
           "add_class", "unknown")
CONFIRM_BELOW = 0.75
MODEL_ALWAYS_CONFIRMS = True        # pilot rule (review item 2): a model reading is shown to the teacher before it acts
GEMMA_TIMEOUT = 20

START_OVER = re.compile(r"(?i)^\W*(start\s+(over|again|afresh)|restart|reset|from\s+the\s+(beginning|start)|new\s+start)\W*$")
MENU = re.compile(r"(?i)^\W*(menu|main\s+menu|show\s+(me\s+)?the\s+menu|options|what\s+can\s+you\s+do)\W*$")
CONTINUE = re.compile(r"(?i)^\W*(continue|go\s+on|carry\s+on|next\s+part|more)\W*$")
SET_LANG = re.compile(r"(?i)\b(?:in|use|prefer|want)\s+(english|tonga|chitonga|both)(?:\s+(?:only|please|languages))?\b|"
                      r"^\W*(english|tonga|chitonga|both)(\s+(only|please|languages))?\W*$")
SET_NAME = re.compile(r"(?i)\bmy\s+name\s+is\s+([A-Za-z][\w'.-]*(?:\s+[A-Z][\w'.-]*)?)|^\W*(?:i\s*am|i'm)\s+"
                      r"((?:mr|mrs|ms|miss|fr|father|sister|teacher|madam|sir)\.?\s+[A-Z][\w'.-]*)")
SET_SCHOOL = re.compile(r"(?i)\b(?:my\s+school\s+is|i\s+teach\s+at|i\s+am\s+at)\s+([A-Z][\w' .-]{2,60}?)(?:\s+in\s+"
                        r"([A-Z][\w' -]{2,30}?)(?:\s+district)?)?\s*[.!]?$")
ADD_CLASS = re.compile(r"(?i)\b(add\s+(a\s+)?(new\s+)?class|another\s+class|new\s+class|i\s+also\s+teach)\b")
GRADES = re.compile(r"(?i)\b(?:grades?|giledi)\s*(\d)(?:\s*(?:and|&|,|/|-)\s*(\d))?|\bgrade\s+(one|two|three|four|five|six|seven)\b")
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
COUNT = re.compile(r"(?i)\b(\d{1,2})\s+(reasons|questions|examples|words|sentences|activities|ideas)\b")


def grades_in(text):
    m = GRADES.search(text or "")
    if not m:
        return []
    if m.group(3):
        return [WORDNUM[m.group(3).lower()]]
    return sorted({int(x) for x in (m.group(1), m.group(2)) if x})


def language_of(text, toi_vocab, en_vocab=frozenset()):
    """'en' | 'toi' | 'mixed': the share of words that are in the plug-in's Tonga material and not English words"""
    ws = [w.lower() for w in re.findall(r"[A-Za-zŋŊ']+", text or "")]
    if not ws:
        return "en"
    toi = sum(1 for w in ws if w in toi_vocab and len(w) > 2 and w not in EN_COMMON and w not in en_vocab)
    share = toi / len(ws)
    return "toi" if share >= 0.4 else "mixed" if share >= 0.15 else "en"


EN_COMMON = set("""a an the of to in on at for from by with and or but is are was be it this that i you we they my our
your their me us how what why which who when where can do does did not no yes please give me lesson grade class
teach teaching one two three ten more less same""".split())


def blank(plugin, lang="en"):
    return {"intent": "unknown", "plugin": plugin, "item": None, "unit": None, "subject": None,
            "slots": {"grade": None, "count": None, "topic": None, "language": lang}, "confidence": 0.0,
            "source": "none", "candidates": [], "why": ""}


def plugin_intents(plugin):
    hk = getattr(plugin, "hooks", None) if plugin is not None else None
    return tuple(getattr(hk, "INTENTS", ()) or ())


def needs_confirm(r):
    """True when this reading must be confirmed before it acts: low confidence, a Tonga / mixed request, a translated
    reading, or (pilot rule) any model reading"""
    return (r["confidence"] < CONFIRM_BELOW or r["slots"]["language"] != "en" or bool(r.get("translated"))
            or (MODEL_ALWAYS_CONFIRMS and str(r.get("source", "")).startswith("model")))


def validate_reading(r, allowed_items=(), plugin=None):
    """[problems]; a reader's output that fails is discarded (the engine falls back to 'unknown'). With the plug-in:
    its own intents are allowed, and the item / unit must exist in it and the reading must name it."""
    p = []
    need = ("intent", "plugin", "item", "unit", "subject", "slots", "confidence", "source", "candidates", "why")
    p += [f"missing {k}" for k in need if k not in r]
    if p:
        return p
    if r["intent"] not in INTENTS + plugin_intents(plugin):
        p.append(f"intent {r['intent']}")
    if plugin is not None:
        if r["plugin"] != plugin.name:
            p.append(f"plugin {r['plugin']} is not {plugin.name}")
        if r["item"] and r["item"] not in plugin.items:
            p.append(f"item {r['item']} not in the plug-in")
        if r["unit"] and r["unit"] not in plugin.units:
            p.append(f"unit {r['unit']} not in the plug-in")
    if not isinstance(r["confidence"], (int, float)) or not 0 <= r["confidence"] <= 1:
        p.append("confidence")
    if r["item"] and r["item"] not in set(r["candidates"]) | set(allowed_items):
        p.append(f"item {r['item']} not among the candidates")
    s = r["slots"]
    if not isinstance(s, dict) or s.get("language") not in ("en", "toi", "mixed"):
        p.append("slots")
    return p


class Reader:
    """the interface: read(text, ctx) -> reading (the fixed structure above). ctx: {'plugin': Plugin, 'index':
    retrieval.Index, 'recall': item id or None, 'grades': [ints], 'number': str}"""
    name = "reader"

    def read(self, text, ctx):
        raise NotImplementedError


class RulesReader(Reader):
    """deterministic; no model. The default."""
    name = "rules"

    def __init__(self, use_lexq=True):
        self.use_lexq = use_lexq
        self._toi = {}                  # per plug-in (a single cache mixed two plug-ins' vocabularies)
        self._en = None

    def en_vocab(self):
        """the English word list the assistant uses to tell Tonga from English (h2_text._en_vocab, 7,886 words; as
        translator_service.detect_direction); EN_COMMON when it cannot load"""
        if self._en is None:
            try:
                import h2_text  # noqa: PLC0415
                self._en = frozenset(h2_text._en_vocab()) | EN_COMMON
            except Exception:  # noqa: BLE001
                self._en = frozenset(EN_COMMON)
        return self._en

    def _unit_scores(self, text, plugin):
        low = " " + re.sub(r"[^\w\s'-]", " ", (text or "").lower()) + " "
        toks = re.findall(r"[a-zŋ']+", low)
        scores = {}
        for uid, kw in plugin.intake.get("units", {}).items():
            en = [k for k in kw.get("en", []) if re.search(r"(?<![a-z])" + re.escape(k.lower().strip()) + r"(?![a-z])", low)]
            toi = [k for k in kw.get("toi", []) if any(k.lower() in t for t in toks)]
            if en or toi:
                scores[uid] = {"en": en, "toi": toi,
                               "w": sum(len(k.split()) + 0.5 for k in en) + 1.0 * len(toi)}
        return sorted(scores.items(), key=lambda kv: -kv[1]["w"])

    def _subject(self, text, plugin):
        low = (text or "").lower()
        for sid, ws in plugin.intake.get("subjects", {}).items():
            if any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low) for w in ws):
                return sid
        return None

    def read(self, text, ctx):
        P = ctx["plugin"]
        if P.name not in self._toi:
            self._toi[P.name] = P.tonga_vocabulary()
        lang = language_of(text, self._toi[P.name], self.en_vocab())
        r = blank(P.name, lang)
        gs = grades_in(text)
        r["slots"]["grade"] = gs[0] if gs else None
        m = COUNT.search(text or "")
        r["slots"]["count"] = int(m.group(1)) if m else None
        t = (text or "").strip()
        # fixed phrasings
        for rx, intent in ((START_OVER, "start_over"), (MENU, "menu"), (CONTINUE, "continue")):
            if rx.search(t):
                return dict(r, intent=intent, confidence=1.0, source="rules:phrase", why=intent)
        m = SET_NAME.search(t)
        if m:
            r["slots"]["topic"] = (m.group(1) or m.group(2)).strip().rstrip(".")
            return dict(r, intent="set_name", confidence=0.95, source="rules:phrase", why="name given")
        m = SET_SCHOOL.search(t)
        if m:
            r["slots"]["topic"] = json.dumps({"school": m.group(1).strip(), "district": (m.group(2) or "").strip() or None})
            return dict(r, intent="set_school", confidence=0.9, source="rules:phrase", why="school given")
        if ADD_CLASS.search(t):
            r["slots"]["topic"] = t
            return dict(r, intent="add_class", confidence=0.9, source="rules:phrase", why="add a class")
        m = SET_LANG.search(t)
        if m and len(t.split()) <= 6:
            v = (m.group(1) or m.group(2)).lower()
            r["slots"]["topic"] = {"english": "en", "tonga": "toi", "chitonga": "toi", "both": "both"}[v]
            return dict(r, intent="set_language", confidence=0.95, source="rules:phrase", why="language preference")
        # the grade to prefer (24 Oct 2026, after Grades 2-3): a grade named in the message, else the class's grade(s).
        # Only a tie-break: a unit of that grade wins over an EQUAL keyword match in another grade ('riddles' is a
        # Grade 2 and a Grade 3 unit); a stronger match elsewhere still wins (long division is Grade 3 by the syllabus,
        # asked by a Grade 2 teacher). Two equal units both of the preferred grades (a multigrade class) stay a tie,
        # so the engine asks.
        pref = set(gs or ctx.get("grades") or [])
        units = sorted(self._unit_scores(t, P),
                       key=lambda kv: (-kv[1]["w"], P.units[kv[0]].get("grade") not in pref))
        subject = self._subject(t, P)
        r["subject"] = subject
        strong = units and units[0][1]["en"]
        # the plug-in's own intents (education: the language / marking routes, the class-struggle phrasings)
        hk = getattr(P, "hooks", None)
        if hk is not None and hasattr(hk, "detect"):
            det = hk.detect(t, r, P, units, strong, use_lexq=self.use_lexq, en_vocab=self.en_vocab)
            if det:
                return det
        # memory of seen requests
        if ctx.get("recall") and ctx["recall"] in P.items:
            it = ctx["recall"]
            return dict(r, intent="lesson", item=it, unit=P.unit_of.get(it), candidates=[it], confidence=0.95,
                        source="memory:recall", why="the same request as before")
        hits = ctx["index"].search(t, grades=sorted(pref) or [1])
        cands = [c[3] for c in hits]
        # 24 Oct 2026 (sample lessons): an off-grade sample (a Grade 5 guide for a Grade 1 class) may be a candidate,
        # after all other material, but never the reading's lead: a Grade 1-3 request reads exactly as before
        lead = [c for c in hits if not (c[3] in P.items and _off_grade(P.items[c[3]], sorted(pref) or [1]))]
        if units:
            uid, sc = units[0]
            u = P.units[uid]
            tie = (len(units) > 1 and units[1][1]["w"] == sc["w"]
                   and (P.units[units[1][0]].get("grade") in pref) == (u.get("grade") in pref))
            conf = 0.85 if sc["en"] else min(0.6 + 0.05 * (len(sc["toi"]) - 1), 0.7)
            if subject and subject != u["subject"]:
                conf -= 0.2
            if tie:
                conf = min(conf, 0.5)
            item = u["items"][0] if u["items"] else None
            return dict(r, intent="lesson", unit=uid, item=item, subject=u["subject"],
                        candidates=list(dict.fromkeys(u["items"] + cands)), confidence=round(conf, 2),
                        source="rules:keyword", why=f"keywords {sc['en'] + sc['toi']}"
                        + (f"; tie with {units[1][0]}" if tie else ""),
                        alternatives=[x[0] for x in units[1:3]])
        if lead:
            top = lead[0]
            conf = 0.8 if top[1] >= 0.8 else 0.6
            return dict(r, intent="lesson", item=top[3], unit=P.unit_of.get(top[3]), candidates=cands,
                        confidence=conf, source="rules:retrieval", why=f"retrieval cover {top[1]}, title hits {top[2]}")
        if subject:
            return dict(r, intent="menu", subject=subject, confidence=0.8, source="rules:subject", why="subject named")
        return r


class TranslatorGemmaReader(Reader):
    """translator toi>en (the served translator through the assistant's backends), then the UNTOUCHED Gemma (llama.cpp
    server, port 8091) with a JSON schema that only allows the retrieved candidates. Not run by default (needs the
    models); `post` can be replaced (tests use a stub). Falls back to the rules reading when anything fails."""
    name = "translator+gemma"

    def __init__(self, backends=None, url=None, post=None, rules=None):
        self.b = backends
        self.url = (url or os.environ.get("ASSISTANT_GEMMA_URL", "http://127.0.0.1:8091")).rstrip("/")
        self.post = post or self._post
        self.rules = rules or RulesReader()

    def _post(self, body):
        import urllib.request  # noqa: PLC0415
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=GEMMA_TIMEOUT) as f:
            return json.loads(f.read())["choices"][0]["message"]["content"]

    def read(self, text, ctx):
        base = self.rules.read(text, ctx)
        if base["confidence"] >= CONFIRM_BELOW or base["intent"] in ("start_over", "menu", "continue", "set_name",
                                                                         "set_language", "set_school", "add_class"):
            return base
        P = ctx["plugin"]
        reading_en = text
        if base["slots"]["language"] != "en" and self.b is not None:
            try:
                reading_en = (self.b.translate(text, "toi>en", model=self.b.cfg.get("readback", "R")).get("text") or text)
            except Exception:  # noqa: BLE001
                reading_en = text
        gs = ctx.get("grades") or [1]
        cands = list(dict.fromkeys(base["candidates"] + [c[3] for c in ctx["index"].search(reading_en, k=8)]))
        cands = [c for c in cands if not (c in P.items and _off_grade(P.items[c], gs))][:8]   # samples: as the rules
        if not cands:
            return dict(base, why=base["why"] + "; no candidates for the model")
        menu = [{"id": c, "title": P.items[c]["title"], "subject": P.items[c].get("subject")} for c in cands]
        schema = {"type": "object", "properties": {
            "intent": {"enum": ["lesson", "struggle", "unknown"]}, "item": {"enum": cands + [None]},
            "grade": {"type": ["integer", "null"]}, "confidence": {"type": "number"}},
            "required": ["intent", "item", "confidence"]}
        msgs = [{"role": "system", "content": "You read a teacher's request and pick ONE item from the list, or null. "
                                              "Answer with JSON only. Never write content."},
                {"role": "user", "content": json.dumps({"request": text, "english_reading": reading_en,
                                                        "items": menu}, ensure_ascii=False)}]
        try:
            raw = self.post({"messages": msgs, "temperature": 0, "max_tokens": 120,
                             "response_format": {"type": "json_object", "schema": schema},
                             "chat_template_kwargs": {"enable_thinking": False}})
            out = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            return dict(base, why=base["why"] + f"; model reader failed: {e}")
        r = dict(base, intent=out.get("intent") or "unknown", item=out.get("item"), unit=P.unit_of.get(out.get("item")),
                 candidates=cands, confidence=max(0.0, min(1.0, float(out.get("confidence") or 0))) * 0.9,
                 source="model:gemma", why=f"read as: {reading_en}")
        r["slots"] = dict(base["slots"], grade=out.get("grade") or base["slots"]["grade"])
        if validate_reading(r, plugin=P):
            return dict(base, why=base["why"] + "; model reading invalid")
        r["translated"] = reading_en != text
        return r
