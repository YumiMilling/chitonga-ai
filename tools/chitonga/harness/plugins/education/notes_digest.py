"""Lesson notes -> a weekly digest per teacher and class, and the internal "what didn't work per lesson" backlog
(24 Oct 2026, the Jesuit).

Chris, 24 Oct 2026: "the notes could then be analyzed by the AI (no personal notes, just about what worked, didn't
work, ideas, kids that need special attention etc)". DECISIONS 24 Oct (route settled): Tonga goes through the
translator; a STANDARD, untouched chat model (Gemma 4 E4B -it on llama.cpp) reads English. It reads, sorts and picks;
it never writes facts or Tonga.

THE STEPS (per teacher and class, one week Monday-Sunday):
  1. Tonga in a note -> English by the served translator (translator_service, toi>en), cached on the note (text_en,
     text_en_by). A note that reads as Tonga is translated whole; an English note with a quoted Tonga span gets the
     span's translation beside it. The note itself is never changed.
  2. The standard model reads the notes (each with a short id N1..Nk, its date, lesson and the teacher's tags) under a
     constrained JSON schema: {worked[], didnt[], ideas[], attention[], repeat[], themes[]}, each item
     {text, notes[ids]} with ids ONLY from the list (enum) and at least one per item.
  3. Code checks every item: it must cite >= 1 listed note; for a category item, at least one cited note must carry
     that tag or say it in so many words (the tag phrases of planner.TAG_RX); and the item must share a content word
     with its cited notes (a cheap invention alarm). An item failing any check is DROPPED and counted (no claim
     without a source note). Notes no kept item cites are listed under "not summarised", quoted as written.
  4. Each 'didnt' and 'attention' item is matched by retrieval (planner.SupportIndex: the plug-in's strategies and
     teaching tips) to at most five candidates; the model may pick ONE id from them or none; it never names a new
     strategy. Without the model, the top candidate stands only above a score threshold.
  5. The digest: short English lines marked "AI summary", each with its note ids; the matches; the dropped count.
  6. The aggregate across teachers, by lesson: notes tagged "didn't work" (plus the didnt items' texts), names
     stripped (planner.probable_names), 'attention' notes about a learner never read, teachers as short hashes ->
     internal/chitonga_train/harness/notes_backlog.json (INTERNAL: our content backlog, never served).

Runs on the Beelink as a heavy job under tools/chitonga/llm/beelink_lock.py ("notes-digest"): on demand
(`--teacher ... [--class ...] [--week YYYY-MM-DD]`, or POST /api/digest/run on the local server) and nightly
(`--nightly`; runtime/install_notes_task.ps1 registers it at 23:40, not installed by this build).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, "..", ".."))
TOOLS = os.path.normpath(os.path.join(HARNESS, ".."))
REPO = os.path.normpath(os.path.join(TOOLS, "..", ".."))
for p in (HARNESS, os.path.join(TOOLS, "llm")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.memory import norm_number  # noqa: E402
from core.retrieval import L  # noqa: E402
from core.school_calendar import d_, monday  # noqa: E402
from plugins.education.planner import TAG_RX, SupportIndex, probable_names  # noqa: E402

CATS = ("worked", "didnt", "ideas", "attention", "repeat", "themes")
CAT_TAG = {"worked": "worked", "didnt": "didnt", "ideas": "idea", "attention": "attention", "repeat": "repeat"}
AI_LABEL = "AI summary: a standard model sorted your notes; every line links to the notes it came from."
NO_MODEL_LABEL = "From your tags (no AI summary this week): your notes, grouped as you tagged them."
BACKLOG = os.path.join(REPO, "internal", "chitonga_train", "harness", "notes_backlog.json")
LLAMA = os.environ.get("CHITONGA_LLAMA_SERVER", r"llama.cpp\vulkan\llama-server.exe")
GGUF = os.environ.get("NOTES_GEMMA_GGUF", r"models\base\gemma-4-E4B-it-Q4_K_M.gguf")
GEMMA_URL = os.environ.get("ASSISTANT_GEMMA_URL", "http://127.0.0.1:8091")
MATCH_MIN = 3.0            # retrieval score the top candidate needs to stand without the model's pick
GENERIC = set("""lesson lessons class learner learners pupil pupils child children today week time well good work
worked didn work need needs again idea ideas teacher one two three day did not""".split())

SYSTEM = (
    "You sort a primary school teacher's own lesson notes from one week into a short weekly digest for that teacher. "
    "You do not add advice, facts or anything the notes do not say. Write in plain English, one short sentence per "
    "item, saying only what the cited notes say. Every item lists the ids of the notes it comes from (only ids from the "
    "list). Sections:\n"
    "worked = what went well; didnt = what did not work; ideas = ideas the teacher wrote for next time; attention = "
    "the class or a learner who needs attention (keep the name as written); repeat = what the teacher wants to teach "
    "again; themes = at most two patterns that run across several notes.\n"
    "Leave a section empty when no note supports it. Reply with ONE JSON object only.")


# ------------------------------------------------------------------------------------------------ translation
class Translator:
    """the served translator, toi>en, loaded on first use (CTranslate2 readers: seconds per note on the Beelink)"""

    def __init__(self, service=None):
        self._svc = service
        self.name = "translator_service (served, toi>en)"

    def svc(self):
        if self._svc is None:
            import translator_service as TS  # noqa: PLC0415
            self._svc = TS.Service()
        return self._svc

    @staticmethod
    def is_tonga(text):
        try:
            import translator_service as TS  # noqa: PLC0415
            return TS.detect_direction(text) == "toi>en"
        except Exception:  # noqa: BLE001
            return False

    def translate(self, text):
        r = self.svc().translate(text, "toi>en")
        return (r.get("text") or "").strip(), list(r.get("alarms") or [])

    def note_english(self, text):
        """(english, lang, how): whole-Tonga notes translated; quoted Tonga spans in an English note glossed beside"""
        t = text or ""
        if not t.strip():
            return t, "en", None
        if self.is_tonga(t):
            en, alarms = self.translate(t)
            return en or t, "toi", "translator" + (" (alarms: " + ", ".join(alarms)[:80] + ")" if alarms else "")
        spans = re.findall(r"['\"‘“]([^'\"’”]{8,200})['\"’”]", t)
        out, done = t, False
        for sp in spans:
            if self.is_tonga(sp):
                en, _ = self.translate(sp)
                if en:
                    out = out.replace(sp, f"{sp} [Tonga; translator: {en}]", 1)
                    done = True
        return out, ("mixed" if done else "en"), ("translator (quoted Tonga)" if done else None)


# ------------------------------------------------------------------------------------------------ the model
class Gemma:
    """the untouched Gemma 4 E4B -it on llama.cpp (OpenAI endpoint). start() launches the Vulkan server when nothing
    answers on the port and stop() ends only a server this object started."""

    def __init__(self, url=GEMMA_URL, post=None, timeout=300):
        self.url = url.rstrip("/")
        self.post_fn = post
        self.timeout = timeout
        self.proc = None
        self.name = "gemma-4-E4B-it Q4_K_M (untouched, llama.cpp)"
        self.calls = []

    def up(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as f:
                return f.status == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self, wait_s=240):
        if self.post_fn or self.up():
            return True
        port = self.url.rsplit(":", 1)[-1]
        log = open(os.path.join(os.path.dirname(GGUF), "..", "..", "llama_server_8091.log"), "a", encoding="utf-8")
        self.proc = subprocess.Popen([LLAMA, "-m", GGUF, "-ngl", "99", "-c", "8192", "--host", "127.0.0.1",
                                      "--port", port, "-np", "1"], stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.up():
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(3)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.proc = None

    def chat(self, messages, schema, max_tokens=900):
        body = {"messages": messages, "temperature": 0, "max_tokens": max_tokens, "seed": 1,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}}
        t0 = time.time()
        if self.post_fn:
            raw = self.post_fn(body)
        else:
            req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as f:
                raw = json.loads(f.read())["choices"][0]["message"]["content"]
        self.calls.append(round(time.time() - t0, 1))
        from core.reader_model import parse  # noqa: PLC0415
        return parse(raw), raw


def _digest_shape_ok(res):
    """True when a parsed model reply has digest_schema's shape: an object whose sections are lists of objects with a
    text string and a list of note-id strings. A server that does not enforce the schema can return strings or nested
    lists instead (reader bake-off, 24 Oct 2026); such a reply is skipped whole, never half-read."""
    if not isinstance(res, dict):
        return False
    for cat in CATS:
        items = res.get(cat, [])
        if items is None:
            continue
        if not isinstance(items, list):
            return False
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("text", ""), str):
                return False
            notes = it.get("notes", [])
            if not isinstance(notes, list) or not all(isinstance(x, str) for x in notes):
                return False
    return True


def digest_schema(ids):
    item = {"type": "object", "properties": {"text": {"type": "string", "maxLength": 200},
                                             "notes": {"type": "array", "items": {"type": "string", "enum": list(ids)},
                                                       "minItems": 1, "maxItems": 6}},
            "required": ["text", "notes"], "additionalProperties": False}
    props = {c: {"type": "array", "items": item, "maxItems": 2 if c == "themes" else 5} for c in CATS}
    return {"type": "object", "properties": props, "required": list(CATS), "additionalProperties": False}


def pick_schema(ids):
    return {"type": "object", "properties": {"pick": {"anyOf": [{"type": "string", "enum": list(ids)},
                                                                {"type": "null"}]}},
            "required": ["pick"], "additionalProperties": False}


def quick_model(gemma):
    """the quick-add hook for planner.parse_quick: the standard model picks a class id, a lesson id and tags ONLY from
    the lists it is given (enum schema); it writes nothing else"""
    from core.timetable import TAGS  # noqa: PLC0415

    def pick(text_en, classes, cands):
        cids = [c["id"] for c in classes]
        schema = {"type": "object", "properties": {
            "class_id": {"anyOf": [{"type": "integer", "enum": cids}, {"type": "null"}]} if cids else {"type": "null"},
            "item": {"anyOf": [{"type": "string", "enum": list(cands)}, {"type": "null"}]} if cands else {"type": "null"},
            "tags": {"type": "array", "items": {"type": "string", "enum": list(TAGS)}, "maxItems": 5}},
            "required": ["class_id", "item", "tags"], "additionalProperties": False}
        cl = "\n".join(f"- {c['id']}: {c['label']} (Grade {', '.join(map(str, c['grades']))})" for c in classes)
        it = "\n".join(f"- {i}" for i in cands) or "- (none)"
        msgs = [{"role": "system", "content": "You read a teacher's one-line lesson note and pick: the class it is "
                                              "about (an id from the list, or null), the lesson (an id from the list, "
                                              "or null), and the tags it shows: worked, didnt (did not work), idea, "
                                              "attention (a learner or the class needs attention), repeat. JSON only."},
                {"role": "user", "content": f"Note: {text_en}\nClasses:\n{cl}\nLessons:\n{it}"}]
        r, _ = gemma.chat(msgs, schema, max_tokens=80)
        return r or {}
    return pick


# ------------------------------------------------------------------------------------------------ building blocks
def note_line(sid, n, P):
    tags = json.loads(n.get("tags") or "[]")
    lesson = (P.items.get(n.get("item") or "") or {}).get("title")
    day = d_(n.get("slot_date") or n["created_at"][:10])
    bits = [f"[{sid}]", day.strftime("%a %d %b")]
    if n.get("subject"):
        bits.append(n["subject"])
    if lesson:
        bits.append(f"lesson: {lesson}")
    if tags:
        bits.append("tags: " + ", ".join("didn't work" if t == "didnt" else t for t in tags))
    if n.get("attention"):
        bits.append(f"attention: {n['attention']}")
    return " | ".join(bits) + " | note: " + (n.get("text_en") or n.get("text") or "(no text)")


def words(t):
    return {w for w in L.content_words(t or "") if w.lower() not in GENERIC}


SUBJECT_WORDS = {"maths": "maths math mathematics numeracy numbers", "science": "science",
                 "english": "english literacy reading", "chitonga": "chitonga tonga", "cts": "creative technology art"}


def supported(cat, item, cited, P=None):
    """(ok, reason): the code's check of one model item against the notes it cites. The shared-word alarm counts the
    notes' words, their subject's names ('maths' for a maths note: the 20-note test's two false drops) and their
    lesson's title."""
    if not cited:
        return False, "cites no listed note"
    if cat != "themes":
        tag = CAT_TAG[cat]
        rx = TAG_RX.get(tag)
        if not any(tag in json.loads(n.get("tags") or "[]") or (rx and rx.search(n.get("text_en") or n.get("text")
                                                                               or "")) for n in cited):
            return False, f"no cited note is tagged or says '{tag}'"
    nw = set()
    for n in cited:
        nw |= words(n.get("text_en") or n.get("text"))
        lesson = n.get("item")
        if lesson:
            nw |= words(lesson.replace("-", " ").replace(".", " "))
            if P is not None:
                nw |= words((P.items.get(lesson) or {}).get("title"))
        if n.get("subject"):
            nw |= words(SUBJECT_WORDS.get(n["subject"], n["subject"]))
    if not words(item.get("text")) & nw:
        return False, "shares no content word with its notes"
    return True, ""


def teacher_hash(number):
    return hashlib.sha256(("zamai-notes|" + norm_number(number)).encode()).hexdigest()[:8]


def redact(text):
    t = text or ""
    for nm in probable_names(t):
        t = re.sub(r"(?<![A-Za-z])" + re.escape(nm) + r"(?![A-Za-z])", "[learner]", t)
    return t


# ------------------------------------------------------------------------------------------------ the job
class DigestJob:
    def __init__(self, P, mem, tt, model=None, translator=None, support=None, lock=True, log=None):
        self.P, self.mem, self.tt = P, mem, tt
        self.model = model
        self.tr = translator
        self.support = support or SupportIndex(P)
        self.lock = lock
        self.log = log or (lambda *a: None)

    def _translate(self, notes):
        n_done = 0
        for n in notes:
            if n.get("text") and not n.get("text_en_by") and self.tr is not None:
                try:
                    en, lang, how = self.tr.note_english(n["text"])
                except Exception as e:  # noqa: BLE001
                    self.log(f"translator failed on {n['id']}: {e}")
                    continue
                if how:
                    self.tt.set_translation(n["id"], en, lang, how)
                    n.update(text_en=en, text_lang=lang, text_en_by=how)
                    n_done += 1
                else:
                    self.tt.set_translation(n["id"], None, "en", "none needed")
                    n.update(text_en_by="none needed")
        return n_done

    def digest(self, teacher, class_id, week):
        """build, check and store one teacher-class-week digest; returns it"""
        week = monday(d_(week))
        end = week + dt.timedelta(days=6)
        cls = self.mem.get_class(int(class_id))
        notes = [n for n in self.tt.notes(teacher, class_id=class_id, since=week.isoformat(), until=end.isoformat())
                 if (n.get("text") or json.loads(n.get("tags") or "[]")) and not n.get("withheld")]
        out = {"teacher_hash": teacher_hash(teacher), "class": {"id": cls["id"], "label": cls["label"]},
               "week": week.isoformat(), "week_end": end.isoformat(), "n_notes": len(notes),
               "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "sections": {c: [] for c in CATS}, "matches": [], "dropped": [], "not_summarised": [],
               "label": AI_LABEL, "ai": True, "model": None, "translated": 0}
        if not notes:
            out.update(label="No notes this week.", ai=False)
            self.tt.put_digest(teacher, class_id, week.isoformat(), out)
            return out
        out["translated"] = self._translate(notes)
        sid = {f"N{i + 1}": n for i, n in enumerate(notes)}
        rid = {n["id"]: s for s, n in sid.items()}
        res = None
        if self.model is not None:
            lines = "\n".join(note_line(s, n, self.P) for s, n in sid.items())
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Class: {cls['label']} (Grade {', '.join(map(str, cls['grades']))}). "
                                                f"Week of {week.isoformat()}. Notes:\n{lines}"}]
            try:
                res, raw = self.model.chat(msgs, digest_schema(list(sid)))
                out["model"] = self.model.name
                out["raw"] = raw
            except Exception as e:  # noqa: BLE001
                self.log(f"model failed: {e}")
                res = None
            # 24 Oct 2026 (reader bake-off: without schema enforcement a reader's reply with string items crashed this
            # job): a reply that is not the schema's shape is never trusted; the model digest is skipped, logged, and
            # the teacher's own tagged notes are shown instead (the no-model path below)
            if res is not None and not _digest_shape_ok(res):
                self.log(f"model digest skipped for class {class_id} week {week.isoformat()}: reply not in the schema's "
                         f"shape ({type(res).__name__}); tagged notes shown instead")
                out["skipped"] = "model reply not in the schema's shape"
                res = None
        if res is None:
            out.update(label=NO_MODEL_LABEL, ai=False)
            for s, n in sid.items():
                for t in json.loads(n.get("tags") or "[]"):
                    cat = {"idea": "ideas"}.get(t, t)
                    out["sections"][cat].append({"text": (n.get("text") or "").strip()[:200] or "(tag only)",
                                                 "notes": [n["id"]], "ai": False, "quoted": True})
        else:
            for cat in CATS:
                for it in (res.get(cat) or [])[:5]:
                    ids = [x for x in dict.fromkeys(it.get("notes") or []) if x in sid]
                    cited = [sid[x] for x in ids]
                    ok, why = supported(cat, it, cited, self.P)
                    rec = {"text": str(it.get("text") or "").strip()[:200], "notes": [sid[x]["id"] for x in ids],
                           "ai": True}
                    if ok and rec["text"]:
                        out["sections"][cat].append(rec)
                    else:
                        out["dropped"].append(dict(rec, section=cat, why=why or "empty text"))
        cited_all = {i for c in CATS for it in out["sections"][c] for i in it["notes"]}
        out["not_summarised"] = [{"id": n["id"], "text": n.get("text"), "tags": json.loads(n.get("tags") or "[]")}
                                 for n in notes if n["id"] not in cited_all]
        out["matches"] = self._match(out, {n["id"]: n for n in notes})
        out["note_refs"] = {n["id"]: rid[n["id"]] for n in notes}
        self.tt.put_digest(teacher, class_id, week.isoformat(), out)
        return out

    def _mistake_text(self, c):
        st = (self.P.strategies or {}).get(c.get("lesson")) or {}
        m = next((x for x in st.get("mistakes", []) if x["id"] == c["id"]), None)
        return (m.get("text") or "").strip()[:160] if m else ""

    def _match(self, out, by_id):
        matches = []
        for cat in ("didnt", "attention"):
            for it in out["sections"][cat]:
                cited = [by_id[i] for i in it["notes"] if i in by_id]
                lesson = next((n.get("item") for n in cited if n.get("item")), None)
                subject = next((n.get("subject") for n in cited if n.get("subject")), None)
                notes_en = [(n.get("text_en") or n.get("text") or "") for n in cited]
                q = it["text"] + " " + " ".join(notes_en)
                # 24 Oct: own lesson first, same subject only, >= 2 shared words from elsewhere (planner.SupportIndex)
                cands = self.support.match(q, lesson=lesson, subject=subject)
                if not cands:
                    continue
                pick, how = None, None
                if self.model is not None:
                    ids = [c["id"] for c in cands]
                    listing = "\n".join(
                        f"- {c['id']}: " + (f"common mistake in the lesson '{(self.P.items.get(c['lesson']) or {}).get('title')}'"
                                            f": {c['title']}. {self._mistake_text(c)} Fix: {c.get('fix')}"
                                            if c["kind"] == "strategy" else f"teaching tip: {c['title']} (when: "
                                                                            f"{c.get('when')})") for c in cands)
                    les = (self.P.items.get(lesson) or {}).get("title") if lesson else None
                    msgs = [{"role": "system", "content": "You pick the ONE listed item that answers the problem in the "
                                                          "teacher's note, or null when none really fits (a shared "
                                                          "word is not enough). Reply with JSON only."},
                            {"role": "user", "content": f"Problem: {it['text']}\nThe teacher's note: "
                                                        f"{' / '.join(notes_en)}\n"
                                                        + (f"Lesson: {les}\n" if les else "")
                                                        + (f"Subject: {subject}\n" if subject else "")
                                                        + f"Items:\n{listing}"}]
                    try:
                        r, _ = self.model.chat(msgs, pick_schema(ids), max_tokens=60)
                        pid = (r or {}).get("pick")
                        pick = next((c for c in cands if c["id"] == pid), None)
                        how = "model pick among retrieved" if pick else "model: none fits"
                    except Exception as e:  # noqa: BLE001
                        self.log(f"pick failed: {e}")
                if pick is None and how is None:
                    b = self.support.best(q, lesson=lesson, subject=subject, min_score=MATCH_MIN)
                    if b:
                        pick, how = b, "retrieval (no model)"
                if pick:
                    matches.append({"section": cat, "for": it["text"], "notes": it["notes"], "match": pick,
                                    "how": how, "candidates": [c["id"] for c in cands]})
        return matches

    # ------------------------------------------------------------------------------------------------ batch
    def run(self, teachers=None, week=None, class_id=None):
        """digests for every (teacher, class) with notes in the week (or the given ones)"""
        week = monday(d_(week or dt.date.today()))
        end = week + dt.timedelta(days=6)
        pairs = set()
        for n in self.tt.all_notes(week.isoformat(), end.isoformat()):
            if n.get("class_id") and (not teachers or n["teacher"] in {norm_number(t) for t in teachers}):
                if class_id is None or n["class_id"] == int(class_id):
                    pairs.add((n["teacher"], n["class_id"]))
        out = []
        for t, c in sorted(pairs):
            try:
                out.append(self.digest(t, c, week))
            except Exception as e:  # noqa: BLE001  (24 Oct 2026: one bad digest never stops the others)
                self.log(f"digest skipped for teacher {teacher_hash(t)} class {c} week {week.isoformat()}: {e!r}"[:300])
        return out

    def aggregate(self, since, until, testers=(), path="default"):
        """what didn't work, per lesson, across teachers: names stripped, learner-attention notes never read"""
        testers = {norm_number(t) for t in testers}
        by = {}
        unlinked = []
        n_read = 0
        for n in self.tt.all_notes(since, until):
            tags = json.loads(n.get("tags") or "[]")
            if "didnt" not in tags or n.get("attention") == "learner" or n.get("withheld"):
                continue
            n_read += 1
            txt = redact(n.get("text_en") or n.get("text") or "")
            rec = {"teacher": teacher_hash(n["teacher"]), "date": n.get("slot_date") or n["created_at"][:10],
                   "text": txt[:300], "tags": [x for x in tags if x != "attention"],
                   "test_only": n["teacher"] in testers or None}
            if n.get("item") and n["item"] in self.P.items:
                by.setdefault(n["item"], []).append(rec)
            else:
                unlinked.append(dict(rec, subject=n.get("subject")))
        lessons = []
        for iid, recs in by.items():
            it = self.P.items[iid]
            real = [r for r in recs if not r["test_only"]]
            lessons.append({"item": iid, "title": it["title"], "grades": it.get("grades"),
                            "subject": it.get("subject") or (self.P.units.get(self.P.unit_of.get(iid) or "") or {})
                            .get("subject"), "n_notes": len(real), "n_teachers": len({r["teacher"] for r in real}),
                            "has_strategy": iid in self.P.strategies, "notes": recs})
        lessons.sort(key=lambda r: (-r["n_teachers"], -r["n_notes"], r["item"]))
        res = {"built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "since": since,
               "until": until, "n_notes_read": n_read, "lessons": lessons, "unlinked": unlinked,
               "label": "INTERNAL content backlog: what teachers noted as not working, per lesson. Names stripped "
                        "(planner.probable_names), notes about one learner never read, teachers as hashes. Never "
                        "served, never published."}
        path = BACKLOG if path == "default" else path      # module value read at call time (tests redirect it)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
        return res


def _testers():
    try:
        sys.path.insert(0, HARNESS)
        from adapter.config import Config  # noqa: PLC0415
        return [e["number"] for e in Config().allowlist().values() if e.get("tester")]
    except Exception:  # noqa: BLE001
        return []


NIGHT_MARK = os.path.join(REPO, "internal", "chitonga_train", "harness", "notes_digest_nightly.json")
NIGHT_LOG = os.path.join(REPO, "internal", "chitonga_train", "harness", "notes_digest_nightly.log")


def try_lock(name, stale_h=4):
    """take the Beelink heavy-job lock WITHOUT waiting: True if taken, False if another job holds it (the nightly run
    then skips and the task retries at its next trigger). Same file, format and stale rule as beelink_lock.acquire."""
    import beelink_lock as BL  # noqa: PLC0415
    for _ in range(2):
        try:
            fd = os.open(BL.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"by": name, "at": time.time()}).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                with open(BL.LOCK, encoding="utf-8") as f:
                    d = json.load(f)
                if time.time() - d.get("at", 0) > stale_h * 3600:
                    os.remove(BL.LOCK)
                    continue
            except Exception:  # noqa: BLE001
                pass
            return False
    return False


def lock_holder():
    try:
        import beelink_lock as BL  # noqa: PLC0415
        with open(BL.LOCK, encoding="utf-8") as f:
            return json.load(f).get("by")
    except Exception:  # noqa: BLE001
        return None


def night_log(msg):
    os.makedirs(os.path.dirname(NIGHT_LOG), exist_ok=True)
    with open(NIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(dt.datetime.now().isoformat(timespec="seconds") + " " + msg + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="memory SQLite (default: the adapter's pilot_memory.sqlite)")
    ap.add_argument("--teacher", action="append", help="number(s); default: every teacher with notes")
    ap.add_argument("--class", dest="class_id", type=int)
    ap.add_argument("--week", help="any date in the week (default: this week)")
    ap.add_argument("--nightly", action="store_true",
                    help="the scheduled run: this week so far (plus last week on a Monday night) and the 28-day "
                         "backlog; takes the lock WITHOUT waiting (held -> skipped and logged; the task retries at "
                         "02:40 and 05:10); once per night (notes_digest_nightly.json)")
    ap.add_argument("--live", action="store_true", help="with --nightly: also the live demo's notes "
                                                       "(web/netlify_notes_sync.py), when ASSISTANT_ADMIN_KEY is set")
    ap.add_argument("--no-model", action="store_true", help="tags only, no AI summary")
    ap.add_argument("--no-translator", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args(argv)
    night = (dt.datetime.now() - dt.timedelta(hours=12)).date()      # 23:40, 02:40 and 05:10 are one night
    if a.nightly:
        try:
            with open(NIGHT_MARK, encoding="utf-8") as f:
                if json.load(f).get("night") == night.isoformat():
                    night_log(f"night {night}: already done")
                    return "done before"
        except (OSError, ValueError):
            pass
    from plugins import education  # noqa: PLC0415
    from core.memory import SQLiteMemory  # noqa: PLC0415
    from core.timetable import TimetableStore  # noqa: PLC0415
    if not a.db:
        from adapter.config import Config  # noqa: PLC0415
        a.db = Config().memory_db
    import beelink_lock  # noqa: PLC0415
    if not a.no_lock:
        if a.nightly:
            if not try_lock("notes-digest"):
                night_log(f"night {night}: SKIPPED, the Beelink lock is held by {lock_holder()}")
                print(f"skipped: the Beelink lock is held by {lock_holder()}", file=sys.stderr)
                return "skipped"
        else:
            beelink_lock.acquire("notes-digest")
    summary = {"night": night.isoformat(), "digests": 0, "items": 0, "dropped": 0}
    gem = None
    try:
        P, _ = education.load()
        mem = SQLiteMemory(a.db)
        tt = TimetableStore(mem)
        gem = None if a.no_model else Gemma()
        job = DigestJob(P, mem, tt, model=gem, translator=None if a.no_translator else Translator(),
                        log=lambda *x: print(*x, file=sys.stderr, flush=True))
        base = night if a.nightly else dt.date.today()
        weeks = [a.week or base.isoformat()]
        if a.nightly and base.weekday() == 0:
            weeks.append((base - dt.timedelta(days=7)).isoformat())
        if gem is not None and not gem.start():
            print("Gemma did not start: digests from tags only", file=sys.stderr)
            job.model = None
            summary["model"] = "did not start"
        for w in weeks:
            for d in job.run(a.teacher, w, a.class_id):
                summary["digests"] += 1
                summary["items"] += sum(len(v) for v in d["sections"].values())
                summary["dropped"] += len(d["dropped"])
                print(json.dumps({"class": d["class"]["label"], "week": d["week"], "notes": d["n_notes"],
                                  "items": sum(len(v) for v in d["sections"].values()), "dropped": len(d["dropped"]),
                                  "matches": len(d["matches"]), "ai": d["ai"]}))
        if a.nightly:
            r = job.aggregate((base - dt.timedelta(days=28)).isoformat(), base.isoformat(), testers=_testers())
            summary["backlog_lessons"] = len(r["lessons"])
            print(f"backlog: {len(r['lessons'])} lessons, {r['n_notes_read']} notes -> {BACKLOG}")
        if gem is not None:
            gem.stop()
        if a.nightly and a.live:
            sys.path.insert(0, os.path.join(HARNESS, "web"))
            import netlify_notes_sync as NS  # noqa: PLC0415
            if NS.admin_key(required=False):
                try:
                    NS.main(["run", "--no-lock", "--week", base.isoformat()] + (["--no-model"] if a.no_model else []))
                    summary["live"] = "done"
                except Exception as e:  # noqa: BLE001
                    summary["live"] = f"failed: {str(e)[:120]}"
            else:
                summary["live"] = "skipped: no ASSISTANT_ADMIN_KEY"
    finally:
        if gem is not None:
            gem.stop()
        if not a.no_lock:
            beelink_lock.release("notes-digest")
    if a.nightly:
        with open(NIGHT_MARK, "w", encoding="utf-8") as f:
            json.dump(dict(summary, done=dt.datetime.now().isoformat(timespec="seconds")), f)
        night_log(f"night {night}: done {json.dumps(summary)}")
    return summary


if __name__ == "__main__":
    main()
