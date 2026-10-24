"""Per-teacher MEMORY for the teacher assistant (20 Oct 2026, the Jesuit; HARNESS_V3_BUILD.md switchboard design;
coordinator for Chris: like a business responder remembers customers). Deterministic store, never in the model's weights.

One JSON file per teacher under internal/chitonga_train/harness/teachers/<id>.json, the id a salted sha256 of the phone
number (the number itself is not written into the file name). It holds:
  profile        TEACHER_PROFILE.md fields, with later corrections (each change dated)
  term           where they are: topics asked about, with dates and the module page we cited
  difficulties   learner difficulties they reported (topic level), the interventions suggested, their feedback
  preferences    language of explanations, answer length
  feedback_log   their feedback lines, also appended to harness/teacher_feedback.jsonl as CANDIDATE training / evaluation
                 data: one teacher's "this worked" informs the next teacher on the same topic only after verification
Rules (binding): no learner names or personal data about children: free text is screened (capitalised names other
than the first word, phone numbers and ages are cut before anything is stored); the teacher can see the whole memory
(view) and have it wiped (wipe).
"""
import datetime as dt
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DIR = os.path.join(REPO, "internal", "chitonga_train", "harness", "teachers")
FEEDBACK = os.path.join(REPO, "internal", "chitonga_train", "harness", "teacher_feedback.jsonl")
SALT = "zamai-teachers-v1"


def tid(phone):
    digits = re.sub(r"\D", "", str(phone or ""))
    return hashlib.sha256((SALT + digits).encode()).hexdigest()[:16]


def _path(phone):
    return os.path.join(DIR, tid(phone) + ".json")


def _now():
    return dt.datetime.now().isoformat(timespec="minutes")


def scrub(text):
    """topic level only: cut what could identify a child (names inside a sentence, phone numbers, ages)"""
    t = text or ""
    t = re.sub(r"\+?\d[\d\s-]{7,}\d", "[number]", t)
    t = re.sub(r"\b\d{1,2}\s*(years?|yrs?)\s*old\b", "[age]", t, flags=re.I)
    words = t.split()
    out = []
    for i, w in enumerate(words):
        prev = words[i - 1] if i else ""
        if i and w[:1].isupper() and not prev.endswith((".", "!", "?", ":")) and w.strip(",.;:!?").lower() not in KEEP_CAPS:
            out.append("[name]")
        else:
            out.append(w)
    return " ".join(out)


KEEP_CAPS = {"i", "english", "chitonga", "tonga", "grade", "form", "term", "mathematics", "maths", "science", "cts",
             "monday", "tuesday", "wednesday", "thursday", "friday", "zambia", "zambian", "ok"}


def load(phone):
    p = _path(phone)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {"id": tid(phone), "created": _now(), "profile": None, "profile_history": [], "term": [], "difficulties": [],
            "preferences": {}, "feedback_log": []}


def save(phone, m):
    os.makedirs(DIR, exist_ok=True)
    m["updated"] = _now()
    json.dump(m, open(_path(phone), "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)


def set_profile(phone, profile, source="screening"):
    m = load(phone)
    if m["profile"] != profile:
        m["profile_history"].append({"at": _now(), "source": source, "profile": profile})
        m["profile"] = profile
    save(phone, m)
    return m


def add_topic(phone, request, task, cites):
    m = load(phone)
    m["term"].append({"at": _now(), "task": task, "request": scrub(request)[:200],
                      "pages": [c.get("cite") for c in (cites or {}).get("pages", [])][:3]})
    m["term"] = m["term"][-50:]
    save(phone, m)
    return m


def add_difficulty(phone, difficulty, intervention=None):
    m = load(phone)
    m["difficulties"].append({"at": _now(), "difficulty": scrub(difficulty)[:200], "intervention": intervention,
                              "feedback": None})
    save(phone, m)
    return m


FEEDBACK_RX = re.compile(r"(?i)\b(it worked|worked well|too long|too short|too hard|too easy|did not work|didn't work|"
                         r"learners liked|they liked|not useful|very useful|helpful)\b")


def add_feedback(phone, text):
    """a feedback line: kept with the last intervention, and logged as CANDIDATE data (verified before any reuse)"""
    m = load(phone)
    fb = {"at": _now(), "text": scrub(text)[:300]}
    m["feedback_log"].append(fb)
    if m["difficulties"] and m["difficulties"][-1].get("feedback") is None:
        m["difficulties"][-1]["feedback"] = fb["text"]
    save(phone, m)
    os.makedirs(os.path.dirname(FEEDBACK), exist_ok=True)
    with open(FEEDBACK, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"teacher": m["id"], **fb, "last_topic": (m["term"][-1] if m["term"] else None),
                            "status": "candidate, unverified"}, ensure_ascii=False) + "\n")
    return m


def failed_interventions(phone):
    m = load(phone)
    return [d["intervention"] for d in m["difficulties"] if d.get("intervention") and d.get("feedback")
            and re.search(r"(?i)did not work|didn't work|not useful|too hard|too long", d["feedback"])]


def set_appreciation(phone, key):
    """the last appreciation line used for this teacher (h3_front.appreciation never repeats it next time)"""
    m = load(phone)
    m["appreciation_last"] = key
    m.setdefault("appreciation_log", []).append({"at": _now(), "key": key})
    m["appreciation_log"] = m["appreciation_log"][-20:]
    save(phone, m)
    return m


def view(phone):
    """everything we keep about this teacher, as they would read it"""
    return load(phone)


def wipe(phone):
    p = _path(phone)
    if os.path.exists(p):
        os.remove(p)
    return True
