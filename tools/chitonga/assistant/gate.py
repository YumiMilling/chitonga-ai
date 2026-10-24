"""Step 2, the scope and safety gate. Deterministic, no model; the model never decides scope and never writes a fixed line.

ORDER (each step for a reason):
  1. safeguarding  h3_front.safeguarding on the ORIGINAL and on the English READING (a Tonga request is caught either
                   way). Outcome: the fixed referral below, never advice (AI_AGENT_POLICY §5; Children's Code 2022).
  2. school signal h3_front.scope on both texts: advice (a classroom situation), teacher_task, social.
  3. content question  NEW here: h3_front.scope sends "What's photosynthesis?", "How many provinces are in Zambia",
                   "How many metres make a kilometer" and "For each correct answer, mark out of 2." to out_of_scope
                   (checked 22 Oct on Fr Musunga's 11 frozen showcase requests: 6 of 11 redirected; "wash our hands"
                   passed only as "pasted Tonga material"). A teacher asking a content question IS the task Chris named
                   ("explaining a topic in Tonga", desk/T01_2026-10-22_answers.md). So: a question form (English or
                   Tonga) that hits no DENY pattern is in scope as 'explain'. The DENY list holds what Chris called out
                   of scope (DECISIONS 20 Oct: not a general chat buddy): news and party politics, money, personal
                   medical advice, jokes / chat about the assistant, weather forecasts, sport results.
                   Trade-off, said plainly: this gate is permissive for questions (a teacher's question is presumed to
                   be for teaching); the scope probes of the showcase set are where that is tested.
  4. else out_of_scope -> the fixed redirect (h3_front.redirect). When the request was Tonga with words we do not know,
     the honest outcome is the FRANK script (DECISIONS 20 Oct: uncertainty = say we do not know, ask), not a redirect.

Kinds in scope: explain | marking | translate | lesson_help | advice | social.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
import h3_front as F  # noqa: E402  (pure regex module: scope, safeguarding, the redirect, the social bank, appreciation)

# ------------------------------------------------------------------------------------------------ fixed lines
# THE REFERRAL, verified destinations (Codex primary-source research 21 Oct 2026,
# llm/council/codex/2026-10-21_1038_safeguarding.md; memory ref-zambia-safeguarding-referral). Order as in the memory:
# the school's own people today, Childline 116, police 991 for immediate danger, the District Social Welfare Office.
# h3_front.REFERRAL predates the verification and carries no number (its unit test asserts that); it is left as is and
# this line is the assistant's. Recheck the numbers before any public release. The Tonga waits for a teacher: never
# machine-rendered, never drafted by a model.
REFERRAL = {
    "en": ("This is about a child's safety, and I cannot give advice on it. Please tell your head teacher, the guidance "
           "and counselling teacher or the school's safeguarding focal person today. You can call Childline on 116 "
           "(free, 24 hours). If the child is in danger now, call the police on 991. The District Social Welfare "
           "Office can also help."),
    "toi": None,
    "status": "English only; the Tonga is to be written and confirmed by a teacher (desk item), never machine-rendered",
    "sources": ["clzambia.org/childline-116 (116, free, 24/7)", "zambiapolice.gov.zm/contact-us (991)",
                "Children's Code Act No. 12 of 2022 s.168(4), s.276", "chomacouncil.gov.zm Social Welfare"],
    "checked": "2026-10-21",
}
REFERRAL_LIVE = False

# THE FRANK SCRIPT (DECISIONS 20 Oct: "be frank that you don't know, ask for explanation"; Tonga only once a teacher
# has confirmed it). No Tonga is drafted here.
FRANK = {"en": "I do not know some of these words. Could you say it in other words?", "toi": None,
         "status": "English meaning only; the Tonga is to be written and confirmed by a teacher"}

# NO CHECKED CONTENT: a health topic with no approved card is never drafted by a model.
# Chris 22 Oct 2026: the no-material line must say "I have no checked material on this yet" (the first showcase run
# answered nine requests with FRANK's "I do not know some of these words", which was false: the words were known, the
# material was missing).
NO_CONTENT = {"en": "I have no checked material on this yet, so I will not guess. Please use the Ministry of Health "
                    "material or ask the health worker at the clinic.",
              "toi": None, "status": "English meaning only; Tonga to be confirmed by a teacher"}
NO_MATERIAL = {"en": "I have no checked material on this yet. If you send me the page you are teaching from, I will "
                     "explain it.",
               "toi": None, "status": "English meaning only; Tonga to be confirmed by a teacher"}

# ------------------------------------------------------------------------------------------------ patterns
Q_EN = re.compile(r"(?i)^\W*(what|what's|whats|why|how|which|where|when|who|whom|whose|give|list|name|explain|describe|"
                  r"define|state|compare|tell\s+me|can\s+you\s+explain|is\s+it\s+true)\b|\?\s*$|difference\s+between")
# Tonga question / explain words, each printed in our sources: pandulula 'explain', nkaambo nzi 'why', ino ... buti 'how', nzi 'what', sena / hena (yes-no question)
Q_TOI = re.compile(r"(?i)\b(\w*pandulul\w*|\w*pandulwid\w*|nkaambo\s+nzi|nzi|buti|ninzi|ncinzi|sena|hena|zyiindene)\b|\?\s*$")
MARK = re.compile(r"(?i)\b(mark(s|ing|ed)?\b|out\s+of\s+\d+|marking\s+(key|scheme|guide)|correct\s+answers?|score\w*|"
                  r"(was|is|were|are)\s+the\s+(learner|pupil|child|student)s?\s+able|rubric|assess\w*)")
TRANSLATE = re.compile(r"(?i)\b(translate|translation\s+of|how\s+do\s+(you|i|we)\s+say|(the\s+)?word\s+for)\b")
HEALTH = re.compile(r"(?i)(cholera|malaria|diarrh|dysenter|\bhiv\b|\baids\b|tubercul|\btb\b|vaccin|medicine|\bdrugs?\b|"
                    r"fever|\bsick|illness|disease|infect|germs?\b|hygien|wash\w*\s+(our\s+|your\s+|their\s+|the\s+)?hands|"
                    r"sanitation|latrine|toilet|nutrition|anaemi|anemi|pregnan|heat\s*stroke|dehydrat|first\s+aid|"
                    r"snake\s*bite|bilharzia|worms|kusoomona|bulwazi|kkolela)")
DENY = [
    ("news_politics", re.compile(r"(?i)\b(latest\s+news|in\s+the\s+news|who\s+will\s+win|which\s+party|vote\s+for|campaign\w*|"
                                 r"(the\s+)?president\s+said|upnd|\bpf\b|hichilema|lungu)\b")),
    ("money", re.compile(r"(?i)\b(loan|borrow\s+money|salary|payslip|my\s+pay|allowance|exchange\s+rate|betting|bet\s+on|"
                         r"lottery|invest(ment)?s?|business\s+idea|make\s+money|"
                         # 24 Oct 2026 (v0.14 held-out run): 'Lend me some money', 'Can you lend me K500', 'Sell me airtime'
                         # passed; a request FOR money or airtime, never the money lesson ('teach my learners about money')
                         r"lend\s+me|borrow\s+(some\s+)?(money|k\d+)|give\s+me\s+(some\s+)?money|send\s+me\s+(some\s+)?money|"
                         # Tonga, the same request (24 Oct: 'Ndipe mali' was served the money LESSON): the phrase
                         # 'give me money' only, never the word mali, which the Grade 1-3 money lessons need
                         r"ndipe\s+mali|ndipa\s+mali|amundipe\s+mali|kondipa\s+mali|"
                         r"airtime|\bk\d{2,}\b)\b")),
    ("personal_health", re.compile(r"(?i)\b(my|i\s+am|i'm|i\s+have|my\s+(child|son|daughter|wife|husband))\b[^.?!]{0,40}\b("
                                   r"sick|ill|fever|pain|headache|cough|diarrh\w*|vomit\w*|medicine|tablets?)\b|"
                                   r"\b(dose|dosage|how\s+many\s+tablets|prescri\w*)\b")),
    ("chat", re.compile(r"(?i)\b(joke|funny\s+story|tell\s+me\s+about\s+yourself|who\s+(made|are)\s+you|are\s+you\s+(a\s+)?"
                        r"(robot|human|real)|girlfriend|boyfriend|love\s+me|marry\s+me)\b")),
    ("weather_forecast", re.compile(r"(?i)(weather|rain)\s+(today|tomorrow|this\s+week)|will\s+it\s+rain|forecast")),
    ("sport", re.compile(r"(?i)\b(score\s+of\s+the\s+match|who\s+won\s+the\s+(match|game)|chipolopolo)\b")),
]
# the advice cards (internal/chitonga_train/advice/cards) by h3_front.SITUATIONS key
SITUATION_CARD = {"large_class": "01", "no_textbooks": "02", "mixed_ability": "03", "discipline": "04",
                  "absenteeism": "05", "parents": "07"}
HUNGRY = re.compile(r"(?i)\b(hungry|hunger|no\s+breakfast|tired\s+learners|learners?\s+(are|is)\s+tired)\b")
MULTIGRADE = re.compile(r"(?i)\bmulti-?grade|two\s+grades|three\s+grades|grades?\s+\d\s+and\s+\d\s+(together|in\s+one)")


def deny(text):
    for name, rx in DENY:
        m = rx.search(text or "")
        if m:
            return name, m.group(0)
    return None, ""


def advice_card(text):
    """the advice card number for a classroom situation, or None"""
    if MULTIGRADE.search(text or ""):
        return "10"
    if HUNGRY.search(text or ""):
        return "06"
    for k in F.situation(text or ""):
        if k in SITUATION_CARD:
            return SITUATION_CARD[k]
    return None


def classify(original, reading, lang="en", unknown=()):
    """{'outcome': 'safeguarding' | 'out_of_scope' | 'frank' | 'in_scope', 'kind', 'why', 'health': bool,
    'advice_card'}. original = the teacher's text, reading = its English reading (== original for English)."""
    texts = [t for t in dict.fromkeys([original or "", reading or ""]) if t]
    for t in texts:
        s = F.safeguarding(t)
        if s:
            return {"outcome": "safeguarding", "kind": None, "why": "child safeguarding: " + s, "health": False}
    both = " \n ".join(texts)
    health = bool(HEALTH.search(both))
    scopes = [F.scope(t) for t in texts]
    lanes = [s[0] for s in scopes]
    asked = bool(Q_EN.search(reading or "") or Q_EN.search(original or "") or Q_TOI.search(original or ""))
    if "advice" in lanes:
        card = advice_card(both)
        return {"outcome": "in_scope", "kind": "advice", "why": scopes[lanes.index("advice")][1], "health": health,
                "advice_card": card}
    if MARK.search(both):
        return {"outcome": "in_scope", "kind": "marking", "why": "a marking request", "health": health}
    if TRANSLATE.search(both) and "teacher_task" in lanes:
        return {"outcome": "in_scope", "kind": "translate", "why": "a translation request", "health": health}
    school = "teacher_task" in lanes and not all(s[1].startswith("pasted Tonga material") for s in scopes
                                                 if s[0] == "teacher_task")
    if asked:
        d, hit = deny(reading) if not school else (None, "")
        if d:
            return {"outcome": "out_of_scope", "kind": None, "why": f"{d}: '{hit}'", "health": health}
        return {"outcome": "in_scope", "kind": "explain",
                "why": ("a question with a school signal" if school else
                        "a content question (no school word; in scope as explaining a topic, gate.py step 3)"),
                "health": health}
    if school or "teacher_task" in lanes:
        return {"outcome": "in_scope", "kind": "lesson_help", "why": scopes[lanes.index("teacher_task")][1],
                "health": health}
    if "social" in lanes:
        return {"outcome": "in_scope", "kind": "social", "why": scopes[lanes.index("social")][1], "health": False}
    if lang in ("toi", "mixed") and unknown:
        return {"outcome": "frank", "kind": None, "why": "Tonga request with unknown words: " + ", ".join(unknown),
                "health": health}
    return {"outcome": "out_of_scope", "kind": None, "why": scopes[0][1] if scopes else "empty", "health": health}


def referral(why=""):
    live = REFERRAL_LIVE and REFERRAL.get("toi")
    return {"reply": REFERRAL["toi"] if live else None, "en_meaning": REFERRAL["en"], "live": bool(live),
            "status": REFERRAL["status"], "sources": REFERRAL["sources"], "why": why, "train": False}


def redirect(why=""):
    return F.redirect(why)


def frank(why=""):
    return {"reply": None, "en_meaning": FRANK["en"], "live": False, "status": FRANK["status"], "why": why,
            "train": False}


def no_content(why=""):
    return {"reply": None, "en_meaning": NO_CONTENT["en"], "live": False, "status": NO_CONTENT["status"], "why": why,
            "train": False}


UNCLEAR = {"en": "I am not sure I understood your question. Could you say it in other words?", "toi": None,
           "status": "English meaning only; the Tonga is to be written and confirmed by a teacher"}


def unclear(why=""):
    return {"reply": None, "en_meaning": UNCLEAR["en"], "live": False, "status": UNCLEAR["status"], "why": why,
            "train": False}


def no_material(why=""):
    return {"reply": None, "en_meaning": NO_MATERIAL["en"], "live": False, "status": NO_MATERIAL["status"], "why": why,
            "train": False}


def social(message):
    return F.social_reply(message)


def warm_line(message, opening, last_key=None):
    """ONE warm line from code (h3_front.appreciation; ASSISTANT_PERSONA superseded 20 Oct: 'A tool', warmth is one
    short line from code at the opening or when a challenge is named). None when neither."""
    key, why = F.appreciation(message, opening, last_key)
    if not key:
        return None
    a = F.APPRECIATE[key]
    return {"key": key, "toi": a["toi"], "en": a["en"], "src": a["src"], "live": F.APPRECIATION_LIVE, "why": why}
