"""The FRONT triage of the teacher path (20 Oct 2026, the Jesuit; coordinator for Chris). Deterministic, no model,
before anything else, borrowed from an earlier WhatsApp business responder by the same author (not in this repo;
read, not imported). Lanes, logged as decision 'front:<lane>':

  media     a photo / document / audio (a page or a worksheet): "needs the page read" (OCR later); never answered blind
  memory    the teacher asks to see or wipe what we keep about them (h3_teachers.view / wipe)
  closer    a thank-you, an ok, a goodbye, or a deferral ("I will get back to you"): no draft and no reply, or one short
            acknowledgement if the profile setting asks for it. A closer that carries feedback ("it worked, thanks")
            is a closer AND its feedback is stored (h3_teachers.add_feedback)
  profile   the answer to the screening questions (h3_session.parse_answers)
  out_of_scope  (20 Oct, Chris: "What we optimize for is the teacher assistant, not a general chat buddy"; "the model
            can simply say, sorry, this is not part of my training, would you want xyz") a request that is not a teacher
            task: no draft; the FIXED redirect from code (REDIRECT below, one place), never improvised by the model;
            logged as its own outcome (type 'redirect'), never training. Its Tonga is a DRAFT until a teacher confirms
            it (REDIRECT_LIVE False): until then the turn is routed with the draft attached, nothing is sent
  task      everything else: the normal converse path with the task mode

Lessons carried over from the responder (its own comments): curly apostrophes split "I’ll" (normalise first); a question
mark or a digit is never a closer; a leading question word is a question; something asked OF US ("send me", "explain")
is never a sign-off. The Tonga closer words are only words printed in our real Tonga (the attestation counts or
Thompson), checked when the module loads: a candidate that is not printed is dropped, never invented.
"""
import functools
import re

ACK_EN = {"ok", "okay", "k", "kk", "alright", "thanks", "thank", "you", "thankyou", "noted", "sure", "fine", "great",
          "good", "nice", "cool", "perfect", "yes", "yeah", "well", "received", "appreciated", "much", "so", "very",
          "sir", "madam", "dear", "bye", "goodbye", "cheers", "blessings", "god", "bless", "regards", "and", "the", "a",
          "it", "is", "that", "this", "all", "for", "help", "your", "lot", "indeed", "true", "will", "do", "too"}
DEFER = re.compile(r"\b(will|would|let me|i'?ll|we'll|i'?m going to|going to|shall)\b.{0,40}?\b(try|get back|come back|"
                   r"let you know|check|see|use|update|report|tell you|write|try it|try them|test|share)\b")
ASK = {"send", "explain", "give", "tell", "show", "make", "write", "prepare", "translate", "help", "need", "want",
       "how", "what", "why", "which", "where", "when", "can", "could", "please"}
QWORDS = {"how", "what", "why", "which", "where", "when", "who", "ino", "nzi", "sena", "hena", "nguni", "kuli"}
MEMORY_RX = re.compile(r"(?i)\b(what do you (know|remember|keep) about me|show (me )?my (memory|data|profile)|"
                       r"(delete|wipe|forget|erase) (my|all my) (memory|data|profile)|forget me)\b")
# Tonga closers: candidates, each kept only if printed in our real Tonga (attest cache) or in Thompson's phrase book
TOI_CANDIDATES = ("twalumba", "ndalumba", "kapati", "loko", "ncibotu", "kabotu", "mbubo", "inzya", "ii", "ee",
                  "kamuciya", "kamucaala", "mweende", "mukasike", "mbuboobo", "kabotu-kabotu")


@functools.lru_cache(maxsize=1)
def toi_closers():
    import h3_detect as D
    pr = D.printed()
    return frozenset(w for w in TOI_CANDIDATES if pr.get(w, 0) >= 2)


def _words(text):
    low = (text or "").lower().replace("’", "'").replace("‘", "'")
    return low, re.findall(r"[a-zŋ'-]+", low)


def lane(message, media=False, screening_pending=False, task_given=False):
    """(lane, why) for one incoming teacher message. task_given: the caller named a teacher task mode (the desk, the
    education set): the scope is known and not judged again"""
    if not media and safeguarding(message):          # always first, whatever else the message is (AI_AGENT_POLICY §5)
        return "safeguarding", "child safeguarding: " + safeguarding(message)
    ln, why = _lane(message, media, screening_pending)
    if ln == "task" and not task_given:
        sc, swhy = scope(message)
        if sc in ("out_of_scope", "social"):
            return sc, swhy
        if sc == "advice":
            return "task", swhy
    return ln, why


def _lane(message, media=False, screening_pending=False):
    if media:
        return "media", "a photo / document / audio: needs the page read (OCR later); never answered blind"
    low, ws = _words(message)
    if MEMORY_RX.search(low):
        return "memory", "the teacher asks about their stored memory"
    if screening_pending:
        return "profile", "the answer to the screening questions"
    if not ws:
        return "closer", "no words (emoji, punctuation)"
    if "?" in low or re.search(r"\d", low) or ws[0] in QWORDS:
        return "task", "a question"
    if any(w in ASK for w in ws) and not DEFER.search(low):
        return "task", "something is asked of us"
    tc = toi_closers()
    if len(ws) <= 15 and has_feedback(message):
        return "closer", "feedback on an earlier answer, asking nothing new (stored in the teacher's memory)"
    if len(ws) <= 12 and all(w in ACK_EN or w in tc for w in ws):
        return "closer", "thanks / ok / goodbye (English or printed Tonga closers)"
    if len(low) <= 120 and DEFER.search(low):
        return "closer", "a deferral (I will try it / get back to you)"
    return "task", "a request"


def has_feedback(message):
    import h3_teachers as TM
    return bool(TM.FEEDBACK_RX.search(message or ""))


# ------------------------------------------------------------------------------------------------ scope (20 Oct)
# Deterministic: a request is a TEACHER TASK if it carries a school / teaching signal (English or Tonga stems that are
# printed in our sources: kuyiisya teach, kwiiya learn, cikolo school, mwiiyi / bai teacher, basicikolo pupils,
# mabbuku books, kubala read, kulemba write, mulaka language, twaano stories, kusandulula / kupandulula); or if it is
# pasted MATERIAL (6+ words of Tonga with no question, the teacher's own text to work on). Everything else (weather,
# news, health advice, money, jokes, greetings without a task) is out of scope. A school signal always wins: "a song
# about the weather for my Grade 2 learners" is a teacher task.
TEACHER_EN = re.compile(
    r"(?i)\b(teach\w*|lesson\w*|learners?|pupils?|students?|class(room)?|grade\s*\d+|form\s*\d|school|homework|"
    r"exercises?|worksheets?|tests?|exams?|examination|quiz|marking|mark\s+scheme|answer\s+key|syllab\w+|curriculum|"
    r"modules?|scheme\s+of\s+work|chalkboard|blackboard|reading|spelling|phonics|syllables?|letter\s+sounds?|vocabulary|"
    r"grammar|nouns?|verbs?|adjectives?|tenses?|plurals?|singular|sentences?|paragraphs?|composition|essay|story|stories|"
    r"poems?|comprehension|dictation|translat\w*|in\s+(chi)?tonga|in\s+english|into\s+(chi)?tonga|into\s+english|"
    r"meaning\s+of|how\s+do\s+(you|i|we)\s+say|word\s+for|maths?|mathematics|fractions?|addition|subtraction|"
    r"multiplication|division|numbers?|counting|science|cts|social\s+studies|literacy|numeracy|chitonga|tonga|english|"
    r"multigrade|adapt|simplif\w*|simpler|clarif\w*|explain\s+(this|the|it)\s+(topic|word|rule|text|passage))\b")
TEACHER_TOI = re.compile(r"(?i)\b\w*(y?iisy|kwiiy|iiye|cikolo|basicikolo|basikwiiy|sikwiiy|mwiiyi|baiyi|bai\b|"
                         r"bbuku|mabbuku|kubala|kulemba|mulaka|giledi|twaano|kaano|sandulul|pandulul|mabala)\w*")
GREET = re.compile(r"(?i)^\W*(hello|hi|hey|good\s+(morning|afternoon|evening|day)|how\s+are\s+you|mwabuka|mwalibuka|"
                   r"mwaswaangana|mwakkala|muli\s+buti|mulibuti|hena\s+muli\s+kabotu|sena\s+muli\s+kabotu)\b")


# ADVICE on hard classroom situations is a teacher task (Chris, DECISIONS 20 Oct): task mode "advice", keyed by the
# SITUATION (one approved answer serves every teacher in the same situation)
SITUATIONS = (
    ("large_class", r"(large|big|huge|overcrowded|crowded)\s+class|too\s+many\s+(learners|pupils|children|students)|"
                    r"\b\d{2,3}\s+(learners|pupils|children|students)\b.*\b(one|1)\s+teacher|class\s+(size|of\s+\d{2,3})"),
    ("discipline", r"disciplin\w*|misbehav\w*|noisy|noise\s+in\s+class|rude|disrespect\w*|fight(s|ing)?\s+in\s+class|"
                   r"cheat\w*|bully\w*|do\s+not\s+listen|don'?t\s+listen|control\s+(my|the)\s+class"),
    ("absenteeism", r"absent\w*|absenteeism|not\s+coming\s+to\s+school|stop(ped)?\s+coming|late\s*-?\s*coming|"
                    r"come\s+late|coming\s+late|drop(ping)?\s*-?\s*out|attendance"),
    ("mixed_ability", r"mixed\s+abilit\w*|slow\s+learners?|fast\s+learners?|different\s+levels|cannot\s+read|can'?t\s+read|"
                      r"struggl\w*|behind\s+the\s+others|multigrade"),
    ("no_textbooks", r"no\s+text\s*books?|few\s+(text)?books|shortage\s+of\s+(text)?books|share\s+(one|a)\s+book|"
                     r"no\s+(books|materials|chalk|paper)|lack\s+of\s+(materials|books)"),
    ("parents", r"parents?|guardians?|pta\b|community\s+meeting"),
    ("motivation", r"motivat\w*|not\s+interested|bored|lose\s+interest"),
)
ADVICE_ASK = re.compile(r"(?i)\b(how|what|advice|advise|help|tips?|ideas?|should|can\s+i|manage|handle|deal\s+with|cope)\b")
# CHILD SAFEGUARDING (AI_AGENT_POLICY.md §5: "anything touching a child's welfare is referred to a person, never advised
# on by an agent alone"): deterministic, erring on the side of catching. Strong words alone; weaker words only with a
# child in the sentence. The reply is a fixed referral line (REFERRAL below), never a model draft.
SAFE_STRONG = re.compile(r"(?i)\b(abus\w*|defile\w*|rap(e|ed|ing)|molest\w*|incest|sexual(ly)?|indecent\w*|"
                         r"self[\s-]*harm\w*|suicid\w*|kill\s+(him|her|them)sel(f|ves)|end\s+(his|her|their)\s+life|"
                         r"cut(s|ting)?\s+(him|her|them)sel(f|ves)|hurt(s|ing)?\s+(him|her|them)sel(f|ves)|"
                         r"early\s+marriage|married\s+off|child\s+marriage|traffick\w*|child\s+labou?r|kulijaya|"
                         r"lijaya|in\s+danger|neglect\w*|pregnan\w*)\b")
SAFE_CHILD = re.compile(r"(?i)\b(child|children|learner|learners|pupil|pupils|girl|boy|student|mwana|bana|kasimbi|kamwana)\b")
SAFE_WEAK = re.compile(r"(?i)\b(beat(s|en|ing)?|hit(s|ting)?|burn(t|ed|s)?|injur\w*|bruis\w*|wound\w*|scars?|"
                       r"touch(ed|es|ing)?\s+(her|him)|runs?\s+away|ran\s+away|missing|threat\w*|violen\w*|"
                       r"starv\w*|chased\s+from\s+home|sleeps?\s+outside|afraid\s+to\s+go\s+home|scared\s+to\s+go\s+home|"
                       # 24 Oct 2026 (chat round v0.14 held-out run, the Jesuit): 'her stepfather hurts her at night' and the
                       # Tonga 'bausyi balamuuma' (his father beats him) passed the gate. Weak words, so a child must be in
                       # the message: hurt(s) her / him / them; Tonga -uuma (beat: balamuuma, bamuuma, kumuuma, wamuuma)
                       r"hurt(s|ing)?\s+(her|him|them|me)|\w*m?uuma|kuuma|balamuuma)\b")
# 24 Oct 2026 (reader bake-off section 3c, the Jesuit): the three regexes above missed seven kinds of disclosure in every
# arm (marriage, transactional sex, forced sex, chased from home, cover-up, images, food deprivation) and every typo'd
# one. The gate is now tools/chitonga/llm/safeguard.py: by category, normalised (SMS spellings, fuzzy key words), with
# benign lesson phrases blanked. The names below stay for their consumers (web/api.py and export_netlify.py ship them to
# the phone and the Netlify function) and are now the unions of the new single-regex rules; the full rule set, the
# normalisation and the combination rules travel as safeguard.export(). safeguarding() below is the new detector.
import safeguard as _SG  # noqa: E402
SAFE_STRONG, SAFE_CHILD, SAFE_WEAK = _SG.legacy_regexes()
SOCIAL = re.compile(r"(?i)\b(tired|exhausted|hard\s+day|tough\s+day|long\s+day|difficult\s+day|stress\w*|"
                    r"overwhelm\w*|discourag\w*|sad|frustrat\w*|worn\s+out|ndakatala|ndalema|mwakatala|"
                    r"ndalibizya|ndabuka|ndakotoka)\b")


def situation(message):
    low = (message or "").lower()
    return [k for k, rx in SITUATIONS if re.search(rx, low)]


def safeguarding(message):
    """'' or the reason (category and the words that caught it). Errs on the side of catching. safeguard.py since
    24 Oct 2026 (the old three-regex body is kept in tools/chitonga/harness/tests/safeguard/score.py for comparison)."""
    return _SG.safeguarding(message)


def scope(message):
    """('safeguarding' | 'advice' | 'teacher_task' | 'social' | 'out_of_scope', why); deterministic, no model.
    advice: a hard classroom situation (SITUATIONS) asked about; social: greetings and how-the-day-was (tired, a hard
    day, too much) without a task: a brief encouragement, then the offer (never a redirect)"""
    low = (message or "").replace("’", "'")
    if safeguarding(low):
        return "safeguarding", "child safeguarding: " + safeguarding(low)
    sit = situation(low)
    if sit and (ADVICE_ASK.search(low) or "?" in low):
        return "advice", "advice on a classroom situation: " + ", ".join(sit)
    if TEACHER_EN.search(low) or TEACHER_TOI.search(low):
        return "teacher_task", "a school / teaching signal"
    if sit:
        return "social", "the teacher tells how it is (" + ", ".join(sit) + "), asking nothing: encouragement + offer"
    ws = re.findall(r"[A-Za-zŋŊ']+", low)
    if SOCIAL.search(low):
        return "social", "the teacher says how the day was: encouragement + offer"
    if GREET.match(low):
        return "social", "a greeting: greeting back + offer"
    try:
        import attest
        eng = sum(1 for v, w, _ in attest.line(low) if v == "english")
    except Exception:  # noqa: BLE001  (no attest: count nothing as English)
        eng = 0
    if len(ws) >= 6 and "?" not in low and eng < 0.3 * len(ws):
        return "teacher_task", "pasted Tonga material (6+ words, no question): the teacher's own text to work on"
    return "out_of_scope", "no teacher task (no school or teaching signal, no pasted material)"


# THE REDIRECT, in one place. Supported task modes: TEACHER_PROFILE.md §2 (teach_tonga, teach_english,
# explain_subject) plus answer keys and adapting / clarifying a text. Tonga only (DECISIONS 20 Oct, like the frank
# script). The Tonga is a DRAFT from sourced pieces, NOT LIVE until a teacher confirms it (desk item
# desk/redirect_2026-10-20/redirect_item.md):
#   "Ndausa loko." = "I'm very sorry." (Thompson 1979 p10, CONVERSATION / CHAT, printed)
#   "eeci tandiciiyide" = "this, I have not learned it": engine2 build iya 1s neg_perf + object c7 (stem iya "learn",
#      Hopgood index p305; kwiiya in Thompson p33-34); engine2 reads tandiciiyide back as iya 1s neg_perf obj c7
#   "Sena muyanda kuti mugwasye ...?" = "Do you want me to help you ...?" (Thompson p23, ENQUIRIES / QUESTIONS, printed
#      exactly so)
#   the task names in Tonga: NOT drafted (no sourced phrase yet); the teacher writes them in the desk item
SUPPORTED_TASKS = ("teach_tonga", "teach_english", "explain_subject", "answer_key", "adapt_clarify")
SUPPORTED_EN = {"teach_tonga": "teach a Chitonga lesson", "teach_english": "teach an English lesson",
                "explain_subject": "explain a topic of another subject (maths, science, CTS)",
                "answer_key": "make an answer key or a short test", "adapt_clarify": "adapt or explain a text for your learners"}
REDIRECT = {
    "toi_draft": "Ndausa loko, eeci tandiciiyide. Sena muyanda kuti mugwasye {tasks_toi}?",
    "toi_tasks": None,              # filled from the teacher's confirmed wording; until then the line is not live
    "en_meaning": "Sorry, this is not part of my training. Would you like help to {tasks_en}?",
    "status": "draft, not confirmed by a teacher",
    "sources": ["Thompson 1979 p10 (Ndausa loko)", "engine2: iya 1s neg_perf obj c7 (Hopgood index p305)",
                "Thompson 1979 p23 (Sena muyanda kuti mugwasye?)"],
}
REDIRECT_LIVE = False               # a teacher confirms the Tonga first (DECISIONS 20 Oct); never improvised by the model


# ENCOURAGEMENT BANK for the social lane (Chris, DECISIONS 20 Oct: "It can keep some general encouragement for the
# teachers, it's a tough job"). Real Tonga only: printed lines (Thompson 1979, page) or engine2 forms of lexicon stems
# (marked). NOT LIVE until a teacher confirms them (desk/social_2026-10-20/); chosen by code (social_reply), never by
# the model; no religion, no preaching.
ENCOURAGE = {
    "thanks_much": {"toi": "Twalumba kapati.", "en": "Thank you very much.", "src": "Thompson 1979 p9 (CONVERSATION / CHAT)"},
    "sorry_for_you": {"toi": "Ndamuusila.", "en": "I am sorry for you.", "src": "Thompson 1979 p9 (CONVERSATION / CHAT)"},
    "rest_little": {"toi": "Kamulyookezya asyoonto.", "en": "Rest a little.", "src": "Thompson 1979 p32 (HOSPITAL)"},
    "keep_going": {"toi": "Kamuciya.", "en": "Keep going.", "src": "Thompson 1979 p28 (GREETINGS: 'Kamuciya. (Keep going.)')"},
    "stay_well": {"toi": "Kamucaala kabotu.", "en": "Stay well.", "src": "Thompson 1979 p28 (GREETINGS)"},
    "meet_again": {"toi": "Tuyoobonana alimwi.", "en": "We shall meet again.", "src": "Thompson 1979 p28 (GREETINGS)"},
    "good_morning": {"toi": "Mwabuka.", "en": "Good morning.", "src": "Thompson 1979 p27 (GREETINGS)"},
    "good_afternoon": {"toi": "Mwalibizya.", "en": "Good afternoon.", "src": "Thompson 1979 p27 (GREETINGS)"},
    "good_evening": {"toi": "Kwasiya.", "en": "Good evening.", "src": "Thompson 1979 p27 (GREETINGS)"},
    "worked_hard": {"toi": "Mwabeleka kapati sunu.", "en": "You have worked hard today.",
                    "src": "DRAFT: engine2 beleka 2p today-past (mwabeleka; stem beleka 'work', T01 F039) + kapati "
                           "'very' (T01 F035) + sunu 'today' (lexicon); not printed as a line"},
}
SOCIAL_LIVE = False

# APPRECIATION (Chris, DECISIONS 20 Oct: the assistant always shows appreciation for the teachers' challenges and their
# value to society). ONE brief line, chosen by code, prepended at the opening of a conversation and whenever the
# teacher's message names a challenge; varied per teacher (the teacher memory keeps the last key); specific, never
# flattery, never pasted onto every answer. NOT LIVE until a teacher confirms the lines (desk/social_2026-10-20/).
APPRECIATE = {
    "app_important_work": {"toi": "Mulaa mulimo mupati kapati.", "en": "You have very important work.",
                           "src": "DRAFT: a printed Grade 4 reader line (chiefs have very important work), person "
                                  "changed ba- > mu-", "when": "both"},
    "app_help_children": {"toi": "Mulagwasya bana kapati.", "en": "You help the children a great deal.",
                          "src": "DRAFT: engine2 gwasya 2p pres (gwasya 'help', Hopgood index p302) + bana + kapati",
                          "when": "challenge"},
    "app_hard_job": {"toi": "Mulimo wanu muyumu.", "en": "Your work is hard.",
                     "src": "DRAFT: mulimo 'work' (Hopgood index p311; 'mulimo mupati', Grade 4 reader) + wanu 'your' "
                            "(Thompson p27 'mukaintu wanu') + -yumu 'hard, difficult' (Hopgood index p319), class 3 mu-",
                     "when": "challenge"},
    "app_work_important_yours": {"toi": "Mulimo wanu mupati.", "en": "Your work is important.",
                                 "src": "DRAFT: as app_hard_job with -pati 'big, great' (Hopgood index p315)",
                                 "when": "both"},
}
# Dropped on the persona's "never" list (no flattery, no praise not tied to anything): "Mulabeleka kabotu" (you work
# well) and a bare "Twalumba kapati" at the opening; appreciation names the value of the work or the challenge named.
APPRECIATION_LIVE = False
CHALLENGE = re.compile(r"(?i)\b(hungry|hunger|tired|exhausted|absent\w*|no\s+(text)?books|few\s+books|large\s+class|"
                       r"too\s+many\s+(learners|pupils)|overcrowd\w*|no\s+(chalk|paper|desks|materials|electricity)|"
                       r"far\s+to\s+walk|walk\s+far|hard|difficult|struggl\w*|alone)\b")


def appreciation(message, opening, last_key=None):
    """(key or None, why): one line at the opening of a conversation or when the message names a challenge; never the
    same key twice in a row for a teacher"""
    ch = bool(CHALLENGE.search(message or "") or situation(message or ""))
    if not (opening or ch):
        return None, "no opening, no challenge named: no appreciation line"
    want = "challenge" if ch else "opening"
    keys = [k for k, v in APPRECIATE.items() if v["when"] in (want, "both")]
    keys = [k for k in keys if k != last_key] or keys
    i = sum(map(ord, (message or "")[:40])) % len(keys)          # varied, deterministic
    return keys[i], ("the message names a challenge" if ch else "the opening of a conversation")


def social_reply(message, hour=None):
    """the bank lines for a social message, chosen by code: a greeting back by the time of day, sympathy and rest for a
    hard / tiring day, recognition for a heavy load, then the task offer (the redirect's offer line). Not live."""
    low = (message or "").lower()
    keys = []
    if GREET.match(low):
        h = hour if hour is not None else __import__("datetime").datetime.now().hour
        keys.append("good_morning" if h < 12 else "good_afternoon" if h < 17 else "good_evening")
    if re.search(r"(?i)tired|exhausted|worn\s+out|ndakatala|ndalema", low):
        keys += ["sorry_for_you", "rest_little"]
    elif re.search(r"(?i)hard|tough|long|difficult|stress|overwhelm|discourag|frustrat|sad", low) or situation(low):
        keys += ["sorry_for_you", "keep_going"]
    elif not keys:
        keys.append("thanks_much")
    keys = keys[:2]
    offer = REDIRECT["toi_draft"].split(". ", 1)[1].format(tasks_toi=REDIRECT.get("toi_tasks") or "[...]")
    toi = " ".join(ENCOURAGE[k]["toi"] for k in keys) + " " + offer
    live = SOCIAL_LIVE and REDIRECT.get("toi_tasks")
    return {"reply": toi if live else None, "draft_toi": toi, "lines": keys, "live": bool(live),
            "en_meaning": " ".join(ENCOURAGE[k]["en"] for k in keys) + " Would you like help with a lesson?",
            "train": False}


# THE SAFEGUARDING REFERRAL: fixed, from code, never a model draft; no helpline number here (to be verified separately).
REFERRAL = {"en": "This is about a child's safety. I cannot advise on it. Please tell your head teacher or the guidance "
                  "and counselling teacher today, or the proper authority.",
            "toi": None,                  # the Tonga waits for a teacher (desk item); nothing is drafted by the model
            "status": "English meaning only; Tonga to be written and confirmed by a teacher"}
REFERRAL_LIVE = False


def referral(why=""):
    live = REFERRAL_LIVE and REFERRAL.get("toi")
    return {"reply": REFERRAL["toi"] if live else None, "en_meaning": REFERRAL["en"], "live": bool(live),
            "status": REFERRAL["status"], "why": why, "train": False}


def redirect(why=""):
    """the redirect record for an out-of-scope request: the reply is None until REDIRECT_LIVE (routed, draft attached)"""
    en = REDIRECT["en_meaning"].format(tasks_en=", ".join(SUPPORTED_EN[t] for t in SUPPORTED_TASKS))
    live = REDIRECT_LIVE and REDIRECT.get("toi_tasks")
    toi = REDIRECT["toi_draft"].format(tasks_toi=REDIRECT.get("toi_tasks") or "[...]")
    return {"reply": toi if live else None, "draft_toi": toi, "en_meaning": en, "live": bool(live),
            "status": REDIRECT["status"], "sources": REDIRECT["sources"], "why": why, "train": False}
