"""The conversation engine: a small state machine driven by the plug-in's menu, with the teacher memory (23 Oct 2026,
the Jesuit; Chris: the harness and the education plug-in; "remember the teacher"; "Not starting from 0 every time";
class memory and the strategy follow-up).

    greet (new: name > language > grade / class)  >  task offer  >  [class]  >  [grade]  >  subject  >  unit  >  item
      >  deliver in parts (Continue / Mistakes / Something else)  >  finished: "Next: <unit>?" / Menu / No, thank you
    hard day > encouragement (fixed lines) > offer          out of scope > fixed redirect > offer
    safeguarding > fixed referral (EVENT stored, never the text)
    struggle ("most of my class ...") > class observation > matched mistake > intervention (+ challenge track)
      > next session: "Did it help?" once (helped / partly / no)
    free text > deterministic intake > reader > confirm when unsure or translated > deliver

RESUME. The state is stored with the teacher memory after every turn. When a teacher comes back after
SESSION_GAP (or greets), the engine offers to resume: an unfinished lesson ("Last time: ..., part 2 of 4. Continue?"),
an interrupted choice ("Last time you were choosing a Maths unit. Continue?") or, after a finished lesson, the next unit
("Next: Subtraction?"). A substantive request is answered directly (the teacher has moved on). A fresh start only on
"start over" or when the old path is finished.

Every reply is data (io.py); every turn is one log record (log.py). Nothing is sent from here.

24 Oct 2026 (the architect's harness review, items 1-3, 5-8; the Jesuit):
  1 one turn at a time per number (a lock per normalised number in handle()); the teacher's language and the active
    plug-in live in the turn's context (a ContextVar), never on the Engine: two teachers served in parallel cannot
    leak into each other's reply, and two messages from one teacher cannot lose each other's state.
  2 every reading is validated here (intake.validate_reading with the plug-in: intent, item among the candidates and
    in the plug-in, unit in the plug-in); an invalid reading becomes 'unknown' with a flag. A model reading always
    confirms (intake.needs_confirm).
  3 "Not this one" on part 1 of a delivery that came straight from free text: one tap back to the alternatives, and
    the tap is a labelled reader error in the turn log (label.kind = reader_error), held-out material for the reader.
  5 the state remembers the plug-in's fingerprint (version@built); after a rebuild an unfinished path drops to the menu
    with "the material was updated" instead of resuming on ids that may be gone.
  6 several plug-ins: Engine([education, other]); a minimal router (rules reading per plug-in, best confidence wins,
    ties stay with the active one; taps on another plug-in's ids switch to it); the first menu level is the plug-in
    when more than one is loaded; flat (non-grade) menus; plug-ins without class memory (manifest memory_unit null).
    A test-only plug-in loads only with allow_test_plugins=True.
  7 plug-in intents through hooks (any 'route:' item is a route), generic status rules (gates.status_rules).
  8 one transaction per turn (memory.transaction): a failed _turn rolls back its writes and the state, then the
    teacher still gets an answer and the state / log are written.
"""
import contextvars
import copy
import datetime
import inspect
import json
import re
import threading
import time

from . import gates as G
from . import io as IO
from . import render as R
from .intake import RulesReader, blank, grades_in, needs_confirm, validate_reading
from .log import TurnLog
from .memory import norm_number, pupil_level
from .retrieval import Index

SESSION_GAP = datetime.timedelta(hours=2)
_TURN = contextvars.ContextVar("harness_turn", default=None)   # the ctx of the turn running in this thread / task
# the state keys of a path in progress (dropped when the plug-in changed under it, item 5)
PATH_KEYS = ("item", "unit", "part", "parts", "finished", "pending", "subject", "grade", "page", "pending_class",
             "struggle_item", "struggle_mistake", "delivered_from", "grade_for_subject")
GENERIC_INTENTS = ("start_over", "continue", "set_language", "set_name", "set_school", "add_class")
LANG_LABEL = {"en": "English", "toi": "Tonga", "both": "Tonga and English"}
STOP = set("the a an of to in on and or my our is are with at for can't cannot cant struggle struggles most many some "
           "learners pupils children class them they it this that not still yet".split())


# words that are never a teacher's name (23 Oct, stream A: 'Literacy', 'Lesson', 'Ok', 'Social studies' were stored as
# names); the plug-in's subject and unit keywords are added at run time (_not_a_name)
NOT_NAMES = set("""help lesson lessons menu hi hello hey ok okay yes no yeah thanks thank please teacher teachers grade
grades class classes reading writing literacy numeracy art arts music sport sports social studies history geography
homework test tests exam exams marking counting numbers number maths math mathematics science english tonga chitonga
citonga subject subjects today tomorrow nothing skip later none idk unknown pupils learners children school""".split())


def _glabel(gs):
    return ("Grade " if len(gs) == 1 else "Grades ") + " and ".join(map(str, gs))


def _words(s):
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2 and w not in STOP}


class Engine:
    def __init__(self, plugin, memory, reader=None, log=None, hooks=None, clock=None, session_gap=SESSION_GAP,
                 drafts=False, allow_test_plugins=False, tonga_ui=False, flywheel=None):
        """plugin: one Plugin or a list (the first is the default). A test-only plug-in (manifest test_only) is
        refused unless allow_test_plugins=True (tests only: never served to real users).

        tonga_ui (24 Oct 2026, Chris: "it's safe to assume that all teachers in Southern Province speak English enough
        that the harness can be all English; it can offer Tonga materials and explanations"): False (default) = the
        INTERFACE (greeting, menus, buttons, confirmations, redirect, follow-ups) is English for everyone; the
        teacher's language preference then governs only the CONTENT (the lessons' Tonga lines, Tonga explanations;
        the Chitonga lessons are Tonga whatever the preference). True = the 23 Oct behaviour: the fixed lines are also
        served in Tonga (core/fixed.py, teacher-written lines only). Intake reads Tonga either way."""
        plugs = list(plugin) if isinstance(plugin, (list, tuple)) else [plugin]
        for p in plugs:
            if getattr(p, "test_only", False) and not allow_test_plugins:
                raise ValueError(f"plug-in {p.name} is TEST ONLY: load it with allow_test_plugins=True, in tests only")
        self.plugins = {p.name: p for p in plugs}
        self.default = plugs[0]
        self.allow_test = allow_test_plugins
        self.tonga_ui = tonga_ui
        self.drafts = drafts        # True (simulator only): also show Tonga still waiting for a teacher, tagged
        self.mem = memory
        self._nonname = None        # built on first use from NOT_NAMES + the plug-ins' subject / unit words
        self.reader = reader or RulesReader()
        self._router = RulesReader()   # deterministic, no model: only decides WHICH plug-in reads the message
        self.log = log if log is not None else TurnLog()
        self.hooks = hooks          # the default plug-in's hooks when the Plugin object carries none (23 Oct callers)
        self.clock = clock or datetime.datetime.now
        self.gap = session_gap
        self.indexes = {p.name: Index(p) for p in plugs}
        self._locks = {}
        self._locks_guard = threading.Lock()
        self.last = None            # the last turn record (diagnostics and tests; not meaningful across threads)
        # 24 Oct 2026 (the flywheel, core/flywheel.py, FLYWHEEL.md): None = off (the harness tests, the CLI); the pilot
        # adapter passes a Flywheel, which may add ONE feedback prompt after a turn and reads the answer to it
        self.fw = flywheel

    # the turn's context: the active plug-in and the teacher's language are per turn, never per Engine (item 1)
    @property
    def P(self):
        ctx = _TURN.get()
        return ctx["P"] if ctx else self.default

    @property
    def lang(self):
        ctx = _TURN.get()
        return ctx["lang"] if ctx else "en"

    @property
    def index(self):
        return self.indexes[self.P.name]

    def _ui_lang(self, pref):
        """the language of the interface lines: English unless tonga_ui (the preference still governs content)"""
        return (pref or "en") if self.tonga_ui else "en"

    def _hooks(self):
        hk = getattr(self.P, "hooks", None)
        if hk is not None:
            return hk
        return self.hooks if self.P is self.default else None

    # ------------------------------------------------------------------------------------------------ helpers
    def b(self, key):
        return self.P.button(key, self.lang, self.drafts)

    def t(self, key, **slots):
        """a fixed line in the language the teacher chose (memory), Tonga from the core table (core/fixed.py)"""
        return self.P.t(key, self.lang, self.drafts, **slots)

    def offer(self, body=None, ctx=None):
        items = [("c:lesson_help", self.b("lesson_help")), ("c:struggle", self.b("struggle")), ("c:lang", self.b("language"))]
        return IO.buttons(body or self.t("task_offer"), items)

    def _state(self, n):
        return self.mem.get_state(n) or {"step": "idle"}

    def _class(self, ctx, grade=None, ask_for=None):
        """(class or None, reply or None): the class this turn is about; asks 'Which class?' only when the teacher has
        more than one and neither the grade nor this session's choice settles it"""
        n, st = ctx["n"], ctx["st"]
        cls = self.mem.classes(n)
        if not cls:
            g = grade or 1
            c = self.mem.add_class(n, f"Grade {g}", [g], plugin=self.P.name)
            self.mem.update_teacher(n, last_class_id=c["id"])
            st["class_id"] = c["id"]
            ctx["path"].append("class:auto-created")
            return c, None
        if len(cls) == 1:
            st["class_id"] = cls[0]["id"]
            return cls[0], None
        if grade:
            fit = [c for c in cls if grade in c["grades"]]
            if len(fit) == 1:
                st["class_id"] = fit[0]["id"]
                return fit[0], None
        if st.get("class_id") and st.get("class_session") == st.get("session"):
            return self.mem.get_class(st["class_id"]), None
        st["pending_class"] = ask_for
        st["step"] = "class"
        rows = [(f"class:{c['id']}", c["label"], "Grades " + ", ".join(map(str, c["grades"]))) for c in cls]
        ctx["path"].append("ask:class")
        return None, IO.choices(self.t("ask_class"), rows, more_id="more:class")

    def _set_class(self, ctx, cid):
        ctx["st"]["class_id"] = cid
        ctx["st"]["class_session"] = ctx["st"].get("session")
        self.mem.update_teacher(ctx["n"], last_class_id=cid)

    # ------------------------------------------------------------------------------------------------ the turn
    def _lock_for(self, n):
        with self._locks_guard:
            return self._locks.setdefault(n, threading.Lock())

    def handle(self, number, msg):
        """one inbound message -> [replies]. One turn at a time per number (item 1), one transaction per turn
        (item 8); the log line is written after the commit."""
        n = norm_number(number)
        with self._lock_for(n):
            return self._handle(n, msg)

    def _handle(self, n, msg):
        t0 = time.time()
        with self.mem.transaction():
            prof, created = self.mem.get_or_create(n)
            st = self._state(n)
            now = self.clock()
            last = datetime.datetime.fromisoformat(st["last_at"]) if st.get("last_at") else None
            new_session = created or last is None or (now - last) > self.gap
            if new_session:
                st["session"] = now.isoformat(timespec="seconds")
            P = self.plugins.get(st.get("active_plugin")) or self.default
            ctx = {"n": n, "prof": prof, "st": st, "created": created, "new_session": new_session, "path": [],
                   "flags": [], "origins": [], "reading": None, "withhold": False, "msg": msg, "now": now,
                   "lang": self._ui_lang(prof.get("language_pref")), "P": P, "label": None}
            token = _TURN.set(ctx)
            try:
                stale = self._check_version(ctx, st.get("active_plugin") not in (None, P.name))
                before, lang0 = copy.deepcopy(st), ctx["lang"]
                try:
                    with self.mem.transaction():            # a savepoint: a failed turn leaves no half-written rows
                        replies = self._turn(ctx, msg)
                except Exception as e:  # noqa: BLE001  (a bug must never leave a teacher without an answer)
                    ctx["flags"].append(f"engine error: {type(e).__name__}: {e}")
                    st.clear()
                    st.update(before)
                    ctx.update(P=P, lang=lang0)
                    replies = [IO.text(self.t("unclear"))]
                if stale:
                    replies = [IO.text(self.t("material_updated"))] + replies
                # the follow-up of last session's suggestion: asked once, at the next contact (never after a referral,
                # never to a teacher who turned feedback questions off: flywheel, 24 Oct)
                if new_session and not created and "safeguarding" not in ctx["path"] and                         not (self.fw and self.fw.is_off(ctx)):
                    sug = self.mem.pending_followup(n, before=st["session"])
                    if sug:
                        self.mem.mark_followup_asked(sug["id"])
                        replies.append(IO.buttons(self.t("followup", what=sug["what"],
                                                         observation=sug["observation"] or ""),
                                                  [(f"fb:{sug['id']}:helped", self.b("helped")),
                                                   (f"fb:{sug['id']}:partly", self.b("partly")),
                                                   (f"fb:{sug['id']}:no", self.b("not_helped"))]))
                        ctx["path"].append(f"followup:asked:{sug['id']}")
                if self.fw:                                  # at most one feedback prompt per answer (flywheel.py)
                    replies = self.fw.after_turn(self, ctx, replies)
                st["last_at"] = now.isoformat(timespec="seconds")
                st["active_plugin"] = self.P.name
                st["plugin_version"] = self.P.fingerprint
                self.mem.set_state(n, st)
                bad = [p for r in replies for p in IO.validate(r)]
                rec = {"when": now.isoformat(timespec="seconds"), "number": n, "plugin": self.P.name,
                       "plugin_version": self.P.fingerprint, "reader": self.reader.name,
                       "inbound": ({"withheld": ctx["withhold"]} if ctx["withhold"] else msg),
                       "new_session": new_session, "reading": ctx["reading"], "path": ctx["path"],
                       "origins": list(dict.fromkeys(ctx["origins"])), "flags": ctx["flags"] + bad,
                       "replies": replies, "state": {k: st.get(k) for k in
                                                     ("step", "subject", "unit", "item", "part", "parts", "class_id",
                                                      "grade", "active_plugin")},
                       "ms": round(1000 * (time.time() - t0))}
                if ctx["label"]:
                    rec["label"] = ctx["label"]
                if ctx.get("flywheel"):
                    rec["flywheel"] = ctx["flywheel"]
            finally:
                _TURN.reset(token)
        self.log.write(rec)
        self.last = rec
        return replies

    def _check_version(self, ctx, plugin_gone=False):
        """item 5: True (and the path dropped) when the state's path was made with other plug-in material: another
        fingerprint (a rebuild), a plug-in no longer loaded, or (state written before 24 Oct, no fingerprint) ids
        that no longer exist"""
        st, P = ctx["st"], self.P
        in_progress = st.get("step") not in (None, "idle") or any(st.get(k) for k in ("item", "unit", "pending"))
        if not in_progress:
            return False
        old = st.get("plugin_version")
        if old is None and not plugin_gone:
            ids_ok = ((not st.get("item") or st["item"] in P.items) and (not st.get("unit") or st["unit"] in P.units)
                      and (not st.get("pending") or st["pending"].get("item") in P.items))
            if ids_ok:
                return False
        elif old == P.fingerprint and not plugin_gone:
            return False
        for k in PATH_KEYS:
            st.pop(k, None)
        st["step"] = "idle"
        ctx["path"].append(f"stale:{old}->{P.fingerprint}")
        return True

    def _turn(self, ctx, msg):
        if msg.get("choice"):
            ctx["path"].append("choice:" + msg["choice"])
            if self.fw and msg["choice"].startswith("fw:"):
                return self.fw.on_choice(self, ctx, msg["choice"])
            return self._choice(ctx, msg["choice"], msg)
        text = (msg.get("text") or "").strip()
        st, n = ctx["st"], ctx["n"]
        # 1 safeguarding: the referral, the EVENT only
        s = G.safeguarding(text)
        if s:
            ctx["withhold"] = "safeguarding message (text not stored)"
            ctx["path"].append("safeguarding")
            self.mem.event(n, "safeguarding_referral")
            if self.fw:
                self.fw.clear(ctx, "safeguarding")
            ref = G.referral(s)
            return [IO.text(ref["reply"] or ref["en_meaning"])]
        # 1b the flywheel (core/flywheel.py): 'feedback off / on', and the teacher's line while a Fix / Write waits.
        # AFTER the safeguarding gate on purpose: a disclosure typed into a feedback step still gets the referral.
        if self.fw and text:
            fwr = self.fw.command(self, ctx, text)
            if fwr is None and self.fw.waiting(ctx):
                fwr = self.fw.on_text(self, ctx, text)
            if fwr is not None:
                return fwr
        # 2 out of scope
        d = G.deny(text)
        if d:
            ctx["path"].append("redirect")
            red = G.redirect(d)
            self.mem.log_request(n, "out_of_scope", outcome="redirect")
            return [self.offer(red["reply"] or self.P.fixed.say("redirect", red["en_meaning"], self.lang, self.drafts))]
        # 3 social
        soc = G.social(text)
        if soc == "hard_day":
            ctx["path"].append("encourage")
            e = G.encourage(text, ctx["now"].hour)
            self.mem.log_request(n, "social", outcome="encourage")
            en = e["en_meaning"].replace(" Would you like help with a lesson?", "")
            return [IO.text(e["reply"] or self.P.fixed.compose(["enc_" + k for k in e["lines"]], en, self.lang,
                                                               self.drafts)), self.offer()]
        if soc == "greet" or not text:
            return self._greet(ctx)
        # 4 onboarding answers in free text
        if st.get("step") == "onboard_name" and self._not_a_name(text):
            # 23 Oct (stream A): a subject or a school word ('Maths', 'Literacy', 'Grade 1', 'Ok') is an answer to a
            # different question, never a name: read it as a request, and say so once with Skip
            st["step"] = "idle"
            ctx["path"].append("name:not_taken")
            r = self._read(ctx, text)
            ctx["reading"] = r
            ctx["path"].append(f"read:{r['intent']}:{r['source']}")
            gated = self._gate_reading(ctx, r, text)      # 24 Oct: this path skipped the reading's gate pass
            if gated is not None:
                return gated
            out = self._intent(ctx, r, text)
            return out + [IO.buttons(self.t("name_not_taken", word=text.strip()), [("c:skip_name", self.b("skip"))])]
        if st.get("step") == "onboard_name" and len(text.split()) <= 4 and re.match(r"^[A-Za-z][\w'. -]*$", text):
            r0 = self._read(ctx, text)
            if r0["intent"] == "unknown":
                gated = self._gate_reading(ctx, r0, text)  # a short Tonga disclosure is never stored as a name
                if gated is not None:
                    return gated
                return self._set_name(ctx, text)
        if st.get("step") == "new_class_label":
            return self._new_class(ctx, text)
        # 5 which plug-in (only when more than one is loaded), then the reading
        self._pick_plugin(ctx, text)
        r = self._read(ctx, text)
        ctx["reading"] = r
        ctx["path"].append(f"read:{r['intent']}:{r['source']}")
        gated = self._gate_reading(ctx, r, text)
        if gated is not None:
            return gated
        return self._intent(ctx, r, text)

    def _gate_reading(self, ctx, r, text):
        """the gates once more on the reading's ENGLISH, BEFORE any routing (_intent): the replies, or None.

        24 Oct 2026 (chat round v0.14, the Jesuit): the gates in _turn read the teacher's own text; a model or
        translator reading carries the message's ENGLISH (read_en), so steps 1 and 2 run once more on it. A Tonga
        safeguarding message the patterns cannot read is still referred, and a Tonga out-of-scope message still
        redirected, by code. 24 Oct (reader bake-off 3c): the safeguarding check normalises spelling itself
        (safeguard.py), so a reader that echoes the typos ('head tr beats childten') no longer lets a disclosure
        through; and every path that reaches _intent passes here (the name-step path did not)."""
        en = str(r.get("read_en") or "").strip()
        if not en or en.lower() == (text or "").strip().lower():
            return None
        n = ctx["n"]
        s = G.safeguarding(en)
        if s:
            ctx["withhold"] = "safeguarding message (text not stored)"
            ctx["path"].append("safeguarding")
            ctx["path"].append("gate:reading")
            self.mem.event(n, "safeguarding_referral")
            if self.fw:
                self.fw.clear(ctx, "safeguarding")
            ref = G.referral(s)
            return [IO.text(ref["reply"] or ref["en_meaning"])]
        d = G.deny(en)
        if d:
            ctx["path"].append("redirect")
            ctx["path"].append("gate:reading")
            red = G.redirect(d)
            self.mem.log_request(n, "out_of_scope", outcome="redirect")
            return [self.offer(red["reply"] or self.P.fixed.say("redirect", red["en_meaning"], self.lang, self.drafts))]
        return None

    def _rctx(self, ctx, P=None):
        P = P or self.P
        cls = self._current_class(ctx)
        text = (ctx["msg"].get("text") or "")
        return {"plugin": P, "index": self.indexes[P.name], "number": ctx["n"],
                "recall": self.mem.recall(ctx["n"], text), "grades": cls["grades"] if cls else [1]}

    def _current_class(self, ctx):
        """the class this turn is most likely about, without asking (24 Oct 2026: the reader prefers its grade): this
        session's class, else the teacher's last class, else the only class; None when nothing settles it"""
        cid = ctx["st"].get("class_id") or (self.mem.get_teacher(ctx["n"]) or {}).get("last_class_id")
        c = self.mem.get_class(cid) if cid else None
        if c:
            return c
        cls = self.mem.classes(ctx["n"])
        return cls[0] if len(cls) == 1 else None

    def _read(self, ctx, text):
        """the reader's reading, VALIDATED here for every reader (item 2): a reading that breaks the contract (an
        intent the core and the plug-in do not know, an item outside the candidates or the plug-in, a unit the
        plug-in lacks, another plug-in's name) is replaced by 'unknown' and flagged"""
        r = self.reader.read(text, self._rctx(ctx))
        bad = validate_reading(r, plugin=self.P) if isinstance(r, dict) else ["not a reading"]
        if bad:
            ctx["flags"].append("reader invalid: " + "; ".join(bad))
            lang = ((r.get("slots") or {}).get("language") if isinstance(r, dict) else None) or "en"
            r = dict(blank(self.P.name, lang if lang in ("en", "toi", "mixed") else "en"),
                     source="invalid:" + str((r or {}).get("source") if isinstance(r, dict) else "?"),
                     why="; ".join(bad))
        return r

    def _pick_plugin(self, ctx, text):
        """the minimal router (item 6): with more than one plug-in, each one's RULES reading (no model) is taken;
        the best confidence among readings that are not 'unknown' wins; ties and the plug-in-independent phrases
        (start over, continue, name, language, school, a class, the bare 'menu') stay with the active plug-in"""
        if len(self.plugins) < 2:
            return
        scored = []
        for P in self.plugins.values():
            r = self._router.read(text, self._rctx(ctx, P))
            if r["intent"] in GENERIC_INTENTS or (r["intent"] == "menu" and r["source"] == "rules:phrase"):
                return
            if r["intent"] != "unknown" and not validate_reading(r, plugin=P):
                scored.append((r["confidence"], P.name == self.P.name, P.name))
        if scored:
            best = max(scored)[2]
            if best != self.P.name:
                ctx["P"] = self.plugins[best]
                ctx["path"].append(f"router:{best}")

    # ------------------------------------------------------------------------------------------------ greeting, resume
    def _greet(self, ctx):
        st, prof = ctx["st"], ctx["prof"]
        ctx["path"].append("greet")
        if ctx["created"] or (not prof.get("name") and st.get("step") in (None, "idle", "onboard_name")
                              and not self.mem.classes(ctx["n"])):
            st["step"] = "onboard_name"
            return [IO.buttons(self.t("greet_new"), [("c:skip_name", self.b("skip"))])]
        hello = self.t("greet_known", name=prof["name"]) if prof.get("name") else self.t("hello")
        res = self._resume(ctx)
        if res:
            return [IO.text(hello)] + res
        if not prof.get("language_pref"):
            return [IO.text(hello)] + self._ask_language(ctx)
        return [self.offer(hello + " " + self.t("task_offer"))]

    def _what(self, unit_id, item_id=None):
        u = self.P.units.get(unit_id) if unit_id else None
        if u:
            return (f"Grade {u['grade']} " if u.get("grade") else "") + f"{u['subject_label']}, {u['label']}"
        it = self.P.items.get(item_id) if item_id else None
        return it["title"] if it else "your last lesson"

    def _resume(self, ctx):
        st = ctx["st"]
        step = st.get("step")
        if step == "deliver" and st.get("item") and not st.get("finished"):
            ctx["path"].append("resume:deliver")
            return [IO.buttons(self.t("resume_deliver", what=self._what(st.get("unit"), st["item"]),
                                        part=st.get("part", 0) + 1, parts=st.get("parts", 1)),
                               [("c:continue", self.b("continue")), ("c:else", self.b("something_else")),
                                ("c:restart", self.b("start_over"))])]
        if step == "deliver" and st.get("finished"):
            nu = self.P.next_unit(st.get("unit")) if st.get("unit") else None
            if nu:
                ctx["path"].append("resume:next")
                return [IO.buttons(self.t("resume_finished", what=self._what(st.get("unit"), st.get("item")),
                                            next=nu["label"]),
                                   [("c:next", self.b("next")), ("c:else", self.b("something_else")),
                                    ("c:restart", self.b("start_over"))])]
            return None
        if step in ("subject", "unit", "item", "grade", "class", "confirm", "plugin"):
            subj = self.P.subject(self._g(st), st.get("subject")) if st.get("subject") else None
            what = {"subject": "a subject", "grade": "a grade", "class": "a class", "plugin": "an area",
                    "unit": f"a {subj['label']} unit" if subj else "a unit",
                    "item": f"from {self._what(st.get('unit'))}", "confirm": "a lesson"}[step]
            ctx["path"].append(f"resume:{step}")
            return [IO.buttons(self.t("resume_step", what=what),
                               [("c:resume_step", self.b("continue")), ("c:else", self.b("something_else")),
                                ("c:restart", self.b("start_over"))])]
        return None

    def _ask_language(self, ctx):
        ctx["st"]["step"] = "onboard_lang"
        return [IO.buttons(self.t("ask_language"), [("lang:en", self.b("lang_en")), ("lang:toi", self.b("lang_toi")),
                                                       ("lang:both", self.b("lang_both"))])]

    def _ask_grade_new_class(self, ctx, label=None):
        ctx["st"]["step"] = "onboard_grade"
        ctx["st"]["new_class_label"] = label
        rows = [(f"newgrade:{g}", f"Grade {g}", "") for g in range(1, 8)]
        rows += [("newgrade:1,2", "Grades 1 and 2", "multigrade"), ("newgrade:2,3", "Grades 2 and 3", "multigrade")]
        return [IO.listing(self.t("ask_grade"), rows, button="Grade")]

    def _not_a_name(self, text):
        """True when a reply to 'What is your name?' is plainly not a name: it has a digit ('Grade 1'), or a word that is
        a subject or a unit keyword of the plug-in ('Maths', 'Literacy', 'Counting') or a common school / chat word
        ('Lesson', 'Ok', 'Help'). Ordinary English words that are also Zambian names (Grace, Precious, Given, Memory,
        Gift) are NOT in the list on purpose: a name is the default for a short unknown reply."""
        t = (text or "").strip().lower()
        if not t or re.search(r"\d", t):
            return bool(t)
        if self._nonname is None:
            ws = set(NOT_NAMES)
            for P in self.plugins.values():
                intake = getattr(P, "intake", {}) or {}
                for words in (intake.get("subjects") or {}).values():
                    ws |= {w.lower() for w in words if " " not in w.strip()}
                for kw in (intake.get("units") or {}).values():
                    ws |= {w.lower().strip() for w in kw.get("en", []) if " " not in w.strip()}
            self._nonname = ws
        return any(w in self._nonname for w in re.findall(r"[a-z']+", t))

    def _set_name(self, ctx, name):
        self.mem.update_teacher(ctx["n"], name=name.strip()[:40])
        ctx["path"].append("memory:name")
        out = [IO.text(self.t("name_set", name=name.strip()))]
        prof = self.mem.get_teacher(ctx["n"])
        if not prof.get("language_pref"):
            return out + self._ask_language(ctx)
        if not self.mem.classes(ctx["n"]):
            return out + self._ask_grade_new_class(ctx)
        ctx["st"]["step"] = "idle"
        return out + [self.offer()]

    def _new_class(self, ctx, text):
        label = re.sub(r"(?i)\b(add|a|new|another|class|i\s+also\s+teach|grades?\s*\d(\s*(and|&|,)\s*\d)?)\b", " ", text)
        label = re.sub(r"[\s,.]+", " ", label).strip() or None
        gs = grades_in(text)
        if gs:
            c = self.mem.add_class(ctx["n"], label or "Grade " + " and ".join(map(str, gs)), gs, plugin=self.P.name)
            self._set_class(ctx, c["id"])
            ctx["st"]["step"] = "idle"
            ctx["path"].append("memory:class_added")
            return [IO.text(self.t("class_added", class_label=c["label"],
                                     grades_label=_glabel(c["grades"]))), self.offer()]
        return self._ask_grade_new_class(ctx, label)

    # ------------------------------------------------------------------------------------------------ taps
    def _choice(self, ctx, cid, msg):
        st, n = ctx["st"], ctx["n"]
        kind, _, rest = cid.partition(":")
        if kind == "plug":
            if rest not in self.plugins:
                ctx["flags"].append(f"unknown plug-in {rest}")
                return [IO.text(self.t("unclear"))]
            ctx["P"] = self.plugins[rest]
            ctx["plugin_chosen"] = True
            for k in PATH_KEYS:
                st.pop(k, None)
            ctx["path"].append(f"plugin:{rest}")
            return self._start_menu(ctx)
        if kind in ("unit", "item") and rest not in (self.P.units if kind == "unit" else self.P.items):
            # a tap on another plug-in's id (an older menu): switch to the plug-in that has it
            other = [P for P in self.plugins.values() if rest in (P.units if kind == "unit" else P.items)]
            if len(other) == 1:
                ctx["P"] = other[0]
                ctx["path"].append(f"router:{other[0].name}")
        if kind == "lang":
            self.mem.update_teacher(n, language_pref=rest)
            ctx["lang"] = self._ui_lang(rest)
            ctx["path"].append("memory:language")
            out = [IO.text(self.t("lang_set", lang_label=LANG_LABEL[rest]))]
            if not self.mem.classes(n):
                return out + self._ask_grade_new_class(ctx)
            if st.get("step") == "onboard_lang":
                st["step"] = "idle"
            return out + [self.offer()]
        if kind == "newgrade":
            gs = [int(x) for x in rest.split(",")]
            label = st.pop("new_class_label", None) or ("Grade " + " and ".join(map(str, gs)))
            c = self.mem.add_class(n, label, gs, plugin=self.P.name)
            self._set_class(ctx, c["id"])
            st["step"] = "idle"
            ctx["path"].append("memory:class_added")
            return [IO.text(self.t("class_added", class_label=c["label"],
                                     grades_label=_glabel(c["grades"]))), self.offer()]
        if kind == "class":
            self._set_class(ctx, int(rest))
            pend = st.pop("pending_class", None)
            st["step"] = "idle"
            return self._after_class(ctx, pend)
        if kind == "grade":
            st["grade"] = int(rest)
            sid = st.pop("grade_for_subject", None)       # the subject named before the multigrade question
            if sid and self.P.subject(int(rest), sid):
                return self._units(ctx, int(rest), sid)
            return self._subjects(ctx, int(rest))
        if kind == "subj":
            st["subject"] = rest
            return self._units(ctx, st.get("grade") or 1, rest)
        if kind == "more":
            _, what, g, s, page = cid.split(":")
            return self._units(ctx, int(g) or None, s, int(page))
        if kind == "unit":
            return self._unit(ctx, rest)
        if kind == "item":
            return self._deliver(ctx, rest, 0)
        if kind == "fb":
            sid, outcome = rest.split(":")
            self.mem.record_outcome(int(sid), outcome)
            ctx["path"].append(f"followup:{outcome}")
            return [IO.text(self.t("followup_thanks"))]
        if kind == "c":
            return self._control(ctx, rest)
        ctx["flags"].append(f"unknown choice {cid}")
        return [IO.text(self.t("unclear"))]

    def _control(self, ctx, what):
        st, n = ctx["st"], ctx["n"]
        if what == "skip_name":
            return self._ask_language(ctx) if not ctx["prof"].get("language_pref") else [self.offer()]
        if what == "lang":
            return self._ask_language(ctx)
        if what in ("lesson_help", "menu", "else"):
            if what == "else":
                for k in ("item", "unit", "part", "parts", "finished", "pending", "subject", "delivered_from"):
                    st.pop(k, None)
                st["step"] = "idle"
                return [self.offer()]
            return self._start_menu(ctx)
        if what == "struggle":
            st["step"] = "struggle_text"
            return [IO.text(self.t("ask_struggle"))]
        if what == "restart":
            keep = {k: st.get(k) for k in ("session", "class_id", "class_session")}
            if "fw" in st:
                keep["fw"] = st["fw"]             # the flywheel's settings survive 'start over' (feedback off stays off)
            st.clear()
            st.update(keep, step="idle")
            ctx["path"].append("start_over")
            return [self.offer(self.t("started_over") + " " + self.t("task_offer"))]
        if what == "continue":
            if st.get("step") == "deliver" and st.get("item") and not st.get("finished"):
                return self._deliver(ctx, st["item"], st.get("part", 0) + 1)
            return self._resume_step(ctx) or [self.offer()]
        if what == "resume_step":
            return self._resume_step(ctx) or [self.offer()]
        if what == "next":
            nu = self.P.next_unit(st.get("unit")) if st.get("unit") else None
            return self._unit(ctx, nu["id"]) if nu else [self.offer()]
        if what in ("yes", "no"):
            pend = st.pop("pending", None) or {}
            if what == "yes" and pend.get("item"):
                ctx["path"].append("confirm:yes")
                return self._deliver(ctx, pend["item"], 0, texts=[pend.get("text", "")])
            if what == "yes" and pend.get("unit"):
                return self._unit(ctx, pend["unit"])
            ctx["path"].append("confirm:no")
            rows = self._alt_rows(pend.get("alternatives", []), pend.get("candidates", []), pend.get("item"))
            if rows:
                return [IO.choices(self.t("perhaps"), rows + [("c:menu", self.b("menu"), "")])]
            return self._start_menu(ctx)
        if what == "not_this":
            # item 3: the reading that delivered this lesson straight from free text was wrong. The tap is a labelled
            # reader error (log 'label'); the request is remembered as 'not_this' so recall stops serving it.
            df = st.pop("delivered_from", None)
            if not df:
                return [self.offer()]
            ctx["label"] = {"kind": "reader_error", "wrong_item": df["item"], "reading": df["reading"],
                            "text": df.get("text"), "plugin": self.P.name}
            ctx["path"].append(f"not_this:{df['item']}")
            self.mem.log_request(n, "lesson", df["item"], outcome="not_this", text=df.get("text"))
            for k in ("item", "part", "parts", "finished"):
                st.pop(k, None)
            st["step"] = "idle"
            rows = self._alt_rows(df.get("alternatives", []), df.get("candidates", []), df["item"])
            if rows:
                return [IO.choices(self.t("not_this"), rows + [("c:menu", self.b("menu"), "")])]
            return [IO.text(self.t("not_this"))] + self._start_menu(ctx)
        if what == "done":
            st["step"] = "idle"
            return [IO.text(self.t("bye"))]
        if what == "mistakes":
            itm = st.get("item")
            s = self.P.strategies.get(itm)
            if not s:
                return [IO.text(self.t("no_checkpoint")), self.offer()]
            body = "Common mistakes:\n" + "\n".join(f"- {m['title']}: \"{m['fix']}\"" for m in s["mistakes"])
            cp = s["checkpoint"]
            body += f"\nCheckpoint: {cp['task']}\nMastered: {cp['mastered']}\nNext step: {cp['next']}"
            if s.get("scope"):
                body += f"\nNote: {s['scope']}"
            ctx["origins"].append(s.get("origin") or "F")   # grade1_new card strategies carry origin C + label
            return [IO.text(body + "\n" + (s.get("label") or "[F: Funda guide, not yet reviewed]")),
                    self._after_buttons(ctx)]
        if what == "challenge":
            s = self.P.strategies.get(st.get("struggle_item") or st.get("item"))
            if not s:
                return [IO.text(self.t("no_challenge")), self.offer()]
            body, org = R.render_challenge(self.P, s, ctx["prof"].get("language_pref") or "en")
            ctx["origins"] += org
            if st.get("class_id"):
                self.mem.add_suggestion(st["class_id"], self.P.name, st.get("observation_id"), s["lesson"],
                                        "challenge", st.get("struggle_mistake"), what="challenge track")
            return [IO.text(body), self.offer(self.t("anything_else"))]
        if what == "add_class":
            st["step"] = "new_class_label"
            return [IO.text(self.t("ask_class_label"))]
        ctx["flags"].append(f"unknown control {what}")
        return [self.offer()]

    def _resume_step(self, ctx):
        st = ctx["st"]
        step = st.get("step")
        if step == "subject":
            return self._subjects(ctx, self._g(st))
        if step == "unit" and st.get("subject"):
            return self._units(ctx, self._g(st), st["subject"], st.get("page", 0))
        if step == "item" and st.get("unit"):
            return self._unit(ctx, st["unit"])
        if step in ("grade", "class", "plugin"):
            return self._start_menu(ctx)
        if step == "confirm" and st.get("pending"):
            return [self._confirm_reply(st["pending"])]
        return None

    # ------------------------------------------------------------------------------------------------ the menu
    def _g(self, st):
        """the grade a menu step works in: the state's, else 1 for a graded plug-in, None for a flat one"""
        return st.get("grade") or (1 if self.P.graded else None)

    def _alt_rows(self, alternatives, candidates, exclude):
        rows = [(f"unit:{a}", self.P.units[a]["label"], self.P.units[a]["subject_label"])
                for a in alternatives if a in self.P.units]
        rows += [(f"item:{c}", self.P.items[c]["title"][:24], self.P.items[c]["kind"])
                 for c in candidates if c != exclude and c in self.P.items
                 and not self.P.items[c]["id"].startswith("route:")][:6]
        return rows

    def _start_menu(self, ctx, grade=None):
        if len(self.plugins) > 1 and not ctx.get("plugin_chosen"):
            ctx["st"]["step"] = "plugin"
            ctx["path"].append("ask:plugin")
            return [IO.choices(self.t("ask_plugin"), [(f"plug:{P.name}", P.label, "") for P in self.plugins.values()])]
        if self.P.memory_unit != "class" or not self.P.graded:
            return self._subjects(ctx, None if not self.P.graded else (grade or 1))
        c, ask = self._class(ctx, grade, ask_for={"do": "menu"})
        if ask:
            return [ask]
        return self._after_class(ctx, {"do": "menu", "grade": grade})

    def _after_class(self, ctx, pend):
        pend = pend or {"do": "menu"}
        st = ctx["st"]
        c = self.mem.get_class(st["class_id"])
        if pend.get("do") == "deliver":
            return self._deliver(ctx, pend["item"], 0, texts=pend.get("texts", []))
        if pend.get("do") == "struggle":
            return self._struggle(ctx, pend["text"], pend.get("unit"))
        grade = pend.get("grade")
        if not grade and len(c["grades"]) > 1:
            st["step"] = "grade"
            ctx["path"].append("ask:grade(multigrade)")
            return [IO.choices(self.t("ask_grade_multi"), [(f"grade:{g}", f"Grade {g}") for g in c["grades"]])]
        return self._subjects(ctx, grade or c["grades"][0])

    def _subjects(self, ctx, grade):
        st = ctx["st"]
        st["grade"] = grade
        subs = self.P.subjects(grade)
        if not subs:
            ctx["path"].append(f"no_content:grade{grade}")
            have = self.P.grades()
            return [IO.choices(self.t("no_content_grade", grade=grade), [(f"grade:{g}", f"Grade {g}") for g in have])]
        st["step"] = "subject"
        return [IO.choices(self.t("ask_subject"), [(f"subj:{s['id']}", s["label"]) for s in subs])]

    def _units(self, ctx, grade, sid, page=0):
        st = ctx["st"]
        st.update(step="unit", subject=sid, grade=grade, page=page)
        s = self.P.subject(grade, sid)
        if not s:
            return self._subjects(ctx, grade)
        # a unit may carry its own status (Chitonga from print: 'ready', 'thin: sy', 'no print'); else items decide
        rows = [(f"unit:{u['id']}", u["label"], (u.get("status") or ("ready" if u["items"] else "no material yet")) +
                 (f" | {u['ref']}" if u.get("ref") else "")) for u in s["units"]]
        return [IO.listing(self.t("ask_unit", grade_label=f"Grade {grade}" if grade else self.P.label,
                                  subject_label=s["label"]), rows,
                           button="Units", page=page, more_id=f"more:units:{grade or 0}:{sid}")]

    def _unit(self, ctx, uid):
        st = ctx["st"]
        u = self.P.units.get(uid)
        if not u:
            return [IO.text(self.t("unclear"))]
        st.update(unit=uid, subject=u["subject"], grade=u["grade"])
        if not u["items"]:
            ctx["path"].append(f"no_content:unit:{uid}")
            have = self.P.units_with_content(u["grade"], u["subject"])
            note = f" ({u['note']})" if u.get("note") else ""
            rows = [(f"unit:{x['id']}", x["label"], x.get("ref") or "") for x in have]
            st["step"] = "unit"
            return [IO.choices(self.t("no_content_unit", unit_label=u["label"], subject_label=u["subject_label"])
                               + note, rows, more_id=f"more:units:{u['grade'] or 0}:{u['subject']}")]
        if len(u["items"]) == 1:
            return self._deliver(ctx, u["items"][0], 0)
        st["step"] = "item"
        rows = [(f"item:{i}", {"lesson": "Lesson", "sample": "Sample lesson"}.get(self.P.items[i]["kind"], "Card") + ": " +
                 self.P.items[i]["title"], "") for i in u["items"]]
        return [IO.listing(self.t("ask_item", unit_label=u["label"]), rows, button="Choose")]

    # ------------------------------------------------------------------------------------------------ delivery
    def _after_buttons(self, ctx):
        st = ctx["st"]
        # part 1 of a lesson read straight from free text: "Not this one" (item 3) takes the Mistakes slot (3 at most)
        not_this = (st.get("part") == 0 and (st.get("delivered_from") or {}).get("item") == st.get("item"))
        if st.get("step") == "deliver" and not st.get("finished"):
            items = [("c:continue", self.b("continue"))]
            if not_this:
                items.append(("c:not_this", self.b("not_this")))
            elif st.get("item") in self.P.strategies:
                items.append(("c:mistakes", self.b("mistakes")))
            items.append(("c:else", self.b("something_else")))
            return IO.buttons(self.t("continue_q"), items)
        nu = self.P.next_unit(st.get("unit")) if st.get("unit") else None
        first = ([("c:not_this", self.b("not_this"))] if not_this else
                 [("c:next", ("Next: " + nu["label"])[:20])] if nu else [])
        items = first + [("c:menu", self.b("menu")), ("c:done", self.b("done"))]
        return IO.buttons(self.t("anything_else"), items)

    def _deliver(self, ctx, item_id, part, texts=()):
        st, n = ctx["st"], ctx["n"]
        it = self.P.items.get(item_id)
        if not it:
            return [IO.text(self.t("unclear"))]
        if it["kind"] == "route":
            return self._route(ctx, it, (ctx["msg"].get("text") or ""))
        if it.get("test_only") and not self.allow_test:
            ctx["flags"].append(f"test-only item {item_id} refused")
            return [IO.text(self.t("unclear"))]
        verdict, why = G.item_allowed(self.P.rules, it, list(texts) + [ctx["msg"].get("text") or ""])
        if verdict != "ok" or it.get("needs_review"):
            ctx["fw_block"] = True                   # no feedback question on a held, health, status or labelled answer
        if verdict == "hold":
            ctx["path"].append(f"hold:{item_id}")
            self.mem.event(n, "safeguarding_referral", note=f"held item {item_id}")
            ref = G.referral(why)
            return [IO.text(ref["reply"] or ref["en_meaning"])]
        if verdict.endswith("_no_card"):               # a status rule (gates.status_rules): no checked material
            ctx["path"].append(f"{verdict}:{item_id}")
            self.mem.log_request(n, "lesson", item_id, outcome=verdict)
            return [self.offer(self.t(verdict) or self.t("no_material"))]
        pre = [IO.text(self.t(verdict))] if verdict.endswith("_plain") and part == 0 else []
        cls = None
        if it["kind"] != "pointer" and self.P.memory_unit == "class":
            cls, ask = self._class(ctx, (it.get("grades") or [None])[0] if it["kind"] == "lesson" else None,
                                   ask_for={"do": "deliver", "item": item_id, "texts": list(texts)})
            if ask:
                return [ask]
        pref = ctx["prof"].get("language_pref") or self.mem.get_teacher(n).get("language_pref") or "en"
        strategy = self.P.strategies.get(item_id)
        body, origins, flags, nparts = R.render(self.P, it, part, pref, strategy)
        ctx["origins"] += origins
        ctx["flags"] += flags
        unit = self.P.unit_of.get(item_id) or st.get("unit")
        finished = part >= nparts - 1
        st.update(step="deliver", item=item_id, unit=unit, part=part, parts=nparts, finished=finished)
        if part == 0:
            r = ctx.get("text_sourced")
            if r and r.get("item") == item_id:      # item 3: what the "Not this one" tap needs, nothing more
                st["delivered_from"] = {"item": item_id, "text": (ctx["msg"].get("text") or "")[:200],
                                        "reading": {k: r.get(k) for k in ("intent", "unit", "confidence", "source",
                                                                          "why")},
                                        "candidates": list(r.get("candidates") or [])[:8],
                                        "alternatives": list(r.get("alternatives") or [])[:3]}
            else:
                st.pop("delivered_from", None)
        if unit:
            st["subject"] = self.P.units[unit]["subject"]
        if cls:
            self.mem.set_position(cls["id"], self.P.name, st.get("subject") or it.get("subject") or "?", unit=unit,
                                  item=item_id, part=part, parts=nparts, finished=finished)
        ctx["path"].append(f"deliver:{item_id}:{part + 1}/{nparts}")
        if part == 0:
            self.mem.log_request(n, "lesson", item_id, outcome="served", text=ctx["msg"].get("text"))
        return pre + [IO.text(body), self._after_buttons(ctx)]

    def _route(self, ctx, it, text):
        ctx["path"].append(f"route:{it['id']}")
        hk = self._hooks()
        res = None
        if hk:
            pref = ctx["prof"].get("language_pref") or "en"
            # 24 Oct 2026: a hook that takes `ctx` (education's route:syllabus) gets the plug-in, the class's grades and
            # the turn's clock; the older hooks are called exactly as before
            if "ctx" in inspect.signature(hk.run).parameters:
                cls = self._current_class(ctx)
                res = hk.run(it["id"], text, pref, ctx={"plugin": self.P, "grades": cls["grades"] if cls else None,
                                                        "now": ctx["now"]})
            else:
                res = hk.run(it["id"], text, pref)
        self.mem.log_request(ctx["n"], it["id"].split(":")[1], it["id"], outcome="served" if res else "no_hook")
        ctx["st"].update(step="idle")
        if not res:
            ctx["path"].append("no_material")
            return [IO.text(self.t("no_material")), self.offer()]
        ctx["origins"] += res.get("origins", [])
        ctx["flags"] += res.get("flags", [])
        return [IO.text(res["body"]), self.offer(self.t("anything_else"))]

    def _confirm_reply(self, pend):
        what = pend.get("what")
        return IO.buttons(self.t("confirm", what=what), [("c:yes", self.b("yes")), ("c:no", self.b("no")),
                                                         ("c:menu", self.b("menu"))])

    # ------------------------------------------------------------------------------------------------ readings
    def _intent(self, ctx, r, text):
        st, n = ctx["st"], ctx["n"]
        it = r["intent"]
        if st.get("step") == "struggle_text" and it not in ("start_over", "menu"):
            it = "struggle"
        if it == "start_over":
            return self._control(ctx, "restart")
        if it == "menu":
            if r.get("subject") and self.P.memory_unit != "class":
                return self._units(ctx, None, r["subject"])
            if r.get("subject"):
                c, ask = self._class(ctx, None, ask_for={"do": "menu"})
                if ask:
                    return [ask]
                g = st.get("grade") if st.get("grade") in c["grades"] else None
                if not g and len(c["grades"]) > 1:      # a multigrade class and no grade yet: ask, do not guess
                    st.update(step="grade", grade_for_subject=r["subject"])
                    ctx["path"].append("ask:grade(multigrade)")
                    return [IO.choices(self.t("ask_grade_multi"), [(f"grade:{x}", f"Grade {x}") for x in c["grades"]])]
                return self._units(ctx, g or c["grades"][0], r["subject"])
            return self._start_menu(ctx)
        if it == "continue":
            return self._control(ctx, "continue")
        if it == "set_name":
            return self._set_name(ctx, r["slots"]["topic"])
        if it == "set_language":
            return self._choice(ctx, "lang:" + r["slots"]["topic"], {})
        if it == "set_school":
            v = json.loads(r["slots"]["topic"])
            self.mem.update_teacher(n, school=v["school"][:60], **({"district": v["district"][:40]} if v.get("district") else {}))
            ctx["path"].append("memory:school")
            return [IO.text(self.t("noted")), self.offer()]
        if it == "add_class":
            return self._new_class(ctx, text)
        if it == "struggle":
            st["step"] = "idle"
            return self._struggle(ctx, text, r.get("unit"))
        if r.get("item") and str(r["item"]).startswith("route:"):   # a plug-in route (education: language, marking)
            return self._route(ctx, self.P.items[r["item"]], text)
        if it == "lesson":
            if r.get("unit") and not self.P.units[r["unit"]]["items"]:
                self.mem.log_request(n, "lesson", None, outcome="no_content_unit")
                return self._unit(ctx, r["unit"])
            item = r.get("item")
            if not item:
                return self._unit(ctx, r["unit"]) if r.get("unit") else [IO.text(self.t("unclear")), self.offer()]
            if needs_confirm(r):
                u = self.P.units.get(r.get("unit")) if r.get("unit") else None
                title = self.P.items[item]["title"]
                what = (f"the Grade {u['grade']} {u['subject_label']} lesson '{title}'" if u else f"'{title}'")
                st["pending"] = {"item": item, "unit": r.get("unit"), "what": what, "text": text,
                                 "alternatives": r.get("alternatives", []), "candidates": r.get("candidates", [])}
                st["step"] = "confirm"
                ctx["path"].append(f"confirm:{item}")
                return [self._confirm_reply(st["pending"])]
            ctx["text_sourced"] = r
            return self._deliver(ctx, item, 0, texts=[text])
        # unknown: health first (never drafted), else honest no-material; the record says a model reader would be tried
        ctx["path"].append("needs_model")
        topic = G.status_hit(self.P.rules, text, str(r.get("read_en") or ""))
        if topic:
            ctx["path"].append(f"{topic}_no_card")        # the gap marker the flywheel's backlog reads (no text kept)
            self.mem.log_request(n, "unknown", outcome=f"{topic}_no_card")
            return [self.offer(self.t(f"{topic}_no_card") or self.t("no_material"))]
        ctx["path"].append("no_material")                # the gap marker of the content backlog (flywheel)
        self.mem.log_request(n, "unknown", outcome="no_material")
        return [self.offer(self.t("no_material"))]

    # ------------------------------------------------------------------------------------------------ struggles
    def _match_mistake(self, text, unit):
        """(strategy, mistake or None, score).
        1 the plug-in's struggle PATTERNS (23 Oct, stream A): every common mistake and struggle topic of every lesson
          carries regexes over the teacher's line ('count past 20', 'skip counting', 'forget the zero in 106'); the
          highest weight wins, +1 when the lesson belongs to the unit the intake read. Shared words alone had sent
          'cannot count past 20' to 'One touch, two numbers' (the word 'count') and many lines to an English lesson.
        2 word overlap: within the unit's lesson >= 1 shared content word, across all lessons >= 2 (one word is not evidence);
        3 the unit's lesson with no named mistake (the general intervention), else nothing (honest 'no strategy')."""
        low = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
        unit_items = set(self.P.units[unit]["items"]) if unit in self.P.units else set()
        best = (None, None, 0)
        for s in self.P.strategies.values():
            for m in list(s.get("mistakes", [])) + list(s.get("struggles", [])):
                pats = m.get("patterns") or []
                if pats and any(re.search(rx, low) for rx in pats):
                    sc = m.get("weight", 2) + (1 if s["lesson"] in unit_items else 0)
                    if sc > best[2]:
                        best = (s, m, sc)
        if best[1]:
            return best
        q = _words(text)
        pools = []
        if unit_items:
            pools.append([self.P.strategies[i] for i in self.P.units[unit]["items"] if i in self.P.strategies])
        pools.append(list(self.P.strategies.values()))
        for k, pool in enumerate(pools):
            need = 1 if (unit_items and k == 0) else 2      # inside the unit's own lesson one shared word may do
            best = (None, None, 0)
            for s in pool:
                for m in s.get("mistakes", []):
                    sc = len(q & _words(m["title"] + " " + m["text"]))
                    if sc >= need and sc > best[2]:
                        best = (s, m, sc)
            if best[1]:
                return best
            if unit_items and pool:
                return pool[0], None, 0
        return None, None, 0

    def _struggle(self, ctx, text, unit):
        st, n = ctx["st"], ctx["n"]
        why = pupil_level(text)
        if why:
            ctx["withhold"] = f"pupil-level data refused ({why}); text not stored"
            ctx["path"].append("pupil_refused")
            return [IO.text(self.t("pupil_refused"))]
        c, ask = self._class(ctx, None, ask_for={"do": "struggle", "text": text, "unit": unit})
        if ask:
            return [ask]
        if not unit:
            pos = self.mem.positions(c["id"])
            unit = pos[0]["unit"] if pos else None
        s, m, score = self._match_mistake(text, unit)
        subject = self.P.units[unit]["subject"] if unit in self.P.units else None
        oid = self.mem.add_observation(c["id"], self.P.name, text, unit=unit, subject=subject,
                                       mistake=m["id"] if m else None)
        ctx["path"].append(f"observation:{oid}" + (f":{m['id']}" if m else ""))
        if not s:
            return [IO.text(self.t("struggle_unmatched", class_label=c["label"])), self.offer()]
        pref = ctx["prof"].get("language_pref") or "en"
        body, org = R.render_intervention(self.P, s, m, pref)
        ctx["origins"] += org
        what = m["title"] if m else self.P.items[s["lesson"]]["title"]
        sid = self.mem.add_suggestion(c["id"], self.P.name, oid, s["lesson"], "intervention", m["id"] if m else None,
                                      what=what)
        st.update(struggle_item=s["lesson"], struggle_mistake=m["id"] if m else None, observation_id=oid, step="idle",
                  item=s["lesson"], unit=unit or self.P.unit_of.get(s["lesson"]))
        ctx["path"].append(f"suggestion:{sid}")
        ack = self.t("struggle_ack", class_label=c["label"], observation=text.strip().rstrip("."))
        return [IO.text(ack + "\n" + body),
                IO.buttons(self.t("anything_else"), [("c:challenge", self.b("challenge")),
                                                       ("c:mistakes", self.b("mistakes")),
                                                       ("c:else", self.b("something_else"))])]
