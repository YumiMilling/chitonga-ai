"""The flywheel inside the engine: every teacher interaction can yield labelled Tonga, with as little friction as we can
manage (24 Oct 2026, the Jesuit; Chris: "I want to adjust the harness for the flywheel, getting inputs from the
teachers"; the one-page account is tools/chitonga/harness/FLYWHEEL.md).

WHAT THIS MODULE DOES. After a turn, it may add ONE short feedback prompt (never two), and it reads the teacher's answer
to it. It writes nothing to disk: every prompt and answer is an EVENT in the turn record (rec["flywheel"]), and the
adapter's FlywheelWriter (adapter/flywheel_log.py) is the one writer that stores it, with the teacher's code, consent,
held-out flag and pay. The engine stays transport-free, as before.

THE PROMPTS (kind -> when -> buttons -> what an answer becomes):
  tonga_check      a lesson part showed Tonga lines, all of them teacher / book / lexicon (T/B/L), and this lesson was
                   not yet checked in this session -> [Tonga OK] [Fix a line] [Stop asking]
                   OK = +1 confidence on the shown lines (reinforcement, unpaid: a one-tap opinion on up to six lines);
                   Fix = "send the line number and your Tonga" -> the teacher's version, VERBATIM, beside the served
                   line (a contradiction is recorded beside it, never overwritten; K2)
  derived_confirm  the part showed a D line (derived from teacher input, needs confirmation) -> the line quoted with
                   its English -> [Right as is] [Write it] [Stop asking]; Right = teacher-confirmed (same weight as
                   authored, Chris 24 Oct; K1, the desk's 'ok' rate); Write = the teacher's own line (K2)
  write_line       the part held a line's Tonga back (its checks rang) -> [Write it] [Not now] [Stop asking] (K2)
  unit_line        a unit with no material yet -> "write one line in Tonga for it?" -> [Write a line] [Not now]
                   [Stop asking] (K2); "Tonga = English" is accepted, the English is optional
CONTROLS (hidden): about 1 in 10 contribution prompts (derived_confirm / write_line / unit_line) is replaced by a
  control item from flywheel_controls.jsonl, shown EXACTLY as a derived_confirm prompt: a known-right line (another
  speaker's teacher line) to confirm, or a known-wrong line (a teacher line with one number changed, so the Tonga no
  longer says what the English says) to fix. Paid as the prompt it looks like. The reply never says it is a control;
  the event carries the control id and the expected answer for scoring.
NEVER: on a turn with safeguarding, a held item, a health / status path, a redirect, a withheld message, encouragement,
  an engine error, or the struggle follow-up ("Did it help?" is already this turn's one question); never while the
  teacher has turned prompts off ("Stop asking", or typing "feedback off"; "feedback on" turns them back on); never
  in the turn right after a prompt was passed by, nor after three passed by in one session; never a paid prompt
  beyond the daily cap (20 paid items per teacher).
FREE TEXT while a "Fix" or "Write" waits: the safeguarding gate runs FIRST (engine._turn step 1); then a numbered line
  ("3 <Tonga>") or the line itself is taken verbatim; anything else (a new request, a menu word) closes the wait and
  is read as a normal message, so a teacher is never trapped in a feedback step.
"""
import datetime
import hashlib
import json
import os
import re
import uuid

from . import io as IO
from . import render as R

PAY = {"tonga_ok": 0, "fix": 2, "right": 1, "write": 2}     # kwacha per answer; the desk's K1 ok / K2 sentence
CONTRIBUTION_KINDS = ("derived_confirm", "write_line", "unit_line")
DAILY_PAID_CAP = 20
CONTROL_EVERY = 10              # about 1 in 10 contribution prompts is a hidden control
IGNORE_LIMIT = 3                # ignored prompts in one session before the flywheel goes quiet for that session
BLOCK_PATH = ("safeguarding", "hold:", "redirect", "gate:", "pupil_refused", "encourage", "health", "_no_card",
              "followup:asked", "stale:")
FEEDBACK_CMD = re.compile(r"(?i)^\s*(feedback|questions|prompts)\s+(off|on)\s*[.!]?\s*$")
STOP_ASKING = re.compile(r"(?i)^\s*(stop\s+asking|no\s+more\s+questions)\s*[.!]?\s*$")
NUMBERED_LINE = re.compile(r"^\s*(?:line\s*)?(\d{1,2})\s*[.):\-]?\s+(\S.*)$", re.S | re.I)
BARE_NUMBER = re.compile(r"^\s*(?:line\s*)?(\d{1,2})\s*[.)]?\s*$", re.I)
COMMAND_WORD = re.compile(r"(?i)^\s*(menu|cancel|stop|skip|back|no|not\s+now|start\s+over|assistant|desk|help|hi|"
                          r"hello|continue)\s*[.!]?\s*$")

BUTTONS = {"ok": "Tonga OK", "fix": "Fix a line", "off": "Stop asking", "right": "Right as is", "write": "Write it",
           "write_unit": "Write a line", "skip": "Not now", "cancel": "Cancel"}
LINES = {
    "tonga_check": "Is the Tonga in this part right (lines {first} to {last})?",
    "derived_confirm": "This Tonga line needs a teacher's check:\n\"{en}\"\n{toi}\nRight as is (K1), or write your own "
                       "line (K2)?",
    "write_line": "Line {n} has no checked Tonga yet:\n\"{en}\"\nWould you like to write it in Tonga? (K2)",
    "unit_line": "Would you like to write one line in Tonga for '{unit_label}'? It helps us build this unit. (K2)",
    "ask_fix": "Which line? Send its number and your Tonga in one message, for example: 3 and then your line. (K2 for "
               "each line you correct)",
    "ask_fix_text": "Please send line {n} in Tonga, after the number.",
    "ask_write": "Please type the Tonga for:\n\"{en}\"",
    "ask_unit_line": "Please type your line in Tonga. You may add the English after an equals sign: Tonga = English.",
    "thanks_ok": "Thank you, noted.",
    "thanks_line": "Thank you. Your line {n} is saved as your own version, beside ours; it is checked before it is "
                   "used.",
    "thanks_write": "Thank you. Your line is saved; it is checked before it is used.",
    "thanks_right": "Thank you, noted as confirmed by you.",
    "off": "OK, I will not ask for feedback any more. Send 'feedback on' to turn it back on.",
    "on": "OK, I will sometimes ask about the Tonga again. Send 'feedback off' to stop.",
    "closed": "That question has closed.",
}


def _today(now):
    return (now or datetime.datetime.now()).date().isoformat()


def load_controls(path):
    """the hidden control items (build_controls.py): [{id, en, toi, expect: right|wrong, author, ...}]"""
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


class Flywheel:
    def __init__(self, controls=None, who=None, control_every=CONTROL_EVERY, daily_cap=DAILY_PAID_CAP, pay=None,
                 enabled=None):
        """controls: list of control items (load_controls) or a path; who(number) -> speaker code (prefers a control
        authored by ANOTHER speaker); control_every: 0 = no controls, 1 = every contribution prompt (tests);
        enabled(number) -> bool: the flywheel speaks to this number at all (run.py keeps it silent towards a number
        whose replies are really SENT, until the main session switches it on: built in shadow, nothing sent)"""
        self.controls = load_controls(controls) if isinstance(controls, str) else list(controls or [])
        self.who = who or (lambda n: None)
        self.every = control_every
        self.cap = daily_cap
        self.pay = dict(PAY, **(pay or {}))
        self.enabled = enabled or (lambda n: True)

    # ------------------------------------------------------------------------------------------------ state
    @staticmethod
    def fw(ctx):
        """the flywheel's part of the conversation state (kept across 'start over', engine._control)"""
        return ctx["st"].setdefault("fw", {})

    def _event(self, ctx, ev, **kw):
        rec = dict({"ev": ev, "at": ctx["now"].isoformat(timespec="seconds")}, **kw)
        ctx.setdefault("flywheel", []).append(rec)
        return rec

    def waiting(self, ctx):
        if not self.enabled(ctx["n"]):
            return False
        o = self.fw(ctx).get("open")
        return bool(o and o.get("await"))

    def clear(self, ctx, why):
        """close an open prompt without an answer (safeguarding: no text is kept, only that it closed)"""
        o = self.fw(ctx).pop("open", None)
        if o:
            self._event(ctx, "closed", pid=o["pid"], kind=o["kind"], why=why)

    def _paid_today(self, ctx):
        p = self.fw(ctx).get("paid") or {}
        return p.get("n", 0) if p.get("date") == _today(ctx["now"]) else 0

    def _count_paid(self, ctx, k):
        if k > 0:
            fw = self.fw(ctx)
            d = _today(ctx["now"])
            fw["paid"] = {"date": d, "n": self._paid_today(ctx) + 1}

    # ------------------------------------------------------------------------------------------------ commands
    def command(self, eng, ctx, text):
        """'feedback off / on', 'stop asking' typed: replies, or None when the text is not such a command"""
        if not self.enabled(ctx["n"]):
            return None
        m = FEEDBACK_CMD.match(text or "")
        off = STOP_ASKING.match(text or "") or (m and m.group(2).lower() == "off")
        if not (m or off):
            return None
        return self._set_off(eng, ctx, bool(off))

    def _set_off(self, eng, ctx, off):
        fw = self.fw(ctx)
        fw["off"] = off
        fw.pop("open", None)
        eng.mem.event(ctx["n"], "feedback_off" if off else "feedback_on")
        self._event(ctx, "setting", feedback="off" if off else "on")
        ctx["path"].append("flywheel:" + ("off" if off else "on"))
        ctx["fw_answered"] = True
        return [IO.text(LINES["off" if off else "on"]), self._nav(eng, ctx)]

    def is_off(self, ctx):
        return bool(self.fw(ctx).get("off"))

    # ------------------------------------------------------------------------------------------------ after a turn
    def after_turn(self, eng, ctx, replies):
        """replies + at most one prompt. Records an open prompt the teacher passed by as 'ignored'."""
        if not self.enabled(ctx["n"]):
            return replies
        fw = self.fw(ctx)
        o = fw.get("open")
        passed_by = False
        if o and not ctx.get("fw_answered") and not ctx.get("fw_touched"):
            fw.pop("open", None)
            self._event(ctx, "ignored", pid=o["pid"], kind=o["kind"], control=o.get("control"))
            q = fw.get("quiet") or {}
            n = (q.get("n", 0) + 1) if q.get("session") == ctx["st"].get("session") else 1
            fw["quiet"] = {"session": ctx["st"].get("session"), "n": n}
            passed_by = True                 # pacing: a question passed by is not followed by another at once
        if passed_by or fw.get("off") or ctx.get("fw_answered") or ctx.get("withhold") or ctx.get("fw_block"):
            return replies
        if any(f.startswith("engine error") for f in ctx.get("flags", [])):
            return replies
        if any(any(b in str(p) for b in BLOCK_PATH) for p in ctx["path"]):
            return replies
        q = fw.get("quiet") or {}
        if q.get("session") == ctx["st"].get("session") and q.get("n", 0) >= IGNORE_LIMIT:
            return replies
        prompt = self._choose(eng, ctx)
        if not prompt:
            return replies
        return replies + [prompt]

    def _deliver_step(self, ctx):
        for p in reversed(ctx["path"]):
            m = re.match(r"deliver:(.+):(\d+)/(\d+)$", str(p))
            if m:
                return m.group(1), int(m.group(2)) - 1
        return None, None

    def _choose(self, eng, ctx):
        st, fw = ctx["st"], self.fw(ctx)
        room = self._paid_today(ctx) < self.cap
        unit_gap = next((str(p).split(":", 2)[2] for p in ctx["path"] if str(p).startswith("no_content:unit:")), None)
        if unit_gap and room and unit_gap in eng.P.units:
            u = eng.P.units[unit_gap]
            return self._open(eng, ctx, "unit_line", unit=unit_gap, item=None, part=None,
                              target={"unit_label": u["label"], "subject": u.get("subject"), "grade": u.get("grade")})
        item_id, part = self._deliver_step(ctx)
        if not item_id:
            return None
        it = eng.P.items.get(item_id)
        if not it or it.get("kind") != "lesson":
            return None
        if any(it.get(r.get("item_flag") or r["name"]) for r in _status_rules(eng.P)):
            return None                              # a health / status item: no feedback questions on it
        pref = ctx["prof"].get("language_pref") or (eng.mem.get_teacher(ctx["n"]) or {}).get("language_pref") or "en"
        lines = R.shown_lines(eng.P, it, part, pref, eng.P.strategies.get(item_id))
        shown = [ln for ln in lines if ln["shown"]]
        held = [ln for ln in lines if ln["held"]]
        done = set(fw.get("done") or [])
        if room:
            h = next((ln for ln in held if f"{item_id}#{ln['n']}" not in done), None)
            if h:
                return self._open(eng, ctx, "write_line", item=item_id, part=part, unit=st.get("unit"),
                                  target=_line(h))
            d = next((ln for ln in shown if "D" in ln["origin"] and f"{item_id}#{ln['n']}" not in done), None)
            if d:
                return self._open(eng, ctx, "derived_confirm", item=item_id, part=part, unit=st.get("unit"),
                                  target=_line(d))
        checked = fw.get("checked") or {}
        if shown and checked.get(item_id) != st.get("session"):
            checked[item_id] = st.get("session")
            fw["checked"] = dict(list(checked.items())[-30:])
            return self._open(eng, ctx, "tonga_check", item=item_id, part=part, unit=st.get("unit"),
                              lines=[_line(x) for x in shown])
        return None

    def _pick_control(self, ctx):
        """a control for this teacher: one they have not seen, preferring another speaker's line"""
        if not self.controls or self.every <= 0:
            return None
        fw = self.fw(ctx)
        k = fw.get("n", 0)
        code = self.who(ctx["n"]) or ctx["n"]
        if self.every > 1:
            # deterministic for the record, unpredictable to the teacher: about 1 in `every`
            hsh = int(hashlib.sha1(f"{code}:{k}".encode()).hexdigest()[:8], 16)
            if hsh % self.every:
                return None
        seen = set(fw.get("controls_seen") or [])
        pool = [c for c in self.controls if c["id"] not in seen]
        if not pool:
            return None
        other = [c for c in pool if c.get("author") not in (code,)]
        pool = other or pool
        c = pool[int(hashlib.sha1(f"{code}:{k}:pick".encode()).hexdigest()[:8], 16) % len(pool)]
        fw["controls_seen"] = (list(seen) + [c["id"]])[-200:]
        return c

    def _open(self, eng, ctx, kind, item=None, part=None, unit=None, target=None, lines=None):
        fw = self.fw(ctx)
        control = None
        if kind in CONTRIBUTION_KINDS:
            c = self._pick_control(ctx)
            fw["n"] = fw.get("n", 0) + 1
            if c:
                control = {"id": c["id"], "expect": c["expect"]}
                kind, item, part, unit = "derived_confirm", None, None, None
                target = {"n": None, "en": c["en"], "toi": c["toi"], "origin": ["D"]}
        pid = uuid.uuid4().hex[:8]
        o = {"pid": pid, "kind": kind, "item": item, "part": part, "unit": unit, "target": target, "lines": lines,
             "control": control, "session": ctx["st"].get("session")}
        fw["open"] = o
        self._event(ctx, "prompt", pid=pid, kind=kind, item=item, part=part, unit=unit, target=target, lines=lines,
                    control=control)
        ctx["path"].append(f"flywheel:prompt:{kind}")
        off = ("fw:off", BUTTONS["off"])
        if kind == "tonga_check":
            ns = [x["n"] for x in lines]
            return IO.buttons(LINES["tonga_check"].format(first=ns[0], last=ns[-1]),
                              [(f"fw:ok:{pid}", BUTTONS["ok"]), (f"fw:fix:{pid}", BUTTONS["fix"]), off])
        if kind == "derived_confirm":
            return IO.buttons(LINES["derived_confirm"].format(en=target["en"], toi=target["toi"]),
                              [(f"fw:right:{pid}", BUTTONS["right"]), (f"fw:write:{pid}", BUTTONS["write"]), off])
        if kind == "write_line":
            return IO.buttons(LINES["write_line"].format(n=target["n"], en=target["en"]),
                              [(f"fw:write:{pid}", BUTTONS["write"]), (f"fw:skip:{pid}", BUTTONS["skip"]), off])
        return IO.buttons(LINES["unit_line"].format(unit_label=target["unit_label"]),
                          [(f"fw:write:{pid}", BUTTONS["write_unit"]), (f"fw:skip:{pid}", BUTTONS["skip"]), off])

    # ------------------------------------------------------------------------------------------------ answers
    def _nav(self, eng, ctx):
        """the navigation the teacher had before the prompt: the lesson's buttons while a lesson is open, else the offer"""
        st = ctx["st"]
        if st.get("step") == "deliver" and st.get("item"):
            return eng._after_buttons(ctx)
        return eng.offer(eng.t("anything_else"))

    def _answer(self, ctx, o, answer, pay_kind=None, teacher_text=None, line=None):
        k = self.pay.get(pay_kind, 0) if pay_kind else 0
        self._count_paid(ctx, k)
        rec = self._event(ctx, "answer", pid=o["pid"], kind=o["kind"], answer=answer, item=o.get("item"),
                          part=o.get("part"), unit=o.get("unit"), served=line or o.get("target"),
                          lines=o.get("lines") if answer == "ok" else None, teacher_text=teacher_text, pay=k,
                          control=o.get("control"))
        if o.get("item") and (line or o.get("target") or {}).get("n") is not None and o["kind"] != "tonga_check":
            fw = self.fw(ctx)
            fw["done"] = (list(fw.get("done") or []) + [f"{o['item']}#{(line or o['target'])['n']}"])[-300:]
        ctx["fw_answered"] = True
        ctx["path"].append(f"flywheel:answer:{o['kind']}:{answer}")
        return rec

    def on_choice(self, eng, ctx, cid):
        """a tap on a flywheel button ('fw:<what>:<pid>' or 'fw:off')"""
        parts = cid.split(":")
        what = parts[1] if len(parts) > 1 else ""
        fw = self.fw(ctx)
        ctx["fw_touched"] = True
        if what == "off":
            return self._set_off(eng, ctx, True)
        if what == "on":
            return self._set_off(eng, ctx, False)
        o = fw.get("open")
        pid = parts[2] if len(parts) > 2 else None
        if not o or (pid and o["pid"] != pid):
            ctx["fw_answered"] = True
            return [IO.text(LINES["closed"]), self._nav(eng, ctx)]
        if what == "cancel":
            fw.pop("open", None)
            self._event(ctx, "closed", pid=o["pid"], kind=o["kind"], why="cancel")
            ctx["fw_answered"] = True
            return [self._nav(eng, ctx)]
        if what == "ok":
            fw.pop("open", None)
            self._answer(ctx, o, "ok", "tonga_ok")
            return [IO.text(LINES["thanks_ok"]), self._nav(eng, ctx)]
        if what == "right":
            fw.pop("open", None)
            self._answer(ctx, o, "right", "right")
            return [IO.text(LINES["thanks_right"]), self._nav(eng, ctx)]
        if what == "skip":
            fw.pop("open", None)
            self._answer(ctx, o, "skip")
            return [self._nav(eng, ctx)]
        if what in ("fix", "write"):
            o["await"] = what
            fw["open"] = o
            ctx["fw_answered"] = True
            ctx["path"].append(f"flywheel:await:{what}")
            if what == "fix":
                body = LINES["ask_fix"]
            elif o["kind"] == "unit_line":
                body = LINES["ask_unit_line"]
            else:
                body = LINES["ask_write"].format(en=o["target"]["en"])
            return [IO.buttons(body, [(f"fw:cancel:{o['pid']}", BUTTONS["cancel"])])]
        ctx["flags"].append(f"unknown flywheel choice {cid}")
        return [IO.text(LINES["closed"]), self._nav(eng, ctx)]

    def on_text(self, eng, ctx, text):
        """free text while a Fix / Write waits: (replies) when taken, None when the text is something else (the wait
        closes and the engine reads the message as usual). The teacher's text is kept VERBATIM (outer spaces only)."""
        fw = self.fw(ctx)
        o = fw.get("open") or {}
        t = (text or "").strip()
        ctx["fw_touched"] = True
        if not t or COMMAND_WORD.match(t) or FEEDBACK_CMD.match(t):
            fw.pop("open", None)
            self._event(ctx, "closed", pid=o.get("pid"), kind=o.get("kind"), why="other message")
            return None
        if o.get("await") == "fix":
            shown = {str(x["n"]): x for x in (o.get("lines") or [])}
            m = NUMBERED_LINE.match(t)
            b = BARE_NUMBER.match(t)
            if b and b.group(1) in shown:
                o["line"] = b.group(1)
                fw["open"] = o
                ctx["fw_answered"] = True
                return [IO.buttons(LINES["ask_fix_text"].format(n=b.group(1)),
                                   [(f"fw:cancel:{o['pid']}", BUTTONS["cancel"])])]
            if m and m.group(1) in shown:
                n, line_text = m.group(1), m.group(2).strip()
            elif o.get("line") in shown:
                n, line_text = o["line"], t
            elif len(shown) == 1 and not m:
                n, line_text = next(iter(shown)), t
            else:
                fw.pop("open", None)
                self._event(ctx, "closed", pid=o.get("pid"), kind=o.get("kind"), why="no line number")
                return None
            fw.pop("open", None)
            self._answer(ctx, o, "fix", "fix", teacher_text=line_text, line=shown[n])
            return [IO.text(LINES["thanks_line"].format(n=n)), self._nav(eng, ctx)]
        if o.get("await") == "write":
            fw.pop("open", None)
            if o["kind"] == "unit_line":
                toi, en = (t.split("=", 1) + [""])[:2] if "=" in t else (t, "")
                self._answer(ctx, o, "write", "write", teacher_text=t,
                             line=dict(o.get("target") or {}, teacher_toi=toi.strip(), teacher_en=en.strip() or None))
            else:
                self._answer(ctx, o, "write", "write", teacher_text=t)
            return [IO.text(LINES["thanks_write"]), self._nav(eng, ctx)]
        fw.pop("open", None)
        return None


def _line(ln):
    return {k: ln.get(k) for k in ("n", "en", "toi", "origin", "src", "en_is_gloss") if ln.get(k) is not None}


def _status_rules(P):
    from . import gates as G
    return G.status_rules(P.rules)
