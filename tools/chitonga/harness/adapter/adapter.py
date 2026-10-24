"""The pilot adapter: one zamai_messages row in, the engine's replies out, every turn logged and labelled (24 Oct 2026,
the Jesuit; design WHATSAPP_ADAPTER.md; Chris: "can we start using the actual harness for the training with our
teachers?"; and, same day: "it's safe to assume that all teachers in Southern Province speak English enough that the
harness can be all English; it can offer Tonga materials and explanations": the engine runs with tonga_ui=False).

ORDER OF THE CHECKS FOR ONE ROW (each one can end the turn; the engine is called only when all pass):
  1 seen before                      -> nothing (a restart never answers twice)
  2 number not on the allow-list     -> IGNORED: no engine call, no reply, logged without the body
  3 the kill switch                  -> desk: no engine call, nothing sent (ntfy once)
  4 the dispatcher (section 4)       -> the start word puts the number in assistant mode; Stop / Desk takes it out;
                                        a desk message (*cd) sent to that teacher since the start takes it out;
                                        2 h of silence takes it out; a desk-numbered answer while a desk batch to
                                        that teacher is open goes to the desk; desk mode -> the desk, untouched
  5 the 24 h window (section 5)      -> a backlog message older than 24 h is HELD unread (the engine does not consume
                                        it); no template unless Chris approved one (config.template_ok, False)
  6 the inbound rate limit           -> held
  7 parse (inbound.py)               -> an unreadable tap: the last menu again as numbered text, no engine call
  8 Engine.handle()                  -> replies + the turn record; the request counted (first 40 held out); labels
  9 send each reply in order         -> before EACH send: the stop, the per-turn cap (6), the hourly cap (60); a
                                        control byte pauses the teacher; 3 failures in 10 min pause the teacher
 10 one log line                     -> pilot_logs/YYYY-MM-DD.jsonl (adapter fields + the engine's turn record)
Safeguarding is the engine's (gates.py; referral, the EVENT only): the adapter adds nothing and stores no text of a
withheld message (the engine marks it withheld; the log and the labels keep only that it happened).
"""
import copy
import datetime
import json
import os
import re
import time

import sys

from . import inbound as IN
from . import payload as PL
from .config import norm_number
from .flywheel_log import FlywheelWriter
from .labels import LabelWriter, pupil_level

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gates as G  # noqa: E402

ZAMBIA = datetime.timezone(datetime.timedelta(hours=2))
START = re.compile(r"(?i)^\s*(assistant|start\s+assistant|start\s+the\s+assistant|teacher\s+assistant)\s*[.!]?\s*$")
STOPW = re.compile(r"(?i)^\s*(stop|desk|stop\s+assistant|stop\s+the\s+assistant)\s*[.!]?\s*$")
DESK_NUMBERED = re.compile(r"^\s*\(?\d{1,2}\s*[.):]\s*\S")
LINES = {   # the adapter's own fixed lines (English: the interface is English, 24 Oct)
    "stopped": "The assistant is off now. Send 'Assistant' any time to start it again.",
    "unreadable_tap": "I could not read that tap on my side. Please reply with the number instead.",
    "media": "I can only read typed messages for now. Please type your request.",
    "pupil": ("Please do not send pupils' names or marks here. I have not kept your message. You can ask again "
              "without the name, for example: how do I mark a test out of 10?"),
}


def parse_ts(s):
    if isinstance(s, datetime.datetime):
        return s if s.tzinfo else s.replace(tzinfo=datetime.timezone.utc)
    d = datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)


def iso(d):
    return d.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


class CaptureLog:
    """the engine's TurnLog stand-in: keeps the record of the turn for the adapter's combined log line"""

    def __init__(self):
        self.last = None

    def write(self, rec):
        self.last = rec


def engine_clock(clock):
    """the engine works in Zambian local time, naive (as its SESSION_GAP and the memory timestamps)"""
    return lambda: clock().astimezone(ZAMBIA).replace(tzinfo=None)


def desk_open_from_files(desk_dir, days=3):
    """a callable(number, now) -> True when a desk batch to that teacher is open (issued_batches.jsonl, teachers.json):
    dated within `days` and its status not answered / closed / paid / done"""
    def is_open(n, now):
        try:
            teachers = json.load(open(os.path.join(desk_dir, "teachers.json"), encoding="utf-8"))
            rows = [json.loads(x) for x in open(os.path.join(desk_dir, "issued_batches.jsonl"), encoding="utf-8")
                    if x.strip()]
        except (OSError, ValueError):
            return False
        codes = {c for c, t in teachers.items() if norm_number(t.get("whatsapp")) == n}
        for r in rows:
            if r.get("to") not in codes:
                continue
            try:
                d = datetime.date.fromisoformat(r.get("date", ""))
            except ValueError:
                continue
            if (now.date() - d).days <= days and not re.search(r"(?i)answered|closed|paid|done", r.get("status", "")):
                return True
        return False
    return is_open


class PilotAdapter:
    def __init__(self, engine, config, store, sender, clock, sleep=time.sleep, notify=None, live=False,
                 desk_open=None, desk_sent=None, mode=None):
        """desk_open(n, now): a desk batch to that teacher is open (issued_batches.jsonl; sparse, a weak signal).
        desk_sent(n, since_iso, until_iso): a desk message ('*cd ...') went to that teacher in that interval (the
        line's own outbound rows: the strong signal, measured on the 10-24 Oct desk traffic, see PILOT_PLAN.md)."""
        if getattr(engine, "tonga_ui", False):
            raise ValueError("pilot mode: the interface is English (Engine(tonga_ui=False)), Chris 24 Oct")
        self.eng, self.cfg, self.store, self.sender = engine, config, store, sender
        self.clock, self.sleep = clock, sleep
        self.notify = notify or (lambda text: None)
        self.live = live
        self.allow = config.allowlist(live=live)
        self.desk_open = desk_open or (lambda n, now: False)
        self.desk_sent = desk_sent or (lambda n, since, until: False)
        self.cap = CaptureLog()
        self.eng.log = self.cap
        self.labels = LabelWriter(config.labels_path, store,
                                  lambda n: (self.allow.get(n) or {}).get("code") or "T?")
        self.flywheel = FlywheelWriter(config, self.allow, mode or ("live" if live else "shadow"))
        self._stop_noted = False

    # ------------------------------------------------------------------ logging
    def _log(self, rec):
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        day = self.clock().astimezone(ZAMBIA).date().isoformat()
        with open(os.path.join(self.cfg.log_dir, f"{day}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def _pause(self, n, why):
        self.store.set_mode(n, "paused", iso(self.clock()))
        self.notify(f"Tonga assistant paused for {(self.allow.get(n) or {}).get('code', '?')}: {why}")

    # ------------------------------------------------------------------ one row
    def process(self, row):
        now = self.clock()
        wa = row.get("wa_message_id") or row.get("id")
        base = {"at": iso(now), "wa_in": wa, "in_at": row.get("created_at"), "msg_type": row.get("msg_type")}
        if self.store.seen(wa):
            return dict(base, route="duplicate")
        self.store.mark_seen(wa, iso(now))
        if row.get("created_at") and row["created_at"] > (self.store.get("mark") or ""):
            self.store.put("mark", row["created_at"])
        n = norm_number(row.get("counterparty"))
        if n not in self.allow:
            return self._log(dict(base, route="ignored", why="not on the pilot allow-list"))   # no body, no number
        code = self.allow[n].get("code")
        base.update(number=n, teacher=code)
        at = parse_ts(row.get("created_at") or now)
        body = (row.get("body") or "").strip()

        if self.cfg.stopped():                                              # 3 the kill switch
            if not self._stop_noted:
                self.notify("Tonga assistant: stop switch on, all messages go to the desk")
                self._stop_noted = True
            return self._log(dict(base, route="desk", why="stop switch"))
        self._stop_noted = False

        # 4 the dispatcher
        m = self.store.mode(n)
        last_in = parse_ts(m["last_in_at"]) if m.get("last_in_at") else None
        self.store.touch(n, iso(at))
        mtype = (row.get("msg_type") or "text").lower()
        if mtype == "text" and START.match(body):
            if m["mode"] == "paused":
                return self._log(dict(base, route="desk", why="paused (automatic stop): a person must clear it"))
            self.store.set_mode(n, "assistant", iso(at))
            return self._turn(n, row, base, at, {"text": ""}, how="start_word")
        if m["mode"] != "assistant":
            return self._log(dict(base, route="desk", why=f"mode {m['mode']}"))
        if m.get("since") and self.desk_sent(n, m["since"], iso(at)):
            # the desk wrote to this teacher after the assistant started: the conversation is the desk's now
            self.store.set_mode(n, "desk", iso(at))
            return self._log(dict(base, route="desk", why="a desk message (*cd) went out since the assistant started"))
        if last_in and (at - last_in) > datetime.timedelta(hours=self.cfg.assistant_idle_h):
            self.store.set_mode(n, "desk", iso(at))
            return self._log(dict(base, route="desk", why=f"assistant idle > {self.cfg.assistant_idle_h} h"))
        if mtype == "text" and STOPW.match(body):
            self.store.set_mode(n, "desk", iso(at))
            sends = self._send(n, at, [{"type": "text", "body": LINES["stopped"]}])
            return self._log(dict(base, route="assistant_stop", sends=sends))
        if mtype == "text" and DESK_NUMBERED.match(body) and (
                self.desk_open(n, now) or self.desk_sent(n, iso(at - datetime.timedelta(hours=72)), iso(at))):
            return self._log(dict(base, route="desk", why="desk-numbered answer while a desk batch is open "
                                                          "(issued_batches, or a *cd message in the last 72 h)"))
        return self._turn(n, row, base, at, None)

    def _turn(self, n, row, base, at, msg, how=None):
        now = self.clock()
        if now - at > datetime.timedelta(hours=self.cfg.window_h):         # 5 window, BEFORE the engine
            return self._log(dict(base, route="held", why="window: inbound older than 24 h; not read by the engine",
                                  template="none (not approved)"))
        hour_ago = iso(now - datetime.timedelta(hours=1))
        if self.store.inbound_since(n, hour_ago) >= self.cfg.max_in_per_hour:   # 6
            return self._log(dict(base, route="held", why="inbound rate limit"))
        if msg is None:                                                     # 7 parse
            msg, how = IN.parse(row, self.store, n)
        if msg.get("unreadable_tap"):
            last = self.store.last_menu(n)
            replies = [{"type": "text", "body": LINES["unreadable_tap"]}]
            if last:
                replies[0]["body"] += "\n" + "\n".join(f"{i + 1}. {t}" for i, (_, t) in enumerate(last["options"]))
            sends = self._send(n, at, replies, keep_menu=False)
            return self._log(dict(base, route="adapter", how=how, sends=sends))
        if msg.get("silent"):
            return self._log(dict(base, route="silent", how=how))
        if msg.get("media"):
            sends = self._send(n, at, [{"type": "text", "body": LINES["media"]}])
            return self._log(dict(base, route="adapter", how=how, sends=sends))
        if msg.get("text") and not G.safeguarding(msg["text"]) and pupil_level(msg["text"]):
            # SAFEGUARDING FIRST: a disclosure is always passed to the engine, whose gate gives the referral and stores
            # only the event (a disclosure names one child too, so this check must never run before that gate).
            # Otherwise nothing pupil-related is stored (pilot rule): the engine's own guard covers only class observations, so
            # a message naming or scoring one learner never reaches the engine, its memory or its log. The row the
            # webhook already wrote in zamai_messages stays until Chris decides the redaction (PILOT_PLAN.md).
            sends = self._send(n, at, [{"type": "text", "body": LINES["pupil"]}])
            return self._log(dict(base, route="adapter", how="pupil_withheld",
                                  inbound={"withheld": "pupil-level data (" + pupil_level(msg["text"]) + "); not kept"},
                                  sends=sends))
        self.store.log_inbound(n, iso(now))
        before = copy.deepcopy(self.eng.mem.get_state(n) or {})
        msg = {k: v for k, v in msg.items() if k in ("text", "choice", "title")}
        replies = self.eng.handle(n, msg)                                   # 8 the engine
        turn = self.cap.last or {}
        request = None
        if any(str(p).startswith("read:") for p in turn.get("path", [])) and not (turn.get("inbound") or {}).get("withheld"):
            k = self.store.next_request(n)
            request = self.labels.request_from(turn, f"{base['teacher']}-{k:04d}", k <= self.cfg.held_out_first)
        recent = json.loads(self.store.get(f"requests:{n}") or "[]")       # the last 10 requests (labels look back)
        labs = self.labels.capture(n, iso(now), msg, turn, before, request, last_req=recent)
        if request:
            self.store.put(f"requests:{n}", json.dumps((recent + [request])[-10:], ensure_ascii=False))
        fws = self.flywheel.capture(n, iso(now), turn, msg, request)        # the flywheel's records (FLYWHEEL.md)
        sends = self._send(n, at, replies)                                  # 9
        withheld = (turn.get("inbound") or {}).get("withheld")
        return self._log(dict(base, route="engine", how=how,
                              inbound=({"withheld": withheld} if withheld else msg),
                              request_id=(request or {}).get("request_id"), held_out=(request or {}).get("held_out"),
                              turn=turn, sends=sends, labels=[x["id"] for x in labs],
                              label_kinds=[x["kind"] for x in labs], flywheel=[x["id"] for x in fws],
                              flywheel_evs=[x["ev"] + ":" + str(x.get("kind") or x.get("what")) for x in fws]))

    # ------------------------------------------------------------------ sending
    def _send(self, n, at, replies, keep_menu=True):
        """send a turn's replies in order. The options of ALL the turn's menus are stored as ONE menu, numbered on from
        one menu to the next (payload.numbered_text offset), so a typed number or a tapped title maps to exactly one
        option of the turn (24 Oct 2026, flywheel: the feedback question is a second menu after the lesson's buttons)"""
        out = []
        now = self.clock()
        combined, last_wa = [], None
        for i, reply in enumerate(replies):
            if self.cfg.stopped():
                out.append({"status": "not sent", "why": "stop switch"})
                break
            if i >= self.cfg.max_out_per_in:
                out.append({"status": "not sent", "why": f"over {self.cfg.max_out_per_in} replies for one message"})
                break
            if self.store.sends_since(n, iso(now - datetime.timedelta(hours=1)), "sent") >= self.cfg.max_out_per_hour:
                self._pause(n, "hourly send cap reached")
                out.append({"status": "not sent", "why": "hourly cap"})
                break
            if now - at > datetime.timedelta(hours=self.cfg.window_h):
                out.append({"status": "not sent", "why": "window"})
                break
            payload, text, opts, flags = PL.build(n, reply, self.cfg.menus, offset=len(combined))
            if "control byte" in flags:
                self._pause(n, "a reply carried a control byte (a bug)")
                out.append({"status": "not sent", "why": "control byte", "text": text})
                break
            if i:
                self.sleep(self.cfg.send_gap_s)
            wa, err = self.sender.send(payload, text, payload["type"])
            status = "failed" if err else "sent"
            self.store.log_send(n, iso(now), status)
            if opts and keep_menu and not err:
                combined += opts
                last_wa = wa
            out.append({"status": "outbox" if str(wa or "").startswith("outbox:") else status, "wa_out": wa,
                        "error": err, "type": payload["type"], "text": text, "flags": flags})
            if err:
                since = iso(now - datetime.timedelta(minutes=self.cfg.fail_window_min))
                if self.store.sends_since(n, since, "failed") >= self.cfg.fail_limit:
                    self._pause(n, f"{self.cfg.fail_limit} send failures in {self.cfg.fail_window_min} min")
                break
        if combined:
            self.store.add_menu(n, iso(now), last_wa, combined)
        return out
