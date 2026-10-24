"""The ONE writer of the flywheel's records and of the pilot pay ledger (24 Oct 2026, the Jesuit; FLYWHEEL.md).

The engine puts its flywheel events in the turn record (core/flywheel.py); this writer stores them, labelled by origin
and consent, one JSON line each, in pilot_logs/flywheel.jsonl:
  {id, at, teacher (code, never the number), speaker (= teacher: the origin of any Tonga in the record), role, tester,
   consent (the allow-list's consent date, or null), mode (shadow | live | sim), request_id, held_out, ev, kind, pid,
   item, part, unit, served / lines / target (what the teacher saw), answer, teacher_text (VERBATIM), pay, control}
  ev: prompt | answer | ignored | closed | setting | gap
GAPS (the content backlog): the engine's path says when it had no material (no_content:unit / no_content:grade /
needs_model + no_material / a route without a hook / <status>_no_card). One 'gap' record each, with the request text
when the turn had one (pupil-level text never reaches the engine; a status (health) gap keeps NO text, only the count).
PAY: every answer with pay > 0 adds one row to the pay ledger in the desk's balances.tsv columns
  date  code  name  what  K  balance_after  note
balance_after is the running total of THIS ledger for that teacher. The ledger is a separate file on purpose (DESIGN:
money is outside the system; the main session moves the rows into desk/balances.tsv when it pays): pay_ledger.tsv in
live mode, pay_ledger_shadow.tsv otherwise. A tester (Chris) earns nothing: K0, with what it would have been.
"""
import csv
import datetime
import json
import os
import re
import uuid

ZAMBIA = datetime.timezone(datetime.timedelta(hours=2))
LEDGER_COLS = ("date", "code", "name", "what", "K", "balance_after", "note")


def gaps_of(turn):
    """[(what, ref, status)] from a turn's path: the honest 'no material' answers. The engine marks them:
    no_content:unit:<id>, no_content:grade<g>, no_material (a request nothing covers, or a route with nothing),
    <status>_no_card[:<item>] (a health / status topic: counted, never with text)"""
    out = []
    for p in (str(x) for x in turn.get("path", [])):
        if p.startswith("no_content:unit:"):
            out.append(("unit", p.split(":", 2)[2], None))
        elif p.startswith("no_content:grade"):
            out.append(("grade", p[len("no_content:grade"):] or None, None))
        elif p == "no_material":
            out.append(("request", None, None))
        elif re.match(r"^\w+_no_card(:|$)", p):
            out.append(("status", p.split(":", 1)[1] if ":" in p else None, p.split("_no_card")[0]))
    return out


class FlywheelWriter:
    def __init__(self, cfg, allow, mode):
        """allow: {number: allow-list entry}; mode: 'shadow' | 'live' | 'sim'"""
        self.cfg, self.allow, self.mode = cfg, allow, mode
        self.path = cfg.flywheel_path
        self.ledger = cfg.pay_ledger_path(mode == "live")

    def _entry(self, n):
        return self.allow.get(n) or {}

    def _write(self, rec):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def _base(self, n, at, request):
        e = self._entry(n)
        return {"id": uuid.uuid4().hex[:12], "at": at, "teacher": e.get("code") or "T?",
                "speaker": e.get("code") or "T?", "role": e.get("role"), "tester": bool(e.get("tester")),
                "consent": (e.get("consent") or {}).get("date"), "mode": self.mode,
                "request_id": (request or {}).get("request_id"), "held_out": (request or {}).get("held_out")}

    def capture(self, n, at, turn, inbound, request):
        """store the turn's flywheel events and gaps -> [records]"""
        out = []
        if (turn.get("inbound") or {}).get("withheld"):
            return out                                  # safeguarding / pupil: nothing of the message is kept
        for ev in turn.get("flywheel") or []:
            rec = dict(self._base(n, at, request), **{k: v for k, v in ev.items() if k != "at"}, engine_at=ev.get("at"))
            out.append(self._write(rec))
            if rec["ev"] == "answer" and rec.get("pay"):
                self._pay(n, rec)
        text = (inbound or {}).get("text") if isinstance(inbound, dict) else None
        for what, ref, status in gaps_of(turn):
            rec = dict(self._base(n, at, request), ev="gap", what=what, ref=ref, status=status,
                       text=None if status else (text or None)[:300] if text else None)
            out.append(self._write(rec))
        return out

    def _balance(self, code):
        total = 0
        try:
            with open(self.ledger, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r.get("code") == code:
                        total += int(r.get("K") or 0)
        except OSError:
            pass
        return total

    def _pay(self, n, rec):
        e = self._entry(n)
        code, k = rec["teacher"], int(rec["pay"])
        note = f"assistant flywheel record {rec['id']}; {self.mode}; not yet in desk/balances.tsv"
        if e.get("tester"):
            note = f"tester, not paid (would be K{k}); record {rec['id']}"
            k = 0
        where = rec.get("item") or (f"unit {rec['unit']}" if rec.get("unit") else "a control line"
                                    if rec.get("control") else "a line")
        line_n = (rec.get("served") or {}).get("n")
        what = f"assistant: {rec['kind']} {rec['answer']}" + (f" line {line_n}" if line_n else "") + f" ({where})"
        today = datetime.datetime.fromisoformat(rec["at"].replace("Z", "+00:00")).astimezone(ZAMBIA).date().isoformat()
        new = not os.path.exists(self.ledger)
        bal = self._balance(code) + k
        os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            if new:
                w.writerow(LEDGER_COLS)
            w.writerow([today, code, e.get("name") or code, what, k, bal, note])
