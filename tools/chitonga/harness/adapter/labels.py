"""Label capture for training: every teacher answer that says whether the harness read the request right (24 Oct 2026,
the Jesuit; coordinator's task item 4; the reader contract of chat round v0.14: the model's output IS the harness
reading, its candidates ARE the retrieval candidates, so a teacher's Yes / No / correction is a labelled reading).

ONE RECORD per label, in pilot_logs/pilot_labels.jsonl:
  {id, at, teacher (code, never the number), request_id, held_out, request (text, or null when withheld),
   request_lang, reading {intent, item, unit, confidence, source, why}, candidates, alternatives, proposed,
   chosen, kind, label, strength, plugin_version, reader}
KINDS (what the teacher did -> what it means):
  confirm_yes        tapped Yes on "You want X?"                      proposed is right        strong
  confirm_no         tapped No                                        proposed is wrong        strong
  not_this           tapped "Not this one" on a lesson read from text proposed is wrong        strong
  correction_pick    after a No / Not this one, picked an alternative chosen = the right one   strong
                     or a menu unit / item
  correction_typed   after a No / Not this one (or instead of answering the confirmation), typed the request again:
                     the new text and its reading; chosen = what that reading delivered, if anything   weak
  accepted           a lesson delivered straight from free text, and the teacher tapped Continue   weak
                     (implicit: they read on; not proof the reading was the best one)
  strategy_feedback  "Did it help?" helped / partly / no            the suggestion's outcome  strong
HELD OUT (test-then-train): a request is every free-text turn the reader read (path 'read:'); the first 40 per teacher
carry held_out=true, and so does every label on them. A training build must drop held_out rows (leak assert) and score
the reader on them.
NEVER STORED: a safeguarding message (the engine withholds it: no label at all) and pupil-level text (the engine's
pupil_level guard: the record keeps the label, the text becomes null).
"""
import hashlib
import json
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory import pupil_level  # noqa: E402

READING_KEYS = ("intent", "item", "unit", "subject", "confidence", "source", "why")
CORRECTION_TEXT = re.compile(r"(?i)^\s*(no\b|not\b|wrong\b|i\s+meant\b|i\s+mean\b|rather\b|actually\b)")


NAVIGATION = ("subj", "grade", "more", "plug", "class")


def _delivered(path):
    """the item a turn delivered (part 1), from its path 'deliver:<kind>:<id>:1/4', or None"""
    for p in path:
        if p.startswith("deliver:"):
            kind, rest = p.split(":", 2)[1:]
            return kind + ":" + rest.rsplit(":", 1)[0]
    return None


def text_key(t):
    """a short key to match a request text without keeping it (pupil-level text is never stored)"""
    return hashlib.sha1((t or "").strip()[:200].encode("utf-8")).hexdigest()[:16] if t else None


def _safe_text(t):
    if not t:
        return None
    return None if pupil_level(t) else t[:300]


class LabelWriter:
    def __init__(self, path, store, teacher_code):
        self.path = path
        self.store = store
        self.code = teacher_code         # number -> 'T01'; the label file never holds a phone number

    def _write(self, rec):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def _base(self, n, at, req, turn):
        return {"id": uuid.uuid4().hex[:12], "at": at, "teacher": self.code(n),
                "request_id": req.get("request_id"), "held_out": req.get("held_out"),
                "request": req.get("text"), "request_lang": req.get("lang"), "reading": req.get("reading"),
                "candidates": req.get("candidates", []), "alternatives": req.get("alternatives", []),
                "proposed": req.get("proposed"), "plugin_version": turn.get("plugin_version"),
                "reader": turn.get("reader")}

    def request_from(self, turn, request_id, held_out):
        """the request this turn made (a free-text turn read by the reader), or None"""
        r = turn.get("reading")
        inbound = turn.get("inbound") or {}
        if not r or inbound.get("withheld") or not any(str(p).startswith("read:") for p in turn.get("path", [])):
            return None
        return {"request_id": request_id, "held_out": held_out, "text": _safe_text(inbound.get("text")),
                "text_key": text_key(inbound.get("text")),
                "lang": (r.get("slots") or {}).get("language"),
                "reading": {k: r.get(k) for k in READING_KEYS},
                "candidates": list(r.get("candidates") or [])[:8], "alternatives": list(r.get("alternatives") or [])[:3],
                "proposed": r.get("item") or (("unit:" + r["unit"]) if r.get("unit") else None)}

    def capture(self, n, at, msg, turn, before_state, request, last_req=None):
        """the labels of one turn -> [records]. before_state: the engine state BEFORE the turn (pending, delivered_from);
        request: this turn's request (request_from) or None; last_req: the adapter's recent requests (a list, newest
        last) before this turn."""
        out = []
        path = [str(p) for p in turn.get("path", [])]
        if (turn.get("inbound") or {}).get("withheld"):
            self.store.set_open_label(n, None, at)
            return out
        openl = self.store.open_label(n)
        # a confirmation waits only while the step is 'confirm' (the engine leaves an answered pending in the state)
        pend = ((before_state or {}).get("pending") or {}) if (before_state or {}).get("step") == "confirm" else {}
        if ("confirm:yes" in path or "confirm:no" in path) and not pend:
            pend = (before_state or {}).get("pending") or {}
        df = (before_state or {}).get("delivered_from") or {}
        choice = msg.get("choice") or ""

        def req_of(src):
            """the original request a pending / delivered_from belongs to: the adapter's last request when the text
            matches (text_key), else rebuilt from the engine state without a request id"""
            hist = last_req if isinstance(last_req, list) else ([last_req] if last_req else [])
            k = text_key(src.get("text"))
            for q in reversed(hist):
                if q and (q.get("text_key") == k or not src.get("text")):
                    return dict(q, proposed=src.get("item") or q.get("proposed"))
            rd = src.get("reading") or {}
            return {"request_id": None, "held_out": None, "text": _safe_text(src.get("text")), "lang": None,
                    "reading": {k: rd.get(k) for k in READING_KEYS} if rd else None,
                    "candidates": list(src.get("candidates") or []), "alternatives": list(src.get("alternatives") or []),
                    "proposed": src.get("item")}

        def typed(src, delivered, req):
            """a correction_typed record: src = the label or request being corrected, req = the new request"""
            same = delivered is not None and delivered == src.get("proposed")
            return dict({k: v for k, v in src.items() if not k.startswith("_")}, id=uuid.uuid4().hex[:12], at=at,
                        kind="correction_typed", label="confirmed_by_retype" if same else "retyped",
                        strength="medium" if CORRECTION_TEXT.match(req.get("text") or "") else "weak",
                        correction_text=req.get("text"), correction_request_id=req.get("request_id"),
                        correction_reading=req.get("reading"), chosen=delivered)

        fresh = bool(openl and openl.get("_fresh"))
        if openl and fresh:                       # an open label is 'fresh' for exactly one turn (typed corrections)
            self.store.set_open_label(n, dict(openl, _fresh=False), at)
            openl = dict(openl, _fresh=False)
        if "confirm:yes" in path and pend.get("item"):
            q = req_of(pend)
            out.append(dict(self._base(n, at, q, turn), kind="confirm_yes", label="correct", strength="strong",
                            chosen=pend["item"]))
            if openl and openl.get("_retype") and openl["_retype"].get("request_id") == q.get("request_id"):
                out.append(typed(openl, pend["item"], openl["_retype"]))   # the retyped request, confirmed: the link
            self.store.set_open_label(n, None, at)
        elif "confirm:no" in path and pend:
            rec = dict(self._base(n, at, req_of(pend), turn), kind="confirm_no", label="wrong", strength="strong",
                       chosen=None)
            out.append(rec)
            self.store.set_open_label(n, dict(rec, _fresh=True), at)
        elif turn.get("label", {}).get("kind") == "reader_error":
            rec = dict(self._base(n, at, req_of(df or {"text": turn["label"].get("text"),
                                                        "reading": turn["label"].get("reading")}), turn),
                       kind="not_this", label="wrong", strength="strong", chosen=None,
                       proposed=turn["label"].get("wrong_item"))
            out.append(rec)
            self.store.set_open_label(n, dict(rec, _fresh=True), at)
        elif choice.startswith("fb:"):
            _, sid, outcome = choice.split(":")
            out.append({"id": uuid.uuid4().hex[:12], "at": at, "teacher": self.code(n), "kind": "strategy_feedback",
                        "suggestion_id": int(sid), "label": outcome, "strength": "strong", "held_out": None,
                        "plugin_version": turn.get("plugin_version")})
        elif openl and (choice.startswith("item:") or choice.startswith("unit:")):
            # the correction picked from the alternatives or through the menu (subject > unit > item): a unit tap keeps
            # the label open until the lesson is chosen, unless the unit delivered at once
            delivered = _delivered(path)
            if choice.startswith("unit:") and not delivered:
                self.store.set_open_label(n, dict(openl, picked_unit=choice.split(":", 1)[1]), at)
            else:
                chosen = delivered or (choice.split(":", 1)[1] if choice.startswith("item:") else choice)
                # picking the very lesson just rejected is a contradiction (a mis-tap, or the lesson was right after
                # all): kept, labelled for a person to look at, never used as a correction
                out.append(dict({k: v for k, v in openl.items() if not k.startswith("_")}, id=uuid.uuid4().hex[:12],
                                at=at, kind="correction_pick",
                                label="same_as_rejected" if chosen == openl.get("proposed") else "corrected",
                                strength="strong", chosen=chosen))
                self.store.set_open_label(n, None, at)
        elif request and (fresh or pend or (df and df.get("item") == (before_state or {}).get("item")
                                            and (before_state or {}).get("part") == 0
                                            and CORRECTION_TEXT.match(request.get("text") or ""))):
            # typed instead of tapping, in the very next turn: after a No / Not this one, instead of answering a
            # waiting confirmation, or ("no, I meant ...") right after a lesson read from text. A teacher who went
            # through the menu first is browsing, not correcting (not counted). When the new text lands on the lesson
            # the reader had proposed, it CONFIRMS the reading (confirmed_by_retype). When it leads to a confirmation
            # instead of a lesson, the link waits for that confirmation's Yes.
            src = openl if fresh else dict(self._base(n, at, req_of(pend) if pend else req_of(df), turn))
            delivered = _delivered(path)
            if delivered:
                out.append(typed(src, delivered, request))
                self.store.set_open_label(n, None, at)
            elif any(p.startswith("confirm:") for p in path):
                self.store.set_open_label(n, dict(src, _retype=request), at)
            else:
                self.store.set_open_label(n, None, at)
        elif choice == "c:continue" and df and (before_state or {}).get("part") == 0                 and df.get("item") == (before_state or {}).get("item"):
            out.append(dict(self._base(n, at, req_of(df), turn), kind="accepted", label="correct", strength="weak",
                            chosen=df["item"]))
        elif openl and not (choice.split(":")[0] in NAVIGATION or choice in ("c:menu", "c:lesson_help")):
            self.store.set_open_label(n, None, at)       # the teacher moved on: the open label stays as written
        return [self._write(r) for r in out]
