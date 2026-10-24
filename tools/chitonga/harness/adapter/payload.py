"""Engine reply dicts -> WhatsApp Cloud API payloads (24 Oct 2026, the Jesuit; WHATSAPP_ADAPTER.md section 2, gap 2).

Two menu modes:
  interactive  reply buttons (max 3, title max 20) and list messages (max 10 rows, row title max 24, description
               max 72, list button max 20, body max 1024): the engine's io.py already builds within these limits and
               io.validate() checks them again here; a reply that fails is sent as numbered plain text, never dropped.
  numbered     the same options as plain text, "1. Continue / 2. Not this one ... Reply with the number." Used until
               the webhook stores a tap's id (today a tap arrives as '[interactive]' with no id or title), and for
               any teacher whose phone shows buttons badly (the kill sign of section 10).
Guards from the Zamai line (zamai_send.py, paid for): no em or en dashes leave the line; and no control bytes (a
backspace once came from a backslash-b in a patch script): a control byte is a bug, so the reply is flagged and the
adapter pauses that teacher.
Every payload comes with its log text (io.as_text) and its options [(id, title)] for the title / number mapping.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import io as IO  # noqa: E402

_D = "[—–]"
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_INTERACTIVE_BODY = 1024
NUMBER_HINT = "Reply with the number."


def strip_dashes(t):
    """as zamai_send.strip_dashes: a leading dash becomes '- ', between digits '-', elsewhere ', '"""
    t = re.sub(r"(?m)^[ \t]*" + _D + r"[ \t]*", "- ", t or "")
    t = re.sub(r"(?<=\d)[ \t]*" + _D + r"[ \t]*(?=\d)", "-", t)
    return re.sub(r"[ \t]*" + _D + r"[ \t]*", ", ", t)


def clean(s):
    return strip_dashes(s or "")


def options(reply):
    if reply.get("type") == "buttons":
        return [(b["id"], b["title"]) for b in reply["buttons"]]
    if reply.get("type") == "list":
        return [(r["id"], r["title"]) for r in reply["rows"]]
    return []


def numbered_text(reply, offset=0):
    """the plain-text form of a menu: the body, the options numbered, the hint. offset (24 Oct 2026, flywheel): when
    one turn sends two menus (a lesson's buttons, then the one feedback question; or the offer, then 'Did it help?'),
    the second menu's numbers CONTINUE the first's (4, 5, 6), so a typed number names exactly one option of the turn
    (before, both menus said '1.' and a typed 1 meant the last menu's first option only)"""
    lines = [clean(reply.get("body") or "")]
    if reply.get("type") == "buttons":
        lines += [f"{offset + i + 1}. {clean(b['title'])}" for i, b in enumerate(reply["buttons"])]
    elif reply.get("type") == "list":
        lines += [f"{offset + i + 1}. {clean(r['title'])}" + (f" ({clean(r['description'])})" if r.get("description")
                                                              else "")
                  for i, r in enumerate(reply["rows"])]
    if reply.get("type") in ("buttons", "list"):
        lines += ["", NUMBER_HINT]
    return "\n".join(lines).strip()


def _text_payload(to, body):
    return {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to, "type": "text",
            "text": {"preview_url": False, "body": body[: IO.MAX_BODY]}}


def build(to, reply, menus="numbered", offset=0):
    """(payload, log_text, opts, flags). flags: 'control byte' (a bug: pause), 'fallback: ...' (sent as text)."""
    flags = []
    raw = " ".join([reply.get("body") or ""] + [t for _, t in options(reply)] +
                   [r.get("description") or "" for r in reply.get("rows") or []])
    if CONTROL.search(raw):
        flags.append("control byte")
    opts = [(i, clean(t)) for i, t in options(reply)]
    t = reply.get("type")
    if t == "text" or not opts:
        body = clean(reply.get("body") or "")
        return _text_payload(to, body), body, [], flags
    problems = IO.validate(reply)
    if len(reply.get("body") or "") > MAX_INTERACTIVE_BODY:
        problems.append("body over 1024 for an interactive message")
    if menus != "interactive" or problems:
        if problems:
            flags.append("fallback: " + "; ".join(problems))
        body = numbered_text(reply, offset)
        return _text_payload(to, body), body, opts, flags
    body = clean(reply["body"])
    if t == "buttons":
        inter = {"type": "button", "body": {"text": body},
                 "action": {"buttons": [{"type": "reply", "reply": {"id": i, "title": tt}} for i, tt in opts]}}
    else:
        inter = {"type": "list", "body": {"text": body},
                 "action": {"button": clean(reply.get("button") or "Choose")[: IO.MAX_BUTTON_TITLE],
                            "sections": [{"title": "Choose",
                                          "rows": [{"id": r["id"], "title": clean(r["title"]),
                                                    **({"description": clean(r["description"])}
                                                       if r.get("description") else {})}
                                                   for r in reply["rows"]]}]}}
    payload = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to, "type": "interactive",
               "interactive": inter}
    return payload, IO.as_text(dict(reply, body=body)), opts, flags
