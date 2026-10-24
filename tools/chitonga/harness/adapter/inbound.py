"""A zamai_messages row -> the engine's inbound message (24 Oct 2026, the Jesuit; WHATSAPP_ADAPTER.md gap 1).

The row as the webhook writes it today: body = text, or '[interactive]' for a tapped button or list row (no id, no
title: both are lost), msg_type = 'text' | 'interactive' | 'button' | media types. The proposal in
../proposals/ makes the webhook store body = the tap's title and choice_id = its id. This parser reads all three
states, so the adapter works before and after the fix:
  1 choice_id present (webhook fixed)                         -> {"choice": id, "title": body}
  2 an interactive row whose body is a TITLE (half fix)       -> the id of that title in the newest menu we sent this
                                                                 number that has it (titles repeat across menus:
                                                                 'Continue' is in many; the newest wins)
  3 a typed number "2" when the last message we sent was a menu -> the id of option 2 (numbered menus, basic phones)
  4 '[interactive]' with nothing else (today's webhook)      -> {"unreadable_tap": True}: the adapter resends the last
                                                                 menu as numbered text, never guesses
  5 text                                                      -> {"text": body}
  6 media (image, audio, sticker ...)                         -> {"media": msg_type}: the engine cannot read it yet
  7 a reaction, an edit, a system notice                      -> {"silent": msg_type}: logged, never answered
"""
import re

# not a message to answer: an emoji reaction, an edit of an earlier message (seen on the desk: 5 reactions, 13 edits in
# 238 inbound rows of 10-24 Oct), Meta's system notices. Logged and left alone, never answered.
SILENT = {"reaction", "edit", "system", "unsupported", "unknown", "revoke", "deleted", "ephemeral"}
NUM = re.compile(r"^\s*(\d{1,2})\s*[.)]?\s*$")


def _norm(t):
    return re.sub(r"\s+", " ", (t or "").replace("…", "...")).strip().lower()


def parse(row, store, number):
    """(msg, how): msg for Engine.handle, or a dict with 'unreadable_tap' / 'media'; how names the rule used"""
    body = row.get("body") or ""
    mtype = (row.get("msg_type") or "text").lower()
    cid = row.get("choice_id")
    if cid:
        return {"choice": cid, "title": body}, "choice_id"
    if mtype in ("interactive", "button"):
        if body and body != "[interactive]" and body != "[button]":
            want = _norm(body)
            for m in store.menus(number):
                for oid, title in m["options"]:
                    if _norm(title) == want:
                        return {"choice": oid, "title": body}, "title_map"
            return {"text": body}, "title_unmapped"
        return {"unreadable_tap": True}, "bare_interactive"
    if mtype in SILENT:
        return {"silent": mtype}, "silent"
    if mtype not in ("text", ""):
        return {"media": mtype}, "media"
    m = NUM.match(body)
    last = store.last_menu(number)
    if m and last and last["options"]:
        k = int(m.group(1))
        if 1 <= k <= len(last["options"]):
            oid, title = last["options"][k - 1]
            return {"choice": oid, "title": title}, "number_map"
    return {"text": body}, "text"
