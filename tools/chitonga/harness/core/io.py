"""Replies as data, WhatsApp-shaped, transport-agnostic (23 Oct 2026, the Jesuit).

The WhatsApp Cloud API limits (checked against the interactive message types of an earlier business line): reply
buttons at most 3, button title at most 20 characters; a list message at most 10 rows in all, row title at most 24
characters, row description at most 72. The engine only builds these dicts; a transport adapter (later) maps them to
the API. Longer lists page: 9 rows + a 'More' row.

    {"type": "text", "body": str}
    {"type": "buttons", "body": str, "buttons": [{"id", "title"}]}
    {"type": "list", "body": str, "button": str, "rows": [{"id", "title", "description"}]}
An inbound message is {"text": str} or {"choice": id} (a tapped button or list row), plus optional "title".
"""
MAX_BUTTONS = 3
MAX_BUTTON_TITLE = 20
MAX_ROWS = 10
MAX_ROW_TITLE = 24
MAX_ROW_DESC = 72
MAX_BODY = 4096


def _cut(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def text(body):
    return {"type": "text", "body": _cut(body, MAX_BODY)}


def buttons(body, items):
    """items: [(id, title)] -> at most 3 (more raises: the caller must use a list)"""
    if len(items) > MAX_BUTTONS:
        raise ValueError(f"{len(items)} buttons > {MAX_BUTTONS}: use a list")
    return {"type": "buttons", "body": _cut(body, 1024),
            "buttons": [{"id": i, "title": _cut(t, MAX_BUTTON_TITLE)} for i, t in items]}


def listing(body, rows, button="Choose", page=0, more_id=None):
    """rows: [(id, title, description)]; more than 10 rows page 9 at a time with a 'More' row (id more_id + ':' +
    next page). Returns the list dict."""
    rows = list(rows)
    if len(rows) > MAX_ROWS:
        start = page * (MAX_ROWS - 1)
        chunk = rows[start: start + MAX_ROWS - 1]
        if start + MAX_ROWS - 1 < len(rows):
            chunk.append((f"{more_id}:{page + 1}", "More ...", f"{len(rows) - start - len(chunk)} more"))
        rows = chunk
    return {"type": "list", "body": _cut(body, 1024), "button": _cut(button, MAX_BUTTON_TITLE),
            "rows": [{"id": i, "title": _cut(t, MAX_ROW_TITLE), "description": _cut(d or "", MAX_ROW_DESC)}
                     for i, t, d in rows]}


def choices(body, items, more_id="more"):
    """buttons when 3 or fewer, else a (paged) list. items: [(id, title)] or [(id, title, description)]"""
    if len(items) <= MAX_BUTTONS and all(len(x[1]) <= MAX_BUTTON_TITLE for x in items):
        return buttons(body, [(x[0], x[1]) for x in items])
    return listing(body, [(x[0], x[1], x[2] if len(x) > 2 else "") for x in items], more_id=more_id)


def validate(reply):
    """[problems] of one reply dict against the limits"""
    p = []
    t = reply.get("type")
    if t == "buttons":
        if not 1 <= len(reply["buttons"]) <= MAX_BUTTONS:
            p.append("buttons count")
        p += [f"button title too long: {b['title']}" for b in reply["buttons"] if len(b["title"]) > MAX_BUTTON_TITLE]
    elif t == "list":
        if not 1 <= len(reply["rows"]) <= MAX_ROWS:
            p.append("rows count")
        p += [f"row title too long: {r['title']}" for r in reply["rows"] if len(r["title"]) > MAX_ROW_TITLE]
    elif t != "text":
        p.append(f"unknown type {t}")
    if len(reply.get("body") or "") > MAX_BODY:
        p.append("body too long")
    return p


def as_text(reply):
    """a plain rendering for the CLI and the logs"""
    out = [reply.get("body") or ""]
    if reply["type"] == "buttons":
        out += [f"  [{i + 1}] {b['title']}" for i, b in enumerate(reply["buttons"])]
    elif reply["type"] == "list":
        out += [f"  ({i + 1}) {r['title']}" + (f" - {r['description']}" if r["description"] else "")
                for i, r in enumerate(reply["rows"])]
    return "\n".join(out)
