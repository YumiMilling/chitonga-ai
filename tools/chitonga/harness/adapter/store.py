"""The adapter's own small state, apart from the teacher memory (24 Oct 2026, the Jesuit).

Kept here, not in the engine's memory, because it is about the TRANSPORT, not the teacher:
  seen        wa_message_id of every inbound handled (a restart never answers twice)
  mark        the high-water mark (created_at of the newest inbound read), so a restart neither replays nor skips
  modes       per number: 'desk' (default) | 'assistant' | 'paused' (an automatic stop for that teacher), last_in_at
  menus       the options of the last outbound messages per number (id, title), to map a tap's TITLE or a typed
              NUMBER back to the engine's id while the webhook stores no id (WHATSAPP_ADAPTER.md gap 1)
  sends       one row per outbound attempt (status, when) for the rate limit and the failure stop
  inbound     one row per inbound turn passed to the engine (for the inbound rate limit)
  requests    per number, the running count of real requests (the first N are held out: test-then-train)
  open_label  per number, the last label that waits for its correction (after a No or a "Not this one")
No message text is stored here except the menu titles we sent ourselves.
"""
import contextlib
import json
import os
import sqlite3
import threading

SCHEMA = """
create table if not exists seen (wa_id text primary key, at text);
create table if not exists kv (k text primary key, v text);
create table if not exists modes (number text primary key, mode text, since text, last_in_at text);
create table if not exists menus (id integer primary key autoincrement, number text, at text, wa_id text, options text);
create table if not exists sends (id integer primary key autoincrement, number text, at text, status text);
create table if not exists inbound (id integer primary key autoincrement, number text, at text);
create table if not exists requests (number text primary key, n integer);
create table if not exists open_label (number text primary key, rec text, at text);
"""


class AdapterStore:
    def __init__(self, path):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def _tx(self):
        with self._lock:
            yield self.db
            self.db.commit()

    # seen / mark
    def seen(self, wa_id):
        return bool(wa_id) and self.db.execute("select 1 from seen where wa_id=?", (wa_id,)).fetchone() is not None

    def mark_seen(self, wa_id, at):
        if wa_id:
            with self._tx() as d:
                d.execute("insert or ignore into seen values (?,?)", (wa_id, at))

    def get(self, k, default=None):
        r = self.db.execute("select v from kv where k=?", (k,)).fetchone()
        return r["v"] if r else default

    def put(self, k, v):
        with self._tx() as d:
            d.execute("insert into kv values (?,?) on conflict(k) do update set v=excluded.v", (k, v))

    # modes
    def mode(self, n):
        r = self.db.execute("select * from modes where number=?", (n,)).fetchone()
        return dict(r) if r else {"number": n, "mode": "desk", "since": None, "last_in_at": None}

    def set_mode(self, n, mode, at):
        with self._tx() as d:
            d.execute("insert into modes values (?,?,?,?) on conflict(number) do update set mode=excluded.mode, "
                      "since=excluded.since", (n, mode, at, None))

    def touch(self, n, at):
        with self._tx() as d:
            d.execute("insert into modes values (?, 'desk', ?, ?) on conflict(number) do update set "
                      "last_in_at=max(coalesce(last_in_at, ''), excluded.last_in_at)", (n, at, at))

    # menus
    def add_menu(self, n, at, wa_id, options):
        with self._tx() as d:
            d.execute("insert into menus (number, at, wa_id, options) values (?,?,?,?)",
                      (n, at, wa_id, json.dumps(options, ensure_ascii=False)))
            d.execute("delete from menus where number=? and id not in (select id from menus where number=? "
                      "order by id desc limit 20)", (n, n))

    def menus(self, n, limit=20):
        rows = self.db.execute("select * from menus where number=? order by id desc limit ?", (n, limit)).fetchall()
        return [dict(r, options=json.loads(r["options"])) for r in rows]

    def last_menu(self, n):
        m = self.menus(n, 1)
        return m[0] if m else None

    # sends and inbound counters
    def log_send(self, n, at, status):
        with self._tx() as d:
            d.execute("insert into sends (number, at, status) values (?,?,?)", (n, at, status))

    def sends_since(self, n, since, status=None):
        q = "select count(*) c from sends where number=? and at>=?" + (" and status=?" if status else "")
        return self.db.execute(q, (n, since) + ((status,) if status else ())).fetchone()["c"]

    def log_inbound(self, n, at):
        with self._tx() as d:
            d.execute("insert into inbound (number, at) values (?,?)", (n, at))

    def inbound_since(self, n, since):
        return self.db.execute("select count(*) c from inbound where number=? and at>=?", (n, since)).fetchone()["c"]

    # requests (held-out counter)
    def next_request(self, n):
        with self._tx() as d:
            d.execute("insert into requests values (?, 1) on conflict(number) do update set n=n+1", (n,))
            return d.execute("select n from requests where number=?", (n,)).fetchone()["n"]

    # the label waiting for its correction
    def open_label(self, n):
        r = self.db.execute("select rec from open_label where number=?", (n,)).fetchone()
        return json.loads(r["rec"]) if r else None

    def set_open_label(self, n, rec, at):
        with self._tx() as d:
            if rec is None:
                d.execute("delete from open_label where number=?", (n,))
            else:
                d.execute("insert into open_label values (?,?,?) on conflict(number) do update set rec=excluded.rec, "
                          "at=excluded.at", (n, json.dumps(rec, ensure_ascii=False), at))
