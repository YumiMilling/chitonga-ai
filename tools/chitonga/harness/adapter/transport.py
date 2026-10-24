"""Where messages come from and go to (24 Oct 2026, the Jesuit).

SOURCES
  ZamaiSource   polls zamai_messages (the line's own message table in a Supabase project) for direction='in' rows from
                the ALLOW-LISTED numbers only (the filter is in the query, so nobody else's message is even read), newer
                than the stored mark. Reads choice_id when the column exists (proposals/), else works without it.
                Read-only: the adapter never edits or deletes a row there.
  a list        simulation and tests: rows in the same shape.
SENDERS
  CloudSender   the WhatsApp Cloud API with the Zamai line's own token and phone id (whatsapp_config line_key='zamai',
                as zamai_send.py), and the log row in zamai_messages (sent and failed alike, as zamai_send.py), body =
                the readable text, msg_type 'interactive' or 'text'. REFUSES unless config.live() at the moment of the
                send (env flag + Chris's go file + no stop).
  MockSender    records the payloads; never touches the network. Used by the simulation, the tests and --dry-run.
The project URL and service key are read from the environment (SUPABASE_URL, ZAMAI_ENV = an env file); no key or
project id is in this repo.
"""
import datetime
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .config import ZAMAIPORT

URL = os.environ.get("SUPABASE_URL", "")
GRAPH = "https://graph.facebook.com/v20.0/{phone_id}/messages"
TIMEOUT = 20


def service_key():
    path = os.environ.get("ZAMAI_ENV") or os.path.join(ZAMAIPORT, ".env")
    for line in open(path, encoding="utf-8"):
        if "SERVICE_ROLE" in line or "SUPABASE_SECRET_KEY" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no Supabase service key in {path}")


class SbError(RuntimeError):
    pass


def sb(path, method="GET", body=None, prefer=None, key=None):
    key = key or service_key()
    headers = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(URL + "/rest/v1/" + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            t = r.read().decode()
        return json.loads(t) if t else None
    except urllib.error.HTTPError as e:
        raise SbError(f"HTTP {e.code} on {method} {path.split('?')[0]}: {e.read().decode(errors='replace')[:300]}")


class ZamaiSource:
    COLS = "id,wa_message_id,created_at,counterparty,msg_type,body"

    def __init__(self, numbers, store):
        self.numbers = sorted(numbers)
        self.store = store
        self.key = service_key()
        self.has_choice = None

    def start_mark(self):
        """first start: from now on (the newest inbound), as zamai_watch.py; later starts: the stored mark"""
        m = self.store.get("mark")
        if m:
            return m
        latest = sb("zamai_messages?direction=eq.in&select=created_at&order=created_at.desc&limit=1", key=self.key)
        m = latest[0]["created_at"] if latest else "2000-01-01T00:00:00+00:00"
        self.store.put("mark", m)
        return m

    def poll(self):
        if not self.numbers:
            return []
        mark = self.start_mark()
        cols = self.COLS + (",choice_id" if self.has_choice is not False else "")
        q = (f"zamai_messages?direction=eq.in&counterparty=in.({','.join(self.numbers)})&select={cols}"
             f"&created_at=gt.{urllib.parse.quote(mark)}&order=created_at.asc&limit=50")
        try:
            rows = sb(q, key=self.key)
            self.has_choice = self.has_choice if self.has_choice is not None else True
        except SbError as e:
            if "choice_id" in str(e) and self.has_choice is not False:
                self.has_choice = False
                return self.poll()
            raise
        return rows or []


    def desk_sent(self, n, since, until):
        """True when a desk message ('*cd ...', the desk's mark on the Zamai line) went to n in (since, until]"""
        rows = sb(f"zamai_messages?direction=eq.out&counterparty=eq.{n}&select=body"
                  f"&created_at=gt.{urllib.parse.quote(since)}&created_at=lte.{urllib.parse.quote(until)}"
                  "&order=created_at.desc&limit=20", key=self.key)
        return any((r.get("body") or "").lstrip().lower().startswith("*cd") for r in rows or [])


class OutboxSender:
    """SHADOW-LIVE (24 Oct 2026, Chris: "the whatsapp adapter needs to be live in as much as that it is possible to test
    data saving and data retrieval"): every reply that would go out is APPENDED to the outbox file with the exact Cloud
    API payload, and nothing is sent. The one exception, a second step on Chris's word: `real` (a CloudSender) is
    used for a TESTER number (allow-list tester=true) and only while config.live() (the three keys) holds; a teacher
    is never sent to from here, whatever the switches say."""

    def __init__(self, path, config, real=None):
        self.path, self.config, self.real = path, config, real
        self.testers = config.testers()

    def send(self, payload, log_text, msg_type):
        if self.real is not None and payload.get("to") in self.testers and self.config.live():
            return self.real.send(payload, log_text, msg_type)
        wa = "outbox:" + uuid.uuid4().hex[:12]
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": utcnow().isoformat(timespec="seconds"), "id": wa, "to": payload.get("to"),
                                "msg_type": msg_type, "text": log_text, "payload": payload}, ensure_ascii=False) + chr(10))
        return wa, None


class MockSender:
    """records every payload; 'fail' makes the next N sends fail (tests)"""

    def __init__(self):
        self.sent = []
        self.fail = 0

    def send(self, payload, log_text, msg_type):
        if self.fail > 0:
            self.fail -= 1
            return None, "mock failure"
        wa = f"wamid.mock{len(self.sent) + 1}"
        self.sent.append({"wa_id": wa, "to": payload["to"], "payload": payload, "text": log_text, "msg_type": msg_type})
        return wa, None


class CloudSender:
    def __init__(self, config):
        self.config = config
        self.key = service_key()
        rows = sb("whatsapp_config?line_key=eq.zamai&select=access_token,phone_number_id", key=self.key)
        if not rows or not rows[0].get("access_token") or not rows[0].get("phone_number_id"):
            raise SystemExit("zamai line not connected (no credentials on whatsapp_config line_key=zamai)")
        self.token, self.phone_id = rows[0]["access_token"], rows[0]["phone_number_id"]

    def send(self, payload, log_text, msg_type):
        if not self.config.live():          # checked at the moment of the send, every time
            return None, "refused: not live (env flag, go file or stop)"
        req = urllib.request.Request(GRAPH.format(phone_id=self.phone_id), method="POST",
                                     headers={"Authorization": "Bearer " + self.token,
                                              "Content-Type": "application/json"},
                                     data=json.dumps(payload).encode())
        wa, err = None, None
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                wa = (json.loads(r.read().decode() or "{}").get("messages") or [{}])[0].get("id")
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")[:300]
        except Exception as e:  # noqa: BLE001
            err = str(e)
        try:
            sb("zamai_messages", method="POST", prefer="return=minimal", key=self.key, body={
                "wa_message_id": wa, "direction": "out", "counterparty": payload["to"], "body": log_text,
                "msg_type": msg_type, "status": "failed" if err else "sent", "error": err})
        except Exception as e:  # noqa: BLE001  (the local log still has it)
            err = (err or "") + f" | log row failed: {e}"
        return wa, err


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)
