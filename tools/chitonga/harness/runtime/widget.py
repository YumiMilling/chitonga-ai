"""The Zamai assistant status widget: a small local page on http://127.0.0.1:8765 (24 Oct 2026, the Jesuit).

    pythonw tools/chitonga/harness/runtime/widget.py          (what the desktop icon runs: serve + open the window)
    python -X utf8 tools/chitonga/harness/runtime/widget.py --text     (the same status in this console, once)
    python -X utf8 tools/chitonga/harness/runtime/widget.py --no-open  (serve only)

WHY A LOCAL PAGE: this Python has no tkinter, and an earlier responder's widget is also a page in a bare Edge window
(--app). Stdlib only (http.server). It binds to 127.0.0.1 only, so nothing outside this machine can reach it.
The buttons are POSTs that must carry a random token printed into the page when it is served, and the Host header
must be 127.0.0.1/localhost: another web page open in the browser cannot press STOP or START (no CSRF, no DNS
rebinding). The server exits by itself after 30 minutes without a request (the open page refreshes every 10 s, so
it lives exactly as long as the window is open). A second start while one serves just opens the window.

Shows: running / stopped (process + heartbeat), mode ("LIVE (tester only)" or shadow, from runtime_mode.json), RUNTIME_STOP and
PILOT_STOP, per allow-listed number its mode, last inbound and today's turns, today's turns, labels, outbox,
the last error line, the watchdog's last line. STOP writes RUNTIME_STOP and stops the process (clean exit at the
next poll, killed after 90 s); START removes RUNTIME_STOP and launches through explorer.exe.
"""
import argparse
import html
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402

PORT = 8765
IDLE_EXIT_S = 1800
TOKEN = secrets.token_urlsafe(16)
LAST = {"t": time.time(), "msg": ""}


def fmt_age(s):
    if s is None:
        return "never"
    s = int(s)
    return f"{s} s" if s < 120 else f"{s // 60} min" if s < 7200 else f"{s // 3600} h"


def page(st):
    e = html.escape
    up = st["running"]
    colour = "#1f6f4a" if up else "#9a5b1e"       # evergreen / copper (no red, the brand rule)
    head = ("RUNNING" if up else "STOPPED" if st["runtime_stop"] else
            "NOT RUNNING" if not st["process"]["alive"] else "HUNG (stale heartbeat)")
    rows = "".join(
        f"<tr><td>{e(n['code'] or '')}</td><td>{e(n['name'] or '')}{' (tester)' if n['tester'] else ''}</td>"
        f"<td>{e(n['mode'] or '')}</td><td>{e((n['last_in_at'] or '-')[:19].replace('T', ' '))}</td>"
        f"<td>{n['turns_today']} / {n['engine_today']}</td></tr>" for n in st["numbers"])
    btn = ("<form method=post action=/stop><input type=hidden name=t value='{t}'><button>STOP</button></form>"
           if not st["runtime_stop"] else
           "<form method=post action=/start><input type=hidden name=t value='{t}'><button>START</button></form>"
           ).format(t=TOKEN)
    if st["process"]["alive"] and st["runtime_stop"]:
        btn += "<p class=s>stopping ...</p>"
    elif not st["runtime_stop"] and not up:
        btn += ("<form method=post action=/start><input type=hidden name=t value='{t}'><button>START now</button>"
                "</form>").format(t=TOKEN)
    return f"""<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=10>
<title>Zamai assistant</title><style>
body{{font:14px Segoe UI,Arial,sans-serif;margin:12px;color:#222}} h1{{font-size:18px;margin:0 0 6px}}
.b{{display:inline-block;padding:3px 10px;border-radius:4px;color:#fff;background:{colour};font-weight:600}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border-bottom:1px solid #ddd;padding:3px 8px;text-align:left}}
.s{{color:#666;font-size:12px}} button{{font-size:14px;padding:4px 18px;margin:6px 6px 0 0}} form{{display:inline}}
code{{font-size:12px;word-break:break-all}}</style></head><body>
<h1>Zamai assistant <span class=b>{head}</span></h1>
<div>mode: <b>{e(st['mode'])}</b></div>
<div class=s>pid {st['process']['pid'] or '-'} ({e(st['process']['owner'])}), heartbeat {fmt_age(st['heartbeat_age_s'])} ago;
RUNTIME_STOP {'ON' if st['runtime_stop'] else 'off'}; PILOT_STOP {'ON (answers nothing, routes to desk)' if st['pilot_stop'] else 'off'}</div>
<table><tr><th></th><th>number</th><th>mode</th><th>last inbound (UTC)</th><th>today turns / engine</th></tr>{rows}</table>
<div>today ({st['day']}): <b>{st['turns_today']}</b> turns, {st['engine_turns_today']} to the engine;
labels <b>{st['labels']}</b> ({st['labels_today']} today); outbox <b>{st['outbox']}</b> (would-be sends)</div>
<div class=s>last error: <code>{e(st['last_error'] or 'none')}</code></div>
<div class=s>watchdog: <code>{e(st['watchdog_last'] or 'no run logged (task installed?)')}</code></div>
{btn}<span class=s>{e(LAST['msg'])}</span>
<div class=s>refreshes every 10 s; {e(st['at'])}</div></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        LAST["t"] = time.time()
        if not self._host_ok():
            return self._send(403, "forbidden")
        st = C.status(C.Paths())
        if self.path.startswith("/status.json"):
            return self._send(200, json.dumps(st, ensure_ascii=False), "application/json")
        return self._send(200, page(st))

    def do_POST(self):
        LAST["t"] = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(min(n, 4096)).decode("utf-8", "replace")
        from urllib.parse import parse_qs
        tok = (parse_qs(body).get("t") or [""])[0]
        if not self._host_ok() or not secrets.compare_digest(tok, TOKEN):
            return self._send(403, "forbidden")
        p = C.Paths()
        if self.path == "/stop":
            open_stop = threading.Thread(target=lambda: LAST.update(msg=C.request_stop(p, "widget")), daemon=True)
            open_stop.start()
            LAST["msg"] = "STOP: RUNTIME_STOP written, the loop exits at its next poll (up to ~15 s)"
            time.sleep(1.5)
        elif self.path == "/start":
            LAST["msg"] = "START: launching ..."
            threading.Thread(target=lambda: LAST.update(msg="START: " + C.request_start(p, "widget")),
                             daemon=True).start()
            time.sleep(1.5)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def open_window():
    url = f"http://127.0.0.1:{PORT}/"
    edge = next((x for x in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if os.path.exists(x)), None)
    if edge:
        subprocess.Popen([edge, f"--app={url}", "--window-size=560,520"], close_fds=True)
    else:
        import webbrowser
        webbrowser.open(url)


def port_in_use():
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def text_status():
    st = C.status(C.Paths())
    print(f"Zamai assistant: {'RUNNING' if st['running'] else 'NOT RUNNING'}  mode: {st['mode']}")
    print(f"  pid {st['process']['pid']} alive={st['process']['alive']} owner={st['process']['owner']}; "
          f"heartbeat {fmt_age(st['heartbeat_age_s'])} ago; RUNTIME_STOP {'ON' if st['runtime_stop'] else 'off'}; "
          f"PILOT_STOP {'ON' if st['pilot_stop'] else 'off'}")
    for n in st["numbers"]:
        print(f"  {n['code']} {n['name']}: mode {n['mode']}, last inbound {n['last_in_at'] or '-'}, "
              f"today {n['turns_today']} turns ({n['engine_today']} engine)")
    print(f"  today {st['day']}: {st['turns_today']} turns, {st['engine_turns_today']} engine; labels {st['labels']} "
          f"({st['labels_today']} today); outbox {st['outbox']}")
    print(f"  last error: {st['last_error'] or 'none'}")
    print(f"  watchdog: {st['watchdog_last'] or 'no run logged'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    if a.text:
        return text_status()
    if port_in_use():
        if not a.no_open:
            open_window()
        return
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if not a.no_open:
        open_window()
    while time.time() - LAST["t"] < IDLE_EXIT_S:
        time.sleep(30)
    srv.shutdown()


if __name__ == "__main__":
    main()
