"""Local server for the teacher web app (24 Oct 2026, the Jesuit). Stdlib only (no Flask on this machine).

    python -X utf8 tools/chitonga/harness/web/server.py                       serve on http://127.0.0.1:8766
    python -X utf8 tools/chitonga/harness/web/server.py --new-key T00         make a key for an allow-listed teacher
    python -X utf8 tools/chitonga/harness/web/server.py --lan                 ALSO answer on the LAN IP (a quick look
                                                                              only: plain http, no offline; prefer adb)
PORT 8766, not 8765: 8765 is the assistant runtime's status widget (runtime/widget.py), which treats anything on
8765 as itself.

SAFETY. Binds 127.0.0.1 unless --lan. Every /api/ call needs the teacher key in the X-Key header (a custom header:
another web page cannot send it without a CORS preflight, which this server never grants). The Host header must be
127.0.0.1 / localhost (or the LAN IP with --lan): DNS rebinding cannot reach the API. noindex on every response,
CSP 'self', no third-party anything. The static shell holds no Tonga; all teacher Tonga comes from the keyed API.
Nothing here is published: site/ is the only published folder and this is not in it.
"""
import argparse
import gzip
import json
import mimetypes
import re
import os
import socket
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api as A  # noqa: E402

STATIC = os.path.join(HERE, "static")
PORT = 8766
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("font/woff2", ".woff2")
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def make_handler(api, hosts):
    class H(BaseHTTPRequestHandler):
        server_version = "zamai-teacher/0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):          # quiet; the API logs what matters
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None, gz_body=None):
            t0 = getattr(self, "_t0", time.perf_counter())
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            use_gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 1200
            if use_gz:
                body = gz_body or gzip.compress(body, 6)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if use_gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding, X-Key")
            self.send_header("X-Robots-Tag", "noindex, nofollow, noarchive")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Server-Timing", f"app;dur={(time.perf_counter() - t0) * 1000:.1f}")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _host_ok(self):
            h = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            return h in hosts

        def _who(self):
            return api.teacher(self.headers.get("X-Key") or "")

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 256_000:
                raise ValueError("too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            self._t0 = time.perf_counter()
            if not self._host_ok():
                return self._send(421, {"error": "host"})
            u = urlparse(self.path)
            if u.path.startswith("/api/"):
                return self._api_get(u)
            return self._static(u.path)

        def do_POST(self):
            self._t0 = time.perf_counter()
            if not self._host_ok():
                return self._send(421, {"error": "host"})
            u = urlparse(self.path)
            who = self._who()
            if not who:
                return self._send(401, {"error": "key"})
            try:
                body = self._body()
                if u.path == "/api/onboard":
                    return self._send(200, api.onboard(who, body))
                if u.path == "/api/sync":
                    return self._send(200, {"results": api.sync_batch(who, body.get("ops") or [])})
                if u.path == "/api/search":
                    return self._send(200, api.search(who, body.get("q") or ""))
                # 24 Oct 2026: quick-add parse and the digest job (TIMETABLE_API.md); a bad parameter is a 400
                if u.path in ("/api/quickadd", "/api/digest/run"):
                    try:
                        fn = api.quickadd if u.path == "/api/quickadd" else api.digest_run
                        return self._send(200, fn(who, body), extra={"Cache-Control": "no-store"})
                    except ValueError as e:
                        return self._send(400, {"error": str(e)[:200]})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._send(500, {"error": str(e)[:200]})

        def _api_get(self, u):
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/api/ping":
                return self._send(200, {"ok": True, "fingerprint": api.fingerprint()})
            who = self._who()
            if not who:
                return self._send(401, {"error": "key"})
            try:
                if u.path == "/api/me":
                    return self._send(200, api.me(who), extra={"Cache-Control": "no-store"})
                if u.path == "/api/home":
                    return self._send(200, api.home(who), extra={"Cache-Control": "no-store"})
                # 24 Oct 2026: generated board plans, lesson maps, tip diagrams and their index (diagrams_build.py), behind
                # the key as on the Netlify demo (they carry lesson text and checked Tonga key words)
                if u.path == "/api/gen":
                    f = q.get("f") or ""
                    if not re.fullmatch(r"(?:(?:board|map|tips)/[A-Za-z0-9_.\-]+\.svg|art_index\.json)", f) or ".." in f:
                        return self._send(400, {"error": "file"})
                    fp = os.path.join(HERE, "diagrams_gen", *f.split("/"))
                    if not os.path.isfile(fp):
                        return self._send(404, {"error": "no such file"})
                    with open(fp, "rb") as fh:
                        data = fh.read()
                    ct = "application/json; charset=utf-8" if f.endswith(".json") else "image/svg+xml; charset=utf-8"
                    return self._send(200, data, ctype=ct, extra={"Cache-Control": "private, max-age=31536000, immutable"})
                # 24 Oct 2026: the bundle in parts (api.WebAPI.split_bundle; the phone fetches what it opens)
                if u.path == "/api/chunk":
                    part = q.get("part") or ""
                    if part not in ("index", "item", "subject", "syllabus", "tips", "search"):
                        return self._send(400, {"error": "part"})
                    try:
                        grade = int(q.get("grade") or (1 if part in ("tips", "search") else 0))
                    except ValueError:
                        return self._send(400, {"error": "grade"})
                    data = api.chunk(grade, part, id=q.get("id"), subject=q.get("subject"))
                    if data is None:
                        return self._send(404, {"error": "no such part"})
                    return self._send(200, data, extra={"Cache-Control": "private, max-age=31536000, immutable"})
                if u.path == "/api/bundle":
                    b = api.bundle(q.get("grade") or 0)
                    if not b:
                        return self._send(404, {"error": "grade"})
                    _, raw, gz = b
                    return self._send(200, raw, gz_body=gz, extra={"Cache-Control": "private, no-cache",
                                                                    "ETag": f'"{api.fingerprint()}-{q.get("grade")}"'})
                # 24 Oct 2026: the syllabus (SYLLABUS_API.md); a bad parameter is a 400, never a 500
                if u.path.startswith("/api/syllabus"):
                    try:
                        if u.path == "/api/syllabus":
                            return self._send(200, api.syllabus(q.get("grade"), q.get("area"), q.get("topic"),
                                                                q.get("sub")))
                        if u.path == "/api/syllabus/item":
                            return self._send(200, api.syllabus_item(q.get("id")))
                        if u.path == "/api/syllabus/week":
                            return self._send(200, api.syllabus_week(q.get("grade"), q.get("term"), q.get("week"),
                                                                     q.get("area")))
                        if u.path == "/api/syllabus/coverage":
                            return self._send(200, api.syllabus_coverage(q.get("grade"), q.get("area")))
                    except ValueError as e:
                        return self._send(400, {"error": str(e)})
                # 24 Oct 2026: timetable, calendar, Today, week, Suggested, notes, digest (TIMETABLE_API.md)
                tt_get = {"/api/timetable": lambda: api.timetable(who, q.get("year")),
                          "/api/calendar": lambda: api.calendar(who, q.get("from"), q.get("to")),
                          "/api/today": lambda: api.today(who, q.get("date")),
                          "/api/week": lambda: api.week(who, q.get("start")),
                          "/api/suggested": lambda: api.suggested(who, q.get("date")),
                          "/api/notes": lambda: api.notes(who, q.get("from"), q.get("to"), q.get("class_id")),
                          "/api/digest": lambda: api.digest(who, q.get("class_id"), q.get("week"))}
                if u.path in tt_get:
                    try:
                        return self._send(200, tt_get[u.path](), extra={"Cache-Control": "no-store"})
                    except ValueError as e:
                        return self._send(400, {"error": str(e)[:200]})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._send(500, {"error": str(e)[:200]})

        def _static(self, path):
            if path in ("", "/"):
                path = "/index.html"
            fp = os.path.normpath(os.path.join(STATIC, path.lstrip("/")))
            if not fp.startswith(STATIC) or not os.path.isfile(fp):
                fp = os.path.join(STATIC, "index.html")       # the app routes by hash; unknown paths get the shell
            with open(fp, "rb") as f:
                body = f.read()
            ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith(("json", "javascript", "xml")):
                ctype += "; charset=utf-8"
            extra = {"Cache-Control": "no-cache"}
            if fp.endswith("sw.js"):
                extra["Service-Worker-Allowed"] = "/"
            return self._send(200, body, ctype, extra)

    return H


def serve(port=PORT, lan=False, api=None):
    api = api or A.WebAPI()
    hosts = {"127.0.0.1", "localhost"}
    bind = "127.0.0.1"
    ip = lan_ip() if lan else None
    if ip:
        hosts.add(ip)
        bind = ip if False else "0.0.0.0"
    srv = ThreadingHTTPServer((bind, port), make_handler(api, hosts))
    srv.daemon_threads = True
    print(f"teacher app on http://127.0.0.1:{port}/" + (f"  and http://{ip}:{port}/ (LAN, plain http)" if ip else ""),
          flush=True)
    srv.serve_forever()


def new_key(code):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
    from adapter.config import Config  # noqa: PLC0415
    allow = Config().allowlist()
    e = next((v for v in allow.values() if v.get("code") == code), None)
    if not e:
        sys.exit(f"{code} is not on the pilot allow-list")
    k = A.KeyStore().create(code, e["number"], e.get("name") or code, e.get("role"), e.get("tester"))
    print(f"key for {code} ({e.get('name')}): {k}\nlocal link: http://127.0.0.1:{PORT}/?k={k}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--lan", action="store_true")
    ap.add_argument("--new-key", metavar="CODE")
    a = ap.parse_args()
    if a.new_key:
        new_key(a.new_key)
    else:
        serve(a.port, a.lan)
