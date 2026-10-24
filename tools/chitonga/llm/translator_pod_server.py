"""The installed translator over HTTP, for ONE rented GPU pod (23 Oct 2026, the Jesuit; Chris: "If the beelink is the
bottleneck, we need to get cheap pods to open up the bottleneck").

What it serves: translator_service.Service().translate(text, direction, keep, use_memory), unchanged: the same code, the
same chitonga-nllb-wise-a050 weights (int8 CT2, made on the pod from the verified HF repo), term guard, P1 layer and
checks. Only the device differs (CT2_DEVICE=cuda); see translator_pod_check.py for what that does to the text.

Why a process pool and not threads: the Service holds one lock per reader (Codex #5, tokenizer state), so threads in one
process queue behind each other; WORKERS separate processes each load their own Service (the 22 Oct weakness map ran
three such processes on one 4090: 20 GB, GPU 100%).

Why jobs: the RunPod HTTPS proxy closes a request after ~100 s, and a long paragraph through the full service can take
longer (showcase answers up to 205 s on the 4090, LOGBOOK 22 Oct). POST /translate waits up to WAIT s and then answers
202 {"job": id}; GET /job/<id> waits again. translate_remote.py hides this.

    POST /translate  {"text", "direction": "auto|en>toi|toi>en", "use_memory": null|bool, "keep": false}
                     -> 200 the service's result dict | 202 {"job": id}
    GET  /job/<id>   -> 200 result | 202 {"job": id} | 404
    GET  /health     -> {"ok": true}; with the token also workers, served, pending, idle_s, model
Every call except the bare /health needs "Authorization: Bearer <TRANSLATOR_TOKEN>". The token is never logged.
Env: TRANSLATOR_TOKEN, PORT (8000), WORKERS (3), STATE_DIR (/root/state: last_request, served for the idle watchdog).
"""
import hmac
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = os.environ.get("TRANSLATOR_TOKEN", "")
PORT = int(os.environ.get("PORT", "8000"))
WORKERS = int(os.environ.get("WORKERS", "3"))
STATE = os.environ.get("STATE_DIR", "/root/state")
WAIT = 75
MAX_CHARS = 20000

_svc = None


def _init():
    global _svc
    sys.path.insert(0, HERE)
    import translator_service as TS
    _svc = TS.service()
    _svc.translate("Wash your hands.", "en>toi", use_memory=False)       # load all readers before the first request
    _svc.translate("Kamusamba maanza.", "toi>en", use_memory=False)


def _work(text, direction, keep, use_memory):
    """errors come back as a plain dict: an exception class from a module the parent never imported (heldout_guard,
    23 Oct) cannot be unpickled there and kills the pool's result thread, so every later request hangs"""
    t = time.time()
    try:
        r = _svc.translate(text, direction, keep=keep, use_memory=use_memory)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"_error": f"{type(e).__name__}: {e}"[:500], "_trace": traceback.format_exc()[-1500:]}
    r = json.loads(json.dumps(r, ensure_ascii=False, default=str))
    r["served_by"] = "pod"
    r["pod_s"] = round(time.time() - t, 2)
    return r


class State:
    def __init__(self):
        self.jobs, self.lock = {}, threading.Lock()
        self.served = self.failed = 0
        self.last = time.time()
        self.pool = None
        os.makedirs(STATE, exist_ok=True)
        self.touch()

    def touch(self):
        self.last = time.time()
        open(os.path.join(STATE, "last_request"), "w").write(str(int(self.last)))

    def count(self, ok):
        with self.lock:
            if ok:
                self.served += 1
            else:
                self.failed += 1
            open(os.path.join(STATE, "served"), "w").write(f"{self.served} {self.failed}")

    def submit(self, text, direction, keep, use_memory):
        jid = uuid.uuid4().hex
        ev = threading.Event()
        job = {"ev": ev, "res": None, "err": None, "t0": time.time()}

        def ok(r):
            if "_error" in r:
                job["err"] = r["_error"]
                print("REQUEST FAILED", r["_error"], r.get("_trace", ""), flush=True)
                self.count(False)
            else:
                job["res"] = r
                self.count(True)
            ev.set()

        def bad(e):
            job["err"] = repr(e)[:500]
            self.count(False)
            ev.set()
        with self.lock:
            self.jobs[jid] = job
            for k in [k for k, j in self.jobs.items() if j["ev"].is_set() and time.time() - j["t0"] > 3600]:
                del self.jobs[k]
        self.pool.apply_async(_work, (text, direction, keep, use_memory), callback=ok, error_callback=bad)
        return jid

    def pending(self):
        return sum(1 for j in self.jobs.values() if not j["ev"].is_set())


S = None


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):       # no request lines with headers in the log
        pass

    def send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def authed(self):
        got = self.headers.get("Authorization", "")
        return bool(TOKEN) and hmac.compare_digest(got.encode(), ("Bearer " + TOKEN).encode())

    def answer(self, jid):
        j = S.jobs.get(jid)
        if not j:
            return self.send(404, {"error": "no such job"})
        j["ev"].wait(WAIT)
        if not j["ev"].is_set():
            return self.send(202, {"job": jid})
        S.jobs.pop(jid, None)
        if j["err"]:
            return self.send(500, {"error": j["err"]})
        return self.send(200, j["res"])

    def do_GET(self):
        if self.path == "/health":
            if not self.authed():
                return self.send(200, {"ok": True})
            return self.send(200, {"ok": True, "workers": WORKERS, "served": S.served, "failed": S.failed,
                                   "pending": S.pending(), "idle_s": int(time.time() - S.last),
                                   "model": os.environ.get("CHITONGA_TRANSLATOR", "chitonga-nllb-wise-a050"),
                                   "device": os.environ.get("CT2_DEVICE", "cpu")})
        if not self.authed():
            return self.send(401, {"error": "token"})
        if self.path.startswith("/job/"):
            S.touch()
            return self.answer(self.path[5:])
        return self.send(404, {"error": "path"})

    def do_POST(self):
        if not self.authed():
            return self.send(401, {"error": "token"})
        if self.path != "/translate":
            return self.send(404, {"error": "path"})
        S.touch()
        try:
            n = int(self.headers.get("Content-Length", "0"))
            q = json.loads(self.rfile.read(n).decode("utf-8"))
            text = q["text"]
            direction = q.get("direction", "auto")
            assert isinstance(text, str) and len(text) <= MAX_CHARS, "text: a string up to 20000 characters"
            assert direction in ("auto", "en>toi", "toi>en"), "direction: auto, en>toi or toi>en"
            um = q.get("use_memory")
            assert um in (None, True, False), "use_memory: null, true or false"
        except Exception as e:  # noqa: BLE001
            return self.send(400, {"error": repr(e)[:300]})
        return self.answer(S.submit(text, direction, bool(q.get("keep", False)), um))


def main():
    global S
    assert len(TOKEN) >= 32, "TRANSLATOR_TOKEN missing or short"
    sys.path[:0] = [HERE, os.path.join(HERE, "..", "train")]    # the parent can unpickle what the workers raise
    S = State()
    t = time.time()
    S.pool = mp.get_context("spawn").Pool(WORKERS, initializer=_init)
    # every worker loads before the port opens: one probe per worker, all at once
    rs = [S.pool.apply_async(_work, ("Good morning, class.", "en>toi", False, um)) for um in [False, None] * WORKERS]
    got = [r.get(timeout=1800) for r in rs]
    bad = [g["_error"] for g in got if "_error" in g]
    assert not bad, "probe failed: " + bad[0]
    print("probe", got[0]["text"], "| memory on:", got[1]["text"], flush=True)
    print(f"SERVER READY port {PORT} workers {WORKERS} load {time.time() - t:.0f}s", flush=True)
    S.touch()
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
