"""The installed translator, called on the translator pod when it is up, on the Beelink when it is not (23 Oct 2026, the
Jesuit; Chris: "If the beelink is the bottleneck, we need to get cheap pods to open up the bottleneck").

Same call shape and result dict as translator_service.service().translate, plus "served_by": "pod" | "beelink":

    import translate_remote as TR
    r = TR.translate("Wash your hands with soap.", "en>toi")          # r["text"], r["alarms"], r["pieces"], ...
    rs = TR.translate_many([(text, "en>toi"), ...], threads=6)          # the pod runs several workers: send in parallel

The pod: tools/chitonga/train/launch_translator_pod.sh writes TRANSLATOR_POD_URL and TRANSLATOR_POD_TOKEN to .env.local
(the token is never printed). Fallback: if the pod does not answer /health, the local Service runs under
beelink_lock.held(...) (one heavy Beelink job at a time). A stream that already HOLDS the Beelink lock passes
lock=False, or it would wait on itself. allow_local=False raises instead of falling back (the identity check uses it).

Note: the pod decodes CT2 int8 on CUDA, the Beelink on the CPU; same weights and code, not always the same text
(translator_pod_check.py measures it; LOGBOOK 22 Oct: about half of the lines differ in a word or more).
"""
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".env.local"))
_health = {"t": 0.0, "up": False}


def _env():
    e = {}
    try:
        for l in open(ENV, encoding="utf-8"):
            if "=" in l and l.split("=", 1)[0].strip() in ("TRANSLATOR_POD_URL", "TRANSLATOR_POD_TOKEN"):
                k, v = l.split("=", 1)
                e[k.strip()] = v.strip().strip('"')
    except FileNotFoundError:
        pass
    for k in ("TRANSLATOR_POD_URL", "TRANSLATOR_POD_TOKEN"):     # the process environment wins (a local test server)
        if os.environ.get(k):
            e[k] = os.environ[k]
    return e.get("TRANSLATOR_POD_URL", "").rstrip("/"), e.get("TRANSLATOR_POD_TOKEN", "")


def _call(method, path, body=None, timeout=110):
    url, tok = _env()
    if not url or not tok:
        raise ConnectionError("no TRANSLATOR_POD_URL / TRANSLATOR_POD_TOKEN in .env.local")
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url + path, data=data, method=method,
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json",
                                          "User-Agent": "zamai-translate-remote"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return e.code, {"error": str(e)}


def pod_up(max_age=60):
    """the pod answers /health with our token (cached for max_age s)"""
    if time.time() - _health["t"] < max_age:
        return _health["up"]
    try:
        code, j = _call("GET", "/health", timeout=20)
        up = code == 200 and "workers" in j
    except Exception:  # noqa: BLE001
        up = False
    _health.update(t=time.time(), up=up)
    return up


def status():
    """the pod's /health with the token: served, pending, idle_s, device (None if down)"""
    try:
        code, j = _call("GET", "/health", timeout=20)
        return j if code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def _remote(text, direction, keep, use_memory, deadline_s=1800):
    code, j = _call("POST", "/translate", {"text": text, "direction": direction, "keep": keep, "use_memory": use_memory})
    t0 = time.time()
    while code == 202 and time.time() - t0 < deadline_s:
        code, j = _call("GET", "/job/" + j["job"])
    if 400 <= code < 500:
        # the request was wrong (bad direction, too long, token): the pod is up; the caller's error, no fallback
        raise ValueError(f"pod refused the request ({code}): {str(j)[:300]}")
    if code != 200:
        raise PodAnswerError(f"pod answered {code}: {str(j)[:300]}")
    return j


class PodAnswerError(RuntimeError):
    """the pod answered, but not with a result (a worker error, 500; a job past the deadline): it is UP, so this does not
    mark it down (local gate 3, 23 Oct: one forced 400 marked the pod down for 60 s and the next good request failed;
    with allow_local=True every stream would have fallen back to the Beelink for a minute after any bad request)"""


_local = {}


def _local_translate(text, direction, keep, use_memory, lock):
    import sys
    sys.path.insert(0, HERE)
    import translator_service as TS
    if lock:
        import beelink_lock
        with beelink_lock.held(f"translate_remote-{os.getpid()}"):
            r = TS.service().translate(text, direction, keep=keep, use_memory=use_memory)
    else:
        r = TS.service().translate(text, direction, keep=keep, use_memory=use_memory)
    r = json.loads(json.dumps(r, ensure_ascii=False, default=str))
    r["served_by"] = "beelink"
    return r


def translate(text, direction="auto", keep=False, use_memory=None, allow_local=True, lock=True):
    """translator_service.Service.translate on the pod, else locally under the Beelink lock"""
    if pod_up():
        try:
            return _remote(text, direction, keep, use_memory)
        except (ConnectionError, OSError, RuntimeError, urllib.error.URLError) as e:
            if not isinstance(e, PodAnswerError):          # only a pod that did not answer counts as down
                _health.update(t=time.time(), up=False)
            if not allow_local:
                raise
            print(f"translate_remote: pod failed ({str(e)[:120]}); local fallback", flush=True)
    elif not allow_local:
        raise ConnectionError("translator pod is down")
    return _local_translate(text, direction, keep, use_memory, lock)


def translate_many(items, threads=6, **kw):
    """items: [(text, direction)]; results in order. Parallel on the pod; one at a time on the Beelink fallback."""
    if not pod_up():
        return [translate(t, d, **kw) for t, d in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(lambda x: translate(x[0], x[1], **kw), items))


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--status"]:
        print(json.dumps(status()))
    else:
        d = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "auto"
        r = translate(sys.argv[1], d)
        print(r["text"])
        print(f"[{r['direction']} {r.get('stage')} {r.get('t')}s served_by {r['served_by']} alarms {r.get('alarms')}]")
