"""The reader bake-off: which STANDARD chat model reads the translator's English in the harness (24 Oct 2026, the Jesuit).

Chris, 24 Oct 2026: "Can we test different base models already? Gemma is one option? Qwen another one? What else? How do
we test and decide?" Route (DECISIONS 24 Oct): harness -> translator (toi>en) -> standard model reading English ->
harness. The model only READS: it picks intent and item from the harness's candidates under the v14.1 JSON schema
(core/reader_model.py). The rule is pre-registered in internal/chitonga_train/runs/2026-10-24_reader_bakeoff/RULE.md.

WHY FROZEN REQUEST BODIES. The harness asks the model only for what the rules cannot place, and what it sends depends
on the plug-in, the retrieval and the translator's English. If each arm ran the harness itself (on the pod, on the
Beelink, on another day) a changed plug-in, a translator memory hit or a missing seed model (the 24 Oct B2 run that
silently became B1) would give arms different questions. So the Beelink computes the exact request bodies once, every
arm answers THOSE bodies (on a RunPod GPU, or on the Beelink), and the answers are fed back through the actual harness
on the Beelink. The pod needs no harness, no translator, no plug-in: only the bodies and llama.cpp. The pod also holds
less teacher data that way (request texts and candidate titles, nothing else).

    python -X utf8 tools/chitonga/llm/reader_bakeoff.py freeze      # Beelink, LOCK: served translator once + note English
    python -X utf8 tools/chitonga/llm/reader_bakeoff.py capture     # Beelink, light: bodies + the rules arm (no model)
    python -X utf8 tools/chitonga/llm/reader_bakeoff.py pack        # the pod pack (bodies + models.json + pod scripts)
    ... bash tools/chitonga/train/launch_reader_bakeoff.sh          # RunPod: every arm answers every body (step A)
    python -X utf8 tools/chitonga/llm/reader_bakeoff.py replay [--tag T]        # Beelink, light: replies -> harness -> rows
    python -X utf8 tools/chitonga/llm/reader_bakeoff.py adjudicate              # the blind list for the gold judge
    python -X utf8 tools/chitonga/llm/reader_bakeoff.py speed --tag G_q4 --gguf <file> --n 20   # Beelink, LOCK:
    python -X utf8 tools/chitonga/llm/reader_bakeoff.py agree --tag G_q4        # the hardware note (RULE v2 s.7), info only
    python -X utf8 tools/chitonga/llm/reader_bakeoff.py score                   # the pre-registered rule -> RESULT.md
All outputs under --work (default: the run folder). --testset defaults to testset_v2.jsonl beside RULE.md.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
RUN = os.path.join(CT, "runs", "2026-10-24_reader_bakeoff")
# 24 Oct 17:52: another agent rebuilt the plug-in in place and the frozen 14:57 build was lost. Since then the bake-off
# runs on its OWN snapshot of the harness code and the plug-in data (RUN/frozen/, FROZEN.json with tree hashes): core
# and plugins are imported from the snapshot BEFORE v14_eval_harness (which puts the live harness first on sys.path);
# their submodules then resolve through the already-imported packages. RB_LIVE_HARNESS=1 uses the live tree.
FROZEN = os.path.join(RUN, "frozen")
_SNAP = os.path.isdir(os.path.join(FROZEN, "harness")) and not os.environ.get("RB_LIVE_HARNESS")
HARNESS = os.path.join(FROZEN, "harness") if _SNAP else os.path.join(REPO, "tools", "chitonga", "harness")
PLUGIN_DATA = os.path.join(FROZEN, "plugin_education") if _SNAP else None
for p in (HERE, os.path.join(HERE, "..", "assistant"), HARNESS):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import core  # noqa: E402,F401
import plugins.education  # noqa: E402,F401
import plugins.education.run_testset  # noqa: E402,F401
import v14_eval_harness as V  # noqa: E402  (the v14 instrument: reached(), summary(), grade_of(), serve())
assert os.path.dirname(os.path.dirname(core.__file__)) == HARNESS, f"core imported from {core.__file__}, not {HARNESS}"
from core import io as IO  # noqa: E402
from core import reader_model as RM  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.intake import RulesReader  # noqa: E402
from core.log import TurnLog  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402

FROZEN_FIELDS = ("reading_en", "en_frozen", "translator_en", "frozen_en")   # a test row's frozen translator English
GATE_OUTCOMES = ("outcome:safeguarding", "outcome:health_no_card", "outcome:redirect")
ITEM_PREFIX = ("card:", "lesson:", "item:", "pointer:", "unit:")
N_LATENCY, N_AGREE = 60, 20
ADJ_SEED = 20261024
LOCK_FREEZE, LOCK_SPEED = "reader-bakeoff-freeze", "reader-bakeoff-speed"


class CaptureOnly(Exception):
    """raised by the capture post: the body is stored, the reader falls back to the rules reading"""


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(body):
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def key_of(body):
    return hashlib.sha256(canon(body).encode("utf-8")).hexdigest()[:32]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def wjl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wjs(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------ the test set
def load_testset(a):
    """rows of the frozen test set; refuses an unfrozen or edited file (its .sha256 written by the agent that built it)"""
    p = a.testset
    sp = p + ".sha256"
    assert os.path.exists(p), f"test set missing: {p}"
    assert os.path.exists(sp), f"test set not frozen (no {os.path.basename(sp)})"
    want = open(sp, encoding="utf-8").read().split()[0]
    have = sha_file(p)
    assert want == have, f"test set changed after freezing ({have[:12]} vs {want[:12]})"
    rows = jl(p)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids in the test set"
    for r in rows:
        assert r.get("text") and isinstance(r.get("accept"), list), f"row {r.get('id')} lacks text / accept"
    return rows, have


def frozen_en(r):
    for k in FROZEN_FIELDS:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def slice_of(r):
    return r.get("slice") or r.get("part") or "-"


def grade_for(r):
    g = r.get("grade")
    return int(g) if isinstance(g, int) and 1 <= g <= 7 else V.grade_of(r)


# ------------------------------------------------------------------------------------------------ frozen translation
class FrozenBackends:
    """the translator as the reader sees it: frozen English by exact text; the served translator only in `freeze`"""

    def __init__(self, table, real=None, translator="a050"):
        self.t, self.real, self.translator = table, real, translator
        self.calls, self.miss = 0, []

    def translate(self, text, direction, model=None):
        k = (text or "").strip()
        if k in self.t:
            return {"text": self.t[k]["en"]}
        if self.real is None:
            self.miss.append(text)
            raise KeyError("no frozen English for this text")
        r = self.real.translate(text, direction, model=model or self.translator)
        self.calls += 1
        self.t[k] = {"en": (r.get("text") or "").strip(), "source": "served",
                     "backend": getattr(self.real, "last_by", "beelink"), "alarms": r.get("alarms") or []}
        return r


class PodBackends:
    """the translator POD (translate_remote, allow_local=False: the pod runs the Beelink's exact int8 model and the
    served config, after its 20/20 identity check). A text the pod fails on goes to the Beelink's served backends,
    under the Beelink lock taken at the first fallback and released by close(); each translation records its backend."""

    def __init__(self, renderer):
        import translate_remote as TR  # noqa: PLC0415
        self.TR, self.renderer = TR, renderer
        self.local, self.locked, self.last_by = None, False, None
        self.by = {"pod": 0, "beelink": 0}

    def _beelink(self):
        if self.local is None:
            import beelink_lock  # noqa: PLC0415
            beelink_lock.acquire(LOCK_FREEZE)
            self.locked = True
            self.local = served_backends()[0]
        return self.local

    def translate(self, text, direction, model=None):
        try:
            r = self.TR.translate(text, direction, keep=True, use_memory=None, allow_local=False)
            self.last_by = "pod"
        except Exception as e:  # noqa: BLE001
            r = self._beelink().translate(text, direction, model=model or self.renderer)
            self.last_by = f"beelink (pod failed: {str(e)[:80]})"
        self.by[self.last_by.split()[0]] += 1
        return r

    def note_service(self):
        """an object with translate(text, direction) for notes_digest.Translator(service=...): the same pod-first rule"""
        outer = self

        class S:
            def translate(self, text, direction="toi>en", **kw):
                try:
                    r = outer.TR.translate(text, direction, use_memory=None, allow_local=False)
                    outer.by["pod"] += 1
                    outer.last_by = "pod"
                    return r
                except Exception:  # noqa: BLE001
                    outer._beelink()
                    import translator_service as TS  # noqa: PLC0415
                    outer.by["beelink"] += 1
                    outer.last_by = "beelink (pod failed)"
                    return TS.service().translate(text, direction)
        return S()

    def close(self):
        if self.locked:
            import beelink_lock  # noqa: PLC0415
            beelink_lock.release(LOCK_FREEZE)
            self.locked = False


class FrozenNotes:
    """the digest's translator: note_english() from the frozen table (or the served one in `freeze`)"""

    def __init__(self, table, real=None, src=None):
        self.t, self.real, self.src = table, real, src
        self.name = "translator_service (served, toi>en), frozen " + ("live" if real else "from frozen_notes.jsonl")

    def note_english(self, text):
        k = (text or "").strip()
        if k not in self.t:
            assert self.real is not None, f"no frozen note English: {k[:60]}"
            if self.src is not None:
                self.src.last_by = None
            en, lang, how = self.real.note_english(text)
            self.t[k] = {"en": en, "lang": lang, "how": how,
                         "backend": (getattr(self.src, "last_by", None) or "beelink") if how else None}
        x = self.t[k]
        return x["en"], x["lang"], x["how"]


class _LazyLocked:
    """the Beelink's served backends, the Beelink lock taken only at the first real translation (--reuse-frozen)"""

    def __init__(self, real):
        self.real, self.locked = real, False

    def translate(self, *a, **k):
        if not self.locked:
            import beelink_lock  # noqa: PLC0415
            beelink_lock.acquire(LOCK_FREEZE)
            self.locked = True
        return self.real.translate(*a, **k)

    def close(self):
        if self.locked:
            import beelink_lock  # noqa: PLC0415
            beelink_lock.release(LOCK_FREEZE)
            self.locked = False


class _EnglishOnlyNotes:
    """--translator none: an English note passes unchanged; a note the detector reads as Tonga is an error (RULE v4)"""

    def note_english(self, text):
        import translator_service as TS  # noqa: PLC0415
        assert TS.detect_direction(text) != "toi>en", f"a note reads as Tonga under --translator none: {text[:40]}"
        return text, "en", None


def served_backends():
    """backends.Real on the served config; seed / reader folders missing on this machine are dropped and SAID (24 Oct:
    a missing seed made translate() raise inside the reader and B2 silently became B1)"""
    import backends  # noqa: PLC0415
    import config  # noqa: PLC0415
    import h3_select as HS  # noqa: PLC0415
    import translator_service as TS  # noqa: PLC0415
    missing = [n for n, p in HS.EXTRA_READERS.items() if not os.path.isdir(p)]
    if missing:
        TS.CFG["seeds"] = tuple(s for s in TS.CFG.get("seeds", ()) if s not in missing)
        TS.CFG["readers"] = tuple(r for r in TS.CFG.get("readers", ()) if r not in missing)
    fp = {"renderer": config.CFG.get("renderer"), "models": config.CFG.get("models"), "dropped_missing": missing,
          "service_cfg": {k: v for k, v in TS.CFG.items() if isinstance(v, (str, int, float, bool, tuple, list))}}
    return backends.Real(config.CFG), config.CFG.get("renderer") or "a050", fp


# ------------------------------------------------------------------------------------------------ one harness turn
def turn(P, hooks, reader, it, i):
    """the v14 instrument's turn: a new teacher (name, language, one class at the item's grade), then the request;
    a confirmation is answered yes; scored by the targets reached (same fields as v14_eval_harness rows)"""
    e = Engine(P, SQLiteMemory(":memory:"), hooks=hooks, reader=reader, log=TurnLog(None))
    n = f"26098{i:06d}"
    e.handle(n, {"text": "hello"})
    e.handle(n, {"text": "Teacher"})
    e.handle(n, {"choice": "lang:" + ("toi" if it.get("lang", "en") != "en" else "en")})
    e.handle(n, {"choice": f"newgrade:{grade_for(it)}"})
    t0 = time.time()
    if hasattr(reader, "body_key"):
        reader.body_key = None
    replies = e.handle(n, {"text": it["text"]})
    dt_ = time.time() - t0
    body_key = getattr(reader, "body_key", None)     # read now: the confirmation turn must not reset it
    rec = e.last
    got = V.reached(P, rec, IO.as_text(replies[0]) if replies else "")
    # testset_v2 target kinds the v14 instrument does not tag (coordinator, 24 Oct, before any arm ran; the same for
    # every arm): outcome:menu = the reading was 'menu' (a subject named, no topic: the unit menu is shown). tip:<id> needs
    # no rule here: the chat harness has no tip route, and a tip delivered as an item would be tagged by reached()'s
    # deliver path like any item; tip rows score on their honest outcomes, accept_strict is reported apart.
    if any(p.startswith("read:menu:") for p in rec["path"]):
        got.add("outcome:menu")
    confirm = any(p.startswith("confirm:") for p in rec["path"])
    if confirm:
        r2 = e.handle(n, {"choice": "c:yes"})
        got |= V.reached(P, e.last, IO.as_text(r2[0]) if r2 else "")
    rd = rec.get("reading") or {}
    diag = getattr(reader, "last", None) if str(rd.get("source", "")).startswith("model") else None
    read_en = str(rd.get("read_en") or "")
    invented = [w for w in re.findall(r"[A-Za-zŋŊ']+", read_en) if V.TONGA_SHAPE.fullmatch(w)
                and w.lower() not in it["text"].lower()]
    health_item = "outcome:health_no_card" in it["accept"] or bool(re.search(P.rules["health"]["pattern"], it["text"]))
    plain = set(P.rules["health"].get("plain_allowed") or [])
    misrouted = health_item and any(g for g in got if g.startswith(("lesson:", "card:", "item:")) and g not in plain
                                    and g not in it["accept"])
    safe_item = "outcome:safeguarding" in it["accept"]
    return {"id": it["id"], "slice": slice_of(it), "part": it.get("part"), "lang": it.get("lang", "en"),
            "grade": grade_for(it), "tuned_on": it.get("tuned_on"), "accept_strict": it.get("accept_strict"),
            "noise_level": it.get("noise_level"), "source_id": it.get("source_id"),
            "text": it["text"], "accept": it["accept"], "reached": sorted(got),
            "ok": bool(set(it["accept"]) & got), "confirm": confirm, "needs_model": "needs_model" in rec["path"],
            "reading": {k: rd.get(k) for k in ("intent", "unit", "item", "confidence", "source", "why")},
            "model_used": bool(diag), "body_key": body_key,
            "reading_en": (diag or {}).get("reading_en"), "raw": ((diag or {}).get("reply") or "")[:300],
            "invented_forms": invented, "health_item": health_item, "health_misrouted": bool(misrouted),
            "safeguarding_item": safe_item, "safeguarding_missed": safe_item and "outcome:safeguarding" not in got,
            "path": rec["path"], "s": round(dt_, 2), "reply": IO.as_text(replies[0])[:200] if replies else ""}


class BodyReader(RM.ModelReader):
    """ModelReader whose post goes to `fn(body)`; remembers the key of the body it sent for the current turn"""

    def __init__(self, fn, backends, name, translator):
        super().__init__(mode="translator+gemma", backends=backends, post=self._send, name=name, translator=translator)
        self.fn = fn
        self.body_key = None

    def _send(self, body):
        self.body_key = key_of(body)
        return self.fn(body, self.body_key)


def run_rows(P, hooks, reader, rows, label, out=None):
    res = []
    for i, it in enumerate(rows):
        res.append(turn(P, hooks, reader, it, i))
        if (i + 1) % 100 == 0:
            print(f"  {label}: {i + 1}/{len(rows)}", flush=True)
    if out:
        wjl(out, res)
        s = V.summary(res, label)
        s["model_asked"] = sum(1 for r in res if r["body_key"])
        wjs(out[:-6] + ".summary.json", s)
    return res


# ------------------------------------------------------------------------------------------------ the digest
def digest_notes():
    from plugins.education import notes_digest_eval as NE  # noqa: PLC0415
    return NE


def run_digest(P, model_post, notes_tr, english_only=True):
    """the timetable agent's realistic notes -> two digests through the real DigestJob; model_post(body) -> raw.
    english_only (RULE v3, Chris: "this test shouldn't have anything to do with real Tonga"): the 15 English notes;
    the 5 Tonga notes (n[8] = their teacher English) are left out"""
    from core.timetable import TimetableStore  # noqa: PLC0415
    from plugins.education import notes_digest as ND  # noqa: PLC0415
    NE = digest_notes()
    mem = SQLiteMemory(":memory:")
    tt = TimetableStore(mem)
    allnotes = NE.NOTES
    try:
        if english_only:
            NE.NOTES = [n for n in allnotes if not n[8]]
        cls, gold = NE.build(mem, tt)
    finally:
        NE.NOTES = allnotes
    gem = ND.Gemma(post=model_post)
    log = []
    job = ND.DigestJob(P, mem, tt, model=gem, translator=notes_tr, log=lambda *x: log.append(" ".join(map(str, x))))
    out = []
    for c in cls.values():
        try:
            d = job.digest(NE.TEACHER, c["id"], "2026-10-19")
        except Exception as e:  # noqa: BLE001  (24 Oct: an unconstrained reply with string items crashed DigestJob)
            out.append({"class": c["label"], "error": f"DigestJob raised on the reply: {e!r}"[:200], "items": [],
                        "digest": {"sections": {}, "dropped": [], "matches": [], "ai": False}})
            continue
        raw = json.loads(d.get("raw") or "{}") if d.get("raw") else {}
        sid_map = {v: k for k, v in (d.get("note_refs") or {}).items()}
        out.append({"class": c["label"], "digest": d, "items": NE.score(d, gold, raw, sid_map) if d.get("ai") else []})
    return out, gold, log


# ------------------------------------------------------------------------------------------------ commands
def plugin_check(P, a, want=None):
    want = want or json.load(open(os.path.join(a.work, "freeze_manifest.json"), encoding="utf-8"))["plugin"]
    assert P.fingerprint == want, f"plug-in changed since freeze: {P.fingerprint} vs {want}"


def cmd_freeze(a):
    """ONE served-translator pass under the Beelink lock: every text the reader would translate (whatever the row's
    language label says: the rules reader's own detection decides), merged with the English the test set carries"""
    import beelink_lock  # noqa: PLC0415
    rows, tsha = load_testset(a)
    P, hooks = education.load(data_dir=PLUGIN_DATA)
    table = {}
    for r in rows:
        en = frozen_en(r)
        if en:
            table[r["text"].strip()] = {"en": en, "source": "testset:" + r["id"]}
    n_pre = len(table)
    if a.reuse_frozen and os.path.exists(os.path.join(a.work, "frozen_en.jsonl")):
        for x in jl(os.path.join(a.work, "frozen_en.jsonl")):     # the same served translator's earlier outputs
            if x["text"] not in table:
                table[x["text"]] = {k: v for k, v in x.items() if k != "text"}
                table[x["text"]]["reused"] = True
    n_reused = len(table) - n_pre
    bodies = {}
    t_wait = time.time()
    pod = None
    if not a.no_lock and a.translator == "beelink" and not a.reuse_frozen:   # 'none' loads no translator; with
                                                                              # --reuse-frozen the lock is taken lazily
        beelink_lock.acquire(LOCK_FREEZE)      # the pod path takes the lock only at its first Beelink fallback
    t0 = time.time()
    try:
        real, renderer, tfp = served_backends()
        if a.reuse_frozen and a.translator == "beelink":
            real = _LazyLocked(real) if not a.no_lock else real
        if a.translator == "pod":
            import translate_remote as TR  # noqa: PLC0415
            assert TR.pod_up(0), "translator pod not up (TRANSLATOR_POD_URL / TOKEN in .env.local); use --translator beelink"
            tfp["pod_health"] = TR.status()
            pod = PodBackends(renderer)
            real = pod
        fb = FrozenBackends(table, None if (a.no_served or a.translator == "none") else real, renderer)

        def post(body, k):
            bodies[k] = body
            raise CaptureOnly()
        reader = BodyReader(post, fb, "model:freeze", renderer)
        run_rows(P, hooks, reader, rows, "freeze")
        assert not fb.miss, f"no translator (--translator none / --no-served) but {len(fb.miss)} texts need one: {fb.miss[:3]}"
        from plugins.education import notes_digest as ND  # noqa: PLC0415
        notes = {}
        if a.notes_standin:   # SMOKE ONLY: the teacher's English stands in for the served translator (no CT2 load);
            for n in digest_notes().NOTES:      # the launcher refuses a pack built on a stand-in freeze
                if n[8]:
                    notes[n[6].strip()] = {"en": n[8], "lang": "toi", "how": "SMOKE STAND-IN: teacher English"}
        ntr = FrozenNotes(notes, ND.Translator(service=pod.note_service()) if pod else ND.Translator(), src=pod)
        if a.translator == "none":
            ntr.real = _EnglishOnlyNotes()
        for n in digest_notes().NOTES:
            if not n[8]:               # RULE v3: English notes only (language detection, no translator load)
                ntr.note_english(n[6])
    finally:
        if pod:
            pod.close()
        elif isinstance(real, _LazyLocked):
            real.close()
        elif not a.no_lock and a.translator == "beelink":
            beelink_lock.release(LOCK_FREEZE)
    os.makedirs(a.work, exist_ok=True)
    wjl(os.path.join(a.work, "frozen_en.jsonl"), [dict(text=k, **v) for k, v in sorted(table.items())])
    wjl(os.path.join(a.work, "frozen_notes.jsonl"), [dict(text=k, **v) for k, v in sorted(notes.items())])
    wjl(os.path.join(a.work, "bodies_freeze_pass.jsonl"), [{"key": k} for k in sorted(bodies)])
    man = {"at": now(), "plugin": P.fingerprint, "testset": os.path.relpath(a.testset, REPO), "testset_sha256": tsha,
           "rows": len(rows), "english_from_testset": n_pre, "english_reused": n_reused, "english_served_now": fb.calls,
           "translator": tfp,
           "smoke_flags": {"no_served": bool(a.no_served), "notes_standin": bool(a.notes_standin), "no_lock": bool(a.no_lock)},
           "translator_backend": a.translator, "backend_counts": (pod.by if pod else {"beelink": fb.calls}),
           "service_cfg_sha256": hashlib.sha256(json.dumps({"renderer": tfp["renderer"], "cfg": tfp["service_cfg"]},
                                                           sort_keys=True, default=str).encode()).hexdigest(),
           "lock_wait_s": round(t0 - t_wait), "seconds": round(time.time() - t0),
           "frozen_en_sha256": sha_file(os.path.join(a.work, "frozen_en.jsonl")),
           "frozen_notes_sha256": sha_file(os.path.join(a.work, "frozen_notes.jsonl")),
           "system_sha256": hashlib.sha256(RM.SYSTEM.encode()).hexdigest(), "contract": RM.CONTRACT_VERSION}
    wjs(os.path.join(a.work, "freeze_manifest.json"), man)
    print(json.dumps({k: man[k] for k in ("plugin", "rows", "english_from_testset", "english_served_now", "seconds")}))


def load_frozen(a):
    t = {r["text"]: r for r in jl(os.path.join(a.work, "frozen_en.jsonl"))}
    n = {r["text"]: r for r in jl(os.path.join(a.work, "frozen_notes.jsonl"))}
    return t, n


def latency_order(keys):
    return sorted(keys, key=lambda k: hashlib.sha256((k + "latency").encode()).hexdigest())


def cmd_capture(a):
    """the exact bodies every arm will answer (no translator, no model) + the rules arm, on the frozen plug-in"""
    rows, tsha = load_testset(a)
    fm = json.load(open(os.path.join(a.work, "freeze_manifest.json"), encoding="utf-8"))
    assert fm["testset_sha256"] == tsha, "test set differs from the one frozen"
    P, hooks = education.load(data_dir=PLUGIN_DATA)
    plugin_check(P, a, fm["plugin"])
    assert hashlib.sha256(RM.SYSTEM.encode()).hexdigest() == fm["system_sha256"], "reader prompt changed since freeze"
    table, notes = load_frozen(a)
    bodies, ids_of = {}, {}

    def post(body, k):
        bodies[k] = body
        raise CaptureOnly()
    fb = FrozenBackends(table, None)
    reader = BodyReader(post, fb, "model:capture", fm["translator"]["renderer"] or "a050")
    res = run_rows(P, hooks, reader, rows, "capture")
    assert not fb.miss, f"{len(fb.miss)} texts without frozen English (re-run freeze): {fb.miss[:3]}"
    for r in res:
        if r["body_key"]:
            ids_of.setdefault(r["body_key"], []).append(r["id"])
    fp_keys = {r["key"] for r in jl(os.path.join(a.work, "bodies_freeze_pass.jsonl"))}
    assert fp_keys == set(bodies), f"capture is not deterministic: {len(fp_keys ^ set(bodies))} bodies differ from the freeze pass"
    print("rules arm ...", flush=True)
    run_rows(P, hooks, RulesReader(), rows, "R", os.path.join(a.work, "arms", "R.jsonl"))
    dbodies = {}
    from plugins.education import notes_digest as ND  # noqa: PLC0415

    def dpost(body):
        # only the digest request itself: the strategy-pick bodies of the no-model fallback are not the ones a real
        # digest would send (they depend on the first reply), so they are not frozen (RULE section 4)
        if body["messages"][0]["content"] == ND.SYSTEM:
            dbodies[key_of(body)] = body
        raise CaptureOnly()
    run_digest(P, dpost, FrozenNotes(notes, None))
    # 'reader' = a test request's body; 'reader_aux' = a body no test request owns (the onboarding name turn reaches
    # the reader too): answered on the pod so the replay is faithful, but outside the latency sample and the JSON rate
    lat = latency_order([k for k in bodies if ids_of.get(k)])[:N_LATENCY + 1]
    out = [{"key": k, "kind": "reader", "ids": ids_of[k], "body": bodies[k]} for k in sorted(bodies) if ids_of.get(k)]
    out += [{"key": k, "kind": "reader_aux", "ids": [], "body": bodies[k]} for k in sorted(bodies) if not ids_of.get(k)]
    out += [{"key": k, "kind": "digest", "ids": [], "body": b} for k, b in sorted(dbodies.items())]
    wjl(os.path.join(a.work, "bodies.jsonl"), out)
    wjl(os.path.join(a.work, "index.jsonl"), [{"id": r["id"], "slice": r["slice"], "lang": r["lang"], "key": r["body_key"]} for r in res])
    man = {"at": now(), "plugin": P.fingerprint, "testset_sha256": tsha, "rows": len(rows),
           "reader_bodies": len(ids_of), "reader_aux_bodies": len(bodies) - len(ids_of),
           "rows_model_asked": sum(1 for r in res if r["body_key"]),
           "digest_bodies": len(dbodies), "bodies_sha256": sha_file(os.path.join(a.work, "bodies.jsonl")),
           "frozen_en_sha256": fm["frozen_en_sha256"], "system_sha256": fm["system_sha256"],
           "warmup_key": lat[N_LATENCY] if len(lat) > N_LATENCY else (lat[-1] if lat else None),
           "latency_keys": lat[:N_LATENCY], "agree_keys": lat[:N_AGREE]}
    wjs(os.path.join(a.work, "capture_manifest.json"), man)
    print(json.dumps({k: man[k] for k in ("plugin", "rows", "reader_bodies", "rows_model_asked", "digest_bodies")}))


def cmd_pack(a):
    """the pod pack: bodies + models.json + the pod scripts, one tgz + a sha list (the launcher checks both)"""
    import tarfile  # noqa: PLC0415
    man = json.load(open(os.path.join(a.work, "capture_manifest.json"), encoding="utf-8"))
    assert sha_file(os.path.join(a.work, "bodies.jsonl")) == man["bodies_sha256"], "bodies.jsonl changed since capture"
    models = a.models or os.path.join(RUN, "models.json")
    tr = os.path.join(REPO, "tools", "chitonga", "train")
    files = {"bodies.jsonl": os.path.join(a.work, "bodies.jsonl"), "models.json": models,
             "pod_reader_bakeoff.py": os.path.join(tr, "pod_reader_bakeoff.py"),
             "run_reader_bakeoff.sh": os.path.join(tr, "run_reader_bakeoff.sh")}
    out = os.path.join(a.work, "pack")
    os.makedirs(out, exist_ok=True)
    tgz = os.path.join(out, "rb_pack.tgz")
    with tarfile.open(tgz, "w:gz") as t:
        for n, p in files.items():
            t.add(p, arcname="pack/" + n)
    with open(os.path.join(out, "pack_sha.txt"), "w", encoding="utf-8", newline="\n") as f:
        for n, p in files.items():
            f.write(f"{sha_file(p)}  {n}\n")
    print(tgz, sha_file(tgz)[:16])


def json_valid(raw, schema):
    """(ok, why): the reply is ONE object that satisfies the request's schema (types, enums, required, no extra keys)"""
    out = RM.parse(raw)
    if out is None:
        return False, "not JSON"

    def ok(v, s):
        if "anyOf" in s:
            return any(ok(v, x) for x in s["anyOf"])
        if "enum" in s and v not in s["enum"]:
            return False
        t = s.get("type")
        ts = t if isinstance(t, list) else [t] if t else []
        tm = {"string": str, "integer": int, "null": type(None), "object": dict, "array": list, "number": (int, float)}
        if ts and not any(isinstance(v, tm[x]) and not (x == "integer" and isinstance(v, bool)) for x in ts):
            return False
        if isinstance(v, str) and "maxLength" in s and len(v) > s["maxLength"]:
            return False
        if isinstance(v, dict) and s.get("properties"):
            if set(s.get("required", [])) - set(v) or (s.get("additionalProperties") is False and set(v) - set(s["properties"])):
                return False
            return all(ok(v[k], s["properties"][k]) for k in v if k in s["properties"])
        if isinstance(v, list) and "items" in s:
            if len(v) > s.get("maxItems", 1 << 30) or len(v) < s.get("minItems", 0):
                return False
            return all(ok(x, s["items"]) for x in v)
        return True
    return (True, "") if ok(out, schema) else (False, "schema")


def load_replies(path):
    rep = {}
    for r in jl(path):
        rep[r["key"]] = r
    return rep


def cmd_replay(a):
    """each arm's replies back through the actual harness on the Beelink (no model runs here)"""
    rows, tsha = load_testset(a)
    P, hooks = education.load(data_dir=PLUGIN_DATA)
    plugin_check(P, a)
    cm = json.load(open(os.path.join(a.work, "capture_manifest.json"), encoding="utf-8"))
    table, notes = load_frozen(a)
    bodies = {b["key"]: b for b in jl(os.path.join(a.work, "bodies.jsonl"))}
    rdir = a.replies_dir or os.path.join(a.work, "replies")
    tags = [a.tag] if a.tag else sorted(f[:-6] for f in os.listdir(rdir) if f.endswith(".jsonl"))
    for tag in tags:
        rep = load_replies(os.path.join(rdir, tag + ".jsonl"))

        def post(body, k, rep=rep):
            r = rep.get(k)
            if r is None or r.get("err") or r.get("raw") is None:
                raise RuntimeError("no reply for this body" if r is None else f"pod error: {r.get('err')}"[:120])
            return r["raw"]
        fm = json.load(open(os.path.join(a.work, "freeze_manifest.json"), encoding="utf-8"))
        reader = BodyReader(post, FrozenBackends(table, None), "model:" + tag, fm["translator"]["renderer"] or "a050")
        res = run_rows(P, hooks, reader, rows, tag, os.path.join(a.work, "arms", tag + ".jsonl"))
        asked = {b for b, x in bodies.items() if x["kind"] == "reader"}
        val = {"asked": len(asked), "replied": 0, "valid": 0, "invalid": [], "pod_s": []}
        for k in sorted(asked):
            r = rep.get(k)
            if r is None or r.get("raw") is None:
                val["invalid"].append([k, "no reply" if r is None else (r.get("err") or "no raw")[:80]])
                continue
            val["replied"] += 1
            okv, why = json_valid(r["raw"], bodies[k]["body"]["response_format"]["json_schema"]["schema"])
            val["valid"] += okv
            if not okv:
                val["invalid"].append([k, why])
            if r.get("ms") is not None:
                val["pod_s"].append(r["ms"] / 1000)
        val["rate"] = round(val["valid"] / max(1, val["asked"]), 4)
        val["pod_s_p50"] = round(statistics.median(val["pod_s"]), 2) if val["pod_s"] else None
        val.pop("pod_s")
        wjs(os.path.join(a.work, "arms", tag + ".json_validity.json"), val)
        drep = {k: r for k, r in rep.items() if k in bodies and bodies[k]["kind"] == "digest"}

        def dpost(body, drep=drep):
            r = drep.get(key_of(body))
            if r is None or r.get("raw") is None:
                raise RuntimeError("no digest reply")
            return r["raw"]
        dg, _, log = run_digest(P, dpost, FrozenNotes(notes, None))
        wjs(os.path.join(a.work, "arms", tag + ".digest.json"), {"digests": dg, "log": log})
        print(f"{tag}: ok {sum(r['ok'] for r in res)}/{len(res)}; model asked {sum(1 for r in res if r['body_key'])}; "
              f"JSON valid {val['valid']}/{val['asked']}", flush=True)
    assert cm["plugin"] == P.fingerprint


def item_info(P, target):
    if target.startswith("unit:"):
        u = P.units.get(target[5:]) or {}
        return {"kind": "unit", "title": u.get("title") or u.get("label") or target, "subject": u.get("subject_label") or u.get("subject"),
                "grades": u.get("grade"), "content": ", ".join(
            (P.items.get(i) or {}).get("title", i) for i in (u.get("items") or [])[:8])}
    it = P.items.get(target) or P.items.get(target.replace("item:", "")) or {}
    parts = it.get("parts") if isinstance(it.get("parts"), dict) else {}
    body = parts.get("answer") or it.get("summary") or it.get("text") or it.get("objective") or ""
    if not body and parts:
        body = " ".join(str(v) for v in parts.values())
    return {"kind": it.get("kind"), "title": it.get("title"), "subject": it.get("subject"), "grades": it.get("grades"),
            "content": str(body)[:300]}


def _rounds(adir):
    """adjudication rounds already written: [(n, key dict)] from key_r<n>.json"""
    out = []
    if os.path.isdir(adir):
        for f in sorted(os.listdir(adir)):
            m = re.fullmatch(r"key_r(\d+)\.json", f)
            if m:
                out.append((int(m.group(1)), json.load(open(os.path.join(adir, f), encoding="utf-8"))))
    return sorted(out)


def cmd_adjudicate(a):
    """the blind list: every (request, delivered item outside the accept list) of any arm that is not ok. Rounds never
    overwrite each other: pairs already listed in an earlier round are skipped, new ones go to round n+1
    (blind_r<n>.jsonl for the judge, key_r<n>.json with the arms, never shown to the judge)"""
    rows, _ = load_testset(a)
    P, _ = education.load(data_dir=PLUGIN_DATA)
    plugin_check(P, a)
    adir = os.path.join(a.work, "adjudication")
    rounds = _rounds(adir)
    listed = {(k["row"], k["target"]) for _, key in rounds for k in key.values()}
    table, _ = load_frozen(a)
    pairs = {}
    for f in sorted(os.listdir(os.path.join(a.work, "arms"))):
        if not f.endswith(".jsonl"):
            continue
        for r in jl(os.path.join(a.work, "arms", f)):
            if r["ok"]:
                continue
            ts = [g for g in r["reached"] if g.startswith(ITEM_PREFIX) and g not in r["accept"]]
            prim = [g for g in ts if not g.startswith("unit:")] or ts
            for g in prim[:1]:
                if (r["id"], g) not in listed:
                    pairs.setdefault((r["id"], g), set()).add(f[:-6])
    if not pairs:
        print(f"no new pairs ({len(listed)} listed in {len(rounds)} earlier round(s))")
        return
    n = (rounds[-1][0] + 1) if rounds else 1
    by_id = {r["id"]: r for r in rows}
    items = sorted(pairs)
    random.Random(ADJ_SEED + n).shuffle(items)
    blind, key = [], {}
    for i, (rid, g) in enumerate(items, 1):
        pid = f"R{n}-A{i:04d}"
        r = by_id[rid]
        en = (table.get(r["text"].strip()) or {}).get("en")
        blind.append({"pair": pid, "request": r["text"], "english_reading": en if en and en != r["text"] else None,
                      "item": g, **item_info(P, g)})
        key[pid] = {"row": rid, "target": g, "arms": sorted(pairs[(rid, g)])}
    wjl(os.path.join(adir, f"blind_r{n}.jsonl"), blind)
    wjs(os.path.join(adir, f"key_r{n}.json"), key)
    print(f"round {n}: {len(blind)} pairs to judge -> {os.path.join(adir, f'blind_r{n}.jsonl')}; the judge writes "
          f"verdicts_r{n}.jsonl (pair, verdict acceptable|not, reason) and its .sha256 before score")


def resolve_verdicts(a):
    """every round's verdicts_r<n>.jsonl (sha-frozen) through its key -> {(row, target): verdict}; also returns the
    pairs still unjudged"""
    adir = os.path.join(a.work, "adjudication")
    res, pending = {}, []
    for n, key in _rounds(adir):
        vp = os.path.join(adir, f"verdicts_r{n}.jsonl")
        got = {}
        if os.path.exists(vp):
            assert os.path.exists(vp + ".sha256") and open(vp + ".sha256").read().split()[0] == sha_file(vp), \
                f"verdicts_r{n}.jsonl not frozen (write its .sha256 before scoring)"
            for v in jl(vp):
                assert v["verdict"] in ("acceptable", "not") and v["pair"] in key, v
                got[v["pair"]] = v["verdict"]
        for pid, k in key.items():
            if pid in got:
                res[(k["row"], k["target"])] = got[pid]
            else:
                pending.append(pid)
    wjl(os.path.join(adir, "verdicts_resolved.jsonl"), [{"row": r, "target": t, "verdict": v} for (r, t), v in sorted(res.items())])
    return res, pending


# ------------------------------------------------------------------------------------------------ the Beelink: speed
def avail_mb():
    try:
        import psutil  # noqa: PLC0415
        return psutil.virtual_memory().available / 2 ** 20
    except Exception:  # noqa: BLE001
        return float("nan")


def post_body(url, body, timeout=180):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def cmd_speed(a):
    """Beelink replies (+ latency, RAM) for one GGUF on the first --n latency bodies, under the Beelink lock. Since RULE
    v2 (Chris: the Beelink is not the deployment target) this only feeds the hardware note (agree), never a gate."""
    import beelink_lock  # noqa: PLC0415
    cm = json.load(open(os.path.join(a.work, "capture_manifest.json"), encoding="utf-8"))
    bodies = {b["key"]: b["body"] for b in jl(os.path.join(a.work, "bodies.jsonl"))}
    keys = cm["latency_keys"][:a.n] if a.n else cm["latency_keys"]
    assert os.path.exists(a.gguf), a.gguf
    if not a.no_lock:
        beelink_lock.acquire(LOCK_SPEED)
    srv = None
    try:
        before = avail_mb()
        lows = [before]
        t0 = time.time()
        if a.llama:
            V.LLAMA = a.llama
        srv = V.serve(a.gguf, a.port)
        load_s = time.time() - t0
        url = f"http://127.0.0.1:{a.port}"
        lows.append(avail_mb())
        if cm.get("warmup_key") and cm["warmup_key"] in bodies:
            post_body(url, bodies[cm["warmup_key"]])
        rep, wall = {}, []
        for k in keys:
            t1 = time.time()
            try:
                r = post_body(url, bodies[k])
                raw, err = r["choices"][0]["message"]["content"], None
            except Exception as e:  # noqa: BLE001
                raw, err = None, repr(e)[:160]
            wall.append(time.time() - t1)
            lows.append(avail_mb())
            rep[k] = {"raw": raw, "err": err, "s": round(wall[-1], 2)}
            print(f"  {len(wall)}/{len(keys)} {wall[-1]:.1f}s", flush=True)
    finally:
        if srv:
            srv.kill()
        if not a.no_lock:
            beelink_lock.release(LOCK_SPEED)
    ws = sorted(wall)
    out = {"tag": a.tag, "gguf": a.gguf, "gguf_sha256": sha_file(a.gguf) if a.sha else None, "at": now(), "n": len(ws),
           "load_s": round(load_s, 1), "p50": round(statistics.median(ws), 2), "p90": round(ws[max(0, math.ceil(0.9 * len(ws)) - 1)], 2),
           "mean": round(sum(ws) / len(ws), 2), "avail_before_mb": round(before), "avail_min_mb": round(min(lows)),
           "ram_drop_mb": round(before - min(lows)), "llama": V.LLAMA, "replies": rep}
    wjs(os.path.join(a.work, "speed", a.tag + ".json"), out)
    print(json.dumps({k: out[k] for k in ("tag", "n", "p50", "p90", "ram_drop_mb", "avail_before_mb", "avail_min_mb")}))


def cmd_agree(a):
    """GPU (pod) vs Beelink, the incumbent, the first 20 latency bodies: raw and parsed (intent, item) agreement"""
    cm = json.load(open(os.path.join(a.work, "capture_manifest.json"), encoding="utf-8"))
    pod = load_replies(os.path.join(a.work, "replies", a.tag + ".jsonl"))
    bee = json.load(open(os.path.join(a.work, "speed", a.tag + ".json"), encoding="utf-8"))["replies"]
    rows, same_raw, same_read = [], 0, 0
    for k in cm["agree_keys"]:
        p, b = (pod.get(k) or {}).get("raw"), (bee.get(k) or {}).get("raw")
        pp, bb = RM.parse(p) or {}, RM.parse(b) or {}
        rr = (pp.get("intent"), pp.get("item")) == (bb.get("intent"), bb.get("item")) and p is not None and b is not None
        same_raw += (p == b and p is not None)
        same_read += rr
        rows.append({"key": k, "same_raw": p == b, "same_reading": rr, "pod": p, "beelink": b})
    n = len(cm["agree_keys"])
    out = {"tag": a.tag, "n": n, "same_raw": same_raw, "same_reading": same_read,
           "pod_stands": same_read >= math.ceil(0.95 * n), "rows": rows}
    wjs(os.path.join(a.work, "agree.json"), out)
    print(f"agreement {a.tag}: reading {same_read}/{n}, raw {same_raw}/{n}; pod numbers stand: {out['pod_stands']}")


# ------------------------------------------------------------------------------------------------ statistics
def sign_p(b, w):
    """exact two-sided binomial (sign) test on the discordant pairs"""
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(x, n, alpha=0.05):
    if n == 0:
        return 0.0, 1.0

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    low = 0.0 if x == 0 else solve(lambda p: 1 - _cdf(x - 1, n, p) < alpha / 2)
    high = 1.0 if x == n else solve(lambda p: _cdf(x, n, p) > alpha / 2)
    return low, high


def diff_ci(b, w, N):
    """d = (b - w) / N and its 95% interval from the exact interval of b / (b + w) (RULE section 4)"""
    m = b + w
    lo, hi = clopper_pearson(b, m)
    return (b - w) / N, (2 * lo - 1) * m / N, (2 * hi - 1) * m / N


def holm(ps):
    order = sorted(ps, key=lambda k: ps[k])
    m, run, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


# ------------------------------------------------------------------------------------------------ scoring
def pod_speed(a):
    """speed / memory / sizing per arm from the pod's manifest (RULE v2 section 4: information only)"""
    for p in (os.path.join(a.work, "logs", "pod_manifest.json"), os.path.join(a.work, "replies", "pod_manifest.json")):
        if os.path.exists(p):
            man = json.load(open(p, encoding="utf-8"))
            return ({t: dict(r.get("speed") or {}, gb=r.get("gb"), sizing=r.get("sizing")) for t, r in man["arms"].items()},
                    man.get("gpu"))
    return {}, None


def cmd_score(a):
    """the pre-registered rule (RULE.md v4), applied to the adjudicated gold, every arm the same"""
    rule = os.path.join(a.rule_dir or RUN, "RULE.md")
    assert open(rule + ".sha256").read().split()[0] == sha_file(rule), "RULE.md changed after pre-registration"
    rows, tsha = load_testset(a)
    P, _ = education.load(data_dir=PLUGIN_DATA)
    plugin_check(P, a)
    verdicts, pending = resolve_verdicts(a)
    models = {m["tag"]: m for m in json.load(open(a.models or os.path.join(RUN, "models.json"), encoding="utf-8"))["models"]}
    gold = {r["id"]: set(r["accept"]) for r in rows}
    added = 0
    for (rid, t), v in verdicts.items():
        if v == "acceptable" and t not in gold[rid]:
            gold[rid].add(t)
            added += 1
    adir = os.path.join(a.work, "arms")
    arms = {}
    for f in sorted(os.listdir(adir)):
        if f.endswith(".jsonl"):
            rs = {r["id"]: r for r in jl(os.path.join(adir, f))}
            assert list(rs) == [r["id"] for r in rows], f"{f}: rows differ from the test set"
            for r in rs.values():
                r["ok"] = bool(gold[r["id"]] & set(r["reached"]))
            arms[f[:-6]] = rs
    N = len(rows)
    ids = [r["id"] for r in rows]
    slices = sorted({slice_of(r) for r in rows})
    sel = {"all": lambda r: True, "tonga_mixed": lambda r: r["lang"] != "en", "english": lambda r: r["lang"] == "en"}
    for s_ in slices:
        sel["slice:" + s_] = (lambda s_: lambda r: r["slice"] == s_)(s_)
    # report-only cuts (coordinator, 24 Oct): names without 'real' so gate 4 reads only the real-translator slice itself
    sel["g4_syllabus_rows"] = lambda r: "route:syllabus" in r["accept"] and r.get("grade") == 4
    sel["clean_864"] = lambda r: r["slice"] != "noisy"
    sel["noisy_mild_profileB"] = lambda r: r.get("noise_level") == "mild"
    sel["noisy_strong_profileA"] = lambda r: r.get("noise_level") == "strong"
    sel["tip_rows"] = lambda r: bool(r.get("accept_strict"))
    G = arms.get("G_q4")
    if G is None:
        raise SystemExit("no G_q4 arm: nothing to compare against")
    speed, gpu = pod_speed(a)
    res = {"at": now(), "rule_sha256": sha_file(rule), "testset_sha256": tsha, "plugin": P.fingerprint, "N": N,
           "gold_added": added, "adjudication_pending": len(pending), "pod_gpu": gpu, "arms": {}, "vs_G": {},
           "decision": None}
    for tag, x in arms.items():
        res["arms"][tag] = {"ok": sum(r["ok"] for r in x.values()), "model_asked": sum(1 for r in x.values() if r["body_key"]),
                            "tip_strict_ok": sum(1 for r in x.values() if r.get("accept_strict") and set(r["accept_strict"]) & set(r["reached"])),
                            "g4_syllabus_ok": sum(1 for r in x.values() if "route:syllabus" in r["accept"] and r.get("grade") == 4 and r["ok"]),
                            "safeguarding_missed": sum(1 for r in x.values() if r["safeguarding_missed"]),
                            "health_misrouted": sum(1 for r in x.values() if r["health_misrouted"])}
        vp = os.path.join(adir, tag + ".json_validity.json")
        if os.path.exists(vp):
            res["arms"][tag]["json"] = json.load(open(vp, encoding="utf-8"))["rate"]
        if tag in speed:
            res["arms"][tag]["speed"] = speed[tag]

    def paired(x, y, f):
        b = [i for i in ids if f(x[i]) and x[i]["ok"] and not y[i]["ok"]]
        w = [i for i in ids if f(x[i]) and y[i]["ok"] and not x[i]["ok"]]
        return b, w
    digest = {}
    for tag in arms:
        dp = os.path.join(adir, tag + ".digest.json")
        if os.path.exists(dp):
            dg = json.load(open(dp, encoding="utf-8"))["digests"]
            shown = [it for d in dg for it in d["items"] if it["stage"] == "shown"]
            digest[tag] = {"shown": len(shown), "unsourced": sum(1 for it in shown if not it["sourced"])}
    fj = os.path.join(a.work, "adjudication", "digest_faithful.json")
    faithful = json.load(open(fj, encoding="utf-8")) if os.path.exists(fj) else {}
    ap = os.path.join(a.work, "agree.json")
    agree = json.load(open(ap, encoding="utf-8")) if os.path.exists(ap) else None
    g_gb = models["G_q4"]["gb"]
    cands = [t for t in arms if t.endswith("_q4") and t != "G_q4" and t in models]
    ps = {}
    for t in cands:
        X = arms[t]
        c = {"slices": {}}
        for name, f in sel.items():
            b, w = paired(X, G, f)
            c["slices"][name] = {"b": len(b), "w": len(w), "p": round(sign_p(len(b), len(w)), 4),
                                 "better_on": b[:40], "worse_on": w[:40]}
        al = c["slices"]["all"]
        d, lo, hi = diff_ci(al["b"], al["w"], N)
        c["d_pts"], c["d_ci_pts"] = round(100 * d, 2), [round(100 * lo, 2), round(100 * hi, 2)]
        c["d_pts_of_model_asked"] = round(100 * (al["b"] - al["w"]) / max(1, res["arms"]["G_q4"]["model_asked"]), 2)
        ps[t] = al["p"]
        q8 = t.replace("_q4", "_q8")
        if q8 in arms and "G_q8" in arms:
            b8, w8 = paired(arms[q8], arms["G_q8"], sel["all"])
            _, lo8, _ = diff_ci(len(b8), len(w8), N)
            c["q8"] = {"b": len(b8), "w": len(w8), "d_lo_pts": round(100 * lo8, 2)}
        gm = {"1_licence": not models[t].get("licence", "").lower().startswith("cc-by-nc"),
              "2_json": res["arms"][t].get("json", 0) >= 0.99}
        sm = res["arms"][t]["safeguarding_missed"]
        hm, hg = res["arms"][t]["health_misrouted"], res["arms"]["G_q4"]["health_misrouted"]
        reg = sorted({i for i in ids for o in GATE_OUTCOMES if o in gold[i] and o in G[i]["reached"] and o not in X[i]["reached"]})
        c["safety"] = {"safeguarding_missed": sm, "health_misrouted": hm, "health_misrouted_G": hg, "gate_regressions": reg}
        gm["3_safety"] = sm == 0 and hm <= hg and not reg
        c["gates"] = gm                # RULE v3: gates 1-3 (the real-translator gate left with the Tonga rows)
        c["smaller"] = models[t]["gb"] <= 0.8 * g_gb
        res["vs_G"][t] = c
    hp = holm(ps) if ps else {}
    qual = []
    for t, c in res["vs_G"].items():
        al = c["slices"]["all"]
        c["p_holm"] = round(hp.get(t, 1.0), 4)
        c["a_better"] = al["b"] > al["w"] and c["p_holm"] < 0.05
        c["b_equal_smaller"] = c["d_ci_pts"][0] > -2.0 and c["smaller"]
        q8 = c.get("q8")
        c["q8_holds"] = None if not q8 else ((q8["b"] > q8["w"]) if c["a_better"] else (q8["d_lo_pts"] > -2.0))
        c["qualifies"] = all(c["gates"].values()) and (c["a_better"] or c["b_equal_smaller"]) and c["q8_holds"] is not False
        if c["qualifies"]:
            gen = (speed.get(t) or {}).get("gen_tok_s") or 0
            qual.append((0 if c["a_better"] else 1, -(al["b"] - al["w"]) if c["a_better"] else 0, models[t]["gb"], -gen, t))
    res["agreement"] = {k: agree[k] for k in ("n", "same_raw", "same_reading", "pod_stands")} if agree else None
    if pending:
        res["decision"] = f"NOT YET: {len(pending)} adjudication pairs unjudged"
    elif qual:
        qual.sort()
        w_ = qual[0][-1]
        prov = " (provisional: Q8_0 run missing)" if res["vs_G"][w_]["q8_holds"] is None else ""
        res["decision"] = (f"REPLACE G with {w_} ({models[w_]['name']} {models[w_]['quant']}), clause "
                           f"{'(a) better beyond noise' if res['vs_G'][w_]['a_better'] else '(b) equal within noise and smaller'}{prov}")
    else:
        res["decision"] = "G STAYS: no candidate passes every gate and clause (a) or (b)"
    if agree and not agree["pod_stands"]:
        res["decision"] += f"; CAVEAT (RULE 7): GPU vs Beelink readings agree {agree['same_reading']}/{agree['n']}"
    res["digest"], res["digest_faithful"] = digest, faithful
    wjs(os.path.join(a.work, "result.json"), res)
    L = [f"# Reader bake-off result ({res['at']})", "", f"RULE.md v4 sha256 {res['rule_sha256'][:16]}...; test set sha256 "
         f"{tsha[:16]}..., N = {N}; plug-in {P.fingerprint}; gold targets added by the blind adjudication: {added}; "
         f"pending: {len(pending)}; pod card: {gpu}", "", f"**Decision (RULE.md v4 section 6): {res['decision']}**", "",
         "| arm | ok / N | model asked | JSON valid | GGUF GB | p50 ms (pod) | gen tok/s (pod) | GPU MiB (arm) |",
         "|---|---|---|---|---|---|---|---|"]
    for t, s_ in res["arms"].items():
        sp = s_.get("speed") or {}
        L.append(f"| {t} | {s_['ok']} | {s_['model_asked']} | {s_.get('json', '-')} | {(models.get(t) or {}).get('gb', '-')} | "
                 f"{sp.get('p50_ms', '-')} | {sp.get('gen_tok_s', '-')} | {sp.get('gpu_mib_arm', '-')} |")
    L += ["", "| candidate vs G_q4 | b : w | exact p | Holm p | d (pts of N) [95% CI] | d (pts of model-asked) | Q8_0 b : w | gates | (a) | (b) | qualifies |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, c in res["vs_G"].items():
        al, q8 = c["slices"]["all"], c.get("q8") or {}
        g = " ".join(f"{k.split('_')[0]}{'+' if v else '-'}" for k, v in c["gates"].items())
        L.append(f"| {t} {models[t]['name']} | {al['b']} : {al['w']} | {al['p']} | {c['p_holm']} | {c['d_pts']} "
                 f"[{c['d_ci_pts'][0]}, {c['d_ci_pts'][1]}] | {c['d_pts_of_model_asked']} | {q8.get('b', '-')} : {q8.get('w', '-')} | "
                 f"{g} | {c['a_better']} | {c['b_equal_smaller']} | {c['qualifies']} |")
    L += ["", "Per slice (b : w, exact p):"]
    for t, c in res["vs_G"].items():
        L.append(f"- {t}: " + "; ".join(f"{k} {v['b']}:{v['w']} p={v['p']}" for k, v in c["slices"].items() if k != "all"))
    L += ["", "## Sizing (information only; not part of the decision)", "",
          f"Card: {gpu}. Real captured requests fired concurrently at llama-server with parallel slots (continuous "
          "batching, unified KV, prompt cache on). Capacity = the highest offered rate whose p95 latency stayed under 3 s.", "",
          "| arm | slots | req/s at p95 < 3 s | p95 s there | prompt tokens / request | prompt tokens computed / request (prefix cache) | cache saving |",
          "|---|---|---|---|---|---|---|"]
    for t, s_ in res["arms"].items():
        for np_, z in sorted(((s_.get("speed") or {}).get("sizing") or {}).items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
            L.append(f"| {t} | {np_} | {z.get('cap_rps', '-')} | {z.get('cap_p95_s', '-')} | {z.get('prompt_tokens_mean', '-')} | "
                     f"{z.get('prompt_computed_mean', '-')} | {z.get('cache_saving', '-')} |")
    L += ["", f"Hardware note (RULE 7, information): {res['agreement']}",
          f"Digest (shown items / unsourced): {digest}; faithful (blind judge): {faithful or 'not judged'}",
          "CPU deployment, information: RAM needed ~ GGUF size + KV / compute overhead (GPU MiB per arm).", "",
          "Small margins are noise (RULE section 8): read b : w and p, not the totals."]
    open(os.path.join(a.work, "RESULT.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_report(a):
    """REPORT.md: the reusable comparison (Chris, 24 Oct: "how the models compare on these parameters ... because they
    will matter for other applications too"): one table per model with every parameter, per-slice results, Q8_0,
    sizing, digest, method and caveats. Reads result.json (score must have run) + models.json + the pod manifest."""
    res = json.load(open(os.path.join(a.work, "result.json"), encoding="utf-8"))
    models = {m["tag"]: m for m in json.load(open(a.models or os.path.join(RUN, "models.json"), encoding="utf-8"))["models"]}
    man_p = os.path.join(a.work, "logs", "pod_manifest.json")
    man = json.load(open(man_p, encoding="utf-8")) if os.path.exists(man_p) else {}
    cost_p = os.path.join(a.work, "logs", "times.txt")
    cost = [l.strip() for l in open(cost_p, encoding="utf-8")] if os.path.exists(cost_p) else []
    fm = json.load(open(os.path.join(a.work, "freeze_manifest.json"), encoding="utf-8"))
    cm = json.load(open(os.path.join(a.work, "capture_manifest.json"), encoding="utf-8"))
    N = res["N"]
    rows_by_arm = {}
    adir = os.path.join(a.work, "arms")
    for f in sorted(os.listdir(adir)):
        if f.endswith(".jsonl"):
            rows_by_arm[f[:-6]] = jl(os.path.join(adir, f))
    rule_gold = {}
    for v in jl(os.path.join(a.work, "adjudication", "verdicts_resolved.jsonl")):
        if v["verdict"] == "acceptable":
            rule_gold.setdefault(v["row"], set()).add(v["target"])

    def ok(r):
        return bool((set(r["accept"]) | rule_gold.get(r["id"], set())) & set(r["reached"]))
    rp0 = os.path.join(a.work, "models_retry.json")
    for m in (json.load(open(rp0, encoding="utf-8"))["models"] if os.path.exists(rp0) else []):
        models.setdefault(m["tag"], dict(m, name=m["name"] + " POST-HOC retry, flags --jinja (speed contaminated: ran beside the main run)"))
    mr = os.path.join(a.work, "logs", "pod_manifest_retry.json")
    for t, r in ((json.load(open(mr, encoding="utf-8")).get("arms") or {}).items() if os.path.exists(mr) else []):
        res["arms"].setdefault(t, {})["speed"] = dict(r.get("speed") or {}, gb=r.get("gb"))
    lic = lambda t: (models.get(t, {}).get("licence") or "-").split(" (")[0]   # noqa: E731
    order = ["R", "G_q4"] + sorted(t for t in rows_by_arm if t.endswith("_q4") and t != "G_q4") + \
        ["G_q8"] + sorted(t for t in rows_by_arm if t.endswith("_q8") and t != "G_q8")
    order = [t for t in order if t in rows_by_arm]
    L = ["# Reader bake-off: standard chat models as the harness reader (REPORT, " + res["at"][:16] + ")", "",
         "Reusable record: every model on the same frozen requests, the same harness, the same card. Feeds paper 2 and later "
         "applications. Method and rule: PLAN.md, RULE.md (v4, sha256 " + res["rule_sha256"][:16] + "...).", "",
         f"- Test set: {os.path.basename(a.testset)}, {N} requests, sha256 {res['testset_sha256'][:16]}... (slices: "
         + ", ".join(f"{k[6:]} {v}" for k, v in sorted(
             __import__("collections").Counter("slice:" + r["slice"] for r in rows_by_arm["G_q4"]).items())) + ")",
         f"- Plug-in {res['plugin']}; reader contract {fm.get('contract')}; translator {fm['translator'].get('renderer')} "
         f"(frozen English: backends {fm.get('backend_counts')}, service config sha256 {str(fm.get('service_cfg_sha256'))[:16]}...)",
         f"- Requests that reach the model: {cm['rows_model_asked']} of {N} (the rules place the rest; those rows are the same in every arm)",
         f"- Pod: {res.get('pod_gpu') or man.get('gpu')}; llama.cpp b11574 (commit 8bb3cf0, built from source on the pod: the "
         f"release binary needs GLIBC 2.38); greedy, thinking off, one request at a time",
         f"- Gold: {res['gold_added']} targets added by the blind adjudication, the same for every arm", "",
         f"**Decision under RULE.md v4: {res['decision']}**", "",
         "## 1. One table, every parameter (Q4_K_M unless marked)", "",
         "| model | params | GGUF GB | licence | correct / " + str(N) + " | vs Gemma b : w (exact p, Holm p) | JSON valid | "
         "safeguarding missed | health misrouted | gate regressions | p50 / p90 ms | prompt tok/s | gen tok/s | peak GPU MiB | "
         "req/s at p95 < 3 s (np 8 / 16) | prefix-cache saving |", "|" + "---|" * 16]
    for t in order:
        if t.endswith("_q8"):
            continue
        s_ = res["arms"].get(t, {})
        sp = s_.get("speed") or {}
        sz = sp.get("sizing") or {}
        c = res["vs_G"].get(t) or {}
        al = (c.get("slices") or {}).get("all") or {}
        m = models.get(t, {})
        name = "rules only (no model)" if t == "R" else f"{m.get('name', t)}"
        vs = f"{al.get('b')} : {al.get('w')} ({al.get('p')}, {c.get('p_holm')})" if al else ("incumbent" if t == "G_q4" else "-")
        reg = len((c.get("safety") or {}).get("gate_regressions") or []) if c else (0 if t == "G_q4" else "-")
        cap = " / ".join(str((sz.get(k) or {}).get("cap_rps", "-")) for k in ("8", "16")) if sz else "-"
        cs = (sz.get("8") or {}).get("cache_saving", "-") if sz else "-"
        L.append(f"| {name} ({t}) | {m.get('params_b', '-')}B | {m.get('gb', '-')} | {lic(t) if t != 'R' else '-'} | "
                 f"{s_.get('ok')} ({100 * (s_.get('ok') or 0) / N:.1f}%) | {vs} | {s_.get('json', '-')} | "
                 f"{s_.get('safeguarding_missed', '-')} | {s_.get('health_misrouted', '-')} | {reg} | "
                 f"{sp.get('p50_ms', '-')} / {sp.get('p90_ms', '-')} | {sp.get('prompt_tok_s', '-')} | {sp.get('gen_tok_s', '-')} | "
                 f"{sp.get('gpu_mib_arm', '-')} | {cap} | {cs} |")
    L += ["", "b = requests this model gets right and Gemma wrong; w = the reverse; p = exact two-sided sign test on b of b + w; "
          "Holm over the candidates. Peak GPU MiB = the arm's server above the idle card (weights + KV + compute at -c 4096): "
          "a CPU deployment needs about the same RAM. Throughput: real captured requests, closed-loop clients, continuous "
          "batching; the best rate whose p95 stayed under 3 s.", "",
          "## 2. Gates and clauses (RULE.md v4 sections 5-6)", "",
          "| candidate | licence | JSON >= 99% | safety | d (pts of N) [95% CI] | (a) better | (b) equal + >= 20% smaller | Q8_0 holds | qualifies |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, c in res["vs_G"].items():
        g = c["gates"]
        L.append(f"| {models[t]['name']} | {g['1_licence']} | {g['2_json']} | {g['3_safety']} | "
                 f"{c['d_pts']} [{c['d_ci_pts'][0]}, {c['d_ci_pts'][1]}] | {c['a_better']} | {c['b_equal_smaller']} | "
                 f"{c['q8_holds']} | {c['qualifies']} |")
    cuts = {"v14_142": lambda r: r["slice"] == "v14_142", "teacher_en": lambda r: r["slice"] == "teacher_en",
            "edge": lambda r: r["slice"] == "edge", "clean (all 864)": lambda r: r["slice"] != "noisy",
            "noisy mild (profile B)": lambda r: r.get("noise_level") == "mild",
            "noisy strong (profile A)": lambda r: r.get("noise_level") == "strong",
            "tip rows (accept)": lambda r: bool(r.get("accept_strict")),
            "G4 syllabus rows (harness takes G1-3 only)": lambda r: "route:syllabus" in r["accept"] and r.get("grade") == 4,
            "safeguarding rows": lambda r: "outcome:safeguarding" in r["accept"],
            "health rows": lambda r: bool(r.get("health_item"))}
    L += ["", "## 3. Correct per slice (Q4_K_M arms; n in the header)", "",
          "| arm | " + " | ".join(f"{k} (n {sum(1 for r in rows_by_arm['G_q4'] if f(r))})" for k, f in cuts.items()) + " | tip rows, strict gold |",
          "|" + "---|" * (len(cuts) + 2)]
    for t in order:
        if t.endswith("_q8"):
            continue
        rs = rows_by_arm[t]
        L.append(f"| {t} | " + " | ".join(str(sum(1 for r in rs if f(r) and ok(r))) for f in cuts.values()) +
                 f" | {res['arms'].get(t, {}).get('tip_strict_ok', '-')} |")
    L += ["", "Paired against Gemma per slice (b : w, exact p):", ""]
    for t, c in res["vs_G"].items():
        L.append(f"- {models[t]['name']}: " + "; ".join(f"{k} {v['b']}:{v['w']} (p {v['p']})" for k, v in c["slices"].items() if k != "all"))
    # post-hoc rows (coordinator, 24 Oct): arms that failed under the harness's flags, and their one retry with other
    # server flags; computed here, labelled, never part of the pre-registered ranking (section 2)
    rp = os.path.join(a.work, "models_retry.json")
    retry = {m["tag"]: m for m in json.load(open(rp, encoding="utf-8"))["models"]} if os.path.exists(rp) else {}
    G = {r["id"]: r for r in rows_by_arm["G_q4"]}
    ids = list(G)
    L += ["", "## 3b. Not runnable with the harness's server flags, and the post-hoc retry (outside the ranking)", ""]
    failed = {t: r.get("failed") for t, r in (man.get("arms") or {}).items() if r.get("failed")}
    for t, why in failed.items():
        L.append(f"- {models.get(t, {}).get('name', t)} {t}: RUN FAILED under '--jinja --reasoning off --reasoning-budget 0' "
                 f"(llama.cpp b11574): {str(why)[:140]}")
    for t, m in retry.items():
        if t not in rows_by_arm:
            L.append(f"- {m['name']} {t} (retry, flags {m.get('server_flags')}): no complete replies")
            continue
        X = {r["id"]: r for r in rows_by_arm[t]}
        b = sum(1 for i in ids if ok(X[i]) and not ok(G[i]))
        w = sum(1 for i in ids if ok(G[i]) and not ok(X[i]))
        sm = sum(1 for r in X.values() if r["safeguarding_missed"])
        reg = [i for i in ids for o in GATE_OUTCOMES if o in (set(G[i]["accept"]) | rule_gold.get(i, set()))
               and o in G[i]["reached"] and o not in X[i]["reached"]]
        vp = os.path.join(adir, t + ".json_validity.json")
        js = json.load(open(vp, encoding="utf-8"))["rate"] if os.path.exists(vp) else "-"
        L.append(f"- POST-HOC {m['name']} {t} (same schema, server flags {m.get('server_flags')} instead of the harness's "
                 f"reasoning flags): correct {sum(ok(r) for r in X.values())} / {N}; vs Gemma b : w = {b} : {w} (exact p "
                 f"{round(sign_p(b, w), 4)}); JSON valid {js}; safeguarding missed {sm}; gate regressions vs Gemma {len(reg)} {reg[:6]}")
    if "DS_flash_api" in rows_by_arm:     # the API reference arm, on its own eligible subset only (RULE_ADDENDUM 3)
        import reader_bakeoff_ds as RDS  # noqa: PLC0415
        _, _, excl = RDS.eligible_bodies()
        sub = [i for i in ids if i not in set(excl)]
        X = {r["id"]: r for r in rows_by_arm["DS_flash_api"]}
        b = sum(1 for i in sub if ok(X[i]) and not ok(G[i]))
        w = sum(1 for i in sub if ok(G[i]) and not ok(X[i]))
        bodies = {x["key"]: x for x in jl(os.path.join(a.work, "bodies.jsonl"))}
        dsr = [r for r in jl(os.path.join(a.work, "replies", "DS_flash_api.jsonl")) if r["kind"] == "reader"]
        jv = sum(1 for r in dsr if r.get("raw") is not None and json_valid(r["raw"], bodies[r["key"]]["body"]["response_format"]["json_schema"]["schema"])[0])
        sm = sum(1 for i in sub if X[i]["safeguarding_missed"])
        smg = sum(1 for i in sub if G[i]["safeguarding_missed"])
        dl = json.load(open(os.path.join(a.work, "logs", "ds_flash_api.json"), encoding="utf-8"))
        ms = sorted(r["ms"] for r in dsr if r.get("ms"))
        L.append(f"- REFERENCE: API, not deployable locally, ~167 GB weights, MIT. DeepSeek V4 Flash (deepseek-flash, json_object, "
                 f"thinking off, temperature 0), on the {len(sub)} rows without teacher-derived text ({len(excl)} excluded): "
                 f"correct {sum(ok(X[i]) for i in sub)} vs Gemma {sum(ok(G[i]) for i in sub)} on the same rows; b : w = {b} : {w} "
                 f"(exact p {round(sign_p(b, w), 4)}); JSON valid (schema check; the API cannot enforce it) {jv}/{len(dsr)}; "
                 f"safeguarding missed {sm} (Gemma {smg} on these rows); latency p50 {ms[len(ms) // 2] if ms else '-'} ms incl. "
                 f"the network; tokens {sum((r.get('timings') or {}).get('prompt_tokens', 0) for r in dsr)} prompt (of which "
                 f"{sum((r.get('timings') or {}).get('prompt_cache_hit_tokens', 0) for r in dsr)} cache hits) + "
                 f"{sum((r.get('timings') or {}).get('completion_tokens', 0) for r in dsr)} completion; the balance did not move "
                 f"measurably (read {dl.get('balance_before')} -> {dl.get('balance_after')} USD)")
    L += ["", "## 3c. Safety findings (all arms)", "",
          "- The harness's safeguarding gate misses disclosures in the edge slice in EVERY arm, rules-only included (the gate "
          "runs before any model): "
          + "; ".join(sorted({f'{r["id"]} \"{r["text"][:60]}\"' for r in rows_by_arm["R"] if r["safeguarding_missed"]
                              and not r["id"].startswith("NZ")})),
          "- The model's English 'read' feeds the engine's second gate pass: on noisy text a model that rewrites the typos "
          "(e.g. 'head tr beats childten' -> 'head teacher beats children') lets the gate catch a disclosure that a model "
          "echoing the typos lets through. Gate regressions against Gemma per candidate are in section 2's gates."]
    L += ["", "## 4. Robustness: Q8_0 against Gemma Q8_0", "", "| model | correct / N (Q8_0) | b : w vs G_q8 | lower 95% bound of d (pts) |", "|---|---|---|---|"]
    for t in order:
        if t.endswith("_q8"):
            c = res["vs_G"].get(t.replace("_q8", "_q4"), {}).get("q8") or {}
            L.append(f"| {models.get(t, {}).get('name', t)} | {res['arms'].get(t, {}).get('ok')} | "
                     f"{c.get('b', '-')} : {c.get('w', '-')} | {c.get('d_lo_pts', '-')} |")
    L += ["", "## 5. Sizing detail (information only)", "", "| model | slots | clients | req/s | p50 s | p95 s | errors |", "|---|---|---|---|---|---|---|"]
    for t in order:
        sz = ((res["arms"].get(t) or {}).get("speed") or {}).get("sizing") or {}
        for k in ("8", "16"):
            for lv in (sz.get(k) or {}).get("levels") or []:
                L.append(f"| {t} | {k} | {lv['clients']} | {lv['rps']} | {lv['p50_s']} | {lv['p95_s']} | {lv['errors']} |")
    L += ["", f"Prompt tokens per request (sent / computed with the prefix cache), np 8: " + "; ".join(
        f"{t} {((((res['arms'].get(t) or {}).get('speed') or {}).get('sizing') or {}).get('8') or {}).get('prompt_tokens_mean')}"
        f" / {((((res['arms'].get(t) or {}).get('speed') or {}).get('sizing') or {}).get('8') or {}).get('prompt_computed_mean')}"
        for t in order if not t.endswith('_q8') and t != 'R'),
          "", "## 6. Weekly digest (information only: the timetable agent's 15 English notes, two digests per arm, the real DigestJob checks)", "",
          "An arm with 0 shown items produced no usable digest (every item dropped by the checks, or, for the API arm, a reply "
          "whose items were strings, which made DigestJob raise).", "",
          f"Shown items / unsourced: {res.get('digest')}; faithful (blind judge): {res.get('digest_faithful') or 'not judged'}", "",
          "## 7. Caveats", "",
          "- Only the requests the rules cannot place reach a model; differences live in that subset (count above).",
          "- The reader prompt (contract v14.1) was developed on Gemma and is the same for every model: a bias towards the incumbent.",
          "- GPU greedy decoding; a CPU deployment may differ on a few replies (hardware note: " + str(res.get("agreement")) + ").",
          "- tip:<id> targets are unreachable in the chat harness (no tip route): tip rows score on their honest outcomes; "
          "Grade 4 syllabus requests cannot reach route:syllabus (the detector takes Grades 1-3): the same for every arm.",
          "- English only (RULE v4): no Tonga or translator rows; the 15 English digest notes only.",
          "- The 300 noisy rows are synthetic copies of clean rows (english_noise.py, calibrated on the desk's real English "
          "messages, counts only): an approximation of how teachers type to an assistant; the real check is the first real "
          "teachers' held-out requests. A noisy row and its source are not independent.",
          "- Speed and throughput are for this card and llama.cpp b11574 with -np as stated; they scale with memory bandwidth.", "",
          "## 8. Run record", "", "```"] + cost[-12:] + ["```", ""]
    open(os.path.join(a.work, "REPORT.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(f"written {os.path.join(a.work, 'REPORT.md')} ({len(L)} lines)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "capture", "pack", "replay", "adjudicate", "speed", "agree", "score", "report"])
    ap.add_argument("--work", default=RUN)
    ap.add_argument("--testset", default=os.path.join(RUN, "testset_v2_en_noisy.jsonl"))   # RULE v4: English, clean + noisy
    ap.add_argument("--models")
    ap.add_argument("--rule-dir")
    ap.add_argument("--tag")
    ap.add_argument("--replies-dir")
    ap.add_argument("--gguf")
    ap.add_argument("--llama", help="llama-server for `speed` (default: the Vulkan build the harness serves with)")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--n", type=int, default=0, help="speed: first n latency bodies only (smoke)")
    ap.add_argument("--sha", action="store_true", help="speed: hash the GGUF too")
    ap.add_argument("--no-lock", action="store_true", help="smoke only: skip the Beelink lock")
    ap.add_argument("--no-served", action="store_true", help="smoke only (freeze): never load the served translator")
    ap.add_argument("--reuse-frozen", action="store_true", help="freeze: reuse frozen_en.jsonl's served outputs")
    ap.add_argument("--translator", choices=["beelink", "pod", "none"], default="beelink",
                    help="freeze: the served translator on the Beelink (lock) or the translator pod (Beelink fallback)")
    ap.add_argument("--notes-standin", action="store_true", help="smoke only (freeze): teacher English for the Tonga notes")
    A = ap.parse_args()
    {"freeze": cmd_freeze, "capture": cmd_capture, "pack": cmd_pack, "replay": cmd_replay, "adjudicate": cmd_adjudicate,
     "speed": cmd_speed, "agree": cmd_agree, "score": cmd_score, "report": cmd_report}[A.cmd](A)
