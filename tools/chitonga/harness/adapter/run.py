"""Run the pilot adapter (24 Oct 2026, the Jesuit). NOTHING IS SENT except in --live, and --live refuses to start
unless Chris's go is on the machine (see config.py: env ZAMAI_ASSISTANT_LIVE=1 + PILOT_GO.json with his words + no
PILOT_STOP). Every send re-checks all three.

    python -X utf8 tools/chitonga/harness/adapter/run.py status
        the switches, the allow-list (consent or not), the menu mode, today's log size
    python -X utf8 tools/chitonga/harness/adapter/run.py replay ROWS.jsonl [--assistant T01,T09]
        replay recorded zamai_messages rows (one JSON object per line) through the dispatcher and the engine with
        the sender MOCKED; prints the routes and the payloads. --assistant puts those teachers in assistant mode
        first (otherwise every row goes to the desk, as live). Use it on a recorded week of desk traffic: the go
        condition is ZERO desk answers routed to the engine.
    python -X utf8 tools/chitonga/harness/adapter/run.py dry-run [--once]
        READ the live zamai_messages (allow-listed numbers only, read-only), run the engine, print what WOULD be
        sent; mocked sender; its own state under pilot_logs/dryrun/ (never the live mark or memory)
    python -X utf8 tools/chitonga/harness/adapter/run.py live
        the real thing. Refuses without the go.
    python -X utf8 tools/chitonga/harness/adapter/run.py pay [--since 2026-10-27] [--cap 20]
        what the teachers earned under the proposed model: K2 per real request that carries the teacher's feedback
        (a Yes / No / Not this one / correction / Did-it-help answer), at most --cap requests per teacher per day
    python -X utf8 tools/chitonga/harness/adapter/run.py export [--since DAY] [--base DIR]
        the flywheel export (../flywheel_export.py; FLYWHEEL.md): the day's teacher inputs -> flywheel_out/<day>/.
        Also started by the shadow loop once a night after 23:00 Lusaka (nightly_export)
STOP: create internal/chitonga_train/harness/PILOT_STOP (any content), or set ZAMAI_ASSISTANT_STOP=1. The next message
is not answered; the loop keeps running and routes everything to the desk.
"""
import argparse
import collections
import datetime
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from core.engine import Engine  # noqa: E402
from core.flywheel import Flywheel  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402

from adapter.adapter import PilotAdapter, desk_open_from_files, engine_clock  # noqa: E402
from adapter.config import REPO, Config  # noqa: E402
from adapter.store import AdapterStore  # noqa: E402
from adapter.transport import MockSender, OutboxSender, utcnow  # noqa: E402

DESK = os.path.join(REPO, "internal", "chitonga_train", "desk")
POLL_S = 5


def build(cfg, sender, live=False, notify=None, desk_sent=None, mode=None, flywheel_silent=()):
    """flywheel_silent: numbers the flywheel never prompts (those whose replies are really sent, unless
    ZAMAI_FLYWHEEL_SEND=1: the flywheel was built in shadow, 24 Oct, and nothing of it is sent without that switch)"""
    P, hooks = education.load()
    allow = cfg.allowlist()
    silent = set(flywheel_silent)
    # the flywheel (FLYWHEEL.md): at most one feedback prompt per answer; hidden controls from the base's control file
    fw = Flywheel(cfg.controls_path, who=lambda n: (allow.get(n) or {}).get("code"), enabled=lambda n: n not in silent)
    eng = Engine(P, SQLiteMemory(cfg.memory_db), hooks=hooks, clock=engine_clock(utcnow),   # tonga_ui False: English
                 flywheel=fw)
    store = AdapterStore(cfg.adapter_db)
    return PilotAdapter(eng, cfg, store, sender, utcnow, notify=notify, live=live,
                        desk_open=desk_open_from_files(DESK), desk_sent=desk_sent, mode=mode), store


def show(res):
    print(f"{res.get('in_at')} {res.get('teacher', '-')} {res.get('route')} {res.get('how') or res.get('why') or ''}"
          f" labels={res.get('label_kinds', [])}")
    for s in res.get("sends", []):
        print("   > " + (s.get("text") or s.get("why") or "").replace("\n", "\n     ")[:800])


def cmd_status(cfg):
    print(f"stop switch: {'ON' if cfg.stopped() else 'off'} ({cfg.stop_path})")
    print(f"go file: {'present' if cfg.go() else 'absent'} ({cfg.go_path})")
    print(f"env ZAMAI_ASSISTANT_LIVE: {os.environ.get('ZAMAI_ASSISTANT_LIVE', '(unset)')}")
    print(f"LIVE: {cfg.live()}")
    print(f"menus: {cfg.menus}")
    allow_all, allow_live = cfg.allowlist(), cfg.allowlist(live=True)
    store = AdapterStore(cfg.adapter_db) if os.path.exists(cfg.adapter_db) else None
    for n, e in allow_all.items():
        m = store.mode(n) if store else {"mode": "desk", "since": None}
        print(f"  {e.get('code')} {e.get('name')}: {'tester (no consent note needed)' if e.get('tester') else 'consent recorded' if n in allow_live else 'NO CONSENT: refused live'}; "
              f"mode {m['mode']} since {m.get('since')}")
    print("The desk session answers ONLY numbers in mode desk (a number in mode assistant is the adapter's).")


def cmd_replay(cfg, path, assistant, base=None, outbox=False):
    """base: reuse a directory (a second run in a NEW process reads the state the first one stored: the restart
    test); outbox: write the would-be sends to that directory's outbox.jsonl instead of a mock list"""
    base = base or os.path.join(cfg.log_dir, "replay_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    rc = Config(base=base, allowlist=cfg.allowlist_path)
    rc.menus = cfg.menus
    sender = OutboxSender(rc.outbox_path, rc) if outbox else MockSender()
    if os.path.exists(cfg.controls_path) and not os.path.exists(rc.controls_path):
        os.makedirs(rc.base, exist_ok=True)
        open(rc.controls_path, "w", encoding="utf-8").write(open(cfg.controls_path, encoding="utf-8").read())
    ad, store = build(rc, sender, mode="sim")
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r.get("direction", "in") == "in"]
    rows.sort(key=lambda r: r.get("created_at") or "")
    codes = {e.get("code"): n for n, e in ad.allow.items()}
    for c in filter(None, (assistant or "").split(",")):
        store.set_mode(codes[c], "assistant", rows[0]["created_at"] if rows else "")
    routes = collections.Counter()
    first_ts = rows[0]["created_at"] if rows else None
    for r in rows:
        ad.clock = (lambda ts=r["created_at"]: datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")))
        ad.eng.clock = engine_clock(ad.clock)
        res = ad.process(r)
        routes[res["route"]] += 1
        show(res)
    print(f"\n{len(rows)} rows from {first_ts}: {dict(routes)}; logs in {base}")


def cmd_loop(cfg, live, once=False):
    from adapter.transport import CloudSender, ZamaiSource
    notify = None
    try:
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import notify as N  # noqa: E402
        notify = (lambda text: N.send(text, title="Tonga assistant")) if live else None
    except Exception:  # noqa: BLE001
        pass
    if live:
        if not cfg.live():
            sys.exit("REFUSED: not live. Needs env ZAMAI_ASSISTANT_LIVE=1 AND PILOT_GO.json with Chris's words AND no "
                     "PILOT_STOP. Run 'status'.")
        if not cfg.allowlist(live=True):
            sys.exit("REFUSED: no allow-listed number has a consent note")
        sender = CloudSender(cfg)
    else:
        cfg = Config(base=os.path.join(cfg.log_dir, "dryrun"), allowlist=cfg.allowlist_path)
        sender = MockSender()
    store0 = AdapterStore(cfg.adapter_db)
    src = ZamaiSource(cfg.allowlist(live=live).keys(), store0)
    silent = set(cfg.allowlist(live=live)) if (live and os.environ.get("ZAMAI_FLYWHEEL_SEND") != "1") else set()
    ad, store = build(cfg, sender, live=live, notify=notify, desk_sent=src.desk_sent, flywheel_silent=silent)
    print(f"{'LIVE' if live else 'DRY RUN (nothing sent)'}: watching {len(ad.allow)} numbers from {src.start_mark()}")
    beat = os.path.join(cfg.log_dir, "heartbeat")
    while True:
        try:
            for r in src.poll():
                show(ad.process(r))
        except Exception as e:  # noqa: BLE001  (the loop must not die on one bad poll)
            print(f"poll error: {e}", flush=True)
            time.sleep(POLL_S * 3)
        os.makedirs(cfg.log_dir, exist_ok=True)
        open(beat, "w").write(utcnow().isoformat(timespec="seconds"))
        if once:
            return
        time.sleep(POLL_S)


def cmd_shadow(cfg, since=None, poll_s=30, send_tester=False, once=False):
    """SHADOW-LIVE (24 Oct 2026, Chris: "the whatsapp adapter needs to be live in as much as that it is possible to
    test data saving and data retrieval"): real inbound (zamai_messages, allow-listed numbers, read-only), the real
    engine and ALL real persistence (pilot_memory.sqlite, pilot_adapter.sqlite, pilot_logs/, pilot_labels.jsonl), and
    NO sends: every reply goes to pilot_logs/outbox.jsonl with its exact payload. The dispatcher is unchanged: a
    teacher's desk traffic is logged, never fed to the engine (it would use up their 40 held-out requests and fill
    the memory with profiles built from desk answers); the engine runs on everything routed to it (any allow-listed
    number after "Assistant"). --send-tester: replies to a TESTER number (Chris) are sent for real, only while the
    three keys hold (config.live()); teachers never."""
    from adapter.transport import CloudSender, ZamaiSource
    real = None
    if send_tester:
        if not cfg.live():
            sys.exit("REFUSED --send-tester: not live (env ZAMAI_ASSISTANT_LIVE=1, PILOT_GO.json with Chris's words, "
                     "no PILOT_STOP). Run 'status'.")
        real = CloudSender(cfg)
    sender = OutboxSender(cfg.outbox_path, cfg, real=real)
    store0 = AdapterStore(cfg.adapter_db)
    if since and not store0.get("mark"):
        store0.put("mark", since)                    # the first start reads from `since` (e.g. today 00:00 Lusaka)
    src = ZamaiSource(cfg.allowlist().keys(), store0)
    silent = cfg.testers() if (real and os.environ.get("ZAMAI_FLYWHEEL_SEND") != "1") else set()
    ad, store = build(cfg, sender, live=False, desk_sent=src.desk_sent, mode="shadow", flywheel_silent=silent)
    os.makedirs(cfg.log_dir, exist_ok=True)
    open(cfg.shadow_pid, "w").write(str(os.getpid()))
    print(f"{utcnow().isoformat(timespec='seconds')} SHADOW-LIVE pid {os.getpid()}: watching {len(ad.allow)} numbers "
          f"({', '.join(e.get('code') for e in ad.allow.values())}) from {src.start_mark()}, every {poll_s} s; "
          f"sends: {'tester only, gated' if real else 'NONE (outbox)'}; flywheel silent for {len(silent)} real-send number(s)", flush=True)
    beat = os.path.join(cfg.log_dir, "heartbeat")
    try:
        while True:
            if os.path.exists(cfg.shadow_exit):
                os.remove(cfg.shadow_exit)
                print(f"{utcnow().isoformat(timespec='seconds')} SHADOW_EXIT found: stopping", flush=True)
                break
            try:
                for r in src.poll():
                    show(ad.process(r))
                    sys.stdout.flush()
            except Exception as e:  # noqa: BLE001  (the loop must not die on one bad poll)
                print(f"{utcnow().isoformat(timespec='seconds')} poll error: {e}", flush=True)
            open(beat, "w").write(utcnow().isoformat(timespec="seconds"))
            nightly_export(cfg, store0)
            if once:
                break
            time.sleep(poll_s)
    finally:
        if os.path.exists(cfg.shadow_pid):
            os.remove(cfg.shadow_pid)


EXPORT_HOUR = 23     # Lusaka: the day's flywheel export runs once after this hour (FLYWHEEL.md, the loop out)


def nightly_export(cfg, store, now=None):
    """once a day after EXPORT_HOUR Lusaka time, start the flywheel export for that day as a SEPARATE process (it loads
    the attestation cache, about a minute: the poll loop must not wait for it, and its failure must not stop the loop)"""
    import subprocess
    now = (now or utcnow()).astimezone(datetime.timezone(datetime.timedelta(hours=2)))
    day = now.date().isoformat()
    if now.hour < EXPORT_HOUR or store.get("export_day") == day:
        return None
    store.put("export_day", day)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log = open(os.path.join(cfg.log_dir, "export_console.log"), "a", encoding="utf-8")
    flags = (0x00000008 | 0x00000200) if os.name == "nt" else 0
    try:
        return subprocess.Popen([sys.executable, "-X", "utf8", os.path.join(os.path.dirname(HERE), "flywheel_export.py"),
                                 "--day", day, "--base", cfg.base], stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    except OSError as e:
        print(f"nightly export not started: {e}", flush=True)
        return None


def cmd_shadow_start(cfg, since, poll_s):
    """start the shadow loop as a detached background process (survives this console); output to shadow_console.log"""
    import subprocess
    if os.path.exists(cfg.shadow_pid):
        sys.exit(f"already running? {cfg.shadow_pid} exists (pid {open(cfg.shadow_pid).read()}); stop it first")
    os.makedirs(cfg.log_dir, exist_ok=True)
    log = open(os.path.join(cfg.log_dir, "shadow_console.log"), "a", encoding="utf-8")
    args = [sys.executable, "-X", "utf8", os.path.abspath(__file__), "shadow", "--poll", str(poll_s)] + \
        (["--since", since] if since else [])
    flags = (0x00000008 | 0x00000200) if os.name == "nt" else 0     # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags,
                         close_fds=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    print(f"shadow started, pid {p.pid}; log {log.name}")


def cmd_shadow_stop(cfg):
    os.makedirs(cfg.log_dir, exist_ok=True)
    open(cfg.shadow_exit, "w").write("stop")
    print(f"SHADOW_EXIT written: the loop stops at its next poll. Hard stop: kill the pid in {cfg.shadow_pid}")


def _lines(path):
    try:
        return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    except OSError:
        return []


def cmd_show(cfg, who=None, turns=5):
    """RETRIEVAL: per allow-listed number, what is stored: profile, classes, positions, resume point, recent requests,
    events, adapter mode, the last N logged turns, the labels and the outbox"""
    mem = SQLiteMemory(cfg.memory_db)
    store = AdapterStore(cfg.adapter_db)
    logs = [r for f in sorted(glob.glob(os.path.join(cfg.log_dir, "20*.jsonl"))) for r in _lines(f)]
    labels, outbox = _lines(cfg.labels_path), _lines(cfg.outbox_path)
    flyw = _lines(cfg.flywheel_path)
    ledger, ledger_path = [], None
    for lp in (cfg.pay_ledger_path(True), cfg.pay_ledger_path(False)):
        if os.path.exists(lp):
            import csv
            ledger_path = lp
            ledger += list(csv.DictReader(open(lp, encoding="utf-8"), delimiter="	"))
    allow = cfg.allowlist()
    print(f"store: {cfg.base}")
    print(f"mark: {store.get('mark')}; turns logged: {len(logs)}; labels: {len(labels)}; outbox: {len(outbox)}; "
          f"flywheel records: {len(flyw)}")
    for n, e in allow.items():
        if who and who not in (n, e.get("code")):
            continue
        prof, st = mem.get_teacher(n), mem.get_state(n) or {}
        print(f"\n=== {e.get('code')} {e.get('name')} ({n}){' TESTER' if e.get('tester') else ''}")
        m = store.mode(n)
        print(f"adapter mode: {m['mode']} since {m.get('since')}; last inbound {m.get('last_in_at')}")
        if not prof:
            print("profile: none stored (never reached the engine)")
        else:
            print("profile: " + json.dumps({k: v for k, v in prof.items() if k != "number"}, ensure_ascii=False,
                                           default=str)[:400])
            for c in mem.classes(n):
                print(f"  class {c['id']}: {c['label']} grades {c['grades']}; positions "
                      + json.dumps(mem.positions(c["id"]), ensure_ascii=False, default=str)[:300])
            resume = {k: st.get(k) for k in ("step", "subject", "unit", "item", "part", "parts", "finished",
                                             "last_at", "session") if st.get(k) is not None}
            print("resume point: " + json.dumps(resume, ensure_ascii=False))
            print("recent requests: " + json.dumps(mem.recent_requests(n, 5), ensure_ascii=False, default=str)[:600])
            print("events: " + json.dumps(mem.events(n), ensure_ascii=False, default=str)[:300])
        mine = [r for r in logs if r.get("number") == n]
        print(f"turns logged: {len(mine)} (last {turns}):")
        for r in mine[-turns:]:
            inbound = r.get("inbound") or {}
            print(f"  {r.get('in_at')} {r.get('route')} {r.get('how') or r.get('why') or ''} "
                  f"in={json.dumps(inbound, ensure_ascii=False)[:80]} path={(r.get('turn') or {}).get('path')}")
        lab = [x for x in labels if x.get("teacher") == e.get("code")]
        print(f"labels: {len(lab)}")
        for x in lab[-turns:]:
            print(f"  {x['at']} {x['kind']} {x['label']} {x.get('request_id')} {x.get('proposed')} -> {x.get('chosen')}")
        fwr = [x for x in flyw if x.get("teacher") == e.get("code")]
        print(f"flywheel records: {len(fwr)} (" + ", ".join(f"{k} {v}" for k, v in collections.Counter(
            x["ev"] + ":" + str(x.get("kind") or x.get("what")) for x in fwr).most_common()) + ")")
        for x in fwr[-turns:]:
            seen = x.get("served") or x.get("target") or {}
            print(f"  {x['at']} {x['ev']} {x.get('kind') or x.get('what') or 'feedback ' + str(x.get('feedback'))} {x.get('answer') or ''} "
                  f"item={x.get('item') or x.get('ref') or '-'} line={seen.get('n') if isinstance(seen, dict) else '-'}"
                  f" pay=K{x.get('pay') or 0}{' CONTROL ' + x['control']['expect'] if x.get('control') else ''}"
                  + (f" teacher_text={x['teacher_text']!r}" if x.get("teacher_text") else "")
                  + (f" text={x['text']!r}" if x.get("text") else ""))
        ctl = [x for x in fwr if x["ev"] == "answer" and x.get("control")]
        if ctl:
            ok = sum(1 for x in ctl if (x["control"]["expect"] == "right") == (x["answer"] == "right"))
            print(f"controls answered: {len(ctl)}, as expected {ok}")
        led = [r for r in ledger if r.get("code") == e.get("code")]
        if led:
            print(f"pay ledger ({os.path.basename(ledger_path)}): {len(led)} rows, balance K{led[-1]['balance_after']}")
            for r in led[-turns:]:
                print(f"  {r['date']} {r['what']} K{r['K']} -> K{r['balance_after']}")
        ob = [x for x in outbox if x.get("to") == n]
        print(f"outbox (would have been sent): {len(ob)}")
        for x in ob[-turns:]:
            print(f"  {x['at']} {x['payload']['type']}: " + x["text"].replace("\n", " / ")[:140])


def cmd_pay(cfg, since, cap):
    """K2 per real request with feedback (proposal, as the desk's K2 per answer), capped per teacher per day"""
    per = collections.defaultdict(set)
    try:
        lines = open(cfg.labels_path, encoding="utf-8").read().splitlines()
    except OSError:
        lines = []
    for x in lines:
        r = json.loads(x)
        if since and r["at"][:10] < since:
            continue
        key = r.get("request_id") or (r["kind"] == "strategy_feedback" and f"fb{r.get('suggestion_id')}")
        if key:
            per[(r["teacher"], r["at"][:10])].add(key)
    total = collections.Counter()
    for (t, day), reqs in sorted(per.items()):
        k = min(len(reqs), cap)
        total[t] += 2 * k
        print(f"{day} {t}: {len(reqs)} requests with feedback, paid {k} x K2 = K{2 * k}")
    print("total: " + ", ".join(f"{t} K{v}" for t, v in total.items()) if total else "nothing earned yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "replay", "dry-run", "live", "pay", "shadow", "shadow-start",
                                    "shadow-stop", "show", "export"])
    ap.add_argument("--base", help="replay / show: a store directory other than the pilot's")
    ap.add_argument("--outbox", action="store_true", help="replay: write would-be sends to the base's outbox.jsonl")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--send-tester", action="store_true")
    ap.add_argument("--who")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--assistant", default="")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--since")
    ap.add_argument("--cap", type=int, default=20)
    a = ap.parse_args()
    cfg = Config()
    if a.cmd == "status":
        cmd_status(cfg)
    elif a.cmd == "replay":
        cmd_replay(cfg, a.path, a.assistant, base=a.base, outbox=a.outbox)
    elif a.cmd == "shadow":
        cmd_shadow(cfg, a.since, a.poll, a.send_tester, a.once)
    elif a.cmd == "shadow-start":
        cmd_shadow_start(cfg, a.since, a.poll)
    elif a.cmd == "shadow-stop":
        cmd_shadow_stop(cfg)
    elif a.cmd == "show":
        cmd_show(Config(base=a.base, allowlist=cfg.allowlist_path) if a.base else cfg, a.who, a.turns)
    elif a.cmd == "export":
        import subprocess
        args = [sys.executable, "-X", "utf8", os.path.join(os.path.dirname(HERE), "flywheel_export.py"),
                "--base", a.base or cfg.base] + (["--day", a.since] if a.since else ["--all"])
        sys.exit(subprocess.call(args))
    elif a.cmd == "dry-run":
        cmd_loop(cfg, live=False, once=a.once)
    elif a.cmd == "live":
        cmd_loop(cfg, live=True)
    else:
        cmd_pay(cfg, a.since, a.cap)


if __name__ == "__main__":
    main()
