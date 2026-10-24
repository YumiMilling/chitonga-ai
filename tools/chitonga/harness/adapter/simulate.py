"""Replay scripted WhatsApp conversations through the pilot adapter with the WhatsApp API MOCKED (24 Oct 2026, the
Jesuit). Nothing leaves the machine: the sender is MockSender, the source is the script, every path (memory, adapter
state, logs, labels, the stop file) points at a fresh directory.

    python -X utf8 tools/chitonga/harness/adapter/simulate.py [script.json ...] [--out DIR] [--quiet]
With no script, every sim/*.json runs. Default --out: internal/chitonga_train/harness/pilot_logs/sim_<time>/ (the
logs and labels a real pilot would write, for reading). Exit code 1 if any expectation fails.

SCRIPT (JSON): {"name", "webhook": "current" | "title" | "full", "menus": "numbered" | "interactive",
                "desk_open": [codes], "steps": [...]}
  webhook  how the inbound row looks after a tap: 'current' = today ('[interactive]', no id, no title), 'title' =
           the half fix (body = the title), 'full' = the proposal (body = title, choice_id = id)
  step     {"from": "T01" | number, "text": str}            a typed message
           {"from": ..., "tap": "Grade 1"}                   a tap on the option with that title in the LAST message
                                                             sent to that number (numbered menus: the teacher types
                                                             its number, as a real teacher would)
           {"from": ..., "type": "image"}                    a media message
           {..., "age_h": 25}                                the row was written 25 h before it is processed (backlog)
           {"advance_h": 20}                                 the clock moves on
           {"kill": true | false}                            create / remove the stop file
           "expect": {"route", "sent" (count), "contains" (in the sent text), "not_contains", "label_kinds",
                      "path_has", "how", "mode"}
"""
import argparse
import datetime
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
sys.path.insert(0, HARNESS)     # 'core', 'plugins' and this package as 'adapter' ('harness' is taken: llm/harness.py)

from core.engine import Engine  # noqa: E402
from core.flywheel import Flywheel  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402

from adapter.adapter import PilotAdapter, engine_clock, iso  # noqa: E402
from adapter.config import Config  # noqa: E402
from adapter.store import AdapterStore  # noqa: E402
from adapter.transport import MockSender  # noqa: E402

ALLOW = {"teachers": [
    {"number": "260970000001", "code": "T01", "name": "Teacher one (simulation)", "role": "teacher",
     "consent": {"words": "SIMULATION", "date": "2026-10-24"}},
    {"number": "260970000009", "code": "T09", "name": "Reviewer nine (simulation)", "role": "reviewer",
     "consent": {"words": "SIMULATION", "date": "2026-10-24"}}]}
CODES = {t["code"]: t["number"] for t in ALLOW["teachers"]}
_PLUGIN = None


class SimClock:
    def __init__(self, start=datetime.datetime(2026, 10, 26, 6, 0, tzinfo=datetime.timezone.utc)):   # 08:00 Lusaka
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += datetime.timedelta(**kw)


class Sim:
    def __init__(self, out_dir, webhook="full", menus="interactive", desk_open=(), env=None, controls=None,
                 control_every=0):
        """controls / control_every: the flywheel's hidden controls (FLYWHEEL.md); none by default, so a script sees
        the prompts a teacher would, minus the 1-in-10 control (a test sets control_every=1 to force one)"""
        global _PLUGIN
        if _PLUGIN is None:
            _PLUGIN = education.load()
        P, hooks = _PLUGIN
        os.makedirs(out_dir, exist_ok=True)
        allow_path = os.path.join(out_dir, "pilot_allowlist.json")
        json.dump(ALLOW, open(allow_path, "w", encoding="utf-8"), indent=1)
        self.cfg = Config(base=out_dir, allowlist=allow_path, env=env or {})
        self.cfg.menus = menus
        self.webhook = webhook
        self.clock = SimClock()
        self.store = AdapterStore(self.cfg.adapter_db)
        self.sender = MockSender()
        self.notes = []
        self.flywheel = Flywheel(controls or [], who=lambda n: {v: k for k, v in CODES.items()}.get(n),
                                 control_every=control_every)
        self.engine = Engine(P, SQLiteMemory(self.cfg.memory_db), hooks=hooks, clock=engine_clock(self.clock),
                             flywheel=self.flywheel)
        desk = {CODES.get(c, c) for c in desk_open}
        self.ad = PilotAdapter(self.engine, self.cfg, self.store, self.sender, self.clock, sleep=lambda s: None,
                               notify=self.notes.append, live=False, desk_open=lambda n, now: n in desk, mode="sim")
        self.k = 0

    def row(self, frm, body, msg_type="text", choice_id=None, age_h=0):
        self.k += 1
        n = CODES.get(frm, frm)
        r = {"id": f"row{self.k}", "wa_message_id": f"wamid.in{self.k}", "counterparty": n, "msg_type": msg_type,
             "body": body, "created_at": iso(self.clock() - datetime.timedelta(hours=age_h))}
        if choice_id and self.webhook == "full":
            r["choice_id"] = choice_id
        return r

    def tap(self, frm, title, age_h=0):
        """the row a phone and today's / the fixed webhook produce for a tap on `title` in the last message"""
        n = CODES.get(frm, frm)
        last = self.store.last_menu(n)
        opts = last["options"] if last else []
        hit = [(i, oid, t) for i, (oid, t) in enumerate(opts) if t.lower().startswith(title.lower())]
        if not hit:
            raise AssertionError(f"no option '{title}' in the last menu to {frm}: {[t for _, t in opts]}")
        i, oid, t = hit[0]
        if self.cfg.menus == "numbered":
            return self.row(frm, str(i + 1), age_h=age_h)
        if self.webhook == "current":
            return self.row(frm, "[interactive]", "interactive", age_h=age_h)
        return self.row(frm, t, "interactive", choice_id=oid, age_h=age_h)

    def step(self, s):
        if "advance_h" in s:
            self.clock.advance(hours=s["advance_h"])
            return None
        if "kill" in s:
            if s["kill"]:
                self.cfg.stop("simulation: kill switch test")
            elif os.path.exists(self.cfg.stop_path):
                os.remove(self.cfg.stop_path)
            return None
        before = len(self.sender.sent)
        if "tap" in s:
            row = self.tap(s["from"], s["tap"], s.get("age_h", 0))
        elif "type" in s:
            row = self.row(s["from"], f"[{s['type']}]", s["type"], age_h=s.get("age_h", 0))
        else:
            row = self.row(s["from"], s["text"], age_h=s.get("age_h", 0))
        res = self.ad.process(row)
        self.clock.advance(seconds=40)
        res["_sent"] = self.sender.sent[before:]
        res["_row"] = row
        return res


def check(res, exp, sim):
    fails = []
    sent_text = "\n".join(x["text"] for x in res.get("_sent", []))
    if "route" in exp and res.get("route") != exp["route"]:
        fails.append(f"route {res.get('route')} != {exp['route']} ({res.get('why')})")
    if "sent" in exp and len(res.get("_sent", [])) != exp["sent"]:
        fails.append(f"sent {len(res.get('_sent', []))} != {exp['sent']}")
    for c in ([exp["contains"]] if isinstance(exp.get("contains"), str) else exp.get("contains", [])):
        if c not in sent_text:
            fails.append(f"'{c}' not in the sent text")
    for c in ([exp["not_contains"]] if isinstance(exp.get("not_contains"), str) else exp.get("not_contains", [])):
        if c in sent_text:
            fails.append(f"'{c}' in the sent text")
    if "label_kinds" in exp and res.get("label_kinds", []) != exp["label_kinds"]:
        fails.append(f"labels {res.get('label_kinds', [])} != {exp['label_kinds']}")
    if "path_has" in exp and not any(exp["path_has"] in str(p) for p in (res.get("turn") or {}).get("path", [])):
        fails.append(f"path {(res.get('turn') or {}).get('path')} lacks {exp['path_has']}")
    if "how" in exp and res.get("how") != exp["how"]:
        fails.append(f"how {res.get('how')} != {exp['how']}")
    if "mode" in exp:
        n = res["_row"]["counterparty"]
        if sim.store.mode(n)["mode"] != exp["mode"]:
            fails.append(f"mode {sim.store.mode(n)['mode']} != {exp['mode']}")
    if "payload_type" in exp and not any(x["payload"]["type"] == exp["payload_type"] for x in res.get("_sent", [])):
        fails.append(f"no {exp['payload_type']} payload sent")
    return fails


def run_script(script, out_dir, quiet=False):
    sim = Sim(out_dir, script.get("webhook", "full"), script.get("menus", "interactive"), script.get("desk_open", ()))
    fails, results = [], []
    for i, s in enumerate(script["steps"]):
        res = sim.step(s)
        if res is None:
            continue
        results.append(res)
        f = check(res, s.get("expect", {}), sim)
        if not quiet:
            what = s.get("text", s.get("tap", s.get("type")))
            print(f"  [{i:02d}] {s.get('from')}: {what!r} -> {res.get('route')} "
                  f"{res.get('how') or res.get('why') or ''} labels={res.get('label_kinds', [])}")
            for x in res["_sent"]:
                print("       | " + x["text"].replace("\n", "\n       | ")[:600])
        fails += [f"step {i}: {x}" for x in f]
    return fails, sim, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scripts", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    scripts = a.scripts or sorted(glob.glob(os.path.join(HERE, "sim", "*.json")))
    out = a.out or os.path.join(Config().log_dir, "sim_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    total = 0
    for p in scripts:
        sc = json.load(open(p, encoding="utf-8"))
        d = os.path.join(out, os.path.splitext(os.path.basename(p))[0])
        shutil.rmtree(d, ignore_errors=True)
        print(f"== {sc['name']} (webhook {sc.get('webhook', 'full')}, menus {sc.get('menus', 'interactive')})")
        fails, sim, _ = run_script(sc, d, a.quiet)
        for f in fails:
            print("  FAIL " + f)
        print(f"  {'OK' if not fails else str(len(fails)) + ' FAILED'}; mock sends {len(sim.sender.sent)}; "
              f"notes {sim.notes}")
        total += len(fails)
    print(f"logs and labels in {out}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
