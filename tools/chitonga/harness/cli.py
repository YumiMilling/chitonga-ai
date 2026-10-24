"""Talk to the harness on the Beelink (23 Oct 2026, the Jesuit). Nothing is sent anywhere; the memory and the turn log
go to internal/chitonga_train/harness/ unless --db / --log say otherwise.

    python -X utf8 tools/chitonga/harness/cli.py [--number 260970000001] [--db PATH|:memory:] [--log PATH|none] [--drafts]
Type a message; type a number to tap button / row N of the last reply; '/who 2609...' switches teacher; '/state' shows
the stored state; '/later H' moves the clock H hours on (to try resuming); '/quit'.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import io as IO  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.log import TurnLog  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--number", default="260970000001")
    ap.add_argument("--db", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--drafts", action="store_true", help="also show fixed-line Tonga still waiting for a teacher, tagged")
    a = ap.parse_args()
    P, hooks = education.load()
    mem = SQLiteMemory(a.db)
    shift = [datetime.timedelta(0)]
    eng = Engine(P, mem, hooks=hooks, log=TurnLog(None if a.log == "none" else (a.log or TurnLog().path)),
                 clock=lambda: datetime.datetime.now() + shift[0], drafts=a.drafts)
    who, last = a.number, []
    print(f"[{P.name} {P.manifest['version']}, teacher {who}]")
    while True:
        try:
            s = input("> ").strip()
        except EOFError:
            break
        if s == "/quit":
            break
        if s.startswith("/who"):
            who = s.split()[1]
            continue
        if s == "/state":
            print(json.dumps({"teacher": mem.get_teacher(who), "classes": mem.classes(who), "state": mem.get_state(who)},
                             indent=1, ensure_ascii=False))
            continue
        if s.startswith("/later"):
            shift[0] += datetime.timedelta(hours=float(s.split()[1]))
            continue
        msg = {"text": s}
        if s.isdigit() and last:
            opts = [o for r in last for o in (r.get("buttons") or r.get("rows") or [])]
            if 1 <= int(s) <= len(opts):
                msg = {"choice": opts[int(s) - 1]["id"]}
        last = eng.handle(who, msg)
        for r in last:
            print(IO.as_text(r))
            print("-" * 40)


if __name__ == "__main__":
    main()
