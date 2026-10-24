"""Pilot configuration: paths, the allow-list, the kill switch and the live gate (24 Oct 2026, the Jesuit).

WHY THREE KEYS FOR A SEND. AI_AGENT_POLICY section 2: nothing goes out without Chris's approval. The adapter is built
and tested before his go, on the same machine that will later run it live, so "not live" must be a property of the
machine state, not of somebody remembering a flag:
  - LIVE needs env ZAMAI_ASSISTANT_LIVE=1 AND the go file PILOT_GO.json holding Chris's words and the date;
  - STOP wins over everything: the file PILOT_STOP (any content) or env ZAMAI_ASSISTANT_STOP set. It is read before
    EVERY single send and before every engine call, so touching the file stops the next message, not the next loop.
The adapter never writes PILOT_GO.json and never removes PILOT_STOP; it only creates PILOT_STOP itself (the automatic
stops of WHATSAPP_ADAPTER.md section 8).

THE ALLOW-LIST (internal/chitonga_train/harness/pilot_allowlist.json): number, code, name, role, consent. A number
without a consent note is loaded for simulation but refused in live mode (the loader says which). Everyone else is
ignored: never answered by the harness, never passed to the engine, logged only as "ignored" with no body.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
HARNESS_DATA = os.path.join(REPO, "internal", "chitonga_train", "harness")
ZAMAIPORT = os.environ.get("ZAMAI_ENV_DIR", os.path.join(REPO, "private"))   # PUBLIC COPY: was a private sibling repo


def norm_number(n):
    d = re.sub(r"\D", "", str(n or ""))
    return "260" + d[-9:] if len(d) >= 9 else d


class Config:
    """Every path the adapter touches, so a test or a simulation can point all of them at a temp directory."""

    def __init__(self, base=None, allowlist=None, env=None):
        self.base = base or HARNESS_DATA
        self.env = os.environ if env is None else env
        self.allowlist_path = allowlist or os.path.join(self.base, "pilot_allowlist.json")
        self.go_path = os.path.join(self.base, "PILOT_GO.json")
        self.stop_path = os.path.join(self.base, "PILOT_STOP")
        self.log_dir = os.path.join(self.base, "pilot_logs")
        self.labels_path = os.path.join(self.base, "pilot_logs", "pilot_labels.jsonl")
        self.outbox_path = os.path.join(self.base, "pilot_logs", "outbox.jsonl")   # shadow: what WOULD be sent
        self.shadow_exit = os.path.join(self.base, "pilot_logs", "SHADOW_EXIT")    # shadow loop: exit at next poll
        self.shadow_pid = os.path.join(self.base, "pilot_logs", "shadow.pid")
        self.memory_db = os.path.join(self.base, "pilot_memory.sqlite")      # teacher memory (the engine's)
        self.adapter_db = os.path.join(self.base, "pilot_adapter.sqlite")    # marks, modes, menus, counters
        # the flywheel (24 Oct 2026, FLYWHEEL.md): its records, the hidden controls, the export folder
        self.flywheel_path = os.path.join(self.base, "pilot_logs", "flywheel.jsonl")
        self.controls_path = os.path.join(self.base, "flywheel_controls.jsonl")
        self.flywheel_out = os.path.join(self.base, "flywheel_out")
        # behaviour (WHATSAPP_ADAPTER.md sections 4, 5, 8; numbers chosen there)
        self.window_h = 24
        self.max_out_per_in = 6
        self.max_out_per_hour = 60
        self.max_in_per_hour = 30
        self.fail_limit, self.fail_window_min = 3, 10
        self.assistant_idle_h = 2          # silence after which a number goes back to the desk
        self.held_out_first = 40           # the first 40 real requests per teacher are test, never train
        self.send_gap_s = 0.5              # WhatsApp does not guarantee the order of near-simultaneous sends
        # menus: 'interactive' (buttons and lists) only once the webhook stores the tap (proposals/); until then
        # 'numbered' (plain text, "Reply with the number"), because a tap today arrives as '[interactive]' with no id
        self.menus = self.env.get("ZAMAI_ASSISTANT_MENUS", "numbered")
        self.template_ok = False           # outside the 24 h window: nothing. A template only on Chris's word.

    def pay_ledger_path(self, live):
        """the pilot pay ledger (balances.tsv columns): live earnings, or the shadow / simulation tally"""
        return os.path.join(self.log_dir, "pay_ledger.tsv" if live else "pay_ledger_shadow.tsv")

    # ------------------------------------------------------------------ switches
    def stopped(self):
        """the kill switch: True stops all outbound and all engine calls, at once"""
        return os.path.exists(self.stop_path) or bool(self.env.get("ZAMAI_ASSISTANT_STOP"))

    def go(self):
        """Chris's go, or None. The file must carry his words and a date; the adapter never writes it."""
        try:
            g = json.load(open(self.go_path, encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return g if g.get("chris_words") and g.get("date") else None

    def live(self):
        """True only with the env flag AND the go file AND no stop"""
        return self.env.get("ZAMAI_ASSISTANT_LIVE") == "1" and self.go() is not None and not self.stopped()

    def stop(self, why):
        """an automatic stop: create PILOT_STOP with the reason (never removed by the adapter)"""
        os.makedirs(self.base, exist_ok=True)
        with open(self.stop_path, "a", encoding="utf-8") as f:
            f.write(why.strip() + "\n")

    def testers(self):
        """the numbers flagged tester=true: the ONLY ones a shadow run may ever send to (and only when live())"""
        return {n for n, e in self.allowlist().items() if e.get("tester")}

    # ------------------------------------------------------------------ the allow-list
    def allowlist(self, live=False):
        """{number: entry}. In live mode an entry without consent is refused (and named in 'refused')."""
        try:
            data = json.load(open(self.allowlist_path, encoding="utf-8"))
        except OSError:
            return {}
        out = {}
        for e in data.get("teachers", []):
            n = norm_number(e.get("number"))
            if len(n) != 12:
                continue
            if live and not e.get("tester") and not (e.get("consent") or {}).get("words"):
                continue                     # a tester (Chris, the owner) needs no consent note; a teacher does
            out[n] = e
        return out
