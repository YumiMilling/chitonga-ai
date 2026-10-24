# Zamai assistant watchdog (24 Oct 2026, the Jesuit), modelled on an earlier responder's watchdog (not in this repo).
#
# Run every 10 minutes by the scheduled task "Zamai assistant watchdog" through pythonw.exe, so no console window
# ever pops up. One check (common.watchdog_check):
#   RUNTIME_STOP present                     -> nothing, ever (STOP in the widget means down until START)
#   process alive, heartbeat <= 3 min old     -> nothing
#   alive, no fresh beat, started < 5 min ago -> nothing (the engine is still loading)
#   alive but heartbeat stale (hung)          -> kill it, then relaunch
#   dead                                      -> relaunch
#   3 relaunches in an hour, never a fresh beat -> stop trying (crash loop), say so in the log
# A relaunch goes `explorer.exe launch_assistant.cmd` (a child of the task would die with the task) and counts only
# when start_assistant.ps1 wrote its start line. Every run writes one line to
# internal/chitonga_train/harness/pilot_logs/watchdog.log. The task must stay "run only when the user is logged on"
# (interactive): the relaunch opens a window on the desktop.
import datetime
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import common as C
    p = C.Paths()
    C.watchdog_check(p)
except Exception:  # never let the watchdog die silently
    log = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "internal", "chitonga_train", "harness",
                                        "pilot_logs", "watchdog.log"))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}  watchdog ERROR: "
                + traceback.format_exc().replace("\n", " | ") + "\n")
