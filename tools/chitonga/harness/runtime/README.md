# Zamai assistant runtime

The always-on window for the teacher assistant on the Beelink, outside VS Code. Built 24 Oct 2026 after Chris said
"for the assistant we need to build something like the ERP that runs outside VS Code in a separate window".
It copies the setup of an earlier WhatsApp responder (not in this repo).

## What runs

- A console window titled **"Zamai assistant"** runs `assistant.py`. It holds the OS lock `pilot_logs/assistant.lock`
  (one copy only) and runs `adapter/run.py shadow --poll 15` inside that same process.
- **Mode.** While `internal/chitonga_train/harness/PILOT_GO.json` holds Chris's words and there is no `PILOT_STOP`,
  it adds `--send-tester` and sets `ZAMAI_ASSISTANT_LIVE=1`. That is **LIVE (tester only)**: real replies go to T00
  (Chris) only. Without the go file it runs plain shadow and nothing is sent. Teachers always go to
  `pilot_logs/outbox.jsonl` until their consent is recorded. run.py still checks all three keys before every send.
- The **watchdog** (`watchdog.pyw`) runs from the scheduled task "Zamai assistant watchdog" every 10 minutes, through
  pythonw, so no window pops up. It relaunches the assistant when the process is dead, or when it is hung (heartbeat
  more than 3 min old: it kills it first). It waits 5 min for a start to finish. After 3 relaunches in an hour without a
  fresh heartbeat it stops trying. It never relaunches while `RUNTIME_STOP` exists. Its log is
  `internal/chitonga_train/harness/pilot_logs/watchdog.log`.
- The **widget** (`widget.py`) is a local page at http://127.0.0.1:8765 in a bare Edge window. It shows running or
  stopped, the mode, last inbound and today's turns per number, labels, the outbox, the last error and the watchdog's
  last line, with a STOP / START button.

## Commands (from the repo root)

| What | Command |
|---|---|
| Start | double-click `tools\chitonga\harness\runtime\launch_assistant.cmd` (or `explorer.exe tools\chitonga\harness\runtime\launch_assistant.cmd`) |
| Check | `python -X utf8 tools/chitonga/harness/runtime/widget.py --text` or the desktop icon |
| Stop (stays down) | STOP in the widget, or create `internal\chitonga_train\harness\RUNTIME_STOP` and then `python -X utf8 tools/chitonga/harness/adapter/run.py shadow-stop` |
| Start after a STOP | START in the widget (removes RUNTIME_STOP, resets the crash-loop guard) |
| Stop every send at once, keep logging | create `internal\chitonga_train\harness\PILOT_STOP` (the adapter's switch; the runtime never touches it) |
| Retrieval | `python -X utf8 tools/chitonga/harness/adapter/run.py show` |
| Install watchdog task | `powershell -ExecutionPolicy Bypass -File tools\chitonga\harness\runtime\install_watchdog_task.ps1` |
| Remove watchdog task | `powershell -ExecutionPolicy Bypass -File tools\chitonga\harness\runtime\uninstall_watchdog_task.ps1` |
| Desktop icon | `powershell -ExecutionPolicy Bypass -File tools\chitonga\harness\runtime\install_assistant_icon.ps1` (`-Startup` also opens the widget at logon; `-Remove` takes both away) |
| Tests | `python -X utf8 -m unittest discover -s tools/chitonga/harness/runtime/tests` |

## Two stop files, on purpose

- `PILOT_STOP` is Chris's kill switch for sends. The loop keeps running and logging, but it answers nothing and
  routes everything to the desk.
- `RUNTIME_STOP` stops the process, and the watchdog will not bring it back. The widget's STOP writes it and START
  removes it.

Closing the console window with X is **not** a stop. The watchdog brings the assistant back within 10 minutes.

## What survives a reboot

- **Survives:** all data (the pilot_memory and pilot_adapter SQLite files, pilot_logs/, labels, outbox), the read
  mark (no message is lost; the loop resumes from it), PILOT_GO.json, PILOT_STOP, RUNTIME_STOP, the scheduled task
  and the icon.
- **Does not survive by itself:** the window. With the task installed it comes back at logon or within 10 min (the
  task is interactive, so Chris must be logged on). Without the task, start it by hand.
  A RUNTIME_STOP left in place keeps it down after a reboot too.

## Files

- `launch_assistant.cmd` is the one hop explorer.exe runs (the job-object lesson: a child of a scheduled task dies
  with the task).
- `start_assistant.ps1` writes its log line first (`pilot_logs/runtime.log`), refuses on RUNTIME_STOP or when the
  runtime already runs, and opens the window.
- `assistant.py` runs in the window. It takes the lock and takes over a shadow loop started some other way
  (SHADOW_EXIT, waits for a clean exit, kills only after 120 s), so two pollers never run together. It writes
  `runtime_mode.json`.
- `common.py` holds the paths, the lock, liveness, the watchdog decision and the status record
  (`python common.py check|status`).
