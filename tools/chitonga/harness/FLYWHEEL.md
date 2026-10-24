# The flywheel in the harness (24 Oct 2026, the Jesuit)

Chris, 24 Oct: "I want to adjust the harness for the flywheel, getting inputs from the teachers." Built on
`internal/chitonga_train/llm/FLYWHEEL_DESIGN.md` (files, not a database; confidence 1 at once, raised by a second
speaker) and the 24 Oct rulings (English interface, Tonga is content; test-then-train; first 40 requests held out;
hidden controls; slips held out of targets but kept as noisy inputs; ok = authored weight). **Built in shadow: the
flywheel never prompts a number whose replies are really sent** (today T00 under `--send-tester`) unless
`ZAMAI_FLYWHEEL_SEND=1`. Teachers' prompts go to the outbox only.

## The loop

Teacher uses the assistant -> the engine adds at most ONE feedback question to an answer (`core/flywheel.py`) -> the
answer becomes an event in the turn record -> `adapter/flywheel_log.py` (the one writer) stores it, labelled by speaker,
consent and mode, in `pilot_logs/flywheel.jsonl`, and books any pay in the pay ledger -> once a night after 23:00
Lusaka the loop starts `flywheel_export.py` for the day -> `flywheel_out/<day>/` (training, test and planning material,
`SUMMARY.md` with the numbers) -> the main session builds the next lesson boxes and training sets from it.

## What each input becomes

| input | when | buttons / text | becomes |
|---|---|---|---|
| 1 reading label | "You want X?", "Not this one", a pick, a retype (exists: `labels.py`) | Yes / No / Not this one | `reader_sft` (strong / medium / weak), `reader_negatives` |
| 2 Tonga in context | a part showed only T/B/L lines, once per lesson per session | Tonga OK / Fix a line / Stop asking | OK: +1 confidence on those lines. Fix: "3 <your Tonga>", kept verbatim beside the served line -> translator pair + box version + word swaps as lexicon hypotheses |
| 3 contribute | a D line shown; a line held back by its checks; a unit with no material | Right as is (K1) / Write it (K2); Write it / Not now; Write a line (K2) | teacher_ok or teacher_written pair (authored weight), box update, monolingual line ("Tonga = English" optional) |
| 4 class reality | struggle reports and "Did it help?" (exists) | Helped / Partly / No | strategy evidence (memory) |
| 5 gaps | every honest "no material" (unit, request, grade; health as a count only) | none | `content_backlog.json`, by count: what we build next |
| 6 controls | about 1 in 10 contribution prompts | shown exactly as a D-line check | `controls_scores.json`: known-wrong caught / missed, known-right agreed / rewrote, per speaker |
| 7 pay | every paid answer | none | one row in `pay_ledger.tsv` (live) or `pay_ledger_shadow.tsv`, balances.tsv columns |
| 8 lesson notes (24 Oct) | the teacher's own notes per timetable slot (app: quick add, slot sheet, "Taught it"; tags worked / didn't / idea / attention / repeat) | unpaid, the teacher's own tool | (a) the teacher's weekly digest; (b) `notes_backlog.json`: what didn't work, per lesson, across teachers (content backlog, beside `content_backlog.json`); (c) "Taught it" = real pacing per class against Funda's estimate; (d) Tonga in notes: monolingual teacher Tonga, NOT a training row (below) |

**Lesson notes as input (24 Oct 2026; core/timetable.py, plugins/education/notes_digest.py).** They are the teacher's
working notes, not contributions: never paid, never prompted, never exported as training material. What flows on:
the cross-teacher backlog (names stripped by `planner.probable_names`, notes tagged attention-to-a-learner never read,
teachers as hashes, testers marked `test_only`), and the counts of taught lessons per unit (pacing reality). A Tonga
note's translator English is a reading aid cached on the note; if a note's Tonga is ever to become data, it goes
through the desk like any line (consent, a check, held-out rules), never straight from the notes table.

## Pay

K1 for "Right as is" on a D line, K2 for a line written or fixed (the desk's ok / sentence rates). Tonga OK is unpaid
(a one-tap opinion on up to six lines; it counts as feedback for the pilot plan's K2 per request). Controls are paid as
the prompt they look like. Cap: 20 paid answers per teacher per day, then no paid prompts. A tester earns K0. The
ledger is a separate file: the main session moves its rows into `desk/balances.tsv` when it pays. Proposal for Chris.

## Controls

`build_controls.py` -> `internal/chitonga_train/harness/flywheel_controls.jsonl`: 30 known-right lines (teacher-written,
15 per speaker; a teacher is shown the OTHER speaker's line first) and 20 known-wrong lines (a 24 Oct synthetic number
line beside a sibling's English: the Tonga says 10, the English says 9; real Tonga, one certain meaning error). Never
from the stage 18 frozen test or dev, never a guard hit, never exported as training. Judge a speaker by counts, not by
one answer.

## Held-out rules (asserted in the export before anything is written)

- reader labels of the first 40 requests per teacher, or with unknown held-out status, never enter reader training;
- any row whose English or Tonga hits `heldout_guard` (incl. the harness v14 test) or the stage 18 frozen 44 / dev 14
  is dropped from every training file;
- a suspected slip (a word attest.py cannot place) is never a Tonga target: kept as `toi>en` noisy input, and the word
  goes to `lexicon_candidates` as a hypothesis;
- controls, testers' records and speakers without recorded consent never become training rows (`excluded.jsonl` says
  why); a printed line's fix is a variant of the print, never a pair with our unverified gloss.

## Friction rules

One question per answer at most; none on safeguarding, held, health / status, redirect, withheld, encouragement or
error turns, nor when "Did it help?" is asked; none right after a question was passed by; quiet for the session after
three passed by; "Stop asking" / "feedback off" turns them off (kept across "start over"; "feedback on" restores);
any new request or menu word closes a waiting Fix / Write. Safeguarding runs before a waiting Fix reads the text.
Numbered menus continue their numbers across a turn's menus (lesson 1-3, question 4-6).

## The web app (DECISIONS 24 Oct: a page, not WhatsApp)

The flywheel sits in the engine and speaks in the engine's reply dicts (buttons with `fw:` ids), so a page renders the
one question as inline buttons under the answer with no change here; the page's server calls
`FlywheelWriter.capture(number, at, turn, inbound, request)` exactly as the WhatsApp adapter does. Offline: a queued
answer keeps its `pid`; an answer whose prompt has closed gets "That question has closed" and is not stored. The
numbered-menu continuation is WhatsApp only.

## Commands

`run.py show [--who T09] [--base DIR]` (records, controls, ledger) · `run.py export [--since DAY] [--base DIR]` ·
`flywheel_export.py --day D | --all [--include-testers] [--no-attest]` · `build_controls.py` ·
tests: `adapter/tests/test_flywheel.py` (21).
