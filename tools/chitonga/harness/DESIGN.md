# The harness: one core, plug-ins as data (v0, 23 Oct 2026, the Jesuit)

Chris, 23 Oct: "let's work on the harness now, a main/base harness, plus the primary education plug in"; "the system
needs to remember the teacher, just as an ERP remembers different customers"; "Not starting from 0 every time".
DECISIONS 23 Oct: behaviour lives in code, the model only READS; content is prepared in advance; plug-ins are data.

**Core** (`core/`, shared by every plug-in; transport-agnostic, WhatsApp-shaped):
- `io.py` replies as data: text, up to 3 buttons, or a list of up to 10 rows (longer lists page with "More").
- `engine.py` the state machine: greet > (name, language, class) > task offer > subject > grade > unit > deliver (in
  parts) > "anything else"; hard day > encouragement > offer; out of scope > fixed redirect; safeguarding > fixed
  referral; struggle > intervention / challenge > follow-up once. State persists in the memory: a teacher who comes
  back resumes ("Last time: ... Continue?"), an interrupted path resumes at its step, a finished one offers the next
  unit; "start over" resets.
- `intake.py` deterministic first (taps, keywords and phrasings from the plug-in, the teacher's own earlier requests),
  then a pluggable READER for free text returning the fixed structure `{intent, plugin, item, unit, slots, confidence}`,
  where `item` must be one of the retrieved candidates. Backends: `RulesReader` (default, no model),
  `TranslatorGemmaReader` (translator toi>en, then the untouched Gemma on llama.cpp 8091 with constrained JSON; not
  run by default), later a trained reader (the chat run plugs in here). Low confidence, or a reading that changes the
  answer, goes to a confirmation step with buttons.
- `gates.py` safeguarding first (the assistant's verified referral: 116, 991, head / guidance teacher, DSWO), then
  the DENY scope list, then the plug-in's domain rules (e.g. health only from a checked card); fixed lines only.
- `retrieval.py` lexical search over the plug-in's items (the assistant library's stemmer; vectors never admit).
- `render.py` templates with slots; every Tonga line carries its origin (T teacher, B book, L lexicon, D derived:
  needs confirmation); Tonga not written by a teacher passes the checks (`checks.py`: numbers, plug-in glossary
  forbids, and, when a backend is given, attest and negation). A failed line is shown in English with a flag.
- `memory.py` teacher memory keyed by the WhatsApp number (normalised 260 + 9 digits, as an earlier
  business responder keys its counterparties): name, school / district, language preference, consent note, classes (1:n; label, grade(s)
  incl. multigrade, subjects, size), syllabus position per class and subject, class-level observations, strategy
  suggestions and their outcome, recent requests (intent and item only), open items, events. SQLite under
  `internal/` behind an interface (`MemoryStore`) so it can move to Supabase. **Never stored:** pupil names, marks or
  cases (a pupil-level observation is refused and not stored); a safeguarding disclosure (only the referral event).
- `log.py` one JSON line per turn: input, reading, path, gates, reply, origins, timing.
- `cli.py` a simulator to talk to the engine on the Beelink.

**Plug-in** (`plugins/<name>/` code-free hooks; data in `internal/.../plugin_<name>/`): `manifest.json` (name, version,
languages, grades), `menu.json` (grade > subject > unit, each unit listing its items or saying honestly that it has
none), `items.jsonl` (cards, lessons with Tonga lines and origins, pointers to human Tonga material, routes, and since
24 Oct the `syllabus` and `sample` items below), `strategies.jsonl`
(per lesson: common mistakes, teacher moves, checkpoint, intervention and challenge steps), `templates.json` (fixed
lines, English meaning + Tonga only where a teacher confirmed it), `glossary.jsonl` (teacher terms; feeds the term
checks), `rules.json` (domain rules, holds), `intake.json` (keywords and phrasings), `testset.jsonl` (frozen held-out
requests). `core/plugin.py` loads and validates all of it; a plug-in that fails validation does not load.

**Education plug-in v0 (Grade 1):** cards v0 + v1 (2 holds respected), the 13 Grade 1 lessons + place value as lesson
objects, the Chitonga index as pointers (the subject is not translated), the teacher glossary of 22-23 Oct, menu =
English, Chitonga, Maths, Science > the GRADE1_MAP units, strategies from the Funda guides (English, Funda ai_draft;
Tonga only where the box already has the line). Built by `plugins/education/build_plugin.py` from the sources, so a
new teacher answer is one rebuild away.

**The syllabus as items (24 Oct 2026, the Jesuit).** One item of kind `syllabus` per specific competence of the Grade 1-3
syllabus: 436 (English 45 / 47 / 45, Mathematics and Science 20 / 25 / 26, Creative and Technology Studies 34 / 32 / 37,
Chitonga Grade 1 125), built by `plugins/education/build_syllabus.py` (called from build_plugin.py) from the Funda JSON
(`internal/funda/content/curriculum/2023/lower-primary/`, Nyanja skipped). Fields: grade, area, topic / sub-topic /
competence numbers and titles, `code` (Funda's position, the form cards use in `syllabus_ref`), `cdc_code` (the code
the book prints, where Funda renumbered: 23 rows), learning activities, expected standard, `pacing` (term, week, weeks
needed, weight, foundational weight: FUNDA'S ESTIMATE, the CDC book has no per-topic pacing), `prov` per field (`cdc` =
verbatim in the book, `cdc_edited` = Funda's transcription differs, checked by `syllabus_check.py` against the decoded
CDC 2024 book: titles 298/311, expected standards 296/311, activities 703/824 verbatim), `links` (cards by their
2024-book `syllabus_ref`, lessons and cards by their menu unit's ref; a conservative title lead only where no code
link exists; Chitonga by the module's own section heading), `covered` / `covered_by` (`code`, `code_broad` = only an
item spanning more than four competences, `title`). Chitonga items are the printed modules' Tonga (B), marked
`auto_extracted`; Funda's Claude-written Tonga (why_it_matters, checkpoint) is not carried. `syllabus_report.json`
holds the coverage and the link report. Syllabus items are never in a menu unit and are OUTSIDE the default retrieval
index (`core/retrieval.py DEFAULT_KINDS`), so the rules reader, the model reader's candidates and the trained chat
round's prompt are unchanged; an index built with them ranks every competence below the qualifying material. A request
ABOUT the syllabus (strong words: syllabus, curriculum, competence, learning activities, expected standard, scheme of
work; soft cues only without a keyword unit: "what should grade 2 do in maths this term", "week 5 of term 2"; never
for a class struggle or a named grade outside 1-3) is the education plug-in's intent `syllabus` -> `route:syllabus`
(hooks.py), answered by `plugins/education/syllabus.py` (topic search, else the term / week view, Funda's pacing
labelled, "this term" from the month). `Engine._route` passes `ctx` (plug-in, class grades, clock) to a hook that takes
it. Web: the grade bundle carries the syllabus; four GET endpoints; search adds a separate `syllabus` key
(`web/SYLLABUS_API.md`).

**Sample lessons (24 Oct 2026, the Jesuit; Chris: "We can include sample lessons from the funda repertoire", "Where we
don't have Tonga words, we show it", "Funda has two streams, the old curriculum and the new one with forms").** One item
of kind `sample` per Funda teacher guide that no Grade 1-3 lesson uses: 52 (maths 5.1-7.2: 15, English E.4.1-E.6.2: 18,
science S.4.1-S.6.2: 19) plus the maths set's introduction (`role: set_intro`, not in the menu), built by
`plugins/education/build_samples.py` (called from build_plugin.py AFTER the syllabus and the intake, so neither sees
them). The number prefix is a GRADE for maths (the set's introduction: "5.3 is Grade 5, Lesson 3"; one grade per
masthead) and a Funda LEVEL for English and science (the masthead prints the target band: E.6.1 "G5-G7", S.4.9
"G4-G7"); `grades` = the band from 4 to 7, `prefix` says which and quotes the masthead. Fields: title, `label_short`,
subject, grades, `band`, `guide_code`, `teaches` (the guide's subtitle), `what_it_is`, `guide` {funda path, `html` =
the body through `web/guide.py` sanitise_funda, label, sha256}, `strategy` (the existing parse_strategies, embedded in
the item and NOT in strategies.jsonl, so the class-struggle matcher never reaches a Grade 4-7 guide), `key_terms`
[{en, toi, origin, badge, source, via, tonga_needed, note}], `curriculum` + `level` (the primary placement),
`placements` [per band grade: 2023 Grade N only where Funda's 2023 upper-primary JSON carries the topic (that sub-topic
in `refs` and `syllabus_ref`, each marked as Funda's ai_draft, backward-designed from the G7 paper, CDC unpublished);
else 2013 Grade N; Grade 7 always 2013, with `in_2023` = Form 1 where the Form 1 JSON carries it], `related_refs`,
`health` (S.4.9, S.5.10, S.6.2; Chris 24 Oct: "Add them in and tag them as need review": these three, and only
these, carry `needs_review` + `review_tag` + `review_note` and are named in rules.json `health.review_allowed`, which
`gates.item_allowed` lets through for a SAMPLE that carries the tag; the loader refuses needs_review anywhere else and
any review_allowed name that is not such a sample; every other health item, the cards included, still needs
`checked`), origin F, label "Sample lesson ... AI draft ...
not yet reviewed". Tonga in key terms ONLY from a teacher (glossary T.., lexicon.tsv rows a teacher attests) or a
printed source in the same sense (glossary B, lexicon.tsv dictionary / school-book rows whose English is the source's;
rows we guessed at c < 0.9, rule-built rows and held-out posters never), every hit read in context (`SENSE_REJECT`:
'body' of a talk, 'bulongo' for the planet Earth, 'mean' as stingy ...); else `tonga_needed` and no Tonga at all.
Menu: grades 4-7 (`samples_only`, Grade 7 labelled "old curriculum, 2013"), one unit per (grade, sample), `sample`,
`curriculum`, `level`, status "sample lesson", ref "2023 curriculum <code>" or "old curriculum (2013)", 2023 units
first. `core/plugin.py` checks a sample (origin F + label; no Tonga lines; every key term either Tonga + origin T/B +
source, or tonga_needed with none; Grade 7 never 2023; a menu unit with a sample says so). Retrieval: `sample` is in
DEFAULT_KINDS but outside the IDF (`n_idf`), so card / lesson / pointer scores are exactly as before; a sample not
for the request's grades ranks after all of them (`retrieval.off_grade`), the rules reader never takes one as its lead,
and `reader_model.candidates()` (unchanged) keeps card / lesson / pointer only. WhatsApp: `render.render_sample`
(overview with the placements, 2023 first; key words with "Tonga word needed"; the guide's checkpoint). Web: the grade
4-7 bundles carry them (`kind: sample`), search returns them in a separate `samples` key (`web/SAMPLES_API.md`); the
Netlify demo's keyword search does not index them. `samples_report.json` holds the counts.

**Grade 4, the whole 2023 syllabus (24 Oct 2026, afternoon, the Jesuit; Chris: "Can we do full curriculum from grade
1-4?").** Checked first: the CDC has NOT published the 2023 Grade 4 syllabi for English, Mathematics, Science or Social
Studies (edu.gov.zm Directorate of Curriculum Development list, ?page_id=1142: Upper Primary only Expressive Arts and
Technology Studies; ?page_id=8082 Primary: "Documents Will be added soon!"; corpus: none). So `build_syllabus.py` adds
Funda's seven Grade 4 files (156 competences): Expressive Arts and Technology Studies from the CDC books, checked field
by field by `plugins/education/syllabus_check_up.py` against the February 2025 uploads (Technology near-verbatim;
Expressive Arts titles mostly, activities rewritten by Funda; printed codes 4.1.2.2 and 4.4.3.2); HEH from the CDC
2024 book, unchecked (not on this machine), printed codes from Funda's note (the book numbers 4.8 twice); English,
Maths, Science, Social Studies as `origin F`, `draft_label` "Funda draft, not a published CDC syllabus" on every item.
Pacing only as Funda's file has it, labelled. Links at Grade 4: cards citing a Grade 4 code of these books (printed
first; the 2013 syllabus's look-alike codes never link), samples only where reading the guide showed they teach the
competence (`SAMPLE_COVER`, 4), cards citing another book but teaching it in full (`READ_COVER`, match `code:read`, 6),
partial material as `related` (never cover). **Printed-code clashes** (the coordinator's find, G2 CTS 2.11.1.1;
also HEH 4.8.1.1): `Codes.by_cdc` keeps every competence under a printed code and resolves by the card's title, else the
Funda code; listed in `syllabus_report.json printed_code_clashes`. **Cards**: `build_plugin.load_cards` reads cards_v0,
v1 and every `cards_v2_*.jsonl` (the Grade 1-3 gaps c303-c478, Grade 4 c2000-c2119), each v2 row checked by
`check_card_v2` (the v1 schema + `key_words_en`, origin C, status draft, optional hold "health review"; no Tonga field;
a non-JSON line, i.e. a file still being written, or a bad row is left out and listed). `key_words_en` reach the card
view as key words with "Tonga word needed". A `"health review"` card becomes a health item (`health`, `review_hold`):
the existing rule withholds it ("no checked health material") and it is no cover; only a body-safety / abuse topic
becomes a referral hold (an item `hold` always answers with the safeguarding referral and logs a referral event).
**Ranking**: new Grade 4+ cards carry `rank_by_grade` and follow the samples' three rules (outside the IDF, after all
other material for a request of another grade, never the rules reader's lead there): measured, without it the frozen
test SC05 ("How many provinces are in Zambia", unknown grade) left its accepted no-material answer for the Grade 4
provinces card; with it every intake-eval row is identical to before. **Menu**: Grade 4 is real material (not
`samples_only`): subjects english, maths, science, social, arts, home_economics, technology; one unit per syllabus
topic with the linked cards in competence order, the grade's sample units after the topic they name (`grade4_menu`).
The Netlify export keeps `rank_by_grade` cards out of the demo keyword index (as samples), so parity holds.

**Labels, not content holds (24 Oct 2026; Chris: "Nothing is held back, just label it for review").** No item is
withheld for its CONTENT any more. `build_plugin.label_content_holds` runs before the syllabus links and labels every
item the gates would have withheld: `needs_review`, `review_tag` "Needs review", `review_note` per reason, and
`review_reason` (`health_unchecked`: health cards no one has checked, cards_v0 13 + cards_v1 22; `health_review`:
cards_v2 "health review" cards, g13 6 + g4 15; `child_protection`: the safeguarding-topic holds c171 / c227, whose
`hold` is dropped and kept as `former_hold`; `content_hold` for any other hold: none today), plus the 3 health samples
(build_samples.py labels every health sample). Counts in the manifest (`needs_review`). `gates.item_allowed` serves a
labelled item ("ok"); the rule's `hold` and `<rule>_no_card` branches remain for an unlabelled item, and
`core/plugin.py` refuses `needs_review` without its tag and note and refuses any content `hold` in the data (the
earlier `review_allowed` whitelist is gone). Every front-end shows the label: `render.review_line` on every WhatsApp
part of every kind, the web views / units / search results, the Netlify meta; no feedback prompt on a labelled item;
a labelled card counts as syllabus cover with `needs_review` on the link. UNCHANGED, because they decide how a MESSAGE
is answered and hide no material: the safeguarding referral on a teacher's message (116 / 991 / DSWO), the scope deny
list, the engine's honest answer to a health-worded message no item matches (`<rule>_no_card` gap), and the
request-wording branch that does not answer a health-worded request with an unlabelled unchecked card (no education
item reaches it now: every health item is labelled).

**Grade-fit content and every card in a menu (24 Oct 2026, evening; persona review REVIEW_PERSONA_2026-10-24.md M2,
and the search agent's count of cards no menu reached).** `build_plugin.content_pass`: (1) a lesson whose Funda guide is
pitched ABOVE it (the guide's masthead grade is higher than the lesson's, or a maths guide's worked example uses numbers
beyond the lesson's scope line, e.g. 347 + 286 in "Addition to 99") no longer shows the guide as its body:
`guide.funda = None`, `guide.older = {funda, band, label "For older classes: sample lesson (Grade N)", why}`, shown
apart (`web/guide.older_for`, the view's `older_guide`; WhatsApp names it in the overview); the lesson keeps its own
card / scope line and its Say-it-in-Tonga box; a checkpoint with numbers beyond the scope moves to `checkpoint_older`;
the strategy's mistakes and moves stay (the class-struggle matcher reads them) with an `older_guide_note`. (2) Exam
talk (ECZ, PSLE, the Grade 6 Composite, Grade 7 papers) is removed from all Grade 1-4 material: guide HTML at serve
time (`guide.strip_exam_talk`), cards, lesson cards, strategies, samples at build time. (3) Every written calculation in
the served English is recomputed (`plugins/education/content_checks.py`); a wrong one must be corrected in
`web/funda_errata.json` (applied by `guide.read_funda` wherever a guide is read; Funda's copy is never edited) or listed
in `content_checks.INTENTIONAL` (a learner mistake shown on purpose), else the build reports a problem. Report:
manifest `content_checks`. `build_plugin.place_cards` (after the syllabus, so it feeds no link) puts every card no menu
held under its grade > subject > unit: the unit already holding material for the same sub-topic, else the same topic,
else a new unit per syllabus topic; cards citing the 2013 syllabus get a unit per cited topic; a missing subject is
added to the grade; a grade that receives cards stops being samples-only. Report: manifest `cards_placed`.

**Fixed lines in Tonga (23 Oct, stream F):** the English of every fixed line and button stays in the plug-in's
templates.json; the Tonga is the CORE's (`core/fixed.py`, data internal/chitonga_train/assistant_library/
core_fixed_lines.json, built by `core/build_fixed_lines.py`: teacher lines reused verbatim, else translator a050 as D,
checked). Served in Tonga only when teacher-written (T) or confirmed; B / L / D wait in
desk/fixed_lines_confirm_2026-10-23/ (8 a message, not sent). The memory's language preference picks en / toi / both
(`Engine.t`, `Engine.b`); the simulator's drafts flag shows the waiting lines tagged. The referral is not in the table.
The assistant addresses one teacher with the respect plural mu- (reasons in build_fixed_lines.py), asked once at the desk.

**Interface language (24 Oct, Chris: "the harness can be all English; it can offer Tonga materials and explanations"):** `Engine(tonga_ui=False)` is the default: the fixed lines and buttons are English for everyone, the language preference governs only content (lesson Tonga boxes, explanations); the Tonga fixed-line table stays, switched off.

**WhatsApp adapter, pilot mode (24 Oct):** `adapter/` (PILOT_PLAN.md), tested in simulation, not live.

**Not here:** live WhatsApp wiring (design: WHATSAPP_ADAPTER.md) (the engine returns reply objects; a transport adapter maps them to the Cloud API later),
model rendering of new Tonga at run time (the old pipeline still does that and keeps working), deployment.

**24 Oct 2026: the architect's review (internal/chitonga_train/llm/reviews/HARNESS_ARCH_REVIEW_2026-10-23.md), items
1-3 and 5-8 built, 9-10 in part.** One turn at a time per number, and the language and active plug-in live in the turn's
context (a ContextVar), never on the Engine (1). Every reading is validated in the engine; a model reading always
confirms (2). "Not this one" on part 1 of a lesson read straight from free text: the tap logs `label.kind =
reader_error` (held-out material for the reader) and cancels recall for those words (3). The state carries the
plug-in's fingerprint (version@built); after a rebuild an unfinished path drops to the menu with "the material was
updated" (5). Several plug-ins: `Engine([education, other])`, a rules-only router (best confidence, ties stay), the
plug-in as the first menu level when more than one is loaded, flat (non-grade) menus (`menu.topics`), plug-ins without
class memory (`memory_unit: null`); proven with the TEST-ONLY `plugins/health_demo` (two checked placeholder cards,
refused unless `allow_test_plugins=True`) (6). Education intents moved to `plugins/education/hooks.py` (`detect()`,
`INTENTS`); any `route:` item is a route; `gates.status_rules` is the generic topic-requires-status rule (education's
`health` block read through a shim until build_plugin.py writes `status_rules`) (7). `classes.kind` / `classes.plugin`
(migration on open), one transaction per turn with a savepoint around `_turn`; a failed turn rolls back its rows and
its state (8). Core lines and buttons in `core/templates_core.json`, plug-ins override (9, in part: fixed.py still
matches Tonga by English, now the core's). Gemma timeout 20 s; teacher origins from the manifest; the adapter's window
check before the engine call (WHATSAPP_ADAPTER.md section 5) (10). Item 4 (pupil guard as a gate) was replaced by Chris's
pupil layer decision (DECISIONS 23 Oct); that layer is NOT built yet (see the LOGBOOK, 24 Oct): the guard behaves as on
23 Oct. Known limits: the pupil guard is English-only (a Tonga line naming a learner passes); a plug-in that owns a DENY
category (personal health) would need a declared exemption.

**Timetable, calendar, lesson notes, weekly digest (24 Oct 2026, the Jesuit).** Chris: "teachers would appreciate if they
could enter their Stundenplan and they can click on lessons in a calendar to keep notes ... analyzed by the AI (no
personal notes, just about what worked, didn't work, ideas, kids that need special attention etc)"; "Term dates are
standardized in Zambia, you can preload them, including public holidays"; the app "smart, easy, not overloaded"
(Microsoft To Do). Back end only (UI: web/UI_SPEC_TIMETABLE.md; contract: web/TIMETABLE_API.md).
- `core/school_calendar.py` (generic): term of a date, teaching week, holidays, breaks, the teacher's own dates winning
  field by field; a blank official date stays blank (flagged `term_end_to_confirm` / `term_dates_to_confirm`), never
  guessed. Data: the plug-in's `school_calendar.json` from `plugins/education/build_calendar.py` (every term date and
  holiday with status official / reported / computed / to_confirm, its source and the day it was checked; the Public
  Holidays Act's Sunday rule; SI 73 of 2026 moved the Day of Prayer to 29 December, 18 October 2026 not observed).
- `core/timetable.py`: periods, term overrides, notes and digests in the SAME memory file, behind their own class (the
  MemoryStore interface the engine uses is unchanged). Offline-first: client ids, last writer wins on the client's
  `updated_at`, tombstones. Notes may name a learner (Chris's rule): kept as written in that teacher's rows only.
- `plugins/education/planner.py`: dated slots from timetable x calendar (holidays and breaks skipped); each future slot's
  suggestion = the teacher's planned pick, else the resume chain (where the class left off, projected one lesson per
  period in menu order), else the "Funda estimate" (pacing sub-topics and their served lessons, always labelled); Today;
  the Suggested tray (next lesson per class, a strategy or tip for a "didn't work" note by retrieval, lessons marked
  repeat); quick add (rules: class by label / grade / the period on now, tags by phrase, lesson by the index; then
  optionally the standard model picking only from the lists); "Taught it" (moves the class's memory position: the same
  one the lesson pages and WhatsApp use).
- `plugins/education/notes_digest.py`: the AI part, a Beelink heavy job under the lock "notes-digest", on demand and
  nightly: Tonga notes through the served translator (cached on the note), the untouched Gemma 4 E4B reading English
  under a JSON schema whose note ids are an enum, code checks that drop any item without a supporting cited note,
  retrieval matches to strategies / tips (the model may pick one candidate or none), a digest marked "AI summary", and
  the cross-teacher "what didn't work per lesson" backlog (internal; names stripped, learner notes never read).
  Measured on 20 test notes: LOGBOOK 24 Oct (timetable entry).
- Web: `web/timetable_api.py` (a mixin of WebAPI: 7 GET endpoints, quick add, digest run, 7 sync op types); the live
  demo: `proposals/netlify_assistant_timetable.patch` (a port of calendar, store, planner and quick add for
  assistant.mjs; notes in monthly Blobs shards; admin pull / push for the Beelink digest via `web/netlify_notes_sync.py`);
  `web/export_timetable.py` writes its data and the parity scenario (`web/tests/netlify_timetable.mjs`).

**Run.** `python -X utf8 tools/chitonga/harness/plugins/education/build_plugin.py` (rebuild the data, ~30 s with attest);
`python -X utf8 tools/chitonga/harness/tests/test_harness.py` (92 tests, ~40 s) and `tests/test_syllabus.py` (40, the
syllabus items, Grade 4, the printed-code clash, the cards_v2 loader; `plugins/education/syllabus_check_up.py` re-checks
the Grade 4 CDC books; `plugins/education/syllabus_check.py` re-checks the Funda text against the CDC book) and
`tests/test_samples.py` (28, the sample lessons; `python -X utf8 plugins/education/build_samples.py` prints every key
term with its Tonga or the need, for reading in context) and `tests/test_timetable.py` (29: calendar, store, planner,
quick add, digest with stub models, web API and HTTP);
`.../plugins/education/run_testset.py`
(end to end, rules reader); `python -X utf8 tools/chitonga/harness/cli.py --db :memory: --log none` (talk to it).

## Later versions (not in v0.5)

Chris, 23 Oct 2026: "It's all a version 0.5, we can also just mention what we think a later version should include."
- **Pupil layer** (design target: a Ministry of Education system that holds pupil data anyway): pupils per class (name or
  school ID), checkpoint results per foundation concept and date (mastered / developing / not yet, from the lessons'
  checkpoints and Funda's mastery loop), small intervention groups by the exact gap, a challenge group, export for a
  ministry system. A teacher sees only their own classes.
- **Talent watch**, not a "gifted" tag: identification through opportunity (a challenge task for every pupil, so
  hidden talent can show), above-level testing in the talent-search tradition (Stanley's SMPY: early masters get the
  next grade's checkpoints), Renzulli's three rings (teacher notes on persistence and creativity next to results); an
  evidence trail over time leading to challenge work, peer teaching, enrichment and referral for further support.
- **Demo data** for any showcase: fictional classes of about 60, marked DEMO, including a pupil who is not ahead at
  first but shines on the challenge tasks.
- WhatsApp adapter (WHATSAPP_ADAPTER.md), voice, and further plug-ins (health, agriculture, councils) on the same core.
