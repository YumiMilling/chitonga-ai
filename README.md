# chitonga-ai

Code and method from a proof of concept: a teacher assistant for Chitonga (Tonga of southern Zambia, ISO 639-3
`toi`), built by one person working with Claude Code between 10 and 24 October 2026, on a small office PC and rented
GPUs by the hour.

This is a research prototype. It is not a product, and nothing here has been checked by a child-protection
professional.

Paper 1 (preprint, October 2026): Lindinger, C. J. *No Data Centre Required: Adapting Open Models to Chitonga with
Paid Bilingual Speakers, Printed Books and a Verifying Harness.* Zenodo. All versions:
https://doi.org/10.5281/zenodo.23487562 (version 2, 25 October 2026: https://doi.org/10.5281/zenodo.23642991).
Paper 2: *Local AI for Zambia 1*, https://doi.org/10.5281/zenodo.23642662

## The architecture

Three parts. Only the translator was trained on Chitonga.

1. **The harness** (`tools/chitonga/harness/`). Deterministic code around the models. It does the following:
   - reads a teacher's WhatsApp message;
   - runs the safety gates first (safeguarding, health, scope);
   - finds checked material in a plug-in (lesson cards, syllabus, sample lessons);
   - keeps a per-teacher memory;
   - renders the reply.

   A plug-in is data, not code (`DESIGN.md`). The flywheel (`FLYWHEEL.md`) turns teacher input into checked data.
2. **The translator** (`tools/chitonga/llm/translator_service.py`). NLLB-200 1.3B, fine-tuned for Chitonga and served
   with CTranslate2 int8 on CPU. Every line of Chitonga goes through it, in both directions, along with a set of
   checks:
   - an n-best rerank;
   - number and negation round trips;
   - an attestation check against printed Tonga;
   - a forbid-only term guard (`term_guard.py`);
   - an exact-match translation memory (`h2_memory.py`).

   These checks raise alarms instead of rewriting.
3. **The reader**: a standard instruction model that has had no Chitonga training (Gemma 4 E4B, Q4_K_M, llama.cpp).
   It reads the translator's English and fills a fixed JSON schema. It never writes Chitonga.

The route was settled on 24 October 2026. Chitonga is trained into the translator only. The chat model is a standard
model, and its job is to read English.

## What we found

Every decision below was tested against a rule written down before any output was read. Each rule had a kill test,
and two seeds where training was involved.

- **Compute is not the bottleneck. Human-checked data is.**
  - The stage 19 kill test ran five candidates on five rented GPUs for about USD 4.75.
  - The reader bake-off (20 model arms) cost about USD 2.
  - What limited every round was the number of Chitonga lines checked by a native speaker.
- **A random gate found a selection effect.** The stage 19 candidate (1.3B) beat the installed translator on the
  frozen set of teacher lines, at 21:11 and 23:15 on two seeds. On 40 fresh, randomly drawn classroom lines it lost,
  6:13 and 2:14. The gain did not carry over to new lines. Without the random gate it would have been installed.
- **3.3B was no better.** NLLB-200 3.3B, trained once with the same data:
  - did worse on the main 174-line set on every path;
  - did worse on documents;
  - was about level on the teacher set;
  - lost on the random lines (7:18, 7:15, 6:14).
- **Domain tags did not work on 1.3B.** A `<cls>` (classroom) tag against no tag scored 20:9 and 13:10 on the teacher
  set, but 3:9 and 2:3 on the random lines.
- **Serving levers are free; use them first.** The term guard, the glossary preference inside the n-best, and the
  exact-match memory need no training. They fix named, dangerous errors at serving time, for example *dizzy* turned
  into *diarrhoea*, or *cholera* into *corona*. Training rounds did not fix those errors.
- **Use the translation memory for exact matches only.**
  - With a loose "near match" rule, 7 of 7 near rows admitted had a different meaning: star/store, door/window,
    stone/toilet, sun/ones. Every check still passed them.
  - With the strict gate, 0 were admitted.
- **An LLM judge is an alarm, not a measure.** On Chitonga, LLM judges were kinder than native speakers in our own
  record. We use them to find errors and route them to speakers, never as the score.
  - The weakness map (22 October): 500 unseen English>Chitonga lines from the installed translator, one blind LLM
    judge, dynamic equivalence. Result: 216 right, 238 partly, 46 wrong.
  - The speaker test (24-25 October): 170 random Grade 1-4 classroom sentences, judged blind by two bilingual
    speakers under a sealed rule with hidden controls. 36% were marked right as written and 68% right or needing only
    a small fix. An AI judge called 53% of the same sentences right.
- **The bigger reader read no better.** The bake-off asked which standard model should read the translator's English.
  The task was English only, clean and noisy, 1,164 requests. Results, correct out of 1,164:

  | Arm | Correct |
  |---|---|
  | Rules only | 517 |
  | Gemma 4 E4B (incumbent) | 613 |
  | Gemma 4 12B | 632 |
  | Qwen3.5 9B | 616 |
  | Qwen3.5 4B | 608 |

  A ~284B reference model, DeepSeek V4 Flash, run on the 1,063 rows without teacher-derived text, scored 522 against
  Gemma's 521. The ceiling sits in the harness (candidates, rules, gold answers), not in the reader. Gemma 4 E4B stays.
- **Pre-registration found a safeguarding gap.** The bake-off's safety gate failed every arm. The harness's
  safeguarding gate missed seven kinds of disclosure, whichever reader was used:
  - child marriage;
  - sexual exploitation for money;
  - abandonment;
  - a cover-up by an adult;
  - images;
  - food deprivation;
  - typo'd disclosures.

  Two of these were routed to a lesson instead. The gate was rebuilt by category (`tools/chitonga/llm/safeguard.py`),
  with nine categories, typo normalisation and benign-phrase blanking. Results:
  - the old gate found 13 of 97 disclosures on a development set we wrote;
  - the new gate finds 97 of 97 with 0 of 89 false alarms (tuned on that set, so an upper bound);
  - on a sealed second set, scored once, it found 20 of 24 with 0 false alarms.

  The gap was found because the safety rule was written before the test.

## Models

Public, on Hugging Face (each URL checked anonymously on 24 October 2026):

- https://huggingface.co/zamai-zambia (the account page).
- https://huggingface.co/zamai-zambia/chitonga-nllb-s13: the stage 13 translator, NLLB-200 1.3B, two seeds averaged.
  Its model card says: "CC BY-NC 4.0, the licence of the base model, Meta's NLLB-200: free for non-commercial use,
  with attribution."
- https://huggingface.co/zamai-zambia/chitonga-chat-run3: the three-stage chat model v0.10r3 on Gemma 4 E4B. It is no
  longer on the route, but kept as a data point. Its model card says: "CC BY-NC 4.0: free for non-commercial use, with
  attribution. The model is a derivative of Google's Gemma 4 (`google/gemma-4-E4B` and `google/gemma-4-E4B-it`), which
  is licensed under the Apache License 2.0."
- https://huggingface.co/zamai-zambia/chitonga-nllb-a050: the installed translator **a050**, the 50/50 weight average
  of stages 13 and 16, CC BY-NC 4.0.
  - Weakness map (22 October): 500 unseen lines, English>Chitonga, one blind LLM judge, dynamic equivalence. Result:
    216 right, 238 partly, 46 wrong.
  - Speaker test (24-25 October, 170 random classroom sentences, two blind speakers): 36% right as written, 68% right
    or needing a small fix. Read the LLM figures as an alarm, not a measure.

Not public: the stage 18 and stage 19 candidates. All of them failed their pre-registered kill tests. They are
available on request.

Code that loaded private model or dataset repos now shows the placeholder `PRIVATE_HF_ORG/...`. Where a public
equivalent exists, it shows the public `zamai-zambia/...` id instead.

## Repository layout

```
tools/chitonga/
  train/        NLLB fine-tuning (train_nllb.py), the leak guard (heldout_guard.py, unit_guard.py), the
                pod/droplet launchers with guard, self-delete, budget cap and overwatch (launch_*.sh, run_*.sh,
                guard_stage19.sh, pod_overwatch.py), pack scripts, the selector (f19_select.py), the stage 18/19
                kill-test runners (final18_*.py, final19_*.py), the weight average (merge_wise.py)
  llm/          translator service, term guard, translation memory, checks, the safeguarding gate, the front triage,
                the reader bake-off (reader_bakeoff.py, english_noise.py), pod server/check, unit tests (h3_tests/)
  grammar/      a small rule engine for Tonga verb derivation from Hopgood's printed grammar
  harness/      core/ (engine, gates, intake, memory, retrieval, render, timetable), plugins/education (code only),
                adapter/ (WhatsApp pilot adapter), web/ (local API and server), runtime/ (always-on window),
                tests/, DESIGN.md, FLYWHEEL.md
  assistant/    gate, library loader, backends and config the harness imports
  space/        p1_space.py (public helpers shared with the translator service)
experiments/
  speaker_test/ draw, held-out check, tagging, translation, blind AI judge, scoring (score.py) of the native-speaker
                test of the installed translator
  weakness_map/ the scripts of the 22 Oct weakness map (decode, blind judging, analysis)
  tm_near/      the near-match translation-memory measurement
```

Notes for reading the code:
- Docstrings carry dates and the reason for each decision.
- "The Jesuit" and "the architect" are the names of two Claude Code agent personas used in the work (developer and
  planner). "The coordinator" is the main Claude Code session. "Chris" is the author.
- Speakers appear by code only (T01, T05, T09, T12).
- Paths are relative to the original working copy. Folders such as `internal/chitonga_train/...` and
  `../zamai-corpus-raw/...` hold data that is not published (see below), so most scripts will not run without it.

## Tests

Run from the repository root with Python 3.12:

```
python -X utf8 tools/chitonga/llm/h3_tests/unit_term_guard.py          # 39 passed
python -X utf8 tools/chitonga/harness/tests/safeguard/score.py          # safeguarding gate, dev set: recall 97/97, false alarms 0/89
python -X utf8 tools/chitonga/harness/tests/safeguard/score.py tools/chitonga/harness/tests/safeguard/sealed_v1.jsonl   # 24/24, 0/24 (seen since)
python -X utf8 tools/chitonga/harness/tests/safeguard/score.py --old    # the old gate: 13/97, 3/89
python -X utf8 tools/chitonga/harness/runtime/tests/test_runtime.py     # 14 passed
```

The safeguarding sets in `harness/tests/safeguard/` are English sentences written for the test. They come from no
real teacher, child or case.

The other suites need the unpublished plug-in data or caches, and they stop with a missing-file error:
- `harness/tests/test_safeguard.py`, `test_timetable.py`, `harness/adapter/tests/test_adapter.py` (14 of 31 pass
  without data);
- `llm/h3_tests/unit_front.py`, `unit_scope.py`, `unit_serving.py`;
- `train/unit_guard.py`. This one refuses to run without the frozen held-out sets, by design: a leak guard that runs
  without its sets would let test items into training.

## What is deliberately not included

- **No speaker data.** No names (except the public credit above), phone numbers, payments, messages, answers or
  corrections. No Chitonga written or corrected by our paid speakers.
- **No training or test data.** No typed books, pair files, synthetic sets, lexicon, glossary, translation-memory
  contents, held-out sets or blind-test keys.
- **No curriculum material.** No Ministry or CDC text and no third-party lesson content. The education plug-in's data
  and its build scripts are left out.
- **No keys, tokens, private repository ids, machine ids or addresses.**
- **Chitonga inside the code.** Single Tonga words and morphemes stay where they are part of a rule: numerals,
  negation, concords, question words, and the child words of the safeguarding gate. They are printed in published
  grammars and dictionaries (Hopgood's grammar, Thompson 1979, the Harris dictionary). Phrases from
  Thompson's printed phrase book appear as fixed lines with page references. Anything a speaker wrote has been removed
  or replaced by the marker `[toi withheld: speaker-written]`. Where a test needed a Chitonga sentence, it now uses
  the public-domain Bible line of Genesis 1:1 or language-neutral tokens. Each such change is marked "PUBLIC COPY" in
  the file.
- **Not yet in this repository**, held back until they have been scrubbed line by line:
  - the full grammar engine (`grammar2/`);
  - the stage 18 and stage 19 pre-registered packs and kill-test results;
  - the reader bake-off rules and report;
  - the speaker-test rule;
  - the weakness-map write-up;
  - the stage 18/19 data builders and probe scripts;
  - the harness tests that quote plug-in content.

## Licence

- **Code and documentation in this repository:** CC BY-NC 4.0, © 2026 Christoph Johannes Lindinger (ZamAi). See
  `LICENSE`. This is the same non-commercial line as the translator's base model.
- **Models** carry their base model's licence:
  - the NLLB-200-derived translators are CC BY-NC 4.0, as Meta's weights;
  - the Gemma-derived chat model is under the terms stated on its model card (CC BY-NC 4.0 for the derivative; Gemma 4
    itself is Apache 2.0).
- **Third-party code and models** keep their own licences: llama.cpp (MIT), CTranslate2 (MIT), Hugging Face
  Transformers (Apache 2.0), and the reader models tested in the bake-off (each under its own licence).

## Credits

- The Chitonga in our models was checked by paid bilingual speakers in Southern Province.
- Funded by Yumi Milling Ltd.
- Contact: chris@zamai.pro.
- Written with Claude Code.
