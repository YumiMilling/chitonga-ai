"""Settings of the teacher-assistant answer pipeline (22 Oct 2026, the Jesuit; council path step 4, Chris approved the
path: "we explain, don't just answer").

Every value says who set it and why. Change a value only with a measured reason and write the reason here.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
MODELS_DIR = os.environ.get("CHITONGA_MODELS", r"models")   # env: a pod run (22 Oct)

# The renderers by short name -> the model directory name under MODELS_DIR (translator_service.SOUP is swapped for it,
# as tools/chitonga/train/negation_sweep.make_service does). R = the released stage 13 soup (DECISIONS 18 Oct).
# a050 = the 50% blend R/stage 16 that DECISIONS 22 Oct prefers for the teacher assistant, pending the negation sweep
# (DECISIONS 22 Oct "NEGATION SWEEP BEFORE INSTALLING A BLEND"): until the sweep says so, R is the default.
RENDERERS = {
    "R": "chitonga-nllb-s13-soup",
    "a050": "chitonga-nllb-wise-a050",
    "a070": "chitonga-nllb-wise-a070",
}

CFG = {
    "renderer": "a050",              # 22 Oct afternoon: the translator is now the 50% blend (translator_service default,
                                     # coordinator's brief for this build); R stays fallback and read-back checker
    "fallback": "R",                 # DECISIONS 22 Oct: "R kept as fallback" when a blend is installed
    "readback": "R",                 # the back-translation checker stays R whatever renders (as in the negation sweep)
    "readback_min": 0.70,            # bge-small cosine, English source vs read-back. 0.70 is the alarm line the 504-line
                                     # comparison used (LOGBOOK, ~line 4070: E>T >= 0.70 for 69-74% of lines). An ALARM
                                     # line, not a measured gate for this content: calibrate on the first real run.
    "max_attempts": 4,               # translator beams tried per sentence before it is left in English
    "fail_verdicts": ("unattested", "not_tonga", "contact", "mixed"),   # attest verdicts that fail an OUTPUT sentence
                                     # (books are the norm for form; mixed / contact = leakage from Bemba / Nyanja;
                                     # brief_parts/related_languages.md). 'english' fails unless the token is allowed.
    "allowed_tokens": ("km", "m", "cm", "mm", "kg", "g", "l", "ml", "k", "h", "min", "s", "x"),  # units copied as typed
    "library_paths": [
        os.path.join(CT, "assistant", "library", "cards.jsonl"),          # the approved / reviewed curriculum cards
        os.path.join(CT, "assistant_library", "cards_v0.jsonl"),          # the 40 English cards v0 (22 Oct; draft)
        os.path.join(CT, "llm", "council", "codex"),                      # Codex drafts: *curriculum_library*.md|jsonl
    ],
    "advice_dir": os.path.join(CT, "advice", "cards"),                   # the 10 advice cards (draft v1, 20 Oct)
    "library_min_cover": 0.5,        # share of the request's content-word weight a card must cover (lexical retrieval)
    "library_min_topic": 1,          # and at least this many content words shared with the card's topic line
    "health_requires_card": True,    # a health topic is never drafted by the local model (MoH / WHO material only):
                                     # Codex prompt 22 Oct "health content must match Zambian MoH / WHO guidance"
    "part_limits": {"answer": 2, "idea": 4, "local_example": 2, "how_to_teach": 5, "check_questions": 3,
                    "try": 3, "why": 1, "avoid": 2},   # sentences per part (ASSISTANT_PERSONA check 4: short for a phone)
    "answer_budget_s": 60,           # council synthesis §3 step 4 kill test: > 60 s per answer
    "gemma_url": os.environ.get("ASSISTANT_GEMMA_URL", "http://127.0.0.1:8091"),  # llama-server, untouched instruct
    "gemma_label": "gemma-4-E4B-it Q4_K_M, untouched (unsloth/gemma-4-E4B-it-GGUF, sha256 85a896a0..., "
                   "models/base; llama.cpp b11574 Vulkan on the iGPU, port 8091)",
    "gemma_timeout_s": 300,          # 21.5 tokens/s on the 780M (22 Oct): a 900-token draft is ~45 s
    "gemma_on": True,
    # Claude via the Claude Code CLI on the subscription (coordinator 22 Oct: the API key answers 401; Chris: "use the
    # subscription for now"). One call per request, 120 s timeout, from an empty folder (no CLAUDE.md, no tools).
    "claude_on": True,
    "claude_cli": os.environ.get("CLAUDE_CLI", "claude"),
    "claude_model": "opus",          # the CLI alias for the latest Opus (claude-opus-5-5 on 22 Oct)
    "claude_timeout_s": 120,
    "claude_cwd": None,              # None = %TEMP%/zamai_claude_empty (must stay empty)
    "dual_log_path": os.path.join(CT, "assistant", "dual_drafts.jsonl"),  # Claude vs Gemma, one line per request
    "render_cache_path": os.path.join(CT, "assistant", "render_cache.json"),
    "glossary_path": os.path.join(CT, "assistant_library", "kept_terms_glossary.jsonl"),
    "log_path": os.path.join(CT, "assistant", "answers.jsonl"),          # this pipeline's answer log (one writer)
    "set_by": "Jesuit 22 Oct 2026, council path step 4; lexicon / template / dual composer routes added 22 Oct pm",
}
