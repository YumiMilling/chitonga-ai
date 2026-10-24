"""Fine-tune NLLB-200 (600M) English -> Chitonga on our pairs, then translate the frozen test English. Runs on the GPU server.

    pip install "transformers==4.46.3" sentencepiece sacrebleu     # keep the image's own torch; do not pip install torch
    python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"
    python train_nllb.py --run <folder with train.jsonl, dev.jsonl, test.en.txt> --tag <name> [options]

Zero-shot baseline (no training, just translate with an existing language code):
    python train_nllb.py --run R --tag zs_bem --steps 0 --tgt-code bem_Latn
Dry run first (20 steps, proves it runs):
    python train_nllb.py --run R --tag dry --steps 20 --eval-every 10 --no-save
Stage 1 smoke test (two seeds: a difference smaller than the gap between them is noise):
    python train_nllb.py --run R --tag ft_bem_s1 --tgt-code bem_Latn --seed 1 --no-save

NLLB has no Chitonga (toi_Latn). Two ways in, both from David Dale's public tutorial "How to fine-tune a NLLB-200
model for translating a new language" (2023):
  --tgt-code bem_Latn            reuse a related language's code as the Chitonga tag (simplest; the default);
  --new-code toi_Latn --init-from bem_Latn   add a new code, its embedding copied from the related language.
Training loop also after that tutorial: Adafactor at a constant rate after a short warmup, random batches, both
directions optional (--both-directions); our additions: early stopping on dev loss with the best weights restored,
seed, and a config file recording every setting and library version. Reviewed 10 Oct 2026 (friendly adversarial review).
Writes into <run>/<tag>/: test.hyp.txt (one translation per test line, same order), dev_loss.tsv, config.json,
and the best model (unless --no-save). Nothing here reads the test references: they never leave Chris's machine.
"""
import argparse
import json
import os
import random
import time

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_constant_schedule_with_warmup
from transformers.optimization import Adafactor

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--base", default="facebook/nllb-200-distilled-600M")
p.add_argument("--eng-code", default="eng_Latn", help="English code: eng_Latn (NLLB) or en (M2M100, 18 Oct backup line)")
p.add_argument("--tgt-code", default="bem_Latn", help="language code used as the Chitonga tag")
p.add_argument("--new-code", default=None, help="add this new language code (e.g. toi_Latn) instead of reusing --tgt-code")
p.add_argument("--init-from", default="bem_Latn", help="with --new-code: copy the embedding of this code")
p.add_argument("--steps", type=int, default=1000, help="~924 pairs / batch 16 = 58 steps an epoch")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max-len", type=int, default=128)
p.add_argument("--eval-every", type=int, default=50)
p.add_argument("--patience", type=int, default=6, help="evaluations without a better dev loss before stopping")
p.add_argument("--both-directions", action="store_true", help="also train Chitonga -> English (half the batches)")
p.add_argument("--beams", type=int, default=4)
p.add_argument("--no-repeat-ngram", type=int, default=3, help="decoding: block repeated 3-grams (11 Oct 2026: the kept "
               "model re-decoded with 3 loses its 2 loop lines, 35.5 -> 35.9; 0 = off, the setting of all runs before)")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--no-save", action="store_true")
p.add_argument("--mix", default=None, help="sample batches by group, e.g. own:0.5,story:0.1,bible:0.2,copy:0.2; groups not "
               "named are not used. Without it every train.jsonl pair is drawn uniformly (the stage 1 behaviour)")
p.add_argument("--tags", action="store_true", help="prefix each source with its group tag (<own>, <story>, <bible>, <copy>); "
               "dev and test use <own> (Kobus, Crego and Senellart 2017 domain control; Caswell, Chelba and Grangier 2019)")
p.add_argument("--concat", action="store_true", help="add a 'concat' pool: 2 and 3 consecutive SmolDoc sentences joined "
               "(training lines only), sampled under the weight given to concat in --mix")
p.add_argument("--polish", type=int, default=0, help="after the main run, up to this many steps on --polish-mix, kept only "
               "if dev loss improves (mixed fine-tuning, Chu, Dabre and Kurohashi 2017)")
p.add_argument("--polish-mix", default="own:0.7,story:0.3")
p.add_argument("--legacy-mix", action="store_true", help="reproduce a run of stage 14 or earlier: skip the 21 Oct mix rules "
               "(every group named, hopgood drawn, no Bible English pairs; mix_groups.py). Never for a new stage")
# 21 Oct 2026, stage 15 (the Jesuit; llm/council/codex/2026-10-20_1842_research_nllb.md section 1): CONTINUED full tuning
# from an adapted checkpoint (the released stage 13 soup) at a low rate, with label smoothing, warmup + linear decay,
# an effective batch of 32 and selection on the dev loss of BOTH directions (the round targets Tonga > English, which
# the English > Tonga-only dev loss cannot see). Every default below reproduces the runs before stage 15.
p.add_argument("--init-model", default=None, help="start from this fine-tuned checkpoint (local folder or HF repo) whose "
               "tokenizer already carries --tgt-code; no new code is added. Stage 15: zamai-zambia/chitonga-nllb-s13")
p.add_argument("--expect-code-id", type=int, default=None, help="with --init-model: assert the id of --tgt-code (s13: 256204)")
p.add_argument("--label-smoothing", type=float, default=0.0)
p.add_argument("--weight-decay", type=float, default=1e-3)
p.add_argument("--schedule", choices=("constant", "linear"), default="constant", help="linear = warmup then linear decay to 0")
p.add_argument("--warmup-frac", type=float, default=None, help="warmup as a share of --steps (default: min(100, steps/10))")
p.add_argument("--grad-accum", type=int, default=1, help="micro-batches per update; each is drawn from the mix on its own")
p.add_argument("--tf32", action="store_true", help="21 Oct (Ada/Ampere): TF32 tensor-core matmuls; weights stay fp32")
p.add_argument("--bf16", action="store_true", help="21 Oct: bf16 autocast for the TRAINING forward pass only (fp32 weights, "
               "fp32 loss); the dev loss and the test decode stay fp32, so they compare with earlier stages")
p.add_argument("--dev-both", action="store_true", help="dev loss = mean of English > Tonga and Tonga > English")
# 21 Oct 2026, the stage 15 diagnostic (Chris: "Ok go"; LOGBOOK 21 Oct 15:00). Both stage 15 arms stopped at step 100 on a
# dev loss that rose and stayed flat, so the kept models were warm-up churn. Three causes are open (label smoothing, a
# dev set that does not match the mix, the learning rate) and the early stop itself hid the evidence: the weights after
# step 100 were thrown away. --save-steps keeps the weights at FIXED steps whatever the dev loss says, so each checkpoint
# can be read blind; the dev loss is only logged. Both default off: every older stage runs exactly as before.
p.add_argument("--save-steps", default=None, help="comma list of update steps, e.g. 250,500,1000,2000: the CURRENT weights "
               "are saved to <run>/<tag>/ckpt_<step>/ (model + tokenizer, then an empty SAVED marker for an uploader). With "
               "it there is NO early stop and NO best-weights restore; the run goes to --steps")
p.add_argument("--log-detail", action="store_true", help="dev_loss.tsv also carries the dev loss per direction (en>toi, "
               "toi>en) and the mean training loss over the updates since the last evaluation (the logged train loss is "
               "otherwise the last micro-batch only), and is rewritten after every evaluation")
# 21 Oct 2026, stage 16 (Chris: "make the run as efficient as possible for the card actually obtained"). All default
# off: every older stage runs exactly as before.
p.add_argument("--token-budget", type=int, default=0, help="LENGTH-GROUPED batches: each micro-batch takes rows of similar "
               "length (a window around a random position of the pool sorted by length) until rows x the longest side "
               "(tokens, capped at --max-len) would pass this budget; at most --max-rows rows. 0 = the old random batch of "
               "--batch rows. Stage 16's mixes padded ~56%% of every random batch (lengths 3 to 256 tokens)")
p.add_argument("--max-rows", type=int, default=64, help="with --token-budget: rows per micro-batch at most (short pools such "
               "as single lexicon words would otherwise fill a batch with 100+ rows)")
p.add_argument("--attn", choices=("default", "sdpa", "eager"), default="default", help="attention implementation "
               "(transformers 4.46 has SDPA for M2M100 / NLLB); the one used is logged")
p.add_argument("--min-tps", type=float, default=0, help="THROUGHPUT GATE: at update --tps-check-step, real (non-pad) source + "
               "target tokens per second since update 20 below this -> exit 3 with THROUGHPUT LOW (a card far below its "
               "expectation is stopped and reported, not run for hours)")
p.add_argument("--tps-check-step", type=int, default=200)
# 24 Oct 2026, stage 19 (Chris: "for the next model training, let's introduce domain tags"). Both default off: every older
# stage runs exactly as before.
p.add_argument("--row-tags", action="store_true", help="prefix each TRAIN row's source with its own 'tag' field (e.g. "
               "'<cls>' for the classroom register), plain text + one space, so it tokenises into existing pieces "
               "('<cls>' -> '<', 'c', 'ls', '>', checked at start-up); rows without the field, task rows, the copy pool, "
               "dev and test stay untagged (Kobus, Crego and Senellart 2017 domain control; Caswell, Chelba and "
               "Grangier 2019 tagged back-translation). Not --tags, which writes one tag per GROUP")
p.add_argument("--tag-drop", type=float, default=0.0, help="with --row-tags: each drawn row loses ALL its tags (domain "
               "and provenance; a '[BT] ' already in the text stays) with this probability, so the model still behaves "
               "sensibly on untagged input (the old readers get untagged input). Stage 19: 0.1")
p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (activations recomputed in the backward "
               "pass): for the 3.3B model on a smaller card; ~30%% slower; not needed on an 80 GB card")
a = p.parse_args()
save_at = sorted({int(s) for s in a.save_steps.split(",") if s.strip()}) if a.save_steps else []
if save_at and (a.polish or a.no_save or max(save_at) > a.steps or min(save_at) < 1):
    raise SystemExit("--save-steps: steps must lie in 1..--steps, and not with --polish or --no-save")

random.seed(a.seed)
torch.manual_seed(a.seed)
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
if a.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
out = os.path.join(a.run, a.tag)
os.makedirs(out, exist_ok=True)
read = lambda n: [json.loads(l) for l in open(os.path.join(a.run, n), encoding="utf-8") if l.strip()]
train, dev = read("train.jsonl"), read("dev.jsonl")
test_en = [l.rstrip("\n") for l in open(os.path.join(a.run, "test.en.txt"), encoding="utf-8")]


def group(o):
    """Stage 2 (11 Oct 2026): which pool a line belongs to, by its origin label."""
    if o in ("liseli:storybook", "storybooks_zambia", "asb:storybook"):
        return "story"
    if o == "jctr:homily":
        return "prose"
    if o.startswith("hopgood:"):
        return "hopgood"
    if o == "liseli:bible":
        return "bible"
    if o.startswith("synthetic:"):
        return o.split(":", 1)[1]          # stage 7: syn_tab, syn_sub, syn_frm, syn_vrb (one group per method)
    return "own"


pools, weights, task_lang = {}, {}, {}   # task_lang outside the if: no --mix crashed on it (Codex 17)
if a.mix:
    for x in train:
        pools.setdefault(group(x["origin"]), []).append(x)
    if os.path.exists(os.path.join(a.run, "mono.jsonl")):
        pools["copy"] = read("mono.jsonl")  # copy task: Chitonga -> Chitonga (Currey et al. 2017)
    if os.path.exists(os.path.join(a.run, "tasks.jsonl")):  # Hopgood structure/lexicon/table tasks (11 Oct 2026)
        for x in read("tasks.jsonl"):
            pools.setdefault(x["pool"], []).append(x)
            task_lang[x["pool"]] = (x["src_lang"], x["tgt_lang"])
    if a.concat:  # consecutive sentences of the same SmolDoc document, all already in train (test documents never are)
        import re as _re
        docs = {}
        for x in pools.get("own", []):
            m = _re.match(r"(smoldoc:[^:]+):(\d+)$", x["id"])
            if m:
                docs.setdefault(m.group(1), {})[int(m.group(2))] = x
        pools["concat"] = []
        for d, sents in docs.items():
            for i in sorted(sents):
                for n in (2, 3):
                    run_ = [sents.get(i + k) for k in range(n)]
                    if all(run_):
                        pools["concat"].append({"id": f"{d}:{i:02d}+{n}", "origin": "concat", "en": " ".join(r["en"] for r in run_),
                                                "toi": " ".join(r["toi"] for r in run_)})
    import math as _math
    for kv in a.mix.split(","):
        g, w = kv.split(":")
        if not (g in pools and pools[g]):
            raise SystemExit("no lines for group " + g)
        weights[g] = float(w)
        if not _math.isfinite(weights[g]) or weights[g] < 0:          # Codex guard review 11: 'story:-1', NaN
            raise SystemExit(f"mix weight must be finite and >= 0: {g}:{w}")
    if not sum(weights.values()) > 0:
        raise SystemExit("mix weights sum to 0")
    print("mix", {g: (len(pools[g]), w) for g, w in weights.items()}, flush=True)
    if not a.legacy_mix:
        # 21 Oct 2026 (audit decisions; mix_groups.py has the why). An unnamed group used to be skipped in silence: the
        # 1,302 Hopgood pairs sat in every train.jsonl from stage 7 to 14 without being drawn. Now every group present
        # must be named (weight 0 = left out on purpose, and visible), 'hopgood' must be present AND drawn, and no Bible
        # English pair may be trained (the Bible is monolingual: copy pool only). Explicit exits, not asserts (python
        # -O would drop those: Codex guard review 16). Checked here, on the pod, before any step.
        unnamed = {g: len(v) for g, v in pools.items() if v and g not in weights and not (g == "concat" and not a.concat)}
        if unnamed:
            raise SystemExit(f"groups present but not named in --mix (weight 0 to leave one out): {unnamed}")
        if not pools.get("hopgood") or weights.get("hopgood", 0) <= 0:
            raise SystemExit("hopgood must be in the build and drawn (DECISIONS 21 Oct (3))")
        if any(pools.get(g) for g in ("bible", "bible_lit")):
            raise SystemExit("Bible English pairs in the pools: the Bible is monolingual")
        import re as _re2
        _wr = _re2.compile(r"^\s*(?:\[[^\]]{0,20}\]|<[^>]{0,20}>)\s*")
        _vn = _re2.compile(r"^\s*\d{1,3}(?:[:.]\d{1,3})*[a-z]?[.)]?\s+")

        def _w(t):
            """mix_groups.verse_key: task wrappers and verse numbers off, then the word key (Codex 8)"""
            t = t or ""
            for _ in range(3):
                u = _vn.sub("", _wr.sub("", t))
                if u == t:
                    break
                t = u
            return " ".join(_re2.findall(r"[\w']+", t.lower()))
        _verses = {_w(x["toi"]) for x in pools.get("copy", []) if str(x.get("origin", "")).startswith("liseli:bible")}
        _verses = {v for v in _verses if len(v.split()) >= 3}
        _bib = [x.get("id") for g, v in pools.items() if g != "copy" for x in v
                if "bible" in (str(x.get("origin", "")) + str(x.get("id", ""))).lower()
                or (x.get("toi") is not None and _w(x.get("toi")) in _verses)
                or (x.get("tgt_lang") == "toi" and _w(x.get("tgt")) in _verses)
                or (x.get("src_lang") == "toi" and _w(x.get("src")) in _verses)]
        if _bib:
            raise SystemExit(f"Bible English pairs in the build ({len(_bib)}): {_bib[:3]}")
else:
    if any(group(x["origin"]) != "own" for x in train):
        raise SystemExit("train.jsonl holds outside pairs: name the groups with --mix")

if not a.legacy_mix:
    # 21 Oct 2026 (Codex guard review 1): the pod checks held-out contamination before step 1. The full guard needs the
    # Beelink's frozen sets, so the builder runs it (mix_groups.check_build: train / mono / tasks / dev through
    # heldout_guard) and writes preflight.json with the sha256 of every file it checked and every mix it checked.
    # Here: the file must exist, every listed file present on the pod must have that sha (an edit after the check is
    # refused), --mix must be one of the checked mixes, and, independently, no test.en.txt / test_hopgood.en.txt line
    # may stand as a train or dev English side (exact, case and spaces folded).
    import hashlib as _hl
    _pf = os.path.join(a.run, "preflight.json")
    if not os.path.exists(_pf):
        raise SystemExit("preflight.json missing: run mix_groups.check_build(run_dir, mix) on the builder first")
    _pre = json.load(open(_pf, encoding="utf-8"))
    for _f, _want in _pre.get("files", {}).items():
        _p = os.path.join(a.run, _f)
        if os.path.exists(_p) and _hl.sha256(open(_p, "rb").read()).hexdigest() != _want:
            raise SystemExit(f"preflight: {_f} differs from the file the guard checked")
    for _f in ("train.jsonl", "dev.jsonl", "test.en.txt"):
        if _f not in _pre.get("files", {}):
            raise SystemExit(f"preflight: {_f} was not checked")
    if a.mix:
        _norm_mix = lambda m: {k: float(v) for k, v in (kv.split(":") for kv in m.split(","))}
        if _norm_mix(a.mix) not in [_norm_mix(m) for m in _pre.get("mixes", [])]:
            raise SystemExit(f"preflight: --mix {a.mix} was not checked (checked: {_pre.get('mixes')})")
    _k = lambda s: " ".join((s or "").lower().split())
    _held = {_k(l) for l in test_en if l.strip()}
    if os.path.exists(os.path.join(a.run, "test_hopgood.en.txt")):
        _held |= {_k(l) for l in open(os.path.join(a.run, "test_hopgood.en.txt"), encoding="utf-8") if l.strip()}
    _leak = [x.get("id") for x in train + dev if _k(x.get("en")) in _held]
    if _leak:
        raise SystemExit(f"held-out test English in train / dev ({len(_leak)}): {_leak[:3]}")

tok = AutoTokenizer.from_pretrained(a.init_model or a.base)
model = AutoModelForSeq2SeqLM.from_pretrained(a.init_model or a.base,
                                              **({"attn_implementation": a.attn} if a.attn != "default" else {}))
print("ATTN", getattr(model.config, "_attn_implementation", "?"), "bf16" if a.bf16 else "fp32", "tf32" if a.tf32 else "",
      f"token-budget {a.token_budget} max-rows {a.max_rows}" if a.token_budget else f"batch {a.batch}", flush=True)
TGT = a.tgt_code
if a.init_model:
    # stage 15: the checkpoint already learned TGT (toi_Latn); adding it again would make a second, untrained row
    assert not a.new_code, "--init-model continues a checkpoint: do not pass --new-code"
    _id = tok.convert_tokens_to_ids(TGT)
    assert _id != tok.unk_token_id and tok.tokenize(TGT) == [TGT], f"{TGT} is not a single known token in {a.init_model}"
    assert a.expect_code_id is None or _id == a.expect_code_id, f"{TGT} has id {_id}, expected {a.expect_code_id}"
    assert model.get_input_embeddings().weight.shape[0] == len(tok), "embedding rows != tokenizer size"
    print(f"init from {a.init_model}: {TGT} id {_id}, vocab {len(tok)}", flush=True)
if a.new_code:
    # Dale (2023), updated recipe for transformers >= 4.38: pass the new code as an extra special token, resize, copy
    old = list(tok.additional_special_tokens)
    assert a.new_code not in old, a.new_code + " already exists"
    tok = AutoTokenizer.from_pretrained(a.base, additional_special_tokens=old + [a.new_code])
    model.resize_token_embeddings(len(tok))
    new_id, src_id = tok.convert_tokens_to_ids(a.new_code), tok.convert_tokens_to_ids(a.init_from)
    assert new_id != tok.unk_token_id and src_id != tok.unk_token_id
    assert tok.tokenize(a.new_code) == [a.new_code], "new code is split into pieces"
    with torch.no_grad():
        model.model.shared.weight[new_id] = model.model.shared.weight[src_id]
    # NLLB ties the output layer and both embedding tables to `shared`, so the copy covers input and output alike
    for w in (model.lm_head.weight, model.model.encoder.embed_tokens.weight, model.model.decoder.embed_tokens.weight):
        assert w.data_ptr() == model.model.shared.weight.data_ptr(), "embeddings not tied: copy the row into each"
    TGT = a.new_code
# 18 Oct (commercial backup, COMMERCIAL_TRANSLATOR_DECISION.md): M2M100 names its codes "en"/"zu" and maps them with
# get_lang_id; NLLB's codes are the tokens themselves
lang_id = (lambda c: tok.get_lang_id(c)) if hasattr(tok, "get_lang_id") else (lambda c: tok.convert_tokens_to_ids(c))
ENG = a.eng_code
for c in (TGT, ENG):
    assert lang_id(c) != tok.unk_token_id, "unknown language code " + c
model.to(dev_)
if a.grad_ckpt:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("GRADIENT CHECKPOINTING on", flush=True)
ROW_TAG_RX = None
if a.row_tags:
    # stage 19: every tag found on a train row must be a short plain-text marker that tokenises into the same pieces
    # whatever follows it (a following SPACE keeps '<cls> Count' -> '<','c','ls','>','Count'; without the space the next
    # word would be cut differently), and must not contain an unknown piece. Checked here, on the pod, before step 1.
    import re as _re3
    ROW_TAG_RX = _re3.compile(r"^<[a-z]{2,6}>$")
    _tags = {}
    for x in train:
        for _k in ("tag", "prov"):
            if x.get(_k):
                _tags[x[_k]] = _tags.get(x[_k], 0) + 1
    for _f in ("tasks.jsonl", "mono.jsonl"):
        if os.path.exists(os.path.join(a.run, _f)):
            for x in read(_f):
                for _k in ("tag", "prov"):
                    if x.get(_k):
                        _tags[x[_k]] = _tags.get(x[_k], 0) + 1
    for _t, _n in _tags.items():
        if not ROW_TAG_RX.match(_t):
            raise SystemExit(f"row tag {_t!r} is not a short <word> marker")
        tok.src_lang = ENG
        _p = tok.tokenize(_t)
        for _w in ("Count the stones.", "Kobala mabwe.", "The project"):
            if tok.tokenize(_t + " " + _w)[:len(_p)] != _p:
                raise SystemExit(f"row tag {_t!r} does not tokenise stably before {_w!r}")
        if tok.unk_token in _p:
            raise SystemExit(f"row tag {_t!r} carries an unknown piece: {_p}")
        print(f"ROW TAG {_t} -> {ascii(_p)} on {_n} rows", flush=True)   # ascii: a cp1252 console cannot print the SentencePiece space mark (smoke, 24 Oct)
    if not _tags:
        print("ROW TAGS: none in train.jsonl (flag has no effect)", flush=True)


def row_prefix(x):
    """stage 19: '<domain> [<prov>] ' from the row's own fields, both directions, SOURCE side only (the target is never
    tagged); --tag-drop drops all of it for this draw. Dev / test rows carry no fields, so they stay untagged."""
    if a.tag_drop and model.training and random.random() < a.tag_drop:
        return ""
    return "".join(x[k] + " " for k in ("tag", "prov") if x.get(k))


TAG = {"own": "<own> ", "concat": "<own> ", "story": "<story> ", "bible": "<bible> ", "copy": "<copy> "}


def batch_loss(pairs, src_lang, tgt_lang, s_key, t_key, g="own"):
    tok.src_lang, tok.tgt_lang = src_lang, tgt_lang
    pre = TAG.get(g or "own", "") if a.tags else ""
    rt = row_prefix if a.row_tags else (lambda x: "")
    enc = tok([pre + rt(x) + x[s_key] for x in pairs], text_target=[x[t_key] for x in pairs], return_tensors="pt", padding=True,
              truncation=True, max_length=a.max_len).to(dev_)
    enc["labels"][enc["labels"] == tok.pad_token_id] = -100
    if model.training:
        TOKS[0] += int(enc["attention_mask"].sum()) + int((enc["labels"] != -100).sum())
        TOKS[1] += int(enc["input_ids"].numel()) + int(enc["labels"].numel())
    if a.label_smoothing and model.training:
        # stage 15: label smoothing on the training loss only; the dev loss stays the plain one (comparable across runs)
        with torch.autocast(device_type=dev_, dtype=torch.bfloat16, enabled=a.bf16):
            logits = model(**enc).logits
        return torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), enc["labels"].view(-1),
                                                 ignore_index=-100, label_smoothing=a.label_smoothing)
    with torch.autocast(device_type=dev_, dtype=torch.bfloat16, enabled=a.bf16 and model.training):
        out = model(**enc)
    return out.loss.float()


dev_parts = {}   # --log-detail: the last dev loss per direction
TOKS = [0, 0]    # real (non-pad) and padded tokens of the training batches, source + target


def lengths_of(pool, g):
    """--token-budget: per row the longer side in tokens (+2 specials), capped at --max-len; indices sorted by it"""
    if g in task_lang:
        a_, b_ = [x["src"] for x in pool], [x["tgt"] for x in pool]
    elif g == "copy":
        a_ = b_ = [x["toi"] for x in pool]
    else:
        a_, b_ = [x["toi"] for x in pool], [x["en"] for x in pool]
    la = [len(v) for v in tok(a_, add_special_tokens=False)["input_ids"]]
    lb = la if b_ is a_ else [len(v) for v in tok(b_, add_special_tokens=False)["input_ids"]]
    L = [min(a.max_len, max(x, y) + 2) for x, y in zip(la, lb)]
    return L, sorted(range(len(pool)), key=lambda i: L[i])


LEN = {}


def draw(pool, g):
    """one micro-batch of rows from a pool: random (the old way) or length-grouped under --token-budget"""
    if not a.token_budget:
        return random.sample(pool, min(a.batch, len(pool)))
    if g not in LEN:
        LEN[g] = lengths_of(pool, g)
    L, order = LEN[g]
    n = len(order)
    i = random.randrange(n)
    w = 4 * a.max_rows
    win = order[max(0, i - w): min(n, i + w)]
    cand = [order[i]] + random.sample(win, min(len(win), 2 * a.max_rows))
    out, mx, seen_ = [], 0, set()
    for j in cand:
        if j in seen_:
            continue
        m2 = max(mx, L[j])
        if out and (len(out) + 1) * m2 > a.token_budget or len(out) >= a.max_rows:
            break
        out.append(j)
        seen_.add(j)
        mx = m2
    return [pool[j] for j in out]


def dev_loss():
    model.eval()
    tot = 0.0
    dirs = [(ENG, TGT, "en", "toi"), (TGT, ENG, "toi", "en")] if a.dev_both else [(ENG, TGT, "en", "toi")]
    with torch.no_grad():
        for sl, tl, sk, tk in dirs:
            part = 0.0
            for i in range(0, len(dev), a.batch):
                b = dev[i:i + a.batch]
                part += batch_loss(b, sl, tl, sk, tk, "own").item() * len(b)
            dev_parts[sk + ">" + tk] = part / max(1, len(dev))
            tot += part
    model.train()
    return tot / max(1, len(dev) * len(dirs))


def translate(texts):
    model.eval()
    tok.src_lang = ENG
    hyps = []
    with torch.no_grad():
        for i in range(0, len(texts), a.batch):
            enc = tok([(TAG["own"] if a.tags else "") + t for t in texts[i:i + a.batch]], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to(dev_)
            gen = model.generate(**enc, forced_bos_token_id=lang_id(TGT), num_beams=a.beams,
                                 max_new_tokens=int(32 + 3 * enc.input_ids.shape[1]),
                                 **({"no_repeat_ngram_size": a.no_repeat_ngram} if a.no_repeat_ngram else {}))
            hyps += tok.batch_decode(gen, skip_special_tokens=True)
    return hyps


best, bad, log = float("inf"), 0, []
T20 = [0.0, 0, 0]   # time and token counters at update 20 (after warm-up of kernels and lengths)
start = time.time()


def write_log():
    if a.log_detail:
        head = "step\ttrain_loss\tdev_loss\ttrain_mean\tdev_en>toi\tdev_toi>en\tminutes\n"
    else:
        head = "step\ttrain_loss\tdev_loss\n"
    open(os.path.join(out, "dev_loss.tsv"), "w").write(head + "".join("\t".join(str(v) for v in r) + "\n" for r in log))


def log_row(step, train_last, dl, train_mean=None):
    if a.log_detail:
        log.append((step, train_last, dl, train_mean, round(dev_parts.get("en>toi", float("nan")), 4),
                    round(dev_parts.get("toi>en", float("nan")), 4), round((time.time() - start) / 60, 1)))
        write_log()
    else:
        log.append((step, train_last, dl))


def save_ckpt(step):
    """--save-steps: the current weights, whatever the dev loss; SAVED is written last, so an uploader never takes a
    half-written folder"""
    d = os.path.join(out, f"ckpt_{step}")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    open(os.path.join(d, "SAVED"), "w").write(f"step {step}\n")
    print(f"CKPT_SAVED step {step} {d}", flush=True)
if a.steps > 0:
    opt = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=a.lr, clip_threshold=1.0,
                    weight_decay=a.weight_decay)
    _warm = max(1, int(a.steps * a.warmup_frac)) if a.warmup_frac is not None else max(1, min(100, a.steps // 10))
    if a.schedule == "linear":
        from transformers import get_linear_schedule_with_warmup
        sched = get_linear_schedule_with_warmup(opt, _warm, a.steps + a.polish)
    else:
        sched = get_constant_schedule_with_warmup(opt, _warm)
    # named_parameters() lists tied weights once: one copy of the embeddings (~1 GB), not four
    snap = lambda: {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    model.train()
    best = dev_loss()
    log_row(0, None, best)
    best_state = None if save_at else snap()   # --save-steps: no restore, so no 5 GB copy on every improvement
    def phase(w, steps, step0):
        global best, best_state
        bad, tsum, tn = 0, 0.0, 0
        for step in range(step0 + 1, step0 + steps + 1):
            for _micro in range(a.grad_accum):          # stage 15: each micro-batch drawn from the mix on its own
                g = random.choices(list(w), list(w.values()))[0] if w else None
                pool = pools[g] if g else train
                b = draw(pool, g or "__train__")
                if g in task_lang:  # the tag is already in the source text
                    code = {"toi": TGT, "eng": ENG}
                    sl, tl = task_lang[g]
                    loss = batch_loss(b, code[sl], code[tl], "src", "tgt", None)
                elif g == "copy":
                    loss = batch_loss(b, TGT, TGT, "toi", "toi", g)
                elif a.both_directions and random.random() < 0.5:
                    loss = batch_loss(b, TGT, ENG, "toi", "en", g)
                else:
                    loss = batch_loss(b, ENG, TGT, "en", "toi", g)
                (loss / a.grad_accum).backward()
                if a.log_detail:
                    tsum += loss.item() / a.grad_accum   # a GPU sync per micro-batch: only when asked
            tn += 1
            if step == 20:
                T20[:] = [time.time(), TOKS[0], TOKS[1]]
            if a.min_tps and step == a.tps_check_step and T20[0]:
                tps = (TOKS[0] - T20[1]) / max(1e-9, time.time() - T20[0])
                pad = 1 - (TOKS[0] - T20[1]) / max(1, TOKS[1] - T20[2])
                print(f"THROUGHPUT CHECK step {step}: {tps:.0f} real tokens/s (gate {a.min_tps:.0f}), padding {pad:.0%}", flush=True)
                if tps < a.min_tps:
                    print("THROUGHPUT LOW: stopping (exit 3) rather than run for hours", flush=True)
                    raise SystemExit(3)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % a.eval_every == 0:
                dl = dev_loss()
                log_row(step, round(loss.item(), 4), round(dl, 4), round(tsum / max(1, tn), 4))
                extra = (f" (en>toi {dev_parts.get('en>toi', 0):.3f} toi>en {dev_parts.get('toi>en', 0):.3f} train_mean "
                         f"{tsum / max(1, tn):.3f})") if a.log_detail else ""
                tsum, tn = 0.0, 0
                if T20[0]:
                    el = time.time() - T20[0]
                    extra += (f" | {(TOKS[0] - T20[1]) / max(1e-9, el):.0f} real tok/s, padding "
                              f"{1 - (TOKS[0] - T20[1]) / max(1, TOKS[1] - T20[2]):.0%}, {el / max(1, step - 20):.3f} s/update")
                if dev_ == "cuda":     # 24 Oct (stage 19, the 3.3B arm): the peak memory is measured, not guessed
                    extra += f" | peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
                print(f"step {step} train {loss.item():.3f} dev {dl:.3f}{extra}", flush=True)
                if dl < best:
                    best, bad = dl, 0
                    if not save_at:
                        best_state = snap()
                elif not save_at:
                    bad += 1
                    if bad >= a.patience:
                        print("early stop", flush=True)
                        return step
            if step in save_at:
                save_ckpt(step)
        return step0 + steps

    last = phase(weights, a.steps, 0)
    if not save_at:
        missing = model.load_state_dict(best_state, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    if a.polish:
        pw = {g: float(v) for g, v in (kv.split(":") for kv in a.polish_mix.split(","))}
        assert all(g in pools and pools[g] for g in pw), "polish group without lines"
        before = best
        print(f"polish {pw} from dev {best:.3f}", flush=True)
        phase(pw, a.polish, last)
        missing = model.load_state_dict(best_state, strict=False)
        print(f"polish: dev {before:.3f} -> {best:.3f}", flush=True)

if os.path.exists(os.path.join(a.run, "test_hopgood.en.txt")):
    th = [l.rstrip("\n") for l in open(os.path.join(a.run, "test_hopgood.en.txt"), encoding="utf-8")]
    hh = translate(th)
    open(os.path.join(out, "test_hopgood.hyp.txt"), "w", encoding="utf-8").write("".join(h.replace("\n", " ") + "\n" for h in hh))
hyps = translate(test_en)
assert len(hyps) == len(test_en)
open(os.path.join(out, "test.hyp.txt"), "w", encoding="utf-8").write("".join(h.replace("\n", " ") + "\n" for h in hyps))
write_log()
json.dump({**vars(a), "target_code_used": TGT, "train_pairs": len(train), "mix_pools": {g: len(pools[g]) for g in weights}, "dev_pairs": len(dev), "test_lines": len(test_en),
           "best_dev_loss": best if a.steps > 0 else None, "minutes": round((time.time() - start) / 60, 1),
           "device": dev_, "gpu": torch.cuda.get_device_name(0) if dev_ == "cuda" else None,
           "torch": torch.__version__, "transformers": transformers.__version__},
          open(os.path.join(out, "config.json"), "w"), indent=1)
if a.steps > 0 and not a.no_save and a.steps not in save_at:   # --save-steps: the last step is already ckpt_<steps>
    model.save_pretrained(os.path.join(out, "model"))
    tok.save_pretrained(os.path.join(out, "model"))
print("wrote", os.path.join(out, "test.hyp.txt"))
