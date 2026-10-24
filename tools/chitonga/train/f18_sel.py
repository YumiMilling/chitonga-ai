"""Selection score for ONE stage 18 sweep checkpoint, on the GPU machine (the Jesuit, 24 Oct 2026; Chris: "also the
weights need to be optimized"; DECISIONS 24 Oct MIX SWEEP). Written BEFORE any sweep output exists.

What it measures, and why each one is here:
  dev     the old dev set (both directions, token-mean NLL): the FORGETTING guard. The new material must not be bought
          with the old (stage 16's rule, PREREG section 2).
  teach   the teacher DEV slice (teacher_dev_final18.jsonl: 14 teacher-written lines drawn after the frozen 44, never
          trained, guarded in the build): token-mean NLL both directions, and a beam decode for the hand read.
          This is the only slice that measures the thing the round is for.
  verb    VERBATIM REPRODUCTION per group: a seeded sample of the trained teacher rows (t_teach, the WRITTEN / better
          rows only: the 'ok' rows are a050's own output confirmed, so reproducing them is not memorisation) and of the
          synthetic group (synth24) decoded en>toi; the share whose output equals the training target, normalised. Read
          by f18_select.py RELATIVE to a050's own share on the same seeded sample (delta review 24 Oct, flaw A).
  probe   the 13 confusable probes (confusable_probes_final18.jsonl) decoded en>toi on the RAW model (the served
          term guard is not here, which is the point: the guard would hide a regression).
  sel     the stage 16 SEL documents (same as stage16_sel.py) for continuity with the earlier rounds: an alarm.

    python f18_sel.py --model DIR --run RUNDIR --out sel/f18_<tag>.json [--batch 16] [--sample 40]

RUNDIR is the unpacked data folder (s18): it must hold dev.jsonl, sel.jsonl, train.jsonl, teacher_dev_final18.jsonl
and confusable_probes_final18.jsonl. Nothing is judged here; f18_select.py applies the pre-registered rule.
"""
import argparse
import json
import os
import random
import re
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--max-len", type=int, default=256)
p.add_argument("--beams", type=int, default=4)
p.add_argument("--sample", type=int, default=40)
p.add_argument("--tgt-code", default="toi_Latn")
a = p.parse_args()
CODE = {"en": "eng_Latn", "toi": a.tgt_code}
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
if dev_ == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(dev_).eval()
t0 = time.time()


def jl(p_):
    return [json.loads(l) for l in open(p_, encoding="utf-8") if l.strip()] if os.path.exists(p_) else []


def nll(pairs):
    tot, n = 0.0, 0
    by = {}
    for s, t, sl, tl in pairs:
        by.setdefault((sl, tl), []).append((s, t))
    with torch.no_grad():
        for (sl, tl), xs in by.items():
            tok.src_lang, tok.tgt_lang = CODE[sl], CODE[tl]
            for i in range(0, len(xs), a.batch):
                b = xs[i:i + a.batch]
                enc = tok([s for s, _ in b], text_target=[t for _, t in b], return_tensors="pt", padding=True,
                          truncation=True, max_length=a.max_len).to(dev_)
                lab = enc["labels"]
                lab[lab == tok.pad_token_id] = -100
                logits = model(**enc).logits.float()
                l = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1),
                                                      ignore_index=-100, reduction="sum")
                tot += l.item()
                n += int((lab != -100).sum().item())
    return (tot / max(1, n), n)


def translate(srcs, sl, tl):
    out = []
    tok.src_lang = CODE[sl]
    with torch.no_grad():
        for i in range(0, len(srcs), a.batch):
            enc = tok(srcs[i:i + a.batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_len).to(dev_)
            g = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODE[tl]), num_beams=a.beams,
                               max_new_tokens=int(32 + 3 * enc.input_ids.shape[1]), no_repeat_ngram_size=3)
            out += tok.batch_decode(g, skip_special_tokens=True)
    return out


def norm(s):
    return " ".join(re.findall(r"[\w']+", (s or "").lower()))


rep = {"model": a.model}
# --- old dev (forgetting) and SEL (continuity alarm)
dev = jl(os.path.join(a.run, "dev.jsonl"))
rep["dev_en>toi"] = round(nll([(x["en"], x["toi"], "en", "toi") for x in dev])[0], 5)
rep["dev_toi>en"] = round(nll([(x["toi"], x["en"], "toi", "en") for x in dev])[0], 5)
rep["dev_both"] = round((rep["dev_en>toi"] + rep["dev_toi>en"]) / 2, 5)
sel = jl(os.path.join(a.run, "sel.jsonl"))
for d in ("en>toi", "toi>en"):
    sl, tl = d.split(">")
    xs = [x for x in sel if x["dir"] == d]
    rep[f"sel_{d}"] = round(nll([(x["src"], x["ref"], sl, tl) for x in xs])[0], 5)
rep["sel_score"] = round((rep["sel_en>toi"] + rep["sel_toi>en"]) / 2, 5)
# --- the teacher dev slice: the loss both ways, and the decode for the hand read
td = jl(os.path.join(a.run, "teacher_dev_final18.jsonl"))
if td:
    rep["teach_en>toi"] = round(nll([(x["en"], x["toi"], "en", "toi") for x in td])[0], 5)
    rep["teach_toi>en"] = round(nll([(x["toi"], x["en"], "toi", "en") for x in td])[0], 5)
    rep["teach_both"] = round((rep["teach_en>toi"] + rep["teach_toi>en"]) / 2, 5)
    rep["teach_items"] = len(td)
    hyp = translate([x["en"] for x in td], "en", "toi")
    rep["teach_hyps"] = [{"id": x["id"], "en": x["en"], "ref": x["toi"], "hyp": h} for x, h in zip(td, hyp)]
    rep["teach_exact"] = sum(norm(h) == norm(x["toi"]) for x, h in zip(td, hyp))
# --- verbatim reproduction per trained group (the memorisation alarm)
train = jl(os.path.join(a.run, "train.jsonl"))
rng = random.Random(1824)
rep["verbatim"] = {}
for g in ("t_teach", "synth24", "t_word"):
    rows = [x for x in train if x.get("origin") == "synthetic:" + g and x.get("en") and x.get("toi")]
    if g == "t_teach":     # delta review: 81 of 136 are a050's own confirmed output; verbatim there is not memorisation
        rows = [x for x in rows if x.get("kind") in ("written", "better")] or rows
    if not rows:
        continue
    s = rng.sample(rows, min(a.sample, len(rows)))
    hyp = translate([x["en"] for x in s], "en", "toi")
    rep["verbatim"][g] = {"n": len(s), "exact": sum(norm(h) == norm(x["toi"]) for x, h in zip(s, hyp)),
                          "examples": [{"en": x["en"], "ref": x["toi"], "hyp": h} for x, h in list(zip(s, hyp))[:5]]}
# --- the confusable probes on the raw model
pr = jl(os.path.join(a.run, "confusable_probes_final18.jsonl"))
if pr:
    hyp = translate([x["en"] for x in pr], "en", "toi")
    out = []
    for x, h in zip(pr, hyp):
        low = h.lower()
        out.append({"id": x["id"], "pair": x["pair"], "en": x["en"], "hyp": h,
                    "forbidden": [w for w in x["forbid"] if re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", low)],
                    "expected": [w for w in x["expect"] if w.strip().lower() in low]})
    rep["probes"] = out
    rep["probe_fail"] = sum(1 for x in out if x["forbidden"])
rep["minutes"] = round((time.time() - t0) / 60, 2)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("F18SEL", json.dumps({k: v for k, v in rep.items() if k not in ("teach_hyps", "probes", "verbatim")}), flush=True)
