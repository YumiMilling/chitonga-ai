"""Selection score for ONE stage 19 checkpoint, on the GPU machine (the Jesuit, 24 Oct 2026). Written BEFORE any stage 19
output exists. As f18_sel.py, plus the TAGS: every slice is read untagged, with <cls> and with <gen> where it matters,
because the round's claim is that the tag routes the register (coordinator, 24 Oct: tags count as working if <cls> beats
no tag on the classroom lines and <gen> does not lose on the general lines).

  dev     the old dev set (both directions, token-mean NLL), untagged and <gen>: the FORGETTING guard
  sel     the stage 16 SEL documents, untagged and <gen>: an alarm, for continuity
  teach   the stage 19 teacher DEV slice (teacher_dev_stage19.jsonl: stage 18's 14 + 22 new written / better lines, never
          trained): NLL both directions untagged / <cls> / <gen>; a beam decode with <cls> for the hand read
  verb    memorisation: (1) the trained teacher rows (t_teach, written / better only: the 'ok' rows are a050's own output)
          decoded with <cls>; (2) the GENERALISATION GAP of the synthetic group: verbatim reproduction on a seeded sample of
          the trained cls_syn rows MINUS verbatim on the held-out synthetic slice (synthetic_s19_heldout.jsonl, never
          trained), both with <cls>. A model that learned the patterns reproduces both; one that memorised the rows
          reproduces only the trained ones (stage 18's absolute gate could not tell these apart)
  probe   the 13 confusable probes decoded RAW untagged, with <cls> and with <gen>
Untagged reads of a050 are the baseline f19_select.py compares against (a050 never saw a tag).

    python f19_sel.py --model DIR --run RUNDIR --out sel/f19_<tag>.json [--batch 16] [--sample 40]
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
TAGS = {"none": "", "cls": "<cls> ", "gen": "<gen> "}
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
if dev_ == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(dev_).eval()
t0 = time.time()
TAGSTR = re.compile(r"<\s*(?:cls|gen|syn)\s*>|\[BT\]")


def jl(p_):
    return [json.loads(l) for l in open(p_, encoding="utf-8") if l.strip()] if os.path.exists(p_) else []


def nll(pairs, tag=""):
    tot, n, by = 0.0, 0, {}
    for s, t, sl, tl in pairs:
        by.setdefault((sl, tl), []).append((tag + s, t))
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
                l = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100,
                                                      reduction="sum")
                tot += l.item()
                n += int((lab != -100).sum().item())
    return tot / max(1, n)


def translate(srcs, sl, tl, tag=""):
    out = []
    tok.src_lang = CODE[sl]
    with torch.no_grad():
        for i in range(0, len(srcs), a.batch):
            enc = tok([tag + s for s in srcs[i:i + a.batch]], return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_len).to(dev_)
            g = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODE[tl]), num_beams=a.beams,
                               max_new_tokens=int(32 + 3 * enc.input_ids.shape[1]), no_repeat_ngram_size=3)
            out += tok.batch_decode(g, skip_special_tokens=True)
    return out


def norm(s):
    return " ".join(re.findall(r"[\w']+", (s or "").lower()))


rep = {"model": a.model}
leaks = []
dev = jl(os.path.join(a.run, "dev.jsonl"))
for tn in ("none", "gen"):
    sfx = "" if tn == "none" else "_" + tn
    e2t = nll([(x["en"], x["toi"], "en", "toi") for x in dev], TAGS[tn])
    t2e = nll([(x["toi"], x["en"], "toi", "en") for x in dev], TAGS[tn])
    rep["dev_both" + sfx] = round((e2t + t2e) / 2, 5)
    rep["dev_en>toi" + sfx], rep["dev_toi>en" + sfx] = round(e2t, 5), round(t2e, 5)
    sel = jl(os.path.join(a.run, "sel.jsonl"))
    ss = []
    for d in ("en>toi", "toi>en"):
        sl, tl = d.split(">")
        ss.append(nll([(x["src"], x["ref"], sl, tl) for x in sel if x["dir"] == d], TAGS[tn]))
    rep["sel_score" + sfx] = round(sum(ss) / 2, 5)
td = jl(os.path.join(a.run, "teacher_dev_stage19.jsonl"))
for tn in ("none", "cls", "gen"):
    sfx = "" if tn == "none" else "_" + tn
    e2t = nll([(x["en"], x["toi"], "en", "toi") for x in td], TAGS[tn])
    t2e = nll([(x["toi"], x["en"], "toi", "en") for x in td], TAGS[tn])
    rep["teach_both" + sfx] = round((e2t + t2e) / 2, 5)
rep["teach_items"] = len(td)
hyp = translate([x["en"] for x in td], "en", "toi", TAGS["cls"])
leaks += [h for h in hyp if TAGSTR.search(h)]
rep["teach_hyps_cls"] = [{"id": x["id"], "en": x["en"], "ref": x["toi"], "hyp": h} for x, h in zip(td, hyp)]
train = jl(os.path.join(a.run, "train.jsonl"))
rng = random.Random(1924)
rep["verbatim"] = {}
tt = [x for x in train if x.get("origin") == "synthetic:t_teach" and x.get("kind") in ("written", "better")]
syn_tr = [x for x in train if x.get("origin") == "synthetic:cls_syn"]
syn_ho = jl(os.path.join(a.run, "synthetic_s19_heldout.jsonl"))
for name, rows, k in (("t_teach", tt, a.sample), ("cls_syn_train", syn_tr, 60), ("cls_syn_heldout", syn_ho, 60)):
    if not rows:
        continue
    s = rng.sample(rows, min(k, len(rows)))
    for tn in (("cls", "none") if name == "t_teach" else ("cls",)):
        h = translate([x["en"] for x in s], "en", "toi", TAGS[tn])
        leaks += [y for y in h if TAGSTR.search(y)]
        rep["verbatim"][f"{name}_{tn}"] = {"n": len(s), "exact": sum(norm(y) == norm(x["toi"]) for x, y in zip(s, h)),
                                           "examples": [{"en": x["en"], "ref": x["toi"], "hyp": y} for x, y in list(zip(s, h))[:4]]}
v = rep["verbatim"]
if "cls_syn_train_cls" in v and "cls_syn_heldout_cls" in v:
    rep["syn_gap"] = round(v["cls_syn_train_cls"]["exact"] / v["cls_syn_train_cls"]["n"]
                           - v["cls_syn_heldout_cls"]["exact"] / v["cls_syn_heldout_cls"]["n"], 4)
pr = jl(os.path.join(a.run, "confusable_probes_final18.jsonl"))
rep["probes"] = {}
for tn in ("none", "cls", "gen"):
    h = translate([x["en"] for x in pr], "en", "toi", TAGS[tn])
    leaks += [y for y in h if TAGSTR.search(y)]
    out = []
    for x, y in zip(pr, h):
        low = y.lower()
        out.append({"id": x["id"], "pair": x["pair"], "en": x["en"], "hyp": y,
                    "forbidden": [w for w in x["forbid"] if re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", low)],
                    "expected": [w for w in x["expect"] if w.strip().lower() in low]})
    rep["probes"][tn] = out
    rep["probe_fail" + ("" if tn == "none" else "_" + tn)] = sum(1 for o in out if o["forbidden"])
rep["tag_in_output"] = len(leaks)            # the target is never tagged: any tag string in an output is a leak
rep["tag_in_output_examples"] = leaks[:5]
rep["minutes"] = round((time.time() - t0) / 60, 2)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("F19SEL", json.dumps({k: v for k, v in rep.items() if k not in ("teach_hyps_cls", "probes", "verbatim",
                                                                      "tag_in_output_examples")}), flush=True)
