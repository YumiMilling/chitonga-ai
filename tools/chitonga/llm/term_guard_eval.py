"""The term guard's acceptance test (22 Oct 2026, the Jesuit; council NEXT_ROUND step 3 rule: "it must fire on the named
items and on NO other line of the 174 + 48 + 687").

Sets: stage 16's P1 174 + El Nino 48 (runs/2026-10-21_stage16/prescreen/gen.jsonl) and the 687 held-out document items
(doctest/gen_docs.jsonl). Only en>toi lines can arm the guard (toi>en is never guarded); every armed line is logged.

  plain     every en>toi line of every set, decoded as the judge decoded it (h2_reader._run: beam 4, max 256, batches
            of 16) once plainly and once through term_guard.translate_batch; every output that differs is listed
            (an unarmed line must be identical: the guard takes it out of its batch, nothing else).
  p1        the armed lines through the full P1 service (translator_service) with the guard on and off: the teacher
            assistant's own path.

    python -X utf8 tools/chitonga/llm/term_guard_eval.py plain --model s13-soup [--model s16-fullA-soup-step8000]
    (R = s13-soup on every en>toi line; any other model on the armed lines only)
    python -X utf8 tools/chitonga/llm/term_guard_eval.py p1
Out: internal/chitonga_train/llm/term_guard/ (eval_<model>.json, p1.json). Named items (the reason for each rule):
e2t:bk:poster_moh_who_cholera_pairs:0 / :1 / :7 (K's *kkolona*) and ce2t:06 (R's and K's *kusoomona*).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import term_guard as TG  # noqa: E402

S16 = os.path.join(REPO, "internal", "chitonga_train", "runs", "2026-10-21_stage16")
OUT = os.path.join(REPO, "internal", "chitonga_train", "llm", "term_guard")
MODELS = r"models"
NAMED = {"e2t:bk:poster_moh_who_cholera_pairs:0", "e2t:bk:poster_moh_who_cholera_pairs:1",
         "e2t:bk:poster_moh_who_cholera_pairs:7", "ce2t:06"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def items():
    its = [dict(x, part=x.get("set", "p1")) for x in jl(os.path.join(S16, "prescreen", "gen.jsonl"))]
    its += [dict(x, part="docs") for x in jl(os.path.join(S16, "doctest", "gen_docs.jsonl"))]
    return its


def armed_log(its):
    return [{"iid": x["iid"], "part": x["part"], "src": x["src"], "rules": [r["name"] for r in TG.rules_for(x["src"], x["dir"])]}
            for x in its if TG.rules_for(x["src"], x["dir"])]


def plain(models, full=("s13-soup",)):
    """full: the models decoded on EVERY en>toi line; the others on the armed lines only (an unarmed line is the
    caller's own plain call by construction, which the full model's sweep checks)"""
    import h2_reader as R
    its = items()
    counts = {}
    for x in its:
        counts[(x["part"], x["dir"])] = counts.get((x["part"], x["dir"]), 0) + 1
    arm = armed_log(its)
    os.makedirs(OUT, exist_ok=True)
    for m in models:
        r = R.Reader(os.path.join(MODELS, "chitonga-nllb-" + m))
        tok = r.tok
        tok.src_lang = "eng_Latn"
        sub = [x for x in its if x["dir"] == "en>toi" and (m in full or TG.rules_for(x["src"], x["dir"]))]
        diffs, reps, n = [], [], 0
        for part in ("p1", "climate", "docs"):
            ps = [x for x in sub if x["part"] == part] if part != "climate" else \
                [x for x in sub if x["part"] not in ("p1", "docs")]
            for i in range(0, len(ps), 16):
                b = ps[i:i + 16]
                toks = [tok.convert_ids_to_tokens(tok.encode(x["src"])) for x in b]
                kw = dict(target_prefix=[["toi_Latn"]] * len(b), beam_size=4, max_decoding_length=256)
                base = [tok.decode(tok.convert_tokens_to_ids(z.hypotheses[0][1:]), skip_special_tokens=True)
                        for z in r.tr.translate_batch(toks, **kw)]
                hy, rp = TG.translate_batch(r.tr, tok, toks, [x["src"] for x in b], "en>toi", **kw)
                for x, o, h, rep in zip(b, base, hy, rp):
                    n += 1
                    g = tok.decode(tok.convert_tokens_to_ids(h[0][0][1:]), skip_special_tokens=True)
                    if rep:
                        reps.append({"iid": x["iid"], "src": x["src"], "named": x["iid"] in NAMED, **rep,
                                     "plain_batch": o, "guarded": g})
                    if g != o:
                        diffs.append({"iid": x["iid"], "armed": bool(rep), "named": x["iid"] in NAMED, "src": x["src"],
                                      "before": o, "after": g, "stored": x["out"].get(m)})
            print(m, part, "done", flush=True)
        res = {"model": m, "decoded_en_toi": n, "counts": {f"{k[0]} {k[1]}": v for k, v in counts.items()},
               "armed": arm, "reports": reps, "changed": diffs,
               "changed_unarmed": [d for d in diffs if not d["armed"]],
               "changed_not_named": [d for d in diffs if not d["named"]]}
        json.dump(res, open(os.path.join(OUT, f"eval_{m}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{m}: decoded {n} en>toi; armed {len(arm)}; changed {len(diffs)} "
              f"(unarmed {len(res['changed_unarmed'])}, not named {len(res['changed_not_named'])})")
        for d in diffs:
            print(" ", d["iid"], "|", d["before"], ">", d["after"])


def p1():
    import translator_service as TS
    arm = armed_log(items())
    on, off = TS.Service(), TS.Service({"term_guard": False})
    off._r = on._r                                     # one set of loaded readers
    rows = []
    for a in arm:
        x = on.translate(a["src"], "en>toi", use_memory=False)
        y = off.translate(a["src"], "en>toi", use_memory=False)
        rows.append({"iid": a["iid"], "named": a["iid"] in NAMED, "src": a["src"], "rules": a["rules"],
                     "off": y["text"], "on": x["text"], "changed": x["text"] != y["text"],
                     "guard": x.get("term_guard"), "alarms_on": x["alarms"]})
        print(a["iid"], "|", y["text"], ">" if x["text"] != y["text"] else "=", x["text"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "p1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "plain":
        plain([a[i + 1] for i, v in enumerate(a) if v == "--model"] or ["s13-soup"])
    elif a and a[0] == "p1":
        p1()
    else:
        print(__doc__)
