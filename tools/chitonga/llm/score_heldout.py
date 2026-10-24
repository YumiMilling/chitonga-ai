"""English > Tonga on the held-out sets for several NLLB models, side by side (15 Oct 2026, stage 10).

Sets: the children's-book and radio test (llm/eval_children/items.tsv) and the Thompson phrase-book lines held out of
every build (runs/<run>/thompson_heldout.tsv; alternatives in brackets count as further references, spelling variants
are data). chrF per set with its 95% range and, against the first model listed, the paired P>base, as score.py does.

    python -X utf8 tools/chitonga/llm/score_heldout.py <run folder> <model dir> [<model dir> ...]

Writes <run>/heldout_<model name>.jsonl (English | references | translation) for a person to read.
"""
import csv
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
from score_children import nllb  # noqa: E402


def sets(run):
    out = {}
    rows = list(csv.DictReader(open(os.path.join(REPO, "internal", "chitonga_train", "llm", "eval_children", "items.tsv"),
                                    encoding="utf-8"), delimiter="\t"))
    out["children+radio"] = [(r["en"], [r["toi"]]) for r in rows]
    rows = list(csv.DictReader(open(os.path.join(run, "thompson_heldout.tsv"), encoding="utf-8"), delimiter="\t"))
    th = []
    for r in rows:
        main = re.sub(r"\s*\([^)]*\)", "", r["toi"]).strip()
        alts = [a.strip() for a in re.findall(r"\(([^)]*)\)", r["toi"]) if a.strip()]
        th.append((re.sub(r"\s*\([^)]*\)", "", r["en"]).strip(), [main] + alts))
    out["Thompson"] = th
    return out


def boot(a, b, k=1000):
    rnd, n, wins, vals = random.Random(1), len(a), 0, []
    for _ in range(k):
        ix = [rnd.randrange(n) for _ in range(n)]
        x = S.corpus([a[i] for i in ix])
        vals.append(x)
        if b is not None and x > S.corpus([b[i] for i in ix]):
            wins += 1
    vals.sort()
    return vals[int(.025 * k)], vals[int(.975 * k)], (wins / k if b is not None else None)


def main():
    run, models = sys.argv[1], sys.argv[2:]
    data = sets(run)
    base = {}
    for m in models:
        name = os.path.basename(m.rstrip("/\\")).replace("chitonga-nllb-", "")
        rec = []
        print("==", name)
        for sname, items in data.items():
            hyps = nllb(m, [e for e, _ in items])
            sts = [S.best_stats(h, refs) for h, (_, refs) in zip(hyps, items)]
            lo, hi, p = boot(sts, base.get(sname))
            base.setdefault(sname, sts)
            print(f"  {sname:15s} n={len(items):3d} chrF {S.corpus(sts):5.1f}  ({lo:.1f}-{hi:.1f})"
                  + (f"  P>base {p:.2f}" if p is not None else "  [BASE]"))
            rec += [{"set": sname, "en": e, "refs": refs, "hyp": h} for h, (e, refs) in zip(hyps, items)]
        with open(os.path.join(run, f"heldout_{name}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rec:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
