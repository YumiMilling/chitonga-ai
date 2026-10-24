"""Score the safeguarding detector on a labelled set (24 Oct 2026, the Jesuit).

    python -X utf8 tools/chitonga/harness/tests/safeguard/score.py [dev_v1.jsonl] [--show] [--old]

Recall per category (disclosures caught), precision (of all messages caught, the share that are disclosures), the
false positives (lesson requests sent to the referral) and the misses, by id. --old scores the gate as it was before
24 Oct (h3_front's three regexes, kept in OLD_* below for comparison only).
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LLM = os.path.normpath(os.path.join(HERE, "..", "..", "..", "llm"))
sys.path.insert(0, LLM)
import safeguard as S  # noqa: E402

# the gate before 24 Oct 2026 (h3_front.SAFE_STRONG / SAFE_CHILD / SAFE_WEAK as they were), for the comparison only
OLD_STRONG = re.compile(r"(?i)\b(abus\w*|defile\w*|rap(e|ed|ing)|molest\w*|incest|sexual(ly)?|indecent\w*|"
                        r"self[\s-]*harm\w*|suicid\w*|kill\s+(him|her|them)sel(f|ves)|end\s+(his|her|their)\s+life|"
                        r"cut(s|ting)?\s+(him|her|them)sel(f|ves)|hurt(s|ing)?\s+(him|her|them)sel(f|ves)|"
                        r"early\s+marriage|married\s+off|child\s+marriage|traffick\w*|child\s+labou?r|kulijaya|"
                        r"lijaya|in\s+danger|neglect\w*|pregnan\w*)\b")
OLD_CHILD = re.compile(r"(?i)\b(child|children|learner|learners|pupil|pupils|girl|boy|student|mwana|bana|kasimbi|kamwana)\b")
OLD_WEAK = re.compile(r"(?i)\b(beat(s|en|ing)?|hit(s|ting)?|burn(t|ed|s)?|injur\w*|bruis\w*|wound\w*|scars?|"
                      r"touch(ed|es|ing)?\s+(her|him)|runs?\s+away|ran\s+away|missing|threat\w*|violen\w*|"
                      r"starv\w*|chased\s+from\s+home|sleeps?\s+outside|afraid\s+to\s+go\s+home|scared\s+to\s+go\s+home|"
                      r"hurt(s|ing)?\s+(her|him|them|me)|\w*m?uuma|kuuma|balamuuma)\b")


def old(text):
    low = (text or "").replace("’", "'")
    m = OLD_STRONG.search(low)
    if m:
        return m.group(0)
    if OLD_CHILD.search(low):
        m = OLD_WEAK.search(low)
        if m:
            return m.group(0) + " +child"
    return ""


def score(rows, fn):
    tot, hit = collections.Counter(), collections.Counter()
    fp, miss = [], []
    for r in rows:
        s = fn(r["text"])
        if r["label"] == "disclose":
            tot[r["cat"]] += 1
            hit[r["cat"]] += bool(s)
            if not s:
                miss.append(r)
        elif s:
            fp.append((r, s))
    tp = sum(hit.values())
    pos = sum(tot.values())
    neg = sum(1 for r in rows if r["label"] != "disclose")
    return {"per_cat": {c: f"{hit[c]}/{tot[c]}" for c in tot}, "recall": f"{tp}/{pos}", "recall_pct": round(100 * tp / pos, 1),
            "fp": f"{len(fp)}/{neg}", "precision_pct": round(100 * tp / (tp + len(fp)), 1) if tp + len(fp) else None,
            "fp_rows": fp, "miss_rows": miss}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(HERE, "dev_v1.jsonl")
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    fn = old if "--old" in sys.argv else S.safeguarding
    res = score(rows, fn)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_rows")}, indent=1))
    if "--show" in sys.argv:
        for r, s in res["fp_rows"]:
            print("FP  ", r["id"], r["text"], "->", s)
        for r in res["miss_rows"]:
            print("MISS", r["id"], r["text"], "| norm:", S.normalise(r["text"]))


if __name__ == "__main__":
    main()
