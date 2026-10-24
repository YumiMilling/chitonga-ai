"""Build the flywheel's hidden control items (24 Oct 2026, the Jesuit; FLYWHEEL.md section 'controls'; DECISIONS 24 Oct:
"no hidden controls in the late-night 'ok' runs: add controls to future batches").

A control is shown to a teacher EXACTLY like a derived line that needs a check ("This Tonga line needs a teacher's
check: <English> / <Tonga>. Right as is, or write your own?"). Two kinds, both made only of Tonga that already exists:
  right  a TEACHER-WRITTEN line (desk items of 21-24 Oct, kinds written / better), with its own English. A teacher who
         taps "Right as is" agrees. The prompting code prefers a line by ANOTHER speaker (author field), so a teacher
         is not just recognising their own sentence. Writing a different version is recorded, and scored as
         'rewrote', not as an error: speaker variation is not error (STRATEGY_TEACHER_DATA C.8).
  wrong  a 24 Oct synthetic number-pattern line (synthetic_t24.jsonl, transforms NUM:*, every word attested, tier A)
         paired with the English of a SIBLING built from the same teacher frame with a DIFFERENT number: "Stop at 9"
         beside the Tonga for "Stop at 10". The Tonga is real and grammatical; it just does not say what the English
         says, by exactly one number. A teacher who taps "Right as is" missed a meaning error; "Write it" caught it.
         (The number swap is the one transformation whose meaning change is certain; noun or tense swaps could be
         borderline, so they are not used.)
Never a control: a line of the stage 18 frozen teacher test or its dev slice (teacher_test_final18 / dev), or any line
the held-out guard knows. Controls are never exported as training data (flywheel_export.py drops them).

    python -X utf8 tools/chitonga/harness/build_controls.py [--out internal/chitonga_train/harness/flywheel_controls.jsonl]
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
RUN = os.path.join(CT, "runs", "2026-10-23_final_translator_data")
OUT = os.path.join(CT, "harness", "flywheel_controls.jsonl")
SYN = os.path.join(RUN, "synthetic_t24.jsonl")
ITEMS = [os.path.join(RUN, f) for f in ("teacher_items_2026-10-21_23.tsv", "teacher_items_2026-10-23_24.tsv")]
PER_FRAME = 4
FROZEN = [os.path.join(RUN, f) for f in ("teacher_test_final18.jsonl", "teacher_dev_final18.jsonl")]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", (s or "").lower())).strip()


def frozen_keys():
    keys = set()
    for p in FROZEN:
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                for s in [r.get("en"), r.get("toi")] + list(r.get("alternatives") or []):
                    if isinstance(s, str) and s.strip():
                        keys.add(norm(s))
    return keys


def guard():
    sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "train"))
    import heldout_guard as HG  # noqa: PLC0415
    return HG.load(RUN)


def known_right(frozen, G, limit):
    out = []
    for p in ITEMS:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                en, toi = (r.get("en") or "").strip(), (r.get("toi") or "").strip()
                if r.get("kind") not in ("written", "better") or not en or not toi:
                    continue
                if any(c in toi for c in "/()[]") or not (3 <= len(toi.split()) <= 10) or len(en.split()) > 12:
                    continue                      # one plain sentence, no alternatives a reader could stumble on
                if norm(en) in frozen or norm(toi) in frozen or G.hit(toi, en):
                    continue
                out.append({"en": en, "toi": toi, "author": r.get("teacher"), "src": r.get("id"),
                            "domain": r.get("domain")})
    seen, keep = set(), []
    by_author = collections.defaultdict(list)
    for x in out:
        if norm(x["toi"]) not in seen:
            seen.add(norm(x["toi"]))
            by_author[x["author"]].append(x)
    # alternate authors so either teacher gets the other's lines
    authors = sorted(by_author)
    while len(keep) < limit and any(by_author.values()):
        for a in authors:
            if by_author[a] and len(keep) < limit:
                keep.append(by_author[a].pop(0))
    return keep


def known_wrong(frozen, G, limit):
    rows = [json.loads(x) for x in open(SYN, encoding="utf-8") if x.strip()]
    groups = collections.defaultdict(list)
    for r in rows:
        t = r.get("transform") or ""
        if not t.startswith("NUM:") or t.startswith("NUM:F08") or r.get("tier") != "A":
            continue                               # F08 (arithmetic sentences) reads oddly out of context
        groups[r["template"]].append(r)
    out = []
    for tpl, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["id"])
        k = 0
        for a, b in zip(rs, rs[1:] + rs[:1]):
            if a is b or a["toi"] == b["toi"] or a["en"] == b["en"]:
                continue
            if any(norm(s) in frozen for s in (a["en"], b["toi"])) or G.hit(b["toi"], a["en"]):
                continue
            out.append({"en": a["en"], "toi": b["toi"], "right_toi": a["toi"], "author": b.get("teacher"),
                        "src": f"{b['id']} toi beside {a['id']} en", "transform": b["transform"],
                        "why": f"the Tonga says the number of '{b['en']}', the English says '{a['en']}'"})
            k += 1
            if k >= PER_FRAME:
                break                              # a few per teacher frame: variety over volume
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--right", type=int, default=30)
    ap.add_argument("--wrong", type=int, default=20)
    a = ap.parse_args()
    fz, G = frozen_keys(), guard()
    right = [dict(x, id=f"ctl:r{i:03d}", expect="right") for i, x in enumerate(known_right(fz, G, a.right))]
    wrong = [dict(x, id=f"ctl:w{i:03d}", expect="wrong") for i, x in enumerate(known_wrong(fz, G, a.wrong))]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for x in right + wrong:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(right)} known-right ({collections.Counter(x['author'] for x in right)}), {len(wrong)} known-wrong -> "
          f"{a.out}")


if __name__ == "__main__":
    main()
