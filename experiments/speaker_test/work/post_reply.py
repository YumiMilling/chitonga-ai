"""After the replies: the two claude -p passes RULE.md names (24 Oct 2026, the Jesuit). Run only after score.py has
parsed real replies (it reads result.json and the replies folder through score.py's parser).

  types       RULE 7: for every corrected primary line, OUR error types from the fixed list, by comparing a050's line
              with the speaker's line (English, a050, speaker; blind to the AI judge). -> error_typing/typed_<utc>.jsonl
  fixcompare  RULE 6 (b): where the AI judge and the speaker both corrected a line, Claude's fix and the speaker's line
              as A / B in seeded random order: "same" (same meaning and wording), "other_wording" (same meaning),
              "different" (different meaning). -> ai_judge/fixcompare_<utc>.jsonl
Claude's fixes stay evaluation only (never shown to a speaker, served, trained on, or put in the memory).
    python -X utf8 post_reply.py types|fixcompare [--replies DIR]
"""
import glob
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
sys.path.insert(0, RUN)
import score as S  # noqa: E402

TYPES = ("negation tense_aspect agreement vocabulary word_sense omission addition names_numbers structure untranslated "
         "garbled").split()
P_TYPES = """Each line of items.jsonl has an English classroom sentence (`en`), a machine translation into Chitonga
(`machine`) and a native speaker's corrected Chitonga (`speaker`). The speaker's line is right. Compare the machine line
with the speaker's line and say WHAT the machine got wrong, using only this fixed list (all that apply, most important
first): negation, tense_aspect, agreement, vocabulary, word_sense, omission, addition, names_numbers, structure,
untranslated, garbled. If the two differ only in wording or spelling with the same meaning, give ["wording_only"].
Write ONLY judged.jsonl, one line per item in order: {"id": "...", "errors": [...], "words": ["machine words at fault"],
"note": "<one short sentence>"}. Then reply DONE <n>."""
P_FIX = """Each line of items.jsonl has an English classroom sentence (`en`) and two Chitonga lines, A and B, in random
order. For each, say how A and B relate: "same" (same meaning and practically the same wording), "other_wording" (same
meaning, different wording), or "different" (the meaning differs). Do not judge which is better. Write ONLY
judged.jsonl, one line per item in order: {"id": "...", "verdict": "same|other_wording|different", "note": "..."}. Then
reply DONE <n>."""


def run(prompt, items, tag):
    wd = os.path.join(os.environ.get("TEMP", "/tmp"), "claude_g14_post", f"{tag}_{int(time.time())}")
    os.makedirs(wd)
    with open(os.path.join(wd, "items.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    open(os.path.join(wd, "prompt.md"), "w", encoding="utf-8").write(prompt)
    subprocess.run(["claude", "-p", "Read prompt.md and do exactly what it says.", "--model", "claude-opus-5-5",
                    "--permission-mode", "acceptEdits", "--disallowedTools", "Bash,WebFetch,WebSearch,Task"],
                   cwd=wd, capture_output=True, text=True, encoding="utf-8", timeout=5400)
    return S.jl(os.path.join(wd, "judged.jsonl"))


def main():
    mode = sys.argv[1]
    rep = sys.argv[sys.argv.index("--replies") + 1] if "--replies" in sys.argv else os.path.join(RUN, "replies")
    sv = {r["id"]: r for r in S.jl(os.path.join(RUN, "served.jsonl"))}
    se = {r["id"]: r for r in S.jl(os.path.join(RUN, "sentences.jsonl"))}
    key = json.load(open(os.path.join(RUN, "drafts", "KEY_lines.json"), encoding="utf-8"))
    sp = json.load(open(os.path.join(RUN, "split.json"), encoding="utf-8"))
    prim = {}
    for p in sorted(glob.glob(os.path.join(rep, "T0*_msg*.txt"))):
        m = os.path.basename(p)[:-4]
        if m not in key:
            continue
        ours = {x["line"]: sv.get(x.get("item") or x.get("toi_id"), {}).get("toi") for x in key[m]}
        got = S.parse_reply(open(p, encoding="utf-8").read(), ours)
        for x in key[m]:
            if x.get("item") and x["line"] in got and sp["shared_primary"].get(x["item"], m[:3]) == m[:3]:
                prim[x["item"]] = got[x["line"]]
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if mode == "types":
        items = [{"id": i, "en": se[i]["en"], "machine": sv[i]["toi"], "speaker": l[1]} for i, l in prim.items()
                 if l[0] != "right" and l[1]]
        out = run(P_TYPES, items, "types")
        os.makedirs(os.path.join(RUN, "error_typing"), exist_ok=True)
        dest = os.path.join(RUN, "error_typing", f"typed_{utc}.jsonl")
        hdr = {"_header": "OUR error types (claude -p), assigned by comparing a050's line with the speaker's line; NOT the speakers' labels."}
    else:
        akey = json.load(open(os.path.join(RUN, "ai_judge", "KEY_do_not_open.json"), encoding="utf-8"))
        ai = {}
        for p in sorted(glob.glob(os.path.join(RUN, "ai_judge", "batch*.jsonl"))):
            for r in S.jl(p):
                if r.get("item") in akey:
                    ai.setdefault(akey[r["item"]], r)
        rng = random.Random(11)
        items = []
        for i, l in prim.items():
            if l[0] != "right" and l[1] and i in ai and ai[i]["label"] != "right" and ai[i].get("fix"):
                pair = [ai[i]["fix"], l[1]]
                rng.shuffle(pair)
                items.append({"id": i, "en": se[i]["en"], "A": pair[0], "B": pair[1]})
        out = run(P_FIX, items, "fix")
        dest = os.path.join(RUN, "ai_judge", f"fixcompare_{utc}.jsonl")
        hdr = {"_header": "Blind second-pass reading of Claude's fix vs the speaker's line. Claude's fixes are EVALUATION ONLY."}
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(hdr) + "\n")
        for r in out:
            r["item"] = r.get("id")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dest, len(items), "items,", len(out), "answered")


if __name__ == "__main__":
    main()
