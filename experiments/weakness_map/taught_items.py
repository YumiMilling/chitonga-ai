"""Which diagnosis items had their error words in stage 16's vocabulary group (22 Oct 2026, the Jesuit).

Why: stage 16 (half of the installed a050 blend) trained a 'vocab' group built FROM this diagnosis's judged error words
(stage16_vocab.py: words the 21 Oct judges named; real human sentences from OUTSIDE the diagnosis sections, never the
diagnosis lines). A vocabulary gain on those items is partly teaching to the test. This re-runs stage16_vocab's own word
extraction (words_from_diag, same stop list) to get item -> words, and marks an item TAUGHT when at least one of its words
got real training sentences (targets.jsonl used > 0). Output: taught_items.json {item: [words]} (taught only) + counts.
"""
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "train"))
import stage16_vocab as V  # noqa: E402

pos = collections.defaultdict(set)
for r in csv.DictReader(open(os.path.join(REPO, "tools", "chitonga", "grammar2", "lexicon.tsv"), encoding="utf-8"),
                        delimiter="\t", quoting=csv.QUOTE_NONE):
    pos[(r["word"] or "").lower()].add(r["pos"] or "")
FUNC = {"dem", "poss", "pron", "conj", "cop", "enum", "num", "n.cop", "prep", "interj"}
PART = set("kuti pe alimwi mbuli inga kuli nkaambo naa kutegwa kuzwa eeci mpoonya nokuba antoomwe kwaamba ambila "
           "nkokuti aboobo elyo lino kale buyo kapati".split())
stop = PART | {w for w, ps in pos.items() if ps and ps <= FUNC}
found = V.words_from_diag(stop)
T = {json.loads(l)["word"]: json.loads(l) for l in open(os.path.join(REPO, "internal", "chitonga_train", "runs",
                                                                     "2026-10-21_stage16", "vocab", "targets.jsonl"), encoding="utf-8")}
# the lexicon changed since 21 Oct (stop list): report how many words differ from the stage 16 targets
missing = sorted(set(T) ^ set(found))
taught = collections.defaultdict(list)
for w, f in found.items():
    if T.get(w, {}).get("used", 0) > 0:
        for it in f["items"]:
            taught[it].append(w)
json.dump({"taught": {k: sorted(v) for k, v in sorted(taught.items())}, "n_taught": len(taught),
           "words_found_now": len(found), "words_stage16": len(T), "word_set_differences": missing[:50],
           "n_word_set_differences": len(missing)},
          open(os.path.join(HERE, "taught_items.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("taught items", len(taught), "words now", len(found), "stage16 targets", len(T), "differences", len(missing))
