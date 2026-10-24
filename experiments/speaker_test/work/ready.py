"""drafts/READY.json: which draft messages are translated AND judged-and-stamped (so they may be sent), with the
draft's sha256 and the stamp that covers it (24 Oct 2026, the Jesuit). The coordinator reads this file; nothing is sent
from here. A message is ready only when every item and known-wrong control in it is in a stamped judge file (repeats
are covered by their original's stamp).
    python -X utf8 ready.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
D = os.path.join(RUN, "drafts")


def main():
    key = json.load(open(os.path.join(D, "KEY_lines.json"), encoding="utf-8"))
    st = [json.loads(l) for l in open(os.path.join(RUN, "ai_judge", "STAMPS.jsonl"), encoding="utf-8") if l.strip()]
    cover = {}
    for s in st:
        for x in s["items"]:
            cover.setdefault(x, s)
    out = {}
    for m, lines in sorted(key.items()):
        need = []
        for L in lines:
            if L.get("item"):
                need.append(L["item"])
            elif L.get("kind") == "known_wrong":
                need.append(L["control"])
            else:
                need.append(L["en_id"])
        miss = [x for x in need if x not in cover]
        p = os.path.join(D, m + ".txt")
        out[m] = {"ready": not miss, "draft_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                  "ids": [L.get("item") or L.get("control") for L in lines],
                  "stamps": sorted({cover[x]["file"] + " " + cover[x]["utc"] for x in need if x in cover}),
                  "not_judged": miss}
    json.dump(out, open(os.path.join(D, "READY.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({m: v["ready"] for m, v in out.items()})


if __name__ == "__main__":
    main()
