"""The concept terms of the teacher search (24 Oct 2026, the Jesuit; core/search_index.py).

For every servable item, a short list of what a teacher might TYPE to find it that its title does not say: the
subject's other names (borrowing / regrouping), the sub-topics inside it (heart, lungs, stomach in "Body: inside
parts"), Zambian English (sums, pupils). Written by Claude (K0, in the session) from the item's own title, key words
and text; conservative: related, never loosely associated (a broader topic goes in search_lexicon.json 'rel', where it
is shown to the teacher as "related", not as a match). Stored in the plug-in as search_concepts.json:
{"note", "by", "items": {item id: [phrases]}}.

    python -X utf8 tools/chitonga/harness/web/search_concepts.py missing [--out FILE]   items with no concept terms yet,
                                                                                         as JSON lines to write them from
    python -X utf8 tools/chitonga/harness/web/search_concepts.py merge FILE             "item id: phrase; phrase" lines
    python -X utf8 tools/chitonga/harness/web/search_concepts.py sample N               N random items with their terms,
                                                                                         for review

An item with no concept terms is still found by its title, key words and body; run `missing` after a plug-in rebuild.
The Chitonga competences (the CDC module's Tonga) get none: the heading glosses in core/search_index.TOI_HEADINGS only.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)


def load_plugin():
    from plugins import education  # noqa: PLC0415
    P, _ = education.load()
    return P


def tips_of(P):
    p = os.path.join(P.dir, "teaching_tips.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def path_of(P):
    return os.path.join(P.dir, "search_concepts.json")


def load(P):
    try:
        with open(path_of(P), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"note": __doc__.split("\n\n")[1].replace("\n", " "), "by": "", "items": {}}


def save(P, obj):
    obj["items"] = dict(sorted(obj["items"].items()))
    with open(path_of(P), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)


def main():
    from core import search_index as S  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("missing")
    m.add_argument("--out")
    g = sub.add_parser("merge")
    g.add_argument("file")
    g.add_argument("--by", default="Claude (Opus), in the session, from each item's own title, key words and text")
    s = sub.add_parser("sample")
    s.add_argument("n", type=int)
    a = ap.parse_args()
    P = load_plugin()
    docs = S.documents(P, tips_of(P))
    obj = load(P)
    if a.cmd == "missing":
        rows = [S.concept_text(d) for d in docs if d["id"] not in obj["items"] and d.get("lang") != "toi"]
        out = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(rows)} of {len(docs)} servable items have no concept terms", file=sys.stderr)
    elif a.cmd == "merge":
        ids = {d["id"] for d in docs}
        n = bad = 0
        with open(a.file, encoding="utf-8") as f:
            for line in f:
                if ":" not in line or line.startswith("#"):
                    continue
                iid, _, rest = line.rstrip("\n").rpartition(": ")
                if iid not in ids:
                    bad += 1
                    print("not a servable item:", iid, file=sys.stderr)
                    continue
                obj["items"][iid] = [p.strip() for p in rest.split(";") if p.strip()]
                n += 1
        obj["by"] = a.by
        save(P, obj)
        print(f"merged {n} items ({bad} unknown ids); {len(obj['items'])} items have concept terms")
    else:
        have = [d for d in docs if d["id"] in obj["items"]]
        random.seed(a.n)
        for d in random.sample(have, min(a.n, len(have))):
            print(f"{d['id']} | {d['title']} | grades {d['grades']}\n    -> {'; '.join(obj['items'][d['id']])}")


if __name__ == "__main__":
    main()
