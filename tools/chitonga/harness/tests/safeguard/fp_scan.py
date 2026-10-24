"""False-positive scan of the safeguarding detector over the lesson-request corpora we hold that are NOT bake-off test
rows (24 Oct 2026, the Jesuit): the plug-in's frozen test set (34), the struggle phrasings (struggle_eval CASES +
HELDOUT), the search queries (web/tests/search_queries.jsonl), the timetable notes (notes_digest_eval.NOTES), the
scope unit lists (h3_tests/unit_scope.py: teacher tasks, advice, social, out of scope), every item / unit title of the
plug-in, and the Tonga text of the plug-in's Tonga lessons (does the fuzzy step turn Tonga into an English key word?).

Prints, per corpus: n, caught by the old gate, caught by the new one, and every message the NEW gate catches that the
old did not (the 'newly sent to safeguarding' list a person must read).

    python -X utf8 tools/chitonga/harness/tests/safeguard/fp_scan.py [--all]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, "..", ".."))
LLM = os.path.normpath(os.path.join(HARNESS, "..", "llm"))
sys.path.insert(0, HARNESS)
sys.path.insert(0, LLM)
sys.path.insert(0, HERE)
import safeguard as S  # noqa: E402
from score import old  # noqa: E402


def corpora():
    from plugins import education  # noqa: PLC0415
    from plugins.education import struggle_eval as SE  # noqa: PLC0415
    from plugins.education import notes_digest_eval as NE  # noqa: PLC0415
    P, _ = education.load()
    out = {"plugin_testset": [t["text"] for t in P.testset],
           "struggle": [c[0] for c in SE.CASES] + [c[0] for c in getattr(SE, "HELDOUT", [])],
           "search_queries": [json.loads(x)["q"] for x in open(os.path.join(HARNESS, "web", "tests", "search_queries.jsonl"),
                                                                encoding="utf-8") if x.strip()],
           "notes": [n[6] for n in NE.NOTES if isinstance(n[6], str)]}
    src = open(os.path.join(LLM, "h3_tests", "unit_scope.py"), encoding="utf-8").read()
    lists = []
    for name in ("TEACHER", "OUT", "SOCIAL", "ADVICE"):
        m = re.search(name + r" = \[(.*?)\n\]", src, re.S)
        if m:
            lists += re.findall(r'"([^"]+)"', m.group(1))
    out["unit_scope_lists"] = lists
    titles = []
    for it in P.items.values():
        if it.get("title"):
            titles.append(str(it["title"]))
    for u in P.units.values():
        if u.get("title"):
            titles.append(str(u["title"]))
    out["plugin_titles"] = sorted(set(titles))
    toi, en = [], []
    for it in P.items.values():
        for k in ("lines", "key_words"):
            for x in it.get(k) or []:
                if isinstance(x, dict):
                    if isinstance(x.get("toi"), str) and x["toi"].strip():
                        toi.append(x["toi"])
                    if isinstance(x.get("en"), str) and x["en"].strip():
                        en.append(x["en"])
    out["plugin_lesson_lines_en"] = sorted(set(en))
    out["plugin_tonga_text"] = sorted(set(toi))
    return out


def main():
    show_all = "--all" in sys.argv
    tot_new = 0
    for name, texts in corpora().items():
        o = sum(1 for t in texts if old(t))
        n = [t for t in texts if S.safeguarding(t)]
        newly = [t for t in texts if S.safeguarding(t) and not old(t)]
        gone = [t for t in texts if old(t) and not S.safeguarding(t)]
        tot_new += len(newly)
        print(f"{name}: n {len(texts)}, old gate {o}, new gate {len(n)}, newly caught {len(newly)}, no longer caught {len(gone)}")
        for t in newly[: (None if show_all else 25)]:
            print("   NEW ", t[:110].replace("\n", " "), "->", S.safeguarding(t))
        for t in gone[: (None if show_all else 10)]:
            print("   GONE", t[:110].replace("\n", " "), "-> old:", old(t))
    print("total newly caught:", tot_new)


if __name__ == "__main__":
    main()
