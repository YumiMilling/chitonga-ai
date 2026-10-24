"""The weakness map of the installed translator (22 Oct 2026, the Jesuit; Chris: "We want to find the weaknesses of the
translation model, which can probably also help with the chat model").

Reads (after score.py has opened the key, so the labels were all written blind):
  diagnosis_500_2026-10-21/pack/<pack>/items.jsonl, KEY_<pack>_do_not_open.json, judged/<pack>__claude.jsonl
  for pack in (a050_vs_s15b, a050_vs_s15b_toi2en); systems a050_served (what users get), a050_raw (same weights, the
  15b protocol: one beam-4 decode per line), s15b_soup1000 (this morning's map).
  The 21 Oct Codex / Fable labels of s15b (judged/s15b_vs_R*__codex|fable.jsonl) for the judge cross-check.
  taught_items.json (taught_items.py): items whose judged error words were in stage 16's vocab group.
Writes analysis.json and prints the tables WEAKNESSES.md quotes.

Paired comparisons are within ONE judge (Claude, one pass, the texts side by side under shuffled letters), so the
judge's strictness cancels; the absolute rates are Claude's and are compared with Codex / Fable only as a cross-check.
"""
import collections
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
DIAG = os.path.join(REPO, "internal", "chitonga_train", "llm", "panel", "diagnosis_500_2026-10-21")
RANK = {"right": 2, "partly": 1, "wrong": 0}
SYS = ("a050_served", "a050_raw", "s15b_soup1000")
VOC = {"vocabulary", "word_sense", "untranslated"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def sign_p(b, w):
    n = b + w
    if not n:
        return 1.0
    k = min(b, w)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def kappa(pairs, cats):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def load_pack(pack):
    items = {x["item"]: x for x in jl(os.path.join(DIAG, "pack", pack, "items.jsonl"))}
    key = json.load(open(os.path.join(DIAG, f"KEY_{pack}_do_not_open.json"), encoding="utf-8"))["key"]
    lab = {x["item"]: x["labels"] for x in jl(os.path.join(DIAG, "judged", f"{pack}__claude.jsonl"))}
    per = {}
    for i in items:
        per[i] = {}
        for L, ss in key[i].items():
            for s in ss:
                s = {"a050_srv_gpu": "a050_served"}.get(s, s)   # pass B decoded on the pod (GPU int8)
                per[i][s] = dict(lab[i][L], text=items[i]["candidates"][L], letter=L)
    return items, per


def load(packA, packB):
    """Two passes, every comparison inside one pass. Pass A (all 500): a050_raw and s15b side by side. Pass B (only the
    lines where the served text differs from the raw decode): served, raw and s15b side by side. Final triple per item:
    pass B where the item is in it, else pass A with served = raw (the same text). Also returns the test-retest pairs
    (the same raw / s15b text labelled in both passes)."""
    itemsA, perA = load_pack(packA)
    per = {i: dict(v, a050_served=v["a050_raw"]) for i, v in perA.items()}
    retest = []
    if os.path.exists(os.path.join(DIAG, "judged", f"{packB}__claude.jsonl")):
        _, perB = load_pack(packB)
        for i, v in perB.items():
            for s in ("a050_raw", "s15b_soup1000"):
                retest.append((perA[i][s]["label"], v[s]["label"]))
            per[i] = v
        inB = set(perB)
    else:
        inB = None
    return itemsA, per, retest, inB


def old_labels(pack, judge):
    """21 Oct labels of s15b_soup1000 by codex / fable"""
    key = json.load(open(os.path.join(DIAG, f"KEY_{pack}_do_not_open.json"), encoding="utf-8"))["key"]
    lab = {x["item"]: x["labels"] for x in jl(os.path.join(DIAG, "judged", f"{pack}__{judge}.jsonl"))}
    out = {}
    for i, k in key.items():
        for L, ss in k.items():
            if "s15b_soup1000" in ss:
                out[i] = lab[i][L]
    return out


def errs(v):
    return set(v.get("errors") or [])


def bw(per, ids, x, y):
    b = w = t = 0
    for i in ids:
        d = RANK[per[i][x]["label"]] - RANK[per[i][y]["label"]]
        b, w, t = b + (d > 0), w + (d < 0), t + (d == 0)
    return {"better": b, "worse": w, "tie": t, "sign_p": sign_p(b, w)}


def main():
    meta = {r["id"]: r for r in jl(os.path.join(DIAG, "set.jsonl"))}
    taught = json.load(open(os.path.join(HERE, "taught_items.json"), encoding="utf-8"))["taught"]
    A = {}
    for (pA, pB), d, old in ((("a050raw_vs_s15b", "a050srv_vs_raw"), "en>toi", "s15b_vs_R"),
                             (("a050raw_vs_s15b_toi2en", "a050srv_vs_raw_toi2en"), "toi>en", "s15b_vs_R_toi2en")):
        items, per, retest, inB = load(pA, pB)
        ids = sorted(items)
        n = len(ids)
        res = {"n": n, "pass_B_items": None if inB is None else len(inB)}
        if retest:
            res["test_retest"] = {"pairs": len(retest), "same": sum(a == b for a, b in retest),
                                  "kappa_3": kappa(retest, tuple(RANK)),
                                  "diff": dict(collections.Counter(f"{a}>{b}" for a, b in retest if a != b))}
        res["labels"] = {s: dict(collections.Counter(per[i][s]["label"] for i in ids)) for s in SYS}
        res["right_ci"] = {s: wilson(res["labels"][s].get("right", 0), n) for s in SYS}
        res["wrong_ci"] = {s: wilson(res["labels"][s].get("wrong", 0), n) for s in SYS}
        res["identical"] = {f"{x}={y}": sum(per[i][x]["text"] == per[i][y]["text"] for i in ids)
                            for x, y in (("a050_served", "a050_raw"), ("a050_raw", "s15b_soup1000"),
                                         ("a050_served", "s15b_soup1000"))}
        res["better_worse"] = {f"{x} vs {y}": bw(per, ids, x, y) for x, y in
                               (("a050_served", "s15b_soup1000"), ("a050_raw", "s15b_soup1000"), ("a050_served", "a050_raw"))}
        T = collections.defaultdict(collections.Counter)
        for i in ids:
            for s in SYS:
                for e in errs(per[i][s]):
                    T[s][e] += 1
        res["error_types"] = {s: dict(T[s].most_common()) for s in SYS}
        fr = {}
        for e in sorted(set(T["s15b_soup1000"]) | set(T["a050_served"])):
            o = {i for i in ids if e in errs(per[i]["s15b_soup1000"])}
            nw = {i for i in ids if e in errs(per[i]["a050_served"])}
            fr[e] = {"fixed": len(o - nw), "remaining": len(o & nw), "new": len(nw - o),
                     "new_items": sorted(nw - o)[:40]}
        res["fixed_remaining_new"] = fr
        tr = collections.Counter((per[i]["s15b_soup1000"]["label"], per[i]["a050_served"]["label"]) for i in ids)
        res["transitions_15b_to_a050served"] = {f"{a}>{b}": c for (a, b), c in sorted(tr.items())}
        sp = {}
        for field in ("domain", "length", "input_kind", "source", "en_by"):
            g = collections.defaultdict(list)
            for i in ids:
                g[str(meta[i][field])[:30]].append(i)
            sp[field] = {k: {"n": len(v), **{s: round(sum(per[i][s]["label"] == "right" for i in v) / len(v), 3) for s in SYS},
                             "wrong_a050_served": sum(per[i]["a050_served"]["label"] == "wrong" for i in v),
                             "bw_served_vs_15b": [bw(per, v, "a050_served", "s15b_soup1000")[k2] for k2 in ("better", "worse")]}
                         for k, v in sorted(g.items())}
        res["splits"] = sp
        vd = {}
        for dom in sorted({meta[i]["domain"] for i in ids}):
            v = [i for i in ids if meta[i]["domain"] == dom]
            vd[dom] = {s: f"{sum('vocabulary' in errs(per[i][s]) for i in v)}/{len(v)}" for s in SYS}
        res["vocabulary_by_domain"] = vd
        tt = {}
        for name, v in (("taught", [i for i in ids if i in taught]), ("not_taught", [i for i in ids if i not in taught])):
            v15 = [i for i in v if "vocabulary" in errs(per[i]["s15b_soup1000"])]
            tt[name] = {"n": len(v), "bw_served_vs_15b": bw(per, v, "a050_served", "s15b_soup1000"),
                        "bw_raw_vs_15b": bw(per, v, "a050_raw", "s15b_soup1000"),
                        "vocab_err_15b": len(v15),
                        "vocab_err_a050_served": sum("vocabulary" in errs(per[i]["a050_served"]) for i in v),
                        "vocab_err_15b_gone_in_a050": sum("vocabulary" not in errs(per[i]["a050_served"]) for i in v15),
                        "vocab_err_new_in_a050": sum("vocabulary" in errs(per[i]["a050_served"]) and
                                                     "vocabulary" not in errs(per[i]["s15b_soup1000"]) for i in v),
                        "right_15b": sum(per[i]["s15b_soup1000"]["label"] == "right" for i in v),
                        "right_a050_served": sum(per[i]["a050_served"]["label"] == "right" for i in v)}
        res["taught_split"] = tt
        xc = {}
        for j in ("codex", "fable"):
            o = old_labels(old, j)
            pairs = [(per[i]["s15b_soup1000"]["label"], o[i]["label"]) for i in ids]
            xc[j] = {"kappa_3": kappa(pairs, tuple(RANK)), "raw": round(sum(a == b for a, b in pairs) / n, 3),
                     "right_claude": sum(a == "right" for a, _ in pairs), f"right_{j}": sum(b == "right" for _, b in pairs),
                     "wrong_claude": sum(a == "wrong" for a, _ in pairs), f"wrong_{j}": sum(b == "wrong" for _, b in pairs)}
        res["judge_crosscheck_s15b"] = xc
        rec = []
        for i in ids:
            v = per[i]["a050_served"]
            if v["label"] != "right":
                rec.append({"item": i, "domain": meta[i]["domain"], "length": meta[i]["length"],
                            "input_kind": meta[i]["input_kind"], "label": v["label"], "errors": v.get("errors"),
                            "error_words": v.get("error_words"), "fix": v.get("fix"), "note": v.get("note"),
                            "sure": v.get("sure"), "src": items[i].get("en_intended") or items[i].get("source_toi"),
                            "ref": items[i].get("reference_toi") or items[i].get("reference_en"), "out": v["text"],
                            "s15b_label": per[i]["s15b_soup1000"]["label"], "s15b_errors": per[i]["s15b_soup1000"].get("errors"),
                            "taught": i in taught})
        res["a050_error_records"] = rec
        A[d] = res
    json.dump(A, open(os.path.join(HERE, "analysis.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for d, res in A.items():
        print("=====", d)
        for k in ("pass_B_items", "test_retest", "labels", "right_ci", "wrong_ci", "identical", "better_worse", "transitions_15b_to_a050served",
                  "error_types", "vocabulary_by_domain", "taught_split", "judge_crosscheck_s15b"):
            print(k, json.dumps(res.get(k), ensure_ascii=False))
        print("fixed_remaining_new", json.dumps({k: {x: y for x, y in v.items() if x != "new_items"}
                                                 for k, v in res["fixed_remaining_new"].items()}))
        for f in ("length", "domain", "input_kind", "en_by", "source"):
            print(f, json.dumps(res["splits"][f], ensure_ascii=False))


if __name__ == "__main__":
    main()
