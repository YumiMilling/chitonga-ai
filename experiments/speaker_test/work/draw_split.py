"""Seeded draw and split for the speaker test (24 Oct 2026, the Jesuit).

1. Pool minus every held-out hit (heldout_report.json, near hits included: conservative).
2. random.Random(SEED).shuffle of the clean pool ids (sorted first): the order IS the draw. The first 200 are the drawn
   set; walking that order, a sentence the live service would serve from memory (whole line or a piece; checked with
   translator_service.Service.memory, the service's own lookup) is skipped and recorded; the first 170 kept are the test,
   the next drawn ones are reserves (known-wrong controls are built from reserves only).
3. Split 170 = 30 shared + 70 T01 + 70 T09, stratified by domain (round-robin over each domain's ids in draw order).
   The 30 shared: a seeded coin gives each a PRIMARY speaker (15 / 15), so every item has exactly one primary judgment
   (85 per speaker) for the pooled estimate; the other judgment counts for agreement only.
4. Controls per speaker (hidden): 2 repeats of the speaker's own items (an item from the first three messages repeated
   at least five messages later) and 2 known-wrong lines (English of one reserve with the a050 Tonga of a DIFFERENT
   reserve, different domain; real served Tonga, never composed).
5. Order per speaker: seeded shuffle; messages of 8; message 1 holds 2 shared items.
"""
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(RUN, "..", "..", "..", ".."))
SEED = 20261024
sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "llm"))


def main():
    pool = [json.loads(l) for l in open(os.path.join(RUN, "pool.jsonl"), encoding="utf-8")]
    hits = json.load(open(os.path.join(RUN, "heldout_report.json"), encoding="utf-8"))["hits"]
    clean = sorted(r["id"] for r in pool if r["id"] not in hits)
    byid = {r["id"]: r for r in pool}
    order = list(clean)
    random.Random(SEED).shuffle(order)
    drawn = order[:200]
    import translator_service as TS
    svc = TS.Service()
    test, mem_skipped, reserves = [], [], []
    for pid in order:
        en = byid[pid]["en"]
        hit = svc.memory(en, "en>toi")
        pieces = [p for p in TS.sentences(en)] if hasattr(TS, "sentences") else [en]
        phit = [p for p in pieces if p != en and svc.memory(p, "en>toi")]
        if hit or phit:
            mem_skipped.append({"id": pid, "en": en, "memory": (hit or {}).get("src") if hit else None, "piece_hits": phit})
            continue
        if len(test) < 170:
            test.append(pid)
        elif len(reserves) < 40:
            reserves.append(pid)
        else:
            break
    # split, stratified by domain
    dom = defaultdict(list)
    for pid in test:
        dom[byid[pid]["domain"]].append(pid)
    seq = []                         # round-robin over domains, in draw order: shared / T01 / T09 interleaved per domain
    for d in sorted(dom):
        seq += dom[d]
    shared, t01, t09 = [], [], []
    slots = ["S", "A", "B", "A", "B", "A", "B"]   # per 17 items: 3 S + 7 A + 7 B -> 30 / 70 / 70 over 170
    cyc = (["S"] * 3 + ["A", "B"] * 7)
    rng = random.Random(SEED + 1)
    for d in sorted(dom):
        ids = dom[d]
        for i, pid in enumerate(ids):
            pass
    # exact counts with domain balance: assign per domain proportionally, remainder by a running deficit
    targets = {"S": 30, "A": 70, "B": 70}
    got = {"S": 0, "A": 0, "B": 0}
    k = 0
    for d in sorted(dom, key=lambda d: -len(dom[d])):
        for pid in dom[d]:
            k += 1
            # the group furthest behind its share at this point
            g = max(("S", "A", "B"), key=lambda g: (targets[g] * k / 170 - got[g], -("SAB".index(g))))
            got[g] += 1
            {"S": shared, "A": t01, "B": t09}[g].append(pid)
    assert (len(shared), len(t01), len(t09)) == (30, 70, 70), (len(shared), len(t01), len(t09))
    coin = list(range(30))
    rng.shuffle(coin)
    primary = {shared[i]: ("T01" if j < 15 else "T09") for j, i in enumerate(coin)}
    # controls: known-wrong pairs from reserves (different domains), built now; Tonga comes from served.jsonl later
    res = list(reserves)
    kw = []
    used = set()
    for i, a in enumerate(res):
        if a in used:
            continue
        for b in res[i + 1:]:
            if b in used or byid[b]["domain"] == byid[a]["domain"]:
                continue
            kw.append({"en_from": a, "toi_from": b})
            used |= {a, b}
            break
        if len(kw) == 4:
            break
    assert len(kw) == 4
    plan = {}
    for spk, own, kws in (("T01", t01, kw[:2]), ("T09", t09, kw[2:])):
        mine_shared = list(shared)
        r2 = random.Random(f"{SEED}-{spk}")
        body = own + mine_shared
        r2.shuffle(body)
        # message 1: 2 shared + 6 own
        first = [x for x in body if x in shared][:2] + [x for x in body if x not in shared][:6]
        rest = [x for x in body if x not in first]
        seqs = first + rest                       # 100 items
        msgs = [seqs[i:i + 8] for i in range(0, 96, 8)] + [seqs[96:]]   # 12 x 8 + 4
        # repeats: one item from message 2, one from message 3, repeated in messages 9 and 11 (>= 5 messages apart)
        rep = [msgs[1][3], msgs[2][5]]
        ctl = [{"cid": f"{spk}-rep1", "kind": "repeat", "of": rep[0]}, {"cid": f"{spk}-rep2", "kind": "repeat", "of": rep[1]},
               {"cid": f"{spk}-kw1", "kind": "known_wrong", **kws[0]}, {"cid": f"{spk}-kw2", "kind": "known_wrong", **kws[1]}]
        msgs[8].insert(5, ctl[0]["cid"])
        msgs[10].insert(2, ctl[1]["cid"])
        msgs[5].insert(6, ctl[2]["cid"])
        msgs[11].insert(1, ctl[3]["cid"])
        # rebalance to <= 9 per message: the last message takes overflow
        flat = [x for m in msgs for x in m]
        msgs = [flat[i:i + 8] for i in range(0, len(flat), 8)]
        plan[spk] = {"messages": msgs, "controls": ctl, "n_items": len(flat)}
    out = {"seed": SEED, "pool": len(pool), "heldout_hits_dropped": len(hits), "clean": len(clean),
           "draw_order_first_200": drawn, "memory_skipped": mem_skipped, "test": test, "reserves": reserves,
           "shared": shared, "shared_primary": primary, "unique": {"T01": t01, "T09": t09},
           "by_domain": {d: {"shared": sum(1 for x in shared if byid[x]["domain"] == d),
                             "T01": sum(1 for x in t01 if byid[x]["domain"] == d),
                             "T09": sum(1 for x in t09 if byid[x]["domain"] == d)} for d in sorted(dom)},
           "plan": plan}
    json.dump(out, open(os.path.join(RUN, "split.json"), "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("pool", "heldout_hits_dropped", "clean")}), "memory skipped", len(mem_skipped),
          "reserves", len(reserves))
    print(json.dumps(out["by_domain"]))
    for s in plan:
        print(s, plan[s]["n_items"], [len(m) for m in plan[s]["messages"]])


if __name__ == "__main__":
    main()
