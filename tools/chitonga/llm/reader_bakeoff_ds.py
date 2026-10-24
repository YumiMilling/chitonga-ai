"""Reader bake-off REFERENCE arm: DeepSeek V4 Flash through the API, a ceiling outside the pre-registered ranking
(24 Oct 2026, the Jesuit; approved by Chris 24 Oct ~18:5x, relayed by the coordinator: "as a CEILING REFERENCE only").

Rules kept here:
  - only bodies whose test rows contain NO teacher-derived text: rows written by Claude (teacher_en, edge) and their
    noisy copies; the 142 v14 rows (the frozen harness test set, incl. Fr Musunga's real requests) and the noisy copies
    of them are excluded; a body shared with an excluded row is excluded too (unsure = exclude). The onboarding body
    ("Message: Teacher") and the two digest bodies (notes written by the timetable agent) are sent.
  - the SAME captured bodies as the pod arms: system prompt and user message unchanged; DeepSeek gets
    response_format json_object (it cannot take the JSON schema; the schema is described in the system prompt),
    thinking disabled, temperature 0 (the pod arms are greedy). Model "deepseek-flash", the key and endpoint of
    tools/news-corpus/ds_tag.py (as tools/literature uses).
  - 20 bodies first, parsed; then the rest; balance read before, during and after; stop if the spend passes USD 2.
  - replies in the pod format: replies/DS_flash_api.jsonl {key, kind, raw, ms, http, err, timings=usage}; ms includes
    the network.

    python -X utf8 tools/chitonga/llm/reader_bakeoff_ds.py [--first 20] [--workers 6]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.join(REPO, "internal", "chitonga_train", "runs", "2026-10-24_reader_bakeoff")
sys.path.insert(0, os.path.join(REPO, "tools", "news-corpus"))
sys.path.insert(0, os.path.join(REPO, "tools", "literature"))
TAG = "DS_flash_api"
STOP_USD = 2.0


def key():
    import ds_tag  # noqa: PLC0415  (its store.env reads the DeepSeek key)
    return ds_tag.store.env("DEEPSEEK_API_KEY", "")


def balance():
    r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + key()})
    return float(json.load(urllib.request.urlopen(r, timeout=30))["balance_infos"][0]["total_balance"])


def ask(body, max_tokens):
    msgs = body["messages"]
    req = {"model": "deepseek-flash", "messages": msgs, "response_format": {"type": "json_object"},
           "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}}
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            r = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(req).encode("utf-8"),
                                       headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=300) as f:
                d = json.load(f)
            c = d["choices"][0]
            return {"raw": c["message"].get("content") or "", "finish": c.get("finish_reason"), "http": 200,
                    "ms": round((time.time() - t0) * 1000), "err": None, "timings": d.get("usage") or {}}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:200]
        time.sleep(5 * (attempt + 1))
    return {"raw": None, "http": None, "ms": round((time.time() - t0) * 1000), "err": last, "timings": None}


def eligible_bodies():
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RUN, "testset_v2_en_noisy.jsonl"), encoding="utf-8")}

    def teacher_derived(r):
        src = (r.get("source") or "") + " " + (r.get("origin") or "")
        if r["slice"] == "v14_142" or "harness_v14_test" in src or r.get("origin"):
            return True
        return not src.strip().startswith("written by Claude")
    excl = {i for i, r in rows.items() if teacher_derived(r)}
    bodies = [json.loads(l) for l in open(os.path.join(RUN, "bodies.jsonl"), encoding="utf-8")]
    keep, dropped = [], 0
    for b in bodies:
        if b["kind"] == "reader" and (not b["ids"] or any(i in excl for i in b["ids"])):
            dropped += 1
            continue
        keep.append(b)
    return keep, dropped, sorted(excl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    keep, dropped, excl = eligible_bodies()
    print(f"rows excluded as teacher-derived or from the frozen harness set: {len(excl)}; bodies sent {len(keep)}, "
          f"bodies excluded {dropped}")
    out_p = os.path.join(RUN, "replies", TAG + ".jsonl")
    done = {}
    if os.path.exists(out_p):
        for l in open(out_p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("raw") is not None:
                done[r["key"]] = r
    b0 = balance()
    print(f"balance before: {b0}")
    todo = [b for b in keep if b["key"] not in done]
    first, rest = todo[:a.first], todo[a.first:]

    def run(batch):
        res = []
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(ask, b["body"], 900 if b["kind"] == "digest" else 300): b for b in batch}
            for f in cf.as_completed(futs):
                b = futs[f]
                res.append(dict(key=b["key"], kind=b["kind"], **f.result()))
        return res
    sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "harness"))
    from core.reader_model import parse  # noqa: PLC0415
    got = run(first)
    ok = sum(1 for r in got if r["raw"] is not None and parse(r["raw"]) is not None)
    print(f"first {len(got)}: {ok} parse as JSON; errors {[r['err'] for r in got if r['err']][:3]}")
    for r in got:
        done[r["key"]] = r
    if ok < max(1, int(0.8 * len(got))):
        print("STOP: too few first replies parse")
    else:
        for i in range(0, len(rest), 100):
            spent = b0 - balance()
            if spent > STOP_USD:
                print(f"STOP: spend {spent:.2f} > {STOP_USD}")
                break
            for r in run(rest[i:i + 100]):
                done[r["key"]] = r
            print(f"  {min(i + 100, len(rest))}/{len(rest)}; spent so far ~{b0 - balance():.3f} USD", flush=True)
    order = {b["key"]: n for n, b in enumerate(keep)}
    with open(out_p, "w", encoding="utf-8", newline="\n") as f:
        for r in sorted(done.values(), key=lambda r: order.get(r["key"], 1 << 30)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    b1 = balance()
    json.dump({"tag": TAG, "model": "deepseek-flash (DeepSeek V4 Flash, API)", "bodies_sent": len(keep),
               "bodies_excluded": dropped, "rows_excluded": len(excl), "replied": sum(1 for r in done.values() if r.get("raw") is not None),
               "balance_before": b0, "balance_after": b1, "spend_usd": round(b0 - b1, 4),
               "settings": "response_format json_object, thinking disabled, temperature 0, same messages as the pod arms",
               "caveat": "ms includes the network; reference only, outside the pre-registered ranking"},
              open(os.path.join(RUN, "logs", "ds_flash_api.json"), "w", encoding="utf-8"), indent=1)
    print(f"written {out_p}: {len(done)} replies; spend {b0 - b1:.4f} USD")


if __name__ == "__main__":
    main()
