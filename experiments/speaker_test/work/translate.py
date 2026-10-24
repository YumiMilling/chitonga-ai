"""a050 EXACTLY AS SERVED on the speaker-test sentences (24 Oct 2026, the Jesuit).

translator_service.service() with NO cfg: the live defaults (memory on, term guard on, glossary_prefer on, tm_near
strict on, the WiSE a050 drafter), the same call the harness makes (harness3: TS.service().translate). Order: the first
message of each speaker first (Chris: send as soon as the first batch exists), then message by message, then reserves.
Writes served.jsonl line by line (append; resumable) and serve_config.json (CFG, file shas, model sha, memory stamp).
Under the Beelink lock "speaker-g14"; other work on the machine comes first (this is a batch script).
    python -X utf8 translate.py
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(RUN, "..", "..", "..", ".."))
LLM = os.path.join(REPO, "tools", "chitonga", "llm")
sys.path.insert(0, LLM)


def sha(p, n=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import beelink_lock as BL
    import translator_service as TS
    sp = json.load(open(os.path.join(RUN, "split.json"), encoding="utf-8"))
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RUN, "pool.jsonl"), encoding="utf-8")}
    order = []
    m01, m09 = sp["plan"]["T01"]["messages"], sp["plan"]["T09"]["messages"]
    for i in range(max(len(m01), len(m09))):
        for msgs in (m01, m09):
            if i < len(msgs):
                for x in msgs[i]:
                    if x in pool and x not in order:
                        order.append(x)
    ctl_res = []
    for s in ("T01", "T09"):
        for c in sp["plan"][s]["controls"]:
            if c["kind"] == "known_wrong":
                ctl_res.append(c["toi_from"])
    order += [x for x in ctl_res if x not in order]
    order += [x for x in sp["test"] if x not in order]
    order += [x for x in sp["reserves"] if x not in order]
    out = os.path.join(RUN, "served.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    BL.acquire("speaker-g14")
    try:
        svc = TS.service()
        soup_bin = os.path.join(TS.SOUP + "-ct2-int8", "model.bin")
        cfgp = os.path.join(RUN, "serve_config.json")
        if not os.path.exists(cfgp):
            files = {f: sha(os.path.join(LLM, f)) for f in ("translator_service.py", "term_guard.py", "h2_memory.py",
                                                           "h2_text.py", "h3_select.py", "h2_reader.py", "p1_space.py")
                     if os.path.exists(os.path.join(LLM, f))}
            gl = os.path.join(REPO, "internal", "chitonga_train", "llm", "glossary_terms.json")
            if os.path.exists(gl):
                files["glossary_terms.json"] = sha(gl)
            import h2_memory as M
            m = M.memory()
            json.dump({"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in svc.cfg.items()},
                       "soup": TS.SOUP, "soup_model_bin_sha256": sha(soup_bin) if os.path.exists(soup_bin) else None,
                       "available": str(svc.available()), "files_sha256": files,
                       "memory": {"stamp": m.get("stamp"), "pairs": m.get("n"), "lessons": m.get("lessons")},
                       "env": {k: v for k, v in os.environ.items() if k.startswith("CHITONGA_")},
                       "at": time.strftime("%Y-%m-%d %H:%M:%S")},
                      open(cfgp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        for pid in order:
            if pid in done:
                continue
            en = pool[pid]["en"]
            t0 = time.time()
            r = svc.translate(en, "en>toi", keep=True)
            rec = {"id": pid, "en": en, "toi": r.get("text"), "stage": r.get("stage"), "alarms": r.get("alarms"),
                   "switched": r.get("switched"), "memory": r.get("memory"), "tm_near": r.get("tm_near"),
                   "memory_blocked": r.get("memory_blocked"),
                   "pieces": [{k: q.get(k) for k in ("src", "chosen", "why", "stage", "guard", "gloss", "tm_near", "alarms")
                               if k in q} for q in (r.get("pieces") or [])],
                   "secs": round(time.time() - t0, 1)}
            gl = [g for g in svc.guard_log[-5:] if isinstance(g, dict) and g.get("ctx") == en] if svc.guard_log else []
            if gl:
                rec["term_guard"] = gl
            with open(out, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(pid, rec["secs"], rec["stage"], (rec["toi"] or "")[:60], flush=True)
    finally:
        BL.release("speaker-g14")
    print("DONE", len(order), flush=True)


if __name__ == "__main__":
    main()
