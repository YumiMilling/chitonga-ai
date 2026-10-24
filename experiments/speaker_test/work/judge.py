"""The blind AI judge for the speaker test, run per batch BEFORE that batch goes to a speaker (24 Oct 2026, the Jesuit;
RULE.md section 6; Chris: "add a time stamp to your evaluation and alternative translation, so we can see you didn't see
the teacher input").

Per call: the items (served Tonga of the given ids) get opaque ids; the key goes to ai_judge/KEY_do_not_open.json
(opened only by score.py). One empty folder outside the repo per call (items, engine readings of the candidates' words,
the full lexicon with speaker names redacted, prompt.md); claude -p (Opus 5.5) with Bash / web / sub-agents off, as
run_claude.py of the weakness map. The answer is written ONCE to ai_judge/<name>.jsonl (first line: the header with the
evaluation-only rule), then its sha256 and UTC time go to ai_judge/STAMPS.jsonl, a hash note to the phone
(tools/notify.py: a third-party time record) and, if installed, an OpenTimestamps proof. A stamped file is never edited.
    python -X utf8 judge.py <name> <id> [<id> ...]        ids: pool ids, or control ids T01-kw1 ...
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(RUN, "..", "..", "..", ".."))
AJ = os.path.join(RUN, "ai_judge")
DIAG = os.path.join(REPO, "internal", "chitonga_train", "llm", "panel", "diagnosis_500_2026-10-21")
sys.path.insert(0, DIAG)
MODEL = "claude-opus-5-5"
SCRATCH = os.path.join(os.environ.get("TEMP", "/tmp"), "claude_judge_g14")
HEADER = {"_header": "AI judge labels and Claude's own Tonga lines (fix) for the speaker test G1-4. EVALUATION ONLY: "
                     "never shown to a speaker, never served, never training data, never in the translation memory "
                     "(RULE.md section 6). Written before the speakers saw these items; sha256 + UTC in STAMPS.jsonl."}
ASK = ("Read prompt.md in this folder and do exactly what it says. items.jsonl holds {n} items; judge every one. Write "
       "the answer to judged.jsonl in this folder (JSON Lines, one line per item, in the order of items.jsonl). When the "
       "file is complete, reply with the single line: DONE <number of lines written>.")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    name, ids = sys.argv[1], list(dict.fromkeys(sys.argv[2:]))
    # an item judged in an earlier stamped file is not judged again (a shared item reaches the second speaker later)
    stp = os.path.join(AJ, "STAMPS.jsonl")
    before = {x for l in open(stp, encoding="utf-8") if l.strip() for x in json.loads(l)["items"]} if os.path.exists(stp) else set()
    skipped = [x for x in ids if x in before]
    ids = [x for x in ids if x not in before]
    if skipped:
        print("already judged and stamped:", skipped)
    out = os.path.join(AJ, f"{name}.jsonl")
    assert not os.path.exists(out), f"{out} exists: stamped files are never rewritten"
    import make_pack as MP
    sp = json.load(open(os.path.join(RUN, "split.json"), encoding="utf-8"))
    ctl = {c["cid"]: c for s in ("T01", "T09") for c in sp["plan"][s]["controls"]}
    sv = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RUN, "served.jsonl"), encoding="utf-8") if l.strip()}
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RUN, "pool.jsonl"), encoding="utf-8")}
    kp = os.path.join(AJ, "KEY_do_not_open.json")
    key = json.load(open(kp, encoding="utf-8")) if os.path.exists(kp) else {}
    items = []
    for x in ids:
        if x in ctl:
            c = ctl[x]
            if c["kind"] != "known_wrong":
                continue                       # a repeat is the same line as its original: judged there
            en, toi = pool[c["en_from"]]["en"], sv[c["toi_from"]]["toi"]
        else:
            en, toi = pool[x]["en"], sv[x]["toi"]
        jid = "j" + hashlib.sha256(f"g14|{x}".encode()).hexdigest()[:8]
        key[jid] = x
        items.append({"item": jid, "en": en, "candidate": toi})
    items.sort(key=lambda it: it["item"])      # order by the opaque id, not by batch position
    json.dump(key, open(kp, "w", encoding="utf-8"), indent=0)
    wd = os.path.join(SCRATCH, f"{name}_{int(time.time())}")
    os.makedirs(wd)
    with open(os.path.join(wd, "items.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    words = {w.lower() for it in items for w in MP.WORD.findall(it["candidate"])}
    open(os.path.join(wd, "engine_readings.txt"), "w", encoding="utf-8", newline="\n").write(MP.readings(words))
    open(os.path.join(wd, "lexicon.tsv"), "w", encoding="utf-8", newline="\n").write(
        re.sub(MP.NAMES, "SPEAKER", open(MP.LEXICON, encoding="utf-8").read()))
    shutil.copy2(os.path.join(AJ, "prompt.md"), os.path.join(wd, "prompt.md"))
    t0 = time.time()
    r = subprocess.run(["claude", "-p", ASK.format(n=len(items)), "--model", MODEL, "--permission-mode", "acceptEdits",
                        "--disallowedTools", "Bash,WebFetch,WebSearch,Task", "--output-format", "json"],
                       cwd=wd, capture_output=True, text=True, encoding="utf-8", timeout=5400)
    got = {}
    jp = os.path.join(wd, "judged.jsonl")
    if os.path.exists(jp):
        for l in open(jp, encoding="utf-8"):
            l = l.strip().strip("`")
            if l.startswith("{"):
                try:
                    d = json.loads(l)
                    got[d["item"]] = d
                except (ValueError, KeyError):
                    pass
    missing = [it["item"] for it in items if it["item"] not in got or got[it["item"]].get("label") not in ("right", "partly", "wrong")]
    meta = {"rc": r.returncode, "secs": round(time.time() - t0), "n": len(items), "complete": len(items) - len(missing),
            "model": MODEL}
    try:
        d = json.loads(r.stdout)
        meta.update({k: d.get(k) for k in ("is_error", "num_turns", "total_cost_usd")})
    except ValueError:
        meta["stdout_tail"] = (r.stdout + r.stderr)[-500:]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(HEADER, name=name, meta=meta, utc=datetime.now(timezone.utc).isoformat(timespec="seconds")),
                           ensure_ascii=False) + "\n")
        for it in items:
            if it["item"] in got:
                f.write(json.dumps(dict(got[it["item"]], judge=MODEL + " (claude -p)"), ensure_ascii=False) + "\n")
    h = sha(out)
    utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = {"file": f"ai_judge/{name}.jsonl", "sha256": h, "utc": utc, "items": sorted(ids), "n": len(items),
             "complete": meta["complete"], "missing": missing}
    with open(os.path.join(AJ, "STAMPS.jsonl"), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(stamp, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, os.path.join(REPO, "tools", "notify.py"),
                    f"speaker test AI judge {name} sha256 {h[:16]}... {utc}"], capture_output=True, timeout=60)
    if shutil.which("ots"):
        subprocess.run(["ots", "stamp", out], capture_output=True, timeout=120)
    print(json.dumps(stamp))


if __name__ == "__main__":
    main()
