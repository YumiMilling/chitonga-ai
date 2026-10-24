"""Chat round v0.14: every reader arm through the ACTUAL harness on the frozen held-out set (24 Oct 2026, the Jesuit).
K0, Beelink, llama.cpp as served. This is the pre-registered kill test's instrument (chat_v14/RUN_PACKAGE.md section 2).

Each held-out request is the first free-text message of a NEW teacher (onboarded: name, language, one class whose
grade is the item's grade: G2*/T2* = 2, G3*/T3* = 3, else 1), through the whole engine: gates (safeguarding, scope,
the 23 Oct health keywords), the deterministic intake, then the arm's reader for what the rules cannot place, the
confirmation, the delivery. Scored by the targets the turn reached (run_testset.reached) against the item's accept list.
Arms:
    rules               RulesReader only (the harness as it runs today without a model)
    gemma               ModelReader(mode=gemma): the UNTOUCHED gemma-4-E4B-it Q4_K_M reads the Tonga itself (baseline 1)
    translator+gemma    the served translator a050 toi>en, then the untouched -it (baseline 2, the 23 Oct winner)
    trained             a chat-round GGUF in the same ModelReader (the kill-test arm)

    python -X utf8 tools/chitonga/llm/v14_eval_harness.py run --arm rules
    python -X utf8 tools/chitonga/llm/v14_eval_harness.py run --arm gemma --gguf <untouched Q4_K_M>
    python -X utf8 tools/chitonga/llm/v14_eval_harness.py run --arm translator+gemma --gguf <untouched Q4_K_M>
    python -X utf8 tools/chitonga/llm/v14_eval_harness.py run --arm trained --gguf <v14 seed GGUF> --label v14_s1
    python -X utf8 tools/chitonga/llm/v14_eval_harness.py report [--pair v14_s1,translator+gemma]
Outputs: internal/chitonga_train/llm/chat_v14/eval/<label>.jsonl and report.md
"""
import argparse
import collections
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
HARNESS = os.path.join(REPO, "tools", "chitonga", "harness")
for p in (HARNESS, HERE, os.path.join(HERE, "..", "assistant")):
    sys.path.insert(0, p)
from core import io as IO  # noqa: E402
from core import reader_model as RM  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.intake import RulesReader  # noqa: E402
from core.log import TurnLog  # noqa: E402
from core.memory import SQLiteMemory  # noqa: E402
from plugins import education  # noqa: E402
from plugins.education.run_testset import reached as _reached  # noqa: E402

TEST = os.path.join(CT, "llm", "chat_v10_tests", "harness_v14_test.jsonl")
OUTD = os.path.join(CT, "llm", "chat_v14", "eval")
LLAMA = r"llama.cpp\vulkan\llama-server.exe"
UNTOUCHED = r"models\base\gemma-4-E4B-it-Q4_K_M.gguf"   # the baselines' model (assistant config.py)
HEALTH_OK = ("outcome:health_no_card", "outcome:no_material", "outcome:unclear", "outcome:safeguarding")
TONGA_SHAPE = re.compile(r"(?i)[a-zŋ]*(zy|bb|kk|cc|ŋ|ng')[a-zŋ]*")


def items():
    return [json.loads(l) for l in open(TEST, encoding="utf-8") if l.strip()]


def serve(gguf, port):
    p = subprocess.Popen([LLAMA, "-m", gguf, "-ngl", "99", "-c", "4096", "--port", str(port), "--host", "127.0.0.1",
                          "-np", "1", "--jinja", "--reasoning", "off", "--reasoning-budget", "0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=os.path.dirname(LLAMA))
    for _ in range(240):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as f:
                if json.load(f).get("status") == "ok":
                    return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    p.kill()
    raise SystemExit("llama-server did not come up")


UNCLEAR_STARTS = ("I am not sure I understood", "Nsiswiilili")   # core/templates_core.json 'unclear' (en) / the served Tonga


def reached(P, rec, reply_text=None):
    """run_testset.reached plus two outcomes it does not tag: the greeting, and (review F3, 24 Oct) the engine's 'unclear'
    line (every no-material / advice / redirect accept list allows it; a reading 'lesson' with no item ends there)"""
    got = _reached(P, rec)
    if "greet" in rec["path"]:
        got.add("outcome:greet")
    text = reply_text if reply_text is not None else " ".join(x.get("body", "") for x in (rec.get("replies") or []))
    if any(text.strip().startswith(u) for u in UNCLEAR_STARTS):
        got.add("outcome:unclear")
    return got


def rescore(a):
    """re-score every stored arm offline from its rows (reached + reply text), no model re-run: review F3"""
    P, _ = education.load()
    for f in sorted(os.listdir(OUTD)):
        if not f.endswith(".jsonl"):
            continue
        rows = [json.loads(l) for l in open(os.path.join(OUTD, f), encoding="utf-8")]
        changed = 0
        for r in rows:
            got = set(r["reached"])
            if any((r.get("reply") or "").strip().startswith(u) for u in UNCLEAR_STARTS):
                got.add("outcome:unclear")
            ok = bool(set(r["accept"]) & got)
            if ok != r["ok"] or got != set(r["reached"]):
                changed += 1
            r["reached"], r["ok"] = sorted(got), ok
        with open(os.path.join(OUTD, f), "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summary(rows, f[:-6])
        json.dump(s, open(os.path.join(OUTD, f[:-6] + ".summary.json"), "w", encoding="utf-8"), indent=1)
        print(f"{f[:-6]}: {changed} rows re-scored -> ok {s['ok']}/{s['n']} (Tonga+mixed {s['tonga_mixed']}, real {s['real']}, g23 {s['g23']})")


def grade_of(it):
    return int(it["id"][1]) if it["id"][:2] in ("G2", "G3", "T2", "T3") else 1


def run(a):
    os.makedirs(OUTD, exist_ok=True)
    P, hooks = education.load()
    # review F6: the trained model must meet the prompt it was trained on; a later edit of SYSTEM would desync them
    man = os.path.join(CT, "llm", "chat_v14", "O", "manifest.json")
    if os.path.exists(man):
        import hashlib  # noqa: PLC0415
        want = json.load(open(man, encoding="utf-8")).get("system_prompt_sha256")
        have = hashlib.sha256(RM.SYSTEM.encode()).hexdigest()
        assert want == have, f"reader_model.SYSTEM changed since the data build ({have[:12]} vs manifest {want[:12]})"
    label = a.label or a.arm
    srv = None
    if a.arm == "rules":
        reader = RulesReader()
    else:
        if not a.gguf and a.arm in ("gemma", "translator+gemma"):
            a.gguf = UNTOUCHED        # the baselines always run the untouched -it; --gguf is for the trained arm
        assert a.gguf and os.path.exists(a.gguf), "a GGUF is needed for this arm"
        url = f"http://127.0.0.1:{a.port}"
        srv = serve(a.gguf, a.port)
        b = None
        if a.arm == "translator+gemma":
            import backends  # noqa: PLC0415
            import config  # noqa: PLC0415
            import h3_select as HS  # noqa: PLC0415
            # 24 Oct: the service's n-best path asks for the s13 SEED models (chitonga-nllb-s13-S13_bi_s1 / _s2), which
            # are no longer on this machine (only the s13 soup, a050 and a070 int8 remain), and the missing folder raises
            # inside translate(), so B2 silently fell back to B1 (0 translator readings in the first run). The seeds and
            # the extra readers are dropped for this arm: a050 alone, term guard on, memory on. Reported in RUN_PACKAGE 7.
            import translator_service as TS  # noqa: PLC0415
            missing = [n for n, p in HS.EXTRA_READERS.items() if not os.path.isdir(p)]
            if missing:   # the service's own CFG, which Service() reads at construction
                TS.CFG["seeds"] = tuple(s for s in TS.CFG.get("seeds", ()) if s not in missing)
                TS.CFG["readers"] = tuple(r for r in TS.CFG.get("readers", ()) if r not in missing)
                print(f"NOTE: translator seeds/readers missing on this machine ({', '.join(missing)}): "
                      f"B2 runs with seeds={TS.CFG['seeds']} readers={TS.CFG['readers']}", flush=True)
            b = backends.Real(config.CFG)
        reader = RM.ModelReader(mode="gemma" if a.arm == "gemma" else a.arm, backends=b, url=url, name="model:" + label)
    out = os.path.join(OUTD, label + ".jsonl")
    rows = []
    t_all = time.time()
    try:
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for i, it in enumerate(items()):
                mem = SQLiteMemory(":memory:")
                e = Engine(P, mem, hooks=hooks, reader=reader, log=TurnLog(None))
                n = f"26099{i:06d}"
                e.handle(n, {"text": "hello"})
                e.handle(n, {"text": "Teacher"})
                e.handle(n, {"choice": "lang:" + ("toi" if it["lang"] != "en" else "en")})
                e.handle(n, {"choice": f"newgrade:{grade_of(it)}"})
                t0 = time.time()
                replies = e.handle(n, {"text": it["text"]})
                dt = time.time() - t0
                rec = e.last
                got = reached(P, rec, IO.as_text(replies[0]) if replies else "")
                confirm = any(p.startswith("confirm:") for p in rec["path"])
                if confirm:
                    r2 = e.handle(n, {"choice": "c:yes"})
                    got |= reached(P, e.last, IO.as_text(r2[0]) if r2 else "")
                ok = bool(set(it["accept"]) & got)
                rd = rec.get("reading") or {}
                diag = getattr(reader, "last", None) if rd.get("source", "").startswith("model") else None
                read_en = str(rd.get("read_en") or "")
                invented = [w for w in re.findall(r"[A-Za-zŋŊ']+", read_en) if TONGA_SHAPE.fullmatch(w)
                            and w.lower() not in it["text"].lower()]
                health_item = any(a_ in it["accept"] for a_ in ("outcome:health_no_card",)) or \
                    bool(re.search(P.rules["health"]["pattern"], it["text"]))
                plain = set(P.rules["health"].get("plain_allowed") or [])
                misrouted = health_item and any(g for g in got if g.startswith(("lesson:", "card:", "item:")) and g not in plain
                                                and g not in it["accept"])
                safe_item = "outcome:safeguarding" in it["accept"]
                safe_missed = safe_item and "outcome:safeguarding" not in got
                rows.append({"id": it["id"], "part": it["part"], "lang": it["lang"], "text": it["text"], "accept": it["accept"],
                             "reached": sorted(got), "ok": ok, "confirm": confirm, "needs_model": "needs_model" in rec["path"],
                             "reading": {k: rd.get(k) for k in ("intent", "unit", "item", "confidence", "source", "why")},
                             "model_used": bool(diag), "candidates": (diag or {}).get("candidates"),
                             "reading_en": (diag or {}).get("reading_en"), "raw": ((diag or {}).get("reply") or "")[:300],
                             "invented_forms": invented, "health_item": health_item, "health_misrouted": bool(misrouted),
                             "safeguarding_item": safe_item, "safeguarding_missed": bool(safe_missed),
                             "path": rec["path"], "flags": rec["flags"], "s": round(dt, 2),
                             "reply": IO.as_text(replies[0])[:200] if replies else ""})
                f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                f.flush()
                print(it["id"], "OK " if ok else "-- ", rd.get("intent"), rd.get("item") or rd.get("unit"), rd.get("source"),
                      f"{dt:.1f}s", flush=True)
    finally:
        if srv:
            srv.kill()
    summ = summary(rows, label)
    summ["s_total"] = round(time.time() - t_all, 1)
    json.dump(summ, open(os.path.join(OUTD, label + ".summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ, indent=1))


def summary(rows, label):
    def n(sel):
        xs = [r for r in rows if sel(r)]
        return f"{sum(r['ok'] for r in xs)}/{len(xs)}"
    return {"arm": label, "n": len(rows), "ok": sum(r["ok"] for r in rows),
            "showcase": n(lambda r: r["part"] == "showcase"), "jesuit23": n(lambda r: r["part"] == "jesuit23"),
            "v13new": n(lambda r: r["part"] == "v13new"), "real": n(lambda r: r["part"] == "real"), "g23": n(lambda r: r["part"] == "g23"),
            "tonga_mixed": n(lambda r: r["lang"] != "en"), "english": n(lambda r: r["lang"] == "en"),
            "model_used": sum(r["model_used"] for r in rows), "needs_model": sum(r["needs_model"] for r in rows),
            "invented_forms": sum(len(r["invented_forms"]) for r in rows),
            "health_misrouted": sum(r["health_misrouted"] for r in rows), "health_items": sum(r["health_item"] for r in rows),
            "safeguarding_missed": sum(r["safeguarding_missed"] for r in rows), "safeguarding_items": sum(r["safeguarding_item"] for r in rows),
            "s_mean": round(sum(r["s"] for r in rows) / max(1, len(rows)), 2),
            "failed_ids": [r["id"] for r in rows if not r["ok"]]}


def load(label):
    p = os.path.join(OUTD, label + ".jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else None


def reachable(P, item, cands):
    """an accepted target is reachable from this candidate list (outcome / intent / route targets always are)"""
    acc = set(item["accept"])
    if any(x.startswith(("outcome:", "intent:", "route:")) for x in acc):
        return True
    return any(c in acc or ("unit:" + (P.unit_of.get(c) or "")) in acc for c in (cands or []))


def tonga_reading_subset(P, rules_rows, b1_rows):
    """review F2, the PRIMARY outcome: the Tonga / mixed items the rules reader gets WRONG, that are asked of the model, and
    whose accepted item is reachable from the NO-translator candidate list. Measured 24 Oct on the frozen 142: 14 items
    (T12, R04, R06, T2A, T2D, T2F, T2G, T2H, T2J, T3A, T3B, T3D, T3E, T3F); B1 5, B2 10 with the translator."""
    return [i for i in rules_rows if rules_rows[i]["lang"] != "en" and not rules_rows[i]["ok"] and b1_rows.get(i, {}).get("model_used")
            and reachable(P, rules_rows[i], b1_rows[i].get("candidates"))
            and any(x.startswith(("unit:", "lesson:", "card:", "route:", "intent:")) for x in rules_rows[i]["accept"])]


def classify_win(i, rules_rows, b2_rows):
    """review F2 (c): what kind of win a trained reader's paired win over B2 is"""
    if rules_rows[i]["ok"] and not b2_rows[i]["ok"]:
        return "avoided B2's override (the rules were already right)"
    if b2_rows[i].get("reading_en"):
        return "read Tonga that B2 needed the translator for"
    return "fixed a rules miss"


def report(a):
    labels = sorted(f[:-6] for f in os.listdir(OUTD) if f.endswith(".jsonl"))
    hdr = ["arm", "ok / 142", "showcase 11", "jesuit23 23", "v13new 50", "real 8", "g23 50", "Tonga+mixed 50", "English 92",
           "model used", "invented", "health misrouted", "safeguarding missed", "s/item"]
    lines = [f"# v0.14 arms through the harness ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for lb in labels:
        rows = load(lb)
        s = summary(rows, lb)
        lines.append("| " + " | ".join(str(x) for x in [lb, s["ok"], s["showcase"], s["jesuit23"], s["v13new"], s["real"], s["g23"],
                                                         s["tonga_mixed"], s["english"], s["model_used"], s["invented_forms"],
                                                         f"{s['health_misrouted']}/{s['health_items']}",
                                                         f"{s['safeguarding_missed']}/{s['safeguarding_items']}", s["s_mean"]]) + " |")
    # the primary outcome (review F2): the Tonga-reading subset, every arm
    rules_rows, b1_rows, b2_rows = (load("rules") or []), (load("gemma") or []), (load("translator+gemma") or [])
    if rules_rows and b1_rows:
        P, _ = education.load()
        RR = {r["id"]: r for r in rules_rows}
        B1 = {r["id"]: r for r in b1_rows}
        sub = tonga_reading_subset(P, RR, B1)
        lines.append(f"\nTonga-reading subset (rules wrong, model asked, target reachable without a translator): {len(sub)} items: {', '.join(sub)}")
        for lb in labels:
            rows = {r["id"]: r for r in load(lb)}
            lines.append(f"  {lb}: {sum(1 for i in sub if rows.get(i, {}).get('ok'))}/{len(sub)}  (ok on: {', '.join(i for i in sub if rows.get(i, {}).get('ok'))})")
    if a.pair:
        x, y = a.pair.split(",")
        rx = {r["id"]: r for r in load(x)}
        ry = {r["id"]: r for r in load(y)}
        for name, sel in (("all", lambda r: True), ("Tonga + mixed", lambda r: r["lang"] != "en"), ("real 8", lambda r: r["part"] == "real"),
                          ("g23", lambda r: r["part"] == "g23")):
            ids = [i for i in rx if i in ry and sel(rx[i])]
            b = sum(1 for i in ids if rx[i]["ok"] and not ry[i]["ok"])
            w = sum(1 for i in ids if ry[i]["ok"] and not rx[i]["ok"])
            lines.append(f"\npaired {x} vs {y} ({name}): better {b} : worse {w}" + (f" = {b / w:.2f}:1" if w else ""))
            lines.append("  better on: " + ", ".join(i for i in ids if rx[i]["ok"] and not ry[i]["ok"]))
            lines.append("  worse on: " + ", ".join(i for i in ids if ry[i]["ok"] and not rx[i]["ok"]))
        if rules_rows and b2_rows and y == "translator+gemma":   # review F2 (c): the kind of each win over B2
            RR = {r["id"]: r for r in rules_rows}
            B2 = {r["id"]: r for r in b2_rows}
            kinds = {}
            for i in rx:
                if i in ry and rx[i]["ok"] and not ry[i]["ok"]:
                    kinds.setdefault(classify_win(i, RR, B2), []).append(i)
            for k, ids in kinds.items():
                lines.append(f"  wins of kind '{k}': {len(ids)} ({', '.join(ids)})")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(OUTD, "report.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--arm", default="rules")
    ap.add_argument("--gguf")
    ap.add_argument("--label")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--pair")
    a = ap.parse_args()
    {"run": run, "report": report, "rescore": rescore}[a.cmd](a)
