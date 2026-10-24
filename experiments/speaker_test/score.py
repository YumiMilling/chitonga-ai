"""Scorer for the speaker test G1-4 (24 Oct 2026, the Jesuit). Executes RULE.md (sha 11e48570...) and ADDENDUM a
(sha 481e50ab...) as sealed; nothing here was tuned on a reply.

Input (all in the run folder unless --replies points elsewhere):
  replies/<SPK>_msgNN.txt   the speaker's reply to that message, VERBATIM (as the desk files answers); several replies
                            to one message go in one file, in order
  replies/received.jsonl    {"spk", "msg": "T01_msg03", "utc": "<first reply time, from zamai_messages>"} per message
  drafts/KEY_lines.json     message line -> item / control (written with the drafts)
  sentences.jsonl, split.json, served.jsonl, ai_judge/ (batch*.jsonl, STAMPS.jsonl, KEY_do_not_open.json)
  ai_judge/fixcompare*.jsonl   optional: the blind second-pass reading of Claude's fix vs the speaker's line
  error_typing/typed*.jsonl    optional: OUR error types per corrected line (RULE 7)
Output: RESULT.md (or --out), and result.json beside it.
Reply forms parsed: "1 ok", "1. ok", "1-4 ok", "1-4 Ok", "all ok", "2 small: <line>", "3 wrong: <line>", "5 <line>"
(untagged = fix_unstated), separated by new lines, ';' or ', <number> '. A returned line equal to ours (case, spaces,
end punctuation) is ok. Numbers are accepted only in rising order and within the message's line count.
    python -X utf8 score.py [--replies DIR] [--out FILE] [--fake]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime

RUN = os.path.dirname(os.path.abspath(__file__))
LAB3 = ("right", "partly", "wrong", "fix_unstated")
FEATS = ("command", "domain_term", "number", "tense_time", "past", "future", "negation", "question", "condition",
         "comparison", "name_place")


def jl(p):
    out = []
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l.startswith("{"):
                out.append(json.loads(l))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.2f} ({lo:.2f}-{hi:.2f})" if n else f"0/0"


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def boot(a, b, f, B=2000, seed=7):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        v = f([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def tnorm(s):
    s = (s or "").lower().replace("’", "'").replace("ng'", "ŋ")
    return " ".join(re.findall(r"[a-zŋ']+", s))


def binary(l):
    return "ok" if l == "right" else "not_ok"


# ------------------------------------------------------------------------------------------------- reply parsing
SPLIT = re.compile(r"(?:^|(?<=[\n;,]))\s*(\d{1,2})(?:\s*(?:-|–|to)\s*(\d{1,2}))?\s*[.):]?\s+", re.I)


def parse_reply(text, ours):
    """ours: {line_no: our Tonga}; returns {line_no: (label, speaker_line)}"""
    n = len(ours)
    t = text.replace("\r", "")
    if re.fullmatch(r"\s*(all\s+(are\s+)?ok(ay)?|ok\s+all|all\s+correct)\W*\s*", t, re.I):
        return {i: ("right", "") for i in ours}
    marks, last = [], 0
    for m in SPLIT.finditer(t):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a > last and a <= b <= n:
            marks.append((m.start(), m.end(), a, b))
            last = b
    out = {}
    for j, (s, e, a, b) in enumerate(marks):
        body = t[e:marks[j + 1][0] if j + 1 < len(marks) else len(t)].strip().strip(",;").strip()
        body = re.split(r"\n\s*pay\b", body, flags=re.I)[0].strip()
        low = body.lower()
        if re.match(r"^(ok|okay|correct|right|yes)\b\W*$", low) or low in ("ok.", "ok,"):
            lab, line = "right", ""
        else:
            m = re.match(r"^(small|wrong)\s*[:\-]?\s*(.*)$", body, re.I | re.S)
            if m:
                lab, line = ("partly" if m.group(1).lower() == "small" else "wrong"), m.group(2).strip()
            else:
                lab, line = "fix_unstated", body
        for i in range(a, b + 1):
            if lab != "right" and line and tnorm(line) == tnorm(ours.get(i)):
                out[i] = ("right", line)
            else:
                out[i] = (lab, line if a == b else ("" if lab == "right" else line))
    return out


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", default=os.path.join(RUN, "replies"))
    ap.add_argument("--out", default=os.path.join(RUN, "RESULT.md"))
    ap.add_argument("--fake", action="store_true", help="the replies are FAKE (scorer test): marked on every line")
    a = ap.parse_args()
    sp = json.load(open(os.path.join(RUN, "split.json"), encoding="utf-8"))
    sent = {r["id"]: r for r in jl(os.path.join(RUN, "sentences.jsonl"))}
    served = {r["id"]: r for r in jl(os.path.join(RUN, "served.jsonl"))}
    key = json.load(open(os.path.join(RUN, "drafts", "KEY_lines.json"), encoding="utf-8"))
    ctl = {c["cid"]: c for s in ("T01", "T09") for c in sp["plan"][s]["controls"]}
    L = []
    W = L.append
    if a.fake:
        W("# FAKE RESULT: scorer test on FAKE replies. NOT A MEASUREMENT. Do not quote.\n")
    else:
        W("# Speaker test G1-4 of a050 as served: RESULT\n")
    W(f"Rule: RULE.md sha256 {open(os.path.join(RUN, 'RULE.md.sha256')).read().split()[0][:16]}...; generated "
      f"{datetime.now().isoformat(timespec='minutes')} by score.py. Speakers by code.\n")

    # ---- judgments
    received = {r["msg"]: r for r in jl(os.path.join(a.replies, "received.jsonl"))}
    J = []          # {spk, msg, line, item | control, label, line_text, utc}
    unparsed = []
    for p in sorted(glob.glob(os.path.join(a.replies, "T0*_msg*.txt"))):
        m = os.path.basename(p)[:-4]
        spk = m[:3]
        if m not in key:
            continue
        ours = {}
        for x in key[m]:
            if x.get("item"):
                ours[x["line"]] = served.get(x["item"], {}).get("toi")
            else:
                ours[x["line"]] = served.get(x["toi_id"], {}).get("toi")
        got = parse_reply(open(p, encoding="utf-8").read(), ours)
        for x in key[m]:
            if x["line"] not in got:
                unparsed.append(f"{m} line {x['line']}")
                continue
            lab, txt = got[x["line"]]
            J.append({"spk": spk, "msg": m, "line": x["line"], "item": x.get("item"), "control": x.get("control"),
                      "label": lab, "speaker_line": txt, "utc": received.get(m, {}).get("utc")})
    items = [j for j in J if j["item"]]
    primary = {}
    for j in items:
        if sp["shared_primary"].get(j["item"], j["spk"]) == j["spk"]:
            primary[j["item"]] = j

    # ---- stamp check (ADDENDUM a, item 3): printed first
    W("## 0. Stamp check (the AI judge wrote before the speakers answered)\n")
    stamps = jl(os.path.join(RUN, "ai_judge", "STAMPS.jsonl"))
    ok_files, bad = 0, []
    first_reply = {}
    for j in items:
        if j["utc"]:
            first_reply[j["item"]] = min(first_reply.get(j["item"], j["utc"]), j["utc"])
    late = []
    stamp_of = {}
    for s in stamps:
        p = os.path.join(RUN, s["file"])
        h = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
        if h == s["sha256"]:
            ok_files += 1
        else:
            bad.append(s["file"])
        for x in s["items"]:
            stamp_of.setdefault(x, s)
            if x in first_reply and not (s["utc"] < _z(first_reply[x])):
                late.append(x)
    no_time = sum(1 for j in items if not j["utc"])
    W(f"- Judge files whose sha256 matches STAMPS.jsonl: {ok_files} of {len(stamps)}" + (f"; MISMATCH: {bad}" if bad else "") + ".")
    W(f"- Items whose judge stamp is NOT earlier than the first speaker reply: {len(late)}" + (f" {sorted(set(late))} (left out of every AI comparison)" if late else "") + ".")
    W(f"- Judgments without a reply time (received.jsonl): {no_time}" + (" (time order NOT VERIFIED for those)" if no_time else "") + ".\n")

    # ---- controls read FIRST (RULE 5: a speaker failing both known-wrong lines leaves the pooled primary estimate)
    pre_excluded = set()
    for spk in ("T01", "T09"):
        kw = [j for j in J if j["spk"] == spk and j["control"] and ctl[j["control"]]["kind"] == "known_wrong"]
        if len(kw) == 2 and all(j["label"] == "right" for j in kw):
            pre_excluded.add(spk)
    primary_all = dict(primary)
    if pre_excluded:
        primary = {k: v for k, v in primary.items() if v["spk"] not in pre_excluded}

    # ---- primary estimates
    W("## 1. The speakers' verdict on a050 (RULE 3)\n")
    if pre_excluded:
        W(f"**{sorted(pre_excluded)} failed both known-wrong controls: left out of the primary estimate below (RULE 5); "
          f"with them: right {fmt(sum(1 for v in primary_all.values() if v['label'] == 'right'), len(primary_all))}.**\n")
    W(f"Judgments parsed: {len(J)} ({len(items)} on items, {len(J) - len(items)} on controls); lines not parsed: {len(unparsed)}"
      + (f" ({', '.join(unparsed[:12])})" if unparsed else "") + ".\n")
    pc = Counter(j["label"] for j in primary.values())
    n = len(primary)
    W(f"**Primary, one judgment per item (n = {n} of 170):**")
    for l in LAB3:
        W(f"- {l}: {fmt(pc[l], n)}")
    p, lo, hi = wilson(pc["right"], n)
    if n:
        verdict = ("AT the bar" if p >= 0.80 else "NEAR the bar" if (p >= 0.70 and hi >= 0.80) else
                   "BELOW the bar" if hi < 0.80 else "between: point under 0.70 but the interval reaches 0.80 (not 'near' by the rule)")
        W(f"- right or partly (usable after a small fix, NOT the bar): {fmt(pc['right'] + pc['partly'], n)}")
        W(f"\n**Against the 80% bar (RULE 3): {verdict}** (right {p:.2f}, Wilson 95% {lo:.2f}-{hi:.2f}).\n")
    for spk in ("T01", "T09"):
        c = Counter(j["label"] for j in items if j["spk"] == spk)
        m = sum(c.values())
        W(f"- {spk}, all own judgments (n = {m}): " + "; ".join(f"{l} {fmt(c[l], m)}" for l in LAB3))
    c = Counter(j["label"] for j in items)
    W(f"- Secondary, all judgments pooled (n = {len(items)}): " + "; ".join(f"{l} {fmt(c[l], len(items))}" for l in LAB3) + "\n")

    # ---- agreement
    W("## 2. Agreement on the 30 shared (RULE 4)\n")
    by = defaultdict(dict)
    for j in items:
        if j["item"] in sp["shared"]:
            by[j["item"]][j["spk"]] = j
    both = [x for x in sp["shared"] if len(by[x]) == 2]
    A = [by[x]["T01"]["label"] for x in both]
    B = [by[x]["T09"]["label"] for x in both]
    if both:
        raw3 = sum(x == y for x, y in zip(A, B)) / len(both)
        rawb = sum(binary(x) == binary(y) for x, y in zip(A, B)) / len(both)
        k3, kb = kappa(A, B), kappa([binary(x) for x in A], [binary(x) for x in B])
        c3 = boot(A, B, kappa)
        cb = boot([binary(x) for x in A], [binary(x) for x in B], kappa)
        W(f"Both answered: {len(both)} of 30. Three-way raw {raw3:.2f}, kappa {k3:.2f} (95% {c3[0]:.2f}-{c3[1]:.2f}); "
          f"binary ok / not ok raw {rawb:.2f}, kappa {kb:.2f} (95% {cb[0]:.2f}-{cb[1]:.2f}).\n")
        W("| item | English | a050 | T01 | T09 |\n|---|---|---|---|---|")
        for x, y, it in zip(A, B, both):
            if x != y:
                W(f"| {it} | {sent[it]['en']} | {served[it]['toi']} | {x}: {by[it]['T01']['speaker_line']} | {y}: {by[it]['T09']['speaker_line']} |")
        W("")
    else:
        W("No shared item answered by both yet.\n")

    # ---- controls
    W("## 3. Controls (RULE 5; in no estimate)\n")
    excluded = set()
    for spk in ("T01", "T09"):
        cj = [j for j in J if j["spk"] == spk and j["control"]]
        kw = [j for j in cj if ctl[j["control"]]["kind"] == "known_wrong"]
        kw_pass = sum(1 for j in kw if j["label"] != "right")
        reps = []
        for j in cj:
            c = ctl[j["control"]]
            if c["kind"] == "repeat":
                orig = [i for i in items if i["spk"] == spk and i["item"] == c["of"]]
                if orig:
                    reps.append(binary(orig[0]["label"]) == binary(j["label"]))
        flag = ""
        if len(kw) == 2 and kw_pass == 0:
            excluded.add(spk)
            flag = " **Both known-wrong marked ok: this speaker is reported apart and left out of the pooled estimate (Chris is told).**"
        elif kw and kw_pass < len(kw):
            flag = " Flag: one known-wrong line marked ok (kept)."
        if len(reps) == 2 and not any(reps):
            flag += " Flag: both repeats inconsistent (kept)."
        W(f"- {spk}: known-wrong passed {kw_pass} of {len(kw)}; repeats consistent {sum(reps)} of {len(reps)}.{flag}")
    assert excluded == pre_excluded
    W("")

    # ---- AI judge
    W("## 4. The AI judge against the speakers (RULE 6)\n")
    akey = json.load(open(os.path.join(RUN, "ai_judge", "KEY_do_not_open.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(RUN, "ai_judge", "KEY_do_not_open.json")) else {}
    ai = {}
    for p in sorted(glob.glob(os.path.join(RUN, "ai_judge", "batch*.jsonl"))):
        for r in jl(p):
            if "item" in r and r["item"] in akey:
                ai.setdefault(akey[r["item"]], r)          # the first stamped judgment counts
    lateset = set(late)
    pairs = [(it, ai[it], primary[it]) for it in primary if it in ai and it not in lateset]
    W(f"AI-judged items: {sum(1 for x in ai if x in sent)} (+ controls {sum(1 for x in ai if x in ctl)}); "
      f"compared with a primary speaker judgment: {len(pairs)}.\n")
    if pairs:
        sa = [binary(x["label"]) for _, x, _ in pairs]
        ss = [binary(y["label"]) for _, _, y in pairs]
        a3 = [x["label"] for _, x, _ in pairs]
        s3 = [y["label"] if y["label"] != "fix_unstated" else "fix_unstated" for _, _, y in pairs]
        W(f"- AI judge alone on these items: " + "; ".join(f"{l} {fmt(sum(1 for v in a3 if v == l), len(a3))}" for l in ("right", "partly", "wrong")))
        W(f"- Binary agreement {sum(x == y for x, y in zip(sa, ss)) / len(pairs):.2f}, kappa {kappa(sa, ss):.2f}; "
          f"three-way agreement (fix_unstated never matches) {sum(x == y for x, y in zip(a3, s3)) / len(pairs):.2f}, kappa {kappa(a3, s3):.2f}")
        conf = Counter((x["label"], y["label"]) for _, x, y in pairs)
        W("\n| AI \\ speaker | " + " | ".join(LAB3) + " |\n|---|" + "---|" * len(LAB3))
        for l in ("right", "partly", "wrong"):
            W(f"| {l} | " + " | ".join(str(conf[(l, m)]) for m in LAB3) + " |")
        W("\n**Calibration by `sure`:**\n")
        for s in ("high", "medium", "low"):
            g = [(x, y) for _, x, y in pairs if x.get("sure") == s]
            W(f"- {s}: n {len(g)}; binary agreement {fmt(sum(binary(x['label']) == binary(y['label']) for x, y in g), len(g))}")
        W("\n**Reliability by p_right band (share the speaker marked ok):**\n")
        for name, lo_, hi_ in (("p >= 80", 80, 101), ("60-79", 60, 80), ("40-59", 40, 60), ("< 40", -1, 40)):
            g = [y for _, x, y in pairs if lo_ <= float(x.get("p_right", -1) or -1) < hi_]
            W(f"- {name}: n {len(g)}; speaker ok {fmt(sum(1 for y in g if y['label'] == 'right'), len(g))}")
        routed = [y for _, x, y in pairs if x["label"] == "right" and x.get("sure") == "high"]
        W(f"\n**Routing test:** lines the judge calls right with sure high: {len(routed)} of {len(pairs)} "
          f"({len(routed) / len(pairs):.2f}) would be served unseen; per 100 served unseen, not ok by the speaker: "
          f"{100 * sum(1 for y in routed if y['label'] != 'right') / max(1, len(routed)):.0f} "
          f"(wrong: {100 * sum(1 for y in routed if y['label'] == 'wrong') / max(1, len(routed)):.0f}); "
          f"share still needing a speaker {1 - len(routed) / len(pairs):.2f}.")
        conf_err = [(it, x, y) for it, x, y in pairs if x["label"] == "right" and y["label"] != "right"]
        W(f"\n**Confident errors (judge right, speaker corrected): {len(conf_err)}**\n")
        for it, x, y in conf_err:
            W(f"- {it} [{x.get('sure')}, p {x.get('p_right')}] {sent[it]['en']} = {served[it]['toi']} -> {y['label']}: {y['speaker_line']}")
        W("\n**Claude's own fixes (evaluation only)**\n")
        need_ai = [binary(x["label"]) == "not_ok" for _, x, _ in pairs]
        need_sp = [binary(y["label"]) == "not_ok" for _, _, y in pairs]
        tp = sum(1 for p_, q in zip(need_ai, need_sp) if p_ and q)
        W(f"- needs a fix: precision {fmt(tp, sum(need_ai))}; recall {fmt(tp, sum(need_sp))}")
        fc = {}
        for p in glob.glob(os.path.join(RUN, "ai_judge", "fixcompare*.jsonl")):
            for r in jl(p):
                fc[r.get("item")] = r.get("verdict")
        both_fix = [(it, x, y) for it, x, y in pairs if x["label"] != "right" and y["label"] != "right" and x.get("fix") and y["speaker_line"]]
        auto = Counter()
        for it, x, y in both_fix:
            A_, B_ = set(tnorm(x["fix"]).split()), set(tnorm(y["speaker_line"]).split())
            j = len(A_ & B_) / max(1, len(A_ | B_))
            auto["same" if tnorm(x["fix"]) == tnorm(y["speaker_line"]) else "close" if j >= 0.5 else "different"] += 1
        W(f"- both corrected: {len(both_fix)}; [A] automatic token comparison: same {auto['same']}, close {auto['close']}, different {auto['different']}")
        W(f"- blind second-pass reading: " + (", ".join(f"{k} {v}" for k, v in Counter(fc.get(it) for it, _, _ in both_fix if fc.get(it)).items()) or "pending (not run yet)"))
        W(f"- judge wrong, speaker ok: {sum(1 for _, x, y in pairs if x['label'] == 'wrong' and y['label'] == 'right')}; "
          f"judge right, speaker wrong: {sum(1 for _, x, y in pairs if x['label'] == 'right' and y['label'] == 'wrong')}")
        kwj = [ai[c] for c in ai if c in ctl]
        W(f"- known-wrong controls judged by the AI: {len(kwj)}; called right: {sum(1 for r in kwj if r['label'] == 'right')}")
    W("")

    # ---- error map
    W("## 5. Where and how a050 errs (RULE 7)\n")
    W("Share NOT right on the primary judgments, Wilson 95%; cells under 10 items marked too few.\n")
    W("| feature | n | not right |\n|---|---|---|")
    for f in FEATS:
        g = [j for it, j in primary.items() if f in sent[it]["features"]]
        k = sum(1 for j in g if j["label"] != "right")
        W(f"| {f} | {len(g)} | {fmt(k, len(g))}{' (too few)' if len(g) < 10 else ''} |")
    W("\n| domain | n | not right |\n|---|---|---|")
    for d in sorted({s["domain"] for s in sent.values()}):
        g = [j for it, j in primary.items() if sent[it]["domain"] == d]
        k = sum(1 for j in g if j["label"] != "right")
        W(f"| {d} | {len(g)} | {fmt(k, len(g))}{' (too few)' if len(g) < 10 else ''} |")
    typed = {}
    for p in glob.glob(os.path.join(RUN, "error_typing", "typed*.jsonl")):
        for r in jl(p):
            typed[r.get("id")] = r.get("errors", [])
    corr = [it for it, j in primary.items() if j["label"] != "right"]
    W(f"\n**Error types (OURS, not the speakers': assigned by comparing a050's line with the speaker's line):** "
      + (", ".join(f"{k} {v}" for k, v in Counter(e for it in corr for e in typed.get(it, [])).most_common())
         if typed else f"pending (error_typing not run; {len(corr)} corrected lines waiting)"))
    flag_g = defaultdict(list)
    for it, j in primary.items():
        al = served.get(it, {}).get("alarms") or []
        flag_g["service alarm" if al else "no alarm"].append(j["label"] == "right")
    W("\nService flags (information only): " + "; ".join(f"{k}: right {fmt(sum(v), len(v))}" for k, v in flag_g.items()))
    W("\n## 6. Every corrected line (primary)\n")
    for it, j in sorted(primary.items()):
        if j["label"] != "right":
            W(f"- {it} [{sent[it]['domain']}] {sent[it]['en']} = {served[it]['toi']} -> {j['spk']} {j['label']}: {j['speaker_line']}"
              + ("  (FAKE)" if a.fake else ""))
    open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    json.dump({"primary": {k: v["label"] for k, v in primary.items()}, "n": n, "counts": dict(pc), "fake": a.fake},
              open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), indent=1)
    print("wrote", a.out, "primary n", n, dict(pc))


def _z(t):
    """reply time as an ISO UTC string comparable with the stamps (both end in +00:00)"""
    t = (t or "").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(t).astimezone().astimezone(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
    except ValueError:
        return t


if __name__ == "__main__":
    main()
