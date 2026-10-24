"""The toolbox on every output before a teacher sees it (15 Oct 2026). Chris: "Is there no test we can run before it
goes to the teachers? ... I want those lines to be fully analysed, and I want to know the certainty the model has for
each ... mix up the high certainty and low certainty ones to see if it is confidently wrong at times" / "use our
established toolbox". chrF is not a test for Tonga (Chris, same day): nothing here compares letters with a reference.

For each held-out English line and each model's Tonga (runs/<run>/heldout_<model>.jsonl, from score_heldout.py):
  certainty   the model's own mean log-probability of its Tonga given the English (CTranslate2 forced scoring), and
              its weakest word (lowest-probability word): where the model itself hesitated
  harness     harness.checks: attestation (is every word real Tonga), form (engine + judge), rules (F045 etc.), the
              meaning map family for flagged words, and the harness triage (send / teacher / block)
  meaning     dynamic equivalence proxy: the Tonga back-translated by the OTHER model (no model grades itself), its
              thought similarity to the English (bge-small; 8% of human translators' lines fall below 0.70)
  gloss       word by word: dictionary meaning, or the engine's reading on a known stem (stem, person, tense)
Then the panel proposal: lines where the models differ, spread over certainty (high / middle / low thirds), with the
CONFIDENT-but-doubtful ones in (high certainty and a harness or meaning flag): the candidates for 'confidently wrong'.

    python -X utf8 tools/chitonga/llm/panel_analyse.py <run folder> <model A dir> <model B dir> [n=20]

    python -X utf8 tools/chitonga/llm/panel_analyse.py refresh <run folder> <model A name> <model B name> [n=20]
        recompute the cheap checks, glosses and triage on the saved analysis (after the dictionary or the attestation
        cache changed), with the NEUTRAL meaning check where present (one back-translator for both models,
        panel/neutral_meaning.py), and write the proposal again. No model is loaded.

Writes <run>/panel/analysis.jsonl and <run>/panel/proposal.md.
"""
import csv
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "tools", "chitonga", "grammar2"), os.path.join(REPO, "tools", "chitonga", "train")]
import harness as H  # noqa: E402
from bt_score import ct2_logprob  # noqa: E402

H.PERSONA = None   # the chat model's self-check needs llama-server; this measures the translators
WORD = re.compile(r"[A-Za-zŋŊ']+")


def load_lex():
    lex, verbs = {}, {}
    for r in csv.DictReader(open(os.path.join(REPO, "tools", "chitonga", "grammar2", "lexicon.tsv"), encoding="utf-8"),
                            delimiter="\t"):
        w, m = (r.get("word") or "").strip().lower(), (r.get("meaning") or "").strip()
        if w and m:
            lex.setdefault(w, m[:50])
            for pl in re.split(r"[,/ ]+", (r.get("plural") or "").strip().lower()):
                if pl:
                    lex.setdefault(pl, "(pl.) " + m[:44])
            if r.get("pos") == "v":
                verbs.setdefault(w, m[:40])
    return lex, verbs


def gloss(w, lex, verbs):
    import engine2 as E
    lw = w.lower().strip("'")
    if lw in lex:
        return lex[lw]
    ps = E.read(lw, known=set(verbs))
    if ps:
        p = ps[0]
        return f"-{p[0]} '{verbs[p[0]]}' ({' '.join(p[1:])})"
    return "?"


class Model:
    def __init__(self, d):
        import ctranslate2
        from transformers import AutoTokenizer
        self.name = os.path.basename(d.rstrip("/\\")).replace("chitonga-nllb-", "")
        self.tok = AutoTokenizer.from_pretrained(d)
        self.tr = ctranslate2.Translator(d + "-ct2-int8", device="cpu", compute_type="int8",
                                         intra_threads=max(1, (os.cpu_count() or 4) - 2))

    def certainty(self, en, toi):
        """mean log-prob per token and the weakest word (lowest summed log-prob over its pieces)"""
        tok = self.tok
        tok.src_lang = "eng_Latn"
        s = tok.convert_ids_to_tokens(tok.encode(en))
        tok.src_lang = "toi_Latn"
        t = tok.convert_ids_to_tokens(tok.encode(toi))
        r = self.tr.score_batch([s], [t])[0]
        lp, toks = list(r.log_probs), t[-len(r.log_probs):]
        words, cur = [], None
        for piece, p in zip(toks, lp):
            if piece in ("toi_Latn", "</s>"):
                continue
            if piece.startswith("▁") or cur is None:
                cur = [piece.lstrip("▁"), 0.0]
                words.append(cur)
            else:
                cur[0] += piece
            cur[1] += p
        weak = min(words, key=lambda x: x[1]) if words else ["", 0.0]
        return round(sum(lp) / max(1, len(lp)), 3), weak[0], round(weak[1], 2)

    def back(self, toi):
        tok = self.tok
        tok.src_lang = "toi_Latn"
        s = tok.convert_ids_to_tokens(tok.encode(toi))
        r = self.tr.translate_batch([s], target_prefix=[["eng_Latn"]], beam_size=4, max_decoding_length=96)[0]
        return tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True)


def main():
    run, da, db = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    A, B = Model(da), Model(db)
    other = {A.name: B, B.name: A}
    lex, verbs = load_lex()
    rows = {}
    for M in (A, B):
        for i, l in enumerate(open(os.path.join(run, f"heldout_{M.name}.jsonl"), encoding="utf-8")):
            r = json.loads(l)
            rows.setdefault(i, {"i": i, "set": r["set"], "en": r["en"], "refs": r["refs"], "out": {}})["out"][M.name] = r["hyp"]
    out = []
    for i, r in sorted(rows.items()):
        for M in (A, B):
            toi = r["out"][M.name]
            mean, weak, weak_lp = M.certainty(r["en"], toi)
            c = H.checks(toi, r["en"], full=False)
            back = other[M.name].back(toi)
            c["meaning"] = {"back": back, "back_by": other[M.name].name, "thought": H.thought(back, r["en"]),
                            "short": len(r["en"].split()) < 4}
            verdict, why = H.triage(c, "translate")
            r.setdefault("an", {})[M.name] = {
                "toi": toi, "certainty": mean, "weakest": [weak, weak_lp], "verdict": verdict, "why": why,
                "attest": c["attest"], "form": c["form"], "rules": c["rules"], "guess": c.get("guess"),
                "meaning": c["meaning"], "gloss": [(w, gloss(w, lex, verbs)) for w in WORD.findall(toi)]}
        r["same"] = r["out"][A.name].strip() == r["out"][B.name].strip()
        out.append(r)
        print(f"{i:3d} {'=' if r['same'] else '≠'} " + " | ".join(
            f"{m}: {r['an'][m]['certainty']:6.2f} {r['an'][m]['verdict']:7s} th {r['an'][m]['meaning']['thought']:.2f}"
            for m in (A.name, B.name)) + f" | {r['en'][:50]}", flush=True)
    os.makedirs(os.path.join(run, "panel"), exist_ok=True)
    with open(os.path.join(run, "panel", "analysis.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    propose(run, out, A.name, B.name, n)


def llm_bad(x):
    """the DeepSeek checks (panel_llm_meaning.py), where they ran: fidelity major, back-translation different"""
    L = x.get("llm") or {}
    return (L.get("fidelity", {}).get("verdict") == "major_issues", L.get("back", {}).get("same") == "different")


def value(r, a, b):
    """what one teacher answer on this line can teach us (Chris: "squeeze as much progress as possible"): the checks
    split the two models (decides the comparison), a confident line some check calls wrong (confidently wrong?),
    the harness would send a line DeepSeek calls wrong (a hole in the harness), the two DeepSeek checks disagree
    (which check to trust)"""
    v = 0
    fa, ba = llm_bad(r["an"][a])
    fb, bb = llm_bad(r["an"][b])
    v += 2 * ((fa, ba) != (fb, bb))
    for m, (f, k) in ((a, (fa, ba)), (b, (fb, bb))):
        x = r["an"][m]
        v += 2 * (f and x["certainty"] >= r["cut_hi"])
        v += 2 * (f and x["verdict"] == "send")
        v += 1 * (f != k)
    return v


def propose(run, rows, a, b, n):
    """lines where the models differ, spread over certainty thirds; in each third the lines one answer teaches most"""
    diff = [r for r in rows if not r["same"]]
    cs = sorted(r["an"][m]["certainty"] for r in rows for m in (a, b))
    for r in diff:
        r["cert"] = min(r["an"][a]["certainty"], r["an"][b]["certainty"])
        r["cut_hi"] = cs[len(cs) * 2 // 3]
        r["value"] = value(r, a, b)
    diff.sort(key=lambda r: -r["cert"])
    k = len(diff) // 3 or 1
    thirds = [("high", diff[:k]), ("middle", diff[k:2 * k]), ("low", diff[2 * k:])]
    pick, rnd = [], random.Random(15)
    for name, grp in thirds:
        want = n // 3 + (1 if name == "high" and n % 3 else 0) + (1 if name == "middle" and n % 3 == 2 else 0)
        grp = sorted(grp, key=lambda r: (-r["value"], rnd.random()))
        for r in grp[:want]:
            r["third"] = name
            pick.append(r)
    lines = [f"# Panel proposal: {a} vs {b} ({len(rows)} held-out lines, {len(diff)} where the models differ)\n",
             "Certainty = the model's own mean log-probability of its Tonga (closer to 0 = surer). Verdict = harness "
             "triage. Thought = meaning of the back-translation vs the English, 0-1: the dynamic equivalence proxy "
             "(who back-translated is named on each line).\n"]
    for r in pick:
        lines.append(f"\n## {r['i']} [{r['third']} certainty, value {r['value']}] {r['en']}\n")
        for m in (a, b):
            x = r["an"][m]
            lines.append(f"- **{m}**: {x['toi']}  \n  certainty {x['certainty']}, weakest word '{x['weakest'][0]}' "
                         f"({x['weakest'][1]}); harness: {x['verdict']}{' (' + '; '.join(x['why']) + ')' if x['why'] else ''}; "
                         f"back ({x['meaning'].get('back_by')}): \"{x['meaning']['back']}\" (thought {x['meaning']['thought']})  \n  gloss: "
                         + " · ".join(f"{w} = {g}" for w, g in x["gloss"])
                         + (f"  \n  DeepSeek: fidelity {x['llm']['fidelity'].get('verdict')}"
                            + "".join(f"; {i['type']}/{i['severity']}: {i['english']} <- {i['tonga']}"
                                      for i in x['llm']['fidelity'].get('issues', []))
                            + f" | back-translation {x['llm']['back'].get('same')} {x['llm']['back'].get('differs', '')}"
                            if x.get("llm") else ""))
    open(os.path.join(run, "panel", "proposal.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"\nproposal: {len(pick)} lines -> {os.path.join(run, 'panel', 'proposal.md')}")


def refresh():
    run, a, b = sys.argv[2], sys.argv[3], sys.argv[4]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    P = os.path.join(run, "panel", "analysis.jsonl")
    rows = [json.loads(l) for l in open(P, encoding="utf-8")]
    lex, verbs = load_lex()
    H.back_translate = lambda t: ""          # meaning comes from the saved checks, nothing is re-translated
    H.thought = lambda x, y: 1.0
    for r in rows:
        for m in (a, b):
            x = r["an"][m]
            c = H.checks(x["toi"], r["en"], full=False)
            mm = x.get("meaning_neutral") or x["meaning"]
            c["meaning"] = {"back": mm["back"], "back_by": mm.get("back_by"), "thought": mm["thought"],
                            "short": len(r["en"].split()) < 4}
            x["verdict"], x["why"] = H.triage(c, "translate")
            x.update({"attest": c["attest"], "form": c["form"], "rules": c["rules"], "guess": c.get("guess"),
                      "meaning": c["meaning"], "gloss": [(w, gloss(w, lex, verbs)) for w in WORD.findall(x["toi"])]})
    with open(P, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    propose(run, rows, a, b, n)


if __name__ == "__main__":
    refresh() if sys.argv[1:2] == ["refresh"] else main()
