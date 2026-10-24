"""Two DeepSeek meaning checks on the panel analysis (15 Oct 2026; Chris: "Ok, run it, we see if it adds value").
The embedding check let 'brother' pass for 'sister' (0.93) and 'book' for 'letter' (0.82); a language model reading
English catches those. Both are dynamic equivalence questions (same meaning, any wording), never letter overlap:

  fidelity   grammar2/fidelity.py as it stands: every Tonga word glossed (dictionary, engine, corpus co-occurrence),
             DeepSeek lists what is missing / changed / added / tense / negation, verdict faithful | minor | major
  back       the neutral back-translation (stage 8) against the English: same | mostly | different, and what differs.
             Reads only English, so it is as good as the back-translator lets it be

    python -X utf8 tools/chitonga/llm/panel_llm_meaning.py <run folder> [limit]

Adds an[<model>]["llm"] = {"fidelity": ..., "back": ...} to <run>/panel/analysis.jsonl; prints token use.
"""
import collections
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "grammar2"))
import fidelity as F  # noqa: E402
import tables_english as T  # noqa: E402

USE = collections.Counter()
BACK_SYS = """Two English sentences: the ORIGINAL, and a machine BACK-TRANSLATION of a Tonga translation of it. Does the
back-translation say the same thing as the original? Judge meaning only (dynamic equivalence): wording, word order,
politeness and small grammar slips do not matter; a different person, thing, number, negation, tense or mood
(a command read back as a statement counts only if the meaning really changes) does.
Answer as JSON: {"same": "same"|"mostly"|"different", "differs": "<what differs, in a few words, or ''>"}"""


def chat(system, user):
    body = {"model": T.MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 2000, "temperature": 0.2, "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"}}
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + T.key()})
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.load(r)
    u = j.get("usage") or {}
    USE["in"] += u.get("prompt_tokens", 0)
    USE["cached"] += u.get("prompt_cache_hit_tokens", 0)
    USE["out"] += u.get("completion_tokens", 0)
    return j["choices"][0]["message"].get("content") or ""


def main():
    run = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    P = os.path.join(run, "panel", "analysis.jsonl")
    rows = [json.loads(l) for l in open(P, encoding="utf-8")]
    F.chat = chat                                   # fidelity's own prompt and parsing, our client (token count)
    CI, TW = F.corpus_index(set())
    M = F.meanings()
    cache = {}
    jobs = {}
    for r in rows:
        for m, x in r["an"].items():
            if "llm" in x:
                continue
            jobs.setdefault((r["en"], x["toi"]), []).append(x)
    keys = list(jobs)[:limit] if limit else list(jobs)
    for en, toi in keys:                            # glosses first, single-threaded (the reader caches)
        F.hints(toi, M, CI, TW, cache)

    def one(k):
        en, toi = k
        fid = F.check(en, toi, M, CI, TW, cache)
        back = jobs[k][0].get("meaning_neutral", jobs[k][0]["meaning"])["back"]
        try:
            b = json.loads(chat(BACK_SYS, f"ORIGINAL: {en}\nBACK-TRANSLATION: {back}"))
        except Exception as e:
            b = {"error": repr(e)[:120]}
        return k, {"fidelity": fid, "back": b}

    with cf.ThreadPoolExecutor(6) as ex:
        for k, res in ex.map(one, keys):
            for x in jobs[k]:
                x["llm"] = res
            print(f"{res['fidelity'].get('verdict', 'error'):13s} {res['back'].get('same', 'error'):9s} | {k[0][:45]} | {k[1][:45]}",
                  flush=True)
    with open(P, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(keys)} pairs; tokens in {USE['in']} (cached {USE['cached']}) out {USE['out']}")


if __name__ == "__main__":
    main()
