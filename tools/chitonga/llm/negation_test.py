"""Negation test (17 Oct 2026; Chris: "We will test it on health stuff, and negatives are super important").

Every held-out pair (book exams, the readers' last lessons, the health set) whose source carries a negative is
translated by each model, in both directions, and checked automatically:
  English > Tonga : the English has not / never / no / n't / nobody / nothing / without / cannot  ->  the Tonga output
                    must contain a word engine 2 reads as a NEGATIVE tense (neg_*, ta-...-i, tii-/tee-, ta-na-, -ninga-,
                    ko-ta-, ku-ta-, nse-/nsi-/tandi-)
  Tonga > English : the Tonga source has such a word  ->  the English output must carry a negation
A dropped negative is a HARD FAIL (it reverses the meaning: "does not kill quickly" > "kills quickly"). Also reported:
the health subset on its own.

    python -X utf8 tools/chitonga/llm/negation_test.py <model dir> [<model dir> ...]
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "grammar2"))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import panel_analyse as PA  # noqa: E402
import reader  # noqa: E402

VOC = os.path.join(REPO, "internal", "chitonga_train", "vocab", "new_books_2026-10-16")
EN_NEG = re.compile(r"\b(not|never|no|nobody|nothing|none|without|cannot|neither|nor|didnt|dont|doesnt|cant|wont|"
                    r"couldnt|isnt|wasnt|arent|werent|havent|hasnt|hadnt|shouldnt|wouldnt|mustnt)\b|n't\b", re.I)
STRICT = re.compile(r"^(tii|tee|taa|too|tandi|nsi|nse|nsy|kata|uuta|uta|muta|bata|kaata|ta[bcklmtuyz][a-z]{3,}|to[cbklmz][a-z]{3,})")
WORD = re.compile(r"[A-Za-zŋŊ']+")
NEG_TAG = re.compile(r"neg|never|not_yet|ninga|proh|ta_ni", re.I)
_cache = {}


# 19 Oct 2026 (the Jesuit): "ccita" 'I don't know' (Thompson p28, p59; lexicon) added: a correct 'Ccita.' for "I don't
# know" was a HARD neg_dropped in h2_checks.negation
LEX_NEG = {"ccita", "kunyina", "kakunyina", "takwe", "taakwe", "tiikwe", "pe", "peepe", "nobaba", "kwiina", "kutaba", "takuli",
           "tazi", "nsizi", "nsyezi", "tandizi", "tiibakali", "taakali", "tiikwakali"}
# 17 Oct 2026, read by hand on s11/s12: negatives the detector missed (citapwilimwi, zitayendelani, ncaatakabwene,
# kakutakonzya, teakamukonzya, Nsei, katakwe, taseluki, nikubula 'even without', wakakilwa 'he failed')
LEX_NEG_RX = re.compile(r"^(nse[iy]|nsi[iy]|k?a?takwe|n?i?kubula|\w*kakilwa|\w*kabula)$")
INNER = re.compile(r"^(ci|zi|zyi|li|lu|bu|tu|ku|kaku|kakuta|nca|ncaa|mpaa|nkwa|nkwaa|ka|no|noci|i|u)ta[a-z]{3,}|^te[a-z]?aka|^tas[a-z]{3,}")
# 19 Oct 2026 (the Jesuit, building gen_grammar_v08): "to" added. ta + u > to- is the 2s negative (toboni, tokalyi,
# tona ula, tokonzyi: rules.tsv, F086 Tonaula); without it every 2s negative read as positive, so the harness's negation
# check (h2_checks.negation, a HARD check) raised neg_dropped on correct 2s lines. The reader still decides (tola 'take')
SURF = re.compile(r"^(ta|to|tii|tee|taa|too|nse|nsi|nsy|nke|tandi|tatu|tamu|taba|tazi|tali|tau|tai|tacu|taci|tabu|taku|kota|kuta|amuta|muta|uta|bata|kata|kaba?ta|cita|zita|lita|ita|ata)")


def toi_neg(text):
    for w in WORD.findall(text.lower()):
        if w in LEX_NEG or LEX_NEG_RX.match(w):
            return True
        if ("tana" in w[2:] or "taana" in w) and len(w) >= 6:     # 17 Oct: nocitana, kalutana were skipped by SURF
            return True
        if not (SURF.match(w) or INNER.match(w)):
            continue
        if w not in _cache:
            try:
                rs = reader.read(w)
            except Exception:  # noqa: BLE001
                rs = []
            neg_read = any(NEG_TAG.search(str(r)) for r in (rs or [])[:4])
            pos_ta_root = any(len(r) > 1 and str(r[1]).startswith(("ta", "ti", "te", "to")) and not NEG_TAG.search(str(r))
                              for r in (rs or []))   # a positive verb whose root starts with ta (talika, tambula...)
            # the reader misses negatives on stems it does not know (taukomena, katasowa): fall back on the strict
            # surface prefix, unless the reader parses the word as a positive verb on a ta-/ti- root
            _cache[w] = ("tana" in w[2:] or "taana" in w) or neg_read or (len(w) >= 6 and bool(STRICT.match(w) or INNER.match(w)) and not pos_ta_root)
        if _cache[w]:
            return True
    return False


def items():
    out = []
    for f in sorted(glob.glob(os.path.join(VOC, "*pairs*.jsonl"))):
        if "fidelity" in f or "SUPERSEDED" in f:
            continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l) if l.strip() else None
            if not r or not r.get("held_out") or not r.get("toi") or not r.get("en_guess"):
                continue
            health = "poster" in f or (r.get("pdf") in (49, 50) and "g10" in f)
            out.append({"toi": r["toi"], "en": r["en_guess"], "src": os.path.basename(f), "health": health})
    return out


def main():
    it = items()
    en_neg = [x for x in it if EN_NEG.search(x["en"])]
    toi_neg_items = [x for x in it if toi_neg(x["toi"])]
    print(f"held-out items {len(it)}; with English negation {len(en_neg)}; with Tonga negation {len(toi_neg_items)}; "
          f"health {sum(x['health'] for x in it)}")
    for d in sys.argv[1:]:
        m = PA.Model(d)
        tok = m.tok
        fails = {"en>toi": [], "toi>en": []}
        tok.src_lang = "eng_Latn"
        for i in range(0, len(en_neg), 16):
            ch = en_neg[i:i + 16]
            res = m.tr.translate_batch([tok.convert_ids_to_tokens(tok.encode(x["en"])) for x in ch],
                                       target_prefix=[["toi_Latn"]] * len(ch), beam_size=4, max_decoding_length=200)
            for x, r in zip(ch, res):
                out = tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True)
                if not toi_neg(out):
                    fails["en>toi"].append((x, out))
        for x in toi_neg_items:
            out = m.back(x["toi"])
            if not EN_NEG.search(out):
                fails["toi>en"].append((x, out))
        name = os.path.basename(d.rstrip("/\\"))
        a, b = len(fails["en>toi"]), len(fails["toi>en"])
        ha = sum(x["health"] for x, _ in fails["en>toi"]); hb = sum(x["health"] for x, _ in fails["toi>en"])
        print(f"\n{name}: negative KEPT  en>toi {len(en_neg) - a}/{len(en_neg)}   toi>en {len(toi_neg_items) - b}/{len(toi_neg_items)}"
              f"   (health fails: en>toi {ha}, toi>en {hb})")
        for k in ("en>toi", "toi>en"):
            for x, out in fails[k][:6]:
                print(f"   DROPPED {k}: {x['toi'][:60]} | {x['en'][:60]} -> {out[:70]}")
        json.dump({k: [{"toi": x["toi"], "en": x["en"], "out": o, "health": x["health"]} for x, o in v] for k, v in fails.items()},
                  open(os.path.join(VOC, f"negation_fails_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
