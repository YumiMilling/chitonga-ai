"""Harness v2, the readers (17 Oct 2026; TODO_ENGINE_HARNESS E1, E6, F2, F3).

E1  the translator that reads the Tonga back: stage 12b seed 1 by default, int8 CTranslate2, loaded exactly as
    panel_analyse.Model loads it (tokenizer from the model folder, weights from <folder>-ct2-int8). One setting swaps
    it: HARNESS_READER=<model folder> (stage 13 when it is kept). The same model drafts Tonga when the harness is
    told to use the translator as the drafter (HARNESS_DRAFTER=nllb).
F2  long text is split (h2_text.split, > 25 words) before either direction and rejoined.
F3  names are protected both ways: Tonga > English through names.safe_translate (neutral stand-ins, gendered stand-ins,
    repair; 71 of 71 names on the 16 Oct test); English > Tonga through h2_text.protect_en / restore.
E6  the chat model's reading of a Tonga line (llama-server, port 8090), a second opinion when the translator drafted;
    None when the server is not up (never an error).
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "tools", "chitonga", "vectors")]
import h2_text as T  # noqa: E402

READER = os.environ.get("HARNESS_READER", os.path.join(os.environ.get("CHITONGA_MODELS", r"models"), "chitonga-nllb-s13-soup"))   # 18 Oct: the stage 13 soup is the base
PORT = int(os.environ.get("HARNESS_PORT", "8090"))
_m = {}


class Reader:
    """the panel_analyse.Model loading, without importing panel_analyse (it imports harness and switches the
    chat model's self-check off as a side effect)"""

    def __init__(self, d=READER):
        import ctranslate2
        from transformers import AutoTokenizer
        self.dir = d
        self.name = os.path.basename(d.rstrip("/\\")).replace("chitonga-nllb-", "")
        self.tok = AutoTokenizer.from_pretrained(d)
        # CT2_DEVICE / CT2_COMPUTE (22 Oct, the negation sweep on a pod): default the Beelink CPU int8, as served
        self.tr = ctranslate2.Translator(d + "-ct2-int8", device=os.environ.get("CT2_DEVICE", "cpu"),
                                         compute_type=os.environ.get("CT2_COMPUTE", "int8"),
                                         intra_threads=int(os.environ.get("HARNESS_THREADS", max(1, (os.cpu_count() or 4) - 2))))

    def _run(self, texts, src, tgt, maxlen=200):
        tok = self.tok
        tok.src_lang = src
        res = self.tr.translate_batch([tok.convert_ids_to_tokens(tok.encode(t)) for t in texts],
                                      target_prefix=[[tgt]] * len(texts), beam_size=4, max_decoding_length=maxlen)
        return [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True) for r in res]

    def back_raw(self, toi):
        return self._run([toi], "toi_Latn", "eng_Latn", 160)[0]

    def back(self, toi):
        """Tonga > English: split, names protected per piece, rejoined. Returns (english, report)"""
        import names as N
        pieces = T.split(toi)
        outs, reps = [], []
        for p in pieces:
            en, rep = N.safe_translate(p, self.back_raw)
            outs.append(en)
            reps.append(rep)
        missing = [m for r in reps for m in r.get("missing", [])]
        return T.join(outs), {"pieces": len(pieces), "names": [n for r in reps for n in r.get("names", [])],
                              "missing": missing, "methods": sorted({r["method"] for r in reps})}

    def translate(self, en):
        """English > Tonga: split, names carried through (h2_text.safe_en), rejoined. Returns (tonga, report)"""
        pieces = T.split(en)
        outs, reps = [], []
        for p in pieces:
            t, rep = T.safe_en(p, lambda x: self._run([x], "eng_Latn", "toi_Latn")[0])
            outs.append(t)
            reps.append(rep)
        return T.join(outs), {"pieces": len(pieces), "names": [n for r in reps for n in r["names"]],
                              "missing": [n for r in reps for n in r["missing"]]}

    def certainty(self, en, toi):
        """mean log-prob per token of the Tonga given the English (a SIGNAL, never a licence to send: E7)"""
        tok = self.tok
        tok.src_lang = "eng_Latn"
        s = tok.convert_ids_to_tokens(tok.encode(en))
        tok.src_lang = "toi_Latn"
        t = tok.convert_ids_to_tokens(tok.encode(toi))
        r = self.tr.score_batch([s], [t])[0]
        return round(sum(r.log_probs) / max(1, len(r.log_probs)), 3)


def reader():
    if "r" not in _m:
        _m["r"] = Reader()
    return _m["r"]


def server_up(port=PORT):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
        return True
    except Exception:  # noqa: BLE001
        return False


def chat(messages, max_tokens=160, port=PORT):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=300))
    return r["choices"][0]["message"]["content"].strip()


READ_ASK = "Translate this Chitonga (Tonga of southern Zambia) into English. Reply with the English only.\n\n"


def chat_read(toi):
    """E6: the chat model's English reading of a Tonga line, or None (server down, or an error)"""
    if not server_up():
        return None
    try:
        return chat([{"role": "user", "content": READ_ASK + toi}], 160).split("\n")[0].strip()
    except Exception:  # noqa: BLE001
        return None
