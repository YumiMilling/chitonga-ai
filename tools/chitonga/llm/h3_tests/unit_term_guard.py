"""term_guard.py (22 Oct 2026, the Jesuit): triggers, exclusions, the suppress list, the regex backstop, the wide
re-decode, and that an unarmed batch is decoded exactly as without the guard. A fake translator and tokenizer, so no
model is loaded; the real models are measured in term_guard_eval.py.
    python -X utf8 tools/chitonga/llm/h3_tests/unit_term_guard.py"""
import os
import re
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import term_guard as TG  # noqa: E402

ok, bad = 0, []


def t(name, got, want):
    global ok
    if got == want:
        ok += 1
    else:
        bad.append((name, got, want))


class Tok:
    """one token per word ('▁' + word); the target-language prefix is token 0 of each hypothesis"""
    def tokenize(self, s):
        return ["▁" + w for w in s.split()]

    def convert_tokens_to_ids(self, toks):
        return list(toks)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(i.lstrip("▁") for i in ids)


class Tr:
    """canned beams per source; a hypothesis holding a suppressed token sequence is dropped (as CT2 would never build it)"""
    def __init__(self, beams):
        self.beams, self.calls = beams, []

    def translate_batch(self, srcs, target_prefix=None, suppress_sequences=None, beam_size=4, num_hypotheses=1, **kw):
        self.calls.append({"n": len(srcs), "sup": suppress_sequences, "beam": beam_size, "k": num_hypotheses})
        sup = [tuple(s) for s in (suppress_sequences or [])]
        out = []
        for s in srcs:
            hs = []
            for h in self.beams[" ".join(x.lstrip("▁") for x in s)][:beam_size]:
                toks = ["toi_Latn"] + ["▁" + w for w in h.split()]
                if any(tuple(toks[i:i + len(q)]) == q for q in sup for i in range(len(toks))):
                    continue
                hs.append(toks)
            hs = hs[:num_hypotheses]
            out.append(NS(hypotheses=hs, scores=[-1.0 - i for i in range(len(hs))]))
        return out


tok = Tok()
src = lambda s: ["▁" + w for w in s.split()]  # noqa: E731

# ------------------------------------------------------------------ triggers
t("cholera arms", [r["name"] for r in TG.rules_for("CHOLERA KILLS!")], ["cholera_not_corona"])
t("cholera + covid does not", TG.rules_for("Cholera and COVID-19 are diseases."), [])
t("cholera + coronavirus does not", TG.rules_for("cholera, not the coronavirus"), [])
t("dizzy arms", [r["name"] for r in TG.rules_for("If you feel dizzy in the sun")], ["dizzy_not_diarrhoea"])
t("dizziness arms", [r["name"] for r in TG.rules_for("Dizziness and headache")], ["dizzy_not_diarrhoea"])
t("dizzy + diarrhoea does not", TG.rules_for("Signs: dizziness, diarrhoea and vomiting"), [])
t("no trigger word", TG.rules_for("Wash your hands before eating."), [])
t("toi>en never armed", TG.rules_for("cholera", "toi>en"), [])
t("wells: no rule", TG.rules_for("Fetch water from the wells."), [])
t("both rules", len(TG.rules_for("Cholera can make you dizzy.")), 2)

# ------------------------------------------------------------------ forbidden forms (regex): precise
R = TG.rules_for("cholera")
t("kkolona caught", TG.bad("Kkolona ilajaya!", R), ["cholera_not_corona"])
t("prefixed caught", TG.bad("bulwazi bwakkolona", R), ["cholera_not_corona"])
t("kkolela passes", TG.bad("KKOLELA ILAJAYA", R), [])
t("kkolera passes", TG.bad("kukwabilila kkolera", R), [])
D = TG.rules_for("dizzy")
t("kusoomona caught", TG.bad("Kuti wamvwa kusoomona zuba", D), ["dizzy_not_diarrhoea"])
t("Harris kusomoona caught", TG.bad("kusomoona", D), ["dizzy_not_diarrhoea"])
t("conjugated caught", TG.bad("ulasoomona", D), ["dizzy_not_diarrhoea"])
t("somona (first birth) not forbidden", TG.bad("mwana wakusomona", D), [])
t("somonona (unsheathe) not forbidden", TG.bad("ulasomonona", D), [])
t("cizunguboombe passes", TG.bad("kumvwa cizunguboombe", D), [])

# ------------------------------------------------------------------ suppress list from the model's own tokenizer
seqs = TG.sequences(tok, R)
t("bare form in list", ["▁kkolona"] in seqs, True)
t("capitalised in list", ["▁Kkolona"] in seqs, True)
t("upper in list", ["▁KKOLONA"] in seqs, True)
t("prefixed in list", ["▁bwakkolona"] in seqs, True)
t("never the right word", any("kkolela" in "".join(s).lower() for s in seqs + TG.sequences(tok, D)), False)

# ------------------------------------------------------------------ decoding
beams = {
    "CHOLERA KILLS": ["Kkolona ilajaya", "Kkolela ilajaya"],
    "Wash hands": ["Kosamba maanza"],
    "If you feel dizzy": ["Kuti wamvwa kusoomona", "Kuti wamvwa cizunguboombe"],
    "dizzy again": ["ndasoomona kapati", "ndasoomona", "ndasoomona zuba", "ndasoomona lino", "kumvwa cizunguboombe"],
    "dizzy hopeless": ["ndasoomona"],
}
tr = Tr(beams)
kw = dict(target_prefix=[["toi_Latn"]] * 3, beam_size=4, num_hypotheses=1, return_scores=True)
h, rep = TG.translate_batch(tr, tok, [src("Wash hands"), src("CHOLERA KILLS"), src("If you feel dizzy")],
                            ["Wash hands", "CHOLERA KILLS", "If you feel dizzy"], "en>toi", **kw)
d = lambda x: tok.decode(x[0][0][1:])  # noqa: E731
t("unarmed untouched", (d(h[0]), rep[0]), ("Kosamba maanza", None))
t("cholera: kkolona blocked", d(h[1]), "Kkolela ilajaya")
t("cholera report", (rep[1]["rules"], rep[1]["changed"], rep[1]["blocked"], rep[1]["redecoded"]),
  (["cholera_not_corona"], True, ["cholera_not_corona"], False))
t("dizzy: kusoomona blocked", d(h[2]), "Kuti wamvwa cizunguboombe")
t("whole batch decoded plainly first (unarmed keep that output)", (tr.calls[0]["n"], tr.calls[0]["sup"]), (3, None))
t("each armed text decoded again under suppress", [(c["n"], bool(c["sup"])) for c in tr.calls[1:]], [(1, True), (1, True)])
tr = Tr(beams)
TG.translate_batch(tr, tok, [src("Wash hands")], ["Wash hands"], "en>toi", target_prefix=[["toi_Latn"]], beam_size=4)
t("nothing armed: one plain call", [(c["n"], c["sup"]) for c in tr.calls], [(1, None)])

tr = Tr(beams)
h, rep = TG.translate_batch(tr, tok, [src("dizzy again")], ["dizzy again"], "en>toi",
                            target_prefix=[["toi_Latn"]], beam_size=4, num_hypotheses=1, return_scores=True)
t("conjugated form: regex backstop, wide re-decode", (d(h[0]), rep[0]["redecoded"], rep[0]["unresolved"]),
  ("kumvwa cizunguboombe", True, False))
t("wide beam is 8", max(c["beam"] for c in tr.calls), 8)

h, rep = TG.translate_batch(Tr(beams), tok, [src("dizzy hopeless")], ["dizzy hopeless"], "en>toi",
                            target_prefix=[["toi_Latn"]], beam_size=4, num_hypotheses=1, return_scores=True)
t("no clean beam: kept, alarm, never a written word", (d(h[0]), rep[0]["unresolved"]), ("ndasoomona", True))

h, rep = TG.translate_batch(Tr(beams), tok, [src("If you feel dizzy")], ["If you feel dizzy"], "toi>en",
                            target_prefix=[["eng_Latn"]], beam_size=4, num_hypotheses=1, return_scores=True)
t("other direction untouched", (d(h[0]), rep[0]), ("Kuti wamvwa kusoomona", None))

# ====================================================================== rule shape
# PUBLIC COPY: the working copy also tests the 13 speaker-derived rules of 24 Oct and the speakers' positive glossary,
# one case each with the speaker's own Tonga line. Those cases quote speaker-written Tonga and are left out here.
t("every rule carries a why", all(len(r.get("why", "")) > 30 for r in TG.RULES), True)
t("every rule has when / unless / forbid / forms", all(all(k in r for k in ("when", "unless", "forbid", "forms"))
                                                       for r in TG.RULES), True)
t("no glossary file in the public copy: nothing armed", TG.gloss_armed("Write the number 7.") if not
  TG.glossary_terms() else [], [])

print(f"{ok} passed, {len(bad)} failed")
for b in bad:
    print("FAIL", b)
sys.exit(1 if bad else 0)
