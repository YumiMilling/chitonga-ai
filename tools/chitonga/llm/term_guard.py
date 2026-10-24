"""Term guard: a conditional, forbid-only decode guard for English > Tonga (22 Oct 2026, the Jesuit; council
NEXT_ROUND_2026-10-22_jesuit.md step 2; Chris: "Go" on step 2, for the translator INSIDE the teacher assistant only).

WHY. Two named, dangerous substitutions survive in the translators and are not training problems we can fix this week:
  - cholera > *kkolona* (corona). Stage 16 K wrote "Kkolona ilajaya!" for "CHOLERA KILLS!" (LOGBOOK 22 Oct morning);
    the source is the Bloom COVID booklet's *Kkolonavailasi* in the same health frame. The school / poster form is
    *kkolela* (MoH/WHO poster; Grade 2 guide p96).
  - dizzy > *kusoomona* (have diarrhoea; Thompson 1979 p85, Harris). R and K both write "Kuti wamvwa kusoomona zuba"
    for "If you feel dizzy in the sun" (El Nino ce2t:06). Telling a child that dizziness is diarrhoea is a health error.

WHAT IT DOES. When the English source triggers a rule, the decode runs with CTranslate2's `suppress_sequences` (checked
in CT2 4.5.0, 22 Oct) holding the tokenisations of the forbidden forms, built at run time from WHATEVER tokenizer the
model has (model-independent: R, K or a merge). A tokenisation the list does not cover (a prefixed or conjugated form)
is caught by a regex on the detokenised text; then the line is re-decoded with a wider beam and the first n-best
hypothesis without the form is taken. If every hypothesis still has it, the guarded top beam is kept and an alarm
(`term_guard_unresolved`) is raised: the guard never writes a word, so nothing is invented (Chris's rule; terminology
constraints as negative constraints: Hokamp & Liu 2017, Dinu et al. 2019, cited from memory).

RULES (forbid-only; each says what it blocks and why; add a rule only on a teacher's ruling or a cited dictionary):
  cholera_not_corona  source says cholera, says neither corona nor covid  > forbid k+o[lr]ona (kkolona, Kkolonavailasi)
  dizzy_not_diarrhoea source says dizzy / dizziness, says nothing of       > forbid soomon / somoon (kusoomona, Harris
                      diarrhoea / dysentery / loose stools                    kusomoona). NOT somona 'first birth' and
                                                                              NOT somonona 'unsheathe' (other verbs)
  wells               NO rule: Harris glosses *mugodi* 'mine; well', so *migodi* may be right. Waits for a teacher.
  + further rules mined from speaker corrections (not included in this public copy, see below).

The trigger reads the WHOLE input (`context`), so a cholera poster line that mentions COVID elsewhere is left alone,
and a piece cut out of a longer cholera line is still guarded. Rules carry a "dir" (default en>toi); toi>en rules since 23 Oct.

PUBLIC COPY. The working copy also holds 13 en>toi and 5 toi>en rules mined on 23-24 Oct from paid speakers' own
corrections ("our model wrote X, the speaker wrote Y"; only a WRONG SENSE became a rule, never a spelling variant,
concord or address form). Those rules quote speaker-written Tonga, so they are left out here; the two rules below come
from printed sources and model outputs only. Before such a rule was written, every speaker-written line of those days
was searched for the form about to be forbidden; where a speaker had used it, no rule was written.

A POSITIVE GLOSSARY, not a forbid (glossary_terms(), used by translator_service's n-best selection): the speakers also
gave the RIGHT word for ~40 classroom terms (the term file is not in the public copy). A forbid cannot write a word (Chris's rule); the n-best preference can
choose a hypothesis that already carries the teacher's form. Forms, never a rewrite.

    import term_guard as TG
    res, reps = TG.translate_batch(translator, tok, texts, contexts, "en>toi", target_prefix=..., beam_size=4, ...)
    # res: per text a list of (tokens, score); reps: per text None or {rules, forbid, redecoded, unresolved, changed}
    terms = TG.gloss_armed(english)            # the glossary terms this English carries
    miss  = TG.gloss_missing(tonga, terms)     # the ones whose accepted forms the Tonga has none of
"""
import json
import os
import re

RULES = [
    {"name": "cholera_not_corona",
     "when": re.compile(r"cholera", re.I),
     "unless": re.compile(r"corona|covid", re.I),
     "forbid": re.compile(r"k+o[lr]ona", re.I),
     "forms": ["kkolona", "kolona", "korona", "kkolonavailasi"],
     "why": "corona for cholera (stage 16 K on the MoH/WHO cholera poster, LOGBOOK 22 Oct)"},
    {"name": "dizzy_not_diarrhoea",
     "when": re.compile(r"(?<![A-Za-z])dizz(?:y|iness|ier|iest)(?![A-Za-z])", re.I),
     "unless": re.compile(r"diarrh|dysenter|loose stool|runny stomach|running stomach", re.I),
     "forbid": re.compile(r"soomon|somoon", re.I),
     "forms": ["kusoomona", "soomona", "kusomoona"],
     "why": "diarrhoea for dizzy (R and K on El Nino ce2t:06; soomona 'have diarrhoea', Thompson 1979 p85)"},

    # (speaker-derived rules of 23-24 Oct: not in the public copy)
]
# prefixes seen on the forbidden forms in the stage 16 training text (akusoomona, bwakusoomona, aakusoomona, ...): their
# tokenisation differs from the bare form's, so they get their own suppress sequences; anything else falls to the regex
PREFIXES = ["", "a", "aa", "bwa", "ka", "mwa", "zya", "lya", "ya", "kwa", "wa", "ku", "mu", "ula", "ba"]


def rules_for(context, direction="en>toi"):
    """the rules a source triggers in this direction (a rule without "dir" is en>toi)"""
    if not context:
        return []
    return [r for r in RULES if r.get("dir", "en>toi") == direction and r["when"].search(context)
            and not r["unless"].search(context)]


# ------------------------------------------------------------------------------------- the positive teacher glossary
GLOSSARY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                         "internal", "chitonga_train", "llm", "glossary_terms.json"))
_GL = {}


def glossary_terms(path=None):
    """the teachers' own word for a classroom term, compiled, cached by the file's stamp.

    A row is {term, when, unless, any, forms, why, src}: `when` is the English that arms it, `unless` the English that
    disarms it (the context-sensitive half: 'number' vs 'amount', 'left' vs 'left over'), `any` the regex of EVERY
    accepted Tonga form (both teachers' variants; a variant is never a miss), `forms` the same list in words for the
    alarm, `why`/`src` the teacher and the date. Nothing here is forbidden and nothing is written: the service may
    only PREFER a hypothesis that already carries one of these forms."""
    p = path or GLOSSARY
    try:
        st = (os.path.getmtime(p), os.path.getsize(p))
    except OSError:
        return []
    if _GL.get("stamp") != st or _GL.get("path") != p:
        rows = []
        for r in json.load(open(p, encoding="utf-8"))["terms"]:
            rows.append(dict(r, when_rx=re.compile(r["when"], re.I),
                             unless_rx=re.compile(r.get("unless") or r"(?!x)x", re.I),
                             any_rx=re.compile(r["any"], re.I)))
        _GL.update(stamp=st, path=p, rows=rows)
    return _GL["rows"]


def gloss_armed(context, direction="en>toi", path=None):
    """the glossary terms this English carries (en>toi only: the teachers gave us the TONGA word for an English term)"""
    if not context or direction != "en>toi":
        return []
    return [t for t in glossary_terms(path) if t["when_rx"].search(context) and not t["unless_rx"].search(context)]


def gloss_missing(toi, terms):
    """the armed terms whose accepted forms the Tonga output has none of"""
    return [t for t in terms if not t["any_rx"].search(toi or "")]


def gloss_alarm(terms):
    """the alarm strings for missing terms: 'glossary_missing: <en> > expected <forms>'"""
    return [f"glossary_missing: {t['term']} > expected {' / '.join(t['forms'])}" for t in terms]


def sequences(tok, rules):
    """the token sequences to suppress, from the model's own tokenizer: every form x case x prefix, deduplicated"""
    seqs, seen = [], set()
    for r in rules:
        for f in r["forms"]:
            for p in r.get("prefixes", PREFIXES):
                w = p + f
                for v in (w, w[:1].upper() + w[1:], w.upper()):
                    t = tuple(tok.tokenize(v))
                    if t and t not in seen:
                        seen.add(t)
                        seqs.append(list(t))
    return seqs


def bad(text, rules):
    """the rules whose forbidden form is in a (detokenised) output"""
    return [r["name"] for r in rules if r["forbid"].search(text or "")]


def _detok(tok, h):
    return tok.decode(tok.convert_tokens_to_ids(h[1:]), skip_special_tokens=True)


def translate_batch(tr, tok, token_lists, contexts, direction, compare=True, detok=None, **kw):  # noqa: C901
    """CTranslate2 translate_batch with the term guard. token_lists: source tokens per text; contexts: the whole English
    input per text (None = the text itself is not known: unguarded). kw: the caller's translate_batch arguments
    (target_prefix per text, beam_size, num_hypotheses, ...). Returns (hyps, reports): hyps per text = [(tokens, score)]
    as translate_batch gives them (score None without return_scores); reports per text = None (no rule) or a dict.
    Unarmed texts get exactly the output they would get without the guard (the whole batch's plain call); armed texts
    are decoded a second time, alone with their rule set, under suppress_sequences. compare is kept for the API: the
    plain output of an armed text is always known now, so every report carries before / after / changed / blocked."""
    detok = detok or (lambda h: _detok(tok, h))
    n = len(token_lists)
    contexts = list(contexts) if contexts is not None else [None] * n
    tp = kw.pop("target_prefix", None)
    groups = {}
    for i in range(n):
        key = tuple(r["name"] for r in rules_for(contexts[i], direction))
        groups.setdefault(key, []).append(i)
    hyps, reps = [None] * n, [None] * n

    def run(idx, extra):
        sub_tp = [tp[i] for i in idx] if tp is not None else None
        res = tr.translate_batch([token_lists[i] for i in idx], target_prefix=sub_tp, **dict(kw, **extra))
        return [list(zip(r.hypotheses, r.scores if r.scores else [None] * len(r.hypotheses))) for r in res]

    if all(not k for k in groups):                     # nothing armed: the caller's own call, unchanged
        return run(list(range(n)), {}), reps
    # Something is armed. The WHOLE batch is first decoded exactly as without the guard: an unarmed text keeps that
    # output (an int8 batch's padding changes outputs, so taking armed texts out of the batch would change their
    # neighbours: measured 22 Oct, R on cholera_pairs:12). An armed text's plain output is its 'before'.
    base = run(list(range(n)), {})
    for i in range(n):
        hyps[i] = base[i]
    for key, idx in groups.items():
        if not key:
            continue
        rules = [r for r in RULES if r["name"] in key]
        seqs = sequences(tok, rules)
        guarded = run(idx, {"suppress_sequences": seqs})
        for j, i in enumerate(idx):
            g = guarded[j]
            rep = {"rules": list(key), "forbid": [r["forbid"].pattern for r in rules], "n_seq": len(seqs),
                   "redecoded": False, "unresolved": False}
            if g and bad(detok(g[0][0]), rules):
                beam = max(8, 2 * kw.get("beam_size", 2))
                wide = run([i], {"suppress_sequences": seqs, "beam_size": beam, "num_hypotheses": beam})[0]
                clean = [h for h in wide if not bad(detok(h[0]), rules)]
                rep["redecoded"] = True
                if clean:
                    k = kw.get("num_hypotheses", 1)
                    pool, seen = clean + [h for h in g if not bad(detok(h[0]), rules)], set()
                    g = [h for h in pool if not (tuple(h[0]) in seen or seen.add(tuple(h[0])))][:max(1, k)]
                else:
                    rep["unresolved"] = True
            hyps[i] = g
            if compare:
                before, after = detok(base[i][0][0]), detok(g[0][0])
                rep.update(before=before, after=after, changed=before != after,
                           blocked=bad(before, rules))
            reps[i] = rep
    return hyps, reps
