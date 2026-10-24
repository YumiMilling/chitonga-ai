"""Harness v3, step 2: checks on a draft POOL, the selection key, and the v3 send gate (19 Oct 2026, the Jesuit;
HARNESS_V3_PLAN.md §1.2-1.5).

Why. v0.9's self-score is a constant 5/5, and without it v2 sends 8 of the 34 wrong drafts of the labelled set (all
short lines the reader rebuilt). v3 replaces the self-check by OUTSIDE evidence: three readers instead of one, the
agreement of the model's own drafts (semantic spread: Farquhar et al., Nature 2024), and attestation. The model's
self-score is never called.

Checks on every draft (answer only, never the thought):
  v2 pieces   attestation, form (judge), rules (error_map.extra), drift, preservation (numbers, names, glossary),
              the soup round trip and the translator's certainty (a SIGNAL, never in the key: E7)
  v3          h3_detect (ide_rule, negfut_mixed, loop3, colour, neg3 in place of v2's negation, askback, lex_align)
  readers     soup (always) + S13 s1 and s2 (lazily, translate mode, stage 2 only): thought against the English,
              gender-neutralised (bigc_score._neutral rule), median of the three
  agreement   each draft's mean thought against the other drafts' soup readings (MBR medoid; Freitag et al. 2022);
              the pool's spread = 1 - mean pairwise thought

Selection key (lexicographic, smaller is better; §1.3): excluded (block) > hard issues > attestation weight > soft form
issues > meaning (translate: -median reader thought rounded to 0.05; converse: echo + low relevance) > -form agreement
(drafts with the same words) > -meaning agreement (rounded 0.05) > repaired > not greedy > length. Certainty is NOT in
the key (E7).

Send gate (§1.4), every rule only ADDS reasons (E7):
  - any block, hard issue, attestation flag, drift, form issue: route (as v2)
  - translate: 2 of the 3 readers at thought >= thought_min (when only the soup has read: the soup >= thought_min)
  - short line (< 4 words, not a memory line): the soup's reading a near copy (>= short_evidence) AND 2 of 3 readers
    AND (pool of >= 3 drafts) at least short_form_agree other drafts with the same words AND spread <= spread_route.
    Agreement alone never sends. Why FORM agreement and not only the plan's meaning spread: set D, 19 Oct (BUILD).
  - a pool spread above spread_route: route (the model disagrees with itself: confabulation)
  - converse: the reply read back as a copy of the user's turn (echo) routes; relevance of the reply's reading to the
    user turn's reading below relevance_min routes [threshold to be set on the replay]
  - a thought signal (never closed, "not sure") routes
Thresholds come from the PROFILE (h3_profile), with the replay that set them.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "tools", "chitonga", "grammar2"), os.path.join(REPO, "tools", "chitonga", "vectors")]
import h2_text as T  # noqa: E402
import h3_detect as D  # noqa: E402

MODELS = os.environ.get("CHITONGA_MODELS", r"models")   # env: a pod run (22 Oct)
EXTRA_READERS = {"s13_s1": os.path.join(MODELS, "chitonga-nllb-s13-S13_bi_s1"),
                 "s13_s2": os.path.join(MODELS, "chitonga-nllb-s13-S13_bi_s2")}
ATTEST_W = {"not_tonga": 3, "contact": 2, "mixed": 2, "unattested": 1, "book_only": 1, "english": 1}
HARD_V2 = {"neg_dropped", "neg_added", "neg_roundtrip", "num_missing", "num_added", "num_roundtrip", "name_missing",
           "name_roundtrip", "gloss_missing", "gloss_wrong", "gloss_roundtrip", "antonym_vs_memory"}
# concord_ref_*: hard since step 4's measurement (19 Oct): 1.9% false alarms on 637 real book-dialogue replies, 0.5% on
# Thompson's conversations, 0% on 2,652 engine replies (plan §3: hard only under 5%)
HARD_V3 = {"askback_unjustified", "colour", "ide_rule", "negfut_mixed", "concord_ref_obj", "concord_ref_subj",
           "english_leak", "format_junk"}
SOFT_V3 = {"obj_missing"}         # counted with the soft form issues in selection; a signal, never a routing reason
BLOCK_V3 = {"loop3"}
SOFT_FORM = {"unknown", "guessed", "person", "timetense", "english", "concord", "subject", "negfut"}
_readers = {}
_emb = {}
_SHE = [(re.compile(r"\bshe\b", re.I), "he"), (re.compile(r"\bherself\b", re.I), "himself"),
        (re.compile(r"\bhers?\b", re.I), "his")]


def neutral(s):
    for rx, to in _SHE:
        s = rx.sub(to, s or "")
    return s


# ------------------------------------------------------------------------------------------------ readers
def reader(name="soup"):
    import h2_reader as R
    if name == "soup":
        return R.reader()
    if name not in _readers:
        _readers[name] = R.Reader(EXTRA_READERS[name])
    return _readers[name]


def back_many(r, texts):
    """Tonga > English for many drafts: pieces without names go through ONE batched call, pieces with names through
    names.safe_translate one by one (h2_reader.Reader.back does the same per draft). Returns [(english, report)]"""
    import names as N
    pieces = [T.split(t) for t in texts]
    plain, plain_at = [], []
    outs = [[None] * len(p) for p in pieces]
    reps = [[None] * len(p) for p in pieces]
    for i, ps in enumerate(pieces):
        for j, p in enumerate(ps):
            spans = N.tag(p)
            if any(s["type"] in N.KEEP_TYPES for s in spans):
                outs[i][j], reps[i][j] = N.safe_translate(p, r.back_raw)
            else:
                plain.append(p)
                plain_at.append((i, j))
    if plain:
        res = r._run(plain, "toi_Latn", "eng_Latn", 160)
        for (i, j), en in zip(plain_at, res):
            outs[i][j], reps[i][j] = en, {"method": "plain", "names": [], "missing": []}
    out = []
    for i in range(len(texts)):
        rp = reps[i]
        out.append((T.join(outs[i]), {"pieces": len(pieces[i]), "names": [n for x in rp for n in x.get("names", [])],
                                      "missing": [m for x in rp for m in x.get("missing", [])],
                                      "methods": sorted({x["method"] for x in rp})}))
    return out


def soup_nbest(en, k=2):
    """the translator's top-k beams for the English (translate mode only; names carried through by h2_text.safe_en
    on the first beam; further beams only when the English has no name). A second opinion by construction."""
    r = reader("soup")
    if T.en_names(en):
        return [r.translate(en)[0]][:k]
    tok = r.tok
    tok.src_lang = "eng_Latn"
    res = r.tr.translate_batch([tok.convert_ids_to_tokens(tok.encode(en))], target_prefix=[["toi_Latn"]],
                               beam_size=max(4, k), num_hypotheses=k, max_decoding_length=200)[0]
    return [tok.decode(tok.convert_tokens_to_ids(h[1:]), skip_special_tokens=True) for h in res.hypotheses]


def embed(texts):
    if not _emb:
        import harness as H
        H.thought("a", "b")                 # loads bge-small the way v1 / v2 do
        _emb["m"] = H._emb["m"]
    v = np.array(list(_emb["m"].embed([t or "" for t in texts])), dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return v


# ------------------------------------------------------------------------------------------------ checks
_memo = {}
MEMO_MAX = 200000


def _check_texts(texts, en, mode, wc, user, readers, certainty, history=()):
    """per-draft checks for texts not yet in the memo (batched round trips). A draft's checks depend only on its text,
    the English / user turn and the readers, so the replay re-uses them across pool sizes (the n-curve)."""
    import attest
    import harness as H
    import h2_checks as C
    kind = "translate" if mode == "translate" else "converse"
    ctx = (mode, en or "", user or "", (wc or {}).get("note", ""), tuple(readers), tuple(history or ())[-2:])
    todo = [t for t in dict.fromkeys(texts) if (ctx, t) not in _memo]
    if todo:
        # an ask-back quotes the USER's word: that word is not the model's Tonga and is not checked
        chk = [re.sub(r"['‘’\"][^'‘’\"]+['‘’\"]", "''", t) if mode == "converse" and _askback(t) else t for t in todo]
        backs = {"soup": back_many(reader("soup"), chk)}
        for rn in readers:
            if rn != "soup":
                backs[rn] = back_many(reader(rn), chk)
        vecs = None
        if en and mode == "translate":
            vecs = embed([neutral(en)] + [neutral(b[0]) for rn in backs for b in backs[rn]])
        soup_v = embed([b[0] for b in backs["soup"]])
        uv = user_back = None
        if mode == "converse" and user:
            key_u = ("user_back", user)
            if key_u not in _memo:
                ub = back_many(reader("soup"), [user])[0][0]
                _memo[key_u] = (ub, embed([ub])[0])
            user_back, uv = _memo[key_u]
        r0 = reader("soup")
        for i, t in enumerate(chk):
            c = {"attest": [list(x) for x in attest.line(t)],
                 "form": [list(x) for x in H.judge.judge(t, en if kind == "translate" else None)],
                 "rules": [list(x) for x in H.EM.extra(t, en or "", kind, set())],
                 "drift": C.drift(t)}
            back, brep = backs["soup"][i]
            c["back"] = {"text": back, "by": r0.name, **brep}
            pres = []
            if kind == "translate" and en:
                pres += [x[:2] for x in D.negation3(en, t, back)] + C.numbers(en, t, back) + C.names(en, t, back)
            else:
                pres += C.names("", t, back)
            pres += C.glossary(en if kind == "translate" else None, t, back)
            if brep.get("missing"):
                pres.append(("name_roundtrip", f"the reader lost name(s) {brep['missing']}"))
            c["preserve"] = [list(x) for x in pres]
            c["v3"] = [x for x in D.detect(t, en, mode, wc=wc, user=user) if not x[0].startswith("neg_")]  # neg3 in preserve
            if mode == "converse" and user:          # step 4: concord across turns (user turn + the last two turns)
                import h3_session as SE
                c["v3"] += [list(x) for x in SE.concord_refs(t, list(history or ())[-2:] + [user])]
            m = {}
            if vecs is not None:
                k = 1
                for rn in backs:
                    m[rn] = round(float(vecs[0] @ vecs[k + i]), 3)
                    k += len(chk)
                vals = sorted(m.values())
                m["median"] = vals[len(vals) // 2] if len(vals) % 2 else round((vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2, 3)
                m["readers"] = {rn: backs[rn][i][0] for rn in backs}
                if certainty:
                    try:
                        m["certainty"] = r0.certainty(en, t)
                    except Exception:  # noqa: BLE001
                        m["certainty"] = None
            if user_back is not None:
                m["user_back"] = user_back
                m["relevance"] = round(float(soup_v[i] @ uv), 3)
                m["echo"] = _echo(t, user)
            if len(_memo) > MEMO_MAX:
                _memo.clear()
            _memo[(ctx, todo[i])] = {"checks": c, "meaning": m, "vec": soup_v[i]}
    return [_memo[(ctx, t)] for t in texts]


def check_pool(pool, en=None, mode="translate", wc=None, user=None, readers=("soup",), certainty=True, history=()):
    """fills every usable draft of the pool with checks; returns pool-level facts (spread, n usable).
    pool: [{'src', 'answer', ...}]; d['text'] = what the user would see (translate mode: the first line)"""
    import copy
    use = [d for d in pool if (d.get("answer") or "").strip()]
    for d in pool:
        if d not in use:
            d["usable"] = False
    if not use:
        return {"spread": None, "n_usable": 0}
    for d in use:
        a = d["answer"].strip()
        d["text"] = a.split("\n")[0].strip() if mode == "translate" else a
    texts = [d["text"] for d in use]
    items = _check_texts(texts, en, mode, wc, user, readers, certainty, history)
    for d, it in zip(use, items):
        d["usable"] = True
        d["checks"] = copy.deepcopy(it["checks"])
        d["meaning"] = copy.deepcopy(it["meaning"])
    # FORM agreement: how many other drafts say exactly the same words (case and punctuation aside). Measured on set D
    # (19 Oct): the readers repair wrong forms (Ndatakala, Nkalala, Ndiimvwi all read back as the English), so the
    # meaning spread does not separate wrong from right short lines; the model's own samples disagreeing in FORM does.
    forms = [tuple(re.findall(r"[a-zŋ']+", t.lower())) for t in texts]
    for i, d in enumerate(use):
        d["form_agree"] = sum(1 for j, f in enumerate(forms) if j != i and f == forms[i])
    # agreement among the drafts' readings (soup)
    n = len(use)
    spread = None
    if n >= 2:
        V = np.stack([it["vec"] for it in items])
        Sm = V @ V.T
        for i, d in enumerate(use):
            d["agreement"] = round(float((Sm[i].sum() - Sm[i, i]) / (n - 1)), 3)
        spread = round(1.0 - float((Sm.sum() - np.trace(Sm)) / (n * (n - 1))), 3)
    else:
        use[0]["agreement"] = None
    return {"spread": spread, "n_usable": n}


def _askback(t):
    import h3_wordcheck as W
    return W.is_askback(t)


def _echo(reply, user):
    a = re.findall(r"[a-zŋ']+", (reply or "").lower())
    b = re.findall(r"[a-zŋ']+", (user or "").lower())
    if not a or not b:
        return False
    if a == b or (len(a) >= 3 and sum(x in b for x in a) / len(a) >= 0.9 and abs(len(a) - len(b)) <= 1):
        return True
    # 'Inzya, ' + the user's whole sentence again (replay 19 Oct: 'Inzya, mwezi ulabala azyalo nyenyezi zilabala.'):
    # parroting, not an answer. A short agreement that takes up the verb ('Inzya, balakomena') is natural and passes.
    if a[0] in ("inzya", "ii", "ee", "peepe", "pe"):
        rest = a[1:]
        return len(rest) >= 4 and sum(x in b for x in rest) / len(rest) >= 0.8 and len(rest) >= 0.8 * len(b)
    return False


# ------------------------------------------------------------------------------------------------ issues and key
def english_parts(text, attest_items):
    """words of the sentences that are English (>= 60% of their words carry attest's 'english' verdict): in task
    teach_english these parts are EXPECTED English (TEACHER_PROFILE.md §2: the language check is per part)"""
    eng = {w.lower() for v, w, _ in attest_items if v == "english"}
    out = set()
    for s in re.split(r"(?<=[.!?:;])\s+|\n+", text or ""):
        ws = [w.lower() for w in re.findall(r"[A-Za-zŋŊ']+", s)]
        if ws and sum(w in eng for w in ws) >= 0.6 * len(ws):
            out.update(ws)
    return out


def issues(d, task=None):
    """the draft's issues by tier: block, hard, attest (weighted), soft, signals. task (TEACHER_PROFILE.md §2):
    teach_english exempts the English parts from the Tonga checks; explain_subject makes English words a signal"""
    c = dict(d.get("checks") or {})
    out = {"block": [], "hard": [], "attest": [], "attest_w": 0, "soft": [], "drift": [], "signals": []}
    if task in ("teach_tonga", "teach_english", "explain_subject", "advice"):
        # TEACHER_PROFILE.md v1.1 (19 Oct, set F smoke): in every teacher task an ENGLISH sentence is allowed (the
        # assistant explains to the teacher in the language the teacher wrote in); every Tonga sentence keeps all checks
        ep = english_parts(d.get("text"), c.get("attest", []))
        if ep:
            c["attest"] = [x for x in c.get("attest", []) if x[1].lower() not in ep]
            c["form"] = [x for x in c.get("form", []) if not (str(x[1]).lower() in ep)]
            c["drift"] = [x for x in c.get("drift", []) if x[0].lower() not in ep]
            rest = " ".join(s for s in re.split(r"(?<=[.!?:;])\s+|\n+", d.get("text") or "")
                            if not set(w.lower() for w in re.findall(r"[A-Za-zŋŊ']+", s)) <= ep)
            # quoted English inside a Tonga sentence is the lesson material ("Kaambo kalatondezya kuti, "She doesn't
            # like ...""): not a leak in a teacher task (replay 20 Oct, B-EC02)
            rest = re.sub(r"[\"“][^\"“”]*[\"”]|‘[^’]*’", " ", rest)
            c["v3"] = [x for x in c.get("v3", []) if x[0] != "english_leak"] + [list(x) for x in D.english_leak(rest)]
            out["signals"].append(f"English parts allowed ({task}): {len(ep)} words")
    if task == "explain_subject":
        en_words = [x for x in c.get("attest", []) if x[0] == "english"]
        if en_words:
            out["signals"].append("English subject terms (explain_subject): " + ", ".join(x[1] for x in en_words))
            c["attest"] = [x for x in c.get("attest", []) if x[0] != "english"]
            c["form"] = [x for x in c.get("form", []) if x[0] != "english"]
    # judge's 'unknown' / 'guessed' predate the shared word check (attest, 19 Oct: KNOWN = attested anywhere, in the
    # lexicon, or analysable, loans included). A word attest knows is not an unknown word here (replay 19 Oct, A140:
    # judge called the loan 'mufwooni' unknown, so a draft saying 'by car' won on 0 soft issues)
    flagged = {str(x[1]).lower() for x in c.get("attest", [])}
    c["form"] = [x for x in c.get("form", []) if x[0] not in ("unknown", "guessed") or str(x[1]).lower() in flagged]
    rel = [x for x in c.get("rules", []) if x[0] == "religion" and not re.match(r"nd\w*komba", x[1])]
    if rel:
        out["block"].append(f"religion unasked: {rel[0][1]}")
    kinds = {x[0] for x in c.get("form", []) + c.get("rules", [])}
    if "repetition" in kinds:
        out["block"].append("decoding loop (a word 3+ times)")
    for x in c.get("v3", []):
        if x[0] in BLOCK_V3:
            out["block"].append(f"{x[0]}: {x[1]}")
        elif x[0] in HARD_V3:
            out["hard"].append(f"{x[0]}: {x[1]}")
        else:
            out["signals"].append(f"{x[0]}: {x[1]}")
    for k, det in c.get("preserve", []):
        if k in HARD_V2:
            out["hard"].append(f"{k}: {det}")
        else:
            out["signals"].append(f"{k}: {det}")
    for v, w, note in c.get("attest", []):
        out["attest"].append(f"attestation: {w} ({v}: {note})")
        out["attest_w"] += ATTEST_W.get(v, 0)
    bad = sorted(kinds & SOFT_FORM)
    if bad:
        out["soft"].append("form/rule: " + ", ".join(bad))
    for w, reason in c.get("drift", []):
        out["drift"].append(f"drift: {w} ({reason})")
    out["n_soft"] = sum(1 for x in c.get("form", []) + c.get("rules", []) if x[0] in SOFT_FORM) + len(out["drift"]) \
        + sum(1 for x in c.get("v3", []) if x[0] in SOFT_V3)
    return out


MEANING_FAR = 0.10


def key(d, mode, best_median=None):
    """meaning_far (replay 19 Oct, A140): a draft whose median reader thought is more than MEANING_FAR below the pool's
    best ranks after every draft near the best, BEFORE the soft form tier: a split word ('Ndila tumina') must not beat
    the right meaning ('kumootokala', by car, for 'by phone' won on 0 soft issues)"""
    if not d.get("usable"):
        return (9,)
    iss = d["issues"]
    m = d.get("meaning") or {}
    far = 0
    if mode == "translate":
        mean = -round(round((m.get("median") or 0) / 0.05) * 0.05, 2)
        if best_median is not None and (m.get("median") or 0) < best_median - MEANING_FAR:
            far = 1
    else:
        mean = (1 if m.get("echo") else 0) + (1 if (m.get("relevance") is not None and m["relevance"] < 0.3) else 0)
    agr = -round(round((d.get("agreement") or 0) / 0.05) * 0.05, 2)
    return (1 if iss["block"] else 0, len(iss["hard"]), iss["attest_w"], far, iss["n_soft"], mean, -(d.get("form_agree") or 0),
            agr, 1 if d.get("repair") else 0, 0 if d.get("src") == "greedy" else 1, len(d.get("text") or ""))


def select(pool, mode, task=None):
    for d in pool:
        if d.get("usable"):
            d["issues"] = issues(d, task)
    meds = [(d.get("meaning") or {}).get("median") for d in pool if d.get("usable")]
    meds = [x for x in meds if x is not None]
    best = max(meds) if meds else None
    for d in pool:
        d["key"] = list(key(d, mode, best))
    order = sorted(range(len(pool)), key=lambda i: (tuple(pool[i]["key"]), i))
    return order


# ------------------------------------------------------------------------------------------------ gate
YES_OPEN = re.compile(r"^\W*(inzya|ii|ee|eeya|ee\s*inzya)\b[,.!]?\s*(.*)$", re.I | re.S)
YESNO_Q = re.compile(r"^\W*(sena|hena)\b", re.I)


def yes_reflex(user, reply):
    """DECISIONS 19 Oct (YES-REFLEX, the harness safety net): a reply opening with Inzya / Ii to a turn that is not a
    yes/no question, and whose rest shares no content word with the turn ('Ndaciswa kapati.' > 'Inzya, twalumba.').
    Agreeing by repeating the content ('Sunu kulapya.' > 'Inzya, kulapya kapati.') is natural and passes."""
    m = YES_OPEN.match(reply or "")
    if not m or YESNO_Q.match(user or "") or (user or "").strip().endswith("?"):
        return False
    uw = {w for w in re.findall(r"[a-zŋ']+", (user or "").lower()) if len(w) > 3}
    rw = {w for w in re.findall(r"[a-zŋ']+", m.group(2).lower()) if len(w) > 3}
    return not (uw & rw)


def gate(d, pool_facts, mode, th, en=None, from_memory=False, thought_signals=(), user=None):
    """(decision, why, signals) for the chosen draft. Only adds reasons (E7)."""
    if not d.get("usable"):
        return "teacher", ["no usable draft"], []
    iss = d["issues"]
    if iss["block"]:
        return "block", iss["block"], []
    why = list(iss["hard"]) + list(iss["attest"]) + list(iss["soft"]) + list(iss["drift"])
    sig = list(iss["signals"]) + list(thought_signals)
    m = d.get("meaning") or {}
    spread = pool_facts.get("spread")
    n = pool_facts.get("n_usable") or 1
    if mode == "translate":
        rs = [v for k, v in m.items() if k in ("soup", "s13_s1", "s13_s2")]
        need = 2 if len(rs) >= 3 else 1
        ok_readers = sum(v >= th["thought_min"] for v in rs)
        if ok_readers < need:
            why.append(f"meaning: {ok_readers} of {len(rs)} readers at thought >= {th['thought_min']} "
                       f"({', '.join(f'{k} {v}' for k, v in m.items() if k in ('soup', 's13_s1', 's13_s2'))}; "
                       f"soup read '{(m.get('readers') or {}).get('soup', '')[:60]}')")
        # BRIEF: under 4 English words, or a Tonga line of 1-2 words for a longer English (set D: Kamwaamba kuti for
        # "Say it in another way, please", Ndiimvwi for "I am on my way": the readers fill in the English)
        short = len((en or "").split()) < 4 or len(re.findall(r"[a-zŋ']+", (d.get("text") or "").lower())) <= 2
        if short and not from_memory:
            if (m.get("soup") or 0) < th["short_evidence"]:
                why.append(f"short line: the soup's reading is not a near copy ({m.get('soup')} < {th['short_evidence']}) (F1)")
            if n >= 3 and spread is not None and spread > th["spread_route"]:
                why.append(f"short line: the drafts disagree (spread {spread} > {th['spread_route']})")
            if n >= 3 and (d.get("form_agree") or 0) < th.get("short_form_agree", 1):
                why.append(f"short line: no other draft of the model says it in these words (form agreement "
                           f"{d.get('form_agree') or 0} < {th.get('short_form_agree', 1)})")
            if n < 3:
                sig.append("short line with fewer than 3 drafts: no agreement evidence")
                if th.get("short_needs_pool", True):
                    why.append("short line: no draft-agreement evidence (fewer than 3 drafts)")
        if m.get("certainty") is not None and m["certainty"] < -3.0:
            sig.append(f"translator certainty {m['certainty']} (signal)")
    else:
        if m.get("echo"):
            why.append("the reply copies the user's turn (echo)")
        if user is not None and yes_reflex(user, d.get("text")):
            why.append("yes-reflex: the reply opens with 'yes' to a turn that is no yes/no question, and takes up "
                       "nothing the user said")
        if m.get("relevance") is not None and m["relevance"] < th["relevance_min"]:
            why.append(f"the reply's reading is far from the user's ({m['relevance']} < {th['relevance_min']}: "
                       f"'{m.get('user_back', '')[:40]}' / '{(d['checks']['back']['text'])[:40]}')")
    if spread is not None and n >= 3 and spread > th["spread_route"] and not any("disagree" in w for w in why):
        why.append(f"the drafts disagree (spread {spread} > {th['spread_route']}: confabulation signal)")
    for s in thought_signals:
        why.append("thought: " + s)
    return ("teacher", why, sig) if why else ("send", [], sig)
