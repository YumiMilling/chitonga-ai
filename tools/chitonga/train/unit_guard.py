"""Tests for the 21 Oct 2026 guards (audit decisions steps 1-3): heldout_guard's relabelled-repeat check, the anchor
files in the guard, mix_groups (every group named, hopgood drawn, no Bible English pairs) and the same rules inside
train_nllb.py. No model, no pod.   python -X utf8 tools/chitonga/train/unit_guard.py"""
import ast
import json
import os
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heldout_guard as HG  # noqa: E402
import mix_groups as X  # noqa: E402
from h2_memory import held_norm  # noqa: E402  (heldout_guard put llm/ on the path)

OK, BAD = [0], []


def t(name, got, want=True):
    if got == want:
        OK[0] += 1
    else:
        BAD.append(name)
        print("FAIL", name, "| got", repr(got)[:200], "| want", repr(want)[:200])


# --- held_norm: one identity whatever the label
same = ["B Having sex with two people only. (1)", "a) Having sex with two people only.",
        "Question 3: Having sex with two people only [2]", "(iv) Having sex with two people only (2 marks)",
        "1.2 Having sex with two people only", "3 Having sex with two people only", "• Having sex with two people only"]
for s in same:
    t("held_norm label: " + s, held_norm(s), "having sex with two people only")
t("blank of any length", held_norm("3 ________ the war began."), held_norm("3. ____ the war began."))
t("Tonga 'Ii' (yes) is not a roman numeral", held_norm("Ii, ndalibizya."), "ii ndalibizya")
t("'I am' keeps I", held_norm("I am going."), "i am going")
t("Tonga maaka mark", held_norm("Alfa beta gamma. (maaka 3)"), held_norm("Alfa beta gamma."))

# --- Guard: exact, relabelled, short lines, 6-gram
G = HG.Guard()
G.add("exam", "B Having sex with two people only.", grams=False)
G.add("exam toi", "Alfa beta gamma delta epsilon zeta eta theta iota.")   # PUBLIC COPY: an exam line in the working copy
G.add("short", "B Yes.", grams=False)
t("exact hit", G.which(en="B Having sex with two people only."), "exam")
t("relabelled hit", G.which(en="a) Having sex with two people only."), "relabelled: exam")
t("relabelled hit via hit()", G.hit(en="(ii) having sex with two people only"), True)
t("different line clean", G.hit(en="Having sex with one person only."), False)
t("1-token key not made (Yes stays trainable)", G.hit(en="Yes."), False)
t("6-gram still works", G.which(toi="Omega psi alfa beta gamma delta epsilon zeta eta."), "6-gram")
t("rsets beside sets", set(G.rsets) <= set(G.sets), True)

# --- load(): the real frozen sets plus the anchor files
GL = HG.load(os.path.join(HG.CT, "runs", "2026-10-18_stage14"))
v1 = [json.loads(l) for l in open(os.path.join(HG.CT, "llm", "panel", "translator_anchor_candidates.jsonl"), encoding="utf-8")]
t("every v1 anchor candidate is guarded", all(GL.hit(r["source"] if r["dir"] == "toi>en" else None,
                                                    r["source"] if r["dir"] == "en>toi" else None) for r in v1), True)
t("anchor set named", any(k.startswith("translator anchor: ") for k in GL.sets), True)
t("the 20 Oct relabelled exam line is caught",
  GL.which(en="a) Having sex with two people only.") not in (None, "6-gram"), True)

# --- mix_groups
tr = [{"id": "a", "origin": "smol", "toi": "Ndili kabotu.", "en": "I am well."},
      {"id": "h", "origin": "hopgood:drill", "toi": "Ndabona.", "en": "I see."},
      {"id": "g", "origin": "synthetic:grammar", "toi": "x y", "en": "x y"}]
mono = [{"id": "m", "origin": "liseli:bible (Tonga only)", "toi": "Mubukulo Leza wakalenga julu anyika yoonse"}]
rep = X.check(tr, "own:0.9,hopgood:0.05,grammar:0,copy:0.05", mono=mono)
t("named mix passes; grammar weight 0 visible", rep["zero_weight"], ["grammar"])


def fails(fn, *a, **k):
    try:
        fn(*a, **k)
        return None
    except (AssertionError, X.MixError) as e:
        return str(e)


t("unnamed group fails", "not named" in (fails(X.check, tr, "own:0.9,hopgood:0.1", mono=mono) or ""), True)
t("hopgood weight 0 fails", "must be drawn" in (fails(X.check, tr, "own:0.9,hopgood:0,grammar:0,copy:0.1", mono=mono) or ""), True)
bl = tr + [{"id": "biblelit:x", "origin": "synthetic:bible_lit", "toi": "q", "en": "q"}]
t("bible_lit group fails", "monolingual" in (fails(X.check, bl, "own:0.8,hopgood:0.1,grammar:0,bible_lit:0.05,copy:0.05",
                                                   mono=mono) or ""), True)
sneak = tr + [{"id": "t9", "origin": "teacher:T09", "toi": "Mubukulo Leza wakalenga julu anyika yoonse.", "en": "In the beginning"}]
t("Bible verse under another origin fails (content)", "content" in (fails(X.check, sneak, "own:0.9,hopgood:0.05,grammar:0,copy:0.05",
                                                                          mono=mono) or ""), True)
task = [{"id": "bt1", "pool": "bt", "src_lang": "eng", "tgt_lang": "toi", "src": "[BT] In the beginning",
         "tgt": "Mubukulo Leza wakalenga julu anyika yoonse", "origin": "bt_mono_soup:zambezi_voice"}]
t("Bible verse as a BT task fails", "content" in (fails(X.check, tr, "own:0.8,hopgood:0.05,grammar:0,copy:0.05,bt:0.1",
                                                        tasks=task, mono=mono) or ""), True)
t("Bible Tonga in the copy pool is allowed", fails(X.check, tr, "own:0.9,hopgood:0.05,grammar:0,copy:0.05", mono=mono), None)
t("stage 14 as built would fail now", fails(X.check_build, os.path.join(HG.CT, "runs", "2026-10-18_stage14"),
  "own:0.068,book_high:0.642,book_med:0.04,copy:0.06,story:0.05,bt:0.04,twins:0.05,lex:0.03,bible_lit:0.02") is not None, True)
t("hopgood counts", X.hopgood_counts(os.path.join(HG.CT, "runs", "2026-10-18_stage14")),
  {"hopgood:drill": 708, "hopgood:exercise": 594})

# --- train_nllb.py: group() identical, and its own copy of the rules behaves the same
src = open(os.path.join(HERE, "train_nllb.py"), encoding="utf-8").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "group")
ns = {}
exec(ast.get_source_segment(src, fn), ns)
for o in ("smol", "hopgood:drill", "liseli:bible", "synthetic:bible_lit", "asb:storybook", "jctr:homily", "teacher:T01"):
    t("group() identical: " + o, ns["group"](o), X.group(o))
blk = src[src.index("    if not a.legacy_mix:"):src.index("else:\n    if any(group(")]
code = textwrap.dedent(blk)


def run_rules(pools, weights, legacy=False):
    env = {"a": types.SimpleNamespace(legacy_mix=legacy, concat=False), "pools": pools, "weights": weights}
    try:
        exec(code, env)
        return None
    except (AssertionError, SystemExit) as e:
        return str(e)


P = {"own": tr[:1], "hopgood": tr[1:2], "copy": mono}
t("pod: named mix passes", run_rules(P, {"own": 0.9, "hopgood": 0.05, "copy": 0.05}), None)
t("pod: unnamed fails", "not named" in (run_rules(P, {"own": 0.95, "hopgood": 0.05}) or ""), True)
t("pod: hopgood 0 fails", "hopgood" in (run_rules(P, {"own": 0.95, "hopgood": 0, "copy": 0.05}) or ""), True)
t("pod: bible_lit fails", "Bible" in (run_rules({**P, "bible_lit": [{"id": "b", "origin": "synthetic:bible_lit", "toi": "q"}]},
                                                {"own": 0.9, "hopgood": 0.05, "copy": 0.03, "bible_lit": 0.02}) or ""), True)
t("pod: verse content fails", "Bible" in (run_rules({**P, "own": sneak[-1:]}, {"own": 0.9, "hopgood": 0.05, "copy": 0.05}) or ""), True)
t("pod: --legacy-mix skips the rules", run_rules(P, {"own": 1.0}, legacy=True), None)

print(f"{OK[0]} passed, {len(BAD)} failed")
sys.exit(1 if BAD else 0)
