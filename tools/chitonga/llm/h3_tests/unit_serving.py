"""Harness v3 step 0, the serving changes (19 Oct 2026): thought split, answer-only checks, word-check note format,
ask-back acceptance. Needs no server (the forced-thought test at the end runs only if the profile's server is up).
    python -X utf8 tools/chitonga/llm/h3_tests/unit_serving.py [profile]"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import h3_profile as P, h3_llm as L, h3_wordcheck as W
ok, bad = 0, []
def t(name, got, want):
    global ok
    if got == want: ok += 1
    else: bad.append((name, got, want))
p = P.load(sys.argv[1] if len(sys.argv) > 1 else "v10r3_s1_thinkon")
# 1 thought split (Gemma 4 format as trained: <|channel>thought\n...\n<channel|>answer<turn|>)
t("split closed", L.split(p, "<|channel>thought\nThe user asks X.\n<channel|>Inzya, ndazitebula.<turn|>"),
  ("The user asks X.", "Inzya, ndazitebula.", True))
t("split none", L.split(p, "Mvwula yatalika kale.<turn|>"), ("", "Mvwula yatalika kale.", None))
t("split never closed", L.split(p, "<|channel>thought\nThe user asks X and then"), ("The user asks X and then", "", False))
t("split keeps answer newlines", L.split(p, "<|channel>thought\nA\n<channel|>Line one.\nLine two.")[1], "Line one.\nLine two.")
# 2 the note is joined as the training rows join it (build_reasoning.ask_rows: content + "\n\n" + note)
rows = [json.loads(l) for l in open(os.path.join(P.REPO, "internal", "chitonga_train", "llm", "reasoning", "ask.jsonl"), encoding="utf-8")]
kn = next(r for r in rows if r["src"].startswith("reasoning:ask:known"))
u = kn["messages"][-2]["content"]
base, note = u.rsplit("\n\n", 1)
t("note format known", W.with_note(base, W.check(base)["note"]) if W.check(base)["all_known"] else u, u)
ab = next(r for r in rows if r["src"] == "reasoning:ask:askback")
ub = ab["messages"][-2]["content"]
bb, nb = ub.rsplit("\n\n", 1)
t("note format askback", W.with_note(bb, W.check(bb)["note"]), ub)
# 3 ask-back acceptance (the harness decides; DECISIONS 19 Oct)
wk = {"note": "[all words known]", "flagged": [], "all_known": True}
wu = {"note": "[unknown word: 'amenshi']", "flagged": ["amenshi"], "all_known": False}  # a hand-built check result
t("askback known rejected", W.askback_justified("Nsemvwide 'kulapya'. Mwati nzi?", wk), False)
t("askback flagged accepted", W.askback_justified("Nsemvwide 'amenshi'. Mwati nzi?", wu), True)
t("askback wrong word rejected", W.askback_justified("Nsemvwide 'muulisya'. Mwati nzi?", wu), False)
t("askback unquoted accepted", W.askback_justified("Tiindamvwa. Mwati nzi?", wu), True)
t("not an askback", W.is_askback("Nsekainki kumushika juunza."), False)
t("Nsemvwide is askback", W.is_askback("Nsemvwide 'x'. Mwati nzi?"), True)
# 4 profile: run 3 off, the smoke profile on; the run 4 pattern refuses to serve
t("run3 think off", P.load("v10r3_s1")["template"]["think"], False)
t("smoke think on", p["template"]["think"], True)
try:
    P.load("run4_template"); t("template refuses", False, True)
except ValueError:
    t("template refuses", True, True)
t("inherit thresholds", P.load("v10O_s1")["thresholds"]["thought_min"], 0.70)
# 5 live: a forced thought on the real server (run 3 does not open one by itself, RUN3_ANALYSIS item 9)
if L.up(p):
    ms = [{"role": "user", "content": "Translate into Chitonga (the Tonga language of southern Zambia). Reply with the translation only.\n\nMy sister is sick."}]
    pr = L.render(p, ms)
    t("switch in prompt", "<|think|>" in pr, True)
    t("switch absent when off", "<|think|>" in L.render(P.load("v10r3_s1"), ms), False)
    r = L._post(p, "/completion", {"prompt": pr + "<|channel>thought\n", "n_predict": 600, "temperature": 0})
    th, ans, closed = L.split(p, "<|channel>thought\n" + r["content"])
    print("forced thought:", repr(th[:160]), "| answer:", repr(ans[:80]), "| closed:", closed)
    t("forced thought parsed", bool(th), True)
print(f"{ok} passed, {len(bad)} failed")
for b in bad: print("FAIL", b)
