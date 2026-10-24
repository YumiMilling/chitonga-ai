"""Front triage + teacher memory tests (20 Oct 2026).  python -X utf8 tools/chitonga/llm/h3_tests/unit_front.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import h3_front as F, h3_teachers as TM
ok, bad = 0, []
def t(name, got, want):
    global ok
    if got == want: ok += 1
    else: bad.append((name, got, want))
L = lambda m, **k: F.lane(m, **k)[0]
t("Twalumba closer", L("Twalumba kapati"), "closer")
t("Ndalumba closer", L("Ndalumba."), "closer")
t("ok thanks closer", L("Ok thank you so much"), "closer")
t("deferral closer", L("I will try it tomorrow and get back to you"), "closer")
t("curly apostrophe deferral", L("Okay I’ll let you know"), "closer")
t("question is a task", L("How do I teach syllables?"), "task")
t("Tonga question is a task", L("Ino ndiyiisya buti tubala?"), "task")
t("request is a task", L("Please make a short test for grade 4 on fractions"), "task")
t("thanks + request is a task", L("Thanks, now explain noun classes"), "task")
t("media", L("", media=True), "media")
t("profile answer", L("grade 3, 60 pupils", screening_pending=True), "profile")
t("memory view", L("What do you remember about me"), "memory")
t("memory wipe", L("Please delete my data"), "memory")
t("closers are printed Tonga only", all(F.toi_closers() <= set(F.TOI_CANDIDATES) for _ in [0]), True)
t("twalumba is printed", "twalumba" in F.toi_closers(), True)
t("feedback seen", F.has_feedback("It worked, thank you"), True)
# memory: scrubbing, profile history, failed interventions, view / wipe
ph = "+260 999 000 111"
TM.wipe(ph)
t("scrub child name", TM.scrub("My learner Mutinta is 8 years old and cannot read"), "My learner [name] is [age] and cannot read")
TM.set_profile(ph, {"grades": [3]})
TM.set_profile(ph, {"grades": [3, 4]}, source="correction")
t("profile history", len(TM.view(ph)["profile_history"]), 2)
TM.add_difficulty(ph, "learners confuse b and d", intervention="letter cards b / d, chalkboard drill")
TM.add_feedback(ph, "it did not work, too long")
t("failed intervention remembered", TM.failed_interventions(ph), ["letter cards b / d, chalkboard drill"])
t("file name is not the number", "999" not in os.path.basename(TM._path(ph)), True)
TM.wipe(ph)
t("wiped", TM.view(ph)["profile"], None)
print(f"{ok} passed, {len(bad)} failed")
for x in bad: print("FAIL", x)
