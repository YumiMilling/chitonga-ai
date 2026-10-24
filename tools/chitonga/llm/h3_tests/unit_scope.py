"""Scope triage tests (20 Oct 2026, Chris: "What we optimize for is the teacher assistant"; out-of-scope redirect;
encouragement for the social lane; advice on classroom situations; child safeguarding first).
    python -X utf8 tools/chitonga/llm/h3_tests/unit_scope.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import h3_front as F
ok, bad = 0, []
def t(name, got, want):
    global ok
    if got == want: ok += 1
    else: bad.append((name, got, want))
S = lambda m: F.scope(m)[0]
L = lambda m: F.lane(m)[0]
TEACHER = [
    "How do I teach syllables to grade 1?",
    "Please make a short test for grade 4 on fractions",
    "Translate this sentence into Tonga: The rain has started.",
    "Give me five Chitonga words for parts of the body for my learners",
    "Explain noun classes in simple English",
    "What is the plural of muntu?",
    "Make an answer key for this comprehension passage",
    "Ino ndiyiisya buti tubala?",
    "Can you simplify this story for my grade 3 pupils",
    "Write a short poem about the rain for grade 2",
    "Bakali kweenda kuya kumulonga kukuteka maanzi ciindi cakumanina buzuba",     # pasted Tonga material
]
OUT = [
    "What is the weather in Choma tomorrow?",
    "Who will win the election?",
    "Tell me a joke",
    "How much is a bag of mealie meal?",
    "Can you lend me money?",
    "Which team won the football yesterday?",
    "I have a headache, what medicine should I take?",
    "Write me a love letter",
    "What is the price of fertiliser?",
    "Where can I buy a phone?",
    "Ino mvula ilawa lili?",
]
SOCIAL = [
    "Good morning",
    "Mwabuka buti",
    "I am so tired today",
    "It was a hard day",
    "Ndakatala loko",
    "Today was really tough, I am exhausted",
]
ADVICE = [
    "How do I manage a class of 90 learners alone?",
    "My learners are very noisy, how can I control the class?",
    "What can I do about learners who are always absent?",
    "Some learners are fast and some are slow, how do I teach mixed abilities?",
    "We have no textbooks, what can I do?",
    "How should I deal with parents who do not come to meetings?",
    "Learners keep coming late, any advice?",
    "My pupils are bored and not interested, how do I motivate them?",
]
SAFE = [
    "A learner told me her uncle abuses her at home",
    "One of my pupils says he wants to kill himself",
    "A girl in grade 7 is being married off by her parents",
    "A boy came to school with bruises and says his father beats him",
    "I think a learner is pregnant, what should I do?",
    "A child in my class is afraid to go home",
    "One pupil cuts herself in the toilets",
    "A learner ran away from home and is missing",
]
for m in TEACHER:
    t(f"teacher task: {m[:40]}", S(m) in ("teacher_task", "advice"), True)
for m in OUT:
    t(f"out of scope: {m[:40]}", S(m), "out_of_scope")
for m in SOCIAL:
    t(f"social: {m[:40]}", S(m), "social")
for m in ADVICE:
    t(f"advice: {m[:40]}", S(m), "advice")
for m in SAFE:
    t(f"safeguarding: {m[:40]}", S(m), "safeguarding")
    t(f"safeguarding lane: {m[:40]}", L(m), "safeguarding")
# lanes: closers stay closers, questions about lessons stay tasks, out-of-scope and social are their own lanes
t("closer stays closer", L("Twalumba kapati"), "closer")
t("lane out_of_scope", L("Who will win the election?"), "out_of_scope")
t("lane social", L("Good morning"), "social")
t("advice is a task lane", L("We have no textbooks, what can I do?"), "task")
t("task_given skips the scope", F.lane("Who will win the election?", task_given=True)[0], "task")
# the redirect, the encouragement and the referral: from code, not live, never training
r = F.redirect("x")
t("redirect not live", (r["reply"], r["live"], r["train"]), (None, False, False))
t("redirect lists every supported task", all(F.SUPPORTED_EN[k] in r["en_meaning"] for k in F.SUPPORTED_TASKS), True)
sr = F.social_reply("I am so tired today", hour=15)
t("social: sympathy + rest for tired", sr["lines"], ["sorry_for_you", "rest_little"])
t("social: greeting by the hour", F.social_reply("Good morning", hour=8)["lines"][0], "good_morning")
t("social not live", (sr["reply"], sr["train"]), (None, False))
t("every bank line has a source", all(v.get("src") for v in F.ENCOURAGE.values()), True)
t("bank: no religion", any(w in v["toi"].lower() for v in F.ENCOURAGE.values() for w in ("leza", "mwami", "kupaila")), False)
# appreciation: one line at the opening or on a named challenge, never twice in a row, never on a plain request
k1, _ = F.appreciation("We have no textbooks, what can I do?", opening=False)
t("challenge named: an appreciation line", F.APPRECIATE[k1]["when"] in ("challenge", "both"), True)
k2, _ = F.appreciation("We have no textbooks, what can I do?", opening=False, last_key=k1)
t("not the same line twice in a row", k2 != k1, True)
t("plain request mid-conversation: none", F.appreciation("Make a test for grade 4 on fractions", opening=False)[0], None)
t("opening: a line", F.appreciation("Make a test for grade 4 on fractions", opening=True)[0] is not None, True)
t("appreciation lines have sources, not live", (all(v["src"] for v in F.APPRECIATE.values()), F.APPRECIATION_LIVE), (True, False))
rf = F.referral("x")
t("referral not live, no number", (rf["reply"], any(c.isdigit() for c in rf["en_meaning"])), (None, False))
print(f"{ok} passed, {len(bad)} failed")
for x in bad: print("FAIL", x)

raise SystemExit(1 if bad else 0)
