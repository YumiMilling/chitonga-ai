"""The struggle-to-mistake matcher, measured (23 Oct 2026, the Jesuit; stream A, harness flaw (i)).

A sample run matched "most of my class cannot count past 20" to the wrong common mistake ("One touch, two numbers"): the
matcher counted shared words ("count") between the teacher's line and the Funda guide's three mistakes, and the guides
have no mistake for counting range, skip counting or zero at all. This file is the yardstick for the fix: 10 realistic
class-struggle phrasings per maths lesson (70), each with the mistake or struggle topic it should reach. Written by me,
as a Zambian Grade 1 teacher would type them on WhatsApp (plain, short, some with numbers); NOT teacher-written, not a
Tonga reference, never training data. A 'hit' = the reply's "Likely: ..." is the expected mistake / topic; a 'lesson
hit' = the right lesson at least.

Path measured = what the engine does: the rules reader picks the unit from the text (intake keywords), then
Engine._match_mistake(text, unit). No memory of a current lesson (the hardest case: the teacher just types).

    python -X utf8 tools/chitonga/harness/plugins/education/struggle_eval.py [--show]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

CASES = [  # (text, lesson, accepted mistake / topic ids)
    # 2.1 counting
    ("Most of my class cannot count past 20", "2.1-counting", ["t-range"]),
    ("they get stuck after 29, they don't know what comes next", "2.1-counting", ["t-range"]),
    ("my learners can count to 10 but not to 100", "2.1-counting", ["t-range"]),
    ("they skip numbers when counting, like 14, 16", "2.1-counting", ["m1"]),
    ("learners forget 15 and 16 when they count", "2.1-counting", ["m1"]),
    ("they count the same stone twice", "2.1-counting", ["m2"]),
    ("they touch one stone and say two numbers", "2.1-counting", ["m2"]),
    ("when I ask how many, they start counting again from one", "2.1-counting", ["m3"]),
    ("the class cannot count in 2s, 5s and 10s", "2.1-counting", ["t-skip"]),
    ("they write 3 and 7 backwards", "2.1-counting", ["t-write"]),
    # 2.2 addition
    ("they always start counting from one again instead of counting on", "2.2-addition", ["t-counton"]),
    ("my pupils cannot add without their fingers", "2.2-addition", ["t-fingers"]),
    ("they don't understand the plus sign and the equals sign", "2.2-addition", ["t-signs"]),
    ("learners struggle with story sums", "2.2-addition", ["t-story"]),
    ("they don't know the number bonds to 10", "2.2-addition", ["t-bonds"]),
    ("when they add 48 and 36 they write 714", "2.2-addition", ["m1"]),
    ("they forget to carry the ten", "2.2-addition", ["m1"]),
    ("they add the carried one twice", "2.2-addition", ["m2"]),
    ("they write the numbers in the wrong columns when adding", "2.2-addition", ["m3"]),
    ("pupils can't add in their heads, they need the stones every time", "2.2-addition", ["t-fingers"]),
    # 2.3 subtraction
    ("for 52 - 7 they do 7 - 2, they take the small number from the big one", "2.3-subtraction-intro", ["m1"]),
    ("they don't know how to borrow", "2.3-subtraction-intro", ["m1"]),
    ("they borrow but forget to cross out the ten", "2.3-subtraction-intro", ["m2"]),
    ("they don't line up the numbers when subtracting", "2.3-subtraction-intro", ["m3"]),
    ("my class mixes up plus and minus", "2.3-subtraction-intro", ["t-signs"]),
    ("they cannot count backwards from 10", "2.3-subtraction-intro", ["t-countback"]),
    ("learners don't understand 'how many more' questions", "2.3-subtraction-intro", ["t-compare"]),
    ("word problems with take away confuse them", "2.3-subtraction-intro", ["t-story"]),
    ("they add instead of subtracting", "2.3-subtraction-intro", ["t-signs"]),
    ("they don't understand what is left after taking away", "2.3-subtraction-intro", ["t-takeaway"]),
    # 3.1 place value
    ("they read 34 as three four", "3.1-place-value-intro", ["m1"]),
    ("they write 205 as 25", "3.1-place-value-intro", ["m2"]),
    ("they think the 4 in 48 is just four", "3.1-place-value-intro", ["m3"]),
    ("they mix up 12 and 21", "3.1-place-value-intro", ["t-reverse"]),
    ("my class can't make bundles of ten", "3.1-place-value-intro", ["t-bundle"]),
    ("they don't understand tens and ones", "3.1-place-value-intro", ["m3"]),
    ("they forget the zero in 106", "3.1-place-value-intro", ["m2"]),
    ("they count a bundle as one, not as ten", "3.1-place-value-intro", ["t-bundle"]),
    ("they write sixty-one when I say sixteen", "3.1-place-value-intro", ["t-reverse"]),
    ("learners don't understand place value at all", "3.1-place-value-intro", ["m1", "m3"]),
    # 3.2 comparing and ordering
    ("they say 58 is bigger than 85", "3.2-comparing-ordering", ["m1"]),
    ("they use the greater than sign the wrong way", "3.2-comparing-ordering", ["m2"]),
    ("they say 27 and 24 are the same because both start with 2", "3.2-comparing-ordering", ["m3"]),
    ("they don't understand more and less", "3.2-comparing-ordering", ["t-moreless"]),
    ("they cannot put numbers in order from smallest to biggest", "3.2-comparing-ordering", ["t-order"]),
    ("ordering numbers is hard for my class", "3.2-comparing-ordering", ["t-order"]),
    ("they think the pile that is spread out has more", "3.2-comparing-ordering", ["t-moreless"]),
    ("the alligator mouth confuses them", "3.2-comparing-ordering", ["m2"]),
    ("they look at the last digit to decide which number is bigger", "3.2-comparing-ordering", ["m1"]),
    ("they don't know what 'the same as' means", "3.2-comparing-ordering", ["t-same"]),
    # 3.7 money
    ("they can't give change", "3.7-money", ["m2"]),
    ("they think K1.50 + K1.50 is K2.100", "3.7-money", ["m1"]),
    ("they don't know 100 ngwee is one kwacha", "3.7-money", ["m1"]),
    ("they mix up the K5 and K10 notes", "3.7-money", ["m3"]),
    ("they cannot recognise the coins and notes", "3.7-money", ["m3"]),
    ("working out change from K20 is too hard for them", "3.7-money", ["m2"]),
    ("they count down instead of counting up for change", "3.7-money", ["m2"]),
    ("they forget to write the K", "3.7-money", ["t-k"]),
    ("they cannot add up the prices of two things", "3.7-money", ["t-total"]),
    ("they confuse ngwee and kwacha", "3.7-money", ["m1"]),
    # 3.8 shapes
    ("they say a square is not a rectangle", "3.8-shapes-2d", ["m1"]),
    ("they mix up sides and lines", "3.8-shapes-2d", ["m2"]),
    ("they only fold the paper top to bottom for symmetry", "3.8-shapes-2d", ["m3"]),
    ("they don't know the names of the shapes", "3.8-shapes-2d", ["t-names"]),
    ("they think a triangle upside down is not a triangle", "3.8-shapes-2d", ["t-rotated"]),
    ("they can't count the sides of a shape", "3.8-shapes-2d", ["t-sides"]),
    ("they call every round thing a circle", "3.8-shapes-2d", ["t-names"]),
    ("they guess the name of the shape without looking", "3.8-shapes-2d", ["t-sides"]),
    ("learners confuse the rectangle and the square", "3.8-shapes-2d", ["m1"]),
    ("they don't understand symmetry", "3.8-shapes-2d", ["m3"]),
]
# A SECOND SET, written BEFORE the struggle patterns (baseline 2/21). NOT a clean held-out: it was in view while I wrote
# the patterns (e.g. 'diamond', 'by fives', 'give back' are in them), so its score is optimistic too. The clean test is
# struggle lines written by teachers (proposed for the next desk batch); until then read both numbers as upper bounds.
HELDOUT = [
    ("after 39 they say 20 again", "2.1-counting", ["t-range"]),
    ("counting by fives is a problem", "2.1-counting", ["t-skip"]),
    ("they say the numbers but their finger is not on the stones", "2.1-counting", ["m2"]),
    ("they use only fingers, they can't do 6 + 3 in the head", "2.2-addition", ["t-fingers"]),
    ("the children start from 1 every time they add", "2.2-addition", ["t-counton"]),
    ("they don't know which pairs make 10", "2.2-addition", ["t-bonds"]),
    ("when subtracting they always take the smaller digit from the bigger one", "2.3-subtraction-intro", ["m1"]),
    ("counting back is hard for them", "2.3-subtraction-intro", ["t-countback"]),
    ("they add when the question says take away", "2.3-subtraction-intro", ["t-signs"]),
    ("they reverse the digits, 13 becomes 31", "3.1-place-value-intro", ["t-reverse"]),
    ("they leave out the zero when writing 302", "3.1-place-value-intro", ["m2"]),
    ("they think a bundle of sticks is one", "3.1-place-value-intro", ["t-bundle"]),
    ("they cannot tell which number is less", "3.2-comparing-ordering", ["t-moreless"]),
    ("arranging numbers from the biggest to smallest", "3.2-comparing-ordering", ["t-order"]),
    ("they get the < and > signs mixed up", "3.2-comparing-ordering", ["m2"]),
    ("when buying they don't know how much to give back", "3.7-money", ["m2"]),
    ("they don't know the notes", "3.7-money", ["m3"]),
    ("they write 50 + 70 ngwee as 120 ngwee", "3.7-money", ["m1"]),
    ("they cannot name a triangle and a circle", "3.8-shapes-2d", ["t-names"]),
    ("if I turn the square they say it is a diamond", "3.8-shapes-2d", ["t-rotated"]),
    ("they think the fold line is a side", "3.8-shapes-2d", ["m2"]),
]


def run(show=False, cases=None, label="dev"):
    from core.engine import Engine  # noqa: PLC0415
    from core.log import TurnLog  # noqa: PLC0415
    from core.memory import SQLiteMemory  # noqa: PLC0415
    from core.retrieval import Index  # noqa: PLC0415
    from plugins import education  # noqa: PLC0415
    P, hooks = education.load()
    e = Engine(P, SQLiteMemory(":memory:"), hooks=hooks, log=TurnLog(None))
    idx = Index(P)
    per, hits, lhits, rows = {}, 0, 0, []
    cases = CASES if cases is None else cases
    for text, lesson, ok in cases:
        r = e.reader.read(text, {"plugin": P, "index": idx, "recall": None, "grades": [1], "number": "x"})
        s, m, _sc = e._match_mistake(text, r.get("unit"))
        got_lesson = s["lesson"].split(":", 1)[1] if s else None
        got = m["id"].split("#", 1)[1] if m else None
        hit = got_lesson == lesson and got in ok
        lh = got_lesson == lesson
        hits += hit
        lhits += lh
        p = per.setdefault(lesson, [0, 0, 0])
        p[0] += hit
        p[1] += lh
        p[2] += 1
        rows.append((hit, text, r.get("unit"), f"{got_lesson}#{got}", f"{lesson}#{'/'.join(ok)}",
                     m["title"] if m else "-"))
    if show:
        for row in rows:
            print(("HIT " if row[0] else "miss"), " | ".join(str(x) for x in row[1:]))
    for k, (h, lh_, n) in per.items():
        print(f"{k:26s} {h}/{n} (lesson {lh_}/{n})")
    print(f"TOTAL {label}: hits {hits}/{len(cases)}, lesson {lhits}/{len(cases)}")
    return hits, lhits, len(cases)


if __name__ == "__main__":
    run("--show" in sys.argv)
    run("--show" in sys.argv, HELDOUT, "second set (seen while writing patterns)")
