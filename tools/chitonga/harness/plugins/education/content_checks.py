"""Content checks on the SERVED English (24 Oct 2026, the Jesuit; persona review REVIEW_PERSONA_2026-10-24.md M2:
"Grade 2 lessons carry Grade 3 to 6 material, and there is one wrong calculation").

Three checks, each used by build_plugin.py and by the tests:
  arith_errors(text)   every written calculation "a + b = c", "a - b = c", "a x b = c", "a ÷ b = c" (also chains
                       "3 + 4 + 5 = 12", and "gives / gets / makes / is c"), with kwacha (K15), units (cm, kg, ml, %)
                       and thousands commas, recomputed. A fraction "1/2 = 2/4" is not a division and is skipped;
                       "07:40" is a time and is skipped. A wrong result is either a FUNDA ERROR (fixed in
                       web/funda_errata.json, applied wherever the guide is read) or an intentional learner mistake
                       the text shows on purpose (INTENTIONAL below, with the reason). An unexplained one fails the build.
  exam_talk(text)      sentences about the ECZ / PSLE / Grade 6 Composite / Grade 7 papers (removed from Grade 1-4
                       material by web/guide.py strip_exam_talk).
  scope_limit(scope)   the largest number a lesson's scope line allows ("adding two 2-digit numbers to 99" -> 99),
                       for the check "the guide's worked example goes beyond this lesson".
"""
import html as H
import re

NUM = r"K?\d{1,3}(?:,\d{3})+(?:\.\d+)?|K?\d+(?:\.\d+)?"
UNIT = r"(?:\s?(?:cm²|cm³|m²|cm|mm|km|kg|ml|g|l|m|%|ngwee|n))?"
OP = r"\s*([+\-−–×x*÷])\s*"
RES = r"\s*(?:=|is|gives|get|gets|makes|equals|→)\s*"
CALC = re.compile(r"(?<![\w.:/,])((?:" + NUM + ")" + UNIT + r"(?:" + OP + r"(?:" + NUM + ")" + UNIT + r")+)" + RES
                  + r"(" + NUM + r")(?![\d/:²³⁴⁵⁶⁷⁸⁹⅓⅔¼½¾⅕⅖⅗⅘⅙⅚⅛])(?!" + UNIT + r"\s*[+\-−–×x*÷]\s*K?\d)")   # not "= 8 + 2 + 3" (a chain)


def _val(s, pct=False):
    s = s.replace("K", "").replace(",", "")
    return float(s) / (100 if pct else 1)


def _calc(expr):
    """the value of 'a op b op c' left to right with × and ÷ first, or None"""
    toks = re.findall(r"(" + NUM + r")(" + UNIT + r")|([+\-−–×x*÷])", expr)
    nums, ops = [], []
    if any(u.strip() == "%" for _, u, _ in toks) and re.search(r"[+\-−–]", expr):
        return None                          # "10% + 5%": a rate, not a sum of amounts
    for n, u, o in toks:
        if n:
            nums.append(_val(n, u.strip() == "%"))
        else:
            ops.append({"−": "-", "–": "-", "x": "*", "×": "*", "÷": "/"}.get(o, o))
    if len(nums) != len(ops) + 1 or not ops:
        return None
    # × and ÷ first
    i = 0
    while i < len(ops):
        if ops[i] in "*/":
            if ops[i] == "/" and nums[i + 1] == 0:
                return None
            nums[i:i + 2] = [nums[i] * nums[i + 1] if ops[i] == "*" else nums[i] / nums[i + 1]]
            ops.pop(i)
        else:
            i += 1
    v = nums[0]
    for o, n in zip(ops, nums[1:]):
        v = v + n if o == "+" else v - n
    return v


def plain(s):
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def arith_errors(text):
    """[(the written calculation, stated result, correct result)] for every calculation that does not hold"""
    out = []
    t = plain(text)
    for m in CALC.finditer(t):
        expr, stated = m.group(1), m.group(m.lastindex)
        if re.search(r"\d\s*/\s*\d", expr):          # a fraction, not a division
            continue
        if re.search(r"(½|¼|¾|\d\s*/\s*\d)\s*[×x*]\s*$", t[:m.start()]):   # "½ × 10 × 6": the half is outside the match
            continue
        if re.search(r"x", expr) and not re.search(r"\d\s*x\s*\d", expr):
            continue
        v = _calc(expr)
        if v is None:
            continue
        sv = _val(stated)
        # a decimal result printed rounded, or a remainder written after the quotient ("847 ÷ 6 = 141 r 1")
        rest = t[m.end():m.end() + 16]
        sp = re.match(r"((?:\s\d)+)", rest)             # a quotient written digit by digit: "824 ÷ 4 = 2 0 6"
        if sp and abs(v - _val(stated + sp.group(1).replace(" ", ""))) < 0.005:
            continue
        if abs(v - sv) < 0.005 or ("÷" in expr and int(v) == sv and re.match(r"(?i)\s*(r|rem|remainder)\s*\d", rest)):
            continue
        if "÷" in expr and abs(round(v, 2) - sv) < 0.011:
            continue
        out.append((m.group(0).strip(), stated, round(v, 2) if v != int(v) else int(v)))
    return out


EXAM = re.compile(r"(?i)\bECZ\b|\bPSLE\b|\bComposite\b|\bG[67]\s+(?:exam|paper)|Grade\s+7\s+(?:exam|paper)|"
                  r"G7 past paper")


def exam_talk(text):
    """the sentences of `text` (plain) that talk about the upper-primary exams"""
    t = plain(text)
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if EXAM.search(s)]


def scope_limit(scope):
    """the largest number the scope line names after 'to' / 'up to' / 'within' / 'sums up to' (99 999 read as one
    number), or None"""
    if not scope:
        return None
    nums = [int(x.replace(" ", "").replace(",", "")) for x in
            re.findall(r"(?i)(?:\bto|\bwithin|\bup to)\s+(\d{1,3}(?:[ ,]\d{3})*|\d+)\b", scope)]
    return max(nums) if nums else None


# learner mistakes the guides show ON PURPOSE (the wrong result is the point): (guide or card id, the calculation as
# found) -> why. Read one by one, 24 Oct 2026.
INTENTIONAL = {
    ("2.2-addition", "48 + 36 = 714"): "common mistake shown on purpose: forgetting to carry",
    ("2.3-subtraction-intro", "2 − 5 = 3"): "common mistake shown on purpose: flipping the digits instead of borrowing",
    ("3.7-money", "K1.50 + K1.50 = K2.100"): "common mistake shown on purpose: forgetting 100 ngwee = K1",
    ("4.1-subtraction-mastery", "0 − 4 = 4"): "common mistake shown on purpose: not borrowing, digits flipped",
}


def unexplained(source_id, text):
    """arith_errors minus the intentional ones of this source (a guide id, or 'lesson:<x>' / 'card:<x>' whose text
    comes from that guide: the guide id is matched inside the source id)"""
    out = []
    for e in arith_errors(text):
        if not any(e[0] == calc and gid in source_id for (gid, calc) in INTENTIONAL):
            out.append(e)
    return out
