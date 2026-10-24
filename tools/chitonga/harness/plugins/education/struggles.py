"""Class struggles for the maths lessons: what a teacher types, and where it should land (23 Oct 2026, the Jesuit;
stream A, harness flaw (i)). Data for build_plugin.py; the engine only runs the regexes (core/engine.py _match_mistake).

Two kinds of entry per lesson:
  PATTERNS[lesson][m1|m2|m3]  regexes (over the teacher's line, lower case) that point at the Funda guide's own common
                              mistakes. Weight 2 by default; 3 where the line is unmistakable.
  TOPICS[lesson]              struggles the Funda guide has NO mistake for, though the Grade 1 syllabus covers them
                              (counting past 20, skip counting, zero, counting on, number bonds, ...). Each carries a
                              short diagnosis and one 'Say' line. origin 'F' when the Say line is the guide's own teacher
                              move (the move named in 'from'); origin 'C' when the diagnosis and the Say line are mine
                              (Claude-drafted English, not in the guide, not reviewed by a teacher: the reply says so).
                              'tonga_en' attaches a Tonga line ONLY when the lesson's box has that very English line
                              (teacher / book / derived, with its origin); 'tonga' gives teacher key words verbatim.
Counters: the lessons now use stones (and sticks for bundles), so the Say lines say stones / sticks.
Measured by struggle_eval.py (70 dev phrasings + 21 held out, written before these patterns).
"""
import re

N = r"(?:\d+|ten|twenty|thirty|forty|fifty|hundred|one hundred)"

PATTERNS = {
    "2.1-counting": {
        "m1": (2, [r"skip(?:s|ping|ped)?\s+(?:a\s+|some\s+|the\s+)?(?:numbers?|\d+)", r"miss(?:es|ing)?\s+(?:out\s+)?(?:a\s+|some\s+)?(?:numbers?|\d+)",
                   r"forget\w*\s+(?:\d+|some numbers|numbers)", r"leave\w*\s+out\s+(?:a\s+|some\s+)?numbers?", r"\bteens?\b|teen numbers"]),
        "m2": (2, [r"(?:same|one)\s+(?:stone|object|thing|cap|counter|bottle top|item)s?\s+twice", r"count\w*\s+(?:the same \w+|it|them|things|objects|stones)\s+twice",
                   r"touch\w*.{0,30}\b(?:two|2)\s+numbers", r"\bone\s+touch", r"lose\s+(?:track|their place)", r"double[- ]count",
                   r"finger\w*\s+(?:is\s+)?not\s+on", r"(?:numbers?|counting)\b.{0,30}\b(?:faster|too fast|ahead) (?:than|of) (?:the|their) finger",
                   r"say\w*\s+the\s+numbers?\s+but"]),
        "m3": (2, [r"how many.{0,40}\b(?:again|start|recount|from one)", r"(?:re-?count|count again).{0,40}(?:how many|from one)", r"last number", r"cardinal"]),
    },
    "2.2-addition": {
        "m1": (2, [r"\bcarry(?:ing)?\b|\bcarried\b(?!.{0,15}twice)", r"\b\d{3}\b.{0,20}(?:answer|instead)|write\w*\s+\d{3}\b", r"regroup"]),
        "m2": (3, [r"carr\w*.{0,25}\btwice\b", r"add\w*\s+the\s+(?:carried|carry)\b.{0,20}(?:twice|again)"]),
        # the column mistake is in both lessons: each claims it only when the line does not name the other operation
        "m3": (2, [r"^(?!.*(?:subtract|take away|taking away|minus|borrow)).*(?:column|line\s+(?:them\s+|the\s+numbers\s+)?up|\bunder\s+the\b)"]),
    },
    "2.3-subtraction-intro": {
        "m1": (2, [r"\bborrow", r"\bflip", r"(?:take|subtract)\w*\s+the\s+small\w*.{0,25}(?:from|than)\s+the\s+(?:big|large)",
                   r"small(?:er)?\s+(?:digit|number)?\s*from\s+the\s+big", r"\b\d+\s*-\s*\d+\b.{0,30}\b(?:they|he|she)\s+(?:do|write|get)"]),
        "m2": (3, [r"cross\w*\s+out\s+the\s+ten", r"borrow\w*\s+but", r"forget\w*\s+to\s+(?:cross|reduce|take\s+one)"]),
        "m3": (2, [r"^(?!.*(?:\badd|plus|carry)).*(?:column|line\s+(?:them\s+|the\s+numbers\s+)?up)"]),
    },
    "3.1-place-value-intro": {
        "m1": (2, [r"read\w*\s+\d{2,3}\s+as\s+\w+\s+\w+", r"separate\s+(?:numbers|digits)", r"one\s+digit\s+at\s+a\s+time",
                   r"left\s+to\s+right", r"place\s+value"]),
        "m2": (3, [r"zero.{0,30}\b\d0\d\b|\b\d0\d\b.{0,30}zero", r"write\w*\s+\d0\d\s+as\s+\d\d\b", r"(?:leave|leaves|left)\s+out\s+the\s+zero",
                   r"(?:forget|skip|miss|drop)\w*\s+the\s+zero", r"\bzero\b.{0,20}\bplace"]),
        "m3": (2, [r"\bthe\s+\d\s+in\s+\d{2,3}\b", r"tens\s+and\s+ones|ones\s+and\s+tens", r"value\s+of\s+(?:the\s+|a\s+)?digit",
                   r"place\s+value"]),
    },
    "3.2-comparing-ordering": {
        "m1": (3, [r"look\w*\s+at\s+the\s+(?:last|second|ones|right)\s*(?:digit|number)?", r"\b(\d)(\d)\b\s+(?:is\s+)?(?:bigger|greater|larger|more)\s+than\s+\2\1\b",
                   r"(?:bigger|greater|larger|smaller).{0,40}\b(?:ones|last digit|second digit)\b"]),
        "m2": (3, [r"\bsigns?\b.{0,40}(?:wrong|mix|confus|other way)|(?:wrong|mix|confus)\w*.{0,30}\bsigns?\b", r"greater\s+than\s+sign|less\s+than\s+sign",
                   r"alligator|crocodile|\bmouth\b", r"[<>]"]),
        "m3": (3, [r"(?:same|equal).{0,40}\b(?:start|begin|first)", r"both\s+(?:start|begin)\s+with"]),
    },
    "3.7-money": {
        "m1": (2, [r"ngwee", r"k\d+\.\d+.{0,20}k\d+\.\d{3}", r"\.100\b"]),
        "m2": (3, [r"\bchange\b", r"count\w*\s+up", r"(?:give|giving|how much to give)\s+back"]),
        "m3": (2, [r"(?:mix|confus)\w*.{0,25}(?:notes?|coins?|k\d+)", r"denomination", r"recogni[sz]\w*\s+(?:the\s+)?(?:coins|notes|money)",
                   r"(?:know|name|read)\w*\s+(?:the\s+)?(?:notes|coins)", r"wrong\s+note"]),
    },
    "3.8-shapes-2d": {
        "m1": (3, [r"square.{0,30}rectangle|rectangle.{0,30}square"]),
        "m2": (3, [r"sides?\s+(?:and|with)\s+lines?|lines?\s+(?:and|with)\s+sides?", r"(?:fold|symmetry)\s+line\s+is\s+a\s+side|line\b.{0,20}\bis\s+a\s+side"]),
        "m3": (2, [r"symmetr", r"\bfold"]),
    },
}

TOPICS = {
    "2.1-counting": [
        {"id": "t-range", "weight": 2, "origin": "C", "title": "Counting past 20 (the tens)",
         "patterns": [r"count\w*\s+(?:past|beyond|above|over|after|more than|up to|to)\s+" + N, r"\bafter\s+\d9\b", r"stuck\s+(?:at|after)\s+\d+",
                      r"(?:big|bigger|large|higher)\s+numbers", r"\b\d9\b.{0,30}(?:what comes next|next|again)", r"to\s+(?:100|a hundred|one hundred)"],
         "text": "Many children count to 20 and stop at the tens: after 29 they do not know that 30 comes next, because each "
                 "new ten needs a new word.",
         "fix": "Count the tens first: ten, twenty, thirty ... one hundred. Then start just before a ten and count on: "
                "twenty-eight, twenty-nine, thirty.",
         "tonga_en": "Ten, twenty, thirty ... one hundred."},
        {"id": "t-skip", "weight": 3, "origin": "C", "title": "Skip counting in 2s, 5s and 10s",
         "patterns": [r"skip[- ]?count", r"count\w*\s+(?:in|by)\s+(?:2|5|10|twos|fives|tens)\b", r"\bin\s+(?:2|5|10)s\b",
                      r"count\w*\s+by\s+(?:twos|fives|tens)|counting\s+by\s+\w+s\b|\bby\s+fives\b"],
         "text": "Skip counting is counting groups: every step jumps by the same number. Children who only count in ones lose "
                 "the jump.",
         "fix": "Put the stones in pairs. Whisper the first stone of each pair, say the second one aloud: (one) TWO, "
                "(three) FOUR, (five) SIX.", "tonga_en": None},
        {"id": "t-zero", "weight": 1, "origin": "C", "title": "The meaning of zero",
         "patterns": [r"\bzero\b", r"\bnought\b", r"\bzero\b"],
         "text": "Zero is the number for none: an empty hand, an empty cup. Children often think zero is not a number at all.",
         "fix": "How many stones in my hand? (open an empty hand) None. None is zero. We write it 0.",
         "tonga": {"toi": "[toi withheld: speaker-written]", "origin": ["T01", "T09"]}},
        {"id": "t-write", "weight": 3, "origin": "C", "title": "Writing numerals the wrong way round",
         "patterns": [r"(?:write|writing|form\w*)\b.{0,40}\b(?:backwards|reversed?|wrong way|mirror)",
                      r"(?:backwards|reversed?|wrong way round|mirror)\b.{0,20}\bnumerals\b",
                      r"\b[3579]\s+(?:and\s+\d\s+)?(?:backwards|the wrong way)"],
         "text": "Reversed 3, 5, 7 and 9 are common at this age: a habit of how the numeral is formed, not a counting problem.",
         "fix": "Write it in the air with me, big, from the top. Then in the sand with one finger. Then on your slate.",
         "tonga_en": None},
    ],
    "2.2-addition": [
        {"id": "t-counton", "weight": 3, "origin": "C", "title": "Counting all instead of counting on",
         "patterns": [r"count\w*\s+on\b", r"from\s+(?:one|1)\s+again|again\s+from\s+(?:one|1)", r"start\w*\s+(?:counting\s+)?from\s+(?:one|1)\b",
                      r"count\w*\s+(?:both|all)\b", r"from\s+(?:one|1)\s+every\s+time"],
         "text": "To add 5 + 3 they count out both piles from one. Counting on starts from the bigger number and counts only "
                 "the smaller pile.",
         "fix": "Keep 5 in your head. Now count on with the 3 stones: six, seven, eight.",
         "tonga_en": "Adding is counting forward."},
        {"id": "t-fingers", "weight": 2, "origin": "C", "title": "Fingers or stones for every sum",
         "patterns": [r"finger", r"without\s+(?:the\s+)?(?:counters|stones|objects|caps)", r"need\w*\s+(?:the\s+)?(?:stones|counters|objects|caps)",
                      r"in\s+(?:their|the)\s+heads?", r"mental\w*"],
         "text": "Fingers and stones are the right start; the aim is to know the small facts by heart, so that the hands are "
                 "free for bigger sums.",
         "fix": "Do not count on your fingers this time.",
         "tonga_en": "Do not count on your fingers this time."},
        {"id": "t-bonds", "weight": 3, "origin": "C", "title": "Number bonds to 10",
         "patterns": [r"\bbonds?\b", r"make\s+(?:a\s+)?(?:10|ten)\b", r"pairs?\s+(?:that\s+|which\s+)?(?:make|to)\s+(?:10|ten)", r"addition facts"],
         "text": "Without the pairs that make 10 (1 + 9, 2 + 8, 3 + 7 ...) every sum is counted from scratch.",
         "fix": "Ten stones, two hands. How many in this hand? So how many are in the other hand?", "tonga_en": None},
        {"id": "t-signs", "weight": 2, "origin": "C", "title": "Reading + and =",
         "patterns": [r"plus\s+sign|equals?\s+sign|the\s+signs\b|\bsymbols?\b"],
         "text": "The signs are new words: + says 'put together', = says 'is the same as'. Children read = as 'the answer comes "
                 "next'.",
         "fix": "Read it with me: 3 plus 2 is the same as 5. Put 3 stones and 2 stones together. Count: 5.",
         "tonga_en": "3 plus 2 gives 5."},
        {"id": "t-story", "weight": 2, "origin": "C", "title": "Story sums",
         "patterns": [r"story\s+(?:sums|problems)", r"word\s+problems?", r"\bstories\b"],
         "text": "In a story sum the child must find the numbers and the action (put together or take away) in the words "
                 "first.",
         "fix": "Tell the story with stones: Chanda has 4 mangoes, Mutinta gives him 3. Show me the 4. Show me the 3. Put "
                "them together. How many?", "tonga_en": None},
    ],
    "2.3-subtraction-intro": [
        {"id": "t-signs", "weight": 4, "origin": "C", "title": "Mixing up adding and taking away",
         "patterns": [r"mix\w*\s+up\s+(?:plus|add\w*|\+)\s+(?:and|with)\s+(?:minus|subtract\w*|tak\w*|-)", r"add\w*\s+instead\s+of\s+(?:subtract|tak)",
                      r"(?:confus|mix)\w*\s+(?:plus|add\w*)\s+(?:and|with)\s+(?:minus|subtract\w*|tak\w*)", r"minus\s+sign",
                      r"add\w*\s+when\s+.{0,25}(?:take\s+away|minus|subtract)"],
         "text": "The two signs look alike to a beginner, and both lessons use the same stones; the action has to be said "
                 "aloud each time.",
         "fix": "Before you start, say what the sign tells you: plus, we put together; minus, we take away.",
         "tonga_en": None},
        {"id": "t-countback", "weight": 2, "origin": "C", "title": "Counting back",
         "patterns": [r"count\w*\s+(?:back|backwards|down)\b", r"backwards\s+from"],
         "text": "Counting back is harder than counting forward: children lose the order of the number names going down.",
         "fix": "Stand ten stones in a row. Take one away each time and say the number that is left: nine, eight, seven.",
         "tonga_en": None},
        {"id": "t-compare", "weight": 2, "origin": "C", "title": "'How many more?' (the difference)",
         "patterns": [r"how\s+many\s+more", r"\bdifference\b", r"more\s+than\s+the\s+other"],
         "text": "'How many more?' asks for the difference between two piles, not for a total: children add the two piles.",
         "fix": "Which pile has more? How many more?",
         "tonga_en": "Which pile has more? How many more?"},
        {"id": "t-story", "weight": 2, "origin": "C", "title": "Story sums with taking away",
         "patterns": [r"story\s+(?:sums|problems)", r"word\s+problems?", r"\bstories\b"],
         "text": "In a story sum the child must find the numbers and the action in the words first.",
         "fix": "Tell the story with stones: there are 7 mangoes, Mutinta eats 2. Show me the 7. Take the 2 away. How many "
                "are left?", "tonga_en": None},
        {"id": "t-takeaway", "weight": 2, "origin": "C", "title": "What 'left' means",
         "patterns": [r"(?:what|how\s+many)\s+(?:is|are)\s+left", r"left\s+over", r"\bremain"],
         "text": "Children count the stones that were taken away instead of the stones that are left.",
         "fix": "Count what is left.", "tonga_en": "Count what is left."},
    ],
    "3.1-place-value-intro": [
        {"id": "t-reverse", "weight": 3, "origin": "C", "title": "Teen numbers written the wrong way round (16 > 61)",
         "patterns": [r"\b(\d)(\d)\b\s+(?:and|with|for|becomes|as)\s+\2\1\b", r"sixty[- ]one.{0,30}sixteen|sixteen.{0,30}sixty[- ]one",
                      r"revers\w*\s+the\s+digits|digits?\s+(?:the\s+)?wrong\s+way", r"\b1\d\b\s+(?:becomes|as)\s+\d1\b"],
         "text": "Teen numbers are said ones first in English (six-teen) but written tens first, so children write 61 for 16.",
         "fix": "Sixteen is one ten and six ones. Show me the bundle first, then the six loose sticks. The bundle is written "
                "first.", "tonga_en": None},
        {"id": "t-bundle", "weight": 3, "origin": "C", "title": "A bundle is ten, not one",
         "patterns": [r"bundl", r"group\w*\s+of\s+(?:10|ten)", r"tie\w*\s+.{0,20}\b(?:10|ten)\b"],
         "text": "Children count a tied bundle as 'one' because it is one thing. The idea of the lesson is that one bundle is "
                 "ten sticks.",
         "fix": "Ten sticks make one bundle. Count the bundles in tens: ten, twenty, thirty.",
         "tonga_en": "Take 10 sticks and tie them together with grass."},
    ],
    "3.2-comparing-ordering": [
        {"id": "t-moreless", "weight": 2, "origin": "C", "title": "More and less with objects",
         "patterns": [r"more\s+(?:and|or)\s+less", r"\bfewer\b|\bless\b", r"spread\s+out", r"(?:longer|bigger)\s+(?:row|pile)"],
         "text": "Young children judge 'more' by how much space a pile takes, not by how many are in it.",
         "fix": "Put the stones in two rows, one under the other. The longer row has more.",
         "tonga_en": "Put the stones in two rows. The longer row has more."},
        {"id": "t-order", "weight": 2, "origin": "C", "title": "Putting numbers in order",
         "patterns": [r"\border", r"\barrang", r"smallest\s+to\s+(?:biggest|largest)|(?:biggest|largest)\s+to\s+smallest", r"ascending|descending"],
         "text": "Ordering is comparing many times: children compare the first two and then lose track.",
         "fix": "Put the numbers in order, smallest first.",
         "tonga_en": "Put the numbers in order, smallest first."},
        {"id": "t-same", "weight": 2, "origin": "C", "title": "'The same as' (equal)",
         "patterns": [r"the\s+same\s+as", r"\bequal", r"same\s+number"],
         "text": "'The same' is a new idea in maths: two piles are the same when every stone has a partner.",
         "fix": "8 and 8 are the same. Nobody has more.",
         "tonga_en": "8 and 8 are the same. Nobody has more."},
    ],
    "3.7-money": [
        {"id": "t-k", "weight": 3, "origin": "F", "from": "teacher move 'Child forgets the K'", "title": "Forgets the K",
         "patterns": [r"forget\w*\s+(?:to\s+write\s+)?the\s+k\b", r"without\s+the\s+k\b", r"write\s+the\s+k\b"],
         "text": "The answer is a number with no unit, so it means nothing as money.",
         "fix": "Seven what? Seven sweets? Seven days? Write the K, otherwise the answer means nothing.",
         "tonga_en": "Always write K before the number."},
        {"id": "t-total", "weight": 2, "origin": "C", "title": "Adding prices (the total)",
         "patterns": [r"\btotal\b", r"add\w*\s+(?:up\s+)?(?:the\s+)?prices", r"how\s+much\s+(?:altogether|in\s+total)", r"two\s+things"],
         "text": "Children work with one price at a time; a total asks them to put two prices together first.",
         "fix": "What does the bread cost? What does the sugar cost? Put the two prices together: that is the total.",
         "tonga_en": None},
    ],
    "3.8-shapes-2d": [
        {"id": "t-names", "weight": 2, "origin": "C", "title": "Naming the shapes",
         "patterns": [r"names?\s+of\s+(?:the\s+)?shapes", r"(?:name|call)\w*\s+.{0,30}(?:circle|triangle|square|rectangle)",
                      r"round\s+thing", r"\boval\b", r"know\s+the\s+shapes"],
         "text": "Children call every round thing a circle and every four-sided thing a square: the names have not yet met "
                 "the counting of sides.",
         "fix": "Show the shape, say its name, the children say it back: circle, square, rectangle, triangle.",
         "tonga": {"toi": "[toi withheld: speaker-written]", "origin": ["T01"]}},
        {"id": "t-rotated", "weight": 3, "origin": "C", "title": "A turned shape is still the same shape",
         "patterns": [r"upside[- ]down", r"\bturn(?:ed)?\b", r"rotat", r"on\s+its\s+(?:side|point)", r"\bdiamond\b"],
         "text": "A square on its point is called a 'diamond', a triangle upside down 'not a triangle': children name the "
                 "picture, not the shape.",
         "fix": "Turn it round. Count the sides again. Still three straight sides? Still a triangle.", "tonga_en": None},
        {"id": "t-sides", "weight": 2, "origin": "F", "from": "teacher move 'Child guesses the name'",
         "title": "Guessing the name before counting the sides",
         "patterns": [r"count\w*\s+(?:the\s+)?sides", r"guess\w*\s+(?:the\s+)?(?:name|shape)", r"without\s+looking", r"how\s+many\s+sides"],
         "text": "The child names the shape from its look and never counts the sides.",
         "fix": "Don't tell me the name yet. Count the sides first. Now tell me.",
         "tonga_en": "Count the sides first. Then tell me the name."},
    ],
}


def attach(lesson_id, mistakes, tonga_for):
    """(mistakes with 'patterns' / 'weight', struggle topics) for one lesson; every regex compiled here so a bad one
    fails the build, not a teacher's turn"""
    pats = PATTERNS.get(lesson_id, {})
    for m in mistakes:
        k = m["id"].split("#", 1)[1]
        if k in pats:
            m["weight"], m["patterns"] = pats[k]
    tops = []
    for t in TOPICS.get(lesson_id, []):
        tg = t.get("tonga") or (tonga_for(t["tonga_en"]) if t.get("tonga_en") else None)
        tops.append({"id": f"{lesson_id}#{t['id']}", "title": t["title"], "text": t["text"], "fix": t["fix"],
                     "fix_tonga": tg, "origin": t["origin"], "from": t.get("from"), "weight": t.get("weight", 2),
                     "patterns": t["patterns"], "kind": "struggle_topic"})
    for m in list(mistakes) + tops:
        for rx in m.get("patterns", []):
            re.compile(rx)
    return mistakes, tops


# the lesson's counters: the boxes use stones (mabwe) and, for bundles, small sticks (tusamu); the Funda guides say bottle
# caps and straws (a speaker gave words for caps on 23 Oct, but the counting lessons keep stones: DECISIONS 23 Oct)
COUNTERS = {"2.1-counting": "stones", "2.2-addition": "stones", "2.3-subtraction-intro": "stones",
            "3.2-comparing-ordering": "stones", "3.1-place-value-intro": "sticks"}


def counters(lesson_id, text):
    """the guide's caps / straws > the lesson's stones / sticks, keeping a capital letter"""
    want = COUNTERS.get(lesson_id)
    if not want:
        return text
    one = want[:-1]
    subs = ([(r"\bbottle[- ](?:caps|tops)\b", want), (r"\bbottle[- ](?:cap|top)\b", one), (r"\bcaps\b", want),
             (r"\bcap\b", one)] if want == "stones" else
            [(r"\b(?:drinking\s+)?straws\b", want), (r"\bstraw\b", one), (r"\bcaps\b", want), (r"\bcap\b", one)])

    def keep(rep):
        return lambda m: rep[0].upper() + rep[1:] if m.group(0)[0].isupper() else rep
    for rx, rep in subs:
        text = re.sub(rx, keep(rep), text, flags=re.I)
    return text
