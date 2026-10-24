"""Renderer: plug-in items into WhatsApp-sized parts, with the origin of every Tonga line (23 Oct 2026, the Jesuit).

Nothing here writes Tonga. Lines come from the plug-in (teacher T01 / T09, book B, lexicon L, derived D = needs
confirmation). A D line whose build-time checks rang an alarm (core/checks.py) is shown in English only, with the flag
in the record; a D line without stored checks is checked here with the pure checks (numbers, glossary forbids).
The teacher's language preference decides what is shown: en = the English lines (Tonga on request), toi = the Tonga
line with its origin, both = Tonga then the English.
"""
import math

from . import checks as CK

ORIGIN_TAG = {"T01": "T", "T09": "T", "B": "B", "L": "L", "D": "D"}


def tag(origins):
    return "[" + "+".join(dict.fromkeys(ORIGIN_TAG.get(o, o) for o in origins)) + "]" if origins else ""


def line_ok(ln, glossary):
    """(ok, flags): may this Tonga line be shown?"""
    if not ln.get("served") or not ln.get("toi"):
        return False, ["not served"]
    if "D" in ln.get("origin", []):
        fl = ln.get("checks")
        if fl is None:
            fl = CK.check_line(ln["en"], ln["toi"], glossary)
        return (not fl), list(fl)
    return True, []


def grade_label(P, item):
    """'Grade 2' / 'Grades 2 and 3' from the item's own grades, else its unit's grade, else '' (24 Oct 2026: the
    header said 'Grade 1' for every lesson after Grades 2-3 were added; the grade now comes from the data)"""
    gs = sorted(set(item.get("grades") or []))
    if not gs:
        u = getattr(P, "units", {}).get(getattr(P, "unit_of", {}).get(item.get("id")))
        gs = [u["grade"]] if u and u.get("grade") else []
    if not gs:
        return ""
    return ("Grade " if len(gs) == 1 else "Grades ") + " and ".join(map(str, gs))


def lesson_parts(item, max_lines=6, strategy=None):
    """the part plan of a lesson: ['overview', ('lines', i, j), ..., 'checkpoint'?]"""
    served = [ln for ln in item.get("lines", []) if ln.get("served")]
    n = max(1, math.ceil(len(served) / max_lines)) if served else 0
    plan = ["overview"] + [("card", c) for c in range(len((item.get("card") or {}).get("parts", [])))]
    plan += [("lines", k * max_lines, min(len(served), (k + 1) * max_lines)) for k in range(n)]
    if strategy and strategy.get("checkpoint"):
        plan.append("checkpoint")
    return plan


def render_lesson(P, item, part, pref, strategy=None):
    """(body, origins, flags, n_parts)"""
    plan = lesson_parts(item, P.rules.get("max_lines_per_part", 6), strategy)
    step = plan[min(part, len(plan) - 1)]
    origins, flags, out = [], [], []
    gl = grade_label(P, item)
    head = f"{item['title']} ({gl + ', ' if gl else ''}{item['subject'].capitalize()}), part {part + 1} of {len(plan)}"
    if step == "overview":
        out.append(head)
        if item.get("scope"):
            out.append(item["scope"])
        if item.get("card"):   # grade1_new (23 Oct): a short English lesson card by Claude stands in for the Funda guide
            out.append(f"Lesson card (English), next part. {item['card']['label']}")
        elif item["guide"].get("funda"):
            out.append(f"Teacher guide (English): {item['guide']['funda'].rsplit('/', 1)[-1]}")
        elif (item["guide"].get("older") or {}).get("label"):   # 24 Oct: a guide pitched above this lesson, kept apart
            out.append(f"{item['guide']['older']['label']}: on the lesson page, not part of this lesson.")
        kws = [k for k in item.get("key_words", []) if k.get("toi") and not k["toi"].startswith("(")]
        if kws:
            out.append("Key words:")
            for k in kws:
                out.append(f"- {k['en']}: {k['toi']} {tag(k['origin'])}")
                origins += k["origin"]
        out.append(P.t("draft_lines"))
    elif step == "checkpoint":
        cp = strategy["checkpoint"]
        out += [head, f"Checkpoint: {cp['task']}", cp["how"], f"Mastered: {cp['mastered']}",
                f"Developing: {cp['developing']}", f"Next step: {cp['next']}",
                strategy.get("label") or "[F: Funda guide, not yet reviewed]"]
        if strategy.get("scope"):
            out.append(f"Note: {strategy['scope']}")
        origins.append(strategy.get("origin") or "F")
    elif step[0] == "card":
        out.append(head)
        for h, t in item["card"]["parts"][step[1]]:
            out.append(f"{h}: {t}")
        out.append("[C: AI draft (Claude) from the CDC syllabus, not yet reviewed by a teacher]")
        origins.append("C")
    else:
        _, i, j = step
        out.append(head)
        out.append("Say it in Tonga:" if pref != "en" else "Classroom lines (Tonga on request: choose Tonga or Both):")
        served = [ln for ln in item["lines"] if ln.get("served")]
        for ln in served[i:j]:
            ok, fl = line_ok(ln, P.glossary)
            if pref == "en":
                out.append(f"{ln['n']}. {ln['en']}")
                continue
            if not ok:
                out.append(f"{ln['n']}. {ln['en']} [Tonga held back: check]")
                flags.append(f"{item['id']} line {ln['n']}: {'; '.join(fl)}")
                continue
            out.append(f"{ln['n']}. {ln['toi']} {tag(ln['origin'])}")
            if pref == "both":
                out.append(f"   ({ln['en']})")
            origins += ln["origin"]
    return "\n".join(out), list(dict.fromkeys(origins)), flags, len(plan)


CARD_LABELS = {"answer": "Answer", "idea": "The idea", "local_example": "A local example", "how_to_teach": "How to teach it"}


def render_card(P, item, part, pref):
    """two parts: (answer, idea, local example) and (how to teach, check questions)"""
    gl = f", Grade {', '.join(map(str, item['grades']))}" if item.get("grades") else ""
    out = [f"{item['title']} (card{gl}), part {part + 1} of 2",
           P.t("draft_card")]
    if pref != "en" and part == 0:
        out.append(P.t("toi_not_ready"))
    keys = ("answer", "idea", "local_example") if part == 0 else ("how_to_teach",)
    for k in keys:
        out.append(f"{CARD_LABELS[k]}: {item['parts'][k]}")
    if part == 1 and item.get("check_questions"):
        out.append("Check questions:")
        out += [f"- {q['q']} ({q['a']})" for q in item["check_questions"][:3]]
    return "\n".join(out), ["C"], [], 2


def render_pointer(P, item):
    out = [P.t("pointer_intro", title=item["title"])] + [f"- {w}" for w in item.get("where", [])]
    if item.get("missing"):
        out.append("Missing: " + "; ".join(item["missing"]))
    return "\n".join(out), ["B"], [], 1


def print_parts(item):
    """the part plan of a lesson from PRINT (toi_primary): overview, the printed lines in parts sized by characters
    (built with the lesson: part_ranges), then the checkpoint if the notes carry one"""
    plan = ["overview"] + [("lines", a, b) for a, b in item.get("part_ranges") or []]
    tn = item.get("teacher_notes") or {}
    if tn.get("checkpoint") or any((f.get("checkpoint") or {}).get("mastered") for f in item.get("funda_notes") or []):
        plan.append("checkpoint")
    return plan


def render_print_lesson(P, item, part, pref):
    """A Chitonga (subject) lesson: the Tonga is the print itself, verbatim, origin B, shown whatever the language
    preference (the subject IS Tonga); English teacher notes are Claude-written (C) or Funda ai_draft (F) and say so;
    the English in brackets beside a Tonga line (pref en / both) is the Jesuit's unverified gloss, never a reference.
    Nothing here writes Tonga. (Stream C, 23 Oct 2026, the Jesuit.)"""
    plan = print_parts(item)
    step = plan[min(part, len(plan) - 1)]
    out, origins = [], []
    gl = grade_label(P, item)
    head = f"{item['title']} ({gl + ', ' if gl else ''}Chitonga), part {part + 1} of {len(plan)}"
    tn = item.get("teacher_notes") or {}
    if step == "overview":
        out += [head, f"From the print: {item['guide'].get('print') or '-'}"]
        if tn.get("what"):
            out.append(f"What it teaches: {tn['what']}")
        if tn.get("steps"):
            out.append("Steps:")
            out += [f"{i}. {s}" for i, s in enumerate(tn["steps"], 1)]
        out.append("[C: teacher notes written by Claude, not yet reviewed by a teacher]")
        for f in (item.get("funda_notes") or [])[:1]:
            if f.get("why_it_matters"):
                out.append(f"Why it matters: {f['why_it_matters']}")
                out.append("[F: Funda draft written by Claude, not reviewed; a Chitonga word inside it is not from "
                           "the print and not checked]")
                origins.append("F")
        for m in item.get("missing") or []:
            out.append(f"Not in our print: {m}")
        if item.get("part_ranges"):
            out.append("Next: the book's own Tonga, as printed [B]" +
                       (" (English in brackets: an unverified gloss, not a translation)" if pref != "toi" else ""))
        origins.append("C")
    elif step == "checkpoint":
        out.append(head)
        cp = tn.get("checkpoint")
        if cp:
            out += [f"Checkpoint: {cp['task']}", f"Mastered: {cp['mastered']}", f"Developing: {cp['developing']}",
                    f"Next step: {cp['next']}", "[C: written by Claude, not yet reviewed]"]
            origins.append("C")
        for f in item.get("funda_notes") or []:
            c = f.get("checkpoint") or {}
            if c.get("mastered"):
                out += [f"Funda checkpoint ({f['funda_subtopic']}): mastered: {c['mastered']}",
                        f"Developing: {c['developing']}", f"Next step: {c['next']}"]
        if item.get("funda_notes"):
            out.append("[F: Funda draft written by Claude, not reviewed; a Chitonga word inside it is not from the "
                       "print and not checked]")
            origins.append("F")
    else:
        _, i, j = step
        out.append(head)
        served = [ln for ln in item.get("lines", []) if ln.get("served") and ln.get("toi")]
        src = None
        for ln in served[i:j]:
            if ln.get("src") != src:
                src = ln.get("src")
                out.append(f"({src})")
            out.append(f"{ln['n']}. {ln['toi']}")
            if pref != "toi" and ln.get("en"):
                out.append(f"   ({ln['en']})")
        out.append("[B: printed book, verbatim]")
        origins.append("B")
    return "\n".join(out), list(dict.fromkeys(origins)), [], len(plan)


def shown_lines(P, item, part, pref, strategy=None):
    """the Tonga box lines a delivered part put in front of the teacher, as data, for the flywheel (24 Oct 2026, the
    Jesuit; tools/chitonga/harness/FLYWHEEL.md): [{n, en, toi, origin, shown, held}] where shown = the Tonga was
    displayed and held = the line's Tonga was held back by its checks ('[Tonga held back: check]'). Mirrors the part
    plans of render_lesson / render_print_lesson exactly (same plan functions, same line_ok), so the feedback prompt
    can only ever ask about a line the teacher has just seen. [] for overview, card, checkpoint and non-lesson parts."""
    if item.get("kind") != "lesson":
        return []
    if item.get("toi_primary"):
        plan = print_parts(item)
        step = plan[min(part, len(plan) - 1)] if plan else None
        if not isinstance(step, tuple):
            return []
        served = [ln for ln in item.get("lines", []) if ln.get("served") and ln.get("toi")]
        return [{"n": ln.get("n"), "en": ln.get("en"), "toi": ln["toi"], "origin": ["B"], "shown": True,
                 "held": False, "src": ln.get("src"), "en_is_gloss": True} for ln in served[step[1]:step[2]]]
    plan = lesson_parts(item, P.rules.get("max_lines_per_part", 6), strategy)
    step = plan[min(part, len(plan) - 1)] if plan else None
    if not (isinstance(step, tuple) and step[0] == "lines"):
        return []
    served = [ln for ln in item.get("lines", []) if ln.get("served")]
    out = []
    for ln in served[step[1]:step[2]]:
        ok = line_ok(ln, P.glossary)[0]
        out.append({"n": ln.get("n"), "en": ln.get("en"), "toi": ln.get("toi") if ok else None,
                    "origin": list(ln.get("origin", [])), "shown": bool(ok and pref != "en"),
                    "held": bool(pref != "en" and not ok), "en_is_gloss": False})
    return out


def sample_parts(item):
    """the part plan of a sample lesson: overview (placement, what it teaches), key words, checkpoint (if the guide has
    one)"""
    plan = ["overview", "terms"]
    if ((item.get("strategy") or {}).get("checkpoint") or {}).get("task"):
        plan.append("checkpoint")
    return plan


def render_sample(P, item, part, pref):
    """a SAMPLE LESSON (24 Oct 2026, build_samples.py): an English Funda guide for Grades 4-7, labelled as a sample,
    with its curriculum placement (2023 first; 2013 marked old) and its key words: Tonga only where a teacher or a
    printed source has the word in this sense, else 'Tonga word needed'. Nothing here writes Tonga."""
    plan = sample_parts(item)
    step = plan[min(part, len(plan) - 1)]
    gs = sorted(item.get("grades") or [])
    gl = (f"Grade {gs[0]}" if len(gs) == 1 else f"Grades {gs[0]}-{gs[-1]}") if gs else ""
    head = (f"Sample lesson: {item['title']} ({gl + ', ' if gl else ''}{(item.get('subject') or '').capitalize()}), "
            f"part {part + 1} of {len(plan)}")
    out, origins = [head], ["F"]
    if step == "overview":
        out.append(item.get("label") or "Sample lesson (English, AI draft, not yet reviewed)")
        for p in sorted(item.get("placements") or [], key=lambda x: (x.get("curriculum") != "2023", x.get("grade"))):
            if p.get("curriculum") == "2023":
                refs = "; ".join(f"{r['code']} {r['sub_topic']}" for r in p.get("refs") or [])
                out.append(f"- {p['level']}: 2023 curriculum (Funda's draft syllabus{': ' + refs if refs else ''})")
            else:
                extra = ""
                if p.get("in_2023"):
                    extra = "; in the 2023 curriculum: Form 1, " + "; ".join(
                        f"{r['code']} {r['sub_topic']}" for r in p["in_2023"].get("refs") or [])
                out.append(f"- {p['level']}: old curriculum (2013){extra}")
        rel = item.get("related_refs") or []
        if rel:
            out.append("Related: " + "; ".join(
                f"{'2023' if r['curriculum'] == '2023' else '2013'} {r['level']} ({r['code']} {r['sub_topic']})"
                for r in rel[:3]))
        if item.get("teaches"):
            out.append(f"What it teaches: {item['teaches']}")
        if item.get("what_it_is"):
            w = item["what_it_is"]
            out.append(w if len(w) <= 600 else w[:597].rsplit(" ", 1)[0] + "...")
        out.append("The full guide is on the lesson page (zamai.pro assistant).")
    elif step == "terms":
        out.append("Key words:")
        for k in item.get("key_terms") or []:
            if k.get("tonga_needed") or not k.get("toi"):
                out.append(f"- {k['en']}: Tonga word needed")
            else:
                out.append(f"- {k['en']}: {k['toi']} [{k.get('badge') or k['origin']}]")
                origins += [o for o in str(k["origin"]).replace(",", "+").replace(" ", "").split("+") if o]
        out.append("Tonga word needed = no teacher and no printed book has given us this word in this sense yet. "
                   "[T: teacher; B: printed book or dictionary, in this sense, not yet checked for this lesson]")
    else:
        cp = item["strategy"]["checkpoint"]
        out += [f"Checkpoint: {cp.get('task', '')}", cp.get("how", ""), f"Mastered: {cp.get('mastered', '')}",
                f"Developing: {cp.get('developing', '')}", f"Next step: {cp.get('next', '')}",
                "[F: Funda guide, sample lesson, not yet reviewed]"]
    return "\n".join(x for x in out if x), list(dict.fromkeys(origins)), [], len(plan)


def review_line(item):
    """'[NEEDS REVIEW] Health content: ...' for a labelled item (Chris 24 Oct: "Nothing is held back, just label it
    for review"), else ''"""
    if not item.get("needs_review"):
        return ""
    return f"[{(item.get('review_tag') or 'Needs review').upper()}] {item.get('review_note') or ''}".strip()


def render(P, item, part, pref, strategy=None):
    body, origins, flags, n = _render(P, item, part, pref, strategy)
    rl = review_line(item)
    if rl:                                   # on every part, so a teacher who only sees part 2 sees it too
        body = body + "\n" + rl
    return body, origins, flags, n


def _render(P, item, part, pref, strategy=None):
    if item["kind"] == "sample":
        return render_sample(P, item, part, pref)
    if item["kind"] == "lesson" and item.get("toi_primary"):
        return render_print_lesson(P, item, part, pref)
    if item["kind"] == "lesson":
        return render_lesson(P, item, part, pref, strategy)
    if item["kind"] == "card":
        return render_card(P, item, part, pref)
    if item["kind"] == "pointer":
        return render_pointer(P, item)
    raise ValueError(item["kind"])


def _tonga_ok(P, st, tg):
    """a Tonga line attached to a strategy step is shown only if its box line may be shown (a D line whose checks
    rang an alarm is held back here too, as in the lesson itself); 'n' names the box line"""
    if not tg or not tg.get("toi"):
        return False
    if "D" not in tg.get("origin", []) or not tg.get("n"):
        return True
    it = P.items.get(st.get("lesson")) or {}
    ln = next((x for x in it.get("lines", []) if x.get("n") == tg["n"]), None)
    return bool(ln) and line_ok(ln, P.glossary)[0]


def render_intervention(P, st, mistake, pref):
    out, origins = [], [st.get("origin") or "F"]
    if mistake:
        out.append(f"Likely: {mistake['title']}. {mistake['text']}")
        out.append(f"Say: \"{mistake['fix']}\"")
        if pref != "en" and _tonga_ok(P, st, mistake.get("fix_tonga")):
            out.append(f"   {mistake['fix_tonga']['toi']} {tag(mistake['fix_tonga']['origin'])}")
            origins += mistake["fix_tonga"]["origin"]
        if mistake.get("origin") == "C":
            origins.append("C")
    out.append("Try this:")
    out += [f"{i}. {s['step']}" for i, s in enumerate(st["intervention"], 1)]
    out += [f"If still not: {s['step']}" for s in st["escalate"]]
    if st.get("scope"):
        out.append(f"Note: {st['scope']}")
    if st.get("label"):   # grade1_new card strategies (origin C) carry their own label
        out.append(st["label"] + " (steps: Funda mastery loop)")
    elif mistake and mistake.get("origin") == "C":
        out.append("[C: this struggle and its 'Say' line drafted by Claude (not in the Funda guide); F: the steps, Funda "
                   "guide and mastery loop; none yet reviewed by a teacher]")
    else:
        out.append("[F: Funda guide and mastery loop, not yet reviewed by a teacher]")
    return "\n".join(out), origins


def render_challenge(P, st, pref):
    out, origins = ["For the learners who already have it:"], [st.get("origin") or "F"]
    for s in st["challenge"]:
        out.append(f"- {s['step']}")
        if pref != "en" and s.get("tonga"):
            out.append(f"   {s['tonga']['toi']} {tag(s['tonga']['origin'])}")
            origins += s["tonga"]["origin"]
    out.append(st.get("label") or "[F: Funda guide and mastery loop, not yet reviewed by a teacher]")
    return "\n".join(out), origins
