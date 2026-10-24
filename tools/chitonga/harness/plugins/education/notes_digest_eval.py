"""Measure the weekly digest on 20 realistic lesson notes (24 Oct 2026, the Jesuit). The task: "Report how often a digest
item lacks a correct source note."

THE NOTES. One teacher, two classes (2C, Grade 2; Grade 3 Blue), the school week 19-23 Oct 2026. 15 English notes
written by me as a teacher would (short, mixed tags, one untagged, learner names in two, a class-level attention). Five
Tonga notes that are REAL teacher-written lesson lines (T01 / T09, plug-in boxes and strategies), never composed by me;
each has its teacher's English, so the translation is checkable. Each note carries GOLD categories (which digest
sections it can support).

THE MEASURE, per digest item:
  sourced   at least one cited note's gold categories include the item's section (themes: any cited note), and
  faithful  (my reading, recorded in review.json after the run) the item says what its cited note(s) say.
Reported twice: for the model's RAW items (before the code's checks) and for the items SHOWN (after the checks).
Plus: matches (didnt / attention items with a retrieved strategy or tip), translations (served route and memory off).

    python -X utf8 tools/chitonga/harness/plugins/education/notes_digest_eval.py [--out DIR] [--no-lock]
Heavy: the served translator and the Gemma 4 E4B server, under the Beelink lock "notes-digest".
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, "..", ".."))
REPO = os.path.normpath(os.path.join(HARNESS, "..", "..", ".."))
for p in (HARNESS, os.path.join(HARNESS, "..", "llm")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(REPO, "internal", "chitonga_train", "harness", "notes_digest_eval_2026-10-24")
TEACHER = "260970000077"

# (class, day, subject, item, tags, attention, text, gold, teacher_en for Tonga)
NOTES = [
    ("2C", "2026-10-19", "maths", "lesson:g2.2.1-counting", ["didnt"], None,
     "Counting past 20 did not work. They lost track after 20 and started again from 1.", ["didnt"], None),
    ("2C", "2026-10-19", "maths", "lesson:g2.2.1-counting", ["worked", "idea"], None,
     "Bottle tops in groups of ten worked much better than counting on fingers. Next time bring enough for every pair.",
     ["worked", "ideas"], None),
    ("2C", "2026-10-20", "english", None, ["attention"], "learner",
     "Mutinta still cannot hear the difference between the b and d sounds. Sit her at the front.", ["attention"], None),
    ("2C", "2026-10-20", "science", None, ["worked"], None,
     "The plant walk outside worked well, everyone found a leaf and a root.", ["worked"], None),
    ("2C", "2026-10-21", "maths", "lesson:g2.2.2-addition", ["didnt", "repeat"], None,
     "Addition with carrying was too fast for most of the class. Repeat it on Friday with sticks.",
     ["didnt", "repeat"], None),
    ("2C", "2026-10-21", "english", None, [], None,
     "Reading groups were noisy; the back group did not read at all.", ["didnt"], None),
    ("2C", "2026-10-22", "maths", None, ["idea"], None,
     "Idea: use a market prices game for addition next week.", ["ideas"], None),
    ("2C", "2026-10-23", "maths", "lesson:g2.2.2-addition", ["worked"], None,
     "Repeated addition with sticks. Most of them got it this time.", ["worked", "repeat"], None),
    ("Grade 3 Blue", "2026-10-19", "english", None, ["didnt"], None,
     "Reading aloud: many learners read word by word without expression.", ["didnt"], None),
    ("Grade 3 Blue", "2026-10-19", "english", None, ["attention"], "class",
     "Half the class has no reader at home, they need more practice time in school.", ["attention"], None),
    ("Grade 3 Blue", "2026-10-20", "maths", None, ["worked"], None,
     "Place value with bundles of sticks worked, they could show 34 as 3 tens and 4 ones.", ["worked"], None),
    ("Grade 3 Blue", "2026-10-21", "science", None, ["didnt"], None,
     "The water experiment did not work because we had no clear bottles.", ["didnt"], None),
    ("Grade 3 Blue", "2026-10-21", None, None, ["attention"], "learner",
     "Chipo and Gift were absent three days this week; check with the head teacher.", ["attention"], None),
    ("Grade 3 Blue", "2026-10-22", "english", None, ["idea"], None,
     "Try pairing strong readers with weak readers for ten minutes every morning.", ["ideas"], None),
    ("Grade 3 Blue", "2026-10-23", "maths", None, ["repeat"], None,
     "Go over subtraction with borrowing again on Monday.", ["repeat"], None),
    # Tonga: real teacher lines in the working copy; PUBLIC COPY: replaced by a marker (speaker-written Tonga is not published)
    ("2C", "2026-10-20", "maths", "lesson:g2.2.1-counting", ["worked"], None,
     "[toi withheld: speaker-written] 1", ["worked"], "Slowly. One stone, one number. (T01)"),
    ("2C", "2026-10-22", "maths", None, ["idea"], None,
     "[toi withheld: speaker-written] 2", ["ideas"], "Check your answer by counting again. (T01)"),
    ("Grade 3 Blue", "2026-10-22", "english", None, ["idea"], None,
     "[toi withheld: speaker-written] 3", ["ideas"], "Say it again so the back row hears. (T09)"),
    ("Grade 3 Blue", "2026-10-21", "english", None, ["idea"], None,
     "[toi withheld: speaker-written] 4", ["ideas"],
     "Use your talking voice, not your playground voice. (T09)"),
    ("2C", "2026-10-23", "science", None, ["worked"], None,
     "[toi withheld: speaker-written] 5", ["worked"], "Water and soap, then rinse. (T09)"),
]
SECTION_OF_GOLD = {"worked": "worked", "didnt": "didnt", "ideas": "ideas", "attention": "attention",
                   "repeat": "repeat"}


def build(mem, tt):
    mem.get_or_create(TEACHER)
    cls = {"2C": mem.add_class(TEACHER, "2C", [2]), "Grade 3 Blue": mem.add_class(TEACHER, "Grade 3 Blue", [3])}
    gold = {}
    for i, (c, day, subj, item, tags, att, text, g, ten) in enumerate(NOTES, 1):
        nid = f"eval-note-{i:02d}"
        tt.upsert_note(TEACHER, {"id": nid, "class_id": cls[c]["id"], "slot_date": day, "subject": subj, "item": item,
                                 "tags": tags, "attention": att, "text": text,
                                 "updated_at": f"{day}T12:{i:02d}:00Z"})
        gold[nid] = {"gold": g, "text": text, "teacher_en": ten, "class": c}
    return cls, gold


def score(d, gold, raw_res, sid_map):
    """per item: sourced by gold; raw (model) and shown (after checks)"""
    rows = []
    for sec, items in (raw_res or {}).items():
        for it in items or []:
            ids = [sid_map.get(x) for x in it.get("notes") or [] if sid_map.get(x)]
            ok = bool(ids) and (sec == "themes" or any(SECTION_OF_GOLD.get(x) == sec for i in ids
                                                     for x in gold[i]["gold"]))
            rows.append({"stage": "raw", "section": sec, "text": it.get("text"), "notes": ids, "sourced": ok})
    for sec, items in d["sections"].items():
        for it in items:
            ok = bool(it["notes"]) and (sec == "themes" or any(SECTION_OF_GOLD.get(x) == sec for i in it["notes"]
                                                             for x in gold[i]["gold"]))
            rows.append({"stage": "shown", "section": sec, "text": it["text"], "notes": it["notes"], "sourced": ok})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args(argv)
    from core.memory import SQLiteMemory  # noqa: PLC0415
    from core.timetable import TimetableStore  # noqa: PLC0415
    from plugins import education  # noqa: PLC0415
    from plugins.education import notes_digest as ND  # noqa: PLC0415
    import beelink_lock  # noqa: PLC0415
    os.makedirs(a.out, exist_ok=True)
    P, _ = education.load()
    mem = SQLiteMemory(":memory:")
    tt = TimetableStore(mem)
    cls, gold = build(mem, tt)
    gem = ND.Gemma()
    tr = ND.Translator()
    log = []
    job = ND.DigestJob(P, mem, tt, model=gem, translator=tr, log=lambda *x: log.append(" ".join(map(str, x))))
    t_wait = time.time()
    if not a.no_lock:
        beelink_lock.acquire("notes-digest")
    t0 = time.time()
    res = {"started": dt.datetime.now().isoformat(timespec="seconds"), "lock_wait_s": round(t0 - t_wait),
           "model": gem.name, "translator": tr.name, "digests": [], "translations": []}
    try:
        res["gemma_started_by_us"] = not gem.up()
        if not gem.start():
            raise SystemExit("Gemma did not start")
        # translations: the served route (memory on, as the job uses) and the model alone (memory off)
        for nid, g in gold.items():
            if g["teacher_en"]:
                served, _ = tr.translate(g["text"])
                r_off = tr.svc().translate(g["text"], "toi>en", use_memory=False)
                res["translations"].append({"id": nid, "toi": g["text"], "teacher_en": g["teacher_en"],
                                            "served": served, "model_only": r_off.get("text"),
                                            "model_only_alarms": r_off.get("alarms")})
        for c in cls.values():
            t1 = time.time()
            d = job.digest(TEACHER, c["id"], "2026-10-19")
            raw = json.loads(d.get("raw") or "{}") if d.get("raw") else {}
            sid_map = {v: k for k, v in d["note_refs"].items()}
            rows = score(d, gold, raw, sid_map)
            res["digests"].append({"class": c["label"], "seconds": round(time.time() - t1, 1), "digest": d,
                                   "items": rows})
    finally:
        gem.stop()
        if not a.no_lock:
            beelink_lock.release("notes-digest")
    res["seconds"] = round(time.time() - t0)
    res["model_calls_s"] = gem.calls
    res["log"] = log
    raw_rows = [r for d in res["digests"] for r in d["items"] if r["stage"] == "raw"]
    shown = [r for d in res["digests"] for r in d["items"] if r["stage"] == "shown"]
    mt = [m for d in res["digests"] for m in d["digest"]["matches"]]
    dm = [it for d in res["digests"] for s in ("didnt", "attention") for it in d["digest"]["sections"][s]]
    cited = {i for d in res["digests"] for s in d["digest"]["sections"].values() for it in s for i in it["notes"]}
    res["summary"] = {
        "notes": len(NOTES), "tonga_notes": sum(1 for n in NOTES if n[8]),
        "raw_items": len(raw_rows), "raw_unsourced": sum(1 for r in raw_rows if not r["sourced"]),
        "shown_items": len(shown), "shown_unsourced": sum(1 for r in shown if not r["sourced"]),
        "dropped_by_checks": sum(len(d["digest"]["dropped"]) for d in res["digests"]),
        "didnt_attention_items": len(dm), "matched": len(mt),
        "notes_cited": len(cited), "notes_not_summarised": len(NOTES) - len(cited)}
    with open(os.path.join(a.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
