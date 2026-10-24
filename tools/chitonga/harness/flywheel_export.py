"""The loop out of the flywheel: the day's teacher inputs -> training / test material, with the held-out rules kept
(24 Oct 2026, the Jesuit; FLYWHEEL.md; Chris: "I want to adjust the harness for the flywheel, getting inputs from the
teachers").

    python -X utf8 tools/chitonga/harness/flywheel_export.py [--day 2026-10-27 | --all] [--base DIR]
                                                              [--include-testers] [--no-attest]
Runs by itself once a night from the shadow / live loop (run.py nightly_export, after 23:00 Lusaka), or by hand
(run.py export). It READS pilot_logs/flywheel.jsonl, pilot_logs/pilot_labels.jsonl, the allow-list and the control file;
it never edits them. It WRITES flywheel_out/<day>/ (or flywheel_out/_TEST_ONLY_<day>/ with --include-testers):

  reader_sft.jsonl          a request the teacher's own answer labelled (Yes on "You want X?", a pick after No / Not
                            this one, a retype that confirmed): text, candidates, target {intent, item, read}; strong /
                            medium / weak tiers; chosen_in_candidates false = a retrieval miss, not a reading error
  reader_negatives.jsonl    a reading the teacher rejected (No / Not this one): text, candidates, the wrong item
  translator_pairs.jsonl    teacher Tonga with its English: a fix of a served line, a line written for a derived or
                            held-back line, a derived line confirmed "Right as is" (same weight as authored, Chris
                            24 Oct), a unit line with the teacher's own English. weight 'authored'. A line with a
                            suspected slip (a word attest.py cannot place) is NOT a Tonga target: it is kept as
                            direction 'toi>en' only, weight 'noisy_input' (DECISIONS 24 Oct)
  monolingual.jsonl         a unit line written without English (Tonga text, origin, unit)
  lexicon_candidates.jsonl  words a teacher wrote that our sources do not place, and word swaps between our line and
                            the teacher's (ours -> theirs), each a HYPOTHESIS with its context and the number of other
                            box lines that carry our form (one to many, by verification: never copied blindly)
  lesson_box_updates.jsonl  per box line, ALL teacher evidence so far (cumulative): versions, confirmations, OKs, by
                            speaker; status teacher_1 (one speaker: confidence 1, used, labelled) / supported (a second
                            speaker agrees) / disputed (speakers differ: both kept, never overwritten)
  content_backlog.json      cumulative: every honest 'no material' answer, by unit, request text and grade, with counts
                            and teachers; health / status gaps as counts only
  controls_scores.json      cumulative, per teacher: known-wrong lines caught / missed, known-right agreed / rewrote
  excluded.jsonl            every record kept OUT of training, with the reason (held out, guard hit, tester, no
                            consent, control, slip as target)
  SUMMARY.md                what moved today, in numbers
THE HELD-OUT RULES (asserted before anything is written, and again on the written rows):
  - a reader label whose request is held out (the first 40 real requests per teacher: test-then-train) or whose held-out
    status is unknown (null) never enters reader_sft / reader_negatives;
  - any row whose English or Tonga (served or written) hits the held-out guard (tools/chitonga/train/heldout_guard.py,
    incl. the harness v14 test) or the stage 18 frozen teacher test / dev slice never enters any training file;
  - controls are never training rows; testers' records (Chris) are never training rows (only with --include-testers,
    into a folder named _TEST_ONLY_, for a pipeline test);
  - no consent recorded in the allow-list = no training row.
"""
import argparse
import collections
import csv
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
RUN = os.path.join(CT, "runs", "2026-10-23_final_translator_data")
FROZEN = [os.path.join(RUN, f) for f in ("teacher_test_final18.jsonl", "teacher_dev_final18.jsonl")]
ZAMBIA = datetime.timezone(datetime.timedelta(hours=2))
SLIP_VERDICTS = ("unattested", "contact", "not_tonga")
READER_POS = ("confirm_yes", "correction_pick", "correction_typed", "accepted")
READER_NEG = ("confirm_no", "not_this")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", (s or "").lower())).strip()


def lusaka_day(at):
    try:
        return datetime.datetime.fromisoformat(str(at).replace("Z", "+00:00")).astimezone(ZAMBIA).date().isoformat()
    except ValueError:
        return str(at)[:10]


def _lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


class FrozenGuard:
    """the held-out guard plus the stage 18 frozen teacher test and dev (the guard itself does not list those two)"""

    def __init__(self, hg=None, frozen_paths=FROZEN, extra=()):
        self.hg = hg
        self.keys = set(norm(x) for x in extra if x)
        for p in frozen_paths:
            for r in _lines(p):
                for s in [r.get("en"), r.get("toi")] + list(r.get("alternatives") or []):
                    if isinstance(s, str) and s.strip():
                        self.keys.add(norm(s))

    def hit(self, toi=None, en=None):
        if any(norm(s) in self.keys for s in (toi, en) if s):
            return "frozen teacher test / dev"
        if self.hg is not None:
            w = self.hg.which(toi, en)
            if w:
                return f"held-out guard: {w}"
        return None


def load_guard(run_dir=RUN):
    sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "train"))
    import heldout_guard as HG  # noqa: PLC0415
    return FrozenGuard(HG.load(run_dir))


def default_attest():
    sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "llm"))
    import attest  # noqa: PLC0415
    return attest.line


def toks(s):
    return re.findall(r"[a-zŋ']+", (s or "").lower())


class Exporter:
    def __init__(self, base, guard, attest_line, allow, plugin_lines=None):
        """guard: FrozenGuard (or anything with hit(toi, en)); attest_line(text) -> [(verdict, word, note)] (None =
        no slip check: every word is taken as placed; the summary says so); allow: {code: allow-list entry};
        plugin_lines: [(item, n, toi)] of the lesson boxes for the one-to-many count"""
        self.base, self.guard, self.attest, self.allow = base, guard, attest_line, allow
        self.plugin_lines = plugin_lines or []
        self.out = collections.defaultdict(list)

    # ------------------------------------------------------------------------------------------------ eligibility
    def why_not(self, rec, include_testers):
        e = self.allow.get(rec.get("teacher")) or {}
        if rec.get("tester") or e.get("tester"):
            return None if include_testers else "tester (pipeline test, not a Tonga speaker's data)"
        if not (rec.get("consent") or (e.get("consent") or {}).get("words")):
            return "no consent recorded"
        return None

    def exclude(self, rec, why, what):
        self.out["excluded"].append({"record": rec.get("id"), "teacher": rec.get("teacher"), "what": what, "why": why})

    # ------------------------------------------------------------------------------------------------ reader
    def reader(self, labels, include_testers):
        for x in labels:
            if x.get("kind") not in READER_POS + READER_NEG:
                continue
            why = self.why_not(x, include_testers)
            if why:
                self.exclude(x, why, "reader label")
                continue
            if x.get("held_out") is not False:
                self.exclude(x, "held out (first 40 requests of the teacher)" if x.get("held_out")
                             else "held-out status unknown (treated as held out)", "reader label")
                continue
            text = x.get("request")
            if not text:
                self.exclude(x, "no request text kept (pupil-level guard)", "reader label")
                continue
            g = self.guard.hit(toi=text, en=text)
            if g:
                self.exclude(x, g, "reader label")
                continue
            base = {"id": "rd:" + x["id"], "text": text, "lang": x.get("request_lang"),
                    "candidates": x.get("candidates") or [], "teacher": x.get("teacher"), "label_kind": x["kind"],
                    "label": x.get("label"), "strength": x.get("strength"), "src_label": x["id"],
                    "request_id": x.get("request_id"), "held_out": False, "origin": f"teacher:{x.get('teacher')}",
                    "channel": "assistant"}
            if x["kind"] in READER_NEG:
                self.out["reader_negatives"].append(dict(base, wrong=x.get("proposed")))
                continue
            if x["kind"] == "correction_pick" and x.get("label") != "corrected":
                self.exclude(x, f"label {x.get('label')} (picked the rejected lesson: a person looks)", "reader label")
                continue
            if x["kind"] == "correction_typed" and x.get("strength") == "weak" and x.get("label") != "confirmed_by_retype":
                self.exclude(x, "weak retype (browsing, not a correction)", "reader label")
                continue
            chosen = x.get("chosen")
            if not chosen:
                continue
            item = chosen if not str(chosen).startswith("unit:") else None
            self.out["reader_sft"].append(dict(base, target={
                "intent": "lesson", "item": item, "unit": chosen[5:] if item is None else None,
                "read": text if x.get("request_lang") == "en" else None},
                needs_read_en=x.get("request_lang") != "en",
                chosen_in_candidates=bool(item and item in (x.get("candidates") or []))))

    # ------------------------------------------------------------------------------------------------ Tonga
    def slips(self, toi):
        if self.attest is None:
            return [], []
        items = self.attest(toi) or []
        slips = [(v, w, n) for v, w, n in items if v in SLIP_VERDICTS]
        mixed = [(v, w, n) for v, w, n in items if v == "mixed"]
        return slips, mixed

    def pair(self, rec, en, toi, kind, served=None):
        """one teacher pair -> translator_pairs (target or noisy input) or excluded"""
        g = self.guard.hit(toi=toi, en=en) or (self.guard.hit(toi=(served or {}).get("toi")) if served else None)
        if g:
            self.exclude(rec, g, kind)
            return None
        slips, mixed = self.slips(toi)
        row = {"id": f"fw:{rec['id']}", "en": en, "toi": toi, "kind": kind, "teacher": rec["teacher"],
               "origin": f"teacher:{rec['teacher']}", "channel": "assistant", "consent": rec.get("consent"),
               "item": rec.get("item"), "line": (served or {}).get("n"), "served_toi": (served or {}).get("toi"),
               "served_origin": (served or {}).get("origin"), "src_record": rec["id"], "held_out": False,
               "contact_words": [w for _, w, _ in mixed] or None}
        if slips:
            row.update(direction="toi>en", weight="noisy_input", slip_suspect=[w for _, w, _ in slips])
            self.exclude(rec, "suspected slip " + ", ".join(w for _, w, _ in slips) + ": not a Tonga target, kept "
                         "as noisy input", kind + " (as target)")
            for v, w, note in slips:
                self.lexicon(rec, w, v, note, toi, served)
        else:
            row.update(direction="both", weight="authored", slip_suspect=None)
        self.out["translator_pairs"].append(row)
        return row

    def lexicon(self, rec, word, verdict, note, toi, served, ours=None):
        n_other = sum(1 for _, _, t in self.plugin_lines if ours and ours in toks(t)) if ours else None
        self.out["lexicon_candidates"].append({
            "word": word, "verdict": verdict, "note": note, "teacher": rec["teacher"], "line": toi,
            "served": (served or {}).get("toi"), "item": rec.get("item"), "replaces": ours,
            "box_lines_with_our_form": n_other, "src_record": rec["id"],
            "status": "hypothesis: a new word, a variant or a slip; one teacher; verify before any change"})

    def diffs(self, rec, served_toi, teacher_toi):
        """word swaps ours -> theirs (same position after aligning the common start and end): variant candidates"""
        a, b = toks(served_toi), toks(teacher_toi)
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        j = 0
        while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
            j += 1
        old, new = a[i:len(a) - j], b[i:len(b) - j]
        if 1 <= len(old) <= 2 and len(old) == len(new):
            for o, n in zip(old, new):
                if o != n:
                    self.lexicon(rec, n, "swap", f"teacher wrote '{n}' where our line has '{o}'", teacher_toi,
                                 {"toi": served_toi}, ours=o)

    def flywheel(self, recs, controls, include_testers):
        ctl = {c["id"]: c for c in controls}
        for r in recs:
            if r.get("ev") != "answer":
                continue
            if r.get("control"):
                self.exclude(r, "control item (scored, never trained)", "control answer")
                continue
            why = self.why_not(r, include_testers)
            if why:
                self.exclude(r, why, f"{r['kind']} {r['answer']}")
                continue
            kind, ans, served = r["kind"], r["answer"], r.get("served") or {}
            if kind == "tonga_check" and ans == "fix" or kind in ("derived_confirm", "write_line") and ans == "write":
                text = (r.get("teacher_text") or "").strip()
                if not text:
                    continue
                if served.get("en_is_gloss") or not served.get("en"):
                    # a printed line (its English is our unverified gloss): the teacher's version is a variant of the
                    # print for the box, never a translator pair with that gloss
                    g = self.guard.hit(toi=text)
                    if g:
                        self.exclude(r, g, "print variant")
                    continue
                self.pair(r, served["en"], text, "teacher_fix" if ans == "fix" else "teacher_written", served)
                if served.get("toi"):
                    self.diffs(r, served["toi"], text)
            elif kind == "derived_confirm" and ans == "right":
                if served.get("en") and served.get("toi"):
                    self.pair(r, served["en"], served["toi"], "teacher_ok", served)
            elif kind == "unit_line" and ans == "write":
                toi, en = (served.get("teacher_toi") or r.get("teacher_text") or "").strip(), served.get("teacher_en")
                if not toi:
                    continue
                if en:
                    self.pair(r, en, toi, "teacher_unit_line", None)
                else:
                    g = self.guard.hit(toi=toi)
                    if g:
                        self.exclude(r, g, "unit line")
                        continue
                    slips, _ = self.slips(toi)
                    self.out["monolingual"].append({"id": f"fw:{r['id']}", "toi": toi, "unit": r.get("unit"),
                                                    "unit_label": served.get("unit_label"), "teacher": r["teacher"],
                                                    "origin": f"teacher:{r['teacher']}", "src_record": r["id"],
                                                    "slip_suspect": [w for _, w, _ in slips] or None})
        return ctl

    # ------------------------------------------------------------------------------------------------ cumulative
    @staticmethod
    def box_updates(recs, eligible):
        """per (item, line): every teacher's evidence so far; the status follows the confidence tally (design 000)"""
        box = {}
        for r in recs:
            if r.get("ev") != "answer" or r.get("control") or not eligible(r) or not r.get("item"):
                continue
            kind, ans = r["kind"], r["answer"]
            lines = r.get("lines") if ans == "ok" else [r.get("served") or {}]
            for ln in lines or []:
                if ln.get("n") is None:
                    continue
                k = (r["item"], str(ln["n"]))
                b = box.setdefault(k, {"item": r["item"], "line": str(ln["n"]), "en": ln.get("en"),
                                       "served_toi": ln.get("toi"), "served_origin": ln.get("origin"),
                                       "ok": [], "confirmed": [], "versions": []})
                if ans == "ok":
                    b["ok"].append(r["teacher"])
                elif ans == "right":
                    b["confirmed"].append(r["teacher"])
                elif ans in ("fix", "write") and r.get("teacher_text"):
                    b["versions"].append({"teacher": r["teacher"], "toi": r["teacher_text"].strip(), "at": r["at"],
                                          "record": r["id"]})
        out = []
        for b in box.values():
            agree_served = set(b["ok"]) | set(b["confirmed"])
            texts = collections.defaultdict(set)
            for v in b["versions"]:
                texts[norm(v["toi"])].add(v["teacher"])
            if texts:
                best = max(texts.values(), key=len)
                speakers = set().union(*texts.values())
                if len(texts) > 1 or agree_served - best:
                    status = "disputed"
                elif len(best) >= 2:
                    status = "supported"
                else:
                    status = "teacher_1"
            else:
                status = "supported" if len(agree_served) >= 2 else ("teacher_1" if agree_served else "none")
                speakers = agree_served
            out.append(dict(b, status=status, speakers=sorted(speakers),
                            note="confidence = number of independent speakers; a contradiction is recorded beside, "
                                 "never overwritten; 'supported' may be served verbatim, 'teacher_1' is used and "
                                 "labelled (Chris 20 Oct: usable at once with confidence 1)"))
        return sorted(out, key=lambda x: (x["item"], x["line"]))

    @staticmethod
    def backlog(recs, eligible_or_tester):
        units, reqs, grades, status = {}, {}, collections.Counter(), collections.Counter()
        for r in recs:
            if r.get("ev") != "gap" or not eligible_or_tester(r):
                continue
            if r["what"] == "unit":
                u = units.setdefault(r["ref"], {"unit": r["ref"], "count": 0, "teachers": [], "held_out": 0})
                u["count"] += 1
                u["held_out"] += 1 if r.get("held_out") else 0
                u["teachers"] = sorted(set(u["teachers"]) | {r["teacher"]})
            elif r["what"] == "request":
                k = norm(r.get("text")) or "(no text)"
                q = reqs.setdefault(k, {"text": r.get("text"), "count": 0, "teachers": [], "held_out": 0,
                                        "last": None})
                q["count"] += 1
                q["held_out"] += 1 if r.get("held_out") else 0
                q["teachers"] = sorted(set(q["teachers"]) | {r["teacher"]})
                q["last"] = r["at"]
            elif r["what"] == "grade":
                grades[str(r.get("ref"))] += 1
            elif r["what"] == "status":
                status[r.get("status") or "?"] += 1
        return {"units": sorted(units.values(), key=lambda x: -x["count"]),
                "requests": sorted(reqs.values(), key=lambda x: -x["count"]),
                "grades": dict(grades), "status_topics_counts_only": dict(status),
                "note": "what teachers asked for and we did not have; drives what we build next. A request marked "
                        "held_out is one of the teacher's first 40 (the reader's test): build the CONTENT, never put "
                        "its wording into reader training."}

    @staticmethod
    def control_scores(recs, controls):
        ctl = {c["id"]: c for c in controls}
        per = collections.defaultdict(lambda: collections.Counter())
        for r in recs:
            c = r.get("control")
            if not c:
                continue
            p = per[r["teacher"]]
            if r["ev"] == "prompt":
                p["shown"] += 1
            elif r["ev"] == "ignored":
                p["ignored"] += 1
            elif r["ev"] == "answer":
                p["answered"] += 1
                if c["expect"] == "wrong":
                    p["wrong_caught" if r["answer"] == "write" else "wrong_missed"] += 1
                else:
                    p["right_agreed" if r["answer"] == "right" else "right_rewrote"] += 1
        out = {}
        for t, p in per.items():
            w = p["wrong_caught"] + p["wrong_missed"]
            rr = p["right_agreed"] + p["right_rewrote"]
            out[t] = dict(p, wrong_catch_rate=round(p["wrong_caught"] / w, 2) if w else None,
                          right_agree_rate=round(p["right_agreed"] / rr, 2) if rr else None,
                          n_controls_in_file=len(ctl),
                          note="judge by counts: a handful of controls moves nothing (calibration rule)")
        return out


def search_gaps(path, include_testers=False):
    """24 Oct 2026: what teachers told us they were looking for when the search found nothing called that (the app's
    "Tell us what you were looking for", contribution kind 'search_gap'; web/api.py and the Netlify function). By what
    they typed, with counts, teachers and their own words; testers only with --include-testers. The Netlify demo's
    gaps sit in its blob store (contrib/...) until they are pulled into this file."""
    out = {}
    for r in _lines(path):
        if r.get("kind") != "search_gap" or (r.get("tester") and not include_testers):
            continue
        k = norm(r.get("en")) or "(no text)"
        g = out.setdefault(k, {"typed": r.get("en"), "count": 0, "teachers": [], "wanted": [], "last": None})
        g["count"] += 1
        g["teachers"] = sorted(set(g["teachers"]) | {r.get("teacher")} - {None})
        if r.get("text") and r["text"] not in g["wanted"]:
            g["wanted"].append(r["text"])
        g["last"] = r.get("at")
    return sorted(out.values(), key=lambda x: -x["count"])


def assert_clean(out, guard):
    """the leak asserts, on the rows about to be written"""
    for r in out["reader_sft"] + out["reader_negatives"]:
        assert r["held_out"] is False, r["id"]
        assert not guard.hit(toi=r["text"], en=r["text"]), r["id"]
    for r in out["translator_pairs"]:
        assert not guard.hit(toi=r["toi"], en=r["en"]), r["id"]
        assert not r.get("control"), r["id"]
        if r.get("slip_suspect"):
            assert r["direction"] == "toi>en" and r["weight"] == "noisy_input", r["id"]


def export(base, day=None, include_testers=False, guard=None, attest_line=None, allow=None, controls=None,
           plugin_lines=None, out_root=None, ledger_paths=None):
    """-> (out_dir, summary dict). day None = all days."""
    from adapter.config import HARNESS_DATA, Config  # noqa: PLC0415
    own = os.path.join(base, "pilot_allowlist.json")      # a replay / demo store uses the pilot's allow-list
    cfg = Config(base=base, allowlist=own if os.path.exists(own) else os.path.join(HARNESS_DATA, "pilot_allowlist.json"))
    allow = allow if allow is not None else {e.get("code"): e for e in cfg.allowlist().values()}
    controls = controls if controls is not None else _lines(cfg.controls_path)
    recs_all = _lines(cfg.flywheel_path)
    labels_all = _lines(cfg.labels_path)
    pick = (lambda r: lusaka_day(r.get("at")) == day) if day else (lambda r: True)
    recs, labels = [r for r in recs_all if pick(r)], [x for x in labels_all if pick(x)]
    ex = Exporter(base, guard, attest_line, allow, plugin_lines)
    ex.reader(labels, include_testers)
    ex.flywheel(recs, controls, include_testers)
    assert_clean(ex.out, guard)
    eligible = lambda r: ex.why_not(r, include_testers) is None  # noqa: E731
    boxes = ex.box_updates(recs_all, eligible)
    backlog = ex.backlog(recs_all, lambda r: True if include_testers else not r.get("tester"))
    backlog["search_gaps"] = search_gaps(os.path.join(HARNESS_DATA, "web_logs", "contributions.jsonl"), include_testers)
    scores = ex.control_scores(recs_all, controls)
    tag = day or "all"
    out_dir = os.path.join(out_root or cfg.flywheel_out, ("_TEST_ONLY_" if include_testers else "") + tag)
    os.makedirs(out_dir, exist_ok=True)
    for name in ("reader_sft", "reader_negatives", "translator_pairs", "monolingual", "lexicon_candidates", "excluded"):
        with open(os.path.join(out_dir, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in ex.out[name]:
                f.write(json.dumps(dict(r, test_only=True) if include_testers else r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "lesson_box_updates.jsonl"), "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    for name, obj in (("content_backlog.json", backlog), ("controls_scores.json", scores)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    # pay of the day, from the ledgers (the flywheel's own; the desk's balances.tsv is the main session's)
    pay = collections.Counter()
    for lp in ledger_paths or (cfg.pay_ledger_path(True), cfg.pay_ledger_path(False)):
        if os.path.exists(lp):
            with open(lp, encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    if not day or row["date"] == day:
                        pay[(os.path.basename(lp), row["code"])] += int(row["K"] or 0)
    ev = collections.Counter(r["ev"] + ":" + str(r.get("kind") or r.get("what")) + (":" + r["answer"] if r.get(
        "answer") else "") for r in recs)
    summary = {"day": tag, "records": len(recs), "labels": len(labels), "events": dict(ev),
               "reader_sft": len(ex.out["reader_sft"]), "reader_negatives": len(ex.out["reader_negatives"]),
               "translator_pairs": len(ex.out["translator_pairs"]),
               "pairs_as_targets": sum(1 for r in ex.out["translator_pairs"] if r["weight"] == "authored"),
               "pairs_noisy_input": sum(1 for r in ex.out["translator_pairs"] if r["weight"] == "noisy_input"),
               "monolingual": len(ex.out["monolingual"]), "lexicon_candidates": len(ex.out["lexicon_candidates"]),
               "excluded": dict(collections.Counter(x["why"].split(":")[0] for x in ex.out["excluded"])),
               "box_lines_with_evidence": len(boxes),
               "box_status": dict(collections.Counter(b["status"] for b in boxes)),
               "backlog_top": [(u["unit"], u["count"]) for u in backlog["units"][:5]] +
                              [(q["text"], q["count"]) for q in backlog["requests"][:5]],
               "controls": scores, "pay": {f"{k[1]} ({k[0]})": v for k, v in pay.items()},
               "slip_check": "attest.py" if attest_line else "OFF (no slip check: every word taken as placed)",
               "test_only": include_testers}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(render_summary(summary))
    return out_dir, summary


def render_summary(s):
    L = [f"# Flywheel export {s['day']}{' (TEST ONLY: testers included, never training)' if s['test_only'] else ''}",
         "", f"Records {s['records']}, reader labels {s['labels']}. Slip check: {s['slip_check']}.", "",
         "## What moved", "",
         f"- reader SFT examples: {s['reader_sft']} (negatives {s['reader_negatives']})",
         f"- translator pairs: {s['translator_pairs']} ({s['pairs_as_targets']} as targets at authored weight, "
         f"{s['pairs_noisy_input']} noisy input only)",
         f"- monolingual unit lines: {s['monolingual']}; lexicon candidates: {s['lexicon_candidates']}",
         f"- lesson box lines with teacher evidence (all days): {s['box_lines_with_evidence']} {s['box_status']}",
         f"- kept out of training: {s['excluded']}", "", "## Events", ""]
    L += [f"- {k}: {v}" for k, v in sorted(s["events"].items())]
    L += ["", "## Content backlog, top", ""] + [f"- {a}: {b}" for a, b in s["backlog_top"]]
    L += ["", "## Controls (all days)", ""] + [f"- {t}: {json.dumps(v)}" for t, v in s["controls"].items()]
    L += ["", "## Pay (this export's ledgers; move to desk/balances.tsv when paid)", ""]
    L += [f"- {k}: K{v}" for k, v in s["pay"].items()] or ["- none"]
    return "\n".join(L) + "\n"


def plugin_box_lines():
    try:
        from plugins import education  # noqa: PLC0415
        P, _ = education.load()
        return [(iid, ln.get("n"), ln.get("toi")) for iid, it in P.items.items() for ln in it.get("lines") or []
                if ln.get("toi")]
    except Exception:  # noqa: BLE001
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--include-testers", action="store_true")
    ap.add_argument("--no-attest", action="store_true")
    a = ap.parse_args()
    from adapter.config import HARNESS_DATA  # noqa: PLC0415
    day = None if a.all else (a.day or datetime.datetime.now(ZAMBIA).date().isoformat())
    guard = load_guard()
    attest_line = None if a.no_attest else default_attest()
    out_dir, s = export(a.base or HARNESS_DATA, day, a.include_testers, guard, attest_line,
                        plugin_lines=plugin_box_lines())
    print(render_summary(s))
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
