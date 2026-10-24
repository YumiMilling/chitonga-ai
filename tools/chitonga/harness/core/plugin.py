"""Plug-in loader and validator (23 Oct 2026, the Jesuit). A plug-in is DATA (DESIGN.md): a folder with manifest.json
and the files it names. A plug-in that fails validation does not load (PluginError with every problem listed): a menu
pointing at an item that does not exist, a Tonga line without an origin, a test set edited after it was frozen, a Tonga
intake keyword that no glossary entry or box line carries (no invented words, even in the matcher), a card on hold that
the rules do not route.

24 Oct 2026 (architecture review items 6, 9, 10): the menu may be graded (`{"grades": {g: {subjects: [...]}}}`, education)
or flat (`{"topics": [{id, label, units}]}`, any plug-in without grades; units then carry grade None); the core's fixed
lines live in core/templates_core.json and a plug-in's templates.json overrides keys instead of copying them; the
origins a line may carry are the manifest's own list (plus the core ones); `test_only` plug-ins (a manifest flag) load
only into an engine built with allow_test_plugins=True; `memory_unit` says whether the plug-in keeps class memory
("class", the default) or none (null).
"""
import hashlib
import json
import os
import re

from .fixed import FixedLines

REQUIRED = ("items", "menu", "templates", "rules", "intake", "glossary", "testset")
# 24 Oct 2026: 'syllabus' = one specific competence of the official syllabus (education: build_syllabus.py). It is
# reference, not material: never in a menu unit, never delivered as a lesson, outside the default retrieval index
# (core/retrieval.py DEFAULT_KINDS), so readers and the trained chat round see the same candidates as before.
# 24 Oct 2026 (later): 'sample' = a SAMPLE LESSON from the Funda teacher guides (education: build_samples.py) for the
# grades the plug-in has no material for (4-7): English guide (origin F, ai_draft), its placement in the 2013 / 2023
# curriculum, key terms with checked Tonga or tonga_needed. No Tonga lines, never in the readers' candidates
# (reader_model keeps card / lesson / pointer), ranked below Grade 1-3 material unless the class's grade is its own.
ITEM_KINDS = ("card", "lesson", "pointer", "route", "syllabus", "sample")
SAMPLE_CURRICULA = ("2013", "2023")
CORE_ORIGINS = ("B", "L", "D")              # book, lexicon, derived; teacher ids (T01, T09 ...) come from the manifest
CORE_TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates_core.json")
_core_tp = None


def core_templates():
    global _core_tp
    if _core_tp is None:
        with open(CORE_TEMPLATES, encoding="utf-8") as f:
            _core_tp = json.load(f)
    return _core_tp


class PluginError(Exception):
    pass


def _jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _j(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Plugin:
    def __init__(self, data_dir, hooks=None, fixed=None):
        self.dir = data_dir
        self.hooks = hooks
        self.fixed = fixed if fixed is not None else FixedLines()   # the core's Tonga for the fixed lines (fixed.py)
        self.manifest = _j(os.path.join(data_dir, "manifest.json"))
        f = self.manifest["files"]
        missing = [k for k in REQUIRED if k not in f]
        if missing:
            raise PluginError(f"manifest misses files: {missing}")
        self.name = self.manifest["name"]
        self.items = {i["id"]: i for i in _jl(os.path.join(data_dir, f["items"]))}
        self.menu = _j(os.path.join(data_dir, f["menu"]))
        tp = _j(os.path.join(data_dir, f["templates"]))
        core = core_templates()
        self.templates = dict(core["lines"], **(tp.get("lines") or {}))          # the plug-in overrides, never copies
        self.button_titles = dict(core["buttons"], **(tp.get("buttons") or {}))
        self.label = self.manifest.get("label") or self.name
        # what the conversation state remembers (review item 5): the version AND the build time, because a rebuild
        # keeps the version number but may drop or rename item / unit ids
        self.fingerprint = f"{self.manifest.get('version')}@{self.manifest.get('built') or '-'}"
        self.test_only = bool(self.manifest.get("test_only"))
        self.memory_unit = self.manifest.get("memory_unit", "class")
        self.origins = tuple(dict.fromkeys(list(CORE_ORIGINS) + [k for k in (self.manifest.get("origins") or {})
                                                                  if k.startswith("T")]))
        if not any(o.startswith("T") for o in self.origins):
            self.origins += ("T01", "T09")   # a manifest without an origins list: the teachers of 23 Oct
        self.rules = _j(os.path.join(data_dir, f["rules"]))
        self.intake = _j(os.path.join(data_dir, f["intake"]))
        self.glossary = _jl(os.path.join(data_dir, f["glossary"]))
        self.strategies = {s["lesson"]: s for s in _jl(os.path.join(data_dir, f["strategies"]))} if "strategies" in f else {}
        tpath = os.path.join(data_dir, f["testset"])
        with open(tpath, "rb") as fh:
            self._tblob = fh.read()
        self.testset = _jl(tpath)
        self.units = {}
        self.unit_of = {}
        self.graded = "grades" in self.menu
        groups = ([(int(g), gm["subjects"]) for g, gm in self.menu["grades"].items()] if self.graded
                  else [(None, self.menu.get("topics") or [])])
        for g, subs in groups:
            for s in subs:
                for u in s["units"]:
                    u = dict(u, subject=s["id"], subject_label=s["label"], grade=g)
                    self.units[u["id"]] = u
                    for it in u["items"]:
                        self.unit_of.setdefault(it, u["id"])
        problems = self.validate()
        if problems:
            raise PluginError(f"{self.name}: {len(problems)} problems: " + "; ".join(problems[:20]))

    # ------------------------------------------------------------------------------------------------ validation
    def validate(self):
        p = []
        for iid, it in self.items.items():
            if it.get("kind") not in ITEM_KINDS:
                p.append(f"{iid}: kind {it.get('kind')}")
            if not it.get("title"):
                p.append(f"{iid}: no title")
            if it.get("kind") == "lesson":
                for ln in it.get("lines", []):
                    if ln.get("served") and ln.get("toi") and not ln.get("origin"):
                        p.append(f"{iid} line {ln.get('n')}: Tonga without an origin")
                    if ln.get("served") and ln.get("toi") and not set(ln["origin"]) <= set(self.origins):
                        p.append(f"{iid} line {ln.get('n')}: unknown origin {ln['origin']}")
                for k in it.get("key_words", []):
                    if k.get("toi") and not k.get("origin") and not k["toi"].startswith("("):
                        p.append(f"{iid} key word {k.get('en')}: no origin")
            if self.test_only and not it.get("test_only"):
                p.append(f"{iid}: a test-only plug-in's item must say test_only")
            if it.get("hold") and self.rules.get("holds", {}).get("action") not in ("referral", "no_material"):
                p.append(f"{iid}: on hold but the rules do not route holds")
            if it.get("kind") == "syllabus":
                if not it.get("code") or not it.get("grades"):
                    p.append(f"{iid}: a syllabus item needs a code and a grade")
                for ln in it.get("links") or []:
                    if ln.get("id") not in self.items or self.items[ln["id"]].get("kind") == "syllabus":
                        p.append(f"{iid}: link to {ln.get('id')}, which is no card / lesson in the plug-in")
                if bool(it.get("covered")) != any(ln.get("served") for ln in it.get("links") or []):
                    p.append(f"{iid}: 'covered' disagrees with its links")
            if it.get("kind") == "sample":
                p += self._validate_sample(iid, it)
        # 24 Oct 2026, Chris: "Nothing is held back, just label it for review." needs_review must carry its tag and a
        # review_note (the line every front-end shows); a content hold in the data is refused (it would hide material)
        for iid, it in self.items.items():
            if it.get("needs_review") and not (it.get("review_note") and it.get("review_tag")):
                p.append(f"{iid}: needs_review without its review_tag and review_note")
            if it.get("hold"):
                p.append(f"{iid}: a content hold ({str(it['hold'])[:40]}); label it needs_review instead")
        for uid, u in self.units.items():
            for it in u["items"]:
                if it not in self.items:
                    p.append(f"menu unit {uid}: item {it} does not exist")
                elif self.items[it].get("kind") == "syllabus":
                    p.append(f"menu unit {uid}: {it} is a syllabus item, not material")
                elif self.items[it].get("kind") == "sample" and not (u.get("sample") and u.get("curriculum")
                                                                      in SAMPLE_CURRICULA):
                    p.append(f"menu unit {uid}: {it} is a sample lesson; the unit must say sample and its curriculum")
            if len(u["label"]) > 24:
                p.append(f"menu unit {uid}: label over 24 characters")
        for uid in self.intake.get("units", {}):
            if uid not in self.units:
                p.append(f"intake: unit {uid} not in the menu")
        known = self.tonga_vocabulary()
        for uid, kw in self.intake.get("units", {}).items():
            for w in kw.get("toi", []):
                lw = w.lower()
                if lw not in known and not any(lw in k for k in known if len(lw) >= 4):   # a stem inside a known word
                    p.append(f"intake {uid}: Tonga keyword '{w}' is in no glossary entry or box line")
        for name in ("greet_new", "task_offer", "ask_subject", "unclear", "no_material"):
            if name not in self.templates:
                p.append(f"template {name} missing")
        sha = hashlib.sha256(self._tblob).hexdigest()
        if self.manifest.get("testset_sha256") and sha != self.manifest["testset_sha256"]:
            p.append("testset.jsonl changed after it was frozen (sha256 differs from the manifest)")
        for t in self.testset:
            if not t.get("frozen") or t.get("train") is not False:
                p.append(f"test {t.get('id')}: not frozen / not held out")
        return p

    def _validate_sample(self, iid, it):
        """a sample lesson: labelled, English-only material with origin F; Tonga only in key terms, each with its origin
        and source, or marked tonga_needed with no Tonga at all (no invented words, not even a placeholder)"""
        p = []
        if it.get("lines"):
            p.append(f"{iid}: a sample lesson carries no Tonga lines")
        if it.get("origin") != "F" or not it.get("label"):
            p.append(f"{iid}: a sample lesson needs origin F and its label")
        if it.get("role") != "set_intro":
            if not it.get("grades"):
                p.append(f"{iid}: a sample lesson needs its grades")
            if it.get("curriculum") not in SAMPLE_CURRICULA or not it.get("level"):
                p.append(f"{iid}: a sample lesson needs its curriculum (2013 / 2023) and level")
            for pl in it.get("placements") or []:
                if pl.get("curriculum") not in SAMPLE_CURRICULA:
                    p.append(f"{iid}: placement curriculum {pl.get('curriculum')}")
                if pl.get("grade") == 7 and pl.get("curriculum") != "2013":
                    p.append(f"{iid}: Grade 7 exists only in the 2013 curriculum")
        for k in it.get("key_terms") or []:
            if k.get("tonga_needed"):
                if k.get("toi"):
                    p.append(f"{iid} key term {k.get('en')}: marked tonga_needed but carries Tonga")
            elif not (k.get("toi") and k.get("origin") and k.get("source")):
                p.append(f"{iid} key term {k.get('en')}: Tonga without an origin and source, and not marked needed")
            elif not (k["origin"] == "B" or str(k["origin"]).startswith("T")):
                p.append(f"{iid} key term {k.get('en')}: origin {k['origin']} (only teacher or book)")
        return p

    def tonga_vocabulary(self):
        """every Tonga word the plug-in's teacher / book / lexicon material carries (glossary forms, box lines, key
        words, test items' Tonga): the matcher may only use these"""
        words = set()

        def add(s):
            for w in re.findall(r"[A-Za-zŋŊ']+", s or ""):
                words.add(w.lower())
            if s:
                words.add(s.lower().strip())
        for g in self.glossary:
            for f in g.get("toi") or []:
                add(f)
            add(g.get("toi_text"))
        for it in self.items.values():
            for ln in it.get("lines", []):
                add(ln.get("toi"))
            for k in it.get("key_words", []):
                add(k.get("toi"))
        return words

    # ------------------------------------------------------------------------------------------------ access
    def t(self, _key, _lang="en", _drafts=False, **slots):
        """a fixed line in the teacher's language (en / toi / both); the Tonga from the core table (fixed.py)"""
        line = self.templates.get(_key) or {}
        return self.fixed.say(_key, line.get("en") or "", _lang, _drafts, **slots)

    def button(self, key, lang="en", drafts=False):
        return self.fixed.button(key, self.button_titles.get(key, key), lang, drafts)

    def grade_menu(self, grade):
        return self.menu["grades"].get(str(grade)) if self.graded else None

    def grades(self):
        return sorted(int(g) for g in self.menu["grades"]) if self.graded else []

    def subjects(self, grade=None):
        """a graded menu: the grade's subjects; a flat menu: its topics (grade ignored)"""
        if not self.graded:
            return self.menu.get("topics") or []
        gm = self.grade_menu(grade)
        return gm["subjects"] if gm else []

    def subject(self, grade, sid):
        return next((s for s in self.subjects(grade) if s["id"] == sid), None)

    def units_with_content(self, grade, sid):
        s = self.subject(grade, sid)
        return [u for u in (s["units"] if s else []) if u["items"]]

    def next_unit(self, unit_id):
        """the next unit WITH content after this one in the same subject's menu order, or None"""
        u = self.units.get(unit_id)
        if not u:
            return None
        s = self.subject(u["grade"], u["subject"])
        ids = [x["id"] for x in s["units"]]
        i = ids.index(unit_id)
        for x in s["units"][i + 1:]:
            if x["items"]:
                return self.units[x["id"]]
        return None
