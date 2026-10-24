"""Education plug-in hooks: the two routes that are code, not data (23 Oct 2026, the Jesuit). Reused, not rewritten:

  route:language  tools/chitonga/assistant/lexq.py: the months (lexicon rows from the readers), word senses and
                  differences, word parts (engine2). No model. Data lines (Tonga) are copied as the lexicon has them.
  route:marking   tools/chitonga/assistant/marking.py: mark scheme out of N, word check of a pupil's Tonga (attest; the
                  nearest lexicon word as a question, never a correction), word-parts rubric. Fixed English wording.

The harness serves these in English with the Tonga data lines; rendering the English frame sentences into Tonga is the
old pipeline's job (tools/chitonga/assistant/pipeline.py, translator + checks) and is not repeated here.

24 Oct 2026 (architecture review item 7): the education intents moved here from core/intake.py: `detect()` is asked
by the core's RulesReader before its generic rules and returns a reading (language / marking route, class struggle)
or None. The struggle phrasings are classroom English, so they belong to this plug-in, not to the core.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.normpath(os.path.join(HERE, "..", "..", "..", "assistant"))
LLM = os.path.normpath(os.path.join(HERE, "..", "..", "..", "llm"))
for p in (ASSIST, LLM):
    if p not in sys.path:
        sys.path.insert(0, p)


class AttestOnly:
    """the one backend call marking.word_check makes (attest.line), without loading any model"""
    cfg = {}

    def words(self, toi):
        import attest  # noqa: PLC0415
        return attest.line(toi)


# moved verbatim from core/intake.py (24 Oct): classroom English, this plug-in's phrasings
STRUGGLE = re.compile(r"(?i)\b(struggl\w*|can'?t|cannot|can\s+not|unable\s+to|not\s+able\s+to|don'?t\s+understand|"
                      r"do\s+not\s+understand|find\s+it\s+(hard|difficult)|keep\s+\w+ing|always\s+(get|write|say)\s+\w+\s+wrong|confus\w*|"
                      r"mix(es|ing)?\s+up|are\s+stuck|still\s+(can'?t|cannot|struggle)|fail(s|ing)?\s+to|weak\s+(in|at)|"
                      r"behind\s+in)\b")
HOWTO = re.compile(r"(?i)^\W*(how\s+(do|can|should)\s+i\s+(teach|explain)|give\s+me|i\s+need\s+a)\b")
# 24 Oct 2026 (the syllabus items): a request ABOUT the syllabus. Strong words name it ("competence for sorting things",
# "Grade 1 English syllabus"); soft cues ask what is planned ("what should grade 2 do in maths this term", "week 5 of
# term 2") and count only when no unit was matched by an English keyword and it is not a class struggle. A grade outside
# 1-3 is not ours to answer from this syllabus (held-out H27, "It is in our grade 4 syllabus", is a snake-bite card).
SYLLABUS_STRONG = re.compile(r"(?i)\b(syllab\w*|curricul\w*|competenc\w*|learning\s+activit\w*|expected\s+standards?|"
                             r"scheme\s+of\s+work|specific\s+outcomes?)\b")
SYLLABUS_SOFT = re.compile(r"(?i)\bwhat\s+(should|do|must|will|does|shall)\b.{0,50}?\b(do|learn|cover|teach|study)\b"
                           r".{0,40}?\b((this|next|current)\s+term|term\s*[123]|week\s*\d{1,2})\b|"
                           r"\bwhat'?s\s+(planned|next)\s+(for|in)\b.{0,30}\b(term|week)\b|"
                           r"\bweek\s*\d{1,2}\s+of\s+term\s*[123]\b|\bplanned\s+for\s+(this|next)\s+(term|week)\b")


class Hooks:
    # this plug-in's own intents (the core accepts them in validate_reading); 'syllabus' added 24 Oct 2026
    INTENTS = ("language", "marking", "syllabus")

    def __init__(self, en_vocab=None):
        self._en = en_vocab

    def detect(self, t, r, P, units, strong, use_lexq=True, en_vocab=None):
        """a reading for the education intents, or None (the core's generic rules go on). Same order and confidence
        as when this lived in core/intake.py: language route (lexq), marking route, class struggle."""
        if use_lexq:
            try:
                import lexq  # noqa: PLC0415
                det = lexq.detect(t, "", en_vocab() if en_vocab else self.en_vocab())
            except Exception as e:  # noqa: BLE001
                det, r["why"] = None, f"lexq unavailable: {e}"
            if det and (det["intent"] == "months" or not strong) and "route:language" in P.items:
                return dict(r, intent="language", item="route:language", candidates=["route:language"],
                            confidence=0.85, source="rules:lexq", why=f"lexq {det['intent']}",
                            subject="chitonga")
        syl = self.syllabus_reading(t, r, P, strong)
        if syl:
            return syl
        try:
            import marking  # noqa: PLC0415
            mk = marking.choose(t, "")
        except Exception:  # noqa: BLE001
            mk = None
        # 24 Oct 2026: `strong` counts English keyword hits only, so a Tonga classroom line that names its lesson
        # (a digraph lesson line) was taken by the word-check template before the
        # Tonga keywords were read. A unit matched through the teachers' Tonga words counts as placed too; the
        # language route above keeps its own rule (a word question about a lesson word is still a word question).
        toi_hit = bool(units and units[0][1].get("toi"))
        if mk and not strong and not toi_hit and not STRUGGLE.search(t) and "route:marking" in P.items:
            return dict(r, intent="marking", item="route:marking", candidates=["route:marking"], confidence=0.85,
                        source="rules:marking", why=f"marking template {mk}")
        if STRUGGLE.search(t) and not HOWTO.search(t):
            u = units[0][0] if units else None
            return dict(r, intent="struggle", unit=u, subject=r.get("subject") or (P.units[u]["subject"] if u else None),
                        confidence=0.85 if u else 0.7, source="rules:struggle",
                        why="a class struggle" + (f" in unit {u}" if u else ""))
        return None

    def syllabus_reading(self, t, r, P, strong):
        """the syllabus route's reading, or None (see SYLLABUS_STRONG / SYLLABUS_SOFT above)"""
        if "route:syllabus" not in P.items:
            return None
        from core.intake import grades_in  # noqa: PLC0415
        if any(g not in (1, 2, 3) for g in grades_in(t)):
            return None
        if STRUGGLE.search(t) and not HOWTO.search(t):
            return None       # "most cannot reach the expected standard for reading" is a class struggle first
        hard = bool(SYLLABUS_STRONG.search(t))
        soft = bool(SYLLABUS_SOFT.search(t)) and not strong
        if not (hard or soft):
            return None
        return dict(r, intent="syllabus", item="route:syllabus", candidates=["route:syllabus"],
                    confidence=0.85 if hard else 0.8, source="rules:syllabus",
                    why="names the syllabus" if hard else "asks what is planned")

    def en_vocab(self):
        if self._en is None:
            try:
                import h2_text  # noqa: PLC0415
                self._en = frozenset(h2_text._en_vocab())
            except Exception:  # noqa: BLE001
                self._en = frozenset()
        return self._en

    def run(self, route_id, text, pref="en", ctx=None):
        """{'body', 'origins', 'flags'} or None. ctx (24 Oct 2026, for route:syllabus): {'plugin', 'grades' (the class's),
        'now'}; the older routes ignore it."""
        if route_id == "route:syllabus":
            ctx = ctx or {}
            P = ctx.get("plugin")
            if P is None:
                return None
            from . import syllabus as SY  # noqa: PLC0415
            return SY.for_plugin(P).answer(text, grades=ctx.get("grades"), now=ctx.get("now"))
        if route_id == "route:language":
            import lexq  # noqa: PLC0415
            det = lexq.detect(text, "", self.en_vocab())
            a = lexq.answer(det, text) if det else None
            if not a or not a.get("parts"):
                return None
            lines = [s for _, ss in a["parts"] for s in ss]
            return {"body": "\n".join(lines) + "\n[L: from our word list and grammar tables; English frame]",
                    "origins": ["L"], "flags": list(a.get("flags") or [])}
        if route_id == "route:marking":
            import marking  # noqa: PLC0415
            name = marking.choose(text, "")
            t = marking.fill(name, text, "", AttestOnly()) if name else None
            if not t:
                return None
            lines = [s for _, ss in t["parts"] for s in ss if ss]
            return {"body": "\n".join(lines) + "\n[template: fixed English wording]", "origins": ["template"],
                    "flags": list(t.get("flags") or [])}
        return None
