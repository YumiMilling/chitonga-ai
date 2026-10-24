"""Lexical retrieval over a plug-in's items (23 Oct 2026, the Jesuit). The assistant library's stemmer and stop list
(tools/chitonga/assistant/library.py) with IDF over the plug-in's items; deterministic, no model, no vectors at run
time (vectors find neighbours, not answers: 'kilometre' sits next to 'kilogram'). Returns ranked candidates for the
reader; it never decides alone: a candidate must cover enough of the request AND share a word with the item's title.

24 Oct 2026 (the syllabus items): `kinds` picks what is indexed. The DEFAULT is unchanged (card, lesson, pointer), so
the engine's index, the readers' candidate lists and the trained chat round's prompt see exactly what they saw before.
An index built with 'syllabus' among its kinds (the syllabus route, the web search) ranks every syllabus item BELOW
every qualifying card / lesson / pointer: a competence is reference, the material that answers the request comes first.
A syllabus item's title words are its competence AND sub-topic titles ('Grouping Things' is how a teacher names it);
its text adds the topic, the activities and the expected standard.

24 Oct 2026 (sample lessons, build_samples.py): 'sample' joins the DEFAULT kinds, under three rules that keep every
Grade 1-3 answer exactly as it was:
  1 IDF is computed over the items that are NOT samples (n_idf, df), so every card / lesson / pointer scores to the
    same number as before the samples existed (the cover of an item depends only on the query and df);
  2 a sample whose grades are not among the request's grades (a Grade 1-3 teacher; no grade given) ranks after EVERY
    qualifying card / lesson / pointer, so it can only fill a slot nothing else wanted (off_grade());
  3 the readers never pick one: reader_model.candidates() keeps card / lesson / pointer (unchanged), and the rules
    reader takes its retrieval lead only from items that are not off-grade samples (intake.py).
A Grade 4-7 class (whose grade is the sample's) gets its samples ranked as material, with the grade bonus.

24 Oct 2026 (Grade 4 cards, cards_v2_g4.jsonl): a card the build marks `rank_by_grade` (a cards_v2 card for Grades 4+
only) follows the same three rules as a sample: outside the IDF, after all other material for a request whose grades
are not its own, never the rules reader's lead there. Measured reason: without it the new "ten provinces" card took
the frozen test SC05 ("How many provinces are in Zambia", a teacher of unknown grade) from its accepted honest
no-material answer. A Grade 4 class gets these cards first, with the grade bonus.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "assistant")))
import library as L  # noqa: E402

DEFAULT_KINDS = ("card", "lesson", "pointer", "sample")
NO_IDF_KINDS = ("sample",)          # indexed, but never part of the IDF: the base scores stay what they were


def ranked_by_grade(it):
    """a sample lesson, or a card the build marked rank_by_grade (new Grade 4+ cards, 24 Oct 2026)"""
    return it.get("kind") == "sample" or bool(it.get("rank_by_grade"))


def off_grade(it, grades):
    """True for a sample lesson (or a rank_by_grade card) that is not for one of these grades (a Grade 1-3 teacher, or
    no grade known): it is searchable, but ranks below all other material and is never the reader's lead"""
    return ranked_by_grade(it) and not (grades and set(it.get("grades") or []) & set(grades))


def item_text(it):
    bits = [it.get("title", "")]
    if it["kind"] == "card":
        bits += [it["parts"].get("answer", ""), it["parts"].get("idea", "")]
    elif it["kind"] == "lesson":
        bits += [it.get("scope") or ""] + [k["en"] for k in it.get("key_words", [])]
    elif it["kind"] == "pointer":
        bits += [it.get("ref") or ""]
    elif it["kind"] == "syllabus":
        bits += [(it.get("sub_topic") or {}).get("title", ""), (it.get("topic") or {}).get("title", ""),
                 it.get("expected_standard") or ""] + list(it.get("learning_activities") or [])
    elif it["kind"] == "sample":
        bits += [it.get("label_short") or "", it.get("teaches") or ""] + [k["en"] for k in it.get("key_terms") or []]
    return " ".join(bits)


def title_text(it):
    if it["kind"] == "syllabus":
        return it.get("title", "") + " " + (it.get("sub_topic") or {}).get("title", "")
    if it["kind"] == "sample":
        return it.get("title", "") + " " + (it.get("label_short") or "")
    return it.get("title", "")


class Index:
    def __init__(self, plugin, kinds=DEFAULT_KINDS):
        self.kinds = tuple(kinds)
        self.items = [it for it in plugin.items.values() if it["kind"] in self.kinds
                      and it.get("role") != "set_intro"]
        self.words = {it["id"]: set(L.content_words(item_text(it))) for it in self.items}
        self.title = {it["id"]: set(L.content_words(title_text(it))) for it in self.items}
        self.df = {}
        self.n_idf = 0
        for it in self.items:
            if it["kind"] in NO_IDF_KINDS or it.get("rank_by_grade"):
                continue
            self.n_idf += 1
            for w in self.words[it["id"]]:
                self.df[w] = self.df.get(w, 0) + 1

    def idf(self, w):
        return math.log((1 + self.n_idf) / (1 + self.df.get(w, 0))) + 1.0

    def search(self, query, k=5, grades=None, min_cover=0.5):
        """[(score, cover, title_hits, item id)] best first, only items that qualify (cover >= min_cover and a title
        word shared); grades: prefer items for these grades (a +0.2 bonus, never a filter: a Grade 6 card can still
        answer a teacher's question). Syllabus items, when indexed, come after all other qualifying items; a sample
        lesson not for these grades (off_grade) comes after every other card / lesson / pointer, before the syllabus."""
        q = list(dict.fromkeys(L.content_words(query)))
        if not q:
            return []
        total = sum(self.idf(w) for w in q)
        out = []
        for it in self.items:
            ws = self.words[it["id"]]
            cover = sum(self.idf(w) for w in q if w in ws) / total
            th = sum(1 for w in q if w in self.title[it["id"]])
            if cover >= min_cover and th >= 1:
                bonus = 0.2 if grades and set(it.get("grades") or []) & set(grades) else 0.0
                bonus += 0.1 if it["kind"] == "lesson" else 0.0
                tier = 2 if it["kind"] == "syllabus" else 1 if off_grade(it, grades) else 0
                out.append((round(cover + 0.25 * th + bonus, 3), round(cover, 3), th, it["id"], tier))
        out.sort(key=lambda r: (r[4], -r[0]))
        return [r[:4] for r in out[:k]]
