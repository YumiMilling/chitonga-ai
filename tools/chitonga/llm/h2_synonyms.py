"""Harness v2, synonyms as evidence (18 Oct 2026; TODO_ENGINE_HARNESS E9).

A word pair confirmed as synonyms in internal/chitonga_train/vocab/synonyms_antonyms.tsv with confirmed = book or
teacher is NOT an error when the draft uses one where a reference (a memory line, a glossary term) has the other.
Pivot-unconfirmed pairs (meaning guessed from the English side) and book-options-unconfirmed pairs (the Jesuit picked
the answer) are never used as evidence, either way. Confirmed ANTONYMS go the other way: the draft having one where the
reference has the other is a hard issue (musankwa for musimbi).

Matching: whole words (lower case); for a verb entry in the infinitive (ku- / kw-), the stem as well, so an inflected
verb in the draft (balabandika) meets its entry (kubandika = kudonda) through the engine 2 reader's stems. Multi-word
entries (kukavwu kasubila) are matched as phrases inside the line. The table is read at run time (it grows).
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TABLE = os.path.join(REPO, "internal", "chitonga_train", "vocab", "synonyms_antonyms.tsv")
CONFIRMED = {"book", "teacher"}
WORD = re.compile(r"[a-zŋ']+")
_t = {}


def _stem(w):
    if w.startswith("kw") and len(w) > 4:
        return w[2:]
    if w.startswith("ku") and len(w) > 4:
        return w[2:]
    return None


def load(path=TABLE):
    """{'syn': {key: class id}, 'members': {id: set}, 'ant': {key: set of keys}, 'phrases': [...]} from confirmed rows"""
    if path in _t:
        return _t[path]
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ant, phrases, n = {}, set(), 0
    for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        if (r.get("confirmed") or "").strip() not in CONFIRMED:
            continue
        a, b = (r.get("word_a") or "").strip().lower(), (r.get("word_b") or "").strip().lower()
        if not a or not b:
            continue
        n += 1
        for x in (a, b):
            if " " in x:
                phrases.add(x)
        if r["relation"] == "synonym":
            parent[find(a)] = find(b)
        elif r["relation"] == "antonym":
            ant.setdefault(a, set()).add(b)
            ant.setdefault(b, set()).add(a)
    cls = {x: find(x) for x in list(parent)}
    keys = {}                                   # word or stem -> class id
    for x, c in cls.items():
        keys[x] = c
        s = _stem(x)
        if s and " " not in x:
            keys.setdefault("STEM:" + s, c)
    members = {}
    for x, c in cls.items():
        members.setdefault(c, set()).add(x)
    _t[path] = {"syn": keys, "members": members, "ant": ant, "phrases": sorted(phrases, key=len, reverse=True), "rows": n}
    return _t[path]


_stems = {}


def stems(w):
    """engine 2 verb stems of a word (cached per process; the engine is read, never copied)"""
    if w not in _stems:
        try:
            sys.path.insert(0, os.path.join(REPO, "tools", "chitonga", "grammar2"))
            import reader
            _stems[w] = sorted({x[1] for x in reader.read(w) if x[0] == "verb"})
        except Exception:  # noqa: BLE001
            _stems[w] = []
    return _stems[w]


def cls_of(w, T=None):
    T = T or load()
    w = w.lower()
    if w in T["syn"]:
        return T["syn"][w]
    for s in stems(w):
        if "STEM:" + s in T["syn"]:
            return T["syn"]["STEM:" + s]
    return None


def same(a, b, T=None):
    """True when a and b are the same word or confirmed synonyms (directly or through the class)"""
    if a.lower() == b.lower():
        return True
    ca, cb = cls_of(a, T), cls_of(b, T)
    return ca is not None and ca == cb


def antonyms(a, b, T=None):
    T = T or load()
    a, b = a.lower(), b.lower()
    ka = {a} | {x for x in T["ant"] if " " not in x and _stem(x) and _stem(x) in stems(a)}
    kb = {b} | {x for x in T["ant"] if " " not in x and _stem(x) and _stem(x) in stems(b)}
    return any(y in T["ant"].get(x, ()) for x in ka for y in kb)


def synonyms_of(w, T=None):
    """the confirmed synonyms of a word (forms as in the table)"""
    T = T or load()
    c = cls_of(w, T)
    return sorted(T["members"].get(c, set()) - {w.lower()}) if c is not None else []


def _units(line, T):
    low = (line or "").lower()
    out = []
    for p in T["phrases"]:
        if p in low:
            out.append(p)
            low = low.replace(p, " ")
    return out + WORD.findall(low)


def compare(draft, ref, T=None):
    """word-level difference of a draft and a reference line, with confirmed synonyms taken out.
    Returns {'only_draft': [...], 'only_ref': [...], 'synonym': [(d, r)], 'antonym': [(d, r)]}"""
    T = T or load()
    d, r = _units(draft, T), _units(ref, T)
    od = [w for w in d if w not in r]
    orf = [w for w in r if w not in d]
    syn, ant = [], []
    for w in list(od):
        for x in list(orf):
            if same(w, x, T):
                syn.append((w, x))
                od.remove(w)
                orf.remove(x)
                break
            if antonyms(w, x, T):
                ant.append((w, x))
                break
    return {"only_draft": od, "only_ref": orf, "synonym": syn, "antonym": ant}
