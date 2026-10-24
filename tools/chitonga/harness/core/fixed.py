"""The core's fixed lines in Tonga (23 Oct 2026, the Jesuit; stream F of the 11 h plan).

WHY A CORE TABLE. Greetings, the AI disclosure, "Anything else?", the honest "no material" line, the redirect, the
follow-up and the buttons are the harness's own words, the same in every plug-in. Their English lives in the plug-in's
templates.json (one place to edit the wording); their Tonga lives here, once, in
internal/chitonga_train/assistant_library/core_fixed_lines.json (built by core/build_fixed_lines.py), keyed by the same
names. Each Tonga entry stores the English it was made for: if the plug-in's English changes, the Tonga no longer
matches and is not served (English instead) until the table is rebuilt and the new line confirmed.

WHAT IS SERVED. status 'served' = written by a teacher (T01 Fr Musunga, T05, T09) or confirmed by one at the
desk (DECISIONS 20 Oct: each teacher input counts at once, confidence 1). status 'confirm' = book (B), word-list (L)
or translator (D, a050) Tonga waiting in the desk queue: shown only with drafts=True (the simulator), tagged [D] / [B].
The safeguarding referral is not here: it stays exactly as the assistant gate has it (English until a teacher writes it).

LANGUAGE. The teacher memory's preference picks the form: en = English; toi = Tonga where there is a servable line,
else English; both = Tonga, then the English in brackets. Buttons (20 characters at most): toi = Tonga; both =
'Tonga / English' when it fits, else Tonga.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "internal", "chitonga_train",
                                        "assistant_library", "core_fixed_lines.json"))
TAG = {"T01": "T", "T05": "T", "T09": "T", "B": "B", "L": "L", "D": "D"}


class FixedLines:
    def __init__(self, path=DEFAULT):
        self.path = path
        self.data = {"lines": {}, "buttons": {}}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)

    def entry(self, kind, key, en, drafts=False):
        """the Tonga entry for this key, or None: missing, made for other English, or not servable"""
        e = (self.data.get(kind) or {}).get(key)
        if not e or not e.get("toi") or (en is not None and e.get("en") != en):
            return None
        if e.get("status") == "served" or (drafts and e.get("status") == "confirm"):
            return e
        return None

    @staticmethod
    def _tag(e):
        if e.get("status") == "served":
            return ""
        return " [" + "+".join(dict.fromkeys(TAG.get(o, o) for o in e.get("origin", []))) + "]"

    def say(self, key, en, lang="en", drafts=False, **slots):
        """the line in the teacher's language; en is the plug-in's English (unformatted)"""
        def fmt(s):
            try:
                return s.format(**slots)
            except (KeyError, IndexError):
                return s
        en_f = fmt(en or "")
        if lang not in ("toi", "both"):
            return en_f
        e = self.entry("lines", key, en, drafts)
        if not e:
            return en_f
        toi = fmt(e["toi"]) + self._tag(e)
        return toi if lang == "toi" else f"{toi}\n({en_f})"

    def button(self, key, en, lang="en", drafts=False):
        if lang not in ("toi", "both"):
            return en
        e = self.entry("buttons", key, en, drafts)
        if not e:
            return en
        both = f"{e['toi']} / {en}"
        return both if lang == "both" and len(both) <= 20 else e["toi"]

    def compose(self, keys, en, lang="en", drafts=False):
        """a line built by code from several bank lines (the encouragement): Tonga only if every part has a line"""
        if lang not in ("toi", "both"):
            return en
        es = [self.entry("lines", k, None, drafts) for k in keys]
        if not all(es):
            return en
        toi = " ".join(e["toi"] for e in es) + "".join(dict.fromkeys(self._tag(e) for e in es))
        return toi if lang == "toi" else f"{toi}\n({en})"
