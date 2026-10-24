"""One JSON line per turn, for the desk, the tests and the next training material (23 Oct 2026, the Jesuit).

Withheld, never written: the text of a safeguarding message and of a message refused as pupil-level data (the record
keeps only that it happened). Default path internal/chitonga_train/harness/turns.jsonl (internal, never site/).
"""
import json
import os

DEFAULT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                                        "internal", "chitonga_train", "harness", "turns.jsonl"))


class TurnLog:
    def __init__(self, path=DEFAULT):
        self.path = path

    def write(self, rec):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
