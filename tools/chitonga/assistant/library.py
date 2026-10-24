"""Step 3a, the English content library and its retrieval.

TWO KINDS OF CARD, one internal shape {'id', 'kind', 'topic', 'parts': [(part, [sentences])], 'keep_english',
'status', 'sources', 'file'}:

  curriculum  the five-part content cards (FLYWHEEL_DESIGN §0b, Chris 22 Oct: "Don't just answer it, we explain"):
              JSON Lines as the Codex library prompt fixes them (council/codex/2026-10-22_1111_curriculum_library_prompt):
              {"topic","grade","subject","answer","idea","local_example","how_to_teach","check_questions":[{"q","a"}],
               "keep_english":[],"sources":[]}  (+ optional "status": approved | reviewed | draft; default draft)
              Read from a .jsonl file or from the JSON lines inside a Codex .md answer (prompt files are skipped).
              keep_english items may be "term" or {"term", "explain"}.
  advice      the 10 advice cards (internal/chitonga_train/advice/cards/*.md, draft v1, 20 Oct), keyed by the
              classroom situation (gate.advice_card). Served short: the first 3 "Try this first" points (their bold
              lead), one "Why it works" sentence, the first 2 "Avoid" points.

RETRIEVAL is lexical and deterministic (no model, no vectors needed at run time): content words of the request,
lightly stemmed, weighted by inverse document frequency over the cards; a card qualifies when it covers at least
CFG library_min_cover of the request's weight AND shares a content word with its topic line. Optional `embed` (bge-small
through h3_select.embed) can be passed to rank; it never admits a card the lexical rule refused (vectors find
neighbours, not answers: 'kilometre' and 'kilogram' sit close).

Status is carried to the answer: a card that is not 'approved' is served as "library card, not yet approved" in the
record; the teacher-facing marker is the same as for a model draft (pipeline.DRAFT_MARK).
"""
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm"))

STOP = set("""a an the of to in on at for from by with and or but if then so as is are was were be been being am do does
did doing have has had having it its this that these those there here what whats what's why how which where when who
whom whose give list name explain describe define state tell me us you your we our they their i my he she his her them
can could would should will shall may might must please some any many much more most very about into out up down over
under than too also just only not no yes make makes made reasons reason way ways one two three four five""".split())
PARTS_CURRICULUM = ("answer", "idea", "local_example", "how_to_teach", "check_questions")


def stem(w):
    w = w.lower().replace("’", "'")
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    return w.replace("metre", "meter").replace("kilometre", "kilometer")


def content_words(text):
    return [stem(w) for w in re.findall(r"[A-Za-z']+", text or "") if w.lower() not in STOP and len(w) > 1]


UNIT_END = re.compile(r"(\d\s?(?:km|cm|mm|kg|ml|m|g|l)\.)\s+(?=[A-Z])")


def split_sentences(text):
    """translator_service.sentences (keeps 'e.g.', 'Dr.' intact), then a cut after a unit that ends a sentence: the
    service reads '100 m. Ask ...' as one sentence (single letter = initial; checked 22 Oct), and NLLB drops second
    sentences (pivot test 18 Oct). Pure fallback when the service is not importable."""
    try:
        import translator_service as TS  # noqa: PLC0415
        parts = [s.strip() for s in TS.sentences(text or "") if s.strip()]
    except Exception:  # noqa: BLE001  (the pure fallback: split after . ! ? followed by a capital)
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", (text or "").strip()) if s.strip()]
    out = []
    for p in parts:
        bits = UNIT_END.split(p)          # [text, unit-with-stop, text, unit-with-stop, ..., rest]
        for i in range(0, len(bits) - 1, 2):
            out.append((bits[i] + bits[i + 1]).strip())
        if bits[-1].strip():
            out.append(bits[-1].strip())
    return out


# ------------------------------------------------------------------------------------------------ loading
def _keep(items):
    out = []
    for k in items or []:
        if isinstance(k, str) and k.strip():
            out.append({"term": k.strip(), "explain": ""})
        elif isinstance(k, dict) and (k.get("term") or "").strip():
            out.append({"term": k["term"].strip(), "explain": (k.get("explain") or k.get("explanation") or "").strip()})
    return out


def validate(rec):
    """the problems with one curriculum record ([] = usable)"""
    bad = [k for k in ("topic", "answer", "idea", "local_example", "how_to_teach") if not (rec.get(k) or "").strip()]
    cq = rec.get("check_questions")
    if not isinstance(cq, list) or not cq or not all(isinstance(x, dict) and x.get("q") and x.get("a") for x in cq):
        bad.append("check_questions")
    return bad


def curriculum_card(rec, file, i):
    parts = [(p, split_sentences(rec[p])) for p in PARTS_CURRICULUM[:4]]
    parts.append(("check_questions", [f"{x['q'].strip()} -> {x['a'].strip()}" for x in rec["check_questions"]]))
    return {"id": f"cur:{os.path.basename(file)}:{i}", "kind": "curriculum", "topic": rec["topic"].strip(),
            "grade": rec.get("grade"), "subject": rec.get("subject"), "parts": parts,
            "keep_english": _keep(rec.get("keep_english")), "status": rec.get("status") or "draft",
            "sources": rec.get("sources") or [], "file": file,
            "text": " ".join([rec["topic"], rec["answer"], rec["idea"]] + [x["q"] for x in rec["check_questions"]])}


def _json_lines(path):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip().strip(",")
            if ln.startswith("{") and ln.endswith("}"):
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    continue


def load_curriculum(paths):
    """(cards, problems). paths: files or directories (a directory is searched for *curriculum_library*.jsonl|md,
    prompt files skipped)"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*curriculum_library*")) if
                            f.endswith((".jsonl", ".md")) and "_prompt" not in f)
        elif os.path.isfile(p):
            files.append(p)
    cards, problems = [], []
    for f in files:
        for i, rec in enumerate(_json_lines(f)):
            bad = validate(rec)
            if bad:
                problems.append({"file": f, "line": i, "topic": rec.get("topic"), "missing": bad})
                continue
            cards.append(curriculum_card(rec, f, i))
    return cards, problems


BOLD = re.compile(r"\*\*(.+?)\*\*")


def _section(md, title):
    m = re.search(r"(?ms)^##\s+" + re.escape(title) + r"\s*$(.*?)(?=^##\s|\Z)", md)
    return m.group(1).strip() if m else ""


def _bullets(sec):
    out = []
    for ln in sec.splitlines():
        if ln.lstrip().startswith(("- ", "* ")):
            b = ln.lstrip()[2:].strip()
            m = BOLD.match(b)
            out.append((m.group(1) if m else b).strip().rstrip(":"))
    return out


def load_advice(folder):
    """{card number: card} from advice/cards/NN_name.md"""
    out = {}
    for f in sorted(glob.glob(os.path.join(folder or "", "[0-9][0-9]_*.md"))):
        with open(f, encoding="utf-8") as fh:
            md = fh.read()
        num = os.path.basename(f)[:2]
        title = (md.splitlines() or [""])[0].lstrip("# ").split("·")[-1].strip()
        why = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\([^)]*\d{4}[^)]*\)", "", _section(md, "Why it works")))
        why = split_sentences(why)[:1]
        parts = [("try", [b if b.endswith((".", "!", "?")) else b + "." for b in _bullets(_section(md, "Try this first"))]),
                 ("why", why),
                 ("avoid", [b if b.endswith((".", "!", "?")) else b + "." for b in _bullets(_section(md, "Avoid"))])]
        out[num] = {"id": f"adv:{num}", "kind": "advice", "topic": title, "parts": parts, "keep_english": [],
                    "status": "draft v1 (teacher review pending)", "sources": [os.path.relpath(f)], "file": f}
    return out


# ------------------------------------------------------------------------------------------------ retrieval
class Library:
    def __init__(self, cards, advice=None, problems=None):
        self.cards = cards
        self.advice = advice or {}
        self.problems = problems or []
        self.df = {}
        for c in cards:
            for w in set(content_words(c["text"])):
                self.df[w] = self.df.get(w, 0) + 1

    @classmethod
    def load(cls, cfg):
        cards, problems = load_curriculum(cfg["library_paths"])
        return cls(cards, load_advice(cfg.get("advice_dir")), problems)

    def idf(self, w):
        return math.log((1 + len(self.cards)) / (1 + self.df.get(w, 0))) + 1.0

    def search(self, query, cfg, k=3, embed=None):
        """(best card or None, ranked [(score, cover, topic_hits, card id)] for the log)"""
        q = list(dict.fromkeys(content_words(query)))
        if not q or not self.cards:
            return None, []
        total = sum(self.idf(w) for w in q)
        ranked = []
        for c in self.cards:
            cw = set(content_words(c["text"]))
            tw = set(content_words(c["topic"]))
            cover = sum(self.idf(w) for w in q if w in cw) / total
            th = sum(1 for w in q if w in tw)
            ranked.append([cover + 0.25 * th, round(cover, 3), th, c])
        if embed is not None:
            vs = embed([query] + [c["text"] for c in self.cards])
            for r, v in zip(ranked, vs[1:]):
                r[0] += 0.5 * float((vs[0] * v).sum())
        ranked.sort(key=lambda r: -r[0])
        log = [(round(r[0], 3), r[1], r[2], r[3]["id"]) for r in ranked[:k]]
        for r in ranked:
            if r[1] >= cfg["library_min_cover"] and r[2] >= cfg["library_min_topic"]:
                return r[3], log
        return None, log
