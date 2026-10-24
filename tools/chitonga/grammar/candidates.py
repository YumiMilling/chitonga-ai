"""Base words for derivation (Chris, 11 Oct 2026): the verbs and adjectives of Hopgood's index (Appendix A, 2nd ed.,
ZEPH 1992), ranked by how often they occur in all the Chitonga text we hold.

    python -X utf8 tools/chitonga/grammar/candidates.py [--top 200]

Counting is exact: the grammar engine generates every form of a verb (infinitive ku-, 14 tenses x all persons and
classes) and of an adjective (all 18 classes), and only whole words equal to one of those forms are counted. Text:
the latest run folder (our translators' pairs, Hopgood, storybooks, JCTR, the Bible, the monolingual pool).
Index entries that are themselves derived forms ("(app. of amba)", "(perf. of jata)", "caus.", "pass.", "recip.")
are not base words; they are written to attested_derivations.tsv as evidence of which derivations really exist.
Writes internal/chitonga_train/derived/candidates.tsv and attested_derivations.tsv.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import concords  # noqa: E402
import tonga  # noqa: E402

REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.join(REPO, "internal", "chitonga_train", "runs", "2026-10-11_stage5")
INDEX = r"..\zamai-corpus-raw\chitonga_books\hopgood_1992\parsed\vision_index.tsv"
OUT = os.path.join(REPO, "internal", "chitonga_train", "derived")
TENSES = ["pres", "a", "remote", "fut_zoo", "fut_yoo", "neg", "neg_fut", "neg_perf", "neg_plup", "neg_futperf",
          "past_cont", "plup", "subj"]
DERIVED = re.compile(r"\((app|perf|caus|pass|recip|rev|intens|intrs|trs)\.?\s+of\s+([a-zŋ\-]+)\)", re.I)


def index_entries():
    for l in open(INDEX, encoding="utf-8"):
        e = l.rstrip("\n").split("\t", 1)[-1]
        if "[?]" in e.split(" ")[0]:   # only an unclear HEADWORD; an unclear lesson reference does not matter
            continue
        m = re.match(r"^(-?[a-zŋ']+)[,.]?\s*(?:\(([IM12])\)\s*)?(v|adj)\.\s*(.+)$", e)
        if m:
            word, mark, pos, rest = m.groups()
            meaning = re.split(r"\.\s*(?:See|cf\.|Cf\.)|\s\d+-\d+|\.\s*$", rest)[0].strip(" .;,")
            yield {"word": word.lstrip("-"), "pos": pos, "mark": mark, "meaning": meaning, "entry": e,
                   "lessons": len(re.findall(r"\b\d{1,2}-\d{1,2}\b", rest))}


def corpus_words():
    c = collections.Counter()
    for name in ("train.jsonl", "mono.jsonl"):
        for l in open(os.path.join(RUN, name), encoding="utf-8"):
            c.update(w.lower() for w in re.findall(r"[a-zA-Zŋ']+", json.loads(l).get("toi") or ""))
    return c


def verb_forms(stem):
    out = {"ku" + stem}
    for t in TENSES:
        for p in tonga.PERSONS:
            try:
                out.update(f.lower() for f in tonga.conjugate(stem, p, t))
            except Exception:
                pass
    return out


def adj_forms(stem):
    return {f.lower() for c in concords.COLS[4:] for f in [concords.adjective(c, stem)] if f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    a = ap.parse_args()
    words = corpus_words()
    rows, attested, seen = [], [], set()
    for e in index_entries():
        if e["word"] in seen:
            continue
        seen.add(e["word"])
        d = DERIVED.search(e["entry"])
        if d or re.search(r"\b(perf|app|caus|pass|recip)\.", e["meaning"]):
            attested.append((e["word"], d.group(1).lower() if d else "?", d.group(2) if d else "", e["meaning"]))
            continue
        if e["word"] in ("nsi", "kwe", "li", "ti", "ba"):  # copula, auxiliaries: not derivation bases
            continue
        forms = verb_forms(e["word"]) if e["pos"] == "v" else adj_forms(e["word"])
        e["freq"] = sum(words.get(f, 0) for f in forms)
        e["seen_forms"] = sum(1 for f in forms if words.get(f))
        rows.append(e)
    rows.sort(key=lambda r: (-r["freq"], -r["lessons"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "candidates.tsv"), "w", encoding="utf-8") as f:
        f.write("rank\tword\tpos\tmeaning\tcorpus_count\tdistinct_forms_seen\thopgood_lessons\n")
        for i, r in enumerate(rows[: a.top], 1):
            f.write(f"{i}\t{r['word']}\t{r['pos']}\t{r['meaning']}\t{r['freq']}\t{r['seen_forms']}\t{r['lessons']}\n")
    with open(os.path.join(OUT, "attested_derivations.tsv"), "w", encoding="utf-8") as f:
        f.write("derived\ttype\tbase\tmeaning\n")
        f.writelines(f"{w}\t{t}\t{b}\t{m}\n" for w, t, b, m in attested)
    print(f"{len(rows)} base verbs/adjectives; {len(attested)} index entries are derived forms (evidence)")
    for i, r in enumerate(rows[:50], 1):
        print(f"  {i:3d} {r['word']:12s} {r['pos']:4s} {r['freq']:6d} ({r['seen_forms']:2d} forms)  {r['meaning'][:48]}")


if __name__ == "__main__":
    main()
