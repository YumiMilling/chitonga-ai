"""Stage 8 data for the NLLB translator (14 Oct 2026; Chris: "let's train the translation model on everything that
wasn't yet included"). The stage 7 run folder plus everything the translator has not seen since 12 Oct:

  own        the translators' lines from a FRESH export (export_pairs.py, leak filters applied): 1,275 vs 1,046 in s7
  teacher    Fr Musunga's corrections of model sentences, v0.55 and v0.6 rounds (desk/teacher_corrections_*.jsonl)
  glossary   the Social Studies picture glossary (Tonga headings, English from the pictures): own group
  grammar    grammar_v065.jsonl: the engine sets rebuilt with his 14 Oct negative future (ta-SC-ka-ROOT-i) and every
             earlier teacher rule (no invented words, Bemba/Nyanja verbs out); its own group so an arm can leave it out
  copy       stage 7's monolingual Tonga + today's typed library books and the Let's Read readers (Tonga lines only)
The stage 7 story, bible and hopgood groups are kept as they were; its syn_* groups stay in the file but no arm uses them.

Held out from everything (asserted): the frozen translators' test, Hopgood's held-out test, the stage 7 dev set, and
the new children's-book + radio test (llm/eval_children/items.tsv): any training or copy line sharing a 6-word sequence
with one of its Tonga lines, or equal to one of its English lines, is dropped.

    python -X utf8 tools/chitonga/train/build_stage8b.py   -> internal/chitonga_train/runs/2026-10-14_stage8/
"""
import collections
import csv
import glob
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CT = os.path.join(REPO, "internal", "chitonga_train")
S7 = os.path.join(CT, "runs", "2026-10-12_stage7")
EXP = os.path.join(CT, "runs", "2026-10-14_export")
S8 = os.path.join(CT, "runs", "2026-10-14_stage8")
RAW = os.path.normpath(os.path.join(REPO, "..", "zamai-corpus-raw", "chitonga_mono"))
WORD = re.compile(r"[a-zŋ']+")
ENG = set("the and of to is in that for with are was this be on by as it at from or have an they which you were their has been "
          "not but all can will usaid ministry education copyright cambridge press university republic zambia sale published "
          "printed isbn rights reserved support people american made possible contents responsibility edc lusaka grade".split())


def g6(t):
    w = WORD.findall(t.lower())
    return {tuple(w[i:i + 6]) for i in range(len(w) - 5)} or ({tuple(w)} if w else set())


def norm(s):
    return " ".join(s.lower().split())


def main():
    os.makedirs(S8, exist_ok=True)
    for f in ("dev.jsonl", "test.en.txt", "test_hopgood.en.txt"):
        shutil.copy(os.path.join(S7, f), S8)
    read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    dev = read(os.path.join(S7, "dev.jsonl"))
    held_toi, held_en = set(), set()
    for r in csv.DictReader(open(os.path.join(CT, "llm", "eval_children", "items.tsv"), encoding="utf-8"), delimiter="\t"):
        held_toi |= g6(r["toi"]); held_en.add(norm(r["en"]))
    for r in dev:
        held_en.add(norm(r["en"]))
    test_en = {norm(l) for f in ("test.en.txt", "test_hopgood.en.txt") for l in open(os.path.join(S7, f), encoding="utf-8")}
    held_en |= test_en
    dev_ids = {r["id"] for r in dev}

    def clean(x):
        return not (g6(x["toi"]) & held_toi) and norm(x.get("en", "")) not in held_en and x.get("id") not in dev_ids

    st, train = collections.Counter(), []
    s7 = read(os.path.join(S7, "train.jsonl"))
    for x in s7:                                   # stage 7 minus its old own lines (replaced by the fresh export)
        if x["origin"].startswith(("smol", "domain")):
            continue
        if clean(x):
            train.append(x); st["s7:" + x["origin"].split(":")[0]] += 1
        else:
            st["dropped (held out): s7"] += 1
    for x in read(os.path.join(EXP, "train.jsonl")):
        if clean(x):
            train.append(x); st["own (fresh export)"] += 1
        else:
            st["dropped (held out): own"] += 1
    for f in sorted(glob.glob(os.path.join(CT, "desk", "teacher_corrections_v0*.jsonl"))):
        for i, x in enumerate(read(f)):
            y = {"id": f"teacher:{os.path.basename(f)}:{i}", "en": x["en"], "toi": x["toi"], "origin": "teacher:T01"}
            if clean(y):
                train.append(y); st["teacher"] += 1
            else:
                st["dropped (held out): teacher"] += 1
    gl = os.path.join(RAW, "library_choma", "social_studies_ece_3_4", "social_studies_mu_chitonga_myaka_3-4.toi.tsv")
    for i, r in enumerate(csv.DictReader(open(gl, encoding="utf-8"), delimiter="\t")):
        if r["page"] in ("imprint", "credits", "cover", "title", "all"):
            continue
        y = {"id": f"glossary:ss34:{i}", "en": r["en_from_picture"], "toi": r["toi"], "origin": "library:glossary"}
        if clean(y):
            train.append(y); st["glossary"] += 1
    for i, x in enumerate(read(os.path.join(CT, "synthetic", "grammar_v065.jsonl"))):
        y = {"id": f"grammar:{i}", "en": x["en"], "toi": x["toi"], "origin": "synthetic:grammar"}
        if clean(y):
            train.append(y); st["grammar"] += 1
        else:
            st["dropped (held out): grammar"] += 1

    mono = [x for x in read(os.path.join(S7, "mono.jsonl")) if not (g6(x["toi"]) & held_toi)]
    st["copy: stage 7"] = len(mono)
    seen = {norm(x["toi"]) for x in mono}
    files = glob.glob(os.path.join(RAW, "library_choma", "*", "*.toi.txt")) + glob.glob(os.path.join(RAW, "edc_letsread", "*.txt"))
    for f in sorted(files):
        for k, line in enumerate(open(f, encoding="utf-8").read().splitlines()):
            ws = WORD.findall(line.lower())
            if len(ws) < 3 or sum(w in ENG for w in ws) / len(ws) >= 0.3 or re.search(r"\d|www|@|copyright", line.lower()):
                continue
            if g6(line) & held_toi or norm(line) in seen:
                st["copy: dropped (held out or duplicate)"] += 1
                continue
            seen.add(norm(line))
            mono.append({"id": f"mono:{os.path.basename(f)}:{k}", "toi": line.strip(), "origin": "library" if "library_choma" in f else "letsread"})
            st["copy: new " + ("library" if "library_choma" in f else "letsread")] += 1

    assert not any(x.get("id") in dev_ids for x in train)
    assert not any(norm(x["en"]) in test_en for x in train)
    assert not any(g6(x["toi"]) & held_toi for x in train + mono)
    with open(os.path.join(S8, "train.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for x in train:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(S8, "mono.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for x in mono:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump({"made": "2026-10-14", "counts": dict(st), "train": len(train), "mono": len(mono), "dev": len(dev),
               "held_out": ["frozen translators' test", "Hopgood test", "stage 7 dev", "eval_children (47)"]},
              open(os.path.join(S8, "manifest.json"), "w"), indent=1)
    for k, v in sorted(st.items()):
        print(f"  {k:40} {v}")
    print("train", len(train), "mono", len(mono), "->", S8)


if __name__ == "__main__":
    main()
