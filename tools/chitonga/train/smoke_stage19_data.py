"""A SMALL copy of the stage 19 data for the CPU smoke of run_stage19.sh (the Jesuit, 24 Oct 2026). Why: the Beelink
stood at 92.5% RAM beside the stage 18 kill test; the full files (train 44 MB, tasks 46 MB) cost ~1.5 GB of Python
objects per trainer process, while the smoke tests the CODE PATH, not the data (the data is checked by 'dry' and on the
droplet by its sha). Every group keeps rows (all of a small group, up to 300 of a big one), every task pool up to 200
rows, mono 500; the same mixes are re-checked by mix_groups.check_build on the copy, so the preflight the trainer
demands is real.
    python -X utf8 tools/chitonga/train/smoke_stage19_data.py <run dir> <smoke s19 dir>
"""
import collections, json, os, random, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix_groups as MG
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
rng = random.Random(7)
def sub(name, key, per):
    by = collections.defaultdict(list)
    for l in open(os.path.join(src, name), encoding="utf-8"):
        if l.strip():
            x = json.loads(l)
            by[key(x)].append(l)
    with open(os.path.join(dst, name), "w", encoding="utf-8", newline="\n") as f:
        for g, ls in by.items():
            for l in (ls if len(ls) <= per else rng.sample(ls, per)):
                f.write(l if l.endswith("\n") else l + "\n")
    return {g: min(len(v), per) for g, v in by.items()}
g1 = sub("train.jsonl", lambda x: MG.group(x["origin"]), 300)
g2 = sub("tasks.jsonl", lambda x: x["pool"], 200)
sub("mono.jsonl", lambda x: "copy", 500)
for f in ("manifest.json", "teacher_dev_stage19.jsonl", "synthetic_s19_heldout.jsonl", "confusable_probes_final18.jsonl"):
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
# the evaluation files are cut short: a random tiny model decodes to the length limit, so the full test decodes at the
# end of every trainer run cost minutes on the CPU and test nothing the first lines do not (smoke only)
for f, n in (("dev.jsonl", 60), ("sel.jsonl", 40), ("test.en.txt", 8), ("test_hopgood.en.txt", 8)):
    with open(os.path.join(src, f), encoding="utf-8") as a, open(os.path.join(dst, f), "w", encoding="utf-8", newline=chr(10)) as b:
        if f == "sel.jsonl":
            ls = [l for l in a if l.strip()]
            for d in ("en>toi", "toi>en"):
                b.writelines([l for l in ls if json.loads(l)["dir"] == d][:n // 2])
        else:
            for i, l in enumerate(a):
                if i >= n:
                    break
                b.write(l)
man = json.load(open(os.path.join(dst, "manifest.json"), encoding="utf-8"))
for m in ("S19a", "S19b", "S19c", "P33"):
    MG.check_build(dst, man["mixes"][m])
print("SMOKE DATA OK", sum(g1.values()), "train rows,", sum(g2.values()), "task rows; 4 mixes checked on the copy")
