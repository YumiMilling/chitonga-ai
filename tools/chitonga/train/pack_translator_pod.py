"""The toolbox tar for the translator pod (23 Oct 2026, the Jesuit). What translator_service needs and nothing else: every
.py under tools/chitonga and tools/collector, plus the repo data files the service actually opened on the Beelink
(runs/2026-10-23_translator_pod/deps.txt, written by translator_pod_check.py local through an audit hook), plus every
.tsv/.json/.jsonl/.txt under tools/chitonga (5.8 MB: the engine tables a request may reach that the 20 lines did not). Never .env.local, never weights (the pod downloads
those from the verified HF repos).

    python -X utf8 tools/chitonga/train/pack_translator_pod.py <out.tgz>
"""
import os
import sys
import tarfile

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEPS = os.path.join(REPO, "internal", "chitonga_train", "runs", "2026-10-23_translator_pod", "deps.txt")


def main(out):
    files = set()
    for top in ("tools/chitonga", "tools/collector"):
        for d, dirs, fs in os.walk(os.path.join(REPO, top)):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "node_modules", ".git")]
            files.update(os.path.relpath(os.path.join(d, f), REPO).replace("\\", "/") for f in fs
                         if f.endswith(".py") or (top == "tools/chitonga" and f.endswith((".tsv", ".json", ".jsonl", ".txt"))))
    for l in open(DEPS, encoding="utf-8"):
        l = l.strip()
        if l and os.path.isfile(os.path.join(REPO, l)) and "__pycache__" not in l and not os.path.basename(l).startswith("."):
            files.add(l)
    # 23 Oct, first serve: the memory's held-out guard (heldout_guard.load on the stage 14 run) refused on the pod because
    # its frozen sets were not in the tar (the Beelink had them cached, so the audit hook never saw them opened). Ship
    # every file the guard and the memory list for themselves: frozen tests travel so they can be EXCLUDED, never trained.
    sys.path[:0] = [os.path.join(REPO, "tools", "chitonga", "llm"), os.path.join(REPO, "tools", "chitonga", "train")]
    import h2_memory as M
    import heldout_guard as HG
    for f in list(HG.dependencies(M.GUARD_RUN)) + list(M.sources()):
        if os.path.isfile(f):
            files.add(os.path.relpath(os.path.abspath(f), REPO).replace("\\", "/"))
    # 24 Oct (the reader bake-off pod): files today's service opens that the 23 Oct audit could not see: the teachers'
    # glossary (term_guard.GLOSSARY, E>T step 6c) and every anchor file + its .sha256 (the memory's forbid list).
    # Not shipped: the harness pilot logs (h2_memory.harness_files): teacher messages; checked 24 Oct 15:30, no line is
    # held_out, so the guard they feed is empty on the Beelink too.
    import term_guard as TGD
    import translator_service as TSV
    for f in [TGD.GLOSSARY] + TSV.ANCHOR_FILES + [a + ".sha256" for a in TSV.ANCHOR_FILES]:
        if os.path.isfile(f):
            files.add(os.path.relpath(os.path.abspath(f), REPO).replace("\\", "/"))
    # 24 Oct: deps.txt was written through os.path.normcase, so on Windows its paths are LOWER CASE; with the real-case
    # names from the globs above the tar held 9 case pairs (kalaula_K1 / kalaula_k1, ..._pairs.SUPERSEDED / .superseded).
    # Windows unpacks them into one file; the Linux pod into two: duplicate memory rows, and the lower-case
    # '.superseded' file passes translator_service's case-sensitive "SUPERSEDED" filter (39 rows a Beelink never
    # serves). Every path now takes its on-disk case (realpath), and one entry per case-folded name is kept.
    real = {}
    root = os.path.realpath(REPO)
    for f in files:
        r = os.path.relpath(os.path.realpath(os.path.join(REPO, f)), os.path.dirname(root)).replace("\\", "/")
        r = r.split("/", 1)[1] if r.split("/", 1)[0].lower() == os.path.basename(root).lower() else "../" + r
        real.setdefault(r.lower(), r)
    files = set(real.values())
    bad = [f for f in files if ".env" in os.path.basename(f) or f.endswith((".safetensors", ".bin", ".gguf"))]
    assert not bad, bad
    size = 0
    with tarfile.open(out, "w:gz") as t:
        for f in sorted(files):
            # files of the sibling corpus folder (../zamai-corpus-raw) keep their place next to zamai-pro
            t.add(os.path.join(REPO, f), arcname=os.path.normpath(os.path.join("zamai-pro", f)).replace("\\", "/"))
            size += os.path.getsize(os.path.join(REPO, f))
    print(f"{len(files)} files, {size / 1e6:.1f} MB raw, tar {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
