"""The tar for the stage 19 kill-test pod (24 Oct 2026, the Jesuit). = pack_translator_pod.py's toolbox tar (everything
translator_service needs; never .env.local, never weights) plus the kill test's own files, with ONE substitution:
tools/chitonga/llm/translator_service.py is replaced by the copy with serving_tags.patch applied (RUN_PACKAGE 2 point 9).
The live Beelink file is never touched; the patch is applied in a scratch folder and only the tar carries the result.

    python -X utf8 tools/chitonga/train/pack_s19kt_pod.py <out.tgz>
"""
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.join(REPO, "internal", "chitonga_train", "runs", "2026-10-24_stage19_data")
SVC = "tools/chitonga/llm/translator_service.py"
EXTRA = ["internal/chitonga_train/runs/2026-10-24_stage19_data/killtest/items.jsonl",
         "internal/chitonga_train/runs/2026-10-24_stage19_data/killtest/pins.json",
         "internal/chitonga_train/runs/2026-10-24_stage19_data/serving_tags.patch"]


def patched_service():
    d = tempfile.mkdtemp(prefix="s19kt_patch_")
    os.makedirs(os.path.join(d, os.path.dirname(SVC)))
    shutil.copy2(os.path.join(REPO, SVC), os.path.join(d, SVC))
    r = subprocess.run(["git", "apply", os.path.join(RUN, "serving_tags.patch")], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    b = open(os.path.join(d, SVC), "rb").read()
    assert b"SERVE_TAGS" in b and b"register=None" in b
    shutil.rmtree(d, ignore_errors=True)
    return b


def main(out):
    sys.path.insert(0, HERE)
    import pack_translator_pod as P
    base = out + ".base.tgz"
    P.main(base)
    svc = patched_service()
    names = set()
    with tarfile.open(base, "r:gz") as src, tarfile.open(out, "w:gz") as dst:
        for m in src.getmembers():
            if m.name == "zamai-pro/" + SVC:
                ti = tarfile.TarInfo(m.name)
                ti.size, ti.mtime, ti.mode = len(svc), m.mtime, m.mode
                dst.addfile(ti, io.BytesIO(svc))
            else:
                dst.addfile(m, src.extractfile(m) if m.isfile() else None)
            names.add(m.name)
        assert "zamai-pro/" + SVC in names, "translator_service.py missing from the base tar"
        assert "zamai-pro/tools/chitonga/train/final19_run.py" in names
        for f in EXTRA:
            if "zamai-pro/" + f not in names:
                dst.add(os.path.join(REPO, f), arcname="zamai-pro/" + f)
    os.remove(base)
    import hashlib
    print("patched translator_service.py sha256", hashlib.sha256(svc).hexdigest())
    print("tar", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
