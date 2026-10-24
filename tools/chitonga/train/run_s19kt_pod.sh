#!/bin/bash
# ON THE POD (24 Oct 2026, the Jesuit; Chris: "check the models on a runpod"). The stage 19 kill test's DECODES: a050 and
# the five stage 19 candidates, raw and served, every set of killtest/METHOD_NOTES.md section 4. A batch job, not a
# server. Started by launch_s19kt_pod.sh with nohup; log /root/run.log; markers SETUP DONE, S19KT_DONE, S19KT_FAILED <why>.
# Setup copied from run_translator_pod.sh (same pins: transformers 4.46.3, ctranslate2 4.5.0 on CUDA, proven 22-23 Oct).
# The pod never converts: every int8 model.bin is downloaded from its HF repo and its sha256 asserted against pins.json.
# Results: /root/w/zamai-pro/internal/chitonga_train/runs/2026-10-24_stage19_data/killtest/dec -> /root/dec.tgz.
set -u
cd /root
fail() { echo "S19KT_FAILED $1"; exit 1; }
set -a; . /root/.env; set +a
[ -n "${HF_TOKEN:-}" ] || fail "no HF_TOKEN"
CAP_MIN=${CAP_MIN:-240}; T0=$(date +%s)
# key-free backstop: the pod ends its own work at the cap even if the launcher is gone (the launcher and the overwatch
# are the ones that DELETE the pod; this only stops the spend of work and leaves the results on disk)
( sleep $(( CAP_MIN * 60 )); echo "BACKSTOP $(date +%H:%M:%S): cap $CAP_MIN min, killing the job"; pkill -f final19_run.py ) >> /root/guard.log 2>&1 &
python -m pip install -q "transformers==4.46.3" sentencepiece huggingface_hub "ctranslate2==4.5.0" fastembed onnxruntime \
  psutil regex > pip.log 2>&1 || fail "pip: $(tail -2 pip.log)"
python -c "import ctranslate2, transformers; print('ct2', ctranslate2.__version__, 'cuda devices', ctranslate2.get_cuda_device_count(), 'transformers', transformers.__version__)" || fail "imports"
mkdir -p /root/w && tar -xzf /root/w.tgz -C /root/w || fail "untar"
W=/root/w/zamai-pro
KT=$W/internal/chitonga_train/runs/2026-10-24_stage19_data/killtest
M=/root/models; mkdir -p $M
HF_HUB_DISABLE_PROGRESS_BARS=1 python - <<'EOF' || fail "model download / sha"
import hashlib, json, os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
M = "/root/models"; U = "PRIVATE_HF_ORG/"
pins = json.load(open("/root/w/zamai-pro/internal/chitonga_train/runs/2026-10-24_stage19_data/killtest/pins.json"))["int8"]
TOK = ["*.json", "*.model"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
def get(int8):
    full = int8[:-len("-ct2-int8")]
    snapshot_download(U + int8, local_dir=os.path.join(M, int8), max_workers=8)
    snapshot_download(U + full, local_dir=os.path.join(M, full), allow_patterns=TOK, max_workers=4)
    got = sha(os.path.join(M, int8, "model.bin"))
    ok = got == pins[int8]
    print(("INT8 SHA OK " if ok else "INT8 SHA MISMATCH ") + int8 + " " + got[:16], flush=True)
    return ok
# 24 Oct, first try: three snapshot_downloads in parallel threads crashed in tqdm (AttributeError: type object 'tqdm'
# has no attribute '_lock'); one repo at a time (8 workers each), progress bars off
res = [get(r) for r in pins]
assert all(res), "sha mismatch"
EOF
echo "ALL INT8 SHA OK (downloaded, not converted)"
# the stage 19 tokenizer.json differs in bytes from a050's (re-saved): assert the pieces are the same for tagged input
python - <<'EOF' || fail "tokenizer check"
from transformers import AutoTokenizer
M = "/root/models/"
names = ["chitonga-nllb-wise-a050", "chitonga-nllb-s19-B13-s1", "chitonga-nllb-s19-B13-s2", "chitonga-nllb33-s19-P1",
         "chitonga-nllb33-s19-B33-s1", "chitonga-nllb33-s19-B33-s2", "chitonga-nllb-s13-soup"]
probe = ["<cls> Count the beans in groups of ten.", "<gen> Kamusamba ansipa kamutana kulya.", "Mwana ulacisa goyooyo."]
ref = None
for n in names:
    t = AutoTokenizer.from_pretrained(M + n)
    out = []
    for sl, s in (("eng_Latn", probe[0]), ("toi_Latn", probe[1]), ("toi_Latn", probe[2])):
        t.src_lang = sl
        out.append(t.convert_ids_to_tokens(t.encode(s)))
    ref = ref or out
    print("TOKENIZER", n, "same as a050" if out == ref else "DIFFERS", out[0][:7], flush=True)
    assert out == ref, n
EOF
echo "SETUP DONE $(date +%H:%M:%S)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
NP=$(nproc); PAR=${PAR:-3}
echo "nproc $NP par $PAR"
export CHITONGA_MODELS=$M CT2_DEVICE=cuda CT2_COMPUTE=int8 HARNESS_THREADS=2 PYTHONUNBUFFERED=1 \
  HARNESS_READER=$M/chitonga-nllb-s13-soup OMP_NUM_THREADS=2
cd $W
# 24 Oct 16:30: several pods share the sets; JOBS (from .env) = "SYSTEM@setspec;..." for this pod (final19_run.parse_spec)
if [ -n "${JOBS:-}" ]; then
  python -X utf8 tools/chitonga/train/final19_run.py podall --par $PAR --jobs "$JOBS" 2>&1 | tee /root/podall.log
else
  python -X utf8 tools/chitonga/train/final19_run.py podall --par $PAR 2>&1 | tee /root/podall.log
fi
tar -czf /root/dec.tgz -C $KT dec || fail "tar dec"
grep -q "^PODALL DONE" /root/podall.log || { echo "S19KT_FAILED podall: $(tail -1 /root/podall.log)"; exit 1; }
echo "S19KT_DONE $(date +%H:%M:%S) wall $(( ($(date +%s) - T0) / 60 )) min"
