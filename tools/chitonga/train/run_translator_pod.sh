#!/bin/bash
# ON THE POD (23 Oct 2026, the Jesuit; Chris: "If the beelink is the bottleneck, we need to get cheap pods to open up the
# bottleneck"). Serves the installed translator (translator_service + chitonga-nllb-wise-a050) over HTTP for the six
# streams. Started by launch_translator_pod.sh with nohup; log /root/run.log; markers: SETUP DONE, SELFKILL yes|no,
# SERVER READY, TPOD_FAILED <why>.
# Setup copied from run_wmap_pod.sh (22 Oct, same pins, same models, same sha checks): a050 FULL from its verified HF
# repo (both shards' sha256 asserted = the Beelink's), int8 DOWNLOADED (the Beelink's model.bin, sha asserted; plan gate 4), R (s13 soup) int8
# + the two s13 seeds for the readers / checker. Then translator_pod_server.py on port 8000 (RunPod HTTPS proxy).
# Pod-side guard, independent of the Beelink: every minute, if no request for IDLE_MIN or the pod is older than CAP_MIN,
# the pod removes itself with the pod's own runpodctl (if the image gives it a key; SELFKILL says which). The launcher
# on the Beelink applies the same rules through the API and is the one that confirms GONE.
set -u
cd /root
fail() { echo "TPOD_FAILED $1"; exit 1; }
set -a; . /root/.env; set +a
[ -n "${HF_TOKEN:-}" ] || fail "no HF_TOKEN"
[ -n "${TRANSLATOR_TOKEN:-}" ] || fail "no TRANSLATOR_TOKEN"
IDLE_MIN=${IDLE_MIN:-60}; CAP_MIN=${CAP_MIN:-640}; T0=$(date +%s)
mkdir -p /root/state; date +%s > /root/state/last_request
# pod-side guard first, so a hang in the setup is covered too (the setup counts as idle time)
if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then echo "SELFKILL yes"; SK=1; else echo "SELFKILL no"; SK=0; fi
(
  while sleep 60; do
    now=$(date +%s); last=$(cat /root/state/last_request 2>/dev/null || echo $now)
    if [ $(( (now - last) / 60 )) -ge $IDLE_MIN ] || [ $(( (now - T0) / 60 )) -ge $CAP_MIN ]; then
      echo "GUARD: idle $(( (now - last) / 60 )) min, age $(( (now - T0) / 60 )) min: removing this pod $(date +%H:%M:%S)"
      [ $SK = 1 ] && runpodctl remove pod "$RUNPOD_POD_ID"
      sleep 300
    fi
  done
) >> /root/guard.log 2>&1 &
# 24 Oct: fastembed / onnxruntime / sentencepiece / regex pinned to the Beelink's versions (bge-small feeds the T>E
# tm-near cosine; an unpinned install could serve another embedding build than the Beelink's)
python -m pip install -q "transformers==4.46.3" "sentencepiece==0.2.2" huggingface_hub "ctranslate2==4.5.0" \
  "fastembed==0.9.0" "onnxruntime==1.31.0" psutil "regex==2026.9.29" > pip.log 2>&1 || fail "pip: $(tail -2 pip.log)"
python -c "import importlib.metadata as m; print('versions', {p: m.version(p) for p in ('ctranslate2', 'transformers', 'sentencepiece', 'fastembed', 'onnxruntime', 'regex', 'numpy')})"
echo "cpu $(nproc) vcpu;$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2); flags: $(grep -m1 flags /proc/cpuinfo | grep -oE 'avx512f|avx512_vnni|avx512_bf16|avx2' | sort -u | tr '\n' ' ')"
python -c "import ctranslate2, transformers; print('ct2', ctranslate2.__version__, 'cuda devices', ctranslate2.get_cuda_device_count(), 'transformers', transformers.__version__)" || fail "imports"
mkdir -p /root/w && tar -xzf /root/w.tgz -C /root/w || fail "untar"
W=/root/w/zamai-pro
M=/root/models; mkdir -p $M /root/out
python - <<'EOF' || fail "model download"
import os, shutil
from huggingface_hub import snapshot_download
M = "/root/models"; U = "PRIVATE_HF_ORG/"
TOK = ["*.json", "*.model"]
def get(repo, local, pats=None):
    snapshot_download(U + repo, local_dir=os.path.join(M, local), allow_patterns=pats, max_workers=8)
    print("got", repo, sorted(os.listdir(os.path.join(M, local))), flush=True)
get("chitonga-nllb-wise-a050", "chitonga-nllb-wise-a050")
get("chitonga-nllb-s13-soup", "chitonga-nllb-s13-soup", TOK)
get("chitonga-nllb-s13-soup-ct2-int8", "chitonga-nllb-s13-soup-ct2-int8")
for s in ("S13_bi_s1", "S13_bi_s2"):
    get(f"chitonga-nllb-s13-{s}-ct2-int8", f"chitonga-nllb-s13-{s}-ct2-int8")
    d = os.path.join(M, f"chitonga-nllb-s13-{s}")
    try:
        get(f"chitonga-nllb-s13-{s}", f"chitonga-nllb-s13-{s}", TOK)
    except Exception as e:  # the reader needs only the tokenizer of the full folder: the NLLB tokenizer, the same files
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(os.path.join(M, "chitonga-nllb-s13-soup")):
            if f.endswith((".json", ".model")) and not f.startswith("model"):
                shutil.copy(os.path.join(M, "chitonga-nllb-s13-soup", f), d)
        print("tokenizer of", s, "copied from s13-soup:", repr(e)[:80], flush=True)
EOF
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  want=$(grep "wise-a050/$f" /root/local_sha.txt | cut -c1-64)
  got=$(sha256sum $M/chitonga-nllb-wise-a050/$f | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "A050 SHA MISMATCH $f (local ${want:-none}, pod $got)"
done
echo "A050 SHA OK (both shards byte-identical to the Beelink's)"
# 23 Oct, first serve: the pod's own int8 conversion was NOT the Beelink's (sha c923... vs de2b...; first line dropped
# 'manza'). TRANSLATOR_POD_PLAN.md gate 4: the pod never converts; it downloads the Beelink's exact model.bin from the HF
# int8 repo and the server does not start unless its sha256 is the Beelink's.
python - <<'EOF' || fail "int8 download"
from huggingface_hub import snapshot_download
snapshot_download("PRIVATE_HF_ORG/chitonga-nllb-wise-a050-ct2-int8", local_dir="/root/models/chitonga-nllb-wise-a050-ct2-int8",
                  max_workers=8)
EOF
want=$(grep "wise-a050-ct2-int8/model.bin" /root/local_sha.txt | cut -c1-64)
got=$(sha256sum $M/chitonga-nllb-wise-a050-ct2-int8/model.bin | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || fail "INT8 SHA MISMATCH a050 (local ${want:-none}, pod $got)"
echo "INT8 SHA OK a050 (the Beelink's model.bin, downloaded, not converted)"
echo "SETUP DONE $(date +%H:%M:%S)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee /root/out/gpu.txt
export CHITONGA_MODELS=$M CT2_DEVICE=cuda CT2_COMPUTE=${CT2_COMPUTE:-int8} HARNESS_THREADS=2 PYTHONUNBUFFERED=1 \
  STATE_DIR=/root/state PORT=8000 WORKERS=${WORKERS:-3}
cd $W
python -X utf8 -c "import sys; sys.path.insert(0, 'tools/collector'); from embed import model; m = model(); print('bge ready', len(list(m.embed(['a']))[0]))" || fail "bge-small"
python -X utf8 tools/chitonga/llm/translator_pod_server.py || fail "server exited"
