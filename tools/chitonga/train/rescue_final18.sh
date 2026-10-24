#!/bin/bash
# FINAL round RESCUE (23 Oct 2026, the Jesuit): called by launch_final18.sh before it terminates a failed or stalled
# machine (feedback-always-keep-trained-models). Every checkpoint or merged folder without UPOK
# (s18/F18sw<dose>/ckpt_<step>, s18/F18B18s<seed>/ckpt_<step>, merged/F18B18m<alpha>s<seed>) is uploaded full to its planned
# private repo, then an int8 copy is attempted. Prints RESCUED <repo> per folder and RESCUE END last.
# SMOKE=1: copies into $SMOKE_DIR/fakehub instead of HF.
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then cd "${SMOKE_DIR:?SMOKE_DIR}"; else cd /root; fi
set -a; [ -f .env ] && . ./.env; set +a
NS=PRIVATE_HF_ORG/chitonga-nllb-f18
put() {
  if [ "$SMOKE" = 1 ]; then mkdir -p "fakehub/$2" && cp -r "$1"/. "fakehub/$2"/
  else timeout 5400 python upload_model.py "$1" "$2" "final round RESCUE upload (run failed or stalled), 23 Oct 2026" facebook/nllb-200-distilled-1.3B
  fi
}
# F12 (pre-launch review): a healthy upload is never killed. If upload_model.py is still running, wait for it (it has
# its own 5,400 s timeout); only then rescue what still lacks UPOK.
w=0
while pgrep -f upload_model.py > /dev/null 2>&1 && [ $w -lt 90 ]; do
  echo "RESCUE: an upload is still running, waiting ($w min)"; sleep 60; w=$((w + 1))
done
for d in s18/F18sw*/ckpt_[0-9]* s18/F18B18s*/ckpt_[0-9]* merged/F18B18m*; do
  case "$d" in *_ct2r|*_ct2r/) continue ;; esac     # an int8 output directory is never a checkpoint (stage 15b lesson)
  [ -f "$d/config.json" ] || continue
  [ -f "$d/UPOK" ] && continue
  python -c "
import json, os, sys
d = sys.argv[1]; i = os.path.join(d, 'model.safetensors.index.json')
ok = os.path.exists(os.path.join(d, 'model.safetensors')) or (os.path.exists(i) and all(
    os.path.exists(os.path.join(d, f)) for f in set(json.load(open(i))['weight_map'].values())))
sys.exit(0 if ok else 1)" "$d" || { echo "RESCUE SKIPPED $d (weights incomplete)"; continue; }
  case "$d" in
    merged/*) run=${d##*/F18B18m}; seed=${run##*s}; alpha=${run%s*}; repo=$NS-B18m$alpha-s$seed ;;   # F18B18m07s1 -> -B18m07-s1
    s18/F18sw*) st=${d##*ckpt_}; run=$(basename "$(dirname "$d")")
                repo=$NS-sweep${run#F18sw}-step$st ;;                      # F18swS18b -> -sweepS18b-step<N>
    *) st=${d##*ckpt_}; run=$(basename "$(dirname "$d")"); run=${run#F18}; arm=${run%s*}; seed=${run##*s}
       repo=$NS-$arm-s$seed-step$st ;;
  esac
  rm -f "$d/SAVED"
  c=${d%/}_ct2r      # F12 (pre-launch review): the int8 goes FIRST (1.3 GB, and it is what the kill test downloads)
  if [ ! -d "$c" ]; then
    ct2-transformers-converter --model "$d" --output_dir "$c" --quantization int8 --force > /dev/null 2>&1 || true
  fi
  [ -d "$c" ] && put "$c" "$repo-ct2-int8" && echo "RESCUED $repo-ct2-int8"
  if put "$d" "$repo"; then
    echo "RESCUED $repo"
  else
    echo "RESCUE FAILED $repo (folder left on the machine)"
  fi
done
echo "RESCUE END"
