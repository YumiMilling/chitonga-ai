#!/bin/bash
# Stage 19 RESCUE (24 Oct 2026, the Jesuit): called by the launcher, the machine-side guard and the key-free backstop
# before the droplet is deleted (feedback-always-keep-trained-models). Every checkpoint or merged folder without UPOK is
# uploaded to its planned private repo, int8 FIRST (what the Beelink downloads), then the full weights. A healthy upload
# is never killed: the rescue waits for it. Prints RESCUE VERIFIED <repo> / RESCUE FAILED <repo>, and RESCUE END last.
# SMOKE=1: copies into $SMOKE_DIR/fakehub instead of HF.
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then cd "${SMOKE_DIR:?SMOKE_DIR}"; else cd "${W:-/root/s19}"; fi
set -a; [ -f .env ] && . ./.env; set +a
[ -d venv ] && . venv/bin/activate
NS13=PRIVATE_HF_ORG/chitonga-nllb-s19; NS33=PRIVATE_HF_ORG/chitonga-nllb33-s19
put() {
  if [ "$SMOKE" = 1 ]; then mkdir -p "fakehub/$2" && cp -r "$1"/. "fakehub/$2"/
  else timeout 5400 python upload_model.py "$1" "$2" "stage 19 RESCUE upload (run failed, stalled or capped), 24 Oct 2026" "$3"
  fi
}
w=0
while pgrep -f "upload_mode[l].py" > /dev/null 2>&1 && [ $w -lt 90 ]; do
  echo "RESCUE: an upload is still running, waiting ($w min)"; sleep 60; w=$((w + 1))
done
for d in s19/S19sw*/ckpt_[0-9]* s19/S19B13s*/ckpt_[0-9]* s19/S19P33/ckpt_[0-9]* s19/S19B33s*/ckpt_[0-9]* merged/S19B13m* merged/S19B33m*; do
  case "$d" in *_ct2|*_ct2/) continue ;; esac
  [ -f "$d/config.json" ] || continue
  [ -f "$d/UPOK" ] && continue
  python -c "
import json, os, sys
d = sys.argv[1]; i = os.path.join(d, 'model.safetensors.index.json')
ok = os.path.exists(os.path.join(d, 'model.safetensors')) or (os.path.exists(i) and all(
    os.path.exists(os.path.join(d, f)) for f in set(json.load(open(i))['weight_map'].values())))
sys.exit(0 if ok else 1)" "$d" || { echo "RESCUE SKIPPED $d (weights incomplete)"; continue; }
  base=facebook/nllb-200-distilled-1.3B
  case "$d" in
    merged/S19B13m*) r=${d##*/S19B13m}; repo=$NS13-B13m${r%s*}-s${r##*s} ;;
    merged/S19B33m*) r=${d##*/S19B33m}; repo=$NS33-B33m${r%s*}-s${r##*s}; base=facebook/nllb-200-3.3B ;;
    s19/S19sw*) run=$(basename "$(dirname "$d")"); repo=$NS13-sweep${run#S19sw} ;;
    s19/S19B13s*) run=$(basename "$(dirname "$d")"); repo=$NS13-B13-s${run##*s} ;;
    s19/S19P33/*) repo=$NS33-P1; base=facebook/nllb-200-3.3B ;;
    s19/S19B33s*) run=$(basename "$(dirname "$d")"); repo=$NS33-B33-s${run##*s}; base=facebook/nllb-200-3.3B ;;
    *) echo "RESCUE SKIPPED $d (no repo rule)"; continue ;;
  esac
  rm -f "$d/SAVED"
  c=${d%/}_ct2r
  if [ "$SMOKE" != 1 ] && [ ! -d "$c" ]; then
    ct2-transformers-converter --model "$d" --output_dir "$c" --quantization int8 --force > /dev/null 2>&1 || true
  fi
  [ -d "$c" ] && put "$c" "$repo-ct2-int8" "$base" && echo "RESCUE VERIFIED $repo-ct2-int8"
  if put "$d" "$repo" "$base"; then touch "$d/UPOK"; echo "RESCUE VERIFIED $repo"; else echo "RESCUE FAILED $repo (folder left on the machine)"; fi
done
echo "RESCUE END"
