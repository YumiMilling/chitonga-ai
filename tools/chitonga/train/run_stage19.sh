#!/bin/bash
# ON THE MACHINE (24 Oct 2026, the Jesuit): stage 19, TWO ARMS on ONE DigitalOcean GPU droplet (H100 80 GB / H200),
# launched only by launch_stage19.sh after Chris's go. Paid from the DO credit (Chris, 24 Oct: "stage 19 ... runs on
# DigitalOcean"). The data, the tags and the dose are the same for both arms: the only variable is model size.
#
# ARM 13 (NLLB 1.3B, from the installed a050):
#   1 SWEEP   the three <cls> doses S19a / S19b / S19c (20 / 25 / 30% of draws under <cls>), $SWEEP_STEPS updates each,
#             seed 1, --row-tags --tag-drop 0.1; scored by f19_sel.py (untagged / <cls> / <gen>)
#   2 SELECT  f19_select.py, the rule written before any output (PACK.md 4): forgetting untagged AND <gen>, teacher
#             memorisation, the synthetic GENERALISATION gap, no new probe confusion, no tag in any output; winner = the
#             lowest <cls> teacher-dev loss; ties to the LOWER dose
#   3 FULL    the winning dose, $STEPS updates, seeds 1 and 2 (Chris's rule), from a050
#   (MERGE    OFF by default, MERGE=1 to switch on. Chris, 24 Oct: "Forget about the blends for a start, we blend after
#             we tested the models." The kill test reads the RAW fine-tunes; blends are made afterwards, only for the
#             candidates that test well, from the uploaded fp32 weights.)
# ARM 33 (NLLB 3.3B, approved by Chris as ONE test arm; no Tonga-tuned 3.3B exists):
#   P1        facebook/nllb-200-3.3B, toi_Latn added from bem_Latn (Dale 2023, as stage 1-13), mix P33 = stage 18's S18b
#             weights on the stage 19 groups, UNTAGGED (as a050 was), lr 5e-5 (stage 6, 11 Oct: 1e-4 over-fits the 3.3B by
#             step ~600, 5e-5 was better), $P1_STEPS updates, seed 1: the 3.3B's a050
#   P2        the 1.3B arm's WINNING dose (same data, same tags), $STEPS updates, seeds 1 and 2, from P1
#   (MERGE    OFF by default, as arm 13. When it is made later, the anchor is P1, NOT facebook/nllb-200-3.3B: the base
#             has no toi_Latn row (256,204 vs 256,205 embeddings: merge_wise refuses the shapes) and no Tonga; P1 is the
#             3.3B's analogue of R / a050, the model the fine-tune started from (Wortsman et al. 2022). P1 is itself a
#             model to test: it is kept, full + int8.)
# KEPT: every model, full weights AND int8 (CTranslate2 int8 made HERE: memory rule convert-int8-on-pod), private repos
#   PRIVATE_HF_ORG/chitonga-nllb-s19-* and chitonga-nllb33-s19-*, verified on HF before the droplet is deleted.
# NO `set -x` (it would print the sourced .env into the fetched log). Greppable lines: CAP PREREQS / SETUP DONE / PACK OK
#   / TRAIN <tag> / step N / THROUGHPUT / CKPT_SAVED / <tag> TRAINING DONE / F19SEL / WINNER / MERGE / UPLOADED <repo> /
#   RUN uploads pending N / F19_DONE | F19_FAILED.
# SMOKE=1 (the Beelink CPU test): data from $SMOKE_DIR/s19, init $SMOKE_INIT (tiny NLLB-shaped 1.3B stand-in),
#   base $SMOKE_BASE (tiny 3.3B stand-in WITHOUT toi_Latn), tiny steps, uploads into fakehub.
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then W=${SMOKE_DIR:?SMOKE_DIR}; else W=${W:-/root/s19}; fi
mkdir -p "$W"; cd "$W" || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export PYTHONIOENCODING=utf-8      # the smoke on Windows died printing a SentencePiece piece; harmless on the droplet
DS=PRIVATE_HF_ORG/chitonga-stage19data
INIT_REPO=${INIT_REPO:-zamai-zambia/chitonga-nllb-a050}; BASE33=${BASE33:-facebook/nllb-200-3.3B}
ARMS=${ARMS:-"13 33"}; STEPS=${STEPS:-2500}; SWEEP_STEPS=${SWEEP_STEPS:-1500}; P1_STEPS=${P1_STEPS:-5000}
LR=2e-5; LR_P1=5e-5; TOKB=1536; MAXROWS=64; MAXLEN=256; BATCH=16; EVAL=250; SELB=16; TAGDROP=0.1
ALPHAS=${ALPHAS:-"0.5 0.7"}; MERGE=${MERGE:-0}; MIN_TPS=${MIN_TPS:-1500}; MIN_TPS33=${MIN_TPS33:-800}; TPS_STEP=200; PREC="--tf32"
DOSES=${DOSES:-"S19a S19b S19c"}; SEEDS=${SEEDS:-"1 2"}; SEEDS33=${SEEDS33:-$SEEDS}; INIT=init_model; B33=base33
NS13=PRIVATE_HF_ORG/chitonga-nllb-s19; NS33=PRIVATE_HF_ORG/chitonga-nllb33-s19
CT2BASE13=facebook/nllb-200-distilled-1.3B
if [ "$SMOKE" = 1 ]; then
  STEPS=${SMOKE_STEPS:-6}; SWEEP_STEPS=${SMOKE_SWEEP:-4}; P1_STEPS=${SMOKE_P1:-6}; EVAL=3; SELB=8; TPS_STEP=3
  INIT=${SMOKE_INIT:?SMOKE_INIT}; B33=${SMOKE_BASE:?SMOKE_BASE}; MIN_TPS=1; MIN_TPS33=1; PREC=""
  DOSES=${SMOKE_DOSES:-"S19a S19b"}; SEEDS=${SMOKE_SEEDS:-1}; ALPHAS=${SMOKE_ALPHAS:-"0.5"}
  echo "SMOKE MODE (Beelink CPU): arms '$ARMS' sweep $SWEEP_STEPS full $STEPS P1 $P1_STEPS doses '$DOSES' seeds '$SEEDS', uploads to $PWD/fakehub"
else
  free -g; df -h .; nvidia-smi --query-gpu=name,memory.total --format=csv
  MID=$(curl -s --max-time 10 http://169.254.169.254/metadata/v1/id)
  echo "CAP PREREQS provider=${PROV:-do} machine_id=${MID:-NONE} curl=$(command -v curl > /dev/null && echo yes || echo NO) pgrep=$(command -v pgrep > /dev/null && echo yes || echo NO) pkill=$(command -v pkill > /dev/null && echo yes || echo NO)"
  # the key-free backstop (v14 lesson): ends the droplet's work at the cap even if every guard with a key has died
  ( sleep $(( ${BACKSTOP_MIN:-420} * 60 )); echo "BACKSTOP $(date)"; bash rescue_stage19.sh > rescue_backstop.log 2>&1; poweroff ) > backstop.log 2>&1 &
  command -v python3 > /dev/null || { echo "F19_FAILED: NO PYTHON3"; exit 1; }
  python3 -m venv --system-site-packages venv 2> venv.log || (apt-get update -qq && apt-get install -y -qq python3-venv > apt.log 2>&1 && python3 -m venv --system-site-packages venv)
  . venv/bin/activate
  python -m pip install -q --upgrade pip > pip0.log 2>&1
  python -c "import torch; assert torch.cuda.is_available()" 2> /dev/null || python -m pip install -q torch > torch.log 2>&1
  # jinja2 pinned (v14 lesson: the DO image's 3.0.3 broke a template call); transformers pinned as every NLLB stage
  python -m pip install -q "transformers==4.46.3" sentencepiece sacrebleu huggingface_hub safetensors "ctranslate2==4.5.0" "jinja2>=3.1.0" protobuf > pip.log 2>&1
  python -c 'import torch,transformers,ctranslate2; print(torch.__version__, transformers.__version__, ctranslate2.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()' || { echo "F19_FAILED: PIP FAILED"; tail -5 pip.log; exit 1; }
  echo "SETUP DONE $(date)"
  ARMS="$ARMS" DS=$DS INIT_REPO=$INIT_REPO BASE33=$BASE33 python - <<'PY' > download.log 2>&1
import gzip, hashlib, json, os, shutil
from huggingface_hub import hf_hub_download, snapshot_download
os.makedirs("s19", exist_ok=True)
DS = os.environ["DS"]
for f in ["train.jsonl.gz", "mono.jsonl.gz", "tasks.jsonl.gz", "dev.jsonl", "test.en.txt", "test_hopgood.en.txt",
          "manifest.json", "preflight.json", "sel.jsonl", "teacher_dev_stage19.jsonl", "synthetic_s19_heldout.jsonl",
          "confusable_probes_final18.jsonl"]:
    p = hf_hub_download(DS, f, repo_type="dataset", local_dir="dl_s19")
    if f.endswith(".gz"):
        with gzip.open(p, "rb") as a, open("s19/" + f[:-3], "wb") as b:
            shutil.copyfileobj(a, b)
    else:
        shutil.copy(p, "s19/" + f)
pre = json.load(open("s19/preflight.json", encoding="utf-8"))
for f, want in pre["files"].items():
    got = hashlib.sha256(open("s19/" + f, "rb").read()).hexdigest()
    assert got == want, f"DATA SHA MISMATCH {f}: {got[:16]} != {want[:16]}"
print("DATA SHA OK", sorted(pre["files"]))
snapshot_download(os.environ["INIT_REPO"], local_dir="init_model")
if "33" in os.environ["ARMS"].split():
    snapshot_download(os.environ["BASE33"], local_dir="base33", allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin"])
print("DOWNLOADS OK")
PY
  grep -q "DOWNLOADS OK" download.log || { echo "F19_FAILED: DOWNLOAD FAILED"; tail -5 download.log; exit 1; }
  [ "$(sha256sum s19/train.jsonl | cut -d' ' -f1)" = "${DATA_SHA:?}" ] || { echo "F19_FAILED: DATA NOT THE LAUNCHED BUILD"; exit 1; }
fi
mixof() { python -c "
import json, sys
man = json.load(open('s19/manifest.json')); pre = json.load(open('s19/preflight.json'))
norm = lambda s: {k: float(v) for k, v in (kv.split(':') for kv in s.split(','))}
m = man['mixes'][sys.argv[1]]; w = norm(m)
assert w.get('choma', 0) == 0 and w.get('choma_med', 0) == 0, 'choma groups must be at weight 0'
assert w in [norm(x) for x in pre['mixes']], 'mix is not a checked mix'
print(m)" "$1"; }
for M in $DOSES P33; do mixof $M > /dev/null || { echo "F19_FAILED: MIX $M NOT CHECKED"; exit 1; }; done
echo "PACK OK data $(sha256sum s19/train.jsonl | cut -c1-16) arms '$ARMS'"
echo "RECIPE arm13: init $INIT_REPO lr $LR sweep $(echo $DOSES | tr -d ' ') x $SWEEP_STEPS, full $STEPS x seeds $(echo $SEEDS | tr -d ' '), --row-tags --tag-drop $TAGDROP, merge $MERGE | arm33: $BASE33 +toi_Latn<-bem_Latn, P1 P33 untagged lr $LR_P1 x $P1_STEPS, P2 = arm13 winner x $STEPS x seeds, merge $MERGE"
set +e
OK=1; UPIDS=""; UPLOGS=""
if [ "$SMOKE" = 1 ]; then
  up() { mkdir -p "fakehub/$2" && cp -r "$1"/. "fakehub/$2"/ && echo "UPLOADED $2"; }
  if [ "${SMOKE_CT2:-1}" = 1 ]; then      # the REAL int8 conversion, on the tiny model (seconds)
    ct2() { ct2-transformers-converter --model "$1" --output_dir "$2" --quantization int8 --force > "$3" 2>&1 && test -s "$2/model.bin"; }
  else ct2() { mkdir -p "$2" && cp "$1"/config.json "$2"/ && echo smoke > "$2/model.bin"; }; fi
else
  up() { timeout 5400 python upload_model.py "$1" "$2" "$3" "$4" && echo "UPLOADED $2"; }
  ct2() { ct2-transformers-converter --model "$1" --output_dir "$2" --quantization int8 --force > "$3" 2>&1; }
fi
upload_both() {   # folder repo description logname base: int8 FIRST (what the Beelink downloads), then the full weights
  local d=$1 c=${1%/}_ct2
  ct2 "$d" "$c" "$4.ct2.log" || { echo "CT2 FAILED $2"; return 1; }
  up "$c" "$2-ct2-int8" "$3 (CTranslate2 int8)" "$5" || { echo "UPLOAD FAILED $2-ct2-int8"; return 1; }
  up "$d" "$2" "$3" "$5" || { echo "UPLOAD FAILED $2"; return 1; }
  rm -rf "$c"; touch "$d/UPOK"; return 0
}
bg_upload() {     # folder repo description logname base
  ( upload_both "$1" "$2" "$3" "$4" "$5" ) > "$4.log" 2>&1 &
  UPIDS="$UPIDS $!"; UPLOGS="$UPLOGS $4"
}
train_one() {     # tag mix steps seed init lr tags(0/1) extra
  local tg=$1 rt=""
  [ "$7" = 1 ] && rt="--row-tags --tag-drop $TAGDROP"
  python train_nllb.py --run s19 --tag $tg --mix "$2" --seed $4 $8 --tgt-code toi_Latn --both-directions --steps $3 \
    --grad-accum 1 --batch $BATCH --token-budget $TOKB --max-rows $MAXROWS --max-len $MAXLEN --attn sdpa --lr $6 \
    --schedule linear --warmup-frac 0.05 --label-smoothing 0 --weight-decay 0.01 --eval-every $EVAL --patience 100000 \
    --dev-both $PREC --save-steps $3 --log-detail --tps-check-step $TPS_STEP $rt \
    $( [ -n "$5" ] && echo "--init-model $5 --expect-code-id 256204" )
}
score() { python f19_sel.py --model "$1" --run s19 --out "sel/f19_$2.json" --batch $SELB --max-len $MAXLEN || echo "SEL FAILED $2"; }
mkdir -p sel merged
# ================================================================ ARM 13
W13=""
if echo " $ARMS " | grep -q " 13 "; then
  score $INIT a050
  for M in $DOSES; do
    tg=S19sw$M; echo "TRAIN $tg (arm 13 sweep, dose $M) seed 1"
    train_one $tg "$(mixof $M)" $SWEEP_STEPS 1 $INIT $LR 1 "--min-tps $MIN_TPS"
    rc=$?; [ "$rc" = 3 ] && { echo "F19_FAILED: THROUGHPUT LOW"; exit 1; }
    d=s19/$tg/ckpt_$SWEEP_STEPS
    if [ $rc != 0 ] || [ ! -f $d/SAVED ]; then echo "SWEEP RUN FAILED $M"; OK=0; continue; fi
    rm -f $d/SAVED; echo "$tg TRAINING DONE"
    score $d $M
    bg_upload $d $NS13-sweep$M "stage 19 arm 13 dose sweep $M ($SWEEP_STEPS updates from $INIT_REPO, seed 1, tags <cls>/<gen>/<syn>), 24 Oct 2026" up_sw$M $CT2BASE13
  done
  python f19_select.py sel "$(echo $DOSES | tr ' ' ',')" | tee sel/select.txt
  W13=$(grep -o '^WINNER S19[a-c]' sel/select.txt | tail -1 | cut -d' ' -f2)
  # SMOKE only: tiny random stand-ins measure nothing, so the smoke forces the winner path (and says so) to exercise the
  # full runs, P2, the uploads, the int8 builds and the rescue; on the droplet this line never fires (SMOKE=0)
  if [ "$SMOKE" = 1 ] && [ -n "${SMOKE_FORCE_WINNER:-}" ]; then
    echo "SMOKE: forcing the winner $SMOKE_FORCE_WINNER (the selector said '${W13:-none}')"; W13=$SMOKE_FORCE_WINNER
  fi
  [ -n "$W13" ] || { echo "F19_FAILED: NO WINNER"; for u in $UPIDS; do wait $u; done; exit 1; }
  echo "WINNER $W13"
  for SEED in $SEEDS; do
    tg=S19B13s$SEED; echo "TRAIN $tg (arm 13 full, dose $W13) seed $SEED"
    train_one $tg "$(mixof $W13)" $STEPS $SEED $INIT $LR 1 "--min-tps $MIN_TPS"
    rc=$?; d=s19/$tg/ckpt_$STEPS
    if [ $rc != 0 ] || [ ! -f $d/SAVED ]; then echo "RUN FAILED $tg"; OK=0; continue; fi
    rm -f $d/SAVED; echo "$tg TRAINING DONE"
    score $d B13_s$SEED
    bg_upload $d $NS13-B13-s$SEED "stage 19 arm 13 FULL run, dose $W13, seed $SEED, $STEPS updates at $LR from $INIT_REPO, tags, 24 Oct 2026" up_$tg $CT2BASE13
    [ "$MERGE" = 1 ] && for A in $ALPHAS; do
      an=$(echo $A | tr -d '.'); m=merged/S19B13m${an}s$SEED; echo "MERGE $tg with a050 at alpha $A"
      if python merge_wise.py $INIT $d $A $m > merge_${tg}_$an.log 2>&1 && grep -q MERGE_OK merge_${tg}_$an.log; then
        score $m B13m${an}_s$SEED
        bg_upload $m $NS13-B13m$an-s$SEED "stage 19 arm 13 candidate: WiSE (1-$A) x $INIT_REPO + $A x run seed $SEED, 24 Oct 2026" up_B13m${an}s$SEED $CT2BASE13
      else echo "MERGE FAILED $tg alpha $A"; OK=0; fi
    done
  done
fi
# ================================================================ ARM 33
if echo " $ARMS " | grep -q " 33 "; then
  W33=${W13:-${W33_DOSE:-S19b}}
  tg=S19P33; echo "TRAIN $tg (arm 33 phase 1: $BASE33 + toi_Latn, P33 untagged) seed 1"
  train_one $tg "$(mixof P33)" $P1_STEPS 1 "" $LR_P1 0 "--base $B33 --new-code toi_Latn --init-from bem_Latn --min-tps $MIN_TPS33"
  rc=$?; [ "$rc" = 3 ] && { echo "F19_FAILED: THROUGHPUT LOW (3.3B)"; exit 1; }
  P1=s19/$tg/ckpt_$P1_STEPS
  if [ $rc != 0 ] || [ ! -f $P1/SAVED ]; then echo "F19_FAILED: RUN FAILED $tg"; for u in $UPIDS; do wait $u; done; exit 1; fi
  rm -f $P1/SAVED; echo "$tg TRAINING DONE"
  score $P1 P33
  bg_upload $P1 $NS33-P1 "stage 19 arm 33 phase 1: $BASE33 + toi_Latn (from bem_Latn), mix P33 untagged, $P1_STEPS updates at $LR_P1, seed 1, 24 Oct 2026" up_P33 $BASE33
  for SEED in $SEEDS33; do
    tg=S19B33s$SEED; echo "TRAIN $tg (arm 33 phase 2, dose $W33) seed $SEED"
    train_one $tg "$(mixof $W33)" $STEPS $SEED $P1 $LR 1 "--min-tps $MIN_TPS33"
    rc=$?; d=s19/$tg/ckpt_$STEPS
    if [ $rc != 0 ] || [ ! -f $d/SAVED ]; then echo "RUN FAILED $tg"; OK=0; continue; fi
    rm -f $d/SAVED; echo "$tg TRAINING DONE"
    score $d B33_s$SEED
    bg_upload $d $NS33-B33-s$SEED "stage 19 arm 33 phase 2, dose $W33, seed $SEED, $STEPS updates at $LR from phase 1, tags, 24 Oct 2026" up_$tg $BASE33
    [ "$MERGE" = 1 ] && for A in $ALPHAS; do
      an=$(echo $A | tr -d '.'); m=merged/S19B33m${an}s$SEED; echo "MERGE $tg with P1 at alpha $A"
      if python merge_wise.py $P1 $d $A $m > merge_${tg}_$an.log 2>&1 && grep -q MERGE_OK merge_${tg}_$an.log; then
        score $m B33m${an}_s$SEED
        bg_upload $m $NS33-B33m$an-s$SEED "stage 19 arm 33 candidate: WiSE (1-$A) x phase 1 + $A x phase 2 seed $SEED, 24 Oct 2026" up_B33m${an}s$SEED $BASE33
      else echo "MERGE FAILED $tg alpha $A"; OK=0; fi
    done
  done
fi
# the uploads are the slow tail: a heartbeat line a minute so the launcher's watchdog is never blind (stage 18 F12)
while :; do
  alive=""; for u in $UPIDS; do kill -0 $u 2> /dev/null && alive="$alive $u"; done
  [ -z "$alive" ] && break
  echo "RUN uploads pending $(echo $alive | wc -w)"; sleep ${UPLOAD_TICK:-60}
done
for u in $UPIDS; do wait $u || OK=0; done
for l in $UPLOGS; do grep -q "UPLOADED" $l.log 2> /dev/null || { echo "NOT UPLOADED: $l"; OK=0; }; done
grep -h "UPLOADED" up_*.log 2> /dev/null | sort
if [ $OK = 1 ]; then echo "F19_DONE, UPLOADS VERIFIED $(date)"; else echo "F19_FAILED (see NOT UPLOADED / FAILED lines): DO NOT TERMINATE before rescue"; fi
