#!/bin/bash
# FINAL translator round of the plateau, on ONE GPU machine (23 Oct 2026, rewritten 24 Oct; the Jesuit).
# Launched only by launch_final18.sh, and only after Chris's go.
#
# 24 Oct, Chris: "We need to leverage on those teacher lines ... also the weights need to be optimized." So the round
# is no longer one recipe at a guessed weight. It is:
#   1. SWEEP   three DOSES of the teacher material (S18a half, S18b planned, S18c double; build_stage18.py writes all
#              three mixes and check_build checks each on these exact files), $SWEEP_STEPS updates each, seed 1, from
#              a050. Scored by f18_sel.py: old-dev NLL (forgetting), the 14-line teacher DEV slice (never trained),
#              verbatim reproduction per group (memorisation), the 13 confusable probes decoded RAW.
#   2. SELECT  f18_select.py applies the rule written before any output existed (eligible = no forgetting, no
#              memorisation, no forbidden probe word; winner = lowest teacher-dev loss; ties to the LOWER dose).
#   3. FULL    the winning mix, $STEPS updates, TWO SEEDS (Chris's rule), from a050.
#   4. MERGE   each seed with a050 at alpha 0.3 / 0.5 / 0.7 (merge_wise.py, Wortsman et al. 2022). The alpha is a free
#              dial: the kill test picks it by the gates (STRATEGY A6). No second training phase.
# The A16 control arm is DROPPED (pre-launch review F7): nothing in the kill test reads it, it cost ~USD 1 and ~100
# pod-minutes, and the sweep's three doses are a better ablation of the same question ("what does the teacher data
# buy?"), measured rather than asserted.
#
# DATA: two private HF datasets, each sha256-checked against its own preflight.json BEFORE training:
#   PRIVATE_HF_ORG/chitonga-stage18data (build_stage18.py: stage 17 + the teacher groups of 21-24 Oct + the synthetic
#   teacher-register group + the confusable contrast), and chitonga-stage16r3 for the dev / sel files' identity check.
# INIT: $INIT_REPO = the installed 50% blend zamai-zambia/chitonga-nllb-a050 (toi_Latn id 256204).
# RECIPE (AB_PREREG, unchanged): Adafactor lr 2e-5, warmup 5% then linear decay, label smoothing 0, token budget 1536 /
#   max 64 rows / max-len 256, SDPA, TF32, dev both directions every 250 (logged), weights kept at the last step.
# UPLOADS: int8 FIRST for every model (F12); full weights for the two full runs; the sweep checkpoints and the merges
#   are uploaded as int8 only. Recorded exception to "always keep trained models": a sweep checkpoint is a selection
#   artefact reproducible from the recorded data + seed + recipe, and a merge is (1-alpha) x a050 + alpha x B18, both
#   of which are kept in full. The rescue uploads whatever lacks UPOK if the job fails.
# Progress lines for the launcher: RECIPE / SETUP DONE / TRAIN / step N / THROUGHPUT / CKPT_SAVED / F18SEL / WINNER /
#   MERGE / UPLOADED / RUN uploads pending N / F18_DONE | F18_FAILED
# SMOKE=1 (the Beelink test): data from $SMOKE_DIR/s18, init $SMOKE_INIT, tiny steps, uploads into fakehub.
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then cd "${SMOKE_DIR:?SMOKE_DIR}"; else cd /root; fi
set -a; [ -f .env ] && . ./.env; set +a
set -e
D16=PRIVATE_HF_ORG/chitonga-stage16r3; D18=PRIVATE_HF_ORG/chitonga-stage18data
INIT_REPO=${INIT_REPO:-zamai-zambia/chitonga-nllb-a050}
STEPS=${STEPS:-2500}; SWEEP_STEPS=${SWEEP_STEPS:-1500}; LR=2e-5; TOKB=1536; MAXROWS=64; MAXLEN=256; BATCH=16; EVAL=250
SELB=16; ALPHAS=${ALPHAS:-"0.3 0.5 0.7"}; MAIN_ALPHA=0.5
MIN_TPS=${MIN_TPS:-1500}; TPS_STEP=200; PREC="--tf32"; INIT=init_model; NS=PRIVATE_HF_ORG/chitonga-nllb-f18
DOSES=${DOSES:-"S18a S18b S18c"}; SEEDS=${SEEDS:-"1 2"}
if [ "$SMOKE" = 1 ]; then
  STEPS=${SMOKE_STEPS:-6}; SWEEP_STEPS=${SMOKE_SWEEP:-4}; EVAL=3; INIT=${SMOKE_INIT:?SMOKE_INIT}; SELB=8; TPS_STEP=3
  MIN_TPS=${SMOKE_MIN_TPS:-1}; PREC=${SMOKE_PREC:-"--tf32"}; DOSES=${SMOKE_DOSES:-"S18a S18b"}; SEEDS=${SMOKE_SEEDS:-1}
  ALPHAS=${SMOKE_ALPHAS:-"0.5 0.7"}
  echo "SMOKE MODE (Beelink test): sweep $SWEEP_STEPS full $STEPS doses '$DOSES' seeds '$SEEDS', uploads to $PWD/fakehub"
else
  command -v python > /dev/null || ln -s "$(command -v python3)" /usr/local/bin/python
  python -m pip --version > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq python3-pip) > apt.log 2>&1
  python -c "import torch" 2> /dev/null || python -m pip install -q torch > torch.log 2>&1
  python -m pip install -q "transformers==4.46.3" sentencepiece sacrebleu huggingface_hub safetensors "ctranslate2==4.5.0" > pip.log 2>&1
  D16=$D16 D18=$D18 INIT_REPO=$INIT_REPO python - <<'PY'
import gzip, hashlib, json, os, shutil
from huggingface_hub import hf_hub_download, snapshot_download
for arm, DS in (("s16", os.environ["D16"]), ("s18", os.environ["D18"])):
    os.makedirs(arm, exist_ok=True)
    for f in ["train.jsonl.gz", "mono.jsonl.gz", "tasks.jsonl.gz", "dev.jsonl", "test.en.txt", "test_hopgood.en.txt",
              "manifest.json", "preflight.json", "sel.jsonl"]:
        p = hf_hub_download(DS, f, repo_type="dataset", local_dir="dl_" + arm)
        if f.endswith(".gz"):
            with gzip.open(p, "rb") as a, open(arm + "/" + f[:-3], "wb") as b: shutil.copyfileobj(a, b)
        else:
            shutil.copy(p, arm + "/" + f)
    pre = json.load(open(arm + "/preflight.json", encoding="utf-8"))
    for f, want in pre["files"].items():
        got = hashlib.sha256(open(arm + "/" + f, "rb").read()).hexdigest()
        assert got == want, f"DATA SHA MISMATCH {arm} {f}: {got[:16]} != {want[:16]}"
    print("DATA SHA OK", arm, DS, sorted(pre["files"]))
# the judging slices travel with the stage 18 data: the teacher DEV slice (never trained) and the 13 probes
for f in ("teacher_dev_final18.jsonl", "confusable_probes_final18.jsonl"):
    shutil.copy(hf_hub_download(os.environ["D18"], f, repo_type="dataset", local_dir="dl_s18"), "s18/" + f)
for f in ("dev.jsonl", "sel.jsonl", "test.en.txt"):
    a, b = (hashlib.sha256(open(x + "/" + f, "rb").read()).hexdigest() for x in ("s16", "s18"))
    assert a == b, f"{f} differs between the datasets"
snapshot_download(os.environ["INIT_REPO"], local_dir="init_model")
assert any(f.endswith(".safetensors") for f in os.listdir("init_model")), "init has no safetensors (merge needs them)"
print("INIT", os.environ["INIT_REPO"], sorted(os.listdir("init_model"))[:6])
PY
  python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"
  # F13 (pre-launch review): a cap that lives ON THIS MACHINE. Every other guard (the launcher's watchdog, its hard-cap
  # subshell, pod_overwatch.py) is a process on the Beelink, which stood at 97.6% RAM during the review; if that box
  # dies the pod would run until the balance is gone. runpodctl and RUNPOD_POD_ID are injected by RunPod; if either is
  # missing the log says so and the operator knows the Beelink guards are the only ones.
  if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    ( sleep $(( ${F18_CAP_MIN:-400} * 60 )); echo "MACHINE CAP ${F18_CAP_MIN:-400} min reached $(date)"
      bash rescue_final18.sh > rescue_cap.log 2>&1; runpodctl remove pod "$RUNPOD_POD_ID" ) > machine_cap.log 2>&1 &
    echo "MACHINE CAP armed: ${F18_CAP_MIN:-400} min, pod $RUNPOD_POD_ID (rescue, then self-remove)"
  else
    echo "MACHINE CAP NOT ARMED: runpodctl=$(command -v runpodctl || echo none) RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset}"
  fi
fi
mixof() { python -c "
import json, sys
arm, name = sys.argv[1], sys.argv[2]
man = json.load(open(arm + '/manifest.json')); pre = json.load(open(arm + '/preflight.json'))
norm = lambda s: {k: float(v) for k, v in (kv.split(':') for kv in s.split(','))}
m = man['mixes'][name]; w = norm(m)
assert w['choma'] == 0 and w['choma_med'] == 0, 'choma groups must be at weight 0 (the children test stories)'
assert w in [norm(x) for x in pre['mixes']], 'mix is not a checked mix'
print(m)" "$1" "$2"; }
for M in $DOSES; do mixof s18 $M > /dev/null || { echo "MIX $M NOT CHECKED"; echo F18_FAILED; exit 1; }; done
echo "RECIPE init $INIT_REPO lr $LR linear warmup 0.05 ls 0 sweep $(echo $DOSES | tr -d ' ') x $SWEEP_STEPS full $STEPS seeds $(echo $SEEDS | tr -d ' ') token-budget $TOKB max-rows $MAXROWS max-len $MAXLEN $PREC sdpa eval $EVAL gate $MIN_TPS wise $(echo $ALPHAS | tr ' ' ',')"
echo "SETUP DONE f18"
set +e
if [ "$SMOKE" = 1 ]; then
  up() { mkdir -p "fakehub/$2" && cp -r "$1"/. "fakehub/$2"/ && echo "UPLOADED $2"; }
else
  up() { timeout 3600 python upload_model.py "$1" "$2" "$3" facebook/nllb-200-distilled-1.3B && echo "UPLOADED $2"; }
fi
ct2() { ct2-transformers-converter --model "$1" --output_dir "$2" --quantization int8 --force > "$3" 2>&1; }
upload_ckpt() {   # folder repo description logname [full]: int8 made HERE (feedback-convert-int8-on-pod), int8 FIRST
  local d=$1 c=${1%/*}/ct2_${1##*/} full=${5:-1}
  ct2 $d $c $4.ct2.log || { echo "CT2 FAILED $2"; return 1; }
  up $c $2-ct2-int8 "$3 (CTranslate2 int8)" || { echo "UPLOAD FAILED $2-ct2-int8 (folders kept)"; return 1; }
  if [ "$full" = 1 ]; then
    up $d $2 "$3" || { echo "UPLOAD FAILED $2 (folders kept)"; return 1; }
  fi
  rm -rf $c; touch $d/UPOK; return 0
}
train_one() {     # tag mix steps seed
  python train_nllb.py --run s18 --tag $1 --mix "$2" --seed $4 --init-model $INIT --tgt-code toi_Latn \
    --expect-code-id 256204 --both-directions --steps $3 --grad-accum 1 --batch $BATCH --token-budget $TOKB \
    --max-rows $MAXROWS --max-len $MAXLEN --attn sdpa --lr $LR --schedule linear --warmup-frac 0.05 \
    --label-smoothing 0 --weight-decay 0.01 --eval-every $EVAL --patience 100000 --dev-both $PREC --save-steps $3 \
    --log-detail --min-tps $MIN_TPS --tps-check-step $TPS_STEP
}
OK=1; UPIDS=""; UPLOGS=""
mkdir -p sel merged
# ---- 0. the init's own scores: every candidate is read against a050, not against zero
python f18_sel.py --model $INIT --run s18 --out sel/f18_a050.json --batch $SELB --max-len $MAXLEN || { echo "SEL FAILED init"; echo F18_FAILED; exit 1; }
# ---- 1. the sweep: one seed per dose, $SWEEP_STEPS updates
for M in $DOSES; do
  tg=F18sw$M; mkdir -p s18/$tg
  echo "TRAIN $tg (sweep, dose $M) seed 1"
  train_one $tg "$(mixof s18 $M)" $SWEEP_STEPS 1
  rc=$?; echo "TRAIN EXIT $tg rc $rc"
  [ "$rc" = 3 ] && { echo "THROUGHPUT LOW: stopping (gate $MIN_TPS real tokens/s)"; echo F18_FAILED; exit 1; }
  d=s18/$tg/ckpt_$SWEEP_STEPS
  if [ $rc != 0 ] || [ ! -f $d/SAVED ]; then echo "SWEEP RUN FAILED $M"; OK=0; continue; fi
  rm -f $d/SAVED
  python f18_sel.py --model $d --run s18 --out sel/f18_$M.json --batch $SELB --max-len $MAXLEN || echo "SEL FAILED $M"
  ( upload_ckpt $d $NS-sweep$M-step$SWEEP_STEPS "final round mix sweep, dose $M, $SWEEP_STEPS updates at lr $LR from $INIT_REPO, seed 1 (selection run), 24 Oct 2026" up_sw$M 0 \
      && echo "RUN sweep$M UPLOADED" ) > up_sw$M.log 2>&1 &
  UPIDS="$UPIDS $!"; UPLOGS="$UPLOGS up_sw$M"
done
# ---- 2. the pre-registered selection
python f18_select.py sel "$(echo $DOSES | tr ' ' ',')" | tee sel/select.txt
W=$(grep -o '^WINNER S18[a-c]' sel/select.txt | tail -1 | cut -d' ' -f2)
[ -n "$W" ] || { echo "NO WINNER"; for u in $UPIDS; do wait $u; done; echo F18_FAILED; exit 1; }
echo "WINNER $W"
MIX="$(mixof s18 $W)"
# ---- 3. the full runs: the winning dose, two seeds
for SEED in $SEEDS; do
  tg=F18B18s$SEED; mkdir -p s18/$tg
  echo "TRAIN $tg (full, mix $W) seed $SEED"
  train_one $tg "$MIX" $STEPS $SEED
  rc=$?; echo "rc $rc" > s18/$tg/TRAIN_EXIT; echo "TRAIN EXIT $tg rc $rc"
  [ "$rc" = 3 ] && { echo "THROUGHPUT LOW: stopping"; echo F18_FAILED; exit 1; }
  d=s18/$tg/ckpt_$STEPS
  if [ $rc != 0 ] || [ ! -f $d/SAVED ]; then echo "RUN FAILED $tg"; OK=0; continue; fi
  rm -f $d/SAVED
  python f18_sel.py --model $d --run s18 --out sel/f18_B18_s$SEED.json --batch $SELB --max-len $MAXLEN || echo "SEL FAILED $tg"
  ( upload_ckpt $d $NS-B18-s$SEED-step$STEPS "final round FULL run, mix $W (chosen by the pre-registered sweep rule), seed $SEED, $STEPS updates at lr $LR from $INIT_REPO, 24 Oct 2026" up_$tg \
      && echo "RUN $tg UPLOADED" ) > up_$tg.log 2>&1 &
  UPIDS="$UPIDS $!"; UPLOGS="$UPLOGS up_$tg"
  # ---- 4. the alpha sweep: the merge ratio is a measured dial, not a guess (STRATEGY A6)
  for A in $ALPHAS; do
    an=$(echo $A | tr -d '.')
    m=merged/F18B18m${an}s$SEED
    echo "MERGE $tg with init at alpha $A"
    if python merge_wise.py $INIT $d $A $m > merge_${tg}_$an.log 2>&1 && grep -q MERGE_OK merge_${tg}_$an.log; then
      printf "# chitonga-nllb final round candidate B18m (alpha %s, seed %s)\n\nWiSE-FT: (1 - %s) x %s + %s x run B18 seed %s (stage 18 data, mix %s, %s updates). run_final18.sh / merge_wise.py, 24 Oct 2026. The alpha is chosen by the kill test's gates (PACK.md 6-7).\n" $A $SEED $A $INIT_REPO $A $SEED $W $STEPS > $m/README.md
      python f18_sel.py --model $m --run s18 --out sel/f18_B18m${an}_s$SEED.json --batch $SELB --max-len $MAXLEN || echo "SEL FAILED B18m$an s$SEED"
      full=0; [ "$A" = "$MAIN_ALPHA" ] && full=1
      ( upload_ckpt $m $NS-B18m$an-s$SEED "final round candidate: WiSE-FT (1-$A) x $INIT_REPO + $A x run B18 seed $SEED, mix $W, 24 Oct 2026" up_B18m${an}s$SEED $full \
          && echo "RUN B18m${an}s$SEED UPLOADED" ) > up_B18m${an}s$SEED.log 2>&1 &
      UPIDS="$UPIDS $!"; UPLOGS="$UPLOGS up_B18m${an}s$SEED"
    else
      echo "MERGE FAILED $tg alpha $A: $(tail -2 merge_${tg}_$an.log)"; OK=0
    fi
  done
done
# F12: the launcher reads progress from the last TRAIN / step / SEL / MERGE / RUN line and the log's size; the uploads
# write only to up_*.log, so without this heartbeat the watchdog is blind for the whole final queue and then kills
# healthy uploads. One line a minute, naming how many are still in flight.
while :; do
  alive=""
  for u in $UPIDS; do kill -0 $u 2> /dev/null && alive="$alive $u"; done
  [ -z "$alive" ] && break
  echo "RUN uploads pending $(echo $alive | wc -w)"
  sleep 60
done
for u in $UPIDS; do wait $u || OK=0; done
cat up_*.log | grep -E "UPLOADED|FAILED" || true
for l in $UPLOGS; do grep -q "UPLOADED" $l.log 2> /dev/null || OK=0; done
NEXP=0
for M in $DOSES; do NEXP=$((NEXP + 1)); done
for SEED in $SEEDS; do NEXP=$((NEXP + 1)); for A in $ALPHAS; do NEXP=$((NEXP + 1)); done; done
[ $(echo $UPLOGS | wc -w) = $NEXP ] || { echo "MODEL COUNT $(echo $UPLOGS | wc -w) != $NEXP expected"; OK=0; }
if [ $OK = 1 ]; then echo F18_DONE; else echo F18_FAILED; fi
