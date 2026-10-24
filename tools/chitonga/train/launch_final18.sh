#!/bin/bash
# FINAL translator round of the plateau: the launcher (23 Oct 2026, the Jesuit; stream D of Chris's 11-hour plan;
# runs/2026-10-23_final_translator_data/PACK.md; data rebuilt 24 Oct with the 23-24 Oct teacher groups and the frozen
# teacher test, RUN_PACKAGE.md). READY, NOT LAUNCHED: the go is Chris's (DECISIONS: no pod without his go). Go command:
#     python -X utf8 tools/chitonga/train/pod_overwatch.py 410 7 &   # the independent overwatch FIRST (cap 410 min)
#     bash tools/chitonga/train/launch_final18.sh data && bash tools/chitonga/train/launch_final18.sh run
# Three commands:
#     bash tools/chitonga/train/launch_final18.sh data [--dry]   # K0: runs/2026-10-23_final_translator_data -> NEW private
#                                                                # HF dataset PRIVATE_HF_ORG/chitonga-stage18data, read back
#     bash tools/chitonga/train/launch_final18.sh run [--dry]    # ONE RunPod 48 GB machine: run_final18.sh (4 runs in turn,
#                                                                # 2 WiSE merges, 6 models x full + int8); --dry = every
#                                                                # local check and the plan, NO machine
#     python -X utf8 tools/chitonga/train/final18_killtest.py ... # afterwards, on the Beelink (PACK.md 6)
# Plan (PACK.md 8, 24 Oct): mix sweep S18a / S18b / S18c x 1,500 updates from a050 (one seed), the pre-registered
#   selection (f18_select.py), the winning mix x seeds 1 2 x 2,500 updates, WiSE merges at 0.3 / 0.5 / 0.7; A16 dropped.
#   EXPECTED ~275 min (the delta review's count: twelve f18_sel runs + training); A40 $0.35-0.59/h = ~$1.6-2.7.
#   (stage 16 measured) + setup, SEL, merges, int8, uploads: EXPECTED ~300 min; A40 community $0.35-0.59/h = ~$1.6-3.0.
# Pod rules (as launch_stage16.sh / launch_ab17.sh; feedback-pod-scripts-guard, -tell-chris-when-pod-starts,
#   -always-keep-trained-models, -convert-int8-on-pod):
#   - RunPod 48 GB only: A40 first, then RTX A6000, L40S, RTX 6000 Ada (never 24 GB); the standard PyTorch image
#     (runpod.py IMAGE, torch preinstalled).
#   - pod_overwatch.py is started by the operator FIRST (cap above this launcher's: --cap 410).
#   - phone notes (tools/notify.py): MACHINE CREATED with one line of purpose; CONFIRMED GONE with results + cost.
#   - SAFETY CAP: min(BUDGET_USD at the card's price, 390 min); the watchdog stops 20 min before it (kill, RESCUE uploads,
#     fetch, terminate); a local timer destroys the machine at the cap.
#   - 20 min allowed to the first SSH; one file per scp, each checked on the machine (test -s); ssh only through X();
#     nohup; EXIT trap: RESCUE (copies every trained / merged folder without UPOK to HF) THEN fetch THEN terminate, and
#     terminate() confirms the machine is gone and phones Chris if not. Never --no-save.
set -u
cd "$(dirname "$0")/../../.."
R=internal/chitonga_train/runs/2026-10-23_final_translator_data
T=$(grep "^HF_TOKEN_CHRIS=" .env.local | cut -d= -f2-)
DS=PRIVATE_HF_ORG/chitonga-stage18data
NS=PRIVATE_HF_ORG/chitonga-nllb-f18
INIT_REPO=${INIT_REPO:-zamai-zambia/chitonga-nllb-a050}
STEPS=${STEPS:-2500}
SWEEP_STEPS=${SWEEP_STEPS:-1500}
MODE=${1:?usage: launch_final18.sh data|run [--dry]}
DRY=$([ "${2:-}" = "--dry" ] && echo 1 || echo 0)
mkdir -p $R/logs
if [ "$MODE" = data ]; then
  F18_DRY=$DRY HF_TOKEN="$T" python -X utf8 - <<'EOF'
"""K0 data step: checks first, then the upload, then every file read back from HF and compared."""
import gzip, hashlib, json, os, random, shutil, tempfile
from collections import Counter
R = "internal/chitonga_train/runs/2026-10-23_final_translator_data"
R16 = "internal/chitonga_train/runs/2026-10-21_stage16"
DS = "PRIVATE_HF_ORG/chitonga-stage18data"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
pre = json.load(open(os.path.join(R, "preflight.json"), encoding="utf-8"))
man = json.load(open(os.path.join(R, "manifest.json"), encoding="utf-8"))
for f, want in pre["files"].items():
    assert sha(os.path.join(R, f)) == want, f"{f} changed since check_build (re-run build_stage18.py)"
assert man["guarded_rows_left"] == 0, "guarded rows left in the build"
for f in ("dev.jsonl", "sel.jsonl", "test.en.txt", "test_hopgood.en.txt"):
    assert sha(os.path.join(R, f)) == sha(os.path.join(R16, f)), f"{f} differs from stage 16: the arms must share tests"
norm = lambda s: {k: float(v) for k, v in (kv.split(":") for kv in s.split(","))}
w = norm(man["mixes"]["S18"])
for _m in ("S18a", "S18b", "S18c"):      # every dose of the sweep must be a mix check_build checked on these files
    assert norm(man["mixes"][_m]) in [norm(x) for x in pre["mixes"]], f"mix {_m} is not a checked mix"
assert w in [norm(x) for x in pre["mixes"]], "mix S18 is not a checked mix"
assert w["choma"] == 0 and w["choma_med"] == 0, "choma groups must be at weight 0"
for g in ("t_teach", "t_teach_flag", "t_word", "t_word_flag", "conf23", "rain24", "synth24"):
    assert w.get(g, 0) > 0, f"new group {g} not drawn"
# 24 Oct: the frozen teacher test (44 lines) must be frozen (sha file) and absent from train / dev on both sides
tt = os.path.join(R, "teacher_test_final18.jsonl")
assert os.path.exists(tt + ".sha256"), "teacher test not frozen: python final18_killtest.py freeze"
assert open(tt + ".sha256").read().split()[0] == hashlib.sha256(open(tt, "rb").read()).hexdigest(), "teacher test edited after freeze"
import sys as _s; _s.path.insert(0, "tools/chitonga/train"); import mix_groups as _MG
tk = set()
for r in (json.loads(l) for l in open(tt, encoding="utf-8")):
    tk.add(_MG._n(r["en"])); tk.add(_MG._n(r["toi"])); tk |= {_MG._n(t) for t in r.get("alternatives", [])}
trows = [json.loads(l) for f in ("train.jsonl", "dev.jsonl") for l in open(os.path.join(R, f), encoding="utf-8") if l.strip()]
tl = [r.get("id") for r in trows if _MG._n(r.get("en")) in tk or _MG._n(r.get("toi")) in tk]
# F3 (pre-launch review): tasks and the copy pool as well, not only train / dev
for r in (json.loads(l) for l in open(os.path.join(R, "tasks.jsonl"), encoding="utf-8") if l.strip()):
    if _MG._n((r.get("src") or "").replace("[BT] ", "")) in tk or _MG._n(r.get("tgt")) in tk:
        tl.append(r.get("id"))
for r in (json.loads(l) for l in open(os.path.join(R, "mono.jsonl"), encoding="utf-8") if l.strip()):
    if _MG._n(r.get("toi")) in tk:
        tl.append(r.get("id"))
assert not tl, f"teacher test lines in train/dev/tasks/mono: {tl[:5]}"
assert man["teacher_test"]["sha256_16"] == hashlib.sha256(open(tt, "rb").read()).hexdigest()[:16], "manifest built on another teacher test"
random.seed(7); groups = list(w)
picks = Counter(random.choices(groups, list(w.values()), k=1_000_000))
assert all(picks[g] == 0 for g in groups if w[g] == 0), {g: picks[g] for g in groups if w[g] == 0}
k = lambda s: " ".join((s or "").lower().split())
held = {k(l) for f in ("test.en.txt", "test_hopgood.en.txt") for l in open(os.path.join(R, f), encoding="utf-8") if l.strip()}
rows = [json.loads(l) for f in ("train.jsonl", "dev.jsonl") for l in open(os.path.join(R, f), encoding="utf-8") if l.strip()]
leak = [r.get("id") for r in rows if k(r.get("en")) in held]
assert not leak, f"held-out English in train/dev: {leak[:5]}"
docs = [json.loads(l) for f in ("sel.jsonl", "doctest/test_doc_items.jsonl") for l in
        open(os.path.join(R16 if f.startswith("doctest") else R, f), encoding="utf-8")]
dk = {k(x["src"]) for x in docs} | {k(x["ref"]) for x in docs}
tasks = [json.loads(l) for l in open(os.path.join(R, "tasks.jsonl"), encoding="utf-8")]
side = [r.get("id") for r in rows if k(r.get("en")) in dk or k(r.get("toi")) in dk] + \
       [t["id"] for t in tasks if k(t["src"].replace("[BT] ", "")) in dk or k(t["tgt"]) in dk]
assert not side, f"held-out document text in training: {side[:5]}"
print("CHECKS OK: sha = preflight, the three dose mixes checked, new groups drawn (t_teach/t_word/synth24/rain24), "
      "teacher test frozen and "
      f"absent, choma 0, dev/SEL/tests = stage 16's, no test English, no held-out document text ({len(rows)} train/dev "
      f"rows, {len(tasks)} task rows)")
if os.environ.get("F18_DRY") == "1":
    raise SystemExit(0)
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
try:
    have = api.list_repo_files(DS, repo_type="dataset")
    assert not [f for f in have if f.endswith((".gz", ".json", ".txt", ".jsonl"))], f"{DS} already holds data: {have}"
except Exception as e:
    if "already holds" in str(e):
        raise
api.create_repo(DS, repo_type="dataset", private=True, exist_ok=True)
assert api.dataset_info(DS).private, f"{DS} is not private"
stage = tempfile.mkdtemp(prefix="s18_up_")
for f in ("train.jsonl", "tasks.jsonl", "mono.jsonl"):
    with open(os.path.join(R, f), "rb") as a, gzip.open(os.path.join(stage, f + ".gz"), "wb") as b:
        shutil.copyfileobj(a, b)
for f in ("dev.jsonl", "test.en.txt", "test_hopgood.en.txt", "manifest.json", "preflight.json", "sel.jsonl",
          "teacher_dev_final18.jsonl", "confusable_probes_final18.jsonl"):
    shutil.copy(os.path.join(R, f), stage)   # the DEV slice and the probes travel with the data: f18_sel.py reads
                                             # them on the machine. The frozen TEST never leaves the Beelink.
# the frozen teacher test does NOT go up: the machine must never hold it (it is judged on the Beelink)
api.upload_folder(folder_path=stage, repo_id=DS, repo_type="dataset",
                  commit_message="stage 18 data (stage 17 + teacher sentences / words of 21-24 Oct + confusable contrast + rainfall), rebuilt 24 Oct 2026")
back = tempfile.mkdtemp(prefix="s18_back_")
for f in sorted(os.listdir(stage)):
    p = hf_hub_download(DS, f, repo_type="dataset", local_dir=back, force_download=True)
    if f.endswith(".gz"):
        with gzip.open(p, "rb") as a:
            got = hashlib.sha256(a.read()).hexdigest()
        assert got == pre["files"][f[:-3]], f"read-back sha differs: {f}"
    else:
        assert sha(p) == sha(os.path.join(stage, f)), f"read-back sha differs: {f}"
shutil.rmtree(stage); shutil.rmtree(back)
print("DATA UPLOADED AND VERIFIED", DS, "private; train sha", pre["files"]["train.jsonl"][:16])
EOF
  exit $?
fi
[ "$MODE" = run ] || { echo "usage: launch_final18.sh data|run [--dry]"; exit 1; }
PROV=runpod
BUDGET_USD=${BUDGET_USD:-4}
POD="chitonga-final18"
EXP=275
S="${TEMP:-/tmp}/claude/final18"; mkdir -p "$S"
TF=$R/logs/times_final18.txt
LOG=run_final18.log
SCRIPTS="train_nllb.py upload_model.py stage16_sel.py f18_sel.py f18_select.py merge_wise.py rescue_final18.sh run_final18.sh"
echo "=== PREFLIGHT $(date) final round: mix sweep S18a/b/c x $SWEEP_STEPS updates, then the winning mix x seeds 1 2 x $STEPS from $INIT_REPO, WiSE 0.3/0.5/0.7"
for f in $SCRIPTS; do
  [ -s tools/chitonga/train/$f ] || { echo "MISSING tools/chitonga/train/$f"; exit 1; }
done
for f in run_final18.sh rescue_final18.sh; do bash -n tools/chitonga/train/$f || { echo "SYNTAX ERROR in $f"; exit 1; }; done
grep -v "^ *#" tools/chitonga/train/run_final18.sh | grep -q -- "--no-save" && { echo "run_final18.sh must never use --no-save"; exit 1; }
HF_TOKEN="$T" INIT_REPO="$INIT_REPO" python -X utf8 -c "
import os
from huggingface_hub import HfApi; api = HfApi()
for ds in ('$DS', 'PRIVATE_HF_ORG/chitonga-stage16r3'):
    fs = api.list_repo_files(ds, repo_type='dataset'); assert {'train.jsonl.gz', 'manifest.json', 'preflight.json', 'sel.jsonl'} <= set(fs), (ds, fs)
    assert api.dataset_info(ds).private, ds
fs = api.list_repo_files(os.environ['INIT_REPO']); assert any(f.endswith('.safetensors') for f in fs), ('init has no weights', fs)
print('PREFLIGHT OK: both private datasets and the init model on HF')" || { echo "PREFLIGHT FAILED (run: launch_final18.sh data; check INIT_REPO)"; [ $DRY = 1 ] || exit 1; }
if [ $DRY = 1 ]; then
  echo "DRY: plan $EXP min expected; cap = min(\$$BUDGET_USD at the card price, 390 min); watchdog 20 min before; first SSH within 20 min"
  echo "DRY: would create a RunPod 48 GB pod $POD (A40 > RTX A6000 > L40S > RTX 6000 Ada), copy:$SCRIPTS .env, then run run_final18.sh"
  echo "DRY: repos to verify: $NS-sweepS18{a,b,c}-step$SWEEP_STEPS (int8), $NS-B18-s{1,2}-step$STEPS (full + int8), $NS-B18m{03,05,07}-s{1,2} (int8; full for 05)"
  echo "DRY DONE (no machine created)"; exit 0
fi
if python -X utf8 tools/chitonga/train/runpod.py status | grep -q "$POD"; then echo "a RunPod pod named $POD exists: stop"; exit 1; fi
H=""; P=""; STARTED=0; VERIFIED=0; DOWN=0; PRICE=""; START=$(date +%s); CAPT=""
O="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i $HOME/.ssh/zamai_sandbox"
X() { timeout 120 ssh -p $P $O $H "$1" < /dev/null; }
wall() { echo $(( ($(date +%s) - START) / 60 )); }
cost() { python -c "print(round(($(date +%s) - $START) / 3600 * ${PRICE:-0}, 2))"; }
note() { python -X utf8 tools/notify.py "$1" --title "Tonga pod" ${2:-} > /dev/null 2>&1; }
times() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S') WALL_MIN $(wall) PRICE_PER_HR ${PRICE:-?} COST_USD_EST $(cost)" >> $TF; }
mstatus() { python -X utf8 tools/chitonga/train/runpod.py status | grep "$POD"; }
RESULT="no results"
fetch() {
  [ -n "$H" ] || return 0
  local D=$R/logs/final18_$(date +%Y%m%d_%H%M%S) f tg; mkdir -p $D
  for f in $LOG rescue.log pip.log; do timeout 120 scp -P $P $O "$H:/root/$f" $D/$f < /dev/null > /dev/null 2>&1 || true; done
  for tg in swS18a swS18b swS18c F18B18s1 F18B18s2 B18m03s1 B18m05s1 B18m07s1 B18m03s2 B18m05s2 B18m07s2; do
    timeout 120 scp -P $P $O "$H:/root/up_$tg.log" $D/up_$tg.log < /dev/null > /dev/null 2>&1 || true
  done
  timeout 120 scp -P $P $O "$H:/root/merge_*.log" $D/ < /dev/null > /dev/null 2>&1 || true
  for f in sel/select.txt machine_cap.log rescue_cap.log; do
    timeout 120 scp -P $P $O "$H:/root/$f" $D/$(basename $f) < /dev/null > /dev/null 2>&1 || true
  done
  timeout 300 scp -r -P $P $O "$H:/root/sel" $D/sel_json < /dev/null > /dev/null 2>&1 || true
  timeout 300 scp -r -P $P $O "$H:/root/sel" $D/sel < /dev/null > /dev/null 2>&1 || true
  for tg in s18/F18swS18a s18/F18swS18b s18/F18swS18c s18/F18B18s1 s18/F18B18s2; do
    for f in dev_loss.tsv config.json TRAIN_EXIT; do
      timeout 120 scp -P $P $O "$H:/root/$tg/$f" $D/$(basename $tg)_$f < /dev/null > /dev/null 2>&1 || true
    done
  done
  RESULT="$(cat $D/*TRAIN_EXIT 2>/dev/null | tr '\n' ' ')$(grep -h 'UPLOADED' $D/up_*.log 2>/dev/null | grep -c 'RUN .* UPLOADED') of 11 models uploaded; $(grep -h '^WINNER' $D/select.txt 2>/dev/null | tail -1)"
  echo "FETCHED into $D: $(ls $D | tr '\n' ' ')"
}
terminate() {
  echo "TERMINATING $POD: $1"; python -X utf8 tools/chitonga/train/runpod.py down $POD | tail -1; DOWN=1
  [ -n "$CAPT" ] && kill $CAPT 2>/dev/null; times "END ($1)"
  local i; for i in 1 2 3 4 5 6; do [ -z "$(mstatus)" ] && { echo "CONFIRMED GONE: $POD"; note "final round: $POD CONFIRMED GONE ($1). Results: $RESULT. Cost ~\$$(cost) ($(wall) min at \$${PRICE:-?}/h)"; return 0; }; sleep 10; done
  echo "STILL RUNNING AFTER TERMINATE: $POD"; note "$POD is STILL RUNNING after terminate: delete it by hand now" --high
}
rescue() {
  [ -n "$H" ] || return 0
  echo "RESCUE: stopping training, uploading every checkpoint / merge without UPOK"
  X "pkill -f train_nllb.py; pkill -f stage16_sel.py; pkill -f merge_wise.py; pkill -f ct2-transformers-converter; sleep 5; cd /root && nohup bash rescue_final18.sh > rescue.log 2>&1 < /dev/null & sleep 2; echo started" > /dev/null || true
  local t=$(date +%s)
  until X "grep -q 'RESCUE END' /root/rescue.log"; do [ $(( $(date +%s) - t )) -gt 5400 ] && { echo "RESCUE: not finished in 90 min (a 5.5 GB upload at ~1 MB/s with margin)"; break; }; sleep 20; done
  X "grep -E 'RESCUED|RESCUE (FAILED|SKIPPED)' /root/rescue.log" || true
}
fail() { echo "FAILURE: $1"; times "FAIL ($1)"; rescue; fetch; terminate "$1"; note "final round stopped: $1" --high; exit 1; }
on_exit() {   # the EXIT trap: copy the weights off, then destroy (the coordinator's rule, 23 Oct)
  rc=$?; [ $DOWN = 1 ] && return
  [ -z "$H" ] && { [ -n "$(mstatus 2>/dev/null)" ] && terminate "launcher exit before ssh"; return; }
  if [ $STARTED = 1 ] && [ $VERIFIED = 0 ]; then rescue; fi
  fetch
  terminate "launcher exit rc=$rc (weights rescued first if the job had started)"
}
trap on_exit EXIT
UP=""
for g in "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA L40S" "NVIDIA RTX 6000 Ada Generation"; do   # 48 GB only
  UP=$(CHITONGA_NAME=$POD CHITONGA_DISK=120 CHITONGA_CLOUD=COMMUNITY python -X utf8 tools/chitonga/train/runpod.py up "$g" 2>&1 | tail -1)
  echo "$UP" | grep -q '"id"' && break
  echo "$g: not available" | tee -a $TF
done
echo "$UP" | grep -q '"id"' || { echo "NO MACHINE DEPLOYED"; exit 1; }
START=$(date +%s)
PRICE=$(echo "$UP" | grep -oE '"costPerHr": [0-9.]*' | grep -o '[0-9.]*$'); PRICE=${PRICE:-0.6}
CARD=$(echo "$UP" | grep -o '"gpuDisplayName": "[^"]*"' | cut -d'"' -f4)
CAP_MIN=$(python -c "print(min(int($BUDGET_USD * 60 / $PRICE) - 2, 390))")
FAIL_MIN=$(( CAP_MIN - 20 ))
echo "MACHINE runpod $POD $CARD created $(date '+%Y-%m-%d %H:%M:%S') price \$$PRICE/h, expect ~$EXP min (~\$$(python -c "print(round($EXP / 60 * $PRICE, 2))")), watchdog stop at $FAIL_MIN min, hard cap $CAP_MIN min (\$$(python -c "print(round($CAP_MIN / 60 * $PRICE, 2))"))" | tee -a $TF
note "MACHINE CREATED: $POD ($CARD, \$$PRICE/h, cap $CAP_MIN min). Purpose: final translator round of the plateau, mix sweep S18a/b/c (3 x $SWEEP_STEPS), then the winning mix x 2 seeds from a050, then WiSE 0.3/0.5/0.7"
( sleep $((CAP_MIN * 60)); echo "HARD CAP $(date)"; python -X utf8 tools/chitonga/train/runpod.py down $POD ) >> $TF 2>&1 &
CAPT=$!
times "START"
s=""; for i in $(seq 1 120); do s=$(mstatus); echo "$s" | grep -q "ssh root" && break; sleep 10; done   # 20 min to first SSH
echo "$s" | grep -q "ssh root" || { terminate "machine never reachable in 20 min"; exit 1; }
H=$(echo "$s" | grep -o "root@[0-9.]*"); P=$(echo "$s" | grep -o "\-p [0-9]*" | cut -d' ' -f2)
ok=0; for i in 1 2 3 4 5 6 8 10 12; do X "echo ready" > /dev/null && { ok=1; break; }; sleep 10; done
[ $ok = 1 ] || fail "ssh never answered"
printf 'HF_TOKEN=%s\n' "$T" > "$S/pod_f18.env"
FILES=""
put() { timeout 300 scp -P $P $O "$1" "$H:/root/$2" < /dev/null || { rm -f "$S/pod_f18.env"; fail "scp failed: $1"; }
        X "test -s /root/$2" || { rm -f "$S/pod_f18.env"; fail "upload check failed: $2"; }; FILES="$FILES $2"; }
for f in $SCRIPTS; do put tools/chitonga/train/$f $f; done
put "$S/pod_f18.env" .env; rm -f "$S/pod_f18.env"
X "cd /root && for f in $FILES; do test -s \$f || { echo MISSING \$f; exit 1; }; done; echo FILES OK" | grep -q "FILES OK" || fail "upload check failed"
echo "FILES OK on the machine:$FILES"
X "cd /root && sed -i 's/\r$//' *.sh *.py && INIT_REPO=$INIT_REPO STEPS=$STEPS SWEEP_STEPS=$SWEEP_STEPS F18_CAP_MIN=$((CAP_MIN + 10)) nohup bash run_final18.sh > $LOG 2>&1 < /dev/null & sleep 3; echo started" | grep -q started || fail "job did not start"
STARTED=1; echo "JOB STARTED $(date +%H:%M)"; times "JOB STARTED"
t0=$(date +%s)
until X "grep -q 'SETUP DONE' /root/$LOG"; do
  X "grep -qE 'Traceback|Error|SHA MISMATCH|NOT CHECKED|differs' /root/$LOG" && fail "error in setup: $(X "grep -E 'Traceback|Error|MISMATCH|NOT CHECKED|differs' /root/$LOG | tail -2")"
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && fail "no SETUP DONE within 20 min"
  [ $(wall) -ge $FAIL_MIN ] && fail "budget: $(wall) min"
  sleep 30
done
X "grep -q SMOKE /root/$LOG" && fail "the job runs in SMOKE mode"
X "grep -q 'RECIPE init $INIT_REPO lr 2e-5 linear warmup 0.05 ls 0 sweep S18aS18bS18c x $SWEEP_STEPS full $STEPS seeds 12 token-budget 1536 max-rows 64 max-len 256 --tf32 sdpa' /root/$LOG" || fail "recipe line is not the final-round recipe: $(X "grep RECIPE /root/$LOG")"
echo "SETUP DONE $(date +%H:%M)"; times "SETUP DONE"
X "grep -a 'MACHINE CAP' /root/$LOG" | tee -a $TF      # F13: armed or NOT ARMED, in the launcher log
last=""; still=0
until X "grep -qE 'F18_(DONE|FAILED)' /root/$LOG"; do
  sleep 60
  X "grep -q Traceback /root/$LOG" && fail "Traceback: $(X "grep -A3 Traceback /root/$LOG | tail -3")"
  prog=$(X "grep -aE '^(TRAIN|step|THROUGHPUT|CKPT_SAVED|F18SEL|SEL |MERGE|WINNER|RUN )' /root/$LOG | tail -1" | cut -c1-200)
  size=$(X "stat -c %s /root/$LOG 2>/dev/null")
  echo "$(date +%H:%M) [$(wall) min] $prog | log $size"
  now="$prog|$size"; if [ "$now" = "$last" ]; then still=$((still+1)); else still=0; fi; last="$now"
  [ $still -ge 45 ] && fail "no progress for ~45 min"   # F12: the final upload queue is ~14 GB and the job now
                                                       # prints 'RUN uploads pending N' every minute, so silence
                                                       # this long is a hang, not a slow host
  [ $(wall) -ge $FAIL_MIN ] && fail "budget: $(wall) min at \$$PRICE/h (stop at $FAIL_MIN)"
done
X "grep -q F18_FAILED /root/$LOG" && fail "job reported F18_FAILED: $(X "grep -E 'FAILED' /root/$LOG | head -3" | tr '\n' ' ')"
times "JOB DONE"
fetch
REPOS="$NS-B18-s1-step$STEPS $NS-B18-s2-step$STEPS"
REPOS_INT8="$NS-B18m03-s1 $NS-B18m05-s1 $NS-B18m07-s1 $NS-B18m03-s2 $NS-B18m05-s2 $NS-B18m07-s2 $NS-sweepS18a-step$SWEEP_STEPS $NS-sweepS18b-step$SWEEP_STEPS $NS-sweepS18c-step$SWEEP_STEPS"
HF_TOKEN="$T" python -X utf8 -c "
import sys
from huggingface_hub import HfApi
api = HfApi(); bad = []
want = [(r, '.safetensors') for r in '$REPOS'.split()]
want += [(r + '-ct2-int8', 'model.bin') for r in '$REPOS'.split() + '$REPOS_INT8'.split()]
for x, need in want:
    try:
        fs = api.list_repo_files(x); priv = api.model_info(x).private
    except Exception as e:
        bad.append((x, repr(e)[:80])); continue
    if not any(f.endswith(need) for f in fs): bad.append((x, 'no weights'))
    if not priv: bad.append((x, 'NOT PRIVATE'))
print(f'VERIFIED {len(want)} repos' if not bad else 'NOT VERIFIED', bad)
sys.exit(1 if bad else 0)" && VERIFIED=1
[ $VERIFIED = 1 ] || fail "repos not verified on HF"
RESULT="$RESULT; sweep 3 + full 2 + merges 6 verified private on HF"
terminate "final round done and verified"
echo "F18 DONE: download the B18m int8 repos to the Beelink and run final18_killtest.py (PACK.md 6)"
