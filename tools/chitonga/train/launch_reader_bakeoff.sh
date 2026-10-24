#!/bin/bash
# The reader bake-off, step A, on ONE RunPod GPU (24 Oct 2026, the Jesuit; Chris: "Can we test different base models
# already? ... How do we test and decide?"; then "Maybe use runpod for it"). Plan and rule:
# internal/chitonga_train/runs/2026-10-24_reader_bakeoff/PLAN.md, RULE.md (sha-frozen). PREPARED, NOT LAUNCHED: the go
# is Chris's (DECISIONS: no pod without his go). The pod answers the frozen request bodies with every pinned GGUF
# (pod_reader_bakeoff.py); no harness, no translator, no model weights of ours on the pod; data deleted with the pod.
# Go:
#     python -X utf8 tools/chitonga/llm/reader_bakeoff.py pack                    # bodies + models.json + pod scripts
#     python -X utf8 tools/chitonga/train/pod_overwatch.py 340 7 &                # ONLY if no overwatch is running (it
#                                                                                 # kills EVERY chitonga-* machine at its cap)
#     bash tools/chitonga/train/launch_reader_bakeoff.sh [--dry]
# Lessons carried over (launch_translator_pod.sh, launch_final18.sh, launch_stage19.sh):
#   - cards: A40 community > A40 secure > L40S community > L40S secure > RTX A6000 community, price <= MAX_PRICE;
#     each card gets 20 min to become reachable over SSH (the image pull alone can take 10+ min), else it is
#     terminated, confirmed gone, its cost taken off the budget, and the next card tried;
#   - phone note (tools/notify.py) the moment a machine exists (purpose, card, price, cap) and on CONFIRMED GONE
#     (results, cost); the run is put FIRST in runs/ACTIVE_RUN.json so the zamai.pro/beelink Training panel shows it;
#   - one file per scp + test -s on the pod; ssh only through X(); nohup on the pod;
#   - a SMOKE reply from the first arm must arrive before any result counts;
#   - 15 min without progress (pod log size + last progress line unchanged) = kill;
#   - hard cap = min(BUDGET_USD at the card's price, MAX_MIN): a local timer destroys the machine at the cap; the pod
#     has its own self-remove / key-free backstop at the cap + 10;
#   - EXIT trap: fetch results, wipe the pod's data, terminate, confirm gone (phone Chris if not);
#   - results fetched after every finished arm, so a dead pod loses at most one arm.
set -u
cd "$(dirname "$0")/../../.."
R=${RB_DIR:-internal/chitonga_train/runs/2026-10-24_reader_bakeoff}   # RB_DIR: the dry test points it at smoke/
MODELS=${RB_MODELS:-$R/models.json}
PACKD=$R/pack
POD=chitonga-rbake
DRY=$([ "${1:-}" = "--dry" ] && echo 1 || echo 0)
BUDGET_USD=${BUDGET_USD:-5}; MAX_MIN=${MAX_MIN:-330}; MAX_PRICE=${MAX_PRICE:-0.90}; SOFT_MIN=${SOFT_MIN:-240}
EXP=${EXP:-220}
TF=$R/logs/times.txt; LOG=run.log
mkdir -p $R/logs $R/replies
LL=$R/logs/launch_rb.log
exec > >(tee -a $LL) 2>&1
HFT=$(grep "^HF_TOKEN_CHRIS=" .env.local | cut -d= -f2- | tr -d '\r')
S="${TEMP:-/tmp}/claude/rb"; mkdir -p "$S"
echo "=== PREFLIGHT $(date '+%Y-%m-%d %H:%M:%S') reader bake-off step A"
for f in tools/chitonga/train/run_reader_bakeoff.sh tools/chitonga/train/pod_reader_bakeoff.py $PACKD/rb_pack.tgz $PACKD/pack_sha.txt \
         internal/chitonga_train/runs/2026-10-24_reader_bakeoff/RULE.md internal/chitonga_train/runs/2026-10-24_reader_bakeoff/RULE.md.sha256 \
         $R/capture_manifest.json $R/freeze_manifest.json $MODELS; do
  [ -s "$f" ] || { echo "PREFLIGHT: missing $f"; exit 1; }
done
bash -n tools/chitonga/train/run_reader_bakeoff.sh || { echo "SYNTAX ERROR in run_reader_bakeoff.sh"; exit 1; }
python -X utf8 - "$R" "$MODELS" <<'EOF' || { echo "PREFLIGHT FAILED"; exit 1; }
"""the frozen things are still frozen: rule sha, bodies = capture, pack = these files, test set = its sha"""
import hashlib, json, os, sys, tarfile
R, MODELS = sys.argv[1], sys.argv[2]
RR = "internal/chitonga_train/runs/2026-10-24_reader_bakeoff"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert open(os.path.join(RR, "RULE.md.sha256")).read().split()[0] == sha(os.path.join(RR, "RULE.md")), "RULE.md changed"
cm = json.load(open(os.path.join(R, "capture_manifest.json"), encoding="utf-8"))
assert sha(os.path.join(R, "bodies.jsonl")) == cm["bodies_sha256"], "bodies.jsonl changed since capture"
ts = json.load(open(os.path.join(R, "freeze_manifest.json"), encoding="utf-8"))["testset"]
assert open(ts + ".sha256").read().split()[0] == sha(ts) == cm["testset_sha256"], "test set changed"
want = {l.split()[1]: l.split()[0] for l in open(os.path.join(R, "pack", "pack_sha.txt")) if l.strip()}
with tarfile.open(os.path.join(R, "pack", "rb_pack.tgz")) as t:
    for m in t.getmembers():
        if m.isfile():
            n = os.path.basename(m.name)
            assert hashlib.sha256(t.extractfile(m).read()).hexdigest() == want[n], f"pack file {n} != pack_sha.txt"
assert want["bodies.jsonl"] == cm["bodies_sha256"], "the pack holds other bodies than the capture"
fm = json.load(open(os.path.join(R, "freeze_manifest.json"), encoding="utf-8"))
assert not any(fm.get("smoke_flags", {}).values()), f"the freeze ran with smoke flags {fm['smoke_flags']}: re-run freeze for real"
assert fm["plugin"] == cm["plugin"] and fm["testset_sha256"] == cm["testset_sha256"], "freeze and capture disagree"
ms = json.load(open(MODELS, encoding="utf-8"))["models"]
assert want["models.json"] == sha(MODELS), "the pack holds another models.json"
assert ms[0]["tag"] == "G_q4" and all(len(m["sha256"]) == 64 for m in ms), "models.json: incumbent first, sha256 pinned"
print(f"PREFLIGHT OK: rule sha, {cm['reader_bodies']} reader + {cm['digest_bodies']} digest bodies, plug-in {cm['plugin']}, "
      f"{len(ms)} arms pinned")
EOF
BAL=$(python -X utf8 tools/chitonga/train/runpod.py me | grep -oE 'balance \$ [0-9.]+' | grep -oE '[0-9.]+$')
python -c "import sys; sys.exit(0 if float('${BAL:-0}') >= $BUDGET_USD + 1 else 1)" || { echo "PREFLIGHT: RunPod balance \$${BAL:-?} < budget \$$BUDGET_USD + 1"; exit 1; }
echo "RunPod balance \$$BAL"
if [ $DRY = 1 ]; then
  echo "DRY: would create $POD (A40 > L40S > RTX A6000, <= \$$MAX_PRICE/h), cap min(\$$BUDGET_USD at the price, $MAX_MIN min), expect ~$EXP min"
  echo "DRY: copy rb_pack.tgz pack_sha.txt run_reader_bakeoff.sh .env (HF token only), run, fetch replies per arm into $R/replies/"
  echo "DRY DONE (no machine created)"; exit 0
fi
if python -X utf8 tools/chitonga/train/runpod.py status | grep -q "$POD"; then echo "a pod named $POD exists: stop"; exit 1; fi
H=""; P=""; DOWN=1; PRICE=""; START=$(date +%s); CAPT=""; CARD=""; PODID=""; RESULT="no arm finished"
O="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i $HOME/.ssh/zamai_sandbox"
X() { timeout 120 ssh -p $P $O $H "$1" < /dev/null; }
wall() { echo $(( ($(date +%s) - START) / 60 )); }
cost() { python -c "print(round(($(date +%s) - $START) / 3600 * ${PRICE:-0}, 2))"; }
note() { python -X utf8 tools/notify.py "$1" --title "Tonga pod" ${2:-} > /dev/null 2>&1; }
times() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S') WALL_MIN $(wall) PRICE_PER_HR ${PRICE:-?} COST_USD_EST $(cost)" >> $TF; }
mstatus() { python -X utf8 tools/chitonga/train/runpod.py status | grep "$POD"; }
fetched=""
fetch() {   # fetch [tag]: one arm's replies, or (no tag) everything; prints UPLOADED for the panel per new arm
  [ -n "$H" ] || return 0
  local t f
  timeout 120 scp -P $P $O "$H:/root/$LOG" $R/logs/pod_run.log < /dev/null > /dev/null 2>&1 || true
  timeout 120 scp -P $P $O "$H:/root/out/manifest.json" $R/logs/pod_manifest.json < /dev/null > /dev/null 2>&1 || true
  for f in machine_cap.log backstop.log pip.log; do timeout 60 scp -P $P $O "$H:/root/$f" $R/logs/$f < /dev/null > /dev/null 2>&1 || true; done
  for t in ${1:-$(X "ls /root/out/ 2>/dev/null | grep -E '^[A-Z][0-9A-Za-z]*_q[48]\.jsonl$' | sed 's/\.jsonl$//'")}; do
    timeout 300 scp -P $P $O "$H:/root/out/$t.jsonl" $R/replies/$t.jsonl.part < /dev/null > /dev/null 2>&1 && mv $R/replies/$t.jsonl.part $R/replies/$t.jsonl || continue
    timeout 60 scp -P $P $O "$H:/root/out/llama_$t.log" $R/logs/llama_$t.log < /dev/null > /dev/null 2>&1 || true
    case " $fetched " in *" $t "*) ;; *) fetched="$fetched $t"; echo "UPLOADED $t (replies fetched to the Beelink: $(grep -c . $R/replies/$t.jsonl) lines)";; esac
  done
  RESULT="$(echo $fetched | wc -w) arms fetched ($fetched ); $(grep -c '^RUN FAILED' $R/logs/pod_run.log 2>/dev/null || echo 0) failed"
}
wipe() { [ -n "$H" ] && X "rm -rf /root/pack /root/out /root/m /root/rb_pack.tgz /root/.env; echo wiped" 2>/dev/null | grep -q wiped && echo "POD DATA WIPED"; return 0; }
terminate() {
  echo "TERMINATING $POD: $1"; python -X utf8 tools/chitonga/train/runpod.py down $POD | tail -1; DOWN=1
  [ -n "$CAPT" ] && kill $CAPT 2>/dev/null; times "END ($1)"
  local i; for i in 1 2 3 4 5 6 9 12; do
    [ -z "$(mstatus)" ] && { echo "CONFIRMED GONE: $POD"; times "CONFIRMED GONE"
      note "reader bake-off: $POD CONFIRMED GONE ($1). Results: $RESULT. Cost ~\$$(cost) ($(wall) min at \$${PRICE:-?}/h)"; return 0; }
    sleep 10; done
  echo "STILL RUNNING AFTER TERMINATE: $POD"; note "$POD is STILL RUNNING after terminate: delete it by hand now" --high
}
fail() { echo "FAILURE: $1"; times "FAIL ($1)"; fetch; wipe; terminate "$1"; note "reader bake-off stopped: $1" --high; exit 1; }
on_exit() { rc=$?; [ $DOWN = 1 ] && return; fetch; wipe; terminate "launcher exit rc=$rc"; }
trap on_exit EXIT
trap 'exit 1' INT TERM HUP
register() {   # put this run FIRST in runs/ACTIVE_RUN.json (the Training panel's registry)
  python -X utf8 - "$PODID" "$CARD" "$PRICE" "$CAP_MIN" "$EXP" "$S/host" "$MODELS" <<'EOF'
import datetime as dt, json, sys
pid, card, price, cap, exp, hostf = sys.argv[1:7]
R = "internal/chitonga_train/runs/2026-10-24_reader_bakeoff"
reg_p = "internal/chitonga_train/runs/ACTIVE_RUN.json"
reg = json.load(open(reg_p, encoding="utf-8"))
ms = json.load(open(f"{R}/models.json" if len(sys.argv) < 8 else sys.argv[7], encoding="utf-8"))["models"]
cm = json.load(open(f"{R}/capture_manifest.json", encoding="utf-8"))
entry = {"id": "reader_bakeoff", "title": "Reader bake-off: standard models answer the frozen harness requests",
         "provider": "runpod", "machine": "chitonga-rbake", "machine_id": pid, "card": card, "usd_hr": float(price),
         "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "estimate_min": int(exp), "cap_min": int(cap),
         "log": f"{R}/logs/launch_rb.log", "ssh": {"host_file": hostf.replace("\\", "/"), "key": "~/.ssh/zamai_sandbox"},
         "remote_log": "/root/run.log", "done_marker": "F20_DONE", "fail_marker": "F20_FAILED", "tail_min": 5,
         "repos": len(ms), "spu": {"Q4_K_M": 1.3, "Q8_0": 1.8},
         "plan": [{"tag": m["tag"], "label": f"{m['name']} {m['quant']}", "arm": m["quant"], "steps": cm["reader_bodies"] + cm.get("reader_aux_bodies", 0)} for m in ms],
         "note": "Not training: 'steps' = request bodies answered, 'UPLOADED' = an arm's replies fetched to the Beelink."}
reg["runs"] = [entry] + [r for r in reg["runs"] if r.get("id") != "reader_bakeoff"]
json.dump(reg, open(reg_p, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print("REGISTERED in ACTIVE_RUN.json")
EOF
}
REACHED=0; SPENT=0
for gc in "NVIDIA A40|COMMUNITY" "NVIDIA A40|SECURE" "NVIDIA L40S|COMMUNITY" "NVIDIA L40S|SECURE" "NVIDIA RTX A6000|COMMUNITY"; do
  g=${gc%|*}; cl=${gc#*|}
  UP=$(CHITONGA_NAME=$POD CHITONGA_DISK=80 CHITONGA_CLOUD=$cl CHITONGA_CLOUD_ONLY=1 python -X utf8 tools/chitonga/train/runpod.py up "$g" 2>&1 | tail -1)
  if ! echo "$UP" | grep -q '"id"'; then echo "$g ($cl): not available" | tee -a $TF; continue; fi
  DOWN=0
  PR=$(echo "$UP" | grep -oE '"costPerHr": [0-9.]*' | grep -o '[0-9.]*$')
  if ! python -c "import sys; sys.exit(0 if float('${PR:-9}') <= $MAX_PRICE else 1)"; then
    echo "$g came at \$${PR:-?}/h > \$$MAX_PRICE: terminated at once" | tee -a $TF
    python -X utf8 tools/chitonga/train/runpod.py down $POD | tail -1; DOWN=1; sleep 15; continue
  fi
  START=$(date +%s); PRICE=$PR
  CARD=$(echo "$UP" | grep -o '"gpuDisplayName": "[^"]*"' | cut -d'"' -f4)
  PODID=$(echo "$UP" | grep -o '"id": "[^"]*"' | head -1 | cut -d'"' -f4)
  CAP_MIN=$(python -c "print(min(int(($BUDGET_USD - $SPENT) * 60 / $PRICE) - 4, $MAX_MIN))")
  echo "MACHINE runpod $POD $CARD created $(date '+%Y-%m-%d %H:%M:%S') price \$$PRICE/h, expect ~$EXP min (~\$$(python -c "print(round($EXP / 60 * $PRICE, 2))")), hard cap $CAP_MIN min (spent before \$$SPENT)"
  note "MACHINE CREATED: $POD ($CARD, \$$PRICE/h, cap $CAP_MIN min). Purpose: reader bake-off step A, $(python -c "import json; print(len(json.load(open('$MODELS'))['models']))") GGUF arms answer the frozen harness requests (no training)"
  register
  ( sleep $(( (CAP_MIN + 2) * 60 )); echo "HARD CAP $(date)"; python -X utf8 tools/chitonga/train/runpod.py down $POD ) >> $TF 2>&1 &
  CAPT=$!
  times "START"
  s=""; for i in $(seq 1 120); do                       # 20 min to the first SSH
    s=$(mstatus); echo "$s" | grep -q "ssh root" && break
    [ $((i % 6)) = 0 ] && echo "  $(date +%H:%M) waiting for ssh: $(python -X utf8 tools/chitonga/train/runpod.py info $POD | cut -c1-200)"
    sleep 10
  done
  if echo "$s" | grep -q "ssh root"; then
    H=$(echo "$s" | grep -o "root@[0-9.]*"); P=$(echo "$s" | grep -o "\-p [0-9]*" | cut -d' ' -f2)
    ok=0; for i in 1 2 3 4 5 6 8 10 12; do X "echo ready" > /dev/null && { ok=1; break; }; sleep 10; done
    [ $ok = 1 ] && { REACHED=1; echo "$H $P" > "$S/host"; break; }
  fi
  H=""; terminate "$CARD never reachable in 20 min"
  SPENT=$(python -c "print(round($SPENT + $(cost), 2))")
done
[ $REACHED = 1 ] || { DOWN=1; echo "NO REACHABLE MACHINE (spent ~\$$SPENT)"; exit 1; }
printf 'HF_TOKEN=%s\nCAP_MIN=%s\nSOFT_MIN=%s\n' "$HFT" "$((CAP_MIN + 10))" "$SOFT_MIN" > "$S/pod.env"
put() { timeout 300 scp -P $P $O "$1" "$H:/root/$2" < /dev/null || { rm -f "$S/pod.env"; fail "scp failed: $1"; }
        X "test -s /root/$2" || { rm -f "$S/pod.env"; fail "empty on the pod: $2"; }; }
put $PACKD/rb_pack.tgz rb_pack.tgz
put $PACKD/pack_sha.txt pack_sha.txt
put tools/chitonga/train/run_reader_bakeoff.sh run_reader_bakeoff.sh
put "$S/pod.env" .env; rm -f "$S/pod.env"
echo "FILES OK $(date +%H:%M)"; times "FILES OK"
X "cd /root && sed -i 's/\r$//' run_reader_bakeoff.sh pack_sha.txt .env && nohup bash run_reader_bakeoff.sh > $LOG 2>&1 < /dev/null & sleep 3; echo started" | grep -q started || fail "job did not start"
echo "JOB STARTED $(date +%H:%M)"; times "JOB STARTED"
t0=$(date +%s)
until X "grep -qE '^(SETUP DONE|F20_FAILED)' /root/$LOG"; do
  [ $(( $(date +%s) - t0 )) -gt 2400 ] && fail "no SETUP DONE within 40 min (a source build of llama-server may run): $(X "tail -2 /root/$LOG")"
  sleep 20
done
X "grep -E '^(MACHINE CAP|PACK OK|BUILD|release binary|SETUP DONE|F20_FAILED)' /root/$LOG" | tee -a $TF
X "grep -q '^F20_FAILED' /root/$LOG" && fail "setup: $(X "grep '^F20_FAILED' /root/$LOG")"
echo "SETUP DONE $(date +%H:%M)"; times "SETUP DONE"
t0=$(date +%s)
until X "grep -qE '^(SMOKE OK|F20_)' /root/$LOG"; do   # the first arm's smoke reply before any result counts
  [ $(( $(date +%s) - t0 )) -gt 900 ] && fail "no SMOKE OK within 15 min of setup: $(X "tail -2 /root/$LOG")"
  sleep 20
done
X "grep -q '^SMOKE OK' /root/$LOG" || fail "first arm failed before its smoke: $(X "grep -E 'RUN FAILED|F20_' /root/$LOG | head -2")"
echo "SMOKE OK $(X "grep -m1 '^SMOKE OK' /root/$LOG" | cut -c1-120)"; times "SMOKE OK"
last=""; still=0; seen_done=0
until X "grep -qE '^F20_(DONE|FAILED)' /root/$LOG"; do
  sleep 60
  prog=$(X "grep -aE '^(TRAIN |THROUGHPUT|SMOKE OK|SKIPPED|DOWNLOADED|RUN FAILED|SPEED|SIZING)|TRAINING DONE' /root/$LOG | tail -1" | cut -c1-160)
  size=$(X "stat -c %s /root/$LOG 2>/dev/null")
  echo "$(date +%H:%M) [$(wall)/$CAP_MIN min, ~\$$(cost)] $prog"
  nd=$(X "grep -c 'TRAINING DONE' /root/$LOG"); if [ "${nd:-0}" -gt "$seen_done" ]; then seen_done=$nd; fetch; fi
  now="$prog|$size"; if [ "$now" = "$last" ]; then still=$((still + 1)); else still=0; fi; last="$now"
  [ $still -ge 15 ] && fail "no progress for 15 min"
  [ $(wall) -ge $CAP_MIN ] && fail "cap $CAP_MIN min (\$$BUDGET_USD at \$$PRICE/h)"
  [ -z "$(mstatus)" ] && { times "POD GONE"; DOWN=1; H=""; echo "FAILURE: pod gone (removed outside the launcher, or by its backstop)"; terminate "pod gone"; exit 1; }
done
fetch
X "grep -E '^(F20_|RUN FAILED|SKIPPED)' /root/$LOG" | tee -a $TF
X "grep -q '^F20_FAILED' /root/$LOG" && fail "$(X "grep '^F20_FAILED' /root/$LOG")"
python -X utf8 - "$R" <<'EOF' || fail "fetched replies incomplete"
import json, os, sys
R = sys.argv[1]
n = sum(1 for l in open(os.path.join(R, "bodies.jsonl"), encoding="utf-8") if l.strip())
man = json.load(open(os.path.join(R, "logs", "pod_manifest.json")))
bad = []
for t, r in man["arms"].items():
    if "seconds" in r:
        p = os.path.join(R, "replies", t + ".jsonl")
        k = sum(1 for l in open(p, encoding="utf-8")) if os.path.exists(p) else 0
        if k != n:
            bad.append((t, k, n))
print("REPLIES COMPLETE" if not bad else f"INCOMPLETE {bad}")
sys.exit(1 if bad else 0)
EOF
grep -lE 'hf_[A-Za-z0-9]{20,}' $R/logs/* 2>/dev/null && { echo "TOKEN IN A FETCHED LOG: redact before anything else"; note "token pattern in a fetched reader bake-off log" --high; }
echo "F20_DONE reader bake-off step A: $RESULT"
wipe
terminate "step A done, replies fetched"
echo "NEXT on the Beelink: reader_bakeoff.py replay, adjudicate (+ the blind judge), the hardware note (speed --tag G_q4 --gguf <file> --n 20, agree), score"
