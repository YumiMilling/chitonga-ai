#!/bin/bash
# The stage 19 kill test's decode pod (24 Oct 2026, the Jesuit; Chris: "check the models on a runpod"; coordinator: A40 or
# L40S, the overwatch first, EXIT trap and hard cap, phone notes at start and at confirmed gone, the Training panel
# registry). A BATCH job: create -> upload the tar -> run_s19kt_pod.sh -> wait for S19KT_DONE -> fetch dec.tgz -> terminate.
# Copied from launch_translator_pod.sh (same guards, same runpod.py, same ssh rules: one file per scp + test -s, ssh only
# through X(), nohup on the pod, a local hard-cap timer that terminates THIS pod by name).
# The overwatch: one pod_overwatch.py already runs for the translator pod (cap 670 min from 15:39); a second one would
# kill every chitonga-* pod once any passed its cap, so this launcher does NOT start one. Our cap (CAP_MIN) is far under
# 670 min, so the running overwatch never fires because of this pod; if it ends earlier (its pods gone for 30 min), this
# launcher's own timer and the EXIT trap remain, and the pod's key-free backstop stops the work at the cap.
#     bash tools/chitonga/train/launch_s19kt_pod.sh <w.tgz>
set -u
cd "$(dirname "$0")/../../.."
TGZ=${1:?w.tgz}
POD=chitonga-s19kt
OUTD=internal/chitonga_train/runs/2026-10-24_stage19_data/killtest/pod; mkdir -p $OUTD
KT=internal/chitonga_train/runs/2026-10-24_stage19_data/killtest
HOSTF="${TEMP:-/tmp}/claude/s19kt/host"
TF=$OUTD/times.txt; LOG=run.log
HFT=$(grep "^HF_TOKEN_CHRIS=" .env.local | cut -d= -f2- | tr -d '\r')
BUDGET_USD=${BUDGET_USD:-4}; MAX_MIN=${MAX_MIN:-300}; MAX_PRICE=${MAX_PRICE:-0.90}
[ -s "$TGZ" ] || { echo "PREFLIGHT: tar missing"; exit 1; }
[ -n "$HFT" ] || { echo "PREFLIGHT: no HF token"; exit 1; }
[ -s tools/chitonga/train/run_s19kt_pod.sh ] || { echo "MISSING run_s19kt_pod.sh"; exit 1; }
if python -X utf8 tools/chitonga/train/runpod.py status | grep -q "$POD"; then echo "a pod named $POD exists: stop"; exit 1; fi
H=""; P=""; DOWN=0; PRICE=""; START=$(date +%s); CAPT=""; PODID=""; CARD=""; RESULT="no result"
O="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i $HOME/.ssh/zamai_sandbox"
X() { timeout 120 ssh -p $P $O $H "$1" < /dev/null; }
wall() { echo $(( ($(date +%s) - START) / 60 )); }
cost() { python -c "print(round(($(date +%s) - $START) / 3600 * ${PRICE:-0}, 2))"; }
note() { python -X utf8 tools/notify.py "$1" --title "Tonga pod" ${2:-} > /dev/null 2>&1; }
times() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S') WALL_MIN $(wall) PRICE_PER_HR ${PRICE:-?} COST_USD_EST $(cost)" | tee -a $TF; }
mstatus() { python -X utf8 tools/chitonga/train/runpod.py status | grep "$POD"; }
fetch() {
  [ -n "$H" ] || return 0
  timeout 120 scp -P $P $O "$H:/root/$LOG" $OUTD/pod_run.log < /dev/null > /dev/null 2>&1 || true
  timeout 120 scp -P $P $O "$H:/root/podall.log" $OUTD/podall.log < /dev/null > /dev/null 2>&1 || true
  timeout 120 scp -P $P $O "$H:/root/guard.log" $OUTD/guard.log < /dev/null > /dev/null 2>&1 || true
}
fetch_dec() {   # the results: packed on the pod (even a partial run), copied, unpacked into killtest/dec
  [ -n "$H" ] || return 1
  X "cd /root/w/zamai-pro/$KT && tar -czf /root/dec_now.tgz dec" > /dev/null 2>&1 || return 1
  timeout 900 scp -P $P $O "$H:/root/dec_now.tgz" $OUTD/dec.tgz < /dev/null > /dev/null 2>&1 || return 1
  [ -s $OUTD/dec.tgz ] && tar -xzf $OUTD/dec.tgz -C $KT && echo "DEC FETCHED $(ls $KT/dec | wc -l) entries" | tee -a $TF
}
deregister() {
  python -X utf8 - "$PODID" <<'EOF' 2>/dev/null || true
import datetime as dt, json, sys
p = "internal/chitonga_train/runs/ACTIVE_RUN.json"
j = json.load(open(p, encoding="utf-8"))
for r in j["runs"]:
    if r.get("id") == "s19kt" and r.get("machine_id") == sys.argv[1]:
        r["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
json.dump(j, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
EOF
}
terminate() {
  echo "TERMINATING $POD: $1"; python -X utf8 tools/chitonga/train/runpod.py down $POD | tail -1; DOWN=1
  [ -n "$CAPT" ] && kill $CAPT 2>/dev/null; times "END ($1)"; deregister
  local i; for i in 1 2 3 4 5 6 9 12; do [ -z "$(mstatus)" ] && { times "CONFIRMED GONE"; note "s19 kill-test pod confirmed gone ($1): $RESULT; cost ~\$$(cost) ($(wall) min at \$${PRICE:-?}/h)"; return 0; }; sleep 10; done
  echo "STILL RUNNING AFTER TERMINATE: $POD"; note "$POD is STILL RUNNING after terminate: delete it by hand now" --high
}
fail() { echo "FAILURE: $1"; times "FAIL ($1)"; fetch; fetch_dec; terminate "$1"; exit 1; }
on_exit() { rc=$?; [ $DOWN = 1 ] && return; fetch; fetch_dec; terminate "launcher exit rc=$rc"; }
trap on_exit EXIT
trap 'exit 1' INT TERM HUP
UP=""; REACHED=0; SPENT=0
for gc in "NVIDIA A40|COMMUNITY" "NVIDIA L40S|COMMUNITY" "NVIDIA A40|SECURE" "NVIDIA L40S|SECURE" "NVIDIA A40|COMMUNITY" "NVIDIA A40|SECURE"; do
  g=${gc%|*}; cl=${gc#*|}
  UP=$(CHITONGA_NAME=$POD CHITONGA_DISK=60 CHITONGA_CLOUD=$cl CHITONGA_CLOUD_ONLY=1 CHITONGA_PORTS="22/tcp" python -X utf8 tools/chitonga/train/runpod.py up "$g" 2>&1 | tail -1)
  if ! echo "$UP" | grep -q '"id"'; then echo "$g ($cl): not available" | tee -a $TF; continue; fi
  PR=$(echo "$UP" | grep -oE '"costPerHr": [0-9.]*' | grep -o '[0-9.]*$')
  if ! python -c "import sys; sys.exit(0 if float('${PR:-9}') <= $MAX_PRICE else 1)"; then
    echo "$g came at \$${PR:-?}/h > \$$MAX_PRICE: terminated at once" | tee -a $TF
    python -X utf8 tools/chitonga/train/runpod.py down $POD | tail -1; sleep 15; continue
  fi
  START=$(date +%s); PRICE=$PR; DOWN=0
  CARD=$(echo "$UP" | grep -o '"gpuDisplayName": "[^"]*"' | cut -d'"' -f4)
  PODID=$(echo "$UP" | grep -o '"id": "[^"]*"' | head -1 | cut -d'"' -f4)
  CAP_MIN=$(python -c "print(min(int(($BUDGET_USD - $SPENT) * 60 / $PRICE) - 6, $MAX_MIN))")
  echo "MACHINE runpod $POD $PODID $CARD created $(date '+%Y-%m-%d %H:%M:%S') price \$$PRICE/h; cap $CAP_MIN min (spent before \$$SPENT)" | tee -a $TF
  note "stage 19 kill-test decodes (a050 + 5 candidates, raw + served), $CARD, \$$PRICE/h, cap \$$BUDGET_USD / $CAP_MIN min"
  times "START"
  python -X utf8 - "$PODID" "${CARD:-?}" "$PRICE" "$CAP_MIN" "$HOSTF" <<'EOF' | tee -a $TF
import datetime as dt, json, re, sys
pid, card, price, cap, hf = sys.argv[1:6]
p = "internal/chitonga_train/runs/ACTIVE_RUN.json"
j = json.load(open(p, encoding="utf-8"))
j["runs"] = [r for r in j["runs"] if r.get("id") != "s19kt"]
hf = re.sub(r"^/([a-zA-Z])/", lambda m: m.group(1).upper() + ":/", hf)
j["runs"].insert(0, {"id": "s19kt", "title": "Stage 19 kill test: decodes (a050 + 5 candidates, raw + served)",
    "provider": "runpod", "machine": "chitonga-s19kt", "machine_id": pid, "card": card, "usd_hr": float(price),
    "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "estimate_min": 120, "cap_min": int(cap),
    "log": "internal/chitonga_train/runs/2026-10-24_stage19_data/killtest/pod/times.txt", "remote_log": "/root/run.log",
    "ssh": {"host_file": hf, "key": "~/.ssh/zamai_sandbox"}, "done_marker": "S19KT_DONE", "fail_marker": "S19KT_FAILED",
    "tail_min": 10, "plan": []})
json.dump(j, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("REGISTERED s19kt first in ACTIVE_RUN.json;", len(j["runs"]), "runs kept")
EOF
  s=""; for i in $(seq 1 120); do                   # 20 min to first ssh
    s=$(mstatus); echo "$s" | grep -q "ssh root" && break
    [ $((i % 5)) = 0 ] && echo "  $(date +%H:%M) $(python -X utf8 tools/chitonga/train/runpod.py info $POD | cut -c1-300)" | tee -a $TF
    sleep 10
  done
  if echo "$s" | grep -q "ssh root"; then
    H=$(echo "$s" | grep -o "root@[0-9.]*"); P=$(echo "$s" | grep -o "\-p [0-9]*" | cut -d' ' -f2)
    ok=0; for i in 1 2 3 4 5 6 8 10 12; do X "echo ready" > /dev/null && { ok=1; break; }; sleep 10; done
    [ $ok = 1 ] && { REACHED=1; mkdir -p "$(dirname "$HOSTF")"; echo "$H $P" > "$HOSTF"; break; }
  fi
  echo "  $(python -X utf8 tools/chitonga/train/runpod.py info $POD | cut -c1-300)" | tee -a $TF
  H=""; terminate "$CARD never reachable"
  SPENT=$(python -c "print(round($SPENT + $(cost), 2))")
done
[ $REACHED = 1 ] || { DOWN=1; echo "NO REACHABLE MACHINE (spent ~\$$SPENT)"; exit 1; }
( sleep $(( (CAP_MIN + 3) * 60 )); echo "HARD CAP $(date)"; python -X utf8 tools/chitonga/train/runpod.py down $POD ) >> $TF 2>&1 &
CAPT=$!
NP=$(X "nproc" | tr -dc 0-9); GB=$(X "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits" | head -1 | tr -dc 0-9)
RAM=$(X "free -g | awk '/Mem:/{print \$2}'" | tr -dc 0-9)
PAR=3; [ "${NP:-0}" -ge 16 ] && [ "${RAM:-0}" -ge 60 ] && PAR=4; [ "${NP:-0}" -lt 8 ] && PAR=2
echo "POD nproc $NP ram ${RAM} GB gpu ${GB} MiB -> par $PAR" | tee -a $TF
S="${TEMP:-/tmp}/s19kt_env_$$"
printf 'HF_TOKEN=%s\nCAP_MIN=%s\nPAR=%s\n' "$HFT" "$((CAP_MIN - 10))" "$PAR" > "$S"
put() { timeout 900 scp -P $P $O "$1" "$H:/root/$2" < /dev/null || { rm -f "$S"; fail "scp failed: $1"; }
        X "test -s /root/$2" || { rm -f "$S"; fail "empty on the pod: $2"; }; }
put "$TGZ" w.tgz
put tools/chitonga/train/run_s19kt_pod.sh run_s19kt_pod.sh
put "$S" .env; rm -f "$S"
times "FILES OK"
X "cd /root && sed -i 's/\r$//' run_s19kt_pod.sh .env && nohup bash run_s19kt_pod.sh > $LOG 2>&1 < /dev/null & sleep 3; echo started" | grep -q started || fail "job did not start"
times "JOB STARTED"
until X "grep -qE '^SETUP DONE|^S19KT_FAILED' /root/$LOG"; do
  sleep 30
  echo "$(date +%H:%M) [$(wall) min, ~\$$(cost)] $(X "tail -1 /root/$LOG" | cut -c1-160)"
  [ $(wall) -ge 45 ] && fail "setup not done after 45 min"
done
X "grep -E 'INT8 SHA|ALL INT8|TOKENIZER|SETUP DONE|S19KT_FAILED|^ct2' /root/$LOG" | tee -a $TF
X "grep -q '^SETUP DONE' /root/$LOG" || fail "setup failed"
times "SETUP DONE"
last=""; same=0
while true; do
  sleep 120
  st=$(X "grep -E '^(S19KT_DONE|S19KT_FAILED)' /root/$LOG | tail -1")
  if [ -n "$st" ]; then echo "$st" | tee -a $TF; break; fi
  prog=$(X "tail -n 3 /root/podall.log; for f in /root/w/zamai-pro/$KT/dec/logs/*.log; do tail -n 1 \$f; done 2>/dev/null" | tr '\n' ' ' | cut -c1-600)
  echo "$(date +%H:%M) [$(wall) min, ~\$$(cost)] $prog" | tee -a $OUTD/progress.log
  if [ "$prog" = "$last" ]; then same=$((same + 1)); else same=0; last="$prog"; fi
  [ $same -ge 10 ] && fail "no progress for 20 min"
  [ -z "$(mstatus)" ] && { times "POD GONE"; DOWN=1; RESULT="pod gone before done"; exit 1; }
  [ $(wall) -ge $CAP_MIN ] && fail "cap $CAP_MIN min"
done
fetch
fetch_dec || fail "results not fetched"
X "grep -q '^S19KT_DONE' /root/$LOG" && RESULT="decodes complete" || RESULT="job failed: $(X "grep S19KT_FAILED /root/$LOG" | cut -c1-120)"
times "RESULTS FETCHED ($RESULT)"
terminate "$RESULT"
exit 0
