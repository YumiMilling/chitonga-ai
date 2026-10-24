#!/bin/bash
# The translator pod for the six streams (23 Oct 2026, the Jesuit; Chris: "If the beelink is the bottleneck, we need to
# get cheap pods to open up the bottleneck"; spend approved in principle: the cheapest card that runs CT2 int8 1.3B,
# community cloud, HARD CAP USD 5 total, max 11 h, then destroy).
# Copied from launch_wmap.sh (same guards): pod_overwatch.py started by the operator FIRST; phone note the moment the pod
# exists ("translator pod for the 6 streams, <card>, <$/h>, cap $5") and on CONFIRMED GONE (requests served, cost); one
# file per scp + test -s; ssh only through X(); nohup on the pod; EXIT trap terminates; a local hard-cap timer; idle
# shutdown after IDLE_MIN without a request (the pod's /health idle_s; the pod also removes itself, run_translator_pod.sh).
# Writes TRANSLATOR_POD_URL (RunPod HTTPS proxy) and TRANSLATOR_POD_TOKEN (random, never printed) to .env.local for
# tools/chitonga/llm/translate_remote.py; the URL line is removed again when the pod goes.
#     bash tools/chitonga/train/launch_translator_pod.sh <w.tgz> <local_sha.txt>
# Log: internal/chitonga_train/runs/2026-10-23_translator_pod/ (times.txt, launch.log, pod_run.log, guard.log).
set -u
cd "$(dirname "$0")/../../.."
TGZ=${1:?w.tgz}; SHA=${2:?local_sha.txt}
POD=chitonga-translator
# 24 Oct (the reader bake-off pod): TPOD_OUTD = this pod's run folder, TPOD_PURPOSE = the phone note's purpose,
# TPOD_RUN_ID = its entry in runs/ACTIVE_RUN.json (the zamai.pro/beelink Training panel), TPOD_HOSTFILE = 'user@ip port'
# for the panel's telemetry (outside the repo; the address never goes into the registry)
OUTD=${TPOD_OUTD:-internal/chitonga_train/runs/2026-10-23_translator_pod}; mkdir -p $OUTD
PURPOSE=${TPOD_PURPOSE:-the 6 streams}; RUNID=${TPOD_RUN_ID:-}; HOSTF=${TPOD_HOSTFILE:-}
TF=$OUTD/times.txt; LOG=run.log
HFT=$(grep "^HF_TOKEN_CHRIS=" .env.local | cut -d= -f2- | tr -d '\r')
BUDGET_USD=${BUDGET_USD:-6}; MAX_MIN=${MAX_MIN:-660}; IDLE_MIN=${IDLE_MIN:-60}; MAX_PRICE=${MAX_PRICE:-0.60}
[ -s "$TGZ" ] && [ -s "$SHA" ] && [ $(grep -c . "$SHA") = 3 ] || { echo "PREFLIGHT: tar or the 3 local sha lines missing"; exit 1; }
[ -n "$HFT" ] || { echo "PREFLIGHT: no HF token"; exit 1; }
[ -s tools/chitonga/train/run_translator_pod.sh ] || { echo "MISSING run_translator_pod.sh"; exit 1; }
if python -X utf8 tools/chitonga/train/runpod.py status | grep -q "$POD"; then echo "a pod named $POD exists: stop"; exit 1; fi
setenv() {  # setenv KEY [VALUE]: set or (no value) remove one line of .env.local; values never echoed
  python -X utf8 - "$1" "${2:-}" <<'EOF'
import sys
k, v = sys.argv[1], sys.argv[2]
p = ".env.local"
ls = [l for l in open(p, encoding="utf-8").read().splitlines() if not l.startswith(k + "=")]
if v:
    ls.append(f"{k}={v}")
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(ls) + "\n")
EOF
}
grep -q "^TRANSLATOR_POD_TOKEN=" .env.local || setenv TRANSLATOR_POD_TOKEN "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
setenv TRANSLATOR_POD_URL
TOK=$(grep "^TRANSLATOR_POD_TOKEN=" .env.local | cut -d= -f2- | tr -d '\r')
H=""; P=""; DOWN=0; PRICE=""; START=$(date +%s); CAPT=""; SERVED="?"; PODID=""; CARD=""
O="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i $HOME/.ssh/zamai_sandbox"
X() { timeout 120 ssh -p $P $O $H "$1" < /dev/null; }
wall() { echo $(( ($(date +%s) - START) / 60 )); }
cost() { python -c "print(round(($(date +%s) - $START) / 3600 * ${PRICE:-0}, 2))"; }
note() { python -X utf8 tools/notify.py "$1" --title "Tonga pod" ${2:-} > /dev/null 2>&1; }
times() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S') WALL_MIN $(wall) PRICE_PER_HR ${PRICE:-?} COST_USD_EST $(cost) SERVED $SERVED" | tee -a $TF; }
mstatus() { python -X utf8 tools/chitonga/train/runpod.py status | grep "$POD"; }
fetch() {
  [ -n "$H" ] || return 0
  timeout 120 scp -P $P $O "$H:/root/$LOG" $OUTD/pod_run.log < /dev/null > /dev/null 2>&1 || true
  timeout 120 scp -P $P $O "$H:/root/guard.log" $OUTD/guard.log < /dev/null > /dev/null 2>&1 || true
}
terminate() {
  setenv TRANSLATOR_POD_URL
  echo "TERMINATING $POD: $1"; python -X utf8 tools/chitonga/train/runpod.py down $POD | tail -1; DOWN=1
  [ -n "$CAPT" ] && kill $CAPT 2>/dev/null; times "END ($1)"
  local i; for i in 1 2 3 4 5 6 9 12; do [ -z "$(mstatus)" ] && { times "CONFIRMED GONE"; note "translator pod confirmed gone ($1): served $SERVED requests, cost ~\$$(cost) ($(wall) min at \$${PRICE:-?}/h)"; return 0; }; sleep 10; done
  echo "STILL RUNNING AFTER TERMINATE: $POD"; note "$POD is STILL RUNNING after terminate: delete it by hand now" --high
}
fail() { echo "FAILURE: $1"; times "FAIL ($1)"; fetch; terminate "$1"; exit 1; }
on_exit() { rc=$?; [ $DOWN = 1 ] && return; fetch; terminate "launcher exit rc=$rc"; }
trap on_exit EXIT
trap 'exit 1' INT TERM HUP
# 23 Oct, first try: a community RTX 3090 ran 14 min and never showed a public ssh port ($0.05). Now each card gets
# ~20 min to become reachable (coordinator: the image pull alone can take 10+ min) (runpod.py info logged every minute); otherwise it is terminated, confirmed gone, and the
# next card is tried; the money of failed tries is taken off the $5 before the cap of the next one.
UP=""; REACHED=0; SPENT=0
# 23 Oct, second try: no community supply for any of six cards; pass 2 takes Secure cloud where its list price is under
# the cap (A5000 0.27, A4000 0.25, RTX 4000 Ada 0.28, A4500 0.25 per hour on 23 Oct); the price check below still rules.
# 23 Oct, Chris: "Use an a40 I suggest" (the card our training runs boot reliably on); cap raised to $6 (coordinator).
for gc in "NVIDIA A40|COMMUNITY" "NVIDIA A40|SECURE" "NVIDIA A40|COMMUNITY" "NVIDIA A40|SECURE"; do
  g=${gc%|*}; cl=${gc#*|}
  UP=$(CHITONGA_NAME=$POD CHITONGA_DISK=40 CHITONGA_CLOUD=$cl CHITONGA_CLOUD_ONLY=1 CHITONGA_PORTS="22/tcp,8000/http" python -X utf8 tools/chitonga/train/runpod.py up "$g" 2>&1 | tail -1)
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
  echo "MACHINE runpod $POD $PODID $CARD created $(date '+%Y-%m-%d %H:%M:%S') price \$$PRICE/h; cap $CAP_MIN min, idle $IDLE_MIN min (spent before \$$SPENT)" | tee -a $TF
  LINE="translator pod for $PURPOSE, $CARD, \$$PRICE/h, cap \$$BUDGET_USD / $CAP_MIN min"
  note "$LINE"
  echo "$LINE ($POD $PODID, created $(date '+%Y-%m-%d %H:%M:%S'), hard cap $CAP_MIN min)" > $OUTD/POD_STARTED.txt
  times "START"
  # the Training panel's registry: this run goes FIRST, every other entry stays as it was (24 Oct)
  [ -n "$RUNID" ] && python -X utf8 tools/chitonga/train/register_run.py "$RUNID" "$PODID" "$CARD" "$PRICE" "$CAP_MIN" \
    "$OUTD/launch.log" "$HOSTF" "$PURPOSE" | tee -a $TF
  s=""; for i in $(seq 1 120); do                   # 20 min to first ssh (TRANSLATOR_POD_PLAN.md)
    s=$(mstatus); echo "$s" | grep -q "ssh root" && break
    [ $((i % 5)) = 0 ] && echo "  $(date +%H:%M) $(python -X utf8 tools/chitonga/train/runpod.py info $POD | cut -c1-300)" | tee -a $TF
    sleep 10
  done
  if echo "$s" | grep -q "ssh root"; then
    H=$(echo "$s" | grep -o "root@[0-9.]*"); P=$(echo "$s" | grep -o "\-p [0-9]*" | cut -d' ' -f2)
    ok=0; for i in 1 2 3 4 5 6 8 10 12; do X "echo ready" > /dev/null && { ok=1; break; }; sleep 10; done
    [ $ok = 1 ] && { REACHED=1; [ -n "$HOSTF" ] && { mkdir -p "$(dirname "$HOSTF")"; echo "$H $P" > "$HOSTF"; }; break; }
  fi
  echo "  $(python -X utf8 tools/chitonga/train/runpod.py info $POD | cut -c1-300)" | tee -a $TF
  H=""; terminate "$CARD never reachable"
  SPENT=$(python -c "print(round($SPENT + $(cost), 2))")
done
[ $REACHED = 1 ] || { DOWN=1; echo "NO REACHABLE MACHINE (spent ~\$$SPENT)"; exit 1; }
( sleep $(( (CAP_MIN + 3) * 60 )); echo "HARD CAP $(date)"; python -X utf8 tools/chitonga/train/runpod.py down $POD ) >> $TF 2>&1 &
CAPT=$!
GB=$(X "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits" | head -1 | tr -dc 0-9)
WK=3; [ "${GB:-0}" -lt 20000 ] && WK=2
S="${TEMP:-/tmp}/tpod_env_$$"
printf 'HF_TOKEN=%s\nTRANSLATOR_TOKEN=%s\nIDLE_MIN=%s\nCAP_MIN=%s\nWORKERS=%s\n' "$HFT" "$TOK" "$IDLE_MIN" "$((CAP_MIN + 2))" "$WK" > "$S"
put() { timeout 900 scp -P $P $O "$1" "$H:/root/$2" < /dev/null || { rm -f "$S"; fail "scp failed: $1"; }
        X "test -s /root/$2" || { rm -f "$S"; fail "empty on the pod: $2"; }; }
put "$TGZ" w.tgz
put "$SHA" local_sha.txt
put tools/chitonga/train/run_translator_pod.sh run_translator_pod.sh
put "$S" .env; rm -f "$S"
echo "FILES OK $(date +%H:%M) gpu ${GB} MiB workers $WK"; times "FILES OK"
X "cd /root && sed -i 's/\r$//' run_translator_pod.sh local_sha.txt .env && nohup bash run_translator_pod.sh > $LOG 2>&1 < /dev/null & sleep 3; echo started" | grep -q started || fail "job did not start"
times "JOB STARTED"
until X "grep -qE '^(SERVER READY|TPOD_FAILED)' /root/$LOG"; do
  sleep 30
  echo "$(date +%H:%M) [$(wall) min, ~\$$(cost)] $(X "tail -1 /root/$LOG" | cut -c1-160)"
  [ $(wall) -ge 40 ] && fail "not ready after 40 min"
done
X "grep -E 'SELFKILL|A050 SHA|INT8 SHA|SETUP DONE|probe|SERVER READY|TPOD_FAILED|^ct2|^versions|^cpu|bge' /root/$LOG" | tee -a $TF
X "grep -q '^SERVER READY' /root/$LOG" || fail "setup failed"
# 23 Oct (plan, after the A40 that served nothing): the URL is written only after ONE real request through the RunPod
# proxy returns a translation; the GPU-vs-Beelink comparison is then logged, not gated (GPU int8 is never byte-identical
# to CPU int8: 56/103 identical on 22 Oct; every pod line carries served_by "pod").
URL="https://$PODID-8000.proxy.runpod.net"
SMOKE=""
for t in 1 2 3 4 5; do                             # the RunPod proxy can lag the server by a minute
  SMOKE=$(cd tools/chitonga/llm && TRANSLATOR_POD_URL="$URL" timeout 300 python -X utf8 -c "
import sys; sys.path.insert(0, '.')   # python -c on this machine does not put the cwd on sys.path (23 Oct, try 4)
import translate_remote as R
try:
    r = R.translate('Wash your hands with soap before eating.', 'en>toi', use_memory=False, allow_local=False)
    print('SMOKE', r.get('served_by'), (r.get('text') or '')[:80])
except Exception as e:
    print('SMOKE-ERR', type(e).__name__, str(e)[:150])" 2>&1 | grep '^SMOKE' | tail -1)
  echo "  smoke try $t: $SMOKE" | tee -a $TF
  echo "$SMOKE" | grep -q "^SMOKE pod [A-Za-z]" && break
  sleep 25
done
echo "$SMOKE" | tee -a $TF
echo "$SMOKE" | grep -q "^SMOKE pod [A-Za-z]" || fail "smoke request through the proxy failed"
setenv TRANSLATOR_POD_URL "$URL"
times "SERVING"
# the identity check compares with a reference made by THE SAME CODE on the Beelink (check_local.json in OUTD); without
# one it is run by hand (24 Oct: the served config changed since the 23 Oct reference)
[ -s $OUTD/check_local.json ] && ( cd tools/chitonga/llm && TPOD_CHECK_DIR=$OUTD timeout 1800 python -X utf8 translator_pod_check.py remote 4 > /dev/null 2>&1 && TPOD_CHECK_DIR=$OUTD python -X utf8 translator_pod_check.py compare 2>&1 | tail -3 ) >> $TF 2>&1 &
miss=0; last_served=0; last_progress=$(date +%s)
while true; do
  sleep 120
  st=$(python -X utf8 tools/chitonga/llm/translate_remote.py --status 2>/dev/null | tail -1)
  if echo "$st" | grep -q '"served"'; then
    miss=0
    SERVED=$(echo "$st" | grep -oE '"served": [0-9]+' | grep -oE '[0-9]+$')
    IDLE=$(echo "$st" | grep -oE '"idle_s": [0-9]+' | grep -oE '[0-9]+$')
    PEND=$(echo "$st" | grep -oE '"pending": [0-9]+' | grep -oE '[0-9]+$')
    echo "$(date +%H:%M) [$(wall) min, ~\$$(cost)] served $SERVED pending $PEND idle $((IDLE / 60)) min" | tee -a $OUTD/serving.log
    # requests waiting but nothing served for 15 min = stuck (the 23 Oct A40): kill, do not idle on the clock
    if [ "${SERVED:-0}" -gt "$last_served" ]; then last_served=$SERVED; last_progress=$(date +%s); fi
    [ "${PEND:-0}" -gt 0 ] && [ $(( $(date +%s) - last_progress )) -ge 900 ] && fail "requests pending, nothing served for 15 min"
    [ "${PEND:-0}" = 0 ] && last_progress=$(date +%s)
    [ "${PEND:-0}" = 0 ] && [ $((IDLE / 60)) -ge $IDLE_MIN ] && { echo "SERVING DONE $(date '+%Y-%m-%d %H:%M:%S') served $SERVED, idle $IDLE_MIN min"; fetch; terminate "idle $IDLE_MIN min"; exit 0; }
  else
    miss=$((miss + 1)); echo "$(date +%H:%M) health: no answer ($miss)" >> $OUTD/serving.log
    [ -z "$(mstatus)" ] && { times "POD GONE"; terminate "pod gone (terminated outside the launcher, or by the pod guard)"; exit 0; }
    [ $miss -ge 3 ] && fail "no health answer for 6 min"
  fi
  [ $(wall) -ge $CAP_MIN ] && { fetch; terminate "cap $CAP_MIN min (\$$BUDGET_USD at \$$PRICE/h)"; exit 0; }
  # 24 Oct: the operator ends a finished job cleanly with a STOP file in the run folder (both users done)
  [ -f $OUTD/STOP ] && { echo "SERVING DONE $(date '+%Y-%m-%d %H:%M:%S') served $SERVED, stop file"; fetch; terminate "stop file: $(head -c 120 $OUTD/STOP)"; exit 0; }
done
