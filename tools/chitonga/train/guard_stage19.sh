#!/bin/bash
# MACHINE-SIDE guard for the stage 19 translator round (both arms) (24 Oct 2026, the Jesuit; stage 18 pre-launch review F13: no guard may
# depend on the Beelink staying alive: it stood at 97.6% RAM that night). Runs ON the DO droplet / RunPod pod under nohup,
# started by the launcher right after the job. It does three things the Beelink launcher also does, independently:
#   1 STOP file 15 min before its cap (informational for this job: train_nllb saves only at its last step),
#   2 at the cap: ends the training processes (train_nllb.py, merge_wise.py, f19_sel.py), NEVER an upload (F12), waits for any upload
#     in flight, runs rescue_stage19.sh (uploads every adapter the log has not verified), and then SELF-DESTROYS,
#   3 when the job is finished (F19_DONE / F19_FAILED) or dead, and the Beelink's HEARTBEAT file is stale
#     (no launcher touch for HB_STALE s), it rescues (if needed) and self-destroys: an orphan machine never bills on.
# Its cap is set 10 min AFTER the launcher's hard cap, so the launcher's cleaner path (fetch logs, verify, destroy) goes
# first whenever the Beelink is alive. Self-destroy: DO API DELETE on the droplet's own id (metadata service) with
# DO_API_TOKEN; RunPod podTerminate with RUNPOD_API_KEY on RUNPOD_POD_ID (both keys arrive in .env for this purpose).
#     nohup bash guard_stage19.sh <CAP_MIN> > guard.out 2>&1 &         log: guard.log (fetched by the launcher)
W=${W:-/root/s19}; cd "$W" || exit 1
set -a; . ./.env; set +a
CAP=${1:?cap minutes}
LOG=run_stage19.log; GL=guard.log
HB_STALE=${HB_STALE:-1500}          # 25 min without a heartbeat = the Beelink launcher is gone
UPLOAD_WAIT=${UPLOAD_WAIT:-45}      # minutes an in-flight upload may take before the guard stops waiting for it
START=$(date +%s); STOPPED=0
log() { echo "$(date '+%F %T') [$(( ($(date +%s) - START) / 60 )) min] $*" >> $GL; }
upload_in_flight() { pgrep -f "upload_mode[l].py|rescue_stage1[9].sh" > /dev/null; }
job_alive() { pgrep -f "run_stage1[9].sh|train_nll[b].py|merge_wis[e].py|f19_se[l].py|upload_mode[l].py|ct2-transformers-converte[r]" > /dev/null; }
done_line() { grep -qE '^F19_(DONE|FAILED)' $LOG 2> /dev/null; }
hb_age() { local t; t=$(cat HEARTBEAT 2> /dev/null || echo 0); echo $(( $(date +%s) - ${t:-0} )); }
wait_uploads() { local i; for i in $(seq 1 $UPLOAD_WAIT); do upload_in_flight || return 0; sleep 60; done; log "an upload still runs after $UPLOAD_WAIT min: not waiting longer"; }
rescue() { log "rescue: uploading every adapter the log has not verified"; bash rescue_stage19.sh >> rescue.log 2>&1; log "rescue done: $(grep -cE 'RESCUE VERIFIED' rescue.log 2>/dev/null) verified, $(grep -cE 'RESCUE FAILED' rescue.log 2>/dev/null) failed"; }
self_destroy() {
  log "SELF-DESTROY: $1"
  if [ "${PROV:-do}" = do ]; then
    ID=$(curl -s --max-time 10 http://169.254.169.254/metadata/v1/id)
    [ -n "$ID" ] || { log "no droplet id from the metadata service"; return 1; }
    curl -s --max-time 30 -o /dev/null -w "DELETE droplets/$ID -> HTTP %{http_code}\n" -X DELETE -H "Authorization: Bearer $DO_API_TOKEN" "https://api.digitalocean.com/v2/droplets/$ID" >> $GL 2>&1
  else
    # RUNPOD_POD_ID is not guaranteed in a non-interactive ssh environment (review F8b): PID 1's environment has it
    PID=${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep '^RUNPOD_POD_ID=' | cut -d= -f2)}
    [ -n "$PID" ] || { log "no RunPod pod id (env or /proc/1/environ)"; return 1; }
    curl -s --max-time 30 -X POST https://api.runpod.io/graphql -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$PID\\\"}) }\"}" >> $GL 2>&1
  fi
}
log "guard up: cap $CAP min, heartbeat stale after $HB_STALE s, provider ${PROV:-do}"
while true; do
  sleep "${GUARD_TICK:-60}"          # GUARD_TICK: the unit test runs the loop at 2 s
  EL=$(( ($(date +%s) - START) / 60 ))
  if done_line; then
    if [ "$(hb_age)" -gt $HB_STALE ]; then
      grep -q "F19_FAILED|NOT UPLOADED" $LOG && rescue
      self_destroy "job done; the Beelink launcher has been silent for $(( $(hb_age) / 60 )) min"; exit 0
    fi
    continue
  fi
  if ! job_alive; then
    log "no job process and no DONE line (heartbeat age $(hb_age) s)"
    if [ "$(hb_age)" -gt $HB_STALE ]; then rescue; self_destroy "job dead and the Beelink launcher silent"; exit 0; fi
    continue
  fi
  if [ $STOPPED = 0 ] && [ $EL -ge $((CAP - 15)) ]; then touch STOP; STOPPED=1; log "STOP file set (cap - 15 min)"; fi
  if [ $EL -ge $CAP ]; then
    log "HARD CAP $CAP min: ending the training (never an upload)"
    pkill -f "train_nll[b].py|merge_wis[e].py|f19_se[l].py" 2> /dev/null; sleep 10
    wait_uploads; rescue; wait_uploads
    self_destroy "hard cap"; exit 0
  fi
done
