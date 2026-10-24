#!/bin/bash
# The DELETE path of guard_stage19.sh, exercised on the Beelink against a FAKE droplet id (the Jesuit, 24 Oct 2026):
# a fake `curl` on PATH answers the metadata call with 99999999 and records every other call instead of sending it.
# The job is marked finished (F19_DONE) and the launcher's heartbeat made stale: the guard must then self-delete,
# i.e. issue DELETE droplets/99999999 with the DO token header. Nothing leaves this machine.
set -u
D=${1:?dir}; command -v cygpath > /dev/null && D=$(cygpath -u "$D")   # a C:/ path would break PATH at its colon (Windows smoke)
rm -rf "$D"; mkdir -p "$D/bin"
cp "$(dirname "$0")/guard_stage19.sh" "$(dirname "$0")/rescue_stage19.sh" "$D/"
cat > "$D/bin/curl" <<'C'
#!/bin/bash
case "$*" in *169.254.169.254*) echo 99999999 ;; *) echo "FAKE CURL $*" >> "$(dirname "$0")/../curl_calls.log"; echo "DELETE droplets/99999999 -> HTTP 204" ;; esac
C
chmod +x "$D/bin/curl"
printf 'DO_API_TOKEN=FAKE_TOKEN_FOR_SMOKE\nPROV=do\nW=%s\n' "$D" > "$D/.env"
echo "F19_DONE, UPLOADS VERIFIED (smoke)" > "$D/run_stage19.log"
echo 0 > "$D/HEARTBEAT"                         # stale: the launcher is 'gone'
( cd "$D" && PATH="$D/bin:$PATH" W="$D" GUARD_TICK=1 HB_STALE=5 SMOKE=1 SMOKE_DIR="$D" timeout 60 bash guard_stage19.sh 30 )
grep -q "DELETE droplets/99999999" "$D/guard.log" "$D/curl_calls.log" 2>/dev/null && grep -q "Authorization: Bearer FAKE_TOKEN_FOR_SMOKE" "$D/curl_calls.log" \
  && echo "GUARD DELETE PATH OK: $(grep -h 'SELF-DESTROY' "$D/guard.log")" || { echo "GUARD DELETE PATH FAILED"; cat "$D/guard.log"; exit 1; }
