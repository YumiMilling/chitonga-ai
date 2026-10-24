#!/bin/bash
# Stage 19 translator round, BOTH ARMS (1.3B from a050; 3.3B from the base), on ONE DigitalOcean GPU droplet, paid from
# the DO credit (Chris, 24 Oct 2026: "stage 19 (both arms, 1.3B and 3.3B) runs on DigitalOcean ... NOT RunPod"). The
# Jesuit, 24 Oct 2026. Built on the proven chat pattern (tools/chitonga/llm/launch_v14.sh + guard_v14.sh + run_v14.sh).
# NOTHING IS CREATED WITHOUT CHRIS'S GO (GO=1).
#
#   bash tools/chitonga/train/launch_stage19.sh dry          K0: bash -n, control bytes, every file, data = manifest =
#                                                            preflight, the frozen 44 / random 40 absent, mixes checked,
#                                                            the plan, the estimate and the cap printed. No HF, no DO call
#   bash tools/chitonga/train/launch_stage19.sh smoke        K0, Beelink CPU: run_stage19.sh end to end with TINY models
#                                                            (both arms, sweep -> select -> full -> merge -> 3.3B P1 with the
#                                                            new language code -> P2 -> merge -> uploads into a fake hub)
#   bash tools/chitonga/train/launch_stage19.sh data [--dry] K0: the private dataset PRIVATE_HF_ORG/chitonga-stage19data,
#                                                            every file read back and compared (the frozen 44 and the random
#                                                            classroom test never go up)
#   bash tools/chitonga/train/launch_stage19.sh prep         HF storage and empty repos; the DO token (list; create/delete
#                                                            rights probed with requests that cannot create or delete)
#   GO=1 bash tools/chitonga/train/launch_stage19.sh run     ONE droplet (H100 80 GB, else H200), run_stage19.sh, every
#                                                            repo verified on HF, the droplet deleted
# Before `run`, in another shell, the independent overwatch: python -X utf8 tools/chitonga/train/pod_overwatch.py 430 7
# (NOT while another chitonga machine runs: it ends EVERY chitonga-* machine once one passes its cap; DECISIONS 24 Oct).
#
# Pod rules (memory feedback-pod-scripts-guard, -tell-chris-when-pod-starts, -always-keep-trained-models,
# -convert-int8-on-pod) and the v14 lessons: one file per scp, each checked (test -s); ssh only through X() (timeout,
# stdin /dev/null); the job under nohup; 20 min to first SSH; SETUP DONE within 30 min; PACK OK within 20 min of it; no
# progress for 25 min = fail (an upload in flight has its own bound); EXIT trap: fetch, then delete unless weights exist
# ONLY on the droplet (then it stays up, a high-priority phone note, the caps still run); a local hard-cap timer deletes
# it at the cap; the MACHINE-SIDE guard (guard_stage19.sh) self-deletes 10 min after that or when the heartbeat stops;
# a key-free backstop timer on the droplet itself powers it off later still; notify.py on CREATED (purpose) and
# CONFIRMED GONE (results + cost). No `set -x` before .env is read. jinja2>=3.1 pinned in run_stage19.sh. The DO token
# cannot read the balance (403): the cost is written; Chris reads the balance in the dashboard.
set -u
cd "$(dirname "$0")/../../.."
R=internal/chitonga_train/runs/2026-10-24_stage19_data
R18=internal/chitonga_train/runs/2026-10-23_final_translator_data
T=$(grep "^HF_TOKEN_CHRIS=" .env.local | cut -d= -f2-)
DS=PRIVATE_HF_ORG/chitonga-stage19data
POD=chitonga-s19
ARMS=${ARMS:-"13 33"}
EXP=${EXP:-250}                 # minutes, both arms, no blends (RUN_PACKAGE.md 5); ARMS=13 alone: ~90
BUDGET_USD=${BUDGET_USD:-45}    # a RUNAWAY cap only, not a budget (Chris, 24 Oct: "the digital ocean budget is there to be used"); DO credit ~$54 minus ~$7 of v14 not yet posted; above that the card pays
SIZES=${SIZES:-"gpu-h100x1-80gb gpu-h200x1-141gb"}
MODE=${1:-dry}
SCRIPTS="train_nllb.py upload_model.py f19_sel.py f19_select.py merge_wise.py run_stage19.sh guard_stage19.sh rescue_stage19.sh"
mkdir -p $R/logs

checks() {
HF_TOKEN="$T" ARMS="$ARMS" EXP=$EXP BUDGET_USD=$BUDGET_USD python -X utf8 - "$@" <<'EOF'
import gzip, hashlib, json, os, shutil, sys, tempfile, urllib.request, urllib.error
sys.path.insert(0, "tools/chitonga/train")
import mix_groups as MG
R = "internal/chitonga_train/runs/2026-10-24_stage19_data"; R18 = "internal/chitonga_train/runs/2026-10-23_final_translator_data"
DS = "PRIVATE_HF_ORG/chitonga-stage19data"
mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for f in ("train.jsonl", "tasks.jsonl", "mono.jsonl", "dev.jsonl", "sel.jsonl", "test.en.txt", "test_hopgood.en.txt",
          "manifest.json", "preflight.json", "teacher_dev_stage19.jsonl", "synthetic_s19_heldout.jsonl",
          "confusable_probes_final18.jsonl", "random_classroom_40.jsonl", "random_classroom_40.jsonl.sha256"):
    assert os.path.getsize(os.path.join(R, f)) > 0, "MISSING " + f
pre = json.load(open(os.path.join(R, "preflight.json"), encoding="utf-8"))
man = json.load(open(os.path.join(R, "manifest.json"), encoding="utf-8"))
for f, want in pre["files"].items():
    assert sha(os.path.join(R, f)) == want, f"{f} changed since check_build (re-run build_stage19.py)"
assert man["guarded_rows_left"] == 0
norm = lambda s: {k: float(v) for k, v in (kv.split(":") for kv in s.split(","))}
for m in ("S19a", "S19b", "S19c", "P33"):
    assert norm(man["mixes"][m]) in [norm(x) for x in pre["mixes"]], f"mix {m} is not a checked mix"
    w = norm(man["mixes"][m]); assert w.get("choma", 0) == 0 and w.get("choma_med", 0) == 0
for f in ("dev.jsonl", "sel.jsonl", "test.en.txt", "test_hopgood.en.txt"):
    assert sha(os.path.join(R, f)) == sha(os.path.join(R18, f)), f"{f} differs from stage 18"
tt = os.path.join(R18, "teacher_test_final18.jsonl")
assert open(tt + ".sha256").read().split()[0] == sha(tt), "the frozen 44 was edited after its freeze"
rc = os.path.join(R, "random_classroom_40.jsonl")
assert open(rc + ".sha256").read().split()[0] == sha(rc), "the random classroom test was edited after its freeze"
held = set()
for p_ in (tt, rc, os.path.join(R, "teacher_dev_stage19.jsonl")):
    for l in open(p_, encoding="utf-8"):
        r = json.loads(l)
        held |= {MG._n(r.get("en")), MG._n(r.get("toi"))} | {MG._n(a) for a in r.get("alternatives", [])}
held.discard("")
bad = []
for f, keys in (("train.jsonl", ("en", "toi")), ("tasks.jsonl", ("src", "tgt")), ("mono.jsonl", ("toi",))):
    for l in open(os.path.join(R, f), encoding="utf-8"):
        r = json.loads(l)
        if any(MG._n((r.get(k) or "").replace("[BT] ", "")) in held for k in keys):
            bad.append((f, r.get("id")))
assert not bad, f"held-out lines in the build: {bad[:5]}"
tags = {}
for l in open(os.path.join(R, "train.jsonl"), encoding="utf-8"):
    r = json.loads(l); tags[(r.get("tag"), r.get("prov"))] = tags.get((r.get("tag"), r.get("prov")), 0) + 1
assert set(t for t, _ in tags) == {"<cls>", "<gen>"}, tags
print(f"CHECKS OK: data = preflight ({pre['files']['train.jsonl'][:16]}), 4 mixes checked, tests = stage 18's, the frozen 44 "
      f"/ random 40 / dev slice absent from train, tasks and mono; tags {dict(((a + ' ' + (b or '')).strip(), n) for (a, b), n in tags.items())}")
arms = os.environ["ARMS"].split(); exp = int(os.environ["EXP"]); price = 4.41; cap = float(os.environ["BUDGET_USD"])
print(f"PLAN: arms {arms}; arm 13 sweep S19a/b/c x 1,500 from a050, select, full x seeds 1 2 x 2,500 (no blends: Chris, 24 Oct); "
      f"arm 33 P1 = facebook/nllb-200-3.3B + toi_Latn, P33 untagged x 5,000 at 5e-5, P2 = arm 13's winner x seeds 1 2 x 2,500, "
      f"P1 kept as a model. ESTIMATE ~{exp} min at ${price}/h = ${exp / 60 * price:.2f}; safety cap ${cap:.0f} "
      f"(hard cap {min(int(cap * 60 / price) - 2, int(1.6 * exp + 30))} min)")
if mode == "dry":
    print("DRY OK (no HF, no DO call)"); sys.exit(0)
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ["HF_TOKEN"])
me = api.whoami()["name"]; assert me == "PRIVATE_HF_ORG", me
if mode in ("data", "data_dry"):
    if mode == "data_dry":
        print("DATA DRY OK (nothing uploaded)"); sys.exit(0)
    try:
        have = api.list_repo_files(DS, repo_type="dataset")
        assert not [f for f in have if f.endswith((".gz", ".json", ".txt", ".jsonl"))], f"{DS} already holds data: {have}"
    except Exception as e:
        if "already holds" in str(e):
            raise
    api.create_repo(DS, repo_type="dataset", private=True, exist_ok=True)
    assert api.dataset_info(DS).private
    stage = tempfile.mkdtemp(prefix="s19_up_")
    for f in ("train.jsonl", "tasks.jsonl", "mono.jsonl"):
        with open(os.path.join(R, f), "rb") as a, gzip.open(os.path.join(stage, f + ".gz"), "wb") as b:
            shutil.copyfileobj(a, b)
    for f in ("dev.jsonl", "test.en.txt", "test_hopgood.en.txt", "manifest.json", "preflight.json", "sel.jsonl",
              "teacher_dev_stage19.jsonl", "synthetic_s19_heldout.jsonl", "confusable_probes_final18.jsonl"):
        shutil.copy(os.path.join(R, f), stage)      # the frozen 44 and the random 40 NEVER go up
    api.upload_folder(folder_path=stage, repo_id=DS, repo_type="dataset", commit_message="stage 19 data (domain tags, scaled synthetic teacher register), 24 Oct 2026")
    back = tempfile.mkdtemp(prefix="s19_back_")
    for f in sorted(os.listdir(stage)):
        p = hf_hub_download(DS, f, repo_type="dataset", local_dir=back, force_download=True, token=os.environ["HF_TOKEN"])
        if f.endswith(".gz"):
            with gzip.open(p, "rb") as a:
                assert hashlib.sha256(a.read()).hexdigest() == pre["files"][f[:-3]], f"read-back differs: {f}"
        else:
            assert sha(p) == sha(os.path.join(stage, f)), f"read-back differs: {f}"
    shutil.rmtree(stage); shutil.rmtree(back)
    json.dump({"data_repo": DS, "train_sha256": pre["files"]["train.jsonl"]}, open(os.path.join(R, "launched_data.json"), "w"), indent=1)
    print("DATA UPLOADED AND VERIFIED", DS, "private; train sha", pre["files"]["train.jsonl"][:16]); sys.exit(0)
# prep / run: storage, empty repos, the DO token's rights
g = lambda i: getattr(i, "used_storage", None) or getattr(i, "usedStorage", None) or 0
used = sum(g(api.model_info(m.id, expand=["usedStorage"])) for m in api.list_models(author=me, expand=["private"]) if m.private)
used += sum(g(api.dataset_info(d.id, expand=["usedStorage"])) for d in api.list_datasets(author=me, expand=["private"]) if d.private)
need = (5 * 6.8 if "13" in arms else 0) + (3 * 16.6 if "33" in arms else 0)     # no blends on the droplet (Chris, 24 Oct)
free = float(os.environ.get("HF_PRIVATE_LIMIT_GB", "1000")) - used / 1e9
print(f"HF private storage: used {used / 1e9:.1f} GB, free {free:.1f}, need {need:.0f} (every model full + int8)")
assert free >= need, "PREFLIGHT FAILED: not enough private storage"
assert set(api.list_repo_files(DS, repo_type="dataset")) >= {"train.jsonl.gz", "preflight.json", "manifest.json"}, "run 'data' first"
for repo in [f"{me}/chitonga-nllb-s19-{x}" for x in ("sweepS19a", "sweepS19b", "sweepS19c", "B13-s1", "B13-s2")] + \
            [f"{me}/chitonga-nllb33-s19-{x}" for x in ("P1", "B33-s1", "B33-s2")]:
    try:
        fs = api.list_repo_files(repo)
        assert not [f for f in fs if f.endswith((".safetensors", ".bin"))], f"{repo} already holds weights"
    except Exception as e:
        if "already holds" in str(e):
            raise
tok = [l.split("=", 1)[1].strip() for l in open(".env.local", encoding="utf-8") if l.startswith("DO_API_TOKEN=")][0]
def do(m, path, body=None):
    r = urllib.request.Request("https://api.digitalocean.com/v2/" + path, method=m, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, json.loads(f.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
c, d = do("GET", "droplets?per_page=100"); assert c == 200, (c, d)
assert not d["droplets"], f"droplets exist: {[x['name'] for x in d['droplets']]}"
c1, _ = do("POST", "droplets", {"name": "rights-probe", "region": "zz9", "size": "no-such-size", "image": "no-such-image"})
c2, _ = do("DELETE", "droplets/1")
print(f"DO token: list 200 (none running); create probe HTTP {c1} (422 = authorised); delete probe HTTP {c2} (404 = authorised)")
assert c1 == 422 and c2 == 404, "DO token lacks create or delete rights"
print("PREFLIGHT OK")
EOF
}

case $MODE in
  dry) for f in $SCRIPTS; do [ -s tools/chitonga/train/$f ] || { echo "MISSING $f"; exit 1; }; done
       for f in run_stage19.sh guard_stage19.sh rescue_stage19.sh; do bash -n tools/chitonga/train/$f || exit 1; done
       bash -n "$0" || exit 1; echo "bash -n OK"
       grep -c $'\x08' tools/chitonga/train/{run_stage19.sh,guard_stage19.sh,rescue_stage19.sh,train_nllb.py,f19_sel.py,f19_select.py,build_stage19.py,synth_t25.py} "$0" | grep -v ":0$" && { echo "CONTROL BYTES"; exit 1; }
       grep -v "^ *#" tools/chitonga/train/run_stage19.sh | grep -q -- "--no-save" && { echo "never --no-save"; exit 1; }
       grep -n "set -x" tools/chitonga/train/run_stage19.sh tools/chitonga/train/guard_stage19.sh | grep -v "#" && { echo "set -x found"; exit 1; }
       python -X utf8 tools/chitonga/train/f19_select.py selftest || exit 1
       checks dry; exit $? ;;
  smoke) SM=${SMOKE_DIR:-${TEMP:-/tmp}/claude/s19_smoke}; rm -rf "$SM"; mkdir -p "$SM/s19"
       # a small, re-checked copy of the data (the RAM note in smoke_stage19_data.py): the code path, not the data
       python -X utf8 tools/chitonga/train/smoke_stage19_data.py $R "$SM/s19" > "$SM/data.log" 2>&1 || { tail -5 "$SM/data.log"; exit 1; }
       tail -1 "$SM/data.log"
       for f in $SCRIPTS; do cp tools/chitonga/train/$f "$SM/$f"; done
       python -X utf8 tools/chitonga/train/smoke_tiny_nllb.py "$SM/tiny" > "$SM/tiny.log" 2>&1 || { cat "$SM/tiny.log"; exit 1; }
       ( cd "$SM" && OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SMOKE=1 SMOKE_DIR="$SM" SMOKE_INIT="$SM/tiny/init_tiny" SMOKE_BASE="$SM/tiny/base_tiny" \
           ARMS="$ARMS" UPLOAD_TICK=5 SMOKE_FORCE_WINNER=${SMOKE_FORCE_WINNER:-S19a} bash run_stage19.sh > run_stage19.log 2>&1 )
       # the whole job log (trainer + selector + scorer output) is kept beside the package, not only the summary lines
       cp "$SM/run_stage19.log" $R/logs/smoke_run_stage19.log; cp "$SM/sel/select.txt" $R/logs/smoke_select.txt 2>/dev/null
       grep -E "^(RECIPE|PACK OK|TRAIN |WINNER|SMOKE:|RULE|MERGE|ROW TAG|F19_|GRADIENT)|TRAINING DONE|UPLOADED|FAILED|Error|Traceback|peak" "$SM/run_stage19.log" | cut -c1-240
       grep -A3 "Traceback" "$SM/run_stage19.log" | head -20
       N=$(ls -d "$SM"/fakehub/*/*/ 2>/dev/null | wc -l); echo "SMOKE: $N repos in the fake hub"
       ( cd "$SM" && SMOKE=1 SMOKE_DIR="$SM" bash rescue_stage19.sh | tail -2 )
       grep -q "^F19_DONE" "$SM/run_stage19.log" || { echo "SMOKE FAILED (log: $SM/run_stage19.log)"; exit 1; }
       ls "$SM"/fakehub/*/*-ct2-int8/model.bin > /dev/null 2>&1 && echo "INT8: $(ls "$SM"/fakehub/*/*-ct2-int8/model.bin | wc -l) real CTranslate2 int8 builds"
       bash tools/chitonga/train/smoke_guard19.sh "$SM/guard_test" || exit 1
       echo "SMOKE OK"
       exit 0 ;;
  data) if [ "${2:-}" = "--dry" ]; then checks data_dry; else checks data; fi; exit $? ;;
  prep) checks prep; exit $? ;;
  run) ;;
  *) echo "usage: launch_stage19.sh dry|smoke|data [--dry]|prep|run"; exit 1 ;;
esac
[ "${GO:-0}" = 1 ] || { echo "run needs GO=1 (Chris's explicit go). Nothing created."; exit 1; }
[ -s $R/launched_data.json ] || { echo "no launched_data.json: run 'data' first"; exit 1; }
checks prep || exit 1
DATA_SHA=$(python -X utf8 -c "import json;print(json.load(open('$R/launched_data.json'))['train_sha256'])")
[ "$(sha256sum $R/train.jsonl | cut -d' ' -f1)" = "$DATA_SHA" ] || { echo "local train.jsonl differs from the uploaded one: 'data' again"; exit 1; }
[ -z "$(python -X utf8 tools/chitonga/train/do_gpu.py status | grep -v '^$')" ] || { echo "a DO droplet is already up: stop"; exit 1; }
WDIR=/root/s19; P=22; PRICE_DEFAULT=4.41
S="${TEMP:-/tmp}/claude/s19"; mkdir -p "$S"
TF=$R/logs/times.txt; LOG=run_stage19.log
H=""; DOWN=0; PRICE=""; START=$(date +%s); RESULT="no result"
O="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i $HOME/.ssh/zamai_sandbox"
X() { timeout 120 ssh -p $P $O $H "$1" < /dev/null; }
wall() { echo $(( ($(date +%s) - START) / 60 )); }
UPLOAD_MAX=${UPLOAD_MAX:-60}
upload_elapsed() { [ -n "$H" ] || { echo ""; return; }; X "p=\$(pgrep -f 'upload_mode[l].py|rescue_stage1[9].sh' | head -1); [ -n \"\$p\" ] && ps -o etimes= -p \$p | tr -d ' '" 2>/dev/null | tr -d '\r'; }
wait_uploads_remote() { local i e; for i in $(seq 1 $UPLOAD_MAX); do e=$(upload_elapsed); [ -n "$e" ] || return 0; echo "upload in flight (${e}s): waiting before delete"; sleep 60; done; }
heartbeat() { [ -n "$H" ] && X "date +%s > $WDIR/HEARTBEAT" > /dev/null 2>&1; }
cost() { python -c "print(round(($(date +%s) - $START) / 3600 * ${PRICE:-$PRICE_DEFAULT}, 2))"; }
mstatus() { python -X utf8 tools/chitonga/train/do_gpu.py status | grep " $POD "; }
mdown() { python -X utf8 tools/chitonga/train/do_gpu.py down $POD | tail -1; }
fetch() {
  [ -n "$H" ] || { echo "FETCH: address unknown"; return 0; }
  local F=$R/logs/f_$(date +%Y%m%d_%H%M%S) f; mkdir -p $F
  for f in $LOG pip.log download.log guard.log guard.out rescue.log rescue_backstop.log backstop.log sel/select.txt \
           $(X "cd $WDIR && ls sel/*.json up_*.log merge_*.log s19/*/dev_loss.tsv s19/*/config.json 2>/dev/null" | tr -d '\r'); do
    mkdir -p "$F/$(dirname $f)"; timeout 120 scp -P $P $O "$H:$WDIR/$f" "$F/$f" < /dev/null > /dev/null 2>&1 || true
  done
  echo "FETCHED into $F"
}
unique() {   # trained or merged folders on the droplet without UPOK (both uploads verified)
  [ -n "$H" ] || { echo 0; return; }
  local n; n=$(X "cd $WDIR && for d in s19/S19*/ckpt_* merged/S19*; do case \$d in *_ct2*) continue;; esac; [ -f \$d/config.json ] && [ ! -f \$d/UPOK ] && echo \$d; done | wc -l" 2>/dev/null | tr -d '\r ')
  [ "${n:-0}" -gt 0 ] 2>/dev/null && echo 1 || echo 0
}
terminate() {
  echo "TERMINATING $POD: $1"; wait_uploads_remote; mdown; DOWN=1
  [ -n "${CAPT:-}" ] && kill $CAPT 2>/dev/null
  echo "END ($1) $(date '+%F %T') WALL_MIN $(wall) COST_USD_EST $(cost)" >> $TF
  local i; for i in 1 2 3 4 5 6 8 10; do
    [ -z "$(mstatus)" ] && { echo "CONFIRMED GONE: $POD"; python -X utf8 tools/notify.py "stage 19 translator (both arms): $POD confirmed gone after $(wall) min, ~\$$(cost) (DO credit). Result: $RESULT" --title "Tonga pod" > /dev/null 2>&1; return 0; }
    sleep 10
  done
  python -X utf8 tools/notify.py "$POD is STILL RUNNING on DO after delete: delete it by hand now" --title "Tonga pod" --high > /dev/null 2>&1
}
fail() {
  echo "FAILURE: $1"; RESULT="FAILED: $1"; fetch
  if [ -n "$H" ]; then X "cd $WDIR && pgrep -f 'train_nll[b].py|merge_wis[e].py|f19_se[l].py' > /dev/null || bash rescue_stage19.sh >> rescue.log 2>&1; tail -3 rescue.log" || true; fi
  if [ "$(unique)" = 1 ]; then
    python -X utf8 tools/notify.py "stage 19: $1. $POD LEFT UP ON PURPOSE: weights not yet on HF. Check it; the caps still run." --title "Tonga pod" --high > /dev/null 2>&1
    DOWN=1; echo "KEEP: weights exist only on the droplet"; exit 2
  fi
  terminate "$1"; exit 1
}
on_exit() {
  rc=$?; [ $DOWN = 1 ] && return
  if [ -z "$H" ]; then [ -n "$(mstatus 2>/dev/null)" ] && terminate "launcher exit before ssh (rc=$rc)"; return; fi
  fetch
  if [ "$(unique)" = 1 ]; then
    python -X utf8 tools/notify.py "stage 19: launcher died (rc=$rc); $POD LEFT UP: weights only on the droplet. The droplet's own guard rescues and deletes it." --title "Tonga pod" --high > /dev/null 2>&1
  else terminate "launcher exit rc=$rc, nothing unique on the droplet"; fi
}
trap on_exit EXIT
UP=$(DO_NAME=$POD python -X utf8 tools/chitonga/train/do_gpu.py up $SIZES 2>&1 | tail -1)
echo "$UP" | cut -c1-200 | tee -a $TF
echo "$UP" | grep -q '"id"' || { echo "NO DROPLET (DO capacity): nothing created"; exit 1; }
PRICE=$(echo "$UP" | grep -oE '"price_hourly": [0-9.]*' | grep -o '[0-9.]*$'); PRICE=${PRICE:-$PRICE_DEFAULT}
START=$(date +%s)
CAP_MIN=$(python -c "print(min(int($BUDGET_USD * 60 / $PRICE) - 2, int(1.6 * $EXP + 30)))")
echo "MACHINE $POD created $(date '+%F %T') on DO \$$PRICE/h; expect ~$EXP min (~\$$(python -c "print(round($EXP / 60 * $PRICE, 2))")), hard cap $CAP_MIN min (\$$(python -c "print(round($CAP_MIN / 60 * $PRICE, 2))"))" | tee -a $TF
python -X utf8 tools/notify.py "stage 19 STARTED: $POD on DO at \$$PRICE/h. For: the translator with domain tags (<cls> classroom) and the scaled teacher synthetic; arms $ARMS (1.3B from a050; 3.3B from the base). ~$EXP min, hard cap $CAP_MIN min." --title "Tonga pod" > /dev/null 2>&1
( sleep $((CAP_MIN * 60)); echo "HARD CAP $(date)"
  if [ -s "$S/host" ]; then HH=$(cut -d' ' -f1 "$S/host")
    for i in $(seq 1 $UPLOAD_MAX); do e=$(timeout 60 ssh -p 22 $O $HH "p=\$(pgrep -f 'upload_mode[l].py|rescue_stage1[9].sh' | head -1); [ -n \"\$p\" ] && echo up" < /dev/null 2>/dev/null); [ "$e" = up ] || break; echo "hard cap: upload in flight ($i)"; sleep 60; done
  fi
  mdown ) >> $TF 2>&1 &
CAPT=$!
s=""; for i in $(seq 1 120); do s=$(mstatus); echo "$s" | grep -q "ssh root" && break; sleep 10; done
echo "$s" | grep -q "ssh root" || fail "machine never reachable in 20 min"
H=$(echo "$s" | grep -o "root@[0-9.]*"); P=22
ok=0; for i in $(seq 1 18); do X "mkdir -p $WDIR" > /dev/null 2>&1 && { ok=1; break; }; sleep 10; done
[ $ok = 1 ] || fail "ssh never answered"
echo "$H $P" > "$S/host"
printf 'HF_TOKEN=%s\nDATA_SHA=%s\nARMS=%s\nW=%s\nPROV=do\nBACKSTOP_MIN=%s\nP1_STEPS=%s\nSEEDS33=%s\nDO_API_TOKEN=%s\n' "$T" "$DATA_SHA" "$ARMS" "$WDIR" "$((CAP_MIN + 30))" "${P1_STEPS:-5000}" "${SEEDS33:-1 2}" \
  "$(grep '^DO_API_TOKEN=' .env.local | cut -d= -f2-)" > "$S/env"
FILES=""
put() { timeout 300 scp -P $P $O "$1" "$H:$WDIR/$2" < /dev/null || { rm -f "$S/env"; fail "scp failed: $1"; }; X "test -s $WDIR/$2" || { rm -f "$S/env"; fail "file empty on the machine: $2"; }; FILES="$FILES $2"; }
for f in $SCRIPTS; do put tools/chitonga/train/$f $f; done
put "$S/env" .env; rm -f "$S/env"
X "grep -q 'row-tags' $WDIR/train_nllb.py && grep -q 'F19_DONE' $WDIR/run_stage19.sh && echo NEWOPT" | grep -q NEWOPT || fail "files on the machine are not the stage 19 versions"
echo "FILES OK:$FILES"
X "cd $WDIR && sed -i 's/\r$//' *.sh *.py && nohup bash run_stage19.sh > $LOG 2>&1 < /dev/null & sleep 3; echo started" | grep -q started || fail "job did not start"
echo "JOB STARTED $(date '+%F %T')" | tee -a $TF
heartbeat
X "cd $WDIR && nohup bash guard_stage19.sh $((CAP_MIN + 10)) > guard.out 2>&1 < /dev/null & sleep 2; pgrep -f 'guard_stage1[9].sh' > /dev/null && echo guardup" | grep -q guardup || fail "the machine-side guard did not start"
echo "MACHINE GUARD UP (cap $((CAP_MIN + 10)) min; key-free backstop $((CAP_MIN + 30)) min)"
t0=$(date +%s)
until X "grep -q 'SETUP DONE' $WDIR/$LOG"; do heartbeat; X "grep -q 'F19_FAILED' $WDIR/$LOG" && fail "setup: $(X "grep FAILED $WDIR/$LOG | tail -1")"; [ $(( $(date +%s) - t0 )) -gt 1800 ] && fail "no SETUP DONE within 30 min"; sleep 30; done
PRQ=$(X "grep '^CAP PREREQS' $WDIR/$LOG | tail -1" | tr -d '\r')
echo "$PRQ" | grep -q "machine_id=NONE\|curl=NO\|pgrep=NO\|pkill=NO" && fail "machine-side cap cannot arm: $PRQ"
echo "SETUP DONE; MACHINE CAP ARMED: $PRQ"
t1=$(date +%s)
until X "grep -q 'PACK OK' $WDIR/$LOG"; do heartbeat; X "grep -q 'F19_FAILED' $WDIR/$LOG" && fail "data: $(X "grep FAILED $WDIR/$LOG | tail -1")"; [ $(( $(date +%s) - t1 )) -gt 1200 ] && fail "no PACK OK within 20 min of SETUP DONE"; sleep 30; done
echo "SMOKE OK (PACK OK: data sha + every mix checked on the droplet) $(date +%H:%M)"
last=""; still=0
until X "grep -qE '^F19_(DONE|FAILED)' $WDIR/$LOG"; do
  sleep 60; heartbeat
  prog=$(X "grep -E '^(TRAIN |step |THROUGHPUT|F19SEL|WINNER|MERGE|RUN uploads|S19.* TRAINING DONE|ROW TAG)' $WDIR/$LOG | tail -1" | cut -c1-200)
  size=$(X "cd $WDIR; cat $LOG up_*.log merge_*.log 2>/dev/null | wc -c")
  upl=$(upload_elapsed)
  echo "$(date +%H:%M) [$(wall)/$CAP_MIN min, ~\$$(cost)] $prog | logs $size${upl:+ | upload in flight ${upl}s}"
  echo "$prog" | grep -q "THROUGHPUT LOW" && fail "$prog"
  if [ -n "$upl" ]; then still=0; [ "$upl" -gt $((UPLOAD_MAX * 60)) ] && fail "an upload has run for over $UPLOAD_MAX min"
  else now="$prog|$size"; if [ "$now" = "$last" ]; then still=$((still + 1)); else still=0; fi; last="$now"; [ $still -ge 25 ] && fail "no progress for 25 min"; fi
done
fetch
X "grep -E '^(WINNER|F19SEL|F19_)|TRAINING DONE|UPLOADED|peak' $WDIR/$LOG | tail -80" | tee $R/logs/summary.txt
X "grep -q '^F19_DONE' $WDIR/$LOG" || fail "the job ended F19_FAILED"
HF_TOKEN="$T" ARMS="$ARMS" python -X utf8 - <<'EOF' || fail "the launcher's own HF check failed"
import os
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"]); bad = []
names = []
if "13" in os.environ["ARMS"].split():
    names += [f"chitonga-nllb-s19-{x}" for x in ("sweepS19a", "sweepS19b", "sweepS19c", "B13-s1", "B13-s2")]
if "33" in os.environ["ARMS"].split():
    names += [f"chitonga-nllb33-s19-{x}" for x in ("P1", "B33-s1", "B33-s2")]
for n in names:
    for repo, need in ((f"PRIVATE_HF_ORG/{n}", ".safetensors"), (f"PRIVATE_HF_ORG/{n}-ct2-int8", "model.bin")):
        try:
            fs = a.list_repo_files(repo); priv = a.model_info(repo).private
        except Exception as e:
            bad.append((repo, repr(e)[:60])); continue
        if not any(f.endswith(need) for f in fs): bad.append((repo, "missing " + need))
        if not priv: bad.append((repo, "NOT PRIVATE"))
print("HF VERIFIED", 2 * len(names), "repos" if not bad else f"NOT VERIFIED {bad}")
raise SystemExit(1 if bad else 0)
EOF
RESULT="arms $ARMS trained, every model full + int8 verified on HF; winner $(grep -o 'WINNER S19[a-c]' $R/logs/summary.txt | tail -1); next: the kill test on the Beelink (PACK.md 4)"
terminate "done and verified"
echo "S19 LAUNCH DONE: $RESULT"
