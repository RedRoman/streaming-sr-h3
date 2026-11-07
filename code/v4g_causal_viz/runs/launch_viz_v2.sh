#!/bin/bash
# Usage: launch_viz_v2.sh TAG PHYSICAL_NPU [--check]
# A worker inherits all three flock descriptors for its entire lifetime.
set -euo pipefail
TAG="${1:?usage: launch_viz_v2.sh TAG PHYSICAL_NPU [--check]}"
PHYSICAL_NPU="${2:?explicit physical NPU index is required}"
MODE="${3:-run}"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || exit 64
[[ "$PHYSICAL_NPU" =~ ^[0-7]$ ]] || exit 64
[[ "$MODE" == run || "$MODE" == --check ]] || exit 64
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${H3_VIZ_ROOT:-$PACKAGE_ROOT}"
PYTHON="${H3_VIZ_PYTHON:-python3}"
: "${H3_VIZ_DEVICE_LOCK:?set the existing shared device-lock path for the selected NPU}"
: "${H3_VIZ_TASK_LOCK:?set the existing shared task-lock path for the selected NPU}"
: "${H3_VIZ_VAE_LOCK:?set the existing shared VAE-lock path for the selected NPU}"
mkdir -p "$ROOT/runs"
ROOT="$(cd -- "$ROOT" && pwd)"
export H3_VIZ_ROOT="$ROOT"
cd "$ROOT"
command -v flock >/dev/null
LOG="runs/viz_v2_${TAG}.log"
EXIT_FILE="$ROOT/runs/viz_v2_${TAG}.exit.json"
OUT="runs/v4g_causal_stream_viz_VIDGEN5S_0086441_${TAG}"
if [[ -e "$OUT" || -e "$LOG" || -e "$EXIT_FILE" ]]; then
  echo '{"event":"aborted_attempt_exists"}' >&2
  exit 73
fi
# Do not truncate shared lock files. Release on refusal; inherit on launch.
exec 200>>"$H3_VIZ_DEVICE_LOCK"
exec 201>>"$H3_VIZ_TASK_LOCK"
exec 202>>"$H3_VIZ_VAE_LOCK"
for FD in 200 201 202; do
  flock -n "$FD" || { echo '{"event":"waiting_for_device_lock"}' >&2; exit 75; }
done
# Parse leading whitespace and colon-separated PIDs; never exempt a foreign PID.
"$PYTHON" - "$PHYSICAL_NPU" <<'PY'
import json, re, subprocess, sys
device = int(sys.argv[1])
r = subprocess.run(['npu-smi', 'info', '-t', 'proc-mem', '-i', str(device)],
                   text=True, capture_output=True, timeout=10)
reported = re.search(r'(?m)^\s*NPU ID\s*:\s*(\d+)\s*$', r.stdout)
if r.returncode or not reported or int(reported[1]) != device:
    print(json.dumps({'event':'device_query_unverified','physical_npu':device}))
    raise SystemExit(75)
pids = sorted(set(int(x) for x in re.findall(r'\bProcess id:\s*(\d+)\b', r.stdout)))
if pids:
    print(json.dumps({'event':'waiting_for_idle_device','physical_npu':device,'pids':pids}))
    raise SystemExit(75)
print(json.dumps({'event':'device_and_locks_verified','physical_npu':device}))
PY
if [[ "$MODE" == --check ]]; then
  echo '{"event":"check_only_no_job_started"}'
  exit 0
fi
# Exclusive creation prevents log overwrites and duplicate tags.
(set -o noclobber; : > "$LOG")
export PYTHONPATH="${H3_VIZ_PYTHONPATH:-${PYTHONPATH:-}}"
export ASCEND_OPP_PATH="${H3_VIZ_ASCEND_OPP_PATH:-${ASCEND_OPP_PATH:-}}"
export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_NPU"
nohup bash -c '
  status_file=$1
  shift
  set +e
  "$@"
  rc=$?
  printf "{\"exit_code\":%s,\"finished_utc\":\"%s\"}\n" "$rc" "$(date -u +%FT%TZ)" > "${status_file}.tmp"
  mv "${status_file}.tmp" "$status_file"
  exit "$rc"
' _ "$EXIT_FILE" "$PYTHON" -u "$PACKAGE_ROOT/experiment/viz_v4g_causal_stream_v2.py" \
  --device npu:0 --tag "$TAG" >> "$LOG" 2>&1 &
WORKER_PID=$!
printf '{"event":"started","tag":"%s","physical_npu":%s,"pid":%s}\n' \
  "$TAG" "$PHYSICAL_NPU" "$WORKER_PID" | tee -a runs/viz_v2_launch.log
