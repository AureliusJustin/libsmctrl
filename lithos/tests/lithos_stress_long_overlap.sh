#!/usr/bin/env bash
set -euo pipefail

# Stress-launch many long-running arbitrary apps under lithosd and capture
# scheduler + nvidia-smi snapshots so overlap can be inspected post-run.

APPS="${1:-40}"
TPCS="${2:-40}"
CYCLES="${3:-20000000000}"
OUTDIR="${4:-/tmp}"
SOCK="${OUTDIR}/lithosd.sock"
MONITOR_TIMEOUT_SEC="${LITHOS_STRESS_MONITOR_TIMEOUT_SEC:-900}"
VERBOSE="${LITHOS_STRESS_VERBOSE:-1}"

mkdir -p "$OUTDIR"

make lithosd lithos_test_arbitrary_app libcuda.so.1 >/dev/null

./lithosd "$SOCK" "$TPCS" >"$OUTDIR/daemon.log" 2>&1 &
DAEMON_PID=$!

cleanup() {
	kill "$DAEMON_PID" 2>/dev/null || true
	wait "$DAEMON_PID" 2>/dev/null || true
	rm -f "$SOCK"
}
trap cleanup EXIT

sleep 0.2

APP_PIDS=()
for i in $(seq 1 "$APPS"); do
	(
		LIBSMCTRL_LITHOS_ENABLE=1 \
		LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE=1 \
		LIBSMCTRL_LITHOSD_SOCK="$SOCK" \
		LIBSMCTRL_LITHOS_TPC_QUOTA_DEFAULT=1 \
		LIBSMCTRL_LITHOS_TEST_LONG_KERNEL_CYCLES="$CYCLES" \
		LD_LIBRARY_PATH=. \
		./lithos_test_arbitrary_app >"$OUTDIR/app_$i.out" 2>"$OUTDIR/app_$i.err"
	) &
	APP_PIDS[${#APP_PIDS[@]}]=$!
done

STATUS_LOG="$OUTDIR/status.log"
SMI_LOG="$OUTDIR/nvidia_smi.log"
PEAK_ACTIVE=0
PEAK_STATUS_FILE="$OUTDIR/status_peak.txt"
MONITOR_START_TS="$(date +%s)"

is_pid_running_non_zombie() {
	local pid="$1"
	local stat

	if ! kill -0 "$pid" 2>/dev/null; then
		return 1
	fi
	stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
	if [[ -z "$stat" ]]; then
		return 1
	fi
	# Zombies still have a PID until parent wait() but should be treated done.
	if [[ "$stat" == Z* ]]; then
		return 1
	fi
	return 0
}

while true; do
	active_apps=0
	for pid in "${APP_PIDS[@]}"; do
		if is_pid_running_non_zombie "$pid"; then
			active_apps=1
			break
		fi
	done
	if (( active_apps == 0 )); then
		break
	fi

	now_ts="$(date +%s)"
	if (( now_ts - MONITOR_START_TS > MONITOR_TIMEOUT_SEC )); then
		echo "monitor_timeout=1 elapsed_sec=$((now_ts - MONITOR_START_TS))" >>"$OUTDIR/summary.txt"
		break
	fi

	TS="$(date +%s.%N)"
	STATUS_LINE="$(./lithosd --status "$SOCK" | head -n1 || true)"
	echo "$TS $STATUS_LINE" >>"$STATUS_LOG"

	ACTIVE="$(echo "$STATUS_LINE" | sed -n 's/.*active_allocs=\([0-9][0-9]*\).*/\1/p')"
	if [[ -n "$ACTIVE" ]] && (( ACTIVE > PEAK_ACTIVE )); then
		PEAK_ACTIVE=$ACTIVE
		./lithosd --status "$SOCK" >"$PEAK_STATUS_FILE" || true
	fi

	if [[ "$VERBOSE" == "1" ]]; then
		echo "[$TS] active_allocs=${ACTIVE:-0} peak=$PEAK_ACTIVE"
	fi

	if command -v nvidia-smi >/dev/null 2>&1; then
		{
			echo "=== $TS ==="
			nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
		} >>"$SMI_LOG"
	fi

	sleep 0.2
done

for pid in "${APP_PIDS[@]}"; do
	wait "$pid" || true
done

FINAL_STATUS="$(./lithosd --status "$SOCK" | head -n1 || true)"

echo "peak_active_allocs=$PEAK_ACTIVE" >"$OUTDIR/summary.txt"
echo "$FINAL_STATUS" >>"$OUTDIR/summary.txt"

awk 'BEGIN{bad=0;nc=0;c=0}
/^RESULT /{
	u=0;s=0;
	for(i=1;i<=NF;i++){
		if($i ~ /^uniq=/){split($i,a,"=");u=a[2]+0}
		if($i ~ /^sms_per_tpc=/){split($i,b,"=");s=b[2]+0}
	}
	if(u<=s)c++; else nc++;
	next
}
{bad++}
END{printf("confined=%d not_confined=%d parse_bad=%d\n",c,nc,bad)}' "$OUTDIR"/app_*.out >>"$OUTDIR/summary.txt"

python3 - "$OUTDIR" >>"$OUTDIR/summary.txt" <<'PY'
import glob, re, sys
outdir = sys.argv[1]
ivs = []
for p in sorted(glob.glob(f"{outdir}/app_*.out")):
    txt = open(p, 'r', encoding='utf-8', errors='ignore').read().strip()
    m = re.search(r"min=(\d+) max=(\d+)", txt)
    if not m:
        continue
    ivs.append((int(m.group(1)), int(m.group(2))))
ivs.sort()
disjoint = True
for i in range(1, len(ivs)):
    if ivs[i][0] <= ivs[i - 1][1]:
        disjoint = False
        break
print(f"pairwise_disjoint={1 if disjoint else 0} intervals={len(ivs)}")
PY

python3 - "$PEAK_STATUS_FILE" >>"$OUTDIR/summary.txt" <<'PY'
import re, sys
path = sys.argv[1]
starts = []
quotas = []
try:
	with open(path, 'r', encoding='utf-8', errors='ignore') as f:
		lines = [l.strip() for l in f if l.strip()]
except FileNotFoundError:
	print("peak_status_disjoint=unknown slots=0 reason=no_peak_snapshot")
	raise SystemExit(0)

for line in lines:
	if line.startswith("lithosd status:") or line.startswith("slot"):
		continue
	parts = line.split()
	if len(parts) < 7:
		continue
	# row format: slot pid launch stream quota start_tpc disable_mask
	try:
		quotas.append(int(parts[4]))
		starts.append(int(parts[5]))
	except ValueError:
		pass

if not starts:
	print("peak_status_disjoint=unknown slots=0 reason=no_rows")
	raise SystemExit(0)

intervals = sorted((s, s + q - 1) for s, q in zip(starts, quotas))
disjoint = True
for i in range(1, len(intervals)):
	if intervals[i][0] <= intervals[i - 1][1]:
		disjoint = False
		break
print(f"peak_status_disjoint={1 if disjoint else 0} slots={len(intervals)}")
PY

echo "Stress run complete: $OUTDIR"
cat "$OUTDIR/summary.txt"
