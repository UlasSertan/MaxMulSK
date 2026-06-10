#!/usr/bin/env bash
# profile_power.sh — wrap powermetrics around a binary run and report
# average CPU power, total energy in Joules, and thermal pressure.
#
# Usage:
#   ./profile_power.sh                   # output saved as traces/<timestamp>_power.txt
#   ./profile_power.sh 4x1               # output saved as traces/4x1_power.txt
#   ./profile_power.sh --no-build 4x1    # skip rebuild
#
# Requires: sudo (powermetrics)
#
# Produces:
#   traces/NAME_power.txt   — summary (avg P/E/combined power, runtime, J, J/GFLOP, thermal)
#   traces/NAME_power.log   — raw binary stdout
#   traces/NAME_power.raw   — raw powermetrics output
#
# Edit main.cpp to leave only the kernel you want to measure active before running.

set -eo pipefail

if ! command -v cmake &>/dev/null; then
    export PATH="/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin:$PATH"
fi

BUILD_DIR="cmake-build-release"
BINARY="$BUILD_DIR/MatrixLibrary"
OUT_DIR="traces"
JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
SAMPLE_INTERVAL_MS=100   # 10 Hz — fine enough to catch a multi-second run

DO_BUILD=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) DO_BUILD=0; shift ;;
        --binary)   BINARY="$2"; DO_BUILD=0; shift 2 ;;
        --args)     IFS=' ' read -r -a EXTRA_ARGS <<< "$2"; shift 2 ;;
        -*)         echo "unknown flag: $1" >&2; exit 1 ;;
        *)          break ;;
    esac
done

NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG="$OUT_DIR/${NAME}_power.log"
PMLOG="$OUT_DIR/${NAME}_power.raw"
SUMMARY="$OUT_DIR/${NAME}_power.txt"

mkdir -p "$OUT_DIR"
rm -f "$LOG" "$PMLOG" "$SUMMARY"

if [[ "$DO_BUILD" == "1" ]]; then
    echo "==> Configuring..."
    cmake -B "$BUILD_DIR" \
          -DCMAKE_BUILD_TYPE=Release \
          --no-warn-unused-cli -Wno-dev \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
          2>&1 | grep -v "^--" || true

    echo "==> Building ($JOBS jobs)..."
    cmake --build "$BUILD_DIR" -j"$JOBS"
fi

[[ -x "$BINARY" ]] || { echo "ERROR: missing $BINARY" >&2; exit 1; }

echo "==> Priming sudo (powermetrics requires it) ..."
sudo -v

# Keep sudo credentials warm for the duration of the run
( while true; do sudo -n true; sleep 30; done ) &
SUDO_KEEPALIVE=$!
trap 'kill $SUDO_KEEPALIVE 2>/dev/null || true' EXIT

echo "==> Sampling power at ${SAMPLE_INTERVAL_MS}ms intervals ..."
# stderr → PMLOG too, so any powermetrics error is visible in the trace.
# No -n bound: we kill on binary completion via SIGTERM, which powermetrics
# handles cleanly (flushes and exits). SIGINT was unreliable through sudo.
# Sampler: cpu_power for power; thermal for pressure level (replaces older smc).
# If `thermal` is also unrecognized on your macOS, drop it and use just cpu_power.
sudo powermetrics --samplers cpu_power,thermal \
    -i "$SAMPLE_INTERVAL_MS" \
    > "$PMLOG" 2>&1 &
PM_PID=$!

# Let powermetrics initialize (it prints the system summary block first).
sleep 0.5

echo "==> Running binary ($BINARY ${EXTRA_ARGS[*]}) ..."
START_NS=$(python3 -c 'import time; print(int(time.time_ns()))')
"$BINARY" "${EXTRA_ARGS[@]}" > "$LOG" 2>&1
END_NS=$(python3 -c 'import time; print(int(time.time_ns()))')

# Brief tail so a final sample lands after the binary finishes.
sleep 0.2

# Send SIGTERM to powermetrics by name (not via $PM_PID, which is the sudo
# wrapper and may not propagate signals). powermetrics flushes on SIGTERM.
sudo pkill -TERM -x powermetrics 2>/dev/null || true
wait $PM_PID 2>/dev/null || true

RUNTIME_S=$(python3 -c "print(f'{($END_NS - $START_NS) / 1e9:.3f}')")

echo "==> Parsing -> $SUMMARY"
python3 - "$LOG" "$PMLOG" "$SUMMARY" "$RUNTIME_S" <<'PY'
import sys, re
from collections import Counter

log_path, pmlog_path, out_path, runtime_s = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])

out = []
out.append("=== Run info ===")
with open(log_path) as f:
    for ln in f:
        s = ln.rstrip()
        if any(k in s for k in ("GFLOPS", "profile runs", "Threads:", "iters=")):
            if s.strip() and "==>" not in s:
                out.append(s)
out.append("")
out.append(f"  runtime           : {runtime_s:.3f} s")

# powermetrics format on Apple Silicon (cpu_power sampler):
#   E-Cluster Power: NNN mW
#   P-Cluster Power: NNN mW         (or P0-Cluster Power / P1-Cluster Power on multi-cluster SoCs)
#   GPU Power:       NNN mW
#   ANE Power:       NNN mW
#   Combined Power (CPU + GPU + ANE): NNN mW
# smc sampler:
#   Current pressure level: Nominal | Fair | Serious | Critical | ...
p_pat     = re.compile(r"^\s*P\d?-?Cluster Power:\s+(\d+)\s*mW",      re.IGNORECASE)
e_pat     = re.compile(r"^\s*E-Cluster Power:\s+(\d+)\s*mW",          re.IGNORECASE)
gpu_pat   = re.compile(r"^\s*GPU Power:\s+(\d+)\s*mW",                re.IGNORECASE)
comb_pat  = re.compile(r"^\s*Combined Power.*:\s+(\d+)\s*mW",         re.IGNORECASE)
therm_pat = re.compile(r"^\s*Current pressure level:\s+(\w+)",        re.IGNORECASE)

p_samples, e_samples, gpu_samples, comb_samples = [], [], [], []
thermal = []

# A single sample window may report P0 and P1 separately; sum them per window.
sample_p_sum = None
def flush_sample():
    global sample_p_sum
    if sample_p_sum is not None:
        p_samples.append(sample_p_sum)
        sample_p_sum = None

with open(pmlog_path, errors="replace") as f:
    for ln in f:
        if ln.startswith("*** Sampled"):
            flush_sample()
            continue
        m = p_pat.match(ln)
        if m:
            sample_p_sum = (sample_p_sum or 0) + int(m.group(1))
            continue
        m = e_pat.match(ln);    e_samples.append(int(m.group(1)))   if m else None
        m = gpu_pat.match(ln);  gpu_samples.append(int(m.group(1))) if m else None
        m = comb_pat.match(ln); comb_samples.append(int(m.group(1))) if m else None
        m = therm_pat.match(ln); thermal.append(m.group(1))         if m else None
flush_sample()

def avg_w(xs):
    return (sum(xs) / len(xs) / 1000.0) if xs else 0.0

avg_p_w   = avg_w(p_samples)
avg_e_w   = avg_w(e_samples)
avg_gpu_w = avg_w(gpu_samples)
avg_c_w   = avg_w(comb_samples)

if comb_samples:
    energy_j   = avg_c_w * runtime_s
    power_basis = "Combined (CPU + GPU + ANE)"
else:
    energy_j   = (avg_p_w + avg_e_w) * runtime_s
    power_basis = "P-cluster + E-cluster"

n_samples = max(len(p_samples), len(e_samples), len(comb_samples))
out.append(f"  power samples     : {n_samples}")
out.append("")
out.append("=== Power averages ===")
out.append(f"  P-cluster avg     : {avg_p_w:>7.3f} W")
out.append(f"  E-cluster avg     : {avg_e_w:>7.3f} W")
if avg_gpu_w > 0:
    out.append(f"  GPU avg           : {avg_gpu_w:>7.3f} W")
out.append(f"  Combined avg      : {avg_c_w:>7.3f} W")
out.append("")
out.append("=== Energy ===")
out.append(f"  basis             : {power_basis}")
out.append(f"  runtime           : {runtime_s:.3f} s")
out.append(f"  energy            : {energy_j:>7.2f} J")

# Try to compute J/GFLOP from `profile()` output lines (M x K x N iters=N)
total_gflops_done = 0.0
peak_gflops = 0.0
size_iter_pat = re.compile(r"(\d+)x(\d+)x(\d+)\s+iters=(\d+)")
gflops_pat    = re.compile(r"([\d.]+)\s*GFLOPS")
with open(log_path) as f:
    for ln in f:
        m = size_iter_pat.search(ln)
        if m:
            M, K, N, it = (int(m.group(i)) for i in (1, 2, 3, 4))
            total_gflops_done += 2.0 * M * K * N * it / 1e9
        m = gflops_pat.search(ln)
        if m:
            try: peak_gflops = max(peak_gflops, float(m.group(1)))
            except ValueError: pass

if total_gflops_done > 0 and energy_j > 0:
    out.append(f"  work done         : {total_gflops_done:>7.1f} GFLOPs")
    out.append(f"  energy / GFLOP    : {energy_j / total_gflops_done:>7.4f} J/GFLOP")
    avg_perf = total_gflops_done / runtime_s
    if avg_c_w > 0:
        out.append(f"  perf / W (combined): {avg_perf / avg_c_w:>7.1f} GFLOPS/W (avg)")
elif peak_gflops > 0 and avg_c_w > 0:
    out.append(f"  peak GFLOPS       : {peak_gflops:.1f}")
    out.append(f"  perf / W (combined): {peak_gflops / avg_c_w:>7.1f} GFLOPS/W (peak)")

if thermal:
    out.append("")
    out.append("=== Thermal pressure (smc sampler) ===")
    for state, cnt in Counter(thermal).most_common():
        out.append(f"  {state:<12}  {cnt} samples")

with open(out_path, "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
PY

echo "==> Summary: $SUMMARY"
echo "==> Log:     $LOG"
echo "==> Raw:     $PMLOG"
