#!/usr/bin/env bash
# bench/energy_bench.sh — energy per implementation per shape: average power,
# Joules, and J/GFLOP.
#
# Each (implementation, shape) gets its OWN powermetrics session wrapped tightly
# around its workload, so every sample belongs to that workload and no timestamp
# alignment is needed. Energy is avg_power x the workload's own measured wall
# time, not the sampler's capture length.
#
# Usage:
#   ./bench/energy_bench.sh                 # defaults below
#   ./bench/energy_bench.sh --seconds 5     # longer window per point
#   ./bench/energy_bench.sh --out DIR
#
# sudo is required by powermetrics. It is primed once at the start and kept
# warm, so the run does not stop to ask again partway through.

set -eo pipefail
cd "$(dirname "$0")/.."

SECONDS_PER_POINT=3
OUT="bench/results/$(date +%F)"
BUILD_DIR="cmake-build-release"
BIN="$BUILD_DIR/energy_bench"
INTERVAL_MS=100

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seconds) SECONDS_PER_POINT="$2"; shift 2 ;;
        --out)     OUT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"
RAW_DIR="$OUT/energy_raw"; mkdir -p "$RAW_DIR"
CSV="$OUT/energy.csv"
TXT="$OUT/energy.txt"

IMPLS=(sme-4x1 sme-1x4sym sme-1x4acc sme-1x4acckc kleidiai-fullpack kleidiai-panel accelerate)
SHAPES=("1024 1024 1024" "2048 2048 2048" "4096 4096 4096" "128 32768 512")

command -v powermetrics >/dev/null || { echo "error: powermetrics not found" >&2; exit 1; }
if [[ ! -x "$BIN" ]]; then
    echo "==> Building energy_bench"
    cmake --build "$BUILD_DIR" --target energy_bench -j"$(sysctl -n hw.logicalcpu)" >/dev/null
fi

echo "==> powermetrics needs root; priming sudo once so the run does not stop later"
sudo -v
( while true; do sudo -n true 2>/dev/null || exit; sleep 30; done ) & KEEPALIVE=$!
cleanup() { kill "$KEEPALIVE" 2>/dev/null || true; sudo pkill -f "powermetrics --samplers" 2>/dev/null || true; }
trap cleanup EXIT

echo "impl,M,K,N,iters,wall_s,gflop,gflops,cpu_w,pcluster_w,ecluster_w,cpu_j,j_per_gflop,samples" > "$CSV"
printf '%-18s %-18s %9s %9s %9s %9s %12s\n' \
    "implementation" "shape" "GFLOP/s" "CPU W" "P-clu W" "CPU J" "J/GFLOP" | tee "$TXT"
printf '%s\n' "$(printf '%.0s-' {1..92})" | tee -a "$TXT"

for shape in "${SHAPES[@]}"; do
    read -r M K N <<< "$shape"
    for impl in "${IMPLS[@]}"; do
        tag="${impl}_${M}x${K}x${N}"
        raw="$RAW_DIR/$tag.raw"

        sudo powermetrics --samplers cpu_power -i "$INTERVAL_MS" > "$raw" 2>/dev/null &
        PM=$!
        sleep 0.35                                   # let the sampler get going
        RES=$("$BIN" "$impl" "$M" "$K" "$N" "$SECONDS_PER_POINT")
        sudo kill "$PM" 2>/dev/null || true
        wait "$PM" 2>/dev/null || true

        wall=$(sed -n 's/.*wall_s=\([0-9.]*\).*/\1/p' <<< "$RES")
        gflop=$(sed -n 's/.*gflop=\([0-9.]*\).*/\1/p' <<< "$RES")
        gflops=$(sed -n 's/.*gflops=\([0-9.]*\).*/\1/p' <<< "$RES")
        iters=$(sed -n 's/.*iters=\([0-9]*\).*/\1/p' <<< "$RES")

        eval "$(python3 bench/energy_parse.py "$raw" "$wall" | sed 's/^/PM_/')" || true
        cpu_w="${PM_cpu_w:-0}"; pcl="${PM_pcluster_w:-0}"; ecl="${PM_ecluster_w:-0}"
        cpu_j="${PM_cpu_j:-0}";  smp="${PM_samples:-0}"
        jpg=$(python3 -c "print(f'{($cpu_j)/($gflop):.6f}' if float('$gflop')>0 else '0')")

        printf '%-18s %-18s %9.1f %9.3f %9.3f %9.2f %12.5f\n' \
            "$impl" "${M}x${K}x${N}" "$gflops" "$cpu_w" "$pcl" "$cpu_j" "$jpg" | tee -a "$TXT"
        echo "$impl,$M,$K,$N,$iters,$wall,$gflop,$gflops,$cpu_w,$pcl,$ecl,$cpu_j,$jpg,$smp" >> "$CSV"
    done
    echo "" | tee -a "$TXT"
done

echo "==> $CSV" | tee -a "$TXT"
echo "==> raw captures in $RAW_DIR"
