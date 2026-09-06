#!/usr/bin/env bash
# bench/grand_benchmark.sh — run every benchmark this repo has, in one pass,
# into one dated directory.
#
# Collects:
#   1. environment      hardware, compiler, and the exact versions of every
#                       third-party thing measured
#   2. bench_compare    our NEON + 4 SME kernels vs Accelerate vs OpenBLAS vs
#                       KleidiAI, plus the KleidiAI packing breakdown
#   3. maxmul_vs_kleidiai
#                       the layered comparison: hot micro-kernel, prepacked full
#                       GEMM, end-to-end, KleidiAI panel-blocked, and both
#                       llama.cpp/ggml CPU paths.  Emits the machine-readable CSV
#                       everything else is summarised from.
#   4. python           NumPy and PyTorch, single-thread
#   5. power            average CPU power and total energy (OPT-IN: needs sudo)
#   6. SUMMARY.md       the consolidated tables
#
# Usage:
#   ./bench/grand_benchmark.sh                 # everything except power
#   ./bench/grand_benchmark.sh --with-power    # everything, plus energy (sudo)
#   ./bench/grand_benchmark.sh --power-only    # ONLY the energy step, reusing an
#                                              # existing results directory
#   ./bench/grand_benchmark.sh --out DIR       # override output directory
#
# Default output: bench/results/<YYYY-MM-DD>/

set -eo pipefail
cd "$(dirname "$0")/.."

WITH_POWER=0
POWER_ONLY=0
OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-power) WITH_POWER=1; shift ;;
        --power-only) WITH_POWER=1; POWER_ONLY=1; shift ;;
        --out)        OUT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$OUT" ]] || OUT="bench/results/$(date +%F)"

if [[ "$POWER_ONLY" -eq 1 ]]; then
    # Reuse an existing run: measure energy and refresh the summary, nothing else.
    [[ -d "$OUT" ]] || { echo "error: $OUT does not exist; run without --power-only first." >&2; exit 1; }
    [[ -f "$OUT/maxmul_vs_kleidiai.csv" ]] || {
        echo "error: $OUT has no maxmul_vs_kleidiai.csv to summarise." >&2; exit 1; }
fi
mkdir -p "$OUT"

BUILD_DIR="cmake-build-release"
JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    for p in /opt/anaconda3/bin/python python3; do
        if command -v "$p" &>/dev/null && "$p" -c "import numpy, torch" 2>/dev/null; then
            PYTHON="$p"; break
        fi
    done
fi

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. environment
# Skipped in --power-only mode: environment.txt belongs to the run being
# reused, and rewriting it would stamp it with the wrong time.
if [[ "$POWER_ONLY" -eq 0 ]]; then
say "Environment"
{
    echo "date                : $(date -u +%Y-%m-%dT%H:%M:%SZ) (UTC)"
    echo "host                : $(sysctl -n machdep.cpu.brand_string)"
    echo "P-cores / E-cores   : $(sysctl -n hw.perflevel0.physicalcpu) / $(sysctl -n hw.perflevel1.physicalcpu)"
    echo "P-core L1d / L2     : $(( $(sysctl -n hw.perflevel0.l1dcachesize) / 1024 )) KiB / $(( $(sysctl -n hw.perflevel0.l2cachesize) / 1048576 )) MiB"
    echo "RAM                 : $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
    echo "SME / SME2          : $(sysctl -n hw.optional.arm.FEAT_SME) / $(sysctl -n hw.optional.arm.FEAT_SME2)"
    echo "OS                  : $(sysctl -n kern.ostype) $(sysctl -n kern.osrelease) ($(sw_vers -productVersion 2>/dev/null))"
    echo "compiler            : $(/opt/homebrew/opt/llvm/bin/clang++ --version | head -1)"
    echo "cmake               : $(cmake --version | head -1)"
    echo "MaxMulSK rev        : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)$( git diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo "KleidiAI (direct)   : $(sed -n 's/.*KLEIDIAI_PINNED_TAG *"\([^"]*\)".*/\1/p' CMakeLists.txt | head -1) (FetchContent, ours)"
    LLAMA_DIR=$(sed -n 's/^LLAMA_CPP_DIR:PATH=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | head -1)
    if [[ -n "$LLAMA_DIR" && -d "$LLAMA_DIR" ]]; then
        echo "llama.cpp / ggml    : $(git -C "$LLAMA_DIR" rev-parse --short HEAD 2>/dev/null)"
        echo "KleidiAI (in ggml)  : $(sed -n 's/.*KLEIDIAI_COMMIT_TAG *"\([^"]*\)".*/\1/p' "$LLAMA_DIR/ggml/src/ggml-cpu/CMakeLists.txt" 2>/dev/null | head -1) (ggml pins its own)"
        echo "ggml CPU flags      : Metal/Accelerate/BLAS OFF, CPU_KLEIDIAI ON, OpenMP OFF"
    fi
    echo "python              : ${PYTHON:-<none with numpy+torch>}"
    [[ -n "$PYTHON" ]] && echo "                      $("$PYTHON" -c 'import numpy,torch;print("numpy",numpy.__version__,"torch",torch.__version__)')"
    echo "OpenBLAS            : $(ls /opt/homebrew/opt/openblas/lib/libopenblas.dylib 2>/dev/null || echo 'not found')"
    echo "threading           : single-thread everywhere (OMP=1, VECLIB=1, OPENBLAS=1, ggml n_threads=1)"
} | tee "$OUT/environment.txt"
fi

# ---------------------------------------------------------------- 2. build
if [[ "$POWER_ONLY" -eq 0 ]]; then
say "Building all targets ($JOBS jobs)"
cmake --build "$BUILD_DIR" -j"$JOBS" > "$OUT/build.log" 2>&1 || {
    echo "build failed; see $OUT/build.log" >&2; exit 1; }
echo "ok"
fi

# ---------------------------------------------------------------- 3. bench_compare
if [[ "$POWER_ONLY" -eq 0 ]]; then
say "bench_compare — our kernels vs Accelerate / OpenBLAS / KleidiAI"
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "./$BUILD_DIR/bench_compare" > "$OUT/bench_compare.txt" 2>&1
echo "-> $OUT/bench_compare.txt"
fi

# ---------------------------------------------------------------- 4. layered
if [[ "$POWER_ONLY" -eq 0 ]]; then
say "benchmark_maxmul_vs_kleidiai — hot / prepacked / end-to-end / panel / ggml"
echo "    (this is the long one, several minutes)"
"./$BUILD_DIR/benchmark_maxmul_vs_kleidiai" "$OUT/maxmul_vs_kleidiai.csv" \
    > "$OUT/maxmul_vs_kleidiai.txt" 2>&1
echo "-> $OUT/maxmul_vs_kleidiai.{txt,csv}"
fi

# ---------------------------------------------------------------- 5. python
if [[ "$POWER_ONLY" -eq 1 ]]; then
    say "Steps 1-5 SKIPPED (--power-only): reusing $OUT"
elif [[ -n "$PYTHON" ]]; then
    say "NumPy / PyTorch baselines (single-thread)"
    "$PYTHON" bench/bench_python.py > "$OUT/python_baselines.txt" 2>&1 || true
    echo "-> $OUT/python_baselines.txt"
else
    say "NumPy / PyTorch — SKIPPED (no interpreter with both installed)"
    echo "no python with numpy+torch found; re-run with PYTHON=/path/to/python" \
        > "$OUT/python_baselines.txt"
fi

# ---------------------------------------------------------------- 6. power
if [[ "$WITH_POWER" -eq 1 ]]; then
    say "Energy — per implementation, per shape (delegates to bench/energy_bench.sh)"
    ./bench/energy_bench.sh --out "$OUT" 2>&1 | tee "$OUT/energy_run.log"
    echo "-> $OUT/energy.{txt,csv}"
else
    say "Energy — SKIPPED (powermetrics needs sudo)"
    {
        echo "Skipped. powermetrics requires root, so this step is opt-in."
        echo
        echo "Run just the energy step against this directory:"
        echo "    ./bench/grand_benchmark.sh --power-only"
        echo "or directly:"
        echo "    ./bench/energy_bench.sh"
    } > "$OUT/energy.txt"
fi

# ---------------------------------------------------------------- 7. summary
say "Aggregating"
"${PYTHON:-python3}" bench/grand_summary.py "$OUT" > "$OUT/SUMMARY.md"
echo "-> $OUT/SUMMARY.md"

say "Done — $OUT"
ls -1 "$OUT"
