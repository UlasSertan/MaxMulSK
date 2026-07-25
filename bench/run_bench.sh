#!/usr/bin/env bash
set -euo pipefail

# Use CLion-bundled cmake if not on PATH
if ! command -v cmake &>/dev/null; then
    # Fall back to a CLion-bundled cmake if one is installed
    for _c in /Applications/CLion*.app/Contents/bin/cmake/mac/*/bin \
              "$HOME"/Applications/CLion*.app/Contents/bin/cmake/mac/*/bin; do
        [ -d "$_c" ] && export PATH="$_c:$PATH" && break
    done
fi
command -v cmake &>/dev/null || {
    echo "error: cmake not found. Install it with 'brew install cmake'." >&2
    exit 1
}

# Run from repo root regardless of where the script is called from
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/cmake-build-release"
BINARY="$BUILD_DIR/bench_compare"
JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

echo "==> Configuring..."
cmake -B "$BUILD_DIR" \
      -S "$REPO_ROOT" \
      -DCMAKE_BUILD_TYPE=Release \
      --no-warn-unused-cli \
      -Wno-dev \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
      2>&1 | grep -v "^--"

echo "==> Building bench_compare ($JOBS jobs)..."
cmake --build "$BUILD_DIR" --target bench_compare -j"$JOBS"

echo "==> Running C++ benchmark (single-threaded)..."
echo ""
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$BINARY"

echo ""
echo "==> Running Python benchmark (NumPy + PyTorch, single-threaded)..."
echo ""
# Needs an interpreter with numpy and torch. Override with e.g.
#   PYTHON=/opt/anaconda3/bin/python ./bench/run_bench.sh
PYTHON="${PYTHON:-python3}"
if "$PYTHON" -c "import numpy, torch" >/dev/null 2>&1; then
    "$PYTHON" "$REPO_ROOT/bench/bench_python.py"
else
    echo "  [skip] '$PYTHON' does not have both numpy and torch installed."
    echo "         Re-run with: PYTHON=/path/to/python ./bench/run_bench.sh"
fi
