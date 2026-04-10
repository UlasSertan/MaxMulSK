#!/usr/bin/env bash
set -euo pipefail

# Use CLion-bundled cmake if not on PATH
if ! command -v cmake &>/dev/null; then
    export PATH="/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin:$PATH"
fi

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
/opt/anaconda3/bin/python "$REPO_ROOT/bench/bench_python.py"
