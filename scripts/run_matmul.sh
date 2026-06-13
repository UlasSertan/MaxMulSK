#!/usr/bin/env bash
set -euo pipefail

# Build and run the `matmul` target: head-to-head N=2048 single-thread
# benchmark of our SME 1x4SymZAInOut kernel vs Apple Accelerate.
#
# Usage:
#   ./scripts/run_matmul.sh            # benchmark only
#   ./scripts/run_matmul.sh --verify   # also run naive ijk reference and check both

# Use CLion-bundled cmake if not on PATH
if ! command -v cmake &>/dev/null; then
    export PATH="/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin:$PATH"
fi

# Anchor to repo root so relative paths resolve regardless of caller's cwd
cd "$(dirname "$0")/.."

BUILD_DIR="cmake-build-release"
BINARY="$BUILD_DIR/matmul"
JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

echo "==> Configuring..."
cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      --no-warn-unused-cli \
      -Wno-dev \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
      2>&1 | grep -v "^--"

echo "==> Building matmul ($JOBS jobs)..."
cmake --build "$BUILD_DIR" --target matmul -j"$JOBS"

echo "==> Running..."
echo ""
"$BINARY" "$@"
