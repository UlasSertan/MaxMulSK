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

# Anchor to repo root so relative paths resolve regardless of caller's cwd
cd "$(dirname "$0")/.."

BUILD_DIR="cmake-build-release"
BINARY="$BUILD_DIR/MatrixLibrary"
JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

echo "==> Configuring..."
cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      --no-warn-unused-cli \
      -Wno-dev \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
      2>&1 | grep -v "^--"

echo "==> Building ($JOBS jobs)..."
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "==> Running..."
echo ""
"$BINARY"
