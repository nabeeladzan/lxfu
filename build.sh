#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CONFIG}" "$@"
cmake --build "${BUILD_DIR}" --config "${CONFIG}"
