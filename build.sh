#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr}"

echo "Configuring build with install prefix: ${INSTALL_PREFIX}"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CONFIG}" -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" "$@"
cmake --build "${BUILD_DIR}" --config "${CONFIG}"
