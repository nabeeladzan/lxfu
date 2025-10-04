#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"

if [[ ! -d "${BUILD_DIR}" ]]; then
    ./build.sh
fi

cmake --install "${BUILD_DIR}" --config "${CONFIG}" "$@"
