#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"
PREFIX="/usr/local"

for arg in "$@"; do
  case $arg in
    --prefix=*)
      PREFIX="${arg#*=}"
      shift
      ;;
    *)
      printf 'Unknown option: %s
' "$arg" >&2
      exit 1
      ;;
  esac

done

rm -f "${PREFIX}/bin/lxfu"
rm -f "${PREFIX}/lib/security/pam_lxfu.so"
rm -f "/etc/lxfu/lxfu.conf"
rm -f "${PREFIX}/share/lxfu/dino.pt"

if [[ -d "${PREFIX}/lib/lxfu" ]]; then
  rm -rf "${PREFIX}/lib/lxfu"
fi

if [[ -d "${PREFIX}/share/lxfu" ]]; then
  rmdir --ignore-fail-on-non-empty "${PREFIX}/share/lxfu"
fi

if [[ -d "/etc/lxfu" ]]; then
  rmdir --ignore-fail-on-non-empty "/etc/lxfu"
fi
