#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"
PREFIX="/usr"

for arg in "$@"; do
  case $arg in
    --prefix=*)
      PREFIX="${arg#*=}"
      shift
      ;;
    *)
      printf 'Unknown option: %s\n' "$arg" >&2
      exit 1
      ;;
  esac

done

echo "Uninstalling LXFU from ${PREFIX}..."

rm -f "${PREFIX}/bin/lxfu"
rm -f "${PREFIX}/lib/security/pam_lxfu.so"
rm -f "${PREFIX}/share/lxfu/dino.pt"

# Remove config from /etc if PREFIX is /usr, otherwise from PREFIX/etc
if [[ "${PREFIX}" == "/usr" ]]; then
  rm -f "/etc/lxfu/lxfu.conf"
  if [[ -d "/etc/lxfu" ]]; then
    rmdir --ignore-fail-on-non-empty "/etc/lxfu"
  fi
else
  rm -f "${PREFIX}/etc/lxfu/lxfu.conf"
  if [[ -d "${PREFIX}/etc/lxfu" ]]; then
    rmdir --ignore-fail-on-non-empty "${PREFIX}/etc/lxfu"
  fi
fi

if [[ -d "${PREFIX}/lib/lxfu" ]]; then
  rm -rf "${PREFIX}/lib/lxfu"
fi

if [[ -d "${PREFIX}/share/lxfu" ]]; then
  rmdir --ignore-fail-on-non-empty "${PREFIX}/share/lxfu"
fi

echo "Uninstall complete!"
echo ""
echo "User data not removed:"
echo "  ~/.lxfu/ (database)"
echo ""
echo "To remove user data: rm -rf ~/.lxfu"
