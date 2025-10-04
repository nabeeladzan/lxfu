#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr}"

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "Build directory not found. Building with prefix ${INSTALL_PREFIX}..."
    INSTALL_PREFIX="${INSTALL_PREFIX}" ./build.sh
fi

# Check if build was configured for the correct prefix
CURRENT_PREFIX=$(grep CMAKE_INSTALL_PREFIX "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d '=' -f2 || echo "/usr/local")

if [[ "${CURRENT_PREFIX}" != "${INSTALL_PREFIX}" ]]; then
    echo "Reconfiguring build for install prefix: ${INSTALL_PREFIX}"
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" -DCMAKE_BUILD_TYPE="${CONFIG}"
    cmake --build "${BUILD_DIR}" --config "${CONFIG}"
fi

echo "Installing to ${INSTALL_PREFIX}..."
cmake --install "${BUILD_DIR}" --config "${CONFIG}" "$@"

# Ensure runtime linker can locate bundled Torch libs for PAM
if [[ "${INSTALL_PREFIX}" == "/usr" ]]; then
    LIBDIR=$(grep -E '^CMAKE_INSTALL_LIBDIR:PATH=' "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | head -n1 | cut -d'=' -f2)
    LIBDIR=${LIBDIR:-lib}
    LXFU_LIB_PATH="${INSTALL_PREFIX}/${LIBDIR}/lxfu"

    if [[ -d "${LXFU_LIB_PATH}" ]]; then
        LDSO_CONF="/etc/ld.so.conf.d/lxfu.conf"
        if [[ ! -f "${LDSO_CONF}" ]] || ! grep -Fxq "${LXFU_LIB_PATH}" "${LDSO_CONF}"; then
            printf '%s\n' "${LXFU_LIB_PATH}" > "${LDSO_CONF}"
            echo "Updated ${LDSO_CONF} with ${LXFU_LIB_PATH}"
        fi

        if command -v ldconfig >/dev/null 2>&1; then
            ldconfig
            echo "Refreshed dynamic linker cache via ldconfig"
        else
            echo "Warning: ldconfig not available; run it manually to refresh linker cache" >&2
        fi
    fi
fi

# Install systemd unit if systemd is present
SERVICE_SRC="packaging/lxfu-face.service"
SERVICE_DEST="/etc/systemd/system/lxfu-face.service"
if [[ -d "/run/systemd/system" ]]; then
    if [[ -f "${SERVICE_SRC}" ]]; then
        tmp_service=$(mktemp)
        sed "s|@PREFIX@|${INSTALL_PREFIX}|g" "${SERVICE_SRC}" > "${tmp_service}"
        install -D -m 0644 "${tmp_service}" "${SERVICE_DEST}"
        rm -f "${tmp_service}"
        echo "Installed systemd unit to ${SERVICE_DEST} (not enabled)"
        echo "Enable with: sudo systemctl enable --now lxfu-face.service"
    else
        echo "Warning: ${SERVICE_SRC} not found; skipping systemd unit installation" >&2
    fi
else
    echo "Systemd not detected; skipping unit installation"
fi

# Install DBus policy for face service
DBUS_POLICY_SRC="packaging/dev.nabeeladzan.lxfu.conf"
DBUS_POLICY_DEST="/etc/dbus-1/system.d/dev.nabeeladzan.lxfu.conf"
if [[ -f "${DBUS_POLICY_SRC}" ]]; then
    install -D -m 0644 "${DBUS_POLICY_SRC}" "${DBUS_POLICY_DEST}"
    if command -v busctl >/dev/null 2>&1; then
        busctl call org.freedesktop.DBus / org.freedesktop.DBus ReloadConfig >/dev/null 2>&1 || true
    fi
    echo "Installed DBus policy to ${DBUS_POLICY_DEST}"
else
    echo "Warning: ${DBUS_POLICY_SRC} not found; skipping DBus policy install" >&2
fi

echo ""
echo "Installation complete!"
echo "  Binary: ${INSTALL_PREFIX}/bin/lxfu"
echo "  Config: $([ "${INSTALL_PREFIX}" = "/usr" ] && echo "/etc/lxfu/lxfu.conf" || echo "${INSTALL_PREFIX}/etc/lxfu/lxfu.conf")"
echo "  Model: ${INSTALL_PREFIX}/share/lxfu/dino.pt"
echo "  PAM module: ${INSTALL_PREFIX}/lib/security/pam_lxfu.so"
