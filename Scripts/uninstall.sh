#!/usr/bin/env bash
# uninstall.sh — Remove Qwegn installed by the macOS .pkg installer.
# Removes /usr/local/bin/qwg, /Applications/Qwegn.app, /usr/local/share/qwegn,
# and forgets the package receipt (ai.qwegn.pkg). User data in ~/.qwg is kept.
set -euo pipefail

ROOT_PREFIX=""
SKIP_PKGUTIL=0
PKG_ID="ai.qwegn.pkg"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 && -n "$2" ]] || { echo "uninstall.sh: --root requires a path" >&2; exit 2; }
      ROOT_PREFIX="${2%/}"
      shift 2
      ;;
    --skip-pkgutil)
      SKIP_PKGUTIL=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--root DIR] [--skip-pkgutil]"
      exit 0
      ;;
    *)
      echo "uninstall.sh: unknown option: $1" >&2
      exit 2
      ;;
  esac
done

APP_PATH="${ROOT_PREFIX}/Applications/Qwegn.app"
CLI_PATH="${ROOT_PREFIX}/usr/local/bin/qwg"
SHARE_PATH="${ROOT_PREFIX}/usr/local/share/qwegn"
VENV_PATH="${HOME}/Library/Application Support/Qwegn"

if [[ -x "${APP_PATH}/Contents/MacOS/qwg-menu" ]]; then
  "${APP_PATH}/Contents/MacOS/qwg-menu" --unregister-login-item >/dev/null 2>&1 || true
fi

if [[ -z "$ROOT_PREFIX" ]]; then
  pkill -TERM -f "${APP_PATH}/Contents/MacOS/qwg-menu" >/dev/null 2>&1 || true
fi

run_priv() {
  if [[ -n "$ROOT_PREFIX" || $EUID -eq 0 ]]; then
    "$@"
  else
    /usr/bin/sudo "$@"
  fi
}

if [[ -e "$CLI_PATH" || -L "$CLI_PATH" ]]; then
  if [[ -w "$(dirname "$CLI_PATH")" ]]; then
    /bin/rm -f "$CLI_PATH"
  else
    run_priv /bin/rm -f "$CLI_PATH"
  fi
fi

if [[ -d "$APP_PATH" || -L "$APP_PATH" ]]; then
  if [[ -w "$(dirname "$APP_PATH")" ]]; then
    /bin/rm -rf "$APP_PATH"
  else
    run_priv /bin/rm -rf "$APP_PATH"
  fi
fi

if [[ -d "$VENV_PATH" && -z "$ROOT_PREFIX" ]]; then
  /bin/rm -rf "$VENV_PATH" || true
fi

if [[ $SKIP_PKGUTIL -eq 0 && -z "$ROOT_PREFIX" ]]; then
  if /usr/sbin/pkgutil --pkgs="$PKG_ID" >/dev/null 2>&1; then
    run_priv /usr/sbin/pkgutil --forget "$PKG_ID" >/dev/null
  fi
fi

if [[ -d "$SHARE_PATH" || -L "$SHARE_PATH" ]]; then
  if [[ -w "$(dirname "$SHARE_PATH")" ]]; then
    /bin/rm -rf "$SHARE_PATH"
  else
    run_priv /bin/rm -rf "$SHARE_PATH"
  fi
fi

echo "Uninstalled Qwegn (removed $CLI_PATH, $APP_PATH, $SHARE_PATH)."
echo "Models, settings, profiles, and caches in ~/.qwg were preserved."
