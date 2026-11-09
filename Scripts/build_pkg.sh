#!/usr/bin/env bash
# build_pkg.sh -- Build Qwegn-<version>-macos-arm64.pkg via pkgbuild + productbuild.
# Installs:
#   /usr/local/bin/qwg
#   /Applications/Qwegn.app (Info.plist + AppIcon.icns + kernel*.metallib)
#   /usr/local/share/qwegn/ (uninstall.sh, quantize scripts, requirements.txt, docs)
#   Product archive HTML screens are pure 7-bit ASCII so Installer.app never misdecodes bytes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="$ROOT/dist"
APP_IDENTITY="${APPLE_DEVELOPER_ID_APP:-${QWG_SIGN_IDENTITY:--}}"
INSTALLER_IDENTITY="${APPLE_DEVELOPER_ID_INSTALLER:-}"
NOTARY_PROFILE="${QWG_NOTARY_PROFILE:-}"
APPLE_ID="${APPLE_ID:-}"
APPLE_TEAM_ID="${APPLE_TEAM_ID:-}"
APPLE_APP_PASSWORD="${APPLE_APP_SPECIFIC_PASSWORD:-}"
SKIP_BUILD=0
VERSION=""
PKG_ID="ai.qwegn.pkg"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_pkg.sh: --output needs a directory" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --app-identity)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_pkg.sh: --app-identity needs a value" >&2; exit 2; }
      APP_IDENTITY="$2"
      shift 2
      ;;
    --installer-identity)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_pkg.sh: --installer-identity needs a value" >&2; exit 2; }
      INSTALLER_IDENTITY="$2"
      shift 2
      ;;
    --notary-profile)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_pkg.sh: --notary-profile needs a value" >&2; exit 2; }
      NOTARY_PROFILE="$2"
      shift 2
      ;;
    --version)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_pkg.sh: --version needs a value" >&2; exit 2; }
      VERSION="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    *)
      echo "Usage: $0 [--output DIR] [--app-identity ID] [--installer-identity ID] [--notary-profile PROFILE] [--version X.Y.Z] [--skip-build]" >&2
      exit 2
      ;;
  esac
done

[[ "$(uname -m)" == "arm64" ]] || { echo "build_pkg.sh: must build on Apple Silicon (arm64)" >&2; exit 1; }

if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/.*static let current: String = "\([^"]*\)".*/\1/p' Sources/Foundation/QWCore/QWVersion.swift)"
fi
VERSION="${VERSION#v}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "build_pkg.sh: invalid version '$VERSION'" >&2; exit 1; }

if [[ $SKIP_BUILD -eq 0 ]]; then
  ./Scripts/build_metallib.sh >/dev/null
  swift build -c release --product qwg
  swift build -c release --product qwg-menu
fi

BIN_DIR="$(swift build -c release --show-bin-path)"
[[ -x "$BIN_DIR/qwg" && -x "$BIN_DIR/qwg-menu" ]] || { echo "build_pkg.sh: release binaries missing in $BIN_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PKGROOT="$WORK/pkgroot"
SCRIPTS_DIR="$WORK/scripts"
RES_DIR="$WORK/resources"
mkdir -p \
  "$PKGROOT/usr/local/bin" \
  "$PKGROOT/Applications" \
  "$PKGROOT/usr/local/share/qwegn/Scripts/quantize" \
  "$PKGROOT/usr/local/share/qwegn/Scripts/oracle" \
  "$SCRIPTS_DIR" \
  "$RES_DIR"

# 1. /Applications/Qwegn.app (with Info.plist, AppIcon.icns, and kernel*.metallib)
./Scripts/build_app.sh --skip-build --version "$VERSION" --identity "$APP_IDENTITY" --output "$PKGROOT/Applications" >/dev/null

# 2. /usr/local/bin/qwg
cp "$BIN_DIR/qwg" "$PKGROOT/usr/local/bin/qwg"
chmod 755 "$PKGROOT/usr/local/bin/qwg"
CLI_SIGN_FLAGS=(--force --sign "$APP_IDENTITY")
if [[ "$APP_IDENTITY" == "-" ]]; then
  CLI_SIGN_FLAGS+=(--timestamp=none)
else
  CLI_SIGN_FLAGS+=(--options runtime --timestamp)
fi
/usr/bin/codesign "${CLI_SIGN_FLAGS[@]}" "$PKGROOT/usr/local/bin/qwg"

# 3. /usr/local/share/qwegn/ support files
cp "$ROOT/Scripts/uninstall.sh" "$PKGROOT/usr/local/share/qwegn/uninstall.sh"
chmod 755 "$PKGROOT/usr/local/share/qwegn/uninstall.sh"
cp "$ROOT/requirements.txt" "$ROOT/LICENSE" "$ROOT/README.md" "$PKGROOT/usr/local/share/qwegn/"
cp "$ROOT/Scripts/convert_lsa_sidecar.py" "$PKGROOT/usr/local/share/qwegn/Scripts/"
cp "$ROOT/Scripts/oracle/emit_pack_identity.py" "$PKGROOT/usr/local/share/qwegn/Scripts/oracle/"
for py in "$ROOT"/Scripts/quantize/*.py; do
  base="$(basename "$py")"
  [[ "$base" == test_* ]] && continue
  cp "$py" "$PKGROOT/usr/local/share/qwegn/Scripts/quantize/$base"
done
if [[ -d "$BIN_DIR/qwegine_QWServer.bundle" ]]; then
  /usr/bin/ditto "$BIN_DIR/qwegine_QWServer.bundle" "$PKGROOT/usr/local/share/qwegn/qwegine_QWServer.bundle"
  /usr/bin/codesign "${CLI_SIGN_FLAGS[@]}" "$PKGROOT/usr/local/share/qwegn/qwegine_QWServer.bundle"
fi

# 4. Installer preinstall / postinstall scripts
cat > "$SCRIPTS_DIR/preinstall" <<'SH'
#!/bin/sh
pkill -TERM -f "/Applications/Qwegn.app/Contents/MacOS/qwg-menu" >/dev/null 2>&1 || true
exit 0
SH

cat > "$SCRIPTS_DIR/postinstall" <<'SH'
#!/bin/sh
chmod 755 "/usr/local/bin/qwg" "/usr/local/share/qwegn/uninstall.sh" 2>/dev/null || true
echo "Qwegn installed (/usr/local/bin/qwg, /Applications/Qwegn.app)."
echo "To remove Qwegn, run: /usr/local/share/qwegn/uninstall.sh"
exit 0
SH
chmod 755 "$SCRIPTS_DIR/preinstall" "$SCRIPTS_DIR/postinstall"

# 5. Component plist (disable bundle relocation so Qwegn.app always lands in /Applications)
/usr/bin/pkgbuild --analyze --root "$PKGROOT" "$WORK/component.plist" >/dev/null
if /usr/libexec/PlistBuddy -c "Print :0:BundleIsRelocatable" "$WORK/component.plist" >/dev/null 2>&1; then
  /usr/libexec/PlistBuddy -c "Set :0:BundleIsRelocatable false" "$WORK/component.plist"
fi

/usr/bin/pkgbuild \
  --root "$PKGROOT" \
  --component-plist "$WORK/component.plist" \
  --scripts "$SCRIPTS_DIR" \
  --identifier "$PKG_ID" \
  --version "$VERSION" \
  --install-location "/" \
  "$WORK/Qwegn-component.pkg" >/dev/null

# 6. Product archive with welcome & conclusion screens (strict 7-bit ASCII + UTF-8 meta)
cat > "$RES_DIR/welcome.html" <<HTML
<!doctype html>
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta charset="utf-8">
</head>
<body style="font-family: -apple-system, Helvetica Neue, sans-serif; font-size: 13px;">
<h3>Qwegn ${VERSION}</h3>
<p>This installer places the following components on your Mac:</p>
<ul>
  <li><code>/usr/local/bin/qwg</code>: command-line engine and server</li>
  <li><code>/Applications/Qwegn.app</code>: menu bar controller and bundled Metal 4 kernels</li>
  <li><code>/usr/local/share/qwegn/</code>: quantizer support files and uninstaller</li>
</ul>
</body>
</html>
HTML

cat > "$RES_DIR/conclusion.html" <<HTML
<!doctype html>
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta charset="utf-8">
</head>
<body style="font-family: -apple-system, Helvetica Neue, sans-serif; font-size: 13px;">
<h3>Qwegn ${VERSION} is installed</h3>
<ul>
  <li>Menu bar app: <code>/Applications/Qwegn.app</code></li>
  <li>CLI: <code>/usr/local/bin/qwg</code></li>
  <li>Python quantizer environment is created automatically on first <code>qwg quantize</code> in <code>~/Library/Application Support/Qwegn/venv/</code> (rebuild anytime with <code>qwg init --python</code>).</li>
</ul>
<p><strong>Uninstall:</strong> run <code>/usr/local/share/qwegn/uninstall.sh</code></p>
</body>
</html>
HTML

if LC_ALL=C grep -n '[^[:print:][:space:]]' "$RES_DIR/welcome.html" "$RES_DIR/conclusion.html" >&2; then
  echo "build_pkg.sh: non-ASCII characters detected in Installer HTML resources" >&2
  exit 1
fi

cat > "$WORK/Distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>Qwegn ${VERSION}</title>
    <options customize="never" require-scripts="false" hostArchitectures="arm64"/>
    <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
    <welcome file="welcome.html"/>
    <conclusion file="conclusion.html"/>
    <choices-outline>
        <line choice="default">
            <line choice="${PKG_ID}"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="${PKG_ID}" visible="false">
        <pkg-ref id="${PKG_ID}"/>
    </choice>
    <pkg-ref id="${PKG_ID}" version="${VERSION}" onConclusion="none">Qwegn-component.pkg</pkg-ref>
</installer-gui-script>
XML

PKG_NAME="Qwegn-${VERSION}-macos-arm64.pkg"
UNSIGNED_PKG="$WORK/${PKG_NAME}.unsigned"
FINAL_PKG="$OUT_DIR/$PKG_NAME"

/usr/bin/productbuild \
  --distribution "$WORK/Distribution.xml" \
  --resources "$RES_DIR" \
  --package-path "$WORK" \
  "$UNSIGNED_PKG" >/dev/null

if [[ -n "$INSTALLER_IDENTITY" ]]; then
  /usr/bin/productsign --sign "$INSTALLER_IDENTITY" --timestamp "$UNSIGNED_PKG" "$FINAL_PKG" >/dev/null
else
  mv "$UNSIGNED_PKG" "$FINAL_PKG"
fi

if [[ -n "$INSTALLER_IDENTITY" ]]; then
  if [[ -n "$NOTARY_PROFILE" ]]; then
    xcrun notarytool submit "$FINAL_PKG" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$FINAL_PKG"
    xcrun stapler validate "$FINAL_PKG"
  elif [[ -n "$APPLE_ID" && -n "$APPLE_TEAM_ID" && -n "$APPLE_APP_PASSWORD" ]]; then
    xcrun notarytool submit "$FINAL_PKG" \
      --apple-id "$APPLE_ID" \
      --team-id "$APPLE_TEAM_ID" \
      --password "$APPLE_APP_PASSWORD" \
      --wait
    xcrun stapler staple "$FINAL_PKG"
    xcrun stapler validate "$FINAL_PKG"
  fi
fi

echo "Built: $FINAL_PKG"
