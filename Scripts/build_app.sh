#!/usr/bin/env bash
# build_app.sh — Assemble Qwegn.app (Info.plist, AppIcon.icns, kernel*.metallib, server UI bundle).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="$ROOT/.build/release"
IDENTITY="${APPLE_DEVELOPER_ID_APP:-${QWG_SIGN_IDENTITY:--}}"
SKIP_BUILD=0
VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_app.sh: --output needs a directory" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --identity)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_app.sh: --identity needs a value" >&2; exit 2; }
      IDENTITY="$2"
      shift 2
      ;;
    --version)
      [[ $# -ge 2 && -n "$2" ]] || { echo "build_app.sh: --version needs a value" >&2; exit 2; }
      VERSION="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    *)
      echo "Usage: $0 [--output DIR] [--identity ID] [--version X.Y.Z] [--skip-build]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/.*static let current: String = "\([^"]*\)".*/\1/p' Sources/Foundation/QWCore/QWVersion.swift)"
fi
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "build_app.sh: invalid version '$VERSION'" >&2; exit 1; }

if [[ $SKIP_BUILD -eq 0 ]]; then
  ./Scripts/build_metallib.sh >/dev/null
  swift build -c release --product qwg
  swift build -c release --product qwg-menu
  swift build -c release --product qwg-smc-helper
fi

BIN_DIR="$(swift build -c release --show-bin-path)"
[[ -x "$BIN_DIR/qwg" ]] || { echo "build_app.sh: missing $BIN_DIR/qwg" >&2; exit 1; }
[[ -x "$BIN_DIR/qwg-menu" ]] || { echo "build_app.sh: missing $BIN_DIR/qwg-menu" >&2; exit 1; }
[[ -x "$BIN_DIR/qwg-smc-helper" ]] || { echo "build_app.sh: missing $BIN_DIR/qwg-smc-helper" >&2; exit 1; }
[[ -f "Sources/QWMetal/kernel.metallib" && -f "Sources/QWMetal/kernel_std.metallib" && -f "Sources/QWMetal/kernel_m3.metallib" && -f "Sources/QWMetal/kernel_m3_std.metallib" ]] || {
  echo "build_app.sh: missing Sources/QWMetal/kernel*.metallib" >&2
  exit 1
}

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
APP="$OUT_DIR/Qwegn.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Library/LaunchDaemons"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleDisplayName</key><string>Qwegn</string>
    <key>CFBundleExecutable</key><string>qwg-menu</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleIconName</key><string>AppIcon</string>
    <key>CFBundleIdentifier</key><string>ai.qwegn.menu</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>Qwegn</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key><string>26.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
/usr/bin/plutil -lint "$APP/Contents/Info.plist" >/dev/null

ICON_WORK="$(mktemp -d)"
trap 'rm -rf "$ICON_WORK"' EXIT
swift "$ROOT/Scripts/gen_app_icon.swift" "$ICON_WORK/AppIcon.iconset"
/usr/bin/iconutil -c icns "$ICON_WORK/AppIcon.iconset" -o "$APP/Contents/Resources/AppIcon.icns"

cp "$BIN_DIR/qwg-menu" "$APP/Contents/MacOS/qwg-menu"
chmod 755 "$APP/Contents/MacOS/qwg-menu"
cp "$BIN_DIR/qwg" "$APP/Contents/MacOS/qwg"
chmod 755 "$APP/Contents/MacOS/qwg"
cp "$BIN_DIR/qwg-smc-helper" "$APP/Contents/MacOS/qwg-smc-helper"
chmod 755 "$APP/Contents/MacOS/qwg-smc-helper"
cp "Sources/QWMetal/kernel.metallib" "Sources/QWMetal/kernel_std.metallib" \
  "Sources/QWMetal/kernel_m3.metallib" "Sources/QWMetal/kernel_m3_std.metallib" \
  "$APP/Contents/Resources/"

cat > "$APP/Contents/Library/LaunchDaemons/ai.qwegn.smc-helper.plist" <<DPLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.qwegn.smc-helper</string>
    <key>BundleProgram</key>
    <string>Contents/MacOS/qwg-smc-helper</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
DPLIST
/usr/bin/plutil -lint "$APP/Contents/Library/LaunchDaemons/ai.qwegn.smc-helper.plist" >/dev/null

if [[ -d "$BIN_DIR/qwegine_QWServer.bundle" ]]; then
  DEST_BUNDLE="$APP/Contents/Resources/qwegine_QWServer.bundle"
  if [[ -f "$BIN_DIR/qwegine_QWServer.bundle/Contents/Info.plist" ]]; then
    /usr/bin/ditto "$BIN_DIR/qwegine_QWServer.bundle" "$DEST_BUNDLE"
  else
    mkdir -p "$DEST_BUNDLE/Contents/Resources"
    /usr/bin/ditto "$BIN_DIR/qwegine_QWServer.bundle" "$DEST_BUNDLE/Contents/Resources"
    rm -rf "$DEST_BUNDLE/Contents/Resources/Info.plist" "$DEST_BUNDLE/Contents/Resources/_CodeSignature"
    cat > "$DEST_BUNDLE/Contents/Info.plist" <<BPLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleIdentifier</key><string>qwegine.QWServer.resources</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>qwegine_QWServer</string>
    <key>CFBundlePackageType</key><string>BNDL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key><string>26.0</string>
</dict>
</plist>
BPLIST
    /usr/bin/plutil -lint "$DEST_BUNDLE/Contents/Info.plist" >/dev/null
  fi
fi

SIGN_FLAGS=(--force --sign "$IDENTITY")
if [[ "$IDENTITY" == "-" ]]; then
  SIGN_FLAGS+=(--timestamp=none)
else
  SIGN_FLAGS+=(--options runtime --timestamp)
fi
/usr/bin/codesign "${SIGN_FLAGS[@]}" "$APP/Contents/MacOS/qwg"
/usr/bin/codesign "${SIGN_FLAGS[@]}" "$APP/Contents/MacOS/qwg-smc-helper"
if [[ -d "$APP/Contents/Resources/qwegine_QWServer.bundle" ]]; then
  /usr/bin/codesign "${SIGN_FLAGS[@]}" "$APP/Contents/Resources/qwegine_QWServer.bundle"
fi
/usr/bin/codesign "${SIGN_FLAGS[@]}" "$APP"
/usr/bin/codesign --verify --deep --strict "$APP"
echo "Built: $APP"
