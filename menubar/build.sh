#!/bin/sh
# Build cachemax-menubar.app from the single Swift source. macOS only.
# No dependencies beyond the Swift toolchain that ships with Xcode CLT.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
app="$here/cachemax-menubar.app"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS"

cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>cachemax</string>
  <key>CFBundleIdentifier</key><string>dev.cachemax.menubar</string>
  <key>CFBundleExecutable</key><string>cachemax-menubar</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

swiftc -O -o "$app/Contents/MacOS/cachemax-menubar" "$here/CacheMaxMenuBar.swift" \
  -framework AppKit

echo "built $app"
echo "run:  open '$app'"
