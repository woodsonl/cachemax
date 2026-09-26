#!/bin/sh
# Run the menu bar render tests. macOS only (needs swiftc).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
bin=$(mktemp -t cachemax-render-tests)
trap 'rm -f "$bin"' EXIT
swiftc -O -o "$bin" "$here/RenderTests.swift"
"$bin"
