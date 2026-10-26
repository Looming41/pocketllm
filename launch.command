#!/usr/bin/env bash
# macOS double-click launcher (Finder runs .command files in Terminal).
# First run: right-click > Open, because files from a USB drive are quarantined by Gatekeeper.
cd "$(dirname "$0")"
xattr -dr com.apple.quarantine bin 2>/dev/null
exec ./launch.sh
