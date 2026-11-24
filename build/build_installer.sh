#!/bin/sh
# Builds dist/BlackboxMask-<version>-Setup.exe on Linux (cross-build, like package_windows.py).
# Requires NSIS 3.x: apt install nsis  (Debian/Ubuntu)  /  dnf install mingw32-nsis (Fedora)
set -eu
cd "$(dirname "$0")/.."
VERSION="${VERSION:-0.1.2}"
mkdir -p dist
makensis -INPUTCHARSET UTF8 -DVERSION="$VERSION" build/installer.nsi
echo "Installer created: dist/BlackboxMask-$VERSION-Setup.exe"
