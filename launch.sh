#!/usr/bin/env bash
# PocketLLM launcher (Linux/macOS). Runs everything from the drive it lives on.
cd "$(dirname "$0")"
export POCKETLLM_ROOT="$PWD"
# a portable Python on the drive (scripts/fetch_python.py) wins over the computer's own
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) TAG=macos-arm64 ;; Darwin-x86_64) TAG=macos-x64 ;;
  Linux-aarch64) TAG=linux-arm64 ;; *) TAG=linux-x64 ;;
esac
PY="$PWD/python/$TAG/bin/python3"
[ -x "$PY" ] || PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "Python 3 is required on this computer (macOS: xcode-select --install)."; exit 1
fi
( sleep 2; (command -v open >/dev/null && open http://127.0.0.1:8080) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8080) ) >/dev/null 2>&1 &
exec "$PY" -m pocketllm serve "$@"
