#!/usr/bin/env bash
set -euo pipefail

# memory guard:
# - reads currently available RAM (kB) from /proc/meminfo
# - adds a configurable buffer in MB (default: 800)
# - applies that limit through `ulimit -v` (kB)
# - runs the provided command, preserving stdout/stderr

usage() {
  cat <<'EOF'
Usage:
  mg.sh [--buffer-mb N] -- <program> [args...]
  mg.sh <program> [args...]

Examples:
  ./mg.sh ./build/main -i assets/random-11-5.scen -N 20
  ./mg.sh --buffer-mb 1200 -- ./build/main -i assets/random-11-5.scen -N 20

Notes:
  - Default buffer is 800 MB.
  - You can also set MG_BUFFER_MB in the environment.
EOF
}

BUFFER_MB="${MG_BUFFER_MB:-400}"

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--buffer-mb" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Error: --buffer-mb requires a value" >&2
    exit 1
  fi
  BUFFER_MB="$2"
  shift 2
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "Error: no program provided" >&2
  usage
  exit 1
fi

if ! [[ "$BUFFER_MB" =~ ^[0-9]+$ ]]; then
  echo "Error: buffer must be a non-negative integer (MB), got: $BUFFER_MB" >&2
  exit 1
fi

if [[ ! -r /proc/meminfo ]]; then
  echo "Error: /proc/meminfo is not readable; cannot determine available memory" >&2
  exit 1
fi

mem_available_kb="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)"
if [[ -z "$mem_available_kb" || ! "$mem_available_kb" =~ ^[0-9]+$ ]]; then
  echo "Error: could not parse MemAvailable from /proc/meminfo" >&2
  exit 1
fi

buffer_kb=$((BUFFER_MB * 1024))
limit_kb=$((mem_available_kb - buffer_kb))

if (( limit_kb <= 0 )); then
  echo "Error: computed non-positive memory limit" >&2
  exit 1
fi

echo "[mg] MemAvailable: ${mem_available_kb} kB; buffer: ${buffer_kb} kB; ulimit -v: ${limit_kb} kB" >&2

# Run command with a virtual memory cap in a subshell so parent shell limits are unaffected.
(
  ulimit -v "$limit_kb"
  exec "$@"
)
