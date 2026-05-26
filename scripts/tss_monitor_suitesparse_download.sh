#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/tss_external/TransSolveStack/datasets/suitesparse_full}"
LOG="$ROOT/download_monitor.log"
STATUS="$ROOT/download_status.txt"
PID_FILE="$ROOT/download.pid"
URLS="$ROOT/metadata/mm_urls.txt"
MM_DIR="$ROOT/MM"

mkdir -p "$ROOT"
{
  echo "monitor_started_at=$(date --iso-8601=seconds)"
  echo "root=$ROOT"
} >> "$LOG"

while true; do
  if pgrep -af "wget .*metadata/mm_urls.txt" >/dev/null 2>&1; then
    sleep 60
    continue
  fi
  break
done

downloaded="$(find "$MM_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
expected="$(wc -l < "$URLS" | tr -d ' ')"
size="$(du -sh "$ROOT" 2>/dev/null | awk '{print $1}')"
finished_at="$(date --iso-8601=seconds)"

{
  echo "finished_at=$finished_at"
  echo "downloaded_files=$downloaded"
  echo "expected_files=$expected"
  echo "size=$size"
  if [ "$downloaded" = "$expected" ]; then
    echo "status=complete"
  else
    echo "status=stopped_incomplete"
  fi
} > "$STATUS"

{
  echo "monitor_finished_at=$finished_at"
  cat "$STATUS"
} >> "$LOG"

if command -v notify-send >/dev/null 2>&1; then
  notify-send "TransSolveStack SuiteSparse download" "$(tail -n 4 "$STATUS" | tr '\n' ' ')" >/dev/null 2>&1 || true
fi

rm -f "$PID_FILE.monitor"
