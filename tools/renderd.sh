#!/usr/bin/env bash
# Start / stop / check the render queue daemon (tools/renderd.py) on the project's private GPU Chrome (CDP :9223).
#   tools/renderd.sh [start|stop|status|restart]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDF="$ROOT/out/queue/renderd.pid"
mkdir -p "$ROOT/out/queue" "$ROOT/logs"
alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
case "${1:-start}" in
  start)
    if alive; then echo "renderd already running (pid $(cat "$PIDF"))"; exit 0; fi
    "$ROOT/tools/gpu_chrome.sh" || exit 1
    cd "$ROOT"
    : > "$ROOT/logs/renderd.log"
    rm -f "$ROOT/out/queue/renderd.lock"
    # supervisor: exit code 3 from the daemon means "Chrome is unhealthy" -> restart Chrome and the daemon; the queue survives
    TSQ_ROOT="$ROOT" BU_NAME=tsq BU_CDP_URL=http://localhost:9223 nohup setsid bash -c '
      while true; do
        browser-use < "$0/tools/renderd.py" >> "$0/logs/renderd.log" 2>&1
        rc=$?
        [ $rc -eq 0 ] && break
        echo "$(date +%T) renderd exited rc=$rc, restarting" >> "$0/logs/renderd.log"
        if [ $rc -eq 3 ]; then "$0/tools/gpu_chrome.sh" stop >> "$0/logs/renderd.log" 2>&1; sleep 2; fi
        "$0/tools/gpu_chrome.sh" >> "$0/logs/renderd.log" 2>&1
        sleep 2
      done' "$ROOT" > /dev/null 2>&1 &
    echo $! > "$PIDF"
    for i in $(seq 1 120); do
      grep -q "renderd up" <(tail -n 5 "$ROOT/logs/renderd.log") 2>/dev/null && { echo "renderd up (pid $(cat "$PIDF"))"; exit 0; }
      alive || { echo "renderd died:"; tail -n 15 "$ROOT/logs/renderd.log"; exit 1; }
      sleep 0.5
    done
    echo "renderd did not come up:"; tail -n 15 "$ROOT/logs/renderd.log"; exit 1 ;;
  stop)
    if alive; then kill -TERM -- "-$(cat "$PIDF")" 2>/dev/null || kill -TERM "$(cat "$PIDF")"; sleep 1; fi
    rm -f "$PIDF"; echo stopped ;;
  restart) "$0" stop; exec "$0" start ;;
  status)
    alive && echo "running (pid $(cat "$PIDF"))" || echo "not running"
    cat "$ROOT/out/queue/daemon.json" 2>/dev/null; echo ;;
esac
