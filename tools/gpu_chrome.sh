#!/usr/bin/env bash
# Private hardware-GPU Chrome for the tsq render daemon (CDP :9223, own profile), so that renders never share a
# browser with anything else on the machine: headless + ANGLE/Vulkan + --disable-vulkan-surface gives hardware
# WebGPU on Linux with an NVIDIA card (checked: adapter vendor "nvidia", Chromium 152). CHROME_BIN names the binary.
#   tools/gpu_chrome.sh         start (idempotent)
#   tools/gpu_chrome.sh stop    stop
set -euo pipefail
# TSQ_CHROME_PORT / TSQ_CHROME_PROFILE select a second instance (town captures use :9224 so that they do not
# share a GPU process - and its queue - with the asset render daemon).
PORT=${TSQ_CHROME_PORT:-9223}
# profiles and logs live on the normal disk (.scratch/, not served): /tmp is a quota'd RAM disk
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROF=${TSQ_CHROME_PROFILE:-$ROOT/.scratch/chrome/tsq-chrome-$PORT}
mkdir -p "$ROOT/.scratch/chrome"
# the instance is whoever serves CDP on $PORT (whatever profile it was started with) or runs on $PROF
pids() { for p in $(pgrep -x chromium || true); do tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -qx -e "--remote-debugging-port=$PORT" -e "--user-data-dir=$PROF" && echo $p; done; return 0; }
if [ "${1:-}" = stop ]; then
  for p in $(pids); do kill $p 2>/dev/null || true; done
  echo "tsq GPU Chrome stopped"; exit 0
fi
if curl -sf "http://localhost:$PORT/json/version" > /dev/null; then
  echo "tsq GPU Chrome already running on port $PORT"; exit 0
fi
CHROME_BIN="${CHROME_BIN:-$(command -v chromium || command -v google-chrome || command -v google-chrome-stable || command -v chromium-browser)}"
env -u DISPLAY nohup setsid "$CHROME_BIN" --headless=new \
  --remote-debugging-port=$PORT \
  --no-first-run --no-default-browser-check \
  --use-angle=vulkan --enable-features=Vulkan,WebGPU --enable-unsafe-webgpu --ignore-gpu-blocklist \
  --disable-vulkan-surface \
  --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows \
  --window-size=1600,1000 \
  --user-data-dir=$PROF \
  --disk-cache-size=67108864 --media-cache-size=1 \
  about:blank > "$ROOT/.scratch/chrome/tsq-chrome-$PORT.log" 2>&1 &
for i in $(seq 1 30); do
  curl -sf "http://localhost:$PORT/json/version" > /dev/null && break
  sleep 0.5
done
echo "tsq GPU Chrome up: http://localhost:$PORT (headless, Vulkan)"
