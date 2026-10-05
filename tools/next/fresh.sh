#!/bin/bash
# restart the next branch's capture Chrome (port 9229): a clean GPU process to measure memory from
cd "$(dirname "$0")/../.." || exit 1
export TSQ_CHROME_PORT=9229 TSQ_CHROME_PROFILE="$PWD/.scratch/chrome/tsq-chrome-9229"
tools/gpu_chrome.sh stop >/dev/null 2>&1
tools/gpu_chrome.sh
