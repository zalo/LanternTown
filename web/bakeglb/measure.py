# measure.py - one cold measurement of the benchmark page, driven over the DevTools protocol. Run inside browser-use:
#
#   BENCH_URL='http://localhost:8765/bakeglb/?v=packed' BENCH_RATE=1 BENCH_OUT=results.jsonl \
#   BU_NAME=<name> BU_CDP_URL=http://localhost:<port> browser-use < web/bakeglb/measure.py
#
# BENCH_RATE   CPU throttle (1, 4)          BENCH_OUT   a .jsonl the result is appended to (aggregate.py reads it)
# BENCH_SHOT   prefix for the PNGs of ?shot=1   BENCH_WAIT  seconds to wait for the page (300)
# What it does: browser cache off and emptied, garbage collected, the tab brought to the front (a hidden tab gets no
# animation frames), JS heap and ArrayBuffer memory polled every 0.2 s for the peaks, GPU-process memory from
# nvidia-smi before and after (the Chrome started by tools/gpu_chrome.sh on the port of BU_CDP_URL).
# run inside browser-use. env: BENCH_URL, BENCH_RATE (cpu throttle), BENCH_OUT (jsonl to append), BENCH_SHOT (png prefix), BENCH_WAIT
import os, json, time, subprocess, base64
url = os.environ['BENCH_URL']; rate = float(os.environ.get('BENCH_RATE', '1'))
def gpu_mb():
    try:
        pids = subprocess.run(['pgrep', '-f', 'type=gpu-process.*tsq-chrome-' + os.environ.get('BU_CDP_URL', ':9222').rsplit(':', 1)[1].strip('/')], capture_output=True, text=True).stdout.split()
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        for line in out.splitlines():
            a = [x.strip() for x in line.split(',')]
            if a[0] in pids: return int(a[1])
    except Exception as e:
        return None
    return None
def heap():
    try:
        h = cdp("Runtime.getHeapUsage")
        return h.get('usedSize', 0), h.get('backingStorageSize', 0), h.get('embedderHeapUsedSize', 0)
    except Exception as e:
        return (0, 0, 0)
goto_url('about:blank')
wait(0.5)
try:
    cdp("Network.enable"); cdp("Network.setCacheDisabled", cacheDisabled=True); cdp("Network.clearBrowserCache")
    cdp("HeapProfiler.collectGarbage")
except Exception as e:
    print('cdp setup', e)
wait(1.5)
g0 = gpu_mb(); h0 = heap()
cdp("Emulation.setCPUThrottlingRate", rate=rate)
t0 = time.time()
goto_url(url)
try:
    cdp("Page.bringToFront")
except Exception as e:
    print('front', e)
peak = [0, 0, 0]; done = None
while time.time() - t0 < float(os.environ.get('BENCH_WAIT', '300')):
    wait(0.2)
    h = heap()
    for i in range(3): peak[i] = max(peak[i], h[i])
    try:
        done = js("window.__bench && window.__bench.done ? 1 : 0")
    except Exception as e:
        done = 0
    if done: break
hx = heap()
try:
    cdp("HeapProfiler.collectGarbage")
except Exception as e:
    pass
wait(0.5)
g1 = gpu_mb(); h1 = heap()
cdp("Emulation.setCPUThrottlingRate", rate=1)
res = json.loads(js("JSON.stringify(Object.assign({}, window.__bench || {}, {shots: undefined}))") or '{}')
res['driver'] = {'rate': rate, 'wall_s': round(time.time() - t0, 1), 'gpu_mb_before': g0, 'gpu_mb_after': g1,
    'heap_mb_before': round(h0[0] / 1048576), 'heap_mb_peak': round(peak[0] / 1048576), 'heap_mb_end': round(h1[0] / 1048576),
    'buffers_mb_before': round(h0[1] / 1048576), 'buffers_mb_peak': round(peak[1] / 1048576), 'buffers_mb_end': round(h1[1] / 1048576)}
res['url'] = url
if os.environ.get('BENCH_SHOT'):
    n = int(js("(window.__bench.shots || []).length") or 0)
    for i in range(n):
        d = js("window.__bench.shots[%d]" % i)
        open(os.environ['BENCH_SHOT'] + '_%d.png' % i, 'wb').write(base64.b64decode(d.split(',', 1)[1]))
if os.environ.get('BENCH_OUT'):
    open(os.environ['BENCH_OUT'], 'a').write(json.dumps(res) + '\n')
short = {k: res.get(k) for k in ('v', 'scene', 'form', 'lazy', 'hidden', 'adapter', 'backend', 'marks', 'errors', 'settled', 'spawn', 'level', 'walk', 'driver', 'warnings', 'failed')}
print(json.dumps(short)[:int(os.environ.get('BENCH_PRINT', '2500'))])
