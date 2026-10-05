import subprocess, re, os
PROF = os.environ.get("TSQ_PERF_PROFILE", "tsq-chrome-9224")
ps = subprocess.run(["ps", "-eo", "pid,rss,args"], capture_output=True, text=True).stdout.splitlines()
mine = [l.split(None, 2) for l in ps if re.search(re.escape(PROF) + r"( |$)", l)]
gpu = [p for p, r, a in mine if "type=gpu-process" in a]
smi = dict(l.replace(" ", "").split(",") for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines() if l)
rend = sorted(int(r) for p, r, a in mine if "type=renderer" in a)
print("VRAM", [smi.get(p) for p in gpu], "renderer_rss_MB", [x // 1024 for x in rend[-2:]], "gpu_proc_rss_MB", [int(r) // 1024 for p, r, a in mine if "type=gpu-process" in a])
