# Traps that cost time (consolidated)

## Blender CLI and the toolkit

- Pin `PYTHONHASHSEED=0` with `--python-use-system-env` (set / dict order drives tie-breaks; Blender's glTF
  importer raises on two custom attributes of different width about every second process otherwise).
- `from tk import *` does not export `np`, `bpy`, `sqrt`, `acos`, `exp`...: import them. A local variable named
  `lod` shadows `lod()`. `P` is dict-like, so a param named `items` clashes.
- `place()` overwrites a `loc` already given to a builder; `transform_verts` is local, `apply_transform()` first;
  `instances_to_mesh` ignores the base's transform and `keep_base=False` deletes the base; `box(loc=)` is lost in
  `instances_to_mesh`; `matrix_world` is stale inside a build (use `matrix_basis`).
- `lathe()` and `recalc_normals` flip standalone inner or open shells (close profiles with on-axis poles; an
  inside-out lining cost a sign 14 points); a 4-segment lathe's radius is the flat distance; `shape_extrude` bevels
  spike at acute tips; `surface(thickness=)` even-solidify spikes on folds (`solidify(even=False)`); `densify`
  after `boolean` or on n-gons explodes; `boolean` applies the whole modifier stack (realise bevels first,
  re-declare `uv_*` and `smooth`).
- `sweep(smooth_path=True)` resamples two-point paths to ~33 rings and does not mitre corners; `chain()` cannot hit
  an exact link count (stamp links along `catenary()`); `text()` size is the em size; `mat(scale=)` was inverted
  against the docs for a while; `rough=` is a multiplier; a gain of 0 is treated as unset (use 0.001 or
  `mode="set"`); `noise_mask(lo, hi)` are thresholds.
- `resolve_params` compares choices strictly (4 is not "4"). `sphere()` face order varies between runs. Custom
  split normals are dropped on export.
- The render queue: jobs take 16 s to 6 minutes; `tools/tsq render --timeout 600` and retry; after a failure the
  PNGs on disk are stale (check mtimes). `tools/tsq render` ignores `--params`; `tsq build --name tmp` leaves
  `tmp.*` files in `out/glb/<id>/` that authors must delete. The daemon's `PAGE_VERSION` must match `web/render.js`.
- A shared module saved mid-edit with an undefined name broke six buildings for twenty minutes; two agents editing
  one asset was noticed only by a diff. One owner per file; build the scene from `release/`.
- Blender's new-camera clip is 100 m (a 490 m sky dome vanishes); the system Python has no numpy but Blender's
  bundled interpreter does (`.../4.1/python/bin/python3.11`); PIL and cv2 are absent, ImageMagick is there.

## three.js r186 / WebGPU / TSL

- `vec3(new THREE.Color(...))` compiles to black: pass r, g, b numbers or a `uniform`. Check weathering swatches
  against the intended colour on day one.
- Hand-driven rendering (captures) must tick `nodeFrame` itself or every frame is identical. Transparent sprites
  need an `mrtNode` with a normal-free layout or they write garbage into the AO pass; they failed to compile in the
  cube-probe capture and had to be hidden there. Transparent materials throw during probe capture.
- A fresh Chrome answers `requestAdapter()` with null for its first seconds: wait, or the page falls silently to
  the WebGL 2 backend with synchronous shader linking.
- WebGL 2 fallback: three's async pixel read-back never resolves (read the drawing buffer synchronously after the
  draw); clustered lights assign in a compute pass that WebGL 2 lacks (CPU cluster assignment into a DataTexture,
  24 x 14 x 20, 0.37 ms per frame); the backend never deletes vertex array objects (delete on drop or dropped cells
  stay on the GPU). A WebGL canvas reads back empty through `toDataURL`: use `Page.captureScreenshot`.
- Shader `time` is wall clock: pin it and animation phases or two captures differ by thousands of pixels and no A/B
  means anything. Supersampled captures hide aliasing: capture at pixel ratio 1 when judging shimmer.
- TRAA's resolve material writes alpha 1 unless `blending = NoBlending`. The mirror camera must get
  `coordinateSystem` and `_reversedDepth` before any custom projection. GPU timestamps are summed since the last
  resolve; divide by frames.
- `localStorage` on `about:blank` throws; clear the remembered tier before judging what a fresh visit opens on.
  Remove unavailable `<option>`s from the DOM (a phone's picker shows hidden ones). Switch post effects with a
  uniform, never a reload.
- A bake dir name must match `^_next\w+$` or `?bakedir=` silently loads the live bake; a half-written bake
  directory hangs the tab (wait for both manifests and `pvs.json`).
- One node material per tint/glow combination exhausts the heap (791 -> 77 by carrying tint, glow, roughness and
  opacity as vertex attributes).

## Browsers, GPU and captures

- Each agent on its **own** Chrome port and profile (`TSQ_CHROME_PORT=92xx tools/gpu_chrome.sh`, then
  `BU_NAME=<name> BU_CDP_URL=http://localhost:92xx browser-use ...` on **every** call); a call without `BU_NAME` goes
  to the shared Chrome and navigated another app's tab away for two minutes. Check the tab's URL before `goto_url`.
- One task, one tab, closed after; never close the last tab (Chrome exits); restart capture browsers on a schedule
  (an idle render browser leaked 5.4 GB over six hours; the asset render page leaked 70-95 MiB per job from
  render-target resizing; a VRAM check must look at the process, not a profile path). Capture profiles grow a
  0.5-0.8 GB HTTP cache: `--disk-cache-size=67108864`.
- Try `--headless=new` with `--use-angle=vulkan --enable-features=Vulkan --enable-unsafe-webgpu
  --ignore-gpu-blocklist` first (it gave hardware WebGPU on the first try here); Xvfb if the adapter is null.
  Verify the adapter is the real GPU before measuring.
- CPU throttling slows only the main thread; worker timings stay optimistic. Measure loads cold, paired, more
  than once; the shared GPU is also running the owner's other work - captures crawl when it is busy.
- A session restart kills the static server, the render daemon and the tunnel: curl the public pages first, then
  resume agents from their on-disk state. A background log watcher expires at 30 minutes; rely on the task's exit.
- Capture settle counts: 20 / 16 / 8 frames once a page is known stable (40 / 90 / 50 cost 7-10 minutes a batch).
  Capture helpers must hide the intro veil and add `body.capture`, or frames come back black or veiled.

## Process

- Showing Blender renders as progress misleads ("are those three.js or Blender?"). A gallery link that 404s or a
  `localhost` link wastes a round trip: check every link, give the proxied form.
- "It builds" is not "it looks right": authors open their own sheets every iteration; the lead opens sheets before
  accepting a round; a passing gate hid torn dials and shards.
- A full forced rebuild (45 min) ran on stale released copies of 44 assets because the release step was skipped;
  the build decided staleness from file dates the release preserved. Print how many working files differ from
  their release; check the sidecar's mtime against the release file before trusting "rebuilt".
- `tools/town.py render` without `--out` overwrites the judged set. Engine agents render to their own folders.
- A 50-minute bake was restarted eight minutes in because two more franchise phrases were found after launch:
  finish the whole grep before any long job. Finish the whole text sweep (layout defaults, every `PARAMS` string).
- Debug states leaked into judged renders twice in an hour (a paint-swatch bollard, `dbg` lamp variants, an
  experimental tunnel mouth) - hence the release pin.
- A test bake named `_hdr` was ignored and a whole comparison row was invalid; check the HUD's bake name.
- Judges handed stale cached GLBs scored an old generator; a stale gallery image hid a redesign; the look changed
  under an author mid-round (sky colour) - freeze look constants during asset rounds or stamp the look version.
- The lead relayed every interim note from agents ("bake still running"): relay state changes only.
- `/tmp` hit its quota from capture caches (13 GB) and the owner asked twice: scratch under the project, deleted
  after. `rm $S/*.png` triggered the owner's wildcard rule; `ls assets/*.png` tripped it once more in a careful
  session. Read-only tools may take variables; `rm`/`mv`/`cp` may not.
- When a control "does nothing" on one device: read the session's URL and `extra.fx` in telemetry before touching
  the control (it was yesterday's script under today's page).
- Agent slips the house rules keep catching: `cd ... &&` chains, backups in `/tmp`, stray `custom.glb` in
  `out/glb/<asset>/`. Add "delete what you built for tests" to every author checklist.
- Ask which element before theorising ("the lamp itself, the pool, the streak or the window?"); reproduce at the
  owner's spot, preset and device; then explain in one paragraph; then fix the rule.
- Compaction loses a subagent's final report; ask a resumed agent to restate its result. Harness prompts are
  truncated in logs at ~2,000 characters: keep briefs short and the long parts in files the agent reads.
