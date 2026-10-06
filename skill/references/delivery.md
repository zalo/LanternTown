# Delivery: bytes, formats, loaders, publishing

## Order of work

Lossless first, measured on the desktop over localhost and an emulated link; lossy only after the scene is judged
right **and the owner lifts the hold** ("Save the gltf-transform stuff to the end; it's not worth any lossy
tradeoffs yet" ... a day later "I think we're pretty far into glb optimization already; let's explore ktx2").

- gzip on the server for models and JSON: Studio models at spawn 156 -> 76 MB. `Cache-Control: immutable` on
  stamped (`?v=`) files; stamp every script URL by mtime (a phone running yesterday's `main.js` under today's page
  produced ghost bugs twice - `Cache-Control: no-cache` behind a proxy was not enough).
- meshopt without quantisation plus one bundle for the far models: 471 -> 106 requests, 72 MB, pixel-identical;
  first frame 54.5 -> 40.1 s on an emulated 50 Mbit link. Not worth it: vertex and index reordering (+16-19 % after
  gzip). Not safe: joining primitives across materials (changes draw order where surfaces share a depth).
- Frame time did not change from any of these; parsing is 2-4 % of load. Texture upload (~3.5 s), uniform setup
  (~2.2 s) and pipeline build (~1.1 s) dominate the dynamic page.
- **Do not coarsen the position grid**: Draco on a 2 mm instead of 1 mm grid passed a hole check and made surfaces
  modelled a millimetre apart coplanar all over the scene. The owner: "No on the quantization, it dramatically
  increases depth fighting." A crack check is not a depth check; judge lossy geometry by walking it.

## Textures

- KTX2/UASTC offline halves texture GPU memory (High 988 -> ~500 MiB) but grows the download (118 -> 185 MB) and is
  tied to the bake (~3 min re-encode each time); ETC1S smears mortar and breaks normal maps - rejected.
- **Ship WebP, compress on the device in workers**: BC7 on desktop (a 240-line single-mode encoder), 4x4 ASTC on
  phones (5x5 loses 4.8 dB and worsens normals; LDR, one partition, dual-plane for the relief alpha and roughness).
  iPhone, Phone preset: 48 layers encoded in 0.4 s, swapped at 1.8 s, 33 -> 4 MB of textures, 60 fps. PSNR against
  uncompressed 39-45 dB; relief height in alpha has the larger worst-case error - check that channel separately.
  An iPhone reports both BC7 and ASTC: force each path (`tcforce=astc`) and print the chosen format on screen;
  the ASTC path had silently never run until telemetry showed it. Keep the uncompressed path as fallback and leave
  review renders uncompressed. Workers `min(sharp ? 3 : 2, max(1, (cores >> 1) - 1))`, 1 under 2 GB.
- Replacing an array texture at run time: shrink it to 1x1x1 before disposing, or the renderer re-creates it for
  meshes that still reference it (a 67 MB array came back after loading).

## GLB exports (`tools/bake2glb.mjs`, `tools/masterglb.mjs`)

- **Exact** (`--mode exact`): the bake's meshopt streams verbatim, custom vertex layout, texture arrays as embedded
  files; what the live page loads per cell and level, and one 250 MB master file with 287 levels for the lazy
  loader (fine 48 %, near 19 %, textures 19 %; the first-frame block of 15.8 MB is laid first in the BIN).
- **Portable** (`--mode portable`): opens in Blender 4.1 and three.js r186 - `KHR_materials_unlit` + `COLOR_0` for
  the baked light, Draco, WebP, no meshopt or KTX2 (Blender refuses them); extensions chosen by an **intersection
  table** made by writing one tiny file per candidate and importing it in both programs. Measured choices:
  `COLOR_0` u8 (geometry -37 %, worst pixel 18/255 on 0.001 % of the frame) over u16; Draco 7.5 MB vs quantised
  19.5 (9.8 gz) vs float 21.6 (10.1 gz) for the lowest level; LOD chains per `region:2` (25 chains, 329 draws at
  spawn) instead of per cell (113, 677); WebP q80, 512 px for the small files, 1024 for the finest; the sky as an
  unlit rounded-box mesh fitted to the scene (no glTF extension gives an environment both importers show; bounds
  grow ~1.1x, parallax 18.6 degrees max over a 49 m walk) with `scenes[0].extras` naming the panorama so an app can
  draw it as a true background. Blender imports the 52 MB finest level in 26 s at 9.4 GB RAM; set the camera clip
  to 1000 m for the dome; parent lower levels under `LOD1_near` / `LOD2_mid` / `LOD3_lo` empties.
- Sizes to expect: finest level 52-55 MB (8 M tris), near 31 MB (3.1 M), middle 12-13 MB, lowest 6 MB, all levels
  87-91 MB; byte shares connectivity + positions 33 %, UVs 25 %, vertex light 23 %, textures 19 %. The owner found
  the middle level "too crunched" for a phone and the near level "passable"; the two-level stream file (finest +
  coarsest at 512 px, 55.5 MB) shows the whole scene after 7 MB and swaps in detail by range request.
- Exact vs portable per one-file scene at equal triangles: ~250 MB vs 88 MB - the exact format keeps HDR light,
  layer indices, light direction and 54 MB of sharp and normal textures; say which comparison is meant.
- `MSFT_lod` + `THREE.LOD` buys nothing at run time over the streaming loader (eager: 150 MB before the first frame
  and 450-545 ms hitches; lazy: no triangle or memory budget, selects by distance to the object centre, never frees
  CPU copies). Its value is portability and an upstream test bed: a kit of synthetic cases, real cells, expected
  results and a check page, then two loader tiers (transparent node -> `THREE.LOD`; opt-in lazy per-level buffers or
  range requests on one file; a JSON chunk over 64 KiB costs a second request; three levels in flight beat four;
  when a server answers 200 instead of 206 read the body as a stream). Prepared as patches for three.js, never
  pushed without the owner's word.

## The loader must survive a dead connection (`web/engine/net.js`)

A phone tab backgrounded while loading comes back with its connections dead: responses neither arrive nor fail,
`fetch()` has no timeout, and the scene stays at "streaming 76" forever. `openBytes` / `fetchSafe`: re-ask a request
that stops arriving (STALL 12 s; WAIT 20 s for a first byte, REWAIT 8 s on a retry; GRACE 4 s after the page was
hidden or stood still; a request queued behind others is tolerated 3x), resume with `Range: bytes=<got>-`, restart
on a 200 or a bad `Content-Range` / `Content-Encoding`, give up after 5 tries (never for `base.glb`), retry failed
levels later with backoff and on `online` / `visibilitychange`. Tested with a local relay that kills every
connection mid-load, not only with a throttled link: level freeze went from stuck forever to all levels by 70 s;
undisturbed loads unchanged (1.1 s, zero retries). The owner's symptom turned out to be a bad phone connection, and
the fix was kept because the hang was real.

## Publishing

- A quick tunnel (`cloudflared tunnel --url http://127.0.0.1:8765`) exposes the **whole** project server - demos,
  renders, docs including reference crops, reviews, the source zip, telemetry from strangers. Check what the server
  serves before tunnelling it; the address lives only as long as its process (check with
  `curl -w "%{http_code} %{time_total}"` before handing it out).
- **Small source repo + data as release assets** (`tools/site.py`, `.github/workflows/pages.yml`): the repository
  holds source only (~24 MB, 878 files); `python3 tools/site.py --upload data` writes the site data as
  content-addressed tar parts (big files alone, the rest bucketed by group and sha1(path), 24 MB / 8 buckets) plus an
  index, and syncs one rolling release (skip present, upload new, delete stale); the Pages workflow downloads the
  release, assembles `_site` and deploys in ~1.5 min on every push. A rebake re-uploads ~17 parts (~430 MB); a new
  picture touches one. **Every part must be byte-deterministic**: two wall-clock stamps in the frozen manifest made
  a 2 MB part re-upload on every run until they were derived from source mtimes. GitHub Pages: 1 GB per site,
  100 MB per file (the 250 MB master GLB cannot be hosted), range requests answered with 206, works from a private
  repository with `public: true` on a paid plan. Cloudflare Pages would be 25 MiB per file and 20,000 files.
- Before open-sourcing: grep the publishable tree for the home path, proxy and tunnel hostnames, secrets and
  franchise names (text cast into geometry needs a rebuild and a rebake, so finish the grep before starting the
  bake); `diff -q` vendored engine files against `node_modules/three`; every tool finds Blender the same way
  (`TSQ_BLENDER`, `BLENDER`, `PATH`); a dry `git init` with a throwaway `--git-dir` lists exactly what the ignore
  rules would publish; test the Pages assembly from a clean checkout (`git checkout-index -a --prefix`) and diff it
  against the direct build. Flag any rewording of the owner's own words (the reference brief named a game; the
  description was reworded unasked and offered for reversal).
- A source-zip button on the gallery (`tools/srczip.py`, `/api/source.zip`): generators, toolkit, materials,
  layouts, tools, engines, pages, docs, reviews, the replication prompt; no generated output, no `node_modules`;
  scanned for credentials; `.yml` and `.github` were silently dropped by an extension whitelist until checked -
  open the zip after adding any new top-level directory or file type.
- README with a **Try it** section listing every demo URL; the live link under the images; the gallery's Demos tab
  current throughout.

## What it cost (for planning)

About 32,500 model calls over four days: 67 M output tokens, 218 M cache writes, 2.8 B cache reads; ~$6,900 at
Fable 5.1 rates. The one-day asset blitz was 72 % of the spend; subagents 91 %. Cache writes were the second-largest
line because each author starts cold on a 5-minute cache: fewer, longer-lived authors on the 1-hour cache and an
even harder reading-list bound are the levers.
