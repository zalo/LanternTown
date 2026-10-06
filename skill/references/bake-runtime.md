# The baked runtime: vertex-colour lighting, cells and levels, streaming, culling

> The pipeline is lighting-agnostic: the bake samples whatever light list and key light the scene declares (lamps,
> sun, moon, sky). Where this file says "lantern", read "emitter"; the ground shadow map follows any light's
> shadow edges, the light-driven splitting any light's falloff.

Why it is fast: almost no draw calls (one mesh per cell and level, one material because every surface samples a
texture **array**), no real-time lights or shadows (the light is in the vertices), a 20-byte vertex, and streaming
that shows a complete coarse scene at once. The owner's reaction to the first baked Low: "works FANTASTICALLY ...
Why is it so fast? Just the fewer draw calls?" - draw calls are part of it; the larger part is that no lighting
happens at run time (418 lights per pixel through clustered lighting -> none).

Tools: `tools/webbake.py` (one command after every `town.py all`; `--jobs`, `--keep-raw`, `--pack-only`, `--pvs`,
`--set key=value`) -> `tools/webbake_bl.py` (Blender: lighting, AO, chunking; lighting workers are forks of one
Blender, CPU only, no VRAM) -> `tools/lsvb_refit.py` -> `node tools/webbake_pack.mjs` (meshopt simplify and encode)
-> WebP pages -> `tools/aomap.py` (ground maps) -> `node tools/bake2glb.mjs --live --verify --drop-packed` ->
`tools/pvs.py`. Runtime: `web/engine/lite.js` (+ `lite_glb.js`, `lite_tc.js`, `natbloom.js`, `pvs.js`, `net.js`).
Reference bake times: full raw bake ~10 min at `--jobs 12`, lean ~4 min, packs 3-8 min, refit +2-8 min, PVS ~20
min for both halves; a bake directory is ~1.8 GB. Bake into `out/web/_next_<name>` and swap it in when verified.

## Before the bake: measure where the time goes

The dynamic page on an RTX 4090 at spawn: `high` 2.9-4.6 GB VRAM, ~3.9 M triangles, ~4,300 draws; `phone` 0.6 GB,
3,437 draws, 1.08 M triangles, yet it crashed the owner's phone and froze a thin laptop at 5 fps. The killers were
**draw calls and texture bytes** (108 MB of 2048² maps downloaded to be shrunk to 512 on the device; 359 proto GLBs),
not triangles; evaluating only 48-96 of ~430 lights made facades dark and wet ground speckled - cutting dynamic
light count is not a usable phone path. Measure CPU ms per frame, not throttled fps (1,800 draws at 28 fps was
CPU-bound; per-material batching halved it).

Targets given to the baking agent per viewpoint (spawn, plaza centre, terrace, canal quay), lightest tier:
<= 250 MB GPU, <= 400 MB RSS, <= 40 MB before the first walkable frame and <= 80 MB total, <= 300 draws, <= 500 k
triangles in view, first frame <= 8 s on localhost, >= 30 fps at 4-6x CPU throttle on WebGL 2. The ones that held
after the fidelity round: Phone <= 180 MiB GPU walking peak, <= 300 draws, first walkable frame <= 1 s at 4x CPU
throttle; Medium <= 700 MB, High <= 1.5 GB; 60 fps on the desktop GPU. State a margin like "176 of 180 MiB", do
not hide it.

## Format

- The scene is cut into **16 m cells** (113 here). Levels: `lo` (every cell, simplified LOD2 at ~13 % with up to
  0.22 m error, shipped in one `base.glb` of 7-9 MB in nearest-the-spawn order so the whole scene appears at once),
  `mid` (LOD2, 98 files, 13.5 MB / 0.87 M tris), `near` (LOD1, 38 files, 47 MB / 2.9 M; Low's top level), `ring`
  (the same for Phone, in clusters with parts ordered by size so a cell is drawn down to what shows at its distance
  and what the budget holds), `fine` (LOD0, 38 files, 117 MB / 7.7 M; Medium and High only). 15 far cells have only
  a lowest level; keep them in every visibility set (17,600 triangles are not worth the risk).
- Vertex record 10 x u16: position on a ~1 mm grid, texture layer, uv, baked light rgb (sqrt-encoded, HDR-capable:
  16 bit x 64), aux (65535 glowing, 65534 glowing-not-pulled, 65533 sign, 65532 sign-not-pulled); `fine` adds a
  dominant light direction so normal maps respond per pixel. Meshopt-compressed streams decoded in workers (one
  worker took Low's worst frame from 67 to 33 ms at 4x throttle). Meshopt v1 level 3 plus transfer gzip won against
  Draco (2x smaller decode time matters more than 25 % fewer bytes; Draco also drops zero-area triangles and
  reorders), zstd, brotli, Morton order and duplicate merging.
- Textures: WebP atlas pages into a 48-layer array; Medium/High add sharper colour sets and normal + roughness
  sets with relief height in alpha. Phone and Low read a leaner bake with only the first array.
- Manifest: cells, levels, glow sprites (position, colour, size, lift), flames, look settings, ground maps, PVS,
  a fingerprint of the scene. Files carry a version stamp and are cached forever.
- **The custom format is not what makes it fast; the loader is.** The same bytes wrapped as GLB (`bake2glb.mjs
  --live`, meshopt streams verbatim under a glTF JSON) are 0.2-0.4 % larger, render pixel-identically and reach the
  first frame 0.1-0.2 s earlier on throttled links **because `base.glb` is requested at the start of `load()` ahead
  of the textures**; the earlier 9-16 % penalty was GLTFLoader's, not the container's. The page now ships only the
  GLBs through a thin reader (`lite_glb.js`) that hands the streams to the same worker decode.
- Vertex-per-triangle census (1.2-1.5 vertices per triangle, ~0.9 positions per triangle even after merging):
  hundreds of thousands of loose parts averaging 7-8 triangles (bricks, planks, shingles) and 22-35 % texture-seam
  duplicates from per-stamp random UV offsets. A simplifier cannot merge across loose parts or UV seams, which is
  why far levels come out rough; a different UV scheme would cut 15-30 % of vertices.

## The lighting bake

- Each vertex is sampled at `P + 0.03 N` against a BVH of the whole scene: lantern light with occlusion (3 rays per
  lantern), a bounce term, 12-ray 3 m AO, moon and sky fill. Keep the raw bake (`--keep-raw`, ~670 MB) - every light
  experiment reads it and should never rewrite it.
- **Topology is adjusted for light before baking**: faces near lanterns are split until an edge is <= half its
  distance to the light (minimum 8 cm, up to 7 passes); on the two reduced levels edges over 0.75 m / 1.5 m are
  split where the corners do not predict the full model's occlusion. Simplifier colour weight 0.6 on fine, 0.15 on
  lo. Splitting everywhere has the best numbers but puts hard bright/dark triangles across plain walls: rejected.
- **Borders are locked** so neighbours and levels agree: cell borders, the ground (one ground mesh shared by all
  levels of a cell), glowing outlines, glass, water, cut-outs. This is what removes seams at cell edges and between
  modular floor pieces.
- **Reduced levels smear and darken for a measured reason**: lit on their own geometry, shipped mid walls carried
  0.85 and lo 0.77 of the correct light (bias -7.5 / -12.5 display steps) - a brightness jump at every level change
  on top of the smudge. Which vertices survive hardly matters (~0.1 step); where the light comes from matters.
- **Least-squares vertex refit** (Kavan et al. 2011, "Least Squares Vertex Baking"): fit each reduced level's vertex
  light so the surface shows the finest level's light. Plain least squares on **linear** light with a per-face mean
  regulariser gives bias 1.00 (tone-slope-squared weights pulled boundaries dark; a fit in sqrt space was worse);
  hidden samples dropped; vertices whose faces land on top-level **tint seams** keep their shipped value (a striped
  awning turned even pink); the ground, border, glow, glass, water and cut-out vertices keep their baked values.
  Results: mid rms 35.6 -> 26.0 steps, lo 44.4 -> 32.6; every level-change jump 0.72-0.97 of before. It must run
  **inside the packer after simplification** (`pack.refit=1`): lo and mid index the same raw buffer and
  `simplify()` only picks survivors, so a relight before packing does nothing for lo. Two passes: the packer
  records each level's index list, numpy fits per cell, the packer applies `<cell>.<level>.fit.bin` (counts and an
  index hash are checked). Report signed bias beside rms every run - a lower error with a worse picture means the
  weights or the measure are wrong. Alternatives measured and dropped: meshoptimizer `simplifyWithUpdate` (0.1 step,
  moves vertices up to 123 cm), lightmaps (12-17 steps at 10 cm texels for +20 % vertices and ~200 B/m²), a hybrid
  (the residual stays over 16 steps on a third of mid and 65 % of lo's walls, so it costs most of a full lightmap).
- **Covering-face rule** (`refit_cover = 0.08`): a sample point with a top-level face of the same facing within
  8 cm above it takes that face's light. Without it a cafe roof whose tiles the far level dropped to the deck was a
  black slab from across the square (the deck's light was baked *under* the tiles). The refit roughly doubles
  (349 -> 770 s). What lies under a detail layer must also carry the layer's material (`lod.md`).
- **One look ships**: far-reaching lamps (`look.lamp 0.43, reach 0, range_scale 3`; lean bake `look.lamp 0.6`),
  chosen by a path-traced Cycles reference of the real scene (`look.md`). Two looks can be baked into one topology
  (`--set look2.*`, union of light edges, six-channel refit, `_COL2`) and switched with one uniform in under half a
  millisecond; it costs +34-43 % geometry bytes (+16-23 % at 8 bits) and 1.6x bake time - fine for a comparison,
  too much to ship; or ship the second look as small files fetched on switch.
- Clamping vertex light above 1 per channel shifts hue (lit windows yellow, lamp glass lime): divide by the largest
  channel; for dark foliage brighten the texture layer by a gain and divide the colour by it.
- Lantern panes are baked nearly unlit because the page's glass shader adds the burner glare at draw time; a
  standalone export must evaluate that per vertex into `COLOR_0`, cutting large panes so vertices hold the hot centre.

## Ground maps: contact occlusion and lantern shadows

Per-vertex AO (one value per vertex, ground vertices 0.6 m apart) cannot hold the dark rim where a bench meets the
paving or a 0.4 m bench shadow, and a denser ground costs 350-780 k triangles per preset. Instead **two top-down
pictures of the whole ground** on one 6.25 cm grid (`tools/aomap.py`, 2400 x 2240 texels here):

- `gao.png`: per texel 96 cosine-weighted 1 m rays from 5 cm above the floor against the top level, a 3 x 3 blur;
  R = open share, G = terrain height. 0.8 MB, 28 s. Runtime: one texture read at `positionWorld.xz`, applied to
  pixels facing up within 0.2 m of the terrain height, multiplied into the baked light **and** the mirror strength
  and glints (on wet night paving diffuse-only occlusion is invisible). `pow 3`, strength 0.6 (the owner chose the
  softer of three links). 14 MB GPU at full size, 3.6 MB at half.
- `gsh.webp`: the lighting pass returns per vertex the light the lanterns' shadows took; ground vertices get it
  added back (the ground holds fill + unshadowed lantern light at every level), and per texel `v` = share of lantern
  light arriving (8 rays per lamp), `f` = fill / (fill + lantern). Runtime factor `R = (f cF + (1-f) v cD) / (f cF +
  (1-f) mix(v, 1, w) cD)`: exactly 1 off the ground; in a shadow the orange leaves and the blue fill stays.
  **The measured share alone is invisible** (many lamps fill each other's shadows, `v` 0.65-0.8 under a bench, and
  wet paving is glints and mirror): deepen locally against a coarse mip of the same texture, `c = clamp(v / (0.95
  vb), 0, 1)^5`, and gate the glints, mirror and half the fill with `c`. The owner could not see the first version
  on the phone; after the deepening: "It all looks great." A quality-menu switch (uniform, localStorage,
  telemetry). GPU memory doubles (RGBA8: 29 MB / 7.2 MB).
- Both can be **folded into a standalone GLB** without a rebake: the exporter cuts the ground where the maps vary
  (`--ground`, `--ground-edge`, `--ground-tol`) and bakes the factor into the vertex light (+4 % bytes, +7 %
  triangles; only the finest level is cut). A recursive cut needs a size floor or a sliver triangle recurses forever.

## Presets, budgets, streaming

- `phone` and `low` read the lean bake; `medium` and `high` add fine, sharp and normal sets, relief and
  reflections. Tier tables carry `near` / `mid` / `fine` distances with hysteresis (a cell leaves a level 20 % + 2 m
  further out than it entered), a view triangle budget spent nearest-first, a resident ceiling, a pixel budget and
  a worker count. Phone's resident ceiling decides how much of the near field stands at the middle level: at
  1.8 M nearly half the cells within 8 m did, at 2.2 M one in eight, still 60 fps. Medium's ceiling rose 2.3 -> 2.5 M
  when modelled far levels made mid heavier (a tailor shop at 13 m fell to mid: broken arch, smudged sign); a 16 k
  triangle change to the lowest level is enough to trip a ceiling, so re-simulate (`tools/perf_sim.py`) after any
  change to `lo`.
- **Pixel ratio 1 everywhere**, every preset and device, TRAA doing the smoothing: measured on the phone's own
  canvas (2.46 M pixels at ratio 3) the mirror pass was a quarter of the frame and the pixels the larger share;
  the owner overruled the desktop-capture verdict that ratio 1 "looks plainly blurred". Do not argue sharpness from
  a desktop crop of a phone canvas. `?ratio=2` to compare.
- **Streaming order**: every nearby cell to its wanted level before any large texture set, texture sets at low
  priority a few at a time (a compressed-texture experiment that fetched 2.5x more texture bytes first left far
  stand-ins on screen for half a minute: "mangled meshes"). Three levels in flight beat four. Measured at 30 Mbit
  cold: stand-ins within 40 m gone at 24.5 s -> 18.5 s with geometry-first order.
- Lighting pools must not pop while walking. The baked light was already identical across levels; what switched
  was the wet-film mask (each level's simplifier placed it differently), per-block nearest-level-only cobble
  lighting, screen-dependent reflections and lantern glass changing inside 30 m. Fixes: one shared ground mesh
  for all levels of a cell, per-pixel distance fades instead of per-block switches, a flat mirror image with the
  screen-space part limited to 8 m, lantern glass identical at every distance. Measured as patch brightness at
  near / mid / far: street-lamp streak 84 / 78 / 79 -> 81 / 81 / 81. Any per-tile runtime light selection (the wet
  shine picked the strongest lanterns per 4 m square) seams exactly like a bake seam: fade the choice smoothly.
- The dynamic Studio renderer stays as an opt-in desktop preset (~4 GB). Its two lessons: never one node material
  per tint/glow combination (791 materials exhausted the heap and locked the main thread; carry tint, glow,
  roughness, opacity as vertex attributes -> 77 materials, 27 s load at 119 fps), and `vec3(new THREE.Color())`
  compiles to black in TSL r186 (rust, moss and verdigris were black for a day without anyone noticing).

## Culling

- Frustum per cell **part** (one mesh per material kind carries its world box; failing parts get `layers.mask = 0`;
  the mirror pass resets the masks).
- **Occlusion is baked, not computed** (a potentially-visible set, `tools/pvs.py`, `?pvs=bake`): flat per-cell ID
  renders from eye points on a 0.5 m grid over everything walkable (pushed out of colliders as the walker is), three
  eye heights, five 512² cube faces, once per level with the whole scene at that level and once per preset at the
  levels it would choose; per 1 m square the cells seen from any point within 1.5 m plus every cell within 8 m.
  A bitset per square (100 kB) with its own version in the URL (remade sets were otherwise served from cache and
  "widening the sets changes nothing" cost an hour). Runtime: one table lookup when the camera changes square;
  a camera off the walkable band gets no set. Everything errs towards seen (glass, water, waterfalls never occlude).
  **Test from random places off the grid** (`--check --n 200`, `--allow 8` px): 256² faces and two heights missed
  slivers through railings; 512², three heights and a 1.5 m pad left none on three presets and 1-5 px slivers in
  four of 200 places on Phone, accepted under a stated tolerance. Measured: 36 % of cells seen from an average
  square (45 % with the skyline rule); the first sets showed 43-65 % because the ID copies took `visible` from the
  page's meshes, which hide a cell's lowest level while a higher one shows - a set that is suspiciously large on
  the bake with more geometry is worth a look at the ID pictures. The checker must `dispose` every ID copy it drops
  (13 GB after 180 places). Budget ~30 minutes after every bake including the check.
- three-mesh-bvh is the route for free cameras or moving occluders; for a static scene walked at eye height the
  rasterised bake is exact for the geometry drawn and costs nothing at run time. It is also a measured opt-in
  **walker** (`?walk=bvh&collide=town_lo.glb`): collide against the coarsest level (0.42 M tris, 18 MB, 0.15-0.3 s
  build, 0.05-0.3 ms per frame), never the finest (200-260 MB, 1.5-2 s); filter transparent, alpha-tested, sky,
  glass, water and fall materials first; measure the push-out on the capsule segment and lift straight up on slopes.
  Honest verdict: better on benches, stoops and real stair steps; stairs up 25-45 % slower, airborne downhill, sticks
  on clutter. Kept opt-in.

## The flicker stack (lit windows and reflections)

"Z-fighting" reports had five causes and five fixes, measured by paired captures 3 cm apart (lit-window
difference High 4.66 % -> 3.28 % -> 1.82 %; in the reflection 9.22 % -> 1.82 %):

1. **4 mm clearance in the generators** between any glowing face and any face behind it or glass in front, enforced
   by a Blender scanner (`tools/checks/zscan.py`: rays from glowing faces along +-normal up to 6 mm, bucketed
   < 0.3 / 1.5 / 3 / 6 mm). 19 of 102 light-bearing models needed fixes; the briefed rule was "a clearance fix, not a
   redesign". Round-to-nearest on the 1 mm grid is monotone so layers never swap; rounding glowing vertices
   outward made it worse.
2. **Reversed float depth buffer** on every preset (`reversedDepthBuffer`; the mirror camera needs
   `coordinateSystem` and `_reversedDepth` set before any custom projection; the screen-space march compares
   view z).
3. **Mirror camera without an oblique near plane**: a plain reversed projection plus a `ClippingGroup` with one
   plane at paving height, enabled only during the mirror draw (the oblique plane was the main cause).
4. **Pull glowing and glass vertices toward the eye** at the `lo` level and in the mirror pass (`t = min(a /
   distance + k, 0.5)`, a = 0.06 m lo / 0.12 m mirror), screen position unchanged.
5. **Packer locks** opaque non-glowing triangles within D of a glowing triangle during simplification (fine 6 mm,
   near 30 mm, mid 45 mm, none at lo). The lock preserves modelled clearance; it cannot create one.

Plus: sub-pixel emissive slits (26-28 mm louvre gaps) flash like z-fighting and no depth offset fixes them - model
glow as wide dim bands, multisample the mirror target, run TRAA on every preset (the owner's call: no toggle,
`?traa=0` for debugging). Diagnose from the geometry (a Blender script listing faces by depth plane around a box;
a meshopt dump around a box), not from frame differences, which move everywhere with clouds and shader time.

## Merge gate and swap checklist

- Gate stated before the result: "no merge until Low/Phone speed is proven within 5 % of live at 4x CPU
  throttle" (met by worker decode, not fewer triangles). Captures from the branch vs live must match within the
  reload-vs-reload floor (two captures of one look differ 2.5-3 % because water and flames move).
- Swap: staging has `manifest.json`, `lean/manifest.json` **and `pvs.json` in both** (PVS finishes 15-20 min after
  the main passes); capture phone, Low and High at the judged views on staging and live and diff; rename live to
  `_next_prev`, staging to `_next`; re-export the standalone GLBs and fix the gallery's size labels; keep one
  previous bake as rollback until the owner has seen the new one on their phone; clear `localStorage`'s remembered
  tier in the test browser. The page must not fall back to the dynamic engine when `town.json` is re-resolved
  before the new bake is ready (compare fingerprints only for review renders). A bake dir name must match
  `^_next\w+$` or `?bakedir=` silently loads the live bake.

## Measuring

- GPU timestamps: `WebGPURenderer({ trackTimestamp: true })`, `resolveTimestampsAsync('render')`, with the GPU kept
  busy (`?gputime=1&rep=13`: repeat whole frames, step `nodeFrame.frameId`, sway the camera; an idle desktop GPU
  clocks down). Timestamps come back summed since the last resolve: divide by frames. Respect `maxQueries`.
- Telemetry (`web/engine/telemetry.js` -> `POST /api/telemetry` -> `tools/telemetry.py --devices | --errors | --sid`):
  a first report the moment the page opens (a tab that dies loading shows up), then ~4 s, 15 s, 30 s, every minute
  and on leave; ids per visit and per browser; device, GPU, screen, memory, load and download timings, frame-time
  statistics, the HUD readouts, which effects ran (`extra.fx`), console warnings and errors. Off on localhost and in
  capture runs. It found: the ASTC path never ran on the iPhone (BC7 tried first), a phone running yesterday's
  `main.js` under today's page, a 7 s blank wait the owner left during, and the heat curve (60 fps for half a minute,
  then 26). Read the session's URL and `extra.fx` before touching a control that "does nothing" on one device.
- `tools/perf_sim.py` simulates the level choice over the whole walkable scene for a tier and bake with
  `tris=` / `resident=` overrides - it found a hanging basket at its crude far level up close before a swap.
- Comparison rows are made on one loaded page by setting uniforms from the console (`__town.eng.P.*.value`) and
  capturing per setting, joined with `magick +append`, the current look always in panel 1; difference images
  (`-compose difference -evaluate multiply 8`, read `%[fx:mean]`) before judging by eye - the first contact-occlusion
  version was "far too faint" and the number said so.
