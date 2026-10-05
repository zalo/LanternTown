# LanternTown — replication prompt

Give this whole file to a coding agent with a shell, Blender, Node, a GPU browser and the ability to
spawn subagents. It contains the original brief, every correction the user made along the way
(folded in as requirements so they are right the first time), the architecture that worked, and the
lessons that cost the most time. Sections 1–3 are *what*; 4–10 are *how*; 11–13 are *rules and traps*.

---

## 1. The brief (from the user; the reference it named is described instead)

> You are a professional 3D modeller and technical director. Please make a realistic realtime
> lantern town square you can walk around in in threejs. Use the lamplit storybook town squares of
> classic adventure games as a reference; but make
> the rendering and detail realistic and lifelike. This needs 100+ unique, exquisitely detailed,
> fully parametric assets, and this environment ends as a composed scene using its assets. Iterate
> on these models until a separate subagent judge rates them at 75/100. Likewise for the town
> square itself.
>
> Subagents author assets. Renders are batched through a shared GPU. Parallel reviewer subagents
> score renders against a quality guide. Iterate until work passes. Everything is parametric: seeds
> and params produce variants, with no hand-placed one-offs. Publish progress to a live updating
> static gallery hosted on the terminal proxy webpage.

Reference images (put them in `assets/`, crop the useful regions into `docs/ref/`):
a storybook-town concept board (districts: gated entrance street, fountain/monument court,
raised café terrace, canal quay), a second board of nine walk-level perspectives the layout is
modelled after, and optionally a third image for a second, separate town assembly.
The boards are painterly: take **design, layout, palette and mood** from them, take **rendering,
construction and materials** from reality.

## 2. Requirements the user added (treat as part of the brief)

Modelling and judging
1. **Model with the Blender command line** (`blender --background --factory-startup --python …`).
   Not an interactive Blender/MCP session, not geometry generated in three.js.
2. **Only the three.js look counts.** Every review image, gallery image and judged shot is a
   capture from the real-time three.js page, never a Blender render. (Blender renders are fine as
   private geometry sanity checks.)
3. **Everything parametric**: each asset is a generator with typed params, a seed and 3–5 named
   variants; the town is a rule-and-seed layout generator. No hand-placed one-offs anywhere.
4. **Low-detail levels are modelled, not decimated.** Each generator (or the toolkit under it)
   produces its own LOD1/LOD2; decimation is only the fallback when a modelled level fails a check.
   LODs do not need a judge gate — the goal is "beautiful cheap models".
5. Landmarks must stay **legible at distance** (clock hands and numerals must survive the far level).

The live demo
6. **The walk-around page must load and run on a phone and a thin laptop.** The first full-detail
   scene crashed the phone and froze the laptop at 5 fps. The default link must pick a light preset
   by itself; heavy presets are opt-in.
7. **Bake for the web**: lighting and geometry are baked into a compact streaming format; the
   fully dynamic "Studio" renderer stays as an opt-in desktop preset.
8. **The baked Low preset is the foundation**; Medium and High are built *up* from it (better
   levels, sharper textures, per-pixel detail, better lighting techniques), not down from Studio.
9. Baked lighting must not **pop** as levels change (light pools appearing/disappearing while
   walking), must not show **seams between modular floor pieces**, and must not **smear** on the
   cheap levels.
10. **No flashing emissive surfaces** — coplanar glowing faces make screen-space reflections
    flicker (scan for coplanar faces; emissive ones first), and so does any bright feature
    **narrower than a few pixels** at the distance it is seen from or in the low-resolution mirror
    image: glow seen through 28 mm slits between louvre blades flashed in the wet paving exactly
    like z-fighting and was reported as such. Model glow as wide, dim bands; multisample the
    mirror image; run temporal anti-aliasing (three.js `TRAANode`) on **every** preset, phone
    included (the user's call after comparing: no toggle in the UI, `?traa=0` only for debugging).
10a. **Things that float part the water.** A boat is not full of the canal: the asset's origin
    sits on its waterline and it declares its waterline outline (a `float` socket, taken inside
    the planking so the cut edge hides in the hull at every level of detail); the terrain step
    collects the outlines of everything placed in water and cuts them out of the water sheet.
    And do not put the water material *inside* a boat: bilge water or a "swamped" variant in a
    town boat reads as the canal coming through the hull and is reported as a broken cut-out.
    Town boats are dry (boards, thwarts, a coil of rope); wet variants stay in the asset's
    parameters for the gallery.
10b. **Water must read as water**: dark body with no diffuse response to speak of, Fresnel
    mirroring, several octaves of *crested* ripples (anisotropic noise moving across its crests,
    mip-mapped so it fades instead of aliasing), and above all the lanterns broken into columns
    of dashes. Blending a sky colour and a town colour by low-frequency noise reads as clouds.
10b. **A phone has a laptop's pixels and a tenth of its cooling.** An iPhone ran Medium and High
    at its own 1206×2043 (pixel ratio 3, 2.46 M pixels — three times what the Phone preset draws)
    at 60 fps for half a minute, grew hot and fell to 26: the telemetry's fps-over-time is how
    throttling shows. The user suspected the screen-space reflections. Measure instead of
    guessing: GPU timestamp queries (`WebGPURenderer( { trackTimestamp: true } )`,
    `resolveTimestampsAsync( 'render' )`, then `info.render.timestamp` = the last frame's passes
    added up) with the GPU kept busy (`?gputime=1&rep=13`: 13 whole frames per displayed one, the
    camera swaying so the mirror image redraws; an idle desktop GPU clocks down and its times
    mean nothing; three.js passes draw once per `nodeFrame.frameId`, so step that between
    repeats). Result at the phone's canvas: the mirroring is a quarter of the frame (High 0.70 ms
    → 0.52 without), its step count does not matter, and the pixels are the larger share (0.40 at
    ratio 2, 0.34 at 0.8 M pixels, 0.30 at ratio 1.5, 0.27 at ratio 1 — under 1.5 nothing more is
    saved on the desktop GPU, and the picture is softer). First remedy: on a phone (coarse
    pointer, short side under 600 px) Medium and High kept their reflections and drew the Phone
    preset's 0.8 M pixels (`hand: { pixels }` in the tier) — the user confirmed the phone then
    ran very well. A desktop GPU's proportions are not a phone's: confirm on the device.
    **The user's final call: pixel ratio 1 everywhere** — every preset, every device, every demo
    page (the town, the one-file town, the LOD demo, the GLB viewer, the gallery's asset viewer),
    so that a Retina laptop gains what the phone gained: one canvas pixel per CSS pixel, the
    temporal anti-aliasing doing the smoothing. In the tier this is a cap beside the budget
    (`ratio = min( devicePixelRatio, tier.ratio || 1, sqrt( pixels / css pixels ) )`);
    `?ratio=2` (or `?pr=2` on the GLB pages) raises it to compare.
11. Distant high-frequency textures (brick) must not alias: mip chain + anisotropy, and fade
    relief/normal detail with distance. Anisotropic filtering is right for the colour and wrong
    for anything a *step* or a *normal* is taken from: a reflection mask or tilt cut from a relief
    height that is only filtered across the pixel's short side turns the joints of receding paving
    into 1-pixel rows and its pits into specks that crawl as one walks (worst on a phone's dense
    screen, and only where reflections are on). Read that height a second time with an isotropic
    footprint the size of the pixel's long side (`textureGrad`), about 1.5 mip levels up.
12. **Glow sprites sit exactly on their fitting** and every lamp that is lit gets one (a globe lamp
    without a glow reads as switched off).
13. Animated parts **swing and sway about the correct physical axis** (gates on their hinge line,
    flags and hanging signs about their top edge).

Composition
14. The plaza must look **intentional, not cluttered**: axes, a symmetric fountain ring, one lamp
    family per zone, feature pieces anchored, loose dressing grouped at edges and activity zones,
    open paving left open.
15. Nothing floating, buried or clipping: lanterns attach to something, bridges meet the ground
    without seams, stairs do not run through fences. Sign and inscription text must have contrast
    against its background.

Delivery
16. **Lossy optimisation comes last.** Start with lossless wins (gzip, meshopt without
    quantisation, bundling). Explore KTX2 / GPU texture compression only after the scene is right.
17. Provide a **GLB exporter from the baked format** for debugging, portability and to verify the
    custom format's gains; use `MSFT_lod` so stock tooling can switch levels.
18. The **gallery** stays live throughout: assets with scores, materials, town shots, references
    and a Demos tab linking every walkable build and experiment.
18a. **Visitors report back by themselves**: every page sends its device type and details (GPU
    name, screen, memory), load and download timings, frame-time statistics, the on-screen
    readouts, console warnings/errors and uncaught errors to the project's own server, with a
    first report before anything heavy loads so a tab that dies loading still shows up. One small
    module (`web/engine/telemetry.js`), one route (`POST /api/telemetry`), one reader
    (`python3 tools/telemetry.py [--devices | --errors | --sid ID]`). Build this early: the
    user's phone is where the scene fails and you cannot see it otherwise. Off on localhost and
    in capture runs so the tools' own browsers do not pollute it; no address stored; the log
    folder is never served; the reader trusts no field's type.

Working rules (the user enforces these)
19. **No scratch on RAM disks** (`/tmp`, `/dev/shm`). Scratch goes in `<project>/.scratch/<task>/`
    on the normal disk and is deleted when the task ends. This applies to every subagent.
20. **No wildcard paths in shell commands** (`*`, `?`, `[..]`) and no `rm`/`mv`/`cp` with a shell
    variable in the path — both raise a permission prompt that halts unattended work. Name paths
    explicitly or do the matching and deleting inside a Python script with absolute paths.
21. **Keep GPU and RAM use low when idle**: the user runs other GPU work on the same machine.
    One tab per task, close what you open, stop your browsers when not capturing; restart Chrome
    when memory creeps (tabs leak between runs).
22. Nothing is pushed, published or uploaded to an external service without the user saying so.

## 3. Definition of done

- ≥ 100 unique parametric assets, each ≥ 75/100 from an independent judge on three.js renders.
  (Reference run: 136 assets in 7 categories — street 39, architecture 26, buildings 18,
  décor 16, plaza 15, lighting 14, vegetation 8 — 133 passing; 57 procedural materials.)
- A composed town ≥ 75/100 from an independent judge on walk-level three.js captures
  (reference run: 81, 14 judged viewpoints, ~530 placed items).
- A walkable page with presets `phone | low | medium | high` (baked) and `studio | studio-medium`
  (dynamic). It opens on a light preset until a heavier one is proven on the owner's real phone;
  the finished page opens on `high` everywhere. Reference device numbers from the user: M4 MacBook Air
  60 fps on `phone`, 30–40 fps on `medium`; desktop GPU 55–60 fps on `high`.
- A live gallery with a Demos tab; every URL handed to the user is one they can actually open
  (see §11 on proxies).

---

## 4. Architecture that worked

```
assets/<category>/<id>.py   parametric generators (Python, toolkit `tk`)
materials/<name>.py         procedural PBR materials (numpy) ─► out/mat/<name>/{albedo,normal,orm}.webp
        │  Blender CLI (tools/bl_build.py, via `tools/tsq build`)
        ▼
out/glb/<id>/<variant>[.lod1|.lod2].glb  + .json sidecar (size, tris, light/fx/attach sockets, parts)
        │
        ├─► asset review renders (three.js page driven by a GPU render daemon) ─► judge subagents
        │
town/layout.py  (rules + seed) ─► out/town/town.json (items, lights, colliders, shots, terrain)
        │
        ├─► Studio runtime: streams GLBs, clustered real lights, SSR        (desktop, opt-in)
        └─► Web bake (Blender + Node): cells × levels, vertex-baked light, texture arrays
                 ─► baked runtime: phone / low / medium / high               (default)
```

Stack used: Blender 4.1 CLI, three.js r186 `WebGPURenderer` + TSL (WebGL2 fallback),
Node 22 with `meshoptimizer`, a stdlib-only Python static server, headless Chrome on the GPU.

### Repo layout

| path | what |
|---|---|
| `tk/` | modelling toolkit: params/RNG, geometry builders (box, cylinder, lathe, sweep, extrude, moulding runs, booleans), surface (materials, UV declarations, smoothing, weathering paint), sockets (`light`, `fx`, `socket`, `part`), composition (`use()`), export finalisation, LOD machinery |
| `assets/` | one file per asset: docstring, `META`, `PARAMS`, `VARIANTS`, `build(p, rng)` |
| `materials/` | one file per procedural material |
| `docs/` | `AUTHORING.md`, `API.md` (generated from `tk`), `MATERIALS.md`, `QUALITY.md`, `TOWN_QUALITY.md`, `LOD.md`, `catalog.json` (id, class, mount, brief, deps, reference crop), `ref/` |
| `tools/tsq` | the one CLI authors use: `build`, `render`, `lod`, `mat`, `matlist`, `api`, `brief`, `assets`, `status` |
| `tools/town.py` | `gen | build | terrain | resolve | render | release | status` for the town |
| `tools/webbake*.{py,mjs}` | the web bake: Blender side (lighting, refinement) and Node side (simplify, pack) |
| `tools/serve.py` | static server + `/api/manifest` + render upload endpoint |
| `tools/site.py` | the same URL layout as a folder of plain files (pages, three.js, the bake, the gallery's pictures, a frozen manifest) for a static host such as GitHub Pages; the gallery hides what needs the server when the manifest has a `static` part |
| `web/gallery/`, `web/town/`, `web/engine/` | gallery, walk page, the two runtimes |
| `reviews/` | judge output JSON per asset / material / town, per round (kept out of the repository) |
| `out/` | everything generated; `.scratch/` everything temporary |

**Use git from the first commit.** The reference run was not a repository and had to imitate
branches with directory copies (`web/next/`, `web/perf/`, `premerge/` backups) — it worked but
cost care. Develop risky runtime work on a branch/worktree and merge when the user says it is good.

## 5. Phases (in this order)

0. **Foundations before any asset**: toolkit + generated API doc, the `tsq` CLI, material library,
   the three.js review-render page and a render daemon that batches captures through one GPU,
   the gallery, the two quality guides, and one hand-polished **exemplar asset** (a lantern head
   composed into a gas lamp post) that every author copies the style of.
1. **Materials** (procedural, judged on swatches rendered in the engine).
2. **Catalogue** (`docs/catalog.json`): 130+ briefs with class, mount, tier, dependencies and the
   reference crop each one answers to. Tier 0 = components other assets compose (`use()`).
3. **Asset waves**: author subagents in parallel (bounded by the Blender cap), judge subagents in
   parallel, iterate each asset to ≥ 75.
4. **Town**: layout generator → build → walk shots → town judge → iterate to ≥ 75. Give the town
   to one long-lived "town builder" subagent that owns `layout.py`.
5. **Performance**: measure where time and memory go, then build the baked runtime (§8) on a
   branch; make the default link safe on a phone *before* showing it to the user.
6. **Modelled LODs** across all assets (§9), then re-bake.
7. **Delivery optimisation** (§10), lossless first.
8. **Experiments** behind flags, each linked from the gallery Demos tab.

Expect the user to walk the town on their own devices and report by eye. Reproduce first (same
link, same preset, capture the spot), explain the cause in one paragraph, then fix.

## 6. Orchestration

Roles
- **Lead (you)**: owns the plan, briefs, locks, merges, the gallery and all communication with the
  user. Does small fixes directly; delegates anything that needs a long independent context.
- **Author** (one per asset): reads a strict reading list (authoring guide, API doc, the exemplar,
  the material table, one reference crop), then build → render → look → fix, 3–6 times.
- **Judge** (fresh context every round, never the author): gets the three.js sheets, the brief,
  the source (to check it is really parametric) and the rubric; writes `reviews/.../r<N>.json`
  with sub-scores, caps, 3–8 concrete problems ("what, where, what instead") and strengths to keep.
- **Specialists** (long-lived): town builder, baking, LOD reducer, texture compression, model-file
  optimisation, format exporter, loader/upstream work. One owner per directory.

Rules that kept it running
- Cap heavy processes machine-wide (reference: ≤ 4 Blender instances) and say so in every brief.
- One GPU render queue for asset sheets; each specialist that needs a browser gets **its own**
  Chrome on its own port and profile and must name it on every call. Never touch a shared browser.
- Lock files for exclusive phases (`BAKING`, `REBUILDING`); bake into a staging directory and swap,
  because other agents read the live output.
- Every brief repeats the house rules (§11) verbatim, states which files the agent owns, what it
  must not edit, and the report format: result first, numbers, what is unverified, files, cleanup.
- Keep author context small: strict reading budget, bounded command output, targeted lookups
  (`tsq api <fn>`) instead of reading toolkit source. Compact or restart long-lived agents.
- Agents can be resumed by message after a session restart; describe their on-disk state and ask
  them to verify half-written files.
- A subagent's "done" is a claim. Check the artefact (open the image, run the check) before
  relaying it; keep "measured" and "estimated" apart in what you tell the user.
- Notify the user (push notification) when a long task ends or a question blocks progress.

## 7. Authoring standard and rubrics (condensed)

Asset file: `"""docstring"""`, `META = {name, class, mount}`, `PARAMS` (default/min/max or choices,
description; `seed` added automatically), `VARIANTS`, `build(p, rng)`. Metres, Z up, front faces −Y.
Mounts: `ground`, `wall`, `wall_ground`, `hang`.

Triangle budgets by class: prop_small 6k, component 12k, prop 16k, furniture 22k,
street_furniture 26k, vegetation 60k, structure 70k, ground 80k, building 160k, landmark 260k.
Build checks: library materials only, budget (2× = error), ≤ 10 material variants, ≤ 6 light
sockets, ground contact, wall assets in front of the wall plane, plausible size.

What makes an asset pass
- **Real dimensions** (door 2.0–2.2 m, storey 2.8–3.2 m, step 0.17 × 0.30, lamp post 3.5–4.5 m).
- **Built the way it is made**: separate boards with gaps, straps with rivets, glazing bars holding
  panes, brackets that reach the wall, nothing paper-thin.
- **Three scales of detail**: silhouette, mouldings/panels/brackets, bevels and fixings with seeded
  irregularity. A bare primitive standing in for a moulded part *will* be noticed (a plain inverted
  pyramid under a bay window read as "weird shapes" — it needed a moulded corbel profile).
- **Library materials** with declared grain/course direction; **weathering that tells a story**.
- **Light**: every glowing thing has a visible emissive mesh *and* a light socket with intensity in
  candela (candle 1–2, wall lantern 25–45, street lantern 50–80, shop window 20–50). Too hot is the
  common mistake: lit ground should be rich amber, not white. Each fitting also declares where its
  glow sprite sits and how big it is.
- Sockets for attachment; `part(..., motion="swing"|"spin"|"flutter")` with an explicit pivot and
  axis for moving parts; all text is a parameter.

Asset rubric (100, pass ≥ 75): silhouette/proportion/scale 20 · geometric detail and construction
20 · materials and weathering 20 · night read and lighting 20 · parametric variation 10 · technical
cleanliness 10. Caps: wrong object or scale ≤ 50; broken geometry/shading ≤ 60; broken or
look-alike variants ≤ 70; blown-out or invisible at night ≤ 70. Sheets per asset: daylight front
and back 3/4, night close-up, clay with a 1.8 m figure and triangle count, night hero, all variants.

Town rubric (100, pass ≥ 75): composition and layout 20 · lighting and atmosphere 20 · realism of
rendering and materials 20 · density and storytelling 15 · faithfulness to the reference 15 ·
technical cleanliness 10. Caps: visible blockout ≤ 60; camera blocked or asset broken/floating/
intersecting ≤ 68; lighting unreadable in more than two shots ≤ 65; not the reference's town ≤ 55;
hand-placed one-offs in the layout ≤ 70. Judged on eye-height walk shots from the same page a
visitor walks.

Tell judges: "75 means it ships without embarrassing the town; 85+ is rare; do not round up —
every problem you fail to name costs a revision round."

## 8. Runtimes

### Studio (dynamic, desktop only)
Streams GLBs and material textures; every lantern is a real light through clustered lighting;
screen-space reflections on wet paving; bloom, fog, animated sky, flames, moving parts.
About 4 GB of GPU memory. Two lessons:
- **Do not create one node material per tint/glow combination.** ~790 materials exhausted the JS
  heap. Carry tint, glow, roughness and opacity as per-vertex attributes; the town then needs ~77
  materials and loads in about a second.
- In TSL r186, `vec3( new THREE.Color() )` compiles to black; pass `r, g, b` numbers or a `uniform`.

### Baked runtime (the default)
Why it is fast: almost no draw calls (one mesh per cell and level, one material because all
surfaces sample a texture **array**), no real-time lights or shadows (lighting is in the
vertices), a 20-byte vertex, and streaming that shows a complete coarse town immediately.

Format (reference numbers)
- The town is cut into ~113 **cells**. Levels: `lo` (every cell, shipped in one ~7 MB file so the
  whole town appears at once), `mid`, `near`, and `fine` (full models; Medium/High only).
- Vertex record 10 × u16: position on a 1/1024 m grid, texture layer, uv, baked light rgb
  (sqrt-encoded), aux; `fine` adds a dominant light direction so normal maps respond per pixel.
- Meshopt-compressed vertex and index buffers, decoded in workers into interleaved buffers.
- Textures: WebP atlas pages into a 48-layer array; Medium/High add sharper colour sets and
  normal+roughness sets; relief height rides in alpha.
- Manifest lists cells, levels, glow sprites (position, colour, size, lift), flames, look settings.
  Files are requested with a version stamp so they can be cached forever.

Lighting bake (Blender side)
- Light is ray-sampled per vertex from the town's light list, with occlusion, a bounce term and AO.
- **Topology is adjusted for light before baking**: big faces near lamps are split until edges are
  short relative to their distance from the light; on reduced levels, long edges are split where
  the corners do not predict the full model's occlusion in between.
- Cell borders, the ground and glowing outlines are **locked** so neighbours and levels agree
  (this is what removes seams between modular floor pieces and at cell edges).
- Simplification weighs baked light as an attribute so vertices carrying a light change are kept.
- To stop cheap levels smearing light, **refit** their vertex light by least squares against the
  finest level of the same cell (Kavan et al. 2011, "Least Squares Vertex Baking") rather than
  keeping whatever value the surviving vertices happened to have. Do this from the start.
  The ground, cell-border vertices, glow, glass, water and cut-outs keep their baked values (all
  levels share one ground; borders must agree with the neighbour).
- The fit asks the top level "what light does this point of the reduced surface show?". Where the
  reduced level has dropped a thin layer (roof tiles, planks) its surface is the deck *under* that
  layer, and the nearest top-level face is the deck too - lit as something buried, i.e. black.
  **Covering-face rule**: a sample point that has a top-level face of the same facing within 8 cm
  above it takes that face's light instead (`refit_cover = 0.08`). Cost: the full bake's refit
  went from 349 s to 770 s. Without it one café roof was a black slab from across the square.
- All levels of a cell must show the *same* light pools; otherwise pools pop as the visitor walks.

Contact occlusion on the ground (the dark rim where a bench, a planter, a wall meets the paving)
- The bake's ambient occlusion is one value per vertex (12 rays, 3 m), and the ground has a vertex
  every 0.6 m at best: the rim under a table falls between vertices, and things look set *on* the
  picture rather than standing in it. It does not need a runtime pass (GTAO costs a depth-normal
  prepass and a blur on every frame - the heat a phone cannot spare), nor decal quads (hundreds of
  transparent draws to sort, z-fighting with modelled setts), nor a denser ground (measured: 10.6 %
  of 20,400 m² has a visible gradient; 8-12 cm edges there are 350-780 thousand more triangles in
  every preset, where Phone's whole view budget is 975 thousand).
- Instead **one top-down picture of the whole ground** (`tools/aomap.py`, last step of
  `tools/webbake.py` when the working data is kept): 6.25 cm texels (2400 × 2240 here), per texel
  96 cosine-weighted rays 1 m long from 5 cm above the floor against the top level, nearer hits
  counting more; red = open share, green = terrain height. A 3 × 3 blur takes the ray noise.
  0.8 MB as PNG, 28 s to bake, named in the manifest (`gao`), shared by both bakes.
- Runtime: the opaque material reads it at `positionWorld.xz` and multiplies its baked light, only
  on pixels facing up within 0.2 m of the terrain height in green (a table top above the spot is
  not the ground). One texture read, no geometry, every level of detail alike. RG8 with mips,
  14 MB of GPU memory at full size (Medium/High), 3.6 MB at half (Phone/Low).
- Tuning learnt by picture: the physically measured share (`pow 1.6`) was all but invisible on wet
  night paving, whose light is mostly mirrored. It reads at `pow 3` **and** applied to the
  reflections too (the mirror strength and the lantern glints take the same factor). `?gao=0` off,
  `?gao=k` (default 0.6 - the user preferred the softer rim over full strength), `?gaopow=`, `?gaospec=`, `?show=gao`.
- Limits: it covers the terrain and floors lying on it; raised floors that are part of an asset
  (stair treads, a terrace deck) keep only the vertex occlusion. A single-file GLB town cannot read
  a world-position map - there the route is to densify the ground at contacts and least-squares
  fit its vertex light against this map.

The lanterns' shadows on the ground (the shape a bench, a post, a planter cuts out of a lamp's pool)
- Same problem, same answer: a bench's shadow is 0.4 m wide, the ground's vertices are 0.6 m
  apart, so the vertex bake smears it into a vague dark patch that also differs between levels of
  detail. The shadows go into a **second picture on the contact map's grid** and out of the
  ground's vertices.
- Baker (`tools/webbake_bl.py`, `gsm*` settings, `ground_shadows`): the lighting pass also returns,
  per vertex, the light the lanterns' shadows took (`T`) and the unshadowed lantern light. Ground
  vertices (opaque, not foliage, facing up, within the height band of the terrain) get `T` added
  back - **the ground's vertices hold fill + unshadowed lantern light, at every level and in both
  bakes**. Then per 6.25 cm texel, 5 cm above the floor, against the top level: `v` = share of
  the lanterns' light that arrives (8 rays per lamp across its size, never under a 0.06 leak),
  `f` = fill / (fill + unshadowed lantern light), smoothed over texels that have a fill. Two
  town-wide hues go to the manifest: `cF` (fill, bluish) and `cD` (lantern, orange). About 60 s for
  5.1 million texels; 41 % of the ground is in lantern light and 0.36 of it arrives on average.
- Map tool (`tools/aomap.py`, `shadow_map`): aligns the two byte planes to the contact map's grid
  and writes `gsh.webp` (lossless, R = sqrt(v), G = sqrt(f)) beside `gao.png`, half size for the
  lean bake; the manifest's `gsh` entry carries `band`, `cF`, `cD`. 0.80 MB (lean 0.28 MB).
- Runtime (`web/engine/lite.js`): the loader merges both pictures into **one RGBA texture**
  [occlusion, height, sqrt v, sqrt f] - still one texture read per pixel. Factor on the ground's
  baked light, `w` = height band x facing up:
  `R = (f cF + (1-f) v cD) / (f cF + (1-f) mix(v, 1, w) cD)` - exactly 1 off the ground, and in a
  shadow the orange leaves and the blue fill stays (a ratio of colours, so the hue shifts as it
  does in the bake). `?gsh=0|1`, `?show=gsh`.
- **The measured share alone is invisible - deepen it locally.** Shipped as above, the owner could
  not see a shadow at all on the phone: many lamps fill each other's shadows (`v` is 0.65-0.8 under
  a bench, more under a post that stands below its own lanterns), and wet paving at night is glints
  and mirror reflections, which a factor on the diffuse light never touches (the contact map taught
  the same). A plain power on `v` shows the shadows but dims and blues the whole square (open
  ground is `v` ~0.8 too). What works: compare `v` with `vb`, the same channel read from a coarse
  mip level of the same texture (texels of ~3 m, one extra low-level fetch):
  `c = clamp(v / (0.95 vb), 0, 1)^5` - 1 on open ground whatever its `v`, falling only under what
  stands close by. `v c` replaces `v` in the factor, and `c` also gates the lanterns' glints
  (together with `v`), the mirror strength, and half of the fill light. Knobs:
  `?gshpow= ?gshspan= ?gshbias= ?gshspec= ?gshmir= ?gshall=`
  (`gshpow=0&gshspec=0.5&gshmir=0&gshall=0` is the measured share only). Judge it at the phone's
  canvas size in a view down at a lamp post's foot, with the switch on and off side by side - not
  in a debug view of the map.
- **A switch in the quality menu** ("Ground shadows: on/off", in the same no-reload group as the
  lens flare): a uniform blends `v` to 1, kept in localStorage, reported in telemetry (`fx.gsh`).
  On by default. Off shows the ground *without* lantern shadows (not the old vertex shadows).
  A bake without `gsh` in its manifest removes the menu entry.
- Costs and limits: the merged texture is RGBA8 instead of RG8 - **GPU memory doubles** (14 -> 29 MB
  at full size, 3.6 -> 7.2 MB at half). The moon's shadows, bounce and vertex occlusion stay in the
  vertices. A bake made with the map *needs* it: read without the page's shader (a standalone GLB
  export, `aomap.py --off`) its ground has no lantern shadows - bake those with `--set gsm=0`,
  which keeps the shadows in the vertices as before.

One look ships: **far-reaching lamps** (`look.lamp 0.43`, `reach 0`, bake `range_scale 3`; the
lean bake `look.lamp 0.6`) — the bake tool's default (`FAR` in `tools/webbake.py`). The earlier
"pools" look was dropped after a side-by-side.

Comparing two lighting looks (how that decision was reached; keep the machinery, not the bytes)
- Do **not** make a second bake and reload the page: the mesh is the same, only the light differs.
  Light the soup twice and store both results in each vertex; the runtime picks with one uniform.
- Bake: `--set look2.lamp=… --set look2.reach=… --set look2.range_scale=…` on top of the normal
  settings. The light-driven splitting uses the **union** of both looks' light edges, so one
  topology serves both; simplification weighs both colours; the least-squares refit solves six
  channels at once.
- Record: 14 × u16 (second rgb at words 10–12, one spare) on levels without a light direction,
  18 × u16 on `fine` (direction 10–11, second rgb 12–14, spare, second direction 16–17). In the
  GLB these are `_COL2` and `_G2` beside `_COL`/`_G`. The manifest gains a `look2` block with its
  own lamp settings, glow sprites and light list.
- Runtime: `lit = mix( col², col2², uLook )`; both looks' lantern grids are rows of one data
  texture; two sets of glow sprites, one visible; the reach term is a uniform. Switching costs
  under half a millisecond (`setLook( i )`, `?look=2`, key `L`). Retake any cached cube probe.
- Cost measured: +36 % geometry bytes on the lean levels, +43 % on the full bake (`fine` +51 %),
  bake time ×1.6 — too much to ship for a choice the visitor does not need; once the user has
  picked, rebake with the one look.

Natural bloom, scattering fog and HDR (asked for by reference: Sonic Ether's "SE Natural Bloom &
Dirty Lens" for Unity, and the fog scattering of his Minecraft shader pack)
- **Licences first.** The Unity effect is MIT: adapt it, credit it in the file header. The
  Minecraft pack (SEUS) forbids any reuse of its source, snippets included: read it for the idea
  only, copy nothing, keep the download out of the project and out of the source zip, delete it.
- Technique (`web/engine/natbloom.js`, one TSL node for WebGPU and WebGL 2): blur the frame into a
  row of ever smaller targets with **no brightness threshold**; bloom is a blend *towards* a
  weighted sum of the levels (`mix( colour, blurred, 0.14 )`), not light added on top - what a pixel
  gives to its halo it loses. The **fog reads the same levels a second time**: a pixel seen through
  a fog share `a` is blended towards the blurred picture by `a`, so fog smears light instead of
  only hiding it. Lens dirt (optional, a texture drawn from a seed) is a third reader. One pyramid
  replaces three's thresholded bloom on the tiers that had one; `?scatter=0` puts the old one back.
- Details that had to be found: weights go by **octave**, not by level index (a pyramid that starts
  at a quarter must give its widths the weights the half-size one gives them, or Phone looks
  washed out); the fog share (exp² of the fog's optical depth, capped at 0.9) rides in the
  **alpha of TRAA's input** so it is antialiased with the
  colour - and TRAA's resolve material writes alpha 1 unless its `blending` is `NoBlending`; a
  guard keeps a bright thing in front from glowing into far fog behind it (per level: the fog
  share of what lit that level, carried in its alpha); levels are for a 720-high picture, one more
  per doubling of the canvas. Tiers: Phone 4 levels from a quarter, the rest 6 from half.
  Strengths as shipped: the user asked for the "bloom scatter" at 70 % of the first tuning, so
  both readers went down together - glow 0.2 -> 0.14 (`?nb_bloom`) and the part of the fog share
  that is smeared 1 -> 0.7 (`?nb_scatter`); the fog's own colour and density are untouched.
- **One fog path for everything, and a sky that is in the world.** Fog is two layers thinning
  with height - the night haze (scale height 18 m, squared in the exponent so the first tens of
  metres stay clear) and an optional ground mist (5 m, plain extinction) - each integrated *along
  the line of sight*: `dist · ( e^(-y0/H) - e^(-y1/H) ) / ( ( y1 - y0 ) / H )`, not looked up at the
  height of the surface. The materials' fog and the scatter share call the same function with
  their own strengths, so "further away" means "through more fog" in both. The sky is no longer a
  backdrop that follows the camera with a special case in the share: it is a dome of 600 m standing
  around the town, written to depth and drawn last (only open pixels are shaded), painted by the
  direction from the eye (moon and stars do not slide) and fogged like any front - clear overhead,
  gone into the haze low over the roofs, and the same falls out for its scatter share.
  `?mist=0.012&fogGain=2` shows the ground mist; the user compared strengths side by side and
  chose **none** (the scatter alone, with the fogged sky) - offer it, do not default to it.
- **Measure before tuning**: the first version looked weak however the sliders stood. A debug view
  (`?show=hdr`: grey under white, one colour per stop above) showed why - lit surfaces all under 1,
  lamp mantles at 1-2. An energy-conserving bloom has nothing to spread in an LDR frame. The bake's
  storage was HDR-capable already (sqrt in 16 bit × 64); the *content* was authored for LDR.
- HDR emitters: glowing vertices (and halo sprite cores, pane glare, flames) are expanded at
  runtime by `k = min( glow · max(L,1)^hot, max( top / L, 1 ) )` (1.5, 2.5, 48): the brighter a
  thing was authored, the more it gains; painted and stained panes under 1 keep their colour.
  Mantles and globes then sit 3-5 stops over white, windows 1-2, and halos, wet-paving streaks and
  fog glow all follow without touching them. `?hdr=0` off, `?hdr_glow= ?hdr_hot= ?hdr_top= …`.
- HDR display output: `new WebGPURenderer( { outputType: HalfFloatType } )` gives an `rgba16float`
  canvas with `toneMapping: { mode: 'extended' }`; ask for it only when
  `matchMedia( '(dynamic-range: high)' )` matches, and check `context.getConfiguration()`. Tone
  curve there: the standard ACES picture (so SDR and HDR match up to white) times a gain that rises
  with the stops over white (1 → `peak` 4 at four stops), then the sRGB curve with
  `renderOutput( …, NoToneMapping, SRGBColorSpace )`. The HUD says "HDR", telemetry records it. A
  headless test machine cannot show it: verify on the user's device (an iPhone on iOS 26 reported
  `display high, canvas extended` and held 60 fps).
- What two further articles (Froyok's custom bloom and lens flare for UE4) added, and did not:
  taken - level count tied to resolution; a lens flare (strength 0.25: six ghosts and a halo
  with a small colour split; off unless switched on in the quality menu, see below). Not taken, with the measurement: the 13-tap downsample and tent
  upsample chain (this pyramid blurs every level with a Gaussian as it shrinks it; a lamp's glow
  varied under 0.5 % over quarter-pixel pans, and the widest level shows no blocks), Karis
  averaging (it dims exactly the small bright lamps the picture is about; TRAA is in front),
  geometry-shader glare stars (no such stage on the web - the per-lamp halo sprites already do it,
  anchored in the world and without the pulsing the article admits to).
- **Lens flare: threshold the picture, not a blurred level, and compress the response.** The first
  version read a quarter-size blurred level over a threshold; the user: "it only activates when I
  get near a light and then it's too strong". Exaggerating it showed why: in a blurred level a
  lantern across the square is thinner than the lit clock face, so the flare came from the clock
  and the windows, never from far lamps, and a lamp overhead (a disc thirty times over the
  threshold) glared. What works: (1) a bright pass on the **full-size** picture - four bilinear taps
  per quarter-size pixel, *each* thresholded on its own at 4 scene units, which is the ceiling of
  every lit or glowing surface (`knee`), so only halo cores and flames pass; (2) each tap rolled
  off towards a ceiling (`x·R/(x+R)`, R = 2); (3) one nine-tap Gaussian at quarter size so scaled
  ghosts cannot step over a one-texel lamp; (4) on the way out `c · (max/R)^(curve-1)`, curve 0.5 -
  a lamp with a tenth of the light gives a third of the ghost. Far lamps then draw small ghosts and
  halo arcs, the near one a soft disc, at about the same brightness. `?nb_fthr ?nb_frange ?nb_fblur
  ?nb_fcurve`. The user halved the first strength (0.5 -> 0.25) and asked for it as an **option in
  the quality menu that does not reload the page**: the passes are always built and skipped while
  `flareIntensity` is 0, the menu entry flips that uniform, the choice is kept in localStorage.
  **Off by default on every tier** (the user's choice after living with it). Rule for every post
  effect: switch it with a uniform, never with a reload (and remove unavailable `<option>`s from
  the DOM - a phone's picker shows hidden ones).
- **Name every script by its date, or a phone will run yesterday's handler under today's page.**
  The flare entry "could not be toggled, it's always on" on the user's iPhone while it worked in
  desktop Chrome. Telemetry showed why: picking it reloaded the page to `?quality=flare` - the
  *previous* `main.js` (which treats every menu value as a tier) running under the new
  `index.html` and the new engine. `Cache-Control: no-cache` + `Last-Modified` was not enough
  behind the proxy. Fix (`tools/serve.py`, `stamped`): a page whose import map carries
  `data-stamp` is sent with `v=<mtime>` on its `<script type="module" src>` and an import-map
  entry for every `.js` beside it and in `web/engine` (`"../engine/lite.js":
  "../engine/lite.js?v=…"`), which every `import` then follows; unchanged files keep their URL and
  stay cached; workers are outside the map and stay unversioned. A static host needs the same done
  at build time. Also clean up what the stale script wrote: an unknown `?quality=` is taken out of
  the address and an unknown remembered tier is forgotten. When a control "does nothing" on one
  device only, read the session's URL and `extra.fx` in telemetry before touching the control.
- **Contrast with physically far-reaching lamps comes from the curve, not from the fill.** With an
  unshortened inverse-square reach the lamps' light overlaps over the whole square and the picture
  goes to one amber mid-level. A second light set without lamps (`--set look2.lamp=0.001`, loaded
  with `?bakedir=_next…` - **the name must match `^_next\w+$` or the page silently loads the live
  bake**) splits every vertex into fill and lamp share at runtime (`?hdr_amb= ?hdr_lamp=`), and
  showed the sky/moon fill is a small part of the square: halving it changes roofs and sky sides,
  not the flatness. What separates pools from roofs and sky is the grade. The first setting
  (`contrast` 1.06 -> 1.5, `exposure` 0.72 -> 0.56, lit-wall ceiling `knee` 1.6 -> 4) looked right
  in SDR captures and the user called it **too dark and too harsh** on their HDR phone, where
  everything over white is boosted again. Half-way (1.25, exposure back at 0.72, knee 2.4) was
  still too much; given links for 1.15 and then trying 1 themselves, the user settled on
  **contrast 1 (no grade), exposure 0.72, knee 2** - shipped. On an HDR display the display supplies
  the contrast; leave the exposure alone, hand over `?contrast=&knee=` links, and let the grade be
  judged on the device, not in captures. Tune such things on **one loaded page** by setting
  the uniforms from the console (`__town.eng.P.contrast.value` …) and capturing a row - reloading
  per variant is slow and drifts with the clouds.
- **Signs and neon need a level of their own.** Every glowing surface rolls off to one ceiling
  (`kneeGlow`), so a neon tube ended no brighter than a lit window, and a back-lit sign face was
  authored dimmer still; the user asked for them "doubled or tripled" for more colour in the
  square. `HDR.neon` (2.5, `?hdr_neon=`, a live uniform) multiplies both the light and its ceiling
  for what glows with no texture of its own (layer 0: the bake's `emissive` and `neon`) **in a
  colour no flame has** - lamps, bulbs and lit rooms run red >= green >= blue; cyan, green, violet
  and pink do not (`?show=neon` paints the mask). The material name would not have done it: the
  hotel's blade sign is `emissive` with a violet tint, not `neon`. Raising only the ceiling changed
  nothing on the sign faces (they sit far under it). The same factor goes on the coloured entries
  of the bake's light list when the town is built, so their streaks on wet paving follow. The hue
  rule cannot see a sign in a flame colour (the HOTEL sign's gold letters and crown, its chaser
  bulbs, a marquee's bulbs), and the signs' light on walls and paving is in the bake; the user
  asked for the rebake that covers both:
  - **An asset declares its signs**: `mat( lib, tint, glow=..., sign=1 )` (the parameter travels in
    the GLB material). Check what the letters are made of first - the hotel's shared the lit rooms'
    material and its sign bulbs the façade bulbs', so the sign needed materials of its own
    (`M_gold`, `M_sbulb`) before it could be told apart. Undeclared, the bake takes `neon` in a
    non-flame hue as a sign - not the material name alone: the globe lamps' opal core is `neon`.
  - **The flag is a vertex aux code**: 65533 (sign) and 65532 (sign, not pulled) beside the glowing
    65535 / 65534; every place the packer and the refit test `== 65535` must take both
    (`GLOWS()` in `webbake_pack.mjs`, the two masks of `lsvb_refit.py`). The page boosts
    `max( hue rule, flag )`, so festoon bulbs keep theirs and a bake without flags still works.
  - **The signs' lights** are the town lights named `neon...`; the bake multiplies them by
    `LOOK.neon`, which `webbake.py` reads from the page's own `HDR.neon` default, and the manifest
    says so (`signs: { k, lights: [rows] }`) - the page scales those rows by `HDR.neon / k`, or
    their glints would be boosted twice.
  - What it bought: the letters, crown and bulbs at the tubes' level, and 2.5 x the cyan / violet
    wash on the walls beside the three signs - real but modest (these are 5-10 m lights), most
    visible on the stone under the hotel sign. In SDR captures gold letters at 2.5 x roll off
    towards cream; judge them on the HDR display.
- Asked what AAA technique was missing, the answer was lamp-lit volumetric fog; the user ruled it
  out as too expensive for phones. Do not build it; the scatter share + flare + grade carry the air.
- Cost: on the desktop GPU on and off differ by 0.01 ms a frame; the test that counts is the
  phone, read from telemetry (`extra.fx` says which effects a session ran).

Water (one material for canal, basins, troughs): on presets with the mirror pass it writes its
Fresnel term and rippled normal into the mirror target and gets the quay walls, hulls and lit
windows from screen-space rays (calm ripples, about half the slope that looks right in a still,
give the long golden streaks; more reads as chop). Presets without the pass get a stand-in from
the same ripples: low rays show a warm "wall" as bright as the baked light on the water under
them, high rays the sky, and each nearby lantern is drawn as its **mirror image** — constant
radiance where the reflected ray passes the burner, no distance falloff, dimmed only by the blur
of sub-pixel ripples. That needs its own light list: nearest lanterns, unmerged, counted far
beyond their lighting range (a highlight computed as a BRDF of a point light with a fixed
roughness shows nothing a few metres away).

Presets: `phone` and `low` read a leaner bake with only the first texture array; `medium` and
`high` add the fine level, sharp and normal sets, relief and reflections. The finished page opens on `high` on every device (the owner's call once
pixel ratio 1 and the cheaper reflections let it hold 60 fps and stay cool on a phone); a choice
made in the selector is remembered. While a tier is unproven on a phone, default to a light one.

Streaming order matters as much as bytes: get every nearby cell to its wanted level **before**
starting large texture sets, fetch those at low priority, a few at a time. (A compressed-texture
experiment that requested 2.5× more texture bytes first left distant stand-ins on screen for half
a minute on a phone — the user read it as "mangled meshes".)

Culling: view frustum **and** occlusion, both per cell part
- Frustum: besides three.js's own per-mesh test, every part of a cell (one mesh per material kind)
  carries its world box; each frame the boxes are tested against the four side planes of the
  camera (0.5 m pad) and failing parts get `layers.mask = 0`. The mirror / probe pass resets the
  masks: it looks elsewhere.
- Occlusion is **baked, not computed per frame** (a potentially-visible set): `tools/pvs.py` opens
  the town page with `?pvs=bake`, which draws the bake in flat per-cell ID colours from eye points
  on a 0.5 m grid over everything walkable (grid points pushed out of colliders the way the walker
  is), three eye heights (1.45 / 1.65 / 1.85 m), five cube faces of 512² each, once per level of
  detail with the whole town at that level, then once per preset with every cell at the level
  that preset's distances give it (and the top level cut to the parts its budget may leave), and
  unions what comes back. Per 1 m square of ground: the cells seen from any eye point in it or
  within 1.5 m, plus every cell within 8 m. Stored as a bitset per square in `pvs.json` (100 kB)
  beside the manifest; the manifest's `pvs` key switches it on and carries a version of its own
  that goes into the file's URL (remade sets for the same bake were otherwise served from the
  browser's cache, and "widening the sets changes nothing" cost an hour). A new bake has none, so
  run the tool after every bake (`webbake.py --pvs` does; ten minutes per bake).
- Runtime: one table lookup when the camera changes square; hidden cells are skipped in the same
  loop as the frustum test. No per-frame cost, nothing for a phone to compute. It only applies to
  a walker: a camera outside the square's height band, off the walkable ground, or squeezed
  between two colliders (the eye can then be inside a wall) gets no set and everything is drawn.
- Everything errs towards "seen": glass, water and waterfalls never occlude, leaf cards are cut
  out as on screen. **Test it** from random places *off* the grid with the page running each
  preset as a visitor would (`tools/pvs.py --check --n 200`, exit 1 on a miss): 256² faces, two
  eye heights and a 0.5 m pad missed slivers through railings (up to 125 px of a 512² × 6 cube);
  512², three heights and 1.5 m left none on High, Medium and Low, and on Phone four places of
  200 where 1–5 pixels (of 1.57 M) of a cell 42–83 m off show through a gap of the coarsest
  level from a narrow range of positions — a sampling limit, accepted under a stated tolerance
  (`--allow 8` px, printed as "sliver"), not explained away. A checker that draws ID copies of
  the page's meshes must send `dispose` to each copy it drops: the renderer's per-mesh records
  hang on shared materials and held 13 GB of dropped levels before it ran out of memory.
- Measured (113 cells of 16 m): both bakes' sets show 36 % of the cells from an average square
  as looked at, 45 % with the skyline rule: the 15 far cells that only have a lowest level are
  in every set. With tight sets the check found one of them left out from one place on the
  terrace (18 pixels, 93 m off, through a gap between two roofs - parallax against a near gap is
  what a 0.5 m grid of eye points samples worst), and all fifteen together are 17,600 triangles:
  culling them is a risk with nothing to gain. After it: no miss in 200 places on any preset,
  two slivers of 1-2 pixels on Phone.
  The first sets showed 43 % (full) and 65 % (lean), and the difference was a baker bug, not the
  town: the ID copy of a level took its `visible` flag from the page's mesh, and the page hides
  a cell's lowest level while a higher one is shown - so in the "whole town at the lowest level"
  pictures those cells were missing as occluders and everything behind them counted as seen.
  Valid sets (they err towards "seen"), but too generous; a set that is suspiciously large on
  the bake with *more* geometry is worth a look at the ID pictures themselves. With the first
  sets triangles drawn fell 5 % (Phone) to 8 % (High) on average and draw calls about 20 %, up
  to 38–56 % in single views — modest on average because the budget already keeps hidden far
  cells cheap (not measured again with the tighter sets). The next step,
  not taken: feed the set into the level budget so hidden cells neither stream nor hold fine
  levels.
- three-mesh-bvh can do this too — raycasts from the camera to sample points of each cell box
  against a BVH of the occluders, or as the ray engine of an offline bake. For a static town
  walked at eye height the rasterised bake is exact to the pixel for the geometry really drawn,
  costs nothing at runtime and needs no CPU copy of the mesh; the BVH route is the one for free
  cameras or moving occluders.

## 9. Levels of detail

- The toolkit carries a current level; builders (`lathe`, `sweep`, bevels, arrays) reduce their own
  segment counts and drop sub-visible profile points at lower levels.
- Authors (or a reducer pass) mark identity parts to keep (`lod_keep`) and supply stand-ins
  (`lod_standin`: e.g. a beaded band becomes a plain ring, a multi-shell globe becomes one sphere).
- Each level is checked against the full model by image gates (silhouette, colour, light leak,
  clearance between surfaces); a failing level is rejected and the decimated copy ships with a
  warning. Reference result: LOD1 ≈ 39 % and LOD2 ≈ 12 % of full triangles, most assets modelled.
- Landmarks get stricter far-level settings; thin dark parts on a face (clock hands) need a keep
  mark *and* a matt contrasting material or they vanish or z-fight at distance.
- Watch for clearance regressions: a stand-in that touches its neighbour creates z-fighting.
- **What lies under a detail layer must look like the layer.** A far level may drop individually
  laid roof tiles (planks, shingles, cobbles) down to the deck they sit on, so that deck carries
  the tile material on its top face (the carpentry's material only below and at its ends). One
  building here had a dark oak deck under its tiles: from across the square on the Phone preset
  its roof was a solid black slab, on Low a scatter of tile shards over black - the largest "hole"
  in the whole town, and no simplifier setting could have fixed it. The material is only half of
  it: with a terracotta deck the roof was still black, because the deck's *light* was baked under
  the tiles; the refit's covering-face rule (section 8) cured that. Verified by picture: Phone
  shows a lit brown roof, Low a tiled one. After any roof work, look at every tiled roof at the
  two lowest levels from 40 m and more.
- **Slivers of sky through reduced levels** (what "gaps" turned out to be). Buildings are hollow
  shells; a simplifier that moves a window rim, a chimney top or an eave by a centimetre opens a
  one-pixel crack and the sky (or the lit room) shows through. Find them by measurement, not by
  eye: `tools/pvs.py --holes` draws each level in ID colours from the walker's eye points with
  the top level's depth beside it and lists every place where a reduced level lets a ray through
  that the top level stops (`.raw/holes.json`: per level and cell, the points). The packer
  (`pack.holes=1`) then finds the source triangles at those places and locks them - whole loose
  parts, kept as they are - before simplifying. Measured: at the far level 86-88 % fewer
  seen-through pixels for 30-32 % more triangles (base file 7 -> 10 MB); at mid only 6-7 % fewer
  for 6-8 % more, at near 17 % for 9 %. So mend the far level only (`pack.holes_levels`,
  default `lo`): mid and near cracks are rarer, smaller on screen and not worth their bytes.
  The census must come from a pack *without* mending, or it cannot see what mending hides.

## 10. Delivery optimisation (reference measurements, desktop, localhost unless noted)

Lossless, adopt early
- **gzip model files on the server**: models at spawn 156 → 76 MB.
- **meshopt without quantisation + one bundle for the far models**: 471 → 106 requests, 72 MB,
  captures pixel-identical; first frame 54.5 → 40.1 s on an emulated 50 Mbit link.
- Not worth it: vertex/index reordering (+16–19 % after gzip). Not safe: joining primitives across
  materials (−16 % draw calls, but changes draw order where surfaces share a depth).
- Frame time did not change measurably from any of these; parsing models is 2–4 % of load.
  Texture upload and shader/pipeline creation dominate.

Textures
- KTX2/UASTC halves texture GPU memory (High 988 → ~500 MiB) but grows the download
  (118 → 185 MB for the whole High page).
- Better on desktop: **ship WebP, compress to BC7 on the client in workers** — same download, same
  memory saving, sharp textures arrive *earlier* (pixel work leaves the main thread). A 240-line
  single-mode BC7 encoder was enough; third-party GPU encoders may carry licence terms — read them.
- Phones need ASTC (and compressing the first-frame array is the only saving on the light presets).
  A 4x4 ASTC encoder in the same workers did it; measured on an iPhone, Phone preset: 48 layers
  encoded in 0.4 s, swapped in at 1.8 s, 33 → 4 MB of textures on the GPU, 60 fps throughout
  (Apple GPUs offer BC7 as well; either path gave the same numbers). Once both paths had run on a
  real phone it became the default, with `?tc=0` to switch it off and review renders left
  uncompressed. Pictures with it on and off differ by 39–45 dB PSNR: not visible.
- Always keep the uncompressed path as the fallback and print the chosen format on screen for
  device testing.

Interchange
- **The custom packed format is not what makes the baked runtime fast — its loader is.** The same
  bytes wrapped as GLB (meshopt streams verbatim, quantised attributes) are 0.2–3 % larger after
  gzip, reach first frame at the same time, show the finer levels 9–16 % later, use the same
  memory and frame time, and render pixel-identically. What a GLB cannot do is stream: the packed
  far-level file is built cell by cell as it arrives. A "portable" GLB (float attributes, unlit
  materials, opens in Blender) is about 2× the bytes after gzip, 5× the draw calls and 2× the time
  from data to first frame. Keep both exports: one proves the format, the other opens anywhere.
  The live page ships the exact GLB (one file per cell and level, one base file parsed as it
  arrives and requested ahead of the textures) through a thin reader that hands the streams to
  the same worker decode — not through GLTFLoader, whose path was the 9–16 % slower one. Result:
  0 differing pixels, +0.2–0.4 % on the wire, first frame 0.1–0.2 s earlier on throttled links.
  Once that held, the packed files were a duplicate: the bake deletes them after the verified GLB
  export (`webbake.py --keep-packed` keeps them; `bake2glb.mjs --live --drop-packed`), the
  manifest says `glb.only`, and the exporter reads its level bytes back out of the `.glb` files.
- **Do not coarsen the position grid to save bytes.** Draco positions on a 2 mm instead of a 1 mm
  grid saved a few per cent and passed a hole check, but surfaces modelled a millimetre or two
  apart became coplanar and fought for the depth buffer all over the town. A crack check is not
  a depth check; judge lossy geometry by walking it.
- The phone preset's ceiling on resident triangles decides how much of the near field stands at
  the middle level (whose small props are crude up close): at 1.8 M nearly half the cells within
  8 m did, at 2.2 M one in eight - still 60 fps, and visibly better.
- **The file that opens in both Blender and three.js**: `KHR_materials_unlit` + `COLOR_0` for the
  baked light, Draco, WebP (Blender refuses meshopt and KTX2). Measured on the 3.1 M-triangle
  level (31 MB): connectivity + positions 33 %, UVs 25 %, vertex light 23 %, textures 19 %; the
  same data as meshopt streams is 53 % larger raw and 19 % larger after gzip. No glTF extension
  gives an environment that both importers show, so the sky is an unlit mesh node: a rounded box
  fitted to the town (bounds grow about 1.1×, +0.19 MB) carrying the panorama mapped by direction
  from the plaza, with `scenes[0].extras` naming the image and projection so an app can draw it
  as a true background instead (no sliding as one walks).
- `MSFT_lod` + `THREE.LOD` buys nothing at run time over the custom streaming loader. Loaded
  eagerly it costs 150 MB before the first frame, 458 MB of geometry in JS and 450–545 ms hitches
  on a throttled CPU; loaded lazily it matches the packed loader's first frame, but `THREE.LOD`
  has no triangle/memory budget, no fades, selects by distance to the object's centre (about 40 %
  more triangles at the spawn) and never frees CPU copies (434 MB against 80 MB). Its value is
  portability and a test bed for upstream loader support.
- three.js `GLTFLoader` does not implement `MSFT_lod`; build a small test kit (synthetic cases +
  real cells + expected results + a check page) before writing loader support, in two tiers:
  transparent (node → `THREE.LOD`, no API change) and opt-in lazy loading (per-level buffers or
  HTTP range requests on one file). Range requests go out uncompressed (the browser asks for
  `identity` encoding), so per-level buffers cost fewer bytes than one ranged file; and the static
  server must actually answer `Range` with 206 — Python's stdlib handler does not.

## 11. House rules for every agent (paste into every brief)

- Scratch only in `<project>/.scratch/<task>/`; never `/tmp` or `/dev/shm`; delete it at the end.
- No wildcard characters in shell paths; no `rm`/`mv`/`cp` with shell variables; tidy up from a
  Python script with explicit absolute paths. Run commands from the project root.
- Keep command output bounded; never print a whole manifest, layout JSON or transcript.
- Browser work through the project's browser CLI, on **your own** Chrome port and session name,
  one tab, reused; verify the GPU adapter is the real GPU; stop the browser when idle.
  Serve local files over localhost HTTP, not `file://`.
- Respect the Blender cap and the lock files. Do not restart shared servers.
- Do not edit outside the directories you were given. Report defects in other owners' files.
- Report: result first; numbers with conditions; what was **not** verified (no real phone, one
  run, emulated network); files changed; what was cleaned up.
- Pages must use **relative URLs** only — they are viewed through a path-prefix reverse proxy.
  Any `localhost` URL given to the user must be rewritten to the public proxied form.

## 12. Traps that cost time

Rendering and capture
- Shader `time` is wall clock: pin it (and animation phases) or two captures of the same build
  differ by thousands of pixels and no A/B comparison means anything.
- Supersampled captures hide aliasing. Capture at device pixel ratio 1 when judging shimmer.
- CPU throttling in the browser slows only the main thread; worker timings stay optimistic.
- Measure loads cold (fresh browser, empty cache), paired, more than once; say when it was once.
- A headless Chrome that survives between runs leaks GPU memory; every open 3D tab keeps its scene
  in VRAM. One task, one tab; restart between heavy rounds.

Geometry and light
- Coplanar faces flicker, and emissive coplanar faces make reflections flicker across the frame.
  Model panes as single sheets, offset decals, and run a coplanarity scan after every town build.
- Not every "z-fighting" report is depth fighting: look at the geometry first. Sub-pixel emissive
  slits flash the same way and no depth offset fixes them (see requirement 10).
- A variant draw seeded by an item's rounded position reshuffles when the item is moved, even by
  a few centimetres in height: check which variants a layout change needs built.
- Glow sprites must come from the fitting's own declared position, never from a guess in the town.
- Moving parts need an explicit pivot and axis; check gates and flags in motion, not in a still.
- Replacing an array texture at run time: disposing the old one can be undone by the renderer
  re-creating it for meshes that still reference it. Shrink it to 1×1×1 before disposing and make
  sure no upload is attempted with null data.
- Per-material node variants explode memory (see §8).

- A checker that redraws the page's meshes in its own scene must not copy state the page changes
  from frame to frame. The visibility bake copied each mesh's `visible` flag, and the page hides a
  cell's lowest level while a higher one is shown - so near cells were missing from the bake's
  "whole town at the lowest level" pass, could not hide anything behind them, and the sets came
  out valid but far too generous. Nothing failed: the test only catches cells wrongly hidden.
  Measure the share of cells culled as well as the misses, and question a low one.

Process
- Showing Blender renders as progress misleads; the user asked "are those three.js or Blender?".
- A gallery link that 404s, or a `localhost` link, wastes a round trip. Check every link you send.
- "It builds" is not "it looks right": authors must open their own sheets every iteration.
- Density added to raise a score reads as clutter; organise by a plan (requirement 14).
- When the user reports a visual problem, ask (or determine) which link, preset and device, and
  reproduce at that spot before theorising.
- Keep experiments behind flags on a preview copy; promote to the live page only with evidence.
- Do not start lossy optimisation early; it hides real problems and the user did not want it.
- If the town builds from released copies of the asset sources (so authors can keep revising),
  releasing is a step of every rebuild, and the build must not decide what is stale from file
  dates the release copy preserves. A full forced rebuild (half an hour) ran on stale copies of 44
  assets here because the release was skipped: have the build print how many working files differ
  from their released copies.
- A bake that is one rebuild behind its town must still be what a visitor gets. The page here
  compared the bake's fingerprint with the town file and fell back to the streaming engine (the
  one that crashes phones) the moment the town was re-resolved, hours before the new bake was
  ready. Keep the strict check for review renders only, and bake into a staging folder that is
  swapped in.
- "Unchanged" must be checked on the file a step writes. Layout generation wrote one file and the
  comparison read another, so a rebuilt asset's larger footprint (which dropped two statues) went
  unnoticed until the next stage complained.

## 13. Reporting to the user

Lead with the result. Give the link (proxied form) and the one or two numbers that matter. Say
plainly what is not verified. Keep internal names, agent identifiers and file dumps out of it
unless asked. When several agents are running, a short table — agent, state, where it is — is the
right shape. Do not stop while work is owed; do not publish, push or upload without being asked.

---

### Kick-off checklist

1. `git init`; create `assets/ materials/ tk/ tools/ web/ docs/ reviews/ out/ .scratch/`.
2. Write the toolkit, generate `docs/API.md`, write `AUTHORING.md`, `QUALITY.md`,
   `TOWN_QUALITY.md`, `MATERIALS.md`; crop references into `docs/ref/`.
3. Build the server, the gallery and the review-render page; start the GPU render daemon; confirm
   the gallery opens through the public URL.
4. Author the exemplar asset by hand to ≥ 85; only then write the catalogue and launch waves.
5. Decide the baked format and presets **before** composing the full town, so the first link the
   user opens runs on their phone.
6. Build LOD support into the toolkit from day one, and the least-squares light refit into the
   first bake.
7. Keep the gallery's Demos tab current; it is how the user finds everything.
