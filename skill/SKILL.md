---
name: parametric-environment
description: Build a realistic real-time walkable 3D environment of any kind (a town square, a harbour, a forest shrine, a market hall, a station, an interior) from 100+ parametric Blender-CLI assets authored and judged by subagents against rubrics, composed by a rule-and-seed layout generator, baked to vertex-colour lighting in streamed cells with modelled LODs, and served as a three.js WebGPU page that runs on a phone. Use for any "model a scene from many procedural assets in Blender and ship it in three.js" project; carries the toolkit, CLI, bake pipeline, runtime, rubric and brief templates, and the measured lessons of a four-day reference run.
---

# Parametric environment pipeline

A complete, measured way to go from a reference board to a judged, baked, phone-ready three.js environment with a
fleet of subagents. The process is the same for any setting; the lighting design (sun, sky, lamps, fire, neon,
underwater caustics) and the asset catalogue are the parts that change. The reference run (LanternTown, October
2026, a lantern-lit old-town square at night) produced 136 parametric assets (133 passing an independent judge at
>= 75/100), 57 procedural materials, a composed scene judged 81/100, and a page that holds 60 fps on an iPhone, in
four days, with most of the rework in the first day. This skill exists so the next environment skips that rework.
Where a lesson below quotes a lantern, a wet cobble or a moon, read it as the reference run's instance of a general
rule; the rule is stated beside it.

## What is in this folder

| path | what |
|---|---|
| `references/process.md` | the phases in order, orchestration rules, definition of done - **read first** |
| `references/authoring-lessons.md` | how to model an asset that passes the first time: construction, real light intensities, material substitutions, weathering at vertex resolution, composition costs, a pre-flight list |
| `references/materials-lessons.md` | procedural PBR textures that read as real and tile |
| `references/judging.md` | running independent judges; what they mark down, ranked; harness artefacts to tell them about |
| `references/composition.md` | the rule-and-seed layout generator; "intentional, not cluttered"; cameras as layout; what raised the scene score |
| `references/lod.md` | reducer-first modelled levels, the trimming pass, image gates, holes and slivers |
| `references/bake-runtime.md` | the vertex-colour light bake, cells and levels, refit, ground maps, streaming, culling, the flicker stack, measuring |
| `references/look.md` | beautification: what worked (with values) and what did not, and how the owner judged it |
| `references/delivery.md` | lossless first, device texture compression, GLB exports, the resilient loader, publishing with incremental uploads, cost |
| `references/traps.md` | everything that cost time, by area |
| `references/project-map.md` | every script and module in `project/`, what it needs running, what to change for a new environment |
| `references/docs/` | the agent-facing contracts **as shipped by the reference run** (`AUTHORING.md`, `QUALITY.md`, `TOWN_QUALITY.md`, `MATERIALS.md`, `LOD.md`, `API.md`, `API_TEX.md`): worked examples with the lantern-town theme in them |
| `templates/` | theme-neutral versions to fill in: `authoring-guide.md`, `rubric-asset.md`, `rubric-scene.md`, `materials-guide.md`; skeletons `asset.py`, `material.py`; briefs for author, judge, scene judge, layout builder, specialist |
| `project/` | a ready-to-copy project: `tk/` (the Blender toolkit), `tools/` (CLI, render daemon, layout pipeline, bake, checks, measurement, server, publishing), `web/` (runtime, pages, gallery, review renderer), `materials/` (57 generators), `assets/` (137 generators of the reference run, reusable or re-themed), `town/` + `town2/` (the reference layout generators), `docs/`, `package.json`, the Pages workflow, `prompt.md` (the reference run's replication prompt) |

In the code, the word **town** means "the composed environment": `tools/town.py` is the layout pipeline,
`town/layout.py` the layout generator, `out/town/town.json` the resolved scene, `/town/` the walk page. The names
were kept so the shipped tools and docs agree; rename them in your copy if you like, consistently.

## Starting a new environment

```bash
cp -r <this skill>/project <your project> && cd <your project>
npm install                                   # three, three-mesh-bvh; meshoptimizer, draco3dgltf
export TSQ_BLENDER=/path/to/blender-4.1       # or `blender` on PATH
python3 tools/gen_api.py                      # docs/API.md, docs/API_TEX.md from the toolkit's docstrings
python3 tools/serve.py &                      # http://127.0.0.1:8765 (gzip, Range 206, manifest, uploads, telemetry)
tools/renderd.sh                              # the GPU render daemon on its own Chrome (:9223)
tools/tsq mat cobble_granite --res 512        # a material in seconds -> out/mat/cobble_granite/{preview,far,sheet}.png
tools/tsq render lamp_post_gas                # a reference asset -> out/renders/lamp_post_gas/{sheet,hero,variants}.png
tools/town.py all && tools/town.py render     # the reference scene (needs its assets built; ~45 min cold)
python3 tools/webbake.py --pvs                # the baked, streamed scene -> out/web/_next ; open /town/
```

Then make it yours, in this order (`references/project-map.md` lists every hard-coded value):

1. **Reference and mood**: crops into `docs/ref/`; decide the lighting design (key light: sun / moon / none;
   emitters: lamps, fire, screens, neon, bioluminescence; atmosphere: clear, haze, fog, dust, rain) - it drives the
   rubrics' "read under the light" criterion, the bake's look constants and the post stack.
2. **The two rubrics and the authoring guide** from `templates/` with the theme filled in; one exemplar asset.
3. **The catalogue** (`tools/catalog_src.py`): components, props, architecture or terrain pieces, hero pieces at
   true size, backdrop; `docs/materials.json` and new or re-tinted `materials/`.
4. **The layout plan** (`town/layout.py` `_generate`, keeping the `Town` machinery): zones, routes, light families,
   focal points, cameras.
5. **Budgets and tiers**: class triangle budgets, light defaults, the bake's cell size and look, the preset tables.

## The process in one screen (details in `references/process.md`)

0. **Ground rules first**: Blender CLI generators; only the three.js look counts; everything parametric and seeded;
   the page must run on the owner's phone before it is shown; lossy optimisation last; house rules in every brief.
1. **Foundations** (~2 h): toolkit + generated API doc, the `tsq` CLI, server, review-render page and GPU render
   daemon, gallery, telemetry, the two rubrics, and one hand-polished exemplar asset to >= 85.
2. **Catalogue**: 100+ briefs with class, mount, tier, dependencies, the reference crop and a CONTRACT paragraph
   (layout-facing param names, placement planes, sockets). Hero pieces at true size, never scaled.
3. **Materials**: author + judge subagents in parallel from the start; counted units per tile, height first.
4. **Asset waves**: an 8-asset pilot, then tier 1, then tier 2 once components exist; fresh judge per round;
   revision briefs quote the judge's numbered problems. Expect 67-74 on the first round and one revision to pass.
   Release pin: the scene builds only from `release/` copies of settled sources.
5. **Layout**: rules and seeds with keep-outs, sight cones and audits; judge on walk-level captures; fan out fixes
   by file ownership; fix rules, not instances. The reference scene went 66 -> 81 in five rounds.
6. **Performance**: measure (draw calls and texture bytes were the phone killers), mitigate on main the same hour,
   then the baked runtime on a branch with numeric targets per viewpoint and a merge gate stated beforehand.
   Higher presets are built *up* from the baked lowest one.
7. **Modelled LODs**: reducer first, authored where it fails, then a trimming pass to a numeric far-level total,
   then a hole census and repair; re-bake.
8. **Delivery**: lossless wins, device texture compression, GLB exports, the stall-resistant loader, a small
   source repo plus content-addressed data parts on GitHub Pages.
9. **Experiments behind URL flags**, each on the gallery's Demos tab, promoted only with paired captures and a
   phone session in telemetry.

## Non-negotiables (each one was learnt the hard way)

- Judge **only** three.js captures; Blender renders are private checks or a measuring instrument (a path-traced
  reference of the real scene decided the reference run's light falloff).
- A subagent's "done" is a claim: open the picture, run the check, compare the mtime before relaying it.
- Authors read exactly five things (`templates/brief-author.md`) and open their own sheets every iteration; any
  intensity guide written for a far-reaching light is 5-20x too hot for emitters near surfaces - calibrate it on
  the exemplar first (`authoring-lessons.md`).
- Vertex masks only darken; bare metals go black away from the key light; coloured weathering (moss, rust,
  verdigris, gilding) is geometry or a substituted material.
- Every light sits at a visible emissive fitting; helper lights never get a glow sprite; glow is wide and dim,
  never a sub-pixel slit; 4 mm clearance between any glowing face and whatever is behind it.
- All levels of a cell share one ground mesh and show the same light; reduced levels get a least-squares light
  refit inside the packer with the covering-face rule; the far level has no loose open-backed strips.
- TRAA on every preset, pixel ratio 1 everywhere, no "enter" button: the veil fades when the scene is ready and the
  first click takes pointer lock.
- Tune looks on one loaded page with uniforms and labelled rows; offer a softer and a stronger link; the owner
  decides on their device, especially for sharpness, contrast and the brightness of saturated emitters.
- Scratch only under the project, never `/tmp`; no wildcard paths; no `rm`/`mv`/`cp` with shell variables; one
  Chrome per agent on its own port and `BU_NAME`; nothing pushed or published without the owner's word; large data
  goes up as content-addressed parts, only what changed.

## What changes with the kind of environment

- **Daylit scenes**: the bake's key light is the sun (directional, with sky fill); the lamp list may be empty, so
  the bake's light-driven topology splitting and the ground shadow map follow the sun's shadow edges instead of
  lamp pools; bloom, lens flare and glow sprites matter less, exposure and sky much more. The HDR emitter expansion
  is for the sun's highlights and the sky, not mantles.
- **Interiors**: smaller cells (4-8 m), occlusion from walls makes the visibility sets far more effective, ceilings
  need the same "what lies under a detail layer" rule as roofs, and the review rig's backing wall and fog scale need
  retuning for small rooms.
- **Natural scenes**: vegetation and terrain dominate; the foliage lessons (mass not cut-outs, card area
  conservation, hemisphere lighting after AO) and `bl_terrain.py` are the centre of the work; the layout's
  "intentional, not cluttered" rules become grouping, paths and clearings.
- **Wet, dusty or hazy atmospheres**: the one-fog-path rule and the scatter reader apply unchanged; reflections
  (wet ground, water, polished floors) were the single biggest realism gain in the reference run and will be in
  most environments that have a reflective ground.
- **Anything with moving things**: the bake is static; moving parts swing about declared pivots, and dynamic
  objects need the opt-in BVH or a dynamic preset. The pipeline does not light dynamic objects from the bake.

## Scale and cost to plan for

One lead, up to ~10 subagents, <= 4 Blender processes, one shared GPU. About 32,500 model calls and ~$6,900 at
Fable 5.1 rates for the whole reference run; the asset blitz was 72 % of it and subagents 91 %. Fewer, longer-lived
authors on a 1-hour cache and a hard reading-list bound are the levers. Bakes take 10-20 min per half plus 20 min of
visibility sets; a full scene rebuild 45 min; an authored far level ~50 min of author time; a judge round minutes.
