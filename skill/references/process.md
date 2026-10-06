# Process: from brief to a walkable, judged, baked scene

The order below is the one that worked, with the corrections the reference run paid for folded in. Times are
from that run (one lead agent, up to ~10 subagents, an RTX 4090 shared with the owner's other GPU work, Blender 4.1
CLI, three.js r186 WebGPU). It took four days and about 32,000 model calls; most of the cost and most of the
rework was in the first day's asset blitz, which is where this process saves the most.

## 0. Fix the ground rules in the first ten minutes

Before any probe or prototype, settle with the owner (or assume, and say so):

- **Modelling route**: Blender on the command line from Python generators (`blender --background
  --factory-startup --python ...`). Not an interactive Blender or MCP session, not geometry made in three.js.
- **What counts**: only the real-time three.js look. Every review image, gallery image and judged shot is a capture
  from the three.js page. Blender renders are private sanity checks or measuring instruments (see §8).
- **Everything parametric**: each asset is a generator with typed params, a seed and 3-5 named variants; the scene is
  a rule-and-seed layout generator. No hand-placed one-offs anywhere.
- **The live page must run on the owner's phone** before it is shown. Decide the baked format and presets before
  composing the full scene (§6), not after the first phone crash.
- **Lossy optimisation last.** Lossless wins first (gzip, meshopt without quantisation, bundling); texture
  compression and Draco only once the scene is judged right and the owner lifts the hold.
- **House rules for every agent** (paste into every brief; `templates/brief-specialist.md` has them): scratch only in
  `<project>/.scratch/<task>/`, never `/tmp`; no wildcard paths and no `rm`/`mv`/`cp` with shell variables; one
  Chrome per agent on its own port, one tab, closed after; keep GPU and RAM low; nothing pushed or published
  without the owner's word.
- **Use git from the first commit.** The reference run was not a repository for three days and imitated branches
  with directory copies (`web/next/`, `web/perf/`, `premerge/`). It worked, but cost care.

## 1. Foundations (about 2 hours before any asset)

In this order, each verified before the next:

1. Repo scaffold: `assets/ materials/ tk/ tools/ web/ docs/ reviews/ out/ .scratch/`, `package.json` (three,
   three-mesh-bvh; meshoptimizer, draco3dgltf as dev deps), `npm install`.
2. The toolkit `tk/` (already written: `project/tk/`) and its generated reference `docs/API.md`
   (`python3 tools/gen_api.py`). Smoke-test it with one asset that exercises every builder once and inspect the
   GLB sidecar (materials, UVs, masks, sockets, parts).
3. `tools/tsq` (the one CLI authors use) and `docs/MATERIALS.md`; launch material author + judge subagents at once
   (the reference run: 9 families, 45 materials) - they run while the rest is built.
4. Static server (`tools/serve.py`: gzip, Range 206, `/api/manifest`, render upload, telemetry), the review-render
   page (`web/render.html`), the GPU render daemon (`tools/renderd.sh` on its **own** Chrome port and profile, never
   the machine's shared Chrome). Validate tone mapping numerically: an 18 % grey plane must land at pixel 128.
5. The gallery (`web/gallery/`, self-refreshing from `/api/manifest`), open through the owner's public URL.
   Telemetry (`web/engine/telemetry.js`, `tools/telemetry.py`) goes in now too: the owner's phone is where the
   scene fails and you cannot see it otherwise.
6. The two quality guides (`docs/QUALITY.md`, `docs/TOWN_QUALITY.md` - the scene one is written after the first
   full scene render so it can name real shots) and `docs/AUTHORING.md`.
7. **One hand-polished exemplar asset** composed from a component (reference run: a gas lamp post composing a
   lantern head; pick whatever carries your environment's light and detail),
   iterated to >= 85 before any author copies it. About an hour.

## 2. Catalogue (the ideation step)

`tools/catalog_src.py` -> `docs/catalog.json`: 130+ entries, each with `id`, `category`, `name`, `class` (sets the
triangle budget), `mount`, `tier` (0 = components others compose with `use()`, 1 = props and architecture, 2 =
buildings and landmarks that compose tier 1), `deps`, `ref` (which reference crop it answers to) and a one-paragraph
`brief` that says what it is, how it is constructed, its params and variants, and what to compose. Add a
**CONTRACT** paragraph wherever the scene will rely on it: layout-facing param names, placement planes ("coping top at
z = 0, water-side edge at y = 0"), socket roles. Its absence cost two judge problems per asset that lacked it.

Derive the list from the reference boards zone by zone (for a town: entrance street, gated plaza, fountain court,
cafe terrace, quay; for a harbour: quays, sheds, slipway, breakwater; for an interior: rooms, passages, fittings):
light sources, furniture, architecture or terrain components (openings, roofs or ceilings, edges, steps), the
large pieces that compose them, hero pieces, decor and clutter, vegetation, backdrop. Every hero piece the
reference makes big needs **its own asset at true size** - a scaled-up smaller asset is caught at once by the judge
(in the reference run: 1.5-2 m stones and 2.5 m lanterns on a 3x gate; cap <= 68).

`tools/tsq brief <id>` prints an entry; keep entries in the source file, never only in the generated JSON (a
regeneration silently dropped one added by hand).

## 3. Asset waves

- **Pilot first**: 8 varied assets spanning the classes (a light fitting, a seat, an opening, a container, a sign, a
  guard rail, a plant, a water or hero feature) through author -> render -> judge -> revise, to validate the guides.
  Only then fan out.
- **Wave A**: all tier-1 assets (reference: 108), one author subagent each, bounded by the Blender cap (<= 4
  Blender processes machine-wide; `tsq` queues renders through the one daemon). **Wave B**: tier-2 buildings and
  landmarks once their components exist on disk, `maxRounds: 4`.
- **Judge** every asset with a fresh subagent (never the author), `templates/brief-judge.md`. Revision briefs quote
  the judge's numbered problems verbatim and say "read the existing file and the current renders first; skip the
  exemplar". Most first rounds land at 67-74 (authors self-estimate 78-84); one revision usually clears 75.
- Expect: 136 assets, 133 passing, in about 20 hours of wall clock with ~10 authors in flight. Buildings took 4.5 h
  for 15 assets. See `authoring-lessons.md` for what makes the first round pass more often.
- Keep author context small: the strict reading list in `templates/brief-author.md`, bounded command output,
  `tsq api <fn>` lookups instead of reading toolkit source. Authors return a 5-10 line report with a "toolkit
  problems" slot - that slot is where most of the toolkit fixes came from.
- **Release pin**: the scene builds from `release/` copies (`tools/town.py release --hold <live authors>`), never from
  a file whose author is still active. Two judged scene renders were spoiled in one hour by a half-saved module and
  leaked debug variants before this existed. Releasing is a step of every rebuild; the build prints how many
  working files differ from their released copies.

## 4. Layout (the composed scene)

- `town/layout.py`: a rule-and-seed generator (`generate(seed, **overrides)`, a `Town` class with rows, runs, rings,
  scatters, attachments, keep-outs, sight cones, an audit). `tools/town.py all` = gen + build + terrain + resolve;
  `tools/town.py render` captures the judged viewpoints from the same page a visitor walks. See `composition.md`.
- Judge with `templates/brief-scene-judge.md`. Reference trajectory: 66 -> 68 -> 74 -> 68 (one floating weathervane
  capped a 75) -> 81, five rounds in about eleven hours.
- After each round, fan out by **file ownership**, not by problem: layout agent (only one allowed to run
  `town.py all`), engine look agent, 2-4 asset-fix agents with exactly named files, each brief quoting the judge
  verbatim with shot names and pixel positions, each told who else is editing what. Water geometry and water
  shading were split across two agents with the vertex-attribute contract written as a comment in the shader.
- Before spending a judge: run the automated audits (nothing on a stair or route, no bounding-box intersections,
  every fitting attached to something, no camera inside geometry). One cap item sinks a round.
- The owner walks the scene on their own devices and reports by eye. Reproduce first (same link, preset, device;
  ask which element: the fitting, its pool on the ground, its reflection, a window), explain the cause in a paragraph, then fix the **rule** that
  produced the defect so the class cannot recur ("nothing is deleted by instance index"). Rule fixes found four
  more defects nobody had reported.

## 5. Performance: measure, then bake

- Measure where time and memory go first (`tools/measure.py`, `tools/perf_*.py`; GPU timestamps with the GPU kept
  busy). In the reference run the phone killers were draw calls (~3,500) and texture bytes (108 MB of 2048² maps
  shrunk to 512 on the device), not triangles.
- Mitigate on main the same hour (default to a light preset chosen by `touch || !navigator.gpu || deviceMemory <=
  4`; cap textures; rename the remembered-preset key so an old "high" cannot come back), then build the baked
  runtime on a branch with **numeric targets per viewpoint** (`bake-runtime.md`) and a merge gate stated before the
  result ("Low/Phone within 5 % of live at 4x CPU throttle").
- The user's words after the first baked Low: "works FANTASTICALLY ... We should be starting from there in a branch
  to make Medium and High look better and better." Medium and High are built **up** from the baked Low (finer
  levels, sharper textures, per-pixel relief, reflections), never down from the dynamic Studio renderer.

## 6. Modelled levels of detail, then re-bake

Reducer-first with authoring where it fails (`lod.md`, `docs/LOD.md`). Then a **trimming pass with a numeric
target**: modelled far levels came out 33 % heavier than the decimated ones until an agent worked down a sorted
growth list to a scene-wide LOD2 total. Then a hole census and repair. Then re-bake.

## 7. Delivery

Lossless first; texture compression on the device; GLB exports (exact for proof, portable for Blender); the loader
that survives dead connections; publish as a small source repo plus content-addressed data parts (`delivery.md`).

## 8. Experiments behind flags

Every experiment is a URL flag on a preview copy, linked from the gallery's Demos tab, promoted only with evidence
(paired captures, numbers, a phone session in telemetry). Look changes are tuned on **one loaded page** by setting
uniforms from the console and capturing a labelled row per variant, never by reloading per variant.

## Orchestration rules that kept it running

- One owner per file. Briefs name the files an agent may edit and everyone else's it must not; cross-owner changes
  are delivered as "replace X with Y" edits where each X occurs exactly once.
- Caps and locks: <= 4 Blender processes (the bake's lighting workers are forks of one Blender, so count processes,
  not cores); `BAKING` / `REBUILDING` lock files; bake into a staging directory and swap, other agents read the
  live output; a lock the lead left behind is the lead's to remove.
- A subagent's "done" is a claim. Open the image, run the check, compare the sidecar mtime, before relaying it.
  Two agents reported passing image gates on torn clock dials and shards; the lead's own look at six sheets caught
  them.
- Keep "measured" and "estimated" apart in everything told to the owner. Report numbers with their conditions
  (device, preset, link speed, runs, once or paired).
- The lead's context compacts every 10-30 minutes on a day like this. Keep orchestration state on disk (wave
  arguments, the live-author list, lock files, scratch scripts such as `shot.py`, `tune.py`, `cmp.py`) so a resumed
  lead re-derives it. Update the replication notes at the end of every feature, before reporting.
- Research questions with their own measurement loop get their own agent; do not queue them behind production
  work (the vertex-light refit sat third in a baking agent's queue with nothing measured until it was split out).
- Compaction loses a subagent's final report; after resuming an agent, ask it to restate its result.
- Status to the owner: a short table (agent, state, where it is) read from files and processes, labelled as such;
  three bullets while jobs run (done / running with minutes left / what lands next); push notification when a long
  task ends or a question blocks.
- Decisions for the owner go as a **numbered list with a full URL per option** ("now" vs "trial"); they answer by
  number. Offer a softer and a stronger link beside every new look setting; the owner picks on their device.

## Definition of done (reference numbers)

- >= 100 unique parametric assets at >= 75 from an independent judge on three.js renders (reference: 136 in 7
  categories, 133 passing; 57 procedural materials).
- A composed scene at >= 75 on walk-level three.js captures (reference: 81, 14 judged viewpoints, ~530 items).
- A walkable page with baked presets `phone | low | medium | high` and the dynamic `studio` as opt-in; the finished
  page opens on `high` everywhere only after it is proven on the owner's phone (60 fps, no throttling in
  telemetry); no "Walk in" button - the veil fades when the scene is ready and the first click takes pointer lock.
- A live gallery with a Demos tab; every URL handed over is one the owner can open (proxied form, checked).
- Telemetry that shows real visits; a source zip; publishing that re-uploads only what changed.
