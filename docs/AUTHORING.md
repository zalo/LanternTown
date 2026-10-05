# Authoring a parametric asset (read fully before writing code)

You are a senior 3D modeller writing ONE parametric asset generator for a realistic, lantern-lit
old-town square (think of a storybook game town rebuilt with real materials and real construction, at night).
The model is built by **Blender on the command line** from your Python file and rendered in the
real-time three.js engine. A separate judge scores the renders against `docs/QUALITY.md`;
you iterate until it passes (>= 75 / 100).

## Reading budget (strict - context is precious; ~15 minutes of reading, then model)

Read exactly these, once, with the Read tool (not `cat`), and nothing else up front:
1. this file,
2. `docs/API.md` (the complete modelling toolkit: every function, signature and behaviour),
3. ONE exemplar: `assets/lighting/lamp_post_gas.py` (composes `lantern_head` with `use()`),
4. `tools/tsq matlist` (compact material table; `tools/tsq matlist wood` filters),
5. the ONE reference crop named in your brief (`docs/ref/<crop>.png`). The reference is stylised
   concept art: take its *design language, proportions and mood* but build it as a real,
   physically plausible object with believable construction and materials.

The judge's sheet (`docs/QUALITY.md`, no need to read it): silhouette / proportion / scale 20,
geometric detail and construction 20, materials and weathering 20, night read and lighting 20,
parametric variation 10, technical cleanliness 10; pass >= 75. Hard caps: does not read as the
briefed object or grossly out of scale <= 50; visibly broken geometry / shading <= 60; a broken
variant or look-alike variants <= 70; blown-out lighting or invisible at night <= 70. The bar is a
hero prop in a current AAA game seen from 1.5 m to 30 m.

Do NOT open `tk/*.py`, `tools/*`, `web/*`, `docs/materials.json`, `docs/catalog.json`,
`docs/ref/board.png`, other docs, or other assets' sources "to understand the system" - this file
and API.md are complete. Targeted lookups when you actually need them:
`tools/tsq api sweep,lathe` (source of single toolkit functions), `tools/tsq assets` (ids available
to `use()`), `tools/tsq brief <id>` (another asset's contract, e.g. its param names).
Keep command output short (`| head`, `| tail -30`); never dump whole files into the terminal.

## The file

`assets/<category>/<asset_id>.py` (the id is the file name and is globally unique):

```python
"""One-paragraph description of the object and how it is constructed."""
from tk import *

META = {"name": "Human name", "class": "prop", "mount": "ground"}
PARAMS = {
    "width": {"default": 1.2, "min": 0.8, "max": 2.0, "desc": "overall width (m)"},
    "style": {"default": "plain", "choices": ["plain", "ornate"], "desc": "..."},
    "paint": {"default": "#2b4a3a", "desc": "paint colour"},
}                                   # an int "seed" param is added automatically
VARIANTS = {                        # named param overrides; "default" always exists implicitly
    "ornate_wide": {"width": 1.8, "style": "ornate", "seed": 7},
}

def build(p, rng):                  # p.width, p.style, p.seed ...; rng = seeded Rng
    ...                             # create mesh objects; everything left in the scene is exported
```

* Units are **metres**, Z up, the **front faces -Y**, rotations are Euler degrees.
* `META["mount"]`:
  * `ground` - origin at the centre of the footprint, on the ground (z = 0 is the paving).
  * `wall` - origin at the fixing point; the wall is the plane y = 0 and the asset projects to -Y.
  * `wall_ground` - stands on the ground against a wall (wall plane y = 0, ground z = 0).
  * `hang` - origin is the suspension point; geometry hangs below (negative z).
  A `mount` PARAM with one of these values overrides META per variant.
* `META["class"]` sets the triangle budget: `prop_small` 6k, `prop` 16k, `component` 12k,
  `furniture` 22k, `street_furniture` 26k, `structure` 70k, `building` 160k, `landmark` 260k,
  `vegetation` 60k, `ground` 80k (override with `"tri_budget"` only with good reason). Over
  budget = warning; 2x over = error. Spend triangles where the silhouette and the close-up need
  them: curved profiles 16-32 segments, small rods 6-10 sides, hidden faces removed.
* Optional META: `"focus"` (height fraction 0..1 or `[x, y, z]`) = where the night close-up camera
  looks (aim it at the most detailed / lit part); `"mount_height"` (m) for wall / hanging assets
  in the review renders; `"allow_below_ground": True` for wells, canal walls, foundations.
* **Everything is driven by PARAMS and the seed.** No magic one-off placement: counts,
  dimensions, style switches, colours, text, wear amounts are parameters; irregularities (plank
  warp, stone sizes, leaning, missing pieces) come from `rng` (`rng.fork("name")` gives a stable
  sub-stream). Every param value inside its min/max must give a sensible object.
  Provide 3-5 `VARIANTS` that look clearly different (size / style / colour / condition).

## Build, render, look, iterate

```
tools/tsq build <asset_id>                 # Blender CLI -> out/glb/<id>/<variant>.glb (+ stats, warnings)
tools/tsq render <asset_id>                # build + render -> out/renders/<id>/{sheet,hero,variants}.png
```

* `print("DBG ...")` lines from your file are shown by `tools/tsq build`; a Python error prints the traceback.
* **Look at the renders with the Read tool** after rendering and criticise them like the judge
  would: `sheet.png` every iteration (daylight front / back, night close-up, clay with a 1.8 m
  figure for scale); `hero.png` (night, as it will be seen in town) and `variants.png` on the
  first and the last iteration and whenever you changed lights or variants.
  Fix what you see: wrong scale next to the figure, flat or blobby shapes, missing thickness,
  floating / intersecting parts, flipped normals (black or see-through faces), stretched or
  mis-scaled textures, grain running the wrong way, lights that are too hot or too weak.
* Expect 3-6 build/render iterations. Do not stop at "it builds".
* Fix every WARN the build prints or have a real reason. What the build checks: every object has
  a `tk.mat()` material (unknown library name / hand-made bpy material = error); triangles vs the
  class budget (> 2x = error; "far too simple" below a few hundred); <= 10 material variants and
  <= 6 light sockets; ground assets touch z = 0 and do not go below it (unless
  `allow_below_ground`); wall assets stay in front of y = 0; overall size plausible for the class.
* On export everything is joined per material into one mesh (plus one per `part()`), UVs /
  smoothing declared with `uv_*` / `smooth` are resolved, automatic cavity grime + edge wear + AO
  are baked into vertex colours, and LODs are made by the town build - you do not handle any of that.

## Modelling standard (what makes it pass)

* **Real dimensions.** Seat 0.45 m, table 0.72 m, door 2.0-2.2 x 0.9 m, storey 2.8-3.2 m, handrail
  0.95-1.1 m, step 0.17 rise x 0.30 going, brick 215 x 65 mm, roof pitch 40-55 deg, lamp post
  3.5-4.5 m, sign lettering 0.12-0.3 m, plank 20-30 mm thick, barrel 0.9 m tall.
* **Build it the way it is made.** Separate boards with gaps, mortise posts and rails, iron straps
  with rivets, hinges with pins, stone blocks with joints and chamfers, glazing bars holding
  panes, brackets that actually reach the wall, feet touching the ground, fixings (bolts, nails,
  collars) where parts meet. Nothing paper-thin: sheet metal >= 2 mm, glass has a frame, cloth
  has folds and a hem, planks have thickness and end grain.
* **Three scales of detail**: primary silhouette (interesting, asymmetric where natural), secondary
  forms (mouldings, panels, brackets, bands, caps and bases), tertiary (bevelled edges everywhere -
  `bevel=` / `bevel()`; bolts, nail heads, chips, dents, sag, slight misalignment from `rng`).
  A hard, perfectly sharp CG edge is a fail; so is a perfectly regular array.
* **Materials** come only from the library: `mat("wood_oak")`, `mat("iron_painted", "#1b2622")`,
  `mat("brass", rough=0.8)`. Pick the most specific material; `plain` / `metal_plain` are last
  resorts. Use 3-8 materials per asset with believable contrast (wood + iron + brass + glass).
  Texture scale is automatic (UVs in metres) - declare direction where it matters
  (`uv_grain(obj, axis=...)` for wood, courses horizontal for brick / tiles; lathe / sweep /
  surface already have good UVs).
* **Weathering tells a story** with `paint(obj, grime=..., wear=..., special=...)`: rain streaks and
  splash-back near the ground, soot above lanterns and chimneys, polished wear where hands and
  feet touch, rust at joints and bases (`special` on iron), moss / lichen on damp north-facing
  stone and roofs (`special` on stone, tiles), sun-fade on paint and fabric. Automatic cavity
  grime, edge wear and AO are added on export - paint the *art-directed* part.
  Night-town mood: old, cared for, lived in - not derelict, not factory-new.
* **Light.** The town is lit almost entirely by the assets themselves, so anything that glows needs
  (a) a visible emissive source mesh (`mat("emissive", "#ffcf8a", glow=20)` flame / mantle / bulb;
  `glass_lantern` / `glass_frosted` with `glow`; `neon`; `window_lit` for occupied rooms;
  `glass_stained`) and (b) a `light("name", loc, intensity=cd, range=m, shadow_near=...)` socket.
  Candela guide: candle 1-2, table oil lamp 5-12, wall lantern 25-45, street lantern 50-80,
  shop-window spill 20-50 (place the light 0.3-0.6 m in front of the glass), neon letters 6-15
  per metre of sign in the tube colour. Too hot is the common mistake: in `hero.png` the lit
  ground should be rich amber, not white. Default lantern colour `#ffbf6a`. Glowing meshes never
  cast shadows; set `shadow_near` just beyond the fitting so its cage does not throw a giant
  shadow. Give flames a `fx("flame", "flame", loc, size=...)` socket only for *open* flames
  (candles, torches, braziers).
* **Moving parts**: `part(objs, "sign", pivot=..., motion="swing")` for hanging signs and lanterns
  on chains, `"spin"` for weather vanes, `"flutter"` for banners, flags and bunting.
* **Sockets** (`socket("name", "attach", loc, rot, role=...)`) mark where other assets attach
  (lantern mount, sign bracket, flower box, door, window) - the town generator and other assets use them.
* **Composition**: build from existing catalogue assets with
  `use("window_casement", width=0.9, seed=p.seed + 3).place((x, 0, z), rot=(0, 0, 180))` - measure
  with `.size`, `.min`, `.max`, find mount points in `.sockets`. Only `use()` assets that exist in
  `assets/` (`ls assets/*/`); if a part you need is missing, model it inline in your file.
* **Text** via `text(name, "CAFE", size=0.18, depth=0.004, font="serif")` - raised, engraved (boolean)
  or painted (thin extrusion 1 mm proud in a contrasting material). All sign text must be a PARAM.
* **Performance**: join repeated small parts with `instances_to_mesh` (bolts, balusters, tiles,
  leaves); 3-8 materials; no hidden interior geometry; stay within the triangle budget.

## Rules

* Create / edit only your own `assets/<category>/<asset_id>.py`. Do not edit `tk/`, `tools/`,
  `docs/`, `web/`, other assets, or anything under `out/` by hand.
* Use only `tools/tsq build` / `tools/tsq render` (they drive the Blender CLI and the shared GPU
  render queue). Never start Blender GUIs, browsers, servers or the Blender MCP tools; never
  download models or textures - everything is procedural.
* If the toolkit lacks something, implement it locally in your file (`mesh_object`, `from_bmesh`,
  `transform_verts`, numpy and `bpy` data API are all available). If you hit a toolkit *bug*,
  work around it and describe it in your final message.
* Final message: 5-10 lines - what you built, params / variants, anything the next round should
  know. No long reports.

## Scratch files

Never write renders, crops, contact sheets or any other scratch to `/tmp` or `/dev/shm` (RAM disks with a quota). Use `.scratch/<your-asset-or-task>/` in the project root (normal disk, not served) and delete that folder when you finish.

Do not use wildcard paths (`*`, `?`, `[..]`) in shell commands: they trigger an approval prompt that stops work. Name paths explicitly or do the matching inside a python script.
