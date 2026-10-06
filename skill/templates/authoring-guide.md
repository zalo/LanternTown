# Authoring a parametric asset (read fully before writing code) - theme-neutral template

Fill the `<...>` fields and save as `docs/AUTHORING.md`. Keep the structure: authors are briefed to read exactly
this file, `docs/API.md`, one exemplar, the material table and one reference crop. The reference run's filled-in
version is `references/docs/AUTHORING.md`; the measured corrections to it are in `references/authoring-lessons.md`
(read them before filling in the light-intensity table below).

---

You are a senior 3D modeller writing ONE parametric asset generator for `<a realistic <environment>, seen <how: in
first person at walking pace>, under <the light>>`. The model is built by **Blender on the command line** from your
Python file and rendered in the real-time three.js engine. A separate judge scores the renders against
`docs/QUALITY.md`; you iterate until it passes (>= 75 / 100).

## Reading budget (strict - context is precious; ~15 minutes of reading, then model)

Read exactly these, once, with the Read tool (not `cat`), and nothing else up front:
1. this file,
2. `docs/API.md` (the complete modelling toolkit: every function, signature and behaviour),
3. ONE exemplar: `<assets/<category>/<exemplar>.py>` (composes `<component>` with `use()`),
4. `tools/tsq matlist` (compact material table; `tools/tsq matlist wood` filters),
5. the ONE reference crop named in your brief (`docs/ref/<crop>.png`). The reference is stylised concept art: take
   its *design language, proportions and mood* but build it as a real, physically plausible object with believable
   construction and materials.

The judge's sheet (no need to read it): silhouette / proportion / scale 20, geometric detail and construction 20,
materials and weathering 20, read under the environment's light 20, parametric variation 10, technical cleanliness
10; pass >= 75. Hard caps: does not read as the briefed object or grossly out of scale <= 50; visibly broken geometry
/ shading <= 60; a broken variant or look-alike variants <= 70; blown-out lighting or invisible <= 70. The bar is a
hero prop in a current AAA game seen from 1.5 m to 30 m.

Do NOT open `tk/*.py`, `tools/*`, `web/*`, `docs/materials.json`, `docs/catalog.json`, other docs or other assets'
sources "to understand the system" - this file and API.md are complete. Targeted lookups when you actually need
them: `tools/tsq api sweep,lathe`, `tools/tsq assets` (ids available to `use()`), `tools/tsq brief <id>` (another
asset's contract). Keep command output short; never dump whole files.

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
VARIANTS = {"ornate_wide": {"width": 1.8, "style": "ornate", "seed": 7}}   # "default" always exists implicitly

def build(p, rng):                  # p.width, p.style, p.seed ...; rng = seeded Rng
    ...                             # create mesh objects; everything left in the scene is exported
```

* Units are **metres**, Z up, the **front faces -Y**, rotations are Euler degrees.
* `META["mount"]`: `ground` (origin at the centre of the footprint, on the ground), `wall` (origin at the fixing
  point; the wall is y = 0 and the asset projects to -Y), `wall_ground`, `hang` (origin is the suspension point).
* `META["class"]` sets the triangle budget: `<class: budget>` ... (override with `"tri_budget"` only with good
  reason; over budget = warning, 2x = error). Spend triangles where the silhouette and the close-up need them.
* Optional META: `"focus"` (where the close-up camera looks), `"mount_height"`, `"allow_below_ground": True` for
  things that go below the ground plane, `"auto_grime"` / `"auto_wear"` (0..1, the automatic weathering gains).
* **Everything is driven by PARAMS and the seed.** No magic one-off placement: counts, dimensions, style switches,
  colours, text, wear amounts are parameters; irregularities come from `rng` (`rng.fork("name")` gives a stable
  sub-stream). Every param value inside its min/max must give a sensible object. Provide 3-5 `VARIANTS` that look
  clearly different in shape, size or colour (not tint only).
* If the brief contains a CONTRACT, its param names, origin and extents are mandatory.

## Build, render, look, iterate

```
tools/tsq build <asset_id>                 # Blender CLI -> out/glb/<id>/<variant>.glb (+ stats, warnings)
tools/tsq render <asset_id>                # build + render -> out/renders/<id>/{sheet,hero,variants}.png
```

* **Look at the renders with the Read tool** after rendering and criticise them like the judge would: `sheet.png`
  every iteration; `hero.png` and `variants.png` on the first and the last iteration and whenever you changed
  lights or variants. The daylight sheet shows only the default variant large: swap another variant into `default`
  to inspect it, and swap back. Expect 3-6 iterations. Do not stop at "it builds". Fix every WARN.
* Stress-build the parameter extremes (`tools/tsq build <id> --params '{...}' --name tmp`) and delete the `tmp.*`
  files afterwards.

## Modelling standard (what makes it pass)

* **Real dimensions**: `<the environment's anchors: seat 0.45 m, door 2.0-2.2 x 0.9 m, storey 2.8-3.2 m, step 0.17 x
  0.30, handrail 0.95-1.1 m, brick 215 x 65 mm, plank 20-30 mm ...>`.
* **Build it the way it is made**: separate boards with gaps, straps with rivets, glazing bars holding panes,
  brackets that reach the wall, feet touching the ground, mortar and joints as geometry. Nothing paper-thin.
* **Three scales of detail**: silhouette; mouldings, panels, brackets, bands, caps and bases (a bare primitive
  standing in for a moulded part will be noticed); bevelled edges, fixings and seeded irregularity.
* **Materials** come only from the library (`mat("wood_oak")`, `mat("iron_painted", "#1b2622")`); 3-8 per asset
  with believable contrast; declare grain and course direction. Substitutions that read under the environment's
  light: `<fill from references/authoring-lessons.md §3 for your light: e.g. gilding as metal_plain gold rough 0.4-0.6
  or dielectric gold paint; iron as grey-brown iron_painted; moss, rust and verdigris as geometry or a second
  material>`.
* **Weathering tells a story** with `paint(obj, grime=, wear=, special=)`, placed by cause at vertex resolution
  (streaks under drips and fixings, splash at the foot, polish where hands and feet touch, `<the environment's
  weathering>`). The masks are thresholded and only darken: keep the art-directed part narrow and strong, re-mesh
  big faces on a 0.3-0.4 m grid before painting, never overlap solids (a buried vertex bakes black), keep geometry at
  z >= 0. Automatic cavity grime, edge wear and AO are added on export.
* **Light.** Anything that glows needs (a) a visible emissive source mesh (`mat("emissive", "<colour>", glow=...)`,
  lit glass with `glow`, `window_lit` with an explicit tint) and (b) a `light("name", loc, intensity=cd, range=m,
  shadow_near=...)` socket **at the fitting**. Intensities measured in this engine for emitters near surfaces (the
  rig has no soft near field, so anything pale within 0.4 m of a light burns to white): `<fill from
  references/authoring-lessons.md §2: candle 0.1-0.5 cd, wall fitting beside a wall 1.5-4.5 (~8 x reach²), sign lamp
  2-3, room spill 6-14 inside the room, free-standing lantern 5-9, tall post head 40-60 ...>`. Default emitter colour
  `<#ffbf6a for flame; 2700-3000 K for tungsten; the tube colour for neon>`. Glowing meshes never cast shadows; set
  `shadow_near` just past the fitting. In `hero.png` the lit ground should be `<rich amber / warm / neutral>`, not
  white.
* **Moving parts**: `part(objs, "sign", pivot=..., motion="swing")`, `"spin"`, `"flutter"`, with an explicit pivot
  and axis. **Sockets** (`socket("name", "attach", loc, rot, role=...)`) mark where other assets attach.
* **Composition**: `use("<component>", width=0.9, seed=p.seed + 3).place(...)`; measure with `.size`, `.min`,
  `.max`, `.sockets`. Components cost `<7-37 k triangles and 3-11 materials each>`; the <= 10 material cap counts
  them. Only `use()` assets that exist (`tools/tsq assets`); otherwise model the part inline.
* **Text** via `text(name, "...", size=, depth=, font=)` - raised, engraved or painted; `size` is the em size
  (cap height ~0.55 x); ~150-500 triangles per glyph, so single-stroke lettering for signs. All text is a PARAM.
* **Performance**: `instances_to_mesh` for repeated small parts; 6-7 segment lathe domes instead of `bolt()`;
  `sweep(smooth_path=False)` for rods; no hidden interior geometry; within budget.

## Rules

* Create / edit only your own `assets/<category>/<asset_id>.py`. Do not edit `tk/`, `tools/`, `docs/`, `web/`,
  other assets, or anything under `out/` by hand.
* Use only `tools/tsq build` / `tools/tsq render`. Never start Blender GUIs, browsers, servers or MCP tools; never
  download models or textures - everything is procedural.
* If the toolkit lacks something, implement it locally in your file. If you hit a toolkit *bug*, work around it and
  describe it in your final message.
* Final message: 5-10 lines - what you built, params / variants, triangle counts, toolkit problems, what the next
  round should know.

## Scratch files

Never write scratch to `/tmp` or `/dev/shm`. Use `.scratch/<asset_id>/` in the project root and delete it when you
finish. Do not use wildcard paths in shell commands.
