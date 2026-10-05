# Authoring materials (procedural PBR textures)

The town square is a **realistic night scene lit by warm lanterns**. Every surface in it uses one of
the shared tiling PBR materials in `docs/materials.json`. Each *baked* material is a small numpy
program `materials/<name>.py` that synthesises a seamlessly tiling texture set. There are no
photographs and no downloads: realism has to come from careful observation of how the real
material is built up, layer by layer.

## The file

```python
from tk.tex import *          # numpy as np, Canvas helpers, colour helpers (see docs/API_TEX.md)

META = {"normal_strength": 1.0}        # optional overrides: tile, res, normal_strength, ao_radius, ao_strength

def build(t):                           # t: Canvas covering ONE tile, t.W x t.H metres, t.w x t.h px
    ...
    return dict(albedo=rgb, height=h, rough=r, metal=m)      # optional: ao=
```

* `albedo`  (h,w,3) **linear** RGB base colour, no lighting / shadows baked in (a touch of cavity
  darkening for dirt is fine - that is dirt, not shading).
* `height`  (h,w) in **metres** (1 mm = 0.001). The normal map and the texture AO are derived from
  it with physically correct slopes, so use true depths: brick joints 3-8 mm, cobble crowns
  10-25 mm, wood grain 0.1-0.6 mm, cast-iron texture 0.2-0.5 mm, weave 0.5-2 mm.
* `rough`   (h,w) or scalar 0..1.  `metal` (h,w) or scalar 0..1.
* Tile size and resolution come from the catalog entry (`tile` in metres, `res`); only override
  them in META with a good reason (e.g. a brick module that must divide the tile).
* Everything wraps: every `t.*` op is periodic, so the result tiles as long as you only build
  from `t.*` ops and per-pixel maths. Never use absolute pixel positions / np.linspace ramps
  across the tile, and never use unseeded randomness.

Bake + look (takes seconds; bakes at catalog res unless `--res` is given):

```bash
tools/tsq mat <name> --res 512      # fast iteration
tools/tsq mat <name>                # final, full resolution  (ALWAYS finish with this)
```

then **Read the images** it lists (you are judged on these):

* `out/mat/<name>/preview.png` - 2x2 tiles, raking daylight: identity, relief, seams.
* `out/mat/<name>/far.png`     - ~7 tiles receding under one warm lantern + faint moonlight: this
  is how the town is actually seen. Shows repetition and whether gloss/relief catch the light.
* `out/mat/<name>/sheet.png`   - albedo | normal | roughness | height | AO.

Exemplars for API usage only (they are NOT the quality bar - they are first drafts and need the
same improvement as everything else): `materials/cobble_granite.py`, `materials/wood_oak.py`,
`materials/iron_wrought.py`.

## What "good" means here (the judge's rubric, 100 points, pass = 75)

1. **Identity & realism (30)** - at a glance it *is* that material at its true physical scale
   (the tile size in metres is given: a 225 mm brick must be 225 mm). Built the way the real
   thing is built: stones are laid, tiles overlap, boards are sawn, paint is brushed on.
2. **Multi-scale detail & relief (20)** - three scales present at once: macro (patchy colour /
   damp / wear over 0.3-1 m), meso (the units: stones, boards, hammer facets, weave) and micro
   (grain, pores, grit). The height field carries real relief so that a low lantern rakes across it.
3. **Colour & PBR plausibility (20)** - measured-looking values. Albedo luminance (linear):
   charcoal/soot 0.02-0.04, dark wet stone 0.05-0.1, old brick 0.12-0.25, weathered wood
   0.15-0.3, limestone / plaster 0.4-0.65, fresh whitewash <= 0.8. Nothing pure black/white.
   Low saturation: real materials are greyer than you think. Roughness varies *spatially* with
   meaning (polished crowns smoother, dirt-filled joints rougher, worn handles shinier). Metal
   is 1.0 only for bare metal; paint, rust, patina, dirt are metal 0.
4. **Tiling (15)** - in `far.png` no feature may read as a repeating grid: no single unique
   stone / knot / stain that the eye can count across the repeats. Spread distinctive features
   evenly, keep large-scale contrast low (the engine adds its own world-space macro variation,
   grime and wear on top), and make sure joints/courses line up across the tile edge.
5. **Night read (15)** - under the warm lantern in `far.png` the surface should look rich: relief
   visible, specular breakup from the roughness map, not flat, not noisy, not plastic.

Common failures the judge will mark down: Voronoi cells that look like cracked mud instead of
stones; uniform noise smeared over everything ("procedural soup"); joints as thin dark lines
with no depth; features far too big or too small for the stated tile size; saturated cartoon
colours; flat roughness; detail only in albedo with a flat normal; visible periodic banding;
height spikes giving a noisy "sandpaper" normal; unreadable blobs.

## Conventions the engine relies on

* **Direction**: U (image X) is "along": grain runs along U for all woods; rope runs along U;
  courses of brick / roof tiles / shingles / slates are horizontal rows with **eaves / bottom at
  V = 0** (image row 0 is the TOP of the texture, V = 1). `bark` ridges run along V.
* **Tintable materials** (`"tintable": true` in the catalog): author them in the natural default
  colour described in the catalog. The engine re-colours with `albedo * tint / mean_albedo`, so
  the mean colour is what gets replaced; keep hue variation moderate and *relative* (darker/
  lighter, slightly warmer/cooler) so the surface still makes sense painted green or blue.
  `wood_painted`, `iron_painted`, `plaster_*`, `canvas`, `ceramic_glazed` are routinely tinted
  to strong colours: their default should be a mid-light neutral-ish colour.
* **Weathering is layered on by the engine** from per-vertex masks (grime in crevices and near
  the ground, wear on exposed edges, a material-specific "special" such as moss / rust /
  verdigris / wetness). So the baked texture should be the material in decent, *moderately*
  aged condition with fine-scale ageing (pores, small chips, subtle stains) - not caked in
  large-scale dirt streaks, which the engine places where they belong.
* Layout units must divide the tile (`t.grid` / `t.stripes` snap for you - check the result
  against the description: e.g. 4 brick lengths x 12 courses per 0.9 m tile).
* Performance: a 2048 bake should stay under ~60 s; iterate at `--res 512`.

## Technique notes

* Start from the **layout** (`t.grid`, `t.cells`, `t.stripes`) -> build the **height** first as
  a sum of physically-meant layers (unit shape via `lay.dome()` / `lay.rounded()`, per-unit
  `lay.tilt()` and `lay.rand()` level differences, surface texture, chips at edges via
  `t.edges` x noise, recessed joints with a slightly rough mortar bed) -> then derive masks
  from the height (`t.cavity`, `t.edges`, `t.curv`, `t.ao`) -> then colour with `ramp()` through
  3-6 observed colours driven by noise, plus per-unit variation, plus cavity dirt and edge
  wear -> then roughness from the same masks.
* Break mechanical regularity with `t.warp()` (a few mm) before thresholding; vary every unit
  (hue, value, height, tilt) with `lay.rand(k)`.
* Use `stretch=` noise for anything directional (grain, brushing, drips, rolled metal).
* Keep per-pixel white noise very low in the height (it turns to sparkle under the lantern);
  build grit from `t.speckle` / high-octave `t.fbm` instead.
* Check the numbers `tsq mat` prints: height range, mean albedo, mean roughness.

## Rules

* Only create / edit your own `materials/<name>.py` files. Do not edit `tk/`, `tools/`, `docs/`.
  If the toolkit is missing something you need, implement it locally in your material file
  (plain numpy on `t.X`, `t.Y`, `t.px` ... is fine as long as it wraps).
* Do not start browsers or servers. Do not download anything.
