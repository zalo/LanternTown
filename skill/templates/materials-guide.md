# Authoring materials (procedural PBR textures) - theme-neutral template

Fill the `<...>` fields and save as `docs/MATERIALS.md`. The reference run's version is
`references/docs/MATERIALS.md`; the lessons are in `references/materials-lessons.md`.

---

The scene is **`<a realistic <environment> under <the light>>`**. Every surface uses one of the shared tiling PBR
materials in `docs/materials.json`. Each *baked* material is a small numpy program `materials/<name>.py` that
synthesises a seamlessly tiling texture set. No photographs, no downloads: realism comes from observing how the
real material is built up, layer by layer.

## The file

```python
from tk.tex import *          # numpy as np, Canvas helpers, colour helpers (see docs/API_TEX.md)
META = {"normal_strength": 1.0}        # optional overrides: tile, res, normal_strength, ao_radius, ao_strength
def build(t):                           # t: Canvas covering ONE tile, t.W x t.H metres, t.w x t.h px
    ...
    return dict(albedo=rgb, height=h, rough=r, metal=m)      # optional: ao=
```

* `albedo` (h,w,3) **linear** RGB, no lighting or shadows baked in. `height` (h,w) in **metres** (true depths: brick
  joints 3-8 mm, cobble crowns 10-25 mm, wood grain 0.1-0.6 mm, weave 0.5-2 mm) - the normal map and texture AO are
  derived from it. `rough`, `metal` (h,w) or scalar.
* Tile size and resolution come from the catalogue entry. Everything wraps: build only from `t.*` ops and per-pixel
  maths; never absolute pixel ramps, never unseeded randomness.

Bake + look: `tools/tsq mat <name> --res 512` to iterate, `tools/tsq mat <name>` to finish (ALWAYS). Then **read**
`out/mat/<name>/preview.png` (2 x 2 tiles, raking daylight: identity, relief, seams), `far.png` (~7 tiles receding
under `<the environment's light: one warm lamp and faint moonlight / low sun / overcast sky>`: repetition, gloss and
relief under the light), `sheet.png` (albedo | normal | roughness | height | AO).

## What "good" means (the judge's rubric, 100 points, pass 75)

1. **Identity & realism (30)** - at a glance it *is* that material at its true physical scale, built the way the real
   thing is built: stones are laid, tiles overlap, boards are sawn, paint is brushed on.
2. **Multi-scale detail & relief (20)** - macro (patchy colour, damp or wear over 0.3-1 m), meso (the units), micro
   (grain, pores, grit) present at once; the height field carries real relief.
3. **Colour & PBR plausibility (20)** - measured-looking values (linear albedo: soot 0.02-0.04, dark wet stone
   0.05-0.1, old brick 0.12-0.25, weathered wood 0.15-0.3, limestone and plaster 0.4-0.65, whitewash <= 0.8); low
   saturation; roughness varies *spatially* with meaning; metal 1.0 only for bare metal.
4. **Tiling (15)** - in `far.png` no feature reads as a repeating grid; distinctive features spread evenly;
   large-scale contrast low (the engine adds world-space macro variation, grime and wear); courses line up across
   the tile edge.
5. **Read under the environment's light (15)** - under `<the light>` in `far.png` the surface looks rich: relief
   visible, specular breakup from the roughness map, not flat, not noisy, not plastic.

Common failures: Voronoi cells for soft forms ("cracked mud"); uniform noise over everything ("procedural soup");
joints as thin dark lines with no depth; features the wrong size for the stated tile; saturated colours; flat
roughness; detail only in albedo with a flat normal; periodic banding; height spikes ("sandpaper" normal); closed
noise contours drawn as ridges ("plywood grain", "topographic map").

## Conventions the engine relies on

* U is "along": grain and rope along U; courses horizontal with the bottom edge at V = 0 (image row 0 is the top).
* **Tintable materials** are authored in their natural default colour; the engine recolours by `albedo * tint /
  mean_albedo`, so keep hue variation moderate and relative.
* **Weathering is layered on by the engine** from per-vertex masks (grime, wear, a material-specific "special"):
  author the material in decent, moderately aged condition with fine-scale ageing, not large-scale dirt.
* Layout units must divide the tile (`t.grid` / `t.stripes` snap; check the count against the description).
* Performance: a 2048 bake stays under ~60 s.

## Technique notes

Layout (`t.grid`, `t.cells`, `t.stripes`) -> height first as a sum of physically meant layers (`lay.dome()`,
`lay.rounded()`, per-unit `lay.tilt()` and `lay.rand()`, surface texture, chips at edges, recessed joints) -> masks
from the height (`t.cavity`, `t.edges`, `t.curv`, `t.ao`) -> colour with `ramp()` through 3-6 observed colours
driven by noise plus per-unit variation, cavity dirt and edge wear -> roughness from the same masks. Break
regularity with `t.warp()` (a few mm); `stretch=` noise for anything directional; grit from `t.speckle` and
high-octave `t.fbm`, never white noise. Riven surfaces as open, sub-parallel steps that pinch out, never fbm
zero-contours. Check the numbers `tsq mat` prints: height range, mean albedo, mean roughness.

## Rules

Only create / edit your own `materials/<name>.py`. Do not edit `tk/`, `tools/`, `docs/`. Implement anything missing
locally (plain numpy on `t.X`, `t.Y` is fine as long as it wraps). No browsers, servers or downloads.
