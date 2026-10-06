# Procedural PBR materials: what made them read as real

`docs/MATERIALS.md` (in `references/docs/`) is the author contract and rubric (identity 30, multi-scale relief 20,
colour and PBR plausibility 20, tiling 15, night read 15; pass 75). `docs/API_TEX.md` is the numpy toolkit. This
is what 57 materials and their judge rounds taught. First rounds scored 63-74 for anything that was not a layout
of counted units; one or two revisions passed most; three (copper patina, wood end grain, rough stone) needed a
change of approach after three rounds of tweaks.

## Construction-first wins, noise-first fails

- **Count the units per tile and lay them**: 4 x 12 bricks at 225 x 75 mm per 0.9 m tile; 8 boards of 125 mm;
  8 x 8 slates at a 150 mm gauge; physically overlapping slabs for tiles and slates; real cylinders over/under for
  wicker; z-buffered leaves for a hedge; ~21,000 dropped stones for gravel; random-rectangular bond on a 150 mm
  module with closers and no four-way joints for York flags (solved offline into a table). `t.grid`, `t.cells`,
  `t.stripes` snap to the tile; check the count against the description.
- **Riven and laminated surfaces are never the zero-contour of fbm** ("plywood grain", "topographic map", "worm
  tracks" were the judge's words for slate, flagstone and bark). Use a few open, sub-parallel delamination steps
  per unit with a sharp drop and a feathered ramp that fades along its length and pinches out; bark as lens-shaped
  strands bundled into ridges with V-section furrows.
- **Multi-scale height that reads at a lantern's raking angle**: rough stone as three levels of tilted planar flake
  scars (14 / 5 / 2 cm) with crisp arrises, grouped chisel furrows on ~20 % of the face, pits; end grain as per-ring
  anatomy (recessed pore band, latewood darkening to a sharp year line) driving albedo, height and roughness
  together. Relief too shallow reads as a flat panel in `far.png` ("quiet").
- **Build height first, then derive masks from it** (`t.cavity`, `t.edges`, `t.curv`, `t.ao`), then colour through a
  ramp of 3-6 observed colours, then roughness from the same masks by cause (polished crowns smoother,
  dirt-filled joints rougher, a 0.5 m footfall lane). Roughness that does not vary spatially is marked down.
- **Soft forms are not Voronoi**: leather, lead flats and copper sheet need rounded pillows or blurred ovals, not
  polygons ("cracked mud").

## Colour discipline

- Warm lantern light pushes everything orange: author cooler and greyer than the photograph. Desaturated
  near-uniform buff with a per-unit value spread of about +-6 % (flags); oak around `#7f6b55` with stave tones within
  10 %; brass F0 near (0.55, 0.40, 0.17) rather than (0.68, 0.39, 0.09); an albedo floor of ~0.025 so furrows are not
  painted black on top of the derived AO (bark); no painted-in shading.
- Judge's luminance table (linear): soot 0.02-0.04, dark wet stone 0.05-0.1, old brick 0.12-0.25, weathered wood
  0.15-0.3, limestone and plaster 0.4-0.65, whitewash <= 0.8. Dark materials with no gloss (basalt setts at 0.09,
  soil at 0.04, hedge at 0.07) vanish at night: give them specular breakup from the roughness map.
- **Metals**: tarnish must change `metal` and `rough`, not only albedo (brass at metal 0.995 everywhere was flat);
  patina as near-dielectric films tied to creases and seams with rubbed metal only on proud edges; verdigris as one
  continuous field (brown oxide -> dark tarnish -> grey-green -> raised crust -> pale cores), not evenly spaced
  ribbons (the evenly spaced runs a judge had prescribed scored lower than the round before).
- Tintable materials are authored in a mid-light neutral: the engine recolours by `albedo * tint / mean_albedo`.

## Tiling

- Gradients cannot tile (heart-side darkening is done as blotches); arcs tile only with a cusp at the edge; one
  pith per tile is a bullseye grid, so model the end of a timber away from the heart.
- Any fixed feature (six nails, three veins, one through-cracked flag) repeats every tile: keep the highest-contrast
  feature plentiful or faint. In `far.png` the eye must not be able to count repeats.
- Check edge rows and columns statistically for seams: a non-periodic `sin(ph * 0.5)` left a 2 px ruled line that
  cost a regression from 73 to 71. Pick an integer number of pixels per yarn for weaves (4, not 4.92) to avoid
  moire; ring pitches under ~2.5 px alias; height spikes give a "sandpaper" normal - build grit from `t.speckle`
  and high-octave `t.fbm`, never white noise.
- The judge's numbers are a starting point, not a spec: prescribed 0.5-1.5 mm plaster lips read as wires (0.35-1.0
  over 18 mm worked), 1-2.5 mm flag steps did not read (1.8-3.2 did), 5-10 % bare copper read as stamped orange
  blobs (3.2 % did). Verify in `far.png` after every change and re-read all three images; several regressions came
  from skipping the final look.

## Engine conventions the texture must respect

- U is "along": grain and rope along U; courses horizontal with the eaves at V = 0 (image row 0 is the top).
- The engine layers grime, wear and a material-specific "special" from per-vertex masks and adds world-space macro
  variation: author the material in decent, moderately aged condition with fine-scale ageing, not caked in
  large-scale dirt.
- Masks only darken in the asset pipeline (see `authoring-lessons.md`), so a material whose identity needs colour
  change (moss, rust, verdigris) must carry that colour itself or be a separate material.

## Workflow

- `tools/tsq mat <name> --res 512` to iterate (seconds), `tools/tsq mat <name>` at full resolution to finish
  (<= 60 s for a 2048 bake); read `preview.png` (identity, relief, seams), `far.png` (how the scene sees it:
  repetition, gloss under one lantern), `sheet.png` (albedo | normal | roughness | height | AO). `tools/tsq
  render-mat <name>` gives the in-engine swatch the judge scores.
- Helpers copied into every ground material (`_courses`, `_dots`, `_drop_stones`, `_drop_chips`) belong in the
  toolkit; so do a riven-step generator and a per-unit layout with closers.
