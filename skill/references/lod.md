# Levels of detail: cheap models that keep their identity

`docs/LOD.md` (in `references/docs/`) is the author guide: the `lod()` tools, rules a-g, the checklist per asset
class and thirteen common mistakes. This is the strategy and the numbers around it.

## Goal and route

The owner's words: "generate them with lower levels of detail as opposed to clobbering them after the fact ...
just looking for more beautiful cheap models". Low levels need no judge gate. The shipped route is **reducer
first, authored where it fails**:

1. The toolkit scales itself at levels 1 and 2 (round sections to a 4 / 18 mm facet tolerance, small bevels
   vanish, profiles simplified, `chain` -> strip, `bolt` -> nothing, thin `text` -> flat glyphs). Alone this passed
   1 of 127 assets (22 at the theoretical best) and changed an awning from 12 stripes to 4: never ship it alone.
2. **The reducer** (`tk/reduce.py`, applied *before* parts are merged): remove faces nobody can see, flatten flat
   areas, drop tiny loose pieces, then collapse. Passed both levels on 86 of 127 and one level on 34 more.
   **Pick-best** keeps whichever of authored / reduced / decimated is closest to the full model by an image metric
   (`tk/lodimage.py`) so nothing ships worse than before.
3. **Authored levels** (`META["lod"] = True` or `"far"`) for the ~30 that still fail: foliage, striped cloth,
   posters, lamps, glazed windows, wire furniture, and the far levels of landmarks, backdrop rows and clocks.
   `lod_standin` / `lod_keep` protect identity parts in assets that are not level-aware (clock hands and
   numerals, a globe lamp's shells as one sphere).

Budgets per variant: LOD1 <= 40 %, LOD2 <= 10 % of its own LOD0 (floors 400 / 150). Pilot of 8 assets: LOD1 16-39 %,
LOD2 2.5-8 %, always at or below the decimated count; wins were largest where decimation tore (a casement window
9,966 -> 1,574 / 254 against decimated 2,719 / 494; a barrel's far level 451 vs 1,222 and whole; a fountain 26 % /
6 %). The gain is identity at the same cost, not fewer triangles.

## Rules that reject a level (the build checks them)

LOD0 frozen to the vertex; the random stream must not shift (draw everything, emit less: `lod_upto`, `lod_thin`,
forks); sockets, lights, parts, footprint and origin identical (measure on LOD0 with `lod_memo`); everything that
glows survives (same material at 1, same kind at 2, >= 30 % of its area); identity survives LOD1 (numerals, hands,
lettering, glazing bars over lit windows, pattern colour changes); same materials (masks transfer by name); **no
holes** (rule g, measured by `tk/holes.py` from street-level viewers 3-15 m away, back faces culled except glass,
plants and cards; limit 0.05 % of samples and 0.2 % of the worst view).

## What a cheap far level is made of (reference authors, 45-55 min each)

- Hanging basket: a 4-sided turned hook, 3-sided chain rods, foliage as a closed 8-sector shell through the leaf
  tips, one diamond card per flower clump with the colour lifted off the mound, ivy as closed 3-sided streamers plus
  outline kites holding the bounds (22,325 -> 8,212).
- Balustrade: coping and plinth prisms per stone with the same stone picks, 4-sided balusters of 16 triangles,
  6-sided finials, moss as 3-sided tents (38,223 -> 6,479).
- Canal wall: one quad per block above the tide line, one per course run below (12,890 -> 4,066). Bench: slats
  merged into planks per finish (28,310 -> 8,784). Gas lamp post (the exemplar): outline, cage, roof and glowing
  mantle; relief, fixings and scrollwork left out; a 6-sided mantle drawn round the full one so it shows as much
  light; the vent, cap and finial as one 5-sided turning.
- Ironwork keeps **every** bar (thinned runs read sparse and strobe) but square 4-sided at 1 and fronts-only
  3-sided at 2. Roofs: tile courses as stepped strips at 1, flat planes with eaves thickness at 2, **the deck under
  dropped tiles carries the tile material** (a dark oak deck under tiles was "the largest hole in the whole scene":
  a black slab from across the square on Phone). Glazing: the lit pane survives; bars fronts-only, then one card.
  Foliage: keep every draw, mesh less; enlarge the kept cards so leaf area is constant and pull them inward.

## The trimming pass nobody planned

Modelled far levels made the scene's LOD2 total **33 % heavier** (890,775 -> 1,186,808 triangles; one pavement-weeds
build 38 -> 1,788), costing Low and Phone 12-16 % more triangles and pushing near cells down a level on Phone's
memory ceiling. A lead plus forks worked down a sorted growth list (`asset_growth.json`: top growers with deltas)
to a numeric target (scene LOD2 <= 950 k) and reached 951,474 (-19.8 %), fixing three named defects on the way. Set
the far budget **before** the wave or identity parts eat the saving; most authored far levels land at 30-45 % of
before when 10-15 % was asked.

## Gates: average metrics pass torn parts

- The image gate passed torn clock dials, a pale shard and dark gaps because it averaged the difference; the lead's
  own look at six of ten sheets caught them. Demand a **worst-region** check as well as the mean, and look at sheets
  before accepting a round.
- The gates are calibrated for mid-size objects: on a basket 16-25 px wide a one-pixel rim error exceeds the 0.08
  silhouette limit, so warnings on objects under ~30 px are advisory; judge in three.js.
- The gate can pass what looks wrong: cards 40-50 % larger than the blades they replace read pale and passed
  brightness. Size each card so its visible front area matches what it replaces, class by class, within 5 %.
- Lit facades: dropping 16 mm glazing bars over lit cards raised lit pixels 2.4x (brightness 1.41). Fewer, thicker
  bars or a darker card - bar area is part of the window's brightness.
- Stand-ins to the millimetre: a far globe at 0.97 R left 0.0021 m² of glow within 4 mm of ironwork; at 0.96 R, none.
- Reproducibility is imperfect: the same generator built twice gave 2,366 then 2,367 triangles; compare LOD0 by
  byte or an order-free geometry hash and expect reduced levels to wobble.

## Holes and slivers

- **Two layers cause them**: the asset reducer deleting faces judged buried and then simplifying away what covered
  them (fixed by rule g's check and a repair that repeats until it passes: 64 of 654 levels over the limit from
  the build's viewers -> 0; 273 -> 220 from stricter independent viewers), and the bake's packer moving a window
  rim or eave by a centimetre at the far level (`tools/pvs.py --holes` draws each level in ID colours beside the top
  level's depth from the walker's eye points and lists where a reduced level lets a ray through; the packer
  (`pack.holes=1`) locks the source triangles - whole loose parts - before simplifying).
- Measured: at the far level 86-88 % fewer see-through pixels for 30-32 % more triangles (base file 7 -> 10 MB);
  at mid 6-7 % fewer for 6-8 %; at near 17 % for 9 %. So mend the far level only (`pack.holes_levels`, default
  `lo`). The census must come from a pack **without** mending, and a 40-viewpoint trial under-reported the cost
  (+11 %) against the 600-point census that ships (+30 %): cost census-driven changes with the census that ships.
- A hole measure must be single-sided except for materials two-sided on purpose, from street-level viewers 3-15 m
  away, against LOD0: the first gates drew everything two-sided and a hole showed the inside of another wall in a
  similar colour. The first sliver census counted removed window frames and balusters as holes: a hole is only where
  the sky or a cell well behind shows through where the top level stops the ray.
- Catalogue of causes in authored far levels: staves each at their own radius and head boards with gaps; stacked
  lathe pieces with different side counts and no caps; kerb blocks without end faces; bar feet missing under a
  bottom rail; a profile helper skipping hidden edges so a die became two loose strips and `recalc_normals` turned
  one inwards; a disc rim with half the segments whose chords lie inside the full rim; limb tubes inscribed in the
  round section; five loose door boards with 4 mm joints. Closing them cost +30 to +60 triangles per level.
  **Nothing at the far level may be a loose open-backed strip**: the bake's hidden-face cull and the 0.22 m
  simplification will prune it (a sky strip along the canal wall survived every asset check for that reason).
- A fixed hole can uncover the next gate (a closed door then reported a 2.3 % light leak): read the whole warning
  list after every fix. Rows with a worst view over 0.2 % under 4 samples pass; report, do not chase.

## Workflow

```
tools/tsq lod <id> [--variant v] [--top N]    # 3 levels + the old decimated ones, budget / OLD / NEW / checks,
                                              # "where the triangles are" per source object, the comparison sheet
tools/tsq lod <id> --no-render                # numbers only
tools/tsq build <id> --lods --out .scratch/<you>/<id>     # every variant as the scene builds them
python3 tools/checks/holes.py town --out FILE  # rule g on every variant the scene uses, worst first
```
Loop: cut the biggest row of the triangle table, rebuild, then **look at the sheet** (LOD0 | decimated | what ships,
with 1:1 crops at 16 m and 44 m). Done when checks are `ok` on every variant, LOD0 unchanged, NEW <= OLD, sheet
looked at. Levels add about one more `build()` to the build time. Freeze the toolkit for a wave: `tk/reduce.py`
changed mid-campaign and ten of eleven comparisons were against the wrong baseline. After any roof or LOD work,
look at every tiled roof from 40 m on the two lowest presets (`roofview`: elevated views, `quality=phone&pixels=2e6`).
