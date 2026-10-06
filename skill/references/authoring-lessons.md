# Authoring parametric assets: what passed, what failed, and why

> Read the lantern, cobble and moon specifics below as the reference run's instances of general rules; each rule is
> stated beside its example and holds for any emitter near a surface, any ground, any key light.

`docs/AUTHORING.md` (in `references/docs/`) is the contract authors read. This file is what 136 authors and ~400
judge rounds learnt on top of it. Give the **pre-flight list** at the end to every author; the rest is for the
lead when writing briefs and fixing the toolkit.

## 1. The shape of a passing asset

- **Every unit is its own slightly out-of-true object.** Stones (over a recessed mortar core 2-4 mm back, joints
  7-15 mm, blocks proud 3-8 mm ashlar / 15-30 mm rubble, per-block tone jitter from two merged tints, one or two
  replaced or spalled blocks per seed), staves, slats, boards with gaps, individually laid 0.26 m slates with
  slipped ones lying on the course below, bricks UV-mapped inside one texture brick with a header course. Seen in
  30+ passing assets; "flat, clean, plasticky" trim and "black slit joints" were marked down until mortar and units
  became geometry.
- **Three scales**: silhouette; mouldings, panels, brackets, bands, caps and bases (a bare primitive standing in for
  a moulded part is noticed - a plain inverted pyramid under a bay window read as "weird shapes" and needed a
  moulded corbel profile, then a *stronger* one because the first was too subtle from the street); bevels, fixings
  and seeded irregularity (lean, chips, sag, a missing piece).
- **Real dimensions that worked as anchors**: step rise 0.165-0.17 m, slate gauge 0.26 m, leaded quarries 12 cm,
  dormers ~1.2 m on bay centres, eaves overhang 0.15-0.25 m, cornice 0.36 m, run-off streaks 3-8 cm wide, rising
  damp 0.4-0.9 m, corbels bedded 0.08 m into the wall, brackets >= 0.25 m clear of lintel ends, plaster undulation
  <= 5 mm, strap legs ~60 mm ending on the frame rail, bracket arms >= 4 cm section (hair-thin arms read as
  floating lanterns).
- **Projection and depth**: a window flat on the wall was marked down; 0.11-0.17 m off the wall with the glass near
  the wall plane and the architrave dying back passes. A shopfront needs ~0.3 m of depth with panelled returns.
- **Lit rooms are recessed boxes** (wallpaper, picture rail, pendant, furniture or goods silhouettes, a pleated
  curtain) with the light **inside the room above the window head**, not emissive cards with a bulb glare on the
  pane. Night read of a building is decided by the share of warm pixels: a lit clock on a navy mass scored 73;
  glowing window bands, lit dormers and a portal light passed.
- **Cast iron reads when it has its own mouldings**: stepped pedestal, torus-and-scotia base, fluted or cannon
  shaft, belt, moulded neck and cap, raised lettering and a maker's mark on every casting.
- **Mechanisms as real linkages** (parasol ribs and runner so `open` 0..1 folds to a furled state; a ladder's lean
  solved against the wall; a bicycle stand whose foot lands on z = 0).
- **Variants must differ in shape, size or colour, not tint only** - foliage tints look alike under amber light; a
  rusty variant that stays the default's red is "all variants look the same" (cap <= 70).
- **Camera-aware**: the review cameras see the -Y side; put the bung, tap, pipe, wound or sign there. The daylight
  sheet shows only the default variant large - temporarily swap another variant into `default` to inspect it, and
  put it back.

## 2. Lighting values that actually read (the guide's candela figures are 5-20x too hot)

The authoring guide's candela table (wall lantern 25-45 cd, street lantern 50-80) was written for a far-reaching
inverse-square light and clipped 28 % of the close-up pixels on the first wall lantern. Point lights in the review
rig and the dynamic engine have no soft near field, so anything pale within 0.3-0.5 m burns to white even at
black albedo. Values that produced rich amber instead of white, measured across ~40 lit assets:

| fitting | cd that passed | notes |
|---|---|---|
| table candle, votive | 0.13-0.5 | one light per fitting, not per flame |
| candle sconce 12 cm from a wall | 0.24 per candle | |
| table oil lamp, step lantern | 0.7-0.9 | |
| wall lantern beside stone | 1.5-4.5 (rule of thumb ~8 x reach²) | 8 cd already burns a pale wall |
| sign lamps, plaque gooseneck | 2.2-2.6, range ~3.2 m | |
| neon | 2.6 per metre per face | guide said 6-15 |
| festoon / fairy bulbs | 1.3 per socket (12-16 total on a string) | |
| door or porch lantern 0.6 m off a pale wall | 4 | down from 30 |
| shop-window spill | 3.8-4 at 0.85 m out, 0.45 m up | so it pools on the step |
| lit room spill | 6-14 placed inside / above the window head | |
| free-standing lantern (well, cart, bicycle) | 5-9 | |
| pier / bridge lantern | 8-22 | |
| gas lamp post head | 42-60 (3.6-4.4 m up) | the exemplar |
| bandstand pendant | 65 | high and central |

Rules that stopped blow-outs: the light sits **at the flame or mantle** (the engine draws a halo sprite at every
socket, so a free-floating helper light reads as "a glow dot in mid-air"); `shadow_near` just past the fitting
(a cage shadow otherwise throws a giant stripe); glass glow low (pane 0.06-0.15, mantle 7-12, `glass_frosted` is the
reliable lit glass; `glass_lantern` / `glass_clear` with a point light 0.1 m behind blow to a white blob); dark
near-black metal or fuel lumps on lamp-facing faces; lamp-lit pale caps get a soot ring. Hidden fill lights break
the "visible source" rule but were the only way for globe posts and parasols - allow them, never give them a glow.
Hero and close-up differ by ~13x in exposure: judge both.

## 3. Materials: substitutions that read at night

| wanted | what was tried | what reads |
|---|---|---|
| gilding, brass letters | `brass`, `bronze_aged` | near-black at night and in the shaded 3/4 view; use `metal_plain` gold at rough 0.4-0.6 with arris wear, or dielectric gold paint `iron_painted "#d9a93c"` on shadow sides, or `brass` with `rough=` 1.25-2.3 |
| wrought iron | `iron_wrought` | black silhouette; grey-brown `iron_painted` (and lighter paint defaults everywhere: `#628c79`, `#40594b`, `#467d62`) |
| zinc, galvanised sheet | `lead` | `lead` is lilac at night and not tintable; light-grey `iron_painted` or `lead` at scale 0.25 |
| moss, lichen, verdigris, rust | `paint(special=...)` | the masks only **darken** (brown-black cushions, "tar" rust); model moss as instanced cushion geometry in joints and on the damp side, rust as flake meshes with a pale rim, verdigris as a tinted second material |
| isotropic stone (lathed urns, big flags) | `stone_smooth` | its 25-30 degree tooling reads as wood grain; tinted `mortar`, or rotate UVs 25-30 degrees per stone |
| white, pink, cream petals | `leaf` | tints a green albedo; use `wax`, `canvas`, `paper`, `plaster_rough` |
| lit window | `window_lit` untinted | renders dark; needs an explicit tint (`#ffc37a`), glow ~7 and per-pane 0..1 UVs |
| water in a trough or basin | `water` | opaque black from the asset side; only the engine's probes / SSR give reflections, so keep the surface and let the engine light it |
| dark paints at night | `#1b2622`-style defaults | lighten the default; `iron_painted` white base albedo is only ~0.55 |

Lantern glass recipe to share, not reinvent per building: `mat("glass_lantern", "#ffa94e", glow=1.4, opacity=0.32)`.
A built-in lantern in a townhouse with `glow=3.6, opacity=0.6` was the "white paned lantern" the scene judge flagged.

## 4. Weathering: by cause, at vertex resolution

- **Paint by cause, not by noise**: run-off under every drip, sill end, anchor foot and strap; soot above lanterns
  and arch crowns; splash band in the bottom 0.3 m; algae tide band with a wavy top; damp foot; rubbed polish at
  hand and tread height; rust tongues under hinges and bolts. Broad airbrushed halos and mid-strength grime read
  as "cloudy blotches", "camouflage", "charred".
- **Masks are thresholded**: invisible below ~0.3-0.4, blotchy to 0.6, black above ~0.65-0.8; `special` keys
  almost binary at 0.5. Keep the art-directed part narrow and strong, or use geometry.
- **Masks live on vertices.** Thin bars need extra rings for rust; a 2-triangle wall is one flat tone; a long box
  smears. Re-mesh walls on a regular 0.3-0.4 m grid **before** jitter and paint (otherwise staining fans out as
  sliver wedges from every window corner); constrained-Delaunay panels with guard vertices around openings;
  vertex rings just inside strand ends.
- **The export AO/cavity bake darkens any vertex buried in or covered by another solid** and smears it across the
  face: never overlap solids; stop plaster 2 mm short under beams; mortar skirts per stone; keep `densify`
  vertices out of junctions; glass counts as an occluder (a candle inside a chimney bakes dark). `META["auto_grime"]`
  / `auto_wear` (0..1) tame the automatic ground splash ("leopard blotches" on a bridge at 1.0; foliage wants 0).
- Keep geometry at **z >= 0**: anything below floats above its contact shadow, and below-ground parts bake black
  (`allow_below_ground` only for wells, canal walls, foundations).

## 5. Composition and budgets

- `use()` of catalogue components costs 7-37 k triangles and 3-11 materials each (wall lantern 9-18 k / 3-6
  materials, shop window 21-37 k, door 11 k, balcony 15 k). A composed building hit 218-303 k triangles and 56-65
  material variants. Building authors ended up modelling most parts inline or remapping every component's slots by
  library name onto a shared set of ~10. Tell authors the costs up front; the <= 10 material cap counts composed
  parts, and there is no per-vertex tint in the asset pipeline.
- Order waves so components exist before their composers run; a brief that names a component that does not exist
  yet gets it modelled inline, differently, in every building.
- **Budget governors**: estimate the triangle count from the params and step detail down in a fixed order
  (segments, optional dressing, tile courses) so every in-range combination builds under budget; stress-build the
  parameter extremes before finishing (`tools/tsq build <id> --params '{...}' --name tmp`, then delete the
  `tmp.*`/`custom.*` files). Judges check the contract range, not only the shipped variants, and silent culling
  of a briefed feature to fit the budget was marked down.
- Triangle costs that surprised authors: `bolt()` 80-164, `text()` 150-500 per glyph (and `size` is the em size,
  cap height is ~0.55 x), `chain()` 160-192 per link, `stone_block` ~300-640, `sweep(smooth_path=True)` resamples a
  two-point path to ~33 rings. Use 6-7 segment lathe domes for rivets, `smooth_path=False` for rods, local torus
  links, single-stroke lettering for signs.
- `roof_gable` pitfalls hit by seven authors: `ridge="y"` rotates every parentless object already in the scene;
  depth caps at 10 m; `gable="parapet"` adds its own gable wall; it lays a ladder and scatters slipped slates with no
  param. Build roofs first or with `ridge="x"` and rotate yourself.
- Determinism: `sphere()` face order varies between runs (use a lathed ball where identity matters); pin
  `PYTHONHASHSEED=0` (the CLI does).

## 6. Vegetation

Foliage must be mass, not cut-outs: twig-cluster cards of 3-6 leaves (2-4 triangles per leaf, 0.12-0.16 m) on
visible twiglets, three layers (dark inner shell or displaced `hedge` shell, mid plumes, paler new shoots, plus
brown die-back), normals bent to the canopy hull, per-card depth darkening, albedo roughly halved and desaturated.
Map each blade once to the whole leaf tile (`uv_keep`, `instances_to_mesh(..., uv_jitter=False)`) or the crown
shows zebra stripes. Leaves are back-face culled: duplicate reversed faces with their own offset vertices. Taper
twigs (13 mm untapered twigs were a judged flaw); root flare and pruning stubs on trunks. Cap leaf count to the
budget and enlarge blades when capped. Set `wear=0` on leaf materials or edge wear paints the cards cream.

## 7. The bar and the self-estimate

Authors said "about 80" for assets that scored 64-74 on nearly every first round (30+ reports): read a
self-estimate as about seven points high. The pass is 75, not 80; a brief that demands 80 sends authors after
unverifiable items (the back 3/4 panel, undersides). What closes the gap is a checklist, not a number.

## Pre-flight list (paste into the author brief, or run as the last iteration)

1. `hero.png` and the night close-up: the lit ground is rich amber, not white; no pale wall or cap within 0.4 m of
   a light is clipping; no glare blob on glass; every light sits at a visible emissive mantle, bulb or flame.
2. Every variant looked at large (swap it into `default`, render, swap back); the plainest and the most ornate
   build under budget; the parameter extremes built.
3. The contract: bounding box exactly `length` x `width`, origin and sockets where the brief says, param names as
   named; nothing below z = 0 (unless allowed), nothing behind y = 0 for wall mounts.
4. Masonry has mortar and joints as geometry; no bare primitive stands in for a moulded part; nothing paper-thin;
   brackets reach the wall; feet touch the ground.
5. Weathering is placed by cause and visible: streaks, soot, splash, polish, rust under fixings - and nothing
   relies on `special` for colour.
6. Metals and dark paints read in the night close-up (substitutions table above).
7. Normals: no inside-out lathe or lofted shell (check standalone inner shells; `recalc_normals` can invert a loft);
   no black faces in the clay panel.
8. Variants clearly differ in shape or size, not tint only; text is a parameter; moving parts have a pivot and axis.
9. No WARN left: materials from the library, within budget, <= 10 materials, <= 6 light sockets.
10. Scratch deleted; no `tmp.*`, `custom.*`, `_chk*` files left in `out/glb/<id>/`.
