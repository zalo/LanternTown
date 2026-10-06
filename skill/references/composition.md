# Composing the scene: a rule-and-seed layout that looks intentional

> The rules are stated for the reference run's town square; read "plaza" as the main open space, "lamps" as the
> environment's light fittings, "fountain" as the focal feature, "backdrop rows" as whatever closes the horizon.

The reference generator is `project/town/layout.py` (3,200 lines, stdlib only); `project/town2/layout.py` is a second
assembly built from one photograph in five hours and is the pattern for a new scene. `tools/town.py` runs it:
`gen` (layout.json) -> `build` (every distinct asset/variant/params, with LODs, `--procs` Blenders from `release/`)
-> `terrain` (ground, skirts, water) -> `resolve` (anchors, stacking, sockets, colliders, lights -> `town.json`)
-> `render` (the judged viewpoints through the walk page). The scene judge rubric is `docs/TOWN_QUALITY.md`.

## Shape of the generator

- `PARAMS` are the zone dimensions (plaza width and depth, axis x, terrace height and width, forecourt, row depths,
  quay and canal, street, fountain diameter, `density`, `storeys_bias`), overridable with `--set k=v`.
- Asset metadata is read from the generators and sidecars (class, mount, bounds, sockets), **release copy first**;
  built GLBs are sliced into footprints (`geo`) so placement validates against real colliders, cached in
  `out/town/geom.json`.
- A `Town` class holds the machinery: `route` (declared walking routes as keep-outs), `stair` (every flight plus a
  0.6 m margin is a keep-out), `cover` / `sight` (review-camera sight cones with a BULK table of plan radius,
  height and canopy lift per bulky prop), `settle` (nudge to the nearest clear spot), `place` (sample candidates in
  zone rectangles, reject occupied / route / stair / over-sight, cheapest wins, re-check built bounds), `bays`
  (edge furnishing that steps off its wall and slides along the bay when something stands in it), socket-driven
  `sockets / doors / fit / front`, `put` (refuses a hung or wall piece with no support within 0.3 m), `floor_gaps`,
  `afloat`, `off_stairs`, `buried`, and `audit` (asserts at generate time). `check_resolved()` repeats the audit in
  `resolve` against the real colliders and exits 1 on a placement failure.
- `_generate` is the plan in commented sections: frame -> walkable regions -> ground -> level changes (stairs,
  retaining walls, balustrades) -> buildings -> walking routes -> canal -> plaza centre -> fixed furniture (lamps
  first) -> one-off pieces -> dressing by zone -> what buildings offer on their sockets (signs, awnings, banners,
  window boxes) -> gate and street -> trees -> review cameras. `generate` iterates up to five passes until the
  cameras stop moving; `town.py all` loops gen + build up to three times because socket rules need built records.
- Deferred placement resolved after the build from real bounds: `anchor "miny"|"maxy"` (a local front or back
  face on `pos`), `on <item>` (stand on its top), `attach {to, role}` (sit on a socket), `nav "stairs"` (the footprint
  becomes a ramp region).
- Output: `items` (asset, variant, params, pos, rot, scale, collide, tag), `regions`, `terrain`, `shots`, `spawn`,
  `missing` (assets that do not exist yet).

## "Intentional, not cluttered" (the owner's complaint after walking round 3)

Fixed without adding or removing a single prop (545 items before and after):

- Derive everything from the plaza frame, the fountain centre and the door and stair sockets: two flagged walks
  crossing in the fountain (gate -> hall door, arcade -> terrace stair); a rectangle of like lamps that defines the
  quarters; one anchored feature per quarter (bandstand, market bay, kiosk + well, cafe); a fountain ring of four
  twin lamps on the diagonals, eight identical benches set exactly radial between them, four identical round beds,
  the same basket and banner on every ring lamp; subtle paving bands along the axes.
- **One lamp family per zone** and a palette by zone (plaza = gas twins; globe lamps only at cafe, forecourt,
  terrace; banners crimson on the ring and hall, green on edge lamps and cafe; one parasol design per zone).
- Feature pieces end an axis, balance another feature, or come as mirrored pairs or even rows: "no lone singles";
  flower beds in sets of 2 or 4, never 3. A `match` helper places a symmetric set all-or-none: a ring with one
  member dropped by a clearance rule reads worse than no ring.
- **Pin the variants that matter to the plan** instead of `"*"`: a random pick had put a 7 m landing stair across
  the lamp line and a `neglected` fountain at the focus.
- Loose dressing aligned to a wall or stall with yaw jitter <= 8 degrees; ladders, cats, bins, churns and the cart
  heap off the open paving; pigeons kept to the kiosk front.
- Symmetry has a cost the next judge sees: mirrored stalls with the same awning, crates and sacks were flagged as
  clones. Mirror the placement, not the seed: a matched pair must still differ in variant and seed. More generally,
  an accent prop kind fronts at most one shot, and a prototype appears once per 30 m sight cone with its variant and
  seed varied.

## Lamps and light budget

- Lights are omnidirectional lantern points; a render with lamps off is a correct moonlit blue, so floods come from
  lamp count and intensity, not ambient. Walls above lanterns get as much light as below unless the engine caps the
  lantern (`capFactor`: 25 % upward). Scale lamp intensity with span (street 970 -> 537 cd total, monument court
  375 -> 221, terrace 941 -> 474); a candle lantern on every cafe table at 1.6 cd; a light budget (448 here, 412 used).
- Reject any lamp position within ~3 m of a shot camera's line to its subject; a lamp post cutting the fountain in
  half and two foreground posts were the second scene's first problems.
- Glow sprites come from the fitting's declared position (one per real burner, sized from sqrt(candela)); helper
  lights (pool, spill, row stand-ins) never get one.

## Cameras are part of the layout

The judged shots are generated with the layout (14 here: approach, gate view, plaza from terrace, cafe, fountain
close, stairs and shops, terrace shops, monument, hall, arcade to canal, canal, bridge back, market, overview),
eye height 1.65 m, FOV 62, 16:9. Bulky props are tested against the sight cones (a general-rule prop may hide at
most 14 % of a frame, a one-off 5 %); nothing loose within 3 m in front of a camera; a standing piece inside 6 m is
wholly in or wholly out of frame; what stands below eye height keeps its foot 7 % of the frame height above the
bottom edge. Re-set a camera when the plan changes (one moved onto the cross axis; one 3 m to clear the bandstand;
one off the bay axis so the clock is not on its subject line). Items nudged by camera frames are counted in the
build report. A bird's-eye shot uses a shift lens (`shift: true`), which needs an off-axis fix in the clustered
lights' compute.

## What raised the scene score, round by round

- 66 -> 68: wet paving with SSR (the judge's top request and the biggest realism gain); the `sourceRadius` soft
  near-field falloff in the clustered lights (walls beside lamps burning to yellow: clipped pixels 4.6 % -> 0.04 %);
  the monument recoloured to the reference's amber star; cameras pulled out of kiosks and from behind posts.
- 68 -> 74 (cap lifted): resolve-time overlap audits (tree guards inside trunks, a bench in a stoop, backdrop rows
  interpenetrating); 3-4x more, smaller leaf cards on real twigs; falling water as narrow overlapping strands with
  irregular lower edges, splash rings and foam; incised gilt inscription on a jointed tablet; bracket arms >= 4 cm;
  feathered puddles with a damp halo and a ripple micro-normal; a highlight knee; a cloud deck with horizon glow.
- 74 -> 68 (capped) -> 81: one floating 0.3 m weathervane (no rod) capped an uncapped 75, hence the automated
  "attached to something" audit; then ambient 1.3 -> 0.8, moon 2.0 -> 1.3, lamp reach falloff `1/(1 + d²/4.5²)`, an
  emissive gradient on lamp glass, SSR of lamps and lit windows on puddles, re-baked stone with a pitted limestone
  micro-normal, inner backdrop rings drawing near variants only (the hazy far variant read as "a windowless navy
  slab" = blockout), a second lower ring so belfries show, a far castle placed where a walk shot sees it over low
  roofs and sunk into the haze (`floodlit 0.3, haze 0.85`; 0.7 / 0.4 was "cartoon-bright").
- Side effect to expect: dropping ambient turned trees into black cut-outs in both scenes; fix with a hemisphere
  floor for foliage (cool sky on tops, warm bounce from lit paving underneath) applied after AO, not with
  self-luminous greens.

## Backdrop and skyline

Backdrop rows stand 40-150 m off, so detail has to be put into the **distance tier** with pins that survive
reduction: belfry stages, string courses, end-gable windows, chimney breasts; "no stage taller than ~3.8 m, no
shaft featureless for more than ~4 m"; window bands as 24 / 34 / 42 % bright / dim / amber with pinned mullions.
Baked glow cards on far rows read as "stickers" and were replaced by real lights capped per row. Give the author
2x upscaled crops of the asset **as seen in the scene**, not its own sheet.

## A second scene from one picture

310 items from 66 assets in ~5 hours, judged 73, 73, 73, then 68 capped and stopped at the owner's request as a
demo. It never passed because the library had no asset for its two landmarks; the keep was a gate scaled 3x, and
the judge measured 1.5-2 m stones and 2.5 m lanterns at once. Do not start a second assembly until the library has
its landmarks at true size; carry the intentional-plaza and lamp-view-line rules from day one; make the first shot
reproduce the reference picture's camera for a side-by-side.

## Rule fixes, not instance fixes

When the owner reports a floating lantern, a bridge off its quay, a stair through a fence: fix the rule that
produced it and add an audit (`put` support check, `floor_gaps`, `afloat`, `off_stairs`, `buried`). In the reference
run the rule fixes caught four more defects nobody had reported (two gate lanterns hanging from nothing, a tunnel
mouth floating, a balustrade 2.8 m through a basin, a shop light off its window). Test three rule values from a
script that rewrites the constant, re-runs `gen` and restores the file; apply the one that passes at rule level.

## Composition traps

- A variant draw seeded by an item's rounded position reshuffles when the item moves even a few centimetres:
  check which variants a layout change needs built.
- "Unchanged" must be checked on the file a step writes: layout generation wrote one file and the comparison read
  another, so a rebuilt asset's larger footprint silently dropped two statues.
- A new asset judged on its own sheet can still fail inside the scene (sockets, footprint): test the swap before
  promising it. A building that lacks a socket the plan relies on (the hotel's upper door) makes a stair vanish.
- Density added to raise a score reads as clutter; empty foreground paving is a real deduction, but the answer is
  organised dressing (a planter pair, a bollard line, paving bands), not scatter.
