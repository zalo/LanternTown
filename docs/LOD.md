# Modelled levels of detail - author guide

The town shows every asset at three levels: LOD0 (full model, near), LOD1 (mid, and the NEAR level of the baked
"Low" tier, so it is seen from a few metres), LOD2 (far). An asset without `META["lod"]` gets LOD1 / LOD2 made
from LOD0 by the build: the REDUCED copy (`tk/reduce.py`, see "Reduced levels" below) or, where that fails, the
decimated one (mushy bars, holed roofs, lost contrast). A converted asset is MODELLED cheap instead.

## How the level reaches you
Put `"lod": True` in `META`. `build(p, rng)` then runs three times with the same params and seed; `lod()` returns
0, 1 or 2. Budgets, relative to this variant's own LOD0 triangles: **LOD1 <= 40 %, LOD2 <= 10 %** (floors 400 / 150
tris for tiny assets). Aim at or below the OLD decimated count printed next to yours; fewer is welcome.
The toolkit already scales itself at lod 1 / 2: round sections (cylinder, sphere, torus, lathe, sweep, rounded_box)
keep a 4 mm / 18 mm facet tolerance; bevels under 6 mm / 30 mm vanish, wider ones become one chamfer; lathe
profiles, sweep paths and extruded 2D outlines are simplified to the same tolerance; `chain` -> a strip; `bolt` ->
nothing; thin `text` -> flat glyph fronts; `subsurf`, `densify`, `surface`, `plank`, `stone_block` get coarser. NOT
scaled: `box(segments=)`, `grid`, counts of 8 or less, hand-built meshes, how many things you instance. That alone
rarely reaches the budget (typically 55-90 % / 40-80 %); the rest is yours. `use()` hands the level to composed
assets (a converted child is modelled at the level; an un-converted one is built in full and reduced inside you by
`tk.reduce.reduce_child` to 40 % / 10 % of its own triangles).

### Only the far level: `META["lod"] = "far"`
For an asset whose LOD1 is fine as it is (the decimated / reduced copy) and whose LOD2 is where the triangles go:
put `"lod": "far"` in `META`. LOD0 and LOD1 are then made exactly as without `META["lod"]` (same files, to the
byte where the build is reproducible), and `build()` runs once more at `lod() == 2` for the far level, which goes
through the same gates as any modelled level 2 (a rejected one falls back to the decimated copy, with a WARN).
`lod()` never returns 1 in such an asset. Composed into a parent: a converted parent gets the modelled level 2 and
the full child reduced at level 1; an un-converted parent builds the child in full, as before. Used by
lamp_post_gas, lamp_post_globe, wall_lantern, lantern_head, pavement_weeds, backdrop_row, bench_park, balustrade,
canal_wall, crate_stack, hanging_basket, banner_crown, window_box, flower_bed, hedge_run.

### A different share for one asset: `META["lod_share"] = {2: 0.7}`
The budget of a modelled level as a share of this asset's own LOD0 instead of 40 % / 10 %. Only for assets whose
LOD0 is itself a far model (backdrop_row: a row of houses that is never seen closer than 60 m, LOD0 5-7k
triangles; 10 % of that is a row of boxes). Every asset without the key has the standard budget.

## Tools (all are no-ops at lod 0; `tools/tsq api <name>` prints the docstring)
| tool | use | example |
|---|---|---|
| `lod()` | branch on the level | `if lod() < 2: knobs()` (only if the block draws no rng, makes no socket / part) |
| `lod_pick(full, mid, far=mid)` | a value per level, used as given | `segments=lod_pick(48, 16, 8)`, `style = lod_pick("carved", "plain")` |
| `lod_exact(v)` | a count / width the toolkit must not rescale | `cylinder("nut", r, h, verts=lod_exact(6))` |
| `lod_upto(n)` | block runs at every level (rng, sockets, parts equal), its meshes emptied above `n` | `with lod_upto(0): rivets(...)` |
| `lod_drop(objs, above=0)` | function form of the same | `lod_drop([hinges, nails])` |
| `lod_thin(seq, mid, far)` | evenly fewer of a repeated thing, ends stay | `for x in lod_thin(xs, 0.5, 0.25): bar(x, w * lod_pick(1, 1.5, 2))` |
| `lod_faces(objs, keep=, drop=, above=0)` | delete unseen faces | `lod_faces(bar, keep=(0, -1, 0))`, `lod_faces(plinth, drop=(0, 0, -1), above=1)` |
| `lod_as(level)` | model a block at another level | `with lod_as(0): numerals()`; `with lod_as(lod() + 1): back_wall()` |
| `lod_segs(n, r)` / `lod_path(pts, closed, k)` | toolkit scaling for rings / profiles you mesh yourself | `seg = lod_segs(32, r)`; `prof = lod_path(prof)` |
| `lod_memo(key, fn)` | a measurement taken on LOD0, replayed at the levels | `z = lod_memo(("roof_z", i), lambda: ray_z(roof, x, y))` |
| `lod_budget()` | triangles this level may have (None at lod 0 / inside `use()`) | `target = lod_budget() or TRI_TARGET` |

## Identity parts without converting: `lod_standin` / `lod_keep` (assets WITHOUT `META["lod"]`)
The reducer cannot know that two slivers of iron are the hands of a clock. A generator that is not level-aware can
say so, and say natively what the part looks like further away - `build()` still runs once, at lod 0:

```python
hands = [hour, minute]                       # finished lod-0 objects (meshes or instancers)
def hands_far():                             # called once, right there; returns finished mesh objects
    return [poly("hour_far", bold(hl, 0.085), dial_m_dark), poly("min_far", bold(ml, 0.06), dial_m_dark)]
lod_standin(hands, mid=hands_mid, far=hands_far)   # LOD1 gets hands_mid(), LOD2 hands_far() in their place
lod_standin(minute_ticks, mid=(), far=())          # () = dropped whole at that level
lod_standin(dial, far=dial_far)                    # mid omitted = the originals stay, protected
lod_keep(bell)                                     # = lod_standin(bell): kept as modelled at both levels
```
- LOD0 is not changed by a vertex and no random draw moves: a stand-in callable runs with the build counter set
  aside (names / texture offsets of everything after it are what they were) and must build from numbers only -
  no `rng`, no fork, no measurement of the scene - with the materials of the originals, in WORLD position. The
  objects it returns are taken out of the scene and wait in the snapshot (`~standin<group>.<level>.<k>`).
  Originals placed by `instances_to_mesh` need the stand-in instanced by the same matrices.
- At LOD1 / LOD2 the reducer removes the tagged originals, links the stand-in, and from then on does not touch it:
  no unseen-face removal, weld, prism, drop, patch or collapse (the same protection `text()` faces have). Kept
  originals are protected the same way. Their triangles count: the rest of the asset gets `budget - identity`.
  So a stand-in has to be cheap, and a kept original small - nothing is reduced for you there.
- The clearance rule holds for stand-ins as for anything else you model: no face parallel to a glowing face or a
  card within 4 mm of it (lay a hand 10 mm or more off the dial; the gates of "Reduced levels" measure it).
- Lettering inside a group that is swapped or dropped at a level is not held to the lettering gate at that level.
- The chooser: a reduced copy that carries declared identity parts is not handed to the decimated copy on image
  difference alone (a stand-in is meant to differ from LOD0, and the decimated copy collapses the part like
  everything else); it still loses to it when it fails a gate the decimated copy passes. Its worst-tile limit is
  1.5 x the usual one (`TILE_IDENT`), for the same reason.
- Inside a `use()` child no stand-in is made (the parent reduces the child as one mesh): the originals are kept.
  In a level-aware asset (`META["lod"]`) and above lod 0 both calls do nothing - model the level instead.
- The mark in the files: every GLB of the asset (lod 0, the reduced and the decimated copies) carries a second
  UV set, glTF `TEXCOORD_1`, whose x is 1.0 on identity vertices and 0.0 elsewhere (`tk.final.mark_keep`; only on
  meshes that have any): stand-ins, kept originals, and lettering - all of it at lod 0 and in the decimated
  copies, at a reduced level only groups 3 px or taller at the level's distance (the smaller ones are allowed to
  go). A later simplification (the web bake) locks those vertices: read accessor `TEXCOORD_1` column 0, or after
  Blender's importer the x of the second UV map. It is NOT a custom attribute `_TSQ_KEEP` next to `_MASK`:
  Blender 4.1's glTF importer keeps the custom attributes of a mesh in a set and their data in a list, and with
  two of different width the import raises ValueError whenever the set iterates the other way round - always
  under `PYTHONHASHSEED=0`, about every second process otherwise (measured) - which would stop
  `tools/checks/zscan.py` and every other tool that imports a built GLB. `TSQ_KEEP=attr` in the environment
  writes the float attribute `_TSQ_KEEP` instead, for a reader that takes accessors straight from the file;
  `TSQ_KEEP=0` writes no mark. The web bake (`tools/webbake_bl.py`, `load_glb`) reads both `TEXCOORD_1` column 0
  and `_TSQ_KEEP`; the default (uv) mark is the one in use (`tools/webbake_keep_test.py`).
- In use: the dials of `clock_tower_hall`, `street_clock`, `wall_clock` (LOD1: flat hands, 12 hour marks, the
  numerals, no minute ticks; LOD2: a 16-sided dial disc, 12 bold marks, bold hands of the same lengths and
  angles, numerals and rings gone). Hands, numerals, hour marks and painted rings of the three are one matt
  non-metal near-black at every level, `mat("wood_painted", "#0c0c0e", rough=1.8, wear=0.0, special=0.0)`
  (wrought iron is metal and took a highlight from a lamp beside the dial).
  `lamp_post_globe`: at LOD1 and LOD2 each globe is one glowing sphere of the opal body's radius (opal body,
  clear skin and smoky caps are three shells 1-7 mm apart; decimated separately they cut through each other:
  0.31 m2 of glowing faces under 3 mm at LOD2, none since); the plinth with its panels, hatch, hinges, bolts and
  lock is one octagonal shell (LOD1 with the eight panel frames, LOD2 ten profile points); and the beaded bands
  of the column are stand-ins as well, because a pool light sits on the axis inside each band: as identity parts
  they keep the shaft wall behind them, which the hidden-face step otherwise removes and the leak gate then
  reads as a 3 mm slit all round the light.

## Rules (the build checks them; a level that breaks one is REJECTED and the decimated copy ships, with a WARN)
a. **lod 0 is frozen.** Not one vertex of the full model may change: no new arithmetic on the lod-0 path
   (`x * lod_pick(1.0, 1.5)` is fine, re-ordering a sum is not), no reordered calls. Check the tri count is the same.
b. **The random stream must not shift.** `rng` must end in the same state at every level. Draw everything, then
   emit less: `lod_upto`, draw-then-`lod_thin`, or `rng.fork("name")` for anything a level skips. A plain
   `if lod() == 0:` around code that draws from `rng` is the classic "rng drift" rejection. Forks are NOT
   checked: a fork whose draws depend on the level's mesh moves things silently (mistake 2).
c. **Sockets, lights, parts (pivot, motion), footprint, origin: identical.** Create them at every level, from
   numbers, never from the level's geometry (ray casts, bounds, vertex positions -> `lod_memo`). Bounds may move
   by 1 % of the diagonal (2 cm at least; warned), 4 % (8 cm at least) rejects.
d. **Everything that glows survives** (`emissive`, `neon`, `window_lit`, `glass_stained`, `glass_frosted`,
   `glass_lantern`): same material at lod 1, same kind at lod 2, covering the same places (a lamp may become a
   card or a low prism, it may not vanish or shrink below 30 % of its area). `lod_upto` never empties glow.
e. **Identity survives LOD1.** Whatever carries the asset's identity or readable contrast - dial numerals and
   hands, lettering, glazing bars over lit windows, pattern colour changes (harlequin gable, striped awning,
   timber on plaster) - stays, as cheap flat geometry (`lod_faces(keep=front)`) or as is (`lod_as(0)`), even when
   tiny. Cut tessellation and hidden / back / inside parts first. At LOD2 keep silhouette, colour zones and glow.
f. **Same materials.** Vertex masks (AO, grime, wear, special) are not painted on a level: they are transferred
   from LOD0 per material NAME, so keep `mat(...)` calls identical; a stand-in takes the finish of what it
   replaces. Masks you paint at a level are overwritten. Leave enough vertices for weathering to live on (a
   2-triangle wall is one flat tone: one coarse `densify(o, lod_pick(0.25, 0.6, 1.5))` is enough).
g. **No holes.** Nobody on the street may look through a level where LOD0 is closed. Measured (`tk/holes.py`)
   as the town draws it - back faces culled, except the materials the engine draws two-sided (glass, plants,
   cards) - by viewers where people stand: eye height 1.6 m at 3, 8 and 15 m around the asset, plus a ring at 5 m
   height (wall-mounted assets: from in front), looking level and up. A sample is a hole when LOD0 shows a front
   face there and the level shows a back face or nothing, further inside the outline than the level's size limit.
   Limit: 0.05 % of the samples and 0.2 % of the worst single view (at least 4 samples; assets with fewer than
   400 samples of closed surface are not judged). In the sidecar: `stats.lods[].holes` (share), `holes_worst`,
   `holes_at` (material and position of the largest group), `closed` (triangles the build put back).
   Usual causes: a face deleted because "nothing sees it" that a later simplification uncovers (the wall behind
   a moulding, the reveal behind a frame, the panel of a door); a closed solid cut open with `lod_faces` whose
   remaining faces do not overlap; a lid, sill or coping dropped as small. For reduced and decimated levels the
   build closes them itself (below) and does not hold the `closed` triangles against the budget (up to a quarter
   of it). An AUTHORED
   level is only measured: the WARN `lod N: holes: ...` names the place - model the face back in.
UVs / tangents are regenerated by your own uv_* calls as at lod 0 - call them on the stand-ins too.

## Commands
    tools/tsq lod <id> [--variant v] [--top N]   # builds 3 levels + OLD decimated ones into out/lod/<id>/, prints
                                                 # budget / OLD / NEW / check / where the triangles are, renders the
                                                 # sheet out/renders/_lod/<id>.png (<id>.<variant>.png for others)
    tools/tsq lod <id> --no-render               # numbers only (fast loop);  --auto: toolkit scaling without META
    tools/tsq build <id> --lods --out .scratch/<you>/<id>     # ALL variants as the town builds them; every line
                                                 # must say "authored" for lod1 and lod2 (an un-converted
                                                 # asset says "reduced" or "decimated")
    python3 tools/checks/holes.py <id>/<variant> ...          # rule g on built files (--dir .scratch/<you>/<id>/..),
    python3 tools/checks/holes.py town --out FILE             # ... on every variant the town uses, worst first
                                                 # (--phase 0.12: other viewers than the build's own; pictures on
                                                 # magenta with back faces culled: tools/checks/holes_view.html)
Loop: read the "where the triangles are" table (per source object, lod 0 / 1 / 2), cut the biggest row, rebuild.
Then LOOK at the sheet (LOD0 | decimated | what ships, enlarged, and 1:1 crops at 16 m / 44 m; the right-hand
column is titled with the mode that ships - `authored`, `reduced`, `reduced (1.5 x budget)`, `decimated ships`):
it must read like LOD0 at the true-size crop and be visibly cleaner than the decimated copy enlarged. Done when: checks `ok` on every variant, lod 0 tris
unchanged, NEW <= OLD, sheet looked at. Levels 1 + 2 add about one more build() of the asset to the build time.

## Reduced levels (what an asset gets without `META["lod"]`, and what yours is measured against)
Per level the build makes up to three candidates and ships one; the sidecar says which in `stats.lods[].mode`
(`authored` | `reduced` | `decimated`; `rejected` only in `tools/tsq lod` comparison builds) and why in `lod_pick`.
- **decimated**: LOD0 merged and collapsed by one ratio (`tk.final._make_lod`) - the old behaviour, always there.
- **reduced** (`tk/reduce.py`, on the un-merged LOD0): unseen faces go first (buried, undersides, backs against the
  wall) - except the ones that cast the shadows of the asset's own lights (`_light_casters`: the engine draws the
  back faces of single-sided materials into a shadow map, so the unseen underside of a lamp's hood is what keeps
  its light off the wall behind; an unseen face stays when it stops a light socket's beam that would otherwise
  land on a face that stays, on the ground or on the mounting wall - tested on the ray to each triangle's centre, and for
  lights flagged `shadow` in their socket on four rays per triangle (its centre and a point towards each corner),
  because a face whose middle lies in the shadow of a part that stays still lets a thin beam out past the rim of
  that part: the lit outline on the paving under street_clock. The four-ray test for every light costs a big
  building thousands of unseen triangles (clock_tower_hall + 3343 at lod 1), so unflagged lights keep the centre
  ray and may still leak such a thin beam where the town gives them a shadow map; `lit_kept` in the reducer log) - coplanar
  faces merge, thin long pieces become 4- / 3-sided prisms, small loose pieces are dropped (never
  the outline, the last piece of a part, or anything near glow), plant cards are thinned and enlarged, fields of
  small pieces are welded, and only the rest is collapsed. A field that will not weld (roof tiles laid over each
  other) and that the collapse would tear into a lattice becomes flat patches instead: its pieces are grouped by
  plane (a cone falls into sectors) and every row of a group is one rectangle in the pieces' material, so
  openings and the outline of the field stay (a rectangle lies where its row lies, sloped with it). A weld or a
  plan that lays a field bare does not stand: points on the flat faces of a field's pieces are looked up again
  along the way the piece faced, and a point is *open* when the first face met there is neither a face of the
  field nor a face of the same material and grime no deeper than a pixel (a roof may lose its tiles to the deck
  under them; a wall must not lose its ashlar to the darker core 1 cm behind). A weld may open 3 % of the points
  (6 % of any one field), and the share a candidate ends with is its `open` (`lod_pick[].cand[].open` of the sidecar, gate below).
  Every step is undone where it moves the bounds, empties
  a part or damages the outline. The collapse works loose piece by loose piece (pieces brought to one size, the
  seams to what is not collapsed pinned) and a piece only takes a step that leaves it whole - the part may end
  over its ratio: what cannot be had without tearing is left. Glow is left as it is; only where glow is most of
  the model (a wall of lit windows, neon tubes, a string of bulbs) is it collapsed with the rest, and only when
  its lit picture holds; glowing panes tiled in one plane (stained-glass quarrels) become one pane per block of
  the level's size limit at lod 2, in the tints of their tiles.
  *Lettering* (`text()` tags its faces: face attribute `tsq_text`, never exported) is never welded, patched or
  collapsed: it loses its backs and the faces bedded on the board, at lod 2 its sides (fronts only), and its flat
  fronts are dissolved; a group under 3 px tall at the level's distance may become its flat front or go - whole,
  never in part. Letters built without `text()` (hand-made glyph meshes, strokes) are not protected: use `text()`.
  *Cards* - glowing faces, and the room card of a window lit or dark (`window_lit` at any glow, `window_dark`:
  the dark card of an unlit room fights the wall like the lit one) - are never welded or snapped onto another
  surface, dropped or collapsed: no step may bring a card nearer than 4 mm (read as 3.75 mm, for rounding) to a
  parallel face of another material than lod 0 has it. What a card hides goes first, a piece takes no collapse
  step that lays more of it onto a card or across one (measured, not sampled: `tk.lodimage.z_pairs` clips every
  near-parallel face to the card, so the thin strip of a bar that comes to cross a pane at a slant counts; a
  thin bar is not replaced by a prism whose new back would lie on the card), and
  what still came to lie on one is pushed 5 mm off along the card's normal (the card never moves).
  Clear glass (`glass_clear`) is seen through by the reducer's probe and by the metric, as on screen: what stands
  behind it (a clock dial, a shop display) is judged, and nothing may be laid over it unseen.
  It draws nothing from `rng`. With `META["lod"]` it is built in comparison builds only (`tools/tsq lod`,
  `tools/lod_reduce_eval.py`): by rule 5 below it could not ship.
- **authored**: your modelled level, when `META["lod"]` is set.

No holes (rule g) in the levels the build makes. Reduced: (1) a face any street viewer sees in LOD0 is never
taken as unseen, whatever the 64 test directions say; (2) the faces that are taken are kept aside; (3) a small
piece whose removal opens a view stays, unless a kept-aside face right behind it closes the view (it then goes
back in); (4) a loose piece the street looks through after the collapse goes back one step of its ladder, then to
what it was (`log["holes_held"]`); (5) at the end (`_close`, also after the forced pass) every remaining hole
sample gets the kept-aside faces lying within 0.3 m behind it or else the LOD0 faces shown there and around,
collapsed as far as nothing opens again; such faces are marked and not touched afterwards. Decimated: the same
last step on the merged copy (`tk.final._close_holes`). Switches: `TSQ_HOLES=0` builds as before (to compare),
`TSQ_HOLES_GATE=0` skips the measure in the metric, `TSQ_REDUCE_TRACE=1` prints the street reading per step.
The last step goes over its viewer sets again while one still fails the gate, `ROUNDS` (4) times at most
(`tk.holes.close`): faces put back for one set can open a view for the other. What it puts back is not held
against the level's triangle budget.
The repair works on sampled rays (two sets of viewers), so it is not a proof: a third set of viewers
(`tools/checks/holes.py town --phase 0.12`) still finds slivers, about a tenth of what was there before. And the
decimated copy is no shelter when it shows through itself: a reduced candidate that does not, with no more
triangles than it and every other gate passed against it, ships instead (`choose`, `"last": True`).

The choice (`tk.final_levels.pick`, `tk.lodimage.choose`), on a point-splat picture of every candidate next to
LOD0 at the distance the level takes over (16 m / 44 m; `tk/lodimage.py`):
1. hard gates as in the rules above (sockets, parts, glow materials, bounds, budget);
2. image gates: outline difference <= 0.03 / 0.08, nothing torn open (<= 0.3 % / 0.5 % of the picture) and no
   piece that touched in LOD0 hanging loose, lit brightness 0.92-1.08 x LOD0, glow coverage >= 0.8 (judged when
   at least 0.5 % of the LOD0 picture glows) - a candidate that misses one only ships when the decimated copy is
   no better on it. Brightness: when both miss the band the image difference decides; when only the reduced copy
   misses it, it still ships if it is within 0.85-1.18 and its image difference is at least 15 % lower;
   *lettering gate*: every tagged group of 3 px or more is drawn alone at a finer scale; front area must stay
   within 0.85-1.15 of LOD0 and the outline difference under 0.10 on average / 0.30 for the worst group, a small
   group must be whole or gone (`stats.lods[].text`);
   *depth-fight gate*: the area of cards (see above) with a parallel face of another material closer than 3 mm
   (the measure of `tools/checks/zscan.py`, buckets A+B+C, opaque and glass partners; zscan.py itself counts
   glowing materials only) must not exceed LOD0's by more than 0.1 % of the card area (5-40 cm2), and neither
   may the area closer than 4 mm (`stats.lods[].zfight`: `z3` against `z0` + `tol`, `z4` against `z40` + `tol`;
   `dark_area` / `dark4`: the cards of unlit rooms and their part of `z4`). That measure samples four points a
   triangle; the same two limits hold for the exact one (`x3` against `x30`, `x4` against `x40`, `xdark4`: the
   area of the cards over which a near-parallel face of another material lies within 3 mm / 4 mm or crosses
   them, `tk.lodimage.z_pairs`). The decimated copy does not win over a candidate that passes these when it
   fails them itself;
   *torn field* (candidates over the budget, which are judged in absolute terms): `open` (above) <= 5 %;
   *local defects* (a mean over the picture and all views hides what is wrong in one place; each is taken from
   the worst of the 12 views around a ground / hanging asset - 8 azimuths at eye height, 4 from above - or the 8
   in front of a wall asset, so every lit and lettered face is looked at): `tile` - the worst tile (1/8 of the
   picture, half-tile steps) may differ by 0.085 / 0.11 on average, unless the decimated copy's worst tile is
   within 15 % of it; `sil_new` - pixels outside LOD0's outline grown by 2 px, <= 1.2 % / 3 % of the outline (a
   shard, a filled hole); `glow_min` - glow coverage in the worst view that shows a glowing face >= 0.85 (a
   dial going dark on one side); `leak` - around each light socket (up to 8) 3000 directions are followed as the
   shadow map would: the share in which LOD0 stops the light and the level lets it land on a face that looks at
   the light and is open to the outside, or on the ground / mounting wall where a viewer sees it, inside a
   shadow of LOD0 and next to another such direction, <= 0.5 % / 1 % (LOD2 stands beyond 44 m, where the town
   gives a lantern no shadow map). All four are relative gates like the others (a candidate that misses one only
   ships when the decimated copy is no better on it: 1.15 x for tiles, 1.5 x for outline and leak, 0.05 for
   glow); `lod_pick[].local` lists the ones the shipped level still misses in absolute terms. Also recorded,
   without a gate: `slit` / `thin` - lines as long as the size limit, however thin, where one looks through the
   level (`thin`: or onto something more than 3 px behind);
3. lowest image difference wins; the decimated copy only when it is better by more than 0.002, and never on that
   ground over a reduced copy of an asset that declares identity parts (`lod_standin` / `lod_keep`, above);
4. a reduced LOD2 may use up to 1.5 x its budget - 2.0 x for `class` building / landmark, where stopping short
   of the budget is better than tearing - when it passes every gate in absolute terms and beats the decimated
   copy; the cheapest step that passes ships (as `reduced`, `lod_pick[].over` 1.5 / 2.0): over a level inside
   the budget that passes every gate too, the 1.5 x one only wins with an image difference at least 15 % lower
   (otherwise: lower by more than 0.002), the 2.0 x one only when nothing cheaper keeps the outline whole.
   The forced collapse that brings a level under a limit goes loose piece by loose piece first; what that leaves
   over is collapsed the plain way (`lod_pick[].cand[].relaxed`), and the gates judge the result. Last resort: when
   nothing passes and the decimated copy itself breaks its lettering or the depth-fight gate, the best reduced
   candidate inside the allowance ships that misses the torn-field limit alone, with `open` <= 10 %
   (`lod_pick[].clean` is false then). And at either level: a decimated copy that fails the depth-fight gate
   in absolute terms is no shelter - when nothing passes, the best reduced candidate inside its allowance ships
   that keeps the clearance, its lettering and the tear / loose / glow gates in absolute terms, with an outline
   difference up to 1.5 x the limit, a brightness of 0.85-1.18 and an image difference up to 1.5 x the
   decimated copy's (`lod_pick[].last`). And when nothing passes and the decimated copy itself misses a
   local-defect gate in absolute terms, it is no shelter from one: the reduced candidate inside its allowance
   with the lowest image difference below the decimated copy's ships that misses nothing but local-defect
   gates against it (`last`, and `lod_pick[].local` says what it misses);
5. a modelled level that keeps its contract ALWAYS ships - when it misses an image gate that another candidate
   passes with a lower image difference, the build warns (`lod 1: lit brightness is 0.84 x lod 0 ... - the reduced
   copy passes ...`): fix the level. One that breaks its contract is replaced by the decimated copy, as before.
   (`tk.final_levels.AUTHORED_YIELDS` / `REJECTED_SHIPS` switch both; `tools/lod_tests/contract.py` pins them.)

`tools/tsq lod <id>` of an un-converted asset shows decimated | what ships and prints every candidate's
numbers; with `META["lod"]` the reduced copy is printed as a third column of numbers to beat. Weathering masks of a
reduced level are fitted per vertex (`finalize_level(seen=, fit=True)`); modelled levels keep the plain transfer.
`tools/lod_reduce_eval.py` measures all assets at once into `out/lod_eval/` (never into `out/glb` or `out/lod`);
`--remeasure 1` builds nothing and scores the candidates already there with the current metric (the earlier pick
is kept as `pick0`, every candidate's absolute gate reasons as `abs`) - the way to try a new gate on all records.
Known limits of the reducer (round 3b): the shadow casters it keeps are not pinned in the later steps (a drop,
weld or collapse may still move one - it shows as `leak`, and the gate decides; a beam thinner than the 3000
directions of `leak` resolve is not measured at all - street_clock's lit outline on the paving was one, look at
the paving round a lamp on the sheet); identity parts (`lod_standin`) take no part in the reducer's reasoning
about what is seen or covered, so what lies behind a kept part is kept too; composed children of modelled
levels (`reduce_child`) get no caster test; and a proud course smaller than the level's size limit whose backing
went as unseen can be flattened by the weld or the forced collapse, which leaves gaps thinner than the size
limit (tower_keep_ornate LOD2, base course by the door: neither the tear gate nor `slit` tells them from small
parts dropped on purpose - look at the sheet of a building that ships a forced collapse). Hairline cracks in a
weld (tunnel_mouth LOD1, round the keystone) have no gate either. The depth-fight gate only lets a reduced copy
through over a depth-fighting decimated one when that reduced copy passes every other gate: where it does not,
the decimated copy ships with coplanar faces lod 0 does not have (lamp_post_globe LOD2 had 0.31 m2 under 3 mm,
measured with `tools/checks/zscan.py`, until its generator got stand-ins) - scan the levels after a rebuild. The
leak gate tests lights whose `shadow_near` lies beyond the whole asset too, which the engine never shadows by the
asset itself (the pool lights of lamp_post_globe). And two builds of the same code do not always agree where a weld or a gate sits on its
limit (buttress LOD2: image difference 0.019-0.023 over four builds; warehouse LOD2 changed its pick) - cause
not found.
Blender is launched with `--python-use-system-env` and `PYTHONHASHSEED=0` by tsq and town.py, so the reducer's and
the generators' set / dict order is the same in every build (`stats.hash_pinned` in the sidecar).
Hand-author a level when the build still ships `decimated` for it, or when the reduced one is not clean
(`lod_pick[].clean` false): typically welded cloth, foliage and blossom, hanging baskets, very thin frames.

## Checklist per asset class
- **Lathed / turned furniture** (bollards, urns, posts, barrels): nothing for round sections (automatic); pick
  segments only where auto leaves > 12 around at lod 2; drop collars / beads under 1 cm into the profile
  (`lod_path`); remove undersides and faces bedded in the ground; fluting / reeding -> plain shaft at lod 2.
  A ring round a faceted body (hoop on staves) takes the body's corners, or its flats cut into the boards.
- **Ironwork** (railings, brackets, gates): keep EVERY bar (thinned runs read sparse and strobe) but make bars
  square 4-sided at lod 1 and fronts-only / 3-sided at lod 2; scrolls via `sweep` scale themselves, set
  `sides=lod_pick(8, 4, 3)`; finials -> a 4-sided spike; rivets, collars, welds -> `lod_upto(0)`.
- **Glazing** (windows, lanterns, shopfronts): the lit pane is rule d; glazing bars are rule e: fronts-only boxes
  at lod 1, at lod 2 one card with the bars or fewer / thicker bars. Keep frame depth (reveal) at lod 1: fronts-only
  frame members open the reveal. Shutters: one slab with a few louvre fronts. Hinges, putty, handles -> nothing.
- **Roofs**: tiles are the cost. lod 1: tile courses as stepped strips (one quad row per course), or stamped
  tiles as top faces only, two neighbours of a course under one face; eaves line, ridge and verge tiles as one
  extruded profile; lod 2: flat planes with eaves / verge thickness, ridge a prism. Keep the roof material(s).
- **Masonry** (walls, plinths, fountains, stairs): bevelled blocks -> chamfer or none; joints stay only as mask /
  material, not geometry, at lod 2; delete bedded and back faces; keep copings and string courses (silhouette).
  Curved basins: `lod_path` the section, `lod_segs` the plan. Water: keep the surface and its masks, drop ripples.
- **Foliage** (trees, hedges, planters): keep the full branching logic and every draw, mesh less: limbs with
  fewer sides and rings, twigs -> ribbons -> nothing; cards: keep a share and enlarge it so leaf AREA is constant.
  Never map one card onto several cells of the `leafy` mosaic (regular dot grid): stretch a one-cell card, or at
  lod 2 use the same `hedge` tints with a larger `leafy=` cell. Pull enlarged cards inward to hold the outline.
- **Cloth** (awnings, banners, flags): folds are mesh resolution: `surface` halves itself, `grid` cuts are yours;
  keep the scallop count if it is the silhouette (`lod_exact`), stripes are colour (rule e), hems / eyelets go.
- **Buildings that compose**: convert the shared children first (roof_gable, window_casement are done; dormer,
  chimney, door_plank, shop_window, wall_lantern are not: they arrive decimated inside you, and a tiled child
  at 10 % looks broken - house_timber's dormer roofs at lod 2). Plaster panels: whole quads plus one coarse
  densify; timbers: fronts and two visible sides; interiors -> nothing; ray casts / child bounds -> `lod_memo`.

## Common mistakes (all hit in the pilot)
1. rng drift: draws inside `if lod()...`; a helper that draws per emitted item (draw for all, emit a subset);
   numpy generators must consume the same shapes at every level (shuffle `(arange(n) * 0.618) % 1 < share`).
2. Measuring the level: ray casts, `world_bbox`, `scene_tris()` vs a cap, outlines sampled at mesh resolution,
   clipping tests, things planted on a primitive's vertices (sprigs on a ball) - all differ per level and move
   sockets or change draws, also inside a fork. `lod_memo` them (the vertex list too).
3. Arithmetic on a `lod_pick` / `lod_exact` value returns a plain number, which the toolkit rescales again.
4. Code that indexes a primitive's vertices by the segment count it asked for, or deforms a lathe / sweep
   afterwards (twist, bulge, periodic waves): pass `lod_exact` counts, or skip the deformation at the level.
5. Hand-built rings, file-local mesh helpers, unit instances and counts of 8 or less (`cylinder(verts=8)`) are
   not auto-scaled: `lod_segs`, `lod_path`, `lod_pick(8, 4)`; simplify an instanced base BEFORE `instances_to_mesh`.
6. A bevel of 6 mm / 30 mm or wider stays as a chamfer; pass `bevel=lod_pick(w, 0.0)` to remove it. `text()` that
   is only measured (a stencil, a brand mask) wants `with lod_as(0):` so every level reads the same glyphs.
7. A local variable named `lod` shadows `lod()` (turret_roof, stairs_plaza): rename it. Emptied objects have zero
   polygons: guard code that reads them (`assign` of a list that became empty, `max()` over no vertices).
8. Flat facets replacing a round part flash under a lantern: keep `smooth()` and at least 6 sides at lod 1.
9. Shadows are cast by back faces: a fronts-only board over a lit pane leaks light, and a one-sided roof plane
   casts none - keep a closed (if crude) volume wherever light must be blocked.
10. Adaptive generators (fill until a triangle target) must take the target from `lod_budget()`, and their own
    tallies from `lod_memo`; padding added to steer the decimator is skipped at the levels (`if lod() == 0`).
11. Skipping placements of a `use()`d child that carries sockets changes sockets: place all, let the child shrink.
12. Only the default variant was checked: budgets are per variant and the plainest one fails first; build all.
13. New material at a level ("uses materials lod 0 does not have"): allowed only as another finish of the SAME
    library material with the same tints (it then inherits the masks of all LOD0 finishes of it).
