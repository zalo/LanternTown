# Quality guide for judging the composed town square

Goal: a **realistic, lifelike, lantern-lit old-town square at night** that a visitor walks through
in first person in a real-time three.js (WebGPU) renderer. Layout and mood follow the reference
board (`docs/ref/board.png`; crops `plaza.png`, `gate.png`, `cafe.png`, `monument.png`,
`canal.png`): an entrance street rising to a gated plaza, a fountain / monument court, a raised
cafe terrace, half-timbered and stone houses with clay roofs, cast-iron lamps, shopfront signs and
neon, golden lantern light against a deep navy night. The board is painterly; the town must take
its *design* from the board and its *rendering* from reality.

A second board, `docs/ref/board2.png`, gives nine perspectives the town is modelled after (crops
`v_overview.png`, `v_items_shop.png`, `v_alley_arches.png`, `v_canal_hotel.png`,
`v_star_monument.png`, `v_accessories_stairs.png`, `v_market_arcade.png`,
`v_skyline_parapet.png`, `v_cafe_court.png`): wet, glossy paving that mirrors the lanterns; a dark
ambient with many small warm pools (pier, wall and bracket lanterns, lit windows); a layered, lit
skyline under a cloudy night sky; the star monument court with flanking stairs and an inscription
wall; shop fronts with painted and neon signs; a quay with a hotel sign over the canal; an alley
spanned by arches and hung with banners; a covered market arcade; a cafe court with parasols and
candle-lit tables; flowers, ivy and banners everywhere. The review viewpoints are chosen to match
these perspectives where the layout has them.

The bar is a night-time environment screenshot from a current AAA game. The judge is looking at
what a player would see - every image is a real-time three.js capture, not an offline render.

## What the judge receives

* `out/renders/_town/<shot>.png` - the review viewpoints (eye-height walk shots plus one or two
  overviews), captured from the same page a visitor walks (`web/town/`), at the `high` tier.
* `out/town/town.json` - the resolved layout (items, lights, colliders, shots) and
  `town/layout.py` - the generator (to verify that everything is placed by rules and seeds).
* The reference crops in `docs/ref/`.
* Asset review scores in `reviews/asset/*/r*.json` (context only: the town is judged on what is
  visible in the shots, not on the scores of its parts).

Look at **every** shot at full resolution before scoring. Score the town as a whole; a single
broken shot limits the total (see caps).

## Score sheet (100 points, pass >= 75)

1. **Composition and layout (20)** `composition` - does it read as a real town square: enclosed
   space, clear routes, layered depth (foreground props, mid-ground facades, skyline), focal
   points (fountain, clock tower, gate, cafe) framed from the walk views; believable street widths,
   storey heights, level changes and stairs; nothing floating, buried, intersecting or blocking a
   route; no empty lots or visible edge of the world.
   *0-8* a scatter of objects on a plane. *9-14* recognisable square with gaps, collisions or
   awkward scale. *15-17* convincing. *18-20* every view is a composed frame.
2. **Lighting and atmosphere (20)** `lighting` - warm lantern pools against cool moonlit shadow;
   readable but genuinely night; light motivated by visible emitters; lit windows; soft falloff,
   contact shadows / AO, fog depth; bloom and exposure controlled (no blown-white lamps, no
   crushed-black facades, no flat ambient grey).
   *0-8* flat, blown or black. *9-14* acceptable but uniform or noisy. *15-17* atmospheric.
   *18-20* you want to stand in it.
3. **Realism of rendering and materials (20)** `realism` - surfaces read as stone, timber,
   plaster, tile, iron, glass, water at walking distance; texture scale consistent between
   neighbouring assets; wear and wetness where expected; no plastic, no untextured or
   placeholder-coloured surfaces, no visible tiling, no cartoon saturation.
   *0-8* blockout. *9-14* game-like but obviously CG. *15-17* convincing. *18-20* photographic.
4. **Density and storytelling (15)** `density` - props, signage, planting, clutter and trim that
   make the place lived-in; variety (no row of identical clones); detail holds up in close shots
   and the far end of each view is not bare.
5. **Faithfulness to the reference (15)** `reference` - the board's districts, landmarks, palette
   and design language are recognisable (lantern motif, gate, fountain court, cafe terrace,
   half-timber + stone + clay roofs, signs and neon), translated to realistic rendering.
6. **Technical cleanliness (10)** `technical` - no z-fighting, popping LOD seams, shadow acne or
   peter-panning, light leaks, missing faces, placeholder / blockout boxes, clipping cameras,
   aliasing shimmer or banding.

**Hard caps** (apply after summing):
* any blockout / placeholder volume or missing building visible in a shot: total <= 60
* a shot whose camera is inside or blocked by geometry, or an asset visibly broken, floating or
  intersecting another in any shot: total <= 68
* lighting flat / blown out / too dark to read in more than two shots: total <= 65
* does not read as the reference's lantern town square: total <= 55
* hand-placed one-off geometry or positions in `town/layout.py` instead of seeded rules: total <= 70

Be strict and specific. 75 means "a stranger shown these captures believes it is a shipped game
environment". 85+ is rare. Do not round up: every problem left unnamed costs a revision round.

## Review output

Write `reviews/town/r<N>.json` (N = round number) and return the same object:

```json
{"round": 1, "score": 62,
 "composition": 14, "lighting": 13, "realism": 12, "density": 9, "reference": 9, "technical": 5,
 "caps": ["blockout visible in 05_gate"],
 "problems": ["most important, concrete, actionable problem: which shot, where in frame, what it should be instead", "..."],
 "strengths": ["what must be kept"],
 "shots": {"01_approach": "one-line verdict", "...": "..."}}
```

`problems`: 5-12 items ordered by impact, each naming the shot(s), the place in frame, the asset
or system responsible when identifiable (asset id, layout rule, lighting rig, post-processing) and
the fix.

## Scratch files

Never write renders, crops, contact sheets or any other scratch to `/tmp` or `/dev/shm` (RAM disks with a quota). Use `.scratch/<your-asset-or-task>/` in the project root (normal disk, not served) and delete that folder when you finish.

Do not use wildcard paths (`*`, `?`, `[..]`) in shell commands: they trigger an approval prompt that stops work. Name paths explicitly or do the matching inside a python script.
