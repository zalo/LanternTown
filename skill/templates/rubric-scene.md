# Quality guide for judging the composed scene - theme-neutral template

Fill the `<...>` fields after the first full set of review captures exists, so the rubric can name real shots.
Keep the scores, caps and output format. The reference run's version is `references/docs/TOWN_QUALITY.md`.

---

Goal: a **realistic, lifelike `<environment>`** that a visitor walks through in first person in a real-time
three.js (WebGPU) renderer. Layout and mood follow the reference board (`docs/ref/board.png`; crops `<...>`):
`<the zones and their order: e.g. an approach rising to an enclosed main space, a focal feature, a raised terrace,
an edge to water or landscape>`, `<the building or landscape vocabulary>`, `<the light: key, emitters, atmosphere>`.
The board is stylised; the scene must take its *design* from the board and its *rendering* from reality.

A second board, `docs/ref/board2.png`, gives `<N>` perspectives the scene is modelled after (crops `v_*.png`):
`<one line each: what the view shows and the mood it carries>`. The review viewpoints match these perspectives
where the layout has them.

The bar is an environment screenshot from a current AAA game. The judge is looking at what a player would see:
every image is a real-time three.js capture, not an offline render.

## What the judge receives

* `out/renders/_town/<shot>.png` - the review viewpoints (eye-height walk shots plus one or two overviews),
  captured from the same page a visitor walks (`web/town/`), at the `<tier>` tier. Shot list: `<names>`.
* `out/town/town.json` - the resolved layout (items, lights, colliders, shots) and `town/layout.py` - the generator
  (to verify that everything is placed by rules and seeds), or a machine-generated layout audit in their place.
* The reference crops in `docs/ref/`.
* Asset review scores in `reviews/asset/*/r*.json` (context only: the scene is judged on what is visible).

Look at **every** shot at full resolution before scoring. Score the scene as a whole; a single broken shot limits
the total (see caps).

## Score sheet (100 points, pass >= 75)

1. **Composition and layout (20)** `composition` - does it read as a real place: enclosed or framed space, clear
   routes, layered depth (foreground, mid-ground, backdrop), focal points framed from the walk views; believable
   dimensions and level changes; nothing floating, buried, intersecting or blocking a route; no empty lots or
   visible edge of the world. *0-8* a scatter of objects on a plane. *9-14* recognisable with gaps, collisions or
   awkward scale. *15-17* convincing. *18-20* every view is a composed frame.
2. **Lighting and atmosphere (20)** `lighting` - `<the intended contrast: warm pools against cool shadow / low sun
   and long shadows / soft overcast with occlusion>`; readable but true to the time of day; light motivated by
   visible sources; soft falloff, contact shadows / AO, atmospheric depth; bloom and exposure controlled (no blown
   whites, no crushed blacks, no flat ambient grey). *0-8* flat, blown or black. *9-14* acceptable but uniform or
   noisy. *15-17* atmospheric. *18-20* you want to stand in it.
3. **Realism of rendering and materials (20)** `realism` - surfaces read as what they are at walking distance;
   texture scale consistent between neighbouring assets; wear and wetness or dust where expected; no plastic, no
   untextured or placeholder-coloured surfaces, no visible tiling, no cartoon saturation. *0-8* blockout. *9-14*
   game-like but obviously CG. *15-17* convincing. *18-20* photographic.
4. **Density and storytelling (15)** `density` - props, signage, planting, clutter and trim that make the place
   lived-in or used; variety (no row of identical clones); detail holds up in close shots and the far end of each
   view is not bare.
5. **Faithfulness to the reference (15)** `reference` - the board's zones, landmarks, palette and design language
   are recognisable, translated to realistic rendering.
6. **Technical cleanliness (10)** `technical` - no z-fighting, popping LOD seams, shadow acne or peter-panning,
   light leaks, missing faces, placeholder volumes, clipping cameras, aliasing shimmer or banding.

**Hard caps** (apply after summing):
* any blockout / placeholder volume or missing structure visible in a shot: total <= 60
* a shot whose camera is inside or blocked by geometry, or an asset visibly broken, floating or intersecting
  another in any shot: total <= 68
* lighting flat / blown out / too dark to read in more than two shots: total <= 65
* does not read as the reference's place: total <= 55
* hand-placed one-off geometry or positions in the layout instead of seeded rules: total <= 70

Be strict and specific. 75 means "a stranger shown these captures believes it is a shipped game environment".
85+ is rare. Do not round up: every problem left unnamed costs a revision round.

## Review output

Write `reviews/town/r<N>.json` and return the same object:

```json
{"round": 1, "score": 62,
 "composition": 14, "lighting": 13, "realism": 12, "density": 9, "reference": 9, "technical": 5,
 "caps": ["blockout visible in 05_gate"],
 "problems": ["most important, concrete, actionable problem: which shot, where in frame, what it should be instead", "..."],
 "strengths": ["what must be kept"],
 "shots": {"01_approach": "one-line verdict", "...": "..."}}
```

`problems`: 5-12 items ordered by impact, each naming the shot(s), the place in frame (coordinates on the
`<1600 x 900>` downscale), the asset or system responsible when identifiable (asset id, layout rule, lighting rig,
post-processing) and the fix.

## Scratch files

Never write scratch to `/tmp` or `/dev/shm`; use `.scratch/<task>/` in the project root and delete it when done. Do
not use wildcard paths in shell commands.
