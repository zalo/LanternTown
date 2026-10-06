# Quality guide for judging assets (and for authors to self-check) - theme-neutral template

Fill the `<...>` fields; keep the scores, caps and output format (the pipeline, the gallery and the briefs rely on
them). The reference run's filled-in version is `references/docs/QUALITY.md`.

---

Goal: a **realistic, lifelike `<environment: e.g. harbour quarter at dusk / forest shrine in morning mist / station
concourse under skylights>`**, walked through in first person in a real-time three.js renderer. Design language
from the reference board (`docs/ref/board.png` and its crops: `<crop names>`) - `<the recurring forms, materials
and motifs>` - but rendered with real materials, real construction and real wear instead of the board's
stylisation. The light: `<key light and emitters, e.g. "low warm sun with a cool sky fill" / "amber lamps and lit
windows against a navy night" / "flat overcast daylight">`.

The bar is a hero prop in a current AAA game or a good archviz kit piece, viewed from 1.5 m to 30 m.

## What the judge receives

For asset `<id>`: `out/renders/<id>/sheet.png` (studio daylight front 3/4 and back 3/4, a close-up under the
environment's light, clay render with a 1.8 m capsule figure for scale + triangle count and size),
`out/renders/<id>/hero.png` (under the environment's light, lit by its own emitters and the scene's key),
`out/renders/<id>/variants.png` (every variant), the asset's catalogue brief, the source file (to check that it is
truly parametric) and the reference crops.

## Score sheet (100 points, pass >= 75)

1. **Silhouette, proportion and scale (20)** - instantly readable as the object it claims to be; convincing
   proportions; correct size next to the 1.8 m figure; an appealing, characterful outline in the reference's design
   language. *0-8* wrong object / primitive blob / wrong scale. *9-14* recognisable but generic or boxy. *15-17* good.
   *18-20* striking, could be concept art for the scene.
2. **Geometric detail and construction (20)** - primary, secondary and tertiary forms; believable construction
   (joints, fixings, thicknesses, how parts meet); bevelled edges catching light; natural irregularity. The clay
   render must already look like a finished model. *0-8* bare primitives, paper-thin or intersecting parts, razor
   edges. *9-14* shapes there but sparse, repetitive or CG-perfect. *15-17* rich. *18-20* exquisite at 1.5 m.
3. **Materials and weathering (20)** - the right library material on each part with correct texture scale and
   direction; believable contrast between materials; art-directed wear that tells a story (`<the environment's
   weathering: salt and rust / moss and damp / dust and sun-fade / scuffs and polish>`); nothing plasticky, nothing
   uniformly noisy, no stretched UVs. *0-8* wrong / flat / stretched. *9-14* correct but uniform, factory-new or
   dirt everywhere. *15-17* convincing. *18-20* photographic.
4. **Read under the environment's light (20)** - in `hero.png` and the close-up: does it look beautiful and real
   under `<the light>`? Emitters (if any) visible and plausible, intensities balanced (`<rich pools, not blown
   white, not dead>`), glass and metal respond, the asset contributes atmosphere. Unlit assets are judged on how
   they take the scene's light (specular breakup, relief, depth). *0-8* blown out, black, or flat. *9-14*
   acceptable. *15-17* atmospheric. *18-20* a screenshot you would post.
5. **Parametric variation (10)** - variants clearly different, all plausible, none broken; params meaningful (size,
   style, colour, condition); seeded irregularity visible; no hand-placed one-offs or dead params in the source.
6. **Technical cleanliness (10)** - no flipped normals / holes / z-fighting / floating or sunken parts / visible
   seams; sits properly on its mount; triangle count within budget and spent sensibly; sensible material count.

**Hard caps** (apply after summing):
* does not read as the briefed object, or is grossly out of scale: total <= 50
* visibly broken geometry or shading in any view (holes, inverted faces, exploded parts): total <= 60
* a variant is broken or all variants look the same: total <= 70
* lighting blown out to white or the asset is essentially invisible under the environment's light: total <= 70

Be strict and specific. A 75 means "ships in the scene without embarrassing it". 85+ is rare and means genuinely
excellent. Do not round up out of kindness: every problem you fail to name costs a revision round later.

## Review output

Write `reviews/asset/<id>/r<N>.json` (N = round number) and return the same object:

```json
{"id": "<id>", "round": 1, "score": 71,
 "silhouette": 15, "detail": 13, "materials": 14, "night": 15, "variation": 7, "technical": 7,
 "caps": [],
 "problems": ["most important, concrete, actionable problem with where it is visible", "..."],
 "strengths": ["what must be kept"]}
```

(`night` is the key of criterion 4 whatever the environment's light; the gallery reads that key.)
`problems`: 3-8 items, ordered by impact, each saying *what* is wrong, *where* (which panel / variant / part) and
*what it should be instead* (with dimensions or references when useful).

## Scratch files

Never write renders, crops, contact sheets or any other scratch to `/tmp` or `/dev/shm`. Use `.scratch/<task>/` in
the project root and delete that folder when you finish. Do not use wildcard paths in shell commands.
