# Judging: running independent subagent judges that actually raise quality

> Criterion 4 is called `night` in the review JSON for historical reasons; it means "read under the environment's
> light" whatever the time of day.

The rubrics are `docs/QUALITY.md` (assets), `docs/MATERIALS.md` §"What good means" (materials) and
`docs/TOWN_QUALITY.md` (the composed scene), all in `references/docs/`. Brief templates: `templates/brief-judge.md`,
`templates/brief-scene-judge.md`. This file is what ~400 judge rounds showed about how to run them.

## Mechanics that mattered

- **Fresh context, never the author.** A judge gets the images, the brief, the rubric and (for assets) the source
  to check it is parametric. Round 4 of the scene onwards: "do NOT read earlier reviews before you have written
  your scores - earlier scores must not anchor you". Allow reading the previous review only to mark each listed
  problem fixed / partly / not.
- **Full size, cropped.** "View every image at full size with the Read tool; do not score from thumbnails or file
  names"; "make a crop with ImageMagick and look closely before you claim it - and equally before you dismiss it";
  "do not apply a cap for something you could not confirm in a crop". Scene captures are 2424 px wide; judges report
  coordinates on a 1600 x 900 downscale, and every brief that passes coordinates on must say so or crops land in the
  wrong place.
- **Score what is there**: "score what you see now, not the effort or the trend"; "75 means it ships without
  embarrassing the scene; 85+ is rare; do not round up - every problem you fail to name costs a revision round";
  "equally do not hold back points for problems you cannot point to in a shot".
- **Caps are explicit**: apply a hard cap only when its condition is visibly met, list it in `caps` with the shot
  or panel that triggers it, and say when a cap is not binding because the sum is already below it.
- **Output is JSON only** (`reviews/<kind>/<id>/r<N>.json`), 3-8 problems for an asset, 8-12 for the scene, each
  "what, where (panel / variant / shot, approximate pixel box), what instead (dimensions, cd, material)", then
  strengths to keep and "the three changes that would raise the score the most". Problems written like this were
  implemented one-for-one; authors reported by problem number.
- A cheap parametric check keeps judge context small: the first ~90 lines of the source plus `grep -c 'rng\.'`
  and `grep -n 'p\.'`, and a CONTRACT check (-4 for a missing contract param). Quick "read-only" judges (JSON in the
  final message, no file) give a number in minutes without file collisions.
- **Re-judge after every revision** and track "stale reviews" (passing scores that predate a rework). In one
  window ~45 of 55 revisions were self-assessed only. The reference run's revise-and-rejudge loop ran as a script
  with up to three cycles against the released source.

## What judges mark down, ranked (assets)

1. **Night read**: blown hotspots beside lanterns, glare blobs on glass, a navy hero with too few warm pixels, lit
   windows as flat cream cards with a bulb star, faceted pools from a bulb close to a low-segment shade.
2. **Weathering that is dirt, not story**: airbrushed halos, grime quantised to the mesh grid ("saw-tooth"), moss
   and rust masks rendering as black blotches ("charred", "tarred"), factory-new paint.
3. **Geometry faults**: inside-out shells, boards piercing a roof, a floating lantern, bright seam lines, an emblem
   overflowing its frame, intersecting parts (each a cap).
4. **Flat or generic surfaces**: plasticky trim with one brushed streak direction, bare rear elevations,
   featureless roof planes, no mortar joints, hair-thin brackets, flat windows with no reveal.
5. **Budget and draw calls**: 28-65 material variants against 10; iron at a third of a house; a gate at 240 k
   triangles; silent culling of a briefed feature to fit.
6. **Material identity**: gilding as flat yellow, iron as a black silhouette, slate chalk-white, lead pale.
7. **Contract honesty**: a fountain whose `diameter` is 1-2.4 m wider than requested; a span that overhangs
   +-length/2; a clamped range narrower than the brief.
8. **Water**: opaque black lids, cascades as "corrugated plastic" or "toothpaste".

Materials: closed-loop noise contours; evenly spread uniform features (leopard dots, pinstripe runs); saturated or
yellow colour and painted-in shading; Voronoi for soft forms; too clean or too flat; countable repeats in `far.png`.

Scene (five rounds, two scenes): water in every round until a water specialist; skyline rows reading as box hulls
(near the blockout cap); stone too clean and uniform or mismatched hue families side by side; lighting too even
("one amber mid-level", lamp panes as flat cream cards, speckle instead of stretched reflections); technical
noise (shadow moire, SSR confetti, roof-tile sparkle, thin-railing aliasing); furniture with no contact shadow;
clones and repetition (the same milk cans in four shots, mirrored stalls, five identical lanterns in a row); empty
foreground paving ("bottom 40 % of the frame is bare cobble"); large blank retaining walls filling half a frame.

## Scores to expect

- Asset first rounds cluster at 67-74; buildings and landmarks at 73-74 (one or two points under). One revision
  passes most; landmarks, vegetation and anything with water took three. Final distribution in the reference run:
  78 x 31, 76 x 28, 77 x 26, 79 x 19, 75 x 19, 80 x 7, 81 x 3; three never passed (a cat at 70 after five rounds, a
  civic hall at 60 held back from the scene).
- **Judge variance is about seven points** on the same asset across rounds (a backdrop row scored 67 / 76 / 70 /
  77 / 70 / 77 across reworks; a second judge dropped a sign from 74 to 60 for an inside-out shell the first had
  missed). Use two independent judges on a final round when the score decides a release.
- Scene: 66 -> 68 -> 74 -> 68 (uncapped 75, one floating weathervane) -> 81. Realism (13-14 / 20) and technical
  (6-7 / 10) were the ceilings every round; composition and lighting sat at 15-16.
- A judge's praise is a constraint for the next round ("no blown lamps or crushed blacks" after two rounds of
  complaints about blown lamps): give look agents the swing history and "aim for the middle".

## Tell judges about the harness, or they score it

- Review-rig artefacts that are not the asset's fault: a halo sprite at every light socket; the rig's key light is
  dimmed 6x when the asset has any light; distance fog scales with asset size, so a >= 20 m asset is washed out in
  daylight and dim at night (landmarks and backdrop rows need their own review scene at true distance); the
  `wall_ground` backing wall hides openings and below-ground parts; `variants.png` shows at most six variants,
  alphabetical, night only, and only the first lamp gets a shadow map; spinning parts take a wall-clock heading
  per render; soft-shadow grain on the wall persists with the lamp off; the hero view is ~13x dimmer than the
  close-up, so "underlit hero" and "blown close-up" can both be true.
- Bare metals render near-black in the studio daylight 3/4 view (no environment reflection) - a known engine limit,
  not a material fault.
- Judges sometimes prescribe what the engine cannot do (green moss via `special`, < 10 draw calls while composing
  catalogue parts, per-vertex emissive dimming). When several authors report "cannot be fixed from the asset", fix
  the harness or the toolkit before another round.
- Stale images: a judge scored a tree from a cached old GLB in the scene, and a redesigned lantern was compared
  against a stale gallery image. Check the capture's mtime against the model's before judging; never judge while
  engine files are mid-edit (shader compile errors looked like asset faults).

## Judge independence vs the one-off cap

Forbidding source access (round 5) meant the "hand-placed one-offs" cap could not be assessed. Hand the judge a
machine-generated layout audit (counts of literal placements, items per rule, nudged and dropped items) instead of
the source.
