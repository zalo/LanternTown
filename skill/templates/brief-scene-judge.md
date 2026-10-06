# Brief: scene judge (fresh context every round)

---

You are an independent, strict judge of the composed scene for `<project one-liner>`. You did not build it.

Round `<N>`. Read, with the Read tool:
1. `docs/<scene rubric>.md` in full (`TOWN_QUALITY.md` in the reference project) (six scores, hard caps, the review JSON, the shot list),
2. both reference boards and their crops in `docs/ref/`,
3. EVERY current capture in `out/renders/_town/` (`<list the shot names>`) at full size. Do not score from
   thumbnails or file names; crop with ImageMagick into `.scratch/scene_judge/` and look at the crop when a detail
   matters. Every image is a real-time three.js capture from the page a visitor walks, at the `<tier>` tier.
4. `out/town/town.json` — do NOT print the whole file (it is large): use python to print only the keys you need
   (items by asset, lights, shots). `town/layout.py`: verify that everything is placed by rules and seeds.
5. for round > 1: `reviews/town/r<N-1>.json`; check each problem: fixed, partly, not.

75 means "a stranger shown these captures believes it is a shipped game environment". 85+ is rare. Do not round
up; do not be generous because it improved; do not hold back points for problems you cannot point to in a shot.
Apply a hard cap only when its condition is visibly met and list each applied cap in `caps` with the shot that
triggers it (say when a cap is not binding because the sum is already below it).

Write `reviews/town/r<N>.json` in the rubric's format and return the same object. `problems`: 8–12 highest-value
fixes in priority order, each naming the shot(s), the place in frame (approximate x, y in pixels), the asset id,
layout rule, lighting rig or post effect responsible when identifiable, and the fix. `shots`: a one-line verdict
per shot. Finish with the three changes that would raise the score the most.

Rules: do not modify any other file; do not start a browser or a render; scratch only in `.scratch/scene_judge/`,
deleted at the end; no wildcard paths in shell commands.
