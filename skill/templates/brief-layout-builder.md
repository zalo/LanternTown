# Brief: layout builder (one long-lived owner of `town/layout.py`, the scene's layout generator)

---

You own `town/layout.py` and nothing else (you may run `tools/town.py gen | build | terrain | resolve | all` and
`tools/town.py render --out <your folder>`; you are the only agent allowed to run `town.py all`). Do not edit assets,
`tk/`, `tools/`, `web/`, `docs/`. Report asset defects; do not fix them.

## Goal

A composed `<environment: a square, a quay, a clearing, a hall, a street>` that an independent judge scores >= 75 on
walk-level three.js captures (`docs/<scene rubric>.md`), from rules and seeds only: no hand-placed one-offs, nothing
deleted or moved by instance index. Reference boards: `docs/ref/<...>`; the walk perspectives in `docs/ref/v_*.png`
are what the review cameras reproduce where the layout has them.

## Read first

The scene rubric; `town/layout.py` as it stands (the `Town` class is the machinery: `route`, `stair`, `cover`,
`place`, `settle`, `bays`, `put`, `audit`, `check_resolved`; `_generate` is the plan); `docs/catalog.json` for what
exists (`tools/tsq assets`); `reviews/town/r<N>.json` for the current judge's problems (quoted below).

## Rules of the plan

- Derive everything from a frame (the main open space, its axes, the focal feature's centre, the sockets of the
  pieces that enclose it): routes as the walks people take, a rectangle or ring of like light fittings that
  defines sub-areas, one anchored feature per sub-area, symmetric sets about the axes (even counts at equal angles,
  seats exactly radial or exactly aligned, planting in 2s or 4s, the same accessory on every member). Feature
  pieces end an axis, balance another, or come as mirrored pairs or even rows - no lone singles. A `match` helper
  places a symmetric set all-or-none.
- One light-fitting family per zone; a palette per zone (two or three recurring accent colours); one design per
  repeated accessory per zone; pin the variants that matter to the plan instead of `"*"`.
- Mirror the placement, not the seed: matched pairs still differ in variant and seed; an accent prop kind fronts at
  most one shot; a prototype appears once per 30 m sight cone.
- Loose dressing aligned to an edge or a host (yaw jitter <= 8 degrees), grouped at edges and activity spots; open
  ground stays open; nothing loose in the foreground 3 m of a camera, nothing cut by the frame edge, foreground
  ground not bare (paired planters, a line of posts, paving or path bands).
- Keep-outs: every stair or ramp plus 0.6 m, every declared route, entrance fronts, thresholds; a hung or
  wall-mounted piece needs support within 0.3 m; nothing floats, sinks, or runs through a guard or a basin; every
  fitting attached to something.
- Light: intensity scaled with the span it lights; no fitting within ~3 m of a camera's line to its subject; small
  table or path lights where the reference has them; a light budget of `<N>`.
- Cameras are part of the layout: eye height 1.65 m, FOV 62, 16:9; bulky pieces tested against sight cones (<= 14 %
  of a frame by rule, <= 5 % for a one-off); re-set a camera when the plan changes and count nudged items.
- Hero pieces the reference makes big need their own asset at true size: report the gap, never scale an asset up.

## Loop

`tools/town.py all` (release first: `town.py release --hold <live authors>`), then `town.py render --out
.scratch/layout/<round>/` and look at every shot with the Read tool; run the audit (`check_resolved` exits 1 on a
placement failure); only then hand the judged set over (the lead renders the judged set). When the judge or the owner
reports a defect, fix the **rule** and add an audit so the class cannot recur; test three rule values from a script
that rewrites the constant, re-runs `gen` and restores the file.

## House rules

Scratch only in `.scratch/layout/`; no wildcard paths; no `rm`/`mv`/`cp` with shell variables; your own Chrome port
`<92xx>` and `BU_NAME=<name>` on every browser call, one tab, stopped when idle; never print `town.json` or
`layout.json` whole; foreground `sleep` is blocked (poll from python).

## Report

Items / builds / assets / lights / colliders / triangles at LOD0; nudged, dropped and sight-blocked counts; which
judge problems are addressed by which rule; assets missing from the library; what was not verified.
