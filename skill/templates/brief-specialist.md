# Brief: specialist / fix agent (long-lived owner of one directory or one problem)

Use for: the layout builder (owns `town/layout.py`), the baking agent, the LOD reducer, texture compression, the
GLB exporter, a scoped fix ("exactly these three assets"), a measured research question. The shape is always:
result wanted, ownership fence, what to read, how to verify, house rules, report format.

---

## Goal (one paragraph, result first)

`<What must be true when you are done, stated as something that can be checked: "every glowing face has >= 4 mm
clearance to any face behind it, measured by tools/checks/zscan.py", "the Low preset's worst frame at 4x CPU
throttle stays within 5 % of the live page", "scene judge problems 3, 5 and 7 (quoted below) no longer appear in
shots 04_cafe and 13_market".>`

This is a `<fix | measurement | feature>`, not a redesign: the look from the front must stay the same to the eye;
variant names, param names, sockets, origins and footprints stay unchanged unless the goal says otherwise.

## You own

- `<files / directories you may edit>`
- Do not edit: `<everything else, named: other agents' files, tk/, docs/, the live page>`. Report defects in other
  owners' files instead of fixing them; if you need a change in a file you do not own, deliver it as "replace X
  with Y" edits where each X occurs exactly once (`assert s.count(X) == 1`).
- Live authors are editing `<assets ...>`: do not touch them; a build failure inside such a file is not yours —
  wait a minute and retry.

## Read first (and nothing else up front)

`<3–6 files, with what to look for in each. Include the judge's JSON or the user's quoted complaint verbatim, with
shot names and pixel positions.>`

## Method and verification

- `<the commands: tools/tsq ..., python3 tools/..., node tools/...>`; render / measure before and after and keep
  both; a claim without its picture or number is not done.
- Answer each question with a measurement, not an opinion: report rms AND signed bias; a worst-region check as
  well as the average (an average image gate passed torn clock hands); paired captures 3 cm apart for flicker;
  pin shader `time` and animation phases for any A/B.
- Measure cold (fresh browser, empty cache), paired, more than once; say when it was once. CPU throttling slows
  the main thread only.
- Machine: at most `<N>` Blender processes (check `pgrep -c blender` before launching); the lock files
  `<BAKING | REBUILDING>` mean wait; bake into a staging directory and swap, other agents read the live output.
- Browser work only through the project's browser CLI on YOUR OWN Chrome: `TSQ_CHROME_PORT=<port>
  tools/gpu_chrome.sh`, then `BU_NAME=<name> BU_CDP_URL=http://localhost:<port> browser-use ...` on every call;
  one tab, reused; check the tab's URL before `goto_url`; verify the adapter is the real GPU; stop the browser when
  idle. Never touch the shared Chrome on :9222 or another agent's port. Serve over localhost HTTP, never `file://`.

## House rules

- Scratch only in `.scratch/<task>/` (never `/tmp`, `/dev/shm`); delete it at the end.
- No wildcard characters in shell paths; no `rm`/`mv`/`cp` with shell variables; tidy up from a python script with
  explicit absolute paths. Run commands from the project root. Foreground `sleep` is blocked: poll from python.
- Keep command output bounded; never print a whole manifest, layout JSON, town.json or transcript.
- Pages use relative URLs only; any localhost URL for the user is rewritten to `<public proxied form>`.
- Nothing is pushed, published or uploaded anywhere.
- Do not restart shared servers or the render daemon; say so if one is down.

## Report

Result first; the numbers with their conditions (device, preset, link speed, runs); what was NOT verified; files
changed; what you cleaned up. Keep "measured" and "estimated" apart. No internal agent names.
