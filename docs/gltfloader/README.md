# GLTFLoader: MSFT_lod

Two patches against three.js `dev` (457581a), kept as local branches in `upstream/three.js`, one commit each. Nothing
has been pushed or proposed upstream. `SUBMIT.md` has the steps to do that.

| file | what |
|---|---|
| `tier-a-gltfloader-msft-lod.patch` | Tier A, branch `gltfloader-msft-lod` (bd87ef4, on `origin/dev` 457581a): `MSFT_lod` read into `THREE.LOD`, no API change |
| `tier-a-PR.md` | pull request description for it |
| `tier-b-gltfloader-lazy-lod-range-requests.patch` | Tier B, branch `gltfloader-msft-lod-lazy` (078fd88, one commit on top of A): `setLazyLOD()`, `setRangeRequests()` |
| `tier-b-PR.md` | pull request description for it |
| `SUBMIT.md` | how to submit the two, Tier A first |

The patches are `git format-patch` output: A applies to `origin/dev`, B applies on top of A. The tips before the last
rebase and squash are kept as the tags `pre-rebase-a` (5fa7ba5, on 2bebd15) and `pre-rebase-b` (0caf99b).

Test kit: `out/web/_nextglb/msft_lod_kit/` (`check.html?tier=a|b`).

Demo: `web/gltflod/`. The default file is `out/web/_nextglb/portable_lod_spawn.glb` (stock materials, Draco, WebP;
no adapter). Parameters:

| parameter | |
|---|---|
| `?tier=a` / `?tier=b` | eager / levels on demand + range requests |
| `&lazy=0`, `&range=0` | tier B with one of the two switched off |
| `&src=glb`, `&src=bins` | the town's own vertex format (`exact_lod_spawn.glb`, or `.gltf` + `.bin` files); needs the material adapter of the demo |
| `&file=<url>` | another file, e.g. `/out/web/_nextglb/town_lod.glb` (the whole town) |
| `&refine=hop`, `&refine=step` | level order experiments, not in the branch: one step off the coarsest level first, or every level on the way |
| `&dist=<0.05..4>` | scales the LOD distances of the file after loading |
| `&all=1`, `&tint=1`, `&view=<name>` | request every level, colour by level, start at another camera |

`web/gltflod/lib/` holds the built loaders: `GLTFLoader.tierA.js` (bd87ef4), `GLTFLoader.tierB.js` (078fd88), and
`GLTFLoader.tierB.hop.js` / `GLTFLoader.tierB.step.js` (still built from 0caf99b, the tip before the rebase, with the
level order experiment; search for `EXPERIMENT`). The page publishes what it measured on `window.__stats` (`transport`,
`ranges_honoured_206`, `fell_back_to_full_200`, `requests`, `bytes_fetched`, `concurrent_peak`, `first_frame_ms`,
`view_complete_ms`, ...).
