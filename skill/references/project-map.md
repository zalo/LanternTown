# Project map: what is in `project/`, what each stage needs, what to change for a new theme

All host-side scripts are Python 3 stdlib only. Blender work runs as `blender --background --factory-startup
--python-use-system-env --python tools/<script>.py -- <args>` with `PYTHONHASHSEED=0`; the Blender side prints
`TSQ_RESULT {json}` lines the host parses. Blender is found by `TSQ_BLENDER`, `BLENDER`, `blender` on PATH, then
`~/.local/bin/blender`. Every tool computes the project root from its own location, so keep the layout.

Browsers: three private headless Chromes with hardware WebGPU, started by `tools/gpu_chrome.sh` (`CHROME_BIN`,
`TSQ_CHROME_PORT`, `TSQ_CHROME_PROFILE`; profiles under `.scratch/chrome/`): :9223 the render daemon, :9224 scene
captures, :9229 the measurement tools. They are driven through the `browser-use` CLI (`BU_CDP_URL=http://localhost:92xx
BU_NAME=<name>`), so that CLI is a dependency. `tools/serve.py` (:8765, `TSQ_PORT`) must run for anything browser-driven.

## Pipeline

| stage | command | scripts | in -> out | needs |
|---|---|---|---|---|
| material bake | `tools/tsq mat <name>[,..] [--res N]`; `tsq mats` | `tsq` -> `bl_mat.py` (Blender as numpy host) -> `materials/<name>.py build(t)` -> `tk/tex.py` | `materials/*.py`, `docs/materials.json` -> `out/mat/<name>/{albedo,normal,orm}.{png,webp}`, `preview/far/sheet.png` | Blender, `magick` |
| material swatch | `tsq render-mat <name>` | queue job -> `renderd.py` -> `web/render.js` | -> `out/renders/_mat/<name>.png` | serve, renderd |
| asset build | `tsq build <id> [--variants] [--params JSON --name x] [--lods] [--out DIR]` | `tsq` -> `bl_build.py` -> `assets/<cat>/<id>.py build(p, rng)` + `tk/` (`tk.final`, with `--lods` `tk.final_levels`, `tk.reduce`, `tk.lodimage`, `tk.holes`) | `assets/**` (or `release/**` with `TSQ_ASSET_DIR=release`) -> `out/glb/<id>/<variant>[.lod1|.lod2].glb` + `.json` sidecar (tris, size, sockets, lights, parts, warnings, lod modes, holes) | Blender 4.1 |
| LOD comparison | `tsq lod <id> [--variant v] [--auto] [--no-render]` | `bl_build.py --lods cmp` + job `lod` | -> `out/lod/<id>/`, `out/renders/_lod/<id>.png` | Blender, serve, renderd |
| review render | `tsq render <id> [--no-build] [--timeout S]` | `tsq` writes `out/queue/<job>.json`; `renderd.py` (inside `browser-use`, from `renderd.sh`) drives one tab of `web/render.html` + `render.js`, which POSTs to `/api/upload` | -> `out/renders/<id>/{sheet,hero,variants}.png`, `out/queue/done/`, heartbeat `out/queue/daemon.json` | serve, Chrome :9223, renderd (`PAGE_VERSION` in `tsq` and `render.js` must match) |
| judge | subagent (`templates/brief-judge.md`) | reads rubric, renders, `tsq brief <id>`, source, ref crop | -> `reviews/asset/<id>/r<N>.json` (`reviews/mat/`, `reviews/town/`) | nothing |
| scene layout | `tools/town.py gen [--seed N] [--set k=v]` | imports `town/layout.py generate()` (`TSQ_TOWN=town2`) | sidecars, catalog -> `out/town/layout.json` | nothing |
| scene builds | `town.py build [--procs 8] [--force] [--only a,b]` | `bl_build.py` per distinct (asset, variant, params) with LODs | layout + `release/` -> `out/glb/<asset>/_<hash>.glb` | Blender x procs |
| terrain | `town.py terrain` | `bl_terrain.py` | layout terrain + water cut-outs -> `out/glb/_town/{terrain,water}.glb` | Blender |
| resolve | `town.py resolve` | `town.py` + `layout.check_resolved()` | layout + sidecars -> `out/town/town.json` (items, protos, lights, colliders, regions, shots) | nothing |
| release | `town.py release [--hold a,b] [--only a,b]` | copies `assets/**` -> `release/**` | | |
| scene captures | `town.py render [--shots] [--quality] [--size] [--out DIR] [--look "k=v&.."]` | Chrome :9224 -> `web/town/?capture=1&token=..` | `town.json`, glbs, mats -> `out/renders/_town/<shot>.png` (ALWAYS `--out` for anything but the judged set) | serve, browser-use |
| web bake | `python3 tools/webbake.py [--out out/web/_next_x] [--jobs N] [--keep-raw] [--pack-only] [--only full|lean] [--pvs] [--no-gao] [--set k=v]` | `webbake.py` -> `webbake_bl.py` (lighting, AO, chunking) -> `lsvb_refit.py` + `lsvb_pack.mjs` -> `node webbake_pack.mjs` -> WebP -> `aomap.py` -> `node bake2glb.mjs --live --verify --drop-packed` -> `pvs.py` | `town.json`, glbs, mats -> `out/web/_next/` (`manifest.json`, `base.glb`, `{m,n,r,f}_<cell>.glb`, texture pages, `gao.png`, `gsh.webp`, `pvs.json`, `lean/`, `.raw/` with `--keep-raw`) | Blender, node + node_modules, `magick`; `--pvs` needs serve + GPU Chrome |
| one-file GLBs | `node tools/bake2glb.mjs <bake> --mode portable|exact --level .. / --lods .. [--ground ..]`; `node tools/masterglb.mjs` | | bake -> `out/web/_nextglb/town_*.glb`, `out/web/_master/town_master.glb` | node (draco3dgltf for portable) |
| checks | `python3 tools/checks/holes.py town|<id>/<variant> [--out F]`; `tools/checks/zscan.py` (Blender worker over a listing); `python3 tools/pvs.py --holes`; `pvs.py --check --n 200 [--allow 8]` | `tk/holes.py`, `web/engine/pvs.js` | built GLBs / bake -> JSON reports, `<bake>/.raw/holes.json` | Blender; serve + Chrome for pvs |
| measurement | `tools/measure.py --quality X [--walk]`; `perf_bench.py`; `perf_walk.py`; `perf_sim.py` (no browser); `perf_capture.py`; `perf_vram.py`; `lightpop.py`; `shoot.py`; `telemetry.py [--days|--sid|--errors|--devices]` | Chrome :9229 via browser-use; `serve.py` writes `out/.telemetry/<day>.jsonl` | | serve, GPU Chrome, nvidia-smi |
| serving | `python3 tools/serve.py [port]` | `serve.py` + `manifest.py` (gallery data, thumbnails) + `srczip.py` | routes `/`, `/out/`, `/node_modules/`, `/api/manifest`, `/api/source.zip`, `/api/upload`, `/api/telemetry`; gzip, Range, import-map stamping | nothing |
| publishing | `python3 tools/site.py [--out DIR] [--dry] [--data-out DIR] [--upload TAG]`; `.github/workflows/pages.yml` | hard-links pages + three + bake + GLBs + frozen manifest; content-hashed tar parts; `gh release` sync; Pages workflow assembles and deploys | -> `out/site/` (~850 MB), `out/site-data/` | `gh` |

Dev-only: `tools/cyclesref/` (path-traced reference of the scene), `tools/probetest/` (light-probe experiment),
`tools/lod_tests/` (toolkit contract tests), `tools/ktx_layers.py` (offline KTX2), `tools/glbopt.mjs` (Studio GLB
release), `tools/webbake_v1*` (the first bake generation).

## The common library `tk/` (imported by path from the project root)

| module | role | public names |
|---|---|---|
| `core.py` | params, seeded `Rng` (`fork`, `u`, `jitter`, `vary`, `pick`), scene and object basics | `P`, `resolve_params`, `params_hash`, `Rng`, `lod`, `clear_scene`, `root`, `empty`, `parent`, `mesh_object`, `from_bmesh`, `place`, `world_matrix`, `apply_transform`, `apply_modifiers`, `delete`, `duplicate`, `mirror`, `descendants`, `world_bbox`, `hex_rgb`, `lerp`, `clamp`, `smoothstep` |
| `geo.py` | builders in metres, Z up, front -Y, metre-scale UVs | `box`, `cylinder`, `cone`, `sphere`, `torus`, `lathe`, `rounded_rect`, `circle_pts`, `arch_pts`, `extrude_profile`, `shape_extrude`, `rounded_box`, paths (`resample_path`, `spiral_path`, `arc_path`, `catenary`, `bezier_path`), `sweep`, `chain`, `surface`, `grid`, `instances_to_mesh`, `trs`, `join`, `text`, `plank`, `stone_block`, `bolt`, `ring_band`; modifiers `bevel`, `subsurf`, `solidify`, `array`, `weld`, `simple_deform`, `boolean`, `subdivide`, `densify`, `displace_noise`, `transform_verts`, `recalc_normals` |
| `surf.py` | library materials, UV declarations, smoothing, weathering masks | `mat`, `assign`, `assign_faces`, `smooth`, `flat`, `uv_box`, `uv_grain`, `uv_cyl`, `uv_planar`, `uv_keep`, `uv_transform`, `realize`, `paint`, `height_ramp`, `facing`, `noise_mask`, `mask_mul`, `material_catalog` |
| `asm.py` | sockets, lights, fx, moving parts, composition | `socket`, `light`, `fx`, `part`, `use`, `Component`, `load_asset`, `variant_params`, `find_asset` |
| `levels.py` | author tools for modelled LODs (no-ops at level 0) | `lod_pick`, `lod_exact`, `lod_thin`, `lod_segs`, `lod_path`, `lod_budget`, `lod_memo`, `lod_upto`, `lod_drop`, `lod_faces`, `lod_as`, `lod_standin`, `lod_keep` |
| `final.py` | build side: merge per material, bake vertex masks (AO, cavity grime, edge wear, ground splash), validate, export GLB + sidecar | `finalize`, `mark_keep` |
| `final_levels.py`, `reduce.py`, `lodimage.py`, `holes.py` | build side: reduced-level candidates, the image metric and gates, pick-best, the no-holes check | (see `docs/API.md` "build side" sections) |
| `tex.py` | seamless procedural PBR synthesis in numpy | `Canvas` (`noise`, `fbm`, `ridged`, `blur`, `warp`, `cavity`, `edges`, `curv`, `ao`, `normal`, `stripes`, `wave`, `grid`, `cells`, `cracks`, `speckle`, `scratches`, `drips`, `fibres`), `Layout` (`rand`, `tilt`, `mask`, `rounded`, `dome`), `ramp`, `mix`, `colour`, `sat`, `smoothstep`, `remap`, `level`, `desaturate`, `overlay_tint`, `render_preview`, `render_far`, `save_material` |

`docs/API.md` and `docs/API_TEX.md` are generated from the docstrings (`python3 tools/gen_api.py`); `tools/tsq api <fn>`
prints one function's source so authors never open `tk/`.

## What a new theme changes

- `tools/catalog_src.py` -> `docs/catalog.json` (the briefs; keep the schema, replace the content) and `docs/ref/`
  (reference crops; the reference run's boards were private and are not shipped).
- `tools/gen_catalog.py` -> `docs/materials.json`: the material table (name, family, tile size, resolution,
  tintable, directional, mask responses); `materials/*.py` are the generators (57 shipped, reusable).
- `tk/final.py` `BUDGETS` per class (prop_small 6 k ... landmark 260 k); `tk/levels.py` `BUDGET` (1.0, 0.40, 0.10)
  and floors; `tk/asm.py` `light()` defaults (`#ffbf6a`, 40 cd, 9 m, flicker 0.12); the glow / two-sided /
  see-through material name lists repeated in `final.py`, `levels.py`, `lodimage.py`, `holes.py`, `final_levels.py`;
  `lodimage.py` LOD switch distances (16 / 44 m).
- `tools/webbake_bl.py` `BK` constants (16 m cells, bounce, moon and sky colours, lantern range scale) and the
  `LOOK` it imports from the engine; `web/engine/lite.js` `BAKED_TIERS` (distances, budgets, pixels, workers per
  preset) and `HDR` defaults; `web/engine/engine.js` night environment and Studio `TIERS`.
- `town/layout.py` `PARAMS`, the `BULK`, `FLAT`, `ACCENT`, `BANNERS`, `LAMP_CD` tables and `_generate` (keep the `Town`
  class); `tools/perf_walk.py`'s route; the judged shot list in `docs/TOWN_QUALITY.md`.
- `tools/site.py` `WEB_SKIP_DIRS/FILES`, `GLBS`, `DROP` (which pages and files the static site ships) and `DATA_TAG`
  in the workflow; `README.md`.
- `docs/AUTHORING.md` names the exemplar and the brief; `docs/QUALITY.md` / `TOWN_QUALITY.md` name the reference
  crops and the lantern-light mood.

## Web runtime (`web/engine/`, three ^0.186.1 WebGPU with WebGL 2 fallback)

`lite.js` the baked runtime (`createLiteEngine`, `BakedTown`, tiers, texture arrays, PVS, TRAA, ground maps);
`lite_glb.js` the thin GLB reader feeding the worker decoder; `lite_tc.js` + `bc7m6.js` / `astc4.js` on-device
texture compression; `natbloom.js` natural bloom + scatter fog + lens dirt/flare; `net.js` stall-resistant fetches;
`pvs.js` visibility-set baking in the page; `telemetry.js` visitor reports; `town.js` the dynamic Studio runtime;
`engine.js` the shared renderer (night environment, flicker lights, flames, GTAO, bloom, grade);
`ClusteredLightsNode.js` vendored Forward+ lights with `sourceRadius`, capped lanterns and a CPU fallback;
`assets.js` / `materials.js` GLB loading with sockets and parts, the TSL material library. Pages: `web/town/` (the
walk, all tiers, `?capture=1`), `web/gallery/`, `web/render.html` + `web/viewer.html` (server-only), `web/master/`
(one-file master GLB through the lazy loader), `web/glbtown/` (stock GLTFLoader + Draco, `?walk=bvh`), `web/gltflod/`
and `web/bakeglb/` (MSFT_lod demos). Import maps point at `../node_modules/three/build/three.webgpu.js`;
`serve.py` stamps script URLs by mtime for pages whose import map carries `data-stamp`.
