# MSFT_lod test kit for three.js GLTFLoader

A test bed for adding `MSFT_lod` to `GLTFLoader`: small synthetic models, one realistic model, the expected result of
each as data, a page that checks a loader against them, and a reference implementation as one plugin file.
The folder is self-contained: copy it anywhere, it refers to nothing outside itself (three.js comes from a URL you give).

| file | bytes | what |
|---|---|---|
| `generate.mjs` | 29,251 | writes the synthetic models and `expected.json` (node, no dependencies) |
| `models/` | 22,743,916 | 13 models in 34 files (list below) |
| `expected.json` | 23,746 | per model: the LODs, their levels, bounds, reference distances, buffers per level |
| `check.html`, `check.js` | 27,279 | loads every model with the loader you name and prints pass / fail per expectation |
| `setup_reference.js` | 801 | the adapter that plugs the reference plugin into the check |
| `GLTFMSFTLodExtension.js` | 33,889 | the reference plugin (tier A and tier B), depends on `three` only |
| `server.mjs` | 5,825 | static server that answers HTTP range requests (node, no dependencies) |

## The extension

Spec: https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/MSFT_lod/README.md (status: complete, vendor
extension). Schema: `ids`, an array of indices, at least one, on a **node** or on a **material**.

```json
{
  "extensionsUsed": [ "MSFT_lod" ],
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [
    { "name": "High", "mesh": 0,
      "extensions": { "MSFT_lod": { "ids": [ 1, 2 ] } },
      "extras": { "MSFT_screencoverage": [ 0.25, 0.05, 0.004 ] } },
    { "name": "Mid", "mesh": 1 },
    { "name": "Low", "mesh": 2 }
  ]
}
```

The node (or material) that carries the extension is level 0, the most detailed one; `ids` lists the lower levels in
order of falling detail. `MSFT_screencoverage` sits in `extras` of the same object, one number per level (so
`ids.length + 1` of them): level `i` is meant for a screen coverage down to value `i`. The spec's own example
`[ 0.5, 0.2, 0.01 ]` ends with "nothing rendered from 0.01 - 0".

What the text says and what it leaves open:

- "If an asset containing this extension is loaded in a client that doesn't implement the extension, then the highest
  LOD level will be loaded and all lower LODs will be ignored." So the extension belongs in `extensionsUsed`, never in
  `extensionsRequired`, and the lower levels must not be reachable from a scene by any other way.
- `MSFT_screencoverage` is "a hint". What coverage means (of what bound, measured how) is not defined, nor what to do
  without it.
- "If both node level and material level LODs are specified then the material level LODs only apply to the specific
  node LOD that they were defined for."
- Not said: whether a lower level node may also be listed in `scene.nodes` or as somebody's child (model `x`: it then
  shows twice in a loader that ignores the extension - taken here as invalid input, checked as optional); what the
  transform of a lower level node means (taken here: it stands in the place of level 0, under the same parent, with
  its own TRS); whether a level may be a node without a mesh (the schema allows any node index, the validator only
  notes `NODE_EMPTY`; model `e`); children of level nodes; hysteresis; any way to load levels separately.

## Two tiers

**Tier A, "transparent".** No change of API or meaning of `GLTFLoader`: `load()` / `parse()` as today, everything
loaded up front. The one difference: a node with `MSFT_lod` comes back as a `THREE.LOD` that holds its levels, with
distances from `MSFT_screencoverage` (or a default), and three.js switches it by itself (`LOD.autoUpdate`, the
renderer calls `lod.update( camera )`). Nothing for the application to call, no option. What "no change" covers, and
what `check.js` tests for it:

- the scene graph is the stock loader's except that the LOD stands where the level 0 node stood, under the same
  parent; the level 0 node itself is unchanged inside it: same `name`, `userData` (extras), translation / rotation /
  scale, children, and the same world position;
- `parser.associations` still maps the level 0 object to `{ nodes: <its index> }`;
- animations and skins that target a node of an LOD still resolve and move that node (model `k`);
- a file without the extension gives the same result as the stock loader, object for object (model `n`);
- files with one buffer per level load as well (all buffers fetched, eagerly).

**Tier B, "luxury".** Opt-in (`{ lazy: true }` in the reference plugin); with the option off the loader is tier A
exactly. `parse()` answers as soon as the coarsest level of every LOD is there; a finer level is fetched, decoded and
built only when the camera first needs it, a bounded number at a time, nearest first, and the coarser level stays on
screen until the finer one is ready. Two delivery forms are covered: one `.bin` per level, and one `.glb` read by HTTP
range requests.

Every expectation in `check.js` is tier A or tier B. A stock loader fails both; a tier A loader passes all A
expectations on every model, the per-level-buffer ones included; a tier B loader passes A and B.

## Models

Synthetic models are plain glTF 2.0 with no other extension. Level 0 is a red sphere (fine), level 1 a green coarser
sphere, level 2 a blue box, a fourth level a yellow tetrahedron: which level is drawn is unmistakable. Node names are
unique per file (`A_High`, `A_Mid`, `A_Low`, ...), materials too.

| model | what | expected (reference distances: sphere of level 0, fov 50, 16:9) |
|---|---|---|
| `a_node_lod.gltf` + `.bin` | (a) node-level, 3 levels, coverage `[0.25, 0.05, 0.004]`, one buffer | 1 LOD; level 1 from 4.94, level 2 from 11.04, nothing beyond 39.04 |
| `a_node_lod.glb` | (f) the same in one `.glb` | the same; one BIN chunk: **eager only** (see "Lazy loading") |
| `b_no_coverage.gltf` | (b) as (a) without `MSFT_screencoverage` | 1 LOD, three levels in order of distance; distances are the loader's own (reference: 4.94, 9.88; the last level never dropped) |
| `c_transforms.gltf` | (c) `C_Left_High`: 2 levels, translation + rotation + scale, a child node `C_Left_Child`, under a scaled parent; `C_Right_High`: 4 levels, coverage `[0.3, 0.1, 0.03, 0.002]` | 2 LODs; left switches at 25.4, right at 4.51 / 7.81 / 14.25 and is dropped at 55.2; world positions as in the stock loader |
| `d_material_lod.gltf` | (d) material-level: one mesh, `D_Mat_High` / `_Mid` / `_Low` | the mesh is drawn with the material of the level |
| `e_empty_lowest.gltf` | (e) the last id is a node without a mesh, coverage `[0.25, 0.02, 0]` | 1 LOD of 3 levels; beyond 17.46 nothing is drawn |
| `g_lazy.gltf` + `g_lazy.lod0/1/2.bin` | (g) one buffer per level, the fine one heavy (368,688 / 1,488 / 648 bytes) | tier A: all loaded; tier B: far away only `lod2.bin` is fetched |
| `i_ranged.glb` | (i) one `.glb` (1,479,320 bytes) laid out for range requests: JSON, then the coarse levels, then the fine one | tier B: first request is a small range; fine level by a second range when near |
| `k_animated.gltf` | (k) an animation channel targets the level 0 node | the clip moves the node inside the LOD |
| `n_plain.gltf` | (n) no extension | identical to the stock loader's result |
| `x_levels_in_scene.gltf` | (x) outside the spec: lower levels also in `scene.nodes` | optional: each level drawn once, not twice |
| `town_cells.gltf` + 10 `.bin` | (h) 3 neighbouring cells of a baked town, 4 levels each (699,351 triangles in all), `KHR_meshopt_compression` + `KHR_mesh_quantization` (both required), the lowest level of all cells in one `.bin`, every other level in its own; 10,334,503 bytes | 3 LODs under their cell nodes; distances 54.7 / 108.8 / 248.0, 60.1 / 114.2 / 253.4, 73.4 / 127.5 / 266.7; needs `setMeshoptDecoder` |
| `town_cells.glb` | (h, ranged) the same in one `.glb` (10,334,064 bytes), coarse levels first | tier B: 2 requests (65,536 + 392,381 bytes) before the first frame |

On (e): the spec does not mention a level without geometry. Nothing forbids it (a node without a mesh is a valid
node), and the coverage array already has a way to say "nothing from here on": a last value above 0. Both are in
the kit: `e` has the empty node, `a` and `c` end their coverage above 0.

The town model is in the form the town exporter writes: quantized `POSITION` plus custom attributes (`_PL`, `_COL`,
`_G`), grey placeholder materials (the page it comes from binds its own). It is a load test for lazy levels with
meshopt, not a material test; `check.js` tints its levels red / green / blue / yellow.

`node generate.mjs` rewrites the synthetic models and `expected.json`; the two `town_cells` files are picked up if
present and described from their JSON (they cannot be regenerated here).

## Coverage to distance

`THREE.LOD` switches on the distance from the camera to the LOD's origin; the file gives screen coverage. The kit
uses this mapping. It is a **choice**, not something the spec gives:

```
coverage( d ) = area of the bounding sphere's disc / area of the screen
              = PI r^2 / ( 4 aspect tan^2( fov / 2 ) d^2 )
distance( c ) = r sqrt( PI / ( 4 aspect tan^2( fov / 2 ) c ) )
```

`r`: half the diagonal of the bounding box of level 0, from the accessors' `min` / `max` with the transforms of the
nodes above it; `fov` 50 degrees, `aspect` 16 / 9 (a reference camera: the distances are fixed at load). Level `i >= 1`
starts at `distance( coverage[ i - 1 ] )`; a last coverage above 0 adds a distance beyond which nothing is drawn.
Without `MSFT_screencoverage`: `coverage[ i ] = 0.25 ^ ( i + 1 )`, the last level kept for ever. The LOD's origin is
the middle of that box, so that the distance is to the object and not to wherever its node's origin happens to be.

Because this is a choice, `check.js` asks for little as required: at 1.05 radii level 0 is drawn, at 630 radii the
last level (or nothing), and on a walk outwards in 24 steps the levels come in order, one at a time, each at least
once. The exact switch distances (`reference_distances` in `expected.json`) are an optional check.

## Running the check

```
node generate.mjs                                   # models/ and expected.json
node server.mjs --mount /three=/path/to/three.js    # http://127.0.0.1:8791/ , answers Range
```

```
check.html?tier=a|b                 which expectations are required (default a)
          &three=URL                three.module.js       (default: three@0.186.1 from jsdelivr)
          &addons=URL               examples/jsm/         (GLTFLoader and the meshopt decoder are taken from here)
          &loader=URL               another GLTFLoader module (export GLTFLoader)
          &setup=URL                an adapter module (below)
          &model=TEXT               only the models whose file name contains TEXT
```

For a three.js checkout: `check.html?tier=a&three=/three/build/three.module.js&addons=/three/examples/jsm/`.
The result is on the page, in the title, and in `window.__result` (`tierA`, `tierB`, `optional`, `rows`, `observed`).

An adapter exports `configure( loader, { tier } )` (called for every `GLTFLoader` the check makes) and, for the ranged
`.glb` models of tier B, `loadRanged( loader, url, { probe } ) -> Promise<gltf>`. A loader with `MSFT_lod` built in
needs no adapter for tier A; for tier B the adapter is where its option is switched on.

Results (three.js r186, `WebGLRenderer`, Chromium 152; `renderer.render()` is what updates the LODs, the check never
calls `LOD.update` itself unless there is no WebGL):

| loader | tier A | tier B | optional |
|---|---|---|---|
| stock `GLTFLoader` (`?tier=b`) | FAIL, 50 of 109 | FAIL, 8 of 30 | 1 of 27 |
| reference plugin, `?tier=a&setup=setup_reference.js` | PASS, 109 of 109 | not asked | 27 of 27 |
| reference plugin, `?tier=b&setup=setup_reference.js` | PASS, 109 of 109 | PASS, 32 of 32 | 27 of 27 |
| `three-gltf-extensions` 0.0.15 `GLTFLodExtension`, all levels up front | FAIL, 91 of 109 | not asked | |
| the same, progressive mode | FAIL, 84 of 109 | FAIL | |

(The stock loader has 30 tier B rows, not 32: two rows only exist once a ranged load has been made.) The community
plugin runs on r186 without errors; it has no material-level LOD and no "nothing drawn" level, and its distances do
not come from the object's size, so on large objects level 0 is not reached at 1.05 radii.

The tier B expectations: after the load with the camera far away only the buffers of the coarsest level have been
requested; far frames fetch nothing more; moving close requests the fine buffer; the coarse level is still drawn
while the fine one is under way (the check delays that request by 600 ms); once there, level 0 is drawn; moving away
and back fetches nothing; no buffer is fetched twice; for a ranged `.glb`: the first request is a small range and not
the file, the bytes before the first frame stay under a bound, the total stays under the file's size, and a server
that ignores `Range` and answers 200 with the whole body is asked once, not once per buffer view.

## What the unmodified GLTFLoader does with each file (observed, r186)

- No `LOD` object anywhere. Only level 0, the finest, is built, and it is drawn at every distance.
- The lower level nodes are never instantiated: nothing reaches them from the scene. Their meshes, accessors and
  materials are not loaded either (the loader resolves dependencies from the scenes down).
- One buffer: the whole `.bin` / `.glb` is fetched, of which only level 0 is used.
- One buffer per level: only level 0's buffers are fetched - the heaviest ones. `g_lazy`: `g_lazy.gltf` +
  `g_lazy.lod0.bin`; `town_cells.gltf`: the three `fine` files (6.4 of 10.3 MB), 20 meshes.
- `x_levels_in_scene.gltf`: all three levels are in the scene and drawn on top of each other.
- `n_plain.gltf`: 2 meshes, as written.
- Material-level: the mesh has the level 0 material.

A loader that ignores the extension and builds from the scenes down therefore shows the most detailed level of
everything - correct, and the worst case for weight.

Not every tool builds from the scenes down. Blender 4.1.1 (its glTF importer, observed): `a_node_lod.glb` comes in as
three mesh objects, `A_High`, `A_Mid` and `A_Low`, all in the scene collection and on top of each other - the importer
creates an object for every node of the file, in a scene or not. `d_material_lod.gltf` comes in as one object with
`D_Mat_High` only. So for node-level LODs "the lower levels are ignored" holds for three.js and does not hold for
Blender.

## Validator

glTF Validator 2.0.0-dev.3.10, every model: **0 errors, 0 warnings**. Infos only:

| model | infos |
|---|---|
| `a_node_lod.gltf`, `a_node_lod.glb`, `b_no_coverage.gltf`, `g_lazy.gltf`, `i_ranged.glb` | 1 `UNSUPPORTED_EXTENSION` (MSFT_lod), 2 `UNUSED_OBJECT` (the lower level nodes) |
| `c_transforms.gltf` | 1 `UNSUPPORTED_EXTENSION`, 4 `UNUSED_OBJECT` |
| `d_material_lod.gltf` | 1 `UNSUPPORTED_EXTENSION`, 2 `UNUSED_OBJECT` (the lower level materials) |
| `e_empty_lowest.gltf` | 1 `UNSUPPORTED_EXTENSION`, 2 `UNUSED_OBJECT`, 1 `NODE_EMPTY` (`/nodes/2`) |
| `k_animated.gltf` | 1 `UNSUPPORTED_EXTENSION`, 1 `UNUSED_OBJECT` |
| `n_plain.gltf` | none |
| `x_levels_in_scene.gltf` | 1 `UNSUPPORTED_EXTENSION` |
| `town_cells.gltf` | 2 `UNSUPPORTED_EXTENSION` (MSFT_lod, KHR_meshopt_compression), 60 `UNUSED_OBJECT` (lower levels, custom attributes without a material that reads them) |
| `town_cells.glb` | 2 `UNSUPPORTED_EXTENSION`, 51 `UNUSED_OBJECT` |

`UNUSED_OBJECT` on every lower level is what the extension looks like to a tool that does not know it.

## Lazy loading: what the file has to look like

A loader can only leave out what it can fetch separately.

1. **One `.bin` per level** (`g_lazy.gltf`, `town_cells.gltf`): each level's buffer views lie in a buffer with a URI
   of its own. Works on any static host, each file is cached and compressed (gzip, brotli) on its own, and the stock
   loader already fetches only what it uses. In the town model the lowest level of all cells shares one `.bin` (it is
   always needed: one request) and every other level of a cell has its own.
2. **One `.glb` read by HTTP range requests** (`i_ranged.glb`, `town_cells.glb`): the JSON chunk first, then in the BIN
   chunk the coarse levels of everything, then the finer ones. The loader asks for the first 64 KiB (header + JSON; a
   longer JSON is fetched with a second request), then for the ranges of the buffer views it needs; ranges asked for
   in the same task that touch or nearly touch (16 KiB) are merged into one request.
3. **One `.glb` without range requests cannot be lazy**: there is one BIN chunk and one request for it.
   `a_node_lod.glb` is that case; a tier B loader loads it whole, as tier A does.

Observed with the reference plugin and `server.mjs --log`:

- `town_cells.gltf`, camera far: the `.gltf` and `town_cells.lo.bin` (392,381 bytes of 10.3 MB). Walking in: the
  `mid`, `near` and `fine` files of the cells that need them, each once.
- `town_cells.glb`: `bytes=0-65535`, then `bytes=44312-436692` - two requests, 458 KB, before the first frame; near the
  cells the fine levels of two cells in one request (`3937281-7422133`), the third in another.
- A server that ignores `Range` (`/norange/...` on `server.mjs`): one request, 200, the whole file, used for
  everything.

Range requests and compression do not go together: a range of a gzip stream is of no use, so a ranged answer is sent
as it is on disk. Chrome sends `Accept-Encoding: identity` by itself on a `fetch()` that carries a `Range` header
(seen in the server log; without `Range` it sends `gzip, deflate, br, zstd`). Meshopt-compressed geometry still gains
some 20 % from gzip (measured on the town: 44.2 MB instead of 55.5 MB for the same walk), which the ranged form gives
up; with per-level files the host can compress each.

Recommendation: per-level `.bin` files as the form to support first. They need nothing from the server and nothing new
in the loader's core, they compress, and they are what glTF-Transform's `partition` already produces. The ranged
single `.glb` is worth having as an opt-in for the case where one file is the requirement; it must fail loudly or
fall back once (not per view) on a server that answers 200.

## The reference plugin

`GLTFMSFTLodExtension.js`, one file, imports `three` only.

```js
import { GLTFMSFTLodExtension, loadRangedGLB } from './GLTFMSFTLodExtension.js';
loader.register( ( parser ) => new GLTFMSFTLodExtension( parser ) );                  // tier A
loader.register( ( parser ) => new GLTFMSFTLodExtension( parser, { lazy: true } ) );  // tier B
const gltf = await loadRangedGLB( loader, 'model.glb' );                               // tier B, one .glb by ranges
```

Options: `lazy` (false), `fov` (50), `aspect` (16 / 9), `hysteresis` (0), `distances` (a function `( info ) =>
{ distance, hysteresis?, center?, cull? }` for an application that has distances of its own), `maxInFlight` (2),
`releaseAfter` (seconds, 0 = never: drop a level that has not been drawn for that long), `keepBuffers` (false),
`onLevel( lod, level, object )`.

How it is built:

- `loadNode`: for a node with the extension the plugin returns a nameless `THREE.LOD` placed at the middle of level
  0's bounds, in the parent's space. Each level is a `Group` at minus that middle holding the glTF node as the stock
  parser builds it, untouched. A last coverage above 0 adds an empty level beyond which nothing shows.
- `beforeRoot`: lower level indices that also appear in `scene.nodes` or as children are taken out (model `x`).
- `afterRoot`: animation tracks that were named after the LOD (the parser names tracks after the object `loadNode`
  returned) are pointed at the level 0 node again; `parser.associations` gets the level 0 node back.
- Material-level: every mesh drawn with such a material becomes an LOD of copies of the mesh, same geometry, one
  material per level.
- Tier B hooks into `THREE.LOD` by **subclassing it and overriding `update( camera )`** (`GLTFLazyLOD`). The renderer
  already calls `update` for every LOD it meets, once per render, with the camera: the override lets `LOD` pick its
  level, notes which level is wanted and how far away, queues the load, and shows the nearest level that is ready
  (the next coarser one, else the next finer) until then. Nothing for the application to call. Alternatives looked
  at: placeholder objects that load on `onBeforeRender` (what the community plugin does: only fires for objects that
  are drawn, so it needs a dummy mesh, and has no distance to order by); an `update()` the application calls every
  frame (an API change); loading everything through a prioritised queue after the first frame (not lazy, only late).
- One queue for all files, `maxInFlight` at a time, nearest first; a wish lapses after a second without updates.
- LOD chains that an animation or a skin points into are loaded whole, as in tier A.
- The ranged `.glb`: `loadRangedGLB` reads header and JSON, hands the JSON to `loader.parse` with buffer 0 marked, and
  the plugin's `loadBufferView` fetches the ranges. Views compressed with `EXT_` / `KHR_meshopt_compression` are
  decoded by the plugin itself with the loader's meshopt decoder (asynchronously, in the decoder's workers when it
  has them), because the stock meshopt plugin's `loadBufferView` would ask for the whole buffer first.

Known limits: a material-level LOD copy does not carry the mesh's children, and meshes with several materials are
left alone; `clone()` of a `GLTFLazyLOD` is a snapshot (what is loaded stays, nothing more is fetched); the distances
are fixed at load for a reference camera (a different fov or a resized window does not move them); a `releaseAfter`
level that is needed again is fetched again.

## Open questions for three.js

**Shape of the result (tier A).**
- Where the LOD's origin is. `LOD.update` measures to the object's origin; a glTF node's origin can be anywhere
  relative to its geometry. The kit moves the LOD to the middle of the bounds and puts the node back with a holder;
  simpler would be to make the LOD the node itself (the mesh of level 0 as a child), at the price of distances to the
  node's origin.
- Whether the object returned for the node may be something else than the node (name, `userData`, associations and
  animation targets all hang on it). The kit keeps the node intact inside a nameless LOD.
- Children of a level 0 node: here they stay its children, so they show with level 0 only.
- Default coverage when the file has none; the reference camera for coverage to distance; whether `hysteresis`
  (r150+) should have a default above 0 - the spec has none, and without it an object at a switch distance flickers.
- "Nothing drawn" below the last coverage: honour it (the kit does) or keep the last level for ever.
- Lower levels listed in a scene as well: clean up (the kit does) or leave as written.

**What makes tier B awkward in GLTFLoader today.**
- `parser.getDependency` caches for ever and loads whole things: an accessor pulls its whole buffer view, a buffer
  view its whole buffer. There is no way to say "this node, later", so the plugin keeps the lower levels out of the
  first pass by returning its own object from `loadNode` and calling `parser.getDependency( 'node', id )` later.
- `loadBuffer` has no plugin hook and one request per URI; a buffer stays in `parser.cache` after its views are cut
  from it (the plugin drops those entries once a level is built, unless `keepBuffers`).
- `loadBufferView` is a first-one-wins hook (`_invokeOne`): a range plugin and the meshopt plugin both want it and
  cannot be combined; the reference plugin does the meshopt decode itself to get around that.
- `associations` are reduced to what is in the returned scenes at the end of `parse`; objects that arrive later have
  to be added by the plugin.
- Every node of a scene is instantiated up front; an `afterRoot` hook is the only place to patch animations.
- `FileLoader` treats 206 as an error and keys its request de-duplication on the URL alone.

**Upstream state of range requests.**
- PR #24580 "FileLoader: Allow HTTP Range requests" (takahirox, open since 2022, approved by Mugen87): accept 206,
  put the `Range` header into the cache / de-duplication key, a 200 answer to a ranged request goes to `onError`
  with the response attached.
- Issue #24506 "GLTFLoader: HTTP Range Requests support" (open): a plugin over the `loadBufferView` hook
  (`GLBRangeRequestsPlugin`, also in `three-gltf-extensions`), with the stated limit that it cannot be combined with
  another plugin on the same hook such as meshopt. donmccurdy prefers partitioning into several `.bin` files over
  range requests, wants range requests opt-in and failing (not silently fetching the whole file) on a 200, and is
  wary of more complexity in `GLTFLoader`; elalish notes that fetching the JSON first lets shader compilation start
  early.
- So an upstream tier B over per-level `.bin` files needs no core change beyond a way to load a node later; over one
  `.glb` it needs PR #24580 (or `fetch` inside the plugin, as here) and a `loadBufferView` that composes: for example
  a separate `loadBufferRange( bufferIndex, byteOffset, byteLength )` hook below it, which the meshopt plugin would
  call instead of `getDependency( 'buffer' )`.
