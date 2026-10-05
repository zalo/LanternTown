# GLTFLoader: Add setLazyLOD() and setRangeRequests().

Related issue: #24506

Related PRs: #24580 (range requests in `FileLoader`), #25077, and the `MSFT_lod` PR (#XXXX), which this one is stacked on: the first commit belongs to that PR, the second one is this PR.

**Description**

With `MSFT_lod` supported, a file can carry several versions of every object, and the loader downloads and decodes all of them before `onLoad()` although the first frame needs one of them. This PR adds two opt-in switches. Both are off by default, and with both off the loader behaves as in the previous PR.

```js
const loader = new GLTFLoader()
	.setMeshoptDecoder( MeshoptDecoder )
	.setDRACOLoader( dracoLoader )
	.setLazyLOD( true )         // load the levels of MSFT_lod nodes when they are needed
	.setRangeRequests( true );  // fetch a .glb in parts (HTTP range requests)

const gltf = await loader.loadAsync( 'town.glb' ); // the coarsest level of every LOD is in the scene
```

This is a large change to a file that is already long (about 1,040 lines added to the loader, 420 to the tests), and #24506 and #24580 voiced exactly that concern. It can be split, see open question 1. I would rather hear "not in the loader" early than polish it further.

**`setLazyLOD( true )`**

- Only the coarsest level (with a mesh) of each `MSFT_lod` node is loaded before `onLoad()`. The result is complete and renderable, just coarse.
- The `LOD` objects are a private subclass, `GLTFLazyLOD`, that overrides `update()`: it lets `LOD` pick the level, and if that level is not loaded yet it shows the nearest level that is (a coarser one if there is one) and asks for the missing one. So levels are requested by the renderer's normal `LOD.autoUpdate`, and nothing has to be called by the application.
- Requests are collected during the frame and started in a microtask after it, the nearest `LOD` first, at most three at a time. A wish that lapses before its turn (the camera moved on, the `LOD` was removed from its parent) is dropped, not loaded.
- The level that is loaded is the one the camera asks for. Levels between it and the one on screen are skipped, so that nothing is downloaded that is not shown in the end (see open question 7 for the alternative).
- A level goes through the normal `getDependency( 'node', … )` path, so plugins and the compression extensions apply to a lazy level as to any other node, and it gets its `parser.associations` entry and `userData` (tested with meshopt, quantization and Draco; KTX2 inside lazy levels follows from the shared path but is not tested). The loaders that were given to the `GLTFLoader` (`DRACOLoader`, `KTX2Loader`) are used for as long as levels are loaded and must not be disposed before; a level that needs a disposed decoder never arrives.
- LODs with a level that is skinned, is a bone, or is targeted by an animation are loaded eagerly, because animations and skins need their nodes at load time. Material LODs are loaded eagerly as well (they share one geometry, there is little to gain).
- This works without range requests. With a `.gltf` whose levels live in different `.bin` files (the layout preferred in the discussion of #24580) only the `.bin` files of the levels that are actually shown are requested. It also works with `parse()`: nothing is fetched then, but decoding and upload are deferred.

The `MSFT_lod` plugin of [three-gltf-extensions](https://github.com/takahirox/three-gltf-extensions/tree/main/loaders/MSFT_lod) has a progressive loading mode as well. The difference here is that the levels are driven by the camera (a level nobody looks at is never fetched) and that it composes with range requests and the compression extensions.

**`setRangeRequests( true )`**

- `load()` requests the first 64 KiB of the file. If the answer is `206` and a GLB, the JSON chunk is parsed from it and the BIN chunk is then read with range requests per buffer view. Ranges that are asked for in the same tick and lie closer than 16 KiB are merged into one request, whether they are served from memory, from a response that is still being read or by a new request.
- If the JSON chunk is longer than 64 KiB, the rest is read from one open-ended request (`bytes=65536-`) as a stream. The JSON is parsed as soon as it is in, the stream is followed for as long as the ranges that are waiting lie in what arrives next, and cancelled after that. A file with a long JSON chunk and its coarsest levels at the start of the BIN chunk shows its first frame after two requests.
- If the server answers `200` with the whole file, that body is used: it is read as it arrives and every range is resolved as soon as its bytes are in. One request, a warning in the console, no error. If a server answers `206` at first and `200` to a later range request, the same warning is logged, that one body is read to its end, the bodies of the other requests are cancelled, and every later range is served from it, so the file is downloaded once.
- A `.gltf` or any other non-GLB file is completed with one more request and parsed the normal way.
- Other answers are errors in the style of `FileLoader` (a response without a status, i.e. not HTTP, is accepted as there); a short answer to a range request rejects the buffer views that waited for it.
- Request headers, `withCredentials`, `manager.resolveURL()`, `manager.itemStart/itemEnd/itemError` and `manager.abortController` are honoured. `onProgress` is not called in this mode.

**How ranges and meshopt compose (#24506)**

The known problem is that range loading via the `loadBufferView` hook and `EXT_meshopt_compression` / `KHR_meshopt_compression` both want that hook, and the meshopt plugin calls `getDependency( 'buffer' )`, which pulls the whole file. This PR puts the ranges *below* the hook instead of into it:

- `GLTFParser.loadBufferRange( bufferIndex, byteOffset, byteLength )` returns a `Uint8Array` with those bytes. For the BIN chunk of a range-loaded GLB it is a range request, for every other buffer it is a view on the buffer loaded as today.
- `GLTFParser.loadBufferView()` uses it for the range-loaded buffer; its existing code path is untouched for all other buffers.
- `GLTFMeshoptCompression.loadBufferView()` reads its compressed source through it instead of `getDependency( 'buffer' )`.

So a meshopt-compressed GLB is fetched and decoded one buffer view at a time, and the fallback buffer is never loaded. Third-party `loadBufferView` plugins keep working unchanged, and can call `loadBufferRange()` to get the same benefit.

**Relation to #24580, and where this differs from what was agreed there**

I did not build on #24580: it has been open since 2022, and this code needs the status (`206`/`200`/`416`), the real length and the body stream of the response, which `FileLoader` does not expose. The cost is about 40 lines (`GLTFBinaryRangeExtension._fetch()`) that repeat what `FileLoader` does for headers, credentials and the manager, and that bypass `Cache` and the merging of concurrent requests in `FileLoader`. If #24580 lands in a form that exposes the response, that method can be replaced by it without touching the rest.

The objections raised there apply here, so for the record:

- *A fallback is risky, and the file may be downloaded several times.* #24580 ended with a `200` answer being an error. Here the `200` body is used instead. The option is opt-in, and I found no path left in which the file is downloaded more than once (the late-`200` case above has a unit test). If you prefer an error, that is a smaller patch than this one (open question 3).
- *Ranges should be opt-in and should fail or warn without server support.* Opt-in: yes. A `200` answer logs a warning ("The server doesn't support range requests, the whole file is loaded.") and the load goes on with that body.
- *Partitioned `.bin` files are the better layout.* `setLazyLOD()` supports exactly that without any range request. `setRangeRequests()` is for the single `.glb`, which is what most pipelines produce.
- *Complexity in `GLTFLoader`.* See open question 1.

**Semantics that are kept**

`onLoad()` / `loadAsync()` / `parse()` / `parseAsync()` are called once, with the same kind of result as before. Errors before that go to `onError`. After `onLoad()` there is no callback left to report to: a level that fails to load is logged with `console.error`, marked as failed (it is not retried), and the `LOD` keeps showing the next best level. Loading a level starts and ends items on the `LoadingManager`, so `manager.onStart`/`onLoad` can fire again later, as with any later load on the same manager.

**Limitations / things I decided not to do**

- Levels are never unloaded. Removing a `LOD` from its parent drops its pending request, and disposing geometries frees GPU memory as usual, but there is no eviction and a disposed level cannot be fetched again (the parser cache holds it). I think this needs a budget/priority policy that belongs to the application.
- The `LOD` objects keep the parser alive (its JSON, its caches, and the bytes that were read; after a `200` answer that is the whole file) for as long as they are referenced.
- No event when a level arrives. Applications that render on demand have to poll (`lod.levels[ i ].object.children.length`), and code that walks the scene once after loading (to set `castShadow`, to change materials, `afterRoot` plugins) does not see the levels that arrive later. A `'levelloaded'`-style event on the `LOD` would be easy to add if wanted.
- `Raycaster` hits what is currently shown, not the level that is still loading.
- Range mode applies to the BIN chunk of a `.glb` only, not to external `.bin` files, and the chunk after the JSON chunk has to be the BIN chunk.
- No `If-Range`: if the file is replaced on the server between two requests, the parts are mixed.
- A `fetch()` with a `Range` header asks for `Accept-Encoding: identity`. So range mode gives up transport compression, also when the answer is `200`. On a server without range support this option can make the download bigger.
- The open-ended request for a long JSON chunk is cancelled when the loader has what it needs. On HTTP/1.1 that closes the connection, and what the server had sent beyond that point is wasted. A `fetch` wrapper or service worker that buffers whole responses turns that request into a download of the rest of the file. Files whose JSON chunk fits into 64 KiB never make that request.
- How many requests it takes depends on the layout of the file. If the coarsest levels are not contiguous in the BIN chunk there is one request per run. An exporter that writes the coarse levels first gets this down to two or three.
- Cross-origin: a simple `Range: bytes=a-b` header is CORS-safelisted in current browsers, older ones send a preflight. Not tested cross-origin.
- Errors are plain `Error` objects, not the `HttpError` of `FileLoader` (which is not exported).

**Tested**

Everything in this list was run on the commit of this PR.

- Unit tests in `test/unit/addons/loaders/GLTFLoader.tests.js` (7 new): lazy levels with `parseAsync()` (coarsest level first, the level arrives after `lod.update()`, the association is registered); range requests against a stubbed `fetch` (`bytes=0-65535`, then exactly the rest, buffer views joined correctly across the two answers); a JSON chunk longer than the head (`bytes=0-65535`, `bytes=65536-`, the stream cancelled shortly after the JSON, the level fetched from where the stream stopped); a server that always answers `200` (one request, read to the end, file loaded); a server that answers `206` and then `200` (two concurrent ranges: one body read, the other cancelled, no request after that); a `.gltf` of exactly 65,536 bytes (the open-ended request is answered with `416`, the file is loaded); Draco data in a lazy level with a stub decoder (the compressed buffer view comes from range requests and is decoded once, when the level is needed). I ran the file under Node with QUnit 2.25.0 and `three` mapped to `src/`, not through the Puppeteer runner: 13/13 pass, 61 assertions.
- The conformance set of the previous PR in Chromium (`WebGLRenderer`, this loader on the r186 build), with both options off: 109/109. With the options on, 32 further checks: only the buffers of the coarsest levels are requested before the first frame; nothing is requested while the camera is far away; the coarse level stays until the fine one has arrived; moving away and back requests nothing; per-level `.bin` files are requested when the level is needed and never twice; a ranged `.glb` starts with one small `206` request, stays below a byte limit before the first frame and below the file size in total, with every answer a `206`; a meshopt-compressed scene as `.gltf` + `.bin` files and as one `.glb`; a server that ignores `Range` (one request, `200`, scene complete): 32/32.
- Regression: the 21 files without the extension from the previous PR, loaded with `dev` and with this branch, with both options off and with both options on (1 to 3 range requests per file): 21/21 structurally identical in both runs. As in the previous PR this run was under Node, with the meshopt decoder as is, image decoding stubbed and a stand-in for `DRACOLoader`.
- A real scene with stock materials: 13 chunks of a town with 2 to 4 levels each (42 levels), `KHR_draco_mesh_compression` + `EXT_texture_webp` + `KHR_materials_unlit`, one 29.1 MiB `.glb` with a JSON chunk of 488 KiB and the coarsest level of every chunk at the start of the BIN chunk. Chromium, a local server without throttling, one run each, a view that needs the finest level of 6 chunks, so the times say little about a real network and the byte and request counts are the result:

| | `onLoad()` | first frame | fetched by then | view complete | fetched by then | levels loaded |
|---|---|---|---|---|---|---|
| eager (previous PR) | 1.44 s | 2.27 s | 29.1 MiB, 1 request | 2.27 s | 29.1 MiB, 1 request | 42 of 42 |
| lazy + ranges | 0.14 s | 0.24 s | 4.1 MiB, 2 requests | 0.81 s | 17.6 MiB, 16 requests | 26 of 42 |

  All 16 requests were answered with `206`, at most 4 were open at the same time, one (the open-ended one for the JSON chunk) was cancelled early. Both runs end with the same levels on screen and without errors or warnings.
- `eslint` with the repository config on both files: clean.

Not tested on this commit: the unit tests through `npm run test-unit-addons`; a throttled or real network (see below); `LoadingManager.abort()` during a range or level load (wired up, not exercised); a response without a status; `WebGPURenderer`; browsers other than Chromium, phones; cross-origin servers; what cancelling the open-ended request costs on a real network.

<details>
<summary>Earlier measurements on a throttled connection (an earlier revision of this branch, not repeated on this commit)</summary>

These were taken before the rebase onto the current `dev` and before the changes to the late-`200` case, with an earlier build of the same scene (27.5 MiB, JSON chunk of 479 kB). Chromium with DevTools throttling, cold cache, the view described above. One run each (two where a range is given); "nearest right" is when the chunk next to the camera shows the level it needs.

| link | | first frame | fetched by then | nearest right | view complete | fetched by then |
|---|---|---|---|---|---|---|
| 20 Mbit/s, 40 ms | eager | 12.6 s | 27.5 MiB, 1 request | 12.6 s | 12.6 s | 27.5 MiB, 1 request |
| | lazy + ranges | 1.4 - 1.5 s | 2.96 MiB, 2 requests | 4.7 s | 7.4 s | 16.6 MiB, 16 requests |
| | lazy + ranges, server ignores `Range` | 1.4 s | 2.97 MiB of that request | 10.7 s | 11.7 s | 27.5 MiB, 1 request |
| 36 Mbit/s, 100 ms | eager | 7.2 s | 27.5 MiB, 1 request | 7.2 s | 7.2 s | 27.5 MiB, 1 request |
| | lazy + ranges | 1.0 - 1.1 s | 2.96 MiB, 2 requests | 2.9 s | 4.5 s | 16.6 MiB, 16 requests |
| 10 Mbit/s, 150 ms | lazy + ranges | 3.0 s | 2.96 MiB, 2 requests | 9.3 s | 14.7 s | 16.6 MiB, 16 requests |

How many levels to load at once (36 Mbit/s, 100 ms; the scene above with 13 LODs and levels of 0.2 to 3 MiB, and the whole town as one 85.2 MiB `.glb` with 98 LODs, 272 levels, many of them small):

| at once | 13 chunks: nearest right | 13 chunks: view complete | whole town: nearest right | whole town: view complete |
|---|---|---|---|---|
| 2 | 2.1 s | 4.8 s | 4.0 s | 16.0 s |
| 3 | 2.8 - 2.9 s | 4.5 s | 4.2 s | 13.4 - 14.2 s |
| 4 | 4.0 s | 4.6 s | 4.8 s | 12.9 s |
| 6 | 4.5 s | 4.5 s | 5.2 s | 12.2 s |
| 8 | | | 5.6 s | 12.2 s |

Large levels share the connection, so that more of them at once only delay the nearest one. Small levels are bound by the round trip of their request, so that fewer at once take longer. Three is the constant that loses the least on both.

Which level to load (open question 7), the 13 chunks at 36 Mbit/s and 100 ms; levels are not additive, so every step is paid in full:

| | no chunk more than 2 levels too coarse | none more than 1 | view complete | fetched by then |
|---|---|---|---|---|
| the wanted level only (this PR) | 4.5 s | 4.5 s | 4.5 s | 16.6 MiB, 16 requests |
| one step off the coarsest level first, then the wanted level | 1.9 s | 5.2 s | 5.2 s | 19.1 MiB, 24 requests |
| every level on the way | 1.8 s | 3.5 s | 6.7 s | 25.5 MiB, 30 requests |

</details>

**Open questions**

1. *Size.* The three parts can be split: (a) `loadBufferRange()` + the meshopt change (small, useful to third-party range plugins on its own), (b) `setRangeRequests()`, (c) `setLazyLOD()`. (b) and (c) could also live in `examples/jsm/loaders/` next to the loader as plugins if (a) is in the core. I kept them in one place because that is what I could test as a whole.
2. *API shape.* Two boolean setters, because the two things are independent (lazy levels work without ranges; ranges have requirements on the server that lazy levels do not) and because every other loader option is a chainable setter. Alternatives: one `setLazyLOD( { ranges: true } )`, or deriving range mode from the lazy option. No options are exposed for the head size (64 KiB), the merge gap (16 KiB) or the number of levels that are loaded at once (3); they are constants at the top of their sections. The right number of levels at once depends on their size (earlier measurements above), which a setter would hand to the application without telling it what to choose. A limit on the bytes in flight would load large levels one after the other and small ones together; it needs the byte length of a level from the JSON and is left out.
3. *Fallback.* Is using the `200` body right, or should a server without range support be an error, as #24580 settled on? And should the loader retry without the `Range` header to get a compressed answer when the file is large?
4. *Failed levels* are not retried, also not after `LoadingManager.abort()`, and `abort()` does not keep levels from being requested later. A retry on a later `update()` is a small change but can hammer a broken server.
5. *Unloading*, and an event for on-demand renderers and for code that post-processes the scene (see limitations).
6. `GLTFLazyLOD` is private and reachable only through `isLOD`. Should it be exported, or should the level states be public on it?
7. *Which level to load.* The loader fetches the level the camera asks for and skips the ones in between. The alternative is to refine in steps, the `LOD` that is the most levels behind first, which is what a viewer expects from a progressive format. In the earlier measurements above, stepping shows something better sooner and costs about 15 % (one step first) or 54 % (every level) more bytes for the same final view. Both alternatives are a change of a few statements in `GLTFLazyLOD.update()` and `_loadLevels()`. I kept the variant that never downloads what is not shown in the end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
