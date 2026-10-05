# GLTFLoader: Add MSFT_lod support.

Related issue: none filed. Related PRs: #25077 (the `loadNode` hook, added so that this extension can be handled by a plugin), #21458.

**Description**

`GLTFLoader` ignores [`MSFT_lod`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/MSFT_lod): a file that uses it renders its level 0 at every distance and the other levels are never instantiated. The extension can be handled by a plugin since #25077, and [three-gltf-extensions](https://github.com/takahirox/three-gltf-extensions/tree/main/loaders/MSFT_lod) has one. I am proposing it for the loader itself because the extension maps directly onto `THREE.LOD` and the plain case needs no API. I am aware that it would be the only vendor extension in the list of built-in ones and that `MSFT_texture_dds` is listed under the plugins; if the answer is that this belongs in a plugin as well, that is a fine answer.

This PR adds a built-in plugin, `GLTFLodExtension`, in the style of the other built-in extensions, and lists the extension in the class documentation. It does not add or change any public API.

- A node with `extensions.MSFT_lod.ids` is returned inside a `LOD`. Level 0 is the node itself, the other levels are the listed nodes. `LOD.autoUpdate` does the switching, nothing else is involved at runtime.
- The distances come from `extras.MSFT_screencoverage`. The last value is the coverage below which the lowest level is no longer drawn; if it is greater than zero an empty last level is added, so that the object disappears as the specification describes. Without `MSFT_screencoverage` each level takes over at a quarter of the previous coverage (the distance doubles).
- Material-level `MSFT_lod` is supported for the simple case: the levels are clones of the node with the listed materials.
- Everything is loaded with the file, as today.
- A file without the extension is loaded as before (see "Tested").

The resulting structure is

```
LOD                 at the center of the bounds of level 0, in the place of the node
├─ Group            level 0, offset back by the center
│  └─ node          the object the loader creates today: same name, transform, userData, association
├─ Group            level 1
│  └─ node ids[ 0 ]
└─ ...
```

**Design notes**

- *Why holders.* `LOD` measures the distance from its own origin. With `KHR_mesh_quantization` (and with exporters that bake an offset into the node) the origin of the node is a corner of the mesh, which makes the switching lopsided. So the `LOD` is placed at the center of the bounding box of level 0 (from the accessor `min`/`max`, no geometry is loaded for it) and each level sits in a `Group` that undoes the offset. The node objects themselves are untouched, so names, transforms, `userData`, `parser.associations` and animation targets are what they are without the extension.
- *Coverage to distance.* `MSFT_screencoverage` is a fraction of the screen, `LOD` wants a distance, and the loader has no camera. The conversion assumes a reference camera with a vertical field of view of 50° (the `PerspectiveCamera` default) and an aspect of 16:9: `distance = radius * 1.42539 / sqrt( coverage )`, where `radius` is the radius of the bounding sphere of level 0 times the largest scale of the ancestors of the node. That is one constant (`LOD_COVERAGE_TO_DISTANCE`), derived in the comment next to it. An application with a different camera can rescale `lod.levels[ i ].distance` after loading.
- *Animation.* A channel that targets a node with `MSFT_lod` has to bind to the node and not to the `LOD` that stands in its place. `loadAnimation` is hooked for that: while the parser resolves the targets of such an animation, the cache entry of the node is the node object, and it is restored right after. This is the least pretty part of the patch (open question 3).
- *Skinning.* Level nodes with a skin go through the normal node path. I have no `MSFT_lod` asset with skinned levels, so this is reasoned, not tested.
- *Existing plugins.* Plugins are stored by name, so a plugin registered under the name `MSFT_lod` (the one of three-gltf-extensions is) replaces the built-in one. That follows from how `GLTFLoader.parse()` collects plugins; I have not run that plugin against this branch.

**Behavior changes for files that use the extension**

- Where there was a mesh there is now a `LOD` (without a name) with that mesh two levels down. Code that finds objects by name or through `parser.associations` keeps working; code that walks `scene.children` by index on such a file does not.
- `userData.gltfExtensions.MSFT_lod` is no longer set on these nodes and materials, because the extension is not an unknown one anymore, and an `extensionsRequired` entry no longer logs "Unknown extension".
- The children of a level-0 node are part of level 0 and are only shown with it.
- `ids` that do not refer to existing nodes or materials make the load fail, where the extension was ignored before.

If that is too much for a default, the plugin could be opt-in through a setter, at the cost of one method.

**Not handled, on purpose**

- Material LODs on a node that has children, is animated or uses `EXT_mesh_gpu_instancing`: the levels are clones and would not follow, so the node is loaded as if its materials had no extension.
- Level nodes that are also listed in a scene or are a child of another node (the specification does not intend that): they are loaded there as today in addition to being levels.
- If no level has `POSITION` bounds (`min`/`max`), there is nothing to place the `LOD` with and the extension is ignored for that node.
- The bounds and distances are fixed when the file is loaded. They do not follow animated ancestors, morph targets, skins, a `LOD` nested in a level of another `LOD`, or a later scale of the scene.

**Tested**

- Unit tests in `test/unit/addons/loaders/GLTFLoader.tests.js` (4 new): node LODs (levels, distances, position, associations, transform of the node); material LODs (one geometry, the listed materials, association of the clone source); an animation that targets a LOD node (the track is bound to the node, the node stays in the `LOD`); the same file without the extension. I ran the file under Node with QUnit 2.25.0 and `three` mapped to `src/`, not through the Puppeteer runner: 6/6 pass.
- A conformance set of small models written for this work (node LOD as `.gltf` and `.glb`, no `MSFT_screencoverage`, transformed and scaled parents, material LOD, an empty lowest level, an animated LOD node, a plain file), each loaded and rendered with `WebGLRenderer` at a series of distances in Chromium, with this loader on the r186 build: 109/109 checks. The one model outside the specification (levels that are also listed in the scene) is the only one with failing optional checks (5 of its checks, 22/27 optional overall).
- Regression: 21 files without the extension (Draco, meshopt + quantization, skins, morph targets, animations, `EXT_mesh_gpu_instancing`; among them `RobotExpressive`, `Soldier`, `Horse`, `Flower`, `DamagedHelmet`, the GPU-instancing `DamagedHelmet`, `AnimatedMorphSphere`, `duck`, `gears`, `collision-world`) loaded with `dev` and with this branch, comparing the complete result structurally: object types, names, hierarchy, matrices, `userData`, associations, geometry attributes and index (type, count, hash of the data), materials and textures, skeletons, instance matrices, animation tracks. 21/21 identical (1487 objects, 1108 meshes, 20 animations). This run was under Node, with the meshopt decoder as is, image decoding stubbed and a stand-in for `DRACOLoader` that records what it is given. The same comparison in Chromium with the real decoders was 21/21 on an earlier revision (before the rebase onto the current `dev` and before three small fixes to the LOD code); it was not repeated on this commit.
- A real scene: 13 chunks of a town with 2 to 4 levels each (42 levels) in one 29 MiB `.glb` with Draco and WebP: loads and renders in Chromium (2.6 M triangles in the test view).
- `eslint` with the repository config on both files: clean.

Not tested: `WebGPURenderer` (nothing here is renderer-specific, `LOD.update()` is called by both renderers), browsers other than Chromium, the unit tests through `npm run test-unit-addons`.

**Open questions**

1. Is a fixed reference camera acceptable for the coverage → distance conversion, or would you rather have a setter or a callback, as in the three-gltf-extensions plugin? I left it out to keep the API surface at zero.
2. The default spacing without `MSFT_screencoverage` (a quarter of the coverage per level) is a guess that looked reasonable on the files I have.
3. The cache swap in `loadAnimation`. It calls `parser.loadAnimation()` itself, so a third-party `loadAnimation` plugin is not asked for animations that target LOD nodes (and likewise a third-party `loadNode` plugin is not asked for the level-0 node). The alternative is a small change in `GLTFParser` so that the object an animation targets can differ from the object that is placed in the hierarchy; cleaner, but it touches core parser code, so I kept it inside the plugin.
4. Should level nodes that are also listed in a scene be removed from it? I did not, because it changes files that work today in a way the specification does not define.
5. No example is added: the glTF examples have no asset with this extension, and I did not want to add a binary to the repository for it. Happy to add one if a suitable sample model exists.

A follow-up (separate PR, on top of this one) adds opt-in loading of the levels on demand and HTTP range requests for `.glb` files, which is where #24506 and #24580 come in. This PR does not depend on it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
