# tk - modelling toolkit API (generated; `from tk import *`)

Units: meters. Z up. Front of an asset faces -Y. Rotations are Euler degrees.


## Parameters, RNG, objects  (`tk/core.py`)

tk.core - params, rng, scene/object basics (Blender 4.1, background mode, no bpy.ops in hot paths).

### class `P`

Resolved parameters with attribute access: p.width == p['width'].

### `resolve_params(schema, overrides=None)`

Merge overrides into a PARAMS schema, coercing types and clamping ranges.

schema entries: {"default": v, "min": a, "max": b, "choices": [...], "desc": "..."}
A "seed" int param is always present (added if the schema omits it).

### `params_hash(asset_id, params)`



### class `Rng`

Seeded RNG. Use rng.fork('name') to get an independent, stable stream for a sub-part so
that changing one part's random draws never reshuffles another part.

- **`.fork(name)`** 

- **`.u(a=0.0, b=1.0)`** 

- **`.jitter(v, amount)`** v +- amount (absolute).

- **`.vary(v, frac)`** v * (1 +- frac).

- **`.chance(p)`** 

- **`.pick(seq)`** 

- **`.weighted(pairs)`** pairs: [(item, weight), ...]

- **`.vec(amount)`** 

### `lod()`

Level of detail being modelled: 0 = the full model (what `tools/tsq build` makes), 1 = mid
(at most 40 % of the triangles of level 0), 2 = far (at most 10 %). build() runs once per level
with the same params and seed; the toolkit already cuts its own tessellation at 1 and 2, the
generator decides what else to skip or replace (`if lod() >= 1:`, lod_upto, lod_pick, lod_thin).
See docs/LOD.md.

### `clear_scene()`



### `root(name)`

Create the root Empty for a model.

### `empty(name, loc=(0, 0, 0), rot=None)`



### `parent(children, parent_obj, keep_transform=True)`



### `mesh_object(name, verts, faces, edges=())`



### `from_bmesh(name, bm)`



### `set_loc_rot(obj, loc=None, rot=None, scale=None)`

rot is Euler degrees (x, y, z).

### `place(obj, loc=None, rot=None, scale=None)`

Alias of set_loc_rot that also works on lists of objects (applied to each).

### `world_matrix(obj)`

World matrix computed from the parent chain without needing a depsgraph update.

### `apply_transform(obj, location=True, rotation=True, scale=True)`

Bake the object's local transform into its mesh data (no bpy.ops).

### `apply_modifiers(obj)`

Evaluate the modifier stack into the mesh (no bpy.ops). Curves/text become meshes.

### `delete(objs)`



### `duplicate(obj, name=None, linked=False)`



### `mirror(obj, axis='X', name=None)`

Duplicate obj mirrored across the plane through the world origin perpendicular to `axis`.

### `descendants(obj)`



### `world_bbox(objs)`

(min Vector, max Vector) over mesh objects (list or a root).

### `hex_rgb(h)`

'#RRGGBB' (sRGB) -> linear RGB tuple. Tuples/lists pass through unchanged (assumed linear).

### `lerp(a, b, t)`



### `clamp(x, a=0.0, b=1.0)`



### `smoothstep(e0, e1, x)`




## Geometry builders  (`tk/geo.py`)

tk.geo - parametric geometry builders (meters, Z up, front = -Y).

Everything returns mesh objects linked to the scene. Builders that know their parametrisation
(lathe, sweep, surface) write proper meter-scale UVs; the rest get box UVs unless you declare
otherwise with uv_grain / uv_cyl / uv_planar.

### `bevel(obj, width=0.003, segments=2, angle=30.0, clamp=True, profile=0.5)`

Bevel modifier on edges sharper than `angle`. Real objects have no razor edges: bevel
anything that catches light (0.5-3 mm metal, 2-8 mm wood, 5-20 mm stone).

### `subsurf(obj, levels=1, simple=False)`



### `solidify(obj, thickness=0.002, offset=-1.0, even=True)`



### `array(obj, count, offset=(1, 0, 0), relative=True, merge=False)`



### `weld(obj, dist=1e-05)`



### `simple_deform(obj, method='BEND', angle_deg=45.0, axis='Z', factor=None)`

Bend / twist / taper / stretch the object (subdivide first so there are verts to bend).

### `boolean(target, cutter, op='DIFFERENCE', delete_cutter=True, solver='EXACT')`

Boolean (applied immediately). Slow on dense meshes: cut before bevelling/subdividing.

### `subdivide(obj, cuts=1)`

Uniformly subdivide all edges `cuts` times (immediate).

### `densify(obj, max_edge=0.25)`

Split edges longer than max_edge (meters) so vertex masks / displacement have resolution.
Use on big flat walls, floors and long beams (immediate).

### `displace_noise(obj, strength=0.005, scale=0.2, seed=0, along_normal=True, axis_mask=(1, 1, 1))`

Organic irregularity: move vertices by smooth 3D noise (feature size ~`scale` m, amplitude
`strength` m). Welded vertices stay welded. Hand-made things (stone, timber, plaster, forged
iron) are never perfectly straight - use small amounts liberally.

### `transform_verts(obj, fn)`

Apply fn(co: Vector) -> Vector to every vertex (local space, immediate). For custom warps,
sags, tapers.

### `recalc_normals(obj, inside=False)`



### `box(name, size=(1, 1, 1), loc=(0, 0, 0), rot=None, bevel=0.0, segments=2)`

Box of full size (x, y, z) centred on loc. bevel = edge chamfer radius (m), baked in.

### `cylinder(name, radius=0.5, depth=1.0, loc=(0, 0, 0), rot=None, verts=32, bevel=0.0, segments=2, cap=True, radius2=None)`

Cylinder / cone frustum along local Z centred on loc. radius2 = top radius (default same).

### `cone(name, r1=0.5, r2=0.0, depth=1.0, loc=(0, 0, 0), rot=None, verts=32)`



### `sphere(name, radius=0.5, loc=(0, 0, 0), segments=32, rings=16, scale=None)`



### `torus(name, major=0.5, minor=0.05, loc=(0, 0, 0), rot=None, major_seg=48, minor_seg=12)`

Torus around local Z. UVs: U along the ring (meters), V around the tube.

### `lathe(name, profile, segments=48, loc=(0, 0, 0), rot=None, angle=360.0, smooth_angle=50.0)`

Revolve a 2D profile [(radius, z), ...] around the Z axis, with meter-scale UVs
(U around, V along the profile).

Order the points bottom-to-top along the visible outer surface; points with radius 0 become
poles. For hollow vessels continue the profile back down the inside to give wall thickness:
    [(0,0),(0.10,0),(0.11,0.01),(0.11,0.15),(0.105,0.15),(0.105,0.005),(0,0.005)]
Build rich turned profiles (beads, coves, fillets) - they are nearly free and read as craft.

### `rounded_rect(w, d, r, seg=6)`

2D rounded rectangle outline [(x, y), ...], CCW, centred.

### `circle_pts(r, n=24, a0=0.0, a1=360.0, center=(0, 0))`

2D arc/circle points. Full circle omits the duplicate end point.

### `arch_pts(width, height, spring=None, kind='round', n=16)`

2D outline of an arched opening (x across, y up), CCW starting bottom-right:
kind: 'round' (semicircle), 'segmental' (shallow arc), 'pointed' (gothic), 'flat'.
width = clear width, height = total height, spring = height where the arch starts
(default: derived). Use for doors, windows, passages, niches.

### `extrude_profile(name, pts2d, depth, loc=(0, 0, 0), rot=None, cap=True, bevel=0.0, segments=2)`

Extrude a closed, simple 2D polygon [(x, y), ...] (CCW) along +Z by depth (base at z=0).
For outlines with holes use shape_extrude().

### `shape_extrude(name, outer, holes=(), depth=0.01, loc=(0, 0, 0), rot=None, bevel=0.0, bevel_res=1)`

Extrude a 2D shape WITH HOLES: outer / each hole is a list of (x, y). The shape lies in the
local XY plane and is extruded from z=0 to z=depth. `bevel` rounds the rim (the outline grows
by `bevel`). Great for cut-out sign boards, pierced ironwork, tracery, plates with slots.
Tip: rotate with rot=(90,0,0) to stand the shape up facing -Y.

### `rounded_box(name, size=(1, 1, 1), radius=0.02, loc=(0, 0, 0), rot=None, seg=5, bevel=0.0)`

Box with rounded vertical edges (radius) and optional small bevel on top/bottom rims.

### `resample_path(points, step=None, n=None, closed=False, smooth_path=True)`

Resample a polyline to roughly uniform spacing (Catmull-Rom smoothed when smooth_path).

### `path_length(points, closed=False)`



### `spiral_path(r0, r1, turns=1.5, n=48, a0_deg=0.0, plane='XZ', center=(0, 0, 0), ccw=True, z_rise=0.0)`

Scroll / volute path from radius r0 to r1 over `turns`. plane: 'XY','XZ','YZ'. The classic
wrought-iron scroll: sweep a flat bar profile along it.

### `arc_path(radius, a0_deg, a1_deg, n=16, plane='XZ', center=(0, 0, 0))`



### `catenary(p0, p1, sag=0.2, n=24)`

Hanging rope/chain/wire between two points with the given sag (m) at mid-span.

### `bezier_path(p0, p1, p2, p3, n=16)`



### `sweep(name, path, profile=0.01, closed=False, sides=10, smooth_path=True, step=None, cap=True, up=(0, 0, 1), scale_fn=None, twist_deg=0.0)`

Sweep a profile along a 3D path -> tube / bar / moulding with meter UVs (U along the path).

path     : [(x,y,z), ...]
profile  : float = round rod radius (with `sides`), or [(a, b), ...] closed CCW cross-section
           in the path's (side, up) frame, e.g. rounded_rect(0.03, 0.008, 0.002, 2) for flat bar,
           or any moulding / gutter / handrail section.
scale_fn : optional fn(t in 0..1) -> scale of the cross-section (tapers: scroll tips, branches).
twist_deg: total twist along the path (twisted square bar balusters!).
up       : reference 'up' for the profile's b axis at the path start.

### `chain(name, path, link_len=0.04, wire=0.004, link_w=None, sides=6)`

Chain of oval links along a path (alternating 90 deg). Returns one mesh.

### `surface(name, fn, nu=16, nv=16, closed_u=False, closed_v=False, thickness=0.0, uv_len=None)`

Mesh from a parametric function fn(u, v) -> (x, y, z), u, v in [0, 1]. UVs are arc-length
based (meters) unless uv_len=(Lu, Lv) is given. thickness > 0 adds a solidify modifier.
Use for cloth (awnings, flags, sacks, tablecloths), terrain, curved roofs, leaves.

### `grid(name, size=(1, 1), cuts=(8, 8), loc=(0, 0, 0), rot=None)`

Flat XY grid (z=0), centred, with meter UVs. Displace / deform it into whatever you need.

### `instances_to_mesh(name, base, matrices, uv_jitter=True, seed=None, keep_base=True)`

Stamp `base` (a mesh object, its own transform ignored) at every 4x4 matrix into ONE mesh.
Fast (numpy) - thousands of roof tiles, leaves, cobbles, bricks, rivets, shingles, petals.
uv_jitter gives each copy a random UV offset so copies don't share the same texture patch.
Materials, UVs and masks of the base are preserved. Give each copy individuality through its
matrix (random rotation / non-uniform scale / tilt).

### `trs(loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0)`

4x4 matrix from location, Euler degrees and scale (float or 3-tuple) - for instances_to_mesh.

### `join(objs, name=None)`

Merge objects into one mesh (world space, identity transform). Modifiers are applied and
UV / smoothing declarations resolved per part first, so each part keeps its own mapping.
Returns the new object; the inputs are removed.

### `text(name, body, size=0.1, depth=0.004, loc=(0, 0, 0), rot=(90, 0, 0), align='CENTER', font='serif', bevel=0.0, spacing=1.0)`

Raised 3D lettering as a mesh. Default rotation stands it up facing -Y (readable from the
front); the text sits on the plane and extrudes `depth` towards the viewer.
font: 'serif' | 'serif_regular' | 'roman' | 'sans' | 'sans_narrow' | 'script' | 'noto_serif'.
size = cap height-ish (m). Returns a mesh object (box UVs).

### `plank(name, length=1.0, width=0.12, thick=0.025, loc=(0, 0, 0), rot=None, bevel=0.003, warp=0.0, rng=None, axis='X', end_mat=None)`

A sawn timber board/beam running along `axis` with wood-grain UVs along its length, slightly
eased edges, optional warp (m of bow/twist) and per-plank random texture offset.
end_mat: optional material for the two end-grain faces (e.g. 'wood_endgrain').

### `stone_block(name, size=(0.4, 0.25, 0.2), loc=(0, 0, 0), rot=None, chamfer=0.012, rough=0.004, rng=None, detail=0.08)`

A dressed stone block: chamfered, subdivided and slightly irregular (each face a little out
of true, arrises knocked). size is the nominal full size. Use for quoins, copings, steps,
kerbs, sills, plinths - individually modelled stones read far better than a textured box.

### `bolt(name, radius=0.008, height=0.006, loc=(0, 0, 0), rot=None, kind='hex')`

Fastener head sitting on z=0 pointing +Z. kind: 'hex' | 'dome' (rivet / clout nail) |
'square' | 'pyramid' (forged nail/stud). Stamp many with instances_to_mesh.

### `ring_band(name, radius=0.3, width=0.04, thick=0.003, loc=(0, 0, 0), rot=None, segments=48, bulge=0.0)`

Flat hoop / band around Z (barrel hoops, straps, collars). width = height of the band.


## Materials, UVs, shading, weathering masks  (`tk/surf.py`)

tk.surf - library materials, world-scale UVs (meters), shading flags and vertex masks.

UV convention: UVs are stored in METERS. The runtime divides by each library material's tile size,
so every surface gets a physically consistent texel density.  For directional materials (wood,
brushed metal, fabric warp, rope) the grain runs along +U.

UV / smoothing are *declared* per object and resolved in realize() after modifiers are applied, so
bevels and booleans never leave you with stale UVs.

### `material_catalog()`

{name: {...}} loaded from docs/materials.json (the list of valid library material names).

### `mat(lib, tint=None, **params)`

Reference a library material by name -> bpy material.

lib    : library material name (see docs/materials.json / `tools/tsq mats`).
tint   : '#RRGGBB' (sRGB) or linear (r,g,b). Multiplies the albedo of tintable materials
         (paint, plaster, fabric, glass colour, emissive colour...). None = material default.
params : optional runtime shader knobs, all floats:
           rough   roughness multiplier (default 1)
           glow    emissive strength for emissive/glass/neon materials (nits-ish, 1..50)
           grime   gain on the grime vertex mask (default 1)
           wear    gain on the edge-wear vertex mask (default 1)
           special gain on the special mask (moss / rust / fade, material dependent)
           scale   texture scale multiplier (default 1; 2 = pattern twice as small)
           opacity for glass-like materials (0..1)
The same (lib, tint, params) always returns the same material, so draw calls are shared.

### `assign(obj, material, slot=None)`

Assign a material (bpy material or library name) to the whole object (or a slot index).

### `assign_faces(obj, material, predicate)`

Assign material to faces where predicate(center: Vector, normal: Vector) is True (local space).
Adds a material slot as needed. Call AFTER the base material is assigned.  Works on the
current base mesh (apply modifiers first if you need the bevelled result).

### `smooth(obj, angle=40.0)`

Smooth shading; edges sharper than `angle` degrees stay sharp (resolved after modifiers).

### `flat(obj)`



### `uv_box(obj, scale=1.0, offset='auto', rot90=False)`

Box (tri-planar) projection in the object's local axes, in meters. V is 'up' (local +Z) on
vertical faces so masonry courses, plaster streaks etc. stay upright.  rot90 swaps U/V.
offset='auto' gives each object a random texture offset so repeated parts don't look cloned;
pass (u, v) in meters or None for an exact placement.

### `uv_grain(obj, axis='X', scale=1.0, offset='auto')`

Directional projection: U runs along local `axis` ('X'|'Y'|'Z' or a 3-vector) on every face
(use for planks, beams, bars: wood grain and brushed streaks follow U). Faces perpendicular to
the axis (end grain) get a planar projection.

### `uv_cyl(obj, axis='Z', scale=1.0, offset='auto', radius=None)`

Cylindrical projection around local `axis`: U = arc length (angle * radius), V = along axis.
radius=None uses the object's mean radius. Caps get a planar projection.

### `uv_planar(obj, u_axis=(1, 0, 0), v_axis=(0, 1, 0), scale=1.0, offset=None)`

Planar projection with explicit local-space U and V directions (meters).

### `uv_keep(obj)`

Keep the UVs the object already has (lathe / sweep / surface generate proper UVs).

### `uv_transform(obj, scale=(1.0, 1.0), offset=(0.0, 0.0), rot_deg=0.0)`

Post-transform existing UVs (applies immediately; object must already have UVs).

### `realize(objs)`

Apply modifiers, resolve smoothing and UV declarations. Accepts one object or a list.
Returns the object(s). Safe to call repeatedly. Bakes object *scale* into the mesh.

### `paint(obj, grime=None, wear=None, special=None, mode='max')`

Paint per-vertex weathering masks (0..1). Each value is a float (uniform) or a vectorised
callable fn(co, normal) -> array, where co / normal are (N,3) numpy arrays in WORLD space.

  grime   : dirt, soot, damp staining (darkens + roughens)        e.g. near the ground, under sills
  wear    : rubbed / chipped areas revealing lighter, smoother substrate   e.g. handles, tread centres
  special : material dependent - moss on stone/roof, rust on iron, verdigris on copper/brass,
            sun-fade on paint/fabric, wetness on water-adjacent stone

mode: 'max' (default), 'add' or 'set'. Auto masks (AO, cavity grime, convex-edge wear, ground
splash) are added on top at export, so only paint the *art-directed* story here.
Modifiers are applied first so masks land on the final vertices.

### `height_ramp(z0, z1)`

Mask helper: 1 at world height z0 fading to 0 at z1 (e.g. paint(o, grime=height_ramp(0, 0.5))).

### `facing(direction=(0, 0, 1), power=1.0)`

Mask helper: how much the surface faces `direction` (e.g. moss / dust on up-facing faces).

### `noise_mask(scale=1.0, seed=0, lo=0.35, hi=0.65)`

Mask helper: smooth 3D value noise (world space, feature size ~`scale` m) remapped lo..hi -> 0..1.
Multiply with other helpers through mask_mul().

### `mask_mul(*fns)`

Combine mask helpers by multiplication (floats allowed).


## Sockets, lights, parts, composition  (`tk/asm.py`)

tk.asm - sockets (lights / attachment points), animated parts and asset composition.

### `socket(name, kind='attach', loc=(0, 0, 0), rot=None, **props)`

A named point exported with the asset (GLB node + sidecar json).

kind 'attach' : a place where the town generator / a parent asset may hang something.
                Convention: the socket's -Y axis points away from the surface (outwards), Z up.
                Give it a `role` prop, e.g. role='sign', 'lantern', 'awning', 'planter',
                'seat', 'prop_top' (things may be put on top), 'door', 'window'.
     'light'  : use light() instead.
     'fx'     : runtime effect anchor, prop fx='flame'|'smoke'|'steam'|'water_jet'|'drip'|'moths'.

### `light(name, loc, color='#ffbf6a', intensity=40.0, range=9.0, flicker=0.12, shadow=False, **props)`

A warm point light emitted by this asset at `loc` (asset space, meters).

intensity : candela. Candle ~1, oil lamp 5-15, gas mantle street lantern 40-90,
            small electric bulb 30-60, big arc/electric globe 150-300. Shop window spill 20-60.
range     : cut-off distance in meters (keep tight: 4-12 m; cost grows with overlap).
flicker   : 0 = steady electric, 0.1-0.2 gas, 0.4-0.7 candle / open flame.
color     : '#RRGGBB'. Flame/gas ~ '#ffbf6a', tungsten ~ '#ffc98a', neon: saturated.
shadow_near=<m> (extra prop): geometry closer to the light than this casts no shadow from it.
            Set it just beyond the light's own housing (lantern cage, shade, bracket) so the
            fitting does not throw a giant shadow of itself: ~0.9 x the housing size.
Every light needs a visible emissive source (flame / mantle / bulb mesh using an emissive
material) at the same spot - never a light from nowhere.
The glow sprite the engine draws on a fitting (a light whose name says lantern / lamp / globe / flame ..., see
tools/town.py light_is_fitting; helper lights get none) follows the light. Two extra props correct it:
halo=<candela>   : the output the glow stands for, when the fitting's light is split over helper lights
                   (default: this light's own intensity; 0 = no glow).
halo_at=(x,y,z)  : where the burner is seen (asset space), when the light itself has to sit off it.

### `fx(name, kind, loc, rot=None, **props)`

Runtime effect anchor: kind = 'flame' | 'smoke' | 'steam' | 'water_jet' | 'drip' | 'moths'.
Useful props: size (m), rate, color.

### `part(objs, name, pivot=(0, 0, 0), motion=None, **props)`

Tag objects as a separately exported, animatable part (everything else merges into 'body').
pivot  : the hinge / suspension point in asset space.
motion : runtime animation hint: 'swing' (hanging signs, lanterns on chains - gentle wind sway),
         'spin' (weather vane, windmill), 'flutter' (flags, bunting, laundry), None = static.
props  : amp (degrees or meters), speed (Hz), axis ('X'|'Y'|'Z').

### `find_asset(asset_id)`



### `load_asset(asset_id)`



### `variant_params(mod, variant=None, overrides=None)`

Resolve params for a named variant (or defaults) plus overrides.

### class `Component`

A sub-asset built inline: one joined mesh (kept out of the scene) + its sockets.
place() stamps a copy into the scene.

- **`.place(loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, sockets=True, name=None)`** Stamp a copy at loc / rot (Euler degrees) / scale. Returns the new mesh object. The component's light / attach / fx sockets are copied too (sockets=False to drop them, or a set of kinds to keep, e.g. {'light'}).

- **`.place_matrix(M, sockets=True, name=None)`** 

### `use(asset_id, variant=None, **overrides)`

Build another asset generator inline and return it as a Component you can place() many
times (buildings use this for windows, doors, shutters, chimneys, lanterns, signs...).

    win = use("window_casement", width=0.9, height=1.4, seed=p.seed + 3)
    for x in xs: win.place((x, 0, 3.2))
    w, h = win.size.x, win.size.z          # measure it to cut matching openings

Every distinct (params) call is a fresh build; reuse the returned Component for repeats.


## Levels of detail - modelled LOD1 / LOD2 (author guide: docs/LOD.md)  (`tk/levels.py`)

tk.levels - levels of detail that are MODELLED, not decimated (author guide: docs/LOD.md).

A level-aware asset (META["lod"] = True) is built three times with the same params and seed:
lod() == 0 is the full model, 1 the mid level (<= 40 % of its triangles), 2 the far level (<= 10 %).
At levels 1 and 2 the toolkit primitives already cut their own tessellation (round sections by a
sagitta tolerance, bevels to one chamfer or none, simplified profiles and paths, chains to a strip,
bolts to nothing); the tools here are for everything only the author can decide.
At level 0 every function in this module is a no-op, so the full model never changes.
The contract of a level (budget, random stream, sockets / lights, parts, glow, bounds) is checked by
`tk/final_levels.py` (`finalize_level`). Its tests: `blender --background --factory-startup --python
tools/lod_tests/contract.py` - run them after any change to tk/levels.py, tk/final_levels.py, tk/final.py or
tools/bl_build.py. `tools/lod_tests/glbdiff.py a.glb b.glb` (Blender's python) compares two GLBs: order-independent
geometry hash and, per material, the triangles without a partner at 1 mm (proof that lod 0 did not change).

### `lod_exact(v)`

Mark a count / bevel width as final: the toolkit will not scale it at any level.
    cylinder("nut", 0.02, 0.01, verts=lod_exact(6))
(lod_pick already returns exact values.)

### `lod_pick(full, mid, far=None)`

A value per level: `full` at lod 0, `mid` at lod 1, `far` at lod 2 (default: `mid`).
    lathe("urn", prof, segments=lod_pick(48, 20, 10))
    style = lod_pick("carved", "chamfered", "plain")
Numbers picked for lod 1 / 2 are used as given (no automatic scaling on top).

### `lod_thin(seq, mid=0.5, far=0.25)`

Fewer of a repeated thing: the whole sequence at lod 0, an evenly spread `mid` fraction at
lod 1 and `far` at lod 2 (first and last always stay; a fraction of 0 gives nothing).
    for x in lod_thin(baluster_xs, 0.5, 0.25): baluster(x, width=bw * lod_pick(1, 1.6, 2.4))
Make the survivors fatter so the run keeps its visual weight. Never draw from the shared `rng`
inside such a loop - draw all values before it, or use rng.fork(f"bal{i}").

### `lod_segs(n, radius)`

The toolkit's own rule for round sections, for rings you build by hand: `n` at lod 0, at
lod 1 / 2 the fewest segments that keep the faceting of a circle of `radius` (m) under
4 mm / 18 mm (never more than n, never under 4 / 3).
    seg = lod_segs(32, r)
    ring = [(r * cos(2 * pi * i / seg), r * sin(2 * pi * i / seg)) for i in range(seg)]

### `lod_path(pts, closed=False, k=1.0)`

A hand-built profile, outline or path with only the points this level needs: the list itself at
lod 0; at lod 1 / 2 the points that keep the line within 4 mm / 18 mm (times k) of the full one. The two
ends (of a ring: at least three points) always stay. For rings and lofts you mesh yourself - lathe() and
sweep() already do this to what they are given.
    prof = lod_path(moulding_profile)                      # [(r, z), ...] or [(x, y, z), ...]
    rings = [[plan(f, d, z) for d, z in prof] for f in stations]

### `lod_budget()`

Triangles this level of the asset may have (40 % / 10 % of what its lod 0 came to with these params),
for generators that fill a triangle target adaptively. None at lod 0 and inside a use()-composed asset.
    target = lod_budget() or TRI_TARGET

### `lod_memo(key, fn)`

A measurement every level must agree on: fn() is evaluated on the full model (lod 0) and that same value
comes back at lod 1 / 2 without calling fn. For whatever is read back from geometry the level changes and then
places or decides other things - a ray cast on a roof, a measured bound, a triangle count that decides
whether optional dressing is added.
    z = lod_memo(("roof_z", round(x, 4), round(y, 4)), lambda: ray_z(roof, x, y))
    room = lod_memo("room_for_dormers", lambda: scene_tris() < CAP)
key: hashable and unique within the asset (give every call its own: include what the value depends on).
The value must be plain data (numbers, strings, tuples / lists, arrays, Vectors), not Blender objects.

### class `lod_upto`

Detail that exists only up to a level: the block runs at EVERY level (so random draws,
sockets, lights and parts made inside are identical at every level), then the mesh objects
created inside are emptied when lod() is above `level`.
    with lod_upto(0):            # full model only
        rivets = instances_to_mesh("rivets", bolt("rv", 0.006), mats)
    with lod_upto(1):            # gone at lod 2
        glazing_bars(...)
Geometry with a glowing material (emissive, neon, window_lit, glass_*) is never emptied.
Use a plain `if lod() <= n:` instead only when the block draws no random numbers, makes no
sockets / parts and is expensive to build.

### `lod_drop(objs, above=0)`

Function form of lod_upto: empty these objects (one or a list, nested lists allowed) when
lod() is above `above`. Returns objs. Emptied objects stay valid (join / instances_to_mesh /
part() keep working); glowing geometry is never emptied.

### `lod_faces(objs, keep=None, drop=None, above=0, cos=0.5)`

Delete the faces nobody sees at the cheaper levels (backs, undersides, faces buried in a wall):
when lod() is above `above`, in these mesh objects (one or a list, nested lists allowed)
    keep=(0, -1, 0)                only faces that look along the direction stay (a box becomes a card)
    drop=[(0, 0, -1), (0, 1, 0)]   faces that look along a direction go (underside and back of a box)
Directions are asset-space vectors (one or a list) at the moment of the call - place / rotate the
object first; a face looks along d when the cosine between its normal and d is above `cos`.
Modifiers are applied first. Returns objs.
    lod_faces(box("bar", (0.5, 0.02, 0.016), loc=...), keep=(0, -1, 0))    # glazing bar: its front
    lod_faces(plinth, drop=(0, 0, -1), above=1)                             # no underside at lod 2

### class `lod_as`

Model a block at another level than the build's: `with lod_as(0):` keeps something exactly
as in the full model (dial numerals, lettering, the one detail that carries the asset's
identity); `with lod_as(lod() + 1):` builds hidden / back / inside parts one level cheaper.
Has no effect in the lod 0 build.

### `lod_standin(objs, mid=None, far=None)`

An IDENTITY PART of an asset that has no META["lod"] (clock hands and hour marks, a shop's emblem, the
pointer of a weathervane ...) and, natively, its cheap stand-ins for the reduced levels. The full model is not
touched: build() still runs once, `objs` are built as always and only get a tag.

  objs      the mesh objects of the part as modelled for lod 0 (one or a list, nested lists allowed). Call
            this BEFORE they are joined / instanced: the tag travels through join() and instances_to_mesh().
  mid, far  what stands in their place at lod 1 / lod 2:
              None                  nothing: the originals stay as modelled and are protected (see below)
              a callable -> objs    builds the stand-in. Called right here, at lod-0 time, with the build
                                    counter set aside, so no name / UV offset of the full model shifts. It
                                    must build from NUMBERS: draw nothing from rng (nor from a fork), use the
                                    materials the originals use (no new mat() variants), and return finished
                                    mesh objects in WORLD position - originals that are instanced need their
                                    stand-in instanced by the same matrices. The objects are taken out of
                                    the scene and kept for the reducer.
              objects               the same, already built (their toolkit calls then do move the counter)
              () / a callable -> () the part is dropped whole at that level

At lod 1 / lod 2 tk.reduce removes the tagged originals of a group that has a stand-in for the level and puts
the stand-in there; stand-ins and kept originals are PROTECTED like lettering: never welded, patched,
collapsed, dropped or replaced by a prism. Their triangles count toward the level's budget (the rest of the
asset gets what is left), so keep them lean. The clearance rule to cards holds for them as for everything: a
hand must not lie within 4 mm of the dial card (the depth-fight gate of the level measures it). Lettering
(text()) inside a group that is swapped at a level is not held to the lettering gate at that level.
In every exported GLB (lod 0, the decimated copies, both levels) the vertices of tagged originals and of
stand-ins carry the float vertex attribute _TSQ_KEEP = 1.0 (see docs/LOD.md), so a later simplification
(the web bake's far pass) can lock them.

No-op in a level-aware asset (META["lod"]: model the levels with lod() there) and above lod 0. Inside a
composed child (tk.asm.use) no stand-in is made: the originals are kept and protected. Returns objs.

    hands = [hour, minute]
    lod_standin(hands, mid=lambda: flat_hands(1.0), far=lambda: flat_hands(2.3))
    lod_standin(minute_ticks, mid=(), far=())

### `lod_keep(objs)`

lod_standin(objs) without stand-ins: these objects are an identity part of the asset - the reducer keeps
them as modelled at lod 1 and lod 2 (never welded, patched, collapsed, dropped or replaced by a prism; their
triangles count toward the budget) and their vertices carry _TSQ_KEEP in the exported GLBs. Returns objs.


## Reduced levels - lod 1 / 2 made from the full model (build side, `from tk import reduce`)  (`tk/reduce.py`)

tk.reduce - cheaper levels (lod 1 / 2) made FROM the full model, for assets that do not model their own.

Where the decimated copy (tk.final._make_lod) collapses everything by one ratio, the reducer works on the
un-merged scene and takes the triangles from where nobody sees them first:

  1. faces no viewer can see (buried in other parts, undersides near the ground, backs against the wall) go
  2. coplanar faces merge (limited dissolve)
  3. thin long pieces (bars, rods, rails, gently bent tubes) become 4-sided (lod 1) / 3-sided (lod 2) prisms
  4. loose pieces smaller than the level's size limit are dropped - never the ones that carry the outline of the
     asset, the last piece of an animated part, or anything next to something that glows
  5. scattered plant cards are thinned (half as hard at lod 1) and the survivors enlarged
  6. fields of small pieces (stones, tiles, cobbles) are welded into one skin, as far as the outline stands it
     and the field stays covered (_open: a weld that lays the darker core behind an ashlar skin bare is undone)
  7. what is still over the budget comes out of the remaining dense meshes by collapse - or, where a field would
     not weld (roof tiles laid over each other) and the collapse would tear it, the field becomes flat patches
     in the planes of its pieces (one rectangle per row of them) and the collapse only gets what is left

The collapse works loose piece by loose piece: a piece only takes a step that leaves it whole (_collapse_part), so
a part may end over its ratio - what cannot be had without tearing is left (see level_candidates: "reduced15",
"reduced20").

Everything that glows is kept as it is (only where it is most of the model it is collapsed along with the rest;
glowing panes tiled in one plane become one pane per block of the size limit, _panes). No step may lay another
surface onto a glowing face: nothing is welded or snapped onto one, a piece takes no collapse step that brings it
within Z_NEAR of one, and _zfix() removes what a glowing card hides and pushes off what still lies on it.
The room card of a window (materials of tk.lodimage.Z_CARD: window_lit whatever its glow, window_dark) is such a
card lit or not: the dark card of an unlit room is kept as it is and kept clear of the wall the same way.

Lettering (tk.geo text(): face attribute "tsq_text") is an identity class: never welded, patched or collapsed. It
loses backs and bedded faces, at lod 2 its sides, its flat fronts are dissolved; a group under tk.lodimage.TEXT_PX
at the level's distance becomes its flat front or goes - whole, never in part.

After every step the bounds and the parts are checked and the step is taken back where it broke them; hidden-face
removal and welding are also checked on a quick image (tk.lodimage.Probe).

    snap = snapshot()                           # right before tk.final.finalize() of lod 0
    ...finalize, reference...
    restore(snap); log = reduce_scene(1, ref, meta, params); tk.final_levels.finalize_level(1, ref, ...)
    cands, log = level_candidates(1, snap, ref, meta, params, stem)     # the three lines above in one, for pick()

    obj = reduce_child(obj, level, meta)        # tk.asm.use(): a composed asset without levels of its own

Nothing here draws from the toolkit's random streams or counters.

### `visibility(objs, mount='ground', allow_below=False)`

From how many of ND directions each polygon can be seen (capped at CAP) -> [int8 array per object].
A polygon is tested at 1-10 points per triangle (by area), on its front (both sides for two-sided materials);
a direction counts when a ray from the point towards the viewer hits nothing that hides (glass, glow, plants
and water hide nothing). Viewers stand on the ground (ground-mounted assets: no view from below EYE_MIN) and in
front of the wall (wall-mounted: no view from behind y = 0). Glow / glass / water / plant faces report CAP.

### `reduce_objects(objs, level, budget, mount='ground', allow_below=False, allow=1.0, cache=None, bounds=None, diag=None, patches=True, lights=None)`

Reduce mesh objects (realized: modifiers, UVs and smoothing resolved) in place to about `budget` triangles.
mount: 'ground' | 'wall' | 'wall_ground' | 'hang' - where the viewers of the asset can stand.
allow: the level may end at up to allow x budget before the last-resort collapse of everything is used.
cache: dict kept between the levels of one asset (the visibility is computed once for both).
patches: False leaves the flat-patch plan of step 8 out.
lights: tk.lodimage.lights_of(sockets) - the asset's own lights: the unseen faces that cast their shadows stay
        (_light_casters; the later steps do not know them - a caster dropped or moved there shows as "leak").
bounds: (min, max) of the full model when the objects are not all of it.
-> log dict; log["objs"] are the objects that now make up the level (some are new).

### `reduce_scene(level, ref, meta, params, allow=1.0, patches=True)`

Reduce the un-merged scene of the full model (restore(snapshot())) to level 1 / 2, ready for
tk.final_levels.finalize_level(). ref: tk.final_levels.reference() of lod 0. -> log dict.

### `force_scene(level, ref, limit=None)`

The scene of reduce_scene(..., allow > 1) that ended over its budget, brought under `limit` triangles (default:
the budget). -> log dict

### `level_candidates(level, snap, ref, meta, params, stem, times=None)`

The reduced candidates of a level for tk.final_levels.pick(): puts the snapshot() back into the scene, reduces
it and exports it through tk.final_levels.finalize_level (cand_file(stem, name, level)).
  "reduced"     the level inside its budget
  "reduced15"   a lod 2 that ended between its budget and tk.lodimage.ALLOW2 x the budget; "reduced" is then the
                same scene forced under the budget
  "reduced20"   a lod 2 of a building / landmark that ended between ALLOW2 and ALLOW2B x the budget (stopping
                short of the budget is better than tearing); "reduced15" and "reduced" are then that scene forced
                further, one after the other
  "..._np"      the same chain without the flat-patch plan, made when the first chain took it: pick() then ships
                the patches only where they give the lower image difference
times: optional dict, CPU seconds are added to its "reduce_s" / "finlevel_s".
-> ([candidate, ...], log of the first chain; log["np"]: of the second)

### `reduce_child(obj, level, meta=None)`

tk.asm.use(): the joined mesh of a composed asset that has no levels of its own, reduced for `level`
(40 % / 10 % of its triangles). Replaces obj; -> the new object.

### `snapshot()`

Call right before tk.final.finalize() of lod 0. Resolves modifiers, UVs and smoothing of the scene in the
order finalize() itself does (so it draws exactly the texture offsets it would have drawn) and returns
unlinked copies of the un-merged mesh objects with their world transform. clear_scene() destroys them.

### `hold()`

unlinked copies of the mesh objects of the scene as they are now (see restore)

### `clear_meshes()`

remove every mesh object from the scene (sockets stay)

### `restore(snap)`

Put fresh copies of a snapshot() / hold() into the scene (after clear_meshes()) -> the new objects

### `release(snap)`

free a snapshot() / hold()


## Level image metric and the choice between candidates (build side, numpy only)  (`tk/lodimage.py`)

tk.lodimage - how a cheaper level LOOKS next to the full model, measured on images, and the choice between
the candidates of a level (modelled / reduced / decimated).

The metric is a point-splat render, not the real renderer: every model is sampled into surface points (a few per
pixel of the distance where the level takes over: 16 m for lod 1, 44 m for lod 2), the points are shaded with the
library material's mean colour, the vertex masks and one fixed sun, culled like the runtime culls (single-sided
unless the material is glass / water / double) and splatted orthographically from a ring of views.  Per candidate:

    sil    1 - |A and B| / |A or B| of the silhouettes                      (mean over the views; sil_max the worst)
    img    mean absolute difference of the blurred sRGB images over the union of both silhouettes
    lum    lit brightness of the candidate / of lod 0 where both cover
    glow   share of the glowing pixels of lod 0 that still glow (None when lod 0 hardly glows; glow_px: the share
           of the lod-0 picture that glows - under GLOW_PX the coverage is not judged)
    tear   share of the lod-0 picture torn open over more than the level's size limit; loose: pieces that touched
           in lod 0 and no longer do
    text   lettering (faces tagged by tk.geo text()), every group drawn alone at a finer scale: see text_measure
    zf     depth fight: glowing area with a parallel face of another material within 3 mm, against lod 0 (zfight;
           the measure of tools/checks/zscan.py)
    ident  smallest share of its lod-0 pixels any material kept (materials with >= 1 % of the picture)
    Local measures (a defect in one place hardly moves a mean over the picture; each is a gate against lod 0 and,
    where the decimated copy is given, only when that copy is clearly better - gates()):
    tile     worst tile of the worst view: mean image difference inside it (TILE; TILE_IDENT times wider for a copy
             carrying tk.lod_standin stand-ins) - a torn dial, a hole
    sil_new  share of the level's silhouette lying outside lod 0's silhouette grown by two pixels, worst view
             (SIL_NEW) - a shard, a filled hoop
    glow_min glow of the view where the least of it is left (GLOW_VIEW) - a lit face gone dark on one side
    leak     share of the directions around a light of the asset in which lod 0 casts a shadow and the level lets
             the light land on something a viewer sees (LEAK; shadow_casts, light_leak) - a missing shadow caster
    uvbad  share of the surface whose texture density is off by more than 2x (reported, never a gate)

    m = Metric("out/x/default.glb", mount="ground")
    a = m.score(1, "out/x/default.lod1.glb"); b = m.score(1, "out/x/default.dec1.glb")
    pick = choose(1, [{"name": "reduced", "kind": "reduced", "hard": [], "tris": 900, "budget": 1000, "m": a},
                      {"name": "decimated", "kind": "decimated", "tris": 1000, "budget": 1000, "m": b}])

Needs numpy only (no bpy), GLB files as written by tk.final (Y-up, _MASK attribute, material extras tsq_lib/tsq_tint).
Probe is the in-memory cousin used by tk.reduce to see whether a reduction step damaged the outline.

### `load_glb(path, cache=True)`

-> [primitive]: {"P" (n,3) z-up metres, "N", "UV" | None, "MASK" (n,4), "I" (m,3), "lib", "tint", "name", "double"}

### class `Metric`

Image comparison against one lod-0 file. The lod-0 renders are made once per level and reused for every
candidate. sockets: the asset's sockets (default: read from the sidecar next to glb0) - the shadows of its
lights are compared too (m["leak"]).

- **`.score(level, glb)`** -> metrics dict of the level file `glb` against lod 0

### `gates(level, m, dec=None)`

Image gates of a level -> reasons it fails (empty = passes). With `dec` (the metrics of the decimated copy,
which is what ships otherwise) a gate only fails when the candidate is also worse than the decimated copy there;
dec=None gives the absolute verdict.
Whole: the silhouette; no hole wider than the level's size limit where lod 0 is closed ("torn open"); what touched
in lod 0 still touches (see loose()); in absolute terms also no more than OPEN of the tiled fields bare ("torn
field", m["open"] from tk.reduce - the decimated copy has no such measure, so this one never counts against it:
it keeps a reduced level over its budget out and lets a building's 2 x candidate win over a cheaper torn one).
Depth fight: the glowing area with a parallel face of another material closer than 3 mm (zfight) may exceed that
of lod 0 by z_tol() at most.
Lit brightness: outside LUM fails on its own; against a decimated copy that is outside LUM too the image
difference decides; against one inside LUM the candidate passes only within LUM_WIDE and with an image difference
at least LUM_GAIN lower.

### `choose(level, candidates)`

Pick what ships at a level.

candidates: [{"name", "kind": "authored" | "reduced" | "decimated", "tris", "budget", "hard": [reasons from
tk.final_levels.finalize_level other than the budget], "m": Metric.score(...) (+ "text": text_measure(...)),
"allow": how far over the budget this asset may go at this level (default ALLOW2 at lod 2, else 1)}].
Order: hard gates, then the image gates, then the lowest image difference wins; the decimated copy (always
eligible - it is the fallback) only wins when it is better by more than DEC_MARGIN, and never over a candidate
whose lettering (or depth-fight gate) passes when its own does not, nor over a reduced copy of an asset that
declares identity parts ("ident": True - tk.lod_standin / lod_keep) and passes its gates.
A reduced lod 2 over its budget stays eligible when it passes every gate in absolute terms (the brightness gate
as above, against the decimated copy) and beats the decimated copy on image difference: up to ALLOW2 x the budget
it wins when it is better than the best candidate inside the budget by more than DEC_MARGIN - and, when that one
passes every gate in absolute terms, by at least ALLOW_GAIN of its image difference; between ALLOW2 and its "allow" (ALLOW2B for buildings and landmarks) only when
nothing cheaper passes or the best cheaper one fails the silhouette or tear gate in absolute terms.
Last resort at lod 2 ("last": True): when nothing passes and the decimated copy fails its lettering or the
depth-fight gate in absolute terms, the reduced candidate inside its allowance with the lowest image difference
ships that passes every gate but the torn-field one and has no more than OPEN_LAST of its fields bare.
At either level ("last": True too): when nothing passes and the decimated copy fails the depth-fight gate in
absolute terms, the reduced candidate inside its allowance with the lowest image difference ships that passes
that gate, its lettering, tears, loose pieces and glow in absolute terms, with a silhouette within Z_MISS[0] x
the limit, a brightness within LUM_WIDE and an image difference within Z_MISS[1] x the decimated copy's.
-> {"level", "winner": name, "kind", "allowance": bool, "over": 0 | 1.5 | 2.0, "why": {name: [reasons it lost]},
    "clean": the winner passes the absolute image gates, "img": its image difference}

### class `Probe`

Outline / depth / glow of a triangle soup against a reference soup, from a handful of views.

    pr = Probe(V0, T0, glow0, two0, mount, level)
    pr.measure(V, T, glow, two, tone) -> {"sil", "hole", "lost", "glow", "lum", "tone"}
tone: the albedo luminance of every triangle (optional); lum is then the brightness of the picture over the
reference's, unlit (what dropping dark details off a light ground, or the reverse, does to it), and "tone" the
mean difference of the two pictures pixel by pixel (gamma space) - which two shifts in opposite directions
cannot hide.
sil: silhouette difference as in Metric; hole: share of the reference pixels where the surface now lies deeper
than `dtol` (something in front went missing); lost: share of the reference pixels with nothing left within one
pixel (a real gap - shaving a sub-pixel rim off thin things does not count); glow: share of the reference's
glowing pixels that still glow.

- **`.wins(V, T, two, spp=8.0, see=None)`** How many pixels each triangle of the soup wins over all views (densely sampled, so that hardly anything shows through the gaps between the samples of what lies in front) -> (len(T),) int array see: clear glass (per triangle) - what lies behind it wins its pixels as well as the glass

- **`.measure(V, T, glow, two, tone=None, see=None)`** 


## Level contract and what ships (build side)  (`tk/final_levels.py`)

tk.final_levels - export and checks of the MODELLED cheaper levels (lod 1 / 2) of a level-aware asset.

tools/bl_build.py builds level 0 exactly as always (tk.final.finalize), captures it with reference(),
then runs build() again at level 1 and 2 and hands each scene to finalize_level():

  * merge per part like level 0, weighted normals, same part pivots
  * the vertex masks (AO / grime / wear / special) are TRANSFERRED from level 0 instead of baked again,
    so a cheaper level keeps the weathering of the full model (no popping when the level switches, and
    edge wear does not vanish with the bevels that produced it)
  * the level is rejected (the caller then ships the decimated fallback) when it is over budget, moves or
    loses a socket / light / part, loses something that glows, changes the bounds, or shifted the random stream

The same finalize_level() exports the REDUCED levels tk.reduce makes from the full model (docs/LOD.md, "Reduced
levels"). What ships at a level is then one of up to three candidates - modelled ("authored"), reduced, decimated -
chosen by pick() from their pictures (tk.lodimage).

### `reference(info, keep)`

Capture level 0 right after tk.final.finalize(): geometry + masks (numpy, survives clear_scene).
"text": the lettering of lod 0 (tk.lodimage.text_prepare), what the levels' lettering is checked against.

### `breakdown()`

{source object name: triangles} of the scene before the merge (modifiers evaluated; Blender's .001 suffix
and trailing counters stripped, so repeated things add up). For `tools/tsq lod`: where the triangles are.

### `budget(level, tris0)`

Triangle budget of a level for an asset of tris0 triangles at level 0.

### `candidate(name, kind, res=None, file=None, **kw)`

One entry of pick()'s candidate list. kind: "authored" | "reduced" | "decimated"; res: the finalize_level()
result of the level (the decimated copy has none: pass tris=, budget=, glb_bytes=); file: its GLB. The caller
adds "m" = tk.lodimage.Metric.score(level, file).

### `pick(level, cands)`

What ships at a level, from the candidate list [candidate(), ...] with their image metrics in "m".

Without a modelled level this is tk.lodimage.choose(): hard gates, image gates, lowest image difference; the
decimated copy only wins when it is better by more than DEC_MARGIN; a reduced lod 2 may use up to ALLOW2 x
the budget. A modelled ("authored") level is the author's decision and is not put up for that vote:
  * one that keeps its contract ships. When it misses an image gate (silhouette, lit brightness, glow) that
    another candidate passes with a lower image difference, that candidate is named in "better" (the build
    prints it as a warning) - and ships instead when AUTHORED_YIELDS is on.
  * one that breaks its contract (budget, sockets, parts, glow, bounds, random stream, build failed) is replaced
    by the decimated copy (REJECTED_SHIPS), as before.
-> the dict of choose() plus "kept": the image gates a modelled level misses although it ships, "better": the
   name of the candidate that passes them (or None).

### `finalize_level(level, ref, glb_path, params, meta, rng=None, rec0=None, build_seconds=0.0, keep_rejected=False, seen=0.3, fit=False, text=False)`

Merge / mask / check / export the scene of a level >= 1 build -> result dict
{"level", "ok", "why": [reasons it was rejected], "warn": [...], "tris", "glb_bytes", "ratio", "budget",
 "cover", "dev", "seconds"}. The file <variant>.lod<level>.glb is written when the level is accepted
(or keep_rejected). The scene is a modelled level (build() at tk.lod() == level) or the reduced full model
(tk.reduce.reduce_scene: seen / fit are the mask transfer of those, see _transfer_masks; text=True adds "text":
tk.lodimage.text_measure() of the level's lettering against lod 0's - only a level made from the lod-0 scene
carries the same text numbers).
