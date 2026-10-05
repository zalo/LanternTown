# tk.tex - texture synthesis API (generated; `from tk.tex import *`)

tk.tex - procedural, seamlessly tiling PBR texture synthesis in numpy (runs in Blender's Python).

All lengths are in METERS on the surface. A canvas `t` covers one tile of `t.tile = (W, H)` meters
with `t.res = (w, h)` pixels. Every operation wraps around, so results always tile.

A material generator is a file materials/<name>.py:

    from tk.tex import *
    META = {...}
    def build(t):
        ...
        return dict(albedo=rgb_linear, height=height_m, rough=r, metal=m)     # arrays or scalars

Arrays are float32, shape (h, w) or (h, w, 3). Image row 0 is the TOP of the texture (V = 1).

### `sat(a)`



### `mix(a, b, t)`

Linear blend; works for scalars, (h,w) and (h,w,3) (t may be (h,w) against colours).

### `smoothstep(e0, e1, x)`



### `remap(a, lo, hi, out_lo=0.0, out_hi=1.0, clip=True)`



### `norm01(a)`

Rescale to exactly 0..1.

### `level(a, lo_pct=1.0, hi_pct=99.0)`

Robust rescale to 0..1 using percentiles (ignores outliers).

### `hexlin(h)`

'#RRGGBB' sRGB -> linear RGB float32 array(3).

### `lin2srgb(a)`



### `ramp(t, stops)`

Colour gradient lookup. t: (h,w) in 0..1; stops: [(pos, '#hex' or (r,g,b) linear), ...].
Interpolates in linear RGB. Returns (h,w,3). The main way to colour things: build a mask or
noise and map it through 3-6 carefully chosen real-world colours.

### `colour(c, shape=None)`

Constant colour image from '#hex'.

### `luminance(rgb)`



### `desaturate(rgb, amount)`



### `overlay_tint(rgb, tint, amount)`

Multiply rgb towards a tint colour by `amount` (0..1 scalar or mask).

### class `Layout`

Result of a cell layout (bricks / stones / planks / tiles).
.dist  distance (m) from each pixel to the nearest cell border (0 at the border, grows inwards)
.id    integer cell id per pixel
.u/.v  cell-local coordinates in meters, origin at the cell centre (u along the cell's length)
.n     number of cells
.rand(k) -> per-pixel array of a per-cell random value in 0..1 (k = channel: different k,
          different numbers) - use for per-stone colour / height / tilt variation
.tilt(amount_m, k) -> per-cell random planar tilt height field (each stone sits a bit crooked)

- **`.rand(k=0)`** 

- **`.tilt(amount=0.003, k=0)`** 

- **`.mask(gap=0.004, soft=0.002)`** 1 inside cells, 0 in the joints (joint half-width = gap/2, edge softness = soft).

- **`.rounded(t, radius=0.01)`** Round off the cells' sharp corners: returns a new Layout whose .dist has corners eased by about `radius` m (river cobbles, worn bricks, pillowy tiles). `t` is the canvas.

- **`.dome(radius=0.02, power=2.0)`** Rounded-edge profile 0..1: 0 at the border rising to 1 at `radius` m inside (pillowed stones / eased brick arrises). Multiply by a height in meters.

### class `Canvas`



- **`.zeros(v=0.0)`** 

- **`.white(seed=None)`** Uniform white noise 0..1 per pixel.

- **`.noise(scale=0.1, seed=None, stretch=(1.0, 1.0), bandwidth=0.6)`** Smooth band-limited noise with features about `scale` meters across. Returns 0..1 (mean 0.5, roughly gaussian). stretch=(sx, sy) elongates features: stretch=(8,1) gives streaks 8x longer along X (wood grain, brushed metal, drips with (1,8)).

- **`.fbm(scale=0.2, octaves=5, gain=0.5, seed=None, stretch=(1.0, 1.0), lacunarity=2.0)`** Fractal noise: sum of noise() octaves starting at `scale` m, each half the size. Returns 0..1 (re-levelled). gain 0.5 = natural; 0.7 = rough/gritty; 0.35 = soft.

- **`.ridged(scale=0.2, octaves=4, gain=0.5, seed=None, stretch=(1.0, 1.0))`** Ridged noise 0..1: sharp crests (veins, cracks when inverted, bark ridges).

- **`.blur(a, sigma, sigma_y=None)`** Gaussian blur with sigma in meters (wraps). sigma_y for anisotropic blur.

- **`.sample(a, X, Y)`** Bilinear, wrapping lookup of image `a` at surface positions X, Y (meters).

- **`.warp(a, amount=0.01, scale=0.1, seed=None)`** Distort image `a` by a smooth random vector field (amount, scale in meters). Breaks up anything that looks too straight / mechanical.

- **`.shift(a, dx=0.0, dy=0.0)`** Translate by (dx, dy) meters (wraps).

- **`.slope(h)`** Gradient (dh/dx, dh/dy) of a height field in m/m.

- **`.curv(h, radius=0.004)`** Convexity of a height field: >0 on ridges / raised edges, <0 in pits / grooves (h - blur(h)), in meters. Use it to put wear on edges and dirt in cavities.

- **`.cavity(h, radius=0.004, depth=0.001)`** 0..1 mask of recesses: 1 where the surface is `depth` m below its surroundings.

- **`.edges(h, radius=0.004, height=0.001)`** 0..1 mask of raised edges / high points (exposed to wear).

- **`.ao(h, radius=0.01, strength=1.0)`** Ambient occlusion 0..1 from a height field (multi-scale cavity estimate).

- **`.normal(h, strength=1.0)`** Tangent-space normal map (h,w,3) in -1..1 from a height field in METERS (physically correct slopes; strength exaggerates).

- **`.stripes(period, axis='x', duty=0.5, soft=0.0)`** Periodic bands 0/1 across `axis` ('x': bands vary along X). period is snapped so it tiles.

- **`.wave(period, axis='x', phase=0.0)`** Sine 0..1 along axis with a period snapped to tile.

- **`.grid(cell_w, cell_h, shift=0.5, jitter_w=0.0, seed=None, shift_rows=True)`** Brick / plank / tile layout: rows of cells `cell_w` x `cell_h` meters (snapped to tile), each row shifted by `shift` cells relative to the previous (0.5 = running bond, 0 = stack, 'random' = random per row). jitter_w (0..0.9) randomises cell lengths within a row. With shift_rows=False the columns are shifted instead (vertical planks).  -> Layout. For joints: lay.mask(gap); the joint is at lay.dist == 0.

- **`.cells(size=0.1, jitter=0.9, seed=None, aspect=1.0)`** Voronoi cell layout: irregular stones / cobbles / cracks / scales about `size` meters across (aspect > 1 stretches cells along X). jitter 0 = regular grid, 1 = fully irregular. -> Layout with exact distance-to-border (.dist), ids, local coords (.u, .v) and extra fields .f1 (distance to the cell's seed point) for round bumps.

- **`.cracks(size=0.3, width=0.0015, seed=None, wander=0.02, density=1.0)`** 0..1 mask of thin wandering cracks (Voronoi borders, warped). density < 1 removes a fraction of the crack network so cracks start and stop.

- **`.speckle(density=2000.0, size=0.002, seed=None, soft=0.5)`** Random dots: `density` per square meter, about `size` m across. 0..1 mask (pores, pits, grit, paint flecks, lichen spots with bigger size).

- **`.scratches(density=60.0, length=0.05, width=0.0006, seed=None, angle=None, angle_spread=180.0)`** 0..1 mask of fine straight scratches (density per m^2, length/width in m). angle (deg, 0 = along X) with +-angle_spread/2 variation; default random directions.

- **`.drips(scale=0.05, length=8.0, seed=None)`** Vertical streak noise 0..1 (rain runs, rust runs, soot washes): noise stretched along Y.

- **`.fibres(scale=0.004, length=20.0, seed=None, axis='x')`** Fine directional fibre / grain noise 0..1 running along `axis`.

### `render_far(t, albedo, nrm, rough, metal, ao, out_w=1024, out_h=576)`

Perspective view across ~7 repeats of the tile lit by one warm lantern over a dim blue night
ambient: shows repetition, macro read and how the relief/gloss catch a low light (the town look).
