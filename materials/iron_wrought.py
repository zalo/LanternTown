"""Hand-forged black iron.

Built the way a smith leaves it: a bar drawn out under the hammer (overlapping dished blows whose
creases stand proud), a thin blue-black crust of fire scale that has spalled off in angular
patches, sparse hammered-in scale pits, then blackened with oil/wax. Crests of the facets are
burnished to dark steel by handling; a faint brown bloom sits in the pits and the scale-free lows.
"""
from tk.tex import *

META = {"normal_strength": 2.0}


def blows(t, size, seed, aspect=(1.0, 1.9), jitter=0.95, level=0.35, tilt=0.25, k=9.0, cur=(0.7, 1.4)):
    """Lower envelope (soft-min) of shallow elliptical bowls = overlapping hammer blows.
    Returns (h, r): h ~0 at a blow centre rising to ~1 half a `size` away; r = per-blow random."""
    rng = np.random.default_rng(seed)
    nx = max(3, round(t.W / size))
    ny = max(3, round(t.H / size))
    cw, ch = t.W / nx, t.H / ny
    px = (np.arange(nx)[None, :] + 0.5 + (rng.random((ny, nx)) - 0.5) * jitter) * cw
    py = (np.arange(ny)[:, None] + 0.5 + (rng.random((ny, nx)) - 0.5) * jitter) * ch
    ang = rng.random((ny, nx)) * math.pi
    asp = aspect[0] + rng.random((ny, nx)) * (aspect[1] - aspect[0])
    off = rng.random((ny, nx)) * level
    cur = cur[0] + (cur[1] - cur[0]) * rng.random((ny, nx))
    tx = (rng.random((ny, nx)) - 0.5) * 2 * tilt
    ty = (rng.random((ny, nx)) - 0.5) * 2 * tilt
    rr = rng.random((ny, nx))
    gx = np.floor(t.X / cw).astype(np.int64)
    gy = np.floor(t.Y / ch).astype(np.int64)
    acc = np.zeros(t.X.shape, dtype=np.float64)
    best = np.full(t.X.shape, 1e9, dtype=F)
    rid = np.zeros(t.X.shape, dtype=F)
    R = 0.5 * math.sqrt(cw * ch)
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            cx, cy = gx + ox, gy + oy
            wx, wy = cx % nx, cy % ny
            dx = t.X - (px[wy, wx] + (cx - wx) * cw)
            dy = t.Y - (py[wy, wx] + (cy - wy) * ch)
            c, s = np.cos(ang[wy, wx]), np.sin(ang[wy, wx])
            u = (dx * c + dy * s) / R
            v = (-dx * s + dy * c) / R
            q = (u * u / asp[wy, wx] + v * v * asp[wy, wx]) * cur[wy, wx] + off[wy, wx] + tx[wy, wx] * u + ty[wy, wx] * v
            acc += np.exp(-k * q)
            closer = q < best
            rid = np.where(closer, rr[wy, wx], rid)
            best = np.where(closer, q, best)
    return (-np.log(acc + 1e-30) / k).astype(F), rid.astype(F)


def build(t):
    # ---- macro: a hand-drawn bar is never flat
    und = (t.fbm(0.16, 3, 0.5, seed=1) - 0.5) * 0.0007

    # ---- meso: hammer blows. Heavy drawing-out blows of very different weight (so some swallow
    #      their neighbours) + lighter planishing marks; soft-min keeps the creases rounded.
    b1, r1 = blows(t, 0.030, 11, level=0.7, tilt=0.75, k=7.0, cur=(0.3, 0.8))
    b2, r2 = blows(t, 0.014, 12, level=0.8, tilt=0.5, k=5.0)
    b1 = t.warp(b1, 0.0015, 0.03, seed=2)
    hammer = np.clip(b1, -1.0, 2.0) * 0.00075 + np.clip(b2, -1.0, 2.0) * 0.00012
    hammer = t.blur(hammer, 0.0009)
    crest = t.edges(hammer, 0.006, 0.00016)                    # creases between blows stand proud
    low = t.cavity(hammer, 0.007, 0.00020)

    # ---- fire scale: thin brittle crust, spalled in angular patches (more on the crests)
    fn = t.warp(t.fbm(0.03, 4, 0.55, seed=3), 0.004, 0.012, seed=4)
    fn2 = t.fbm(0.010, 3, 0.6, seed=5)
    cover = fn * 0.72 + fn2 * 0.28
    e = 1.6 * t.px / 0.0009                                    # edge softness ~1.6 px at any res
    scale1 = smoothstep(0.30, 0.30 + 0.012 * e, cover)         # main crust (~88 % cover)
    scale2 = smoothstep(0.585, 0.585 + 0.012 * e, cover + (t.fbm(0.02, 3, 0.5, seed=6) - 0.5) * 0.3)   # second lamina
    crust = scale1 * 0.00012 + scale2 * 0.00007
    peb = t.fbm(0.0035, 3, 0.6, seed=7)                        # pebbly scale skin
    craze = t.cracks(0.011, 0.0005, seed=8, wander=0.002, density=0.6)
    crust_tex = ((peb - 0.5) * 0.00006 - craze * 0.00004) * scale1
    bare_tex = (t.fbm(0.002, 2, 0.5, seed=9) - 0.5) * 0.00002 * (1 - scale1)

    # ---- pits: scale hammered into the surface and since fallen out; clustered
    pclu = smoothstep(0.42, 0.72, t.fbm(0.06, 3, 0.5, seed=10))
    pit_a = t.speckle(density=7000, size=0.0030, seed=13, soft=0.7) * (0.2 + 0.8 * pclu)
    pit_b = t.speckle(density=35000, size=0.0014, seed=14, soft=0.6) * (0.1 + 0.9 * pclu)
    pits = sat(pit_a + pit_b * 0.6)
    # ---- a few file / drift marks
    scr = t.scratches(density=220, length=0.035, width=0.0005, seed=15)

    height = und + hammer + crust + crust_tex + bare_tex - pits * 0.00026 - scr * 0.00003

    # ---- masks
    cav = t.cavity(height, 0.0025, 0.00018)
    hand = smoothstep(0.35, 0.8, t.fbm(0.09, 3, 0.5, seed=16))
    burn = sat(crest * hand) * (1 - pits)       # handled / rubbed crests (soft, patchy)
    bloom_n = smoothstep(0.5, 0.82, t.fbm(0.07, 4, 0.55, seed=17))
    bloom = sat(bloom_n * (0.2 + 0.8 * low) * (1 - 0.7 * scale1) + pits * 0.6 * (0.3 + 0.7 * bloom_n)
                + (1 - scale1) * 0.25 * bloom_n)
    oil = sat(low * 0.55 + cav * 0.5) * (1 - bloom)             # oil / soot / wax held in the lows

    # ---- colour (linear). Scale = blue-black magnetite; bare = dark steel; crests = burnished steel
    tone = sat(0.5 + (peb - 0.5) * 0.8 + (t.fbm(0.11, 3, 0.5, seed=18) - 0.5) * 0.7 + (fn2 - 0.5) * 0.4)
    c_scale = ramp(tone, [(0.0, "#2b2c30"), (0.45, "#383a40"), (0.75, "#43464d"), (1.0, "#505157")])
    c_scale = mix(c_scale, c_scale * hexlin("#bcc4d8") * 1.45, scale2 * 0.40)     # thicker lamina: bluer
    c_bare = ramp(t.fbm(0.02, 4, 0.6, seed=19), [(0.0, "#4f4c49"), (0.5, "#625e59"), (1.0, "#746f68")])
    albedo = mix(c_bare, c_scale, scale1)
    albedo = mix(albedo, hexlin("#171615"), sat(oil * 0.7 + craze * scale1 * 0.35))
    albedo = mix(albedo, ramp(peb, [(0, "#38231a"), (0.5, "#52331f"), (1, "#684327")]), bloom * 0.75)
    albedo = mix(albedo, hexlin("#7d7a77"), burn * 0.3)
    albedo = mix(albedo, hexlin("#a8a39c"), scr * 0.45)

    rough = 0.58 + 0.16 * (peb - 0.5) * scale1 + 0.05 * (r1 - 0.5)
    rough = mix(rough - 0.13, rough, scale1)                    # bare metal smoother than the crust
    rough = mix(rough, 0.70, oil * 0.5)
    rough = mix(rough, 0.9, bloom * 0.85)
    rough = mix(rough, 0.40, burn * 0.5)
    rough = mix(rough, 0.34, scr * 0.5)

    metal = 0.92 - 0.12 * scale1
    metal = mix(metal, 0.4, oil * 0.6)
    metal = mix(metal, 0.0, bloom * 0.95)
    metal = mix(metal, 1.0, sat(burn * 0.7 + scr * 0.5))
    return dict(albedo=albedo, height=height, rough=rough, metal=metal)
