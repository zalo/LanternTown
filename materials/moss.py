"""Dense cushion moss: pillow-shaped colonies packed shoulder to shoulder, each one a velvet of
tiny upright shoot tips (little star rosettes 2-4 mm across). Deep green down in the creases
where it stays wet, yellow-green on the sun-side crowns, a few cushions dried olive-brown with a
rusty haze of spore capsules, crumbs of litter caught in the hollows."""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.012, "ao_strength": 0.8}


def _rosettes(t, size, seed, arms=5):
    """Shoot tips: every Voronoi cell is one shoot seen from above - a small dome whose leaves
    radiate from the centre. Returns (bump 0..1, per-shoot random, centre mask)."""
    lay = t.cells(size, 1.0, seed=seed)
    r = lay.f1 / (lay.f1 + lay.dist + 1e-9)                # 0 at the shoot centre, 1 at its rim
    ang = np.arctan2(lay.v, lay.u)
    star = 0.5 + 0.5 * np.cos(ang * arms + lay.rand(5) * 6.283)
    bump = (1 - r ** 1.6) * (0.72 + 0.28 * star * smoothstep(0.15, 0.7, r))
    bump = bump * (0.55 + 0.45 * lay.rand(1))             # shoots are not all equally tall
    return bump.astype(F), lay.rand(2), (1 - smoothstep(0.0, 0.45, r))


def _cushions(t, rng, n, r_rng, k=380.0):
    """Soft union of `n` dome-shaped colonies (spherical-ish caps). Returns (height m, blended
    per-cushion random a, random b)."""
    E = np.ones((t.h, t.w), dtype=np.float64)
    A = np.full((t.h, t.w), 0.5, dtype=np.float64)
    B = np.full((t.h, t.w), 0.5, dtype=np.float64)
    g = int(np.ceil(np.sqrt(n)))
    cells = rng.permutation(g * g)[:n]
    for i in range(n):
        cx = ((cells[i] % g) + rng.random()) / g * t.W
        cy = ((cells[i] // g) + rng.random()) / g * t.H
        R = rng.uniform(*r_rng) * (0.75 + 0.5 * rng.random())
        H = R * rng.uniform(0.32, 0.46)
        ex, rot = rng.uniform(0.75, 1.3), rng.random() * 3.1416
        ra, rb = rng.random(), rng.random()
        hw = int(R * 1.35 / t.px) + 2
        ix, iy = int(cx / t.px), int((t.H - cy) / t.py)
        xs, ys = np.arange(ix - hw, ix + hw + 1), np.arange(iy - hw, iy + hw + 1)
        X = (xs[None, :] + 0.5) * t.px - cx
        Y = (t.H - (ys[:, None] + 0.5) * t.py) - cy
        c, sn = math.cos(rot), math.sin(rot)
        u, v = (X * c + Y * sn) * ex, (-X * sn + Y * c) / ex
        q = (u * u + v * v) / (R * R)
        h = H * np.clip(1.0 - q, 0.0, 1.0) ** 0.72
        w = np.exp(k * h) - 1.0
        sel = np.ix_(ys % t.h, xs % t.w)
        E[sel] += w
        A[sel] += w * ra
        B[sel] += w * rb
    return (np.log(E) / k).astype(F), (A / E).astype(F), (B / E).astype(F)


def build(t):
    rng = np.random.default_rng(90210)
    # ---- macro: cushions = soft union of domed colonies of three sizes, on a thin moss mat
    big, ra1, rb1 = _cushions(t, rng, 40, (0.032, 0.056))
    mid, ra2, rb2 = _cushions(t, rng, 190, (0.014, 0.027))
    k = 380.0
    macro = np.log(np.exp(k * big) + np.exp(k * mid * 0.9)) / k
    wsel = sat((big - mid * 0.9) * 400 + 0.5)
    crand = mix(ra2, ra1, wsel)
    crand2 = mix(rb2, rb1, wsel)
    macro = t.warp(macro, 0.0028, 0.02, seed=2)                         # colonies are never true circles
    crand = t.warp(crand, 0.0028, 0.02, seed=2)
    crand2 = t.warp(crand2, 0.0028, 0.02, seed=2)
    lumps = (t.fbm(0.018, 3, 0.5, seed=6) - 0.5) * 0.0032              # daughter lobes on every cushion
    swell = (t.fbm(0.16, 3, 0.5, seed=7) - 0.5) * 0.005
    macro = t.blur(macro, 0.0011) + lumps + swell
    hi = smoothstep(0.002, 0.020, macro)                               # crowns of the cushions
    lowm = t.blur(macro, 0.006)
    crease = sat((lowm - macro) / 0.003)                               # seams where colonies meet

    # ---- meso / micro: shoot tips. two moss habits: fine velvet and a coarser starry one
    b1, r1, c1 = _rosettes(t, 0.0030, 10, arms=5)
    b2, r2, c2 = _rosettes(t, 0.0052, 11, arms=6)
    coarse = smoothstep(0.6, 0.72, crand2 + (t.fbm(0.05, 2, seed=12) - 0.5) * 0.2)
    bump = mix(b1, b2, coarse)
    srand = mix(r1, r2, coarse)
    centre = mix(c1, c2, coarse)
    tip_h = mix(0.0011, 0.0021, coarse)
    fuzz = (t.fbm(0.0018, 2, 0.6, seed=13) - 0.5) * 0.00025
    height = macro + bump * tip_h + fuzz

    # ---- litter: crumbs of bark / grit / dead leaf caught on the moss, mostly in the hollows
    low = 1 - smoothstep(0.004, 0.013, t.blur(macro, 0.004))
    lit = t.speckle(density=700, size=0.0036, seed=14, soft=0.35)
    lit = lit * smoothstep(0.45, 0.6, t.fbm(0.03, 3, seed=15) * 0.6 + low * 0.4)
    height = height + lit * 0.0009

    # ---- masks
    cav = t.cavity(height, 0.0016, 0.0005)                             # between individual shoots
    damp = sat(low * 0.8 + crease * 0.6)

    # ---- colour
    patch = t.fbm(0.22, 3, 0.5, seed=20)
    mott = t.fbm(0.02, 3, 0.55, seed=21)
    tone = sat(0.20 + 0.52 * hi + 0.34 * (crand - 0.5) + 0.26 * (patch - 0.5) + 0.16 * (mott - 0.5)
               + 0.16 * (srand - 0.5) + 0.16 * bump - 0.18 * crease)
    albedo = ramp(tone, [(0.0, "#222f18"), (0.25, "#32461c"), (0.5, "#4a5e24"), (0.75, "#66762d"), (1.0, "#878f3e")])
    # shoot tips are the newest, palest growth; between shoots you look down into old dark stems
    albedo = mix(albedo, albedo * hexlin("#fff4b0") * 1.25, centre * 0.30 * (0.4 + 0.6 * hi))
    albedo = mix(albedo, albedo * hexlin("#5a5a34"), sat(cav) * 0.7)
    # some cushions are dry: olive-brown, with a rusty haze of capsules on a few of them
    dry = smoothstep(0.60, 0.74, (crand * 7.31) % 1.0 * 0.6 + 0.4 * t.fbm(0.07, 3, seed=22)) * hi
    albedo = mix(albedo, luminance(albedo)[..., None] * hexlin("#b89a52") * 1.9, dry * 0.42)
    caps = t.speckle(density=42000, size=0.0016, seed=23, soft=0.4) * dry
    caps = caps + t.speckle(density=5000, size=0.0015, seed=24, soft=0.4) * hi * 0.6
    albedo = mix(albedo, hexlin("#6b3a1c"), sat(caps) * 0.7)
    litc = ramp(t.white(seed=25) * 0.0 + t.fbm(0.004, 2, seed=26), [(0.0, "#2a2018"), (0.5, "#4a3a28"), (1.0, "#6b5a44")])
    albedo = mix(albedo, litc, sat(lit) * 0.85)

    # ---- roughness: moss is matte velvet; wet hollows carry a little sheen, dry cushions none
    rough = 0.90 - 0.20 * damp + 0.05 * (mott - 0.5) + 0.06 * dry - 0.05 * centre
    rough = np.clip(rough, 0.6, 0.98)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
