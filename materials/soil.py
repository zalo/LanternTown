"""Dark crumbly planting soil with bits of bark (pots and planters).

Construction: a loose, lumpy bed of soil aggregates - thousands of individual crumbs and clods in
three size classes dropped onto an uneven surface, each with a granular skin - then composted bark
chips (flat, slightly curved, fibrous slivers 10-45 mm long) are scattered on top at random angles,
tilted and half-buried by the crumbs around them, then a few rootlets / straw fibres and pale sand
grains.  Dry crumb tops are a little paler than the damp crevices; bark is the only thing with any
sheen.
"""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.007, "ao_strength": 1.0}


def _dots(t, density, size, seed, soft=0.6):
    """Fast speckle: one dot per jittered grid cell (3x3 search).  0..1 mask, plus a per-dot random."""
    cell = max(1.0 / np.sqrt(density), t.px * 3)
    nx, ny = max(2, int(round(t.W / cell))), max(2, int(round(t.H / cell)))
    cw, ch = t.W / nx, t.H / ny
    rng = np.random.default_rng(seed)
    jx, jy, rr = rng.random((ny, nx)).astype(F), rng.random((ny, nx)).astype(F), rng.random((ny, nx)).astype(F)
    gx = np.floor(t.X / cw).astype(np.int64)
    gy = np.floor(t.Y / ch).astype(np.int64)
    out = np.zeros(t.X.shape, dtype=F)
    rv = np.zeros(t.X.shape, dtype=F)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = gx + ox, gy + oy
            wx, wy = cx % nx, cy % ny
            dx = (cx + jx[wy, wx]) * cw - t.X
            dy = (cy + jy[wy, wx]) * ch - t.Y
            r = size * 0.5 * (0.5 + rr[wy, wx])
            m = 1 - smoothstep(r * (1 - soft), r, np.sqrt(dx * dx + dy * dy))
            upd = m > out
            out = np.where(upd, m, out)
            rv = np.where(upd, rr[wy, wx], rv)
    return out, rv


def _drop_stones(t, H, ID, Q, stones, id0):
    """Press stones into the height field H (max-composite).  stones: rows of
    (cx, cy, a, b, ang, c, top, pw, f1x, f1y, f1h, lump, lph).  Writes the owner id into ID and the
    normalised radial position (0 centre .. 1 outline) into Q."""
    px, py, w, h = t.px, t.py, t.w, t.h
    for k in range(stones.shape[0]):
        cx, cy, a, b, ang, c, top, pw, fx, fy, fh, lump, lph = stones[k]
        rad = max(a, b) * (1.0 + lump) + px
        nxp, nyp = int(rad / px) + 1, int(rad / py) + 1
        ix, iy = int(cx / px), int((t.H - cy) / py)
        xs = np.arange(ix - nxp, ix + nxp + 1)
        ys = np.arange(iy - nyp, iy + nyp + 1)
        X = (xs[None, :] + 0.5) * px - cx
        Y = (t.H - (ys[:, None] + 0.5) * py) - cy
        ca, sa = np.cos(ang), np.sin(ang)
        xl = (X * ca + Y * sa) / a
        yl = (-X * sa + Y * ca) / b
        th = np.arctan2(yl, xl)
        rho = 1.0 + lump * (np.cos(3 * th + lph) + 0.6 * np.cos(5 * th + 2.3 * lph) + 0.5 * np.cos(2 * th + 4.1 * lph))
        q = (xl * xl + yl * yl) / (rho * rho)
        m = q < 1.0
        if not m.any():
            continue
        z = np.sqrt(np.clip(1.0 - q, 0.0, 1.0)) ** pw                 # 1 at the centre .. 0 at the outline
        z = np.minimum(z, fh + fx * xl + fy * yl)                      # fracture facet
        z = top - c + c * z
        yy, xx = ys % h, xs % w
        sub = H[np.ix_(yy, xx)]
        upd = m & (z > sub)
        if not upd.any():
            continue
        H[np.ix_(yy, xx)] = np.where(upd, z, sub)
        sid = ID[np.ix_(yy, xx)]
        ID[np.ix_(yy, xx)] = np.where(upd, id0 + k, sid)
        sq = Q[np.ix_(yy, xx)]
        Q[np.ix_(yy, xx)] = np.where(upd, q, sq)
    return id0 + stones.shape[0]


def _drop_chips(t, H, ID, CU, CV, Hs, chips, id0, rag):
    """Lay bark chips on the bed (max-composite, so higher crumbs half-bury them).  chips: rows of
    (cx, cy, a, b, ang, th, tx, ty, curv).  Writes id and chip-local coords (-1..1)."""
    px, py, w, h = t.px, t.py, t.w, t.h
    for k in range(chips.shape[0]):
        cx, cy, a, b, ang, th, tx, ty, curv = chips[k]
        rad = max(a, b) * 1.15 + px
        nxp, nyp = int(rad / px) + 1, int(rad / py) + 1
        ix, iy = int(cx / px), int((t.H - cy) / py)
        xs = np.arange(ix - nxp, ix + nxp + 1)
        ys = np.arange(iy - nyp, iy + nyp + 1)
        yy, xx = ys % h, xs % w
        X = (xs[None, :] + 0.5) * px - cx
        Y = (t.H - (ys[:, None] + 0.5) * py) - cy
        ca, sa = np.cos(ang), np.sin(ang)
        xl = (X * ca + Y * sa) / a
        yl = (-X * sa + Y * ca) / b
        q = (np.abs(xl) ** 3.2 + np.abs(yl) ** 2.4) * (1.0 + 0.9 * (rag[np.ix_(yy, xx)] - 0.5))
        m = q < 1.0
        if not m.any():
            continue
        z0 = Hs[iy % h, ix % w]
        z = z0 + tx * xl * a + ty * yl * b + th * (1.0 - curv * yl * yl) - th * 0.9 * smoothstep(0.55, 1.0, q)
        sub = H[np.ix_(yy, xx)]
        upd = m & (z > sub)
        if not upd.any():
            continue
        H[np.ix_(yy, xx)] = np.where(upd, z, sub)
        ID[np.ix_(yy, xx)] = np.where(upd, id0 + k, ID[np.ix_(yy, xx)])
        CU[np.ix_(yy, xx)] = np.where(upd, xl, CU[np.ix_(yy, xx)])
        CV[np.ix_(yy, xx)] = np.where(upd, yl, CV[np.ix_(yy, xx)])
    return id0 + chips.shape[0]


def build(t):
    rng = np.random.default_rng(606)
    A = t.W * t.H

    # ------------------------------------------------------------------ loose bed
    macro = t.fbm(0.30, 3, 0.5, seed=11)                                # drier / damper patches
    und = (t.noise(0.15, seed=12) - 0.5) * 0.008 + (t.noise(0.07, seed=13) - 0.5) * 0.006
    fine = t.fbm(0.0035, 3, 0.65, seed=14)
    bed_h = und + (t.fbm(0.02, 3, 0.55, seed=15) - 0.5) * 0.0030 + (fine - 0.5) * 0.0008

    def make(n, r_lo, r_hi, proud):
        cx, cy = rng.random(n) * t.W, rng.random(n) * t.H
        r = r_lo * (r_hi / r_lo) ** (rng.random(n) ** 1.5)
        asp = 1.0 + 0.5 * rng.random(n) ** 1.5
        a, b = r * np.sqrt(asp), r / np.sqrt(asp)
        ang = rng.random(n) * np.pi
        c = b * (0.60 + 0.35 * rng.random(n))
        base = und[np.clip(((t.H - cy) / t.py).astype(int), 0, t.h - 1), np.clip((cx / t.px).astype(int), 0, t.w - 1)]
        top = base + r * proud * (0.20 + 0.80 * rng.random(n)) - 0.0003
        pw = 0.55 + 0.45 * rng.random(n)
        z = np.zeros(n)
        lump = 0.07 + 0.11 * rng.random(n)
        return np.stack([cx, cy, a, b, ang, c, top, pw, z, z, z + 9.0, lump, rng.random(n) * 6.28], axis=1)

    H = bed_h.astype(np.float64).copy()
    ID = np.zeros(H.shape, dtype=np.int64)
    Q = np.ones(H.shape, dtype=np.float64)
    n = 1
    for g in (make(int(30000 * A), 0.0016, 0.0040, 1.00),               # crumbs 3-8 mm
              make(int(5200 * A), 0.0040, 0.0085, 1.00),                # 8-17 mm
              make(int(520 * A), 0.0085, 0.0150, 0.80)):                # clods 17-30 mm
        n = _drop_stones(t, H, ID, Q, g, n)
    n_soil = n

    # ------------------------------------------------------------------ bark chips (evenly scattered, never clumped)
    gx, gy = 11, 10
    jx = (np.arange(gx)[None, :] + rng.random((gy, gx))) * (t.W / gx)
    jy = (np.arange(gy)[:, None] + rng.random((gy, gx))) * (t.H / gy)
    nc = gx * gy
    half_l = 0.0055 + 0.0185 * rng.random(nc) ** 1.3                    # 11-48 mm long
    half_w = np.minimum(0.0026 + 0.0075 * rng.random(nc) ** 1.3, half_l * 0.66)
    chips = np.stack([jx.ravel(), jy.ravel(), half_l, half_w, rng.random(nc) * np.pi,
                      0.0014 + 0.0026 * rng.random(nc), (rng.random(nc) - 0.5) * 0.30, (rng.random(nc) - 0.5) * 0.30,
                      0.15 + 0.5 * rng.random(nc)], axis=1)
    CU = np.zeros(H.shape)
    CV = np.zeros(H.shape)
    Hs = t.blur(H.astype(F), 0.005).astype(np.float64) + 0.0006
    rag = t.fbm(0.006, 3, 0.6, seed=16).astype(np.float64)
    n = _drop_chips(t, H, ID, CU, CV, Hs, chips, n, rag)
    # a last sprinkle of loose crumbs, some landing on the bark
    sprinkle = make(int(5000 * A), 0.0012, 0.0030, 1.0)
    cxp = np.clip((sprinkle[:, 0] / t.px).astype(int), 0, t.w - 1)
    cyp = np.clip(((t.H - sprinkle[:, 1]) / t.py).astype(int), 0, t.h - 1)
    sprinkle[:, 6] = H[cyp, cxp] + sprinkle[:, 5] * (0.35 + 0.5 * rng.random(sprinkle.shape[0]))
    ID2 = np.zeros(H.shape, dtype=np.int64)
    _drop_stones(t, H, ID2, Q, sprinkle, 1)
    on_top = ID2 > 0
    ID = np.where(on_top, (ID2 % (n_soil - 1)) + 1, ID)

    H = H.astype(F)
    Q = Q.astype(F)
    bark = (ID >= n_soil).astype(F)
    crumb = ((ID > 0) & (ID < n_soil)).astype(F)
    cu, cv = CU.astype(F), CV.astype(F)

    pr = np.random.default_rng(99)
    tab = pr.random((n + 1, 5)).astype(F)
    tab[0] = 0.5
    k_tone, k_hue, k_dry, k_a, k_b = [tab[ID, i] for i in range(5)]

    # granular skin on the aggregates; fibrous furrows along the bark
    gran = t.fbm(0.0050, 4, 0.65, seed=21)
    caul = t.fbm(0.013, 3, 0.55, seed=22)
    furrow = 0.5 + 0.5 * np.sin(cv * (5.0 + 7.0 * k_a) + k_b * 6.28 + (t.noise(0.012, seed=23) - 0.5) * 5.0 + cu * 0.8)
    furrow = furrow * (0.6 + 0.4 * t.fbm(0.004, 2, 0.6, seed=24))
    height = H + (1 - bark) * ((gran - 0.5) * 0.0013 + (caul - 0.5) * 0.0014) + bark * (furrow - 0.5) * 0.00045
    # rootlets / straw fibres lying on the surface
    fib = t.scratches(density=420, length=0.022, width=0.0007, seed=25)
    fib = fib * (1 - bark)
    height = height + fib * 0.0005

    # ------------------------------------------------------------------ colour
    cav = t.cavity(height, 0.005, 0.0015)
    high = t.edges(height, 0.006, 0.0018)
    dry = sat(0.22 + 0.28 * (macro - 0.4) + 0.55 * high - 0.6 * cav + 0.40 * (k_dry - 0.5) * crumb)
    soilc = ramp(dry, [(0.0, "#2a211a"), (0.35, "#3a2e24"), (0.7, "#4b3c2f"), (1.0, "#5c4b3b")])
    soilc = soilc * (0.80 + 0.40 * gran)[..., None] * (0.90 + 0.20 * k_tone * crumb + 0.10 * (1 - crumb))[..., None]
    # pale mineral grains (sharp sand in the mix) and specks of lighter organic matter
    sandg, sand_r = _dots(t, 42000, 0.0019, 26, 0.6)
    soilc = mix(soilc, ramp(sand_r, [(0, "#5e5347"), (0.6, "#84786a"), (1, "#a59b8c")]), (sandg * (0.25 + 0.5 * sand_r))[..., None])
    org, org_r = _dots(t, 16000, 0.0032, 27, 0.7)
    soilc = mix(soilc, ramp(org_r, [(0, "#5a4532"), (1, "#7a6148")]), (org * 0.45 * (org_r > 0.35))[..., None])
    soilc = mix(soilc, hexlin("#8a775a"), (fib * 0.75)[..., None])

    barkc = ramp(k_hue, [(0.00, "#4d3929"), (0.35, "#5f4532"), (0.60, "#6c4e37"), (0.80, "#765841"),
                         (0.93, "#876a4d"), (1.00, "#544338")])
    barkc = barkc * (0.80 + 0.36 * k_tone)[..., None]
    barkc = barkc * (0.72 + 0.50 * furrow)[..., None]                         # dark furrows, paler fibre ridges
    edge_c = smoothstep(0.45, 1.0, np.abs(cu) ** 3.2 + np.abs(cv) ** 2.4)
    barkc = mix(barkc, hexlin("#3a2f26"), (edge_c * 0.55 + smoothstep(0.55, 0.85, t.fbm(0.012, 3, 0.55, seed=28)) * 0.45)[..., None])   # soil smeared on
    albedo = mix(soilc, barkc, bark[..., None])
    albedo = albedo * (1 - 0.35 * cav)[..., None]                             # damp, shaded crevices
    albedo = albedo * (0.97 + 0.06 * macro)[..., None]

    # ------------------------------------------------------------------ roughness
    damp = smoothstep(0.55, 0.15, dry)                                         # moist, dark soil in the hollows has a faint sheen
    r_soil = 0.97 - 0.17 * damp + 0.04 * (gran - 0.5) - 0.12 * sandg
    r_bark = 0.78 - 0.14 * furrow + 0.14 * edge_c + 0.10 * (k_dry - 0.5)
    rough = mix(r_soil, r_bark, bark)
    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0)
