"""Weathered copper roofing sheet (plain, no seams): brown oxide turning to verdigris.

Built in the order the real thing gets its surface:
  1. smooth rolled sheet dressed over boarding: broad oil-canning, a few very shallow dresser
     dishes and soft dents - nothing sharp, no pebbling;
  2. brown cuprite oxide over everything (dull dielectric skin, satin only in rain-washed lanes),
     blackened to tarnish in its own soft clouds;
  3. patina = how long each spot stays wet, summed from separate layers of different size:
     isotropic warped blotches (0.4 m -> 25 mm), eight individual run-offs that start at a drip point
     and widen / feather / break into strands down the slope (-V), washed channels that stay brown,
     drip spots with short tails, tide rings and dew held in the hollows.
     Wetness is rank-transformed (exact coverage) and read through one continuous ramp:
     brown -> olive bloom -> dusty grey-green film -> verdigris crust -> chalky carbonate. The film
     feathers over centimetres with a clumped grainy edge; only the crust has a defined (lobed) edge.
     The field is equalised so every 0.3 m patch and every column / row carries about the same
     luminance (the engine adds world-space variation; a tile must not carry its own);
  4. relief follows the patina: the crust stands 0.2-0.35 mm proud with a thicker tide-line rim and
     a pitted granular top, the film is faintly grainy, the oxide is smooth;
  5. no bare metal - the engine's wear mask supplies bright copper where it belongs.
"""
import math
from tk.tex import *

META = {"normal_strength": 4.5}

# luminance of each patina state (linear) - used while equalising the field
L_BROWN, L_TAR, L_FILM, L_VERD, L_PALE = 0.052, 0.032, 0.125, 0.27, 0.38


def zs(a):
    a = a - float(a.mean())
    return (a / max(float(a.std()), 1e-9)).astype(F)


def ovals(t, spacing, r_lo, r_hi, seed, aspect=1.6, plateau=0.35, keep=1.0, nb=1, ring=0.0, small=1.0):
    """Oval marks on a jittered grid (random orientation / size / strength); max-combined so overlaps
    do not pile up. `small` > 1 skews the sizes towards r_lo (many small, few large); `ring` > 0
    hollows the centre (tide rings). r_hi must stay below nb * spacing."""
    rng = np.random.default_rng(t.seed * 1000 + 7000 + seed)
    nx, ny = max(3, round(t.W / spacing)), max(3, round(t.H / spacing))
    cw, ch = t.W / nx, t.H / ny
    sh = (ny, nx)
    px, py = rng.random(sh).astype(F), rng.random(sh).astype(F)
    an = (rng.random(sh) * math.pi).astype(F)
    rr = (r_lo * (r_hi / r_lo) ** (rng.random(sh) ** small)).astype(F)
    asp = (1.0 + (aspect - 1.0) * rng.random(sh)).astype(F)
    amp = ((0.55 + 0.45 * rng.random(sh)) * (rng.random(sh) < keep)).astype(F)
    gx, gy = t.X / cw, t.Y / ch
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = (gx - ix).astype(F), (gy - iy).astype(F)
    acc = t.zeros()
    for oy in range(-nb, nb + 1):
        for ox in range(-nb, nb + 1):
            jx, jy = (ix + ox) % nx, (iy + oy) % ny
            dx = (fx - ox - px[jy, jx]) * cw
            dy = (fy - oy - py[jy, jx]) * ch
            ca, sa = np.cos(an[jy, jx]), np.sin(an[jy, jx])
            a, b = dx * ca + dy * sa, (-dx * sa + dy * ca) * asp[jy, jx]
            d = np.sqrt(a * a + b * b) / rr[jy, jx]
            prof = 1 - smoothstep(plateau, 1.0, d)
            if ring > 0:
                prof = prof * (1 - ring * (1 - smoothstep(0.0, 0.75, d)))
            acc = np.maximum(acc, amp[jy, jx] * prof)
    return acc.astype(F)


def scatter(rng, t, n, sep):
    """n points kept at least `sep` apart (wrapping): even density, no grid."""
    pts, tries = [], 0
    while len(pts) < n and tries < 8000:
        tries += 1
        x, y = rng.random() * t.W, rng.random() * t.H
        if all(math.hypot((x - qx + t.W * 0.5) % t.W - t.W * 0.5, (y - qy + t.H * 0.5) % t.H - t.H * 0.5) >= sep
               for qx, qy in pts):
            pts.append((x, y))
    return pts


def local(t, cx, cy, ang=0.0):
    """Wrapped coordinates (along, across) in metres about a point, rotated by ang."""
    dx = (t.X - cx + t.W * 0.5) % t.W - t.W * 0.5
    dy = (t.Y - cy + t.H * 0.5) % t.H - t.H * 0.5
    ca, sa = math.cos(ang), math.sin(ang)
    return dx * ca + dy * sa, -dx * sa + dy * ca


def dents(t, seed, n=9, r_lo=0.035, r_hi=0.06, sep=0.27):
    """Broad soft dishes (handling / hail / ladder knocks), scattered evenly. 0..1 profile."""
    rng = np.random.default_rng(t.seed * 1000 + 5000 + seed)
    out = t.zeros()
    for (x, y) in scatter(rng, t, n, sep):
        a, b = local(t, x, y, rng.random() * math.pi)
        r, asp = r_lo + (r_hi - r_lo) * rng.random(), 1.0 + 0.5 * rng.random()
        d = np.sqrt(a * a + (b * asp) ** 2) / r
        out = np.maximum(out, (0.6 + 0.4 * rng.random()) * np.exp(-2.2 * d * d))
    return out.astype(F)


def runoffs(t, seed, n=9, w_lo=0.006, w_hi=0.013, grow=0.018, l_lo=0.3, l_hi=0.95):
    """Individual run-offs: each starts at a drip point, runs down the slope (-V) widening and fading,
    with a wandering, slightly leaning centre line (so no column of the tile is owned by one run).
    Stratified in X with jitter, heads spread in Y. Returns 0..1 (soft)."""
    rng = np.random.default_rng(t.seed * 1000 + 4000 + seed)
    out = t.zeros()
    ys = rng.permutation(n)
    for i in range(n):
        cx = (i + 0.1 + 0.8 * rng.random()) * t.W / n
        y0 = (ys[i] + rng.random()) * t.H / n
        L = l_lo + (l_hi - l_lo) * rng.random() ** 1.2
        w0 = w_lo + (w_hi - w_lo) * rng.random()                           # half width at the head
        w1 = w0 + grow * (0.4 + rng.random()) * (L / 0.6)                  # ... at the foot
        d = (y0 - t.Y) % t.H                                               # metres below the head
        s = d / L
        lean = (rng.random() - 0.5) * 0.16                                 # +-4.5 deg
        k1, k2 = 4.0 + 5.0 * rng.random(), 13.0 + 9.0 * rng.random()
        xc = cx + lean * d + 0.014 * (0.4 + 0.6 * rng.random()) * np.sin(k1 * d + rng.random() * 6.28) \
            + 0.003 * np.sin(k2 * d + rng.random() * 6.28)
        dx = (t.X - xc + t.W * 0.5) % t.W - t.W * 0.5
        w = w0 + (w1 - w0) * np.clip(s, 0, 1) ** 0.8
        env = smoothstep(0.0, 0.04, s) * (1 - smoothstep(0.55, 1.0, s)) * (s < 1.0)
        body = env * np.exp(-(np.abs(dx) / w) ** 1.6)
        a, b = local(t, cx, y0 - 0.02 * L)
        head = 0.9 * np.exp(-((a / (w0 * 2.4)) ** 2 + (b / (w0 * 3.0)) ** 2))
        out = np.maximum(out, (0.6 + 0.4 * rng.random()) * np.maximum(body, 0.5 * head))
    return out.astype(F)


def trail(t, a, length):
    """Drag a mask down the slope (-V): one-sided exponential smear, `length` metres (wraps)."""
    k = np.exp(-(np.arange(t.h) * t.py / length) ** 1.5)
    k = k / k.sum()
    return np.fft.ifft(np.fft.fft(a, axis=0) * np.fft.fft(k)[:, None], axis=0).real.astype(F)


def build(t):
    # --- 1. the sheet: smooth rolled metal, soft forms only -------------------------------------------
    oilcan = t.fbm(0.45, 2, 0.4, seed=1)                                   # buckles between fixings
    oil2 = t.fbm(0.14, 2, 0.4, seed=2)
    dish = t.blur(ovals(t, 0.10, 0.020, 0.034, 2, aspect=1.5, plateau=0.0, keep=0.65), 0.006)   # dresser marks
    dent = t.warp(dents(t, 4), 0.006, 0.06, seed=5)
    form = (oilcan - 0.5) * 0.0024 + (oil2 - 0.5) * 0.0005 - dish * 0.00032 - dent * 0.0007
    low = sat(dent * 1.1 + dish * 0.7 + sat(-t.curv(form, 0.04) / 0.0005) * 0.5)    # hollows hold the dew
    lowb = t.blur(low, 0.006)

    def streak(src, length, spread=0.0):
        """What a wet source leaves below it: a sharp-headed smear fading down the slope."""
        a = trail(t, src, length)
        if spread > 0:
            a = a + 0.7 * t.blur(trail(t, src, length * 2.2), spread, 0.0)
        return (a / max(float(np.percentile(a, 99.7)), 1e-6)).astype(F)

    # --- 3. the patina field ("time wet"), layer by layer ---------------------------------------------
    qs = np.linspace(0, 100, 101)

    def rank(a):
        return np.interp(a, np.percentile(a, qs), qs / 100.0).astype(F)

    # (a) blotches: isotropic, every size from 0.4 m down to 12 mm, domain-warped so nothing is pointed
    blot = 0.25 * t.noise(0.40, seed=30) + 1.0 * t.noise(0.20, seed=31) + 0.7 * t.noise(0.10, seed=32) \
        + 0.4 * t.noise(0.05, seed=33, stretch=(1.0, 1.8)) + 0.2 * t.noise(0.025, seed=34)
    blot = zs(t.warp(blot, 0.015, 0.2, seed=41))
    # (b) run-offs: a handful of individual streaks from drip points, widening and feathering downslope,
    #     their feet breaking into strands; and rain-washed channels that stay brown
    vfil = t.fbm(0.010, 2, 0.5, seed=38, stretch=(1.0, 16.0))              # strands 5-10 mm wide
    vf = zs(vfil)
    runs = runoffs(t, 1, n=8, l_lo=0.4, l_hi=1.0) * (0.62 + 0.38 * smoothstep(0.25, 0.7, vfil))
    chan = runoffs(t, 2, n=6, w_lo=0.005, w_hi=0.010, grow=0.012, l_lo=0.3, l_hi=0.7)
    # (c) spots where drops sit and dry (5-20 mm), each with a short tail down the slope; tide rings
    spots = ovals(t, 0.085, 0.0035, 0.010, 50, aspect=1.3, plateau=0.45, keep=0.7, small=1.5)
    spots = np.maximum(spots, 0.8 * t.blur(streak(spots, 0.03), 0.0015, 0.0))
    rings = ovals(t, 0.15, 0.008, 0.016, 51, aspect=1.25, plateau=0.6, keep=0.6, ring=0.85)
    con = 0.25 + 1.05 * smoothstep(0.3, 0.7, t.fbm(0.3, 2, 0.4, seed=44))   # calm passages vs. busy ones
    base = con * blot + 2.5 * runs - 1.1 * chan + 1.2 * spots + 1.0 * rings + 0.6 * lowb

    lobes = zs(t.fbm(0.06, 3, 0.55, seed=36))                              # crust lobes 60 -> 15 mm
    clump = zs(t.fbm(0.009, 2, 0.5, seed=37))                              # clumped grains 4-9 mm
    mott = zs(t.fbm(0.05, 3, 0.5, seed=39))
    cdev = zs(t.fbm(0.09, 3, 0.5, seed=48))                                # where the crust has taken hold
    tfield = zs(t.fbm(0.12, 3, 0.5, seed=49))                              # where the oxide has blackened

    def states(w):
        """Nothing here has a plateau: the film thickens steadily with wetness; the oxide blackens in its
        own soft clouds; only the crust (grains and lobes that coalesce where it stays wet longest) has an edge."""
        u = rank(w)                                                        # 0 driest .. 1 wettest
        tar = smoothstep(0.25, 0.8, rank(0.45 * zs(w) + tfield))
        film = smoothstep(0.20, 0.68, u + 0.035 * lobes + 0.03 * clump + 0.015 * vf) ** 1.2
        v = rank(u + 0.15 * lobes + 0.03 * clump + 0.06 * cdev + 0.02 * vf + 0.2 * runs + 0.1 * spots)
        verd = smoothstep(0.775, 0.855, v)
        pale = smoothstep(0.90, 1.0, v + 0.03 * mott) * verd               # oldest: chalky
        return tar, film, verd, pale

    # equalise: every 0.3 m patch and every column / row gets about the same luminance
    cmask = 0.35 + 1.3 * smoothstep(-0.8, 0.8, zs(t.fbm(0.09, 3, 0.5, seed=47)))   # breaks column corrections up
    cmask = cmask / (cmask.mean(axis=0, keepdims=True))
    off = t.zeros()
    for _ in range(7):
        tar, film, verd, pale = states(base + off)
        lum = mix(mix(mix(mix(L_BROWN, L_TAR, tar * 0.85), L_FILM, film), L_VERD, verd), L_PALE, pale * 0.6)
        rel = lum / float(lum.mean()) - 1.0
        col = t.blur(np.broadcast_to(rel.mean(axis=0, keepdims=True), rel.shape).astype(F), 0.012, 0.0)
        row = t.blur(np.broadcast_to(rel.mean(axis=1, keepdims=True), rel.shape).astype(F), 0.0, 0.01)
        off = off - 0.75 * t.blur(rel, 0.14) - 0.6 * col * cmask - 0.4 * row
    w = base + off
    tar, film, verd, pale = states(w)
    film = np.maximum(film, verd)
    thick = sat(t.blur(verd, 0.006) * 1.3 - 0.3)                           # interior of the crust
    rim = sat(verd - t.blur(verd, 0.004)) * 2.2                            # tide line: thickest at the edge

    # --- 4. relief follows the patina -----------------------------------------------------------------
    crust = t.fbm(0.009, 3, 0.55, seed=15)                                 # granular top 3-6 mm
    pits = t.speckle(density=3200, size=0.005, seed=16, soft=0.8)
    fgrain = t.fbm(0.005, 2, 0.5, seed=12)
    roll = t.fbm(0.004, 2, 0.5, seed=13, stretch=(1.0, 14.0))              # mill finish, barely there
    h = form + (roll - 0.5) * 0.000012 + (fgrain - 0.5) * 0.000018
    h = h + t.blur(verd, 0.0016) * (0.00022 + 0.00014 * thick) + t.blur(rim, 0.0015) * 0.00008 \
        + verd * ((crust - 0.5) * 0.00022 - pits * 0.00010) \
        + film * (1 - verd) * ((fgrain - 0.5) * 0.00007 + 0.00005)

    # --- 2. brown oxide ground -----------------------------------------------------------------------
    bn = t.fbm(0.07, 4, 0.5, seed=18)
    bfine = t.fbm(0.008, 2, 0.5, seed=17)
    lanes = t.fbm(0.09, 3, 0.5, seed=19, stretch=(1.0, 3.5))               # rain-washed lanes vs dew-etched areas
    bt = sat(0.5 + 0.9 * (bn - 0.5) + 0.3 * (bfine - 0.5) + 0.5 * (lanes - 0.5))
    brown = ramp(bt, [(0.0, "#40362e"), (0.3, "#473b31"), (0.55, "#4e4135"), (0.8, "#554638"), (1.0, "#5c4c3c")])
    cast = t.fbm(0.13, 3, 0.5, seed=20)                                    # faint tarnish casts: plum / olive
    brown = mix(brown, brown * hexlin("#b9a3b0") * 2.1, smoothstep(0.62, 0.9, cast) * 0.22)
    brown = mix(brown, brown * hexlin("#a8a98a") * 2.0, smoothstep(0.38, 0.1, cast) * 0.28)
    brown = mix(brown, brown * 0.86, lowb * 0.5)                           # dirt settles in the hollows

    tarn = ramp(sat(0.5 + 1.2 * (bfine - 0.5) + 0.5 * (bn - 0.5)), [(0.0, "#2d2924"), (1.0, "#3a352d")])
    dusty = ramp(sat(0.15 + 0.6 * film + 0.07 * clump + 0.05 * vf + 0.10 * mott + 0.08 * cdev),   # thin: olive; thick: grey-green
                 [(0.0, "#3b3d33"), (0.3, "#434c41"), (0.6, "#4a5f53"), (1.0, "#537061")])
    hue = t.fbm(0.11, 3, 0.5, seed=22)                                     # brochantite green <-> bluer
    green = ramp(sat(0.42 + 0.2 * thick + 0.14 * mott + 0.5 * (crust - 0.5) + 0.08 * vf),
                 [(0.0, "#4d7c6d"), (0.5, "#5f9785"), (1.0, "#72a592")])
    green = green * mix(hexlin("#f2ffe6"), hexlin("#e4f6ff"), smoothstep(0.3, 0.7, hue))
    chalk = ramp(sat(0.5 + 0.2 * clump + 0.5 * (crust - 0.5)), [(0.0, "#7fa898"), (1.0, "#99baab")])
    green = mix(green, chalk, pale * 0.65)
    green = mix(green, hexlin("#3f5347"), pits * 0.45)                     # pinholes down to the dark oxide
    green = mix(green, green * 1.12, sat(rim) * 0.5)                       # paler tide line

    albedo = mix(brown, tarn, tar * 0.88)
    albedo = mix(albedo, dusty, sat(film * (0.92 + 0.05 * clump)))
    albedo = mix(albedo, green, verd)

    # --- PBR ------------------------------------------------------------------------------------------
    rough = mix(0.40, 0.57, smoothstep(0.3, 0.7, lanes)) + 0.05 * (bfine - 0.5) + 0.05 * (bn - 0.5) + 0.05 * lowb
    rough = mix(rough, 0.61 + 0.06 * (bfine - 0.5), tar)
    rough = mix(rough, 0.75 + 0.08 * (fgrain - 0.5), film)
    rough = mix(rough, 0.88 + 0.07 * (crust - 0.5) + 0.03 * pale, verd)
    metal = 0.0

    ao = t.ao(h - form, radius=0.003, strength=1.6)                        # pits and the foot of the crust edge
    return dict(albedo=albedo, height=h, rough=sat(rough), metal=metal, ao=ao)
