"""Hand-trowelled lime plaster under several coats of limewash.

Built the way it is made:
  1. the wall behind (rubble / lath) is gently out of flat;
  2. the finish coat is laid on and closed with a steel trowel in overlapping arc sweeps.  Each
     sweep leaves a flat fan at its own level, canted slightly across the blade, with a raised lip
     along the edge that dug in and a feathered trailing edge; the steel drags sand in concentric
     arcs and burnishes the surface more or less hard (-> gloss and tone vary sweep by sweep);
  3. sand grains stand in the surface; pin-holes;
  4. limewash is brushed on in ~110 mm swathes whose direction wanders; a few blebs and short runs;
  5. age: a few long tapering shrinkage cracks with one edge lifted, two or three patches of fine
     map-crazing in the wash skin, and one or two later repairs standing a millimetre proud with a
     feathered rim, a touch cleaner and more open in texture.
"""
from tk.tex import *
import math


def aniso(t, scale, angle, stretch, seed, bandwidth=0.6, octaves=1, gain=0.5):
    """Band-limited noise 0..1 (std 0.18), features `scale` m across and `stretch` x longer along
    `angle` degrees.  Built in the Fourier domain so it wraps at any angle."""
    rng = np.random.default_rng(t.seed * 1000 + 9000 + seed)
    wn = np.fft.fft2(rng.standard_normal((t.h, t.w)))
    FX, FY = np.meshgrid(np.fft.fftfreq(t.w, d=t.px), np.fft.fftfreq(t.h, d=t.py))
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    fu = FX * c - FY * s
    fv = FX * s + FY * c
    f0 = np.sqrt((fu * stretch) ** 2 + fv ** 2)
    filt = np.zeros_like(f0)
    amp, sc = 1.0, scale
    for _ in range(octaves):
        with np.errstate(divide='ignore'):
            lf = np.where(f0 > 0, np.log2(np.maximum(f0 * sc, 1e-12) / 0.5), -99)
        filt += amp * np.exp(-0.5 * (lf / bandwidth) ** 2)
        amp *= gain
        sc *= 0.5
    filt[0, 0] = 0
    out = np.fft.ifft2(wn * filt).real
    return np.clip(0.5 + out / max(out.std(), 1e-12) * 0.18, 0, 1).astype(F)


def _tab(rng, n, k=3):
    """Smooth periodic random table (std ~1)."""
    a = rng.standard_normal(n)
    ker = np.ones(k) / k
    a = np.convolve(np.concatenate([a, a, a]), ker, mode='same')[n:2 * n]
    return a / max(a.std(), 1e-9)


def trowel(t, n, seed):
    """Painter's-algorithm trowel sweeps.  Returns (height m, burnish 0..1, tone -1..1, lip 0..1):
    burnish is how hard the steel closed the surface, tone a per-sweep random value (how wet the
    coat was when it was closed -> how the limewash took), lip the raised blade-edge lines."""
    rng = np.random.default_rng(t.seed * 1000 + 4000 + seed)
    h = t.zeros()
    burn = t.zeros(0.5)
    tone = t.zeros()
    lipm = t.zeros()
    for i in range(n):
        cx, cy = rng.random() * t.W, rng.random() * t.H
        r = rng.random(12)
        R = 0.30 + 0.55 * r[0]                         # forearm / shoulder radius of the sweep
        Lb = 0.16 + 0.10 * r[1]                        # blade contact length
        arc = 0.20 + 0.36 * r[2]                       # length of the sweep along its path
        th = arc / R
        ang = (r[3] - 0.5) * math.radians(150) + (math.pi * 0.5 if r[4] < 0.75 else -math.pi * 0.5)
        pxv, pyv = cx - R * math.cos(ang), cy - R * math.sin(ang)       # pivot
        half = 0.5 * arc + 0.5 * Lb + 0.03
        hw = int(half / t.px) + 2
        ix, iy = int(cx / t.px), int((t.H - cy) / t.py)
        xs = np.arange(ix - hw, ix + hw + 1)
        ys = np.arange(iy - hw, iy + hw + 1)
        dx = ((xs[None, :] + 0.5) * t.px - pxv).astype(F)
        dy = ((t.H - (ys[:, None] + 0.5) * t.py) - pyv).astype(F)
        rr = np.sqrt(dx * dx + dy * dy)
        phi = np.arctan2(dy, dx) - ang
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        s = (rr - R) / (0.5 * Lb)                      # -1..1 across the blade
        a = phi / (0.5 * th)                           # -1..1 along the sweep
        tilt = (1.0 if r[6] < 0.5 else -1.0) * (0.0007 + 0.0019 * r[5])                   # blade canted onto one edge (m at the edge)
        e = 1.0 if tilt < 0 else -1.0                  # the edge that digs in (low side)
        se = s * e                                     # +1 at the digging edge, -1 at the trailing edge
        # the digging edge cuts a crisp boundary, the trailing edge feathers out; ends lift off
        m = (1 - smoothstep(0.74, 1.0, se)) * smoothstep(-1.0, -0.62, se) * (1 - smoothstep(0.5, 1.0, np.abs(a)))
        # sand grains dragged under the blade: fine concentric scores, constant along the sweep
        score = np.interp(np.clip(s * 0.5 + 0.5, 0, 1) * 127, np.arange(128), _tab(rng, 128)).astype(F)
        level = (r[7] - 0.5) * 0.0024
        target = level + s * tilt + score * 0.00009
        sel = np.ix_(ys % t.h, xs % t.w)
        cur = h[sel]
        cur = cur + (target - cur) * m
        # the lip: plaster squeezed out past the digging edge, ragged, fading as the trowel lifts
        if r[10] < 0.50:
            rag = np.interp((np.clip(a * 0.5 + 0.5, 0, 1) * 63), np.arange(64), _tab(rng, 64, 5)).astype(F)
            dist = (se - 0.93) * 0.5 * Lb
            lh = (0.00035 + 0.00065 * r[8] ** 1.5) * np.clip(0.65 + 0.5 * rag, 0.0, 1.5)
            lip = np.exp(-(dist / np.where(dist < 0, 0.012, 0.0065)) ** 2) * (1 - smoothstep(0.30, 0.92, np.abs(a))) * lh
            cur = cur + lip
            l0 = lipm[sel]
            lipm[sel] = np.maximum(l0 * (1 - m), sat(lip / 0.0006))
        else:
            lipm[sel] = lipm[sel] * (1 - m)
        h[sel] = cur
        b = burn[sel]
        burn[sel] = b + ((0.15 + 0.85 * r[9]) + score * 0.05 - b) * m
        tn = tone[sel]
        tone[sel] = tn + ((r[11] - 0.5) * 2 - tn) * m
    return h, sat(burn), tone, lipm


def contour_cracks(t, scale, seed, levels=((0.5, 0.20, 0.0015), (0.34, 0.10, 0.0009)), wmin=0.0004):
    """Long wandering shrinkage cracks: level lines of a smooth low-frequency noise (so they never
    form a tidy polygon net), gated so they start and stop, tapering to nothing at both ends.
    levels = ((level, fraction kept, max width m), ...).  Returns (crack 0..1, lifted-bank 0..1)."""
    n = t.fbm(scale, 2, 0.35, seed=seed)                            # smooth: long open level lines
    n = t.warp(n, 0.006, 0.05, seed=seed + 1)                       # wander
    n = t.warp(n, 0.0015, 0.008, seed=seed + 4)                     # jagged, grain-scale kinks
    gx, gy = t.slope(n)
    g = np.sqrt(gx * gx + gy * gy) + 1e-6
    wv = 0.5 + 0.5 * sat(t.noise(0.05, seed=seed + 3))
    gs = t.blur(g, 0.012)
    gm = float(np.median(gs))
    steep = smoothstep(0.45 * gm, 0.9 * gm, gs)                     # flat spots of the field make curly closed loops: no crack there
    ck = t.zeros()
    bank = t.zeros()
    for i, (lv, keep, wmax) in enumerate(levels):
        v = n - lv
        d = np.abs(v) / g                                           # ~distance to the line (m)
        gate = t.noise(scale * 0.8, seed=seed + 10 + i)
        thr = np.percentile(gate, 100 * (1 - keep))
        along = smoothstep(thr, thr + 0.07, gate)                   # 0 at the tips -> 1 mid-crack
        w = (wmin + (wmax - wmin) * along * wv) * 0.5
        ck = np.maximum(ck, (1 - smoothstep(w * 0.6, w * 1.4 + t.px * 0.5, d)) * smoothstep(0.0, 0.25, along) * steep)
        bank = np.maximum(bank, (v > 0) * np.exp(-(d / 0.004) ** 2) * along)
    return ck.astype(F), bank.astype(F)


def build(t):
    A = t.W * t.H
    # ================================================================== height (metres)
    # 1. the wall behind: slow undulation; only a hint of float marks at 0.15 m
    wall = (t.fbm(0.60, 3, 0.5, seed=1) - 0.5) * 0.0070 + (t.fbm(0.15, 3, 0.5, seed=2) - 0.5) * 0.0015

    # 2. trowel sweeps
    sweep, burn, stone, lipm = trowel(t, int(30 * A), 1)
    sweep = t.blur(sweep, 0.0008)                              # softened by the limewash coats

    # 5a. later repairs: 1.0-1.2 mm proud, rim feathered over 30-40 mm, own (tighter) finish
    pn = t.warp(t.fbm(0.55, 2, 0.5, seed=3), 0.05, 0.14, seed=4)
    patch = (pn > np.percentile(pn, 95.5)).astype(F)
    patch_soft = smoothstep(0.08, 0.92, t.blur(patch, 0.016))  # feathered rim ~35 mm
    patch_in = smoothstep(0.55, 0.95, patch_soft)
    rim = sat(1 - np.abs(patch_soft - 0.5) * 2)
    patch_tex = (t.fbm(0.05, 3, 0.5, seed=5) - 0.5) * 0.0005
    sweep = sweep * (1 - patch_in * 0.75)                      # the repair buries the old sweeps
    patch_h = patch_soft * 0.0012 + patch_in * patch_tex + rim ** 2 * 0.00028

    # 5b. shrinkage cracks: few, long, tapering, one bank lifted a hair
    ck, cklip = contour_cracks(t, 0.70, 6)
    ck = ck * (1 - patch_in * 0.9)
    cklip = cklip * (1 - patch_in)
    # map-crazing of the limewash skin: 20-40 mm cells, in two or three patches only
    cz = t.cells(0.030, 0.95, seed=7)
    czd = t.warp(cz.dist, 0.0015, 0.012, seed=71)
    cg = t.fbm(0.40, 2, 0.5, seed=8)
    craze_g = smoothstep(0.0, 0.05, cg - np.percentile(cg, 90)) * (1 - patch_soft)
    craze = (1 - smoothstep(0.0003, 0.0012, czd)) * craze_g * (0.45 + 0.55 * cz.rand(1))
    curl = smoothstep(0.004, 0.0, czd) * craze_g               # cell edges curl up very slightly

    # 4. brushed limewash: bristle streaks in ~110 mm swathes, each at its own angle (+-20 deg)
    sw = t.grid(0.11, 0.55, shift='random', jitter_w=0.5, seed=9, shift_rows=False)
    angs = (-20, -10, 0, 10, 20)
    k = np.minimum((sw.rand(1) * len(angs)).astype(np.int64), len(angs) - 1)
    brush = t.zeros()
    wsum = t.zeros()
    for j, da in enumerate(angs):
        wj = t.blur((k == j).astype(F), 0.007)
        bj = aniso(t, 0.017, 90 + da, 9, 10 + j, octaves=2, gain=0.7)
        brush += wj * (bj - 0.5)
        wsum += wj
    brush = brush / np.maximum(wsum, 1e-6) / 0.13              # ~ -1..1
    load = sw.rand(2) - 0.5                                    # how loaded the brush was, per stroke
    load = t.blur(load, 0.012)
    # one scatter serves both the limewash blebs and the pin-holes
    dots = t.cells(1.0 / math.sqrt(430.0), 1.0, seed=13)
    kind = dots.rand(5) < 0.26
    blebs = (1 - smoothstep(0.0005, 0.0024 * (0.5 + dots.rand(3)), dots.f1)) * kind      # limewash blebs ...
    runs = sat(t.blur(t.shift(blebs, 0, -0.006), 0.0011, 0.0055) * 2.2)  # ... some sagging into short runs
    bleb = np.maximum(blebs, runs * (sw.rand(3) > 0.45))

    # 3. sand standing in the surface; pin-holes
    grit_n = t.fbm(0.0035, 3, 0.62, seed=12)
    gn = t.noise(0.0016, seed=15, bandwidth=0.45)             # ~20000 grains per m2, 0.8-1.5 mm
    grain = smoothstep(0.70, 0.84, gn)
    grain = grain * (0.35 + 0.65 * (1 - burn))                 # hard-trowelled areas are closed up
    pin = (1 - smoothstep(0.0013 * 0.6 * (0.5 + dots.rand(3)), 0.0013 * (0.5 + dots.rand(3)), dots.f1)) * (~kind)

    height = (wall + sweep + patch_h
              + brush * 0.00005 + (grit_n - 0.5) * 0.00016 + grain * 0.00015
              + bleb * 0.00035 - pin * 0.00045
              - ck * 0.0007 + cklip * 0.00012 - craze * 0.00020 + curl * 0.00005)

    # ================================================================== masks
    cav = t.cavity(height, 0.006, 0.0004)
    high = t.edges(height, 0.006, 0.0004)

    # ================================================================== colour
    # Limewash over lime plaster is very even.  What variation there is has a cause: each sweep
    # took the wash a little differently (trowel-shaped tone patches), the brush loaded more or
    # less per stroke, and one broad soft cloud of thicker / thinner wash.  About +-3 % locally.
    cloud = t.fbm(0.75, 2, 0.5, seed=20) - 0.5
    mid = t.fbm(0.09, 3, 0.55, seed=21) - 0.5
    band = t.fbm(0.085, 3, 0.45, seed=25, stretch=(1.0, 6.0)) - 0.5
    d = (cloud * 0.085 + stone * 0.020 - (burn - 0.5) * 0.030 + load * 0.022
         + band * 0.022 + mid * 0.016 + brush * 0.006 + (grit_n - 0.5) * 0.030)
    d = d * (1 - patch_in * 0.6) + patch_soft * 0.035 - rim ** 2 * 0.012
    base = hexlin("#d6d2c7")
    alb = base[None, None, :] * (1 + d)[..., None]
    # thin wash lets the warmer plaster ghost through; thick wash is whiter and cooler
    warm = sat(0.5 - d * 9)
    alb = overlay_tint(alb, "#e6dcc6", warm * 0.16)
    # dust under the trowel lips and in the hollows, rubbed-clean high lines
    alb = mix(alb, alb * hexlin("#e9e2d4"), sat(cav * 0.55))
    alb = mix(alb, alb * 1.035, sat(high * 0.6 + bleb * 0.5))
    alb = mix(alb, hexlin("#857d6e"), ck * 0.45)
    alb = mix(alb, hexlin("#8f8779"), craze * 0.10)
    alb = mix(alb, hexlin("#8d8576"), pin * 0.42)
    alb = mix(alb, hexlin("#8c8475"), np.roll(pin[::-1, ::-1], (t.h // 3, t.w // 5), (0, 1)) * 0.22)     # fly specks
    alb = alb * (1 - grain * 0.035)[..., None]
    alb = alb * (np.array([0.655, 0.632, 0.578], dtype=F) / alb.reshape(-1, 3).mean(0))

    # ================================================================== roughness
    # burnished sweeps are satin (0.66-0.72); open, brushed, gritty wash is chalk-matt (0.9+)
    rvar = t.fbm(0.12, 3, 0.55, seed=30) - 0.5
    rough = 0.915 - burn * 0.25 + rvar * 0.05 + brush * 0.035 + (grit_n - 0.5) * 0.05
    rough = rough + grain * 0.07 + cav * 0.05 + sat(ck + craze) * 0.08 - high * 0.05 - lipm * 0.05
    rough = rough + patch_soft * 0.06 - bleb * 0.04
    return dict(albedo=alb, height=height, rough=np.clip(rough, 0.55, 0.98), metal=0.0)
