"""Dressed fine-grained limestone face (no joints), a century out in the weather.

Built the way a banker mason leaves a sill or coping and the way rain then takes it back: the block
is worked flat with a broad chisel (shallow facets, soft arrises) and finished with the point of a
fine punch - short pecks a centimetre long lying every way, never in rows - then rubbed.  Weather has
since opened the stone: the softer beds have sunk a fraction of a millimetre and pitted, the hard
skin has scaled off in shallow sharp-edged patches, pores and shell moulds stand open.  NOTHING here
is directional (no drag, no brushing): the only anisotropy is the faint, broken natural bedding.
Tone, pores and shell hash are organised in those beds; a few short calcite veins cross them.
"""
from tk.tex import *
import math

BED_ANG = 3.0        # natural bedding of the stone (deg, CCW from U; near level on upright faces)
CHECK = False        # True: replay every dressing facet and report any that still wins at its window border


def aniso(t, scale, angle, stretch, seed, bandwidth=0.6, octaves=1, gain=0.5):
    """Band-limited noise 0..1 (std 0.18) with features `scale` m across and `stretch` x longer
    along `angle` degrees.  Built in the Fourier domain so it wraps at any angle."""
    rng = np.random.default_rng(t.seed * 1000 + 9000 + seed)
    wn = np.fft.fft2(rng.standard_normal((t.h, t.w)))
    FX, FY = np.meshgrid(np.fft.fftfreq(t.w, d=t.px), np.fft.fftfreq(t.h, d=t.py))
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    fu = FX * c - FY * s            # image rows run downward: flip so angle is CCW on the surface
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


def sig(n):
    """0..1 noise (std 0.18) -> roughly -1..1 (2 sigma)."""
    return np.clip((n - 0.5) / 0.36, -1.4, 1.4)


def veins(t, a, b, width, seed, wander=0.17, lo=0.58):
    """Calcite veins: zero lines of a periodic plane wave with integer wave vector (a, b) (so they
    tile for ANY integer a, b: the distance field uses |tan(ph/2)| and the analytic phase gradient,
    both of which are periodic).  Real veins are healed fractures: straight with slight kinks,
    swelling and pinching along their length, and they start and stop.  -> (core, halo) 0..1."""
    bend = (t.fbm(0.5, 2, 0.5, seed=seed) - 0.5) + (t.noise(0.09, seed=seed + 1) - 0.5) * 0.16
    ph = 2 * math.pi * (a * t.X / t.W + b * t.Y / t.H) + bend * 2 * math.pi * wander
    bx, by = t.slope(bend)
    gpx = 2 * math.pi * a / t.W + 2 * math.pi * wander * bx
    gpy = 2 * math.pi * b / t.H + 2 * math.pi * wander * by
    d = np.abs(np.tan(ph * 0.5)) / (0.5 * np.sqrt(gpx * gpx + gpy * gpy) + 1e-6)
    d = np.minimum(d, 0.05)                                   # ~distance (m) to the vein centre line
    gate = smoothstep(lo, lo + 0.20, t.fbm(0.40, 2, 0.5, seed=seed + 2))
    w = width * (0.35 + 1.3 * t.noise(0.05, seed=seed + 3))
    core = (1 - smoothstep(w * 0.25, w * 0.9, d)) * gate
    halo = (1 - smoothstep(w * 0.5, w * 2.4, d)) * gate
    return core.astype(F), halo.astype(F)


def _win(t, cx, cy, half):
    ix, iy = int(math.floor(cx / t.px)), int(math.floor((t.H - cy) / t.py))
    hw = min(int(half / t.px) + 2, t.w // 2 - 1)
    xs = np.arange(ix - hw, ix + hw + 1)
    ys = np.arange(iy - hw, iy + hw + 1)
    Xl = ((xs[None, :] + 0.5) * t.px - cx).astype(F)
    Yl = ((t.H - (ys[:, None] + 0.5) * t.py) - cy).astype(F)
    return np.ix_(ys % t.h, xs % t.w), Xl, Yl


def flakes(t, n_side, seed, wx, wy):
    """A few small spalls where a corner of the chisel lifted a flake: a shallow tilted scar,
    1-1.5 mm deep under the blow with a crisp little step there, feathering out to nothing.
    One per cell of an n_side x n_side jittered grid (so they are spread, never clumped).
    Returns depth >= 0 (metres)."""
    rng = np.random.default_rng(t.seed * 1000 + 7300 + seed)
    out = t.zeros()
    cw, ch = t.W / n_side, t.H / n_side
    for j in range(n_side):
        for i in range(n_side):
            r = rng.random(10)
            if r[0] < 0.2:
                continue
            cx = ((i + 0.15 + 0.7 * r[1]) * cw) % t.W
            cy = ((j + 0.15 + 0.7 * r[2]) * ch) % t.H
            a = r[3] * 2 * math.pi
            ca, sa = math.cos(a), math.sin(a)
            L = 0.016 + 0.020 * r[4]
            Wd = L * (0.55 + 0.4 * r[5])
            D = 0.0010 + 0.0005 * r[6]
            idx, Xl, Yl = _win(t, cx, cy, L * 1.6)
            X2, Y2 = Xl + wx[idx], Yl + wy[idx]
            u = (X2 * ca + Y2 * sa) / L                         # -0.5 blow .. +0.5 feather edge
            v = (-X2 * sa + Y2 * ca) / (Wd * 0.5)
            dep = D * np.clip(0.5 - u, 0, 1) * np.clip(1 - v * v, 0, 1) ** 0.6 * smoothstep(-0.56, -0.50, u)
            out[idx] = np.maximum(out[idx], dep.astype(F))
    return out


def dressing(t, pitch, seed):
    """Broad-chisel dressing: every stroke leaves a very shallow, slightly hollow, slightly tilted
    facet; the face is the LOWER envelope of all strokes, so it is continuous and the facets meet
    in low arrises (no steps).  Returns height (m), zero mean."""
    rng = np.random.default_rng(t.seed * 1000 + 7200 + seed)
    n = max(2, int(round(t.W / pitch)))
    m = max(2, int(round(t.H / pitch)))
    cw, ch = t.W / n, t.H / m
    R = 0.5 * math.sqrt(cw * ch)
    out = np.full((t.h, t.w), 1e3, dtype=F)
    bad = 0
    for pas in range(2 if CHECK else 1):
      rng = np.random.default_rng(t.seed * 1000 + 7200 + seed)
      for j in range(m):
        for i in range(n):
            r = rng.random(8)
            cx = ((i + 0.5 + (r[0] - 0.5) * 0.6) * cw) % t.W
            cy = ((j + 0.5 + (r[1] - 0.5) * 0.6) * ch) % t.H
            a = r[2] * math.pi
            ca, sa = math.cos(a), math.sin(a)
            sl = 0.008 + 0.013 * r[3]
            ta = r[4] * 2 * math.pi
            D = 0.0004 + 0.0005 * r[5]
            c = (r[6] - 0.5) * 0.0010
            idx, Xl, Yl = _win(t, cx, cy, R * 3.3)
            u = Xl * ca + Yl * sa
            v = -Xl * sa + Yl * ca
            e2 = (u / (1.25 * R)) ** 2 + (v / (0.85 * R)) ** 2
            h = (c + sl * (Xl * math.cos(ta) + Yl * math.sin(ta)) + D * e2 + 0.00012 * e2 * e2).astype(F)
            if pas == 1:
                win = h <= out[idx]
                bad += int(win[0].sum() + win[-1].sum() + win[:, 0].sum() + win[:, -1].sum())
                continue
            out[idx] = np.minimum(out[idx], h)
    assert out.max() < 1.0
    if CHECK:
        print('DBG dressing: window-edge wins', bad, 'range mm', float(out.max() - out.min()) * 1000)
    return out - out.mean()


def pecks(t, density, length, width, seed, keep=0.6, spread=180.0, ang0=0.0):
    """Punch / peck marks: short elongated dabs (length x width metres) lying every way.  One
    candidate per jittered grid cell, a fraction `keep` exists.  Returns (profile 0..1 with a soft
    V section, per-mark random 0..1)."""
    cell = max(1.0 / math.sqrt(density), length * 0.75, t.px * 4)
    nx, ny = max(2, int(round(t.W / cell))), max(2, int(round(t.H / cell)))
    cw, ch = t.W / nx, t.H / ny
    rng = np.random.default_rng(t.seed * 1000 + 5100 + seed)
    jx, jy, ra, rl, rk, rw = (rng.random((ny, nx)).astype(F) for _ in range(6))
    gx = np.floor(t.X / cw).astype(np.int64)
    gy = np.floor(t.Y / ch).astype(np.int64)
    out = np.zeros(t.X.shape, dtype=F)
    rv = np.zeros(t.X.shape, dtype=F)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = gx + ox, gy + oy
            wx, wy = cx % nx, cy % ny
            dx = t.X - (cx + jx[wy, wx]) * cw
            dy = t.Y - (cy + jy[wy, wx]) * ch
            a = np.radians(ang0) + (ra[wy, wx] - 0.5) * math.radians(spread)
            ca, sa = np.cos(a), np.sin(a)
            al = dx * ca + dy * sa
            ac = -dx * sa + dy * ca
            L = length * (0.45 + 0.8 * rl[wy, wx])
            Wd = width * (0.6 + 0.7 * rw[wy, wx])
            u = al / L + 0.5                                   # 0 where the point bit .. 1 run-out
            prof = smoothstep(0.0, 0.2, u) * (1 - smoothstep(0.35, 1.0, u))
            m = prof * np.clip(1 - np.abs(ac) / (Wd * 0.5), 0, 1) * (rk[wy, wx] < keep)
            upd = m > out
            out = np.where(upd, m, out)
            rv = np.where(upd, rl[wy, wx], rv)
    return out.astype(F), rv.astype(F)


def build(t):
    # ================================================================== height (metres)
    # 1. broad dressing relief: the face was brought flat with a 40-50 mm chisel -> shallow
    #    facets 12-25 cm across, each very slightly out of plane, arrises softened by rubbing
    facet = t.blur(dressing(t, 0.19, 3), 0.009)
    und = sig(t.fbm(0.22, 2, 0.5, seed=1)) * 0.00055 + sig(t.noise(0.45, seed=2)) * 0.00035
    broad = facet + und

    # 2. the stone: natural bedding (faint, broken, a few cm thick), grain, pores, shell moulds
    grain = t.fbm(0.006, 4, 0.62, seed=8)
    bedgate = aniso(t, 0.12, BED_ANG, 3.2, 5, octaves=2)                    # 2-3 softer beds per metre
    porous = smoothstep(0.54, 0.74, bedgate)
    lamina = sig(aniso(t, 0.022, BED_ANG, 5.0, 7, octaves=2))                # fine bedding, broken up
    lam_gate = 0.35 + 0.65 * smoothstep(0.4, 0.7, t.fbm(0.16, 2, 0.5, seed=64))
    pn = t.fbm(0.011, 3, 0.6, seed=9)
    vugs = smoothstep(0.80, 0.92, pn) * (0.18 + 0.82 * porous)
    pores = smoothstep(0.80, 0.94, t.fbm(0.0045, 2, 0.6, seed=11)) * (0.40 + 0.60 * porous)
    shell = smoothstep(0.80, 0.92, aniso(t, 0.007, BED_ANG, 2.2, 4)) * (0.15 + 0.85 * smoothstep(0.5, 0.7, aniso(t, 0.12, BED_ANG, 5, 6)))

    # 3. weathering relief: the face is no longer a plane at the centimetre scale.  Lumpy
    #    differential erosion (1-4 cm), soft beds sunk, and shallow scaled patches where the hard
    #    skin has let go: a crisp ragged edge 0.4-0.8 mm high that dies out round the patch
    ero = sig(t.fbm(0.034, 4, 0.55, seed=65)) * 0.00042 + sig(t.fbm(0.012, 3, 0.6, seed=66)) * 0.00020
    sk_n = t.fbm(0.15, 4, 0.55, seed=67) + (t.fbm(0.02, 3, 0.6, seed=68) - 0.5) * 0.16
    thr = np.percentile(sk_n, 80)
    scaled = smoothstep(thr, thr + 0.012, sk_n)                               # ~20 % of the face has scaled
    sc_dep = (0.00045 + 0.00035 * smoothstep(0.3, 0.7, t.fbm(0.3, 2, 0.5, seed=69)))
    sc_floor = sig(t.fbm(0.009, 3, 0.65, seed=71)) * 0.00016                  # the scar floor is granular
    # a second, smaller generation of scaling inside and beside the first
    sk2 = t.fbm(0.055, 3, 0.55, seed=72) + (t.fbm(0.012, 2, 0.6, seed=73) - 0.5) * 0.14
    thr2 = np.percentile(sk2, 90)
    scaled2 = smoothstep(thr2, thr2 + 0.012, sk2)

    # 4. the mason's finish where it survives (on the un-scaled skin, ~half the face): short punch
    #    pecks lying every way.  Rubbing and rain have taken them down to a few tenths of a mm.
    tool_n = t.fbm(0.40, 3, 0.5, seed=60) + (t.fbm(0.05, 2, 0.5, seed=61) - 0.5) * 0.3
    tooled = smoothstep(np.percentile(tool_n, 40), np.percentile(tool_n, 62), tool_n) * (1 - scaled)
    pk1, pr1 = pecks(t, 2600, 0.013, 0.0042, 1, keep=0.62)
    pk2, pr2 = pecks(t, 5200, 0.007, 0.0030, 2, keep=0.45)
    peck = np.maximum(pk1 * (0.55 + 0.45 * pr1), pk2 * 0.6 * (0.5 + 0.5 * pr2))
    peck = t.warp(peck, 0.0012, 0.008, seed=74)

    # a few small spalls (a flake that came away under the chisel), evenly spread
    wx = (t.fbm(0.02, 3, 0.55, seed=14) - 0.5) * 0.006
    wy = (t.fbm(0.02, 3, 0.55, seed=15) - 0.5) * 0.006
    spd = flakes(t, 3, 1, wx, wy)
    spall = sat(spd / 0.0005)

    c1, h1 = veins(t, 1, 2, 0.0022, 20, lo=0.66)
    c2, h2 = veins(t, -2, 1, 0.0015, 30, lo=0.70)
    vein = np.maximum(c1, c2 * 0.85)
    vhalo = np.maximum(h1, h2 * 0.85)
    vein = vein * (0.45 + 0.55 * smoothstep(0.3, 0.7, t.fbm(0.03, 3, 0.6, seed=37)))   # sugary, broken crystal fill

    height = (broad + ero
              - porous * 0.00030 + lamina * lam_gate * 0.00014 * (0.5 + 0.5 * porous + 0.5 * scaled)
              - scaled * (sc_dep + sc_floor) - scaled2 * 0.00030
              - peck * tooled * 0.00050
              + (grain - 0.5) * 0.00030
              - pores * 0.00042 - vugs * 0.00085 - shell * 0.00040
              - spd + vein * 0.00010)

    # ================================================================== masks
    cav = t.cavity(height, 0.003, 0.0005)
    cav_w = t.cavity(height, 0.010, 0.0006)
    high = t.edges(height, 0.012, 0.0005)
    pit = sat(vugs + pores * 0.5 + shell * 0.6)
    scar = sat(scaled + scaled2 * 0.6)
    rim = sat(t.edges(height, 0.0025, 0.00025) * scar * 1.5)   # the crisp lip of every scaled patch

    # ================================================================== colour
    # tone lies in soft beds and soft clouds; the old skin is a shade darker and warmer (patina)
    # than the paler, chalkier stone exposed where it has scaled
    bed1 = aniso(t, 0.034, BED_ANG, 6, 40) - 0.5
    bed2 = aniso(t, 0.10, BED_ANG, 4, 41) - 0.5
    cloud = t.fbm(0.6, 2, 0.5, seed=42) - 0.5
    mott = t.fbm(0.05, 4, 0.55, seed=43) - 0.5
    fine = t.fbm(0.0034, 3, 0.65, seed=39) - 0.5
    tone = sat(0.5 + bed1 * 0.14 + bed2 * 0.12 + cloud * 0.24 + mott * 0.36 + (grain - 0.5) * 0.30 + fine * 0.30
               + scar * 0.10 - (1 - scar) * 0.03)
    alb = ramp(tone, [(0.0, "#9f998c"), (0.3, "#aea89b"), (0.55, "#bab5a8"), (0.8, "#c4bfb3"), (1.0, "#cdc9be")])
    # iron blush / cooler skin: broad, weak
    ochre = smoothstep(0.50, 0.80, aniso(t, 0.42, BED_ANG, 2.0, 44, octaves=2))
    alb = overlay_tint(alb, "#d6c09a", ochre * 0.20 * (1 - 0.5 * scar))
    cool = smoothstep(0.52, 0.82, aniso(t, 0.38, BED_ANG, 2.0, 45, octaves=2))
    alb = overlay_tint(alb, "#bfc5c8", cool * 0.16)
    # grey weathering film on the surviving skin: soft patches, a little darker
    film = smoothstep(0.45, 0.8, t.fbm(0.26, 4, 0.55, seed=75)) * (1 - scar)
    alb = overlay_tint(alb, "#a9a69f", film * 0.26)
    # shelly / oolitic flecks (irregular, mixed 1-4 mm, never evenly spaced)
    hash_p = smoothstep(0.72, 0.90, t.fbm(0.0052, 3, 0.62, seed=48))
    hash_d = smoothstep(0.76, 0.92, t.fbm(0.0044, 3, 0.62, seed=49))
    big_p = smoothstep(0.80, 0.90, t.fbm(0.011, 3, 0.6, seed=46)) * smoothstep(0.45, 0.7, t.noise(0.06, seed=47))
    alb = mix(alb, hexlin("#d8d4c9"), sat(hash_p * 0.22 + big_p * 0.16))
    alb = mix(alb, hexlin("#8f8a7e"), hash_d * 0.22)
    # shell hash: thin curved fragments (contour arcs of a fine noise), a few mm long, lying
    # every way in the bed; grey calcite ones and paler chalky ones
    shelly = 0.35 + 0.65 * smoothstep(0.45, 0.65, aniso(t, 0.12, BED_ANG, 4, 6))
    n1, n2 = t.noise(0.020, seed=80), t.noise(0.028, seed=81)
    arc_d = (1 - smoothstep(0.012, 0.045, np.abs(n1 - 0.5))) * smoothstep(0.62, 0.76, t.noise(0.022, seed=82)) * shelly
    arc_p = (1 - smoothstep(0.012, 0.045, np.abs(n2 - 0.5))) * smoothstep(0.63, 0.77, t.noise(0.030, seed=83)) * shelly
    alb = mix(alb, hexlin("#918c81"), arc_d * 0.34)
    alb = mix(alb, hexlin("#d6d2c7"), arc_p * 0.30)
    # crustose lichen: small pale grey rosettes with a darker centre, a few ochre, in loose colonies
    # on the old skin only
    lgate = smoothstep(0.50, 0.70, t.fbm(0.33, 3, 0.5, seed=76)) * (1 - scar)
    lich = t.speckle(density=420, size=0.022, seed=77, soft=0.4) * lgate
    lich = t.warp(lich, 0.002, 0.012, seed=78) * (0.55 + 0.45 * smoothstep(0.35, 0.65, grain))
    lcol = mix(hexlin("#c9c9bf"), hexlin("#b8a468"), smoothstep(0.62, 0.72, t.noise(0.13, seed=79)))
    alb = mix(alb, lcol * (0.88 + 0.24 * grain)[..., None], lich * 0.42)
    # black micro-lichen / soot specks peppering the skin, thicker in the damp soft beds
    spk = t.speckle(density=9000, size=0.0034, seed=84, soft=0.6)
    spk = spk * smoothstep(0.40, 0.66, t.fbm(0.08, 3, 0.55, seed=85)) * (1 - scar) * (0.4 + 0.6 * porous)
    alb = mix(alb, hexlin("#5d5a52"), spk * 0.42)
    # bruised stone in the peck marks is paler; dirt sits in the pits and at the foot of every scar lip
    alb = mix(alb, alb * 1.07 + 0.004, sat(peck * tooled * 1.2) * 0.5)
    alb = mix(alb, hexlin("#dcd8ce"), sat(vhalo) * 0.10)
    alb = mix(alb, hexlin("#dedbd1"), sat(vein) * 0.09)
    alb = mix(alb, alb * hexlin("#cfc5b2") * 1.02, sat(pit * 0.95 + cav * 0.45 + cav_w * 0.25))
    alb = mix(alb, hexlin("#c9c4b8"), spall * 0.35)            # fresh stone in the flake scar
    alb = mix(alb, alb * 1.07, sat(high * 0.4 + rim * 0.6))
    # limestone is greyer than you think: pull toward grey, then pin the mean to the target
    alb = desaturate(alb, 0.25)
    alb = alb * (np.array([0.470, 0.450, 0.392], dtype=F) / alb.reshape(-1, 3).mean(0))

    # ================================================================== roughness
    # weathered limestone is matt; the rubbed skin that survives is a little tighter, lichen,
    # scars and dirt-filled pits are the roughest; calcite veins and worn high points smoother
    rvar = t.fbm(0.07, 3, 0.55, seed=50) - 0.5
    rough = 0.70 + rvar * 0.09 + (grain - 0.5) * 0.09 - (1 - scar) * (1 - tooled) * 0.07
    rough = rough + scar * 0.07 + film * 0.05 + lich * 0.10 - high * 0.05 - vein * 0.012
    rough = rough + cav * 0.08 + pit * 0.14 + spall * 0.08 + peck * tooled * 0.06
    return dict(albedo=np.clip(alb, 0.02, 0.9), height=height, rough=np.clip(rough, 0.35, 0.97), metal=0.0)
