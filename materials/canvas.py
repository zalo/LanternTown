"""Heavy cotton canvas (numbered duck): tight plain weave of plied cotton yarn, unbleached off-white.

256 warp ends x 256 weft picks per 250 mm tile (0.98 mm pitch, exactly 4 px per yarn at 1024 px - the
finest sett the bake resolves, close to a real No. 8-10 duck; a whole number of pixels per yarn so
the mip chain averages the weave out cleanly instead of beating into bands).  Height is the smooth max of two crimped yarn systems
with wide crowns that touch, so the face is the closed, brick-like field of little oval floats that
dense duck shows - not separate beads in a dark grid.  Slubs are few and short.  Cloth-scale relief is
kept under ~1.2 mm and follows fold logic: the cloth has been folded and rolled along warp and weft,
so it carries a handful of long, soft creases within +-10 degrees of the yarn directions (placed by
stratified position so no single mark is countable), over a gentle billow.
Authored light and nearly neutral: the engine tints it (and paints stripes across U)."""
from tk.tex import *

META = {"normal_strength": 1.0}

NWARP = 256      # ends across U   (even)
NWEFT = 256      # picks across V  (even)


def _smax(a, b, k):
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


def _creases(t, seed, n_each=5):
    """Long soft fold lines along warp and weft.  Returns (signed profile ~-1..1, sharp-line mask
    of outward folds, sharp-line mask of inward folds)."""
    rng = np.random.default_rng(seed)
    prof = t.zeros()
    out_l = t.zeros()
    in_l = t.zeros()
    # folds are never dead straight: bend the coordinate frame by a millimetre or two
    bx = (t.noise(0.09, seed=31) - 0.5) * 2 * 0.006
    by = (t.noise(0.09, seed=32) - 0.5) * 2 * 0.006
    k = 0
    for axis in (0, 1):
        for i in range(n_each):
            p = (i + 0.15 + 0.7 * rng.random()) / n_each          # stratified across the fold direction
            a = rng.random()
            cx, cy = (a * t.W, p * t.H) if axis == 0 else (p * t.W, a * t.H)
            ang = math.radians((rng.random() - 0.5) * 20.0) + (0.0 if axis == 0 else math.pi / 2)
            L = 0.15 + 0.07 * rng.random()
            sign = 1.0 if (k % 2 == 0) else -1.0                   # alternate mountain / valley folds
            amp = 0.55 + 0.45 * rng.random()
            k += 1
            dx = np.mod(t.X + bx - cx + t.W / 2, t.W) - t.W / 2
            dy = np.mod(t.Y + by - cy + t.H / 2, t.H) - t.H / 2
            al = dx * math.cos(ang) + dy * math.sin(ang)
            ac = -dx * math.sin(ang) + dy * math.cos(ang)
            taper = sat(1 - (2 * al / L) ** 2) ** 1.5              # the crease dies away at both ends
            sh = 0.50 * np.exp(-np.abs(ac) / 0.0011) + 0.50 * np.exp(-(ac / 0.0058) ** 2)
            prof += sign * amp * sh * taper
            line = np.exp(-(ac / 0.0011) ** 2) * taper * amp
            if sign > 0:
                out_l = np.maximum(out_l, line)
            else:
                in_l = np.maximum(in_l, line)
    return prof.astype(F), out_l.astype(F), in_l.astype(F)


def build(t):
    pi = math.pi
    pw, pf = t.W / NWARP, t.H / NWEFT

    # --- the cloth is never perfectly square: yarns drift a fraction of a pitch
    gx = (t.noise(0.06, seed=1) - 0.5) * 2 * 0.0007 + (t.noise(0.006, seed=2, stretch=(1, 14)) - 0.5) * 2 * 0.00016
    gy = (t.noise(0.06, seed=3) - 0.5) * 2 * 0.0007 + (t.noise(0.006, seed=4, stretch=(14, 1)) - 0.5) * 2 * 0.00019
    u = (t.X + gx) / pw
    v = (t.Y + gy) / pf
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu - 0.5, v - iv - 0.5
    paru = np.where((iu % 2) == 0, 1.0, -1.0)
    parv = np.where((iv % 2) == 0, 1.0, -1.0)

    # --- slub: now and then a yarn runs a little thick for 1-2 cm
    sl_w = t.noise(pf * 1.9, seed=5, stretch=(12, 1))             # weft yarns run along U
    sl_p = t.noise(pw * 1.9, seed=6, stretch=(1, 13))             # warp yarns run along V
    slub_w = smoothstep(0.78, 0.92, sl_w)
    slub_p = smoothstep(0.80, 0.93, sl_p)
    thick_w = 1.0 + 0.05 * (sl_w - 0.5) * 4 + 0.25 * slub_w
    thick_p = 1.0 + 0.04 * (sl_p - 0.5) * 4 + 0.18 * slub_p

    # --- two crimped yarn systems (m). warp crimps more than weft in a duck. Wide crowns: yarns touch.
    ridge_w = sat(np.cos(pi * fv))                                     # across-yarn crown, 1 on the axis
    ridge_p = sat(np.cos(pi * fu))
    crimp_w = np.cos(pi * (u - 0.5)) * parv                       # +1 where the pick rides over an end
    crimp_p = -np.cos(pi * (v - 0.5)) * paru
    z_w = 0.000105 * crimp_w + 0.00026 * thick_w * ridge_w ** 0.5
    z_p = 0.000150 * crimp_p + 0.00025 * thick_p * ridge_p ** 0.5
    weave = _smax(z_w, z_p, 0.00004)
    isw = z_w >= z_p
    crimp = np.where(isw, crimp_w, crimp_p)
    ridge = np.where(isw, ridge_w, ridge_p)
    slub = np.where(isw, slub_w, slub_p)

    # ply twist + cotton fuzz (micro)
    fib = np.where(isw, t.fbm(0.0009, 2, 0.6, seed=7, stretch=(5, 1)), t.fbm(0.0009, 2, 0.6, seed=8, stretch=(1, 5)))
    fuzz = t.fbm(0.0018, 3, 0.6, seed=9)
    micro = (fib - 0.5) * 0.00005 + (fuzz - 0.5) * 0.00004

    # --- cloth-scale relief (<= ~1.2 mm in all): gentle billow, soft handling, long fold creases
    billow = t.fbm(0.11, 2, 0.45, seed=10)
    # handling: soft buckles that follow the yarn directions (cloth bends along warp or weft)
    buck = 0.5 * (t.fbm(0.022, 2, 0.45, seed=11, stretch=(3.0, 1)) + t.fbm(0.022, 2, 0.45, seed=12, stretch=(1, 3.0)))
    # fine set-in wrinkles from being rolled up: a few mm apart, along warp or weft, in patches
    wr_w = t.fbm(0.0070, 2, 0.45, seed=13, stretch=(6.0, 1))
    wr_p = t.fbm(0.0070, 2, 0.45, seed=14, stretch=(1, 6.0))
    pw_ = smoothstep(0.42, 0.62, t.noise(0.05, seed=15))
    pp_ = smoothstep(0.42, 0.62, t.noise(0.05, seed=16))
    wrinkle = (wr_w - 0.5) * pw_ + (wr_p - 0.5) * pp_
    fold, ridge_fold, valley_fold = _creases(t, 7103, 6)
    cloth = (billow - 0.5) * 0.00040 + (buck - 0.5) * 0.00050 + wrinkle * 0.00020 + fold * 0.00035

    height = weave + micro + cloth

    # --- masks
    w_lo, w_hi = np.percentile(weave, 0.5), np.percentile(weave, 99.5)
    wv = sat((weave - w_lo) / (w_hi - w_lo))
    pit = sat(1 - wv / 0.45)                                       # the tiny interstices between yarns
    crown = smoothstep(0.3, 1.0, crimp) * ridge                    # yarn crowns: what gets rubbed
    macro = t.fbm(0.10, 4, 0.5, seed=17)
    stain = t.fbm(0.05, 4, 0.55, seed=18)

    # --- colour: unbleached cotton - warm grey-white; faint yarn-to-yarn tone, pits a touch greyer
    yt_w = t.noise(pf * 1.6, seed=19, stretch=(14, 1))
    yt_p = t.noise(pw * 1.6, seed=20, stretch=(1, 14))
    yt = np.where(isw, yt_w, yt_p)
    tn = sat(0.5 + (yt - 0.5) * 0.4 + (macro - 0.5) * 0.22 + (fuzz - 0.5) * 0.30 + slub * 0.07)
    base = ramp(tn, [(0.0, "#bcb39e"), (0.35, "#c9c1ad"), (0.65, "#d2cbb8"), (1.0, "#dcd5c4")])
    base = base * (1.0 - 0.12 * pit)[..., None]
    base = mix(base, base * hexlin("#cdbfa6") * 1.12, (pit * 0.3)[..., None])       # dust in the weave
    base = mix(base, base * 1.04 + 0.006, crown * 0.30)
    # faint weathering: slightly yellowed / greyed tide marks, very soft
    base = mix(base, base * hexlin("#e9dfc6") * 1.05, smoothstep(0.55, 0.9, stain)[..., None] * 0.22)
    base = mix(base, base * hexlin("#d9d6cf") * 1.04, smoothstep(0.6, 0.95, 1 - macro)[..., None] * 0.12)
    # motes: bits of cotton seed / leaf trash spun into unbleached yarn
    neps = t.speckle(density=22000, size=0.0010, seed=21, soft=0.6) * smoothstep(0.35, 0.7, t.noise(0.004, seed=22))
    base = mix(base, hexlin("#8b7654"), neps * 0.45)
    # creases: rubbed a shade paler on outward folds, a faint grey line of dust in inward folds
    base = mix(base, base * 1.05 + 0.008, ridge_fold * 0.2)
    base = mix(base, base * hexlin("#d8d1c3") * 1.03, valley_fold * 0.12)
    albedo = base

    # --- roughness: matt cotton; crowns and rubbed arrises a little smoother, dusty pits rougher,
    #     plus a slow +-0.04 handling term (where hands and rope have polished the cloth)
    handled = t.fbm(0.09, 3, 0.5, seed=23)
    rough = (0.90 - 0.07 * crown - 0.08 * ridge_fold + 0.05 * pit + 0.04 * valley_fold
             + 0.08 * (handled - 0.5) + 0.05 * (fib - 0.5) - 0.03 * slub)

    # AO only from the weave and the inward creases: billows are open, they do not occlude
    ao = sat(1.0 - 0.20 * pit - 0.05 * valley_fold)
    return dict(albedo=sat(albedo), height=height, rough=sat(rough), metal=0.0, ao=ao)
