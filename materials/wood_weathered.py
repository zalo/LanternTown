"""Silver-grey weathered softwood (pine / spruce left out for decades), grain along +U (X).

Construction: flat-sawn softwood -> a ring-phase field with slow "heart" axes gives cathedral
arches down the board and tight straight grain between them. Weather has eaten the soft pale
earlywood away and left the hard latewood standing as ridges (raised grain, saw-tooth profile
across each ring). UV has bleached the fibres silver; dirt and mildew sit in the eroded valleys.
Drying checks and a few long splits follow the fibres, knots stand proud and are cracked, and a
few rusted nail heads have bled blue-black iron stain along the grain.
"""
from tk.tex import *

META = {"normal_strength": 2.0}

M_X = 10          # ring cycles gained along the tile length (taper of the log) - integer so it tiles

KNOTS = [(0.21, 0.31, 0.012), (0.58, 0.83, 0.010), (0.86, 0.52, 0.014)]
NAILS = [(0.07, 0.10), (0.56, 0.60), (0.33, 0.90), (0.80, 0.24), (0.42, 0.42), (0.93, 0.76)]


def _wrapd(a, L):
    return (a + L / 2) % L - L / 2


def build(t):
    X, Y, W, H = t.X, t.Y, t.W, t.H

    # ------------------------------------------------------------------ knots: fibre deflection
    DY = t.zeros()
    krad = t.zeros() + 9.0
    kang = t.zeros()
    khalo = t.zeros()
    for (ku, kv, r) in KNOTS:
        dx = _wrapd(X - ku * W, W)
        dy = _wrapd(Y - kv * H, H)
        g = np.exp(-((dx / (r * 8.0)) ** 2 + (dy / (r * 3.0)) ** 2))
        DY += dy * 0.95 * g
        rr = np.sqrt((dx / 1.2) ** 2 + dy ** 2) / r
        kang = np.where(rr < krad, np.arctan2(dy, dx), kang)
        krad = np.minimum(krad, rr)
        khalo = np.maximum(khalo, np.exp(-((dx / (r * 4.0)) ** 2 + (dy / (r * 1.8)) ** 2)))
    kwob = (t.noise(0.007, seed=40) - 0.5) * 0.45
    kr = krad + kwob
    kmask = 1 - smoothstep(0.85, 1.0, kr)
    Ys = Y - DY

    # ------------------------------------------------------------------ ring phase
    heart = (t.noise(0.16, seed=1, stretch=(14.0, 1.0), bandwidth=0.45) - 0.5) * 2 * 17.0   # heart axes
    P = heart
    P += (t.noise(0.07, seed=2, stretch=(5.0, 1.0)) - 0.5) * 2 * 1.6                       # wavy figure
    P += (t.noise(0.014, seed=3, stretch=(8.0, 1.0)) - 0.5) * 2 * 0.22                     # ragged lines
    phi = t.sample(P, X, Ys) + M_X * X / W
    gx, gy = t.slope(P)
    gxs = t.sample(gx, X, Ys) + M_X / W
    gys = t.sample(gy, X, Ys)
    dens = np.sqrt(gxs ** 2 + gys ** 2) * (1 - 0.85 * khalo) + 2.0
    dens = t.blur(dens, 0.004)
    aa = 1 - smoothstep(0.25, 0.5, dens * t.px)
    # good and lean years: ring character cycles with period M_X (keeps the X seam exact)
    w = 2 * np.pi / M_X
    phi = phi + 0.42 * np.sin(phi * w * 1 + 0.7) + 0.20 * np.sin(phi * w * 3 + 2.9)
    rid = np.floor(phi)
    f = (phi - rid).astype(np.float32)
    ryear = 0.5 + 0.5 * np.sin(rid * w * 3 + 1.1) * np.cos(rid * w * 1 + 0.3)              # per ring 0..1

    # softwood ring: wide soft earlywood, gradual transition, thin hard latewood, abrupt ring end
    lw0 = 0.60 - 0.12 * ryear
    late = smoothstep(lw0, 0.90, f) * (1 - smoothstep(0.965, 1.0, f))
    crest = (smoothstep(0.15, 0.92, f) ** 1.6) * (1 - smoothstep(0.955, 1.0, f))            # eroded profile
    late *= aa
    crest = crest * aa + 0.45 * (1 - aa)

    # ------------------------------------------------------------------ fibres and erosion
    fib = t.sample(t.fibres(0.0038, 30.0, seed=4), X, Ys)
    fib2 = t.sample(t.fbm(0.012, 3, 0.55, seed=5, stretch=(16.0, 1.0)), X, Ys)
    erode = t.fbm(0.22, 3, 0.5, seed=6, stretch=(2.0, 1.0))                # how hard the weather has bitten
    depth = 0.55 + 0.45 * smoothstep(0.25, 0.75, erode)

    # checks: follow the fibres (contours of a strongly stretched field), start, taper and stop
    def fibre_cracks(scale, stretch, gate_scale, gate_lo, gate_hi, wmax, seed):
        c = t.noise(scale, seed=seed, stretch=(stretch, 1.0))
        _, cgy = t.slope(c)
        c = t.sample(c, X, Ys)
        cgy = np.abs(t.sample(cgy, X, Ys))
        valid = smoothstep(0.12 / scale, 0.40 / scale, cgy)                # no U-turns: cracks only run along the fibres
        d = np.abs(c - 0.5) / np.maximum(cgy, 0.12 / scale)
        gate = smoothstep(gate_lo, gate_hi, t.noise(gate_scale, seed=seed + 1, stretch=(5.0, 1.0)))
        wd = wmax * gate * valid
        return sat(1 - smoothstep(wd * 0.3, wd + 1e-6, d)) * (gate > 0.01)

    split = fibre_cracks(0.045, 45.0, 0.07, 0.60, 0.86, 0.0030, 20)        # few long open splits
    check = fibre_cracks(0.014, 40.0, 0.035, 0.52, 0.80, 0.0016, 22)       # many short surface checks
    check2 = fibre_cracks(0.007, 36.0, 0.02, 0.50, 0.78, 0.0012, 24) * 0.7
    check = np.maximum(check, check2) * (1 - kmask)
    split = split * (1 - kmask)

    # knot: hard, stands proud, own tight rings, radial star check, loose dark rim
    krings = 0.5 + 0.5 * np.sin(kr * 6.0 * 2 * np.pi + t.noise(0.012, seed=41) * 6)
    kstar = smoothstep(0.82, 0.96, np.abs(np.sin(kang * 1.5 + 0.4 + (t.noise(0.02, seed=42) - 0.5) * 3)))
    kstar = kstar * smoothstep(0.05, 0.25, kr) * (1 - smoothstep(0.75, 0.98, kr)) * kmask
    krim = np.exp(-((kr - 1.02) / 0.10) ** 2)

    # nails: small rusted heads slightly sunk, iron stain bleeding along the grain
    nail = t.zeros()
    stain = t.zeros()
    rust = t.zeros()
    sn = t.fibres(0.006, 18.0, seed=30)
    for i, (nu, nv) in enumerate(NAILS):
        dx = _wrapd(X - nu * W, W)
        dy = _wrapd(Y - nv * H, H)
        rr = np.sqrt(dx * dx + dy * dy)
        nail = np.maximum(nail, 1 - smoothstep(0.0032, 0.0044, rr))
        L = 0.05 + 0.04 * ((i * 37) % 5) / 4.0
        sy = 0.008 + 0.004 * ((i * 53) % 4) / 3.0
        s = np.exp(-np.abs(dx) / L) * np.exp(-(dy / sy) ** 2)
        stain = np.maximum(stain, s * (0.55 + 0.45 * ((i * 29) % 3) / 2.0))
        rust = np.maximum(rust, np.exp(-(rr / 0.008) ** 2))
    stain = sat(stain * (0.35 + 1.5 * sn))

    # ------------------------------------------------------------------ height (metres)
    macro = (t.fbm(0.30, 3, 0.5, seed=7, stretch=(2.5, 1.0)) - 0.5) * 0.0012
    height = macro + (crest - 0.45) * 0.0011 * depth
    height = height + (fib - 0.5) * 0.00040 + (fib2 - 0.5) * 0.00030
    trough = t.blur(sat(check + split), 0.0016)                            # eroded crack lips
    height = height - check * 0.0012 - split * 0.0026 - trough * 0.0005
    height = height + kmask * (0.0005 + 0.0002 * (krings - 0.5)) - kstar * 0.0012 - krim * 0.0005 * (1 - kmask)
    height = height - nail * 0.0006 - rust * 0.0002
    pit = t.speckle(density=5000, size=0.0028, seed=8) * (1 - late)        # small rot / insect pits
    height = height - pit * 0.0004

    # ------------------------------------------------------------------ colour
    cav = t.cavity(height, 0.004, 0.0005)
    ridge = t.edges(height, 0.004, 0.0004)
    tone = t.fbm(0.30, 4, 0.5, seed=9, stretch=(2.2, 1.0))
    streak = t.sample(t.fbm(0.05, 3, 0.5, seed=10, stretch=(12.0, 1.0)), X, Ys)
    v = 0.5 + 0.42 * (streak - 0.5) + 0.34 * (tone - 0.5) + 0.26 * (fib - 0.5) + 0.20 * (fib2 - 0.5)
    albedo = ramp(sat(v), [(0.0, "#6f6b64"), (0.3, "#8b8780"), (0.6, "#a19d94"), (1.0, "#b8b4a9")])
    # sheltered zones keep a little of the wood's brown
    brown = smoothstep(0.45, 0.85, t.fbm(0.10, 3, 0.5, seed=11, stretch=(8.0, 1.0))) * (1 - 0.6 * erode)
    albedo = mix(albedo, albedo * hexlin("#c9b08f") * 1.55, brown * 0.40)
    albedo = mix(albedo, albedo * hexlin("#b9a58c") * 1.18, late * 0.75)                 # resinous latewood
    albedo = albedo * (1.0 + 0.16 * ridge - 0.30 * cav)[..., None]                       # bleached crests, dirty valleys
    mild = t.speckle(density=26000, size=0.0026, seed=12) * smoothstep(0.45, 0.7, t.fbm(0.12, 3, 0.5, seed=13))
    albedo = mix(albedo, hexlin("#4c4a46"), mild * 0.38 * (1 - late))                    # mildew specks
    albedo = mix(albedo, albedo * hexlin("#77767a") * 1.9 * 0.50, sat(stain * 1.1))          # blue-black iron stain
    albedo = mix(albedo, hexlin("#5c4638"), rust * 0.5)
    kcol = ramp(krings, [(0, "#4d443b"), (1, "#6e6154")])
    albedo = mix(albedo, albedo * 0.82, khalo * (1 - kmask) * 0.5)
    albedo = mix(albedo, kcol, kmask * 0.9)
    albedo = mix(albedo, hexlin("#3a3531"), sat(krim * 0.55 * (1 - kmask) + kstar))
    albedo = mix(albedo, albedo * 0.78, trough * 0.6)
    albedo = mix(albedo, hexlin("#2e2b28"), sat(check * 0.75 + split * 0.95))
    albedo = mix(albedo, hexlin("#33291f") * (0.8 + 0.5 * t.noise(0.002, seed=31))[..., None], nail)

    # ------------------------------------------------------------------ roughness
    rough = 0.86 + 0.08 * (fib - 0.5) - 0.14 * late * (0.5 + 0.5 * ridge) - 0.06 * ridge + 0.08 * cav
    rough = rough - 0.12 * kmask + 0.08 * sat(check + split) - 0.05 * brown
    rough = mix(rough, 0.80, nail)
    rough = np.clip(rough, 0.5, 0.97)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
