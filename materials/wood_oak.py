"""Sawn European oak, one continuous face, grain along +U (X).

Built the way a board is: a ring-phase field (tree rings cut obliquely by the saw) gives the grain
lines; where the cut runs flat to the rings the lines open into wide figure, where it runs steep
they tighten into straight rift grain with pale medullary ray fleck. Ring-porous structure: each
ring starts with a band of open vessels (dark dashed grooves) followed by dense smooth latewood.
On top: band-saw kerf marks across the grain, fine drying checks that follow the grain, a few
sound knots the grain flows around. Tintable: warm mid brown, moderate relative variation only.
"""
from tk.tex import *

META = {"normal_strength": 2.2}

N_RINGS = 66                      # ring cycles across the 0.6 m tile height (integer -> tiles)

# (u, v) in tile fractions, radius m  - spread so that no single knot can be counted in repeats
KNOTS = [(0.14, 0.24, 0.011), (0.49, 0.74, 0.013), (0.83, 0.41, 0.010)]


def _hash(i, k=0.0):
    x = np.sin(i * 12.9898 + k * 78.233) * 43758.5453
    return (x - np.floor(x)).astype(np.float32)


def _wrapd(a, L):
    return (a + L / 2) % L - L / 2


def build(t):
    X, Y, W, H = t.X, t.Y, t.W, t.H

    # ------------------------------------------------------------------ knots: grain deflection
    DY = t.zeros()
    kmask = t.zeros()            # knot body 0..1
    krad = t.zeros() + 9.0       # normalised radius inside nearest knot
    kang = t.zeros()
    khalo = t.zeros()
    for i, (ku, kv, r) in enumerate(KNOTS):
        dx = _wrapd(X - ku * W, W)
        dy = _wrapd(Y - kv * H, H)
        g = np.exp(-((dx / (r * 9.0)) ** 2 + (dy / (r * 3.2)) ** 2))
        DY += dy * 0.96 * g
        rr = np.sqrt((dx / 1.15) ** 2 + dy ** 2) / r
        krad = np.minimum(krad, rr)
        kang = np.where(rr < 3.0, np.arctan2(dy, dx), kang)
        khalo = np.maximum(khalo, np.exp(-((dx / (r * 4.5)) ** 2 + (dy / (r * 1.9)) ** 2)))
    kwob = (t.noise(0.006, seed=40) - 0.5) * 0.5
    kmask = 1 - smoothstep(0.82, 1.0, krad + kwob)

    # ------------------------------------------------------------------ ring phase (the grain)
    Ys = Y - DY
    P = (t.noise(0.10, seed=1, stretch=(6.0, 1.0)) - 0.5) * 2 * 4.2          # cut angle wanders
    P += (t.noise(0.07, seed=21, stretch=(60.0, 1.0)) - 0.5) * 2 * 2.6       # bands of tighter / wider rings
    P += (t.noise(0.045, seed=2, stretch=(7.0, 1.0)) - 0.5) * 2 * 0.9        # local waviness
    P += (t.noise(0.012, seed=3, stretch=(9.0, 1.0)) - 0.5) * 2 * 0.16       # ragged ring lines
    Pd = t.sample(P, X, Ys)
    phi = N_RINGS * Ys / H + Pd
    # year-to-year ring width variation (monotone 1-D warp of the phase, periodic in N_RINGS)
    w = 2 * np.pi / N_RINGS
    phi = phi + 0.55 * np.sin(phi * w * 3 + 1.3) + 0.38 * np.sin(phi * w * 7 + 4.1) + 0.22 * np.sin(phi * w * 13 + 0.4)
    rid = np.floor(phi)
    f = (phi - rid).astype(np.float32)
    rid = np.mod(rid, N_RINGS)
    r1, r2 = _hash(rid, 1.0), _hash(rid, 2.0)

    # local ring density (rings / m) -> flat-sawn vs rift/quarter zones, and anti-aliasing
    gx, gy = t.slope(P)
    dens = np.abs(N_RINGS / H + t.sample(gy, X, Ys)) * (1 - 0.9 * khalo) + 1e-3
    dens = t.blur(dens, 0.006)
    aa = 1 - smoothstep(0.22, 0.5, dens * t.py)
    flat = 1 - smoothstep(45.0, 95.0, dens)          # wide open figure
    quarter = smoothstep(95.0, 150.0, dens)          # tight straight grain

    # ring profile: earlywood vessel band at the start of the ring, dense latewood after it
    ew_w = 0.30 + 0.16 * r1
    early = (1 - smoothstep(ew_w * 0.55, ew_w, f)) * smoothstep(0.0, 0.05, f)
    late = smoothstep(ew_w, 0.75, f) * (1 - smoothstep(0.93, 1.0, f))
    early *= aa
    late *= aa

    # ------------------------------------------------------------------ fine structure
    fib = t.fibres(0.0045, 26.0, seed=4)                                     # fibre streaks
    fib2 = t.fbm(0.014, 3, 0.55, seed=5, stretch=(14.0, 1.0))                # broader streaks
    # open vessels: short dark dashes along the grain, concentrated in the earlywood band
    ves = t.noise(0.0022, seed=6, stretch=(8.0, 1.0), bandwidth=0.7)
    ves_m = smoothstep(0.52, 0.70, ves) * (0.22 + 0.78 * early)
    ves_m *= 1 - 0.6 * kmask
    # rays: pale fleck across the grain on rift/quarter zones, dark spindles along it on flat zones
    rayn = t.fbm(0.011, 2, 0.5, seed=7, stretch=(1.0, 3.2))
    rayn = t.warp(rayn, 0.005, 0.035, seed=8)
    fleck = smoothstep(0.62, 0.84, rayn) * smoothstep(0.35, 0.6, t.noise(0.10, seed=9, stretch=(2.5, 1))) * quarter
    spin = smoothstep(0.66, 0.82, t.noise(0.0034, seed=10, stretch=(4.0, 1.0), bandwidth=0.6))
    spin *= (0.35 + 0.65 * flat) * (1 - early * 0.5)

    # band-saw kerf marks straight across the board, strength patchy (partly planed / worn off)
    sawn = t.noise(0.0065, seed=11, stretch=(1.0, 45.0), bandwidth=0.9)
    sawn2 = t.noise(0.019, seed=12, stretch=(1.0, 30.0), bandwidth=0.8)
    saw_amt = smoothstep(0.35, 0.75, t.fbm(0.25, 3, 0.5, seed=13, stretch=(1.0, 2.0)))
    saw = ((sawn - 0.5) * 0.7 + 0.8 * (sawn2 - 0.5)) * (0.15 + 0.85 * saw_amt)

    # drying checks: follow ring boundaries on selected rings, start/stop with a gate, tapered
    gate = t.noise(0.05, seed=14, stretch=(4.0, 1.0))
    gate = smoothstep(0.60, 0.80, gate)
    sel = (_hash(rid, 5.0) < 0.42).astype(np.float32)
    dist_m = np.minimum(f, 1 - f) / dens                                     # metres to ring boundary
    wchk = 0.0011 * gate * sel
    check = (1 - smoothstep(wchk * 0.35, wchk + 1e-6, dist_m)) * (gate > 0) * sel
    check = sat(check) * (1 - kmask)

    # knot body: its own little end-grain rings, dark rim, a star check
    kr = krad + kwob
    krings = 0.5 + 0.5 * np.sin(kr * 5.5 * 2 * np.pi + t.noise(0.01, seed=41) * 6)
    kstar = smoothstep(0.86, 0.97, np.abs(np.sin(kang * 1.5 + 0.7 + (t.noise(0.02, seed=42) - 0.5) * 3))) * smoothstep(0.1, 0.3, kr) * (1 - smoothstep(0.7, 0.95, kr))
    krim = smoothstep(0.72, 0.95, kr) * kmask + (1 - smoothstep(1.0, 1.35, kr)) * (1 - kmask) * 0.7

    # ------------------------------------------------------------------ height (metres)
    macro = (t.fbm(0.28, 3, 0.5, seed=15, stretch=(2.0, 1.0)) - 0.5) * 0.0007
    height = macro
    height = height + late * 0.00016 - early * 0.00020
    height = height - ves_m * 0.00045
    height = height + (fib - 0.5) * 0.00030 + (fib2 - 0.5) * 0.00016
    height = height + saw * 0.00040
    height = height - spin * 0.00008 + fleck * 0.00005
    height = height - check * 0.0015
    height = height + kmask * (0.00025 * (krings - 0.5) - 0.0002 - kstar * 0.0008) - krim * 0.00025

    # ------------------------------------------------------------------ colour
    tone = t.fbm(0.33, 3, 0.5, seed=16, stretch=(1.6, 1.0))                  # macro patchiness
    streak = t.fbm(0.07, 3, 0.5, seed=17, stretch=(9.0, 1.0))                # heartwood streaks
    v = 0.5 + 0.55 * (streak - 0.5) + 0.30 * (tone - 0.5) + 0.22 * (fib - 0.5) + 0.16 * (fib2 - 0.5) + 0.10 * (r2 - 0.5)
    albedo = ramp(sat(v), [(0.0, "#7a5a3d"), (0.35, "#906c48"), (0.65, "#a27d55"), (1.0, "#b39168")])
    albedo = mix(albedo, albedo * hexlin("#b89474") * 1.75, late * 0.40)                 # denser, slightly darker/redder
    albedo = mix(albedo, hexlin("#6a4e37"), early * 0.38)
    albedo = mix(albedo, hexlin("#3e2c1f"), ves_m * 0.62)
    albedo = mix(albedo, hexlin("#59402c"), spin * 0.40)
    albedo = mix(albedo, hexlin("#b3956d"), fleck * 0.30)
    albedo = albedo * (1.0 + 0.07 * saw)[..., None]                           # fuzzy kerf crests catch light
    kcol = ramp(krings, [(0, "#5e432e"), (1, "#7d5b3d")])
    albedo = mix(albedo, albedo * 0.80, khalo * (1 - kmask) * 0.5)
    albedo = mix(albedo, kcol, kmask * 0.92)
    albedo = mix(albedo, hexlin("#3d2b1f"), sat(krim * 0.5 + kstar * kmask * 0.8))
    albedo = mix(albedo, hexlin("#2a1e16"), check * 0.85)
    # faint grey dust lodged in the open grain
    cav = t.cavity(height, 0.003, 0.0004)
    albedo = mix(albedo, desaturate(albedo, 0.5) * 0.82, cav * 0.25)

    # ------------------------------------------------------------------ roughness
    polish = t.fbm(0.16, 4, 0.5, seed=18, stretch=(3.5, 1.0))                                     # handled / rubbed patches
    rough = 0.72 - 0.20 * smoothstep(0.30, 0.85, polish) + 0.07 * (fib - 0.5)
    rough = rough - 0.07 * late - 0.12 * fleck + 0.10 * early + 0.16 * ves_m + 0.10 * np.abs(saw) * 2
    rough = rough + 0.2 * check - 0.10 * kmask
    rough = np.clip(rough, 0.35, 0.95)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
