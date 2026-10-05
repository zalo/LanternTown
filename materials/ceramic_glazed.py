"""Glazed ceramic: glossy off-white tin glaze over an earthenware body, finely crazed.

Construction: a glass layer ~0.3 mm thick that flowed in the kiln - it pooled and sagged a little
(soft drips with a rounded leading edge), kept a faint orange-peel, a few pinhole clusters and
glaze-skip dimples.  Crazing is a hierarchy: a few long old cracks split the surface into fields;
each field cracked into long gently curved parallel primaries (6-9 mm apart), bridged at right
angles by rungs that END on the primaries (T-junctions), and the resulting cells split again
across their long axis (final mesh 2.5-4 mm).  Cracks are hairlines: they show as a change of
gloss with a slightly raised lip, and only where the piece stood in tea / dirt are they stained.
Tintable: every colour variation is a multiplier of the base glaze colour.
"""
from tk.tex import *

META = {"normal_strength": 2.0, "ao_radius": 0.002, "ao_strength": 0.35}

TARGET = (0.685, 0.662, 0.585)          # mean linear albedo of the white glaze (~#d9d6ca)
TINT_TEST = None                        # e.g. (0.05, 0.22, 0.10) to preview a tinted use


def _net(a, ga, b, gb, n1, kmin, kmax, seed):
    """Right-angle crack hierarchy inside one field.
    a: across-strip coordinate (tile units), b: along-strip coordinate, ga / gb their metric
    (1/m).  Returns distances in metres to: primaries, rungs, strip-wise splits, rung-wise splits."""
    rng = np.random.default_rng(seed)
    ks = rng.integers(kmin, kmax + 1, n1)
    off = rng.random(n1).astype(np.float32)
    ms = rng.integers(4, 11, n1).astype(np.float32)
    mp = rng.random(n1).astype(np.float32)
    T = rng.random((n1, kmax + 1, 6)).astype(np.float32)
    ph1 = n1 * a
    s0 = np.floor(ph1)
    f = ph1 - s0
    s = np.mod(s0.astype(np.int64), n1)
    w1 = 1.0 / (n1 * np.maximum(ga, 1e-6))                 # strip width (m)
    d1 = np.minimum(f, 1 - f) * w1
    ki = ks[s]
    k = ki.astype(np.float32)
    # rungs: own pitch and offset per strip, and unevenly spaced along it
    wob = 0.30 * np.sin(2 * np.pi * (ms[s] * b + mp[s]))
    ph2 = k * b + off[s] + wob
    c0 = np.floor(ph2)
    g = ph2 - c0
    c = np.mod(c0.astype(np.int64), ki)
    dw = 1 + 0.30 * 2 * np.pi * ms[s] / k * np.cos(2 * np.pi * (ms[s] * b + mp[s]))
    w2 = 1.0 / (k * np.maximum(dw, 0.2) * np.maximum(gb, 1e-6))   # local rung pitch (m)
    d2 = np.minimum(g, 1 - g) * w2
    r = T[s, c]
    long_cell = w2 > 0.8 * w1
    # cells split along the strip (a line that ends on the two rungs) ...
    split3 = r[..., 0] < 0.80
    f3 = 0.34 + 0.32 * r[..., 1]
    d3 = np.where(split3, np.abs(f - f3) * w1, 1.0)
    half = split3 & (f > f3)
    # ... and each half splits again across its long axis (ends on the primary and the split)
    rh = np.where(half, r[..., 2], r[..., 3])
    rg = np.where(half, r[..., 4], r[..., 5])
    split4 = (rh < 0.70) & long_cell
    g4 = 0.30 + 0.40 * rg
    d4 = np.where(split4, np.abs(g - g4) * w2, 1.0)
    return [x.astype(np.float32) for x in (d1, d2, d3, d4)]


def _line(d, w):
    return 1 - smoothstep(0.0001, w, d)


def build(t):
    # ------------------------------------------------------------------ crazing geometry
    dx = (t.noise(0.075, seed=1, bandwidth=0.5) - 0.5) * 0.036 + (t.noise(0.016, seed=2) - 0.5) * 0.0024
    dy = (t.noise(0.075, seed=3, bandwidth=0.5) - 0.5) * 0.036 + (t.noise(0.016, seed=4) - 0.5) * 0.0024
    gxx, gxy = t.slope(dx)
    gyx, gyy = t.slope(dy)
    a_x = (t.X + dx) / t.W
    a_y = (t.Y + dy) / t.H
    jx = (t.noise(0.011, seed=9, stretch=(1.0, 7.0), bandwidth=0.5) - 0.5) * 0.0075   # uneven strip widths
    jy = (t.noise(0.011, seed=10, stretch=(7.0, 1.0), bandwidth=0.5) - 0.5) * 0.0075
    g_x = np.sqrt((1 + gxx) ** 2 + gxy ** 2) / t.W
    g_y = np.sqrt(gyx ** 2 + (1 + gyy) ** 2) / t.H

    # fields: a few long old cracks; each field has its own primary direction and phase
    zl = t.cells(0.115, jitter=0.9, seed=5)

    def wz(x):
        return t.warp(x, 0.009, 0.05, seed=6)

    d0 = wz(zl.dist)
    ori = wz(zl.rand(1)) > 0.5
    oa = wz(zl.rand(2))
    ob = wz(zl.rand(3))
    N = 54                                                   # 0.4 m / 54 = 7.4 mm primaries
    jgx = t.slope(jx)[0]
    jgy = t.slope(jy)[1]
    A = _net(a_y + jy / t.H + oa, g_y * np.maximum(1 + jgy, 0.25), a_x + ob, g_x, N, 34, 56, 11)    # primaries run along U
    B = _net(a_x + jx / t.W + oa, g_x * np.maximum(1 + jgx, 0.25), a_y + ob, g_y, N, 34, 56, 12)    # primaries run along V
    d1, d2, d3, d4 = [np.where(ori, b, a) for a, b in zip(A, B)]

    fine_z = smoothstep(0.35, 0.60, t.fbm(0.09, 3, 0.5, seed=7))             # where the mesh went on subdividing
    along = 0.60 + 0.40 * smoothstep(0.3, 0.7, t.fbm(0.02, 3, 0.55, seed=8))  # cracks open and close along their length
    crack = np.maximum.reduce([
        _line(d0, 0.00032) * 1.00,
        _line(d1, 0.00031) * 0.90,
        _line(d2, 0.00028) * 0.78,
        _line(d3, 0.00025) * 0.60 * (0.25 + 0.75 * fine_z),
        _line(d4, 0.00023) * 0.48 * fine_z,
    ]) * along
    old = np.maximum(_line(d0, 0.00030), _line(d1, 0.00030) * 0.8) * along   # the open, older cracks

    # ------------------------------------------------------------------ glaze thickness / flow
    broad = t.fbm(0.14, 3, 0.5, seed=20)
    runs_n = t.noise(0.042, seed=21, stretch=(1.0, 1.7), bandwidth=0.5)
    thr = float(np.percentile(runs_n, 93.0))
    blob = smoothstep(thr - 0.02, thr + 0.08, runs_n)                        # the fat lower end of a sag
    tail = blob
    for k in range(1, 7):                                                    # thinning trail above it
        tail = np.maximum(tail, t.shift(blob, dy=0.007 * k) * (1 - k / 7.0) ** 1.2)
    sag = t.blur(tail, 0.0045) * 0.6 + sat(t.blur(blob, 0.0028)) * 0.4     # rounded bead at the leading edge
    wavy = t.fbm(0.045, 3, 0.45, seed=22)
    thick = t.blur(0.65 * broad + 0.30 * wavy + 0.22 * tail, 0.006)
    thick = level(thick)

    peel = t.fbm(0.0024, 2, 0.5, seed=23)                                    # orange-peel, 1.5-3 mm
    peel_z = 0.4 + 0.6 * smoothstep(0.3, 0.7, t.noise(0.10, seed=24, bandwidth=0.5))

    # pinholes: tiny burst bubbles, in a few clusters; glaze skips: shallow dimples
    pin_cl = smoothstep(0.62, 0.78, t.fbm(0.07, 3, 0.5, seed=25))
    pin = t.speckle(density=60000, size=0.0009, seed=26, soft=0.5) * pin_cl
    pin = np.maximum(pin, t.speckle(density=700, size=0.0011, seed=27, soft=0.5))
    pin_rim = sat(t.blur(pin, 0.0005) * 2.5 - pin)
    skip = t.blur(t.speckle(density=260, size=0.0034, seed=28, soft=0.9), 0.0007)

    # use: micro-scratch haze, a few longer scratches
    haze = np.maximum(t.scratches(density=25000, length=0.010, width=0.0004, seed=30),
                      t.scratches(density=25000, length=0.016, width=0.0004, seed=31))
    scr = t.scratches(density=260, length=0.045, width=0.0004, seed=32)

    lipm = sat(t.blur(crack, 0.00045) * 2.0 - crack)                         # glaze curls up beside a crack
    height = (wavy - 0.5) * 0.00016 + (broad - 0.5) * 0.00010 + sag * 0.00020 \
        + (peel - 0.5) * 0.000055 * peel_z \
        - crack * 0.000030 + lipm * 0.000008 \
        - pin * 0.00006 + pin_rim * 0.000008 - skip * 0.00004 - scr * 0.000005

    # ------------------------------------------------------------------ colour (all relative to the base)
    base = ramp(thick, [(0.0, "#dcd6c6"), (0.5, "#dad7cb"), (1.0, "#d8d8d0")])   # thin: body warms it; thick: cooler
    base = base * (0.985 + 0.03 * t.fbm(0.03, 3, 0.5, seed=40))[..., None]
    # dirt / tea stain soaked into the crazing: only in the zones that stood wet
    sn = t.fbm(0.10, 3, 0.5, seed=41)
    lo, hi = float(np.percentile(sn, 60.0)), float(np.percentile(sn, 72.0))
    sz = smoothstep(lo, hi, sn)
    stain = crack * (0.06 + 0.25 * sz * (0.7 + 0.3 * old))
    bleed = t.blur(old * sz, 0.0008) * 0.10                                  # faint halo creeping under the glaze
    st_col = np.array([0.36, 0.27, 0.17], dtype=np.float32)                  # multiplier: brown tea / iron
    base = base * (1 - sat(stain + bleed)[..., None] * (1 - st_col))
    # body showing through pinholes and thin skips; sparse iron specks burnt through the glaze
    body = np.array([0.62, 0.42, 0.27], dtype=np.float32)
    base = base * (1 - (pin * 0.55 + skip * 0.10)[..., None] * (1 - body))
    iron = t.speckle(density=1800, size=0.0008, seed=42, soft=0.7)
    base = base * (1 - iron[..., None] * 0.45 * (1 - np.array([0.45, 0.30, 0.18], dtype=np.float32)))
    # use-dulled zones carry a trace of grey film
    dn = t.fbm(0.12, 4, 0.5, seed=43)
    dull = t.blur(smoothstep(0.35, 0.90, dn), 0.006)
    base = base * (1 - 0.035 * dull)[..., None]

    mean = base.reshape(-1, 3).mean(0)
    albedo = base * (np.array(TARGET, dtype=np.float32) / mean)
    if TINT_TEST is not None:
        albedo = albedo * (np.array(TINT_TEST, dtype=np.float32) / np.array(TARGET, dtype=np.float32))
    albedo = np.clip(albedo, 0.01, 0.92)

    # ------------------------------------------------------------------ roughness
    rough = 0.075 + 0.025 * (peel - 0.5) * peel_z - 0.012 * (thick - 0.5)
    rough = rough + 0.20 * dull * (0.75 + 0.25 * t.fbm(0.012, 2, 0.5, seed=44))
    rough = rough + 0.07 * haze * (0.35 + 0.65 * dull) + 0.13 * scr
    rough = rough + 0.10 * crack
    rough = np.maximum(rough, 0.35 * sat(stain / 0.25))
    rough = np.maximum(rough, 0.6 * pin)
    rough = rough + 0.06 * skip
    rough = np.clip(rough, 0.05, 0.7)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
