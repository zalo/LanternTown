"""One broad leaf blade, upper side, filling the tile. Base at U = 0, tip at U = 1, midrib along U
through the middle of the tile; the tile's top / bottom edges are the margin zone.
Built like a leaf: midrib -> alternate secondary veins that leave it at ~50 degrees and sweep
towards the tip, thinning out before the margin -> cross (ladder) tertiaries between neighbouring
secondaries -> polygonal areoles -> cell-scale texture. Veins are impressed, the blade is quilted
up between them. Tintable: veins / blade differ in value and warmth, not in a fixed hue."""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.0035, "ao_strength": 0.7}

NSEC = 8                 # secondary veins per side over the tile length


def _wrap(d, L):
    return (d + 0.5 * L) % L - 0.5 * L


def build(t):
    rng = np.random.default_rng(2718)
    W, H = t.W, t.H
    # gentle organic wobble of the whole vein system (a few tenths of a mm .. 1.5 mm)
    wob_x = (t.noise(0.04, seed=1) - 0.5) * 0.007 + (t.noise(0.016, seed=2) - 0.5) * 0.0007
    wob_y = (t.noise(0.05, seed=3) - 0.5) * 0.004
    X = t.X + wob_x
    v = (t.Y - 0.5 * H) + wob_y * np.sin(6.283 * t.Y / H) ** 2      # no wobble at the midrib rows' wrap partner
    s = np.abs(v)                                                    # distance from the midrib
    u01 = (t.X / W) % 1.0

    # ---- midrib: thick at the base, fine at the tip (taper wraps through a short swell at the base)
    taper = (1.0 - u01) * smoothstep(0.0, 0.06, u01)
    wm = 0.0009 + 0.0017 * taper                                   # half width
    d_mid = s

    # ---- secondary veins, both sides, alternate
    sp = W / NSEC
    d_sec = np.full(X.shape, 1e3, dtype=F)
    w_sec = t.zeros()
    behind = np.full(X.shape, 1e3, dtype=F)
    ahead = np.full(X.shape, 1e3, dtype=F)
    bid = np.zeros(X.shape, dtype=np.int32)
    fade_end = 0.118
    for side in (0, 1):
        on = (v >= 0) if side == 0 else (v < 0)
        for i in range(NSEC):
            u0 = (i + (0.5 if side else 0.0) + rng.uniform(-0.14, 0.14)) * sp
            cot = rng.uniform(0.50, 0.68)
            cur = rng.uniform(2.6, 4.4)
            w0 = rng.uniform(0.00075, 0.00100)
            run = s * cot + cur * s * s
            du = _wrap(X - u0 - run, W)
            dperp = np.abs(du) / np.sqrt(1.0 + (cot + 2 * cur * s) ** 2)
            better = on & (dperp < d_sec)
            d_sec = np.where(better, dperp, d_sec)
            w_sec = np.where(better, w0, w_sec)
            # interval bookkeeping for the ladder veins: nearest vein behind (towards base) / ahead
            pos = np.where(du >= 0, du, du + W)                      # distance ahead of this vein
            neg = W - pos
            b2 = on & (pos < behind)
            behind = np.where(b2, pos, behind)
            bid = np.where(b2, i + side * NSEC, bid)
            ahead = np.where(on & (neg < ahead), neg, ahead)
    thin = 1.0 - 0.70 * smoothstep(0.0, 0.11, s)                     # veins thin towards the margin
    live = 1.0 - smoothstep(0.084, fade_end, s)                      # ... and dissolve into the network
    w_sec = w_sec * thin
    # where a secondary joins the midrib it flares slightly
    w_sec = w_sec * (1.0 + 0.6 * np.exp(-(s / 0.004) ** 2))

    # ---- tertiary "ladder" veins: rungs between neighbouring secondaries, roughly square to them
    frac = behind / np.maximum(behind + ahead, 1e-6)                 # 0 at the vein behind .. 1 at the next
    ph = np.random.default_rng(99).random(2 * NSEC + 1).astype(F)[bid]
    rsp = 0.0060
    rr = s / rsp + ph + 2.2 * frac + (t.noise(0.03, seed=4) - 0.5) * 0.9 + (t.noise(0.008, seed=16) - 0.5) * 0.35
    d_rung = np.abs(rr - np.floor(rr) - 0.5) * rsp
    net = t.cells(0.0085, jitter=1.0, seed=5)
    d_net = t.warp(net.dist, 0.0012, 0.02, seed=6)
    d_ter = mix(d_rung, d_net, smoothstep(0.075, 0.105, s))
    d_ter = np.where(d_mid < 0.004, 1.0, d_ter)
    # ---- areoles: the smallest closed meshes
    are = t.cells(0.0034, jitter=1.0, seed=7)
    d_are = t.warp(are.dist, 0.0004, 0.01, seed=15)

    # ---- vein masks (1 on the vein)
    m_mid = np.exp(-(d_mid / wm) ** 2)
    m_sec = np.exp(-(d_sec / np.maximum(w_sec, 1e-5)) ** 2) * live
    marg = smoothstep(0.075, 0.105, s)                               # margin zone: only a faint mesh
    m_ter = np.exp(-(d_ter / 0.00034) ** 2) * (0.55 + 0.45 * t.noise(0.02, seed=8)) * (1.0 - 0.55 * marg)
    m_are = np.exp(-(d_are / 0.00016) ** 2) * (0.5 + 0.5 * t.noise(0.01, seed=9))

    # ---- height (m)
    # blade quilted up between the secondaries and away from the midrib
    panel = sat(np.minimum(d_sec, d_mid * 0.8) / (sp * 0.33))
    panel = 1 - (1 - panel) ** 2.2
    quilt = mix(0.35 * sat(d_mid / 0.01), panel, live)
    pane3 = sat(d_ter / 0.0032)
    pane3 = 1 - (1 - pane3) ** 2
    height = quilt * 0.00090 + pane3 * 0.00013 * (1.0 - 0.5 * marg) + sat(d_are / 0.0009) * 0.000030
    # the midrib: a cord lying in a shallow channel; secondaries are finer cords in creases
    chan = np.exp(-(d_mid / (wm * 2.6)) ** 2)
    height += -0.00085 * chan + 0.00095 * m_mid * (0.55 + 0.45 * taper)
    crease = np.exp(-(d_sec / np.maximum(w_sec * 2.6, 1e-5)) ** 2) * live
    height += -0.00036 * crease + 0.00026 * m_sec
    # the blade is never flat: a shallow keel fold and broad waviness, strongest towards the margin
    height += 0.0011 * (0.5 - 0.5 * np.cos(6.283 * (t.Y - 0.5 * H) / H))
    height += (t.fbm(0.07, 3, 0.5, seed=10) - 0.5) * 0.0009 * (0.4 + 0.6 * smoothstep(0.02, 0.11, s))
    # cell-scale texture and a little hard grit on top
    cellt = t.fbm(0.0011, 2, 0.6, seed=11)
    height += (cellt - 0.5) * 0.000014

    # ---- blemishes: a few pin galls / necrotic specks with a chlorotic halo, tiny dust
    spot = t.speckle(density=420, size=0.0026, seed=12, soft=0.5)
    halo = sat(t.blur(spot, 0.0016) * 3.0)
    height -= spot * 0.00006
    dust = t.speckle(density=9000, size=0.0006, seed=13, soft=0.6) * smoothstep(0.5, 0.7, t.fbm(0.05, 2, seed=14))

    # ---- colour
    macro = t.fbm(0.11, 3, 0.5, seed=20)
    mott = t.fbm(0.006, 3, 0.55, seed=21)
    tone = sat(0.42 + 0.34 * (macro - 0.5) + 0.22 * (mott - 0.5) + 0.30 * (quilt - 0.5) + 0.10 * (are.rand(1) - 0.5)
               - 0.10 * smoothstep(0.09, 0.125, s))
    albedo = ramp(tone, [(0.0, "#27401c"), (0.35, "#34521f"), (0.65, "#416226"), (1.0, "#52732e")])
    # blade next to the big veins is a touch deeper green (thicker tissue)
    albedo = albedo * (1.0 - 0.16 * crease - 0.14 * chan)[..., None]
    vein_c = hexlin("#93a55e")
    albedo = mix(albedo, mix(albedo, vein_c, 0.30), m_are * 0.4)
    albedo = mix(albedo, mix(albedo, vein_c, 0.55), m_ter * 0.55)
    albedo = mix(albedo, vein_c * 0.92, m_sec * 0.8)
    albedo = mix(albedo, hexlin("#a3b06c"), m_mid * 0.9)
    albedo = mix(albedo, albedo * hexlin("#fff09a") * 1.25, halo * 0.35)
    albedo = mix(albedo, hexlin("#4a3520"), spot * 0.8)
    albedo = mix(albedo, albedo * 1.12 + 0.01, dust * 0.5)

    # ---- roughness: waxy cuticle with a slight gloss on the swollen panels, duller veins and specks
    sheen = t.fbm(0.03, 3, 0.5, seed=22)
    rough = 0.40 - 0.08 * quilt + 0.10 * (sheen - 0.5) + 0.05 * (cellt - 0.5)
    rough = rough + 0.12 * m_ter + 0.16 * m_sec + 0.16 * m_mid + 0.3 * spot + 0.25 * dust + 0.06 * crease
    rough = np.clip(rough, 0.24, 0.85)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
