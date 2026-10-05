"""Jute hessian (burlap) sacking: coarse open plain weave of hairy, slubby single jute yarns.

68 x 68 yarns per 150 mm tile (2.2 mm pitch, ~11.5 threads per inch = ordinary 10 oz hessian).
Every yarn is a flattened cylinder with a crimped centreline; the cloth is the z-buffer (max) of
warp and weft, so the over/under checker, the open holes and the uneven spacing are geometric.
On top: S-twist striation in each yarn, slubs, and a layer of loose stray fibres (the hair)."""
from tk.tex import *

META = {"normal_strength": 1.0}

N = 68        # yarns per tile each way (even -> plain weave closes across the seam)


def _rnd(n, k):
    return np.random.default_rng(4711 + 7919 * k).random(n).astype(F)


def _yarns(t, c, along, wob, slub, rid, k0, pitch, sign):
    """One yarn system. c: across coordinate in yarn pitches, along: coordinate along in pitches.
    Returns z (m, -1 where empty), across (-1..1), id, crimp phase (+1 on top)."""
    pi = math.pi
    ic = np.floor(c)
    best = None
    for k in (-1, 0, 1):
        i = ic + k
        ii = i.astype(np.int64) % N
        par = np.where((i % 2) == 0, 1.0, -1.0) * sign
        off = (_rnd(N, k0)[ii] - 0.5) * 0.26                     # uneven sett: yarns bunch in pairs
        hw = pitch * 0.5 * (0.64 + 0.18 * _rnd(N, k0 + 1)[ii]) * (0.62 + 0.80 * slub) * (0.9 + 0.2 * rid)
        th = hw * 0.80                                           # jute yarn beds down flat-ish
        d = (c - (i + 0.5 + off)) * pitch + wob
        ph = np.cos(pi * (along - 0.5)) * par
        zc = 0.00042 * ph
        ins = np.abs(d) < hw
        zz = np.where(ins, zc + th * np.sqrt(np.maximum(1 - (d / hw) ** 2, 0)) - 0.0004, -1.0)
        cur = dict(z=zz, a=np.where(ins, d / hw, 0), id=ii.astype(F), ph=ph, tone=_rnd(N, k0 + 2)[ii])
        if best is None:
            best = cur
        else:
            w = zz > best["z"]
            best = {q: np.where(w, cur[q], best[q]) for q in cur}
    return best


def build(t):
    p = t.W / N
    # --- per-yarn wander / slubs: noise long along the yarn, about one yarn wide across it
    wob_w = (t.noise(0.0045, seed=1, stretch=(9, 1)) - 0.5) * 2 * 0.00042      # weft (runs along X)
    wob_p = (t.noise(0.0045, seed=2, stretch=(1, 9)) - 0.5) * 2 * 0.00042      # warp (runs along Y)
    slub_w = t.noise(0.0042, seed=3, stretch=(4.5, 1))
    slub_p = t.noise(0.0042, seed=4, stretch=(1, 4.5))
    rag = t.fbm(0.0009, 2, 0.6, seed=5)                                        # ragged hairy edges
    # cloth distorts as a whole (bias pull)
    gx = (t.noise(0.06, seed=6) - 0.5) * 2 * 0.0016
    gy = (t.noise(0.06, seed=7) - 0.5) * 2 * 0.0016
    u = (t.X + gx) / p
    v = (t.Y + gy) / p

    weft = _yarns(t, v, u, wob_w, slub_w, rag, 10, p, +1.0)
    warp = _yarns(t, u, v, wob_p, slub_p, rag, 20, p, -1.0)
    floor_z = -0.0011
    isw = weft["z"] >= warp["z"]
    z = np.maximum(weft["z"], warp["z"])
    yarn = (z > floor_z).astype(F)
    acr = np.where(isw, weft["a"], warp["a"])
    ph = np.where(isw, weft["ph"], warp["ph"])
    tone = np.where(isw, weft["tone"], warp["tone"])
    yid = np.where(isw, weft["id"], warp["id"] + N)

    # --- yarn structure: S-twist (fibres lie ~25 deg to the yarn axis) + individual fibres
    tw = 2 * math.pi
    tw_w = np.sin(tw * (t.X / t.W * 150 + weft["a"] * 0.55 + weft["id"] * 0.37))
    tw_p = np.sin(tw * (t.Y / t.H * 150 + warp["a"] * 0.55 + warp["id"] * 0.37))
    twist = np.where(isw, tw_w, tw_p)
    fib_w = t.fbm(0.0011, 3, 0.65, seed=8, stretch=(7, 1))
    fib_p = t.fbm(0.0011, 3, 0.65, seed=9, stretch=(1, 7))
    fib = np.where(isw, fib_w, fib_p)
    fibd = t.sample(fib, t.X + acr * 0.0004 * np.where(isw, 1, 0), t.Y + acr * 0.0004 * np.where(isw, 0, 1))
    lump = t.fbm(0.003, 3, 0.55, seed=10)

    height = z + yarn * (twist * 0.000035 * (1 - acr * acr) + (fibd - 0.5) * 0.00016 + (lump - 0.5) * 0.00012)
    holen = t.fbm(0.002, 3, 0.6, seed=11)
    height = np.where(yarn > 0.5, height, floor_z + (holen - 0.5) * 0.00025)

    # --- hair: loose bast fibres lying over the cloth and bridging the holes
    hair = np.maximum(
        t.scratches(density=170000, length=0.0050, width=0.00011, seed=12),
        t.scratches(density=170000, length=0.0090, width=0.00010, seed=13))
    hair = t.warp(hair, 0.0007, 0.006, seed=14)
    hair2 = t.warp(t.scratches(density=170000, length=0.0035, width=0.00010, seed=15), 0.0005, 0.004, seed=16)
    hair = sat(np.maximum(hair, hair2 * 0.8))
    hair_top = np.maximum(t.blur(np.maximum(height, floor_z + 0.0006), 0.0012), height) + 0.00010
    height = mix(height, np.maximum(height, hair_top), hair)

    # soft handling rumples of the cloth itself
    rumple = t.fbm(0.07, 3, 0.45, seed=17)
    fold = t.ridged(0.05, 2, 0.5, seed=21)                        # soft slack folds of loose sacking
    height = height + (rumple - 0.5) * 0.0032 + (fold - 0.5) * 0.0014
    height = mix(height, t.blur(height, t.px * 0.7), 0.5)

    # --- masks
    crown = sat(1 - acr * acr) ** 0.6 * yarn
    top = smoothstep(0.2, 1.0, ph) * crown                        # yarn riding over a crossing
    cav = t.cavity(z, 0.0010, 0.0005) * yarn

    # --- colour: raw jute, golden grey-brown; each yarn a little different, fibres lighter/darker
    macro = t.fbm(0.08, 3, 0.5, seed=18)
    ydr = np.where(isw, level(t.noise(0.0035, seed=30, stretch=(22, 1))), level(t.noise(0.0035, seed=31, stretch=(1, 22))))
    tn = sat(0.5 + (tone - 0.5) * 0.30 + (ydr - 0.5) * 0.30 + (fibd - 0.5) * 0.75 + (lump - 0.5) * 0.35 + (macro - 0.5) * 0.18)
    base = ramp(tn, [(0.0, "#5c4a30"), (0.25, "#7a6543"), (0.5, "#927b55"),
                     (0.75, "#a58e66"), (1.0, "#b9a47c")])
    # retted-fibre colour drift: some yarns greyer, some redder-brown
    base = mix(base, desaturate(base, 0.40) * 0.98, sat((0.28 - ydr) * 3.0)[..., None])
    base = mix(base, base * hexlin("#ecd0b4") * 1.15, sat((ydr - 0.74) * 3.0)[..., None])
    base = base * (0.93 + 0.07 * twist * (1 - acr * acr))[..., None]
    base = base * (0.78 + 0.22 * crown)[..., None]                # dust sits in the flanks
    base = mix(base, base * hexlin("#8a7458"), sat(cav) * 0.6)
    base = mix(base, base * 1.10 + 0.01, top * 0.45)              # rubbed crowns paler
    hole_col = ramp(holen, [(0, "#2a2218"), (1, "#3b3023")])
    albedo = mix(hole_col, base, yarn)
    hair_col = ramp(t.noise(0.01, seed=19), [(0.2, "#9c8762"), (0.8, "#c9b78f")])
    albedo = mix(albedo, hair_col, hair * 0.85)

    # --- roughness: dry bast fibre, faint silky sheen on the crowns and on loose fibres
    rough = 0.90 - 0.13 * top - 0.05 * crown + 0.10 * (fibd - 0.5) + 0.06 * (macro - 0.5)
    rough = mix(0.98, rough, yarn)
    rough = mix(rough, 0.72, hair * 0.7)

    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0)
