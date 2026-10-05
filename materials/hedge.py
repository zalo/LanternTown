"""Clipped box / privet hedge face: thousands of small oval leaves, built the way the plant
builds them - short twigs carrying opposite pairs of leaves, each leaf with its own pitch, roll
and cupping - z-buffered in depth from the clipped outer shell down into the dark, twiggy
interior. Glossy upper sides, paler matt undersides, bright young tip growth, a few yellowed
leaves. Tintable: hue variation is kept relative (lighter / darker, a little warmer / cooler)."""
from tk.tex import *

META = {"normal_strength": 1.0}

DEPTH = 0.034          # how far into the hedge you can see (m)


def _leaves(t, rng, n_sprigs, z_rng, len_rng, young=0.0, flush=None):
    """Generate the leaf list for one depth layer. Returns an (N, 12) array:
    cx, cy, angle, length, width, z0, pitch, roll, cup, tone, kind, twist."""
    out = []
    g = int(np.ceil(np.sqrt(n_sprigs)))
    order = rng.permutation(g * g)[:n_sprigs]
    for c in order:
        sx = ((c % g) + rng.random()) / g * t.W
        sy = ((c // g) + rng.random()) / g * t.H
        sa = rng.random() * 6.283                    # twig direction on the hedge face
        fore = rng.uniform(0.25, 1.0)                # twig points more or less out at the viewer
        z0 = rng.uniform(*z_rng)
        npair = rng.integers(2, 5)
        L0 = rng.uniform(*len_rng)
        s_tone = rng.random()
        is_young = rng.random() < young
        if flush is not None:
            fv = flush[int((t.H - sy) / t.py) % t.h, int(sx / t.px) % t.w]
            is_young = is_young and fv > 0.52
        step = rng.uniform(0.008, 0.013) * fore
        for j in range(npair):
            along = (j - (npair - 1) * 0.5) * step
            px_, py_ = sx + math.cos(sa) * along, sy + math.sin(sa) * along
            spread = rng.uniform(0.7, 1.35)          # leaf angle off the twig (rad)
            flip = (j % 2) * 1.5708 * (1 - fore)     # decussate pairs
            for side in (-1, 1):
                la = sa + side * spread + flip + rng.normal(0, 0.18)
                L = L0 * (1.0 - 0.10 * j) * rng.uniform(0.82, 1.12)
                pitch = rng.normal(0.0, 0.27) + (1 - fore) * 0.2      # leaves on outward twigs tip down at the tip
                roll = rng.normal(0.0, 0.26)
                pitch = float(np.clip(pitch, -1.05, 1.05))
                roll = float(np.clip(roll, -0.95, 0.95))
                Lp = L * math.cos(pitch)
                W = L * rng.uniform(0.42, 0.56)
                # the leaf is attached at its base: centre sits half a (projected) length out
                cx = px_ + math.cos(la) * Lp * 0.5
                cy = py_ + math.sin(la) * Lp * 0.5
                zc = z0 + j * 0.0012 + rng.normal(0, 0.0015) + math.tan(pitch) * Lp * 0.5 * 0.6
                tone = float(np.clip(s_tone * 0.55 + rng.random() * 0.45, 0, 1))
                kind = 0.0
                r = rng.random()
                if is_young and j >= npair - 2:
                    kind = 1.0                       # fresh tip growth
                elif r < 0.014:
                    kind = 2.0                       # yellowing / tired leaf
                elif roll > 0.6 or r < 0.07:
                    kind = 3.0                       # underside showing
                out.append((cx, cy, la, L, W, zc, pitch, roll, rng.uniform(0.1, 0.5), tone, kind, rng.normal(0, 0.12)))
    return np.array(out, dtype=np.float64)


def _raster(t, leaves, zbuf, attr):
    """Z-buffered rasterisation of leaf sprites. attr: dict of per-pixel arrays that get written."""
    px, py = t.px, t.py
    for (cx, cy, la, L, W, zc, pitch, roll, cup, tone, kind, twist) in leaves:
        cp, cr = math.cos(pitch), math.cos(roll)
        hl = L * 0.5 * cp                 # projected half length
        hwid = W * 0.5 * cr               # projected half width
        R = max(hl, hwid) + 1.5 * px
        n = int(R / px) + 1
        ix, iy = int(cx / px), int((t.H - cy) / py)
        xs, ys = np.arange(ix - n, ix + n + 1), np.arange(iy - n, iy + n + 1)
        X = (xs[None, :] + 0.5) * px - cx
        Y = (t.H - (ys[:, None] + 0.5) * py) - cy
        c, s = math.cos(la), math.sin(la)
        al = (X * c + Y * s) / hl                     # -1 (base) .. 1 (tip)
        ac = (-X * s + Y * c) / hwid                  # -1 .. 1 across
        ac = ac - twist * (al * al - 0.3)             # slight sickle bend
        # box leaf outline: oval, widest just below the middle, blunt tip
        a2 = np.clip(1.0 - al * al, 0.0, 1.0)
        half = a2 ** 0.55 * (1.0 - 0.16 * al)
        e = half - np.abs(ac)                         # > 0 inside (in half-width units)
        inside = (e > 0.0) & (np.abs(al) < 1.0)
        if not inside.any():
            continue
        q = np.clip(np.abs(ac) / np.maximum(half, 1e-4), 0.0, 1.0)     # 0 midrib .. 1 margin
        # leaf surface height: plane (pitch, roll) + cupping (margins curl down) + arch along the length
        z = (zc + al * hl * math.tan(pitch) + ac * hwid * math.tan(roll)
             - cup * W * 0.22 * q * q - 0.10 * L * cp * al * al * 0.35
             - 0.00018 * np.exp(-(q / 0.12) ** 2))                     # the midrib is a fine groove
        sel = np.ix_(ys % t.h, xs % t.w)
        zb = zbuf[sel]
        win = inside & (z > zb)
        if not win.any():
            continue
        zbuf[sel] = np.where(win, z, zb)
        for k, v in (("al", al), ("q", q), ("edge", e * hwid)):
            a = attr[k][sel]
            attr[k][sel] = np.where(win, v, a)
        for k, v in (("tone", tone), ("kind", kind), ("pitch", pitch), ("rnd", (tone * 7.77 + L * 913.0) % 1.0)):
            a = attr[k][sel]
            attr[k][sel] = np.where(win, v, a)


def _twigs(t, rng, n, zbuf, twig):
    """Thin woody stems down in the interior (only seen through the gaps)."""
    px = t.px
    for _ in range(n):
        cx, cy, a = rng.random() * t.W, rng.random() * t.H, rng.random() * 3.1416
        L, wd, z0 = rng.uniform(0.02, 0.06), rng.uniform(0.0010, 0.0022), -DEPTH * rng.uniform(0.72, 0.98)
        n_ = int((L / 2 + wd) / px) + 2
        ix, iy = int(cx / px), int((t.H - cy) / t.py)
        xs, ys = np.arange(ix - n_, ix + n_ + 1), np.arange(iy - n_, iy + n_ + 1)
        X = (xs[None, :] + 0.5) * px - cx
        Y = (t.H - (ys[:, None] + 0.5) * t.py) - cy
        al = X * math.cos(a) + Y * math.sin(a)
        ac = -X * math.sin(a) + Y * math.cos(a)
        m = (np.abs(ac) < wd) & (np.abs(al) < L / 2)
        z = z0 + np.sqrt(np.clip(wd * wd - ac * ac, 0, None))
        sel = np.ix_(ys % t.h, xs % t.w)
        zb = zbuf[sel]
        win = m & (z > zb)
        zbuf[sel] = np.where(win, z, zb)
        twig[sel] = np.where(win, z, twig[sel])


def build(t):
    rng = np.random.default_rng(1848)
    zbuf = t.zeros(-DEPTH)
    attr = {k: t.zeros() for k in ("al", "q", "edge", "tone", "kind", "pitch", "rnd")}
    twig = t.zeros(-1.0)
    _twigs(t, rng, 140, zbuf, twig)
    # the shell follows the shears only roughly: the outer surface billows by a few mm
    area = t.W * t.H
    flush = t.fbm(0.12, 2, 0.5, seed=9)                 # new growth comes in flushes, not as confetti
    deep = _leaves(t, rng, int(800 * area), (-DEPTH * 0.82, -DEPTH * 0.52), (0.020, 0.028))
    midl = _leaves(t, rng, int(720 * area), (-DEPTH * 0.50, -DEPTH * 0.26), (0.021, 0.030), young=0.25, flush=flush)
    top = _leaves(t, rng, int(640 * area), (-DEPTH * 0.20, -0.002), (0.022, 0.032), young=0.6, flush=flush)
    allv = np.concatenate([deep, midl, top])
    # billow: push whole regions of the shell in / out so the clipped face is not a plane
    bil = t.fbm(0.14, 3, 0.5, seed=1)
    ixs = (allv[:, 0] / t.px).astype(int) % t.w
    iys = ((t.H - allv[:, 1]) / t.py).astype(int) % t.h
    allv[:, 5] += (bil[iys, ixs] - 0.5) * 0.010
    _raster(t, allv, zbuf, attr)

    twig = (np.abs(zbuf - twig) < 1e-7).astype(F)        # twigs only where no leaf covers them
    leaf = (zbuf > -DEPTH + 1e-5) & (twig < 0.5)
    leaf_m = leaf.astype(F)
    depth = sat(-zbuf / DEPTH)                           # 0 at the shell, 1 at the back
    height = zbuf.copy()
    # interior: nothing is flat in there - give the back a rough, twiggy bed
    back = (t.fbm(0.01, 3, 0.6, seed=2) - 0.5) * 0.004
    height = np.where(leaf | (twig > 0.5), height, -DEPTH + back)
    # leaf surface micro-relief: faint blistering of the blade
    height = height + leaf_m * (t.fbm(0.003, 2, 0.5, seed=3) - 0.5) * 0.00012
    height = t.blur(height, t.px * 0.55)

    al, q, tone, kind, rnd = attr["al"], attr["q"], attr["tone"], attr["kind"], attr["rnd"]
    # ---- colour
    mott = t.fbm(0.004, 3, 0.55, seed=4)
    patch = t.fbm(0.2, 3, 0.5, seed=5)
    tt = sat(0.12 + 0.62 * tone + 0.14 * (mott - 0.5) + 0.14 * (patch - 0.5) - 0.20 * depth)
    mature = ramp(tt, [(0.0, "#24351b"), (0.3, "#324920"), (0.6, "#405a26"), (1.0, "#526b2e")])
    young = ramp(tt, [(0.0, "#4a622a"), (0.5, "#5c752f"), (1.0, "#718838")])
    tired = ramp(tt, [(0.0, "#5e6129"), (0.5, "#757331"), (1.0, "#857a3a")])
    under = ramp(tt, [(0.0, "#42522e"), (0.5, "#55653a"), (1.0, "#6a7848")])
    albedo = mature
    albedo = mix(albedo, young, (kind == 1.0).astype(F))
    albedo = mix(albedo, tired, (kind == 2.0).astype(F))
    albedo = mix(albedo, under, (kind == 3.0).astype(F))
    # midrib: a pale line on top, a strong pale keel underneath; margin a touch lighter and thicker
    rib = np.exp(-(q / 0.17) ** 2) * smoothstep(-1.0, -0.5, -al) * leaf_m
    albedo = mix(albedo, albedo * hexlin("#f2f0b0") * 1.8, rib * 0.55)
    # the blade is thinner and a shade darker towards the margin, with faint side veins
    side = 0.5 + 0.5 * np.cos((al * 7.0 - q * 3.0) * 3.1416)
    albedo = albedo * (1.06 - 0.16 * smoothstep(0.2, 0.95, q) - 0.05 * side * smoothstep(0.15, 0.4, q))[..., None]
    rim = (1 - smoothstep(0.0, 0.0012, attr["edge"])) * leaf_m
    albedo = mix(albedo, albedo * hexlin("#f0e8a8") * 1.35, rim * 0.35)
    # leaf base (petiole end) slightly yellower, tips of a few leaves scorched brown
    albedo = mix(albedo, albedo * hexlin("#fff2a0") * 1.15, smoothstep(-0.5, -1.0, al) * leaf_m * 0.35)
    scorch = smoothstep(0.72, 0.98, al) * smoothstep(0.86, 0.93, rnd) * leaf_m
    albedo = mix(albedo, hexlin("#5a4426"), scorch * 0.8)
    # depth: leaves further in are older, dustier and stand in each other's shade
    albedo = albedo * (1.0 - 0.50 * smoothstep(0.15, 0.95, depth))[..., None]
    twig_c = ramp(mott, [(0.0, "#2a2118"), (1.0, "#4a3c2a")])
    gap_c = ramp(t.fbm(0.012, 3, 0.6, seed=6), [(0.0, "#0f120b"), (1.0, "#1a1c12")])
    albedo = np.where(leaf[..., None], albedo, np.where((twig > 0.5)[..., None], twig_c, gap_c))

    # ---- roughness: waxy cuticle on top, matt beneath, dull in the interior
    gloss = 0.30 + 0.26 * rnd + 0.10 * (mott - 0.5)
    rough = gloss
    rough = np.where(kind == 1.0, rough - 0.05, rough)
    rough = np.where(kind == 2.0, rough + 0.18, rough)
    rough = np.where(kind == 3.0, 0.68 + 0.08 * (mott - 0.5), rough)
    rough = rough + 0.12 * rib + 0.25 * scorch + 0.22 * smoothstep(0.3, 1.0, depth)
    rough = np.where(leaf, rough, 0.92)
    rough = np.clip(rough, 0.25, 0.96)

    # ---- AO: the hedge is self-shadowing foliage; combine fine leaf-on-leaf and broad depth occlusion
    ao = t.ao(height, radius=0.006, strength=0.9) * (1.0 - 0.45 * smoothstep(0.1, 1.0, depth))
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0, ao=sat(ao))
