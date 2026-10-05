"""Aged laid paper / parchment.

Construction: a hand-made sheet, formed on a wire mould (chain lines every ~25 mm, visible mostly
as relief), built from a felt of rag fibres with a cloudy formation.  A sheet that has been
handled keeps its history as a crumple network: nearly flat facets that tilt by a fraction of a
millimetre and meet in sharp hairline folds (mountain or valley, as the facets dictate), the folds
running from node to node.  Damp has cockled it gently and left foxing: mostly tiny tan-brown
spots in a few clusters, and a few faint toning patches with darker tide-line rims.
"""
from tk.tex import *

META = {"normal_strength": 2.2, "ao_radius": 0.003, "ao_strength": 0.5}

TARGET = (0.605, 0.548, 0.408)          # mean linear albedo: aged cream, greyer than yellow


def _crumple(t, n, jitter, amp, seed, fold_ref=0.012):
    """Piecewise-planar facet surface on a periodic triangulation (jittered n x n grid, random
    diagonals, random node heights +-amp m) plus a sharp fold line along every facet edge whose
    sign / strength follow the actual dihedral angle between the two facets.
    Returns (facet_height_m, fold_height_unit, fold_mask 0..1)."""
    rng = np.random.default_rng(seed)
    cw, ch = t.W / n, t.H / n
    jx = (rng.random((n, n)) - 0.5) * jitter
    jy = (rng.random((n, n)) - 0.5) * jitter
    hz = (rng.random((n, n)) * 2 - 1) * amp

    def vert(i, j):
        return ((i + 0.5 + jx[j % n, i % n]) * cw, (j + 0.5 + jy[j % n, i % n]) * ch, hz[j % n, i % n], (i % n, j % n))

    tris = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = vert(i, j), vert(i + 1, j), vert(i + 1, j + 1), vert(i, j + 1)
            if rng.random() < 0.5:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]

    Hh = np.full((t.h, t.w), np.nan, dtype=np.float32)
    edges = {}
    for tri in tris:
        (x0, y0, z0, _), (x1, y1, z1, _), (x2, y2, z2, _) = tri
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        gx = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / det
        gy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det
        ix0 = int(np.floor(min(x0, x1, x2) / t.px)) - 1
        ix1 = int(np.ceil(max(x0, x1, x2) / t.px)) + 1
        iy0 = int(np.floor((t.H - max(y0, y1, y2)) / t.py)) - 1
        iy1 = int(np.ceil((t.H - min(y0, y1, y2)) / t.py)) + 1
        xs = np.arange(ix0, ix1 + 1)
        ys = np.arange(iy0, iy1 + 1)
        Xl = ((xs + 0.5) * t.px)[None, :]
        Yl = (t.H - (ys + 0.5) * t.py)[:, None]
        l1 = ((Xl - x0) * (y2 - y0) - (x2 - x0) * (Yl - y0)) / det
        l2 = ((x1 - x0) * (Yl - y0) - (Xl - x0) * (y1 - y0)) / det
        inside = (l1 >= -1e-4) & (l2 >= -1e-4) & (l1 + l2 <= 1 + 1e-4)
        val = (z0 + gx * (Xl - x0) + gy * (Yl - y0)).astype(np.float32)
        sub = Hh[np.ix_(ys % t.h, xs % t.w)]
        Hh[np.ix_(ys % t.h, xs % t.w)] = np.where(inside, val, sub)
        for p, q, r in ((tri[0], tri[1], tri[2]), (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1])):
            key = (p[3], q[3]) if p[3] < q[3] else (q[3], p[3])
            edges.setdefault(key, []).append((p, q, r, gx, gy))
    Hh = np.where(np.isnan(Hh), 0.0, Hh)

    fold = t.zeros()
    mask = t.zeros()
    for key in sorted(edges):
        lst = edges[key]
        if len(lst) != 2:
            continue
        (p, q, r, gax, gay), (_, _, _, gbx, gby) = lst
        ex, ey = q[0] - p[0], q[1] - p[1]
        L = np.hypot(ex, ey)
        ex, ey = ex / L, ey / L
        nx_, ny_ = -ey, ex
        side = np.sign((r[0] - p[0]) * nx_ + (r[1] - p[1]) * ny_)       # which side facet a lies on
        ox, oy = -side * nx_, -side * ny_                                # from a towards b
        ang = (gax - gbx) * ox + (gay - gby) * oy                        # > 0: mountain fold
        u = rng.random()
        if u < 0.22:                                                     # a share of edges never creased sharply
            continue
        a = np.sign(ang) * (0.35 + 0.65 * min(1.0, abs(ang) / fold_ref)) * (0.55 + 0.45 * u)
        R = 0.008
        ix0 = int(np.floor((min(p[0], q[0]) - R) / t.px))
        ix1 = int(np.ceil((max(p[0], q[0]) + R) / t.px))
        iy0 = int(np.floor((t.H - max(p[1], q[1]) - R) / t.py))
        iy1 = int(np.ceil((t.H - min(p[1], q[1]) + R) / t.py))
        xs = np.arange(ix0, ix1 + 1)
        ys = np.arange(iy0, iy1 + 1)
        Xl = ((xs + 0.5) * t.px)[None, :] - p[0]
        Yl = (t.H - (ys + 0.5) * t.py)[:, None] - p[1]
        al = Xl * ex + Yl * ey
        ac = np.abs(Xl * nx_ + Yl * ny_)
        over = np.maximum(np.maximum(-al, al - L), 0.0)
        d = np.sqrt(ac * ac + over * over)
        s = np.clip(al / L, 0, 1)
        taper = 0.55 + 0.45 * np.clip(np.minimum(s, 1 - s) / 0.2, 0, 1)
        prof = 0.62 * np.exp(-d / 0.0006) + 0.38 * np.exp(-(d / 0.0024) ** 2)
        sl = np.ix_(ys % t.h, xs % t.w)
        fold[sl] += (a * prof * taper).astype(np.float32)
        mask[sl] = np.maximum(mask[sl], (abs(a) * np.exp(-d / 0.0007) * taper).astype(np.float32))
    return Hh, fold, mask


def _fibres(t, n, lmin, lmax, width, seed, curl=0.6):
    """Curved fibres: n arcs of length lmin..lmax m, about `width` m wide (anti-aliased)."""
    rng = np.random.default_rng(seed)
    out = t.zeros()
    for _ in range(n):
        cx, cy = rng.random() * t.W, rng.random() * t.H
        a = rng.random() * np.pi
        L = lmin + (lmax - lmin) * rng.random()
        c = (rng.random() * 2 - 1) * curl * 2.0 / L                       # sag ~ curl * L / 8
        c2 = (rng.random() * 2 - 1) * curl * 6.0 / (L * L)                # a little S-bend
        wd = max(width * (0.6 + 0.8 * rng.random()), t.px * 0.75)
        hw = int((L / 2) / t.px) + 3
        ix, iy = int(cx / t.px), int((t.H - cy) / t.py)
        xs = np.arange(ix - hw, ix + hw + 1)
        ys = np.arange(iy - hw, iy + hw + 1)
        Xl = ((xs + 0.5) * t.px)[None, :] - cx
        Yl = (t.H - (ys + 0.5) * t.py)[:, None] - cy
        al = Xl * np.cos(a) + Yl * np.sin(a)
        ac = -Xl * np.sin(a) + Yl * np.cos(a) - c * al * al - c2 * al * al * al
        m = np.clip(1 - np.abs(ac) / wd, 0, 1) * np.clip((L / 2 - np.abs(al)) / (L * 0.25), 0, 1)
        m = m * (0.35 + 0.65 * rng.random())
        sl = np.ix_(ys % t.h, xs % t.w)
        out[sl] = np.maximum(out[sl], m.astype(np.float32))
    return out


def _spots(t, density, size, seed, soft, wob):
    """Irregular round spots with a per-spot random strength.  Returns (mask, strength)."""
    lay = t.cells(max(1.0 / np.sqrt(density), t.px * 3), 1.0, seed=seed)
    r = size * 0.5 * (0.35 + 1.0 * lay.rand(3) ** 2)
    f = lay.f1 * (1 + wob * (t.fbm(size * 0.5, 2, 0.5, seed=seed + 1) - 0.5))
    return 1 - smoothstep(r * (1 - soft), r, f), lay.rand(5)


def build(t):
    # ------------------------------------------------------------------ sheet deformation (macro)
    f1, c1, m1 = _crumple(t, 7, 0.5, 0.00026, 7)                            # ~7 cm facets
    f2, c2, m2 = _crumple(t, 15, 0.6, 0.00009, 8, fold_ref=0.008)          # ~4 cm secondary crumple
    zone = smoothstep(0.42, 0.62, t.noise(0.14, seed=3, bandwidth=0.5))      # where the sheet was crushed finer
    facet = f1 + f2 * (0.25 + 0.75 * zone)
    foldu = c1 + 0.55 * c2 * (0.15 + 0.85 * zone)
    fmask = np.maximum(m1, 0.55 * m2 * (0.15 + 0.85 * zone))
    facet = t.warp(facet, 0.005, 0.17, seed=9)
    foldu = t.warp(foldu, 0.005, 0.17, seed=9)
    fmask = t.warp(fmask, 0.005, 0.17, seed=9)
    facet = t.blur(facet, 0.0006)

    cockle = (t.fbm(0.22, 2, 0.45, seed=4) - 0.5) * 0.0005                   # damp cockling, long and low
    ripple = (t.noise(0.03, seed=5, stretch=(1.0, 3.5)) - 0.5) * 0.00016     # drying waves
    macro_h = facet + 0.00019 * foldu + cockle + ripple

    # ------------------------------------------------------------------ mould structure (meso)
    # Chain lines ~25 mm apart (each wire a little off its pitch), seen as a thin spot in the
    # sheet; they come and go along their length.
    rng = np.random.default_rng(31)
    nch = 20
    pos = (np.arange(nch) + 0.5) * (t.W / nch) + (rng.random(nch) * 2 - 1) * 0.001
    xw = t.X + (t.fbm(0.15, 2, 0.5, seed=6) - 0.5) * 0.0012
    dd = xw[..., None] - pos[None, None, :]
    dd = np.abs((dd + t.W / 2) % t.W - t.W / 2)
    k = dd.argmin(-1)
    cdist = np.take_along_axis(dd, k[..., None], -1)[..., 0]
    fadeA = t.noise(0.05, seed=32, stretch=(0.3, 2.5), bandwidth=0.5)
    fadeB = t.noise(0.05, seed=33, stretch=(0.3, 2.5), bandwidth=0.5)
    fade = smoothstep(0.35, 0.65, np.where(k % 2 == 0, fadeA, fadeB))
    chain = np.exp(-(cdist / 0.0007) ** 2) * fade
    chain_sh = np.exp(-(cdist / 0.0035) ** 2) * fade                          # pulp heaps beside the wire

    # ------------------------------------------------------------------ fibres + formation (micro)
    form = t.fbm(0.0035, 3, 0.55, seed=10)                                   # cloudy pulp formation, 2-4 mm
    flock = t.fbm(0.012, 3, 0.5, seed=11)                                    # larger flocs
    felt = np.maximum(t.scratches(16000, 0.0045, 0.0004, seed=12), t.scratches(16000, 0.003, 0.0004, seed=13))
    fib_l = _fibres(t, 2200, 0.008, 0.015, 0.0006, 14)                       # long pale rag fibres
    fib_d = _fibres(t, 1500, 0.007, 0.014, 0.0006, 15)                      # long dark fibres / hairs
    shive = np.maximum(t.scratches(1600, 0.0025, 0.0007, seed=16), t.speckle(3500, 0.0012, seed=17) * 0.8)
    tooth = t.fbm(0.0016, 2, 0.6, seed=18)

    height = macro_h - 0.000045 * chain + 0.00002 * chain_sh
    height = height + (form - 0.5) * 0.00005 + (flock - 0.5) * 0.00004 + felt * 0.000018
    height = height + fib_l * 0.00003 + fib_d * 0.000025 + shive * 0.00003 + (tooth - 0.5) * 0.00003

    # ------------------------------------------------------------------ colour
    tone = 0.5 * t.fbm(0.20, 3, 0.5, seed=19) + 0.5 * flock
    base = ramp(level(tone), [(0.0, "#cfc3a6"), (0.35, "#d6cbb0"), (0.7, "#dcd2b9"), (1.0, "#e1d8c1")])
    base = base * (0.955 + 0.09 * form)[..., None]
    base = base * (1 - 0.02 * chain + 0.008 * chain_sh)[..., None]
    base = base * (1 + 0.04 * felt + 0.10 * fib_l - 0.125 * fib_d)[..., None]
    base = mix(base, base * hexlin("#8a6f4c") * 1.9, shive * 0.5)

    # toning patches left by damp: faint inside, a darker tide line at the rim
    tn = t.fbm(0.085, 4, 0.55, seed=20)
    thr = 0.70
    patch = smoothstep(thr - 0.01, thr + 0.02, tn)
    rim = np.exp(-((tn - thr) / 0.012) ** 2) * (0.5 + 0.5 * t.fbm(0.02, 2, 0.5, seed=21))
    tide = sat(0.35 * patch + 0.75 * rim)
    base = mix(base, base * hexlin("#b89f76") * 1.55, tide * 0.22)

    # foxing: hard-clustered, most spots tiny and faint, a few larger blooms with a darker core
    clus = smoothstep(0.56, 0.74, t.fbm(0.10, 3, 0.5, seed=22))
    clus = np.maximum(clus, patch * 0.6)
    s_t, k_t = _spots(t, 60000, 0.0022, 23, 0.7, 0.8)
    s_m, k_m = _spots(t, 9000, 0.0060, 25, 0.8, 1.0)
    s_l, k_l = _spots(t, 420, 0.0130, 27, 0.9, 1.2)
    fox = sat(s_t * k_t ** 2 * 0.85 * (0.04 + 0.96 * clus) + s_m * k_m ** 2 * 0.8 * (0.02 + 0.98 * clus)
              + s_l * k_l * 0.55 * clus)
    core = sat(s_m ** 3 * k_m ** 2 * clus + s_l ** 4 * k_l * clus * 0.6)
    fox = sat(fox * (0.55 + 0.9 * form) + t.blur(fox, 0.0025) * 0.45)
    base = mix(base, base * hexlin("#a8875c") * 1.62, fox * 0.5)
    base = mix(base, base * hexlin("#8f6f48") * 1.62, core * 0.35)

    # handling: valley folds hold a trace of grime, ridges are scuffed to a paler fuzz
    grime = smoothstep(0.45, 0.8, t.fbm(0.05, 3, 0.55, seed=28))
    ridge = sat(foldu) * fmask
    vall = sat(-foldu) * fmask
    base = base * (1 - 0.045 * sat(fmask * 1.2) * (0.35 + 0.65 * grime) - 0.03 * vall)[..., None]
    base = mix(base, base * 1.05 + 0.01, sat(ridge * 1.5) * 0.5)
    mean = base.reshape(-1, 3).mean(0)
    albedo = np.clip(base * (np.array(TARGET, dtype=np.float32) / mean), 0.02, 0.86)

    # ------------------------------------------------------------------ roughness
    sized = smoothstep(0.2, 0.8, 0.7 * t.fbm(0.09, 4, 0.55, seed=29) + 0.3 * (1 - flock))   # gelatine size / burnish left
    rough = 0.81 - 0.19 * sized * (0.6 + 0.4 * (1 - form)) + 0.10 * (form - 0.5) + 0.05 * (tooth - 0.5)
    rough = rough + 0.06 * felt + 0.07 * fib_l + 0.22 * sat(fmask * 1.4) + 0.05 * fox + 0.06 * tide - 0.03 * chain
    rough = np.clip(rough, 0.58, 0.97)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
