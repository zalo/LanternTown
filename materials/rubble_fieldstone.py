"""Random rubble wall: mixed grey / ochre field stones bedded in deep, struck-back lime mortar.

Built like the wall: stones of four size classes are placed big-first (a mason sets the large
stones, then fits smaller ones and pinnings into the gaps), each lying roughly on its flat with
its own length, lean and lumpy outline. Stones are packed with a size-weighted distance so big
stones keep convex faces and little ones are squeezed between them. Each stone then rises as its
own rounded, faceted lump out of a rough mortar bed 15-30 mm behind the faces.
"""
from tk.tex import *

META = {"normal_strength": 1.0}

# grey granite / gneiss, blue-grey, ochre sandstone, rusty brown, pale, dark
PALETTE = ["#84817a", "#7a7872", "#8d8a82", "#6f7375", "#958567", "#9d8d6e", "#877761",
           "#7d6a59", "#a39c8c", "#989183", "#66645f", "#8e836b"]
GRANITIC = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]             # speckled crystalline stones
BEDDED = [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]               # stones that show bedding / banding


AX, AY = 1.25, 0.80            # stones lie on their flat: longer than tall


def lay_stones(t, seed=7):
    rng = np.random.default_rng(seed)
    W, H = t.W, t.H
    s = W / 2.0
    classes = [(0.150, 0.185, 9), (0.110, 0.140, 26), (0.075, 0.100, 55), (0.042, 0.062, 130)]
    pts = []
    for rmin, rmax, n in classes:
        fails = placed = 0
        while placed < n and fails < 2500:
            x, y, r = rng.random() * W, rng.random() * H, (rmin + rng.random() * (rmax - rmin)) * s
            ok = True
            for (qx, qy, qr) in pts:
                dx = abs(x - qx)
                dx = min(dx, W - dx)
                dy = abs(y - qy)
                dy = min(dy, H - dy)
                if (dx / AX) ** 2 + (dy / AY) ** 2 < (0.76 * (r + qr)) ** 2:
                    ok = False
                    break
            if ok:
                pts.append((x, y, r))
                placed += 1
            else:
                fails += 1
    n = len(pts)
    shp = (t.h, t.w)
    p1 = np.full(shp, 9.0, dtype=F)
    p2 = np.full(shp, 9.0, dtype=F)
    i1 = np.zeros(shp, dtype=np.int64)
    i2 = np.zeros(shp, dtype=np.int64)
    own = np.full(shp, 9.0, dtype=F)
    lu = np.zeros(shp, dtype=F)
    lv = np.zeros(shp, dtype=F)
    for i, (x, y, r) in enumerate(pts):
        asp = 0.9 + 0.7 * rng.random()                       # on top of the common "lying flat" stretch
        phi = (rng.random() - 0.5) * 0.5
        m2, q2 = 0.03 + 0.05 * rng.random(), rng.random() * 6.283
        m3, q3 = 0.03 + 0.05 * rng.random(), rng.random() * 6.283
        m5, q5 = 0.01 + 0.02 * rng.random(), rng.random() * 6.283
        ext = r * 2.6
        hwp = int(ext * AX / t.px) + 2
        hhp = int(ext * AY / t.py) + 2
        ix, iy = int(x / t.px), int((H - y) / t.py)
        xs = np.arange(ix - hwp, ix + hwp + 1)
        ys = np.arange(iy - hhp, iy + hhp + 1)
        dx = ((xs[None, :] + 0.5) * t.px - x).astype(F)
        dy = ((H - (ys[:, None] + 0.5) * t.py) - y).astype(F)
        pw = (dx / AX) ** 2 + (dy / AY) ** 2 - r * r           # power distance: straight shared faces
        c, sn = math.cos(phi), math.sin(phi)
        u = dx * c + dy * sn
        v = -dx * sn + dy * c
        eu, ev = u / (AX * math.sqrt(asp)), v * math.sqrt(asp) / AY
        th = np.arctan2(ev, eu)
        d = np.sqrt(eu * eu + ev * ev) / r
        d = d * (1 + m2 * np.cos(2 * th + q2) + m3 * np.cos(3 * th + q3) + m5 * np.cos(5 * th + q5))
        sel = np.ix_(ys % t.h, xs % t.w)
        a1, a2, j1, j2 = p1[sel], p2[sel], i1[sel], i2[sel]
        closer = pw < a1
        second = (~closer) & (pw < a2)
        p2[sel] = np.where(closer, a1, np.where(second, pw, a2))
        i2[sel] = np.where(closer, j1, np.where(second, i, j2))
        p1[sel] = np.where(closer, pw, a1)
        i1[sel] = np.where(closer, i, j1)
        own[sel] = np.where(closer, (1.50 - d) * r, own[sel])
        lu[sel] = np.where(closer, u, lu[sel])
        lv[sel] = np.where(closer, v, lv[sel])
    P = np.array(pts, dtype=np.float64)
    ddx = np.abs(P[i1, 0] - P[i2, 0])
    ddx = np.minimum(ddx, W - ddx) / AX
    ddy = np.abs(P[i1, 1] - P[i2, 1])
    ddy = np.minimum(ddy, H - ddy) / AY
    sep = np.sqrt(ddx * ddx + ddy * ddy) + 1e-4
    delta = ((p2 - p1) / (2.0 * sep)).astype(F)             # metres from the face shared with the neighbour
    lay = Layout(np.minimum(delta, own), i1, lu, lv, n, seed, 0.2, 0.2)
    lay.r = P[i1, 2].astype(F)
    return lay


def build(t):
    lay = lay_stones(t)
    R = lay.rand
    r = lay.r
    ox, oy = R(1) * t.W, R(2) * t.H                          # per-stone offsets: no texture flows across stones

    # ------------------------------------------------------------------ stone outlines
    dist = (lay.dist + (t.noise(0.14, seed=8) - 0.5) * 0.030 * sat(r / 0.16)
            + (t.noise(0.06, seed=1) - 0.5) * 0.014 + (t.noise(0.022, seed=2) - 0.5) * 0.006)
    lay2 = Layout(np.maximum(dist, 0.0), lay.id, lay.u, lay.v, lay.n, lay.seed, 0.2, 0.2).rounded(t, 0.018)
    gap = 0.004 + 0.007 * R(3)                               # half joint 4-11 mm
    e = lay2.dist - gap                                      # metres inside the stone outline

    # ------------------------------------------------------------------ stone shape
    angular = smoothstep(0.25, 0.5, R(10))                   # field boulders vs split / broken faces
    # rounded boulder: long shoulder; split stone: short broken arris then a face
    rr = r * mix(0.42 + 0.2 * R(4), 0.16 + 0.14 * R(4), angular) * (0.75 + 0.5 * t.noise(0.05, seed=9))
    x = sat(e / rr)
    dome = 1 - (1 - x) ** (2.0 + 1.0 * R(5))
    # broken faces: the crown is the lower envelope of three tilted planes meeting in soft ridges
    un, vn = lay.u / r, lay.v / r
    kk = 22.0
    acc = np.exp(-kk * dome)
    for j in range(3):
        cj = 0.80 + 0.40 * R(20 + 3 * j)
        gx = (R(21 + 3 * j) - 0.5) * 1.0
        gy = (R(22 + 3 * j) - 0.5) * 1.3
        acc = acc + np.exp(-kk * np.maximum(cj + gx * un + gy * vn, 0.12))
    facet = -np.log(acc) / kk + math.log(4.0) / kk
    shape = mix(dome, np.maximum(facet, 0.0), angular)
    lump = t.sample(t.fbm(0.10, 3, 0.5, seed=3), t.X + ox, t.Y + oy)
    knob = t.sample(t.fbm(0.03, 4, 0.55, seed=4), t.X + oy, t.Y + ox)
    grain = t.fbm(0.008, 3, 0.6, seed=5)
    proud = (0.018 + 0.014 * R(11)) * (0.8 + 0.8 * r / 0.18)    # big stones stand further out
    pits = t.speckle(density=2200, size=0.007, seed=7) * smoothstep(0.4, 0.6, knob)
    # every stone has its own skin: water-worn smooth, knobbly, pitted, or bedded with small ledges
    knobby = 0.25 + 1.5 * R(30) ** 1.5
    pitted = smoothstep(0.45, 0.75, R(31))
    bang = (R(13) - 0.5) * 1.2                               # bedding lies near the horizontal
    bu = lay.u * np.cos(bang) + lay.v * np.sin(bang)
    bv = -lay.u * np.sin(bang) + lay.v * np.cos(bang)
    band = t.sample(t.fbm(0.03, 3, 0.55, seed=22, stretch=(8.0, 1.0)), bu + ox, bv + oy)
    # a ledge where a flake has come off the face of some split stones
    lc = (R(33) - 0.5) * 0.5
    lq = (R(34) - 0.5) * 6.283
    lp = un * np.cos(lq) + vn * np.sin(lq) - lc + (lump - 0.5) * 0.25
    ledge = smoothstep(-0.05, 0.05, lp) * (R(35) > 0.55) * angular
    stone_h = (shape * proud
               + dome * ((lump - 0.5) * 0.009 + (knob - 0.5) * 0.0045 * knobby - ledge * (0.003 + 0.004 * R(36)))
               + (grain - 0.5) * 0.0009 * (0.5 + knobby) - pits * 0.0014 * pitted
               + lay.tilt(0.006, 3) * dome - 0.006)

    # ------------------------------------------------------------------ mortar bed
    mlow = t.fbm(0.07, 3, 0.5, seed=10)
    msand = t.fbm(0.009, 4, 0.65, seed=11)
    pebble = t.speckle(density=5000, size=0.008, seed=12)
    fillet = smoothstep(-0.022, 0.0, e)                      # mortar pressed up against each stone
    mortar_h = (mlow - 0.5) * 0.009 + (msand - 0.5) * 0.003 + pebble * 0.0016 + fillet * 0.006 - 0.004
    height = np.maximum(stone_h, mortar_h)
    is_stone = smoothstep(0.0, 0.0018, stone_h - mortar_h)

    cav = t.cavity(height, 0.006, 0.0012)
    edg = t.edges(height, 0.012, 0.003)
    # how close to the mortar line (dirt and damp collect round the base of every stone)
    rim = (1 - smoothstep(0.0, 0.008, stone_h - mortar_h)) * is_stone

    # ------------------------------------------------------------------ stone colour
    pal = np.stack([hexlin(c) for c in PALETTE])
    rs = np.random.default_rng(101)
    k = rs.integers(0, len(PALETTE), lay.n + 1)
    base = pal[k][lay.id]
    granitic = np.array(GRANITIC, dtype=F)[k][lay.id]
    bedded = np.array(BEDDED, dtype=F)[k][lay.id]
    mott = t.sample(t.fbm(0.07, 4, 0.55, seed=20), t.X + ox, t.Y + oy)
    fine = t.fbm(0.014, 3, 0.6, seed=21)
    # bedding / gneissic banding in the stone's own direction
    val = (0.86 + 0.28 * R(14)) * (1.0 + (mott - 0.5) * 0.34 + (fine - 0.5) * 0.16 + bedded * (band - 0.5) * 0.30)
    stone = base * val[..., None]
    # crystalline speckle on the granitic stones
    fl = t.speckle(density=30000, size=0.0035, seed=23)
    stone = mix(stone, hexlin("#34322f"), fl * 0.5 * granitic)
    stone = mix(stone, hexlin("#c4bfb3"), np.roll(fl, (t.h // 3, t.w // 7), axis=(0, 1)) * 0.45 * granitic)
    # iron weathering rind, strongest on the sandy stones
    iron = smoothstep(0.5, 0.8, t.fbm(0.1, 3, 0.5, seed=24))
    stone = mix(stone, stone * hexlin("#c79b62") * 1.75, iron * (0.18 + 0.22 * (1 - granitic)))
    # pale crustose lichen blotches and a grey weathering skin on exposed crowns
    lich = t.speckle(density=500, size=0.03, seed=25, soft=0.4) * smoothstep(0.52, 0.68, t.fbm(0.25, 3, 0.5, seed=26))
    stone = mix(stone, hexlin("#b0b0a2") * (0.8 + 0.4 * fine)[..., None], lich * 0.5)
    stone = mix(stone, desaturate(stone, 0.3) * 1.12 + 0.01, edg * 0.5)
    stone = stone * (1 - 0.40 * cav)[..., None]
    stone = mix(stone, stone * hexlin("#8a7f70") * 1.2, rim * 0.6)

    # ------------------------------------------------------------------ mortar colour
    mcol = ramp(msand, [(0.0, "#6f6657"), (0.5, "#8f8572"), (1.0, "#aaa08a")])
    mcol = mix(mcol, hexlin("#5c574f"), pebble * 0.5)
    mcol = mcol * (0.78 + 0.34 * mlow)[..., None]
    deep = t.cavity(height, 0.03, 0.012)
    mcol = mcol * (1 - 0.30 * deep)[..., None]               # dirt in the deepest pockets

    albedo = np.clip(mix(mcol, stone, is_stone), 0.02, 0.8)

    # ------------------------------------------------------------------ roughness
    rsto = (0.80 - 0.12 * (R(15) - 0.5) - 0.12 * edg + (fine - 0.5) * 0.14 - 0.06 * granitic
            + 0.10 * cav + 0.08 * lich + 0.10 * rim)
    rough = mix(0.95 - 0.04 * pebble, rsto, is_stone)
    return dict(albedo=albedo, height=height, rough=np.clip(rough, 0.3, 1.0), metal=0.0)
