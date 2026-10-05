"""Stepped stone wall buttress. A rock-faced plinth course with a splayed cap carries one to four
stages of coursed masonry; every stage is closed by a moulded drip course (throated, returned along
both flanks) and a sloping weathering of bedded slabs that sets the pier back towards the wall. The
corners are bonded long-and-short ashlar quoins; the faces between are either full ashlar, random
rubble or brick. The head is a long weathering dressed into the wall under a lead flashing, or a
small gabled cap (gablet). Every stone is its own chamfered block; behind each course lies a lime-mortar bed
4 mm back from the stone faces, so the joints read as pointed mortar, and rubble / brick walling is a skin
12 mm behind the quoins. Weathering (rain streaks under the drips, damp foot, moss film, knocked-off arrises)
is painted where the water goes.
"""
from tk import *
import numpy as np
from math import atan, atan2, sqrt, degrees, radians, tan, sin, cos, pi, hypot

META = {"name": "Stone buttress", "class": "component", "mount": "wall_ground", "focus": 0.5}
PARAMS = {
    "height": {"default": 4.2, "min": 2.2, "max": 8.0, "desc": "overall height where the head dies into the wall (m)"},
    "width": {"default": 0.86, "min": 0.5, "max": 1.6, "desc": "overall width at the plinth (m)"},
    "depth": {"default": 1.0, "min": 0.45, "max": 2.0, "desc": "overall projection from the wall at the plinth (m)"},
    "stages": {"default": 2, "min": 1, "max": 4, "desc": "number of stages (set-backs + 1)"},
    "moss": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "moss on weatherings, ledges and the damp foot"},
    "infill": {"default": "ashlar", "choices": ["ashlar", "rubble", "brick"], "desc": "walling between the ashlar quoins"},
    "cap": {"default": "weathered", "choices": ["weathered", "gablet"], "desc": "head: sloping weathering into the wall, or a gabled cap"},
    "slope": {"default": 56.0, "min": 35.0, "max": 68.0, "desc": "pitch of the weatherings (deg)"},
    "course": {"default": 0.29, "min": 0.2, "max": 0.42, "desc": "masonry course height (m)"},
    "plinth": {"default": 0.5, "min": 0.25, "max": 0.9, "desc": "plinth height (m)"},
    "stone": {"default": "#b8ab94", "desc": "stone colour"},
    "date": {"default": "1687", "desc": "raised date / inscription on a tablet stone on the front ('' = none)"},
    "ring": {"default": True, "desc": "wrought-iron tethering ring on the front"},
    "wear": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "chipped arrises, out-of-true stones, staining"},
}
VARIANTS = {
    "tall_gablet": {"height": 6.4, "width": 0.95, "depth": 1.35, "stages": 3, "cap": "gablet", "slope": 58.0,
                    "course": 0.33, "moss": 0.3, "stone": "#cdb891", "date": "1702", "seed": 4},
    "rubble_squat": {"height": 2.9, "width": 1.15, "depth": 1.25, "stages": 1, "infill": "rubble", "slope": 50.0,
                     "moss": 0.82, "stone": "#a9a08c", "plinth": 0.42, "wear": 0.8, "date": "", "seed": 11},
    "brick_quoined": {"height": 4.8, "width": 0.8, "depth": 0.85, "stages": 2, "infill": "brick", "moss": 0.2,
                      "stone": "#c9bea8", "ring": False, "date": "MDCCXL", "seed": 23},
    "slender_four": {"height": 7.4, "width": 0.62, "depth": 1.1, "stages": 4, "slope": 62.0, "course": 0.36,
                     "moss": 0.55, "stone": "#8f8c84", "wear": 0.6, "date": "", "seed": 31},
}

NOSE = 0.072    # projection of the drip-mould nose beyond the face below
NS = 0.042      # projection of the foot of a weathering (it stands on the weathered top of the mould)
HM = 0.135      # height of the drip-mould course
NN = 0.035      # square nose of a weathering slab
EMB = 0.012     # slab bedded this far into the mould top
GAP = 0.008     # mortar joint
MR = 0.004      # mortar bed sits this far behind the stone faces
CH = 0.005      # chamfer of the block arrises

# weathered string / drip mould section (out, up): fillet, cove, soffit, throat, drip edge, rounded nose, weathered top
MOULD = [(-0.025, 0.0), (0.005, 0.0), (0.005, 0.018), (0.011, 0.034), (0.024, 0.046), (0.040, 0.050), (0.043, 0.061),
         (0.052, 0.061), (0.055, 0.049), (0.066, 0.049), (NOSE, 0.056), (NOSE, 0.082), (0.067, 0.092),
         (NS + 0.004, HM - EMB + 0.003), (0.006, HM), (-0.025, HM)]


def _shade(hexc, k, warm=0.0):
    h = hexc.lstrip("#")
    f = (k * (1 + warm), k, k * (1 - warm))
    c = [max(0, min(255, int(round(int(h[i:i + 2], 16) * f[j])))) for j, i in enumerate((0, 2, 4))]
    return "#%02x%02x%02x" % tuple(c)


def _sm(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _inset(prof, d):
    """Open section polyline moved d inwards (the points on the back line stay): the mortar bed behind a stone."""
    amin = min(a for a, _ in prof)
    out = []
    for i, (a, b) in enumerate(prof):
        if a <= amin + 1e-6:
            out.append((a, b))
            continue
        nx = ny = 0.0
        k = 0
        for j in (i - 1, i):
            if 0 <= j < len(prof) - 1:
                dx, dy = prof[j + 1][0] - prof[j][0], prof[j + 1][1] - prof[j][1]
                l = hypot(dx, dy) or 1.0
                nx += dy / l
                ny -= dx / l
                k += 1
        l = hypot(nx, ny) or 1.0
        m = min(2.0, k / l) / l
        out.append((a - d * nx * m, b - d * ny * m))
    return out


def _upoint(W, D, s):
    """Point and outward normal on the U-shaped plan path wall -> left flank -> front -> right flank -> wall."""
    if s <= D:
        return (-W / 2, -s), (-1.0, 0.0)
    if s <= D + W:
        return (-W / 2 + (s - D), -D), (0.0, -1.0)
    return (W / 2, -D + (s - D - W)), (1.0, 0.0)


def _piece(name, W, D, z, prof, s0, s1, ch=CH, cc=0.008, da=(0.0, 0.0), pw=None, bulge=0.0, rough=0.0,
           rng=None, step=0.3, caps=True, sq=False, spall=None):
    """One stone running from s0 to s1 along the U path with cross-section prof [(out, up), ...].
    Ends are chamfered (sq = cut square), a wrapped corner gets a chamfered arris, stones ending at the wall are
    cut square. spall = (corner 0|1, {section index: (e1, e2)}) knocks the arris back by e1 / e2 along the two faces."""
    S = 2 * D + W
    amin = min(a for a, _ in prof)
    npf = len(prof)
    sam = []                                     # (s, point, normal, chamfer direction | None, end shrink, corner, side)

    def st(s, shrink=0.0):
        P, n = _upoint(W, D, min(max(s, 0.0), S))
        sam.append((s, P, n, None, shrink, -1, 0))

    def fill(a, b):
        k = int((b - a) / step) if b > a else 0
        for i in range(1, k + 1):
            st(a + (b - a) * i / (k + 1))

    wall0, wall1 = s0 <= 1e-5, s1 >= S - 1e-5
    if wall0:
        st(0.0)
        prev = 0.0
        if s1 > 0.25 and D > 0.25:               # a vertex row close to the wall: the damp junction is painted there
            st(0.09)
            prev = 0.09
    elif sq:
        st(s0)
        prev = s0
    else:
        st(s0, ch)
        st(s0 + ch)
        prev = s0 + ch
    for ci, (sc, C, n1, n2) in enumerate(((D, (-W / 2, -D), (-1.0, 0.0), (0.0, -1.0)), (D + W, (W / 2, -D), (0.0, -1.0), (1.0, 0.0)))):
        if s0 + 0.03 < sc < s1 - 0.03:
            em1 = em2 = 0.0
            if spall is not None and spall[0] == ci:
                em1 = max(e[0] for e in spall[1].values()) + 0.008
                em2 = max(e[1] for e in spall[1].values()) + 0.008
            fill(prev, sc - cc - em1)
            u = (n1[0] + n2[0], n1[1] + n2[1])
            sam.append((sc, C, u, n2, 0.0, ci, 0))
            sam.append((sc, C, u, n1, 0.0, ci, 1))
            prev = sc + cc + em2
    if wall1:
        if S - prev > 0.25 and D > 0.25:
            fill(prev, S - 0.09)
            st(S - 0.09)
        else:
            fill(prev, S)
        st(S)
    elif sq:
        fill(prev, s1)
        st(s1)
    else:
        fill(prev, s1 - ch)
        st(s1 - ch)
        st(s1, ch)
    ns = len(sam)
    L = max(s1 - s0, 1e-6)
    verts = []
    for j, (s, P, u, kn, shr, ci, side) in enumerate(sam):
        t = clamp((s - s0) / L)
        d0 = da[0] + (da[1] - da[0]) * t
        bl = bulge * (max(0.0, sin(pi * t)) ** 0.6) if bulge else 0.0
        for i, (a, b) in enumerate(prof):
            a2 = a
            if a > amin + 1e-6:
                a2 = a + d0 - shr
                if pw is not None and pw[i] > 0:
                    a2 += pw[i] * (bl + (rng.u(-rough, rough) if (rough and rng is not None) else 0.0))
            kx = ky = 0.0
            if kn is not None:
                e = cc
                if spall is not None and spall[0] == ci and i in spall[1]:
                    e += spall[1][i][side]
                kx, ky = -e * kn[0], -e * kn[1]
            verts.append((P[0] + a2 * u[0] + kx, P[1] + a2 * u[1] + ky, z + b))
    faces = []
    for j in range(ns - 1):
        for i in range(npf - 1):
            faces.append((j * npf + i, (j + 1) * npf + i, (j + 1) * npf + i + 1, j * npf + i + 1))
    if caps:
        faces.append(tuple(range(npf)))
        faces.append(tuple((ns - 1) * npf + i for i in reversed(range(npf))))
    return mesh_object(name, verts, faces)


def _spall(base, npf, top, e1, e2, rest=0.8):
    """Section indices -> arris set-back for a corner knocked off at the top (or bottom) of a stone whose face runs
    over the section points base (bottom), base + 1 (break line), base + 2 (top)."""
    d = {base + 1: (0.08 * e1, 0.08 * e2)}
    hit = base + 2 if top else base
    for k in (range(base + 2, npf) if top else range(0, base + 1)):
        f = 1.0 if k == hit else rest
        d[k] = (e1 * f, e2 * f)
    return d


def _layout(r, W, D, aL, bL, bR, aR, unit=0.44, fill=True, narrow="centre", front=True):
    """Joint positions for one course: two corner stones (flank leg a, front leg b) + filler stones."""
    S = 2 * D + W

    def fit(a):
        a = min(a, D)
        return D if D - a < 0.16 else a

    aL, aR = fit(aL), fit(aR)
    one = False
    if W - bL - bR < 0.17:                       # front too narrow for two quoins and a filler stone
        if narrow == "through" and W <= 0.82:    # one stretcher right across, its ends showing on both flanks
            one = True
        else:
            f = 0.5 + (r.pick([-1, 1]) * r.u(0.1, 0.19) if narrow == "offset" else r.u(-0.05, 0.05))
            bL = W * f
            bR = W - bL

    def split(a, b):
        k = max(1, int(round((b - a) / unit)))
        cuts = [a] + [a + (b - a) * (i + r.u(-0.18, 0.18)) / k for i in range(1, k)] + [b]
        return list(zip(cuts[:-1], cuts[1:]))

    segs = []
    if fill and D - aL > 1e-4:
        segs += split(0.0, D - aL)
    if one:
        segs.append((D - aL, D + W + aR))
    else:
        segs.append((D - aL, D + bL))
        if fill and front and W - bL - bR > 1e-4:
            segs += split(D + bL, D + W - bR)
        segs.append((D + W - bR, D + W + aR))
    if fill and D - aR > 1e-4:
        segs += split(D + W + aR, S)
    return [(a + GAP / 2 if a > 1e-5 else 0.0, b - GAP / 2 if b < S - 1e-5 else S) for a, b in segs]


def _block_prof(h, t=0.07, ch=CH, extra=()):
    """Face stone section. t <= ch gives a 'tile' whose chamfers die into the mortar bed (no hidden beds)."""
    b0, b1 = GAP / 2, h - GAP / 2
    face = [(0.0, b0 + ch)] + [(0.0, b) for b in sorted(extra)] + [(0.0, b1 - ch)]
    if t <= ch:
        return [(-ch, b0)] + face + [(-ch, b1)]
    return [(-t, b0), (-ch, b0)] + face + [(-ch, b1), (-t, b1)]


def _prism(name, pts, c0, c1, axis="X", skip=None, flip=False):
    """Prism of a 2D polygon: axis 'X' -> pts are (y, z) extruded x = c0..c1; 'Y' -> pts are (x, z) extruded
    along y. skip(p, q) -> True drops the side face on that polygon edge (hidden beds / backs)."""
    clean = []
    for q in pts:
        if not clean or abs(q[0] - clean[-1][0]) > 1e-5 or abs(q[1] - clean[-1][1]) > 1e-5:
            clean.append(q)
    if abs(clean[0][0] - clean[-1][0]) < 1e-5 and abs(clean[0][1] - clean[-1][1]) < 1e-5:
        clean.pop()
    k = len(clean)
    area = sum(clean[i][0] * clean[(i + 1) % k][1] - clean[(i + 1) % k][0] * clean[i][1] for i in range(k))
    if area < 0:
        clean.reverse()
    if axis == "X":
        verts = [(c0, a, b) for a, b in clean] + [(c1, a, b) for a, b in clean]
    else:
        verts = [(a, c0, b) for a, b in clean] + [(a, c1, b) for a, b in clean]
    faces = []
    for i in range(k):
        j = (i + 1) % k
        if skip is not None and skip(clean[i], clean[j]):
            continue
        faces.append((i, j, k + j, k + i) if axis == "X" else (i, k + i, k + j, j))
    if axis == "X":
        faces.append(tuple(reversed(range(k))))
        faces.append(tuple(range(k, 2 * k)))
    else:
        faces.append(tuple(range(k)))
        faces.append(tuple(reversed(range(k, 2 * k))))
    if flip:
        faces = [tuple(reversed(fc)) for fc in faces]
    return mesh_object(name, verts, faces)


def _dome(name):
    """Low-poly moss cushion: 5-sided dome, rim slightly below z = 0 so it beds into the stone."""
    vs = [(cos(2 * pi * k / 5), sin(2 * pi * k / 5), -0.12) for k in range(5)]
    vs += [(0.64 * cos(2 * pi * (k + 0.5) / 5), 0.64 * sin(2 * pi * (k + 0.5) / 5), 0.6 + 0.1 * (k % 2)) for k in range(5)]
    vs.append((0.06, -0.05, 1.0))
    fs = []
    for k in range(5):
        k1 = (k + 1) % 5
        fs += [(k, k1, 5 + k), (k1, 5 + k1, 5 + k), (5 + k, 5 + k1, 10)]
    return mesh_object(name, vs, fs)


def build(p, rng):
    H = p.height
    cap = p.cap
    infill = p.infill
    ashlar = infill == "ashlar"
    wearp = p.wear
    mossp = p.moss
    seed = p.seed
    fl = 0.075 if cap == "weathered" else 0.0         # lead flashing upstand above the stone
    Hs = H - fl
    pp = min(clamp(0.085 * p.depth, 0.05, 0.09), p.width * 0.12)     # plinth projection
    W = p.width - 2 * pp
    D0 = p.depth - pp
    hp = min(p.plinth, H * 0.22)
    hcap = min(0.12, hp * 0.35)
    rec = 0.012                                       # rubble / brick infill sits this far behind the quoin faces
    course = max(p.course, (H - hp) / 19.0)          # tall piers are built of bigger stones

    # ---------------- solve the stage layout so the head lands exactly on `height`
    n = int(p.stages)
    ta = tan(radians(p.slope))
    vmin = 0.34
    while True:
        if cap == "weathered":
            Dtop = D0 if n == 1 else min(D0, max(0.22, D0 * 0.45))
            c0 = n * (NN - EMB)
            c1 = D0 + n * NS
        else:
            Dtop = D0 if n == 1 else min(D0, max(0.26, D0 * 0.5))
            c0 = (n - 1) * (NN - EMB) + (0.05 - EMB)
            c1 = (D0 - Dtop) + (n - 1) * NS + W / 2 + NS
        t = min(ta, (Hs - hp - n * HM - n * vmin - c0) / c1)
        if t >= 0.5 or n == 1:
            break
        n -= 1
    if t < 0.3:
        t = max(0.2, min(ta, (Hs - hp - n * HM - n * 0.15 - c0) / c1))
    Ds = [D0 + (Dtop - D0) * (i / (n - 1)) if n > 1 else D0 for i in range(n)]
    V = max(0.1 * n, Hs - hp - n * HM - c0 - c1 * t)
    ws = [1.3 ** (n - 1 - i) for i in range(n)]
    vs = [V * w / sum(ws) for w in ws]
    zs, zvis, zm, zt = [], [], [], []
    z = hp
    for i in range(n):
        rise = 0.0 if i == 0 else (NN - EMB) + (Ds[i - 1] + NS - Ds[i]) * t
        zs.append(z)
        zvis.append(z + rise)
        zm.append(z + rise + vs[i])
        zt.append(zm[i] + HM)
        z = zt[i]
    ang = atan(t)
    ca, sa = cos(ang), sin(ang)
    a_deg = degrees(ang)
    sl = sqrt(1 + t * t)

    # ---------------- materials: three stone tones (colour and polish differ stone to stone), lime mortar, walling
    st_m = [mat("stone_smooth", p.stone), mat("stone_smooth", _shade(p.stone, 0.76, 0.05), rough=1.2),
            mat("stone_smooth", _shade(p.stone, 1.12, -0.02), rough=0.8)]
    plinth_m = mat("stone_rough", _shade(p.stone, 0.74, 0.03))
    mortar_m = mat("mortar", "#ddd4bf")
    infill_m = None if ashlar else mat({"rubble": "rubble_fieldstone", "brick": "brick_red"}[infill])
    stones = [[], [], []]
    plinth_objs = []
    mortar = []                                        # mortar beds behind every stone
    infill_objs = []
    chips = []                                         # (x, y, z) of knocked-off corners: fresh pale stone
    moss_m = []                                        # matrices of moss cushions

    def bucket(r):
        return stones[r.weighted([(0, 5), (1, 3), (2, 2)])]

    def tone(r, o):
        """Every stone weathers to its own tone: some dirtier, some scoured pale."""
        k = r.u()
        if k < 0.5:
            paint(o, grime=r.u(0.05, 0.34) * (0.6 + 0.7 * wearp))
        elif k < 0.75:
            paint(o, wear=r.u(0.1, 0.4))

    def moss_at(r, loc, rot, size, flat=0.4, tu=None, sat=0):
        """A moss cushion plus `sat` smaller ones crowding it along the surface tangent tu."""
        moss_m.append(trs(loc, (rot[0], rot[1], rot[2] + r.u(0, 72)) if rot[0] == 0 and rot[1] == 0 else rot,
                          (size * r.u(0.9, 1.5), size * r.u(0.9, 1.5), size * flat * r.u(0.7, 1.2))))
        for _ in range(sat if tu is not None else 0):
            du = r.pick([-1, 1]) * r.u(0.9, 1.9) * size
            q = (loc[0] + tu[0] * du, loc[1] + tu[1] * du, loc[2] + tu[2] * du)
            s2 = size * r.u(0.45, 0.8)
            moss_m.append(trs(q, rot, (s2 * r.u(0.9, 1.4), s2 * r.u(0.9, 1.4), s2 * flat * r.u(0.7, 1.2))))

    def moss_on_path(r, Wm, Dm, zc, a, size, sides_only=False):
        S = 2 * Dm + Wm
        s = r.u(0.03, S - 0.03)
        if sides_only and Dm < s < Dm + Wm:
            s = r.u(0.03, Dm - 0.03) if r.chance(0.5) else r.u(Dm + Wm + 0.03, S - 0.03)
        P, nrm = _upoint(Wm, Dm, s)
        moss_at(r, (P[0] + nrm[0] * a, P[1] + nrm[1] * a, zc), (0, 0, r.u(0, 360)), size, tu=(-nrm[1], nrm[0], 0), sat=r.pick([0, 1, 1]))

    dj = 0.001 + 0.0025 * wearp

    # ---------------- plinth: rock-faced base course(s) + splayed cap course, each on its mortar bed
    r = rng.fork("plinth")
    S0 = 2 * D0 + W
    hb = hp - hcap
    nb = 2 if hb > 0.46 else 1
    pf = pp - 0.016
    for c in range(nb):
        h = hb / nb
        g0 = 0.0 if c == 0 else GAP / 2
        prof = [(-0.06, g0), (pf - 0.014, g0), (pf, g0 + 0.016), (pf, h * 0.36), (pf, h * 0.68),
                (pf, h - 0.014 - GAP / 2), (pf - 0.014, h - GAP / 2), (-0.06, h - GAP / 2)]
        pw = [0, 0, 0.25, 1.0, 1.0, 0.25, 0, 0]
        legs = [r.u(0.3, 0.46) for _ in range(4)]
        for (a, b) in _layout(r, W, D0, *legs, unit=0.52):
            o = _piece("plinth", W, D0, c * h, prof, a, b, ch=0.012, cc=0.014, da=(r.u(-0.003, 0.003), r.u(-0.003, 0.003)),
                       pw=pw, bulge=r.u(0.008, 0.016), rough=0.005, rng=r, step=0.2, caps=False)
            tone(r, o)
            plinth_objs.append(o)
        mortar.append(_piece("plinth_bed", W, D0, c * h, [(-0.06, 0.0), (pf - 0.014 - MR, 0.0), (pf - 0.014 - MR, h), (-0.06, h)],
                             0.0, S0, sq=True, caps=False, cc=0.014, step=2.0))
    prof = [(-0.06, GAP / 2), (pp - 0.005, GAP / 2), (pp, 0.006 + GAP / 2), (pp, 0.036), (0.008, hcap - 0.004), (0.0, hcap), (-0.06, hcap)]
    legs = [r.u(0.32, 0.5) for _ in range(4)]
    for (a, b) in _layout(r, W, D0, *legs, unit=0.62):
        o = _piece("plinth_cap", W, D0, hb, prof, a, b, da=(r.u(-dj, dj), r.u(-dj, dj)), step=0.4, caps=False)
        tone(r, o)
        stones[2 if r.chance(0.6) else 0].append(o)
    mortar.append(_piece("plinth_cap_bed", W, D0, hb, _inset([(-0.06, 0.0)] + prof[1:], MR), 0.0, S0, sq=True, caps=False, step=2.0))

    # ---------------- tethering ring height: centred in one course of the lowest stage
    zr_ = 1.05 if (zm[0] - 0.2 > 1.05 > hp + 0.2) else (hp + zm[0]) / 2
    has_ring = bool(p.ring) and zm[0] - hp > 0.35
    nc0 = max(1, int(round((zm[0] - zs[0]) / course)))
    c_ring = int(clamp(int((zr_ - hp) / ((zm[0] - hp) / nc0)), 0, nc0 - 1))
    if has_ring:
        zr_ = hp + (c_ring + 0.66) * (zm[0] - hp) / nc0

    # ---------------- date tablet: a stone of its own built into one course of the front
    def courses(i):
        """Coursing of the part of a stage that shows on the front: (bottom, number of courses, course height)."""
        z0 = zvis[i] if zvis[i] - zs[i] > 0.02 else zs[i]
        k = max(1, int(round((zm[i] - z0) / course)))
        return z0, k, (zm[i] - z0) / k

    kt = min(1, n - 1)
    zBt, nck, hck = courses(kt)
    body = str(p.date).strip()
    tab = None                                         # (course, width)
    if body and W > 0.36:
        cand = [c for c in range(max(1, nck - 1)) if not (kt == 0 and has_ring and c <= c_ring)]
        if cand:
            mid = (zBt + zm[kt]) / 2 + 0.1
            ct = min(cand, key=lambda c: abs(zBt + (c + 0.5) * hck - mid))
            tab = (ct, min(W - 0.2, max(hck * 1.5, hck * 0.5 * len(body) + 0.14)))

    # ---------------- stages: coursed blocks with long-and-short quoins, drip-mould course, weathering slabs
    # the mould's bed: a simpler section just inside the stone, clear of the throat
    mould_bed = [(-0.025, 0.0), (0.005 - MR, 0.0), (0.005 - MR, 0.02), (0.022, 0.05), (0.04, 0.066), (0.06, 0.066),
                 (NOSE - MR, 0.084), (NS, HM - EMB - 0.001), (0.006, HM - MR), (-0.025, HM - MR)]
    weath = []                                         # (yf, yb, zb, ztop, bed-joint fractions, half width, covered-from fraction)
    joints_low = []                                    # (W, D, z) bed joints near the ground, for moss
    Ll = 0.11                                          # lap of the lead flashing over the top slab
    nchip = 0
    hcs = []
    for i in range(n):
        r = rng.fork(f"stage{i}")
        Di = Ds[i]
        Si = 2 * Di + W
        zB, nB, hB = courses(i)
        hcs.append(hB)
        crs = []                                      # (bottom, height, flanks only, index among the front courses)
        if zB - zs[i] > 1e-4:                         # beside the weathering of the stage below only the flanks show
            nA = max(1, int(round((zB - zs[i]) / course)))
            crs += [(zs[i] + k * (zB - zs[i]) / nA, (zB - zs[i]) / nA, True, -1) for k in range(nA)]
        crs += [(zB + k * hB, hB, False, k) for k in range(nB)]
        runs = []                                     # walling skins [z0, z1, flanks only]
        for j, (zc, hc, fo, c) in enumerate(crs):
            if zc < 1.3 and c > 0:
                joints_low.append((W, Di, zc))
            tabc = tab is not None and i == kt and c == tab[0]
            full = ashlar or tabc or (i == 0 and has_ring and c == c_ring)     # a through course of ashlar carries the ring
            even = (j + i) % 2 == 0
            if full:
                lo1, lo2, sh1, sh2 = r.u(0.36, 0.5), r.u(0.36, 0.5), r.u(0.17, 0.25), r.u(0.17, 0.25)
                legs = (sh1, lo1, lo2, sh2) if even else (lo1, sh1, sh2, lo2)
                if tabc:                              # the two quoins run up to the tablet stone
                    bl = (W - tab[1]) / 2
                    legs = (legs[0], bl, bl, legs[3])
            else:       # quoins only: long-and-short, never more than half of the front between them
                fa = [clamp(0.42 * Di, 0.12, 0.42) * r.u(0.88, 1.05) for _ in range(2)]
                fs_ = [clamp(0.22 * Di, 0.08, 0.22) * r.u(0.9, 1.1) for _ in range(2)]
                fb = max(0.09, min(0.36, 0.3 * W)) * r.u(0.92, 1.04)
                fbs = max(0.07, min(0.2, 0.19 * W)) * r.u(0.9, 1.08)
                legs = (fa[0], fbs, fb, fs_[1]) if even else (fs_[0], fb, fbs, fa[1])
                if runs and runs[-1][2] == fo and abs(runs[-1][1] - zc) < 1e-6:
                    runs[-1][1] = zc + hc
                else:
                    runs.append([zc, zc + hc, fo])
            segs = _layout(r, W, Di, *legs, fill=full, narrow="through" if even else "offset", front=not tabc)
            if fo:                                    # stop the stones at the front arris, behind the slabs
                cl = []
                for (a, b) in segs:
                    if a < Di - 0.02:
                        cl.append((a, min(b, Di)))
                    if b > Di + W + 0.02:
                        cl.append((max(a, Di + W + 1e-6), b))
                segs = cl
            dense = c == nB - 1                       # the course under a drip mould carries the rain streaks
            step = 0.065 if dense else 0.5
            tk_ = 0.0 if ashlar else 0.03
            base = 1 if ashlar else 2
            zmid = zc + hc / 2
            pc = (0.22 + 0.55 * wearp) if 0.4 < zmid < 1.7 else 0.12 * wearp
            inzone = 0.4 < zmid < 1.7
            if inzone and wearp > 0.15 and nchip < 2:
                pc = 1.0                              # at least two knocked corners where carts and people pass
            c_spall, c_extra = None, ()
            for (a, b) in segs:
                cis = [ci for ci, sc in ((0, Di), (1, Di + W)) if a + 0.03 < sc < b - 0.03]
                extra, spall = ((hc * 0.3, hc * 0.62) if dense else ()), None
                if cis and not dense and c_spall is None and wearp > 0.02 and r.chance(pc):
                    # a knocked-off corner: the arris retreats along both faces leaving a broad spalled facet
                    ci = r.pick(cis)
                    sc = Di if ci == 0 else Di + W
                    top = r.chance(0.6)
                    lc = min(hc * 0.6, r.u(0.09, 0.18))
                    e1 = min(0.4 * (sc - a), 0.075, r.u(0.035, 0.065) * (0.75 + 0.6 * wearp))
                    e2 = min(0.4 * (b - sc), 0.075, r.u(0.035, 0.065) * (0.75 + 0.6 * wearp))
                    nchip += 1 if inzone else 0
                    extra = ((hc - GAP / 2 - CH - lc) if top else (GAP / 2 + CH + lc),)
                    spall = (ci, _spall(base, 5 if ashlar else 7, top, e1, e2))
                    c_spall, c_extra = (ci, _spall(1, 5, top, e1, e2, rest=1.0)), extra
                    chips.append((-W / 2 if ci == 0 else W / 2, -Di, zc + (hc - 0.015 if top else 0.015)))
                o = _piece("block", W, Di, zc, _block_prof(hc, tk_, CH, extra), a, b, da=(r.u(-dj, dj) * 0.6, r.u(-dj, dj) * 0.6),
                           step=step, caps=not ashlar, spall=spall)
                tone(r, o)
                bucket(r).append(o)
                if not full:        # mortar bed behind a quoin, pointed up against the walling
                    bp = [(-0.03, 0.0), (-MR, 0.0), (-MR, extra[0] if extra else hc / 2), (-MR, hc), (-0.03, hc)]
                    if not spall:
                        bp.pop(2)
                    a2, b2 = max(0.0, a - GAP / 2 - 0.005), min(Si, b + GAP / 2 + 0.005)
                    if fo:
                        a2, b2 = (a2, min(b2, Di)) if b <= Di + 1e-5 else (max(a2, Di + W + 1e-6), b2)
                    mortar.append(_piece("quoin_bed", W, Di, zc, bp, a2, b2, sq=True, step=2.0,
                                         spall=(spall[0], _spall(1, 5, top, e1, e2, rest=1.0)) if spall else None))
            if full:                # the course's mortar bed, following any spalled corner
                bp = [(-0.07, 0.0), (-MR, 0.0), (-MR, c_extra[0] if c_extra else hc / 2), (-MR, hc), (-0.07, hc)]
                if not c_spall:
                    bp.pop(2)
                spans = [(0.0, Si)]
                if fo:
                    spans = [(0.0, Di), (Di + W + 1e-6, Si)]
                elif tabc:                            # the bed stops at the tablet; its bed joints get their own fillets
                    e = (W - tab[1]) / 2 + GAP / 2 - 0.001
                    spans = [(0.0, Di + e), (Di + W - e, Si)]
                    for zj in (zc + GAP / 4, zc + hc - GAP / 4):
                        mortar.append(box("tablet_bed", (tab[1] - GAP, 0.02, GAP / 2 + 0.004), loc=(0, -Di + MR + 0.01, zj)))
                for (a, b) in spans:
                    mortar.append(_piece("course_bed", W, Di, zc, bp, a, b, sq=True, caps=False, step=2.0, spall=c_spall))
        for (z0, z1, fo) in runs:   # walling between the quoins: a skin just behind the quoin faces, one piece per
            hh = z1 - z0            # run of courses between the through courses of ashlar
            nk = max(1, int(round(hh / 0.15)))
            sp = [(-0.08, 0.0)] + [(-rec, hh * k / nk) for k in range(nk + 1)] + [(-0.08, hh)]
            for (a, b) in ([(0.0, Di), (Di + W + 1e-6, Si)] if fo else [(0.0, Si)]):
                o = _piece("walling", W, Di, z0, sp, a, b, sq=True, caps=False, cc=0.05, step=0.15)
                if infill == "rubble":
                    displace_noise(o, strength=0.005, scale=0.11, seed=seed + 60 + i)
                infill_objs.append(o)
        # drip mould returned round the three faces
        legs = [r.u(0.26, 0.42) for _ in range(4)]
        for (a, b) in _layout(r, W, Di, *legs, unit=0.62):
            o = _piece("drip", W, Di, zm[i], MOULD, a, b, cc=0.004, da=(r.u(-dj, dj) * 0.5, r.u(-dj, dj) * 0.5), step=0.7,
                       sq=True, caps=False)
            smooth(o, 50)
            tone(r, o)
            stones[2 if r.chance(0.7) else 0].append(o)
        mortar.append(_piece("drip_bed", W, Di, zm[i], mould_bed, 0.0, Si, sq=True, caps=False, cc=0.004, step=2.0))
        # weathering: bedded slabs with a square nose, sloping back to the next stage (or to the wall)
        last = i == n - 1
        if last and cap == "gablet":
            continue
        yf = -(Di + NS)
        ybv = 0.0 if last else -Ds[i + 1]
        yback = 0.0 if last else -Ds[i + 1] - 0.001      # the slabs butt against the next stage
        zb = zt[i] - EMB
        ztop = zb + NN + (yback - yf) * t
        nbz = max(1, int(round((ztop - zb - NN) / 0.3)))
        xw = W / 2 + 0.006
        xcuts = [-xw, xw] if 2 * xw < 0.98 else [-xw, r.u(-0.12, 0.12), xw]
        weath.append((yf, ybv, zb, zb + NN + (ybv - yf) * t, [(yback - yf) * k / nbz / (ybv - yf) for k in range(1, nbz)], xw,
                      1.0 - (Ll * ca / (ybv - yf) if last else 0.0)))
        hid = lambda q0, q1: (abs(q0[0] - yback) < 1e-6 and abs(q1[0] - yback) < 1e-6) or (abs(q0[1] - zb) < 1e-6 and abs(q1[1] - zb) < 1e-6)
        ys = lambda zz: yf + (zz - zb - NN) / t
        for k in range(nbz):
            za = zb if k == 0 else zb + NN + (ztop - zb - NN) * k / nbz + GAP / 2
            z2 = zb + NN + (ztop - zb - NN) * (k + 1) / nbz - (GAP / 2 if k < nbz - 1 else 0.0)
            if k == 0:
                poly = [(yf, zb), (yback, zb), (yback, z2), (ys(z2), z2), (yf, zb + NN)]
            else:
                poly = [(ys(za), za), (yback, za), (yback, z2), (ys(z2), z2)]
            xc = xcuts if k % 2 == 0 or len(xcuts) == 2 else [-xw, -xcuts[1], xw]
            for j in range(len(xc) - 1):
                x0 = xc[j] + (GAP / 2 if j > 0 else 0.0)
                x1 = xc[j + 1] - (GAP / 2 if j < len(xc) - 2 else 0.0)
                o = _prism("weathering", poly, x0, x1, skip=hid)
                densify(o, 0.15)
                displace_noise(o, strength=0.0012 + 0.0012 * wearp, scale=0.16, seed=seed + 7 * i)
                bevel(o, 0.006, segments=1)
                tone(r, o)
                stones[2 if r.chance(0.6) else 0].append(o)
        # the bed the slabs lie in: same solid 5 mm inside, seen in the slab joints
        dm = 0.005
        o = _prism("weathering_bed", [(yf + dm, zb), (yback, zb), (yback, ztop - dm * sl), (yf + dm, zb + NN + dm * t - dm * sl)],
                   -(xw - dm), xw - dm, skip=hid)
        densify(o, 0.18)
        mortar.append(o)

    # ---------------- head
    r = rng.fork("head")
    lead_objs = []
    Dn = Ds[-1]
    if cap == "weathered":
        # dressed lead: turned into a chase in the wall (pointed with a mortar fillet), down the wall, lapped over the
        # top slab and finished with a welted edge
        tt = 0.004
        top = H - 0.012
        xl = W / 2 + 0.002
        sp = lambda l, off=0.0: (-l * ca - off * sa, Hs - l * sa + off * ca)
        Q = [(0.0, top), (-tt, top), (-tt - 0.0015, Hs + 0.03), (-tt - 0.004, Hs + 0.012), sp(0.022, tt), sp(Ll - 0.017, tt),
             sp(Ll - 0.009, tt + 0.004), sp(Ll - 0.002, tt + 0.0045), sp(Ll + 0.004, tt * 0.55), sp(Ll + 0.003, 0.0), (0.0, Hs)]
        online = lambda q: abs(q[1] - Hs - t * q[0]) < 1e-6
        o = _prism("flashing", Q, -xl, xl, skip=lambda q0, q1: (online(q0) and online(q1)) or (abs(q0[0]) < 1e-6 and abs(q1[0]) < 1e-6))
        densify(o, 0.1)
        displace_noise(o, strength=0.0011, scale=0.06, seed=seed + 3)
        smooth(o, 62)
        lead_objs.append(o)
        mortar.append(_prism("chase_fillet", [(0.0, H), (-0.011, top - 0.004), (0.0, top - 0.004)], -xl - 0.006, xl + 0.006,
                             skip=lambda q0, q1: abs(q0[0]) < 1e-6 and abs(q1[0]) < 1e-6))
    else:
        hw = W / 2 + NS
        zb = zt[-1] - EMB
        zr = zb + 0.05 + hw * t
        xz = [(-hw, zb), (hw, zb), (hw, zb + 0.05), (0.0, zr), (-hw, zb + 0.05)]
        y0 = -(Dn + NS)
        ng = max(1, int(round(-y0 / 0.42)))
        cuts = [y0] + [y0 * (1 - (k + r.u(-0.15, 0.15)) / ng) for k in range(1, ng)] + [0.0]
        for k in range(ng):
            o = _prism("gablet", xz, cuts[k] + (GAP / 2 if k > 0 else 0), cuts[k + 1] - (GAP / 2 if k < ng - 1 else 0), axis="Y",
                       skip=lambda q0, q1: abs(q0[1] - zb) < 1e-6 and abs(q1[1] - zb) < 1e-6)
            densify(o, 0.16)
            displace_noise(o, strength=0.0012 + 0.0012 * wearp, scale=0.16, seed=seed + 40)
            bevel(o, 0.007, segments=1)
            tone(r, o)
            stones[2 if r.chance(0.6) else 0].append(o)
        dm = 0.005
        o = _prism("gablet_bed", [(-hw + dm, zb + 0.004), (hw - dm, zb + 0.004), (hw - dm, zb + 0.05 - dm * 0.5),
                                  (0.0, zr - dm * sl), (-hw + dm, zb + 0.05 - dm * 0.5)], y0 + dm, 0.0, axis="Y")
        densify(o, 0.2)
        mortar.append(o)
        # carved boss on the gable face
        rb = min(0.085, hw * 0.28)
        if hw * t > 0.3:
            boss = lathe("boss", [(0.0, -0.01), (rb, -0.01), (rb, 0.006), (rb * 0.86, 0.013), (rb * 0.7, 0.008), (rb * 0.52, 0.01),
                                  (rb * 0.36, 0.022), (rb * 0.16, 0.026), (0.0, 0.024)], segments=16,
                         loc=(0, y0, zb + 0.05 + hw * t * 0.36), rot=(90, 0, 0))

            def petals(co):
                a = atan2(co.y, co.x)
                rr = hypot(co.x, co.y)
                k = 1.0 - 0.16 * (0.5 + 0.5 * cos(a * 4)) * smoothstep(rb * 0.4, rb * 0.9, rr)
                return Vector((co.x * k, co.y * k, co.z))
            transform_verts(boss, petals)
            stones[2].append(boss)
        for sx in (-1, 1):
            for _ in range(int(0.5 + 5 * mossp)):
                x = sx * r.u(0.25, 0.98) * hw
                moss_at(r, (x, r.u(y0 + 0.03, -0.03) * r.u(0.2, 1.0), zr - abs(x) * t - 0.003), (0, sx * a_deg, 0), r.u(0.018, 0.04))

    # ---------------- hidden backstop core + shadow liner
    def sil(rc_, dv_):
        """Side silhouette (y, z) of the pier, rc_ inside the faces and dv_ below the weathering surfaces."""
        q = [(0.0, 0.0), (-(D0 - rc_), 0.0)]
        for i in range(n):
            yi = -(Ds[i] - rc_)
            if i == n - 1 and cap == "gablet":
                q += [(yi, zt[i] - 0.02), (0.0, zt[i] - 0.02)]
                break
            yf = -(Ds[i] + NS)
            zsl = lambda yy: zt[i] - EMB + NN + (yy - yf) * t - dv_
            yn = 0.0 if i == n - 1 else -(Ds[i + 1] - rc_)
            q += [(yi, zsl(yi)), (yn, zsl(yn))]
        return q

    rc = 0.06
    back = lambda q0, q1: (abs(q0[0]) < 1e-6 and abs(q1[0]) < 1e-6) or (abs(q0[1]) < 1e-6 and abs(q1[1]) < 1e-6)
    core = _prism("core", sil(rc, 0.03 * sl), -(W / 2 - rc), W / 2 - rc, skip=back)
    # shadow liner: the pier's own shape just behind the mortar with its faces turned inwards. The engine casts
    # shadows from the faces turned away from the light, with a generous bias - for a pier standing against a wall
    # those are its back, i.e. nothing - so the liner gives the lit sides a caster as far out as possible. It is
    # fully enclosed and faces inwards: never visible. Its corners are cut back clear of the spalled arrises.
    lo = 0.012 if ashlar else rec + 0.01
    lc_ = 0.085
    liners = [_prism("shadow_liner", [(y, max(zz, 0.03)) for (y, zz) in sil(lo, lo * sl + 0.002)],
                     -(W / 2 - lc_), W / 2 - lc_, flip=True, skip=back)]
    for i in range(n):          # the vertical faces of each stage, round the (cut) corners
        z1 = zm[i] + HM * 0.5
        if z1 - zs[i] > 0.1:
            liners.append(_piece("shadow_liner", W - 2 * lo, Ds[i] - lo, zs[i] + 0.03,
                                 [(-0.02, z1 - zs[i] - 0.03), (0.0, z1 - zs[i] - 0.03), (0.0, 0.0), (-0.02, 0.0)],
                                 0.0, 2 * (Ds[i] - lo) + W - 2 * lo, sq=True, caps=False, cc=lc_ - lo, step=2.0))
    for o in [core] + liners:
        assign(o, mortar_m)
        paint(o, grime=0.7)

    # ---------------- tethering ring: pattress plate, forged eye, loose ring resting on the stone
    iron_objs = []
    if has_ring:
        r = rng.fork("ring")
        xr = r.u(-0.12, 0.12) * W
        yw = -D0
        iron_objs.append(box("ring_plate", (0.078, 0.007, 0.078), loc=(xr, yw - 0.0035, zr_), rot=(0, 45, 0), bevel=0.0015, segments=1))
        iron_objs.append(torus("ring_eye", major=0.02, minor=0.0062, loc=(xr, yw - 0.02, zr_), rot=(0, 90, 0), major_seg=10, minor_seg=5))
        iron_objs.append(cylinder("ring_shank", 0.008, 0.02, loc=(xr, yw - 0.008, zr_), rot=(90, 0, 0), verts=6))
        iron_objs.append(torus("ring", major=0.064, minor=0.0088, loc=(xr + 0.004, yw - 0.0165, zr_ - 0.069), rot=(97.0, 0, r.u(-6, 6)),
                               major_seg=16, minor_seg=5))
        nail = bolt("ring_nail", 0.006, 0.004, kind="pyramid")
        iron_objs.append(instances_to_mesh("ring_nails", nail, [trs((xr + dx, yw - 0.007, zr_ + dz), (90, 0, 0))
                                                                 for dx, dz in ((0.036, 0), (-0.036, 0), (0, 0.036), (0, -0.036))], keep_base=False))
        im = mat("iron_wrought")
        for o in iron_objs:
            assign(o, im)
            smooth(o, 40)
            paint(o, special=mask_mul(noise_mask(0.03, seed=seed + 9, lo=0.25, hi=0.7), 0.8), wear=0.4)

    # ---------------- moss cushions, clustered where water lingers: slab bed joints, the head of each weathering,
    # the ledges of the drip moulds, the plinth splay, low bed joints and the damp foot
    r = rng.fork("moss")
    if mossp > 0.04:
        for (yf, yb, zb, ztp, fr, xw, vl) in weath:
            spots = list(fr) * 2 + [vl - 0.025]
            for _ in range(int(1 + 5 * mossp)):
                v = clamp(r.pick(spots) + r.u(-0.012, 0.02), 0.03, vl - 0.012)
                y = yf + v * (yb - yf)
                moss_at(r, (r.u(-0.92, 0.92) * xw, y, zb + NN + (y - yf) * t - 0.002), (a_deg, 0, 0), r.u(0.016, 0.034),
                        tu=(1, 0, 0), sat=r.pick([0, 1, 1, 2]))
        for i in range(n):
            if i == n - 1 and cap == "gablet":
                continue
            for _ in range(int(0.5 + 3 * mossp)):
                a = r.u(0.014, 0.036)
                moss_on_path(r, W, Ds[i], zt[i] - 0.005 - 0.22 * (a - 0.006), a, r.u(0.011, 0.02), sides_only=True)
        for _ in range(int(1 + 4 * mossp)):
            a = r.u(0.012, pp - 0.012)
            moss_on_path(r, W, D0, hp - hcap + 0.034 + (hcap - 0.036) * (1 - a / pp) - 0.004, a, r.u(0.012, 0.024))
        for _ in range(int(2 + 8 * mossp)):
            sz = r.u(0.018, 0.034)
            s = r.u(0.02, S0 - 0.02) if r.chance(0.4) else (r.u(0.02, 0.3 * D0) if r.chance(0.5) else S0 - r.u(0.02, 0.3 * D0))
            P, nrm = _upoint(W, D0, s)
            f = r.u(0.3, 0.45)
            el = r.u(1.0, 1.7)                         # stretched along the foot of the plinth, hugging the stone
            moss_m.append(trs((P[0] + nrm[0] * (pf - 0.6 * sz), P[1] + nrm[1] * (pf - 0.6 * sz), 0.0), (0, 0, 0),
                              (sz * (el if abs(nrm[1]) > 0.5 else 1.0), sz * (el if abs(nrm[0]) > 0.5 else 1.0), sz * f)))
        if joints_low:
            for _ in range(int(5 * mossp)):
                Wj, Dj, zj = r.pick(joints_low)
                s = r.u(0.03, 2 * Dj + Wj - 0.03)
                P, nrm = _upoint(Wj, Dj, s)
                rot = (90, 0, 0) if nrm[1] < -0.5 else (0, 90 * nrm[0], 0)
                moss_at(r, (P[0] - nrm[0] * 0.002, P[1] - nrm[1] * 0.002, zj), rot, r.u(0.01, 0.018), flat=0.5,
                        tu=(-nrm[1], nrm[0], 0), sat=r.pick([1, 2]))
    if moss_m:
        base = _dome("moss_base")
        assign(base, mat("moss"))
        mo = instances_to_mesh("moss_cushions", base, moss_m, keep_base=False, seed=seed)
        smooth(mo, 80)
        paint(mo, grime=mask_mul(noise_mask(0.05, seed=seed + 12, lo=0.2, hi=0.8), 0.22))

    # ---------------- hart's-tongue fern tufts rooted in damp joints (corner above a weathering, plinth splay)
    tufts = []
    if mossp >= 0.25:
        for i in range(n - 1):
            tufts.append(((r.u(-0.3, 0.3) * W, -Ds[i + 1] - 0.004, zvis[i + 1] - 0.012), 0.0, 1.0))
        side = 1 if r.chance(0.6) else -1
        tufts.append(((side * (W / 2 + 0.006), -r.u(0.08, 0.3) * D0, hp - 0.012), 90.0 * side, min(0.7, pp / 0.14)))
        if n == 1:
            tufts.append(((-side * r.u(0.0, 0.3) * W, -D0 - 0.006, hp - 0.012), 0.0, min(0.7, pp / 0.14)))
        tufts = tufts[:1 + int(mossp > 0.5) + int(mossp > 0.8)]
    if tufts:
        def blade_fn(u, v):
            hwid = 0.014 * (1.0 - v ** 2.4) * (0.45 + 0.55 * min(1.0, v * 4.0)) + 0.0008
            x = (u - 0.5) * 2.0 * hwid
            return (x, -0.12 * v * v - 0.35 * abs(x), 0.2 * (v - 0.28 * v ** 3))
        blade = surface("fern_blade", blade_fn, 2, 3, thickness=0.0012, uv_len=(0.25, 0.25))
        realize(blade)
        assign(blade, mat("leaf", "#b9cf8e"))
        fm = []
        for (loc, az0, sc) in tufts:
            for _ in range(r.pick([5, 6, 7])):
                fm.append(trs((loc[0] + r.u(-0.012, 0.012), loc[1] + r.u(-0.004, 0.004), loc[2]),
                              (r.u(-28, 12), r.u(-8, 8), az0 + r.u(-75, 75)), sc * r.u(0.5, 1.05)))
        instances_to_mesh("ferns", blade, fm, keep_base=False, uv_jitter=False)

    # ---------------- art-directed weathering: where the water goes
    n_fine = noise_mask(0.045, seed=seed + 1, lo=0.42, hi=0.72)      # sampled squashed in z -> vertical streaks
    n_run = noise_mask(0.085, seed=seed + 8, lo=0.36, hi=0.7)
    n_broad = noise_mask(0.13, seed=seed + 2, lo=0.3, hi=0.7)
    n_up = noise_mask(0.16, seed=seed + 3, lo=0.3, hi=0.7)
    n_damp = noise_mask(0.3, seed=seed + 4, lo=0.3, hi=0.75)
    n_side = noise_mask(0.45, seed=seed + 5, lo=0.4, hi=0.8)
    n_blot = noise_mask(0.09, seed=seed + 6, lo=0.5, hi=0.8)
    n_spk = noise_mask(0.06, seed=seed + 7, lo=0.35, hi=0.75)
    drips = [(zm[i] + 0.012, min(0.6, hcs[i])) for i in range(n)] + [(hp - hcap + 0.006, 0.28)]
    vsq = np.array([1.0, 1.0, 0.06])
    ssq = np.array([1.0, 0.1, 0.1])
    film_amp = min(1.0, 0.62 + 0.8 * mossp)
    edge = 0.1 + 0.85 * mossp

    def moss_fn(co, nr):
        m = np.zeros(len(co))
        if mossp <= 0.02:
            return m
        x, y, zz = co[:, 0], co[:, 1], co[:, 2]
        nz = nr[:, 2]
        vert = np.clip(1.0 - np.abs(nz) * 2.0, 0.0, 1.0)
        nu = np.asarray(n_up(co, nr))
        nd = np.asarray(n_damp(co, nr))
        spk = np.asarray(n_spk(co, nr))
        onw = np.zeros(len(co), dtype=bool)
        for (yf, yb, zb, ztp, fr, xw, vl) in weath:
            dz = zz - (zb + NN + (y - yf) * t)      # on the sloping face: slabs, their bevels, the mortar in the joints
            ins = (dz > -0.022 * sl) & (dz < 0.03) & (y > yf - 0.01) & (y < yb + 0.012) & (np.abs(x) < xw + 0.01) & (nz > -0.75)
            ins &= ~((y > yb - 0.004) & (nz < 0.15))        # not the face of the stage standing on it
            v = np.clip((y - yf) / max(1e-3, yb - yf), 0.0, 1.2)
            # a continuous film on the upper part thinning down-slope, tongues running down the bed joints and
            # along the cheeks where the water sheds last
            val = _sm(1.0 - edge - 0.42, 1.0 - edge + 0.16, v + 0.15 * (nu - 0.5) + 0.1 * (spk - 0.5)) * (0.9 + 0.1 * spk)
            for f in fr:
                val = np.maximum(val, np.exp(-((v - f) / 0.035) ** 2) * (0.5 + 0.5 * nu) * min(1.0, 0.55 + mossp))
            val = np.maximum(val, np.exp(-((xw - np.abs(x)) / 0.03) ** 2) * (0.3 + 0.5 * nu) * _sm(0.0, 0.5, v + mossp - 0.3))
            m = np.where(ins, np.maximum(m, val * film_amp), m)
            onw |= ins
        # ledges: drip-mould tops, plinth splay, gablet
        oth = (~onw) & (nz > 0.3)
        m = np.where(oth, np.maximum(m, (0.42 + 0.55 * mossp) * (0.55 + 0.45 * nu) + 0.25 * np.clip(1.0 + y / 0.2, 0.0, 1.0)), m)
        # the damp junction with the wall, the foot, a little on the sunless flank
        hf = 0.55 + 0.45 * np.clip(1.0 - zz / (0.45 * H), 0.0, 1.0)
        m = np.maximum(m, vert * np.clip(1.0 + y / (0.1 + 0.16 * mossp), 0.0, 1.0) * (0.62 + 0.38 * mossp) * (0.3 + 0.7 * nd * hf + 0.4 * (hf - 0.55)))
        m = np.maximum(m, np.clip(1.0 - zz / (0.2 + 0.45 * mossp), 0.0, 1.0) * (0.55 + 0.45 * nd) * min(1.0, 0.6 + 0.6 * mossp))
        m = np.maximum(m, 0.55 * mossp * np.clip(nr[:, 0], 0.0, 1.0) * np.clip(1.0 - zz / (0.5 * H), 0.0, 1.0) * np.asarray(n_side(co, nr)))
        return np.clip(m, 0.0, 1.0)

    def grime_fn(co, nr):
        y, zz = co[:, 1], co[:, 2]
        vert = np.clip(1.0 - np.abs(nr[:, 2]) * 2.0, 0.0, 1.0)
        upf = np.clip(nr[:, 2], 0.0, 1.0)
        col = np.asarray(n_run(co * vsq, nr))              # constant down a face: one value per rain run
        runl = np.asarray(n_fine(co * vsq, nr))
        face = np.clip(1.7 - np.abs(nr[:, 2]) * 2.0, 0.0, 1.0)     # vertical faces and their chamfers
        damp = np.asarray(n_damp(co, nr))
        # splash-back and rising damp on the bottom half metre, a dirt line where the plinth meets the cobbles
        g = 0.52 * np.clip(1.0 - zz / (0.6 + 0.3 * damp), 0.0, 1.0) ** 1.1
        g = np.maximum(g, 0.55 * np.clip(1.0 - zz / 0.06, 0.0, 1.0))
        # under every drip: a sheltered sooty line right below the mould, rain runs of different length below it
        for zd, ln in drips:
            d = zd - zz
            crust = 0.38 * np.clip(1.0 - d / 0.075, 0.0, 1.0)
            fall = (0.2 + 0.7 * col) * np.clip(1.0 - d / (ln * (0.7 + 0.7 * runl)), 0.0, 1.0) ** 0.6 * (0.75 + 0.25 * wearp)
            g = np.maximum(g, np.where(d > -0.004, np.maximum(crust, fall) * face, 0.0))
        # damp, dirty junction with the wall
        g = np.maximum(g, 0.38 * np.clip(1.0 + y / 0.13, 0.0, 1.0) * vert * (0.4 + 0.6 * damp))
        # up-facing stone: dirt washed down the slope in soft streaks, a little lichen crust, damp under the moss
        g = np.maximum(g, upf * (0.06 + (0.2 + 0.08 * wearp) * np.asarray(n_broad(co * ssq, nr)) + 0.1 * np.asarray(n_blot(co, nr))))
        g = np.maximum(g, 0.5 * moss_fn(co, nr))
        # sooty undersides (soffits, throats), soft unevenness across every face
        g = np.maximum(g, 0.42 * np.clip(-nr[:, 2], 0.0, 1.0))
        g = np.maximum(g, 0.17 * vert * np.asarray(n_broad(co, nr)))
        return np.clip(g, 0.0, 1.0)

    def wear_fn(co, nr):
        zz = co[:, 2]
        band = np.clip(1.0 - np.abs(zz - 1.0) / 0.7, 0.0, 1.0)
        w = band * np.asarray(n_blot(co * 0.7, nr)) * (0.2 + 0.45 * wearp)
        w = np.maximum(w, 0.3 * np.clip(nr[:, 2], 0.0, 1.0) * np.asarray(n_spk(co * 1.7, nr)))      # pale lichen on up-facing stone
        for (cx, cy, cz) in chips:                      # fresh pale stone where a corner was knocked off
            d2 = (co[:, 0] - cx) ** 2 + (co[:, 1] - cy) ** 2 + (zz - cz) ** 2
            w = np.maximum(w, 0.8 * np.exp(-d2 / 0.1 ** 2))
        return np.clip(w, 0.0, 1.0)

    for k in range(3):
        if stones[k]:
            o = join(stones[k], f"stone_{k}")
            assign(o, st_m[k])
            paint(o, grime=grime_fn, special=moss_fn, wear=wear_fn)
    if plinth_objs:
        o = join(plinth_objs, "plinth_stones")
        assign(o, plinth_m)
        paint(o, grime=grime_fn, special=moss_fn)
    if mortar:
        o = join(mortar, "mortar_beds")
        assign(o, mortar_m)
        paint(o, grime=lambda co, nr: 0.75 * grime_fn(co, nr), special=moss_fn)
    if infill_objs:
        o = join(infill_objs, "walling")
        assign(o, infill_m)
        paint(o, grime=grime_fn, special=moss_fn)
    for o in lead_objs:
        assign(o, mat("lead", rough=1.5))
        paint(o, grime=lambda co, nr: 0.04 + 0.16 * np.asarray(n_broad(co * vsq, nr)),
              wear=lambda co, nr: 0.75 * np.asarray(n_fine(co * vsq, nr)))        # white oxide runs

    # ---------------- date tablet (a proud stone with a sunk, framed field and raised numerals) + sockets
    k, zl = 0, clamp(2.6, zvis[0] + 0.2, max(zvis[0] + 0.2, zm[0] - 0.2))
    for i in range(1, n):                               # lantern bracket: the stage face nearest to 2.6 m up
        zi = clamp(2.6, zvis[i] + 0.2, max(zvis[i] + 0.2, zm[i] - 0.2))
        if zm[i] - zvis[i] > 0.45 and abs(zi - 2.6) < abs(zl - 2.6):
            k, zl = i, zi
    if tab is not None:
        # the margin of the stone is left standing as a frame round a sunk, dirt-holding field with raised numerals
        ct, tw = tab
        th = hck - GAP
        bw = tw - GAP
        zc = zBt + (ct + 0.5) * hck
        yv = -Ds[kt]
        sunk = 0.011
        fs = min(th * 0.42, (bw - 0.11) / max(1, len(body)) / 0.62)
        tb = box("tablet", (bw, 0.06, th), loc=(0, yv + sunk + 0.03, zc))
        fr = shape_extrude("tablet_frame", rounded_rect(bw - 0.003, th - 0.003, 0.004, 1), [rounded_rect(bw - 0.062, th - 0.062, 0.003, 1)],
                           depth=sunk + 0.006, loc=(0, yv + sunk, zc), rot=(90, 0, 0), bevel=0.004)
        sh = shape_extrude("tablet_margin", rounded_rect(bw - 0.058, th - 0.058, 0.003, 1), [rounded_rect(bw - 0.09, th - 0.09, 0.002, 1)],
                           depth=0.0015, loc=(0, yv + sunk, zc), rot=(90, 0, 0))
        tx = text("tablet_date", body, size=fs, depth=0.012, loc=(0, yv + sunk, zc - fs * 0.5), font="roman")
        assign(tb, st_m[1])
        paint(tb, grime=0.24, special=mask_mul(noise_mask(0.1, seed=seed + 8), 0.6 * mossp))
        assign(sh, st_m[1])
        paint(sh, grime=0.5)
        assign(fr, st_m[0])
        paint(fr, grime=mask_mul(height_ramp(zc + th / 2, zc - th / 2), 0.2), wear=0.3)
        assign(tx, st_m[2])
        paint(tx, wear=0.7)
        if kt == k:                                     # a lantern bracket goes above the tablet, not over it
            zl = max(zl, min(zc + th / 2 + 0.3, zm[k] - 0.1))
    socket("lantern", "attach", (0.0, -Ds[k], zl), role="lantern")
    socket("top", "attach", (0.0, -Ds[-1] * 0.5, H), role="prop_top")
