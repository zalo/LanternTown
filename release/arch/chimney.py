"""Masonry chimney stack with terracotta pots, to be sunk into a roof. A brick (or rubble-stone with dressed
quoins) shaft rises from a wider base over a stone weathering offset, carries a projecting string, a dentil
course and corbelled oversailing courses (moulded stone cornice on the stone stack), and is finished with a
hand-trowelled mortar flaunching in which 1-4 hollow clay pots of different patterns are bedded. Code-4 style
lead flashing (front apron, stepped side pieces, back gutter) marks the roof line; an iron tie band holds the
old stack together. Soot, damp and moss are painted from the params; the whole stack can lean.
"""
from math import tan, radians, degrees, cos, sin, pi, sqrt, ceil, exp

import numpy as np
from mathutils import Vector

from tk import *

META = {"name": "Chimney stack with pots", "class": "component", "mount": "ground", "focus": 0.8}
PARAMS = {
    "width": {"default": 0.9, "min": 0.45, "max": 1.8, "desc": "width of the stack at its base, along X (m)"},
    "depth": {"default": 0.56, "min": 0.4, "max": 1.0, "desc": "depth of the stack at its base, along Y (m)"},
    "height": {"default": 2.4, "min": 1.0, "max": 4.5,
               "desc": "base to the top of the mortar collar the pots are bedded in (m); the pots rise above it"},
    "material": {"default": "brick_red", "choices": ["brick_red", "brick_ochre", "stone"], "desc": "stack masonry"},
    "pots": {"default": 2, "min": 1, "max": 4, "desc": "number of chimney pots"},
    "pot_style": {"default": "mixed", "choices": ["mixed", "roll", "cannon", "louvre", "crown", "castle", "octagon", "cowl"],
                  "desc": "pot pattern (mixed = a different design per flue)"},
    "pot_height": {"default": 0.55, "min": 0.3, "max": 0.95, "desc": "nominal pot height (m)"},
    "corbels": {"default": 3, "min": 1, "max": 4, "desc": "number of oversailing courses of the cap"},
    "dentils": {"default": True, "desc": "dentil course of projecting headers under the oversailing courses (brick)"},
    "shoulder": {"default": 0.42, "min": 0.0, "max": 0.6,
                 "desc": "height fraction where the stack steps in over a stone weathering (0 = straight shaft)"},
    "lean": {"default": 1.2, "min": -6.0, "max": 6.0, "desc": "lean of the top of the stack towards +X (deg)"},
    "soot": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "soot staining of pots and cap"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "moss / damp on ledges and the back (north) face"},
    "strap": {"default": True, "desc": "wrought-iron tie band round the shaft"},
    "flashing": {"default": True, "desc": "lead flashing at the roof line"},
    "sink": {"default": 0.5, "min": 0.2, "max": 1.8,
             "desc": "height above the base where the roof surface crosses the stack centre (m)"},
    "pitch": {"default": 45.0, "min": 20.0, "max": 60.0, "desc": "roof pitch the flashing is dressed to; the roof falls towards -Y (deg)"},
    "roof_stub": {"default": "none", "choices": ["none", "tiles", "slate"],
                  "desc": "patch of roof around the stack for stand-alone use (none when a building provides the roof)"},
    "smoke": {"default": True, "desc": "smoke fx socket on the first pot"},
}
VARIANTS = {
    "ochre_four_slate": {"material": "brick_ochre", "width": 1.3, "depth": 0.6, "height": 3.1, "pots": 4, "corbels": 4,
                         "lean": 0.0, "soot": 0.95, "sink": 1.0, "roof_stub": "slate", "pot_height": 0.7, "seed": 4},
    "stone_squat_crown": {"material": "stone", "width": 0.85, "depth": 0.72, "height": 1.7, "pots": 1, "pot_style": "crown",
                          "shoulder": 0.0, "strap": False, "moss": 0.95, "soot": 0.4, "lean": -0.5, "pitch": 35.0,
                          "sink": 0.42, "pot_height": 0.62, "seed": 11},
    "leaning_single": {"width": 0.5, "depth": 0.5, "height": 2.7, "pots": 1, "pot_style": "cannon", "corbels": 2,
                       "lean": 5.0, "soot": 1.0, "shoulder": 0.3, "pot_height": 0.85, "seed": 23},
    "tiled_roof_three": {"width": 1.2, "depth": 0.6, "height": 2.9, "pots": 3, "pitch": 50.0, "sink": 1.05,
                         "roof_stub": "tiles", "lean": 2.0, "moss": 0.7, "pot_height": 0.5, "seed": 31},
}

COURSE = 0.075      # brick course module of the brick_* materials (m)
HALF = 0.1125       # half-brick module along a course (m)
JOINT = 0.010       # mortar joint (m)
STEP = 0.028        # oversail of one corbel course (m)

# (tangent, outward normal) of the four faces: 0 front (-Y), 1 right (+X), 2 back (+Y), 3 left (-X).
# Corner k is where face k starts: 0 front-left, 1 front-right, 2 back-right, 3 back-left.
SIDES = (((1, 0), (0, -1)), ((0, 1), (1, 0)), ((-1, 0), (0, 1)), ((0, -1), (-1, 0)))

# Whole stretchers of the brick_ochre texture (row, first cell, last cell in half-brick cells): single modelled
# bricks are UV-mapped inside one of these so no mortar joint of the texture ever crosses a modelled brick.
OCHRE_S = ((11, 1.87, 3.70), (11, 4.85, 6.74), (7, 1.87, 3.70), (7, 4.85, 6.74), (10, 0.34, 2.20), (10, 3.34, 5.18),
           (8, 0.34, 2.20), (8, 3.34, 5.18), (9, 1.87, 3.69), (9, 4.85, 6.67))


# --------------------------------------------------------------------------------------- helpers
def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _s1(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0 + 1e-9)))
    return t * t * (3.0 - 2.0 * t)


def mesh_uv(name, verts, faces, uvs):
    """Mesh with hand-written per-corner UVs (meters), one (u, v) per face corner in face order."""
    o = mesh_object(name, verts, faces)
    me = o.data
    if len(me.loops) == len(uvs):
        lay = me.uv_layers.new(name="UVMap")
        lay.data.foreach_set("uv", [float(v) for uv in uvs for v in uv])
        uv_keep(o)
    else:
        uv_box(o)
    return o


def rect_loft(name, rings, top=True):
    """Loft through rectangles [(half_x, half_y, z), ...]: perfectly mitred masonry profiles."""
    verts, faces = [], []
    for (hx, hy, z) in rings:
        verts += [(-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)]
    for i in range(len(rings) - 1):
        a, b = i * 4, (i + 1) * 4
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((a + k, a + k2, b + k2, b + k))
    if top:
        e = (len(rings) - 1) * 4
        faces.append((e, e + 1, e + 2, e + 3))
    return mesh_object(name, verts, faces)


def prism_x(name, yz, x0, x1, tip=0.0, drop=0.0):
    """Convex polygon in the YZ plane extruded from x0 to x1 (lead sheet pieces). With tip > 0 the last `tip`
    metres at both ends are dressed down by `drop` (sheet ends beaten down into the roof covering)."""
    n = len(yz)
    if tip > 0:
        m = max(2, int(ceil((x1 - x0 - 2 * tip) / 0.09)))
        st = [(x0, drop)] + [(lerp(x0 + tip, x1 - tip, i / m), 0.0) for i in range(m + 1)] + [(x1, drop)]
    else:
        st = [(x0, 0.0), (x1, 0.0)]
    verts = [(x, y, z - d) for (x, d) in st for (y, z) in yz]
    e = (len(st) - 1) * n
    faces = [tuple(range(n)), tuple(range(e + n - 1, e - 1, -1))]
    for k in range(len(st) - 1):
        faces += [(k * n + i, k * n + (i + 1) % n, (k + 1) * n + (i + 1) % n, (k + 1) * n + i) for i in range(n)]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def taper(r0, z0, r1, z1, k):
    return [(lerp(r0, r1, (i + 1) / k), lerp(z0, z1, (i + 1) / k)) for i in range(k)]


def arc(cr, cz, r, a0, a1, k):
    return [(cr + r * cos(radians(lerp(a0, a1, i / (k - 1)))), cz + r * sin(radians(lerp(a0, a1, i / (k - 1))))) for i in range(k)]


# --------------------------------------------------------------------------------------- masonry
def shaft_grid(name, secs, spalls, zoff, warp):
    """Plain brick shaft: one quad grid per section (half_x, half_y, z0, z1), bed joints of the texture on the
    course grid (V = z - zoff) and U fitted per face to a whole number of half bricks, so every course turns the
    arris with a stretcher on one face and its header on the other. spalls[section] = [(corner, z, depth)]
    knocks a brick-sized piece out of an arris; warp(x, y, z) -> (dx, dy) puts the faces out of true."""
    verts, faces, uvs = [], [], []
    c = COURSE
    for si, (hx, hy, z0, z1) in enumerate(secs):
        cols, ubase, usc, lens = [], [], [], []
        u0 = 0.005
        for side in range(4):
            L = 2 * (hx if side % 2 == 0 else hy)
            e = min(0.07, L * 0.2)
            n_ = max(1, int(ceil((L - 2 * e) / 0.17)))
            cols += [(side, t) for t in [0.0] + [e + (L - 2 * e) * i / n_ for i in range(n_ + 1)]]
            N = max(1, int(round(L / HALF)))
            ubase.append(u0)
            usc.append((N * HALF - 0.010) / L)
            lens.append(L)
            u0 = (u0 + (N + 1) * HALF) % 0.9
        zs = {round(z0, 4), round(z1, 4)}
        k = int(ceil((z0 + 0.04 - zoff) / (2 * c)))
        while zoff + k * 2 * c < z1 - 0.04:
            zs.add(round(zoff + k * 2 * c, 4))
            k += 1
        moved = {}
        for (corner, za, d) in spalls.get(si, ()):
            for zz in (za, za + 0.007, za + c - 0.007, za + c):
                zs.add(round(zz, 4))
            moved[(round(za + 0.007, 4), corner)] = d
            moved[(round(za + c - 0.007, 4), corner)] = d * 0.45
        zs = sorted(zs)
        base, M = len(verts), len(cols)
        for z in zs:
            for (side, t) in cols:
                (tx, ty), (nx_, ny_) = SIDES[side]
                ht, hn = (hx, hy) if side % 2 == 0 else (hy, hx)
                x, y = (t - ht) * tx + hn * nx_, (t - ht) * ty + hn * ny_
                d = moved.get((z, side)) if t == 0.0 else None
                if d:
                    x -= d * (1 if x > 0 else -1)
                    y -= d * (1 if y > 0 else -1)
                dx, dy = warp(x, y, z)
                verts.append((x + dx, y + dy, z))
        for r_ in range(len(zs) - 1):
            for i in range(M):
                i2 = (i + 1) % M
                side, t = cols[i]
                t2 = cols[i2][1] if cols[i2][0] == side else lens[side]
                faces.append((base + r_ * M + i, base + r_ * M + i2, base + (r_ + 1) * M + i2, base + (r_ + 1) * M + i))
                ua, ub = ubase[side] + t * usc[side], ubase[side] + t2 * usc[side]
                va, vb = zs[r_] - zoff, zs[r_ + 1] - zoff
                uvs += [(ua, va), (ub, va), (ub, vb), (ua, vb)]
    return mesh_uv(name, verts, faces, uvs)


class Bricks:
    """Individually modelled bricks (front, beds and ends; the back is buried in the mortar core). Each brick is
    UV-mapped inside ONE brick of the wall texture, so projecting courses always show whole bricks."""

    def __init__(self, mname, r):
        self.v, self.f, self.uv = [], [], []
        self.mname, self.r = mname, r

    def rect(self):
        r = self.r
        if self.mname == "brick_ochre":
            j, a, b = r.pick(OCHRE_S)
            u0, u1 = (a + 0.14) * HALF, (b - 0.14) * HALF
        else:
            j, k = min(11, int(r.u(0.0, 12.0))), min(3, int(r.u(0.0, 4.0)))
            cx = k * 0.225 + (0.0 if j % 2 else HALF)
            u0, u1 = cx - 0.096, cx + 0.096
        return u0, u1, j * COURSE + 0.010, (j + 1) * COURSE - 0.010

    def add(self, side, hx, hy, a0, a1, proj, z0, z1, dp, sag=0.0, chip=None):
        """Brick on face `side` of the rectangle (hx, hy) grown by proj; a0..a1 along the face, dp deep."""
        (tx, ty), (nx_, ny_) = SIDES[side]
        f = (hy if side % 2 == 0 else hx) + proj
        U0, U1, V0, V1 = self.rect()
        r = self.r
        wu = U1 - U0
        ua = (U0, U1)
        if a1 - a0 < 0.16:
            w_ = min(wu, (a1 - a0) * 0.92)
            s_ = U0 + r.u(0.0, wu - w_)
            ua = (s_, s_ + w_)
        w_ = min(wu, dp * 0.92)
        s_ = U0 + r.u(0.0, wu - w_)
        ue = (s_, s_ + w_)
        V = (V0, V1)
        base = len(self.v)
        for ia in (0, 1):
            for inn in (0, 1):
                for iz in (0, 1):
                    a = a1 if ia else a0
                    n = f if inn else f - dp
                    z = (z1 if iz else z0) - (sag if ia else 0.0)
                    if chip and inn and chip[0] == ia and chip[1] == iz:
                        a += chip[2] * (-1 if ia else 1)
                        n -= chip[2] * 0.7
                        z += chip[2] * (-0.5 if iz else 0.5)
                    self.v.append((a * tx + n * nx_, a * ty + n * ny_, z))
        quads = (("f", ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))), ("t", ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))),
                 ("b", ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))), ("e", ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),
                 ("e", ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))))
        for kind, q in quads:
            self.f.append(tuple(base + ia * 4 + inn * 2 + iz for (ia, inn, iz) in q))
            for (ia, inn, iz) in q:
                if kind == "f":
                    self.uv.append((ua[ia], V[iz]))
                elif kind == "t":
                    self.uv.append((ua[ia], V[inn]))
                elif kind == "b":
                    self.uv.append((ua[ia], V[1 - inn]))
                else:
                    self.uv.append((ue[inn], V[iz]))

    def course(self, hx, hy, proj, z0, z1, owner, kind="stretcher", events=()):
        """One course round the rectangle (hx, hy) grown by proj. The faces of parity `owner` run through to the
        arrises (their end bricks show as headers on the other two faces), as in real bond. kind: 'stretcher'
        (odd half-brick counts get one header), 'header', or 'dentil' (headers, every other one projecting)."""
        r = self.r
        HX, HY = hx + proj, hy + proj
        for side in range(4):
            ht, ha = (HX, HY) if side % 2 == 0 else (HY, HX)
            L, La = 2 * ht, 2 * ha
            own = (side % 2) == owner
            items = []                                   # (a0, a1, projecting, corner depth or None)
            if kind == "dentil":
                N = max(3, int(round(L / HALF)))
                pt = L / N
                Na = max(3, int(round(La / HALF)))
                cd = (La / Na if Na % 2 else La / Na / 2) - JOINT / 2
                if N % 2:
                    for i in range(N):
                        if own or 0 < i < N - 1:
                            items.append((-ht + i * pt, -ht + (i + 1) * pt, i % 2 == 1, cd if i in (0, N - 1) else None))
                else:
                    if own:
                        items.append((-ht, -ht + pt / 2, False, cd))
                    for i in range(N - 1):
                        items.append((-ht + pt / 2 + i * pt, -ht + pt / 2 + (i + 1) * pt, i % 2 == 0, None))
                    if own:
                        items.append((ht - pt / 2, ht, False, cd))
            else:
                if own:
                    lo, hi = -ht, ht
                else:
                    cw = min(0.1025, L * 0.3) + JOINT / 2
                    lo, hi = -ht + cw, ht - cw
                N = max(1, int(round((hi - lo) / HALF)))
                u = (hi - lo) / N
                if kind == "header" or N == 1:
                    seq = [1] * N
                else:
                    seq = [2] * (N // 2)
                    if N % 2:
                        seq.insert(min(len(seq), int(r.u(0.0, len(seq) + 1.0))), 1)
                a = lo
                for idx, w in enumerate(seq):
                    corner = own and idx in (0, len(seq) - 1)
                    items.append((a, a + w * u, False, min(0.1025, La * 0.3) if corner else None))
                    a += w * u
            for (a0, a1, pr, cd) in items:
                b0 = a0 if a0 < -ht + 1e-6 else a0 + JOINT / 2
                b1 = a1 if a1 > ht - 1e-6 else a1 - JOINT / 2
                pj = proj + r.jitter(0.0, 0.0015)
                dp = cd if cd else 0.055
                if kind == "dentil":
                    pj += (STEP - 0.003) if pr else 0.0015
                    dp += STEP if (pr and not cd) else 0.0
                dz = r.jitter(0.0, 0.001)
                sag, chip = 0.0, None
                for (ek, es, ef) in events:
                    pos = -ht + ef * L
                    if es != side or not (a0 - 1e-6 <= pos <= a1 + 1e-6):
                        continue
                    if ek == "slip":
                        pj += 0.014
                        sag = 0.008
                    elif ek == "gap":
                        b1 -= 0.016
                    elif ek == "chip" and cd:
                        chip = (0 if ef < 0.5 else 1, 1 if r.chance(0.65) else 0, r.u(0.024, 0.036))
                self.add(side, hx, hy, b0, b1, pj, z0 + JOINT / 2 + dz, z1 - JOINT / 2 + dz, dp, sag, chip)

    def mesh(self, name):
        return mesh_uv(name, self.v, self.f, self.uv)


def moulding(name, prof, hx, hy, r, max_len=0.5):
    """Dressed-stone band round the rectangle (hx, hy): profile [(out, z), ...] (bottom-inner -> outside -> top-inner)
    cut into separate stones per face with open joints and knocked arrises at the joints; mitred returns at the corners."""
    verts, faces = [], []
    P = len(prof)
    amax = max(a for (a, _) in prof)
    for side in range(4):
        (tx, ty), (nx_, ny_) = SIDES[side]
        ht, hn = (hx, hy) if side % 2 == 0 else (hy, hx)
        L = 2 * ht
        ns = max(1, int(round(L / max_len)))
        cuts = [-ht] + [-ht + L * (i + r.jitter(0.0, 0.13)) / ns for i in range(1, ns)] + [ht]
        for si in range(ns):
            dz = r.jitter(0.0, 0.002)
            base = len(verts)
            for (tt, mitre, sg) in ((cuts[si], si == 0, -1), (cuts[si + 1], si == ns - 1, 1)):
                kn = r.u(0.004, 0.013) if r.chance(0.7) else 0.0
                for (a, z) in prof:
                    t = tt + sg * a if mitre else tt - sg * (0.003 + (kn if a > amax - 0.012 else 0.0))
                    verts.append((t * tx + (hn + a) * nx_, t * ty + (hn + a) * ny_, z + dz))
            for i in range(P - 1):
                faces.append((base + i, base + P + i, base + P + i + 1, base + i + 1))
            if si > 0:
                faces.append(tuple(base + i for i in range(P)))
            if si < ns - 1:
                faces.append(tuple(base + P + i for i in reversed(range(P))))
    return mesh_object(name, verts, faces)


def flaunching(name, rings, hxi, hyi, z_top, holes, ds=0.06):
    """Mortar haunch lofted through `rings` up to a flat top (hxi, hyi, z_top) that is a fine grid cut out and
    dressed up round each pot (holes = [(x, y, r)]), so no vertex is buried under a pot."""
    nx, ny = max(2, int(ceil(2 * hxi / ds))), max(2, int(ceil(2 * hyi / ds)))
    verts, faces = [], []
    snapped = set()
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y, z = -hxi + 2 * hxi * i / nx, -hyi + 2 * hyi * j / ny, z_top
            for (px, py, rr) in holes:
                d = sqrt((x - px) ** 2 + (y - py) ** 2)
                if d < rr:
                    snapped.add(i * (ny + 1) + j)
                    if 0 < i < nx and 0 < j < ny and d > 1e-4:
                        x, y, z = px + (x - px) / d * rr, py + (y - py) / d * rr, z_top + 0.004
                    break
            verts.append((x, y, z))
    for i in range(nx):
        for j in range(ny):
            q = (i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1)
            if not all(k in snapped for k in q):
                faces.append(q)
    per = [(i, 0) for i in range(nx)] + [(nx, j) for j in range(ny)] + [(i, ny) for i in range(nx, 0, -1)] + [(0, j) for j in range(ny, 0, -1)]
    M = len(per)
    prev = None
    for (hx, hy, z) in rings:
        cur = list(range(len(verts), len(verts) + M))
        verts += [((2 * i / nx - 1) * hx, (2 * j / ny - 1) * hy, z) for (i, j) in per]
        if prev:
            faces += [(prev[k], prev[(k + 1) % M], cur[(k + 1) % M], cur[k]) for k in range(M)]
        prev = cur
    cur = [i * (ny + 1) + j for (i, j) in per]
    faces += [(prev[k], prev[(k + 1) % M], cur[(k + 1) % M], cur[k]) for k in range(M)]
    return mesh_object(name, verts, faces)


# --------------------------------------------------------------------------------------- pots
def pot_roll(rb, hp, seg, w):
    rt = rb * 0.8
    zi = max(0.04, hp - 0.3)
    prof = [(rb, 0.0), (rb, 0.03), (rb - 0.008, 0.045)] + taper(rb - 0.008, 0.045, rt, hp - 0.05, 3)
    prof += arc(rt + 0.002, hp - 0.024, 0.024, -70, 115, 6)
    prof += [(rt - w, hp - 0.012), (rt - w, zi), (0.0, zi)]
    return [lathe("pot_roll", prof, segments=seg)], None, []


def pot_cannon(rb, hp, seg, w):
    r0, rn, zn = rb * 0.92, rb * 0.66, hp * 0.86
    zi = max(0.04, hp - 0.32)

    def rr(z):
        return lerp(r0, rn, (z - 0.055) / (zn - 0.055))

    prof = [(rb, 0.0), (rb, 0.04), (r0, 0.055)]
    for zb in (hp * 0.3, hp * 0.6):
        prof += [(rr(zb), zb), (rr(zb) + 0.009, zb + 0.008), (rr(zb) + 0.009, zb + 0.028), (rr(zb + 0.036), zb + 0.036)]
    prof += [(rn, zn), (rn + 0.004, hp * 0.9), (rb * 0.8, hp * 0.96), (rb * 0.87, hp - 0.012), (rb * 0.88, hp),
             (rb * 0.88 - w, hp), (rn - w + 0.006, hp * 0.91), (rn - w, zn), (rn - w, zi), (0.0, zi)]
    return [lathe("pot_cannon", prof, segments=seg)], None, []


def pot_octagon(rb, hp, seg, w):
    rt = rb * 0.74
    zi = max(0.04, hp - 0.3)
    prof = [(rb * 1.04, 0.0), (rb * 1.04, 0.06), (rb * 0.94, 0.085)] + taper(rb * 0.94, 0.085, rt, hp - 0.13, 3)
    prof += [(rt + 0.006, hp - 0.12), (rb * 0.93, hp - 0.075), (rb * 0.95, hp - 0.06), (rb * 0.95, hp - 0.035),
             (rb * 0.88, hp - 0.028), (rb * 0.88, hp), (rb * 0.88 - w, hp), (rt - w, hp - 0.12), (rt - w, zi), (0.0, zi)]
    return [lathe("pot_oct", prof, segments=8, rot=(0, 0, 22.5), smooth_angle=20)], None, []


def pot_crown(rb, hp, seg, w, castle=False):
    zn, rn = hp * 0.58, rb * 0.68
    zi = max(0.04, zn - 0.25)
    prof = [(rb, 0.0), (rb, 0.035), (rb * 0.93, 0.05)] + taper(rb * 0.93, 0.05, rn, zn, 3)
    prof += [(rn + 0.012, zn + 0.008), (rn + 0.017, zn + 0.02), (rn + 0.012, zn + 0.034), (rn - w, zn + 0.03), (rn - w, zi), (0.0, zi)]
    body = lathe("pot_crown", prof, segments=seg)
    nt = 8 if rb > 0.1 else 6
    nu = nt * 6
    z0 = zn + 0.02
    h_hi = hp - z0
    h_lo = h_hi * (0.66 if castle else 0.58)
    r0, r1 = rn + 0.004, rb * 1.04
    pos = [0.0, 0.05, 0.25, 0.45, 0.5, 0.75]

    def fn(u, v):
        i = int(round(u * nu)) % nu
        k, j = divmod(i, 6)
        if castle:
            ang = 2 * pi * (k + pos[j]) / nt
            top = h_hi if j in (1, 2, 3) else h_lo
        else:
            ang = 2 * pi * i / nu
            top = h_lo + (h_hi - h_lo) * (1.0 - abs(j / 6.0 - 0.5) * 2.0) ** 0.85
        zz = v * top
        r = r0 + (r1 - r0) * (zz / h_hi) ** 1.5
        return (r * cos(ang), r * sin(ang), z0 + zz)

    crown = surface("pot_crown_top", fn, nu, 4, closed_u=True, thickness=w * 0.9)
    return [body, crown], None, []


def pot_louvre(rb, hp, seg, w):
    z1 = hp * 0.46
    rc, ro = rb * 0.5, rb * 0.92
    n = 3
    pp = (hp - 0.05 - z1) / (n + 0.7)
    prof = [(rb, 0.0), (rb, 0.035), (rb * 0.93, 0.05)] + taper(rb * 0.93, 0.05, rb * 0.84, z1 - 0.03, 2)
    prof += [(rb * 0.84 + 0.012, z1 - 0.02), (rb * 0.84 + 0.012, z1 - 0.005), (rc, z1 + 0.012), (rc, z1 + n * pp + 0.6 * pp), (0.0, z1 + n * pp + 0.6 * pp)]
    objs = [lathe("pot_louvre", prof, segments=seg)]
    for k in range(n):
        zk = z1 + k * pp
        objs.append(lathe("louvre", [(ro - 0.014, zk + 0.3 * pp), (ro, zk + 0.3 * pp), (ro + 0.004, zk + 0.36 * pp), (rc + 0.012, zk + pp),
                                     (rc, zk + pp), (rc, zk + 0.9 * pp)], segments=seg))
    zk = z1 + n * pp
    objs.append(lathe("louvre_cap", [(ro - 0.014, zk + 0.3 * pp), (ro, zk + 0.3 * pp), (ro + 0.004, zk + 0.36 * pp), (rb * 0.34, hp - 0.036),
                                     (rb * 0.24, hp - 0.03), (rb * 0.2, hp - 0.012), (rb * 0.12, hp - 0.002), (0.0, hp)], segments=seg))
    zr0, zr1 = z1 - 0.005, zk + 0.5 * pp
    for k in range(4):
        a = radians(45 + 90 * k)
        R = (rc + ro) * 0.5 - 0.008
        objs.append(box("louvre_rib", (0.016, ro - rc, zr1 - zr0), loc=(R * cos(a), R * sin(a), (zr0 + zr1) / 2), rot=(0, 0, degrees(a) - 90), bevel=0.003, segments=1))
    return objs, (rc * 1.25, z1, zk + 0.6 * pp), []


def pot_cowl(rb, hp, seg, w):
    """Plain roll-top pot with a lead-sheet rain cowl riveted on three iron legs."""
    objs, core, _ = pot_roll(rb, hp, seg, w)
    rt = rb * 0.8
    rh = rb * 1.12
    zb = hp + 0.085
    hat = lathe("cowl_hat", [(0.0, zb + 0.062), (rh - 0.004, zb), (rh, zb), (rh + 0.003, zb + 0.006), (0.022, zb + 0.078), (0.016, zb + 0.1), (0.0, zb + 0.104)], segments=seg)
    recalc_normals(hat)
    extras = [(hat, "lead")]
    for k in range(3):
        a = radians(30 + 120 * k)
        r0, r1 = rt + 0.024, rh * 0.8
        leg = sweep("cowl_leg", [(r0 * cos(a), r0 * sin(a), hp - 0.11), (r0 * cos(a), r0 * sin(a), hp - 0.03), (r1 * cos(a), r1 * sin(a), zb + 0.012)],
                    0.0045, sides=6, smooth_path=False)
        extras.append((leg, "iron_wrought"))
    extras.append((ring_band("cowl_clamp", radius=rt + 0.024, width=0.022, thick=0.003, loc=(0, 0, hp - 0.09), segments=seg), "iron_wrought"))
    return objs, core, extras


POTS = {"roll": (pot_roll, 0.62), "cowl": (pot_cowl, 0.6), "cannon": (pot_cannon, 1.3), "louvre": (pot_louvre, 1.0),
        "crown": (pot_crown, 0.95), "castle": (lambda rb, hp, seg, w: pot_crown(rb, hp, seg, w, True), 0.9), "octagon": (pot_octagon, 1.12)}


# --------------------------------------------------------------------------------------- build
def build(p, rng):
    W, D, H = float(p.width), float(p.depth), float(p.height)
    stone = p.material == "stone"
    seed = int(p.seed)
    c = COURSE
    soot, mossy = float(p.soot), float(p.moss)

    m_body = mat("rubble_fieldstone") if stone else mat(p.material)
    m_dress = mat("stone_smooth", "#b3a892" if stone else "#a89c87")
    m_mortar = mat("mortar", "#a39c90")
    m_lead = mat("lead")
    m_iron = mat("iron_wrought")

    # ---- vertical layout: everything lands on brick courses counted down from the cap
    fl = 0.06                                      # flaunching height
    z_mas = H - fl
    zoff = z_mas % c                               # course grid offset (the cut course is at the hidden base)
    n = int(min(p.corbels, max(1, int((H - 0.7) / 0.3))))
    z_c0 = z_mas - (n + 2) * c                     # underside of the first oversailing course
    has_string = H >= 1.5
    zs1 = z_c0 - 3 * c
    zs0 = zs1 - c
    shaft_top = zs0 if has_string else z_c0
    hw0, hd0 = W / 2, D / 2
    dent_on = (not stone) and bool(p.dentils) and z_c0 - c > (zs1 if has_string else 0.0) + 0.01
    z_bt = z_c0 + 0.03 if stone else (z_c0 - c if dent_on else z_c0)       # top of the plain shaft

    # ---- roof line (falls towards -Y); pitch / sink are fitted so the flashing always sits on the shaft
    T = tan(radians(p.pitch))
    toff = 0.014 if stone else 0.003               # lead stands this far off the masonry face
    th = 0.004                                     # lead thickness
    zb_max = shaft_top - 0.24
    run_f = 0.16 / sqrt(1 + T * T) + toff + th
    T = min(T, max(0.05, (zb_max - 0.03) / (D + run_f)))
    cs = 1 / sqrt(1 + T * T)
    run_f = 0.16 * cs + toff + th
    s = min(max(float(p.sink), T * (hd0 + run_f) + 0.03), zb_max - T * hd0)

    def zr(y):
        return s + T * y

    z_rb = zr(hd0)

    # ---- shoulder: the shaft steps in above a weathered stone offset
    sh, z_sh = 0.0, 0.0
    if p.shoulder > 0.01:
        z_sh = z_mas - round((z_mas - max(p.shoulder * H, z_rb + 0.26)) / c) * c
        if z_sh <= shaft_top - 0.36 and min(W, D) - 0.112 >= 0.3:
            sh = 0.056
    hw1, hd1 = hw0 - sh, hd0 - sh

    # ---- tie band heights (centred on a course)
    z_lo = (z_sh + 0.1) if sh else (z_rb + 0.22)
    bands = []
    if p.strap and shaft_top - z_lo > 0.3:
        fr = [(0.7, 0.3), (0.28, -0.3)] if shaft_top - z_lo > 1.05 else [(0.55, 0.3)]
        bands = [(zoff + (round((lerp(z_lo, shaft_top, f) - zoff) / c - 0.5) + 0.5) * c, hw1 * fx_) for (f, fx_) in fr]

    # ---- the faces are a few mm out of true: slow wander plus a belly, dying out at the roof line and the cap
    r_w = rng.fork("wander")
    ph = [r_w.u(0.0, 6.283) for _ in range(4)]
    w_amp = 0.0 if stone else 1.0
    z_bel = lerp(z_rb + 0.3, shaft_top - 0.25, r_w.u(0.3, 0.7))

    def warp(x, y, z):
        env = w_amp * _s1(z_rb + 0.08, z_rb + 0.5, z) * (1.0 - _s1(shaft_top - 0.3, shaft_top - 0.05, z))
        if env <= 0.0:
            return 0.0, 0.0
        bel = 0.007 * exp(-((z - z_bel) / 0.3) ** 2)
        dx = 0.005 * (sin(2.3 * z + ph[0]) + 0.6 * sin(5.1 * z + 1.7 * y + ph[1])) + bel * max(-1.0, min(1.0, x / hw1))
        dy = 0.004 * (sin(2.9 * z + ph[2]) + 0.6 * sin(4.3 * z + 1.9 * x + ph[3])) + bel * max(-1.0, min(1.0, y / hd1))
        return dx * env, dy * env

    r_j = rng.fork("courses")
    r_e = rng.fork("erosion")
    body_objs, brick_objs, core_objs, dress_objs, lead_objs, iron_objs, mortar_objs, rust_objs = [], [], [], [], [], [], [], []

    # ---- masonry body
    if stone:
        rings = [(hw0, hd0, 0.0)]
        if sh:
            rings += [(hw0, hd0, z_sh), (hw1, hd1, z_sh)]
        rings += [(hw1, hd1, z_bt)]
        body = rect_loft("stack", rings, top=False)
        densify(body, 0.17 if W * H < 3.2 else 0.22)
        displace_noise(body, strength=0.007, scale=0.3, seed=seed)
        bevel(body, width=0.012, segments=1)
        uv_box(body, offset=None)
        top_off = 0.014                                        # half-size growth of the cap top
    else:
        secs = [(hw0, hd0, 0.0, z_sh), (hw1, hd1, z_sh, z_bt)] if sh else [(hw0, hd0, 0.0, z_bt)]
        up_i = len(secs) - 1

        def course_z(lo, hi, f):
            zz = zoff + round((lerp(lo, hi, f) - zoff) / c) * c
            for (zb_, _x) in bands:                            # never under a tie band
                if abs(zz + c / 2 - zb_) < 0.09:
                    zz -= 2 * c
            return zz

        spalls = {}
        lo_u, hi_u = max(secs[up_i][2] + 0.15, z_rb + 0.3), shaft_top - 0.25
        front = r_e.pick([0, 1])
        if hi_u - lo_u > 0.15:                                 # frost-spalled brick corners on the arrises
            za = course_z(lo_u, hi_u, r_e.u(0.3, 0.85))
            spalls[up_i] = [(front, za, r_e.u(0.032, 0.046)), (r_e.pick([2, 3]), za, r_e.u(0.028, 0.04))]
            if sh and (z_sh - 0.3) - (z_rb + 0.3) > 0.1:
                spalls[0] = [(1 - front, course_z(z_rb + 0.3, z_sh - 0.3, r_e.u(0.2, 0.9)), r_e.u(0.032, 0.046))]
            elif hi_u - lo_u > 0.7:
                zb_ = course_z(lo_u, hi_u, r_e.u(0.0, 0.2))
                if abs(zb_ - za) > 0.2:
                    spalls[up_i].append((1 - front, zb_, r_e.u(0.032, 0.046)))
        body = shaft_grid("stack", secs, spalls, zoff, warp)
        bevel(body, width=0.005, segments=1)
        top_off = (n - 1) * STEP + 0.004
    body_objs.append(body)

    # ---- brick string, dentil course and oversailing cap: individually modelled bricks on a recessed mortar core
    if not stone:
        B = Bricks(p.material, rng.fork("bricks"))
        rec = 0.007
        if has_string:
            B.course(hw1, hd1, 0.02, zs0, zs1, 1, events=[("gap", r_e.pick([0, 1]), r_e.u(0.3, 0.7))])
            core_objs.append(rect_loft("string_core", [(hw1 - 0.01, hd1 - 0.01, zs0 + 0.001), (hw1 + 0.02 - rec, hd1 + 0.02 - rec, zs0 + 0.001),
                                                       (hw1 + 0.02 - rec, hd1 + 0.02 - rec, zs1 - 0.001), (hw1 - 0.01, hd1 - 0.01, zs1 - 0.001)], top=False))
        courses = [(0.0, z_c0 - c, z_c0, "dentil")] if dent_on else []
        for k in range(1, n + 1):
            courses.append((k * STEP, z_c0 + (k - 1) * c, z_c0 + k * c, "header" if (k == n and n >= 3) else "stretcher"))
        courses.append((n * STEP, z_c0 + n * c, z_c0 + (n + 1) * c, "stretcher"))
        courses.append((top_off, z_c0 + (n + 1) * c, z_mas, "stretcher"))
        first = 1 if dent_on else 0
        events = {i: [] for i in range(len(courses))}
        i_slip = first + min(n - 1, int(r_e.u(0.0, n)))                               # one slipped brick in the oversail
        events[i_slip].append(("slip", r_e.pick([0, 1]), r_e.u(0.3, 0.7)))
        events[len(courses) - 2].append(("gap", r_e.pick([0, 1]), r_e.u(0.25, 0.75)))  # one open perpend
        for i in sorted(range(first, len(courses)), key=lambda _i: r_e.u())[:3]:       # knocked-off corner bricks
            events[i].append(("chip", 0, r_e.pick([0.0, 1.0])) if i % 2 == 0 else r_e.pick([("chip", 1, 0.0), ("chip", 3, 1.0)]))
        rings = [(hw1 - 0.015, hd1 - 0.015, z_bt - 0.002)]
        for i, (pj, za, zb_, kind) in enumerate(courses):
            B.course(hw1, hd1, pj, za, zb_, i % 2, kind=kind, events=events[i])
            rings += [(hw1 + pj - rec, hd1 + pj - rec, za), (hw1 + pj - rec, hd1 + pj - rec, zb_ - (0.004 if i == len(courses) - 1 else 0.0))]
        brick_objs.append(B.mesh("cap_bricks"))
        core_objs.append(rect_loft("cap_core", rings, top=True))

    # ---- dressed stone: shoulder weathering, and on the stone stack the string + moulded cornice + quoins
    r_s = rng.fork("stones")

    def dress(name, prof, hx, hy, max_len, sd):
        o = moulding(name, prof, hx, hy, r_s, max_len)
        densify(o, 0.085)
        displace_noise(o, strength=0.0045, scale=0.2, seed=seed + sd)
        displace_noise(o, strength=0.002, scale=0.11, seed=seed + sd + 1)
        bevel(o, width=0.005, segments=1)
        smooth(o, 50)
        dress_objs.append(o)
        back = [(a - 0.007, z + (0.005 if z < prof[0][1] + 0.002 else -0.004)) for (a, z) in prof]
        core_objs.append(rect_loft(name + "_bed", [(hx + a, hy + a, z) for (a, z) in back], top=False))

    if sh:
        wz = z_sh
        dress("weathering", [(sh - 0.012, wz - 0.072), (sh + 0.028, wz - 0.072), (sh + 0.034, wz - 0.064), (sh + 0.034, wz - 0.02),
                             (sh + 0.016, wz - 0.004), (0.02, wz + 0.052), (-0.008, wz + 0.066)], hw1, hd1, 0.5, 3)
    if stone:
        if has_string:
            dress("string", [(-0.008, zs0 - 0.01), (0.022, zs0 - 0.01), (0.032, zs0 + 0.004), (0.032, zs1 - 0.022), (-0.008, zs1 + 0.006)],
                  hw1, hd1, 0.45, 5)
        ch = z_mas - z_c0
        proj = n * STEP + 0.03
        prof = [(-0.008, z_c0), (0.014, z_c0), (0.014, z_c0 + 0.03)]
        for i in range(1, 6):                                            # cavetto bed mould
            t = i / 5
            prof.append((0.014 + (proj - 0.03) * (1 - cos(t * pi / 2)), z_c0 + 0.03 + ch * 0.3 * sin(t * pi / 2)))
        zc1 = z_c0 + 0.03 + ch * 0.3
        zc2 = zc1 + 0.006 + ch * 0.24
        prof += [(proj - 0.024, zc1 + 0.008), (proj, zc1 + 0.008), (proj, zc2), (proj - 0.012, zc2 + 0.012),     # fillet + corona
                 (proj * 0.45, z_mas - (z_mas - zc2) * 0.3), (0.016, z_mas), (-0.03, z_mas)]                  # weathered top
        dress("cornice", prof, hw1, hd1, 0.42, 7)
        # quoins, long and short alternately, standing proud of the rubble
        r_q = rng.fork("quoins")
        ql, qs = (0.3, 0.17) if min(W, D) > 0.62 else (0.2, 0.12)
        for (za, zb, hx, hy) in ([(0.0, z_sh - 0.07, hw0, hd0), (z_sh + 0.06, shaft_top, hw1, hd1)] if sh else [(0.0, shaft_top, hw0, hd0)]):
            rows = max(1, min(5 if sh else 7, int(round((zb - za) / 0.23))))
            hq = (zb - za) / rows
            for ci, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                for i in range(rows):
                    lx, ly = (ql, qs) if (i + ci) % 2 == 0 else (qs, ql)
                    lx, ly = r_q.vary(lx, 0.14), r_q.vary(ly, 0.14)
                    pr = 0.008 + r_q.u(0.0, 0.006)
                    q = stone_block("quoin", (lx, ly, hq - 0.01), loc=(sx * (hx + pr - lx / 2), sy * (hy + pr - ly / 2), za + (i + 0.5) * hq),
                                    rot=(0, 0, r_q.jitter(0.0, 0.8)), rng=r_q, chamfer=0.013, rough=0.005, detail=0.3)
                    dress_objs.append(q)

    # ---- lead flashing at the roof line. Every piece is a solid body sunk 30 mm below the roof plane, and without a
    #      roof patch nothing is wider than the stack, so no loose sheet ever floats beside it
    yf, yb = -hd0, hd0
    xw = hw0 + toff + th
    roofed = p.roof_stub != "none"
    xf = xw + (0.06 if roofed else 0.0)             # the apron only runs out over the roof covering when there is one
    y_ap, y_gt = yf, yb
    if p.flashing:
        up = 0.15
        lift = 0.007
        # front apron: upstand turned into a joint + flap dressed down over the tiles
        y_o = yf - toff - th
        zk = zr(y_o) + lift
        lead_objs.append(prism_x("apron_up", [(y_o, zk - th), (yf - toff, zk - th), (yf - toff, zr(yf) + up), (y_o, zr(yf) + up)], -xw, xw))
        y_ap = y_o - (0.16 if roofed else 0.1) * cs
        edge = th if roofed else 0.018               # free edge: thin over the tiles, a blunt welt when it stands alone
        flap = prism_x("apron_flap", [(y_ap, zr(y_ap) + lift - edge), (y_ap + 0.035, zr(y_ap + 0.035) - 0.03), (y_o + 0.002, zr(y_o) - 0.03),
                                      (y_o + 0.002, zk), (y_ap, zr(y_ap) + lift)], -xf, xf, tip=0.03, drop=0.005 if roofed else 0.0)
        displace_noise(flap, strength=0.0025, scale=0.09, seed=seed + 8)
        lead_objs.append(flap)
        # stepped side flashing: one piece per step, lapped over the piece below
        nst = max(2, int(round((yb - yf) * T / 0.15)))
        run = (yb - yf) / nst
        for sx in (-1, 1):
            for j in range(nst):
                ya = yf + j * run - (0.025 if j else toff + th)
                ye = yf + (j + 1) * run
                zt = zr(ye) + 0.07 + r_j.jitter(0.0, 0.004)
                if not stone:                                  # top edge turned into the bed joint above
                    zt = zoff + ceil((zr(ye) + 0.05 - zoff) / c) * c - 0.003
                xo = hw0 + toff + (0.0016 if j % 2 else 0.0)
                lead_objs.append(prism_x("step_flash", [(ya, zr(ya) + 0.004), (ye, zr(ye) + 0.004), (ye, zt), (ya, zt)], sx * xo, sx * (xo + th)))
        # back gutter: upstand + lead-clad sole running up the slope, stopped flush with the side flashing. Without a
        # roof patch only the 50 mm cant at the foot of the upstand is built (the host roof laps over the rest)
        y_o = yb + toff + th
        lead_objs.append(prism_x("back_up", [(yb + toff, zr(yb) + lift - th), (y_o, zr(y_o) + lift - th), (y_o, zr(yb) + up), (yb + toff, zr(yb) + up)], -xw, xw))
        y_gt = y_o + (0.15 if roofed else 0.05) * cs
        sole = prism_x("back_sole", [(y_o - 0.002, zr(y_o) - 0.03), (y_gt, zr(y_gt) - 0.03), (y_gt + 0.004, zr(y_gt) + lift - 0.004),
                                     (y_gt - 0.004, zr(y_gt) + lift), (y_o - 0.002, zr(y_o) + lift)], -xw, xw)
        lead_objs.append(sole)

    # ---- optional roof patch (stand-alone use): one continuous grid, the courses run through; only the stack pierces it
    if p.roof_stub != "none":
        mg = 0.62
        x1, y_hi = hw0 + mg, hd0 + mg
        y_lo = max(-(hd0 + mg), (0.02 - s) / T)
        tkr = 0.07

        def lines(lo, hi, forced):
            ks = sorted({round(v, 4) for v in forced if lo + 0.04 < v < hi - 0.04} | {lo, hi})
            out = []
            for a_, b_ in zip(ks[:-1], ks[1:]):
                m = max(1, int(ceil((b_ - a_) / 0.19)))
                out += [a_ + (b_ - a_) * i / m for i in range(m)]
            return out + [hi]

        hx_h, hy_h = hw0 - 0.01, hd0 - 0.01
        xs = lines(-x1, x1, [-xf - 0.035, -hx_h, hx_h, xf + 0.035])
        ys = lines(y_lo, y_hi, [y_ap - 0.035, -hy_h, hy_h, y_gt + 0.04])
        nx, ny = len(xs) - 1, len(ys) - 1
        verts = [(x, y, zr(y)) for x in xs for y in ys]
        faces = []
        for i in range(nx):
            for j in range(ny):
                if not (abs(xs[i] + xs[i + 1]) / 2 < hx_h and abs(ys[j] + ys[j + 1]) / 2 < hy_h):
                    faces.append((i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1))
        per = [(i, 0) for i in range(nx)] + [(nx, j) for j in range(ny)] + [(i, ny) for i in range(nx, 0, -1)] + [(0, j) for j in range(ny, 0, -1)]
        b0 = len(verts)
        verts += [(xs[i], ys[j], max(0.0, zr(ys[j]) - tkr)) for (i, j) in per]
        M = len(per)
        for k, (i, j) in enumerate(per):
            i2, j2 = per[(k + 1) % M]
            faces.append((b0 + k, b0 + (k + 1) % M, i2 * (ny + 1) + j2, i * (ny + 1) + j))
        u0 = len(verts)                                        # underside, so the patch is a slab from every angle
        verts += [(sx * x1, y, max(0.0, zr(y) - tkr)) for y in ys for sx in (-1, 1)]
        faces += [(u0 + 2 * j, u0 + 2 * j + 2, u0 + 2 * j + 3, u0 + 2 * j + 1) for j in range(ny)]
        roof = mesh_object("roof_stub", verts, faces)
        uv_planar(roof, u_axis=(1, 0, 0), v_axis=(0, cs, T * cs))
        assign(roof, mat("roof_tile_clay" if p.roof_stub == "tiles" else "roof_slate"))
        paint(roof, special=mask_mul(noise_mask(0.5, seed=seed + 12, lo=0.35, hi=0.9), 0.12 * mossy),
              grime=lambda co, nn: np.maximum(0.8 * (nn[:, 2] < -0.3),
                                              0.45 * soot * _ss(hd0 + 0.1, -hd0 - 0.5, co[:, 1]) * _ss(hw0 + 0.4, hw0, np.abs(co[:, 0]))))

    # ---- wrought-iron tie bands: 46 x 8 mm strap standing proud, drawn tight by a bolted lug pair on the front
    so = (0.02 if stone else 0.0) + 0.011
    runs = []                                       # rust runs below the fixings: (face, along, z_top, length, width)
    for bi, (zb_, xt) in enumerate(bands):
        ring = rounded_rect(2 * (hw1 + so), 2 * (hd1 + so), 0.022, 3)
        iron_objs.append(sweep(f"tie_band{bi}", [(x, y, zb_) for (x, y) in ring], rounded_rect(0.010, 0.048, 0.0025, 1), closed=True, smooth_path=False))
        yl = -(hd1 + so + 0.005)                    # outer face of the strap on the front
        for sx in (-1, 1):
            iron_objs.append(box(f"tie_lug{bi}", (0.009, 0.064, 0.052), loc=(xt + sx * 0.022, yl - 0.027, zb_), bevel=0.002, segments=1))
        yb_ = yl - 0.036
        iron_objs.append(cylinder(f"tie_bolt{bi}", 0.0065, 0.1, loc=(xt, yb_, zb_), rot=(0, 90, 0), verts=8))
        iron_objs.append(bolt(f"tie_nut{bi}", 0.0135, 0.013, loc=(xt + 0.0265, yb_, zb_), rot=(0, 90, 0)))
        iron_objs.append(bolt(f"tie_head{bi}", 0.013, 0.009, loc=(xt - 0.0265, yb_, zb_), rot=(0, -90, 0), kind="square"))
        rv = bolt("tie_rivet", 0.009, 0.006, kind="dome")
        ms = []
        for fx_ in (-0.62, 0.62):
            ms.append(trs((fx_ * hw1, yl, zb_), (90, 0, 0)))
            ms.append(trs((fx_ * hw1, -yl, zb_), (-90, 0, 0)))
            runs.append((0, fx_ * hw1, zb_, r_j.u(0.2, 0.36), r_j.u(0.014, 0.024)))
        for sx in (-1, 1):
            ms.append(trs((sx * (hw1 + so + 0.005), 0.0, zb_), (0, sx * 90, 0)))
            runs.append((1 if sx > 0 else 3, r_j.jitter(0.0, 0.02), zb_, r_j.u(0.2, 0.36), r_j.u(0.014, 0.024)))
        iron_objs.append(instances_to_mesh(f"tie_rivets{bi}", rv, ms, keep_base=False))
        runs.append((0, xt + r_j.jitter(0.0, 0.01), zb_, r_j.u(0.36, 0.5), r_j.u(0.03, 0.042) * min(1.0, hw1 / 0.36)))       # the long run below the lug
        runs.append((0, hw1 - 0.012, zb_, r_j.u(0.25, 0.4), 0.018))
        runs.append((1, hd1 - 0.014, zb_, r_j.u(0.2, 0.34), 0.018))
        runs.append((1, -hd1 + 0.012, zb_, r_j.u(0.2, 0.4), 0.016))
    if runs and not stone:
        verts, faces = [], []
        for (side, t, zt, ln, w) in runs:
            (tx, ty), (nx_, ny_) = SIDES[side]
            hn = (hd1 if side % 2 == 0 else hw1) + 0.002
            ln = min(ln, zt - (z_sh + 0.08 if sh else z_rb + 0.2))
            if ln < 0.08:
                continue
            base = len(verts)
            drift = r_j.jitter(0.0, 0.012)
            for i in range(5):
                f = i / 4
                ww = max(0.002, w * (1.0 - f) ** 0.8 * (1.0 if i != 1 else 0.8))
                tc = t + drift * f * f
                z = zt - 0.012 - ln * f
                for a in (tc - ww / 2, tc + ww / 2):
                    x, y = a * tx + hn * nx_, a * ty + hn * ny_
                    dx, dy = warp(x, y, z)
                    verts.append((x + dx, y + dy, z))
            for i in range(4):
                faces.append((base + 2 * i + 2, base + 2 * i + 3, base + 2 * i + 1, base + 2 * i))
        if faces:
            o = mesh_object("rust_runs", verts, faces)
            assign(o, mat("mortar", "#5e331d"))
            paint(o, grime=lambda co, nn: 0.35 + 0.3 * noise_mask(0.04, seed=seed + 4)(co, nn))
            rust_objs.append(o)

    # ---- pots: layout on the cap, each a different pattern, bedded in mortar flaunching
    npots = int(p.pots)
    best = None
    for rows in (1, 2):
        cols = int(ceil(npots / rows))
        sx_, sy_ = (2 * hw1 - 0.05) / cols, (2 * hd1 - 0.05) / rows
        r = min(0.15, 0.44 * sx_, 0.44 * sy_)
        if best is None or r > best[0] + 0.004:
            best = (r, rows, cols, sx_, sy_)
    r_pot, rows, cols, sx_, sy_ = best
    r_p = rng.fork("pots")
    styles = ["roll", "crown", "louvre", "cannon", "octagon", "castle", "cowl"]
    order = sorted(styles, key=lambda _s: r_p.u())
    tints = ["#ffffff", "#d6a183", "#efd9b4"]
    seg = 24 if npots <= 2 else (20 if npots == 3 else 16)
    pots, pot_info, pot_extras = [], [], []
    for i in range(npots):
        row, col = divmod(i, cols)
        cnt = min(cols, npots - row * cols)
        px = (col - (cnt - 1) / 2) * sx_ + r_p.jitter(0.0, 0.006)
        py = (row - (rows - 1) / 2) * sy_ + r_p.jitter(0.0, 0.006)
        style = order[i % len(order)] if p.pot_style == "mixed" else p.pot_style
        fnc, hf = POTS[style]
        rb = max(0.07, r_pot * r_p.u(0.88, 1.0))
        hp = min(p.pot_height * hf * r_p.vary(1.0, 0.07), rb * 8.5)
        hp = max(hp, 0.26)
        w = max(0.014, rb * 0.14)
        objs, core, extras = fnc(rb, hp, seg, w)
        tint = tints[(i + seed) % len(tints)] if p.pot_style == "mixed" else tints[(i * 2 + seed) % len(tints)]
        for o in objs:
            assign(o, mat("terracotta", tint))
        pot = join(objs, f"pot{i}")
        zp = H - 0.035
        prot = (r_p.jitter(0.0, 1.1), r_p.jitter(0.0, 1.1), r_p.u(0, 360))      # pots stay within 2 deg of the cap normal
        place(pot, loc=(px, py, zp), rot=prot)
        pots.append(pot)
        for (eo, em) in extras:
            assign(eo, mat(em))
            place(eo, loc=(px, py, zp), rot=prot)
            paint(eo, special=mask_mul(noise_mask(0.05, seed=seed + 21, lo=0.3, hi=0.7), 0.7), grime=0.35 * soot)
            pot_extras.append(eo)
        pot_info.append((px, py, zp, rb, hp, core))
        mortar_objs.append(lathe("pot_bed", [(rb + 0.045, H - 0.05), (rb + 0.036, H - 0.012), (rb + 0.013, H + 0.02), (rb - 0.004, H + 0.028)],
                                 segments=16, loc=(px, py, 0)))

    # flaunching: weathered mortar haunch from the cap edge up to the collar round the pots
    hx_t, hy_t = hw1 + top_off, hd1 + top_off
    ins = min(0.11, 0.45 * min(hx_t, hy_t))
    fla = flaunching("flaunching", [(hx_t - 0.012, hy_t - 0.012, z_mas - 0.02), (hx_t - 0.02, hy_t - 0.02, z_mas + 0.014),
                                    (hx_t - ins, hy_t - ins, H - 0.006)], hx_t - ins * 1.25, hy_t - ins * 1.25, H,
                     [(q[0], q[1], q[3] + 0.032) for q in pot_info])
    displace_noise(fla, strength=0.003, scale=0.1, seed=seed + 17, along_normal=False)
    smooth(fla, 50)
    mortar_objs.append(fla)

    # ---- materials
    for o in body_objs + brick_objs:
        assign(o, m_body)
    for o in dress_objs:
        assign(o, m_dress)
    for o in lead_objs:
        assign(o, m_lead)
    for o in iron_objs:
        assign(o, m_iron)
    for o in mortar_objs + core_objs:
        assign(o, m_mortar)
        if o.name.startswith("pot_bed"):
            displace_noise(o, strength=0.003, scale=0.06, seed=seed + 19)

    # ---- weathering (painted on the plumb stack, the lean is applied afterwards)
    streak = noise_mask(0.25, seed=seed + 1, lo=0.3, hi=0.75)
    blot = noise_mask(0.45, seed=seed + 2, lo=0.25, hi=0.8)
    aniso = np.array([4.0, 4.0, 0.3])
    fine = np.array([9.0, 9.0, 0.22])
    drip = np.array([3.2, 3.2, 0.25])

    def stack_grime(co, nn):
        z = co[:, 2]
        top = _ss(z_c0 - 1.1, z_mas, z)
        st = streak(co * aniso, nn)
        under = _ss(z_c0 - 0.3, z_c0, z) * (z < z_c0 + 0.02)
        g = soot * (top * (0.2 + 0.65 * st) + 0.3 * under)
        st2 = streak(co * fine, nn)
        for q in pot_info:                           # soot washed down the cap below each flue
            dt = np.where(np.abs(nn[:, 1]) > np.abs(nn[:, 0]), np.abs(co[:, 0] - q[0]), np.abs(co[:, 1] - q[1]))
            g = np.maximum(g, soot * 0.9 * _ss(q[3] * 2.0, q[3] * 0.4, dt) * _ss(z_mas - 0.85, z_mas + 0.02, z) * (0.3 + 0.7 * st2))
        for (zb_, _x) in bands:                      # damp rust wash below the tie bands
            g = np.maximum(g, 0.5 * st * _ss(zb_ - 0.45, zb_ - 0.02, z) * (z < zb_))
        if sh:                                       # damp streaks below the drip of the stone weathering
            g = np.maximum(g, 0.75 * (0.25 + 0.75 * streak(co * drip, nn)) * _ss(z_sh - 0.55, z_sh - 0.07, z) * (z < z_sh - 0.05))
        if has_string:
            g = np.maximum(g, 0.35 * (0.3 + 0.7 * st2) * _ss(zs0 - 0.3, zs0, z) * (z < zs0 + 0.01))
        return np.clip(g, 0.0, 1.0)

    def damp(co, nn):                                # damp north face low down
        north = np.clip(nn[:, 1], 0.0, 1.0)
        low = 1.0 - _ss(z_rb, z_rb + 1.0, co[:, 2])
        return 0.9 * north * low * blot(co, nn)

    def stack_moss(co, nn):
        upf = np.clip(nn[:, 2], 0.0, 1.0)
        b = blot(co, nn)
        return np.clip(mossy * (0.75 * upf * b * b + damp(co, nn)), 0.0, 0.6)           # cushions on ledges, damp north face

    def stone_grime(co, nn):
        nz = nn[:, 2]
        g = 0.12 + 0.24 * blot(co * 2.3 + 5.0, nn)                                      # patchy, stone by stone
        g = g + 0.45 * np.clip(-nz, 0.0, 1.0)                                          # dirty soffits and throats
        g = np.maximum(g, (0.15 + 0.5 * streak(co * drip, nn)) * (np.abs(nz) < 0.6))    # run-off streaks down the faces
        g = np.maximum(g, np.clip(nz, 0.0, 1.0) * (0.26 + 0.18 * soot + 0.3 * streak(co * 3.0, nn)))   # dirt + soot lying on the tops
        return np.clip(np.maximum(g, 0.6 * stack_grime(co, nn)), 0.0, 0.72)

    def stone_moss(co, nn):                          # cushions on the weathered tops, green film on the north side
        upf = np.clip(nn[:, 2], 0.0, 1.0)
        b = blot(co * 1.7 + 3.0, nn)
        return np.clip(mossy * (1.1 * upf * b + 0.5 * np.clip(nn[:, 1], 0.0, 1.0) * b + damp(co, nn)), 0.0, 0.65)

    def fl_grime(co, nn):                            # soot only as a graded halo round each pot base
        g = 0.1 + 0.15 * streak(co * 3.0, nn)
        for q in pot_info:
            rad = np.sqrt((co[:, 0] - q[0]) ** 2 + (co[:, 1] - q[1]) ** 2)
            g = np.maximum(g, soot * 0.6 * _ss(q[3] + 0.17, q[3] + 0.02, rad))
        return np.clip(g, 0.0, 0.6)

    def fl_moss(co, nn):                             # soft cushions in the fillets, away from the hot flues
        far = np.ones(len(co))
        for q in pot_info:
            rad = np.sqrt((co[:, 0] - q[0]) ** 2 + (co[:, 1] - q[1]) ** 2)
            far = far * _ss(q[3] + 0.03, q[3] + 0.15, rad)
        return np.clip(mossy * 0.55 * blot(co * 1.6, nn) * (0.35 + 0.65 * far), 0.0, 0.5)

    for o in body_objs:
        paint(o, grime=stack_grime, special=stack_moss)
    for o in brick_objs:
        paint(o, grime=stack_grime, special=lambda co, nn: np.clip(mossy * damp(co, nn), 0.0, 0.6))
    for o in core_objs:
        paint(o, grime=lambda co, nn: np.clip(0.08 + stack_grime(co, nn), 0.0, 0.7))
    for o in dress_objs:
        paint(o, grime=stone_grime, special=stone_moss)
    for o in mortar_objs:
        paint(o, grime=fl_grime, special=fl_moss)
    for o in lead_objs:
        paint(o, grime=mask_mul(noise_mask(0.12, seed=seed + 6), 0.45))
    for o in iron_objs:
        paint(o, special=lambda co, nn: np.clip(0.06 + 0.6 * noise_mask(0.07, seed=seed + 7, lo=0.4, hi=0.9)(co, nn), 0.0, 0.7))
    for pot, (px, py, zp, rb, hp, core) in zip(pots, pot_info):
        def pot_grime(co, nn, px=px, py=py, zp=zp, rb=rb, hp=hp, core=core):
            dx, dy = co[:, 0] - px, co[:, 1] - py
            rad = np.sqrt(dx * dx + dy * dy) + 1e-6
            inward = (-(nn[:, 0] * dx + nn[:, 1] * dy) / rad > 0.35) & (co[:, 2] > zp + 0.06)
            top = _ss(zp + hp * 0.35, zp + hp, co[:, 2])
            g = soot * 1.2 * top * (0.35 + 0.65 * streak(co * aniso, nn))
            g = np.maximum(g, soot * 1.1 * _ss(zp + hp - 0.13, zp + hp - 0.02, co[:, 2]))        # blackened rim
            g = np.maximum(g, inward * 1.0)
            if core is not None:
                g = np.maximum(g, ((rad < core[0]) & (co[:, 2] > zp + core[1]) & (co[:, 2] < zp + core[2] + 0.02)) * 1.0)
            return np.clip(g, 0.0, 1.0)

        paint(pot, grime=pot_grime,
              special=lambda co, nn, zp=zp, hp=hp: np.clip(mossy * 0.8 * blot(co * 2.0, nn) * (1.0 - _ss(zp, zp + hp * 0.5, co[:, 2])), 0, 0.6))

    # ---- lean: the masonry bows over between the roof line and 60 % of the shaft, everything above (cap, flaunching,
    #      pots) is carried rigidly at the lean angle, so the pots stay square to the cap
    ang = radians(float(p.lean))
    z_k = s + 0.6 * (shaft_top - s)
    a = ang / (z_k - s)

    def bend(co):
        if abs(a) < 1e-5 or co.z <= s:               # plumb inside the roof
            return co
        t = a * (min(co.z, z_k) - s)
        e = max(0.0, co.z - z_k)
        return Vector(((1.0 - cos(t)) / a + co.x * cos(t) + e * sin(t), co.y, s + sin(t) / a - co.x * sin(t) + e * cos(t)))

    def shape(co):
        dx, dy = warp(co.x, co.y, co.z)
        return bend(Vector((co.x + dx, co.y + dy, co.z)))

    pre = set(o.name for o in body_objs + rust_objs)             # already warped when built
    for o in body_objs + brick_objs + core_objs + dress_objs + lead_objs + iron_objs + rust_objs + mortar_objs + pots + pot_extras:
        if abs(a) < 1e-5 and (stone or o.name in pre):
            continue
        realize(o)
        uv_keep(o)
        apply_transform(o)
        transform_verts(o, bend if o.name in pre else shape)

    if p.smoke:
        px, py, zp, rb, hp, core = pot_info[0]
        v = bend(Vector((px, py, zp + hp + 0.03)))
        fx("smoke", "smoke", tuple(v), size=rb * 2.4)
    socket("roof_line", "attach", (0.0, 0.0, s), role="roof", pitch=degrees(np.arctan(T)))
