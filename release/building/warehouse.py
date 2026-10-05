"""Brick canal warehouse. Solid brick box on a dressed-stone plinth: a tall ground storey with barred
windows and an arched cart door, upper storeys divided into bays by brick pilasters standing on a
jointed stone string course, cast-iron small-pane windows under gauged-brick segmental arches, a column
of arched loading doors and - cantilevered over them on raking struts - a weatherboarded timber hoist
housing (lucam) whose gabled roof dies into the main slate roof. A limewashed sign band with big flaked
letters runs round the building at the lucam floor. Corbelled dentil cornice, cast-iron gutters and
downpipes, coped parapet gables with kneelers, a folded jib crane and a cathead beam on the canal wall,
six wall lanterns (cart door, pilaster, under the lucam, both gables, canal wall).
"""
from tk import *
from math import sin, cos, tan, atan, atan2, asin, sqrt, radians, degrees, pi, ceil, floor
import numpy as np
import bmesh
import bpy
from mathutils import Matrix, Vector

META = {"name": "Brick canal warehouse", "class": "building", "mount": "ground", "focus": 0.36}
PARAMS = {
    "width": {"default": 10.0, "min": 8.0, "max": 16.0, "desc": "front width along X (m)"},
    "depth": {"default": 8.0, "min": 7.0, "max": 11.0, "desc": "depth front to back, y = 0 .. depth (m)"},
    "storeys": {"default": 3, "min": 3, "max": 4, "desc": "number of storeys (the top one carries the lucam)"},
    "storey_height": {"default": 3.1, "min": 2.9, "max": 3.4, "desc": "upper storey height (m); the ground storey is 4 m"},
    "brick": {"default": "brick_red", "choices": ["brick_red", "brick_ochre"], "desc": "wall brick"},
    "sign_text": {"default": "MERIDIAN WHARF", "desc": "name painted in big letters on the sign band (front and canal wall)"},
    "sign_sub": {"default": "BONDED STORES", "desc": "trade description painted on the sign band of both gable walls"},
    "sign_height": {"default": 0.62, "min": 0.45, "max": 0.7, "desc": "cap height of the sign letters (m); the band is 0.95 m"},
    "sign_font": {"default": "sans_narrow", "choices": ["sans_narrow", "sans", "serif", "roman"], "desc": "letter style of the sign"},
    "sign_style": {"default": "light", "choices": ["light", "dark"],
                   "desc": "light = limewashed band with dark letters, dark = tarred band with pale letters"},
    "lit_ratio": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "fraction of windows with a lit room behind"},
    "hoist_bay": {"default": "centre", "choices": ["centre", "left", "right"],
                  "desc": "which bay carries the loading doors and the lucam"},
    "lucam_storeys": {"default": 1, "min": 1, "max": 2, "desc": "height of the lucam in storeys (2 needs a four-storey building)"},
    "canal_doors": {"default": True, "desc": "loading doors, a jib crane and a cathead beam on the back (canal) wall"},
    "lucam_reach": {"default": 1.1, "min": 0.8, "max": 1.24, "desc": "projection of the timber hoist housing (m)"},
    "pitch": {"default": 43.0, "min": 38.0, "max": 50.0, "desc": "roof pitch (deg)"},
    "paint": {"default": "#7d9a86", "desc": "paint colour of the timber: loading doors, lucam boarding, fascia"},
    "iron": {"default": "#1d2723", "desc": "paint colour of the cast iron: window frames, gutters, pipes, lanterns, sign letters"},
    "stone": {"default": "#c2b59c", "desc": "tint of the dressed stone: strings, sills, plinth, copings and of the limewashed sign band"},
    "chimney": {"default": True, "desc": "counting-house chimney stack on the back slope"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0,
            "desc": "condition: soot and drip stains, flaked sign, missing and slipped slates, sagging ridge, patched and "
                    "boarded-up windows, slipped weatherboards"},
    "lantern_cd": {"default": 28.0, "min": 2.0, "max": 45.0, "desc": "intensity of the wall lanterns (cd); the one under the lucam is 1.25x"},
    "mullion": {"default": 0.042, "min": 0.0, "max": 0.07,
                "desc": "width (m) of the heavier primary glazing bars: a regular grid that still reads from far away and in the "
                        "low LODs; below 0.02 = fine bars only"},
    "lod_pins": {"default": True, "desc": "lock the lit window panes, the primary glazing bars of lit windows and the lucam lining "
                                          "so the town's LOD decimation leaves them as clean shapes"},
}
VARIANTS = {
    "ochre_four": {"width": 12.5, "depth": 9.0, "storeys": 4, "brick": "brick_ochre", "hoist_bay": "right",
                   "paint": "#a34d37", "iron": "#231c1a", "sign_text": "TEA & SPICE", "sign_sub": "IMPORTERS", "lucam_storeys": 2,
                   "lit_ratio": 0.4, "pitch": 46.0, "sign_font": "serif", "seed": 7},
    "narrow_tall": {"width": 8.0, "depth": 7.0, "storeys": 4, "storey_height": 2.95, "paint": "#5f86a8",
                    "sign_text": "No 3", "sign_sub": "GRANARY", "lucam_reach": 1.24, "lit_ratio": 0.2, "sign_height": 0.7, "seed": 13},
    "long_quay": {"width": 16.0, "depth": 10.5, "storeys": 3, "hoist_bay": "left", "paint": "#b9b29c",
                  "iron": "#2a2f3a", "sign_text": "CANAL COMPANY", "sign_sub": "FLYBOATS DAILY", "lit_ratio": 0.5,
                  "pitch": 40.0, "storey_height": 3.3, "age": 0.25, "seed": 21},
    "old_tarred": {"width": 11.5, "depth": 7.0, "storeys": 3, "storey_height": 2.9, "brick": "brick_ochre", "paint": "#4b433a",
                   "stone": "#a89f8c", "sign_text": "SAIL LOFT", "sign_sub": "& CHANDLERY", "age": 1.0, "canal_doors": False,
                   "hoist_bay": "left", "pitch": 50.0, "lit_ratio": 0.15, "chimney": False, "lucam_reach": 0.9, "lantern_cd": 24.0,
                   "sign_style": "dark", "seed": 34},
}


# ------------------------------------------------------------------ small local geometry helpers
def _arch_y(x, w, h, rise):
    """Height of a segmental-arched opening (clear width w, crown height h) at offset x from its centre."""
    if rise < 1e-4:
        return h
    R = (w * w / 4 + rise * rise) / (2 * rise)
    return h - R + sqrt(max(R * R - x * x, 0.0))


def _arch(cx, z0, w, h, rise, n=10):
    """Opening outline, CCW seen from the front, from bottom-right to bottom-left."""
    pts = [(cx + w / 2, z0)]
    for i in range(n + 1):
        x = w / 2 - w * i / n
        pts.append((cx + x, z0 + _arch_y(x, w, h, rise)))
    pts.append((cx - w / 2, z0))
    return pts


def _prism(name, prof, x0, x1, m0=0.0, m1=0.0, cap0=True, cap1=True):
    """Extrude a (y, z) profile along X; m0 / m1 = 1 mitres that end at 45 deg (outwards = -y)."""
    n = len(prof)
    a = sum(prof[i][0] * prof[(i + 1) % n][1] - prof[(i + 1) % n][0] * prof[i][1] for i in range(n))
    if a < 0:
        prof = prof[::-1]
    verts = [(x0 + y * m0, y, z) for (y, z) in prof] + [(x1 - y * m1, y, z) for (y, z) in prof]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    if cap1:
        faces.append(tuple(n + i for i in range(n)))
    if cap0:
        faces.append(tuple(reversed(range(n))))
    return mesh_object(name, verts, faces)


def _lines(must, step, fine=()):
    """Sorted grid lines through every mandatory value; the gaps are filled evenly with a spacing <= step
    (<= 0.26 m inside the `fine` intervals - the openings, whose arched heads need a few more columns)."""
    vals = sorted(must)
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] > 2e-3:
            out.append(v)
    res = []
    for a, b in zip(out[:-1], out[1:]):
        m = (a + b) / 2
        st = 0.26 if any(f0 < m < f1 for f0, f1 in fine) else step
        n = max(1, int(ceil((b - a) / st - 1e-6)))
        res += [a + (b - a) * i / n for i in range(n)]
    res.append(out[-1])
    return res


def _cut(iv, excl):
    """Interval (a, b) minus a list of excluded intervals -> list of intervals."""
    out = [iv]
    for e0, e1 in excl:
        nxt = []
        for a, b in out:
            if e1 <= a or e0 >= b:
                nxt.append((a, b))
                continue
            if e0 - a > 0.02:
                nxt.append((a, e0))
            if b - e1 > 0.02:
                nxt.append((e1, b))
        out = nxt
    return out


class _MB:
    """Mesh builder for lots of hexahedra (door boards, voussoirs, glazing bars, roof slabs)."""

    def __init__(self):
        self.v, self.f, self.open = [], [], False

    def hexa(self, pts, back=True, sides=(0, 1, 2, 3)):
        """pts = front quad (4) + back quad (4). sides: which of the four faces between them to build
        (0 = along the front edge 0-1, 1 = edge 1-2, 2 = edge 2-3, 3 = edge 3-0)."""
        b = len(self.v)
        self.v += [tuple(q) for q in pts]
        self.f.append((b, b + 1, b + 2, b + 3))
        sf = ((b, b + 4, b + 5, b + 1), (b + 1, b + 5, b + 6, b + 2), (b + 2, b + 6, b + 7, b + 3), (b + 3, b + 7, b + 4, b))
        for i in sides:
            self.f.append(sf[i])
        if back and len(sides) == 4:
            self.f.append((b + 7, b + 6, b + 5, b + 4))
        else:
            self.open = True

    def quad(self, pts):
        b = len(self.v)
        self.v += [tuple(q) for q in pts]
        self.f.append(tuple(range(b, b + len(pts))))
        self.open = True

    def obj(self, name):
        o = mesh_object(name, self.v, self.f)
        if not self.open:
            recalc_normals(o)
        return o


class _PB:
    """Faces that have to survive the town's LOD decimation untouched (glazing-bar mask, lit panes, lucam lining):
    every face is a fan whose diagonals carry a tiny hidden fin triangle; the non-manifold edge locks its vertices."""

    def __init__(self, pins=True):
        self.v, self.f, self.pins = [], [], pins

    def face(self, pts, n):
        P = [Vector(q) for q in pts]
        k = len(P)
        n = Vector(n).normalized()
        nn = Vector((0.0, 0.0, 0.0))
        for i in range(k):
            a, b = P[i], P[(i + 1) % k]
            nn += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y)))
        if nn.dot(n) < 0:
            P.reverse()
        b0 = len(self.v)
        self.v += [tuple(q) for q in P]
        if not self.pins:
            self.f.append(tuple(range(b0, b0 + k)))
            return
        off = min(0.012, 0.2 * min((P[i] - P[(i + 1) % k]).length for i in range(k)))
        for j in range(1, k - 1):
            self.f.append((b0, b0 + j, b0 + j + 1))
        if k == 3:
            e = (P[1] - P[0]).normalized()
            m = (P[0] + P[1]) / 2
            for q in (m - n * off, m + n.cross(e) * off):
                self.v.append(tuple(q))
                self.f.append((b0, b0 + 1, len(self.v) - 1))
        else:
            for j in range(2, k - 1):
                self.v.append(tuple((P[0] + P[j]) / 2 - n * off))
                self.f.append((b0, b0 + j, len(self.v) - 1))

    def obj(self, name):
        return mesh_object(name, self.v, self.f)


def _drop_faces(obj, pred):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    dead = [f for f in bm.faces if pred(f.calc_center_median(), f.normal)]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context='FACES')
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def _ring(mb, cx, z0, w, h, rise, d, proud, rng, step=0.085):
    """Gauged-brick arch ring: individual voussoirs standing `proud` of the wall face (wall-local coords);
    only the face and the intrados of every voussoir can be seen."""
    R = (w * w / 4 + rise * rise) / (2 * rise)
    phi = asin(min(1.0, w / (2 * R)))
    n = max(5, int(round(2 * phi * R / step))) | 1
    cz = z0 + h - R
    g = 0.004 / R
    for i in range(n):
        a0 = -phi + 2 * phi * i / n + g
        a1 = -phi + 2 * phi * (i + 1) / n - g
        r0, r1 = R - 0.004, R + d + rng.u(-0.004, 0.004)
        yf = -proud - rng.u(0.0, 0.005)
        q = [(cx + r0 * sin(a0), cz + r0 * cos(a0)), (cx + r0 * sin(a1), cz + r0 * cos(a1)),
             (cx + r1 * sin(a1), cz + r1 * cos(a1)), (cx + r1 * sin(a0), cz + r1 * cos(a0))]
        mb.hexa([(x, yf, z) for x, z in q] + [(x, 0.03, z) for x, z in q], back=False, sides=(0,))


def _beam(name, p0, p1, w, th, **kw):
    """Timber (or lead strip) between two points; w = horizontal-ish width, th = depth."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    o = plank(name, d.length, w, th, axis='X', **kw)
    e = d.to_track_quat('X', 'Z').to_euler()
    place(o, loc=(p0 + p1) / 2, rot=tuple(degrees(a) for a in e))
    return o


def _rod(name, p0, p1, r, sides=6):
    return sweep(name, [tuple(p0), tuple(p1)], r, sides=sides, smooth_path=False)


def _basis(e, s, n, origin):
    m = Matrix(((e[0], s[0], n[0], origin[0]), (e[1], s[1], n[1], origin[1]), (e[2], s[2], n[2], origin[2]), (0, 0, 0, 1)))
    return m


def _block(name, size, bev, loc=(0, 0, 0), rot=None, drop_back=True, drop_bottom=False):
    """Cheap dressed block: chamfered box, optionally without the faces that sit in / on the masonry."""
    o = box(name, size, bevel=bev, segments=1)
    if drop_back or drop_bottom:
        _drop_faces(o, lambda c, n: (drop_back and n.y > 0.9) or (drop_bottom and n.z < -0.9))
    place(o, loc=loc, rot=rot)
    return o


def _loft(name, secs, closed=True, cap0=False, cap1=False):
    """Quads between consecutive sections (lists of points, the sections follow each other along +X and are
    drawn in the (y, z) plane; the winding is fixed here so the faces look outwards)."""
    q0 = secs[0]
    n = len(q0)
    a = sum(q0[i][1] * q0[(i + 1) % n][2] - q0[(i + 1) % n][1] * q0[i][2] for i in range(n))
    if a < 0:
        secs = [s[::-1] for s in secs]
    verts = [tuple(q) for s in secs for q in s]
    faces = []
    for k in range(len(secs) - 1):
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            faces.append((k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i))
    if cap0:
        faces.append(tuple(reversed(range(n))))
    if cap1:
        faces.append(tuple((len(secs) - 1) * n + i for i in range(n)))
    return mesh_object(name, verts, faces)


def _tube(mb, x0, x1, y0, y1, zb, zt, rows):
    """Four gridded sides of an upright box (outward faces) into a mesh builder."""
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for q in range(rows):
        za, zc = zb + (zt - zb) * q / rows, zb + (zt - zb) * (q + 1) / rows
        for i in range(4):
            a, b = ring[i], ring[(i + 1) % 4]
            mb.quad([(a[0], a[1], za), (b[0], b[1], za), (b[0], b[1], zc), (a[0], a[1], zc)])


def _chain(name, p0, p1, link=0.13, wire=0.011):
    """Light chain: eight-sided links of triangular wire, alternately turned 90 deg."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    n = max(2, int(round(d.length / (link * 0.72))))
    hl, hw = link / 2 - wire, link * 0.31 - wire
    loop = [(hl, -hw * 0.5, 0), (hl, hw * 0.5, 0), (hl * 0.5, hw, 0), (-hl * 0.5, hw, 0), (-hl, hw * 0.5, 0), (-hl, -hw * 0.5, 0),
            (-hl * 0.5, -hw, 0), (hl * 0.5, -hw, 0)]
    base = sweep(name + "_link", loop, wire, closed=True, sides=3, smooth_path=False)
    q = d.to_track_quat('X', 'Z').to_matrix().to_4x4()
    mats = [Matrix.Translation(p0 + d * ((i + 0.5) / n)) @ q @ Matrix.Rotation(pi / 2 * (i % 2), 4, 'X') for i in range(n)]
    return instances_to_mesh(name, base, mats, uv_jitter=False, keep_base=False)


def _n1(scale, seed, lo=0.3, hi=0.7):
    """1D value noise along a wall (vertical streaks)."""
    f = noise_mask(scale, seed=seed, lo=lo, hi=hi)

    def fn(t):
        co = np.zeros((len(t), 3), dtype=np.float64)
        co[:, 0] = t
        co[:, 1] = 3.7
        co[:, 2] = 1.3
        return f(co, None)
    return fn


# =====================================================================================================
def build(p, rng):
    W, D, NS = float(p.width), float(p.depth), int(p.storeys)
    hs, age = float(p.storey_height), float(p.age)
    lod = clamp((2 * (W + D) * NS - 165.0) / 50.0, 0.0, 1.0)   # the biggest sheds: coarser paint tessellation, larger slates (budget)
    pitch = radians(p.pitch)
    tanp, cosp, sinp = tan(pitch), cos(pitch), sin(pitch)
    F1 = 4.0
    F = [0.0] + [F1 + hs * k for k in range(NS - 1)]        # floor level of every storey
    Ftop = F[-1]
    LS = 2 if (int(p.lucam_storeys) >= 2 and NS >= 4) else 1   # storeys covered by the lucam
    kb = NS - LS                                             # storey whose floor carries the lucam and the sign band
    z_e = Ftop + hs + 0.15                                   # top of the eaves cornice
    z_fs = z_e - 0.53                                        # underside of the frieze string
    T, PJ, PWC, PWI = 0.4, 0.10, 0.56, 0.45                  # wall thickness, pilaster projection / widths
    OV, HP, TV = 0.24, 0.28, 0.15                            # eaves overhang, parapet upstand, roof slab
    zR0 = z_e + 0.2                                          # slate plane height above the wall face
    z_r = zR0 + D / 2 * tanp                                 # ridge
    z_edge = zR0 - OV * tanp                                 # slate plane at the eaves edge
    z_apex = z_r + HP
    zb0, zb1 = F[kb] - 0.235, F[kb] + 0.715                  # painted sign band
    cd = float(p.lantern_cd)
    dark_sign = str(p.sign_style) == "dark"

    # ten materials = ten draw calls
    brick = mat(p.brick)
    stone = mat("stone_smooth", p.stone)
    slate = mat("slate")                                     # slates and the lead flashings
    wood = mat("wood_painted", p.paint)
    iron = mat("iron_painted", p.iron)
    terra = mat("terracotta", "#84503f")
    dark_m = mat("window_dark")
    lit_a = mat("window_lit", "#ff9c3c", glow=2.4, curtain=0.0)     # oil-lamp lit store rooms
    lit_b = mat("window_lit", "#ffdfa6", glow=3.8, curtain=0.15)    # brighter, whiter gas light; lamp flames
    glass_m = mat("glass_lantern", glow=2.2)

    G = {"brick": [], "stone": [], "wood": [], "iron": [], "lead": [], "terra": [], "slate": []}
    MATS = {"brick": brick, "stone": stone, "wood": wood, "iron": iron, "lead": slate, "terra": terra, "slate": slate}
    TONE = {}                                                # object name -> its own base grime (stone by stone tone shifts)

    def add(kind, o, M=None, world_uv=False, tone=None):
        """Register an object built in wall-local coordinates; M moves it to its wall."""
        if not world_uv:
            realize(o)
        apply_transform(o)
        if M is not None:
            o.data.transform(M)
            o.data.update()
        if world_uv:
            uv_box(o, offset=None)
        assign(o, MATS[kind])
        G[kind].append(o)
        if tone is not None:
            TONE[o.name] = float(tone)
        return o

    # ---- bays
    nb = max(3, int(W / 2.9 + 0.5))
    B = W / nb
    ww = clamp(B * 0.42, 1.05, 1.4)
    mid = nb // 2
    hidx = {"centre": mid, "left": 1 if nb >= 4 else mid, "right": nb - 2 if nb >= 4 else mid}[p.hoist_bay]
    xh = -W / 2 + (hidx + 0.5) * B
    ns = max(2, int(round(D / 3.4)))
    cart_w, cart_h, cart_r = min(2.4, B - 0.9), 2.9, 0.38
    ld_w, ld_h, ld_r = min(1.8, B - 1.1), min(2.4, hs - 0.62), 0.26
    pins = bool(p.lod_pins)
    mw = float(p.mullion) if float(p.mullion) >= 0.02 else 0.0
    cdoors = bool(p.canal_doors)
    ub = -W / 2 + (mid + 0.5) * B                             # canal-door bay (back wall coordinates)
    ar = rng.fork("age")

    # ------------------------------------------------------------------ window kits (instanced)
    class Kit:
        pass

    kr = rng.fork("kits")

    def make_kit(tag, w, h, rise, barred=False):
        k = Kit()
        k.w, k.h, k.rise, k.tag = w, h, rise, tag
        rv, fw = 0.12, 0.05
        wi, hi = w - 2 * fw, h - 2 * fw
        ri = rise * wi / w
        mb = _MB()
        # frame: only the face and the inner rim can be seen
        po, pn = _arch(0, -0.01, w + 0.03, h + 0.025, rise * (w + 0.03) / w, n=8), _arch(0, fw, wi, hi, ri, n=8)
        for i in range(len(po)):
            j = (i + 1) % len(po)
            mb.quad([(po[i][0], rv, po[i][1]), (po[j][0], rv, po[j][1]), (pn[j][0], rv, pn[j][1]), (pn[i][0], rv, pn[i][1])])
            mb.quad([(pn[i][0], rv, pn[i][1]), (pn[j][0], rv, pn[j][1]), (pn[j][0], rv + 0.05, pn[j][1]), (pn[i][0], rv + 0.05, pn[i][1])])

        def vbar(x, bw, y0, y1, z0, z1, sides=(1, 3)):
            mb.hexa([(x - bw / 2, y0, z0), (x + bw / 2, y0, z0), (x + bw / 2, y0, z1), (x - bw / 2, y0, z1),
                     (x - bw / 2, y1, z0), (x + bw / 2, y1, z0), (x + bw / 2, y1, z1), (x - bw / 2, y1, z1)], back=False, sides=sides)

        def hbar(x0, x1, bw, y0, y1, z, sides=(0, 2)):
            mb.hexa([(x0, y0, z - bw / 2), (x1, y0, z - bw / 2), (x1, y0, z + bw / 2), (x0, y0, z + bw / 2),
                     (x0, y1, z - bw / 2), (x1, y1, z - bw / 2), (x1, y1, z + bw / 2), (x0, y1, z + bw / 2)], back=False, sides=sides)

        ncol = max(3, int(round(wi / 0.25)))
        nrow = max(2, int(round((hi - ri) / 0.3)))
        # primary bars: a heavier regular grid (on the lines of the ventilator) between the fine glazing bars; their
        # faces are repeated as locked quads on the lit windows, so the mullion pattern survives every LOD
        c0 = (ncol - 2) // 2
        pv = set() if mw <= 0 else ({c0, c0 + 2} if ncol >= 4 else set(range(1, ncol)))
        ph = set() if mw <= 0 else {j for j in (nrow - 1, nrow - 3) if j >= 1}
        mk = _PB(True)
        ym = rv + 0.0134
        for i in range(1, ncol):
            x = -wi / 2 + wi * i / ncol
            zt = _arch_y(x, wi, hi, ri)
            if i in pv:
                vbar(x, mw, rv + 0.014, rv + 0.048, fw, fw + zt)
                mk.face([(x - mw / 2, ym, fw), (x + mw / 2, ym, fw), (x + mw / 2, ym, fw + zt), (x - mw / 2, ym, fw + zt)], (0, -1, 0))
            else:
                vbar(x, 0.016, rv + 0.018, rv + 0.046, fw, fw + zt)
        for j in range(1, nrow + 1):
            z = (hi - ri) * j / nrow
            if j in ph:
                hbar(-wi / 2, wi / 2, mw, rv + 0.014, rv + 0.048, fw + z)
                mk.face([(-wi / 2, ym, fw + z - mw / 2), (wi / 2, ym, fw + z - mw / 2), (wi / 2, ym, fw + z + mw / 2), (-wi / 2, ym, fw + z + mw / 2)],
                        (0, -1, 0))
            else:
                hbar(-wi / 2, wi / 2, 0.016, rv + 0.018, rv + 0.046, fw + z)
        k.mask = None
        if pins and mk.v:
            k.mask = mk.obj(tag + "_mask")
            assign(k.mask, iron)
        if ri > 0.3:                                          # round head: one more bar inside the arch
            z = hi - ri * 0.5
            R = (wi * wi / 4 + ri * ri) / (2 * ri)
            hx_ = sqrt(max(R * R - (z - (hi - R)) ** 2, 0.0))
            hbar(-hx_, hx_, 0.016, rv + 0.018, rv + 0.046, fw + z)
        if ncol >= 4 and nrow >= 4 and not barred:            # pivot-hung ventilator: a heavier sub-frame
            x0, x1 = -wi / 2 + wi * c0 / ncol, -wi / 2 + wi * (c0 + 2) / ncol
            z0, z1 = fw + (hi - ri) * (nrow - 3) / nrow, fw + (hi - ri) * (nrow - 1) / nrow
            for s_ in (-1, 1):
                vbar((x0 + x1) / 2 + s_ * ((x1 - x0) / 2 - 0.003), 0.03, rv + 0.01, rv + 0.03, z0 - 0.012, z1 + 0.012)
                hbar(x0 + 0.018, x1 - 0.018, 0.03, rv + 0.01, rv + 0.03, (z0 + z1) / 2 + s_ * ((z1 - z0) / 2 - 0.003))
        if barred:                                            # security bars leaded into the stone sill
            nbar = max(4, int(round(w / 0.2)))
            for i in range(1, nbar):
                x = -w / 2 + w * i / nbar
                vbar(x, 0.017, 0.042, 0.059, 0.0, _arch_y(x, w, h, rise) + 0.01)
            for fz in (0.33, 0.7):
                hbar(-w / 2 - 0.03, w / 2 + 0.03, 0.03, 0.045, 0.056, (h - rise) * fz)
        k.frame = mb.obj(tag + "_frame")
        assign(k.frame, iron)

        # glass: every pane its own slightly tilted quad (they catch the lantern light one by one); the variants 1, 2
        # have a few panes replaced by bits of board
        yg = rv + 0.036
        zl = [fw - 0.01] + [fw + (hi - ri) * j / nrow for j in range(1, nrow + 1)]

        def panes(pp):
            gm, pm = _MB(), _MB()
            for i in range(ncol):
                xa_, xb_ = -wi / 2 + wi * i / ncol, -wi / 2 + wi * (i + 1) / ncol
                if i == 0:
                    xa_ -= 0.01
                if i == ncol - 1:
                    xb_ += 0.01
                for j in range(nrow + 1):
                    if j < nrow:
                        q = [(xa_, zl[j]), (xb_, zl[j]), (xb_, zl[j + 1]), (xa_, zl[j + 1])]
                    else:
                        ta = fw + _arch_y(clamp(xa_, -wi / 2, wi / 2), wi, hi, ri) + 0.012
                        tb = fw + _arch_y(clamp(xb_, -wi / 2, wi / 2), wi, hi, ri) + 0.012
                        q = [(xa_, zl[nrow]), (xb_, zl[nrow]), (xb_, tb), (xa_, ta)]
                    xc, zc = (xa_ + xb_) / 2, (q[0][1] + q[2][1]) / 2
                    if pp > 0 and j < nrow and kr.chance(pp):
                        pm.quad([(x, yg - 0.007, z) for x, z in q])
                    else:
                        tx, tz = kr.u(-0.056, 0.056), kr.u(-0.056, 0.056)
                        ex = max(abs(tx * (x - xc) + tz * (z - zc)) for x, z in q)
                        if ex > 0.013:                        # no corner nearer than 5 mm to the bar fronts (was down to 2 mm)
                            tx, tz = tx * 0.013 / ex, tz * 0.013 / ex
                        gm.quad([(x, yg + tx * (x - xc) + tz * (z - zc), z) for x, z in q])
            g = gm.obj(tag + "_glass")
            assign(g, dark_m)
            pt = None
            if pm.v:
                pt = pm.obj(tag + "_patch")
                uv_grain(pt, axis='Z')
                realize(pt)
                assign(pt, wood)
                paint(pt, wear=1.0, grime=0.35, mode='set')
            return g, pt

        k.glass = [panes(0.0), panes(0.09), panes(0.2)]
        k.sill = _block(tag + "_sill", (w + 0.26, 0.3, 0.12), 0.014, loc=(0, 0.055, -0.045))
        apply_transform(k.sill)
        assign(k.sill, stone)
        mbv = _MB()
        _ring(mbv, 0, 0, w, h, rise, 0.225, 0.012, kr, step=0.085 * (1 + 0.4 * lod))
        k.vous = mbv.obj(tag + "_vous")
        assign(k.vous, terra)
        k.Mf, k.Ml, k.Md = [], [], [[], [], []]
        return k

    def lit_card(kit, Mi, base, m_):
        """The lit room of one window: a locked card behind the bars, dimmed unevenly (dirty glass, dark towards the sill)."""
        rv, fw = 0.12, 0.05
        wi, hi = kit.w - 2 * fw, kit.h - 2 * fw
        ri = kit.rise * wi / kit.w
        n3 = Mi.to_3x3() @ Vector((0, -1, 0))
        nx_, nz_ = 3, 3
        xs_ = [-wi / 2 - 0.01 + (wi + 0.02) * i / nx_ for i in range(nx_ + 1)]

        def zz(i, j):
            zt = fw + _arch_y(clamp(xs_[i], -wi / 2, wi / 2), wi, hi, ri) + 0.012
            return fw - 0.01 + (zt - fw + 0.01) * j / nz_
        pb = _PB(pins)
        for i in range(nx_):
            for j in range(nz_):
                pb.face([Mi @ Vector((xs_[a], rv + 0.036, zz(a, b))) for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))], n3)
        o = pb.obj(kit.tag + "_lit")
        assign(o, m_)
        z0w, hh = (Mi @ Vector((0, 0, 0))).z, kit.h
        nz = noise_mask(0.45, seed=int(kr.u(0, 9999)), lo=0.25, hi=0.8)
        paint(o, grime=lambda co, n: np.clip(base + 0.34 * np.clip(1.0 - (co[:, 2] - z0w) / hh, 0, 1) ** 1.6 + 0.24 * nz(co, n), 0, 1), mode='set')
        return o

    # ---- lit hoist loft: a warm glow card set back behind the window with the hoisting gear standing in front of it
    lit_polys = []
    sil, sil_f = _MB(), _MB()
    sil_p = _PB(True)                                         # locked faces of the big silhouettes (they outlive the LODs)

    def hq(mb, Mi, quad, y0, y1, pin=False):
        mb.hexa([tuple(Mi @ Vector((x, y0, z))) for x, z in quad] + [tuple(Mi @ Vector((x, y1, z))) for x, z in quad])
        if pin and pins:
            sil_p.face([Mi @ Vector((x, y0 - 0.0006, z)) for x, z in quad], Mi.to_3x3() @ Vector((0, -1, 0)))

    def hx(mb, Mi, x0, x1, y0, y1, z0, z1, sx=0.0, pin=False):
        hq(mb, Mi, [(x0, z0), (x1, z0), (x1 + sx, z1), (x0 + sx, z1)], y0, y1, pin)

    def lit_scene(Mi, w, h, ya, yb, r, margin, kind):
        """Mi = frame of the opening (x across, +y into the room, z up from the sill); ya..yb = free depth."""
        m_ = margin
        lit_polys.append(([tuple(Mi @ Vector(q)) for q in ((-w / 2 - m_, yb, -m_), (w / 2 + m_, yb, -m_), (w / 2 + m_, yb, h + m_), (-w / 2 - m_, yb, h + m_))],
                          Mi.to_3x3() @ Vector((0, -1, 0)), (Mi @ Vector((0, 0, 0))).z, h))
        y0, y1 = ya + 0.1, ya + 0.15
        if kind == "wheel":                                   # spoked hoist wheel on its axle
            cxw, czw, rw = r.u(-0.06, 0.06), h * r.u(0.36, 0.46), 0.41
            nsg = 14
            for i in range(nsg):
                a0, a1 = 2 * pi * i / nsg, 2 * pi * (i + 1) / nsg
                hq(sil, Mi, [(cxw + rr * sin(a), czw + rr * cos(a)) for rr, a in ((rw - 0.045, a0), (rw, a0), (rw, a1), (rw - 0.045, a1))], y0, y1)
            a_off = r.u(0, 1.0)
            for i in range(6):
                a = a_off + 2 * pi * i / 6
                dx, dz, px, pz = sin(a), cos(a), cos(a) * 0.016, -sin(a) * 0.016
                hq(sil, Mi, [(cxw + 0.03 * dx - px, czw + 0.03 * dz - pz), (cxw + 0.03 * dx + px, czw + 0.03 * dz + pz),
                             (cxw + (rw - 0.03) * dx + px, czw + (rw - 0.03) * dz + pz), (cxw + (rw - 0.03) * dx - px, czw + (rw - 0.03) * dz - pz)],
                   y0 + 0.012, y1 - 0.012)
            hx(sil, Mi, cxw - 0.05, cxw + 0.05, y0 - 0.03, y1 + 0.03, czw - 0.05, czw + 0.05)
            hx(sil, Mi, -w / 2 - 0.2, w / 2 + 0.2, y1 + 0.03, y1 + 0.13, czw - 0.06, czw + 0.06, pin=True)
        else:                                                 # windlass: rope drum with flanges, crank and a taut fall
            czw, rd = h * r.u(0.3, 0.38), 0.105
            hx(sil, Mi, -w / 2 - 0.2, w / 2 + 0.2, y0, y0 + 2 * rd, czw - rd, czw + rd, pin=True)
            for fx_ in (-0.3, 0.16):
                hx(sil, Mi, fx_ * w - 0.022, fx_ * w + 0.022, y0 - 0.08, y0 + 2 * rd + 0.08, czw - rd - 0.085, czw + rd + 0.085, pin=True)
            for q_ in range(4):
                xq = (-0.2 + 0.07 * q_) * w
                hx(sil, Mi, xq - 0.016, xq + 0.016, y0 - 0.02, y0 + 2 * rd + 0.02, czw - rd - 0.02, czw + rd + 0.02)
            xr = (-0.2 + 0.07 * 4) * w
            hx(sil, Mi, xr - 0.014, xr + 0.014, y0 - 0.03, y0 - 0.002, czw, h + 0.25)                   # the fall, up to the cathead sheave
            xk = 0.33 * w                                     # crank
            hx(sil, Mi, xk - 0.018, xk + 0.018, y0 - 0.02, y0 + 0.02, czw - 0.02, czw + 0.3, sx=0.1)
            hx(sil, Mi, xk + 0.085, xk + 0.25, y0 - 0.03, y0 + 0.0, czw + 0.27, czw + 0.31)
            hx(sil, Mi, -0.09 * w - 0.02, -0.09 * w + 0.02, y0 + 2 * rd, y0 + 2 * rd + 0.1, czw - 0.75, czw - rd, sx=-0.12)   # pawl post
        for s_ in (-1, 1):                                    # timber cheeks carrying the axle
            hx(sil, Mi, s_ * (w / 2 + 0.02) - 0.06, s_ * (w / 2 + 0.02) + 0.06, y1 + 0.03, y1 + 0.13, -0.3, h + 0.3, pin=True)
        xq = (r.u(0.2, 0.36) * (1 if r.chance(0.5) else -1)) * w  # hanging oil lamp: the hot spot in the room
        zl_ = h * r.u(0.55, 0.74)
        yl = ya + 0.05
        hx(sil, Mi, xq - 0.004, xq + 0.004, yl - 0.004, yl + 0.004, zl_ + 0.2, h + 0.05)
        hx(sil, Mi, xq - 0.08, xq + 0.08, yl - 0.05, yl + 0.05, zl_ + 0.185, zl_ + 0.2)
        hx(sil, Mi, xq - 0.045, xq + 0.045, yl - 0.04, yl + 0.04, zl_, zl_ + 0.045)
        hx(sil_f, Mi, xq - 0.016, xq + 0.016, yl - 0.016, yl + 0.016, zl_ + 0.06, zl_ + 0.13)

    kg = make_kit("wg", ww, 1.3, 0.16, barred=True)
    km = make_kit("wm", ww, hs - 1.35, 0.18)
    kt = make_kit("wt", ww, hs - 1.6, 0.18)
    ka = make_kit("wa", 0.9, 1.35, 0.45)

    # ------------------------------------------------------------------ plinth stones (instanced, three tones)
    pr = rng.fork("plinth")
    pl_base = []
    for i in range(3):
        b = _block("plinth%d" % i, (0.9, 0.17, 0.55), 0.016, drop_bottom=True)
        assign(b, stone)
        paint(b, grime=(0.0, 0.2, 0.4)[i], mode='set')
        pl_base.append(b)
    pl_M = [[], [], []]

    def plinth_run(M, a, b):
        if b - a < 0.15:
            return
        n = max(1, int(round((b - a) / pr.u(1.0, 1.35))))
        cuts = [a] + sorted(a + (b - a) * (i + pr.u(-0.18, 0.18)) / n for i in range(1, n)) + [b]
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            pl_M[int(pr.u(0, 2.999))].append(M @ trs(((x0 + x1) / 2, 0.035 + pr.u(-0.006, 0.004), 0.275),
                                                    scale=((x1 - x0 - 0.01) / 0.9, 1.0, 1.0)))

    dent = box("dentil", (0.11, 0.07, 0.075))
    _drop_faces(dent, lambda c, n: n.y > 0.5 or n.z > 0.5)
    assign(dent, brick)
    dent_M = []
    plate = lathe("pattress", [(0.135, 0.0), (0.125, 0.014), (0.055, 0.024), (0.03, 0.04), (0.028, 0.058), (0.0, 0.06)], segments=6)
    assign(plate, iron)
    plate_M = []
    bolt_b = bolt("stud", 0.013, 0.01, kind="pyramid")
    assign(bolt_b, iron)
    bolt_M = []

    win_all = {}                                               # wall tag -> [(kit, M, cx, z0, storey)]
    pipes = []                                                 # (M, x) downpipe positions
    bands = {}                                                 # wall tag -> (M, L, pilaster intervals, excluded intervals)
    str_z = [F1 - 0.2, z_fs] + ([Ftop + 0.835] if NS >= 4 else [])

    def string_course(M, L, zb, hgt, proj, gaps=()):
        """Moulded stone string, stone by stone: open joints, each stone a hair out of line and of its own tone."""
        prof = [(0.03, zb), (-proj * 0.62, zb), (-proj * 0.66, zb + hgt * 0.2), (-proj * 0.9, zb + hgt * 0.34), (-proj, zb + hgt * 0.42),
                (-proj, zb + hgt * 0.84), (-proj * 0.8, zb + hgt), (0.03, zb + hgt)]
        xs = [-L / 2] + [c for gx, hw in sorted(gaps) for c in (gx - hw, gx + hw)] + [L / 2]
        for i in range(0, len(xs), 2):
            a, b = xs[i], xs[i + 1]
            first, last = i == 0, i == len(xs) - 2
            n = max(1, int(round((b - a) / pr.u(1.0, 1.4))))
            cuts = [a] + sorted(a + (b - a) * (q + pr.u(-0.14, 0.14)) / n for q in range(1, n)) + [b]
            for q in range(n):
                e0, e1 = first and q == 0, last and q == n - 1
                dy, dz = pr.u(-0.005, 0.004), pr.u(-0.003, 0.003)
                pf = [(y + (dy if y < 0 else 0.0), z + dz) for y, z in prof]
                o = _prism("string", pf, cuts[q] + (0.0 if q == 0 else 0.007), cuts[q + 1] - (0.0 if q == n - 1 else 0.007),
                           1.0 if e0 else 0.0, 1.0 if e1 else 0.0, not e0, not e1)
                add("stone", o, M, tone=pr.u(0.0, 0.5) ** 1.3)

    # ------------------------------------------------------------------ brick wall: an even quad grid around the openings
    def grid_wall(tag, L, ops, gable):
        """ops = (cx, z0, w, h, rise) arched openings (z0 = 0: open to the ground). Even quad grid whose lines run through
        every jamb, sill and springing; the rows above an opening bend up to its arch. Reveals 0.3 m deep."""
        zg = zR0 + HP
        mx, mz, fine = [-L / 2, L / 2], [0.0, z_e], []
        if gable:
            mx.append(0.0)
            mz += [zg, z_apex]
        for (cx, z0, w, h, rise) in ops:
            mx += [cx - w / 2, cx + w / 2]
            mz += [z0, z0 + h - rise, z0 + h + 0.1]
            fine.append((cx - w / 2, cx + w / 2))
        xs, zs = _lines(mx, 0.34 * (1 + 0.5 * lod), fine), _lines(mz, 0.42 * (1 + 0.4 * lod))
        nx, nz = len(xs), len(zs)

        def idx(vals, v):
            return min(range(len(vals)), key=lambda i: abs(vals[i] - v))
        Z = [[zs[j]] * nx for j in range(nz)]
        dead = set()
        rev = _MB()
        dp = 0.3
        for (cx, z0, w, h, rise) in ops:
            il, ir = idx(xs, cx - w / 2), idx(xs, cx + w / 2)
            jl, js, jh = idx(zs, z0), idx(zs, z0 + h - rise), idx(zs, z0 + h + 0.1)
            for i in range(il + 1, ir):
                a = z0 + _arch_y(xs[i] - cx, w, h, rise)
                for j in range(js, jh):
                    t = (zs[j] - zs[js]) / (zs[jh] - zs[js])
                    Z[j][i] = a + t * (zs[jh] - a)
            for i in range(il, ir):
                for j in range(jl, js):
                    dead.add((i, j))
            x0, x1 = xs[il], xs[ir]
            for j in range(jl, js):                           # jambs
                za, zc = zs[j], zs[j + 1]
                rev.quad([(x0, 0, za), (x0, dp, za), (x0, dp, zc), (x0, 0, zc)])
                rev.quad([(x1, 0, zc), (x1, dp, zc), (x1, dp, za), (x1, 0, za)])
            for i in range(il, ir):                           # sill and soffit
                if z0 > 0.01:
                    rev.quad([(xs[i], 0, z0), (xs[i + 1], 0, z0), (xs[i + 1], dp, z0), (xs[i], dp, z0)])
                za, zc = Z[js][i], Z[js][i + 1]
                rev.quad([(xs[i], 0, za), (xs[i], dp, za), (xs[i + 1], dp, zc), (xs[i + 1], 0, zc)])
        vmap, verts, faces = {}, [], []

        def vid(x, z):
            key = (round(x, 4), round(z, 4))
            if key not in vmap:
                vmap[key] = len(verts)
                verts.append((x, 0.0, z))
            return vmap[key]

        def under(q):
            return zg + (L / 2 - abs(q[0])) * tanp - q[1]

        for j in range(nz - 1):
            for i in range(nx - 1):
                if (i, j) in dead:
                    continue
                poly = [(xs[i], Z[j][i]), (xs[i + 1], Z[j][i + 1]), (xs[i + 1], Z[j + 1][i + 1]), (xs[i], Z[j + 1][i])]
                if gable and zs[j + 1] > zg - 1e-6:           # raking top of the parapet gable
                    out = []
                    for q in range(len(poly)):
                        a, b = poly[q], poly[(q + 1) % len(poly)]
                        fa, fb = under(a), under(b)
                        if fa >= -1e-7:
                            out.append(a)
                        if (fa > 1e-7 and fb < -1e-7) or (fa < -1e-7 and fb > 1e-7):
                            t = fa / (fa - fb)
                            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                    poly = out
                ids = []
                for x, z in poly:
                    v = vid(x, z)
                    if v not in ids:
                        ids.append(v)
                if len(ids) >= 3:
                    faces.append(tuple(ids))
        if gable:                                             # inner face of the parapet above the slates, wall head
            rev.quad([(-L / 2, T, zR0 - 0.1), (-L / 2, T, zg), (0.0, T, z_apex), (0.0, T, z_r - 0.1)])
            rev.quad([(L / 2, T, zR0 - 0.1), (0.0, T, z_r - 0.1), (0.0, T, z_apex), (L / 2, T, zg)])
        else:
            rev.quad([(-L / 2, 0, z_e), (L / 2, 0, z_e), (L / 2, T, z_e), (-L / 2, T, z_e)])
        nv = len(verts)
        return mesh_object(tag + "_wall", verts + rev.v, faces + [tuple(i + nv for i in f) for f in rev.f])

    drips = []                                                 # (wall matrix, x, z) ends of the window sills: the stains start here

    # ------------------------------------------------------------------ one masonry wall
    def make_wall(tag, M, L, nbay, gable, hoist, canal=False):
        Bw = L / nbay
        ops, wins, notch = [], [], None
        for i in range(nbay):
            cx = -L / 2 + (i + 0.5) * Bw
            for k in range(NS):
                if i == hoist and canal and k >= 1:
                    ops.append((cx, F[k], ld_w, ld_h, ld_r))
                    continue
                if i == hoist and not canal:
                    if k == 0:
                        notch = cx
                        ops.append((cx, 0.0, cart_w, cart_h, cart_r))
                    elif k < NS - LS:
                        ops.append((cx, F[k], ld_w, ld_h, ld_r))
                    continue
                kit, z0 = (kg, 1.55) if k == 0 else ((kt, F[k] + 0.95) if k == NS - 1 else (km, F[k] + 0.85))
                ops.append((cx, z0, kit.w, kit.h, kit.rise))
                wins.append((kit, cx, z0, k))
        if gable:
            ops.append((0.0, z_e + 0.55, ka.w, ka.h, ka.rise))
            wins.append((ka, 0.0, z_e + 0.55, NS))
        add("brick", grid_wall(tag, L, ops, gable), M, world_uv=True)

        # pilasters / piers: gridded like the wall, chamfered arrises
        pil = []

        def pier(x0, x1, zb, zt):
            c, y1 = 0.008, -PJ
            n = max(1, int(ceil((zt - zb) / (0.45 * (1 + 0.4 * lod)))))
            mb = _MB()
            P = [(x0, 0.0), (x0, y1 + c), (x0 + c, y1), (x1 - c, y1), (x1, y1 + c), (x1, 0.0)]
            for q in range(n):
                za, zc = zb + (zt - zb) * q / n, zb + (zt - zb) * (q + 1) / n
                for a, b in zip(P[:-1], P[1:]):
                    mb.quad([(a[0], a[1], za), (b[0], b[1], za), (b[0], b[1], zc), (a[0], a[1], zc)])
            mb.quad([(x0, y1, zt), (x1, y1, zt), (x1, 0.0, zt), (x0, 0.0, zt)])
            add("brick", mb.obj(tag + "_pier"), M, world_uv=True)
            pil.append((x0, x1))

        if gable:
            pier(-L / 2 - PJ, -L / 2 + PWC, F1 - 0.02, z_fs + 0.02)
            pier(L / 2 - PWC, L / 2 + PJ, F1 - 0.02, z_fs + 0.02)
        else:
            pier(-L / 2, -L / 2 + PWC, 0.0, z_fs + 0.02)
            pier(L / 2 - PWC, L / 2, 0.0, z_fs + 0.02)
        for i in range(1, nbay):
            x = -L / 2 + i * Bw
            pier(x - PWI / 2, x + PWI / 2, F1 - 0.02, z_fs + 0.02)
        excl = []
        if hoist is not None:
            hx_ = -L / 2 + (hoist + 0.5) * Bw
            excl = [(hx_ - ld_w / 2 - 0.32, hx_ + ld_w / 2 + 0.32)] if canal else [(hx_ - Bw / 2 + PWI / 2, hx_ + Bw / 2 - PWI / 2)]
        bands[tag] = (M, L, sorted(pil), excl, gable)
        # tie-rod pattress plates at the upper floors
        pxs = [-L / 2 + i * Bw for i in range(1, nbay)] + [-L / 2 + PWC / 2, L / 2 - PWC / 2]
        for k in range(2, NS):
            for x in pxs:
                plate_M.append(M @ trs((x, -PJ, zb0 - 0.2 if k == kb else F[k] - 0.12), (90, 0, rng.u(0, 360))))
        if gable:
            plate_M.append(M @ trs((-L * 0.2, 0, z_e + 0.3), (90, 0, 0)))
            plate_M.append(M @ trs((L * 0.2, 0, z_e + 0.3), (90, 0, 0)))

        # plinth
        if gable:
            plinth_run(M, -L / 2, L / 2)
        else:
            for sx in (-1, 1):
                o = _block(tag + "_pierbase", (PWC + 0.07, 0.2, 0.62), 0.018, loc=(sx * (L / 2 - PWC / 2 + 0.015), -PJ - 0.014 + 0.1, 0.31),
                           drop_bottom=True)
                add("stone", o, M, tone=pr.u(0.05, 0.4))
            a, b = -L / 2 + PWC + 0.04, L / 2 - PWC - 0.04
            if notch is not None:
                plinth_run(M, a, notch - cart_w / 2 - 0.01)
                plinth_run(M, notch + cart_w / 2 + 0.01, b)
            else:
                plinth_run(M, a, b)

        # stone strings
        gaps = []
        if not gable:
            gaps = [(-(L / 2 - 0.4), 0.1), (L / 2 - 0.4, 0.1)] if tag == "front" else [(L / 2 - 0.4, 0.1)]
            for gx, _hw in gaps:
                pipes.append((M, gx))
        string_course(M, L, F1 - 0.2, 0.2, 0.17, gaps)
        string_course(M, L, z_fs, 0.14, 0.155, [(-L / 2 + (hoist + 0.5) * Bw, 0.085)] if (canal and hoist is not None) else [])
        if NS >= 4:                                           # sill band under the top storey windows
            if canal and hoist is not None:
                gaps = gaps + [(-L / 2 + (hoist + 0.5) * Bw, ld_w / 2 + 0.03)]
            string_course(M, L, Ftop + 0.95 - 0.115, 0.1, 0.135, gaps)

        # corbelled brick cornice with dentils (eaves walls)
        if not gable:
            prof = [(0.02, z_e - 0.225), (-0.05, z_e - 0.225), (-0.05, z_e - 0.15), (-0.105, z_e - 0.15), (-0.105, z_e - 0.075),
                    (-0.16, z_e - 0.075), (-0.16, z_e), (0.02, z_e)]
            nsg = max(2, int(ceil(L / (0.7 * (1 + 0.5 * lod)))))
            secs = [[(-L / 2 + L * q / nsg, y, z) for y, z in prof] for q in range(nsg + 1)]
            add("brick", _loft(tag + "_cornice", secs, closed=False), M, world_uv=True)
            nd = int((L - 0.2) / 0.225)
            for i in range(nd + 1):
                dent_M.append(M @ trs((-nd * 0.1125 + i * 0.225, -0.033, z_e - 0.2625)))
        win_all[tag] = [(kit, M, cx, z0, k) for (kit, cx, z0, k) in wins]

    M_front = Matrix.Identity(4)
    M_back = Matrix.Translation((0, D, 0)) @ Matrix.Rotation(pi, 4, 'Z')
    M_left = Matrix.Translation((-W / 2, D / 2, 0)) @ Matrix.Rotation(-pi / 2, 4, 'Z')
    M_right = Matrix.Translation((W / 2, D / 2, 0)) @ Matrix.Rotation(pi / 2, 4, 'Z')
    make_wall("front", M_front, W, nb, False, hidx)
    make_wall("back", M_back, W, nb, False, mid if cdoors else None, canal=True)
    make_wall("left", M_left, D, ns, True, None)
    make_wall("right", M_right, D, ns, True, None)

    # ---- windows: lit / dark / patched / boarded up
    lr = rng.fork("lit")
    p_board = clamp((age - 0.5) * 0.8, 0.0, 0.4)
    p_patch = clamp(age * 0.7 - 0.1, 0.0, 0.65)

    def boarded(M, cx, z0, kit):
        """Rough boards nailed across the opening inside the reveal."""
        mb = _MB()
        z = z0 + 0.02 + ar.u(0, 0.08)
        ztop = z0 + kit.h - kit.rise * 0.5
        hl = kit.w / 2 - 0.012
        while z < ztop - 0.2:
            bw = ar.u(0.16, 0.25)
            if not ar.chance(0.14):
                dl, dr_ = ar.u(-0.035, 0.035), ar.u(-0.035, 0.035)
                y0 = 0.035 + ar.u(0, 0.012)
                q = [(cx - hl, z + dl), (cx + hl, z + dr_), (cx + hl, z + bw + dr_), (cx - hl, z + bw + dl)]
                mb.hexa([(x, y0, zz_) for x, zz_ in q] + [(x, y0 + 0.024, zz_) for x, zz_ in q], back=False, sides=(0, 2))
            z += bw + ar.u(0.008, 0.06)
        if not mb.v:
            return
        o = mb.obj("boarding")
        uv_grain(o, axis='X')
        add("wood", o, M)
        paint(o, wear=ar.u(0.8, 1.0), grime=0.25, mode='set')

    n_lit = 0
    for tag, wl_ in win_all.items():
        flags = [lr.chance(p.lit_ratio) for _ in wl_]
        on = p.lit_ratio > 0.02
        low = [i for i, w_ in enumerate(wl_) if w_[4] <= 1]
        if on and low and not any(flags[i] for i in low):
            flags[low[int(lr.u(0, len(low) - 0.001))]] = True  # someone is always working late on the lower floors
        up = [i for i, w_ in enumerate(wl_) if 2 <= w_[4] < NS]
        if on and tag == "back" and up and not any(flags[i] for i in up):
            flags[up[int(lr.u(0, len(up) - 0.001))]] = True    # and a light high up over the canal
        ch = None
        c1 = [i for i, w_ in enumerate(wl_) if w_[4] == 1]
        if on and tag == "front" and c1:                       # the counting house on the first floor: the brightest window
            ch = c1[int(lr.u(0, len(c1) - 0.001))]
            flags[ch] = True
        for i, ((kit, M, cx, z0, k), f) in enumerate(zip(wl_, flags)):
            Mi = M @ Matrix.Translation((cx, 0, z0))
            kit.Mf.append(Mi)
            for sx in (-1, 1):
                drips.append((M, cx + sx * kit.w / 2, z0 - 0.1))
            if f:
                n_lit += 1
                kit.Ml.append(Mi)
                if i == ch:
                    lit_card(kit, Mi, 0.0, lit_b)
                else:
                    lit_card(kit, Mi, lr.u(0.08, 0.6) + (0.14 if k >= NS - 1 else 0.0), lit_a if lr.chance(0.7) else lit_b)
            else:
                if kit is km or kit is kt:
                    if ar.chance(p_board):
                        boarded(M, cx, z0, kit)
                kit.Md[(1 + int(ar.u(0, 1.999))) if ar.chance(p_patch) else 0].append(Mi)
    for kit in (kg, km, kt, ka):
        if kit.Mf:
            G["iron"].append(instances_to_mesh(kit.tag + "_frames", kit.frame, kit.Mf))
            G["stone"].append(instances_to_mesh(kit.tag + "_sills", kit.sill, kit.Mf))
            G["terra"].append(instances_to_mesh(kit.tag + "_arches", kit.vous, kit.Mf))
            for v in range(3):
                if kit.Md[v]:
                    g, pt = kit.glass[v]
                    dg = instances_to_mesh(kit.tag + "_dark%d" % v, g, kit.Md[v], uv_jitter=False)
                    paint(dg, grime=mask_mul(noise_mask(0.3, seed=p.seed + 31 + v, lo=0.35, hi=0.85), 0.75), mode='set')
                    if pt is not None:
                        instances_to_mesh(kit.tag + "_patched%d" % v, pt, kit.Md[v])
            if kit.mask is not None and kit.Ml:
                G["iron"].append(instances_to_mesh(kit.tag + "_mask", kit.mask, kit.Ml))
        delete([kit.frame, kit.sill, kit.vous] + [o for gp in kit.glass for o in gp if o is not None] + ([kit.mask] if kit.mask is not None else []))
    for i, b in enumerate(pl_base):
        if pl_M[i]:
            G["stone"].append(instances_to_mesh("plinth_stones%d" % i, b, pl_M[i]))
    delete(pl_base)
    G["brick"].append(instances_to_mesh("dentils", dent, dent_M, keep_base=False))
    G["iron"].append(instances_to_mesh("pattress_plates", plate, plate_M, keep_base=False))

    # ------------------------------------------------------------------ loading doors (wall coordinates)
    dr = rng.fork("doors")

    def make_door(tag, Mw, cx, z0, w, h, rise, cart):
        def add_(kind, o, **kw):
            return add(kind, o, Mw, **kw)

        rv, fwd = 0.17, 0.09
        wi, hi = w - 2 * fwd, h - fwd
        ri = rise * wi / w
        outer = _arch(cx, z0, w + 0.03, h + 0.015, rise)
        inner = _arch(cx, z0, wi, hi, ri)[::-1]
        add_("wood", shape_extrude(tag + "_frame", outer + inner, [], depth=0.11, loc=(0, rv + 0.1, 0), rot=(90, 0, 0), bevel=0.004))
        mb = _MB()
        for s in (-1, 1):
            xa, xb = (cx - wi / 2 + 0.004, cx - 0.004) if s < 0 else (cx + 0.004, cx + wi / 2 - 0.004)
            nbd = max(3, int(round((xb - xa) / 0.19)))
            for i in range(nbd):
                x0 = xa + (xb - xa) * i / nbd + 0.002
                x1 = xa + (xb - xa) * (i + 1) / nbd - 0.002
                yf = rv + 0.036 + dr.u(-0.004, 0.004)
                zb = z0 + 0.012 + dr.u(0, 0.012 + 0.02 * age)
                t0 = z0 + _arch_y(x0 - cx, wi, hi, ri) - 0.006
                t1 = z0 + _arch_y(x1 - cx, wi, hi, ri) - 0.006
                mb.hexa([(x0, yf, zb), (x1, yf, zb), (x1, yf, t1), (x0, yf, t0), (x0, rv + 0.08, zb), (x1, rv + 0.08, zb),
                         (x1, rv + 0.08, t1), (x0, rv + 0.08, t0)], back=False)
        boards = mb.obj(tag + "_boards")
        uv_grain(boards, axis='Z')
        add_("wood", boards)
        bo = _arch(cx, z0, wi + 0.02, hi + 0.01, ri)
        back = mesh_object(tag + "_back", [(x, rv + 0.075, z) for x, z in bo], [list(range(len(bo)))])
        add_("wood", back)
        add_("wood", plank(tag + "_stile", hi - 0.04, 0.075, 0.022, loc=(cx + 0.03, rv + 0.026, z0 + (hi - 0.04) / 2 + 0.01), rot=(0, -90, 0), bevel=0.004))
        # strap hinges with forged spear ends, studs and pintles
        Ls = wi / 2 * 0.74
        for s in (-1, 1):
            xp = cx + s * (wi / 2 - 0.012)
            for fz in ((0.14, 0.48, 0.8) if cart else (0.2, 0.74)):
                zc = z0 + (hi - ri) * fz + (0.0 if cart else 0.05)
                pts = [(0, -0.032), (Ls * 0.78, -0.02), (Ls * 0.86, -0.046), (Ls, 0), (Ls * 0.86, 0.046), (Ls * 0.78, 0.02), (0, 0.032)]
                if s > 0:
                    pts = [(-x, z) for x, z in pts][::-1]
                add_("iron", shape_extrude(tag + "_strap", pts, [], depth=0.009, loc=(xp, rv + 0.034, zc), rot=(90, 0, 0)))
                add_("iron", cylinder(tag + "_pintle", 0.02, 0.11, loc=(xp + s * 0.014, rv + 0.02, zc), verts=8))
                for fx_ in (0.1, 0.4, 0.68):
                    bolt_M.append(Mw @ trs((xp - s * Ls * fx_, rv + 0.025, zc), (90, 0, dr.u(0, 90))))
        # arch ring
        mb = _MB()
        _ring(mb, cx, z0, w, h, rise, 0.34 if cart else 0.225, 0.016, dr)
        add_("terra", mb.obj(tag + "_ring"))
        if cart:
            add_("stone", _block(tag + "_key", (0.3, 0.16, 0.62), 0.02, loc=(cx, -0.02, z0 + h + 0.24)), tone=dr.u(0.0, 0.2))
            for s in (-1, 1):
                add_("stone", _block(tag + "_impost", (0.42, 0.14, 0.2), 0.016, loc=(cx + s * (w / 2 + 0.2), -0.02, z0 + h - rise - 0.1)), tone=dr.u(0.1, 0.45))
                add_("stone", lathe(tag + "_guard", [(0.19, 0.0), (0.19, 0.07), (0.15, 0.3), (0.1, 0.5), (0.05, 0.6), (0.0, 0.62)], segments=10,
                                   loc=(cx + s * (w / 2 + 0.03), -0.03, 0.0)), tone=dr.u(0.3, 0.6))
            add_("stone", _block(tag + "_threshold", (w + 0.5, 0.62, 0.05), 0.012, loc=(cx, -0.05, 0.025), drop_back=False, drop_bottom=True), tone=0.35)
            zb = z0 + 1.25
            add_("iron", box(tag + "_dropbar", (wi * 0.62, 0.012, 0.06), loc=(cx, rv + 0.012, zb), bevel=0.003, segments=1))
            for dx in (-wi * 0.27, -0.12, 0.12, wi * 0.27):
                add_("iron", box(tag + "_staple", (0.03, 0.035, 0.1), loc=(cx + dx, rv + 0.012, zb), bevel=0.004, segments=1))
            add_("iron", box(tag + "_padlock", (0.07, 0.03, 0.085), loc=(cx + 0.12, rv - 0.005, zb - 0.1), bevel=0.008, segments=1))
            add_("iron", torus(tag + "_shackle", 0.028, 0.007, loc=(cx + 0.12, rv - 0.005, zb - 0.05), rot=(90, 0, 0), major_seg=10, minor_seg=5))
            add_("iron", torus(tag + "_ring_pull", 0.075, 0.009, loc=(cx - wi * 0.2, rv + 0.014, z0 + 1.0), rot=(78, 0, 0), major_seg=14, minor_seg=5))
        else:
            if z0 > F1 + 0.1:
                add_("stone", _block(tag + "_sill", (w + 0.36, 0.42, 0.15), 0.016, loc=(cx, -0.01, z0 - 0.07)), tone=dr.u(0.1, 0.45))
                for s in (-1, 1):
                    drips.append((Mw, cx + s * w / 2, z0 - 0.15))
            add_("iron", box(tag + "_wearplate", (w - 0.06, 0.26, 0.012), loc=(cx, 0.0, z0 + 0.007), bevel=0.002, segments=1))
            add_("iron", cylinder(tag + "_guardbar", 0.017, w + 0.12, loc=(cx, 0.07, z0 + 1.02), rot=(0, 90, 0), verts=8))
            add_("iron", cylinder(tag + "_guardbar2", 0.013, w + 0.12, loc=(cx, 0.07, z0 + 0.55), rot=(0, 90, 0), verts=6))
            for s in (-1, 1):
                add_("iron", torus(tag + "_pull", 0.055, 0.008, loc=(cx + s * 0.16, rv + 0.014, z0 + 1.2), rot=(80, 0, 0), major_seg=12, minor_seg=5))

    make_door("cart", M_front, xh, 0.0, cart_w, cart_h, cart_r, True)
    for k in range(1, NS - LS):
        make_door("load%d" % k, M_front, xh, F[k], ld_w, ld_h, ld_r, False)

    def hook_set(tag, M, x, y, zt, drop, chain_parts=None):
        """Chain with a ball weight and a hook hanging from (x, y, zt)."""
        zh = zt - drop
        objs = [_chain(tag + "_chain", (x, y, zt), (x, y, zh + 0.36)),
                lathe(tag + "_ball", [(0.0, 0.0), (0.03, 0.01), (0.066, 0.06), (0.078, 0.13), (0.066, 0.2), (0.03, 0.25), (0.02, 0.3), (0.0, 0.3)], segments=8,
                      loc=(x, y, zh + 0.08)),
                sweep(tag + "_hook", [(x, y, zh + 0.1), (x, y, zh - 0.02), (x + 0.05, y, zh - 0.11), (x, y, zh - 0.18), (x - 0.06, y, zh - 0.13),
                                      (x - 0.065, y, zh - 0.06)], 0.013, sides=6, scale_fn=lambda t: 1.0 - 0.5 * t ** 3)]
        for o in objs:
            if M is not None:
                apply_transform(o)
                realize(o)
                o.data.transform(M)
            assign(o, iron)
            paint(o, special=0.15 + 0.3 * age)
        return objs

    if cdoors:                                                 # canal side: doors on every upper floor, wall crane, cathead beam
        for k in range(1, NS):
            make_door("canal%d" % k, M_back, ub, F[k], ld_w, ld_h, ld_r, False)
        # folded wall crane on the pilaster beside the lowest door: post in two gudgeon brackets, jib with tie rod and knee
        # brace swung back against the wall, sheave, chain and hook, a hand winch on the post
        uc = ub + B / 2
        zc0, zc1 = F1 + 0.35, F1 + 2.5
        zj, LJ = zc1 - 0.55, 0.85
        Mp = M_back @ Matrix.Translation((uc, -PJ - 0.17, 0.0))
        Mj = Mp @ Matrix.Rotation(radians(198.0), 4, 'Z')
        cw = [cylinder("crane_post", 0.065, zc1 - zc0, loc=(0, 0, (zc0 + zc1) / 2), verts=10),
              lathe("crane_cap", [(0.066, 0.0), (0.08, 0.012), (0.08, 0.04), (0.05, 0.07), (0.05, 0.1), (0.066, 0.14), (0.04, 0.2), (0.0, 0.22)], segments=8,
                    loc=(0, 0, zc1))]
        for zb_ in (zc0 + 0.12, zc1 - 0.12):
            cw.append(box("crane_bracket", (0.17, 0.2, 0.075), loc=(0, 0.085, zb_), bevel=0.005, segments=1))
            cw.append(box("crane_plate", (0.27, 0.014, 0.32), loc=(0, 0.163, zb_), bevel=0.003, segments=1))
            cw.append(cylinder("crane_collar", 0.088, 0.1, loc=(0, 0, zb_), verts=10))
            for bx, bz in ((-0.1, -0.12), (0.1, -0.12), (-0.1, 0.12), (0.1, 0.12)):
                bolt_M.append(Mp @ trs((bx, 0.156, zb_ + bz), (90, 0, 0), scale=1.5))
        zw = zc0 + 0.85                                        # hand winch
        cw += [cylinder("crane_drum", 0.08, 0.24, loc=(0, -0.125, zw), rot=(0, 90, 0), verts=10),
               sweep("crane_crank", [(0.14, -0.125, zw), (0.2, -0.125, zw), (0.2, -0.125, zw - 0.2), (0.31, -0.125, zw - 0.2)], 0.012, sides=5,
                     smooth_path=False),
               cylinder("crane_cog", 0.11, 0.022, loc=(-0.145, -0.125, zw), rot=(0, 90, 0), verts=12)]
        for sx in (-1, 1):
            cw.append(box("crane_cheek", (0.016, 0.2, 0.26), loc=(sx * 0.13, -0.08, zw), bevel=0.003, segments=1))
        cj = [box("crane_jib", (LJ, 0.09, 0.13), loc=(LJ / 2 - 0.02, 0, zj), bevel=0.006, segments=1),
              _rod("crane_tie", (0.03, 0, zc1 - 0.06), (LJ - 0.1, 0, zj + 0.065), 0.04),
              _rod("crane_knee", (0.05, 0, zj - 0.55), (LJ * 0.55, 0, zj - 0.06), 0.034),
              cylinder("crane_sheave", 0.11, 0.045, loc=(LJ - 0.02, 0, zj - 0.1), rot=(90, 0, 0), verts=12)]
        for sy in (-1, 1):
            cj.append(box("crane_strap", (0.075, 0.012, 0.3), loc=(LJ - 0.02, sy * 0.034, zj - 0.05), bevel=0.003, segments=1))
        for o in cw:
            add("iron", o, Mp)
        for o in cj:
            add("iron", o, Mj)
        hook_set("crane", Mj, LJ + 0.09, 0.0, zj - 0.1, 1.15)
        # cathead beam over the top door: timber through the wall, iron strap and stays, a hanging block with its fall
        zc = z_fs + 0.07
        add("wood", plank("cathead", 0.87, 0.15, 0.2, loc=(ub, -0.185, zc), axis='Y', bevel=0.008, rng=dr), M_back)
        add("iron", box("cathead_strap", (0.172, 0.06, 0.222), loc=(ub, -0.5, zc), bevel=0.003, segments=1), M_back)
        for sx in (-1, 1):
            add("iron", _rod("cathead_stay", (ub + sx * 0.085, -0.5, zc), (ub + sx * 0.55, -0.01, zc), 0.014, sides=5), M_back)
            add("iron", box("cathead_stayplate", (0.11, 0.014, 0.11), loc=(ub + sx * 0.55, -0.007, zc), bevel=0.003, segments=1), M_back)
            add("wood", box("cathead_block", (0.016, 0.15, 0.25), loc=(ub + sx * 0.03, -0.5, zc - 0.33), bevel=0.004, segments=1), M_back)
        add("iron", _rod("cathead_eye", (ub, -0.5, zc - 0.1), (ub, -0.5, zc - 0.24), 0.012, sides=5), M_back)
        add("iron", cylinder("cathead_sheave", 0.085, 0.036, loc=(ub, -0.5, zc - 0.33), rot=(0, 90, 0), verts=10), M_back)
        hook_set("cathead", M_back, ub, -0.5, zc - 0.44, 0.75)

    # ------------------------------------------------------------------ painted sign band right round the building, big flaked letters
    band_k, let_k = ("iron", "stone") if dark_sign else ("stone", "iron")
    S1 = _n1(0.5, p.seed + 11, 0.3, 0.75)
    fl_big = noise_mask(0.55, seed=p.seed + 12, lo=0.2, hi=0.9)
    fl_fine = noise_mask(0.16, seed=p.seed + 13, lo=0.2, hi=0.9)
    n_mid = noise_mask(0.6, seed=p.seed + 2, lo=0.4, hi=0.8)
    frac = 0.03 + 0.12 * age
    zones_of = {}
    for wi_, (tag, (M, L, pil, excl, gable)) in enumerate(bands.items()):
        x_lo, x_hi = (-L / 2 - PJ, L / 2 + PJ) if gable else (-L / 2, L / 2)
        zones, cur = [], x_lo
        for (a, b) in pil:
            if a - cur > 0.02:
                zones += [(c, d, 0.0) for c, d in _cut((cur, a), excl)]
            zones += [(c, d, -PJ) for c, d in _cut((a, b), excl)]
            cur = b
        if x_hi - cur > 0.02:
            zones += [(c, d, 0.0) for c, d in _cut((cur, x_hi), excl)]
        zones_of[tag] = zones
        mb = _MB()
        zr = [zb0 + (zb1 - zb0) * j / 3 for j in range(4)]
        for (a, b, y) in zones:
            n = max(1, int(ceil((b - a) / 0.4)))
            for i in range(n):
                xa_, xb_ = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
                for j in range(3):
                    mb.quad([(xa_, y - 0.006, zr[j]), (xb_, y - 0.006, zr[j]), (xb_, y - 0.006, zr[j + 1]), (xa_, y - 0.006, zr[j + 1])])
        for (a, b) in pil:                                    # the band returns round the cheeks of the pilasters
            for xe, sg in ((a, -1), (b, 1)):
                if not gable and abs(abs(xe) - L / 2) < 1e-3:
                    continue
                x = xe + sg * 0.006
                for j in range(3):
                    q = [(x, 0.0, zr[j]), (x, -PJ - 0.006, zr[j]), (x, -PJ - 0.006, zr[j + 1]), (x, 0.0, zr[j + 1])]
                    mb.quad(q if sg < 0 else q[::-1])
        o = mb.obj(tag + "_signband")
        o.data.transform(M)
        uv_box(o)
        assign(o, MATS[band_k])
        if dark_sign:
            paint(o, special=mask_mul(n_mid, 0.25 + 0.3 * age), grime=0.25, wear=mask_mul(fl_fine, 0.3 * age))
        else:
            def band_grime(co, n):
                t = co[:, 0] * np.abs(n[:, 1]) + co[:, 1] * np.abs(n[:, 0])
                top = np.clip((co[:, 2] - zb0) / (zb1 - zb0), 0, 1)
                return np.clip(0.06 + (0.12 + 0.36 * age) * S1(t) * (0.35 + 0.65 * top) + 0.16 * n_mid(co, n) * (0.4 + age), 0, 1)
            paint(o, grime=band_grime)
        # flaked paint: brick by brick the wall shows through band and letters
        cells = []
        for (a, b, y) in zones:
            for c in range(int(floor(zb0 / 0.075)), int(ceil(zb1 / 0.075))):
                za, zc = max(zb0, c * 0.075), min(zb1, (c + 1) * 0.075)
                if zc - za < 0.012:
                    continue
                off = 0.1125 * (c % 2)
                for q in range(int(floor((a - off) / 0.225)), int(ceil((b - off) / 0.225))):
                    xa_, xb_ = max(a, off + q * 0.225), min(b, off + (q + 1) * 0.225)
                    if xb_ - xa_ > 0.012:
                        cells.append((xa_, xb_, za, zc, y, c))
        co = np.array([[(c_[0] + c_[1]) / 2 + 37.0 * wi_, 1.7, (c_[2] + c_[3]) / 2] for c_ in cells], dtype=np.float64)
        v = 0.3 * fl_big(co, None) + 0.7 * fl_fine(co, None) + 0.1 * np.clip(1.0 - (co[:, 2] - zb0) / (zb1 - zb0), 0, 1)
        on_ = v > np.quantile(v, 1.0 - frac)
        fl = _MB()
        i = 0
        while i < len(cells):
            if not on_[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(cells) and on_[j + 1] and cells[j + 1][5] == cells[i][5] and cells[j + 1][4] == cells[i][4] \
                    and abs(cells[j + 1][0] - cells[j][1]) < 1e-6:
                j += 1
            xa_, xb_, za, zc, y = cells[i][0], cells[j][1], cells[i][2], cells[i][3], cells[i][4]
            fl.quad([(xa_, y - 0.014, za), (xb_, y - 0.014, za), (xb_, y - 0.014, zc), (xa_, y - 0.014, zc)])
            i = j + 1
        if fl.v:
            add("brick", fl.obj(tag + "_signflakes"), M, world_uv=True)

    def sign_text(tag, body, x0, x1):
        """Letters of cap height sign_height centred in x0..x1 of a wall, cut and stepped where they cross a pilaster."""
        M = bands[tag][0]
        body = str(body).strip()
        if not body or x1 - x0 < 0.5:
            return
        o = text("sign_src", body, size=1.0, depth=0.01, loc=(0, 0, 0), font=str(p.sign_font))
        apply_transform(o)
        _drop_faces(o, lambda c, n: n.y > -0.5)                # paint has no edges: keep the faces only
        bm = bmesh.new()
        bm.from_mesh(o.data)
        delete([o])
        if len(bm.faces) == 0:
            bm.free()
            return
        xs_, zs_ = [v.co.x for v in bm.verts], [v.co.z for v in bm.verts]
        xa_, xb_, za_, zc_ = min(xs_), max(xs_), min(zs_), max(zs_)
        s = float(p.sign_height) / max(zc_ - za_, 1e-3)
        w0, avail = (xb_ - xa_) * s, x1 - x0
        sx_, sz_ = s, s
        if w0 > avail:
            if w0 * 0.74 <= avail:
                sx_ = s * avail / w0
            else:
                f = avail / (w0 * 0.74)
                sx_, sz_ = s * 0.74 * f, s * f
        wf = (xb_ - xa_) * sx_
        xs0, zm = (x0 + x1) / 2 - wf / 2, (zb0 + zb1) / 2
        for v in bm.verts:
            v.co.x = xs0 + (v.co.x - xa_) * sx_
            v.co.z = zm + (v.co.z - (za_ + zc_) / 2) * sz_
            v.co.y = 0.0
        for (a, b, y) in zones_of[tag]:
            if b <= xs0 or a >= xs0 + wf:
                continue
            bz = bm.copy()
            for pc, pn in (((a, 0, 0), (-1, 0, 0)), ((b, 0, 0), (1, 0, 0))):
                bmesh.ops.bisect_plane(bz, geom=bz.verts[:] + bz.edges[:] + bz.faces[:], dist=1e-5, plane_co=pc, plane_no=pn,
                                       clear_outer=True, clear_inner=False)
            if len(bz.faces) == 0:
                bz.free()
                continue
            for v in bz.verts:
                v.co.y = y - 0.010
            add(let_k, from_bmesh("sign_letters", bz), M)
        bm.free()

    def runs(body, regs):
        """Where the name goes: split either side of the hoist bay, repeated when short, else whole on the longer side."""
        body = str(body).strip()
        words = body.split()
        regs = [r_ for r_ in regs if r_[1] - r_[0] >= 0.5]
        if not regs or not body:
            return []
        if len(regs) == 2:
            l0, l1 = regs[0][1] - regs[0][0], regs[1][1] - regs[1][0]
            if len(body) <= 5 and min(l0, l1) >= 1.6:
                return [(body, regs[0]), (body, regs[1])]
            if min(l0, l1) >= 1.9 and max(l0, l1) / min(l0, l1) < 1.7 and len(words) >= 2:
                best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
                return [(" ".join(words[:best]), regs[0]), (" ".join(words[best:]), regs[1])]
            if l1 >= l0:
                a, b = regs[1]
                return [(body, (a, min(b, a + 9.0)))]
            a, b = regs[0]
            return [(body, (max(a, b - 9.0), b))]
        a, b = regs[0]
        m = (a + b) / 2
        return [(body, (max(a, m - 3.4), min(b, m + 3.4)))]

    for body, (a, b) in runs(p.sign_text, [(-W / 2 + 0.78, xh - B / 2 + PWI / 2 - 0.14), (xh + B / 2 - PWI / 2 + 0.14, W / 2 - 0.78)]):
        sign_text("front", body, a, b)
    regs_b = [(-W / 2 + 0.78, ub - ld_w / 2 - 0.46), (ub + ld_w / 2 + 0.46, W / 2 - 0.78)] if cdoors else [(-W / 2 + 0.78, W / 2 - 0.78)]
    for body, (a, b) in runs(p.sign_text, regs_b):
        sign_text("back", body, a, b)
    for tag in ("left", "right"):                              # trade description along both gable walls
        for body, (a, b) in runs(p.sign_sub, [(-D / 2 + 0.3, D / 2 - 0.3)]):
            sign_text(tag, body, a, b)

    # ------------------------------------------------------------------ lucam (timber hoist housing)
    tr = rng.fork("lucam")
    xl, R = xh, min(float(p.lucam_reach), 1.24)
    wl = min(2.5, B - 0.55)
    z_lf = F[NS - LS]
    z_le = z_e + 0.15
    lp = radians(45.0)
    tl = tan(lp)
    hsl = wl / 2 + 0.2
    z_lr = z_le + wl / 2 * tl
    z_eL = z_lr - hsl * tl
    yf = -R - 0.26
    bd = clamp((age - 0.55) / 0.45, 0.0, 1.0)                 # board damage

    def y_back(dx):
        return max(-OV, (z_lr - abs(dx) * tl - zR0) / tanp)

    yb0 = y_back(0.0)
    # cantilever beams, bressummer, raking struts on stone corbels
    for s in (-1, 1):
        xb_ = xl + s * (wl / 2 - 0.1)
        add("wood", plank("lucam_beam", R + 0.32, 0.2, 0.24, loc=(xb_, -(R + 0.32) / 2 + 0.16, z_lf - 0.12), axis='Y', bevel=0.008, warp=0.004, rng=tr))
        p0, p1 = (xb_, -0.07, z_lf - 1.28), (xb_, -R + 0.16, z_lf - 0.26)
        add("wood", _beam("lucam_strut", p0, p1, 0.16, 0.17, bevel=0.008, rng=tr))
        add("stone", _block("lucam_corbel", (0.3, 0.26, 0.34), 0.03, loc=(xb_, -0.07, z_lf - 1.42)), tone=tr.u(0.1, 0.4))
        add("iron", box("lucam_strap", (0.215, 0.05, 0.3), loc=(xb_, -R + 0.16, z_lf - 0.16), bevel=0.003, segments=1))
        for dz in (-0.1, 0.06):
            for sx in (-1, 1):
                bolt_M.append(trs((xb_ + sx * 0.109, -R + 0.16, z_lf - 0.16 + dz), (0, sx * 90, 0)))
    add("wood", plank("lucam_bressummer", wl + 0.06, 0.17, 0.22, loc=(xl, -R + 0.02, z_lf - 0.11), axis='X', bevel=0.008, rng=tr))
    # floor: boards across, a hoist trap with two drooping flaps in the middle
    st = min(0.8, R - 0.3)
    yc = -R / 2 - 0.03
    nfb = max(3, int(round((R - 0.12) / 0.19)))
    mbf = _MB()

    def fboard(x0, x1, y, bw):
        dz = tr.u(-0.003, 0.003)
        q = [(x0, y - bw / 2, z_lf - 0.035 + dz), (x1, y - bw / 2, z_lf - 0.035 + dz), (x1, y + bw / 2, z_lf - 0.035 + dz), (x0, y + bw / 2, z_lf - 0.035 + dz)]
        mbf.hexa(q[::-1] + [(x, y_, z + 0.03) for x, y_, z in q[::-1]], back=False)
    for i in range(nfb):
        y = -0.03 - (R - 0.12) * (i + 0.5) / nfb
        bw = (R - 0.12) / nfb - 0.006
        if abs(y - yc) < st / 2:
            ln = (wl - 0.4 - st) / 2
            for s in (-1, 1):
                xc_ = xl + s * (st / 2 + ln / 2)
                fboard(xc_ - ln / 2, xc_ + ln / 2, y, bw)
        else:
            fboard(xl - wl / 2 + 0.2, xl + wl / 2 - 0.2, y, bw)
    fl_o = mbf.obj("lucam_floor")
    uv_grain(fl_o, axis='X')
    add("wood", fl_o)
    for s in (-1, 1):
        add("wood", plank("lucam_trimmer", R - 0.1, 0.08, 0.15, loc=(xl + s * (st / 2 + 0.05), -R / 2, z_lf - 0.11), axis='Y', rng=tr))
        dro = tr.u(5, 12)
        fwid = st / 2 - 0.07
        add("wood", plank("lucam_flap", st - 0.02, fwid, 0.03, loc=(xl + s * (st / 2 - fwid / 2 * cos(radians(dro))), yc, z_lf - 0.02 - fwid / 2 * sin(radians(dro))),
                          rot=(0, s * dro, 0), axis='Y', rng=tr))
    # weatherboarding: overlapping feather-edge boards, each one its own slightly irregular piece; with age some have
    # slipped off one nail and a few are gone (the bare lining shows)
    wwl, whl = min(0.8, wl - 1.0), 1.0
    zws = [Ftop + 1.35] + ([z_lf + 1.35] if LS == 2 else [])
    exp_ = 0.16
    mb_f, mb_s, mb_gap = _MB(), _MB(), _MB()

    def wboard(mb, c, e, out, ln, wd=0.19, th=0.021):
        c, e, out = Vector(c), Vector(e), Vector(out)
        ta = radians(6.5 + tr.u(-0.8, 0.8))
        up = Vector((0, 0, 1)) * cos(ta) - out * sin(ta)
        ot = out * cos(ta) + Vector((0, 0, 1)) * sin(ta)
        c = c + Vector((0, 0, tr.u(-0.004, 0.004))) + e * tr.u(-0.006, 0.006)
        if ln > 0.8 and bd > 0:
            u = ar.u()
            if u < 0.1 * bd:                                  # gone
                q = [c - e * ln / 2 - up * wd / 2, c + e * ln / 2 - up * wd / 2, c + e * ln / 2 + up * wd / 2, c - e * ln / 2 + up * wd / 2]
                if e.cross(up).dot(out) < 0:
                    q.reverse()
                mb_gap.quad([v + out * 0.006 for v in q])
                return
            if u < 0.24 * bd:                                 # hanging by one end
                sg = 1 if ar.chance(0.5) else -1
                a = radians(ar.u(4.0, 9.0)) * min(1.0, 1.6 / ln)
                piv = c - e * (sg * ln / 2)
                e2 = (e * cos(a) - Vector((0, 0, 1)) * (sg * sin(a))).normalized()
                c = piv + e2 * (sg * ln / 2) + out * 0.012
                e = e2
        q = [c - e * ln / 2 - up * wd / 2, c + e * ln / 2 - up * wd / 2, c + e * ln / 2 + up * wd / 2, c - e * ln / 2 + up * wd / 2]
        mb.hexa([v + ot * th for v in q] + [v + ot * 0.0 for v in q])

    nbo = int(ceil((z_le - z_lf + 0.02) / exp_))
    for i in range(nbo):
        zc = z_lf - 0.02 + (i + 0.5) * exp_
        if any(zw - 0.1 < zc < zw + whl + 0.1 for zw in zws):
            ln = (wl - wwl) / 2
            for s in (-1, 1):
                wboard(mb_f, (xl + s * (wwl / 2 + ln / 2), -R - 0.012, zc), (1, 0, 0), (0, -1, 0), ln)
        else:
            wboard(mb_f, (xl, -R - 0.012, zc), (1, 0, 0), (0, -1, 0), wl + tr.u(0.0, 0.02))
        for s in (-1, 1):
            wboard(mb_s, (xl + s * (wl / 2 + 0.002), -R / 2, zc), (0, 1, 0), (s, 0, 0), R + 0.02)
    zc = z_lf - 0.02 + (nbo + 0.5) * exp_
    while True:                                               # gable boards
        half = (z_lr - zc - 0.1) / tl
        if half < 0.16:
            break
        wboard(mb_f, (xl, -R - 0.012, zc), (1, 0, 0), (0, -1, 0), 2 * half)
        zc += exp_
    bf, bs = mb_f.obj("lucam_boards_front"), mb_s.obj("lucam_boards_side")
    uv_grain(bf, axis='X')
    uv_grain(bs, axis='Y')
    add("wood", bf)
    add("wood", bs)
    if mb_gap.v:
        o = mb_gap.obj("lucam_gaps")
        uv_grain(o, axis='X')
        realize(o)
        assign(o, wood)
        paint(o, wear=1.0, grime=0.7, mode='set')
    hcb = z_le - z_lf + 0.1
    for s in (-1, 1):                                         # corner boards
        add("wood", plank("lucam_corner_f", hcb, 0.11, 0.026, loc=(xl + s * (wl / 2 - 0.02), -R - 0.05, z_lf - 0.05 + hcb / 2), rot=(0, -90, 0), rng=tr))
        add("wood", plank("lucam_corner_s", hcb, 0.11, 0.026, loc=(xl + s * (wl / 2 + 0.04), -R + 0.0, z_lf - 0.05 + hcb / 2), rot=(0, -90, 90), rng=tr))
    # windows of the hoist loft: the hoist wheel behind the top one, the windlass behind the lower one
    for iw, zw0 in enumerate(zws):
        if p.lit_ratio > 0.02 and (iw == 0 or lr.chance(max(float(p.lit_ratio), 0.6))):
            lit_scene(Matrix.Translation((xl, -R, zw0)), wwl, whl, 0.0, 0.42, lr, 0.3, "wheel" if iw == 0 else "drum")
        else:
            gp = [(xl - wwl / 2 - 0.03, -R - 0.02, zw0 - 0.03), (xl + wwl / 2 + 0.03, -R - 0.02, zw0 - 0.03),
                  (xl + wwl / 2 + 0.03, -R - 0.02, zw0 + whl + 0.03), (xl - wwl / 2 - 0.03, -R - 0.02, zw0 + whl + 0.03)]
            assign(mesh_object("lucam_glass", gp, [(0, 1, 2, 3)]), dark_m)
        for s in (-1, 1):
            add("wood", plank("lucam_wjamb", whl + 0.1, 0.075, 0.05, loc=(xl + s * (wwl / 2 + 0.03), -R - 0.05, zw0 + whl / 2), rot=(0, -90, 0), rng=tr))
        add("wood", plank("lucam_whead", wwl + 0.2, 0.06, 0.07, loc=(xl, -R - 0.055, zw0 + whl + 0.06), rng=tr))
        add("wood", plank("lucam_wsill", wwl + 0.24, 0.1, 0.05, loc=(xl, -R - 0.07, zw0 - 0.05), rot=(-8, 0, 0), rng=tr))
        add("wood", plank("lucam_wmull", whl, 0.03, 0.03, loc=(xl, -R - 0.04, zw0 + whl / 2), rot=(0, -90, 0), rng=tr))
        for fz in (0.34, 0.67):
            add("wood", plank("lucam_wbar", wwl, 0.03, 0.025, loc=(xl, -R - 0.04, zw0 + whl * fz), rng=tr))
    # lining behind the weatherboards: when the separate boards break up in the low LODs the housing stays a closed
    # painted box and the glow of the loft window cannot shine through the gaps
    lin = _PB(pins)
    xo, xi = wl / 2 - 0.03, wwl / 2 + 0.02
    zb_ = z_lf
    for zw0 in sorted(zws) + [None]:
        zt_ = z_le - 0.02 if zw0 is None else zw0 - 0.04
        if zt_ > zb_ + 0.01:
            lin.face([(xl - xo, -R + 0.004, zb_), (xl + xo, -R + 0.004, zb_), (xl + xo, -R + 0.004, zt_), (xl - xo, -R + 0.004, zt_)], (0, -1, 0))
        if zw0 is not None:
            for s in (-1, 1):
                lin.face([(xl + s * xi, -R + 0.004, zt_), (xl + s * xo, -R + 0.004, zt_), (xl + s * xo, -R + 0.004, zw0 + whl + 0.04),
                          (xl + s * xi, -R + 0.004, zw0 + whl + 0.04)], (0, -1, 0))
            zb_ = zw0 + whl + 0.04
    for s in (-1, 1):
        lin.face([(xl + s * (wl / 2 - 0.012), y, z) for y, z in ((-R + 0.004, z_lf), (-0.02, z_lf), (-0.02, z_le - 0.02), (-R + 0.004, z_le - 0.02))], (s, 0, 0))
    add("wood", lin.obj("lucam_lining"))
    for pts, n_, z0w, hh in lit_polys:                         # glow of the loft: brightest at lamp height, dirty
        P = [Vector(q) for q in pts]
        pb = _PB(pins)

        def bl(u, v):
            return (P[0] * (1 - u) + P[1] * u) * (1 - v) + (P[3] * (1 - u) + P[2] * u) * v
        for i in range(3):
            for j in range(3):
                pb.face([bl(i / 3, j / 3), bl((i + 1) / 3, j / 3), bl((i + 1) / 3, (j + 1) / 3), bl(i / 3, (j + 1) / 3)], n_)
        o = pb.obj("lit_glow")
        assign(o, lit_a)
        nzg = noise_mask(0.4, seed=int(lr.u(0, 9999)), lo=0.25, hi=0.8)
        paint(o, grime=lambda co, n, z0w=z0w, hh=hh, nzg=nzg: np.clip(0.05 + 0.5 * np.abs((co[:, 2] - z0w) / hh - 0.62) ** 1.4 + 0.22 * nzg(co, n), 0, 1),
              mode='set')
    if sil.v:
        add("iron", sil.obj("lit_room_gear"))                 # unlit things against the glow: tarred black
    if sil_p.v:
        add("iron", sil_p.obj("lit_room_pins"))
    if sil_f.v:
        assign(sil_f.obj("lit_room_flames"), lit_b)
    # roof of the lucam: boarded slab, bargeboards, finial
    mb = _MB()
    for s in (-1, 1):
        top = [(xl + s * hsl, yf, z_eL - 0.02), (xl + s * hsl, y_back(hsl) + 0.25, z_eL - 0.02), (xl, yb0 + 0.25, z_lr - 0.02), (xl, yf, z_lr - 0.02)]
        mb.hexa(top + [(x, y, z - 0.075) for (x, y, z) in top])
    add("wood", mb.obj("lucam_roof_slab"))
    for s in (-1, 1):
        add("wood", _beam("lucam_barge", (xl + s * (hsl + 0.02), yf - 0.012, z_eL - 0.11), (xl, yf - 0.012, z_lr - 0.09), 0.032, 0.2, bevel=0.005, rng=tr))
    add("wood", lathe("lucam_finial", [(0.0, -0.3), (0.02, -0.29), (0.045, -0.24), (0.03, -0.19), (0.04, -0.16), (0.04, 0.3), (0.055, 0.33), (0.035, 0.37),
                                       (0.03, 0.47), (0.055, 0.53), (0.06, 0.6), (0.035, 0.68), (0.0, 0.76)], segments=8, loc=(xl, yf - 0.03, z_lr - 0.2)))
    # hoist chain through the trap with a ball weight and hook (swings)
    z_hook = max(F1 + 1.15, z_lf - 2.4)
    part(hook_set("hoist", None, xl, yc, z_lf + 0.3, z_lf + 0.3 - z_hook), "hoist", pivot=(xl, yc, z_lf), motion="swing", amp=1.2, speed=0.25)

    # ------------------------------------------------------------------ roof: boarded slab, slates one by one, ridge
    xa, xb = -W / 2 + T, W / 2 - T
    SAG = 0.26 * clamp((age - 0.25) / 0.75, 0.0, 1.0) ** 1.3

    def sag(x, v):
        """Old roofs sag between the trusses; stiff at the gables and where the lucam roof ties in."""
        if SAG <= 0:
            return 0.0
        u = clamp((x - xa) / (xb - xa), 0.0, 1.0)
        m = clamp((abs(x - xl) - hsl - 0.1) / 1.5, 0.0, 1.0)
        m = m * m * (3 - 2 * m)
        return -(SAG * sin(pi * u) + 0.25 * SAG * abs(sin(4 * pi * u))) * clamp(v, 0.0, 1.0) * m

    nsx = max(2, int(ceil((xb - xa + 0.06) / 0.5)))
    secs = []
    for i in range(nsx + 1):
        x = xa - 0.03 + (xb - xa + 0.06) * i / nsx
        sg = sag(x, 1.0)
        secs.append([(x, -OV, z_edge - 0.02), (x, D / 2, z_r - 0.02 + sg), (x, D + OV, z_edge - 0.02), (x, D + OV, z_edge - 0.02 - TV),
                     (x, D / 2, z_r - 0.02 - TV + sg), (x, -OV, z_edge - 0.02 - TV)])
    slab = _loft("roof_slab", secs)
    uv_grain(slab, axis='X')
    realize(slab)
    assign(slab, wood)
    paint(slab, wear=1.0, grime=0.75, mode='set')              # bare sarking boards: dark where slates are missing
    ch_x, ch_y, ch_w, ch_d = xb - 0.85, D / 2 + 1.15, 0.95, 0.6
    SW, SG, STH = 0.32 * (1 + 0.22 * lod), 0.26 * (1 + 0.22 * lod), 0.016
    s_base = mesh_object("slate", [(-SW / 2 + 0.003, 0, 0), (SW / 2 - 0.003, 0, 0), (SW / 2 - 0.003, 2 * SG, 0), (-SW / 2 + 0.003, 2 * SG, 0),
                                   (-SW / 2 + 0.003, 0, STH), (SW / 2 - 0.003, 0, STH), (SW / 2 - 0.003, 2 * SG, STH), (-SW / 2 + 0.003, 2 * SG, STH)],
                         [(4, 5, 6, 7), (0, 1, 5, 4)])
    assign(s_base, slate)
    sr = rng.fork("slates")
    # tone classes: (share, grime, wear, lichen) - weathered dark ones, pale ones, a few new replacements, lichened ones
    CLS = [(0.56, 0.0, 0.0, 0.0), (0.17, 0.3, 0.0, 0.0), (0.08, 0.55, 0.0, 0.1), (0.1, 0.0, 0.16, 0.0), (0.02 + 0.02 * age, 0.0, 0.42, 0.0),
           (0.03 + 0.04 * age, 0.15, 0.0, 0.45)]
    cum, acc = [], 0.0
    for c_ in CLS:
        acc += c_[0]
        cum.append(acc)
    s_M = [[] for _ in CLS]
    tilt = atan(2 * STH / (2 * SG))
    holes = []
    for _ in range(int(age ** 2.5 * 8 + 0.3)):                  # patches of missing slates
        back_ = sr.chance(0.5)
        hx_, hv = sr.u(xa + 0.7, xb - 0.7), sr.u(0.15, 0.88)
        hy = (D + OV - hv * (D / 2 + OV)) if back_ else (-OV + hv * (D / 2 + OV))
        holes.append((Vector((hx_, hy, z_edge + hv * (z_r - z_edge))), sr.u(0.22, 0.3 + 0.38 * age)))

    def lay(P0, e, s, n, wid, slen, keep, main=False):
        P0, e, s, n = Vector(P0), Vector(e), Vector(s), Vector(n)
        nc = int(slen / SG) + 1
        na = int(wid / SW)
        off0 = (wid - na * SW) / 2
        for j in range(nc):
            sj = j * SG - 0.04
            ky = 1.0
            if sj + 2 * SG > slen + 0.02:
                ky = (slen + 0.02 - sj) / (2 * SG)
                if ky < 0.25:
                    continue
            for i in range(na + (j % 2)):
                a = off0 + (i + 0.5 - 0.5 * (j % 2)) * SW
                kx = 1.0
                if a < SW * 0.4 or a > wid - SW * 0.4:        # half slates at the verges
                    kx = 0.5
                    a = a + SW * 0.25 if a < wid / 2 else a - SW * 0.25
                c = P0 + e * a + s * (sj + SG * 0.5)
                if not keep(c):
                    continue
                u, u2, u3 = sr.u(), sr.u(), sr.u()
                near = 9.0
                if main:
                    for hc, hr in holes:
                        near = min(near, (c - hc).length / hr)
                    if near < 0.75 + 0.5 * u2 or u < 0.015 * age:
                        continue                               # missing slate
                slip = 0.0
                if main and (u > 1.0 - 0.05 * age or (near < 1.9 and u3 < 0.3)):
                    slip = sr.u(0.06, 0.2)
                P = P0 + e * a + s * (sj - slip) + n * (2 * STH + sr.u(-0.003, 0.004))
                if main:
                    P.z += sag(c.x, (c.z - z_edge) / (z_r - z_edge))
                Mi = _basis(e, s, n, P) @ Matrix.Rotation(-tilt + radians(sr.u(-0.6, 0.6)), 4, 'X') \
                    @ Matrix.Rotation(radians(sr.u(-1.2, 1.2) + (sr.u(-9.0, 9.0) if slip else 0.0)), 4, 'Z') \
                    @ Matrix.Diagonal((kx * sr.u(0.97, 1.0), ky, sr.u(0.8, 1.25), 1.0))
                uc_ = sr.u()
                s_M[next((q for q, cv in enumerate(cum) if uc_ < cv), 0)].append(Mi)

    def keep_main(c):
        dx = abs(c.x - xl)
        if c.y < D / 2 and dx < hsl + 0.03 and c.z < z_lr - dx * tl + 0.02:
            return False
        if p.chimney and abs(c.x - ch_x) < ch_w / 2 + 0.03 and abs(c.y - ch_y) < ch_d / 2 + 0.16:
            return False
        return True

    Ls = (D / 2 + OV) / cosp
    lay((xa, -OV, z_edge), (1, 0, 0), (0, cosp, sinp), (0, -sinp, cosp), xb - xa, Ls, keep_main, True)
    lay((xb, D + OV, z_edge), (-1, 0, 0), (0, -cosp, sinp), (0, sinp, cosp), xb - xa, Ls, keep_main, True)
    cl, sl = cos(lp), sin(lp)

    def keep_lucam(c):
        return c.z > zR0 + max(c.y, -OV) * tanp + 0.01 or c.y < -OV
    lay((xl + hsl, yf, z_eL), (0, 1, 0), (-cl, 0, sl), (sl, 0, cl), yb0 - yf, hsl / cl, keep_lucam)
    lay((xl - hsl, yb0, z_eL), (0, -1, 0), (cl, 0, sl), (-sl, 0, cl), yb0 - yf, hsl / cl, keep_lucam)
    for q, c_ in enumerate(CLS):
        if not s_M[q]:
            continue
        b = mesh_object("slate%d" % q, [tuple(v.co) for v in s_base.data.vertices], [tuple(f.vertices) for f in s_base.data.polygons])
        assign(b, slate)
        paint(b, grime=c_[1], wear=c_[2], special=c_[3], mode='set')
        G["slate"].append(instances_to_mesh("slates%d" % q, b, s_M[q], keep_base=False, seed=p.seed + q))
    delete([s_base])
    # ridge tiles (half-round, red clay) on the main ridge and the lucam ridge
    rt = lathe("ridge_tile", [(0.118, 0.0), (0.132, 0.0), (0.132, 0.4), (0.146, 0.4), (0.146, 0.46), (0.118, 0.46)], segments=6, angle=180.0)
    assign(rt, terra)
    r_M = []
    nrt = int(ceil((xb - xa) / 0.45))
    for i in range(nrt):
        x = xa + (xb - xa) * i / nrt
        if 1 < i < nrt - 2 and ar.chance(0.1 * clamp((age - 0.5) * 2, 0.0, 1.0)):
            continue                                           # a ridge tile has gone
        dx_ = (xb - xa) / nrt
        slope = (sag(x + dx_, 1.0) - sag(x, 1.0)) / dx_
        r_M.append(Matrix.Translation((x, D / 2, z_r - 0.075 + sag(x, 1.0) + sr.u(-0.005, 0.005))) @ Matrix.Rotation(-atan(slope) + radians(sr.u(-1.5, 1.5)), 4, 'Y')
                   @ Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))) @ Matrix.Diagonal((1, 1, dx_ / 0.43, 1)))
    nlt = int(ceil((yb0 + 0.12 - yf) / 0.45))
    for i in range(nlt):
        y = yf + (yb0 + 0.12 - yf) * i / nlt
        r_M.append(Matrix.Translation((xl, y, z_lr - 0.07)) @ Matrix(((-1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
                   @ Matrix.Diagonal((0.85, 0.85, (yb0 + 0.12 - yf) / nlt / 0.43, 1)))
    G["terra"].append(instances_to_mesh("ridge_tiles", rt, r_M, keep_base=False))
    # lead: valleys of the lucam, abutment flashings against the parapets
    for s in (-1, 1):
        add("lead", _beam("valley", (xl + s * hsl, y_back(hsl) - 0.05, z_eL + 0.05), (xl, yb0 + 0.06, z_lr + 0.05), 0.3, 0.012, bevel=0.003))
    for (ya, yb_) in ((-OV, D / 2), (D + OV, D / 2)):
        for xs, s in ((xa, 1), (xb, -1)):
            add("lead", _beam("flash", (xs + s * 0.07, ya, z_edge + 0.075), (xs + s * 0.07, yb_, z_r + 0.075), 0.17, 0.012, bevel=0.003))
            add("lead", _beam("flash_up", (xs + s * 0.006, ya + (0.2 if ya < 0 else -0.2), z_edge + 0.14 + 0.2 * tanp),
                              (xs + s * 0.006, yb_, z_r + 0.14), 0.012, 0.17, bevel=0.003))

    # ------------------------------------------------------------------ parapet gables: kneelers, copings, apex stones
    cr = rng.fork("coping")
    SWAP = Matrix(((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

    def zt(du):                                               # top line of kneeler / coping, du = distance outside the wall corner
        return z_e + 0.60 - du * tanp

    for M in (M_left, M_right):
        for s in (-1, 1):
            kp = [(-0.34, z_e - 0.2), (0.07, z_e - 0.2), (0.11, z_e - 0.13), (0.19, z_e - 0.08), (0.19, z_e - 0.01), (OV + 0.13, z_e + 0.05),
                  (OV + 0.13, zt(OV + 0.13)), (-0.34, zt(-0.34))]
            o = _prism("kneeler", [(D / 2 + du, z) for du, z in kp], -0.055, T + 0.055)
            o.data.transform(SWAP)
            if s < 0:
                o.data.transform(Matrix.Diagonal((-1, 1, 1, 1)))
            recalc_normals(o)
            bevel(o, 0.014, segments=1)
            add("stone", o, M, tone=cr.u(0.1, 0.5))
            add("stone", _block("kneeler_corbel", (0.34, T * 0.5 + 0.1, 0.17), 0.025, loc=(s * (D / 2 - 0.02), T * 0.25 - 0.02, z_e - 0.28)), M,
                tone=cr.u(0.1, 0.5))
            u0, u1 = D / 2 - 0.335, 0.2
            slen = (u0 - u1) / cosp
            nco = max(3, int(round(slen / 0.8)))
            cuts = [0.0] + [(i + cr.u(-0.15, 0.15)) / nco for i in range(1, nco)] + [1.0]
            for i in range(nco):
                f0, f1 = cuts[i], cuts[i + 1]
                um = u0 + (u1 - u0) * (f0 + f1) / 2
                zm = zR0 + HP + (D / 2 - um) * tanp + 0.035 / cosp
                add("stone", _block("coping", (slen * (f1 - f0) - 0.012 + 0.03, T + 0.12 + cr.u(-0.012, 0.012), 0.1), 0.018,
                                    loc=(s * um, T / 2 + cr.u(-0.006, 0.006), zm + cr.u(-0.006, 0.006)),
                                    rot=(0, s * degrees(pitch) + cr.u(-0.5, 0.5), 0), drop_back=False, drop_bottom=True), M, tone=cr.u(0.0, 0.55))
        add("stone", _block("apex", (0.52, T + 0.16, 0.42), 0.035, loc=(0, T / 2, z_apex + 0.0), drop_back=False, drop_bottom=True), M, tone=cr.u(0.1, 0.4))
        add("stone", lathe("apex_ball", [(0.0, 0.0), (0.11, 0.0), (0.12, 0.03), (0.07, 0.06), (0.06, 0.1), (0.1, 0.14), (0.135, 0.22), (0.12, 0.31),
                                         (0.07, 0.37), (0.0, 0.39)], segments=10, loc=(0, T / 2, z_apex + 0.2)), M, tone=cr.u(0.1, 0.4))

    # ------------------------------------------------------------------ gutters, outlets and downpipes
    gy, gz, gr = -OV - 0.05, z_edge - 0.045, 0.068
    gprof = [(gy + gr * cos(radians(a)), gz + gr * sin(radians(a))) for a in range(180, 361, 36)] + \
            [(gy + (gr - 0.007) * cos(radians(a)), gz + (gr - 0.007) * sin(radians(a))) for a in range(360, 179, -36)]
    z_h = z_fs - 0.16                                          # rim of the hopper heads
    yp = -PJ - 0.078                                           # axis of the downpipes, clear of the pier face
    col = join([lathe("pipe_collar", [(0.046, -0.055), (0.059, -0.05), (0.061, 0.03), (0.054, 0.045), (0.046, 0.05)], segments=8),
                box("pipe_ear", (0.23, 0.014, 0.05), loc=(0, 0.066, 0))], "pipe_collar")
    assign(col, iron)
    col_M = []

    def downpipe(M, gx):
        mb = _MB()
        zt_, zb_ = z_h, z_h - 0.27
        top = [(gx - 0.16, yp - 0.125), (gx + 0.16, yp - 0.125), (gx + 0.16, -PJ - 0.004), (gx - 0.16, -PJ - 0.004)]
        bot = [(gx - 0.07, yp - 0.062), (gx + 0.07, yp - 0.062), (gx + 0.07, yp + 0.062), (gx - 0.07, yp + 0.062)]
        mb.hexa([(top[0][0], top[0][1], zt_), (top[1][0], top[1][1], zt_), (bot[1][0], bot[1][1], zb_), (bot[0][0], bot[0][1], zb_),
                 (top[3][0], top[3][1], zt_), (top[2][0], top[2][1], zt_), (bot[2][0], bot[2][1], zb_), (bot[3][0], bot[3][1], zb_)])
        add("iron", mb.obj("hopper"), M)
        add("iron", box("hopper_rim", (0.35, 0.155, 0.045), loc=(gx, (yp - 0.125 - PJ - 0.004) / 2 - 0.004, z_h), bevel=0.008, segments=1), M)
        add("iron", box("hopper_band", (0.25, 0.11, 0.022), loc=(gx, yp - 0.03, z_h - 0.15), bevel=0.005, segments=1), M)
        add("iron", cylinder("downpipe", 0.045, zb_ - 0.33 + 0.04, loc=(gx, yp, (zb_ + 0.04 + 0.33) / 2), verts=8, cap=False), M)
        add("iron", sweep("pipe_shoe", [(gx, yp, 0.36), (gx, yp, 0.26), (gx, yp - 0.04, 0.17), (gx, yp - 0.13, 0.1)], 0.045, sides=6, step=0.07), M)
        nj = max(2, int(round((zb_ - 0.4) / 1.83)))
        for i in range(nj + 1):
            col_M.append(M @ trs((gx, yp, 0.42 + (zb_ - 0.12 - 0.42) * i / nj)))
        add("stone", _block("gully", (0.4, 0.34, 0.08), 0.015, loc=(gx, yp - 0.1, 0.04), drop_back=False, drop_bottom=True), M, tone=0.4)
        add("iron", box("gully_grate", (0.22, 0.16, 0.012), loc=(gx, yp - 0.13, 0.083), bevel=0.003, segments=1), M)

    brk_M = []
    brk = box("gutter_bracket", (0.03, 2 * gr + 0.07, 0.014))
    assign(brk, iron)
    for M, gxs in ((M_front, [g for (m_, g) in pipes if m_ is M_front]), (M_back, [g for (m_, g) in pipes if m_ is M_back])):
        segs = [(xa + 0.06, xl - wl / 2 - 0.035), (xl + wl / 2 + 0.035, xb - 0.06)] if M is M_front else [(xa + 0.06, xb - 0.06)]
        for (g0, g1) in segs:
            add("iron", _prism("gutter", gprof, g0, g1), M)
            nbk = max(2, int((g1 - g0) / 0.9))
            for i in range(nbk + 1):
                brk_M.append(M @ trs((g0 + 0.1 + (g1 - g0 - 0.2) * i / nbk, gy + 0.03, gz - gr - 0.006)))
        for gx in gxs:
            sx = 1 if gx < 0 else -1
            x0 = gx + sx * 0.14
            add("iron", sweep("gutter_outlet", [(x0, gy, gz - gr + 0.01), (x0, gy, gz - gr - 0.12), (gx + sx * 0.05, -PJ - 0.2, (gz + z_h) / 2 + 0.1),
                                                (gx, yp - 0.03, z_h + 0.2), (gx, yp - 0.03, z_h - 0.03)], 0.04, sides=6, step=0.07), M)
            downpipe(M, gx)
    G["iron"].append(instances_to_mesh("pipe_collars", col, col_M, keep_base=False))
    G["iron"].append(instances_to_mesh("gutter_brackets", brk, brk_M, keep_base=False))

    # ------------------------------------------------------------------ chimney stack of the counting-house flue (back slope)
    if p.chimney:
        cb = rng.fork("chimney")
        zs = lambda y: zR0 + (D - y) * tanp + 0.05             # top of the slates on the back slope
        y0c, y1c = ch_y - ch_d / 2, ch_y + ch_d / 2
        z_lo, z_top = zs(y1c) - 0.3, z_r + 0.95
        mb = _MB()
        _tube(mb, ch_x - ch_w / 2, ch_x + ch_w / 2, y0c, y1c, z_lo, z_top, max(3, int(ceil((z_top - z_lo) / 0.45))))
        add("brick", mb.obj("chimney_stack"), None, world_uv=True)
        for dz, pj_ in ((-0.62, 0.03), (-0.225, 0.035), (-0.15, 0.07), (-0.075, 0.04)):
            add("brick", _block("chimney_course", (ch_w + 2 * pj_, ch_d + 2 * pj_, 0.075), 0.005, loc=(ch_x, ch_y, z_top + dz + 0.0375), drop_back=False),
                None, world_uv=True)
        add("stone", _block("chimney_cap", (ch_w + 0.1, ch_d + 0.1, 0.075), 0.02, loc=(ch_x, ch_y, z_top + 0.035), drop_back=False), tone=0.45)
        for sx in (-1, 1):
            hp_ = cb.u(0.48, 0.66)
            pot = lathe("chimney_pot", [(0.075, 0.25), (0.075, hp_ - 0.01), (0.1, hp_), (0.118, hp_ - 0.02), (0.118, hp_ - 0.07), (0.098, hp_ - 0.1),
                                        (0.09, 0.1), (0.125, 0.06), (0.13, 0.0)], segments=10,
                        loc=(ch_x + sx * ch_w * 0.24, ch_y + cb.u(-0.02, 0.02), z_top + 0.05), rot=(cb.u(-2, 2), cb.u(-2, 2), 0))
            add("terra", pot)
        for sx in (-1, 1):                                     # lead: side soakers, apron and back gutter
            xf = ch_x + sx * (ch_w / 2 + 0.007)
            add("lead", _beam("chimney_flash", (xf, y0c - 0.04, zs(y0c - 0.04) + 0.06), (xf, y1c + 0.04, zs(y1c + 0.04) + 0.06), 0.012, 0.19, bevel=0.003))
        add("lead", box("chimney_apron", (ch_w + 0.16, 0.012, 0.2), loc=(ch_x, y1c + 0.007, zs(y1c) + 0.07), bevel=0.003, segments=1))
        add("lead", box("chimney_apron2", (ch_w + 0.16, 0.2, 0.012), loc=(ch_x, y1c + 0.1 * cosp, zs(y1c) + 0.02 - 0.1 * sinp), rot=(-float(p.pitch), 0, 0),
                        bevel=0.003, segments=1))
        add("lead", box("chimney_backgutter", (ch_w + 0.16, 0.012, 0.16), loc=(ch_x, y0c - 0.007, zs(y0c) + 0.05), bevel=0.003, segments=1))

    # ------------------------------------------------------------------ wall lanterns (six lights): cart door, pilaster, under
    # the lucam, both gables, canal wall. Square tapered gas lanterns on forged brackets.
    lamp_pos = []

    def lantern(tag, M, style, cdl, reach, s, lname):
        ir = []
        if style == "scroll":                                 # standing on a bracket arm with a brace and a scroll
            hb = (0.0, -reach, 0.118)
            ir.append(box(tag + "_plate", (0.085, 0.012, 0.5), loc=(0, -0.006, -0.08), bevel=0.003, segments=1))
            ir.append(box(tag + "_arm", (0.014, reach + 0.07, 0.036), loc=(0, -(reach + 0.07) / 2, 0.1), bevel=0.003, segments=1))
            ir.append(sweep(tag + "_brace", bezier_path((0, -0.012, -0.29), (0, -reach * 0.3, -0.26), (0, -reach * 0.8, -0.06), (0, -reach * 0.88, 0.084), n=8),
                            0.011, sides=5, smooth_path=False))
            ir.append(sweep(tag + "_scroll", spiral_path(0.1, 0.028, turns=1.3, n=14, a0_deg=90.0, plane="YZ", center=(0, -0.125, -0.018)), 0.008,
                            sides=4, smooth_path=False))
            ir.append(lathe(tag + "_tip", [(0.0, -0.05), (0.02, -0.03), (0.012, -0.01), (0.012, 0.0)], segments=6, loc=(0, -reach - 0.05, 0.1), rot=(90, 0, 0)))
        elif style == "swan":                                 # hanging from a swan-neck
            hb = (0.0, -reach, 0.555 - 0.5 * s)
            ir.append(box(tag + "_plate", (0.08, 0.012, 0.42), loc=(0, -0.006, 0.1), bevel=0.003, segments=1))
            ir.append(sweep(tag + "_neck", bezier_path((0, -0.012, -0.02), (0, -reach * 0.3, 0.62), (0, -reach * 1.05, 0.95), (0, -reach, 0.6), n=12),
                            0.0135, sides=6, smooth_path=False))
            ir.append(sweep(tag + "_scroll", spiral_path(0.085, 0.025, turns=1.2, n=12, a0_deg=180.0, plane="YZ", center=(0, -0.1, 0.2)), 0.007,
                            sides=4, smooth_path=False))
            ir.append(torus(tag + "_ring", 0.02, 0.005, loc=(0, -reach, 0.578), rot=(0, 90, 0), major_seg=8, minor_seg=4))
        else:                                                 # hanging under a timber on a flat arm
            hb = (0.0, -reach, -0.075 - 0.5 * s)
            ir.append(box(tag + "_arm", (0.036, reach + 0.16, 0.012), loc=(0, (0.14 - reach) / 2 - 0.01, -0.006), bevel=0.003, segments=1))
            ir.append(_rod(tag + "_stay", (0, -0.004, 0.14), (0, -reach * 0.85, 0.0), 0.007, sides=4))
            ir.append(_rod(tag + "_drop", (0, -reach, -0.004), (0, -reach, -0.08), 0.007, sides=4))
        for o in ir:
            add("iron", o, M)
        Mh = M @ Matrix.Translation(hb) @ Matrix.Rotation(pi / 4, 4, 'Z') @ Matrix.Diagonal((s, s, s, 1.0))
        hd = [lathe(tag + "_pan", [(0.0, 0.0), (0.04, 0.0), (0.082, 0.036), (0.086, 0.05), (0.0, 0.05)], segments=4, smooth_angle=30),
              lathe(tag + "_roof", [(0.0, 0.31), (0.136, 0.3), (0.152, 0.3), (0.152, 0.318), (0.062, 0.4), (0.04, 0.41), (0.04, 0.44), (0.056, 0.452),
                                    (0.03, 0.485), (0.0, 0.5)], segments=4, smooth_angle=30),
              cylinder(tag + "_burner", 0.02, 0.06, loc=(0, 0, 0.08), verts=6)]
        for a in range(4):
            c_, s_ = cos(a * pi / 2), sin(a * pi / 2)
            hd.append(_rod(tag + "_bar", (0.08 * c_, 0.08 * s_, 0.05), (0.131 * c_, 0.131 * s_, 0.3), 0.006, sides=4))
        for o in hd:
            add("iron", o, Mh)
        gl = lathe(tag + "_glass", [(0.08, 0.05), (0.131, 0.3)], segments=4, smooth_angle=30)
        fm = lathe(tag + "_flame", [(0.0, 0.112), (0.016, 0.13), (0.02, 0.15), (0.01, 0.185), (0.0, 0.2)], segments=6)
        for o, m_ in ((gl, glass_m), (fm, lit_b)):
            realize(o)
            apply_transform(o)
            o.data.transform(Mh)
            assign(o, m_)
        pos = Mh @ Vector((0, 0, 0.15))
        light(lname, tuple(pos), color="#ffbf6a", intensity=cdl, range=9.5, flicker=0.12, shadow_near=0.26 * s)
        lamp_pos.append((pos.x, pos.y, pos.z))

    xs2 = xh + (B / 2 if xh <= 0 else -B / 2)
    Bs = D / ns
    lantern("lamp_cart", M_front @ Matrix.Translation((xh, -0.10, 3.3)), "scroll", cd, 0.8, 1.12, "lamp_cart")
    lantern("lamp_pier", M_front @ Matrix.Translation((xs2, -PJ, z_lf - 1.5)), "swan", cd, 0.5, 1.0, "lamp_pier")
    lantern("lamp_lucam", M_front @ Matrix.Translation((xl, -R - 0.065, z_lf - 0.22)), "hang", 1.25 * cd, clamp(1.38 - R, 0.1, 0.3), 1.05, "lamp_lucam")
    lantern("lamp_side_l", M_left @ Matrix.Translation((-D / 2 + (ns - ns // 2) * Bs, -PJ, F[kb] - 1.45)), "swan", 0.85 * cd, 0.46, 1.0, "lamp_side_l")
    lantern("lamp_side_r", M_right @ Matrix.Translation((-D / 2 + (ns // 2) * Bs, -PJ, F[kb] - 1.45)), "swan", 0.85 * cd, 0.46, 1.0, "lamp_side_r")
    lantern("lamp_canal", M_back @ Matrix.Translation((ub - B / 2, -PJ, F[1] + 1.75)), "swan", 0.85 * cd, 0.46, 1.0, "lamp_canal")
    G["iron"].append(instances_to_mesh("studs", bolt_b, bolt_M, keep_base=False))

    # ------------------------------------------------------------------ art-directed weathering
    n_big = noise_mask(2.2, seed=p.seed + 1, lo=0.35, hi=0.8)
    n_sml = noise_mask(0.18, seed=p.seed + 3, lo=0.4, hi=0.75)
    n_blot = noise_mask(0.9, seed=p.seed + 4, lo=0.42, hi=0.75)
    n_grn = noise_mask(1.3, seed=p.seed + 5, lo=0.4, hi=0.75)
    S_a = _n1(0.42, p.seed + 21, 0.25, 0.8)
    S_b = _n1(1.5, p.seed + 22, 0.2, 0.8)
    drip_w = []
    for (M, x, z) in drips:
        P = M @ Vector((x, 0.0, z))
        nr = M.to_3x3() @ Vector((0, -1, 0))
        drip_w.append((P.x, P.y, P.z, abs(nr.y) > 0.5, ar.u(1.1, 1.9), ar.u(0.3, 0.75)))

    def brick_grime(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        t = np.where(np.abs(n[:, 1]) >= np.abs(n[:, 0]), x, y + 51.0)      # coordinate along the wall: stains run straight down
        g = np.clip(1.0 - z / 1.5, 0, 1) ** 1.5 * 0.75                     # splash-back from the paving
        st_ = 0.15 + 0.6 * S_a(t) * (0.4 + 0.6 * S_b(t))
        for zs_ in str_z:                                                  # run-off below the stone strings
            d = zs_ - z
            g = np.maximum(g, (d > -0.02) * np.clip(1.0 - d / 1.6, 0, 1) ** 1.3 * st_)
        g = np.maximum(g, np.clip((z - (z_e - 0.5)) / 0.5, 0, 1) * (z < z_e + 0.4) * 0.45)   # soot under the eaves
        for (px, py, pz, fb, fall, amp) in drip_w:                         # dark drips below the ends of the sills
            dt, dpl = (x - px, np.abs(y - py)) if fb else (y - py, np.abs(x - px))
            d = pz - z
            g = np.maximum(g, (dpl < 0.3) * (d > -0.05) * np.exp(-(dt / 0.2) ** 2) * np.clip(1.0 - d / fall, 0, 1) * amp)
        for (lx, ly, lz) in lamp_pos:                                      # lamp black above the lanterns
            d = z - lz
            g = np.maximum(g, (d > -0.1) * np.exp(-((x - lx) ** 2 + (y - ly) ** 2) / 0.62) * np.clip(1.0 - d / 1.9, 0, 1) * 0.6)
        g = np.maximum(g, n_big(co, n) * 0.5)
        g = np.maximum(g, np.clip((z - (z_r + 0.25)) / 0.7, 0, 1) * 0.9)   # soot at the chimney head
        return np.clip(g * (0.45 + 0.8 * age), 0, 1)

    def brick_special(co, n):                                                           # damp on the canal (back) side, low down
        back = np.clip(n[:, 1], 0, 1)
        return np.clip(np.clip(1.0 - co[:, 2] / 2.6, 0, 1) ** 1.5 * (0.15 + 0.6 * back) * n_mid(co, n) * (0.4 + 0.8 * age), 0, 1)

    def stone_grime(co, n):
        hi_ = np.clip((co[:, 2] - (z_e - 0.3)) / 0.3, 0, 1)                # copings, kneelers, apex stones: black weathering
        lb = np.zeros(len(co))
        for (lx, ly, lz) in lamp_pos:                                      # lamp black on the stones next to a lantern
            lb = np.maximum(lb, np.exp(-((co[:, 0] - lx) ** 2 + (co[:, 1] - ly) ** 2 + (co[:, 2] - lz) ** 2) / 1.0))
        return np.clip((np.clip(1.0 - co[:, 2] / 0.8, 0, 1) * 0.7 + np.clip(-n[:, 2], 0, 1) * 0.45 + n_mid(co, n) * 0.3
                        + hi_ * n_blot(co, n) * 0.6) * (0.5 + 0.7 * age) + 0.65 * lb, 0, 1)

    def stone_moss(co, n):
        hi_ = np.clip((co[:, 2] - (z_e - 0.3)) / 0.3, 0, 1)
        return np.clip(np.clip(n[:, 2], 0, 1) * n_mid(co, n) * (0.3 + 0.9 * age) + np.clip(n[:, 1], 0, 1) * np.clip(1.0 - co[:, 2] / 2.0, 0, 1) * 0.5 * age
                       + hi_ * n_grn(co, n) * (0.45 + 0.5 * age), 0, 1)

    def wood_wear(co, n):
        return np.clip(n_sml(co, n) * (0.15 + 0.75 * age) * (0.5 + 0.5 * np.clip(1.0 - (co[:, 2] - 0.0) / 2.5, 0, 1) + 0.4 * n_big(co, n)), 0, 1)

    def wood_grime(co, n):
        return np.clip((np.clip(1.0 - co[:, 2] / 0.9, 0, 1) * 0.7 + n_mid(co, n) * 0.3 + np.clip(-n[:, 2], 0, 1) * 0.3) * (0.4 + 0.7 * age), 0, 1)

    def slate_lichen(co, n):
        return np.clip(n_mid(co, n) * n_big(co, n) * (0.2 + 0.8 * age) * (0.5 + 0.5 * np.clip(n[:, 1], 0, 1)), 0, 1)

    def slate_grime(co, n):
        g = n_big(co, n) * 0.3
        if p.chimney:
            d = np.sqrt((co[:, 0] - ch_x) ** 2 + (co[:, 1] - ch_y) ** 2)
            g = np.maximum(g, np.clip(1.0 - d / 2.2, 0, 1) * 0.6)
        return np.clip(g * (0.5 + 0.7 * age), 0, 1)

    def iron_rust(co, n):
        return np.clip(n_sml(co, n) * (0.15 + 0.6 * age) + np.clip(-n[:, 2], 0, 1) * 0.25 * age, 0, 1)

    for o in G["brick"]:
        paint(o, grime=brick_grime, special=brick_special)
    for o in G["stone"]:
        tn = TONE.get(o.name, 0.0)
        paint(o, grime=lambda co, n, tn=tn: np.clip(stone_grime(co, n) + tn, 0, 1), special=stone_moss, mode='add')
    for o in G["wood"]:
        paint(o, wear=wood_wear, grime=wood_grime, special=mask_mul(n_big, 0.3 + 0.5 * age))
    for o in G["slate"]:
        paint(o, grime=slate_grime, special=slate_lichen, mode='add')
    for o in G["terra"]:
        paint(o, grime=mask_mul(n_mid, 0.3 + 0.5 * age), special=mask_mul(n_big, 0.2 + 0.5 * age))
    for o in G["iron"]:
        paint(o, special=iron_rust, grime=mask_mul(n_mid, 0.3 * age))
    for o in G["lead"]:
        paint(o, grime=mask_mul(n_mid, 0.4))

    # ------------------------------------------------------------------ sockets for the town generator
    xs_sign = xh - B / 2 if xh <= 0 else xh + B / 2
    socket("sign_front", "attach", (xs_sign, -PJ, F1 + 1.5), role="sign")
    socket("lantern_left", "attach", (-W / 2, D * 0.5, 3.3), rot=(0, 0, -90), role="lantern")
    socket("lantern_right", "attach", (W / 2, D * 0.5, 3.3), rot=(0, 0, 90), role="lantern")
    socket("hoist_load", "attach", (xl, yc, z_hook - 0.15), role="prop_top")
