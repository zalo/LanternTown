"""Cast-iron pillar letter box. A dark moulded plinth (bolted ground flange, half-round torus at kick
height, die with the founder's name) carries a round or hexagonal body cast in one piece; the front has
a door set in a rebate inside a raised architrave, hung on three knuckle hinges, with brass lock
escutcheon and ring pull, an enamel collection plate in a brass frame and a cast post horn in a moulded
panel. Above it the letter aperture is cut through the body under an eyebrow hood, with raised lettering
on the frieze. The moulded cap has a beaded fascia and a turned finial (shallow dome, ogee bell or
Penfold-style acanthus leaves). Paint damage is modelled: flakes of bare rusty iron with a pale undercoat
edge sit where a box really gets knocked (door edge, lock, aperture lip, cap rim and beads, plinth torus,
foot of the body) and rust bleeds from the hinges, hood, bosses and base. Optional stone footing.
"""
from tk import *
import bmesh
import numpy as np
from math import sin, cos, tan, atan, pi, radians, degrees, atan2, hypot, ceil

META = {"name": "Pillar post box", "class": "prop", "mount": "ground", "focus": 0.67}
PARAMS = {
    "plan": {"default": "round", "choices": ["round", "hex"], "desc": "body plan: cylindrical or hexagonal (Penfold)"},
    "height": {"default": 1.6, "min": 1.3, "max": 1.9, "desc": "overall height above the paving incl. finial (m)"},
    "width": {"default": 0.45, "min": 0.38, "max": 0.56, "desc": "body diameter / width across flats (m)"},
    "paint": {"default": "#8f1b14", "desc": "body paint colour"},
    "base_paint": {"default": "#3a3d42", "desc": "plinth paint colour"},
    "letter_paint": {"default": "#d2a63c", "desc": "gold paint used to pick out the lettering and emblem (if gilding)"},
    "cap": {"default": "dome", "choices": ["dome", "ogee", "acanthus"], "desc": "cap style"},
    "legend": {"default": "POST OFFICE", "desc": "raised lettering on the frieze above the aperture"},
    "plate_text": {"default": "NEXT COLLECTION|7.30 PM|MON - SAT", "desc": "collection plate text, lines separated by |"},
    "maker": {"default": "ANNO 1887", "desc": "founder's name cast on the plinth ('' = none)"},
    "emblem": {"default": "horn", "choices": ["horn", "none"], "desc": "cast emblem in the door panel"},
    "panels": {"default": True, "desc": "raised panel mouldings and bosses on the sides and back"},
    "gilding": {"default": True, "desc": "lettering and emblem picked out in gold"},
    "footing": {"default": 0.05, "min": 0.0, "max": 0.15, "desc": "height of the stone footing slab (0 = set straight into the paving)"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "paint chips and rust (0 = freshly painted, 1 = neglected)"},
}
VARIANTS = {
    "penfold_green": {"plan": "hex", "width": 0.5, "height": 1.5, "paint": "#35805a", "cap": "acanthus", "letter_paint": "#e2c979",
                      "legend": "POST OFFICE", "plate_text": "LETTERS|CLEARED DAILY|6 AM - 9 PM", "wear": 0.35, "seed": 4},
    "blue_ogee": {"plan": "round", "width": 0.42, "height": 1.85, "paint": "#33609c", "cap": "ogee", "legend": "AIR MAIL",
                  "plate_text": "BY AIRSHIP|TUES & FRI|NOON", "footing": 0.1, "wear": 0.25, "seed": 9},
    "old_rusty": {"plan": "hex", "width": 0.46, "height": 1.38, "paint": "#8a3524", "base_paint": "#4a4038", "cap": "dome", "gilding": False,
                  "legend": "LETTERS", "plate_text": "NO COLLECTION|ON SUNDAYS", "footing": 0.0, "wear": 1.0, "seed": 17},
    "ochre_wide": {"plan": "round", "width": 0.54, "height": 1.45, "paint": "#b07a14", "base_paint": "#3b352e", "cap": "acanthus",
                   "legend": "POSTES", "plate_text": "LEVEE|8 H - 19 H", "emblem": "horn", "panels": False, "wear": 0.6, "seed": 23},
}

STEP = 0.03          # tessellation along the surface for parts wrapped round the body


def _mixhex(h1, h2, t):
    try:
        a = [int(h1[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(h2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(int(round(lerp(x, y, t))) for x, y in zip(a, b))
    except (ValueError, IndexError, TypeError):
        return h2


class _Poly:
    """Profile polyline [(radius, z)] ordered bottom-to-top along the outer surface (as for lathe)."""

    def __init__(self, pts):
        self.p = [(float(r), float(z)) for r, z in pts]
        self.cum = [0.0]
        for k in range(1, len(self.p)):
            self.cum.append(self.cum[-1] + hypot(self.p[k][0] - self.p[k - 1][0], self.p[k][1] - self.p[k - 1][1]))
        self.tot = self.cum[-1]

    def seg(self, q):
        k = 0
        while k < len(self.p) - 2 and q > self.cum[k + 1]:
            k += 1
        return k

    def nrm(self, q):
        k = self.seg(clamp(q, 0.0, self.tot))
        dr, dz = self.p[k + 1][0] - self.p[k][0], self.p[k + 1][1] - self.p[k][1]
        L = hypot(dr, dz) or 1.0
        return dz / L, -dr / L

    def at(self, q, blur=1e-5):
        """(r, z, nr, nz) at arc length q; the normal is averaged over +-blur (rounds the corners)."""
        q = clamp(q, 0.0, self.tot)
        k = self.seg(q)
        f = (q - self.cum[k]) / max(self.cum[k + 1] - self.cum[k], 1e-9)
        a, b = self.nrm(q - blur), self.nrm(q + blur)
        nr, nz = a[0] + b[0], a[1] + b[1]
        L = hypot(nr, nz) or 1.0
        return lerp(self.p[k][0], self.p[k + 1][0], f), lerp(self.p[k][1], self.p[k + 1][1], f), nr / L, nz / L


def _cuts(o, cs, axis=0):
    """Slice the mesh with planes x (axis 0) or z (axis 2) = each value in cs."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    vs = [v.co[axis] for v in bm.verts]
    if vs:
        c0, c1 = min(vs), max(vs)
        no = (1, 0, 0) if axis == 0 else (0, 0, 1)
        for c in sorted(set(round(c, 5) for c in cs)):
            if c0 + 1e-4 < c < c1 - 1e-4:
                geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
                bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=(c, 0, 0) if axis == 0 else (0, 0, c), plane_no=no)
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()


def _steps(c0, c1, step):
    out = []
    c = ceil((c0 + 1e-4) / step) * step
    while c < c1 - 1e-4:
        out.append(c)
        c += step
    return out


def _bounds(o, axis):
    vs = [v.co[axis] for v in o.data.vertices]
    return (min(vs), max(vs)) if vs else (0.0, 0.0)


def _prism(name, prof, x0, x1, nx):
    """Extrude a (d, z) section along x in 'face space' (y = -d)."""
    n = len(prof)
    verts, faces = [], []
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        verts += [(x, -d, z) for (d, z) in prof]
    for i in range(nx):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(range(nx * n, nx * n + n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _rr(w, h, r, seg=3, step=STEP, zcuts=()):
    """Rounded-rect loop [(x, z, nx, nz)], CCW seen from the front; horizontal runs split at k * step,
    vertical runs at the given z values (relative to the centre) so vertex masks have something to hold."""
    hw, hh = w / 2.0, h / 2.0
    r = max(min(r, hw - 1e-4, hh - 1e-4), 1e-4)
    out = []
    zc = sorted(z for z in zcuts if -hh + r + 0.004 < z < hh - r - 0.004)

    def arc(cx, cz, a0):
        for i in range(seg + 1):
            a = radians(a0 + 90.0 * i / seg)
            out.append((cx + r * cos(a), cz + r * sin(a), cos(a), sin(a)))

    def run(xa, xb, z, nz):
        xs = _steps(min(xa, xb), max(xa, xb), step)
        if xb < xa:
            xs.reverse()
        for x in xs:
            out.append((x, z, 0.0, nz))
    arc(hw - r, -hh + r, -90)
    out.extend((hw, z, 1.0, 0.0) for z in zc)
    arc(hw - r, hh - r, 0)
    run(hw - r, -hw + r, hh, 1.0)
    arc(-hw + r, hh - r, 90)
    out.extend((-hw, z, -1.0, 0.0) for z in reversed(zc))
    arc(-hw + r, -hh + r, 180)
    run(-hw + r, hw - r, -hh, -1.0)
    return out


def _frame(name, loop, prof, x0=0.0, z0=0.0, cap=False):
    """Moulding / plate in face space. prof = [(offset along the loop normal, outward depth)] from the
    outer back edge across the face; cap fills the last loop (solid plate)."""
    n, m = len(loop), len(prof)
    verts = [(x0 + x + nx * off, -d, z0 + z + nz * off) for (off, d) in prof for (x, z, nx, nz) in loop]
    faces = []
    for j in range(m - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap:
        faces.append(tuple(range((m - 1) * n, m * n)))
    return mesh_object(name, verts, faces)


def _ring(name, w, h, r, bar, D, x0=0.0, z0=0.0, d0=0.0, step=STEP, seg=3, zcuts=(), plain=False):
    """Raised moulded frame, outer size w x h, bar width `bar`, standing D proud of depth d0."""
    loop = _rr(w - bar, h - bar, max(r - bar / 2, 0.002), seg=seg, step=step, zcuts=zcuts)
    b = bar / 2
    prof = [(b, d0 - 0.0015), (b, d0 + D * 0.55), (b - min(0.004, bar * 0.22), d0 + D), (-b + min(0.006, bar * 0.3), d0 + D),
            (-b, d0 - 0.0015)]
    if plain:
        del prof[1]
    return _frame(name, loop, prof, x0, z0)


def _relief(name, pts, wfn, hfn, y0, na=4):
    """Half-round cast relief along a 2D path [(x, z)] in face space: width wfn(t), standing hfn(t) proud
    of depth y0 (t = 0..1 along the path), closed with flat ends; the foot is sunk 1 mm."""
    P = np.asarray(pts, dtype=float)
    n = len(P)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    m = na + 1
    verts, faces = [], []
    for i in range(n):
        d = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        d = d / (np.linalg.norm(d) or 1.0)
        sx, sz = d[1], -d[0]
        t = cum[i] / cum[-1]
        w, h = wfn(t) / 2.0, hfn(t)
        for k in range(m):
            a = pi * k / na
            foot = k in (0, na)
            verts.append((P[i][0] + sx * w * cos(a), -(y0 - 0.001 if foot else y0 + h * sin(a)), P[i][1] + sz * w * cos(a)))
    for i in range(n - 1):
        for k in range(na):
            faces.append((i * m + k, (i + 1) * m + k, (i + 1) * m + k + 1, i * m + k + 1))
    for i, flip in ((0, False), (n - 1, True)):
        verts.append((P[i][0], -(y0 - 0.001), P[i][1]))
        c = len(verts) - 1
        for k in range(na):
            faces.append((c, i * m + k + 1, i * m + k) if flip else (c, i * m + k, i * m + k + 1))
    return mesh_object(name, verts, faces)


def build(p, rng):
    hexa = p.plan == "hex"
    W = p.width
    R = W / 2.0                                   # radius (round) or apothem (hex)
    ws = W / 0.45
    kf = 1.0 / cos(pi / 6) if hexa else 1.0
    T30 = tan(pi / 6)
    flat = 2 * R * T30                            # width of one hex face
    zf = p.footing if p.footing >= 0.015 else 0.0
    z0 = max(zf - 0.003, 0.0)                     # underside of the ironwork
    Ht = p.height
    wr = p.wear
    NS = 24 if hexa else 32                       # segments of the turned parts

    # scale < 1 shrinks the paint texture: fine orange peel and pin-prick pitting instead of big flecks
    red = mat("iron_painted", p.paint, scale=0.12)
    blk = mat("iron_painted", p.base_paint, rough=0.75, scale=0.4)
    brass = mat("brass")
    gold = mat("iron_painted", p.letter_paint, rough=0.8) if p.gilding else red
    enamel = mat("ceramic_glazed", "#e9e2cf")
    dark = mat("iron_painted", "#16130f", rough=1.4)                              # aperture void, door rebate, plate lettering
    rustm = mat("iron_painted", "#a85a28", rough=1.7, scale=0.3)   # red-oxide primer / rust (its rust mask is painted on):                                                              # bare iron where the paint has flaked off
    edge = mat("iron_painted", _mixhex(p.paint, "#d8c4ae", 0.45), rough=1.5, scale=0.3)      # undercoat showing round a flake
    groups = {"red": [], "blk": [], "brass": [], "gold": [], "enamel": [], "dark": [], "rust": [], "edge": []}
    mats = {"red": red, "blk": blk, "brass": brass, "gold": gold, "enamel": enamel, "dark": dark, "rust": rustm, "edge": edge}

    def add(o, kind):
        assign(o, mats[kind])
        groups[kind].append(o)
        return o

    # ---- helpers: hexagonal lathes and wrapping flat-built parts onto the front face
    def hexk(th):
        return 1.0 / cos((th % (pi / 3)) - pi / 6) if hexa else 1.0

    def hexify(o):
        if not hexa:
            return o

        def f(co):
            if hypot(co.x, co.y) < 1e-6:
                return co
            k = hexk(atan2(co.y, co.x))
            return Vector((co.x * k, co.y * k, co.z))
        transform_verts(o, f)
        bevel(o, 0.003, segments=1, angle=50)
        return o

    def lat(name, prof, seg=NS, sm=50, hx=True):
        o = lathe(name, prof, segments=seg, smooth_angle=sm)
        return hexify(o) if hx else o

    stp = 10.0 if hexa else STEP                 # flat faces need no tessellation along the surface

    def mring(*a, **k):
        return _ring(*a, step=stp, **k)

    def onface(o, rad=None, cut=STEP, xs=(), zs=(), sm=35):
        """Part built in face space (x = along the surface, y = -outward, z = up) -> onto the body front.
        xs / zs: extra slicing planes so vertex masks and AO have resolution on big plates."""
        rr = R if rad is None else rad
        apply_modifiers(o)
        apply_transform(o)
        if zs:
            _cuts(o, zs, 2)
        if hexa:
            if xs:
                _cuts(o, xs, 0)
            transform_verts(o, lambda co: Vector((co.x, co.y - rr, co.z)))
        else:
            if cut or xs:
                _cuts(o, list(xs) + (_steps(*_bounds(o, 0), cut) if cut else []), 0)
            transform_verts(o, lambda co: Vector(((rr - co.y) * sin(co.x / rr), -(rr - co.y) * cos(co.x / rr), co.z)))
            smooth(o, sm)
        return o

    def label(name, body, size, depth, maxw, font="serif", sides=True, weld=0.0):
        """Text centred on (0, 0) in face space, back on y = 0, shrunk to fit maxw. Outlines are thinned
        and the hidden back faces removed."""
        o = text(name, body, size=size, depth=depth, font=font)
        apply_transform(o)
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.normal_update()
        kill = [f for f in bm.faces if f.normal.y > 0.5 or (not sides and f.normal.y > -0.5)]
        bmesh.ops.delete(bm, geom=kill, context='FACES')
        if weld:                        # thin the over-dense curve outlines
            bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=weld)
            bm.normal_update()          # ...then part the faces from the walls again so the arris stays crisp
            hard = [e for e in bm.edges if len(e.link_faces) == 2 and e.link_faces[0].normal.dot(e.link_faces[1].normal) < 0.5]
            bmesh.ops.split_edges(bm, edges=hard)
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
        mn, mx = world_bbox([o])
        w = max(mx.x - mn.x, 1e-6)
        k = min(1.0, maxw / w)
        cx, cz, y0 = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2, mx.y
        transform_verts(o, lambda co: Vector(((co.x - cx) * k, co.y - y0, (co.z - cz) * k)))
        return o

    # ---- vertical layout
    cs = min(ws, 1.15) * clamp((Ht - z0) / 1.5, 0.8, 1.1)       # cap scale
    c = cs
    ov = 0.046 * ws                                             # cap overhang
    hd = {"dome": 0.085, "ogee": 0.25, "acanthus": 0.14}[p.cap] * cs
    fin_h = {"dome": 0.04, "ogee": 0.13, "acanthus": 0.11}[p.cap] * cs
    rim_h = 0.062 * cs
    zc = Ht - fin_h - hd - rim_h                                # top of the body / cap soffit
    hp = z0 + 0.165 * (Ht - z0)                                 # top of the plinth
    kp = clamp((hp - z0) / 0.257, 0.75, 1.1)                    # plinth moulding scale
    fb = 0.02                                                   # door architrave width
    dw = (flat - 2 * fb - 0.016) if hexa else 1.46 * R          # door width (along the surface)
    zl = zc - 0.05 * cs - 0.037                                 # frieze lettering centre
    sh = 0.036                                                  # aperture height
    sw = (dw - 0.04) if hexa else min(0.25, dw - 0.07)          # aperture width
    zs = zl - 0.093                                             # aperture centre
    zd1 = zl - 0.17                                             # door top
    zd0 = hp + 0.045                                            # door bottom
    dh = zd1 - zd0
    zm = (zd0 + zd1) / 2
    dd = 0.006                                                  # door stands this proud of the body
    hw = sw + 0.07                                              # hood width
    zh = zs + sh / 2 + 0.013                                    # hood underside
    xk = -(dw / 2 + 0.0008)                                     # hinge line = middle of the door gap
    nh = 3 if dh > 0.5 else 2
    hz = [zd0 + 0.09 + (dh - 0.18) * i / (nh - 1) for i in range(nh)]
    ph = 0.112 if dh > 0.5 else 0.08                            # collection plate
    pw = min(0.19, dw - 0.075)
    zpl = zd1 - 0.03 - ph / 2
    zpb = zpl - ph / 2
    zlk = zpb - 0.052                                           # lock rail
    xl = dw / 2 - 0.038
    Hr = 0.17 + 0.2 * wr                                        # how far the rust creeps up from the plinth
    if hexa:
        pangs = [60.0 * k for k in range(1, 6)]                 # side / back panels (degrees from the front)
        spw = flat - 0.06
    else:
        fh = (dw + 2 * fb + 0.04) / 2
        pitch = (2 * pi * R - 2 * fh) / 4
        pangs = [degrees((fh + pitch * (k + 0.5)) / R) for k in range(4)]
        spw = pitch - 0.05
    has_panels = p.panels and dh > 0.2

    # rust runs: (angle of the face from the front, x on it, top z, half width, length)
    streaks = [(0.0, xk + 0.021, z - 0.03, 0.010, 0.10 + 0.22 * wr) for z in hz]
    streaks += [(0.0, sx * (hw / 2 - 0.006), zh + 0.002, 0.010, 0.09 + 0.16 * wr) for sx in (-1, 1)]
    streaks += [(0.0, sx * (sw / 2 + 0.012), zs - sh / 2 - 0.02, 0.009, 0.05 + 0.1 * wr) for sx in (-1, 1)]
    streaks += [(0.0, xl, zlk - 0.027, 0.009, 0.08 + 0.15 * wr), (0.0, -pw / 2, zpb - 0.008, 0.008, 0.05 + 0.1 * wr),
                (0.0, pw / 2, zpb - 0.008, 0.008, 0.04 + 0.1 * wr)]
    if has_panels:
        streaks += [(radians(a), 0.0, zs - 0.03, 0.012, 0.12 + 0.22 * wr) for a in pangs]
    rcuts = [hp + Hr * f for f in (0.2, 0.42, 0.68, 1.0)]             # where the creeping rust needs vertices

    # ---- stone footing: two dressed slabs with a joint
    stones = []
    if zf > 0:
        st = mat("stone_rough", "#b3ada0")
        S = 2 * (R + 0.056) * kf + 0.09
        r = rng.fork("footing")
        for sx in (-1, 1):
            hh = zf - (0.0 if sx < 0 else r.u(0.001, 0.003))
            b = stone_block("footing", (S / 2 - 0.004, S + r.u(-0.01, 0.01), hh), loc=(sx * S / 4, r.u(-0.004, 0.004), hh / 2),
                            rng=r, chamfer=0.012, rough=0.003, detail=0.16)
            assign(b, st)
            stones.append(b)

    # ---- plinth: bolted ground flange, scotia, half-round torus at kick height, die, cavetto up to the body
    def pz(v):
        return z0 + v * kp
    zt_ = hp - 0.05 * kp
    tor = [(R + 0.041, pz(0.053)), (R + 0.048, pz(0.062)), (R + 0.05, pz(0.075)), (R + 0.048, pz(0.088)), (R + 0.041, pz(0.097))]
    pl = [(R + 0.058, z0), (R + 0.058, pz(0.028)), (R + 0.054, pz(0.034)), (R + 0.037, pz(0.037)), (R + 0.035, pz(0.048))] + tor
    pl += [(R + 0.033, pz(0.103)), (R + 0.024, pz(0.112))]
    nd = 1
    pl += [(R + 0.024, lerp(pz(0.112), zt_, i / nd)) for i in range(1, nd + 1)]
    pl += [(R + 0.031, zt_ + 0.006 * kp), (R + 0.031, zt_ + 0.016 * kp), (R + 0.022, zt_ + 0.024 * kp), (R + 0.009, zt_ + 0.04 * kp),
           (R + 0.004, hp), (R - 0.004, hp + 0.001)]
    add(lat("plinth", pl), "blk")
    bt = join([cylinder("nut", 0.0085, 0.008, loc=(0, 0, 0.006), verts=6), cylinder("washer", 0.0125, 0.002, loc=(0, 0, 0.001), verts=8)], "hd_bolt")
    add(instances_to_mesh("hd_bolts", bt, [trs(((R + 0.046) * cos(radians(30 + 60 * k)), (R + 0.046) * sin(radians(30 + 60 * k)), pz(0.0355)),
                                               (0, 0, rng.u(0, 60))) for k in range(6)], keep_base=False), "rust")
    die_h = zt_ - pz(0.112)
    if p.maker.strip() and die_h > 0.03:
        mk = label("maker", p.maker.strip(), min(0.024, die_h * 0.42), 0.002, flat * 0.9 if hexa else 1.5 * R, font="sans", sides=False)
        place(mk, loc=(0, 0.0012, pz(0.112) + die_h * 0.5))
        add(onface(mk, rad=R + 0.024, cut=None), "blk")

    # ---- body: one casting (columns added where rust runs need them), letter aperture cut through it
    base_a = [2 * pi * i / NS for i in range(NS)]
    extra_a = []
    for (a, u, ztop, sg, L) in streaks:
        if a == 0.0 and abs(u) < dw / 2 and ztop < zd1:
            continue                                            # that run is on the door
        for du in (-1.4 * sg, 0.0, 1.4 * sg):
            extra_a.append((a + (atan((u + du) / R) if hexa else (u + du) / R) - pi / 2) % (2 * pi))
    cols = list(extra_a)
    for i, b in enumerate(base_a):
        corner = hexa and i % 4 == 0
        if corner or all(abs(((b - e + pi) % (2 * pi)) - pi) * R > 0.008 for e in extra_a):
            cols.append(b)
    cols.sort()
    zrows = [hp - 0.012, hp + 0.004] + rcuts
    nrow = max(3, int(round((zc - rcuts[-1]) / 0.14)))
    zrows += [lerp(rcuts[-1], zc, i / nrow) for i in range(1, nrow + 1)]
    shade = [zs - sh / 2 - 0.036, zh + 0.046] + ([zd1 - 0.022, zd1 - 0.055] if has_panels else [])
    for zx in shade:                                            # keep the baked shade under hood and mouldings tight
        if all(abs(zx - q) > 0.012 for q in zrows) and zx < zc:
            zrows.append(zx)
    zrows.sort()
    nc = len(cols)
    bv = [(R * cos(a), R * sin(a), z) for z in zrows for a in cols]
    bf = [(j * nc + i, j * nc + (i + 1) % nc, (j + 1) * nc + (i + 1) % nc, (j + 1) * nc + i) for j in range(len(zrows) - 1) for i in range(nc)]
    bf += [tuple(range(nc - 1, -1, -1)), tuple(range((len(zrows) - 1) * nc, len(zrows) * nc))]
    body = mesh_object("body", bv, bf)
    if hexa:
        transform_verts(body, lambda co: Vector((co.x * hexk(atan2(co.y, co.x)), co.y * hexk(atan2(co.y, co.x)), co.z)))
    cw = sw if hexa else 2 * R * sin(sw / 2 / R)
    cutter = box("slot_cut", (cw, 0.2, sh), loc=(0, -R - 0.025, zs))
    boolean(body, cutter)
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.normal_update()
    kill = [f for f in bm.faces if abs(f.normal.z) > 0.9 and (f.calc_center_median().z < hp or f.calc_center_median().z > zc - 0.004)]
    bmesh.ops.delete(bm, geom=kill, context='FACES')            # the ends are hidden inside the plinth and the cap
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    assign(body, red)
    # only the cut pocket (front half of the body) is the dark void
    assign_faces(body, dark, lambda fc, n: fc.y < -R * 0.5 and abs(fc.x) <= cw / 2 + 1e-3 and abs(fc.z - zs) <= sh / 2 + 1e-3
                 and (abs(n.y) < 0.5 or fc.y > -R + 0.04))
    bevel(body, 0.004 if hexa else 0.002, segments=1, angle=50)
    smooth(body, 40)
    if hexa:
        uv_box(body)
    else:
        uv_cyl(body)
    groups["red"].append(body)

    # ---- aperture surround and eyebrow hood
    fr = mring("slot_frame", sw + 0.046, sh + 0.042, 0.014, 0.021, 0.009, z0=zs)
    add(onface(fr), "red")
    hood = _prism("hood", [(-0.004, zh), (0.05, zh), (0.055, zh + 0.005), (0.055, zh + 0.011), (0.03, zh + 0.024), (-0.004, zh + 0.03)], -hw / 2, hw / 2, 8)

    def brow(co):
        e = min(1.0, abs(co.x) / (hw / 2)) ** 2
        y = co.y * (1.0 - 0.55 * e) if co.y < 0 else co.y
        return Vector((co.x, y, co.z + 0.01 * (1.0 - e)))
    transform_verts(hood, brow)
    bevel(hood, 0.002, segments=1)
    add(onface(hood, cut=None), "red")

    # ---- frieze lettering
    if p.legend.strip():
        lg = label("legend", p.legend.strip(), 0.039, 0.005, flat * 0.86 if hexa else 2.0 * R, weld=0.0011)
        place(lg, loc=(0, 0.001, zl))
        add(onface(lg, cut=None), "gold")

    # ---- door in its architrave, with a real shadow gap all round
    fcuts = [z - zm for z in rcuts]
    arch = mring("architrave", dw + 2 * fb + 0.006, dh + 2 * fb + 0.006, 0.02, fb, 0.0095, z0=zm, zcuts=fcuts)
    add(onface(arch), "red")
    reb = _frame("rebate", _rr(dw + 0.0015, dh + 0.0015, 0.0125, step=stp), [(0.0045, 0.0007), (-0.0045, 0.0007)], z0=zm)
    add(onface(reb), "dark")
    door = _frame("door", _rr(dw - 0.003, dh - 0.003, 0.011, step=stp), [(0.0, -0.002), (0.0, dd - 0.0014), (-0.0016, dd)], z0=zm, cap=True)
    dxs = []
    for (a, u, ztop, sg, L) in streaks:
        if a == 0.0 and abs(u) < dw / 2 - 0.004 and ztop < zd1:
            dxs += [clamp(u + du, -dw / 2 + 0.004, dw / 2 - 0.004) for du in (-1.4 * sg, 0.0, 1.4 * sg)]
    if hexa:
        dxs += [x for x in _steps(-dw / 2, dw / 2, 0.045) if all(abs(x - e) > 0.008 for e in dxs)]
    # rows of the door plate: none underneath the plate or the emblem (their baked shade would smear out
    # over the plate as a dark blot), tight ones just outside them and under the panel moulding instead
    dzs = _steps(zd0, zd1, 0.12) + rcuts + [zpb - 0.045, zpb - 0.124, zpb - 0.155]
    hide = [(zpb - 0.012, zpb + ph + 0.012)]
    zq0, zq1 = zd0 + 0.035, zpb - 0.105
    eq = min(dw - 0.14, 0.2, (zq1 - zq0 - 0.05) * 1.3)
    if zq1 - zq0 > 0.09 and p.emblem == "horn" and eq > 0.06:
        hide.append(((zq0 + zq1) / 2 - 0.23 * eq - 0.012, (zq0 + zq1) / 2 + 0.39 * eq + 0.01))
    for (h0, h1) in hide:
        dzs = [z for z in dzs if not (h0 - 0.004 < z < h1 + 0.004)] + [h0, h1]
    add(onface(door, xs=dxs, zs=[z for z in dzs if zd0 + 0.004 < z < zd1 - 0.004]), "red")

    # knuckle hinges on the left edge: barrel in three knuckles with pin caps, riveted strap on the door
    kn = lathe("knuckle", [(0, -0.0345), (0.0048, -0.0335), (0.0048, -0.0305), (0.008, -0.03), (0.008, -0.0112), (0.0062, -0.0098), (0.008, -0.0084),
                           (0.008, 0.0084), (0.0062, 0.0098), (0.008, 0.0112), (0.008, 0.03), (0.0048, 0.0305), (0.0048, 0.0335), (0, 0.0345)],
               segments=6, loc=(xk, -0.0105, 0))
    lf = _frame("leaf", _rr(0.034, 0.056, 0.0065, seg=1, step=10.0), [(0.0, dd - 0.001), (-0.0008, dd + 0.003), (-0.0048, dd + 0.003)], x0=xk + 0.019, cap=True)
    rv = lathe("rivet", [(0.0036, 0.0), (0.003, 0.0015), (0.0, 0.0024)], segments=5)
    rvs = instances_to_mesh("rivets", rv, [trs((xk + 0.027, -(dd + 0.003), sz * 0.016), (90, 0, 0)) for sz in (-1, 1)], keep_base=False)
    hinge = join([kn, lf, rvs], "hinge")
    hinges = instances_to_mesh("hinges", hinge, [trs((0, 0, z)) for z in hz], keep_base=False)
    add(onface(hinges, cut=None, sm=50), "red")

    # collection plate: enamel tablet in a screwed brass frame
    pf = mring("plate_frame", pw + 0.018, ph + 0.018, 0.008, 0.013, 0.0045, z0=zpl, d0=dd, seg=2, plain=True)
    add(onface(pf), "brass")
    tab = box("plate", (pw, 0.002, ph), loc=(0, -(dd + 0.001), zpl))
    add(onface(tab, xs=_steps(-pw / 2, pw / 2, 0.04) if hexa else (), zs=_steps(zpb, zpb + ph, 0.04)), "enamel")
    sc = lathe("screw", [(0.0031, 0.0), (0.0026, 0.0012), (0.0, 0.0019)], segments=6)
    scs = instances_to_mesh("plate_screws", sc, [trs((sx * (pw / 2 + 0.002), -(dd + 0.0048), zpl + sz * (ph / 2 + 0.002)), (90, 0, 0))
                                                 for sx in (-1, 1) for sz in (-1, 1)])
    add(onface(scs, cut=None), "brass")
    lines = [s.strip() for s in p.plate_text.split("|") if s.strip()][:4]
    if lines:
        lpitch = (ph - 0.022) / len(lines)
        for i, s in enumerate(lines):
            big = (i == 1) or len(lines) == 1
            t = label("plate_text%d" % i, s, min(0.028, lpitch * (0.78 if big else 0.56)), 0.0012, pw - 0.024, font="sans", sides=False)
            place(t, loc=(0, -(dd + 0.002), zpl + (ph - 0.022) / 2 - lpitch * (i + 0.5)))
            add(onface(t, cut=None), "dark")

    # lock rail: brass escutcheon with keyhole, ring pull
    key = circle_pts(0.0042, 7, -50, 230, center=(0, 0.006)) + [(-0.003, -0.011), (0.003, -0.011)]
    es = shape_extrude("escutcheon", rounded_rect(0.03, 0.052, 0.013, 3), [key], depth=0.004, loc=(xl, -dd + 0.0005, zlk), rot=(90, 0, 0))
    add(onface(es, cut=None), "brass")
    add(onface(box("keyhole", (0.013, 0.001, 0.026), loc=(xl, -(dd + 0.0006), zlk - 0.002)), cut=None), "dark")
    ess = instances_to_mesh("lock_screws", sc, [trs((xl, -(dd + 0.0042), zlk + sz * 0.02), (90, 0, 0)) for sz in (-1, 1)], keep_base=False)
    add(onface(ess, cut=None), "brass")
    xh = xl - 0.062 if dw > 0.2 else xl - 0.05
    boss = lathe("pull_boss", [(0, 0), (0.011, 0), (0.012, 0.002), (0.009, 0.005), (0.006, 0.006), (0.006, 0.012), (0.004, 0.014), (0, 0.0145)], segments=8,
                 loc=(xh, -dd, zlk + 0.012), rot=(90, 0, 0))
    ring = torus("pull_ring", 0.015, 0.003, loc=(xh, -(dd + 0.0085), zlk - 0.004), rot=(90 + rng.u(4, 9), 0, 0), major_seg=12, minor_seg=4)
    add(onface(boss, cut=None), "brass")
    add(onface(ring, cut=None), "brass")

    # moulded panel with the cast post horn (bold half-round relief: lead pipe, coil, bell, cord and tassels)
    inset = 0.045
    pnw = dw - 2 * inset
    zp0, zp1 = zd0 + 0.035, zpb - 0.105
    if zp1 - zp0 > 0.09:
        pn = mring("door_panel", pnw, zp1 - zp0, 0.014, 0.014, 0.004, z0=(zp0 + zp1) / 2, d0=dd, seg=2, zcuts=[z - (zp0 + zp1) / 2 for z in rcuts], plain=True)
        add(onface(pn), "red")
        ew = min(pnw - 0.05, 0.2, (zp1 - zp0 - 0.05) * 1.3)
        if p.emblem == "horn" and ew > 0.06:
            ke = min(1.0, ew / 0.16)
            rl = ew * 0.2                                       # coil radius
            ztop = (zp0 + zp1) / 2 + ew * 0.25
            cx = -ew * 0.04
            xa, xb = -ew / 2, ew / 2
            path = [(lerp(xa, cx, i / 5), ztop) for i in range(5)]
            path += [(cx + rl * cos(radians(a)), ztop - rl + rl * sin(radians(a))) for a in range(90, -271, -18)]
            path += [(lerp(cx, xb, i / 7), ztop) for i in range(1, 8)]
            tl = (cx - xa) + 2 * pi * rl + (xb - cx)
            tb = 1.0 - (xb - cx) * 0.8 / tl                     # where the bell starts to flare

            def hw_(t):
                if t < 0.022:
                    return 0.0085 * ke
                w = lerp(0.0056, 0.0098, min(t / tb, 1.0)) * ke
                if t > tb:
                    w += (0.25 * ew - 0.0098 * ke) * ((t - tb) / (1.0 - tb)) ** 2.1
                return w * (1.07 if t > 0.985 else 1.0)

            def hh_(t):
                return clamp(0.55 * hw_(t), 0.0042, 0.0075) * (1.15 if t > 0.985 else 1.0)
            horn = _relief("horn", path, hw_, hh_, dd)
            add(onface(horn, cut=None, sm=70), "gold")
            # cord slung under the coil, tied round the lead pipe and the bell throat, with two tassels
            x1, x2 = xa + ew * 0.13, cx + (xb - cx) * 0.42
            zb_ = ztop - 2 * rl - 0.062 * ew
            cpath = [(lerp(x1, x2, i / 14), ztop - (ztop - zb_) * sin(pi * i / 14) ** 0.7) for i in range(15)]
            parts = [_relief("cord", cpath, lambda t: 0.0047 * ke, lambda t: 0.004 * ke, dd, na=3)]
            for xt, sg_, ln in ((x1, -1, 0.2 * ew), (x2, 1, 0.15 * ew)):
                parts.append(_relief("tie", [(xt, ztop + 0.008 * ke), (xt, ztop), (xt, ztop - 0.008 * ke)], lambda t: 0.006 * ke,
                                     lambda t: 0.0085 * ke, dd, na=3))
                tz = ztop - 0.009 * ke
                parts.append(_relief("tassel", [(xt + sg_ * 0.007 * ke, tz - ln * f) for f in (0.0, 0.3, 0.42, 0.55, 0.8, 1.0)],
                                     lambda t: ke * (0.0045 if t < 0.36 else (0.0085 if t < 0.5 else lerp(0.007, 0.011, (t - 0.5) / 0.5))),
                                     lambda t: ke * (0.0038 if t < 0.36 else 0.005), dd, na=3))
            add(onface(join(parts, "horn_cord"), cut=None, sm=70), "gold")

    # ---- raised panels and bosses on the other faces
    if has_panels:
        sp_ = mring("side_panel", spw, dh, 0.016, 0.017, 0.007, z0=zm, seg=2, zcuts=fcuts, plain=True)
        onface(sp_)
        add(instances_to_mesh("side_panels", sp_, [trs((0, 0, 0), (0, 0, a)) for a in pangs], keep_base=False), "red")
        bs = lathe("boss", [(0.031, 0), (0.032, 0.003), (0.027, 0.007), (0.02, 0.006), (0.012, 0.014), (0, 0.016)],
                   segments=8, loc=(0, 0.001, zs), rot=(90, 0, 0))
        onface(bs, cut=None)
        add(instances_to_mesh("bosses", bs, [trs((0, 0, 0), (0, 0, a)) for a in pangs], keep_base=False), "red")

    # ---- cap: neck moulding, overhanging beaded fascia, roof, finial
    capp = [(R - 0.004, zc - 0.05 * c), (R + 0.005, zc - 0.046 * c), (R + 0.005, zc - 0.036 * c), (R + 0.011, zc - 0.028 * c),
            (R + ov * 0.6, zc - 0.007 * c), (R + ov * 0.8, zc + 0.002 * c), (R + ov * 0.84, zc + 0.011 * c),
            (R + ov, zc + 0.015 * c), (R + ov, zc + 0.05 * c), (R + ov - 0.005 * ws, zc + 0.058 * c), (R + ov - 0.016 * ws, zc + rim_h)]
    r0 = R + ov - 0.016 * ws
    zt = zc + rim_h
    if p.cap == "dome":
        top = []
        for i in range(1, 6):
            a = i / 6 * pi / 2
            top.append((r0 * cos(a), zt + hd * sin(a)))
            if i == 3:                        # raised roll round the crown
                top += [(r0 * cos(a) - 0.002, zt + hd * sin(a) + 0.006 * cs), (r0 * cos(a) - 0.01, zt + hd * sin(a) + 0.008 * cs)]
    else:
        ctrl = ([(1, 0), (0.97, 0.14), (0.84, 0.33), (0.6, 0.52), (0.36, 0.67), (0.2, 0.8), (0.12, 0.92), (0.09, 1.0)] if p.cap == "ogee"
                else [(1, 0), (0.92, 0.2), (0.72, 0.46), (0.46, 0.72), (0.25, 0.9), (0.12, 1.0)])
        pts = resample_path([(r0 * a, 0.0, zt + hd * b) for a, b in ctrl], n=10 if p.cap == "ogee" else 8)
        top = [(q[0], q[2]) for q in pts][1:]
    add(lat("cap", capp + top + [(0.0, zt + hd)], sm=45), "red")
    za = zt + hd
    if p.cap == "dome":
        fp = [(0.036, -0.014), (0.038, 0.0), (0.03, 0.006), (0.02, 0.009), (0.016, 0.016), (0.023, 0.022), (0.023, 0.028), (0.013, 0.036), (0, 0.04)]
    elif p.cap == "ogee":
        fp = [(0.03, -0.02), (0.032, 0.0), (0.022, 0.008), (0.014, 0.016), (0.014, 0.026), (0.024, 0.03), (0.026, 0.036), (0.016, 0.043), (0.012, 0.05)]
        fp += [(0.03 * cos(radians(a)), 0.078 + 0.03 * sin(radians(a))) for a in range(-66, 67, 22)]
        fp += [(0.009, 0.108), (0.006, 0.12), (0, 0.13)]
    else:
        fp = [(0.032, -0.02), (0.036, 0.0), (0.026, 0.008), (0.017, 0.014), (0.015, 0.024), (0.026, 0.034), (0.036, 0.05), (0.037, 0.062), (0.029, 0.08),
              (0.017, 0.096), (0.007, 0.106), (0, 0.11)]
    add(lathe("finial", [(a * c, za + b * c) for a, b in fp], segments=12), "gold" if p.cap != "dome" else "red")

    # beads on the fascia (domes; the ones that lost their paint are bare rusty iron)
    cr = rng.fork("chips")
    br = 0.0105 * ws
    rb = R + ov - br * 0.15
    zbd = zc + 0.0325 * c
    bead = lathe("bead", [(br, 0.0), (br * 0.74, br * 0.66), (0.0, br)], segments=6)
    smooth(bead, 80)
    bm_ = []
    if hexa:
        L = 2 * rb * T30
        n = max(3, int(round(L / (br * 3.0))))
        for j in range(6):
            th = radians(30 + 60 * j)
            for i in range(n):
                o_ = (i / n - 0.5) * L
                bm_.append(((rb * cos(th) - sin(th) * o_, rb * sin(th) + cos(th) * o_, zbd), (0, 90, 30 + 60 * j)))
    else:
        n = int(2 * pi * rb / (br * 2.9))
        bm_ = [((rb * cos(2 * pi * i / n), rb * sin(2 * pi * i / n), zbd), (0, 90, 360.0 * i / n)) for i in range(n)]
    lost, run = [], 0
    for i in range(len(bm_)):                # knocked beads come in short runs
        run = cr.pick([1, 1, 2, 3]) if (run == 0 and cr.chance(0.02 + 0.09 * wr)) else max(run - 1, 0)
        lost.append(run > 0)
    good = [trs(l, r_) for (l, r_), k in zip(bm_, lost) if not k]
    bad = [trs(l, r_, 0.9) for (l, r_), k in zip(bm_, lost) if k]
    if bad:
        add(instances_to_mesh("beads_bare", bead, bad, uv_jitter=False), "rust")
    add(instances_to_mesh("beads", bead, good, keep_base=False, uv_jitter=False), "red")

    # acanthus leaves: thick cast leaves that follow the roof, with a raised midrib, scalloped lobes and
    # tips that turn down over the edge onto the fascia
    if p.cap == "acanthus":
        nl = 6 if hexa else 8
        th0 = pi / 6 if hexa else 0.0             # hex: one leaf lying on each face of the roof
        aw = (0.88 if hexa else 1.0) * pi / nl
        lp = _Poly([(R + ov, zc + 0.043 * c), (R + ov, zc + 0.05 * c), (R + ov - 0.005 * ws, zc + 0.058 * c), (r0, zt)] + [q for q in top if q[0] >= 0.05 * c])
        qr = lp.cum[3]
        TL = 0.008 * c
        NVL = 9
        rowsL = [(0.0006, 0.17, -1.0, 0.0), (0.0022, 0.21, 0.8, 0.0), (lp.cum[1], 0.3, 1.05, 0.0), (lp.cum[2], 0.37, 1.1, 0.0)]
        for j in range(NVL, -1, -1):
            v = j / NVL
            A = lerp(0.62, 0.84, v / 0.4) if v < 0.4 else (lerp(0.84, 0.66, (v - 0.4) / 0.35) if v < 0.75 else lerp(0.66, 0.3, (v - 0.75) / 0.25))
            nt = abs(sin(3 * pi * v)) ** 0.7
            rowsL.append((qr + (lp.tot - qr) * (1.0 - v), A * (1.0 - 0.34 * nt), 1.0, nt))
        rowsL.append((lp.tot, 0.5, -1.0, 0.0))
        ucols = [-1.05, -1.0, -0.16, 0.0, 0.16, 1.0, 1.05]
        lv, lfc = [], []
        for (q, A, tf, nt) in rowsL:
            r_, z_, nr, nz = lp.at(q, blur=0.005)
            for k, u in enumerate(ucols):
                if tf < 0 or k in (0, 6):
                    h = -0.0015
                elif k in (1, 5):
                    h = TL * 0.55 * (1.0 - 0.4 * nt) * tf
                elif k in (2, 4):
                    h = TL * tf
                else:
                    h = (TL + 0.0045 * c) * tf
                th = th0 + u * A * aw
                kk = hexk(th)
                lv.append(((r_ + h * nr) * kk * cos(th), (r_ + h * nr) * kk * sin(th), z_ + h * nz))
        m = len(ucols)
        for j in range(len(rowsL) - 1):
            for k in range(m - 1):
                lfc.append((j * m + k, j * m + k + 1, (j + 1) * m + k + 1, (j + 1) * m + k))
        leaf = mesh_object("leaf", lv, lfc)
        smooth(leaf, 42)
        add(instances_to_mesh("leaves", leaf, [trs((0, 0, 0), (0, 0, k * 360.0 / nl + (0 if hexa else 22.5))) for k in range(nl)], keep_base=False), "red")

    # ---- paint flakes: thin modelled patches of bare rusty iron (with the pale undercoat showing round the
    #      bigger ones) that follow the turned / wrapped surfaces, placed where a pillar box gets knocked
    chipbuf = {"rust": ([], []), "edge": ([], [])}

    def chip(poly, a, u0, s, size, asp=1.0, pale=True, off=0.0008, facets=0, rw=None):
        """Flake centred at arc length s on the profile `poly`, on the face / at the angle `a` from the
        front (radians, CCW), u0 metres along the surface from there. size = height, asp = width / height.
        rw = radius the surface was wrapped at (parts built flat), facets = segments of a turned part."""
        s0, s1 = max(s - size / 2, 0.0004), min(s + size / 2, poly.tot - 0.0004)
        if s1 - s0 < 0.0015:
            return
        Wd = size * asp
        e = clamp(size * 0.1, 0.001, 0.0022) if pale else 0.0
        if s1 - s0 < 4.5 * e or Wd < 4.5 * e or size < 0.007:
            e = 0.0
        rows = [s0, s1] + ([s0 + e, s1 - e] if e else [])
        nsub = 1 if size < 0.009 else (2 if size < 0.02 else 3)
        for q in list(poly.cum) + [lerp(s0, s1, i / nsub) for i in range(1, nsub)]:
            if s0 < q < s1 and all(abs(q - x) > 0.0007 for x in rows):
                rows.append(q)
        rows.sort()
        ncol = max(1, int(ceil(Wd / 0.02)))
        fq = [cr.u(0.6, 1.4), cr.u(1.8, 3.2), cr.u(0.6, 1.4), cr.u(1.8, 3.2)]
        ph_ = [cr.u(), cr.u(), cr.u(), cr.u()]
        shr = cr.u(-0.3, 0.3)
        rc = rw or poly.at(s)[0]
        ca, sa = cos(a), sin(a)
        V, grid = [], []
        for j, q in enumerate(rows):
            t = (q - s0) / (s1 - s0)
            env = 0.34 + 0.66 * max(0.0, 1.0 - (2 * t - 1) ** 2) ** 0.5
            Lw = Wd / 2 * env * (1 + 0.28 * sin(2 * pi * (fq[0] * t + ph_[0])) + 0.15 * sin(2 * pi * (fq[1] * t + ph_[1])))
            Rw = Wd / 2 * env * (1 + 0.28 * sin(2 * pi * (fq[2] * t + ph_[2])) + 0.15 * sin(2 * pi * (fq[3] * t + ph_[3])))
            xc_ = shr * Wd * (t - 0.5)
            xa_, xb_ = xc_ - Lw, xc_ + Rw
            if e:
                end = j in (0, len(rows) - 1)
                ee = 0.0 if end else min(e, (xb_ - xa_) * 0.25)
                xs_ = [xa_, xa_ + ee] + [lerp(xa_ + ee, xb_ - ee, i / ncol) for i in range(1, ncol)] + [xb_ - ee, xb_]
            else:
                xs_ = [lerp(xa_, xb_, i / ncol) for i in range(ncol + 1)]
            r_, z_, nr, nz = poly.at(q)
            ids = []
            for k, x in enumerate(xs_):
                if k and abs(x - xs_[k - 1]) < 1e-7:
                    ids.append(ids[-1])
                    continue
                u = u0 + x
                rho = r_ + off * nr
                if hexa:
                    V.append((rho * sa + u * ca, -rho * ca + u * sa, z_ + off * nz))
                else:
                    th = a + u / rc
                    if facets:
                        rho *= cos(pi / facets) / cos(((th - pi / 2) % (2 * pi / facets)) - pi / facets)
                    V.append((rho * sin(th), -rho * cos(th), z_ + off * nz))
                ids.append(len(V) - 1)
            grid.append(ids)
        m = len(grid[0])
        for j in range(len(rows) - 1):
            eb = e and (rows[j + 1] <= s0 + e + 1e-7 or rows[j] >= s1 - e - 1e-7)
            for k in range(m - 1):
                f = []
                for vi in (grid[j][k], grid[j][k + 1], grid[j + 1][k + 1], grid[j + 1][k]):
                    if vi not in f:
                        f.append(vi)
                if len(f) < 3:
                    continue
                vs, fs = chipbuf["edge" if (eb or (e and k in (0, m - 2))) else "rust"]
                base = len(vs)
                vs.extend(V[vi] for vi in f)
                fs.append(tuple(range(base, base + len(f))))

    nmul = min(2.0 * wr ** 0.8, 1.7)

    def cnt(base):
        return int(base * nmul + cr.u())

    def csz(k=1.0, mx=0.03):
        return min(0.0035 * 9 ** (cr.u() ** 1.1) * k * lerp(0.85, 1.2, wr), mx)

    def around(rad, wd, front_gap=0.0):
        """Random (a, u0) for a flake of width wd on a surface of radius rad; front_gap keeps clear of the door."""
        if hexa:
            lim = rad * T30 - 0.75 * wd - 0.011
            if lim < 0:
                return None
            return radians(60.0 * (cr.pick([1, 2, 3, 4, 5]) if front_gap else cr.pick([0, 1, 2, 3, 4, 5]))), cr.u(-lim, lim)
        return cr.u(front_gap, 2 * pi - front_gap), 0.0

    doorP = _Poly([(R + dd, zd0 + 0.002), (R + dd, zd1 - 0.002)])
    for _ in range(cnt(12)):                                 # door: lock side edge, thickest round the lock
        sz, asp = csz(), cr.u(0.5, 1.1)
        z = zlk + cr.u(-0.1, 0.07) if cr.chance(0.5) else cr.u(zd0 + 0.02, zd1 - 0.03)
        chip(doorP, 0.0, dw / 2 - 0.0036 - sz * asp * 0.75 - cr.u() ** 2 * 0.018, z - zd0 - 0.002, sz, asp, rw=R)
    for _ in range(cnt(8)):                                  # door: bottom edge
        sz = csz()
        asp = min(cr.u(1.0, 2.4), 0.034 / sz)
        chip(doorP, 0.0, cr.u(-1, 1) * (dw / 2 - 0.016 - sz * asp * 0.75), 0.0025 + sz * 0.5 + cr.u() ** 2 * 0.02, sz, asp, rw=R)
    for _ in range(cnt(3)):                                  # door: hinge side, between the hinges
        sz, asp = csz(0.8), cr.u(0.5, 1.0)
        i = cr.pick(list(range(nh - 1)))
        chip(doorP, 0.0, -dw / 2 + 0.0036 + sz * asp * 0.75 + cr.u() ** 2 * 0.012, lerp(hz[i], hz[i + 1], cr.u(0.3, 0.7)) - zd0 - 0.002, sz, asp, rw=R)
    lipP = _Poly([(R + 0.009, zs - sh / 2 - 0.0165), (R + 0.009, zs - sh / 2 - 0.0065)])
    for _ in range(cnt(7)):                                  # aperture lip, scuffed by parcels and sleeves
        sz = min(csz(0.8), 0.0085)
        asp = cr.u(1.5, 3.5)
        chip(lipP, 0.0, cr.u(-1, 1) * (sw / 2 - sz * asp * 0.6), cr.u(0.35, 0.65) * lipP.tot, sz, asp, rw=R)
    for _ in range(cnt(7)):                                  # hood: the sloping top just behind its front edge
        x = cr.u(-0.6, 0.6) * hw / 2
        e_ = (abs(x) / (hw / 2)) ** 2
        sc_, zz = 1.0 - 0.55 * e_, zh + 0.01 * (1.0 - e_)
        hoodP = _Poly([(R + 0.052 * sc_, zz + 0.0126), (R + 0.034 * sc_, zz + 0.022)])
        sz = min(csz(0.8), 0.014)
        chip(hoodP, 0.0, x, cr.u(0.3, 0.6) * hoodP.tot, sz, min(cr.u(1.0, 2.2), 0.016 / sz), off=0.0012, rw=R)
    archP = _Poly([(R + 0.0095, zd0 + 0.012), (R + 0.0095, zd1 - 0.012)])
    for _ in range(cnt(5)):                                  # architrave on the lock side
        sz = min(csz(), 0.022)
        asp = min(cr.u(0.4, 0.9), 0.006 / sz)
        chip(archP, 0.0, dw / 2 + 0.003 + fb / 2 + 0.001, cr.u(0.02, 0.98) * archP.tot, sz, asp, rw=R)
    archB = _Poly([(R + 0.0095, zd0 - 0.0186), (R + 0.0095, zd0 - 0.0094)])
    for _ in range(cnt(6)):                                  # architrave sill
        sz = min(csz(), 0.0082)
        asp = cr.u(1.5, 3.5)
        chip(archB, 0.0, cr.u(-1, 1) * (dw / 2 - 0.012), cr.u(0.4, 0.6) * archB.tot, sz, asp, rw=R)
    fgap = (dw / 2 + fb + 0.03) / R
    bodyP = _Poly([(R, hp + 0.003), (R, hp + 0.32)])
    for _ in range(cnt(16)):                                 # foot of the body, all round
        sz, asp = csz(), cr.u(0.7, 1.8)
        au = around(R, sz * asp, fgap)
        if au:
            chip(bodyP, au[0], au[1], sz / 2 + 0.002 + cr.u() ** 2 * 0.26, sz, asp)
    bodyM = _Poly([(R, hp + 0.3), (R, zc - 0.08)])
    for _ in range(cnt(2)):                                  # the odd small one on the broad panels
        sz = min(csz(0.7), 0.008)
        au = around(R, sz, fgap)
        if au:
            chip(bodyM, au[0], au[1], cr.u(0.05, 0.95) * bodyM.tot, sz, cr.u(0.7, 1.4))
    fz0, fz1 = zc + 0.0155 * c, zc + 0.0495 * c
    for zz0, zz1 in ((fz0, zbd - br * 0.97), (zbd + br * 0.97, fz1)):   # cap fascia, either side of the beads
        if zz1 - zz0 > 0.0035:
            fasP = _Poly([(R + ov, zz0), (R + ov, zz1)])
            for _ in range(cnt(6)):
                sz = min(csz(), fasP.tot - 0.001)
                asp = cr.u(1.5, 4.0)
                au = around(R + ov, sz * asp)
                if au:
                    chip(fasP, au[0], au[1], fasP.tot / 2, sz, asp, facets=NS)
    if p.cap != "acanthus":                                  # roof, mostly near the edge
        domeP = _Poly([(r0, zt)] + top)
        for _ in range(cnt(8)):
            sz, asp = csz(), cr.u(0.7, 1.6)
            s = 0.004 + sz / 2 + cr.u() ** 1.5 * 0.5 * domeP.tot
            au = around(domeP.at(s)[0], sz * asp)
            if au:
                chip(domeP, au[0], au[1], s, sz, asp, facets=NS)
    torP = _Poly(tor)
    for _ in range(cnt(14)):                                 # plinth torus: boots, barrows, cart wheels
        sz, asp = csz(1.25), cr.u(0.8, 2.2)
        au = around(R + 0.05, sz * asp)
        if au:
            chip(torP, au[0], au[1], cr.u(0.25, 0.75) * torP.tot, sz, asp, pale=False, facets=NS)
    flP = _Poly([(R + 0.058, z0 + 0.003), (R + 0.058, pz(0.028))])
    for _ in range(cnt(6)):                                  # arris of the ground flange
        sz, asp = min(csz(), 0.02), cr.u(1.0, 2.5)
        au = around(R + 0.058, sz * asp)
        if au:
            chip(flP, au[0], au[1], flP.tot - sz / 2 - cr.u(0, 0.003), sz, asp, pale=False, facets=NS)
    dieP = _Poly([(R + 0.024, pz(0.114)), (R + 0.024, zt_)])
    for _ in range(cnt(4)):                                  # die, away from the founder's name
        sz, asp = csz(), cr.u(0.7, 1.6)
        au = around(R + 0.024, sz * asp, 1.0)
        if au:
            chip(dieP, au[0], au[1], cr.u(0.15, 0.85) * dieP.tot, sz, asp, pale=False, facets=NS)
    for kind in ("rust", "edge"):
        vs, fs = chipbuf[kind]
        if fs:
            o = mesh_object("flakes_" + kind, vs, fs)
            smooth(o, 60)
            add(o, kind)

    # ---- weathering: rust creeping up from the plinth in streaks, bleeding below hinges, hood, lock,
    #      plate and bosses, sitting on ledges and round the flange bolts; the lip rubbed by hands
    rustn = noise_mask(0.09, seed=p.seed + 5, lo=0.25, hi=0.75)
    finen = noise_mask(0.035, seed=p.seed + 8, lo=0.45, hi=0.7)
    strk = noise_mask(0.07, seed=p.seed + 6, lo=0.3, hi=0.7)
    patchn = noise_mask(0.16, seed=p.seed + 21, lo=0.58, hi=0.74)
    squash = np.array([1.0, 1.0, 0.12])
    flat0 = np.array([1.0, 1.0, 0.0])

    def streakn(co, n):                 # vertical streaks: noise sampled on squashed heights
        return strk(co * squash, n)

    def runs(co):
        z = co[:, 2]
        m = np.zeros(len(co))
        for (a, u, ztop, sg, L) in streaks:
            if hexa:
                xr = co[:, 0] * cos(a) + co[:, 1] * sin(a)
                out = co[:, 0] * sin(a) - co[:, 1] * cos(a)
                d = np.where(out > 0.6 * R, xr - u, 1.0)
            else:
                d = (((np.arctan2(co[:, 0], -co[:, 1]) - (a + u / R)) + pi) % (2 * pi) - pi) * R
            g = np.exp(-(d / sg) ** 2) * np.clip(1.0 - (ztop - z) / L, 0, 1) ** 1.2 * (z <= ztop + 0.004)
            m = np.maximum(m, g)
        return m * (0.65 + 0.35 * wr)

    def rust_red(co, n):
        z = co[:, 2]
        rad = np.hypot(co[:, 0], co[:, 1])
        tong = 0.34 + 0.66 * strk(co * flat0, n)        # creeps up in tongues of uneven height
        base = np.clip(1.0 - (z - hp) / (Hr * tong), 0, 1)
        m = np.clip(base * 1.7, 0, 1) * (0.55 + 0.45 * finen(co, n)) * (0.75 + 0.25 * wr)
        m = np.maximum(m, runs(co))
        if wr > 0.5:                    # neglected: streaky patches breaking out higher up
            m = np.maximum(m, patchn(co, n) * streakn(co, n) * (wr - 0.5) * 1.5)
        low = z < zc - 0.06 * c
        m = np.maximum(m, low * np.clip(n[:, 2], 0, 1) * rustn(co, n) * (0.25 + 0.4 * wr))            # ledges hold water
        proud = (-co[:, 1] > R + 0.03) if hexa else ((rad > R + 0.03) & (co[:, 1] < 0))
        hood_e = (z > zh - 0.003) & (z < zh + 0.04) & proud
        m = np.maximum(m, hood_e * finen(co, n) * (0.3 + 0.5 * wr))                                    # hood edge
        rim = (z > zc + 0.008 * c) & (z < zc + 0.056 * c)
        m = np.maximum(m, rim * finen(co, n) * rustn(co, n) * (0.25 + 0.55 * wr))                      # cap rim and beads
        lip = (z < zs - sh / 2 + 0.002) & (z > zs - sh / 2 - 0.024) & (np.abs(co[:, 0]) < sw / 2 + 0.02) & (co[:, 1] < -R * 0.8)
        m = np.maximum(m, lip * (0.2 + 0.35 * wr) * (0.4 + 0.6 * rustn(co, n)))
        return np.clip(m, 0, 0.95)

    def wear_red(co, n):                # bare metal only where hands and parcels rub the aperture
        z = co[:, 2]
        lip = (np.abs(z - (zs - sh / 2 - 0.004)) < 0.006) & (np.abs(co[:, 0]) < sw / 2) & (co[:, 1] < -R * 0.8)
        return lip * (0.25 + 0.4 * wr) * finen(co, n)

    def grime_red(co, n):
        z = co[:, 2]
        return np.clip(1.0 - (z - z0) / 0.5, 0, 1) * 0.28 + np.clip(n[:, 2], 0, 1) * 0.18 * (z < zc)

    def rust_blk(co, n):
        z = co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        m = up * (0.3 + 0.7 * rustn(co, n)) * (0.4 + 0.5 * wr)                                        # ledges
        m = np.maximum(m, np.clip(1.0 - (z - z0) / (0.05 + 0.1 * wr), 0, 1) * streakn(co, n) * (0.3 + 0.5 * wr))
        m = np.maximum(m, (z > pz(0.1)) * streakn(co, n) * finen(co, n) * (0.15 + 0.5 * wr))          # runs down the die
        ang = np.arctan2(co[:, 1], co[:, 0])
        for k in range(6):                                                                             # holding-down bolts
            d = ((ang - radians(30 + 60 * k) + pi) % (2 * pi) - pi) * (R + 0.046)
            m = np.maximum(m, np.exp(-(d / 0.024) ** 2) * (z < pz(0.05)) * (0.6 + 0.35 * wr))
        if wr > 0.5:
            m = np.maximum(m, patchn(co, n) * (wr - 0.5) * 1.6)
        return np.clip(m, 0, 0.95)

    def grime_blk(co, n):
        return np.clip(1.0 - (co[:, 2] - z0) / 0.2, 0, 1) * 0.2

    for o in groups["red"]:
        paint(o, wear=wear_red, special=rust_red, grime=grime_red)
    for o in groups["blk"]:
        paint(o, special=rust_blk, grime=grime_blk)
    for o in groups["rust"]:            # flake cores, rusted bolts and beads: blotchy rust over the primer
        paint(o, special=mask_mul(noise_mask(0.012, seed=p.seed + 31, lo=0.3, hi=0.7), 0.9), grime=0.2, mode="set")
    for o in groups["gold"]:
        if p.gilding:
            paint(o, wear=mask_mul(finen, 0.25 * wr), grime=0.1 + 0.2 * wr)
        else:
            paint(o, special=rust_red, grime=grime_red)
    for o in groups["brass"]:
        paint(o, wear=0.3, special=mask_mul(noise_mask(0.03, seed=p.seed + 7), 0.3 + 0.5 * wr), grime=0.2)
    for o in groups["enamel"]:
        paint(o, wear=mask_mul(noise_mask(0.025, seed=p.seed + 9, lo=0.72, hi=0.8), 0.3 + 0.7 * wr), grime=0.15 + 0.3 * wr)
    for b in stones:
        paint(b, grime=mask_mul(noise_mask(0.15, seed=p.seed + 14, lo=0.2, hi=0.8), 0.45), special=mask_mul(noise_mask(0.2, seed=p.seed + 13), 0.5))
