"""Tiered plaza fountain - the centrepiece of the square.

A dressed-stone basin (octagon / round / barbed quatrefoil plan) on a stepped kerb: moulded plinth
course, die stones with sunk-and-fielded panels and carved paterae, bull-nosed coping, corner piers
with ball finials or small cast-iron lantern standards. In the pool stands a pedestal with bronze
lion-mask spouts carrying one to three scalloped bowls on turned baluster stems, crowned by a carved
finial with a bronze nozzle. Clear water lies in the pool and in every bowl, falls in fraying sheets
from each lobe of the scalloped lips, arcs from the lion mouths and plays from the finial; every
landing has its foam and splash crown. Bronze bulkhead lights glow under the pool surface. The stone
carries a green-black algae line, moss in the joints, wet run-off and lime crust where the water runs.

`diameter` is the footprint: the outer diameter of the kerb (outermost step). Everything is built per
stone (individually lofted blocks with V-joints) along an analytic plan curve, so every plan /
diameter / step count yields properly jointed masonry.

Levels: every stone, bowl, sheet of water and ornament is rebuilt with what its distance shows (the
file's own mesh builders take the level: _stone, _stack, _polar callers, _sheet_mesh, _Froth, _Moss,
_patera). lod 1: stones without spalls meeting in a plain V-joint, no faces under the water or between
courses, half the columns / rows of every sheet, star-cut paterae, pyramid moss. lod 2: stones butted
(their tones carry the coursing), one sunk bevel for the panels, flat octagonal paterae, three-strip
sheets, no moss cushions or flung drops. The random streams are drawn in full at every level.
"""
from tk import *
import bmesh
import numpy as np
from math import exp, ceil
from mathutils import Vector

META = {"name": "Tiered plaza fountain", "class": "landmark", "mount": "ground",
        "focus": [0.0, -0.7, 1.5], "max_lights": 8, "auto_grime": 0.35, "lod": True}

PARAMS = {
    "diameter": {"default": 6.0, "min": 2.5, "max": 7.0, "desc": "outer diameter of the basin kerb = footprint of the outermost step (m)"},
    "plan": {"default": "octagon", "choices": ["octagon", "round", "quatrefoil"], "desc": "basin plan shape"},
    "tiers": {"default": 3, "min": 1, "max": 3, "desc": "number of bowls on the central stem (small basins carry at most 2)"},
    "spouts": {"default": 4, "min": 0, "max": 8, "desc": "bronze lion-mask spouts on the pedestal"},
    "scallops": {"default": 12, "min": 0, "max": 16, "desc": "lobes of the lowest bowl (0 = plain turned bowls)"},
    "stem_height": {"default": 1.0, "min": 0.8, "max": 1.3, "desc": "vertical stretch of the central stem"},
    "steps": {"default": 2, "min": 1, "max": 3, "desc": "steps of the kerb the basin stands on (fewer on small diameters)"},
    "finial": {"default": "pinecone", "choices": ["pinecone", "urn", "ball"], "desc": "crowning ornament"},
    "stone": {"default": "#ab9f89", "desc": "stone tint of basin and bowls"},
    "paint": {"default": "#1b2622", "desc": "paint colour of the lantern standards"},
    "rim_lanterns": {"default": 4, "min": 0, "max": 4, "desc": "small lantern standards on the rim piers"},
    "pool_lights": {"default": 2, "min": 0, "max": 4, "desc": "submerged bronze bulkhead lights in the pool"},
    "lit": {"default": True, "desc": "lanterns and pool lights burning"},
    "flow": {"default": 1.0, "min": 0.0, "max": 1.0, "desc": "water flow: 0 = switched off / stagnant, 1 = full sheets"},
    "moss": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age: moss, algae and damp staining"},
    "inscription": {"default": "ANNO MDCCCLXXIV", "desc": "bronze lettering of the dedication tablet on the front panel ('' = no tablet)"},
}

VARIANTS = {
    "round_village": {"plan": "round", "diameter": 4.4, "tiers": 2, "scallops": 0, "spouts": 3, "steps": 1,
                      "finial": "ball", "stone": "#a8a59b", "rim_lanterns": 3, "pool_lights": 1,
                      "inscription": "AQVA PVRA", "seed": 7},
    "quatrefoil_grand": {"plan": "quatrefoil", "diameter": 7.0, "tiers": 3, "scallops": 16, "spouts": 4, "steps": 3,
                         "stem_height": 1.15, "finial": "urn", "stone": "#b9a684", "pool_lights": 4,
                         "inscription": "S P Q R  MDCCXCVI", "seed": 11},
    "granite_single": {"plan": "octagon", "diameter": 4.8, "tiers": 1, "scallops": 8, "spouts": 8, "steps": 1,
                       "stem_height": 1.2, "stone": "#8e9196", "rim_lanterns": 2, "pool_lights": 4, "paint": "#20232a",
                       "inscription": "", "moss": 0.25, "seed": 19},
    "neglected": {"plan": "round", "diameter": 6.4, "tiers": 2, "scallops": 10, "spouts": 6, "steps": 2,
                  "finial": "urn", "stone": "#a9a48f", "flow": 0.0, "moss": 1.0, "rim_lanterns": 2,
                  "pool_lights": 0, "paint": "#2a1d14", "seed": 23},
    "pocket_court": {"plan": "octagon", "diameter": 2.5, "tiers": 2, "scallops": 8, "spouts": 2, "steps": 1,
                     "finial": "ball", "stone": "#b3a68c", "rim_lanterns": 2, "pool_lights": 1,
                     "inscription": "", "moss": 0.7, "seed": 31},
}

G = 9.81


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------

def _ss(a, b, x):
    """smoothstep that also works for a > b (falling edge)."""
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _ssn(a, b, x):
    """numpy smoothstep (a > b = falling edge)."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _shade(hexcol, k):
    h = hexcol.lstrip("#")
    c = [min(255, max(0, int(round(int(h[i:i + 2], 16) * k)))) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(c)


def _mix(ha, hb, t):
    a, b = ha.lstrip("#"), hb.lstrip("#")
    c = [int(round(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t)) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(c)


def _q(p0, c, p1, n=5):
    """Quadratic bezier p0 -> p1 (control c), without the start point."""
    out = []
    for i in range(1, n + 1):
        t = i / n
        a, b, d = (1 - t) ** 2, 2 * t * (1 - t), t * t
        out.append((a * p0[0] + b * c[0] + d * p1[0], a * p0[1] + b * c[1] + d * p1[1]))
    return out


def _c3(p0, c1, c2, p1, n=8):
    """Cubic bezier without the start point."""
    out = []
    for i in range(1, n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
        out.append((a * p0[0] + b * c1[0] + c * c2[0] + d * p1[0], a * p0[1] + b * c1[1] + c * c2[1] + d * p1[1]))
    return out


def _vstreak(seed, lo=0.42, hi=0.72, fine=0.07):
    """Vertical run-off streak mask (noise stretched along Z)."""
    nm = noise_mask(fine, seed=seed, lo=lo, hi=hi)
    sc = np.array([1.0, 1.0, 0.09])
    return lambda co, n: nm(co * sc, n)


_UVN = [0]


def _mesh_uv(name, verts, faces, fuv):
    """mesh_object with explicit per-corner UVs (meters)."""
    o = mesh_object(name, verts, faces)
    me = o.data
    flat = np.array([c for uvs in fuv for uv in uvs for c in uv], dtype=np.float32)
    if len(flat) == len(me.loops) * 2:
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", flat)
        uv_keep(o)
    else:
        uv_box(o)
    return o


def _mesh_vuv(name, verts, faces, uv, wear=None, special=None, sm=70.0):
    """mesh_object with per-vertex UVs (meters) and, for falling water, the two shader hints of the
    `water_fall` material written straight into the weathering masks: wear = break-up (0 = glassy film at
    the lip .. 1 = torn, aerated threads), special = foam (1 = white water)."""
    o = _mesh_uv(name, verts, faces, [[uv[i] for i in f] for f in faces])
    smooth(o, sm)
    if wear is not None:
        _vmask(o, "wear", wear)
    if special is not None:
        _vmask(o, "special", special)
    return o


def _vmask(o, name, vals):
    """Write a weathering mask (grime / wear / special) straight from a per-vertex list."""
    me = o.data
    if len(vals) != len(me.vertices):
        return
    at = me.attributes.get("m_" + name) or me.attributes.new("m_" + name, 'FLOAT', 'POINT')
    at.data.foreach_set("value", np.clip(np.asarray(vals, dtype=np.float32), 0.0, 1.0))


def _h1(a, b, c=0.0):
    """Stable 0..1 hash: every stone gets its own tone without touching the layout streams."""
    v = sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453
    return v - int(v) if v >= 0 else v - int(v) + 1.0


def _dress(o, k=1.0, a=0.0):
    """Fix the face-aligned meter UVs of a dressed stone. The library stone is isotropic (pitted, no tooling
    direction), so by default nothing is sheared; k / a squeeze U and turn the UVs for a directional finish."""
    realize(o)
    if o.data.uv_layers:
        if k != 1.0 or a != 0.0:
            uv_transform(o, scale=(k, 1.0), rot_deg=a)
        uv_keep(o)
    return o


# ----------------------------------------------------------------------------------------------
# plan curve: every masonry course is lofted along pt(segment, f, offset)
# ----------------------------------------------------------------------------------------------

class _Plan:
    """Closed CCW plan of the outer die face. Segment k runs from pier k-1 to pier k, f in 0..1;
    d offsets the curve outwards (m). Segment 0 faces the front (-Y)."""

    def __init__(self, kind, R):
        self.kind, self.R = kind, R
        if kind == "octagon":
            self.n, self.sub = 8, 1
            self.L = 2 * R * tan(pi / 8)
        elif kind == "round":
            self.n = 8 if R >= 2.0 else (6 if R >= 1.3 else 4)
            self.sub = max(6, int(ceil(360.0 / self.n / 4.0)))
            self.L = 2 * pi * R / self.n
        else:
            self.n, self.sub = 4, 36
            q = 0.56 + 0.06 * clamp((1.6 - R) / 0.6, 0.0, 1.0)     # lobe radius / R (fatter lobes when small)
            self.rho = q * R
            self.c = R - self.rho                                  # distance of the lobe centres
            self.th0 = self._th(0.0)
            self.L = 2 * self.th0 * self.rho

    def cusp(self, d=0.0):
        """Quatrefoil: coordinate (along both lobe axes) of the cusp between two lobes at offset d."""
        rr = self.rho + d
        return (self.c + sqrt(max(2 * rr * rr - self.c * self.c, 1e-6))) / 2

    def _th(self, d):
        x = self.cusp(d)
        return atan2(x, x - self.c)                # half opening of the trimmed lobe arc

    def pt(self, k, f, d=0.0):
        R = self.R
        if self.kind == "octagon":
            a0 = radians(-112.5 + 45.0 * k)
            a1 = a0 + pi / 4
            rr = (R + d) / cos(pi / 8)
            x0, y0, x1, y1 = rr * cos(a0), rr * sin(a0), rr * cos(a1), rr * sin(a1)
            return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
        if self.kind == "round":
            a = -pi / 2 + (k - 0.5 + f) * 2 * pi / self.n
            return ((R + d) * cos(a), (R + d) * sin(a))
        rr = self.rho + d
        th = self._th(d)
        ax = radians(-90.0 + 90.0 * k)
        a = ax + min(th, max(-th, (2 * f - 1) * self.th0))
        return (self.c * cos(ax) + rr * cos(a), self.c * sin(ax) + rr * sin(a))

    def normal(self, k, f):
        e = 1e-3
        f0, f1 = max(0.02, f - e), min(0.98, f + e)
        a, b = self.pt(k, f0), self.pt(k, f1)
        tx, ty = b[0] - a[0], b[1] - a[1]
        l = hypot(tx, ty) or 1.0
        return (ty / l, -tx / l)

    def outline(self, d, per=None):
        per = per or max(self.sub, 4)
        return [self.pt(k, i / per, d) for k in range(self.n) for i in range(per)]

    def pier_angle(self, k):
        """Plan angle (deg) of pier k (the junction between segment k and k+1)."""
        if self.kind == "octagon":
            return -67.5 + 45.0 * k
        if self.kind == "round":
            return -90.0 + (k + 0.5) * 360.0 / self.n
        return -45.0 + 90.0 * k

    def offs(self, co):
        """Signed offset d of world points (N,3) from the plan curve (numpy; < 0 = inside)."""
        x, y = co[:, 0], co[:, 1]
        if self.kind == "octagon":
            d = np.full(len(co), -1e9)
            for i in range(8):
                a = radians(-90.0 + 45.0 * i)
                d = np.maximum(d, x * cos(a) + y * sin(a))
            return d - self.R
        if self.kind == "round":
            return np.hypot(x, y) - self.R
        d = np.full(len(co), 1e9)
        for k in range(4):
            ax = radians(-90.0 + 90.0 * k)
            d = np.minimum(d, np.hypot(x - self.c * cos(ax), y - self.c * sin(ax)) - self.rho)
        return d


# ----------------------------------------------------------------------------------------------
# masonry builders
# ----------------------------------------------------------------------------------------------

def _loft(name, rings, material=None, sm=35.0, along_u=True, caps=(True, True), skip=()):
    """Closed solid through a list of rings (each a list of xyz, same count), end-capped. UVs are
    face-aligned meters: one axis runs along the loft, the other around the section (along_u: the
    loft direction is U - stones lofted along the wall; else V - blocks stacked upwards).
    caps = (start, end) to build; skip = section edges j (ring point j -> j + 1) left open: what a
    neighbouring stone or the water hides (levels only)."""
    n, m = len(rings), len(rings[0])
    P = np.array(rings, dtype=float)
    sa = np.concatenate([[0.0], np.cumsum(np.linalg.norm(P[1:] - P[:-1], axis=2).mean(axis=1))])
    mid = P[n // 2]
    sb = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.roll(mid, -1, 0) - mid, axis=1))])
    _UVN[0] += 1
    ou, ov = ((_UVN[0] * 0.6180339) % 1.0) * 3.0, ((_UVN[0] * 0.3819660) % 1.0) * 3.0
    verts = [tuple(q) for q in P.reshape(-1, 3)]
    faces, fuv = [], []
    for i in range(n - 1):
        a, b = i * m, (i + 1) * m
        for j in range(m):
            if j in skip:
                continue
            j2 = (j + 1) % m
            faces.append((a + j, a + j2, b + j2, b + j))
            q = ((sa[i], sb[j]), (sa[i], sb[j + 1]), (sa[i + 1], sb[j + 1]), (sa[i + 1], sb[j]))
            fuv.append([(u + ou, v + ov) for u, v in q] if along_u else [(v + ou, u + ov) for u, v in q])

    def cap_uv(ring):
        if np.ptp(ring[:, 2]) < 1e-4:
            return [(float(x) + ou, float(y) + ov) for x, y, _ in ring]
        dxy = ring[:, :2] - ring[0, :2]
        far = dxy[int(np.argmax((dxy ** 2).sum(1)))]
        l = float(np.hypot(far[0], far[1]))
        ax = far / l if l > 1e-6 else np.array([1.0, 0.0])
        return [(float(dxy[i] @ ax) + ou, float(ring[i, 2]) + ov) for i in range(len(ring))]

    c0, c1 = cap_uv(P[0]), cap_uv(P[-1])
    if caps[0]:
        faces.append(tuple(range(m - 1, -1, -1)))
        fuv.append([c0[j] for j in range(m - 1, -1, -1)])
    if caps[1]:
        faces.append(tuple((n - 1) * m + j for j in range(m)))
        fuv.append([c1[j] for j in range(m)])
    o = _mesh_uv(name, verts, faces, fuv)
    if skip:
        # strips with nothing between them: turned each on its own, the one that is hollow towards the viewer (the
        # pool side of a die on a round plan) ends up facing into the stone and the far wall of the pool is a hole.
        # They face the way the closed stone does.
        A, B = P[:-1], P[1:]
        A2, B2 = np.roll(A, -1, 1), np.roll(B, -1, 1)
        out = (np.cross(B2 - A, B - A2) * ((A + A2 + B + B2) / 4.0 - P.mean(axis=(0, 1)))).sum()
        if out < 0:
            o.data.flip_normals()
    else:
        recalc_normals(o)
    smooth(o, sm)
    if material is not None:
        assign(o, material)
    return o


def _inset(prof, c):
    """Mitre-inset a closed 2D polygon by c (m)."""
    P = np.array(prof, dtype=float)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    sgn = 1.0 if area > 0 else -1.0

    def nr(e):
        t = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        return np.stack([-t[:, 1], t[:, 0]], 1) * sgn

    n1, n2 = nr(P - np.roll(P, 1, 0)), nr(np.roll(P, -1, 0) - P)
    m = n1 + n2
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    k = 1.0 / np.maximum((m * n1).sum(1), 0.5)
    return [tuple(q) for q in (P + m * (k * c)[:, None])]


def _stone(name, plan, k, f0, f1, prof=None, prof_fn=None, breaks=(), cham=0.006, material=None, sm=35.0,
           chips=(), tone=0.0, knocks=(None, None), jw=0.0028, hide=None):
    """One stone: a closed (offset, z) profile lofted along plan segment k from f0 to f1. Its ends are cut back
    so neighbouring stones meet in a real, recessed joint: an open channel 2 * jw wide and `cham` deep whose bed
    lies dark with dirt, with a dirty margin on the faces either side. prof_fn(f) may vary the profile (sunk
    panels); breaks = extra stations (f) where the profile has creases.
    chips = [(station m from f0, half length m, profile index, depth m, reach m, fresh)]: spalls knocked out
    of an arris - a short V-notch whose deepest point sits off-centre. knocks = (start, end), each None or
    (profile index, depth m, reach m, length m, fresh): the corner of an arris knocked off at the joint.
    tone = the block's own shade (> 0 darker = grime, < 0 paler = wear).
    Levels: no spalls, the profile and the stations along a curve only as fine as the level shows, no end caps,
    and none of the profile edges hide(pa, pb) says nothing can see (bedded on a course, under the water).
    lod 1 keeps the joint as a plain V; at lod 2 the stones butt and their tones carry the coursing."""
    Lv = lod()
    if Lv:
        chips, knocks = (), (None, None)
        if prof is not None:
            prof = lod_path(prof, closed=True)
    L = plan.L
    g, gj, sl = 0.0004 / L, jw / L, 0.002 / L
    a, b = f0 + gj + sl, f1 - gj - sl
    nsub = max(1, int(ceil((f1 - f0) * plan.sub - 1e-6)))
    if Lv and plan.kind != "octagon":                     # the facets a circle of that radius needs at this level
        rho = plan.R if plan.kind == "round" else plan.rho
        nsub = max(1, min(nsub, int(ceil((f1 - f0) * L / (2 * pi * rho) * lod_segs(2048, rho) - 1e-6))))
    cand = [f0 + (f1 - f0) * i / nsub for i in range(1, nsub)]
    pf0 = prof_fn if prof_fn is not None else (lambda f: prof)
    cmap = {}                                             # station -> (profile index, depth, reach, fresh)
    long_ = (b - a) * L > 0.22
    ma, mb = a + 0.03 / L, b - 0.03 / L                   # dirt margin either side of a joint
    if long_:
        if knocks[0]:
            idx, dep_, reach, ln, fresh = knocks[0]
            cmap[a] = (idx, dep_, reach, fresh)
            ma = a + min(ln, 0.3 * (b - a) * L) / L
        if knocks[1]:
            idx, dep_, reach, ln, fresh = knocks[1]
            cmap[b] = (idx, dep_, reach, fresh)
            mb = b - min(ln, 0.3 * (b - a) * L) / L
        if not Lv:
            cmap[ma] = cmap[mb] = None
    lo = (ma if long_ else a) + 0.012 / L
    hi = (mb if long_ else b) - 0.012 / L
    for (sm_, hl, idx, dep_, reach, fresh) in chips:
        fc, w = f0 + sm_ / L, hl / L
        if fc - w < lo or fc + w > hi:
            continue
        cmap[fc - w] = None
        cmap[fc - 0.22 * w] = (idx, dep_, reach, fresh)
        cmap[fc + 0.38 * w] = (idx, 0.55 * dep_, reach, fresh)
        cmap[fc + w] = None
        lo = fc + w + 0.012 / L
    gap = 0.02 / L
    inner = [x for x in cmap if a < x < b]
    br = sorted([x for x in breaks if a + gap < x < b - gap and all(abs(x - y) > 0.004 / L for y in inner)] + inner)
    fs = br + [x for x in cand if a + gap < x < b - gap and all(abs(x - y) > gap for y in br)]
    fs = [a] + sorted(fs) + [b]

    def pf(f):
        pr = pf0(f)
        ch = cmap.get(f)
        if not ch:
            return pr
        idx, dep_, reach, _ = ch
        idx = idx % len(pr)
        cx, cz = sum(q[0] for q in pr) / len(pr), sum(q[1] for q in pr) / len(pr)
        px, pz = pr[idx]
        out = []
        for (d, z) in pr:
            wq = max(0.0, 1.0 - hypot(d - px, z - pz) / reach) ** 1.5
            l = hypot(cx - d, cz - z) or 1.0
            out.append((d + (cx - d) / l * dep_ * wq, z + (cz - z) / l * dep_ * wq))
        return out

    def ring(f, pr):
        out = []
        for d, z in pr:
            x, y = plan.pt(k, f, d)
            out.append((x, y, z))
        return out

    if Lv:
        pa, pb = pf(a), pf(b)
        g = 0.0                                           # no end caps: neighbours end on the very same station
        if Lv == 1:
            rings = [ring(f0 + g, _inset(pa, cham))] + [ring(f, pf(f)) for f in fs] + [ring(f1 - g, _inset(pb, cham))]
        else:
            rings = [ring(f0 + g, pa)] + [ring(f, pf(f)) for f in fs[1:-1]] + [ring(f1 - g, pb)]
        skip = [j for j in range(len(pa)) if hide is not None and hide(pa[j], pa[(j + 1) % len(pa)])]
        o = _loft(name, rings, material, sm, caps=(False, False), skip=skip)
        paint(o, grime=max(float(tone), 0.0), wear=max(-float(tone), 0.0))   # (the level takes its masks from lod 0)
        return o
    ia, ib = _inset(pf(a), cham), _inset(pf(b), cham)
    rings = [ring(f0 + g, ia), ring(f0 + gj, ia)]         # the bed of the joint channel and its side
    rings += [ring(f, pf(f)) for f in fs]
    rings += [ring(f1 - gj, ib), ring(f1 - g, ib)]
    o = _loft(name, rings, material, sm)
    m = len(rings[0])
    dark, pale = max(float(tone), 0.0), max(-float(tone), 0.0)
    jm = np.full(len(rings) * m, dark)
    wm = np.full(len(rings) * m, pale)
    jm[:2 * m] = jm[-2 * m:] = 0.92                       # dirt lying in the joints: they read as one dark line
    wm[:2 * m] = wm[-2 * m:] = 0.0
    for i, f in enumerate(fs):
        r0_ = (i + 2) * m
        if i == 0 or i == len(fs) - 1:                    # the arris of the joint: dirt creeps out of it
            jm[r0_:r0_ + m] = min(0.8, dark + 0.42)
            wm[r0_:r0_ + m] = 0.0
        ch = cmap.get(f)
        if ch:                                           # the broken face: pale where fresh, dirty where old
            idx, dep_, reach, fresh = ch
            pr = pf0(f)
            px, pz = pr[idx % len(pr)]
            for j, (d, z) in enumerate(pr):
                wq = max(0.0, 1.0 - hypot(d - px, z - pz) / reach)
                if wq > 0:
                    if fresh:
                        wm[r0_ + j] = max(wm[r0_ + j], 0.8 * wq ** 0.5)
                        jm[r0_ + j] *= 1.0 - wq ** 0.5
                    else:
                        jm[r0_ + j] = max(jm[r0_ + j], 0.6 * wq ** 0.5)
    paint(o, grime=lambda co, n: jm if len(co) == len(jm) else np.full(len(co), dark),
          wear=lambda co, n: wm if len(co) == len(wm) else np.full(len(co), pale))
    return o


def _chips(r, length, idx_choices, p_each=0.5, n_max=2, dep=(0.006, 0.016)):
    """Seeded spalls along one stone of the given length: [(station, half length, profile index, depth, reach, fresh)]."""
    out = []
    n = sum(1 for _ in range(n_max) if r.u() < p_each)
    for j in range(n):
        s0 = length * (j + r.u(0.12, 0.88)) / n
        d = r.u(*dep)
        out.append((s0, r.u(0.022, 0.05) + 1.5 * d, idx_choices[r.randint(0, len(idx_choices) - 1)], d, r.u(0.022, 0.04), r.u() < 0.45))
    return sorted(out)


def _knock(r, idx_choices, p=0.3, dep=(0.007, 0.018)):
    """Seeded knock at one end of a stone (the corner of an arris broken off at the joint), or None."""
    if r.u() > p:
        return None
    d = r.u(*dep)
    return (idx_choices[r.randint(0, len(idx_choices) - 1)], d, r.u(0.024, 0.045) + 0.5 * d, r.u(0.035, 0.08) + 2.0 * d, r.u() < 0.4)


def _offset_outline(pts, e):
    """Mitre-offset a closed CCW 2D polygon outwards by e."""
    P = np.array(pts, dtype=float)

    def nr(v):
        t = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        return np.stack([t[:, 1], -t[:, 0]], 1)

    n1, n2 = nr(P - np.roll(P, 1, 0)), nr(np.roll(P, -1, 0) - P)
    m = n1 + n2
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    k = 1.0 / np.maximum((m * n1).sum(1), 0.45)
    return P + m * (k * e)[:, None]


def _stack(name, outline, profile, material=None, sm=30.0, caps=(True, True)):
    """'Polygon lathe': a moulded block over an arbitrary plan outline; profile = [(offset, z)]."""
    rings = [[(x, y, z) for x, y in _offset_outline(outline, e)] for e, z in profile]
    return _loft(name, rings, material, sm, along_u=False, caps=caps)


def _polar(name, phis, edges, r0, r_noms, z_fn, material, sm=80.0):
    """Sheet between the circle r0 and a closed outline (edges[j] = radius at angle phis[j], CCW).
    r_noms = ring radii measured along the shortest ray (r0 .. min(edges)); rings morph from the
    circle to the outline. z_fn(r, phi, e) -> z with e = distance to the edge along the ray.
    Normals up, planar meter UVs."""
    re = min(edges)
    m = len(phis)
    verts, faces = [], []
    for rn in r_noms:
        t = (rn - r0) / max(re - r0, 1e-6)
        for j in range(m):
            r = r0 + (edges[j] - r0) * t
            verts.append((r * cos(phis[j]), r * sin(phis[j]), z_fn(r, phis[j], edges[j] - r)))
    for i in range(len(r_noms) - 1):
        a, b = i * m, (i + 1) * m
        for j in range(m):
            j2 = (j + 1) % m
            faces.append((a + j, b + j, b + j2, a + j2))
    o = mesh_object(name, verts, faces)
    uv_planar(o, (1, 0, 0), (0, 1, 0))
    smooth(o, sm)
    assign(o, material)
    return o


def _water_marks(o, hits):
    """Shader hints of the `water` material, painted on a (flat) water surface from the landings on it.
    hits = [(x0, y0, x1, y1, strength, foam radius)]: every landing is a segment (the foot of a sheet) or a point.
    wear = agitation: `strength` at the landing, falling linearly to 0 over 1.2 m (the material runs rings
    outwards along its gradient and roughens the water with it); special = foam: 1 within the foam radius of
    the landing, dying away over about as much again (the material breaks that edge into lace)."""
    me = o.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ag, fo = np.zeros(nv), np.zeros(nv)
    for (x0, y0, x1, y1, k, rf) in hits:
        dx, dy = x1 - x0, y1 - y0
        l2 = dx * dx + dy * dy
        tq = np.clip(((co[:, 0] - x0) * dx + (co[:, 1] - y0) * dy) / l2, 0.0, 1.0) if l2 > 1e-9 else 0.0
        d = np.hypot(co[:, 0] - x0 - tq * dx, co[:, 1] - y0 - tq * dy)
        ag = np.maximum(ag, np.clip(k - d / 1.2, 0.0, 1.0))
        fo = np.maximum(fo, np.clip(1.0 - (d - 0.55 * rf) / (1.25 * rf + 0.03), 0.0, 1.0) ** 1.4)
    _vmask(o, "wear", ag)
    _vmask(o, "special", fo)


def _water_glow(o, pts, rad):
    """grime > 0.5 on a `water` surface = the light of a submerged lamp standing in the water (needs glow on the
    material): strongest over the lamp, gone `rad` away."""
    me = o.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    g = np.zeros(nv)
    for (x, y) in pts:
        g = np.maximum(g, np.clip(1.0 - np.hypot(co[:, 0] - x, co[:, 1] - y) / rad, 0.0, 1.0))
    _vmask(o, "grime", np.where(g > 0, 0.5 + 0.5 * g, 0.0))


# ----------------------------------------------------------------------------------------------
# kerb steps, basin wall, piers
# ----------------------------------------------------------------------------------------------

_MOULD = [(0.0, 0.0), (0.012, -0.012), (0.030, -0.012), (0.050, -0.032), (0.085, -0.032), (0.105, -0.016)]


# the same panel per level: lod 1 without its flats, lod 2 one bevel down to the field (the field stays where it is:
# tablet and paterae are set on it)
_MOULD_LOD = (_MOULD, [(0.0, 0.0), (0.012, -0.012), (0.050, -0.032), (0.105, -0.016)], [(0.0, 0.0), (0.035, -0.016)])


def _mould(t, M=_MOULD):
    """Depth of the sunk, moulded and fielded panel at distance t inside its border."""
    if t <= 0.0:
        return 0.0
    for (t0, d0), (t1, d1) in zip(M, M[1:]):
        if t <= t1:
            return d0 + (d1 - d0) * (t - t0) / (t1 - t0)
    return M[-1][1]


def _build_steps(plan, n_steps, rng, T, z_pod, rise, d_land, going, material):
    objs = []
    r = rng.fork("steps")
    rc = rng.fork("step_chips")
    for c in range(n_steps):
        z1 = z_pod - c * rise
        z0 = z1 - rise - 0.02
        d_out = d_land + c * going
        d_in = (-T - 0.08) if c == 0 else d_out - going - 0.08
        Ld = plan.L * (plan.R + d_out) / plan.R
        n = max(1, int(Ld / 1.15 + 0.5)) + (c % 2)
        for k in range(plan.n):
            cuts = [0.0] + [(i + r.u(-0.13, 0.13)) / n for i in range(1, n)] + [1.0]
            for i in range(n):
                dz, dd = r.u(-0.002, 0.002), r.u(-0.003, 0.0)
                if lod():
                    # the stones of a course share one profile, so with no end caps they still close on each other:
                    # a tread 2 mm proud of its neighbour is a slot into the hollow stone for a grazing sight line
                    dz, dd = 0.0, -0.0015
                do, zt = d_out + dd, z1 + dz
                prof = [(d_in, z0), (do - 0.006, z0), (do, z0 + 0.008), (do, zt - 0.030), (do - 0.004, zt - 0.013),
                        (do - 0.014, zt - 0.003), (do - 0.032, zt), (d_in, zt + 0.003)]
                ls = (cuts[i + 1] - cuts[i]) * Ld
                # (levels: the bed and the back, under the course above / the water, are left open)
                hid = lambda a_, b_, z0=z0, d_in=d_in: max(a_[1], b_[1]) <= z0 + 0.01 or max(a_[0], b_[0]) <= d_in + 1e-6
                objs.append(_stone("step", plan, k, cuts[i], cuts[i + 1], prof, cham=0.008, material=material, hide=hid,
                                   chips=[(q[0] * plan.L / Ld,) + q[1:] for q in _chips(rc, ls, (5, 5, 4, 6), 0.6, 3, (0.009, 0.024))],
                                   tone=rc.u(-0.22, 0.36), jw=rc.u(0.003, 0.0045),
                                   knocks=(_knock(rc, (5, 4, 6), 0.4, (0.01, 0.024)), _knock(rc, (5, 4, 6), 0.4, (0.01, 0.024)))))
    return objs


def _build_wall(plan, rng, T, z_pod, hp, hd, hc, fa, material):
    """Plinth course, panelled die stones and coping between the piers. Returns (objs, panel anchors)."""
    objs, anchors = [], []
    r = rng.fork("wall")
    rc = rng.fork("wall_chips")
    L = plan.L
    z_d0, z_c0 = z_pod + hp, z_pod + hp + hd
    z_rim = z_c0 + hc
    run = (1 - 2 * fa) * L
    n_pan = max(1, int(run / 1.55 + 0.5))
    oi = min(0.05, 0.25 * T)
    MD = _MOULD_LOD[lod()]
    for k in range(plan.n):
        # plinth course
        n = max(1, int(run / 1.3 + 0.5))
        cuts = [fa] + [fa + (1 - 2 * fa) * (i + r.u(-0.12, 0.12)) / n for i in range(1, n)] + [1 - fa]
        for i in range(n):
            dd = r.u(-0.002, 0.002)
            z0, z1 = z_pod - 0.01, z_d0 + r.u(-0.001, 0.001)
            prof = [(-T, z0), (0.05 + dd, z0), (0.05 + dd, z0 + 0.075)]
            prof += _q(prof[-1], (0.05 + dd, z0 + 0.105), (0.03 + dd, z0 + 0.112), 3)
            prof += [(0.022 + dd, z1 - 0.022)]
            prof += _q(prof[-1], (0.006 + dd, z1 - 0.018), (0.004 + dd, z1), 3)
            prof += [(-T, z1)]
            # (levels: open where it beds on the step and inside the pool, below the water; the top stays - the die
            # stands a few mm back from its edge, and with the ledge gone one looks down into the wall there)
            hid = lambda a_, b_, z0=z0: max(a_[1], b_[1]) <= z0 + 1e-6 or max(a_[0], b_[0]) <= -T + 1e-6
            objs.append(_stone("plinth", plan, k, cuts[i], cuts[i + 1], prof, cham=0.006, material=material, hide=hid,
                               chips=_chips(rc, (cuts[i + 1] - cuts[i]) * L, (2, 2, 1), 0.45, 2, (0.006, 0.016)),
                               tone=rc.u(-0.1, 0.38), jw=rc.u(0.0022, 0.0034),
                               knocks=(_knock(rc, (2, 1), 0.25), _knock(rc, (2, 1), 0.25))))
        # die stones, one sunk panel each
        mx, mz = 0.10, 0.05
        cuts = [fa + (1 - 2 * fa) * i / n_pan for i in range(n_pan + 1)]
        for i in range(n_pan):
            fA, fB = cuts[i], cuts[i + 1]
            dd = r.u(-0.002, 0.002)
            sa, sb = fA * L + mx, fB * L - mx
            za, zb = z_d0 + mz, z_c0 - mz
            zs = [z_d0] + [za + t for t, _ in MD] + [zb - t for t, _ in reversed(MD)] + [z_c0]
            breaks = [(sa + t) / L for t, _ in MD] + [(sb - t) / L for t, _ in MD]

            def pf(f, sa=sa, sb=sb, za=za, zb=zb, zs=zs, dd=dd):
                s = f * L
                ts = min(s - sa, sb - s)
                return [(_mould(min(ts, z - za, zb - z), MD) + dd, z) for z in zs] + [(-T, zs[-1]), (-T, zs[0])]

            # (levels: open under the coping and on the plinth)
            hid = lambda a_, b_: min(a_[1], b_[1]) >= z_c0 - 1e-6 or max(a_[1], b_[1]) <= z_d0 + 1e-6
            objs.append(_stone("die", plan, k, fA, fB, prof_fn=pf, breaks=breaks, cham=0.006, material=material, hide=hid,
                               tone=rc.u(-0.2, 0.32), jw=rc.u(0.002, 0.003)))
            fc = (fA + fB) / 2
            anchors.append((plan.pt(k, fc, _MOULD[-1][1] + dd - 0.002), plan.normal(k, fc), (za + zb) / 2,
                            (sb - sa), zb - za))
        # coping
        n = n_pan + 1
        cuts = [fa] + [fa + (1 - 2 * fa) * (i + r.u(-0.15, 0.15)) / n for i in range(1, n)] + [1 - fa]
        for i in range(n):
            dd, dz = r.u(-0.003, 0.003), r.u(-0.002, 0.002)
            z0, z1 = z_c0, z_rim + dz
            prof = [(-T - oi, z0 + 0.012), (-T - oi + 0.012, z0), (0.012 + dd, z0)]
            if lod() >= 2:
                # far: the underside lies flat on the die. With the small chamfer of the inner arris simplified away
                # it rose from the outer face to the inner one, a slot over the open top of the die
                prof = [(-T - oi, z0), (0.012 + dd, z0)]
            prof += _q(prof[-1], (0.016 + dd, z0 + 0.024), (0.036 + dd, z0 + 0.030), 3)
            prof += [(0.044 + dd, z0 + 0.032), (0.044 + dd, z0 + 0.046)]
            prof += _c3(prof[-1], (0.082 + dd, z0 + 0.050), (0.088 + dd, z1 - 0.012), (0.052 + dd, z1 - 0.002), 6)
            prof += [(0.02 + dd, z1 + 0.002), (-T - oi + 0.035, z1 + 0.006)]
            prof += _q(prof[-1], (-T - oi, z1 + 0.004), (-T - oi, z1 - 0.03), 3)
            nose = max(range(len(prof)), key=lambda q: prof[q][0])
            arr = (nose, nose + 1, nose - 1, len(prof) - 2, len(prof) - 3, nose)
            objs.append(_stone("coping", plan, k, cuts[i], cuts[i + 1], prof, cham=0.007, material=material,
                               chips=_chips(rc, (cuts[i + 1] - cuts[i]) * L, arr, 0.62, 3, (0.009, 0.026)),
                               tone=rc.u(-0.22, 0.28), jw=rc.u(0.0024, 0.0038),
                               knocks=(_knock(rc, arr, 0.38, (0.008, 0.02)), _knock(rc, arr, 0.38, (0.008, 0.02)))))
    return objs, anchors


def _pier_outline(plan, k, T, hw, po, pin):
    if plan.kind == "quatrefoil":
        a0, a1 = radians(-90.0 + 90.0 * k), radians(90.0 * k)
        u0, u1 = (cos(a0), sin(a0)), (cos(a1), sin(a1))
        lo, hi = plan.cusp(-T) - pin, plan.cusp(0.0) + po
        return [(s * u0[0] + t * u1[0], s * u0[1] + t * u1[1]) for s, t in ((hi, lo), (hi, hi), (lo, hi), (lo, lo))]
    k2 = (k + 1) % plan.n
    fa = hw / plan.L
    return [plan.pt(k, 1 - fa, po), plan.pt(k, 1.0, po), plan.pt(k2, fa, po),
            plan.pt(k2, fa, -T - pin), plan.pt(k, 1.0, -T - pin), plan.pt(k, 1 - fa, -T - pin)]


def _build_piers(plan, T, z_pod, z_rim, hw, po, pin, ci, material):
    objs, tops = [], []
    zc = z_rim
    zj = (z_pod + 0.176, zc + 0.002)                      # bed joints: base block / shaft / cap
    jd = 0.007
    prof = [(0.035, z_pod - 0.01), (0.035, z_pod + 0.122), (0.028, z_pod + 0.134), (0.004, z_pod + 0.158),
            (0.0, z_pod + 0.165), (0.0, zj[0] - 0.004), (-jd, zj[0] - 0.0028), (-jd, zj[0] + 0.0028), (0.0, zj[0] + 0.004),
            (0.0, zj[1] - 0.004), (-jd, zj[1] - 0.0028), (-jd, zj[1] + 0.0028), (0.0, zj[1] + 0.004),
            (0.0, zc + 0.012), (0.012, zc + 0.018), (0.017, zc + 0.029), (0.012, zc + 0.040),
            (0.0, zc + 0.046), (0.004, zc + 0.060), (0.014, zc + 0.078), (0.032, zc + 0.088), (0.038, zc + 0.090),
            (0.038, zc + 0.114), (0.024, zc + 0.128), (-ci, zc + 0.150)]
    prof = lod_path(prof)
    for k in range(plan.n):
        out = _pier_outline(plan, k, T, hw, po, pin)
        A = sum(out[i][0] * out[(i + 1) % len(out)][1] - out[(i + 1) % len(out)][0] * out[i][1] for i in range(len(out)))
        if A < 0:
            out = out[::-1]
        objs.append(_stack("pier", out, prof, material, caps=(lod() == 0, True)))
        cx = sum(q[0] for q in out) / len(out)
        cy = sum(q[1] for q in out) / len(out)
        tn = [0.42 * _h1(cx, cy, 1.7 + q) - 0.12 for q in range(3)]      # base block, shaft and cap: each its own shade

        def pier_tone(co, n, tn=tn, sgn=1.0):
            z = co[:, 2]
            t = np.where(z < zj[0], tn[0], np.where(z < zj[1], tn[1], tn[2]))
            return np.clip(sgn * t, 0, 1)

        def pier_bed(co, n):
            z = co[:, 2]
            dz = np.minimum(np.abs(z - zj[0]), np.abs(z - zj[1]))
            return np.maximum(0.92 * (dz < 0.0034), 0.42 * (dz < 0.0046))

        paint(objs[-1], grime=pier_tone, wear=lambda co, n, f=pier_tone: f(co, n, sgn=-1.0))
        paint(objs[-1], grime=pier_bed)
        tops.append((cx, cy, zc + 0.150))
    return objs, tops


# ----------------------------------------------------------------------------------------------
# central stem: pedestal, balusters, scalloped bowls, finial
# ----------------------------------------------------------------------------------------------

def _lobe_fn(n, phase):
    if n <= 0:
        return lambda phi: 1.0
    k = n / (2 * pi)
    return lambda phi: abs(sin(pi * (((phi - phase) * k) % 1.0))) ** 0.6


def _stem(name, z0, z1, rf, rb, rn, rt, gad, material, seg=72):
    H = z1 - z0

    def Z(f):
        return z0 + f * H

    pr = [(rf, Z(0.0)), (rf, Z(0.05)), (rf * 0.93, Z(0.058))]
    pr += _q(pr[-1], (rf * 1.06, Z(0.085)), (rf * 0.92, Z(0.112)), 5)
    pr += [(rf * 0.82, Z(0.118))]
    pr += _q(pr[-1], (0.64 * rb, Z(0.13)), (0.62 * rb, Z(0.185)), 4)
    pr += [(0.70 * rb, Z(0.190)), (0.75 * rb, Z(0.203)), (0.70 * rb, Z(0.216)), (0.62 * rb, Z(0.222))]
    pr += _c3(pr[-1], (1.02 * rb, Z(0.235)), (1.14 * rb, Z(0.40)), (0.82 * rb, Z(0.52)), 9)
    pr += _q(pr[-1], (0.58 * rb, Z(0.62)), (rn, Z(0.76)), 5)
    pr += [(rn * 1.22, Z(0.766)), (rn * 1.32, Z(0.782)), (rn * 1.22, Z(0.798)), (rn, Z(0.804))]
    pr += _q(pr[-1], (rn * 1.02, Z(0.90)), (rt * 0.88, Z(0.95)), 5)
    pr += [(rt * 0.88, Z(0.965)), (rt, Z(0.972)), (rt, Z(1.0))]
    Lv = lod()
    if gad > 0 and Lv == 1:
        seg = lod_exact(gad * 2)              # one ridge, one hollow: the gadroons stay in step with the facets
    o = lathe(name, pr, segments=seg, smooth_angle=38)
    if gad > 0 and Lv < 2:
        za, zb = Z(0.232), Z(0.56)

        def fn(co):
            if za < co.z < zb:
                w = sin(pi * (co.z - za) / (zb - za)) ** 0.7
                s = 1.0 + 0.085 * w * (abs(cos(atan2(co.y, co.x) * gad / 2)) - 0.55)
                co.x *= s
                co.y *= s
            return co

        transform_verts(o, fn)
    assign(o, material)
    return o


def _bowl_inner(Rb, zb, h, t):
    """Inner profile of a bowl, from just inside the lip crest down to the flat of the floor."""
    lip = zb + h
    pr = [(Rb - 0.45 * t, lip - 0.06 * t), (Rb - 0.72 * t, lip - 0.30 * t), (Rb - 0.85 * t, lip - 0.70 * t)]
    pr += _c3(pr[-1], (0.96 * Rb - t, zb + 0.34 * h + 0.5 * t), (0.68 * Rb, zb + t), (0.25 * Rb, zb + t), 9)
    return pr


def _lobe_warp(o, Rb, n, phase, A, dip):
    """Scallop a turned bowl: pinch the plan between the lobes, dip the lip at the lobe tips."""
    if n <= 0:
        return o
    lobe = _lobe_fn(n, phase)
    r_in, r_out = 0.30 * Rb, 0.88 * Rb

    def fn(co):
        r = hypot(co.x, co.y)
        if r > r_in:
            c = lobe(atan2(co.y, co.x))
            w = _ss(r_in, r_out, r)
            s = 1.0 - A * w * (1.0 - c)
            co.x *= s
            co.y *= s
            co.z -= dip * w * w * c * c
        return co

    transform_verts(o, fn)
    return o


def _bowl(name, Rb, zb, h, t, rc, n, phase, A, dip, material, mseg, zw=None):
    """zw = level of the water standing in it: at lod 1 / 2 the stone stops just below it (the floor is never seen)
    and is closed flat there (between the edge of the water and the lip a sight line gets under the water, and with
    the bowl left open it met the back of the belly); the lip gets 6 (8 for an odd lobe count) / 4 facets a lobe, so
    creases and tips stay on the facets."""
    Lv = lod()
    lip = zb + h
    pr = [(rc, zb - 0.004), (rc, zb)]
    pr += _c3(pr[-1], (0.66 * Rb, zb), (0.97 * Rb, zb + 0.30 * h), (Rb - 0.25 * t, lip - 1.55 * t), 10)
    pr += [(Rb - 0.03 * t, lip - 1.46 * t), (Rb - 0.03 * t, lip - 1.18 * t),
           (Rb + 0.22 * t, lip - 0.97 * t), (Rb + 0.36 * t, lip - 0.66 * t), (Rb + 0.37 * t, lip - 0.36 * t),
           (Rb + 0.23 * t, lip - 0.10 * t), (Rb - 0.08 * t, lip)]
    inner = _bowl_inner(Rb, zb, h, t)
    if Lv and zw is not None:
        cut = next((q for q, pt_ in enumerate(inner) if pt_[1] < zw - 0.015), len(inner) - 1)
        pr += inner[:cut + 1]
        pr += [(0.0, inner[cut][1])]
    else:
        pr += inner
        pr += [(0.0, zb + t)]
    seg = (n * mseg) if n > 0 else 72
    if Lv and n > 0:
        seg = lod_exact(n * ((6 if n % 2 == 0 else 8) if Lv == 1 else 4))
    o = lathe(name, pr, segments=seg, smooth_angle=42)
    _lobe_warp(o, Rb, n, phase, A, dip)
    assign(o, material)
    return o


def _radial_joints(o, angles, r_min, half=0.003, depth=0.006):
    """Cut the dry joints of a bowl built from sections: on each radial plane (through the axis, at `angles`)
    an open channel 2 * half wide and `depth` deep runs right round the section - under the bowl, over the lip
    roll and down the inside."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    for a in angles:
        rad, tng = Vector((cos(a), sin(a), 0.0)), Vector((-sin(a), cos(a), 0.0))
        for off in (-half, half, 0.0):
            fs = [f for f in bm.faces if f.calc_center_median().dot(rad) > 0.0]
            geom = fs + list({e for f in fs for e in f.edges}) + list({v for f in fs for v in f.verts})
            bmesh.ops.bisect_plane(bm, geom=geom, dist=2e-5, plane_co=tng * off, plane_no=tng)
        bm.normal_update()
        for v in bm.verts:
            if abs(v.co.dot(tng)) < 4e-5 and v.co.dot(rad) > r_min:
                v.co -= v.normal * depth
    bm.to_mesh(o.data)
    bm.free()
    return o


def _joint_angles(b):
    """Radial joints of a bowl: large bowls are built from sections of two lobes (six sections when plain),
    the joints lying in the creases between lobes; small ones are turned from one block."""
    if b["Rb"] < 0.55 or b["i"] > 1:
        return []
    n = b["n"]
    if n <= 0:
        cnt = 6 if b["i"] == 0 else 4
        return [radians(15.0) + 2 * pi * j / cnt for j in range(cnt)]
    step = 2 if n >= 8 else 1
    return [b["phase"] + 2 * pi * j / n for j in range(0, n, step)]


def _bowl_bed(name, Rb, zb, h, t, r0, n, phase, A, dip, z_top, material, mseg):
    """The bowl floor as seen through the water: a skin 4 mm inside the stone, up to the waterline."""
    inner = _bowl_inner(Rb, zb, h, t)[1:]
    pr = [(r0, zb + t + 0.004)] + [(r - 0.004, z + 0.003) for r, z in reversed(inner)]
    pr = [(r, min(z, z_top - 0.002)) for r, z in pr]
    seg = (n * mseg) if n > 0 else 72
    o = lathe(name, pr, segments=seg, smooth_angle=60)
    _lobe_warp(o, Rb, n, phase, A, dip)
    assign(o, material)
    return o


# V of falling water is "time-like": the `water_fall` material scrolls its pattern down V at _V_REF units/s, so
# advancing V by ds * _V_REF / speed makes the pattern travel with the water (never slower than _V_MIN m/s).
_V_REF, _V_MIN = 2.9, 1.2


# Falling water and standing water are both see-through and both drawn: where a sheet, a jet or a piece of froth lay
# on a water surface (the film over the lip ran 1.5 mm above the bowl's water for a hand's width, the feet of the
# sheets and the froth mounds ended in it) the two fought for the depth buffer. Nothing that falls comes nearer
# than 4 mm to the surface it falls on (kept with half a millimetre to spare), and the film starts where the
# bowl's water ends.
_W_CLEAR = 0.0045


def _fall_time(drop, vd):
    return (-vd + sqrt(vd * vd + 2 * G * max(0.02, drop))) / G


def _sheet_mesh(name, b, z_end, A, material, K=12, z_min=None):
    """All the falling water of one bowl as one mesh. Every spill point sheds ONE thin, continuous sheet: a glassy
    film drawn over the lip roll that leaves it on a ballistic curve, necks in under surface tension as it
    speeds up, and reaches the water below still in one piece - only its two edges and its foot fray. It has no
    thickness (double-sided material) and stays all but flat across its width: the middle is thrown a little
    further than the edges and a few faint flutes run down it. Tearing is the material's job; the mesh says
    where. UV: U across (m), V along the flow measured up from the foot, so the `water_fall` shader scrolls
    down the fall; masks: wear = break-up (glassy at the lip, rising towards the foot and the two edges),
    special = foam (only where the sheet strikes the water) - see _mesh_vuv.
    Returns the landings [(r, angle, half width, sheet#)]: three points along the foot of every sheet.
    Levels: every other column and half the rows at lod 1, five columns and four rows at lod 2; the foot of
    EVERY column is still worked out (the landings must not move).
    z_min: no point of a sheet lies below this (the water it falls on, plus _W_CLEAR)."""
    Lv = lod()
    K = (K, 6, 4)[Lv]
    Rb, tw, lip, dip, n = b["Rb"], b["tw"], b["lip"], b["dip"], b["n"]
    lobe = _lobe_fn(n, b["phase"])
    # the film starts 2 mm outside the edge of the bowl's water (Rb - 0.30 tw where it spills; build() lays the
    # water no further out anywhere), on the line it always rose along: no part of it lies over that water
    kt = (0.20 * tw + 0.002) / (0.45 * tw)
    top = [(Rb - 0.30 * tw + 0.002, b["zw"] + 0.0015 + kt * (lip - dip + 0.004 - b["zw"] - 0.0015), 0.0), (Rb - 0.05 * tw, lip - dip + 0.004, 1.0),
           (Rb + 0.28 * tw, lip - dip - 0.08 * tw, 1.0), (Rb + 0.42 * tw, lip - dip - 0.36 * tw, 1.0)]
    r0, z0, vd = Rb + 0.43 * tw, lip - dip - 0.5 * tw, 0.35
    tt = _fall_time(z0 - z_end, vd)
    hk = clamp(sqrt(max(z0 - z_end, 0.0) / 0.8), 0.5, 1.0)          # a short fall arrives all but unbroken
    kv = _V_REF / _V_MIN
    V, F, UV, WR, SP, lands = [], [], [], [], [], []

    def P(phi, sl, r, z, wz):
        a = phi + sl / Rb
        if n > 0:
            cl = lobe(a)
            r -= A * Rb * (1.0 - cl)
            z += dip * (1.0 - cl * cl) * wz
        return (r * cos(a), r * sin(a), z), r, a

    def traj(vo, t):
        tau = tt * t
        return r0 + vo * tau, z0 - vd * tau - 0.5 * G * tau * tau

    def foot_len(vo, M=14):
        """V of the trajectory at fraction t, measured up from the foot: arc length weighted by 1 / speed (see
        _V_REF), so the pattern of the material runs slowly over the lip and is drawn out as the water speeds up"""
        pts = [traj(vo, q / M) for q in range(M + 1)]
        cum = [0.0]
        for q in range(M, 0, -1):
            sp = hypot(vo, vd + G * tt * (q - 0.5) / M)
            cum.append(cum[-1] + hypot(pts[q][0] - pts[q - 1][0], pts[q][1] - pts[q - 1][1]) * _V_REF / max(sp, _V_MIN))
        cum = cum[::-1]

        def f(t):
            x = clamp(t, 0.0, 1.0) * M
            q = min(int(x), M - 1)
            return cum[q] + (cum[q + 1] - cum[q]) * (x - q)
        return f

    tq = (0, 1, 2, 3) if Lv < 2 else (1, 3)
    nt = len(tq)
    for si, s in enumerate(b["sheets"]):
        phi, w, vs = s["phi"], s["w"], s["v_out"]
        lf = foot_len(vs)
        lt = [lf(0.0) + kv * hypot(top[3][0] - r0, top[3][1] - z0)]
        for q in (2, 1, 0):
            lt.append(lt[-1] + kv * hypot(top[q + 1][0] - top[q][0], top[q + 1][1] - top[q][1]))
        lt = lt[::-1]
        nc = len(s["dv"]) - 1
        nr = nt + K + 1
        base = len(V)
        foot = []
        cols = list(range(nc + 1)) if Lv == 0 else sorted(set(range(0, nc + 1, 2) if Lv == 1 else (int(round(nc * q / 4.0)) for q in range(5))) | {nc})
        for j in range(nc + 1):
            e = 2.0 * j / nc - 1.0                               # -1 .. 1 across the sheet
            ed = abs(e) ** 3
            vo = vs * (1.0 + 0.14 * (1.0 - e * e) + s["dv"][j])   # deepest, fastest water in the middle of the lobe
            if j not in cols:                                    # (levels) a column left out: only its foot
                r, z = traj(vo, 1.0)
                sl = 0.5 * e * w * (1.0 - s["neck"] * 1.0 ** 0.85 + s["wob"] * sin(s["ph"] + 7.0 * 1.0) * 1.0 * abs(e)) + s["drift"] * 1.0 * 1.0
                foot.append(P(phi, sl, r, z, 0.0)[1:])
                continue
            for q in tq:                                         # the film over the lip roll
                sl = 0.5 * e * w
                V.append(P(phi, sl, top[q][0], top[q][1], top[q][2])[0])
                UV.append((s["uo"] + sl, lt[q]))
                WR.append(0.02 + 0.12 * ed)
                SP.append(0.0)
            for q in range(K + 1):                               # the free fall
                t = q / K
                r, z = traj(vo, t)
                neck = 1.0 - s["neck"] * t ** 0.85
                wob = s["wob"] * sin(s["ph"] + 7.0 * t) * t * abs(e)
                sl = 0.5 * e * w * (neck + wob) + s["drift"] * t * t
                pt, rr, aa = P(phi, sl, r, z, (1 - t) ** 2)
                if z_min is not None and pt[2] < z_min:          # the foot stops short of the water it strikes
                    pt = (pt[0], pt[1], z_min)
                V.append(pt)
                UV.append((s["uo"] + sl, lf(t)))
                # one glassy sheet for the first half of the fall, rents and ropes below, threads and drops at the
                # foot; the two edges fray first
                brk = (0.02 + hk * (0.74 * t ** 1.6 + s["cw"][j] * t) + (0.08 + 0.32 * t) * ed * (0.3 + 0.7 * t)
                       + 0.16 * _ss(0.86, 1.0, t))
                WR.append(clamp(brk, 0.0, 1.0))
                SP.append(clamp(0.6 * _ss(0.87, 1.0, t) + 0.12 * ed * _ss(0.5, 1.0, t), 0.0, 1.0))
            foot.append((rr, aa))
        for j in range(len(cols) - 1):
            for q in range(nr - 1):
                i0 = base + j * nr + q
                F.append((i0, i0 + nr, i0 + nr + 1, i0 + 1))
        wf = 0.5 * w * (1.0 - s["neck"])
        for fq in (0.2, 0.5, 0.8):
            rr, aa = foot[int(round(fq * nc))]
            lands.append((rr, aa, max(0.3 * wf, 0.006), si))
    if not V:
        return None, lands
    o = _mesh_vuv(name, V, F, UV, WR, SP)
    assign(o, material)
    return o, lands


def _stream_mesh(name, path, w0, w1, material, uo=0.0, foam_len=0.1, z_min=None):
    """A solid jet as two thin crossed ribbons along `path` = [(x, y, z)] from the nozzle to the foot (a tube would
    show its far wall through the near one): one lying across the flow, a narrower one standing in the plane
    of the arc; both widen from w0 to w1. UV / masks as for the sheets: V up from the foot, wear = break-up
    rising down the jet, special = foam at the foot only."""
    Pp = np.array(path, dtype=float)
    K = len(Pp) - 1
    seg = np.linalg.norm(Pp[1:] - Pp[:-1], axis=1)
    below = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])       # arc length down to the foot
    tg = np.gradient(Pp, axis=0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
    hd = Pp[-1, :2] - Pp[0, :2]
    hd = hd / max(float(np.linalg.norm(hd)), 1e-6)
    side = np.array([-hd[1], hd[0], 0.0])                              # horizontal, across the flow
    V, F, UV, WR, SP = [], [], [], [], []
    es = (-1.0, 0.0, 1.0) if lod() == 0 else (-1.0, 1.0)
    ne = len(es)
    for (ax_k, wk, du) in ((0, 1.0, 0.0), (1, 0.62, 0.37)):
        base = len(V)
        for q in range(K + 1):
            t = q / K
            ax = side if ax_k == 0 else np.cross(tg[q], side)
            hw = 0.5 * wk * (w0 + (w1 - w0) * t ** 1.4)
            for e in es:
                v = Pp[q] + ax * (e * hw)
                V.append((float(v[0]), float(v[1]), float(v[2]) if z_min is None else max(float(v[2]), z_min)))
                UV.append((uo + du + e * hw, float(below[q])))
                WR.append(clamp(0.04 + 0.5 * t ** 1.3 + (0.22 if e else 0.0), 0.0, 0.95))
                SP.append(0.9 * _ss(foam_len, 0.0, float(below[q])))
        for q in range(K):
            for e in range(ne - 1):
                i0 = base + q * ne + e
                F.append((i0, i0 + 1, i0 + ne + 1, i0 + ne))
    o = _mesh_vuv(name, V, F, UV, WR, SP)
    assign(o, material)
    return o


class _Froth:
    """Collects the water thrown up where falling water lands - boiling froth mounds, the splash crown (one ragged
    band of spray) and ribbons of drops flung clear - into one `water_fall` mesh. The flat foam on the surface
    itself (boil, lacy edge, rings) is NOT modelled: it is painted on the water (_water_marks) and drawn by the
    `water` material. wear / special carry break-up / whiteness; V runs DOWN every throw, so the pattern of the
    material travels up and outwards with the spray."""

    def __init__(self, rng):
        self.v, self.f, self.uv, self.wr, self.sp, self.r = [], [], [], [], [], rng
        self.floor = None                    # nothing is added below this: the water being struck, plus _W_CLEAR

    def _add(self, p, uv, wr, sp):
        if self.floor is not None and p[2] < self.floor:
            p = (p[0], p[1], self.floor)
        self.v.append(p)
        self.uv.append(uv)
        self.wr.append(wr)
        self.sp.append(sp)
        return len(self.v) - 1

    def _uv(self, uo, vo, dx, dy, dz=0.0):
        return (uo + dx + 0.6 * dz, vo + 8.0 * (dy + 0.8 * dz))

    def skirt(self, x, y, z, ang, ra, rt, h, seg=14):
        """Splash crown: one band of spray thrown up and outwards all round a landing (semi-axes ra along `ang`,
        rt across; height h, ragged). White and nearly whole at its root, torn to threads half way up and to
        single drops at its upper edge."""
        r = self.r
        ca, sa = cos(ang), sin(ang)
        uo, vo = r.u(0, 3), r.u(0, 3)
        p1, p2, p3 = r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28)
        per = pi * (ra + rt)
        rows = ((1.0, 0.0, 0.93, 0.4), (1.1, 0.4, 0.96, 0.2), (1.24, 0.78, 0.98, 0.08), (1.4, 1.0, 1.0, 0.0))
        Lv = lod()
        if Lv:                                                   # fewer, larger pieces of the same crown
            seg = max(5, seg // 2) if Lv == 1 else 5
            rows = (rows[0], rows[2], rows[3]) if Lv == 1 else (rows[0], rows[3])
        b = len(self.v)
        for i in range(seg + 1):
            th = 2 * pi * i / seg
            kh = 0.72 + 0.3 * sin(3 * th + p1) * sin(2 * th + p2) + 0.16 * sin(7 * th + p3)
            for (fr, fz, wr, sp) in rows:
                kr = 1.0 + (fr - 1.0) * kh
                u, v = ra * kr * cos(th), rt * kr * sin(th)
                self._add((x + u * ca - v * sa, y + u * sa + v * ca, z + h * fz * kh),
                          (uo + 2.5 * per * i / seg, vo - 9.0 * h * kh * (fz + 0.5 * (fr - 1.0))), wr, sp)
        nr = len(rows)
        for i in range(seg):
            for q in range(nr - 1):
                i0 = b + i * nr + q
                self.f.append((i0, i0 + nr, i0 + nr + 1, i0 + 1))

    def spray(self, x, y, z, ang, v0, elev, w, K=6):
        """Drops flung clear of a landing: a thin ribbon along their ballistic arc (leaving (x, y, z) towards
        `ang` at v0, `elev` above the horizontal) that the material tears into separate drops travelling
        along it."""
        r = self.r
        ca, sa = cos(ang), sin(ang)
        vr, vz = v0 * cos(elev), v0 * sin(elev)
        tf = 2 * vz / G * r.u(0.7, 0.96)
        uo, vv = r.u(0, 3), r.u(0, 3)
        tilt = r.u(-0.7, 0.7)
        Lv = lod()
        if Lv >= 2:                                              # (drawn above, so the stream stays in step)
            return
        K = (K, 3)[Lv]
        b = len(self.v)
        prev = None
        for q in range(K + 1):
            t = tf * q / K
            pr, pz = vr * t, vz * t - 0.5 * G * t * t
            if prev is not None:
                vv -= hypot(pr - prev[0], pz - prev[1]) * _V_REF / max(hypot(vr, vz - G * t), 0.8)
            prev = (pr, pz)
            hw = 0.5 * w * (0.55 + 0.9 * q / K)
            for e in (-1.0, 1.0):
                self._add((x + pr * ca - sa * hw * e, y + pr * sa + ca * hw * e, z + pz + tilt * hw * e),
                          (uo + hw * e, vv), 0.93 + 0.07 * q / K, 0.15 * (1.0 - q / K) ** 2)
        for q in range(K):
            i0 = b + 2 * q
            self.f.append((i0, i0 + 1, i0 + 3, i0 + 2))

    def mound(self, x, y, z, rb, h, seg=8):
        """Low boiling mound of froth where a strand strikes the water."""
        r = self.r
        uo, vo = r.u(0, 3), r.u(0, 3)
        ph = r.u(0, 6.28)
        rings = ((1.0, -0.004, 0.96, 0.2), (0.78, 0.5, 0.84, 0.5), (0.45, 0.88, 0.76, 0.6))
        pts = []                                                 # every point is drawn at every level ...
        for fr, fz, wr, sp in rings:
            for i in range(seg):
                th = ph + 2 * pi * i / seg
                k = fr * r.u(0.92, 1.08)
                dx, dy, dz = rb * k * cos(th), rb * k * sin(th), h * fz * r.u(0.9, 1.05) if fz > 0 else fz
                pts.append(((x + dx, y + dy, z + dz), self._uv(uo, vo, dx, dy, dz), wr, sp))
        Lv = lod()
        if Lv >= 2:
            return
        isel = list(range(seg)) if Lv == 0 else list(range(0, seg, 2))     # ... and only some of them meshed
        jsel = (0, 1, 2) if Lv == 0 else (0, 2)
        b = len(self.v)
        for j in jsel:
            for i in isel:
                self._add(*pts[j * seg + i])
        top = self._add((x, y, z + h), self._uv(uo, vo, 0.0, 0.0, h), 0.7, 0.6)
        m = len(isel)
        for j in range(len(jsel) - 1):
            for i in range(m):
                i2 = (i + 1) % m
                self.f.append((b + j * m + i, b + j * m + i2, b + (j + 1) * m + i2, b + (j + 1) * m + i))
        j = len(jsel) - 1
        for i in range(m):
            self.f.append((b + j * m + i, b + j * m + (i + 1) % m, top))

    def mesh(self, name, material):
        if not self.v:
            return None
        o = _mesh_vuv(name, self.v, self.f, self.uv, self.wr, self.sp, sm=80.0)
        assign(o, material)
        return o


class _Moss:
    """Collects modelled moss cushions (low irregular domes, `moss` material) into one mesh."""

    def __init__(self, rng, cap=900):
        self.v, self.f, self.uv, self.r, self.n, self.cap = [], [], [], rng, 0, cap

    def tuft(self, x, y, z, ang, ra, rt, h, seg=7):
        """One cushion: ra along `ang`, rt across, h high, its skirt sunk 5 mm into the stone."""
        if self.n >= self.cap:
            return
        self.n += 1
        r = self.r
        ca, sa = cos(ang), sin(ang)
        ph = r.u(0, 6.28)
        uo, vo = r.u(0, 2), r.u(0, 2)
        b = len(self.v)
        uvs, vs = [], []
        for fr, fz in ((1.0, -0.005), (0.66, 0.7)):
            for i in range(seg):
                th = ph + 2 * pi * i / seg
                k = fr * r.u(0.78, 1.22)
                u, v = ra * k * cos(th), rt * k * sin(th)
                vs.append((x + u * ca - v * sa, y + u * sa + v * ca, z + (h * fz * r.u(0.8, 1.15) if fz > 0 else fz)))
                uvs.append((uo + 2.2 * u, vo + 2.2 * v))
        vs.append((x + r.u(-0.2, 0.2) * ra * ca, y + r.u(-0.2, 0.2) * ra * sa, z + h))
        uvs.append((uo, vo))
        Lv = lod()
        if Lv:                                # every cushion is drawn in full (the stream is shared with the layout)
            if Lv == 1:                       # ... and meshed as a low pyramid; too small to show at lod 2
                self.v += vs[:seg] + vs[-1:]
                for i in range(seg):
                    q = (i, (i + 1) % seg, 2 * seg)
                    self.f.append((b + q[0], b + q[1], b + seg))
                    self.uv.append([uvs[k] for k in q])
            return
        self.v += vs
        for i in range(seg):
            i2 = (i + 1) % seg
            q = (i, i2, seg + i2, seg + i)
            self.f.append(tuple(b + k for k in q))
            self.uv.append([uvs[k] for k in q])
            q = (seg + i, seg + i2, 2 * seg)
            self.f.append(tuple(b + k for k in q))
            self.uv.append([uvs[k] for k in q])

    def mesh(self, name, material):
        if not self.v:
            return None
        o = _mesh_uv(name, self.v, self.f, self.uv)
        smooth(o, 80)
        assign(o, material)
        return o


# ----------------------------------------------------------------------------------------------
# carved and cast ornament: lion masks, paterae, tablet, finial
# ----------------------------------------------------------------------------------------------

def _lion_face(x, y):
    """Relief of the lion's face (unit = mask radius) as (height, recess mask, highlight mask).
    Bold forms only - every feature spans at least two cells of the face grid."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def g(cx, cy, sx, sy):
        return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))

    def g2(cx, cy, sx, sy):
        return g(cx, cy, sx, sy) + g(-cx, cy, sx, sy)

    ax = np.abs(x)
    # head: broad skull above, narrower projecting muzzle below
    skull = 0.40 * np.sqrt(np.clip(1.0 - (x / 0.60) ** 2 - ((y - 0.08) / 0.56) ** 2, 0.0, None))
    muzz = 0.50 * np.sqrt(np.clip(1.0 - (x / 0.40) ** 2 - ((y + 0.36) / 0.42) ** 2, 0.0, None))
    face = np.maximum(skull, muzz)
    fm = _ssn(0.0, 0.12, face)
    yb = 0.24 + 0.42 * (ax - 0.08)                                     # brows rise outwards: the lion's frown
    brow = np.exp(-(((ax - 0.27) / 0.20) ** 2 + ((y - yb - 0.07) / 0.075) ** 2))
    eyes = np.exp(-(((ax - 0.25) / 0.105) ** 2 + ((y - yb + 0.07) / 0.062) ** 2))
    bridge = np.exp(-((x / (0.10 + 0.10 * np.clip(0.25 - y, 0.0, None))) ** 2)) * _ssn(0.48, 0.30, y) * _ssn(-0.22, -0.05, y)
    nose = g(0, -0.13, 0.17, 0.085)                                    # broad wedge of the nose
    nostr = g2(0.105, -0.185, 0.045, 0.035)
    pads = g2(0.17, -0.34, 0.16, 0.12)                                 # whisker pads
    cleft = g(0, -0.33, 0.03, 0.10)
    mouth = g(0, -0.55, 0.19, 0.075)                                   # open mouth round the pipe
    chin = g(0, -0.74, 0.17, 0.075)
    cheek = g2(0.36, -0.04, 0.11, 0.16)
    f = face + fm * (0.17 * brow - 0.26 * eyes + 0.13 * bridge + 0.20 * nose - 0.07 * nostr + 0.17 * pads
                     - 0.10 * cleft - 0.42 * mouth + 0.15 * chin + 0.07 * cheek - 0.05 * g(0, 0.50, 0.035, 0.12))
    dark = np.clip(fm * (1.3 * eyes + 1.3 * mouth + 0.9 * nostr + 0.5 * cleft), 0, 1)       # sockets, nostrils, mouth
    high = np.clip(fm * (0.25 + 0.9 * brow + 0.7 * bridge + nose + 0.8 * pads + 0.8 * chin + 0.5 * cheek), 0, 1)
    return np.maximum(0.0, f), dark, high


# the mane: two overlapping rows of ten pointed locks on a polar grid whose spokes follow the locks
_MANE_R = (0.40, 0.46, 0.56, 0.66, 0.73, 0.75, 0.82, 0.89, 0.95, 0.975, 1.0)
_MANE_A = {0: (0.10, 0.10, 0.0), 1: (0.37, 0.34, 0.0), 2: (0.41, 0.30, 0.0), 3: (0.41, 0.25, 0.2), 4: (0.37, 0.0, 0.55)}
_MANE_B = {4: (0.22, 0.17, 0.0), 5: (0.23, 0.16, 0.0), 6: (0.25, 0.13, 0.0), 7: (0.24, 0.10, 0.2), 8: (0.19, 0.05, 0.55),
           9: (0.045, 0.045, 0.0), 10: (0.0, 0.0, 0.0)}
_MANE_NS = 60


def _lion_mane(i, j):
    """Vertex (x, y, height, groove, crest) of the mane at spoke i, ring j (unit = mask radius)."""
    r = _MANE_R[j]
    th0 = pi / 2 - 2 * pi * i / _MANE_NS
    th = th0 - 0.55 * (r - 0.40) * cos(th0)                 # the locks hang: they sweep down on both sides
    x, y = r * cos(th), r * sin(th)
    h, c_ = 0.0, 0.0
    flat = 1.0 - 0.27 * max(0.0, -sin(th0)) ** 2             # the locks under the jaw lie flatter
    for tab, k, fl in ((_MANE_A, i % 6, flat if j > 0 else 1.0), (_MANE_B, (i + 3) % 6, 1.0)):
        if j in tab:
            cr, va, t = tab[j]
            c = max(0.0, (abs(cos(pi * k / 6)) - t) / (1 - t))
            hh = (va + (cr - va) * c) * fl
            if hh >= h:
                h, c_ = hh, c
    groove = (1.0 - c_) if 0 < j < 9 else 0.0
    if 0 < j < 9:                                                         # round ears standing out of the mane
        h += 0.16 * (exp(-(((x - 0.50) / 0.12) ** 2 + ((y - 0.62) / 0.11) ** 2))
                     + exp(-(((x + 0.50) / 0.12) ** 2 + ((y - 0.62) / 0.11) ** 2)))
    return x, y, h, groove, (c_ if 0 < j < 9 else 0.0)


def _lion_mask(Rm, bronze, stone, moss_amt, seed, N=22):
    """Bronze lion mask on a stone roundel, facing -Y, centred on the origin (y = 0 is the wall face).
    Returns the joined base mesh and the local position of the spout tip."""
    depth = Rm * 0.95
    y0 = -0.018
    Rf, yc = 0.72, -0.08

    def face_xy(u, v):
        a, b = 2 * u - 1, 2 * v - 1
        a, b = a * (0.66 + 0.34 * a * a), b * (0.66 + 0.34 * b * b)      # finer grid over eyes and nose
        return Rf * a * sqrt(1 - b * b / 2), yc + Rf * b * sqrt(1 - a * a / 2)

    def fn(u, v):
        x, y = face_xy(u, v)
        return (x * Rm, y0 - float(_lion_face(x, y)[0]) * depth, y * Rm)

    face = surface("lion_face", fn, nu=N, nv=N)
    uv_box(face)
    smooth(face, 179)
    assign(face, bronze)

    nr = len(_MANE_R) - 1

    def fm_(u, v):
        x, y, h, gr, cr = _lion_mane(int(round(u * _MANE_NS)) % _MANE_NS, int(round(v * nr)))
        return (x * Rm, y0 - h * depth, y * Rm)

    mane = surface("lion_mane", fm_, nu=_MANE_NS, nv=nr, closed_u=True)
    uv_box(mane)
    smooth(mane, 50)
    assign(mane, bronze)

    # patina painted from the relief itself: black-green in the eye sockets, nostrils and mouth, dirt lying in
    # the grooves between the locks, and the proud forms (brow, nose, muzzle, chin, lock crests) rubbed bright.
    def face_wear(co, n):
        f, dark, high = _lion_face(co[:, 0] / Rm, co[:, 2] / Rm)
        return np.clip(0.6 * high - dark, 0, 0.6)

    paint(face, special=lambda co, n: 1.0 * (_lion_face(co[:, 0] / Rm, co[:, 2] / Rm)[1] > 0.45), wear=face_wear)
    mv = np.array([_lion_mane(i, j)[3:] for i in range(_MANE_NS) for j in range(nr + 1)])
    paint(mane, grime=lambda co, n: mv[:, 0] ** 2 if len(co) == len(mv) else 0.0 * co[:, 0],
          wear=lambda co, n: 0.5 * mv[:, 1] ** 3 if len(co) == len(mv) else 0.0 * co[:, 0])
    zm = -0.54 * Rm
    pipe = lathe("spout_pipe", [(0.0, 0.0), (0.085 * Rm, 0.0), (0.085 * Rm, 0.42 * Rm), (0.115 * Rm, 0.43 * Rm),
                                (0.115 * Rm, 0.47 * Rm), (0.07 * Rm, 0.47 * Rm), (0.07 * Rm, 0.40 * Rm), (0.0, 0.40 * Rm)],
                 segments=12, loc=(0, 0, zm), rot=(90, 0, 0))
    assign(pipe, bronze)
    paint(pipe, wear=lambda co, n: 0.3 + 0.4 * (co[:, 1] < y0 - 0.38 * Rm))
    rd = lathe("roundel", [(1.24 * Rm, -0.07), (1.24 * Rm, 0.004), (1.20 * Rm, 0.014), (1.13 * Rm, 0.018),
                           (1.09 * Rm, 0.012), (1.04 * Rm, 0.012), (1.0 * Rm, 0.016), (0.0, 0.016)],
               segments=36, rot=(90, 0, 0))
    assign(rd, stone)
    paint(rd, special=lambda co, n: np.clip(-co[:, 2] / Rm - 0.35, 0, 1) * (0.5 + 0.4 * min(moss_amt, 1.0)))
    base = join([face, mane, pipe, rd], "lion_mask")
    return base, (0.0, y0 - 0.47 * Rm, zm)


def _patera(name, rad, material, npet=8, per=5, fine=False):
    """Carved patera: a dished flower of cupped petals with raised edges, V-grooves between them,
    a hemispherical central boss and a raised outer bead. Lies in XY facing +Z, back at z = 0."""
    if fine:
        rs = [0.07, 0.14, 0.19, 0.215, 0.30, 0.42, 0.54, 0.65, 0.74, 0.80, 0.84, 0.865, 0.895, 0.93, 0.965, 1.0]
    else:
        rs = [0.10, 0.19, 0.215, 0.36, 0.52, 0.66, 0.77, 0.84, 0.865, 0.93, 1.0]
    Lv = lod()
    if Lv == 1:                               # a star: petal crest and groove alternate
        rs, per = [0.19, 0.215, 0.52, 0.84, 0.93, 1.0], 2
    elif Lv >= 2:                             # a low octagonal boss
        rs, per = [1.0], 1
    nseg = npet * per

    def h(r, phi):
        boss = (0.10 + 0.24 * sqrt(max(0.0, 1.0 - (r / 0.20) ** 2))) if r < 0.20 else 0.0
        bead = (0.05 + 0.11 * sqrt(max(0.0, 1.0 - ((r - 0.93) / 0.07) ** 2))) if r >= 0.86 else 0.0
        pet = 0.0
        if 0.20 <= r < 0.86:
            lr = (r - 0.20) / 0.64
            w = 1.0 if lr < 0.6 else sqrt(max(0.0, 1.0 - ((lr - 0.6) / 0.4) ** 2))
            at = abs(((phi * npet / (2 * pi)) % 1.0) - 0.5) * 2.0
            if w > 1e-3 and at < w:
                pet = 0.19 * float(np.interp(at / w, [0.0, 0.62, 0.9, 1.0], [0.55, 1.0, 0.65, 0.0])) * (0.5 + 0.5 * lr)
        return max(boss, bead, pet) + 0.06 * r * r

    verts = [(0.0, 0.0, (h(0.0, 0.0) + 0.07) * rad)]
    for r in rs:
        for j in range(nseg):
            phi = 2 * pi * j / nseg
            verts.append((r * rad * cos(phi), r * rad * sin(phi), (h(r, phi) + 0.07) * rad))
    for j in range(nseg):
        phi = 2 * pi * j / nseg
        verts.append((rad * cos(phi), rad * sin(phi), -0.004))
    faces = []
    for j in range(nseg):
        j2 = (j + 1) % nseg
        faces.append((0, 1 + j, 1 + j2))
    for i in range(len(rs)):
        a, b = 1 + i * nseg, 1 + (i + 1) * nseg
        for j in range(nseg):
            j2 = (j + 1) % nseg
            faces.append((a + j, b + j, b + j2, a + j2))
    o = mesh_object(name, verts, faces)
    uv_planar(o, (1, 0, 0), (0, 1, 0))
    smooth(o, 48)
    assign(o, material)
    return o


def _plaque(body, w, h, tablet, bronze, seed):
    """Dedication tablet: a moulded stone slab let into the panel, with applied cast-bronze lettering and
    four bronze fixing studs. Faces -Y, origin = centre of its back on the wall face (the slab is packed
    out 3 cm behind for curved walls). Returns [stone part, bronze part]."""
    def rect(e, y):
        a, b = w / 2 + e, h / 2 + e
        return [(-a, y, -b), (a, y, -b), (a, y, b), (-a, y, b)]

    slab = _loft("tablet", [rect(0.0, 0.03), rect(0.0, -0.012), rect(-0.006, -0.018), rect(-0.016, -0.018),
                            rect(-0.022, -0.011), rect(-0.026, -0.011)], tablet, 30)
    uv_box(slab)
    densify(slab, 0.06)
    nm = noise_mask(0.05, seed=seed + 41, lo=0.3, hi=0.75)
    vs = _vstreak(seed + 42, 0.5, 0.7, 0.03)
    # verdigris-stained damp under the letters, clean rubbed frame
    paint(slab, grime=lambda co, n: np.clip(0.75 * vs(co, n) * (co[:, 2] < 0.0) * (co[:, 1] > -0.0125), 0, 0.75),
          wear=lambda co, n: 0.6 * (co[:, 1] < -0.0165) * (0.5 + 0.5 * nm(co * 1.7, n)))
    size = min(0.105, (w - 0.13) / (0.50 * max(len(body), 6)), (h - 0.07) / 0.52)
    letters = [text("tablet_text", body, size=size, depth=0.007, loc=(0, -0.011, 0), font="roman")]
    for sx in (-1, 1):
        for sz in (-1, 1):
            letters.append(sphere("tablet_stud", 0.008, loc=(sx * (w / 2 - 0.011), -0.0175, sz * (h / 2 - 0.011)), segments=8, rings=4))
    for o in letters:
        assign(o, bronze)
        paint(o, wear=lambda co, n: 0.6 * (co[:, 1] < -0.0165))
    return [slab, join(letters, "tablet_letters")]


def _finial(kind, z0, ks, kv, stone, bronze):
    """Carved crowning ornament standing in the top bowl. Returns (objects, nozzle tip z)."""
    objs = []
    h_st = 0.26 * kv
    st = _stem("finial_stem", z0, z0 + h_st, 0.085 * ks, 0.062 * ks, 0.032 * ks, 0.07 * ks, 0, stone, seg=40)
    objs.append(st)
    zb = z0 + h_st
    if kind == "pinecone":
        Hc, Rc = 0.34 * ks, 0.105 * ks
        pr = [(0.07 * ks, zb), (0.074 * ks, zb + 0.012 * ks), (0.05 * ks, zb + 0.02 * ks)]
        Lv = lod()
        nprof = (22, 11, 6)[Lv]
        for i in range(nprof + 1):
            t = i / nprof
            rr = Rc * (sin(pi * (0.16 + 0.84 * t) ** 0.8) ** 0.85) * (1.0 - 0.25 * t)
            pr.append((max(rr, 0.006 * ks), zb + 0.02 * ks + Hc * t))
        pr.append((0.0, zb + 0.02 * ks + Hc + 0.004))
        with lod_as(0 if Lv < 2 else 2):       # the scales are pressed into these rings: their count is chosen here
            o = lathe("pinecone", pr, segments=(64, 24, 12)[Lv], smooth_angle=60)
        zc0 = zb + 0.03 * ks
        nsc, q = 8, 46.0 / ks

        def fn(co):
            if co.z > zc0:
                phi = atan2(co.y, co.x)
                zz = (co.z - zc0) * q
                s1 = 0.5 + 0.5 * cos(nsc * phi + zz)
                s2 = 0.5 + 0.5 * cos(nsc * phi - zz)
                bump = sqrt(s1 * s2)
                s = 1.0 + 0.20 * (bump - 0.45) * _ss(zc0, zc0 + 0.03 * ks, co.z)
                co.x *= s
                co.y *= s
            return co

        if Lv < 2:
            transform_verts(o, fn)
        assign(o, stone)
        objs.append(o)
        top = zb + 0.02 * ks + Hc
    elif kind == "urn":
        Hu = 0.40 * ks
        Z = lambda f: zb + f * Hu
        pr = [(0.07 * ks, Z(0.0)), (0.045 * ks, Z(0.03))]
        pr += _q(pr[-1], (0.035 * ks, Z(0.08)), (0.06 * ks, Z(0.12)), 3)
        pr += _c3(pr[-1], (0.14 * ks, Z(0.16)), (0.155 * ks, Z(0.36)), (0.125 * ks, Z(0.50)), 8)
        pr += [(0.135 * ks, Z(0.515)), (0.135 * ks, Z(0.545)), (0.118 * ks, Z(0.56))]
        pr += _q(pr[-1], (0.07 * ks, Z(0.60)), (0.06 * ks, Z(0.70)), 4)
        pr += [(0.075 * ks, Z(0.71)), (0.075 * ks, Z(0.735))]
        pr += _q(pr[-1], (0.07 * ks, Z(0.80)), (0.03 * ks, Z(0.86)), 4)
        pr += _q(pr[-1], (0.05 * ks, Z(0.91)), (0.022 * ks, Z(0.985)), 4)
        pr += [(0.0, Z(1.0))]
        o = lathe("urn", pr, segments=lod_pick(64, 32, 16), smooth_angle=40)
        za, zc = Z(0.13), Z(0.50)

        def fn(co):
            if za < co.z < zc:
                w = sin(pi * (co.z - za) / (zc - za)) ** 0.6
                s = 1.0 + 0.09 * w * (abs(cos(atan2(co.y, co.x) * 8)) - 0.55)
                co.x *= s
                co.y *= s
            return co

        if lod() < 2:
            transform_verts(o, fn)
        assign(o, stone)
        objs.append(o)
        top = Z(1.0)
    else:
        rs = 0.115 * ks
        pr = [(0.07 * ks, zb), (0.05 * ks, zb + 0.02 * ks), (0.045 * ks, zb + 0.04 * ks)]
        zc = zb + 0.04 * ks + rs * 0.92
        for i in range(1, 15):
            a = -pi / 2 + 0.40 + (pi - 0.55) * i / 14
            pr.append((rs * cos(a), zc + rs * sin(a)))
        pr += [(0.03 * ks, zc + rs + 0.01 * ks), (0.022 * ks, zc + rs + 0.045 * ks), (0.0, zc + rs + 0.05 * ks)]
        o = lathe("ball", pr, segments=48, smooth_angle=50)
        assign(o, stone)
        objs.append(o)
        top = zc + rs + 0.05 * ks
    nz = lathe("nozzle", [(0.0, top - 0.03), (0.016 * ks, top - 0.03), (0.020 * ks, top - 0.012), (0.014 * ks, top + 0.004),
                          (0.012 * ks, top + 0.035 * ks), (0.016 * ks, top + 0.04 * ks), (0.016 * ks, top + 0.052 * ks),
                          (0.009 * ks, top + 0.052 * ks), (0.009 * ks, top + 0.03 * ks), (0.0, top + 0.03 * ks)], segments=16)
    assign(nz, bronze)
    objs.append(nz)
    return objs, top + 0.052 * ks


# ----------------------------------------------------------------------------------------------
# metalwork: lantern standard for the rim piers, bulkhead light for the pool
# ----------------------------------------------------------------------------------------------

def _lantern_standard(p, rng):
    """Small cast-iron lantern standard, origin = centre of its foot. A square foot plate bolted to the
    pier cap, a stepped collar, a vase and a fluted 50 mm shaft carrying a four-pane lantern.
    Returns (base mesh, light z)."""
    iron = mat("iron_painted", p.paint)
    parts = []
    foot = lathe("foot", [(0.128, 0.0), (0.128, 0.012), (0.120, 0.018), (0.0, 0.018)], segments=4, rot=(0, 0, 45), smooth_angle=25)
    collar = lathe("collar", [(0.080, 0.016), (0.080, 0.030), (0.074, 0.036), (0.064, 0.038), (0.064, 0.048), (0.0, 0.048)],
                   segments=28, smooth_angle=30)
    parts += [foot, collar]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(bolt("foot_bolt", radius=0.0115, height=0.009, loc=(sx * 0.066, sy * 0.066, 0.017), kind="hex"))
    pr = [(0.060, 0.046), (0.060, 0.056), (0.054, 0.062)]
    pr += _q(pr[-1], (0.050, 0.092), (0.037, 0.104), 4)
    pr += [(0.041, 0.108), (0.047, 0.120), (0.041, 0.132), (0.034, 0.137)]
    pr += _c3(pr[-1], (0.056, 0.160), (0.055, 0.245), (0.030, 0.295), 7)
    dh = 0.26                                    # extra shaft length
    pr += [(0.035, 0.300), (0.038, 0.312), (0.035, 0.324), (0.027, 0.330), (0.0245, 0.56 + dh), (0.031, 0.565 + dh),
           (0.035, 0.577 + dh), (0.031, 0.589 + dh), (0.0245, 0.594 + dh)]
    pr += _q(pr[-1], (0.0245, 0.64 + dh), (0.050, 0.665 + dh), 4)
    pr += [(0.056, 0.668 + dh), (0.056, 0.68 + dh), (0.0, 0.68 + dh)]
    post = lathe("post", pr, segments=20, smooth_angle=40)

    def flute(co):
        if 0.34 < co.z < 0.55 + dh:
            s = 1.0 - 0.06 * (0.5 + 0.5 * cos(atan2(co.y, co.x) * 10))
            co.x *= s
            co.y *= s
        return co

    if lod() == 0:                              # (the flutes need the full 20 sides)
        transform_verts(post, flute)
    zl = 0.68 + dh
    zt = zl + 0.30
    c0, c1 = 0.112, 0.160                       # corner radii of the cage bottom / top
    tray = lathe("tray", [(0.03, zl - 0.012), (0.10, zl), (c0 + 0.008, zl + 0.004), (c0 + 0.008, zl + 0.02), (c0 - 0.006, zl + 0.026),
                          (0.0, zl + 0.026)], segments=4, rot=(0, 0, 45), smooth_angle=25)
    rail = lathe("rail", [(c1 - 0.016, zt - 0.008), (c1 + 0.008, zt - 0.008), (c1 + 0.008, zt + 0.012), (c1 - 0.016, zt + 0.012),
                          (c1 - 0.016, zt - 0.008)], segments=4, rot=(0, 0, 45), smooth_angle=25)
    zr = zt + 0.012
    roof = lathe("roof", [(0.0, zr), (c1 + 0.022, zr), (c1 + 0.022, zr + 0.006), (0.115, zr + 0.04), (0.066, zr + 0.072), (0.046, zr + 0.105),
                          (0.050, zr + 0.11), (0.050, zr + 0.122), (0.028, zr + 0.132), (0.016, zr + 0.152), (0.026, zr + 0.168),
                          (0.016, zr + 0.184), (0.0, zr + 0.192)], segments=4, rot=(0, 0, 45), smooth_angle=25)
    parts += [post, tray, rail, roof]
    h0, h1 = c0 * cos(pi / 4), c1 * cos(pi / 4)
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        bar = sweep("bar", [(sx * h0, sy * h0, zl + 0.02), (sx * h1, sy * h1, zt - 0.004)], rounded_rect(0.013, 0.013, 0.003, 2))
        parts.append(bar)
    burner = lathe("burner", [(0.014, zl + 0.024), (0.014, zl + 0.10), (0.024, zl + 0.105), (0.024, zl + 0.115), (0.0, zl + 0.115)], segments=12)
    parts.append(burner)
    for o in parts:
        assign(o, iron)
    nm = noise_mask(0.09, seed=p.seed + 31, lo=0.5, hi=0.7)
    for o in parts:
        paint(o, special=lambda co, n: nm(co, n) * (0.3 + 0.6 * np.clip(1 - co[:, 2] / 0.25, 0, 1)) * (0.4 + 0.6 * p.moss))
    g0, g1 = h0 - 0.006, h1 - 0.006
    zg0, zg1 = zl + 0.026, zt - 0.008
    gv, gf = [], []
    for i in range(4):
        a = radians(90 * i)
        ca, sa = cos(a), sin(a)
        quad = [(-g0, -g0, zg0), (g0, -g0, zg0), (g1, -g1, zg1), (-g1, -g1, zg1)]
        b = len(gv)
        gv += [(x * ca - y * sa, x * sa + y * ca, z) for x, y, z in quad]
        gf.append((b, b + 1, b + 2, b + 3))
    glass = mesh_object("glass", gv, gf)
    uv_box(glass)
    assign(glass, mat("glass_lantern", glow=0.42 if p.lit else 0.0))
    zm = zl + 0.115
    # CLEARANCE RULE: the foot of the glowing mantle stands 8 mm inside the burner (it lay in the plane of the burner's top)
    mantle = lathe("mantle", [(0.0, zm - 0.008), (0.015, zm - 0.008), (0.015, zm), (0.021, zm + 0.02), (0.017, zm + 0.05), (0.0, zm + 0.062)], segments=12)
    assign(mantle, mat("emissive", "#ffcf8a", glow=20) if p.lit else mat("glass_frosted", glow=0.0))
    parts += [glass, mantle]
    base = join(parts, "lantern_std")
    return base, zm + 0.03


def _bulkhead(s, bronze, lens_mat, lamp_mat):
    """Submerged bronze bulkhead light: bolted flange, drum, bezel, domed lens under a crossed guard,
    and a cable gland towards the pedestal (local -X). Origin = centre of its foot.
    Returns (base mesh, height of the lens top)."""
    parts = []
    body = lathe("pl_body", [(0.118 * s, 0.0), (0.118 * s, 0.010 * s), (0.110 * s, 0.016 * s), (0.092 * s, 0.018 * s),
                             (0.088 * s, 0.046 * s), (0.095 * s, 0.050 * s), (0.095 * s, 0.062 * s), (0.087 * s, 0.068 * s),
                             (0.071 * s, 0.068 * s), (0.069 * s, 0.054 * s), (0.0, 0.054 * s)], segments=24, smooth_angle=35)
    # CLEARANCE RULE: the floor of the drum lies 7 mm under the pool surface that runs through the fitting (it was 1 mm
    # under that glowing sheet), and the foot of the lamp 8 mm under the floor (it was 2 mm under it)
    parts.append(body)
    for i in range(6):
        a = radians(60.0 * i + 30.0)
        parts.append(bolt("pl_bolt", radius=0.008 * s, height=0.006 * s, loc=(0.104 * s * cos(a), 0.104 * s * sin(a), 0.012 * s), kind="dome"))
    for ang in (45.0, 135.0):
        ca, sa = cos(radians(ang)), sin(radians(ang))
        path = [(0.092 * s * t * ca, 0.092 * s * t * sa, (0.066 + 0.030 * (1 - t * t)) * s) for t in (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)]
        parts.append(sweep("pl_guard", path, rounded_rect(0.011 * s, 0.005 * s, 0.0015 * s, 1)))
    # cable gland on the side facing the pedestal (the feed runs under the floor)
    parts.append(lathe("pl_gland", [(0.0, 0.0), (0.024 * s, 0.0), (0.024 * s, 0.028 * s), (0.017 * s, 0.034 * s), (0.0, 0.034 * s)],
                       segments=10, loc=(-0.086 * s, 0, 0.030 * s), rot=(0, -90, 0)))
    for o in parts:
        assign(o, bronze)
        paint(o, special=lambda co, n: 0.9 * (co[:, 2] < 0.014 * s), wear=lambda co, n: 0.2 + 0.4 * (co[:, 2] > 0.06 * s))
    lens = lathe("pl_lens", [(0.070 * s, 0.058 * s), (0.065 * s, 0.070 * s), (0.049 * s, 0.079 * s), (0.026 * s, 0.084 * s),
                             (0.0, 0.085 * s)], segments=20)
    assign(lens, lens_mat)
    lamp = lathe("pl_lamp", [(0.0, 0.046 * s), (0.020 * s, 0.046 * s), (0.019 * s, 0.068 * s), (0.011 * s, 0.076 * s),
                             (0.0, 0.078 * s)], segments=10)
    assign(lamp, lamp_mat)
    return join(parts + [lens, lamp], "pool_light"), 0.085 * s


# ----------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------

def build(p, rng):
    _UVN[0] = 0
    seed = p.seed
    flow = p.flow
    flowing = flow > 0.02
    Lv = lod()

    # ---- footprint contract: `diameter` = outer diameter of the kerb (outermost step) -----------
    D = clamp(p.diameter, 2.5, 7.0)
    n_steps = min(int(p.steps), 1 if D < 3.3 else (2 if D < 5.4 else 3))
    rise = 0.16
    going = 0.30 if D < 4.0 else 0.32
    d_land = clamp(0.065 * D, 0.24, 0.40)                 # landing between the basin and the first nosing
    R = D / 2 - d_land - going * (n_steps - 1) - 0.006    # outer radius of the basin wall (die face)
    T = clamp(0.115 * R + 0.02, 0.16, 0.30)               # wall thickness
    plan = _Plan(p.plan, R)
    pin = 0.05
    hw = clamp(0.10 * R, 0.13, 0.21)                      # half width of a pier
    po = clamp(0.42 * hw, 0.06, 0.088)                    # pier projection in front of the die
    if p.plan == "quatrefoil":
        r_clear = (plan.cusp(-T) - pin) * sqrt(2.0) - 0.04
    else:
        r_clear = R - T - pin
    ks = min((2 * R / 5.0) ** 0.75, (r_clear - 0.12) / 1.35)   # scale of the central stem
    kv = ks * p.stem_height
    tiers = min(int(p.tiers), 2 if R < 1.35 else 3)
    mg = clamp(2.0 * p.moss, 0.0, 1.4)                    # gain on moss / algae (1 at the default age)
    mg1 = min(mg, 1.0)
    heavy = clamp(2.0 * p.moss - 1.0, 0.0, 1.0)           # neglect: growth spreading over open faces
    Rext = D / 2

    stone = mat("stone_smooth", p.stone, bump=1.7)       # crisper tooling: the dressed faces must not read as clay
    kerb = mat("stone_rough", _shade(p.stone, 0.84), bump=1.3)
    bronze = mat("brass", "#dcbc92")                     # golden statuary bronze (bronze_aged goes near-black under lantern light)
    murk = clamp((1.0 - flow) * 0.6 + 0.5 * heavy, 0.0, 1.0)
    # the water surface: dark, glossy and rippling, so it mirrors the lanterns, the lit stone and the falls
    # depth: the lit bed of the basin shows through the water; glow: the light of the submerged lamps in it
    surf_m = mat("water", _mix("#182c2c", "#2b2d15", murk), ripple=0.3 + 0.45 * flow, depth=0.3,
                 glow=0.8 if (p.lit and int(p.pool_lights) > 0) else 0.0, grime=0.0)
    # thin sheets: mostly clear, so the stone and the lamps show through them (an untinted water_fall renders black)
    fall = mat("water_fall", "#e4f0f4", opacity=0.7, glow=0.14, grime=0.0)

    z_pod = rise * n_steps
    hp, hc = 0.13, 0.15
    hd = 0.38 if R >= 1.45 else 0.33
    z_d0, z_c0 = z_pod + hp, z_pod + hp + hd
    z_rim = z_c0 + hc
    z_floor = z_pod + 0.02
    z_w = z_rim - 0.19 - (1 - flow) * 0.08
    dep = 0.24                                            # visible depth of the pool

    nzA = noise_mask(0.5, seed=seed + 1, lo=0.45, hi=0.6)        # big sharp-edged patches
    nzB = noise_mask(0.16, seed=seed + 2, lo=0.4, hi=0.6)        # small patches
    nzC = noise_mask(0.22, seed=seed + 3, lo=0.35, hi=0.75)      # soft
    streak = _vstreak(seed + 5, 0.55, 0.7, 0.07)
    streak2 = _vstreak(seed + 6, 0.5, 0.7, 0.04)

    def northf(co):
        return _ssn(-0.3, 0.7, co[:, 1] / Rext)

    # ---- kerb steps -------------------------------------------------------------------------
    steps = join(_build_steps(plan, n_steps, rng, T, z_pod, rise, d_land, going, kerb), "steps")
    densify(steps, lod_pick(0.14, 0.30, 0.60))            # (levels: what the weathering masks need, no more)
    displace_noise(steps, strength=0.003, scale=0.35, seed=seed, along_normal=False)

    def step_moss(co, n):
        """damp black-green line in the joint under every riser (the moss cushions grow out of it)"""
        z = co[:, 2]
        d = plan.offs(co)
        nb, ncs, nrt = nzB(co, n), nzC(co, n), northf(co)
        jm = np.zeros(len(co))
        wide = np.zeros(len(co))
        for c in range(n_steps + 1):
            zt = z_pod - c * rise
            d_r = (d_land + (c - 1) * going) if c > 0 else 0.05
            riser = np.clip(1 - (z - zt) / 0.05, 0, 1) * (z > zt - 0.012) * (d < d_r + 0.04) * (d > d_r - 0.06)
            tread = np.clip(1 - (d - d_r) / 0.07, 0, 1) * (np.abs(z - zt) < 0.025)
            jm = np.maximum(jm, np.maximum(riser, tread))
            wide = np.maximum(wide, np.clip(1 - (d - d_r) / 0.22, 0, 1) * (np.abs(z - zt) < 0.025))
        m = jm * (0.35 + 0.65 * nrt) * (0.45 + 0.55 * nb) * 0.9 * mg1
        return np.clip(np.maximum(m, heavy * 0.62 * wide * ncs ** 1.5), 0, 1)

    def step_grime(co, n):
        """trodden-in dirt: blotches on the treads, damp standing against every riser"""
        z = co[:, 2]
        d = plan.offs(co)
        m = 0.34 * nzA(co * 1.7, n)
        for c in range(n_steps):
            d_r = (d_land + (c - 1) * going) if c > 0 else 0.05
            m = np.maximum(m, np.clip(1 - (d - d_r) / 0.16, 0, 1) * (np.abs(z - (z_pod - c * rise)) < 0.03) * (0.3 + 0.35 * nzC(co, n)))
        # the foot of the kerb: street dirt washed against the lowest riser
        m = np.maximum(m, np.clip(1 - z / 0.09, 0, 1) * (z > -0.005) * (0.42 + 0.3 * nzC(co * 1.5, n)))
        return np.clip(m, 0, 0.7)

    paint(steps, special=step_moss, grime=step_grime, wear=lambda co, n: 0.3 * np.clip(n[:, 2], 0, 1) * nzC(co * 1.3, n))

    # ---- basin wall and piers -----------------------------------------------------------------
    hw_eff = po if p.plan == "quatrefoil" else hw
    fa = (hw_eff - 0.012) / plan.L
    ci = min(0.075, 0.36 * hw)
    wall_objs, anchors = _build_wall(plan, rng, T, z_pod, hp, hd, hc, fa, stone)
    pier_objs, pier_tops = _build_piers(plan, T, z_pod, z_rim, hw, po, pin, ci, stone)
    basin = join(wall_objs + pier_objs, "basin")
    _dress(basin)
    densify(basin, lod_pick(0.16, 0.30, 0.60))
    displace_noise(basin, strength=0.0025, scale=0.3, seed=seed + 4, along_normal=False)

    def basin_moss(co, n):
        z = co[:, 2]
        d = plan.offs(co)
        inner, outer = d < -0.55 * T, d > -0.05
        side, up = 1 - np.abs(n[:, 2]), np.clip(n[:, 2], 0, 1)
        nrt, nb, ncs = northf(co), nzB(co, n), nzC(co, n)
        band = inner * np.clip(1 - (z - (z_w + 0.02)) / (0.05 + 0.11 * ncs), 0, 1) * (0.75 + 0.2 * mg1)   # ragged algae line of the pool
        foot = outer * np.clip(1 - (z - z_pod) / 0.04, 0, 1) * (0.35 + 0.65 * nrt) * (0.45 + 0.55 * nb) * 0.9 * mg1
        hv = heavy * 0.62 * ncs ** 1.5 * outer * (0.4 + 0.6 * nrt) * (up * (z < z_d0 + 0.05) + side * np.clip(1 - (z - z_pod) / (hp + 0.5 * hd), 0, 1))
        return np.clip(np.maximum(band, np.maximum(foot, hv)), 0, 1)

    def basin_grime(co, n):
        z = co[:, 2]
        d = plan.offs(co)
        inner, outer = d < -0.55 * T, d > -0.05
        side = 1 - np.abs(n[:, 2])
        below = np.clip(1 - (z_c0 + 0.02 - z) / 0.30, 0, 1) * (z < z_c0 + 0.03)
        up = np.clip(n[:, 2], 0, 1)
        ncs = nzC(co, n)
        drip = outer * side * below * streak(co, n) * 0.74                    # run-off under the coping drip
        wet = inner * np.clip(1 - (z - (z_w + 0.06)) / 0.09, 0, 1) * (0.6 + 0.24 * ncs)     # wet-dark stone above the waterline
        # the inner half of the coping is splashed and stays damp in patches; water stands in its joints
        damp = up * (z > z_rim - 0.03) * _ssn(-0.35 * T, -0.8 * T, d) * (0.2 + 0.4 * nzA(co * 1.6, n)) * (0.45 + 0.55 * flow)
        foot = outer * np.clip(1 - (z - z_pod) / 0.16, 0, 1) ** 0.8 * (0.36 + 0.34 * ncs)   # rising damp and dirt off the landing
        blot = 0.26 * nzA(co * 1.3, n) * side
        return np.clip(np.maximum(np.maximum(drip, wet), np.maximum(np.maximum(damp, foot), blot)), 0, 0.82)

    def basin_wear(co, n):
        """lime: the pale tide line the pool leaves above its wet band, white runs down the die between the
        dark ones, and the rubbed top of the coping"""
        z = co[:, 2]
        d = plan.offs(co)
        inner, outer = d < -0.55 * T, d > -0.05
        side = 1 - np.abs(n[:, 2])
        tide = inner * side * np.clip(1 - np.abs(z - (z_w + 0.17)) / 0.035, 0, 1) * (0.45 + 0.4 * nzC(co * 2.0, n))
        below = np.clip(1 - (z_c0 + 0.02 - z) / 0.36, 0, 1) * (z < z_c0 + 0.03)
        runs = outer * side * below * streak2(co * 1.31 + 0.37, n) * 0.5
        # the top of the coping: sat on for generations - the outer two thirds are rubbed smooth and pale in
        # seat-wide patches, most of all along the rounded nose; lichen-pale flecks on the rest
        up = np.clip(n[:, 2], 0, 1)
        seat = _ssn(-0.75 * T, -0.45 * T, d) * (0.34 + 0.5 * nzA(co * 0.8 + 3.1, n))
        nose_ = (d > 0.03) * (z > z_rim - 0.045) * np.clip(n[:, 2] + 0.5, 0, 1) * (0.5 + 0.3 * nzC(co * 1.2, n))
        top = np.maximum(up * (z > z_rim - 0.03) * seat, nose_)
        fleck = 0.5 * up * outer * (z < z_rim - 0.05) * nzB(co * 1.7 + 5.0, n) ** 2
        return np.clip(np.maximum(np.maximum(tide, runs), np.maximum(top, fleck)), 0, 0.85)

    paint(basin, grime=basin_grime, special=basin_moss, wear=basin_wear)

    # dedication tablet on the front panel, carved paterae on the others (long panels get a pair)
    plq = None
    if p.inscription.strip() and anchors:
        plq = min(range(len(anchors)), key=lambda i: (round(anchors[i][0][1], 3), abs(anchors[i][0][0])))
        (px, py), (nx, ny), zc, pw, ph = anchors[plq]
        w_pl, h_pl = min(0.82, pw - 0.34), min(0.16, ph - 0.10)
        if w_pl < 0.3:
            plq = None
        else:
            for part_ in _plaque(p.inscription.strip(), w_pl, h_pl, stone, bronze, seed):
                instances_to_mesh(part_.name, part_, [trs((px, py, zc), (0, 0, degrees(atan2(ny, nx)) + 90.0), 1.0)], keep_base=False)
            pc, pn = np.array([px, py]), np.array([nx, ny])

            def plaque_run(co, n):
                """verdigris / damp run-off on the stone below the tablet"""
                dd = co[:, :2] - pc
                along = np.abs(dd[:, 0] * -pn[1] + dd[:, 1] * pn[0])
                out = dd[:, 0] * pn[0] + dd[:, 1] * pn[1]
                m = (along < w_pl / 2) * (np.abs(out) < 0.12) * (co[:, 2] < zc - h_pl / 2 + 0.01) * (co[:, 2] > zc - h_pl / 2 - 0.3)
                return m * 0.42 * streak2(co, n) * np.clip(1 - (zc - h_pl / 2 - co[:, 2]) / 0.2, 0, 1)

            paint(basin, grime=plaque_run)
    ros_m = []
    for ia, ((px, py), (nx, ny), zc, pw, ph) in enumerate(anchors):
        if ia == plq:
            continue
        sc = min(1.0, (pw - 0.2) / 0.19, ph / 0.28)
        if sc < 0.45:
            continue
        ang = degrees(atan2(ny, nx)) + 90.0
        offs = [-pw * 0.27, pw * 0.27] if (pw >= 1.0 and p.plan == "octagon") else [0.0]   # a pair would float off a curve
        for dx in offs:
            ros_m.append(trs((px - ny * dx, py + nx * dx, zc), (90, 0, ang), sc))
    if ros_m:
        instances_to_mesh("paterae_wall", _patera("patera", 0.088, stone), ros_m, keep_base=False)

    # ---- pedestal -----------------------------------------------------------------------------
    n_sp = int(p.spouts)
    ped_round = (p.plan == "round") or (n_sp not in (0, 1, 2, 4, 8))
    rp = 0.46 * ks
    rr = rp if ped_round else rp / cos(pi / 8)
    r_ped = 1.42 * rr
    z_pl = z_rim + 0.03                          # plinth top, well clear of the water
    z_b = z_pl + 0.10 * ks
    z_dd = z_b + 0.50 * kv
    e = ks
    fl = max(0.014 * e, 0.009)                   # fillet steps of the cornice: never thinner than 9 mm
    pr = [(1.42 * rr, z_floor - 0.01), (1.42 * rr, z_pl - 0.03), (1.385 * rr, z_pl - 0.006), (1.30 * rr, z_pl)]
    pr += _q(pr[-1], (1.40 * rr, z_pl + 0.03 * e), (1.26 * rr, z_pl + 0.056 * e), 5)
    pr += [(1.20 * rr, z_pl + 0.06 * e)]
    pr += _q(pr[-1], (1.03 * rr, z_pl + 0.066 * e), (rr, z_b), 4)
    zj_p = (z_b + 0.012, z_dd - 0.010)                    # bed joints: base course / die / cornice block
    for q, zq in enumerate(zj_p):
        pr += [(rr, zq - 0.0038), (rr - 0.0065, zq - 0.0026), (rr - 0.0065, zq + 0.0026), (rr, zq + 0.0038)]
    pr += [(rr, z_dd), (1.045 * rr, z_dd + 0.006 * e), (1.075 * rr, z_dd + 0.019 * e), (1.045 * rr, z_dd + 0.032 * e), (rr, z_dd + 0.038 * e)]
    pr += _q(pr[-1], (1.01 * rr, z_dd + 0.076 * e), (1.19 * rr, z_dd + 0.084 * e), 4)
    zf = z_dd + 0.084 * e + fl
    zt_ = zf + 0.036 * e
    z_c = zt_ + fl + 0.024 * e
    pr += [(1.19 * rr, zf), (1.31 * rr, zf), (1.31 * rr, zt_), (1.25 * rr, zt_), (1.25 * rr, zt_ + fl), (0.72 * rr, z_c), (0.0, z_c)]
    if ped_round:
        ped = lathe("pedestal", pr, segments=64, smooth_angle=38)
    else:
        ped = lathe("pedestal", pr, segments=8, rot=(0, 0, 22.5), smooth_angle=25)
    _dress(ped)
    densify(ped, 0.11)
    assign(ped, stone)
    mask_angles = [radians(-90.0 + 360.0 * i / n_sp) for i in range(n_sp)] if n_sp > 0 else []
    z_mask = z_b + 0.27 * kv
    Rm = min(0.165 * ks, 0.21 * kv, (rp * tan(pi / 8) * 0.80) if not ped_round else 1.0)

    def under_mouth(co, wk):
        """1 on the trail below each lion mouth (widening downwards), 0 elsewhere."""
        if not mask_angles:
            return np.zeros(len(co))
        phi = np.arctan2(co[:, 1], co[:, 0])
        z = co[:, 2]
        fal = np.clip((z_mask - z) / max(z_mask - z_w, 0.05), 0, 1)
        m = np.zeros(len(co))
        for a in mask_angles:
            dphi = np.abs(((phi - a + np.pi) % (2 * np.pi)) - np.pi)
            m = np.maximum(m, np.clip(1 - dphi * rp / (wk * Rm * (1 + 1.2 * fal)), 0, 1))
        return m * (z < z_mask - 0.3 * Rm)

    def ped_moss(co, n):
        z = co[:, 2]
        up, nb, ncs = np.clip(n[:, 2], 0, 1), nzB(co, n), nzC(co, n)
        band = np.clip(1 - (z - (z_w + 0.015)) / (0.04 + 0.10 * ncs), 0, 1) * (0.75 + 0.2 * mg1)   # ragged algae line
        run = under_mouth(co, 1.0) * (0.45 + 0.3 * nb) * (0.6 + 0.4 * mg1)                   # green trail below each mouth
        return np.clip(np.maximum(np.maximum(band, run), heavy * 0.6 * ncs ** 1.5 * (0.3 + 0.7 * up)), 0, 1)

    def ped_grime(co, n):
        """dark wet core of the run-off below each lion mouth"""
        z = co[:, 2]
        run = under_mouth(co, 0.42) * (0.55 + 0.45 * streak2(co, n)) * 0.68
        # splashed foot: the falling sheets and the mouth jets keep the base of the pedestal wet
        wet = np.clip(1 - (z - (z_w + 0.04)) / (0.16 + 0.2 * nzC(co, n)), 0, 1) * (0.48 + 0.2 * nzB(co, n)) * (0.5 + 0.5 * flow)
        drip = (z < z_dd) * (z > z_b) * np.clip(1 - (z_dd - z) / (0.34 * kv), 0, 1) * streak(co, n) * 0.5      # under the cornice
        blot = 0.24 * nzA(co * 1.5, n)
        dz = np.minimum(np.abs(z - zj_p[0]), np.abs(z - zj_p[1]))
        bed = np.maximum(0.92 * (dz < 0.0032), 0.4 * (dz < 0.0045))                  # dirt in the bed joints
        course = np.where(z < zj_p[0], 0.2, np.where(z < zj_p[1], 0.0, 0.14)) * (z > z_w + 0.2)
        return np.clip(np.maximum(np.maximum(np.maximum(run, wet), np.maximum(drip, blot + course)), 0.0), 0, 0.78) * (bed < 0.3) + bed

    def ped_wear(co, n):
        """lime bloom above the wet foot and pale runs down the die"""
        z = co[:, 2]
        tide = np.clip(1 - np.abs(z - (z_w + 0.26)) / 0.05, 0, 1) * (0.35 + 0.45 * nzC(co * 2.0, n)) * (1 - under_mouth(co, 1.0))
        runs = (z < z_dd) * (z > z_b) * streak2(co * 1.27 + 0.5, n) * 0.42
        return np.clip(np.maximum(tide, runs), 0, 0.8)

    paint(ped, grime=ped_grime, special=ped_moss, wear=ped_wear)

    # ---- lion masks and paterae -------------------------------------------------------------------
    r_face = rp + 0.004
    tip_r = tip_z = None
    if n_sp > 0:
        mbase, tip = _lion_mask(Rm, bronze, stone, mg, seed)
        mats_ = [trs((r_face * cos(a), r_face * sin(a), z_mask), (0, 0, degrees(a) + 90.0), 1.0) for a in mask_angles]
        instances_to_mesh("lion_masks", mbase, mats_, keep_base=False)
        tip_r = r_face - tip[1]
        tip_z = z_mask + tip[2]
    if not ped_round:
        # carved paterae on the faces of the die that carry no mask
        blank = [radians(-90.0 + 45.0 * i) for i in range(8)]
        blank = [a for a in blank if all(abs(((a - b + pi) % (2 * pi)) - pi) > 0.05 for b in mask_angles)]
        if blank:
            r_pat = min(0.14 * ks, 0.72 * rp * tan(pi / 8))
            instances_to_mesh("paterae", _patera("patera_p", r_pat, stone, per=8, fine=True),
                                    [trs((rp * cos(a), rp * sin(a), z_mask), (90, 0, degrees(a) + 90.0), 1.0) for a in blank],
                                    keep_base=False)

    # ---- tiers: dimensions first (the sheets are planned before the stone is stained) --------------
    spec = [(1.02, 0.30, 0.42, 0.235, 0.105, 0.27, 0.055),      # rim radius, depth, stem height, bulb r, neck r, flare r, wall
            (0.64, 0.21, 0.62, 0.135, 0.062, 0.155, 0.042),
            (0.40, 0.15, 0.50, 0.095, 0.045, 0.105, 0.034)]
    sc_ = int(p.scallops)
    n_lobes = [sc_, max(6, sc_ - 2) if sc_ else 0, max(6, sc_ - 4) if sc_ else 0]
    msegs = [12, 8, 8]
    A = 0.085
    rw = rng.fork("water")
    td = []
    z = z_c
    r_seat = 0.33 * ks
    for i in range(tiers):
        Rb, hb, hs, rb, rn, rt, tw = spec[i]
        Rb, hb, rb, rn, rt, tw = Rb * ks, hb * ks, rb * ks, rn * ks, rt * ks, tw * ks
        zb = z + hs * kv
        n = n_lobes[i]
        dip = 0.012 * ks if n > 0 else 0.0
        lip = zb + hb
        zw = (lip - dip + 0.002) if flowing else (lip - 0.045 * ks)
        td.append(dict(i=i, z0=z, zb=zb, Rb=Rb, hb=hb, rb=rb, rn=rn, rt=rt, tw=tw, n=n, dip=dip, lip=lip, zw=zw,
                       phase=-pi / 2 - (pi / n if n else 0.0), r_seat=r_seat, mseg=msegs[i], sheets=[]))
        z = zb + tw
        r_seat = (0.17, 0.12, 0.085)[i] * ks
    z_top = z
    if flowing:
        kq = 0.6 + 0.4 * ks
        for b in td:
            n = b["n"]
            cnt = n if n > 0 else (9, 8, 7)[b["i"]]
            arc = 2 * pi * b["Rb"] / cnt
            off = rw.u(0.0, 1.0) if n == 0 else 0.0
            for j in range(cnt):
                if flow < 0.99 and rw.u() > 0.35 + flow * 0.75:
                    continue
                if n > 0:
                    phi = -pi / 2 + 2 * pi * j / cnt
                    w = arc * rw.u(0.60, 0.84) * (0.45 + 0.55 * flow)
                else:                                                # plain lip: broken sheets, about half the rim
                    phi = -pi / 2 + 2 * pi * (j + off + rw.u(-0.13, 0.13)) / cnt
                    w = arc * rw.u(0.55, 0.78) * (0.45 + 0.55 * flow)
                # one continuous sheet per spill point: columns across it carry faint flutes (dv) and their own
                # readiness to tear (cw); a weak flow necks in harder
                nc = int(clamp(round(w / (0.03 * kq)), 4, 16))
                cw, c_ = [], rw.u(-0.05, 0.12)
                for _ in range(nc + 1):
                    c_ = clamp(c_ + rw.u(-0.09, 0.09), -0.08, 0.2)
                    cw.append(c_)
                b["sheets"].append(dict(phi=phi, w=w, v_out=rw.u(0.24, 0.40) * (0.5 + 0.5 * ks), uo=rw.u(0, 3),
                                        neck=rw.u(0.18, 0.34) * (1.5 - 0.5 * flow), drift=rw.u(-0.02, 0.02) * (0.5 + 0.5 * ks),
                                        wob=rw.u(0.03, 0.08), ph=rw.u(0, 6.28), cw=cw,
                                        dv=[rw.u(-0.03, 0.03) for _ in range(nc + 1)]))

    # ---- stems and bowls ------------------------------------------------------------------------
    for b in td:
        i, Rb, hb, tw, n, lip, dip, zb = b["i"], b["Rb"], b["hb"], b["tw"], b["n"], b["lip"], b["dip"], b["zb"]
        st = _stem("stem%d" % i, b["z0"] - 0.004, zb, b["r_seat"], b["rb"], b["rn"], b["rt"], 12 if i == 0 else (10 if i == 1 else 0), stone)
        _dress(st)
        bw = _bowl("bowl%d" % i, Rb, zb, hb, tw, b["rt"] * 0.985, n, b["phase"], A, dip, stone, b["mseg"], zw=b["zw"])
        _dress(bw)
        jang = _joint_angles(b)
        if jang and Lv < 2:
            _radial_joints(bw, jang, max(0.34 * Rb, 1.3 * b["rt"]), half=0.0028, depth=0.0065)
        ja = np.sort(np.array([(a - jang[0]) % (2 * pi) for a in jang])) if jang else None

        def sect(co, ja=ja, a0=(jang[0] if jang else 0.0), i=i):
            """(distance to the nearest radial joint in m, shade of the vertex's own section)"""
            if ja is None:
                return np.full(len(co), 1e9), np.zeros(len(co))
            r = np.hypot(co[:, 0], co[:, 1])
            rel = (np.arctan2(co[:, 1], co[:, 0]) - a0) % (2 * np.pi)
            dj = np.min(np.abs(((rel[:, None] - ja[None, :] + np.pi) % (2 * np.pi)) - np.pi), axis=1) * r
            tones = np.array([0.46 * _h1(q, 3.1 + i, seed * 0.37) - 0.18 for q in range(len(ja) + 1)])
            return dj, tones[np.searchsorted(ja, rel, side="right")]

        def cfun(co, b=b):
            """1 at every spill point of the lip (lobe tips / under the sheets of a plain bowl), 0 between."""
            phi = np.arctan2(co[:, 1], co[:, 0])
            if b["n"] > 0:
                return np.abs(np.sin(np.pi * (((phi - b["phase"]) * b["n"] / (2 * np.pi)) % 1.0))) ** 0.6
            c = np.zeros(len(co))
            for s in b["sheets"]:
                dphi = np.abs(((phi - s["phi"] + np.pi) % (2 * np.pi)) - np.pi)
                c = np.maximum(c, np.clip(1 - dphi * b["Rb"] / (0.75 * s["w"]), 0, 1) ** 0.6)
            return c

        def ext_of(co, nrm, Rb=Rb, tw=tw):
            r = np.hypot(co[:, 0], co[:, 1])
            rad = (co[:, 0] * nrm[:, 0] + co[:, 1] * nrm[:, 1]) / np.maximum(r, 1e-6)
            return ((nrm[:, 2] < 0.25) & (rad > -0.1) & (r > 0.3 * Rb)) | (r > Rb - 0.1 * tw)

        def bowl_wear(co, nrm, cfun=cfun, ext_of=ext_of, lip=lip, hb=hb, tw=tw, dip=dip, sect=sect):
            """limescale: a pale crust on the lip roll and a tongue below every spill point"""
            c, ext = cfun(co), ext_of(co, nrm)
            zrel = np.clip(1 - (lip - co[:, 2]) / hb, 0, 1)
            roll = np.clip(1 - (lip - dip - co[:, 2]) / (1.35 * tw), 0, 1)
            wetc = _ssn(0.4, 0.75, c) if flowing else np.zeros(len(co))
            crust = roll * (0.6 + 0.4 * nzC(co * 2.0, nrm)) * (0.55 + 0.45 * c) * 0.95 * (1 - wetc)      # dry stretches of the lip
            below = (lip - dip - co[:, 2]) / tw
            fringe = ext * np.clip(1 - np.abs(below - 1.9) / 0.9, 0, 1) * (0.5 + 0.5 * nzC(co * 2.0, nrm)) * (0.45 + 0.55 * c) * 0.9
            tongue = ext * c ** 6 * _ssn(0.05, 0.6, zrel) * (0.55 + 0.45 * nzC(co * 2.5, nrm)) * 0.95 * (below > 1.3)
            dj, tn = sect(co)
            own = 0.9 * np.clip(-tn, 0, 1) * ext * (dj > 0.004)                         # paler blocks
            return np.clip(np.maximum(np.maximum(np.maximum(crust, fringe), tongue) * (0.8 + 0.2 * flow), own), 0, 0.95)

        def bowl_grime(co, nrm, cfun=cfun, ext_of=ext_of, lip=lip, hb=hb, b=b, sect=sect, Rb=Rb):
            """dark wet run-off flanking each lime tongue"""
            c, ext = cfun(co), ext_of(co, nrm)
            zrel = np.clip(1 - (lip - co[:, 2]) / hb, 0, 1)
            flank = ext * _ssn(0.3, 0.7, c) * (1 - 0.9 * c ** 10) * (0.4 + 0.6 * zrel) * (0.5 + 0.5 * streak2(co, nrm)) * 0.9
            if flowing:                                              # the lip roll is wet-dark wherever the water goes over it
                roll = np.clip(1 - (lip - co[:, 2]) / (1.5 * b["tw"]), 0, 1) ** 0.5
                flank = np.maximum(flank, roll * _ssn(0.4, 0.75, c) * 0.66)
            blot = 0.22 * nzA(co * 1.8, nrm) * ext
            # the dry joints between the sections: a dark line, damp creeping out of it, each block its own shade
            dj, tn = sect(co)
            r = np.hypot(co[:, 0], co[:, 1])
            joint = np.maximum(0.92 * (dj < 0.0016), 0.5 * np.clip(1 - dj / 0.035, 0, 1) ** 1.5) * (r > 0.3 * Rb)
            # damp, dirty soffit: the underside of the bowl never dries
            soffit = ext * np.clip(-nrm[:, 2], 0, 1) * (0.3 + 0.3 * nzC(co * 1.4, nrm))
            m = np.maximum(np.maximum(flank, blot + np.clip(tn, 0, 1) * ext), np.maximum(soffit, joint))
            return np.clip(m, 0, 0.92)

        def bowl_moss(co, nrm, cfun=cfun, ext_of=ext_of, Rb=Rb, zw=b["zw"]):
            c, ext = cfun(co), ext_of(co, nrm)
            r = np.hypot(co[:, 0], co[:, 1])
            crease = ext * (1 - c) ** 3 * (r > 0.45 * Rb) * (0.45 + 0.55 * nzB(co, nrm)) * 0.5 * mg
            inside = (~ext) * (r > 0.3 * Rb) * (0.7 + 0.25 * mg1)             # algae inside the bowl
            under = ext * np.clip(-nrm[:, 2], 0, 1) * nzC(co * 1.6, nrm) ** 1.5 * 0.6 * heavy
            return np.clip(np.maximum(inside, crease + under), 0, 1)

        paint(bw, grime=bowl_grime, special=bowl_moss, wear=bowl_wear)

        zfoot = td[i - 1]["zw"] if i > 0 else None

        def stem_moss(co, nrm, zfoot=zfoot):
            m = heavy * 0.6 * nzC(co, nrm) ** 1.5 * np.clip(0.4 + nrm[:, 2], 0, 1)
            if zfoot is not None:
                m = np.maximum(m, np.clip(1 - (co[:, 2] - (zfoot + 0.03 * ks)) / (0.05 * ks), 0, 1) * (0.75 + 0.2 * mg1))
            return np.clip(m, 0, 1)

        paint(st, special=stem_moss, grime=lambda co, nrm: 0.3 * nzA(co * 1.8, nrm),
              wear=lambda co, nrm: 0.4 * streak2(co * 1.2, nrm) * (0.4 + 0.6 * flow))

    # ---- finial -----------------------------------------------------------------------------------
    fin_objs, z_tip = _finial(p.finial, z_top - 0.004, ks, kv, stone, bronze)
    zfin = td[-1]["zw"]
    for o in fin_objs[:-1]:
        _dress(o)
        paint(o, special=lambda co, n: np.clip(1 - (co[:, 2] - (zfin + 0.03 * ks)) / (0.04 * ks), 0, 1) * 0.85,
              wear=lambda co, n: 0.55 * streak2(co, n) * (0.4 + 0.6 * flow) * (co[:, 2] > zfin + 0.08 * ks))
    paint(fin_objs[-1], wear=0.45)

    # ---- pier ornaments that are stone: balls on the piers without a lantern ------------------------
    N = plan.n
    n_lan = int(p.rim_lanterns)
    while n_lan > 0 and N % n_lan != 0:
        n_lan -= 1
    lan_idx = [int(i * N / n_lan + 1e-6) for i in range(n_lan)] if n_lan else []
    ball_idx = [k for k in range(N) if k not in lan_idx]
    s_orn = clamp(hw / 0.21, 0.72, 1.0)
    if ball_idx:
        rb_ = 0.105
        prb = [(0.085, -0.004), (0.085, 0.012), (0.06, 0.022), (0.048, 0.034), (0.052, 0.045)]
        zc_ = 0.045 + rb_ * 0.86
        for i in range(1, 15):
            a = -pi / 2 + 0.52 + (pi - 0.52) * i / 14
            prb.append((max(rb_ * cos(a), 0.0), zc_ + rb_ * sin(a)))
        ball = lathe("pier_ball", prb, segments=28, smooth_angle=50)
        _dress(ball)
        assign(ball, stone)
        bm_ = [trs((pier_tops[k][0], pier_tops[k][1], pier_tops[k][2] - 0.002), (0, 0, rng.u(0, 360)), s_orn) for k in ball_idx]
        balls = instances_to_mesh("pier_balls", ball, bm_, keep_base=False)
        if heavy > 0:
            paint(balls, special=lambda co, n: heavy * 0.6 * np.clip(n[:, 2], 0, 1) * nzC(co * 1.4, n) ** 1.5)

    # ---- moss: green cushions in the damp joints (under every riser, round the foot of the basin), thicker
    # to the north; with neglect also on the treads and the coping. Nothing grows beyond the footprint.
    if mg > 0.05:
        rm_ = rng.fork("moss")
        moss = _Moss(rm_)
        lines = [(0.05, z_pod, fa + 0.02, 1.0 - fa - 0.02)]
        lines += [(d_land + c * going, z_pod - (c + 1) * rise, 0.0, 1.0) for c in range(n_steps - 1)]
        for d_j, z_j, f_lo, f_hi in lines:
            for k in range(plan.n):
                a_, b_ = plan.pt(k, f_lo, d_j), plan.pt(k, f_hi, d_j)
                seg_len = max(hypot(b_[0] - a_[0], b_[1] - a_[1]), 0.2) * (1.0 if p.plan == "octagon" else 1.12)
                t = rm_.u(0.0, 0.25)
                while t < seg_len - 0.06:
                    f = f_lo + (f_hi - f_lo) * t / seg_len
                    x0, y0 = plan.pt(k, f, d_j)
                    nw = _ss(-0.5, 0.7, y0 / Rext)
                    if rm_.u(0, 1) < (0.3 + 0.7 * nw) * mg1 * 0.8 + 0.3 * heavy:
                        for _ in range(rm_.randint(2, 7) + (2 if heavy > 0.3 else 0)):
                            f = f_lo + (f_hi - f_lo) * t / seg_len
                            if f > f_hi:
                                break
                            nx, ny = plan.normal(k, f)
                            big = rm_.u(0.7, 1.25) * (1.0 + 0.35 * heavy)
                            ra, rt = rm_.u(0.045, 0.085) * big, rm_.u(0.022, 0.040) * big
                            x0, y0 = plan.pt(k, f, d_j + 0.75 * rt)
                            moss.tuft(x0, y0, z_j, atan2(nx, -ny), ra, rt, rm_.u(0.014, 0.028) * big)
                            t += ra * rm_.u(1.0, 1.6)
                    t += rm_.u(0.12, 0.5)
        # scattered cushions: treads (north and neglected), coping top (neglected)
        n_sc = int((26 + 110 * heavy) * mg1 * D / 6.0)
        for _ in range(n_sc * 3):
            k, f = rm_.randint(0, plan.n - 1), rm_.u(0.03, 0.97)
            c = rm_.randint(0, n_steps - 1)
            d0 = 0.05 if c == 0 else d_land + (c - 1) * going
            d1 = d_land + c * going
            dd_ = d0 + (d1 - d0 - 0.07) * rm_.u(0.05, 0.75) ** 1.6
            if c == 0 and not (fa + 0.04 < f < 1 - fa - 0.04):
                continue
            x0, y0 = plan.pt(k, f, dd_)
            nw = _ss(-0.5, 0.7, y0 / Rext)
            if rm_.u(0, 1) > 0.12 + 0.5 * nw + 0.38 * heavy:
                continue
            rr_ = rm_.u(0.024, 0.05) * (1.0 + 0.4 * heavy)
            moss.tuft(x0, y0, z_pod - c * rise, rm_.u(0, 6.28), rr_ * rm_.u(1.0, 1.5), rr_, rr_ * rm_.u(0.35, 0.55))
        # the damp nook where the coping runs into a pier
        for k in range(plan.n):
            for f in (fa + 0.012, 1.0 - fa - 0.012):
                x0, y0 = plan.pt(k, f, -0.5 * T)
                if rm_.u(0, 1) > (0.3 + 0.5 * _ss(-0.5, 0.7, y0 / Rext)) * mg1 + 0.3 * heavy:
                    continue
                nx, ny = plan.normal(k, f)
                dd_ = rm_.u(-T + 0.07, -0.10)
                for _ in range(rm_.randint(1, 3)):
                    x0, y0 = plan.pt(k, f, dd_)
                    ra = rm_.u(0.022, 0.04)
                    moss.tuft(x0, y0, z_rim + 0.002, atan2(ny, nx), ra, rm_.u(0.011, 0.017), rm_.u(0.008, 0.014))
                    dd_ += ra * rm_.u(1.2, 1.7)
                    if dd_ > -0.05:
                        break
        if heavy > 0.05:
            for _ in range(int(70 * heavy * D / 6.0)):
                k, f = rm_.randint(0, plan.n - 1), rm_.u(fa + 0.05, 1 - fa - 0.05)
                x0, y0 = plan.pt(k, f, rm_.u(-T + 0.05, -0.02))
                if rm_.u(0, 1) > 0.25 + 0.75 * _ss(-0.5, 0.7, y0 / Rext):
                    continue
                rr_ = rm_.u(0.02, 0.045)
                moss.tuft(x0, y0, z_rim + 0.003, rm_.u(0, 6.28), rr_ * rm_.u(1.0, 1.6), rr_, rr_ * rm_.u(0.35, 0.5))
        moss.mesh("moss", mat("moss"))

    # ---- water: the outline of the pool first, then everything that falls (it tells the pool where to ripple) --
    per = max(plan.sub, int(ceil(232.0 / plan.n))) if flowing else max(plan.sub, 12)
    out = plan.outline(-T + 0.02, per)
    phis = [atan2(y, x) for x, y in out]
    edges = [hypot(x, y) for x, y in out]
    r_e = min(edges)
    r0 = 0.9 * r_ped * cos(pi / 8)

    def edge_at(a):
        j = min(range(len(phis)), key=lambda q: abs(((phis[q] - a + pi) % (2 * pi)) - pi))
        return edges[j]

    # ---- falling water: torn sheets, lion-mouth arcs, finial jets; foam, rings, crowns and drops at every landing --
    imps = []                                            # (x, y, strength, phase) of everything that strikes the pool
    bimps = [[] for _ in td]                             # ... and the water of each bowl
    hits = []                                            # the same landings as _water_marks wants them
    bhits = [[] for _ in td]
    if flowing:
        froth = _Froth(rng.fork("froth"))
        sk = 0.55 + 0.45 * ks

        def splash(x, y, z, am, ra_, rt_, k_, n_arc=2):
            """what one landing throws up: a crown of spray round it and a few strings of drops flung clear (the
            boil, the foam and the rings are painted on the water itself - see hits / _water_marks)"""
            froth.skirt(x, y, z - 0.005, am, 0.5 * ra_, 0.9 * rt_, rw.u(0.06, 0.1) * k_, seg=int(clamp(rt_ / 0.02, 10, 20)))
            for _ in range(n_arc):
                a2 = am + rw.u(-1.3, 1.3) + (pi if rw.u() < 0.3 else 0.0)
                froth.spray(x + 0.5 * ra_ * cos(a2), y + 0.5 * ra_ * sin(a2), z, a2, rw.u(0.9, 1.6) * sqrt(k_),
                            rw.u(0.9, 1.25), rw.u(0.009, 0.016) * k_)

        for b in td:
            i = b["i"]
            z_end = (td[i - 1]["zw"] if i > 0 else z_w)
            o, lands = _sheet_mesh("sheets%d" % i, b, z_end - 0.012, A, fall, z_min=z_end + _W_CLEAR)
            froth.floor = z_end + _W_CLEAR
            k_i = sk * (1.0 if i == 0 else 0.68)
            zs_ = z_end + (0.011 if i == 0 else 0.007)
            if i == 0:
                r_max = r_e - 0.10
            else:
                r_max = (td[i - 1]["Rb"] - 0.65 * td[i - 1]["tw"]) * (1.0 - A) - 0.03
            by_sheet = {}
            for (r, a, hwid, si) in lands:
                r = min(r, r_max)
                by_sheet.setdefault(si, []).append((r, a, hwid))
                froth.mound(r * cos(a), r * sin(a), zs_ - 0.005, 0.02 * k_i + 0.9 * hwid, rw.u(0.012, 0.024) * k_i, seg=9)
            for si, ls in by_sheet.items():
                rm = sum(q[0] for q in ls) / len(ls)
                a_lo, a_hi = min(q[1] for q in ls), max(q[1] for q in ls)
                am = 0.5 * (a_lo + a_hi)
                rt_ = 0.5 * (a_hi - a_lo) * rm + max(q[2] for q in ls) + rw.u(0.06, 0.09) * k_i
                ra_ = min(rw.u(0.11, 0.16) * k_i, max(0.03, r_max + 0.02 - rm))
                splash(rm * cos(am), rm * sin(am), zs_, am, ra_, rt_, k_i, n_arc=2 if i == 0 else 1)
                seg_ = (rm * cos(a_lo), rm * sin(a_lo), rm * cos(a_hi), rm * sin(a_hi), 1.0 if i == 0 else 0.8,
                        min(ra_, 0.11 * k_i))
                (hits if i == 0 else bhits[i - 1]).append(seg_)
                # every sheet is one line source of rings on the water it strikes
                hit = (rm * cos(am), rm * sin(am), rw.u(0.7, 1.0), rw.u(0, 6.28))
                if i == 0:
                    imps.append(hit)
                else:
                    bimps[i - 1].append(hit)
        # lion-mouth arcs
        froth.floor = z_w + _W_CLEAR
        if n_sp > 0:
            for a in mask_angles:
                v0 = rw.u(0.85, 1.05) * (0.6 + 0.4 * flow) * sqrt(ks)
                tt = sqrt(2 * (tip_z - (z_w - 0.01)) / G)
                path = []
                for q in range(15):
                    t = tt * q / 14
                    rr_ = tip_r - 0.01 + v0 * t
                    path.append((rr_ * cos(a), rr_ * sin(a), tip_z - 0.5 * G * t * t))
                _stream_mesh("mouth_arc", lod_thin(path, 0.5, 0.2), 0.024 * ks, 0.05 * ks, fall, uo=rw.u(0, 3), foam_len=0.12 * ks,
                             z_min=z_w + _W_CLEAR)
                rl_ = min(tip_r - 0.01 + v0 * tt, r_e - 0.12)
                x_, y_ = rl_ * cos(a), rl_ * sin(a)
                froth.mound(x_, y_, z_w + 0.005, 0.04 * sk, 0.04 * sk, seg=8)
                splash(x_, y_, z_w + 0.011, a, 0.13 * sk, 0.10 * sk, sk)
                imps.append((x_, y_, 1.3, rw.u(0, 6.28)))
                hits.append((x_, y_, x_, y_, 1.0, 0.085 * sk))
        # finial jets: thin streams playing out of the nozzle into the top bowl
        tb = td[-1]
        froth.floor = tb["zw"] + _W_CLEAR
        nj = 8
        for j in range(nj):
            a = 2 * pi * (j + 0.5) / nj
            vu = rw.u(1.25, 1.55) * sqrt(ks)
            vo = min(rw.u(0.34, 0.42), 0.78 * tb["Rb"] / 0.6) * sqrt(ks)
            z_land = tb["zw"] - 0.01
            tt = (vu + sqrt(vu * vu + 2 * G * (z_tip - z_land))) / G
            if vo * tt > 0.8 * tb["Rb"]:
                vo = 0.8 * tb["Rb"] / tt

            def jet_fn(u, v, a=a, vu=vu, vo=vo, tt=tt, zfl=tb["zw"] + _W_CLEAR):
                t = tt * (1 - v)
                rr_ = 0.006 + vo * t
                zz = z_tip + vu * t - 0.5 * G * t * t
                w = 0.010 * ks + 0.035 * ks * (t / tt)
                tx = (u - 0.5) * w
                return (rr_ * cos(a) - tx * sin(a), rr_ * sin(a) + tx * cos(a), max(zz, zfl))

            o = surface("jet", jet_fn, nu=2, nv=14)
            assign(o, fall)
            paint(o, wear=lambda co, n, a=a: 0.12 + 0.5 * np.clip(np.hypot(co[:, 0], co[:, 1]) / max(vo * tt, 0.05), 0, 1),
                  special=lambda co, n, zl=z_land: 0.8 * _ssn(zl + 0.09, zl + 0.01, co[:, 2]) * (np.hypot(co[:, 0], co[:, 1]) > 0.05), mode="set")
            xj, yj = vo * tt * cos(a), vo * tt * sin(a)
            froth.mound(xj, yj, tb["zw"] + 0.001, 0.02 * sk, 0.016 * sk, seg=6)
            bimps[-1].append((xj, yj, 0.55, rw.u(0, 6.28)))
            bhits[-1].append((xj, yj, xj, yj, 0.6, 0.022 * sk))
            froth.skirt(xj, yj, tb["zw"], a, 0.016 * sk, 0.016 * sk, 0.03 * sk, seg=6)
        froth.mesh("froth", fall)
        fx("jet", "water_jet", (0, 0, z_tip), size=0.3 * ks)
        for k, s in enumerate(td[0]["sheets"][:: max(1, len(td[0]["sheets"]) // 3)][:3]):
            fx("drip%d" % k, "drip", (td[0]["Rb"] * cos(s["phi"]), td[0]["Rb"] * sin(s["phi"]), td[0]["lip"] - td[0]["tw"]), size=0.2)

    # ---- the pool: rings spreading from every landing, cross-hatching where they meet ----------------------
    if imps:
        ri = [hypot(q[0], q[1]) for q in imps]
        r_land = sum(ri) / len(ri)
        ra, rb2 = max(r_ped + 0.06, min(ri) - 0.42), min(max(ri) + 0.62, r_e - 0.04)
    else:
        r_land = 0.5 * (r_ped + r_e)
        ra, rb2 = r_ped + 0.06, r_e - 0.04

    def pool_z(r, a, e_):
        # dead flat: the `water` material draws the lop, the rings and the foam (flat water also keeps the baked
        # edge-wear mask, which carries the agitation, clean)
        return z_w

    if imps:
        rn_s = [r0, r_ped + 0.03]
        kf = max(2, int((rb2 - ra) / 0.03 + 0.5))
        rn_s += [ra + (rb2 - ra) * k / kf for k in range(kf + 1)]
        while rn_s[-1] < r_e - 0.13:
            rn_s.append(rn_s[-1] + 0.11)
        rn_s.append(r_e)
    else:
        rn_s = [r0, r_ped + 0.05, 0.5 * (r_ped + r_e), r_e - 0.1, r_e]
    rn_s = sorted(set(round(r, 4) for r in rn_s if r <= r_e + 1e-6))
    if Lv:
        # the same sheet with fewer spokes and rings (corners of the plan stay); its edge lies deeper in the wall,
        # whose facets are coarser too. edge_at / r_e above keep the full outline: the pool lights hang on them.
        sa_ = (2, 5 if p.plan == "octagon" else 3)[Lv - 1]
        out_m = plan.outline(-T + 0.06, per)
        ix = [k * per + q for k in range(plan.n) for q in range(0, per, sa_) if q == 0 or per - q >= 0.5 * sa_]
        ph_m, ed_m = [atan2(out_m[q][1], out_m[q][0]) for q in ix], [hypot(out_m[q][0], out_m[q][1]) for q in ix]
        rn_m = rn_s[:2] + rn_s[2:-1][::(2, 5)[Lv - 1]] + [min(ed_m)]
        pool_o = _polar("pool_water", ph_m, ed_m, r0, rn_m, pool_z, surf_m)
    else:
        pool_o = _polar("pool_water", phis, edges, r0, rn_s, pool_z, surf_m)
    if hits:
        _water_marks(pool_o, hits)

    # ---- water in the bowls ---------------------------------------------------------------------------
    for b in td:
        i, Rb, tw, n = b["i"], b["Rb"], b["tw"], b["n"]
        nseg = (n * 12 if n > 0 else 96) if flowing else (n * 6 if n > 0 else 60)
        lobe = _lobe_fn(n, b["phase"])
        ph_b = [b["phase"] + 2 * pi * j / nseg for j in range(nseg)]
        ed_b = []
        for a in ph_b:
            c = lobe(a)
            if flowing:
                ed_b.append((Rb - (0.60 - 0.30 * c) * tw) * (1.0 - A * (1.0 - c)) if n > 0 else Rb - 0.30 * tw)
            else:
                ed_b.append((Rb - 0.80 * tw) * (1.0 - A * (1.0 - c)))
        r_in = (td[i + 1]["r_seat"] if i + 1 < len(td) else 0.085 * ks) * 0.8
        re_b = min(ed_b)
        # the tier above (or the finial jets) lands here
        if i + 1 < len(td):
            up_ = td[i + 1]
            rl = up_["Rb"] + 0.43 * up_["tw"] + 0.31 * (0.5 + 0.5 * ks) * _fall_time(up_["lip"] - b["zw"], 0.35)
        else:
            rl = 0.55 * Rb
        rl = min(rl, re_b - 0.05)
        BI = np.array(bimps[i], dtype=float) if bimps[i] else np.zeros((0, 4))
        if len(BI):
            rl = min(float(np.mean(np.hypot(BI[:, 0], BI[:, 1]))), re_b - 0.05)
        if flowing:
            ra, rb2 = max(r_in + 0.03, rl - 0.2), min(rl + 0.2, re_b - 0.02)
            kf = max(2, int((rb2 - ra) / 0.026 + 0.5))
            rn_b = [r_in] + [ra + (rb2 - ra) * k / kf for k in range(kf + 1)] + [re_b]
        else:
            rn_b = [r_in, 0.5 * (r_in + re_b), re_b]
        rn_b = sorted(set(round(r, 4) for r in rn_b))
        if Lv:                                               # as many spokes a lobe as the stone has facets
            sb_ = max(1, (nseg // n) // ((6 if n % 2 == 0 else 8) if Lv == 1 else 4)) if n > 0 else (2, 4)[Lv - 1]
            ph_b, ed_b = ph_b[::sb_], ed_b[::sb_]
            rn_b = rn_b[:1] + rn_b[1:-1][::(2, 4)[Lv - 1]] + rn_b[-1:]
        bw_o = _polar("bowl_water%d" % i, ph_b, ed_b, r_in, rn_b, lambda r, a, e_, zw=b["zw"]: zw, surf_m)
        if bhits[i]:
            _water_marks(bw_o, bhits[i])

    # ---- lantern standards on the rim piers (after the water: their panes must draw over it) -----------
    light_specs = []
    if lan_idx:
        lbase, lz = _lantern_standard(p, rng.fork("lantern"))
        lm = []
        for k in lan_idx:
            cx, cy, cz = pier_tops[k]
            ang = plan.pier_angle(k) + (45.0 if p.plan == "quatrefoil" else 0.0)
            lm.append(trs((cx, cy, cz - 0.003), (0, 0, ang + 90.0 + rng.u(-2, 2)), s_orn))
            light_specs.append(("lantern", (cx, cy, cz + lz * s_orn)))
        instances_to_mesh("lanterns", lbase, lm, keep_base=False)

    # ---- submerged pool lights: bronze bulkheads on the pool floor --------------------------------------
    n_pl = int(p.pool_lights)
    pool_pos = []
    if n_pl > 0:
        s_pl = clamp(ks, 0.65, 1.0)
        # (material budget: the lens shares the lantern glass and the lamp shares the gas mantle)
        lens_m = mat("glass_lantern", glow=0.42 if p.lit else 0.0)
        lamp_m = mat("emissive", "#ffcf8a", glow=5) if p.lit else mat("glass_frosted", glow=0.0)
        r_pls = []
        for i in range(n_pl):
            a = radians(360.0 * i / n_pl)
            ea = edge_at(a)
            r_pls.append(max(min(0.5 * (r_land + 0.1 + ea), ea - 0.30), r_ped + 0.30))
        r_pl = min(r_pls)
        fbase, h_lens = _bulkhead(s_pl, bronze, lens_m, lamp_m)
        fm = []
        for i in range(n_pl):
            a = radians(360.0 * i / n_pl)
            x, y = r_pl * cos(a), r_pl * sin(a)
            # set just under the skin on a riser: only the bezel and the glowing lens break the dark surface
            fm.append(trs((x, y, z_w - 0.061 * s_pl), (0, 0, degrees(a)), 1.0))
            # the light itself sits well under the water: it lights the stone and the falling water from below
            # (the water surface casts no shadow) and is far enough from the bezel, the lens and the foam not
            # to burn a hot spot into them
            pool_pos.append((x, y, z_w - 0.16))
        instances_to_mesh("pool_lights", fbase, fm, keep_base=False, uv_jitter=False)
        if p.lit:
            _water_glow(pool_o, [(q[0], q[1]) for q in pool_pos], 0.55 * s_pl)

    # ---- light sockets (first two cast shadows in the review: nearest lantern, then a pool light) -----
    if p.lit:
        order = []
        lan = [q for _, q in light_specs]
        if lan:
            order.append(("lantern", lan[0]))
        if pool_pos:
            order.append(("pool", pool_pos[0]))
        order += [("lantern", q) for q in lan[1:]] + [("pool", q) for q in pool_pos[1:]]
        cnt = {"lantern": 0, "pool": 0}
        for kind, q in order:
            nm_ = "%s_%d" % (kind, cnt[kind])
            cnt[kind] += 1
            if kind == "lantern":
                light(nm_, q, color="#ffbf6a", intensity=14.0, range=7.5, flicker=0.14, shadow_near=0.24)
            else:
                light(nm_, q, color="#ffd7a4", intensity=3.0, range=1.2 + 3.0 * ks, flicker=0.03, shadow_near=0.2)

    socket("coping", "attach", (0, -R + T / 2, z_rim + 0.006), role="prop_top")
