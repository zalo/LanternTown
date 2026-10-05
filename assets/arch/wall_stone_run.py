"""A run of old masonry wall along X. Stone walls are a lime-mortar core faced on one or both sides
with individually laid stones (random rubble brought to courses with jumpers, or fine-jointed ashlar)
that sit a few millimetres proud of flush buff pointing; brick walls are hand-made brick with
pilaster strips, the odd proud brick and a double clay-tile creasing. The wall stands on a weathered
plinth course and is finished with separate coping stones (flat slabs cramped with iron, saddle-back
or half-round) of uneven length and thickness, each bedded on its own wedge of mortar so the top
line steps and tilts with `age`. Free ends are bonded with alternating long / short dressed quoins
and finish dead flat on x = +-length/2 so runs butt together; an end may instead carry a coursed
pier with a moulded cap (pyramid, ball finial or a small gas lantern hung from a cast crook standard).
Retaining walls get a battered face drained by clay weep pipes with algae streaks below them.
Moss film and cushions on the coping, an algae band above the plinth and a damp tide line at the
foot are all driven by `moss` / `age`.
"""
from tk import *
import math
import numpy as np

META = {"name": "Stone wall run", "class": "component", "mount": "ground", "focus": 0.66}
PARAMS = {
    "length": {"default": 4.0, "min": 1.0, "max": 14.0, "desc": "run length along X (m); the run spans exactly x = -length/2 .. +length/2"},
    "height": {"default": 1.15, "min": 0.45, "max": 3.0, "desc": "nominal height to the top of the coping (m)"},
    "thickness": {"default": 0.42, "min": 0.25, "max": 0.8, "desc": "wall thickness at the top (m)"},
    "facing": {"default": "rubble", "choices": ["rubble", "ashlar", "brick"], "desc": "face construction"},
    "coping": {"default": "flat", "choices": ["flat", "saddle", "half_round"], "desc": "coping stone section"},
    "pier_ends": {"default": "left", "choices": ["none", "left", "right", "both"], "desc": "which ends finish in a pier (other ends are flat quoined ends that butt to the next run)"},
    "pier_width": {"default": 0.56, "min": 0.4, "max": 0.9, "desc": "pier side (m)"},
    "pier_rise": {"default": 0.24, "min": 0.05, "max": 0.8, "desc": "pier shaft height above the coping (m)"},
    "pier_cap": {"default": "pyramid", "choices": ["pyramid", "ball", "lantern"], "desc": "what finishes the pier cap"},
    "lantern_cd": {"default": 10.0, "min": 5.0, "max": 80.0, "desc": "pier lantern intensity (cd)"},
    "kind": {"default": "freestanding", "choices": ["freestanding", "retaining"], "desc": "freestanding = faced both sides; retaining = faced on -Y only, back parged, drained by weep pipes"},
    "batter": {"default": 0.0, "min": 0.0, "max": 0.12, "desc": "front face batter (m of lean per m of height; the base gets thicker)"},
    "plinth": {"default": 0.18, "min": 0.0, "max": 0.35, "desc": "height of the projecting plinth course (0 = none)"},
    "weep_spacing": {"default": 1.5, "min": 0.0, "max": 4.0, "desc": "spacing of the clay weep pipes of a retaining wall (0 = none; free-standing walls never have them)"},
    "stone_size": {"default": 1.0, "min": 0.7, "max": 1.6, "desc": "scale of the facing stones / courses"},
    "stone": {"default": "#d6cab2", "desc": "stone tint"},
    "moss": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "moss / algae amount on coping, joints and damp base"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "settlement: stepped and tilted coping stones, chipped arrises, knocked corners"},
    "tie_ring": {"default": True, "desc": "wrought-iron tethering ring on the first pier"},
}
VARIANTS = {
    "ashlar_gate_piers": {"facing": "ashlar", "coping": "saddle", "pier_ends": "both", "pier_cap": "ball", "length": 5.0,
                          "height": 1.45, "pier_rise": 0.4, "stone": "#e2d6ba", "moss": 0.35, "age": 0.3, "seed": 4},
    "brick_garden": {"facing": "brick", "coping": "half_round", "pier_ends": "right", "length": 4.6, "height": 1.9,
                     "thickness": 0.34, "pier_width": 0.5, "pier_rise": 0.3, "moss": 0.5, "seed": 11},
    "retaining_terrace": {"kind": "retaining", "batter": 0.09, "height": 1.75, "length": 6.0, "thickness": 0.5, "stone_size": 1.35,
                          "pier_ends": "none", "weep_spacing": 1.3, "moss": 1.0, "age": 0.9, "stone": "#c8bda6", "seed": 23},
    "parapet_lantern": {"facing": "ashlar", "height": 0.8, "length": 3.6, "thickness": 0.36, "pier_ends": "both", "pier_cap": "lantern",
                        "pier_rise": 0.3, "pier_width": 0.48, "stone": "#e0d0ac", "moss": 0.45, "lantern_cd": 7.0, "seed": 31},
}

TOOL_DEG = 27.0        # stone_smooth's veining runs ~27 deg off U: turn the UVs so it reads as level bedding
BRICK = (0.225, 0.075)  # brick_red module (tile 0.9 m = 4 x 12 bricks)


# ------------------------------------------------------------------ small mesh kit (numpy -> one mesh per material)
def _tint(h, k):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 * k[j] for j, i in enumerate((0, 2, 4))]
    return "#%02x%02x%02x" % tuple(int(round(min(1.0, max(0.0, v)) * 255)) for v in c)


def _nrm(P):
    Q = np.roll(P, -1, axis=0)
    return np.array([np.sum((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])),
                     np.sum((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])),
                     np.sum((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1]))])


class _Kit:
    """Collects pieces per bucket; fixes winding, applies the batter shear, clamps to the run, writes
    metre-scale box UVs (random offset per piece, optional rotation per bucket) and carries
    per-vertex weathering values (grime / wear / special) decided while modelling."""

    def __init__(self, half, shear, rng):
        self.b, self.half, self.shear, self.r = {}, half, shear, rng
        self.tris = 0
        self.rot = {}

    def add(self, key, V, F, out=None, inward=False, shear=True, g=0.0, w=0.0, sp=0.0, uvoff=True, fix=True):
        V = np.array(V, dtype=float)
        c = V.mean(axis=0)
        F2 = []
        for f in F:
            P = V[list(f)]
            ref = out if out is not None else (P.mean(axis=0) - c)
            d = float(np.dot(_nrm(P), ref))
            if inward:
                d = -d
            F2.append(tuple(f) if (d >= 0 or not fix) else tuple(reversed(f)))
        if shear:
            V = self.shear(V)
        V[:, 0] = np.clip(V[:, 0], -self.half, self.half)
        bk = self.b.setdefault(key, {"v": [], "f": [], "uv": [], "n": 0, "g": [], "w": [], "s": []})
        off = np.array([self.r.u(0, 3.0), self.r.u(0, 3.0)])
        if not uvoff:
            off = np.zeros(2)
        ang = self.rot.get(key, 0.0)
        ca, sa = math.cos(ang), math.sin(ang)
        for f in F2:
            P = V[list(f)]
            a = np.abs(_nrm(P))
            if a[1] >= a[0] and a[1] >= a[2]:
                uv = P[:, [0, 2]]
            elif a[0] >= a[2]:
                uv = P[:, [1, 2]]
            else:
                uv = P[:, [0, 1]]
            if ang:
                uv = np.stack([uv[:, 0] * ca - uv[:, 1] * sa, uv[:, 0] * sa + uv[:, 1] * ca], axis=1)
            bk["uv"].extend((uv + off).ravel().tolist())
            bk["f"].append(tuple(i + bk["n"] for i in f))
            self.tris += len(f) - 2
        bk["v"].append(V)
        bk["n"] += len(V)
        for nm, val in (("g", g), ("w", w), ("s", sp)):
            bk[nm].append(np.broadcast_to(np.asarray(val, dtype=float), (len(V),)).copy())

    def finish(self, key, material, angle):
        bk = self.b.get(key)
        if not bk:
            return None
        o = mesh_object("w_" + key, np.concatenate(bk["v"]).tolist(), bk["f"])
        me = o.data
        uvl = me.uv_layers.new(name="UVMap")
        if uvl is not None and len(uvl.data) * 2 == len(bk["uv"]):
            uvl.data.foreach_set("uv", bk["uv"])
            uv_keep(o)
        else:
            print("DBG uv fallback", key)
            uv_box(o)
        for nm, attr in (("g", "m_grime"), ("w", "m_wear"), ("s", "m_special")):
            arr = np.clip(np.concatenate(bk[nm]), 0.0, 1.0).astype(np.float32)
            if float(arr.max()) > 0.0:
                if len(arr) == len(me.vertices):
                    at = me.attributes.get(attr) or me.attributes.new(attr, 'FLOAT', 'POINT')
                    at.data.foreach_set("value", arr)
                else:
                    print("DBG mask size mismatch", key, len(arr), len(me.vertices))
        assign(o, material)
        smooth(o, angle)
        return o


def _block(x0, x1, y0, y1, z0, z1, ch=0.01, skip=(), r=None, jit=0.0):
    """Chamfered box (24 verts). skip = {(axis, sign)} faces that are buried."""
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    h = np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])
    sg = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    if ch <= 0:
        V = [c + np.array(s) * h for s in sg]
        ix = {s: i for i, s in enumerate(sg)}
        F = []
        for a in range(3):
            b, d = (a + 1) % 3, (a + 2) % 3
            for s in (-1, 1):
                if (a, s) in skip:
                    continue
                q = []
                for sb, sd in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    t = [0, 0, 0]
                    t[a], t[b], t[d] = s, sb, sd
                    q.append(ix[tuple(t)])
                F.append(tuple(q))
        return np.array(V), F
    ch = min(ch, float(h.min()) * 0.45)
    V, ix = [], {}
    for s in sg:
        jv = np.array([r.u(-jit, jit), r.u(-jit, jit), r.u(-jit, jit)]) if (r is not None and jit > 0) else 0.0
        for a in range(3):
            pnt = np.array(s, dtype=float) * h
            for b in range(3):
                if b != a:
                    pnt[b] = s[b] * (h[b] - ch)
            ix[(a,) + s] = len(V)
            V.append(c + pnt + jv)
    F = []
    for a in range(3):
        b, d = (a + 1) % 3, (a + 2) % 3
        for s in (-1, 1):
            if (a, s) in skip:
                continue
            q = []
            for sb, sd in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                t = [0, 0, 0]
                t[a], t[b], t[d] = s, sb, sd
                q.append(ix[(a,) + tuple(t)])
            F.append(tuple(q))
    for a in range(3):
        for b in range(a + 1, 3):
            d = 3 - a - b
            for sa in (-1, 1):
                for sb in (-1, 1):
                    if (a, sa) in skip and (b, sb) in skip:
                        continue
                    t0 = [0, 0, 0]
                    t0[a], t0[b], t0[d] = sa, sb, -1
                    t1 = list(t0)
                    t1[d] = 1
                    t0, t1 = tuple(t0), tuple(t1)
                    F.append((ix[(a,) + t0], ix[(a,) + t1], ix[(b,) + t1], ix[(b,) + t0]))
    for s in sg:
        if sum(1 for a in range(3) if (a, s[a]) in skip) >= 2:
            continue
        F.append((ix[(0,) + s], ix[(1,) + s], ix[(2,) + s]))
    return np.array(V), F


def _prism(prof, xs, inset, bottom=True, caps=(True, True), zfn=None):
    """Extrude a convex (y, z) section along X through rings xs; inset[k] shrinks ring k (end chamfers).
    The edge from the last profile point back to the first is the bed (dropped when bottom=False)."""
    prof = np.array(prof, dtype=float)
    n = len(prof)
    cy = (prof[:, 0].min() + prof[:, 0].max()) / 2
    cz = (prof[:, 1].min() + prof[:, 1].max()) / 2
    hy = max((prof[:, 0].max() - prof[:, 0].min()) / 2, 1e-4)
    hz = max((prof[:, 1].max() - prof[:, 1].min()) / 2, 1e-4)
    rings = []
    for k, x in enumerate(xs):
        e = inset[k]
        y = cy + (prof[:, 0] - cy) * (1 - e / hy)
        z = cz + (prof[:, 1] - cz) * (1 - e / hz) + (zfn(x) if zfn else 0.0)
        rings.append(np.stack([np.full(n, x), y, z], axis=1))
    V = np.concatenate(rings)
    F = []
    for k in range(len(xs) - 1):
        for j in range(n):
            if not bottom and j == n - 1:
                continue
            j2 = (j + 1) % n
            F.append((k * n + j, k * n + j2, (k + 1) * n + j2, (k + 1) * n + j))
    if caps[0]:
        F.append(tuple(range(n)))
    if caps[1]:
        F.append(tuple((len(xs) - 1) * n + j for j in range(n)))
    return V, F




def _knock(V, r, e, corners=(0, 1, 2, 3, 4, 5, 6, 7)):
    """Knock one corner off a chamfered _block (24 verts): its three verts slide back along the arrises."""
    si = r.pick(list(corners))
    s = (((si >> 2) & 1) * 2 - 1, ((si >> 1) & 1) * 2 - 1, (si & 1) * 2 - 1)
    for a in range(3):
        for b in range(3):
            if b != a:
                V[3 * si + a][b] -= s[b] * e * r.u(0.6, 1.0)
    return V


class _Line:
    """Bed line z(x): constant, or piecewise linear on the knot grid shared by the whole wall, so the
    stones above and below a bed meet it along exactly the same polyline (no cracks in the pointing)."""

    def __init__(self, z=0.0, x0=0.0, dx=1.0, vals=None):
        self.z, self.x0, self.dx, self.v = z, x0, dx, vals

    def __call__(self, x):
        if self.v is None:
            return self.z
        t = (x - self.x0) / self.dx
        i = min(max(int(math.floor(t)), 0), len(self.v) - 2)
        f = min(max(t - i, 0.0), 1.0)
        return self.v[i] * (1 - f) + self.v[i + 1] * f

    def knots(self, a, b):
        if self.v is None:
            return []
        i = int(math.floor((a - self.x0) / self.dx)) + 1
        out = []
        while self.x0 + i * self.dx < b - 2e-4:
            if self.x0 + i * self.dx > a + 2e-4:
                out.append(self.x0 + i * self.dx)
            i += 1
        return out


def _strip(Vm, Fm, P, Q):
    """Stitch two polylines of (u, z) points that run the same way into a triangle strip."""
    def par(S):
        d = [0.0]
        for k in range(1, len(S)):
            d.append(d[-1] + math.hypot(S[k][0] - S[k - 1][0], S[k][1] - S[k - 1][1]))
        return [v / max(d[-1], 1e-9) for v in d]
    b0 = len(Vm)
    Vm.extend((q[0], q[1], 0.0) for q in P)
    b1 = len(Vm)
    Vm.extend((q[0], q[1], 0.0) for q in Q)
    tp, tq = par(P), par(Q)
    i = j = 0
    while i < len(P) - 1 or j < len(Q) - 1:
        if j >= len(Q) - 1 or (i < len(P) - 1 and tp[i + 1] <= tq[j + 1]):
            Fm.append((b0 + i, b0 + i + 1, b1 + j))
            i += 1
        else:
            Fm.append((b0 + i, b1 + j + 1, b1 + j))
            j += 1


def _pillow(r, xs4, zb, zt, hi, joint, c, d1, bulge, rnd, tilt, chip=0.0, sub=None):
    """One facing stone seen from the front, as (u, v, depth) with depth 0 = the pointing plane.
    The cell is a quad: xs4 = (left bottom, left top, right bottom, right top) x between the bed lines
    zb / zt it shares with its neighbours. The stone keeps `joint` clear of the cell edge, has rounded
    corners (rnd), an arris of width c rising d1 above the pointing and a tilted, slightly bulged
    face; chip > 0 breaks one corner of the face. Returns stone verts, faces and the mortar skirt
    (verts, faces) that fills the cell around it flush to the cell edge."""
    xm = (xs4[0] + xs4[1] + xs4[2] + xs4[3]) / 4
    w = max((xs4[2] + xs4[3] - xs4[0] - xs4[1]) / 2, 1e-3)
    h = max(zt(xm) - zb(xm), 1e-3)
    cu, cv = min(c, w * 0.3), min(c, h * 0.3)
    ju, jv = max(0.3, 1 - joint / w), max(0.3, 1 - joint / h)
    au, av = 1 - 2 * cu / w, 1 - 2 * cv / h
    if hi:
        us = [-1, -au, r.u(-0.25, 0.25), au, 1]
        vs = [-1, -av, r.u(-0.25, 0.25), av, 1]
    elif sub:                                     # dressed face cut into a grid so stains can cross it softly
        nu, nv = max(1, int(round(w / sub[0]))), max(1, int(round(h / sub[1])))
        us = [-1] + [-au + 2 * au * k / nu for k in range(nu + 1)] + [1]
        vs = [-1] + [-av + 2 * av * k / nv for k in range(nv + 1)] + [1]
    else:
        us, vs = [-1, -au, au, 1], [-1, -av, av, 1]
    n, m = len(us), len(vs)
    kq = {(sa, sb): r.u(rnd[0], rnd[1]) for sa in (-1, 1) for sb in (-1, 1)}
    tu, tv = r.u(-tilt, tilt), r.u(-tilt, tilt)
    V = []
    for j in range(m):
        for i in range(n):
            a, b = us[i], vs[j]
            k = kq[(1 if a >= 0 else -1, 1 if b >= 0 else -1)]
            a2 = a * math.sqrt(max(0.0, 1 - k * b * b / 2)) * ju
            b2 = b * math.sqrt(max(0.0, 1 - k * a * a / 2)) * jv
            tb = (b2 + 1) / 2
            xa = xs4[0] + (xs4[1] - xs4[0]) * tb
            xb = xs4[2] + (xs4[3] - xs4[2]) * tb
            u = xa + (xb - xa) * (a2 + 1) / 2
            z0, z1 = zb(u), zt(u)
            if i in (0, n - 1) or j in (0, m - 1):
                d = 0.0
            else:
                ring1 = i in (1, n - 2) or j in (1, m - 2)
                d = d1 + (0.0 if ring1 else bulge) + tu * a * w / 2 + tv * b * h / 2
                d = max(d, 0.0025)
            V.append([u, z0 + (z1 - z0) * tb, d])
    if chip > 0:
        i, j = r.pick([1, n - 2]), r.pick([1, m - 2])
        q = V[j * n + i]
        dist = max(math.hypot(q[0] - xm, q[1] - (zb(xm) + zt(xm)) / 2), 1e-3)
        f = min(0.4, chip / dist)
        q[0] += (xm - q[0]) * f
        q[1] += ((zb(xm) + zt(xm)) / 2 - q[1]) * f
        q[2] *= 0.3
    F = []
    for j in range(m - 1):
        for i in range(n - 1):
            a, b, cc, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            cj, ci = j in (0, m - 2), i in (0, n - 2)
            if cj and ci:
                if (j == 0) == (i == 0):          # outer corner is a or cc -> split a-cc
                    F += [(a, b, cc), (a, cc, d)]
                else:
                    F += [(a, b, d), (b, cc, d)]
            else:
                F.append((a, b, cc, d))
    Vm, Fm = [], []
    lb, lt, rb_, rt = (xs4[0], zb(xs4[0])), (xs4[1], zt(xs4[1])), (xs4[2], zb(xs4[2])), (xs4[3], zt(xs4[3]))
    _strip(Vm, Fm, [lb] + [(x, zb(x)) for x in zb.knots(xs4[0], xs4[2])] + [rb_], [V[i] for i in range(n)])
    _strip(Vm, Fm, [lt] + [(x, zt(x)) for x in zt.knots(xs4[1], xs4[3])] + [rt], [V[(m - 1) * n + i] for i in range(n)])
    _strip(Vm, Fm, [lb, lt], [V[j * n] for j in range(m)])
    _strip(Vm, Fm, [rb_, rt], [V[j * n + n - 1] for j in range(m)])
    return np.array(V), F, np.array(Vm), Fm


def _dome(r, cx, cy, cz, rx, ry, rz, n=6):
    """Low moss cushion: apex, shoulder ring, skirt ring sunk a little into what it grows on."""
    a0 = r.u(0, 6.28)
    V = [(cx + r.u(-0.15, 0.15) * rx, cy + r.u(-0.15, 0.15) * ry, cz + rz)]
    for f, zf in ((0.62, 0.7), (1.0, -0.15)):
        for k in range(n):
            a = a0 + k * 2 * math.pi / n
            q = f * r.u(0.9, 1.1)
            V.append((cx + rx * q * math.cos(a), cy + ry * q * math.sin(a), cz + rz * zf))
    F = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)]
    for k in range(n):
        a, b = 1 + k, 1 + (k + 1) % n
        F.append((a, b, b + n, a + n))
    return np.array(V), F


def _tube(P, R, n=8):
    """Round bar along a path lying in the YZ plane (outward winding)."""
    P = np.array(P, dtype=float)
    m = len(P)
    V, F = [], []
    n1 = np.array([1.0, 0.0, 0.0])
    for k in range(m):
        t = P[min(k + 1, m - 1)] - P[max(k - 1, 0)]
        t = t / max(np.linalg.norm(t), 1e-9)
        n2 = np.cross(t, n1)
        for j in range(n):
            a = 2 * math.pi * j / n
            V.append(P[k] + R[k] * (math.cos(a) * n1 + math.sin(a) * n2))
    for k in range(m - 1):
        for j in range(n):
            j2 = (j + 1) % n
            F.append((k * n + j, k * n + j2, (k + 1) * n + j2, (k + 1) * n + j))
    return np.array(V), F


# ------------------------------------------------------------------ the wall
def build(p, rng):
    L, H, T = float(p.length), float(p.height), float(p.thickness)
    xl, xr = -L / 2, L / 2
    fac, cop = p.facing, p.coping
    rubble, ashlar, brick = fac == "rubble", fac == "ashlar", fac == "brick"
    free = p.kind == "freestanding"
    sides = (-1, 1) if free else (-1,)
    age, mossk, seed = float(p.age), float(p.moss), int(p.seed)
    mk = min(1.0, 1.3 * mossk)
    lantern = p.pier_cap == "lantern" and p.pier_ends != "none"

    # ---- vertical layout: plinth | body (bands) | coping
    ov = clamp(0.1 * T, 0.03, 0.05)                 # coping overhang each side
    W = T + 2 * ov
    if cop == "flat":
        edge, rise = clamp(0.25 * T, 0.085, 0.14), 0.0
    elif cop == "saddle":
        edge, rise = clamp(0.14 * T, 0.05, 0.08), 0.27 * W
    else:
        edge, rise = 0.035, min(W * 0.5, 0.3) * 0.94
    k = min(1.0, 0.36 * H / (edge + rise))
    edge, rise = edge * k, rise * k
    tch = edge + rise
    hp = min(float(p.plinth), 0.26 * H)
    if hp < 0.04:
        hp = 0.0
    zc = H - tch
    Hb = zc - hp
    bt = float(p.batter)
    proj, pov = 0.035, 0.04                         # plinth projection, pier cap / plinth overhang

    def shear(V):
        if bt > 0:
            m = (V[:, 1] < -1e-6) & (V[:, 2] < zc)
            V[m, 1] -= bt * (zc - V[m, 2])
        return V

    kit = _Kit(L / 2, shear, rng.fork("uv"))
    ta = math.radians(TOOL_DEG)
    kit.rot["dress"] = kit.rot["cope"] = ta
    if ashlar:
        kit.rot["face0"] = kit.rot["face1"] = kit.rot["face2"] = ta

    # ---- smooth pseudo-noise along the wall for the weathering lines
    rn = rng.fork("wx")
    NPH = [(rn.u(2.0, 4.0), rn.u(0, 6.28), rn.u(5.0, 9.0), rn.u(0, 6.28), rn.u(11.0, 17.0), rn.u(0, 6.28)) for _ in range(8)]

    def wob(x, kk):                                 # 0..1
        a = NPH[kk]
        return 0.5 + 0.25 * np.sin(x * a[0] + a[1]) + 0.15 * np.sin(x * a[2] + a[3]) + 0.10 * np.sin(x * a[4] + a[5])

    # ---- piers
    pl, pr = p.pier_ends in ("left", "both"), p.pier_ends in ("right", "both")
    npier = int(pl) + int(pr)
    Px = min(float(p.pier_width), max(0.22, (L - 0.3) / max(npier, 1) - 2 * pov)) if npier else 0.0
    Py = max(float(p.pier_width), T + 0.12)
    yfp, ybp = -Py / 2 - bt * zc, Py / 2
    zpt = H + float(p.pier_rise)
    fxa = xl + (pov + Px if pl else 0.0)            # face / coping run between these
    fxb = xr - (pov + Px if pr else 0.0)
    Lf = fxb - fxa

    def in_pier(x, m=0.03):
        return (pl and x < xl + 2 * pov + Px + m) or (pr and x > xr - 2 * pov - Px - m)

    # ---- stone scale and detail level against the triangle budget
    s = float(p.stone_size)
    area = len(sides) * Lf * max(Hb, 0.1)
    hi = False
    afine = False
    nlan = (int(pl) + int(pr)) if lantern else 0
    ovh = 370.0 * L + 700.0 + 3600.0 * nlan         # coping, plinth, ends, cramps; lantern heads and ironwork
    if (not free) and p.weep_spacing > 0:
        ovh += 160 * (int(Lf / max(float(p.weep_spacing), 0.6)) + 1) + 350
    fb = clamp(11200.0 - ovh, 2000.0, 7600.0)       # what is left for the facing stones
    if rubble:
        est = area / (0.045 * s * s)
        hi = est * 58 <= fb
        if not hi and est * 36 > fb:
            s *= min(3.2, math.sqrt(est * 36 / fb))
    elif ashlar:
        est = area / (0.157 * s * s)
        afine = est * 80 <= fb                      # every block gridded, else only the damp foot course
        if not afine and est * 36 > fb - 750:
            s *= min(3.2, math.sqrt(est * 36 / (fb - 750)))
    rec = 0.008 if rubble else (0.004 if ashlar else 0.0)       # pointing plane behind the nominal face

    rb = rng.fork("bands")
    bh = (0.30 if rubble else 0.27) * s
    nb = max(1, int(round(Hb / bh)))
    wts = [rb.u(0.88, 1.12) for _ in range(nb)]
    zs = [hp]
    for wv in wts:
        zs.append(zs[-1] + Hb * wv / sum(wts))
    zs[-1] = zc
    qlens = (0.46, 0.27) if rubble else (0.62, 0.34)
    qmax = max(0.12, Lf * 0.3)

    def qlen(i, par):
        return 0.0 if brick else min(qlens[(i + par) % 2] * s, qmax)

    # ---- clay weep pipes of a retaining wall, just above the plinth
    ww, ro, ri, pjp = 0.15, 0.05, 0.037, 0.028
    weeps = []
    b0h = zs[1] - zs[0]
    if (not free) and p.weep_spacing > 0 and Hb > 0.2:
        wa, wb = fxa + (0.0 if pl else qlens[0] * s) + 0.3, fxb - (0.0 if pr else qlens[0] * s) - 0.3
        if wb - wa > 0.0:
            nw = int((wb - wa) / max(float(p.weep_spacing), 0.6)) + 1
            rw = rng.fork("weep")
            for kk in range(nw):
                weeps.append(wa + (wb - wa) * (kk + 0.5) / nw + rw.u(-0.08, 0.08) * min(1.0, (wb - wa)))
    wh = 0.14 if b0h - 0.14 >= 0.07 else b0h        # height of the mortar pocket the pipe is bedded in
    whz = hp + (min(wh / 2, 0.08) if not brick else 0.085)
    tdep = (min(0.16, T / 2 - 0.05) + pjp) if not brick else pjp - 0.004

    # ---- coping stone layout (needed early: the joints stain the face below them)
    rc = rng.fork("coping")
    cxa = fxa - (0.006 if pl else 0.0)
    cxb = fxb + (0.006 if pr else 0.0)
    cs = clamp(T / 0.42, 0.85, 1.3)
    cb = [cxa]
    while cb[-1] < cxb - 1e-4:
        rem = cxb - cb[-1]
        if rem <= 1.15 * cs:
            w = rem
        elif rem < 1.6 * cs:
            w = rem * rc.u(0.42, 0.58)
        else:
            w = rc.u(0.5, 1.1) * cs
        cb.append(min(cb[-1] + w, cxb))
    cb[-1] = cxb
    cjoints = cb[1:-1]
    rs = rng.fork("streak")
    cstreak = {sd: [(xj + rs.u(-0.02, 0.02), rs.u(0.05, 0.09), rs.u(0.22, 0.6) * min(1.0, Hb / 0.7), rs.u(0.5, 1.0))
                    for xj in cjoints if rs.chance(0.8)] for sd in (-1, 1)}

    # ---- weathering decided per vertex: damp tide line at the foot, algae band above the plinth,
    #      runs below the coping joints and below the weep pipes
    kd = 0.32 + 0.16 * age

    def wmask(V, side=-1):
        x, z = V[:, 0], V[:, 2]
        g = np.clip((hp + 0.08 + 0.11 * wob(x, 0) - z) / 0.14, 0.0, 1.0) * kd
        sp = np.clip((hp + 0.09 + 0.11 * wob(x, 1) - z) / 0.07, 0.0, 1.0) * mk * (0.55 + 0.45 * wob(x * 2.3, 2)) * (1.0 if side > 0 else 0.85)
        for (xj, wj, lj, sj) in cstreak[side]:
            m = np.clip(1 - np.abs(x - xj) / wj, 0.0, 1.0) * np.clip(1 - (zc - z) / lj, 0.0, 1.0) * (z < zc + 0.005) * sj
            g = np.maximum(g, 0.4 * m)
            sp = np.maximum(sp, 0.6 * m * mk)
        for xw in weeps:
            if side < 0:
                m = np.clip(1 - np.abs(x - xw) / 0.075, 0.0, 1.0) * (z < whz + 0.01)
                g = np.maximum(g, 0.55 * m)
                sp = np.maximum(sp, m * clamp(0.3 + mossk, 0.0, 1.0))
        return g, sp

    # ---- stone colour: one dominant stone, a few clusters of warmer / greyer ones
    ntint = 2 if lantern else 3
    rt_ = rng.fork("tint")
    clusters = {}
    for sd in sides:
        ncl = max(1, int(round(0.75 * Lf * max(Hb, 0.1))))
        clusters[sd] = [(rt_.u(fxa, fxb), rt_.u(hp, zc), rt_.u(0.22, 0.42) * s, rt_.u(0.12, 0.22) * s, 1 + (kk + (sd > 0)) % 2)
                        for kk in range(ncl)]

    def tint_of(side, x, z, r):
        stray = r.u(0, 1)
        for (cx, cz, rx, rz, ty) in clusters[side]:
            if ((x - cx) / rx) ** 2 + ((z - cz) / rz) ** 2 < 1.0 and stray < 0.85:
                return min(ty, ntint - 1)
        if stray > 0.965:
            return 1 if stray > 0.983 else min(2, ntint - 1)
        return 0

    # ---- bed lines shared by the courses
    wavy = rubble and hi
    KDX = 0.12
    nk = int(math.ceil(L / KDX)) + 2

    def mkline(fn):
        return _Line(x0=xl, dx=KDX, vals=[fn(xl + kk * KDX) for kk in range(nk)])

    qa_, qb_ = (0.0 if (pl or brick) else qlens[0] * s), (0.0 if (pr or brick) else qlens[0] * s)
    lines = []
    for i in range(nb + 1):
        if not wavy or i == 0 or i == nb:
            lines.append(_Line(z=zs[i]))
        else:
            a_, b_, k1, k2 = rb.u(0, 6.28), rb.u(0, 6.28), rb.u(4.0, 7.0), rb.u(9.0, 13.0)

            def fn(x, i=i, a_=a_, b_=b_, k1=k1, k2=k2):
                env = clamp(min(x - (fxa + qa_), (fxb - qb_) - x) / 0.3, 0.0, 1.0)
                return zs[i] + 0.011 * s * env * (0.6 * math.sin(k1 * x + a_) + 0.4 * math.sin(k2 * x + b_))
            lines.append(mkline(fn))

    # ---- facing stones, each with the pointing around it
    def stone(side, xs4, zb, zt, r):
        wd = (xs4[2] + xs4[3] - xs4[0] - xs4[1]) / 2
        xm = (xs4[0] + xs4[1] + xs4[2] + xs4[3]) / 4
        zmid = (zb(xm) + zt(xm)) / 2
        hz = zt(xm) - zb(xm)
        ym = side * (T / 2 - rec)
        outv = np.array([0.0, side, 0.0])

        def Wd(P):
            P = np.asarray(P, dtype=float)
            return np.stack([P[:, 0], ym + side * P[:, 2], P[:, 1]], axis=1)
        if wd < 0.035 or hz < 0.03:                 # a sliver: just pointing
            Q = Wd([(xs4[0], zb(xs4[0]), 0.0), (xs4[2], zb(xs4[2]), 0.0), (xs4[3], zt(xs4[3]), 0.0), (xs4[1], zt(xs4[1]), 0.0)])
            g, sp = wmask(Q, side)
            kit.add("joint", Q, [(0, 1, 2, 3)], out=outv, g=g * 1.6, sp=sp)
            return
        if rubble:
            chip = r.u(0.015, 0.035) if r.chance(0.08 + 0.2 * age) else 0.0
            V, F, Vm, Fm = _pillow(r, xs4, zb, zt, hi, r.u(0.008, 0.015), r.u(0.006, 0.012), r.u(0.0035, 0.008),
                                   r.u(0.002, 0.010), (0.03, 0.32), 0.035, chip)
        else:
            chip = r.u(0.02, 0.05) if r.chance(0.2 + 0.3 * age) else 0.0
            sub = (max(0.15, Lf / 40.0), 0.1 * max(1.0, s)) if (afine or zb(xm) < hp + 0.3) else None
            V, F, Vm, Fm = _pillow(r, xs4, zb, zt, False, 0.006, 0.008, r.u(0.002, 0.007), 0.0, (0.0, 0.0), 0.012, chip, sub)
        ti = tint_of(side, xm, zmid, r)
        dg = 0.2 * r.u(0, 1) ** 2
        dw = 0.22 * r.u(0, 1) ** 2 if r.chance(0.5) else 0.0
        Q = Wd(V)
        g, sp = wmask(Q, side)
        kit.add("face%d" % ti, Q, F, out=outv, g=g + dg, w=dw, sp=sp)
        Q = Wd(Vm)
        g, sp = wmask(Q, side)
        pm = mossk * (0.12 + 0.3 * (zmid > zc - 0.24 * s) + 0.12 * (side > 0))
        if r.chance(pm):                            # moss holding in the bed joint under the stone
            zbv = np.array([zb(u) for u in Vm[:, 0]])
            bed = np.clip(1 - (Vm[:, 1] - zbv) / 0.03, 0.0, 1.0)
            sp = np.maximum(sp, bed * r.u(0.5, 1.0) * np.clip(0.2 + 1.3 * wob(Vm[:, 0] * 3.0, 5), 0.0, 1.0))
        kit.add("joint", Q, Fm, out=outv, g=g * 1.6, sp=sp)

    def fill_row(side, x0, x1, zb, zt, r, wlo, whi_):
        bnd = [(x0, x0)]
        x = x0
        sl = 0.022 * s if rubble else 0.0
        while x < x1 - 1e-4:
            w = r.u(wlo, whi_) * s
            if r.chance(0.12) and rubble:
                w *= 1.5
            if (x1 - x) - w < wlo * s * 0.75:
                w = x1 - x
            x += w
            bnd.append((x1, x1) if x >= x1 - 1e-4 else (x + r.u(-sl, sl), x + r.u(-sl, sl)))
        for kk in range(len(bnd) - 1):
            stone(side, (bnd[kk][0], bnd[kk][1], bnd[kk + 1][0], bnd[kk + 1][1]), zb, zt, r)

    def fill_band(side, x0, x1, zb, zt, r):
        if x1 - x0 < 1e-4:
            return
        if ashlar:
            fill_row(side, x0, x1, zb, zt, r, 0.42, 0.78)
            return
        hb = zt((x0 + x1) / 2) - zb((x0 + x1) / 2)
        two = hb > 0.2 * s
        x = x0
        while x < x1 - 1e-4:
            rem = x1 - x
            if two and r.chance(0.22) and rem > 0.16 * s:
                w = r.u(0.15, 0.3) * s
                if rem - w < 0.12 * s:
                    w = rem
                stone(side, (x, x, x + w, x + w), zb, zt, r)
                x += w
            else:
                seg = r.u(0.45, 1.0) * s
                if rem - seg < 0.2 * s:
                    seg = rem
                if two:
                    f_, ph_, kk_ = r.u(0.36, 0.64), r.u(0, 6.28), r.u(7.0, 11.0)
                    if wavy:
                        zm = mkline(lambda xx, f_=f_, ph_=ph_, kk_=kk_: zb(xx) + (zt(xx) - zb(xx)) * f_ + 0.008 * s * math.sin(kk_ * xx + ph_))
                    else:
                        zm = _Line(z=zb(x) + (zt(x) - zb(x)) * f_)
                    fill_row(side, x, x + seg, zb, zm, r, 0.14, 0.38)
                    fill_row(side, x, x + seg, zm, zt, r, 0.14, 0.38)
                else:
                    fill_row(side, x, x + seg, zb, zt, r, 0.14, 0.38)
                x += seg

    def weep_pipe(side, xw, ym, pocket_top=None):
        """Clay drain pipe bedded in a pocket of mortar; pocket_top = bed line closing the pocket."""
        N = 12
        outv = np.array([0.0, side, 0.0])

        def Wd(P):
            P = np.asarray(P, dtype=float)
            return np.stack([P[:, 0], ym + side * P[:, 2], P[:, 1]], axis=1)
        if pocket_top is not None:
            u0, u1 = xw - ww / 2, xw + ww / 2
            circ = [(xw + (ro - 0.003) * math.cos(math.radians(225 + 30 * kk)), whz + (ro - 0.003) * math.sin(math.radians(225 + 30 * kk)))
                    for kk in range(N + 1)]
            tl, tr = (u0, pocket_top(u0)), (u1, pocket_top(u1))
            top = [tr] + [(x, pocket_top(x)) for x in reversed(pocket_top.knots(u0, u1))] + [tl]
            Vm, Fm = [], []
            _strip(Vm, Fm, [(u0, hp), (u1, hp)], circ[0:4])
            _strip(Vm, Fm, [(u1, hp), tr], circ[3:7])
            _strip(Vm, Fm, top, circ[6:10])
            _strip(Vm, Fm, [tl, (u0, hp)], circ[9:13])
            Q = Wd(Vm)
            g, sp = wmask(Q, side)
            kit.add("joint", Q, Fm, out=outv, g=g * 1.6, sp=sp)

        def ring(rad, d):
            return [(xw + rad * math.cos(2 * math.pi * kk / N), whz + rad * math.sin(2 * math.pi * kk / N), d) for kk in range(N)]
        low = np.array([max(0.0, -math.sin(2 * math.pi * kk / N)) for kk in range(N)])      # 1 at the invert
        wet = clamp(0.35 + mossk, 0.0, 1.0)
        V = ring(ro, -0.006) + ring(ro, pjp) + ring(ro - 0.004, pjp + 0.004) + ring(ri, pjp + 0.004)
        F = [(a * N + kk, a * N + (kk + 1) % N, (a + 1) * N + (kk + 1) % N, (a + 1) * N + kk) for a in range(3) for kk in range(N)]
        kit.add("pipe", Wd(V), F, g=np.concatenate([0.25 + 0.4 * low, 0.2 + 0.45 * low, 0.2 + 0.5 * low, 0.45 + 0.4 * low]),
                sp=np.concatenate([low * wet, low * wet, 0.8 * low * wet, 0.6 * low * wet]))
        V = ring(ri, pjp + 0.004) + ring(ri, pjp - tdep)
        F = [(kk, (kk + 1) % N, N + (kk + 1) % N, N + kk) for kk in range(N)]
        kit.add("pipe", Wd(V), F, inward=True, g=np.concatenate([0.45 + 0.4 * low, np.ones(N)]), sp=np.concatenate([0.6 * low * wet, np.zeros(N)]))
        kit.add("pipe", Wd(ring(ri, pjp - tdep)), [tuple(range(N))], out=outv, g=1.0)

    if not brick:
        for side in sides:
            r = rng.fork("face%d" % side)
            for i in range(nb):
                xa = fxa + (0.0 if pl else qlen(i, 0))
                xb = fxb - (0.0 if pr else qlen(i, 1))
                zb, zt = lines[i], lines[i + 1]
                if i == 0 and weeps:
                    x = xa
                    for xw in weeps:
                        fill_band(side, x, xw - ww / 2, zb, zt, r)
                        if wh < b0h - 1e-6:
                            ptop = _Line(z=hp + wh)
                            stone(side, (xw - ww / 2, xw - ww / 2, xw + ww / 2, xw + ww / 2), ptop, zt, r)
                        else:
                            ptop = zt
                        weep_pipe(side, xw, side * (T / 2 - rec), ptop)
                        x = xw + ww / 2
                    fill_band(side, x, xb, zb, zt, r)
                else:
                    fill_band(side, xa, xb, zb, zt, r)
    else:
        for xw in weeps:
            weep_pipe(-1, xw, -T / 2)
    if not free and not brick:                      # parged back of a retaining wall
        nx_ = max(2, int(round(L / max(0.3, L / 24.0))))
        zr_ = [hp + t for t in (0.0, 0.07, 0.14, 0.22, 0.32) if hp + t < zc - 0.1]
        nz_ = max(1, int(round((zc - zr_[-1]) / 0.4)))
        zr_ += [zr_[-1] + (zc - zr_[-1]) * (j + 1) / nz_ for j in range(nz_)]
        V = np.array([(xl + L * i / nx_, T / 2, z) for z in zr_ for i in range(nx_ + 1)])
        F = [(j * (nx_ + 1) + i, j * (nx_ + 1) + i + 1, (j + 1) * (nx_ + 1) + i + 1, (j + 1) * (nx_ + 1) + i)
             for j in range(len(zr_) - 1) for i in range(nx_)]
        g, sp = wmask(V, 1)
        kit.add("joint", V, F, out=np.array([0.0, 1.0, 0.0]), g=0.1 + 0.6 * g + 0.12 * wob(V[:, 0] * 1.7 + V[:, 2] * 2.1, 6), sp=0.6 * sp)

    # ---- quoins on the flat (butting) ends, with their bed joints pointed
    rq = rng.fork("quoin")
    if not brick:
        qp = 0.001 if ashlar else 0.006
        for e, has_pier, par in ((-1, pl, 0), (1, pr, 1)):
            if has_pier:
                continue
            inner = [4 * (1 if e < 0 else 0) + 2 * iy + iz for iy in (0, 1) for iz in (0, 1)]
            for i in range(nb):
                q = qlen(i, par)
                x0, x1 = (xl, xl + q - 0.004) if e < 0 else (xr - q + 0.004, xr)
                dd = rq.u(-0.0015, 0.0025)
                yb = T / 2 + qp + dd if free else T / 2 + 0.002
                V, F = _block(x0, x1, -T / 2 - qp - dd, yb, zs[i] + 0.004, zs[i + 1] - 0.004, 0.008,
                              skip={(0, -e), (2, -1), (2, 1)}, r=rq, jit=0.0012)
                if rq.chance(0.15 + 0.45 * age):
                    _knock(V, rq, rq.u(0.012, 0.03), inner)
                g, sp = wmask(V, 1)
                kit.add("dress", V, F, g=g + 0.15 * rq.u(0, 1) ** 2, sp=sp)
            for i in range(nb + 1):
                ql = min(qlen(max(i - 1, 0), par), qlen(min(i, nb - 1), par))
                x0, x1 = (xl + 0.004, xl + ql) if e < 0 else (xr - ql, xr - 0.004)
                ys = T / 2 + qp - 0.006
                V, F = _block(x0, x1, -ys, ys if free else T / 2 - 0.002, max(zs[i] - 0.007, 0.0), zs[i] + 0.007, 0.0,
                              skip={(0, -e), (2, -1), (2, 1)})
                g, sp = wmask(V, 1)
                kit.add("joint", V, F, g=g * 1.6, sp=sp)

    # ---- plinth course: long weathered stones through the wall
    rp = rng.fork("plinth")
    pxa = xl + (2 * pov + Px - 0.01 if pl else 0.0)
    pxb = xr - (2 * pov + Px - 0.01 if pr else 0.0)
    if hp > 0 and pxb - pxa > 0.05:
        yf, yb = -T / 2 - proj, T / 2 + (proj if free else 0.0)
        wth = min(0.03, hp * 0.3)
        if free:
            prof = [(yb, 0), (yb, hp - wth), (yb - wth, hp), (yf + wth, hp), (yf, hp - wth), (yf, 0)]
        else:
            prof = [(yb, 0), (yb, hp), (yf + wth, hp), (yf, hp - wth), (yf, 0)]
        x = pxa
        while x < pxb - 1e-4:
            w = rp.u(0.55, 1.05)
            if (pxb - x) - w < 0.4:
                w = pxb - x
            a, b = x, x + w
            ea = 0.0 if abs(a - xl) < 1e-6 else 0.007
            eb = 0.0 if abs(b - xr) < 1e-6 else 0.007
            a2, b2 = a + (0.004 if ea else 0.0), b - (0.004 if eb else 0.0)
            nmid = max(2, int(round((b2 - a2) / 0.2)) - 1)
            mids = sorted(set([round(xw + dx, 4) for xw in weeps for dx in (-0.075, 0.0, 0.075) if a2 + 0.02 < xw + dx < b2 - 0.02]
                              + [round(a2 + (b2 - a2) * (t + 1) / (nmid + 1), 4) for t in range(nmid)]))
            xs = [a2] + ([a2 + ea] if ea else []) + [m for m in mids if a2 + 0.012 < m < b2 - 0.012] + ([b2 - eb] if eb else []) + [b2]
            ins = [ea] + [0.0] * (len(xs) - 2) + [eb]
            V, F = _prism(prof, xs, ins, bottom=False)
            V[:, 2] += rp.u(0.0, 0.004) * (V[:, 2] > hp * 0.5)
            x_, z_ = V[:, 0], V[:, 2]
            nn = wob(x_ * 1.4, 2)
            g = kd * (0.7 + 0.3 * wob(x_, 0)) + 0.12 * rp.u(0, 1)
            sp = mk * np.where(z_ > hp - wth - 1e-4, 0.45 + 0.55 * nn, 0.15 + 0.6 * nn) * (z_ > 0.02) * np.where(V[:, 1] > 0, 1.0, 0.85)
            for xw in weeps:
                m = np.clip(1 - np.abs(x_ - xw) / 0.08, 0.0, 1.0) * (V[:, 1] < 0)
                g = np.maximum(g, 0.6 * m)
                sp = np.maximum(sp, m * clamp(0.35 + mossk, 0.0, 1.0))
            kit.add("dress", V, F, g=g, sp=sp)
            x += w

    # ---- coping stones: uneven lengths and thicknesses, each bedded on its own wedge of mortar
    hwid = W / 2
    if cop == "flat":
        cch = 0.016
        cprof = [(hwid, 0), (hwid, edge - cch), (hwid - cch, edge - 0.004), (hwid * 0.45, edge - 0.001), (0.0, edge),
                 (-hwid * 0.45, edge - 0.001), (-hwid + cch, edge - 0.004), (-hwid, edge - cch), (-hwid, 0)]
    elif cop == "saddle":
        cprof = [(hwid, 0), (hwid, edge - 0.008), (hwid - 0.008, edge), (hwid * 0.5, edge + rise * 0.52), (0.022, tch), (-0.022, tch),
                 (-hwid * 0.5, edge + rise * 0.52), (-hwid + 0.008, edge), (-hwid, edge - 0.008), (-hwid, 0)]
    else:
        cprof = [(hwid, 0), (hwid, edge)] + [(hwid * math.cos(math.radians(a)), edge + rise * math.sin(math.radians(a)))
                                              for a in (20, 40, 60, 80, 100, 120, 140, 160)] + [(-hwid, edge), (-hwid, 0)]
    npf = len(cprof)

    def ctop(y):                                    # coping top above its bed at y
        t = min(1.0, abs(y) / hwid)
        if cop == "flat":
            return edge
        if cop == "saddle":
            return edge + rise * (1 - t)
        return edge + rise * math.sqrt(max(0.0, 1 - t * t))

    cstep = 0.14 if L <= 6 else (0.24 if L <= 9 else 0.4)
    ya = T / 2 + 0.004
    cstones = []
    for ci in range(len(cb) - 1):
        a, b = cb[ci], cb[ci + 1]
        enda, endb = abs(a - xl) < 1e-6, abs(b - xr) < 1e-6
        a2, b2 = a + (0.0 if enda else 0.004), b - (0.0 if endb else 0.004)
        wl = b2 - a2
        mode = rc.weighted([(0, 0.6), (1, 0.2), (2, 0.2)])          # 1 = bedded proud, 2 = thin / settled
        tf = 1.0 + rc.u(-0.09, 0.09) * (0.3 + age) - (0.1 * age if mode == 2 else 0.0)
        pmax = math.degrees(math.atan(0.03 / max(wl, 0.2)))
        pitch = math.radians(clamp(rc.u(-2.2, 2.2) * age, -pmax, pmax))
        roll = math.radians(rc.u(-1.8, 1.8) * age)
        yaw = 0.0 if (enda or endb) else math.radians(rc.u(-0.5, 0.5) * age)
        lift = 0.003 + abs(math.tan(pitch)) * wl / 2 + abs(math.tan(roll)) * hwid + age * rc.u(0.0, 0.006) + (0.016 * age if mode == 1 else 0.0)
        c0 = np.array([(a + b) / 2, 0.0, zc])
        Rr = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Rp = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        M = Ry @ Rp @ Rr
        tv = np.array([0.0, rc.u(-0.008, 0.008) * age, lift])

        def xf(V, c0=c0, M=M, tv=tv):
            return (np.asarray(V, dtype=float) - c0) @ M.T + c0 + tv

        def top_at(x, y, xf=xf, tf=tf):
            return float(xf(np.array([[x, y, zc + ctop(y) * tf]]))[0, 2])
        # rings: regular ones plus three tight ones round every chipped arris
        nm = max(1, int(round(wl / cstep)) - 1)
        mids = [a2 + wl * (t + 1) / (nm + 1) for t in range(nm)]
        chips = []
        if wl > 0.3:
            for _ in range(int(rc.chance(0.3 + 0.5 * age)) + int(rc.chance(0.35 * age))):
                chips.append((rc.u(a2 + 0.07, b2 - 0.07), rc.pick([1, -1]), rc.u(0.012, 0.026), rc.u(0.02, 0.035)))
        if len(chips) == 2 and abs(chips[0][0] - chips[1][0]) < 0.12:
            chips.pop()
        for (xc_, sg_, d_, hl_) in chips:
            mids = [m for m in mids if abs(m - xc_) > hl_ + 0.02] + [xc_ - hl_, xc_, xc_ + hl_]
        ec = 0.008
        xs = [a2, a2 + ec] + sorted(mids) + [b2 - ec, b2]
        ins = [ec] + [0.0] * (len(xs) - 2) + [ec]
        V, F = _prism([(y, zc + z * tf) for (y, z) in cprof], xs, ins)
        nr = len(xs)
        x_, y_ = V[:, 0].copy(), V[:, 1].copy()
        jidx = np.tile(np.arange(npf), nr)
        bottom = (jidx == 0) | (jidx == npf - 1)
        sidev = (jidx == 1) | (jidx == npf - 2)
        upw = np.where(bottom, 0.0, np.where(sidev, 0.45, 1.0))
        jp = np.clip(1 - np.minimum(x_ - a, b - x_) / 0.10, 0.0, 1.0)
        n_ = np.clip((0.5 * wob(x_, 3) + 0.5 * wob(1.7 * x_ + 5.0 * y_, 4) - 0.33) / 0.34, 0.0, 1.0)
        film = np.clip(mossk * 1.5 * (0.2 + 0.45 * np.clip(y_ / hwid, 0.0, 1.0) + 0.55 * jp) * (0.2 + 0.8 * n_), 0.0, 1.0) * upw
        backf = np.clip(mossk * 1.2 * (0.35 + 0.65 * n_), 0.0, 1.0) * (y_ > 0) * np.where(sidev, 1.0, np.where(bottom, 0.5, 0.0))
        sp = np.maximum(film, backf)
        g = 0.3 * sp + 0.1 * wob(x_ * 1.3, 5) + np.where(bottom, 0.3, 0.0) + np.where(sidev, 0.12 + 0.25 * jp, 0.0)
        wr = 0.18 * (1 - sp) * upw * wob(x_, 6)
        for (xc_, sg_, d_, hl_) in chips:           # a spall off the top arris
            kc = min(range(nr), key=lambda q: abs(xs[q] - xc_))
            j1, j2 = (1, 2) if sg_ > 0 else (npf - 2, npf - 3)
            V[kc * npf + j1] += np.array([0.0, 0.0, -d_ * 0.8])
            V[kc * npf + j2] += np.array([0.0, -sg_ * d_, -d_ * 0.12])
            for j in (j1, j2):
                wr[kc * npf + j], sp[kc * npf + j] = 0.7, 0.0
        ends_ok = [q for q, isend in ((0, enda), (1, endb)) if not isend]
        if ends_ok and rc.chance(0.3 + 0.5 * age):  # knocked corner at a joint
            atb = rc.pick(ends_ok)
            sg_ = rc.pick([1, -1])
            j1, j2 = (1, 2) if sg_ > 0 else (npf - 2, npf - 3)
            dv = np.array([(-1 if atb else 1) * rc.u(0.01, 0.03), -sg_ * rc.u(0.006, 0.02), -rc.u(0.006, 0.02)])
            for rg, f_ in (((nr - 1) if atb else 0, 1.0), ((nr - 2) if atb else 1, 0.7)):
                for j in (j1, j2):
                    V[rg * npf + j] += dv * f_
                    wr[rg * npf + j] = 0.6
        kit.add("cope", xf(V), F, shear=False, g=g, w=wr, sp=sp)
        cstones.append({"a": a, "b": b, "top": top_at})
        # its bed: a wedge of mortar between the wall head and the stone
        tq = xf(np.array([[xx, yy, zc + 0.004] for xx in (a2 + 0.002, b2 - 0.002) for yy in (-ya, ya)]))
        Vb = [(xx, yy, zc - 0.004) for xx in (a2 + 0.002, b2 - 0.002) for yy in (-ya, ya)] + \
             [(q[0], q[1], max(q[2], zc + 0.002)) for q in tq]
        Fb = [(0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        xb_ = np.array([q[0] for q in Vb])
        kit.add("joint", Vb, Fb, shear=False, g=0.3, sp=mossk * (0.35 + 0.6 * wob(xb_, 4)))
    # pointing in the joints between the coping stones, kept a little low
    for ci, xj in enumerate(cjoints):
        SL, SR = cstones[ci], cstones[ci + 1]
        dz = min(S_["top"](xj, yy) - (zc + ctop(yy)) for S_ in (SL, SR) for yy in (-0.9 * hwid, 0.0, 0.9 * hwid))
        pf = [(y * (1 - 0.012 / hwid), zc - 0.004 if (j in (0, npf - 1)) else zc + z + dz - 0.006) for j, (y, z) in enumerate(cprof)]
        V, F = _prism(pf, [xj - 0.008, xj + 0.008], [0.0, 0.0], caps=(False, False))
        kit.add("joint", V, F, shear=False, g=0.35, sp=min(1.0, mossk * 1.2))

    def stone_at(x):
        for S_ in cstones:
            if x <= S_["b"]:
                return S_
        return cstones[-1]

    # iron cramps let into the top across the joints of flat copings that still sit level
    rcr = rng.fork("cramp")
    if cop == "flat":
        for ci, xj in enumerate(cjoints):
            yy = rcr.u(-0.25, 0.25) * T
            z0_, z1_ = cstones[ci]["top"](xj - 0.06, yy), cstones[ci + 1]["top"](xj + 0.06, yy)
            if rcr.chance(0.8) and abs(z0_ - z1_) < 0.007:
                zt_ = max(z0_, z1_)
                V, F = _block(xj - 0.085, xj + 0.085, yy - 0.013, yy + 0.013, min(z0_, z1_) - 0.008, zt_ + 0.003, 0.003 if L <= 7 else 0.0, skip={(2, -1)})
                kit.add("iron", V, F, shear=False)
                for sx in (-1, 1):
                    V, F = _block(xj + sx * 0.07 - 0.016, xj + sx * 0.07 + 0.016, yy - 0.018, yy + 0.018, min(z0_, z1_) - 0.008, zt_ + 0.0045, 0.003 if L <= 7 else 0.0, skip={(2, -1)})
                    kit.add("iron", V, F, shear=False)

    # ---- body: lime-mortar hearting behind the stones, or the brickwork itself
    extra = []                                      # toolkit-built objects: (obj, kind)
    cxl = (xl + pov + Px * 0.5) if pl else (xl if brick else xl + 0.004)
    cxr = (xr - pov - Px * 0.5) if pr else (xr if brick else xr - 0.004)
    mortarm = mat("mortar", _tint(p.stone, (1.05, 1.10, 1.20)), grime=0.55)

    def core_fix(co):
        y = co.y
        if bt > 0 and y < -1e-6 and co.z < zc:
            y -= bt * (zc - co.z)
        return Vector((min(max(co.x, xl), xr), y, co.z))

    if not brick:
        cz0, cz1 = (hp - 0.01 if hp > 0 else 0.0), zc - 0.002
        y0c, y1c = -(T / 2 - rec - 0.004), (T / 2 - rec - 0.004 if free else T / 2 - 0.004)
        core = box("core", (cxr - cxl, y1c - y0c, cz1 - cz0), loc=((cxl + cxr) / 2, (y0c + y1c) / 2, (cz0 + cz1) / 2), segments=1)
        for xw in weeps:
            cut = cylinder("cut", radius=0.044, depth=tdep + 0.05, loc=(xw, y0c + (tdep + 0.05) / 2 - 0.03, whz), rot=(90, 0, 0), verts=10, segments=1)
            boolean(core, cut)
        apply_transform(core)
        transform_verts(core, core_fix)
        assign(core, mortarm)
        uv_box(core)
        extra.append((core, "mortar"))
    else:
        z0b, z1b = (hp - 0.01 if hp > 0 else 0.0), zc - 0.062
        nx_ = max(2, int(round((cxr - cxl) / max(0.16, (cxr - cxl) / 45.0))))
        zr_ = [z0b] + [hp + t for t in (0.06, 0.12, 0.18, 0.25, 0.34) if hp + t < z1b - 0.1]
        nz_ = max(1, int(round((z1b - zr_[-1]) / 0.3)))
        zr_ += [zr_[-1] + (z1b - zr_[-1]) * (j + 1) / nz_ for j in range(nz_)]
        nz_ = len(zr_) - 1
        rbk = rng.fork("brick")
        ph = [rbk.u(0, 6.28) for _ in range(4)]
        for side in (-1, 1):
            V = []
            for j in range(nz_ + 1):
                for i in range(nx_ + 1):
                    x, z = cxl + (cxr - cxl) * i / nx_, zr_[j]
                    env = clamp(min(x - xl, xr - x) / 0.25, 0.0, 1.0)
                    d = 0.0035 * (0.4 + age) * env * (math.sin(1.9 * x + ph[0] + side) * math.sin(2.3 * z + ph[1]) + 0.5 * math.sin(4.1 * x + 3.3 * z + ph[2]))
                    V.append((x, side * (T / 2 + d), z))
            F = [(j * (nx_ + 1) + i, j * (nx_ + 1) + i + 1, (j + 1) * (nx_ + 1) + i + 1, (j + 1) * (nx_ + 1) + i) for j in range(nz_) for i in range(nx_)]
            V = np.array(V)
            g, sp = wmask(V, side)
            kit.add("body", V, F, out=np.array([0.0, side, 0.0]), g=g, sp=sp, uvoff=False)
        for e, has_pier in ((-1, pl), (1, pr)):
            if not has_pier:
                xe = xl if e < 0 else xr
                kit.add("body", [(xe, -T / 2, z0b), (xe, T / 2, z0b), (xe, T / 2, z1b), (xe, -T / 2, z1b)], [(0, 1, 2, 3)],
                        out=np.array([float(e), 0.0, 0.0]), g=[kd, kd, 0.0, 0.0], uvoff=False)
        # the odd brick laid proud of the face
        bw, bz = BRICK
        npil = max(0, int(round(Lf / 2.4)) - 1) if Lf > 2.6 else 0
        pilx = [fxa + Lf * (kk + 1) / (npil + 1) for kk in range(npil)]
        for side in (-1, 1):
            for _ in range(max(2, int(round(Lf * (z1b - hp) * 0.9)))):
                Mc = int(rbk.u(hp / bz + 3, z1b / bz - 1))
                kx = int(math.floor(rbk.u(fxa + 0.25, fxb - 0.5) / bw))
                jc = bw * kx + (0.1035 if Mc % 2 == 0 else -0.009)
                d = rbk.u(0.009, 0.016)
                if Mc * bz > z1b - 0.05 or Mc * bz - 0.06 < hp + 0.05 or any(abs(jc + 0.11 - px_) < 0.32 for px_ in pilx):
                    continue
                if any(Mc * bz - 0.085 < zz < Mc * bz + 0.025 for zz in zr_):     # keep clear of the face grid rows
                    continue
                y0, y1 = (-(T / 2 + d), -(T / 2 - 0.02)) if side < 0 else (T / 2 - 0.02, T / 2 + d)
                V, F = _block(jc + 0.007, jc + 0.218, y0, y1, Mc * bz - 0.06, Mc * bz, 0.003, skip={(1, -side)})
                kit.add("body", V, F, uvoff=False, w=0.25)

    # ---- brick dressings: double tile creasing under the coping, pilaster strips with weathered stone heads
    if brick:
        rt = rng.fork("tile")
        V, F = _block(cxa + 0.006, cxb - 0.006, -T / 2 - 0.02, T / 2 + 0.02, zc - 0.062, zc - 0.003, 0.0, skip={(2, -1), (2, 1)})
        kit.add("joint", V, F, g=0.35, sp=0.5 * mossk)
        tl = 0.265 * max(1.0, L / 7.0)
        for c_i, (z0, pj) in enumerate(((zc - 0.060, 0.058), (zc - 0.031, 0.036))):
            x = cxa + 0.004 - (tl / 2 if c_i else 0.0)
            while x < cxb - 1e-4:
                a, b = max(x, cxa + 0.004), min(x + tl, cxb - 0.004)
                if b - a > 0.02:
                    dz = rt.u(-0.0015, 0.0015)
                    V, F = _block(a + 0.003, b - 0.003, -T / 2 - pj - rt.u(0, 0.007), T / 2 + pj + rt.u(0, 0.007), z0 + dz, z0 + 0.026 + dz, 0.004)
                    kit.add("tile", V, F, g=0.2 + 0.35 * rt.u(0, 1), sp=mossk * rt.u(0.0, 0.7) * (0.5 + 0.5 * c_i))
                x += tl
        pw_, pp = 0.3375, 0.08
        ztop = zc - 0.062 - 0.004
        for xc in pilx:
            for side in sides:
                zh = ztop - 0.12
                y0, y1 = (-(T / 2 + pp), -(T / 2 - 0.02)) if side < 0 else (T / 2 - 0.02, T / 2 + pp)
                zq = [(hp - 0.01 if hp > 0 else 0.0)] + [hp + t for t in (0.09, 0.2, 0.34) if hp + t < zh - 0.1] + [zh]
                for kq_ in range(len(zq) - 1):
                    V, F = _block(xc - pw_ / 2, xc + pw_ / 2, y0, y1, zq[kq_], zq[kq_ + 1], 0.0, skip={(1, -side), (2, -1), (2, 1)})
                    g, sp = wmask(V, side)
                    kit.add("body", V, F, g=g, sp=sp, uvoff=False)
                yf_ = side * (T / 2 + pp + 0.006)
                yi_ = side * (T / 2 - 0.01)
                wprof = [(yi_, zh), (yf_, zh), (yf_, zh + 0.035), (side * (T / 2 + 0.004), ztop), (yi_, ztop)]
                V, F = _prism(wprof, [xc - pw_ / 2 - 0.008, xc - pw_ / 2 - 0.002, xc + pw_ / 2 + 0.002, xc + pw_ / 2 + 0.008], [0.006, 0.0, 0.0, 0.006])
                kit.add("dress", V, F, g=0.2, sp=0.5 * mossk * (V[:, 2] > zh + 0.03))

    # ---- piers: coursed dressed blocks (or brick shaft), plinth block, moulded cap, finial
    rpi = rng.fork("pier")
    pier_tops = []
    ring_done = False
    head = None
    for e, has_pier in ((-1, pl), (1, pr)):
        if not has_pier:
            continue
        xe = xl if e < 0 else xr
        xo, xi = xe - e * pov, xe - e * (pov + Px)          # outer / inner shaft face
        x0, x1 = min(xo, xi), max(xo, xi)
        pxc, pyc = (x0 + x1) / 2, (yfp + ybp) / 2
        zb0 = hp if hp > 0 else 0.0
        if hp > 0:
            V, F = _block(x0 - pov, x1 + pov, yfp - pov, ybp + pov, 0.0, hp + 0.03, 0.02, skip={(2, -1)}, r=rpi, jit=0.002)
            if rpi.chance(0.3 + 0.4 * age):
                _knock(V, rpi, rpi.u(0.02, 0.04), (1, 3, 5, 7))
            kit.add("dress", V, F, shear=False, g=kd * 0.9, sp=mk * np.where(V[:, 2] > hp * 0.5, 0.75, 0.2))
        if brick:
            zq = [zb0] + [hp + t for t in (0.09, 0.2, 0.34) if hp + t < zpt - 0.1] + [zpt]
            for kq_ in range(len(zq) - 1):
                V, F = _block(x0, x1, yfp, ybp, zq[kq_], zq[kq_ + 1], 0.0, skip={(2, -1), (2, 1)})
                g, sp = wmask(V, 1)
                kit.add("body", V, F, shear=False, g=g, sp=sp, uvoff=False)
        else:
            ne = max(1, int(round((zpt - zc) / bh)))
            pzs = list(zs) + [zc + (zpt - zc) * (t + 1) / ne for t in range(ne)]
            for i in range(len(pzs) - 1):
                za, zb = pzs[i] + 0.004, pzs[i + 1] - 0.004
                if zb - za < 0.02:
                    continue
                if i % 2 == 1 and Px >= 0.4:
                    xm = x0 + Px * rpi.u(0.38, 0.62)
                    parts = [(x0, xm - 0.004, {(0, 1)}, (0, 1, 2, 3)), (xm + 0.004, x1, {(0, -1)}, (4, 5, 6, 7))]
                    V, F = _block(xm - 0.024, xm + 0.024, yfp + 0.006, ybp - 0.006, za - 0.004, zb + 0.004, 0.0, skip={(2, -1), (2, 1), (0, -1), (0, 1)})
                    kit.add("joint", V, F, shear=False, g=0.15)
                else:
                    parts = [(x0, x1, set(), (0, 1, 2, 3, 4, 5, 6, 7))]
                for (a, b, sk, crn) in parts:
                    dd = rpi.u(-0.003, 0.005)
                    V, F = _block(a, b, yfp - dd, ybp + dd, za, zb, 0.011, skip=sk | {(2, -1), (2, 1)}, r=rpi, jit=0.002)
                    if rpi.chance(0.18 + 0.4 * age):
                        _knock(V, rpi, rpi.u(0.006, 0.018), crn)
                    g, sp = wmask(V, 1)
                    kit.add("dress", V, F, shear=False, g=g + 0.15 * rpi.u(0, 1) ** 2, sp=sp)
            for i in range(1, len(pzs)):
                V, F = _block(x0 + 0.006, x1 - 0.006, yfp + 0.006, ybp - 0.006, pzs[i] - 0.024, pzs[i] + 0.024, 0.0, skip={(2, -1), (2, 1)})
                g, sp = wmask(V, 1)
                kit.add("joint", V, F, shear=False, g=g * 1.6 + 0.1, sp=sp)
        # cap: necking, oversailing slab, weathered top
        V, F = _block(x0 - 0.014, x1 + 0.014, yfp - 0.014, ybp + 0.014, zpt, zpt + 0.045, 0.012, skip={(2, 1)}, r=rpi, jit=0.001)
        kit.add("cope", V, F, shear=False, g=0.3)
        sl = clamp(0.16 * Px, 0.07, 0.11)
        zs0 = zpt + 0.045
        V, F = _block(x0 - pov, x1 + pov, yfp - pov, ybp + pov, zs0, zs0 + sl, 0.012, r=rpi, jit=0.001)
        if rpi.chance(0.3 + 0.5 * age):
            _knock(V, rpi, rpi.u(0.015, 0.035))
        up_ = (V[:, 2] > zs0 + sl * 0.5)
        kit.add("cope", V, F, shear=False, g=0.15 + 0.2 * up_, sp=mossk * up_ * np.where(V[:, 1] > pyc, 0.75, 0.3))
        bx, by = (x1 - x0) / 2 + pov - 0.016, (ybp - yfp) / 2 + pov - 0.016
        if p.pier_cap == "pyramid":
            tx = ty = 0.035
            rz = 0.42 * min(bx, by)
        else:
            tx, ty = bx * 0.5, by * 0.5
            rz = 0.045
        z0_, z1_ = zs0 + sl - 0.003, zs0 + sl + rz
        V = [(pxc - bx, pyc - by, z0_), (pxc + bx, pyc - by, z0_), (pxc + bx, pyc + by, z0_), (pxc - bx, pyc + by, z0_),
             (pxc - tx, pyc - ty, z1_), (pxc + tx, pyc - ty, z1_), (pxc + tx, pyc + ty, z1_), (pxc - tx, pyc + ty, z1_)]
        F = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
        if p.pier_cap == "lantern":                 # lead-dressed weathering under the lamp standard, turned down over the arris
            bx, by = bx + 0.022, by + 0.022
            zd = z0_ - 0.014
            V = [(pxc - bx, pyc - by, zd), (pxc + bx, pyc - by, zd), (pxc + bx, pyc + by, zd), (pxc - bx, pyc + by, zd),
                 (pxc - bx, pyc - by, z0_ + 0.006), (pxc + bx, pyc - by, z0_ + 0.006), (pxc + bx, pyc + by, z0_ + 0.006), (pxc - bx, pyc + by, z0_ + 0.006),
                 (pxc - tx, pyc - ty, z1_), (pxc + tx, pyc - ty, z1_), (pxc + tx, pyc + ty, z1_), (pxc - tx, pyc + ty, z1_)]
            F = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 9, 8), (5, 6, 10, 9), (6, 7, 11, 10), (7, 4, 8, 11), (8, 9, 10, 11)]
            kit.add("lead", V, F, shear=False)
        else:
            kit.add("cope", V, F, shear=False, g=0.15, sp=[mossk * 0.6, mossk * 0.6, mossk * 0.8, mossk * 0.8, 0.0, 0.0, mossk * 0.2, mossk * 0.2])
        zcap = z1_
        pier_tops.append((pxc, pyc, zcap, e))
        if p.pier_cap == "ball":
            rbl = 0.42 * min(Px, Py) * 0.5 + 0.02
            prof = [(rbl * 0.62, -0.002), (rbl * 0.62, 0.012), (rbl * 0.5, 0.02), (rbl * 0.36, 0.045), (rbl * 0.4, 0.06)]
            zc_b = 0.06 + rbl * 0.9
            for t in range(11):
                a = math.radians(-64 + (90 + 64) * t / 10)
                prof.append((rbl * math.cos(a), zc_b + rbl * math.sin(a)))
            prof[-1] = (0.0, zc_b + rbl)
            ball = lathe("pier_ball", prof, segments=20, loc=(pxc, pyc, zcap))
            extra.append((ball, "cope"))
        elif p.pier_cap == "lantern":
            # cast crook standard: bolted flange and moulded socket on the cap, the crook carries the
            # lantern out over the street face, hung by its finial so nothing stands between the flame
            # and the stone: the lamp washes the pier shaft and the wall below it
            if head is None:
                head = use("lantern_head", width=0.24, height=0.3, crown=False, intensity=float(p.lantern_cd), paint="#1b1f1c", seed=seed + 3)
            hs = 0.8
            yo = (ybp - yfp) / 2 + pov + 0.15
            zb_ = 0.07 - hs * head.min.z                # lantern origin above the cap
            ztop_ = zb_ + hs * head.max.z               # top of its finial
            zA = ztop_ + 0.03
            V, F = _block(pxc - 0.078, pxc + 0.078, pyc - 0.078, pyc + 0.078, zcap - 0.003, zcap + 0.012, 0.004, skip={(2, -1)})
            kit.add("iron", V, F, shear=False)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    V, F = _block(pxc + sx * 0.057 - 0.011, pxc + sx * 0.057 + 0.011, pyc + sy * 0.057 - 0.011, pyc + sy * 0.057 + 0.011,
                                  zcap + 0.011, zcap + 0.022, 0.0, skip={(2, -1)})
                    kit.add("iron", V, F, shear=False)
            foot = lathe("lantern_socket", [(0.0, 0.01), (0.056, 0.01), (0.056, 0.024), (0.044, 0.034), (0.032, 0.06), (0.027, 0.11),
                                            (0.037, 0.12), (0.037, 0.136), (0.024, 0.146), (0.0, 0.146)], segments=10, loc=(pxc, pyc, zcap))
            extra.append((foot, "iron"))
            col = lathe("lantern_collar", [(0.0, -0.03), (0.02, -0.03), (0.029, -0.018), (0.029, -0.006), (0.021, 0.004), (0.027, 0.016), (0.019, 0.03), (0.0, 0.03)],
                        segments=8, loc=(pxc, pyc, zcap + zA - 0.02))
            extra.append((col, "iron"))
            Q0, Q1, Q2, Q3 = np.array([0.0, zA]), np.array([0.0, zA + 0.27]), np.array([-yo, zA + 0.31]), np.array([-yo, ztop_ + 0.05])
            path, rad = [(pxc, pyc, zcap + 0.13)], [0.021]
            for kk in range(15):
                t = kk / 14.0
                q = (1 - t) ** 3 * Q0 + 3 * t * (1 - t) ** 2 * Q1 + 3 * t * t * (1 - t) * Q2 + t ** 3 * Q3
                path.append((pxc, pyc + q[0], zcap + q[1]))
                rad.append(0.019 - 0.008 * t)
            V, F = _tube(path, rad, 8)
            kit.add("iron", V, F, shear=False, fix=False)
            knob = lathe("lantern_knob", [(0.0, -0.02), (0.012, -0.015), (0.018, 0.0), (0.012, 0.015), (0.0, 0.02)],
                         segments=8, loc=(pxc, pyc - yo, zcap + ztop_ + 0.05))
            extra.append((knob, "iron"))
            link = torus("lantern_link", major=0.021, minor=0.0045, loc=(pxc, pyc - yo, zcap + ztop_ + 0.016), rot=(90, 0, 0), major_seg=10, minor_seg=4)
            extra.append((link, "iron"))
            scr = torus("lantern_scroll", major=0.085, minor=0.0085, loc=(pxc, pyc - 0.097, zcap + zA + 0.08), rot=(0, 90, 0), major_seg=12, minor_seg=4)
            extra.append((scr, "iron"))
            drop = lathe("lantern_drop", [(0.0, -0.05), (0.007, -0.045), (0.013, -0.034), (0.007, -0.022), (0.014, -0.01), (0.024, 0.0), (0.0, 0.0)],
                         segments=8, loc=(pxc, pyc - yo, zcap + zb_ + hs * head.min.z + 0.004))
            extra.append((drop, "iron"))
            zl_ = zcap + 0.2 + 0.45 * (zA - 0.2)        # ladder bar on the standard
            V, F = _block(pxc - 0.125, pxc + 0.125, pyc - 0.006, pyc + 0.006, zl_ - 0.006, zl_ + 0.006, 0.0)
            kit.add("iron", V, F, shear=False)
            for sx in (-1, 1):
                V, F = _block(pxc + sx * 0.13 - 0.011, pxc + sx * 0.13 + 0.011, pyc - 0.011, pyc + 0.011, zl_ - 0.011, zl_ + 0.011, 0.0)
                kit.add("iron", V, F, shear=False)
            head.place((pxc, pyc - yo, zcap + zb_), scale=hs, sockets={"light"})
        # tethering ring on the street face of the first pier
        if p.tie_ring and not ring_done and zpt > 0.7:
            ring_done = True
            zr = min(0.86, zb0 + 0.62 * (zpt - zb0))
            V, F = _block(pxc - 0.032, pxc + 0.032, yfp - 0.007, yfp + 0.004, zr - 0.032, zr + 0.032, 0.003, skip={(1, 1)})
            kit.add("iron", V, F, shear=False)
            V, F = _block(pxc - 0.011, pxc + 0.011, yfp - 0.03, yfp, zr - 0.012, zr + 0.012, 0.004, skip={(1, 1)})
            kit.add("iron", V, F, shear=False)
            tr = torus("tie_ring", major=0.052, minor=0.0065, loc=(pxc, yfp - 0.024, zr - 0.05), rot=(84, 0, 0), major_seg=18, minor_seg=6)
            extra.append((tr, "iron"))

    # ---- moss cushions: coping joints and back edge, the bed under the coping, plinth ledge, foot of the wall
    rm = rng.fork("moss")

    def cushion(x, y, z, rx, ry, rz, sh=False, ground=False):
        if kit.tris > 11300 - 3000 * nlan:
            return
        x = clamp(x, xl + rx * 1.15 + 0.004, xr - rx * 1.15 - 0.004)
        V, F = _dome(rm, x, y, z, rx, ry, rz)
        if ground:
            V[:, 2] = np.maximum(V[:, 2], 0.0)
        kit.add("moss", V, F, shear=sh)

    for ci, xj in enumerate(cjoints):
        for _ in range(int(mossk * 3.0 + rm.u(0, 1))):
            y = clamp(hwid * (1 - 1.9 * rm.u(0, 1) ** 1.5), -0.9 * hwid, 0.9 * hwid)
            zt_ = min(cstones[ci]["top"](xj, y), cstones[ci + 1]["top"](xj, y)) - 0.007
            cushion(xj + rm.u(-0.004, 0.004), y, zt_, rm.u(0.012, 0.022), rm.u(0.02, 0.05), rm.u(0.009, 0.016))
    for _ in range(int((cxb - cxa) * 1.6 * mossk + rm.u(0, 1))):        # patches hanging on the back arris
        x = (rm.pick(cjoints) + rm.u(-0.2, 0.2)) if (cjoints and rm.chance(0.6)) else rm.u(cxa + 0.1, cxb - 0.1)
        x = clamp(x, cxa + 0.1, cxb - 0.1)
        sg_ = 1 if (rm.chance(0.8) or not free) else -1
        for q in range(rm.pick([2, 3, 3])):
            xx = x + q * rm.u(0.04, 0.06)
            y = sg_ * hwid * rm.u(0.84, 0.96)
            cushion(xx, y, stone_at(xx)["top"](xx, y) - 0.004, rm.u(0.045, 0.075), rm.u(0.02, 0.032), rm.u(0.008, 0.014))
    for sd in sides:
        for _ in range(int(Lf * (0.8 if sd < 0 else 1.6) * mossk + rm.u(0, 1))):
            x = rm.u(fxa + 0.08, fxb - 0.08)
            cushion(x, sd * (T / 2 + 0.01), zc - 0.011, rm.u(0.02, 0.045), rm.u(0.01, 0.016), rm.u(0.008, 0.013), sh=True)
        if hp > 0:
            for _ in range(int(Lf * 1.4 * mossk + rm.u(0, 1))):
                x = rm.u(pxa + 0.06, pxb - 0.06)
                rr = rm.u(0.02, 0.045)
                yy = sd * (T / 2 + proj * 0.45) + (-bt * (zc - hp) if sd < 0 else 0.0)
                cushion(x, yy, hp - 0.004, rr * rm.u(1.2, 2.2), rr * 0.7, rr * 0.5)
        for _ in range(int(Lf * 1.2 * mossk + rm.u(0, 1))):
            x = rm.u(pxa + 0.06, pxb - 0.06)
            rr = rm.u(0.03, 0.07)
            yy = sd * (T / 2 + (proj if hp > 0 else 0.0)) + (-bt * zc if sd < 0 else 0.0)
            cushion(x, yy + sd * rr * 0.3, 0.0, rr * rm.u(1.2, 2.4), rr * 0.8, rr * 0.5, ground=True)
    for xw in weeps:                                # where the weeps drip
        yy = -(T / 2 + (proj * 0.5 if hp > 0 else 0.0)) - bt * (zc - hp)
        cushion(xw + rm.u(-0.02, 0.02), yy, hp - 0.004, rm.u(0.04, 0.06), 0.022, 0.016)
        cushion(xw + rm.u(-0.03, 0.03), -(T / 2 + (proj if hp > 0 else 0.0)) - bt * zc - 0.02, 0.0, rm.u(0.06, 0.1), 0.05, 0.028, ground=True)

    # ---- realise buckets, materials
    if rubble:
        fk = ((1.0, 1.0, 1.0), (1.01, 0.965, 0.9), (0.86, 0.87, 0.895))
        fmats = [mat("stone_rough", _tint(p.stone, kk), grime=0.75) for kk in fk]
    else:
        fk = ((1.0, 1.0, 1.0), (1.0, 0.97, 0.92), (0.9, 0.905, 0.925))
        fmats = [mat("stone_smooth", _tint(p.stone, kk)) for kk in fk]
    dressm = fmats[0] if ashlar else mat("stone_smooth", _tint(p.stone, (0.98, 0.96, 0.92)))
    objs = {}
    for i in range(3):
        objs["face%d" % i] = kit.finish("face%d" % i, fmats[i], 38 if rubble else 20)
    objs["dress"] = kit.finish("dress", dressm, 30)
    objs["cope"] = kit.finish("cope", dressm, 30)
    objs["joint"] = kit.finish("joint", mortarm, 30)
    objs["moss"] = kit.finish("moss", mat("moss"), 80)
    ironm = mat("iron_wrought")
    objs["iron"] = kit.finish("iron", ironm, 40)
    objs["lead"] = kit.finish("lead", mat("lead"), 30)
    tilem = mat("terracotta", "#cf8a62")
    objs["tile"] = kit.finish("tile", tilem, 30)
    objs["pipe"] = kit.finish("pipe", tilem, 50)
    objs["body"] = kit.finish("body", mat("brick_red"), 30)
    for o, kind in extra:
        if kind == "cope":
            assign(o, dressm)
            paint(o, special=mask_mul(facing((0, 0, 1), 0.6), noise_mask(0.2, seed=seed + 5, lo=0.3, hi=0.7), mossk), grime=0.2)
        elif kind == "iron":
            assign(o, ironm)
            paint(o, special=mask_mul(noise_mask(0.04, seed=seed + 10, lo=0.25, hi=0.7), 0.7))
    if objs.get("lead") is not None:
        paint(objs["lead"], grime=mask_mul(noise_mask(0.12, seed=seed + 12, lo=0.2, hi=0.8), 0.7))
    if objs.get("iron") is not None:
        paint(objs["iron"], special=mask_mul(noise_mask(0.05, seed=seed + 10, lo=0.2, hi=0.7), 0.8))

    # ---- sockets
    for pt in pier_tops:
        if p.pier_cap == "pyramid":
            socket("pier_top_" + ("l" if pt[3] < 0 else "r"), "attach", (pt[0], pt[1], pt[2]), role="prop_top")
    if cop == "flat" and Lf > 0.8:
        socket("coping_top", "attach", ((fxa + fxb) / 2, 0.0, H + 0.01), role="prop_top")
    socket("end_l", "attach", (xl, 0.0, 0.0), rot=(0, 0, -90), role="wall_end")
    socket("end_r", "attach", (xr, 0.0, 0.0), rot=(0, 0, 90), role="wall_end")
