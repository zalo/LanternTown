"""Canal tunnel mouth: the portal where the canal disappears under the buildings. A face wall of
individually cut rock-faced blocks (pitched bosses inside drafted margins, cut to the extrados of
the arch) carries a ring of dressed voussoirs with a projecting dated keystone on moulded imposts
and long-and-short jamb quoins; a moulded string course and weathered coping finish the wall and
cast-iron cross pattress plates tie it back. Behind it a brick barrel vault on stone ribs recedes
over black water into an unlit, soot-black last bay; a kerbed, flagged towpath ledge on a stone
pier runs along one side with a boatman's hauling chain on the wall, and one chain lantern hangs
from the crown of the first bay. Algae grows in a tide band above the water line, moss cushions
sit in the joints and on the ledges. Portal face at y = 0, vault to +Y, quay level z = 0.
"""
from math import asin, atan, atan2, ceil, cos, degrees, floor, hypot, pi, radians, sin, sqrt

import numpy as np
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt

from tk import *

META = {"name": "Canal tunnel mouth", "class": "structure", "mount": "wall_ground",
        "allow_below_ground": True, "focus": 0.6}
PARAMS = {
    "width": {"default": 3.6, "min": 2.4, "max": 6.0, "desc": "clear opening width at the jambs (m)"},
    "height": {"default": 3.0, "min": 2.2, "max": 5.0, "desc": "clear opening height, quay level to crown (m)"},
    "arch": {"default": "round", "choices": ["round", "horseshoe", "segmental"], "desc": "arch form"},
    "depth": {"default": 3.4, "min": 2.0, "max": 4.0, "desc": "length of the vault behind the face (m)"},
    "lantern": {"default": True, "desc": "chain lantern hanging in the first bay of the vault"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "moss / algae on the damp stone"},
    "wing": {"default": 1.1, "min": 0.7, "max": 2.5, "desc": "face wall width either side of the opening (m)"},
    "top": {"default": 0.65, "min": 0.3, "max": 2.0, "desc": "masonry between the arch ring and the string course (m)"},
    "channel": {"default": 1.5, "min": 1.0, "max": 3.0, "desc": "channel bed depth below the quay (m)"},
    "water_level": {"default": 0.75, "min": 0.3, "max": 2.5, "desc": "water surface below the quay (m)"},
    "water": {"default": True, "desc": "water surface inside the vault"},
    "towpath": {"default": 0.85, "min": 0.5, "max": 1.4, "desc": "towpath ledge width inside the vault (m)"},
    "towpath_side": {"default": "right", "choices": ["right", "left"], "desc": "side of the towpath ledge"},
    "rustication": {"default": "rock", "choices": ["rock", "vee"], "desc": "face blocks: rock-faced bossage / smooth with V joints"},
    "stone": {"default": "#b5a78f", "desc": "stone colour"},
    "vault": {"default": "brick", "choices": ["brick", "ochre_brick", "stone"], "desc": "lining of the vault and tunnel walls"},
    "date": {"default": "1784", "desc": "cast numerals on the keystone ('' = none)"},
    "lantern_style": {"default": "cylinder", "choices": ["cylinder", "hex", "globe"], "desc": "lantern body"},
    "intensity": {"default": 34.0, "min": 10, "max": 90, "desc": "lantern intensity (cd)"},
    "mount": {"default": "ground", "choices": ["ground", "wall_ground"],
              "desc": "'ground' = free-standing: the masonry backing of the vault (abutments, haunching, closed end) is built; "
                      "'wall_ground' = let into a host quay / building wall that covers the back, so the backing is left out"},
}
VARIANTS = {
    "horseshoe_sandstone": {"width": 2.9, "height": 3.7, "arch": "horseshoe", "rustication": "vee", "stone": "#c8a577",
                            "towpath_side": "left", "towpath": 0.7, "moss": 0.15, "depth": 2.6, "top": 0.45,
                            "lantern_style": "hex", "date": "1691", "seed": 4},
    "segmental_wide": {"width": 5.4, "height": 2.7, "arch": "segmental", "depth": 4.0, "wing": 1.4, "stone": "#8f8c88",
                       "vault": "stone", "moss": 0.85, "towpath": 1.2, "top": 0.9, "water_level": 0.55,
                       "lantern_style": "globe", "intensity": 48, "date": "1812", "seed": 9},
    "tall_grand": {"width": 4.4, "height": 4.6, "arch": "round", "depth": 3.8, "wing": 1.6, "top": 1.3, "stone": "#bfb6a6",
                   "rustication": "vee", "moss": 0.3, "channel": 2.2, "water_level": 1.1, "intensity": 50, "date": "MDCCL",
                   "seed": 15},
    "low_culvert": {"width": 2.5, "height": 2.3, "arch": "segmental", "depth": 2.2, "wing": 0.8, "top": 0.35,
                    "stone": "#9c8f7c", "moss": 1.0, "towpath": 0.5, "channel": 1.1, "water_level": 0.45, "date": "",
                    "lantern_style": "hex", "intensity": 26, "seed": 23},
}


def _shade(h, f=1.0, warm=0.0):
    h = str(h).lstrip("#")
    try:
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        r, g, b = 0.71, 0.65, 0.56
    r, g, b = min(1.0, r * f * (1 + warm)), min(1.0, g * f), min(1.0, b * f * (1 - warm))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


class _Acc:
    """Accumulates many cut stones into one mesh (+ per-vertex stone tone / arris flags for the masks)."""

    def __init__(self):
        self.v, self.f, self.tone, self.edge = [], [], [], []

    def add(self, co, tone=0.5, edge=0.0):
        self.v.append(co)
        self.tone.append(tone)
        self.edge.append(edge)
        return len(self.v) - 1

    def obj(self, name):
        return mesh_object(name, self.v, self.f) if self.f else None


def _dedupe(pts, tol=0.012):
    out = []
    for q in pts or ():
        if not out or hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > tol:
            out.append((float(q[0]), float(q[1])))
    while len(out) > 3 and hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def _pdist(P, A):
    """Signed distance (positive inside) from points P (m, 2) to the closed polygon A (n, 2)."""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    A = np.asarray(A, dtype=float)
    B = np.roll(A, -1, axis=0)
    d = B - A
    L2 = np.maximum((d * d).sum(1), 1e-18)
    PA = P[:, None, :] - A[None, :, :]
    t = np.clip((PA * d[None]).sum(2) / L2[None], 0.0, 1.0)
    C = PA - t[..., None] * d[None]
    dist = np.sqrt((C * C).sum(2)).min(1)
    x, y = P[:, 0:1], P[:, 1:2]
    ax, ay, bx, by = A[None, :, 0], A[None, :, 1], B[None, :, 0], B[None, :, 1]
    den = np.where(np.abs(by - ay) < 1e-12, 1e-12, by - ay)
    hit = ((ay > y) != (by > y)) & (x < ax + (y - ay) * (bx - ax) / den)
    return np.where(hit.sum(1) % 2 == 1, dist, -dist)


def _inset(poly, d):
    n = len(poly)
    out = []
    for i in range(n):
        p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % n]
        e0 = (p1[0] - p0[0], p1[1] - p0[1])
        e1 = (p2[0] - p1[0], p2[1] - p1[1])
        l0 = hypot(*e0) or 1e-9
        l1 = hypot(*e1) or 1e-9
        n0 = (-e0[1] / l0, e0[0] / l0)
        n1 = (-e1[1] / l1, e1[0] / l1)
        k = max(0.5, 1.0 + n0[0] * n1[0] + n0[1] * n1[1])
        out.append((p1[0] + d * (n0[0] + n1[0]) / k, p1[1] + d * (n0[1] + n1[1]) / k))
    return out


def _inset_ok(poly, d):
    ring = _inset(poly, d)
    return ring if float(_pdist(ring, poly).min()) > 0.6 * d else None


def _walk(poly, i, step, s):
    """Point at arc length s from vertex i walking in direction step (+1 / -1); also the vertices passed."""
    n = len(poly)
    acc = 0.0
    k = 0
    while k < n - 1:
        a, b = poly[(i + step * k) % n], poly[(i + step * (k + 1)) % n]
        L = hypot(b[0] - a[0], b[1] - a[1])
        if acc + L >= s:
            f = (s - acc) / max(L, 1e-9)
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), k
        acc += L
        k += 1
    return None, k


def _blunt(poly, w0=0.05, ang=55.0):
    """Masons never leave a feather edge: cut every tip sharper than `ang` back to where the stone
    is w0 wide (the little triangle is pinned up with mortar). Returns None if nothing useful is left."""
    ca = cos(radians(ang))
    for _ in range(6):
        n = len(poly)
        if n < 4:
            return poly if n == 3 else None
        tip = None
        for i in range(n):
            p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % n]
            ax, az = p0[0] - p1[0], p0[1] - p1[1]
            bx, bz = p2[0] - p1[0], p2[1] - p1[1]
            la, lb = hypot(ax, az), hypot(bx, bz)
            if la < 1e-6 or lb < 1e-6:
                continue
            if (ax * bx + az * bz) / (la * lb) > ca and (-ax) * bz - (-az) * bx > 0:
                tip = i
                break
        if tip is None:
            return poly
        per = sum(hypot(poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1]) for i in range(n))
        s = 0.02
        found = None
        while s < per * 0.4:
            a, kf = _walk(poly, tip, 1, s)
            b, kb = _walk(poly, tip, -1, s)
            if a is None or b is None:
                break
            if hypot(a[0] - b[0], a[1] - b[1]) >= w0:
                found = (a, kf, b, kb)
                break
            s += 0.012
        if not found:
            return None
        a, kf, b, kb = found
        keep = n - (kf + 1) - (kb + 1) + 1
        if keep < 1:
            return None
        poly = _dedupe([a] + [poly[(tip + kf + 1 + q) % n] for q in range(keep)] + [b])
    return poly


def _boss_fn(r, lo, hi, h, flats):
    """Height field of a pitched (rock-faced) boss: a few irregular chiselled facets, a conchoidal
    spall or two and fine tooling ripple. Dressed flat where an iron plate is seated (flats)."""
    planes = []
    for _ in range(int(r.u(3.0, 5.99))):
        a = r.u(0, 2 * pi)
        sl = r.u(0.14, 0.42)
        planes.append((r.u(lo[0], hi[0]), r.u(lo[1], hi[1]), cos(a) * sl, sin(a) * sl, h * r.u(0.8, 1.3)))
    waves = [(r.u(0, 2 * pi), 2 * pi / r.u(0.045, 0.15), r.u(0, 2 * pi), r.u(0.03, 0.08) * h) for _ in range(4)]
    spalls = [(r.u(lo[0], hi[0]), r.u(lo[1], hi[1]), r.u(0.06, 0.15), h * r.u(0.3, 0.75)) for _ in range(2) if r.chance(0.45)]

    def fn(P):
        P = np.asarray(P, dtype=float)
        X, Z = P[:, 0], P[:, 1]
        Hh = np.full(len(P), 9.0)
        for (cx, cz, gx, gz, hh) in planes:
            Hh = np.minimum(Hh, hh + gx * (X - cx) + gz * (Z - cz))
        Hh = np.maximum(Hh, 0.3 * h)
        for (a, kk, ph, amp) in waves:
            Hh = Hh + amp * np.sin(kk * (X * cos(a) + Z * sin(a)) + ph)
        for (sx, sz, sr, sd) in spalls:
            Hh = Hh - sd * np.clip(1.0 - np.hypot(X - sx, Z - sz) / sr, 0.0, 1.0) ** 1.5
        Hh = np.maximum(Hh, 0.15 * h)
        for (fx_, fz_, fr) in flats:
            q = np.clip((np.hypot(X - fx_, Z - fz_) - fr) / 0.09, 0.0, 1.0)
            Hh = Hh * q * q * (3 - 2 * q)
        return Hh
    return fn


def _stone(acc, poly, y_face, y_back, ch, bulge, r, cell=0.075, margin=0.03, flats=(), chip=0.3):
    """One cut stone from a CCW outline (x, z) in the face plane: sides back to y_back, chamfered
    arris, drafted margin and a pitched rock-faced boss (bulge > 0) or a tooled flat face."""
    poly = _dedupe(poly)
    poly = _blunt(poly) if len(poly) >= 3 else None
    if not poly or len(poly) < 3:
        return False
    A = np.array(poly, dtype=float)
    n = len(poly)
    lo, hi = A.min(0), A.max(0)
    nx = max(2, int(ceil((hi[0] - lo[0]) / cell)))
    nz = max(2, int(ceil((hi[1] - lo[1]) / cell)))
    rs = np.random.RandomState(int(r.u(0, 2 ** 31 - 2)))
    cw = np.array([(hi[0] - lo[0]) / nx, (hi[1] - lo[1]) / nz])
    G = np.stack(np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cw[0], lo[1] + (np.arange(nz) + 0.5) * cw[1]), -1).reshape(-1, 2)
    G = G + rs.uniform(-0.27, 0.27, G.shape) * cw
    sdG = _pdist(G, A)
    inr = float(sdG.max())
    if inr < 0.03:
        return False
    tone = r.u(0, 1)
    k = min(1.0, inr / 0.12)
    c1 = min(ch, 0.3 * inr)
    b0 = len(acc.v)
    for q in poly:
        acc.add((q[0], y_back, q[1]), tone, 1.0)
    for q in poly:
        acc.add((q[0], y_face + c1, q[1]), tone, 1.0)
    rings = 2
    inner, dtot = poly, 0.0
    r2 = _inset_ok(poly, c1) or _inset_ok(poly, c1 * 0.5)
    hf = None
    if r2 is not None:
        for q in r2:
            acc.add((q[0], y_face + rs.uniform(-1, 1) * 0.0012, q[1]), tone, 0.75)
        rings, inner, dtot = 3, r2, c1
        if bulge > 1e-4:
            mg = min(margin * r.u(0.8, 1.2), 0.25 * inr)
            st = min(0.02, 0.15 * inr)
            r3, r4 = _inset_ok(poly, c1 + mg), _inset_ok(poly, c1 + mg + st)
            if r3 is not None and r4 is not None:
                hf = _boss_fn(r, lo, hi, bulge * k, flats)
                for q in r3:
                    acc.add((q[0], y_face + rs.uniform(-1, 1) * 0.0015, q[1]), tone, 0.3)
                h4 = hf(np.array(r4))
                for q, hh in zip(r4, h4):
                    acc.add((q[0], y_face - 0.55 * hh * r.u(0.8, 1.15), q[1]), tone, 0.0)
                rings, inner, dtot = 5, r4, c1 + mg + st
    for i in range(n):
        j = (i + 1) % n
        for m in range(rings - 1):
            a_, b_ = b0 + m * n, b0 + (m + 1) * n
            acc.f.append((a_ + i, a_ + j, b_ + j, b_ + i))
    # face inside the innermost ring: constrained Delaunay over jittered grid points
    ib = b0 + (rings - 1) * n
    pts = G[sdG > dtot + 0.42 * cell]
    done = False
    try:
        vin = [Vector(q) for q in inner] + [Vector((float(q[0]), float(q[1]))) for q in pts]
        res = delaunay_2d_cdt(vin, [], [list(range(n))], 1, 1e-6)
        ov, of, orig = res[0], res[2], res[3]
        if len(of) >= n - 2:
            OP = np.array([(v.x, v.y) for v in ov])
            if hf is not None:
                dd = np.clip(_pdist(OP, np.array(inner)), 0.0, 1.0)
                e = np.clip(dd / 0.045, 0.0, 1.0)
                yy = y_face - hf(OP) * (0.55 + 0.45 * e * e * (3 - 2 * e)) - rs.uniform(-1, 1, len(OP)) * 0.0018
            else:
                yy = y_face + rs.uniform(-1, 1, len(OP)) * 0.0014
            idx = []
            for i, v in enumerate(ov):
                src = [s for s in orig[i] if s < n]
                if src:
                    idx.append(ib + src[0])
                else:
                    idx.append(acc.add((float(OP[i, 0]), float(yy[i]), float(OP[i, 1])), tone, 0.0))
            for tri in of:
                if len(tri) != 3:
                    continue
                a_, b_, c_ = (OP[q] for q in tri)
                if (b_[0] - a_[0]) * (c_[1] - a_[1]) - (b_[1] - a_[1]) * (c_[0] - a_[0]) > 0:
                    acc.f.append((idx[tri[0]], idx[tri[1]], idx[tri[2]]))
                else:
                    acc.f.append((idx[tri[0]], idx[tri[2]], idx[tri[1]]))
            done = True
    except Exception as ex:                    # degenerate outline: fall back to a fan
        print("DBG cdt fallback", ex)
    if not done:
        c = G[int(np.argmax(sdG))]
        ci = acc.add((float(c[0]), y_face, float(c[1])), tone, 0.0)
        for i in range(n):
            acc.f.append((ib + i, ib + (i + 1) % n, ci))
    # knocked arrises / chipped corners
    if chip > 0 and r.chance(chip):
        for _ in range(1 if r.chance(0.65) else 2):
            q = poly[min(n - 1, int(r.u(0, n)))]
            rc, dc = r.u(0.05, 0.12), r.u(0.012, 0.03)
            for vi in range(b0 + n, len(acc.v)):
                x, y, z = acc.v[vi]
                d = hypot(x - q[0], z - q[1])
                if d < rc:
                    acc.v[vi] = (x, min(y_back - 0.004, y + dc * (1.0 - d / rc) ** 0.6), z)
    return True


def _rect_pts(x0, x1, z0, z1, step=0.18):
    pts = []
    c = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        m = max(1, int(ceil(hypot(b[0] - a[0], b[1] - a[1]) / step)))
        for q in range(m):
            t = q / m
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return pts


def _clip_disc(pts, cz, rr):
    """Cut the disc (centre (0, cz), radius rr) out of a CCW polygon given as dense boundary points."""
    def inside(q):
        return q[0] * q[0] + (q[1] - cz) ** 2 < rr * rr

    fl = [inside(q) for q in pts]
    if not any(fl):
        return pts
    if all(fl):
        return None
    n = len(pts)

    def cut(po, pi_):
        for _ in range(22):
            m = ((po[0] + pi_[0]) * 0.5, (po[1] + pi_[1]) * 0.5)
            if inside(m):
                pi_ = m
            else:
                po = m
        return po

    s = next(i for i in range(n) if not fl[i] and fl[i - 1])
    out = []
    ent = None
    for q in range(n):
        i = (s + q) % n
        j = (i + 1) % n
        if not fl[i]:
            out.append(pts[i])
            if fl[j]:
                ent = cut(pts[i], pts[j])
                out.append(ent)
        elif not fl[j]:
            ex = cut(pts[j], pts[i])
            t1 = atan2(ent[1] - cz, ent[0])
            t2 = atan2(ex[1] - cz, ex[0])
            d = (t2 - t1 + pi) % (2 * pi) - pi
            m = max(1, int(ceil(abs(d) / radians(6.0))))
            for k in range(1, m):
                th = t1 + d * k / m
                out.append((rr * cos(th), cz + rr * sin(th)))
            out.append(ex)
    return out


def _run(r, xa, xb, lmin, lmax, first=None):
    if xb - xa < 0.08:
        return []
    out = []
    x = xa
    L = first if first else r.u(lmin, lmax)
    while True:
        x1 = x + L
        if xb - x1 < lmin * 0.6:
            out.append((x, xb))
            break
        out.append((x, x1))
        x = x1
        L = r.u(lmin, lmax)
    return out


def _wedge(name, a0, a1, r0, r1, y0, y1, cz, bev=0.014, bseg=2, arc=0.2):
    seg = max(1, int(ceil(abs(a1 - a0) * r1 / arc)))
    verts, faces = [], []
    for k in range(seg + 1):
        a = a0 + (a1 - a0) * k / seg
        c, s = cos(a), sin(a)
        for (rr, y) in ((r0, y0), (r1, y0), (r1, y1), (r0, y1)):
            verts.append((rr * c, y, cz + rr * s))
    for k in range(seg):
        b, n = 4 * k, 4 * (k + 1)
        for q in range(4):
            faces.append((b + q, b + (q + 1) % 4, n + (q + 1) % 4, n + q))
    faces.append((3, 2, 1, 0))
    e = 4 * seg
    faces.append((e, e + 1, e + 2, e + 3))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    bevel(o, bev, segments=bseg, angle=30)
    return o


def _prism(acc, prof, x0, x1, z):
    n = len(prof)
    b = len(acc.v)
    for x in (x0, x1):
        for (y, zz) in prof:
            acc.add((x, y, z + zz))
    for i in range(n):
        j = (i + 1) % n
        acc.f.append((b + i, b + j, b + n + j, b + n + i))
    acc.f.append(tuple(b + i for i in range(n))[::-1])
    acc.f.append(tuple(b + n + i for i in range(n)))


def _levels(z0, z1, zf0, zf1, fine=0.07, coarse=0.19):
    """z stations of a wall: fine rows where the tide band / wet line needs mask resolution."""
    out = [z0]
    z = z0
    while z < z1 - 1e-6:
        z = min(z1, z + (fine if zf0 - coarse < z < zf1 else coarse))
        if z1 - z < 0.04:
            z = z1
        out.append(z)
    return out


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build(p, rng):
    W, H, D = p.width, p.height, p.depth
    free = p.mount == "ground"
    ts = 1.0 if p.towpath_side == "right" else -1.0
    a_style = {"round": 0.0, "horseshoe": -24.0, "segmental": 46.0}[p.arch]
    a_lim = 90.0 - 2.0 * degrees(atan(2.0 * (H - 0.5) / W))      # keeps the springing >= 0.5 m above the quay
    a0 = radians(min(62.0, max(a_style, a_lim)))
    R = (W / 2) / cos(a0)
    czc = H - R                                # arch centre height
    zs = czc + R * sin(a0)                     # springing height
    span = pi - 2 * a0
    t = clamp(0.13 * W, 0.42, 0.62)            # ring thickness
    Rext = R + t
    rd = 0.5                                   # ring depth into the tunnel
    yb = 0.62                                  # back of the face wall
    ch = p.channel
    wl = min(p.water_level, ch - 0.25)
    tw = min(p.towpath, W * 0.34)
    xl = ts * (W / 2 - tw)                     # channel edge of the towpath kerb
    xw = W / 2 + 0.08                          # interior wall plane
    Rv = R + 0.08                              # vault lining radius
    kc = max(1, int(round(zs / 0.36)))
    hc = zs / kc                               # course height (a bed joint lands on the springing)
    z_corn = H + t + p.top
    xring = Rext if a0 < 0 else Rext * cos(a0)
    X = max(W / 2 + p.wing, xring + 0.3)       # half width of the face wall
    zb = -ch - 0.25
    y_dark = D - clamp(0.24 * D, 0.55, 0.9)    # the unlit, soot-black last bay starts behind the last rib
    nr = max(1, int(round((y_dark - rd) / 1.15)))
    bay = (y_dark - rd) / nr
    yl = rd + (bay - 0.14) * 0.5               # lantern: middle of the first bay
    mk = p.moss

    m_face = mat("stone_rough", _shade(p.stone, 0.9))
    m_dress = mat("stone_rough", _shade(p.stone, 1.1, 0.02), scale=1.8)
    m_wall = mat({"brick": "brick_red", "ochre_brick": "brick_ochre", "stone": "ashlar_limestone"}[p.vault])
    m_iron = mat("iron_wrought")
    m_dk = mat("stone_rough", "#0d0c0b")       # soot-black masonry of the last bay and the closed end

    def xe(z):
        return sqrt(max(0.0, Rext * Rext - (z - czc) ** 2))

    def jw(j):                                 # long-and-short jamb quoins
        return t + 0.12 if (kc - 1 - j) % 2 == 0 else t - 0.1

    courses = []
    j = int(floor(zb / hc))
    while True:
        z0, z1 = max(j * hc, zb), (j + 1) * hc
        if z1 >= z_corn - 0.16:
            courses.append((j, z0, z_corn))
            break
        if z1 - z0 > 0.05:
            courses.append((j, z0, z1))
        j += 1

    # tie-rod pattress plates on the spandrels (the bosses are dressed flat where they are seated)
    rock = p.rustication == "rock"
    zp = max(zs + 0.45, H - 0.45)
    plates = []
    if X - xring > 0.45:
        for sx in (1, -1):
            plates.append((sx * (X + max(xring, xe(zp))) / 2, zp))
    flats = [(px, pz, 0.2) for (px, pz) in plates]

    # ------------------------------------------------------------------ face wall blocks
    chf = 0.014 if rock else 0.024
    bul = 0.05 if rock else 0.0
    area = 2 * X * (z_corn - zb) - W * (zs + ch) - 0.5 * pi * R * R * (span / pi)
    cell = clamp(sqrt(max(area, 1.0) / 8000.0), 0.048, 0.115) * (1.0 if rock else 1.35)
    bstep = clamp(3.4 * cell, 0.18, 0.4)      # outline step of a block (big walls: coarser)
    acc = _Acc()
    rf = rng.fork("face")
    g = 0.004
    lmin, lmax = 1.5 * hc, 2.8 * hc
    for (j, z0, z1) in courses:
        if z1 <= zs + 1e-4:
            xin = W / 2 + jw(j)
            runs = [(sx, _run(rf, xin, X, lmin, lmax)) for sx in (1, -1)]
            clip = False
        elif z0 >= czc + Rext - 1e-4:
            runs = [(1, _run(rf, -X, X, lmin, lmax))]
            clip = False
        elif z1 >= czc + Rext - 0.03:
            runs = [(1, _run(rf, -X, X, lmin, lmax))]
            clip = True
        else:
            es = [xe(z0 + (z1 - z0) * i / 10) for i in range(11)]
            ein, eout = min(es), max(es)
            runs = [(sx, _run(rf, ein, X, lmin, lmax, first=(eout - ein) + rf.u(0.3, 0.3 + hc))) for sx in (1, -1)]
            clip = True
        for (sx, blocks) in runs:
            for (xa, xb) in blocks:
                xa, xb = (xa, xb) if sx > 0 else (-xb, -xa)
                pts = _rect_pts(xa + g, xb - g, z0 + g, z1 - g, bstep)
                if clip:
                    pts = _clip_disc(pts, czc, Rext + g)
                    if pts is None:
                        continue
                yf = -0.012 - (rf.u(0, 0.02) if rock else rf.u(0, 0.004))     # every block sits a little differently
                _stone(acc, pts, yf, 0.04, chf, bul * rf.u(0.6, 1.4), rf, cell=cell, flats=flats)
    # stone pier carrying the towpath at the portal (below the threshold stone)
    pier = []
    for (j, z0, z1) in courses:
        a_, b_ = max(z0, -ch), min(z1, -0.158)
        if b_ - a_ > 0.02:
            pier.append([a_, b_])
    if len(pier) > 1 and pier[-1][1] - pier[-1][0] < 0.15:
        pier[-2][1] = pier[-1][1]
        pier.pop()
    pxa, pxb = sorted((xl + ts * 0.03, ts * W / 2))
    for q, (z0, z1) in enumerate(pier):
        cuts = [pxa, pxb]
        if pxb - pxa > 0.7:
            cuts = [pxa, pxa + (pxb - pxa) * (0.38 if q % 2 else 0.62), pxb]
        for c in range(len(cuts) - 1):
            _stone(acc, _rect_pts(cuts[c] + g, cuts[c + 1] - g, z0 + g, z1 - g), -0.02 - rf.u(0, 0.012), 0.33, chf,
                   bul * rf.u(0.6, 1.3), rf, cell=cell)
    # sill course under the channel mouth
    for (xa_, xb_) in _run(rf, -W / 2, W / 2, 0.7, 1.2):
        _stone(acc, _rect_pts(xa_ + g, xb_ - g, zb + g, -ch - g), -0.02 - rf.u(0, 0.012), 0.04, chf, bul * rf.u(0.6, 1.3), rf, cell=cell)
    face = acc.obj("face_blocks")
    assign(face, m_face)
    smooth(face, 48)
    f_tone, f_edge = np.array(acc.tone), np.array(acc.edge)

    # backing wall (lime mortar shows in the joints; the opening is cut through it)
    e = 0.1
    xlr = W / 2 - tw
    hole = [(-(W / 2 + e), -ch), (xlr + 0.06, -ch), (xlr + 0.06, -0.165), (W / 2 + e, -0.165), (W / 2 + e, zs)]
    na = max(8, int(span / radians(7)))
    for k in range(1, na):
        th = a0 + span * k / na
        hole.append(((R + e) * cos(th), czc + (R + e) * sin(th)))
    hole.append((-(W / 2 + e), zs))
    if ts < 0:
        hole = [(-x, z) for (x, z) in hole][::-1]
    slab = shape_extrude("wall_slab", [(-X, zb), (X, zb), (X, z_corn + 0.2), (-X, z_corn + 0.2)], [hole],
                         depth=yb - 0.012, loc=(0, yb, 0), rot=(90, 0, 0))
    apply_transform(slab)
    assign(slab, m_wall)
    assign_faces(slab, m_face, lambda c, n: n.y < -0.5)
    densify(slab, 0.9)
    uv_box(slab)
    paint(slab, grime=0.35)

    # ------------------------------------------------------------------ dressings
    dress = []
    rv = rng.fork("ring")
    nv = int(round(R * span / 0.38))
    nv += 1 if nv % 2 == 0 else 0
    nv = max(nv, 7)
    kw = 1.2
    da = span / (nv - 1 + kw)
    a = a0
    for i in range(nv):
        key = i == nv // 2
        w = da * (kw if key else 1.0)
        if key:
            dress.append(_wedge("keystone", a, a + w, R - 0.035, Rext + 0.17, -0.1, rd, czc, bev=0.018))
        else:
            dress.append(_wedge("voussoir", a, a + w, R, Rext, -0.05 - rv.u(0, 0.012), rd, czc, bev=0.016))
        a += w
    ktop = H + t + 0.17
    kwid = da * kw * (Rext + 0.17)
    dress.append(stone_block("key_cap", (kwid + 0.07, 0.21, 0.075), loc=(0, -0.015, ktop + 0.03), rng=rv, chamfer=0.014, detail=0.2))
    # skewbacks (segmental arches): the stone between the impost and the first voussoir
    if a0 > 0.03:
        a1 = asin(clamp((zs - czc) / Rext, -1.0, 1.0))
        sk = [(W / 2, zs), ((W / 2 + Rext * cos(a1)) / 2, zs)]
        ns = max(2, int((a0 - a1) / radians(6)) + 1)
        for k in range(ns + 1):
            th = a1 + (a0 - a1) * k / ns
            sk.append((Rext * cos(th), czc + Rext * sin(th)))
        for k in (1, 2):
            f = k / 3.0
            sk.append((lerp(Rext * cos(a0), W / 2, f), lerp(czc + Rext * sin(a0), zs, f)))
        acd = _Acc()
        _stone(acd, sk, -0.05, rd, 0.016, 0.0, rv, cell=0.12, chip=0)
        _stone(acd, [(-x, z) for (x, z) in sk][::-1], -0.05, rd, 0.016, 0.0, rv, cell=0.12, chip=0)
        dress.append(acd.obj("skewbacks"))
    # jamb quoins and moulded imposts
    rq = rng.fork("quoin")
    q_det = clamp(0.36 * (zs + ch + 0.25) / 2.95, 0.36, 0.7)      # tall jambs: coarser stones, same triangle budget
    for sx in (1, -1):
        for (j, z0, z1) in courses:
            if z1 <= zs + 1e-4:
                wq = jw(j)
                dress.append(stone_block("quoin", (wq - 0.006, rd + 0.045, z1 - z0 - 0.006),
                                         loc=(sx * (W / 2 + wq / 2), (rd - 0.045) / 2, (z0 + z1) / 2),
                                         rng=rq, chamfer=0.016, rough=0.003, detail=q_det))
        dress.append(stone_block("abacus", (t + 0.24, rd + 0.11, 0.13), loc=(sx * (W / 2 + (t + 0.24) / 2 - 0.06), (rd - 0.11) / 2, zs - 0.065),
                                 rng=rq, chamfer=0.02, rough=0.002, detail=0.2))
        dress.append(stone_block("necking", (t + 0.17, rd + 0.075, 0.075), loc=(sx * (W / 2 + (t + 0.17) / 2 - 0.03), (rd - 0.075) / 2, zs - 0.165),
                                 rng=rq, chamfer=0.022, rough=0.002, detail=0.2))
    # string course (moulded stones) and coping
    rc = rng.fork("cornice")
    prof = [(0.10, 0.0), (-0.02, 0.0), (-0.02, 0.05), (-0.05, 0.07), (-0.05, 0.10), (-0.085, 0.125), (-0.125, 0.15),
            (-0.16, 0.168), (-0.16, 0.23), (-0.14, 0.26), (0.10, 0.26)]
    acn = _Acc()
    for (xa, xb) in _run(rc, -X - 0.05, X + 0.05, 0.7, 1.15):
        _prism(acn, prof, xa + 0.002, xb - 0.002, z_corn + rc.u(-0.003, 0.003))
    corn = acn.obj("string_course")
    recalc_normals(corn)
    dress.append(corn)
    cop_j = []
    for (xa, xb) in _run(rc, -X - 0.07, X + 0.07, 0.75, 1.2):
        cop_j.append(xa)
        dress.append(stone_block("coping", (xb - xa - 0.006, 0.9, 0.15), loc=((xa + xb) / 2, 0.31 + rc.u(-0.006, 0.006), z_corn + 0.335 + rc.u(-0.004, 0.004)),
                                 rng=rc, chamfer=0.022, rough=0.004, detail=0.3))

    # ------------------------------------------------------------------ interior: vault, walls, ribs, towpath
    y0i = rd - 0.03
    lined, dark = [], []

    def vault_part(ya, yb_, fine):
        def vf(u, v):
            th = (pi - a0) - span * v
            return (Rv * cos(th), ya + (yb_ - ya) * u, czc + Rv * sin(th))
        return surface("vault", vf, nu=max(2, int((yb_ - ya) / (0.15 if fine else 0.45))),
                       nv=max(10, int(Rv * span / (0.15 if fine else 0.3))))

    def wall_part(sx, ya, yb_, fine):
        zlo = -0.03 if sx == ts else -ch
        zl = _levels(zlo, zs, -wl - 0.05, -wl + 0.75) if fine else _levels(zlo, zs, 9, 9, coarse=0.5)
        nl = len(zl) - 1
        ys = (yb_, ya) if sx > 0 else (ya, yb_)

        def fn(u, v):
            return (sx * xw, ys[0] + (ys[1] - ys[0]) * u, zl[min(nl, int(round(v * nl)))])
        return surface("wall_in", fn, nu=max(2, int((yb_ - ya) / (0.15 if fine else 0.45))), nv=nl)

    up_b = max(0.0, (Rv - R) * sin(a0)) + 0.025                  # the band closes the joint between wall top and vault foot
    lined.append(vault_part(y0i, y_dark, True))
    dark.append(vault_part(y_dark, D, False))
    for sx in (1, -1):
        lined.append(wall_part(sx, y0i, y_dark, True))
        dark.append(wall_part(sx, y_dark, D, False))
        dress.append(box("impost_band", (0.14, y_dark - rd, 0.14 + up_b), loc=(sx * xw, (y_dark + rd) / 2, zs - 0.07 + up_b / 2), bevel=0.012))
        dark.append(box("impost_band_d", (0.14, D - y_dark, 0.14 + up_b), loc=(sx * xw, (D + y_dark) / 2, zs - 0.07 + up_b / 2)))
    # ribs (stone arches across the vault on wall piers); the last one frames the black bay
    rr_ = rng.fork("rib")
    nrs = max(5, int(round(Rv * span / 0.62)) | 1)
    for i in range(nr):
        yc = rd + (i + 1) * bay - (0.14 if i < nr - 1 else 0.0)
        for k in range(nrs):
            dress.append(_wedge("rib", a0 + span * k / nrs, a0 + span * (k + 1) / nrs, R + 0.012, Rv + 0.1, yc - 0.14, yc + 0.14, czc,
                                bev=0.012, bseg=1, arc=0.3))
        for sx in (1, -1):
            z = -0.02 if sx == ts else -ch
            ztop = zs - 0.14
            while z < ztop - 0.02:
                z1 = z + max(hc, (zs + ch) / 8.0) * rr_.u(0.9, 1.25)
                if ztop - z1 < 0.18:
                    z1 = ztop
                dress.append(stone_block("rib_pier", (0.13, 0.28, z1 - z - 0.004), loc=(sx * xw, yc, (z + z1) / 2), rng=rr_, chamfer=0.012,
                                         rough=0.002, detail=q_det + 0.14))
                z = z1
    # towpath: threshold stone bedded on the pier, masonry ledge, kerb stones with a worn nosing, flags behind
    rk = rng.fork("kerb")
    y_th = 0.30
    dress.append(stone_block("threshold", (abs(ts * W / 2 - xl) - 0.008, y_th + 0.06 - 0.006, 0.15),
                             loc=((xl + ts * W / 2) / 2, (y_th - 0.06) / 2, -0.075), rng=rk, chamfer=0.02, rough=0.003, detail=0.12))
    xa, xb = sorted((xl + ts * 0.035, ts * (xw + 0.05)))
    zl_ = _levels(-ch, -0.09, -wl - 0.05, -wl + 0.75)
    ys_ = (y_dark, 0.3) if ts > 0 else (0.3, y_dark)
    lined.append(surface("ledge", lambda u, v: (xl + ts * 0.035, ys_[0] + (ys_[1] - ys_[0]) * u, zl_[min(len(zl_) - 1, int(round(v * (len(zl_) - 1))))]),
                         nu=max(2, int((y_dark - 0.3) / 0.15)), nv=len(zl_) - 1))
    dark.append(box("ledge_d", (xb - xa, D - y_dark, ch - 0.004), loc=((xa + xb) / 2, (D + y_dark) / 2, (-ch - 0.004) / 2)))
    kerbs, flags = [], []
    wf = (xw - 0.005) - (W / 2 - tw + 0.345)
    kw_ = 0.34 if wf > 0.15 else 0.34 + wf
    y = y_th
    while y < y_dark - 0.01:
        y1 = y + rk.u(0.6, 0.95)
        if y_dark - y1 < 0.35:
            y1 = y_dark
        kerbs.append(stone_block("kerb", (kw_, y1 - y - 0.006, 0.17), loc=(xl + ts * kw_ / 2, (y + y1) / 2, -0.085 + rk.u(-0.006, 0.003)),
                                 rng=rk, chamfer=0.024, rough=0.004, detail=0.17))
        y = y1
    if wf > 0.15:
        y = y_th
        xf0 = W / 2 - tw + 0.345
        while y < y_dark - 0.01:
            y1 = y + rk.u(0.45, 0.8)
            if y_dark - y1 < 0.3:
                y1 = y_dark
            cuts = [xf0, xf0 + wf * rk.u(0.4, 0.6), xf0 + wf] if wf > 0.7 else [xf0, xf0 + wf]
            for q in range(len(cuts) - 1):
                flags.append(stone_block("flag", (cuts[q + 1] - cuts[q] - 0.008, y1 - y - 0.008, 0.07),
                                         loc=(ts * (cuts[q] + cuts[q + 1]) / 2, (y + y1) / 2, -0.035 - rk.u(0, 0.007)),
                                         rng=rk, chamfer=0.009, rough=0.003, detail=0.25))
            y = y1
    # channel bed, water, closed end
    xa, xb = sorted((-ts * xw, xl + ts * 0.035))
    bed = grid("bed", size=(xb - xa, D), cuts=(1, 1), loc=((xa + xb) / 2, D / 2, -ch))
    dark.append(bed)
    ei = 0.02
    endp = [(-(xw + ei), -ch - ei), (xw + ei, -ch - ei), (xw + ei, zs)]
    for k in range(1, na):
        th = a0 + span * k / na
        endp.append(((Rv + ei) * cos(th), czc + (Rv + ei) * sin(th)))
    endp.append((-(xw + ei), zs))
    dark.append(extrude_profile("end_dark", endp, depth=0.03, loc=(0, D + 0.03, 0), rot=(90, 0, 0), cap=True))
    for o in dark:
        assign(o, m_dk)
        paint(o, grime=1.0)
    for o in lined:
        assign(o, m_wall)
    if p.water:
        m_water = mat("water", "#040606", opacity=1.0)
        wat = grid("water", size=(xb - xa, D - 0.003), cuts=(max(4, int((xb - xa) / 0.22)), max(6, int(D / 0.22))),
                   loc=((xa + xb) / 2, (D + 0.003) / 2, -wl))
        assign(wat, m_water)
        # the cut through the water body at the portal (hidden below the canal surface in town)
        sk_ = mesh_object("water_section", [(xa, 0.003, -ch), (xb, 0.003, -ch), (xb, 0.003, -wl), (xa, 0.003, -wl)], [(0, 1, 2, 3)])
        assign(sk_, m_water)
        if p.lantern:
            # the lantern's broken reflection on the ripples: a column of thin horizontal glints towards the portal
            rg = rng.fork("glint")
            ag = _Acc()
            xg0, xg1 = min(xa, xb) + 0.12, max(xa, xb) - 0.12

            def dash(gx, gy, ga, gb):
                gx = clamp(gx, xg0 + ga, xg1 - ga)
                rot = radians(rg.u(-7, 7))
                b0 = len(ag.v)
                for (u_, v_) in ((-1, 0), (-0.5, -0.85), (0.05, -1), (0.55, -0.8), (1, 0), (0.5, 0.85), (-0.05, 1), (-0.55, 0.8)):
                    dx, dy = u_ * ga, v_ * gb * rg.u(0.6, 1.2)
                    ag.add((gx + dx * cos(rot) - dy * sin(rot), gy + dx * sin(rot) + dy * cos(rot), -wl + 0.004))
                ag.f.append(tuple(range(b0, b0 + 8)))

            ya_, yb_ = 0.06, min(y_dark - 0.1, yl + 0.45)
            for _ in range(int(clamp(64 * (yb_ - ya_) / 1.4, 30, 90))):       # the lamp itself: a broken column towards the viewer
                f = rg.u(0, 1) ** 0.8
                gy = lerp(yb_, ya_, f)
                core = 1.0 - min(1.0, abs(gy - (yl - 0.25)) / 0.7)             # brightest patch just in front of the nadir
                dash(rg.u(-1, 1) * rg.u(0.1, 1) * (0.07 + 0.2 * f), gy, rg.u(0.03, 0.09) * (0.7 + 1.5 * core), rg.u(0.005, 0.011) * (0.8 + 1.0 * core))
            for sx in (1, -1):                                                 # the lit walls: sparse sparkle along their foot
                xwall = xl + ts * 0.035 if sx == ts else sx * xw
                for _ in range(int(clamp(9 * (y_dark - 0.3), 6, 26))):
                    gy = rg.u(0.25, min(y_dark - 0.2, yl + 1.3))
                    dash(xwall - sx * rg.u(0.1, 0.5), gy, rg.u(0.02, 0.06), rg.u(0.004, 0.008))
            gl = ag.obj("water_glints")
            assign(gl, mat("emissive", "#ff9a3a", glow=1.5))

    # masonry backing of the vault (free-standing only): abutment walls, haunching, brick extrados, closed back
    if free:
        Ro = Rv + 0.34
        xo = min(X - 0.04, max(xw, Rv if a0 < 0 else 0.0) + 0.5)
        ah = max(a0 + radians(14), radians(28))
        if xo < Ro * cos(ah) + 0.08:
            ah = max(ah, atan2(sqrt(max(Ro * Ro - xo * xo, 0.0)), xo))
            sp = [(xo, zb)]
        else:
            sp = [(xo, zb), (xo, czc + Ro * sin(ah) - 0.12)]
        nh = len(sp)
        nsh = max(8, int((pi - 2 * ah) / radians(8)))
        for k in range(nsh + 1):
            th = ah + (pi - 2 * ah) * k / nsh
            sp.append((Ro * cos(th), czc + Ro * sin(th)))
        sp += [(-x, z) for (x, z) in sp[:nh]][::-1]
        sp = _dedupe(sp)
        shell = extrude_profile("backing", sp, depth=D + 0.3 - yb + 0.01, loc=(0, D + 0.3, 0), rot=(90, 0, 0), cap=False)
        apply_transform(shell)
        shell = join([shell, mesh_object("backing_end", [(x, D + 0.3, z) for (x, z) in sp][::-1], [tuple(range(len(sp)))])], "backing")
        densify(shell, 0.5)
        assign(shell, m_wall)
        uv_box(shell)
        paint(shell, grime=lambda co, n: np.clip(0.1 + 0.4 * np.clip(n[:, 2], 0, 1) + 0.3 * np.clip(1 - (co[:, 2] - zb) / 1.2, 0, 1), 0, 1))

    # ------------------------------------------------------------------ ironwork
    iron = []

    def ring_fix(x, y, z, r=0.085):
        iron.append(box("ring_plate", (0.11, 0.012, 0.13), loc=(x, y - 0.006, z + 0.01), bevel=0.003))
        for sx_ in (-1, 1):
            iron.append(bolt("ring_nail", 0.011, 0.007, loc=(x + sx_ * 0.036, y - 0.012, z + 0.055), rot=(90, 0, 0), kind="pyramid"))
        iron.append(torus("ring_eye", 0.028, 0.009, loc=(x, y - 0.034, z), rot=(0, 90, 0), major_seg=14, minor_seg=6))
        iron.append(torus("ring", r, 0.012, loc=(x, y - 0.04, z - r + 0.014), rot=(96, 0, 0), major_seg=22, minor_seg=7))

    stains_r = []
    if zs > 0.95:
        for sx in (1, -1):
            ring_fix(sx * (W / 2 + 0.2), -0.045, 0.62)
            stains_r.append((sx * (W / 2 + 0.2), 0.5))
    ring_fix(ts * (W / 2 - tw * 0.5), -0.062, -0.075, r=0.06)
    # cast-iron cross pattress plates with a boss, square nut and the end of the tie rod
    yp_ = -0.04 if rock else -0.022
    stains = []
    for (xp, zp_) in plates:
        for sgn in (1, -1):
            dx, dz = cos(radians(45)) * sgn, sin(radians(45))
            nx_, nz_ = -dz, dx
            if sgn < 0:
                nx_, nz_ = -nx_, -nz_
            L_, w_, we_ = 0.23, 0.021, 0.043
            o_ = [(-L_, -we_), (-0.6 * L_, -w_), (0.6 * L_, -w_), (L_, -we_), (L_, we_), (0.6 * L_, w_), (-0.6 * L_, w_), (-L_, we_)]
            bar = [(xp + u_ * dx + v_ * nx_, zp_ + u_ * dz + v_ * nz_) for (u_, v_) in o_]
            if sgn < 0:
                bar = bar[::-1]
            iron.append(extrude_profile("pattress_bar", bar, depth=0.016, loc=(0, yp_, 0), rot=(90, 0, 0), cap=True))
        iron.append(cylinder("pattress_boss", 0.058, 0.03, loc=(xp, yp_ - 0.022, zp_), rot=(90, 0, 0), verts=14))
        iron.append(bolt("pattress_nut", 0.03, 0.022, loc=(xp, yp_ - 0.036, zp_), rot=(90, 0, 22), kind="square"))
        iron.append(cylinder("tie_rod", 0.012, 0.03, loc=(xp, yp_ - 0.07, zp_), rot=(90, 0, 0), verts=8))
        stains.append((xp, zp_ - 0.12))
    # boatmen's hauling chain along the towpath wall
    zh = min(0.98, zs - 0.32)
    if zh > 0.6:
        xc_ = ts * (xw - 0.095)
        yce = y_dark - 0.32
        ne = max(2, int(round((yce - rd - 0.2) / 1.1)) + 1)
        eyes = [rd + 0.2 + (yce - rd - 0.2) * k / (ne - 1) for k in range(ne)]
        path = []
        for k, ye in enumerate(eyes):
            iron.append(cylinder("eye_stem", 0.008, 0.1, loc=(ts * (xw - 0.045), ye, zh), rot=(0, 90, 0), verts=7))
            iron.append(torus("eye", 0.022, 0.006, loc=(xc_, ye, zh), rot=(90, 0, 0), major_seg=12, minor_seg=5))
            if k:
                path += catenary((xc_, eyes[k - 1], zh - 0.018), (xc_, ye, zh - 0.018), sag=0.07, n=12)[(1 if k > 1 else 0):]
        iron.append(chain("haul_chain", path, link_len=0.11, wire=0.008, sides=4))
    if p.date:
        iron.append(text("date", str(p.date), size=min(0.12, 0.72 * kwid / max(1, len(str(p.date))) / 0.62), depth=0.008,
                         loc=(0, -0.1, H + t * 0.42), font="roman"))
    for o in iron:
        assign(o, m_iron)

    # ------------------------------------------------------------------ lantern
    if p.lantern:
        ln = use("hanging_lantern", drop=clamp(0.25 * H, 0.4, 1.1), style=p.lantern_style, size=clamp(0.085 * W, 0.22, 0.36),
                 intensity=p.intensity, seed=p.seed + 11)
        ln.place((0, yl, czc + Rv - 0.004))
    socket("vault_lantern", "attach", (0, yl, czc + Rv), role="lantern")
    for sx, nm in ((1, "r"), (-1, "l")):
        socket("face_lantern_" + nm, "attach", (sx * (W / 2 + (X - W / 2) * 0.55), -0.06, min(2.5, z_corn - 0.4)), role="lantern")
    socket("sign", "attach", (0, -0.13, z_corn - min(0.35, p.top * 0.5)), role="sign")

    # ------------------------------------------------------------------ moss cushions in joints and on ledges
    if mk > 0.05:
        rm = rng.fork("moss")
        mm = []

        def clump(x, y, z, sx_, sy_, sz_):
            mm.append(trs((x, y, z), rot=(rm.u(-12, 12), rm.u(-12, 12), rm.u(0, 360)), scale=(sx_, sy_, sz_)))

        def pick(seq):
            return seq[min(len(seq) - 1, int(rm.u(0, len(seq))))]

        for _ in range(int(round(24 * mk * X / 3.0))):               # coping: back edge and joints
            x = (pick(cop_j) + rm.u(-0.03, 0.03)) if (cop_j and rm.chance(0.45)) else rm.u(-X, X)
            clump(x, rm.u(-0.08, 0.7), z_corn + 0.408, rm.u(0.05, 0.16), rm.u(0.05, 0.14), rm.u(0.025, 0.06))
        joints = [jj * hc for jj in range(-8, 3) if -wl + 0.03 < jj * hc < 0.5]
        for _ in range(int(round(30 * mk))):                         # bed joints of the damp lower courses
            if not joints:
                break
            sx = pick((1, -1))
            x = sx * rm.u(W / 2 + 0.12, X - 0.08)
            clump(x, -0.012, pick(joints) + rm.u(-0.008, 0.008), rm.u(0.07, 0.2), rm.u(0.03, 0.05), rm.u(0.03, 0.055))
        for sx in (1, -1):                                           # impost ledges
            for _ in range(int(round(5 * mk))):
                clump(sx * (W / 2 + rm.u(0.0, t + 0.1)), -0.085 + rm.u(-0.01, 0.02), zs, rm.u(0.05, 0.11), rm.u(0.04, 0.06), rm.u(0.025, 0.045))
        for _ in range(int(round(9 * mk))):                          # foot of the towpath wall
            clump(ts * (xw - 0.02), rm.u(rd + 0.1, y_dark - 0.3), -0.002, rm.u(0.04, 0.07), rm.u(0.08, 0.2), rm.u(0.025, 0.05))
        if mm:
            mb = sphere("moss_base", 0.5, segments=7, rings=4)
            assign(mb, m_face)                 # the stone's own moss layer (special mask) - no extra draw call
            smooth(mb, 80)
            mo = instances_to_mesh("moss_cushions", mb, mm, keep_base=False)
            paint(mo, special=1.0, grime=lambda co, n: 0.5 * np.clip(-n[:, 2], 0, 1))

    # ------------------------------------------------------------------ weathering
    dress = join([o for o in dress if o is not None], "dressings")
    assign(dress, m_dress)
    smooth(dress, 35)
    kerb = join(kerbs, "kerbs")
    assign(kerb, m_dress)
    objs_flag = join(flags, "flags") if flags else None
    if objs_flag:
        assign(objs_flag, m_face)
    nz1 = noise_mask(0.55, seed=p.seed + 3)
    nz2 = noise_mask(0.25, seed=p.seed + 5, lo=0.3, hi=0.75)
    nzb = noise_mask(1.1, seed=p.seed + 9, lo=0.2, hi=0.8)
    nzs = noise_mask(0.13, seed=p.seed + 7, lo=0.38, hi=0.78)
    nze = noise_mask(0.45, seed=p.seed + 13, lo=0.45, hi=0.75)
    amp_t = 0.0 if mk < 0.03 else min(1.0, 0.4 + 1.5 * mk)
    band_t = 0.2 + 0.42 * mk

    def tide(co, n, band):                     # algae band above the water line with a soft, wandering upper edge
        hgt = co[:, 2] + wl
        b = band * (0.7 + 0.6 * nzb(co, n))
        return np.clip(1.15 * (1.0 - hgt / b), 0.0, 1.0) * (hgt > -0.1)

    def wet(co, n):                            # dark wet line right at the water
        return np.clip(1.0 - (co[:, 2] + wl) / 0.13, 0.0, 1.0)

    def streak(co, n):                         # rain runs below the string course
        c2 = co.copy()
        c2[:, 2] *= 0.1
        return nzs(c2, n) * np.clip(1.0 - (z_corn - co[:, 2]) / 1.7, 0.0, 1.0)

    def rust_run(co, n):                       # dirty rust runs below the iron fixings
        out = np.zeros(len(co))
        for (sx_, sz_) in stains + stains_r:
            dz = sz_ - co[:, 2]
            out = np.maximum(out, np.exp(-((co[:, 0] - sx_) / 0.06) ** 2) * np.clip(1.0 - dz / 1.0, 0.0, 1.0) * (dz > -0.05))
        return out * (co[:, 1] < 0.02)

    def depth_g(co, n):                        # the vault darkens towards the black bay
        return 0.92 * _ss(yl + 0.35, y_dark + 0.05, co[:, 1])

    def s_ext(co, n):
        up = np.clip(n[:, 2], 0.0, 1.0)
        return np.clip(amp_t * tide(co, n, band_t) + 1.5 * mk * up * (0.35 + 0.65 * nz1(co, n)), 0.0, 1.0)

    def g_ext(co, n):
        g_ = np.maximum(wet(co, n), 0.4 * streak(co, n) + 0.16 * nz1(co, n))
        g_ = np.maximum(g_, 0.75 * rust_run(co, n))
        return np.clip(np.maximum(g_, depth_g(co, n) * (co[:, 1] > rd)), 0.0, 1.0)

    def arrays(co):
        if len(co) == len(f_tone):
            return f_tone, f_edge
        return np.full(len(co), 0.5), np.zeros(len(co))

    def s_face(co, n):                         # + moss creeping along the joints of the damp lower courses
        tone_, edge_ = arrays(co)
        low = np.clip(1.0 - (co[:, 2] + wl) / (0.9 + 1.6 * mk), 0.0, 1.0)
        return np.clip(s_ext(co, n) + 1.5 * mk * edge_ * low * (0.4 + 0.6 * nz2(co, n)), 0.0, 1.0)

    def g_face(co, n):                         # + every stone weathers to its own tone
        tone_, edge_ = arrays(co)
        return np.clip(np.maximum(g_ext(co, n), 0.55 * np.clip(0.5 - tone_, 0.0, 1.0)), 0.0, 1.0)

    def w_face(co, n):
        tone_, edge_ = arrays(co)
        return 0.6 * np.clip(tone_ - 0.62, 0.0, 1.0)

    def tread(co, n):
        return 0.6 * nz1(co, n) * np.clip(n[:, 2], 0.0, 1.0) * np.clip(1.0 - np.abs(co[:, 2]) / 0.05, 0.0, 1.0)

    def soot(co, n):                           # above the lantern, drawn a little towards the portal by the draught
        d2 = co[:, 0] ** 2 + 0.55 * (co[:, 1] - yl + 0.12) ** 2
        return np.exp(-d2 / 0.3) * np.clip((co[:, 2] - (H - 0.8)) / 0.45, 0.0, 1.0)

    def seep(co, n):                           # damp running down from the springing and through the haunches
        c2 = np.stack([np.full(len(co), 3.1), co[:, 1], co[:, 2] * 0.07], 1)
        s_ = nzs(c2, n)
        below = np.clip(1.0 - (zs - co[:, 2]) / 1.5, 0.0, 1.0) * (co[:, 2] <= zs + 0.02)
        haunch = np.clip((np.abs(co[:, 0]) - 0.4 * Rv) / (0.55 * Rv), 0.0, 1.0) * (co[:, 2] > zs + 0.02)
        return s_ * np.maximum(below, haunch)

    def g_in(co, n):
        g_ = np.maximum(wet(co, n), 0.95 * seep(co, n) + 0.22 * nz1(co, n))
        g_ = np.maximum(g_, soot(co, n))
        return np.clip(np.maximum(g_, depth_g(co, n)), 0.0, 1.0)

    def s_in(co, n):
        return np.clip(amp_t * tide(co, n, 0.85 * band_t), 0.0, 1.0)

    def w_in(co, n):                           # efflorescence: salts bloom above the damp zone and beside the seeps
        hgt = co[:, 2] + wl
        zone = _ss(band_t * 0.6, band_t + 0.25, hgt) * (1.0 - _ss(band_t + 0.6, band_t + 1.3, hgt))
        return np.clip(0.9 * nze(co, n) * zone + 0.5 * nze(co, n) * seep(co, n), 0.0, 1.0) * (1.0 - depth_g(co, n))

    paint(face, grime=g_face, special=s_face, wear=w_face)
    paint(dress, grime=g_ext, special=lambda co, n: 0.9 * s_ext(co, n), wear=tread)
    paint(kerb, grime=g_ext, wear=tread, special=lambda co, n: 0.7 * s_ext(co, n))
    if objs_flag:
        paint(objs_flag, grime=lambda co, n: np.maximum(0.25, depth_g(co, n)), wear=tread, special=lambda co, n: 0.5 * mk * nz2(co, n))
    for o in lined:
        paint(o, grime=g_in, special=s_in, wear=w_in)
    iron = join(iron, "ironwork")
    paint(iron, special=lambda co, n: np.clip(0.28 + 0.45 * nz2(co, n) + 0.6 * wet(co, n), 0.0, 1.0), grime=0.3)
