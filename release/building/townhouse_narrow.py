"""Tall, narrow rendered merchant's townhouse in pastel limewash. A hardstone plinth runs round all
four sides; the raised ground floor has a stone stoop with wrought handrails up to a panelled door
in a moulded stone case with a lit fanlight, and a cellar bulkhead under the parlour window. Above
are 2-3 storeys of sash windows set in real reveals with hinged shutters, iron wall anchors at every
floor, a jointed stone cornice and a shaped parapet gable (stepped / bell / neck / spout) coped
with individual stone blocks, loft doors flanked by iron date numerals and a long hoist beam with
pulley block, chain and hook. Behind the gable a steep roof of individually laid pantiles oversails
the long walls on open rafter feet; flanks and rear have banded windows, a painted yard door in a
stone case on a threshold, and a tiled lean-to store.
The four wall faces are triangulated meshes that carry the weathering exactly where it belongs:
thin run-off streaks from sill ends, anchor feet and cornice drips, a soft rising-damp band and
faint limewash mottling, with a slight bulge so no arris is razor straight.
Ten shared materials - the whole house is ten draw calls.
"""
from tk import *
import numpy as np
from math import sin, cos, tan, atan2, radians, degrees, pi, hypot, ceil, floor
from mathutils import Matrix, Vector
from mathutils.geometry import delaunay_2d_cdt

META = {"name": "Narrow pastel townhouse", "class": "building", "mount": "ground", "focus": [0.0, 0.0, 2.6]}
PARAMS = {
    "width": {"default": 4.4, "min": 3.5, "max": 5.5, "desc": "frontage width (m); walls span x = -width/2 .. +width/2"},
    "depth": {"default": 8.0, "min": 7.0, "max": 10.0, "desc": "depth of the body from the front wall plane y = 0 (m)"},
    "storeys": {"default": 3, "min": 3, "max": 4, "desc": "storeys including the raised ground floor"},
    "storey_height": {"default": 2.85, "min": 2.7, "max": 3.1, "desc": "upper storey height (m)"},
    "gable": {"default": "bell", "choices": ["stepped", "bell", "neck", "triangular"], "desc": "shape of the street gable"},
    "pitch": {"default": 52.0, "min": 45.0, "max": 57.0, "desc": "roof pitch (deg)"},
    "tint": {"default": "#e3c98a", "desc": "limewash colour of the walls"},
    "trim": {"default": "#cfc6b2", "desc": "stone colour of plinth, cornice, coping, surrounds and steps"},
    "shutters": {"default": "louvred", "choices": ["louvred", "panelled", "none"], "desc": "window shutters on the upper storeys"},
    "shutter_paint": {"default": "#2f5645", "desc": "paint colour of shutters, front door, loft doors, hatch and yard door"},
    "joinery": {"default": "#ebe5d6", "desc": "paint colour of the window joinery, door frame, eaves and hoist beam"},
    "roof_tint": {"default": "#c7714d", "desc": "clay colour of the pantiles"},
    "door_side": {"default": "left", "choices": ["left", "right"], "desc": "bay with the front door and stoop"},
    "lit_ratio": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "fraction of windows with a lit room behind"},
    "hoist": {"default": True, "desc": "hoist beam with pulley block, chain and hook in the gable"},
    "dormer": {"default": True, "desc": "tiled dormer on one roof slope (omitted automatically when the roof is too low for it)"},
    "cellar": {"default": True, "desc": "sloping cellar bulkhead hatch at the foot of the front wall"},
    "date": {"default": "1687", "desc": "year in iron numerals on the gable ('' = none)"},
    "number": {"default": "12", "desc": "house number on the door keystone ('' = none)"},
    "lantern_cd": {"default": 11.0, "min": 0.0, "max": 60.0, "desc": "wall lantern by the door (cd, 0 = no lantern)"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: run-off streaks, damp, patched render, moss on the roof"},
}
VARIANTS = {
    "stepped_rose": {"width": 3.6, "storeys": 4, "gable": "stepped", "tint": "#d9a093", "shutters": "panelled",
                     "shutter_paint": "#5a1f1c", "door_side": "right", "depth": 7.5, "roof_tint": "#a9583f", "dormer": False,
                     "date": "1642", "number": "7", "seed": 4},
    "neck_blue": {"width": 5.3, "storeys": 4, "gable": "neck", "tint": "#a7bfca", "shutters": "none",
                  "shutter_paint": "#1f2f3b", "storey_height": 3.0, "depth": 9.5, "pitch": 50.0, "date": "1731",
                  "number": "31", "lit_ratio": 0.65, "roof_tint": "#6d7480", "seed": 9},
    "spout_mint": {"width": 4.0, "storeys": 3, "gable": "triangular", "tint": "#b4ccae", "shutters": "panelled",
                   "shutter_paint": "#27384f", "depth": 7.0, "pitch": 55.0, "wear": 0.8, "trim": "#bdb39c",
                   "date": "1598", "number": "3", "lit_ratio": 0.35, "seed": 15},
    "bell_wide_cream": {"width": 5.0, "storeys": 3, "gable": "bell", "tint": "#ead9c0", "shutters": "louvred",
                        "shutter_paint": "#7a2f25", "door_side": "right", "depth": 9.0, "storey_height": 3.05,
                        "date": "1704", "number": "18", "lit_ratio": 0.8, "seed": 23},
}

GT = 0.35          # thickness of the street gable wall (m)
RV = 0.11          # depth of the window reveals (m)


# ------------------------------------------------------------------------------------------ helpers
def prism(name, prof, axis, c0, c1, m=None):
    """Closed polygon prof [(u, v)] extruded along 'X' (verts (c, u, v)) or 'Y' (verts (u, c, v))."""
    n = len(prof)

    def pt(c, u, v):
        q = (c, u, v) if axis == 'X' else (u, c, v)
        return tuple(m @ Vector(q)) if m is not None else q

    verts = [pt(c0, u, v) for u, v in prof] + [pt(c1, u, v) for u, v in prof]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def rect(w, h, cx=0.0, cy=0.0):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def frame_m(xl, yl, zl, pos):
    m = Matrix.Identity(4)
    for r in range(3):
        m[r][0], m[r][1], m[r][2], m[r][3] = xl[r], yl[r], zl[r], pos[r]
    return m


def bez2(p0, p1, p2, p3, n=14):
    return [(q[0], q[2]) for q in bezier_path((p0[0], 0, p0[1]), (p1[0], 0, p1[1]), (p2[0], 0, p2[1]), (p3[0], 0, p3[1]), n=n)]


_BF = {'z': (0, 3, 2, 1), 'Z': (4, 5, 6, 7), 'y': (0, 1, 5, 4), 'Y': (2, 3, 7, 6), 'x': (0, 4, 7, 3), 'X': (1, 2, 6, 5)}


class Acc:
    """Collects open boxes / frusta / quads into ONE flat-shaded mesh: only the faces that can be seen
    are built, so joinery, ironwork and masonry cost a handful of triangles each."""

    def __init__(self, m=None):
        self.v, self.f, self.m = [], [], m

    def polys(self, pts, faces, m=None):
        m = m if m is not None else self.m
        used = sorted(set(i for f in faces for i in f))
        re = {i: len(self.v) + k for k, i in enumerate(used)}
        for i in used:
            self.v.append(tuple(m @ Vector(pts[i])) if m is not None else tuple(pts[i]))
        self.f += [tuple(re[i] for i in f) for f in faces]
        return self

    def box(self, size, loc, faces="xXyYzZ", m=None, g=0.0):
        """g > 0: the box is cut g from both ends of its long axis, so the vertices that sit in a junction
        (and bake dark) only shade the last few centimetres instead of the whole length."""
        if g and max(size) > 3 * g:
            ax = list(size).index(max(size))
            L, lo, cf = size[ax], loc[ax] - size[ax] / 2, ("xX", "yY", "zZ")[ax]
            for k, (s0, s1) in enumerate(((0.0, g), (g, L - g), (L - g, L))):
                sz, lc = list(size), list(loc)
                sz[ax], lc[ax] = s1 - s0, lo + (s0 + s1) / 2
                fs = faces.replace(cf[0], "") if k > 0 else faces
                fs = fs.replace(cf[1], "") if k < 2 else fs
                self.box(sz, lc, fs, m)
            return self
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        cx, cy, cz = loc
        pts = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
               (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz), (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
        return self.polys(pts, [_BF[c] for c in faces], m)

    def frustum(self, w0, h0, w1, h1, y0, y1, cx, cz, m=None):
        """Rect (w0 x h0) in the XZ plane at y0 tapering to (w1 x h1) at y1 < y0: face + four flanks."""
        pts = [(cx - w0 / 2, y0, cz - h0 / 2), (cx + w0 / 2, y0, cz - h0 / 2), (cx + w0 / 2, y0, cz + h0 / 2), (cx - w0 / 2, y0, cz + h0 / 2),
               (cx - w1 / 2, y1, cz - h1 / 2), (cx + w1 / 2, y1, cz - h1 / 2), (cx + w1 / 2, y1, cz + h1 / 2), (cx - w1 / 2, y1, cz + h1 / 2)]
        return self.polys(pts, [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], m)

    def stud(self, s, y0, y1, cx, cz, m=None):
        """Forged pyramid nail head of side s on the plane y0, point at y1 < y0."""
        pts = [(cx - s / 2, y0, cz - s / 2), (cx + s / 2, y0, cz - s / 2), (cx + s / 2, y0, cz + s / 2), (cx - s / 2, y0, cz + s / 2), (cx, y1, cz)]
        return self.polys(pts, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], m)

    def obj(self, name):
        return mesh_object(name, self.v, self.f)


def uv_mesh(name, verts, faces, uvs):
    """Mesh with explicit per-vertex UVs that are kept as they are."""
    o = mesh_object(name, verts, faces)
    uvl = o.data.uv_layers.new(name="UVMap")
    for poly in o.data.polygons:
        for li, vi in zip(poly.loop_indices, poly.vertices):
            uvl.data[li].uv = uvs[vi]
    uv_keep(o)
    return o


def pt_in_poly(P, poly):
    P = np.asarray(P, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if abs(y1 - y0) < 1e-12:
            continue
        inside ^= ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
    return inside


def edge_dist(P, poly):
    P = np.asarray(P, dtype=np.float64)
    d = np.full(len(P), 1e9)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        l2 = ex * ex + ey * ey
        if l2 < 1e-14:
            continue
        t = np.clip(((P[:, 0] - ax) * ex + (P[:, 1] - ay) * ey) / l2, 0, 1)
        d = np.minimum(d, np.hypot(P[:, 0] - ax - t * ex, P[:, 1] - ay - t * ey))
    return d


def offset_poly(poly, d):
    """CCW polygon pushed outwards by d (inwards for d < 0), mitred."""
    n = len(poly)
    out = []
    for i in range(n):
        p0, p1, p2 = np.array(poly[i - 1]), np.array(poly[i]), np.array(poly[(i + 1) % n])
        e1, e2 = p1 - p0, p2 - p1
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        n1 = np.array([e1[1], -e1[0]]) / l1 if l1 > 1e-9 else None
        n2 = np.array([e2[1], -e2[0]]) / l2 if l2 > 1e-9 else None
        if n1 is None or n2 is None:
            nn = n1 if n1 is not None else n2
            out.append(tuple(p1 + nn * d) if nn is not None else tuple(p1))
            continue
        b = n1 + n2
        lb = np.linalg.norm(b)
        if lb < 1e-6:
            out.append(tuple(p1 + n1 * d))
        else:
            b /= lb
            out.append(tuple(p1 + b * d / max(0.45, float(b @ n1))))
    return out


def sample_loop(poly, step):
    out = []
    n = len(poly)
    for i in range(n):
        p0, p1 = poly[i], poly[(i + 1) % n]
        k = max(1, int(ceil(hypot(p1[0] - p0[0], p1[1] - p0[1]) / step)))
        for j in range(k):
            out.append((p0[0] + (p1[0] - p0[0]) * j / k, p0[1] + (p1[1] - p0[1]) * j / k))
    return out


class Panel:
    """One wall face as a constrained Delaunay mesh in its own (u, z) plane. Openings are real holes with
    reveals; every part that sits on the wall is ringed with vertices so the baked contact shadow stays
    a thin line; run-off streaks are narrow three-column strips so they stay 3-8 cm wide."""

    def __init__(self, fm, outline, zb):
        self.fm = fm                       # world = fm @ (u, -out, z)
        self.outline = [tuple(q) for q in outline]
        self.zb = zb
        self.holes, self.keeps, self.patches, self.streaks = [], [], [], []
        self.block, self.sbox = [], []

    @staticmethod
    def _bbox(poly):
        us, zs = [q[0] for q in poly], [q[1] for q in poly]
        return (min(us), min(zs), max(us), max(zs))

    def hole(self, poly, depth=RV):
        self.holes.append(([tuple(q) for q in poly], depth))
        self.block.append(self._bbox(poly))

    def keep(self, u0, z0, u1, z1):
        self.keep_poly([(u0, z0), (u1, z0), (u1, z1), (u0, z1)])

    def keep_poly(self, poly):
        self.keeps.append([tuple(q) for q in poly])
        self.block.append(self._bbox(poly))

    def patch(self, poly):
        self.patches.append([tuple(q) for q in poly])
        self.block.append(self._bbox(poly))

    def streak(self, u, zt, hw, ln, s, rng):
        """Run-off streak starting at (u, zt) and running down; it stops above whatever lies below."""
        wd = 0.01
        x0, x1 = u - hw - wd - 0.012, u + hw + wd + 0.012
        zmin = self.zb + 0.06
        for (b0, c0, b1, c1) in self.block + self.sbox:
            if b1 < x0 or b0 > x1:
                continue
            if c0 < zt - 0.004 and c1 > zt + 0.004:
                return False
            if c1 <= zt + 0.004:
                zmin = max(zmin, c1 + 0.03)
        ln = min(ln, zt - zmin)
        if ln < 0.18:
            return False
        corners = np.array([(x0, zt + 0.02), (x1, zt + 0.02), (x0, zt - ln), (x1, zt - ln)])
        if not pt_in_poly(corners, self.outline).all() or edge_dist(corners, self.outline).min() < 0.02:
            return False
        T = [-0.02 / ln, 0.0, 0.08, 0.2, 0.38, 0.62, 1.0]
        UC = [u, u]
        for _ in T[2:]:
            UC.append(clamp(UC[-1] + rng.u(-1, 1) * wd * 0.6, u - wd, u + wd))
        self.streaks.append({"zt": zt, "ln": ln, "hw": hw, "s": s, "T": np.array(T), "UC": np.array(UC)})
        self.sbox.append((x0, zt - ln, x1, zt + 0.02))
        return True

    def build(self, name, wob, rng, grid=0.42, rim=(0.05, 0.17)):
        out = self.outline
        cp, ed = [], []

        def loop(poly, closed=True):
            b = len(cp)
            cp.extend(poly)
            n = len(poly)
            ed.extend((b + i, b + (i + 1) % n) for i in range(n if closed else n - 1))
            return b

        loop(out)
        for poly, _ in self.holes:
            loop(poly)
        for poly in self.patches:
            loop(poly)
        for s in self.streaks:
            nr = len(s["T"])
            cols = [[], [], [], [], []]
            for t, uc in zip(s["T"], s["UC"]):
                z = s["zt"] - t * s["ln"]
                hw = s["hw"] * (1 - 0.45 * max(float(t), 0.0))
                for c, off in enumerate((-hw, -0.5 * hw, 0.0, 0.5 * hw, hw)):
                    cols[c].append((float(uc) + off, z))
            b = [loop(c, closed=False) for c in cols]
            for c in range(4):
                ed.extend([(b[c], b[c + 1]), (b[c] + nr - 1, b[c + 1] + nr - 1)])

        us, zs = [q[0] for q in out], [q[1] for q in out]
        u0, u1, ztop = min(us), max(us), max(zs)
        free = []
        for poly in self.keeps:
            for d in (0.03, 0.1):
                free += sample_loop(offset_poly(poly, d), 0.25)
        for poly in self.patches:
            free += sample_loop(offset_poly(poly, 0.035), 0.25) + sample_loop(offset_poly(poly, -0.035), 0.25)
        for d in rim:
            free += sample_loop(offset_poly(out, -d), 0.3)
        u = u0 + 0.06
        while u < u1:                                                      # contact row just above the plinth cap
            free.append((u, self.zb + 0.056))
            u += 0.2
        for off in (0.12, 0.2, 0.32, 0.46, 0.62, 0.8, 1.0, 1.25):          # fine rows for the rising-damp gradient
            u = u0 + rng.u(0.05, 0.25)
            while u < u1:
                free.append((u, self.zb + off + rng.u(-0.02, 0.02)))
                u += rng.u(0.22, 0.34)
        gridp = []
        z, r = self.zb + 1.25 + grid * 0.8, 0
        while z < ztop:
            u = u0 + grid * (0.5 if r % 2 else 0.2)
            while u < u1:
                gridp.append((u + rng.u(-0.25, 0.25) * grid, z + rng.u(-0.25, 0.25) * grid))
                u += grid
            z += grid * 0.87
            r += 1
        G = np.array(gridp)
        okg = np.ones(len(G), dtype=bool)
        for poly in self.keeps:
            b0, c0, b1, c1 = self._bbox(poly)
            okg &= ~((G[:, 0] > b0 - 0.02) & (G[:, 0] < b1 + 0.02) & (G[:, 1] > c0 - 0.02) & (G[:, 1] < c1 + 0.02))
        F = np.concatenate([np.array(free), G[okg]])
        ok = pt_in_poly(F, out) & (edge_dist(F, out) > 0.025)
        for poly, _ in self.holes:
            ok &= ~pt_in_poly(F, poly) & (edge_dist(F, poly) > 0.012)
        for poly in self.patches:
            ok &= edge_dist(F, poly) > 0.012
        for (b0, c0, b1, c1) in self.sbox:
            ok &= ~((F[:, 0] > b0) & (F[:, 0] < b1) & (F[:, 1] > c0) & (F[:, 1] < c1))
        seen, fk = set(), []
        for q in F[ok]:
            key = (int(round(q[0] / 0.025)), int(round(q[1] / 0.025)))
            if key not in seen:
                seen.add(key)
                fk.append((float(q[0]), float(q[1])))

        res = delaunay_2d_cdt([Vector(q) for q in cp + fk], ed, [], 0, 1e-6)
        V2 = np.array([(v[0], v[1]) for v in res[0]], dtype=np.float64)
        tri = np.array([tuple(f) for f in res[2] if len(f) == 3], dtype=np.int64)
        cen = V2[tri].mean(axis=1)
        okc = pt_in_poly(cen, out)
        for poly, _ in self.holes:
            okc &= ~pt_in_poly(cen, poly)
        tri = tri[okc]
        p0, p1, p2 = V2[tri[:, 0]], V2[tri[:, 1]], V2[tri[:, 2]]
        ar = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        tri[ar < 0] = tri[ar < 0][:, ::-1]
        tri = tri[np.abs(ar) > 1e-9]

        fm = np.array(self.fm, dtype=np.float64)
        S = np.stack([V2[:, 0], np.zeros(len(V2)), V2[:, 1], np.ones(len(V2))], axis=1) @ fm.T
        Pw = wob(S[:, :3])
        inward = fm[:3, :3] @ np.array([0.0, 1.0, 0.0])
        verts = [tuple(q) for q in Pw]
        faces = [tuple(int(i) for i in t) for t in tri]
        for poly, dep in self.holes:
            idx = [int(np.argmin(np.hypot(V2[:, 0] - q[0], V2[:, 1] - q[1]))) for q in poly]
            b = len(verts)
            for i in idx:
                verts.append(tuple(Pw[i] + inward * dep))
            n = len(poly)
            for k in range(n):
                k2 = (k + 1) % n
                faces.append((idx[k], idx[k2], b + k2, b + k))
        o = mesh_object(name, verts, faces)
        smooth(o, 35)
        return o


def gable_outline(style, a, R):
    """Parapet height above the eaves line as a function of x -> [(x, h)] for x = -a .. a.
    Every style stays above the roof plane behind it (cover) so the roof never shows through."""
    def cover(x):
        return R * (1 - x / a) + 0.42

    brk = []
    if style == "stepped":
        tw = 0.42
        ns = max(2, int(round((a - tw) / 0.52)))
        sw = (a - tw) / ns
        rise = R * sw / a

        def f(x):
            if x <= tw:
                return R * (a - tw) / a + 0.47 + rise
            i = min(ns - 1, int(floor((a - x) / sw + 1e-6)))
            return R * (i + 1) * sw / a + 0.47
        brk = [a - i * sw for i in range(1, ns + 1)]
    elif style == "triangular":
        tw = 0.38

        def f(x):
            if x <= tw:
                return R + 1.2
            h = R * (1 - x / a) + 0.55
            return max(h, 0.98) if x > a - 0.34 else h
        brk = [tw, a - 0.34]
    elif style == "bell":
        Ht = R + 1.2
        sh = 0.22
        hs = cover(a - sh)
        c1 = bez2((a - sh, hs), (0.70 * a, hs + 0.02), (0.56 * a, 0.28 * Ht), (0.56 * a, 0.55 * Ht))
        c2 = bez2((0.56 * a, 0.55 * Ht), (0.56 * a, 0.86 * Ht), (0.34 * a, Ht), (0.0, Ht), n=18)
        pl = sorted(c1 + c2)
        xs_, hs_ = [q[0] for q in pl], [q[1] for q in pl]

        def f(x):
            if x > a - sh:
                return hs
            return float(np.interp(x, xs_, hs_))
        brk = [a - sh]
    else:  # neck
        xn = clamp(0.44 * a, 0.8, 1.15)
        hn = R + 0.8
        sh = 0.18
        hs = cover(a - sh)
        c1 = bez2((a - sh, hs), (xn + 0.45 * (a - xn), hs), (xn + 0.06, 0.34 * hn), (xn, 0.7 * hn))
        pl = sorted(c1)
        xs_, hs_ = [q[0] for q in pl], [q[1] for q in pl]

        def f(x):
            if x <= xn:
                return hn + 0.1 + 0.46 * (1 - x / xn)
            if x > a - sh:
                return hs
            return float(np.interp(x, xs_, hs_))
        brk = [xn, a - sh]

    xs = set(round(k * 0.055, 4) for k in range(int(a / 0.055) + 1))
    xs.add(round(a, 4))
    for b in brk:
        xs.add(round(b - 2e-4, 4))
        xs.add(round(b + 2e-4, 4))
    xs = sorted(x for x in xs if 0 <= x <= a + 1e-6)
    half = [(x, max(f(x), cover(x) if style in ("bell", "neck") else 0.0)) for x in xs]
    out = [half[0]]
    for i in range(1, len(half) - 1):          # drop collinear points
        (x0, h0), (x1, h1), (x2, h2) = out[-1], half[i], half[i + 1]
        cr = (x1 - x0) * (h2 - h0) - (h1 - h0) * (x2 - x0)
        if abs(cr) > 1e-5 or hypot(x1 - x0, h1 - h0) > 0.5:
            out.append(half[i])
    out.append(half[-1])
    return [(-x, h) for x, h in reversed(out) if x > 1e-6] + out


def make_tile(wt, lt, amp=0.021, th=0.015, lift=0.022):
    """One clay pantile: trough with a roll on each edge, lower end lifted onto the course below."""
    nx = 4
    verts = []
    for (y, zo) in ((0.0, 0.03 + lift), (lt, 0.03)):
        for j in range(nx + 1):
            u = j / nx
            verts.append(((u - 0.5) * wt, y, -amp * cos(2 * pi * u) + zo))
    for j in range(nx + 1):
        u = j / nx
        verts.append(((u - 0.5) * wt, 0.0, -amp * cos(2 * pi * u) + 0.03 + lift - th))
    faces = []
    for j in range(nx):
        faces.append((j, j + 1, nx + 1 + j + 1, nx + 1 + j))
        faces.append((2 * (nx + 1) + j, 2 * (nx + 1) + j + 1, j + 1, j))
    o = mesh_object("pantile", verts, faces)
    smooth(o, 60)
    return o


def make_shutter(kind, w, h, m):
    """Framed shutter leaf centred on x, standing on z = 0, 32 mm thick around y = 0."""
    st, t = 0.058, 0.032
    A = Acc()
    for sx in (-1, 1):
        A.box((st, t, h), (sx * (w - st) / 2, 0, h / 2))
    for zc, rh in [(0.05, 0.10), (h * 0.5, 0.07), (h - 0.04, 0.08)]:
        A.box((w - 2 * st, t, rh), (0, 0, zc), "yYzZ")
    iw = w - 2 * st
    for z0, z1 in [(0.10, h * 0.5 - 0.035), (h * 0.5 + 0.035, h - 0.08)]:
        if kind == "louvred":
            n = max(3, int((z1 - z0) / 0.056))
            for i in range(n):
                z = z0 + (z1 - z0) * (i + 0.5) / n
                A.box((iw, 0.046, 0.009), (0, 0, 0), "yZz", m=Matrix.Translation((0, 0, z)) @ Matrix.Rotation(radians(38), 4, 'X'))
        else:
            A.box((iw, 0.012, z1 - z0), (0, 0.004, (z0 + z1) / 2), "yY")
            A.frustum(iw - 0.03, z1 - z0 - 0.03, iw - 0.1, z1 - z0 - 0.1, -0.002, -0.015, 0, (z0 + z1) / 2)
    o = A.obj("shutter")
    assign(o, m)
    return o


def make_window(name, w, h, cols, style, mode, M, y0=0.045, uvo=(0.0, 0.0)):
    """Double-hung sash window; origin = bottom centre of the wall opening (w x h) on the wall plane,
    +y into the wall. The frame face sits at y0 (recessed in the reveal; y0 < 0 = planted on a wall).
    style: 'timber' (architrave + drip cap, shuttered windows), 'stone' (stone architrave, keystone,
    bracketed sill), 'band' (plain stone band) or 'plain' (frame and sill only).
    mode : None = dark room, 'open' = lit room, 'drawn' = curtains drawn, 'blind' = blind half down,
    'half' = one curtain drawn."""
    m_join, m_stone, m_lit, m_dim, m_dark = M
    fw, e = 0.055, 0.004
    proud = y0 < 0
    A, S = Acc(), Acc()
    for sx in (-1, 1):
        A.box((fw, 0.06, h + 2 * e), (sx * (w / 2 + e - fw / 2), y0 + 0.03, h / 2), "xXy" if proud else ("yX" if sx < 0 else "yx"))
    A.box((w - 2 * fw + 2 * e, 0.06, fw), (0, y0 + 0.03, h + e - fw / 2), "yzZ" if proud else "yz")
    A.box((w - 2 * fw + 2 * e, 0.06, fw), (0, y0 + 0.03, fw / 2 - e), "yZ")
    iw, z_lo, z_hi = w - 2 * fw, fw - e, h - fw + e
    zm = (z_lo + z_hi) / 2 + 0.02 * (h > 1.5)

    def sash(z0, z1, yf, br):
        sw_ = 0.036
        for sx in (-1, 1):
            A.box((sw_, 0.022, z1 - z0), (sx * (iw / 2 - sw_ / 2), yf + 0.011, (z0 + z1) / 2), "yX" if sx < 0 else "yx")
        A.box((iw - 2 * sw_, 0.022, 0.04), (0, yf + 0.011, z1 - 0.02), "yz")
        A.box((iw - 2 * sw_, 0.022, br), (0, yf + 0.011, z0 + br / 2), "yZz")
        gx = iw / 2 - sw_
        g0, g1 = z0 + br, z1 - 0.04
        for i in range(1, cols):
            A.box((0.018, 0.016, g1 - g0), (-gx + 2 * gx * i / cols, yf + 0.011, (g0 + g1) / 2), "yxX")
        A.box((2 * gx, 0.016, 0.018), (0, yf + 0.011, (g0 + g1) / 2), "yzZ")
        return (-gx, gx, g0, g1, yf + 0.017)

    panes = [sash(zm - 0.02, z_hi, y0 + 0.012, 0.036), sash(z_lo, zm + 0.016, y0 + 0.034, 0.07)]
    quads = {}

    def pane(mt, x0, x1, g0, g1, yg):
        vs, fs, uvs = quads.setdefault(mt, ([], [], []))
        b = len(vs)
        for (x, z) in ((x0, g0), (x1, g0), (x1, g1), (x0, g1)):
            vs.append((x, yg, z))
            uvs.append(((x + iw / 2) / iw * 0.96 + 0.02 + uvo[0], z / h * 0.96 + 0.02 + uvo[1]))
        fs.append((b, b + 1, b + 2, b + 3))

    for k, (x0, x1, g0, g1, yg) in enumerate(panes):       # k = 0: upper sash
        if mode is None:
            pane(m_dark, x0, x1, g0, g1, yg)
        elif mode == "open":
            pane(m_lit, x0, x1, g0, g1, yg)
        elif mode == "drawn":
            pane(m_dim, x0, x1, g0, g1, yg)
        elif mode == "blind":
            pane(m_dim if k == 0 else m_lit, x0, x1, g0, g1, yg)
        else:
            xs = x0 + (x1 - x0) * 0.44
            pane(m_dim, x0, xs, g0, g1, yg)
            pane(m_lit, xs, x1, g0, g1, yg)
    parts = []
    for mt, (vs, fs, uvs) in quads.items():
        g = uv_mesh("wglass", vs, fs, uvs)
        assign(g, mt)
        parts.append(g)

    stone = []
    sill_w = w + (0.34 if style == "stone" else 0.14)
    yb = max(y0 + 0.008, 0.03)
    stone.append(prism("wsill", [(yb, -0.085), (-0.105, -0.085), (-0.115, -0.075), (-0.115, -0.035), (-0.06, 0.0), (yb, 0.0 if proud else 0.012)],
                       'X', -sill_w / 2, sill_w / 2))
    wood = []
    if style == "timber":
        for sx in (-1, 1):
            A.box((0.07, 0.052, h + 0.07), (sx * (w / 2 + 0.031), 0.004, h / 2 + 0.035), "xXy", g=0.06)
        A.box((w + 0.132, 0.052, 0.07), (0, 0.004, h + 0.031), "yz")
        wood.append(prism("wcap", [(0.03, h + 0.066), (-0.05, h + 0.066), (-0.075, h + 0.09), (-0.085, h + 0.09), (-0.085, h + 0.105), (0.03, h + 0.125)],
                          'X', -w / 2 - 0.085, w / 2 + 0.085))
    elif style == "stone":
        for sx in (-1, 1):
            S.box((0.13, 0.07, h + 0.02), (sx * (w / 2 + 0.062), -0.005, h / 2 + 0.01), "xXy", g=0.06)
        S.box((w + 0.254, 0.07, 0.13), (0, -0.005, h + 0.062), "xXyz")
        stone.append(prism("wkey", [(-0.06, h - 0.01), (0.06, h - 0.01), (0.09, h + 0.2), (-0.09, h + 0.2)], 'Y', -0.075, 0.03))
        stone.append(prism("whead", [(0.03, h + 0.127), (-0.045, h + 0.127), (-0.06, h + 0.15), (-0.09, h + 0.165), (-0.09, h + 0.185), (0.03, h + 0.2)],
                           'X', -w / 2 - 0.16, w / 2 + 0.16))
        for sx in (-1, 1):
            stone.append(prism("wbrk", [(0.03, -0.085), (-0.09, -0.085), (-0.07, -0.13), (-0.03, -0.2), (0.03, -0.22)], 'X',
                               sx * (w / 2 + 0.04) - 0.045, sx * (w / 2 + 0.04) + 0.045))
    elif style == "band":
        for sx in (-1, 1):
            S.box((0.09, 0.055, h + 0.01), (sx * (w / 2 + 0.041), 0.0025, h / 2 + 0.005), "xXy", g=0.06)
        S.box((w + 0.172, 0.06, 0.11), (0, 0.0, h + 0.051), "xXyzZ")
    wood.append(A.obj("wframe"))
    if S.v:
        stone.append(S.obj("wband"))
    for o in wood:
        assign(o, m_join)
    for o in stone:
        assign(o, m_stone)
    return join(wood, name), join(stone, name + "_stone"), join(parts, name + "_glass")


# -------------------------------------------------------------------------------------------- build
def build(p, rng):
    W, D = float(p.width), float(p.depth)
    a = W / 2
    N = int(p.storeys)
    wear = float(p.wear)
    sd = int(p.seed)
    pitch = radians(p.pitch)
    tp, cp_, sp_ = tan(pitch), cos(pitch), sin(pitch)
    steps = 4
    rise = 0.17
    zp = steps * rise                       # plinth / raised ground floor level
    hg = p.storey_height + 0.25             # ground storey is the tall one
    hu = p.storey_height
    floors = [zp] + [zp + hg + i * hu for i in range(N - 1)]     # floor level of each storey
    z0 = zp + hg + (N - 1) * hu             # eaves line / base of the gable
    R = a * tp                              # roof rise
    ov = 0.24                               # eaves overhang
    xe = a + ov
    ze = z0 + 0.10 - ov * tp                # top of the roof deck at the eaves edge
    zr = z0 + 0.10 + R                      # top of the roof deck at the ridge
    door_l = p.door_side == "left"

    # ---- the ten materials
    m_wall = mat("plaster_lime", p.tint)
    m_stone = mat("stone_smooth", p.trim, grime=0.6)
    m_tile = mat("terracotta", p.roof_tint)
    m_shut = mat("wood_painted", p.shutter_paint)
    m_join = mat("wood_painted", p.joinery)
    m_iron = mat("iron_wrought")
    m_lit = mat("window_lit", "#ffc37a", glow=2.0, curtain=0.25, grime=0.0)      # grime gain 0: the baked cavity dirt must not blotch the panes
    m_dim = mat("window_lit", "#f0a25c", glow=1.0, curtain=0.95, grime=0.0)
    m_dark = mat("window_dark", grime=0.0)
    m_lglass = mat("glass_lantern", "#ffa94e", glow=1.4, opacity=0.32)      # warm lit panes (same recipe as wall_lantern), not a white card
    MW = (m_join, m_stone, m_lit, m_dim, m_dark)

    nz_b = noise_mask(2.3, seed=sd + 2, lo=0.25, hi=0.8)       # broad limewash mottling
    nz_m = noise_mask(0.8, seed=sd + 3, lo=0.3, hi=0.75)
    nz_d = noise_mask(1.1, seed=sd + 5, lo=0.2, hi=0.8)        # height of the rising damp
    nz_w = noise_mask(2.0, seed=sd + 6, lo=0.15, hi=0.85)      # wall bulge
    nz_f = noise_mask(0.25, seed=sd + 7, lo=0.3, hi=0.75)      # fine: moss, rust
    nz_c = noise_mask(0.9, seed=sd + 8, lo=0.74, hi=0.9)       # sparse chipped render
    nz_p = noise_mask(0.15, seed=sd + 9, lo=0.55, hi=0.85)     # paint chips

    def stone_w(o, g=0.1, moss=0.5):
        for q in (o if isinstance(o, (list, tuple)) else [o]):
            assign(q, m_stone)
            paint(q, grime=lambda co, n, g=g: np.clip(g + 0.12 * nz_m(co, n), 0, 1) * (0.6 + 0.6 * wear),
                  special=mask_mul(facing((0, 0, 1)), nz_f, moss * (0.3 + wear)))
        return o

    def wood_w(o, m, g=0.15, chip=0.4, fade=0.3):
        assign(o, m)
        paint(o, grime=g, wear=mask_mul(nz_p, chip * (0.4 + 0.8 * wear)), special=fade * (0.4 + 0.6 * wear))
        return o

    def iron_w(o, rust=0.22, bright=0.0):
        for q in (o if isinstance(o, (list, tuple)) else [o]):
            assign(q, m_iron)
            paint(q, special=mask_mul(nz_f, rust + 0.5 * wear), wear=bright)
        return o

    # ---- wall faces: face matrix maps (u along the wall, -y outwards, z up) to the world
    FM = {"front": Matrix.Identity(4),
          "back": Matrix.Translation((0, D, 0)) @ Matrix.Rotation(pi, 4, 'Z'),
          "right": Matrix.Translation((a, 0, 0)) @ Matrix.Rotation(pi / 2, 4, 'Z'),
          "left": Matrix.Translation((-a, 0, 0)) @ Matrix.Rotation(-pi / 2, 4, 'Z')}

    def wob(S):
        """Walls bulge up to ~2 cm inwards so no face is dead flat and no corner razor straight."""
        S = np.asarray(S, dtype=np.float64)
        w = nz_w(S.astype(np.float32), None).astype(np.float64)
        out = S.copy()
        out[:, 0] -= np.sign(S[:, 0]) * (np.abs(np.abs(S[:, 0]) - a) < 2e-4) * 0.022 * w
        out[:, 1] += (np.abs(S[:, 1]) < 2e-4) * 0.012 * w
        out[:, 1] -= (np.abs(S[:, 1] - D) < 2e-4) * 0.022 * w
        return out

    def band(name, prof, face, u0, u1):
        """Moulding with section prof [(out, z)] run along a wall face from u0 to u1."""
        return prism(name, [(-o_, z) for o_, z in prof], 'X', u0, u1, m=FM[face])

    # ---- layout of the street front
    nbay = 2 if W < 4.95 else 3
    mg = 0.2
    bw = (W - 2 * mg) / nbay
    bays = [-a + mg + bw * (i + 0.5) for i in range(nbay)]
    door_bay = 0 if door_l else nbay - 1
    nx_bay = bays[1 if door_bay == 0 else nbay - 2]
    dx = bays[door_bay]
    shut = p.shutters != "none"
    ww = clamp((bw - 0.22) / 2, 0.6, 1.05) if shut else clamp(bw - 0.78, 0.72, 1.15)
    wh = clamp(hu - 1.2, 1.4, 1.95)
    wgw, wgh = clamp(bw - 0.68, 0.8, 1.15), clamp(hg - 1.3, 1.6, 2.1)
    c3 = 3 if ww >= 0.78 else 2
    ust = "timber" if shut else "stone"
    kinds = {"upper": (ww, wh, c3, ust), "top": (ww, wh * 0.86, c3, ust), "ground": (wgw, wgh, 3, "stone"),
             "side": (0.8, 1.45, 2, "band"), "side_top": (0.8, 1.25, 2, "band"), "attic": (0.62, 0.95, 2, "band"),
             "dormer": (0.62, 0.95, 2, "plain")}
    ow = clamp(bw - 0.48, 0.98, 1.14)                        # clear width of the door opening
    lh_ = clamp(hg - 0.9 - ow / 2, 1.95, 2.15)               # leaf height
    spr = lh_ + 0.08                                         # springing line above the threshold
    Hd = spr + ow / 2
    key_top = min(Hd + 0.3, hg - 0.1)
    # rear: yard door in one bay, lean-to store in the other
    rw_ = rng.fork("sidewin")
    bxs = [-a * 0.46, a * 0.46]
    bdoor = rw_.pick([0, 1])
    ybx, lnx = bxs[bdoor], bxs[1 - bdoor]
    ydw, ydh, zth = 0.88, 1.98, 0.17
    ydt = zth + ydh
    lw_ = clamp(0.62 * a, 1.0, 1.5)
    ld, lz1, lz0 = 0.56, 2.3, 1.98
    yd = D - 0.55                                            # downpipes on the flanks
    hop_z = ze - 0.62

    # ---- gable geometry
    pts = gable_outline(p.gable, a, R)
    htop = max(h for _, h in pts)
    hend = pts[-1][1]

    def halfw(h):
        xs_ = [abs(x) for x, hh in pts if hh >= h]
        return max(xs_) if xs_ else 0.0

    lw = 0.78
    lh = clamp(0.5 * R, 1.0, 1.35)
    lzz = z0 + 0.36
    zb = lzz + lh + 0.36                                     # hoist beam
    Ht = R + 1.2
    bandz = z0 + max(0.56 * Ht, zb - z0 + 0.27)              # bell gable: waist band and oculus
    ocz = max(z0 + 0.745 * Ht, bandz + 0.42)
    oculus = p.gable == "bell" and ocz + 0.3 < z0 + Ht - 0.22 and halfw(ocz - z0 + 0.27) > 0.42

    # ---- the four wall panels
    zc = [zp + o_ for o_ in (0.0, 0.11, 0.2, 0.32, 0.46, 0.62, 0.8, 1.0, 1.25)]
    nzc = int(ceil((z0 - zc[-1]) / 0.5))
    zc += [zc[-1] + (z0 - zc[-1]) * i / nzc for i in range(1, nzc + 1)]
    zc[-1] = z0

    def edge_up(u):
        return [(u, z) for z in zc]

    def edge_dn(u):
        return [(u, z) for z in reversed(zc)][:-1]

    notch = [(dx + x, zp + z) for x, z in arch_pts(ow + 0.06, Hd + 0.03, spring=spr, kind="round", n=10)][::-1]
    PN = {"front": Panel(FM["front"], [(-a, zp)] + notch + edge_up(a) + [(x, z0 + h) for x, h in reversed(pts)] + edge_dn(-a), zp),
          "back": Panel(FM["back"], [(-a, zp), (-ybx - ydw / 2, zp), (-ybx - ydw / 2, ydt), (-ybx + ydw / 2, ydt), (-ybx + ydw / 2, zp)]
                        + edge_up(a) + [(0.0, z0 + R)] + edge_dn(-a), zp),
          "right": Panel(FM["right"], [(0.0, zp)] + edge_up(D) + edge_dn(0.0), zp),
          "left": Panel(FM["left"], [(-D, zp)] + edge_up(0.0) + edge_dn(-D), zp)}
    wins, sills = {}, []
    rl = rng.fork("lit")
    lights = {"front": 2, "back": 1, "side": 1}
    nlight = [0]

    def pick_mode(lit, hero=False):
        if not lit:
            return None
        r = rl.u(0, 1)
        if hero:
            return "open" if r < 0.55 else "half"
        return "open" if r < 0.34 else "blind" if r < 0.56 else "half" if r < 0.8 else "drawn"

    def put(kind, face, c, z, mode, light_ok=True, shutters=False):
        w_, h_, cols, style = kinds[kind]
        u = c if face in ("front", "right") else -c
        wins.setdefault((kind, mode), []).append(FM[face] @ Matrix.Translation((u, 0, z)))
        pn = PN[face]
        pn.hole([(u - w_ / 2, z), (u + w_ / 2, z), (u + w_ / 2, z + h_), (u - w_ / 2, z + h_)])
        sw2 = (w_ + (0.34 if style == "stone" else 0.14)) / 2
        top = {"stone": 0.21, "timber": 0.13, "band": 0.115}.get(style, 0.0)
        kx = (w_ + 0.065) if shutters else max(sw2, w_ / 2 + 0.092)
        pn.keep(u - kx, z - 0.092, u + kx, z + h_ + top)
        if style == "stone":
            pn.keep(u - w_ / 2 - 0.1, z - 0.225, u + w_ / 2 + 0.1, z - 0.093)
        sills.append((face, u, z - 0.098, kx if shutters else sw2 + 0.004))
        grp = face if face in ("front", "back") else "side"
        if mode and light_ok and lights[grp] > 0:
            lights[grp] -= 1
            nlight[0] += 1
            lp = FM[face] @ Vector((u, 0.36, z + h_ * 0.5))
            light(f"window{nlight[0]}", tuple(lp), color="#ffc98a", intensity=5.0 if kind == "ground" else 3.4, range=5.5, flicker=0.04)

    # front: ground floor (the parlour is usually lit), upper storeys
    for i, bx in enumerate(bays):
        if i != door_bay:
            lit = (rl.chance(0.5 + 0.9 * p.lit_ratio) if bx == nx_bay else rl.chance(p.lit_ratio)) and p.lit_ratio > 0.01
            put("ground", "front", bx, zp + 0.95, pick_mode(lit, hero=True))
    nup = (N - 1) * nbay
    order = sorted(range(nup), key=lambda i: rl.u(0, 1))
    nlit = int(round(p.lit_ratio * nup))
    lit_it = iter([order.index(i) < nlit for i in range(nup)])
    sh_m, hinge_m, dog_m = {}, [], []
    rs = rng.fork("shutters")
    for si in range(1, N):
        kind = "top" if si == N - 1 else "upper"
        zs = floors[si] + 0.85
        h = kinds[kind][1]
        for bi, bx in enumerate(bays):
            put(kind, "front", bx, zs, pick_mode(next(lit_it)), light_ok=(si == 1), shutters=shut)
            if si == 1:
                socket(f"planter_{bi}", "attach", (bx, -0.11, zs - 0.09), role="planter")
            if shut:
                hx = ww / 2 + 0.05
                swd = ww / 2
                for sx in (1, -1):
                    ang = rs.u(0.5, 5.0) if not rs.chance(0.12) else rs.u(8, 18)
                    rot = Matrix.Rotation(radians(-sx * ang), 4, 'Z')
                    sh_m.setdefault(kind, []).append(Matrix.Translation((bx + sx * hx, -0.038, zs + rs.u(-0.004, 0.004))) @ rot @ Matrix.Translation((sx * swd / 2, 0, 0)))
                    for hz in (0.2, h - 0.2):
                        hinge_m.append(Matrix.Translation((bx + sx * hx, -0.058, zs + hz)) @ rot @ Matrix.Rotation(0 if sx > 0 else pi, 4, 'Z'))
                    dog_m.append(trs((bx + sx * (hx + swd - 0.06), 0.0, zs - 0.06), (0, rs.u(-20, 20), 0)))
    # flanks: banded windows on the upper storeys, a few lit
    for face in ("right", "left"):
        for si in range(1, N):
            kind = "side_top" if si == N - 1 else "side"
            put(kind, face, rw_.u(1.5, D * 0.5 - 0.6), floors[si] + 0.95, pick_mode(rl.chance(p.lit_ratio * 0.8)))
            if rw_.chance(0.55):
                put(kind, face, rw_.u(D * 0.5 + 0.6, D - 1.6), floors[si] + 0.95, pick_mode(rl.chance(p.lit_ratio * 0.8)))
    for si in range(1, N):
        for bx in bxs:
            if rw_.chance(0.12):
                continue
            put("side", "back", bx, floors[si] + 0.95, pick_mode(rl.chance(p.lit_ratio * 0.9)))
    if R > 2.0:
        put("attic", "back", 0.0, z0 + 0.3, pick_mode(rl.chance(p.lit_ratio * 0.6)), light_ok=False)

    # parts that sit on the walls: contact rings + stops for the run-off
    pf, pb = PN["front"], PN["back"]
    pf.keep(-a - 0.1, z0 - 0.205, a + 0.1, z0 + 0.15)                               # cornice
    pf.keep(-a - 0.1, floors[1] - 0.075, a + 0.1, floors[1] + 0.065)                # string course
    pf.keep_poly([(dx + x, zp + z) for x, z in arch_pts(ow + 0.44, Hd + 0.22, spring=spr, kind="round", n=10)])
    pf.keep(dx - 0.16, zp + Hd + 0.1, dx + 0.16, zp + key_top + 0.005)
    pf.hole([(-lw / 2, lzz), (lw / 2, lzz), (lw / 2, lzz + lh), (-lw / 2, lzz + lh)], 0.1)
    pf.keep(-lw / 2 - 0.16, lzz - 0.095, lw / 2 + 0.16, lzz + lh + 0.115)
    if p.hoist:
        pf.keep(-0.16, zb - 0.18, 0.16, zb + 0.18)
    if p.gable == "bell":
        hwb = halfw(bandz - z0) + 0.05
        pf.keep(-hwb, bandz - 0.055, hwb, bandz + 0.075)
        if oculus:
            pf.hole([(0.165 * cos(radians(t)), ocz + 0.165 * sin(radians(t))) for t in range(0, 360, 30)], 0.08)
            pf.keep(-0.26, ocz - 0.26, 0.26, ocz + 0.26)
    if p.gable in ("neck", "bell"):
        for sx in (-1, 1):
            pf.keep(sx * (a - 0.5) - 0.34, z0 + 0.17, sx * (a - 0.5) + 0.34, z0 + 0.86)
    lx = (dx + nx_bay) / 2                                   # lantern bracket on the bay line just above the string course:
    lz = floors[1] + 0.4                                     # the flame hangs above the parlour glass, so no pane mirrors it
    lan_on = p.lantern_cd > 0.5
    if p.lantern_cd > 0.5:
        pf.keep(lx - 0.045, lz - 0.34, lx + 0.045, lz + 0.14)
    for sx, face in ((1, "right"), (-1, "left")):
        PN[face].keep(sx * yd - 0.09, zp, sx * yd + 0.09, hop_z + 0.3)
    pb.keep(-ybx - ydw / 2 - 0.14, zp, -ybx + ydw / 2 + 0.14, ydt + 0.2)            # yard door case
    pb.keep(-lnx - lw_ / 2 - 0.02, zp, -lnx + lw_ / 2 + 0.02, lz1 + 0.2)            # lean-to

    # patched render on the flanks and rear
    rpt = rng.fork("patch")
    for face, (pu0, pu1) in (("right", (0.8, D - 1.2)), ("left", (-D + 1.2, -0.8)), ("back", (-a + 0.5, a - 0.5))):
        for _ in range(2 if face != "back" else 1):
            for _try in range(8):
                pw_, ph_ = rpt.u(0.8, 1.5), rpt.u(0.6, 1.2)
                cu, cz_ = rpt.u(pu0 + pw_ / 2, pu1 - pw_ / 2), rpt.u(zp + 0.9, z0 - 1.0)
                bb = (cu - pw_ / 2 - 0.08, cz_ - ph_ / 2 - 0.08, cu + pw_ / 2 + 0.08, cz_ + ph_ / 2 + 0.08)
                if any(not (bb[2] < b0 or bb[0] > b1 or bb[3] < c0 or bb[1] > c1) for (b0, c0, b1, c1) in PN[face].block):
                    continue
                n_ = 12                                      # squarish trowelled repair with a wandering edge

                def sq(v):
                    return (1.0 if v > 0 else -1.0) * abs(v) ** 0.45

                PN[face].patch([(cu + pw_ / 2 * sq(cos(2 * pi * (k + 0.5) / n_)) * rpt.u(0.82, 1.0), cz_ + ph_ / 2 * sq(sin(2 * pi * (k + 0.5) / n_)) * rpt.u(0.82, 1.0))
                                for k in range(n_)])
                break

    # run-off streaks: sill ends, anchor feet, cornice / string course drips
    rk_ = rng.fork("streaks")
    sgain = 0.42 + 0.3 * wear
    for (face, u, zt, sw2) in sills:
        for sx in (-1, 1):
            if rk_.chance(0.85):
                PN[face].streak(u + sx * (sw2 - 0.012), zt, rk_.u(0.026, 0.042), rk_.u(0.45, 1.25), sgain * rk_.u(0.6, 1.0), rk_)
    axs = [-a + 0.13, a - 0.13] + [(bays[i] + bays[i + 1]) / 2 for i in range(nbay - 1)]
    anc_front = [(x, floors[si] + 0.33) for si in range(1, N) for x in axs if not (lan_on and si == 1 and abs(x - lx) < 0.2)] + [(x, z0 - 0.5) for x in axs]
    for (x, z) in anc_front:
        pf.streak(x - 0.012, z - 0.25, rk_.u(0.022, 0.032), rk_.u(0.4, 1.0), sgain * rk_.u(0.7, 1.1), rk_)
    for _ in range(nbay + 2):
        pf.streak(rk_.u(-a + 0.3, a - 0.3), z0 - 0.21, rk_.u(0.025, 0.045), rk_.u(0.35, 1.1), sgain * rk_.u(0.45, 0.8), rk_)
    for _ in range(nbay + 1):
        pf.streak(rk_.u(-a + 0.3, a - 0.3), floors[1] - 0.08, rk_.u(0.022, 0.036), rk_.u(0.3, 0.9), sgain * rk_.u(0.4, 0.75), rk_)
    for sx in (-1, 1):
        pf.streak(sx * (lw / 2 + 0.14), lzz - 0.1, rk_.u(0.022, 0.034), rk_.u(0.2, 0.4), sgain * 0.8, rk_)
        if p.gable == "bell":
            pf.streak(sx * (hwb - 0.04), bandz - 0.06, 0.022, rk_.u(0.4, 0.9), sgain * 0.8, rk_)
    anc_side = []
    ra = rng.fork("anchors")
    for sx, face in ((1, "right"), (-1, "left")):
        for si in range(1, N):
            for yy in (D * 0.3 + ra.u(-0.4, 0.4), D * 0.7 + ra.u(-0.4, 0.4)):
                zz = floors[si] + 0.26
                if any(not (sx * yy + 0.1 < b0 or sx * yy - 0.1 > b1 or zz + 0.3 < c0 or zz - 0.3 > c1) for (b0, c0, b1, c1) in PN[face].block):
                    continue
                anc_side.append((sx, yy, zz))
                PN[face].keep(sx * yy - 0.03, zz - 0.27, sx * yy + 0.03, zz + 0.27)
                PN[face].streak(sx * yy, zz - 0.27, rk_.u(0.022, 0.032), rk_.u(0.5, 1.3), sgain * rk_.u(0.5, 0.85), rk_)
        PN[face].streak(sx * yd + sx * 0.13, hop_z - rk_.u(0.3, 1.2), 0.03, rk_.u(0.8, 1.6), sgain * 0.8, rk_)      # leaking collar
    for _ in range(3):
        pb.streak(rk_.u(-a + 0.3, a - 0.3), z0 + rk_.u(-1.0, 0.2), rk_.u(0.025, 0.04), rk_.u(0.6, 1.4), sgain * rk_.u(0.4, 0.7), rk_)

    def paint_wall(pn, o, dirt=1.0):
        inv = np.array(pn.fm.inverted(), dtype=np.float64)

        def uz(co):
            L = np.asarray(co, dtype=np.float64) @ inv[:3, :3].T + inv[:3, 3]
            return L[:, 0], L[:, 2]

        def damp(co, n, z):
            flat = np.array(co, dtype=np.float32)
            flat[:, 2] = 0.0
            hd = 0.5 + 0.4 * nz_d(flat, n)
            return np.clip(1.0 - (z - zp) / hd, 0, 1) ** 1.1

        def patch(u, z):
            m = np.zeros(len(u))
            P = np.stack([u, z], axis=1)
            for poly in pn.patches:
                m = np.maximum(m, (pt_in_poly(P, poly) | (edge_dist(P, poly) < 3e-4)).astype(np.float64))
            return m

        def grime(co, n):
            u, z = uz(co)
            g = (0.1 * nz_b(co, n) + 0.05 * nz_m(co, n)) * dirt + 0.3 * damp(co, n, z)
            g = g * (0.55 + 0.6 * wear)
            for s in pn.streaks:
                t = (s["zt"] - z) / s["ln"]
                okk = (t >= s["T"][0]) & (t <= 1.0)
                if not okk.any():
                    continue
                uc = np.interp(t, s["T"], s["UC"])
                hw = s["hw"] * (1 - 0.45 * np.clip(t, 0, 1))
                fall = np.where(t < 0, 1 - t / s["T"][0], (1 - np.clip(t, 0, 1)) ** 0.85)
                prof = np.interp(np.abs(u - uc) / hw, [0.0, 0.5, 1.0], [1.0, 0.85, 0.0])
                g = np.maximum(g, okk * s["s"] * fall * prof + g * 0.5)
            return np.clip(g * (1 - 0.6 * patch(u, z)) + 0.03 * patch(u, z), 0, 1)

        def special(co, n):
            u, z = uz(co)
            return np.zeros(len(u))

        def chip(co, n):
            u, z = uz(co)
            return np.clip(0.2 * patch(u, z) + nz_c(co, n) * 0.5 * wear * np.clip(1 - (z - zp) / 3.5, 0, 1), 0, 1)

        assign(o, m_wall)
        paint(o, grime=grime, special=special, wear=chip)

    rgrid = rng.fork("wallmesh")
    for face in ("front", "right", "back", "left"):
        wo_ = PN[face].build(f"wall_{face}", wob, rgrid)
        paint_wall(PN[face], wo_, 1.0 if face == "front" else 1.3)

    # back of the street gable above the roof, its end faces and the risers of a stepped outline
    gb = Acc()

    def wy(x, z):
        return tuple(wob(np.array([(x, 0.0, z)]))[0])

    for i in range(len(pts) - 1):
        (xa, ha), (xb, hb) = pts[i], pts[i + 1]
        if xb - xa < 2e-3:
            if abs(hb - ha) > 0.01:
                zl, zh = z0 + min(ha, hb) - 0.02, z0 + max(ha, hb)
                q = [wy(xa, zl), (xa, GT, zl), (xa, GT, zh), wy(xa, zh)]
                gb.polys(q if xa < 0 else q[::-1], [(0, 1, 2, 3)])
            continue
        za, zb_ = z0 + ha, z0 + hb
        ma = min(za - 0.02, z0 + 0.1 + R * (1 - abs(xa) / a) + 0.22)
        mb = min(zb_ - 0.02, z0 + 0.1 + R * (1 - abs(xb) / a) + 0.22)
        gb.polys([(xb, GT, z0), (xa, GT, z0), (xa, GT, ma), (xb, GT, mb), (xa, GT, za), (xb, GT, zb_)], [(0, 1, 2, 3), (3, 2, 4, 5)])
    gb.polys([wy(a, z0), (a, GT, z0), (a, GT, z0 + hend), wy(a, z0 + hend)], [(0, 1, 2, 3)])
    gb.polys([(-a, GT, z0), wy(-a, z0), wy(-a, z0 + hend), (-a, GT, z0 + hend)], [(0, 1, 2, 3)])
    gbo = gb.obj("gable_back")
    assign(gbo, m_wall)
    paint(gbo, grime=lambda co, n: np.clip(0.12 + 0.1 * nz_b(co, n), 0, 1) * (0.55 + 0.6 * wear))

    # ---- hardstone plinth round all four sides: coursed blocks under a moulded cap
    pl = Acc()
    pl.box((W, D, zp), (0, D / 2, zp / 2), "xXy")
    for (xa, xb) in ((-a, ybx - ydw / 2), (ybx + ydw / 2, a)):
        pl.polys([(xb, D, 0), (xa, D, 0), (xa, D, zp), (xb, D, zp)], [(0, 1, 2, 3)])
    core = pl.obj("plinth_core")
    assign(core, m_stone)
    paint(core, grime=0.75)
    rp = rng.fork("plinth")
    chh = (zp - 0.05) / 2
    pbk = Acc()

    def plinth_run(face, ua, ub, out_):
        for ci in range(2):
            x, k = ua, 0
            while x < ub - 1e-3:
                ln = rp.u(0.55, 0.95) if not (ci == 1 and k == 0) else rp.u(0.3, 0.45)
                if ub - (x + ln) < 0.3:
                    ln = ub - x
                pbk.frustum(ln - 0.008, chh - 0.008, ln - 0.036, chh - 0.036, 0.0, -out_ - rp.u(-0.004, 0.004), x + ln / 2, chh * (ci + 0.5), m=FM[face])
                x += ln
                k += 1

    cap_f = [(-0.03, zp - 0.05), (0.115, zp - 0.05), (0.115, zp - 0.022), (0.095, zp), (0.04, zp + 0.035), (-0.03, zp + 0.04)]
    cap_s = [(-0.03, zp - 0.05), (0.075, zp - 0.05), (0.075, zp - 0.022), (0.06, zp), (0.03, zp + 0.035), (-0.03, zp + 0.04)]
    caps = []
    plinth_run("front", -a - 0.045, a + 0.045, 0.07)
    for (ua, ub) in ((-a - 0.115, dx - ow / 2 - 0.2), (dx + ow / 2 + 0.2, a + 0.115)):
        caps.append(band("plinth_cap", cap_f, "front", ua, ub))
    plinth_run("right", 0.0, D, 0.045)
    plinth_run("left", -D, 0.0, 0.045)
    caps.append(band("plinth_cap", cap_s, "right", 0.0, D))
    caps.append(band("plinth_cap", cap_s, "left", -D, 0.0))
    segs = [(-a - 0.045, a + 0.045)]
    for (ca, cb) in ((-ybx - ydw / 2 - 0.13, -ybx + ydw / 2 + 0.13), (-lnx - lw_ / 2 + 0.03, -lnx + lw_ / 2 - 0.03)):
        segs = [q for (sa, sb) in segs for q in ((sa, min(sb, ca)), (max(sa, cb), sb)) if q[1] - q[0] > 0.12]
    for (ua, ub) in segs:
        plinth_run("back", ua, ub, 0.045)
    for (ua, ub) in ((-a - 0.075, -ybx - ydw / 2 - 0.13), (-ybx + ydw / 2 + 0.13, a + 0.075)):
        caps.append(band("plinth_cap", cap_s, "back", ua, ub))
    pbo = pbk.obj("plinth_blocks")
    assign(pbo, m_stone)
    paint(pbo, grime=lambda co, n: np.clip(0.16 + 0.3 * (1 - co[:, 2] / zp) + 0.15 * nz_m(co, n), 0, 1),
          special=lambda co, n: np.clip(nz_f(co, n) * (0.25 + 0.5 * wear) * (1.2 - co[:, 2] / zp), 0, 1))
    stone_w(caps, g=0.1, moss=0.7)

    # ---- stone trim: jointed cornice at the eaves line, string course at first-floor level
    rtr = rng.fork("trim")
    corn = [(-0.03, -0.2), (0.045, -0.2), (0.05, -0.13), (0.08, -0.11), (0.085, -0.06), (0.15, -0.01), (0.2, 0.0), (0.2, 0.06), (0.225, 0.065),
            (0.225, 0.11), (-0.03, 0.145)]
    strc = [(-0.03, -0.07), (0.035, -0.07), (0.04, -0.03), (0.07, -0.015), (0.07, 0.035), (-0.03, 0.06)]
    for nm, prof, zz, ext, ln0 in (("cornice", corn, z0, 0.05, 1.3), ("string_course", strc, floors[1], 0.0, 1.6)):
        nl_ = max(2, int(round((W + 2 * ext) / ln0)))
        cuts = [-a - ext] + sorted(-a - ext + (W + 2 * ext) * (k + rtr.u(-0.12, 0.12)) / nl_ for k in range(1, nl_)) + [a + ext]
        for k in range(nl_):
            o = band(nm, [(o_, zz + z + rtr.u(-0.0015, 0.0015)) for o_, z in prof], "front", cuts[k] + 0.003, cuts[k + 1] - 0.003)
            stone_w(o, g=0.04 + rtr.u(0.0, 0.1), moss=0.8)

    # coping: individual moulded stone blocks with joints, oversailing both faces of the gable wall
    hwc = GT / 2 + 0.07
    cop = [(-hwc, -0.012), (hwc, -0.012), (hwc, 0.085), (hwc - 0.03, 0.11), (0.06, 0.145), (-0.06, 0.145), (-hwc + 0.03, 0.11), (-hwc, 0.085)]
    runs, cur = [], [pts[0]]
    for i in range(1, len(pts)):
        if abs(pts[i][0] - pts[i - 1][0]) < 2e-3:
            runs.append(cur)
            cur = [pts[i]]
        else:
            cur.append(pts[i])
    runs.append(cur)
    nco = 0
    for run in runs:
        if len(run) < 2 or run[-1][0] - run[0][0] < 0.05:
            continue
        flat_run = max(h for _, h in run) - min(h for _, h in run) < 1e-3
        P = np.array([(x, z0 + h) for x, h in run])
        if flat_run:              # flat runs oversail the drop at their ends, like real coping slabs
            P = np.array([(run[0][0] - 0.045, z0 + run[0][1]), (run[-1][0] + 0.045, z0 + run[0][1])])
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        L = float(cum[-1])
        nb_ = max(1, int(round(L / 0.55)))
        if not flat_run and abs(run[0][0] + run[-1][0]) < 1e-3:
            nb_ = max(2, 2 * int(round(L / 1.1)))          # joint on the crown
        for b in range(nb_):
            s0 = L * b / nb_ + (0.004 if b > 0 else 0.0)
            s1 = L * (b + 1) / nb_ - (0.004 if b < nb_ - 1 else 0.0)
            inner = [float(s) for s in cum if s0 + 0.04 < s < s1 - 0.04]
            if len(inner) > 3:
                inner = inner[::int(ceil(len(inner) / 3))]
            St = np.array([s0] + inner + [s1])
            Q = np.stack([np.interp(St, cum, P[:, 0]), np.interp(St, cum, P[:, 1])], axis=1)
            T = np.gradient(Q, axis=0)
            T /= np.linalg.norm(T, axis=1)[:, None]
            dh, dy = rtr.u(-0.006, 0.006), rtr.u(-0.004, 0.004)
            verts = []
            for (qx, qz), (tx, tz) in zip(Q, T):
                for (s_, b_) in cop:
                    bb_ = b_ + (dh if b_ > 0.05 else 0.0)
                    verts.append((qx - tz * bb_, GT / 2 + dy + s_, qz + tx * bb_))
            m_ = len(cop)
            faces = [(i * m_ + k, i * m_ + (k + 1) % m_, (i + 1) * m_ + (k + 1) % m_, (i + 1) * m_ + k) for i in range(len(St) - 1) for k in range(m_)]
            faces += [tuple(range(m_)), tuple(range((len(St) - 1) * m_, len(St) * m_))]
            cb = mesh_object(f"coping{nco}", verts, faces)
            recalc_normals(cb)
            smooth(cb, 30)
            stone_w(cb, g=0.03 + rtr.u(0.0, 0.11), moss=0.9)
            nco += 1

    # finial on the crown of the gable
    if p.gable in ("bell", "neck"):
        fin = lathe("finial", [(0.0, 0.0), (0.12, 0.0), (0.125, 0.03), (0.075, 0.055), (0.06, 0.1), (0.075, 0.13), (0.14, 0.2), (0.165, 0.28),
                               (0.15, 0.35), (0.09, 0.4), (0.05, 0.42), (0.045, 0.46), (0.08, 0.49), (0.07, 0.53), (0.0, 0.6)], segments=10)
    else:
        fin = lathe("finial", [(0.0, 0.0), (0.11, 0.0), (0.11, 0.045), (0.06, 0.07), (0.055, 0.11), (0.1, 0.16), (0.14, 0.24), (0.14, 0.3),
                               (0.1, 0.38), (0.0, 0.42)], segments=10)
    place(fin, loc=(0, GT / 2, z0 + htop + 0.135))
    stone_w(fin, moss=0.8)
    if p.gable == "neck":
        xn = clamp(0.44 * a, 0.8, 1.15)
        hn = R + 0.8
        nc = box("neck_cornice", (2 * xn + 0.24, GT + 0.22, 0.1), loc=(0, GT / 2, z0 + hn + 0.05), bevel=0.02, segments=1)
        stone_w(nc)
    if p.gable == "bell":
        nb_ = band("bell_band", [(-0.03, bandz - 0.05), (0.035, bandz - 0.05), (0.04, bandz - 0.02), (0.075, bandz), (0.075, bandz + 0.04), (-0.03, bandz + 0.07)],
                   "front", -hwb, hwb)
        stone_w(nb_, moss=0.8)
        if oculus:
            oc = shape_extrude("bell_oculus", [(0.25 * cos(radians(t)), 0.25 * sin(radians(t))) for t in range(0, 360, 30)],
                               [[(0.16 * cos(radians(t)), 0.16 * sin(radians(t))) for t in range(0, 360, 30)]], depth=0.075, loc=(0, 0.03, ocz), rot=(90, 0, 0))
            stone_w(oc)
            ocg = cylinder("bell_oculus_glass", 0.17, 0.01, loc=(0, 0.06, ocz), rot=(90, 0, 0), verts=12)
            assign(ocg, m_dark)
            ob = Acc()
            ob.box((0.34, 0.016, 0.02), (0, 0.047, ocz), "yzZ").box((0.02, 0.016, 0.34), (0, 0.047, ocz), "yxX")
            assign(ob.obj("oculus_bars"), m_join)
    if p.gable in ("neck", "bell"):
        for sx in (-1, 1):          # carved volutes at the shoulders
            spth = spiral_path(0.03, 0.27, 1.6, n=26, a0_deg=90, plane="XZ", center=(sx * (a - 0.5), -0.012, z0 + 0.52), ccw=sx > 0)
            v = sweep(f"volute{sx}", spth, [(-0.03, -0.045), (0.03, -0.045), (0.03, 0.03), (-0.03, 0.03)], up=(0, 1, 0), smooth_path=False,
                      scale_fn=lambda t: 0.6 + 0.5 * t)
            stone_w(v)

    # ---- roof deck oversailing the long walls on rafter feet, pantiles, ridge, verges
    tv = 0.09 / cp_
    ys, ye = GT - 0.05, D + 0.2
    deck = prism("roof_deck", [(xe, ze), (0.0, zr), (-xe, ze), (-xe, ze - tv), (0.0, zr - tv), (xe, ze - tv)], 'Y', ys, ye)
    assign(deck, m_tile)
    assign_faces(deck, m_join, lambda c, n: n.z < -0.2 or abs(n.x) > 0.9 or n.y > 0.9)
    paint(deck, grime=0.4, special=0.3)
    ls = xe / cp_
    nrow = max(4, int(ceil((ls - 0.1) / 0.31)))
    ex = (ls - 0.1) / nrow
    lt = ex + 0.085
    ty0, ty1 = GT + 0.005, ye
    ncol = max(4, int(round((ty1 - ty0) / 0.232)))
    cw = (ty1 - ty0) / ncol
    tile = make_tile(cw + 0.03, lt)
    assign(tile, m_tile)
    rt = rng.fork("tiles")
    mats = []
    for sx in (1, -1):
        up = Vector((-sx * cp_, 0, sp_))
        nrm = Vector((sx * sp_, 0, cp_))
        xl = Vector((0, sx, 0))
        for r in range(nrow):
            for c in range(ncol):
                s = r * ex - 0.05
                slip = 0.0
                jz = rt.u(-1.2, 1.2)
                if rt.chance(0.03 * (0.5 + wear)):
                    slip = rt.u(0.02, 0.06)
                    jz = rt.u(-4, 4)
                pos = Vector((sx * xe, ty0 + cw * (c + 0.5) + rt.u(-0.004, 0.004), ze)) + up * (s - slip) + nrm * rt.u(0.0, 0.006)
                mats.append(frame_m(xl, up, nrm, pos) @ Matrix.Rotation(radians(jz), 4, 'Z'))
    # rafter feet under the eaves
    rf_ = Acc()
    nrf = max(4, int(round((D - GT - 0.3) / 0.55)))
    for sx in (1, -1):
        up = Vector((-sx * cp_, 0, sp_))
        nrm = Vector((sx * sp_, 0, cp_))
        yl = nrm.cross(up)
        xm = a + ov / 2 - 0.03
        for i in range(nrf):
            yy = GT + 0.25 + (D - GT - 0.4) * i / (nrf - 1)
            pos = Vector((sx * xm, yy, ze - tv + (xe - xm) * tp)) - nrm * 0.05
            rf_.box(((ov + 0.1) / cp_, 0.065, 0.1), (0, 0, 0), "xXyYz", m=frame_m(up, yl, nrm, pos))
    wood_w(rf_.obj("rafter_feet"), m_join, g=0.3, chip=0.5)
    # dormer on the slope away from the chimney: rendered cheeks, tiled gablet roof, small sash
    rdm = rng.fork("dormer")
    dsx = rdm.pick([1, -1])
    dq = radians(40)
    de = (2 * ex + 0.075) * cos(dq)             # half span of the dormer roof, sized to two tile courses
    dhw = de - 0.08
    drr = de * tan(dq)
    xf = a - 0.30
    zf = ze + (xe - xf) * tp
    ztop = zf + 1.25
    dyc = D * rdm.u(0.27, 0.36)
    dridge = []
    irons = []
    if p.dormer and zr - (ztop + drr) > 0.4:
        xr = xe - (ztop + drr - ze) / tp - 0.05
        xb = xe - (ztop - ze) / tp
        x0 = xf + 0.13
        dbody = prism("dormer_body", [(dsx * xf, zf - 0.25), (dsx * xf, ztop), (dsx * (xb - 0.06), ztop)], 'Y', dyc - dhw, dyc + dhw)
        dped = prism("dormer_pediment", [(dyc - dhw, ztop - 0.01), (dyc + dhw, ztop - 0.01), (dyc, ztop + drr - 0.07)], 'X', dsx * xf, dsx * xr)
        for o in (dbody, dped):
            assign(o, m_wall)
            paint(o, grime=lambda co, n: np.clip(0.1 + 0.25 * np.clip(1 - (co[:, 2] - zf) / 0.4, 0, 1), 0, 1) * (0.5 + 0.5 * wear), special=0.1 + 0.2 * wear)
        ddeck = prism("dormer_deck", [(dyc - de, ztop), (dyc, ztop + drr), (dyc + de, ztop), (dyc + de, ztop - 0.05), (dyc, ztop + drr - 0.06), (dyc - de, ztop - 0.05)],
                      'X', dsx * x0, dsx * xr)
        assign(ddeck, m_tile)
        paint(ddeck, grime=0.6)
        ndc = int(ceil((x0 - xr) / cw))
        for sy in (1, -1):
            up = Vector((0, -sy * cos(dq), sin(dq)))
            nrm = Vector((0, sy * sin(dq), cos(dq)))
            xl = Vector((-sy, 0, 0))
            for r in range(2):
                for c in range(ndc):
                    pos = Vector((dsx * (x0 - cw * (c + 0.5)), dyc + sy * de, ztop)) + up * (r * ex - 0.04) + nrm * rt.u(0.0, 0.005)
                    mats.append(frame_m(xl, up, nrm, pos) @ Matrix.Rotation(radians(rt.u(-1.2, 1.2)), 4, 'Z'))
            bb = box("dormer_barge", (0.03, de / cos(dq) + 0.04, 0.11), loc=(dsx * (x0 + 0.012), dyc + sy * de / 2, ztop + drr / 2 - 0.03), rot=(-sy * 40, 0, 0))
            wood_w(bb, m_join, g=0.25, chip=0.5)
        nd = int(ceil((x0 - xr) / 0.34))
        for i in range(nd):
            dridge.append(Matrix.Translation((dsx * (x0 + 0.02 - 0.34 * i), dyc, ztop + drr - 0.03)) @ Matrix.Rotation(dsx * pi / 2, 4, 'Z') @ Matrix.Scale(0.8, 4))
        irons.append(lathe("dormer_finial", [(0.0, 0.0), (0.03, 0.0), (0.035, 0.03), (0.02, 0.05), (0.04, 0.09), (0.035, 0.13), (0.0, 0.2)], segments=6,
                           loc=(dsx * (x0 - 0.03), dyc, ztop + drr + 0.08)))
        wins.setdefault(("dormer", pick_mode(rl.chance(p.lit_ratio))), []).append(Matrix.Translation((dsx * xf, dyc, zf + 0.22)) @ Matrix.Rotation(dsx * pi / 2, 4, 'Z'))
    # lean-to store at the rear: mono-pitch of the same pantiles
    lq = atan2(lz1 - lz0, ld)
    le = Vector((0, D + ld + 0.06, lz0 + 0.05 - 0.06 * tan(lq)))
    ncl = max(3, int(round((lw_ + 0.1) / cw)))
    cwl = (lw_ + 0.1) / ncl
    upl, nrl_, xll = Vector((0, -cos(lq), sin(lq))), Vector((0, sin(lq), cos(lq))), Vector((-1, 0, 0))
    for r in range(int(ceil((ld + 0.1) / cos(lq) / ex))):
        for c in range(ncl):
            pos = Vector((lnx - (lw_ + 0.1) / 2 + cwl * (c + 0.5), le.y, le.z)) + upl * (r * ex - 0.04) + nrl_ * rt.u(0.0, 0.004)
            mats.append(frame_m(xll, upl, nrl_, pos) @ Matrix.Rotation(radians(rt.u(-1.2, 1.2)), 4, 'Z') @ Matrix.Diagonal((cwl / cw, 1, 1, 1)))
    tiles = instances_to_mesh("pantiles", tile, mats, keep_base=False)
    paint(tiles, grime=mask_mul(noise_mask(1.5, seed=sd + 21, lo=0.2, hi=0.9), 0.3 + 0.35 * wear),
          special=mask_mul(noise_mask(0.8, seed=sd + 22, lo=0.5, hi=0.8), 0.15 + 0.55 * wear))
    rtile = surface("ridge_tile", lambda u, v: (-0.155 * cos(pi * u), v * 0.44, 0.155 * sin(pi * u) * (1.0 + 0.06 * v)), 6, 1, thickness=0.016)
    assign(rtile, m_tile)
    realize(rtile)
    nrid = max(3, int(round((ty1 - ty0) / 0.4)))
    rw = (ty1 - ty0) / nrid
    rmats = [trs((rt.u(-0.006, 0.006), ty0 + rw * i - 0.01, zr - 0.035 + rt.u(0, 0.006)), (rt.u(-1, 1), 0, rt.u(-1.5, 1.5))) for i in range(nrid)]
    ridge = instances_to_mesh("ridge_tiles", rtile, rmats + dridge, keep_base=False)
    paint(ridge, grime=0.3 + 0.3 * wear, special=mask_mul(noise_mask(0.6, seed=sd + 23), 0.3 + 0.6 * wear))
    gp = [(0.07 * cos(radians(t)), 0.07 * sin(radians(t))) for t in (180, 225, 270, 315, 360)] + [(0.05, 0.0), (0.0, -0.05), (-0.05, 0.0)]
    for sx in (1, -1):
        mx, mz = sx * xe / 2, (ze + zr) / 2
        # barge board on the rear verge, mortar fillet against the street gable
        bb = box("barge", (ls + 0.06, 0.035, 0.2), loc=(mx + sx * 0.03 * sp_, ye + 0.012, mz - 0.03), rot=(0, sx * p.pitch, 0))
        wood_w(bb, m_join, g=0.25, chip=0.5)
        fl = box("verge_fillet", (ls - 0.25, 0.14, 0.05), loc=(mx - sx * 0.1 * cp_ + sx * 0.07 * sp_, GT + 0.06, mz + 0.1 * sp_ + 0.07 * cp_), rot=(0, sx * p.pitch, 0))
        stone_w(fl, g=0.25, moss=1.0)
        # half-round gutter laid to a slight fall towards the rear
        gx, gz = sx * (xe + 0.075), ze - 0.06
        irons.append(sweep(f"gutter{sx}", [(gx, ys + 0.02, gz + 0.014), (gx, (ys + ye) / 2, gz + 0.004), (gx, ye + 0.03, gz - 0.006)], gp, smooth_path=False))
    gbk = Acc()
    nb = max(3, int((ye - ys) / 0.9))
    for sx in (1, -1):
        for i in range(nb):
            gbk.box((0.17, 0.024, 0.008), (sx * (xe + 0.06), ys + 0.25 + (ye - ys - 0.5) * i / (nb - 1), ze - 0.135), "xXyYz")
    irons.append(gbk.obj("gutter_brackets"))

    # ---- rendered chimney stack through one roof slope: oversailing stone cap, two clay pots
    rc = rng.fork("chimney")
    csx = -dsx
    cx = csx * a * 0.42
    cy = D * rc.u(0.5, 0.66)
    cwx, cwy = 0.52, 0.92
    z_lo = zr - R * (abs(cx) + cwx / 2) / a - 0.25
    z_top = zr + rc.u(0.75, 1.0)
    ck = Acc()
    ck.box((cwx, cwy, z_top - z_lo), (cx, cy, (z_top + z_lo) / 2), "xXyY")
    stack = ck.obj("chimney_stack")
    densify(stack, 0.5)
    assign(stack, m_wall)
    paint(stack, grime=lambda co, n: np.clip(0.12 + 0.55 * np.clip(1 - (z_top - co[:, 2]) / 0.9, 0, 1), 0, 1) * (0.6 + 0.5 * wear), special=0.15 + 0.3 * wear)
    zk = z_top
    for k, (grow, hh) in enumerate(((0.05, 0.07), (0.1, 0.07), (0.06, 0.05))):
        c = box(f"chimney_cap{k}", (cwx + 2 * grow, cwy + 2 * grow, hh), loc=(cx, cy, zk + hh / 2), bevel=0.012, segments=1)
        stone_w(c, g=0.3, moss=0.8)
        zk += hh
    zc_ = z_top + 0.19
    fill = box("chimney_flaunch", (cwx + 0.02, cwy + 0.02, 0.06), loc=(cx, cy, zc_ + 0.02), bevel=0.025, segments=1)
    stone_w(fill, g=0.45)
    for k, sy in enumerate((-1, 1)):
        ph = rc.u(0.42, 0.6)
        r0 = rc.u(0.095, 0.11)
        if k == 0:
            prof = [(r0 + 0.02, 0.0), (r0 + 0.02, 0.05), (r0, 0.07), (r0 - 0.012, ph - 0.07), (r0 + 0.012, ph - 0.05), (r0 + 0.016, ph - 0.02), (r0 + 0.004, ph),
                    (r0 - 0.02, ph), (r0 - 0.02, ph - 0.12)]
        else:
            prof = [(r0 + 0.02, 0.0), (r0 + 0.02, 0.06), (r0 + 0.004, 0.08), (r0 - 0.02, ph * 0.55), (r0 - 0.026, ph - 0.1), (r0 - 0.005, ph - 0.07), (r0 - 0.002, ph - 0.03),
                    (r0 - 0.018, ph), (r0 - 0.04, ph), (r0 - 0.04, ph - 0.12)]
        pot = lathe(f"chimney_pot{k}", prof, segments=10, loc=(cx + rc.u(-0.01, 0.01), cy + sy * cwy * 0.24, zc_ + 0.03), rot=(rc.u(-2, 2), rc.u(-2, 2), 0))
        assign(pot, m_tile)
        paint(pot, grime=lambda co, n: np.clip(0.35 + 1.4 * (co[:, 2] - zc_ - 0.1), 0, 1) * (0.6 + 0.4 * wear))
    fx("smoke", "smoke", (cx, cy - cwy * 0.24, zc_ + 0.6), size=0.35, rate=0.5)

    # ---- cast-iron downpipes on the flanks near the rear corners
    dpc = Acc()
    for sx in (1, -1):
        xh = sx * (a + 0.1)
        irons.append(lathe(f"dp_hopper{sx}", [(0.044, 0.0), (0.052, 0.02), (0.06, 0.05), (0.115, 0.17), (0.135, 0.18), (0.135, 0.2), (0.125, 0.21), (0.125, 0.25), (0.14, 0.26),
                                             (0.14, 0.29), (0.12, 0.29), (0.11, 0.2), (0.0, 0.2)], segments=4, loc=(xh, yd, hop_z), rot=(0, 0, 45), smooth_angle=20))
        irons.append(sweep(f"dp_pipe{sx}", [(xh, yd, hop_z + 0.02), (xh, yd, 0.4), (xh + sx * 0.03, yd, 0.3), (xh + sx * 0.12, yd, 0.2), (xh + sx * 0.17, yd, 0.16)],
                           0.043, sides=6, smooth_path=False))
        gx, gz = sx * (xe + 0.075), ze - 0.06
        irons.append(sweep(f"gutter_drop{sx}", [(gx, yd, gz - 0.05), (gx, yd, gz - 0.15), (lerp(gx, xh, 0.5), yd, lerp(gz - 0.15, hop_z + 0.3, 0.5)),
                                               (xh, yd, hop_z + 0.3), (xh, yd, hop_z + 0.16)], 0.036, sides=6, smooth_path=False))
        ncl_ = max(2, int(hop_z / 1.8))
        for i in range(ncl_ + 1):
            zz = 0.55 + (hop_z - 0.75) * i / ncl_
            dpc.box((0.105, 0.105, 0.03), (xh, yd, zz), "xXyYzZ")
            dpc.box((0.08, 0.2, 0.024), (sx * (a + 0.02), yd, zz), "xXyYzZ")
        gul = box(f"gully{sx}", (0.36, 0.4, 0.09), loc=(sx * (a + 0.2), yd, 0.045), bevel=0.015, segments=1)
        assign(gul, m_stone)
        paint(gul, grime=0.5, special=0.5 + 0.4 * wear)
    irons.append(dpc.obj("dp_collars"))

    # ---- windows: every size / room state is one mesh, stamped on the four walls
    # CLEARANCE RULE: a lit pane keeps 4 mm or more from whatever lies behind / in front of it. The planted dormer
    # sash stands 57 mm proud, so its lower pane (y0 + 0.051) is 6 mm in front of the dormer's rendered face.
    for gi, ((kind, mode), ms) in enumerate(wins.items()):
        w_, h_, cols, style = kinds[kind]
        base, sbase, gbase = make_window(f"win_{kind}_{mode}", w_, h_, cols, style, mode, MW, y0=-0.057 if kind == "dormer" else 0.045, uvo=(float(gi * 3 % 7), float(gi * 5 % 11)))
        wo = instances_to_mesh(f"windows_{kind}_{mode}", base, ms, keep_base=False)
        go_ = instances_to_mesh(f"windows_{kind}_{mode}_glass", gbase, ms, keep_base=False, uv_jitter=False)
        uv_keep(go_)
        paint(go_, grime=0.0, wear=0.0, special=0.0)       # clean panes: the room behind must read
        paint(wo, grime=0.04 + 0.06 * wear, wear=mask_mul(nz_p, 0.06 + 0.22 * wear), special=0.1 + 0.2 * wear)
        so_ = instances_to_mesh(f"windows_{kind}_{mode}_stone", sbase, ms, keep_base=False)
        paint(so_, grime=lambda co, n: np.clip(0.04 + 0.1 * nz_m(co, n), 0, 1) * (0.6 + 0.6 * wear),
              special=mask_mul(facing((0, 0, 1)), nz_f, 0.3 + 0.4 * wear))
    if shut:
        for kind, ms in sh_m.items():
            base = make_shutter(p.shutters, ww / 2, kinds[kind][1], m_shut)
            so = instances_to_mesh(f"shutters_{kind}", base, ms, keep_base=False)
            paint(so, wear=mask_mul(nz_p, 0.25 + 0.6 * wear), special=mask_mul(noise_mask(0.5, seed=sd + 72), 0.3 + 0.5 * wear), grime=0.1 + 0.15 * wear)
        hb = Acc()
        hb.box((0.17, 0.006, 0.034), (0.085, 0, 0), "xXyzZ").box((0.02, 0.02, 0.06), (0, 0, 0), "xXyzZ")
        irons.append(instances_to_mesh("shutter_hinges", hb.obj("hinge"), hinge_m, keep_base=False))
        dg = Acc()
        dg.box((0.02, 0.006, 0.13), (0, -0.045, 0), "xXyzZ").box((0.014, 0.075, 0.014), (0, -0.0075, 0), "xXzZ")
        dg.box((0.05, 0.006, 0.02), (0.012, -0.045, 0.06), "xXyzZ").box((0.05, 0.006, 0.02), (-0.012, -0.045, -0.06), "xXyzZ")
        irons.append(instances_to_mesh("shutter_dogs", dg.obj("dog"), dog_m, keep_base=False))

    # ---- front door: moulded stone case with jambs, cream frame, six-panel leaf, lit fanlight
    angs = [0, 7, 21, 36, 51, 64, 74, 90, 106, 116, 129, 144, 159, 173, 180]
    zj = [0.0, 0.27, spr - 0.075]                            # jamb cuts above the base block / below the impost

    def hpath(r):
        return ([(r, z) for z in zj] + [(r * cos(radians(t)), spr + r * sin(radians(t))) for t in angs]
                + [(-r, z) for z in reversed(zj)])

    def arch_band(name, r_in, r_out, yf, yb_in=None, yb_out=None):
        """Horseshoe band between two concentric door-head paths: face at yf, reveal back to yb_in,
        outer return back to yb_out."""
        pi_, po_ = hpath(r_in), hpath(r_out)
        n = len(pi_)
        V = [(dx + x, yf, zp + z) for x, z in pi_] + [(dx + x, yf, zp + z) for x, z in po_]
        F = [(i, n + i, n + i + 1, i + 1) for i in range(n - 1)]
        if yb_in is not None:
            b = len(V)
            V += [(dx + x, yb_in, zp + z) for x, z in pi_]
            F += [(i, i + 1, b + i + 1, b + i) for i in range(n - 1)]
        if yb_out is not None:
            b = len(V)
            V += [(dx + x, yb_out, zp + z) for x, z in po_]
            F += [(n + i, b + i, b + i + 1, n + i + 1) for i in range(n - 1)]
        o = mesh_object(name, V, F)
        smooth(o, 40)
        return o

    r0 = ow / 2
    case = [arch_band("door_case", r0, r0 + 0.15, -0.105, yb_in=0.18),
            arch_band("door_backband", r0 + 0.13, r0 + 0.22, -0.14, yb_in=-0.105, yb_out=0.03),
            arch_band("door_bead", r0 + 0.045, r0 + 0.075, -0.12, yb_in=-0.105, yb_out=-0.105)]
    kh = key_top - (Hd - 0.03)
    key = prism("door_key", [(-0.105, Hd - 0.03), (0.105, Hd - 0.03), (0.15, key_top), (-0.15, key_top)], 'Y', -0.185, 0.03)
    place(key, loc=(dx, 0, zp))
    case.append(key)
    for sx in (-1, 1):
        case.append(box("door_impost", (0.27, 0.2, 0.085), loc=(dx + sx * (ow / 2 + 0.11), -0.065, zp + spr), bevel=0.012, segments=1))
        case.append(box("door_base", (0.26, 0.2, 0.24), loc=(dx + sx * (ow / 2 + 0.11), -0.062, zp + 0.12), bevel=0.012, segments=1))
    thr = Acc()
    thr.box((ow + 0.01, 0.3, 0.03), (dx, 0.03, zp + 0.006), "yZ")
    case.append(thr.obj("door_threshold"))
    stone_w(case, g=0.05, moss=0.2)
    dfr = arch_band("door_frame", r0 - 0.065, r0 + 0.01, 0.03, yb_in=0.1)
    ldw = ow - 0.13
    zc_f = zp + spr
    rf = ldw / 2
    fb = Acc()
    fb.box((ldw + 0.01, 0.075, 0.08), (dx, 0.0675, zp + lh_ + 0.04), "yzZ")
    for k in range(1, 6):
        fb.box((rf - 0.12, 0.02, 0.02), ((rf + 0.12) / 2, 0, 0), "yzZ", m=Matrix.Translation((dx, 0.062, zc_f)) @ Matrix.Rotation(radians(-30 * k), 4, 'Y'))
    hub = sweep("fan_hub", arc_path(0.13, 0, 180, n=8, plane="XZ", center=(dx, 0.062, zc_f)), rect(0.022, 0.022), up=(0, 1, 0), smooth_path=False)
    for o in (dfr, fb.obj("fanlight_bars"), hub):
        wood_w(o, m_join, g=0.08, chip=0.25)
    # leaf: rails and stiles are one welded grid (edge vertices take the contact shadow), panels are
    # sunk with a raised field - all in the same paint
    yL = 0.052
    L_ = ldw + 0.01
    gx_ = [-L_ / 2, -L_ / 2 + 0.03, -L_ / 2 + 0.105, -0.045, 0.045, L_ / 2 - 0.105, L_ / 2 - 0.03, L_ / 2]
    gz_ = [0.0, 0.03, 0.17, 0.30 * lh_, 0.30 * lh_ + 0.11, 0.70 * lh_, 0.70 * lh_ + 0.11, lh_ - 0.11, lh_ - 0.03, lh_]
    nxg = len(gx_)
    GV = [(dx + x, yL, zp + z) for z in gz_ for x in gx_]
    GF = []
    lf = Acc()
    for j in range(len(gz_) - 1):
        for i in range(nxg - 1):
            if i in (2, 4) and j in (2, 4, 6):
                pw_, ph_ = gx_[i + 1] - gx_[i], gz_[j + 1] - gz_[j]
                cx_, cz_ = dx + (gx_[i] + gx_[i + 1]) / 2, zp + (gz_[j] + gz_[j + 1]) / 2
                lf.frustum(pw_, ph_, pw_ - 0.036, ph_ - 0.036, yL, yL + 0.016, cx_, cz_)
                lf.frustum(pw_ - 0.085, ph_ - 0.085, pw_ - 0.15, ph_ - 0.15, yL + 0.016, yL + 0.004, cx_, cz_)
            else:
                GF.append((j * nxg + i, j * nxg + i + 1, (j + 1) * nxg + i + 1, (j + 1) * nxg + i))
    lf.polys(GV, GF)
    dleaf = lf.obj("front_door")
    assign(dleaf, m_shut)
    paint(dleaf, wear=lambda co, n: np.clip(nz_p(co, n) * 0.2 * wear + 0.45 * np.clip(1 - (co[:, 2] - zp) / 0.14, 0, 1) * (0.3 + 0.7 * wear), 0, 1),
          grime=lambda co, n: 0.02 + 0.15 * np.clip(1 - (co[:, 2] - zp) / 0.4, 0, 1), special=0.05 + 0.15 * wear)
    nf = 12
    yfg = 0.078                                             # fanlight glass 5 mm behind the hub, 6 mm behind the bars (clearance rule)
    fv = [(dx, yfg, zc_f)] + [(dx + rf * cos(pi * i / nf), yfg, zc_f + rf * sin(pi * i / nf)) for i in range(nf + 1)]
    fan = uv_mesh("fanlight_glass", fv, [(0, i + 1, i + 2) for i in range(nf)], [((q[0] - dx) / (2 * rf) * 0.96 + 0.5 + 2.0, (q[2] - zc_f) / (2 * rf) * 0.96 + 0.26 + 9.0) for q in fv])
    hall_lit = p.lit_ratio > 0.01
    assign(fan, m_dim if hall_lit else m_dark)
    if hall_lit:
        light("hall", (dx, 0.45, zc_f + rf * 0.3), color="#ffc98a", intensity=4.0, range=5.5, flicker=0.05)
    hs = -1 if door_l else 1                                # hinge side
    kx_ = dx - hs * (ldw / 2 - 0.055)
    df = Acc()
    df.box((0.05, 0.006, 0.17), (kx_, yL - 0.003, zp + 1.0), "xXyzZ")
    df.box((0.21, 0.008, 0.05), (dx, yL - 0.004, zp + 0.30 * lh_ + 0.055), "xXyzZ")
    df.box((0.05, 0.022, 0.05), (dx, yL - 0.011, zp + 0.70 * lh_ + 0.055 + 0.04), "xXyzZ")
    dfi = [df.obj("door_plates"),
           lathe("door_knob", [(0.0, 0.0), (0.024, 0.0), (0.026, 0.006), (0.012, 0.012), (0.012, 0.034), (0.03, 0.048), (0.035, 0.062), (0.024, 0.078), (0.0, 0.082)],
                 segments=8, loc=(kx_, yL - 0.003, zp + 1.04), rot=(90, 0, 0)),
           torus("door_knocker", 0.05, 0.009, loc=(dx, yL - 0.022, zp + 0.70 * lh_ + 0.045), rot=(80, 0, 0), major_seg=10, minor_seg=4)]
    if p.number and kh > 0.17:
        ns_ = min(0.15, kh - 0.09)
        dfi.append(text("house_number", str(p.number), size=ns_, depth=0.016, loc=(dx, -0.183, zp + Hd - 0.03 + (kh - ns_) / 2), font="serif"))
    iron_w(dfi, rust=0.05, bright=0.75)

    # ---- stoop: stone flight with slab treads, wrought handrails, boot scraper
    going, landing = 0.28, 0.64
    sw_ = ow + 0.5
    rstp = rng.fork("stoop")
    sb_ = Acc()
    for j in range(steps):
        dj = landing + (steps - 1 - j) * going
        zt = (j + 1) * rise
        sb_.box((sw_ - 0.05, dj - 0.03, rise - 0.045), (dx, -(dj - 0.03) / 2, zt - 0.05 - (rise - 0.045) / 2 + 0.009), "xXy")
        tread = stone_block(f"step_tread{j}", (sw_, dj + 0.012, 0.052), loc=(dx + rstp.u(-0.004, 0.004), -(dj + 0.012) / 2, zt - 0.026), rng=rstp, chamfer=0.014, rough=0.003, detail=0.5)
        assign(tread, m_stone)
        paint(tread, grime=lambda co, n: np.clip(0.06 + 0.16 * (1 - co[:, 2] / zp) + 0.2 * (np.abs(co[:, 0] - dx) > sw_ / 2 - 0.12), 0, 1),
              wear=lambda co, n: np.clip(n[:, 2], 0, 1) * np.exp(-((co[:, 0] - dx) / 0.3) ** 2) * (0.1 + 0.15 * wear),
              special=lambda co, n: np.clip((np.abs(co[:, 0] - dx) > sw_ / 2 - 0.2) * nz_f(co, n) * (0.4 + 0.6 * wear), 0, 1))
    sbo = sb_.obj("step_bodies")
    assign(sbo, m_stone)
    paint(sbo, grime=0.4, special=mask_mul(nz_f, 0.3 + 0.5 * wear))
    rail_i = []
    ri = Acc()
    y_top = -landing + 0.06
    y_bot = -(landing + (steps - 1) * going) + 0.13
    z_bot = rise
    hr = 0.92
    for sx in (-1, 1):
        rx = dx + sx * (sw_ / 2 - 0.06)
        path = [(rx, 0.02, zp + hr), (rx, y_top, zp + hr), (rx, y_bot, z_bot + hr), (rx, y_bot - 0.1, z_bot + hr - 0.06)]
        rail_i.append(sweep(f"rail{sx}", path, rect(0.036, 0.012), smooth_path=False))
        sc = spiral_path(0.012, 0.062, 1.35, n=13, a0_deg=80, plane="YZ", center=(rx, y_bot - 0.115, z_bot + hr - 0.125))
        rail_i.append(sweep(f"rail_scroll{sx}", sc, rect(0.008, 0.03), up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t))
        posts = [(y_top, zp), (y_bot, z_bot)]
        for k in range(1, steps - 1):
            posts.append((-(landing + (k - 0.5) * going), zp - k * rise))
        posts.append((-landing * 0.45, zp))
        for i, (py, pz) in enumerate(posts):
            zt_ = zp + hr if py >= y_top else z_bot + hr + (py - y_bot) / (y_top - y_bot) * (zp - z_bot)
            if i < 2:
                ri.box((0.024, 0.024, zt_ - pz), (rx, py, (pz + zt_) / 2), "xXyY")
                ri.box((0.05, 0.05, 0.014), (rx, py, pz + 0.005), "xXyYZ")
                rail_i.append(lathe("post_knob", [(0.0, 0.0), (0.016, 0.0), (0.012, 0.012), (0.026, 0.03), (0.026, 0.042), (0.0, 0.06)], segments=6, loc=(rx, py, zt_ + 0.004)))
            else:
                rail_i.append(sweep(f"post{sx}_{i}", [(rx, py, pz - 0.01), (rx, py, zt_ - 0.004)], rect(0.015, 0.015), twist_deg=360, step=0.1, smooth_path=False))
                ri.box((0.034, 0.034, 0.01), (rx, py, pz + 0.004), "xXyYZ")
    sxs = 1 if door_l else -1
    bsx = dx + sxs * (sw_ / 2 + 0.16)
    if abs(bsx) < a - 0.15 and not p.cellar:
        for sy in (-0.09, 0.09):
            ri.box((0.018, 0.018, 0.21), (bsx, -0.3 + sy, 0.105), "xXyYZ")
        ri.box((0.006, 0.19, 0.05), (bsx, -0.3, 0.11), "xXzZ")
    rail_i.append(ri.obj("rail_posts"))
    irons += rail_i

    # ---- wall lantern on a scrolled bracket over the string course, between door and parlour window
    if p.lantern_cd > 0.5:
        reach = 0.54
        la = Acc()
        la.box((0.07, 0.04, 0.46), (lx, -0.008 + 0.014, lz - 0.1), "xXyzZ")
        for bz in (0.09, -0.29):
            la.stud(0.024, -0.014, -0.026, lx, lz + bz)
        li = [sweep("lan_arm", [(lx, -0.01, lz), (lx, -reach * 0.5, lz + 0.015), (lx, -reach, lz), (lx, -reach - 0.07, lz - 0.03)], rect(0.02, 0.02), up=(1, 0, 0), smooth_path=False),
              sweep("lan_tip", spiral_path(0.008, 0.04, 1.2, n=10, a0_deg=90, plane="YZ", center=(lx, -reach - 0.075, lz - 0.07)), rect(0.007, 0.02),
                    up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.55 + 0.45 * t),
              sweep("lan_stay", [(lx, -0.012, lz - 0.3), (lx, -0.12, lz - 0.25), (lx, -0.26, lz - 0.1), (lx, -0.33, lz - 0.012)], rect(0.007, 0.02), up=(1, 0, 0), smooth_path=False),
              sweep("lan_scroll", spiral_path(0.012, 0.085, 1.5, n=14, a0_deg=200, plane="YZ", center=(lx, -0.115, lz - 0.115)), rect(0.007, 0.02),
                    up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.5 + 0.5 * t)]
        ly = -reach + 0.04
        zt_l = lz - 0.13                                   # top of the lantern cage
        hb_, t_, b_ = 0.27, 0.105, 0.07
        li.append(torus("lan_ring", 0.022, 0.005, loc=(lx, ly, lz - 0.03), rot=(0, 90, 0), major_seg=8, minor_seg=4))
        la.box((0.012, 0.012, 0.05), (lx, ly, lz - 0.065), "xXyY")
        li.append(cone("lan_roof", r1=t_ * 1.75, r2=0.035, depth=0.085, loc=(lx, ly, zt_l + 0.04), rot=(0, 0, 45), verts=4))
        li.append(cylinder("lan_vent", 0.034, 0.035, loc=(lx, ly, zt_l + 0.095), verts=6))
        li.append(cone("lan_cap", r1=0.05, r2=0.008, depth=0.03, loc=(lx, ly, zt_l + 0.125), verts=6))
        li.append(shape_extrude("lan_top", rect(2 * t_ + 0.03, 2 * t_ + 0.03), [rect(2 * t_ - 0.02, 2 * t_ - 0.02)], depth=0.014, loc=(lx, ly, zt_l - 0.014)))
        la.box((2 * b_ + 0.024, 2 * b_ + 0.024, 0.014), (lx, ly, zt_l - hb_ - 0.007))
        li.append(cone("lan_drop", r1=0.03, r2=0.006, depth=0.04, loc=(lx, ly, zt_l - hb_ - 0.034), rot=(180, 0, 0), verts=6))
        for sx in (-1, 1):
            for sy in (-1, 1):
                li.append(sweep("lan_bar", [(lx + sx * b_, ly + sy * b_, zt_l - hb_), (lx + sx * t_, ly + sy * t_, zt_l - 0.004)], rect(0.01, 0.01), smooth_path=False))
        la.box((0.022, 0.022, 0.06), (lx, ly, zt_l - hb_ + 0.03), "xXyYZ")
        li.append(la.obj("lan_plate"))
        gv, gf = [], []
        for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            gv.append((lx + sx * (b_ - 0.003), ly + sy * (b_ - 0.003), zt_l - hb_ + 0.004))
            gv.append((lx + sx * (t_ - 0.003), ly + sy * (t_ - 0.003), zt_l - 0.012))
        for k in range(4):
            k2 = (k + 1) % 4
            gf.append((2 * k, 2 * k2, 2 * k2 + 1, 2 * k + 1))
        lg = mesh_object("lan_glass", gv, gf)
        assign(lg, m_lglass)
        fl_ = lathe("lan_mantle", [(0.0, 0.0), (0.014, 0.004), (0.024, 0.024), (0.02, 0.05), (0.008, 0.078), (0.0, 0.092)], segments=6, loc=(lx, ly, zt_l - hb_ + 0.065))
        assign(fl_, m_lglass)
        light("lantern", (lx, ly - 0.06, zt_l - hb_ + 0.1), color="#ffbf6a", intensity=float(p.lantern_cd), range=10.0, flicker=0.14, shadow=True, shadow_near=0.26)
        irons += li

    # ---- wrought-iron wall anchors: S-anchors on the street front, plain bars on the flanks
    anch = sweep("anchor", [(0.05, 0, 0.185), (0.025, 0, 0.23), (-0.02, 0, 0.215), (-0.018, 0, 0.16), (0.0, 0, 0.12), (0.0, 0, -0.12), (0.018, 0, -0.16),
                            (0.02, 0, -0.215), (-0.025, 0, -0.23), (-0.05, 0, -0.185)], rect(0.03, 0.012), up=(0, 1, 0), smooth_path=False)
    ab = Acc()
    ab.box((0.024, 0.04, 0.024), (0, 0.012, 0), "xXzZ").stud(0.04, -0.006, -0.024, 0, 0)
    anc = join([anch, ab.obj("anchor_pin")], "anchor")
    am = [trs((x, -0.012, z), (0, ra.u(-4, 4), 0)) for (x, z) in anc_front]
    irons.append(instances_to_mesh("wall_anchors", anc, am, keep_base=False))
    sa = Acc()
    for (sx, yy, zz) in anc_side:
        m_ = FM["right" if sx > 0 else "left"] @ Matrix.Translation((sx * yy, 0, zz)) @ Matrix.Rotation(radians(ra.u(-3, 3)), 4, 'Y')
        sa.box((0.034, 0.014, 0.52), (0, -0.016, 0), "xXyzZ", m=m_)
        sa.box((0.024, 0.045, 0.024), (0, 0.006, 0), "xXzZ", m=m_)
        sa.stud(0.042, -0.023, -0.04, 0, 0, m=m_)
    if sa.v:
        irons.append(sa.obj("flank_anchors"))

    # ---- gable: loft doors, hoist beam with pulley block and chain, date numerals
    lst = Acc()
    for sx in (-1, 1):
        lst.box((0.1, 0.065, lh + 0.02), (sx * (lw / 2 + 0.046), -0.0025, lzz + lh / 2 + 0.01), "xXy", g=0.06)
    lst.box((lw + 0.2, 0.07, 0.11), (0, -0.005, lzz + lh + 0.051), "xXyzZ")
    lsill = prism("loft_sill", [(0.06, -0.085), (-0.105, -0.085), (-0.115, -0.075), (-0.115, -0.035), (-0.06, 0.0), (0.06, 0.01)], 'X', -lw / 2 - 0.15, lw / 2 + 0.15)
    place(lsill, loc=(0, 0, lzz))
    stone_w([lst.obj("loft_frame"), lsill], g=0.06)
    lb = Acc()
    li_ = Acc()
    rb = rng.fork("loft")
    for k in range(6):
        pw = lw / 6
        lb.box((pw - 0.005, 0.03, lh), (-lw / 2 + pw * (k + 0.5), 0.065 + rb.u(-0.003, 0.003), lzz + lh / 2), "xXy")
    lo = lb.obj("loft_doors")
    wood_w(lo, m_shut, g=0.12, chip=0.6, fade=0.4)
    for sx in (-1, 1):
        for hz in (0.2, lh - 0.2):
            li_.box((lw / 2 - 0.05, 0.007, 0.04), (sx * (lw / 4 + 0.015), 0.046, lzz + hz), "xXyzZ")
            li_.box((0.02, 0.02, 0.07), (sx * (lw / 2 - 0.012), 0.042, lzz + hz), "xXyzZ")
    li_.box((0.012, 0.012, 0.16), (0.05, 0.044, lzz + lh * 0.48), "xXyzZ")
    irons.append(li_.obj("loft_iron"))

    if p.hoist:
        by0, by1 = -1.06, 0.15                              # the beam runs through the gable wall
        beam = box("hoist_beam", (0.16, by1 - by0, 0.2), loc=(0, (by0 + by1) / 2, zb), bevel=0.01, segments=1)
        assign(beam, m_join)
        paint(beam, grime=0.2, wear=lambda co, n: np.clip(nz_p(co, n) * (0.35 + 0.5 * wear) + (co[:, 1] < by0 + 0.2) * 0.4, 0, 1), special=0.3)
        hy = by0 + 0.14
        hi_ = Acc()
        hi_.box((0.176, 0.05, 0.216), (0, hy, zb))
        hi_.box((0.176, 0.05, 0.216), (0, -0.2, zb))
        hi_.box((0.3, 0.04, 0.34), (0, -0.012 + 0.02 - 0.008, zb), "xXyzZ")
        for bx_, bz_ in ((-0.11, 0.12), (0.11, 0.12), (-0.11, -0.12), (0.11, -0.12)):
            hi_.stud(0.03, -0.02, -0.034, bx_, zb + bz_)
        hi_.box((0.014, 0.05, 0.1), (0, hy, zb - 0.15), "xXyY")      # eye strap under the beam
        irons.append(hi_.obj("hoist_iron"))
        zt = zb - 0.2
        fall = clamp(lh + 0.3, 1.3, 1.7)
        tk_ = Acc()
        for sx in (-1, 1):                                  # pulley block: two cheeks round a sheave
            tk_.box((0.012, 0.11, 0.2), (sx * 0.028, hy, zt - 0.1))
        tk_.box((0.07, 0.03, 0.03), (0, hy, zt - 0.012))
        nl_ = int(fall / 0.075)
        for i in range(nl_):
            tk_.box((0.032, 0.008, 0.088) if i % 2 == 0 else (0.008, 0.032, 0.088), (0, hy - 0.05, zt - 0.14 - 0.075 * (i + 0.5)), "xXyY")
        zh = zt - 0.14 - 0.075 * nl_
        tko = tk_.obj("hoist_block")
        sheave = cylinder("hoist_sheave", 0.06, 0.036, loc=(0, hy, zt - 0.11), rot=(0, 90, 0), verts=8)
        hook = sweep("hoist_hook", [(0, hy - 0.05, zh + 0.02), (0, hy - 0.05, zh - 0.09), (0, hy - 0.01, zh - 0.17), (0, hy + 0.06, zh - 0.17), (0, hy + 0.09, zh - 0.1)],
                     0.014, sides=5, scale_fn=lambda t: 1.0 - 0.5 * t ** 3)
        tack = [tko, sheave, hook]
        iron_w(tack, rust=0.3)
        part(tack, "hoist_tackle", pivot=(0, hy, zt), motion="swing", amp=1.5, speed=0.25)
    if p.date:
        ds = str(p.date)
        nh = (len(ds) + 1) // 2
        hd0 = lzz - z0 + lh * 0.5
        avail = halfw(hd0 + 0.22) - 0.14 - (lw / 2 + 0.2)
        size = min(0.33, avail / (0.6 * nh))
        if p.gable in ("neck", "bell"):
            size = min(size, (a - 0.5 - 0.36 - (lw / 2 + 0.2)) / (0.6 * nh)) if hd0 - 0.17 < 0.86 else size
        if size >= 0.2:
            xc = lw / 2 + 0.2 + 0.3 * nh * size
            for sx, body in ((-1, ds[:nh]), (1, ds[nh:])):
                if body:
                    irons.append(text(f"date{sx}", body, size=size, depth=0.05, loc=(sx * xc, 0.03, z0 + hd0 - size / 2), font="serif"))
        elif htop - (zb - z0) > 0.75:
            irons.append(text("date", ds, size=min(0.3, 0.75 * halfw(zb - z0 + 0.6)), depth=0.05, loc=(0, 0.03, zb + 0.26), font="serif"))

    # ---- cellar bulkhead under the parlour window
    if p.cellar:
        hw_, hp_, hh_ = clamp(bw - 0.5, 0.85, 1.15), 0.95, zp - 0.1
        ckp = [(-0.07, 0.0), (-0.07, hh_ + 0.03), (-hp_ + 0.1, 0.22), (-hp_, 0.2), (-hp_, 0.0)]
        rk = rng.fork("cellar")
        for sx in (-1, 1):
            xo = nx_bay + sx * (hw_ / 2 + 0.075)
            o = prism(f"cellar_cheek{sx}", ckp, 'X', xo - 0.075, xo + 0.075)
            assign(o, m_stone)
            paint(o, grime=lambda co, n: np.clip(0.15 + 0.35 * (1 - co[:, 2] / 0.5), 0, 1), special=mask_mul(nz_f, 0.4 + 0.5 * wear))
        kerb = box("cellar_kerb", (hw_ + 0.3, 0.16, 0.19), loc=(nx_bay, -hp_ + 0.08, 0.095), bevel=0.014, segments=1)
        assign(kerb, m_stone)
        paint(kerb, grime=0.4, special=mask_mul(nz_f, 0.4 + 0.5 * wear))
        y_lo, z_lo2, y_hi, z_hi2 = -hp_ + 0.13, 0.2, -0.075, hh_ + 0.01
        sl = hypot(y_hi - y_lo, z_hi2 - z_lo2)
        th = atan2(z_hi2 - z_lo2, y_hi - y_lo)
        mh = Matrix.Translation((nx_bay, y_lo, z_lo2)) @ Matrix.Rotation(th, 4, 'X')
        cw_, ci_ = Acc(mh), Acc(mh)
        for sx in (-1, 1):
            lwid = hw_ / 2 - 0.006
            for k in range(3):
                pw = lwid / 3
                cw_.box((pw - 0.005, sl, 0.034), (sx * (0.006 + pw * (k + 0.5)), sl / 2 + rk.u(-0.006, 0.006), 0.017 + rk.u(0, 0.003)), "xXyYZ")
            for t in (0.22, 0.78):
                ci_.box((lwid - 0.03, 0.045, 0.007), (sx * (lwid / 2 + 0.02), sl * t, 0.038), "xXyYZ")
                ci_.box((0.022, 0.08, 0.022), (sx * (lwid + 0.004), sl * t, 0.042), "xXyYZ")
            ci_.box((0.014, 0.12, 0.014), (sx * 0.1, sl * 0.36, 0.046), "xXyYZ")
        ci_.box((0.11, 0.035, 0.008), (0.0, sl * 0.12, 0.04), "xXyYZ")
        wo = cw_.obj("cellar_doors")
        wood_w(wo, m_shut, g=0.25 + 0.2 * wear, chip=0.7, fade=0.4)
        irons.append(ci_.obj("cellar_iron"))

    # ---- yard door at the rear: stone case on a threshold, boarded leaf in the shutter paint
    my = FM["back"] @ Matrix.Translation((-ybx, 0, 0))
    ys_, yw_, yi_ = Acc(my), Acc(my), Acc(my)
    for sx in (-1, 1):
        ys_.box((0.13, 0.16, ydt), (sx * (ydw / 2 + 0.065), 0.03, ydt / 2), "xXy", g=0.2)
    ys_.box((ydw + 0.34, 0.17, 0.18), (0, 0.025, ydt + 0.09), "xXyzZ")
    ys_.box((ydw + 0.5, 0.36, zth), (0, -0.07, zth / 2), "xXyZ")
    stone_w(ys_.obj("yard_case"), g=0.1, moss=0.6)
    for k in range(5):
        pw = ydw / 5
        yw_.box((pw - 0.005, 0.03, ydh + 0.01), (-ydw / 2 + pw * (k + 0.5), 0.075 + rb.u(-0.002, 0.002), zth + ydh / 2), "xXy")
    for hz in (0.3, ydh - 0.3):
        yi_.box((ydw * 0.7, 0.007, 0.045), (-ydw * 0.14, 0.057, zth + hz), "xXyzZ")
        yi_.box((0.022, 0.022, 0.08), (-ydw / 2 + 0.012, 0.052, zth + hz), "xXyzZ")
    yi_.box((0.04, 0.008, 0.14), (ydw / 2 - 0.09, 0.056, zth + 1.0), "xXyzZ")
    yi_.box((0.02, 0.05, 0.02), (ydw / 2 - 0.09, 0.03, zth + 1.03), "xXyzZ")
    wood_w(yw_.obj("yard_door"), m_shut, g=0.06, chip=0.2, fade=0.1)
    irons.append(yi_.obj("yard_iron"))

    # ---- lean-to store against the rear wall
    lbody = prism("leanto_body", [(D - 0.02, 0.0), (D + ld, 0.0), (D + ld, lz0), (D - 0.02, lz1)], 'X', lnx - lw_ / 2, lnx + lw_ / 2)
    densify(lbody, 0.45)
    assign(lbody, m_wall)
    paint(lbody, grime=lambda co, n: np.clip(0.14 + 0.1 * nz_b(co, n) + 0.3 * np.clip(1 - co[:, 2] / 0.6, 0, 1), 0, 1) * (0.55 + 0.6 * wear),
          special=lambda co, n: np.clip(1 - co[:, 2] / 0.7, 0, 1) * (0.3 + 0.45 * wear))
    ldeck = prism("leanto_deck", [(D - 0.02, lz1 + 0.05), (D + ld + 0.06, le.z), (D + ld + 0.06, le.z - 0.05), (D - 0.02, lz1)], 'X', lnx - lw_ / 2 - 0.05, lnx + lw_ / 2 + 0.05)
    assign(ldeck, m_tile)
    assign_faces(ldeck, m_join, lambda c, n: n.z < -0.2 or n.y > 0.9 or abs(n.x) > 0.9)
    paint(ldeck, grime=0.4, special=0.3)
    ml = FM["back"] @ Matrix.Translation((-lnx, 0, 0))
    lts, ltw, lti, ltj = Acc(ml), Acc(ml), Acc(ml), Acc(ml)
    lts.box((lw_ + 0.04, ld + 0.02, 0.18), (0, -ld / 2 - 0.01, 0.09), "xXy")
    lts.box((lw_ + 0.12, 0.05, 0.12), (0, -0.02, lz1 + 0.1), "xXyZ")              # flashing fillet under the wall
    stone_w(lts.obj("leanto_base"), g=0.2, moss=0.8)
    ldw_, ldh_ = 0.62, 1.52
    for k in range(4):
        pw = ldw_ / 4
        ltw.box((pw - 0.005, 0.03, ldh_), (-ldw_ / 2 + pw * (k + 0.5), -ld - 0.012 + rb.u(-0.002, 0.002), 0.2 + ldh_ / 2), "xXy")
    for sx in (-1, 1):
        ltj.box((0.06, 0.04, ldh_ + 0.06), (sx * (ldw_ / 2 + 0.03), -ld - 0.012, 0.2 + ldh_ / 2 + 0.03), "xXy", g=0.06)
    ltj.box((ldw_ + 0.12, 0.04, 0.06), (0, -ld - 0.012, 0.2 + ldh_ + 0.03), "xXyZ")
    for hz in (0.25, ldh_ - 0.25):
        lti.box((ldw_ * 0.7, 0.007, 0.04), (ldw_ * 0.14, -ld - 0.031, 0.2 + hz), "xXyzZ")
    lti.box((0.03, 0.02, 0.1), (-ldw_ / 2 + 0.08, -ld - 0.036, 0.2 + 0.85), "xXyzZ")
    wood_w(ltw.obj("leanto_door"), m_shut, g=0.08, chip=0.3, fade=0.15)
    wood_w(ltj.obj("leanto_frame"), m_join, g=0.15, chip=0.5)
    irons.append(lti.obj("leanto_iron"))

    iron_w(irons)

    # ---- sockets for the town generator
    socket("sign_front", "attach", (nx_bay, 0, floors[1] - 0.5), role="sign")
    socket("door", "attach", (dx, 0, zp), role="door")
