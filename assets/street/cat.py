"""Domestic cat for the square: sitting upright, loafing with the paws tucked in, or curled asleep.
The body is one continuous smooth-shaded skin: a signed-distance field of anatomical masses (leaning
spine with a ridge, shoulder blades, narrow chest, thighs with raised knees, long fore legs with elbows
and wrists, paws with four splayed toes, flat-crowned skull, cheeks, muzzle, a blended tail root) is
meshed with surface nets, snapped back onto the field, decimated (dense on the face and paws), relaxed
and ironed free of folded triangles. Fur: slim pointed tufts (7-17 mm, one triangle each) are rooted
in the skin and lie along the hair flow, lifted a few degrees; they share the skin's material, tint
and pattern, break up the silhouette on the cheek ruff, chest, haunches and tail (more and longer with
fluff) and feather the border of the white markings into lobes. The tail is a round tube that grows
out of a blended root, tapers to a rounded tip and lies pressed against the body (tail_wrap) or
trails in an S-curve on the stones. The coat is a matte woven-fibre material whose texture is sampled
along a single line of the weave, so it turns into fine strands that follow the hair flow (round the
spine and each limb, radial from the nose on the head); every tint change is a smooth per-vertex mask
(underfur break-up, paler belly / chest / muzzle, darker saddle and spine line, soft sheen on the
upward faces of black fur). The mackerel tabby pattern is a texture mask driven by one continuous
stripe coordinate solved over the skin (least-squares Poisson) from a prescribed band direction: the
skin is cut along the centre line of every bar, so narrow broken bars fall away from the spine down
the flanks, ring the legs and tail, run up the forehead as an 'M' and sweep back over the cheeks at
their real-world width whatever the size. White bib, socks, muzzle and tail tip are one field with a
lobed edge, cut into the mesh. Ears are cupped pinnae with a rolled rim (coat on the back, paler
hollow, soft furnishings at the inner base) that stay upright whatever the pitch of the head; eyes
are almond openings with a dark lid margin, a shallow-domed iris (darker limbal ring), round night
pupil (optional tapetum glow inside the pupil only) and a clear cornea; whiskers are pale hair-thin
tubes; the collar is an 11 mm woven strap fitted to the tightest closed section of the neck (2 mm
above the skin all round) with a brass buckle, keeper and bell.
"""
from tk import *
import numpy as np
from math import sin, cos, tan, radians, pi
from mathutils import Euler
from mathutils.bvhtree import BVHTree

META = {"name": "Sitting cat", "class": "prop_small", "mount": "ground", "tri_budget": 9000, "focus": 0.72, "auto_wear": 0.0, "auto_grime": 0.0}
PARAMS = {
    "pose": {"default": "sit", "choices": ["sit", "loaf", "curl"], "desc": "sitting upright / loaf with paws tucked / curled asleep"},
    "coat": {"default": "ginger_tabby", "choices": ["ginger_tabby", "black", "grey", "grey_tabby", "cream"], "desc": "coat colour / pattern"},
    "size": {"default": 1.0, "min": 0.7, "max": 1.3, "desc": "scale; 1 = average adult (0.44 m to the ear tips when sitting)"},
    "build": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "lean (0) .. stocky (1)"},
    "head_scale": {"default": 1.1, "min": 0.9, "max": 1.35, "desc": "head size relative to the body (kittens ~1.3)"},
    "fluff": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "fur length: sleek short hair .. fluffy (thicker coat, more and longer tufts)"},
    "white": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of white bib, socks and muzzle (0 = none)"},
    "head_turn": {"default": 18.0, "min": -65.0, "max": 65.0, "desc": "head yaw (deg, + = towards its left / +X)"},
    "tail_wrap": {"default": 0.9, "min": 0.0, "max": 1.0, "desc": "tail trailing loose in an S (0) .. wrapped tight round the body (1)"},
    "mirror": {"default": False, "desc": "mirror the pose left / right (tail side, curl direction)"},
    "eyes": {"default": "auto", "choices": ["auto", "open", "half", "closed"], "desc": "auto: sit = open, loaf = half, curl = closed"},
    "eye_colour": {"default": "#b5a83c", "desc": "iris colour"},
    "pupil": {"default": 0.8, "min": 0.1, "max": 1.0, "desc": "pupil dilation (slit .. round night pupil)"},
    "eyeshine": {"default": 0.0, "min": 0.0, "max": 1.0, "desc": "tapetum glow inside the pupils at night (0 = none)"},
    "collar": {"default": True, "desc": "woven collar strap with a brass buckle"},
    "collar_colour": {"default": "#7c2018", "desc": "collar colour"},
    "bell": {"default": True, "desc": "brass bell on the collar"},
}
VARIANTS = {
    "tuxedo_loaf": {"pose": "loaf", "coat": "black", "white": 0.85, "eye_colour": "#9dbb3c", "eyes": "open", "eyeshine": 0.25,
                    "collar_colour": "#b3261e", "head_turn": -28.0, "mirror": True, "build": 0.7, "seed": 4},
    "grey_curl": {"pose": "curl", "coat": "grey", "white": 0.0, "fluff": 0.75, "size": 1.1, "collar": False, "seed": 9},
    "tabby_kitten": {"coat": "grey_tabby", "size": 0.72, "white": 0.3, "head_turn": -34.0, "tail_wrap": 0.2, "mirror": True, "head_scale": 1.3, "pupil": 0.9,
                     "fluff": 0.55, "build": 0.2, "bell": False, "collar_colour": "#2f5a3a", "seed": 15},
    "cream_sleeper": {"pose": "loaf", "coat": "cream", "fluff": 1.0, "build": 0.8, "size": 1.15, "eyes": "closed", "white": 0.0,
                      "head_turn": -24.0, "collar": False, "seed": 23},
}

# tint = palest tone of the coat (belly, chest, muzzle) = colour of the material; ground = dark tone of the plain coat (0..1),
# pale = how far belly / chest are lifted back to the tint, saddle = darker back, brk = faint break-up, face = paler muzzle
# and eye rings, rim = lighter upward faces (the sheen of black fur), edge = tone at the border of a white marking,
# gap / hw = spacing and half width of the tabby bars (m), bar / spine = depth of the bars and of the dorsal line,
# skin = nose and ear hollow, rough = roughness multiplier, whisk = whisker radius (m; thinner on a pale coat)
COATS = {
    "ginger_tabby": {"tint": "#e3a05c", "ground": 0.12, "pale": 0.9, "saddle": 0.08, "brk": 0.06, "face": 0.6, "rim": 0.0, "edge": 0.30,
                     "gap": 0.029, "hw": 0.0030, "bar": 0.36, "spine": 0.40, "skin": "#dbb0a2", "rough": 1.0},
    "black": {"tint": "#5a504a", "ground": 0.62, "pale": 0.0, "saddle": 0.0, "brk": 0.08, "face": 0.0, "rim": 0.50, "edge": 0.80,
              "skin": "#33292b", "rough": 0.8},
    "grey": {"tint": "#8f949c", "ground": 0.20, "pale": 0.6, "saddle": 0.10, "brk": 0.07, "face": 0.4, "rim": 0.0, "edge": 0.25,
             "skin": "#a98a8c", "rough": 0.85},
    "grey_tabby": {"tint": "#b9ae9c", "ground": 0.14, "pale": 0.9, "saddle": 0.08, "brk": 0.06, "face": 0.6, "rim": 0.0, "edge": 0.25,
                   "gap": 0.027, "hw": 0.0028, "bar": 0.45, "spine": 0.50, "skin": "#cfaea6", "rough": 1.0},
    "cream": {"tint": "#ecd3a8", "ground": 0.10, "pale": 0.8, "saddle": 0.08, "brk": 0.05, "face": 0.6, "rim": 0.0, "edge": 0.10,
              "skin": "#e6b0a4", "rough": 1.0, "whisk": 0.00040},
}
WHITE = "#f6f2ea"
FUR_TEX = 4.0                                      # texture scale of the coat material (fine fibres)
STRETCH = 4.0                                      # the fibres are drawn out this much along the hair flow
TIP_KEEP = 0.35                                    # share of its stand-off a lock keeps where the coat curves away under it
FEATHER_W = 0.005                                  # soft inner edge of a white marking (m)
BAR_TRIS = 1500                                    # triangles a tabby spends on its bars
H_GRID = 0.0024                                    # field sampling step (m)
TRI_CAP = 8900
RELAX = 3

# ------------------------------------------------------------------ signed distance toolkit (numpy)
def _smin(a, b, k):
    if k <= 1e-6:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * (k * 0.25)

def _smax(a, b, k):
    return -_smin(-a, -b, k)

def _rot(rx=0.0, ry=0.0, rz=0.0, order="XYZ"):
    return np.array(Euler((radians(rx), radians(ry), radians(rz)), order).to_matrix(), np.float32)

def _ell(c, r, R=None):
    c = np.asarray(c, np.float32)
    r = np.asarray(r, np.float32)
    rmin = float(r.min())

    def fn(P):
        q = P - c
        if R is not None:
            q = q @ R
        k0 = np.sqrt(((q / r) ** 2).sum(-1))
        k1 = np.sqrt(((q / (r * r)) ** 2).sum(-1)) + 1e-9
        return np.where(k0 < 0.5, (k0 - 1.0) * rmin, k0 * (k0 - 1.0) / k1)

    rad = float(r.max())
    return fn, c - rad, c + rad

def _chain(pts, radii):
    pts = np.asarray(pts, np.float32)
    radii = np.asarray(radii, np.float32)

    def fn(P):
        d = None
        for i in range(len(pts) - 1):
            a = pts[i]
            ab = pts[i + 1] - a
            t = np.clip(((P - a) @ ab) / float(ab @ ab + 1e-12), 0.0, 1.0)
            di = np.linalg.norm(P - a - t[..., None] * ab, axis=-1) - (radii[i] + (radii[i + 1] - radii[i]) * t)
            d = di if d is None else np.minimum(d, di)
        return d

    rm = float(radii.max())
    return fn, pts.min(0) - rm, pts.max(0) + rm

def _path(ctrl, radii, n):
    """Smooth resampled centre line + radii interpolated along it."""
    pts = np.array([tuple(q) for q in resample_path([tuple(float(v) for v in c) for c in ctrl], n=n)], np.float32)
    c = np.asarray(ctrl, np.float32)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    sp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    r = np.interp(sp / sp[-1], s / s[-1], radii).astype(np.float32)
    return pts, r

class _Field:
    def __init__(self):
        self.ops = []

    def add(self, prim, k=0.02):
        self.ops.append(("u", prim, float(k)))

    def cut(self, prim, k=0.004):
        self.ops.append(("s", prim, float(k)))

    def inflate(self, amount):
        self.ops.append(("o", float(amount), 0.0))

    def at(self, P, ground=True):
        P = np.asarray(P, np.float32)
        d = np.full(P.shape[:-1], 1.0, np.float32)
        for kind, a, k in self.ops:
            if kind == "o":
                d = d - a
            elif kind == "u":
                d = _smin(d, a[0](P), k)
            else:
                d = _smax(d, -a[0](P), k)
        return np.maximum(d, -P[..., 2]) if ground else d

    def grad(self, P, eps=0.0012, ground=True):
        P = np.asarray(P, np.float32)
        g = []
        for i in range(3):
            e = np.zeros(3, np.float32)
            e[i] = eps
            g.append((self.at(P + e, ground) - self.at(P - e, ground)) / (2 * eps))
        return np.stack(g, -1)

    def normal(self, P, eps=0.0022):
        """Shading normal of the skin: gradient of the field without the ground plane, taken over a few millimetres
        so that the decimated triangles never show."""
        return _unit(self.grad(P, eps, False))

    def project(self, P, it=3, off=0.0):
        P = np.array(P, np.float32)
        for _ in range(it):
            d = self.at(P)
            g = self.grad(P)
            g2 = (g * g).sum(-1, keepdims=True) + 1e-9
            P = P - np.clip(d[..., None] * g / g2, -0.004, 0.004)
        if off:
            g = self.grad(P)
            P = P + off * g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)
        return P

    def hit(self, a, b):
        """First surface crossing going from a (inside) to b (outside)."""
        a = np.array(a, np.float32)
        b = np.array(b, np.float32)
        ts = np.linspace(0.0, 1.0, 48).astype(np.float32)
        d = self.at(a[None] + ts[:, None] * (b - a)[None])
        out = d > 0
        if not out.any():
            return b
        i = int(np.argmax(out))
        lo, hi = float(ts[max(i - 1, 0)]), float(ts[i])
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if float(self.at(a + mid * (b - a))) > 0:
                hi = mid
            else:
                lo = mid
        return a + 0.5 * (lo + hi) * (b - a)

    def hits(self, A, B, n=40, it=11):
        """Vectorised hit(): rays A[i] (inside) -> B[i]. Returns (points, ok)."""
        A = np.asarray(A, np.float32)
        B = np.asarray(B, np.float32)
        ts = np.linspace(0.0, 1.0, n).astype(np.float32)
        out = self.at(A[:, None, :] + ts[None, :, None] * (B - A)[:, None, :]) > 0
        ok = out.any(1) & ~out[:, 0]
        i = np.argmax(out, 1)
        lo, hi = ts[np.maximum(i - 1, 0)], ts[i]
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            o = self.at(A + mid[:, None] * (B - A)) > 0
            hi = np.where(o, mid, hi)
            lo = np.where(o, lo, mid)
        return A + (0.5 * (lo + hi))[:, None] * (B - A), ok

    def grid(self, h):
        los = np.array([a[1] for kind, a, k in self.ops if kind == "u"])
        his = np.array([a[2] for kind, a, k in self.ops if kind == "u"])
        lo = (los.min(0) - 0.02).astype(np.float32)
        hi = (his.max(0) + 0.02).astype(np.float32)
        lo[2] = -2.5 * h
        n = np.ceil((hi - lo) / h).astype(int) + 1
        ax = [lo[i] + h * np.arange(n[i], dtype=np.float32) for i in range(3)]
        D = np.full(tuple(n), 1.0, np.float32)
        for kind, a, k in self.ops:
            if kind == "o":
                D -= a
                continue
            pad = 2 * k + 0.02 + 2 * h
            i0 = np.clip(np.floor((a[1] - pad - lo) / h).astype(int), 0, n - 1)
            i1 = np.clip(np.ceil((a[2] + pad - lo) / h).astype(int) + 1, 1, n)
            sl = tuple(slice(int(i0[i]), int(i1[i])) for i in range(3))
            X, Y, Z = np.meshgrid(ax[0][sl[0]], ax[1][sl[1]], ax[2][sl[2]], indexing="ij")
            di = a[0](np.stack([X, Y, Z], -1))
            D[sl] = _smin(D[sl], di, k) if kind == "u" else _smax(D[sl], -di, k)
        D = np.maximum(D, -ax[2][None, None, :])
        return D, lo

def _surface_nets(D, lo, h):
    """Naive surface nets (vectorised): one vertex per sign-changing cell, one quad per crossing edge."""
    nx, ny, nz = D.shape
    ins = D < 0
    cnt = np.zeros((nx - 1, ny - 1, nz - 1), np.int8)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                cnt += ins[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz]
    act = (cnt > 0) & (cnt < 8)
    acc = np.zeros((nx - 1, ny - 1, nz - 1, 3), np.float32)
    num = np.zeros((nx - 1, ny - 1, nz - 1), np.float32)
    edges = []
    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax] = slice(0, -1)
        s1[ax] = slice(1, None)
        F0 = D[tuple(s0)]
        F1 = D[tuple(s1)]
        cross = (F0 < 0) != (F1 < 0)
        t = np.where(cross, F0 / np.where(cross, F0 - F1, 1.0), 0.0).astype(np.float32)
        shp = F0.shape
        idx = np.meshgrid(np.arange(shp[0], dtype=np.float32), np.arange(shp[1], dtype=np.float32),
                          np.arange(shp[2], dtype=np.float32), indexing="ij")
        idx[ax] = idx[ax] + t
        o1, o2 = [a for a in range(3) if a != ax]
        w = cross.astype(np.float32)
        for d1 in (0, 1):
            for d2 in (0, 1):
                es = [slice(None)] * 3
                es[o1] = slice(d1, shp[o1] - 1 + d1)
                es[o2] = slice(d2, shp[o2] - 1 + d2)
                es = tuple(es)
                num += w[es]
                for c in range(3):
                    acc[..., c] += idx[c][es] * w[es]
        edges.append((ax, o1, o2, np.argwhere(cross), F0))
    vid = np.full(act.shape, -1, np.int64)
    vid[act] = np.arange(int(act.sum()))
    V = acc[act] / num[act][:, None] * h + lo
    quads = []
    for ax, o1, o2, ci, F0 in edges:
        m = (ci[:, o1] >= 1) & (ci[:, o1] <= D.shape[o1] - 2) & (ci[:, o2] >= 1) & (ci[:, o2] <= D.shape[o2] - 2)
        ci = ci[m]

        def cell(d1, d2):
            c = ci.copy()
            c[:, o1] -= d1
            c[:, o2] -= d2
            return vid[c[:, 0], c[:, 1], c[:, 2]]

        q = np.stack([cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)], 1)
        flip = F0[ci[:, 0], ci[:, 1], ci[:, 2]] < 0
        if ax == 1:
            flip = ~flip
        q[~flip] = q[~flip][:, ::-1]
        quads.append(q)
    return V.astype(np.float32), np.concatenate(quads, 0)

# ------------------------------------------------------------------ small mesh helpers
def _blob(name, c, r, R=None, seg=12, rings=8, cap=1.0):
    """Ellipsoid (cap < 1: only the polar cap around local +Z, open at the back)."""
    vs = [(0.0, 0.0, 1.0)]
    for i in range(1, rings + (1 if cap < 1.0 else 0)):
        ph = cap * pi * i / rings
        for j in range(seg):
            th = 2 * pi * j / seg
            vs.append((sin(ph) * cos(th), sin(ph) * sin(th), cos(ph)))
    vs.append((0.0, 0.0, -1.0))
    V = np.array(vs, np.float32) * np.asarray(r, np.float32)
    if R is not None:
        V = V @ R.T
    V = V + np.asarray(c, np.float32)
    faces = [(0, 1 + j, 1 + (j + 1) % seg) for j in range(seg)]
    for i in range(rings - (1 if cap < 1.0 else 2)):
        a = 1 + i * seg
        b = a + seg
        for j in range(seg):
            j2 = (j + 1) % seg
            faces.append((a + j, b + j, b + j2, a + j2))
    if cap < 1.0:
        return mesh_object(name, V[:-1].tolist(), faces)
    last = len(vs) - 1
    a = 1 + (rings - 2) * seg
    faces += [(last, a + (j + 1) % seg, a + j) for j in range(seg)]
    return mesh_object(name, V.tolist(), faces)

def _relax(F, V, T, wgt, it=3):
    """Even out the decimated triangles (weighted umbrella smoothing) and snap them back onto the skin."""
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = np.unique(np.sort(E, 1), axis=0)
    cnt = np.zeros(len(V), np.float32)
    np.add.at(cnt, E[:, 0], 1.0)
    np.add.at(cnt, E[:, 1], 1.0)
    mv = wgt > 0.01
    for _ in range(it):
        acc = np.zeros_like(V)
        np.add.at(acc, E[:, 0], V[E[:, 1]])
        np.add.at(acc, E[:, 1], V[E[:, 0]])
        Vn = V + (acc / np.maximum(cnt, 1.0)[:, None] - V) * (0.5 * wgt)[:, None]
        V = V.copy()
        V[mv] = F.project(Vn[mv], it=2)
    return V

def _hairs(name, strands, sides=3):
    V, Fc = [], []
    for pts, r0, r1 in strands:
        pts = np.asarray(pts, np.float32)
        n = len(pts)
        base = len(V)
        for i in range(n):
            t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            t = t / (np.linalg.norm(t) + 1e-9)
            u = np.cross(t, (0.0, 0.0, 1.0))
            if np.linalg.norm(u) < 1e-3:
                u = np.cross(t, (1.0, 0.0, 0.0))
            u = u / np.linalg.norm(u)
            v = np.cross(t, u)
            r = r0 + (r1 - r0) * i / (n - 1)
            for s in range(sides):
                a = 2 * pi * s / sides
                V.append(tuple(float(x) for x in (pts[i] + r * (cos(a) * u + sin(a) * v))))
        for i in range(n - 1):
            for s in range(sides):
                s2 = (s + 1) % sides
                Fc.append((base + i * sides + s, base + i * sides + s2, base + (i + 1) * sides + s2, base + (i + 1) * sides + s))
        Fc.append(tuple(base + s for s in range(sides))[::-1])
        Fc.append(tuple(base + (n - 1) * sides + s for s in range(sides)))
    obj = mesh_object(name, V, Fc)
    recalc_normals(obj)
    return obj

def _nrm(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def _unit(A):
    A = np.asarray(A, np.float32)
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-9)

def _sstep(a, b, x):
    t = np.clip((np.asarray(x, np.float32) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _pv(arr, base=None):
    """paint() callable returning a precomputed per-vertex array (plus an optional position based fn)."""
    arr = np.asarray(arr, np.float32)

    def fn(co, nr):
        b = base(co, nr) if base is not None else 0.0
        if len(co) != len(arr):
            return np.clip(b + np.zeros(len(co), np.float32), 0.0, 1.0)
        return np.clip(arr + b, 0.0, 1.0)

    return fn

def _split_iso(V, T, fn):
    """Cut triangles along the zero contour of the scalar field fn (solved on the edges, not interpolated)
    so a coat patch gets a smooth border. Returns (verts, tris, label: 1 where fn < 0, parent triangle)."""
    V = np.asarray(V, np.float32)
    T = np.asarray(T, np.int64)
    f = fn(V)
    neg = f < 0
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    E = E[neg[E[:, 0]] != neg[E[:, 1]]]
    if len(E) == 0:
        return V, T, neg[T].all(1).astype(np.int32), np.arange(len(T))
    a = np.where(neg[E[:, 0]], E[:, 0], E[:, 1])
    b = np.where(neg[E[:, 0]], E[:, 1], E[:, 0])
    lo = np.zeros(len(E), np.float32)
    hi = np.ones(len(E), np.float32)
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        ng = fn(V[a] + mid[:, None] * (V[b] - V[a])) < 0
        lo = np.where(ng, mid, lo)
        hi = np.where(ng, hi, mid)
    t = np.clip(0.5 * (lo + hi), 0.07, 0.93)
    nv = V[a] + t[:, None] * (V[b] - V[a])
    idx = {(int(min(i, j)), int(max(i, j))): len(V) + k for k, (i, j) in enumerate(zip(a, b))}
    V2 = np.concatenate([V, nv])
    oT, oM, par = [], [], []
    for ti, (a_, b_, c_) in enumerate(T.tolist()):
        sa, sb, sc = bool(neg[a_]), bool(neg[b_]), bool(neg[c_])
        n = sa + sb + sc
        if n == 0 or n == 3:
            oT.append((a_, b_, c_))
            oM.append(1 if n == 3 else 0)
            par.append(ti)
            continue
        if sa == sb:
            a_, b_, c_ = c_, a_, b_
        elif sa == sc:
            a_, b_, c_ = b_, c_, a_
        lone = 1 if neg[a_] else 0
        ab = idx[(min(a_, b_), max(a_, b_))]
        ac = idx[(min(a_, c_), max(a_, c_))]
        oT += [(a_, ab, ac), (ab, b_, c_), (ab, c_, ac)]
        oM += [lone, 1 - lone, 1 - lone]
        par += [ti, ti, ti]
    return V2, np.array(oT, np.int64), np.array(oM, np.int32), np.array(par, np.int64)

def _disc(name, c, n, up, rx, rz, sph=0.0, seg=12, rings=2):
    """Elliptical disc facing n (radii rx across / rz up), wrapped onto a sphere of radius sph (0 = flat)."""
    c = np.asarray(c, np.float32)
    n = _nrm(n)
    lat = _nrm(np.cross(up, n))
    upv = np.cross(n, lat)
    vs = [c]
    for i in range(1, rings + 1):
        f = i / rings
        for j in range(seg):
            a = 2 * pi * j / seg
            x, z = f * rx * cos(a), f * rz * sin(a)
            vs.append(c + x * lat + z * upv - n * ((x * x + z * z) / (2 * sph) if sph else 0.0))
    faces = [(0, 1 + j, 1 + (j + 1) % seg) for j in range(seg)]
    for i in range(rings - 1):
        a = 1 + i * seg
        b = a + seg
        for j in range(seg):
            j2 = (j + 1) % seg
            faces.append((a + j, b + j, b + j2, a + j2))
    return mesh_object(name, [tuple(float(x) for x in v) for v in vs], faces)

def _tri_mesh(name, V, T):
    return mesh_object(name, np.asarray(V, np.float32).tolist(), np.asarray(T, np.int64).tolist())

# ------------------------------------------------------------------ hair flow: skeleton chains -> stripe coordinate, UVs, direction
def _fchain(C, pts, rad, ref, u0=None, use="all"):
    pts = np.asarray(pts, np.float32)
    rad = np.broadcast_to(np.asarray(rad, np.float32), (len(pts),)).copy()
    S = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]).astype(np.float32)
    k = int(round(float(S[-1]) / 0.004)) + 1
    if k > len(pts) and len(pts) > 2:                       # fine smooth curve: the arc length has no kinks at the joints
        fine = np.array([tuple(q) for q in resample_path([tuple(float(v) for v in q) for q in pts], n=k)], np.float32)
        Sf = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))]).astype(np.float32)
        rad = np.interp(Sf / Sf[-1], S / S[-1], rad).astype(np.float32)
        pts, S = fine, Sf
    rbar = float(rad.mean())
    C.flow.append(dict(pts=pts, rad=rad, ref=_nrm(ref), S=S, rbar=rbar, u0=u0, use=use))
    return C.flow[-1]

def _chain_eval(ch, P):
    """Capsule distance, arc length (extended past the ends), angle round the axis and tangent of the
    closest chain segment for the points P."""
    pts, rad, ref, S = ch["pts"], ch["rad"], ch["ref"], ch["S"]
    P = np.asarray(P, np.float32)
    n = len(pts)
    N = len(P)
    best = np.full(N, 1e9, np.float32)
    s = np.zeros(N, np.float32)
    th = np.zeros(N, np.float32)
    tan_ = np.zeros((N, 3), np.float32)
    for i in range(n - 1):
        a = pts[i]
        ab = pts[i + 1] - a
        L = float(np.linalg.norm(ab)) + 1e-9
        t_ = ab / L
        tr = ((P - a) @ t_) / L
        tc = np.clip(tr, 0.0, 1.0)
        q = P - a - tc[:, None] * ab
        dc = np.linalg.norm(q, axis=1) - (rad[i] + (rad[i + 1] - rad[i]) * tc)
        te = np.clip(tr, -0.3 / L if i == 0 else 0.0, 1.0 + 0.3 / L if i == n - 2 else 1.0)
        e1 = ref - float(ref @ t_) * t_
        e1 = e1 / (np.linalg.norm(e1) + 1e-9)
        e2 = np.cross(t_, e1)
        m = dc < best
        best = np.where(m, dc, best)
        s = np.where(m, S[i] + te * L, s)
        th = np.where(m, np.arctan2(q @ e2, q @ e1), th)
        tan_[m] = t_
    return best, s, th, tan_

def _chain_s(ch, P, sig=0.035):
    """Arc length along a chain as a smooth function of space: the projections onto all its segments, blended with
    Gaussian weights. Unlike the closest-point arc length it has no jumps or flats far from the axis (haunches,
    belly), so the bands interpolate cleanly over the triangles of the skin."""
    pts, S = ch["pts"], ch["S"]
    P = np.asarray(P, np.float32)
    mid = 0.5 * (pts[1:] + pts[:-1])
    t = _unit(pts[1:] - pts[:-1])
    Sm = 0.5 * (S[1:] + S[:-1])
    out = np.zeros(len(P), np.float32)
    for i0 in range(0, len(P), 16000):
        d = P[i0:i0 + 16000, None, :] - mid[None]
        d2 = (d * d).sum(-1)
        w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2.0 * sig * sig))
        out[i0:i0 + 16000] = (w * (Sm[None] + (d * t[None]).sum(-1))).sum(1) / w.sum(1)
    return out

def _chain_t(ch, P, sig=0.05):
    """Smooth unit tangent field of a chain (Gaussian blend of the segment directions)."""
    pts = ch["pts"]
    P = np.asarray(P, np.float32)
    mid = 0.5 * (pts[1:] + pts[:-1])
    t = _unit(pts[1:] - pts[:-1])
    out = np.zeros((len(P), 3), np.float32)
    for i0 in range(0, len(P), 16000):
        d = P[i0:i0 + 16000, None, :] - mid[None]
        d2 = (d * d).sum(-1)
        w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2.0 * sig * sig))
        out[i0:i0 + 16000] = w @ t
    return _unit(out)

HEAD_N = np.array([0.0, -0.105, -0.012], np.float32)       # pole of the head pattern: in front of the nose leather
HEAD_E = np.array([0.060, 0.062, 0.050], np.float32)

def _head_eval(ch, P):
    """Head: distance to the skull, meridian angle round the line nose -> nape (signed by the side), head-space
    position, distance from that line (m) and weight of the head region. The meridians leave the nose bridge, run
    up the forehead and back over the crown, and sweep back along the cheeks: the lie of the hair on a cat's head."""
    c, R, hs = ch["c"], ch["R"], ch["hs"]
    q = ((np.asarray(P, np.float32) - c) @ R) / hs
    k0 = np.sqrt(((q / HEAD_E) ** 2).sum(1))
    r = q - HEAD_N
    e = _nrm(np.array([0.0, 0.05, -0.06], np.float32) - HEAD_N)
    wd = np.array([0.0, -e[2], e[1]], np.float32)
    wd = wd if wd[2] > 0 else -wd
    h = r @ wd
    return (k0 - 1.0) * 0.055 * hs, np.arctan2(q[:, 0], h), q, np.hypot(q[:, 0], h) * hs, 1.0 - _sstep(1.25, 1.8, k0)

def _stripe_solve(C, V, T):
    """Coordinate that carries the tabby bars over the body, solved on the skin. Arc length along the nearest chain
    would crowd the bars into a fan wherever the chain bends (haunch of a sitting cat, throat), so only the bar
    DIRECTION is prescribed: a unit gradient per triangle (planar bands across a straight spine that fall away down
    both flanks, the smooth tangent of a curled spine, of a free leg near it) and one continuous coordinate is
    integrated from it in the least-squares sense (Poisson problem on the mesh)."""
    T = np.asarray(T, np.int64)
    P = np.asarray(V, np.float32)
    nv = len(P)
    P64 = P.astype(np.float64)
    p0, p1, p2 = P64[T[:, 0]], P64[T[:, 1]], P64[T[:, 2]]
    nn = np.cross(p1 - p0, p2 - p0)
    A2 = np.maximum(np.linalg.norm(nn, axis=1), 1e-12)
    nu = nn / A2[:, None]
    G = np.stack([np.cross(nu, p2 - p1), np.cross(nu, p0 - p2), np.cross(nu, p1 - p0)], 1) / A2[:, None, None]   # hat function gradients
    A = 0.5 * A2
    Pc = P[T].mean(1)
    W, D = [], []
    body = C.flow[0]
    limbs = [ch for ch in C.flow[1:] if not ch.get("head") and ch["use"] == "all"]
    chord = body["pts"][-1] - body["pts"][0]
    if float(np.linalg.norm(chord)) > 0.8 * float(body["S"][-1]):          # fairly straight spine: flat parallel bands, shared with the legs
        t0 = _nrm(chord)
        if limbs:
            tl = _nrm(sum(_nrm(ch["pts"][-1] - ch["pts"][0]) for ch in limbs))
            t0 = _nrm(t0 + (tl if float(tl @ t0) > 0 else -tl))
        W.append(np.ones(len(T), np.float64))
        pm = body["pts"].mean(0)
        sv = np.linalg.svd((body["pts"] - pm).astype(np.float64))[2]
        nl = sv[2] if abs(float(sv[2] @ t0)) < 0.3 else np.cross(t0, (0.0, 0.0, 1.0))
        nl = nl - float(nl @ t0) * t0.astype(np.float64)
        nl = nl / (np.linalg.norm(nl) + 1e-9)                # across the body
        xl = (Pc.astype(np.float64) - pm) @ nl
        D.append(t0.astype(np.float64)[None] - 0.32 * (xl / np.sqrt(xl * xl + 0.004 ** 2))[:, None] * nl[None])   # the bars fall away from the spine on both flanks
    else:
        W.append(np.ones(len(T), np.float64))
        D.append(_chain_t(body, Pc, 0.08).astype(np.float64))
        for ch in limbs:
            W.append(8.0 * np.exp(-np.maximum(_chain_eval(ch, Pc)[0], 0.0).astype(np.float64) / 0.004))
            D.append(_chain_t(ch, Pc, 0.02).astype(np.float64))
    W = np.stack(W)
    W = W / W.sum(0)
    g = sum(w[:, None] * d for w, d in zip(W, D))
    g = g - (g * nu).sum(1, keepdims=True) * nu
    M = A[:, None, None] * (G[:, :, None, :] * G[:, None, :, :]).sum(-1)
    L = np.zeros((nv, nv))
    np.add.at(L, (T[:, :, None], T[:, None, :]), M)
    b = np.zeros(nv)
    np.add.at(b, T, A[:, None] * (G * g[:, None, :]).sum(-1))
    va = np.zeros(nv)
    np.add.at(va, T, A[:, None] / 3.0)
    pin = 4.0 * va + 1e-12                                  # only fixes the free constant (and loose pieces)
    L[np.arange(nv), np.arange(nv)] += pin
    b += pin * _chain_s(C.flow[0], P)
    return np.linalg.solve(L, b).astype(np.float32)

def _own(ch, d, first, tube=False):
    """Distance used to pick the chain that combs a piece of skin: a limb only owns the skin right on it and the
    tail (a mesh of its own) none, so a haunch or a foot lying next to them follows the spine in one sweep instead
    of a patchwork of three hair directions."""
    if first or tube or ch.get("head"):
        return d
    if ch["use"] == "tube":
        return np.full_like(d, 1e3)
    return np.where(d < 0.014, d, 1e3)

def _across(C, V, T):
    """Per-corner coordinates ACROSS the hair flow (m; angle x radius round the nearest chain, meridian angle on the
    head) and ALONG it (arc length of that chain, distance from the nose on the head). Each triangle takes both
    from one chain; the seams between chains run along the hair, where the drawn-out fibres hide them."""
    V = np.asarray(V, np.float32)
    D, TH, SS, RB = [], [], [], []
    for i, ch in enumerate(C.flow):
        if ch.get("head"):
            d, a, _q, _rho, _w = _head_eval(ch, V)
            TH.append(a)
            SS.append(np.linalg.norm(V - ch["radial"], axis=1))
            RB.append(ch["rbar"])
        else:
            d, s, th, _tn = _chain_eval(ch, V)
            TH.append(th)
            SS.append(s)
            RB.append(ch["rbar"])
        D.append(_own(ch, d, i == 0))
    D, TH, SS = np.stack(D), np.stack(TH), np.stack(SS)
    kf = np.argmin(D[:, T].mean(2), 0)
    th = TH[kf[:, None], T]
    thw = th[:, :1] + (th - th[:, :1] + pi) % (2 * pi) - pi
    return thw * np.array(RB, np.float32)[kf][:, None], SS[kf[:, None], T]

def _near(u, cen):
    """Index of the bar (sorted centres cen) nearest to each coordinate."""
    k = np.clip(np.searchsorted(cen, u), 1, len(cen) - 1)
    return np.where(u - cen[k - 1] < cen[k] - u, k - 1, k)

def _bar_val(d, a, e):
    """Profile of a tabby bar at distance d from its centre line: full up to the half width a, fading evenly to
    nothing at e (the skin has vertices on all four of these lines, so the fade is exactly the feathered edge)."""
    return np.clip((e - np.abs(d)) / np.maximum(e - a, 1e-6), 0.0, 1.0)

def _bar_lines(cen, a):
    """Both edge lines of the core of every bar as one sorted list (bar k owns the lines 2k, 2k + 1) and the half
    widths, limited so that neighbouring bars keep a clear gap."""
    gap = np.diff(cen)
    a = np.minimum(a, 0.28 * np.minimum(np.concatenate([gap[:1], gap]), np.concatenate([gap, gap[-1:]]))).astype(np.float32)
    return np.stack([cen - a, cen + a], 1).ravel().astype(np.float32), a

def _feather(V, T, on, fmax, free):
    """Soft edge of the bars: every vertex next to a bar (on = vertices of the bars) is drawn in to at most fmax
    from it, so the tint fades from the bar to the plain coat over 2 - 6 mm whatever the size of the triangles.
    Returns the moved vertices and which ones moved."""
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    E = np.concatenate([E, E[:, ::-1]])
    E = E[~on[E[:, 0]] & on[E[:, 1]]]
    acc, cnt = np.zeros_like(V), np.zeros(len(V), np.float32)
    np.add.at(acc, E[:, 0], V[E[:, 1]])
    np.add.at(cnt, E[:, 0], 1.0)
    Q = acc / np.maximum(cnt, 1.0)[:, None]
    d = np.linalg.norm(V - Q, axis=1)
    mv = (cnt > 0) & (d > fmax) & free
    V = V.copy()
    V[mv] = Q[mv] + (V[mv] - Q[mv]) * ((fmax + 0.0 * d)[mv] / d[mv])[:, None]
    return V, mv

def _squeeze(u, cen, k, s):
    """Coordinate across the bars with every bar narrowed by the factor s (per point): the lines of bar k drawn
    in this coordinate run together where s gets small, so a bar tapers to a point where it fades out."""
    gap = np.diff(cen)
    lim = 0.49 * np.minimum(np.concatenate([gap[:1], gap]), np.concatenate([gap, gap[-1:]]))[k]
    return (cen[k] + np.clip((u - cen[k]) / s, -lim, lim)).astype(np.float32)

def _split_edges(V, T, cut):
    """Re-triangulate after edges got a new vertex (cut: sorted vertex pair -> new vertex). Returns tris, parents."""
    oT, par = [], []
    for ti, (a, b, c) in enumerate(T.tolist()):
        ab, bc, ca = cut.get((min(a, b), max(a, b))), cut.get((min(b, c), max(b, c))), cut.get((min(c, a), max(c, a)))
        n = (ab is not None) + (bc is not None) + (ca is not None)
        if n == 0:
            oT.append((a, b, c))
        elif n == 1:
            if bc is not None:
                a, b, c, ab = b, c, a, bc
            elif ca is not None:
                a, b, c, ab = c, a, b, ca
            oT += [(a, ab, c), (ab, b, c)]
        elif n == 2:
            if ab is None:                                   # lone corner c
                a, b, c, ab, ca = c, a, b, ca, bc
            elif ca is None:                                 # lone corner b
                a, b, c, ab, ca = b, c, a, bc, ab
            if np.linalg.norm(V[ab] - V[c]) < np.linalg.norm(V[b] - V[ca]):
                oT += [(a, ab, ca), (ab, b, c), (ab, c, ca)]
            else:
                oT += [(a, ab, ca), (ab, b, ca), (b, c, ca)]
        else:
            oT += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        par += [ti] * (n + 1)
    return np.array(oT, np.int64), np.array(par, np.int64)

def _ridge_cut(V, T, coord, cen, present, extra=()):
    """Cut the skin along the centre lines coord = cen[k] of the bars (where present(P, k)), so each bar gets a
    row of vertices exactly on its centre line. Vertices already close to a line are taken onto it instead of
    making sliver triangles. Returns verts, tris, parent triangle, coord and the extra per-vertex arrays."""
    coord = np.array(coord, np.float32)
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)

    def crossing():
        ua, ub = coord[E[:, 0]], coord[E[:, 1]]
        lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
        k = np.clip(np.searchsorted(cen, lo, side="right"), 0, len(cen) - 1)
        ok = (cen[k] > lo) & (cen[k] < hi)
        t = np.where(ok, (cen[k] - ua) / np.where(ok, ub - ua, 1.0), 0.5)
        return ok, k, t

    ok, k, t = crossing()
    Pm = V[E[:, 0]] + t[:, None] * (V[E[:, 1]] - V[E[:, 0]])
    ok &= present(Pm, k)
    near = ok & ((t < 0.2) | (t > 0.8))
    vi, first = np.unique(np.where(t[near] < 0.5, E[near, 0], E[near, 1]), return_index=True)
    V = np.array(V, np.float32)
    V[vi] = Pm[near][first]                                  # the vertex slides along its edge onto the line
    coord[vi] = cen[k[near][first]]
    ne = np.flatnonzero(near)[first]
    extra = [np.array(x, np.float32) for x in extra]
    for x in extra:
        x[vi] = x[E[ne, 0]] + t[ne] * (x[E[ne, 1]] - x[E[ne, 0]])
    ok2, k, t = crossing()
    ok = ok & ok2
    Pm = (V[E[:, 0]] + t[:, None] * (V[E[:, 1]] - V[E[:, 0]]))[ok]
    cut = {(int(i), int(j)): len(V) + n for n, (i, j) in enumerate(E[ok].tolist())}
    V2 = np.concatenate([V, Pm]).astype(np.float32)
    T2, par = _split_edges(V2, T, cut)
    ex = [np.concatenate([x, x[E[ok, 0]] + t[ok] * (x[E[ok, 1]] - x[E[ok, 0]])]).astype(np.float32) for x in extra]
    return V2, T2, par, np.concatenate([coord, cen[k[ok]]]).astype(np.float32), ex

def _bar_list(rng, lo, hi, gap, jit, hw):
    cen, hws = [], []
    u = lo + rng.u(0.2, 1.0) * gap
    while u < hi:
        cen.append(u)
        hws.append(hw * rng.u(0.75, 1.3))
        u += gap * (1.0 + rng.u(-jit, jit))
    if len(cen) < 2:
        cen, hws = [lo, hi], [hw, hw]
    return np.array(cen, np.float32), np.array(hws, np.float32)

def _iso_lines(V, T, coord, cen, span):
    """Level lines coord = cen[k] traced over the mesh (triangles whose coord varies by more than span are skipped).
    Returns a list of (k, points along the line)."""
    c = coord[T]
    lo, hi = c.min(1), c.max(1)
    k0 = np.searchsorted(cen, lo, side="right")
    k1 = np.searchsorted(cen, hi, side="left")
    nxt, pos = {}, {}
    for ti in np.flatnonzero((k1 > k0) & (hi - lo < span)).tolist():
        a, b, c_ = (int(x) for x in T[ti])
        for k in range(int(k0[ti]), int(k1[ti])):
            lv = float(cen[k])
            pts = []
            for i, j in ((a, b), (b, c_), (c_, a)):
                ui, uj = float(coord[i]), float(coord[j])
                if (ui < lv) != (uj < lv):
                    key = (k, min(i, j), max(i, j))
                    if key not in pos:
                        pos[key] = V[i] + (lv - ui) / (uj - ui) * (V[j] - V[i])
                    pts.append(key)
            if len(pts) == 2:
                nxt.setdefault(pts[0], []).append(pts[1])
                nxt.setdefault(pts[1], []).append(pts[0])
    lines, seen = [], set()
    for start in [q for q, n in nxt.items() if len(n) == 1] + list(nxt.keys()):
        if start in seen:
            continue
        chain, cur = [start], start
        seen.add(start)
        while True:
            nx = [q for q in nxt[cur] if q not in seen]
            if not nx:
                break
            cur = nx[0]
            seen.add(cur)
            chain.append(cur)
        if len(chain) >= 3:
            lines.append((chain[0][0], np.array([pos[q] for q in chain], np.float32)))
    return lines

def _runs(on):
    """(start, end) of every run of at least two True values."""
    out, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= 2:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out

def _curve(S, P, s):
    """Smooth curve through the points P (arc lengths S) at the arc lengths s (Catmull-Rom)."""
    S = np.asarray(S, np.float32)
    s = np.clip(np.asarray(s, np.float32), S[0], S[-1])
    i = np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(S) - 2)
    ia, ib = np.maximum(i - 1, 0), np.minimum(i + 2, len(S) - 1)
    h = S[i + 1] - S[i]
    t = ((s - S[i]) / h)[:, None]
    m0 = (P[i + 1] - P[ia]) * (h / (S[i + 1] - S[ia]))[:, None]
    m1 = (P[ib] - P[i]) * (h / (S[ib] - S[i]))[:, None]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * P[i] + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * P[i + 1] + (t3 - t2) * m1).astype(np.float32)

def _flow_dir(C, P, nrm, fall=0.22, tube=False):
    """Unit hair direction on the surface (tangent of the nearest chain, radial from the nose on the head)."""
    P = np.asarray(P, np.float32)
    N = len(P)
    best = np.full(N, 1e9, np.float32)
    tan_ = np.zeros((N, 3), np.float32)
    for i, ch in enumerate(C.flow):
        if ch.get("head"):
            dc = _head_eval(ch, P)[0]
            tn = _unit(P - ch["radial"])
        else:
            dc, _s, _th, tn = _chain_eval(ch, P)
        dc = _own(ch, dc, i == 0, tube)
        m = dc < best
        best = np.where(m, dc, best)
        tan_[m] = tn[m]
    tan_ = tan_ + np.array([0.0, 0.0, -fall], np.float32)
    return _unit(tan_ - (tan_ * nrm).sum(1, keepdims=True) * nrm)

def _snap(C, P, off):
    """Points moved onto the finished (decimated) skin mesh, lifted by off along its face normal."""
    out = np.array(P, np.float32)
    for i, q in enumerate(out):
        loc, nr, _idx, _d = C.bvh.find_nearest((float(q[0]), float(q[1]), float(q[2])))
        if loc is not None:
            out[i] = (loc.x + nr.x * off, loc.y + nr.y * off, loc.z + nr.z * off)
    return out


# ------------------------------------------------------------------ the cat
class _Cat:
    pass

def _paw(C, x, y, yaw, s=1.0, k=0.010):
    """Paw resting on the ground: pad mass and four splayed toes (local -Y = forward)."""
    F = C.F
    R = _rot(0, 0, yaw)
    o = np.array([x, y, 0.0], np.float32)

    def W(v):
        return o + R @ (np.asarray(v, np.float32) * s)

    F.add(_ell(W((0, 0, 0.0122)), (0.0160 * s, 0.0205 * s, 0.0122 * s), R), k=k)
    for j in range(4):
        d = j - 1.5
        Rt = R @ _rot(0, 0, d * 9.0)
        F.add(_ell(W((d * 0.0084, -0.0170 + 0.0042 * abs(d), 0.0086)), (0.0049 * s, 0.0108 * s, 0.0082 * s), Rt), k=0.0028)
    for j in range(3):                                       # clefts between the toes
        d = j - 1.0
        a = W((d * 0.0086, -0.0300 + 0.0035 * abs(d), 0.0040))
        b = W((d * 0.0080, -0.0200, 0.0165))
        c = W((d * 0.0076, -0.0120, 0.0200))
        F.cut(_chain([a, b, c], [0.0015 * s, 0.0013 * s, 0.0004 * s]), k=0.0034)
    C.paws.append((float(W((0, -0.012, 0))[0]), float(W((0, -0.012, 0))[1])))

def _envelope(F, c0, phi0, turn, zqs, dists, span, n=80):
    """Outline of the body seen from above, offset outwards: points at angles phi0 + turn * [0, span] from c0. The
    outline is taken at several heights (each with its own offset) and the widest wins, so a tail laid along it is
    not tucked away under the bulge of a flank."""
    th = np.linspace(0.0, span, n)
    ph = radians(phi0) + turn * th
    dirs = np.stack([np.cos(ph), np.sin(ph)], 1).astype(np.float32)
    ts = np.linspace(0.55, 0.0, 150).astype(np.float32)
    t = np.zeros(n, np.float32)
    for zq, dist in zip(zqs, dists):
        P = np.zeros((n, len(ts), 3), np.float32)
        P[..., 0] = c0[0] + dirs[:, None, 0] * ts[None]
        P[..., 1] = c0[1] + dirs[:, None, 1] * ts[None]
        P[..., 2] = zq
        d = F.at(P) - dist
        i = np.clip(np.argmax(d < 0, 1), 1, len(ts) - 1)
        ar = np.arange(n)
        d0, d1 = d[ar, i - 1], d[ar, i]
        t = np.maximum(t, ts[i - 1] + (ts[i] - ts[i - 1]) * d0 / (d0 - d1 + 1e-9))
    for _ in range(2):                                        # soften kinks where the outline jumps between masses
        t[1:-1] = np.maximum(t[1:-1], 0.25 * t[:-2] + 0.5 * t[1:-1] + 0.25 * t[2:])
    return np.stack([c0[0] + dirs[:, 0] * t, c0[1] + dirs[:, 1] * t], 1)

def _add_tail(C, p, rng, c0, phi0, turn, L, span=1.55 * pi, z_root=0.038):
    """Tail: a round tube growing out of a root that is blended into the rump. It lies on the stones pressed
    3 mm into the coat of the body outline (tail_wrap = 1) or trails loose in an S (0); its radius tapers from
    13 mm to a rounded tip of 6 mm (thicker with fluff)."""
    F = C.F
    f = 1.0 + 0.5 * C.fluff
    infl = 0.001 + 0.002 * C.fluff                           # inflation the field still gets in build()
    rt = 0.0135 * f
    env = _envelope(F, c0, phi0, turn, (rt, 0.040, 0.060, 0.085), (rt - 0.0035 + infl, rt - 0.0035 + infl, rt - 0.0045 + infl, rt - 0.0100 + infl), span)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(env, axis=0), axis=1))])
    o = np.array([cos(radians(phi0)), sin(radians(phi0))], np.float32)
    tau = turn * np.array([-o[1], o[0]], np.float32)
    back = rt + 0.045
    Lr = L - back
    m = 15
    a = np.linspace(0.0, Lr, m)
    u = a / Lr
    wr = np.stack([np.interp(a, s, env[:, 0]), np.interp(a, s, env[:, 1])], 1)
    # loose tail: sweeps out to the side, round towards the front and the tip flicks back out -> an S on the stones
    al = np.radians(32.0 + rng.u(-6, 6) - 48.0 * (1.0 - np.cos(2 * pi * (0.8 + rng.u(-0.05, 0.05)) * u)))
    dirs = np.cos(al)[:, None] * tau[None] + np.sin(al)[:, None] * o[None]
    step = np.concatenate([np.zeros((1, 2)), 0.5 * (dirs[1:] + dirs[:-1]) * (Lr / (m - 1))])
    lo = env[0][None] + np.cumsum(step, axis=0)
    w = p.tail_wrap
    xy = lo + (wr - lo) * w
    uu = np.concatenate([[0.0], (back + a) / L])
    r = np.interp(uu, [0.0, 0.10, 0.24, 0.62, 0.86, 1.0], [0.023, 0.0170 * f, rt, rt * 0.90, rt * 0.72, rt * 0.46])
    z = r[1:] * 0.93 + 0.012 * np.exp(-a / 0.022)
    root = env[0] - o * back
    ctrl = [(float(root[0]), float(root[1]), z_root)] + [(float(xy[i, 0]), float(xy[i, 1]), float(z[i])) for i in range(m)]
    pts, rr = _path(ctrl, r, 25)
    nb = 7
    F.add(_chain(pts[:nb], rr[:nb] + np.linspace(0.0025, -0.0025, nb).astype(np.float32)), k=0.018)   # root, dives into the tube
    C.white.append((_ell(pts[-1], (0.040, 0.040, 0.040)), 0.8))
    C.tailch = _fchain(C, pts, rr, (0, 0, 1), use="tube")
    C.zones.append(dict(fn=_chain(pts[1:6], rr[1:6])[0], fall=0.008, boost=2.0, lift=1.2, length=1.2))
    C.tail = (pts, rr)

def _tube(pts, rr, stn, sides=8):
    """Round tube along pts with a hemispherical tip (open at the start, which is buried in the body).
    Returns verts, tris, angle round the tube, arc length (stn = arc length of pts) and radial normal per vertex."""
    pts = np.asarray(pts, np.float32)
    rr = np.asarray(rr, np.float32)
    te = _nrm(pts[-1] - pts[-2])
    re = float(rr[-1])
    capa = np.radians([32.0, 58.0, 78.0]).astype(np.float32)
    P = np.concatenate([pts, pts[-1][None] + te[None] * (re * np.sin(capa))[:, None]])
    R = np.concatenate([rr, re * np.cos(capa)])
    n = len(P)
    tg = np.concatenate([_unit(np.gradient(pts, axis=0)), np.repeat(te[None], 3, 0)])
    sd = _unit(np.cross(tg, np.array([0.0, 0.0, 1.0], np.float32)))
    up = np.cross(sd, tg)
    ang = (2 * pi * (np.arange(sides) + 0.5) / sides).astype(np.float32)
    V = P[:, None, :] + R[:, None, None] * (np.cos(ang)[None, :, None] * sd[:, None, :] + np.sin(ang)[None, :, None] * up[:, None, :])
    V = np.concatenate([V.reshape(-1, 3), (pts[-1] + te * re)[None]]).astype(np.float32)
    T = []
    for i in range(n - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a, b, c, d = i * sides + j, i * sides + j2, (i + 1) * sides + j2, (i + 1) * sides + j
            T += [(a, b, c), (a, c, d)]
    last = (n - 1) * sides
    for j in range(sides):
        T.append((last + j, last + (j + 1) % sides, n * sides))
    T = np.array(T, np.int64)
    ax = np.concatenate([np.repeat(P, sides, 0), P[-1:]])
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    if float((fn * (V[T].mean(1) - ax[T].mean(1))).sum()) < 0:
        T = T[:, ::-1].copy()
    Nn = _unit(V - ax + te[None] * 1e-5)
    V[:, 2] = np.maximum(V[:, 2], 0.0)
    sv = np.concatenate([np.repeat(np.concatenate([stn, stn[-1] + re * np.sin(capa)]), sides), [stn[-1] + re]])
    return V, T, np.concatenate([np.tile(ang, n), [0.0]]).astype(np.float32), sv.astype(np.float32), Nn

def _add_head(C, c, R, rng, ear_lean=0.0, ear_h=1.0, ear_yaw=0.0):
    F = C.F
    hs = C.hs
    c = np.asarray(c, np.float32) + np.array([0, 0, (hs - 1.0) * 0.022], np.float32)
    fl = C.fluff

    def W(v):
        return c + R @ (np.asarray(v, np.float32) * hs)

    def E(v, r, k):
        F.add(_ell(W(v), tuple(x * hs for x in r), R), k=k * hs)

    E((0, 0.005, -0.002), (0.053, 0.052, 0.044), 0.03)                                      # skull: low rounded crown
    for sx in (-1, 1):
        E((sx * 0.028, -0.019, -0.014), (0.0245 + 0.005 * fl, 0.027, 0.024), 0.016)         # cheeks
    E((0, -0.044, -0.015), (0.025, 0.024, 0.019), 0.014)                                    # muzzle
    F.add(_chain([W((0, -0.024, 0.012)), W((0, -0.060, -0.006))], [0.0125 * hs, 0.0085 * hs]), k=0.012 * hs)  # nose bridge
    for sx in (-1, 1):
        E((sx * 0.011, -0.057, -0.019), (0.0135, 0.012, 0.0115), 0.005)                     # whisker pads
        E((sx * 0.022, -0.027, 0.0195), (0.015, 0.012, 0.006), 0.016)                       # brows: soft, no shelf
    E((0, -0.050, -0.031), (0.011, 0.012, 0.009), 0.007)                                    # chin
    es = 1.0 + (hs - 1.0) * 0.45
    pitch = float(np.degrees(np.arcsin(np.clip(-(R @ np.array([0, -1.0, 0], np.float32))[2], -1.0, 1.0))))    # nose down = +
    for sx in (-1, 1):
        # large ears set high on the corners of the crown, opening forwards and a little outwards; they stand nearly
        # upright whatever the pitch of the head (laid back against the skull when the nose points down)
        lean = float(np.clip(7.0 - pitch, -24.0, -3.0)) + rng.u(-3, 3) + ear_lean
        yw, tw = sx * (27 + ear_yaw + rng.u(-3, 3)), sx * (20 + rng.u(-5, 5))
        Re = R @ _rot(0, yw, 0) @ _rot(lean, 0, 0) @ _rot(0, 0, tw)
        for _ in range(9):                                   # the opening never faces the ground (head tucked / rolled)
            if float((Re @ np.array([0, -1.0, 0], np.float32))[2]) > 0.04 + 0.22 * float(np.clip((pitch - 15.0) / 20.0, 0.0, 1.0)):
                break
            lean -= 5.0
            Re = R @ _rot(0, yw, 0) @ _rot(lean, 0, 0) @ _rot(0, 0, tw)
        base = W((sx * 0.0345, 0.006, 0.026))
        w_, h_ = 0.046 * es * hs, 0.041 * es * hs * ear_h
        F.add(_ell(base + Re @ np.array([0, 0.012, -0.002], np.float32), (w_ * 0.45, 0.011, 0.014 * es), Re), k=0.013)       # ear root: thick base seated in the skull
        C.ears.append(dict(base=base, R=Re, w=w_, h=h_, sx=sx, es=es))
        C.zones.append(dict(fn=_ell(W((sx * 0.043, -0.008, -0.024)), (0.016 * hs, 0.022 * hs, 0.020 * hs), R)[0],
                            fall=0.010, boost=7.0, lift=1.6, length=1.3))                    # cheek ruff
    C.head = (c, R)
    C.white.append((_ell(W((0, -0.052, -0.022)), (0.027 * hs, 0.03 * hs, 0.024 * hs), R), 0.25))
    C.flow.append(dict(head=True, c=c, R=R, hs=hs, radial=W((0, -0.105, -0.012)), use="all",
                       rbar=0.045 * hs))

def _lens(c, Rs, ew, hu, hl, depth):
    """Almond eye opening: intersection of two flat ellipsoids, so the upper and the lower lid are arcs that
    meet in pointed corners. Returns the field primitive and a function giving the lid outline."""
    du = (ew * ew - hu * hu) / (2 * hu)
    Ru = du + hu
    dl = (ew * ew - hl * hl) / (2 * hl)
    Rl = dl + hl
    zl = Rs[:, 2]
    fa = _ell(c - zl * du, (Ru, depth / (1.0 - (du / Ru) ** 2) ** 0.5, Ru), Rs)[0]
    fb = _ell(c + zl * dl, (Rl, depth / (1.0 - (dl / Rl) ** 2) ** 0.5, Rl), Rs)[0]

    def fn(P):
        return np.maximum(fa(P), fb(P))

    def outline(m):
        x = np.linspace(-ew, ew, m)
        zu = np.sqrt(Ru * Ru - x * x) - du
        zd = -(np.sqrt(Rl * Rl - x * x) - dl)
        return np.concatenate([x, x[::-1][1:-1]]), np.concatenate([zu, zd[::-1][1:-1]])

    r = ew + 0.012
    return (fn, np.asarray(c, np.float32) - r, np.asarray(c, np.float32) + r), outline

def _face(C, rng, p):
    """Almond eye openings carved into the finished skin, with a dished iris, pupil and clear cornea (or closed lids)."""
    F = C.F
    c, R = C.head
    hs = C.hs

    def W(v):
        return c + R @ (np.asarray(v, np.float32) * hs)

    op = C.eye_open
    up = R @ np.array([0, 0, 1], np.float32)
    fwd = R @ np.array([0, -1, 0], np.float32)
    for sx in (-1, 1):
        a = W((sx * 0.012, 0.0, 0.003))
        b = W((sx * 0.029, -0.090, 0.007))
        hit = F.hit(a, b)
        n = _nrm(0.65 * _nrm(F.grad(hit)) + 0.35 * _nrm(b - a))
        xl = _nrm(np.cross(up, n))
        zl = np.cross(xl, -n)
        Rs = np.stack([xl, -n, zl], 1).astype(np.float32) @ _rot(0, -12 * sx, 0)        # outer corner higher
        C.eye_pts.append(hit)
        if op > 0.05:
            ew = 0.0118 * hs
            hu = max(0.0074 * op, 0.0032) * hs
            hl = max(0.0056 * op, 0.0028) * hs
            prim, _outline = _lens(hit, Rs, ew, hu, hl, 0.0062 * hs)
            F.cut(prim, k=0.0016)
            C.lens.append(prim[0])                           # the wall of the opening becomes the dark lid margin
            re = 0.026 * hs                                  # shallow dome set 2 mm behind the lids
            ec = hit - n * (re + 0.0020 * hs)
            g = _nrm(0.80 * n + 0.20 * fwd)
            xg = _nrm(np.cross(up, n))
            Rg = np.stack([xg, np.cross(n, xg), n], 1).astype(np.float32)
            C.eyes.append((_blob("eye", ec, (re,) * 3, Rg, seg=12, rings=5, cap=0.17), ec, g, re))
            ph = min(0.0052 * hs, 0.46 * (hu + hl))
            pw = min(lerp(0.0011, 0.0046, p.pupil) * hs, ph)
            C.pupils.append(_disc("pupil", ec + g * (re + 0.00025), g, up, pw, ph, sph=re, seg=12, rings=2))
            if p.eyeshine > 0.01:                              # tapetum: a small glow deep inside the pupil, black all round it
                C.shines.append(_disc("shine", ec + g * (re + 0.00045), g, up, pw * 0.34, ph * 0.34, sph=re, seg=10, rings=1))
            C.corneas.append(_blob("cornea", ec, (re + 0.0009,) * 3, Rg, seg=10, rings=3, cap=0.17))
        else:
            arc = [hit + Rs @ (np.array([t * 0.0125, 0.0, -0.0036 * (1 - t * t)], np.float32) * hs) for t in np.linspace(-1, 1, 7)]
            arc = F.project(np.array(arc), it=4)
            F.cut(_chain(arc, [0.0019] * len(arc)), k=0.004)
            C.lid_arcs.append(arc)
    na = W((0, -0.030, -0.008))
    nb = W((0, -0.095, -0.0045))
    C.nose_pts = (F.hit(na, nb), _nrm(nb - na))

def _spine_back(spine, sr, ey=(0.0, 1.0, 0.0)):
    """Dorsal unit vectors along a spine path (perpendicular to it, towards ey)."""
    t = _unit(np.gradient(spine, axis=0))
    e = np.asarray(ey, np.float32)[None]
    return _unit(e - (e * t).sum(1, keepdims=True) * t)

def _sit(C, p, rng):
    """Sitting upright: leaning torso narrowing to the chest, thighs with raised knees beside the belly, long straight
    fore legs standing clear of the belly from the elbows down."""
    F = C.F
    b = C.b
    fl = C.fluff
    spine, sr = _path([(0, 0.086, 0.082), (0, 0.062, 0.150), (0, 0.022, 0.215), (0, -0.020, 0.268), (0, -0.044, 0.306), (0, -0.056, 0.336)],
                      [0.083 * b, 0.074 * b, 0.064 * b ** 0.5, 0.056, 0.052, 0.050], 13)
    F.add(_chain(spine, sr), k=0.0)
    dors = _spine_back(spine, sr)
    C.dorsal = (spine, dors)
    ridge = spine + dors * (sr - 0.0075)[:, None]
    F.add(_chain(ridge[1:7], [0.0100] * 6), k=0.012)                                                              # spine ridge of the lower back
    for sx in (-1, 1):
        F.add(_ell((sx * 0.064 * b, 0.040, 0.080), (0.041, 0.084, 0.073), _rot(-20, 0, -sx * 9)), k=0.020)        # thigh
        F.add(_ell((sx * 0.073 * b, -0.040, 0.072), (0.026, 0.035, 0.046)), k=0.014)                              # knee
        F.add(_ell((sx * 0.082 * b, 0.060, 0.030), (0.022, 0.034, 0.028)), k=0.012)                               # hock
        hf, hr = _path([(sx * 0.083 * b, 0.030, 0.021), (sx * 0.080 * b, -0.035, 0.0175), (sx * 0.076 * b, -0.078, 0.0145)], [0.020, 0.0172, 0.0145], 5)
        F.add(_chain(hf, hr), k=0.008)                                                                            # hind foot
        _paw(C, sx * 0.075 * b, -0.091, sx * 8.0, 1.04)
        sc = spine[8] + dors[8] * (sr[8] - 0.009) + np.array([sx * 0.031, 0, 0], np.float32)
        F.add(_ell(sc, (0.019, 0.017, 0.046), _rot(20, 0, sx * 10)), k=0.010)                                     # shoulder blade
        F.add(_ell((sx * 0.036, -0.036, 0.238), (0.025, 0.040, 0.052)), k=0.024)                                  # shoulder
        fp, fr = _path([(sx * 0.035, -0.050, 0.222), (sx * 0.037, -0.054, 0.165), (sx * 0.036, -0.063, 0.100), (sx * 0.034, -0.074, 0.042),
                        (sx * 0.033, -0.080, 0.018)], [0.026, 0.0212, 0.0170, 0.0132, 0.0134], 10)
        F.add(_chain(fp, fr), k=0.012)                                                                            # fore leg
        F.add(_ell((sx * 0.041, -0.036, 0.160), (0.012, 0.016, 0.022)), k=0.010)                                  # elbow
        _paw(C, sx * 0.033, -0.092, sx * 7.0)
        C.legs.append((np.concatenate([fp, np.array([[sx * 0.033, -0.130, 0.008]], np.float32)]), np.concatenate([fr, [0.012]]), (sx, -0.5, 0), "all"))
        C.white.append((_ell((sx * 0.033, -0.095, 0.0), (0.034, 0.05, 0.02 + 0.11 * p.white)), 0.0))
        C.white.append((_ell((sx * 0.075 * b, -0.09, 0.0), (0.03, 0.04, 0.035)), 0.3))
        C.zones.append(dict(fn=_ell((sx * 0.100 * b, 0.045, 0.075), (0.03, 0.085, 0.075))[0], fall=0.02, boost=2.2, lift=1.25, length=1.2))
        C.zones.append(dict(fn=_ell((sx * 0.044, -0.030, 0.165), (0.014, 0.02, 0.04))[0], fall=0.012, boost=0.9, lift=1.2, length=1.1))
        C.zones.append(dict(fn=_ell((sx * 0.040, -0.030, 0.290), (0.02, 0.03, 0.03))[0], fall=0.012, boost=3.0 * max(fl - 0.45, 0.0), lift=1.3, length=1.3))   # neck ruff
    F.cut(_chain(spine[6:11] + dors[6:11] * (sr[6:11] + 0.0030)[:, None], [0.0080] * 5), k=0.012)                 # furrow of the spine between the blades
    F.add(_ell((0, -0.060, 0.240), (0.038 + 0.012 * fl, 0.028, 0.052)), k=0.030)                                  # chest
    C.white.append((_ell((0, -0.104, 0.240), (0.052, 0.052, 0.05 + 0.09 * p.white)), 0.0))
    C.zones.append(dict(fn=_ell((0, -0.080, 0.238), (0.042, 0.028, 0.058))[0], fall=0.02, boost=4.5, lift=1.5, length=1.25))
    C.zones.append(dict(fn=_ell((0, 0.13, 0.06), (0.09, 0.05, 0.06))[0], fall=0.03, boost=1.2, lift=1.1, length=1.1))
    F.inflate(0.006 * fl)
    _fchain(C, spine[::-1], sr[::-1], (0, 1, 0.25))
    _add_tail(C, p, rng.fork("tail"), (0.0, 0.03), 90.0, -1.0, 0.37)
    yaw = p.head_turn
    R = _rot(5 + rng.u(-3, 3), rng.u(-4, 4), yaw, "YXZ")
    _add_head(C, np.array([0, -0.050, 0.316], np.float32) + _rot(0, 0, yaw) @ np.array([0, -0.018, 0.044], np.float32), R, rng.fork("head"))
    hc, hR = C.head
    nape = np.array([hc + hR @ np.array([0, 0.024, -0.008], np.float32) * C.hs, spine[10] + dors[10] * 0.008], np.float32)
    F.add(_chain(nape, [0.041 * C.hs, 0.049]), k=0.030)                                                           # nape: the occiput runs into a thick neck
    thr = np.array([hc + hR @ np.array([0, -0.014, -0.034], np.float32) * C.hs, (0, -0.066, 0.268)], np.float32)
    F.add(_chain(thr, [0.034 * C.hs, 0.040]), k=0.030)                                                            # throat: the chin meets the chest fur
    C.neck = (np.array([0, -0.040, 0.300], np.float32), np.array([0, -0.25, 1.0], np.float32), np.array([1.0, 0, 0], np.float32), 0.070)

def _loaf(C, p, rng):
    """Loaf: belly and flanks on the stones, a flat back that dips behind the shoulder blades, rounded haunches with
    the hind feet lying along the flanks, fore arms folded under the chest so that only the wrists show, and the
    chest ruff filling the space under the chin. The tail is tucked along one flank."""
    F = C.F
    b = C.b
    fl = C.fluff
    sleep = 1.0 - C.eye_open
    up_ = 1.0 - sleep                                                                                         # awake: head carried on a short neck
    spine, sr = _path([(0, 0.135, 0.078), (0, 0.085, 0.100), (0, 0.005, 0.090), (0, -0.075, 0.106), (0, -0.125, 0.128), (0, -0.160, 0.150),
                       (0, -0.167 - 0.015 * up_, 0.156 + 0.026 * up_)],
                      [0.060 * b, 0.066 * b, 0.064 * b, 0.064 * b, 0.052, 0.047, 0.044], 15)
    F.add(_chain(spine, sr), k=0.0)
    dors = _spine_back(spine, sr, (0, 0, 1))
    C.dorsal = (spine, dors)
    ridge = spine + dors * (sr - 0.0070)[:, None]
    F.add(_chain(ridge[1:6], [0.0095] * 5), k=0.012)                                                          # spine ridge over the rump
    for sx in (-1, 1):
        F.add(_chain(np.array([(sx * 0.046 * b, 0.105, 0.056), (sx * 0.048 * b, -0.085, 0.056)], np.float32), [0.058, 0.056]), k=0.022)   # flank on the stones
        F.add(_ell((sx * 0.088 * b, 0.064, 0.078), (0.047, 0.090, 0.080), _rot(6, 0, -sx * 8)), k=0.008)      # haunch: rounded, higher than the back
        F.add(_ell((sx * 0.046 * b, 0.092, 0.132), (0.030, 0.044, 0.028)), k=0.020)                           # hip
        F.add(_ell((sx * 0.098 * b, -0.012, 0.048), (0.027, 0.040, 0.044)), k=0.014)                          # knee
        hf, hr = _path([(sx * 0.108 * b, 0.000, 0.017), (sx * 0.106 * b, -0.050, 0.015), (sx * 0.102 * b, -0.078, 0.013)], [0.018, 0.0155, 0.0135], 5)
        F.add(_chain(hf, hr), k=0.008)                                                                        # hind foot along the flank
        _paw(C, sx * 0.101 * b, -0.088, sx * 10.0, 0.92)
        F.add(_ell((sx * 0.034, -0.078, 0.152 * b ** 0.3), (0.024, 0.042, 0.040), _rot(-6, 0, sx * 12)), k=0.012)   # shoulder blade: stands over the back
        F.add(_ell((sx * 0.072 * b, -0.088, 0.054), (0.030, 0.046, 0.048)), k=0.012)                          # elbow / upper arm
        wp, wr = _path([(sx * 0.072 * b, -0.098, 0.030), (sx * 0.060, -0.158, 0.024), (sx * 0.036, -0.206, 0.021)], [0.024, 0.021, 0.0195], 7)
        F.add(_chain(wp, wr), k=0.010)                                                                        # fore arm folded under: wrist showing
        C.legs.append((wp, wr, (0, -0.4, 1), "none"))
        C.white.append((_ell((sx * 0.036, -0.206, 0.0), (0.04, 0.04, 0.03 + 0.03 * p.white)), 0.0))
        C.white.append((_ell((sx * 0.103 * b, -0.100, 0.0), (0.032, 0.040, 0.034)), 0.3))
        C.zones.append(dict(fn=_ell((sx * 0.115 * b, 0.07, 0.07), (0.03, 0.09, 0.07))[0], fall=0.02, boost=2.6, lift=1.25, length=1.2))
        C.zones.append(dict(fn=_ell((sx * 0.085 * b, -0.10, 0.03), (0.03, 0.06, 0.03))[0], fall=0.015, boost=2.0, lift=1.2, length=1.2))
    F.add(_ell((0, -0.128, 0.086), (0.050 + 0.012 * fl, 0.040, 0.050)), k=0.03)                                # chest
    F.add(_chain(np.array([(0, -0.166, 0.136 - 0.010 * sleep), (0, -0.168, 0.100), (0, -0.156, 0.062)], np.float32), [0.044, 0.046, 0.042]), k=0.028)   # chest ruff under the chin
    C.white.append((_ell((0, -0.190, 0.090), (0.058, 0.06, 0.045 + 0.07 * p.white)), 0.0))
    C.zones.append(dict(fn=_ell((0, -0.195, 0.095), (0.05, 0.03, 0.05))[0], fall=0.02, boost=4.0, lift=1.5, length=1.25))
    C.zones.append(dict(fn=_ell((0, 0.19, 0.06), (0.08, 0.05, 0.06))[0], fall=0.03, boost=1.2, lift=1.1, length=1.1))
    F.inflate(0.006 * fl)
    _fchain(C, spine[::-1], sr[::-1], (0, 0, 1))
    _add_tail(C, p, rng.fork("tail"), (0.0, 0.03), 122.0, 1.0, 0.34)
    piv = np.array([0, -0.170, 0.146], np.float32)
    yaw = p.head_turn * 0.75
    R = _rot(4 + 7 * sleep + rng.u(-3, 3), rng.u(-4, 4), yaw, "YXZ")
    hc = piv + _rot(0, 0, yaw) @ np.array([0, -0.032 - 0.018 * up_, 0.018 + 0.052 * up_], np.float32)
    _add_head(C, hc, R, rng.fork("head"), ear_lean=-6.0 * sleep, ear_h=1.0 - 0.14 * sleep, ear_yaw=10.0 * sleep)
    thr = np.array([hc + R @ np.array([0, -0.014, -0.034], np.float32) * C.hs, (0, -0.186, 0.098)], np.float32)
    F.add(_chain(thr, [0.038 * C.hs, 0.044]), k=0.034)                                                          # throat: soft junction of jaw and chest
    Ry = _rot(0, 0, yaw * 0.6)
    C.neck = (piv + Ry @ np.array([0, 0.002 - 0.012 * up_, 0.022 * up_], np.float32), Ry @ np.array([0, -0.58, 0.81], np.float32),
              Ry @ np.array([1.0, 0, 0], np.float32), 0.070)

def _curl(C, p, rng):
    """Curled asleep: the spine runs round in a ring, the haunch with the tucked hind leg closes it at the front,
    the head rests on the fore paws against the haunch and the tail lies round the front, under the chin."""
    F = C.F
    b = C.b
    cx, cy, R0 = 0.0, 0.015, 0.100

    def arc(a, r=R0, z=0.0):
        return (cx + r * cos(radians(a)), cy + r * sin(radians(a)), z)

    angs = [212, 185, 140, 90, 40, 5, -22]
    rr = [0.050, 0.064, 0.076 * b ** 0.5, 0.082 * b, 0.084 * b, 0.084 * b, 0.078 * b]
    spine, sr = _path([arc(a, R0, r * 0.9) for a, r in zip(angs, rr)], rr, 18)
    F.add(_chain(spine, sr), k=0.0)
    cc = np.array([cx, cy, 0.0], np.float32)
    q = (spine - cc) * np.array([1, 1, 0], np.float32)
    dv = _unit(q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9) * 0.75 + np.array([0, 0, 1], np.float32))
    C.dorsal = (spine[::-1].copy(), dv[::-1].copy())
    F.add(_chain((spine + dv * (sr - 0.0075)[:, None])[4:17], [0.0100] * 13), k=0.016)           # spine ridge
    F.add(_ell((cx, cy - 0.012, 0.045), (0.085, 0.085, 0.052)), k=0.04)                           # belly
    F.add(_ell(arc(-38, 0.102, 0.068), (0.060, 0.076, 0.064), _rot(0, 0, 52)), k=0.030)           # hind thigh
    hf, hr = _path([arc(-58, 0.105, 0.03), arc(-80, 0.100, 0.024), arc(-98, 0.098, 0.02)], [0.022, 0.019, 0.017], 6)
    F.add(_chain(hf, hr), k=0.014)                                                                # hind foot
    ff, fr = _path([(-0.075, -0.020, 0.05), (-0.045, -0.060, 0.03), (-0.004, -0.070, 0.022)], [0.028, 0.021, 0.0185], 7)
    F.add(_chain(ff, fr), k=0.016)                                                                # fore leg under the chin
    F.add(_ell((-0.072, -0.008, 0.088), (0.045, 0.05, 0.046)), k=0.024)                            # shoulder
    C.legs.append((hf, hr, (0, 0, 1), "none"))
    C.legs.append((ff, fr, (0, 0, 1), "none"))
    C.white.append((_ell(arc(-98, 0.098, 0.02), (0.035, 0.035, 0.03)), 0.2))
    C.white.append((_ell((-0.004, -0.070, 0.022), (0.035, 0.03, 0.03)), 0.0))
    C.white.append((_ell((-0.03, -0.045, 0.03), (0.05, 0.04, 0.03 + 0.03 * p.white)), 0.0))
    F.inflate(0.006 * C.fluff)
    _fchain(C, spine, sr, (0, 0, 1))
    R = _rot(10 + rng.u(-3, 3), 4 + rng.u(-4, 4), 42 + rng.u(-5, 5), "YXZ")
    hc = np.array([-0.058, -0.087, 0.068], np.float32)
    _add_head(C, hc, R, rng.fork("head"), ear_lean=-8.0, ear_h=0.9, ear_yaw=12.0)
    nape = np.array([hc + R @ np.array([0, 0.034, 0.004], np.float32) * C.hs, (-0.074, -0.018, 0.086)], np.float32)
    F.add(_chain(nape, [0.044 * C.hs, 0.050]), k=0.035)                                          # nape: no crease behind the skull
    # tail hugging the outside of the curl, round in front of the nose
    _add_tail(C, p, rng.fork("tail"), (cx, cy), -24.0, -1.0, 0.30, span=lerp(0.9, 1.75, p.tail_wrap) + 0.6, z_root=0.045)
    C.neck = (np.array([-0.072, -0.062, 0.060], np.float32), np.array([0.03, -0.05, 0.012], np.float32), np.array([0, 0, 1.0], np.float32), 0.066)

# ------------------------------------------------------------------ ears
def _ear_mesh(e, k):
    """Cupped pinna: a closed shell whose section is a crescent - convex coat at the back, a rolled rim 3 mm thick all
    round the opening, the bare hollow dropping away behind it - so it reads as a cup from the front, the back and
    edge-on. Its two lowest rows lie inside the skull, drawn in towards the middle, so no edge shows at the base.
    Returns the ear, the height (0 base .. 1 tip) and hollow flag of every vertex, and the faces of the hollow."""
    w, h, sx, Re, base = e["w"], e["h"], e["sx"], e["R"], e["base"]
    vs = [-0.34, -0.10, 0.14, 0.38, 0.60, 0.80, 0.94]
    us = [(-1.0, 0), (-0.62, 0), (-0.22, 0), (0.22, 0), (0.62, 0), (1.0, 0), (0.88, 1), (0.50, 1), (0.0, 1), (-0.50, 1), (-0.88, 1)]
    nu = len(us)

    def pos(u, v, front):
        vv, root = max(v, 0.0), min(v, 0.0)
        hw = ((w / 2) * (1.0 - vv ** 1.6) ** 0.75 + 0.0012) * (1.0 + 1.3 * root)
        y = ((0.50 - 0.20 * vv) * hw + 0.0015) * (1.0 - u * u) - 0.05 * h * vv * vv
        if front:
            y -= 0.0018 + 0.0014 * float(_sstep(0.45, 0.88, abs(u)))
        return (-sx * 0.10 * w * vv + u * hw + sx * 0.030 * root, y, v * h)

    V, vv_, inner = [], [], []
    for v in vs:
        for u, front in us:
            V.append(pos(u, v, front))
            vv_.append(v)
            inner.append(1.0 if front and abs(u) < 0.6 else 0.0)
    V.append((-sx * 0.10 * w, -0.05 * h, h))
    vv_.append(1.0)
    inner.append(0.0)
    tip = len(V) - 1
    faces, hollow = [], []
    for j in range(len(vs) - 1):
        for i in range(nu):
            i2 = (i + 1) % nu
            if 6 <= i <= 9:
                hollow.append(len(faces))
            faces.append((j * nu + i, j * nu + i2, (j + 1) * nu + i2, (j + 1) * nu + i))
    top = (len(vs) - 1) * nu
    for i in range(nu):
        faces.append((top + i, top + (i + 1) % nu, tip))
    Vl = np.array(V, np.float32)
    obj = mesh_object("cat_ear", (base + Vl @ Re.T).tolist(), faces)
    recalc_normals(obj)
    me = obj.data
    lv = np.zeros(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    _set_uv(obj, np.stack([Vl[lv, 0] + 0.23 * k, Vl[lv, 2] / STRETCH], 1))
    return obj, np.array(vv_, np.float32), np.array(inner, np.float32), hollow

# ------------------------------------------------------------------ nose, mouth, lids, whiskers, collar
def _neck_ring(C, n=26):
    """Collar seat: the tightest closed section of the skin round the neck. Planes near the pose's neck hint are
    tried (pitched and shifted along the neck); a plane counts only when every ray from its centre leaves the skin
    close by, so the strap can never stand free of the coat. Returns (centre, axis, ray directions, radii) or None."""
    F = C.F
    nc0, na0, side, rmax = C.neck
    na0 = _nrm(na0)
    side = _nrm(side - float(side @ na0) * na0)
    fw = np.cross(side, na0)
    ang = 2 * pi * np.arange(n) / n
    best = None
    for pitch in (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0):
        ax = _nrm(na0 * cos(radians(pitch)) + fw * sin(radians(pitch)))
        v = np.cross(ax, side)
        dirs = (np.cos(ang)[:, None] * side + np.sin(ang)[:, None] * v).astype(np.float32)
        for off in (-0.020, -0.012, -0.004, 0.004, 0.012, 0.020):
            o = (nc0 + na0 * off).astype(np.float32)
            r = None
            for _ in range(3):
                if float(F.at(o)) > -0.012:
                    r = None
                    break
                H, ok = F.hits(o + dirs * 0.004, o + dirs * (rmax * 1.5), n=60)
                if not ok.all():
                    r = None
                    break
                r = np.linalg.norm(H - o, axis=1)
                o = (o + 0.8 * (H.mean(0) - o)).astype(np.float32)
            if r is None:
                continue
            H, ok = F.hits(o + dirs * 0.004, o + dirs * (rmax * 1.5), n=60)
            if not ok.all():
                continue
            r = np.linalg.norm(H - o, axis=1)
            cost = float(r.mean() + 1.0 * r.std() + 0.5 * (r.max() - r.mean()) + 0.10 * off)
            if best is None or cost < best[0]:
                best = (cost, o, ax, dirs, r)
    return None if best is None else best[1:]

def _accessories(C, rng, p):
    F = C.F
    c, R = C.head
    hs = C.hs

    def W(v):
        return c + R @ (np.asarray(v, np.float32) * hs)

    out = {}
    nh, nd = C.nose_pts
    out["nose"] = [_blob("nose", nh - nd * 0.0030 * hs, (0.0064 * hs, 0.0044 * hs, 0.0045 * hs), R, seg=8, rings=5)]
    lines = []
    ph = F.project(np.array([W((0, -0.068, -0.0125 - 0.0023 * i)) for i in range(5)]), it=4, off=0.0005)
    lines.append((ph, 0.00055, 0.00045))
    for sx in (-1, 1):
        lip = [W((sx * 0.012 * t, -0.068 + 0.012 * t * t, -0.0215 - 0.004 * sin(t * pi * 0.9))) for t in np.linspace(0, 1, 5)]
        lines.append((F.project(np.array(lip), it=4, off=0.0004), 0.00048, 0.00025))
    for arc in C.lid_arcs:
        if C.eye_open < 0.25:                                  # closed lids: keep the line on the finished skin, never half buried
            lines.append((_snap(C, F.project(np.array(arc, np.float32), it=3), 0.0002), 0.0008, 0.0008))
        else:
            lines.append((F.project(np.array(arc), it=3, off=-0.0003), 0.0008, 0.0008))
    out["lines"] = [_hairs("face_lines", lines, sides=3)]
    wr = rng.fork("whiskers")
    strands = []
    for sx in (-1, 1):
        for j in range(5):
            root = F.hit(W((sx * 0.010, -0.050, -0.019)), W((sx * 0.034, -0.078 + 0.004 * (j % 2), -0.027 + 0.0052 * j)))
            dl = _nrm(np.array([sx * 1.0, -0.42 + wr.u(-0.12, 0.12), -0.16 + 0.10 * j + wr.u(-0.05, 0.05)], np.float32))
            L = wr.u(0.048, 0.068) * hs
            dw = R @ dl
            pts = [root - dw * 0.002 + dw * (L * t) + np.array([0, 0, -0.009 * t * t], np.float32) for t in np.linspace(0, 1, 5)]
            strands.append((pts, C.whisk, 0.45 * C.whisk))
    out["whiskers"] = [_hairs("whiskers", strands, sides=3)]
    out["collar"], out["brass"] = [], []
    C.collar_ring = None
    ring_fit = C.ring_fit
    if ring_fit is not None:
        nc, na_, dirs, rad = ring_fit
        n = len(dirs)
        C.collar_ring = (nc, na_, float(rad.max()))
        TH, WD, GAP = 0.0022, 0.0110, 0.0020
        # the strap lies on the skin like a real collar: its width runs along the neck inside the local tangent plane,
        # so on the cone of a hunched (loaf) neck it tilts to face outwards instead of standing on edge
        Pm = F.project(nc + dirs * rad[:, None], it=2)
        nr = _unit(F.grad(Pm))
        wd = na_ - (nr @ na_)[:, None] * nr
        wd = _unit(np.where(np.linalg.norm(wd, axis=1, keepdims=True) > 0.2, wd, na_ + 0.0 * wd))
        for _ in range(2):
            wd = _unit(np.roll(wd, 1, axis=0) + 2 * wd + np.roll(wd, -1, axis=0))
        E0 = F.project(Pm - wd * (WD / 2), it=3)
        E1 = F.project(Pm + wd * (WD / 2), it=3)
        E1 = E0 + _unit(E1 - E0) * WD
        for _ in range(3):                                    # both edges sit GAP above the coat all round
            E0 = E0 + _unit(F.grad(E0)) * (GAP - F.at(E0))[:, None]
            E1 = E1 + _unit(F.grad(E1)) * (GAP - F.at(E1))[:, None]
            E1 = E0 + _unit(E1 - E0) * WD
        wdir = _unit(E1 - E0)
        tg = _unit(np.roll(E0 + E1, -1, axis=0) - np.roll(E0 + E1, 1, axis=0))
        nrm = _unit(np.cross(tg, wdir))
        nrm *= np.sign((nrm * nr).sum(1) + 1e-9)[:, None]
        ring = 0.5 * (E0 + E1) + nrm * (TH / 2)
        lift = np.maximum(0.004 - np.minimum(E0[:, 2], E1[:, 2]), 0.0)[:, None] * np.array([0, 0, 1.0], np.float32)
        E0, E1, ring = E0 + lift, E1 + lift, ring + lift
        cv = np.concatenate([E0, E1, E1 + nrm * TH, E0 + nrm * TH]).astype(np.float32)
        i = np.arange(n)
        j = (i + 1) % n
        cf = []
        for a_, b_ in ((0, 1), (1, 2), (2, 3), (3, 0)):
            cf.append(np.stack([a_ * n + i, a_ * n + j, b_ * n + j, b_ * n + i], 1))
        strap = mesh_object("collar", cv.tolist(), np.concatenate(cf).tolist())
        recalc_normals(strap)
        out["collar"].append(strap)
        low = _nrm(0.5 * (R @ np.array([0, -1.0, 0], np.float32)) + np.array([0, -0.5, -0.9], np.float32))
        i0 = int(np.argmax((ring - nc) @ low))
        # buckle to one side of the bell: brass frame + tongue, keeper loop
        ib = (i0 + n // 6) % n
        tng = _nrm(ring[(ib + 1) % n] - ring[ib - 1])
        rd = _nrm(nrm[ib] - float(nrm[ib] @ tng) * tng)
        wb = _nrm(np.cross(rd, tng))
        q = ring[ib] + rd * (TH / 2 + 0.0006)
        tl = lambda x: tuple(float(y) for y in x)
        fr = [q + tng * a + wb * b for a, b in ((-0.0070, -0.0074), (0.0070, -0.0074), (0.0070, 0.0074), (-0.0070, 0.0074))]
        out["brass"].append(sweep("buckle", [tl(x) for x in fr], 0.0010, closed=True, sides=4, smooth_path=False, up=tl(rd)))
        out["brass"].append(sweep("buckle_tongue", [tl(q - tng * 0.0070 + rd * 0.0006), tl(q + tng * 0.0078 + rd * 0.0012)], 0.0007, sides=4,
                                  smooth_path=False, up=tl(rd)))
        ik = (ib + 2) % n
        tk_ = _nrm(ring[(ik + 1) % n] - ring[ik - 1])
        kp = [(-0.0020, -WD / 2 - 0.0008), (0.0020, -WD / 2 - 0.0008), (0.0020, WD / 2 + 0.0008), (-0.0020, WD / 2 + 0.0008)]
        out["collar"].append(sweep("collar_keeper", [tl(ring[ik] - tk_ * 0.0022), tl(ring[ik] + tk_ * 0.0022)], kp, smooth_path=False, up=tl(wdir[ik])))
        if p.bell:
            bp = ring[i0] + dirs[i0] * 0.004 + np.array([0, 0, -0.0105], np.float32)
            for _ in range(6):                                # keep the bell outside the fur
                d = float(F.at(bp))
                if d < 0.0078:
                    bp = bp + _nrm(F.grad(bp)) * (0.0078 - d)
            bp[2] = max(bp[2], 0.0092)
            out["brass"].append(_blob("bell", bp, (0.0085, 0.0085, 0.0085), seg=10, rings=6))
            top = ring[i0] + dirs[i0] * 0.0016
            out["brass"].append(sweep("bell_loop", [tl(top), tl(0.5 * (top + bp) + dirs[i0] * 0.002), tl(bp + _nrm(top - bp) * 0.0075)], 0.0008, sides=4,
                                      smooth_path=False))
            out["brass"].append(torus("bell_band", major=0.0086, minor=0.0010, loc=tl(bp), major_seg=12, minor_seg=3))
    return out

def _set_uv(obj, luv):
    me = obj.data
    lay = me.uv_layers.new(name="UVMap")
    lay.data.foreach_set("uv", np.ascontiguousarray(luv, np.float32).ravel())
    uv_keep(obj)

def _plain_uv(obj, k=0.0):
    """Texture coordinates of a small coat part: fibres running up its length."""
    me = obj.data
    nv = len(me.vertices)
    co = np.zeros(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lv = np.zeros(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    _set_uv(obj, np.stack([co[lv, 0] * 0.8 + co[lv, 1] * 0.6 + 0.071 * k, co[lv, 2] / STRETCH], 1))

def build(p, rng):
    S = p.size * 0.97
    coat = COATS[p.coat]
    tabby = "gap" in coat
    C = _Cat()
    C.F = F = _Field()
    C.b = lerp(0.9, 1.14, p.build)
    C.fluff = p.fluff
    C.hs = p.head_scale
    C.whisk = coat.get("whisk", 0.00055)                     # wide enough to stay one unbroken line at 1.5 m
    for k in ("white", "eyes", "pupils", "shines", "corneas", "lid_arcs", "lens", "ears", "flow", "zones", "legs", "paws", "eye_pts"):
        setattr(C, k, [])
    C.eye_open = {"auto": {"sit": 1.0, "loaf": 0.5, "curl": 0.0}[p.pose], "open": 1.0, "half": 0.5, "closed": 0.0}[p.eyes]
    {"sit": _sit, "loaf": _loaf, "curl": _curl}[p.pose](C, p, rng.fork("pose"))
    for pts, rad, ref, use in C.legs:
        _fchain(C, pts, rad, ref, use=use)
    F.inflate(0.001 + 0.002 * p.fluff)
    _face(C, rng.fork("face"), p)
    C.ring_fit = _neck_ring(C) if p.collar else None
    hc, hR = C.head
    hs = C.hs
    headch = next(ch for ch in C.flow if ch.get("head"))
    eyes_c = np.array(C.eye_pts, np.float32)

    # ---- white markings: one smooth field (bib, socks, muzzle, tail tip melted together, lobed edge)
    zones = [fn for (fn, _lo, _hi), thr in C.white if p.white > thr and p.white > 0.01]
    wnA = noise_mask(0.040, seed=p.seed + 11, lo=0.0, hi=1.0)
    wnB = noise_mask(0.016, seed=p.seed + 12, lo=0.0, hi=1.0)

    def wfield(P):
        P = np.asarray(P, np.float32)
        fw = np.full(len(P), 1.0, np.float32)
        if not zones:
            return fw
        for fn in zones:
            fw = _smin(fw, fn(P), 0.018)
        return fw - (np.asarray(wnA(P, P), np.float32) - 0.5) * 0.020 - (np.asarray(wnB(P, P), np.float32) - 0.5) * 0.007

    # ---- skin: field -> surface nets -> snap to the field -> drop the hidden underside -> decimate
    h = H_GRID
    D, lo = F.grid(h)
    V, Q = _surface_nets(D, lo, h)
    V = F.project(V, it=2)
    under = (V[:, 2] < 0.3 * h)[Q].all(1)
    V[np.unique(Q[under]), 2] = 0.0
    Q = Q[~under]
    used = np.unique(Q)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    V = V[used]
    Q = remap[Q]
    V[:, 2] = np.maximum(V[:, 2], 0.0)
    body = mesh_object("cat_skin", V.tolist(), Q.tolist())
    recalc_normals(body)
    # keep the face, the paws and the border of the white dense, let the broad body masses go coarse
    dh = np.linalg.norm(V - hc, axis=1)
    q = ((V - hc) @ hR) / hs
    wv = np.ones(len(V), np.float32)
    wv[dh < 0.105 * hs] = 0.45
    wv[(dh < 0.105 * hs) & (q[:, 1] < -0.012)] = 0.32
    for e in C.eye_pts:
        wv[np.linalg.norm(V - e, axis=1) < 0.020 * hs] = 0.14
    for px, py in C.paws:
        wv[(np.hypot(V[:, 0] - px, V[:, 1] - py) < 0.036) & (V[:, 2] < 0.035)] = 0.22
    if zones:
        wv[np.abs(wfield(V)) < 0.007] = np.minimum(wv[np.abs(wfield(V)) < 0.007], 0.55)
    vg = body.vertex_groups.new(name="coarse")
    for val in np.unique(wv):
        vg.add(np.nonzero(wv == val)[0].tolist(), float(val), "REPLACE")
    m = body.modifiers.new("dec", "DECIMATE")
    opened = C.eye_open > 0.05
    n_lock = int(230 + 520 * p.fluff)                         # locks (two triangles each), a fifth of them on the tail
    n_tail = int(0.22 * n_lock)
    n_bar = BAR_TRIS if tabby else 0
    other = (1380 if tabby else 1120) + 2 * n_lock + n_bar + (460 + (130 if p.bell else 0) if C.ring_fit is not None else 0)
    other += (300 if opened else 80) + (24 if opened and p.eyeshine > 0.01 else 0)
    target = (TRI_CAP - other) / (1.27 if zones else 1.0)
    m.ratio = min(1.0, target / (2.0 * len(Q)))
    m.vertex_group = "coarse"
    m.vertex_group_factor = 1.0
    apply_modifiers(body)
    me = body.data
    me.calc_loop_triangles()
    V = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    V[:, 2] = np.maximum(V[:, 2], 0.0)
    T = np.zeros(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3).astype(np.int64)
    delete([body])
    dh = np.linalg.norm(V - hc, axis=1)
    rw = np.clip((dh - 0.10 * hs) / 0.03, 0.0, 1.0) * np.clip((V[:, 2] - 0.03) / 0.02, 0.0, 1.0)
    if RELAX:
        V = _relax(F, V, T, rw, RELAX)
    nb = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    if float((nb * F.grad(V[T].mean(1))).sum()) < 0:                      # outward winding
        T = T[:, ::-1].copy()
    Ed = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    cn = np.zeros(len(V), np.float32)
    np.add.at(cn, Ed[:, 0], 1.0)
    np.add.at(cn, Ed[:, 1], 1.0)
    for _ in range(6):                                                    # iron out triangles folded over by the decimation
        ct = V[T].mean(1)
        fnr = _unit(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]))
        bad = ((fnr * _unit(F.grad(ct))).sum(1) < 0.3) & (ct[:, 2] > 0.006)
        for e in C.eye_pts:                                               # the lid edges are meant to be sharp
            bad &= np.linalg.norm(ct - e, axis=1) > 0.024 * hs
        if not bad.any():
            break
        bv = np.zeros(len(V), bool)
        bv[T[bad].ravel()] = True
        ac = np.zeros_like(V)
        np.add.at(ac, Ed[:, 0], V[Ed[:, 1]])
        np.add.at(ac, Ed[:, 1], V[Ed[:, 0]])
        V = V.copy()
        V[bv] = F.project((ac / np.maximum(cn, 1.0)[:, None])[bv], it=2)
    lab = np.zeros(len(T), np.int32)
    if zones and (wfield(V) < 0).any():
        V, T, lab, par = _split_iso(V, T, wfield)
        V, T, _l, par = _split_iso(V, T, lambda P: wfield(P) + FEATHER_W)      # row of vertices where the soft edge of the white ends
        lab = lab[par]
    for fn in C.lens:                                                     # lid margin: the wall of the eye opening is dark skin
        lab[(fn(V[T].mean(1)) < -0.0004) & (fn(V)[T].max(1) < 0.0013)] = 2    # never a triangle that reaches out over the lid
    C.bvh = BVHTree.FromPolygons(V.tolist(), T.tolist())

    # ---- materials: matte fibre coat (texture coordinates = across / along the hair flow). The colour of the material is
    # the palest tone of the coat; every darker tone (ground colour, saddle, bars, contact shading) is ONE smooth
    # per-vertex mask, so no tone ever changes in a step and nothing is left to the automatic edge / cavity masks
    fur = mat("paper", coat["tint"], scale=FUR_TEX, rough=coat["rough"])
    fur_w = mat("paper", WHITE, scale=FUR_TEX, rough=coat["rough"])
    dark = mat("wood_painted", "#0b0807", rough=1.5, wear=0.001, grime=0.001)
    skin = mat("wood_painted", coat["skin"], rough=0.7, wear=0.001, grime=0.4)

    nzB = noise_mask(0.065, seed=p.seed + 6, lo=0.0, hi=1.0)
    brk = noise_mask(0.042, seed=p.seed + 8, lo=0.0, hi=1.0)
    body0 = C.flow[0]

    def face_terms(co):
        qh = ((co - hc) @ hR) / hs
        mz = 1.0 - _sstep(0.7, 1.3, np.sqrt((((qh - np.array([0, -0.050, -0.022], np.float32)) / np.array([0.030, 0.034, 0.026], np.float32)) ** 2).sum(1)))
        de = np.linalg.norm(co[:, None, :] - eyes_c[None], axis=2).min(1)
        return mz, 1.0 - _sstep(0.013 * hs, 0.024 * hs, de)

    def hpat(P):
        _d, _a, qh, _rho, wh = _head_eval(headch, P)
        return wh * (1.0 - _sstep(0.014, 0.038, qh[:, 1]))

    def coat_g(co, nr=None, tail=False):
        """Tone of the fur at the points co (0 = the pale colour of the material .. 1 = darkest): ground colour, paler
        belly / chest / muzzle / eye rings, darker saddle, a faint 6 cm break-up, lighter upward faces on a black coat,
        the coat's own contact shadow on the stones, and the 5 mm feather on the inside of a white marking."""
        co = np.asarray(co, np.float32)
        return np.clip(np.floor((co[:, 2] - 0.04) / 0.026) / 10.0, 0.0, 1.0).astype(np.float32)   # DBG ramp
        b = np.asarray(nzB(co, co), np.float32)
        if tail:
            g = coat["ground"] + 0.6 * coat["saddle"] + coat["brk"] * (b - 0.5)
        else:
            d, _s, th, _tn = _chain_eval(body0, co)
            wb, sad = np.exp(-np.maximum(d, 0.0) / 0.02), 0.5 + 0.5 * np.cos(th)
            mz, ey = face_terms(co)
            calm = 1.0 - 0.7 * _head_eval(headch, co)[4]
            g = coat["ground"] * (1.0 - coat["pale"] * (0.25 + 0.75 * wb) * (1.0 - sad) ** 2) + coat["saddle"] * wb * sad ** 2
            g = (g + coat["brk"] * (b - 0.5) * calm) * (1.0 - coat["face"] * np.maximum(mz, 0.6 * ey))
        if coat["rim"]:
            nr = F.normal(co) if nr is None else np.asarray(nr, np.float32)
            g = g - coat["rim"] * np.clip(nr[:, 2], 0.0, 1.0) ** 1.5
        if zones:
            wf = wfield(co)
            g = np.where(wf < 0.0004, coat["edge"] * (1.0 - _sstep(0.0, FEATHER_W, -wf)) + 0.03 * b, g)
        g = g + 0.12 * (1.0 - _sstep(0.0, 0.016, co[:, 2]))
        return np.clip(g, 0.0, 1.0).astype(np.float32)

    def place(P, off):
        """Points put on the skin (the field surface, lifted wherever a face of the mesh lies above it) + off."""
        Q = F.project(np.asarray(P, np.float32).reshape(-1, 3), it=3)
        n = F.normal(Q)
        lift = np.zeros(len(Q), np.float32)
        for i, q in enumerate(Q.tolist()):
            loc, _nr, _i, _d = C.bvh.find_nearest(q)
            if loc is not None:
                lift[i] = max(0.0, (loc.x - q[0]) * n[i, 0] + (loc.y - q[1]) * n[i, 1] + (loc.z - q[2]) * n[i, 2])
        return Q + n * (off + lift)[:, None], n

    # ---- overlay ribbons: every dark marking of a tabby is a slim spindle of three rows of vertices lying 1 mm above
    # the skin: the middle row carries the dark tone, both outer rows the tone of the coat below them, so the edges are
    # soft whatever the size of the skin triangles, and each one runs out into a point at both ends
    RV, RT, RG, RS = [], [], [], []
    OFF_C, OFF_E = 0.0015, 0.0012

    def ribbon_line(pts, hw, depth, taper):
        """Centre line of a marking laid on the skin every 3 mm, with its half width, depth and pointed ends."""
        pts = np.asarray(pts, np.float32)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        L = float(s[-1])
        if L < 0.010:
            return None
        n = max(5, int(np.ceil(L / 0.003)) + 1)
        si = np.linspace(0.0, L, n)
        c = np.stack([np.interp(si, s, pts[:, i]) for i in range(3)], 1).astype(np.float32)
        for _ in range(3):
            c[1:-1] = 0.25 * c[:-2] + 0.5 * c[1:-1] + 0.25 * c[2:]
        c, nr = place(c, OFF_C)
        t0, t1 = min(taper[0], 0.45 * L), min(taper[1], 0.45 * L)
        f = np.clip(np.minimum(si / t0, (L - si) / t1), 0.0, 1.0)
        forced = sorted({0, int(round(t0 / L * (n - 1))), int(round((L - t1) / L * (n - 1))), n - 1})
        return dict(c=c, nr=nr, si=si, f=f, forced=forced, tg=_unit(np.gradient(c, axis=0)),
                    hw=np.interp(si, s, np.broadcast_to(np.asarray(hw, np.float32), (len(pts),))),
                    dep=np.interp(si, s, np.broadcast_to(np.asarray(depth, np.float32), (len(pts),))))

    def stations(rb, tol, dmax):
        """As few stations along the line as keep every chord within tol of the skin (and of the line, sideways)."""
        c, si, nr = rb["c"], rb["si"], rb["nr"]
        out = [0]
        for a, b in zip(rb["forced"][:-1], rb["forced"][1:]):
            i = a
            while i < b:
                j = i + 1
                while j < b:
                    k = j + 1
                    seg = c[k] - c[i]
                    q = c[i + 1:k] - c[i]
                    q = q - np.outer(q @ seg / (float(seg @ seg) + 1e-12), seg)
                    if si[k] - si[i] > dmax or np.abs((q * nr[i + 1:k]).sum(1)).max() > tol or np.linalg.norm(q, axis=1).max() > 4.0 * tol:
                        break
                    j = k
                out.append(j)
                i = j
        return out

    def ribbon(rb, idx):
        idx = np.asarray(idx)
        c, nr, tg = rb["c"][idx], rb["nr"][idx], rb["tg"][idx]
        n = len(idx)
        wd = (2.0 * rb["hw"] * rb["f"])[idx].astype(np.float32)
        dep = (rb["dep"] * rb["f"] ** 0.5)[idx].astype(np.float32)
        sd = _unit(np.cross(nr, tg))
        l = place(c + sd * wd[:, None], OFF_E)[0]
        r = place(c - sd * wd[:, None], OFF_E)[0]
        wide = wd > 0.0004
        base = sum(len(v) for v in RV)
        ic = base + np.arange(n)
        il = np.where(wide, base + n + np.cumsum(wide) - 1, ic)
        ir = np.where(wide, base + n + int(wide.sum()) + np.cumsum(wide) - 1, ic)
        g0 = coat_g(c)
        RV.append(np.concatenate([c, l[wide], r[wide]]))
        RG.append(np.concatenate([np.clip(g0 + dep, 0.0, 1.0), coat_g(l[wide]), coat_g(r[wide])]))
        tt = []
        for i in range(n - 1):
            tt += [(il[i], ic[i], ic[i + 1]), (il[i], ic[i + 1], il[i + 1]), (ic[i], ir[i], ir[i + 1]), (ic[i], ir[i + 1], ic[i + 1])]
        tt = np.array([t_ for t_ in tt if len(set(t_)) == 3], np.int64).reshape(-1, 3)
        RT.append(tt)
        RS.append((c[:-1], c[1:], wd[:-1], wd[1:], dep[:-1], dep[1:]))

    def bar_g(P):
        """Dark tone the ribbons add at the points P (the same tent profile the ribbons show)."""
        P = np.asarray(P, np.float32)
        out = np.zeros(len(P), np.float32)
        if not RS:
            return out
        A, B, WA, WB, DA, DB = (np.concatenate([r_[i] for r_ in RS]) for i in range(6))
        ab = B - A
        ab2 = (ab * ab).sum(1) + 1e-12
        for i0 in range(0, len(P), 1500):
            q = P[i0:i0 + 1500, None, :] - A[None]
            t = np.clip((q * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)
            d = np.linalg.norm(q - t[..., None] * ab[None], axis=-1)
            out[i0:i0 + 1500] = ((DA + (DB - DA) * t) * np.clip(1.0 - d / np.maximum(WA + (WB - WA) * t, 1e-5), 0.0, 1.0)).max(1)
        return out

    wr_ = rng.fork("warp")
    kw_ = [(_nrm([wr_.u(-1, 1), wr_.u(-1, 1), wr_.u(-1, 1)]) * (2 * pi / lam), wr_.u(0, 2 * pi), a_) for lam, a_ in ((0.105, 0.0075), (0.060, 0.0035))]
    uwarp = lambda P: sum(a_ * np.sin(np.asarray(P, np.float32) @ k_ + p_) for k_, p_, a_ in kw_) / S
    koff = np.array([0.371, 0.113, 0.237], np.float32)
    br = rng.fork("bars")

    def breaks(P, k, lo_, hi_):                                 # 1 where bar k is drawn, 0 where it is broken
        return _sstep(lo_, hi_, np.asarray(brk(np.asarray(P, np.float32) + np.asarray(k, np.float32)[:, None] * koff, None), np.float32))

    def limb_d(P):
        d = np.full(len(P), 1.0, np.float32)
        for ch in C.flow[1:]:
            if not ch.get("head") and ch["use"] == "all":
                d = np.minimum(d, _chain_eval(ch, P)[0])
        return d

    def body_w(P):
        """Strength of the body bars: full where the coat faces the back, a little less on the flanks, fading out
        towards the belly and the chest; rings on the outer faces of the free legs; none on the head (it has its own
        lines), on the white, low on the feet or right on the dorsal line."""
        nrv = F.normal(P)
        _d0, _s, th0, tn = _chain_eval(body0, P)
        e1 = _unit(body0["ref"][None] - (tn @ body0["ref"])[:, None] * tn)
        nd = (nrv * e1).sum(1)
        base = _sstep(-0.70, -0.20, nd) * (0.62 + 0.38 * _sstep(-0.1, 0.7, nd)) * _sstep(0.07, 0.13, np.abs(th0))
        onl = 1.0 - _sstep(0.004, 0.012, limb_d(P))
        outer = nrv[:, 0] * np.sign(P[:, 0]) - 0.35 * nrv[:, 1]
        base = base * (1.0 - onl) + 0.8 * _sstep(-0.15, 0.35, outer) * onl
        base = base * (1.0 - hpat(P)) * _sstep(0.008, 0.026, P[:, 2])
        return base * _sstep(0.002, 0.010, wfield(P)) if zones else base

    def head_w(P, par):
        _d, _a, qh, _rho, wh = _head_eval(headch, P)
        mz, ey = face_terms(P)
        win = _sstep(par[:, 1] - 0.004, par[:, 1] + 0.018, qh[:, 1]) * (1.0 - _sstep(par[:, 2] - 0.014, par[:, 2] + 0.014, qh[:, 1]))
        w = win * wh * (1.0 - _sstep(0.014, 0.038, qh[:, 1])) * (1.0 - np.maximum(mz, ey))
        return w * _sstep(0.002, 0.010, wfield(P)) if zones else w

    if tabby:
        jobs = []                                                # (points, visible half width, depth, segment, stand-off, tapers)
        # one continuous dorsal line: tail root -> rump -> back -> nape -> occiput
        dp, dd = C.dorsal
        sd_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dp, axis=0), axis=1))])
        sf = np.linspace(0.0, sd_[-1], int(sd_[-1] / 0.006) + 2)
        A = np.stack([np.interp(sf, sd_, dp[:, i]) for i in range(3)], 1).astype(np.float32)
        Dd = _unit(np.stack([np.interp(sf, sd_, dd[:, i]) for i in range(3)], 1))
        t0_ = _nrm(dp[1] - dp[0])
        fan = np.radians([68.0, 52.0, 36.0, 20.0, 8.0])
        A = np.concatenate([np.repeat(A[:1], len(fan), 0), A])
        Dd = np.concatenate([np.cos(fan)[:, None] * Dd[:1] - np.sin(fan)[:, None] * t0_[None], Dd])
        occ = hc + hR @ np.array([0, 0.022, 0.004], np.float32) * hs
        do = hR @ _nrm([0.0, 0.80, 0.60])
        tn_ = np.linspace(0.15, 1.0, 8)[:, None]
        A = np.concatenate([A, A[-1][None] + (occ - A[-1])[None] * tn_]).astype(np.float32)
        Dd = np.concatenate([Dd, _unit(Dd[-1][None] + (do - Dd[-1])[None] * tn_)]).astype(np.float32)
        H, ok = F.hits(A, A + Dd * 0.17, n=90)
        ok &= H[:, 2] > 0.012
        if ok.sum() > 4:
            jobs.append((H[ok], 0.0032 / S, coat["spine"], (0.012 / S, 0.030 / S)))
        u = _stripe_solve(C, V, T) + uwarp(V)
        cen, hws = _bar_list(br, float(u.min()) - 0.02, float(u.max()) + 0.02, coat["gap"] / S, 0.30, coat["hw"] / S)
        for k, pts in _iso_lines(V, T, u, cen, 2.0 * coat["gap"] / S):
            w = body_w(pts) * breaks(pts, np.full(len(pts), k), 0.22, 0.48)
            w[1:-1] = 0.25 * w[:-2] + 0.5 * w[1:-1] + 0.25 * w[2:]
            for a_, b_ in _runs(w > 0.14):
                q = pts[a_:b_]
                jobs.append((q, hws[k] * (0.50 + 0.50 * w[a_:b_]), coat["bar"] * np.clip(w[a_:b_] / 0.45, 0.4, 1.0), (0.030 / S, 0.030 / S)))
        # head: lines up the forehead ('M'), one over the crown, bars sweeping back over the cheeks
        hb = [(0.0, 0.0021, -0.006, 0.075), (0.21, 0.0023, -0.041, 0.062), (0.47, 0.0024, -0.033, 0.050), (0.74, 0.0019, -0.020, 0.030),
              (1.02, 0.0025, -0.030, 0.016), (1.36, 0.0025, -0.026, 0.008)]
        hb = sorted([(-a, w, y0, y1) for a, w, y0, y1 in hb if a > 0] + hb)
        hcen = np.array([b_[0] for b_ in hb], np.float32)
        hpar = np.array([b_[1:] for b_ in hb], np.float32)
        _dh, ah, _qh, _rho, wh_ = _head_eval(headch, V)
        for k, pts in _iso_lines(V, T[(wh_[T] > 0.02).all(1)], ah, hcen, 0.5):
            w = head_w(pts, hpar[np.full(len(pts), k)]) * breaks(pts, np.full(len(pts), k + 40), 0.16, 0.38)
            for a_, b_ in _runs(w > 0.14):
                jobs.append((pts[a_:b_], hpar[k, 0] * hs * (0.55 + 0.45 * w[a_:b_]), 0.85 * coat["bar"] * np.clip(w[a_:b_] / 0.45, 0.4, 1.0), (0.012, 0.012)))
        lines = [rb for rb in (ribbon_line(*j) for j in jobs) if rb is not None]
        for tol in (0.00045, 0.0006, 0.0008, 0.0010):
            sts = [stations(rb, tol, 0.024) for rb in lines]
            if sum(4 * len(s_) - 8 for s_ in sts) <= n_bar:
                break
        for rb, s_ in zip(lines, sts):
            ribbon(rb, s_)

    def fur_mesh(name, V, T, lab, uvc, g_):
        """Coat mesh (skin, ribbons and locks of one part): smooth shaded, tinted by the per-vertex tone g_."""
        V = np.array(V, np.float32)
        V[:, 2] = np.maximum(V[:, 2], 0.0)
        o = _tri_mesh(name, V, T)
        _set_uv(o, (np.asarray(uvc, np.float32) * np.array([1.0, 1.0 / STRETCH], np.float32)).reshape(-1, 2))
        assign(o, fur)
        slot = np.zeros(3, np.int32)
        for li, m_ in ((1, fur_w), (2, dark)):
            if (lab == li).any():
                o.data.materials.append(m_)
                slot[li] = len(o.data.materials) - 1
        o.data.polygons.foreach_set("material_index", slot[lab])
        smooth(o, 180)
        paint(o, grime=_pv(g_), wear=0.0, special=0.0, mode="set")
        return o

    gen = np.random.default_rng(int(rng.fork("tufts").u(1, 2 ** 31 - 1)))

    def tufts(V, T, lab, uvc, g_, z, nmax, tail=None):
        """Fur tufts: a skin triangle is drawn out into a pointed clump - a new vertex 1 - 2 cm down the hair flow,
        lying 1 - 2 mm above the coat and standing off where the coat curves away under it (which is what breaks the
        outline) - closed underneath. A tuft is part of the skin: its point takes the normal of the triangle it grew
        from and the tone of the coat right under it, so inside the outline it is shaded exactly like the skin.
        z = weight per triangle (-1 = none; 1 = cheek ruff, chest, back of the thighs). Returns the new mesh."""
        if True or nmax <= 0 or not (z > 0).any():
            return V, T, lab, uvc, g_
        e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
        cr = np.cross(e1, e2)
        ar = 0.5 * np.linalg.norm(cr, axis=1)
        nt = _unit(cr)
        per = np.linalg.norm(e1, axis=1) + np.linalg.norm(e2, axis=1) + np.linalg.norm(e2 - e1, axis=1)
        wz = ar * np.where((z < 0.0) | (20.8 * ar / (per * per + 1e-12) < 0.4), 0.0, (0.25 + np.maximum(z, 0.0)) ** 1.5)
        m_ = min(int(1.5 * nmax) + 8, int((wz > 0).sum()))
        if m_ < 1:
            return V, T, lab, uvc, g_
        idx = gen.choice(len(T), m_, replace=False, p=wz / wz.sum())
        root = V[T[idx]].mean(1)
        ni = nt[idx]
        fl = p.fluff
        zi = z[idx]
        g = _flow_dir(C, root, ni, tube=tail is not None)
        aj = np.radians(gen.uniform(-14.0, 14.0, m_))[:, None]
        g = np.cos(aj) * g + np.sin(aj) * np.cross(ni, g)
        L = gen.uniform(0.012, 0.022, m_) * (0.8 + 0.6 * fl) / S * (0.85 + 0.3 * zi) * np.clip(root[:, 2] / 0.05, 0.45, 1.0)
        if tail is None:
            qh = ((root - hc) @ hR) / hs
            L = np.where(np.sqrt(((qh / HEAD_E) ** 2).sum(1)) < 1.3, 0.62 * L, L)       # shorter hair on the skull
        L = np.minimum(L, 0.005 + 3.0 * np.sqrt(ar[idx]))
        sd = np.cross(ni, g)
        sg = np.where(sd[:, 2] < 0.0, 1.0, -1.0) * np.where(gen.random(m_) < 0.8, 1.0, -1.0)   # most tips droop
        bend = sg * gen.uniform(0.05, 0.22, m_) * L
        P0 = (root + L[:, None] * g + bend[:, None] * sd).astype(np.float32)
        lift = (gen.uniform(0.0008, 0.0020, m_) * (0.7 + 0.6 * zi)).astype(np.float32)
        keep = np.ones(m_, bool)
        if tail is None:
            out = np.minimum(np.maximum(F.at(P0), 0.0) * (TIP_KEEP + 0.25 * fl), 0.007)
            apex, _n = place(P0, lift + out)
            ga = coat_g(P0)
            if RS:                                                # never across a bar: it would cut the bar in two
                keep &= np.maximum(bar_g(apex), bar_g(0.5 * (root + apex))) < 0.05
        else:
            apex = P0 + ni * (lift + gen.uniform(0.0005, 0.0035, m_) * (0.5 + fl))[:, None]
            ga = tail(apex, ni)
        if zones:                                                 # nor across the edge of a white marking
            keep &= wfield(root) * wfield(P0) > 0.0
        keep &= np.cumsum(keep) <= nmax
        idx, apex, ga, L, bend = (a[keep] for a in (idx, apex, ga, L, bend))
        k = len(idx)
        if k == 0:
            return V, T, lab, uvc, g_
        x = len(V) + np.arange(k)
        a_, b_, c_ = T[idx, 0], T[idx, 1], T[idx, 2]
        ux = uvc[idx].mean(1) + np.stack([bend, L], 1)
        T2 = np.concatenate([T, np.stack([b_, c_, x], 1), np.stack([c_, a_, x], 1)])
        T2[idx] = np.stack([a_, b_, x], 1)
        uv2 = np.concatenate([uvc, np.stack([uvc[idx, 1], uvc[idx, 2], ux], 1), np.stack([uvc[idx, 2], uvc[idx, 0], ux], 1)])
        uv2[idx] = np.stack([uvc[idx, 0], uvc[idx, 1], ux], 1)
        return (np.concatenate([V, apex]).astype(np.float32), T2, np.concatenate([lab, lab[idx], lab[idx]]), uv2,
                np.concatenate([g_, ga]).astype(np.float32))

    # skin + tufts + ribbons
    Pc = V[T].mean(1)
    z = np.full(len(T), 0.03 + 0.30 * p.fluff, np.float32)
    for zn in C.zones:
        z = np.maximum(z, min(1.0, zn["boost"] / 4.5) * (1.0 - _sstep(0.0, zn["fall"], zn["fn"](Pc))))
    mz_t, ey_t = face_terms(Pc)
    qt = ((Pc - hc) @ hR) / hs
    face = (np.sqrt(((qt / HEAD_E) ** 2).sum(1)) < 1.25) & (qt[:, 1] < -0.020) & (np.abs(qt[:, 0]) < 0.036)
    z = z * (1.0 - np.maximum(mz_t, ey_t))
    z[face | (lab == 2) | (Pc[:, 2] < 0.016)] = -1.0
    for px, py in C.paws:
        z[(np.hypot(Pc[:, 0] - px, Pc[:, 1] - py) < 0.040) & (Pc[:, 2] < 0.040)] = -1.0
    for e in C.ears:
        z[np.linalg.norm(Pc - e["base"], axis=1) < 0.024 * hs] = -1.0
    if zones:                                                              # the border of the white stays a clean soft line
        z[np.abs(wfield(Pc)) < 0.006] = -1.0
    nt_ = _unit(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]))
    opn = np.minimum(F.at(Pc + nt_ * 0.012) / 0.012, F.at(Pc + nt_ * 0.035) / 0.035)
    z[opn < 0.72] = -1.0                                                   # none in the folds
    if C.ring_fit is not None:
        nc, na_, _dirs, rad_ = C.ring_fit
        ax_ = (Pc - nc) @ na_
        z[(np.abs(ax_) < 0.016) & (np.linalg.norm(Pc - nc - ax_[:, None] * na_, axis=1) < float(rad_.max()) + 0.02)] = -1.0
    g_skin = coat_g(V, F.normal(V))
    ac3, al3 = _across(C, V, T)
    Vb, Tb, labb, uvb, gb = tufts(V, T, lab, np.stack([ac3, al3], -1), g_skin, z, n_lock - n_tail)
    if RV:
        Vr, Tr = np.concatenate(RV), np.concatenate(RT)
        ac3, al3 = _across(C, Vr, Tr)
        Tb = np.concatenate([Tb, Tr + len(Vb)])
        Vb = np.concatenate([Vb, Vr])
        labb = np.concatenate([labb, np.zeros(len(Tr), np.int32)])
        uvb = np.concatenate([uvb, np.stack([ac3, al3], -1)])
        gb = np.concatenate([gb, np.concatenate(RG)])
    body = fur_mesh("cat_body", Vb, Tb, labb, uvb, gb)

    # tail: round tube on a smooth centre line; the rings of a tabby (and its dark tip) are tone only
    tp, tr = C.tail[0][2:], C.tail[1][2:]
    Sx = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(tp, axis=0), axis=1))]).astype(np.float32)
    Lt = float(Sx[-1])
    if tabby:
        cen_t, hws_t = _bar_list(br, 0.012, Lt - 0.050 / S, 0.024 / S, 0.3, 0.0042 / S)
        gt_ = np.diff(cen_t)
        gt_ = np.minimum(np.concatenate([gt_[:1], gt_]), np.concatenate([gt_, gt_[-1:]]))
        et = np.minimum(hws_t + 0.0045 / S, 0.44 * gt_).astype(np.float32)
        lines_t = np.stack([cen_t - et, cen_t, cen_t + et], 1).ravel()
        stn = np.concatenate([[0.0], lines_t, np.arange(float(lines_t[-1]) + 0.012, Lt - 0.006, 0.013), [Lt]])
        stn = np.sort(stn[(stn >= 0.0) & (stn <= Lt)])
        stn = stn[np.concatenate([[True], np.diff(stn) > 0.003])]
        cen_t, et = np.concatenate([cen_t, [Lt + 0.004]]).astype(np.float32), np.concatenate([et, [0.050 / S]]).astype(np.float32)
    else:
        stn = np.linspace(0.0, Lt, 24)

    def tail_g(P, N, s=None):
        g = coat_g(P, N, tail=True)
        if tabby:
            if s is None:                                       # arc length of a point near the tube
                s = Sx[np.argmin(np.linalg.norm(P[:, None, :] - tp[None], axis=2), 1)]
                s = s + ((P - _curve(Sx, tp, s)) * _unit(_curve(Sx, tp, s + 0.002) - _curve(Sx, tp, s - 0.002))).sum(1)
            k = _near(s, cen_t)
            ring = (0.45 + 0.55 * breaks(P, k + 80, 0.15, 0.40)) * np.clip(1.0 - np.abs(s - cen_t[k]) / et[k], 0.0, 1.0)
            g = np.where(wfield(P) < 0.0004, g, g + coat["bar"] * ring) if zones else g + coat["bar"] * ring
        return np.clip(g, 0.0, 1.0).astype(np.float32)

    Vt, Tt, At, st, Nt = _tube(_curve(Sx, tp, stn), np.interp(stn, Sx, tr).astype(np.float32), stn)
    vt3 = At[Tt] * C.tailch["rbar"]
    vt3 = vt3[:, :1] + (vt3 - vt3[:, :1] + pi * C.tailch["rbar"]) % (2 * pi * C.tailch["rbar"]) - pi * C.tailch["rbar"]
    labt = (wfield(Vt[Tt].mean(1)) < 0).astype(np.int32) if zones else np.zeros(len(Tt), np.int32)
    zt = np.where((Vt[Tt].mean(1)[:, 2] > 0.010) & (st[Tt].mean(1) < Lt - 0.016 / S), 0.5, -1.0).astype(np.float32)   # the tip stays round
    uvt = np.stack([vt3, st[Tt]], -1)
    tail = fur_mesh("cat_tail", *tufts(Vt, Tt, labt, uvt, tail_g(Vt, Nt, st), zt, n_tail, tail=tail_g))
    furs = [body, tail]
    g_ear = float(coat_g(np.array([e["base"] for e in C.ears], np.float32)).mean())
    for k, e in enumerate(C.ears):
        o, vv, inner, hollow = _ear_mesh(e, k)
        assign(o, fur)
        o.data.materials.append(skin)
        mi = np.zeros(len(o.data.polygons), np.int32)
        mi[hollow] = 1
        o.data.polygons.foreach_set("material_index", mi)
        smooth(o, 180)
        paint(o, grime=_pv(np.where(inner > 0.5, 0.18 + 0.40 * (1.0 - np.clip(vv, 0.0, 1.0)), g_ear + 0.10 * np.clip(vv, 0.0, 1.0))), wear=0.0, special=0.0, mode="set")
        furs.append(o)

    # ---- face, whiskers, collar
    acc = _accessories(C, rng.fork("acc"), p)
    iris = mat("ceramic_glazed", p.eye_colour, grime=1.15)
    up = hR @ np.array([0, 0, 1], np.float32)
    eyes = []
    for o, ec, g, re in C.eyes:
        assign(o, iris)
        smooth(o, 180)
        cf = ec + g * re

        def limbal(co, nr, cf=cf):
            d = np.asarray(co, np.float32) - cf
            return np.clip(1.0 * _sstep(0.0045 * hs, 0.0095 * hs, np.linalg.norm(d, axis=1)) + 0.6 * _sstep(0.000 * hs, 0.005 * hs, d @ up), 0.0, 1.0)

        paint(o, grime=limbal)
        eyes.append(o)
    for o in C.pupils:
        assign(o, dark)
        smooth(o, 180)
    for o in C.shines:
        assign(o, mat("emissive", "#c3d98a", glow=0.02 + 0.16 * p.eyeshine))
        smooth(o, 180)
    for o in C.corneas:
        assign(o, mat("glass_clear", opacity=0.12))
        smooth(o, 180)
    for o in acc["nose"]:
        assign(o, skin)
        smooth(o, 180)
        paint(o, grime=0.35)
    for o in acc["lines"]:
        assign(o, dark)
        smooth(o, 80)
    for o in acc["whiskers"]:
        assign(o, fur_w)
        smooth(o, 80)
        paint(o, grime=0.0, wear=0.0, special=0.0, mode="set")
    for o in acc["collar"]:
        assign(o, mat("canvas", p.collar_colour, wear=0.001, grime=0.15, scale=4.0))
        smooth(o, 35)
        paint(o, grime=0.0, wear=0.0, special=0.0, mode="set")                    # one colour inside and out
    for o in acc["brass"]:
        apply_transform(o)
        assign(o, mat("brass"))
        smooth(o, 60)
    everything = furs + eyes + C.pupils + C.shines + C.corneas + acc["nose"] + acc["lines"] + acc["whiskers"] + acc["collar"] + acc["brass"]

    # ---- size / mirror
    realize(everything)
    mx = -1.0 if p.mirror else 1.0
    for o in everything:
        transform_verts(o, lambda co: co.__class__((co.x * S * mx, co.y * S, max(co.z, 0.0) * S)))
        if p.mirror:
            o.data.flip_normals()
