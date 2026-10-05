"""A small flock of feral town pigeons on the paving. Every bird is a skinned body (rump - belly - crop - neck -
head) lofted along a posed spine and then feathered the way a real wing folds: four shingled tiers of scalloped
coverts / secondaries riding on the flank (the two wing bars are simply the dark tips of the greater coverts and
of the secondaries), a stack of separate dark primaries lying over the rump, and a fan of overlapping tail
feathers with the dark terminal band on a thin tail core. Cere, beak, eyes, feathered thighs, scaly red tarsi
and four-toed clawed feet finish it. All plumage colour is painted per vertex on ONE matte feather material
(slate head, pale wing shield, white rump, charcoal bars, pied patches, dark / checker morphs); the neck gloss is
a single metallic patch that shifts purple -> green and fades into the slate. Poses, morphs, headings, head
turns and placement come from the seed; torn bread crumbs, droppings and a shed feather tell why they are here.
"""
from tk import *
import math
import numpy as np

META = {"name": "Pigeons", "class": "prop_small", "mount": "ground", "focus": [0.0, 0.0, 0.12]}
PARAMS = {
    "count": {"default": 5, "min": 2, "max": 7, "desc": "number of birds"},
    "spread": {"default": 1.2, "min": 0.6, "max": 3.0, "desc": "diameter of the patch the group occupies (m)"},
    "poses": {"default": "mixed", "choices": ["mixed", "feeding", "resting", "alert"],
              "desc": "behaviour of the group: mixed / pecking at crumbs / fluffed-up roosting / heads up, about to fly"},
    "white": {"default": 1, "min": 0, "max": 7, "desc": "how many of the birds are white"},
    "plumage": {"default": "mixed", "choices": ["mixed", "blue_bar", "checker", "dark", "pied"],
                "desc": "plumage morph of the non-white birds"},
    "size": {"default": 1.0, "min": 0.85, "max": 1.15, "desc": "overall bird size factor (1 = 32 cm beak to tail)"},
    "fluff": {"default": 0.25, "min": 0.0, "max": 1.0, "desc": "how puffed-up the feathers are (cold night = 1)"},
    "crumbs": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "amount of scattered bread crumbs"},
    "droppings": {"default": 3, "min": 0, "max": 8, "desc": "number of droppings on the paving"},
    "sheen": {"default": "#8f84ad", "desc": "base tint of the iridescent neck gloss (the nape shifts to green)"},
}
VARIANTS = {
    "feeding_flock": {"count": 7, "spread": 1.7, "poses": "feeding", "white": 1, "crumbs": 1.0, "seed": 11},
    "roosting_pair": {"count": 2, "spread": 0.6, "poses": "resting", "white": 0, "fluff": 1.0, "crumbs": 0.0,
                      "plumage": "blue_bar", "droppings": 6, "size": 1.08, "seed": 4},
    "alert_dark_trio": {"count": 3, "spread": 1.0, "poses": "alert", "white": 0, "plumage": "dark", "crumbs": 0.15,
                        "fluff": 0.0, "sheen": "#7f8ab8", "seed": 23},
    "white_doves": {"count": 4, "spread": 0.9, "poses": "mixed", "white": 4, "crumbs": 0.4, "droppings": 1,
                    "size": 0.94, "seed": 31},
}

# materials: feathers (grey), feathers (white), neck gloss, bare skin.  Vertex masks: (grime, wear, special).
FEA, WHT, IRI, SKIN = range(4)

POSES = {
    #        body pitch, body centre height, head centre (x, z), head pitch, neck girth, feet x (L, R)
    "stand": dict(beta=20.0, zc=9.7, head=(8.0, 21.0), hp=-6.0, neck=1.06, feet=(0.5, 0.1), legs=True, fl=1.0),
    "alert": dict(beta=31.0, zc=10.3, head=(7.2, 24.2), hp=3.0, neck=0.86, feet=(0.6, 0.2), legs=True, fl=0.94),
    "walk": dict(beta=11.0, zc=9.5, head=(10.2, 19.6), hp=-9.0, neck=0.94, feet=(3.0, -2.4), legs=True, fl=1.0),
    "peck": dict(beta=-27.0, zc=9.5, head=(11.2, 3.9), hp=-63.0, neck=0.9, feet=(0.8, -0.6), legs=True, fl=1.0),
    "roost": dict(beta=13.0, zc=5.9, head=(6.2, 14.6), hp=-4.0, neck=1.36, feet=(0, 0), legs=False, fl=1.17),
}
POSE_MIX = {
    "mixed": [("peck", 0.34), ("stand", 0.24), ("walk", 0.2), ("roost", 0.12), ("alert", 0.1)],
    "feeding": [("peck", 0.62), ("walk", 0.24), ("stand", 0.14)],
    "resting": [("roost", 0.62), ("stand", 0.3), ("alert", 0.08)],
    "alert": [("alert", 0.62), ("stand", 0.2), ("walk", 0.18)],
}
POSE_LEAD = {"mixed": ["stand", "peck", "roost", "walk"], "feeding": ["peck", "peck", "walk"],
             "resting": ["roost", "roost", "stand"], "alert": ["alert", "alert", "stand"]}

# level of detail: up to five birds get the full feather count, bigger flocks a leaner bird (stays inside 6k tris)
DETAIL = {
    "hi": dict(n=12, ss=[0, 0.5, 1, 2, 3, 3.6, 4.2, 4.7, 5.15, 5.6, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10.25, 11],
               nf=(5, 6, 6, 5), npr=5, nt=8, sides=6, leg=5, toe=3, core=6),
    "lo": dict(n=10, ss=[0, 1, 2, 3, 3.7, 4.4, 5.0, 5.6, 6.2, 6.8, 7.4, 8, 8.6, 9.3, 10.1, 11],
               nf=(4, 5, 5, 4), npr=4, nt=6, sides=5, leg=4, toe=2, core=5),
}

# folded wing carrier: tables over the nominal spine parameter s (rump 0 ... shoulder 4.5)
WS = [0.42, 0.86, 1.56, 1.9, 2.6, 3.4, 4.05, 4.5]
W_AT = [11, 10, 8, 8, 9, 12, 22, 40]              # dorsal edge angle (deg from the top of the back)
W_AB = [98, 102, 110, 115, 119, 116, 104, 78]     # ventral edge angle
W_MW = [2.7, 3.2, 3.9, 4.2, 0, 0, 0, 0]           # the wing keeps its own width where the body tapers to the rump
W_MH = [2.5, 2.8, 3.2, 3.4, 0, 0, 0, 0]
W_PR = [1, 1, 1, 1, 1, 1, 0.9, 0.75]              # how proud of the body (0 = tucked into the breast feathers)
W_SL = [0.4, 0.5, 0.55, 0.55, 0.3, 0.1, 0, 0]     # feather rows slant: dorsal end further forward
W_FR = [0.0, 0.14, 0.45, 0.82, 1.0]
W_DJ = [0.14, 0.32, 0.46, 0.43, 0.14]
G_S = [0.0, 1.2, 2.0, 3.0, 3.7, 4.3, 4.9, 5.5, 7.4, 8.0, 9.0, 11.0]            # body darkness along the spine
G_V = [0.4, 0.34, 0.24, 0.3, 0.48, 0.66, 0.76, 0.8, 0.8, 0.72, 0.64, 0.66]
IR_RIM, IR_IN, IR_GREEN, IR_LO = 0.8, 0.46, 0.57, 0.24             # neck gloss: darkness at its rim / inside, green strength


def sst(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Soup:
    """Accumulates lofted tubes / feather patches (verts, faces, per-face material, per-vertex masks) into one mesh."""

    def __init__(self):
        self.v, self.k, self.f, self.m, self.uv = [], [], [], [], []
        self.uvo = (0.0, 0.0)
        self.tris = 0

    def _emit(self, V, K, F, M, UV, inside=None):
        V = np.asarray(V, float)
        ref = V.mean(axis=0) if inside is None else np.asarray(inside, float)
        tot = 0.0
        for f in F:                                   # orient away from the reference point
            q = V[list(f)]
            nr = np.cross(q[2] - q[0], q[3] - q[1]) if len(f) == 4 else np.cross(q[1] - q[0], q[2] - q[0])
            tot += float(np.dot(nr, q.mean(axis=0) - ref))
        if tot < 0:
            F = [tuple(reversed(f)) for f in F]
            UV = [tuple(reversed(u)) for u in UV]
        o = len(self.v)
        self.v.extend(V.tolist())
        self.k.extend(np.broadcast_to(np.asarray(K, float), (len(V), 3)).tolist())
        self.f.extend(tuple(i + o for i in f) for f in F)
        self.m.extend(M)
        self.uv.extend(UV)
        self.tris += sum(len(f) - 2 for f in F)

    def tube(self, rings, mat_, K, caps=(True, True), cap_mat=None, inside=None, skip=None):
        """rings: list of (n,3) arrays; mat_: id | per-segment list | fn(i, j); K: masks (3,) | (R,3) | (R,n,3).
        UVs in metres: U along the loft (feather direction), V around it."""
        R, n = len(rings), len(rings[0])
        rings = [np.asarray(r, float) for r in rings]
        V = np.concatenate(rings, axis=0)
        K = np.asarray(K, float)
        if K.ndim == 2:
            K = np.repeat(K[:, None, :], n, axis=1)
        K = np.broadcast_to(K, (R, n, 3)).reshape(-1, 3)
        cen = [r.mean(axis=0) for r in rings]
        Ls = [0.0]
        for i in range(1, R):
            Ls.append(Ls[-1] + float(np.linalg.norm(cen[i] - cen[i - 1])))
        per = [float(np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1).sum()) for r in rings]
        u0, v0 = self.uvo

        def uvp(i, j):
            return (u0 + Ls[i], v0 + (j - n * 0.5) * per[i] / n)

        def fm(i, j):
            return mat_(i, j) if callable(mat_) else mat_[i] if isinstance(mat_, (list, tuple)) else mat_

        F, M, UV = [], [], []
        for i in range(R - 1):
            for j in range(n):
                if skip is not None and skip[i][j]:   # left open: another piece covers it
                    continue
                a, b = i * n + j, i * n + (j + 1) % n
                F.append((a, b, b + n, a + n))
                UV.append((uvp(i, j), uvp(i, j + 1), uvp(i + 1, j + 1), uvp(i + 1, j)))
                M.append(fm(i, j))
        if caps[0]:
            F.append(tuple(range(n - 1, -1, -1)))
            UV.append(tuple(uvp(0, j) for j in range(n - 1, -1, -1)))
            M.append(fm(0, 0) if cap_mat is None else cap_mat[0])
        if caps[1]:
            F.append(tuple(range((R - 1) * n, R * n)))
            UV.append(tuple(uvp(R - 1, j) for j in range(n)))
            M.append(fm(R - 2, 0) if cap_mat is None else cap_mat[1])
        self._emit(V, K, F, M, UV, inside)

    def patch(self, V, F, mat_, K, UVv, inside=None):
        """Free-form piece: verts, faces, material id | per-face list, masks (3,) | (nv,3), per-vertex UVs (m)."""
        u0, v0 = self.uvo
        UV = [tuple((u0 + UVv[i][0], v0 + UVv[i][1]) for i in f) for f in F]
        self._emit(V, K, F, list(mat_) if isinstance(mat_, (list, tuple)) else [mat_] * len(F), UV, inside)

    def ribbon(self, rows, mat_, K, tip=None, inside=None):
        """Quad strip through rows of m points running across (blade / tent sections); optional closing tip point."""
        rows = [np.asarray(r, float) for r in rows]
        R, m = len(rows), len(rows[0])
        V, UVv, F = [], [], []
        u = 0.0
        for i, rw in enumerate(rows):
            if i:
                u += float(np.linalg.norm(rw.mean(axis=0) - rows[i - 1].mean(axis=0)))
            acc = 0.0
            for j in range(m):
                if j:
                    acc += float(np.linalg.norm(rw[j] - rw[j - 1]))
                V.append(rw[j])
                UVv.append((u, acc))
        for i in range(R - 1):
            for j in range(m - 1):
                a = i * m + j
                F.append((a, a + 1, a + m + 1, a + m))
        if tip is not None:
            tip = np.asarray(tip, float)
            u += float(np.linalg.norm(tip - rows[-1].mean(axis=0)))
            V.append(tip)
            UVv.append((u, UVv[-1][1] * 0.5))
            for j in range(m - 1):
                F.append(((R - 1) * m + j, (R - 1) * m + j + 1, R * m))
        K = np.asarray(K, float)
        if K.ndim == 2 and len(K) != len(V):          # per row (+ tip)
            if tip is not None and len(K) == R:
                K = np.concatenate([K, K[-1:]], axis=0)
            K = np.concatenate([np.repeat(K[:R], m, axis=0), K[R:]], axis=0)
        self.patch(np.array(V), F, mat_, K, UVv, inside)

    def finish(self, name, mats):
        if not self.f:
            return None
        V = np.array(self.v)
        V[:, 2] = np.maximum(V[:, 2], 0.0)
        obj = mesh_object(name, [tuple(v) for v in V], self.f)
        me = obj.data
        used = sorted(set(self.m))
        remap = {k: i for i, k in enumerate(used)}
        for k in used:
            me.materials.append(mats[k])
        if len(me.polygons) == len(self.f):
            for poly, mi in zip(me.polygons, self.m):
                poly.material_index = remap[mi]
            uvl = me.uv_layers.new(name="UVMap")
            for poly, uvs in zip(me.polygons, self.uv):
                for li, uv in zip(poly.loop_indices, uvs):
                    uvl.data[li].uv = uv
            uv_keep(obj)
        if len(me.vertices) == len(self.k):           # plumage / dirt masks straight onto the vertices
            Kk = np.clip(np.array(self.k, dtype=np.float32), 0.0, 1.0)
            for c, nm in enumerate(("m_grime", "m_wear", "m_special")):
                at = me.attributes.get(nm) or me.attributes.new(nm, 'FLOAT', 'POINT')
                at.data.foreach_set("value", np.ascontiguousarray(Kk[:, c]))
        return obj


def cr_sample(K, s):
    """Catmull-Rom interpolation of key rows K at (fractional index) parameters s."""
    K = np.asarray(K, float)
    n = len(K)
    out = []
    for t in s:
        i = int(min(max(math.floor(t), 0), n - 2))
        f = t - i
        p0, p1, p2, p3 = K[max(i - 1, 0)], K[i], K[i + 1], K[min(i + 2, n - 1)]
        out.append(0.5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (3 * p1 - p0 - 3 * p2 + p3) * f ** 3))
    return np.array(out)


def ring_xz(c, th, w, hu, hd, n):
    """Section ring in the bird's sagittal frame: centre c=(x,z), plane perpendicular to the direction th (rad).
    Half-width w, radii hu (dorsal) / hd (ventral). Point 0 is the top of the back."""
    ux, uz = -math.sin(th), math.cos(th)
    pts = []
    for j in range(n):
        a = 2 * math.pi * j / n
        ca, sa = math.cos(a), math.sin(a)
        h = hu if ca >= 0 else hd
        pts.append((c[0] + h * ca * ux, w * sa, c[1] + h * ca * uz))
    return np.array(pts)


def ring_dir(c, d, ra, rb, n, ref=(0.0, 1.0, 0.0)):
    """Elliptic ring around centre c perpendicular to direction d (ra along ref x d, rb along the other)."""
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    a = np.cross(ref, d)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = np.cross(d, a)
    c = np.asarray(c, float)
    return np.array([c + a * ra * math.cos(2 * math.pi * j / n) + b * rb * math.sin(2 * math.pi * j / n) for j in range(n)])


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-9)


def make_bird(S, pose, plum, r, size, fluff, pos, yaw_deg, head_yaw, D):
    P = POSES[pose]
    fl = (1.0 + 0.1 * fluff) * P["fl"]
    beta = math.radians(P["beta"] + r.jitter(0, 3.0))
    zc = P["zc"] * (1.0 + 0.04 * (fl - 1.0) * 10 * (0 if P["legs"] else 1))
    cb, sb = math.cos(beta), math.sin(beta)

    def bl(u, v):                     # body-local (cm) -> bird sagittal plane
        return np.array([u * cb - v * sb, u * sb + v * cb + zc])

    # ---- spine keys: [x, z, half-width, dorsal r, ventral r, direction]
    B = np.array([[-9.0, 0.6, 1.3, 0.8, 0.9], [-7.0, 0.5, 3.0, 2.0, 2.3], [-3.8, 0.2, 4.8, 3.5, 4.2], [0.0, 0.0, 5.6, 4.2, 5.2],
                  [3.6, 0.3, 5.3, 4.2, 5.3], [6.2, 1.6, 4.0, 3.6, 4.9]])
    B[1:, 2:5] *= fl
    hp = math.radians(P["hp"] + r.jitter(0, 5.0))
    hc = np.array(P["head"], float) + np.array([r.jitter(0, 0.5), r.jitter(0, 0.5)])
    f, g = np.array([math.cos(hp), math.sin(hp)]), np.array([-math.sin(hp), math.cos(hp)])
    tipz = (hc + f * 4.2 - g * 1.07)[1]
    if tipz < 0.3:
        hc[1] += 0.3 - tipz
    H = [(-1.3, -0.9, 1.95, 1.9, 1.9), (0.0, 0.0, 1.9, 1.95, 1.8), (1.3, -0.1, 1.45, 1.45, 1.3), (2.1, -0.35, 0.72, 0.7, 0.6)]
    Pb = bl(B[5, 0], B[5, 1])
    Qh = hc + f * H[0][0] + g * H[0][1]
    d = (Qh - Pb) / (np.linalg.norm(Qh - Pb) + 1e-9)
    fw = np.array([d[1], -d[0]])
    nk = P["neck"] * (1.0 + 0.06 * fluff)
    keys = [[*bl(b[0], b[1]), b[2], b[3], b[4]] for b in B]
    keys.append([*(Pb + (Qh - Pb) * 0.36 + fw * 0.25), 2.75 * nk, 2.45 * nk, 2.6 * nk])
    keys.append([*(Pb + (Qh - Pb) * 0.72 - fw * 0.05), 2.2 * nk, 2.0 * nk, 2.05 * nk])
    for h in H:
        keys.append([*(hc + f * h[0] + g * h[1]), h[2], h[3], h[4]])
    K = np.array(keys)
    nd = math.atan2(d[1], d[0])
    th = []
    for i in range(len(K)):
        a, b = K[max(i - 1, 0)], K[min(i + 1, len(K) - 1)]
        th.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    for k, t in zip(range(8, 12), (0.35, 0.6, 0.85, 1.0)):
        th[k] = nd + (hp - nd) * t
    th = np.unwrap(np.array(th))
    K = np.column_stack([K, th])

    piv = K[7, 0]
    hy = math.radians(head_yaw)

    def yawed(pts, wgt):              # turn the head about the upper neck
        if abs(hy * wgt) < 1e-4:
            return np.array(pts, float)
        a = hy * wgt
        ca, sa = math.cos(a), math.sin(a)
        q = np.array(pts, float)
        x, y = q[:, 0] - piv, q[:, 1].copy()
        q[:, 0] = piv + x * ca - y * sa
        q[:, 1] = x * sa + y * ca
        return q

    sc = size * 0.01
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def W(pts):                       # bird-local cm -> asset space m
        q = np.atleast_2d(np.array(pts, float)) * sc
        x, y = q[:, 0].copy(), q[:, 1].copy()
        q[:, 0] = pos[0] + x * cy - y * sy
        q[:, 1] = pos[1] + x * sy + y * cy
        q[:, 2] = np.maximum(q[:, 2], 0.0012 if not P["legs"] else 0.0)
        return q

    # ---- plumage: everything is one feather material; the look is painted into the vertex masks
    white = plum == "white"
    FM = WHT if white else FEA
    dgl = r.jitter(0.0, 0.04) + (0.06 if plum == "checker" else 0.0)
    pk = [np.array([0.42, 0.31, 0.23]) * r.pick([-1, 1]), np.array([-0.27, 0.45, 0.36]) * r.pick([-1, 1]), np.array([0.19, -0.38, 0.5])]
    pph = [r.u(0, 6.283) for _ in range(3)]
    pbias = r.u(0.05, 0.4)

    def tone(Pl, g_, w_=0.0, d_=0.0, pm=1.0):
        """masks (grime = darkness, wear = white, special = street dirt) for feather vertices at local points Pl"""
        Pl = np.atleast_2d(np.asarray(Pl, float))
        m = len(Pl)
        g_ = np.broadcast_to(np.asarray(g_, float), (m,)).copy()
        w_ = np.broadcast_to(np.asarray(w_, float), (m,)).copy()
        d_ = np.broadcast_to(np.asarray(d_, float), (m,)).copy()
        if white:
            return np.column_stack([g_ * 0.0, w_ * 0.0, d_])
        if plum == "dark":            # sooty 'spread' morph: dark all over, bars and tail band barely darker
            g_ = 0.76 + 0.24 * g_
            w_ *= 0.0
        g_ = np.clip(g_ + dgl, 0.0, 1.0)
        if plum == "pied":            # the grey pales towards the white patches (those faces use the white feathers)
            pw = pied(Pl) * pm
            g_ = g_ * (1.0 - pw)
            w_ = np.maximum(w_, 0.3 * pw)
        return np.column_stack([g_, w_, d_])

    def pied(Pl):
        """0..1 whiteness of the seeded pied pattern: irregular patches wandering over head, wings, rump, flights"""
        Pl = np.atleast_2d(np.asarray(Pl, float))
        if plum != "pied":
            return np.zeros(len(Pl))
        nz = sum(np.sin(Pl @ k + ph) for k, ph in zip(pk, pph))
        return sst(0.25, 0.7, nz * 0.55 + pbias)

    def fmat(Pl, pm=1.0):             # feather material of a face / a whole feather
        return WHT if white or float(pied(Pl).mean()) * pm > 0.5 else FM

    c0 = bl(0.0, 0.0)
    centre = W([[c0[0], 0.0, c0[1]]])[0]

    # ---- body skin
    n, ss = D["n"], D["ss"]
    SK = cr_sample(K, ss)
    A = np.arange(n) * 360.0 / n
    dev = np.minimum(A, 360.0 - A)                    # angle from the top of the back
    dors = np.cos(np.radians(A))
    # wandering lower / upper edge of the neck gloss over the ring angle (see below)
    ph1, ph2 = r.u(0, 6.283), r.u(0, 6.283)
    an = np.radians(np.append(A, 360.0))
    dv_ = np.minimum(np.append(A, 360.0), 360.0 - np.append(A, 360.0))
    lo_v = (5.3 if pose == 'roost' else 4.95) + 0.3 * np.sin(2.0 * an + ph1) + 0.16 * np.sin(5.0 * an + ph2) - 0.35 * sst(100.0, 180.0, dv_)
    hi_v = 7.7 + 0.16 * np.sin(3.0 * an + ph2)
    rings, masks, lring = [], [], []

    def pmb(s):                       # pied white keeps off the dark breast and neck: head, back, rump and belly only
        return 1.0 - float(sst(3.2, 3.9, s) * sst(8.3, 7.7, s))

    for s, k in zip(ss, SK):
        rg = yawed(ring_xz((k[0], k[1]), k[5], k[2], k[3], k[4], n), float(smoothstep(6.0, 8.2, s)))
        rings.append(W(rg))
        lring.append(rg)
        trunk = 1.0 if 1.0 < s < 4.6 else 0.0
        g_ = np.interp(s, G_S, G_V) + trunk * (0.06 * np.clip(dors, 0, 1) - 0.07 * np.clip(-dors, 0, 1))
        # dark slate hood: the feathers around the gloss are as dark as its rim, fading out over breast and head
        g_ = np.maximum(g_, 0.93 * sst(lo_v[:n] - 0.95, lo_v[:n] - 0.15, s) * sst(hi_v[:n] + 0.75, hi_v[:n] + 0.1, s))
        w_ = 0.85 * sst(2.4, 1.4, s) * sst(-0.1, 0.5, dors)                    # pale rump
        d_ = 0.55 * sst(0.25, 0.9, -dors) * sst(0.0, 0.6, s) * sst(4.2, 3.2, s) * np.array([r.u(0.45, 1.0) for _ in range(n)])
        masks.append(tone(rg, g_, w_, d_, pm=pmb(s)))
    masks = np.array(masks)
    Rn = len(ss)
    isI = np.zeros((Rn - 1, n), bool)
    if not white:
        # iridescent neck: ONE patch from the crop to under the head with wandering edges. It is its own skin
        # piece of metallic gloss let into the body: at its rim it is as dark as the slate around it, further in
        # it opens to purple and then to green on the nape and the sides of the upper neck - no outline.
        for j in range(n):
            lo, hi = 0.5 * (lo_v[j] + lo_v[j + 1]), 0.5 * (hi_v[j] + hi_v[j + 1])
            for i in range(Rn - 1):
                isI[i, j] = lo < 0.5 * (ss[i] + ss[i + 1]) < hi
        vid, Vi, Ki, UVi, Fi = {}, [], [], [], []

        def iv(i, j):
            j %= n
            if (i, j) not in vid:
                adj = [isI[a, b % n] for a in (i - 1, i) for b in (j - 1, j) if 0 <= a < Rn - 1]
                e = 0.0
                if all(adj):
                    e = float(sst(0.0, 1.0, min((ss[i] - lo_v[j]) / 1.3, (hi_v[j] - ss[i]) / 0.9)))
                gz = float(sst(4.9, 7.3, ss[i]) * (1.0 - 0.8 * sst(40.0, 170.0, dev[j]))) * (0.86 + 0.14 * math.sin(3.0 * an[j] + ph1 + 0.9 * ss[i]))
                vid[(i, j)] = len(Vi)
                Vi.append(rings[i][j])
                Ki.append((lerp(IR_RIM, IR_IN, e), 0.0, (IR_LO + (IR_GREEN - IR_LO) * gz) * float(sst(0.0, 0.85, e)) * r.u(0.93, 1.0)))
                UVi.append((ss[i] * 0.024 * size, j * 0.02 * size))
            return vid[(i, j)]

        for i in range(Rn - 1):
            for j in range(n):
                if isI[i, j]:
                    Fi.append((iv(i, j), iv(i, j + 1), iv(i + 1, j + 1), iv(i + 1, j)))
        if Fi:
            S.patch(np.array(Vi), Fi, IRI, np.array(Ki), UVi)
    S.tube(rings, lambda i, j: fmat([lring[i][j], lring[i][(j + 1) % n], lring[i + 1][j], lring[i + 1][(j + 1) % n]], pmb(0.5 * (ss[i] + ss[i + 1]))), masks, skip=isI)

    # ---- folded wings: shingled tiers of scalloped feathers riding on the body surface
    def wing_pt(sn, v, lift, side):
        s = max(sn + float(np.interp(sn, WS, W_SL)) * (0.5 - v), 0.0)
        a = math.radians(lerp(float(np.interp(sn, WS, W_AT)), float(np.interp(sn, WS, W_AB)), v))
        dd = -0.35 + (float(np.interp(v, W_FR, W_DJ)) + 0.35) * float(np.interp(sn, WS, W_PR)) + lift
        k = cr_sample(K, [s])[0]
        w_, hu_, hd_ = max(k[2], float(np.interp(sn, WS, W_MW)) * fl), max(k[3], float(np.interp(sn, WS, W_MH)) * fl), k[4]
        ux, uz = -math.sin(k[5]), math.cos(k[5])
        ca, sa = math.cos(a), math.sin(a)
        h = (hu_ if ca >= 0 else hd_) + dd
        return np.array([k[0] + h * ca * ux, side * (w_ + dd) * sa, k[1] + h * ca * uz])

    def s_of(t):
        return 4.5 - 4.05 * t

    chk = plum == "checker"
    g_sh = 0.07                       # pale wing shield
    #        t0,   t1,   shape row,  bar,  base tone, tip tone, lift
    tiers = [(0.0, 0.27, 0.11, False, 0.42, 0.62 if chk else g_sh, 0.13),          # lesser coverts / shoulder
             (0.19, 0.46, None, False, g_sh, 0.66 if chk else g_sh + 0.02, 0.14),  # median coverts
             (0.37, 0.67, -1, True, g_sh + 0.02, 1.0, 0.15),                       # greater coverts: front bar
             (0.57, 1.05, -1, True, g_sh + 0.06, 1.0, 0.16)]                       # secondaries / tertials: rear bar
    for side in (-1, 1):
        for (t0, t1, tm, bar, gb, gt, lift), nf in zip(tiers, D["nf"]):
            vb = [0.0] + [(j + r.jitter(0, 0.1)) / nf for j in range(1, nf)] + [1.0]
            ed = [0.0] + [1.0] * (nf - 1) + [0.0]
            bs = [float(sst(0.0, 0.24, v)) if bar else 1.0 for v in vb]         # bars fade out towards the back
            V, Tn, UVv, F = [], [], [], []

            def add(t, v, lf, gg):
                V.append(wing_pt(s_of(t), v, lf, side))
                Tn.append(gg + r.jitter(0, 0.025))
                UVv.append((t * 0.16 * size, v * 0.09 * size))
                return len(V) - 1

            # the shoulder tier starts with a rounded leading edge (bend of the wing) standing just proud of the breast
            base = [add(t0 + (0.1 * (2.0 * v - 1.0) ** 2 if t0 == 0.0 else 0.0), v, 0.0 if t0 == 0.0 else -0.06, gb) for v in vb]
            mid = None
            if tm is not None:
                if bar:           # the dark feather tips: ~1 cm, a little wider towards the lower wing edge
                    mid = [add(t1 - (0.062 + 0.03 * v), v, lift * 0.6 * e, gb + 0.06 * b) for v, e, b in zip(vb, ed, bs)]
                else:
                    mid = [add(tm, v, lift * 0.45 * e, lerp(g_sh, gt, 0.15)) for v, e in zip(vb, ed)]
            gn = g_sh if tm == 0.11 else gb
            notch = [add(t1 - 0.05, v, lift * 0.8 * e, lerp(gn, gt * 0.88, b) + 0.14) for v, e, b in zip(vb, ed, bs)]
            for j in range(nf):
                dv = vb[j + 1] - vb[j]
                tj = t1 + r.jitter(0, 0.012) - 0.03 * (j / max(nf - 1, 1) - 0.5) ** 2
                bj = 0.5 * (bs[j] + bs[j + 1]) if bar else 1.0
                tl = add(tj, vb[j] + 0.27 * dv, lift, lerp(gb, gt, bj))
                tr = add(tj + r.jitter(0, 0.006), vb[j] + 0.73 * dv, lift, lerp(gb, gt, bj))
                lo_l, lo_r = (mid[j], mid[j + 1]) if mid else (base[j], base[j + 1])
                if mid:
                    F.append((base[j], base[j + 1], mid[j + 1], mid[j]))
                F.append((lo_l, lo_r, tr, tl))
                F.append((lo_l, tl, notch[j]))
                F.append((lo_r, notch[j + 1], tr))
            S.patch(W(V), F, [fmat([V[i_] for i_ in f_]) for f_ in F], tone(V, Tn), UVv, inside=centre)

    # ---- tail: thin core + a rounded fan of overlapping feathers with the dark terminal band
    root = bl(-7.0, 0.5)
    Lt = 11.7
    rr = -8.0 + r.jitter(0, 3.0)
    while root[1] + Lt * math.sin(math.radians(rr) - beta) < 2.4 and rr < 40:
        rr += 2.0
    psi = math.radians(rr) - beta
    td = np.array([-math.cos(psi), 0.0, math.sin(psi)])
    tn = np.array([math.sin(psi), 0.0, math.cos(psi)])
    ey = np.array([0.0, 1.0, 0.0])
    root3 = np.array([root[0], 0.0, root[1]])
    fan = (1.18 if pose == "peck" else 1.0) * r.vary(1.0, 0.08)

    def hwid(dist):
        return float(np.interp(dist, [0.0, 3.5, 8.0, Lt], [2.0, 2.5, 3.0 * fan, 3.3 * fan]))

    def thick(dist):
        return float(np.interp(dist, [0.0, 3.5, 8.0, Lt], [1.25, 0.85, 0.5, 0.22]))

    def tail_top(dist, y):            # upper surface of the core (flat-topped hexagon section)
        q = abs(y) / (hwid(dist) * 0.9)
        return thick(dist) * 0.74 * float(np.clip((1.0 - q) / 0.5, 0.0, 1.0))

    nc = D["core"]
    cr_, ck = [], []
    for dist in ([0.0, 3.5, 8.0, Lt - 0.9] if nc == 6 else [0.0, 5.0, Lt - 0.9]):
        c = root3 + td * dist
        ring = []
        for j in range(nc):
            a = 2 * math.pi * j / nc
            ring.append(c + ey * hwid(dist) * 0.9 * math.cos(a) + tn * thick(dist) * (0.85 if math.sin(a) >= 0 else 0.7) * math.sin(a))
        cr_.append(W(ring))
        ck.append(tone(ring, 0.34 + 0.55 * float(sst(7.0, 10.0, dist)), 0.0, 0.3 * float(sst(6.0, 10.5, dist))))
    S.tube(cr_, FM, np.array(ck), caps=(False, True))
    nt = D["nt"]
    order = sorted(range(nt), key=lambda k: (abs(k + 0.5 - nt / 2), k))
    for k in range(nt):
        ck_ = (k + 0.5 - nt / 2) / (nt / 2)
        Lk = Lt * (1.0 - 0.15 * ck_ * ck_) + r.jitter(0, 0.12)
        zs = 0.07 + 0.05 * (nt - 1 - order.index(k))
        rows = []
        for fr, wsc in ((0.1, 1.0), (0.62, 1.0), (0.76, 1.0), (0.95, 0.72)):
            dist = Lk * fr
            yc = ck_ * hwid(dist) * 0.84
            hw_ = hwid(dist) * 0.84 / nt * 1.32 * wsc
            rows.append([root3 + td * dist + ey * y + tn * (tail_top(dist, y) + zs) for y in (yc - hw_, yc + hw_)])
        tipp = root3 + td * Lk + ey * ck_ * hwid(Lk) * 0.84 + tn * (tail_top(Lk, 0.0) + zs)
        tk_ = tone([rw[0] for rw in rows] + [tipp], [0.36, 0.4, 0.97, 0.95, 0.8], 0.0, [0.0, 0.0, 0.1, 0.25, 0.3])
        S.ribbon([W(rw) for rw in rows], fmat([rw[0] for rw in rows[1:]]), tk_, tip=W([tipp])[0], inside=W([root3 + td * 6.0 - tn * 4.0])[0])

    # ---- primaries: separate dark blades from under the secondaries, stacked and staggered, lying over the tail
    npr = D["npr"]
    over = r.pick([-1, 1])            # which wing tip lies on top where they cross
    for side in (-1, 1):
        for i in range(npr):
            vi = 0.9 - 0.5 * i / (npr - 1)
            R0 = wing_pt(1.35, vi, -0.1 + 0.03 * i, side)
            R0w = unit(wing_pt(1.35, vi + 0.06, 0.0, side) - wing_pt(1.35, vi - 0.06, 0.0, side))
            M0 = wing_pt(0.3, vi * 0.85 + 0.05, 0.0 + 0.04 * i, side)
            M0w = unit(unit(wing_pt(0.3, vi * 0.85 + 0.11, 0.0, side) - wing_pt(0.3, vi * 0.85 - 0.01, 0.0, side)) + ey * side * 0.4)
            di = Lt - 1.7 - 1.15 * i * (4.0 / (npr - 1))
            lat = 1.0 + 0.12 * i - (1.3 if side == over else 0.0)
            T0 = root3 + td * di + ey * side * lat + tn * (tail_top(di, 0.0) + 0.07 + 0.05 * nt + 0.1 + 0.06 * i + (0.14 if side == over else 0.0))
            T0w = unit(ey * side - tn * 0.25)
            P0 = lerp(M0, T0, 0.84)
            rows = [[c - w * hw_, c + w * hw_] for c, w, hw_ in ((R0, R0w, 0.8), (M0, M0w, 0.74), (P0, T0w, 0.5))]
            pk_ = tone([R0, M0, P0, T0], [0.74, 0.82, 0.92, 0.96])
            S.ribbon([W(rw) for rw in rows], fmat([M0, P0, T0]), pk_, tip=W([T0])[0], inside=W([root3 + td * 2.0 - tn * 6.0])[0])

    # ---- head: beak, cere, eyes
    def hw(a, b, y=0.0):
        q = hc + f * a + g * b
        return np.array([q[0], y, q[1]])

    ns = D["sides"]
    bk = [((1.85, -0.3), 0.58, 0.54), ((3.1, -0.5), 0.41, 0.36), ((3.9, -0.82), 0.23, 0.19), ((4.2, -1.07), 0.05, 0.05)]
    br = []
    for i, (c2, ra, rb) in enumerate(bk):
        a2 = bk[min(i + 1, len(bk) - 1)][0]
        b2 = bk[max(i - 1, 0)][0]
        dv = hw(a2[0] - b2[0], a2[1] - b2[1]) - hw(0, 0)
        br.append(W(yawed(ring_dir(hw(*c2), dv, ra, rb, ns), 1.0)))
    if white:                         # flesh-pink bill on the white birds, dark horn otherwise
        S.tube(br, SKIN, np.array([(0.0, 0.8, 0.0), (0.0, 0.85, 0.0), (0.1, 0.9, 0.0), (0.3, 0.9, 0.0)]), caps=(False, True))
    else:
        S.tube(br, FEA, np.array([(0.9, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.9, 0.15, 0.0)]), caps=(False, True))
    cc = hw(2.25, 0.22)
    f3, g3 = hw(1, 0) - hw(0, 0), hw(0, 1) - hw(0, 0)
    cer = []
    for ph in (0.16, 0.5, 0.84):
        ctr = cc + f3 * 0.55 * math.cos(ph * math.pi)
        cer.append(W(yawed(ring_dir(ctr, f3, 0.32 * math.sin(ph * math.pi), 0.52 * math.sin(ph * math.pi), 5), 1.0)))
    S.tube(cer, WHT, (0.06, 0.0, 0.0))
    for side in (-1, 1):
        er = []
        for (lat, rad) in [(1.5, 0.4), (1.8, 0.27), (1.87, 0.19)]:
            c3 = hw(0.6, 0.3, side * lat)
            er.append(W(yawed([c3 + (f3 * math.cos(2 * math.pi * j / ns) + g3 * math.sin(2 * math.pi * j / ns)) * rad for j in range(ns)], 1.0)))
        ik = (0.75, 0.0, 0.0) if white else (0.0, 0.55, 0.0)
        S.tube(er[:2], SKIN, np.array([(0.25, 0.3, 0.0), ik]), caps=(False, False))
        S.patch(er[2], [tuple(range(ns))], FEA, (1.0, 0.0, 0.0), [(0.0, 0.0)] * ns, inside=W(yawed([hw(0.6, 0.3)], 1.0))[0])

    # ---- legs and feet: feathered thigh, scaly red tarsus, three toes forward and one back, dark claws
    beak_tip = W(yawed([hw(4.2, -1.07)], 1.0))[0]
    if P["legs"]:
        nl = D["leg"]
        for side, fx in zip((1, -1), P["feet"]):
            hip = bl(-0.6, -2.4)
            hip = np.array([hip[0], side * 2.1, hip[1]])
            xf = fx + r.jitter(0, 0.4)
            yf = side * (2.5 + r.jitter(0, 0.3))
            ank = np.array([xf - 0.15, yf, 0.2])
            hock = np.array([xf - 1.3, yf * 0.97, min(3.6, hip[2] * 0.48)])
            ax = hock - ank
            S.tube([W(ring_dir(ank, ax, 0.27, 0.27, nl)), W(ring_dir(ank + ax * 0.5, ax, 0.21, 0.21, nl)), W(ring_dir(hock + ax * 0.12, ax, 0.27, 0.27, nl))],
                   SKIN, np.array([(0.12, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.0)]), caps=(False, False))
            thd = hip - hock
            trg = [ring_dir(hock - thd * 0.02, thd, 0.5, 0.48, nl), ring_dir(hock + thd * 0.45, thd, 1.2, 1.0, nl), ring_dir(hip, thd, 1.6, 1.3, nl)]
            if D["toe"] < 3:
                trg = [trg[0], trg[2]]
            S.tube([W(t_) for t_ in trg], fmat(trg[0]), np.array([tone(t_, 0.16, 0.0, 0.4) for t_ in trg]), caps=(False, False))
            fyaw = r.jitter(0, 12.0) + side * 6.0
            for ang, ln in ((-33, 2.5), (0, 3.0), (33, 2.5), (180, 1.4)):
                a = math.radians(ang + fyaw)
                dx, dy = math.cos(a), math.sin(a)
                secs = [(-0.25, 0.25, 0.36, 0.0), (0.5 * ln, 0.2, 0.3, 0.0), (0.84 * ln, 0.14, 0.2, 0.3), (ln + 0.3, 0.03, 0.04, 1.0)]
                if ang == 180 or D["toe"] < 3:
                    secs = [secs[0], (0.8 * ln, 0.16, 0.22, 0.2), secs[3]]
                rows = []
                for (dist, hw_, hh, _) in secs:
                    bx, by = xf + dx * dist, yf + dy * dist
                    rows.append(W([(bx - dy * hw_, by + dx * hw_, 0.0), (bx, by, hh), (bx + dy * hw_, by - dx * hw_, 0.0)]))
                S.ribbon(rows, SKIN, np.array([(gk, 0.0, 0.0) for (_, _, _, gk) in secs]))
    return beak_tip


def build(p, rng):
    mats = [mat("paper", "#9eaac2", scale=3.0, rough=1.15), mat("paper", "#f3f1ea", scale=3.0, rough=1.15),
            mat("brass", p.sheen, scale=9.0, rough=1.3, wear=0.0, grime=1.5), mat("leather", "#a8646c", scale=5.0, grime=0.6)]
    n = int(p.count)
    R = p.spread * 0.5
    D = DETAIL["hi" if n <= 5 else "lo"]

    # ---- who does what
    pr = rng.fork("poses")
    poses = [POSE_LEAD[p.poses][i] if i < len(POSE_LEAD[p.poses]) and i < max(2, n - 1) else pr.weighted(POSE_MIX[p.poses]) for i in range(n)]
    cr = rng.fork("colour")
    nwhite = min(int(p.white), n)
    plums = []
    for i in range(n):
        if p.plumage == "mixed":
            plums.append("blue_bar" if i == 0 else cr.weighted([("blue_bar", 0.4), ("checker", 0.28), ("dark", 0.17), ("pied", 0.15)]))
        else:
            plums.append(p.plumage)
    for k in [w for w in [1, 3, 0, 2, 4, 5, 6] if w < n][:nwhite]:
        plums[k] = "white"

    # ---- where: bird 0 at the feeding spot in the middle, the rest scattered with breathing room
    lr = rng.fork("layout")
    feed = (lr.jitter(0, 0.04) + 0.13, lr.jitter(0, 0.04) - 0.06)
    pts = [(0.0, 0.0)]
    dmin = 0.34
    tries = 0
    while len(pts) < n:
        a, rad = lr.u(0, 2 * math.pi), R * math.sqrt(lr.u(0.05, 1.0))
        q = (rad * math.cos(a), rad * math.sin(a))
        # keep the view from the front onto the lead bird open (nobody stands right in front of it)
        if q[1] < -0.12 and abs(q[0]) < 0.12 - q[1] * 0.9:
            tries += 1
            continue
        if all(math.hypot(q[0] - o[0], q[1] - o[1]) > dmin for o in pts) and math.hypot(q[0] - feed[0], q[1] - feed[1]) > 0.1:
            pts.append(q)
        tries += 1
        if tries % 60 == 0:
            dmin *= 0.9
    threat = lr.u(0, 360)
    S = Soup()
    beaks = []
    for i in range(n):
        r = rng.fork(f"bird{i}")
        pose = poses[i]
        to_feed = math.degrees(math.atan2(feed[1] - pts[i][1], feed[0] - pts[i][0]))
        if i == 0:
            yaw = r.pick([0.0, 180.0]) + r.jitter(0, 20.0)      # side-on to the front: the classic pigeon profile
        elif pose in ("peck", "walk"):
            yaw = to_feed + r.jitter(0, 30.0 if pose == "peck" else 55.0)
        elif p.poses == "alert":
            yaw = threat + r.jitter(0, 50.0)
        else:
            yaw = r.u(0, 360)
        hy = {"stand": r.jitter(0, 55.0), "alert": r.jitter(0, 70.0), "walk": r.jitter(0, 20.0), "peck": r.jitter(0, 14.0), "roost": r.jitter(0, 35.0)}[pose]
        S.uvo = (r.u(0, 3.0), r.u(0, 3.0))
        bt = make_bird(S, pose, plums[i], r, p.size * r.vary(1.0, 0.06), p.fluff, pts[i], yaw, hy, D)
        if pose == "peck":
            beaks.append(bt)

    # ---- a shed feather or two (quill + cupped vane)
    dr = rng.fork("mess")
    for k in range(1 + (1 if n >= 5 else 0)):
        a, rad, fa = dr.u(0, 2 * math.pi), R * dr.u(0.35, 1.0), dr.u(0, 2 * math.pi)
        c = np.array([rad * math.cos(a), rad * math.sin(a), 0.0])
        ax, sd_ = np.array([math.cos(fa), math.sin(fa), 0.0]), np.array([-math.sin(fa), math.cos(fa), 0.0])
        rows = []
        for (t, wd, zz) in [(0.0, 0.0012, 0.001), (0.016, 0.007, 0.004), (0.04, 0.0095, 0.007), (0.06, 0.007, 0.0055)]:
            cc = c + ax * t + sd_ * 0.004 * math.sin(t * 40.0)
            rows.append([cc + sd_ * wd + (0, 0, zz), cc + (0, 0, zz + 0.0012 + wd * 0.12), cc - sd_ * wd + (0, 0, zz * 0.6)])
        S.ribbon(rows, FEA if k else WHT, np.array([(0.5 if k else 0.0, 0.0, 0.0), (0.2 if k else 0.0, 0.0, 0.0), (0.1, 0.0, 0.2), (0.1, 0.0, 0.3)]),
                 tip=c + ax * 0.072 + (0, 0, 0.003), inside=c - (0, 0, 0.05))
    flock = S.finish("pigeons", mats)
    smooth(flock, 75)

    # ---- bread: a couple of torn crust pieces at the feeding spot and crumbs under the pecking beaks
    crr = rng.fork("crumbs")
    left = 5950 - S.tris - int(p.droppings) * 18
    ncr = min(int(round(36 * p.crumbs)), max(left // 8, 0))
    if ncr > 0:
        jr = crr.fork("shape")
        ov = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 0.75), (0, 0, -0.75)]
        ov = [(x * jr.u(0.7, 1.15) + jr.jitter(0, 0.18), y * jr.u(0.7, 1.15) + jr.jitter(0, 0.18), z + jr.jitter(0, 0.12)) for x, y, z in ov]
        chunk = mesh_object("crumb", ov, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (1, 0, 5), (2, 1, 5), (3, 2, 5), (0, 3, 5)])
        assign(chunk, mats[WHT])
        smooth(chunk, 180)
        # pale crumb with a darker baked crust on one flank
        paint(chunk, special=lambda co, nr: np.clip(co[:, 0] * 0.9 + co[:, 1] * 0.5 + 0.25, 0.0, 1.0),
              grime=lambda co, nr: 0.3 * np.clip(co[:, 0] + 0.2, 0.0, 1.0))
        ms = []
        for k in range(ncr):
            if beaks and k % 3 == 0:
                b = crr.pick(beaks)
                c = (b[0] + crr.jitter(0, 0.035), b[1] + crr.jitter(0, 0.035))
            else:
                a, rad = crr.u(0, 2 * math.pi), abs(crr.jitter(0, 1.0)) ** 1.6 * (0.1 + 0.16 * R)
                c = (feed[0] + rad * math.cos(a), feed[1] + rad * math.sin(a))
            s = crr.u(0.0016, 0.004) * (2.6 if k < 3 else 1.0)          # 3-8 mm crumbs, a few 1-2 cm torn pieces
            ms.append(trs((c[0], c[1], s * 0.62), (crr.jitter(0, 35), crr.jitter(0, 35), crr.u(0, 360)), (s * crr.u(0.8, 1.5), s, s * crr.u(0.7, 1.0))))
        cm_ = instances_to_mesh("crumbs", chunk, ms, keep_base=False)
        transform_verts(cm_, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))

    # ---- droppings
    for k in range(int(p.droppings)):
        a, rad = dr.u(0, 2 * math.pi), R * math.sqrt(dr.u(0.0, 1.0)) * 1.05
        rd = dr.u(0.011, 0.024)
        sp = lathe(f"dropping{k}", [(0.0, 0.0028), (rd * 0.5, 0.0024), (rd, 0.0)], segments=6,
                   loc=(rad * math.cos(a), rad * math.sin(a), 0.0), rot=(0, 0, dr.u(0, 360)))
        sd = dr.u(0, 100)
        transform_verts(sp, lambda co, sd=sd: Vector((co.x * (1.0 + 0.3 * math.sin(3.0 * math.atan2(co.y, co.x) + sd)),
                                                      co.y * (1.0 + 0.25 * math.cos(2.0 * math.atan2(co.y, co.x) + sd * 1.7)), co.z)))
        assign(sp, mats[WHT])
        smooth(sp, 60)
        paint(sp, grime=lambda co, nr: 0.55 * np.clip((co[:, 2] - 0.0018) / 0.001, 0.0, 1.0))
