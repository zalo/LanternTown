"""Ivy (Hedera) climbing a wall from a planting pocket at its foot. Woody stems grow over the wall plane
(y = 0) by a seeded space-colonisation walk inside a ragged fan-shaped envelope, thicken towards the
ground by the pipe model and are skinned as flattened bark tubes (a braided companion stem winds round
the main trunk). Every thin shoot carries shingled, individually bent lobed leaf blades on petioles,
standing 2-12 cm off the wall in several tint classes that turn yellow / orange / crimson with `autumn`.
Leader shoots with small young leaves run out past the mass; soil, moss and fallen leaves at the foot.
"""
from tk import *
import numpy as np

META = {"name": "Climbing ivy patch", "class": "vegetation", "mount": "wall_ground", "focus": 0.45}
PARAMS = {
    "width": {"default": 1.9, "min": 0.6, "max": 3.2, "desc": "overall width of the patch on the wall (m)"},
    "height": {"default": 2.7, "min": 0.8, "max": 4.5, "desc": "height the ivy has climbed (m)"},
    "density": {"default": 1.0, "min": 0.3, "max": 1.5, "desc": "leaf density / overlap (1 = closed cover)"},
    "leaf_size": {"default": 0.068, "min": 0.045, "max": 0.1, "desc": "typical leaf blade length (m)"},
    "autumn": {"default": 0.0, "min": 0.0, "max": 1.0, "desc": "autumn tint amount (0 = evergreen, 1 = all leaves turned)"},
    "coverage": {"default": 0.9, "min": 0.35, "max": 1.0, "desc": "how solid the mass is (low = gappy, wall shows through)"},
    "bare": {"default": 0.12, "min": 0.0, "max": 0.6, "desc": "fraction of the height with bare woody stems at the base"},
    "lean": {"default": 0.0, "min": -1.0, "max": 1.0, "desc": "sideways growth bias (-1 left .. +1 right)"},
    "stems": {"default": 3, "min": 1, "max": 4, "desc": "number of stems rising from the ground"},
    "trunk": {"default": 0.026, "min": 0.008, "max": 0.045, "desc": "radius of the main stem at the ground (m)"},
    "depth": {"default": 0.11, "min": 0.05, "max": 0.14, "desc": "maximum projection of the leaves from the wall (m)"},
    "trails": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount / length of leader shoots running out of the mass"},
    "berries": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "amount of flower / berry umbels on the adult upper shoots (pale green buds, blue-black berries in autumn)"},
    "litter": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "fallen leaves on the ground at the foot"},
    "opening_w": {"default": 0.0, "min": 0.0, "max": 2.4, "desc": "width of a trimmed opening for a door / window (0 = none)"},
    "opening_h": {"default": 2.1, "min": 0.4, "max": 3.5, "desc": "height of the trimmed opening (m)"},
    "opening_z": {"default": 0.0, "min": 0.0, "max": 3.0, "desc": "sill height of the trimmed opening (0 = doorway)"},
    "green": {"default": "#3f7433", "desc": "base leaf colour"},
}
VARIANTS = {
    "autumn_tall": {"width": 1.8, "height": 3.9, "autumn": 0.85, "litter": 1.0, "bare": 0.1, "trails": 0.8, "seed": 4},
    "young": {"width": 1.4, "height": 1.9, "density": 0.7, "coverage": 0.62, "leaf_size": 0.055, "stems": 1, "trunk": 0.011,
                     "bare": 0.0, "trails": 1.0, "litter": 0.1, "seed": 11},
    "doorway": {"width": 3.0, "height": 3.3, "opening_w": 1.15, "opening_h": 2.25, "stems": 3, "density": 1.15, "coverage": 0.95,
                "autumn": 0.3, "seed": 23},
    "old_leaning": {"width": 3.2, "height": 2.3, "lean": 0.85, "bare": 0.42, "trunk": 0.042, "stems": 2, "leaf_size": 0.085,
                    "density": 1.3, "coverage": 1.0, "green": "#2f5f3a", "trails": 0.3, "seed": 37},
}

# leaf outlines: x across (-0.5..0.5), y along the blade (0 = base, 1 = tip); vertex 0 = petiole foot
_HI = {"pts": np.array([(0, -0.42), (-0.016, 0.17), (0.016, 0.17),                    # petiole needle
                        (0, 0.16), (0, 0.52), (0, 1.0),                               # n, m, t on the midrib
                        (0.25, 0.02), (0.34, 0.22), (0.48, 0.42), (0.2, 0.6),          # right a, b, c, d
                        (-0.25, 0.02), (-0.34, 0.22), (-0.48, 0.42), (-0.2, 0.6)]),
       "tris": np.array([(1, 0, 2), (3, 6, 7), (3, 7, 4), (4, 7, 8), (4, 8, 9), (4, 9, 5),
                         (3, 11, 10), (3, 4, 11), (4, 12, 11), (4, 13, 12), (4, 5, 13)])}
_LO = {"pts": np.array([(0, -0.42), (-0.016, 0.14), (0.016, 0.14),
                        (0, 0.12), (0, 1.0), (0.46, 0.3), (0.17, 0.55), (-0.46, 0.3), (-0.17, 0.55)]),
       "tris": np.array([(1, 0, 2), (3, 5, 6), (3, 6, 4), (3, 8, 7), (3, 4, 8)])}


def _sm(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vnoise(seed):
    g = np.random.RandomState(seed % (2 ** 31 - 1)).rand(64, 64)

    def f(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64) + 0.0 * x
        x = x + 0.0 * y
        xi = np.floor(x).astype(int)
        yi = np.floor(y).astype(int)
        fx, fy = x - xi, y - yi
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        a, b = g[xi % 64, yi % 64], g[(xi + 1) % 64, yi % 64]
        c, d = g[xi % 64, (yi + 1) % 64], g[(xi + 1) % 64, (yi + 1) % 64]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    return f


def _fbm(nf, x, y, octaves=3):
    s, a, tot = 0.0, 0.5, 0.0
    for o in range(octaves):
        s = s + a * nf(np.asarray(x) * 2 ** o + 17.3 * o, np.asarray(y) * 2 ** o + 5.1 * o)
        tot += a
        a *= 0.5
    return s / tot


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _np_mesh(name, verts, faces, vuv):
    """Mesh from numpy arrays with per-vertex UVs (metres) kept as the object's UV map."""
    o = mesh_object(name, np.asarray(verts, dtype=np.float64).reshape(-1, 3).tolist(), np.asarray(faces).tolist())
    me = o.data
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", np.asarray(vuv, dtype=np.float32)[lv].ravel())
    uv_keep(o)
    return o


class _Tubes:
    """Accumulates tapered, slightly flattened tubes (seam on the wall side, V along the stem for bark)."""

    def __init__(self):
        self.V, self.F, self.UV, self.n, self.tris = [], [], [], 0, 0

    def add(self, P, R, sides, rs, flat=0.86):
        P = np.asarray(P, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        n = len(P)
        if n < 2:
            return
        T = np.gradient(P, axis=0)
        if P[0, 2] < 0.02:
            T[0] = (0, 0, 1)
        T = _unit(T)
        B = np.cross(T, np.array([0.0, -1.0, 0.0]))
        bl = np.linalg.norm(B, axis=1, keepdims=True)
        B = np.where(bl < 1e-5, np.array([1.0, 0.0, 0.0]), B / np.maximum(bl, 1e-9))
        N = np.cross(B, T)
        th = -pi / 2 + 2 * pi * np.arange(sides + 1) / sides
        if sides >= 5:                                       # gnarled, knobbly old wood
            sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            kn = np.interp(sl, np.arange(0, sl[-1] + 0.08, 0.035), rs.rand(int(sl[-1] / 0.035) + 4)[:len(np.arange(0, sl[-1] + 0.08, 0.035))])
            R = R * (0.88 + 0.3 * kn ** 2)
            lump = 1 + rs.normal(0, 0.07, (n, sides + 1))
            lump[:, -1] = lump[:, 0]
        else:
            lump = np.ones((n, sides + 1))
        ring = P[:, None, :] + (R[:, None] * lump)[:, :, None] * (np.cos(th)[None, :, None] * B[:, None, :] + flat * np.sin(th)[None, :, None] * N[:, None, :])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        circ = 2 * pi * max(float(R.mean()), 0.004)
        uv = np.stack([np.broadcast_to((np.arange(sides + 1) / sides * circ + rs.rand() * 2)[None, :], (n, sides + 1)),
                       np.broadcast_to((s + rs.rand() * 3)[:, None], (n, sides + 1))], axis=-1)
        m = sides + 1
        i = np.arange(n - 1)[:, None]
        k = np.arange(sides)[None, :]
        q = np.stack([i * m + k, (i + 1) * m + k, (i + 1) * m + k + 1, i * m + k + 1], axis=-1).reshape(-1, 4) + self.n
        self.V.append(ring.reshape(-1, 3))
        self.UV.append(uv.reshape(-1, 2))
        self.F.append(q)
        self.n += n * m
        self.tris += (n - 1) * sides * 2

    def build(self, name):
        if not self.V:
            return None
        V = np.concatenate(self.V)
        V[:, 1] = np.minimum(V[:, 1], -0.0005)
        V[:, 2] = np.maximum(V[:, 2], 0.0)
        o = _np_mesh(name, V, np.concatenate(self.F), np.concatenate(self.UV))
        smooth(o, 80)
        return o


def _chaikin(P, R):
    P2, R2 = [P[0]], [R[0]]
    for i in range(len(P) - 1):
        P2 += [0.75 * P[i] + 0.25 * P[i + 1], 0.25 * P[i] + 0.75 * P[i + 1]]
        R2 += [0.75 * R[i] + 0.25 * R[i + 1], 0.25 * R[i] + 0.75 * R[i + 1]]
    P2.append(P[-1])
    R2.append(R[-1])
    return np.array(P2), np.array(R2)


def _grow(A, roots, D, di, dk, rs, blocked, max_nodes):
    """2D space colonisation: attractors A (M,2) pull a branching tree out of the root nodes."""
    pos = [np.array(r, dtype=np.float64) for r in roots]
    par = [-1] * len(roots)
    M = len(A)
    nd = np.full(M, 1e9)
    ni = np.zeros(M, dtype=int)
    alive = np.ones(M, dtype=bool)
    new_from = 0
    for it in range(2500):
        if len(pos) >= max_nodes:
            break
        Pn = np.array(pos[new_from:])
        if len(Pn):
            d = np.linalg.norm(A[:, None, :] - Pn[None, :, :], axis=2)
            j = d.argmin(1)
            dm = d[np.arange(M), j]
            upd = dm < nd
            nd[upd] = dm[upd]
            ni[upd] = j[upd] + new_from
        new_from = len(pos)
        alive &= nd > dk
        if not alive.any():
            break
        act = alive & (nd < di)
        P = np.array(pos)
        if not act.any():
            ai = np.where(alive)[0]
            k = ai[nd[ai].argmin()]
            src = ni[k]
            v = _unit(A[k] - P[src] + rs.normal(0, 0.15, 2) * np.linalg.norm(A[k] - P[src]))
            q = P[src] + D * v
            if blocked(q[None, :])[0] or np.linalg.norm(P - q, axis=1).min() < 0.4 * D:
                alive[k] = False
                continue
            pos.append(q)
            par.append(int(src))
            continue
        vec = _unit(A[act] - P[ni[act]])
        acc = np.zeros((len(P), 2))
        np.add.at(acc, ni[act], vec)
        src = np.unique(ni[act])
        dirs = _unit(_unit(acc[src]) + rs.normal(0, 0.27, (len(src), 2)) + np.array([0.0, 0.1]))
        newp = P[src] + D * dirs
        ok = ~blocked(newp)
        ok &= np.linalg.norm(newp[:, None, :] - P[None, :, :], axis=2).min(1) > 0.5 * D
        added = 0
        for q, s, good in zip(newp, src, ok):
            if good:
                pos.append(q)
                par.append(int(s))
                added += 1
        if added == 0:
            alive[act] = False
    return np.array(pos), np.array(par, dtype=int)


def _leaves(shape, base, e, tilt, roll, s, wid, fold, curl, droop, lobe=None):
    """Vectorised leaf blades. base (K,3) petiole end, e (K,3) in-plane tip direction, tilt = angle off the
    wall plane, s = blade length. Returns verts (K*nv,3), tris, per-vertex uv (metres on the 0.25 m tile)."""
    K = len(base)
    out = np.array([0.0, -1.0, 0.0])
    ca, sa = np.cos(tilt)[:, None], np.sin(tilt)[:, None]
    Y = ca * e + sa * out
    Z = _unit(ca * out - sa * e)
    X = np.cross(Y, Z)
    cr, sr = np.cos(roll)[:, None], np.sin(roll)[:, None]
    X, Z = cr * X + sr * Z, -sr * X + cr * Z
    return _leaf_frames(shape, base, X, Y, Z, s, wid, fold, curl, droop, lobe)


def _leaf_frames(shape, base, X, Y, Z, s, wid, fold, curl, droop, lobe=None):
    K = len(base)
    pts, tris = shape["pts"], shape["tris"]
    nv = len(pts)
    x = np.repeat(pts[None, :, 0], K, axis=0)
    y = np.repeat(pts[None, :, 1], K, axis=0)
    if lobe is not None and nv == 14:                # 0 = deeply 5-lobed juvenile leaf .. 1 = nearly entire, heart-shaped
        for i, sg in ((7, 1.0), (11, -1.0)):
            x[:, i] += sg * 0.08 * lobe
        for i, sg in ((9, 1.0), (13, -1.0)):
            x[:, i] += sg * 0.13 * lobe
            y[:, i] += 0.03 * lobe
        for i, sg in ((8, 1.0), (12, -1.0)):
            x[:, i] -= sg * 0.05 * lobe
        for i, sg in ((6, 1.0), (10, -1.0)):
            x[:, i] -= sg * 0.03 * lobe
            y[:, i] += 0.04 * lobe
    z = fold[:, None] * np.abs(x) - curl[:, None] * (y - 0.4) ** 2 - droop[:, None] * x * x
    z[:, 0] = -0.3                                   # petiole foot drops back towards the stem
    y0 = pts[3, 1]
    lx, ly, lz = x * (wid * s)[:, None], (y - y0) * s[:, None], z * s[:, None]
    V = base[:, None, :] + lx[..., None] * X[:, None, :] + ly[..., None] * Y[:, None, :] + lz[..., None] * Z[:, None, :]
    F = tris[None, :, :] + (np.arange(K) * nv)[:, None, None]
    uv = np.stack([(0.5 + 0.92 * pts[:, 0]) * 0.25, np.clip(0.04 + 0.92 * pts[:, 1], 0.03, 0.97) * 0.25], axis=-1)
    uv[0] = (0.125, 0.03)
    uv[1], uv[2] = (0.118, 0.2), (0.132, 0.2)
    return V.reshape(-1, 3), F.reshape(-1, 3), np.tile(uv, (K, 1))


def _mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def build(p, rng):
    W, H = float(p.width), float(p.height)
    seed = int(p.seed)
    rs = np.random.RandomState((seed * 7919 + 13) % (2 ** 31 - 1))
    nz, nh, nc = _vnoise(seed * 3 + 1), _vnoise(seed * 3 + 2), _vnoise(seed * 3 + 3)
    k_sz = clamp(sqrt(W * H / 5.0), 0.8, 1.7)            # coarser growth for big patches
    D, dk, di = 0.045 * k_sz, 0.1 * k_sz, 0.5 * k_sz
    lean = float(p.lean)
    bare_t = float(p.bare)
    n_stems = max(1, min(int(p.stems), int(W / 0.3)))         # a narrow strip cannot hold four trunks
    trunk = min(float(p.trunk), 0.008 + 0.013 * sqrt(W * H))  # ... nor can a small young patch have arm-thick wood
    ow, oh, oz = float(p.opening_w), float(p.opening_h), float(p.opening_z)
    has_open = ow > 0.05 and ow < W - 0.5 and oz < H - 0.4

    def blocked(q, margin=0.03):
        q = np.asarray(q)
        b = (np.abs(q[:, 0]) > W / 2 - 0.03) | (q[:, 1] > H - 0.03) | (q[:, 1] < 0.0)
        if has_open:
            b |= (np.abs(q[:, 0]) < ow / 2 + margin) & (q[:, 1] > oz - margin) & (q[:, 1] < oz + oh + margin)
        return b

    # ---- envelope: ragged fan from a narrow foot, tongues at the top, holes towards the margins
    xb = -lean * W * 0.3
    base_frac = min(0.26 + 0.07 * n_stems, 0.6)

    def envelope(x, z):
        t = np.clip(z / H, 0, 1)
        xc = xb + lean * W * 0.45 * t
        fan = base_frac + (1 - base_frac) * _sm(bare_t * 0.75, bare_t + 0.22, t)
        if has_open:
            fan = np.maximum(fan, 0.999 * (z < oz + oh + 0.5))
        rag_r = 1 - 0.34 * _fbm(nz, z / 1.1 + 3.1, 0.5, 1) - 0.2 * _fbm(nz, z / 0.3 + 13.1, 4.5, 2)
        rag_l = 1 - 0.34 * _fbm(nz, z / 1.1 + 40.7, 9.5, 1) - 0.2 * _fbm(nz, z / 0.3 + 53.1, 7.5, 2)
        hr = np.maximum(fan * (W / 2 - xc) * rag_r, 0.04)
        hl = np.maximum(fan * (W / 2 + xc) * rag_l, 0.04)
        u = np.where(x >= xc, (x - xc) / hr, (xc - x) / hl)
        ztop = H * (1 - 0.5 * _fbm(nz, x / (0.2 * W + 0.18) + 70.0, 3.3)) * (1 - 0.12 * np.clip(u, 0, 1) ** 2)
        e = np.clip(np.minimum((1 - u) * np.minimum(hr, hl) / 0.35, (ztop - z) / 0.4), 0, 1)
        thr = 0.5 + 0.7 * (1 - p.coverage)
        acc = _sm(thr - 0.08, thr + 0.08, _fbm(nh, x / 0.38, z / 0.38) + 0.45 * e)
        acc = np.where(t < bare_t, np.maximum(acc, 0.6), acc)
        acc = np.where((u < 1) & (z < ztop) & (z > 0.02), acc, 0.0)
        # a skirt of runners creeping along the foot of the wall
        sk_w = min(0.3 + 0.12 * n_stems, 0.8) * W / 2 * (0.75 + 0.5 * _fbm(nz, x / 0.6 + 17.0, 8.8, 1))
        sk = (np.abs(x - xb) < sk_w) & (z > 0.02) & (z < 0.06 + 0.3 * _fbm(nz, x / 0.22 + 90.0, 1.7, 2) * (1 - 0.6 * np.abs(x - xb) / np.maximum(sk_w, 1e-3)))
        return np.maximum(acc, np.where(sk, 0.8 * (1 - bare_t), 0.0))

    sp = 0.05 * k_sz
    nc_pts = int(W * H / (sp * sp))
    cx = rs.uniform(-W / 2, W / 2, nc_pts)
    cz = rs.uniform(0, H, nc_pts)
    keep = rs.rand(nc_pts) < envelope(cx, cz)
    A = np.stack([cx[keep], cz[keep]], axis=1)
    if len(A) < 12:
        A = np.stack([rs.uniform(-W * 0.3, W * 0.3, 60), rs.uniform(0.05, H * 0.8, 60)], axis=1)
    # fit the cloud to the requested extents (leaves overhang the stems by a few cm)
    mg = 0.07 + 0.5 * p.leaf_size
    x0, x1 = np.percentile(A[:, 0], [1.5, 98.5])              # stray outliers must not shrink the mass
    mgx, mgz = mg + 0.05 * p.trails * W, mg + 0.06 * p.trails * H        # room for the leaders
    A[:, 0] = (A[:, 0] - x0) / max(x1 - x0, 1e-3) * (W - 2 * mgx) - (W / 2 - mgx)
    A[:, 1] *= (H - mgz) / max(np.percentile(A[:, 1], 99.6), 1e-3)
    A = A[(np.abs(A[:, 0]) < W / 2 - mg) & (A[:, 1] < H - mg)]
    A = A[~blocked(A, 0.07)]
    low = A[A[:, 1] < max(0.3, 0.12 * H)]
    if len(low) < n_stems:
        low = A[np.argsort(A[:, 1])[:max(n_stems, 8)]]
    lx = np.sort(low[:, 0])
    roots = [(float(lx[int((i + 0.5) / n_stems * (len(lx) - 1))]) + rs.uniform(-0.02, 0.02), 0.0) for i in range(n_stems)]
    for i in range(1, len(roots)):                       # keep the feet apart
        if roots[i][0] - roots[i - 1][0] < 0.06:
            roots[i] = (roots[i - 1][0] + 0.06 + rs.uniform(0, 0.04), 0.0)

    pos, par = _grow(A, roots, D, di, dk, rs, blocked, 1700)
    N0 = len(pos)

    # ---- relax the kinks of the growth walk, then send leader shoots out of some tips
    for _ in range(2):
        nchild = np.bincount(par[par >= 0], minlength=len(pos))
        csum = np.zeros_like(pos)
        np.add.at(csum, par[par >= 0], pos[par >= 0])
        m = (par >= 0) & (nchild > 0)
        avg = csum[m] / nchild[m][:, None]
        pos[m] = 0.5 * pos[m] + 0.25 * pos[par[m]] + 0.25 * avg
    # old stems snake up the wall instead of standing like posts
    root_id = np.arange(len(pos))
    for i in range(len(pos)):
        if par[i] >= 0:
            root_id[i] = root_id[par[i]]
    s_ph, s_wl, s_am = rs.uniform(0, 2 * pi, len(pos)), rs.uniform(0.3, 0.75, len(pos)), rs.uniform(0.012, 0.04, len(pos))
    zb = bare_t * H + 0.25
    pos[:, 0] += (s_am[root_id] * _sm(0.0, 0.14, pos[:, 1]) * (1 - _sm(zb, zb + 0.5, pos[:, 1]))
                  * np.sin(2 * pi * pos[:, 1] / s_wl[root_id] + s_ph[root_id]))
    # runners snake rather than grow straight: low-frequency warp (feet stay put)
    wamp = 0.045 * k_sz * _sm(0.0, 0.35, pos[:, 1])
    pos[:, 0] += wamp * 2 * (_fbm(nc, pos[:, 0] / 0.32 + 11.0, pos[:, 1] / 0.32, 2) - 0.5)
    pos[:, 1] += wamp * 1.2 * (_fbm(nc, pos[:, 0] / 0.32 + 51.0, pos[:, 1] / 0.32 + 23.0, 2) - 0.5)
    pos[:, 0] = np.clip(pos[:, 0], -W / 2 + 0.03, W / 2 - 0.03)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, H - 0.03)
    pos_l, par_l, leader = list(pos), list(par), [False] * N0
    nchild = np.bincount(par[par >= 0], minlength=N0)
    tips = np.array([i for i in range(N0) if nchild[i] == 0 and par[i] >= 0], dtype=int)
    if len(tips):
        # the outermost tips (fewest neighbours) throw the leaders, heading away from the mass
        crowd = (np.linalg.norm(pos[tips][:, None, :] - pos[None, :, :], axis=2) < 0.24 * k_sz).sum(1) + rs.rand(len(tips)) * 6
        tips = tips[np.argsort(crowd)]
    cen = pos.mean(axis=0)
    n_lead = int(round(p.trails * (4 + 1.3 * len(tips) ** 0.75)))
    stack = [(int(i), None, int(rs.randint(4, 7 + int(20 * p.trails)))) for i in tips[:n_lead]]
    while stack:
        i, ang, steps = stack.pop()
        if ang is None:
            h = _unit(pos_l[i] - pos_l[par_l[i]]) + 0.8 * _unit(pos_l[i] - cen) + np.array([0.0, 0.2])
            ang = atan2(h[1], h[0])
        cur, q = i, np.array(pos_l[i], dtype=np.float64)
        turn = rs.normal(0, 0.08)
        for s_ in range(steps):
            ang += turn + rs.normal(0, 0.2) + 0.04 * np.sign(sin(pi / 2 - ang))    # wander, drift upwards
            q = q + D * np.array([cos(ang), sin(ang)])
            if blocked(q[None, :], 0.02)[0] or q[1] < 0.06:
                break
            pos_l.append(q.copy())
            par_l.append(cur)
            leader.append(True)
            cur = len(pos_l) - 1
            if steps > 6 and 1 < s_ < steps - 3 and rs.rand() < 0.1:                 # side shoot
                stack.append((cur, ang + rs.choice([-1.0, 1.0]) * rs.uniform(0.6, 1.1), int(rs.randint(3, max(4, steps // 2)))))
    pos, par, leader = np.array(pos_l), np.array(par_l, dtype=int), np.array(leader)
    N = len(pos)

    # ---- pipe model radii
    children = [[] for _ in range(N)]
    for i in range(N):
        if par[i] >= 0:
            children[par[i]].append(i)
    ntip = np.zeros(N)
    for i in range(N - 1, -1, -1):
        if not children[i]:
            ntip[i] = 1.0
        if par[i] >= 0:
            ntip[par[i]] += ntip[i]
    load = np.ones(N)                                        # wood carried = length of shoot above the node
    for i in range(N - 1, -1, -1):
        if par[i] >= 0:
            load[par[i]] += load[i]
    r_tip = 0.0017
    rad = r_tip + (trunk - r_tip) * (load / load.max()) ** 0.6
    for i in range(N):
        if par[i] < 0:
            rad[i] = max(rad[i], 0.45 * trunk)
    lift = 0.004 * _fbm(nc, pos[:, 0] / 0.11, pos[:, 1] / 0.11) * (rad < 0.006)
    P3 = np.stack([pos[:, 0], -(rad * 0.86 + 0.0008 + lift), pos[:, 1]], axis=1)
    DBG_nodes = N

    # ---- chains -> tubes
    tubes = _Tubes()
    main = [max(c, key=lambda j: ntip[j]) if c else -1 for c in children]
    starts = [i for i in range(N) if par[i] < 0]
    for i in range(N):
        for c in children[i]:
            if c != main[i]:
                starts.append(c)
    trunk_chain = None

    def sides_for(r):
        return 7 if r >= 0.012 else 5 if r >= 0.006 else 4 if r >= 0.0032 else 3

    for s0 in starts:
        idx = [s0]
        while main[idx[-1]] >= 0:
            idx.append(main[idx[-1]])
        Pc = P3[idx].copy()
        Rc = rad[idx].copy()
        is_root = par[s0] < 0
        if not is_root:
            Pc = np.vstack([P3[par[s0]][None, :], Pc])
            Rc = np.concatenate([[rad[s0]], Rc])
        else:
            Rc = Rc * (1 + 0.9 * np.exp(-Pc[:, 2] / 0.07))                   # root flare
            Pc[:, 1] = -(Rc * 0.86 + 0.0008)
            if trunk_chain is None or ntip[s0] > trunk_chain[2]:
                trunk_chain = (Pc.copy(), Rc.copy(), ntip[s0])
        Rc[-1] *= 0.5
        if len(Pc) < 2:
            continue
        # split where the radius class changes so thin ends do not pay for 8 sides
        cls = [sides_for(r) for r in Rc]
        a = 0
        while a < len(Pc) - 1:
            b = a + 1
            while b < len(Pc) - 1 and cls[b] == cls[a]:
                b += 1
            Pa, Ra, sd = Pc[a:b + 1], Rc[a:b + 1], cls[a]
            if sd >= 5:
                Pa, Ra = _chaikin(Pa, Ra)
            elif sd == 3 and len(Pa) > 3:
                sel = sorted(set(list(range(0, len(Pa), 2)) + [len(Pa) - 1]))
                Pa, Ra = Pa[sel], Ra[sel]
            tubes.add(Pa, Ra, sd, rs)
            a = b

    # ---- a companion stem braided round the main trunk
    if trunk_chain is not None and len(trunk_chain[0]) > 6:
        Pt, Rt, _ = trunk_chain
        Pt, Rt = _chaikin(Pt, Rt)
        st = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pt, axis=0), axis=1))])
        for b in range(1 if trunk < 0.02 else 2):
            Lb = min(st[-1] * rs.uniform(0.45, 0.8), rs.uniform(0.7, 1.6) + H * 0.15)
            nb = int(np.searchsorted(st, Lb))
            if nb < 5:
                continue
            sb = st[:nb]
            tt = sb / sb[-1]
            rb = Rt[:nb] * rs.uniform(0.36, 0.5) * (1 - 0.75 * tt ** 1.5)
            rb = rb * (1 + 0.5 * np.exp(-Pt[:nb, 2] / 0.06))
            phi = (pi if b else 0.0) + sb * (2 * pi / rs.uniform(0.45, 0.8)) * (1 if b else -1)
            Tt = _unit(np.gradient(Pt[:nb], axis=0))
            Bt = _unit(np.cross(Tt, np.array([0.0, -1.0, 0.0])))
            Rh = Rt[:nb] * 0.9 + rb * 0.8 + 0.03 * tt ** 3
            Pb = Pt[:nb] + Bt * (Rh * np.cos(phi))[:, None]
            Pb[:, 1] = -(rb * 0.86 + (1.7 * Rt[:nb] - 0.1 * rb) * np.sin(phi) ** 2)
            tubes.add(Pb, rb, 6, rs)

    # ---- berry umbels standing out of the mass on the adult upper shoots
    n_berry_tris = 0
    cand = np.where((rad < 0.005) & (par >= 0) & ~leader & (pos[:, 1] > 0.3 * H))[0]
    n_umb = min(int(round(p.berries * (5 + 18.0 * len(A) * sp * sp))), len(cand), 44)
    if n_umb > 0:
        bc, br = [], []
        for j in rs.choice(cand, n_umb, replace=False):
            c = np.array([pos[j, 0] + rs.uniform(-0.03, 0.03), -rs.uniform(0.95, 1.2) * p.depth, pos[j, 1] + rs.uniform(0.0, 0.04)])
            p0 = P3[j]
            stalk = np.array([p0, p0 + (c - p0) * 0.35 + np.array([0, 0.004, 0.012]), p0 + (c - p0) * 0.75 + np.array([0, 0.0, 0.01]), c])
            tubes.add(stalk, np.array([0.0019, 0.0017, 0.0015, 0.0013]), 3, rs)
            nb = int(rs.randint(12, 19))
            dirs = _unit(rs.normal(0, 1, (nb, 3)))
            bc.append(c[None, :] + dirs * rs.uniform(0.015, 0.024, (nb, 1)))
            br.append(rs.uniform(0.005, 0.0068, nb))
        bc, br = np.concatenate(bc), np.concatenate(br)
        bc[:, 1] = np.minimum(bc[:, 1], -0.008)
        bc[:, 2] = np.maximum(bc[:, 2], 0.008)
        axes = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.float64)
        bv = bc[:, None, :] + br[:, None, None] * axes[None, :, :]
        of = []
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    a_, b_, c_ = (0 if sx > 0 else 1), (2 if sy > 0 else 3), (4 if sz > 0 else 5)
                    of.append((a_, b_, c_) if sx * sy * sz > 0 else (a_, c_, b_))
        bfc = np.array(of)[None, :, :] + (np.arange(len(bc)) * 6)[:, None, None]
        bo = _np_mesh("berries", bv.reshape(-1, 3), bfc.reshape(-1, 3), np.full((len(bc) * 6, 2), 0.06))
        smooth(bo, 180)
        assign(bo, mat("wax", _mix(hex_rgb("#b9c468"), hex_rgb("#2b2540"), float(_sm(0.25, 0.8, p.autumn)))))
        n_berry_tris = len(bc) * 8

    # ---- leaves: a flat, dark inner layer shingled against the wall + the main mass standing off on petioles
    a_env = len(A) * sp * sp
    # old wood is bare near the ground (the `bare` zone) but hidden under foliage higher up
    leafy = np.maximum(np.maximum(1 - _sm(0.0045, 0.012, rad), 1 - _sm(1.5, 4.5, ntip)), 0.8 * _sm(0.0, 0.3, pos[:, 1] - (0.12 + bare_t * H))) * (par >= 0)
    w_main = leafy * np.where(leader, 0.5, 1.0)
    w_in = leafy * np.where(leader, 0.0, 1.0)
    n_old = max(int(((par >= 0) & ~leader).sum()), 1)
    cov = clamp(w_in.sum() / (0.5 * n_old), 0.08, 1.0)       # mostly bare wood -> proportionally fewer leaves
    n_main = p.density * 2.5 * a_env / (0.42 * p.leaf_size ** 2) * cov * min(w_main.sum() / max(w_in.sum(), 1e-6), 1.6)
    n_in = p.density ** 0.5 * 1.1 * a_env / (0.42 * (1.25 * p.leaf_size) ** 2) * cov
    tri_budget = 56500 - tubes.tris - 2600 - n_berry_tris
    fit = min(1.0, tri_budget / (5.0 * (n_main + n_in)))                 # hard cap for huge dense patches
    n_main, n_in = int(n_main * fit), int(n_in * fit)
    big = min(fit ** -0.5, 2.2)                              # ... and larger blades keep the cover closed
    n_main = max(n_main, int(25 + 40 * W * H))               # never a naked skeleton
    if w_main.sum() < 1e-6:
        w_main = (par >= 0) * 1.0
    cnt_m = rs.multinomial(n_main, w_main / max(w_main.sum(), 1e-9))
    cnt_i = rs.multinomial(n_in, w_in / max(w_in.sum(), 1e-9)) if w_in.sum() > 0 else np.zeros(N, dtype=int)
    li = np.concatenate([np.repeat(np.arange(N), cnt_m), np.repeat(np.arange(N), cnt_i)])
    K = len(li)
    inner = np.arange(K) >= int(cnt_m.sum())
    pdir = _unit(pos - pos[np.maximum(par, 0)])
    pdir[par < 0] = (0.0, 1.0)
    dv, ld = pdir[li], leader[li]
    perp = np.stack([-dv[:, 1], dv[:, 0]], axis=1)
    side = np.where(rs.rand(K) < 0.5, -1.0, 1.0)
    latd = np.where(ld, rs.uniform(0.004, 0.02, K), 0.006 + np.abs(rs.normal(0, 0.55 * dk, K)))
    q2 = pos[li] + dv * (rs.uniform(-0.6, 0.6, K) * D)[:, None] + perp * (side * latd)[:, None]
    q2[:, 0] = np.clip(q2[:, 0], -W / 2 + 0.035, W / 2 - 0.035)
    q2[:, 1] = np.clip(q2[:, 1], 0.03, H - 0.03)
    e2 = _unit(perp * side[:, None] * 0.55 + np.array([0.0, -1.0]) * rs.uniform(0.7, 1.4, K)[:, None] + rs.normal(0, 0.36, (K, 2)))
    if has_open:                                             # trimmed: blades do not hang into the opening
        ins = (np.abs(q2[:, 0]) < ow / 2 + 0.09) & (q2[:, 1] > oz - 0.05) & (q2[:, 1] < oz + oh + 0.1)
        e2[ins, 0] = np.abs(e2[ins, 0]) * np.sign(q2[ins, 0])
        e2[ins, 1] = np.abs(e2[ins, 1]) * np.where(q2[ins, 1] > oz + oh, 0.3, -1.0)
        e2 = _unit(e2)
    size = p.leaf_size * big * np.clip(np.exp(rs.normal(0, 0.22, K)), 0.6, 1.45) * np.where(ld, rs.uniform(0.5, 0.8, K), 1.0)
    size *= 0.75 + 0.25 * _sm(0.0, 0.25, 1 - (q2[:, 1] / H))                # smaller towards the growing top
    size = np.where(inner, size * 1.25, size)
    edge = W / 2 - np.abs(q2[:, 0]) < 1.2 * size             # blades at the margin turn inwards: stay inside `width`
    e2[edge, 0] = -np.abs(e2[edge, 0]) * np.sign(q2[edge, 0])
    q2[:, 0] = np.clip(q2[:, 0], -W / 2 + 0.5 * size, W / 2 - 0.5 * size)
    tilt = np.radians(np.where(inner, rs.uniform(3, 13, K), rs.uniform(14, 54, K)))
    y_out = max(p.depth, 0.6 * p.leaf_size * big + 0.03) - size * np.sin(tilt) - 0.004
    y0 = np.where(ld, rs.uniform(0.012, 0.03, K), 0.022 + np.maximum(y_out - 0.022, 0.006) * rs.rand(K) ** 0.75)
    y0 = np.where(inner, rs.uniform(0.011, 0.024, K), y0)
    base = np.stack([q2[:, 0], -y0, q2[:, 1]], axis=1)
    e3 = np.stack([e2[:, 0], np.zeros(K), e2[:, 1]], axis=1)
    # tint classes: 0 = most exposed (fresh green, first to turn) .. 4 = deep inner leaves (dark, last to turn)
    score = (0.4 * (y0 / p.depth) + 0.25 * ld + 0.2 * rs.rand(K) + 0.95 * (_fbm(nc, q2[:, 0] / 0.55, q2[:, 1] / 0.55) - 0.5)
             + 0.15 * q2[:, 1] / H)
    rank = np.argsort(np.argsort(-score)) / max(K, 1)
    lcls = np.searchsorted(np.array([0.08, 0.26, 0.5, 0.74]), rank)
    lcls = np.where(inner, np.maximum(lcls, 3), lcls)
    # detailed 5-lobed blades where the triangle budget allows (leader shoots first), simple blades for the rest
    extra, n_ld, n_oth = tri_budget - 5.0 * K, int((ld & ~inner).sum()), int((~ld & ~inner).sum())
    hi_ld = clamp(extra / (6.0 * max(n_ld, 1)), 0.0, 1.0)
    hi_frac = clamp((extra - 6.0 * n_ld) / (6.0 * max(n_oth, 1)), 0.0, 1.0)
    use_hi = (rs.rand(K) < np.where(ld, hi_ld, hi_frac)) & ~inner
    mature = _sm(0.55, 0.95, q2[:, 1] / H + 0.5 * (y0 / p.depth) - 0.25)     # adult shoots high up carry unlobed leaves
    g = hex_rgb(p.green)
    greens = [_mix(g, hex_rgb("#7da045"), 0.42), _mix(g, hex_rgb("#6f9a3c"), 0.4), g, tuple(c * 0.8 for c in g), tuple(c * 0.58 for c in g)]
    falls = [hex_rgb(c) for c in ("#8a2418", "#b04a1c", "#bf8424", "#a39a36", "#6a7028")]
    leaf_mats = []
    for c in range(5):
        t = float(_sm(0.18 * c - 0.02, 0.18 * c + 0.28, p.autumn))
        leaf_mats.append(mat("leaf", _mix(greens[c], falls[c], t)))
    leaf_objs = []
    n_leaf_tris = 0
    for c in range(5):
        Vs, Fs, UVs, off = [], [], [], 0
        for shape, sel in ((_HI, use_hi), (_LO, ~use_hi)):
            m = (lcls == c) & sel
            k = int(m.sum())
            if k == 0:
                continue
            V, F, uv = _leaves(shape, base[m], e3[m], tilt[m], rs.normal(0, 0.32, k), size[m], rs.uniform(0.92, 1.15, k),
                               rs.uniform(-0.08, 0.38, k), rs.uniform(0.12, 0.62, k), rs.uniform(0.0, 0.8, k),
                               np.clip(rs.normal(0.3, 0.3, k) + 0.5 * mature[m], 0.0, 1.0))
            Vs.append(V)
            Fs.append(F + off)
            UVs.append(uv)
            off += len(V)
        if not Vs:
            continue
        V = np.concatenate(Vs)
        V[:, 1] = np.minimum(V[:, 1], -0.003)
        V[:, 2] = np.maximum(V[:, 2], 0.004)
        o = _np_mesh(f"leaves{c}", V, np.concatenate(Fs), np.concatenate(UVs))
        n_leaf_tris += sum(len(f) for f in Fs)
        smooth(o, 55)
        assign(o, leaf_mats[c])
        paint(o, grime=lambda co, n: 0.55 * np.clip(1.0 + co[:, 1] / 0.05, 0, 1))        # leaves deep in the mass are darker
        leaf_objs.append(o)

    # ---- stems
    stems = tubes.build("stems")
    if stems is not None:
        assign(stems, mat("bark", scale=2.5))
        paint(stems, grime=mask_mul(height_ramp(0.0, 0.3), 0.45),
              special=lambda co, n: np.clip(0.75 * height_ramp(0.0, 0.9)(co, n) * noise_mask(0.12, seed=seed + 5, lo=0.35, hi=0.7)(co, n)
                                            + 0.45 * noise_mask(0.2, seed=seed + 9, lo=0.45, hi=0.75)(co, n), 0, 1))

    # ---- planting pocket: soil, moss cushions and fallen leaves at the foot
    rx = [r[0] for r in roots]
    groups = [[rx[0]]]
    for x in rx[1:]:
        if x - groups[-1][-1] < 0.5:
            groups[-1].append(x)
        else:
            groups.append([x])
    for gi, grp in enumerate(groups):
        cxm, half = 0.5 * (grp[0] + grp[-1]), 0.5 * (grp[-1] - grp[0]) + 0.13 + 2 * trunk
        half = min(half, W / 2 - abs(cxm) + 0.02)
        nf = _vnoise(seed * 5 + gi)

        def mound(u, v, cxm=cxm, half=half, nf=nf):
            a = 2 * pi * u
            k = 1 + 0.4 * (float(nf(cos(a) * 1.3 + 5, sin(a) * 1.3 + 5)) - 0.5)
            x, y = cxm + half * v * k * cos(a), -0.075 - 0.07 * v * k * sin(a)
            z = (0.012 + 2.2 * trunk * 0.5) * (1 - v ** 2.2) * (0.75 + 0.5 * float(nf(x * 30 + 3, y * 30 + 7)))
            return (x, min(y, -0.003), max(z, 0.0))
        so = surface(f"soil{gi}", mound, nu=18, nv=5, closed_u=True)
        recalc_normals(so)
        assign(so, mat("soil"))
        uv_box(so)
        paint(so, special=0.4)
    r_moss = rng.fork("moss")
    for i in range(int(3 + W * 2.2)):
        gx = r_moss.pick(rx) + r_moss.u(-0.45, 0.45) if r_moss.chance(0.65) else r_moss.u(-W / 2 + 0.08, W / 2 - 0.08)
        gx = clamp(gx, -W / 2 + 0.06, W / 2 - 0.06)
        if has_open and oz < 0.05 and abs(gx) < ow / 2 + 0.05:
            continue
        mr, mh = r_moss.u(0.025, 0.06), r_moss.u(0.01, 0.022)
        gy = -r_moss.u(0.6, 1.0) * mr * 0.7 - (0.1 if r_moss.chance(0.3) else 0.0)
        mo = sphere(f"moss{i}", 1.0, loc=(gx, gy, 0.0), segments=8, rings=5, scale=(mr, mr * 0.7, mh))
        realize(mo)
        transform_verts(mo, lambda co: Vector((co.x, min(co.y, -0.002), abs(co.z))))
        displace_noise(mo, strength=0.004, scale=0.03, seed=seed + i)
        transform_verts(mo, lambda co: Vector((co.x, min(co.y, -0.002), max(co.z, 0.0))))
        assign(mo, mat("moss"))
    # fallen leaves lying flat on the paving
    kf = int(p.litter * (8 + 46 * p.autumn) * (0.6 + 0.4 * W))
    if kf > 0:
        fx_ = rs.uniform(-W / 2 + 0.06, W / 2 - 0.06, kf)
        fy_ = -0.04 - rs.exponential(0.14, kf)
        fy_ = np.maximum(fy_, -0.55)
        psi = rs.uniform(0, 2 * pi, kf)
        Yf = np.stack([np.cos(psi), np.sin(psi), np.zeros(kf)], axis=1)
        Zf = _unit(np.stack([rs.normal(0, 0.1, kf), rs.normal(0, 0.1, kf), np.ones(kf)], axis=1))
        Xf = _unit(np.cross(Yf, Zf))
        Yf = np.cross(Zf, Xf)
        fs = p.leaf_size * rs.uniform(0.6, 1.1, kf)
        bf = np.stack([fx_, fy_, 0.006 + 0.3 * fs * 0.1], axis=1)
        V, F, uv = _leaf_frames(_HI, bf, Xf, Yf, Zf, fs, rs.uniform(0.9, 1.1, kf), rs.uniform(0.0, 0.3, kf),
                                -rs.uniform(0.05, 0.4, kf), -rs.uniform(0.0, 0.5, kf))
        V = V.reshape(kf, -1, 3)
        V[:, 0, :] = V[:, 3, :] - Yf * (fs * 0.35)[:, None]                 # petiole lies flat too
        V = V.reshape(-1, 3)
        V[:, 1] = np.minimum(V[:, 1], -0.004)
        V[:, 2] = np.maximum(V[:, 2], 0.003)
        cut = int(kf * (0.35 + 0.5 * p.autumn))
        nvl = len(_HI["pts"])
        for nm, a, b, m_ in (("litter_a", 0, cut, leaf_mats[0] if p.autumn > 0.25 else mat("leaf", "#7a5630")),
                             ("litter_b", cut, kf, mat("leaf", "#7a5630"))):
            if b - a <= 0:
                continue
            Fa = F.reshape(kf, -1, 3)[a:b].reshape(-1, 3) - a * nvl
            lo = _np_mesh(nm, V[a * nvl:b * nvl], Fa, uv[a * nvl:b * nvl])
            smooth(lo, 55)
            assign(lo, m_)
            paint(lo, grime=0.35)
    print(f"DBG umb={n_umb} nodes={DBG_nodes} attractors={len(A)} stem_tris={tubes.tris} leaves={K} hi_frac={hi_frac:.2f} leaf_tris={n_leaf_tris}")
