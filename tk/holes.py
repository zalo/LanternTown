"""tk/holes.py - "no holes": a cheaper level must not let the viewer look through the model where LOD 0 is closed.

The town draws opaque surfaces single-sided with the sky behind them, so a face that went missing shows as a patch of
sky - under a sill, along a string course, in a door. This module measures exactly that, with rays (docs/LOD.md,
"No holes"):

    viewers   stand where people stand: at eye height on rings 3 / 8 / 15 m around a ground-mounted asset (and one
              ring from a terrace above), in front of and below a wall-mounted one, around and below a hanging one
    a ray     is drawn the way the engine draws: the back of a single-sided face is not there (the ray goes on),
              leaves and clear glass are looked through, the ground stops it
    a sample  is a ray whose first surface in LOD 0 is a front face, and whose neighbours in the view are too (the
              outline of a level may move by a cell)
    a hole    is a sample where the level shows nothing at all: `back` - only the backs of faces lay on the ray (the
              viewer looks into the model), `none` - nothing lay on it (counted only further than the level's size
              limit inside the LOD 0 outline: a piece under the size limit may go)

    H = Holes(V0, T0, two0, thru0, mount)               # LOD 0 once
    m = H.measure(V, T, two, thru, level)               # -> {"holes": share of the samples, "back", "none", "worst",
                                                        #     "n", "rays", "deep", "where": [...]}
    m = glb_holes(glb0, glb, mount, level)              # the same on files

Runs under Blender's python (numpy, mathutils); draws nothing from the toolkit's random streams.
"""
import json
import math
import numpy as np
from mathutils.bvhtree import BVHTree

PLANT = ("moss", "hedge", "leaf")
TWO_SIDED = ("glass_clear", "glass_frosted", "glass_lantern", "water", "water_fall")
THROUGH = PLANT + ("glass_clear", "glass_lantern", "water_fall")    # looked through: neither a surface nor a hole

EYE = 1.6                       # eye height of a viewer on the street (m)
RINGS = ((3.0, EYE, 8), (8.0, EYE, 8), (15.0, EYE, 8), (10.0, 5.0, 6))      # distance to the footprint, eye height, viewers at least
CELL = 0.4                      # ray spacing (degrees): 2 cm at 3 m
CELL_MIN = 0.03                 # ... a small asset is sampled finer, down to this
RMAX = 12000                    # rays per viewer at most (the spacing grows)
RMIN = 2500                     # ... and at least, over the asset's box (the spacing shrinks down to CELL_MIN)
MIN_COUNT = 4                   # fewer hole samples than this are no finding
MIN_N = 400                     # ... and an asset with fewer samples than this in all its views is not judged (a tuft
                                # of grass, a string of lights: next to nothing of it is a closed surface)
LIMIT = (0.0, 0.0005, 0.0005)   # the gate: share of the samples that may be holes (0.05 %)
LIMIT_WORST = (0.0, 0.002, 0.002)   # ... and in the worst single view (0.2 %)
MIN_VIEW = 150                  # a view with fewer samples has no say in `worst`
DEEP = 1.0                      # `deep`: the level's surface lies this much further (m) than LOD 0's - no gate
PHASE = 0.37                    # where on their rings the viewers of the gate stand (turns of one viewer spacing)
PHASES = (PHASE, 0.87)
FILL = 1.5                      # a repair that does not hold for the second set of viewers (DENSE) also takes the
FILL_MAX = 0.25                 #   LOD 0 faces within this x the spacing of the rays of a hole sample (.. m at most)
DENSE = 0.5                     # ... the second set still sees this x the holes the first saw, and fails the gate
ROUNDS = 4                      # the repair goes over its viewer sets this often at most (close())
THIN = (0.15, 0.3, 0.55)        # the faces put back come at LOD 0's density: collapsed to the first of these ratios
THIN_MIN = 40                   #   that opens nothing again (patches of at least this many triangles,
THIN_ALL = 100                  #   when the repair took at least this many)


def _erode(m, r=1):
    out = m.copy()
    for _ in range(int(r)):
        o = out.copy()
        o[1:, :] &= out[:-1, :]
        o[:-1, :] &= out[1:, :]
        o[:, 1:] &= out[:, :-1]
        o[:, :-1] &= out[:, 1:]
        o[0, :] = o[-1, :] = False
        o[:, 0] = o[:, -1] = False
        out = o
    return out


class _Soup:
    """triangles the rays can meet; single-sided ones only from their front"""

    def __init__(self, V, T, two=None, thru=None, only=None):
        V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(T, dtype=np.int64).reshape(-1, 3)
        two = np.zeros(len(T), dtype=bool) if two is None else np.asarray(two, dtype=bool)
        keep = np.ones(len(T), dtype=bool) if thru is None else ~np.asarray(thru, dtype=bool)
        if only is not None:
            keep = np.asarray(only, dtype=bool).copy()
        if len(T):
            n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
            ln = np.linalg.norm(n, axis=1)
            keep &= ln > 1e-12
        else:
            n, ln = np.zeros((0, 3)), np.zeros(0)
        self.src = np.flatnonzero(keep)                 # row here -> triangle of the caller
        self.N = n[keep] / np.maximum(ln[keep], 1e-30)[:, None]
        self.two = two[keep]
        self.bvh = None
        if len(self.src):
            used = np.unique(T[keep])
            re = np.zeros(len(V), dtype=np.int64)
            re[used] = np.arange(len(used))
            self.bvh = BVHTree.FromPolygons(V[used].tolist(), re[T[keep]].tolist(), all_triangles=True, epsilon=0.0)

    def cast(self, O, D, far):
        """first surface on each ray -> distance (inf: none), triangle of the caller (-1), went through backs"""
        n = len(D)
        t = np.full(n, np.inf)
        tri = np.full(n, -1, dtype=np.int64)
        back = np.zeros(n, dtype=bool)
        if self.bvh is None or n == 0:
            return t, tri, back
        rc, near = self.bvh.ray_cast, self.bvh.find_nearest_range
        o = np.array(O, dtype=np.float64).reshape(-1, 3)
        if len(o) == 1:
            o = np.repeat(o, n, axis=0)
        else:
            o = o.copy()
        base = np.zeros(n)
        lim = np.array(far, dtype=np.float64) * np.ones(n)
        act = np.flatnonzero(lim > 0.0)
        for _ in range(24):
            if not len(act):
                break
            res = [rc(a, b, c) for a, b, c in zip(o[act].tolist(), D[act].tolist(), lim[act].tolist())]
            idx = np.fromiter((-1 if r[2] is None else r[2] for r in res), dtype=np.int64, count=len(act))
            dist = np.fromiter((0.0 if r[3] is None else r[3] for r in res), dtype=np.float64, count=len(act))
            h = idx >= 0
            a, i, d = act[h], idx[h], dist[h]
            front = self.two[i] | ((self.N[i] * D[a]).sum(1) < 0.0)
            if (~front).any():
                # a back face on top: a front face in the same place (two skins on one plane) is what the engine shows
                for k in np.flatnonzero(~front).tolist():
                    p = o[a[k]] + D[a[k]] * d[k]
                    for _co, _n, j, _d in near(p.tolist(), 1e-4):
                        if self.two[j] or float(self.N[j] @ D[a[k]]) < 0.0:
                            i[k], front[k] = j, True
                            break
            v = a[front]
            t[v] = base[v] + d[front]
            tri[v] = self.src[i[front]]
            c = a[~front]
            back[c] = True
            step = d[~front] + 1e-4
            o[c] += D[c] * step[:, None]
            base[c] += step
            lim[c] -= step
            act = c[lim[c] > 0.0]
        return t, tri, back


def viewers(mount, mn, mx, phase=PHASE):
    """where people stand to look at the asset -> [(eye, ring)]"""
    sh = phase - PHASE
    mn, mx = np.asarray(mn, dtype=np.float64), np.asarray(mx, dtype=np.float64)
    c, w = 0.5 * (mn + mx), mx - mn
    out = []
    if mount in ("ground", "wall_ground"):
        for ri, (d, eye, nmin) in enumerate(RINGS):
            hx, hy = 0.5 * w[0] + d, 0.5 * w[1] + d
            n = int(np.clip(round(4.0 * (hx + hy) / (2.0 * d + 2.0)), nmin, 16))
            for i in range(n):
                a = 2.0 * math.pi * (i + phase) / n
                ca, sa = math.cos(a), math.sin(a)
                if mount == "wall_ground" and sa > -0.1:
                    continue                    # the wall stands behind y = 0
                r = min(hx / max(abs(ca), 1e-9), hy / max(abs(sa), 1e-9))
                out.append((np.array([c[0] + r * ca, c[1] + r * sa, eye]), ri))
    elif mount == "wall":
        rad = 0.5 * float(max(w[0], w[2]))
        for ri, d in enumerate((3.0, 8.0)):
            for el in (10.0, -20.0, -45.0):
                for az in (-60.0, -30.0, 0.0, 30.0, 60.0):
                    a, e = math.radians(az + 30.0 * sh), math.radians(el + 20.0 * sh)
                    out.append((c + (d + rad) * np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)]), ri))
    else:
        rad = 0.5 * float(max(w))
        for ri, d in enumerate((3.0, 8.0)):
            for el in (0.0, -25.0, -50.0):
                for i in range(8):
                    a, e = 2.0 * math.pi * (i + phase) / 8, math.radians(el + 20.0 * sh)
                    out.append((c + (d + rad) * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]), ri))
    return out


def _grid(eye, mn, mx, cell=CELL, rmax=RMAX):
    """rays from the eye over the asset's box -> directions (rows, columns, 3), spacing (radians)"""
    c = 0.5 * (mn + mx)
    cs = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]) - eye
    f = c - eye
    az0 = math.atan2(f[1], f[0])
    az = np.arctan2(cs[:, 1], cs[:, 0]) - az0
    az = (az + math.pi) % (2.0 * math.pi) - math.pi
    el = np.arctan2(cs[:, 2], np.hypot(cs[:, 0], cs[:, 1]))
    a0, a1 = max(float(az.min()), math.radians(-75.0)), min(float(az.max()), math.radians(75.0))
    e0, e1 = max(float(el.min()), math.radians(-80.0)), min(float(el.max()), math.radians(80.0))
    if a1 <= a0 or e1 <= e0:
        return np.zeros((0, 0, 3)), math.radians(cell)
    area = (a1 - a0) * (e1 - e0)
    step = max(math.radians(cell), math.sqrt(area / rmax))
    step = min(step, max(math.radians(CELL_MIN), math.sqrt(area / RMIN)))
    na, ne = int(math.ceil((a1 - a0) / step)) + 3, int(math.ceil((e1 - e0) / step)) + 3
    A = az0 + a0 + (np.arange(na) - 1.0) * step
    E = np.clip(e0 + (np.arange(ne) - 1.0) * step, -1.55, 1.55)
    ce = np.cos(E)[:, None]
    D = np.stack([ce * np.cos(A)[None, :], ce * np.sin(A)[None, :], np.sin(E)[:, None] * np.ones((1, na))], axis=2)
    return D, step


def size_limit(level, diag):
    k = float(np.clip(diag / 6.0, 0.6, 2.5))
    return (0.0, 0.035 * k, 0.14 * k)[level]


class Holes:
    """LOD 0 as the viewers see it; measure() compares a level against it."""

    def __init__(self, V, T, two=None, thru=None, mount="ground", bounds=None, mat=None, names=None, cell=CELL, rmax=RMAX,
                 rings=None, phase=PHASE):
        V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(T, dtype=np.int64).reshape(-1, 3)
        self.mount = mount
        if bounds is None:
            u = np.unique(T) if len(T) else np.zeros(0, dtype=np.int64)
            bounds = (V[u].min(0), V[u].max(0)) if len(u) else (np.zeros(3), np.zeros(3))
        self.mn, self.mx = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
        self.diag = float(np.linalg.norm(self.mx - self.mn))
        self.mat = None if mat is None else np.asarray(mat, dtype=np.int64)
        self.names = list(names) if names is not None else None
        self.ground = mount in ("ground", "wall_ground")
        s0 = _Soup(V, T, two, thru)
        self._s0, self._V, self._T, self._kd = s0, V, T, None
        far = self.diag + 1.0
        self.views = []
        self.rays = 0
        for eye, ring in viewers(mount, self.mn, self.mx, phase):
            if rings is not None and ring not in rings:
                continue
            D, step = _grid(eye, self.mn, self.mx, cell, rmax)
            if D.size == 0:
                continue
            sh = D.shape[:2]
            Df = D.reshape(-1, 3)
            reach = np.linalg.norm(0.5 * (self.mn + self.mx) - eye) + far
            lim = np.full(len(Df), reach)
            if self.ground:
                dn = Df[:, 2] < -1e-6
                lim[dn] = np.minimum(lim[dn], (eye[2] - min(0.0, float(self.mn[2]))) / -Df[dn, 2])   # a quay goes below z = 0
            t, tri, _ = s0.cast(eye, Df, lim)
            hit = np.isfinite(t).reshape(sh)
            inner = _erode(hit, 1)
            if not inner.any():
                continue
            self.rays += len(Df)
            self.views.append({"eye": eye, "ring": ring, "D": Df, "lim": lim, "t": t, "tri": tri, "hit": hit, "inner": inner,
                               "step": step, "sh": sh})
        self.n = int(sum(int(v["inner"].sum()) for v in self.views))

    def measure(self, V, T, two=None, thru=None, level=1, detail=True, veil=None, grow=0, box=None):
        """-> {"holes", "back", "none", "worst", "n", "deep", "where"}; with detail the hole points too ("pts": (k, 3)
        on LOD 0, "eyes": the viewer of each, "kind": 0 back | 1 none).  `veil`: per triangle, leaves and other cut-out
        cards of the level - looked through like in LOD 0, but a ray that met one is not a hole (a basket that became
        a tuft of leaves is a change of the picture, not a view through the model).  grow: also "grown" - the LOD 0
        triangles (rows of T as given to Holes) shown at the hole samples and their neighbours `grow` rays around.
        box: (min, max) - only the samples whose LOD 0 point lies in it (a step that changed one part asks for those:
        "count" and the points are what it reads, the shares are still of all samples)"""
        s = _Soup(V, T, two, thru)
        sv = None
        if veil is not None and np.asarray(veil).any():
            sv = _Soup(V, T, np.ones(len(np.asarray(T).reshape(-1, 3)), dtype=bool), None, only=veil)
        ms = size_limit(level, self.diag)
        nb = nn = nd = 0
        worst, wv = 0.0, None
        pts, eyes, kind, mats, grown, svi, sri = [], [], [], [], [], [], []
        for vi, v in enumerate(self.views):
            sel = np.flatnonzero(v["inner"].ravel())
            if box is not None:
                if "P" not in v:
                    v["P"] = v["eye"][None, :] + v["D"] * np.where(np.isfinite(v["t"]), v["t"], 0.0)[:, None]
                p_ = v["P"][sel]
                sel = sel[((p_ >= box[0]) & (p_ <= box[1])).all(1)]
                if not len(sel):
                    continue
            t, _tri, back = s.cast(v["eye"], v["D"][sel], v["lim"][sel])
            gone = ~np.isfinite(t)
            hb = gone & back
            hn = gone & ~back
            if hn.any():
                # nothing on the ray: only further inside the LOD 0 outline than the size limit
                tm = float(np.median(v["t"][sel]))
                r = int(np.clip(math.ceil(ms / max(tm * v["step"], 1e-9)), 1, 40))
                deep_in = _erode(v["hit"], r).ravel()[sel]
                hn &= deep_in
            if sv is not None and (hb.any() or hn.any()):
                q = np.flatnonzero(hb | hn)
                tv, _t, _b = sv.cast(v["eye"], v["D"][sel[q]], v["lim"][sel[q]])
                cov = np.zeros(len(sel), dtype=bool)
                cov[q] = np.isfinite(tv)
                hb &= ~cov
                hn &= ~cov
            k = int(hb.sum()) + int(hn.sum())
            nb += int(hb.sum())
            nn += int(hn.sum())
            nd += int((np.isfinite(t) & (t > v["t"][sel] + DEEP)).sum())
            if len(sel) >= MIN_VIEW and k / len(sel) > worst:
                worst, wv = k / len(sel), vi
            if k and grow:
                g = np.zeros(v["sh"], dtype=bool).ravel()
                g[sel[hb | hn]] = True
                g = ~_erode(~g.reshape(v["sh"]), grow)
                g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = False
                g = g.ravel() & (v["tri"] >= 0)
                grown.append(v["tri"][g])
            if k and detail:
                for m_, kd in ((hb, 0), (hn, 1)):
                    if m_.any():
                        r_ = sel[m_]
                        pts.append(v["eye"][None, :] + v["D"][r_] * v["t"][r_][:, None])
                        eyes.append(np.repeat(v["eye"][None, :], len(r_), axis=0))
                        kind.append(np.full(len(r_), kd))
                        mats.append(v["tri"][r_])
                        svi.append(np.full(len(r_), vi))
                        sri.append(r_)
        n = max(self.n, 1)
        out = {"holes": round((nb + nn) / n, 6), "back": round(nb / n, 6), "none": round(nn / n, 6), "worst": round(worst, 5),
               "n": self.n, "deep": round(nd / n, 5), "count": nb + nn}
        if wv is not None:
            out["worst_eye"] = [round(float(x), 2) for x in self.views[wv]["eye"]]
        if grow:
            out["grown"] = np.unique(np.concatenate(grown)) if grown else np.zeros(0, dtype=np.int64)
        if detail:
            if pts:
                P, E, K, TR = np.vstack(pts), np.vstack(eyes), np.concatenate(kind), np.concatenate(mats)
            else:
                P, E, K, TR = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
            out["pts"], out["eyes"], out["kind"], out["tri"] = P, E, K, TR
            out["view"] = np.concatenate(svi) if svi else np.zeros(0, dtype=np.int64)
            out["ray"] = np.concatenate(sri) if sri else np.zeros(0, dtype=np.int64)
            out["where"] = self.where(P, TR)
        return out

    def around(self, view, ray, grow=1, fill=0.0):
        """the LOD 0 triangles (rows) shown at the samples (measure(): "view", "ray") and `grow` rays around them -
        and, where LOD 0 is finer than the rays are apart (a figure, a rope, a carved stone), the triangles between
        them: those within `fill` x the spacing of the rays of a sample's point that face its viewer and that he
        sees. Without these the patch is a scatter of small faces and the next viewer looks between them."""
        rows = []
        for vi in np.unique(view).tolist():
            v = self.views[vi]
            r0 = ray[view == vi]
            g = np.zeros(v["sh"], dtype=bool).ravel()
            g[r0] = True
            if grow:
                g = ~_erode(~g.reshape(v["sh"]), grow)
                g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = False
                g = g.ravel()
            g &= v["tri"] >= 0
            rows.append(v["tri"][g])
            if fill and len(self._s0.src):
                if self._kd is None:
                    from mathutils.kdtree import KDTree
                    C = self._V[self._T[self._s0.src]].mean(1)
                    kd = KDTree(len(C))
                    for i, c in enumerate(C.tolist()):
                        kd.insert(c, i)
                    kd.balance()
                    self._kd, self._C = kd, C
                r0 = r0[np.isfinite(v["t"][r0])]
                P = v["eye"][None, :] + v["D"][r0] * v["t"][r0][:, None]
                rad = np.minimum(fill * v["t"][r0] * v["step"], FILL_MAX)
                cand = set()
                for p, r in zip(P.tolist(), rad.tolist()):
                    cand.update(i for _c, i, _d in self._kd.find_range(p, r))
                cand = np.fromiter(cand, dtype=np.int64, count=len(cand))
                if len(cand):
                    d = self._C[cand] - v["eye"]
                    dist = np.linalg.norm(d, axis=1)
                    d = d / np.maximum(dist, 1e-12)[:, None]
                    ok = self._s0.two[cand] | ((self._s0.N[cand] * d).sum(1) < 0.0)
                    cand, d, dist = cand[ok], d[ok], dist[ok]
                    t, _tri, _b = self._s0.cast(v["eye"], d, dist + 1e-3)
                    rows.append(self._s0.src[cand[~(t < dist - 2e-3)]])
        return np.unique(np.concatenate(rows)) if rows else np.zeros(0, dtype=np.int64)

    def where(self, P, TR, top=5):
        """the holes by LOD 0 material and place -> [{"mat", "n", "at": [x, y, z], "box": [min, max]}] (largest first)"""
        if not len(P):
            return []
        if self.mat is not None and self.names is not None:
            key = self.mat[TR]
        else:
            key = np.zeros(len(P), dtype=np.int64)
        cellsz = max(0.5, self.diag / 12.0)
        cell = np.floor((P - self.mn) / cellsz).astype(np.int64)
        code = ((key * 4096 + cell[:, 0]) * 4096 + cell[:, 1]) * 4096 + cell[:, 2]
        u, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
        rows = []
        for j in np.argsort(-cnt)[:top].tolist():
            q = P[inv == j]
            rows.append({"mat": self.names[int(key[inv == j][0])] if self.names is not None and self.mat is not None else "",
                         "n": int(cnt[j]), "at": [round(float(x), 2) for x in np.median(q, axis=0)],
                         "box": [[round(float(x), 2) for x in q.min(0)], [round(float(x), 2) for x in q.max(0)]]})
        return rows


def ok(level, m):
    """the gate -> [reason] (empty: no holes)"""
    if not m or m.get("n", 0) < MIN_N:
        return []
    why = []
    if m.get("count", MIN_COUNT) >= MIN_COUNT and (m["holes"] > LIMIT[level] or m["worst"] > LIMIT_WORST[level]):
        w = (m.get("where") or [{}])[0]
        why.append(f"holes: {100 * m['holes']:.3f} % of the views show through the model (limit {100 * LIMIT[level]:.2f} %), "
                   f"worst view {100 * m['worst']:.2f} % (limit {100 * LIMIT_WORST[level]:.1f} %)"
                   + (f"; most at {w.get('at')} on {w.get('mat')}" if w else ""))
    return why


def brief(m):
    """what goes into the sidecar"""
    if not m:
        return None
    return {k: m[k] for k in ("holes", "back", "none", "worst", "n", "deep", "count", "worst_eye", "where") if k in m}


# ----------------------------------------------------------------------------
# on files
# ----------------------------------------------------------------------------

def soup_of(prims):
    """tk.lodimage.load_glb() primitives -> V, T, two-sided, looked through, material row per triangle, material names,
    veil (the triangles of plant materials, see Holes.measure)"""
    Vs, Ts, W, H, M, L, names, off = [], [], [], [], [], [], [], 0
    for p in prims:
        n = len(p["I"])
        if not n:
            continue
        lib = p.get("lib", "")
        Vs.append(p["P"])
        Ts.append(np.asarray(p["I"], dtype=np.int64) + off)
        off += len(p["P"])
        W.append(np.full(n, bool(p.get("double")) or lib.startswith(TWO_SIDED), dtype=bool))
        H.append(np.full(n, lib.startswith(THROUGH), dtype=bool))
        L.append(np.full(n, lib.startswith(PLANT), dtype=bool))
        nm = f"{p.get('name') or lib} ({lib})" if p.get("name") and p.get("name") != lib else lib
        if nm not in names:
            names.append(nm)
        M.append(np.full(n, names.index(nm), dtype=np.int64))
    if not Vs:
        z = np.zeros(0, dtype=bool)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), z, z, np.zeros(0, dtype=np.int64), [], z
    return np.vstack(Vs), np.vstack(Ts), np.concatenate(W), np.concatenate(H), np.concatenate(M), names, np.concatenate(L)


def of_prims(prims0, mount, **kw):
    V, T, W, H, M, names, _veil = soup_of(prims0)
    return Holes(V, T, W, H, mount, mat=M, names=names, **kw)


def measure_prims(H, prims, level, detail=True):
    V, T, W, Hh, _M, _names, veil = soup_of(prims)
    return H.measure(V, T, W, Hh, level, detail=detail, veil=veil)


def glb_holes(glb0, glb, mount, level):
    from . import lodimage as _im
    return measure_prims(of_prims(_im.load_glb(glb0), mount), _im.load_glb(glb), level)


# ----------------------------------------------------------------------------
# on the scene (Blender objects), and the repair
# ----------------------------------------------------------------------------

def _mat_flags(m):
    """material -> (two-sided in the engine, looked through, veil)"""
    lib = str(m.get("tsq_lib", "")) if m is not None else ""
    two = lib.startswith(TWO_SIDED) or lib.startswith(PLANT)
    if not two and m is not None:
        p = m.get("tsq_params")
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                p = None
        try:
            two = bool(p and p.get("double"))
        except Exception:
            two = False
    return two, lib.startswith(THROUGH), lib.startswith(PLANT), lib


def obj_soup(objs):
    """mesh objects -> {"V", "T", "two", "thru", "veil", "own": index into objs per triangle, "poly": its polygon,
    "mat": row of "names" per triangle, "names"} in world space"""
    from .core import world_matrix
    Vs, Ts, W, H, L, O, P, M, names, off = [], [], [], [], [], [], [], [], [], 0
    for oi, o in enumerate(objs):
        me = o.data
        npo = len(me.polygons)
        if not npo:
            continue
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        mw = np.array(world_matrix(o))        # (not matrix_world: that is stale until the depsgraph runs - tk.reduce._prep)
        if not np.allclose(mw, np.eye(4), atol=1e-9):
            co = co @ mw[:3, :3].T + mw[:3, 3]
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        t = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", t)
        tp = np.empty(nt, dtype=np.int32)
        me.loop_triangles.foreach_get("polygon_index", tp)
        mi = np.empty(npo, dtype=np.int32)
        me.polygons.foreach_get("material_index", mi)
        mats = list(me.materials) or [None]
        fl = [_mat_flags(m) for m in mats]
        mi = np.clip(mi, 0, len(mats) - 1)[tp]
        row = []
        for m, f in zip(mats, fl):
            nm = f"{m.name} ({f[3]})" if m is not None and m.name != f[3] else f[3]
            if nm not in names:
                names.append(nm)
            row.append(names.index(nm))
        Vs.append(co)
        Ts.append(t.reshape(-1, 3).astype(np.int64) + off)
        off += len(co)
        W.append(np.array([f[0] for f in fl], dtype=bool)[mi])
        H.append(np.array([f[1] for f in fl], dtype=bool)[mi])
        L.append(np.array([f[2] for f in fl], dtype=bool)[mi])
        M.append(np.array(row, dtype=np.int64)[mi])
        O.append(np.full(nt, oi, dtype=np.int64))
        P.append(tp.astype(np.int64))
    if not Vs:
        z, zi = np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
        return {"V": np.zeros((0, 3)), "T": np.zeros((0, 3), dtype=np.int64), "two": z, "thru": z, "veil": z, "own": zi, "poly": zi,
                "mat": zi, "names": []}
    return {"V": np.vstack(Vs), "T": np.vstack(Ts), "two": np.concatenate(W), "thru": np.concatenate(H), "veil": np.concatenate(L),
            "own": np.concatenate(O), "poly": np.concatenate(P), "mat": np.concatenate(M), "names": names}


def of_objs(objs, mount, bounds=None, phases=PHASES, **kw):
    """LOD 0 as it stands in the scene -> (the obj_soup(), [Holes per phase])"""
    s = obj_soup(objs)
    return s, [Holes(s["V"], s["T"], s["two"], s["thru"], mount, bounds=bounds, mat=s["mat"], names=s["names"], phase=p, **kw)
               for p in phases]


def measure_objs(H, objs, level, detail=False, grow=0, box=None):
    s = obj_soup(objs)
    return H.measure(s["V"], s["T"], s["two"], s["thru"], level, detail=detail, veil=s["veil"], grow=grow, box=box)


def seen_rows(Hs):
    """the LOD 0 triangles (rows) any viewer's ray shows"""
    rows = [v["tri"][v["tri"] >= 0] for H in Hs for v in H.views]
    return np.unique(np.concatenate(rows)) if rows else np.zeros(0, dtype=np.int64)


def thin_mesh(me, ratio):
    """collapse-decimated copy of a patch (tk.reduce._close, tk.final._close_holes) -> new mesh"""
    import bpy
    import bmesh
    w = me.copy()
    bm = bmesh.new()
    bm.from_mesh(w)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
    bm.to_mesh(w)
    bm.free()
    o = bpy.data.objects.new("_tsq_thin", w)
    bpy.context.scene.collection.objects.link(o)
    m = o.modifiers.new("THIN", 'DECIMATE')
    m.decimate_type = 'COLLAPSE'
    m.ratio = float(ratio)
    m.use_collapse_triangulate = True
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    out = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    bpy.data.objects.remove(o)
    bpy.data.meshes.remove(w)
    out.validate()
    return out


def mesh_tris(me):
    n = len(me.polygons)
    if not n:
        return 0
    lt = np.empty(n, dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    return int((lt - 2).sum())


def thin(Hs, level, objs, patches, swap):
    """The repair again, cheaper: patches = [(key, mesh)] put back by close(); swap(key, mesh) shows that mesh in
    place of the patch. The patches are collapsed together to the first ratio of THIN at which the street sees no
    more holes than with them whole; otherwise they stay whole. -> triangles saved (the meshes given stay the
    caller's to free)"""
    import bpy
    big = [(k, me) for k, me in patches if mesh_tris(me) >= THIN_MIN]
    if not big or sum(mesh_tris(me) for _k, me in patches) < THIN_ALL:
        return 0

    def count():
        return sum(measure_objs(H, objs(), level)["count"] for H in Hs)

    c0 = count()
    for r in THIN:
        new = [(k, me, thin_mesh(me, r)) for k, me in big]
        for k, _me, t in new:
            swap(k, t)
        if count() <= c0:
            return sum(mesh_tris(me) - mesh_tris(t) for _k, me, t in new)
        for k, me, t in new:
            swap(k, me)
            bpy.data.meshes.remove(t)
    return 0


def close(Hs, level, objs, put, behind=None):
    """The repair: where a viewer looks through the level, the faces LOD 0 shows him there are put back.
    objs() -> the level's objects now; put(rows) puts the polygons of those LOD 0 triangles back (it is told the
    rows of the hole samples and of the rays around them) -> triangles added.
    behind(eyes, points) -> (per sample: closed, triangles added): the caller's cheaper way, tried first - a face
    it took away earlier that lies right behind what LOD 0 shows there (tk.reduce).
    One pass per viewer set: a face put back closes its ray for good. It can open another, though: a ray past the
    edge of what was left of a thin shell (a bucket, a fire basket) met nothing, which this close to the outline is
    no hole, and now meets the back of a face put back for a viewer on the other side - which is one. So the passes
    are gone through again, ROUNDS times at most, as long as a set still fails the gate.
    -> (the first set's reading before, triangles added)"""
    first, added, fill, was = None, 0, 0.0, None
    for rnd in range(ROUNDS):
        if rnd and not (added and any(ok(level, measure_objs(H, objs(), level)) for H in Hs)):
            break
        for k, H in enumerate(Hs):
            m = measure_objs(H, objs(), level, detail=True)
            if first is None:
                first = m
            if not m["count"]:
                continue
            if not rnd and k and was is not None and m["count"] > DENSE * len(was[0]) and ok(level, m):
                # what closed the views of the first set did nothing for the second: LOD 0 is finer here than the
                # rays are apart, and the faces put back are a scatter. Solid patches then (around(fill=)), for the
                # first too.
                fill = FILL
                added += int(put(Hs[0].around(was[0], was[1], 1, fill)))
            left = np.ones(m["count"], dtype=bool)
            if behind is not None:
                done, n = behind(m["eyes"], m["pts"])
                left &= ~done
                added += int(n)
            if left.any():
                added += int(put(H.around(m["view"][left], m["ray"][left], 1, fill)))
            if not rnd and k == 0:
                was = (m["view"], m["ray"])
    return first, added
