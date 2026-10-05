"""Complete pitched roof for a rectangular plan (gable / hip / half-hip / gambrel), built the way a
carpenter and a tiler would: every course of the covering is a real step (sawtooth course strips
registered to the tile texture, wavy and sagging with the rafters), a doubled eaves course of
individual tiles, bedded half-round ridge / hip tiles (or a lead roll), sprocketed (bell-cast) eaves on
shaped oak rafter feet over a wall plate - or a moulded stone cornice with modillions -, half-round
gutters on strap brackets, and gables closed with bargeboards over a half-timbered / plastered / brick
/ boarded tympanum with a lit attic window, or with a coped parapet on kneelers. Verges carry a tile
undercloak on a mortar bed; a cast-iron rooflight sits on a lead-clad kerb with apron, back gutter and
stepped soakers; slipped and replaced tiles / split shingles lie on the courses below them.
Origin: centre of the plan at the wall head (z = 0 is the lowest point; walls of the building stop at
z = 0, `width` along X and `depth` along Y at the wall faces, the eaves overhang beyond).
Levels: lod 1 keeps a real step at every other course and every piece of the eaves / verge / ridge kit in a
leaner section without the faces nobody sees; lod 2 is a flat sheet per slope that keeps the eaves line, the
bell-cast and the sag, with the kit as single faces. Windows, timbering and the rooflight stay at every level.
"""
from tk import *
import bpy
import numpy as np
from math import sin, cos, tan, atan2, acos, radians, degrees, pi, ceil, floor, hypot, sqrt
from mathutils import Vector, Matrix

META = {"name": "Tiled gable roof module", "class": "component", "mount": "ground",
        "tri_budget": 16000, "focus": [2.4, -2.0, 1.6], "lod": True}   # a whole roof: ~50 real courses per slope + eaves / ridge / verge pieces
PARAMS = {
    "width": {"default": 7.0, "min": 3.0, "max": 14.0, "desc": "plan size along X at the wall faces (m)"},
    "depth": {"default": 5.6, "min": 3.0, "max": 10.0, "desc": "plan size along Y at the wall faces (m)"},
    "pitch": {"default": 50.0, "min": 30.0, "max": 62.0, "desc": "main roof pitch (deg)"},
    "type": {"default": "gable", "choices": ["gable", "hip", "half_hip", "gambrel"], "desc": "roof form"},
    "ridge": {"default": "x", "choices": ["x", "y"], "desc": "ridge direction (y = gable faces the front)"},
    "overhang": {"default": 0.42, "min": 0.15, "max": 0.8, "desc": "eaves overhang beyond the wall face (m)"},
    "covering": {"default": "roof_tile_clay", "choices": ["roof_tile_clay", "roof_slate", "wood_shingle"], "desc": "roof covering"},
    "sag": {"default": 0.1, "min": 0.0, "max": 0.35, "desc": "ridge sag at mid span (m); also scales the rafter undulation"},
    "kick": {"default": 0.7, "min": 0.0, "max": 1.0, "desc": "bell-cast (sprocketed) eaves: 0 = straight slope, 1 = strong kick"},
    "gable": {"default": "bargeboard", "choices": ["bargeboard", "scalloped", "parapet"], "desc": "gable verge: moulded bargeboards, scalloped bargeboards or coped parapet"},
    "gable_wall": {"default": "timber", "choices": ["timber", "plaster", "brick", "boards", "none"], "desc": "gable tympanum infill"},
    "attic_window": {"default": "lit", "choices": ["lit", "dark", "none"], "desc": "small attic casement in each gable"},
    "rooflight": {"default": "lit", "choices": ["lit", "dark", "none"], "desc": "small cast-iron rooflight on the front slope"},
    "ladder": {"default": "front", "choices": ["front", "back", "none"], "desc": "sweep's cat ladder lying on a slope up to the ridge"},
    "eaves": {"default": "rafters", "choices": ["rafters", "cornice"], "desc": "exposed shaped rafter feet or moulded stone cornice"},
    "ridge_tile": {"default": "half_round", "choices": ["half_round", "crested", "lead_roll"], "desc": "ridge / hip capping"},
    "finial": {"default": "none", "choices": ["none", "ball", "spike"], "desc": "finials at the ridge ends"},
    "gutter": {"default": "zinc", "choices": ["zinc", "copper", "iron", "none"], "desc": "half-round eaves gutter metal"},
    "wall_color": {"default": "#d8c9a6", "desc": "plaster / limewash colour of the gable and wall head"},
    "timber_color": {"default": "#6a4c36", "desc": "stain of the oak carpentry"},
    "trim_color": {"default": "#4a2a22", "desc": "paint of bargeboards, window frames and boarding"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "moss / lichen on the covering"},
}
VARIANTS = {
    "hip_slate": {"type": "hip", "covering": "roof_slate", "width": 8.0, "depth": 6.0, "pitch": 42.0, "eaves": "cornice",
                  "ridge_tile": "lead_roll", "finial": "spike", "gutter": "copper", "sag": 0.02, "kick": 0.9, "overhang": 0.5, "ladder": "none", "seed": 4},
    "half_hip_shingle": {"type": "half_hip", "covering": "wood_shingle", "width": 6.0, "depth": 5.0, "pitch": 55.0, "gable": "scalloped",
                         "gable_wall": "boards", "ridge_tile": "lead_roll", "sag": 0.16, "trim_color": "#2f4a3c", "gutter": "iron",
                         "overhang": 0.55, "moss": 0.8, "rooflight": "none", "ladder": "none", "seed": 11},
    "gambrel_front": {"type": "gambrel", "ridge": "y", "width": 6.4, "depth": 7.0, "pitch": 48.0, "ridge_tile": "crested", "finial": "ball",
                      "gable_wall": "plaster", "wall_color": "#8d836a", "trim_color": "#5a2620", "sag": 0.09, "kick": 0.5, "seed": 23},
    "parapet_slate": {"gable": "parapet", "gable_wall": "brick", "covering": "roof_slate", "width": 5.2, "depth": 6.4, "pitch": 57.0,
                      "eaves": "cornice", "finial": "ball", "kick": 0.25, "overhang": 0.32, "sag": 0.03, "gutter": "iron", "moss": 0.25, "seed": 31},
}

COVER = {
    "roof_tile_clay": dict(g=0.10, th=0.014, tw=0.17, tt=0.013, wav=0.13, ridge=("terracotta", "#a35a3e"), r=0.125),
    "roof_slate": dict(g=0.15, th=0.008, tw=0.25, tt=0.007, wav=0.07, ridge=("terracotta", "#5f555c"), r=0.115),
    "wood_shingle": dict(g=0.10, th=0.017, tw=0.13, tt=0.014, wav=0.24, ridge=("terracotta", "#8a5a44"), r=0.115),
}
T_SK, T_R = 0.045, 0.11          # covering + battens + sarking above the rafters / rafter depth


class VNoise:
    """Small seeded 2D value noise (-1..1)."""

    def __init__(self, seed, n=64):
        self.t = np.random.default_rng(int(seed) + 1234).uniform(-1.0, 1.0, (n, n))
        self.n = n

    def __call__(self, x, y=0.0):
        n = self.n
        xi, yi = floor(x), floor(y)
        fx, fy = x - xi, y - yi
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        t = self.t
        x0, x1, y0, y1 = xi % n, (xi + 1) % n, yi % n, (yi + 1) % n
        return float((t[x0, y0] * (1 - fx) + t[x1, y0] * fx) * (1 - fy) + (t[x0, y1] * (1 - fx) + t[x1, y1] * fx) * fy)


class Profile:
    """Roof section from the eaves edge to the ridge: d = plan distance inwards from the wall face, z = height
    of the batten plane; the pitch may vary with d (bell-cast eaves, gambrel break)."""

    def __init__(self, d0, z0, d1, pitch_fn, step=0.02):
        n = max(2, int(ceil((d1 - d0) / step)))
        self.d = np.linspace(d0, d1, n + 1)
        z = [z0]
        for i in range(n):
            dm = 0.5 * (self.d[i] + self.d[i + 1])
            z.append(z[-1] + tan(radians(pitch_fn(dm))) * (self.d[i + 1] - self.d[i]))
        self.z = np.array(z)
        self.s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.d), np.diff(self.z)))])
        self.len = float(self.s[-1])
        self.pf = pitch_fn

    def at_s(self, s):
        return float(np.interp(s, self.s, self.d)), float(np.interp(s, self.s, self.z))

    def z_of_d(self, d):
        return float(np.interp(d, self.d, self.z))

    def d_of_z(self, z):
        return float(np.interp(z, self.z, self.d))

    def s_of_d(self, d):
        return float(np.interp(d, self.d, self.s))

    def ang_d(self, d):
        return radians(self.pf(min(max(d, float(self.d[0])), float(self.d[-1]))))

    def pt(self, s, lift=0.0):
        d, z = self.at_s(s)
        a = self.ang_d(d)
        return d - sin(a) * lift, z + cos(a) * lift


def make_mesh(name, verts, faces, uvs=None):
    obj = mesh_object(name, verts, faces)
    if uvs is not None:
        me = obj.data
        li = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", li)
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(uvs, dtype=np.float32)[li].ravel())
        uv_keep(obj)
    return obj


def setF(obj, F):
    obj.matrix_basis = F @ obj.matrix_basis
    return obj


def _used(V, fs, UV=None):
    """Drop the vertices no face uses (cheaper levels leave faces out)."""
    idx = sorted({i for f_ in fs for i in f_})
    m = {i: k for k, i in enumerate(idx)}
    return [V[i] for i in idx], [tuple(m[i] for i in f_) for f_ in fs], (None if UV is None else [UV[i] for i in idx])


def ribbon(name, F, top, bot, d0, d1, sides="fktbe"):
    """Solid between two polylines top / bot [(u, z)] (u increasing) from d0 (outer) to d1, in face space.
    sides: the faces that exist - f(ront, at d0), bac(k, at d1), t(op), b(ottom), e(nds); the cheaper levels
    leave out what is hidden."""
    n = len(top)
    V = []
    for (u, z) in top:
        V += [(u, d0, z), (u, d1, z)]
    for (u, z) in bot:
        V += [(u, d0, z), (u, d1, z)]
    fs = []
    for k in range(n - 1):
        t0, t1, t0n, t1n = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        b0, b1, b0n, b1n = 2 * n + 2 * k, 2 * n + 2 * k + 1, 2 * n + 2 * k + 2, 2 * n + 2 * k + 3
        q4 = [(b0, b0n, t0n, t0), (b1, t1, t1n, b1n), (t0, t0n, t1n, t1), (b0, b1, b1n, b0n)]
        fs += q4 if len(sides) == 5 else [q for q, c in zip(q4, "fktb") if c in sides]
    e = n - 1
    if "e" in sides:
        fs.append((2 * n, 0, 1, 2 * n + 1))
        fs.append((2 * n + 2 * e, 2 * n + 2 * e + 1, 2 * e + 1, 2 * e))
    if len(sides) < 5:
        V, fs, _ = _used(V, fs)
    return mesh_object(name, [F @ Vector(v) for v in V], fs)


def eaves_run(name, F, section, u0, u1, m0=0.0, m1=0.0, nst=2, caps=(True, True), skip=()):
    """Extrude a closed section [(q, z)] (q = distance OUT from the wall face) along the eaves from u0 to u1.
    m0 / m1 = 1 mitres that end at 45 deg through the wall corner (hip roofs). UVs: U along the run.
    skip: edges of the section (index of their first point) that get no face - hidden sides at the cheaper levels."""
    sec = list(section)
    area = sum(sec[i][0] * sec[(i + 1) % len(sec)][1] - sec[(i + 1) % len(sec)][0] * sec[i][1] for i in range(len(sec)))
    n = len(sec)
    if area < 0:
        sec.reverse()
        skip = {(n - 2 - j) % n for j in skip}
    per = [0.0]
    for j in range(n):
        q0, z0 = sec[j]
        q1, z1 = sec[(j + 1) % n]
        per.append(per[-1] + hypot(q1 - q0, z1 - z0))
    V, UV, fs = [], [], []
    for k in range(nst):
        t = k / (nst - 1)
        for j in range(n + 1):
            q, z = sec[j % n]
            u = u0 + (u1 - u0) * t
            if k == 0:
                u -= m0 * q
            if k == nst - 1:
                u += m1 * q
            V.append(F @ Vector((u, -q, z)))
            UV.append((u, per[j]))
    for k in range(nst - 1):
        for j in range(n):
            a_, b_ = k * (n + 1) + j, (k + 1) * (n + 1) + j
            if j not in skip:
                fs.append((a_, b_, b_ + 1, a_ + 1))
    if caps[0]:
        fs.append(tuple(range(n)))
    if caps[1]:
        base = (nst - 1) * (n + 1)
        fs.append(tuple(base + j for j in reversed(range(n))))
    if skip:
        V, fs, UV = _used(V, fs, UV)
    return make_mesh(name, V, fs, UV)


def slope_sheet(name, F, prof, ext, i0, s_top, g, th, nseg, vn, close, uoff, wav, stag=0.0, crs=None, flat=False):
    """One roof plane as stepped course strips. Row A = foot of the butt on the batten plane, row B = top of
    the butt (lifted by the tile thickness); V of the texture is registered to the courses.
    crs: the courses that get a real step (default: every one). flat: no steps at all - one face from each of
    these course lines to the next, lying at half the tile thickness (the texture still shows every course)."""
    N = max(i0 + 1, int(ceil(s_top / g - 1e-6)))
    V, UV, fs = [], [], []
    info = {}
    base_l = 0.5 * th if flat else 0.0

    def row(i, lifted, fixed):
        idx = len(V)
        s0 = min(i * g, s_top)
        uL, uR = ext(prof.at_s(s0)[1])
        if uR - uL < 0.06:
            mid = 0.5 * (uL + uR)
            uL, uR = mid - 0.03, mid + 0.03
        e = stag * (0.35 + 0.65 * (i % 2) + 0.35 * vn(i * 3.7 + 1.3, 5.5))     # verge ends break joint course by course
        if close[0]:
            uL -= e
        if close[1]:
            uR += e
        ss_, ls_ = [], []
        for j in range(nseg + 1):
            u = uL + (uR - uL) * j / nseg
            s = s0 if fixed else min(max(s0 + wav * vn(u * 1.9 + 3.1, i * 0.77) + 1.6 * wav * vn(u * 0.55 + 11.0, i * 0.12) + 0.5 * wav * vn(i * 5.3, 9.5), 0.0), s_top)
            lift = th * (1.0 + 0.35 * vn(u * 2.7 + 40.0, i * 1.31)) if lifted else base_l
            d, z = prof.pt(s, lift)
            V.append(F @ Vector((u, d, z)))
            UV.append((u + uoff, s0 + (0.008 if lifted else 0.0)))
            ss_.append(s)
            ls_.append(lift)
        if lifted or i not in info:
            info[i] = (uL, uR, ss_, ls_)
        return idx

    def at(i, u):
        """(s, lift) of the butt line of course i at u, exactly as meshed."""
        if i not in info:                                   # a course without a step of its own (cheaper levels)
            lo, hi = [k for k in info if k < i], [k for k in info if k > i]
            if flat or not lo or not hi:
                return min(i * g, s_top), base_l
            return min(i * g, s_top), th * (min(hi) - i) / (min(hi) - max(lo))
        uL, uR, ss_, ls_ = info[i]
        t = min(max((u - uL) / (uR - uL), 0.0), 1.0) * nseg
        j = min(int(t), nseg - 1)
        f = t - j
        return ss_[j] + (ss_[j + 1] - ss_[j]) * f, ls_[j] + (ls_[j + 1] - ls_[j]) * f

    if flat:
        rw = [row(i, False, True) for i in sorted(set([i0] + [i for i in crs if i0 < i < N]))] + [row(N, False, True)]
        for r in range(len(rw) - 1):
            fs += [(rw[r] + j, rw[r] + j + 1, rw[r + 1] + j + 1, rw[r + 1] + j) for j in range(nseg)]
        return make_mesh(name, V, fs, UV), N, at
    rows = [(row(i, False, i == i0), row(i, True, i == i0)) for i in (range(i0, N) if crs is None else sorted(set([i0] + [i for i in crs if i0 < i < N])))]
    top = row(N, False, True)
    for r, (A, B) in enumerate(rows):
        nxt = rows[r + 1][0] if r + 1 < len(rows) else top
        for j in range(nseg):
            fs.append((A + j, A + j + 1, B + j + 1, B + j))
            fs.append((B + j, B + j + 1, nxt + j + 1, nxt + j))
        if close[0]:
            fs.append((A, B, nxt))
        if close[1]:
            fs.append((A + nseg, nxt + nseg, B + nseg))
    return make_mesh(name, V, fs, UV), N, at


def wbox(name, M, size, c, only=None):
    """Box given directly in world space (so it can be warped with the roof skin).
    only: the faces that exist (0 underside, 1 top, 2 / 3 the -y / +y side, 4 / 5 the ends)."""
    V = [M @ Vector((c[0] + i * size[0] / 2, c[1] + j * size[1] / 2, c[2] + k * size[2] / 2)) for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)]
    fs = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    if only is not None:
        V, fs, _ = _used(V, [fs[i] for i in only])
    return mesh_object(name, V, fs)


def tiles_mesh(name, items, sides=True):
    """Individual tiles / slates / shingles as ONE mesh. item = (M, w, ln, tk, U0, V0, dV): a wedge lying in the
    local XY plane (x across, y up the slope, butt of thickness tk at y = 0, tail feathering out at y = ln), with
    UVs taken from the covering texture (U0 at the tile centre, V0 at the butt, dV up the tile)."""
    V, UV, fs = [], [], []
    for (M, w, ln, tk, U0, V0, dV) in items:
        hw, b_ = w / 2, len(V)
        for (x, y, z, dv) in ((-hw, 0, 0, -0.006), (hw, 0, 0, -0.006), (-hw, 0, tk, 0.0), (hw, 0, tk, 0.0), (hw, ln, tk, dV), (-hw, ln, tk, dV)):
            V.append(M @ Vector((x, y, z)))
            UV.append((U0 + x, V0 + dv))
        fs += [(b_ + 2, b_ + 3, b_ + 4, b_ + 5), (b_, b_ + 1, b_ + 3, b_ + 2), (b_ + 1, b_ + 4, b_ + 3), (b_, b_ + 2, b_ + 5)][:4 if sides else 2]
    return make_mesh(name, V, fs, UV)


def rake_tiles(name, F, line, d0, d1, t_top, t_bot, step, gap, rng, merge=0):
    """A course of separate tiles laid along a raking polyline [(u, z)] (verge undercloak): outer edge at d0.
    merge 1: one length per straight of the line (edge and underside only); 2: nothing. The draws are the same."""
    L_ = [0.0]
    for k in range(1, len(line)):
        L_.append(L_[-1] + hypot(line[k][0] - line[k - 1][0], line[k][1] - line[k - 1][1]))
    us_, zs_ = [q[0] for q in line], [q[1] for q in line]
    n = max(1, int(round(L_[-1] / step)))
    V, fs = [], []
    if merge:
        for k in range(n):
            rng.jitter(0, 0.004), rng.jitter(0, 0.0015)
        if merge > 1:
            return None
        for k in range(len(line) - 1):
            b_ = len(V)
            for (u, z) in (line[k], line[k + 1]):
                V += [F @ Vector((u, d0, z + t_bot)), F @ Vector((u, d1, z + t_bot)), F @ Vector((u, d0, z + t_top))]
            fs += [(b_, b_ + 3, b_ + 5, b_ + 2), (b_, b_ + 1, b_ + 4, b_ + 3)]
        return mesh_object(name, V, fs)
    for k in range(n):
        ta, tb_ = L_[-1] * k / n + gap / 2, L_[-1] * (k + 1) / n - gap / 2
        ua, za, ub, zb = float(np.interp(ta, L_, us_)), float(np.interp(ta, L_, zs_)), float(np.interp(tb_, L_, us_)), float(np.interp(tb_, L_, zs_))
        dd, jz = d0 + rng.jitter(0, 0.004), rng.jitter(0, 0.0015)
        b_ = len(V)
        for (u, z) in ((ua, za), (ub, zb)):
            V += [F @ Vector((u, dd, z + t_bot + jz)), F @ Vector((u, d1, z + t_bot + jz)), F @ Vector((u, d1, z + t_top + jz)), F @ Vector((u, dd, z + t_top + jz))]
        fs += [(b_, b_ + 4, b_ + 7, b_ + 3), (b_, b_ + 1, b_ + 5, b_ + 4), (b_, b_ + 3, b_ + 2, b_ + 1), (b_ + 4, b_ + 5, b_ + 6, b_ + 7), (b_ + 3, b_ + 7, b_ + 6, b_ + 2)]
    return mesh_object(name, V, fs)


def tile_mtx(F, u, pa, pb, yaw=0.0, roll=0.0, sx=1.0, sy=1.0, slip=0.0):
    yd, yz = pb[0] - pa[0], pb[1] - pa[1]
    l = hypot(yd, yz)
    yd, yz = yd / l, yz / l
    M = Matrix(((1, 0, 0, u), (0, yd, -yz, pa[0]), (0, yz, yd, pa[1]), (0, 0, 0, 1)))
    J = Matrix.Translation((0, -slip, 0)) @ Matrix.Rotation(radians(yaw), 4, 'Z') @ Matrix.Rotation(radians(roll), 4, 'Y') @ Matrix.Diagonal((sx, sy, 1, 1))
    return F @ M @ J


def ridge_base(name, r0, r1, ln, tt, n=6, shell=False):
    """Half-round ridge tile along +X: head (x = 0) is the exposed, thicker end; the tail tucks under the next.
    shell: the outer face only (cheaper levels)."""
    V, fs = [], []
    if shell:
        for k in range(n + 1):
            c, s = cos(pi * k / n), sin(pi * k / n)
            V += [(0, r0 * c, r0 * s), (ln, r1 * c, r1 * s)]
        return mesh_object(name, V, [(2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1) for k in range(n)])
    for k in range(n + 1):
        th_ = pi * k / n
        c, s = cos(th_), sin(th_)
        V += [(0, r0 * c, r0 * s), (ln, r1 * c, r1 * s), (0, (r0 - tt) * c, (r0 - tt) * s), (ln, (r1 - tt) * c, (r1 - tt) * s)]
    for k in range(n):
        o0, o1, i0 = 4 * k, 4 * k + 1, 4 * k + 2
        p0, p1, j0 = 4 * k + 4, 4 * k + 5, 4 * k + 6
        fs.append((o0, p0, p1, o1))
        fs.append((o0, i0, j0, p0))
    fs.append((0, 1, 3, 2))
    e = 4 * n
    fs.append((e, e + 2, e + 3, e + 1))
    return mesh_object(name, V, fs)


def axis_mtx(origin, xdir, up):
    x = Vector(xdir).normalized()
    y = Vector(up).cross(x).normalized()
    z = x.cross(y)
    return Matrix(((x.x, y.x, z.x, origin[0]), (x.y, y.y, z.y, origin[1]), (x.z, y.z, z.z, origin[2]), (0, 0, 0, 1)))


class Path:
    def __init__(self, pts):
        self.p = np.array([tuple(q) for q in pts], dtype=float)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))])
        self.len = float(self.s[-1])

    def at(self, t):
        return Vector([float(np.interp(t, self.s, self.p[:, k])) for k in range(3)])


def thin(pts, tol=0.002):
    """Drop the points of a polyline [(u, z)] that lie within tol of the chord (Douglas-Peucker)."""
    if len(pts) < 3:
        return list(pts)
    (u0, z0), (u1, z1) = pts[0], pts[-1]
    l = max(hypot(u1 - u0, z1 - z0), 1e-9)
    dev = [abs((u1 - u0) * (z - z0) - (z1 - z0) * (u - u0)) / l for (u, z) in pts]
    k = max(range(1, len(pts) - 1), key=lambda i: dev[i])
    if dev[k] <= tol:
        return [pts[0], pts[-1]]
    return thin(pts[:k + 1], tol)[:-1] + thin(pts[k:], tol)


def split_runs(pts, tol=11.0):
    runs, start = [], 0
    for k in range(1, len(pts) - 1):
        a0 = degrees(atan2(pts[k][1] - pts[k - 1][1], pts[k][0] - pts[k - 1][0]))
        a1 = degrees(atan2(pts[k + 1][1] - pts[k][1], pts[k + 1][0] - pts[k][0]))
        if abs(a1 - a0) > tol:
            runs.append((start, k))
            start = k
    runs.append((start, len(pts) - 1))
    return runs


def build(p, rng):
    typ = p.type
    Lv = lod()
    TOLV = (0.002, 0.006, 0.02)[Lv]                     # how closely polylines follow the verge / eaves curves
    ridge_y = (p.ridge == "y")
    L, S = (p.depth, p.width) if ridge_y else (p.width, p.depth)
    if typ in ("hip", "half_hip") and L < S - 1e-6:      # a hipped ridge always runs the long way
        L, S = S, L
        ridge_y = not ridge_y
    a, b = S / 2.0, L / 2.0
    o = p.overhang
    C = COVER[p.covering]
    g, th, tw, tt, rr = C["g"], C["th"], C["tw"], C["tt"], C["r"]
    clay, slate_c, shingle = (p.covering == "roof_tile_clay"), (p.covering == "roof_slate"), (p.covering == "wood_shingle")
    seed = int(p.seed)
    vn = VNoise(seed)
    parapet = (p.gable == "parapet") and typ != "hip"
    cornice = (p.eaves == "cornice")
    lead_roll = (p.ridge_tile == "lead_roll")
    pw = 0.24                                           # parapet wall thickness
    ov = min(max(o * 0.7, 0.16), 0.4)                   # verge overhang at the gables
    xv = (b - pw) if parapet else (b + ov)              # where the covering stops at a gable
    pitch = float(p.pitch)

    # ------------------------------------------------------------------ materials
    COV = mat(p.covering, special=1.0, rough=0.82) if slate_c else mat(p.covering, special=1.0)
    RIDGE = mat("lead", rough=0.9, grime=1.6) if lead_roll else mat(*C["ridge"])
    OAK = mat("wood_oak", p.timber_color)
    TRIM = mat("wood_painted", p.trim_color)
    PLASTER = lambda: mat("plaster_rough", p.wall_color, scale=1.4)
    WALLM = {"timber": PLASTER, "plaster": PLASTER, "none": PLASTER, "brick": lambda: mat("brick_red"),
             "boards": lambda: mat("wood_planks", p.trim_color)}[p.gable_wall]()
    BAND = WALLM if p.gable_wall in ("brick",) else PLASTER()
    STONE = mat("stone_smooth", "#b9ad96")
    MORTAR = STONE if parapet else mat("mortar", "#8f8676")
    LIT = mat("window_lit", "#ffc37a", glow=1.4, curtain=0.9)
    LEAD = mat("lead", rough=0.9, grime=1.6)                 # old dark dull lead
    IRON = mat("iron_painted", "#20242a")
    GUT = {"zinc": lambda: mat("zinc"), "copper": lambda: mat("copper_patina"), "iron": lambda: IRON, "none": lambda: mat("zinc")}[p.gutter]()
    UCM = mat(*C["ridge"]) if clay else (LEAD if slate_c else OAK)          # verge undercloak

    # ------------------------------------------------------------------ section
    kick = p.kick
    if typ == "gambrel":
        p_lo, p_up = min(74.0, pitch + 20.0), max(20.0, pitch - 20.0)
        d_br = a * 0.36
    else:
        p_lo, p_up, d_br = pitch, pitch, 1e9
    pk = max(20.0, p_lo - 27.0 * kick)                # sprocket (bell-cast) pitch over the last ~0.6 m of the slope
    k0 = -o + 0.17
    k1 = k0 + 0.26 + 0.2 * kick

    def pitch_fn(d):
        v = pk + (p_lo - pk) * smoothstep(k0, k1, d)
        if typ == "gambrel":
            v = v + (p_up - v) * smoothstep(d_br - 0.04, d_br + 0.04, d)
        return v

    ZE = 0.27 if cornice else max(0.3, (T_SK + T_R) / cos(radians(pk)) + 0.04)
    prof = Profile(-o, ZE, a, pitch_fn)
    H = float(prof.z[-1])
    p_top = radians(pitch_fn(a - 0.05))
    gr = 0.5 * rr
    s_top = prof.s_of_d(a - gr)
    zcap = 1e9
    prof_end = prof
    if typ == "half_hip":
        z_h0 = ZE + (H - ZE) * 0.56
        dh0 = b - xv
        prof_end = Profile(dh0, z_h0, dh0 + (H - z_h0) / tan(radians(pitch)), lambda d: pitch)
        zcap = z_h0

    def Xe(z):
        if typ == "hip":
            return b - prof.d_of_z(z)
        if typ == "half_hip" and z > z_h0:
            return min(xv, b - prof_end.d_of_z(z))
        return xv

    def Ye(z):
        return a - prof.d_of_z(z)

    Xr = max(0.0, Xe(H - 1e-4))
    hipped = typ in ("hip", "half_hip")

    def frame(ox, oy, ang):
        return Matrix.Translation((ox, oy, 0)) @ Matrix.Rotation(radians(ang), 4, 'Z')

    F_f, F_b, F_px, F_nx = frame(0, -a, 0), frame(0, a, 180), frame(b, 0, 90), frame(-b, 0, -90)
    mains = [(F_f, "f"), (F_b, "b")]
    ends = [(F_px, "px"), (F_nx, "nx")]

    # ------------------------------------------------------------------ sag / undulation warp (skin only)
    Xw = b + ov
    z_w0 = prof.z_of_d(0.0)
    und = 0.012 + 0.22 * p.sag

    def warp(co):
        t = min(max((co.z - z_w0) / max(H - z_w0, 1e-3), 0.0), 1.0)
        if t <= 0.0:
            return co
        fx = cos(min(abs(co.x) / Xw, 1.0) * pi / 2) ** 2
        dz = -p.sag * fx * t ** 1.4
        dz -= und * sqrt(fx) * (0.5 + 0.5 * vn(co.x * 0.45 + 7.3, co.y * 0.13 + 2.1)) * sin(pi * t) ** 0.8
        return Vector((co.x, co.y, co.z + dz))

    skin = []
    n_crs = max(1.0, prof.len / g)
    nseg_of = lambda w: int(min(18, max(4, round(w / 0.62)), max(4, 600 // n_crs)))
    if Lv:
        nseg_0 = nseg_of
        nseg_of = lambda w: max(3, int(round(nseg_0(w) * (0.67 if Lv == 1 else 0.5))))

    def courses(pr, st):
        """Course lines that are meshed at the cheaper levels. lod 1: a real step at every other course.
        lod 2: rows of a flat sheet - close together where the section bends (bell-cast, gambrel break),
        then about every metre so the sheet follows the sag."""
        N = max(2, int(ceil(st / g - 1e-6)))
        if Lv == 1:
            return list(range(1, N, 2))
        brk = []
        if pr is prof:
            brk = [pr.s_of_d(d) for d in (k0, 0.5 * (k0 + k1), k1)]
            if typ == "gambrel":
                brk += [pr.s_of_d(d_br - 0.04), pr.s_of_d(d_br + 0.04)]
        brk = sorted(set([g] + [s_ for s_ in brk if g < s_ < st - 0.3]))
        out = []
        for sa_, sb_ in zip(brk, brk[1:] + [st]):
            m_ = max(1, int(ceil((sb_ - sa_) / 1.0)))
            out += [sa_ + (sb_ - sa_) * k / m_ for k in range(m_)]
        return sorted({min(N - 1, max(1, int(round(s_ / g)))) for s_ in out})

    # ------------------------------------------------------------------ covering: stepped course sheets
    r_t = rng.fork("tiles")
    sheets = []
    gab = (typ != "hip")
    for F, tag in mains:
        uoff = r_t.u(0, 7.0)
        sh, N, at = slope_sheet("cov_" + tag, F, prof, lambda z: (-Xe(z), Xe(z)), 1, s_top, g, th, nseg_of(2 * Xe(ZE)), vn,
                            (gab, gab), uoff, C["wav"] * g, stag=0.009 if (gab and not parapet) else 0.0,
                            crs=courses(prof, s_top) if Lv else None, flat=Lv >= 2)
        sheets.append((sh, F, prof, lambda z: (-Xe(z), Xe(z)), N, s_top, uoff, tag, at))
    if hipped:
        s_top_e = prof_end.len if typ == "half_hip" else s_top
        for F, tag in ends:
            uoff = r_t.u(0, 7.0)
            sh, N, at = slope_sheet("cov_" + tag, F, prof_end, lambda z: (-Ye(z), Ye(z)), 1, s_top_e, g, th, nseg_of(2 * Ye(prof_end.z[0])), vn,
                                (False, False), uoff, C["wav"] * g, crs=courses(prof_end, s_top_e) if Lv else None, flat=Lv >= 2)
            sheets.append((sh, F, prof_end, lambda z: (-Ye(z), Ye(z)), N, s_top_e, uoff, tag, at))
    for sh, *_ in sheets:
        assign(sh, COV)
        smooth(sh, 30)
        skin.append(sh)

    # ---- doubled eaves course: individual tiles carrying the covering texture (registered like the field) on a
    #      continuous under-eaves course; butts, lifts and joints jitter tile by tile
    def surf(s):                                        # lift of the stepped covering above the batten plane
        return th * (1.0 - (s / g - floor(s / g)))

    def tile_item(F, pr, u, s_a, l_a, s_b, l_b, w, tk, U0, V0, yaw=0.0, roll=0.0, dv=None):
        pa, pb = pr.pt(s_a, l_a), pr.pt(s_b, l_b)
        return (tile_mtx(F, u, pa, pb, yaw=yaw, roll=roll), w, hypot(pb[0] - pa[0], pb[1] - pa[1]), tk, U0, V0, (s_b - s_a) if dv is None else dv)

    eaves_items = []
    for sh, F, pr, ext, N, st, uoff, tag, at in sheets:
        uL, uR = ext(pr.z[0])
        n = max(1, int(round((uR - uL) / tw)))
        w = (uR - uL) / n
        i_s = len(eaves_items)
        for k in range(n):
            sl = r_t.u(0.0, 0.008) + (r_t.u(0.008, 0.02) if r_t.chance(0.08) else 0.0)
            uc = uL + (k + 0.5) * w
            eaves_items.append(tile_item(F, pr, uc, -sl, tt + r_t.u(0.0, 0.004), g + 0.03 - sl, -0.9 * tt, w - r_t.u(0.002, 0.004), tt,
                                         uc + uoff, 0.004, yaw=r_t.jitter(0, 0.8), roll=r_t.jitter(0, 1.5)))
        if Lv:                                              # the same course as a few wide lengths (the draws stay)
            del eaves_items[i_s:]
            grp = 3 if Lv == 1 else max(3, int(ceil(n / 4)))
            for k in range(0, n, grp):
                m_ = min(grp, n - k)
                uc = uL + (k + 0.5 * m_) * w
                eaves_items.append(tile_item(F, pr, uc, -0.004, tt + 0.002, g + 0.03 - 0.004, -0.9 * tt, m_ * w - 0.003, tt, uc + uoff, 0.004))
    et_ = tiles_mesh("eaves_tiles", eaves_items, sides=not Lv)
    assign(et_, COV)
    skin.append(et_)

    # ---- underside lining (sarking seen from inside / below; also keeps the shell closed for shadows)
    for sh, F, pr, ext, N, st, uoff, tag, at in sheets:
        nl = max(2, nseg_of(ext(pr.z[0])[1] * 2) // 2) if not Lv else nseg_of(ext(pr.z[0])[1] * 2)     # levels: the sheet's own columns, so the sag matches
        ss = sorted(set([float(x) for x in np.linspace(pr.s_of_d(min(0.0, float(pr.d[-1]) - 0.5)) if pr is prof else 0.0, st, max(3, int(st / 0.7) + 1) if not Lv else 4)]
                        + ([prof.s_of_d(d_br)] if (typ == "gambrel" and pr is prof) else [])))
        V, UV, fs = [], [], []
        for s_ in ss:
            d_, z_ = pr.pt(s_, -0.05)
            uL, uR = ext(pr.at_s(s_)[1])
            uL, uR = (uL + 0.03, uR - 0.03) if uR - uL > 0.1 else (0.5 * (uL + uR) - 0.02, 0.5 * (uL + uR) + 0.02)
            for j in range(nl + 1):
                u = uL + (uR - uL) * j / nl
                V.append(F @ Vector((u, d_, z_)))
                UV.append((u, s_))
        for i in range(len(ss) - 1):
            for j in range(nl):
                a_, b_ = i * (nl + 1) + j, (i + 1) * (nl + 1) + j
                fs.append((a_, b_, b_ + 1, a_ + 1))
        ln = make_mesh("lining_" + sh.name[4:], V, fs, UV)
        assign(ln, OAK)
        skin.append(ln)

    # ------------------------------------------------------------------ eaves kit per eaves side
    q_gc, R_g = o + 0.02, 0.062
    z_gc = ZE - 0.022
    ang0 = prof.ang_d(0.0)
    z_soff = prof.z_of_d(0.0) - 0.022 / cos(ang0)        # top of the sarking boards at the wall face
    z_rt = prof.z_of_d(0.0) - T_SK / cos(ang0)           # rafter top at the wall face
    z_rb = z_rt - T_R / cos(ang0)
    r_e = rng.fork("eaves")
    gutter_objs, oak_objs, band_objs, stone_objs = [], [], [], []
    sides = [(F_f, b, "f", True), (F_b, b, "b", True)]
    if typ == "hip":
        sides += [(F_px, a, "px", False), (F_nx, a, "nx", False)]
    mit = 1.0 if typ == "hip" else 0.0
    caps = (mit == 0.0, mit == 0.0)
    raft_m, brk_m, mod_m = [], [], []
    for F, hl, tag, is_main in sides:
        if mit:
            eL = eP = eS = eG = hl
        else:
            eL = xv                                       # covering
            eP = (b - pw + 0.01) if parapet else (xv - 0.05)   # wall plate (carries the bargeboard)
            eS = (b - pw) if parapet else (xv - 0.05)     # soffit boards
            eG = xv - 0.012                               # gutter
        hb = (b - pw + 0.01) if (parapet and not mit) else hl
        # under-eaves course (continuous) - gives the thick doubled eaves edge
        Pq = lambda s_, l_: (-prof.pt(s_, l_)[0], prof.pt(s_, l_)[1])
        sec = [Pq(-0.006, 0.0), Pq(0.45 * g, -0.3 * tt), Pq(0.45 * g, 0.25 * tt), Pq(-0.006, tt)]
        uc = eaves_run("undercourse_" + tag, F, sec, -eL, eL, mit, mit, caps=caps)
        assign(uc, COV)
        # sarking boards seen from below
        qs = np.linspace(-0.03, o - 0.02, 5 if not Lv else 4 - Lv)
        HID = dict(caps=(False, False), skip=(0, 2, 3)) if Lv else dict(caps=caps)        # levels: only the face that shows
        top = [(float(q), prof.z_of_d(-q) - 0.022 / cos(prof.ang_d(-q))) for q in qs]
        bot = [(float(q), prof.z_of_d(-q) - T_SK / cos(prof.ang_d(-q))) for q in qs]
        if not cornice:
            sf = eaves_run("soffit_" + tag, F, top + bot[::-1], -eS, eS, mit, mit, caps=caps,
                           skip=tuple(range(len(top) - 1)) + (2 * len(top) - 1,) if Lv else ())
            oak_objs.append(sf)
            # wall plate, beam filling between the rafters, wall band below the plate
            z_pl = z_rb + 0.03
            z_pb = max(0.0, z_pl - 0.15)
            pl = eaves_run("plate_" + tag, F, [(-0.12, z_pb), (0.035, z_pb), (0.035, z_pl), (-0.12, z_pl)], -eP, eP, mit, mit, nst=3 if not Lv else 2, caps=caps, skip=(2, 3) if Lv else ())
            oak_objs.append(pl)
            bf = eaves_run("beamfill_" + tag, F, [(-0.12, z_pl - 0.01), (0.004, z_pl - 0.01), (0.004, z_rt + 0.01), (-0.12, z_rt + 0.01)], -hb, hb, mit, mit, **HID)
            band_objs.append(bf)
            if z_pb > 0.012:
                wb = eaves_run("wallband_" + tag, F, [(-0.12, 0.0), (0.0, 0.0), (0.0, z_pb + 0.005), (-0.12, z_pb + 0.005)], -hb, hb, mit, mit, **HID)
                band_objs.append(wb)
            # rafter feet
            nr = max(2, int(round(2 * (hb - 0.12) / 0.4)))
            for k in range(nr + 1):
                u = -(hb - 0.12) + 2 * (hb - 0.12) * k / nr + r_e.jitter(0, 0.015)
                if is_main and p.gutter != "none" and abs(u - (hl - 0.55)) < 0.1:      # keep clear of the gutter outlet
                    u -= 0.13
                raft_m.append(F @ Matrix.Translation((u, 0, r_e.jitter(0, 0.004))) @ Matrix.Rotation(radians(r_e.jitter(0, 0.8)), 4, 'Z'))
            if mit and is_main:
                for sx in (-1, 1):
                    raft_m.append(F @ Matrix.Translation((sx * hl, 0, -0.004)) @ Matrix.Rotation(radians(45 * sx), 4, 'Z') @ Matrix.Diagonal((1.3, sqrt(2), 1.0, 1)))
        else:
            ztop_c = lambda q: prof.z_of_d(-q) - 0.03 / cos(prof.ang_d(-q))
            q_out = o - 0.085
            z_edge = ztop_c(q_out)
            z_s = max(0.1, z_edge - 0.115)
            sec = [(-0.12, 0.0), (0.022, 0.0), (0.022, 0.034), (0.034, 0.040), (0.040, 0.062), (0.060, z_s - 0.03), (0.082, z_s - 0.008), (0.088, z_s),
                   (q_out - 0.022, z_s), (q_out - 0.022, z_s - 0.012), (q_out, z_s - 0.012), (q_out, z_edge - 0.05), (q_out + 0.01, z_edge - 0.042),
                   (q_out + 0.026, z_edge - 0.018), (q_out + 0.032, z_edge - 0.004), (q_out + 0.032, z_edge)]
            if Lv:                                          # the moulding in a few strokes; no bed, top or back
                sec = [(-0.12, 0.0), (0.022, 0.0), (0.060, z_s - 0.03), (0.088, z_s), (q_out - 0.022, z_s), (q_out, z_s - 0.012), (q_out, z_edge - 0.05),
                       (q_out + 0.032, z_edge)] if Lv == 1 else [(-0.12, 0.0), (0.022, 0.0), (0.088, z_s), (q_out, z_s - 0.012), (q_out + 0.032, z_edge)]
            for q in (np.linspace(q_out - 0.02, -0.12, 4) if not Lv else [-0.12]):
                sec.append((float(q), ztop_c(float(q))))
            cn = eaves_run("cornice_" + tag, F, sec, -hb, hb, mit, mit, nst=max(2, int(2 * hb / 1.2)) if Lv < 2 else 2, caps=caps,
                           skip=(0, len(sec) - 2, len(sec) - 1) if Lv else ())
            stone_objs.append(cn)
            m_len = q_out - 0.05 - 0.088
            if m_len > 0.08:
                nm = max(2, int(round(2 * (hb - 0.1) / 0.42)))
                for k in range(nm + 1):
                    u = -(hb - 0.1) + 2 * (hb - 0.1) * k / nm
                    mod_m.append((F @ Matrix.Translation((u, -0.088, z_s)), m_len))
        # gutter
        if p.gutter != "none":
            stp = (30, 45, 60)[Lv]
            na = 180 // stp + 1
            sec = [(q_gc + R_g * cos(radians(t)), z_gc + R_g * sin(radians(t))) for t in range(180, 361, stp)]
            sec += [(q_gc + (R_g - 0.005) * cos(radians(t)), z_gc + (R_g - 0.005) * sin(radians(t))) for t in range(360, 179, -stp)]
            sec = [(q_gc - R_g - 0.006, z_gc + 0.004)] + sec[:na] + [(q_gc + R_g + 0.008, z_gc + 0.006), (q_gc + R_g + 0.006, z_gc - 0.004)] + sec[na:]
            if Lv >= 2:                                     # far: a solid half-round bar
                sec = sec[:na + 2]
            gt = eaves_run("gutter_" + tag, F, sec, -eG, eG, mit, mit, nst=2, caps=caps)
            gutter_objs.append(gt)
            nb = max(2, int(round(2 * eG / 1.15)))
            for k in range(nb + 1):
                u = -(eG - 0.12) + 2 * (eG - 0.12) * k / nb + r_e.jitter(0, 0.03)
                brk_m.append(F @ Matrix.Translation((u, 0, 0)))
            if is_main:
                zb = z_gc - R_g
                zl = max(0.012, zb - 0.13)
                ol = cylinder("outlet_" + tag, 0.034, zb - zl + 0.01, loc=(hl - 0.55, -q_gc, (zb + zl) / 2 + 0.005), verts=10)
                gutter_objs.append(setF(ol, F))
                if Lv < 2:
                    col = cylinder("outlet_collar_" + tag, 0.04, 0.02, loc=(hl - 0.55, -q_gc, zl + 0.03), verts=10)
                    gutter_objs.append(setF(col, F))
    if raft_m:
        # sprocketed rafter feet: run out past the tile edge, are notched to cradle the gutter and end in a
        # shaped (ovolo-cut) nose that shows below / beyond the gutter
        a_k = prof.ang_d(-o)
        bz = lambda q: prof.z_of_d(-q) if q <= o else ZE - (q - o) * tan(a_k)
        rtop = lambda q: bz(q) - T_SK / cos(prof.ang_d(-min(q, o)))
        depv = lambda q: T_R / cos(prof.ang_d(-min(q, o)))
        ntop = lambda q: max(rtop(q), ZE - 0.135)
        q_end = o + 0.24
        Rc = R_g + 0.003
        tq = [0.0, q_gc - Rc - 0.004]
        if p.gutter != "none":
            tq += [q_gc + Rc * cos(radians(t)) for t in (228, 300)]
        tq.append(q_end)
        sec = []
        for q in tq:
            z = ntop(q)
            if p.gutter != "none" and abs(q - q_gc) < Rc:
                z = min(z, z_gc - sqrt(max(Rc * Rc - (q - q_gc) ** 2, 0.0)))
            sec.append((q, z))
        nt = ntop(q_end)
        qn = q_end - 0.07
        zbn = max(rtop(qn) - depv(qn), 0.02)
        D_ = max(0.05, nt - zbn)
        sec += [(q_end + 0.003, nt - 0.34 * D_), (q_end - 0.028, nt - 0.7 * D_), (qn, nt - D_)]
        sec += [(q, max(rtop(q) - depv(q), 0.02)) for q in (0.0,)]
        if Lv:      # cheeks, shaped nose and underside only: the top lies under the soffit boards and the gutter, the back in the wall
            sec = [(0.0, ntop(0.0)), (q_gc, min(ntop(q_gc), z_gc - Rc) if p.gutter != "none" else ntop(q_gc)), (q_end, nt)]
            sec += ([(q_end + 0.003, nt - 0.34 * D_)] if Lv == 1 else []) + [(qn, nt - D_), (0.0, max(rtop(0.0) - depv(0.0), 0.02))]
            if Lv >= 2:                                     # far: a wedge - two cheeks and the underside
                sec = [sec[0], (q_end, nt - 0.3 * D_), sec[-1]]
        rb = eaves_run("rafter_base", Matrix.Identity(4), sec, -0.045, 0.045, skip=((), (0, 1, len(sec) - 1), (0, 2))[Lv])
        uv_grain(rb, axis=(0, -cos(a_k), -sin(a_k)))
        assign(rb, OAK)
        rf_ = instances_to_mesh("rafter_feet", rb, raft_m, keep_base=False, seed=seed + 2)
        assign(rf_, OAK)
        paint(rf_, wear=mask_mul(noise_mask(0.35, seed=seed + 60, lo=0.15, hi=0.7), 0.42), grime=mask_mul(noise_mask(0.5, seed=seed + 6), 0.3))
    if mod_m:
        msec = [(0, 0), (1.0, 0), (1.0, -0.028), (0.78, -0.036), (0.45, -0.066), (0.2, -0.078), (0, -0.09)]
        if Lv:
            msec = [msec[0], msec[1], msec[2], msec[4], msec[6]] if Lv == 1 else [msec[0], msec[1], msec[6]]       # far: a wedge
        mb = eaves_run("modillion", Matrix.Identity(4), msec, -0.045, 0.045, skip=(0, len(msec) - 1) if Lv else ())
        assign(mb, STONE)
        stone_objs.append(instances_to_mesh("modillions", mb, [M @ Matrix.Diagonal((1, ml, 1, 1)) for M, ml in mod_m], keep_base=False, seed=seed + 3))
    if brk_m and Lv < 2:
        pth = [(0, -(q_gc - R_g - 0.085), z_gc + 0.05), (0, -(q_gc - R_g - 0.005), z_gc + 0.012)]
        pth += [(0, -(q_gc + (R_g + 0.004) * cos(radians(t))), z_gc + (R_g + 0.004) * sin(radians(t))) for t in (range(195, 346, 50) if not Lv else (240, 300))]
        pth += [(0, -(q_gc + R_g + 0.012), z_gc + 0.012)]
        if Lv:                                              # the strap as one flat strip, facing out
            bk = mesh_object("bracket", [(sx * 0.012, y_, z_) for (_x, y_, z_) in pth for sx in (-1, 1)],
                             [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(len(pth) - 1)])
        else:
            bk = sweep("bracket", pth, [(-0.002, -0.012), (0.002, -0.012), (0.002, 0.012), (-0.002, 0.012)], smooth_path=False, up=(1, 0, 0))
        assign(bk, GUT)
        gutter_objs.append(instances_to_mesh("gutter_brackets", bk, brk_m, keep_base=False, seed=seed + 4))
    for o_ in gutter_objs:
        assign(o_, GUT)
        smooth(o_, 50)
        paint(o_, special=mask_mul(noise_mask(0.5, seed=seed + 5, lo=0.35, hi=0.75), 0.7), grime=0.35)
    for o_ in oak_objs:
        assign(o_, OAK)
        paint(o_, grime=mask_mul(noise_mask(0.7, seed=seed + 6), 0.5))
    for o_ in stone_objs:
        assign(o_, STONE)
        paint(o_, grime=mask_mul(noise_mask(0.6, seed=seed + 7, lo=0.3, hi=0.8), 0.6), special=mask_mul(facing((0, 0, 1)), 0.5))
    for o_ in band_objs:
        assign(o_, BAND)
        paint(o_, grime=0.3)

    # ------------------------------------------------------------------ ridge and hips
    r_r = rng.fork("ridge")
    lines = []                       # (Path, normal m, beta)
    if Xr > 0.2:
        lines.append((Path([(x, 0.0, H) for x in np.linspace(-Xr, Xr, max(2, int(2 * Xr / 0.45) + 1))]), Vector((0, 0, 1)), p_top, True))
    if hipped:
        zlo = prof_end.z[0]
        for sx in (-1, 1):
            for sy in (-1, 1):
                zs_ = np.linspace(zlo, H, 14)
                pts = [(sx * (b - prof_end.d_of_z(z)), sy * (a - prof.d_of_z(z)), float(z)) for z in zs_]
                zm = 0.5 * (zlo + H)
                a1, a2 = prof.ang_d(prof.d_of_z(zm)), prof_end.ang_d(prof_end.d_of_z(zm))
                n1, n2 = Vector((0, sy * sin(a1), cos(a1))), Vector((sx * sin(a2), 0, cos(a2)))
                lines.append((Path(pts), (n1 + n2).normalized(), acos(max(-1.0, min(1.0, n1.dot(n2)))) / 2, False))
    rt_m, cr_m = [], []
    LT, EXPO = 0.33, 0.295
    for path, m, beta, is_ridge in lines:
        c_off = -rr * tan(beta) + (0.014 if not lead_roll else 0.0)
        if lead_roll:
            # wood-cored lead roll: round roll on narrow wings dressed over the covering, lapped every ~1.5 m
            wl, rc = 0.115, 0.032
            pr_ = [(-wl * cos(beta), -wl * sin(beta) + 0.008), (-0.03, 0.01)]
            pr_ += [(rc * cos(radians(t)), 0.038 + rc * sin(radians(t))) for t in (range(215, -36, -50) if not Lv else ((215, 90, -35) if Lv == 1 else (90,)))]
            pr_ += [(0.03, 0.01), (wl * cos(beta), -wl * sin(beta) + 0.008), (wl * cos(beta), -wl * sin(beta) + 0.003), (0, 0.0), (-wl * cos(beta), -wl * sin(beta) + 0.003)]
            if Lv:
                pr_ = pr_[:-3]                              # the underside lies in the covering
            if Lv >= 2:
                pr_ = [pr_[0], pr_[2], pr_[4]]              # far: a tent over the hip
            pr_.reverse()
            pts = [path.at(t) + m * (th + 0.004) for t in np.linspace(0, path.len, max(2, int(path.len / (0.65 if not Lv else 1.0)) + 1))]
            with lod_as(0):                                 # every point of the path stays: the roll sags with the ridge (warp below)
                ld = sweep("lead_roll", pts, pr_, smooth_path=False, cap=True, up=tuple(m))
            lds = [ld]
            n_lap = max(1, int(round(path.len / 1.5)))
            lap_p = [(x_ * 1.09, y_ * 1.07 + 0.004) for (x_, y_) in pr_]
            for k in range(1, (n_lap + (0 if is_ridge else 1)) if Lv < 2 else 1):
                t0 = path.len * k / n_lap - (0.0 if k < n_lap else 0.1)
                ends_ = [path.at(max(0.0, t0 - 0.045)) + m * (th + 0.004), path.at(min(path.len, t0 + 0.045)) + m * (th + 0.004)]
                lds.append(sweep("lead_lap", ends_, lap_p, smooth_path=False, cap=True, up=tuple(m)))
            for ld in lds:
                assign(ld, RIDGE)
                smooth(ld, 50)
                paint(ld, grime=mask_mul(noise_mask(0.4, seed=seed + 8, lo=0.2, hi=0.8), 0.75))
                skin.append(ld)
            continue
        n_t = max(1, int(ceil((path.len - 0.03) / EXPO)))
        ex = (path.len - 0.03) / n_t
        i_s = len(rt_m)
        for k in range(n_t):
            t0 = k * ex
            P0, P1 = path.at(t0), path.at(min(path.len, t0 + ex))
            org = P0 + m * (c_off + r_r.jitter(0, 0.004))
            M = axis_mtx(org, P1 - P0, m) @ Matrix.Rotation(radians(r_r.jitter(0, 2.2)), 4, 'X') @ Matrix.Rotation(radians(r_r.jitter(0, 1.0)), 4, 'Z') \
                @ Matrix.Diagonal((ex / EXPO * r_r.vary(1.0, 0.02), 1, 1, 1))
            rt_m.append(M)
            if p.ridge_tile == "crested" and is_ridge:
                cr_m.append(M @ Matrix.Translation((0, 0.008, rr - 0.008)) @ Matrix.Rotation(radians(90), 4, 'X'))
        if Lv >= 2:                                         # far: one tile of double length for every two
            two = rt_m[i_s:]
            del rt_m[i_s:]
            rt_m += [two[k] @ Matrix.Diagonal(((EXPO + LT) / LT, 1, 1, 1)) if k + 1 < len(two) else two[k] for k in range(0, len(two), 2)]
        pm = [((rr - 0.028) * cos(radians(t)), (rr - 0.028) * sin(radians(t))) for t in range(0, 181, 36 if not Lv else 90)]
        pm += [(-(rr + 0.01), -0.004), (-(rr + 0.012), -0.04), ((rr + 0.012), -0.04), ((rr + 0.01), -0.004)][::1 if Lv < 2 else 3]
        pts = [path.at(t) + m * c_off for t in np.linspace(0.012, path.len - 0.012, max(2, int(path.len / (0.6 if not Lv else 1.0)) + 1))]
        with lod_as(0):                                     # as the roll: the bed sags with the ridge
            mo = sweep("ridge_mortar", pts, pm, smooth_path=False, cap=True, up=tuple(m))
        assign(mo, MORTAR)
        paint(mo, grime=0.5)
        lod_drop(mo, above=1)                               # far: the ridge tiles sit on the covering
        skin.append(mo)
    if rt_m:
        rbm = ridge_base("ridge_base", rr, rr - 0.02, LT, 0.016) if not Lv else ridge_base("ridge_base", rr, rr - 0.02, LT, 0.016, n=6 - 2 * Lv, shell=True)
        assign(rbm, RIDGE)
        smooth(rbm, 45 if not Lv else 50)
        rt = instances_to_mesh("ridge_tiles", rbm, rt_m, keep_base=False, seed=seed + 9)
        skin.append(rt)
        if cr_m:
            cpf = [(0.03, 0), (0.29, 0), (0.265, 0.045), (0.215, 0.06), (0.16, 0.13), (0.105, 0.06), (0.055, 0.045)]
            cb = extrude_profile("crest_base", cpf if not Lv else ([cpf[0], cpf[1], cpf[3], cpf[4], cpf[5]] if Lv == 1 else [cpf[0], cpf[1], cpf[4]]), 0.016)
            if Lv >= 2:                                     # far: the two faces of a pointed card
                lod_faces(cb, keep=[(0, 0, 1), (0, 0, -1)])
            assign(cb, RIDGE)
            skin.append(instances_to_mesh("ridge_crests", cb, cr_m, keep_base=False, seed=seed + 10))

    # ------------------------------------------------------------------ gables
    r_g = rng.fork("gable")
    lights = []
    win_socks = []

    def zs_fn(u):
        return prof.z_of_d(a - abs(u))

    def ztop(u):
        return min(zs_fn(u), zcap)

    def pitch_u(u):                         # local pitch (rad) of the capped verge line
        if zs_fn(u) > zcap:
            return 0.0
        return prof.ang_d(a - abs(u))

    def u_samples(u_max, step=0.7):
        sp_ = {0.0}
        for d in list(np.linspace(-o, k1, 7)) + ([d_br - 0.045, d_br, d_br + 0.045] if typ == "gambrel" else []) + [0.0]:
            sp_.update([a - float(d), -(a - float(d))])
        if typ == "half_hip":
            uh = a - prof.d_of_z(zcap)
            sp_.update([-uh, uh])
        us = set(sp_) | {-u_max, u_max}
        n = max(2, int(ceil(2 * u_max / step)))
        for k in range(n + 1):
            us.add(-u_max + 2 * u_max * k / n)
        us = sorted(u for u in us if abs(u) <= u_max + 1e-6)
        out = [us[0]]
        for u in us[1:]:
            if u - out[-1] > 0.035:
                out.append(u)
            elif u in sp_ and out[-1] not in sp_ and len(out) > 1:
                out[-1] = u
            elif u == us[-1]:
                out[-1] = u
        return out

    def lean(us_, keep_=()):
        """The verge line with fewer points (cheaper levels); the u in keep_ (+-) stay."""
        ks = {q[0] for q in thin([(u, ztop(u)) for u in us_], TOLV)}
        return [u for u in us_ if u in ks or any(abs(abs(u) - k_) < 1e-6 for k_ in keep_)]

    if typ != "hip":
        def notch_us(us_, hw_):
            e_ = 0.0006
            return sorted([u for u in us_ if abs(abs(u) - hw_) > 0.03] + [-hw_ - e_, -hw_ + e_, hw_ - e_, hw_ + e_])

        for F, tag in ends:
            wall_objs = []
            out_ = F.to_3x3() @ Vector((0, -1, 0))          # looking out of this gable
            # ---- attic window opening (cut through the wall so the casement sits back in a real reveal)
            zt = lambda u: ztop(u) - 0.05
            ww, wh, fw = 0.52, 0.72, 0.045
            z_sill = max(0.42, 0.26 * zt(0.0))
            wh = min(wh, zt(ww / 2 + 0.14) - 0.34 - z_sill)
            has_win = p.attic_window != "none" and wh > 0.36 and not (p.gable_wall == "none" and not parapet)
            d_face = -0.03 if parapet else 0.0
            hw_o, z_head = ww / 2 + fw, z_sill + wh + fw
            # ---- tympanum wall
            if parapet:
                ko = min(o, 0.22)
                us = u_samples(a + ko) if not Lv else lean(u_samples(a + ko), (a,))
                hp = 0.27
                topw = [(u, ztop(u) + hp) for u in us]
                botw = [(u, 0.0 if abs(u) <= a else min(ztop(u) + hp - 0.24, (abs(u) - a) * 1.6 + 0.02)) for u in us]
                if has_win:
                    us_w = notch_us(us, hw_o)
                    botf = dict(botw)
                    walls = [ribbon("parapet_" + tag, F, [(u, ztop(u) + hp) for u in us_w],
                                    [(u, z_head if abs(u) < hw_o else (0.0 if abs(u) <= a else botf.get(u, 0.0))) for u in us_w], -0.03, pw, ("fktbe", "fkbe", "f")[Lv]),
                             ribbon("parapet_apron_" + tag, F, [(-hw_o, z_sill), (hw_o, z_sill)], [(-hw_o, 0.0), (hw_o, 0.0)], -0.03, pw, ("fktbe", "fkt", "f")[Lv])]
                else:
                    walls = [ribbon("parapet_" + tag, F, topw, botw, -0.03, pw, ("fktbe", "fke", "f")[Lv])]
                if Lv >= 2:                                 # far: of the inner face only the upstand above the covering
                    walls.append(ribbon("parapet_up_" + tag, F, topw, [(u, ztop(u)) for u in us], -0.03, pw, "k"))
                for wl_ in walls:
                    densify(wl_, 0.75)
                    assign(wl_, WALLM)
                    uv_box(wl_, offset=None)
                    paint(wl_, grime=mask_mul(noise_mask(0.9, seed=seed + 20), 0.5), special=mask_mul(noise_mask(0.7, seed=seed + 21, lo=0.45, hi=0.8), 0.5))
                # coping stones, kneelers and apex stone
                cm = []
                for (i0, i1) in split_runs(topw):
                    (ua, za), (ub, zb) = topw[i0], topw[i1]
                    Lr = hypot(ub - ua, zb - za)
                    phi = degrees(atan2(zb - za, ub - ua))
                    nst_ = max(1, int(ceil(Lr / 0.56)))
                    i_s = len(cm)
                    for k in range(nst_):
                        t = (k + 0.5) / nst_
                        uu, zz = ua + (ub - ua) * t, za + (zb - za) * t
                        cm.append(F @ Matrix.Translation((uu, (pw - 0.03) / 2, zz + 0.03 / max(0.3, cos(radians(phi))))) @ Matrix.Rotation(radians(-phi + r_g.jitter(0, 0.5)), 4, 'Y')
                                  @ Matrix.Diagonal((Lr / nst_ - 0.007, pw + 0.03 + 0.09 + r_g.jitter(0, 0.006), r_g.vary(0.075, 0.06), 1)))
                    if Lv >= 2:                             # far: one stone per straight of the rake (the draws stay)
                        del cm[i_s:]
                        cm.append(F @ Matrix.Translation(((ua + ub) / 2, (pw - 0.03) / 2, (za + zb) / 2 + 0.03 / max(0.3, cos(radians(phi))))) @ Matrix.Rotation(radians(-phi), 4, 'Y')
                                  @ Matrix.Diagonal((Lr - 0.007, pw + 0.03 + 0.09, 0.075, 1)))
                cb_ = box("coping_base", (1, 1, 1), bevel=0.0)
                if Lv:
                    lod_faces(cb_, drop=(0, 0, -1))         # plain stones on their bed
                else:
                    bevel(cb_, 0.06, segments=1)
                st = [instances_to_mesh("coping_" + tag, cb_, cm, keep_base=False, seed=seed + 22)]
                for sx in (-1, 1):
                    ue = sx * (a + ko)
                    zk = ztop(ue) + hp
                    st.append(setF(box("kneeler", (0.44, pw + 0.15, 0.21), loc=(ue - sx * 0.17, (pw - 0.03) / 2, zk - 0.035), bevel=0.012, segments=1), F))
                    st.append(lod_faces(setF(box("corbel", (0.2, pw + 0.07, 0.12), loc=(sx * (a + ko * 0.45), (pw - 0.03) / 2, zk - 0.2), bevel=0.012, segments=1), F),
                                        drop=(0, 0, 1), above=1))         # far: its top lies under the kneeler
                if typ != "half_hip":
                    st.append(setF(box("apex_stone", (0.3, pw + 0.15, 0.3), loc=(0, (pw - 0.03) / 2, ztop(0) + hp - 0.05), rot=(0, 45, 0), bevel=0.012, segments=1), F))
                for s_ in st:
                    assign(s_, STONE)
                    paint(s_, grime=mask_mul(noise_mask(0.5, seed=seed + 23), 0.6), special=mask_mul(facing((0, 0, 1)), noise_mask(0.4, seed=seed + 24), 0.8 * p.moss + 0.2))
                # lead flashing against the parapet
                usf = [u for u in us if abs(u) <= a + min(o, ko) - 0.02]
                fl = ribbon("flash_up_" + tag, F, [(u, ztop(u) + 0.13) for u in usf], [(u, ztop(u) + 0.0) for u in usf], pw, pw + 0.005, "k" if Lv else "fktbe")
                fl2 = ribbon("flash_cover_" + tag, F, [(u, ztop(u) + 0.036) for u in usf], [(u, ztop(u) + 0.03) for u in usf], pw, pw + 0.11, "t" if Lv else "fktbe")
                for f_ in (fl, fl2):
                    assign(f_, LEAD)
                    paint(f_, grime=0.6)
                    skin.append(f_)
            else:
                us = u_samples(a) if not Lv else lean(u_samples(a))
                if p.gable_wall != "none":
                    topf = lambda u: max(0.03, ztop(u) - 0.05)
                    if has_win:
                        us_w = notch_us(us, hw_o)
                        walls = [ribbon("gable_wall_" + tag, F, [(u, topf(u)) for u in us_w], [(u, z_head if abs(u) < hw_o else 0.0) for u in us_w], 0.0, 0.2, "fb" if Lv else "fktbe"),
                                 ribbon("gable_apron_" + tag, F, [(-hw_o, z_sill), (hw_o, z_sill)], [(-hw_o, 0.0), (hw_o, 0.0)], 0.0, 0.2, "ft" if Lv else "fktbe")]
                    else:
                        walls = [ribbon("gable_wall_" + tag, F, [(u, topf(u)) for u in us], [(u, 0.0) for u in us], 0.0, 0.2, "f" if Lv else "fktbe")]
                    for wl_ in walls:
                        densify(wl_, 0.75)
                        assign(wl_, WALLM)
                        uv_box(wl_, rot90=(p.gable_wall == "boards"), offset=None)
                        paint(wl_, grime=mask_mul(noise_mask(0.8, seed=seed + 25, lo=0.3, hi=0.8), 0.32),
                              special=mask_mul(height_ramp(0.0, 0.9), noise_mask(0.5, seed=seed + 26), 0.5))
                # ---- verge: soffit boards, a continuous undercloak course projecting past the bargeboard, the mortar
                #      bed / fillet the verge tiles sit in, moulded bargeboards
                usv = [q[0] for q in thin([(u, ztop(u)) for u in u_samples(a + o - 0.03)], TOLV)]
                sfv = ribbon("verge_soffit_" + tag, F, [(u, ztop(u) - 0.026) for u in usv], [(u, ztop(u) - 0.046) for u in usv], -(ov - 0.075), 0.0, "b" if Lv else "fktbe")
                assign(sfv, OAK)
                vline = [(u, ztop(u)) for u in usv]
                for (i0, i1) in split_runs(vline):
                    uc_ = rake_tiles("undercloak_" + tag, F, vline[i0:i1 + 1], -ov - 0.004, -ov + 0.12, -0.011, -0.024,
                                     0.268 if clay else 1.4, 0.004, r_g, merge=Lv)
                    if uc_ is not None:
                        assign(uc_, UCM)
                        paint(uc_, grime=0.5, special=0.3 * p.moss)
                if Lv < 2:
                    mb_ = ribbon("verge_bed_" + tag, F, [(u, ztop(u) - 0.0015) for u in usv], [(u, ztop(u) - 0.011) for u in usv], -ov + 0.002, -ov + 0.1, "f" if Lv else "fktbe")
                    assign(mb_, MORTAR)
                    paint(mb_, grime=0.5)
                scal = p.gable == "scalloped" and Lv < 2    # far: a plain board of the scallops' mean depth
                usb = u_samples(a + o - 0.045, step=0.07 if scal else 0.7)
                line = [(u, ztop(u) - 0.024) for u in usb]
                if not scal:
                    line = thin(line, TOLV)
                dep_b = 0.21 if (scal or p.gable != "scalloped") else 0.185
                for (i0, i1) in split_runs(line):
                    seg = line[i0:i1 + 1]
                    du, dz_ = seg[-1][0] - seg[0][0], seg[-1][1] - seg[0][1]
                    cs = max(0.3, cos(atan2(abs(dz_), abs(du))))
                    Lr = hypot(du, dz_)
                    if scal:
                        acc, bot = 0.0, []
                        for k, (u, z) in enumerate(seg):
                            if k:
                                acc += hypot(u - seg[k - 1][0], z - seg[k - 1][1])
                            bot.append((u, max(0.004, z - (dep_b - 0.05 * abs(sin(pi * acc / 0.28))) / cs)))
                    else:
                        bot = [(u, max(0.004, z - dep_b / cs)) for (u, z) in seg]
                    bd = ribbon("barge_" + tag, F, seg, bot, -ov + 0.045, -ov + 0.077, ("fktbe", "fkbe", "fk")[Lv])
                    ax = F.to_3x3() @ Vector((du, 0, dz_)).normalized()
                    uv_grain(bd, axis=tuple(ax))
                    assign(bd, TRIM)
                    wall_objs.append(bd)
                    coarse = [seg[0]] + [q for q in seg[1:-1][::3]] + [seg[-1]] if scal else seg
                    if Lv < 2:
                        cp = ribbon("barge_cap_" + tag, F, coarse, [(u, z - 0.075 / cs) for (u, z) in coarse], -ov + 0.027, -ov + 0.045, "fb" if Lv else "fktbe")
                        uv_grain(cp, axis=tuple(ax))
                        assign(cp, TRIM)
                        wall_objs.append(cp)
                    if Lr > 2.6 and abs(dz_) > 0.3 and Lv < 2:          # purlin end carrying the bargeboard
                        um, zm_ = 0.5 * (seg[0][0] + seg[-1][0]), 0.5 * (seg[0][1] + seg[-1][1])
                        pu = box("purlin_end", (0.11, ov - 0.085, 0.15), loc=(um, -(ov - 0.085) / 2, zm_ - 0.022 - 0.1 / cs),
                                 rot=(0, -degrees(atan2(dz_, du)), 0), bevel=0.006 if not Lv else 0.0, segments=1)
                        uv_grain(pu, axis='Y')
                        assign(setF(pu, F), OAK)
                for bd in wall_objs:
                    paint(bd, wear=mask_mul(noise_mask(0.25, seed=seed + 27, lo=0.45, hi=0.8), 0.7), special=mask_mul(noise_mask(0.8, seed=seed + 28), 0.6), grime=0.2)
                if typ != "half_hip":                        # hip-knob / pendant post where the bargeboards meet
                    zt0 = zs_fn(0.0)
                    po = box("barge_post", (0.085, 0.085, 0.56), loc=(0, -ov + 0.05, zt0 - 0.075 - 0.28), bevel=0.008 if not Lv else 0.0, segments=1)
                    uv_grain(po, axis='Z')
                    ppf = [(0.0, -0.16), (0.012, -0.15), (0.03, -0.12), (0.038, -0.09), (0.03, -0.06), (0.018, -0.045), (0.03, -0.03), (0.046, -0.015), (0.046, 0.0)]
                    if Lv:
                        ppf = [ppf[0], ppf[2], ppf[3], ppf[5], ppf[7], ppf[8]] if Lv == 1 else [ppf[0], ppf[3], ppf[5], ppf[8]]
                    with lod_as(0):
                        pd = lathe("barge_pendant", ppf, segments=8 - 2 * Lv, loc=(0, -ov + 0.05, zt0 - 0.075 - 0.56))
                    for q_ in (po, pd):
                        assign(setF(q_, F), TRIM)
                        paint(q_, wear=mask_mul(noise_mask(0.2, seed=seed + 29, lo=0.5, hi=0.8), 0.6))

            # ---- attic casement set back in the reveal, lit from INSIDE the attic (only panes, reveals and sill glow)
            if has_win:
                zc = z_sill + wh / 2
                d_g, d_fr = d_face + 0.078, d_face + 0.056
                fr = []
                glass = setF(box("attic_glass_" + tag, (ww + 0.02, 0.008, wh + 0.02), loc=(0, d_g, zc)), F)
                assign(glass, LIT if p.attic_window == "lit" else mat("window_dark"))
                # the pane is one card at every level (a thin glowing box is two sheets that fight each other).
                # CLEARANCE RULE: nothing lies parallel within 4 mm of it - mullion and bars run through it, 6 mm behind
                lod_faces(glass, keep=out_, above=-1)
                for sx in (-1, 1):
                    fr.append(box("win_jamb", (fw, 0.055, wh + fw), loc=(sx * (ww / 2 + fw / 2), d_fr, zc + fw / 2)))
                fr.append(box("win_head", (ww, 0.055, fw), loc=(0, d_fr, z_sill + wh + fw / 2)))
                fr.append(box("win_rail", (ww, 0.05, 0.036), loc=(0, d_fr, z_sill + 0.018)))
                fr.append(box("win_mullion", (0.034, 0.053, wh), loc=(0, d_fr - 0.0025, zc)))
                for sx in (-1, 1):
                    for kz in (1, 2):
                        fr.append(box("win_bar", (ww / 2 - 0.017, 0.028, 0.016), loc=(sx * (ww / 4 + 0.008), d_g - 0.010, z_sill + wh * kz / 3)))
                for f_ in fr:
                    assign(setF(f_, F), TRIM)
                    if Lv:                                  # glazing bars as cards on the pane; frame without backs (far: fronts only)
                        lod_faces(f_, keep=out_) if (Lv >= 2 or f_.name.startswith("win_bar")) else lod_faces(f_, drop=-out_)
                    paint(f_, wear=mask_mul(noise_mask(0.15, seed=seed + 30, lo=0.5, hi=0.8), 0.6))
                stone_sill = p.gable_wall in ("brick",) or parapet
                sill = box("win_sill", (2 * hw_o + 0.14, 0.15, 0.05), loc=(0, d_face - 0.02, z_sill - 0.025), bevel=0.008 if not Lv else 0.0, segments=1)
                if not stone_sill:
                    uv_grain(sill, axis='X')
                assign(setF(sill, F), STONE if stone_sill else OAK)
                paint(sill, grime=0.3, wear=0.25)
                if stone_sill:
                    lin = box("win_lintel", (2 * hw_o + 0.24, 0.09, 0.13), loc=(0, d_face + 0.02, z_head + 0.065), bevel=0.008 if not Lv else 0.0, segments=1)
                    assign(setF(lin, F), STONE)
                    paint(lin, grime=0.35)
                if p.attic_window == "lit":
                    lights.append((tag, F @ Vector((0, d_face + 0.42, zc + 0.04))))
            else:
                win_socks.append((tag, F @ Vector((0, d_face, max(0.6, 0.4 * zt(0.0))))))
            if p.gable_wall == "timber" and not parapet:
                tim = []
                pr_bot = lambda u: ztop(u) - 0.06 - 0.17 / max(0.3, cos(pitch_u(u)))

                def beam(name, size, loc, rot=None, ax='X', proud=0.04, bev=0.0):
                    bx = box(name, (size[0], proud + 0.03, size[1]) if ax == 'X' else (size[0], proud + 0.03, size[1]), loc=(loc[0], 0.015 - proud / 2 - 0.0, loc[1]), rot=rot, bevel=bev if not Lv else 0.0, segments=1)
                    uv_grain(bx, axis=ax)
                    tim.append(setF(bx, F))
                    if Lv:                                  # the timbers lie in the wall: no backs; far: their faces only
                        lod_faces(bx, keep=out_) if Lv >= 2 else lod_faces(bx, drop=-out_)
                    return bx

                beam("tie_beam", (2 * a + 0.07, 0.2), (0, 0.1), proud=0.05, bev=0.008)
                # principals follow the verge line
                usp = u_samples(a - 0.02)
                pl_line = [(u, ztop(u) - 0.06) for u in usp]
                for (i0, i1) in split_runs(pl_line):
                    seg = pl_line[i0:i1 + 1] if not Lv else thin(pl_line[i0:i1 + 1], TOLV)
                    du, dz_ = seg[-1][0] - seg[0][0], seg[-1][1] - seg[0][1]
                    cs = max(0.3, cos(atan2(abs(dz_), abs(du))))
                    pr2 = ribbon("principal_" + tag, F, seg, [(u, max(0.0, z - 0.17 / cs)) for (u, z) in seg], -0.042, 0.02, ("fktbe", "fb", "f")[Lv])
                    uv_grain(pr2, axis=tuple(F.to_3x3() @ Vector((du, 0, dz_)).normalized()))
                    tim.append(pr2)
                sw = 0.115
                u_w = ww / 2 + 0.045 + sw / 2 if has_win else 0.0
                z_head = z_sill + wh + 0.045 if has_win else 0.0
                studs = [u_w, -u_w] if has_win else [0.0]
                k = 1
                sp = max(0.55, min(0.8, (a - u_w) / max(1, round((a - u_w) / 0.72))))
                while u_w + k * sp < a - 0.3:
                    studs += [u_w + k * sp, -(u_w + k * sp)]
                    k += 1
                z_col = None
                if has_win and pr_bot(0.0) - z_head > 0.75:
                    z_col = z_head + 0.09
                    hw_c = a - prof.d_of_z(z_col + 0.3) if z_col + 0.3 < min(H, zcap) else u_w + 0.2
                    hw_c = max(u_w + 0.1, min(hw_c, a - 0.2))
                    beam("collar", (2 * hw_c, 0.15), (0, z_col), proud=0.036)
                    if pr_bot(0.0) - (z_col + 0.075) > 0.3:
                        beam("king_post", (0.12, pr_bot(0.0) + 0.04 - (z_col + 0.07)), (0, 0.5 * (pr_bot(0.0) + 0.04 + z_col + 0.07)), ax='Z', proud=0.03)
                elif has_win:
                    beam("win_head_beam", (2 * u_w + sw, 0.1), (0, z_head + 0.05), proud=0.034)
                if has_win:
                    beam("win_sill_beam", (2 * u_w + sw, 0.09), (0, z_sill - 0.05 - 0.045), proud=0.034)
                for u in studs:
                    ztp = pr_bot(u) + 0.04
                    if z_col is not None and abs(u) < hw_c and abs(u) > u_w + 0.01:
                        pass
                    if ztp - 0.2 > 0.22:
                        beam("stud", (sw * r_g.vary(1.0, 0.08), ztp - 0.19), (u + r_g.jitter(0, 0.008), 0.19 + (ztp - 0.19) / 2), ax='Z', proud=0.03)
                # raking braces in the outer bays
                if a > 1.9 and typ != "gambrel":
                    for sx in (-1, 1):
                        u0_, u1_ = sx * (a - 0.38), sx * max(u_w + sw, a * 0.42)
                        z1_ = min(pr_bot(u1_) - 0.1, 0.2 + abs(u1_ - u0_) * 1.25)
                        Lb = hypot(u1_ - u0_, z1_ - 0.2)
                        beam("brace", (Lb, 0.1), ((u0_ + u1_) / 2, (0.2 + z1_) / 2), rot=(0, -degrees(atan2(z1_ - 0.2, u1_ - u0_)), 0), proud=0.024)
                for t_ in tim:
                    assign(t_, OAK)
                    paint(t_, grime=mask_mul(noise_mask(0.5, seed=seed + 31), 0.5), wear=mask_mul(noise_mask(0.2, seed=seed + 32, lo=0.55, hi=0.85), 0.4))

    # ------------------------------------------------------------------ rooflight and vent tiles on the slopes
    def wz(v):
        return warp(Vector(v))

    def slope_mtx(F, u, s0, s1, lift=0.0):
        M = tile_mtx(F, u, prof.pt(s0, lift), prof.pt(s1, lift))
        c = M.to_translation()
        return Matrix.Translation((0, 0, wz(c).z - c.z)) @ M

    r_s = rng.fork("slope")
    rl_light = None
    rl_on = False
    rl_w, rl_l = 0.5, round(0.7 / g) * g
    s_rl = floor(0.5 * s_top / g) * g
    u_rl = 0.42 * Xe(prof.at_s(s_rl)[1])
    if p.rooflight != "none" and s_top > 2.4 and Xe(prof.at_s(s_rl + rl_l)[1]) > abs(u_rl) + 0.7:
        # cast-iron rooflight on a lead-clad kerb: apron dressed over the course below, back gutter tucked under
        # the course above, stepped side flashings course by course, lid propped ajar on its stay
        rl_on = True
        M = slope_mtx(F_f, u_rl, s_rl, s_rl + rl_l)
        hk, kt = th + 0.105, 0.045
        if Lv >= 2:                                         # far: the kerb as one block, no flashings
            lead_p = [box("rl_kerb", (rl_w, rl_l, hk), loc=(0, rl_l / 2, hk / 2))]
        else:
            lead_p = [box("rl_kerb", (kt, rl_l, hk), loc=(sx * (rl_w / 2 - kt / 2), rl_l / 2, hk / 2)) for sx in (-1, 1)]
            lead_p += [box("rl_kerb", (rl_w - 2 * kt, kt, hk), loc=(0, y_, hk / 2)) for y_ in (kt / 2, rl_l - kt / 2)]
        lead_p = [setF(f_, M) for f_ in lead_p]
        zM = M.to_3x3() @ Vector((0, 0, 1))
        lod_faces(lead_p, drop=[zM, -zM], above=1)
        la = g + 0.05
        sl_a = surf(s_rl - la)
        if Lv < 2:
            lead_p.append(eaves_run("rl_apron", M, [(0.0, th + 0.007), (g, th + 0.009), (la, sl_a + 0.008), (la, sl_a + 0.002), (g, th + 0.002), (0.0, 0.002)],
                                    -(rl_w / 2 + 0.15), rl_w / 2 + 0.15, nst=6 if not Lv else 2))
            lead_p.append(eaves_run("rl_back", M, [(-rl_l, th + 0.014), (-(rl_l + g - 0.004), 0.009), (-(rl_l + g - 0.004), 0.003), (-rl_l, th + 0.006)],
                                    -(rl_w / 2 + 0.1), rl_w / 2 + 0.1))
        for sx in ((-1, 1) if Lv == 1 else ()):             # the stepped soakers as one strip per side
            ua_, ub_ = sorted((sx * (rl_w / 2 - 0.002), sx * (rl_w / 2 + 0.09)))
            lead_p.append(eaves_run("rl_step", M, [(0.014, th + 0.009), (-(rl_l - 0.004), th + 0.009), (-(rl_l - 0.004), 0.003), (0.014, 0.002)],
                                    ua_, ub_, caps=(sx > 0, sx < 0)))
        for sx in ((-1, 1) if not Lv else ()):
            for j in range(int(round(rl_l / g))):
                ua_, ub_ = sorted((sx * (rl_w / 2 - 0.002), sx * (rl_w / 2 + 0.085 + 0.012 * (j % 2))))
                lead_p.append(eaves_run("rl_step", M, [(-(j * g - 0.014), th + 0.009), (-((j + 1) * g - 0.004), 0.008), (-((j + 1) * g - 0.004), 0.003), (-(j * g - 0.014), th + 0.002)],
                                        ua_, ub_, caps=(sx > 0, sx < 0)))
        for f_ in lead_p:
            assign(f_, LEAD)
        lead_o = join(lead_p, name="rooflight_lead")
        paint(lead_o, grime=lambda co, n: 0.62 + 0.38 * noise_mask(0.25, seed=seed + 13, lo=0.2, hi=0.8)(co, n))
        ML = M @ Matrix.Translation((0, rl_l, hk)) @ Matrix.Rotation(radians(-4.5), 4, 'X')
        rw_, rt_, ov_ = 0.05, 0.024, 0.01
        lid = [box("rl_stile", (rw_, rl_l + 2 * ov_, rt_), loc=(sx * (rl_w / 2 + ov_ - rw_ / 2), -rl_l / 2, rt_ / 2)) for sx in (-1, 1)]
        lid += [box("rl_rail", (rl_w + 2 * ov_ - 2 * rw_, rw_, rt_), loc=(0, y_, rt_ / 2)) for y_ in (ov_ - rw_ / 2, -rl_l - ov_ + rw_ / 2)]
        lid.append(box("rl_bar", (0.022, rl_l - 2 * rw_ + 0.03, 0.03), loc=(0, -rl_l / 2, 0.015)))
        if not Lv:
            lid += [box("rl_hinge", (0.07, 0.03, 0.026), loc=(sx * 0.14, ov_ + 0.006, 0.006)) for sx in (-1, 1)]
            lid.append(box("rl_stay", (0.016, 0.15, 0.006), loc=(0.0, -rl_l - ov_ - 0.05, 0.004), rot=(-14, 0, 0)))
            lid.append(box("rl_stay_knob", (0.034, 0.034, 0.012), loc=(0.0, -rl_l - ov_ - 0.118, -0.012)))
        zL = ML.to_3x3() @ Vector((0, 0, 1))
        for f_ in lid:
            assign(setF(f_, ML), IRON)
            if Lv:                                          # the lid frame without its underside; far: its top only
                lod_faces(f_, keep=zL) if Lv >= 2 else lod_faces(f_, drop=-zL)
        lid_o = join(lid, name="rooflight_lid")
        paint(lid_o, grime=0.3, special=mask_mul(noise_mask(0.12, seed=seed + 14, lo=0.35, hi=0.8), 0.6), wear=0.15)
        gw_, gl_ = (rl_w - 2 * rw_ + 0.03) / 2, (rl_l - 2 * rw_ + 0.03) / 2
        # obscured glass: the pane samples one small patch of the lit-window texture, so it glows dim and even
        gl = make_mesh("rl_glass", [ML @ Vector(v) for v in ((-gw_, -rl_l / 2 - gl_, 0.012), (gw_, -rl_l / 2 - gl_, 0.012), (gw_, -rl_l / 2 + gl_, 0.012), (-gw_, -rl_l / 2 + gl_, 0.012))],
                       [(0, 1, 2, 3)], [(0.30, 0.36), (0.32, 0.36), (0.32, 0.39), (0.30, 0.39)])
        assign(gl, LIT if p.rooflight == "lit" else mat("window_dark"))
        if p.rooflight == "lit":
            rl_light = M @ Vector((0, rl_l / 2, -0.3))
    # sweep's cat ladder: two rails that follow the sagging slope, rungs nailed on top
    u_ld = None
    if p.ladder != "none" and Xr > 0.7:
        F_l = F_f if p.ladder == "front" else F_b
        s0_l = max(prof.s_of_d(k1) + 0.15, 0.24 * s_top)
        if typ == "gambrel":
            s0_l = prof.s_of_d(d_br + 0.05) + 0.12
        s1_l = s_top - 0.1
        u_ld = -min(0.45 * Xe(prof.at_s(s0_l)[1]), Xr - 0.45)
        if s1_l - s0_l > 1.0:
            lift = th + 0.012
            nsl = max(2, int((s1_l - s0_l) / 0.75))
            lad = []
            for sx in (-1, 1):
                V, fs = [], []
                for k in range(nsl + 1):
                    s_ = s0_l + (s1_l - s0_l) * k / nsl
                    for hh, du in (((0.0, -0.02), (0.0, 0.02), (0.065, 0.02), (0.065, -0.02)) if Lv < 2 else ((0.0, -0.025), (0.0, 0.025), (0.065, 0.0))):
                        d_, z_ = prof.pt(s_, lift + hh)
                        V.append(F_l @ Vector((u_ld + sx * 0.17 + du, d_, z_)))
                    if k:
                        nq = 4 if Lv < 2 else 3
                        b0, b1 = nq * (k - 1), nq * k
                        fs += [(b0 + i, b1 + i, b1 + (i + 1) % nq, b0 + (i + 1) % nq) for i in range(1 if Lv else 0, nq)]   # levels: no underside
                if Lv < 2:
                    fs += [(0, 1, 2, 3), (4 * nsl + 3, 4 * nsl + 2, 4 * nsl + 1, 4 * nsl)]
                lad.append(recalc_normals(mesh_object("ladder_rail", V, fs)) if not Lv else mesh_object("ladder_rail", V, fs))
            nrg = int((s1_l - s0_l - 0.16) / 0.3)
            for k in range(nrg + 1):
                s_ = s0_l + 0.08 + k * 0.3
                M = tile_mtx(F_l, u_ld + r_s.jitter(0, 0.006), prof.pt(s_, lift + 0.065), prof.pt(s_ + 0.1, lift + 0.065), yaw=r_s.jitter(0, 1.5))
                if Lv >= 2 and k % 2:                       # far: every second rung, a wider board (all are drawn)
                    continue
                lad.append(wbox("ladder_rung", M, (0.42, 0.05 if Lv < 2 else 0.09, 0.024), (0, 0, 0.012), only=(None, [1, 2, 3, 4, 5], [1])[Lv]))
            for o_ in lad:
                assign(o_, OAK)
                paint(o_, grime=0.55, wear=0.3)
                skin.append(o_)
            zc_r = H - rr * tan(p_top) + 0.014                # ridge tile centre height
            hp_ = [prof.pt(s1_l - 0.14, lift + 0.07), prof.pt(s1_l - 0.02, lift + 0.075), (a - rr * 0.8, zc_r + rr * 0.6 + 0.03), (a - rr * 0.35, zc_r + rr + 0.012),
                   (a + rr * 0.35, zc_r + rr + 0.012), (a + rr * 0.85, zc_r + rr * 0.5 + 0.02), (a + rr + 0.03, zc_r - 0.03)]
            if lead_roll:
                hp_ = hp_[:2] + [(a - 0.06, H + 0.07), (a + 0.06, H + 0.07), (a + 0.16, H - 0.16 * tan(p_top) + 0.03)]
            if Lv:
                hp_ = hp_[::2] if len(hp_) == 7 else [hp_[0], hp_[2], hp_[3], hp_[4]]
            for sx in ((-1, 1) if Lv < 2 else ()):            # iron hooks over the ridge
                hk_ = sweep("ladder_hook", [tuple(F_l @ Vector((u_ld + sx * 0.17, d_, z_))) for d_, z_ in hp_],
                            [(-0.003, -0.014), (0.003, -0.014), (0.003, 0.014), (-0.003, 0.014)], smooth_path=False, up=tuple(F_l.to_3x3() @ Vector((1, 0, 0))))
                assign(hk_, GUT)
                paint(hk_, special=0.5, grime=0.4)
                skin.append(hk_)
    # ---- slipped tiles resting on the course below, clustered replacement tiles, loose / curled shingles:
    #      all individual wedges carrying the covering texture, registered to the meshed (wavy) butt lines
    def clear(tag, u, s_):
        if tag == "f" and rl_on and abs(u - u_rl) < rl_w / 2 + 0.32 and s_rl - 0.3 < s_ < s_rl + rl_l + 0.32:
            return False
        if u_ld is not None and tag == ("f" if p.ladder == "front" else "b") and abs(u - u_ld) < 0.36:
            return False
        return True

    loose = []
    budget_t = 330 if shingle else 150
    for sh, F, pr, ext, N, st, uoff, tag, at in sheets:
        i_lo = (int(ceil(pr.s_of_d(k1) / g)) + 1) if pr is prof else 2
        i_hi = N - 3
        if i_hi - i_lo < 3:
            continue
        wm = ext(pr.at_s(0.5 * st)[1])
        area = max(0.0, wm[1] - wm[0]) * st

        def put(i, u, kind):
            eL, eR = ext(pr.at_s(min((i + 1) * g, st))[1])
            if not (i_lo <= i <= i_hi) or u < eL + 0.3 or u > eR - 0.3 or not clear(tag, u, i * g) or len(loose) >= budget_t:
                return
            (sa, la_), (sb, lb_) = at(i, u), at(i + 1, u)
            if kind == "slid":                                # slid down 30-80 mm and skewed, butt resting on the course below
                sl = r_s.u(0.03, 0.08)
                loose.append(tile_item(F, pr, u, sa - sl, th * min(1.0, sl / g) * 1.2 + 0.003, sb + 0.012, 0.45 * lb_ - tt, tw - 0.006, tt,
                                       u + uoff, i * g + 0.008, yaw=r_s.u(1.5, 4.0) * (1 if r_s.chance(0.5) else -1), roll=r_s.jitter(0, 1.0), dv=g - 0.014))
            elif kind == "repl":                              # later replacement: another tile of the same stock, sitting a little proud
                loose.append(tile_item(F, pr, u, sa - r_s.u(0.0, 0.006), la_ + r_s.u(0.002, 0.004) - tt, sb + 0.012, 0.45 * lb_ - tt, tw - r_s.u(0.004, 0.01), tt,
                                       u + uoff + r_s.u(0.0, 5.0), (i + int(r_s.u(0, 5.99))) * g + 0.008, yaw=r_s.jitter(0, 0.7), roll=r_s.jitter(0, 1.2), dv=g - 0.014))
            else:                                             # shingle: random width, butt, thickness and curl
                w_ = r_s.u(0.08, 0.2)
                tk_ = r_s.u(0.009, 0.017)
                sl = r_s.u(-0.008, 0.03)
                loose.append(tile_item(F, pr, u, sa - sl, max(la_ + r_s.u(0.002, 0.005) - tk_, th * max(0.0, sl) / g + 0.002), sb + 0.012, 0.45 * lb_ - tk_, w_, tk_,
                                       u + uoff + r_s.u(0.0, 5.0), (i + int(r_s.u(0, 5.99))) * g + 0.008, yaw=r_s.jitter(0, 1.6), roll=r_s.jitter(0, 4.5), dv=g - 0.014))

        if shingle:
            for k in range(int(area * 5.5)):
                i = int(r_s.u(1, N - 1.01))
                eL, eR = ext(pr.at_s(min((i + 1) * g, st))[1])
                if eR - eL > 0.8:
                    i_lo_, i_lo = i_lo, 1
                    put(i, r_s.u(eL + 0.32, eR - 0.32), "shingle")
                    i_lo = i_lo_
        else:
            for c in range(max(1, int(round(area * (0.3 if clay else 0.22))))):
                i = int(r_s.u(i_lo, i_hi + 0.99))
                eL, eR = ext(pr.at_s(min((i + 1) * g, st))[1])
                if eR - eL < 1.0:
                    continue
                u0_ = r_s.u(eL + 0.35, eR - 0.35)
                if r_s.chance(0.42):
                    put(i, u0_, "slid")
                    if r_s.chance(0.35):
                        put(i, u0_ + tw, "slid")
                else:
                    for k in range(int(r_s.u(2, 5.99))):
                        di = int(r_s.u(0, 2.99))
                        put(i + di, u0_ + (int(r_s.u(0, 2.99)) + 0.5 * (di % 2)) * tw, "repl")
    if loose and Lv < 2:                                      # lod 1: tops and butts (every other shingle); far: none
        lt_ = tiles_mesh("loose_tiles", loose if not Lv else loose[::(2 if shingle else 1)], sides=not Lv)
        assign(lt_, COV)
        skin.append(lt_)

    # ------------------------------------------------------------------ finials at the ridge ends

    fin_pts = []
    if p.finial != "none":
        xs = [-(Xr - 0.14), Xr - 0.14] if Xr > 0.3 else [0.0]
        for x in xs:
            P = wz((x, 0.0, H))
            if p.finial == "ball":
                fpf = [(0.0, 0.0), (0.085, 0.0), (0.09, 0.02), (0.06, 0.035), (0.04, 0.07), (0.05, 0.085), (0.036, 0.1), (0.05, 0.13), (0.082, 0.17),
                       (0.094, 0.215), (0.082, 0.26), (0.05, 0.295), (0.026, 0.315), (0.03, 0.34), (0.016, 0.365), (0.0, 0.385)]
                if Lv >= 2:                                 # far: base, neck, ball, tip
                    fpf = [(0.09, 0.0), (0.04, 0.07), (0.04, 0.115), (0.094, 0.215), (0.04, 0.31), (0.0, 0.385)]
                f_ = lathe("finial_ball", fpf, segments=lod_pick(8, 6, 4), loc=(P.x, P.y, P.z - 0.012))
                assign(f_, mat(*C["ridge"]) if not lead_roll else RIDGE)
            else:
                fpf = [(0.0, 0.0), (0.07, 0.0), (0.07, 0.015), (0.03, 0.05), (0.02, 0.12), (0.035, 0.14), (0.02, 0.16), (0.014, 0.3), (0.04, 0.34),
                       (0.048, 0.38), (0.04, 0.42), (0.012, 0.46), (0.008, 0.7), (0.0, 0.78)]
                if Lv >= 2:                                 # far: base, shaft, knop, needle
                    fpf = [(0.07, 0.0), (0.022, 0.1), (0.016, 0.3), (0.048, 0.38), (0.012, 0.46), (0.0, 0.78)]
                f_ = lathe("finial_spike", fpf, segments=lod_pick(8, 6, 4), loc=(P.x, P.y, P.z - 0.012))
                assign(f_, GUT)
            paint(f_, grime=0.3, special=0.3)
            fin_pts.append(P)

    # ------------------------------------------------------------------ warp the skin, weather it
    for o_ in skin:
        transform_verts(o_, warp)
    zr = max(H - ZE, 0.5)
    nm_a, nm_b, nm_c = noise_mask(1.3, seed=seed + 40, lo=0.3, hi=0.75), noise_mask(0.35, seed=seed + 41, lo=0.35, hi=0.8), noise_mask(2.6, seed=seed + 42, lo=0.2, hi=0.8)
    moss_amt = p.moss

    nm_d = noise_mask(0.45, seed=seed + 43, lo=0.2, hi=0.8)

    def eav_fn(co):                                           # 1 at the eaves edge, fading out ~0.5 m (in height) above it
        return np.clip(1.0 - (co[:, 2] - ZE) / 0.5, 0, 1)

    def moss_fn(co, n):
        north = np.clip(n[:, 1] * 0.9 + 0.5, 0.2, 1.0)
        low = 1.0 - 0.55 * np.clip((co[:, 2] - ZE) / zr, 0, 1) + 0.6 * eav_fn(co)
        return np.clip(moss_amt * 0.95 * north * low * nm_a(co, n) * (0.4 + 0.6 * nm_b(co, n)), 0, 0.85)

    def grime_fn(co, n):
        top = np.clip((co[:, 2] - ZE) / zr, 0, 1)
        return np.clip(0.15 + 0.38 * nm_c(co, n) * (0.5 + 0.5 * top) + 0.36 * eav_fn(co) * (0.45 + 0.55 * nm_d(co, n)), 0, 1)

    nm_w = noise_mask(0.3, seed=seed + 44, lo=0.3, hi=0.8)
    for o_ in skin:
        if o_.name.startswith(("cov_", "eaves_tiles", "loose_tiles", "ridge_tiles", "ridge_crests")):
            paint(o_, special=moss_fn, grime=grime_fn, wear=(mask_mul(nm_w, 0.2 if slate_c else 0.3) if (slate_c or shingle) else None))
    for o_ in bpy.data.objects:
        if o_.type == 'MESH' and o_.name.startswith("undercourse"):
            paint(o_, special=moss_fn, grime=0.6)

    # ------------------------------------------------------------------ orientation, sockets, lights
    R = Matrix.Rotation(radians(-90), 4, 'Z') if ridge_y else Matrix.Identity(4)
    rz = -90.0 if ridge_y else 0.0
    if ridge_y:
        for o_ in list(bpy.context.scene.objects):
            if o_.type == 'MESH' and o_.parent is None:
                o_.matrix_basis = R @ o_.matrix_basis
    W = lambda v: tuple(R @ Vector(v))
    for tag, P in lights:
        light("attic_" + tag, W(P), color="#ffc37a", intensity=2.2, range=4.5, flicker=0.05)
    if rl_light is not None:
        light("rooflight", W(rl_light), color="#ffc37a", intensity=1.2, range=2.5, flicker=0.05)
    face_rot = {"f": 0.0, "b": 180.0, "px": 90.0, "nx": -90.0}
    for tag, P in win_socks:
        socket("gable_window_" + tag, "attach", W(P), rot=(0, 0, face_rot[tag] + rz), role="window")
    nd = max(1, int(L // 2.8))
    for F, tag in mains:
        s_d = 0.36 * s_top
        d_, z_ = prof.at_s(s_d)
        half = Xe(z_) * 0.8
        for k in range(nd):
            u = -half + 2 * half * (k + 0.5) / nd
            socket(f"dormer_{tag}{k}", "attach", W(wz(F @ Vector((u, d_, z_)))), rot=(0, 0, face_rot[tag] + rz), role="dormer",
                   pitch=round(degrees(prof.ang_d(d_)), 1))
        d_, z_ = prof.at_s(0.8 * s_top)
        for k, sx in enumerate((-1, 1)):
            socket(f"chimney_{tag}{k}", "attach", W(wz(F @ Vector((sx * Xe(z_) * 0.6, d_, z_)))), rot=(0, 0, face_rot[tag] + rz), role="chimney",
                   pitch=round(degrees(prof.ang_d(d_)), 1))
        if p.gutter != "none":
            socket("downpipe_" + tag, "attach", W(F @ Vector(((b if typ != "hip" else b) - 0.55, -q_gc, 0.012))), rot=(0, 0, face_rot[tag] + rz), role="downpipe")
    for k, x in enumerate((-max(Xr - 0.14, 0.0), max(Xr - 0.14, 0.0))):
        socket(f"ridge_end_{k}", "attach", W(wz((x, 0.0, H)) + Vector((0, 0, 0.02))), rot=(0, 0, rz), role="weathervane")
