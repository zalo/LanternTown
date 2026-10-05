"""Clinker-built wooden rowing skiff moored on the canal. The hull is lofted from a parametric
surface (sheer, keel, stem and section curves): overlapping strakes with gains at the ends, copper
rivets along the lands, steam-bent ribs, keel / skeg / curved stem with a brass stem band, a
wineglass transom with sculling notch and painted name, capped gunwales with rubbing strakes, risers
carrying the thwarts with grown knees, stern sheets, a small foredeck with the coiled painter led to
the bow ring, bottom boards over bilge water, swivel rowlocks on pads, oars stowed on the thwarts and
optional cargo (slatted crates and a sack, or a brass hurricane lantern). Origin on the waterline.
"""
from tk import *
import math
import numpy as np
from mathutils import Vector

META = {"name": "Canal rowing boat", "class": "furniture", "mount": "ground", "allow_below_ground": True,
        "focus": [0.72, 0.0, 0.55]}
PARAMS = {
    "length": {"default": 3.8, "min": 2.8, "max": 5.2, "desc": "length over all (m)"},
    "beam": {"default": 1.0, "min": 0.85, "max": 1.15, "desc": "beam factor (1 = normal skiff proportions)"},
    "strakes": {"default": 7, "min": 5, "max": 9, "desc": "number of clinker strakes per side"},
    "paint": {"default": "#9fc4b4", "desc": "topside paint colour"},
    "trim": {"default": "#d8ccac", "desc": "sheer strake / lettering / foredeck paint colour"},
    "cargo": {"default": "lantern", "choices": ["empty", "crates", "lantern"], "desc": "what is carried"},
    "oars": {"default": 2, "min": 0, "max": 2, "desc": "number of oars (stowed on the thwarts, afloat when swamped)"},
    "name": {"default": "MARIGOLD", "desc": "name painted on the transom and bow ('' = none)"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "how far the paint has flaked (1 = about half bare wood)"},
    "algae": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "strength of the slime band at the waterline"},
    "bilge": {"default": 0.4, "min": 0.0, "max": 1.0,
              "desc": "bilge water (0 = dry, 0.4 = pool round the bottom boards, 1 = swamped: water over the boards, boat sits low)"},
    "fenders": {"default": 2, "min": 0, "max": 3, "desc": "rope fenders hung on the quay (+Y) side"},
    "intensity": {"default": 8.0, "min": 2.0, "max": 25.0, "desc": "lantern intensity (cd), cargo = lantern"},
}
VARIANTS = {
    "crates_blue": {"length": 4.5, "paint": "#5f83ad", "trim": "#cdbb8a", "cargo": "crates", "name": "HERON",
                    "strakes": 8, "wear": 0.3, "seed": 4},
    "dinghy_red": {"length": 2.9, "beam": 1.12, "strakes": 6, "paint": "#a84c36", "trim": "#ddd3b6", "cargo": "empty",
                   "oars": 1, "name": "PIP", "wear": 0.65, "fenders": 1, "bilge": 0.15, "seed": 11},
    "tarred_long": {"length": 5.1, "beam": 0.9, "strakes": 8, "paint": "#3a3d41", "trim": "#8a2f22", "cargo": "lantern",
                    "name": "OLD TOM", "algae": 0.9, "wear": 0.25, "fenders": 3, "seed": 23},
    "swamped_white": {"length": 3.4, "paint": "#d3ccb6", "trim": "#7f9bb0", "cargo": "empty", "oars": 1, "name": "",
                      "wear": 0.9, "bilge": 1.0, "algae": 1.0, "fenders": 0, "seed": 31},
}


def loft(name, loops, cap=True):
    """Skin a list of closed loops (equal point counts) into a closed solid."""
    n = len(loops[0])
    verts = [tuple(v) for lp in loops for v in lp]
    faces = []
    for j in range(len(loops) - 1):
        a0, b0 = j * n, (j + 1) * n
        for i in range(n):
            i2 = (i + 1) % n
            faces.append((a0 + i, a0 + i2, b0 + i2, b0 + i))
    if cap:
        faces.append(tuple(range(n - 1, -1, -1)))
        m = (len(loops) - 1) * n
        faces.append(tuple(range(m, m + n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def zband(z0, z1, z2, z3):
    """Mask: 0 below z0, 1 between z1..z2, 0 above z3 (world height)."""
    def f(co, n):
        z = co[:, 2]
        return np.clip(np.minimum((z - z0) / (z1 - z0), (z3 - z) / (z3 - z2)), 0.0, 1.0)
    return f


def keep_front(o, d):
    """Keep only the faces of a thin extrusion that lie furthest along d (painted lettering)."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(o.data)
    dv = Vector(d).normalized()
    cs = [(f, f.calc_center_median().dot(dv)) for f in bm.faces]
    mx = max(c for _, c in cs)
    bmesh.ops.delete(bm, geom=[f for f, c in cs if c < mx - 2e-4], context="FACES")
    bm.normal_update()
    bmesh.ops.reverse_faces(bm, faces=[f for f in bm.faces if f.normal.dot(dv) < 0])
    bm.to_mesh(o.data)
    bm.free()


def euler_to(n):
    e = Vector((0, 0, 1)).rotation_difference(Vector(n).normalized()).to_euler()
    return tuple(math.degrees(a) for a in e)


def by_idx(vals):
    """Mask given per vertex index (for meshes whose vertex order we built ourselves)."""
    arr = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)

    def f(co, n):
        if len(co) != len(arr):
            print("rowboat: by_idx size mismatch", len(co), len(arr))
            return np.zeros(len(co))
        return arr
    return f


def noise1(r, n=12):
    """Smooth 1D value noise on 0..1 with n cells, values 0..1."""
    vals = [r.u() for _ in range(n + 2)]

    def f(x):
        x = min(max(x, 0.0), 1.0) * n
        j = int(min(x, n - 1e-6))
        a = x - j
        a = a * a * (3 - 2 * a)
        return vals[j] * (1 - a) + vals[j + 1] * a
    return f


class Decal:
    """Thin surface patches (flaked paint, slime, streaks) collected into one mesh."""

    def __init__(self):
        self.v, self.f, self.area = [], [], 0.0

    def add(self, pts, nrm):
        q = [pts[0]]
        for v in pts[1:]:
            if np.linalg.norm(v - q[-1]) > 2e-4:
                q.append(v)
        if len(q) > 2 and np.linalg.norm(q[0] - q[-1]) < 2e-4:
            q.pop()
        if len(q) < 3:
            return
        c = np.cross(q[1] - q[0], q[2] - q[0])
        if float(np.dot(c, nrm)) < 0:
            q = q[::-1]
        for j in range(1, len(q) - 1):
            self.area += 0.5 * float(np.linalg.norm(np.cross(q[j] - q[0], q[j + 1] - q[0])))
        n0 = len(self.v)
        self.v += [tuple(float(x) for x in v) for v in q]
        self.f.append(tuple(range(n0, n0 + len(q))))

    def make(self, name, material, grain="X"):
        if not self.f:
            return None
        o = mesh_object(name, self.v, self.f)
        uv_grain(o, grain)
        assign(o, material)
        return o


def build(p, rng):
    L = p.length
    N = int(p.strakes)
    k = (L / 3.8) ** 0.4
    Bm = (0.26 * L + 0.36) * p.beam / 2 - 0.03       # moulded half breadth
    draft = 0.13 * k
    sink = 0.07 * smoothstep(0.5, 1.0, p.bilge)        # a swamped boat sits low
    wz = -sink                                         # world z of the hull's own (design) waterline
    zs_mid = 0.31 * k + wz
    bow_rise, stern_rise = 0.17 * k, 0.07 * k
    th = 0.015                                         # plank thickness -> 12 mm step at every land
    lap = 0.026
    rt, rw = 0.014, 0.026                              # rib moulded / sided
    y0 = 0.010
    rake_s, rake_t = 0.11 * L, 0.08 * k
    xT = -L / 2 + rake_t
    xF = L / 2 - 0.055 - rake_s
    Lt = xF - xT
    tm = 0.42
    TH, bz = math.radians(75.0), 1.5
    ZN, SN = 1 - math.cos(TH) ** bz, math.sin(TH)
    zk_mid = -draft + wz
    seed = int(p.seed)
    wr = p.wear

    m_paint = mat("wood_painted", p.paint)
    m_trim = mat("wood_painted", p.trim)
    m_oak = mat("wood_oak")
    m_anti = mat("wood_weathered", "#b4634c")       # red-oxide antifouling / slime (special = algae)
    m_weath = mat("wood_weathered")                 # silver bare wood: flaked paint, thwarts, boards
    m_brass = mat("brass", "#f0c8a0")
    m_rope = mat("rope")
    m_water = mat("water", "#5f5d45", flow=0.3, opacity=0.78)

    # ------------------------------------------------------------------ hull surface
    def zkeel(t):
        z = zk_mid
        if t < 0.35:
            z += 0.08 * k * ((0.35 - t) / 0.35) ** 2
        if t > 0.6:
            z += 0.05 * k * ((t - 0.6) / 0.4) ** 2.5
        return z

    def zsheer(t):
        if t > 0.4:
            return zs_mid + bow_rise * ((t - 0.4) / 0.6) ** 2
        return zs_mid + stern_rise * ((0.4 - t) / 0.4) ** 2

    def halfb(t):
        if t < tm:
            return Bm * (1 - 0.30 * ((tm - t) / tm) ** 2)
        u = (t - tm) / (1 - tm)
        return Bm * (1 - u ** 1.8)

    def aexp(t):
        if t < tm:
            return lerp(0.62, 0.8, ((tm - t) / tm) ** 2)
        return lerp(0.62, 1.3, smoothstep(tm, 1.0, t))

    def zfr(s):
        return (1 - math.cos(s * TH) ** bz) / ZN

    def hull(t, s, sg=1.0):
        t = min(max(t, 0.0), 1.0)
        s = min(max(s, 0.0), 1.0)
        f = zfr(s)
        y = y0 + halfb(t) * (math.sin(s * TH) / SN) ** aexp(t)
        zk = zkeel(t)
        z = zk + (zsheer(t) - zk) * f
        wb = max(0.0, (t - 0.5) / 0.5) ** 2
        wt = max(0.0, 1 - t / 0.25) ** 2
        x = xT + t * (xF - xT) + wb * rake_s * math.sin(s * math.pi / 2) ** 0.75 - wt * rake_t * f
        return Vector((x, sg * y, z))

    def hull_n(t, s, sg=1.0):
        e = 1e-3
        pt = hull(min(t + e, 1), s) - hull(max(t - e, 0), s)
        ps = hull(t, min(s + e, 1)) - hull(t, max(s - e, 0))
        n = ps.cross(pt)
        if n.length < 1e-12:
            n = Vector((0, 0, -1))
        n.normalize()
        return Vector((n.x, sg * n.y, n.z))

    def s_at_z(t, z):
        zk = zkeel(t)
        f = min(max((z - zk) / (zsheer(t) - zk), 0.0), 1.0)
        return math.acos(min(1.0, max(0.0, (1 - f * ZN)) ** (1 / bz))) / TH

    def s_at_y(t, y):
        b = halfb(t)
        if y <= y0 or b < 1e-6:
            return 0.0
        return math.asin(min(1.0, SN * min(1.0, (y - y0) / b) ** (1 / aexp(t)))) / TH

    def t_at_x(x, s):
        a, b = 0.0, 1.0
        for _ in range(26):
            m = (a + b) / 2
            if hull(m, s).x < x:
                a = m
            else:
                b = m
        return (a + b) / 2

    def ts_at(x, z):
        t = min(max((x - xT) / (xF - xT), 0.0), 1.0)
        s = 0.5
        for _ in range(5):
            s = s_at_z(t, z)
            t = t_at_x(x, s)
        return t, s

    def ymold(x, z):
        t, s = ts_at(x, z)
        return hull(t, s).y

    def x_of(t):
        return xT + t * (xF - xT)

    # strake edges: equal girth on the midship section
    ss = np.linspace(0, 1, 241)
    mp = np.array([tuple(hull(tm, s)) for s in ss])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(mp, axis=0), axis=1))])
    G = [cum[-1] * i / N for i in range(N + 1)]

    def s_of_g(g):
        return float(np.interp(g, cum, ss))

    S = [s_of_g(g) for g in G]
    i_paint = N - 2
    for i in range(1, N - 1):
        if hull(tm, S[i]).z - wz >= 0.03:
            i_paint = i
            break

    def gain(t):
        return smoothstep(0.0, 0.06, t) * (1 - smoothstep(0.86, 1.0, t))

    def strake_sec(i, t, sg):
        kk = gain(t)
        if i > 0:
            s_lo, off = s_of_g(G[i] - lap * kk), th * kk
        else:
            s_lo, off = 0.0, 0.0
        pl, nl = hull(t, s_lo, sg), hull_n(t, s_lo, sg)
        ph, nh = hull(t, S[i + 1], sg), hull_n(t, S[i + 1], sg)
        return pl + nl * off, pl + nl * (off + th), ph, ph + nh * th, nl, nh

    def fvis(i, t, W):
        """Fraction of the strake face (lower edge -> upper edge) that is not covered by the next land / the rubbing strake."""
        if i == N - 1:
            return max(0.3, 1 - 0.044 / max(W, 0.05))
        return max(0.3, 1 - lap * gain(t) / max(W, 0.03))

    nu = 26
    tst = [1 - (1 - j / (nu - 1)) ** 1.3 for j in range(nu)]

    def inside_grime(z, rib=False):
        """Interior: dry and merely grubby under the gunwale, dark and damp low in the bilge, dirt in the rib corners."""
        damp = clamp(1 - (z - zk_mid) / (0.25 * k))
        return min(1.0, 0.14 + 0.58 * damp ** 0.8 + (0.38 if rib else 0.0))

    def bilge_grime(co, n):
        return np.clip(0.1 + 0.6 * np.clip(1 - (co[:, 2] - zk_mid) / (0.25 * k), 0, 1) ** 0.8, 0, 1)

    def g_wl(z):
        """Outside: grubby boot-top just above the water."""
        return 0.5 * clamp(1 - (z - wz - 0.03) / 0.12)

    def inward(c, n):
        return (c - Vector((c.x, 0.0, zs_mid + 0.5))).dot(n) < 0

    # ribs first: the planking stations sit on the ribs so the dirt in the rib / plank corners has vertices
    n_rib = max(6, int(round(0.86 * (xF - xT) / (0.21 * (L / 3.8) ** 0.5))))
    rib_ts = [0.07 + 0.86 * j / (n_rib - 1) for j in range(n_rib)]
    st_d = {0.0: False, 0.03: False, 0.95: False, 0.968: False, 0.984: False, 1.0: False}
    bays = n_rib <= 13                                 # long boats: no mid-bay stations, every other rib bay reads dirty
    for j_, a_ in enumerate(rib_ts):
        st_d[a_] = True if bays else j_ % 2 == 0
    if bays:
        for a_, b_ in zip(rib_ts, rib_ts[1:]):
            st_d[(a_ + b_) / 2] = False
    st_p = sorted(st_d.items())

    ip = max(1, i_paint - 1)                           # first painted strake: the one crossing the waterline
    tst_b = [1 - (1 - j / 19) ** 1.3 for j in range(20)]   # the bottom is under water in the canal
    orng_ = rng.fork("planks")
    for sg in (1.0, -1.0):
        for i in range(N):
            painted = i >= ip
            loops, gr, spc = [], [], []
            if painted:
                gj = orng_.u(0.0, 0.1)                 # every plank weathers a little differently
                for t, is_rib in st_p:
                    D, C, A, Bp, nl, nh = strake_sec(i, t, sg)
                    kk = gain(t)
                    fv = fvis(i, t, (Bp - C).length)
                    F1, F2 = C.lerp(Bp, fv - 0.085), C.lerp(Bp, fv - 0.012)
                    loops.append([D, C, F1, F2, Bp, A])
                    sh = 0.5 if i == N - 1 else 0.7 * kk        # narrow dirt line tucked under the land above
                    gr += [max(0.9 * kk, inside_grime(D.z, is_rib)), gj, gj, sh, 0.8, inside_grime(A.z, is_rib)]
                    spc += [0.0, 0.9, 0.22, 0.0, 0.0, 0.0]     # paint chalked pale along the exposed lower edge of every plank
            else:
                for t in tst_b:
                    D, C, A, Bp, nl, nh = strake_sec(i, t, sg)
                    loops.append([D, C, Bp, A])
                    al = (0.25 + 0.75 * p.algae) * clamp((Bp.z - wz + 0.2) / 0.16) * orng_.u(0.5, 1.0)
                    gr += [max(0.7, inside_grime(D.z)), orng_.u(0.05, 0.3), orng_.u(0.05, 0.3), inside_grime(A.z)]
                    spc += [0.0, 0.5 * al * orng_.u(0.3, 1.0), al, 0.0]
            o = loft(f"strake{i}", loops)
            uv_grain(o, "X")
            smooth(o, 35)
            assign(o, (m_trim if i == N - 1 else m_paint) if painted else m_anti)
            assign_faces(o, m_oak, inward)
            paint(o, grime=by_idx(gr), special=by_idx(spc))

    # ------------------------------------------------------------------ strake face tables (+Y side) for surface patches
    NT = 181
    tabC, tabB, tabN, tabW, tabF = [], [], [], [], []
    for i in range(N):
        cs, bs, ns, ws, fs = [], [], [], [], []
        for j in range(NT):
            t = j / (NT - 1)
            D, C, A, Bp, nl, nh = strake_sec(i, t, 1.0)
            W = (Bp - C).length
            cs.append(tuple(C)); bs.append(tuple(Bp)); ns.append(tuple(nl)); ws.append(W); fs.append(fvis(i, t, W))
        tabC.append(np.array(cs)); tabB.append(np.array(bs)); tabN.append(np.array(ns)); tabW.append(ws); tabF.append(fs)

    def tab_at(t):
        x = min(max(t, 0.0), 1.0) * (NT - 1)
        j = min(int(x), NT - 2)
        return j, x - j

    def face_w(i, t):
        j, a = tab_at(t)
        return tabW[i][j] * (1 - a) + tabW[i][j + 1] * a, tabF[i][j] * (1 - a) + tabF[i][j + 1] * a

    def face_z(i, t, f):
        j, a = tab_at(t)
        cz = tabC[i][j][2] * (1 - a) + tabC[i][j + 1][2] * a
        bz_ = tabB[i][j][2] * (1 - a) + tabB[i][j + 1][2] * a
        return cz + (bz_ - cz) * f

    def sadd(dec, i, sg, tf, off):
        """Add a polygon given in (t, f) strake-face coordinates, `off` metres proud of the plank."""
        pts = []
        for (t, f) in tf:
            j, a = tab_at(t)
            C = tabC[i][j] * (1 - a) + tabC[i][j + 1] * a
            B = tabB[i][j] * (1 - a) + tabB[i][j + 1] * a
            Nn = tabN[i][j] * (1 - a) + tabN[i][j + 1] * a
            P = C + (B - C) * f + Nn * off
            pts.append(np.array((P[0], sg * P[1], P[2])))
        dec.add(pts, np.array((Nn[0], sg * Nn[1], Nn[2])))

    # transom plane helpers (aft face)
    ta = 0.004
    tf = ta + 0.03 / (xF - xT)
    D0 = zsheer(0) - zkeel(0)
    slope = rake_t / D0

    def tr_x(z):
        return hull(ta, s_at_z(ta, z)).x

    def tr_y(z):
        return hull(ta, s_at_z(ta, z)).y

    def tadd(dec, yz, off):
        dec.add([np.array((tr_x(z) - off, y, z)) for (y, z) in yz], np.array((-1.0, 0.0, 0.0)))

    nm = str(p.name).strip()
    size_b = min(0.062, 0.62 * (G[N] - G[N - 1] - 0.03))
    tcn = 0.80
    nm_ht = (0.45 * size_b * max(len(nm), 1) + 0.03) / Lt if nm else 0.0     # half length of the bow name in t
    zc_n = zkeel(0) + D0 * 0.74
    size_t = min(0.075, 0.62 * 2 * tr_y(zc_n) / (0.78 * max(len(nm), 3)))

    # ------------------------------------------------------------------ flaked paint: hard-edged chips down to silver wood
    # Paint goes first along the lands (lower plank edges), under the land above, round the rivets, at the stem,
    # the quarters and the boot-top; slivers follow the grain. Bare area is capped near 50 % at wear = 1.
    riv_ts = sorted(rib_ts + [(a + b) / 2 for a, b in zip(rib_ts, rib_ts[1:])] + [0.025, 0.955])
    if wr > 0.02:
        fl, tfl = Decal(), Decal()
        frg = rng.fork("flakes")
        vis_area = 0.0
        for sg in (1.0, -1.0):
            for i in range(ip, N):
                r = frg.fork(f"s{i}_{int(sg)}")
                nz = noise1(r, max(6, int(L / 0.32)))
                nz2 = noise1(r, max(6, int(L / 0.22)))
                vis_area += sum(tabW[i][j] * tabF[i][j] for j in range(NT)) / NT * Lt

                def env(t, i=i, nz=nz):
                    zf = face_z(i, t, 0.3) - wz
                    b = 0.30 * smoothstep(0.88, 0.99, t) + 0.20 * (1 - smoothstep(0.0, 0.07, t)) + 0.22 * clamp(1 - abs(zf - 0.07) / 0.11)
                    if i == N - 1:
                        b += 0.06
                    return wr * (0.2 + 1.0 * nz(t)) + b * (0.3 + 0.7 * wr)

                def in_name(t, f, i=i):
                    return bool(nm) and i == N - 1 and abs(t - tcn) < nm_ht and 0.12 < f < 0.72

                # ragged bare band creeping up from the land
                t = 0.006 + r.u(0, 0.01)
                while t < 0.992:
                    dt = r.u(0.012, 0.062) / Lt
                    e = env(t + dt / 2)
                    W, fv = face_w(i, t + dt / 2)
                    if e > 0.15 and r.chance(0.93):
                        h = clamp((e - 0.08) * 0.44 * r.u(0.5, 1.35), 0.02, 0.42) * fv
                        if in_name(t + dt / 2, h):
                            h = min(h, 0.1)
                        if r.chance(0.25):       # splinter
                            sadd(fl, i, sg, [(t, 0.0), (t + dt, 0.0), (t + dt * r.u(0.15, 0.85), h * 1.25)], 0.0009)
                        else:
                            sadd(fl, i, sg, [(t, 0.0), (t + dt, 0.0), (t + dt * r.u(0.8, 1.0), h * r.u(0.75, 1.1)),
                                             (t + dt * r.u(0.0, 0.2), h * r.u(0.75, 1.1))], 0.0009)
                    # ... and hanging down from under the land above / the rubbing strake
                    eu = 0.62 * e
                    if eu > 0.2 and r.chance(0.8):
                        h = clamp((eu - 0.12) * 0.4 * r.u(0.4, 1.3), 0.015, 0.2) * fv
                        if not in_name(t + dt / 2, fv - h):
                            sadd(fl, i, sg, [(t + dt * r.u(0.0, 0.2), fv - h * r.u(0.7, 1.1)), (t + dt * r.u(0.8, 1.0), fv - h * r.u(0.7, 1.1)),
                                             (t + dt, fv + 0.012), (t, fv + 0.012)], 0.0009)
                    t += dt + (r.u(0.002, 0.012) / Lt if r.chance(0.3) else 0.0)

                # loose slivers and small stepped patches in the field
                def sliver(tc, fc, ln, hh, fv):
                    dl = ln / 2 / Lt
                    fc = min(max(fc, hh + 0.01), fv - hh - 0.01)
                    if in_name(tc, fc):
                        return
                    if r.chance(0.5):
                        q = [(tc - dl, fc + r.jitter(0, hh * 0.5)), (tc + r.jitter(0, dl * 0.5), fc - hh), (tc + dl, fc + r.jitter(0, hh * 0.5)),
                             (tc + r.jitter(0, dl * 0.5), fc + hh)]
                    else:
                        q = [(tc - dl, fc - hh), (tc + dl * r.u(0.55, 1.0), fc - hh * r.u(0.6, 1.0)), (tc + dl, fc + hh),
                             (tc - dl * r.u(0.55, 1.0), fc + hh * r.u(0.6, 1.0))]
                    sadd(fl, i, sg, q, 0.0011)

                for _ in range(int(Lt * (4 + 26 * wr) * r.u(0.8, 1.2))):
                    t0 = r.u(0.012, 0.988)
                    if r.u() > 0.25 + 0.75 * nz2(t0):
                        continue
                    W, fv = face_w(i, t0)
                    f0 = (0.06 + 0.68 * r.u() ** 1.4) * fv
                    hh0 = r.u(0.002, 0.0055) * (1 + 0.6 * wr) / W
                    for sl in range(r.pick([1, 1, 2, 2, 3])):
                        sliver(t0 + r.jitter(0, 0.012) / Lt, f0 + sl * hh0 * 1.8, r.u(0.010, 0.032) * (1 + 0.9 * wr), hh0 * r.u(0.7, 1.1), fv)
                # chips round the rivet heads
                if i >= 1:
                    for t_r in riv_ts:
                        if r.chance(0.18 + 0.5 * wr):
                            W, fv = face_w(i, t_r)
                            for _ in range(r.pick([1, 2])):
                                sliver(t_r + r.jitter(0, 0.012) / Lt, r.u(0.05, 0.2) * fv, r.u(0.008, 0.026), r.u(0.002, 0.005) / W, fv)
        # transom (grain runs athwartships)
        r = frg.fork("transom")
        zt0, zt1 = zkeel(0) + 0.05, zsheer(0) - 0.012
        for _ in range(int((8 + 90 * wr) * (0.6 + 0.4 * p.beam))):
            z = lerp(zt0, zt1, r.u() ** r.pick([0.6, 1.0, 1.8]))
            yw = tr_y(z) - 0.012
            ln, hh = r.u(0.012, 0.05) * (1 + 0.8 * wr), r.u(0.002, 0.006) * (1 + 0.6 * wr)
            if yw < ln * 0.7:
                continue
            yc = (yw - ln / 2) * (r.pick([-1, 1]) * (1 - r.u() ** 2.2) if r.chance(0.6) else r.u(-1, 1))
            if nm and abs(z - zc_n) < size_t * 0.9 and abs(yc) < 0.42 * size_t * len(nm) + ln / 2 + 0.02:
                continue
            tadd(tfl, [(yc - ln / 2, z + r.jitter(0, hh * 0.5)), (yc + r.jitter(0, ln / 4), z - hh), (yc + ln / 2, z + r.jitter(0, hh * 0.5)),
                       (yc + r.jitter(0, ln / 4), z + hh)], 0.0010)
        fo = [fl.make("paint_flakes", m_weath), tfl.make("paint_flakes_transom", m_weath, grain="Y")]
        for o in fo:
            if o is not None:
                # bare wood stays silver-grey: only a breath of dirt, a little more at the boot-top
                paint(o, grime=mask_mul(noise_mask(0.25, seed=seed + 15, lo=0.3, hi=0.8), 0.12), wear=0.5)
                paint(o, grime=lambda co, n: 0.3 * np.clip(1 - (co[:, 2] - wz - 0.03) / 0.1, 0, 1))

    # ------------------------------------------------------------------ slime band at the waterline, dried scum edge, weeping rivets
    if p.algae > 0.03:
        slm, scm = Decal(), Decal()
        ar = rng.fork("algae")
        nb = 46
        tbs = [0.5 - 0.5 * math.cos(math.pi * (0.03 + 0.95 * j / (nb - 1))) for j in range(nb)]
        zlo = wz - 0.045
        hs = 0.72 + 0.38 * p.algae
        for sg in (1.0, -1.0):
            nzb = noise1(ar, 13)
            # top edge follows the water the boat actually floats in (a swamped hull has slimed up to its new line)
            zt = [wz + 0.45 * sink + (0.06 + 0.05 * nzb(t) + ar.jitter(0, 0.007)) * hs for t in tbs]
            for i in range(N):
                prev = None
                for j, t in enumerate(tbs):
                    jj, a = tab_at(t)
                    cz = tabC[i][jj][2] * (1 - a) + tabC[i][jj + 1][2] * a
                    bz_ = tabB[i][jj][2] * (1 - a) + tabB[i][jj + 1][2] * a
                    fv = tabF[i][jj]

                    def fz(z):
                        if bz_ - cz < 1e-4:
                            return fv if z > cz else 0.0
                        return min(max((z - cz) / (bz_ - cz), 0.0), fv)

                    cur = (t, fz(zlo), fz(zt[j] - 0.017), fz(zt[j]))
                    if prev is not None:
                        if max(prev[2] - prev[1], cur[2] - cur[1]) > 0.004:
                            sadd(slm, i, sg, [(prev[0], prev[1]), (cur[0], cur[1]), (cur[0], cur[2]), (prev[0], prev[2])], 0.0016)
                        if max(prev[3] - prev[2], cur[3] - cur[2]) > 0.004:
                            sadd(scm, i, sg, [(prev[0], prev[2]), (cur[0], cur[2]), (cur[0], cur[3]), (prev[0], prev[3])], 0.0016)
                    prev = cur
            # weeping streaks below the rivets on the lower strakes
            for i in range(1, N):
                for t_r in riv_ts:
                    W, fv = face_w(i - 1, t_r)
                    zr = face_z(i - 1, t_r, fv) - wz
                    if zr < 0.02 or zr > 0.2 or not ar.chance(0.3 + 0.4 * p.algae) or gain(t_r) < 0.5:
                        continue
                    ln = min(ar.u(0.025, 0.08), zr) / max(W, 0.03)
                    hw = ar.u(0.003, 0.006) / Lt
                    sadd(slm, i - 1, sg, [(t_r - hw, fv), (t_r + hw, fv), (t_r + ar.jitter(0, hw), fv - ln)], 0.0013)
        # across the transom
        zb0 = max(zlo, zkeel(0) + 0.012)
        ztt = wz + 0.45 * sink + (0.07 + ar.u(0.0, 0.03)) * hs
        ny = 4
        for j in range(ny):
            ua, ub = -1 + 2 * j / ny, -1 + 2 * (j + 1) / ny
            dza, dzb = 0.012 * math.sin(j * 2.1 + seed), 0.012 * math.sin((j + 1) * 2.1 + seed)

            def py(u, z):
                return u * (tr_y(z) - 0.004)
            za, zb_ = ztt + dza, ztt + dzb
            tadd(slm, [(py(ua, zb0), zb0), (py(ub, zb0), zb0), (py(ub, zb_ - 0.017), zb_ - 0.017), (py(ua, za - 0.017), za - 0.017)], 0.0016)
            tadd(scm, [(py(ua, za - 0.017), za - 0.017), (py(ub, zb_ - 0.017), zb_ - 0.017), (py(ub, zb_), zb_), (py(ua, za), za)], 0.0016)
        so = slm.make("slime_band", m_weath)             # weathered wood's special mask is algae
        if so is not None:
            paint(so, special=1.0, grime=lambda co, n: 0.3 * np.clip(1 - (co[:, 2] - wz + 0.045) / 0.045, 0.0, 1.0), wear=0.0, mode="set")
        co_ = scm.make("slime_scum", m_weath)
        if co_ is not None:
            paint(co_, wear=1.0, special=0.0, grime=0.0, mode="set")

    # ------------------------------------------------------------------ ribs + rivets
    rprof = [(-rt / 2, -rw / 2), (rt / 2, -rw / 2), (rt / 2, rw / 2), (-rt / 2, rw / 2)]
    ribs = []
    for j, t in enumerate(rib_ts):
        side = []
        for i in range(N, 0, -1):
            side.append(hull(t, S[i]) - hull_n(t, S[i]) * (rt / 2))
        c = hull(t, 0.0)
        path = side + [Vector((c.x, 0, c.z + 0.013 + rt / 2))] + [Vector((v.x, -v.y, v.z)) for v in reversed(side)]
        ribs.append(sweep(f"rib{j}", [tuple(v) for v in path], rprof, smooth_path=False, up=(1, 0, 0)))
    ribs = join(ribs, "ribs")
    assign(ribs, m_oak)
    paint(ribs, grime=lambda co, n: 0.9 * np.clip(1 - (co[:, 2] - zk_mid) / (0.2 * k), 0, 1) ** 1.2,
          wear=mask_mul(lambda co, n: np.clip((co[:, 2] - zk_mid) / (0.3 * k), 0.0, 1.0), noise_mask(0.12, seed=seed + 5, lo=0.2, hi=0.7), 0.6))

    rv = [(0.0062 * math.cos(a), 0.0062 * math.sin(a), 0.0) for a in np.radians(np.arange(4) * 90.0)] + [(0, 0, 0.0034)]
    riv = mesh_object("rivet", rv, [(j, (j + 1) % 4, 4) for j in range(4)])
    rm = []
    for sg in (1.0, -1.0):
        for i in range(1, N):
            fl_ = 0.5 * lap / (G[i + 1] - G[i] + lap)
            for t in riv_ts:
                D, C, A, Bp, nl, nh = strake_sec(i, t, sg)
                pos = C.lerp(Bp, fl_ * gain(t) + 0.04)
                if pos.z - wz < -0.01:
                    continue
                rm.append(trs(tuple(pos + nl * 0.0012), euler_to(nl)))
    # rove heads inside, where the upper lands cross the ribs
    for sg in (1.0, -1.0):
        for t in rib_ts:
            for i in range(max(2, N - 2), N):
                nn = hull_n(t, S[i], sg)
                rm.append(trs(tuple(hull(t, S[i], sg) - nn * (rt + 0.0003)), euler_to(-nn)))
    rivets = instances_to_mesh("rivets", riv, rm, keep_base=False)
    assign(rivets, m_brass)
    paint(rivets, special=mask_mul(noise_mask(0.2, seed=seed + 3), 0.7), grime=0.3)

    # ------------------------------------------------------------------ keel, skeg, stem
    kw = 0.027
    loops = []
    for j in range(15):
        t = j / 14 * 0.985
        c = hull(t, 0.0)
        zo = min(c.z - 0.045, zk_mid - 0.045) if t <= 0.6 else c.z - 0.045
        loops.append([(c.x, -kw, c.z + 0.012), (c.x, kw, c.z + 0.012), (c.x, kw, zo), (c.x, -kw, zo)])

    def stem_pt(s):
        c = hull(1.0, s)
        tg = hull(1.0, min(s + 2e-3, 1.0)) - hull(1.0, max(s - 2e-3, 0.0))
        tg = Vector((tg.x, 0, tg.z)).normalized()
        return Vector((c.x, 0, c.z)), tg, Vector((tg.z, 0, -tg.x))

    stem_ss = [0.03 + 0.97 * (j / 13) ** 0.9 for j in range(14)]
    for s in stem_ss:
        c, tg, d = stem_pt(s)
        din = lerp(0.012, 0.032, smoothstep(0.0, 0.3, s))
        dout = lerp(0.045, 0.052, s)
        a, b = c - d * din, c + d * dout
        loops.append([(a.x, -kw, a.z), (a.x, kw, a.z), (b.x, kw, b.z), (b.x, -kw, b.z)])
    c1, tg1, d1 = stem_pt(1.0)
    for up_, sc in ((0.05, 1.0), (0.075, 0.8)):
        a, b = c1 + tg1 * up_ - d1 * 0.032 * sc, c1 + tg1 * up_ + d1 * 0.052 * sc
        loops.append([(a.x, -kw * sc, a.z), (a.x, kw * sc, a.z), (b.x, kw * sc, b.z), (b.x, -kw * sc, b.z)])
    keel = loft("keel", loops)
    bevel(keel, 0.005, segments=1)
    smooth(keel, 35)
    uv_grain(keel, "X")
    assign(keel, m_oak)
    assign_faces(keel, m_anti, lambda c, n: c.z - wz < 0.035 and not (abs(n.z) > 0.7 and c.z > zk_mid))
    zwb = zband(wz - 0.2, wz - 0.03, wz + 0.05, wz + 0.1)
    paint(keel, grime=lambda co, n: np.where(co[:, 2] - wz > 0.035, 0.9 * np.clip(1 - (co[:, 2] - wz - 0.035) / 0.12, 0, 1), 0.2))
    paint(keel, special=mask_mul(zwb, 0.35 + 0.65 * p.algae),
          wear=mask_mul(height_ramp(zsheer(1.0) + 0.1, zsheer(1.0) - 0.15), 0.5))
    # brass stem band
    bp = []
    for s in [0.06 + 0.94 * j / 11 for j in range(12)]:
        c, tg, d = stem_pt(s)
        bp.append(tuple(c + d * (lerp(0.045, 0.052, s) + 0.001)))
    bp.append(tuple(c1 + tg1 * 0.04 + d1 * 0.053))
    band = sweep("stem_band", bp, [(-0.002, -0.009), (0.002, -0.009), (0.002, 0.009), (-0.002, 0.009)], up=(0, 1, 0), smooth_path=False)
    assign(band, m_brass)
    paint(band, special=lambda co, n: np.clip(0.3 + 0.7 * (1 - (co[:, 2] - wz) / 0.25), 0.3, 1.0), grime=height_ramp(wz, wz + 0.2))

    # ------------------------------------------------------------------ transom
    def transom_loop(t):
        pts = [hull(t, s) for s in np.linspace(0, 1, 9)]
        top = pts[-1]
        crown = []
        for fy in (0.75, 0.5, 0.28):
            crown.append((fy * top.y, 0.03 * k * (1 - fy * fy)))
        r = 0.032
        notch = [(r * math.cos(a), 0.03 * k - r * math.sin(a) * 1.1) for a in np.radians([0, 35, 70, 90, 110, 145, 180])]
        notch[0] = (r * 1.25, 0.03 * k * (1 - (r * 1.25 / top.y) ** 2))
        notch[-1] = (-notch[0][0], notch[0][1])
        mid = crown + notch + [(-y, dz) for (y, dz) in reversed(crown)]
        lp = [tuple(v) for v in pts]
        lp += [(top.x - slope * dz, y, top.z + dz) for (y, dz) in mid]
        lp += [(v.x, -v.y, v.z) for v in reversed(pts)]
        return lp

    transom = loft("transom", [transom_loop(ta), transom_loop(tf)])
    bevel(transom, 0.004, segments=1)
    uv_grain(transom, "Y")
    assign(transom, m_paint)
    assign_faces(transom, m_oak, lambda c, n: n.x > 0.5)
    paint(transom, grime=lambda co, n: np.where(n[:, 0] > 0.5, 0.45, 0.6 * np.clip(1 - (co[:, 2] - wz - 0.03) / 0.12, 0, 1)),
          special=mask_mul(facing((0, 0, 1)), 0.6))

    # ------------------------------------------------------------------ layout of thwarts / fenders (needed for the chafe marks)
    z_tt = round(zs_mid - 0.15, 3)
    t_th = 0.03
    z_tb = z_tt - t_th
    cap_h = 0.022
    thw_ts = [0.45, 0.70] if L < 4.4 else [0.36, 0.55, 0.73]
    thw = [(x_of(t), 0.21 if j < len(thw_ts) - 1 else 0.185) for j, t in enumerate(thw_ts)]      # (x centre, width)
    row_x = [xc - w / 2 - 0.29 for (xc, w) in thw[:-1]]
    frng = rng.fork("fenders")
    nf = int(p.fenders)
    fen_ts = [lerp(0.2, 0.72, (jf + 0.5) / nf) + frng.jitter(0.0, 0.03) for jf in range(nf)]
    fen_x = [hull(t, 1.0).x for t in fen_ts]

    # ------------------------------------------------------------------ gunwale cap, rubbing strake, riser
    for sg in (1.0, -1.0):
        cts = list(tst)
        for xr in row_x:                               # dirt gathers against the rowlock pads
            cts += [t_at_x(xr + d_, 1.0) for d_ in (-0.2, -0.14, 0.14, 0.2)]
        if sg > 0:
            for xf_ in fen_x:                          # the fender lanyards saw a groove into the cap
                cts += [t_at_x(xf_ + d_, 1.0) for d_ in (-0.045, -0.013, 0.0, 0.013, 0.045)]
        cts = sorted(cts)
        cts = [t for j, t in enumerate(cts) if j == 0 or t - cts[j - 1] > 1e-4]
        cp, rp, sp = [], [], []
        for t in cts:
            P, n = hull(t, 1.0, sg), hull_n(t, 1.0, sg)
            cp.append((P.x, P.y, P.z + 0.009))
        for t in tst:
            P, n = hull(t, 1.0, sg), hull_n(t, 1.0, sg)
            nh = Vector((n.x, n.y, 0)).normalized()
            q = P + nh * (th + 0.009)
            rp.append((q.x, q.y, q.z - 0.026))
        for j in range(15):
            t = 0.02 + 0.9 * j / 14
            s = s_at_z(t, z_tb - 0.024)
            q = hull(t, s, sg) - hull_n(t, s, sg) * (rt + 0.009)
            sp.append(tuple(q))
        cap = sweep("gunwale", cp, [(-0.024, -cap_h / 2), (0.024, -cap_h / 2), (0.024, cap_h / 2 - 0.005), (0.019, cap_h / 2), (-0.019, cap_h / 2), (-0.024, cap_h / 2 - 0.005)],
                    smooth_path=False)
        rub = sweep("rubrail", rp, [(-0.01, -0.015), (0.01, -0.009), (0.01, 0.009), (-0.01, 0.015)], smooth_path=False)
        ris = sweep("riser", sp, [(-0.009, -0.024), (0.009, -0.024), (0.009, 0.024), (-0.009, 0.024)], smooth_path=False)
        for o in (cap, rub, ris):
            assign(o, m_oak)
        fx_ = fen_x if sg > 0 else []

        def cap_wear(co, n, fx_=fx_):
            w_ = 0.8 + 0.2 * noise_mask(0.3, seed=seed + 21, lo=0.2, hi=0.8)(co, n)         # sun-bleached, handled
            for xf_ in fx_:
                w_ = np.where(np.abs(co[:, 0] - xf_) < 0.006, 1.0, w_)
            return w_

        def cap_grime(co, n, fx_=fx_):
            g_ = 0.05 * np.ones(len(co))
            for xf_ in fx_:
                d_ = np.abs(co[:, 0] - xf_)
                g_ = np.where((d_ > 0.006) & (d_ < 0.03), 0.85, g_)
            for xr in row_x:
                d_ = np.abs(co[:, 0] - xr)
                g_ = np.where((d_ > 0.1) & (d_ < 0.17), 0.7, g_)
            return g_

        paint(cap, wear=cap_wear, grime=cap_grime, mode="set")
        paint(rub, wear=mask_mul(noise_mask(0.18, seed=seed + 22, lo=0.35, hi=0.8), 0.5), grime=0.45)
        paint(ris, grime=0.5)

    # ------------------------------------------------------------------ seats: stern sheets, thwarts, knees
    def seat(name, xa, xb, n=5, inset=0.004):
        xs = [lerp(xa, xb, i / (n - 1)) for i in range(n)]
        ys = [ymold(x, z_tb) - inset for x in xs]
        lps_ = []
        for u_ in (-1.0, -0.86, -0.62, -0.34, 0.0, 0.34, 0.62, 0.86, 1.0):   # stations across the boat carry the wear / dirt gradient
            lps_.append([(x, u_ * y, z_tt) for x, y in zip(xs, ys)] + [(x, u_ * y, z_tb) for x, y in zip(reversed(xs), reversed(ys))])
        o = loft(name, lps_)
        bevel(o, 0.004, segments=1)
        uv_grain(o, "Y")
        assign(o, m_weath)                              # scrubbed grey softwood, polished pale where people sit
        yc = sum(ys) / len(ys)
        xm = (xa + xb) / 2
        paint(o, wear=mask_mul(lambda co, nn: np.exp(-(co[:, 1] / (yc * 0.42)) ** 2) * np.clip(1.25 - np.abs(co[:, 0] - xm) / (xb - xa) * 1.2, 0, 1),
                               noise_mask(0.09, seed=seed + 30, lo=0.15, hi=0.6), 0.95),
              grime=lambda co, nn: np.clip(0.22 + 0.75 * (np.abs(co[:, 1]) / yc - 0.55) / 0.45, 0.22, 0.9))
        return o

    xs_a = hull(tf, s_at_z(tf, z_tt)).x - 0.002
    xs_b = xs_a + 0.36 * k
    seat("stern_sheets", xs_a, xs_b, n=4)
    for j, (xc, w) in enumerate(thw):
        t = thw_ts[j]
        seat(f"thwart{j}", xc - w / 2, xc + w / 2, n=2)
        # pillar under the thwart
        zf = zkeel(t) + 0.012
        if not (p.cargo == "lantern" and j == len(thw_ts) - 1):
            st = plank("pillar", z_tb - zf, 0.04, 0.04, loc=(xc, 0, (z_tb + zf) / 2), axis="Z", bevel=0.0)
            assign(st, m_oak)
            paint(st, grime=bilge_grime)
        # grown knees on top of the thwart
        tt = (xc - xT) / (xF - xT)
        zg = zsheer(tt) - 0.003
        for sg in (1.0, -1.0):
            lps = []
            for xx in (xc - 0.015, xc + 0.015):
                yh = lambda z: ymold(xx, z) - 0.002
                a0, a2 = Vector((yh(z_tt), z_tt)), Vector((yh(zg), zg))
                a1 = Vector((yh((z_tt + zg) / 2), (z_tt + zg) / 2))
                arm = min(0.24, a0.x * 0.5)
                b0, b2 = a2 + Vector((-0.05, 0)), Vector((a0.x - arm, z_tt + 0.014))
                bc = Vector((a0.x - 0.07, z_tt + 0.045))
                crv = [(1 - u) ** 2 * b0 + 2 * u * (1 - u) * bc + u * u * b2 for u in (0, 0.25, 0.5, 0.75, 1.0)]
                poly = [a0, a1, a2] + crv + [Vector((a0.x - arm, z_tt))]
                lps.append([(xx, sg * q.x, q.y) for q in poly])
            kn = loft("knee", lps)
            bevel(kn, 0.004, segments=1)
            uv_grain(kn, "Y")
            assign(kn, m_oak)
            paint(kn, grime=0.3)

    # quarter knees at the transom corners
    xq = hull(tf, 1.0).x
    zq = zsheer(0.0)
    for sg in (1.0, -1.0):
        top = []
        for t in (tf, 0.045, 0.09):
            P = hull(t, 1.0)
            top.append((P.x, sg * (P.y - 0.004)))
        top.append((xq + 0.05, sg * hull(tf, 1.0).y * 0.72))
        top.append((xq, sg * hull(tf, 1.0).y * 0.5))
        qk = loft("quarter_knee", [[(x, y, zq - 0.026) for x, y in top], [(x, y, zq + 0.016) for x, y in top]])
        bevel(qk, 0.004, segments=1)
        uv_grain(qk, "Y")
        assign(qk, m_oak)
        paint(qk, grime=0.3)

    # ------------------------------------------------------------------ foredeck (painted) + painter
    td0, td1 = 0.86, 0.985
    dts = [lerp(td0, td1, j / 5) for j in range(6)]

    def deck_loop(dz):
        a = [hull(t, 1.0) for t in dts]
        lp = [(v.x, v.y - 0.006, v.z + dz) for v in a]
        lp += [(v.x, -(v.y - 0.006), v.z + dz) for v in reversed(a)]
        lp.append((a[0].x + 0.07, 0.0, a[0].z + dz + 0.004))
        return lp

    deck = loft("foredeck", [deck_loop(-0.01), deck_loop(0.016)])
    bevel(deck, 0.004, segments=1)
    uv_grain(deck, "X")
    assign(deck, m_trim)
    paint(deck, grime=mask_mul(noise_mask(0.2, seed=seed + 41, lo=0.2, hi=0.8), 0.6), special=0.5)

    rr = rng.fork("rope")
    tc = 0.895
    Pc = hull(tc, 1.0)
    Pa, Pb = hull(0.87, 1.0), hull(0.95, 1.0)
    pitch = math.atan2(Pb.z - Pa.z, Pb.x - Pa.x)
    cx, cy, cz = Pc.x, 0.015, Pc.z + 0.016
    rope_r = 0.0085
    rc = min(0.085, Pc.y - 0.07)
    turns = 4
    ph = rr.u(0, 6.28)

    def tilt(lx, ly, lz):
        return (cx + lx * math.cos(pitch) - lz * math.sin(pitch), cy + ly, cz + lx * math.sin(pitch) + lz * math.cos(pitch))

    cpath = []
    na = turns * 14
    for j in range(na + 1):
        a = j / 14 * 2 * math.pi
        r = rc + 0.010 * math.sin(a * 0.37 + ph) + 0.006 * math.sin(a * 1.3 + ph * 2)
        cpath.append(tilt(r * math.cos(a + ph), r * math.sin(a + ph), rope_r + 0.0155 * a / (2 * math.pi) * 0.92))
    # lead: off the coil, forward along the deck, round the stemhead, down to the ring
    c_r, tg_r, d_r = stem_pt(0.80)
    eye_c = c_r + d_r * (0.052 + 0.012)
    ring_c = eye_c + Vector((0.008, 0, -0.03))
    knot = ring_c + Vector((0.012, 0.012, -0.022))
    Pd = hull(0.975, 1.0)
    lead = [cpath[-1], tilt(rc + 0.05, 0.05, 0.05), (Pd.x - 0.06, 0.05, Pd.z + 0.03),
            (c1.x - 0.01, 0.045, c1.z + 0.034), (c1.x + 0.05, 0.036, c1.z + 0.012), (c1.x + 0.075, 0.018, c1.z - 0.05),
            tuple(knot + Vector((0.004, 0.0, 0.035))), tuple(knot)]
    rope = sweep("painter_coil", cpath + lead[1:], rope_r, sides=5, step=0.035)
    kn = sphere("painter_knot", 0.019, loc=tuple(knot), segments=8, rings=5, scale=(1, 1, 1.25))
    tail = sweep("painter_tail", [tuple(knot), tuple(knot + Vector((0.012, 0.02, -0.04))), tuple(knot + Vector((0.01, 0.028, -0.09)))], rope_r * 0.9, sides=6, step=0.03)
    for o in (rope, kn, tail):
        assign(o, m_rope)
    paint(rope, grime=mask_mul(noise_mask(0.1, seed=seed + 44), 0.5))
    # eye bolt + ring on the stem
    eb = cylinder("eye_shank", 0.006, 0.02, loc=tuple(c_r + d_r * 0.055), rot=euler_to(d_r), verts=8)
    eye = torus("eye", 0.012, 0.0045, loc=tuple(eye_c), rot=(90, 0, 0), major_seg=10, minor_seg=5)
    ring = torus("bow_ring", 0.03, 0.0055, loc=tuple(ring_c), rot=(0, 90, 30), major_seg=14, minor_seg=5)
    for o in (eb, eye, ring):
        assign(o, m_brass)
        paint(o, special=0.5)
    socket("mooring_bow", "attach", tuple(ring_c), role="rope")
    socket("mooring_stern", "attach", (hull(0, 1).x, 0, zsheer(0) + 0.03), rot=(0, 0, 90), role="rope")
    # where the hull parts the water it floats in (z = 0): the section of the planking there, taken inside the
    # thickness of the planks, the transom and the stem. Whoever lays the water (tools/town.py terrain) cuts this out
    # of it, so the canal ends in the planking instead of running through the boat.
    wl = []
    for t in np.linspace(ta, 0.985, 26):
        s = s_at_z(t, 0.0)
        c, n = hull(t, s), hull_n(t, s)
        h = math.hypot(n.x, n.y) or 1.0
        sg = -1.0 if n.y < 0 else 1.0
        wl.append((c.x + sg * n.x / h * th * 0.8, c.y + sg * n.y / h * th * 0.8))
    wl[0] = (tr_x(0.0) + 0.012, wl[0][1])
    c, tg, d = stem_pt(s_at_z(1.0, 0.0))
    x_tip = c.x + 0.045 / max(d.x, 0.6) - 0.012
    wl = [q for q in wl if q[0] < x_tip - 0.01] + [(x_tip, kw - 0.008)]
    socket("waterline", "float", (0, 0, 0), outline=[round(float(v), 4) for q in wl + [(x, -y) for (x, y) in reversed(wl)] for v in q])

    # ------------------------------------------------------------------ bottom boards (a level platform on the ribs) + bilge water
    def yref(t):
        return hull(t, s_at_z(t, zkeel(t) + 0.035 * k)).y

    yr0 = yref(tm)
    fts = [0.15 + 0.63 * j / 9 for j in range(10)]
    tb = 0.016
    z_floor = zk_mid + 0.064                           # top of the boards amidships
    n_fb = 3 if Bm < 0.78 else 5
    pitch_b, wid_b = 0.128, 0.102                      # 26 mm gaps: the bilge water shows between the boards
    br = rng.fork("boards")
    for jb in range(n_fb):
        jj = jb - (n_fb - 1) / 2
        ja, jz = br.jitter(0.0, 0.005), br.jitter(0.0, 0.002)
        loops = []
        for t in fts:
            f = min(1.0, max(0.4, yref(t) / yr0))
            yc, hw = jj * pitch_b * f + ja, wid_b / 2 * f
            zb = max(z_floor - tb, zkeel(t) + 0.03) + jz
            xx = x_of(t)
            loops.append([(xx, yc - hw, zb), (xx, yc + hw, zb), (xx, yc + hw, zb + tb), (xx, yc - hw, zb + tb)])
        fb = loft("bottom_board", loops)
        bevel(fb, 0.003, segments=1)
        uv_grain(fb, "X")
        assign(fb, m_weath)
        paint(fb, grime=mask_mul(noise_mask(0.25, seed=seed + 50 + jb, lo=0.1, hi=0.7), 0.3 + 0.25 * min(1.0, p.bilge * 2)),
              special=mask_mul(noise_mask(0.12, seed=seed + 60), 0.25 + 0.5 * p.algae * p.bilge),
              wear=mask_mul(noise_mask(0.2, seed=seed + 61), lambda co, n: np.clip(1 - np.abs(co[:, 1]) / 0.2, 0, 1), 0.6))
    # two cleats hold the boards together
    for t in (fts[2], fts[-3]):
        f = min(1.0, max(0.4, yref(t) / yr0))
        cl = plank("board_cleat", ((n_fb - 1) * pitch_b + wid_b) * f, 0.04, 0.012, loc=(x_of(t), 0, max(z_floor, zkeel(t) + 0.03 + tb) + 0.006), axis="Y",
                   bevel=0.0)
        assign(cl, m_weath)
        paint(cl, grime=0.5)

    bl_ = p.bilge
    z_w = zk_mid + (0.02 + 0.095 * bl_ if bl_ <= 0.4 else lerp(0.058, 0.160, (bl_ - 0.4) / 0.6))
    afloat = z_w > z_floor + 0.03                      # deep enough to float oars and gear
    wv, wf, wst = [], [], []
    prev = None
    wrng = rng.fork("bilge")
    scum = Decal()                                     # pale scum line where the water meets the planking + floating flecks
    up_ = np.array((0.0, 0.0, 1.0))
    for j in range(45):
        t = lerp(tf + 0.001, 0.97, j / 44)
        if zkeel(t) + 0.004 >= z_w:
            prev = None
            continue
        sw_ = s_at_z(t, z_w)
        yw = max(0.004, hull(t, sw_).y + th * 0.8)
        xx = hull(t, sw_).x
        wv += [(xx, -yw, z_w), (xx, yw, z_w)]
        cur = len(wv) - 2
        ws_ = (xx, yw, min(yw * 0.6, wrng.u(0.01, 0.032)), min(yw * 0.6, wrng.u(0.01, 0.032)), t)
        if prev is not None:
            wf.append((prev, cur, cur + 1, prev + 1))
            a_ = wst[-1]
            for sg, kw_ in ((1.0, 2), (-1.0, 3)):
                scum.add([np.array((a_[0], sg * (a_[1] - a_[kw_]), z_w + 0.0015)), np.array((ws_[0], sg * (ws_[1] - ws_[kw_]), z_w + 0.0015)),
                          np.array((ws_[0], sg * ws_[1], z_w + 0.0015)), np.array((a_[0], sg * a_[1], z_w + 0.0015))], up_)
        wst.append(ws_)
        prev = cur
    if wf and bl_ > 0.02:                              # (bilge 0: a dry boat - bare boards and planking, no water in it at all)
        water = mesh_object("bilge_water", wv, wf)
        assign(water, m_water)
        # leaves, sawdust and foam flecks drift on the surface (they draw the water plane even where it only mirrors the dark hull)
        for _ in range(int(36 + 60 * bl_)):
            a_ = wrng.pick(wst[1:-1])
            f = min(1.0, max(0.4, yref(a_[4]) / yr0))
            y = wrng.pick([-1, 1]) * a_[1] * (1 - wrng.u() ** 1.6 * 0.9)
            if abs(y) > a_[1] - 0.03:
                continue
            if not afloat and abs(y) < ((n_fb - 1) / 2 * pitch_b + wid_b / 2) * f + 0.012:
                d_ = (abs(y) / (pitch_b * f) - ((n_fb - 1) / 2 % 1)) % 1.0        # only in the gaps between the boards
                if abs(d_ - 0.5) > 0.07:
                    continue
            ln, wd = wrng.u(0.016, 0.04), wrng.u(0.008, 0.018)
            an = wrng.u(0, math.pi) if afloat or abs(y) > 0.3 * f else wrng.jitter(0.0, 0.25)
            ca, sa = math.cos(an), math.sin(an)
            scum.add([np.array((a_[0] + ca * dx - sa * dy, y + sa * dx + ca * dy, z_w + 0.002))
                      for dx, dy in ((-ln, 0.0), (ln * wrng.u(-0.3, 0.3), -wd), (ln, 0.0), (ln * wrng.u(-0.3, 0.3), wd))], up_)
        sc_ = scum.make("bilge_scum", m_weath)
        if sc_ is not None:
            paint(sc_, wear=1.0, special=0.0, grime=0.0, mode="set")

    # ------------------------------------------------------------------ rowlocks on pads
    for xr in row_x:
        t = t_at_x(xr, 1.0)
        for sg in (1.0, -1.0):
            P = hull(t, 1.0, sg)
            Pn = hull(t + 0.02, 1.0, sg)
            yaw = math.degrees(math.atan2(Pn.y - P.y, Pn.x - P.x))
            zt = P.z + 0.009 + cap_h / 2
            pad = plank("rowlock_pad", 0.25, 0.052, 0.024, loc=(P.x, P.y, zt + 0.011), rot=(0, 0, yaw), bevel=0.006)
            assign(pad, m_oak)
            paint(pad, wear=0.5, grime=0.25)
            zp = zt + 0.023
            parts = [box("rl_plate", (0.09, 0.034, 0.004), loc=(P.x, P.y, zp + 0.002), rot=(0, 0, yaw), bevel=0.001, segments=1),
                     cylinder("rl_boss", 0.0115, 0.012, loc=(P.x, P.y, zp + 0.008), verts=10),
                     cylinder("rl_shank", 0.006, 0.04, loc=(P.x, P.y, zp + 0.03), verts=6)]
            zu = zp + 0.08
            parts.append(sweep("rl_crutch", arc_path(0.03, -230, 50, n=9, plane="XZ", center=(P.x, P.y, zu)), 0.0058, sides=5, smooth_path=False))
            for a in (50, 130):
                parts.append(sphere("rl_tip", 0.0085, loc=(P.x + 0.03 * math.cos(math.radians(a)), P.y, zu + 0.03 * math.sin(math.radians(a))), segments=6, rings=4))
            for o in parts:
                assign(o, m_brass)
                paint(o, special=mask_mul(noise_mask(0.03, seed=seed + 70), 0.6), wear=0.3)

    # ------------------------------------------------------------------ oars: varnished looms, leathers, flat painted blades with a rib
    n_oars = int(p.oars)
    Lo = min(2.75, max(1.8, 0.62 * L))
    orng = rng.fork("oars")
    has_staff = p.cargo == "lantern"
    for io in range(n_oars):
        sg = -1.0 if io == 0 else 1.0
        bl = clamp(0.27 * Lo, 0.6, 0.7)
        xb0 = Lo - bl
        shaft = lathe("oar_loom", [(0.0, 0.0), (0.013, 0.0), (0.0165, 0.004), (0.0165, 0.12), (0.020, 0.14), (0.0225, 0.2), (0.0225, Lo * 0.36),
                                   (0.0195, xb0), (0.017, xb0 + 0.05), (0.0, xb0 + 0.05)], segments=10, rot=(0, 90, 0))
        uv_grain(shaft, "Z")
        assign(shaft, m_oak)
        bloops = []
        for u in (0.0, 0.05, 0.11, 0.2, 0.45, 0.7, 0.9, 0.975, 1.0):
            w = (lerp(0.042, 0.098, smoothstep(0.0, 0.17, u)) + 0.027 * u) * (1 - 0.1 * smoothstep(0.94, 1.0, u))
            e, tf_ = 0.0032, 0.0046
            rwd = lerp(0.015, 0.006, u)
            rh = max(tf_ + 0.0005, lerp(0.0185, 0.004, u ** 0.7))
            ya, yb_ = min(rwd + 0.013, w * 0.42), min(rwd * 0.5, w * 0.2)
            x = xb0 + bl * u
            bloops.append([(x, -w / 2, -e), (x, -ya, -tf_), (x, -yb_, -rh), (x, yb_, -rh), (x, ya, -tf_), (x, w / 2, -e),
                           (x, w / 2, e), (x, ya, tf_), (x, yb_, rh), (x, -yb_, rh), (x, -ya, tf_), (x, -w / 2, e)])
            w_tip = w
        blade = loft("oar_blade", bloops)
        smooth(blade, 32)
        uv_grain(blade, "X")
        assign(blade, m_paint)
        paint(blade, grime=mask_mul(noise_mask(0.15, seed=seed + 82 + io, lo=0.2, hi=0.8), 0.5), special=0.4)
        paint(shaft, wear=lambda co, n: 0.85 * np.clip(1 - co[:, 0] / 0.22, 0, 1), grime=mask_mul(noise_mask(0.2, seed=seed + 84 + io), 0.45))
        xcl = 0.3 * Lo
        collar = cylinder("oar_leather", 0.0255, 0.17, loc=(xcl, 0, 0), rot=(0, 90, 0), verts=10)
        uv_cyl(collar, "Z")
        button = torus("oar_button", 0.028, 0.0085, loc=(xcl - 0.085, 0, 0), rot=(0, 90, 0), major_seg=10, minor_seg=5)
        assign(collar, m_rope)
        assign(button, m_rope)
        paint(collar, grime=0.55)
        tip = box("oar_tip", (0.026, w_tip + 0.005, 0.0105), loc=(Lo - 0.013, 0, 0), bevel=0.0015, segments=1)
        assign(tip, m_brass)
        paint(tip, special=0.5, grime=0.2)
        oar = join([shaft, blade, collar, button, tip], f"oar{io}")
        xh = xs_b - 0.16 + orng.jitter(0.0, 0.05)
        xt_ = xh + Lo
        if afloat:
            # swamped: the oar drifts in the water under the thwarts
            yh = min(0.2, ymold(xh + 0.3, z_w) - 0.12)
            yt = max(0.07, min(0.13, ymold(min(xt_ + 0.25, x_of(0.86)), z_w) - 0.11))
            yaw = math.degrees(math.atan2(sg * (yt - yh), Lo))
            place(oar, loc=(xh + 0.25, sg * yh, z_w - 0.002), rot=(orng.jitter(0.0, 25.0), 0.0, yaw))
        else:
            # handle on the stern sheets, blade over the forward thwart (clear of the lantern staff), converging towards the bow
            yt_lim = ymold(min(xt_, x_of(0.9)), z_tt + 0.03) - 0.105
            yh = min(0.25, ymold(xh, z_tt) - 0.09) + orng.jitter(0.0, 0.015)
            yt = max(0.095, min(0.13, yt_lim)) + orng.jitter(0.0, 0.006)
            if has_staff:
                # lantern boat: stowed blades aft, flat on the stern sheets, so nothing lies round the foot of the staff
                xbt = xs_a + 0.07 + orng.u(0.0, 0.04)                 # blade tip
                yb_ = min(0.2, ymold(xbt + 0.2, z_tt) - 0.12)
                yg = max(0.1, min(0.16, ymold(min(xbt + Lo, x_of(0.9)), z_tt + 0.03) - 0.1))
                yaw = math.degrees(math.atan2(sg * (yb_ - yg), -Lo))
                place(oar, loc=(xbt + Lo, sg * yg, z_tt + 0.0235), rot=(orng.jitter(0.0, 3.0), -0.3, yaw))
            else:
                yaw = math.degrees(math.atan2(sg * (yt - yh), Lo))
                place(oar, loc=(xh, sg * yh, z_tt + 0.0235), rot=(orng.jitter(0.0, 3.0), 0.45, yaw))

    # ------------------------------------------------------------------ bailer (and flotsam when swamped)
    gaps = []
    edges = [xs_b] + [v for (xc, w) in thw for v in (xc - w / 2, xc + w / 2)]
    for j in range(0, len(edges) - 1, 2):
        gaps.append((edges[j], edges[j + 1]))
    drng = rng.fork("gear")
    if p.cargo != "crates" or afloat:
        g_ = gaps[-1] if p.cargo == "crates" else gaps[0]
        bps = [plank("bailer_bottom", 0.22, 0.11, 0.008, loc=(0, 0, 0.004), bevel=0.0),
               plank("bailer_side", 0.22, 0.008, 0.06, loc=(0, 0.051, 0.03), bevel=0.0),
               plank("bailer_side", 0.22, 0.008, 0.06, loc=(0, -0.051, 0.03), bevel=0.0),
               plank("bailer_back", 0.094, 0.06, 0.008, loc=(-0.106, 0, 0.03), axis="Y", rot=(90, 0, 0), bevel=0.0),
               cylinder("bailer_handle", 0.013, 0.13, loc=(-0.165, 0, 0.05), rot=(0, 72, 0), verts=8)]
        for o in bps:
            assign(o, m_oak)
        bailer = join(bps, "bailer")
        paint(bailer, wear=mask_mul(noise_mask(0.08, seed=seed + 66), 0.7), grime=0.3)
        xb_ = lerp(g_[0], g_[1], drng.u(0.35, 0.65))
        if afloat:
            place(bailer, loc=(xb_, max(0.05, min(drng.u(0.12, 0.26), ymold(xb_, z_w) - 0.27)), z_w - 0.022),
                  rot=(drng.jitter(0, 6), drng.jitter(-7, 3), drng.u(0, 360)))
            # a couple of offcuts adrift
            for j in range(2):
                gq = gaps[min(len(gaps) - 1, j + 1)] if len(gaps) > 1 else gaps[0]
                oc = plank("flotsam", drng.u(0.22, 0.4), drng.u(0.05, 0.09), 0.016, bevel=0.003,
                           loc=(lerp(gq[0], gq[1], drng.u(0.3, 0.7)), (0.5 - j) * 2 * drng.u(0.1, 0.22), z_w - 0.004),
                           rot=(drng.jitter(0, 3), drng.jitter(0, 2), drng.u(0, 180)))
                assign(oc, m_weath)
                paint(oc, wear=0.8, grime=0.1)
        else:
            place(bailer, loc=(xb_, drng.pick([-1, 1]) * drng.u(0.12, 0.2), z_floor + 0.001), rot=(0, 0, drng.u(-40, 40)))

    # ------------------------------------------------------------------ rope fenders on the quay side
    for jf in range(nf):
        t = fen_ts[jf]
        P, n = hull(t, 1.0), hull_n(t, 1.0)
        nh = Vector((n.x, n.y, 0)).normalized()
        top = P + nh * (th + 0.03) + Vector((0, 0, -0.03))
        fl_ = 0.26
        zc = top.z - 0.045 - fl_ / 2
        # hangs plumb from the rubbing strake: clear of the flared planking over its whole length
        yb = max(hull(t, s_at_z(t, zc + fl_ * u)).y for u in (-0.5, -0.25, 0.0, 0.25, 0.42)) + th * 2 + 0.052
        fen = lathe("fender", [(0.0, -fl_ / 2), (0.03, -fl_ / 2 + 0.01), (0.048, -fl_ / 2 + 0.05), (0.05, 0.0), (0.048, fl_ / 2 - 0.05), (0.03, fl_ / 2 - 0.012),
                               (0.012, fl_ / 2), (0.0, fl_ / 2)], segments=10, loc=(P.x, yb, zc), rot=(frng.jitter(-6.0, 3.0), frng.jitter(0, 4.0), 0))
        uv_cyl(fen, "Z")
        assign(fen, m_rope)
        paint(fen, grime=mask_mul(height_ramp(zc - fl_ / 2, zc + fl_ / 2), 0.8))
        ln = sweep("fender_line", [(P.x, P.y - 0.03, P.z + 0.022), (P.x, P.y + 0.012, P.z + 0.034), (P.x, P.y + th + 0.036, P.z + 0.0),
                                   (P.x, yb - 0.004, zc + fl_ / 2 + 0.0)], 0.005, sides=5, step=0.03)
        assign(ln, m_rope)

    # ------------------------------------------------------------------ name: transom + both bows
    if nm:
        xa_n = tr_x(zc_n)
        tx = text("name_transom", nm, size=size_t, depth=0.0015, loc=(xa_n - 0.0006, 0, zc_n - size_t * 0.4), rot=(90, 0, -90), font="serif")
        apply_transform(tx)
        keep_front(tx, (-1, 0, 0))
        transform_verts(tx, lambda co: Vector((co.x - slope * (co.z - zc_n), co.y, co.z)))
        assign(tx, m_trim)
        # bows: wrapped on to the sheer strake
        i = N - 1
        Pm = hull(tcn, (S[i] + 1) / 2)
        for sg in (1.0, -1.0):
            tb_ = text("name_bow", nm, size=size_b, depth=0.0012, loc=(Pm.x, 0, 0), rot=(90, 0, 0 if sg < 0 else 180), font="serif")
            apply_transform(tb_)
            keep_front(tb_, (0, -1 if sg < 0 else 1, 0))

            def wrap(co, sg=sg):
                dep = abs(co.y)
                tq = t_at_x(co.x, 0.93)
                D, C, A, Bp, nl, nh = strake_sec(i, tq, sg)
                f = 0.36 + (co.z - size_b * 0.5) / (C - Bp).length
                q = C.lerp(Bp, min(max(f, 0.05), 0.95))
                return q + nl * (dep + 0.0004)

            transform_verts(tb_, wrap)
            assign(tb_, m_paint)

    # ------------------------------------------------------------------ cargo
    if p.cargo == "crates":
        m_burlap = mat("burlap", scale=0.55)             # coarse weave, big enough to read
        m_bottle = mat("glass_clear", "#3f6b3a", opacity=0.8)
        crng = rng.fork("crates")

        def crate(name, w, d, h, loc, yaw, lid=True, fill=None):
            ts_, parts, extra = 0.012, [], []
            sw = (h - 0.02) / 3 - 0.012

            def bx(nm_, size, loc_, ax, bev=0.002, rz=0.0):
                o = box(nm_, size, loc=loc_, rot=(0, 0, rz), bevel=bev, segments=1)
                uv_grain(o, ax)
                parts.append(o)

            for sx in (-1, 1):
                for sy in (-1, 1):
                    bx("post", (0.03, 0.03, h), (sx * (w / 2 - 0.027), sy * (d / 2 - 0.027), h / 2), "Z", bev=0.0)
            for j in range(3):
                z = 0.012 + sw / 2 + j * (sw + 0.012)
                for sy in (-1, 1):
                    bx("side", (w, ts_, sw * crng.u(0.9, 1.0)), (crng.jitter(0, 0.003), sy * (d / 2 - ts_ / 2), z), "X", bev=0.0)
                for sx in (-1, 1):
                    bx("end", (ts_, d - 2 * ts_, sw * crng.u(0.9, 1.0)), (sx * (w / 2 - ts_ / 2), 0, z), "Y", bev=0.0)
            nl_ = 4
            lw = d / nl_ - 0.008
            skip = crng.pick([None, None, 1, 2])
            for j in range(nl_):
                y = -d / 2 + (j + 0.5) * d / nl_
                bx("base", (w - 2 * ts_, lw, ts_), (0, y, ts_ / 2), "X", bev=0.0)
                if lid and j != skip:
                    bx("lid", (w + 0.004, lw, ts_), (crng.jitter(0, 0.004), y, h + ts_ / 2), "X", rz=crng.jitter(0, 0.8))
            for o in parts:
                assign(o, m_weath)
            wi, di = w - 2 * ts_ - 0.012, d - 2 * ts_ - 0.012
            if fill == "bottles":
                # wine bottles stood in straw sacking, necks proud of the rim
                bt = lathe("bottle", [(0.036, 0.0), (0.037, 0.15), (0.031, 0.185), (0.015, 0.215), (0.0125, 0.27), (0.016, 0.272), (0.016, 0.284), (0.0, 0.284)],
                           segments=7)
                nx_, ny_ = max(2, int(wi / 0.082)), max(2, int(di / 0.082))
                ms = []
                for a_ in range(nx_):
                    for b_ in range(ny_):
                        if crng.chance(0.12):
                            continue
                        ms.append(trs(((a_ + 0.5) / nx_ * wi - wi / 2, (b_ + 0.5) / ny_ * di - di / 2, ts_ + crng.u(0.0, 0.012)),
                                      (crng.jitter(0, 3), crng.jitter(0, 3), crng.u(0, 360))))
                bts = instances_to_mesh("bottles", bt, ms, keep_base=False)
                smooth(bts, 50)
                assign(bts, m_bottle)
                extra.append(bts)
                pk = box("packing", (wi, di, h * 0.55), loc=(0, 0, ts_ + h * 0.275), bevel=0.0)
                assign(pk, m_burlap)
                extra.append(pk)
            elif fill == "mound":
                # heaped produce under a sacking cover, bulging through the open top
                pm = [crng.u(0, 6.28) for _ in range(4)]

                def fn(u, v):
                    ed = min(1.0, 4.5 * min(u, 1 - u)) * min(1.0, 4.5 * min(v, 1 - v))
                    lump = 0.5 + 0.25 * math.sin(u * 9.0 + pm[0]) * math.sin(v * 7.0 + pm[1]) + 0.2 * math.sin(u * 17.0 + pm[2]) * math.cos(v * 13.0 + pm[3])
                    return ((u - 0.5) * wi, (v - 0.5) * di, h * 0.7 + 0.065 * ed ** 0.6 * (0.55 + lump * 0.6))
                md = surface("produce", fn, nu=9, nv=7)
                smooth(md, 70)
                assign(md, m_burlap)
                paint(md, grime=mask_mul(noise_mask(0.06, seed=seed + 93), 0.5))
                extra.append(md)
            else:
                pk = box("packing", (wi, di, h - 0.03), loc=(0, 0, ts_ + (h - 0.03) / 2), bevel=0.0)
                assign(pk, m_burlap)
                paint(pk, grime=0.5)
                extra.append(pk)
            c = join(parts, name)
            paint(c, grime=mask_mul(noise_mask(0.2, seed=seed + 90), 0.5), wear=mask_mul(noise_mask(0.1, seed=seed + 91), 0.4))
            place([c] + extra, loc=loc, rot=(0, 0, yaw))
            return c

        def sack(name, loc, yaw, hx, hy):
            """Filled sack lying on a crate of half extents (hx, hy): seamed bottom with ears, sagging over the edges,
            neck gathered into pleats and tied with twine."""
            st = [(-0.30, 0.150, 0.008), (-0.278, 0.172, 0.05), (-0.20, 0.186, 0.088), (-0.07, 0.19, 0.10), (0.06, 0.186, 0.097), (0.15, 0.168, 0.086),
                  (0.215, 0.118, 0.066), (0.258, 0.05, 0.04), (0.28, 0.04, 0.034), (0.30, 0.06, 0.048), (0.325, 0.08, 0.06), (0.332, 0.05, 0.036)]
            nr = 12
            ph_ = [crng.u(0, 6.28) for _ in range(4)]
            lps = []
            for (x, a_, b_) in st:
                fold = smoothstep(0.08, 0.23, x)
                ringp = []
                for q in range(nr):
                    an = 2 * math.pi * q / nr
                    c_, s_ = math.cos(an), math.sin(an)
                    rr_ = 1 + 0.06 * math.sin(3 * an + ph_[0] + x * 9) * (1 - fold) + 0.17 * fold * (1 if q % 2 else -1)
                    lump = (0.012 * math.sin(x * 21 + ph_[2] + 2 * an) + 0.009 * math.sin(x * 37 + ph_[3] - an)) * (1 - fold)
                    y = (a_ * rr_ + lump) * math.copysign(abs(c_) ** 0.8, c_)
                    z = (b_ * rr_ + lump) * (s_ if s_ > 0 else 0.5 * s_) + 0.5 * b_
                    z -= 0.8 * max(0.0, abs(x) - hx) ** 1.15 + 0.6 * max(0.0, abs(y) - hy) ** 1.1
                    ringp.append((x, y, z))
                lps.append(ringp)
            sk = loft(name, lps)
            smooth(sk, 55)
            assign(sk, m_burlap)
            paint(sk, grime=mask_mul(noise_mask(0.09, seed=seed + 95, lo=0.3, hi=0.8), 0.45), special=0.3)
            xn = 0.27
            zn = 0.5 * 0.036 - 0.8 * max(0.0, xn - hx) ** 1.15
            tie = [torus("sack_tie", 0.04, 0.0055, loc=(xn, 0, zn + 0.004), rot=(0, 90, 0), major_seg=10, minor_seg=5),
                   sweep("sack_tie_end", [(xn, 0.03, zn + 0.02), (xn + 0.012, 0.055, zn - 0.01), (xn + 0.006, 0.062, zn - 0.06)], 0.004, sides=5),
                   sweep("sack_tie_end", [(xn, 0.036, zn + 0.01), (xn - 0.015, 0.066, zn - 0.02), (xn - 0.01, 0.07, zn - 0.045)], 0.004, sides=5)]
            for o in tie:
                assign(o, m_rope)
            place([sk] + tie, loc=loc, rot=(0, 0, yaw))

        zfl = z_floor + 0.002
        g0 = gaps[0]
        w0 = min(0.52, (g0[1] - g0[0]) - 0.16)
        x0c = (g0[0] + g0[1]) / 2 + crng.jitter(0.0, 0.03)
        h0 = 0.30
        hb = min(0.21, z_tt - zfl - 0.03)
        hd = 0.21
        yd = crng.jitter(0, 0.015)
        crate("crate_a", w0, 0.34, h0, (x0c, crng.jitter(0, 0.02), zfl), crng.jitter(0.0, 6.0))
        # a second, smaller crate stacked askew on the first: stands proud of the gunwale
        wd_ = min(0.38, w0 - 0.08)
        crate("crate_d", wd_, 0.3, hd, (x0c - 0.03, yd, zfl + h0 + 0.013), crng.pick([-1, 1]) * crng.u(7.0, 14.0))
        if len(gaps) > 1:
            g1 = gaps[1]
            w1 = min(0.42, (g1[1] - g1[0]) - 0.14)
            crate("crate_b", w1, 0.3, hb, ((g1[0] + g1[1]) / 2, crng.jitter(0, 0.02), zfl + 0.004), crng.jitter(0.0, 9.0), lid=False, fill="bottles")
        crate("crate_c", 0.26, 0.3, 0.17, (xs_a + 0.15 * k, crng.jitter(0, 0.03), z_tt), crng.jitter(0.0, 8.0), lid=False, fill="mound")
        # a tied sack slumped over the stack
        zs0 = zfl + h0 + 0.013 + hd + 0.012
        sack("sack", (x0c - 0.03 + crng.jitter(0, 0.02), yd, zs0 + 0.002), crng.jitter(0.0, 12.0) + crng.pick([0, 180]), wd_ / 2, 0.15)

    if p.cargo == "lantern":
        m_glass = mat("glass_lantern", glow=0.5, opacity=0.22)
        m_flame = mat("emissive", "#ffb45e", glow=5.5)
        lrng = rng.fork("lantern")
        xc, w = thw[-1]
        tq = (xc - xT) / (xF - xT)
        zf = zkeel(tq) + 0.012
        zs = z_tt + 0.80
        # ash staff stepped on the hog, through the forward thwart; tarred, with a dull lashed-on crook
        staff = cylinder("lan_staff", 0.019, zs - zf, loc=(xc, 0, (zs + zf) / 2), verts=10, radius2=0.015)
        uv_grain(staff, "Z")
        smooth(staff, 50)
        assign(staff, m_oak)
        # sooted and hand-darkened towards the lantern, so the staff does not flare beside the flame
        paint(staff, grime=lambda co, n: np.maximum(bilge_grime(co, n), 0.55 + 0.45 * np.clip((co[:, 2] - z_tt - 0.1) / 0.3, 0, 1)), mode="set")
        metal = [lathe("lan_flange", [(0.019, 0.0), (0.036, 0.0), (0.036, 0.004), (0.026, 0.006), (0.023, 0.02), (0.018, 0.022)], segments=12, loc=(xc, 0, z_tt)),
                 lathe("lan_ferrule", [(0.0155, -0.07), (0.019, -0.068), (0.019, -0.05), (0.0175, -0.046), (0.0175, -0.004), (0.021, 0.0), (0.021, 0.008),
                                       (0.012, 0.016), (0.0, 0.018)], segments=12, loc=(xc, 0, zs))]
        reach = 0.27
        cr = [(0, 0, zs - 0.01), (0, 0, zs + 0.05)]
        cr += [(reach / 2 - reach / 2 * math.cos(a), 0, zs + 0.05 + reach / 2 * math.sin(a)) for a in np.radians(np.arange(20, 181, 20))]
        cr += [(reach, 0, zs + 0.01)]
        hr = 0.018
        cr += [(reach - hr + hr * math.cos(a), 0, zs + 0.01 + hr * math.sin(a)) for a in np.radians([-35, -75, -115, -155, -195, -235])]
        crook = sweep("lan_crook", [(xc + x, y, z) for x, y, z in cr], 0.006, sides=6, smooth_path=False,
                      scale_fn=lambda u: 1.0 if u < 0.8 else 1.0 - 0.45 * (u - 0.8) / 0.2)
        metal.append(crook)
        for o in metal:
            assign(o, m_brass)
        # aged, sooty brass: the ferrule sits a hand's breadth from the flame and must not mirror it
        paint(metal[0], special=mask_mul(noise_mask(0.05, seed=seed + 98), 0.6), grime=0.4)
        paint(metal[1], grime=1.0, special=0.35, mode="set")
        paint(crook, grime=0.85, special=mask_mul(noise_mask(0.05, seed=seed + 98), 0.5), mode="set")
        hx, hz = xc + reach - hr, zs + 0.01 - hr + 0.006      # where the bail hangs on the hook
        bail_h = 0.15
        lz = hz - 0.2 - bail_h - 0.0028
        lx, ly = hx, 0.0
        font = lathe("lan_font", [(0.0, 0.0), (0.060, 0.0), (0.064, 0.004), (0.064, 0.012), (0.058, 0.018), (0.062, 0.03), (0.062, 0.05), (0.05, 0.062),
                                  (0.03, 0.066), (0.026, 0.08), (0.032, 0.083), (0.032, 0.09), (0.0, 0.09)], segments=14)
        globe = lathe("lan_globe", [(0.03, 0.09), (0.045, 0.105), (0.056, 0.14), (0.053, 0.18), (0.037, 0.21), (0.03, 0.222)], segments=16)
        flame = lathe("lan_flame", [(0.0, 0.099), (0.007, 0.105), (0.0095, 0.116), (0.005, 0.132), (0.0, 0.146)], segments=8)
        wick = cylinder("lan_wick", 0.011, 0.012, loc=(0, 0, 0.094), verts=8)
        top = lathe("lan_top", [(0.03, 0.22), (0.041, 0.224), (0.041, 0.232), (0.031, 0.24), (0.027, 0.262), (0.045, 0.268), (0.047, 0.274),
                                (0.022, 0.29), (0.009, 0.297), (0.0, 0.297)], segments=14)
        metal = [font, wick, top]
        for sx in (-1, 1):
            metal.append(sweep("lan_tube", [(sx * 0.058, 0, 0.05), (sx * 0.074, 0, 0.09), (sx * 0.076, 0, 0.2), (sx * 0.06, 0, 0.245), (sx * 0.036, 0, 0.262)], 0.005, sides=6, smooth_path=False))
        for z in (0.125, 0.178):
            metal.append(torus("lan_hoop", 0.059, 0.0018, loc=(0, 0, z), major_seg=16, minor_seg=4))
        for a in (45, 135, 225, 315):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            metal.append(sweep("lan_wire", [(0.05 * ca, 0.05 * sa, 0.092), (0.06 * ca, 0.06 * sa, 0.14), (0.058 * ca, 0.058 * sa, 0.18), (0.04 * ca, 0.04 * sa, 0.222)], 0.0016, sides=4, smooth_path=False))
        hp = []
        for j in range(13):
            a = math.radians(4 + 172 * j / 12)
            hp.append((0, 0.078 * math.cos(a), 0.2 + bail_h * math.sin(a) ** 0.8))
        metal.append(sweep("lan_bail", hp, 0.0028, sides=5, smooth_path=False))
        for o in metal:
            assign(o, m_brass)
        assign(globe, m_glass)
        assign(flame, m_flame)
        body = join(metal, "lantern_body")
        # (painted at the origin, before placing) burner, cage and chimney are sooted dark: they sit centimetres from
        # the flame and would otherwise burn out; the font below keeps a little tarnished brass
        paint(body, grime=lambda co, n: np.where(co[:, 2] > 0.06, 0.95, 0.45), special=mask_mul(noise_mask(0.04, seed=seed + 99), 0.5), mode="set")
        tiltl = lrng.jitter(0.0, 2.5)
        for o in (body, globe, flame):
            place(o, loc=(lx, ly, lz), rot=(tiltl, 0, 0))
        light("lantern", (lx, ly, lz + 0.125), color="#ffbf6a", intensity=p.intensity, range=6.5, flicker=0.3, shadow_near=0.1)
