"""One storey-high bay of half-timbered (oak box-frame) wall. A sill beam, posts and wall plate carry
rails, studs and braces in one of five panel patterns; every timber is hewn (adze facets, waney
arrises, drying shakes), cut to fit between its neighbours and draw-pegged with proud riven pegs.
The infill is triangulated exactly against the timber outlines: lime plaster set back 15-25 mm with
a small irregular fillet against the oak, a shallow pillow, broken patches showing daub and laths
and re-limed repairs - or brick nogging of individual bricks in level / diagonal coursing on a lime
bed. Weathering is placed, not random: rising damp from the sill, run-off streaks under rails and
plate, bleached sill and post feet, tannin stains under the ironwork. Optional framed window / door
opening, stone plinth, forged fishtail L-straps and a jetty: joists with bull-nosed ends, moulded
bressummer with a sunk lozenge frieze and ogee brackets on the posts.
Origin bottom centre on the wall plane (front face y = 0, 0.18 m thick towards +Y).
"""
from tk import *
import math
import numpy as np
import bmesh
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt

META = {"name": "Half-timbered wall bay", "class": "component", "mount": "wall_ground", "focus": 0.55}
PARAMS = {
    "width": {"default": 3.2, "min": 1.2, "max": 6.0, "desc": "bay width (m), spans x in [-width/2, width/2]"},
    "height": {"default": 2.9, "min": 2.2, "max": 4.2, "desc": "storey height (m), spans z in [0, height]"},
    "pattern": {"default": "curved_braces", "choices": ["close_studding", "square_panels", "herringbone", "st_andrew", "curved_braces"],
                "desc": "panel pattern of the secondary timbers"},
    "plaster": {"default": "#e8dcc0", "desc": "limewash tint of the infill panels"},
    "timber": {"default": "#66503e", "desc": "timber tone (stain on the oak)"},
    "jetty": {"default": False, "desc": "jetty at the top: joists, moulded bressummer and carved brackets projecting to -Y"},
    "jetty_depth": {"default": 0.4, "min": 0.25, "max": 0.6, "desc": "projection of the jetty (m)"},
    "irregularity": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "hand-hewn waviness, leaning studs, uneven sections"},
    "posts": {"default": "both", "choices": ["both", "left", "right", "none"], "desc": "which end posts this bay owns (avoid doubled posts when tiling bays)"},
    "timber_size": {"default": 0.2, "min": 0.15, "max": 0.28, "desc": "face width of the main frame timbers (m)"},
    "infill": {"default": "plaster", "choices": ["plaster", "brick"], "desc": "panel infill: pillowed lime plaster or brick nogging"},
    "recess": {"default": 0.02, "min": 0.012, "max": 0.03, "desc": "set-back of the infill behind the studs (m)"},
    "opening_w": {"default": 0.0, "min": 0.0, "max": 2.4, "desc": "framed opening width (0 = none)"},
    "opening_h": {"default": 1.25, "min": 0.4, "max": 2.6, "desc": "framed opening height (m)"},
    "opening_sill": {"default": 0.95, "min": 0.0, "max": 2.2, "desc": "opening sill height above z = 0 (0 = doorway, cuts the sill beam)"},
    "opening_x": {"default": 0.0, "min": -2.5, "max": 2.5, "desc": "opening centre offset along x (m)"},
    "plinth": {"default": 0.0, "min": 0.0, "max": 0.8, "desc": "height of a dressed stone plinth under the sill beam (0 = none)"},
    "straps": {"default": True, "desc": "forged iron L-straps nailed over the post joints"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: stains, damp, bleached and chipped surfaces"},
    "mount": {"default": "ground", "choices": ["ground", "wall_ground"],
              "desc": "ground = free-standing (review renders: the bay IS the wall, its body lies behind y = 0, so the "
                      "review backing wall of wall_ground would cover it); wall_ground = set against a building core"},
}
VARIANTS = {
    "jettied_herringbone": {"pattern": "herringbone", "jetty": True, "width": 3.6, "height": 3.1, "plaster": "#dcb878",
                            "timber": "#4a3424", "seed": 4},
    "window_cross": {"pattern": "st_andrew", "opening_w": 0.95, "opening_h": 1.2, "opening_sill": 1.05, "width": 2.9,
                     "plaster": "#efe9dc", "timber": "#3d2c20", "seed": 8},
    "brick_studded": {"pattern": "close_studding", "infill": "brick", "plinth": 0.4, "height": 3.3, "width": 2.4,
                      "timber": "#57493d", "irregularity": 0.3, "seed": 13},
    "crooked_squares": {"pattern": "square_panels", "irregularity": 1.0, "width": 4.4, "height": 2.6, "plaster": "#d9a58c",
                        "timber": "#5a4534", "age": 0.9, "jetty": True, "jetty_depth": 0.3, "seed": 21},
}

D = 0.18        # wall thickness towards +Y
DB = D - 0.002  # back of the timbers (the inside plaster sheet covers them)
OV = 0.008      # how far a tenoned timber is let into its host (hides the joint line)
CH = 0.009      # eased arris of hewn timber


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _mix(h1, h2, t):
    a, b = _hex(h1), _hex(h2)
    return "#" + "".join("%02x" % int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def _wob(r, n, amp):
    """Smooth 1D random wander in [-amp, amp] over t in 0..1."""
    ks = [r.u(-1.0, 1.0) for _ in range(n + 2)]

    def f(t):
        x = min(max(t, 0.0), 1.0) * n
        i = min(int(x), n - 1)
        fr = x - i
        fr = fr * fr * (3 - 2 * fr)
        return amp * (ks[i] * (1 - fr) + ks[i + 1] * fr)
    return f


def _dense(pts, step):
    a = np.array(pts, dtype=float)
    seg = np.linalg.norm(a[1:] - a[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(s[-1] / step) + 1)
    t = np.linspace(0.0, s[-1], n)
    P = np.stack([np.interp(t, s, a[:, 0]), np.interp(t, s, a[:, 1])], axis=1)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    return P, T


def _run(mask):
    best = (0, -1)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j + 1
        else:
            i += 1
    return best


def _qbez(p0, c, p2, n=24):
    out = []
    for i in range(n + 1):
        t = i / n
        out.append(((1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * c[0] + t * t * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * c[1] + t * t * p2[1]))
    return out


def _cbez(p0, p1, p2, p3, n=10):
    out = []
    for i in range(n + 1):
        t = i / n
        k = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3)
        out.append((k[0] * p0[0] + k[1] * p1[0] + k[2] * p2[0] + k[3] * p3[0],
                    k[0] * p0[1] + k[1] * p1[1] + k[2] * p2[1] + k[3] * p3[1]))
    return out


def _extend(pts, e):
    def unit(a, b):
        dx, dz = b[0] - a[0], b[1] - a[1]
        l = math.hypot(dx, dz) or 1.0
        return dx / l, dz / l
    u0 = unit(pts[1], pts[0])
    u1 = unit(pts[-2], pts[-1])
    return [(pts[0][0] + u0[0] * e, pts[0][1] + u0[1] * e)] + list(pts) + [(pts[-1][0] + u1[0] * e, pts[-1][1] + u1[1] * e)]


def _prism_x(name, poly_yz, x0, x1):
    """Extrude a (y, z) polygon across X."""
    n = len(poly_yz)
    verts = [(x0, y, z) for y, z in poly_yz] + [(x1, y, z) for y, z in poly_yz]
    faces = [list(range(n))[::-1], [n + i for i in range(n)]]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return recalc_normals(mesh_object(name, verts, faces))


def _dedup(seq):
    out = []
    for i in seq:
        if not out or out[-1] != i:
            out.append(i)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _inpoly(px, pz, poly):
    """Vectorised point-in-polygon (even-odd)."""
    px, pz = np.asarray(px, dtype=float), np.asarray(pz, dtype=float)
    ins = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        if z0 == z1:
            continue
        cond = (z0 > pz) != (z1 > pz)
        xi = x0 + (pz - z0) * (x1 - x0) / (z1 - z0)
        ins ^= cond & (px < xi)
    return ins


def _blob(r, cx, cz, rad, n=12, sx=1.0):
    out = []
    for k in range(n):
        a = (k + r.u(-0.3, 0.3)) / n * 2 * math.pi
        q = rad * r.u(0.62, 1.2)
        out.append((cx + math.cos(a) * q * sx, cz + math.sin(a) * q))
    return out


def _scaled(poly, c, s):
    return [(c[0] + (x - c[0]) * s, c[1] + (z - c[1]) * s) for x, z in poly]


class Frame:
    """Collects the timbers (hewn beams in the wall plane), their peg positions, their footprints
    (for the infill) and the constraint lines the infill is triangulated against."""

    def __init__(self, irr, rng, oak, endm, oak2=None):
        self.irr, self.rng, self.oak, self.endm, self.oak2 = irr, rng, oak, endm, oak2 or oak
        self.objs, self.segs, self.pegs, self.n = [], [], [], 0
        self.lines, self.rings, self.shk, self.drips, self.g0 = [], [], [], [], {}

    def member(self, pts, w, y0=0.0, clip=None, pegs=(0, 0), peg_y=0.0, endgrain=False, wav=(1.0, 1.0),
               name="timber", spacing=0.45, alt=0.0, mid=True, fine=0.16, wane=0.5, shake=0.0, material=None):
        """Beam of face width w along the (x, z) path. clip = (x0, x1, z0, z1): cut each end where the
        timber meets the surrounding members (mitres, with the corner let in where two hosts meet).
        The edges are straight between coarse knots (so the infill can follow them exactly); between
        the knots the face is hewn: adze facets, wind, waney arrises and the odd drying shake."""
        r = self.rng.fork("m%d" % self.n)
        self.n += 1
        irr = self.irr
        step = 0.003
        P, T = _dense(pts, step)
        Nn = np.stack([-T[:, 1], T[:, 0]], axis=1)
        hw = w / 2.0
        E = [P + hw * Nn, P - hw * Nn]
        idx = []
        for e in E:
            if clip is None:
                idx.append((0, len(P) - 1))
            else:
                x0, x1, z0, z1 = clip
                mk = (e[:, 0] >= x0 - OV) & (e[:, 0] <= x1 + OV) & (e[:, 1] >= z0 - OV) & (e[:, 1] <= z1 + OV)
                i0, i1 = _run(mk)
                if i1 - i0 < 14:
                    return None
                idx.append((i0, i1))
        lenmax = max(i1 - i0 for i0, i1 in idx) * step
        nk = max(2, int(round(lenmax / spacing)) + 1)
        m = max(1, int(round(lenmax / (nk - 1) / fine)))
        nst = (nk - 1) * m + 1
        wl0 = _wob(r, nk, irr * 0.0055 * wav[0])
        wr0 = _wob(r, nk, irr * 0.0055 * wav[1])
        tl, tr_ = r.u(-1, 1) * irr * 0.07 * hw * wav[0], r.u(-1, 1) * irr * 0.07 * hw * wav[1]
        K = [[], []]
        for i in range(nk):
            u = i / (nk - 1)
            for s, (i0, i1) in enumerate(idx):
                f = i0 + u * (i1 - i0)
                k = min(int(f), len(P) - 2)
                q = f - k
                e = E[s][k] * (1 - q) + E[s][k + 1] * q
                off = (wl0(u) + tl * (2 * u - 1)) if s == 0 else -(wr0(u) + tr_ * (2 * u - 1))
                K[s].append(e + Nn[k] * off)

        front = y0 < 1e-6
        hew = 0.002 + 0.006 * irr
        nw = max(2, nst // 2)
        wy = _wob(r, nw, irr * 0.0016)
        wt = _wob(r, nw, irr * 0.0018)
        cl = _wob(r, nw, 0.75)
        cr = _wob(r, nw, 0.75)
        wm = _wob(r, nw, 0.14)
        wc = [_wob(r, nw, 1.0) for _ in range(3)]
        dy = r.u(0.0, irr * 0.002)

        def bumps():
            out = []
            nb = int(lenmax * wane * r.u(0.4, 1.3) + r.u(0.0, 0.9))
            for _ in range(nb):
                out.append((r.u(0.06, 0.94), r.u(0.1, 0.26) / max(lenmax, 0.1), r.u(0.012, 0.028) * (0.5 + 0.7 * irr)))
            return out

        def wane_at(bs, u):
            c = 0.0
            for uc, hl, amp in bs:
                t = abs(u - uc) / hl
                if t < 1.0:
                    c += amp * (1 - t * t) ** 2
            return c

        bl, br = bumps(), bumps()
        sk = None
        if mid and shake > 0 and lenmax > 0.7 and r.chance(shake):
            ls = min(r.u(0.4, 1.0), lenmax * 0.7) / lenmax
            u0 = r.u(0.08, 0.92 - ls)
            sk = (u0, ls, r.u(0.0028, 0.0045), r.u(0.008, 0.013))

        verts, rows, yfs = [], [], []

        def add(q, yy):
            verts.append((float(q[0]), float(yy), float(q[1])))
            return len(verts) - 1

        for j in range(nst):
            u = j / (nst - 1)
            i = min(j // m, nk - 2)
            t = (j - i * m) / m
            L = K[0][i] * (1 - t) + K[0][i + 1] * t
            R = K[1][i] * (1 - t) + K[1][i + 1] * t
            wd = max(float(np.linalg.norm(R - L)), 1e-6)
            inn = (R - L) / wd
            yf = y0 + dy + wy(u) + (hew if front else 0.0)
            tw = wt(u)
            a = [hew * (0.55 * wc[k](u) + 0.45 * r.u(-1, 1)) for k in range(3)]
            c1 = min(CH * (1.0 + (0.3 + 0.7 * irr) * cl(u)) + wane_at(bl, u), 0.3 * wd)
            c2 = min(CH * (1.0 + (0.3 + 0.7 * irr) * cr(u)) + wane_at(br, u), 0.3 * wd)
            ymin = 0.0 if front else y0 - 0.005
            yl, ym, yr = max(yf + tw + a[0], ymin), max(yf + a[1], ymin), max(yf - tw + a[2], ymin)
            row = [add(L, DB), add(L, yl + c1), add(L + c1 * inn, yl)]
            if mid:
                tm = 0.5 + wm(u) + r.u(-0.03, 0.03)
                Mp = (L + c1 * inn) * (1 - tm) + (R - c2 * inn) * tm
                kk = 0.0
                if sk and sk[0] < u < sk[0] + sk[1]:
                    kk = math.sin(math.pi * (u - sk[0]) / sk[1]) ** 0.6
                if kk > 0.15:
                    row += [add(Mp - sk[2] * kk * inn, ym), add(Mp, ym + sk[3] * kk), add(Mp + sk[2] * kk * inn, ym)]
                    self.shk.append((float(Mp[0]), float(ym + sk[3] * kk), float(Mp[1])))
                else:
                    im = add(Mp, ym)
                    row += [im, im, im] if sk else [im]
            row += [add(R - c2 * inn, yr), add(R, yr + c2), add(R, DB)]
            rows.append(row)
            yfs.append(yf)
        faces = []
        nr = len(rows[0])
        for j in range(nst - 1):
            for k in range(nr - 1):
                f = _dedup([rows[j][k], rows[j][k + 1], rows[j + 1][k + 1], rows[j + 1][k]])
                if len(f) >= 3:
                    faces.append(tuple(f))

        def side_of(pt, clipped):
            if clip is None or not clipped:
                return None
            x0, x1, z0, z1 = clip
            c = [("x0", abs(pt[0] - (x0 - OV))), ("x1", abs(pt[0] - (x1 + OV))), ("z0", abs(pt[1] - (z0 - OV))), ("z1", abs(pt[1] - (z1 + OV)))]
            s, d = min(c, key=lambda t: t[1])
            return s if d < 0.008 else None

        EXP = 0.0003
        inK = []
        for i in range(nk):
            d = K[1][i] - K[0][i]
            inK.append(d / max(float(np.linalg.norm(d)), 1e-6))
        eL = [(float(q[0]), float(q[1])) for q in K[0]]
        eR = [(float(q[0]), float(q[1])) for q in K[1]]
        xL = [tuple((K[0][i] - inK[i] * EXP).tolist()) for i in range(nk)]
        xR = [tuple((K[1][i] + inK[i] * EXP).tolist()) for i in range(nk)]
        self.lines += [xL, xR]
        for s, pl in ((0, K[0]), (1, K[1])):
            if wav[s] > 0:
                sg = -1.0 if s == 0 else 1.0
                self.rings.append([tuple((pl[i] + sg * inK[i] * r.u(0.0055, 0.009)).tolist()) for i in range(nk)])

        for e in (0, 1):
            st = 0 if e == 0 else nst - 1
            ke = 0 if e == 0 else nk - 1
            ur = _dedup(rows[st])
            iL = idx[0][e]
            iR = idx[1][e]
            sL = side_of(E[0][iL], iL > 0 if e == 0 else iL < len(P) - 1)
            sR = side_of(E[1][iR], iR > 0 if e == 0 else iR < len(P) - 1)
            vL, vR = K[0][ke], K[1][ke]
            corner = sL and sR and sL != sR and sL[0] != sR[0]
            tin = T[idx[0][0]] if e == 0 else -T[idx[0][1]]
            if corner:
                x0, x1, z0, z1 = clip
                val = {"x0": x0 - OV, "x1": x1 + OV, "z0": z0 - OV, "z1": z1 + OV}
                kx = val[sL] if sL[0] == "x" else val[sR]
                kz = val[sL] if sL[0] == "z" else val[sR]
                kf = add((kx, kz), yfs[st])
                kb = add((kx, kz), DB)
                self.segs.append((eL[ke], (kx, kz), (kx, kz), eR[ke]))
                self.lines.append([xL[ke], (kx, kz), xR[ke]])
                for k in range(1, len(ur) - 2):
                    faces.append((ur[k], ur[k + 1], kf))
                faces += [(ur[0], ur[1], kf, kb), (ur[-2], ur[-1], kb, kf), (ur[-1], ur[0], kb)]
                if pegs[e]:
                    q = (vL + vR + np.array([kx, kz])) / 3 + tin * 0.075
                    self.pegs.append((float(q[0]), float(q[1]), y0 + 0.002))
            else:
                faces.append(tuple(ur))
                self.lines.append([xL[ke], xR[ke]])
                if pegs[e]:
                    c0 = (vL + vR) / 2
                    if sL and sL == sR:
                        dr = np.array({"x0": (-1, 0), "x1": (1, 0), "z0": (0, -1), "z1": (0, 1)}[sL], dtype=float)
                    else:
                        dr = -tin
                    pos = c0 + dr * (0.036 + r.u(-0.004, 0.006))
                    offs = [0.0] if pegs[e] == 1 else [-0.21, 0.21]
                    for o in offs:
                        q = pos + (vR - vL) * (o + r.u(-0.03, 0.03))
                        self.pegs.append((float(q[0]), float(q[1]), peg_y))
        o = recalc_normals(mesh_object("%s%d" % (name, self.n), verts, faces))
        d = P[-1] - P[0]
        d = d / max(np.linalg.norm(d), 1e-6)
        dx, dz = float(d[0]), float(d[1])
        uv_grain(o, axis=(dx, 0.0, dz))
        assign(o, material or (self.oak2 if (alt > 0 and r.chance(alt)) else self.oak))
        if endgrain:
            assign_faces(o, self.endm, lambda c, n: abs(n.x * dx + n.z * dz) > 0.92)
        for i in range(nk - 1):
            self.segs.append((eL[i], eL[i + 1], eR[i + 1], eR[i]))
        self.g0[o.name] = r.u(0.0, 1.0)
        self.objs.append(o)
        return o

    def near(self, X, Z):
        """Signed distance of points to the nearest timber edge (negative under a timber) and the
        closest point on that edge. Works on the real footprints (strip quads) of the members."""
        sh = X.shape
        x, z = X.ravel()[None, :], Z.ravel()[None, :]
        n = x.shape[1]
        ar = np.arange(n)
        bd, bx, bz = np.full(n, 1e9), np.zeros(n), np.zeros(n)
        Q = np.array(self.segs, dtype=float)                    # (q, 4, 2): L0, L1, R1, R0

        def segd(A, B):
            vx, vz = (B[:, 0] - A[:, 0])[:, None], (B[:, 1] - A[:, 1])[:, None]
            ax, az = A[:, 0][:, None], A[:, 1][:, None]
            t = np.clip(((x - ax) * vx + (z - az) * vz) / (vx * vx + vz * vz + 1e-12), 0.0, 1.0)
            px, pz = ax + t * vx, az + t * vz
            return np.hypot(x - px, z - pz), px, pz

        for c in range(0, len(Q), 96):
            q = Q[c:c + 96]
            dl, lx, lz = segd(q[:, 0], q[:, 1])
            dr, rx, rz = segd(q[:, 3], q[:, 2])
            pos = np.ones((len(q), n), dtype=bool)
            neg = np.ones((len(q), n), dtype=bool)
            for k in range(4):
                A, B = q[:, k], q[:, (k + 1) % 4]
                cr = (B[:, 0] - A[:, 0])[:, None] * (z - A[:, 1][:, None]) - (B[:, 1] - A[:, 1])[:, None] * (x - A[:, 0][:, None])
                pos &= cr >= 0
                neg &= cr <= 0
            ul = dl <= dr
            d = np.where(ul, dl, dr)
            d = np.where(pos | neg, -d, d)
            k = np.argmin(d, axis=0)
            dk = d[k, ar]
            up = dk < bd
            ulk = ul[k, ar]
            bd = np.where(up, dk, bd)
            bx = np.where(up, np.where(ulk, lx[k, ar], rx[k, ar]), bx)
            bz = np.where(up, np.where(ulk, lz[k, ar], rz[k, ar]), bz)
        return bd.reshape(sh), bx.reshape(sh), bz.reshape(sh)

    def dist(self, X, Z):
        return self.near(X, Z)[0]


def _lines(a, b, cell, extra=()):
    n = max(1, int(math.ceil((b - a) / cell)))
    v = [a + (b - a) * i / n for i in range(n + 1)]
    for e in extra:
        if a + 1e-4 < e < b - 1e-4:
            if len(v) > 2:
                k = min(range(1, len(v) - 1), key=lambda i: abs(v[i] - e))
                if abs(v[k] - e) < cell * 0.45:
                    v[k] = e
                    continue
            v.append(e)
    return sorted(set(v))


def build(p, rng):
    W, H = p.width, p.height
    irr, age = p.irregularity, p.age
    ts = p.timber_size
    oak = mat("wood_oak", p.timber)
    oak2 = mat("wood_oak", _mix(p.timber, "#8f8478", 0.32))
    sillm = mat("wood_weathered", _mix(p.timber, "#d6d0c4", 0.55))
    endm = mat("wood_endgrain", _mix(p.timber, "#a88a66", 0.4))
    fr = Frame(irr, rng.fork("frame"), oak, endm, oak2)

    pw = ts                      # posts
    sill_h = ts
    plate_h = ts * 0.9
    sw = ts * 0.72               # studs / jambs
    rw = ts * 0.76               # rails
    dw = ts * 0.58               # herringbone / cross timbers
    bw = ts * 0.85               # curved braces
    Y_RAIL, Y_STUD, Y_DIA = 0.006, 0.012, 0.019
    zb = p.plinth
    bress_h, joist_h = 0.2, 0.15
    jd = p.jetty_depth
    zt = H - (bress_h + joist_h if p.jetty else 0.0)     # top of the wall plate
    iz0, iz1 = zb + sill_h, zt - plate_h
    has_l = p.posts in ("both", "left")
    has_r = p.posts in ("both", "right")
    ix0 = -W / 2 + (pw if has_l else 0.0)
    ix1 = W / 2 - (pw if has_r else 0.0)

    # ---- opening (window, or doorway when the sill height is 0)
    jw = sw
    ow = min(p.opening_w, (ix1 - ix0) - 2 * jw - 0.02)
    has_open = p.opening_w > 0.05 and ow > 0.3
    door = has_open and p.opening_sill < 0.03
    ox0 = ox1 = oz0 = oz1 = 0.0
    sill_rail = head_rail = False
    if has_open:
        oxc = clamp(p.opening_x, ix0 + jw + ow / 2, ix1 - jw - ow / 2)
        ox0, ox1 = oxc - ow / 2, oxc + ow / 2
        if door:
            oz0 = 0.0
        else:
            oz0 = max(p.opening_sill, iz0)
            if oz0 < iz0 + rw:
                oz0 = iz0                       # the sill beam is the window sill
            else:
                sill_rail = True
        oz1 = oz0 + p.opening_h
        if oz1 >= iz1 - 0.02:
            oz1 = iz1                           # the wall plate is the lintel
        else:
            oz1 = min(oz1, iz1 - rw)
            head_rail = True

    # ---- main frame: sill beam (bleached, weathered), posts, wall plate
    zs = zb + sill_h / 2
    if door:
        for a, b, pg in ((-W / 2, ox0 - jw + OV, (0, 2)), (ox1 + jw - OV, W / 2, (2, 0))):
            if b - a > 0.08:
                fr.member([(a, zs), (b, zs)], sill_h, 0.0, endgrain=True, wav=(1, 0), pegs=pg, name="sill", wane=0.9, shake=0.8, material=sillm)
    else:
        fr.member([(-W / 2, zs), (W / 2, zs)], sill_h, 0.0, endgrain=True, wav=(1, 0), name="sill", wane=0.9, shake=1.0, material=sillm)
    fr.member([(-W / 2, zt - plate_h / 2), (W / 2, zt - plate_h / 2)], plate_h, 0.0, endgrain=True, wav=(0, 1), name="plate", wane=0.5, shake=0.7)
    fr.drips.append((ix0, ix1, iz1, 0.35 if p.jetty else 1.0))
    if has_l:
        fr.member([(-W / 2 + pw / 2, iz0 - OV), (-W / 2 + pw / 2, iz1 + OV)], pw, 0.0, pegs=(2, 2), wav=(0, 1), name="post", wane=1.0, shake=1.0)
    if has_r:
        fr.member([(W / 2 - pw / 2, iz0 - OV), (W / 2 - pw / 2, iz1 + OV)], pw, 0.0, pegs=(2, 2), wav=(1, 0), name="post", wane=1.0, shake=1.0)
    # wide bays get intermediate posts (they also carry the jetty brackets)
    mids = []
    if not has_open:
        nmid = max(0, int(math.ceil(W / 4.0 - 1e-6)) - 1)
        mids = [-W / 2 + W * (k + 1) / (nmid + 1) for k in range(nmid)]
    for mx in mids:
        fr.member([(mx, iz0 - OV), (mx, iz1 + OV)], pw, 0.0, pegs=(2, 2), name="post", wane=1.0, shake=1.0)

    # ---- secondary timbers
    def stud(r, x, z0, z1, w=sw, y=Y_STUD, pegs=(1, 1), lean=1.0):
        dx = r.u(-1, 1) * irr * 0.02 * lean
        return fr.member([(x - dx / 2, z0 - OV), (x + dx / 2, z1 + OV)], w * r.vary(1.0, 0.1 * irr), y, pegs=pegs, peg_y=0.004, name="stud",
                         alt=0.4, wane=0.5, shake=0.3)

    def rail(r, x0, x1, z, w=rw, y=Y_RAIL, pegs=(2, 2)):
        dz = r.u(-1, 1) * irr * 0.012
        ww = w * r.vary(1.0, 0.08 * irr)
        fr.drips.append((x0, x1, z - ww / 2, 1.0))
        return fr.member([(x0 - OV, z - dz / 2), (x1 + OV, z + dz / 2)], ww, y, pegs=pegs, peg_y=0.004, name="rail", alt=0.25,
                         wane=0.5, shake=0.6)

    def tiers_of(r, z0, z1, hmin=1.9, frac=0.5, x0=0.0, x1=1.0):
        h = z1 - z0
        if h <= hmin:
            return [(z0, z1)]
        zc = z0 + h * (frac + r.u(-0.03, 0.03) * irr)
        rail(r, x0, x1, zc)
        return [(z0, zc - rw / 2), (zc + rw / 2, z1)]

    def brace(r, cell, corner, share=1.0):
        """Curved (natural crook) arch brace: the foot sits shouldered in the corner between the
        post and the rail below, the head is tenoned into the plate."""
        cx0, cx1, a, b = cell
        cw, chh = cx1 - cx0, b - a
        sx = -1 if corner[1] == "l" else 1
        xc = cx0 if sx < 0 else cx1
        zc = b
        bwid = bw * r.vary(1.0, 0.08 * irr)
        run = min(cw * 0.94 * share, 1.2)
        on_rail = chh <= 1.7
        rise = chh if on_rail else min(chh * 0.85, 1.7)
        if run < 0.3 or rise < 0.35:
            return
        p0 = (xc - sx * bwid * (0.42 if on_rail else 0.0), zc - rise)
        p2 = (xc - sx * run, zc)
        mid = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
        k = 0.3 + r.u(-0.06, 0.08) * irr
        c = (mid[0] + (xc - mid[0]) * k, mid[1] + (zc - mid[1]) * k)
        fr.member(_extend(_qbez(p0, c, p2), 0.3), bwid, Y_DIA, clip=cell, pegs=(1, 2), peg_y=0.004,
                  name="brace", spacing=0.14, fine=0.2, alt=0.3, wane=1.2, shake=0.0)

    def fill(r, x0, x1, z0, z1, lp, rp, par=0):
        w, h = x1 - x0, z1 - z0
        if w < 0.12 or h < 0.15:
            return
        pat = p.pattern
        if pat == "close_studding":
            tiers = tiers_of(r, z0, z1, x0=x0, x1=x1)
            ng = max(1, int(round(w / 0.42)))
            xs = [x0 + w * k / ng + r.u(-1, 1) * irr * 0.02 for k in range(1, ng)]
            for a, b in tiers:
                for x in xs:
                    stud(r, x, a, b)
        elif pat == "square_panels":
            nz = max(1, int(round(h / 0.9)))
            nx = max(1, int(round(w / 0.85)))
            zr = [z0 + h * k / nz + r.u(-1, 1) * irr * 0.02 for k in range(1, nz)]
            for z in zr:
                rail(r, x0, x1, z)
            edges = [z0] + zr + [z1]
            xs = [x0 + w * k / nx + r.u(-1, 1) * irr * 0.025 for k in range(1, nx)]
            for ti in range(nz):
                a = edges[ti] + (rw / 2 if ti > 0 else 0)
                b = edges[ti + 1] - (rw / 2 if ti < nz - 1 else 0)
                for x in xs:
                    stud(r, x, a, b)
                if ti == nz - 1 and b - a > 0.45:
                    # straight knee braces from the posts up to the plate
                    bounds = [x0] + xs + [x1]
                    for side, ok in (("l", lp), ("r", rp)):
                        if not ok or (side == "r" and nx == 1 and lp):
                            continue
                        j = 0 if side == "l" else nx - 1
                        cx0 = bounds[j] + (sw / 2 if j > 0 else 0)
                        cx1 = bounds[j + 1] - (sw / 2 if j < nx - 1 else 0)
                        e = min(cx1 - cx0, b - a) * (0.8 + r.u(-0.06, 0.06) * irr)
                        sx = 1 if side == "l" else -1
                        xc = cx0 if side == "l" else cx1
                        pth = _extend([(xc, b - e), (xc + sx * e, b)], 0.3)
                        fr.member(pth, dw * 1.1, Y_DIA, clip=(cx0, cx1, a, b), pegs=(1, 1), peg_y=0.004, name="knee", alt=0.3, wane=0.8)
        elif pat == "herringbone":
            tiers = tiers_of(r, z0, z1, hmin=2.0, x0=x0, x1=x1)
            nx = max(1, int(round(w / 0.56)))
            xs = [x0 + w * k / nx for k in range(1, nx)]
            bounds = [x0] + xs + [x1]
            for ti, (a, b) in enumerate(tiers):
                for x in xs:
                    stud(r, x, a, b, lean=0.3)
                npc = max(1, int(round((b - a) / 0.37)))
                pitch = (b - a) / npc
                for j in range(nx):
                    cx0 = bounds[j] + (sw / 2 if j > 0 else 0)
                    cx1 = bounds[j + 1] - (sw / 2 if j < nx - 1 else 0)
                    cw = cx1 - cx0
                    if cw < 0.1:
                        continue
                    s = 1 if (j + ti + par) % 2 == 0 else -1
                    xm = (cx0 + cx1) / 2
                    ext = cw / 2 + 0.35
                    for k in range(-2, npc + 2):
                        zk = a + pitch * (k + 0.5) + r.u(-1, 1) * irr * 0.012
                        sl = 1.0 + r.u(-1, 1) * irr * 0.06
                        fr.member([(xm - ext, zk - s * sl * ext), (xm + ext, zk + s * sl * ext)], dw * r.vary(1.0, 0.1 * irr), Y_DIA,
                                  clip=(cx0, cx1, a, b), pegs=(1, 0) if k % 2 else (0, 1), peg_y=0.01, name="chev", alt=0.4,
                                  mid=False, fine=0.3, wane=0.4)
        elif pat == "st_andrew":
            if h > 1.55:
                zc = z0 + min(1.05, h * 0.44)
                rail(r, x0, x1, zc + rw / 2)
                ct = (z0, zc)
                ut = (zc + rw, z1)
            else:
                ct, ut = (z0, z1), None
            chh = ct[1] - ct[0]
            nx = max(1, int(round(w / clamp(chh * 1.05, 0.5, 1.1))))
            xs = [x0 + w * k / nx for k in range(1, nx)]
            bounds = [x0] + xs + [x1]
            for x in xs:
                stud(r, x, ct[0], ct[1], lean=0.3)
            for j in range(nx):
                cx0 = bounds[j] + (sw / 2 if j > 0 else 0)
                cx1 = bounds[j + 1] - (sw / 2 if j < nx - 1 else 0)
                cw = cx1 - cx0
                if cw < 0.3 or chh < 0.3:
                    continue
                e = 0.08
                for _ in range(5):
                    th = math.atan2(chh, max(cw - 2 * e, 0.02))
                    e = min(dw / (2 * math.sin(th)), cw * 0.4)
                for q, sgn in enumerate((1, -1)):
                    xa, xb = (cx0 + e, cx1 - e) if sgn > 0 else (cx1 - e, cx0 + e)
                    pth = _extend([(xa, ct[0]), (xb, ct[1])], 0.3)
                    fr.member(pth, dw, Y_DIA + 0.005 * q, clip=(cx0, cx1, ct[0], ct[1]), pegs=(1, 1), peg_y=0.004, name="cross", alt=0.3,
                              mid=False, fine=0.25, wane=0.6)
            if ut and ut[1] - ut[0] > 0.2:
                ng = max(1, int(round(w / 0.45)))
                if nx > 1:
                    ng = nx * max(1, int(round((w / nx) / 0.45)))
                for k in range(1, ng):
                    stud(r, x0 + w * k / ng + r.u(-1, 1) * irr * 0.015, ut[0], ut[1])
        else:  # curved_braces
            tiers = tiers_of(r, z0, z1, frac=0.47, x0=x0, x1=x1)
            nx = max(1, int(round(w / 0.82)))
            xs = [x0 + w * k / nx + r.u(-1, 1) * irr * 0.02 for k in range(1, nx)]
            bounds = [x0] + xs + [x1]
            for ti, (a, b) in enumerate(tiers):
                for x in xs:
                    stud(r, x, a, b)
                top = ti == len(tiers) - 1
                for j in range(nx):
                    cx0 = bounds[j] + (sw / 2 if j > 0 else 0)
                    cx1 = bounds[j + 1] - (sw / 2 if j < nx - 1 else 0)
                    left = j == 0 and lp and top
                    right = j == nx - 1 and rp and top
                    if left or right:
                        share = 0.5 if (left and right) else 1.0
                        if left:
                            brace(r, (cx0, cx1, a, b), "tl", share)
                        if right:
                            brace(r, (cx0, cx1, a, b), "tr", share)
                    elif cx1 - cx0 > 0.6:
                        stud(r, (cx0 + cx1) / 2 + r.u(-1, 1) * irr * 0.02, a, b, w=sw * 0.85)

    rf = rng.fork("fill")
    if has_open:
        jz0 = 0.0 if door else iz0
        for xj in (ox0 - jw / 2, ox1 + jw / 2):
            fr.member([(xj, jz0 - (0 if door else OV)), (xj, iz1 + OV)], jw, Y_RAIL, pegs=(0 if door else 2, 2), name="jamb", wane=0.6, shake=0.5)
        if sill_rail:
            rail(rf, ox0, ox1, oz0 - rw / 2, y=Y_STUD)
        if head_rail:
            rail(rf, ox0, ox1, oz1 + rw / 2, y=Y_STUD)
        fill(rf.fork("l"), ix0, ox0 - jw, iz0, iz1, True, False, 0)
        fill(rf.fork("r"), ox1 + jw, ix1, iz0, iz1, False, True, 1)
        if sill_rail:
            fill(rf.fork("b"), ox0, ox1, iz0, oz0 - rw, False, False, 0)
        if head_rail:
            fill(rf.fork("a"), ox0, ox1, oz1 + rw, iz1, False, False, 1)
    else:
        xa = ix0
        secs = []
        for mx in mids:
            secs.append((xa, mx - pw / 2))
            xa = mx + pw / 2
        secs.append((xa, ix1))
        for i, (a, b) in enumerate(secs):
            fill(rf.fork("s%d" % i) if mids else rf, a, b, iz0, iz1, True, True, i)

    # ---- pegs: riven oak, tapered, each a little proud and askew
    if fr.pegs:
        rp_ = rng.fork("pegs")
        pg = cylinder("peg", radius=0.0145, depth=0.05, verts=5, radius2=0.0115)
        assign(pg, endm)
        ms = []
        for x, z, py in fr.pegs:
            proud = 0.006 + rp_.u(0.0, 0.007)
            ms.append(trs((x, py + 0.025 - proud, z), (90 + rp_.u(-5, 5), rp_.u(0, 360), rp_.u(-5, 5)), rp_.u(0.85, 1.15)))
        pegs = instances_to_mesh("pegs", pg, ms, keep_base=False)
        paint(pegs, wear=0.35, grime=0.15 * age)

    # ---- infill panels: triangulated exactly against the timber outlines (constrained Delaunay), so
    # the lime runs up to every timber with a small fillet and nothing hides under a beam
    gx0 = -W / 2 + (pw / 2 if has_l else 0.0)
    gx1 = W / 2 - (pw / 2 if has_r else 0.0)
    gz0, gz1 = zb + sill_h / 2, zt - plate_h / 2
    hole = None
    if has_open:
        hole = (ox0 - jw / 2, ox1 + jw / 2, -1.0 if door else oz0 - min(rw, sill_h) / 2, oz1 + min(rw, plate_h) / 2)
    brick = p.infill == "brick"
    B = 0.0 if brick else 0.004 + 0.003 * irr                       # pillow of a panel (m)
    y_pl = max(Y_STUD + p.recess + 0.003, Y_DIA + 0.014 + B)        # plaster face at the timbers
    rpl = rng.fork("plaster")
    ph = [rpl.u(0, 6.28) for _ in range(8)]

    def fdist(x, z):
        return fr.dist(np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(z, dtype=float)))

    def in_hole(x, z):
        if not hole:
            return np.zeros(np.shape(x), dtype=bool)
        return (x > hole[0]) & (x < hole[1]) & (z > hole[2]) & (z < hole[3])

    def triangulate(lines, loops, points):
        pts, edges = [], []
        for ln in lines:
            b = len(pts)
            pts.extend((float(x), float(z)) for x, z in ln)
            edges.extend((b + i, b + i + 1) for i in range(len(ln) - 1))
        for ln in loops:
            b = len(pts)
            pts.extend((float(x), float(z)) for x, z in ln)
            edges.extend((b + i, b + (i + 1) % len(ln)) for i in range(len(ln)))
        pts.extend((float(x), float(z)) for x, z in points)
        res = delaunay_2d_cdt([Vector(q) for q in pts], edges, [], 0, 1e-5)
        V = np.array([(v[0], v[1]) for v in res[0]], dtype=float).reshape(-1, 2)
        F = np.array([tuple(f) for f in res[2] if len(f) == 3], dtype=int).reshape(-1, 3)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        ar = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        F[ar < 0] = F[ar < 0][:, ::-1]                              # CCW in (x, z) = facing -Y
        cx, cz = (a[:, 0] + b[:, 0] + c[:, 0]) / 3, (a[:, 1] + b[:, 1] + c[:, 1]) / 3
        keep = (np.abs(ar) > 2e-8) & (cx > gx0) & (cx < gx1) & (cz > gz0) & (cz < gz1) & ~in_hole(cx, cz)
        keep &= fr.dist(cx, cz) > -0.001
        return V, F[keep], cx[keep], cz[keep]

    rect = [(gx0, gz0), (gx1, gz0), (gx1, gz1), (gx0, gz1)]
    loops = [rect]
    if hole:
        loops.append([(hole[0], hole[2]), (hole[1], hole[2]), (hole[1], hole[3]), (hole[0], hole[3])])

    def grid_pts(x0, x1, z0, z1, g, r):
        out = []
        nx, nz = max(1, int(round((x1 - x0) / g))), max(1, int(round((z1 - z0) / g)))
        for j in range(nz + 1):
            for i in range(nx + (0 if j % 2 else 1)):
                out.append((x0 + (x1 - x0) * (i + (0.5 if j % 2 else 0.0)) / nx + r.u(-0.2, 0.2) * g,
                            z0 + (z1 - z0) * j / nz + r.u(-0.2, 0.2) * g))
        return out

    # rain run-off streaks under the rails and the wall plate
    rs_ = rng.fork("streaks")
    streaks = []
    for a, b, zbot, kk in fr.drips:
        n = int(round((b - a) * (1.3 + 3.0 * age) * kk + rs_.u(-0.5, 0.5)))
        for _ in range(max(0, n)):
            x = rs_.u(a + 0.03, b - 0.03)
            L = rs_.u(0.3, 0.8) * (0.6 + 0.8 * age)
            wst = rs_.u(0.016, 0.036)
            I = rs_.u(0.5, 0.66)
            zs_ = zbot - 0.014 - np.arange(48) * 0.02
            dd = np.minimum(fdist(np.full(48, x), zs_), np.minimum(fdist(np.full(48, x - 2.2 * wst), zs_), fdist(np.full(48, x + 2.2 * wst), zs_)))
            if dd[0] < 0.004 or in_hole(x, zbot - 0.03) or any(abs(x - q[0]) < 0.16 and abs(zbot - q[1]) < 0.3 for q in streaks):
                continue
            blk = np.nonzero(dd < 0.004)[0]
            if len(blk):
                L = min(L, blk[0] * 0.02)
            if L >= 0.07:
                streaks.append((x, zbot, L, wst, I))
    if streaks:
        SA = np.array(streaks, dtype=float)

    def streak_mask(x, z):
        if not streaks:
            return np.zeros(x.shape)
        t = (SA[None, :, 1] - z[:, None]) / SA[None, :, 2]
        dx = np.abs(x[:, None] - SA[None, :, 0]) / (SA[None, :, 3] * (1.0 - 0.72 * np.clip(t, 0.0, 1.0)))   # tapers downwards
        mk = np.clip((1.7 - dx) / 1.15, 0.0, 1.0) * np.clip((1.0 - t) / 0.3, 0.0, 1.0) ** 0.6 * (t > -0.05) * SA[None, :, 4]
        return mk.max(axis=1)

    def tide(x):
        return (0.21 + 0.05 * np.sin(3.1 * x + ph[0]) + 0.04 * np.sin(7.7 * x + ph[1]) + 0.025 * np.sin(17.0 * x + ph[2])) * (0.55 + 0.9 * age)

    # the panels between the timbers as connected regions of a 2 cm raster (coursing direction of
    # the bricks / re-limed panels are chosen per panel)
    cs = 0.02
    nxr, nzr = max(1, int(math.ceil((gx1 - gx0) / cs))), max(1, int(math.ceil((gz1 - gz0) / cs)))
    RX, RZ = np.meshgrid(gx0 + (np.arange(nxr) + 0.5) * cs, gz0 + (np.arange(nzr) + 0.5) * cs, indexing="ij")
    fl = ((fr.dist(RX, RZ) > 0.0) & ~in_hole(RX, RZ)).tolist()
    labl = [[0] * nzr for _ in range(nxr)]
    nlab = 0
    for i in range(nxr):
        for j in range(nzr):
            if fl[i][j] and not labl[i][j]:
                nlab += 1
                labl[i][j] = nlab
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    for a2, b2 in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= a2 < nxr and 0 <= b2 < nzr and fl[a2][b2] and not labl[a2][b2]:
                            labl[a2][b2] = nlab
                            stack.append((a2, b2))
    lab = np.array(labl)

    def lab_at(x, z):
        return lab[np.clip(((x - gx0) / cs).astype(int), 0, nxr - 1), np.clip(((z - gz0) / cs).astype(int), 0, nzr - 1)]

    def lab_near(x, z):
        out = lab_at(x, z)
        for dx_, dz_ in ((cs, 0), (-cs, 0), (0, cs), (0, -cs), (2 * cs, 0), (-2 * cs, 0), (0, 2 * cs), (0, -2 * cs)):
            if out.all():
                break
            out = np.where(out == 0, lab_at(x + dx_, z + dz_), out)
        return out

    # broken patches showing the daub and laths, and re-limed repairs in a fresher tint
    chips, patches = [], []
    if not brick:
        rc = rng.fork("chips")
        tries = 0
        while len(chips) < int(age * 2.4 + 0.3) and tries < 40:
            tries += 1
            x, z = rc.u(gx0 + 0.25, gx1 - 0.25), iz0 + rc.u(0.0, 0.07)
            if fdist(x, z + 0.07)[0] < 0.03 or in_hole(x, z + 0.12) or any(abs(x - c[0][0]) < 0.5 for c in chips):
                continue
            rad = rc.u(0.085, 0.13) * (0.8 + 0.5 * age)
            outer = _blob(rc, x, z, rad, 12, 1.35)
            chips.append(((x, z), outer, _scaled(outer, (x, z), 0.76)))
        # one or two panels were re-limed later: fresher, cleaner limewash
        sizes = np.bincount(lab.ravel(), minlength=nlab + 1) * cs * cs
        cand = [c for c in range(1, nlab + 1) if 0.1 < sizes[c] < 0.3 * max(sizes[1:].sum(), 1e-6)] or \
            [c for c in range(1, nlab + 1) if sizes[c] > 0.1]
        for _ in range((1 if age > 0.25 else 0) + (1 if age > 0.7 else 0)):
            if cand:
                patches.append(cand.pop(rc.randrange(len(cand))))

    pts, spts = [], []
    if not brick:
        pts += grid_pts(gx0, gx1, gz0, gz1, 0.21, rpl)
        pts += grid_pts(gx0, gx1, iz0 + 0.03, iz0 + 0.03 + 0.4 * (0.6 + 0.8 * age), 0.105, rpl)
        for x, zbot, L, wst, I in streaks:
            spts += [(x + s * 0.5 * wst * (1 - 0.72 * t), zbot - 0.012 - t * L) for t in (0.0, 0.22, 0.45, 0.7, 0.88) for s in (-1, 1)]
            spts += [(x + s * 1.75 * wst * (1 - 0.72 * t), zbot - 0.012 - t * L) for t in (0.0, 0.22, 0.45, 0.7, 0.88) for s in (-1, 1)]
            spts.append((x, zbot - 0.012 - 1.03 * L))
    else:
        pts += grid_pts(gx0, gx1, gz0, gz1, 0.3, rpl)
    if pts:
        PA = np.array(pts)
        ok = fdist(PA[:, 0], PA[:, 1]) > (0.03 if not brick else 0.05)
        pts = [q for q, k in zip(pts, ok) if k]
    if spts:                                                        # streak points may sit close under the rail
        PA = np.array(spts)
        ok = fdist(PA[:, 0], PA[:, 1]) > 0.009
        if pts:
            QA = np.array(pts)                                      # drop grid points that would cut a streak
            far = np.ones(len(QA), dtype=bool)
            for x, zbot, L, wst, I in streaks:
                far &= ~((np.abs(QA[:, 0] - x) < 2.6 * wst) & (QA[:, 1] < zbot) & (QA[:, 1] > zbot - L - 0.03))
            pts = [q for q, k in zip(pts, far) if k]
        pts += [q for q, k in zip(spts, ok) if k]
    cloops = [c[1] for c in chips] + [c[2] for c in chips]
    V, F, fcx, fcz = triangulate(fr.lines + fr.rings, loops + cloops, pts)
    vx, vz = V[:, 0], V[:, 1]
    vd = fr.dist(vx, vz)
    fw, fh = (0.013, 0.009) if brick else (0.0065, 0.0)
    fil = (fh + 0.001 * np.sin(31.0 * vx + ph[3]) * np.sin(27.0 * vz + ph[4])) if brick else \
        (0.0034 + 0.0016 * np.sin(23.0 * vx + ph[3]) + 0.0012 * np.sin(29.0 * vz + ph[4]))
    und = 0.0012 * (np.sin(vx * 3.1 + ph[5]) * np.sin(vz * 2.6 + ph[6]) + 0.5 * np.sin(vx * 6.3 + ph[7]) * np.sin(vz * 5.1 + ph[5]))
    ft = np.clip(1.0 - vd / fw, 0.0, 1.0)
    pil = np.clip((vd - fw) / 0.2, 0.0, 1.0)
    pil = pil * pil * (3 - 2 * pil)
    VY = y_pl - fil * ft - B * pil + (0.0 if brick else 1.0) * und * np.clip(vd / 0.1, 0.0, 1.0)
    in_chip = np.zeros(len(V), dtype=bool)
    rim = np.zeros(len(V), dtype=bool)
    for c0, outer, inner in chips:
        ins = _inpoly(vx, vz, _scaled(inner, c0, 1.04))
        in_chip |= ins
        rim |= _inpoly(vx, vz, _scaled(outer, c0, 1.04)) & ~ins
        VY = np.where(ins, y_pl + 0.007 + 0.0015 * np.sin(57.0 * vx + ph[0]) * np.sin(49.0 * vz), VY)
    in_patch = np.zeros(len(V), dtype=bool)
    if patches:
        in_patch[F[np.isin(lab_near(fcx, fcz), patches)].ravel()] = True
    infill = mesh_object("infill", [(float(vx[i]), float(VY[i]), float(vz[i])) for i in range(len(V))], [tuple(int(k) for k in f) for f in F])
    uv_box(infill)
    smooth(infill, 60)
    if brick:
        pm = mat("mortar", "#b9ae98")
    else:
        pm = mat("plaster_lime", p.plaster, scale=3.0)
    assign(infill, pm)
    if chips:
        daub = mat("plaster_rough", "#8f7556")
        cps = [c[1] for c in chips]
        assign_faces(infill, daub, lambda c, n: any(bool(_inpoly(c.x, c.z, q)) for q in cps))
        # riven laths showing in the break
        lath = []
        for c0, outer, inner in chips:
            for zl in (c0[1] + 0.028, c0[1] + 0.082):
                xs_ = np.linspace(c0[0] - 0.3, c0[0] + 0.3, 90)
                okl = _inpoly(xs_, np.full(90, zl), _scaled(inner, c0, 0.9)) & (fdist(xs_, np.full(90, zl - 0.016)) > 0.004) \
                    & (fdist(xs_, np.full(90, zl + 0.016)) > 0.004)
                i0, i1 = _run(okl)
                if i1 - i0 > 8:
                    lb = box("lath", (xs_[i1] - xs_[i0], 0.005, 0.026 + rc.u(0, 0.006)), loc=((xs_[i0] + xs_[i1]) / 2, y_pl + 0.004, zl),
                             rot=(0, rc.u(-1.5, 1.5), 0))
                    uv_grain(lb, axis="X")
                    assign(lb, oak2)
                    lath.append(lb)
        paint(lath, grime=0.35, wear=0.3)
    if patches:
        pf = lambda c, n: int(lab_near(np.array([c.x]), np.array([c.z]))[0]) in patches and \
            not (chips and any(bool(_inpoly(c.x, c.z, q)) for q in cps))
        assign_faces(infill, mat("plaster_lime", _mix(p.plaster, "#f4efe4", 0.3), scale=3.0), pf)
    low_k = 0.55 if zb > 0.01 else 1.0

    def pl_grime(co, n):
        x, z = co[:, 0], co[:, 2]
        g = streak_mask(x, z) * (0.7 + 0.3 * age)
        splash = np.clip(1.0 - (z - iz0) / tide(x), 0.0, 1.0) ** 0.6 * low_k * (0.42 + 0.24 * age)
        broad = (0.03 + 0.09 * age) * (0.5 + 0.5 * np.sin(1.7 * x + ph[6]) * np.sin(1.3 * z + ph[7]))
        g = np.clip(g + splash + broad, 0.0, 1.0)
        if len(g) == len(in_patch):
            g = np.where(in_patch, g * 0.3, g)
            g = np.where(in_chip, 0.45, g)
        return g

    def pl_damp(co, n):
        x, z = co[:, 0], co[:, 2]
        return np.clip(1.0 - (z - iz0) / (0.45 * tide(x)), 0.0, 1.0) ** 1.5 * 0.8 * age * low_k * (0.55 + 0.45 * np.sin(9.0 * x + ph[3]))

    def pl_chip(co, n):
        if len(co) == len(rim):
            return np.where(rim & ~in_chip, 0.75, 0.0)
        return np.zeros(len(co))

    paint(infill, grime=pl_grime, special=pl_damp, wear=pl_chip)

    # ---- brick nogging: individual hand-made bricks standing proud of the lime bedding, coursed
    # level or diagonally (herringbone from panel to panel), a few lime-washed or soot-dark
    if brick:
        rb = rng.fork("bricks")
        ML, MV, J = 0.236, 0.074, 0.011
        pats, bricks, bl_loops = {}, {}, []

        su, sv = np.meshgrid(np.linspace(0.04, 0.96, 6), np.linspace(0.1, 0.9, 3), indexing="ij")
        for c in range(1, nlab + 1):
            ii, jj = np.nonzero(lab == c)
            if len(ii) < 6:
                continue
            bx0, bx1 = gx0 + ii.min() * cs, gx0 + (ii.max() + 1) * cs
            bz0, bz1 = gz0 + jj.min() * cs, gz0 + (jj.max() + 1) * cs
            th = 0.0
            if rb.chance(0.45):
                th = math.radians(rb.pick([40.0, 45.0, 50.0])) * (1 if c % 2 else -1)
            ct, st = math.cos(th), math.sin(th)
            ox, oz = bx0 + rb.u(-0.1, 0.0), bz0 + rb.u(-0.03, 0.0)
            cu = [(x - ox) * ct + (z - oz) * st for x in (bx0, bx1) for z in (bz0, bz1)]
            cv = [-(x - ox) * st + (z - oz) * ct for x in (bx0, bx1) for z in (bz0, bz1)]
            k0, k1 = int(math.floor(min(cv) / MV)), int(math.ceil(max(cv) / MV))
            offs = {k: (k % 2) * ML / 2 + rb.u(-0.025, 0.025) for k in range(k0, k1 + 1)}
            pats[c] = (ct, st, ox, oz, offs)
            for k in range(k0, k1 + 1):
                for j in range(int(math.floor((min(cu) - offs[k]) / ML)), int(math.ceil((max(cu) - offs[k]) / ML)) + 1):
                    ua, va = j * ML + offs[k] + J / 2, k * MV + J / 2
                    uu, vv = ua + su * (ML - J), va + sv * (MV - J)
                    if not (lab_at(ox + uu * ct - vv * st, oz + uu * st + vv * ct) == c).any():
                        continue
                    tone = rb.weighted([(0, 0.81), (1, 0.11), (2, 0.08)])
                    bricks[(c, k, j)] = (tone, rb.u(-0.0025, 0.0025), rb.u(-0.012, 0.012), rb.u(-0.02, 0.02), rb.u(0.05, 0.5))
                    bl_loops.append([(ox + u * ct - v * st, oz + u * st + v * ct) for u, v in
                                     ((ua, va), (ua + ML - J, va), (ua + ML - J, va + MV - J), (ua, va + MV - J))])
        V2, F2, c2x, c2z = triangulate(fr.lines, loops + bl_loops, [])
        fl2 = lab_at(c2x, c2z)
        yb = y_pl - 0.0075
        bv, bf, bg, vid = ([], [], []), ([], [], []), ([], [], []), {}
        for fi in range(len(F2)):
            c = int(fl2[fi])
            if c == 0:
                ci, cj = int((c2x[fi] - gx0) / cs), int((c2z[fi] - gz0) / cs)
                win = lab[max(0, ci - 2):ci + 3, max(0, cj - 2):cj + 3]
                nzv = win[win > 0]
                if not len(nzv):
                    continue
                c = int(np.bincount(nzv).argmax())
            if c not in pats:
                continue
            ct, st, ox, oz, offs = pats[c]
            u = (c2x[fi] - ox) * ct + (c2z[fi] - oz) * st
            v = -(c2x[fi] - ox) * st + (c2z[fi] - oz) * ct
            k = int(math.floor(v / MV))
            if k not in offs:
                continue
            j = int(math.floor((u - offs[k]) / ML))
            fu, fv = u - offs[k] - j * ML, v - k * MV
            key = (c, k, j)
            if key not in bricks or not (J / 2 - 1e-4 < fu < ML - J / 2 + 1e-4 and J / 2 - 1e-4 < fv < MV - J / 2 + 1e-4):
                continue
            tone, yo, tu, tv, gg = bricks[key]
            tri = []
            for q in F2[fi]:
                kq = (int(q), key)
                if kq not in vid:
                    x, z = V2[q]
                    lu = (x - ox) * ct + (z - oz) * st - offs[k] - j * ML - ML / 2
                    lv = -(x - ox) * st + (z - oz) * ct - k * MV - MV / 2
                    vid[kq] = len(bv[tone])
                    bv[tone].append((float(x), float(yb + yo + tu * lu + tv * lv), float(z)))
                    bg[tone].append(gg)
                tri.append(vid[kq])
            bf[tone].append(tuple(tri))
        bmats = (mat("terracotta", "#a98f86"), mat("terracotta", "#786863"), pm)
        for tone in range(3):
            if not bf[tone]:
                continue
            bo = mesh_object("bricks%d" % tone, bv[tone], bf[tone])
            uv_box(bo)
            assign(bo, bmats[tone])
            ga = np.array(bg[tone])

            def br_grime(co, n, ga=ga, tone=tone):
                base = ga if len(ga) == len(co) else 0.25
                return np.clip(base * (0.5 if tone == 2 else 1.0) + pl_grime(co, n), 0.0, 1.0)

            paint(bo, grime=br_grime, special=pl_damp)

    # plain plastered inside face (the room side) so the bay is closed from behind
    bxs = _lines(-W / 2, W / 2, 0.9, (ox0, ox1) if has_open else ())
    bzs = _lines(zb, zt, 0.9, ((oz0, oz1) if not door else (oz1,)) if has_open else ())
    bv = [(x, D, z) for x in bxs for z in bzs]
    nzb = len(bzs)
    bf = []
    for i in range(len(bxs) - 1):
        for j in range(nzb - 1):
            cx, cz = (bxs[i] + bxs[i + 1]) / 2, (bzs[j] + bzs[j + 1]) / 2
            if has_open and ox0 < cx < ox1 and (oz0 if not door else -1.0) < cz < oz1:
                continue
            bf.append((i * nzb + j, i * nzb + j + 1, (i + 1) * nzb + j + 1, (i + 1) * nzb + j))
    back = mesh_object("inside_plaster", bv, bf)
    uv_box(back)
    assign(back, pm)
    paint(back, grime=lambda co, n: np.clip(1.0 - (co[:, 2] - zb) / 0.5, 0.0, 1.0) * 0.4 + 0.1)

    # ---- jetty: joists over the plate, bressummer on their ends, carved brackets on the posts
    extra = []
    if p.jetty:
        rj = rng.fork("jetty")
        xe = W / 2 - pw / 2
        nj = max(1, int(round(2 * xe / 0.46)))
        jx = [-xe + 2 * xe * k / nj for k in range(nj + 1)]
        z0j, z1j = zt, zt + joist_h
        yfb = -jd - 0.015
        band, qk = 0.042, 0.014
        rr = joist_h - band
        for k, x in enumerate(jx):
            jw_ = 0.115 * rj.vary(1.0, 0.1 * irr + 0.03)
            yo = -jd + 0.012 + rj.u(0.0, 0.012) * irr
            dz = rj.u(0.0, 0.006) * irr
            # bull-nosed end: square fillet band over a quirk and a full quarter-round roll
            poly = [(0.05, z0j - dz)]
            poly += [(yo + qk + rr + rr * math.cos(math.radians(a)), z0j - dz + rr + rr * math.sin(math.radians(a))) for a in (270, 252, 234, 216, 198, 180)]
            poly += [(yo, z0j - dz + rr), (yo, z1j), (0.05, z1j)]
            j = _prism_x("joist%d" % k, poly, x - jw_ / 2, x + jw_ / 2)
            assign(j, oak)
            assign_faces(j, endm, lambda c, n: n.y < -0.5)
            uv_grain(j, axis="Y")
            extra.append(j)
        # blocking between the joists on the wall line, floor boards over them
        blk = box("blocking", (W - 0.004, D - 0.03, joist_h), loc=(0, 0.03 + (D - 0.03) / 2, zt + joist_h / 2))
        uv_grain(blk, axis="X")
        assign(blk, oak)
        extra.append(blk)
        fb = box("floor_boards", (W - 0.004, jd + D - 0.19, 0.03), loc=(0, (-jd + 0.17 + D) / 2, z1j + 0.015))
        uv_grain(fb, axis="Y")
        assign(fb, oak2)
        extra.append(fb)
        # bressummer: moulded section lofted along x with a little sag; the fascia has a sunk
        # frieze field (11 mm) left standing as a row of lozenges and nail-head bosses
        fz0, fz1, fdp = 0.089, 0.153, 0.011
        prof = [(0.2, 0.0), (0.04, 0.0), (0.022, 0.006), (0.009, 0.018), (0.002, 0.036), (0.0, 0.05), (0.0, 0.058), (0.009, 0.064),
                (0.009, 0.074), (0.0, 0.08), (0.0, fz0 - 0.004), (fdp, fz0), (fdp, fz1), (0.0, fz1 + 0.004), (0.0, 0.163),
                (0.008, 0.168), (0.008, 0.179), (0.0, 0.184), (0.0, 0.194), (0.006, 0.2), (0.2, 0.2)]
        nb = max(2, int(round(W / 0.5)) + 1)
        sag = _wob(rj, nb, 0.004 * irr)
        wyb = _wob(rj, nb, 0.004 * irr)

        def bz_off(u):
            return -abs(sag(u)) - 0.004 * irr * math.sin(math.pi * u)

        bv, bf = [], []
        npf = len(prof)
        sc = bress_h / 0.2
        for i in range(nb):
            u = i / (nb - 1)
            x = -W / 2 + W * u
            dzz = bz_off(u)
            for k, (py, pz) in enumerate(prof):
                top = pz > 0.19
                bv.append((x, yfb + py + (wyb(u) if py < 0.1 else 0.0), z1j + pz * sc + (0.0 if top else dzz * (1 - pz / 0.2))))
        for i in range(nb - 1):
            for k in range(npf):
                bf.append((i * npf + k, i * npf + (k + 1) % npf, (i + 1) * npf + (k + 1) % npf, (i + 1) * npf + k))
        bf.append(tuple(range(npf)))
        bf.append(tuple(range((nb - 1) * npf, nb * npf)))
        br = recalc_normals(mesh_object("bressummer", bv, bf))
        uv_grain(br, axis="X")
        assign(br, oak)
        assign_faces(br, endm, lambda c, n: abs(n.x) > 0.92)
        zf = z1j + (fz0 + fz1) / 2 * sc
        paint(br, grime=lambda co, n: np.where((np.abs(co[:, 2] - zf) < 0.045) & (co[:, 1] > yfb + 0.006) & (co[:, 1] < yfb + 0.02), 0.85, 0.0))
        extra.append(br)
        fhh = (fz1 - fz0) / 2 * sc - 0.003
        lz = [(-0.058, 0), (0, -fhh), (0.058, 0), (0, fhh)]
        loz = mesh_object("lozenge", [(x, y, 0.0) for x, y in lz] + [(x * 0.62, y * 0.5, fdp) for x, y in lz],
                          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)])
        boss = mesh_object("boss", [(-0.016, 0, 0), (0, -0.016, 0), (0.016, 0, 0), (0, 0.016, 0), (0, 0, 0.009)],
                           [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
        assign(loz, oak)
        assign(boss, oak)
        nl = max(1, int(round((W - 0.24) / 0.2)))
        lm, bm_ = [], []
        for k in range(2 * nl + 1):
            u = 0.5 + ((k / (2 * nl)) - 0.5) * (W - 0.24) / W
            m4 = trs((-W / 2 + W * u, yfb + fdp + wyb(u) + 0.0005, zf + bz_off(u) * 0.4), (90, 0, 0), (1.0, 1.0, 1.0))
            if k % 2:
                lm.append(m4)
            else:
                bm_.append(m4)
        fz_ = [recalc_normals(instances_to_mesh("frieze", loz, lm, keep_base=False)), recalc_normals(instances_to_mesh("bosses", boss, bm_, keep_base=False))]
        paint(fz_, wear=lambda co, n: np.where(co[:, 1] < yfb + 0.004, 0.55, 0.0))
        extra += fz_
        # ogee brackets: one on every post this bay owns
        A = jd - 0.03
        Hb = clamp(jd * 1.35, 0.4, 0.62)
        zbk = zt - Hb
        Bk = Hb - 0.07 - 0.075
        o_pts = [(-0.012, zt), (A, zt), (A, zt - 0.05), (A - 0.014, zt - 0.058)]
        o_pts += _cbez((A - 0.014, zt - 0.07), (A - 0.02, zt - 0.07 - 0.6 * Bk), (0.05, zt - 0.07 - 0.45 * Bk), (0.05, zbk + 0.078), n=8)
        o_pts += [(0.07, zbk + 0.064), (0.08, zbk + 0.044), (0.072, zbk + 0.022), (0.05, zbk + 0.008), (-0.012, zbk)]
        bxs = ([-xe] if has_l else []) + ([xe] if has_r else []) + list(mids)
        for k, x in enumerate(bxs):
            bt = 0.11
            bk = _prism_x("bracket%d" % k, [(-o, z) for o, z in o_pts], x - bt / 2, x + bt / 2)
            uv_grain(bk, axis=(0, -0.6, 0.8))
            assign(bk, oak)
            bevel(bk, 0.007, segments=1)
            extra.append(bk)
        if bxs:
            pg2 = cylinder("peg", radius=0.014, depth=0.05, verts=5, radius2=0.011)
            assign(pg2, endm)
            pm2 = []
            for x in bxs:
                for zq in (zbk + 0.05, zbk + 0.16):
                    pm2.append(trs((x - 0.055 + 0.022, -0.045 - rj.u(0, 0.01), zq), (rj.u(0, 360), 90, 0), 1.0))
            extra.append(instances_to_mesh("bracket_pegs", pg2, pm2, keep_base=False))
        socket("jetty_front", "attach", (0, yfb, H), role="wall_above", depth=jd)

    # ---- forged iron L-straps over the post joints: fishtail ends, rose-head nails
    irons = []
    stains = []
    if p.straps and (has_l or has_r):
        ri = rng.fork("iron")
        h2 = 0.023
        nail = bolt("nail", 0.0115, 0.009, kind="pyramid")
        nms = []
        top = not p.jetty
        zj = (zt - plate_h / 2) if top else zs
        sv = -1 if top else 1          # the leg on the post runs down from the plate / up from the sill
        for side, ok in ((-1, has_l), (1, has_r)):
            if not ok:
                continue
            xp = side * (W / 2 - pw / 2)
            lv = 0.36 + ri.u(0, 0.07)
            lh = 0.33 + ri.u(0, 0.08)
            sh = -side
            st_ = [(-h2 + 0.012, -h2), (lh - 0.07, -h2), (lh, -1.75 * h2), (lh - 0.024, 0.0), (lh, 1.75 * h2), (lh - 0.07, h2),
                   (h2, h2), (h2, lv - 0.07), (1.75 * h2, lv), (0.0, lv - 0.024), (-1.75 * h2, lv), (-h2, lv - 0.07), (-h2, -h2 + 0.012)]
            pts_ = [(xp + sh * s, zj + sv * t) for s, t in st_]
            if (sh * sv) < 0:
                pts_ = pts_[::-1]
            s = shape_extrude("strap", pts_, depth=0.011, loc=(0, 0.005, 0), rot=(90, 0, 0), bevel=0.0015)
            irons.append(s)
            for t in (0.07, 0.19, lh - 0.085):
                nms.append(trs((xp + sh * t, -0.0058, zj + ri.u(-0.004, 0.004)), (90, ri.u(0, 90), 0), ri.u(0.85, 1.15)))
                stains.append((xp + sh * t, zj - h2, ri.u(0.1, 0.22), 0.03))
            for t in (0.0, 0.13, 0.24, lv - 0.085):
                nms.append(trs((xp + ri.u(-0.004, 0.004), -0.0058, zj + sv * t), (90, ri.u(0, 90), 0), ri.u(0.85, 1.15)))
            stains.append((xp, zj + sv * lv if sv < 0 else zj - h2, ri.u(0.22, 0.4), 0.045))
        irons.append(instances_to_mesh("nails", nail, nms, keep_base=False))
        rust = noise_mask(0.06, seed=p.seed + 23, lo=0.25, hi=0.75)
        for o in irons:
            assign(o, mat("iron_wrought"))
            paint(o, special=lambda co, n: np.clip(0.12 + rust(co, n) * (0.3 + 0.3 * age), 0, 1), grime=0.2,
                  wear=lambda co, n: np.clip(-n[:, 1], 0, 1) * 0.12)

    # ---- stone plinth under the sill beam
    if zb > 0.02:
        rs = rng.fork("plinth")
        stone = mat("stone_rough", "#a89f8f")
        courses = 1 if zb <= 0.34 else 2
        chh = zb / courses
        spans = [(-W / 2, W / 2)]
        if door:
            spans = [(-W / 2, ox0 - jw), (ox1 + jw, W / 2)]
        for ci in range(courses):
            for a, b in spans:
                if b - a < 0.1:
                    continue
                nbk = max(1, int(round((b - a) / (0.55 if ci == 0 else 0.42))))
                cuts = [a] + [a + (b - a) * (k + rs.u(-0.18, 0.18)) / nbk for k in range(1, nbk)] + [b]
                for k in range(len(cuts) - 1):
                    L = cuts[k + 1] - cuts[k]
                    proj = 0.04 + rs.u(-0.006, 0.006)
                    blk = stone_block("plinth", (L - 0.008, D + proj, chh - 0.008),
                                      loc=((cuts[k] + cuts[k + 1]) / 2, (D - proj) / 2, ci * chh + chh / 2 + 0.002), rng=rs,
                                      chamfer=0.014, rough=0.003, detail=0.2)
                    assign(blk, stone)
                    paint(blk, grime=height_ramp(0.0, 0.45),
                          special=mask_mul(facing((0, 0, 1)), noise_mask(0.2, seed=p.seed + 3, lo=0.3, hi=0.7), 0.5 + 0.5 * age))
        # lime mortar bed behind the block faces
        for a, b in spans:
            if b - a < 0.1:
                continue
            bed = box("bedding", (b - a - 0.01, D + 0.02, zb - 0.006), loc=((a + b) / 2, (D - 0.02) / 2, zb / 2 + 0.001))
            assign(bed, stone)

    # ---- art-directed ageing of the timber: dark damp along the ground line, the sill and the lower
    # 0.4 m bleached silver, sheltered soot under the plate, tannin stains under the ironwork,
    # dirt in the shakes
    bleach = noise_mask(0.3, seed=p.seed + 31, lo=0.4, hi=0.85)
    mossn = noise_mask(0.25, seed=p.seed + 37, lo=0.4, hi=0.75)
    SH = np.array(fr.shk, dtype=float) if fr.shk else None
    ST = np.array(stains, dtype=float) if stains else None

    def tm_masks(g0, lowf=None, edgef=None, hb=0.0):
        def tm_grime(co, n):
            x, z = co[:, 0], co[:, 2]
            low = np.clip(1.0 - (z - zb) / 0.11, 0.0, 1.0) ** 0.8 * 0.85
            topk = np.clip(1.0 - (zt - z) / 0.5, 0.0, 1.0) * 0.3
            g = low + topk + 0.05 + 0.22 * g0
            if ST is not None:
                t = (ST[None, :, 1] - z[:, None]) / ST[None, :, 2]
                dx = (x[:, None] - ST[None, :, 0]) / ST[None, :, 3]
                g = g + (np.exp(-dx * dx) * np.clip(1.0 - t, 0.0, 1.0) * (t > -0.15) * (co[:, 1] < 0.03)[:, None]).max(axis=1) * 0.75
            g = g * (0.45 + 0.75 * age)
            if lowf is not None and len(lowf) == len(co):
                # sun- and rain-bleached feet: the pale wood darkens back to the stained oak at the top of the zone
                u = np.where(edgef, 1.0, np.clip((z - iz0) / max(hb - iz0, 0.05), 0.0, 1.0) ** 1.3)
                g = np.where(lowf, np.maximum(G_MATCH * u, low * 0.8) + 0.08 * g0, g)
            if SH is not None:
                dmin = np.sqrt(((co[:, None, :] - SH[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
                g = np.where(dmin < 0.0015, 1.0, g)
            return np.clip(g, 0.0, 1.0)

        def tm_wear(co, n):
            return np.clip(bleach(co, n) * (0.05 + 0.2 * age) * (0.4 + 0.6 * g0), 0.0, 1.0)

        def tm_moss(co, n):
            low = np.clip(1.0 - (co[:, 2] - zb) / 0.22, 0.0, 1.0)
            return low * mossn(co, n) * 0.5 * age

        return tm_grime, tm_wear, tm_moss

    G_MATCH = 0.5
    bleachm = mat("wood_oak", _mix(p.timber, "#bdb7ac", 0.46))
    rbz = rng.fork("bleach")
    for o in fr.objs:
        lowf, edgef, hb = None, None, 0.0
        zs_ = [v.co.z for v in o.data.vertices]
        if age > 0.05 and o.name.split(".")[0].rstrip("0123456789") in ("post", "stud", "jamb", "brace", "knee", "chev", "cross") and min(zs_) < iz0 + 0.2 \
                and max(zs_) > iz0 + 0.5:
            hb = iz0 + rbz.u(0.28, 0.5) * (0.5 + age)
            jp, jf = rbz.u(0, 6.28), rbz.u(90.0, 160.0)
            # ragged upper limit: the bleaching follows the grain further up on some facets
            assign_faces(o, bleachm, lambda c, n, hb=hb, jp=jp, jf=jf: c.z < hb + 0.16 * math.sin(jf * (c.x + 0.6 * c.y) + jp))
            li = list(o.data.materials).index(bleachm)
            bm = bmesh.new()
            bm.from_mesh(o.data)
            cut = [e for e in bm.edges if len({f.material_index == li for f in e.link_faces}) == 2]
            if cut:
                bmesh.ops.split_edges(bm, edges=cut)
            bm.verts.index_update()
            lowf = np.array([any(f.material_index == li for f in v.link_faces) for v in bm.verts], dtype=bool)
            up = {(round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)) for v, lo in zip(bm.verts, lowf) if not lo}
            edgef = np.array([bool(lo) and (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)) in up for v, lo in zip(bm.verts, lowf)], dtype=bool)
            if edgef.any():
                hb = float(np.mean([v.co.z for v, e in zip(bm.verts, edgef) if e]))
                # ragged limit: slide the border vertices (both copies) along the grain
                bk = {(round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)) for v, e in zip(bm.verts, edgef) if e}
                for v in bm.verts:
                    if (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)) in bk:
                        v.co.z += 0.085 * math.sin(137.0 * v.co.x + 211.0 * v.co.y + jp)
            bm.to_mesh(o.data)
            bm.free()
        g, w_, s_ = tm_masks(fr.g0.get(o.name, 0.5), lowf, edgef, hb)
        paint(o, grime=g, wear=w_, special=s_)
    for o in extra:
        g, w_, s_ = tm_masks(0.5)
        paint(o, grime=g, wear=w_, special=s_)

    # ---- sockets
    if has_open:
        socket("window" if not door else "door", "attach", ((ox0 + ox1) / 2, 0.0, oz0), role="door" if door else "window",
               width=ox1 - ox0, height=oz1 - oz0)
        if not door:
            socket("planter", "attach", ((ox0 + ox1) / 2, 0.0, oz0 - 0.02), role="planter")
    socket("lantern", "attach", (-W / 2 + pw / 2, 0.0, min(iz1 - 0.25, zb + 2.25)), role="lantern")
    socket("sign", "attach", (W / 2 - pw / 2, 0.0, min(iz1 - 0.2, zb + 2.5)), role="sign")
