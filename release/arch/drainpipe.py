"""Cast-iron rainwater downpipe as a plumber would hang it: an ornate hopper head on a shaped back plate
(dated front, crest, moulded cornice, fed by a dressed lead wall spout or a length of beaded half-round eaves
gutter on strap brackets), a one-piece swan-neck offset (socket, flattened bends, spigot bead), socketed pipe
lengths whose sockets carry flat pierced lugs bolted to the wall over distance bobbins, and a shoe at the foot
that discharges over a flush dished gully stone with a barred grate and a low kerb against the wall.
Weathering is built into the surface: every casting is skinned ring by ring and the skin is cut along noisy
iso-lines into paint / rust (or verdigris) / algae, so rust fans down from the socket joints and lug bolts and
green-black slime creeps up the foot and weeps down the back of the swan-neck.
Wall plane y = 0, ground z = 0, hopper rim at z = height; the pipe run stands at x = offset.
"""
from math import sin, cos, pi, radians, degrees, atan2, hypot, sqrt, ceil

import numpy as np
from tk import *

META = {"name": "Gutter and downpipe", "class": "component", "mount": "wall", "mount_height": 0.0, "focus": 0.9}
PARAMS = {
    "height": {"default": 4.4, "min": 2.2, "max": 12.0, "desc": "height of the hopper head rim above the ground (m)"},
    "section": {"default": "round", "choices": ["round", "square"], "desc": "pipe cross-section"},
    "size": {"default": 0.1, "min": 0.075, "max": 0.15, "desc": "pipe diameter / side (m)"},
    "offset": {"default": 0.3, "min": -0.8, "max": 0.8, "desc": "sideways swan-neck offset of the pipe run from the hopper (m, 0 = straight drop)"},
    "hopper": {"default": "box", "choices": ["box", "gothic", "urn"], "desc": "hopper head style: moulded box, crenellated half-octagon, gadrooned bowl"},
    "hopper_size": {"default": 1.0, "min": 0.75, "max": 1.4, "desc": "hopper head scale"},
    "date": {"default": "1887", "desc": "date cast on the hopper front"},
    "crest": {"default": "shield", "choices": ["shield", "rosette", "none"], "desc": "crest cast on the hopper"},
    "gilded": {"default": True, "desc": "date and shield picked out in gold leaf (painted iron / zinc only)"},
    "inlet": {"default": "spout", "choices": ["spout", "gutter", "none"], "desc": "what feeds the hopper: lead wall spout or eaves gutter"},
    "gutter_len": {"default": 2.0, "min": 0.6, "max": 6.0, "desc": "length of the eaves gutter (m, inlet = gutter)"},
    "fascia": {"default": True, "desc": "painted timber fascia board behind the eaves gutter"},
    "pipe_len": {"default": 1.83, "min": 0.9, "max": 2.5, "desc": "nominal length of one socketed pipe (m)"},
    "metal": {"default": "cast_iron", "choices": ["cast_iron", "copper", "zinc"], "desc": "painted cast iron, patinated copper or zinc"},
    "paint": {"default": "#4c5e4c", "desc": "paint colour (cast iron)"},
    "ageing": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "rust bleeding / green staining amount"},
    "gully": {"default": True, "desc": "dished stone gully with grate under the shoe"},
}
VARIANTS = {
    "gothic_square": {"section": "square", "hopper": "gothic", "paint": "#7d2a22", "height": 5.2, "offset": -0.36,
                      "date": "1862", "size": 0.11, "seed": 4},
    "copper_urn": {"metal": "copper", "hopper": "urn", "height": 3.6, "offset": 0.0, "date": "1904", "crest": "rosette",
                   "hopper_size": 1.15, "ageing": 0.7, "seed": 9},
    "eaves_gutter": {"inlet": "gutter", "gutter_len": 2.6, "height": 3.0, "offset": 0.22, "paint": "#3d5676",
                     "crest": "rosette", "date": "1911", "hopper_size": 0.85, "seed": 15},
    "neglected_tall": {"height": 6.8, "ageing": 1.0, "paint": "#5a6062", "gilded": False, "offset": 0.5, "hopper": "gothic",
                       "date": "1848", "seed": 23},
}

SOCK = 0.122            # height of a pipe socket (bell)
# (growth over the barrel radius, height above the socket foot, flag of the row that starts here)
SOCK_PROF = [(0.0045, 0.008, ""), (0.011, 0.02, ""), (0.011, 0.058, ""), (0.011, 0.096, "h"), (0.0165, 0.103, "h"),
             (0.0165, 0.113, ""), (0.0125, 0.122, ""), (0.0048, 0.116, "i")]
B, HI, ST, AL = 0, 1, 2, 3   # material roles of a skin: paint, worn edge paint, stain (rust / verdigris), algae


def mix_hex(a, b, t):
    try:
        pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
        pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(int(round(lerp(x, y, t))) for x, y in zip(pa, pb))
    except Exception:
        return a


def vnoise(P, scale, seed):
    """Smooth value noise 0..1 at points (N,3)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    return np.asarray(noise_mask(scale, int(seed), 0.0, 1.0)(P, None), dtype=np.float64)


def streak_field(ang, d, stk):
    """Runs that start at d = 0 and taper to a point: stk = [(angle, half width rad, length m, strength, wander)]."""
    f = np.zeros_like(d)
    for a0, w, L, amp, wob in stk:
        da = (ang - a0 - wob * np.sin(d * 11.0 + a0 * 5.0) + pi) % (2 * pi) - pi
        u = np.clip(d / L, 0.0, 1.0)                        # the run spreads a little and thins out as it goes down
        f = np.maximum(f, np.where(d > -0.006, amp * (1.0 - u) ** 0.7 * np.exp(-(da / (w * (0.6 + 0.7 * u))) ** 2), 0.0))
    return f


def skin(name, P, F, rows, mats, grime=None, closed=True, cap_first=False, cap_last=False, sm=40.0, T=0.5):
    """Skin a stack of rings P[R][n] and cut every quad along the iso-line F = T (marching squares):
    rows[k] = (material outside, material inside or None) for the row between ring k and k + 1.
    Shared, welded vertices - the two materials meet on one watertight surface."""
    P = np.asarray(P, dtype=np.float64)
    R, n = P.shape[0], P.shape[1]
    Ff = (np.zeros(R * n) if F is None else np.asarray(F, dtype=np.float64).reshape(-1)).tolist()
    vg = (np.zeros(R * n) if grime is None else np.asarray(grime, dtype=np.float64).reshape(-1)).tolist()
    verts = [tuple(v) for v in P.reshape(-1, 3).tolist()]
    faces, fmat, cross = [], [], {}

    def xv(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cross:
            fa, fb = Ff[key[0]], Ff[key[1]]
            t = min(0.88, max(0.12, (T - fa) / (fb - fa)))
            pa, pb = verts[key[0]], verts[key[1]]
            verts.append(tuple(pa[i] + (pb[i] - pa[i]) * t for i in range(3)))
            vg.append(vg[key[0]] + (vg[key[1]] - vg[key[0]]) * t)
            cross[key] = len(verts) - 1
        return cross[key]

    if cap_first:
        faces.append(tuple(reversed(range(n))))
        fmat.append(rows[0][0])
    for k in range(R - 1):
        mo, mi = rows[k]
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            q = [k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i]
            ins = [mi is not None and Ff[v] >= T for v in q]
            c = sum(ins)
            if c == 0 or c == 4:
                faces.append(q)
                fmat.append(mi if c == 4 else mo)
            elif c == 2 and ins[0] == ins[2]:                # saddle: four corner triangles + centre quad
                for s_ in range(4):
                    faces.append([xv(q[s_ - 1], q[s_]), q[s_], xv(q[s_], q[(s_ + 1) % 4])])
                    fmat.append(mi if ins[s_] else mo)
                faces.append([xv(q[0], q[1]), xv(q[1], q[2]), xv(q[2], q[3]), xv(q[3], q[0])])
                fmat.append(mo)
            else:
                pin, pout = [], []
                for s_ in range(4):
                    a, b = q[s_], q[(s_ + 1) % 4]
                    (pin if ins[s_] else pout).append(a)
                    if ins[s_] != ins[(s_ + 1) % 4]:
                        x = xv(a, b)
                        pin.append(x)
                        pout.append(x)
                faces += [pin, pout]
                fmat += [mi, mo]
    if cap_last:
        faces.append(tuple(range((R - 1) * n, R * n)))
        fmat.append(rows[-1][0])
    o = mesh_object(name, verts, faces)
    used = sorted(set(fmat))
    for m in used:
        o.data.materials.append(mats[m])
    if len(o.data.polygons) == len(faces):
        o.data.polygons.foreach_set("material_index", [used.index(m) for m in fmat])
    else:
        print("DBG skin lost faces", name, len(o.data.polygons), len(faces))
    smooth(o, sm)
    garr = np.clip(np.asarray(vg, dtype=np.float32), 0.0, 1.0)
    if len(o.data.vertices) == len(garr):
        paint(o, grime=lambda co, nrm: garr)
    return o


def shield_pts(w, h, n=5):
    pts = [(0.0, -h / 2)]
    for k in range(n - 1, -1, -1):
        a = radians(90.0 * k / n)
        pts.append((w / 2 * cos(a) ** 0.8, 0.1 * h - 0.6 * h * sin(a)))
    pts += [(w / 2, h / 2), (0.0, h / 2 - 0.08 * h), (-w / 2, h / 2)]
    for k in range(0, n):
        a = radians(90.0 * k / n)
        pts.append((-w / 2 * cos(a) ** 0.8, 0.1 * h - 0.6 * h * sin(a)))
    return pts


def build(p, rng):
    H, s, r, A = p.height, p.hopper_size, p.size / 2, p.ageing
    square = p.section == "square"
    copper, zinc = p.metal == "copper", p.metal == "zinc"
    off = p.offset if abs(p.offset) > 0.06 else 0.0
    yc = -(r + 0.03)                  # pipe axis stands 30 mm off the wall on bobbins
    yb = -0.006                       # front of the hopper back plate
    sd = int(p.seed) * 7 + 3

    # ------------------------------------------------------------------ materials
    paint_hex = p.paint if isinstance(p.paint, str) and p.paint.startswith("#") and len(p.paint) == 7 else "#4c5e4c"
    rusty = mat("iron_rusty")
    if copper:
        body = mat("iron_painted", "#8a583b", rough=1.1, scale=2.0)   # brown oxidised copper (dull, sheltered faces)
        hi = mat("brass", "#eeb08a")                         # rubbed bright copper on beads and rims
        stain = mat("copper_patina")                         # verdigris where the water runs
        algae, fixm, gk, gka, heavy_k = stain, body, 0.2, 0.2, 1.7
    elif zinc:
        body = mat("zinc")
        hi = body
        stain, algae, fixm, gk, gka, heavy_k = rusty, mat("iron_painted", "#39422b", rough=3.0), rusty, 0.9, 1.0, 0.6
    else:
        body = mat("iron_painted", paint_hex, rough=1.3, scale=2.0)    # worn satin paint over sand-cast iron
        hi = mat("iron_painted", mix_hex(paint_hex, "#d2d6cc", 0.3), rough=0.9, scale=2.0)   # paint rubbed thin on beads, rims, lugs
        stain, algae, fixm, gk, gka, heavy_k = rusty, mat("iron_painted", "#343d25", rough=3.0), rusty, 1.0, 1.0, 1.0
    MATS = [body, hi, stain, algae]
    # copper heads are never gilded: date and crest stay in the metal as raised relief
    gilt = body if copper else (mat("metal_plain", "#d4a84b", rough=0.55) if p.gilded else hi)
    lead = body if copper else mat("iron_painted", "#756c5e", rough=1.6, scale=1.5)   # dull dark lead, no blue in it
    AG = 0.35 + 0.9 * A               # algae / weep strength
    ALG_TOP = 0.55 + 0.5 * A          # how far the green creeps up the foot of the pipe

    iron, his, gold, fixings, leads, skins, mosses = [], [], [], [], [], [], []

    # ------------------------------------------------------------------ pipe section
    if square:
        def sec(g=0.0, fx_=1.0, fy_=1.0):
            hx = r + g
            rad = max(0.003, min(0.014 + g, hx - 0.002))
            arcs = []
            for sx, sy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
                arcs.append([(sx * (hx - rad) + rad * cos(radians(a0 + 45 * k)), sy * (hx - rad) + rad * sin(radians(a0 + 45 * k)))
                             for k in range(3)])
            pts = []
            for c in range(4):
                pts += arcs[c]
                a, b = arcs[c][-1], arcs[(c + 1) % 4][0]
                pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            return [(x * fx_, y * fy_) for x, y in pts]
    else:
        NS = 12

        def sec(g=0.0, fx_=1.0, fy_=1.0):
            return [((r + g) * cos(2 * pi * k / NS) * fx_, (r + g) * sin(2 * pi * k / NS) * fy_) for k in range(NS)]
    ANG = np.array([atan2(dy, dx) for dx, dy in sec(0.0)])
    NP = len(ANG)

    def vring(cx, z, g=0.0):
        return np.array([(cx + dx, yc + dy, z) for dx, dy in sec(g)])

    def rust_F(P, d, stk, sdk, rim=0.0, ang=None):
        nz = vnoise(P * np.array([1.0, 1.0, 0.4]), 0.035, sdk)
        f = streak_field(ANG if ang is None else ang, d, stk) * (0.6 + 0.8 * nz)
        if stk:                                             # chipped paint: small rusty scabs round the joint
            chip = np.clip((vnoise(P, 0.024, sdk + 7) - 0.68) * 7.0, 0.0, 1.1) * np.clip(1.0 - d / (0.3 + 0.5 * A), 0.0, 1.0) ** 0.5
            f = np.maximum(f, np.where(d > 0.0, chip, 0.0))
        if rim > 0.0:
            f = np.maximum(f, np.where(d < 0.03, rim * (0.25 + 0.9 * vnoise(P, 0.05, sdk + 3)), 0.0))
        return f

    def algae_F(P):
        z = P[:, 2]
        return 0.1 + AG * (0.55 * np.clip(1.0 - z / ALG_TOP, 0.0, 1.0) ** 0.8 + 0.45 * vnoise(P, 0.06, sd + 11)
                           + 0.12 * (0.5 + 0.5 * np.sin(ANG)))

    def stain_g(F):
        return gk * (0.22 * np.clip((F - 0.1) / 0.4, 0.0, 1.0) + 0.24 * np.clip((F - 0.55) / 0.35, 0.0, 1.0))

    def alg_g(F):
        return gka * 0.22 * np.clip((F - 0.15) / 0.35, 0.0, 1.0)

    def base_g(P):
        return 0.03 + 0.1 * vnoise(P, 0.09, sd + 13) + 0.34 * np.clip(1.0 - P[:, 2] / 0.8, 0.0, 1.0)

    def make_streaks(rs, heavy, ears=True, lmax=9.0):
        stk = []
        if A < 0.04 or heavy <= 0.0:
            return stk
        for _ in range(int(round((0.7 + 2.6 * A) * heavy * heavy_k * rs.u(0.7, 1.3)))):
            stk.append((radians(rs.u(-170.0, -10.0)), rs.u(0.42, 0.7) * (0.8 + 0.5 * A), min(lmax, SOCK + rs.u(0.22, 0.7) * (0.4 + 1.1 * A)),
                        rs.u(0.9, 1.2), rs.u(0.04, 0.16)))
        if ears:                                             # the lug bolts bleed too
            for a0 in (radians(-12.0), radians(-168.0)):
                if rs.chance(0.35 + 0.6 * A):
                    stk.append((a0, rs.u(0.32, 0.5), min(lmax, SOCK + rs.u(0.08, 0.3) * (0.4 + 1.1 * A)), 1.05, 0.05))
        return stk

    def add_socket(Ps, Fs, Gs, RM, cx, zs, stk, sdk, xof=None):
        """Bell socket rings on top of a barrel ring at zs (the row below must already be (B, ST))."""
        rim = (0.36 + 0.6 * A) if A > 0.04 else 0.0
        for g, dz, fl in SOCK_PROF:
            z = zs + dz
            P = vring(cx if xof is None else xof(z), z, g)
            if fl == "i":
                Ps.append(P), Fs.append(np.zeros(NP)), Gs.append(np.full(NP, 0.5)), RM.append((ST, None))
            else:
                F = rust_F(P, zs + SOCK - P[:, 2], stk, sdk, rim)
                Ps.append(P), Fs.append(F), Gs.append(base_g(P) + stain_g(F)), RM.append((HI if fl == "h" else B, ST))

    def finish(name, Ps, Fs, Gs, RM, vertical=True, **kw):
        o = skin(name, np.array(Ps), np.array(Fs), RM[:len(Ps) - 1], MATS, np.array(Gs), **kw)
        if copper:                                           # pull the patina texture out into runs
            realize(o)
            uv_transform(o, scale=(1.0, 0.3) if vertical else (0.55, 0.55))
            uv_keep(o)
        skins.append(o)
        return o

    def pipe_length(name, cx, z_bot, z_top, rs, heavy=1.0, shear=0.0, sdk=0):
        """One cast length: spigot, barrel, bell socket at the top (the spigot of the piece above sits in it)."""
        zs, zb = z_top - SOCK, z_bot - 0.03
        stk = make_streaks(rs, heavy)
        lmax = max([q[2] for q in stk], default=0.0)
        zr = max(zb + 0.02, z_top - lmax - 0.02) if stk else zs
        za = min(ALG_TOP, zr - 0.04)
        zl = []
        if za > zb + 0.08:
            na = int(ceil((za - zb) / 0.1))
            zl += [(zb + (za - zb) * i / na, "a" if i < na else "") for i in range(na + 1)]
        else:                                               # the spigot end hides in the socket below: keep its shade local
            zl += [(zb, ""), (z_bot + 0.015, "")]
        if zr < zs - 0.01:
            zz, stp = [zs], 0.028
            while zz[-1] - stp > zr + 0.012:
                zz.append(zz[-1] - stp)
                stp *= 1.42
            zl.append((zr, "r0"))
            zl += [(z, "r") for z in reversed(zz)]
        else:
            zl.append((zs, "r"))

        def xof(z):
            return cx + shear * (z - z_top)
        Ps, Fs, Gs, RM = [], [], [], []
        for z, tag in zl:
            P = vring(xof(z), z)
            if tag == "a":
                F = algae_F(P)
                Ps.append(P), Fs.append(F), Gs.append(base_g(P) + alg_g(F)), RM.append((B, AL))
            elif tag == "r":
                F = rust_F(P, z_top - P[:, 2], stk, sdk)
                Ps.append(P), Fs.append(F), Gs.append(base_g(P) + stain_g(F)), RM.append((B, ST))
            else:
                Ps.append(P), Fs.append(np.zeros(NP)), Gs.append(base_g(P)), RM.append((B, ST if tag == "r0" else None))
        add_socket(Ps, Fs, Gs, RM, cx, zs, stk, sdk, xof)
        return finish(name, Ps, Fs, Gs, RM, cap_last=True)

    # =================================================================== hopper head
    st = p.hopper
    if st == "box":
        Bhw, Bd, Hh = 0.19 * s, 0.21 * s, 0.32 * s
        prof = [(-0.006, -0.66), (0.004, -0.655), (0.008, -0.63), (0.004, -0.605), (0, -0.60), (0, -0.21), (0.004, -0.205),
                (0.008, -0.18), (0.004, -0.155), (0.006, -0.14), (0.012, -0.10), (0.024, -0.05), (0.028, -0.04), (0.028, 0)]
        band = (-0.60, -0.21)
    elif st == "gothic":
        Bhw, Bd, Hh = 0.16 * s, 0.20 * s, 0.42 * s
        prof = [(-0.004, -0.50), (0.005, -0.495), (0.008, -0.475), (0.004, -0.455), (0, -0.45), (0, -0.14), (0.004, -0.135),
                (0.008, -0.115), (0.004, -0.095), (0.008, -0.08), (0.02, -0.03), (0.02, 0)]
        band = (-0.45, -0.14)
    else:
        Bhw, Bd, Hh = 0.185 * s, 0.20 * s, 0.36 * s
        prof = [(0.016, -0.485), (0.012, -0.47), (0, -0.455), (0, -0.17), (0.005, -0.165), (0.008, -0.145), (0.004, -0.125),
                (0.004, -0.11), (0.01, -0.07), (0.024, -0.03), (0.03, -0.015), (0.03, 0)]
        band = (-0.455, -0.17)
    hw0, d0 = r + 0.024, 2 * (r + 0.024)
    Bhw, Bd = max(Bhw, hw0 + 0.05), max(Bd, d0 + 0.03)
    rimg = prof[-1][0] * s
    NA = 14

    def hbase(hw, d, inset=0.0):
        y0 = yb - inset
        if st == "urn":
            pts = [(hw, y0), (-hw, y0)]
            for k in range(1, NA):
                a = pi + pi * k / NA
                pts.append((hw * cos(a), y0 + (d - inset) * sin(a)))
            return pts
        cx_, cy_ = (0.5 * hw, 0.55 * d) if st == "gothic" else (min(0.024 * s, hw * 0.3), min(0.024 * s, d * 0.3))
        return [(hw, y0), (-hw, y0), (-hw, yb - d + cy_), (-hw + cx_, yb - d), (hw - cx_, yb - d), (hw, yb - d + cy_)]

    nom = hbase(Bhw, Bd)
    HC = []
    for i in range(len(nom)):
        a_, b_ = nom[i], nom[(i + 1) % len(nom)]
        HC.append(1 if (i == 0 or st == "urn") else max(1, int(round(hypot(b_[0] - a_[0], b_[1] - a_[1]) / 0.055))))

    def hsec(hw, d, inset=0.0):
        b = hbase(hw, d, inset)
        out = []
        for i in range(len(b)):
            a_, c_ = b[i], b[(i + 1) % len(b)]
            out += [(lerp(a_[0], c_[0], k / HC[i]), lerp(a_[1], c_[1], k / HC[i])) for k in range(HC[i])]
        return out

    zhb = H - Hh                                           # underside of the hopper
    z_p0 = H + prof[0][1] * Hh
    hwp0, dp0 = Bhw + prof[0][0] * s, Bd + prof[0][0] * s
    spec = [(hw0, d0, 0.0, zhb, "f", False), (hw0 + 0.006, d0 + 0.008, 0.0, zhb + 0.014 * s, "f", False)]
    if st == "urn":
        for i in range(1, 6):
            t = i / 6
            e = sin(t * pi / 2) ** 0.85
            spec.append((lerp(hw0 + 0.006, Bhw + 0.016 * s, e), lerp(d0 + 0.008, Bd + 0.016 * s, e), 0.0,
                         H + Hh * lerp(-0.95, -0.485, t), "f", False))
    else:
        for t in (0.34, 0.67):
            spec.append((lerp(hw0 + 0.006, hwp0, t), lerp(d0 + 0.008, dp0, t), 0.0, lerp(zhb + 0.014 * s, z_p0, t), "f", False))
    gs = [g for g, _ in prof]

    def crown(j):
        return gs[j] > 0 and (j == 0 or gs[j] > gs[j - 1]) and (j == len(gs) - 1 or gs[j] > gs[j + 1])

    for i, (g, zr_) in enumerate(prof):
        last = i == len(prof) - 1
        hi_row = last or crown(i) or crown(i + 1) or i == len(prof) - 2
        spec.append((Bhw + g * s, Bd + g * s, 0.0, H + zr_ * Hh, "o0" if i == 0 else "o", hi_row))
        if not last and zr_ == band[0] and prof[i + 1][1] == band[1]:
            for t in (1 / 3, 2 / 3):                        # the plain band: two more rings so stains have something to bite
                spec.append((Bhw, Bd, 0.0, H + lerp(band[0], band[1], t) * Hh, "o", False))
    ins = 0.008 * s
    for g, zr_ in ((rimg - 0.012 * s, 0.0), (rimg - 0.014 * s, -0.06), (-0.008 * s, -0.22), (-0.008 * s, band[0])):
        spec.append((Bhw + g, Bd + g, ins, H + zr_ * Hh, "i", False))
    spec.append((hw0 + 0.012, d0 + 0.012, ins, H - 0.9 * Hh, "i", False))

    rh = rng.fork("hopper_stain")
    hstk = []
    if A > 0.04:
        for _ in range(int(round(rh.u(0.7, 1.3) * ((2.0 + 5.0 * A) if copper else (0.6 + 2.6 * A))))):
            if copper:
                ph = rh.u(0.12, pi - 0.12)
            else:                                           # runs keep to the cheeks and corners, clear of the date
                ph = rh.u(0.06, 0.8)
                ph = ph if rh.chance(0.5) else pi - ph
            hstk.append((ph, rh.u(0.2, 0.34) * (1.3 if copper else 1.0), Hh * rh.u(0.45, 0.95) * (0.55 + A) * (1.5 if copper else 1.0),
                         rh.u(0.9, 1.2), 0.04))
    zf1 = z_p0
    Ps, Fs, Gs, RM = [], [], [], []
    for hw, d, inset, z, kind, hi_row in spec:
        P = np.array([(x, y, z) for x, y in hsec(hw, d, inset)])
        if st == "urn":                                     # gadroons on the bowl
            t = (z - zhb) / (0.5 * Hh)
            if 0.03 < t < 1.0 and kind != "i":
                a = np.arctan2(yb - P[:, 1], P[:, 0])
                k = 1.0 + 0.075 * sin(pi * t) ** 0.7 * (np.abs(np.cos(7 * a)) - 0.55)
                P[:, 0] *= k
                P[:, 1] = yb + (P[:, 1] - yb) * k
        phi = np.arctan2(yb - P[:, 1], P[:, 0])
        nz = vnoise(P, 0.05, sd + 21)
        Fr = streak_field(phi, H - P[:, 2], hstk) * (0.7 + 0.6 * nz)
        Fw = 0.12 + AG * (0.5 * np.clip(1.0 - (z - zhb) / max(zf1 - zhb, 1e-3), 0.0, 1.0) ** 1.2 + 0.5 * nz)
        in_band = kind == "o" and H + band[0] * Hh - 1e-4 <= z <= H + band[1] * Hh + 1e-4
        # dull film: matt in the sunk panel, satin on the mouldings (a clean flat face flashes like foil under a lantern)
        Gb = ((0.3 + 0.16 * nz) if in_band else (0.15 + 0.22 * vnoise(P, 0.08, sd + 5))) * (0.5 if hi_row else 1.0)
        if kind == "i":
            Ps.append(P), Fs.append(np.zeros(len(P))), Gs.append(np.full(len(P), 0.42)), RM.append((B, None))
        elif kind == "f":
            F = np.maximum(Fw, Fr) if copper else Fw
            Ps.append(P), Fs.append(F), Gs.append(Gb + (stain_g(F) if copper else alg_g(F))), RM.append((B, ST if copper else AL))
        else:
            F = Fr if (copper or kind == "o") else np.zeros(len(P))
            Ps.append(P), Fs.append(F), Gs.append(Gb + stain_g(F)), RM.append((HI if hi_row else B, ST))
    hop = skin("hopper", np.array(Ps), np.array(Fs), RM[:len(Ps) - 1], MATS, np.array(Gs), cap_last=True, sm=38.0)
    if copper:
        realize(hop)
        uv_transform(hop, scale=(1.0, 0.3))
        uv_keep(hop)
    skins.append(hop)
    yf = yb - Bd                                           # front of the band
    zmid = H + Hh * (band[0] + band[1]) / 2
    bh = Hh * (band[1] - band[0])

    def slim(t):
        """Lettering sits on the casting: drop its hidden back and the outline points a 5 cm digit does not need."""
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(t.data)
        bmesh.ops.delete(bm, geom=[f_ for f_ in bm.faces if all(v.co.z < 1e-5 for v in f_.verts)], context='FACES')
        bmesh.ops.dissolve_limit(bm, angle_limit=radians(11.0), verts=bm.verts[:], edges=bm.edges[:])
        bm.to_mesh(t.data)
        bm.free()

    def cast_text(name, body_txt, size, cx, yfront, cz, max_w):
        t = text(name, body_txt, size=size, depth=0.006 * s, loc=(0, 0, 0), font="serif")
        mn, mx = world_bbox([t])
        w = mx.x - mn.x
        if w > max_w:
            delete([t])
            t = text(name, body_txt, size=size * max_w / w, depth=0.006 * s, loc=(0, 0, 0), font="serif")
            mn, mx = world_bbox([t])
        slim(t)
        place(t, loc=(cx - (mn.x + mx.x) / 2, yfront + 0.0015 - mx.y, cz - (mn.z + mx.z) / 2))
        gold.append(t)

    def rose(name, R, loc, rot, seg=8):
        return lathe(name, [(R, 0.0), (R * 1.04, R * 0.12), (R * 0.78, R * 0.3), (R * 0.5, R * 0.26),
                            (R * 0.38, R * 0.52), (0.0, R * 0.62)], segments=seg, loc=loc, rot=rot)

    def crest_at(loc, yaw, w):
        """Gilt shield with raised border and boss, or a cast rosette (left in the paint), facing -Y turned by yaw."""
        if p.crest == "none":
            return
        ca, sa = sin(radians(yaw)), -cos(radians(yaw))      # outward direction
        if p.crest == "shield":
            h = w * 1.22
            gold.append(extrude_profile("crest", shield_pts(w, h), 0.007 * s, loc=loc, rot=(90, 0, yaw), bevel=0.0015, segments=1))
            iron.append(extrude_profile("crest_field", shield_pts(w * 0.7, h * 0.7), 0.004 * s,
                                        loc=(loc[0] + ca * 0.006 * s, loc[1] + sa * 0.006 * s, loc[2] + 0.004 * s), rot=(90, 0, yaw)))
            gold.append(rose("crest_boss", w * 0.16, (loc[0] + ca * 0.0095 * s, loc[1] + sa * 0.0095 * s, loc[2] + 0.02 * w),
                             (90, 0, yaw), 8))
        else:
            R = w * 0.56
            (iron if copper else his).append(rose("crest_rose", R, loc, (90, 0, yaw), 12))
            # eight petals round the boss: a cast flower rather than a button
            pet = sphere("petal", R * 0.3, segments=6, rings=4, scale=(1.0, 1.0, 0.45))
            pm = [trs((loc[0] + (-sa) * R * 0.66 * cos(radians(a)) + ca * R * 0.16, loc[1] + ca * R * 0.66 * cos(radians(a)) + sa * R * 0.16,
                       loc[2] + R * 0.66 * sin(radians(a))), (90, 0, yaw)) for a in range(0, 360, 45)]
            (iron if copper else his).append(instances_to_mesh("crest_petals", pet, pm, keep_base=False))

    date = str(p.date).strip()[:8]
    if st == "box":
        fw = 2 * (Bhw - 0.024 * s) - 0.014 * s
        his.append(shape_extrude("front_frame", rounded_rect(fw, bh - 0.014 * s, 0.012 * s, 3),
                                 [rounded_rect(fw - 0.022 * s, bh - 0.036 * s, 0.006 * s, 3)], depth=0.005 * s,
                                 loc=(0, yf + 0.001, zmid), rot=(90, 0, 0)))
        tsz = min(0.09 * s, bh * 0.68)
        if p.crest != "none" and len(date) >= 2:
            h2 = len(date) // 2
            cast_text("date_l", date[:h2], tsz, -0.106 * s, yf, zmid, 0.112 * s)
            cast_text("date_r", date[h2:], tsz, 0.106 * s, yf, zmid, 0.112 * s)
            crest_at((0, yf + 0.001, zmid - 0.002), 0.0, min(0.074 * s, bh * 0.66))
        elif date:
            cast_text("date", date, tsz, 0.0, yf, zmid, fw - 0.05 * s)
        for sx in (-1, 1):                                  # rosettes on the cheeks
            his.append(rose("cheek_rose", 0.034 * s, (sx * Bhw, yb - Bd * 0.48, zmid), (90, 0, sx * 90)))
        for sx in (-1, 1):                                  # drop pendants under the front corners
            iron.append(lathe("pendant", [(0.0, -0.05 * s), (0.012 * s, -0.036 * s), (0.006 * s, -0.016 * s),
                                          (0.014 * s, -0.004 * s), (0.014 * s, 0.0)], segments=6,
                              loc=(sx * (Bhw - 0.012 * s), yf + 0.014 * s, H + band[0] * Hh - 0.012 * s)))
    elif st == "gothic":
        if date:
            cast_text("date", date, min(0.05 * s, bh * 0.5), 0.0, yf, zmid, Bhw - 0.03 * s)
        nx, ny = 0.55 * Bd, -0.5 * Bhw
        yaw = degrees(atan2(nx, -ny))
        for sx in (-1, 1):
            crest_at((sx * (0.75 * Bhw - 0.0005), yb - Bd + 0.275 * Bd + 0.0005, zmid), sx * yaw, 0.062 * s)
        # crenellated rim: merlons on every facet
        top = hbase(Bhw + rimg, Bd + rimg)
        edges = [(top[1], top[2]), (top[2], top[3]), (top[3], top[4]), (top[4], top[5]), (top[5], top[0])]
        mer = []
        mw, mt, mh = 0.046 * s, 0.014 * s, 0.036 * s
        for a_, b_ in edges:
            dx, dy = b_[0] - a_[0], b_[1] - a_[1]
            Le = hypot(dx, dy)
            cnt = max(1, int(round(Le / (0.085 * s))))
            for i in range(cnt):
                f = (i + 0.5) / cnt
                mer.append(box("merlon", (min(mw, Le / cnt * 0.6), mt, mh),
                               loc=(a_[0] + dx * f - dy / Le * mt * 0.5, a_[1] + dy * f + dx / Le * mt * 0.5, H + mh / 2 - 0.002),
                               rot=(0, 0, degrees(atan2(dy, dx))), bevel=0.002, segments=1))
        mj = join(mer, "merlons")
        assign(mj, hi)
        his.append(mj)
        # moulded string under the band
        sr = [np.array([(x, y, H + Hh * (-0.30) + dz * s) for x, y in hbase(Bhw + g * s, Bd + g * s)])
              for g, dz in ((0.0, -0.012), (0.006, -0.008), (0.008, 0.0), (0.006, 0.008), (0.0, 0.012))]
        his.append(skin("gothic_string", np.array(sr), None, [(HI, None)] * 4, MATS, sm=38.0))
    else:
        cw, chh = 0.165 * s, 0.07 * s
        sag = Bd * (1.0 - sqrt(max(0.0, 1.0 - (cw / 2 / Bhw) ** 2)))
        car = extrude_profile("cartouche", rounded_rect(cw, chh, 0.014 * s, 3), sag + 0.01 * s, loc=(0, yf + sag, zmid),
                              rot=(90, 0, 0), bevel=0.002, segments=1)
        assign(car, body)
        paint(car, grime=0.42)
        if date:
            cast_text("date", date, 0.042 * s, 0.0, yf - 0.01 * s, zmid, cw - 0.03 * s)
        for sx in (-1, 1):
            ph = radians(62.0)
            x_, y_ = Bhw * sin(ph), yb - Bd * cos(ph)
            yaw = degrees(atan2(sin(ph) / Bhw, cos(ph) / Bd))
            crest_at((sx * (x_ - 0.002), y_ + 0.002, zmid), sx * yaw, 0.058 * s)

    # back plate with ogee cresting, bolted to the wall
    W = Bhw + rimg + 0.03 * s
    crest_h = (0.215 if p.inlet == "spout" else 0.075) * s
    bp = [(-(hw0 + 0.035), -Hh * 0.97), (hw0 + 0.035, -Hh * 0.97), (W, -Hh * 0.5), (W, 0.03 * s)]
    for k in range(1, 9):
        t = k / 9
        bp.append((W * (1 - t ** 0.85) + 0.045 * s * t, 0.03 * s + (crest_h - 0.03 * s) * (0.5 - 0.5 * cos(pi * t))))
    bp += [(-x, z) for x, z in reversed(bp[2:])]
    bpl = extrude_profile("back_plate", bp, 0.006, loc=(0, 0, H), rot=(90, 0, 0), bevel=0.0012, segments=1)
    assign(bpl, body)
    paint(bpl, grime=lambda co, n: 0.33 + 0.14 * vnoise(co, 0.07, sd + 15))
    def dome(name, R, h, seg=6):
        return lathe(name, [(R, 0.0), (R * 0.8, h * 0.65), (0.0, h)], segments=seg)

    bb = dome("bp_bolt", 0.011, 0.008)
    bms = [trs((sx * W * 0.8, -0.006, H + 0.016 * s), (90, 0, 0)) for sx in (-1, 1)]
    bms += [trs((sx * (hw0 + 0.012), -0.006, H - Hh * 0.9), (90, 0, 0)) for sx in (-1, 1)]
    if p.inlet != "gutter":
        iron.append(cylinder("back_finial", 0.047 * s, 0.006, loc=(0, -0.003, H + crest_h + 0.012 * s), rot=(90, 0, 0), verts=12))
        bms.append(trs((0, -0.006, H + crest_h + 0.012 * s), (90, 0, 0)))
    fixings.append(instances_to_mesh("bp_bolts", bb, bms, keep_base=False))

    # outlet spigot under the hopper: the weep that feeds the slime on the swan-neck starts here
    Ps, Fs, Gs, RM = [], [], [], []
    for g, dz in ((0, -0.1), (0, -0.05), (0, -0.022), (0.007, -0.016), (0.009, -0.008), (0.014, 0.004)):
        P = vring(0.0, zhb + dz, g)
        F = (0.1 + AG * (0.45 + 0.6 * vnoise(P, 0.045, sd + 23))) if dz > -0.09 else np.zeros(NP)
        Ps.append(P), Fs.append(F), Gs.append(0.1 + (stain_g(F) if copper else alg_g(F))), RM.append((B, AL))
    finish("outlet", Ps, Fs, Gs, RM)

    # =================================================================== inlet
    rr = rng.fork("inlet")
    if p.inlet == "spout":
        zc = H + 0.108 * s
        rsp = 0.034 * s
        lip = np.array([0.0, -0.108 * s, zc - 0.08 * s])
        sp = sweep("spout", bezier_path((0, -0.001, zc), (0, -0.06 * s, zc), (0, -0.095 * s, zc - 0.02 * s), tuple(lip), n=6),
                   rsp, sides=8, cap=False, up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 1.0 + 0.12 * smoothstep(0.7, 1.0, t))
        solidify(sp, 0.004, offset=-1.0)
        leads.append(sp)
        # hand-dressed lead flange tacked to the back plate, and the wiped (soldered) joint swelling round the pipe
        fl = extrude_profile("spout_flange", rounded_rect(0.156 * s, 0.15 * s, 0.03 * s, 3), 0.004, loc=(0, -0.006, zc - 0.004 * s),
                             rot=(90, 0, 0), bevel=0.0015, segments=1)
        displace_noise(fl, strength=0.0035 * s, scale=0.045, seed=p.seed + 3, along_normal=False, axis_mask=(1, 0, 1))
        leads.append(fl)
        wj = lathe("wiped_joint", [(rsp + 0.03 * s, 0.0), (rsp + 0.027 * s, 0.005), (rsp + 0.016 * s, 0.013), (rsp + 0.008 * s, 0.024),
                                   (rsp + 0.004 * s, 0.036), (rsp + 0.0005, 0.044)], segments=10, loc=(0, -0.0095, zc), rot=(90, 0, 0))
        displace_noise(wj, strength=0.002, scale=0.03, seed=p.seed + 4)
        leads.append(wj)
        nb = dome("clout", 0.006, 0.003, 5)
        fixings.append(instances_to_mesh("flange_nails", nb, [trs((sx * 0.06 * s, -0.0105, zc - 0.004 * s + sz * 0.057 * s), (90, 0, 0))
                                                              for sx in (-1, 1) for sz in (-1, 1)], keep_base=False))
    elif p.inlet == "gutter":
        Rg, stl = 0.075, 0.02                               # stilted half-round with rolled rims
        yw = -0.024 if p.fascia else 0.0                    # face the brackets screw to
        yg = min(yw - Rg - 0.03, yc - 0.01)
        L = p.gutter_len
        x0 = -L * rr.pick([0.22, 0.5, 0.78])
        x1 = x0 + L
        zr0 = H + 0.15 + Rg + stl                            # rim level at the outlet

        def rim_z(x):
            return zr0 + 0.005 * abs(x)                    # laid to a fall towards the outlet

        def trough(R, t, bead):
            pr = [(R * cos(radians(a)), -stl + R * sin(radians(a))) for a in np.arange(180.0, 360.1, 22.5)]
            pr += [(R + bead, -0.012), (R + bead, 0.0), (R - t, 0.0)]
            pr += [((R - t) * cos(radians(a)), -stl + (R - t) * sin(radians(a))) for a in np.arange(360.0, 179.9, -22.5)]
            pr += [(-R - bead, 0.0), (-R - bead, -0.012)]
            return pr

        nseg = max(2, int(L / 0.55))
        xs = sorted(set([x0, x1, 0.0] + [x0 + L * i / nseg for i in range(1, nseg)]))
        gut = sweep("gutter", [(x, yg, rim_z(x)) for x in xs], trough(Rg, 0.005, 0.007), cap=True, up=(0, 0, 1), smooth_path=False)
        smooth(gut, 35)
        iron.append(gut)
        # silt and leaf mould lying in the sole: what you see when you look into an old gutter
        ws = 0.8 * (Rg - 0.005)
        sr_ = [[(x, yg + ws * rr.u(0.85, 1.0), rim_z(x) - stl - 0.6 * (Rg - 0.005) + rr.u(-0.004, 0.004)),
                (x, yg - ws * rr.u(0.85, 1.0), rim_z(x) - stl - 0.6 * (Rg - 0.005) + rr.u(-0.004, 0.004))]
               for x in np.linspace(x0 + 0.004, x1 - 0.004, max(4, int(L / 0.25)))]
        silt = skin("gutter_silt", np.array(sr_), None, [(0, None)] * (len(sr_) - 1), [mat("soil")], closed=False, sm=60.0)
        paint(silt, grime=lambda co, n: 0.35 * vnoise(co, 0.12, sd + 37))
        hd = [(-Rg - 0.011, 0.004)] + [((Rg + 0.006) * cos(radians(a)), -stl + (Rg + 0.006) * sin(radians(a))) for a in range(180, 361, 30)]
        hd += [(Rg + 0.011, 0.004)]
        for xe in (x0 - 0.005, x1 - 0.001):
            his.append(extrude_profile("stop_end", hd, 0.006, loc=(xe, yg, rim_z(xe)), rot=(90, 0, 90)))
        # drop outlet (nozzle) from the sole of the gutter into the hopper
        ztop, zbot = zr0 - stl - Rg + 0.01, H - 0.05 * s
        no = [np.array([(0.04 * g * cos(2 * pi * k / 12), yg + 0.04 * g * sin(2 * pi * k / 12), z) for k in range(12)])
              for g, z in ((0.88, zbot), (1.0, zbot), (1.0, ztop - 0.04), (1.12, ztop - 0.034), (1.12, ztop - 0.02), (1.45, ztop))]
        iron.append(skin("gutter_outlet", np.array(no), None, [(B, None), (B, None), (B, None), (HI, None), (B, None)], MATS))
        # union clips where lengths meet
        nj = int(L / 1.83)
        for i in range(1, nj + 1):
            xj = x0 + L * i / (nj + 1)
            if abs(xj) < 0.15:
                continue
            his.append(sweep("gutter_union", [(xj - 0.028, yg, rim_z(xj) + 0.001), (xj + 0.028, yg, rim_z(xj) + 0.001)],
                             trough(Rg + 0.005, 0.004, 0.006), cap=True, up=(0, 0, 1), smooth_path=False))
        # strap brackets screwed to the fascia / wall
        Rb = Rg + 0.0035
        path = [(0, yw - 0.004, 0.006), (0, yg + Rb + 0.008, 0.006), (0, yg + Rb, -stl)]
        path += [(0, yg + Rb * cos(radians(a)), -stl + Rb * sin(radians(a))) for a in range(-45, -181, -45)]
        path += [(0, yg - Rb - 0.008, 0.008)]
        flatbar = [(-0.0028, -0.014), (0.0028, -0.014), (0.0028, 0.014), (-0.0028, 0.014)]
        strap = sweep("strap", path, flatbar, up=(1, 0, 0), smooth_path=False)
        plate = box("bk_plate", (0.036, 0.006, 0.2), loc=(0, yw - 0.003, -0.07))
        brace = sweep("bk_brace", bezier_path((0, yw - 0.005, -0.155), (0, lerp(yw, yg, 0.4), -0.155), (0, lerp(yw, yg, 0.8), -stl - Rb - 0.035),
                                              (0, yg - 0.01, -stl - Rb - 0.001), n=4), flatbar, up=(1, 0, 0), smooth_path=False)
        b1 = box("bk_b1", (0.012, 0.005, 0.012), loc=(0, yw - 0.0085, -0.03))
        b2 = box("bk_b2", (0.012, 0.005, 0.012), loc=(0, yw - 0.0085, -0.135))
        for o in (strap, plate, brace):
            assign(o, body)
        for o in (b1, b2):
            assign(o, fixm)
        bk = join([strap, plate, brace, b1, b2], "bracket")
        nbk = max(2, int(round(L / 0.85)) + 1)
        bxs = [x0 + 0.1 + (L - 0.2) * i / (nbk - 1) for i in range(nbk)]
        bxs = [x if abs(x) > W + 0.03 else (W + 0.05) * (1 if x >= 0 else -1) for x in bxs]
        bxs = [x for x in bxs if x0 + 0.03 < x < x1 - 0.03]
        brk = instances_to_mesh("brackets", bk, [trs((x, 0, rim_z(x))) for x in bxs], keep_base=False)
        paint(brk, grime=lambda co, n: 0.08 + 0.2 * vnoise(co, 0.08, sd + 31))
        if p.fascia:
            fb = box("fascia", (L + 0.16, 0.024, 0.19), loc=((x0 + x1) / 2, -0.012, zr0 + 0.0025 * L - 0.075), bevel=0.003, segments=1)
            densify(fb, 0.2)
            uv_grain(fb, axis="X")
            assign(fb, mat("wood_painted", "#e6e1d2"))
            paint(fb, wear=lambda co, n: 0.42 * vnoise(co, 0.09, sd + 33) * (0.5 + A),
                  grime=lambda co, n: 0.1 + 0.3 * np.clip((zr0 - 0.03 - co[:, 2]) / 0.12, 0.0, 1.0) * vnoise(co * np.array([1.0, 1.0, 0.3]), 0.07, sd + 35))

    # =================================================================== swan neck + pipe run + shoe
    z_shoe = 0.47                                           # top of the shoe socket
    rp = rng.fork("pipes")
    if off:
        zn = zhb - 0.05                                     # top of the swan-neck socket
        avail = zn - SOCK - 0.08 - z_shoe - 0.3
        drop = max(0.12, min(0.24 + 0.95 * abs(off), avail))
        zB = zn - SOCK - drop - 0.08                        # mouth of the first pipe socket
        zA = zn - SOCK                                      # foot of the neck's own socket
        kk = drop * 0.55
        cl = [(off, zB - 0.035, 0.0, ""), (off, zB + 0.008, 0.0, "h"), (off, zB + 0.013, 0.0065, "h"), (off, zB + 0.025, 0.0065, "h"),
              (off, zB + 0.03, 0.0, "")]
        for x, _, z in bezier_path((off, yc, zB + 0.045), (off, yc, zB + 0.045 + kk), (0, yc, zA - 0.035 - kk), (0, yc, zA - 0.035), n=9):
            cl.append((x, z, 0.0, "w"))
        cl.append((0.0, zA, 0.0, "s"))
        nb0 = 5
        # tangents and how hard the path turns (flattened section in the bends, as a cast offset has)
        tang = []
        for i in range(len(cl)):
            a_, b_ = cl[max(i - 1, 0)], cl[min(i + 1, len(cl) - 1)]
            dx, dz = b_[0] - a_[0], b_[1] - a_[1]
            ln = hypot(dx, dz) or 1.0
            tang.append((dx / ln, dz / ln))
        tang[0] = tang[1] = tang[2] = tang[3] = tang[4] = (0.0, 1.0)
        tang[-1] = (0.0, 1.0)
        turn = [0.0] * len(cl)
        for i in range(1, len(cl) - 1):
            turn[i] = abs(atan2(tang[i + 1][0], tang[i + 1][1]) - atan2(tang[i - 1][0], tang[i - 1][1]))
        tmax = max(max(turn), 1e-3)
        a_c = radians(-132.0) if off > 0 else radians(-48.0)   # under-front flank where the weep clings
        WE = 0.45 + 1.0 * A
        Ps, Fs, Gs, RM = [], [], [], []
        nw = sum(1 for c in cl if c[3] == "w")
        iw = 0
        for i, (x, z, g, tag) in enumerate(cl):
            tx, tz = tang[i]
            fl_ = turn[i] / tmax
            P = np.array([(x + dx * tz, yc + dy, z - dx * tx) for dx, dy in sec(g, 1.0 - 0.1 * fl_, 1.0 + 0.035 * fl_)])
            if tag == "w":
                t = iw / (nw - 1)                           # 0 at the lower end, 1 under the hopper
                iw += 1
                da = (ANG - a_c + pi) % (2 * pi) - pi
                F = WE * (0.3 + 0.8 * t ** 0.6) * np.exp(-(da / 0.8) ** 2) * (0.6 + 0.8 * vnoise(P, 0.05, sd + 41))
                if iw == 1:
                    F *= 0.0
                Ps.append(P), Fs.append(F), Gs.append(base_g(P) + (stain_g(F) if copper else alg_g(F)) + 0.1 * np.clip(-np.cos(da), 0, 1) * 0),
                RM.append((B, AL))
            elif tag == "s":
                Ps.append(P), Fs.append(np.zeros(NP)), Gs.append(base_g(P) + 0.08), RM.append((B, ST))
            else:
                Ps.append(P), Fs.append(np.zeros(NP)), Gs.append(base_g(P)), RM.append((HI if tag == "h" else B, None))
        add_socket(Ps, Fs, Gs, RM, 0.0, zA, make_streaks(rp.fork("neck"), 0.5, ears=False, lmax=SOCK * 0.95), sd + 43)
        finish("swan_neck", Ps, Fs, Gs, RM, vertical=False, cap_last=True)
        z_top0 = zB
    else:
        z_top0 = zhb - 0.05
    Lrun = z_top0 - z_shoe
    n_len = max(1, int(Lrun / p.pipe_len + 0.8))
    Lp = Lrun / n_len
    ear_at = []
    for i in range(n_len):
        zt = z_shoe + (i + 1) * Lp
        pipe_length(f"pipe{i}", off, z_shoe + i * Lp, zt, rp.fork(f"len{i}"), shear=rp.jitter(0.0, 0.0022), sdk=sd + 50 + i * 3)
        ear_at.append(zt)

    # shoe: socket + bend kicking the water out over the gully, cast in one
    zs_sh = z_shoe - SOCK
    Rbend, ang = 0.115, radians(57.0)
    zb1 = zs_sh - 0.05
    ye, ze = yc - Rbend * (1 - cos(ang)), zb1 - Rbend * sin(ang)
    Lm = 0.085
    ym, zm = ye - Lm * sin(ang), ze - Lm * cos(ang)         # centre of the mouth

    def shoe_ring(cy, cz, a, g, sc=1.0):
        return np.array([(off + dx * sc, cy + dy * sc * cos(a), cz - dy * sc * sin(a)) for dx, dy in sec(g)])

    def along(dist):                                        # point on the spout axis, dist up from the mouth
        return ym + dist * sin(ang), zm + dist * cos(ang)
    Ps, Fs, Gs, RM = [], [], [], []
    Z0 = np.zeros(NP)
    for dist, g, sc, mats_, gr in ((0.055, -0.006, 1.03, (ST, None), 0.55), (0.0, -0.006, 1.1, (ST, None), 0.4)):
        cy, cz = along(dist)
        Ps.append(shoe_ring(cy, cz, ang, g, sc)), Fs.append(Z0), Gs.append(np.full(NP, gr)), RM.append(mats_)
    lip_rim = (0.5 + 0.7 * A) if A > 0.04 else 0.0
    for dist, g, sc, mats_ in ((0.0, 0.004, 1.1, (B, ST)), (0.013, 0.004, 1.08, (B, AL)), (0.022, 0.0, 1.06, (B, AL))):
        cy, cz = along(dist)
        P = shoe_ring(cy, cz, ang, g, sc)
        F = lip_rim * (0.25 + 0.9 * vnoise(P, 0.04, sd + 61)) if dist < 0.01 else algae_F(P)
        Ps.append(P), Fs.append(F), Gs.append(base_g(P) + (stain_g(F) if dist < 0.01 else alg_g(F))), RM.append(mats_)
    for i in range(5, -1, -1):
        a = ang * i / 5
        P = shoe_ring(yc - Rbend * (1 - cos(a)), zb1 - Rbend * sin(a), a, 0.0)
        F = algae_F(P)
        Ps.append(P), Fs.append(F), Gs.append(base_g(P) + alg_g(F)), RM.append((B, AL))
    P = vring(off, zs_sh)
    Ps.append(P), Fs.append(Z0), Gs.append(base_g(P)), RM.append((B, ST))
    add_socket(Ps, Fs, Gs, RM, off, zs_sh, make_streaks(rp.fork("shoe"), 0.45, lmax=SOCK * 0.95), sd + 63)
    finish("shoe", Ps, Fs, Gs, RM, vertical=False, cap_first=True, cap_last=True)
    ear_at.append(z_shoe)

    # flat pierced lugs cast on both sides of every socket, square washer + bolt driven into the wall over a bobbin
    Rs = r + 0.011
    y_l = min(-0.03, yc + 0.72 * Rs)                        # mid-plane of the lugs
    xr0 = Rs * (0.92 if square else 0.62)
    xr1 = Rs + 0.044
    bits = []
    for sx in (-1, 1):
        lp = [(xr0, -0.019), (xr1 - 0.007, -0.0145), (xr1, -0.0075), (xr1, 0.0075), (xr1 - 0.007, 0.0145), (xr0, 0.019)]
        if sx < 0:
            lp = [(-x, z) for x, z in reversed(lp)]
        lug = extrude_profile("lug", lp, 0.009, loc=(0, y_l + 0.0045, 0), rot=(90, 0, 0))
        assign(lug, hi)
        xb = sx * (Rs + 0.024)
        wsh = box("washer", (0.024, 0.0025, 0.024), loc=(xb, y_l - 0.0045 - 0.00125, 0), rot=(0, 8 * sx, 0))
        bt = cylinder("ear_bolt", 0.0085, 0.008, loc=(xb, y_l - 0.007 - 0.004, 0), rot=(90, 0, 0), verts=6)
        bob = cylinder("bobbin", 0.0115, -(y_l + 0.0045), loc=(xb, (y_l + 0.0045) / 2, 0), rot=(90, 0, 0), verts=6, cap=False)
        # rust-stained packing where the bobbin bears on the wall: the halo every old fixing leaves
        halo = extrude_profile("fix_halo", [(xb + 0.021 * (1.0 + 0.35 * sin(2.3 * k + sx)) * cos(2 * pi * k / 7),
                                             -0.004 + 0.023 * (1.0 + 0.3 * cos(1.7 * k + sx)) * sin(2 * pi * k / 7)) for k in range(7)],
                               0.0025, loc=(0, 0, 0), rot=(90, 0, 0))
        for o in (wsh, bt, bob, halo):
            assign(o, fixm)
        bits += [lug, wsh, bt, bob, halo]
    ear = join(bits, "ear_set")
    ears = instances_to_mesh("ears", ear, [trs((off, 0, z - 0.062), (0, rp.jitter(0.0, 1.5), 0)) for z in ear_at], keep_base=False)
    paint(ears, grime=lambda co, n: 0.1 + 0.22 * vnoise(co, 0.03, sd + 71) + 0.25 * np.clip(1.0 + co[:, 1] / 0.03, 0.0, 1.0))

    # =================================================================== stone gully
    if p.gully:
        stone_m = mat("stone_rough", "#857f73")
        moss_m = mat("moss")
        gw, gh, kh = 0.46, 0.016, 0.05                     # width, height above the paving, kerb height
        gx, gy = off, ym - 0.045                           # the water lands here
        gd = -gy + 0.185                                   # depth from the wall
        gq = 0.072                                         # half size of the grate rebate
        g2 = gq + 0.01
        xs = [-gw / 2, -gw / 2 + 0.014, -0.16, -g2, -gq, 0.0, gq, g2, 0.16, gw / 2 - 0.014, gw / 2]
        ys = [-gd, -gd + 0.014, -gd + 0.066, gy - g2, gy - gq, gy, gy + gq, gy + g2, (gy + g2 - 0.082) / 2,
              -0.082, -0.066, 0.0]
        rg = rng.fork("gully")
        MG = 0.45 + 0.9 * A
        Ps, Fs, Gs = [], [], []
        for y in ys:
            row = []
            for xr in xs:
                edge = abs(xr) >= gw / 2 - 1e-6 or y <= -gd + 1e-6
                dist = hypot(xr, (y - gy) * 1.1)
                hole = abs(xr) <= gq + 1e-4 and abs(y - gy) <= gq + 1e-4
                if edge:
                    z = 0.0
                elif y >= -0.0661:
                    z = kh
                elif hole:
                    z = 0.002
                elif y >= -0.0821:
                    z = gh
                else:
                    z = gh - 0.011 * smoothstep(0.17, 0.05, dist)
                row.append((gx + xr, min(y, 0.0), z))
            P = np.array(row)
            wob = vnoise(P, 0.11, sd + 81) - 0.5
            for i, xr in enumerate(xs):                     # hand-dressed: nothing quite true
                if abs(xr) >= gw / 2 - 1e-6:
                    P[i, 0] += 0.012 * wob[i]
                elif P[i, 2] > 0.004:
                    P[i, 2] += 0.004 * wob[i]
            if y <= -gd + 1e-6:
                P[:, 1] += 0.012 * wob
            dist = np.hypot(P[:, 0] - gx, (P[:, 1] - gy) * 1.1)
            nz = vnoise(P, 0.07, sd + 83)
            ringm = np.exp(-((dist - 0.135) / 0.05) ** 2) * (0.35 + 1.0 * vnoise(P, 0.13, sd + 85))
            F = MG * (0.55 * ringm + 0.5 * nz + 0.5 * np.exp(-((P[:, 1] + 0.088) / 0.028) ** 2))
            inner = (np.abs(P[:, 0] - gx) <= g2 + 5e-4) & (np.abs(P[:, 1] - gy) <= g2 + 5e-4)
            F = np.where(inner | (P[:, 2] < 0.001) | (P[:, 1] > -0.07), 0.0, F)
            G = 0.05 + 0.12 * nz + 0.55 * np.clip((0.21 - dist) / 0.15, 0.0, 1.0) ** 0.8 + 0.22 * np.clip((F - 0.2) / 0.3, 0.0, 1.0)
            G = np.where((np.abs(P[:, 0] - gx) <= gq + 1e-4) & (np.abs(P[:, 1] - gy) <= gq + 1e-4), 0.9, G)
            G = np.where(P[:, 1] > -0.07, 0.1 + 0.2 * nz, G)
            Ps.append(P), Fs.append(F), Gs.append(G)
        gul = skin("gully_stone", np.array(Ps), np.array(Fs), [(0, 1)] * (len(ys) - 1), [stone_m, moss_m], np.array(Gs), closed=False, sm=50.0)
        sq = [(-gq, -gq), (gq, -gq), (gq, gq), (-gq, gq)]
        gbits = [shape_extrude("grate", sq, [[(x * 0.84, y * 0.84) for x, y in sq]], depth=0.008, loc=(gx, gy, 0.001))]
        gbits += [box("grate_bar", (0.012, gq * 1.7, 0.007), loc=(gx + (i - 2) * 0.0235, gy, 0.005)) for i in range(5)]
        grate = join(gbits, "grate")
        assign(grate, rusty)
        paint(grate, grime=lambda co, n: 0.12 + 0.2 * vnoise(co, 0.03, sd + 87))

        def moss_strip(name, xa, xb, ybase, zbase, hmax, sdk, thr=0.45):
            """Ragged line of moss tufts growing in a joint: rises and dies away with the noise."""
            nu = max(4, int((xb - xa) / 0.02))
            rings_ = []
            gg = []
            for i in range(nu + 1):
                x = lerp(xa, xb, i / nu)
                q = np.array([[x, ybase, zbase]])
                h = hmax * float(np.clip((vnoise(q, 0.06, sdk)[0] * 0.7 + vnoise(q, 0.02, sdk + 1)[0] * 0.5 - thr) / (1.0 - thr), 0.0, 1.0)) ** 0.7
                if i in (0, nu):
                    h = 0.0
                j1, j2 = rg.u(0.7, 1.3), rg.u(0.7, 1.3)
                if h <= 1e-5:
                    rings_.append([(x, ybase - 1e-4 * k, zbase + 1e-4 * (3 - k)) for k in range(4)])
                else:
                    rings_.append([(x, ybase, zbase + 1.5 * h * j1 + 0.004), (x, ybase - (0.35 * h + 0.004), zbase + 1.15 * h * j2),
                                   (x, ybase - (1.1 * h + 0.01) * j1, zbase + 0.45 * h), (x, ybase - (1.7 * h + 0.012) * j2, zbase + 0.0005)])
                gg.append([0.3, 0.05 + 0.2 * rg.u(), 0.12, 0.3])
            o = skin(name, np.array(rings_), None, [(0, None)] * nu, [moss_m], np.array(gg), closed=False, sm=60.0)
            mosses.append(o)

        if A > 0.03:
            moss_strip("moss_wall_joint", gx - gw / 2 + 0.015, gx + gw / 2 - 0.015, -0.0005, kh + 0.001, 0.012 + 0.02 * A, sd + 91, 0.5 - 0.2 * A)
            moss_strip("moss_kerb_foot", gx - gw / 2 + 0.03, gx + gw / 2 - 0.03, -0.079, gh + 0.002, 0.008 + 0.014 * A, sd + 93, 0.58 - 0.2 * A)
            for sx in (-1, 1):                              # and along the foot of the wall either side of the stone
                xa = gx + sx * (gw / 2 + 0.004)
                xb_ = xa + sx * rg.u(0.12, 0.3)
                moss_strip("moss_wall_foot", min(xa, xb_), max(xa, xb_), -0.0005, 0.0005, 0.012 + 0.018 * A, sd + 95 + sx, 0.55 - 0.2 * A)
        fx("drip", "drip", (off, ym - 0.02, zm - r * 0.6), rate=0.5, size=0.02)

    # a little moss lodged on socket rims and the hopper rim once things are neglected
    rm = rng.fork("moss")

    def lump(name, loc, mr, sc, k):
        m = sphere(name, mr, loc=loc, segments=6, rings=4, scale=sc)
        displace_noise(m, strength=mr * 0.35, scale=mr * 1.1, seed=p.seed + k)
        assign(m, mat("moss"))
        paint(m, grime=lambda co, n: 0.08 + 0.25 * vnoise(co, 0.02, sd + k))
        mosses.append(m)

    for i, z in enumerate(ear_at[:-1]):
        if rm.chance(-0.3 + 0.8 * A):
            a = radians(rm.u(200, 340))
            mr = rm.u(0.012, 0.02)
            lump("moss_joint", (off + (r + 0.012) * cos(a), yc + (r + 0.012) * sin(a), z + 0.002), mr, (rm.u(1.3, 2.0), 1.0, 0.6), 40 + i)
    if A > 0.7:
        for k, sx in enumerate([rm.pick([-1, 1])]):
            mr = rm.u(0.016, 0.026) * s
            lump("moss_rim", (sx * Bhw * rm.u(0.55, 0.95), yb - 0.012 * s, H + 0.003), mr, (rm.u(1.5, 2.3), 0.8, 0.55), 31 + k)

    # =================================================================== materials + weathering of the loose castings
    def film(co, n):                                        # soot and dust film: matt in the hollows, satin where the rain washes
        return 0.12 + 0.17 * vnoise(co, 0.1, sd + 13) + 0.1 * np.clip(-n[:, 2], 0.0, 1.0)

    for o in iron:
        if not o.data.materials:
            assign(o, body)
        paint(o, grime=film)
    for o in his:
        if not o.data.materials:
            assign(o, hi)
        paint(o, grime=lambda co, n: 0.6 * film(co, n))
    for o in gold:
        assign(o, gilt)
        paint(o, grime=lambda co, n: (0.08 + 0.3 * A) * vnoise(co, 0.045, sd + 11))
    for o in fixings:
        if not o.data.materials:
            assign(o, fixm)
        paint(o, grime=0.32)
    if leads:
        zc_, ly, lz = H + 0.108 * s, -0.108 * s, H + 0.028 * s

        def lead_g(co, n):                                  # dull dark lead: white bloom on the weathered top, wet-dark lip
            g = 0.33 + 0.1 * (vnoise(co, 0.035, sd + 17) - 0.5)
            g -= 0.2 * np.clip(n[:, 2], 0.0, 1.0) * np.clip((vnoise(co, 0.018, sd + 19) - 0.35) * 3.0, 0.0, 1.0)   # patches of white bloom
            g += 0.08 * np.clip(-n[:, 2], 0.0, 1.0)
            dl = np.hypot(co[:, 1] - ly, co[:, 2] - lz)
            return g + 0.22 * np.clip(1.0 - dl / (0.05 * s), 0.0, 1.0)
        for o in leads:
            assign(o, lead)
            if not copper:
                paint(o, grime=lead_g)
            else:
                paint(o, grime=film)

    socket("inlet", "attach", (0, yb - Bd * 0.5, H), role="gutter")
