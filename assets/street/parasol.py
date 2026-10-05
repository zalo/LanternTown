"""Large canvas market / cafe parasol. A two-piece turned timber pole (brass ferrule joint) stands in a
cast-iron base with clamp screws; a fixed crown hub carries timber ribs strung on a wire ring, a sliding
runner pushes them up with timber stretchers (brass saddles, peg through the pole, hoist cord over a
pulley to a cleat). The canopy is one continuous cloth sewn from a canvas gore per bay: flat seam tape
over every rib and round the rim, slack that sags between the ribs, tension creases fanning out of the
rib-tip pockets, a rippling scalloped valance with a bound hem, a soft raised vent cap under a turned
finial and tie cords. Small oil lanterns hang from the stretchers; their light glows through the canvas.
`open` folds the real linkage: the cloth hangs in swags and finally wraps round the pole in spiral
pleats, tied with the cord.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, degrees, sqrt, atan2, hypot, ceil, exp
from mathutils import Vector, Matrix

META = {"name": "Cafe parasol", "class": "furniture", "mount": "ground", "focus": [0.25, -0.6, 1.76]}
PARAMS = {
    "diameter": {"default": 3.0, "min": 2.0, "max": 4.5, "desc": "canopy diameter over the rib tips when fully open (m)"},
    "height": {"default": 2.7, "min": 2.3, "max": 3.6, "desc": "height of the crown hub (m); raised automatically to keep ~1.8 m head room under the valance"},
    "ribs": {"default": 8, "choices": [6, 8, 10], "desc": "number of ribs / canvas gores"},
    "pitch": {"default": 22.0, "min": 12.0, "max": 34.0, "desc": "slope of the open canopy (deg)"},
    "open": {"default": 1.0, "min": 0.0, "max": 1.0, "desc": "1 = fully open, 0 = furled round the pole and tied with the cord"},
    "sag": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "slack of the canvas between the ribs"},
    "pattern": {"default": "plain", "choices": ["plain", "stripes", "awning", "panels"],
                "desc": "plain | stripes (concentric bands) | awning (stripes running down each gore) | panels (alternating gores)"},
    "colour": {"default": "#d8c8a2", "desc": "main canvas colour"},
    "colour2": {"default": "#8a2f25", "desc": "second canvas colour (stripes / alternate gores)"},
    "trim": {"default": "#96362a", "desc": "colour of seam tape and hem binding"},
    "stripe": {"default": 0.14, "min": 0.05, "max": 0.35, "desc": "stripe width (m)"},
    "valance": {"default": 0.2, "min": 0.0, "max": 0.36, "desc": "valance drop (m), 0 = none"},
    "valance_style": {"default": "scallop", "choices": ["scallop", "pointed", "straight"], "desc": "cut of the valance edge"},
    "tassels": {"default": True, "desc": "small tassels hung from the valance between the scallops"},
    "vent": {"default": True, "desc": "raised vent cap over an opening at the crown"},
    "base": {"default": "round", "choices": ["round", "cross", "stone"], "desc": "cast-iron dome base | four-legged cast cross | drilled stone block with iron socket"},
    "lanterns": {"default": 2, "min": 0, "max": 3, "desc": "oil lanterns hung from the stretchers under the canopy (only when open)"},
    "rim_lanterns": {"default": 2, "min": 0, "max": 3, "desc": "oil lanterns on iron hooks at the rib tips (only when open)"},
    "intensity": {"default": 5.0, "min": 1.0, "max": 20.0, "desc": "light per hanging oil lantern (cd); the small rim lanterns give a fifth of it (min 0.8)"},
    "wood": {"default": "#a9825a", "desc": "stain of pole and ribs"},
    "paint": {"default": "#35584a", "desc": "paint of the cast-iron base"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "sun-fade, dirt and rust amount"},
    "lean": {"default": 0.7, "min": 0.0, "max": 2.5, "desc": "how far the pole leans in its socket (deg)"},
}
VARIANTS = {
    "bistro_stripe": {"diameter": 2.5, "height": 2.55, "pitch": 27, "pattern": "stripes", "colour": "#e6dcc2", "colour2": "#9a2d24",
                      "trim": "#e6dcc2", "stripe": 0.13, "valance_style": "pointed", "valance": 0.17, "base": "cross", "lanterns": 1,
                      "wood": "#6b4a2e", "paint": "#2a3040", "seed": 4},
    "market_green": {"diameter": 4.0, "height": 3.0, "ribs": 10, "pitch": 18, "pattern": "awning", "colour": "#e3d9bd", "colour2": "#2f5a3c",
                     "trim": "#24452f", "stripe": 0.11, "valance_style": "straight", "valance": 0.26, "base": "stone", "lanterns": 3, "rim_lanterns": 1,
                     "intensity": 8.0, "sag": 0.8, "age": 0.75, "seed": 9},
    "harlequin": {"diameter": 2.3, "height": 2.5, "ribs": 6, "pitch": 30, "pattern": "panels", "colour": "#c9a24a", "colour2": "#27466b",
                  "trim": "#e8dcc0", "valance": 0.15, "vent": False, "base": "round", "lanterns": 1, "rim_lanterns": 3, "tassels": False, "paint": "#5a1c16", "seed": 15},
    "furled": {"open": 0.0, "colour": "#b9552f", "trim": "#e3d3ad", "base": "round", "paint": "#2a3040", "age": 0.8, "lean": 1.2, "seed": 22},
}


def _rect(w, h):
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def _bar(name, p0, p1, prof, up):
    return sweep(name, [tuple(p0), tuple(p1)], prof, smooth_path=False, up=tuple(up))


def _remap_uv(obj, fn):
    """fn(u01, v01 arrays) -> (U, V) in metres; the surface was built with uv_len=(1, 1)."""
    lay = obj.data.uv_layers.active or obj.data.uv_layers[0]
    arr = np.zeros(len(lay.data) * 2, dtype=np.float32)
    lay.data.foreach_get("uv", arr)
    a = arr.reshape(-1, 2)
    U, V = fn(a[:, 0].copy(), a[:, 1].copy())
    a[:, 0], a[:, 1] = U, V
    lay.data.foreach_set("uv", a.ravel())
    uv_keep(obj)


def _quad_mesh(name, verts, quads, uvs, midx=None, angle=60.0):
    """Mesh from explicit quads with per-corner UVs (metres) and an optional material slot per quad."""
    o = mesh_object(name, verts, quads)
    me = o.data
    assert len(me.polygons) == len(quads), name
    lay = me.uv_layers.new(name="UVMap")
    lay.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32).ravel())
    if midx is not None:
        me.polygons.foreach_set("material_index", np.asarray(midx, dtype=np.int32))
    uv_keep(o)
    smooth(o, angle)
    return o


def _vnormals(o):
    arr = np.empty(len(o.data.vertices) * 3, dtype=np.float32)
    o.data.vertices.foreach_get("normal", arr)
    return arr.reshape(-1, 3).astype(np.float64)


def _slots(o, mats):
    for k, m in enumerate(mats):
        assign(o, m, slot=k)


def build(p, rng):
    n = int(p.ribs)
    D = float(p.diameter)
    pitch = radians(p.pitch)
    opn = clamp(float(p.open))
    q = 1.0 - opn
    th_closed = radians(85.0)
    theta = pitch + (th_closed - pitch) * q
    sw = smoothstep(0.5, 0.92, q)              # how far the slack is wrapped round the pole
    furled = q > 0.75
    age = float(p.age)
    pole_r = 0.0215 + 0.0025 * D               # 58 mm pole under a 3 m canopy
    r_h = pole_r + 0.03                        # rib pivot radius
    r_run = pole_r + 0.034                     # stretcher pivot radius on the runner
    L = (D / 2 - r_h) / cos(pitch)             # rib length
    val_h = float(p.valance)
    z_h = max(float(p.height), L * sin(pitch) + val_h + 1.8)
    dphi = 2 * pi / n
    phi = [-pi / 2 + dphi / 2 + i * dphi for i in range(n)]      # a gore faces the front
    rj = rng.fork("ribs")
    th = [theta + radians(rj.jitter(0.0, 0.9)) * (0.25 + 0.75 * opn) for _ in range(n)]
    slack = 0.004 + 0.013 * p.sag
    c_off = 0.02                               # cloth line above the rib centre line (ribs are 28 mm deep)
    j_off = 0.027                              # stretcher joint below the rib centre line
    a_st = 0.5 * L                             # stretcher joint along the rib
    reach_o = r_h + a_st * cos(pitch) - r_run
    S = hypot(reach_o, 0.24 * L)               # stretcher length

    def z_runner(t):
        return z_h - a_st * sin(t) - j_off * cos(t) - sqrt(max(S * S - (r_h + a_st * cos(t) - r_run) ** 2, 1e-6))

    z_run = z_runner(theta)
    z_run_open = z_runner(pitch)
    z_run_closed = z_runner(th_closed)
    z_tube = 0.5                                      # top of the base socket tube
    bh = 0.24                                         # height of the stone block base
    tube_top = bh + 0.3 if p.base == "stone" else z_tube
    z_cl = max(clamp(z_run_closed - 0.18, 0.5, 1.0), tube_top + 0.27)       # cleat height
    z_j = max(min(0.78, z_cl - 0.16), tube_top + 0.1)                       # ferrule joint of the two pole halves
    s_tie = 0.6 * L
    vent = bool(p.vent)
    s0 = 0.2 * L if vent else 0.014
    s_cap = 0.31 * L
    cap_lift = 0.075

    # ------------------------------------------------------------------ materials
    striped = p.pattern in ("stripes", "awning")
    # grime gain down: the automatic cavity pass reads the sag of every gore as a dirt trap and blotches the coarse cloth mesh
    # (and every pleat of the furled bundle even more so)
    cg = 0.5 * (1.0 - 0.75 * sw)
    cw = 0.15                                  # low edge-wear gain: canvas frays little, and folds should not read as worn edges
    cA = mat("canvas", p.colour, stripe=float(p.stripe), tint2=p.colour2, grime=cg, wear=cw) if striped else mat("canvas", p.colour, grime=cg, wear=cw)
    cB = mat("canvas", p.colour2, grime=cg, wear=cw)
    cT = mat("canvas", p.trim, grime=cg, wear=cw)
    wood = mat("wood_oak", p.wood)
    iron = mat("iron_painted", p.paint, rough=0.75, wear=1.5, special=1.4)      # glossy thick paint, chipped and rust-bled
    wrought = mat("iron_wrought")
    brass = mat("brass", rough=0.85)
    rope_m = mat("rope", "#cbb48a")

    upper = []                                 # everything that leans with the pole
    cloths, woods, brasses, ropes = [], [], [], []

    def rib_pt(i, s, off=0.0):
        t = th[i % n]
        ph = phi[i % n]
        r = r_h + s * cos(t) + off * sin(t)
        return Vector((r * cos(ph), r * sin(ph), z_h - s * sin(t) + off * cos(t)))

    def rib_up(i):
        t = th[i % n]
        ph = phi[i % n]
        return Vector((sin(t) * cos(ph), sin(t) * sin(ph), cos(t)))

    def sst(x):
        return smoothstep(0.0, 1.0, x)

    def cloth(i, u, s, lift_z=0.0, lift_r=0.0):
        """Point of gore i (between rib i and i+1) at u across, slope distance s."""
        i0, i1 = i % n, (i + 1) % n
        phA = phi[i0]
        phB = phA + dphi
        tA, tB = th[i0], th[i1]
        rA = r_h + s * cos(tA) + c_off * sin(tA) + lift_r
        zA = z_h - s * sin(tA) + c_off * cos(tA) + lift_z
        rB = r_h + s * cos(tB) + c_off * sin(tB) + lift_r
        zB = z_h - s * sin(tB) + c_off * cos(tB) + lift_z
        ax, ay, bx, by = rA * cos(phA), rA * sin(phA), rB * cos(phB), rB * sin(phB)
        cx, cy, cz = ax + (bx - ax) * u, ay + (by - ay) * u, zA + (zB - zA) * u
        c = sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (zB - zA) ** 2)
        w = 2 * (r_h + s * cos(pitch)) * sin(pi / n) * (1 + slack)
        ex = sqrt(max(w * w - c * c, 0.0))
        f = 4 * u * (1 - u)
        hx, hy, hz = cx, cy, cz - 0.435 * ex * f           # slack hangs between the ribs
        if sw <= 0.0:
            return (hx, hy, hz)
        # furled: every gore is a flap laid round the bundle, all in the same direction, each over the next
        r_r = rA + (rB - rA) * u
        loose = 0.55 + 1.6 * (s / L) * (1.0 - 0.75 * exp(-((s - s_tie) / 0.16) ** 2))
        t_o, gap, kk = 0.006 * loose, 0.005 * loose, 0.013 * loose / dphi
        alpha = min(0.5 * ex / (r_r + 0.02), 1.1 * pi)
        phT = phA + 0.5 * dphi + alpha
        ph = phA + (phT - phA) * 2 * u if u < 0.5 else phT + (phB - phT) * (2 * u - 1)
        lo = t_o * sst((ph - phA) * r_r / 0.05) + kk * (ph - phA)
        li = max(0.0, (t_o + gap) * sst((ph - phB) * r_r / 0.05) + kk * (ph - phB))
        lvl = lo + (li - lo) * smoothstep(0.4, 0.6, u)
        r = r_r + lvl
        fx_, fy_, fz_ = r * cos(ph), r * sin(ph), cz - 0.05 * f * (s / L)
        return (hx + (fx_ - hx) * sw, hy + (fy_ - hy) * sw, hz + (fz_ - hz) * sw)

    # ------------------------------------------------------------------ canopy
    chord_rim = 2 * (D / 2) * sin(pi / n)
    k_sc = max(2, int(round(chord_rim / 0.36)))            # scallops per gore
    half = float(p.stripe) / 2
    panels = p.pattern == "panels"
    tassel_pts = []
    weathered = []                                         # (object, grime, fade) of the open canopy, painted after the lean

    def wid(s):
        return 2 * (r_h + s * cos(pitch)) * sin(pi / n)

    s_under = min(s_cap + 0.06, 0.5 * L)

    def guv(s, u):
        """Canvas UVs (m) of a gore point: weave square to the gore axis; `stripes` runs the bands round the canopy."""
        ac = (u - 0.5) * wid(s)
        return (L - s, ac) if p.pattern == "stripes" else (ac + half, s)

    def gore_mat(i):
        return cB if panels and i % 2 else cA

    def build_open():
        """One continuous cloth: a welded top skin (seam tape, rim tape = faces of the same surface), a plain under
        skin a few mm below, a rippling valance hung from the rim line and a soft vent cap."""
        rc = rng.fork("crease")
        tw = 0.017                                         # half width of the seam tape
        nh = int(clamp(round(wid(L) / 0.15), 6, 8))        # columns per half gore, crowding towards the ribs
        nu_g = 2 * nh
        ncol = n * nu_g
        kd = 0.6 + 0.8 * float(p.sag)
        # tension creases fanning out of each rib-tip pocket / out of the cap ties: (amplitude, waves, phase, reach)
        cr = [[(rc.u(0.012, 0.019) * kd * rc.pick((-1, 1)), rc.u(4.5, 7.0), rc.u(0, 6.28), rc.u(0.3, 0.45)) for _ in range(2)] for _ in range(n)]
        cr2 = [[(rc.u(0.006, 0.01) * kd, rc.u(5.0, 7.5), rc.u(0, 6.28), rc.u(0.2, 0.3)) for _ in range(2)] for _ in range(n)]
        bil = [(rc.u(0.004, 0.008) * kd, rc.u(0, 6.28), rc.u(1.0, 1.7)) for _ in range(n)]
        s_tie_c = s_cap - 0.02

        def detail(i, u, s, billow=True):
            i %= n
            W = wid(s)
            b = L - s
            z = 0.0
            for side in (0, 1):
                a = (u if side == 0 else 1.0 - u) * W
                edge = sst((a - tw) / 0.04)                # nothing on the tape: the cloth lies flat on the rib
                if edge <= 0.0:
                    continue
                amp, kf, ph, ln = cr[i][side]
                d = hypot(a, b)
                if d < ln:
                    z += amp * edge * sst((d - 0.02) / 0.06) * (1.0 - sst((d - 0.4 * ln) / (0.6 * ln))) * sin(kf * atan2(b, a) + ph)
                if vent:
                    b2 = s - s_tie_c
                    amp, kf, ph, ln = cr2[i][side]
                    d = hypot(a, b2)
                    if b2 > -0.02 and d < ln:
                        z += (amp * edge * sst((b2 + 0.02) / 0.05) * sst((d - 0.02) / 0.05) * (1.0 - sst((d - 0.4 * ln) / (0.6 * ln)))
                              * sin(kf * atan2(max(b2, 0.0), a) + ph))
            if not billow:
                return z
            amp, ph, fq = bil[i]
            return z + amp * sin(pi * u) ** 2 * sin(2 * pi * fq * s / L + ph)

        def top_pt(i, u, s):
            x, y, z = cloth(i, u, s)
            return (x, y, z + detail(i, u, s))

        def u_cols(s):
            W = wid(s)
            twe = min(tw, 0.3 * W)
            a = [0.0, twe] + [twe + (W / 2 - twe) * ((k - 1) / (nh - 1)) ** 1.6 for k in range(2, nh + 1)]
            us = [x / W for x in a]
            return us + [1.0 - x for x in reversed(us[:-1])]

        # rows: fine near the rim (rim tape, creases), coarse towards the hub; the first row of quads lies under the cap
        s_top = s_under if vent else s0
        span = L - s_top
        m = int(clamp(round(span / 0.105), 7, 11))
        bs = [0.0, 0.022] + [0.022 + (span - 0.022) * (k / (m - 1)) ** 1.6 for k in range(1, m)]
        S = ([s0] if vent else []) + [L - b for b in reversed(bs)]
        NR = len(S)
        Uc = [u_cols(s) for s in S]
        age_f = 0.35 + 0.9 * age

        def vid(c, k):
            return (c % ncol) * NR + k

        verts, quads, uvs, midx, g_top, d_top = [], [], [], [], [], []
        for i in range(n):
            for j in range(nu_g):
                for k in range(NR):
                    s, u = S[k], Uc[k][j]
                    verts.append(top_pt(i, u, s))
                    a = min(u, 1.0 - u) * wid(s)
                    g = 0.05 + 0.16 * clamp(1.0 - (L - s) / 0.12)                                   # dust settling at the rim
                    g += 0.24 * exp(-((a - 0.035) / 0.03) ** 2) * sst((s - s_top) / (0.6 * span))    # water running down beside the seams
                    if vent and s > s_cap - 0.03:                                                   # drip line under the cap edge
                        g += 0.2 * exp(-(s - s_cap + 0.03) / 0.14) * (0.45 + 0.55 * sin(u * 19.0 + i * 1.7) ** 2)
                    dz = detail(i, u, s, billow=False)
                    g += 0.3 * clamp(-dz / 0.012)                                                   # dirt lying in the creases
                    d_top.append(clamp(dz / 0.014))
                    g_top.append(g * age_f)
                c = i * nu_g + j
                for k in range(NR - 1):
                    quads.append((vid(c, k), vid(c, k + 1), vid(c + 1, k + 1), vid(c + 1, k)))
                    uvs += [guv(S[k], Uc[k][j]), guv(S[k + 1], Uc[k + 1][j]), guv(S[k + 1], Uc[k + 1][j + 1]), guv(S[k], Uc[k][j + 1])]
                    midx.append(1 if (j == 0 or j == nu_g - 1 or k == NR - 2) else (2 if panels and i % 2 else 0))
        top = _quad_mesh("canopy", verts, quads, uvs, midx)
        _slots(top, [cA, cT] + ([cB] if panels else []))
        g_top, d_top = np.array(g_top), np.array(d_top)
        weathered.append((top, lambda co, nr: g_top * (0.6 + 0.4 * nzb(co, nr)),      # ...and the crease ridges bleach first
                          lambda co, nr: np.clip(nr[:, 2] * 1.3 * (0.3 + 0.6 * nz2(co, nr)) * age + 0.15 * d_top, 0, 1)))
        P_top = verts

        # under skin: plain sagging cloth, coarse, clear of the creases above it
        nuc = 7
        S_u = S[::2] if (NR - 1) % 2 == 0 else S[::2] + [S[-1]]
        NRu = len(S_u)
        ncol_u = n * nuc

        def u_under(s):
            e2 = min((tw + 0.045) / wid(s), 0.2)
            return [0.0, e2] + [e2 + (1 - 2 * e2) * k / (nuc - 2) for k in range(1, nuc - 2)] + [1.0 - e2, 1.0]

        Uu = [u_under(s) for s in S_u]

        def vidu(c, k):
            return (c % ncol_u) * NRu + k

        verts, quads, uvs, midx = [], [], [], []
        for i in range(n):
            e_max = 1.15 * (max(abs(cr[i][0][0]), abs(cr[i][1][0]), cr2[i][0][0], cr2[i][1][0]) + bil[i][0])
            for j in range(nuc):
                for k in range(NRu):
                    s, u = S_u[k], Uu[k][j]
                    x, y, z = cloth(i, u, s)
                    sag_m = 0.5 * (cloth(i, 0.0, s)[2] + cloth(i, 1.0, s)[2]) - cloth(i, 0.5, s)[2]
                    edge = sst((min(u, 1.0 - u) * wid(s) - tw) / 0.04)
                    verts.append((x, y, z - 0.004 - edge * (e_max + 1.3 * max(sag_m, 0.0) / 25.0)))
                c = i * nuc + j
                for k in range(NRu - 1):
                    quads.append((vidu(c, k), vidu(c + 1, k), vidu(c + 1, k + 1), vidu(c, k + 1)))
                    uvs += [guv(S_u[k], Uu[k][j]), guv(S_u[k], Uu[k][j + 1]), guv(S_u[k + 1], Uu[k + 1][j + 1]), guv(S_u[k + 1], Uu[k + 1][j])]
                    midx.append(1 if panels and i % 2 else 0)
        under = _quad_mesh("canopy_in", verts, quads, uvs, midx)
        _slots(under, [cA] + ([cB] if panels else []))
        weathered.append((under, "soot", None))
        P_und = verts
        new = [top, under]

        # ---- valance: hangs from the rim line, flutes in and out, bound hem
        rv = rng.fork("valance")
        has_val = val_h > 0.02
        style = p.valance_style if has_val else "straight"
        vh = val_h if has_val else 0.03                    # no valance: just the turned-down bound edge
        hb = min(0.018, 0.4 * vh) if has_val else 0.02     # width of the binding
        m_sc = 12 if style != "straight" else 8            # segments per scallop
        Cv = k_sc * m_sc
        ncv = n * Cv
        f2 = int(round(2.3 * k_sc * n))
        f3 = max(2, int(round(0.6 * n)))
        ph1, ph2, ph3, ph4 = rv.u(-0.5, 0.5) - pi / 2, rv.u(0, 6.28), rv.u(0, 6.28), rv.u(0, 6.28)
        kv = clamp(vh / 0.2, 0.25, 1.3)
        A1, A2, A3 = rv.u(0.008, 0.012) * kv, rv.u(0.004, 0.007) * kv, rv.u(0.005, 0.009) * kv
        ue = [rv.u(-0.09, 0.09) for _ in range(n * k_sc)]
        verts, vin, quads, qin, uvs, uvin, midx, g_val = [], [], [], [], [], [], [], []
        W_rim = wid(L)
        for i in range(n):
            ut = np.array(Uc[-1])
            pt = np.array([P_top[vid(i * nu_g + j, NR - 1)] for j in range(nu_g + 1)])
            uu_ = np.array(Uu[-1])
            pu = np.array([P_und[vidu(i * nuc + j, NRu - 1)] for j in range(nuc + 1)])
            ng = Vector((cos(phi[i] + dphi / 2), sin(phi[i] + dphi / 2), 0.0))
            ug, dg = [], []
            for jj in range(Cv + 1):
                t = jj / Cv
                sc = min(int(t * k_sc + 1e-9), k_sc - 1)
                tl = t * k_sc - sc
                un = 1.0 + ue[i * k_sc + sc]
                if style == "scallop":                     # even steps round the arc, not across it: no kinked cusps
                    u = (sc + (1 - cos(pi * tl)) / 2) / k_sc
                    d = vh * (0.5 + 0.5 * sin(pi * tl) * un)
                elif style == "pointed":
                    u = t
                    d = vh * (0.42 + 0.58 * (1 - abs(2 * tl - 1)) * un)
                else:
                    u = t
                    d = vh
                d += 0.03 * vh * sin(2 * pi * f3 * (i + u) / n + ph4)        # the hem never hangs quite level
                ug.append(u)
                dg.append(d)
            for jj in range(Cv):
                u, d = ug[jj], dg[jj]
                U = i + u
                win = sst(u / 0.08) * sst((1 - u) / 0.08)                    # sewn tight at the rib corners
                w = win * (A1 * sin(2 * pi * k_sc * U + ph1) + A2 * sin(2 * pi * f2 * U / n + ph2) + A3 * sin(2 * pi * f3 * U / n + ph3)) + 0.003
                T = Vector([float(np.interp(u, ut, pt[:, a_])) for a_ in range(3)])
                Ti = Vector([float(np.interp(u, uu_, pu[:, a_])) for a_ in range(3)])
                verts += [tuple(T), tuple(T + ng * (w * (d - hb) / d) - Vector((0, 0, d - hb))), tuple(T + ng * (w + 0.002) - Vector((0, 0, d)))]
                fr_ = clamp((T.z - Ti.z) / max(d - hb, 1e-4))                 # keep the inner skin inside the fluted outer one
                vin.append(tuple(Vector((T.x, T.y, Ti.z)) + ng * (w * fr_ - 0.0025)))
                st = 0.75 + 0.25 * sin(U * 41.0 + ph2) * sin(U * 17.0 + ph3)
                g_val += [0.07 * age_f, 0.2 * age_f * st, 0.4 * age_f * st]  # dirt line along the hem
                c = i * Cv + jj
                if (style == "pointed" and jj % m_sc == m_sc // 2) or (style != "pointed" and jj % m_sc == 0 and jj > 0):
                    tassel_pts.append(verts[-1])
                for k in range(2):
                    v00, v01 = (c % ncv) * 3 + k, (c % ncv) * 3 + k + 1
                    v10, v11 = ((c + 1) % ncv) * 3 + k, ((c + 1) % ncv) * 3 + k + 1
                    quads.append((v00, v01, v11, v10))
                    qin.append((v00, v10, v11, v01))
                    dr = [(0.0, 0.0), (dg[jj] - hb, dg[jj + 1] - hb), (dg[jj], dg[jj + 1])]
                    ua, ub = (ug[jj] - 0.5) * W_rim + half, (ug[jj + 1] - 0.5) * W_rim + half
                    q = [(ua, L + dr[k][0]), (ua, L + dr[k + 1][0]), (ub, L + dr[k + 1][1]), (ub, L + dr[k][1])]
                    uvs += q
                    uvin += [q[0], q[3], q[2], q[1]]
                    midx.append(1 if (k == 1 or not has_val) else (2 if panels and i % 2 else 0))
        val = _quad_mesh("valance", verts, quads, uvs, midx)
        nrm = _vnormals(val)
        vi = []
        for c in range(ncv):                               # inner skin: hung from the under skin, pinched to the outer one at the hem
            vi.append(vin[c])
            for k, t_ in ((1, 0.0025), (2, 0.001)):
                vi.append(tuple(np.array(verts[c * 3 + k]) - nrm[c * 3 + k] * t_))
        val_in = _quad_mesh("valance_in", vi, qin, uvin, midx)
        vmats = [cB if p.pattern == "stripes" else cA, cT] + ([cB] if panels else [])
        g_val = np.array(g_val)
        for o in (val, val_in):
            _slots(o, vmats)
            weathered.append((o, lambda co, nr: g_val, age * 0.3))
        new += [val, val_in]

        # ---- rib-tip pockets: a reinforcing patch sewn over the end of every seam, lapping over the rim
        verts, quads, uvs = [], [], []
        for i in range(n):
            er_ = Vector((cos(phi[i]), sin(phi[i]), 0.0))
            b0 = len(verts)
            rows = [(L - 0.105, 0.021, 0.0), (L - 0.05, 0.028, 0.0), (L, 0.034, 0.0), (L, 0.034, 0.022), (L, 0.03, 0.042)]
            for (s, hw, drop) in rows:
                for a in (-hw, -hw / 3, hw / 3, hw):
                    u = a / wid(s)
                    pt_ = Vector(top_pt(i, u, s) if a >= 0 else top_pt(i - 1, 1.0 + u, s))
                    if drop > 0 or s >= L:
                        pt_ = pt_ + er_ * 0.003
                    verts.append(tuple(pt_ + Vector((0, 0, 0.0018 - drop))))
                    uvs.append((a, s + drop))
            for k in range(len(rows) - 1):
                for j in range(3):
                    quads.append((b0 + k * 4 + j, b0 + (k + 1) * 4 + j, b0 + (k + 1) * 4 + j + 1, b0 + k * 4 + j + 1))
        puv = []
        for q_ in quads:
            puv += [uvs[v] for v in q_]
        pockets = _quad_mesh("rib_pockets", verts, quads, puv)
        assign(pockets, cT)
        weathered.append((pockets, 0.18 * age_f, age * 0.3))
        new.append(pockets)

        # ---- vent cap: a small canopy of its own, drooping between its ties, bound edge
        if vent:
            s_a = 0.014
            ncg = 6
            vs = [0.0, 0.35, 0.68, 1.0 - 0.016 / (s_cap - s_a), 1.0]
            NRc = len(vs)
            ncc = n * ncg
            fl = [(rc.u(0.004, 0.012), rc.u(-0.005, 0.005)) for _ in range(n)]

            def u_cap(s):
                W = wid(s)
                twe = min(tw, 0.3 * W)
                us = [0.0, twe / W, (twe + (W / 2 - twe) * 0.45) / W, 0.5]
                return us + [1.0 - x for x in reversed(us[:-1])]

            def cap_pt(i, u, v):
                s = s_a + (s_cap - s_a) * v
                x, y, z = cloth(i, u, s, lift_z=cap_lift - 0.045 * v * v)
                dr_, asym = fl[i % n]
                return (x, y, z - dr_ * 4 * u * (1 - u) * v ** 3 + asym * sin(2 * pi * u) * sin(pi * u) * v * v)

            Ucp = [u_cap(s_a + (s_cap - s_a) * v) for v in vs]
            verts, quads, qin, uvs, uvin, midx, g_cap = [], [], [], [], [], [], []
            for i in range(n):
                for j in range(ncg):
                    for k in range(NRc):
                        verts.append(cap_pt(i, Ucp[k][j], vs[k]))
                        g_cap.append((0.05 + 0.16 * vs[k] ** 2) * age_f)
                    c = i * ncg + j
                    for k in range(NRc - 1):
                        v00, v01 = (c % ncc) * NRc + k, (c % ncc) * NRc + k + 1
                        v10, v11 = ((c + 1) % ncc) * NRc + k, ((c + 1) % ncc) * NRc + k + 1
                        quads.append((v00, v01, v11, v10))
                        qin.append((v00, v10, v11, v01))
                        sk = [s_a + (s_cap - s_a) * vs[k], s_a + (s_cap - s_a) * vs[k + 1]]
                        q = [guv(sk[0], Ucp[k][j]), guv(sk[1], Ucp[k + 1][j]), guv(sk[1], Ucp[k + 1][j + 1]), guv(sk[0], Ucp[k][j + 1])]
                        uvs += q
                        uvin += [q[0], q[3], q[2], q[1]]
                        midx.append(1 if (j == 0 or j == ncg - 1 or k == NRc - 2) else (2 if panels and i % 2 else 0))
            cap = _quad_mesh("vent_cap", verts, quads, uvs, midx)
            nrm = _vnormals(cap)
            vi = [tuple(np.array(verts[q_]) - nrm[q_] * (0.001 if q_ % NRc == NRc - 1 else 0.0025)) for q_ in range(len(verts))]
            cap_in = _quad_mesh("vent_cap_in", vi, qin, uvin, midx)
            g_cap = np.array(g_cap)
            cmats = [cB if panels else cA, cT] + ([cA] if panels else [])       # the cap gores alternate against the canopy
            for o in (cap, cap_in):
                _slots(o, cmats)
            weathered.append((cap, lambda co, nr: g_cap, lambda co, nr: (0.4 + 0.6 * nz2(co, nr)) * age))
            weathered.append((cap_in, lambda co, nr: g_cap, None))
            new += [cap, cap_in]
        return new

    def build_folded():
        """Half-furled / furled: every gore is its own flap with real thickness, laid round the pole (see cloth())."""
        m_sc = max(4, int(round(120 / (n * k_sc))))
        if sw > 0.3:
            m_sc = max(4, min(int(ceil(m_sc * 1.4)), 150 // (n * k_sc)))      # pleats need columns, but stay in budget
        nu = k_sc * m_sc
        nv = int(clamp(round(L / 0.19), 7, 8 if sw > 0.3 else 9))

        def s_gore(v):
            if not vent:
                return s0 + (L + 0.003 - s0) * v
            v1 = 1.0 / nv
            return np.where(v <= v1, s0 + (s_under - s0) * v / v1, s_under + (L + 0.003 - s_under) * (v - v1) / (1.0 - v1))

        def gore_uv(sa, sb, smap=None):
            def fn(U, V):
                s = smap(V) if smap else sa + (sb - sa) * V
                ac = (U - 0.5) * wid(s)
                if p.pattern == "stripes":
                    return (L - s), ac
                return ac + half, s
            return fn

        def sheet(name, fn, nu_, nv_, uvfn, m_, t=0.0025):
            g = surface(name, fn, nu_, nv_, thickness=t, uv_len=(1.0, 1.0))
            _remap_uv(g, uvfn)
            assign(g, m_)
            cloths.append(g)

        for i in range(n):
            sheet(f"gore{i}", lambda u, v, i=i: cloth(i, u, float(s_gore(v))), nu, nv, gore_uv(s0, L, s_gore), gore_mat(i))
            tape = _bar(f"seam{i}", rib_pt(i, s0, c_off + 0.0022), rib_pt(i, L, c_off + 0.0022), _rect(0.03, 0.003), rib_up(i))
            assign(tape, cT)
            cloths.append(tape)
        if vent:
            nuc = max(6, nu // 2)
            for i in range(n):
                def capf(u, v, i=i):
                    s = 0.014 + (s_cap - 0.014) * v
                    return cloth(i, u, s, lift_z=cap_lift - 0.045 * v * v, lift_r=0.012 * sw)
                sheet(f"cap{i}", capf, nuc, 3, gore_uv(0.014, s_cap), gore_mat(i + 1))
                edge = [Vector(capf(j / nuc, 1.0)) + Vector((0, 0, -0.0125)) for j in range(nuc + 1)]
                hem = sweep(f"cap_hem{i}", [tuple(e) for e in edge], _rect(0.006, 0.022), smooth_path=False, up=(0, 0, 1))
                assign(hem, cT)
                cloths.append(hem)
        if val_h > 0.02:
            rv = rng.fork("valance")
            for i in range(n):
                ph0 = rv.u(0, 6.28)
                amp = rv.u(0.015, 0.026)

                def edge_depth(u):
                    fr_ = (u * k_sc) % 1.0
                    if u >= 1.0 - 1e-9:
                        fr_ = 1.0
                    if p.valance_style == "scallop":
                        return val_h * (0.5 + 0.5 * sqrt(max(1 - (2 * fr_ - 1) ** 2, 0.0)))
                    if p.valance_style == "pointed":
                        return val_h * (0.42 + 0.58 * (1 - abs(2 * fr_ - 1)))
                    return val_h

                def valf(u, v, i=i, ph0=ph0, amp=amp):
                    x, y, z = cloth(i, u, L)
                    r = hypot(x, y)
                    wob = (amp * sin(2 * pi * k_sc * u + ph0) * sin(pi / 2 * min(1.0, u * 6)) * sin(pi / 2 * min(1.0, (1 - u) * 6)) + 0.004) * v
                    k = (r + wob) / r
                    return (x * k, y * k, z + 0.001 - v * (edge_depth(u) + 0.001))

                sheet(f"valance{i}", valf, nu, 3, lambda U, V: ((U - 0.5) * wid(L) + half, L + V * val_h),
                      cB if p.pattern == "stripes" else gore_mat(i))
                path = [Vector(valf(j / nu, 1.0)) + Vector((0, 0, 0.008)) for j in range(nu + 1)]
                hem = sweep(f"hem{i}", [tuple(e) for e in path], _rect(0.007, 0.02), smooth_path=False, up=(0, 0, 1))
                assign(hem, cT)
                cloths.append(hem)
                us = [(j + 0.5) / k_sc for j in range(k_sc)] if p.valance_style == "pointed" else [j / k_sc for j in range(1, k_sc)]
                tassel_pts.extend(valf(u_, 1.0) for u_ in us)
        # folded cloth: even-thickness solidify shoots the vertices on a hairpin fold metres away -> plain normal offset
        for o in cloths:
            for m in o.modifiers:
                if m.type == "SOLIDIFY":
                    m.use_even_offset = False
                    m.use_rim = False              # no knife-edge rim faces: the exporter would paint them as worn-white edges

    nzb = noise_mask(0.7, seed=p.seed + 13, lo=0.3, hi=0.8)
    nz2 = noise_mask(0.9, seed=p.seed + 12, lo=0.2, hi=0.8)
    if sw > 0.0:
        build_folded()
    else:
        upper += build_open()
    upper += cloths

    # ------------------------------------------------------------------ ribs, stretchers, hubs
    rib_prof = rounded_rect(0.02, 0.028, 0.004, 1)
    str_prof = rounded_rect(0.017, 0.024, 0.003, 1)
    for i in range(n):
        up = rib_up(i)
        rib = _bar(f"rib{i}", rib_pt(i, 0.0), rib_pt(i, L - 0.012), rib_prof, up)
        woods.append(rib)
        ph = phi[i]
        q0 = Vector((r_run * cos(ph), r_run * sin(ph), z_run))
        q1 = rib_pt(i, a_st, -j_off)
        tang = Vector((-sin(ph), cos(ph), 0.0))
        sup = (q1 - q0).cross(tang).normalized()
        if sup.z < 0 and not furled:
            sup = -sup
        woods.append(_bar(f"stretcher{i}", q0, q1, str_prof, sup))
        # brass saddle where the stretcher meets the rib, brass shoe at the rib tip
        brasses.append(_bar(f"saddle{i}", rib_pt(i, a_st - 0.024, -0.007), rib_pt(i, a_st + 0.024, -0.007), _rect(0.026, 0.046), up))
        brasses.append(_bar(f"shoe{i}", rib_pt(i, L - 0.04), rib_pt(i, L - 0.005), _rect(0.025, 0.033), up))
        brasses.append(_bar(f"fork{i}", q0 - Vector((0, 0, 0.004)), q0 + (q1 - q0).normalized() * 0.04, _rect(0.022, 0.03), sup))
    # crown hub: turned disc, ribs strung on a wire ring
    woods.append(lathe("crown", [(pole_r + 0.001, z_h - 0.075), (pole_r + 0.009, z_h - 0.07), (pole_r + 0.009, z_h - 0.03), (pole_r + 0.014, z_h - 0.02),
                                 (r_h + 0.007, z_h - 0.014), (r_h + 0.009, z_h - 0.006), (r_h + 0.009, z_h + 0.008), (r_h + 0.004, z_h + 0.014),
                                 (pole_r + 0.012, z_h + 0.018), (pole_r + 0.001, z_h + 0.03)], segments=14))
    brasses.append(torus("crown_wire", r_h + 0.006, 0.0032, loc=(0, 0, z_h), major_seg=14, minor_seg=4))
    brasses.append(ring_band("crown_band", pole_r + 0.0095, 0.022, 0.0015, loc=(0, 0, z_h - 0.05), segments=14))
    # runner
    woods.append(lathe("runner", [(pole_r + 0.0015, z_run - 0.085), (pole_r + 0.01, z_run - 0.08), (pole_r + 0.01, z_run - 0.03), (pole_r + 0.016, z_run - 0.02),
                                  (r_run + 0.006, z_run - 0.014), (r_run + 0.008, z_run - 0.006), (r_run + 0.008, z_run + 0.008), (r_run + 0.003, z_run + 0.014),
                                  (pole_r + 0.013, z_run + 0.018), (pole_r + 0.0015, z_run + 0.03)], segments=14))
    brasses.append(torus("runner_wire", r_run + 0.004, 0.0032, loc=(0, 0, z_run), major_seg=14, minor_seg=4))
    brasses.append(ring_band("runner_band", pole_r + 0.0105, 0.024, 0.0015, loc=(0, 0, z_run - 0.055), segments=14))
    # peg through the pole that holds the open runner, with a pull ring
    z_peg = z_run_open - 0.1
    a_peg = phi[0] + dphi * 0.5 + pi / 2
    brasses.append(cylinder("peg", 0.0045, 2 * pole_r + 0.05, loc=(0.008 * cos(a_peg), 0.008 * sin(a_peg), z_peg), rot=(0, 90, degrees(a_peg)), verts=8))
    brasses.append(torus("peg_ring", 0.014, 0.0022, loc=((pole_r + 0.047) * cos(a_peg), (pole_r + 0.047) * sin(a_peg), z_peg - 0.01),
                         rot=(90, 0, degrees(a_peg) + 90), major_seg=12, minor_seg=5))

    # ------------------------------------------------------------------ pole, ferrule, finial
    z_sock = 0.06
    lift0 = cap_lift if vent else 0.0
    z_cl_top = z_h + c_off * cos(theta) + lift0 + 0.004           # cloth apex
    lo = cylinder("pole_lower", pole_r, z_j - z_sock, loc=(0, 0, (z_j + z_sock) / 2), verts=18)
    hi = cylinder("pole_upper", pole_r * 0.98, z_cl_top - z_j, loc=(0, 0, (z_cl_top + z_j) / 2), verts=18, radius2=pole_r * 0.93)
    for o in (lo, hi):
        smooth(o, 50)
        uv_grain(o, axis="Z")
        displace_noise(o, 0.0006, 0.3, seed=p.seed)
    woods += [lo, hi]
    fr = pole_r + 0.0005
    brasses.append(lathe("ferrule", [(fr, z_j - 0.075), (fr + 0.005, z_j - 0.071), (fr + 0.005, z_j - 0.056), (fr + 0.0032, z_j - 0.053), (fr + 0.0032, z_j - 0.004),
                                     (fr + 0.0045, z_j - 0.002), (fr + 0.0045, z_j + 0.002), (fr + 0.0032, z_j + 0.004), (fr + 0.0032, z_j + 0.053),
                                     (fr + 0.005, z_j + 0.056), (fr + 0.005, z_j + 0.071), (fr, z_j + 0.075)], segments=14))
    brasses.append(bolt("ferrule_pin", 0.006, 0.004, loc=(0, -(fr + 0.003), z_j + 0.03), rot=(90, 0, 0), kind="dome"))
    brasses.append(bolt("ferrule_pin2", 0.006, 0.004, loc=(0, -(fr + 0.003), z_j - 0.03), rot=(90, 0, 0), kind="dome"))
    zc = z_cl_top
    rp = pole_r * 0.93
    brasses.append(lathe("crown_plate", [(rp * 0.5, zc - 0.002), (r_h + 0.036, zc - 0.002), (r_h + 0.04, zc + 0.003), (r_h + 0.034, zc + 0.008),
                                         (rp + 0.016, zc + 0.012), (rp + 0.012, zc + 0.02), (rp * 0.5, zc + 0.02)], segments=18))
    zf = zc + 0.02
    woods.append(lathe("finial", [(rp + 0.006, zf), (rp + 0.016, zf + 0.006), (rp + 0.018, zf + 0.014), (rp + 0.008, zf + 0.024), (rp - 0.006, zf + 0.034),
                                  (rp - 0.008, zf + 0.044), (rp + 0.004, zf + 0.056), (rp + 0.014, zf + 0.074), (rp + 0.016, zf + 0.09),
                                  (rp + 0.01, zf + 0.108), (rp - 0.004, zf + 0.124), (0.008, zf + 0.136), (0.01, zf + 0.146), (0.0, zf + 0.152)], segments=16))

    # ------------------------------------------------------------------ hoist cord, pulley, cleat
    i_r = n // 4
    ph_r = phi[i_r] + dphi / 2
    er = Vector((cos(ph_r), sin(ph_r), 0.0))
    tr = Vector((-sin(ph_r), cos(ph_r), 0.0))
    z_p = z_h - 0.115
    rp_c = pole_r + 0.03

    def rz(r, z, t=0.0):
        v = er * r + tr * t
        return (v.x, v.y, z)

    brasses.append(cylinder("pulley", 0.015, 0.009, loc=rz(rp_c, z_p), rot=(90, 0, degrees(ph_r) + 180), verts=14))
    brasses.append(_bar("pulley_strap", rz(pole_r - 0.002, z_p + 0.004), rz(rp_c + 0.004, z_p), _rect(0.016, 0.012), (0, 0, 1)))
    cord = [rz(pole_r + 0.014, z_run + 0.03), rz(pole_r + 0.013, z_p - 0.02)]
    for k in range(7):
        t = pi - pi * k / 6
        cord.append(rz(rp_c + 0.0175 * cos(t), z_p + 0.0175 * sin(t)))
    zc_top = z_cl + 0.055
    span = z_p - zc_top
    bow = 0.02 + 0.05 * q
    for k in range(1, 9):
        t = k / 8
        cord.append(rz(rp_c + 0.0175 + (pole_r + 0.022 - rp_c - 0.0175) * t ** 3 + bow * sin(pi * t), z_p - span * t, 0.012 * sin(pi * t)))
    ropes.append(sweep("hoist_cord", cord, 0.0042, sides=6, smooth_path=False))
    brasses.append(torus("runner_eye", 0.009, 0.0025, loc=rz(pole_r + 0.014, z_run + 0.024), rot=(90, 0, degrees(ph_r) + 90), major_seg=10, minor_seg=5))
    # cleat on the lower pole half, cord made off in figure-of-eights, tail hanging
    brasses.append(_bar("cleat_stem", rz(pole_r - 0.003, z_cl), rz(pole_r + 0.024, z_cl), rounded_rect(0.014, 0.03, 0.003, 1), (0, 0, 1)))
    brasses.append(sweep("cleat_horns", [rz(pole_r + 0.026, z_cl - 0.06), rz(pole_r + 0.022, z_cl - 0.03), rz(pole_r + 0.022, z_cl + 0.03), rz(pole_r + 0.026, z_cl + 0.06)],
                         0.0075, sides=8, scale_fn=lambda t: 0.55 + 0.45 * sin(pi * t)))
    brasses.append(_bar("cleat_plate", rz(pole_r - 0.001, z_cl - 0.03), rz(pole_r - 0.001, z_cl + 0.03), _rect(0.022, 0.006), tuple(er)))
    for k in range(2):
        pts = []
        for j in range(14):
            t = 2 * pi * j / 14
            pts.append(rz(pole_r + 0.011 + 0.0045 * cos(t) + 0.004 * k, z_cl + (0.036 + 0.008 * k) * sin(t), (0.013 + 0.003 * k) * sin(2 * t)))
        ropes.append(sweep(f"cleat_turns{k}", pts, 0.004, closed=True, sides=4, smooth_path=False))
    tl = 0.3 + 0.25 * q
    tail = [rz(pole_r + 0.014, z_cl - 0.03, 0.006), rz(pole_r + 0.03, z_cl - 0.03 - tl * 0.3, 0.012), rz(pole_r + 0.022, z_cl - 0.03 - tl * 0.7, 0.02),
            rz(pole_r + 0.018, z_cl - 0.03 - tl, 0.016)]
    ropes.append(sweep("cord_tail", resample_path(tail, n=10), 0.0042, sides=6))
    ropes.append(sphere("cord_knot", 0.009, loc=rz(pole_r + 0.018, z_cl - 0.035 - tl, 0.016), segments=8, rings=6, scale=(1, 1, 1.3)))

    # ------------------------------------------------------------------ tie cord
    rt = rng.fork("tie")
    if furled:
        bins = 44
        env = [0.0] * bins
        zs = []
        for i in range(n):
            for j in range(41):
                x, y, z = cloth(i, j / 40, s_tie)
                b = int((atan2(y, x) % (2 * pi)) / (2 * pi) * bins) % bins
                env[b] = max(env[b], hypot(x, y))
                zs.append(z)
        mean_r = sum(e for e in env if e > 0) / max(1, sum(1 for e in env if e > 0))
        env = [e if e > 0 else mean_r for e in env]
        env = [max(env[b - 1], env[b], env[(b + 1) % bins]) for b in range(bins)]
        env = [(env[b - 1] + 2 * env[b] + env[(b + 1) % bins]) / 4 for b in range(bins)]
        zt = sum(zs) / len(zs)
        for k in range(2):
            pts = [((env[b] + 0.006) * cos(2 * pi * (b + 0.5) / bins), (env[b] + 0.006) * sin(2 * pi * (b + 0.5) / bins),
                    zt + 0.011 * k + 0.006 * sin(2 * pi * b / bins + k * 2.0)) for b in range(bins)]
            ropes.append(sweep(f"tie_wrap{k}", pts, 0.0048, closed=True, sides=6, smooth_path=False))
        bf = int(0.75 * bins)                                  # bow at the front
        rf = env[bf] + 0.012
        af = 2 * pi * (bf + 0.5) / bins
        kx, ky = rf * cos(af), rf * sin(af)
        ropes.append(sphere("tie_knot", 0.013, loc=(kx, ky, zt + 0.006), segments=8, rings=6, scale=(1.2, 0.8, 1.0)))
        for k, sx in enumerate((-1, 1)):
            ln = rt.u(0.2, 0.34)
            pts = [(kx, ky, zt + 0.004), (kx + sx * 0.035, ky - 0.012, zt - 0.03), (kx + sx * 0.045, ky - 0.008, zt - ln * 0.6), (kx + sx * 0.04, ky - 0.004, zt - ln)]
            ropes.append(sweep(f"tie_tail{k}", resample_path(pts, n=8), 0.0045, sides=6))
            lp = [(kx, ky, zt + 0.006), (kx + sx * 0.05, ky - 0.014, zt + 0.03), (kx + sx * 0.075, ky - 0.01, zt + 0.004), (kx + sx * 0.04, ky - 0.012, zt - 0.014), (kx, ky - 0.004, zt + 0.002)]
            ropes.append(sweep(f"tie_loop{k}", resample_path(lp, n=10), 0.0045, sides=6))
    else:
        # the furling cord is sewn to one seam at the rim and hangs loose while the parasol is up
        it = n - 1
        tip = rib_pt(it, L - 0.03, -0.02)
        for k in range(2):
            ln = rt.u(0.3, 0.46)
            dx, dy = rt.u(-0.03, 0.03), rt.u(-0.03, 0.03)
            pts = [tuple(tip), (tip.x + dx * 0.4, tip.y + dy * 0.4, tip.z - ln * 0.35), (tip.x + dx, tip.y + dy, tip.z - ln * 0.75), (tip.x + dx * 0.8, tip.y + dy * 0.8, tip.z - ln)]
            ropes.append(sweep(f"tie_cord{k}", resample_path(pts, n=8), 0.0045, sides=6))
            ropes.append(sphere(f"tie_end{k}", 0.009, loc=(tip.x + dx * 0.8, tip.y + dy * 0.8, tip.z - ln - 0.004), segments=8, rings=6, scale=(1, 1, 1.4)))

    # tassels on the valance
    if p.tassels and tassel_pts:
        tb = lathe("tassel", [(0.0, -0.078), (0.0095, -0.075), (0.0062, -0.043), (0.0085, -0.03), (0.004, -0.02), (0.0016, 0.004)], segments=4)
        assign(tb, cT)
        rtas = rng.fork("tassels")
        tm = [trs((x, y, z + 0.004), (rtas.u(-7, 7), rtas.u(-7, 7), rtas.u(0, 72)), rtas.u(0.9, 1.1)) for (x, y, z) in tassel_pts]
        cloths_t = instances_to_mesh("tassels", tb, tm, keep_base=False)
        upper.append(cloths_t)
        cloths.append(cloths_t)
    for o in woods:
        assign(o, wood)
    for o in brasses:
        assign(o, brass)
    for o in ropes:
        assign(o, rope_m)
    upper += woods + brasses + ropes

    # ------------------------------------------------------------------ lean: the pole never stands quite true in its socket
    rl = rng.fork("lean")
    la = radians(p.lean) * rl.u(0.6, 1.0)
    ld = rl.u(0, 2 * pi)
    piv = Vector((0, 0, 0.12))
    M = Matrix.Translation(piv) @ Matrix.Rotation(la, 4, Vector((cos(ld), sin(ld), 0))) @ Matrix.Translation(-piv)
    for o in upper:
        apply_transform(o)
        transform_verts(o, lambda co: M @ co)

    # ------------------------------------------------------------------ lanterns: under the canopy on the stretchers, and on hooks at the rib tips
    lant_xy = []
    lit = opn > 0.8
    n_in = int(p.lanterns) if lit else 0
    n_rim = int(p.rim_lanterns) if lit else 0
    hooks = []
    if n_in + n_rim > 0:
        glass = mat("glass_lantern", glow=1.5)
        flame_m = mat("emissive", "#ffb45e", glow=9)

    def lantern(k, hp, ring_deg, drop, cd):
        x, y, z = hp.x, hp.y, hp.z
        lo_ = [torus(f"lant_ring{k}", 0.017, 0.002, loc=(x, y, z - 0.004), rot=(0, 90, ring_deg), major_seg=10, minor_seg=4)]
        z_cap = z - 0.021 - drop - 0.046
        lo_.append(cylinder(f"lant_link{k}", 0.002, drop, loc=(x, y, z - 0.021 - drop / 2), verts=5))
        lo_.append(sweep(f"lant_bail{k}", arc_path(0.046, 0, 180, n=9, plane="XZ", center=(x, y, z_cap)), 0.0019, sides=4))
        zb = z_cap - 0.168
        zg = zb + 0.05
        lo_.append(lathe(f"lant_font{k}", [(0.0, zb), (0.04, zb), (0.045, zb + 0.005), (0.046, zb + 0.022), (0.042, zb + 0.032), (0.03, zb + 0.04),
                                           (0.029, zb + 0.05), (0.012, zb + 0.052), (0.01, zb + 0.064), (0.0, zb + 0.066)], segments=12, loc=(x, y, 0)))
        lo_.append(lathe(f"lant_cap{k}", [(0.029, zg + 0.098), (0.033, zg + 0.102), (0.03, zg + 0.112), (0.017, zg + 0.124), (0.02, zg + 0.13),
                                          (0.012, zg + 0.138), (0.0, zg + 0.14)], segments=12, loc=(x, y, 0)))
        for sx in (-1, 1):
            lo_.append(cylinder(f"lant_guard{k}", 0.0019, z_cap - zb - 0.02, loc=(x + sx * 0.046, y, (z_cap + zb + 0.02) / 2), verts=4))
        lo_.append(cylinder(f"lant_bar{k}", 0.0019, 0.092, loc=(x, y, zg + 0.108), rot=(0, 90, 0), verts=4))
        for o in lo_:
            assign(o, brass)
        gl = lathe(f"lant_glass{k}", [(0.028, zg), (0.036, zg + 0.018), (0.041, zg + 0.045), (0.037, zg + 0.075), (0.028, zg + 0.1)], segments=12, loc=(x, y, 0))
        assign(gl, glass)
        fl = sphere(f"lant_flame{k}", 0.008, loc=(x, y, zg + 0.034), segments=6, rings=5, scale=(1, 1, 2.0))
        assign(fl, flame_m)
        part(lo_ + [gl, fl], f"lantern{k}", pivot=(x, y, z), motion="swing", amp=2.5, speed=0.35)
        light(f"lantern{k}", (x, y, zg + 0.036), color="#ffbf6a", intensity=cd, range=6.5, flicker=0.2, shadow_near=0.085)

    n_glow = min(n_in, 6 - n_in - n_rim)        # light budget: 6 sockets
    glow_h, glow_k = 0.15, 0.03
    used = []
    for k in range(n_in):
        i = (int(round(k * n / n_in)) + 1) % n
        while i in used:
            i = (i + 1) % n
        used.append(i)
        ph = phi[i]
        q0 = Vector((r_run * cos(ph), r_run * sin(ph), z_run))
        q1 = rib_pt(i, a_st, -j_off)
        t = 0.6 + 0.08 * ((k % 2) - 0.5)
        hp = M @ (q0 + (q1 - q0) * t)
        lantern(k, hp, degrees(ph), 0.07 + 0.05 * ((k * 7 + p.seed) % 3) / 2, float(p.intensity))
        lant_xy.append((hp.x, hp.y))
        if k < n_glow:
            # the lantern shining through the canvas: a soft warm pool on the top of the cloth right above it
            gk_ = 1.0 + 0.28 / hypot(hp.x, hp.y)
            zc_ = z_h + c_off - (gk_ * hypot(hp.x, hp.y) - r_h) * np.tan(theta)
            light(f"canvas_glow{k}", (hp.x * gk_, hp.y * gk_, zc_ + glow_h), color="#ffb062", intensity=glow_k * float(p.intensity),
                  range=1.6, flicker=0.2, shadow=False)
    used_r = []
    for k in range(n_rim):
        i = (int(round((k + 0.5) * n / n_rim)) + n - 2) % n
        while i in used_r:
            i = (i + 1) % n
        used_r.append(i)
        ph = phi[i]
        e = Vector((cos(ph), sin(ph), 0.0))
        tip = M @ rib_pt(i, L - 0.02, -0.02)
        reach = 0.15
        pts = [tip - e * 0.07, tip + e * 0.03, tip + e * (reach * 0.6) + Vector((0, 0, 0.022)), tip + e * (reach * 0.96) + Vector((0, 0, 0.01)),
               tip + e * (reach + 0.012) + Vector((0, 0, -0.012)), tip + e * (reach + 0.002) + Vector((0, 0, -0.03)), tip + e * (reach - 0.014) + Vector((0, 0, -0.022))]
        hooks.append(sweep(f"rim_hook{k}", resample_path([tuple(v) for v in pts], n=14), 0.0036, sides=6, scale_fn=lambda t: 1.0 - 0.35 * t))
        hp = tip + e * reach + Vector((0, 0, -0.008))
        lantern(10 + k, hp, degrees(ph), 0.03, max(0.8, float(p.intensity) * 0.2))
    for o in hooks:
        assign(o, wrought)

    # ------------------------------------------------------------------ base
    rb_ = rng.fork("base")
    r_t = pole_r + 0.011
    Rb = 0.22 + 0.04 * D                       # 0.68 m cast-iron base under a 3 m canopy
    irons, stones, forged = [], [], []
    iron_b = wrought if p.base == "stone" else iron

    def tube(zb0, zt0):
        return [(r_t + 0.012, zb0), (r_t + 0.012, zb0 + 0.02), (r_t + 0.004, zb0 + 0.032), (r_t, zb0 + 0.05), (r_t, zt0 - 0.11), (r_t + 0.006, zt0 - 0.105),
                (r_t + 0.008, zt0 - 0.09), (r_t + 0.006, zt0 - 0.075), (r_t, zt0 - 0.07), (r_t, zt0 - 0.03), (r_t + 0.007, zt0 - 0.025),
                (r_t + 0.007, zt0 - 0.005), (r_t + 0.003, zt0), (pole_r + 0.0015, zt0), (pole_r + 0.0015, zt0 - 0.04)]

    def clamp_screws(zs_, az0):
        for k, zc_ in enumerate(zs_):
            az = az0 + k * radians(90)
            e = Vector((cos(az), sin(az), 0))
            irons.append(cylinder(f"screw_boss{k}", 0.016, 0.016, loc=tuple(e * (r_t + 0.006) + Vector((0, 0, zc_))), rot=(0, 90, degrees(az)), verts=10))
            forged.append(cylinder(f"screw{k}", 0.0065, 0.055, loc=tuple(e * (r_t + 0.038) + Vector((0, 0, zc_))), rot=(0, 90, degrees(az)), verts=8))
            tilt = rb_.u(-50, 50)
            forged.append(cylinder(f"screw_bar{k}", 0.005, 0.09, loc=tuple(e * (r_t + 0.062) + Vector((0, 0, zc_))), rot=(tilt, 0, degrees(az)), verts=6))
            for sgn in (-1, 1):
                off = Vector((0, 0, sgn * 0.045))
                off.rotate(Matrix.Rotation(radians(tilt), 3, "X"))
                off.rotate(Matrix.Rotation(az, 3, "Z"))
                forged.append(sphere(f"screw_ball{k}", 0.0085, loc=tuple(e * (r_t + 0.062) + Vector((0, 0, zc_)) + off), segments=6, rings=4))

    if p.base == "round":
        # heavy cast disc: foot bead, 7 cm rim with a moulded shoulder, gadrooned dome, socket tube with gussets
        rim_h = 0.07
        prof = [(Rb - 0.004, 0.0), (Rb, 0.004), (Rb, 0.017), (Rb - 0.005, 0.023), (Rb - 0.007, 0.05), (Rb - 0.003, 0.055), (Rb - 0.003, 0.064),
                (Rb - 0.01, rim_h), (Rb - 0.05, rim_h + 0.004), (Rb - 0.058, rim_h + 0.012)]
        z_b, r_o, r_in, nd = rim_h + 0.012, Rb - 0.058, r_t + 0.05, 5
        for k in range(1, nd + 1):
            t = k / nd
            prof.append((r_o + (r_in - r_o) * t, z_b + 0.062 * sin(t * pi / 2) ** 1.2))
        z_d = z_b + 0.062
        prof += [(r_t + 0.03, z_d + 0.006), (r_t + 0.026, z_d + 0.02), (r_t + 0.014, z_d + 0.028)]
        prof += tube(z_d + 0.028, z_tube)[2:]
        dome = lathe("base_dome", prof, segments=32, smooth_angle=40)
        ng = 8

        def gadroon(co):
            r = hypot(co.x, co.y)
            m = smoothstep(r_in, r_in + 0.04, r) * (1 - smoothstep(r_o - 0.03, r_o, r))
            if m <= 0 or co.z < z_b - 0.004:
                return co
            a = atan2(co.y, co.x)
            return Vector((co.x, co.y, co.z + 0.011 * m * (abs(cos(a * ng / 2)) ** 0.6 - 0.5)))

        transform_verts(dome, gadroon)
        irons.append(dome)
        bt = bolt("base_bolt", 0.012, 0.009, kind="hex")
        irons.append(instances_to_mesh("base_bolts", bt, [trs(((Rb - 0.03) * cos(radians(45 + 90 * k)), (Rb - 0.03) * sin(radians(45 + 90 * k)), rim_h + 0.002),
                                                              (0, 0, rb_.u(0, 60))) for k in range(4)], keep_base=False))
        gus = extrude_profile("gusset", [(0.0, 0.0), (0.085, 0.0), (0.055, 0.022), (0.012, 0.11), (0.0, 0.12)], 0.013)
        gm = []
        for k in range(4):
            a = radians(90 * k)
            base_pt = Vector((cos(a), sin(a), 0)) * (r_t - 0.002) + Vector((0, 0, z_d + 0.012))
            gm.append(trs(tuple(base_pt - Vector((sin(a), -cos(a), 0)) * 0.0065), (90, 0, degrees(a))))
        irons.append(instances_to_mesh("gussets", gus, gm, keep_base=False))
        clamp_screws((z_tube - 0.04, z_tube - 0.17), radians(-65))
    elif p.base == "cross":
        irons.append(lathe("base_tube", [(r_t + 0.03, 0.0), (r_t + 0.032, 0.012), (r_t + 0.014, 0.024)] + tube(0.024, z_tube)[2:], segments=24))
        leg_prof = rounded_rect(0.04, 0.058, 0.009, 2)
        pad = lathe("pad", [(0.058, 0.0), (0.062, 0.006), (0.06, 0.02), (0.044, 0.03), (0.024, 0.034), (0.0, 0.034)], segments=12)
        pm = []
        for k in range(4):
            a = radians(45 + 90 * k)
            e = Vector((cos(a), sin(a), 0))
            pts = bezier_path(tuple(e * (r_t - 0.004) + Vector((0, 0, 0.27))), tuple(e * (Rb * 0.5) + Vector((0, 0, 0.31))),
                              tuple(e * (Rb * 0.62) + Vector((0, 0, 0.055))), tuple(e * (Rb - 0.01) + Vector((0, 0, 0.055))), n=12)
            irons.append(sweep(f"leg{k}", pts, leg_prof, scale_fn=lambda t: 1.0 - 0.3 * t, up=(0, 0, 1)))
            # lower strut makes each leg a rigid triangle
            irons.append(_bar(f"strut{k}", e * (r_t - 0.004) + Vector((0, 0, 0.08)), e * (Rb * 0.68) + Vector((0, 0, 0.092)), rounded_rect(0.024, 0.034, 0.005, 1), (0, 0, 1)))
            pm.append(trs(tuple(e * (Rb - 0.012)), (0, 0, rb_.u(0, 90))))
        irons.append(instances_to_mesh("pads", pad, pm, keep_base=False))
        irons.append(lathe("leg_collar", [(r_t - 0.001, 0.215), (r_t + 0.013, 0.22), (r_t + 0.018, 0.236), (r_t + 0.018, 0.295), (r_t + 0.013, 0.31), (r_t - 0.001, 0.315)], segments=24))
        clamp_screws((z_tube - 0.04, z_tube - 0.1), radians(-90))
    else:
        bs = 0.4 + 0.055 * D
        blk = stone_block("base_stone", (bs, bs, bh), loc=(0, 0, bh / 2), chamfer=0.024, rough=0.006, rng=rb_)
        place(blk, rot=(0, 0, rb_.u(-8, 8)))
        stones.append(blk)
        plate = lathe("base_plate", [(0.0, bh - 0.004), (r_t + 0.085, bh - 0.004), (r_t + 0.088, bh + 0.005), (r_t + 0.084, bh + 0.012), (r_t + 0.03, bh + 0.014),
                                     (r_t + 0.014, bh + 0.026)] + tube(bh + 0.026, bh + 0.3)[2:], segments=24)
        irons.append(plate)
        bt = bolt("base_bolt", 0.011, 0.009, kind="hex")
        irons.append(instances_to_mesh("base_bolts", bt, [trs(((r_t + 0.062) * cos(radians(45 + 90 * k)), (r_t + 0.062) * sin(radians(45 + 90 * k)), bh + 0.012),
                                                              (0, 0, rb_.u(0, 60))) for k in range(4)], keep_base=False))
        rz0 = blk.rotation_euler.z
        for sx in (-1, 1):
            e = Vector((cos(rz0), sin(rz0), 0)) * sx
            c0 = e * (bs / 2 + 0.004) + Vector((0, 0, bh * 0.68))
            irons.append(torus(f"lift_staple{sx}", 0.02, 0.006, loc=tuple(c0), rot=(90, 0, degrees(rz0)), major_seg=10, minor_seg=5))
            irons.append(torus(f"lift_ring{sx}", 0.05, 0.007, loc=tuple(c0 + e * 0.012 + Vector((0, 0, -0.05))), rot=(0, 90 - sx * 9, degrees(rz0)), major_seg=16, minor_seg=5))
        clamp_screws((bh + 0.25, bh + 0.15), radians(-70))
    for o in irons:
        assign(o, iron_b)
    for o in forged:
        assign(o, wrought)
    if stones:
        st = mat("stone_rough", "#a9a08d")
        for o in stones:
            assign(o, st)
            paint(o, grime=height_ramp(0.0, 0.3), special=mask_mul(facing((0, 0, 1)), noise_mask(0.18, seed=p.seed + 3), 0.4 + 0.6 * age))

    # ------------------------------------------------------------------ weathering
    R_out = D / 2
    z_rim = z_h - L * sin(theta)
    age_f = 0.35 + 0.9 * age

    def soot(co, nr):
        # faint soot bloom above each lantern on the underside
        g = np.full(len(co), 0.06 * age_f)
        for (lx, ly) in lant_xy:
            g = g + 0.5 * np.exp(-((co[:, 0] - lx) ** 2 + (co[:, 1] - ly) ** 2) / (0.24 ** 2))
        return np.clip(g, 0, 1)

    for (o, g, f) in weathered:
        paint(o, grime=soot if isinstance(g, str) else g, special=f)

    def cloth_grime(co, nr):
        # folded cloth / tassels: soft dirt gathering towards the rim and the hem, nothing fine (vertex masks are coarse)
        r = np.hypot(co[:, 0], co[:, 1])
        rim = np.clip((r - 0.3 * R_out) / (0.7 * R_out), 0, 1)
        g = (0.05 + 0.2 * rim * rim * (0.5 + 0.5 * nzb(co, nr))) * age_f
        if opn > 0.8:
            g = g + 0.25 * age_f * np.clip((z_rim - 0.5 * val_h - co[:, 2]) / max(0.5 * val_h, 0.02), 0, 1)
        return np.clip(g, 0, 1)

    def cloth_fade(co, nr):
        if sw > 0.0:
            return np.clip(0.3 + 0.5 * nr[:, 2], 0, 1) * 0.4 * age       # furled: an even, slightly bleached outside
        return np.clip(nr[:, 2] * 1.3, 0, 1) * (0.3 + 0.7 * nz2(co, nr)) * age

    for o in cloths:
        paint(o, grime=cloth_grime, special=cloth_fade)
    for o in woods:
        if o.name.startswith("pole"):
            paint(o, grime=mask_mul(height_ramp(0.3, 1.0), 0.6),
                  wear=mask_mul(height_ramp(1.2, 1.65), height_ramp(1.2, 0.85), noise_mask(0.06, seed=p.seed + 4), 0.7))
    # ironwork: rust creeping up from the wet paving and standing on ledges, paint knocked off the rim by chair legs and boots,
    # bare bright steel where hands turn the clamp screws
    nzr = noise_mask(0.05, seed=p.seed + 2, lo=0.3, hi=0.62)
    nzr2 = noise_mask(0.2, seed=p.seed + 5, lo=0.35, hi=0.65)
    nzc = noise_mask(0.035, seed=p.seed + 7, lo=0.45, hi=0.7)

    def rust(co, nr):
        foot = np.clip(1.0 - co[:, 2] / 0.11, 0, 1)
        ledge = np.clip(nr[:, 2], 0, 1) * nzr2(co, nr)
        return np.clip((0.8 * foot + 0.45 * ledge) * (0.3 + 0.7 * nzr(co, nr)) * (0.5 + 0.8 * age), 0, 1)

    def chips(co, nr):
        rim = np.clip(1.0 - np.abs(co[:, 2] - 0.05) / 0.06, 0, 1) * (np.hypot(co[:, 0], co[:, 1]) > 0.15)
        return np.clip((0.15 + 0.85 * rim) * nzc(co, nr) * (0.5 + 0.8 * age), 0, 1)

    for o in irons:
        paint(o, special=rust, wear=chips, grime=mask_mul(height_ramp(0.0, 0.2), 0.5))
    for o in forged:
        paint(o, wear=0.75 if "bar" in o.name or "ball" in o.name else 0.2, special=mask_mul(nzr, 0.3 + 0.5 * age))
    for o in hooks:
        paint(o, special=mask_mul(nzr2, 0.3 + 0.5 * age), wear=0.15)
    socket("canopy_centre", "attach", (0, 0, z_run - 0.1), role="lantern")
