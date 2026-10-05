"""Victorian park bandstand. A polygonal plinth of individually dressed stones (base course, die, bull-nosed
coping) carries a boarded timber deck reached by a flight of stone steps between raked cheek walls. Slender
fluted cast-iron columns on octagonal pedestals stand on the coping at every corner; they carry an iron ring
beam with a pierced frieze and scrolled spandrel brackets, and a wrought railing spans every bay but the
stair bay. On the beam sits a swept (tent) or bell-cast roof of sheet metal with batten-roll seams and hip
rolls (or slates with lead hips), boarded underneath with hip ribs, a scalloped eaves board, a coronet of
cresting, a finial and optionally a louvred cupola. A glazed pendant lantern hangs from the central boss and
a festoon of bulbs is strung along the eaves. Everything fits inside the `diameter` footprint circle.
"""
from tk import *
from math import pi, cos, sin, tan, radians, degrees, atan2, hypot, sqrt, exp
import numpy as np
from mathutils import Vector

META = {"name": "Octagonal bandstand", "class": "landmark", "mount": "ground", "focus": 0.52}
PARAMS = {
    "diameter": {"default": 6.0, "min": 4.0, "max": 8.0, "desc": "footprint diameter incl. steps and eaves (m), centred on the origin"},
    "columns": {"default": 8, "choices": [6, 8, 10, 12], "desc": "number of columns = sides of the plan polygon"},
    "plinth_height": {"default": 0.66, "min": 0.3, "max": 1.0, "desc": "deck height above the paving (m); sets the number of steps"},
    "column_height": {"default": 2.9, "min": 2.5, "max": 3.6, "desc": "deck to underside of the ring beam (m)"},
    "roof": {"default": "zinc", "choices": ["zinc", "copper", "lead", "slate"], "desc": "roof covering"},
    "roof_form": {"default": "tent", "choices": ["tent", "bell"], "desc": "swept concave tent or bell-cast ogee"},
    "roof_rise": {"default": 0.66, "min": 0.4, "max": 1.0, "desc": "roof height as a fraction of the eaves radius"},
    "cupola": {"default": False, "desc": "louvred ventilating cupola with its own small roof"},
    "railing": {"default": "scroll", "choices": ["scroll", "bars", "none"], "desc": "railing infill between the columns"},
    "paint": {"default": "#1f3a30", "desc": "paint colour of the ironwork"},
    "trim": {"default": "#d8ccb0", "desc": "paint colour of the eaves boards and roof timbers"},
    "soffit": {"default": "#d9c3a0", "desc": "stain colour of the boarded soffit"},
    "gilding": {"default": True, "desc": "gilded capitals, bosses, date and finial"},
    "date": {"default": "1887", "desc": "raised date on the ring beam above the steps ('' = none)"},
    "lit": {"default": True, "desc": "lantern and festoon switched on"},
    "festoon_bulbs": {"default": 5, "min": 0, "max": 8, "desc": "festoon bulbs per bay along the eaves (0 = no festoon)"},
    "music_stands": {"default": 4, "min": 0, "max": 7, "desc": "music stands and chairs round the conductor's rostrum (0 = bare deck)"},
    "intensity": {"default": 65.0, "min": 20, "max": 200, "desc": "central lantern intensity (cd)"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of weathering (moss, rust, patina, worn paint)"},
}
VARIANTS = {
    "copper_bell": {"roof": "copper", "roof_form": "bell", "roof_rise": 0.8, "paint": "#5a1616", "trim": "#e3d6b4",
                    "railing": "bars", "diameter": 6.6, "wear": 0.8, "seed": 4},
    "hex_kiosk": {"diameter": 4.6, "columns": 6, "roof": "slate", "roof_rise": 0.85, "paint": "#20232a", "column_height": 2.6,
                  "plinth_height": 0.45, "festoon_bulbs": 4, "gilding": False, "date": "", "music_stands": 0, "seed": 11},
    "grand_cupola": {"diameter": 8.0, "columns": 10, "cupola": True, "roof": "lead", "paint": "#1c2f4a", "column_height": 3.4,
                     "plinth_height": 0.85, "roof_rise": 0.6, "festoon_bulbs": 6, "date": "1902", "music_stands": 7, "seed": 23},
    "copper_cupola": {"diameter": 7.0, "columns": 8, "cupola": True, "roof": "copper", "roof_form": "bell", "roof_rise": 0.7,
                      "paint": "#f0ead8", "trim": "#7d2a22", "soffit": "#a8c0b8", "railing": "bars", "wear": 0.3, "seed": 31},
}


def _cr(ctrl, per=14):
    """Catmull-Rom through 2D control points -> dense (n, 2) array."""
    P = np.array([ctrl[0]] + list(ctrl) + [ctrl[-1]], float)
    P[0] = 2 * P[1] - P[2]
    P[-1] = 2 * P[-2] - P[-3]
    out = []
    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, per, endpoint=False):
            out.append(0.5 * ((2 * P[i]) + (-P[i - 1] + P[i + 1]) * t + (2 * P[i - 1] - 5 * P[i] + 4 * P[i + 1] - P[i + 2]) * t * t
                              + (-P[i - 1] + 3 * P[i] - 3 * P[i + 1] + P[i + 2]) * t ** 3))
    out.append(P[-2])
    return np.array(out)


def _volute(cx, cz, r_out, r_in, a0, turns, ccw, n=20):
    """Spiral in the XZ plane from the outer end (angle a0, degrees) inwards."""
    sg = 1.0 if ccw else -1.0
    pts = []
    for i in range(n):
        t = i / (n - 1)
        a = radians(a0) + sg * turns * 2 * pi * t
        r = r_out + (r_in - r_out) * t
        pts.append((cx + r * cos(a), 0.0, cz + r * sin(a)))
    return pts


def _rect(a, b):
    return [(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)]


def build(p, rng):
    N = int(p.columns)
    R = p.diameter / 2
    half = pi / N
    cosN, tanN = cos(half), tan(half)
    A0, dA = -pi / 2, 2 * pi / N
    W = clamp(p.wear, 0.0, 1.0)
    lit = bool(p.lit)

    # ------------------------------------------------------------------ materials
    stone = mat("stone_smooth", "#b9ae9a")
    deck_m = mat("wood_oak", "#a5977f")
    iron = mat("iron_painted", p.paint)
    trim = mat("wood_painted", p.trim)
    soff = mat("wood_planks", p.soffit)
    gilt = mat("brass", rough=0.8) if p.gilding else iron
    roofm = mat({"zinc": "zinc", "copper": "copper_patina", "lead": "lead", "slate": "roof_slate"}[p.roof])
    flash = mat("lead") if p.roof == "slate" else roofm
    glow = mat("emissive", "#ffd9a0", glow=8) if lit else mat("glass_frosted", "#e8e0d0", glow=0)
    glass = mat("glass_lantern", glow=0.8 if lit else 0.0)

    # ------------------------------------------------------------------ dimensions
    ph = min(p.plinth_height, 0.125 * p.diameter)
    n_ris = max(2, int(round(ph / 0.165)))
    rise = ph / n_ris
    going = 0.30 if p.diameter >= 5.0 else 0.27
    proj = (n_ris - 1) * going
    ov = clamp(0.07 * p.diameter, 0.3, 0.5)                   # eaves overhang beyond the column line
    nd = 0.30                                                 # the top tread is recessed into the plinth
    ap = min(R - proj + nd - 0.08, (R - 0.02) * cosN - ov + 0.17)   # plinth wall apothem
    ca = ap - 0.17                                            # column line apothem
    ea = ca + ov                                              # eaves apothem
    Rc = ca / cosN
    bay = 2 * ca * tanN
    zd = ph
    Hc = p.column_height
    zc = zd + Hc
    bd = 0.2
    zb = zc + bd
    fh = 0.2                                                  # frieze height
    zr = 0.98                                                 # handrail height above the deck

    def fdir(k):
        a = A0 + k * dA
        return (cos(a), sin(a)), (-sin(a), cos(a)), a

    def P2(k, xl, rho):
        a = A0 + k * dA
        return (cos(a) * rho - sin(a) * xl, sin(a) * rho + cos(a) * xl)

    def fblock(name, k, f0, f1, rho_i, rho_o, z0, h, bevel=0.0, gap=0.0):
        pts = [P2(k, f0 * rho_o * tanN + gap, rho_o), P2(k, f1 * rho_o * tanN - gap, rho_o),
               P2(k, f1 * rho_i * tanN - gap, rho_i), P2(k, f0 * rho_i * tanN + gap, rho_i)]
        return extrude_profile(name, pts, h, loc=(0, 0, z0), bevel=bevel, segments=1)

    def face_m(k, rho, z, xl=0.0):
        x, y = P2(k, xl, rho)
        return trs((x, y, z), (0, 0, degrees(A0 + k * dA) + 90))

    def cuts(n, r, jit=0.1):
        return [-1.0] + [-1 + 2 * (i + r.u(-jit, jit)) / n for i in range(1, n)] + [1.0]

    # ------------------------------------------------------------------ plinth: dressed stones
    r = rng.fork("plinth")
    st = []
    Ws = min(bay - 0.34, 3.0)                                 # clear stair width
    ct = 0.24                                                 # cheek wall thickness
    f_s = (Ws / 2 + ct - 0.02) / (ap * tanN)

    def stones(k, c):
        """(f0, f1) spans of the stones on face k; the stair bay keeps only what lies outside the cheeks."""
        if k != 0:
            return list(zip(c[:-1], c[1:]))
        return [] if f_s > 0.93 else [(-1.0, -f_s), (f_s, 1.0)]

    side_w = 2 * ap * tanN
    bh = min(0.2, ph * 0.32)
    chh = 0.12
    nb = max(2, int(round(side_w / 0.8)))
    for k in range(N):
        c = cuts(nb, r)
        for (fa_, fb_) in stones(k, c):
            st.append(fblock("base", k, fa_, fb_, ap - 0.06, ap + 0.055 + r.u(-0.003, 0.003), 0.0, bh + r.u(-0.003, 0.0),
                             bevel=0.014, gap=0.003))
    wall_h = zd - chh - bh
    ncr = max(1, int(round(wall_h / 0.28)))
    cht = wall_h / ncr
    nw = max(2, int(round(side_w / 0.62)))
    for j in range(ncr):
        for k in range(N):
            if j % 2 == 0:
                c = cuts(nw, r)
            else:
                c = [-1.0] + [-1 + 2 * (i - 0.5 + r.u(-0.1, 0.1)) / nw for i in range(1, nw + 1)] + [1.0]
            for (fa_, fb_) in stones(k, c):
                st.append(fblock("wall", k, fa_, fb_, ap - 0.08, ap + r.u(-0.004, 0.003), bh + j * cht + 0.003, cht - 0.006,
                                 bevel=0.007, gap=0.003))
    ncp = max(2, int(round(side_w / 0.95)))
    for k in range(N):
        c = cuts(ncp, r, 0.07)
        for (fa_, fb_) in stones(k, c):
            st.append(fblock("coping", k, fa_, fb_, ap - 0.34, ap + 0.06, zd - chh + 0.002, chh - 0.002 + r.u(-0.002, 0.0015),
                             bevel=0.016, gap=0.003))
    st.append(extrude_profile("core", [((ap - 0.3) / cosN * cos(A0 + (k + 0.5) * dA), (ap - 0.3) / cosN * sin(A0 + (k + 0.5) * dA))
                                       for k in range(N)], zd - 0.032))
    for o in st:
        assign(o, stone)
    plinth = join(st, "plinth")

    def north_low(co, nr):
        return np.clip(nr[:, 1] * 0.7 + 0.5, 0, 1) * np.clip(1.15 - co[:, 2] / (zd + 0.2), 0, 1)

    paint(plinth, grime=mask_mul(height_ramp(0.0, 0.5), 0.45 + 0.5 * W),
          special=mask_mul(north_low, noise_mask(0.45, seed=p.seed + 3, lo=0.35, hi=0.7), 0.35 + 0.9 * W))

    # ------------------------------------------------------------------ deck: boards cut to the polygon
    y_top = -(ap + 0.06 - nd)                                 # nosing of the top tread (deck level)
    y_thr = y_top + 0.3 + 0.004
    rho_d = ap - 0.345
    Rd = rho_d / cosN

    def hw(y):
        m = 1e9
        for k in range(N):
            a = A0 + k * dA
            if cos(a) > 1e-6:
                m = min(m, (rho_d - y * sin(a)) / cos(a))
        return m

    corners_y = sorted(set(round(Rd * sin(A0 + (k + 0.5) * dA), 5) for k in range(N)))
    r = rng.fork("deck")
    ny = max(4, int(round(2 * rho_d / 0.135)))
    bw = 2 * rho_d / ny
    boards = []
    for i in range(ny):
        y0 = -rho_d + i * bw + 0.002
        y1 = y0 + bw - 0.004
        x0, x1 = hw(y0), hw(y1)
        if min(x0, x1) < 0.03:
            continue
        mids = [yc for yc in corners_y if y0 + 1e-4 < yc < y1 - 1e-4]
        ym = mids[0] if mids else None
        xm = hw(ym) if mids else None
        xs = [-max(x0, x1) - 1.0]
        if min(x0, x1) > 1.0:
            xs.append(r.u(-0.55, 0.55) * min(x0, x1))
            if min(x0, x1) > 1.9 and r.chance(0.5):
                xs.append(xs[-1] + (1.0 if xs[-1] < 0 else -1.0) * r.u(0.9, 1.3))
        xs = sorted(xs) + [max(x0, x1) + 1.0]
        dz = r.u(-0.0015, 0.0015)
        spans = [(xs[j] + 0.0015, xs[j + 1] - 0.0015) for j in range(len(xs) - 1)]
        if y0 < y_thr:                                        # make room for the threshold stone and the cheeks
            spans = [(-99.0, -(Ws / 2 + ct + 0.012)), (Ws / 2 + ct + 0.012, 99.0)]
        for (xa, xb) in spans:
            if min(xb, x0) - max(xa, -x0) < 0.06 or min(xb, x1) - max(xa, -x1) < 0.06:
                continue
            pts = [(max(xa, -x0), y0), (min(xb, x0), y0)]
            if ym is not None and xm < xb:
                pts.append((xm, ym))
            pts += [(min(xb, x1), y1), (max(xa, -x1), y1)]
            if ym is not None and -xm > xa:
                pts.append((-xm, ym))
            b = extrude_profile("board", pts, 0.03, loc=(0, 0, zd - 0.03 + dz), bevel=0.003, segments=1)
            uv_grain(b, "X")
            boards.append(b)
    for o in boards:
        assign(o, deck_m)
    deck = join(boards, "deck")
    densify(deck, 0.45)

    def deck_wear(co, nr):
        x, y = co[:, 0], co[:, 1]
        c = np.exp(-(x * x + y * y) / (0.45 * rho_d) ** 2) * 0.55
        s = np.exp(-(x / 0.6) ** 2) * np.clip(-y / rho_d, 0, 1) * 0.7
        return np.clip(c + s, 0, 1)

    def deck_edge(co, nr):
        return np.clip((np.hypot(co[:, 0], co[:, 1]) / rho_d - 0.72) * 3.0, 0, 1)

    paint(deck, wear=mask_mul(deck_wear, noise_mask(0.3, seed=p.seed + 5, lo=0.2, hi=0.8), 0.4 + 0.6 * W),
          grime=mask_mul(deck_edge, 0.35 + 0.5 * W))

    # ------------------------------------------------------------------ steps and cheek walls (front, -Y)
    rs = rng.fork("steps")
    stn = []
    for i in range(0, n_ris):
        top = zd - i * rise
        yf = y_top - i * going
        dpt = going + 0.035 if i else 0.3
        stn.append(stone_block(f"step{i}", (Ws + 0.02, dpt, rise), loc=(0, yf + dpt / 2, top - rise / 2), rng=rs, chamfer=0.012, detail=0.14))
    y_f = y_top - proj
    y_b = y_top + 0.3
    zck = zd + 0.004
    pitch = rise / going

    def cheek_top(y):
        return zd + (y - y_top) * pitch + 0.14

    y_k = y_top - 0.14 / pitch
    y_p = y_f + 0.17
    if y_p < y_k - 0.03:
        z_p = cheek_top(y_p)
        cprof = [(y_f, 0.0), (y_b, 0.0), (y_b, zck), (y_k, zck), (y_p, z_p), (y_f, z_p)]
    else:
        z_p = zck
        cprof = [(y_f, 0.0), (y_b, 0.0), (y_b, zck), (y_f, zck)]
    xs_cheek = Ws / 2 + ct / 2 + 0.01
    for sx in (-1, 1):
        ck = extrude_profile("cheek", cprof, ct, loc=(sx * xs_cheek - ct / 2, 0, 0), rot=(90, 0, 90), bevel=0.014, segments=1)
        apply_transform(ck)
        densify(ck, 0.3)
        displace_noise(ck, 0.003, 0.3, seed=p.seed + sx)
        stn.append(ck)
    for o in stn:
        assign(o, stone)
    steps = join(stn, "steps")

    def tread_wear(co, nr):
        return np.exp(-(co[:, 0] / (0.3 * Ws)) ** 2) * np.clip(nr[:, 2], 0, 1) * (co[:, 1] < y_top + 0.02)

    paint(steps, wear=mask_mul(tread_wear, 0.35 + 0.6 * W), grime=mask_mul(height_ramp(0.0, 0.4), 0.5 + 0.4 * W),
          special=mask_mul(noise_mask(0.3, seed=p.seed + 8, lo=0.5, hi=0.8), lambda co, nr: (np.abs(co[:, 0]) > Ws * 0.42) * 1.0, 0.3 + 0.7 * W))

    # stair handrails: newel on the cheek pad, raking rail to the column
    ir = []
    y_n = y_f + 0.085
    hn = max(0.72, (rise + 0.92 + 0.085 * pitch) - z_p)
    z_n = z_p + hn
    for sx in (-1, 1):
        x = sx * xs_cheek
        ir.append(lathe("newel", [(0.0, 0.0), (0.05, 0.0), (0.05, 0.02), (0.034, 0.035), (0.03, 0.07), (0.022, 0.09), (0.02, hn - 0.14), (0.03, hn - 0.125),
                                  (0.03, hn - 0.105), (0.02, hn - 0.09), (0.018, hn - 0.06), (0.036, hn - 0.04), (0.044, hn - 0.012), (0.036, hn + 0.016),
                                  (0.015, hn + 0.034), (0.0, hn + 0.038)], segments=14, loc=(x, y_n, z_p)))
        pa = Vector((x, y_n, z_n - 0.1))
        pb = Vector((x, -ca - 0.055, zd + zr))
        ir.append(sweep("stair_rail", [tuple(pa), tuple(pb)], 0.02, sides=10, smooth_path=False))
        la, lb = pa - Vector((0, 0, 0.5)), pb - Vector((0, 0, 0.62))
        la.z = max(la.z, z_p + 0.07)
        ir.append(sweep("stair_rail_low", [tuple(la), tuple(lb)], _rect(0.03, 0.012), smooth_path=False))
        nbal = max(2, int((pb.y - pa.y) / 0.13))
        for i in range(1, nbal):
            t = i / nbal
            q0, q1 = la.lerp(lb, t), pa.lerp(pb, t)
            ir.append(box("stair_bal", (0.014, 0.014, q1.z - q0.z), loc=(x, q0.y, (q0.z + q1.z) / 2)))

    # ------------------------------------------------------------------ cast-iron vent grilles in the plinth
    if wall_h >= 0.2:
        vw, vh = 0.36, min(0.2, wall_h - 0.07)
        nsl = 7
        slots = [[(x + (-vw / 2 + 0.04 + i * (vw - 0.08) / (nsl - 1)), y) for x, y in rounded_rect(0.018, vh - 0.05, 0.008, 1)] for i in range(nsl)]
        vent = shape_extrude("vent", rounded_rect(vw, vh, 0.012, 2), slots, depth=0.012, rot=(90, 0, 0))
        apply_transform(vent)
        vback = box("vent_back", (vw - 0.03, 0.003, vh - 0.03), loc=(0, -0.003, 0))
        vent = join([vent, vback], "vent")
        ir.append(instances_to_mesh("vents", vent, [face_m(k, ap + 0.005, bh + wall_h / 2) for k in range(1, N, 2)], keep_base=False))

    # ------------------------------------------------------------------ conductor's rostrum and music stands
    nst = int(p.music_stands)
    if nst > 0:
        rf_ = rng.fork("stands")
        py = -0.4 * rho_d
        rw = 0.8
        rost = []
        for i in range(6):
            rost.append(plank("rostrum_board", length=rw, width=rw / 6 - 0.004, thick=0.024,
                              loc=(0, py - rw / 2 + (i + 0.5) * rw / 6, zd + 0.168), rng=rf_, warp=0.002))
        for sy in (-1, 1):
            rost.append(plank("rostrum_side", length=rw - 0.02, width=0.155, thick=0.024, loc=(0, py + sy * (rw / 2 - 0.02), zd + 0.078),
                              rot=(90, 0, 0), rng=rf_))
        for sx in (-1, 1):
            rost.append(plank("rostrum_end", length=rw - 0.07, width=0.155, thick=0.024, loc=(sx * (rw / 2 - 0.02), py, zd + 0.078),
                              rot=(90, 0, 90), rng=rf_))
        for o in rost:
            assign(o, deck_m)
        rostrum = join(rost, "rostrum")
        paint(rostrum, wear=mask_mul(facing((0, 0, 1)), noise_mask(0.2, seed=p.seed + 21, lo=0.3, hi=0.8), 0.5 + 0.4 * W))
        sparts = [cylinder("stand_stem", 0.009, 0.82, loc=(0, 0, 0.71), verts=8),
                  cylinder("stand_tube", 0.013, 0.42, loc=(0, 0, 0.5), verts=8),
                  lathe("stand_hub", [(0.013, 0.26), (0.024, 0.27), (0.024, 0.31), (0.013, 0.32)], segments=8),
                  lathe("stand_clamp", [(0.013, 0.69), (0.022, 0.695), (0.022, 0.715), (0.013, 0.72)], segments=8)]
        for q in range(3):
            aq = radians(90 + q * 120)
            sparts.append(sweep("stand_leg", [(0.012 * cos(aq), 0.012 * sin(aq), 0.3), (0.27 * cos(aq), 0.27 * sin(aq), 0.006)], _rect(0.016, 0.006),
                                smooth_path=False))
            sparts.append(sweep("stand_stay", [(0.012 * cos(aq), 0.012 * sin(aq), 0.12), (0.15 * cos(aq), 0.15 * sin(aq), 0.145)], 0.004, sides=5,
                                smooth_path=False))
        dparts = [box("desk", (0.46, 0.004, 0.3), loc=(0, 0, 0.15), bevel=0.0015), box("desk_lip", (0.46, 0.04, 0.004), loc=(0, -0.02, 0.002)),
                  box("desk_rib", (0.02, 0.012, 0.3), loc=(0, 0.008, 0.15))]
        dsk = join(dparts, "desk")
        place(dsk, loc=(0, -0.012, 1.0), rot=(-20, 0, 0))
        pages = join([box("page", (0.2, 0.0014, 0.27), loc=(sx * 0.103, -0.0034 - 0.0008 * sx, 0.142), rot=(0, 0, sx * 2.0)) for sx in (-1, 1)], "sheet_music")
        place(pages, loc=(0, -0.012, 1.0), rot=(-20, 0, 0))
        apply_transform(pages)
        stand = join(sparts + [dsk], "stand")
        ra = min(1.35, 0.66 * rho_d)
        sm_ = []
        for i in range(nst):
            ph_ = (-58 + 116 * (i + 0.5) / nst) + rf_.u(-5, 5)
            rr2 = ra * rf_.u(0.9, 1.05)
            sm_.append(trs((rr2 * sin(radians(ph_)), py + 0.3 + rr2 * cos(radians(ph_)), zd), (0, 0, 180 - ph_ + rf_.u(-14, 14))))
        ir.append(instances_to_mesh("music_stands", stand, sm_, keep_base=False))
        # bentwood chairs behind the stands (only where they stay clear of the railing)
        arch = [(-0.17, -0.2, 0.0), (-0.145, -0.15, 0.45), (-0.15, -0.185, 0.72), (-0.09, -0.205, 0.86), (0.0, -0.21, 0.9),
                (0.09, -0.205, 0.86), (0.15, -0.185, 0.72), (0.145, -0.15, 0.45), (0.17, -0.2, 0.0)]
        inner = [(-0.085, -0.15, 0.46), (-0.09, -0.18, 0.68), (-0.05, -0.195, 0.77), (0.0, -0.2, 0.79), (0.05, -0.195, 0.77), (0.09, -0.18, 0.68),
                 (0.085, -0.15, 0.46)]
        cparts = [lathe("chair_seat", [(0.0, 0.43), (0.165, 0.43), (0.195, 0.44), (0.2, 0.452), (0.19, 0.463), (0.1, 0.468), (0.0, 0.462)], segments=16),
                  sweep("chair_back", arch, 0.013, sides=6, step=0.06),
                  sweep("chair_hoop", inner, 0.009, sides=5, step=0.05),
                  torus("chair_ring", 0.185, 0.007, loc=(0, -0.01, 0.2), major_seg=14, minor_seg=4)]
        for sx in (-1, 1):
            cparts.append(sweep("chair_leg", [(sx * 0.14, 0.13, 0.44), (sx * 0.17, 0.175, 0.0)], 0.013, sides=6, smooth_path=False))
        chair = join(cparts, "chair")
        chm = []
        gone = int(rf_.u(0, 2.5 * nst)) + 1                    # at most one chair has been carried off
        for m in sm_:
            mc = m @ trs((rf_.u(-0.08, 0.08), -0.62 + rf_.u(-0.05, 0.08), 0.0), (0, 0, rf_.u(-25, 25)))
            cpos = mc.translation
            if hypot(cpos.x, cpos.y) + 0.22 < ca - 0.08 and len(chm) + 1 != gone:
                chm.append(mc)
        if chm:
            ir.append(instances_to_mesh("chairs", chair, chm, keep_base=False))
        else:
            delete(chair)
        pm_ = [m for m in sm_ if rf_.chance(0.8)]
        if pm_:
            sheets = instances_to_mesh("sheet_music", pages, pm_, keep_base=False)
            assign(sheets, mat("paper", "#efe6cf") if not (p.roof == "slate" and p.gilding) else trim)   # stay within 10 materials
            paint(sheets, special=mask_mul(noise_mask(0.15, seed=p.seed + 22, lo=0.5, hi=0.9), 0.3 + 0.5 * W))
        else:
            delete(pages)

    # ------------------------------------------------------------------ columns
    hp, rp = 0.5, 0.118
    r0, r1 = 0.06, 0.049
    hcap = 0.36
    zs0, zs1 = hp, Hc - hcap

    def rad(z):
        return r0 + (r1 - r0) * (z - zs0) / (zs1 - zs0)

    ped = lathe("ped", [(0.0, 0.0), (rp + 0.035, 0.0), (rp + 0.035, 0.045), (rp + 0.014, 0.06), (rp, 0.085), (rp, 0.115), (rp - 0.014, 0.13),
                        (rp - 0.014, hp - 0.11), (rp - 0.002, hp - 0.095), (rp + 0.012, hp - 0.07), (rp + 0.012, hp - 0.04), (rp - 0.02, hp - 0.02),
                        (rp - 0.05, hp), (0.0, hp)], segments=8, rot=(0, 0, 22.5), smooth_angle=25)
    sp = [(r0 + 0.036, zs0 - 0.004), (r0 + 0.042, zs0 + 0.012), (r0 + 0.036, zs0 + 0.028), (r0 + 0.02, zs0 + 0.036), (r0 + 0.015, zs0 + 0.05),
          (r0 + 0.024, zs0 + 0.062), (r0 + 0.022, zs0 + 0.076), (r0 + 0.006, zs0 + 0.088)]
    sp += [(rad(z), z) for z in (zs0 + 0.1, zs0 + 0.14, zr - 0.09, zr - 0.05)]
    rr = rad(zr)
    sp += [(rr + 0.012, zr - 0.04), (rr + 0.012, zr - 0.024), (rr + 0.024, zr - 0.012), (rr + 0.027, zr), (rr + 0.024, zr + 0.012),
           (rr + 0.012, zr + 0.024), (rr + 0.012, zr + 0.04)]
    sp += [(rad(z), z) for z in (zr + 0.05, zr + 0.09, zs1 - 0.09, zs1 - 0.05)]
    sp += [(r1 + 0.012, zs1 - 0.04), (r1 + 0.017, zs1 - 0.028), (r1 + 0.012, zs1 - 0.016), (r1 + 0.003, zs1 - 0.008), (r1 + 0.003, zs1 + 0.03),
           (r1 + 0.008, zs1 + 0.1), (r1 + 0.022, zs1 + 0.17), (r1 + 0.05, zs1 + 0.235), (r1 + 0.075, zs1 + 0.275), (r1 + 0.08, zs1 + 0.29),
           (0.0, zs1 + 0.29)]
    nf = 10
    shaft = lathe("shaft", sp, segments=nf * 3, smooth_angle=60)

    def flute(co):
        z = co.z
        if z < zr:
            e = min(smoothstep(zs0 + 0.1, zs0 + 0.14, z), 1.0 - smoothstep(zr - 0.09, zr - 0.05, z))
        else:
            e = min(smoothstep(zr + 0.05, zr + 0.09, z), 1.0 - smoothstep(zs1 - 0.09, zs1 - 0.05, z))
        rxy = hypot(co.x, co.y)
        if e <= 0.0 or rxy < 1e-6:
            return co
        c = 0.5 + 0.5 * cos(atan2(co.y, co.x) * nf)
        kk = (rxy - 0.0055 * e * (1.0 - c ** 3)) / rxy
        return Vector((co.x * kk, co.y * kk, co.z))

    transform_verts(shaft, flute)
    aba = box("abacus", (0.27, 0.27, hcap - 0.29), loc=(0, 0, Hc - (hcap - 0.29) / 2), bevel=0.008)
    Lb = min(0.6, 0.38 * bay)                                 # spandrel bracket reach / drop
    Hb = 1.2 * Lb
    z_sp = Hc - fh - Hb
    rsp = rad(min(max(z_sp, zs0), zs1))
    col_parts = [ped, shaft, aba,
                 lathe("spring", [(rsp - 0.002, z_sp - 0.05), (rsp + 0.01, z_sp - 0.04), (rsp + 0.012, z_sp - 0.02), (rsp + 0.022, z_sp - 0.01),
                                  (rsp + 0.022, z_sp + 0.008), (rsp + 0.01, z_sp + 0.02), (rsp - 0.002, z_sp + 0.035)], segments=20)]
    hd = bolt("hd", 0.011, 0.009)
    col_parts.append(instances_to_mesh("hds", hd, [trs(((rp + 0.018) * cos(radians(22.5 + q * 90)), (rp + 0.018) * sin(radians(22.5 + q * 90)), 0.045))
                                                   for q in range(4)], keep_base=False))
    colbase = join(col_parts, "colbase")
    leaf = surface("leaf", lambda u, v: ((u - 0.5) * 0.075 * (1 - v * v * 0.8), -0.016 * sin(v * pi * 0.9) - 0.05 * v ** 3 - 0.005 * cos((u - 0.5) * pi),
                                         v * 0.24), 3, 6, thickness=0.004)
    realize(leaf)
    cm, lm = [], []
    for k in range(N):
        ang = A0 + (k + 0.5) * dA
        M = trs((Rc * cos(ang), Rc * sin(ang), zd), (0, 0, degrees(ang) + 90))
        cm.append(M)
        for q in range(8):
            aq = q * 45.0 + 22.5
            lm.append(M @ trs(((r1 + 0.006) * sin(radians(aq)), -(r1 + 0.006) * cos(radians(aq)), zs1 + 0.025), (0, 0, aq), 1.0 + 0.1 * (q % 2)))
    cols = instances_to_mesh("columns", colbase, cm, keep_base=False)
    leaves = instances_to_mesh("leaves", leaf, lm, keep_base=False)
    assign(cols, iron)
    assign(leaves, gilt)
    ir.append(cols)

    # ------------------------------------------------------------------ ring beam, frieze, spandrel brackets
    for k in range(N):
        d, t, a = fdir(k)
        bm = fblock("beam", k, -1, 1, ca - 0.06, ca + 0.06, zc, bd, bevel=0.004)
        ir.append(bm)
        ir.append(fblock("beam_cornice", k, -1, 1, ca - 0.085, ca + 0.085, zb - 0.035, 0.035, bevel=0.006))
        ir.append(fblock("beam_fillet", k, -1, 1, ca - 0.075, ca + 0.075, zc, 0.028, bevel=0.004))
    rv = sphere("rivet", 0.009, segments=6, rings=4, scale=(1, 1, 0.65))
    apply_transform(rv)
    rvm = []
    nrv = max(3, int(bay / 0.22))
    for k in range(N):
        d, t, a = fdir(k)
        for i in range(nrv):
            xl = (i + 0.5) / nrv * bay - bay / 2
            for sgn in (1, -1):
                x, y = P2(k, xl, ca + sgn * 0.06)
                rvm.append(trs((x, y, zc + 0.065), (90, 0, degrees(a) + 90 + (0 if sgn > 0 else 180))))
    ir.append(instances_to_mesh("rivets", rv, rvm, keep_base=False))

    wf = bay - 0.22
    nh = max(2, int(round(wf / 0.17)))
    holes = []
    for i in range(nh):
        cx = -wf / 2 + (i + 0.5) * wf / nh
        holes.append(circle_pts(0.062, 10, center=(cx, -fh / 2)))
        if i < nh - 1:
            cx2 = cx + 0.5 * wf / nh
            holes.append([(cx2, -fh / 2 - 0.05), (cx2 + 0.016, -fh / 2), (cx2, -fh / 2 + 0.05), (cx2 - 0.016, -fh / 2)])
    fr = shape_extrude("frieze", [(-wf / 2, -fh), (wf / 2, -fh), (wf / 2, 0.0), (-wf / 2, 0.0)], holes, depth=0.016, loc=(0, 0.008, 0),
                       rot=(90, 0, 0), bevel=0.002)
    apply_transform(fr)
    ir.append(instances_to_mesh("friezes", fr, [face_m(k, ca, zc) for k in range(N)], keep_base=False))

    # bracket in its local XZ plane: x away from the column axis, z = 0 at the beam underside
    x0 = r1 + 0.004
    zt = -fh
    arc = [(x0 + Lb * (1 - cos(radians(th))), 0.0, zt - Hb + Hb * sin(radians(th))) for th in np.linspace(0, 90, 15)]
    bar = _rect(0.012, 0.034)
    bparts = [sweep("br_arc", arc, bar, up=(0, 1, 0), smooth_path=False)]
    arc_np = np.array(arc)
    lo, hi = 0.01, Lb
    for _ in range(24):
        mid = (lo + hi) / 2
        cxr, czr = x0 + mid + 0.004, zt - mid - 0.004
        if np.min(np.hypot(arc_np[:, 0] - cxr, arc_np[:, 2] - czr)) > mid + 0.006:
            lo = mid
        else:
            hi = mid
    rg = lo
    cxr, czr = x0 + rg + 0.004, zt - rg - 0.004
    bparts.append(sweep("br_ring", [(cxr + rg * cos(q), 0.0, czr + rg * sin(q)) for q in np.linspace(0, 2 * pi, 20, endpoint=False)],
                        _rect(0.009, 0.03), closed=True, up=(0, 1, 0)))
    for q in (45, 135):
        bparts.append(box("br_spoke", (2 * rg - 0.004, 0.012, 0.007), loc=(cxr, 0, czr), rot=(0, q, 0)))
    taper = lambda tt: 1.0 - 0.5 * tt
    thv = radians(40)
    wv = Lb * (1 - cos(thv))
    rv1 = 0.4 * wv
    zv = zt - Hb + Hb * sin(thv)
    vol = [(x0 + 0.004, 0.0, czr)] + _volute(x0 + 0.004 + rv1, zv, rv1, rv1 * 0.2, 180, 1.35, True)
    bparts.append(sweep("br_vol_v", vol, bar, up=(0, 1, 0), scale_fn=taper, step=0.02))
    thh = radians(50)
    xh = x0 + Lb * (1 - cos(thh))
    rv2 = 0.4 * Hb * (1 - sin(thh))
    vol = [(cxr, 0.0, zt - 0.004)] + _volute(xh, zt - 0.004 - rv2, rv2, rv2 * 0.2, 90, 1.35, False)
    bparts.append(sweep("br_vol_h", vol, bar, up=(0, 1, 0), scale_fn=taper, step=0.02))
    bparts.append(box("br_top", (Lb, 0.03, 0.01), loc=(x0 + Lb / 2, 0, zt - 0.005)))
    brk = join(bparts, "bracket")
    boss = sphere("br_boss", 0.032, loc=(cxr, 0, czr), segments=12, rings=8, scale=(1, 0.7, 1))
    apply_transform(boss)
    bm_, bsm = [], []
    for k in range(N):
        ang = A0 + (k + 0.5) * dA
        cpos = (Rc * cos(ang), Rc * sin(ang), zc)
        for (kk, sgn) in ((k, -1), ((k + 1) % N, 1)):
            d, t, a = fdir(kk)
            M = trs(cpos, (0, 0, degrees(atan2(sgn * t[1], sgn * t[0]))))
            bm_.append(M)
            bsm.append(M)
    ir.append(instances_to_mesh("brackets", brk, bm_, keep_base=False))
    bosses = instances_to_mesh("bosses", boss, bsm, keep_base=False)
    assign(bosses, gilt)

    if p.date:
        dt = text("date", str(p.date), size=0.1, depth=0.007, loc=(0, -(ca + 0.06), zc + 0.052), font="serif")
        assign(dt, gilt)

    # ------------------------------------------------------------------ railing (one panel, stamped into every bay but the stair bay)
    if p.railing != "none":
        wb = bay - 2 * (rad(zr) + 0.01)
        rp_ = []
        hr_prof = [(-0.027, -0.012), (0.027, -0.012), (0.032, 0.0), (0.027, 0.012), (0.013, 0.021), (-0.013, 0.021), (-0.027, 0.012), (-0.032, 0.0)]
        rp_.append(sweep("rail_top", [(-wb / 2, 0, zr), (wb / 2, 0, zr)], hr_prof, smooth_path=False))
        z_mid, z_low = 0.8, 0.115
        rp_.append(box("rail_mid", (wb, 0.032, 0.012), loc=(0, 0, z_mid)))
        rp_.append(box("rail_low", (bay - 0.16, 0.036, 0.014), loc=(0, 0, z_low)))
        rr_ = (zr - 0.012 - z_mid - 0.006) / 2
        nr = max(2, int(wb / (2 * rr_ + 0.012)))
        for i in range(nr):
            rp_.append(torus("rail_ring", rr_ - 0.005, 0.0055, loc=(-wb / 2 + (i + 0.5) * wb / nr, 0, z_mid + 0.006 + rr_), rot=(90, 0, 0),
                             major_seg=12, minor_seg=4))
        zb0, zb1 = z_low + 0.007, z_mid - 0.006
        sq = _rect(0.014, 0.014)
        if p.railing == "bars":
            nbar = max(3, int(round(wb / 0.115)))
            for i in range(nbar):
                x = -wb / 2 + (i + 0.5) * wb / nbar
                if i % 2 == 0:
                    rp_.append(sweep("bar_tw", [(x, 0, zb0), (x, 0, zb1)], sq, step=0.05, smooth_path=False, twist_deg=540, up=(0, 1, 0)))
                else:
                    rp_.append(box("bar", (0.013, 0.013, zb1 - zb0), loc=(x, 0, (zb0 + zb1) / 2)))
                    rp_.append(sphere("knop", 0.019, loc=(x, 0, (zb0 + zb1) / 2), segments=8, rings=6, scale=(1, 1, 1.5)))
        else:
            nu_ = max(1, int(round(wb / 0.27)))
            un = wb / nu_
            rb_ = min(0.048, un * 0.19)
            sbar = _rect(0.007, 0.018)
            tp2 = lambda tt: 0.6 + 0.4 * min(1.0, 5.0 * min(tt, 1.0 - tt))
            for i in range(nu_ + 1):
                x = -wb / 2 + i * un
                if 0 < i < nu_:
                    rp_.append(box("bar", (0.014, 0.014, zb1 - zb0), loc=(x, 0, (zb0 + zb1) / 2)))
                if i < nu_:
                    xc = x + un / 2
                    for sg in (-1, 1):
                        e = 0.006
                        bot = _volute(sg * (e + rb_), zb0 + rb_ + 0.004, rb_, rb_ * 0.22, 180 if sg > 0 else 0, 1.3, sg > 0, n=16)
                        topv = _volute(sg * (e + rb_), zb1 - rb_ - 0.004, rb_, rb_ * 0.22, 180 if sg > 0 else 0, 1.3, sg < 0, n=16)
                        zm = (zb0 + zb1) / 2
                        path = list(reversed(bot)) + [(sg * e, 0, zm)] + topv
                        path = [(xc + q[0], 0.0, q[2]) for q in path]
                        rp_.append(sweep("scroll", path, sbar, up=(0, 1, 0), scale_fn=tp2, step=0.03))
                    for zz in (zb0 + 2 * rb_ + 0.03, (zb0 + zb1) / 2, zb1 - 2 * rb_ - 0.03):
                        rp_.append(box("collar", (0.032, 0.024, 0.014), loc=(xc, 0, zz)))
        panel = join(rp_, "rail_panel")
        ir.append(instances_to_mesh("railing", panel, [face_m(k, ca, zd) for k in range(1, N)], keep_base=False))

    # ------------------------------------------------------------------ roof
    if p.roof_form == "tent":
        curve = _cr([(0.0, 0.0), (0.25, 0.085), (0.5, 0.26), (0.75, 0.57), (1.0, 1.0)])
    else:
        curve = _cr([(0.0, 0.0), (0.1, 0.03), (0.22, 0.2), (0.4, 0.48), (0.62, 0.7), (0.82, 0.83), (1.0, 1.0)])
    CS = np.maximum.accumulate(curve[:, 0])
    CF = curve[:, 1]

    def fcurve(s):
        return float(np.interp(s, CS, CF))

    metal = p.roof != "slate"
    roof_tops, roof_sofs, roof_metal, roof_trim = [], [], [], []

    def roof_stage(tag, ea_, rt_, z0, H, th_, seam_sp, nv=14):
        """N curved facets from eaves apothem ea_ (z0) to rt_ (z0 + H), with soffit, seams and hip rolls."""
        def rho(s):
            return ea_ + (rt_ - ea_) * s

        for k in range(N):
            d, t, a = fdir(k)
            vax = Vector((-d[0] * (ea_ - rt_), -d[1] * (ea_ - rt_), H)).normalized()

            def top_fn(u, v, k=k):
                rh = rho(v)
                x, y = P2(k, (2 * u - 1) * rh * tanN, rh)
                return (x, y, z0 + H * fcurve(v))

            def sof_fn(u, v, k=k):
                rh = rho(v)
                x, y = P2(k, (1 - 2 * u) * rh * tanN, rh)
                return (x, y, z0 + H * fcurve(v) - th_)

            o = surface(f"{tag}_top{k}", top_fn, 2, nv)
            uv_planar(o, u_axis=(t[0], t[1], 0), v_axis=tuple(vax))
            roof_tops.append(o)
            o = surface(f"{tag}_sof{k}", sof_fn, 2, nv)
            uv_planar(o, u_axis=(t[0], t[1], 0), v_axis=tuple(vax))
            roof_sofs.append(o)
            if metal and seam_sp > 0:
                wmax = ea_ * tanN - 0.1
                offs = [0.0]
                j = 1
                while j * seam_sp < wmax:
                    offs += [j * seam_sp, -j * seam_sp]
                    j += 1
                for xo in offs:
                    s_max = min(1.0, (ea_ - (abs(xo) + 0.03) / tanN) / (ea_ - rt_))
                    if s_max < 0.06:
                        continue
                    ss = [s for s in np.linspace(0, 1, nv + 1) if s < s_max - 0.02] + [s_max]
                    path = []
                    for s in ss:
                        x, y = P2(k, xo, rho(s))
                        path.append((x, y, z0 + H * fcurve(s) - 0.002))
                    roof_metal.append(sweep(f"{tag}_seam", path, [(-0.012, 0.0), (0.012, 0.0), (0.015, 0.026), (0.0, 0.036), (-0.015, 0.026)],
                                            smooth_path=False))
            # hip roll + rib under the soffit
            hip, rib = [], []
            ang = a + half
            for s in np.linspace(0, 1, nv + 1):
                rc_ = rho(s) / cosN
                hip.append((rc_ * cos(ang), rc_ * sin(ang), z0 + H * fcurve(s) + 0.004))
                if s > 0.02:
                    rib.append(((rc_ - 0.03) * cos(ang), (rc_ - 0.03) * sin(ang), z0 + H * fcurve(s) - th_ - 0.02))
            roof_metal.append(sweep(f"{tag}_hip", hip, 0.032 if ea_ > 1.0 else 0.022, sides=8, smooth_path=False))
            if th_ > 0.06:
                o = sweep(f"{tag}_rib", rib, _rect(0.05, 0.075), smooth_path=False)
                roof_trim.append(o)
            # eaves: drip roll (covering), fascia board (trim)
            roof_metal.append(fblock(f"{tag}_drip", k, -1, 1, ea_ - 0.035, ea_ + 0.012, z0 - 0.028, 0.032, bevel=0.005))
            fa = fblock(f"{tag}_fascia", k, -1, 1, ea_ - 0.05, ea_ - 0.012, z0 - th_ - 0.05, th_ + 0.03, bevel=0.003)
            uv_grain(fa, (t[0], t[1], 0))
            roof_trim.append(fa)
        return lambda rr_: z0 + H * fcurve(clamp((ea_ - rr_) / (ea_ - rt_), 0, 1))

    th = 0.09
    Hr_full = p.roof_rise * ea
    cup = bool(p.cupola)
    rcu = max(0.42, 0.2 * ea)                                  # cupola drum apothem
    rt = rcu if cup else 0.09
    Hr = Hr_full * (0.8 if cup else 1.0)
    sb = (ea - ca) / (ea - rt)
    z_e = zb + 0.09 + th - Hr * fcurve(sb)
    seam_sp = clamp(ea * tanN * 2 / 4.6, 0.36, 0.5)
    zroof = roof_stage("main", ea, rt, z_e, Hr, th, seam_sp)
    z_top = z_e + Hr
    # wall plate between the beam and the soffit
    for k in range(N):
        d, t, a = fdir(k)
        wp = fblock("wall_plate", k, -1, 1, ca - 0.045, ca + 0.045, zb, 0.15)
        uv_grain(wp, (t[0], t[1], 0))
        roof_trim.append(wp)

    # scalloped eaves board
    rho_s = ea - 0.03
    wsc = 2 * rho_s * tanN - 0.03
    nsc = max(3, int(round(wsc / 0.14)))
    psc = wsc / nsc
    outl = [(-wsc / 2, 0.0), (-wsc / 2, -0.04)]
    sholes = []
    for i in range(nsc):
        cx = -wsc / 2 + (i + 0.5) * psc
        for q in np.linspace(180, 360, 6)[1:-1]:
            outl.append((cx + (psc / 2 - 0.004) * cos(radians(q)), -0.04 + (psc / 2 - 0.004) * 0.9 * sin(radians(q))))
        outl.append((cx + psc / 2, -0.04))
        sholes.append(circle_pts(0.014, 6, center=(cx, -0.055)))
    outl.append((wsc / 2, 0.0))
    sc = shape_extrude("scallop", outl, sholes, depth=0.016, loc=(0, 0.008, 0), rot=(90, 0, 0))
    apply_transform(sc)
    scal = instances_to_mesh("scallops", sc, [face_m(k, rho_s, z_e - th - 0.045) for k in range(N)], keep_base=False)
    roof_trim.append(scal)

    # hip-end knobs at the eaves
    knob = lathe("knob", [(0.0, -0.05), (0.02, -0.04), (0.03, -0.015), (0.02, 0.0), (0.03, 0.012), (0.05, 0.04), (0.052, 0.06), (0.04, 0.09),
                          (0.016, 0.11), (0.012, 0.135), (0.0, 0.16)], segments=12)
    km = []
    for k in range(N):
        ang = A0 + (k + 0.5) * dA
        km.append(trs(((ea / cosN + 0.005) * cos(ang), (ea / cosN + 0.005) * sin(ang), z_e + 0.01)))
    roof_metal.append(instances_to_mesh("knobs", knob, km, keep_base=False))

    # cupola: louvred drum and its own small roof
    z_boss = z_top - th                                        # underside level where the lantern hangs from
    fin_r = rt
    if cup:
        hdm = clamp(0.2 * Hr_full, 0.3, 0.5)
        roof_metal.append(lathe("cup_flash", [(rcu / cosN + 0.07, z_top - 0.06), (rcu / cosN + 0.02, z_top + 0.015), (rcu / cosN - 0.03, z_top + 0.03)],
                                segments=N, rot=(0, 0, degrees(A0 + half))))
        for k in range(N):
            d, t, a = fdir(k)
            ir.append(fblock("cup_sill", k, -1, 1, rcu - 0.07, rcu + 0.015, z_top, 0.04, bevel=0.004))
            ir.append(fblock("cup_head", k, -1, 1, rcu - 0.07, rcu + 0.015, z_top + hdm - 0.05, 0.05, bevel=0.004))
            ir.append(fblock("cup_core", k, -1, 1, 0.02, rcu - 0.06, z_top, hdm))
            nl = max(3, int((hdm - 0.09) / 0.055))
            for i in range(nl):
                zl = z_top + 0.04 + (i + 0.5) * (hdm - 0.09) / nl
                xq, yq = P2(k, 0, rcu - 0.03)
                ir.append(box("louvre", (2 * rcu * tanN - 0.07, 0.07, 0.012), loc=(xq, yq, zl), rot=(-32, 0, degrees(a) + 90)))
            ang = a + half
            ir.append(box("cup_post", (0.055, 0.055, hdm), loc=((rcu - 0.01) / cosN * cos(ang), (rcu - 0.01) / cosN * sin(ang), z_top + hdm / 2),
                          rot=(0, 0, degrees(ang)), bevel=0.004))
        ea2 = rcu + 0.2
        H2 = Hr_full * 0.42
        z_e2 = z_top + hdm + 0.03
        zroof = roof_stage("cup", ea2, 0.06, z_e2, H2, 0.04, 0.0, nv=8)
        # boarded ceiling closing the main roof under the drum
        cl = extrude_profile("cup_ceiling", [((rcu + 0.03) / cosN * cos(A0 + (k + 0.5) * dA), (rcu + 0.03) / cosN * sin(A0 + (k + 0.5) * dA))
                                             for k in range(N)], 0.03, loc=(0, 0, z_top - th - 0.03))
        uv_grain(cl, "X")
        roof_trim.append(cl)
        z_boss = z_top - th - 0.03
        z_top = z_e2 + H2
        fin_r = 0.06

    # finial: flashing skirt (roof metal), spindle, gilt ball, spike
    r_sk = fin_r / cosN + (0.14 if not cup else 0.1)
    roof_metal.append(lathe("apex_flash", [(r_sk + 0.012, zroof(r_sk * cosN) - 0.03), (r_sk + 0.016, zroof(r_sk * cosN) - 0.005),
                                           (r_sk * 0.62, z_top + 0.0), (0.085, z_top + 0.11), (0.07, z_top + 0.2), (0.09, z_top + 0.215),
                                           (0.09, z_top + 0.235), (0.05, z_top + 0.25), (0.0, z_top + 0.25)], segments=24))
    fs = 1.0 if R > 2.6 else 0.8
    zf = z_top + 0.245
    ir.append(lathe("finial", [(0.045, zf), (0.03, zf + 0.03 * fs), (0.024, zf + 0.2 * fs), (0.05, zf + 0.22 * fs), (0.055, zf + 0.24 * fs),
                               (0.03, zf + 0.26 * fs), (0.026, zf + 0.3 * fs)], segments=16))
    ball = lathe("finial_ball", [(0.0, zf + 0.29 * fs)] + [(0.105 * fs * sin(q), zf + (0.395 - 0.105 * cos(q)) * fs) for q in np.linspace(0.25, pi - 0.2, 10)]
                 + [(0.03, zf + 0.53 * fs), (0.04, zf + 0.55 * fs), (0.022, zf + 0.6 * fs), (0.016, zf + 0.8 * fs), (0.03, zf + 0.84 * fs),
                    (0.012, zf + 0.9 * fs), (0.0, zf + 1.12 * fs)], segments=20)
    assign(ball, gilt)
    # coronet of cresting round the apex
    r_cr = r_sk + 0.07
    z_cr = zroof(r_cr * cosN) + 0.02
    ir.append(ring_band("coronet_band", r_cr, 0.035, 0.006, loc=(0, 0, z_cr + 0.012), segments=32))
    spike = shape_extrude("spike", [(-0.008, 0.0), (0.008, 0.0), (0.008, 0.09), (0.03, 0.11), (0.035, 0.135), (0.021, 0.152), (0.012, 0.14), (0.01, 0.16),
                                    (0.022, 0.19), (0.0, 0.25), (-0.022, 0.19), (-0.01, 0.16), (-0.012, 0.14), (-0.021, 0.152), (-0.035, 0.135),
                                    (-0.03, 0.11), (-0.008, 0.09)], depth=0.008, loc=(0, 0.004, 0), rot=(90, 0, 0), bevel=0.001)
    apply_transform(spike)
    ncs = 2 * N
    ir.append(instances_to_mesh("coronet", spike, [trs((r_cr * cos(q), r_cr * sin(q), z_cr), (-12, 0, degrees(q) + 90), fs)
                                                   for q in np.linspace(0, 2 * pi, ncs, endpoint=False)], keep_base=False))

    # ---- roof materials and weathering
    for o in roof_tops:
        assign(o, roofm)
    for o in roof_metal:
        assign(o, flash)
    for o in roof_sofs:
        assign(o, soff)
    for o in roof_trim:
        assign(o, trim)
    rtop = join(roof_tops, "roof_top")
    densify(rtop, 0.35)
    if metal:
        displace_noise(rtop, 0.002, 0.6, seed=p.seed + 11)
    rmet = join(roof_metal, "roof_metal")
    rsof = join(roof_sofs, "soffit")
    densify(rsof, 0.4)
    rtrim = join(roof_trim, "roof_trim")

    def streaks(co, nr):
        a = np.arctan2(co[:, 1], co[:, 0])
        return 0.5 + 0.5 * np.sin(a * 37.0 + 3.0 * np.sin(a * 11.0 + p.seed))

    patina = {"copper": 0.45 + 0.55 * W, "zinc": 0.25 * W, "lead": 0.0, "slate": 0.3 + 0.6 * W}[p.roof]
    rmask = mask_mul(noise_mask(0.7, seed=p.seed + 13, lo=0.25, hi=0.8), lambda co, nr: 0.55 + 0.45 * streaks(co, nr), patina)
    for o in (rtop, rmet):
        paint(o, special=rmask, grime=mask_mul(streaks, noise_mask(0.8, seed=p.seed + 14, lo=0.3, hi=0.9), 0.07 + 0.2 * W))

    def soot(co, nr):
        return np.exp(-(co[:, 0] ** 2 + co[:, 1] ** 2) / (0.3 * ea) ** 2)

    paint(rsof, grime=mask_mul(soot, 0.3 + 0.5 * W))
    paint(rtrim, wear=mask_mul(noise_mask(0.25, seed=p.seed + 15, lo=0.5, hi=0.85), 0.3 + 0.7 * W), special=0.3 * W,
          grime=mask_mul(soot, 0.5))

    # ------------------------------------------------------------------ central boss and pendant lantern
    z_lb = zd + max(2.25, Hc - 0.72)                           # underside of the lantern body
    hl = 0.66
    z_lt = z_lb + hl
    boss_p = [(0.0, z_boss - 0.2), (0.025, z_boss - 0.19), (0.045, z_boss - 0.15), (0.03, z_boss - 0.11), (0.07, z_boss - 0.08),
              (0.13, z_boss - 0.05), (0.15, z_boss - 0.01), (0.2, z_boss + 0.03), (0.22, z_boss + 0.1 if not cup else z_boss + 0.03)]
    cb = lathe("ceiling_boss", boss_p, segments=20)
    assign(cb, trim)
    paint(cb, grime=0.4)
    lp = []
    lp.append(cylinder("lantern_rod", 0.011, (z_boss - 0.19) - (z_lt + 0.2), loc=(0, 0, ((z_boss - 0.19) + (z_lt + 0.2)) / 2), verts=8))
    zk = (z_boss - 0.19 + z_lt + 0.2) / 2
    lp.append(lathe("rod_knop", [(0.011, zk - 0.05), (0.02, zk - 0.035), (0.034, zk), (0.02, zk + 0.035), (0.011, zk + 0.05)], segments=12))
    rt_l, rb_l = 0.24, 0.16
    lp.append(lathe("lantern_cap", [(0.0, z_lt + 0.22), (0.022, z_lt + 0.21), (0.03, z_lt + 0.17), (0.07, z_lt + 0.15), (0.075, z_lt + 0.12),
                                    (0.05, z_lt + 0.105), (0.1, z_lt + 0.07), (rt_l + 0.01, z_lt + 0.025), (rt_l + 0.035, z_lt - 0.004),
                                    (rt_l + 0.035, z_lt - 0.016), (rt_l - 0.01, z_lt - 0.016), (rt_l - 0.012, z_lt + 0.0), (0.0, z_lt + 0.03)],
                    segments=6, smooth_angle=25))
    lp.append(lathe("lantern_foot", [(rb_l - 0.012, z_lb + 0.012), (rb_l + 0.012, z_lb + 0.012), (rb_l + 0.016, z_lb), (rb_l + 0.006, z_lb - 0.014),
                                     (0.07, z_lb - 0.04), (0.035, z_lb - 0.075), (0.026, z_lb - 0.1), (0.04, z_lb - 0.118), (0.026, z_lb - 0.14),
                                     (0.0, z_lb - 0.17)], segments=6, smooth_angle=25))
    gv, gf = [], []
    for q in range(6):
        aq = radians(q * 60)
        lp.append(sweep("lantern_bar", [(rb_l * cos(aq), rb_l * sin(aq), z_lb), (rt_l * cos(aq), rt_l * sin(aq), z_lt)], _rect(0.014, 0.014),
                        smooth_path=False, up=(cos(aq), sin(aq), 0)))
        gv += [((rb_l - 0.006) * cos(aq), (rb_l - 0.006) * sin(aq), z_lb + 0.01), ((rt_l - 0.006) * cos(aq), (rt_l - 0.006) * sin(aq), z_lt - 0.01)]
        gf.append((2 * q, (2 * q + 2) % 12, (2 * q + 3) % 12, 2 * q + 1))
    lg = mesh_object("lantern_glass", gv, gf)
    recalc_normals(lg)
    assign(lg, glass)
    lp.append(lathe("burner", [(0.0, z_lb - 0.01), (0.03, z_lb), (0.03, z_lb + 0.03), (0.014, z_lb + 0.05), (0.014, z_lb + 0.19), (0.03, z_lb + 0.2),
                               (0.03, z_lb + 0.225), (0.0, z_lb + 0.225)], segments=10))
    for o in lp:
        assign(o, gilt if p.gilding else iron)
    z_l = z_lb + 0.3
    mantle = lathe("mantle", [(0.0, z_l - 0.075), (0.02, z_l - 0.07), (0.034, z_l - 0.03), (0.037, z_l + 0.015), (0.027, z_l + 0.06), (0.01, z_l + 0.085),
                              (0.0, z_l + 0.09)], segments=12)
    assign(mantle, glow)
    if lit:
        light("lantern", (0, 0, z_l), color="#ffbf6a", intensity=float(p.intensity), range=clamp(2.2 * p.diameter, 9, 16), flicker=0.1,
              shadow=True, shadow_near=0.42)

    # ------------------------------------------------------------------ festoon along the eaves
    nbul = int(p.festoon_bulbs)
    if nbul > 0:
        rf = rng.fork("festoon")
        z_h = z_e - th - 0.03
        wires, bm2, hm2 = [], [], []
        nl = 0
        for k in range(N):
            d, t, a = fdir(k)
            wq = (ea + 0.0) * tanN - 0.03
            x0_, y0_ = P2(k, -wq, ea + 0.004)
            x1_, y1_ = P2(k, wq, ea + 0.004)
            sag = 0.09 * bay ** 0.5 + rf.u(-0.02, 0.03)
            pts = np.asarray(catenary((x0_, y0_, z_h), (x1_, y1_, z_h), sag=sag, n=nbul * 4))
            wires.append(sweep("festoon_wire", [tuple(q) for q in pts], 0.004, sides=5, smooth_path=False))
            for i in range(nbul):
                q = pts[min(len(pts) - 1, int(round((i + 0.5) / nbul * (len(pts) - 1))))]
                tilt = (rf.u(-8, 8), rf.u(-8, 8), rf.u(0, 360))
                hm2.append(trs((q[0], q[1], q[2] - 0.004), tilt))
                if not rf.chance(0.03 + 0.07 * W):
                    bm2.append(trs((q[0], q[1], q[2] - 0.004), tilt))
            ir.append(box("festoon_hook", (0.012, 0.03, 0.012), loc=(*P2(k, wq + 0.02, ea), z_h + 0.004), rot=(0, 0, degrees(a) + 90)))
            if lit and (k % 2 == 0) and nl < 5:
                xm_, ym_ = P2(k, 0, ea + 0.15)                 # just clear of the bulbs so the eaves board is not burnt out
                light(f"festoon{k}", (xm_, ym_, z_h - sag - 0.2), color="#ffc98a", intensity=1.6 * nbul, range=5.5, flicker=0.03, shadow_near=0.2)
                nl += 1
        holder = cylinder("holder", 0.014, 0.034, loc=(0, 0, -0.02), verts=8)
        apply_transform(holder)
        bulb = sphere("bulb", 0.03, loc=(0, 0, -0.066), segments=8, rings=5, scale=(1, 1, 1.25))
        apply_transform(bulb)
        hs = instances_to_mesh("holders", holder, hm2, keep_base=False)
        assign(hs, gilt if p.gilding else iron)
        ir += wires
        if bm2:
            bs = instances_to_mesh("bulbs", bulb, bm2, keep_base=False)
            assign(bs, glow)
        else:
            delete(bulb)

    # ------------------------------------------------------------------ ironwork paint and ageing
    for o in ir:
        assign(o, iron)
    iron_all = join(ir, "ironwork")

    def base_rust(co, nr):
        z = co[:, 2] - zd
        return np.clip(1 - z / 0.4, 0, 1) * (z > -0.02) + np.clip(1 - np.abs(z - 0.115) / 0.06, 0, 1) * 0.6

    def hand_polish(co, nr):
        z = co[:, 2] - zd
        return np.clip(1 - np.abs(z - zr - 0.01) / 0.03, 0, 1) * np.clip(nr[:, 2], 0, 1)

    paint(iron_all, special=mask_mul(base_rust, noise_mask(0.08, seed=p.seed + 2, lo=0.3, hi=0.75), 0.3 + 0.9 * W),
          grime=mask_mul(lambda co, nr: np.clip(1 - (co[:, 2] - zd) / 0.5, 0, 1), 0.4 + 0.4 * W),
          wear=mask_mul(hand_polish, 0.4 + 0.5 * W))

    socket("stage", "attach", (0, 0, zd), role="prop_top")
    for k in (1, N - 1):
        ang = A0 + (k + 0.5) * dA
        socket(f"banner{k}", "attach", ((Rc + 0.07) * cos(ang), (Rc + 0.07) * sin(ang), zd + Hc * 0.78), rot=(0, 0, degrees(ang) + 90), role="banner")
