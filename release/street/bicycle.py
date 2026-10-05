"""Vintage roadster bicycle propped on its side stand. Lugged steel diamond (or loop / step-through)
frame, 36-spoke cross-laced wheels on Westwood rims, valanced mudguards on wire stays with a white
tail patch, hockey-stick chain guard or full oil-bath chain case, cottered cranks with rubber block
pedals, a sprung leather saddle with tool roll, North Road bars with rod-brake levers and stirrups,
a strapped-on wicker basket with parcels and a small brass oil lamp. The front end is steered about
the real head-tube axis and the whole machine leans onto the stand.
"""
from math import sin, cos, radians, degrees, pi, atan2, hypot, sqrt

from mathutils import Matrix, Vector

from tk import *

META = {"name": "Old bicycle", "class": "prop", "mount": "ground", "tri_budget": 20500,
        "focus": [0.4, -0.05, 0.74]}
PARAMS = {
    "paint": {"default": "#2f6d4d", "desc": "frame paint colour"},
    "basket": {"default": True, "desc": "wicker basket strapped to the handlebars (lamp moves to the fork)"},
    "lean": {"default": 9.0, "min": 0.0, "max": 16.0, "desc": "lean onto the side stand (deg)"},
    "wheel_size": {"default": 0.70, "min": 0.56, "max": 0.74, "desc": "outer tyre diameter (m); 0.70 = 28 in roadster"},
    "frame": {"default": "diamond", "choices": ["diamond", "step_through"], "desc": "gents diamond or ladies loop frame"},
    "steer": {"default": -24.0, "min": -40.0, "max": 40.0, "desc": "handlebar turn (deg, + = towards the stand side)"},
    "chain_guard": {"default": "hockey", "choices": ["hockey", "case"], "desc": "open hockey-stick guard or full chain case"},
    "rack": {"default": False, "desc": "rear luggage carrier"},
    "saddlebag": {"default": True, "desc": "leather tool roll behind the saddle"},
    "cargo": {"default": True, "desc": "parcel and paper rolls in the basket"},
    "lamp_lit": {"default": True, "desc": "oil lamp burning"},
    "lamp_power": {"default": 5.0, "min": 1.0, "max": 15.0, "desc": "oil lamp intensity (cd)"},
    "tyres": {"default": "#1c1b1a", "desc": "tyre rubber colour (black, or cream #cfc4a6)"},
    "saddle": {"default": "#6e3f20", "desc": "leather colour of saddle, grips and straps"},
    "condition": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "0 = cared for, 1 = rusty and grimy"},
}
VARIANTS = {
    "ladies_burgundy": {"frame": "step_through", "paint": "#611519", "chain_guard": "case", "rack": True, "wheel_size": 0.66,
                        "lean": 6.0, "steer": -12.0, "tyres": "#a8997a", "lamp_power": 3.5, "saddle": "#3a2416", "saddlebag": False, "seed": 4},
    "gents_black": {"paint": "#15181b", "basket": False, "rack": True, "lean": 12.0, "steer": -32.0, "wheel_size": 0.72,
                    "saddle": "#2c1c12", "condition": 0.25, "seed": 11},
    "errand_cream": {"paint": "#cdbf9c", "frame": "step_through", "wheel_size": 0.6, "rack": True, "lean": 10.0, "steer": 12.0,
                     "saddle": "#7a4a2a", "chain_guard": "case", "condition": 0.45, "seed": 23},
    "rusty_blue": {"paint": "#2b4d6e", "basket": False, "lamp_lit": False, "saddlebag": False, "lean": 15.0, "steer": 34.0,
                   "condition": 1.0, "wheel_size": 0.68, "seed": 31},
}

X = Vector((1.0, 0.0, 0.0))
Y = Vector((0.0, 1.0, 0.0))
Z = Vector((0.0, 0.0, 1.0))


# ------------------------------------------------------------------------------------------ helpers
def _rot_to(d):
    e = Vector(d).normalized().to_track_quat('Z', 'Y').to_euler()
    return (degrees(e.x), degrees(e.y), degrees(e.z))


def _tube(name, a, b, r, sides=10, r2=None, cap=True):
    a = Vector(a)
    b = Vector(b)
    d = b - a
    o = cylinder(name, r, d.length, loc=tuple((a + b) * 0.5), rot=_rot_to(d), verts=sides, cap=cap, radius2=r2)
    smooth(o, 62)
    return o


def _rod_m(a, b, r):
    a = Vector(a)
    b = Vector(b)
    d = b - a
    q = d.normalized().to_track_quat('Z', 'Y')
    return Matrix.Translation((a + b) * 0.5) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, d.length, 1.0))


def _rods(name, segs, r, sides=5, cap=False):
    base = cylinder(name + "_b", 1.0, 1.0, verts=sides, cap=cap)
    o = instances_to_mesh(name, base, [_rod_m(a, b, r) for a, b in segs], uv_jitter=False, keep_base=False)
    smooth(o, 80)
    return o


def _bake(objs, M):
    for o in objs:
        realize(o)
        apply_transform(o)
        o.data.transform(M)
        o.data.update()


def _loft(name, rings, cap_first=False, cap_last=False):
    n = len(rings[0])
    verts = [tuple(v) for r in rings for v in r]
    faces = []
    for i in range(len(rings) - 1):
        for k in range(n):
            a = i * n + k
            b = i * n + (k + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap_first:
        faces.append(tuple(range(n - 1, -1, -1)))
    if cap_last:
        faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _tangent(C1, r1, C2, r2):
    """Outer tangent normals (upper, lower) of two circles in the XZ plane."""
    ex, ez = C2[0] - C1[0], C2[1] - C1[1]
    d = hypot(ex, ez)
    ex, ez = ex / d, ez / d
    px, pz = -ez, ex
    if pz < 0:
        px, pz = -px, -pz
    sg = (r1 - r2) / d
    cg = sqrt(max(1e-9, 1.0 - sg * sg))
    return (ex * sg + px * cg, ez * sg + pz * cg), (ex * sg - px * cg, ez * sg - pz * cg)


def _belt(C1, r1, C2, r2, step):
    """Points (x, z, tangent angle) around two pulleys: C1 front, C2 rear."""
    nu, nd = _tangent(C1, r1, C2, r2)
    au, ad = atan2(nu[1], nu[0]), atan2(nd[1], nd[0])
    while ad < au:
        ad += 2 * pi
    out = []

    def line(p0, p1):
        L = hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(1, int(round(L / step)))
        ta = atan2(p1[1] - p0[1], p1[0] - p0[0])
        for i in range(n):
            t = i / n
            out.append((p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t, ta))

    def arc(C, r, a0, a1):
        n = max(1, int(round(r * (a1 - a0) / step)))
        for i in range(n):
            a = a0 + (a1 - a0) * i / n
            out.append((C[0] + r * cos(a), C[1] + r * sin(a), a + pi / 2))

    line((C1[0] + r1 * nu[0], C1[1] + r1 * nu[1]), (C2[0] + r2 * nu[0], C2[1] + r2 * nu[1]))
    arc(C2, r2, au, ad)
    line((C2[0] + r2 * nd[0], C2[1] + r2 * nd[1]), (C1[0] + r1 * nd[0], C1[1] + r1 * nd[1]))
    arc(C1, r1, ad, au + 2 * pi)
    return out


def _wheel(pre, A, R, rt, rear, phase, M):
    """Wheel centred on A with its axle along Y. Returns [(obj, material)]."""
    out = []
    Rc = R - rt
    Rr = R - 2 * rt + 0.004
    prof = []
    for k in range(11):
        ph = radians(-138 + 276 * k / 10)
        rr = rt + (0.0015 if abs(ph) < radians(34) else 0.0) - (0.0012 if k == 5 else 0.0)
        prof.append((Rc + rr * cos(ph), rr * sin(ph)))
    out.append((lathe(pre + "tyre", prof, segments=40, loc=tuple(A), rot=(90, 0, 0), smooth_angle=60), M["rubber"]))
    rim = [(Rr + 0.008, 0.0145), (Rr + 0.001, 0.0165), (Rr - 0.006, 0.0135), (Rr - 0.0105, 0.006), (Rr - 0.0105, -0.006),
           (Rr - 0.006, -0.0135), (Rr + 0.001, -0.0165), (Rr + 0.008, -0.0145)]
    out.append((lathe(pre + "rim", rim, segments=40, loc=tuple(A), rot=(90, 0, 0), smooth_angle=50), M["nickel"]))
    if rear:
        yf, rf, yo = 0.029, 0.031, 0.0545
        hub = [(0.0, -0.072), (0.0045, -0.072), (0.0045, -0.066), (0.0105, -0.066), (0.0105, -0.059), (0.008, -0.059),
               (0.008, -0.036), (0.016, -0.034), (rf + 0.003, -0.031), (rf + 0.003, -0.027), (0.027, -0.024), (0.0285, 0.0),
               (0.027, 0.024), (rf + 0.003, 0.027), (rf + 0.003, 0.031), (0.016, 0.034), (0.013, 0.05), (0.008, 0.052),
               (0.008, 0.059), (0.0105, 0.059), (0.0105, 0.066), (0.0045, 0.066), (0.0045, 0.072), (0.0, 0.072)]
    else:
        yf, rf, yo = 0.031, 0.021, 0.048
        hub = [(0.0, -0.066), (0.0045, -0.066), (0.0045, -0.060), (0.010, -0.060), (0.010, -0.0535), (0.0075, -0.0535),
               (0.0075, -0.037), (0.012, -0.035), (rf + 0.003, -0.033), (rf + 0.003, -0.029), (0.0125, -0.026), (0.0105, 0.0),
               (0.0125, 0.026), (rf + 0.003, 0.029), (rf + 0.003, 0.033), (0.012, 0.035), (0.0075, 0.037), (0.0075, 0.0535),
               (0.010, 0.0535), (0.010, 0.060), (0.0045, 0.060), (0.0045, 0.066), (0.0, 0.066)]
    out.append((lathe(pre + "hub", hub, segments=12, loc=tuple(A), rot=(90, 0, 0), smooth_angle=50), M["nickel"]))
    r_in = Rr - 0.0095
    segs = []
    for i in range(36):
        side = 1 if i % 2 == 0 else -1
        k = i // 2
        ah = 20.0 * k + (0.0 if side > 0 else 10.0) + phase
        ar = ah + (62.0 if k % 2 == 0 else -62.0)
        # snap the rim end onto the 10 degree drilling
        ar = round((ar - phase) / 10.0) * 10.0 + phase
        hx, hz = rf * cos(radians(ah)), rf * sin(radians(ah))
        rx, rz = r_in * cos(radians(ar)), r_in * sin(radians(ar))
        yh = side * (yf + (0.002 if k % 2 == 0 else -0.002))
        segs.append((A + Vector((hx, yh, hz)), A + Vector((rx, side * 0.003, rz))))
    out.append((_rods(pre + "spokes", segs, 0.0013, sides=4), M["nickel"]))
    va = radians(phase + 5.0)
    vd = Vector((cos(va), 0, sin(va)))
    out.append((_tube(pre + "valve", A + vd * (r_in + 0.004), A + vd * (r_in - 0.03), 0.003, 6), M["brass"]))
    return out


# -------------------------------------------------------------------------------------------- build
def build(p, rng):
    R = p.wheel_size * 0.5
    rt = 0.019
    cond = float(p.condition)
    seed = int(p.seed)
    ha = radians(68.0)
    u = Vector((-cos(ha), 0.0, sin(ha)))            # up the head / seat tube
    fw = Vector((sin(ha), 0.0, cos(ha)))            # perpendicular to it, forwards
    wb = 1.12 + 1.1 * (R - 0.35)
    Ar = Vector((-wb / 2, 0, R))
    Af = Vector((wb / 2, 0, R))
    BB = Vector((Ar.x + sqrt((R + 0.12) ** 2 - 0.065 ** 2), 0, R - 0.065))
    ST = BB + u * 0.60
    Q = Af - fw * 0.07                               # foot of the steering axis
    s_cr = R + 0.05
    CR = Q + u * s_cr
    HB = Q + u * (s_cr + 0.024)
    HT = Q + u * ((ST.z + 0.01 - Q.z) / u.z)
    Rg = R + 0.02                                    # mudguard radius

    M = {
        "frame": mat("iron_painted", p.paint, wear=0.4),
        "nickel": mat("brass", "#e6d9c0", rough=0.8),
        "brass": mat("brass"),
        "rubber": mat("plain", p.tyres),
        "leather": mat("leather", p.saddle),
        "iron": mat("iron_wrought"),
    }
    rear, front = [], []

    def Rr_(o, m):
        assign(o, M[m] if isinstance(m, str) else m)
        rear.append(o)
        return o

    def Fr_(o, m):
        assign(o, M[m] if isinstance(m, str) else m)
        front.append(o)
        return o

    # ================================================================ frame (lugged steel tubes)
    Rr_(_tube("seat_tube", BB, ST, 0.0143, 12), "frame")
    DT1 = HB + u * 0.028
    Rr_(_tube("down_tube", BB, DT1, 0.0143, 12), "frame")
    Rr_(_tube("head_tube", HB, HT, 0.0175, 14), "frame")
    T1 = HT - u * 0.03
    ddt = (BB - DT1).normalized()
    if p.frame == "diamond":
        T0 = ST - u * 0.03
        Rr_(_tube("top_tube", T0, T1, 0.0127, 12), "frame")
        dtt = (T0 - T1).normalized()
        Rr_(_tube("lug_tt_s", T0 - dtt * 0.014, T0 - dtt * 0.05, 0.0152, 12), "frame")
        # frame pump on pegs under the top tube
        pz = Vector((0, 0, -0.031))
        pa, pb = T0 - dtt * 0.09 + pz, T0 - dtt * 0.47 + pz
        Rr_(_tube("pump", pa, pb, 0.0095, 10), "nickel")
        Rr_(_tube("pump_handle", pb, pb - dtt * 0.035, 0.011, 10), "iron")
        Rr_(_rods("pump_pegs", [(pa + dtt * 0.012, pa + dtt * 0.012 - pz), (pb - dtt * 0.045, pb - dtt * 0.045 - pz)], 0.004, 6), "frame")
    else:
        S = BB + u * 0.24
        pts = bezier_path(T1, T1 + ddt * 0.32, S + fw * 0.24 + u * 0.035, S, n=16)
        Rr_(sweep("loop_tube", pts, 0.0127, sides=10, smooth_path=False), "frame")
        dtt = ddt
        Rr_(_tube("lug_loop_s", S + fw * 0.012, S + fw * 0.045 + u * 0.004, 0.0152, 12), "frame")
    # lugs: sockets at every joint, bottom bracket shell, seat pinch bolt
    Rr_(_tube("lug_h0", HB, HB + u * 0.052, 0.0198, 14), "frame")
    Rr_(_tube("lug_h1", HT - u * 0.052, HT, 0.0198, 14), "frame")
    Rr_(_tube("lug_dt_h", DT1 + ddt * 0.016, DT1 + ddt * 0.055, 0.0166, 12), "frame")
    Rr_(_tube("lug_tt_h", T1 + dtt * 0.016, T1 + dtt * 0.052, 0.0152, 12), "frame")
    Rr_(_tube("lug_seat", ST - u * 0.055, ST + u * 0.002, 0.0168, 12), "frame")
    Rr_(_tube("seat_bolt", ST - u * 0.012 - fw * 0.019 - Y * 0.017, ST - u * 0.012 - fw * 0.019 + Y * 0.017, 0.0045, 6), "nickel")
    Rr_(_tube("bb_shell", BB - Y * 0.036, BB + Y * 0.036, 0.0215, 14), "frame")
    Rr_(_tube("lug_bb_s", BB + u * 0.016, BB + u * 0.05, 0.0166, 12), "frame")
    Rr_(_tube("lug_bb_d", BB - ddt * 0.016, BB - ddt * 0.05, 0.0166, 12), "frame")
    Rr_(_tube("headset0", HB - u * 0.011, HB, 0.0205, 14), "nickel")
    Rr_(_tube("headset1", HT, HT + u * 0.011, 0.0205, 14), "nickel")
    # brass head badge wrapped on the head tube
    hm = (HB + HT) * 0.5

    def badge(a, b):
        w = 0.62 * (1.0 if b > 0.35 else 0.35 + 0.65 * b / 0.35) * (1.0 if b < 0.8 else 1.0 - 0.5 * (b - 0.8) / 0.2)
        ang = (a - 0.5) * 2.0 * w
        q = hm + u * ((b - 0.5) * 0.05) + fw * (0.0186 * cos(ang)) + Y * (0.0186 * sin(ang))
        return (q.x, q.y, q.z)

    Rr_(surface("head_badge", badge, 6, 5, thickness=0.0012), "brass")

    cs_dir = (Ar - BB).normalized()
    for s in (-1, 1):
        ax = Ar + Y * (0.057 * s)
        scp = ST - u * 0.022 + Y * (0.019 * s) - fw * 0.005
        mid = ax.lerp(scp, 0.78)
        mid.y = 0.036 * s
        Rr_(sweep("seat_stay", [ax + Vector((0.004, 0, 0.012)), mid, scp], 0.0072, sides=8, step=0.06,
                  scale_fn=lambda t: 0.8 + 0.2 * t), "frame")
        Rr_(sweep("chain_stay", [BB + Vector((-0.014, 0.033 * s, 0.0)), ax + Vector((0.016, 0, 0))], 0.0098, sides=8,
                  smooth_path=False, scale_fn=lambda t: 1.0 - 0.3 * t), "frame")
        Rr_(box("dropout", (0.05, 0.004, 0.032), loc=tuple(ax + Vector((0.006, 0, 0.005))), bevel=0.0012), "frame")
    mid0 = Ar.lerp(ST - u * 0.022, 0.78)
    Rr_(_tube("stay_bridge", mid0 - Y * 0.036, mid0 + Y * 0.036, 0.0055, 8), "frame")
    cbp = BB + cs_dir * 0.07
    Rr_(_tube("cs_bridge", cbp - Y * 0.037, cbp + Y * 0.037, 0.006, 8), "frame")

    # ================================================================ fork, stem (steered group)
    tilt = -(90.0 - degrees(ha))
    Fr_(box("fork_crown", (0.034, 0.118, 0.022), loc=tuple(CR), rot=(0, tilt, 0), bevel=0.005), "frame")
    blade_pts = {}
    for s in (-1, 1):
        yb = Y * (0.0475 * s)
        pts = bezier_path(CR + yb, Q + u * (s_cr * 0.45) + yb, Q + u * 0.05 + yb, Af + Y * (0.0505 * s), n=12)
        blade_pts[s] = pts
        Fr_(sweep("fork_blade", pts, 0.0118, sides=8, smooth_path=False, scale_fn=lambda t: 1.0 - 0.42 * t), "frame")
        Fr_(box("fork_tip", (0.024, 0.004, 0.03), loc=tuple(Af + Y * (0.0505 * s) + Vector((-0.002, 0, 0.003))), bevel=0.0012), "frame")
    Fr_(_tube("locknut", HT + u * 0.011, HT + u * 0.021, 0.0165, 6), "nickel")
    stem_top = HT + u * 0.125
    Hc = stem_top + fw * 0.05                        # handlebar clamp centre
    Fr_(_tube("stem", HT + u * 0.015, stem_top + u * 0.006, 0.0108, 10), "nickel")
    Fr_(_tube("stem_ext", stem_top - fw * 0.004, Hc, 0.0105, 10), "nickel")
    Fr_(_tube("bar_clamp", Hc - Y * 0.02, Hc + Y * 0.02, 0.0158, 12), "nickel")
    Fr_(_tube("stem_bolt", stem_top + u * 0.006, stem_top + u * 0.013, 0.0065, 6), "nickel")

    # North Road handlebars, grips, rod-brake levers
    def bar_side(s, dz=0.0):
        return [Hc + Vector((x, y * s, z + dz)) for (x, y, z) in
                ((0.0, 0.06, 0.0), (0.02, 0.13, 0.004), (0.026, 0.19, 0.012), (-0.012, 0.236, 0.022), (-0.09, 0.256, 0.026),
                 (-0.20, 0.262, 0.018))]

    bar = list(reversed(bar_side(-1))) + [Hc] + bar_side(1)
    Fr_(sweep("handlebar", bar, 0.0108, sides=8, step=0.03), "nickel")
    for s in (-1, 1):
        e = bar_side(s)
        g0, g1 = e[-1], e[-1] + (e[-2] - e[-1]).normalized() * 0.105
        Fr_(_tube("grip", g0 - (g1 - g0).normalized() * 0.004, g1, 0.0158, 10), "leather")
        Fr_(_tube("grip_end", g1 - (g1 - g0).normalized() * 0.002, g1 + (g1 - g0).normalized() * 0.006, 0.0182, 10), "leather")
        lv = [Hc + Vector((x, y * s, z)) for (x, y, z) in
              ((-0.175, 0.262, -0.014), (-0.09, 0.255, -0.008), (-0.02, 0.233, -0.012), (0.014, 0.19, -0.022), (0.008, 0.12, -0.03),
               (0.0, 0.03, -0.034))]
        Fr_(sweep("brake_lever", lv, 0.0038, sides=5, step=0.035), "nickel")
        Fr_(_rods("lever_clips", [(Hc + Vector((0.022, 0.16 * s, 0.01)), Hc + Vector((0.012, 0.16 * s, -0.03))),
                                  (Hc + Vector((0.0, 0.045 * s, 0.0)), Hc + Vector((0.0, 0.045 * s, -0.036)))], 0.0035, 5), "nickel")
    # bell
    bl = Hc + Vector((0.021, 0.105, 0.0105))
    Fr_(lathe("bell", [(0.0, 0.0), (0.012, 0.0), (0.012, 0.004), (0.026, 0.004), (0.0265, 0.009), (0.023, 0.018), (0.013, 0.025),
                       (0.004, 0.027), (0.004, 0.031), (0.0, 0.032)], segments=14, loc=tuple(bl)), "brass")

    # front rod brake: pull rod down the head, stirrup round the tyre, blocks under the Westwood rim
    Rr0 = R - 2 * rt + 0.004
    S0 = Q + u * (s_cr + 0.045) + fw * 0.048
    Fr_(_tube("pull_rod_f", Hc + Vector((0.0, -0.022, -0.034)), S0 + Y * -0.022, 0.0035, 6), "nickel")
    Fr_(_tube("stirrup_top_f", S0 - Y * 0.036, S0 + Y * 0.036, 0.004, 6), "nickel")
    dpad = (S0 - Af).normalized()
    pad_c = Af + dpad * (Rr0 - 0.019)
    for s in (-1, 1):
        pth = [S0 + Y * (0.034 * s), pad_c + Y * (0.034 * s) - dpad * 0.004, pad_c + Y * (0.013 * s) - dpad * 0.004]
        Fr_(sweep("stirrup_f", pth, 0.0036, sides=5, smooth_path=False), "nickel")
        Fr_(box("brake_block_f", (0.012, 0.011, 0.044), loc=tuple(pad_c + Y * (0.011 * s) + dpad * 0.002), rot=_rot_to(fw.cross(Y) * 0 + Vector((-dpad.z, 0, dpad.x))),
                bevel=0.002), "rubber")
        # guide clip on the fork blade
        bp = Vector(blade_pts[s][5])
        Fr_(_tube("stirrup_guide", bp, Vector((bp.x + 0.03, 0.034 * s, bp.z + 0.012)), 0.003, 5), "nickel")

    # ================================================================ wheels
    wr = rng.fork("wheels")
    for o, m in _wheel("r_", Ar, R, rt, True, wr.u(0, 360), M):
        Rr_(o, m)
    for o, m in _wheel("f_", Af, R, rt, False, wr.u(0, 360), M):
        Fr_(o, m)
    Rr_(cylinder("sprocket", 0.0375, 0.003, loc=(Ar.x, -0.047, Ar.z), rot=(90, 0, 0), verts=18), "iron")

    # ================================================================ mudguards
    hw, dep, th = 0.026, 0.014, 0.0025
    gp_o, gp_i = [], []
    for k in range(7):
        a = radians(70.0 - 140.0 * k / 6)
        aa = hw * sin(a) / sin(radians(70))
        bb = -dep * (1 - cos(a)) / (1 - cos(radians(70)))
        gp_o.append((aa, bb))
        gp_i.append((aa * (1 - th / hw), -th + bb * (dep - th) / dep))
    gprof = gp_o + list(reversed(gp_i))

    def guard(name, A, a0, a1):
        n = max(10, int(abs(a1 - a0) / 7.5))
        path = [(A.x + Rg * cos(radians(a0 + (a1 - a0) * i / n)), 0.0, A.z + Rg * sin(radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
        return sweep(name, path, gprof, up=(cos(radians(a0)), 0, sin(radians(a0))), smooth_path=False,
                     scale_fn=lambda t: min(1.0, 0.5 + 0.5 * min(t, 1 - t) / 0.03))

    def gpt(A, a, s, dr=-0.008):
        return A + Vector(((Rg + dr) * cos(radians(a)), hw * s, (Rg + dr) * sin(radians(a))))

    ra0, ra1 = 14.0, 196.0
    g = Rr_(guard("guard_rear", Ar, ra0, ra1), "frame")
    cream = mat("iron_painted", "#e6dfca", wear=0.4)
    assign_faces(g, cream, lambda c, n: atan2(c.z - Ar.z, c.x - Ar.x) > radians(163) or atan2(c.z - Ar.z, c.x - Ar.x) < radians(-90))
    fa0, fa1 = 84.0, 224.0
    Fr_(guard("guard_front", Af, fa0, fa1), "frame")
    st = []
    for s in (-1, 1):
        st += [(Ar + Y * (0.061 * s) + Vector((-0.006, 0, 0.006)), gpt(Ar, 142.0, s)),
               (Ar + Y * (0.061 * s) + Vector((-0.008, 0, 0.0)), gpt(Ar, ra1 - 6.0, s))]
    Rr_(_rods("guard_stays_r", st, 0.0024, 5), "nickel")
    st = []
    for s in (-1, 1):
        st += [(Af + Y * (0.054 * s), gpt(Af, fa1 - 8.0, s)), (Af + Y * (0.054 * s), gpt(Af, fa0 + 8.0, s))]
    Fr_(_rods("guard_stays_f", st, 0.0024, 5), "nickel")
    # red-lead mud flap on the front guard
    fl0 = Af + Vector((Rg * cos(radians(fa1 - 2)), 0, Rg * sin(radians(fa1 - 2))))
    Fr_(box("mud_flap", (0.004, 0.06, 0.075), loc=tuple(fl0 + Vector((0.004, 0, -0.03))), rot=(0, -12, 0), bevel=0.0015), "leather")

    # ================================================================ transmission
    yc = -0.047                                      # chain line
    r1, r2 = 0.092, 0.0365
    C1, C2 = (BB.x, BB.z), (Ar.x, Ar.z)
    cr = rng.fork("crank")
    phi = radians(cr.u(205, 335))
    Rr_(_tube("bb_axle", BB - Y * 0.074, BB + Y * 0.071, 0.008, 8), "nickel")
    if p.chain_guard == "hockey":
        outer = []
        for k in range(72):
            a = 2 * pi * k / 72 + phi
            rr = r1 + (0.004 if k % 2 == 0 else -0.003)
            outer.append((rr * cos(a), rr * sin(a)))
        holes = []
        for k in range(5):
            a = phi + radians(36 + 72 * k)
            holes.append([(0.056 * cos(a) + 0.023 * cos(a + 2 * pi * j / 9) * (1.0 if j % 9 else 1.25),
                           0.056 * sin(a) + 0.023 * sin(a + 2 * pi * j / 9) * (1.0 if j % 9 else 1.25)) for j in range(9)])
        Rr_(shape_extrude("chainring", outer, holes, depth=0.003, loc=(BB.x, yc + 0.0015, BB.z), rot=(90, 0, 0)), "nickel")
        links = _belt(C1, r1, C2, r2, 0.019)
        lb = box("link_b", (1, 1, 1))
        mats = []
        for i, (x, z, ta) in enumerate(links):
            sc = (0.0195, 0.0085, 0.0088) if i % 2 == 0 else (0.0195, 0.005, 0.0064)
            mats.append(trs((x, yc, z), (0, -degrees(ta), 0), sc))
        Rr_(instances_to_mesh("chain", lb, mats, uv_jitter=False, keep_base=False), "iron")
        # hockey-stick guard: plate over the top run and round the front of the chainwheel, with a rolled top flange
        nu, _nd = _tangent(C1, r1, C2, r2)
        au = atan2(nu[1], nu[0])
        ro, ri = r1 + 0.024, r1 - 0.022
        a_lo = radians(-62)
        E = (C2[0] + r2 * nu[0] + 0.01, C2[1] + r2 * nu[1])
        poly, top_edge = [], []
        for k in range(15):
            a = a_lo + (au - a_lo) * k / 14
            poly.append((C1[0] + ro * cos(a), C1[1] + ro * sin(a)))
        top_edge = list(poly)
        for k in range(8):
            a = au + pi * k / 7
            poly.append((E[0] + 0.022 * cos(a), E[1] + 0.022 * sin(a)))
            if k < 3:
                top_edge.append(poly[-1])
        for k in range(13):
            a = au + (a_lo - au) * k / 12
            poly.append((C1[0] + ri * cos(a), C1[1] + ri * sin(a)))
        Rr_(shape_extrude("chain_guard", poly, (), depth=0.0022, loc=(0, -0.0535, 0), rot=(90, 0, 0), bevel=0.0006), "frame")
        Rr_(sweep("guard_flange", [(x, -0.0445, z) for x, z in top_edge], rounded_rect(0.0022, 0.022, 0.0008, 1), up=(0, 1, 0),
                  smooth_path=False), "frame")
        bt = lathe("cg_bolt", [(0.0046, 0.0), (0.0036, 0.002), (0.0, 0.003)], segments=6)
        bm = [trs((C1[0] + (r1 + 0.001) * cos(a), -0.0557, C1[1] + (r1 + 0.001) * sin(a)), (90, 0, 0)) for a in (au, radians(-30))]
        bm.append(trs((E[0] + 0.02, -0.0557, E[1]), (90, 0, 0)))
        Rr_(instances_to_mesh("cg_bolts", bt, bm, keep_base=False), "nickel")
        Rr_(_rods("cg_stay", [(Vector((E[0] + 0.02, -0.054, E[1])), Vector((E[0] + 0.03, -0.05, E[1] - 0.05)))], 0.003, 5), "frame")
    else:
        R1, R2 = r1 + 0.024, r2 + 0.026
        nu, nd = _tangent(C1, R1, C2, R2)
        au, ad = atan2(nu[1], nu[0]), atan2(nd[1], nd[0])
        poly = []
        for k in range(21):
            a = (ad if ad < au else ad - 2 * pi) + ((au - (ad if ad < au else ad - 2 * pi))) * k / 20
            poly.append((C1[0] + R1 * cos(a), C1[1] + R1 * sin(a)))
        ad2 = ad if ad > au else ad + 2 * pi
        for k in range(11):
            a = au + (ad2 - au) * k / 10
            poly.append((C2[0] + R2 * cos(a), C2[1] + R2 * sin(a)))
        Rr_(shape_extrude("chain_case", poly, (), depth=0.024, loc=(0, -0.0345, 0), rot=(90, 0, 0), bevel=0.002), "frame")
        Rr_(sweep("case_bead", [(x, -0.0595, z) for x, z in poly], 0.0035, closed=True, sides=5, smooth_path=False), "frame")
        Rr_(lathe("case_disc", [(0.0, 0.0), (0.05, 0.0), (0.054, 0.002), (0.05, 0.0045), (0.03, 0.005), (0.0, 0.005)], segments=20,
                  loc=(BB.x, -0.0595, BB.z), rot=(90, 0, 0)), "frame")
        Rr_(lathe("case_cap", [(0.0, 0.0), (0.02, 0.0), (0.022, 0.002), (0.018, 0.005), (0.0, 0.006)], segments=12,
                  loc=(Ar.x + 0.0, -0.0595, Ar.z), rot=(90, 0, 0)), "nickel")
    # cranks + rubber block pedals
    pr = rng.fork("pedals")
    for s, ang in ((-1, phi), (1, phi + pi)):
        yk = -0.0665 if s < 0 else 0.063
        c0 = Vector((BB.x, yk, BB.z))
        c1 = c0 + Vector((cos(ang), 0, sin(ang))) * 0.165
        Rr_(sweep("crank", [c0 - (c1 - c0) * 0.1, c1 + (c1 - c0) * 0.07], rounded_rect(0.022, 0.011, 0.004, 2), up=(0, 1, 0),
                  smooth_path=False, scale_fn=lambda t: 1.0 - 0.32 * t), "nickel")
        Rr_(_tube("crank_boss", c0 - Y * 0.0075, c0 + Y * 0.0075, 0.0175, 12), "nickel")
        tl = pr.u(-28, 28)
        ped = []
        y0 = yk + s * 0.006
        ped.append((_tube("ped_axle", (0, y0, 0), (0, y0 + s * 0.108, 0), 0.0048, 6), "nickel"))
        for dx in (-0.03, 0.03):
            ped.append((box("ped_block", (0.02, 0.082, 0.021), loc=(dx, y0 + s * 0.06, 0), bevel=0.004), "rubber"))
        for dy in (0.016, 0.104):
            ped.append((box("ped_plate", (0.088, 0.003, 0.024), loc=(0, y0 + s * dy, 0), bevel=0.001), "nickel"))
        _bake([o for o, _ in ped], Matrix.Translation((c1.x, 0, c1.z)) @ Matrix.Rotation(radians(tl), 4, 'Y'))
        for o, m in ped:
            Rr_(o, m)

    # rear rod brake: rod under the down tube, bell crank at the bracket, stirrup under the chain stays
    nrm = Vector((-ddt.z, 0, ddt.x))
    if nrm.z > 0:
        nrm = -nrm
    b0 = DT1 + ddt * 0.03 + nrm * 0.03
    b1 = BB - ddt * 0.045 + nrm * 0.032
    cross_c = Ar - cs_dir * -(R + 0.032) + Vector((0, 0, -0.022))
    padr = Ar + cs_dir * (Rr0 - 0.019) + Vector((0, 0, -0.022))
    Rr_(_rods("brake_rod_r", [(b0, b1), (b1, cross_c), (HB - u * 0.004 + fw * 0.02, b0)], 0.0033, 5), "nickel")
    Rr_(_tube("stirrup_top_r", cross_c - Y * 0.034, cross_c + Y * 0.034, 0.004, 6), "nickel")
    for s in (-1, 1):
        Rr_(sweep("stirrup_r", [cross_c + Y * (0.032 * s), padr + Y * (0.032 * s), padr + Y * (0.013 * s)], 0.0036, sides=5,
                  smooth_path=False), "nickel")
        Rr_(box("brake_block_r", (0.044, 0.011, 0.012), loc=tuple(padr + Y * (0.011 * s) + Vector((0, 0, 0.004))), bevel=0.002), "rubber")

    # ================================================================ saddle (built about the clamp)
    SC = ST + u * 0.06
    Rr_(_tube("seat_post", ST - u * 0.02, SC, 0.0118, 10), "nickel")
    sr = rng.fork("saddle")
    sad = []

    def sfn(a, b):
        x = 0.155 - 0.285 * a
        w = 0.021 + 0.087 * smoothstep(0.16, 0.8, a)
        if a > 0.84:
            w *= sqrt(max(0.05, 1.0 - 0.6 * ((a - 0.84) / 0.16) ** 2))
        if a < 0.08:
            w *= 0.55 + 0.45 * sqrt(a / 0.08)
        hc = 0.02 + 0.03 * smoothstep(0.15, 0.9, a)
        an = (b - 0.5) * 2.0 * 1.72
        y = w * sin(an)
        z = 0.02 * ((a - 0.45) / 0.55) ** 2 + 0.008 * smoothstep(0.7, 1.0, a) - hc * (1 - cos(an))
        if a > 0.9:
            k = (a - 0.9) / 0.1
            z -= 0.034 * k * k
            x += 0.008 * k * k
        if a < 0.07:
            k = (0.07 - a) / 0.07
            z -= 0.022 * k * k
            x -= 0.006 * k * k
        return (x, y, z)

    top = surface("saddle_top", sfn, 14, 12, thickness=0.005)
    sad.append((top, "leather"))
    for s in (-1, 1):
        sad.append((sweep("saddle_rail", [(0.135, 0.01 * s, -0.03), (0.06, 0.018 * s, -0.058), (0.0, 0.02 * s, -0.062), (-0.06, 0.04 * s, -0.07),
                                          (-0.1, 0.06 * s, -0.088)], 0.0034, sides=5, step=0.04), "iron"))
        hel = [(-0.1 + 0.016 * cos(t * 2 * pi * 3.5), 0.06 * s + 0.016 * sin(t * 2 * pi * 3.5), -0.088 + 0.058 * t) for t in
               [i / 30.0 for i in range(31)]]
        sad.append((sweep("saddle_spring", hel, 0.0029, sides=4, smooth_path=False), "iron"))
    sad.append((sweep("cantle", [(-0.085, -0.082, -0.036), (-0.118, -0.045, -0.026), (-0.128, 0.0, -0.022), (-0.118, 0.045, -0.026),
                                 (-0.085, 0.082, -0.036)], rounded_rect(0.003, 0.016, 0.001, 1), up=(0, 0, 1), step=0.03), "iron"))
    sad.append((box("seat_clamp", (0.03, 0.05, 0.022), loc=(0, 0, -0.064), bevel=0.003), "iron"))
    sad.append((_tube("clamp_bolt", (0, -0.032, -0.064), (0, 0.032, -0.064), 0.005, 6), "nickel"))
    sad.append((_tube("nose_bolt", (0.11, 0, -0.03), (0.15, 0, -0.026), 0.004, 6), "nickel"))
    rv = lathe("rivet", [(0.0046, 0.0), (0.0036, 0.0019), (0.0, 0.0028)], segments=6)
    rm = [trs((-0.128 + 0.012 * abs(k) ** 1.5, 0.03 * k, 0.006 - 0.004 * abs(k)), (0, -62, 0)) for k in (-2, -1, 0, 1, 2)]
    rm += [trs((0.142, 0.009 * k, 0.012), (0, 55, 0)) for k in (-1, 1)]
    sad.append((instances_to_mesh("rivets", rv, rm, keep_base=False), "brass"))
    if p.saddlebag:
        bc = Vector((-0.172, 0.0, -0.085))
        sad.append((cylinder("tool_roll", 0.046, 0.2, loc=tuple(bc), rot=(90, 0, 0), verts=14, bevel=0.01), "leather"))
        sad.append((box("roll_flap", (0.05, 0.19, 0.006), loc=tuple(bc + Vector((-0.012, 0, 0.044))), rot=(0, 14, 0), bevel=0.002), "leather"))
        for s in (-1, 1):
            sad.append((ring_band("roll_strap", radius=0.0468, width=0.015, thick=0.003, loc=(bc.x, 0.06 * s, bc.z), rot=(90, 0, 0), segments=16),
                        "leather"))
            sad.append((box("roll_buckle", (0.004, 0.02, 0.016), loc=(bc.x - 0.049, 0.06 * s, bc.z + 0.004), bevel=0.001), "brass"))
            sad.append((box("roll_hanger", (0.004, 0.013, 0.06), loc=(bc.x + 0.04, 0.06 * s, bc.z + 0.036), rot=(0, 38, 0)), "leather"))
    Msad = Matrix.Translation(SC + Z * 0.064) @ Matrix.Rotation(radians(-3.0 + sr.u(-1.5, 1.5)), 4, 'Y')
    _bake([o for o, _ in sad], Msad)
    for o, m in sad:
        if o.name.startswith("saddle_top"):
            smooth(o, 70)
        Rr_(o, m)

    # ================================================================ rear carrier
    if p.rack:
        zr = 2 * R + 0.052
        x0, x1 = Ar.x - 0.2, Ar.x + 0.15
        yr = 0.066
        loop = [(x1, -yr + 0.02, zr), (x1 - 0.02, -yr, zr), (x0 + 0.02, -yr, zr), (x0, -yr + 0.02, zr), (x0, yr - 0.02, zr),
                (x0 + 0.02, yr, zr), (x1 - 0.02, yr, zr), (x1, yr - 0.02, zr)]
        Rr_(sweep("rack_loop", loop, 0.005, closed=True, sides=6, smooth_path=False), "frame")
        rs = [(Vector((x0, 0.0, zr)), Vector((x1, 0.0, zr)))]
        for kx in (0.25, 0.5, 0.75):
            xx = x0 + (x1 - x0) * kx
            rs.append((Vector((xx, -yr, zr)), Vector((xx, yr, zr))))
        for s in (-1, 1):
            rs.append((Vector((Ar.x - 0.01, yr * s, zr)), Ar + Y * (0.064 * s) + Vector((-0.012, 0, 0.004))))
            rs.append((Vector((x0 + 0.02, yr * s, zr)), Ar + Y * (0.064 * s) + Vector((-0.014, 0, 0.0))))
            sp_ = ST - u * 0.06
            rs.append((Vector((x1, 0.03 * s, zr)), Vector((sp_.x - 0.012, 0.024 * s, zr + 0.01))))
        Rr_(_rods("rack_rods", rs, 0.0042, 6, cap=True), "frame")
        Rr_(sweep("rack_clip", [(x1 - 0.02, 0, zr + 0.006), (x1 - 0.06, 0, zr + 0.03), (x0 + 0.1, 0, zr + 0.012), (x0 + 0.03, 0, zr + 0.02)],
                  rounded_rect(0.07, 0.003, 0.001, 1), step=0.04), "frame")

    # ================================================================ oil lamp (built about its own centre, facing +X)
    lamp = []
    lamp.append((lathe("lamp_body", [(0.0, -0.038), (0.024, -0.038), (0.031, -0.032), (0.031, 0.016), (0.034, 0.018), (0.034, 0.022),
                                     (0.039, 0.028), (0.041, 0.034), (0.041, 0.039), (0.037, 0.042), (0.035, 0.039)], segments=18,
                       rot=(0, 90, 0)), "brass"))
    lens_m = mat("glass_lantern", glow=9.0 if p.lamp_lit else 0.0, opacity=0.75)
    lamp.append((lathe("lamp_lens", [(0.0355, 0.039), (0.029, 0.0435), (0.016, 0.047), (0.0, 0.048)], segments=18, rot=(0, 90, 0)), lens_m))
    lamp.append((lathe("lamp_chimney", [(0.017, 0.026), (0.017, 0.046), (0.021, 0.048), (0.021, 0.052), (0.013, 0.058), (0.013, 0.063),
                                        (0.024, 0.064), (0.024, 0.067), (0.013, 0.076), (0.004, 0.078), (0.0, 0.082)], segments=14), "brass"))
    lamp.append((lathe("lamp_font", [(0.0, -0.06), (0.021, -0.06), (0.026, -0.056), (0.026, -0.036), (0.021, -0.03), (0.019, -0.024)],
                       segments=14), "brass"))
    lamp.append((_tube("lamp_wick", (0.0, -0.024, -0.04), (0.0, -0.04, -0.04), 0.0055, 8), "brass"))
    for s in (-1, 1):
        lamp.append((lathe("lamp_jewel", [(0.0115, 0.0), (0.0115, 0.004), (0.008, 0.007), (0.0, 0.008)], segments=10, loc=(0.0, 0.0295 * s, 0.0),
                           rot=(-90 * s, 0, 0)), lens_m))
    lamp.append((sweep("lamp_bail", [(0.0, 0.031 * cos(radians(a)) * 1.25, 0.03 + 0.062 * sin(radians(a))) for a in range(0, 181, 20)], 0.002,
                       sides=4, smooth_path=False), "nickel"))
    if p.basket:
        # lamp bracket carried on the basket support, low in front of the basket and clear of the tyre
        zb_ = Hc.z + 0.02 - 0.25 - 0.006
        bx_ = Hc.x + 0.21
        Lc = Vector((bx_ + 0.095 + 0.078, -0.075, zb_ + 0.066))
        mount = [(Vector((bx_ + 0.01, -0.075, zb_)), Vector((Lc.x - 0.05, -0.075, zb_))),
                 (Vector((Lc.x - 0.05, -0.075, zb_)), Lc - X * 0.05 + Z * 0.012), (Lc - X * 0.05, Lc - X * 0.036)]
    else:
        Lc = HT + u * 0.03 + fw * 0.12
        hb0 = HT + u * 0.006
        mount = [(hb0 + fw * 0.012, hb0 + fw * 0.06), (hb0 + fw * 0.06, Lc - X * 0.04 - Z * 0.01), (Lc - X * 0.04 - Z * 0.02, Lc - X * 0.036 + Z * 0.01)]
    _bake([o for o, _ in lamp], Matrix.Translation(Lc))
    for o, m in lamp:
        Fr_(o, m)
    Fr_(_rods("lamp_bracket", mount, 0.0045, 6, cap=True), "frame")

    # ================================================================ wicker basket + cargo
    bk_top = Hc.z + 0.02
    bk_h = 0.25
    bx = Hc.x + 0.075 + 0.135                        # basket centre (x)
    if p.basket:
        wick = mat("wicker")

        def ring(t, inset=0.0):
            w = lerp(0.29, 0.385, t) + 0.016 * sin(pi * t) - 2 * inset
            d = lerp(0.19, 0.27, t) + 0.012 * sin(pi * t) - 2 * inset
            return w, d

        nl = 10
        rings = []
        for i in range(nl + 1):
            t = i / nl
            w, d = ring(t)
            bulge = 0.0022 if i % 2 else 0.0
            zz = bk_top - bk_h + bk_h * t
            rings.append([(bx + py * (1 + bulge / d * 2), px * (1 + bulge / w * 2), zz) for px, py in rounded_rect(w, d, 0.055, 4)])
        w, d = ring(1.0, 0.004)
        rings.append([(bx + py, px, bk_top + 0.004) for px, py in rounded_rect(w, d, 0.052, 4)])
        for i in range(nl, -1, -2):
            t = i / nl
            w, d = ring(t, 0.008)
            zz = bk_top - bk_h + bk_h * t + (0.012 if i == 0 else 0.0)
            rings.append([(bx + py, px, zz) for px, py in rounded_rect(w, d, 0.048, 4)])
        bsk = _loft("basket", rings, cap_first=True, cap_last=True)
        smooth(bsk, 50)
        uv_box(bsk)
        Fr_(bsk, wick)
        for t, rr, dz in ((1.0, 0.0105, 0.004), (0.0, 0.008, 0.004), (0.62, 0.0055, 0.0)):
            w, d = ring(t)
            pth = [(bx + py, px, bk_top - bk_h + bk_h * t + dz) for px, py in rounded_rect(w + 0.004, d + 0.004, 0.057, 4)]
            Fr_(sweep("basket_roll", pth, rr, closed=True, sides=6, smooth_path=False, twist_deg=0.0), wick)
        back = bx - 0.135
        for s in (-1, 1):
            yy = 0.075 * s
            pth = [(back + 0.004, yy, bk_top - 0.06), (back - 0.006, yy, bk_top + 0.004), (Hc.x + 0.012, yy, Hc.z + 0.0135),
                   (Hc.x - 0.0135, yy, Hc.z + 0.002), (Hc.x - 0.002, yy, Hc.z - 0.0135), (back - 0.004, yy, bk_top - 0.045)]
            Fr_(sweep("basket_strap", pth, rounded_rect(0.003, 0.018, 0.001, 1), up=(0, 1, 0), step=0.03), "leather")
            Fr_(box("strap_buckle", (0.005, 0.024, 0.02), loc=(back - 0.008, yy, bk_top - 0.03), bevel=0.0015), "brass")
        # carrier struts from the axle and a cross bar under the basket
        zb = bk_top - bk_h - 0.006
        sup = [(Vector((bx + 0.01, -0.11, zb)), Vector((bx + 0.01, 0.11, zb)))]
        for s in (-1, 1):
            sup.append((Vector((bx + 0.01, 0.1 * s, zb)), Af + Y * (0.058 * s)))
        Fr_(_rods("basket_struts", sup, 0.004, 6, cap=True), "frame")
        if p.cargo:
            cg = rng.fork("cargo")
            paper = mat("paper", "#b9a377")
            zf = bk_top - bk_h + 0.012
            # parcel at the front, rolled papers behind it: everything stays in the shade of the lamp housing
            Fr_(box("parcel", (0.1, 0.19, 0.19), loc=(bx + 0.03, 0.02, zf + 0.1), rot=(cg.u(-4, 4), cg.u(3, 7), cg.u(-5, 5)), bevel=0.007), paper)
            Fr_(box("parcel_tie", (0.104, 0.012, 0.194), loc=(bx + 0.03, 0.02, zf + 0.1), rot=(0, 5, 0)), "leather")
            for k in range(3):
                a0 = Vector((bx - 0.06, -0.09 + 0.06 * k + cg.u(-0.01, 0.01), zf + 0.02))
                dr = Vector((cg.u(-0.1, 0.03), cg.u(-0.22, 0.22), 1.0)).normalized()
                Fr_(_tube("paper_roll", a0, a0 + dr * cg.u(0.28, 0.42), cg.u(0.013, 0.019), 9), paper)

    # ================================================================ pose: steer, lean, drop onto the paving
    Msteer = Matrix.Translation(Q) @ Matrix.Rotation(radians(p.steer), 4, u) @ Matrix.Translation(-Q)
    Mlean = Matrix.Rotation(radians(-p.lean), 4, 'X')
    _bake(front, Mlean @ Msteer)
    _bake(rear, Mlean)
    mn, mx = world_bbox(rear + front)
    sh = Vector((-(mn.x + mx.x) / 2, -(mn.y + mx.y) / 2, -mn.z))
    Msh = Matrix.Translation(sh)
    _bake(rear + front, Msh)
    Mrear = Msh @ Mlean
    Mfront = Mrear @ Msteer

    # side stand: clamped behind the bracket on the left chain stay, foot on the ground
    kp = Mrear @ (BB + cs_dir * 0.085 + Vector((0, 0.046, -0.012)))
    la = radians(p.lean + 21.0)
    kd = Vector((-0.22, sin(la), -cos(la))).normalized()
    foot = kp + kd * ((kp.z - 0.006) / -kd.z)
    stand = [_tube("stand_leg", kp, foot, 0.0075, 8, r2=0.006), _tube("stand_foot", Vector((foot.x, foot.y, 0.0)), Vector((foot.x, foot.y, 0.008)), 0.013, 10),
             box("stand_plate", (0.05, 0.012, 0.03), loc=tuple(kp + Vector((0.012, -0.006, 0.006))), rot=(-p.lean, 0, 0), bevel=0.002)]
    for o in stand:
        assign(o, M["iron"])

    # ================================================================ weathering
    nz = noise_mask(0.06, seed=seed + 3, lo=0.62 - 0.4 * cond, hi=0.95 - 0.35 * cond)
    nz2 = noise_mask(0.025, seed=seed + 5, lo=0.5, hi=0.8)
    low = height_ramp(0.0, 0.45)
    for o in rear + front + stand:
        nm = o.name
        ms = o.data.materials[0].name if o.data.materials else ""
        if nm.startswith(("r_tyre", "f_tyre")):
            paint(o, grime=mask_mul(noise_mask(0.05, seed=seed + 9, lo=0.3, hi=0.8), 0.35 + 0.5 * cond))
        elif nm.startswith(("guard_", "chain_guard", "chain_case", "fork_blade", "chain_stay", "dropout", "fork_tip")):
            paint(o, grime=mask_mul(low, 0.3 + 0.6 * cond), special=mask_mul(nz, 0.08 + 0.92 * cond * cond), wear=mask_mul(nz2, 0.1 + 0.5 * cond))
        elif nm.startswith(("saddle_top", "tool_roll", "grip")):
            paint(o, wear=mask_mul(facing((0, 0, 1), 2.0), 0.35 + 0.4 * cond), grime=0.15 + 0.3 * cond)
        elif nm.startswith(("chain", "sprocket", "stand", "saddle_spring", "saddle_rail")):
            paint(o, special=mask_mul(noise_mask(0.03, seed=seed + 7, lo=0.3, hi=0.7), 0.2 + 0.8 * cond), grime=0.3)
        elif nm.startswith("basket"):
            paint(o, grime=mask_mul(height_ramp(bk_top - bk_h + sh.z - 0.05, bk_top + sh.z), 0.5))
        elif "iron_painted" in ms:
            paint(o, special=mask_mul(nz, 0.04 + 0.96 * cond * cond), wear=mask_mul(nz2, 0.06 + 0.45 * cond), grime=0.2 * cond)
        elif nm.startswith(("r_rim", "f_rim", "r_spokes", "f_spokes", "r_hub", "f_hub", "handlebar", "crank", "chainring")):
            paint(o, grime=mask_mul(noise_mask(0.04, seed=seed + 11, lo=0.35, hi=0.75), 0.25 + 0.7 * cond))

    # ================================================================ light + sockets
    if p.lamp_lit:
        # at the lower lip of the lens: the housing shades the basket, bars and saddle behind / above it while the beam still
        # reaches the paving under the machine (shadow_near is inside the bezel)
        lp = Mfront @ (Lc + X * 0.062 - Z * 0.026)
        light("oil_lamp", tuple(lp), color="#ffb85e", intensity=float(p.lamp_power), range=5.5, flicker=0.3, shadow=True, shadow_near=0.012)
    if p.basket:
        socket("basket", "attach", tuple(Mfront @ Vector((bx, 0, bk_top - bk_h + 0.02))), role="prop_top")
    if p.rack:
        socket("rack", "attach", tuple(Mrear @ Vector((Ar.x - 0.03, 0, 2 * R + 0.06))), role="prop_top")
