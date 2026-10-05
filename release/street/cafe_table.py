"""Round bistro / gueridon table for the cafe terrace. A cast-iron tripod pedestal (turned baluster column
with three cabriole legs ending in scrolled toes on levelling pads; or a wrought flat-bar hourglass tripod;
or a heavy cast disc with a gadrooned ogee, a turned vase column and three big scroll feet) carries a cast
spider screwed under the top. The top is a veined, ring-stained marble slab or a boarded oak disc held in a
pinned brass rim, or an enamelled iron top under a draped tablecloth (cone folds, gathered corner pleats,
turned hem). A param-driven setting (porcelain coffee cups, saucers and spoons, creamer and bud vase / wine
bottle, glasses and a sprig) and a candle in a brass chamberstick or a small oil lamp sit on it.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, sqrt, acos, atan2, ceil
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt

META = {"name": "Cafe table", "class": "furniture", "mount": "ground", "focus": 0.8}
PARAMS = {
    "diameter": {"default": 0.7, "min": 0.5, "max": 1.1, "desc": "diameter of the round top (m)"},
    "height": {"default": 0.73, "min": 0.68, "max": 0.78, "desc": "height of the top surface (m)"},
    "top": {"default": "marble", "choices": ["marble", "wood", "cloth"], "desc": "marble slab / oak boards in a brass rim, or a draped tablecloth"},
    "base": {"default": "cabriole", "choices": ["cabriole", "scroll", "disc"],
             "desc": "cast cabriole tripod / wrought flat-bar scroll tripod / cast gadrooned disc pedestal on three scroll feet"},
    "setting": {"default": "coffee", "choices": ["none", "coffee", "wine"], "desc": "table setting"},
    "places": {"default": 2, "min": 1, "max": 3, "desc": "number of place settings (cups or glasses)"},
    "lamp": {"default": "candle", "choices": ["candle", "oil", "none"], "desc": "table light: candle in a brass chamberstick / small oil lamp"},
    "glow": {"default": 1.0, "min": 0.3, "max": 2.5, "desc": "multiplier on the table light intensity"},
    "paint": {"default": "#3c5f50", "desc": "paint colour of the cast ironwork"},
    "marble": {"default": "#f2efe9", "desc": "marble tint (dark tints get light veins)"},
    "stain": {"default": "#a47a52", "desc": "stain of the timber top"},
    "cloth": {"default": "round", "choices": ["round", "square"], "desc": "tablecloth cut"},
    "cloth_color": {"default": "#e9e1cf", "desc": "tablecloth colour"},
    "cloth_stripe": {"default": 0.0, "min": 0.0, "max": 0.2, "desc": "woven stripe width of the tablecloth (m), 0 = plain"},
    "cloth_color2": {"default": "#e6dcc6", "desc": "second colour of a striped tablecloth"},
    "cloth_drop": {"default": 0.24, "min": 0.12, "max": 0.42, "desc": "how far the cloth hangs below the top (m)"},
    "flower": {"default": "#c2382c", "desc": "petal colour of the flower in the bud vase"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of rust, stains and polish wear"},
}
VARIANTS = {
    "wine_wrought": {"top": "wood", "base": "scroll", "setting": "wine", "lamp": "oil", "diameter": 0.8, "seed": 4},
    "gueridon_small": {"top": "marble", "marble": "#2c4337", "base": "disc", "diameter": 0.55, "height": 0.75, "setting": "none",
                       "paint": "#8a2a22", "seed": 11},
    "bistro_cloth": {"top": "cloth", "cloth": "square", "cloth_color": "#9a2f28", "cloth_stripe": 0.05, "cloth_drop": 0.2, "base": "disc",
                     "setting": "coffee", "places": 3, "diameter": 0.8, "flower": "#e8c23a", "seed": 23},
    "terrace_large": {"top": "cloth", "cloth": "round", "cloth_drop": 0.32, "diameter": 1.0, "setting": "wine", "places": 3,
                      "base": "cabriole", "paint": "#2f4a66", "lamp": "oil", "seed": 31},
}


# ------------------------------------------------------------------------------------------------ helpers
def _rz(x, y, z, ang):
    a = radians(ang)
    c, s = cos(a), sin(a)
    return (x * c - y * s, x * s + y * c, z)


def _cr(P, sub=6, last_ghost=False):
    """Catmull-Rom through P (3D). last_ghost: the last point only steers the end tangent."""
    P = [np.array(q, dtype=float) for q in P]
    ext = [2 * P[0] - P[1]] + P + ([] if last_ghost else [2 * P[-1] - P[-2]])
    nseg = len(ext) - 3
    out = []
    for i in range(nseg):
        p0, p1, p2, p3 = ext[i:i + 4]
        for k in range(sub):
            t = k / sub
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(ext[nseg + 1])
    return [tuple(float(c) for c in q) for q in out]


def _scroll_leg(name, ctrl, Rf, zc, r0, r1, turns, profile, scales, sub=5, nsc=28, a0=235.0):
    """A leg drawn in the XZ plane (x = distance from the axis) that ends in a volute toe around (Rf, zc)."""
    sp = spiral_path(r0, r1, turns, n=nsc, a0_deg=a0, plane="XZ", center=(Rf, 0.0, zc))
    main = _cr([(x, 0.0, z) for x, z in ctrl] + [sp[0], sp[2]], sub, last_ghost=True)
    path = main + [tuple(q) for q in sp[1:]]
    A = np.asarray(path)
    seg = np.linalg.norm(np.diff(A, axis=0), axis=1)
    ts = float(seg[:len(main) - 1].sum() / seg.sum())
    xs = [0.0, 0.3 * ts, ts, 1.0]
    return sweep(name, path, profile, smooth_path=False, up=(0, 0, 1), scale_fn=lambda t: float(np.interp(t, xs, scales)))


def _put(objs, x, y, z, a):
    for o in objs:
        apply_transform(o)
        place(o, loc=(x, y, z), rot=(0, 0, a))
    return objs


def _lum(hx):
    h = hx.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.3 * r + 0.59 * g + 0.11 * b


def _rust_mask(zr, k):
    """Rust creeping up from the wet paving: a soft ramp (the material breaks it into blotches), strongest at the toes."""
    def fn(co, nrm):
        return k * np.clip(1.0 - co[:, 2] / zr, 0.0, 1.0) ** 1.3

    return fn


def _stud(name, r, h, seg=6):
    """Cheap domed pin / screw head sitting on z = 0."""
    return lathe(name, [(r, 0.0), (r * 0.86, h * 0.7), (0.0, h)], segments=seg)


# ------------------------------------------------------------------------------------------------ bases
LEG_ANGLES = (-90.0, 30.0, 150.0)


def _foot_pads(Rf, hp, rp, out):
    pad = lathe("pad", [(0.0, 0.0), (rp, 0.0), (rp + 0.0015, 0.0015), (rp + 0.0015, hp * 0.6), (rp - 0.002, hp), (0.0, hp)], segments=10)
    out.append(instances_to_mesh("foot_pads", pad, [trs(_rz(Rf, 0, 0, a)) for a in LEG_ANGLES], keep_base=False))


def _spider(R, zu, out):
    """Cast three-arm bracket screwed up into the underside of the top."""
    La = max(0.6 * R - 0.03, 0.08)
    out.append(cylinder("spider_hub", 0.064, 0.007, loc=(0, 0, zu - 0.0035), verts=20, bevel=0.0015, segments=1))
    heads = []
    for a in LEG_ANGLES:
        aa = a + 60.0
        out.append(box("spider_arm", (La, 0.03, 0.007), loc=_rz(0.045 + La / 2, 0, zu - 0.0035, aa), rot=(0, 0, aa), bevel=0.002, segments=1))
        heads.append(trs(_rz(0.045 + La - 0.022, 0, zu - 0.007, aa), (180, 0, aa)))
        heads.append(trs(_rz(0.045 + La * 0.45, 0, zu - 0.007, aa), (180, 0, aa)))
    out.append(instances_to_mesh("spider_screws", _stud("screw", 0.0055, 0.003), heads, keep_base=False))


def _base_cabriole(R, H, zu, out, gilt):
    zt = zu - 0.006
    zh = 0.40 * H
    Rf = clamp(0.80 * R, 0.2, 0.38)
    zb, zn = zh + 0.10, zt - 0.105
    bal = [(0.0, 0.025), (0.07, 0.034), (0.17, 0.041), (0.30, 0.040), (0.46, 0.031), (0.62, 0.023), (0.8, 0.0195), (1.0, 0.0185)]
    prof = [(0.0, zh - 0.135), (0.008, zh - 0.133), (0.017, zh - 0.122), (0.021, zh - 0.108), (0.017, zh - 0.094), (0.011, zh - 0.084),
            (0.013, zh - 0.078), (0.024, zh - 0.07), (0.038, zh - 0.06), (0.045, zh - 0.05), (0.047, zh - 0.042), (0.047, zh + 0.042),
            (0.043, zh + 0.05), (0.036, zh + 0.054), (0.036, zh + 0.062), (0.041, zh + 0.068), (0.041, zh + 0.076), (0.031, zh + 0.084)]
    prof += [(r, zb + (zn - zb) * t) for t, r in bal]
    prof += [(0.0185, zn + 0.004), (0.026, zn + 0.01), (0.028, zn + 0.018), (0.026, zn + 0.026), (0.019, zn + 0.032), (0.019, zn + 0.05),
             (0.024, zn + 0.066), (0.04, zn + 0.084), (0.054, zn + 0.094), (0.056, zn + 0.098), (0.056, zt), (0.0, zt)]
    out.append(lathe("column", prof, segments=24))
    # three cabriole legs: knee high at the hub, ankle, volute toe
    r0 = 0.034
    zc = r0 + 0.017 * 0.62 + 0.006
    fx_ = [0.0, 0.19, 0.385, 0.565, 0.68, 0.79]
    zz = [zh + 0.005, zh + 0.040, zh + 0.030, zh - 0.040, zh * 0.50, zh * 0.24]
    ctrl = [(0.028 + f * (Rf - 0.028), z) for f, z in zip(fx_, zz)]
    sec = [(-0.013, -0.012), (-0.009, -0.017), (0.009, -0.017), (0.013, -0.012), (0.013, 0.0085), (0.0062, 0.012),
           (0.005, 0.0165), (0.002, 0.019), (-0.002, 0.019), (-0.005, 0.0165), (-0.0062, 0.012), (-0.013, 0.0085)]
    ros, bolts = [], []
    for k, a in enumerate(LEG_ANGLES):
        leg = _scroll_leg(f"leg{k}", ctrl, Rf, zc, r0, 0.009, 1.3, sec, [0.95, 1.08, 0.62, 0.36], sub=4, nsc=21)
        place(leg, rot=(0, 0, a))
        out.append(leg)
        kx, kz = ctrl[1][0] + 0.004, ctrl[1][1] - 0.004
        ros.append(trs(_rz(kx, -0.0136, kz, a), (90, 0, a)))
        ros.append(trs(_rz(kx, 0.0136, kz, a), (-90, 0, a)))
        bolts.append(trs(_rz(0.0465, 0, zh, a + 60), (0, 90, a + 60)))
    rosette = lathe("rosette", [(0.0125, 0.0), (0.0135, 0.002), (0.009, 0.0045), (0.0055, 0.0042), (0.0, 0.0085)], segments=8)
    gilt.append(instances_to_mesh("leg_rosettes", rosette, ros, keep_base=False))
    hb = bolt("hub_bolt", 0.008, 0.006, kind="hex")
    out.append(instances_to_mesh("hub_bolts", hb, bolts, keep_base=False))
    _foot_pads(Rf, 0.009, 0.013, out)
    _spider(R, zu, out)


def _base_scroll(R, H, zu, out, gilt):
    """Wrought flat-bar hourglass tripod: three bars from lugs under the top pinch at a collared waist and
    splay to scrolled toes; a hoop under the top and a round-bar ring low down brace it."""
    ru = 0.66 * R
    zw = 0.5 * H
    Rf = clamp(0.80 * R, 0.2, 0.38)
    r0, th = 0.03, 0.010
    zc = r0 + th * 0.5 * 0.9 + 0.005
    zl = zu - th * 0.5
    ring_pt = (0.021 + 0.42 * (Rf - 0.021), zw * 0.48)
    ctrl = [(ru, zl), (ru - 0.055, zl), (ru - 0.105, zl - 0.024), (ru * 0.38, zu - 0.105), (0.045, zw + 0.11), (0.024, zw + 0.04),
            (0.021, zw), (0.024, zw - 0.04), (0.05, zw - 0.12), ring_pt, (0.021 + 0.72 * (Rf - 0.021), zw * 0.2)]
    sec = rounded_rect(0.03, th, 0.0028, 1)
    screws, rivets = [], []
    for k, a in enumerate(LEG_ANGLES):
        leg = _scroll_leg(f"bar{k}", ctrl, Rf, zc, r0, 0.008, 1.35, sec, [1.0, 1.0, 1.0, 0.55], sub=4, nsc=22)
        place(leg, rot=(0, 0, a))
        out.append(leg)
        screws.append(trs(_rz(ru - 0.02, 0, zu - th, a), (180, 0, a)))
        rivets.append(trs(_rz(ring_pt[0] + th * 0.5, 0, ring_pt[1] - 0.003, a), (0, 115, a)))
        rivets.append(trs(_rz(ru - 0.03 + 0.002, 0, zu - th - 0.011, a), (0, 90, a)))
        # C-scroll stay between the bar and the hoop under the top
        cs = spiral_path(0.007, 0.03, 1.2, n=19, a0_deg=40, plane="XZ", center=(ru - 0.1, 0.0, zu - 0.06), ccw=False)
        s = sweep(f"c_scroll{k}", cs, rounded_rect(0.018, 0.006, 0.002, 1), smooth_path=False, up=(0, 0, 1),
                  scale_fn=lambda t: 0.6 + 0.4 * t)
        place(s, rot=(0, 0, a))
        out.append(s)
    out.append(ring_band("top_hoop", radius=ru - 0.03, width=0.022, thick=0.004, loc=(0, 0, zu - th - 0.011), segments=40))
    out.append(torus("brace_ring", major=ring_pt[0] - th * 0.5 - 0.005, minor=0.005, loc=(0, 0, ring_pt[1]), major_seg=32, minor_seg=6))
    out.append(lathe("collar", [(0.012, zw - 0.02), (0.031, zw - 0.02), (0.034, zw - 0.016), (0.034, zw - 0.007), (0.038, zw), (0.034, zw + 0.007),
                                (0.034, zw + 0.016), (0.031, zw + 0.02), (0.012, zw + 0.02)], segments=16))
    out.append(lathe("spindle", [(0.0, zw - 0.2), (0.005, zw - 0.19), (0.011, zw - 0.17), (0.006, zw - 0.15), (0.0065, zw - 0.14), (0.011, zw - 0.132),
                                 (0.0065, zw - 0.124), (0.007, zw), (0.0065, zw + 0.124), (0.011, zw + 0.132), (0.0065, zw + 0.14), (0.006, zw + 0.15),
                                 (0.011, zw + 0.17), (0.005, zw + 0.19), (0.0, zw + 0.2)], segments=10))
    out.append(instances_to_mesh("lug_screws", _stud("screw", 0.0055, 0.003), screws + rivets, keep_base=False))
    _foot_pads(Rf, 0.008, 0.012, out)


def _base_disc(R, H, zu, out, gilt):
    """Heavy cast disc: moulded flange, gadrooned ogee, collared vase column; three big scroll feet clasp the rim."""
    zt = zu - 0.006
    Rb = clamp(0.62 * R, 0.19, 0.30)
    r0, sh = 0.038, 0.019                                  # volute radius / bar thickness of the feet
    zc = r0 + sh * 0.5 + 0.007
    z0 = 0.05                                              # underside of the cast disc
    Rf = Rb + 0.048
    out.append(lathe("disc_flange", [(0.0, z0), (Rb - 0.006, z0), (Rb, z0 + 0.005), (Rb, z0 + 0.019), (Rb - 0.004, z0 + 0.024),
                                     (Rb - 0.013, z0 + 0.026), (Rb - 0.0105, z0 + 0.031), (Rb - 0.0105, z0 + 0.035), (Rb - 0.013, z0 + 0.039),
                                     (Rb - 0.02, z0 + 0.041), (Rb - 0.032, z0 + 0.041)], segments=40))
    ng = 14
    rg0, rg1 = Rb - 0.024, 0.069
    zg0, hd = z0 + 0.037, 0.066

    def gad(u, v):
        th = 2 * pi * u
        r = lerp(rg0, rg1, v)
        z = zg0 + hd * (0.6 * sin(min(v / 0.7, 1.0) * pi / 2) + 0.4 * smoothstep(0.5, 1.0, v) ** 1.5)
        dz = 0.009 * sin(pi * clamp(v * 1.02)) ** 0.5 * abs(cos(ng * th / 2)) ** 0.55
        return ((r + 0.4 * dz) * cos(th), (r + 0.4 * dz) * sin(th), z + dz)

    dome = surface("gadroons", gad, ng * 6, 6, closed_u=True)
    smooth(dome, 38)
    out.append(dome)
    zk = zg0 + hd
    prof = [(0.058, zk - 0.008), (0.073, zk - 0.005), (0.077, zk + 0.001), (0.075, zk + 0.007), (0.066, zk + 0.011), (0.058, zk + 0.014),
            (0.052, zk + 0.022), (0.051, zk + 0.03), (0.056, zk + 0.035), (0.058, zk + 0.041), (0.055, zk + 0.047), (0.046, zk + 0.052)]
    zb0, zb1 = zk + 0.058, zt - 0.106
    bal = [(0.0, 0.040), (0.08, 0.050), (0.19, 0.0585), (0.32, 0.058), (0.46, 0.049), (0.6, 0.039), (0.75, 0.032), (0.9, 0.0285), (1.0, 0.0275)]
    prof += [(r, lerp(zb0, zb1, t)) for t, r in bal]
    prof += [(0.0275, zb1 + 0.004), (0.036, zb1 + 0.009), (0.0395, zb1 + 0.017), (0.036, zb1 + 0.025), (0.028, zb1 + 0.031), (0.0265, zb1 + 0.05),
             (0.031, zb1 + 0.066), (0.044, zb1 + 0.084), (0.056, zb1 + 0.096), (0.058, zb1 + 0.10), (0.058, zt), (0.0, zt)]
    out.append(lathe("pedestal", prof, segments=24))
    # scroll feet: a flat cast bar from under the disc that drops and curls up into a volute beside the rim
    ctrl = [(Rb - 0.10, z0 - 0.0085), (Rb - 0.05, z0 - 0.009), (Rb - 0.008, z0 - 0.012)]
    sec = [(-0.021, -0.006), (-0.017, -0.0095), (0.017, -0.0095), (0.021, -0.006), (0.021, 0.006), (0.017, 0.0095), (-0.017, 0.0095), (-0.021, 0.006)]
    for k, a in enumerate(LEG_ANGLES):
        foot = _scroll_leg(f"foot{k}", ctrl, Rf, zc, r0, 0.011, 1.2, sec, [1.0, 1.0, 0.92, 0.5], sub=3, nsc=20)
        place(foot, rot=(0, 0, a))
        out.append(foot)
    _foot_pads(Rf, 0.008, 0.015, out)
    _spider(R, zu, out)


# ------------------------------------------------------------------------------------------------ tops
def _brass_rim(R, H, tt, n_pins, out):
    z0 = H - tt
    rim = lathe("brass_rim", [(R - 0.009, z0 - 0.0015), (R - 0.001, z0 - 0.0015), (R + 0.0016, z0 + 0.001), (R + 0.0016, z0 + 0.005),
                              (R + 0.0004, z0 + 0.007), (R + 0.0004, H - 0.007), (R + 0.0016, H - 0.005), (R + 0.0016, H - 0.0005),
                              (R - 0.0004, H + 0.0014), (R - 0.007, H + 0.0014), (R - 0.008, H + 0.0002)], segments=56, smooth_angle=60)
    out.append(rim)
    out.append(instances_to_mesh("rim_pins", _stud("rim_pin", 0.0032, 0.0016), [
        trs(_rz(R + 0.0003, 0, H - tt * 0.5, 360.0 * k / n_pins + 11), (0, 90, 360.0 * k / n_pins + 11)) for k in range(n_pins)], keep_base=False))


def _board_poly(x0, x1, Rw, seg=64):
    a0, a1 = acos(clamp(x1 / Rw, -1, 1)), acos(clamp(x0 / Rw, -1, 1))
    n = max(2, int(ceil((a1 - a0) / (2 * pi / seg))))
    up = [(Rw * cos(a0 + (a1 - a0) * i / n), Rw * sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
    pts = up + [(x, -y) for x, y in reversed(up)]
    out = []
    for q in pts:
        if not out or (abs(q[0] - out[-1][0]) + abs(q[1] - out[-1][1])) > 1e-5:
            out.append(q)
    if (abs(out[0][0] - out[-1][0]) + abs(out[0][1] - out[-1][1])) < 1e-5:
        out.pop()
    return out


def _fractal(r_, p0, p1, rough, n):
    """Wandering polyline p0 -> p1 by midpoint displacement (n = 2^k + 1 points)."""
    pts = [np.array(p0, dtype=float), np.array(p1, dtype=float)]
    amp = rough * float(np.linalg.norm(pts[1] - pts[0]))
    while len(pts) < n:
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            L = max(float(np.linalg.norm(d)), 1e-9)
            new += [(a + b) / 2 + np.array([-d[1], d[0]]) / L * r_.u(-amp, amp), b]
        pts = new
        amp *= 0.55
    return np.array(pts)


def _marble_bands(r_, Rv, R, wr, stain_at):
    """Veins and ring stains as soft bands: (centre line (n,2), half width (n), strength (n), closed)."""
    veins, stains = [], []
    phi = r_.u(0, pi)                                            # bedding direction of the slab
    d0 = np.array([cos(phi), sin(phi)])
    n0 = np.array([-d0[1], d0[0]])
    n_main = int(clamp(round(2.0 + 3.0 * (2 * R - 0.5)), 2, 4))
    for k in range(n_main):
        off = (-0.62 + 1.24 * (k + 0.5) / n_main + r_.u(-0.1, 0.1)) * Rv
        a = phi + r_.u(-0.3, 0.3)
        dd = np.array([cos(a), sin(a)])
        c0 = n0 * off + d0 * r_.u(-0.1, 0.1) * Rv
        b = float(c0 @ dd)
        disc = sqrt(max(b * b - (float(c0 @ c0) - Rv * Rv), 1e-9))
        c = _fractal(r_, c0 + dd * (-b - disc), c0 + dd * (-b + disc), 0.12, 17)
        rr = np.linalg.norm(c[1:-1], axis=1)
        c[1:-1] *= np.minimum(1.0, 0.985 * Rv / np.maximum(rr, 1e-9))[:, None]
        m = len(c)
        ph1, ph2 = r_.u(0, 6.3), r_.u(0, 6.3)
        hw = r_.u(0.009, 0.014) * (0.55 + 0.9 * (0.5 + 0.5 * np.sin(np.arange(m) * 0.9 + ph1)))
        st = r_.u(0.6, 0.85) * (0.6 + 0.4 * (0.5 + 0.5 * np.sin(np.arange(m) * 0.6 + ph2)))
        veins.append((c, hw, st, False))
        for _ in range(2 + (1 if r_.chance(0.5) else 0)):               # feathered side branches
            i = int(r_.u(2, m - 3))
            tg = c[i + 1] - c[i - 1]
            ang = atan2(tg[1], tg[0]) + radians(r_.u(18, 50)) * (1 if r_.chance(0.5) else -1) + (pi if r_.chance(0.5) else 0.0)
            end = c[i] + np.array([cos(ang), sin(ang)]) * r_.u(0.3, 0.75) * Rv
            re = float(np.linalg.norm(end))
            if re > 0.96 * Rv:
                end *= 0.96 * Rv / re
            cb = _fractal(r_, c[i], end, 0.14, 9)
            rr = np.linalg.norm(cb, axis=1)
            cb *= np.minimum(1.0, 0.975 * Rv / np.maximum(rr, 1e-9))[:, None]
            tb = np.linspace(0, 1, len(cb))
            hwb = r_.u(0.003, 0.0048) * np.sin(pi * tb) ** 0.5
            veins.append((cb, hwb, np.full(len(cb), r_.u(0.5, 0.85)), False))
    for (sx, sy) in stain_at:                                    # dull rings left by wet cups and glasses
        r0 = r_.u(0.03, 0.036)
        m = 14
        a_ = np.arange(m) * (2 * pi / m)
        c = np.stack([sx + r0 * np.cos(a_), sy + r0 * np.sin(a_)], 1)
        st = (0.35 + 0.4 * wr) * (0.25 + 0.75 * (0.5 + 0.5 * np.cos(a_ - r_.u(0, 6.3))) ** 1.5)
        stains.append((c, np.full(m, 0.0034), st, True))
    return veins, stains


def _band_geo(bands, Rv, V, E):
    """Centre line + two edge lines per band as constraint edges for the triangulation."""
    for c, hw, st, closed in bands:
        n = len(c)
        t = (np.roll(c, -1, 0) - np.roll(c, 1, 0)) if closed else np.gradient(c, axis=0)
        t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
        nr = np.stack([-t[:, 1], t[:, 0]], 1)
        for L in (c, c + nr * hw[:, None], c - nr * hw[:, None]):
            o = len(V)
            for q in L:
                r = sqrt(q[0] * q[0] + q[1] * q[1])
                k = Rv / r if r > Rv else 1.0
                V.append((float(q[0] * k), float(q[1] * k)))
            E += [(o + i, o + i + 1) for i in range(n - 1)]
            if closed:
                E.append((o + n - 1, o))


def _band_mask(bands, z_top):
    """Vectorised mask: 1 on the band centre lines falling to 0 at their edges (top face only)."""
    A, B, HA, HB, SA, SB = [], [], [], [], [], []
    for c, hw, st, closed in bands:
        idx = list(range(len(c) - 1)) + ([len(c) - 1] if closed else [])
        for i in idx:
            j = (i + 1) % len(c)
            A.append(c[i]); B.append(c[j]); HA.append(hw[i]); HB.append(hw[j]); SA.append(st[i]); SB.append(st[j])
    if not A:
        return lambda co, nrm: np.zeros(len(co))
    A, B = np.array(A), np.array(B)
    HA, HB, SA, SB = np.array(HA), np.array(HB), np.array(SA), np.array(SB)
    D = B - A
    DD = np.maximum((D * D).sum(1), 1e-12)

    def fn(co, nrm):
        P = co[:, None, :2]
        rel = P - A[None]
        t = np.clip((rel * D[None]).sum(2) / DD[None], 0.0, 1.0)
        dist = np.linalg.norm(rel - t[..., None] * D[None], axis=2)
        hw = HA[None] + (HB - HA)[None] * t
        st = SA[None] + (SB - SA)[None] * t
        v = st * np.clip(1.0 - dist / (hw + 1e-7), 0.0, 1.0) * np.clip(hw / 0.0012, 0.0, 1.0)
        return v.max(axis=1) * (co[:, 2] > z_top - 1e-4)

    return fn


def _marble_top(R, H, bands):
    """Flat top face triangulated around the vein / stain bands so vertex masks can draw them crisply."""
    Rv = R - 0.004
    nseg = 64
    V = [(Rv * cos(2 * pi * k / nseg), Rv * sin(2 * pi * k / nseg)) for k in range(nseg)]
    E = [(k, (k + 1) % nseg) for k in range(nseg)]
    s = max(0.05, 0.16 * R)                                     # sparse fill so broad stains have resolution
    ny = int(Rv / (s * 0.866)) + 1
    for j in range(-ny, ny + 1):
        for i in range(-ny - 1, ny + 2):
            x, y = (i + 0.5 * (j % 2)) * s, j * s * 0.866
            if x * x + y * y < (Rv - 0.55 * s) ** 2:
                V.append((x, y))
    _band_geo(bands, Rv, V, E)
    res = delaunay_2d_cdt([Vector(v) for v in V], E, [], 0, 1e-5)
    vo, fo = res[0], res[2]
    faces = []
    for f in fo:
        a, b, c = vo[f[0]], vo[f[1]], vo[f[2]]
        area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if abs(area) < 1e-12:
            continue
        faces.append(tuple(f) if area > 0 else tuple(reversed(f)))
    return mesh_object("marble_top", [(v.x, v.y, H) for v in vo], faces)


def _cloth(p, rng, R, H):
    """Draped tablecloth: flat on the top, a soft break over the edge, then cone folds that deepen towards a
    turned hem (round cut) or four gathered corner tails with pleats (square cut). Cloth length is kept along
    every meridian so the hem scallops; UVs are the flat cloth so weave and stripes follow the drape."""
    square = p.cloth == "square"
    er = 0.011
    zc = H + 0.0016
    Rc = R + 0.003
    hmax = H - 0.06
    drop = min(p.cloth_drop, hmax)
    rot0 = radians(rng.u(-20, 20))
    hs = Rc + drop
    q = [rng.u(0, 2 * pi) for _ in range(3)]
    A0 = clamp(0.034 + 0.19 * drop, 0.036, 0.1)
    folds = []                                               # (theta0, theta1, amplitude at the hem, skew, segments)
    if square:
        Lc = min(hs * sqrt(2) - Rc, hmax) - drop
        cuts = []
        for k in range(4):
            pc = rot0 + pi / 4 + k * pi / 2
            j = [radians(rng.u(-2.5, 2.5)) for _ in range(4)]
            cuts.append((pc - radians(28) + j[0], pc - radians(9.5) + j[1], pc + radians(9.5) + j[2], pc + radians(28) + j[3]))
        for k in range(4):
            b1, b2, b3, b4 = cuts[k]
            nb1 = cuts[(k + 1) % 4][0] + (2 * pi if k == 3 else 0.0)
            folds.append((b1, b2, (0.03 + 0.11 * Lc) * rng.u(0.8, 1.15), rng.u(-0.15, 0.3), 6))
            folds.append((b2, b3, (0.045 + 0.2 * Lc) * rng.u(0.9, 1.1), rng.u(-0.15, 0.15), 7))
            folds.append((b3, b4, (0.03 + 0.11 * Lc) * rng.u(0.8, 1.15), rng.u(-0.3, 0.15), 6))
            folds.append((b4, nb1, 0.75 * A0 * rng.u(0.7, 1.2), rng.u(-0.3, 0.3), 6))
    else:
        nf = int(clamp(round(2 * pi * R / 0.21), 8, 14))
        ws = [rng.u(0.6, 1.5) for _ in range(nf)]
        tot = sum(ws)
        th = rng.u(0, 2 * pi)
        for w in ws:
            dth = 2 * pi * w / tot
            folds.append((th, th + dth, A0 * rng.u(0.7, 1.25) * clamp(w * nf / tot, 0.7, 1.3), rng.u(-0.35, 0.35), 7))
            th += dth
    nfold = len(folds)
    vd = [0.22 * min(folds[k - 1][2], folds[k][2]) for k in range(nfold)]      # valley depth at the start of fold k
    TH, F = [], []
    for k, (t0, t1, amp, skew, ns) in enumerate(folds):
        v0, v1 = vd[k], vd[(k + 1) % nfold]
        for i in range(ns):
            s = i / ns
            TH.append(t0 + (t1 - t0) * s)
            F.append(amp * sin(pi * s ** (2.0 ** skew)) ** 0.85 - lerp(v0, v1, s))
    TH = np.array(TH[::-1])                                  # descending angle -> outward facing quads
    F = np.array(F[::-1])
    nu = len(TH)
    c, s = np.cos(TH), np.sin(TH)
    if square:
        Ld = np.minimum(hs / np.maximum(np.maximum(np.abs(np.cos(TH - rot0)), np.abs(np.sin(TH - rot0))), 1e-6) - Rc, hmax)
    else:
        Ld = np.minimum(drop * (1 + 0.06 * np.sin(TH + q[0]) + 0.03 * np.sin(3 * TH + q[1])), hmax)

    rows, rho = [], []

    def ring(r, z, ro):
        rows.append(np.stack([r * c, r * s, np.broadcast_to(z, (nu,)).astype(float)], 1))
        rho.append(np.broadcast_to(ro, (nu,)).astype(float))

    for f in (0.0, 0.58, 1.0):
        ring((Rc - er) * f, zc, (Rc - er) * f)
    for a in (30.0, 60.0, 90.0):
        ring(Rc - er + er * sin(radians(a)), zc - er * (1 - cos(radians(a))), Rc - er + er * radians(a))
    nd = 7
    j_h0 = len(rows)
    for k in range(1, nd + 1):
        t = k / nd
        l = t * Ld
        o = np.clip(F * t ** 0.85, -0.45 * l, 0.85 * l)
        if k == 3:
            o = o + 0.0022                                   # the pressed crease of a cloth kept folded in the linen press
        ring(Rc + o, zc - er - np.sqrt(np.maximum(l * l - o * o, 0.0)), Rc - er + er * pi / 2 + l)
    P = np.stack(rows, 1)                                    # (nu, rows, 3)
    RHO = np.stack(rho, 1)
    # inner skin of the hanging part (turned hem at the bottom), offset along the surface normal
    thk = 0.0022
    inner, inner_rho = [], []
    for k in range(nd, 1, -1):
        j = j_h0 + k - 1
        tu = np.roll(P[:, j], -1, 0) - np.roll(P[:, j], 1, 0)
        tv = (P[:, min(j + 1, P.shape[1] - 1)] - P[:, j - 1])
        nn = np.cross(tu, tv)
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        inner.append(P[:, j] - nn * thk)
        inner_rho.append(RHO[:, j])
    P = np.concatenate([P, np.stack(inner, 1)], 1)
    RHO = np.concatenate([RHO, np.stack(inner_rho, 1)], 1)
    nrow = P.shape[1]
    nv = nrow - 1
    cl = surface("tablecloth", lambda u, v: tuple(P[int(round(u * nu)) % nu, int(round(v * nv))]), nu, nv, closed_u=True)
    # flat-cloth UVs (metres): the weave runs straight across the top and follows the drape down the folds
    UV = np.stack([RHO * np.cos(TH - rot0)[:, None], RHO * np.sin(TH - rot0)[:, None]], 2).reshape(-1, 2)
    me = cl.data
    li = np.zeros(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", li)
    me.uv_layers[0].data.foreach_set("uv", UV[li].astype(np.float32).ravel())
    uv_keep(cl)
    weld(cl, 1e-4)
    smooth(cl, 55)
    return cl, zc, zc - er - drop


# ------------------------------------------------------------------------------------------------ table ware
def _cup_set(M):
    cz = 0.005
    cup = lathe("cup", [(0.0, cz), (0.019, cz), (0.020, cz + 0.003), (0.019, cz + 0.006), (0.026, cz + 0.012), (0.033, cz + 0.025),
                        (0.037, cz + 0.042), (0.039, cz + 0.056), (0.0375, cz + 0.0565), (0.035, cz + 0.042)], segments=14)
    coffee = lathe("coffee", [(0.0355, cz + 0.046), (0.0, cz + 0.0463)], segments=14)
    handle = sweep("cup_handle", [(0.035, 0, cz + 0.049), (0.049, 0, cz + 0.051), (0.058, 0, cz + 0.041), (0.054, 0, cz + 0.028),
                                  (0.041, 0, cz + 0.019), (0.030, 0, cz + 0.021)], 0.0032, sides=5, step=0.0085)
    saucer = lathe("saucer", [(0.0, 0.0), (0.022, 0.0), (0.024, 0.002), (0.045, 0.007), (0.066, 0.013), (0.068, 0.0145), (0.066, 0.0155),
                              (0.04, 0.0085), (0.024, 0.005), (0.0, 0.005)], segments=18)
    spoon = sweep("spoon", [(-0.044, -0.052, 0.0185), (-0.02, -0.052, 0.0172), (0.008, -0.052, 0.0166), (0.022, -0.052, 0.0166),
                            (0.032, -0.052, 0.0166), (0.042, -0.052, 0.0172)], [(-0.003, 0.0), (0.0, -0.0008), (0.003, 0.0), (0.0, 0.0008)],
                  up=(0, 0, 1), step=0.0088,
                  scale_fn=lambda t: float(np.interp(t, [0, 0.1, 0.55, 0.68, 0.82, 0.95, 1.0], [0.9, 1.3, 0.75, 1.0, 2.7, 2.2, 0.8])))
    for o in (cup, handle, saucer, coffee):
        assign(o, M("ceramic"))
    paint(coffee, wear=1.0, grime=0.85, mode="set")          # black coffee: the same glaze, stained right down
    assign(spoon, M("brass"))
    return [cup, handle, saucer, coffee, spoon]


def _creamer(M):
    jug = lathe("creamer", [(0.0, 0.0), (0.02, 0.0), (0.022, 0.003), (0.03, 0.02), (0.031, 0.035), (0.024, 0.055), (0.022, 0.066),
                            (0.026, 0.074), (0.0245, 0.075), (0.0205, 0.066), (0.0225, 0.055), (0.0285, 0.035), (0.0, 0.03)],
                segments=14)

    def spout(co):                                       # pull the lip out into a pouring spout towards -X
        if co.z > 0.06 and co.x < 0:
            f = smoothstep(0.06, 0.075, co.z) * max(0.0, -co.x / 0.026) ** 3
            return Vector((co.x - 0.011 * f, co.y * (1 - 0.35 * f), co.z + 0.003 * f))
        return co

    transform_verts(jug, spout)
    handle = sweep("creamer_handle", [(0.024, 0, 0.064), (0.04, 0, 0.066), (0.048, 0, 0.05), (0.042, 0, 0.03), (0.029, 0, 0.024)], 0.003, sides=5, step=0.0085)
    for o in (jug, handle):
        assign(o, M("ceramic"))
    return [jug, handle]


GLASS_OUT = [(0.006, 0.078), (0.02, 0.09), (0.034, 0.11), (0.038, 0.135), (0.035, 0.165), (0.033, 0.175)]


def _wine_glass(M, fill):
    g = lathe("wine_glass", [(0.0, 0.0), (0.031, 0.0), (0.0315, 0.0016), (0.013, 0.0045), (0.0052, 0.009), (0.0045, 0.07)] + GLASS_OUT, segments=16)
    assign(g, M("glass"))
    objs = [g]
    if fill > 0.02:
        zf = lerp(0.098, 0.14, fill)
        zs_, rs_ = [z for _, z in GLASS_OUT], [r for r, _ in GLASS_OUT]
        pr = [(0.0, 0.0825)] + [(r - 0.0014, z + 0.001) for r, z in GLASS_OUT[1:] if z < zf - 0.006]
        pr += [(float(np.interp(zf, zs_, rs_)) - 0.0014, zf), (0.0, zf + 0.0003)]
        w = lathe("wine", pr, segments=16)
        assign(w, M("wine"))
        objs.append(w)
    return objs


def _bottle(M, label_dir):
    b = lathe("bottle", [(0.0, 0.004), (0.03, 0.0), (0.036, 0.004), (0.0375, 0.012), (0.0375, 0.165), (0.034, 0.19), (0.024, 0.215),
                         (0.0165, 0.235), (0.0145, 0.255), (0.0145, 0.283), (0.0165, 0.285), (0.0165, 0.293), (0.0145, 0.295),
                         (0.0145, 0.30), (0.0, 0.30)], segments=18)
    lab = lathe("label", [(0.0379, 0.06), (0.0383, 0.062), (0.0383, 0.128), (0.0379, 0.13)], segments=28, angle=150.0, rot=(0, 0, label_dir - 75.0))
    neck = lathe("neck_label", [(0.0169, 0.236), (0.0151, 0.256), (0.015, 0.262)], segments=20, angle=120.0, rot=(0, 0, label_dir - 60.0))
    for o in (b, lab, neck):
        assign(o, M("bottle"))
    for o in (lab, neck):
        paint(o, wear=1.0, mode="set")                    # paper label: the matt pale side of the same material
    return [b, lab, neck]


def _vase(M, rng, sprig):
    if sprig:
        v = lathe("vase", [(0.0, 0.0), (0.021, 0.0), (0.023, 0.003), (0.019, 0.03), (0.0135, 0.07), (0.0125, 0.1), (0.016, 0.112),
                           (0.0145, 0.1125), (0.0105, 0.1), (0.0, 0.096)], segments=14)
        assign(v, M("brass"))                  # opaque: a glass vase lets the stem cast a hairline lamp shadow
        ztop = 0.112
    else:
        v = lathe("vase", [(0.0, 0.0), (0.019, 0.0), (0.021, 0.003), (0.029, 0.022), (0.031, 0.04), (0.023, 0.068), (0.0125, 0.092),
                           (0.0115, 0.106), (0.0155, 0.118), (0.014, 0.1185), (0.0095, 0.106), (0.0, 0.1)], segments=14)
        assign(v, M("ceramic"))
        ztop = 0.118
    objs = [v]
    green = M("bottle") if sprig else M("leaf")
    lean, side = rng.u(-0.03, 0.03), rng.u(-0.02, 0.02)
    hs = ztop + rng.u(0.075, 0.11)
    ctrl = [(0.0, 0.0, ztop - 0.07), (lean * 0.15, side * 0.1, ztop), (lean * 0.6, side * 0.6, ztop + (hs - ztop) * 0.6), (lean, side, hs)]
    stem_pts = _cr(ctrl, 4)
    stem = sweep("stem", stem_pts, 0.0022, sides=4, smooth_path=False, scale_fn=lambda t: 1.0 - 0.35 * t)
    assign(stem, green)
    objs.append(stem)
    leaf = surface("leaf_blade", lambda u, v: ((u - 0.5) * 0.024 * sin(pi * clamp(v * 0.92 + 0.05)) ** 0.8, v * 0.06,
                                               0.016 * sin(v * 2.0) - 0.02 * v * v - 0.004 * cos((u - 0.5) * pi)), 2, 4,
                   thickness=0.0008, uv_len=(0.25, 0.25))
    assign(leaf, green)
    realize(leaf)
    lm = []
    idx = list(range(5, len(stem_pts) - (0 if sprig else 1), 1 if sprig else 3))
    for n_, i in enumerate(idx):
        if not sprig and n_ >= 2:
            break
        lm.append(trs(stem_pts[i], (rng.u(15, 40), 0, n_ * 137.5 + rng.u(-20, 20)), rng.u(0.75, 1.05) * (0.8 if sprig else 1.0)))
    objs.append(instances_to_mesh("leaves", leaf, lm, keep_base=False))
    if not sprig:
        petal = surface("petal", lambda u, v: ((u - 0.5) * 0.017 * sin(pi * clamp(v * 0.85 + 0.1)), 0.004 + v * 0.03, 0.012 * v * v - 0.002 * cos((u - 0.5) * pi)),
                        2, 2, thickness=0.0008, uv_len=(0.25, 0.25))
        assign(petal, M("petal"))
        realize(petal)
        top = stem_pts[-1]
        npet = 9
        pm = [trs(top, (rng.u(-4, 14), 0, k * 360.0 / npet + rng.u(-6, 6)), rng.u(0.9, 1.08)) for k in range(npet)]
        pm += [trs((top[0], top[1], top[2] + 0.002), (rng.u(22, 34), 0, (k + 0.5) * 360.0 / 5), 0.62) for k in range(5)]
        objs.append(instances_to_mesh("petals", petal, pm, keep_base=False))
        eye = sphere("flower_eye", 0.0065, loc=(top[0], top[1], top[2] + 0.002), segments=8, rings=4, scale=(1, 1, 0.6))
        assign(eye, M("brass"))
        objs.append(eye)
    return objs


def _candle(M, rng):
    """Brass chamberstick with a part-burnt candle whose top glows through. Returns (objects, flame height, candle length)."""
    dish = lathe("chamberstick", [(0.0, 0.0), (0.03, 0.0), (0.04, 0.003), (0.045, 0.0095), (0.0462, 0.011), (0.0445, 0.0112), (0.038, 0.005),
                                  (0.017, 0.004), (0.0155, 0.007), (0.0155, 0.021), (0.0185, 0.023), (0.0185, 0.027), (0.0135, 0.028)], segments=20)
    ring = torus("thumb_ring", major=0.0115, minor=0.0024, loc=(0.056, 0, 0.0115), rot=(90, 0, 0), major_seg=12, minor_seg=5)
    hc = rng.u(0.10, 0.15)
    zt = 0.024 + hc
    rc, n = 0.0113, 20
    zh, zl = zt - 0.017, zt - 0.06                        # lit-through wax: solid glow above zh, fading out down to zl
    body = lathe("candle", [(0.0112, 0.024), (rc, zl)], segments=n, rot=(0, 0, 180.0 / n))
    top = lathe("candle_top", [(rc, zh), (rc, zt - 0.006), (0.0118, zt - 0.001), (0.0108, zt + 0.0006), (0.0085, zt - 0.0015),
                               (0.0, zt - 0.004)], segments=n)
    # the glow thins out downwards as fine interleaved needles of the two wax materials (reads as a gradient)
    T = [(rc * cos(2 * pi * i / n), rc * sin(2 * pi * i / n), zh) for i in range(n)]
    B = [(rc * cos(2 * pi * (i + 0.5) / n), rc * sin(2 * pi * (i + 0.5) / n), zl) for i in range(n)]
    fade_hot = mesh_object("candle_fade_hot", T + B, [(i, n + i, (i + 1) % n) for i in range(n)])
    fade_dim = mesh_object("candle_fade", T + B, [(n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)])
    for o in (fade_hot, fade_dim):
        uv_box(o)
        smooth(o, 80)
    drips = []
    for k in range(3):
        a = rng.u(0, 360)
        ln = rng.u(0.02, min(0.06, hc * 0.7))
        c, s = cos(radians(a)), sin(radians(a))
        d = sweep(f"wax_drip{k}", [(0.0105 * c, 0.0105 * s, zt - 0.001), (0.0122 * c, 0.0122 * s, zt - ln * 0.4), (0.0126 * c, 0.0126 * s, zt - ln)],
                  0.0021, sides=4, step=ln / 4, scale_fn=lambda t: 0.75 + 0.6 * t * t)
        drips.append(d)
    wick = cylinder("wick", 0.0007, 0.009, loc=(0, 0, zt), verts=4)
    for o in (dish, ring):
        assign(o, M("brass"))
    for o in [body, fade_dim] + drips:
        assign(o, M("wax"))
    for o in (top, fade_hot):
        assign(o, M("wax_hot"))
    assign(wick, M("iron"))
    return [dish, ring, body, top, fade_hot, fade_dim, wick] + drips, zt + 0.015, hc


def _oil_lamp(M):
    """Small brass table oil lamp with a clear glass chimney. Returns (objects, flame height)."""
    font = lathe("oil_lamp", [(0.0, 0.0), (0.04, 0.0), (0.042, 0.004), (0.03, 0.01), (0.016, 0.018), (0.014, 0.03), (0.02, 0.036), (0.04, 0.045),
                              (0.047, 0.06), (0.044, 0.075), (0.03, 0.085), (0.02, 0.088), (0.02, 0.096), (0.027, 0.098), (0.028, 0.108),
                              (0.025, 0.109), (0.025, 0.101), (0.008, 0.101), (0.008, 0.114), (0.0, 0.114)], segments=18)
    knob_rod = cylinder("wick_rod", 0.0018, 0.03, loc=(0.03, 0, 0.093), rot=(0, 90, 0), verts=5)
    knob = cylinder("wick_wheel", 0.007, 0.003, loc=(0.045, 0, 0.093), rot=(0, 90, 0), verts=10)
    chim = lathe("chimney", [(0.0255, 0.104), (0.031, 0.122), (0.034, 0.142), (0.031, 0.165), (0.024, 0.19), (0.021, 0.215), (0.0205, 0.262)], segments=16)
    soot = lathe("chimney_soot", [(0.0198, 0.2625), (0.02, 0.253)], segments=16)       # smoked band inside the top of the glass
    flame = lathe("lamp_flame", [(0.0, 0.114), (0.006, 0.118), (0.0085, 0.128), (0.006, 0.142), (0.0025, 0.154), (0.0, 0.16)], segments=8)
    for o in (font, knob_rod, knob):
        assign(o, M("brass"))
    assign(chim, M("glass"))
    assign(soot, M("iron"))
    paint(soot, grime=1.0, mode="set")
    assign(flame, M("flame"))
    return [font, knob_rod, knob, chim, soot, flame], 0.136


# ------------------------------------------------------------------------------------------------ build
def build(p, rng):
    R = p.diameter / 2.0
    H = p.height
    wr = p.wear
    seed = int(p.seed)
    wrought = p.base == "scroll"
    mats = {}
    makers = {
        "iron": lambda: mat("iron_wrought", rough=0.75, grime=0.45, wear=1.4) if wrought else mat("iron_painted", p.paint, rough=0.65, grime=0.45),
        "marble": lambda: mat("stone_smooth", p.marble, rough=0.2, grime=1.1, wear=3.0),
        "wood": lambda: mat("wood_oak", p.stain),
        "brass": lambda: mat("brass", rough=1.45),
        "cloth": lambda: (mat("canvas", p.cloth_color, stripe=p.cloth_stripe, tint2=p.cloth_color2, wear=0.3) if p.cloth_stripe > 0.005
                          else mat("canvas", p.cloth_color, wear=0.3)),
        "ceramic": lambda: mat("ceramic_glazed", "#f8f5ee", rough=0.45, grime=1.0),
        "glass": lambda: mat("glass_clear", "#ffffff", opacity=0.05, rough=0.5),
        "bottle": lambda: mat("ceramic_glazed", "#102c1c", rough=0.35),
        "wine": lambda: mat("glass_clear", "#5a0e1c", opacity=0.93),
        "wax": lambda: mat("wax", "#f6eedc", glow=0.3),
        "wax_hot": lambda: mat("wax", "#ffe9c4", glow=1.3),
        "leaf": lambda: mat("leaf", "#5f8a45"),
        "petal": lambda: mat("leaf", p.flower),
        "flame": lambda: mat("emissive", "#ffb45e", glow=9.0),
    }

    def M(k):
        if k not in mats:
            mats[k] = makers[k]()
        return mats[k]

    tt = {"marble": 0.024, "wood": 0.028, "cloth": 0.014}[p.top]
    zu = H - tt                                           # underside of the top

    # ---- layout of what stands on the top (needed first: the marble keeps the rings of earlier cups)
    lay = rng.fork("layout")
    n = int(p.places)
    a0 = {1: -90.0, 2: 0.0, 3: 90.0}[n] + lay.u(-8, 8)
    place_angles = [a0 + k * 360.0 / n for k in range(n)]
    aux_angles = {1: [a0 + 180, a0 + 95, a0 - 95], 2: [a0 + 90, a0 - 90], 3: [a0 + 60, a0 + 180, a0 + 300]}[n]
    rp = min(0.6 * R, R - 0.085)
    ra = min(0.36 * R, 0.16)

    # ---- base
    iron_objs, gilt_objs = [], []
    {"cabriole": _base_cabriole, "scroll": _base_scroll, "disc": _base_disc}[p.base](R, H, zu, iron_objs, gilt_objs)
    rust = _rust_mask(0.2 + 0.2 * wr, 0.42 + 0.3 * wr)
    for o in iron_objs:
        assign(o, M("iron"))
        if wrought:
            paint(o, special=rust, wear=mask_mul(facing((0, 0, 1), 1.0), noise_mask(0.06, seed=seed + 5, lo=0.3, hi=0.7), 0.35 + 0.5 * wr))
        else:
            paint(o, special=rust, wear=mask_mul(height_ramp(0.0, 0.3), noise_mask(0.03, seed=seed + 5, lo=0.5, hi=0.8), 0.3 + 0.7 * wr))
    for o in gilt_objs:
        assign(o, M("brass"))
        paint(o, wear=0.25)

    # ---- top
    zs = H                                                # surface things stand on
    if p.top == "marble":
        sr = rng.fork("marble")
        stain_at = []
        if p.setting != "none":
            for a in place_angles:
                stain_at.append(_rz(rp * sr.u(0.45, 0.8), 0, 0, a + sr.u(22, 48) * (1 if sr.chance(0.5) else -1))[:2])
        stain_at.append(_rz(R * sr.u(0.25, 0.6), 0, 0, sr.u(0, 360))[:2])
        veins, stains = _marble_bands(sr, R - 0.004, R, wr, stain_at)
        slab = _marble_top(R, H, veins + stains)
        # closed solid under the mask-carrying face: an open two-sheet slab drops out of the sun shadow on small tops
        under = lathe("marble_under", [(0.0, zu), (R - 0.007, zu), (R - 0.005, zu + 0.005), (R - 0.0045, H - 0.002), (0.0, H - 0.002)], segments=32)
        dark = _lum(p.marble) < 0.5
        vm, sm = _band_mask(veins, H), _band_mask(stains, H)
        etch = noise_mask(0.14, seed=seed + 7, lo=0.45, hi=0.85)
        ke = 0.16 + 0.3 * wr
        for o in (slab, under):
            uv_box(o)
            assign(o, M("marble"))
        if dark:
            paint(slab, wear=vm, grime=lambda co, nrm: np.maximum(sm(co, nrm), etch(co, nrm) * ke))
        else:
            paint(slab, grime=lambda co, nrm: np.maximum(np.maximum(vm(co, nrm), sm(co, nrm)), etch(co, nrm) * ke))
        rim = []
        _brass_rim(R, H, tt, max(6, int(round(2 * pi * R / 0.12))), rim)
    elif p.top == "wood":
        r_ = rng.fork("boards")
        Rw = R - 0.003
        nb = max(3, int(round(2 * Rw / 0.125)))
        ws = [r_.u(0.85, 1.15) for _ in range(nb)]
        ws[0] *= 0.8
        ws[-1] *= 0.8
        tot = sum(ws)
        x = -Rw
        for k in range(nb):
            w = ws[k] / tot * 2 * Rw
            x0, x1 = x + (0.0011 if k else 0.0), x + w - (0.0011 if k < nb - 1 else 0.0)
            x += w
            b = extrude_profile(f"board{k}", _board_poly(x0, x1, Rw, 56), tt - 0.001, loc=(0, 0, zu + r_.u(-0.0006, 0.0004)), bevel=0.0014, segments=1)
            uv_grain(b, axis="Y")
            assign(b, M("wood"))
            paint(b, wear=mask_mul(facing((0, 0, 1)), noise_mask(0.11, seed=seed + 9, lo=0.4, hi=0.8), 0.3 + 0.7 * wr),
                  grime=mask_mul(noise_mask(0.2, seed=seed + 3, lo=0.45, hi=0.85), 0.3 + 0.4 * wr))
        zs = H - 0.0008
        rim = []
        _brass_rim(R, H, tt, max(8, nb * 2), rim)
    else:
        under = lathe("iron_top", [(0.0, zu), (R - 0.004, zu), (R - 0.001, zu + 0.003), (R - 0.001, H - 0.002), (R - 0.006, H - 0.0005)], segments=40)
        assign(under, M("iron"))
        cloth, zs, zhem = _cloth(p, rng.fork("cloth"), R, H)
        assign(cloth, M("cloth"))
        hem = mask_mul(height_ramp(zhem - 0.15, zhem + 0.12), noise_mask(0.09, seed=seed + 1, lo=0.3, hi=0.8), 0.2 + 0.5 * wr)

        def cloth_grime(co, nrm):                         # the unlit inside of the skirt stays dark
            rr = np.maximum(np.linalg.norm(co[:, :2], axis=1), 1e-6)
            inward = -(nrm[:, 0] * co[:, 0] + nrm[:, 1] * co[:, 1]) / rr
            return np.maximum(hem(co, nrm), np.clip(inward * 4.0, 0.0, 1.0) * (co[:, 2] < zs - 0.02))

        paint(cloth, grime=cloth_grime, special=mask_mul(facing((0, 0, 1)), 0.2 + 0.4 * wr))
        rim = []
    for o in rim:
        assign(o, M("brass"))
        paint(o, wear=mask_mul(facing((0, 0, 1)), 0.3 + 0.4 * wr), special=mask_mul(noise_mask(0.04, seed=seed + 6, lo=0.55, hi=0.85), facing((0, 0, -1)), 0.5 * wr))

    # ---- what stands on it
    if p.setting == "coffee":
        for k, a in enumerate(place_angles):
            x, y, _ = _rz(rp, 0, 0, a + lay.u(-4, 4))
            _put(_cup_set(M), x, y, zs, a + 90 + lay.u(-25, 25))
        x, y, _ = _rz(min(ra + 0.045, R - 0.06), 0, 0, aux_angles[0] + lay.u(-10, 10))
        _put(_vase(M, rng.fork("vase"), False), x, y, zs, lay.u(0, 360))
        x, y, _ = _rz(ra + 0.005, 0, 0, aux_angles[1] + lay.u(-12, 12))
        _put(_creamer(M), x, y, zs, lay.u(0, 360))
    elif p.setting == "wine":
        for k, a in enumerate(place_angles):
            x, y, _ = _rz(rp + 0.012, 0, 0, a + lay.u(-14, 14))
            _put(_wine_glass(M, lay.u(0.25, 1.0) if k < 2 else 0.0), x, y, zs, lay.u(0, 360))
        ab = aux_angles[0] + lay.u(-10, 10)
        x, y, _ = _rz(ra + 0.03, 0, 0, ab)
        _put(_bottle(M, ab + lay.u(-50, 50)), x, y, zs, 0.0)
        x, y, _ = _rz(ra + 0.012, 0, 0, aux_angles[1] + lay.u(-12, 12))
        _put(_vase(M, rng.fork("vase"), True), x, y, zs, lay.u(0, 360))
    if p.lamp != "none":
        lx, ly, _ = _rz(lay.u(0.0, 0.05) * R, 0, 0, aux_angles[-1] + (0 if n == 1 else 90) + lay.u(-30, 30))
        if p.lamp == "candle":
            objs, zf, hc = _candle(M, rng.fork("candle"))
            _put(objs, lx, ly, zs, lay.u(0, 360))
            light("candle", (lx, ly, zs + zf), color="#ffb964", intensity=0.25 * p.glow, range=3.0, flicker=0.5, shadow=True,
                  shadow_near=0.045)
            fx("flame", "flame", (lx, ly, zs + zf - 0.012), size=0.03)
            spill, fl = 0.5, 0.5
        else:
            objs, zf = _oil_lamp(M)
            _put(objs, lx, ly, zs, lay.u(0, 360))
            light("oil_lamp", (lx, ly, zs + zf), color="#ffbf6a", intensity=0.7 * p.glow, range=4.0, flicker=0.15, shadow=True, shadow_near=0.06)
            spill, fl = 0.85, 0.15
        # the glow that gets past the top: warm spill on the pedestal and the paving around the table
        # (on the axis, inside the pedestal, so it rakes the legs and the foot without burning out the column)
        light("spill", (0.0, 0.0, zu - 0.12), color="#ffb964", intensity=spill * (0.8 if p.top == "cloth" else 1.0) * p.glow, range=2.4,
              flicker=fl, shadow=False)
        # and a faint bounce from the front so the pedestal is not a black silhouette under the lit top
        light("bounce", (0.0, -(0.5 * R + 0.06), zu - 0.22), color="#ffb964", intensity=0.07 * spill * p.glow, range=1.2,
              flicker=fl, shadow=False)

    # ---- sockets: free spot on the top, and where chairs go (the socket's -Y looks at the table)
    socket("top", "attach", (0, 0, zs), role="prop_top")
    for k, a in enumerate(place_angles):
        sx, sy, _ = _rz(R + 0.3, 0, 0, a)
        socket(f"seat{k}", "attach", (sx, sy, 0.0), rot=(0, 0, a - 90), role="seat")
