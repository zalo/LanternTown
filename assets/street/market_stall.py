"""Timber-framed market stall. Two sledge sole plates carry four chamfered posts with knee braces, an eave
beam and a higher back beam; round rafter poles are rope-lashed across them and a striped canvas awning
is stretched over the poles (sagging between them), with a scalloped, piped valance on three sides and
guy ropes down to cast-iron stall weights. Under it: a plank counter on two or three sawhorse trestles
with a pleated drape tacked to its edge, a boarded back with a shelf dresser and a name board, goods on
counter / shelves / ground and hanging from hooks in the eave beam (fruit, bread, flowers or trinkets),
and one or more glazed lanterns on chains from a purlin.
"""
from tk import *
import numpy as np
from math import sin, cos, tan, radians, degrees, atan2, asin, sqrt, pi, copysign
from mathutils import Vector

META = {"name": "Market stall with awning", "class": "structure", "mount": "ground", "focus": [0.0, -0.45, 1.3]}
PARAMS = {
    "width": {"default": 2.6, "min": 1.8, "max": 3.8, "desc": "frame width, post to post (m)"},
    "depth": {"default": 1.7, "min": 1.3, "max": 2.2, "desc": "frame depth, post to post (m)"},
    "eave_height": {"default": 2.2, "min": 2.05, "max": 2.5, "desc": "height of the front (eave) beam (m)"},
    "pitch": {"default": 16.0, "min": 8.0, "max": 28.0, "desc": "awning slope, rising to the back (deg)"},
    "overhang": {"default": 0.42, "min": 0.2, "max": 0.7, "desc": "awning overhang in front of the posts (m)"},
    "counter_height": {"default": 0.86, "min": 0.74, "max": 0.96, "desc": "counter top height (m)"},
    "canvas": {"default": "#9a2f25", "desc": "awning canvas colour"},
    "canvas2": {"default": "#e6dcc2", "desc": "awning stripe colour"},
    "stripe": {"default": 0.16, "min": 0.0, "max": 0.35, "desc": "stripe width (m), 0 = plain canvas"},
    "drape": {"default": "#2c4660", "desc": "colour of the counter drape and the valance piping"},
    "frame_stain": {"default": "#7c5a3a", "desc": "stain colour of the oak frame"},
    "valance": {"default": "scallop", "choices": ["scallop", "zigzag", "tabs", "plain"], "desc": "valance edge style"},
    "valance_depth": {"default": 0.21, "min": 0.12, "max": 0.32, "desc": "valance drop (m)"},
    "goods": {"default": "fruit", "choices": ["fruit", "bread", "flowers", "trinkets"], "desc": "what the stall sells"},
    "goods_colour": {"default": "auto", "desc": "main goods colour (#hex): apples / crust / blooms / glaze; auto = by goods"},
    "goods_colour2": {"default": "auto", "desc": "second goods colour (#hex): oranges / wicker / foliage / brass; auto = by goods"},
    "stock": {"default": 0.8, "min": 0.2, "max": 1.0, "desc": "how full the stall is (0.2 nearly sold out .. 1)"},
    "sign_text": {"default": "auto", "desc": "lettering on the name board (auto = by goods, empty = no board)"},
    "lanterns": {"default": 1, "min": 1, "max": 3, "desc": "number of hanging lanterns"},
    "intensity": {"default": 18.0, "min": 8.0, "max": 60.0, "desc": "light per hanging lantern (cd)"},
    "post_lantern": {"default": True, "desc": "small lantern on a wrought-iron bracket on a front post"},
    "guy_ropes": {"default": True, "desc": "guy ropes to iron stall weights at the four corners"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: canvas sag and fade, grime, warped boards"},
}
VARIANTS = {
    "bakery_green": {"goods": "bread", "width": 2.2, "depth": 1.5, "canvas": "#2f5a43", "canvas2": "#e9dfc4", "stripe": 0.22,
                     "drape": "#7a2a22", "valance": "tabs", "pitch": 12.0, "guy_ropes": False, "seed": 4},
    "florist_wide": {"goods": "flowers", "width": 3.6, "depth": 1.9, "canvas": "#c79a3a", "canvas2": "#efe6cf", "stripe": 0.12,
                     "drape": "#3d5a3a", "valance": "zigzag", "valance_depth": 0.26, "pitch": 20.0, "lanterns": 2,
                     "frame_stain": "#4a3526", "seed": 9},
    "curio_blue": {"goods": "trinkets", "width": 2.9, "depth": 1.6, "canvas": "#27456b", "canvas2": "#d9cfb4", "stripe": 0.3,
                   "drape": "#6b2436", "valance": "scallop", "valance_depth": 0.28, "pitch": 24.0, "eave_height": 2.1,
                   "overhang": 0.55, "lanterns": 2, "age": 0.7, "seed": 15},
    "old_plain_small": {"goods": "fruit", "width": 1.9, "depth": 1.35, "canvas": "#b9a57e", "stripe": 0.0, "drape": "#5b4a36",
                        "valance": "plain", "valance_depth": 0.16, "pitch": 10.0, "stock": 0.45, "age": 1.0,
                        "frame_stain": "#8a6a48", "goods_colour": "#7d9a3a", "seed": 23},
}

_AUTO = {"fruit": ("#b5382a", "#e39a2b", "FRUIT"), "bread": ("#d9a763", "#d8c8a4", "BAKER"),
         "flowers": ("#d8558a", "#8fae6a", "FLOWERS"), "trinkets": ("#2f5f93", "#ffffff", "CURIOS")}


# ------------------------------------------------------------------------------------------- helpers
def _timber(name, a, b, w, t, m, bevel=0.005, warp=0.0, rng=None, roll=0.0, light=False):
    """Sawn timber from a to b; w = section across (local Y), t = section along local Z."""
    a, b = Vector(a), Vector(b)
    d = b - a
    L = d.length
    rot = (roll, -degrees(asin(max(-1.0, min(1.0, d.z / L)))), degrees(atan2(d.y, d.x)))
    if light:
        return _board(name, L, w, t, tuple((a + b) / 2), rot, m, bevel=bevel)
    o = plank(name, L, w, t, loc=tuple((a + b) / 2), rot=rot, bevel=bevel, warp=warp, rng=rng)
    assign(o, m)
    return o


def _pole(name, a, b, r, m, rng, bow=0.004, sides=10, taper=0.0):
    """Peeled round pole: slightly bowed, optional taper."""
    a, b = Vector(a), Vector(b)
    pts = []
    for i in range(5):
        q = a.lerp(b, i / 4)
        if 0 < i < 4:
            q = q + Vector((rng.jitter(0, bow), 0, rng.jitter(0, bow * 0.3)))
        pts.append(tuple(q))
    o = sweep(name, pts, r, sides=sides, smooth_path=False, scale_fn=(lambda t: 1.0 - taper * t) if taper else None)
    assign(o, m)
    return o


def _wraps(name, c, u, v, a, b, turns, m, r=0.0042, sq=0.5, pitch=0.0095, seg=6, sides=4):
    """Rope wound `turns` times round a (rounded-rectangular) section: loops in the plane (u, v)."""
    c, u, v = Vector(c), Vector(u).normalized(), Vector(v).normalized()
    w = u.cross(v)
    n = max(3, int(turns * seg))
    pts = []
    for i in range(n + 1):
        th = 2 * pi * i / seg
        cu, sv = cos(th), sin(th)
        q = c + u * (a * copysign(abs(cu) ** sq, cu)) + v * (b * copysign(abs(sv) ** sq, sv)) + w * (pitch * (i / seg - turns / 2))
        pts.append(tuple(q))
    o = sweep(name, pts, r, sides=sides, smooth_path=False)
    assign(o, m)
    return o


def _resample2d(pts, n):
    P = np.array(pts, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n)
    Q = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9
    return Q, T, t


def _board(name, L, w, t, loc, rot, m, bevel=0.003):
    """Light sawn board along local X (one-segment chamfer) with grain UVs."""
    o = box(name, (L, w, t), loc=loc, rot=rot, bevel=bevel, segments=1)
    uv_grain(o, "X")
    assign(o, m)
    return o


class _Bank:
    """Collects instance matrices per base mesh; flush() stamps each into one mesh."""

    def __init__(self):
        self.base, self.mats = {}, {}

    def put(self, key, maker, M):
        if key not in self.base:
            self.base[key] = maker()
            self.mats[key] = []
        self.mats[key].append(M)

    def flush(self, seed):
        out = {}
        for i, (k, b) in enumerate(self.base.items()):
            out[k] = instances_to_mesh("g_" + k, b, self.mats[k], seed=seed + i, keep_base=False)
        return out


def _crate(m, cw, cd, chh):
    t = 0.009
    parts = []
    for i in range(3):
        parts.append(box("cb", (cw - 0.004, (cd - 0.03) / 3, t), loc=(0, (i - 1) * (cd - 0.01) / 3, t / 2)))
    for sy in (-1, 1):
        for z in (0.04, chh - 0.032):
            parts.append(box("cs", (cw, t, 0.055), loc=(0, sy * (cd / 2 - t / 2), z)))
    for sx in (-1, 1):
        parts.append(box("ce", (0.014, cd - 2 * t, chh - 0.004), loc=(sx * (cw / 2 - 0.007), 0, chh / 2 + 0.002), bevel=0.002, segments=1))
        for sy in (-1, 1):
            parts.append(box("cc", (0.022, 0.022, chh - 0.012), loc=(sx * (cw / 2 - 0.025), sy * (cd / 2 - t - 0.011), chh / 2)))
    for o in parts:
        assign(o, m)
    return join(parts, "crate")


def _lathe_c(name, prof, seg, m, dz=0.0, sm=80.0):
    o = lathe(name, prof, segments=seg, smooth_angle=sm)
    if dz:
        transform_verts(o, lambda c: Vector((c.x, c.y, c.z + dz)))
    assign(o, m)
    return o


def _lantern(idx, hook, drop, lw, iron, glass, flame_m, intensity, shadow=False, rng_=7.5):
    """Square glazed lantern on a chain; everything swings as one part about the hook."""
    x, y, z = hook
    o = []
    o.append(torus("l_eye", 0.013, 0.003, loc=(x, y, z - 0.013), rot=(90, 0, 0), major_seg=10, minor_seg=5))
    zt = z - drop
    o.append(chain("l_chain", [(x, y, z - 0.026), (x, y, (z - 0.026 + zt + 0.03) / 2), (x, y, zt + 0.03)], link_len=0.054, wire=0.0036, sides=4))
    o.append(torus("l_ring", 0.02, 0.0035, loc=(x, y, zt + 0.012), rot=(90, 0, 0), major_seg=12, minor_seg=5))
    hw = lw / 2
    bh = lw * 1.4
    q = 1.4142
    zr = zt - 0.1                        # eaves of the lantern roof
    o.append(lathe("l_roof", [(0.0, -0.004), (hw * q * 1.22, -0.004), (hw * q * 1.24, 0.002), (hw * q * 0.78, 0.036), (hw * q * 0.42, 0.054),
                              (hw * q * 0.4, 0.07), (hw * q * 0.52, 0.074), (hw * q * 0.2, 0.09), (0.008, 0.096), (0.0, 0.1)],
                   segments=4, loc=(x, y, zr), rot=(0, 0, 45), smooth_angle=20))
    zb = zr - bh
    o.append(lathe("l_base", [(0.0, -0.03), (0.006, -0.028), (0.012, -0.018), (hw * q * 0.5, -0.014), (hw * q * 1.12, -0.008), (hw * q * 1.14, 0.0),
                              (hw * q * 1.05, 0.006), (0.0, 0.006)], segments=4, loc=(x, y, zb), rot=(0, 0, 45), smooth_angle=20))
    for sx in (-1, 1):
        for sy in (-1, 1):
            o.append(box("l_bar", (0.009, 0.009, bh), loc=(x + sx * hw, y + sy * hw, zb + bh / 2), bevel=0.001, segments=1))
    for zz in (zb + 0.012, zr - 0.01):
        for s in (-1, 1):
            o.append(box("l_rail", (lw, 0.007, 0.012), loc=(x, y + s * hw, zz), bevel=0.001, segments=1))
            o.append(box("l_rail", (0.007, lw, 0.012), loc=(x + s * hw, y, zz), bevel=0.001, segments=1))
    o.append(cylinder("l_font", 0.022, 0.03, loc=(x, y, zb + 0.021), verts=10, bevel=0.003, segments=1))
    o.append(cylinder("l_burner", 0.008, 0.02, loc=(x, y, zb + 0.046), verts=8))
    for ob in o:
        assign(ob, iron)
    g = []
    # CLEARANCE RULE (depth fighting): glowing glass is ONE sheet (materials are double sided - a 2 mm box is two sheets
    # that fight), on the centre plane of the rails and cropped to the opening between them, so no iron lies over it
    for s in (-1, 1):
        g.append(lod_faces(box("l_glass", (lw - 0.008, 0.002, bh - 0.033), loc=(x, y + s * (hw - 0.001), zb + bh / 2 + 0.001)),
                           keep=(0, s, 0), above=-1))
        g.append(lod_faces(box("l_glass", (0.002, lw - 0.008, bh - 0.033), loc=(x + s * (hw - 0.001), y, zb + bh / 2 + 0.001)),
                           keep=(s, 0, 0), above=-1))
    for ob in g:
        assign(ob, glass)
    fl = lathe("l_flame", [(0.0, 0.0), (0.008, 0.006), (0.012, 0.018), (0.008, 0.038), (0.0, 0.058)], segments=8, loc=(x, y, zb + 0.056))
    assign(fl, flame_m)
    part(o + g + [fl], f"lantern{idx}", pivot=hook, motion="swing", amp=2.5, speed=0.3)
    light(f"lantern{idx}", (x, y, zb + 0.08), color="#ffbf6a", intensity=intensity, range=rng_, flicker=0.14,
          shadow=shadow, shadow_near=lw * 1.25)
    return zb


# --------------------------------------------------------------------------------------------- build
def build(p, rng):
    W, D, He, age, stock = p.width, p.depth, p.eave_height, p.age, p.stock
    phi = radians(p.pitch)
    tp, cph = tan(phi), cos(phi)
    ps = 0.075                               # post section
    hx = W / 2 - ps / 2
    yf, yb = -D / 2 + ps / 2, D / 2 - ps / 2
    Hb = He + (yb - yf) * tp                 # top of the back beam
    bw, bh = 0.065, 0.09                     # beam section
    rp = 0.023                               # rafter pole radius
    ch = p.counter_height
    c1 = _AUTO[p.goods][0] if p.goods_colour in ("", "auto") else p.goods_colour
    c2 = _AUTO[p.goods][1] if p.goods_colour2 in ("", "auto") else p.goods_colour2

    oak = mat("wood_oak", p.frame_stain)
    pine = mat("wood_weathered")
    canvas = mat("canvas", p.canvas, stripe=p.stripe, tint2=p.canvas2) if p.stripe > 0.02 else mat("canvas", p.canvas)
    cloth = mat("canvas", p.drape)
    rope_m = mat("rope")
    iron = mat("iron_wrought")
    glass = mat("glass_lantern", glow=0.4)
    flame_m = mat("emissive", "#ffb45e", glow=3.5)

    bank = _Bank()
    ropes = []

    # ================================================================ frame (stained oak)
    rf = rng.fork("frame")
    fr = []
    kb = 0.36
    for sx in (-1, 1):
        x = sx * hx
        fr.append(_timber("sole", (x, yf - 0.17, 0.035), (x, yb + 0.17, 0.035), 0.095, 0.07, oak, bevel=0.008, warp=0.002, rng=rf))
        fr.append(_timber("post_f", (x, yf, 0.07), (x, yf, He - bh), ps, ps, oak, bevel=0.007, warp=0.003, rng=rf))
        fr.append(_timber("post_b", (x, yb, 0.07), (x, yb, Hb - bh), ps, ps, oak, bevel=0.007, warp=0.003, rng=rf))
        fr.append(_timber("side_rail", (x, yf + ps / 2, ch + 0.1), (x, yb - ps / 2, ch + 0.1), 0.04, 0.07, oak, warp=0.002, rng=rf))
        for (y, H) in ((yf, He), (yb, Hb)):
            fr.append(_timber("knee", (sx * (hx - 0.025), y, H - bh - kb), (sx * (hx - kb), y, H - bh - 0.022), 0.045, 0.05, oak))
        fr.append(_timber("foot_brace", (x, yf + 0.32, 0.08), (x, yf + 0.025, 0.4), 0.04, 0.05, oak))
        fr.append(_timber("foot_brace", (x, yb - 0.32, 0.08), (x, yb - 0.025, 0.4), 0.04, 0.05, oak))
    ext = 0.17
    fr.append(_timber("beam_f", (-hx - ext, yf, He - bh / 2), (hx + ext, yf, He - bh / 2), bw, bh, oak, bevel=0.007, warp=0.003, rng=rf))
    fr.append(_timber("beam_b", (-hx - ext, yb, Hb - bh / 2), (hx + ext, yb, Hb - bh / 2), bw, bh, oak, bevel=0.007, warp=0.003, rng=rf))

    # round rafter poles lashed across the two beams, cantilevering to carry the front overhang
    n_r = max(3, int(round(2 * hx / 0.62)) + 1)
    rxs = [-hx + 2 * hx * i / (n_r - 1) for i in range(n_r)]
    bay = 2 * hx / (n_r - 1)
    oh_f, oh_b = p.overhang, 0.14
    y_e, y_k = yf - oh_f, yb + oh_b

    def zraf(y):
        return He + rp / cph + (y - yf) * tp

    for i, x in enumerate(rxs):
        e0, e1 = rf.u(0.0, 0.03), rf.u(0.0, 0.05)
        fr.append(_pole(f"rafter{i}", (x, y_e + 0.012, zraf(y_e + 0.012)), (x, y_k + 0.02 + e1, zraf(y_k + 0.02 + e1)), rp * rf.vary(1.0, 0.06), oak, rf, taper=0.12))
        for (y, top) in ((yf, He), (yb, Hb)):
            zc = (zraf(y) + rp + top - bh) / 2
            hb = (zraf(y) + rp - (top - bh)) / 2 + 0.004
            for k, d in enumerate(((1, 1, 0), (1, -1, 0))):
                ropes.append(_wraps("lash", (x, y, zc), d, (0, 0, 1), bw * 0.7071 + 0.008 + k * 0.004, hb + k * 0.003, 1.45, rope_m, sq=0.35))
    # purlin under the rafters: carries the lanterns
    y_l = yf + min(0.3, (yb - yf) * 0.25)
    rpu = 0.02
    z_pu = zraf(y_l) - rp - rpu
    fr.append(_pole("purlin", (-hx - 0.09, y_l, z_pu), (hx + 0.07, y_l, z_pu), rpu, oak, rf, bow=0.002))
    for x in (rxs[0], rxs[-1]) + ((rxs[n_r // 2],) if n_r > 3 else ()):
        ropes.append(_wraps("lash_p", (x, y_l, z_pu + rpu), (1, 1, 0), (0, 0, 1), 0.04, rp + rpu + 0.005, 2.5, rope_m, sq=0.7))

    frame = join(fr, "frame")
    paint(frame, grime=lambda co, n: np.clip(0.6 * np.clip(1 - co[:, 2] / 0.4, 0, 1) + 0.15 * age, 0, 1),
          wear=mask_mul(height_ramp(1.25, 1.6), height_ramp(1.25, 0.9), noise_mask(0.08, seed=p.seed + 1), 0.5),
          special=mask_mul(height_ramp(0.0, 0.3), noise_mask(0.12, seed=p.seed + 2), 0.4 + 0.6 * age))

    # coach bolts at the joints
    def mk_bolt():
        b = box("bolt", (0.017, 0.017, 0.007), loc=(0, 0, 0.0035), bevel=0.0015, segments=1)
        apply_transform(b)
        assign(b, iron)
        return b

    def mk_nail():
        b = lathe("nail", [(0.006, 0.0), (0.0045, 0.0022), (0.0, 0.0032)], segments=5)
        assign(b, iron)
        return b

    for sx in (-1, 1):
        for (y, H) in ((yf, He), (yb, Hb)):
            for (bx, bz) in ((sx * (hx - 0.05), H - bh - kb + 0.03), (sx * (hx - kb + 0.035), H - bh - 0.05), (sx * hx, H - bh / 2)):
                bank.put("bolt", mk_bolt, trs((bx, y - (bw / 2 if bz > H - bh else 0.0225) - 0.0005, bz), (90, 0, 45 * (bz < H - bh))))
        for y in (yf, yb):
            bank.put("bolt", mk_bolt, trs((sx * (hx + 0.0475), y, 0.04), (0, sx * 90, 0)))
            bank.put("bolt", mk_bolt, trs((sx * (hx + 0.0205), y + (0.3 if y < 0 else -0.3), ch + 0.1), (0, sx * 90, 0)))

    # ================================================================ awning canvas
    rc = rng.fork("canvas")
    sag = 0.016 + 0.034 * age
    side_oh = 0.13
    xa = hx + side_oh
    zc0 = He + (2 * rp + 0.013) / cph
    ph = [rc.u(0, 6.283) for _ in range(6)]

    def awn(x, y):
        z = zc0 + (y - yf) * tp
        ax = abs(x)
        if ax >= hx:
            z -= ((ax - hx) / side_oh) ** 1.6 * 0.05
            sx_ = 0.0
        else:
            sx_ = sin(pi * (((x + hx) / bay) % 1.0)) ** 1.3
        if y < yf:
            wy = 0.5 + 1.1 * (yf - y) / oh_f
        elif y > yb:
            wy = 0.5
        else:
            wy = 0.5 + 0.5 * sin(pi * (y - yf) / (yb - yf))
        z -= sag * sx_ * wy
        return z + 0.0025 * sin(x * 19 + ph[0]) * sin(y * 13 + ph[1])

    nu_a = max(28, int(2 * xa / 0.05))
    awning = surface("awning", lambda u, v: (-xa + 2 * xa * u, y_e + (y_k - y_e) * v, awn(-xa + 2 * xa * u, y_e + (y_k - y_e) * v)),
                     nu_a, 10, thickness=0.003)
    assign(awning, canvas)

    vd = p.valance_depth
    style = p.valance

    def vdepth(s):
        if style == "scallop":
            return vd * (0.4 + 0.6 * sqrt(max(0.0, 1 - (2 * s - 1) ** 2)))
        if style == "zigzag":
            return vd * (0.42 + 0.58 * (1 - abs(2 * s - 1)))
        if style == "tabs":
            return vd * (0.5 + 0.5 * clamp((0.5 - abs(s - 0.5) - 0.05) / 0.05))
        return vd * 0.8

    spc = 8
    n_s = max(4, int(round(2 * xa / vd)))

    def val_front(u, v):
        x = -xa + 2 * xa * u
        d = vdepth((u * n_s) % 1.0)
        wav = 0.012 * sin(x * 6.3 + ph[2]) + 0.006 * sin(x * 19 + ph[3])
        return (x, y_e - 0.002 - (0.012 + wav) * v, awn(x, y_e) - 0.001 - d * v)

    Ls = y_k - y_e
    n_ss = max(3, int(round(Ls / vd)))
    cloth_objs = [awning]
    pipes = []
    vf = surface("valance_f", val_front, n_s * spc, 3, thickness=0.003)
    cloth_objs.append(vf)
    pipes.append(sweep("piping_f", [val_front(i / (n_s * spc), 1.0) for i in range(n_s * spc + 1)], 0.0055, sides=4, smooth_path=False))
    for sx in (-1, 1):
        def val_side(u, v, sx=sx):
            y = y_e + Ls * u
            d = vdepth((u * n_ss) % 1.0) * 0.9
            wav = 0.01 * sin(y * 7.1 + ph[4] + sx) + 0.005 * sin(y * 23 + ph[5])
            return (sx * (xa + 0.002 + (0.01 + wav) * v), y, awn(sx * xa, y) - 0.001 - d * v)
        cloth_objs.append(surface("valance_s", val_side, n_ss * spc, 3, thickness=0.003))
        pipes.append(sweep("piping_s", [val_side(i / (n_ss * spc), 1.0) for i in range(n_ss * spc + 1)], 0.0055, sides=4, smooth_path=False))
        # corner tie cords from the canvas down to the beam ends
        for (yy, yb_, H) in ((y_e, yf, He), (y_k, yb, Hb)):
            a = (sx * (xa - 0.01), yy, awn(sx * xa, yy) - 0.02)
            b = (sx * (hx + ext - 0.04), yb_, H - 0.01)
            ropes.append(sweep("tie", catenary(a, b, sag=0.02, n=6), 0.0035, sides=4, smooth_path=False))
            ropes.append(_wraps("tie_w", (sx * (hx + ext - 0.04), yb_, H - bh / 2), (0, 1, 0), (0, 0, 1), bw / 2 + 0.004, bh / 2 + 0.004, 2.0, rope_m, sq=0.3))
    for o in cloth_objs[1:]:
        assign(o, canvas)
    for o in pipes:
        assign(o, cloth)

    lxs = [0.0] if p.lanterns == 1 else [(-1 + 2 * i / (p.lanterns - 1)) * (hx - 0.55) * 0.75 for i in range(int(p.lanterns))]

    def canvas_grime(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        streak = (0.5 + 0.5 * np.sin(x * 31 + 2.1 * np.sin(x * 9.0 + ph[0]))) ** 3
        front = np.clip((y_k - y) / (y_k - y_e), 0, 1)
        g = (0.12 + 0.45 * age) * streak * (0.25 + 0.75 * front)
        f = ((np.clip(x, -hx, hx) + hx) / bay) % 1.0
        g += (0.1 + 0.3 * age) * np.sin(np.pi * f) ** 4                       # dirt pooling in the sags
        for lx in lxs:                                                        # lamp soot
            g = np.maximum(g, 0.6 * (n[:, 2] < 0) * np.exp(-((x - lx) ** 2 + (y - y_l) ** 2) / (2 * 0.2 ** 2)))
        return np.clip(g, 0, 1)

    paint(awning, grime=canvas_grime, special=mask_mul(noise_mask(0.45, seed=p.seed + 5, lo=0.25, hi=0.75), 0.25 + 0.6 * age))
    for o in cloth_objs[1:]:
        paint(o, grime=lambda co, n: np.clip((0.2 + 0.5 * age) * (0.5 + 0.5 * np.sin(co[:, 0] * 23 + co[:, 1] * 19)) ** 2, 0, 1),
              special=mask_mul(noise_mask(0.3, seed=p.seed + 6), 0.2 + 0.5 * age))

    # ================================================================ counter on trestles
    rk = rng.fork("counter")
    xc = hx - ps / 2 - 0.035
    cdp = min(0.74, (yb - yf) * 0.46)
    y0 = yf - 0.10
    y1 = y0 + cdp
    tt = 0.032
    pw = []
    nb = 5
    bwid = cdp / nb
    for i in range(nb):
        pw.append(plank("ctop", 2 * xc + rk.jitter(0, 0.012), bwid - 0.004, tt,
                        loc=(rk.jitter(0, 0.005), y0 + bwid * (i + 0.5), ch - tt / 2 + rk.jitter(0, 0.0012)), warp=0.0015 + 0.003 * age, rng=rk, bevel=0.004))
    n_t = 2 if W < 3.0 else 3
    txs = [(-1 + 2 * i / (n_t - 1)) * (xc - 0.34) for i in range(n_t)]
    zt_ = ch - tt
    for xt in txs:
        pw.append(_timber("tr_beam", (xt, y0 + 0.05, zt_ - 0.035), (xt, y1 - 0.05, zt_ - 0.035), 0.06, 0.07, pine, light=True))
        for yy in (y0 + 0.13, y1 - 0.13):
            for s in (-1, 1):
                pw.append(_timber("tr_leg", (xt + s * 0.035, yy, zt_ - 0.05), (xt + s * 0.25, yy, 0.009), 0.04, 0.055, pine, bevel=0.004, light=True))
            so = 1 if yy > (y0 + y1) / 2 else -1
            pw.append(_timber("tr_tie", (xt - 0.21, yy + so * 0.03, 0.36), (xt + 0.21, yy + so * 0.03, 0.36), 0.02, 0.07, pine, bevel=0.003, light=True))
            for s in (-1, 1):
                bank.put("nail", mk_nail, trs((xt + s * 0.155, yy + so * 0.04, 0.36), (-90 * so, 0, 0)))
    pw.append(_timber("stretcher", (txs[0] - 0.3, y1 - 0.13 + 0.03, 0.405), (txs[-1] + 0.3, y1 - 0.13 + 0.03, 0.405), 0.09, 0.022, pine, warp=0.004, rng=rk))
    sx_, sy_ = txs[-1] - 0.25 + rk.jitter(0, 0.1), min(y1 + 0.3, y1 + (yb - 0.38 - y1) * 0.5)
    pw.append(lathe("stool_seat", [(0.0, 0.43), (0.15, 0.43), (0.158, 0.438), (0.158, 0.458), (0.15, 0.466), (0.06, 0.462), (0.0, 0.458)], segments=14, loc=(sx_, sy_, 0)))
    for k in range(3):
        a = radians(k * 120 + rk.u(0, 60))
        pw.append(_timber("stool_leg", (sx_ + 0.09 * cos(a), sy_ + 0.09 * sin(a), 0.435), (sx_ + 0.17 * cos(a), sy_ + 0.17 * sin(a), 0.006), 0.032, 0.032, pine, bevel=0.006, light=True))
    for o in pw:
        assign(o, pine)
    counter = join(pw, "counter")
    paint(counter, grime=lambda co, n: np.clip(0.7 * np.clip(1 - co[:, 2] / 0.4, 0, 1) + 0.15 * age, 0, 1),
          wear=mask_mul(facing((0, 0, 1)), height_ramp(ch - 0.01, ch - 0.2), noise_mask(0.1, seed=p.seed + 8, lo=0.3, hi=0.8), 0.7))

    # drape: lapped over the counter edge, tacked, hanging in pleats on three sides
    e, cr = 0.007, 0.035
    raw = [(-xc - e, y1 - 0.03)]
    for k in range(5):
        a = radians(180 + 90 * k / 4)
        raw.append((-xc - e + cr + cr * cos(a), y0 - e + cr + cr * sin(a)))
    for k in range(5):
        a = radians(270 + 90 * k / 4)
        raw.append((xc + e - cr + cr * cos(a), y0 - e + cr + cr * sin(a)))
    raw.append((xc + e, y1 - 0.03))
    Ld = 2 * (cdp - 0.03) + 2 * xc
    lam = 0.17
    nu_d = int(Ld / lam) * (6 if W < 3.2 else 5)
    nv_d = 7
    Q, T, S = _resample2d(raw, nu_d + 1)
    dph = [rk.u(0, 6.283) for _ in range(3)]
    z_hem = 0.075

    def drape_fn(u, v):
        i = min(nu_d, int(round(u * nu_d)))
        j = int(round(v * nv_d))
        qx, qy = Q[i]
        nx_, ny_ = T[i][1], -T[i][0]
        if j == 0:
            return (qx - nx_ * 0.032, qy - ny_ * 0.032, ch + 0.0035)
        if j == 1:
            return (qx + nx_ * 0.001, qy + ny_ * 0.001, ch + 0.003)
        vv = (j - 1) / (nv_d - 1)
        s = S[i]
        th = 2 * pi * s / lam + 1.4 * sin(2 * pi * s / (lam * 3.7) + dph[0])
        a = 0.003 + 0.04 * vv ** 0.75
        off = a * (0.5 + 0.5 * sin(th)) + 0.012 * vv
        z = ch + 0.003 - vv * (ch - z_hem) + 0.012 * vv * sin(th * 0.5 + dph[1])
        return (qx + nx_ * off, qy + ny_ * off, z)

    drape = surface("drape", drape_fn, nu_d, nv_d, thickness=0.003)
    assign(drape, cloth)
    paint(drape, grime=lambda co, n: np.clip(0.55 * np.clip(1 - (co[:, 2] - 0.06) / 0.3, 0, 1) + 0.2 * age * (co[:, 2] < ch - 0.02), 0, 1),
          special=mask_mul(noise_mask(0.35, seed=p.seed + 9), 0.3 + 0.5 * age))
    n_tack = int(2 * xc / 0.14)
    for i in range(n_tack + 1):
        bank.put("nail", mk_nail, trs((-xc + 0.03 + (2 * xc - 0.06) * i / n_tack, y0 + 0.012, ch + 0.0045)))

    # ================================================================ boarded back, shelf dresser, name board
    rb = rng.fork("back")
    bk = []
    y_bw = yb + ps / 2 + 0.011            # centre of the back boards
    wall_top = 1.6
    x = -hx - ps / 2
    while x < hx + ps / 2 - 0.02:
        w = min(rb.u(0.12, 0.18), hx + ps / 2 - x)
        if w < 0.05:
            break
        top = wall_top + rb.jitter(0, 0.012) - (rb.u(0.05, 0.25) if rb.chance(0.1 * age) else 0.0)
        bk.append(_board("bboard", top - 0.1, w - 0.005, 0.02, (x + w / 2, y_bw + rb.jitter(0, 0.002 + 0.003 * age), 0.1 + (top - 0.1) / 2),
                         (rb.jitter(0, 0.3 + age), -90 + rb.jitter(0, 0.25), 90), pine))
        for zz in (0.3, 1.45):
            for dx in ((0.03, w - 0.03) if w > 0.09 else (w / 2,)):
                bank.put("nail", mk_nail, trs((x + dx, y_bw + 0.0325, zz + rb.jitter(0, 0.012)), (-90, 0, 0)))
        x += w
    for zz in (0.3, 1.45):
        bk.append(_timber("brail", (-hx + ps / 2, yb, zz), (hx - ps / 2, yb, zz), 0.045, 0.07, pine, warp=0.002, rng=rb))
    for zz in (0.3, 1.45):          # outside ledger battens and Z braces
        bk.append(_board("batten", 2 * hx + ps - 0.03, 0.085, 0.022, (rb.jitter(0, 0.005), y_bw + 0.0215, zz), (90, 0, rb.jitter(0, 0.15)), pine, bevel=0.004))
    nz = 1 if W < 3.0 else 2
    for k in range(nz):
        xa0 = -hx + 0.02 + k * (2 * hx - 0.04) / nz
        xa1 = xa0 + (2 * hx - 0.04) / nz
        if k % 2:
            xa0, xa1 = xa1, xa0
        bk.append(_timber("zbrace", (xa0, y_bw + 0.0215, 0.36), (xa1, y_bw + 0.0215, 1.39), 0.022, 0.075, pine, bevel=0.004, light=True))
        for t_ in (0.06, 0.5, 0.94):
            bank.put("nail", mk_nail, trs((xa0 + (xa1 - xa0) * t_, y_bw + 0.0325, 0.36 + 1.03 * t_), (-90, 0, 0), 1.3))
    # rolled-up rain sheet tied under the back beam
    rr0 = 0.05
    xr0, xr1 = -hx + 0.02, hx - 0.02
    rc_y, rc_z = yb + 0.012, Hb - bh - rr0 - 0.006
    rph = [rb.u(0, 6.283) for _ in range(3)]
    n_ties = max(2, len(rxs) - 1)
    tie_x = [(rxs[i] + rxs[i + 1]) / 2 + rb.jitter(0, 0.05) for i in range(len(rxs) - 1)]

    def roll_fn(u, v):
        x_ = xr0 + (xr1 - xr0) * u
        a = 2 * pi * v
        e = min(u, 1 - u)
        r = rr0 * (1 + 0.12 * sin(x_ * 7 + rph[0]) + 0.07 * sin(3 * a + x_ * 9 + rph[1])) * (0.02 + 0.98 * min(1.0, e / 0.03) ** 0.5)
        pinch = min([abs(x_ - tx) for tx in tie_x])
        r *= 0.86 + 0.14 * min(1.0, pinch / 0.05)
        dz = -0.018 * min(1.0, pinch / 0.25)
        return (x_, rc_y + r * cos(a), rc_z + dz + r * sin(a))

    roll = surface("rain_sheet", roll_fn, max(20, int((xr1 - xr0) / 0.08)), 8, closed_v=True)
    assign(roll, canvas)
    paint(roll, grime=mask_mul(noise_mask(0.1, seed=p.seed + 14), 0.15 + 0.25 * age), special=0.3 + 0.4 * age)
    for tx in tie_x:
        ropes.append(_wraps("roll_tie", (tx, yb + 0.004, Hb - (bh + 2 * rr0) / 2), (0, 1, 0), (0, 0, 1), max(bw / 2, rr0 * 0.9) + 0.004, (bh + 2 * rr0 * 0.9) / 2 + 0.004, 1.6, rope_m, sq=0.6))
    # dresser: cheeks + shelves standing in front of the boards
    sh_d = 0.3
    y_s1 = yb - ps / 2 - 0.004            # back of the shelves
    y_s0 = y_s1 - sh_d
    xs_in = hx - ps / 2 - 0.004
    shelf_z = [0.5, 0.95, 1.36]
    dres_top = shelf_z[-1] + 0.2
    cheeks = [-xs_in + 0.011, xs_in - 0.011] + ([0.0] if W > 3.0 else [])
    for cx in cheeks:
        bk.append(_board("cheek", dres_top - 0.02, sh_d, 0.022, (cx, (y_s0 + y_s1) / 2, 0.02 + (dres_top - 0.02) / 2), (0, -90, 0), pine, bevel=0.004))
    for z in shelf_z:
        for k in range(2):
            bk.append(_board("shelf", 2 * xs_in - 0.05 + rb.jitter(0, 0.01), sh_d / 2 - 0.003, 0.024,
                             (0, y_s0 + sh_d / 4 + k * sh_d / 2, z - 0.012 + rb.jitter(0, 0.001)), (rb.jitter(0, 0.4), 0, 0), pine, bevel=0.004))
        bk.append(_board("shelf_lip", 2 * xs_in - 0.05, 0.014, 0.04, (0, y_s0 - 0.006, z - 0.004), None, pine))
    bk.append(_board("kick", 2 * xs_in - 0.05, 0.02, 0.1, (0, y_s0 + 0.03, 0.05), None, pine))
    for o in bk:
        assign(o, pine)
    sign = p.sign_text
    if sign == "auto":
        sign = _AUTO[p.goods][2]
    sign = sign.strip()
    if sign:
        tsize = min(0.135, (2 * hx - 0.5) / (max(len(sign), 3) * 0.82))
        sl = min(2 * hx + 0.05, len(sign) * tsize * 0.82 + 0.42)
        sz = wall_top + 0.03 + 0.11
        ys = yb - ps / 2 - 0.013
        bk.append(plank("sign_board", sl, 0.21, 0.024, loc=(0, ys, sz), rot=(90 + rb.jitter(0, 0.4), 0, 0), bevel=0.005))
        assign(bk[-1], pine)
        for s in (-1, 1):          # shaped ends
            c = cylinder("sign_end", 0.105, 0.024, loc=(s * sl / 2, ys, sz), rot=(90, 0, 0), verts=16, bevel=0.004, segments=1)
            assign(c, pine)
            bk.append(c)
            bank.put("bolt", mk_bolt, trs((s * (sl / 2 - 0.02), ys - 0.0125, sz), (90, 0, 20)))
        t = text("sign_text", sign, size=tsize, depth=0.004, loc=(0, ys - 0.012, sz), font="serif")
        assign(t, cloth)                 # lettering painted in the stall's drape colour
    back = join(bk, "back")
    paint(back, grime=lambda co, n: np.clip(0.5 * np.clip(1 - co[:, 2] / 0.35, 0, 1) + (0.04 + 0.16 * age), 0, 1),
          special=mask_mul(height_ramp(0.05, 0.6), noise_mask(0.15, seed=p.seed + 11), 0.3 + 0.7 * age, lambda co, n: (n[:, 1] > 0.5) * 1.0))

    # ================================================================ lanterns
    for i, lx in enumerate(lxs):
        hook = (lx, y_l, z_pu - rpu)
        drop = max(0.1, min(0.3, (z_pu - rpu) - (ch + 1.24)))
        _lantern(i, hook, drop, 0.15, iron, glass, flame_m, p.intensity, shadow=(i == 0))
        ropes.append(_wraps("lash_l", (lx, y_l, z_pu), (0, 1, 0), (0, 0, 1), rpu + 0.003, rpu + 0.003, 1.5, rope_m, sq=1.0, pitch=0.02))

    pl_side = 0
    if p.post_lantern:
        rl = rng.fork("post_lantern")
        sx = pl_side = rl.pick([-1, 1])
        px, py, zbar = sx * hx, yf - ps / 2, min(1.8, He - bh - kb - 0.02)
        reach = 0.27
        br = [box("pl_plate", (0.034, 0.005, 0.3), loc=(px, py - 0.0025, zbar - 0.09), bevel=0.0015, segments=1),
              box("pl_bar", (0.012, reach, 0.012), loc=(px, py - reach / 2, zbar), bevel=0.0015, segments=1),
              sweep("pl_stay", bezier_path((px, py - 0.004, zbar - 0.22), (px, py - 0.03, zbar - 0.1), (px, py - reach * 0.55, zbar - 0.1), (px, py - reach * 0.78, zbar - 0.008), n=8),
                    rounded_rect(0.012, 0.005, 0.0015, 1), up=(1, 0, 0), smooth_path=False),
              sweep("pl_scroll", spiral_path(0.006, 0.03, 1.25, n=14, a0_deg=90, plane="YZ", center=(px, py - reach - 0.002, zbar + 0.03), ccw=False),
                    rounded_rect(0.012, 0.005, 0.0015, 1), up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.55 + 0.45 * t)]
        for o in br:
            assign(o, iron)
        brj = join(br, "post_bracket")
        paint(brj, special=mask_mul(noise_mask(0.05, seed=p.seed + 31, lo=0.35, hi=0.75), 0.4 + 0.4 * age))
        for zz in (zbar + 0.03, zbar - 0.21):
            bank.put("bolt", mk_bolt, trs((px, py - 0.005, zz), (90, 0, 0), 0.7))
        _lantern(9, (px, py - reach + 0.045, zbar - 0.006), 0.07, 0.115, iron, glass, flame_m, p.intensity * 0.5, rng_=5.0)

    # balance scales hanging from the purlin over the counter
    if p.goods != "flowers":
        rs = rng.fork("scales")
        ssx = -pl_side if pl_side else rs.pick([-1, 1])
        bx = ssx * hx * 0.58
        if any(abs(bx - lx) < 0.3 for lx in lxs):
            bx = ssx * (hx - 0.3)
        zt_s, zbm, arm = z_pu - rpu, ch + 0.98, 0.2
        tl = rs.pick([-1, 1]) * rs.u(4, 9)
        sc = [torus("sc_hook", 0.012, 0.0028, loc=(bx, y_l, zt_s - 0.012), rot=(90, 0, 0), major_seg=8, minor_seg=4),
              sweep("sc_rod", [(bx, y_l, zt_s - 0.024), (bx, y_l, zbm + 0.075)], 0.0028, sides=4, smooth_path=False),
              box("sc_fork", (0.012, 0.004, 0.085), loc=(bx, y_l - 0.006, zbm + 0.035), bevel=0.001, segments=1),
              box("sc_beam", (2 * arm + 0.03, 0.007, 0.018), loc=(bx, y_l, zbm), rot=(0, tl, 0), bevel=0.002, segments=1),
              box("sc_pointer", (0.005, 0.004, 0.075), loc=(bx - 0.0375 * sin(radians(tl)), y_l, zbm + 0.0375 * cos(radians(tl))), rot=(0, tl, 0))]
        for s_ in (-1, 1):
            ex, ez = bx + s_ * arm * cos(radians(tl)), zbm - s_ * arm * sin(radians(tl))
            zp = ez - 0.3
            sc.append(lathe("sc_pan", [(0.0, 0.0), (0.06, 0.003), (0.092, 0.018), (0.098, 0.03), (0.094, 0.03), (0.06, 0.007), (0.0, 0.004)], segments=12, loc=(ex, y_l, zp)))
            for k in range(3):
                a = radians(90 + k * 120)
                sc.append(sweep("sc_wire", [(ex, y_l, ez - 0.008), (ex + 0.095 * cos(a), y_l + 0.095 * sin(a), zp + 0.03)], 0.0014, sides=3, smooth_path=False))
            if s_ * tl > 0:                # the low pan carries the weights
                for (dx, r_, h_) in ((-0.025, 0.021, 0.034), (0.03, 0.015, 0.025)):
                    sc.append(cylinder("sc_wt", r_, h_, loc=(ex + dx, y_l + rs.jitter(0, 0.015), zp + 0.005 + h_ / 2), verts=8, bevel=0.003, segments=1))
        for o in sc:
            assign(o, iron)
        scj = join(sc, "scales")
        paint(scj, special=mask_mul(noise_mask(0.04, seed=p.seed + 33, lo=0.4, hi=0.8), 0.3 + 0.4 * age))

    # ================================================================ guy ropes and stall weights
    if p.guy_ropes:
        rg = rng.fork("guy")
        wt = []
        for sx in (-1, 1):
            for (yy, H, oy) in ((yf, He, -0.22), (yb, Hb, 0.4)):
                top = (sx * (hx + ext - 0.09), yy, H - bh / 2)
                gx, gy = sx * (hx + ext + rg.u(0.58, 0.72)), yy + oy + rg.jitter(0, 0.05)
                ring_z = 0.168
                ropes.append(_wraps("guy_w", top, (0, 1, 0), (0, 0, 1), bw / 2 + 0.005, bh / 2 + 0.005, 3.0, rope_m, r=0.006, sq=0.3, pitch=0.013))
                pts = catenary((top[0] + sx * 0.02, top[1], top[2] - bh / 2), (gx, gy, ring_z + 0.02), sag=0.03 + 0.03 * age, n=12)
                ropes.append(sweep("guy", pts, 0.006, sides=5, smooth_path=False))
                # wooden tensioner (slide) on the rope
                k = 4
                a, b = Vector(pts[k]), Vector(pts[k + 1])
                d = (b - a).normalized()
                sl_ = _timber("guy_slide", a - d * 0.05, a + d * 0.05, 0.035, 0.016, oak, bevel=0.003)
                # the tail of the rope coming back from the weight to the slide
                ropes.append(sweep("guy_tail", [tuple(a + Vector((0, 0.012, 0.004))), tuple(a.lerp(Vector((gx, gy, ring_z)), 0.5) + Vector((0, 0.02, -0.03))),
                                                (gx, gy + 0.008, ring_z + 0.015)], 0.006, sides=5, step=0.2))
                kn = sphere("guy_knot", 0.017, loc=(gx, gy, ring_z + 0.022), segments=8, rings=5, scale=(1, 1.2, 0.9))
                assign(kn, rope_m)
                ropes.append(kn)
                w = lathe("weight", [(0.0, 0.0), (0.1, 0.0), (0.108, 0.008), (0.108, 0.02), (0.1, 0.028), (0.085, 0.1), (0.078, 0.122), (0.06, 0.134),
                                     (0.03, 0.14), (0.0, 0.14)], segments=8, loc=(gx, gy, 0.0), rot=(0, 0, rg.u(0, 45)), smooth_angle=30)
                wt.append(w)
                wt.append(torus("weight_ring", 0.03, 0.007, loc=(gx, gy, ring_z), rot=(90, 0, rg.u(-30, 30) + 90), major_seg=14, minor_seg=6))
        for o in wt:
            assign(o, iron)
        wts = join(wt, "weights")
        paint(wts, special=mask_mul(noise_mask(0.06, seed=p.seed + 13, lo=0.3, hi=0.7), 0.5 + 0.4 * age), grime=height_ramp(0.0, 0.1))

    # ================================================================ goods
    rgd = rng.fork("goods")
    cw, cd_, chh = 0.46, 0.32, 0.14
    tilt = 15.0
    n_slot = max(2, int((2 * xc - 0.1) / (cw + 0.05)))
    slot_x = [(-(n_slot - 1) / 2 + i) * ((2 * xc - 0.1) / n_slot) for i in range(n_slot)]
    yfr = y0 + 0.05 + cd_ / 2            # front row centre
    ybk = y1 - 0.03 - cd_ / 2            # back row centre
    two_rows = (ybk - yfr) > cd_ * 0.8
    y_sh = (y_s0 + y_s1) / 2
    hang_x = []
    xh = -(hx - kb - 0.12)
    while xh < hx - kb - 0.1:
        if all(abs(xh - lx) > 0.3 for lx in lxs):        # keep the goods out of the lantern's swing
            hang_x.append(xh)
        xh += rgd.u(0.24, 0.32)
    zh0 = He - bh                        # underside of the eave beam

    z_step = ch + chh
    if two_rows:                         # raised step for the back row of the display
        st = []
        Ls_ = 2 * xc - 0.14
        for k in range(2):
            st.append(_board("step", Ls_ + rgd.jitter(0, 0.02), cd_ / 2 - 0.004, 0.022, (rgd.jitter(0, 0.008), ybk - cd_ / 4 + k * cd_ / 2, z_step - 0.011), (rgd.jitter(0, 0.5), 0, 0), pine))
        nbk = max(3, int(Ls_ / 0.6) + 1)
        for k in range(nbk):
            st.append(_board("step_block", 0.05, cd_ - 0.05, chh - 0.022, (-Ls_ / 2 + 0.06 + (Ls_ - 0.12) * k / (nbk - 1), ybk, ch + (chh - 0.022) / 2), None, pine))
        stj = join(st, "step")
        paint(stj, grime=0.2 + 0.2 * age, wear=mask_mul(facing((0, 0, 1)), noise_mask(0.1, seed=p.seed + 17), 0.5))

    def mk_crate():
        return _crate(pine, cw, cd_, chh)

    def mk_hook():
        o = torus("hook", 0.011, 0.0025, rot=(90, 0, 0), major_seg=8, minor_seg=4)
        apply_transform(o)
        assign(o, iron)
        return o

    def hang_string(x, ln, y=None):
        y = yf if y is None else y
        bank.put("hook", mk_hook, trs((x, y, zh0 - 0.011), (0, 0, rgd.u(-40, 40))))
        s = sweep("string", [(x, y, zh0 - 0.02), (x, y, zh0 - ln)], 0.003, sides=4, smooth_path=False)
        assign(s, rope_m)
        ropes.append(s)
        return (x, y, zh0 - ln)

    def tilted_crate(x, y, zbase):
        M = trs((x + rgd.jitter(0, 0.01), y, zbase + (cd_ / 2) * sin(radians(tilt)) + 0.002), (tilt, 0, rgd.jitter(0, 2.0)))
        bank.put("crate", mk_crate, M)
        return M

    def flat_crate(x, y, zbase, rz=0.0, flip=False):
        M = trs((x, y, zbase + (chh if flip else 0.0)), (180 if flip else 0, 0, rz))
        bank.put("crate", mk_crate, M)
        return M

    def pile(key, maker, M, r, full, zb=0.009, layers=3, sq=1.0):
        """Fill a crate (matrix M) with round things of radius r."""
        nx = max(1, int((cw - 0.045) / (2 * r)))
        ny = max(1, int((cd_ - 0.035) / (2 * r)))
        zb = zb + chh - 2 * r * sq * 0.82 - 0.009            # top layer pokes above the rim; hidden lower layers are not built
        for L in range(layers):
            ax, ay = nx - L, ny - L
            if ax < 1 or ay < 1:
                break
            for i in range(ax):
                for j in range(ay):
                    if L > 0 and not rgd.chance(full):
                        continue
                    lx_, ly_ = (i - (ax - 1) / 2) * 2 * r * 1.03, (j - (ay - 1) / 2) * 2 * r * 1.03
                    bank.put(key, maker, M @ trs((lx_ + rgd.jitter(0, r * 0.12), ly_ + rgd.jitter(0, r * 0.12), zb + r * sq * (1 + L * 1.45)),
                                                 (rgd.u(-35, 35), rgd.u(-35, 35), rgd.u(0, 360)), rgd.vary(1.0, 0.09)))

    # riser batten propping the tilted front crates
    def riser():
        o = plank("riser", 2 * xc - 0.12, 0.05, cd_ * sin(radians(tilt)) - 0.004, loc=(0, yfr + cd_ / 2 * cos(radians(tilt)) - 0.04, ch + cd_ * sin(radians(tilt)) / 2 - 0.004), bevel=0.003)
        assign(o, pine)

    # spare crates under the counter (seen from the vendor's side)
    ux = txs[0] + 0.52 if n_t == 2 else (txs[0] + txs[1]) / 2
    flat_crate(ux, (y0 + y1) / 2 + 0.03, 0.0, rz=rgd.jitter(0, 8))
    flat_crate(ux + rgd.jitter(0, 0.02), (y0 + y1) / 2 + 0.03, chh + 0.002, rz=rgd.jitter(0, 8))

    gx0 = xc - 0.3                        # ground stock positions in front of the drape
    ground = [(-gx0, y0 - 0.3), (gx0 - 0.1, y0 - 0.27), (-gx0 + 0.5, y0 - 0.34), (gx0 - 0.62, y0 - 0.33)]
    ground = ground[:max(1, int(round(stock * (4 if W > 2.3 else 2))))]

    if p.goods == "fruit":
        mA, mB = mat("ceramic_glazed", c1), mat("ceramic_glazed", c2)     # glossy skins; 'wax' is translucent and shades per facet

        fseg = 8 if W * D * stock < 4.6 else 6          # very big, fully stocked stalls: lighter fruit

        def mk_apple():
            return _lathe_c("apple", [(0, 0.008), (0.022, 0.0), (0.045, 0.03), (0.036, 0.068), (0.014, 0.078), (0, 0.068)], fseg, mA, dz=-0.039)

        def mk_orange():
            o = sphere("orange", 0.045, segments=fseg, rings=5 if fseg > 6 else 4)
            assign(o, mB)
            return o

        def mk_pumpkin():
            o = lathe("pumpkin", [(0, 0.01), (0.05, 0.0), (0.115, 0.03), (0.14, 0.09), (0.12, 0.15), (0.06, 0.18), (0.02, 0.172), (0, 0.168)], segments=20)
            transform_verts(o, lambda c: Vector((c.x * (1 + 0.07 * cos(10 * atan2(c.y, c.x))), c.y * (1 + 0.07 * cos(10 * atan2(c.y, c.x))), c.z)))
            assign(o, mB)
            st = cylinder("stem", 0.012, 0.05, loc=(0.004, 0, 0.19), rot=(0, 12, 0), verts=6, radius2=0.008)
            assign(st, oak)
            return join([o, st], "pumpkin")

        def mk_onions():
            core = sweep("braid", [(0, 0, 0), (0.004, 0, -0.2), (0, 0, -0.46)], 0.007, sides=5)
            assign(core, rope_m)
            b = _lathe_c("bulb", [(0, -0.03), (0.026, -0.022), (0.035, 0.0), (0.014, 0.03), (0.004, 0.05), (0, 0.052)], 8, mB)
            ms = [trs((0.03 * cos(i * 2.4), 0.03 * sin(i * 2.4), -0.1 - i * 0.04), (55 * sin(i * 2.4), -55 * cos(i * 2.4), 0), 1.05 - 0.02 * (i % 3)) for i in range(9)]
            return join([core, instances_to_mesh("bulbs", b, ms, keep_base=False)], "onions")

        def mk_chili():
            core = sweep("braid", [(0, 0, 0), (0, 0.003, -0.2), (0, 0, -0.4)], 0.005, sides=5)
            assign(core, rope_m)
            b = _lathe_c("chili", [(0, 0), (0.013, 0.012), (0.009, 0.075), (0.002, 0.113), (0, 0.115)], 5, mA)
            ms = [trs((0.012 * cos(i * 2.1), 0.012 * sin(i * 2.1), -0.07 - i * 0.024), (140 * sin(i * 2.1) + 0, -140 * cos(i * 2.1), 0), 1.0 + 0.1 * (i % 2)) for i in range(13)]
            return join([core, instances_to_mesh("chilis", b, ms, keep_base=False)], "chili")

        kinds = [("apple", mk_apple, 0.045), ("orange", mk_orange, 0.045)]
        riser()
        for i, x in enumerate(slot_x):
            k = kinds[i % 2]
            pile(k[0], k[1], tilted_crate(x, yfr, ch), k[2], stock, layers=2)
            if two_rows:
                k = kinds[(i + 1) % 2]
                pile(k[0], k[1], flat_crate(x + rgd.jitter(0, 0.015), ybk, ch + chh + 0.002, rz=rgd.jitter(0, 3)), k[2], stock, layers=2)
        for si, z in enumerate(shelf_z):
            xs = -xs_in + 0.33
            while xs < xs_in - 0.3:
                if si < 2 and rgd.chance(0.35 + 0.6 * stock):
                    k = kinds[(si + int(xs * 7)) % 2]
                    pile(k[0], k[1], flat_crate(xs, y_sh - 0.005, z, rz=rgd.jitter(0, 4)), k[2], stock * 0.6, layers=1)
                    xs += cw + rgd.u(0.05, 0.14)
                elif rgd.chance(0.4 + 0.5 * stock):
                    bank.put("pumpkin", mk_pumpkin, trs((xs - 0.08, y_sh, z), (0, 0, rgd.u(0, 360)), rgd.u(0.7, 0.95)))
                    xs += rgd.u(0.27, 0.36)
                else:
                    xs += 0.3
        for i, x in enumerate(hang_x):
            if rgd.chance(0.3 + 0.7 * stock):
                at = hang_string(x, rgd.u(0.06, 0.16))
                if i % 2:
                    bank.put("onions", mk_onions, trs(at, (0, 0, rgd.u(0, 360)), rgd.vary(1.0, 0.08)))
                else:
                    bank.put("chili", mk_chili, trs(at, (0, 0, rgd.u(0, 360)), rgd.vary(1.0, 0.1)))
        for gi, (gx, gy) in enumerate(ground):
            if gi % 2 == 0:
                rz = rgd.jitter(0, 12)
                flat_crate(gx, gy, 0.0, rz=rz)
                k = kinds[gi // 2 % 2]
                pile(k[0], k[1], flat_crate(gx + 0.01, gy, chh + 0.002, rz=rz + rgd.jitter(0, 6)), k[2], stock, layers=2)
            else:
                bank.put("pumpkin", mk_pumpkin, trs((gx, gy, 0), (0, 0, rgd.u(0, 360)), 1.15))
                bank.put("pumpkin", mk_pumpkin, trs((gx + 0.3, gy + 0.06, 0), (0, 0, rgd.u(0, 360)), 0.9))

    elif p.goods == "bread":
        mA = mat("terracotta", c1)
        mB = mat("wicker", c2)

        def mk_boule():
            o = _lathe_c("boule", [(0, 0), (0.05, 0.002), (0.082, 0.02), (0.09, 0.05), (0.07, 0.085), (0.03, 0.102), (0, 0.105)], 10, mA, dz=-0.05)
            displace_noise(o, 0.004, 0.05, seed=3)
            return o

        def mk_baguette():
            o = _lathe_c("baguette", [(0, -0.3), (0.018, -0.29), (0.032, -0.22), (0.036, 0.0), (0.032, 0.22), (0.018, 0.29), (0, 0.3)], 7, mA)
            displace_noise(o, 0.004, 0.06, seed=5)
            return o

        def mk_miche():
            o = _lathe_c("miche", [(0, 0), (0.1, 0.002), (0.15, 0.022), (0.158, 0.05), (0.13, 0.08), (0.06, 0.095), (0, 0.098)], 14, mA)
            displace_noise(o, 0.005, 0.07, seed=7)
            return o

        def mk_basket():
            o = _lathe_c("basket", [(0, 0), (0.13, 0), (0.15, 0.015), (0.185, 0.2), (0.196, 0.214), (0.186, 0.222), (0.172, 0.2), (0.14, 0.03), (0, 0.026)], 16, mB)
            return o

        def mk_tall_basket():
            o = _lathe_c("tbasket", [(0, 0), (0.1, 0), (0.115, 0.015), (0.14, 0.36), (0.152, 0.376), (0.142, 0.384), (0.128, 0.36), (0.105, 0.03), (0, 0.026)], 16, mB)
            return o

        def mk_rings():
            core = sweep("cord", [(0, 0, 0), (0.0, 0.0, -0.36)], 0.004, sides=4)
            assign(core, rope_m)
            b = torus("ring", 0.07, 0.024, major_seg=12, minor_seg=6)
            assign(b, mA)
            ms = [trs((0.066 * (1 if i % 2 else -1) * 0.25, 0.0, -0.11 - i * 0.075), (90, 0, 28 * (1 if i % 2 else -1)), 1.0 - 0.05 * i) for i in range(4)]
            return join([core, instances_to_mesh("rings", b, ms, keep_base=False)], "rings")

        def mk_pretzels():
            core = sweep("cord", [(0, 0, 0), (0.0, 0.0, -0.4)], 0.004, sides=4)
            assign(core, rope_m)
            pz = [(0.035, 0, -0.07), (0.0, 0.004, -0.02), (-0.045, 0, 0.012), (-0.078, 0, 0.055), (-0.045, 0, 0.1), (0, 0, 0.11), (0.045, 0, 0.1),
                  (0.078, 0, 0.055), (0.045, 0, 0.012), (0.0, -0.004, -0.02), (-0.035, 0, -0.07)]
            b = sweep("pretzel", pz, 0.012, sides=6, scale_fn=lambda t: 0.75 + 0.6 * sin(pi * t))
            assign(b, mA)
            ms = [trs((0, 0.008 * (1 if i % 2 else -1), -0.19 - i * 0.16), (0, 0, 18 * (1 if i % 2 else -1)), 1.0) for i in range(2)]
            return join([core, instances_to_mesh("pretzels", b, ms, keep_base=False)], "pretzels")

        riser()
        for i, x in enumerate(slot_x):
            M = tilted_crate(x, yfr, ch)
            if i % 2 == 0:
                pile("boule", mk_boule, M, 0.092, stock, layers=2, sq=0.55)
            else:
                n_bg = 3 + int(round(4 * stock))
                for k in range(n_bg):
                    row, col = (0, k) if k < 4 else (1, k - 4)
                    bank.put("baguette", mk_baguette, M @ trs((rgd.jitter(0, 0.015) + 0.03, (col - (1.5 if row == 0 else 1.0)) * 0.073, 0.046 + row * 0.062),
                                                              (rgd.u(0, 360), 90, rgd.jitter(0, 4)), (1, 1, rgd.u(0.78, 0.9))))
            if two_rows:
                if i % 2 == 0:
                    tb = trs((x + rgd.jitter(0, 0.03), ybk + 0.01, z_step), (0, 0, rgd.u(0, 360)), 0.9)
                    bank.put("tbasket", mk_tall_basket, tb)
                    for k in range(3 + int(round(4 * stock))):
                        a = rgd.u(0, 360)
                        bank.put("baguette", mk_baguette, tb @ trs((0.05 * cos(radians(a)), 0.05 * sin(radians(a)), 0.33), (rgd.u(4, 13), 0, a + 90), rgd.u(0.85, 1.0)))
                else:
                    for k in range(1 + int(round(2 * stock))):
                        bank.put("miche", mk_miche, trs((x + rgd.jitter(0, 0.02), ybk + rgd.jitter(0, 0.015), ch + chh + 0.003 + k * 0.088), (0, 0, rgd.u(0, 360)), rgd.u(0.9, 1.0)))
        for si, z in enumerate(shelf_z):
            xs = -xs_in + 0.2
            while xs < xs_in - 0.2:
                r_ = rgd.random()
                if not rgd.chance(0.3 + 0.7 * stock):
                    xs += 0.2
                elif r_ < 0.4:
                    bk_ = trs((xs + 0.08, y_sh - 0.005, z), (0, 0, rgd.u(0, 360)), 0.72)
                    bank.put("basket", mk_basket, bk_)
                    for k in range(4):
                        bank.put("boule", mk_boule, bk_ @ trs((0.07 * cos(k * 1.7), 0.07 * sin(k * 1.7), 0.14 + 0.05 * (k == 3)), (rgd.u(-30, 30), rgd.u(-30, 30), rgd.u(0, 360)), 0.95))
                    xs += 0.36
                elif r_ < 0.75:
                    for k in range(rgd.pick([2, 3, 3])):
                        bank.put("boule", mk_boule, trs((xs, y_sh + rgd.jitter(0, 0.02), z + 0.085), (72, 0, 90 + rgd.jitter(0, 8)), (1.0, 1.25, 1.0)))
                        xs += 0.085
                    xs += 0.1
                else:
                    for k in range(rgd.pick([1, 2])):
                        bank.put("miche", mk_miche, trs((xs + 0.1, y_sh, z + k * 0.088), (0, 0, rgd.u(0, 360)), 0.85))
                    xs += 0.38
        for i, x in enumerate(hang_x):
            if rgd.chance(0.3 + 0.7 * stock):
                at = hang_string(x, rgd.u(0.05, 0.12))
                bank.put("rings" if i % 2 else "pretzels", mk_rings if i % 2 else mk_pretzels, trs(at, (0, 0, rgd.jitter(0, 25)), rgd.vary(1.0, 0.06)))
        for gi, (gx, gy) in enumerate(ground[:3]):
            if gi % 2 == 0:
                tb = trs((gx, gy, 0), (0, 0, rgd.u(0, 360)), 1.15)
                bank.put("tbasket", mk_tall_basket, tb)
                for k in range(4 + int(round(4 * stock))):
                    a = rgd.u(0, 360)
                    bank.put("baguette", mk_baguette, tb @ trs((0.05 * cos(radians(a)), 0.05 * sin(radians(a)), 0.32), (rgd.u(4, 13), 0, a + 90), rgd.u(0.85, 1.0)))
            else:
                bk_ = trs((gx, gy, 0), (0, 0, rgd.u(0, 360)), 1.1)
                bank.put("basket", mk_basket, bk_)
                for k in range(6):
                    bank.put("boule", mk_boule, bk_ @ trs((0.085 * cos(k * 1.26), 0.085 * sin(k * 1.26), 0.15 + 0.07 * (k == 5)) if k < 5 else (0, 0, 0.21),
                                                          (rgd.u(-30, 30), rgd.u(-30, 30), rgd.u(0, 360)), 0.95))

    elif p.goods == "flowers":
        mA = mat("leaf", c1)
        mB = mat("hedge", c2)

        def mk_tub():
            o = _lathe_c("tub", [(0, 0.0), (0.1, 0.0), (0.13, 0.2), (0.121, 0.203), (0.118, 0.19), (0, 0.19)], 8, pine, sm=30)
            uv_grain(o, "Z")
            h = [lathe("hoop", [(rr, z0 - 0.012), (rr + 0.005, z0 - 0.01), (rr + 0.005, z0 + 0.01), (rr, z0 + 0.012)], segments=8)
                 for (rr, z0) in ((0.1065, 0.045), (0.124, 0.16))]
            for b in h:
                assign(b, iron)
            return join([o] + h, "tub")

        def mk_foliage():
            o = sphere("foliage", 0.15, segments=9, rings=5, scale=(1, 1, 0.85))
            apply_transform(o)
            displace_noise(o, 0.03, 0.09, seed=11)
            assign(o, mB)
            return o

        def mk_bloom(m=mA, n="bloom"):
            o = _lathe_c(n, [(0, 0), (0.03, 0.018), (0.036, 0.032), (0.014, 0.034), (0, 0.022)], 5, m, sm=25)
            return o

        def mk_bloom2():
            o = mk_bloom(canvas, "bloom2")
            uv_transform(o, scale=(0.0, 0.0))        # one texel per bloom: each takes one of the awning's two stripe colours
            return o

        def mk_spray():
            o = cone("spray", 0.03, 0.004, 0.34, loc=(0, 0, 0.17), verts=6)
            apply_transform(o)
            assign(o, mB)
            b = mk_bloom()
            ms = [trs((0.028 * cos(i * 2.4) * (1 - i / 14), 0.028 * sin(i * 2.4) * (1 - i / 14), 0.1 + i * 0.024), (70 * sin(i * 2.4), -70 * cos(i * 2.4), 0), 0.85 - 0.035 * i) for i in range(0, 10, 3)]
            return join([o, instances_to_mesh("sb", b, ms, keep_base=False)], "spray")

        def mk_bunch():       # dried bunch hung upside-down
            o = cone("stems", 0.012, 0.07, 0.3, loc=(0, 0, -0.17), verts=7)
            apply_transform(o)
            assign(o, mB)
            tie = torus("btie", 0.016, 0.005, loc=(0, 0, -0.045), major_seg=8, minor_seg=4)
            assign(tie, rope_m)
            b = mk_bloom()
            ms = []
            for i in range(9):
                a, rr = i * 2.4, 0.075 * sqrt((i + 0.5) / 9)
                ms.append(trs((rr * cos(a), rr * sin(a), -0.31 - 0.02 * cos(i)), (180 + 50 * sin(a) * rr / 0.075, -50 * cos(a) * rr / 0.075, 0), 1.1))
            return join([o, tie, instances_to_mesh("bb", b, ms, keep_base=False)], "bunch")

        def bouquet(Mt, sc, n):
            bank.put("tub", mk_tub, Mt @ trs((0, 0, 0), (0, 0, 0), sc))
            c = Vector((0, 0, 0.27 * sc))
            R = 0.15 * sc
            bank.put("foliage", mk_foliage, Mt @ trs(tuple(c), (0, 0, rgd.u(0, 360)), sc))
            kind = rgd.pick(["a", "b", "ab", "ab"])
            for i in range(n):
                th, a = rgd.u(0, 78), rgd.u(0, 360)
                d = Vector((sin(radians(th)) * cos(radians(a)), sin(radians(th)) * sin(radians(a)), cos(radians(th)) * 0.85))
                pos = c + d * (R * rgd.u(0.92, 1.08))
                use_b = kind == "b" or (kind == "ab" and i % 3 == 0)
                bank.put("bloom2" if use_b else "bloom", mk_bloom2 if use_b else mk_bloom,
                         Mt @ trs(tuple(pos), (0, th, a), sc * rgd.u(0.85, 1.25)))
            for i in range(rgd.pick([0, 1, 2]) if sc > 0.7 else 0):
                bank.put("spray", mk_spray, Mt @ trs((rgd.jitter(0, 0.05) * sc, rgd.jitter(0, 0.05) * sc, 0.2 * sc), (rgd.jitter(0, 14), rgd.jitter(0, 14), rgd.u(0, 360)), sc * rgd.u(0.9, 1.15)))

        nblo = int(7 + 8 * stock)
        for i, x in enumerate(slot_x):
            for dx in ((-0.12, 0.12) if i % 2 == 0 else (0.0,)):
                if rgd.chance(0.3 + 0.6 * stock):
                    bouquet(trs((x + dx + rgd.jitter(0, 0.015), yfr - 0.02 + rgd.jitter(0, 0.02), ch), (0, 0, rgd.u(0, 360))), rgd.u(0.72, 0.85), nblo)
            if two_rows:
                bouquet(trs((x + rgd.jitter(0, 0.04), ybk, ch + chh + 0.002), (0, 0, rgd.u(0, 360))), rgd.u(1.0, 1.15), nblo + 6)
        for si, z in enumerate(shelf_z):
            xs = -xs_in + 0.18
            while xs < xs_in - 0.15:
                if rgd.chance(0.3 + 0.65 * stock):
                    bouquet(trs((xs, y_sh - 0.01, z), (0, 0, rgd.u(0, 360))), rgd.u(0.55, 0.7) if si < 2 else 0.5, int(nblo * 0.45))
                xs += rgd.u(0.38, 0.54)
        for i, x in enumerate(hang_x):
            if rgd.chance(0.3 + 0.7 * stock):
                at = hang_string(x, rgd.u(0.06, 0.2))
                bank.put("bunch", mk_bunch, trs(at, (rgd.jitter(0, 4), rgd.jitter(0, 4), rgd.u(0, 360)), rgd.vary(1.0, 0.12)))
        for gi, (gx, gy) in enumerate(ground):
            bouquet(trs((gx, gy, 0), (0, 0, rgd.u(0, 360))), rgd.u(1.35, 1.6), nblo + 8)

    else:  # trinkets
        mA = mat("ceramic_glazed", c1)
        mB = mat("brass", c2)

        def mk_vase():
            return _lathe_c("vase", [(0, 0), (0.036, 0), (0.04, 0.008), (0.036, 0.016), (0.062, 0.07), (0.066, 0.11), (0.05, 0.16), (0.026, 0.195), (0.023, 0.22),
                                     (0.036, 0.24), (0.03, 0.242), (0.016, 0.222), (0, 0.22)], 12, mA)

        def mk_jug():
            o = _lathe_c("jug", [(0, 0), (0.05, 0), (0.054, 0.006), (0.07, 0.05), (0.068, 0.1), (0.045, 0.15), (0.042, 0.175), (0.05, 0.19), (0.044, 0.19),
                                 (0.034, 0.172), (0, 0.168)], 12, mA)
            h = sweep("jug_h", [(0.046, 0, 0.165), (0.075, 0, 0.176), (0.1, 0, 0.16), (0.105, 0, 0.12), (0.09, 0, 0.085), (0.066, 0, 0.07)], 0.008, sides=5, smooth_path=False)
            assign(h, mA)
            return join([o, h], "jug")

        def mk_plate():
            return _lathe_c("plate", [(0, 0.004), (0.04, 0.0), (0.06, 0.004), (0.11, 0.02), (0.113, 0.024), (0.104, 0.025), (0.06, 0.011), (0, 0.009)], 16, mA, dz=0)

        def mk_urn():
            return _lathe_c("urn", [(0, 0), (0.1, 0), (0.108, 0.012), (0.1, 0.03), (0.17, 0.2), (0.185, 0.3), (0.15, 0.42), (0.09, 0.49), (0.085, 0.53),
                                    (0.11, 0.56), (0.1, 0.565), (0.07, 0.535), (0, 0.53)], 16, mA)

        def mk_bowl():
            return _lathe_c("bowl", [(0, 0), (0.035, 0), (0.038, 0.008), (0.03, 0.014), (0.085, 0.05), (0.1, 0.072), (0.096, 0.074), (0.08, 0.054), (0.03, 0.02), (0, 0.018)], 14, mB)

        def mk_goblet():
            return _lathe_c("goblet", [(0, 0), (0.034, 0), (0.036, 0.005), (0.012, 0.016), (0.009, 0.06), (0.016, 0.068), (0.04, 0.1), (0.044, 0.14), (0.041, 0.14),
                                       (0.036, 0.102), (0.0, 0.078)], 12, mB)

        def mk_stick():
            return _lathe_c("candlestick", [(0, 0), (0.05, 0), (0.054, 0.006), (0.03, 0.02), (0.014, 0.03), (0.011, 0.07), (0.02, 0.085), (0.011, 0.1), (0.01, 0.17),
                                            (0.018, 0.182), (0.034, 0.19), (0.036, 0.196), (0.014, 0.2), (0.013, 0.24), (0.0, 0.24)], 12, mB)

        def mk_pot():
            o = _lathe_c("pot", [(0, 0), (0.07, 0), (0.09, 0.02), (0.1, 0.08), (0.092, 0.13), (0.1, 0.136), (0.094, 0.14), (0.084, 0.13), (0.09, 0.08), (0.0, 0.014)], 14, mB)
            h = sweep("pot_h", [(0.092, 0, 0.115), (0.2, 0, 0.13), (0.33, 0, 0.125)], rounded_rect(0.02, 0.006, 0.002, 1), up=(0, 1, 0), smooth_path=False)
            assign(h, mB)
            return join([o, h], "pot")

        def mk_bell():
            o = _lathe_c("bell", [(0.058, -0.1), (0.062, -0.096), (0.048, -0.07), (0.036, -0.04), (0.03, -0.018), (0.016, -0.008), (0.006, -0.004), (0, -0.004)], 12, mB)
            o2 = _lathe_c("bell_in", [(0, -0.012), (0.026, -0.022), (0.042, -0.07), (0.055, -0.098), (0.058, -0.1)], 12, mB)
            cl = sphere("clapper", 0.011, loc=(0, 0, -0.098), segments=8, rings=5)
            assign(cl, iron)
            lp = torus("bell_loop", 0.011, 0.003, loc=(0, 0, 0.004), rot=(90, 0, 0), major_seg=8, minor_seg=4)
            assign(lp, mB)
            return join([o, o2, cl, lp], "bell")

        def mk_casket():
            b = box("casket", (0.2, 0.13, 0.085), loc=(0, 0, 0.0425), bevel=0.004, segments=1)
            assign(b, oak)
            lid = cylinder("casket_lid", 0.065, 0.2, loc=(0, 0, 0.085), rot=(0, 90, 0), verts=14, bevel=0.003, segments=1)
            assign(lid, oak)
            st = [box("strap", (0.018, 0.136, 0.09), loc=(sx * 0.07, 0, 0.045), bevel=0.001, segments=1) for sx in (-1, 1)]
            st += [cylinder("strap_l", 0.068, 0.018, loc=(sx * 0.07, 0, 0.085), rot=(0, 90, 0), verts=14) for sx in (-1, 1)]
            st.append(box("lock", (0.03, 0.008, 0.04), loc=(0, -0.066, 0.08), bevel=0.002, segments=1))
            for s in st:
                assign(s, mB)
            return join([b, lid] + st, "casket")

        def mk_beads():
            bd = sphere("bead", 0.011, segments=6, rings=4)
            assign(bd, mB)
            c = instances_to_mesh("beads", bd, [trs((0.004 * sin(i), 0, -0.012 - i * 0.021), (0, 0, 0), 1.0 if i % 4 else 1.5) for i in range(16)], keep_base=False)
            d = _lathe_c("pendant", [(0, 0.0), (0.03, 0.004), (0.045, 0.008), (0.03, 0.012), (0, 0.016)], 10, mA)
            place(d, loc=(0, -0.008, -0.365), rot=(90, 0, 0))
            return join([c, d], "beads")

        small = [("bowl", mk_bowl), ("goblet", mk_goblet), ("casket", mk_casket), ("jug", mk_jug), ("stick", mk_stick), ("vase", mk_vase)]
        tall = [("vase", mk_vase), ("stick", mk_stick), ("jug", mk_jug), ("urn", mk_urn)]
        xs = -xc + 0.14
        while xs < xc - 0.1:                      # front rows directly on the counter
            for row, yy in enumerate((yfr - 0.07, yfr + 0.1)):
                if rgd.chance(0.25 + 0.7 * stock):
                    k = rgd.pick(small[:4] if row == 0 else small)
                    bank.put(k[0], k[1], trs((xs + rgd.jitter(0, 0.02) + 0.08 * row, yy + rgd.jitter(0, 0.02), ch + 0.001), (0, 0, rgd.u(0, 360)), rgd.u(0.85, 1.1)))
            xs += rgd.u(0.17, 0.23)
        if two_rows:
            for i, x in enumerate(slot_x):
                for dx in (-0.14, 0.02, 0.16):
                    if rgd.chance(0.3 + 0.7 * stock):
                        k = rgd.pick(tall)
                        sc = 0.52 if k[0] == "urn" else rgd.u(0.95, 1.2)
                        bank.put(k[0], k[1], trs((x + dx + rgd.jitter(0, 0.015), ybk + rgd.jitter(0, 0.04), ch + chh + 0.003), (0, 0, rgd.u(0, 360)), sc))
        for si, z in enumerate(shelf_z):
            xs = -xs_in + 0.16
            while xs < xs_in - 0.12:
                if rgd.chance(0.3 + 0.7 * stock):
                    if rgd.chance(0.42):
                        sc = rgd.u(0.8, 1.05)
                        bank.put("plate", mk_plate, trs((xs, y_s1 - 0.045, z + 0.113 * sc * cos(radians(14))), (76, 0, 0), sc))
                        if rgd.chance(0.6):
                            k = rgd.pick(small[:2])
                            bank.put(k[0], k[1], trs((xs + rgd.jitter(0, 0.03), y_s0 + 0.09, z), (0, 0, rgd.u(0, 360)), 0.8))
                    else:
                        k = rgd.pick(small)
                        bank.put(k[0], k[1], trs((xs, y_sh + rgd.jitter(0, 0.04), z), (0, 0, rgd.u(0, 360) if k[0] != "casket" else rgd.jitter(0, 12)), rgd.u(0.85, 1.05)))
                xs += rgd.u(0.2, 0.28)
        for i, x in enumerate(hang_x):
            if rgd.chance(0.3 + 0.7 * stock):
                k = i % 3
                at = hang_string(x, rgd.u(0.1, 0.26) if k != 2 else 0.03)
                if k == 0:
                    bank.put("bell", mk_bell, trs(at, (0, 0, rgd.u(0, 360)), rgd.u(0.85, 1.3)))
                elif k == 1:
                    bank.put("pot", mk_pot, trs((at[0], at[1], at[2] - 0.33), (0, -90, rgd.jitter(90, 25)), rgd.u(0.85, 1.0)))
                else:
                    bank.put("beads", mk_beads, trs(at, (0, 0, rgd.jitter(0, 30))))
        for gi, (gx, gy) in enumerate(ground[:3]):
            if gi % 2 == 0:
                bank.put("urn", mk_urn, trs((gx, gy, 0), (0, 0, rgd.u(0, 360)), rgd.u(0.95, 1.15)))
            else:
                flat_crate(gx, gy, 0.0, rz=rgd.jitter(0, 15), flip=True)
                bank.put("pot", mk_pot, trs((gx - 0.05, gy, chh + 0.003), (0, 0, rgd.u(200, 340)), 1.1))

    made = bank.flush(p.seed)
    for k in ("apple", "orange", "pumpkin", "onions", "chili", "boule", "miche", "foliage", "beads"):
        if k in made:
            smooth(made[k], 180)
    # goods weathering: a little variety from the masks (bruised fruit, darker crusts, tarnish)
    for k, o in made.items():
        if k in ("apple", "orange", "pumpkin", "onions", "chili"):       # broad patches: whole fruits riper / duller, never per-face blotches
            paint(o, grime=mask_mul(noise_mask(0.35, seed=p.seed + 21, lo=0.4, hi=0.9), 0.3))
        elif k in ("boule", "baguette", "miche", "rings", "pretzels"):
            paint(o, grime=lambda co, n: np.clip(0.2 + 0.45 * np.clip(n[:, 2], 0, 1), 0, 1), wear=mask_mul(noise_mask(0.03, seed=p.seed + 22, lo=0.5, hi=0.8), 0.5))
        elif k in ("bowl", "goblet", "stick", "pot", "bell", "beads"):
            paint(o, grime=mask_mul(noise_mask(0.05, seed=p.seed + 23), 0.5), wear=mask_mul(noise_mask(0.03, seed=p.seed + 24, lo=0.4, hi=0.8), 0.6))
        elif k == "crate":
            paint(o, grime=mask_mul(noise_mask(0.2, seed=p.seed + 25), 0.5 + 0.3 * age))
        elif k in ("bolt", "nail", "hook"):
            paint(o, special=mask_mul(noise_mask(0.3, seed=p.seed + 26, lo=0.3, hi=0.7), 0.4 + 0.5 * age))

    for o in ropes:
        assign(o, rope_m)
    rp_all = join(ropes, "ropes")
    paint(rp_all, grime=0.15 + 0.3 * age)

    socket("festoon_l", "attach", (-hx - ext, yf, He - bh / 2), rot=(0, 0, 90), role="festoon")
    socket("festoon_r", "attach", (hx + ext, yf, He - bh / 2), rot=(0, 0, -90), role="festoon")
    socket("sign_front", "attach", (0, yf - bw / 2, He - bh / 2), role="sign")
