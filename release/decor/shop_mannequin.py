"""Tailor's display mannequin: a linen-covered dress form (no arms) with a turned neck cap and, when it
carries a hat, an ovoid milliner's head block, on a telescoping turned-oak pole with brass collar and
thumbscrew, standing on cabriole tripod legs with bun feet. It wears one of three outfits, each cut the
way the garment is made: a harlequin (jester) doublet pieced from quilted wool and satin diamonds with
stitched seam grooves, gathered under a keepered belt, laced up the back, dagged belled peplum, a
starched linen ruff set in figure-of-eight flutes, puffed shoulder caps with the empty sleeves hanging
limp and creased, and a two-horned fool's cap; an evening tailcoat (wool body with waist seam, tails
growing out of the back and lapping at the vent, satin lapels, waistcoat, shirt bib, wing collar, bow
tie, silk top hat, tape measure); or a gown (fitted bodice laced at the back, lace bertha, gathered
puff sleeves, bell skirt with turned hem, petticoat frill, satin sash with a bow, bound straw hat).
A paper price tag hangs on a string. Dust, faded dye and soiled hems scale with `wear`.
"""
from tk import *
import math
from math import sin, cos, pi, floor, copysign, atan2, degrees, hypot, asin, exp
import numpy as np
from mathutils import Matrix

META = {"name": "Dress mannequin display", "class": "prop", "mount": "ground", "focus": 0.7}
PARAMS = {
    "height": {"default": 1.56, "min": 1.40, "max": 1.80, "desc": "height to the top of the neck cap finial (m); the pole telescopes, the torso keeps its size"},
    "outfit": {"default": "jester", "choices": ["jester", "tailcoat", "gown"], "desc": "garment on the form"},
    "colour_a": {"default": "#a3242b", "desc": "main cloth colour (jester diamonds / coat / gown)"},
    "colour_b": {"default": "#1c6470", "desc": "second colour (jester diamonds / waistcoat / sash and ribbons)"},
    "colour_c": {"default": "#e0bd58", "desc": "accent colour (third jester diamond + pom-poms / bow tie + pocket square / lace trim)"},
    "hat": {"default": True, "desc": "wooden head block carrying the matching hat (fool's cap / top hat / straw hat); False = bare neck finial"},
    "girth": {"default": 1.0, "min": 0.88, "max": 1.15, "desc": "fullness of the torso form"},
    "legs": {"default": 3, "min": 3, "max": 5, "desc": "number of legs of the stand (3 = tripod)"},
    "wood": {"default": "#7a5433", "desc": "stain of the turned stand"},
    "diamonds": {"default": 10, "min": 6, "max": 14, "desc": "jester: harlequin diamonds around the body (rounded to even)"},
    "skirt_folds": {"default": 11, "min": 7, "max": 16, "desc": "gown: number of deep folds in the skirt"},
    "tape": {"default": True, "desc": "tailcoat: tailor's tape measure draped round the neck"},
    "tag_text": {"default": "45", "desc": "price written on the paper tag ('' = no tag)"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "dust, fading and shop wear"},
}
VARIANTS = {
    "tailcoat": {"outfit": "tailcoat", "height": 1.64, "colour_a": "#2f3a66", "colour_b": "#b3924e", "colour_c": "#8c1f2a",
                 "girth": 1.06, "wood": "#4a3222", "seed": 4},
    "gown": {"outfit": "gown", "height": 1.52, "colour_a": "#7b2346", "colour_b": "#d8a640", "colour_c": "#efe6d2",
             "girth": 0.94, "skirt_folds": 12, "seed": 8},
    "jester_green": {"outfit": "jester", "height": 1.44, "colour_a": "#2f7a3c", "colour_b": "#6a2c80", "colour_c": "#e8892c",
                     "hat": False, "diamonds": 8, "legs": 4, "wood": "#3b2a1e", "wear": 0.8, "seed": 13},
    "gown_ivory": {"outfit": "gown", "height": 1.72, "colour_a": "#d9cdb0", "colour_b": "#3f6f8f", "colour_c": "#f3ecdc",
                   "hat": False, "skirt_folds": 15, "girth": 1.02, "wood": "#5c4630", "wear": 0.1, "seed": 21},
}

# torso sections of the form: height above the hip plate, half width, front depth, back depth, superellipse exponent
_KT = [0.0, 0.10, 0.28, 0.46, 0.58, 0.615, 0.65, 0.68, 0.74]
_KW = [0.158, 0.172, 0.124, 0.158, 0.178, 0.184, 0.115, 0.058, 0.050]
_KF = [0.098, 0.106, 0.090, 0.124, 0.104, 0.084, 0.064, 0.054, 0.050]
_KB = [0.108, 0.120, 0.086, 0.104, 0.106, 0.086, 0.064, 0.054, 0.050]
_KN = [2.5, 2.5, 2.3, 2.4, 2.7, 2.7, 2.3, 2.0, 2.0]
_HEAD_SQ = 1.13          # the head block (and every hat on it) is this much deeper than wide


def _ss(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def _mix(a, b, t):
    return a + (b - a) * t


def _pw(c, e):
    return copysign(abs(c) ** e, c)


def _wrap(a):
    return (a + pi) % (2 * pi) - pi


def _rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _hex(c):
    return "#" + "".join("%02x" % int(round(255 * min(1.0, max(0.0, v)))) for v in c)


def _tone(h, wr, k=1.0):
    """A dyed cloth that has stood in a shop window: knocked back in saturation and faded towards dust."""
    c = _rgb(h)
    g = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    d = 0.15 + 0.13 * wr
    c = [g + (v - g) * (1 - d) for v in c]
    f = 0.04 + 0.20 * wr
    return _hex([(v + (dv - v) * f) * k for v, dv in zip(c, (0.70, 0.66, 0.58))])


def _arr(a, gain=1.0):
    """Per-vertex mask from a precomputed array (falls back to nothing if the vertex count changed)."""
    a = np.clip(np.asarray(a, dtype=np.float32) * gain, 0.0, 1.0)
    return lambda co, n: a if len(co) == len(a) else np.zeros(len(co), dtype=np.float32)


def _orient(loc, nrm):
    """Matrix putting a +Z-up fastener at loc with its axis along the surface normal."""
    yaw = degrees(atan2(nrm.x, -nrm.y))
    pitch = degrees(asin(max(-1.0, min(1.0, nrm.z))))
    return trs(tuple(loc), (90 - pitch, 0, yaw))


class _Form:
    """The torso: pt(a, t, off) = point at angle a (radians from the front, towards +X) and height t
    above the hip plate, pushed `off` metres off the surface (how the garments are draped on it)."""

    def __init__(self, zh, g):
        self.zh, self.g = zh, g
        self._c = {}

    def sect(self, t):
        tt = min(0.74, max(0.0, t))
        key = round(tt, 4)
        if key in self._c:
            return self._c[key]
        d = 0.016

        def f(tab):
            return float(np.interp(tt - d, _KT, tab) + 2 * np.interp(tt, _KT, tab) + np.interp(tt + d, _KT, tab)) * 0.25

        gs = 1 + (self.g - 1) * (1 - _ss(0.58, 0.66, tt))
        r = (f(_KW) * gs, f(_KF) * gs, f(_KB) * gs, f(_KN))
        self._c[key] = r
        return r

    def pt(self, a, t, off=0.0):
        hw, hf, hb, n = self.sect(t)
        s, c = sin(a), cos(a)
        hd = hf if c >= 0 else hb
        e = 2.0 / n
        x = hw * _pw(s, e)
        y = -hd * _pw(c, e)
        if off:
            e2 = 2.0 * (n - 1) / n
            nx = _pw(s, e2) / hw
            ny = -_pw(c, e2) / hd
            L = hypot(nx, ny) or 1.0
            x += off * nx / L
            y += off * ny / L
        z = self.zh + t + off * 0.9 * _ss(0.60, 0.66, t)
        return Vector((x, y, z))

    def nrm(self, a, t):
        v = self.pt(a, t, 0.02) - self.pt(a, t, 0.0)
        v.normalize()
        return v

    def a_for_x(self, t, x):
        hw, _, _, n = self.sect(t)
        return asin(min(1.0, max(0.0, x / hw)) ** (n / 2.0))


def _harlequin(name, fn, na, nr, k, mats, dag=False, puff=0.008, line_v=0.0):
    """Cloth pieced from diamonds. The mesh is a diagonal lattice and every diamond owns its vertices, so
    each patch is a little quilted pillow and every seam is a real crease (stitched groove) between two
    patches. fn(u, v, extra_offset) -> point. dag=True leaves the lowest row as free hanging points;
    the rows below line_v get an inner face (the lining seen from underneath).
    Returns (object, seam mask, pillow mask) - per-vertex arrays for the weathering."""
    I, J = 2 * na * k, 2 * nr * k
    idx, verts, seam, faces, cols = {}, [], [], [], []
    jl = int(line_v * J)

    def big(ci, cj):
        return floor((ci + cj) / (2.0 * k)), floor((ci - cj) / (2.0 * k))

    def vid(i, j, D, back=False):
        key = (i % I, j, D, back)
        if key not in idx:
            a, b = (i + j) / (2.0 * k), (i - j) / (2.0 * k)
            q = abs(sin(pi * a)) * abs(sin(pi * b))
            ex = puff * q ** 0.5 - (0.004 if back else 0.0)
            idx[key] = len(verts)
            verts.append(tuple(fn((i % I) / I, j / J, ex)))
            seam.append(1.0 if q < 1e-6 else 0.0)
        return idx[key]

    for j in range(0, J + 1):
        for i in range(I):
            if (i + j) % 2 == 0:
                continue
            if 0 < j < J:
                A, B = big(i, j)
                if dag and A - B <= 0:
                    continue
                D = ((A + B) % (2 * na), A - B)
                f = (vid(i - 1, j, D), vid(i, j - 1, D), vid(i + 1, j, D), vid(i, j + 1, D))
            elif j == 0:
                if dag:
                    continue
                A, B = big(i, 0.4)
                D = ((A + B) % (2 * na), A - B)
                f = (vid(i - 1, 0, D), vid(i + 1, 0, D), vid(i, 1, D))
            else:
                A, B = big(i, J - 0.4)
                D = ((A + B) % (2 * na), A - B)
                f = (vid(i - 1, J, D), vid(i, J - 1, D), vid(i + 1, J, D))
            c = 0 if (A + B) % 2 == 0 else 1 + (B % 2)
            faces.append(f)
            cols.append(c)
            if 0 < j <= jl:
                faces.append((vid(i, j + 1, D, True), vid(i + 1, j, D, True), vid(i, j - 1, D, True), vid(i - 1, j, D, True)))
                cols.append(c)
    o = mesh_object(name, verts, faces)
    for s, m in enumerate(mats):
        assign(o, m, slot=s)
    if len(o.data.polygons) == len(cols):
        o.data.polygons.foreach_set("material_index", np.array(cols, dtype=np.int32))
    smooth(o, 60)
    sm = np.array(seam, dtype=np.float32)
    return o, sm, 1.0 - sm


def _ribbon(name, cfn, nfn, width, n, thick=0.0015, closed=False):
    """Flat strip lying on a surface: cfn(v) centre line, nfn(v) outward surface normal."""

    def fn(u, v):
        e = 2e-3
        if closed:
            T = cfn((v + e) % 1.0) - cfn((v - e) % 1.0)
        else:
            T = cfn(min(1.0, v + e)) - cfn(max(0.0, v - e))
        W = T.cross(nfn(v))
        W.normalize()
        return cfn(v) + W * ((u - 0.5) * width)

    return surface(name, fn, nu=1, nv=n, closed_v=closed, thickness=thick)


def _frill(name, c, rin, rout, N, amp, droop, ph=0.0, nseg=6, thick=0.0025):
    """Small goffered frill / dagged brim: a ring whose waves grow towards the rim."""

    def fn(u, v):
        a = -2 * pi * u
        rx, ry = _mix(rin[0], rout[0], v), _mix(rin[1], rout[1], v)
        w = amp * (0.12 + 0.88 * v) * sin(N * a + ph)
        return (c[0] + rx * sin(a), c[1] - ry * cos(a), c[2] - droop * v ** 1.5 + w)

    return surface(name, fn, nu=N * nseg, nv=2, closed_u=True, thickness=thick)


def _ruff(name, c, rin, rout, N, amp, droop, gap=0.07, thick=0.0032, spp=8):
    """Starched linen ruff: one long band gathered into the neckband and set with a poking stick into
    figure-of-eight flutes, so the outer edge is a chain of full rounded loops `2 * amp` deep. Built as
    a closed thin shell (real cloth thickness, one consistent set of normals); the closure is at the back."""
    pitch = (2 * pi - gap) / N

    def mid(u, v):
        th = 2 * pi * N * u
        g = 0.14 + 0.86 * v
        a = -pi + gap / 2 + (2 * pi - gap) * u - 0.115 * pitch * g * sin(2 * th)
        rx, ry = _mix(rin[0], rout[0], v), _mix(rin[1], rout[1], v)
        return Vector((c[0] + rx * sin(a), c[1] - ry * cos(a), c[2] - droop * v + amp * g * sin(th)))

    st = ((0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (0.0, -1.0))

    def fn(u, w):
        v, sd = st[int(round(w * 3)) % 4]
        e = 0.2 / (N * spp)
        P = mid(u, v)
        n = (mid(u + e, v) - mid(u - e, v)).cross(mid(u, 1.0) - mid(u, 0.0))
        n.normalize()
        return P + n * (sd * thick * 0.5)

    o = surface(name, fn, nu=N * spp, nv=3)
    smooth(o, 75)
    return o


def _sleeve_fn(F, sx, L, kind, r):
    """An EMPTY sleeve on a form that has no arms: the head is set into the armscye (kind 'puff': a
    stiffened, gathered shoulder cap down to a band), below that the cloth hangs limp and flat
    (about 10 x 4 cm) in a few long drape folds, collapses in a crease where the elbow would be and
    swings / twists a little, differently left and right.
    Returns f(u, s, k=1, ex=0) with s = 0 at the shoulder ... 1 at the cuff, and f.centre(s)."""
    hw = F.sect(0.60)[0]
    puff = kind == "puff"
    zs = F.zh + (0.628 if puff else 0.618)
    ph, ph2 = r.u(0, 6.28), r.u(0, 6.28)
    swing = r.jitter(0.0, 0.030)
    th0 = r.u(0.25, 0.55)
    th1 = th0 + r.u(0.25, 0.7) * r.pick((1.0, 1.0, -0.7))
    se = (0.63 if puff else 0.53) + r.jitter(0.0, 0.035)
    kick = r.u(0.014, 0.032)
    out = r.u(-0.012, 0.008)
    sb = 0.25
    x0 = hw + (0.022 if puff else 0.004)

    def centre(s):
        st = min(1.0, s / 0.09)
        low = _ss(se - 0.03, se + 0.3, s)
        cx = x0 - (0.034 if puff else 0.03) * (1 - st) ** 2 + out * s + 0.5 * kick * low
        cy = swing * s * s - kick * low - 0.006 * s
        return Vector((sx * cx, cy, zs - L * s))

    def f(u, s, k=1.0, ex=0.0):
        phi = 2 * pi * u * sx
        cp, sp = cos(phi), sin(phi)
        fold = 0.0
        if puff and s < sb:
            q = s / sb
            m = sin(pi * (0.07 + 0.83 * q)) ** 0.7
            g = abs(cos(pi * q)) ** 1.4                     # gathered into armscye and band
            m *= 1 + 0.085 * g * sin(6 * phi + ph)
            ra, rb = 0.071 * m, 0.060 * m
            th = 0.0
        else:
            if puff:
                w = _ss(sb, sb + 0.11, s)
                ra = _mix(0.031, 0.053 - 0.007 * s, w)
                rb = _mix(0.026, 0.0185, w)
            else:
                m = 0.42 + 0.58 * min(1.0, s / 0.09) ** 0.5
                w = _ss(0.10, 0.36, s)
                ra = (0.059 - 0.011 * s) * m
                rb = _mix(0.036, 0.0205, w) * m
            pin = exp(-((s - se) / 0.04) ** 2)              # the collapsed elbow
            rb *= 1 - 0.5 * pin
            ra *= 1 + 0.09 * pin
            th = _mix(th0, th1, s) * w
            fold = w * (0.0062 * sin(4 * phi + ph + 1.6 * s) + 0.0034 * sin(7 * phi - 3.0 * s + ph2)) * (1 + 0.9 * pin)
            fold -= 0.009 * pin * cos(phi - ph)             # diagonal crease across the fold
            fold += 0.004 * w * sin(26 * s + 2 * sin(phi + ph2)) * exp(-((s - se) / 0.09) ** 2)
        la = (ra + ex) * sp * k
        lb = ((rb + ex) * cp + fold) * k
        c = centre(s)
        st_, ct_ = sin(th), cos(th)
        return Vector((c.x + sx * (ct_ * lb + st_ * la), c.y - st_ * lb + ct_ * la, c.z))

    f.centre = centre
    f.band = sb
    return f


def _buttons(name, pts, radius, material, height=None):
    """pts: [(location, outward normal)] -> one mesh of fastener heads."""
    h = height or radius * 0.7
    b = lathe(name + "_b", [(radius, 0.0), (radius, h * 0.35), (radius * 0.62, h), (0.0, h)], segments=7)
    o = instances_to_mesh(name, b, [_orient(loc, nrm) for loc, nrm in pts], keep_base=False)
    assign(o, material)
    return o


def _lacing(name, F, ta, tb, n, off, da, cord, brass):
    """Centre-back closure: two rows of brass eyelets either side of the back seam, a cord laced
    criss-cross through them and its two ends left hanging from the lowest pair."""
    eye = lathe(name + "_eye", [(0.0026, 0.0), (0.0058, 0.0006), (0.005, 0.0024), (0.0026, 0.0024)], segments=6)
    ms, rows = [], ([], [])
    for i in range(n):
        t = _mix(ta, tb, i / (n - 1.0))
        for sd in (0, 1):
            a = pi + (2 * sd - 1) * da
            ms.append(_orient(F.pt(a, t, off), F.nrm(a, t)))
            rows[sd].append((a, t))
    ey = instances_to_mesh(name + "_eyes", eye, ms, keep_base=False)
    assign(ey, brass)
    paint(ey, wear=0.6)
    for k in (0, 1):
        path = []
        for i in range(n):
            a, t = rows[(i + k) % 2][i]
            path.append(tuple(F.pt(a, t, off + 0.0028)))
            if i < n - 1:
                path.append(tuple(F.pt(pi, _mix(t, rows[0][i + 1][1], 0.5), off + 0.004 + 0.003 * k)))
        c = sweep(f"{name}_cord{k}", path, 0.0021, sides=4, smooth_path=False)
        assign(c, cord)
    for sd in (0, 1):
        a, t = rows[sd][0]
        q = F.pt(a, t, off + 0.003)
        sg = 2 * sd - 1
        _cord(f"{name}_end{sd}", [q, (q.x + sg * 0.008, q.y + 0.014, q.z - 0.022), (q.x + sg * 0.014, q.y + 0.02, q.z - 0.05),
                                  (q.x + sg * 0.012, q.y + 0.02, q.z - 0.085 - 0.02 * sd)], 0.0021, cord, curve=True)
    _cord(name + "_seam", [F.pt(pi, t, off + 0.0005) for t in np.linspace(ta - 0.03, tb + 0.03, 8)], 0.0022, cord)


def _cord(name, pts, r, material, sides=4, curve=False):
    """Thin cord / piping through the given points (optionally rounded off as a bezier)."""
    pts = [tuple(q) for q in pts]
    if curve and len(pts) == 4:
        pts = [tuple(q) for q in bezier_path(*pts, n=7)]
    o = sweep(name, pts, r, sides=sides, smooth_path=False)
    assign(o, material)
    return o


def _pompom(name, loc, radius, seed, material):
    """Yarn pom-pom: a tufted, uneven ball."""
    o = sphere(name, radius, loc=loc, segments=10, rings=7)
    rs = np.random.RandomState(seed)
    transform_verts(o, lambda co: co * float(1 + rs.uniform(-0.1, 0.13)))
    smooth(o, 42)
    assign(o, material)
    return o


# ------------------------------------------------------------------------------------------ stand
def _stand(p, rng, F, tcut, wood, brass):
    H, zh = p.height, F.zh + tcut
    r = rng.fork("stand")
    woods, metals = [], []
    R = 0.27 + 0.06 * (H - 1.4) / 0.4
    ztop = zh - 0.10
    z0 = 0.345
    Lm = ztop - z0
    prof = [(0.0, 0.10), (0.014, 0.104), (0.02, 0.118), (0.013, 0.132), (0.018, 0.14), (0.04, 0.15), (0.047, 0.162),
            (0.047, 0.275), (0.04, 0.288), (0.028, 0.293), (0.026, 0.30), (0.036, 0.31), (0.039, 0.32), (0.036, 0.33),
            (0.026, 0.34)]
    mid = [(0.028, 0.0), (0.040, 0.14), (0.042, 0.24), (0.035, 0.40), (0.026, 0.58)]
    if Lm > 0.3:
        mid += [(0.0245, 0.63), (0.031, 0.645), (0.031, 0.66), (0.0245, 0.675)]
    mid += [(0.0225, 0.78), (0.021, 1.0)]
    prof += [(rr, z0 + f * Lm) for rr, f in mid] + [(0.0, ztop)]
    pole = lathe("pole", prof, segments=16, smooth_angle=40)
    uv_grain(pole, axis='Z')
    woods.append(pole)
    # brass socket collar with thumbscrew; inner rod telescopes up into the hip plate
    metals.append(lathe("collar", [(0.0215, ztop - 0.036), (0.028, ztop - 0.034), (0.029, ztop - 0.026), (0.0265, ztop - 0.022),
                                   (0.0265, ztop - 0.008), (0.032, ztop - 0.003), (0.032, ztop + 0.006), (0.016, ztop + 0.010),
                                   (0.0, ztop + 0.010)], segments=18))
    metals.append(cylinder("rod", 0.011, zh - 0.02 - ztop, loc=(0, 0, (zh - 0.02 + ztop) / 2), verts=10))
    metals.append(lathe("flange", [(0.011, zh - 0.036), (0.022, zh - 0.032), (0.04, zh - 0.026), (0.042, zh - 0.02), (0.0, zh - 0.02)], segments=14))
    sa = r.u(215, 325)                      # the thumbscrew faces the shop front
    d = (cos(math.radians(sa)), sin(math.radians(sa)))
    metals.append(cylinder("screw", 0.0045, 0.034, loc=(d[0] * 0.042, d[1] * 0.042, ztop - 0.015), rot=(0, 90, sa), verts=8))
    metals.append(box("screw_wing", (0.006, 0.032, 0.02), loc=(d[0] * 0.062, d[1] * 0.062, ztop - 0.015), rot=(0, 0, sa), bevel=0.002))
    # hip plate closing the form
    outline = [tuple(F.pt(2 * pi * i / 28, tcut))[:2] for i in range(28)]
    plate = extrude_profile("hip_plate", outline, 0.02, loc=(0, 0, zh - 0.02), bevel=0.004)
    uv_grain(plate, axis='X')
    woods.append(plate)
    # cabriole legs mortised into the hub, bun feet, pegged joints
    n = int(p.legs)
    a_off = 90 + r.jitter(0, 6)
    sec = rounded_rect(0.05, 0.03, 0.009, 2)
    pegs = []
    for kk in range(n):
        a = math.radians(a_off + 360.0 * kk / n)
        ca, sn = cos(a), sin(a)
        pts2 = bezier_path((0.03, 0, 0.225), (0.15, 0, 0.278), (R - 0.08, 0, 0.16), (R, 0, 0.044), n=10)
        path = [(q[0] * ca, q[0] * sn, q[2]) for q in pts2]
        leg = sweep(f"leg{kk}", path, sec, up=(-sn, ca, 0), scale_fn=lambda t: 1.0 - 0.36 * t, smooth_path=False)
        woods.append(leg)
        woods.append(lathe(f"foot{kk}", [(0.0, 0.0), (0.021, 0.0), (0.031, 0.008), (0.033, 0.021), (0.027, 0.036), (0.019, 0.046),
                                          (0.0, 0.052)], segments=10, loc=(R * ca, R * sn, 0)))
        q = pts2[2]
        for sd in (-1, 1):
            pegs.append(trs((q[0] * ca - sd * sn * 0.0145, q[0] * sn + sd * ca * 0.0145, q[2]), (-90 * sd, 0, degrees(a))))
    pg = lathe("peg", [(0.005, 0.0), (0.0045, 0.002), (0.0, 0.003)], segments=6)
    metals.append(instances_to_mesh("pegs", pg, pegs, keep_base=False))
    assign(woods, wood)
    assign(metals, brass)
    k = 0.4 + p.wear
    for o in woods:
        paint(o, grime=mask_mul(height_ramp(0.0, 0.3), 0.8 * k), wear=mask_mul(height_ramp(0.0, 0.07), 0.9 * k))
    paint(metals, wear=mask_mul(noise_mask(0.03, seed=p.seed + 2, lo=0.2, hi=0.7), 0.85 - 0.4 * p.wear), grime=0.25 * p.wear)
    return woods, metals


def _head(F, hat, wood, wr):
    """Neck cap, then either a turned finial or an ovoid milliner's head block (about 165 mm wide,
    190 mm deep, 210 mm tall) on a short turned neck. Returns (z of the block base, z of the hat line)."""
    zc = F.zh + 0.74
    cap = lathe("neck_cap", [(0.049, 0.0), (0.056, 0.004), (0.057, 0.013), (0.051, 0.02), (0.0, 0.02)], segments=20, loc=(0, 0, zc))
    zm, hu, R = 0.150, 0.088, 0.083          # widest point above the block base, crown height, half width
    if hat:
        # a real neck's width, a jaw line, the widest point at the temples, a domed crown
        prof = [(0.049, 0.0), (0.052, 0.004), (0.052, 0.010), (0.046, 0.014), (0.0435, 0.024), (0.043, 0.040), (0.047, 0.054),
                (0.057, 0.068), (0.067, 0.084), (0.0745, 0.102), (0.080, 0.124), (R, zm)]
        for i in range(1, 7):
            th = 0.5 * pi * i / 7.0
            prof.append((R * cos(th) ** 0.92, zm + hu * sin(th)))
        prof.append((0.0, zm + hu))
    else:
        prof = [(0.02, 0.0), (0.014, 0.008), (0.014, 0.016), (0.022, 0.024), (0.027, 0.037), (0.023, 0.05), (0.012, 0.061), (0.0, 0.07)]
    top = lathe("head_block" if hat else "finial", prof, segments=18 if hat else 14, loc=(0, 0, zc + 0.02), smooth_angle=45)
    if hat:
        transform_verts(top, lambda co: Vector((co.x, co.y * (1 + (_HEAD_SQ - 1) * _ss(0.045, 0.11, co.z)), co.z)))
    uv_grain(top, axis='Z')
    uv_grain(cap, axis='X')
    assign([cap, top], wood)
    paint(top, wear=mask_mul(facing((0, 0, 1), 1.0), 0.5), grime=0.2 * wr)
    return zc + 0.02, zc + 0.02 + zm + 0.030


# ------------------------------------------------------------------------------------------ outfits
def _jester(p, rng, F, mA, mB, mC, cream, yarn, brass, leather, cloth, wr):
    r = rng.fork("jester")
    na = max(6, 2 * int(round(p.diamonds / 2.0)))
    k = 3 if na <= 10 else 2
    t0, t1, tb = -0.17, 0.668, 0.275
    nr = max(4, int(round((t1 - t0) / (1.0 / na * 1.22))))
    ph, ph2 = r.u(0, 6.28), r.u(0, 6.28)
    Ng = 13 if k == 3 else 9
    nf = 6

    def body(u, v, ex):
        a = 2 * pi * u + pi
        t = t0 + (t1 - t0) * v
        d = t - tb
        cinch = exp(-(d / 0.026) ** 2)
        ad = abs(d) / 0.042
        gath = ad * exp(1 - ad)                                 # cloth bunching either side of the belt
        gw = (0.5 + 0.5 * sin(Ng * a + ph + 1.3 * sin(3 * a + ph2))) ** 1.5
        fl = _ss(0.24, -0.15, t)                                # the peplum flaring into flutes
        off = 0.009 + ex * (1 - 0.8 * cinch) + 0.007 * _ss(0.27, 0.19, t)
        off += 0.016 * gath * gw * (0.8 if d > 0 else 1.0)
        off += 0.05 * fl * fl + 0.026 * fl * _pw(sin(nf * a + ph), 0.7)
        off += 0.0045 * sin(3 * a + 13 * t + ph) * _ss(0.33, 0.45, t)      # soft drag folds across the chest
        aw = _wrap(a)
        ds, dt = (abs(aw) - pi / 2) * 0.17, t - 0.53             # tension wrinkles radiating from the armscye
        rho = hypot(ds, dt)
        off += 0.005 * sin(6 * atan2(dt, ds) + ph2 + (0 if aw > 0 else 1.7)) * exp(-((rho - 0.105) / 0.04) ** 2)
        return F.pt(a, t, off)

    doublet, sm, pil = _harlequin("doublet", body, na, nr, k, [mA, mB, mC], dag=True, line_v=0.24)
    paint(doublet, grime=_arr(sm, 0.5 + 0.4 * wr), wear=_arr(pil, 0.12 + 0.3 * wr))
    cloth.append(doublet)
    # bells on the points of the dagged peplum
    bell = lathe("bell", [(0.0, 0.0), (0.008, 0.002), (0.012, 0.011), (0.008, 0.02), (0.003, 0.025), (0.003, 0.03), (0.0, 0.03)], segments=6)
    bm = []
    for i in range(na):
        q = body(i / na, 0.0, 0.0)
        bm.append(trs((q.x, q.y, q.z - 0.027), (0, 0, r.u(0, 360))))
    # belt: brass buckle, the strap end pushed through a keeper, punched eyelet holes, cloth belt loops
    bo = 0.0215
    belt = _ribbon("belt", lambda v: F.pt(2 * pi * v, tb, bo), lambda v: F.nrm(2 * pi * v, tb), 0.036, 40, thick=0.004, closed=True)
    q = F.pt(0, tb, bo + 0.004)
    buckle = shape_extrude("buckle", rounded_rect(0.056, 0.05, 0.008, 3), [rounded_rect(0.038, 0.032, 0.004, 2)], depth=0.005,
                           loc=(q.x, q.y, q.z), rot=(90, 0, 0))
    prong = box("buckle_prong", (0.036, 0.004, 0.004), loc=(q.x + 0.004, q.y - 0.006, q.z), bevel=0.001)
    assign([buckle, prong], brass)
    paint([buckle, prong], wear=0.65)
    a_end = 0.62 + r.u(0, 0.12)

    def tip(u, v):
        a = 0.06 + (a_end - 0.06) * v
        wv = 0.016 * min(1.0, (1 - v) / 0.09) ** 0.6
        return F.pt(a, tb + (u - 0.5) * 2 * wv - 0.002 * v, bo + 0.0045 + 0.004 * v * v)

    strap = surface("belt_tip", tip, nu=2, nv=10, thickness=0.0035)
    kq, kn = F.pt(0.36, tb, bo + 0.004), F.nrm(0.36, tb)
    keeper = box("belt_keeper", (0.011, 0.01, 0.044), loc=tuple(kq), rot=(0, 0, degrees(atan2(kn.x, -kn.y))), bevel=0.002)
    assign([belt, strap, keeper], leather)
    hole = lathe("belt_hole", [(0.0016, 0.0), (0.0042, 0.0004), (0.0036, 0.0016), (0.0016, 0.0016)], segments=6)
    hm = []
    for i in range(4):
        a = 0.14 + 0.055 * i + (0.05 if i > 2 else 0.0)
        hm.append(_orient(F.pt(a, tb - 0.001, bo + 0.0065 + 0.004 * ((a - 0.06) / (a_end - 0.06)) ** 2), F.nrm(a, tb)))
    for a in (pi - 0.42, pi + 0.42):
        hm.append(_orient(F.pt(a, tb, bo + 0.002), F.nrm(a, tb)))
    bh = instances_to_mesh("belt_holes", hole, hm, keep_base=False)
    assign(bh, brass)
    paint(bh, wear=0.6)
    for i, a in enumerate((pi - 0.2, pi + 0.2, 1.05, -1.05)):
        lq, ln = F.pt(a, tb, bo + 0.001), F.nrm(a, tb)
        lp = box(f"belt_loop{i}", (0.013, 0.007, 0.052), loc=tuple(lq), rot=(0, 0, degrees(atan2(ln.x, -ln.y))), bevel=0.002)
        assign(lp, mC)
        cloth.append(lp)
    # yarn pom-pom buttons down the front, piped centre-front seam between them
    for i, t in enumerate((0.365, 0.455, 0.545)):
        cloth.append(_pompom(f"pom{i}", tuple(F.pt(0, t, 0.03)), 0.022, p.seed + i, yarn))
    seam = sweep("front_seam", [tuple(F.pt(0, t, 0.0185)) for t in np.linspace(0.30, 0.66, 9)], 0.003, sides=4, smooth_path=False)
    assign(seam, mC)
    # laced closure up the back
    _lacing("lace", F, 0.335, 0.60, 5, 0.0185, 0.19, mC, brass)

    # sleeves: gathered shoulder cap, then the empty sleeve hanging limp; cuff band, linen frill, a bell
    for sx in (-1, 1):
        f = _sleeve_fn(F, sx, 0.57, "puff", r.fork(f"sleeve{sx}"))
        sl, ssm, spl = _harlequin(f"sleeve{sx}", lambda u, v, ex, f=f: f(u, 1 - v, 1.0, ex), 4, 4, 3,
                                  [mB, mA, mC] if sx > 0 else [mA, mB, mC], puff=0.0028)
        zc_ = f.centre(1.0).z
        paint(sl, grime=_arr(ssm, 0.4 + 0.4 * wr), wear=_arr(spl, 0.1 + 0.25 * wr))
        paint(sl, grime=mask_mul(height_ramp(zc_ - 0.02, zc_ + 0.14), 0.3 + 0.55 * wr))
        cloth.append(sl)
        for nm, s_, kk, rad in (("band", f.band, 1.1, 0.0055), ("cuff", 0.985, 1.07, 0.005)):
            bd = sweep(f"sleeve_{nm}{sx}", [tuple(f(u, s_, kk)) for u in np.linspace(0, 1, 12, endpoint=False)], rad, sides=4,
                       closed=True, smooth_path=False)
            assign(bd, mC)
        phf = r.u(0, 6.28)

        def frill(u, v, f=f, phf=phf):
            s_ = 0.99 + 0.075 * v
            c, q = f.centre(s_), f(u, s_, 1.03)
            g = 1 + 0.2 * v + 0.11 * v * sin(10 * 2 * pi * u + phf)
            return Vector((c.x + (q.x - c.x) * g, c.y + (q.y - c.y) * g, q.z))

        fr = surface(f"cuff_frill{sx}", lambda u, v, frill=frill: frill(u, 1 - v), nu=30, nv=2, closed_u=True, thickness=0.002)
        assign(fr, cream)
        paint(fr, grime=0.25 + 0.5 * wr)
        cloth.append(fr)
        c = f.centre(1.0)
        bm.append(trs((c.x, c.y - 0.004, c.z - 0.10), (0, 0, r.u(0, 360))))
        _cord(f"bell_cord{sx}", [(c.x, c.y - 0.004, c.z), (c.x, c.y - 0.004, c.z - 0.072)], 0.0015, mC, sides=3)

    # the ruff: neckband, one deep tier of figure-of-eight flutes, tie strings at the back
    zn = F.zh + 0.702
    ruff = _ruff("ruff", (0, 0.004, zn), (0.050, 0.048), (0.170, 0.147), 22, 0.026, 0.016)
    nb = surface("neckband", lambda u, v: F.pt(2 * pi * u + pi, 0.676 + 0.05 * v, 0.006), nu=20, nv=1, closed_u=True, thickness=0.003)
    assign([ruff, nb], cream)
    paint(ruff, grime=lambda co, n: np.clip(1.0 - (np.hypot(co[:, 0], co[:, 1]) - 0.055) / 0.075, 0, 1) * (0.3 + 0.4 * wr))
    cloth += [ruff, nb]
    for sd in (-1, 1):
        _cord(f"ruff_tie{sd}", [(sd * 0.006, 0.06, zn - 0.004), (sd * 0.016, 0.085, zn - 0.03), (sd * 0.022 + r.jitter(0, 0.008), 0.118, zn - 0.075),
                                (sd * 0.02, 0.118, zn - 0.12 - r.u(0, 0.03))], 0.0022, cream, curve=True)
    return bm, bell


def _fool_cap(p, rng, mA, mB, mC, yarn, brass, bell, cloth_extra):
    """Two-horned fool's cap: each horn two gores sewn along a piped seam, creased where it bends over,
    a coif round the block, rolled headband, dagged brim, a bell on each tip."""
    r = rng.fork("cap")
    parts, bm = [], []
    for sx, m in ((1, mA), (-1, mB)):
        dr = r.u(-0.04, 0.05)
        fw = r.u(-0.05, 0.03)
        ph = r.u(0, 6.28)
        P = [Vector(q) for q in ((sx * 0.012, 0, 0.0), (sx * 0.08, 0, 0.16), (sx * 0.21, fw * 0.5, 0.235), (sx * 0.31, fw, 0.09 + dr))]

        def frame(t, P=P):
            s = 1 - t
            c = P[0] * (s ** 3) + P[1] * (3 * s * s * t) + P[2] * (3 * s * t * t) + P[3] * (t ** 3)
            T = (P[1] - P[0]) * (3 * s * s) + (P[2] - P[1]) * (6 * s * t) + (P[3] - P[2]) * (3 * t * t)
            T.normalize()
            S = Vector((0, 1, 0))
            N2 = S.cross(T)
            N2.normalize()
            S = T.cross(N2)
            return c, N2, S

        def horn(u, v, sx=sx, ph=ph, frame=frame):
            t = v * 0.985
            c, N2, S = frame(t)
            phi = 2 * pi * u
            rr = 0.09 * (1.0 - 0.88 * t ** 0.72)
            inner = max(0.0, sx * cos(phi)) ** 1.5
            rr *= 1 + 0.10 * sin(34 * t + ph) * inner * _ss(0.12, 0.3, t) * _ss(0.95, 0.6, t)      # creases inside the bend
            rr *= 1 - 0.07 * abs(sin(phi)) ** 0.5 * _ss(0.05, 0.4, t) + 0.05 * sin(3 * phi + 9 * t + ph) * _ss(0.2, 0.6, t)
            return c + (N2 * cos(phi) + S * sin(phi)) * rr

        h = surface(f"horn{sx}", horn, nu=12, nv=14)
        assign(h, m)
        parts.append(h)
        pipe = sweep(f"horn_seam{sx}", [tuple(frame(t)[0] - frame(t)[1] * (sx * 0.09 * (1.0 - 0.88 * t ** 0.72) * 1.01)) for t in np.linspace(0.06, 0.97, 11)],
                     0.0034, sides=4, smooth_path=False)
        assign(pipe, mC)
        parts.append(pipe)
        e = frame(0.985)[0]
        bm.append(trs((e.x + sx * 0.004, e.y, e.z - 0.034)))
    coif = lathe("coif", [(0.085, 0.0), (0.086, 0.022), (0.078, 0.046), (0.055, 0.066), (0.0, 0.076)], segments=16)
    assign(coif, mC)
    phb = 0.0

    def dags(u, v):
        a = -2 * pi * u
        tri = abs(((8 * u + phb) * 2) % 2 - 1)
        rr = 0.0835 + 0.012 * v + 0.004 * v * tri
        return (rr * sin(a), -rr * cos(a), 0.012 - v * 0.062 * (0.3 + 0.7 * tri ** 1.3))

    brim = surface("cap_brim", dags, nu=32, nv=2, closed_u=True, thickness=0.003)
    assign(brim, mC)
    roll = torus("cap_roll", 0.089, 0.013, loc=(0, 0, 0.014), major_seg=20, minor_seg=6)
    assign(roll, yarn)
    bl = instances_to_mesh("cap_bells", bell, bm)
    assign(bl, brass)
    paint(bl, wear=0.6)
    return parts + [coif, brim, roll, bl]


def _tailcoat(p, rng, F, coat, satin, mB, shirt, mC, brass, paper, ink, cloth):
    r = rng.fork("coat")
    tw = 0.258                                  # the waist seam

    def x_open(t):
        return float(np.interp(t, [0.26, 0.30, 0.56, 0.67], [0.030, 0.034, 0.072, 0.060]))

    def a_open(t):
        tt = min(0.67, max(0.26, t))
        return F.a_for_x(tt, x_open(tt))

    # shirt bib and wing collar
    bib = surface("shirt_bib", lambda u, v: F.pt(_mix(-1.25, 1.25, u), _mix(0.34, 0.674, v), 0.004), nu=12, nv=8)
    col = surface("wing_collar", lambda u, v: F.pt(2 * pi * u, 0.664 + 0.062 * v, 0.004 + 0.007 * v), nu=20, nv=2, closed_u=True, thickness=0.002)
    assign([bib, col], shirt)
    cloth += [bib, col]

    # waistcoat: V neck, two points at the bottom, a strap and buckle across the back
    def wc(u, v):
        a = _mix(-1.9, 1.9, u)
        xr = abs(sin(a))
        tb = 0.184 - 0.044 * max(0.0, 1 - abs(xr - 0.33) / 0.33) + 0.016 * _ss(0.75, 1.0, xr)
        tt = 0.405 + 0.205 * _ss(0.0, 0.62, xr)
        t = _mix(tb, tt, v)
        return F.pt(a, t, 0.0085 + 0.002 * sin(52 * t + 2 * a) * exp(-((t - 0.27) / 0.05) ** 2))

    wco = surface("waistcoat", wc, nu=26, nv=9)
    solidify(wco, 0.002)
    assign(wco, mB)
    cloth.append(wco)

    # the coat body: two halves joined at the centre-back seam, cut straight across at the waist
    def coat_half(sx, ph):
        def fn(u, v):
            if sx < 0:
                u = 1 - u
            tb = tw + 0.012 * (1 - min(1.0, u / 0.3))
            t = _mix(tb, 0.668, v)
            a = _mix(a_open(t), pi, u)
            off = 0.014 + 0.003 * sin(5 * a + 17 * t + ph) * _ss(0.3, 0.4, t)
            ds, dt = (a - pi / 2) * 0.17, t - 0.53                # strain creases pulling from the armscye
            off += 0.0045 * sin(6 * atan2(dt, ds) + ph) * exp(-((hypot(ds, dt) - 0.105) / 0.04) ** 2)
            off += 0.003 * sin(64 * t + 3 * sin(2 * a + ph)) * exp(-((t - 0.31) / 0.04) ** 2) * _ss(0.5, 1.2, a)   # and across the waist
            return F.pt(sx * a, t, off)

        o = surface(f"coat{sx}", fn, nu=18, nv=13)
        solidify(o, 0.003)
        return o

    # the tails: sewn into the waist seam from the side-back round to the centre vent, tapering as the
    # front edge sweeps back, one lapping over the other at the vent, hanging in soft folds
    def tail(sx, ph):
        tt = -0.27 + r.jitter(0, 0.012)
        cross = r.u(0.04, 0.07)

        def fn(u, v):
            if sx < 0:
                u = 1 - u
            af = pi - _mix(1.50, 0.58, _ss(0.0, 1.0, v ** 0.55))
            ab = pi + (cross if sx > 0 else -0.004) * _ss(0.0, 0.2, v)
            a = _mix(af, ab, u)
            t = _mix(tw + 0.008, tt + 0.075 * (1 - u) ** 2.2, v)
            off = 0.0105 + 0.004 * _ss(0.0, 0.1, v) + (0.0055 if sx > 0 else 0.0) * _ss(0.0, 0.14, v) * _ss(0.3, 1.0, u)
            off += 0.008 * v * sin(7 * a + ph) + 0.004 * v * sin(15 * a - 2 * ph)
            off += 0.024 * v * v * (1 - u) ** 2 - 0.012 * v * v * u
            return F.pt(sx * a, t, off)

        o = surface(f"tail{sx}", fn, nu=8, nv=12)
        solidify(o, 0.003)
        return o

    for sx in (1, -1):
        c = coat_half(sx, r.u(0, 6.28))
        tl = tail(sx, r.u(0, 6.28))
        assign([c, tl], coat)
        cloth += [c, tl]

        # satin-faced peaked lapel lying on the coat front
        def lap(u, v, sx=sx):
            if sx < 0:
                u = 1 - u
            t = _mix(0.287, 0.655, v)
            w = float(np.interp(v, [0, 0.72, 0.80, 1.0], [0.010, 0.070, 0.050, 0.028]))
            pk = 0.032 * max(0.0, 1 - abs(v - 0.74) / 0.1) * u * u
            a = F.a_for_x(t, x_open(t) - 0.003 + u * w)
            return F.pt(sx * a, t + pk, 0.019 + 0.005 * sin(pi * u))

        lp = surface(f"lapel{sx}", lap, nu=4, nv=12)
        solidify(lp, 0.003)
        assign(lp, satin)
        cloth.append(lp)

        # the empty sleeve, shirt cuff showing, cuff buttons
        f = _sleeve_fn(F, sx, 0.60, "coat", r.fork(f"sleeve{sx}"))
        sl = surface(f"sleeve{sx}", lambda u, v, f=f: f(u, 1 - v), nu=20, nv=18, closed_u=True)
        assign(sl, coat)
        zc_ = f.centre(1.0).z
        paint(sl, grime=mask_mul(height_ramp(zc_ - 0.02, zc_ + 0.12), 0.25 + 0.5 * p.wear))
        cloth.append(sl)
        cf = surface(f"shirt_cuff{sx}", lambda u, v, f=f: f(u, 1.035 - 0.07 * v, 0.86), nu=20, nv=2, closed_u=True, thickness=0.002)
        assign(cf, shirt)
        bp = []
        for s in (0.90, 0.935, 0.97):
            q = f(0.13, s)
            nn = q - f.centre(s)
            nn.z = 0
            nn.normalize()
            bp.append((q, nn))
        _buttons(f"cuff_buttons{sx}", bp, 0.006, brass)

    # collar round the back of the neck, from lapel to lapel
    a1 = 1.15
    cpath = [tuple(F.pt(a, 0.668, 0.019) + Vector((0, 0, 0.012))) for a in np.linspace(a1, 2 * pi - a1, 16)]
    collar = sweep("coat_collar", cpath, rounded_rect(0.008, 0.044, 0.003, 1), up=(0, 0, 1), smooth_path=False)
    assign(collar, coat)
    cloth.append(collar)

    # seams: centre back, the curved side-backs, the waist seam the tails hang from
    seams = [_cord("back_seam", [F.pt(pi, t, 0.0152) for t in np.linspace(tw, 0.655, 9)], 0.0026, coat),
             _cord("waist_seam", [F.pt(a, tw + 0.002, 0.0158) for a in np.linspace(pi - 1.5, pi + 1.5, 15)], 0.0028, coat)]
    for sx in (-1, 1):
        sp = [(0.98, 0.57), (0.72, 0.50), (0.48, 0.40), (0.34, 0.30), (0.30, tw)]
        sp = [(_mix(sp[i][0], sp[i + 1][0], f), _mix(sp[i][1], sp[i + 1][1], f)) for i in range(4) for f in (0.0, 0.5)] + [sp[-1]]
        seams.append(_cord(f"side_seam{sx}", [F.pt(pi + sx * a, t, 0.0152) for a, t in sp], 0.0026, coat))
    # buttons: shirt studs, waistcoat, coat fronts, two at the back waist above the tail pleats
    _buttons("studs", [(F.pt(0, t, 0.0045), F.nrm(0, t)) for t in (0.46, 0.525, 0.59)], 0.0045, brass)
    _buttons("wc_buttons", [(F.pt(0.03, t, 0.0085), F.nrm(0.03, t)) for t in (0.21, 0.25, 0.29, 0.33, 0.37)], 0.0065, brass)
    cb = []
    for sx in (-1, 1):
        for t in (0.288, 0.328, 0.368):
            a = F.a_for_x(t, 0.088 + 0.2 * (t - 0.288))
            cb.append((F.pt(sx * a, t, 0.0145), F.nrm(sx * a, t)))
        cb.append((F.pt(pi + sx * 0.3, tw + 0.004, 0.0155), F.nrm(pi + sx * 0.3, tw)))
    _buttons("coat_buttons", cb, 0.0115, brass, height=0.005)

    # bow tie
    q = F.pt(0, 0.672, 0.016)
    for sx in (-1, 1):
        def lobe(u, v, sx=sx):
            uu = u if sx > 0 else 1 - u
            x = 0.006 + 0.05 * uu
            h = 0.008 + 0.021 * uu ** 0.8
            return (q.x + sx * x, q.y - 0.004 - 0.006 * sin(pi * v) * uu + 0.012 * uu * uu, q.z + (2 * v - 1) * h)

        lb = surface(f"bow{sx}", lobe, nu=4, nv=4, thickness=0.005)
        assign(lb, mC)
        cloth.append(lb)
    kn = sphere("bow_knot", 0.011, loc=(q.x, q.y - 0.005, q.z), segments=8, rings=6, scale=(1.0, 0.8, 1.2))
    assign(kn, mC)

    # pocket square in the breast welt (wearer's left)
    a = F.a_for_x(0.5, 0.118)
    q = F.pt(a, 0.5, 0.016)
    yaw = degrees(atan2(F.nrm(a, 0.5).x, -F.nrm(a, 0.5).y))
    welt = box("welt", (0.07, 0.004, 0.012), loc=tuple(q), rot=(0, -6, yaw), bevel=0.001)
    assign(welt, satin)
    for i, dx in enumerate((-0.014, 0.012)):
        sq = extrude_profile(f"pocket_square{i}", [(-0.017, 0.0), (0.017, 0.0), (0.004 * (1 - 2 * i), 0.03 + 0.008 * i)], 0.003,
                             loc=(q.x + dx * cos(math.radians(yaw)), q.y + dx * sin(math.radians(yaw)) - 0.002 - 0.002 * i, q.z + 0.004),
                             rot=(90, 0, yaw))
        assign(sq, mC)

    # tape measure round the neck, ends hanging down the lapels
    if p.tape:
        nodes = [(-F.a_for_x(0.33, 0.058), 0.31 + r.u(0, 0.05)), (-F.a_for_x(0.47, 0.066), 0.47), (-F.a_for_x(0.60, 0.085), 0.60),
                 (-1.25, 0.672), (-2.2, 0.69), (-pi, 0.694), (-2 * pi + 2.2, 0.69), (-2 * pi + 1.25, 0.672),
                 (-2 * pi + F.a_for_x(0.60, 0.082), 0.60), (-2 * pi + F.a_for_x(0.47, 0.06), 0.47), (-2 * pi + F.a_for_x(0.36, 0.052), 0.38 + r.u(0, 0.05))]
        cur = resample_path([(a, t, 0.0) for a, t in nodes], n=40)

        def at(v):
            x = v * (len(cur) - 1)
            i = min(len(cur) - 2, int(x))
            fr = x - i
            return _mix(cur[i][0], cur[i + 1][0], fr), _mix(cur[i][1], cur[i + 1][1], fr)

        def tc(v):
            a, t = at(v)
            return F.pt(a, t, 0.0285 + 0.012 * _ss(0.62, 0.69, t))

        tape = _ribbon("tape", tc, lambda v: F.nrm(*at(v)), 0.016, 39, thick=0.0012)
        assign(tape, paper)
        tick = mesh_object("tick", [(-0.0035, -0.0006, 0), (0.0035, -0.0006, 0), (0.0035, 0.0006, 0), (-0.0035, 0.0006, 0)], [(0, 1, 2, 3)])
        tm = []
        for i in range(1, 72):
            v = i / 72.0
            a, t = at(v)
            if t > 0.63:
                continue
            N = F.nrm(a, t)
            T = tc(min(1.0, v + 0.004)) - tc(max(0.0, v - 0.004))
            T.normalize()
            W = T.cross(N)
            W.normalize()
            lg = 1.0 if i % 4 else 1.9
            M = Matrix((W * lg, T, N)).transposed().to_4x4()
            M.translation = tc(v) + N * 0.0012 + W * (0.0045 - 0.0035 * lg * 0.5)
            tm.append(M)
        tk_ = instances_to_mesh("tape_ticks", tick, tm, keep_base=False)
        assign(tk_, ink)
        tips = []
        for v in (0.0, 1.0):
            q = tc(v)
            tips.append(box("tape_tip", (0.018, 0.003, 0.012), loc=tuple(q + F.nrm(*at(v)) * 0.0005), rot=(0, 0, degrees(atan2(F.nrm(*at(v)).x, -F.nrm(*at(v)).y))), bevel=0.0008))
        assign(tips, brass)


def _top_hat(p, rng, silk, band):
    h = lathe("top_hat", [(0.079, 0.0), (0.134, 0.004), (0.141, 0.011), (0.137, 0.018), (0.087, 0.013), (0.083, 0.022), (0.0805, 0.09),
                          (0.087, 0.160), (0.0865, 0.169), (0.081, 0.174), (0.0, 0.175)], segments=28, smooth_angle=45)

    def curl(co):
        rr = hypot(co.x, co.y)
        if co.z < 0.02 and rr > 0.086:
            return Vector((co.x, co.y, co.z + 0.022 * (co.x / 0.14) ** 2 * (rr - 0.086) / 0.055))
        return co

    transform_verts(h, curl)
    assign(h, silk)
    paint(h, wear=mask_mul(noise_mask(0.05, seed=p.seed, lo=0.3, hi=0.8), 0.35))
    b = ring_band("hat_band", radius=0.0835, width=0.032, thick=0.002, loc=(0, 0, 0.04), segments=28)
    assign(b, band)
    return [h, b]


def _straw_hat(p, rng, straw, mB, mA):
    r = rng.fork("hat")
    ph = r.u(0, 6.28)
    h = lathe("straw_hat", [(0.079, 0.0), (0.2, -0.003), (0.206, 0.0), (0.2, 0.003), (0.14, 0.006), (0.087, 0.008), (0.085, 0.02),
                            (0.084, 0.07), (0.07, 0.087), (0.0, 0.093)], segments=32, smooth_angle=50)
    bind = lathe("hat_binding", [(0.197, -0.0055), (0.2075, -0.0048), (0.2105, 0.0), (0.2075, 0.0048), (0.197, 0.0055)], segments=32)

    def wave(co):
        rr = hypot(co.x, co.y)
        if rr > 0.09:
            f = ((rr - 0.088) / 0.115)
            return Vector((co.x, co.y, co.z + (0.02 * sin(2 * atan2(co.y, co.x) + ph) - 0.026) * f * f))
        return co

    transform_verts(h, wave)
    transform_verts(bind, wave)
    assign(h, straw)
    paint(h, grime=lambda co, n: np.clip((np.hypot(co[:, 0], co[:, 1]) - 0.15) / 0.06, 0, 1) * (0.1 + 0.25 * p.wear))
    b = ring_band("hat_ribbon", radius=0.0865, width=0.034, thick=0.002, loc=(0, 0, 0.03), segments=28)
    assign([b, bind], mB)
    out = [h, b, bind]
    # ribbon tails at the back and fabric roses at the side
    for sx in (-1, 1):
        tl = surface(f"hat_tail{sx}", lambda u, v, sx=sx: (sx * (0.012 + 0.03 * v) + (u - 0.5) * 0.03, 0.087 + 0.11 * v,
                                                             0.03 - 0.012 * v - 0.05 * v * v + 0.006 * sin(6 * v + sx)), nu=1, nv=6, thickness=0.0015)
        assign(tl, mB)
        out.append(tl)
    for i in range(3):
        a = math.radians(205 + i * 28)
        rs = lathe(f"rose{i}", [(0.0, 0.0), (0.024, 0.002), (0.027, 0.01), (0.018, 0.013), (0.02, 0.02), (0.011, 0.022), (0.012, 0.028), (0.0, 0.03)],
                   segments=9, loc=(0.098 * cos(a), 0.098 * sin(a), 0.03), rot=(r.u(-25, 25) + 0, r.u(-25, 25), 0))
        assign(rs, mA if i != 1 else mB)
        out.append(rs)
    return out


def _gown(p, rng, F, mA, mB, mC, brass, cloth):
    r = rng.fork("gown")
    Nf = int(p.skirt_folds)
    ph1, ph2, ph3 = r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28)
    tw = 0.275
    zw = F.zh + tw
    zhem = min(0.24, max(0.16, F.zh - 0.5))
    Ls = zw - zhem
    Rh = 0.20 + 0.19 * Ls
    w_hw, w_hf, w_hb, _ = F.sect(tw)

    def fold(a):
        f = sin(Nf * a + ph1 + 0.9 * sin(2 * a + ph2))
        return _pw(f, 0.75)

    def skirt(a, s, grow=1.0, dz=0.0):
        b = 0.55 * (1 - (1 - min(s, 1.0)) ** 2.2) + 0.45 * s
        sa, ca = sin(a), cos(a)
        rx = _mix(w_hw + 0.013, Rh, b)
        ry = _mix((w_hf if ca >= 0 else w_hb) + 0.013, Rh * (0.93 if ca >= 0 else 1.08), b)
        g = fold(a)
        m = (1 + (0.012 + 0.13 * s ** 1.3) * g + 0.018 * s * sin(2 * Nf * a + ph3)) * grow
        e = 2.0 / _mix(2.3, 2.0, min(1.0, b))
        return Vector((rx * m * _pw(sa, e), -ry * m * _pw(ca, e), zw - s * Ls + 0.012 * s * g + dz))

    def skirt_n(a, s):
        e = 0.01
        n = (skirt(a, s + e) - skirt(a, s - e)).cross(skirt(a + e, s) - skirt(a - e, s))
        n.normalize()
        return n

    def sk(u, v):
        a = -2 * pi * u
        if v <= 0.9:
            return skirt(a, v / 0.9)
        q = (v - 0.9) / 0.1
        return skirt(a, 1.0, grow=1 - 0.014 * q, dz=0.02 * q)

    nsk = max(72, Nf * 6)
    so = surface("skirt", sk, nu=nsk, nv=11, closed_u=True)
    assign(so, mA)
    cloth.append(so)
    # lace petticoat frill peeping under the hem
    # (same angular sampling as the skirt, so the two shells stay parallel and never cross in the folds)
    pet = surface("petticoat", lambda u, v: skirt(-2 * pi * u, 0.93 + 0.135 * v, grow=0.95 * (1 + 0.012 * v * sin(2 * Nf * 2 * pi * u))),
                  nu=nsk, nv=2, closed_u=True, thickness=0.002)
    assign(pet, mC)
    cloth.append(pet)
    # ribbon band sewn round the skirt
    rb = _ribbon("skirt_band", lambda v: skirt(2 * pi * v, 0.8, grow=1.012), lambda v: skirt_n(2 * pi * v, 0.8), 0.04, nsk, thick=0.0, closed=True)
    assign(rb, mB)
    cloth.append(rb)

    # bodice with pointed waist and wide neckline, strained a little towards the lacing
    def tn(a):
        xr = abs(sin(a)) ** 1.6
        return _mix(0.56 if cos(a) >= 0 else 0.61, 0.648, xr)

    def bod(u, v):
        a = 2 * pi * u
        tb = 0.262 - (0.05 * max(0.0, 1 - abs(sin(a)) / 0.5) if cos(a) > 0 else 0.0)
        t = _mix(tb, tn(a), v)
        back = _ss(0.9, 0.2, abs(_wrap(a - pi)))
        return F.pt(a, t, 0.008 + 0.002 * sin(7 * a + 20 * v) * (1 - v) + 0.003 * back * sin(95 * t) * _ss(0.3, 0.36, t) * _ss(0.6, 0.55, t))

    bo = surface("bodice", bod, nu=40, nv=12, closed_u=True)
    assign(bo, mA)
    cloth.append(bo)
    for sx in (-1, 1):
        _cord(f"bodice_seam{sx}", [F.pt(sx * (0.3 + 0.5 * (t - 0.25)), t, 0.0092) for t in np.linspace(0.25, 0.56, 8)], 0.0025, mA)
    _buttons("bodice_buttons", [(F.pt(0, t, 0.0085), F.nrm(0, t)) for t in np.linspace(0.25, 0.525, 7)], 0.0065, brass)
    _lacing("lace", F, 0.335, 0.575, 5, 0.0095, 0.2, mB, brass)

    # lace bertha round the neckline
    Nb = 17
    phb = r.u(0, 6.28)

    def bertha(u, v):
        a = 2 * pi * u
        drop = 0.085 - 0.045 * sin(a) ** 2
        return F.pt(a, tn(a) + 0.01 - drop * v, 0.011 + 0.03 * v + 0.018 * v * _pw(sin(Nb * a + phb), 0.8))

    be = surface("bertha", bertha, nu=Nb * 5, nv=2, closed_u=True, thickness=0.002)
    assign(be, mC)
    cloth.append(be)

    # puff sleeves: cloth gathered into the armscye and into a ribbon cuff band, lace frill below
    for sx in (-1, 1):
        cx, cz = sx * (F.sect(0.6)[0] + 0.034), F.zh + 0.535
        php = r.u(0, 6.28)

        def puff(u, v, cx=cx, cz=cz, php=php, sx=sx, k=1.0):
            phi = 2 * pi * u
            g = abs(cos(pi * v)) ** 1.3
            rr = 0.079 * sin(pi * (0.11 + 0.83 * v)) ** 0.75 * (1 + (0.035 + 0.11 * g) * sin(10 * phi + php) + 0.035 * sin(3 * phi - php)) * k
            zz = -0.064 + 0.132 * v
            return (cx + rr * cos(phi) - sx * 0.25 * (zz + 0.06), rr * sin(phi) * 1.05, cz + zz)

        pf = surface(f"puff{sx}", puff, nu=40, nv=8, closed_u=True)
        assign(pf, mA)
        cloth.append(pf)
        cb = sweep(f"puff_band{sx}", [puff(u, 0.0, k=1.06) for u in np.linspace(0, 1, 14, endpoint=False)], rounded_rect(0.005, 0.016, 0.002, 1),
                   closed=True, up=(0, 0, 1), smooth_path=False)
        assign(cb, mB)
        fr = _frill(f"puff_frill{sx}", (cx + sx * 0.001, 0, cz - 0.066), (0.027, 0.029), (0.054, 0.058), 7, 0.007, 0.034, ph=php, nseg=6, thick=0.002)
        assign(fr, mC)
        cloth.append(fr)

    # sash with a bow and hanging tails at the back
    sash = _ribbon("sash", lambda v: F.pt(2 * pi * v, 0.272, 0.0165), lambda v: F.nrm(2 * pi * v, 0.272), 0.052, 40, thick=0.003, closed=True)
    assign(sash, mB)
    cloth.append(sash)
    K = F.pt(pi, 0.272, 0.02)
    sec = [(-0.0015, -0.032), (0.0015, -0.032), (0.0015, 0.032), (-0.0015, 0.032)]
    for sx in (-1, 1):
        up = r.u(0.0, 0.03)
        path = [K + Vector(d) for d in ((0, 0.012, 0), (sx * 0.06, 0.016, 0.018 + up), (sx * 0.125, 0.03, 0.026 + up), (sx * 0.135, 0.052, 0.012 + up),
                                        (sx * 0.07, 0.058, -0.004), (0, 0.04, 0))]
        lp = sweep(f"bow_loop{sx}", [tuple(q) for q in path], sec, up=(0, 0, 1), scale_fn=lambda t: 0.45 + 0.55 * sin(pi * t))
        assign(lp, mB)
        cloth.append(lp)
        L = r.u(0.3, 0.4)
        tl = _ribbon(f"bow_tail{sx}", lambda v, sx=sx, L=L: skirt(pi + sx * (0.05 + 0.05 * v), 0.015 + L * v, grow=1.035 + 0.02 * sin(9 * v + sx)),
                     lambda v, sx=sx, L=L: skirt_n(pi + sx * (0.05 + 0.05 * v), 0.015 + L * v), 0.05, 9, thick=0.002)
        assign(tl, mB)
        cloth.append(tl)
    kn = sphere("bow_knot", 0.024, loc=(K.x, K.y + 0.03, K.z), segments=10, rings=6, scale=(1.0, 0.8, 1.2))
    assign(kn, mB)
    return skirt, zhem


# ------------------------------------------------------------------------------------------ build
def build(p, rng):
    H = p.height
    zh = H - 0.83
    F = _Form(zh, p.girth)
    wr = p.wear
    wood = mat("wood_oak", p.wood, rough=0.62)
    brass = mat("brass", rough=0.8)
    linen = mat("canvas", "#b9a886", scale=1.5)
    leather = mat("leather", "#1d1712")
    paper = mat("paper", "#e9d68f")
    cloth = []

    # a gentleman's form stops at the hip; the others are hidden by the garment below the bust anyway
    tcut = 0.15 if p.outfit == "tailcoat" else 0.0
    _stand(p, rng, F, tcut, wood, brass)
    ztop, zhat = _head(F, p.hat, wood, wr)

    # the linen-covered form with piped centre seams
    tf = {"tailcoat": tcut, "jester": 0.62, "gown": 0.5}[p.outfit]
    form = surface("form", lambda u, v: F.pt(2 * pi * u, _mix(tf, 0.74, v)), nu=28, nv=max(3, int((0.74 - tf) / 0.065)), closed_u=True)
    assign(form, linen)
    for a in (0.0, pi):
        _cord("form_seam", [F.pt(a, t, 0.001) for t in np.linspace(tf, 0.735, 14 if tf < 0.3 else 5)], 0.0022, linen)
    paint(form, grime=mask_mul(noise_mask(0.12, seed=p.seed, lo=0.4, hi=0.8), 0.25 + 0.5 * wr))

    hat_parts = []
    tag_at = None
    if p.outfit == "jester":
        # motley: matt wool against satin, so the diamonds differ in sheen as well as colour
        mA = mat("canvas", _tone(p.colour_a, wr), rough=1.12)
        mB = mat("canvas", _tone(p.colour_b, wr), scale=3.5, rough=0.32)
        mC = mat("canvas", _tone(p.colour_c, wr, 0.92), scale=3.5, rough=0.4)
        yarn = mat("burlap", _tone(p.colour_c, wr, 1.05), scale=2.0)
        cream = mat("canvas", _tone("#ebe4d2", wr * 0.6), scale=2.5)
        bm, bell = _jester(p, rng, F, mA, mB, mC, cream, yarn, brass, leather, cloth, wr)
        bl = instances_to_mesh("bells", bell, bm)
        assign(bl, brass)
        paint(bl, wear=mask_mul(facing((0, 0, 1), 1.0), 0.8))
        if p.hat:
            hat_parts = _fool_cap(p, rng, mA, mB, mC, yarn, brass, bell, cloth)
            hat_tilt = (rng.u(-6, 2), rng.u(-7, 7), rng.u(-20, 20))
        delete(bell)
        a = 0.95
        tag_at = (F.pt(a, 0.262, 0.03), F.pt(a, 0.19, 0.062), F.nrm(a, 0.2), -12.0)
    elif p.outfit == "tailcoat":
        coat = mat("canvas", _tone(p.colour_a, wr), scale=2.0, rough=1.12)          # wool broadcloth
        satin = mat("canvas", _tone(p.colour_a, wr * 0.5, 0.42), scale=4.0, rough=0.3)   # silk facing and hat
        mB = mat("canvas", _tone(p.colour_b, wr), scale=3.0, rough=0.6)
        shirt = mat("canvas", _tone("#ebe6d8", wr * 0.5), scale=3.0)
        mC = mat("canvas", _tone(p.colour_c, wr), scale=4.0, rough=0.42)
        _tailcoat(p, rng, F, coat, satin, mB, shirt, mC, brass, paper, leather, cloth)
        if p.hat:
            hat_parts = _top_hat(p, rng, satin, coat)
            hat_tilt = (rng.u(-8, -3), rng.u(-6, 6), rng.u(-20, 20))
        a = -F.a_for_x(0.29, 0.088)
        tag_at = (F.pt(a, 0.288, 0.02), F.pt(a - 0.1, 0.205, 0.032), F.nrm(a - 0.1, 0.2), -4.0)
    else:
        mA = mat("canvas", _tone(p.colour_a, wr), scale=2.5, rough=0.8)             # taffeta
        mB = mat("canvas", _tone(p.colour_b, wr), scale=4.0, rough=0.32)            # satin ribbon
        mC = mat("canvas", _tone(p.colour_c, wr * 0.6), scale=1.5)                  # lace / linen
        skirt, zhem = _gown(p, rng, F, mA, mB, mC, brass, cloth)
        if p.hat:
            hat_parts = _straw_hat(p, rng, mat("wicker", "#d8bd84", scale=13.0), mB, mC)
            hat_tilt = (rng.u(-11, -5), rng.u(-8, 8), rng.u(-30, 30))
        a = 0.75
        q0, q1 = skirt(a, 0.10, grow=1.04), skirt(a, 0.22, grow=1.04)
        slope = degrees(atan2(hypot(q1.x - q0.x, q1.y - q0.y), q0.z - q1.z))
        nn = Vector((q0.x, q0.y, 0.0))
        nn.normalize()
        tag_at = (F.pt(a, 0.26, 0.02), q0, nn, -slope)

    if hat_parts:
        hat = join(hat_parts, "hat")
        place(hat, loc=(0, 0, zhat), rot=hat_tilt, scale=(1.0, _HEAD_SQ, 1.0))
        apply_transform(hat)
        cloth.append(hat)

    # price tag on a string
    if p.tag_text and tag_at:
        anchor, P, nn, tilt = tag_at
        yaw = degrees(atan2(nn.x, -nn.y))
        card = extrude_profile("tag", [(-0.024, -0.086), (0.024, -0.086), (0.024, -0.02), (0.012, -0.006), (-0.012, -0.006), (-0.024, -0.02)],
                               0.0016, loc=(0, 0.0008, 0), rot=(90, 0, 0))
        assign(card, paper)
        tx = text("tag_text", str(p.tag_text)[:4], size=0.026, depth=0.0008, loc=(0, -0.0008, -0.064), font="serif")
        assign(tx, leather)
        eye = torus("tag_eyelet", 0.0035, 0.0012, loc=(0, -0.0008, -0.015), rot=(90, 0, 0), major_seg=8, minor_seg=4)
        assign(eye, brass)
        tag = join([card, tx, eye], "price_tag")
        place(tag, loc=tuple(P), rot=(tilt, rng.jitter(0, 5), yaw))
        apply_transform(tag)
        hole = P + Vector((0, 0, -0.012))
        mid = (anchor + hole) * 0.5 + nn * 0.012 + Vector((0, 0, -0.012))
        _cord("tag_string", [anchor, (anchor + mid) * 0.5 + Vector((0, 0, -0.004)), mid, hole], 0.0011, linen, sides=3, curve=True)

    # shop wear: dust settling on shoulders, ruff and hat (and a little everywhere), hems soiled
    zl = min(world_bbox(cloth)[0].z, zh) if cloth else zh
    nz = noise_mask(0.16, seed=p.seed + 5, lo=0.25, hi=0.75)
    up = facing((0, 0, 1), 1.4)

    def dust(co, n):
        return np.clip((0.12 + 0.55 * nz(co, n)) * wr + up(co, n) * (0.25 + 0.75 * wr), 0.0, 1.0)

    for o in cloth:
        paint(o, grime=mask_mul(height_ramp(zl, zl + 0.24), 0.2 + 0.6 * wr), special=dust)

    socket("top", "attach", (0, 0, ztop + (0.24 if p.hat else 0.07)), role="prop_top")
