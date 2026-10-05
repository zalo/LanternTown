"""Small projecting first-floor balcony. A moulded stone slab (or a boarded timber deck on joists) is
carried by forged iron brackets: wall bar + top bar with curled ends, a big C-scroll brace bound on
with collars and a ring in the corner, bolted to the wall. The railing is square corner posts with
newel blocks and finials, flat bottom / frieze / top rails, a frieze of forged rings and repeating
scroll / lyre / heart / twisted-bar panels between square bars, capped by a moulded handrail.
Optional geranium pots hang outside the front rail in hooked ring holders.
Origin: centre of the deck's wall edge at floor level; wall plane y = 0, balcony projects to -Y.
"""
import bmesh
from tk import *
import numpy as np
from math import sin, cos, pi, atan2, acos, radians, hypot, ceil, floor

META = {"name": "Wrought-iron balcony", "class": "component", "mount": "wall", "mount_height": 3.1,
        "focus": [0.45, -0.7, 0.85], "tri_budget": 26000}       # dense forged scrollwork: every scroll is real bar
PARAMS = {
    "width": {"default": 1.8, "min": 1.0, "max": 3.6, "desc": "overall deck width along the wall (m)"},
    "projection": {"default": 0.75, "min": 0.45, "max": 1.3, "desc": "overall deck projection from the wall (m)"},
    "rail_height": {"default": 1.0, "min": 0.9, "max": 1.15, "desc": "top of handrail above the floor (m)"},
    "pattern": {"default": "scroll", "choices": ["scroll", "lyre", "bars", "hearts"], "desc": "railing panel motif"},
    "deck": {"default": "stone", "choices": ["stone", "timber"], "desc": "deck construction"},
    "brackets": {"default": 2, "min": 2, "max": 4, "desc": "number of scrolled iron brackets"},
    "frieze": {"default": True, "desc": "band of forged rings under the handrail"},
    "handrail": {"default": "oak", "choices": ["oak", "iron"], "desc": "handrail capping"},
    "finial": {"default": "ball", "choices": ["ball", "acorn", "none"], "desc": "corner post finials"},
    "gilt": {"default": True, "desc": "finials picked out in brass"},
    "paint": {"default": "#1c2622", "desc": "paint colour of the ironwork"},
    "stone": {"default": "#b7ab97", "desc": "stone / timber stain colour of the deck"},
    "pots": {"default": 3, "min": 0, "max": 6, "desc": "flower pots hung on the front rail"},
    "flower": {"default": "#c8322b", "desc": "flower colour"},
    "rust": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "rust bleeding at joints and feet"},
    "lantern": {"default": "right", "choices": ["none", "left", "right"], "desc": "oil lantern on a forged crook rising from a corner post"},
    "lantern_cd": {"default": 4.0, "min": 1.0, "max": 30.0, "desc": "lantern intensity (cd)"},
}
VARIANTS = {
    "lyre_wide": {"width": 2.7, "projection": 0.9, "pattern": "lyre", "brackets": 3, "paint": "#3b5f86",
                  "finial": "acorn", "pots": 4, "flower": "#e6dfd0", "lantern": "left", "seed": 4},
    "hearts_timber": {"width": 1.5, "projection": 0.65, "pattern": "hearts", "deck": "timber", "stone": "#6a4a2f",
                      "paint": "#6e1f18", "handrail": "iron", "pots": 2, "flower": "#e8b23a", "seed": 9},
    "bars_juliet": {"width": 1.2, "projection": 0.48, "pattern": "bars", "frieze": False, "handrail": "iron",
                    "finial": "none", "gilt": False, "paint": "#14161a", "pots": 0, "rust": 0.8, "lantern": "none", "seed": 15},
    "cream_long": {"width": 3.0, "projection": 1.0, "rail_height": 1.08, "brackets": 3, "frieze": False, "paint": "#a49f8b",
                   "stone": "#a9a294", "handrail": "iron", "pots": 4, "flower": "#d86a9a", "lantern": "left", "seed": 23},
}

ST, SD, BS = 0.007, 0.018, 0.013          # scroll bar thickness / depth, square baluster size


def _rect(a, b):
    return [(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)]


def _cham(a, b, c):
    ha, hb = a / 2, b / 2
    return [(-ha + c, -hb), (ha - c, -hb), (ha, -hb + c), (ha, hb - c), (ha - c, hb), (-ha + c, hb), (-ha, hb - c), (-ha, -hb + c)]


def _volute(c, R, turns, th, s, inner, chord=0.026):
    """Volute points from its outer end (angle th on radius R) winding inwards in direction s."""
    A = turns * 2 * pi
    pts, a = [], 0.0
    while True:
        t = min(a / A, 1.0)
        r = R * (1 - (1 - inner) * t ** 1.7)
        pts.append(c + r * np.array([cos(th + s * a), sin(th + s * a)]))
        if a >= A - 1e-9:
            break
        a = min(A, a + clamp(chord / r, 2 * pi / 18, 2 * pi / 7))
    return pts


def _scroll2d(c1, R1, turns1, c2, R2, turns2, kind="S", side=1, inner=0.22, bulge=0.0, chord=0.026):
    """Forged scroll: volute round c1 (wound up to radius R1), common tangent, volute round c2.
    kind 'S' = volutes turn opposite ways, 'C' = the same way. R = 0 gives a plain end at the point.
    bulge bows the body sideways (towards the tangent side). Returns (points, tangent pt 1, tangent pt 2)."""
    c1 = np.array(c1, float)
    c2 = np.array(c2, float)
    d = c2 - c1
    L = float(hypot(d[0], d[1]))
    phi = atan2(d[1], d[0])
    k = (R1 + R2) / L if kind == "S" else (R1 - R2) / L
    th = phi + side * acos(max(-1.0, min(1.0, k)))
    n = np.array([cos(th), sin(th)])
    n2 = -n if kind == "S" else n
    P1 = c1 + R1 * n
    P2 = c2 + R2 * n2
    T = P2 - P1

    def crs(a, b):
        return a[0] * b[1] - a[1] * b[0]

    pts = []
    has1 = R1 > 1e-6 and turns1 > 0
    if has1:
        s1 = 1.0 if crs(n, T) > 0 else -1.0
        pts += list(reversed(_volute(c1, R1, turns1, th, -s1, inner, chord)))
    else:
        pts.append(P1)
    nl = 8 if abs(bulge) > 1e-6 else 1
    for i in range(1, nl):
        t = i / nl
        f = sin(pi * t) ** 2 if has1 else 6.75 * t * (1 - t) ** 2
        pts.append(P1 + T * t + n * bulge * f)
    if R2 > 1e-6 and turns2 > 0:
        s2 = 1.0 if crs(n2, T) > 0 else -1.0
        pts += _volute(c2, R2, turns2, atan2(n2[1], n2[0]), s2, inner, chord)
    else:
        pts.append(P2)
    return [(float(q[0]), float(q[1])) for q in pts], P1, P2


def _pick(c1, R1, t1, c2, R2, t2, kind, key, **kw):
    """Both tangent solutions; keep the one minimising key(P1, P2)."""
    best = None
    for side in (1, -1):
        r = _scroll2d(c1, R1, t1, c2, R2, t2, kind, side, **kw)
        if best is None or key(r[1], r[2]) < key(best[1], best[2]):
            best = r
    return best


def _taper(mode):
    if mode == "both":
        return lambda t: 0.5 + 0.5 * smoothstep(0.0, 0.12, min(t, 1 - t))
    if mode == "end":
        return lambda t: 0.5 + 0.5 * smoothstep(0.0, 0.14, 1 - t)
    return None


def _bar(name, pts2, prof, plane="XZ", taper="both", closed=False):
    """Sweep a flat bar along a 2D path lying in the panel plane (XZ) or the bracket plane (YZ, u = -y)."""
    if plane == "XZ":
        P, up = [(u, 0.0, v) for u, v in pts2], (0, 1, 0)
    else:
        P, up = [(0.0, -u, v) for u, v in pts2], (1, 0, 0)
    return sweep(name, P, prof, smooth_path=False, up=up, scale_fn=_taper(taper), closed=closed)


def _mirror2(pts, cx):
    return [(2 * cx - u, v) for u, v in pts]


# ------------------------------------------------------------------ railing panel motifs (local XZ, u 0..m, v 0..h)
def _mod_scroll(m, h, flip):
    """Tall S-scroll with two full volutes and a pair of tendrils welded to its body."""
    prof = _rect(ST, SD)
    R = min((m - BS) / 2 - ST / 2, h / 4.4)
    cx = m / 2
    c1, c2 = (cx, 0.975 * R + ST / 2), (cx, h - 0.975 * R - ST / 2)
    pts, P1, P2 = _scroll2d(c1, R, 1.75, c2, R, 1.75, "S", 1)
    Rs = 0.31 * R
    b0 = P1 + (P2 - P1) * 0.27
    cs = (BS / 2 + ST / 2 + Rs, h / 2 + 0.27 * R)
    tp, _, _ = _pick((b0[0] - 0.004, b0[1] + 0.004), 0, 0, cs, Rs, 1.2, "C", lambda p1, p2: p2[0], inner=0.3, chord=0.022)
    tp2 = [(m - u, h - v) for u, v in tp]
    sprof = _rect(ST * 0.85, SD * 0.9)
    paths = [(pts, prof, "both"), (tp, sprof, "end"), (tp2, sprof, "end")]
    cols = [(0.0045, c1[1] + 0.3 * R), (m - 0.0045, c2[1] - 0.3 * R), (0.0045, cs[1]), (m - 0.0045, h - cs[1])]
    if flip:
        paths = [(_mirror2(q, cx), pr, tm) for q, pr, tm in paths]
        cols = [(m - u, v) for u, v in cols]
    objs = [_bar("s", q, pr, taper=tm) for q, pr, tm in paths]
    for u, v in cols:
        objs.append(box("col", (0.026, SD + 0.007, 0.014), loc=(u, 0, v)))
    return objs


def _mod_lyre(m, h, flip):
    """Lyre: two mirrored S-arms (foot volutes in, head volutes out), yokes and rod strings."""
    prof = _rect(ST, SD)
    cx = m / 2
    a = m / 2 - BS / 2 - ST / 2
    Rb, Rt = min(a * 0.45, h * 0.2), min(a * 0.3, h * 0.12)
    c1 = (cx - 0.86 * Rb - ST / 2 - 0.0005, 0.965 * Rb + ST / 2)
    c2 = (cx - a + 0.86 * Rt, h - 0.965 * Rt - ST / 2)
    pts, P1, P2 = _pick(c1, Rb, 1.4, c2, Rt, 1.3, "S", lambda p1, p2: p1[0], bulge=0.5 * (a - 1.9 * Rb))
    objs = [_bar("armL", pts, prof), _bar("armR", _mirror2(pts, cx), prof)]

    def ua(v):
        return P1[0] + (P2[0] - P1[0]) * (v - P1[1]) / (P2[1] - P1[1])

    vt, vb = P2[1] - 0.012, 2 * Rb + ST + 0.016
    for v in (vt, vb):
        w = 2 * (cx - ua(v)) + 0.004
        objs.append(box("yoke", (w, SD * 0.8, ST), loc=(cx, 0, v)))
    ns = 3 if m < 0.36 else 5
    sp = min(0.03, (cx - ua(vt) - 0.012) / ((ns - 1) / 2))
    for k in range(ns):
        u = cx + (k - (ns - 1) / 2) * sp
        objs.append(cylinder("string", 0.0034, vt - vb, loc=(u, 0, (vt + vb) / 2), verts=5, cap=False))
    tl = max(0.02, h - vt - 0.012)
    objs.append(cone("tip", 0.009, 0.0, tl, loc=(cx, 0, vt + tl / 2 + ST / 2), verts=4))
    objs.append(box("col", (0.024, SD + 0.007, 0.014), loc=(cx, 0, c1[1])))
    for u in (0.0045, m - 0.0045):
        objs.append(box("col", (0.026, SD + 0.007, 0.014), loc=(u, 0, c2[1])))
    return objs


def _mod_hearts(m, h, flip):
    """Heart of two bowed halves on a stem, over a pair of C-scrolls whose backs lie against the bars."""
    prof = _rect(ST, SD)
    cx = m / 2
    a = m / 2 - BS / 2 - ST / 2
    Rh = min((a - ST / 2) / 1.86, h * 0.2)
    c2 = (cx - 0.855 * Rh - ST / 2 - 0.0005, h - 0.965 * Rh - ST / 2)
    vp = c2[1] - 3.1 * Rh
    pts, _, _ = _pick((cx - 0.003, vp), 0, 0, c2, Rh, 1.3, "C", lambda p1, p2: p2[0], inner=0.25, bulge=0.5 * Rh)
    objs = [_bar("hL", pts, prof, taper="end"), _bar("hR", _mirror2(pts, cx), prof, taper="end")]
    objs.append(box("stem", (0.012, 0.012, vp + 0.012), loc=(cx, 0, (vp + 0.012) / 2)))
    objs.append(box("knot", (0.028, SD + 0.008, 0.018), loc=(cx, 0, vp + 0.009)))
    objs.append(box("col", (0.024, SD + 0.007, 0.014), loc=(cx, 0, c2[1])))
    R1 = min((a - 0.006 - ST) / 1.86, vp * 0.2)
    R2 = R1 * 0.74
    cl = (cx - 0.006 - ST / 2 - 0.86 * R1, 0.965 * R1 + ST / 2)
    ch = (cx - 0.006 - ST / 2 - 0.86 * R2, vp - 0.035 - R2)
    cp, Q1, Q2 = _pick(cl, R1, 1.3, ch, R2, 1.3, "C", lambda p1, p2: p1[0], chord=0.022)
    sprof = _rect(ST * 0.9, SD * 0.9)
    objs += [_bar("cL", cp, sprof), _bar("cR", _mirror2(cp, cx), sprof)]
    vm = (Q1[1] + Q2[1]) / 2
    for u in (0.0045, m - 0.0045):
        objs.append(box("col", (0.026, SD + 0.007, 0.014), loc=(u, 0, vm)))
    for v in (cl[1], ch[1]):
        objs.append(box("col", (0.036, SD + 0.005, 0.012), loc=(cx, 0, v)))
    return objs


def _mod_bars(m, h, flip):
    cx = m / 2
    objs = []
    if flip:
        o = sweep("tw", [(cx, 0, h * i / 18) for i in range(19)], _rect(0.013, 0.013), smooth_path=False, up=(0, 1, 0))
        z0, z1 = h * 0.2, h * 0.8

        def tw(co):
            a = 2 * pi * 1.75 * clamp((co.z - z0) / (z1 - z0))
            x, y = co.x - cx, co.y
            return Vector((cx + x * cos(a) - y * sin(a), x * sin(a) + y * cos(a), co.z))

        transform_verts(o, tw)
        objs.append(o)
        for z in (z0 - 0.008, z1 + 0.008):
            objs.append(box("bead", (0.021, 0.021, 0.011), loc=(cx, 0, z)))
    else:
        objs.append(sweep("pl", [(cx, 0, 0), (cx, 0, h)], _rect(0.013, 0.013), smooth_path=False, up=(0, 1, 0)))
        objs.append(lathe("knuckle", [(0.0066, -0.024), (0.011, -0.017), (0.0155, -0.006), (0.0155, 0.006), (0.011, 0.017), (0.0066, 0.024)],
                          segments=8, loc=(cx, 0, h * 0.5)))
        for z in (h * 0.5 - 0.04, h * 0.5 + 0.04):
            objs.append(box("bead", (0.019, 0.019, 0.007), loc=(cx, 0, z)))
    return objs


MODS = {"scroll": (_mod_scroll, 0.29, True), "lyre": (_mod_lyre, 0.34, True), "hearts": (_mod_hearts, 0.255, True),
        "bars": (_mod_bars, 0.108, False)}


# ------------------------------------------------------------------ bracket (local: wall x = 0 plane YZ, top at z = 0)
def _bracket(L, D, iron):
    bt, bw = 0.011, 0.042
    prof = _rect(bt, bw)
    u0 = bt / 2 + 0.001
    objs = []
    # top bar under the deck, outer end curled down; wall bar with the tail curled out
    Rt = 0.032
    pts, _, _ = _pick((0.001, -bt / 2), 0, 0, (L - Rt, -bt / 2 - Rt), Rt, 0.95, "C", lambda p1, p2: -p2[1])
    objs.append(_bar("top", pts, prof, "YZ", "end"))
    Rw = 0.036
    pts, _, _ = _pick((u0, -bt), 0, 0, (u0 + Rw, -D), Rw, 1.0, "C", lambda p1, p2: p2[0])
    objs.append(_bar("wall", pts, prof, "YZ", "end"))
    # main C-scroll brace: volutes bear on the wall bar and under the top bar
    s = min(L, D)
    R1, R2 = 0.2 * s, 0.155 * s
    c1 = (u0 + bt + 0.94 * R1, -D + 0.085 + R1)
    c2 = (L - 0.07 - R2, -1.5 * bt - 0.94 * R2)
    corner = np.array([u0 + bt / 2, -bt])
    pts, P1, P2 = _pick(c1, R1, 1.5, c2, R2, 1.5, "C", lambda p1, p2: -float(np.linalg.norm((p1 + p2) / 2 - corner)), chord=0.03)
    objs.append(_bar("brace", pts, _rect(bt, bw - 0.004), "YZ"))
    # ring wedged in the corner between bars, brace and volutes
    T = (P2 - P1) / np.linalg.norm(P2 - P1)
    nrm = np.array([-T[1], T[0]])
    lo, hi = 0.01, 0.4 * s
    for _ in range(24):
        r = (lo + hi) / 2
        c = corner + np.array([bt / 2 + r, -bt / 2 - r])
        ok = (abs(float(np.dot(c - P1, nrm))) >= r + bt and np.linalg.norm(c - np.array(c1)) >= R1 + r + bt
              and np.linalg.norm(c - np.array(c2)) >= R2 + r + bt)
        lo, hi = (r, hi) if ok else (lo, r)
    rr = lo
    rc = corner + np.array([bt / 2 + rr, -bt / 2 - rr])
    ring = [(rc[0] + rr * cos(2 * pi * i / 18), rc[1] + rr * sin(2 * pi * i / 18)) for i in range(18)]
    objs.append(_bar("ring", ring, _rect(0.009, 0.03), "YZ", None, closed=True))
    # second small ring / quatre-foil filler further out when there is room
    # collars binding scroll to bars
    objs.append(box("col", (bw + 0.014, 0.034, 0.024), loc=(0, -(u0 + bt * 0.75), c1[1]), bevel=0.003, segments=1))
    objs.append(box("col", (bw + 0.014, 0.024, 0.034), loc=(0, -c2[0], -bt * 1.25), bevel=0.003, segments=1))
    objs.append(box("col", (0.04, 0.02, 0.026), loc=(0, -(u0 + bt * 0.7), rc[1]), bevel=0.002, segments=1))
    objs.append(box("col", (0.04, 0.026, 0.02), loc=(0, -rc[0], -bt * 1.2), bevel=0.002, segments=1))
    # fixings: coach bolts through the wall bar, bolts up into the deck
    bl = cylinder("b", 0.0105, 0.008, verts=6, loc=(0, 0, 0.004))
    apply_transform(bl)
    ms = [trs((0, -(u0 + bt / 2), z), (90, 0, 0)) for z in (-0.05, rc[1] - rr - 0.03 if rc[1] - rr - 0.03 > c1[1] + R1 else -D * 0.3, -D + 0.06)]
    ms += [trs((0, -u, -bt), (180, 0, 0)) for u in (rc[0] + rr + 0.035, L - 0.045)]
    objs.append(instances_to_mesh("bolts", bl, ms, keep_base=False))
    for o in objs:
        assign(o, iron)
    return join(objs, "bracket_base")


def _u_slab(name, x0, x1, proj, prof, end_l, end_r):
    """Stone slab with a moulded nosing run round the front and the free ends (mitred corners)."""
    verts, faces = [], []
    for (e, z) in prof:
        xl, xr, yf = x0 + (e if end_l else 0.0), x1 - (e if end_r else 0.0), -(proj - e)
        verts += [(xl, 0.0, z), (xl, yf, z), (xr, yf, z), (xr, 0.0, z)]
    for i in range(len(prof) - 1):
        for k in range(3):
            if (k == 0 and not end_l) or (k == 2 and not end_r):
                if abs(prof[i][1] - prof[i + 1][1]) < 1e-6:
                    continue
            a, b = i * 4 + k, i * 4 + k + 1
            faces.append((a, b, b + 4, a + 4))
    n = (len(prof) - 1) * 4
    faces.append((3, 2, 1, 0))
    faces.append((n, n + 1, n + 2, n + 3))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _pot(r, flower_col, zpot):
    """Terracotta pot with a geranium; base centre at the origin."""
    ph = r.vary(0.15, 0.08)
    k = ph / 0.15
    prof = [(0.0, 0.0), (0.054 * k, 0.0), (0.058 * k, 0.005), (0.079 * k, ph - 0.03), (0.088 * k, ph - 0.03), (0.09 * k, ph - 0.024),
            (0.09 * k, ph - 0.003), (0.086 * k, ph), (0.08 * k, ph), (0.078 * k, ph - 0.016), (0.0, ph - 0.016)]
    pot = lathe("pot", prof, segments=10)
    assign(pot, mat("terracotta", "#c9764a"))
    paint(pot, grime=mask_mul(noise_mask(0.05, seed=int(r.u(0, 99))), 0.5), special=mask_mul(height_ramp(0.0, ph * 0.8), 0.6))
    objs = [pot]
    cz = ph + 0.035
    cr = 0.085 * k
    core = sphere("core", cr, loc=(0, 0, cz), segments=9, rings=5, scale=(1.0, 1.0, 0.72))
    displace_noise(core, strength=0.018, scale=0.07, seed=int(r.u(0, 999)))
    assign(core, mat("hedge", "#6f8f55"))
    objs.append(core)
    # round scalloped leaves as shallow cups, uv mapped to the whole leaf tile
    lv, lf = [(0.0, 0.0, -0.006)], []
    nseg = 7
    for i in range(nseg):
        a = 2 * pi * i / nseg
        rr = 0.034 * (1.0 if i % 2 == 0 else 0.9)
        lv.append((rr * cos(a), rr * sin(a), 0.004 if i % 2 else 0.0))
    for i in range(nseg):
        lf.append((0, 1 + i, 1 + (i + 1) % nseg))
    leaf = mesh_object("leaf", lv, lf)
    uvl = leaf.data.uv_layers.new(name="UVMap")
    for li, loop in enumerate(leaf.data.loops):
        co = leaf.data.vertices[loop.vertex_index].co
        uvl.data[li].uv = (0.125 + co.x / 0.034 * 0.11, 0.125 + co.y / 0.034 * 0.11)
    uv_keep(leaf)
    assign(leaf, mat("leaf", "#7da052"))
    ms = []
    for i in range(15):
        az = r.u(0, 360)
        tilt = r.u(15, 115)
        d = (sin(radians(tilt)) * cos(radians(az)), sin(radians(tilt)) * sin(radians(az)), cos(radians(tilt)))
        rad = cr * r.u(1.0, 1.25)
        ms.append(trs((d[0] * rad, d[1] * rad, cz + d[2] * rad * 0.72 - (0.03 if tilt > 90 else 0.0)), (0, tilt * 0.75, az), r.u(0.75, 1.25)))
    objs.append(instances_to_mesh("leaves", leaf, ms, uv_jitter=False, keep_base=False))
    # flower heads: umbels of small florets on thin stalks
    fl = join([sphere("fl", 0.017 * sc, loc=(dx, dy, dz), segments=5, rings=3, scale=(1, 1, 0.75))
               for dx, dy, dz, sc in ((0.006, -0.008, 0.004, 1.0), (0.012, 0.012, -0.003, 0.85), (-0.014, 0.002, -0.002, 0.9))],
              "floret")
    assign(fl, mat("leaf", flower_col))
    fm, stalks = [], []
    for i in range(int(r.u(4, 7))):
        az = r.u(0, 360)
        tilt = r.u(5, 70)
        d = (sin(radians(tilt)) * cos(radians(az)), sin(radians(tilt)) * sin(radians(az)), cos(radians(tilt)))
        rad = cr * r.u(1.35, 1.8)
        pos = (d[0] * rad, d[1] * rad, cz + d[2] * rad * 0.8)
        fm.append(trs(pos, (r.u(-20, 20), tilt * 0.6, az), r.u(0.75, 1.2)))
        stalks.append(sweep("stalk", [(d[0] * cr * 0.5, d[1] * cr * 0.5, cz), pos], 0.002, sides=4, smooth_path=False, cap=False))
    objs.append(instances_to_mesh("flowers", fl, fm, keep_base=False))
    for s_ in stalks:
        assign(s_, mat("leaf", "#7da052"))
    objs += stalks
    return objs, ph


def build(p, rng):
    W, PR, H = p.width, p.projection, p.rail_height
    iron = mat("iron_painted", p.paint)
    brass = mat("brass", rough=0.85)
    jr = rng.fork("jitter")
    fn, m_target, bound_bars = MODS[p.pattern]
    iron_objs = []                    # railing ironwork (rust near the deck)

    # ================================================================ deck
    if p.deck == "stone":
        t = 0.13
        st_mat = mat("stone_smooth", p.stone)
        prof = [(0.018, 0.0), (0.005, -0.004), (0.0, -0.013), (0.0, -0.05), (0.006, -0.056), (0.013, -0.058), (0.015, -0.066),
                (0.021, -0.079), (0.033, -0.092), (0.049, -0.1), (0.055, -0.104), (0.055, -0.117), (0.063, -0.125), (0.078, -0.13)]
        nst = 1 if W <= 2.3 else (2 if W <= 3.1 else 3)
        cuts = [-W / 2] + [-W / 2 + W * (i + 1) / nst + rng.fork("joint").jitter(0, 0.06) for i in range(nst - 1)] + [W / 2]
        for i in range(nst):
            gl, gr = (0.0 if i == 0 else 0.002), (0.0 if i == nst - 1 else 0.002)
            s = _u_slab(f"slab{i}", cuts[i] + gl, cuts[i + 1] - gr, PR - rng.fork(f"slab{i}").u(0, 0.004) * (nst > 1), prof, i == 0, i == nst - 1)
            densify(s, 0.24)
            displace_noise(s, strength=0.0016, scale=0.22, seed=p.seed + i)
            bevel(s, 0.003, segments=1, angle=55)
            assign(s, st_mat)
            smooth(s, 35)
            paint(s, grime=lambda co, n: np.clip(0.25 + 0.6 * (n[:, 2] < -0.5) + 0.5 * np.clip((co[:, 1] + 0.12) / 0.12, 0, 1), 0, 1) * 0.8,
                  special=mask_mul(facing((0, 0, 1)), noise_mask(0.22, seed=p.seed + 11, lo=0.4, hi=0.75),
                                   lambda co, n: np.clip((-co[:, 1] - PR * 0.35) / (PR * 0.5), 0, 1), 0.9))
    else:
        jt_, bt_ = 0.11, 0.028
        t = jt_ + bt_
        wr = rng.fork("timber")
        oak = mat("wood_oak", p.stone)
        oak_d = oak
        eg = mat("wood_endgrain", p.stone)
        nb_ = int(p.brackets)
        ib = min(0.22, W * 0.14)
        jx = sorted(set([round(-W / 2 + 0.06, 4), round(W / 2 - 0.06, 4)] + [round(-W / 2 + ib + (W - 2 * ib) * i / (nb_ - 1), 4) for i in range(nb_)]))
        for i, x in enumerate(jx):
            j = plank(f"joist{i}", PR - 0.075, 0.085, jt_, loc=(x, -(PR - 0.075) / 2 - 0.001, -bt_ - jt_ / 2), axis="Y", bevel=0.006, rng=wr)
            assign(j, oak_d)
            paint(j, grime=0.35)
        tr = plank("trimmer", W - 0.03, 0.07, jt_, loc=(0, -(PR - 0.045), -bt_ - jt_ / 2), axis="X", bevel=0.008, rng=wr, end_mat=eg)
        assign(tr, oak_d)
        paint(tr, grime=0.3, wear=mask_mul(facing((0, -1, 0)), noise_mask(0.2, seed=p.seed), 0.4))
        # moulded cover strip planted on the trimmer
        cs = plank("cover", W - 0.01, 0.018, 0.045, loc=(0, -(PR - 0.045) - 0.043, -bt_ - 0.03), axis="X", bevel=0.006, rng=wr, end_mat=eg)
        assign(cs, oak_d)
        nbd = max(3, int(round(PR / 0.128)))
        bw_ = (PR - 0.004) / nbd
        nails = []
        for i in range(nbd):
            ln = W + 0.02 + wr.u(-0.008, 0.008)
            y = -(i + 0.5) * bw_ - 0.004 - (0.008 if i == nbd - 1 else 0.0)
            bd = plank(f"board{i}", ln, bw_ - 0.005, bt_, loc=(wr.u(-0.004, 0.004), y, -bt_ / 2 - wr.u(0, 0.002)), axis="X",
                       bevel=0.004, warp=0.003, rng=wr, end_mat=eg)
            assign(bd, oak)
            paint(bd, grime=mask_mul(noise_mask(0.3, seed=p.seed + i), 0.5), wear=mask_mul(facing((0, 0, 1)), noise_mask(0.35, seed=p.seed + 40), 0.5),
                  special=mask_mul(facing((0, 0, 1)), noise_mask(0.18, seed=p.seed + 7, lo=0.5, hi=0.8), 0.5))
            for x in jx:
                for dy in (-bw_ * 0.28, bw_ * 0.28):
                    nails.append(trs((x + wr.u(-0.006, 0.006), y + dy, 0.0)))
        nl = bolt("nail", 0.0045, 0.002, kind="pyramid")
        nm = instances_to_mesh("nails", nl, nails, keep_base=False)
        assign(nm, iron)

    # ================================================================ brackets
    nb = int(p.brackets)
    ib = min(0.22, W * 0.14)
    Lb = PR - 0.09
    Db = clamp(PR * 0.78, 0.38, 0.85)
    bb = _bracket(Lb, Db, iron)
    bx = [-W / 2 + ib + (W - 2 * ib) * i / (nb - 1) for i in range(nb)]
    brk = instances_to_mesh("brackets", bb, [trs((x, 0, -t)) for x in bx], keep_base=False)
    paint(brk, special=mask_mul(lambda co, n: np.clip(np.maximum(1 - (co[:, 2] + t + Db + 0.04) / 0.3, 1 - (-co[:, 1]) / 0.06 * 0.0), 0, 1),
                                noise_mask(0.08, seed=p.seed + 3, lo=0.3, hi=0.7), min(1.0, p.rust * 1.6)),
          grime=mask_mul(noise_mask(0.15, seed=p.seed + 5), 0.4))
    brk2 = None

    # ================================================================ railing
    inset = 0.06
    hx, py = W / 2 - inset, PR - inset
    ps = 0.03
    oakrail = p.handrail == "oak"
    hr_h = 0.034 if oakrail else 0.017
    zb = 0.08
    ztr = H - hr_h - 0.005
    fh = 0.1
    zfr = ztr - 0.01 - fh if p.frieze else ztr
    v0 = zb + 0.005
    h = zfr - 0.005 - v0

    bays = max(1, int(ceil((2 * hx) / 1.5)))
    px = [-hx + 2 * hx * i / bays for i in range(bays + 1)]
    spans = []                                         # (start xy, angle deg, length)
    for i in range(bays):
        spans.append(((px[i] + ps / 2, -py), 0.0, px[i + 1] - px[i] - ps))
    side_len = py - 0.008 - ps / 2
    spans.append(((-hx, -0.008), -90.0, side_len))
    spans.append(((hx, -0.008), -90.0, side_len))

    # posts
    lant_sx = {"none": 0, "left": -1, "right": 1}[p.lantern]
    for sx in (-1, 1):
        iron_objs.append(box("post", (ps, ps, H - 0.03), loc=(sx * hx, -py, (H - 0.03) / 2), bevel=0.002, segments=1))
        iron_objs.append(box("post_base", (0.054, 0.054, 0.02), loc=(sx * hx, -py, 0.012), bevel=0.004, segments=1))
        iron_objs.append(box("post_band", (0.042, 0.042, 0.014), loc=(sx * hx, -py, zb + 0.03), bevel=0.003, segments=1))
        iron_objs.append(box("newel", (0.076, 0.076, 0.06), loc=(sx * hx, -py, H - 0.012), bevel=0.005, segments=2))
        if p.finial != "none" and sx != lant_sx:
            if p.finial == "ball":
                fp = [(0.0, 0.0), (0.024, 0.0), (0.026, 0.005), (0.016, 0.011), (0.011, 0.02), (0.015, 0.026)]
                fp += [(0.031 * cos(radians(a)), 0.056 + 0.031 * sin(radians(a))) for a in (-60, -35, -10, 15, 40, 62, 80)] + [(0.0, 0.087)]
            else:
                fp = [(0.0, 0.0), (0.024, 0.0), (0.026, 0.005), (0.016, 0.012), (0.012, 0.02), (0.022, 0.027), (0.03, 0.038), (0.032, 0.05),
                      (0.028, 0.055), (0.027, 0.068), (0.021, 0.086), (0.011, 0.1), (0.005, 0.106), (0.006, 0.113), (0.0, 0.118)]
            f = lathe("finial", fp, segments=12, loc=(sx * hx, -py, H + 0.018))
            assign(f, brass if p.gilt else iron)
            if p.gilt:
                paint(f, special=mask_mul(noise_mask(0.03, seed=p.seed + sx), 0.5), grime=0.25)
    for x in px[1:-1]:
        iron_objs.append(box("ipost", (0.024, 0.024, ztr), loc=(x, -py, ztr / 2), bevel=0.002, segments=1))
        iron_objs.append(box("ipost_base", (0.044, 0.044, 0.016), loc=(x, -py, 0.01), bevel=0.003, segments=1))
    pads = [box("pad", (0.075, 0.075, 0.006), loc=(x, -py, 0.003), bevel=0.002, segments=1) for x in px]
    for o in pads:
        assign(o, iron)

    # wall plates with rivets where the side rails are let into the wall
    wb = lathe("wr", [(0.0065, 0.0), (0.005, 0.003), (0.0, 0.0045)], segments=6)
    wm = []
    for sx in (-1, 1):
        for z in ([zb, ztr] + ([zfr] if p.frieze else [])):
            iron_objs.append(box("wplate", (0.07, 0.008, 0.05), loc=(sx * hx, -0.0045, z), bevel=0.002, segments=1))
            wm += [trs((sx * hx + dx, -0.0085, z), (90, 0, 0)) for dx in (-0.025, 0.025)]
        iron_objs.append(box("wplate_hr", (0.08, 0.01, 0.06), loc=(sx * hx, -0.0055, H - hr_h / 2 - 0.004), bevel=0.003, segments=1))
    iron_objs.append(instances_to_mesh("wrivets", wb, wm, keep_base=False))

    # rails, feet, handrail
    hr_objs = []
    if oakrail:
        hp = [(-0.022, 0.0), (0.022, 0.0), (0.029, 0.006), (0.03, 0.014), (0.026, 0.022), (0.017, 0.03), (0.006, 0.034),
              (-0.006, 0.034), (-0.017, 0.03), (-0.026, 0.022), (-0.03, 0.014), (-0.029, 0.006)]
    else:
        hp = [(-0.019, 0.0), (0.019, 0.0), (0.024, 0.004), (0.024, 0.008), (0.019, 0.013), (0.009, 0.017), (-0.009, 0.017),
              (-0.019, 0.013), (-0.024, 0.008), (-0.024, 0.004)]
    feet = []
    for si, (st_, ang, ln) in enumerate(spans):
        ca, sa = cos(radians(ang)), sin(radians(ang))
        mid = (st_[0] + ca * ln / 2, st_[1] + sa * ln / 2)
        for z in ([zb, ztr] + ([zfr] if p.frieze else [])):
            iron_objs.append(box("rail", (ln + 0.004, 0.03, 0.01), loc=(mid[0], mid[1], z), rot=(0, 0, ang), bevel=0.0015, segments=1))
        nf = max(1, int(round(ln / 0.5)))
        for k in range(1, nf):
            feet.append(trs((st_[0] + ca * ln * k / nf, st_[1] + sa * ln * k / nf, 0.0), (0, 0, ang)))
    if feet:
        ft = lathe("foot", [(0.0, 0.0), (0.015, 0.0), (0.016, 0.004), (0.009, 0.009), (0.0075, zb * 0.45), (0.012, zb * 0.55), (0.0075, zb * 0.68),
                            (0.0075, zb - 0.005)], segments=8)
        iron_objs.append(instances_to_mesh("feet", ft, feet, keep_base=False))
    # handrail: front between the newels, sides from the wall to the newels
    runs = [((-hx + 0.036, -py), (hx - 0.036, -py)), ((-hx, -0.001), (-hx, -(py - 0.036))), ((hx, -0.001), (hx, -(py - 0.036)))]
    for a, b in runs:
        ln = hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(ln / 0.14) + 1)
        path = [(a[0] + (b[0] - a[0]) * i / (n - 1), a[1] + (b[1] - a[1]) * i / (n - 1), H - hr_h) for i in range(n)]
        o = sweep("handrail", path, hp, smooth_path=False, up=(0, 0, 1))
        smooth(o, 50)
        hr_objs.append(o)
    if oakrail:
        hm = mat("wood_oak", p.stone if p.deck == "timber" else "#5a3d27", **({} if p.deck == "timber" else {"rough": 0.75}))
        for o in hr_objs:
            assign(o, hm)
            paint(o, wear=mask_mul(facing((0, 0, 1)), noise_mask(0.3, seed=p.seed + 21, lo=0.25, hi=0.7), 0.75), grime=mask_mul(facing((0, 0, -1)), 0.5))
    else:
        for o in hr_objs:
            assign(o, iron)
            paint(o, wear=mask_mul(facing((0, 0, 1)), noise_mask(0.25, seed=p.seed + 21, lo=0.3, hi=0.7), 0.8))

    # panels
    bases = {}
    bar_m, ring_m = [], []
    for si, (st_, ang, ln) in enumerate(spans):
        ca, sa = cos(radians(ang)), sin(radians(ang))
        n = max(1, int(round(ln / m_target)))
        m = ln / n
        key = round(m, 4)
        if key not in bases:
            pair = []
            for flip in (False, True):
                objs = fn(m, h, flip)
                for o in objs:
                    assign(o, iron)
                pair.append([join(objs, f"mod_{key}_{int(flip)}"), []])
            bases[key] = pair
        for i in range(n):
            u = m * i
            # mirror about the span centre so the motifs pair up symmetrically
            flip = (i % 2 == 1) if i < n / 2 else ((n - 1 - i) % 2 == 0 if p.pattern == "scroll" else (n - 1 - i) % 2 == 1)
            bases[key][int(flip)][1].append(trs((st_[0] + ca * u, st_[1] + sa * u, v0 + jr.jitter(0, 0.0008)), (0, jr.jitter(0, 0.25), ang)))
            if bound_bars and i > 0:
                bar_m.append(trs((st_[0] + ca * u, st_[1] + sa * u, v0), (0, jr.jitter(0, 0.35), ang + jr.pick([0, 90, 180, 270]))))
        if p.frieze:
            nr = max(1, int(floor(ln / (fh + 0.014))))
            sp = ln / nr
            for i in range(nr):
                u = (i + 0.5) * sp
                ring_m.append(trs((st_[0] + ca * u, st_[1] + sa * u, zfr + 0.005 + fh / 2), (0, jr.u(0, 360), ang)))
                if i > 0:
                    bar_m_fr = trs((st_[0] + ca * i * sp, st_[1] + sa * i * sp, zfr + 0.005 + fh / 2), (0, 0, ang))
                    ring_m.append(None)
                    ring_m[-1] = ("stud", bar_m_fr)
    for key, pair in bases.items():
        for base, ms in pair:
            if ms:
                iron_objs.append(instances_to_mesh(f"panels_{key}", base, ms, keep_base=False))
            else:
                delete(base)
    if bar_m:
        bbar = sweep("bbar", [(0, 0, 0), (0, 0, h)], _cham(BS, BS, 0.0015), smooth_path=False, up=(0, 1, 0))
        iron_objs.append(instances_to_mesh("bars", bbar, bar_m, keep_base=False))
    if p.frieze:
        rr = fh / 2 - 0.0025
        rb = sweep("fring", [(rr * cos(2 * pi * i / 12), 0, rr * sin(2 * pi * i / 12)) for i in range(12)], _rect(0.005, 0.015),
                   closed=True, smooth_path=False, up=(0, 1, 0))
        iron_objs.append(instances_to_mesh("frieze", rb, [m_ for m_ in ring_m if not isinstance(m_, tuple)], keep_base=False))
        studs = [m_[1] for m_ in ring_m if isinstance(m_, tuple)]
        if studs:
            sb = box("stud", (0.016, 0.02, 0.012))
            iron_objs.append(instances_to_mesh("frieze_studs", sb, studs, keep_base=False))

    for o in iron_objs:
        if not o.data.materials:
            assign(o, iron)
        paint(o, special=mask_mul(lambda co, n: np.clip(np.maximum(1 - co[:, 2] / 0.2, 0.55 * (1 - np.abs(co[:, 2] - zfr) / 0.05)), 0, 1),
                                  noise_mask(0.06, seed=p.seed + 9, lo=0.3, hi=0.7), min(1.0, p.rust * 1.5)),
              grime=mask_mul(noise_mask(0.2, seed=p.seed + 13), 0.35))

    # ================================================================ flower pots in hooked ring holders on the front rail
    npot = int(p.pots)
    pr = rng.fork("pots")
    for i in range(npot):
        r = pr.fork(f"pot{i}")
        keep = min(0.5, 0.55 * hx) if lant_sx else 0.0          # keep foliage out of the lantern's hot spot
        xa, xb = (-hx + keep, hx) if lant_sx < 0 else (-hx, hx - keep)
        x = xa + (i + 0.5) * ((xb - xa) / npot) + r.jitter(0, min(0.08, (xb - xa) / npot * 0.25))
        objs, ph = _pot(r, p.flower, 0.0)
        k = ph / 0.15
        ring_r = 0.083 * k
        yc = -(py + 0.03 + ring_r + 0.012)
        zp = H - 0.03 - ph + r.jitter(0, 0.01)
        zr = ph - 0.034
        hold = [torus("pring", ring_r, 0.0042, loc=(0, 0, zr), major_seg=12, minor_seg=4)]
        for dx in (-0.045 * k, 0.045 * k):
            yb = sqrt(max(ring_r ** 2 - dx ** 2, 0.0))
            y_r = -(yc + py)                      # rail centre in pot space
            top = H - zp + 0.004
            path = [(dx, yb, zr), (dx, y_r - 0.036, zr + 0.02), (dx, y_r - 0.036, top - 0.006), (dx, y_r - 0.03, top), (dx, y_r + 0.03, top),
                    (dx, y_r + 0.036, top - 0.006), (dx, y_r + 0.036, top - 0.05)]
            hold.append(sweep("hook", path, _rect(0.014, 0.004), smooth_path=False, up=(1, 0, 0)))
        for o in hold:
            assign(o, iron)
            paint(o, special=mask_mul(noise_mask(0.04, seed=p.seed + i), min(1.0, 0.4 + p.rust)))
        grp = join(objs + hold, f"pot{i}")
        place(grp, loc=(x, yc, zp), rot=(0, 0, r.u(0, 360) * 0.0 + r.jitter(0, 6)))

    # ================================================================ oil lantern hung from a forged crook on one corner post
    if lant_sx:
        xl = lant_sx * hx
        pts, _, _ = _pick((py, H + 0.016), 0, 0, (py + 0.12, H + 0.56), 0.12, 0.62, "C", lambda p1, p2: p2[0], inner=0.45, chord=0.018)
        tip = pts[-1]
        sp_, _, _ = _pick((py + 0.011, H + 0.2), 0, 0, (py + 0.047, H + 0.455), 0.03, 1.3, "C", lambda p1, p2: p2[0], inner=0.28, chord=0.014)
        crook = [_bar("crook", pts, _rect(0.015, 0.015), "YZ", "end"), _bar("crook_scroll", sp_, _rect(ST, SD * 0.8), "YZ", "end")]
        place(crook, loc=(xl, 0, 0))
        crook.append(box("crook_col", (0.03, 0.034, 0.02), loc=(xl, -py - 0.002, H + 0.2), bevel=0.003, segments=1))
        crook.append(lathe("crook_base", [(0.0, 0.0), (0.026, 0.0), (0.028, 0.005), (0.017, 0.012), (0.012, 0.022), (0.016, 0.03), (0.011, 0.04)],
                           segments=10, loc=(xl, -py, H + 0.018)))
        for o in crook:
            assign(o, iron)
            paint(o, special=mask_mul(noise_mask(0.06, seed=p.seed + 31, lo=0.4, hi=0.75), min(1.0, p.rust)))
        lo = [torus("l_link", 0.014, 0.0028, loc=(0, 0, -0.012), rot=(90, 0, 90), major_seg=8, minor_seg=4),
              torus("l_link", 0.014, 0.0028, loc=(0, 0, -0.035), rot=(90, 0, 0), major_seg=8, minor_seg=4),
              torus("l_link", 0.014, 0.0028, loc=(0, 0, -0.058), rot=(90, 0, 90), major_seg=8, minor_seg=4),
              torus("l_ring", 0.011, 0.003, loc=(0, 0, -0.083), rot=(90, 0, 0), major_seg=8, minor_seg=4),
              cylinder("l_vent", 0.021, 0.016, loc=(0, 0, -0.099), verts=8),
              cone("l_roof", 0.096, 0.02, 0.05, loc=(0, 0, -0.13), rot=(0, 0, 45), verts=4),
              box("l_eave", (0.14, 0.14, 0.008), loc=(0, 0, -0.158), bevel=0.002, segments=1),
              box("l_base", (0.125, 0.125, 0.012), loc=(0, 0, -0.316), bevel=0.003, segments=1),
              lathe("l_drop", [(0.0, -0.034), (0.007, -0.028), (0.011, -0.018), (0.005, -0.008), (0.014, 0.0)], segments=8, loc=(0, 0, -0.322))]
        for ax in (-1, 1):
            for ay in (-1, 1):
                lo.append(box("l_bar", (0.007, 0.007, 0.15), loc=(ax * 0.053, ay * 0.053, -0.236)))
        for o in lo:
            assign(o, iron)
        burner = lathe("l_burner", [(0.0, 0.0), (0.024, 0.0), (0.027, 0.008), (0.022, 0.02), (0.008, 0.026), (0.007, 0.04), (0.0, 0.04)], segments=10, loc=(0, 0, -0.31))
        assign(burner, brass)
        glass = box("l_glass", (0.1, 0.1, 0.15), loc=(0, 0, -0.236))
        # CLEARANCE RULE (depth fighting): a glowing sheet keeps >= 4 mm to any parallel surface - the top and bottom of
        # the glass box lay 1 mm inside the eave / base plates (never seen), so only the four panes are kept
        bm_g = bmesh.new()
        bm_g.from_mesh(glass.data)
        bmesh.ops.delete(bm_g, geom=[f for f in bm_g.faces if abs(f.normal.z) > 0.5], context="FACES")
        bm_g.to_mesh(glass.data)
        bm_g.free()
        assign(glass, mat("glass_lantern", glow=0.55))
        flame = sphere("l_flame", 0.011, loc=(0, 0, -0.25), segments=6, rings=4, scale=(1, 1, 2.2))
        assign(flame, mat("emissive", "#ffb45e", glow=7))
        lo += [burner, glass, flame]
        hook = (xl, -tip[0], tip[1] - 0.008)
        for o in lo:
            o.location = (o.location[0] + hook[0], o.location[1] + hook[1], o.location[2] + hook[2])
        part(lo, "lantern", pivot=hook, motion="swing", amp=2.5, speed=0.35)
        light("lantern", (hook[0], hook[1], hook[2] - 0.25), intensity=p.lantern_cd, range=5.0, flicker=0.3, shadow_near=0.45)

    # ================================================================ sockets
    socket("door", "attach", (0, 0, 0), role="door")
    socket("planter_front", "attach", (0, -(py + 0.03), H), role="planter")
    socket("lantern_under", "attach", (0, -(PR - 0.12), -t), role="lantern")
    socket("deck", "attach", (0, -PR / 2, 0), role="prop_top")
