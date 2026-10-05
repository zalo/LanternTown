"""Roof ornaments for a ridge running along X: a turned finial (ball-and-spike, crossed fleur-de-lis,
pineapple or star-topped) and / or a run of pierced ridge cresting, in three real constructions:
* terracotta - socketed half-round ridge tiles bedded in lime mortar, each with its own pierced crest fin;
  the finial is thrown in sections threaded on an iron rod;
* lead - wood-cored lead roll with dressed wings, lapped sheets and tacks, cast lead cresting, beaten finial;
* iron - zinc ridge saddle carrying a bolted base rail, cast-iron cresting panels between spear-headed
  standards, a slender forged finial braced by two scrolls.
Origin: on the ridge line at the middle of the run (z = 0 is the bed the ridge capping sits on).
"""
from tk import *
import numpy as np
from math import sin, cos, tan, pi, radians, sqrt, hypot

META = {"name": "Roof finial and ridge cresting", "class": "prop_small", "mount": "ground",
        "tri_budget": 15000,     # up to 3 m of individually pierced cresting units plus one or two turned finials
        "focus": [-0.6, 0.0, 0.4]}
PARAMS = {
    "mode": {"default": "both", "choices": ["finial", "cresting", "both"], "desc": "finial only, cresting run only, or a run with finial(s)"},
    "height": {"default": 0.78, "min": 0.45, "max": 1.6, "desc": "finial height from the ridge bed to the tip (m)"},
    "material": {"default": "terracotta", "choices": ["terracotta", "lead", "iron"], "desc": "construction / material"},
    "motif": {"default": "ball_spike", "choices": ["ball_spike", "fleur", "pineapple", "star"], "desc": "finial motif"},
    "cresting_length": {"default": 1.8, "min": 0.6, "max": 3.0, "desc": "length of the cresting run along the ridge (m)"},
    "crest_style": {"default": "trefoil", "choices": ["trefoil", "spear", "rings"], "desc": "pierced cresting pattern"},
    "crest_height": {"default": 0.2, "min": 0.1, "max": 0.34, "desc": "height of the cresting above the ridge crown (m)"},
    "finial_at": {"default": "end", "choices": ["end", "centre", "both_ends"], "desc": "where the finial stands on the run (mode both)"},
    "ridge_width": {"default": 0.23, "min": 0.19, "max": 0.3, "desc": "width of the ridge capping (m)"},
    "gilt": {"default": False, "desc": "gold-leafed ball / star / fleur (metal finials; the star on clay ones)"},
    "clay_tint": {"default": "#e3a57f", "desc": "tint of the terracotta"},
    "paint": {"default": "#1f2a22", "desc": "paint colour of the ironwork"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: moss, staining, settled tiles, broken crests above 0.6"},
}
VARIANTS = {
    "lead_fleur": {"material": "lead", "motif": "fleur", "height": 1.05, "crest_style": "spear", "crest_height": 0.24,
                   "cresting_length": 2.4, "finial_at": "both_ends", "gilt": True, "age": 0.35, "seed": 4},
    "iron_star": {"material": "iron", "motif": "star", "height": 1.3, "crest_style": "rings", "crest_height": 0.22,
                  "cresting_length": 2.25, "finial_at": "centre", "gilt": True, "paint": "#2e4a45", "age": 0.5, "seed": 9},
    "pineapple_solo": {"mode": "finial", "motif": "pineapple", "height": 0.95, "clay_tint": "#e8c9a4", "ridge_width": 0.26,
                       "age": 0.3, "seed": 12},
    "old_cresting": {"mode": "cresting", "crest_style": "spear", "crest_height": 0.26, "cresting_length": 2.7,
                     "clay_tint": "#b8856d", "age": 0.95, "seed": 23},
}


# ----------------------------------------------------------------------------------------- helpers
def _sm(ctrl, n):
    """Smooth (Catmull-Rom) lathe profile section through control points [(r, z), ...]."""
    P = resample_path([(r, 0.0, z) for r, z in ctrl], n=n)
    return [(max(float(q[0]), 0.0005), float(q[2])) for q in P]


def _simplify(pts, tol):
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        dx, dy = c[0] - a[0], c[1] - a[1]
        l = hypot(dx, dy)
        d = abs(dx * (b[1] - a[1]) - dy * (b[0] - a[0])) / l if l > 1e-9 else 1.0
        if d > tol and hypot(b[0] - a[0], b[1] - a[1]) > 2e-4:
            out.append(b)
    out.append(pts[-1])
    return out


def _run_piece(name, prof, xa, xb, z=0.0, lap=0.0, lap_scale=1.0, step=None):
    """A length of ridge capping (cross-section `prof`) from xa to xb; the first `lap` metres are a
    slightly bigger socket / lap that covers the end of the previous piece."""
    xs = [xa]
    if lap > 0:
        xs += [xa + lap, xa + lap + 0.0015]
    x = xs[-1]
    k = max(1, int(round((xb - x) / step))) if step else 1
    xs += [x + (xb - x) * (i + 1) / k for i in range(k)]
    tot = xb - xa

    def sc(t):
        return lap_scale if (lap > 0 and t * tot <= lap + 1e-5) else 1.0

    return sweep(name, [(q, 0.0, z) for q in xs], prof, smooth_path=False, scale_fn=sc, up=(0, 0, 1))


def _crest_shape(style, wu, h, yb, hole_k, jag=None):
    """Outline and holes of one cresting unit: x in [-wu/2, wu/2], y in [0, h]; yb = top of the solid band.
    The top edge is a height field built from round lobes and pointed (two-centred) arches."""
    hp = h - yb
    circles, goths, holes, brk = [], [], [], []
    if style == "trefoil":
        m = max(1, int(round(wu / (hp * 1.75))))
        wm = wu / m
        for k in range(m):
            cx = -wu / 2 + wm * (k + 0.5)
            rc = min(0.30 * hp, 0.2 * wm)
            rs = rc * 0.88
            sx = min(rc + rs * 0.5, wm / 2 - rs - 0.003)
            circles += [(cx, h - rc, rc), (cx - sx, yb + rs * 0.8, rs), (cx + sx, yb + rs * 0.8, rs)]
            holes += [(cx, h - rc, rc * hole_k, "c"), (cx - sx, yb + rs * 0.8, rs * hole_k, "c"), (cx + sx, yb + rs * 0.8, rs * hole_k, "c")]
        for k in range(m + 1):
            circles.append((-wu / 2 + wm * k, yb, 0.11 * hp))
    elif style == "spear":
        m = max(1, int(round(wu / (hp * 0.85))))
        wm = wu / m
        a = 0.37 * wm
        for k in range(m):
            cx = -wu / 2 + wm * (k + 0.5)
            tall = (k % 2 == 0)
            top = h if tall else yb + 0.7 * hp
            rk = 0.075 * hp if tall else 0.0
            ht = top - yb - 1.3 * rk
            goths.append((cx, a, ht))
            if rk > 0:
                circles.append((cx, top - rk, rk))
            holes.append((cx, yb + 0.16 * ht, (a * (hole_k + 0.02), ht * (hole_k + 0.02)), "g"))
        for k in range(m + 1):
            circles.append((-wu / 2 + wm * k, yb, 0.13 * hp))
    else:                                    # rings
        rk = 0.065 * hp
        m = max(1, int(np.ceil(wu / (hp - 1.4 * rk + 0.003))))
        wm = wu / m
        rr = min(wm / 2 - 0.0015, 0.5 * (hp - 1.4 * rk))
        cy = h - 1.4 * rk - rr
        for k in range(m):
            cx = -wu / 2 + wm * (k + 0.5)
            circles += [(cx, cy, rr), (cx, cy + rr + 0.3 * rk, rk)]
            holes.append((cx, cy, rr * min(hole_k + 0.14, 0.8), "c"))
        rsp = min(0.36 * (cy - yb), 0.28 * wm)
        if rsp > 0.007 and wm / 2 - rr < 0.004:
            for k in range(1, m):
                holes.append((-wu / 2 + wm * k, yb + (cy - yb) * 0.42, rsp, "c"))

    def ytop(x):
        y = yb
        for (cx, cy_, r) in circles:
            d = abs(x - cx)
            if d <= r:
                y = max(y, cy_ + sqrt(r * r - d * d))
        for (cx, a_, ht) in goths:
            d = abs(x - cx)
            if d < a_:
                rg = (ht * ht + a_ * a_) / (2 * a_)
                y = max(y, yb + sqrt(max(rg * rg - (d + rg - a_) ** 2, 0.0)))
        return y

    xl, xr = -wu / 2, wu / 2
    xs = list(np.linspace(xl, xr, 80))
    for (cx, cy_, r) in circles:
        brk += [cx - r - 1e-4, cx - r + 1e-4, cx + r - 1e-4, cx + r + 1e-4, cx]
    for (cx, a_, ht) in goths:
        brk += [cx - a_, cx + a_, cx - 1e-4, cx + 1e-4]
    xs = sorted(set([float(x) for x in xs] + [x for x in brk if xl < x < xr]))
    top = [(x, ytop(x)) for x in xs]
    if jag is not None:                      # snapped-off unit: clip the top with a jagged break line
        jx, jy = jag
        top = [(x, min(y, float(np.interp(x, jx, jy)))) for x, y in top]
    top = _simplify(top, 0.0008)
    outer = [(xl, 0.0), (xr, 0.0)] + list(reversed(top))
    hl = []
    for (cx, cy_, r, kind) in holes:
        if kind == "c":
            ext_top, hw = cy_ + r, r
        else:
            ext_top, hw = cy_ + r[1], r[0]
        if jag is not None and ext_top + 0.012 > float(np.min(np.interp(np.linspace(cx - hw - 0.01, cx + hw + 0.01, 9), jag[0], jag[1]))):
            continue
        if cx - hw < xl + 0.006 or cx + hw > xr - 0.006:
            continue
        if kind == "c":
            hl.append(circle_pts(r, n=10, center=(cx, cy_)))
        else:                                # lancet: pointed top, round foot
            a_, ht = r
            rg = (ht * ht + a_ * a_) / (2 * a_)
            pts = []
            for i in range(7):               # right side going up
                d = a_ * (1 - i / 6.0)
                pts.append((cx + d, cy_ + a_ * 0.6 + sqrt(max(rg * rg - (d + rg - a_) ** 2, 0.0))))
            for i in range(1, 7):            # left side going down
                d = a_ * i / 6.0
                pts.append((cx - d, cy_ + a_ * 0.6 + sqrt(max(rg * rg - (d + rg - a_) ** 2, 0.0))))
            for i in range(1, 6):            # round foot
                an = pi + pi * i / 6.0
                pts.append((cx + a_ * cos(an), cy_ + a_ * 0.6 + a_ * 0.6 * sin(an)))
            hl.append(pts)
    return outer, hl


_FLEUR = [(0.0, 0.0), (0.05, 0.04), (0.125, 0.0), (0.17, 0.035), (0.135, 0.11), (0.09, 0.19), (0.075, 0.25), (0.125, 0.255),
          (0.125, 0.335), (0.2, 0.40), (0.26, 0.405), (0.285, 0.37), (0.33, 0.335), (0.38, 0.39), (0.39, 0.48), (0.345, 0.58),
          (0.27, 0.63), (0.2, 0.61), (0.155, 0.54), (0.125, 0.46), (0.095, 0.40), (0.06, 0.40), (0.062, 0.48), (0.10, 0.6),
          (0.12, 0.7), (0.10, 0.82), (0.05, 0.93), (0.0, 1.0)]


def _fleur_outline(hfl, n=54):
    pts = _FLEUR + [(-x, y) for x, y in reversed(_FLEUR[1:-1])]
    P = resample_path([(x * hfl * 0.95, y * hfl, 0.0) for x, y in pts], n=n, closed=True)
    return [(float(q[0]), float(q[1])) for q in P]


def _star(name, rs, th, edge, loc, rz):
    vs = [(0.0, -th, 0.0), (0.0, th, 0.0)]
    for k in range(10):
        a = radians(90 + 36 * k)
        r = rs if k % 2 == 0 else rs * 0.42
        vs += [(r * cos(a), -edge, r * sin(a)), (r * cos(a), edge, r * sin(a))]
    faces = []
    for k in range(10):
        j = (k + 1) % 10
        f0, b0, f1, b1 = 2 + 2 * k, 3 + 2 * k, 2 + 2 * j, 3 + 2 * j
        faces += [(0, f1, f0), (1, b0, b1), (f0, f1, b1, b0)]
    o = mesh_object(name, vs, faces)
    recalc_normals(o)
    flat(o)
    bevel(o, 0.0012, 1, angle=20)
    place(o, loc=loc, rot=(0, 0, rz))
    return o


def _leaf(name, ln, wd, curl, th):
    def fn(u, v):
        wv = wd * (sin(pi * (0.14 + 0.86 * v))) ** 0.7
        x = (u - 0.5) * wv
        return (x, -curl * ln * v * v + 0.3 * abs(x), v * ln)
    o = surface(name, fn, 2, 4, thickness=th)
    realize(o)
    return o


# ------------------------------------------------------------------------------------------- build
def build(p, rng):
    M = p.material
    slim = (M == "iron")
    age = float(p.age)
    R = p.ridge_width / 2
    H = float(p.height)
    seed = int(p.seed)

    gold = mat("brass", "#ffe9b8", rough=0.75)
    iron_w = mat("iron_wrought")
    if M == "terracotta":
        body = mat("terracotta", p.clay_tint)
        saddle_m = body
        star_m = gold if p.gilt else iron_w
        ball_m = body
    elif M == "lead":
        body = mat("lead")
        saddle_m = body
        star_m = ball_m = gold if p.gilt else body
    else:
        body = mat("iron_painted", p.paint)
        saddle_m = mat("zinc")
        star_m = ball_m = gold if p.gilt else body

    # ---- ridge capping section and crown height
    tt = 0.016                               # clay tile thickness
    lift = 0.006                             # mortar bed under the clay tiles
    if M == "terracotta":
        zc = lift + R
    else:
        rr = 0.03                            # lead roll radius (timber core + sheet)
        pa = radians(32)
        xw = 0.906 * rr
        ya = 0.005 + (R - xw) * tan(pa)
        cyr = ya + 0.42 * rr
        zc = cyr + rr

    # ---- layout of the run in units
    u0 = {"terracotta": 0.30, "lead": 0.42, "iron": 0.45}[M]
    if p.mode == "finial":
        L = {"terracotta": 0.36, "lead": 0.5, "iron": 0.52}[M]
        n = 1
        fin_units = [0]
    else:
        L = float(p.cresting_length)
        n = max(2, int(round(L / u0)))
        fin_units = []
        if p.mode == "both":
            if p.finial_at == "centre":
                n += (n % 2 == 0)
                fin_units = [n // 2]
            elif p.finial_at == "both_ends":
                n = max(n, 3)
                fin_units = [0, n - 1]
            else:
                fin_units = [0]
    u = L / n
    x0 = -L / 2

    body_objs, saddle_objs, bed_objs, iron_objs, gold_objs, ball_objs, star_objs, moss_objs = [], [], [], [], [], [], [], []

    # ---- unit transforms (hand-laid: nothing is dead straight)
    rj = rng.fork("lay")
    amp = 0.35 + 0.65 * age
    unit_M = []
    for i in range(n):
        if M == "terracotta":
            unit_M.append(trs((x0 + i * u + rj.jitter(0, 0.0015), rj.jitter(0, 0.002 * amp), rj.u(0, 0.003 * amp)),
                              (rj.jitter(0, 1.6 * amp), rj.jitter(0, 0.5 * amp), rj.jitter(0, 0.7 * amp))))
        else:
            unit_M.append(trs((x0 + i * u, 0, 0), (rj.jitter(0, 1.3 * amp), 0, rj.jitter(0, 0.3 * amp))))

    # ---- ridge capping
    if M == "terracotta":
        arc_o = [(R * cos(radians(a)), R * sin(radians(a))) for a in np.linspace(0, 180, 9)]
        arc_i = [((R - tt) * cos(radians(a)), (R - tt) * sin(radians(a))) for a in np.linspace(180, 0, 9)]
        tile = _run_piece("tile", arc_o + arc_i, 0.0, u - 0.002, z=lift, lap=0.052, lap_scale=1.1)
        bevel(tile, 0.0025, 1)
        saddle_objs.append(instances_to_mesh("ridge_tiles", tile, unit_M, keep_base=False))
        # lime mortar: bedding squeezed out along both edges and filling the open ends of the run
        ri = R - tt + 0.001
        core = [(R + 0.024, 0.0), (R + 0.006, 0.02), (ri, 0.02)] + \
               [(ri * cos(radians(a)), ri * sin(radians(a)) + lift) for a in np.linspace(15, 165, 6)] + \
               [(-ri, 0.02), (-R - 0.006, 0.02), (-R - 0.024, 0.0)]
        bed = _run_piece("bedding", core, x0 + 0.005, x0 + L - 0.006, step=0.1)
        displace_noise(bed, 0.004, 0.06, seed=seed + 1, axis_mask=(0.6, 1, 0.5))
        transform_verts(bed, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
        bed_objs.append(bed)
    else:
        prof = [(R, 0.0), (R, 0.005), (xw, ya)] + \
               [(rr * cos(radians(a)), cyr + rr * sin(radians(a))) for a in np.linspace(-25, 205, 9)][1:-1] + \
               [(-xw, ya), (-R, 0.005), (-R, 0.0)]
        nsh = max(1, int(np.ceil(L / 1.05)))
        for k in range(nsh):
            xa, xb = x0 + L * k / nsh, x0 + L * (k + 1) / nsh
            sh = _run_piece(f"sheet{k}", prof, xa, xb, lap=0.07 if k > 0 else 0.0, lap_scale=1.045, step=0.09)
            displace_noise(sh, 0.0022 if M == "lead" else 0.0012, 0.07, seed=seed + 2 + k)
            transform_verts(sh, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
            saddle_objs.append(sh)
        # tacks (clips) holding the free edges of the wings
        clip = box("clip", (0.034, 0.03, 0.004), bevel=0.001, segments=1)
        nc = max(2, int(round(L / 0.42)))
        cm = []
        for k in range(nc):
            xk = x0 + L * (k + 0.5) / nc + rj.jitter(0, 0.03)
            for sy in (-1, 1):
                cm.append(trs((xk, sy * (R - 0.01), 0.011), (sy * 32.0, 0, rj.jitter(0, 4))))
        saddle_objs.append(instances_to_mesh("tacks", clip, cm, keep_base=False))

    # ---- cresting
    ch = float(p.crest_height)
    bury = 0.014
    band = ch * {"terracotta": 0.2, "lead": 0.16, "iron": 0.1}[M]
    th = {"terracotta": 0.02, "lead": 0.011, "iron": 0.009}[M]
    hole_k = {"terracotta": 0.5, "lead": 0.56, "iron": 0.64}[M]
    gap = {"terracotta": 0.005, "lead": 0.003, "iron": 0.011}[M]
    bev = {"terracotta": 0.004, "lead": 0.003, "iron": 0.002}[M]
    z_rail = zc + 0.011                      # top of the iron base rail
    fin_z0 = (z_rail if slim else zc) - bury
    crest_units = [i for i in range(n) if i not in fin_units] if p.mode != "finial" else []
    if crest_units:
        rb_ = rng.fork("break")
        broken = [i for i in crest_units if age > 0.6 and rb_.chance((age - 0.5) * 0.55)]
        if age > 0.85 and not broken:
            broken = [crest_units[len(crest_units) // 2]]
        wu = u - 2 * gap
        h = ch + bury
        yb = bury + band

        def fin_mesh(name, jag=None):
            outer, holes = _crest_shape(p.crest_style, wu, h, yb, hole_k, jag)
            o = shape_extrude(name, outer, holes, depth=th, loc=(u / 2, th / 2, fin_z0), rot=(90, 0, 0), bevel=bev, bevel_res=0)
            apply_transform(o)
            if not slim:                     # roll moulding where the fin springs from the capping
                rolls = [cylinder(f"{name}_roll{sy}", 0.008, wu - 0.004, loc=(u / 2, sy * (th / 2 + 0.0035), zc + 0.0035), rot=(0, 90, 0),
                                  verts=6, cap=True) for sy in (-1, 1)]
                for q in rolls:
                    smooth(q, 80)
                o = join([o] + rolls, name)
            return o

        whole = [unit_M[i] for i in crest_units if i not in broken]
        if whole:
            body_objs.append(instances_to_mesh("crest", fin_mesh("fin"), whole, keep_base=False))
        for i in broken:
            jx = np.linspace(-wu / 2, wu / 2, 9)
            jy = np.array([yb + (h - yb) * rb_.u(0.05, 0.5) for _ in jx])
            jy[rb_.pick([0, 8])] = h + 0.01                      # one side still stands to full height
            body_objs.append(instances_to_mesh(f"crest_broken{i}", fin_mesh(f"finb{i}", (jx, jy)), [unit_M[i]], keep_base=False))

    if slim:
        # base rail on the roll, coach bolts, spear-headed standards between the panels
        rail = box("rail", (L - 0.012, 0.036, 0.012), loc=(0, 0, zc + 0.005), bevel=0.002, segments=1)
        densify(rail, 0.25)
        body_objs.append(rail)
        bt = cylinder("bolt", 0.0062, 0.0048, loc=(0, 0, 0.0024), verts=6)
        apply_transform(bt)
        bm_ = []
        for i in range(n):
            for fx_ in (0.09, 0.5, 0.91):
                for sy in (-1, 1):
                    bm_.append(trs((x0 + (i + fx_) * u, sy * 0.0115, z_rail - 0.0005), (0, 0, rj.u(0, 60))))
        iron_objs.append(instances_to_mesh("bolts", bt, bm_, keep_base=False))
        if p.mode != "finial":
            hw = 0.008
            ph = ch * 0.86
            pst = box("post_bar", (hw * 2, hw * 2, ph), loc=(0, 0, ph / 2), bevel=0.0015, segments=1)
            col = box("post_collar", (hw * 2 + 0.008, hw * 2 + 0.008, 0.01), loc=(0, 0, ph - 0.012), bevel=0.002, segments=1)
            foot = box("post_foot", (hw * 2 + 0.012, 0.034, 0.012), loc=(0, 0, 0.006), bevel=0.002, segments=1)
            sp = lathe("post_spear", [(hw * 0.8, ph - 0.004), (hw * 2.3, ph + 0.022), (hw * 1.2, ph + 0.05), (0.0, ph + 0.085)],
                       segments=4, rot=(0, 0, 45), smooth_angle=20)
            post = join([pst, col, foot, sp], "post")
            pm = [trs((x0 + i * u + (0.006 if i == 0 else (-0.006 if i == n else 0)), 0, z_rail - 0.001),
                      (rj.jitter(0, 1.2 * amp), rj.jitter(0, 1.0 * amp), 0)) for i in range(n + 1)]
            body_objs.append(instances_to_mesh("posts", post, pm, keep_base=False))

    # ---- finial(s)
    def finial(xf, r, idx):
        fo = {"body": [], "ball": [], "star": [], "iron": []}
        hf = H - zc
        kr = clamp((hf / 0.63) ** 0.55, 0.72, 1.7)
        seg = 20 if not slim else 16
        fr = {"ball_spike": 0.44, "fleur": 0.38, "pineapple": 0.36, "star": 0.42}[p.motif]
        rb = min(0.05 * kr, R - 0.03) if slim else min(0.082 * kr, R - 0.014)
        ball_r = rb * (1.15 if slim else 0.92)
        hu = hf * (1 - fr)                   # the motif keeps sane proportions on tall finials: the stem takes the rest
        if p.motif == "fleur":
            hu = min(hu, 0.44 * kr)
        elif p.motif == "pineapple":
            hu = min(hu, 6.3 * ball_r)
        hs = hf - hu
        zs = zc + hs

        def Z(f):
            return zc + f * hs

        if not slim:
            rsk = min(rb * 1.32, R - 0.004)
            zsk = max(0.012, (lift + sqrt(R * R - rsk * rsk) if M == "terracotta" else 0.005 + (R - rsk) * tan(pa)) - 0.008)
            # apron dressed / moulded down onto the ridge capping
            pr = _sm([(rsk, zsk), (rb * 1.14, zsk + 0.55 * (zc - zsk)), (rb * 1.04, zc + 0.006)], 5)
            pr += [(rb * 1.02, Z(0.05)), (rb * 1.1, Z(0.07)), (rb * 1.14, Z(0.105)), (rb * 1.1, Z(0.14)), (rb * 0.95, Z(0.16))]
            pr += _sm([(rb * 0.93, Z(0.165)), (rb * 0.68, Z(0.21)), (rb * 0.52, Z(0.30)), (rb * 0.5, Z(0.37)), (rb * 0.56, Z(0.405))], 6)
            pr += [(rb * 0.7, Z(0.415)), (rb * 0.76, Z(0.445)), (rb * 0.7, Z(0.475))]
            pr += _sm([(rb * 0.54, Z(0.49)), (rb * 0.62, Z(0.54)), (rb * 0.92, Z(0.63)), (rb * 1.0, Z(0.72)), (rb * 0.9, Z(0.80)),
                       (rb * 0.6, Z(0.89)), (rb * 0.44, Z(0.94))], 9)
            pr += [(rb * 0.58, Z(0.95)), (rb * 0.6, Z(0.975)), (rb * 0.42, Z(1.0))]
            rn = rb * 0.42
        else:
            zb = max((R - rb * 1.2) * tan(pa) - 0.004, 0.012)
            pr = [(rb * 1.2, zb), (rb * 1.2, Z(0.015)), (rb * 1.05, Z(0.03))]
            pr += _sm([(rb * 1.0, Z(0.035)), (rb * 0.66, Z(0.07)), (rb * 0.42, Z(0.13)), (rb * 0.34, Z(0.19))], 5)
            pr += [(rb * 0.52, Z(0.2)), (rb * 0.6, Z(0.22)), (rb * 0.52, Z(0.24)), (rb * 0.32, Z(0.25)), (rb * 0.32, Z(0.44))]
            pr += _sm([(rb * 0.34, Z(0.46)), (rb * 0.5, Z(0.54)), (rb * 0.6, Z(0.62)), (rb * 0.46, Z(0.72)), (rb * 0.32, Z(0.8))], 7)
            pr += [(rb * 0.32, Z(0.92)), (rb * 0.52, Z(0.93)), (rb * 0.6, Z(0.95)), (rb * 0.52, Z(0.97)), (rb * 0.32, Z(1.0))]
            rn = rb * 0.32
        lower = lathe("finial_lower", pr, segments=seg)
        fo["body"].append(lower)

        if p.motif == "ball_spike":
            Rb = min(ball_r, hu * 0.22)
            zbc = zs + Rb * 0.9
            fo["ball"].append(sphere("finial_ball", Rb, loc=(0, 0, zbc), segments=seg, rings=10, scale=(1, 1, 0.94)))
            za = zbc + Rb * 0.9
            ztip = H - (0.035 if M == "terracotta" else 0.0)
            ls = ztip - za - 0.03
            sp = [(rn * 0.8, za - 0.012), (rn * 1.3, za + 0.006), (rn * 1.42, za + 0.016), (rn * 1.1, za + 0.026)]
            if slim:
                sp += _sm([(rn * 0.8, za + 0.03), (rn * 0.7, za + 0.03 + 0.3 * ls), (rn * 1.7, za + 0.03 + 0.5 * ls),
                           (rn * 1.1, za + 0.03 + 0.72 * ls), (rn * 0.3, za + 0.03 + 0.96 * ls)], 9)
            else:
                sp += _sm([(rn * 0.86, za + 0.03), (rn * 0.74, za + 0.03 + 0.2 * ls), (rn * 0.5, za + 0.03 + 0.5 * ls),
                           (rn * 0.3, za + 0.03 + 0.8 * ls), (rn * 0.16, za + 0.03 + 0.97 * ls)], 7)
            sp += [(0.0, ztip)]
            fo["body"].append(lathe("finial_spike", sp, segments=12))
            if M == "terracotta":            # the iron rod the sections are threaded on shows at the tip
                fo["iron"].append(cylinder("finial_rod", 0.0045, 0.06, loc=(0, 0, H - 0.03), verts=6))
        elif p.motif == "fleur":
            rk = rn * 1.9
            fo["ball"].append(sphere("finial_knop", rk, loc=(0, 0, zs + rk * 0.7), segments=14, rings=8, scale=(1, 1, 0.85)))
            zp0 = zs + rk * 0.75
            hfl = H - zp0
            tf = {"terracotta": 0.03, "lead": 0.018, "iron": 0.011}[M]
            ol = _fleur_outline(hfl - 0.004)
            a = shape_extrude("fleur_a", ol, depth=tf, loc=(0, tf / 2, zp0), rot=(90, 0, 0), bevel=tf * 0.3, bevel_res=1)
            b = shape_extrude("fleur_b", ol, depth=tf, loc=(-tf / 2, 0, zp0), rot=(90, 0, 90), bevel=tf * 0.3, bevel_res=1)
            fo["ball"] += [a, b]
            rbnd = 0.125 * 0.95 * hfl + 0.004
            fo["body"].append(lathe("fleur_band", [(rbnd - 0.012, zp0 + 0.25 * hfl), (rbnd, zp0 + 0.262 * hfl), (rbnd + 0.004, zp0 + 0.295 * hfl),
                                                   (rbnd, zp0 + 0.328 * hfl), (rbnd - 0.012, zp0 + 0.34 * hfl)], segments=16))
        elif p.motif == "pineapple":
            Rp = min(ball_r * 1.12, hu * 0.25)
            hp_ = min(hu * 0.6, Rp * 2.9)
            zc0 = zs + 0.012 + hp_ / 2
            ns, ms = 8, 4

            def pine(uu, vv):
                ph = pi * (0.07 + 0.88 * vv)
                env = sin(ph) ** 0.72
                a1 = (ns * uu + ms * vv) % 1.0 - 0.5
                a2 = (ns * uu - ms * vv) % 1.0 - 0.5
                b = max(0.0, 1.0 - 2.0 * max(abs(a1), abs(a2)))          # one little pyramid per diamond scale
                rad = Rp * env * (0.87 + 0.24 * b ** 0.75)
                an = 2 * pi * uu
                return (rad * cos(an), rad * sin(an), zc0 - hp_ / 2 * cos(ph) * (1.0 + 0.0 * b))
            pb = surface("pine_body", pine, 48, 24, closed_u=True)
            recalc_normals(pb)
            smooth(pb, 70)
            fo["ball"].append(pb)
            ztop = zc0 + hp_ / 2 * cos(pi * 0.05)
            # calyx of leaves under the fruit and the tuft on top
            lf = _leaf("calyx_leaf", Rp * 1.0, Rp * 0.7, 0.42, 0.011 if not slim else 0.005)
            lm = [trs((rn * 0.9 * sin(radians(k * 45)), -rn * 0.9 * cos(radians(k * 45)), zs - 0.004), (38 + r.jitter(0, 5), 0, k * 45), r.vary(1.0, 0.08))
                  for k in range(8)]
            fo["body"].append(instances_to_mesh("calyx", lf, lm, keep_base=False))
            lc = max(H - ztop, 0.03)
            tl = _leaf("tuft_leaf", lc * 0.98, lc * 0.5, 0.3, 0.011 if not slim else 0.005)
            tm = [trs((0.012 * sin(radians(k * 60)), -0.012 * cos(radians(k * 60)), ztop - 0.012), (46 + r.jitter(0, 6), 0, k * 60), 0.62) for k in range(6)]
            tm += [trs((0.006 * sin(radians(k * 72 + 30)), -0.006 * cos(radians(k * 72 + 30)), ztop - 0.01), (20 + r.jitter(0, 5), 0, k * 72 + 30), 0.86) for k in range(5)]
            tm += [trs((0, 0.004, ztop - 0.01), (3, 0, r.u(0, 360)), 1.0), trs((0, -0.004, ztop - 0.01), (4, 0, r.u(0, 360)), 0.95)]
            fo["body"].append(instances_to_mesh("tuft", tl, tm, keep_base=False))
            fo["body"].append(lathe("pine_collar", [(rn, zs - 0.004), (Rp * 0.42, zs + 0.006), (Rp * 0.46, zs + 0.02), (Rp * 0.3, zs + 0.035)], segments=14))
        else:                                # star
            rk = rn * 1.75
            Rs = min(0.15 * kr, hu * 0.3)
            zsc = H - Rs
            fo["ball"].append(sphere("finial_knop", rk, loc=(0, 0, zs + rk * 0.75), segments=14, rings=8, scale=(1, 1, 0.9)))
            za = zs + rk * 1.55
            zt = zsc - Rs * 0.36
            ls = zt - za
            sp = [(rn * 0.8, za - 0.012), (rn * 1.25, za + 0.005), (rn * 1.35, za + 0.014), (rn * 1.05, za + 0.023)]
            sp += _sm([(rn * 0.82, za + 0.027), (rn * 0.6, za + 0.4 * ls), (rn * 0.42, za + 0.8 * ls), (max(rn * 0.36, 0.007), zt)], 6)
            sp += [(max(rn * 0.5, 0.01), zt + 0.004), (max(rn * 0.5, 0.01), zt + 0.012), (0.0, zt + 0.016)]
            fo["body"].append(lathe("finial_stem", sp, segments=12))
            fo["star"].append(cylinder("star_rod", 0.0055, Rs * 0.5, loc=(0, 0, zt + Rs * 0.2), verts=6))
            fo["star"].append(_star("star_a", Rs, Rs * 0.17, 0.0025, (0, 0, zsc), 0))
            fo["star"].append(_star("star_b", Rs, Rs * 0.17, 0.0025, (0, 0, zsc), 90))

        if slim:                             # two forged scrolls brace the stem along the ridge
            r1 = min(0.105 * kr, u / 2 - 0.035, hs * 0.5)
            barp = rounded_rect(0.009, 0.024, 0.002, 1)
            cxs = rn + 0.006 + r1
            czs = z_rail + r1 * 0.86 + 0.004
            for sx in (-1, 1):
                pts = spiral_path(0.013, r1, 1.5, n=26, a0_deg=0.0, plane="XZ", center=(cxs, 0, czs), ccw=False)
                pts = [(sx * q[0], q[1], q[2]) for q in pts]
                fo["body"].append(sweep(f"scroll{sx}", pts, barp, up=(0, 1, 0), scale_fn=lambda t: 0.55 + 0.45 * min(1.0, t * 2.5)))
                fo["body"].append(lathe(f"scroll_tie{sx}", [(0.006, 0), (0.012, 0.004), (0.012, 0.012), (0.006, 0.016)], segments=8,
                                        loc=(sx * (rn + 0.012), 0, czs - 0.008), rot=(0, sx * 90, 0)))
        if M == "terracotta":
            displace_noise(lower, 0.0016, 0.05, seed=seed + 11 + idx)
        elif M == "lead":
            for o in fo["body"] + fo["ball"]:
                if o.name.startswith(("finial_lower", "finial_ball", "finial_knop")):
                    displace_noise(o, 0.0022, 0.045, seed=seed + 13 + idx)
        lean = (r.jitter(0, 0.5 + 1.6 * age), r.jitter(0, 0.5 + 1.6 * age), r.u(0, 360) if p.motif in ("ball_spike", "pineapple") else r.jitter(0, 6))
        for k in fo:
            for o in fo[k]:
                apply_transform(o)
                place(o, loc=(xf, 0, 0), rot=lean)
        body_objs.extend(fo["body"])
        ball_objs.extend(fo["ball"])
        star_objs.extend(fo["star"])
        iron_objs.extend(fo["iron"])
        socket(f"tip{idx}" if idx else "tip", "attach", (xf, 0, H), role="prop_top")

    if p.mode != "cresting":
        for k, i in enumerate(fin_units):
            finial(x0 + (i + 0.5) * u, rng.fork(f"finial{k}"), k)

    # ---- moss cushions on old clay ridges (in the lee of the crests and on the bedding)
    if M == "terracotta" and age > 0.3:
        rm = rng.fork("moss")
        tuft = sphere("moss_tuft", 0.5, segments=7, rings=4)
        displace_noise(tuft, 0.12, 0.5, seed=seed + 5)
        mm = []
        for k in range(int(L * 7 * (age - 0.2))):
            xk = rm.u(x0 + 0.03, x0 + L - 0.03)
            s = rm.u(0.018, 0.04)
            if rm.chance(0.55):
                mm.append(trs((xk, R + rm.u(0.0, 0.012), 0.012), (0, 0, rm.u(0, 360)), (s * rm.u(1.2, 2.2), s, s * 0.6)))
            else:
                mm.append(trs((xk, rm.pick([-1, 1, 1]) * (th / 2 + 0.004), zc - 0.003), (0, 0, rm.u(-20, 20)), (s * rm.u(1.2, 2.4), s * 0.7, s * 0.55)))
        if mm:
            moss_objs.append(instances_to_mesh("moss", tuft, mm, keep_base=False))

    # ---- materials
    for o in body_objs:
        assign(o, body)
    for o in saddle_objs:
        assign(o, saddle_m)
    for o in bed_objs:
        assign(o, mat("mortar", "#b9b2a3"))
    for o in iron_objs:
        assign(o, mat("iron_rusty") if slim else iron_w)
    for o in ball_objs:
        assign(o, ball_m)
    for o in star_objs:
        assign(o, star_m)
    for o in moss_objs:
        assign(o, mat("moss"))

    # ---- art-directed weathering
    zr = zc + max(ch if crest_units else 0.1, 0.1)
    n_a = noise_mask(0.11, seed=seed + 3, lo=0.78 - 0.5 * age, hi=1.02 - 0.42 * age)
    n_b = noise_mask(0.05, seed=seed + 4, lo=0.3, hi=0.75)
    n_c = noise_mask(0.2, seed=seed + 6, lo=0.25, hi=0.8)

    def damp(co, nr):                        # moss / lichen: tops, the north (+Y) side, low down
        up = np.clip(0.4 + 0.4 * nr[:, 2] + 0.4 * nr[:, 1], 0, 1)
        low = np.clip(1.3 - co[:, 2] / (zr + 0.2), 0.3, 1)
        return np.clip(n_a(co, nr) * up * low * 1.3, 0, 1) * min(1.0, age * 1.5)

    def streak(co, nr):                      # rain-washed dirt: under overhangs and towards the bed
        low = np.clip(1.0 - co[:, 2] / (zr + 0.05), 0, 1)
        under = np.clip(-nr[:, 2], 0, 1)
        return np.clip((0.25 + 0.5 * low + 0.6 * under) * (0.35 + 0.65 * n_c(co, nr)) * (0.4 + 0.8 * age), 0, 1)

    def rust(co, nr):                        # at the fixings and water traps
        low = np.clip(1.0 - (co[:, 2] - zc) / 0.09, 0, 1)
        return np.clip((0.75 * low + 0.3) * n_b(co, nr) * (0.35 + 0.9 * age), 0, 1)

    def knocks(co, nr):
        return np.clip(n_b(co, nr) * np.clip(nr[:, 2], 0, 1) * (0.2 + 0.5 * age), 0, 1)

    if M == "terracotta":
        for o in body_objs + saddle_objs + ball_objs:
            paint(o, grime=streak, special=damp, wear=knocks)
        for o in bed_objs:
            paint(o, grime=mask_mul(n_c, 0.5 + 0.4 * age), special=mask_mul(n_a, min(1.0, age * 1.6)))
    elif M == "lead":
        for o in body_objs + saddle_objs:
            paint(o, grime=streak, wear=knocks)
    else:
        for o in body_objs:
            paint(o, grime=streak, special=rust, wear=mask_mul(n_b, 0.25 * age))
        for o in saddle_objs:
            paint(o, grime=streak, special=mask_mul(rust, 0.6))
    for o in ball_objs + star_objs:
        if p.gilt and M != "terracotta" or o in star_objs:
            paint(o, grime=mask_mul(n_c, 0.25 + 0.35 * age))
