"""Retractable canvas shop awning (lateral folding-arm type, c. 1900): a painted timber blind box with a
painted sheet-zinc capping and shaped end cheeks is coach-bolted to the wall; inside it a roller carries the wound
canvas, driven by a worm gear with a brass winding eye. Below the box a square iron bar on wall brackets
carries two folding arms (shoulder, elbow knuckle with spring and cable, wrist) that push out the front
bar. The striped canvas runs from the roller over the front bar with a slight belly and drooping hems
and ends in a scalloped, bound valance with appliqued lettering.
Origin = centre of the top fixing rail on the wall plane (y = 0); the awning projects to -Y by
`projection` and is `width` wide overall.
"""
from tk import *
import numpy as np
from bisect import bisect_right

META = {"name": "Striped shop awning", "class": "component", "mount": "wall", "mount_height": 3.0,
        "tri_budget": 14000,       # appliqued lettering alone is 2-3k triangles
        "focus": [0.0, -1.35, -0.74]}
PARAMS = {
    "width": {"default": 3.0, "min": 1.6, "max": 5.0, "desc": "overall width of the blind box (m)"},
    "projection": {"default": 1.5, "min": 0.8, "max": 2.5, "desc": "horizontal reach from the wall to the valance (m)"},
    "pitch": {"default": 22.0, "min": 5.0, "max": 35.0, "desc": "fall of the canvas below horizontal (deg)"},
    "sag": {"default": 0.55, "min": 0.0, "max": 1.0, "desc": "slackness of the canvas (0 = drum tight, 1 = tired and baggy)"},
    "stripe": {"default": "block", "choices": ["block", "regency", "plain"], "desc": "stripe pattern: equal blocks, wide + narrow, or plain canvas"},
    "stripe_width": {"default": 0.15, "min": 0.08, "max": 0.3, "desc": "width of a (wide) stripe = width of a scallop (m)"},
    "stripe_a": {"default": "#8a2424", "desc": "main canvas colour"},
    "stripe_b": {"default": "#e6dcc2", "desc": "second stripe colour"},
    "valance": {"default": "solid", "choices": ["solid", "striped"], "desc": "valance in the main colour or striped like the canvas"},
    "valance_height": {"default": 0.3, "min": 0.2, "max": 0.42, "desc": "overall drop of the valance (m)"},
    "scallop": {"default": "round", "choices": ["round", "wave", "point", "tab", "straight"], "desc": "shape of the valance edge"},
    "scallop_depth": {"default": 0.065, "min": 0.02, "max": 0.1, "desc": "depth of the scallops (m)"},
    "text": {"default": "BOULANGERIE", "desc": "lettering on the valance ('' = none)"},
    "font": {"default": "serif", "choices": ["serif", "roman", "sans", "sans_narrow", "script"], "desc": "lettering font"},
    "trim": {"default": "#e2c57c", "desc": "colour of the valance binding and the lettering"},
    "paint": {"default": "#2a4a3c", "desc": "paint colour of the blind box and ironwork"},
    "capping": {"default": "#7f8a86", "desc": "paint colour of the sheet-zinc weather capping on the box"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "sun-fade, dirt and rust (0 = new, 1 = tired)"},
}
VARIANTS = {
    "regency_green": {"width": 2.2, "projection": 1.15, "pitch": 30.0, "stripe": "regency", "stripe_width": 0.2,
                      "stripe_a": "#2f5a43", "stripe_b": "#e9e2cc", "valance": "striped", "scallop": "wave",
                      "text": "", "paint": "#2a1c14", "capping": "#6c9685", "trim": "#e9e2cc", "sag": 0.35, "age": 0.3, "seed": 4},
    "cafe_blue_wide": {"width": 4.6, "projection": 2.1, "pitch": 15.0, "stripe_width": 0.26, "stripe_a": "#243f66",
                       "stripe_b": "#ece6d6", "scallop": "point", "scallop_depth": 0.09, "valance_height": 0.34,
                       "text": "CAFE DE LA PLACE", "font": "roman", "trim": "#f0e7cf", "paint": "#15181c", "capping": "#5d6670",
                       "sag": 0.8, "age": 0.5, "seed": 11},
    "tabac_ochre": {"width": 1.8, "projection": 0.9, "pitch": 33.0, "stripe": "plain", "stripe_a": "#b9862f",
                    "valance": "striped", "scallop": "tab", "scallop_depth": 0.05, "valance_height": 0.3,
                    "text": "TABAC", "font": "sans", "trim": "#5a1f17", "paint": "#4a1a16", "capping": "#8c7a60", "sag": 1.0,
                    "age": 0.8, "seed": 23},
    "narrow_stripe_straight": {"width": 3.6, "projection": 1.3, "pitch": 26.0, "stripe_width": 0.09,
                               "stripe_a": "#6b3d7a", "stripe_b": "#d9c9a0", "valance": "striped",
                               "scallop": "straight", "valance_height": 0.22, "text": "", "paint": "#20232a",
                               "trim": "#d9c9a0", "sag": 0.2, "age": 0.15, "seed": 31},
}

TH = 0.003          # canvas thickness


def _stripes(Wc, kind, sw):
    """-> (edges [n+1], colour index per stripe [n], scallop boundaries)."""
    if kind == "regency":
        nb = 0.3
        k = max(1, int(round((Wc - sw) / (sw * (1 + nb)))))
        widths, cols = [sw], [0]
        for _ in range(k):
            widths += [sw * nb, sw]
            cols += [1, 0]
    else:
        n = max(3, int(round(Wc / sw)))
        if n % 2 == 0:
            n += 1
        widths = [1.0] * n
        cols = [0] * n if kind == "plain" else [i % 2 for i in range(n)]
    s = Wc / sum(widths)
    edges = [-Wc / 2]
    for w in widths:
        edges.append(edges[-1] + w * s)
    edges[-1] = Wc / 2
    if kind == "regency":
        sc = [edges[0]] + [(edges[i] + edges[i + 1]) / 2 for i in range(len(cols)) if cols[i] == 1] + [edges[-1]]
    else:
        sc = list(edges)
    return edges, cols, sc


def _shape(style, t):
    if style == "round":
        return sqrt(max(0.0, 1.0 - (2 * t - 1) ** 2))
    if style == "wave":
        return 0.5 - 0.5 * cos(2 * pi * t)
    if style == "point":
        return 1.0 - abs(2 * t - 1)
    if style == "tab":
        return clamp((0.5 - abs(t - 0.5) - 0.1) / 0.06, 0.0, 1.0)
    return 1.0


_TS = {
    "round": [(1 - cos(pi * k / 5)) / 2 for k in range(5)],
    "wave": [k / 5 for k in range(5)],
    "point": [0.0, 0.25, 0.5, 0.75],
    "tab": [0.0, 0.1, 0.16, 0.5, 0.84, 0.9],
    "straight": [0.0, 0.5],
}


def _merge(xs, tol=1e-4):
    out = []
    for x in sorted(xs):
        if not out or x - out[-1] > tol:
            out.append(x)
    return out


def _set_face_mats(obj, mats, fn_index):
    """Assign materials per base-mesh face: fn_index(face_number) -> index into mats."""
    assign(obj, mats[0])
    for k in range(1, len(mats)):
        counter = iter(range(10 ** 9))
        assign_faces(obj, mats[k], lambda c, n, k=k, counter=counter: fn_index(next(counter)) == k)


def build(p, rng):
    W, P = p.width, p.projection
    th = radians(p.pitch)
    cth, sth = cos(th), sin(th)
    age = p.age
    cA, cB, cT = mat("canvas", p.stripe_a), mat("canvas", p.stripe_b), mat("canvas", p.trim)
    iron = mat("iron_painted", p.paint)
    wood = mat("wood_painted", p.paint)
    zinc = mat("wood_painted", p.capping)        # painted sheet zinc (bare metal reads black from above at night)
    brass = mat("brass")
    forged = mat("iron_wrought")
    irons, woods, zincs = [], [], []

    # ------------------------------------------------------------------ key section points (y, z)
    CH = 0.03                                   # end cheek thickness
    Wc = W - 0.2                                # canvas width
    yc, zc, r_roll = -0.125, -0.147, 0.042      # roller axis, radius of the wound canvas
    Ty, Tz = yc - r_roll * sth, zc + r_roll * cth          # canvas leaves the roller here
    hv = p.valance_height
    sd = 0.0 if p.scallop == "straight" else min(p.scallop_depth, hv - 0.15)
    hb = hv - sd                                # drop to the scallop cusps
    fold_amp = 0.018 + 0.024 * p.sag
    r_c = 0.014                                 # corner radius of the front bar
    rc = r_c + TH / 2 + 0.0006
    y_v = -P + fold_amp + 0.005                 # mid-plane of the valance
    cby = y_v + rc                              # centre of the bar's top front corner
    T2y = cby - rc * sth
    Ls = (Ty - T2y) / cth                       # free span of the canvas
    T2z = Tz - Ls * sth
    cbz = T2z - rc * cth
    ztop = cbz                                  # valance starts here
    zbot = ztop - hv
    BY, BZ = 0.05, 0.07                         # front bar section
    y_ff = cby - r_c                            # bar front face
    bar_y, bar_z = y_ff + BY / 2, cbz + r_c - BZ / 2

    # ------------------------------------------------------------------ canvas
    edges, cols, scb = _stripes(Wc, p.stripe, p.stripe_width)
    r = rng.fork("cloth")
    lam1, lam2 = r.vary(0.52, 0.18), r.vary(0.29, 0.18)
    ph1, ph2, ph3, ph4 = (r.u(0, 2 * pi) for _ in range(4))
    sagm = p.sag * 0.024 * Ls + 0.004

    def stripe_col(x):
        i = min(max(bisect_right(edges, x) - 1, 0), len(cols) - 1)
        return cols[i]

    def fold(x, z):
        t = clamp((ztop - BZ - z) / max(hv - BZ, 1e-3), 0.0, 1.0)
        w = 0.6 * sin(2 * pi * x / lam1 + ph1) + 0.4 * sin(2 * pi * x / lam2 + ph2)
        return -fold_amp * t * (0.5 + 0.5 * w)

    def belly(x, s):
        xn = x / (Wc / 2)
        b = sin(pi * s)
        dz = -sagm * b * (0.8 + 0.2 * cos(pi * xn * 0.5))
        dz -= (0.012 + 0.022 * p.sag) * b * abs(xn) ** 6                        # hems droop
        dz += (0.005 + 0.01 * p.sag) * b * (sin(2 * pi * (x / 0.47 + 0.35 * s) + ph3) * 0.6
                                               + sin(2 * pi * (x / 0.83 - 0.5 * s) + ph4) * 0.4)   # tension ripples
        return dz

    rows = []                                   # each: fn(x) -> (y, z)
    nroll = 7
    for k in range(nroll):
        a = th - radians(372.0) * (1 - k / nroll)
        rr = r_roll - 0.007 * (1 - k / nroll)
        rows.append(lambda x, a=a, rr=rr: (yc - rr * sin(a), zc + rr * cos(a)))
    nspan = 7
    for k in range(nspan + 1):
        s = k / nspan
        rows.append(lambda x, s=s: (Ty + (T2y - Ty) * s, Tz + (T2z - Tz) * s + belly(x, s)))
    for k in range(1, 4):
        a = th + (pi / 2 - th) * k / 3
        rows.append(lambda x, a=a: (cby - rc * sin(a), cbz + rc * cos(a)))
    j_val = len(rows) - 1                       # the valance is sewn on here (front face of the bar)
    z2 = ztop - hb + 0.03

    xs = list(edges)
    if len(cols) <= 27:
        for i in range(len(cols)):
            if edges[i + 1] - edges[i] > 0.1:
                xs.append((edges[i] + edges[i + 1]) / 2)
    xs = _merge(xs)
    nu, nv = len(xs) - 1, len(rows) - 1

    def cloth_fn(u, v):
        x = xs[int(round(u * nu))]
        y, z = rows[int(round(v * nv))](x)
        return (x, y, z)

    cloth = surface("canvas", cloth_fn, nu, nv, thickness=TH)
    solid_val = p.valance == "solid"

    def cloth_mat(f):
        i, j = divmod(f, nv)
        return stripe_col((xs[i] + xs[i + 1]) / 2)

    _set_face_mats(cloth, [cA, cB], cloth_mat)
    # doubled side hems (a thicker folded band enclosing the raw edge)
    hem_rows = rows[nroll:j_val + 1]
    hems = []
    for sx in (-1, 1):
        hx = [sx * (Wc / 2 - 0.03), sx * (Wc / 2 + 0.0015)]
        hm = surface("side_hem", lambda u, v, hx=hx: (hx[int(round(u))],) + tuple(hem_rows[int(round(v * (len(hem_rows) - 1)))](hx[int(round(u))])),
                     1, len(hem_rows) - 1, thickness=TH + 0.0035)
        assign(hm, [cA, cB][cols[0 if sx < 0 else -1]])
        hems.append(hm)

    # ---- valance: its own piece sewn on along the bar, fine columns for the scalloped outline, bound hem
    xf = list(edges)
    ts = _TS[p.scallop]
    for a, b in zip(scb[:-1], scb[1:]):
        xf += [a + (b - a) * t for t in ts]
    xf = _merge(xf)

    def edge_z(x):
        k = min(max(bisect_right(scb, x) - 1, 0), len(scb) - 2)
        t = (x - scb[k]) / (scb[k + 1] - scb[k])
        return ztop - hb - sd * _shape(p.scallop, clamp(t, 0.0, 1.0))

    bot = [(x, edge_z(x)) for x in xf]
    tw = 0.017                                  # binding width
    trim_row = []
    for i, (x, z) in enumerate(bot):
        x0, z0 = bot[max(i - 1, 0)]
        x1, z1 = bot[min(i + 1, len(bot) - 1)]
        tx, tz = x1 - x0, z1 - z0
        l = hypot(tx, tz) or 1.0
        nx, nz = -tz / l, tx / l
        nz = max(nz, 0.45)
        l2 = hypot(nx, nz)
        xt = clamp(x + tw * nx / l2, xf[0], xf[-1])
        if trim_row:
            xt = max(xt, trim_row[-1][0] + 0.002)          # never fold back on itself
        trim_row.append((xt, clamp(z + tw * nz / l2, z + 0.004, z2 - 0.004)))
    trim_row[0] = (xf[0], trim_row[0][1])
    for i in range(len(trim_row) - 1, -1, -1):             # keep inside the cloth at the far end
        lim = xf[-1] - 0.002 * (len(trim_row) - 1 - i)
        if trim_row[i][0] > lim:
            trim_row[i] = (lim, trim_row[i][1])
        else:
            break
    nuf = len(xf) - 1

    def edge_fn(u, v):
        i = int(round(u * nuf))
        j = int(round(v * 4))
        if j <= 2:
            x, z = xf[i], (ztop, ztop - BZ, z2)[j]
        elif j == 3:
            x, z = trim_row[i]
        else:
            x, z = bot[i]
        return (x, y_v + fold(x, z), z)

    edge = surface("valance", edge_fn, nuf, 4)
    solidify(edge, TH, offset=0.0, even=False)

    def edge_mat(f):
        i, j = divmod(f, 4)
        if j == 3:
            return 2
        return 0 if solid_val else stripe_col((xf[i] + xf[i + 1]) / 2)

    _set_face_mats(edge, [cA, cB, cT], edge_mat)

    # ---- lettering appliqued on the valance (follows its folds)
    label = str(p.text).strip()
    if label:
        band = hb - tw - 0.012
        zmid = ztop - 0.008 - band / 2
        size = min(0.64 * band, 0.18)
        t = text("lettering", label, size=size, depth=0.003, loc=(0, y_v - TH / 2 + 0.0008, zmid), font=p.font, spacing=1.06)
        apply_transform(t)
        lo, hi = world_bbox([t])
        k = min(1.0, 0.84 * Wc / max(hi.x - lo.x, 1e-3), 0.8 * band / max(hi.z - lo.z, 1e-3))
        cz = (lo.z + hi.z) / 2

        def fit(co):
            x, z = co.x * k, zmid + (co.z - cz) * k
            return Vector((x, co.y + fold(x, z), z))

        transform_verts(t, fit)
        assign(t, cT)
        paint(t, special=0.5 * age, grime=0.25 * age)

    # ---- weathering of the cloth: sun-fade on top, dirt in the belly, rain streaks, grubby hem
    nm = noise_mask(0.55, seed=p.seed + 3, lo=0.3, hi=0.75)
    st = noise_mask(0.07, seed=p.seed + 5, lo=0.4, hi=0.8)
    sq = np.array([1.0, 0.06, 0.06])

    def c_grime(co, n):
        s = np.clip((Ty - co[:, 1]) / max(Ty - T2y, 1e-3), 0.0, 1.0)
        on_span = (co[:, 2] > ztop + 0.01) & (co[:, 1] < Ty)
        pool = np.sin(np.pi * s) ** 2 * on_span * (n[:, 2] > 0.2)
        streak = st(co * sq, n) * on_span * (0.25 + 0.75 * s)
        hem = np.clip(1.0 - (co[:, 2] - zbot) / 0.2, 0.0, 1.0) ** 1.5
        sides = np.clip((np.abs(co[:, 0]) - (Wc / 2 - 0.12)) / 0.12, 0.0, 1.0) * on_span
        g = (0.15 + 0.85 * age) * (0.35 * pool * nm(co, n) + 0.3 * streak + 0.3 * hem + 0.3 * sides)
        return np.clip(g, 0.0, 1.0)

    def c_fade(co, n):
        up = np.clip(n[:, 2], 0.0, 1.0)
        front = np.clip(-n[:, 1], 0.0, 1.0)
        return np.clip(age * (0.7 * up + 0.35 * front) * (0.5 + 0.5 * nm(co, n)), 0.0, 1.0)

    paint([cloth, edge] + hems, grime=c_grime, special=c_fade)

    # ------------------------------------------------------------------ blind box (painted timber)
    back = box("back_board", (W, 0.025, 0.275), loc=(0, -0.0125, -0.0925), bevel=0.003, segments=1)
    uv_grain(back, "X")
    woods.append(back)
    slope = 7.0
    tb_y, tb_z = -0.131, -0.027
    # (the sloped top board is modelled as one piece with its sheet-metal covering, see the capping below)

    def top_under(y):                           # underside of the top board at depth y
        return tb_z - 0.011 / cos(radians(slope)) + (y - tb_y) * tan(radians(slope))

    yf = -0.243
    prof = [(-0.025, -0.23), (-0.025, top_under(-0.025)), (yf, top_under(yf)), (yf, -0.125)]
    prof += [(y, z) for (y, z, _) in bezier_path((yf, -0.125, 0), (yf, -0.2, 0), (-0.2, -0.23, 0), (-0.14, -0.23, 0), n=7)][1:]
    area = sum(prof[i][0] * prof[(i + 1) % len(prof)][1] - prof[(i + 1) % len(prof)][0] * prof[i][1] for i in range(len(prof)))
    if area < 0:
        prof.reverse()
    for sx in (-1, 1):
        x0 = -W / 2 if sx < 0 else W / 2 - CH
        ck = extrude_profile("cheek", prof, CH, loc=(x0, 0, 0), rot=(90, 0, 90), bevel=0.004, segments=1)
        uv_grain(ck, "Z")
        woods.append(ck)
    fas = box("fascia", (W - 2 * CH + 0.004, 0.02, 0.05), loc=(0, yf + 0.012, top_under(yf + 0.012) - 0.024), bevel=0.004, segments=1)
    uv_grain(fas, "X")
    woods.append(fas)
    bead = cylinder("fascia_bead", 0.009, W - 2 * CH + 0.004, loc=(0, yf + 0.004, top_under(yf) - 0.047), rot=(0, 90, 0), verts=8)
    uv_grain(bead, "Z")
    woods.append(bead)
    # sheet-metal capping (batten-roll joints, upstand against the wall, rolled drip edge)
    cs = cos(radians(slope))
    sn = sin(radians(slope))
    ncap = max(8, int(round(W / 0.17)))        # tessellated so the baked dirt / AO stays local to the rolls and edges
    zincs.append(surface("zinc_cap", lambda u, v: (W * (u - 0.5), tb_y - 0.008 + 0.262 * (v - 0.5) * cs,
                                                   tb_z + 0.00275 / cs + 0.262 * (v - 0.5) * sn), ncap, 4, thickness=0.0275))
    zincs.append(box("zinc_upstand", (W, 0.004, 0.03), loc=(0, -0.027, 0.006), bevel=0.001, segments=1))
    dy = tb_y - 0.008 - 0.131 * cs
    zincs.append(cylinder("zinc_drip", 0.007, W, loc=(0, dy, tb_z + 0.0135 / cs - 0.131 * sin(radians(slope)) - 0.004), rot=(0, 90, 0), verts=8))
    nseam = max(2, int(round(W / 0.8)))
    for i in range(1, nseam):
        x = -W / 2 + W * i / nseam + rng.jitter(0, 0.02)
        zincs.append(box("zinc_roll", (0.016, 0.266, 0.013), loc=(x, tb_y - 0.008, tb_z + 0.0135 / cs + 0.007), rot=(slope, 0, 0), bevel=0.004, segments=1))
    # coach bolts through the fixing rail into the wall
    nb = max(3, int(round(W / 0.55)) + 1)
    hx = cylinder("coach_bolt", 0.0125, 0.008, loc=(0, 0, 0.004), verts=4, radius2=0.009)
    apply_transform(hx)
    flat(hx)
    bm = [trs((-W / 2 + 0.12 + (W - 0.24) * i / (nb - 1) + rng.jitter(0, 0.01), -0.025, 0.031), (90, 0, rng.jitter(0, 25))) for i in range(nb)]
    bolts = [instances_to_mesh("coach_bolts", hx, bm, keep_base=False)]

    # ------------------------------------------------------------------ roller and winding gear
    xin = W / 2 - CH                            # inner face of the cheeks
    irons.append(cylinder("roller_tube", 0.031, 2 * xin, loc=(0, yc, zc), rot=(0, 90, 0), verts=16))
    for sx in (-1, 1):
        irons.append(cylinder("roller_flange", 0.05, 0.008, loc=(sx * (Wc / 2 + 0.012), yc, zc), rot=(0, 90, 0), verts=12))
    irons.append(cylinder("bearing", 0.04, 0.03, loc=(-xin + 0.015, yc, zc), rot=(0, 90, 0), verts=14, bevel=0.004, segments=1))
    gx = xin - 0.027
    irons.append(box("gear_case", (0.054, 0.085, 0.092), loc=(gx, yc, zc), bevel=0.008, segments=2))
    irons.append(cylinder("gear_boss", 0.022, 0.03, loc=(gx, yc - 0.045, zc - 0.03), rot=(155, 0, 0), verts=12, bevel=0.003, segments=1))
    da = radians(25.0)
    dv = Vector((0, -sin(da), -cos(da)))
    g0 = Vector((gx, yc - 0.045, zc - 0.03))
    sh_len = 0.17
    bolts.append(cylinder("crank_shaft", 0.0065, sh_len, loc=tuple(g0 + dv * (sh_len / 2)), rot=(155, 0, 0), verts=8))
    eye_c = g0 + dv * (sh_len + 0.017)
    crank_eye = torus("crank_eye", 0.019, 0.005, loc=tuple(eye_c), rot=(0, 90, 0), major_seg=12, minor_seg=5)
    assign(crank_eye, brass)
    paint(crank_eye, wear=0.6, special=0.3 * age)

    # ------------------------------------------------------------------ square wall bar on brackets
    ybar, zbar, sqb = -0.066, -0.318, 0.04
    irons.append(box("wall_bar", (W - 0.1, sqb, sqb), loc=(0, ybar, zbar), bevel=0.003, segments=1))
    nbr = max(2, int(round(W / 1.5)) + 1)
    bx = [(-1 + 2 * i / (nbr - 1)) * (W / 2 - 0.11) for i in range(nbr)]
    bolt_m = []
    hexb = cylinder("hex", 0.0085, 0.007, loc=(0, 0, 0.0035), verts=6)
    apply_transform(hexb)
    flat(hexb)
    for x in bx:
        irons.append(box("bracket_plate", (0.1, 0.01, 0.19), loc=(x, -0.005, zbar - 0.005), bevel=0.003, segments=1))
        irons.append(box("bracket_web", (0.016, ybar + sqb / 2 + 0.012, 0.11), loc=(x, (ybar + sqb / 2 - 0.008) / 2 - 0.004, zbar), bevel=0.002, segments=1))
        irons.append(box("bracket_clamp", (0.056, sqb + 0.016, sqb + 0.016), loc=(x, ybar, zbar), bevel=0.004, segments=1))
        for sxx in (-1, 1):
            for szz in (-1, 1):
                bolt_m.append(trs((x + sxx * 0.033, -0.01, zbar - 0.005 + szz * 0.074), (90, 0, rng.u(0, 60))))
        bolt_m.append(trs((x, ybar - sqb / 2 - 0.008, zbar), (90, 0, rng.u(0, 60))))

    # ------------------------------------------------------------------ front bar
    fb = sweep("front_bar", [(-W / 2 + 0.012, bar_y, bar_z), (W / 2 - 0.012, bar_y, bar_z)], rounded_rect(BY, BZ, r_c, 3), smooth_path=False)
    irons.append(fb)
    for sx in (-1, 1):
        irons.append(rounded_box("bar_cap", (0.012, BY + 0.012, BZ + 0.012), radius=0.004, loc=(sx * (W / 2 - 0.006), bar_y, bar_z), seg=2))
    # pull eye for the boat hook
    irons.append(box("pull_plate", (0.07, 0.03, 0.005), loc=(0, bar_y, bar_z - BZ / 2 - 0.0025), bevel=0.001, segments=1))
    pull = torus("pull_eye", 0.02, 0.0055, loc=(0, bar_y, bar_z - BZ / 2 - 0.022), rot=(0, 90, 0), major_seg=12, minor_seg=5)
    assign(pull, brass)
    paint(pull, wear=0.7, special=0.25 * age)
    for sx in (-1, 1):
        bolt_m.append(trs((sx * 0.024, bar_y, bar_z - BZ / 2 - 0.005), (180, 0, rng.u(0, 60))))

    # ------------------------------------------------------------------ folding arms
    xs_arm = max(W / 2 - 0.26, 0.35)
    S_yz = (ybar - sqb / 2 - 0.055, zbar)
    Wp_yz = (y_ff + BY + 0.05, bar_z - 0.028)
    D = hypot(Wp_yz[0] - S_yz[0], Wp_yz[1] - S_yz[1])
    beta = atan2(S_yz[1] - Wp_yz[1], S_yz[0] - Wp_yz[0])       # fall of the arm plane
    bdeg = degrees(beta)
    e2 = Vector((0, -cos(beta), -sin(beta)))
    n3 = Vector((0, -sin(beta), cos(beta)))
    half = D / 2
    off = min(half * sqrt(1.055 ** 2 - 1.0), xs_arm - 0.09)

    def knuckle(name, c, s=1.0):
        pr = [(0.0, -0.033), (0.026, -0.033), (0.03, -0.029), (0.03, -0.011), (0.026, -0.008),
              (0.026, 0.008), (0.03, 0.011), (0.03, 0.029), (0.026, 0.033), (0.0, 0.033)]
        return lathe(name, [(a * s, b * s) for a, b in pr], segments=10, loc=tuple(c), rot=(bdeg, 0, 0), smooth_angle=35)

    up_prof = rounded_rect(0.018, 0.046, 0.004, 2)
    fo_prof = rounded_rect(0.016, 0.04, 0.004, 2)
    cable_objs = []
    for sx in (-1, 1):
        ra = rng.fork(f"arm{sx}")
        xs0 = sx * xs_arm
        xw0 = sx * (xs_arm - 0.03)
        S = Vector((xs0, S_yz[0], S_yz[1]))
        Wp = Vector((xw0, Wp_yz[0], Wp_yz[1]))
        mid = (S + Wp) / 2
        E = mid + Vector((-sx, 0, 0)) * (off * ra.vary(1.0, 0.06))
        # shoulder: clamp block on the square bar + lug
        irons.append(box("shoulder_clamp", (0.085, sqb + 0.022, sqb + 0.026), loc=(xs0, ybar, zbar), bevel=0.005, segments=1))
        irons.append(box("shoulder_lug", (0.052, 0.06, 0.036), loc=(xs0, ybar - sqb / 2 - 0.028, zbar), rot=(bdeg, 0, 0), bevel=0.004, segments=1))
        bolt_m.append(trs((xs0 - 0.026, ybar - sqb / 2 - 0.011, zbar + 0.02), (90, 0, ra.u(0, 60))))
        bolt_m.append(trs((xs0 + 0.026, ybar - sqb / 2 - 0.011, zbar + 0.02), (90, 0, ra.u(0, 60))))
        irons.append(knuckle("shoulder_knuckle", S))
        irons.append(sweep("upper_arm", [tuple(S), tuple(E)], up_prof, smooth_path=False, up=tuple(n3)))
        irons.append(knuckle("elbow_knuckle", E, 1.08))
        irons.append(sweep("forearm", [tuple(E), tuple(Wp)], fo_prof, smooth_path=False, up=tuple(n3)))
        irons.append(knuckle("wrist_knuckle", Wp, 0.85))
        irons.append(box("wrist_lug", (0.046, 0.058, 0.03), loc=(xw0, Wp.y - 0.027, Wp.z + 0.004), bevel=0.004, segments=1))
        irons.append(box("wrist_plate", (0.1, 0.008, 0.06), loc=(xw0, y_ff + BY + 0.004, bar_z), bevel=0.002, segments=1))
        for sxx in (-1, 1):
            bolt_m.append(trs((xw0 + sxx * 0.036, y_ff + BY + 0.008, bar_z + 0.012), (-90, 0, ra.u(0, 60))))
        for c, s in ((S, 1.0), (E, 1.08), (Wp, 0.85)):
            bolt_m.append(trs(tuple(c + n3 * (0.033 * s)), (bdeg, 0, ra.u(0, 60)), 1.5 * s))
        # tension spring along the upper arm, cable over the elbow to the forearm
        du = (E - S).normalized()
        df = (Wp - E).normalized()
        inw = Vector((-sx, 0, 0))
        side_u = (inw - du * inw.dot(du)).normalized()            # in-plane, towards the fold side
        side_f = (inw - df * inw.dot(df)).normalized()
        La = (E - S).length
        a0 = S + du * (La * 0.2) - side_u * 0.022
        a1 = S + du * (La * 0.5) - side_u * 0.022
        turns = min(13, max(8, int((a1 - a0).length / 0.011)))
        hel = []
        for i in range(turns * 5 + 1):
            tt = i / (turns * 5)
            ang = 2 * pi * turns * tt
            hel.append(tuple(a0 + (a1 - a0) * tt + (side_u * cos(ang) + n3 * sin(ang)) * 0.0085))
        cable_objs.append(sweep("arm_spring", hel, 0.0024, sides=3, smooth_path=False))
        irons.append(box("spring_lug", (0.014, 0.014, 0.02), loc=tuple(a0 - du * 0.012 + side_u * 0.008), rot=(bdeg, 0, 0), bevel=0.002, segments=1))
        c_end = E + df * (La * 0.22) - side_f * 0.02
        cab = [tuple(a1), tuple(E - du * 0.05 - side_u * 0.024), tuple(E - (side_u + side_f).normalized() * 0.036),
               tuple(E + df * 0.05 - side_f * 0.024), tuple(c_end)]
        cable_objs.append(sweep("arm_cable", cab, 0.003, sides=4, step=0.035))
        irons.append(box("cable_lug", (0.014, 0.014, 0.02), loc=tuple(c_end + df * 0.008 + side_f * 0.008), rot=(bdeg, 0, 0), bevel=0.002, segments=1))

    # ------------------------------------------------------------------ forged scroll brackets under the box ends
    flat_prof = [(-0.003, -0.015), (0.003, -0.015), (0.003, 0.015), (-0.003, 0.015)]
    zt = -0.2335
    for sx in (-1, 1):
        x = sx * (W / 2 - 0.016)
        rs = rng.fork(f"scroll{sx}")
        drop = rs.vary(0.3, 0.05)
        forged_objs = [box("scroll_wall_leg", (0.03, 0.007, drop + 0.03), loc=(x, -0.0035, zt - (drop + 0.03) / 2 + 0.003), bevel=0.0015, segments=1),
                       box("scroll_top_leg", (0.03, 0.215, 0.007), loc=(x, -0.1075, zt), bevel=0.0015, segments=1)]
        pth = []
        cy0, cz0 = -0.01 - 0.03, zt - drop
        for i in range(15):
            tt = i / 14
            a = radians(-450 + 450 * tt)
            rr = 0.008 + 0.022 * tt
            pth.append((x, cy0 + rr * cos(a), cz0 + rr * sin(a)))
        pth += bezier_path((x, -0.01, cz0), (x, -0.012, zt - drop * 0.45), (x, -0.1, zt - 0.006), (x, -0.185, zt - 0.006), n=10)[1:]
        cy1, cz1 = -0.185, zt - 0.006 - 0.034
        for i in range(1, 15):
            tt = i / 14
            a = radians(90 + 430 * tt)
            rr = 0.034 - 0.025 * tt
            pth.append((x, cy1 + rr * cos(a), cz1 + rr * sin(a)))
        forged_objs.append(sweep("scroll", pth, flat_prof, smooth_path=False, up=(1, 0, 0), scale_fn=lambda t: 1.0 - 0.35 * abs(2 * t - 1) ** 3))
        cable_objs += forged_objs
        for zz in (zt - 0.05, zt - drop + 0.02):
            bolt_m.append(trs((x, -0.007, zz), (90, 0, rs.u(0, 60)), 0.9))

    bolts.append(instances_to_mesh("hex_bolts", hexb, bolt_m, keep_base=False))

    # ------------------------------------------------------------------ materials and weathering
    rust = noise_mask(0.12, seed=p.seed + 7, lo=0.35, hi=0.8)
    rust_f = noise_mask(0.05, seed=p.seed + 8, lo=0.3, hi=0.7)
    for o in irons:
        assign(o, iron)
    paint(irons, special=mask_mul(rust, 0.2 + 0.7 * age), wear=mask_mul(rust_f, 0.15 + 0.4 * age))
    for o in woods:
        assign(o, wood)
    # paint on the box flakes and fades where the rain runs off the front, and stays sound under the capping
    paint(woods, wear=mask_mul(noise_mask(0.09, seed=p.seed + 9, lo=0.45, hi=0.8), facing((0, -1, -0.3)), 0.25 + 0.75 * age),
          special=mask_mul(facing((0, -1, 0.2)), 0.3 + 0.5 * age),
          grime=mask_mul(noise_mask(0.2, seed=p.seed + 10), facing((0, 0, -1)), 0.3 + 0.5 * age))
    for o in zincs:
        assign(o, zinc)
    # the capping takes the weather: paint chalked by the sun, dirt lying in patches and washed towards the drip edge
    streak = lambda co, n: noise_mask(0.1, seed=p.seed + 13, lo=0.35, hi=0.8)(co * np.array([1.0, 0.3, 0.3]), n)
    drip = lambda co, n: np.clip((-0.1 - co[:, 1]) / 0.14, 0.15, 1.0)
    paint(zincs, special=mask_mul(facing((0, -0.2, 1)), 0.35 + 0.5 * age),
          grime=lambda co, n: np.clip((0.1 + 0.3 * age) * noise_mask(0.22, seed=p.seed + 12, lo=0.4, hi=0.85)(co, n)
                                      + (0.1 + 0.45 * age) * streak(co, n) * drip(co, n), 0.0, 1.0))
    for o in bolts + cable_objs:
        assign(o, forged)
    paint(bolts + cable_objs, special=mask_mul(rust_f, 0.3 + 0.7 * age))

    # ------------------------------------------------------------------ sockets
    socket("hang_front", "attach", (0, bar_y, bar_z - BZ / 2 - 0.045), role="lantern")
    for sx, nm_ in ((-1, "hang_l"), (1, "hang_r")):
        socket(nm_, "attach", (sx * (Wc / 2 - 0.25), ybar, zbar - sqb / 2 - 0.01), role="prop_hang")
