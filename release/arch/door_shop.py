"""Half-glazed, panelled shop door in a projecting timber doorcase: panelled pilasters on plinth blocks,
scrolled consoles carrying a dentilled cornice, panelled reveals and soffit, a rectangular lit fanlight
(transom sash with glazing bars and a gilt house number) over a framed-and-panelled leaf (pair of leaves when
wide) with bolection-moulded lower panels, glazing bars, brass knob / pull, letter plate and screwed kick
plate, a roller blind and an 'OPEN' card hung behind the glass, a shop bell on a coiled spring bracket and
one worn stone step with a dished tread and bull nose. The case is finished on its outer cheeks too (sunk-panel
mouldings, plinth and necking returns, cornice mitred back to the wall under lead end aprons). The glazing is
obscured glass glowing evenly with a shadow-play of counter, shelves and bottles behind it; the optional stained
fanlight is a leaded light of separate quarries in modelled cames. Origin: bottom centre on the wall plane;
everything stays inside width x height.
"""
import bmesh
import numpy as np
from math import exp, cos, sin, radians, pi, hypot
from tk import *

META = {"name": "Glazed shop door", "class": "component", "mount": "wall_ground", "focus": [0.0, -0.25, 1.1]}
PARAMS = {
    "width": {"default": 1.3, "min": 1.1, "max": 1.9, "desc": "outer frame width (m); wide frames get a pair of leaves"},
    "height": {"default": 2.95, "min": 2.6, "max": 3.5, "desc": "outer frame height, pavement to top of cornice (m)"},
    "paint": {"default": "#1d4a3b", "desc": "paint colour of the door leaf"},
    "trim": {"default": "#738268", "desc": "paint colour of the doorcase (pilasters, cornice, linings)"},
    "glazing_bars": {"default": "margin", "choices": ["none", "cross", "grid", "margin"], "desc": "glazing bar pattern of the door light"},
    "sign_text": {"default": "OPEN", "desc": "text on the card hung behind the glass ('' = no card)"},
    "number": {"default": "12", "desc": "gilt house number on the fanlight ('' = none)"},
    "lit": {"default": True, "desc": "shop lit inside: glowing glazing and light spill"},
    "fanlight": {"default": "plain", "choices": ["plain", "stained"], "desc": "fanlight glazing: plain obscured glass or leaded coloured quarries"},
    "handle": {"default": "knob", "choices": ["knob", "pull"], "desc": "brass door furniture"},
    "hand": {"default": "right", "choices": ["right", "left"], "desc": "side of the handle / latch"},
    "blind": {"default": 0.16, "min": 0.0, "max": 0.6, "desc": "how far the roller blind is drawn (fraction of the glass)"},
    "reveal": {"default": 0.24, "min": 0.16, "max": 0.36, "desc": "projection of the doorcase = depth of the recess (m)"},
    "step": {"default": 0.16, "min": 0.1, "max": 0.2, "desc": "rise of the stone step (m)"},
    "scraper": {"default": True, "desc": "wrought-iron boot scraper leaded into the step"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age: chipped paint, scuffed brass, hollowed step"},
    "intensity": {"default": 4.0, "min": 2.0, "max": 30.0, "desc": "light spilling out through the door glass onto the step and paving (cd)"},
}
VARIANTS = {
    "oxblood_grid": {"paint": "#4d1414", "trim": "#4d1414", "glazing_bars": "grid", "handle": "pull", "hand": "left",
                     "fanlight": "stained", "number": "7", "width": 1.2, "height": 2.8, "blind": 0.0, "seed": 4},
    "double_navy": {"width": 1.75, "height": 3.2, "paint": "#1f3558", "trim": "#9a968a", "glazing_bars": "cross",
                    "handle": "pull", "number": "24", "blind": 0.3, "reveal": 0.3, "seed": 9},
    "closed_dark": {"lit": False, "sign_text": "CLOSED", "paint": "#46545c", "trim": "#8d8573", "glazing_bars": "none",
                    "blind": 0.55, "wear": 1.0, "height": 2.7, "number": "3", "step": 0.12, "seed": 15},
    "tall_stained": {"height": 3.4, "width": 1.42, "paint": "#6a4a1c", "trim": "#27362e", "fanlight": "stained",
                     "glazing_bars": "margin", "number": "108", "blind": 0.08, "wear": 0.25, "seed": 23},
}


def _bx(name, x0, x1, y0, y1, z0, z1, bev=0.003, seg=1, grain=None):
    o = box(name, (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), loc=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
            bevel=bev, segments=seg)
    if grain:
        uv_grain(o, axis=grain)
    return o


def _sheet(o, keep=lambda n: n.y < -0.5):
    """Keep only the faces of o that face the viewer (-Y). Glowing glass, blinds and cards are single sheets: every
    exported material is double sided, so a thin glowing box is two sheets a few mm apart that fight each other, and
    its edge faces lie coincident with the timber it is bedded in.
    CLEARANCE RULE: a glowing sheet keeps >= 4 mm to whatever lies parallel behind or in front of it."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not keep(f.normal)], context="FACES")
    bm.to_mesh(o.data)
    bm.free()
    return o


def _mould(name, x0, x1, z0, z1, y, prof, side=0):
    """Mitred moulding frame round the rectangle (x0..x1, z0..z1) on the plane y (facing -Y), or with side=+-1 in
    the YZ plane x = y facing +-X (x0..x1 are then y coordinates).
    prof = [(inset from the outer edge, proud towards the viewer), ...] - a closed cross-section."""
    xa, xb, za, zb = min(x0, x1), max(x0, x1), min(z0, z1), max(z0, z1)
    verts, faces = [], []
    k = len(prof)
    for i, d in prof:
        for u, v in ((xa + i, za + i), (xb - i, za + i), (xb - i, zb - i), (xa + i, zb - i)):
            verts.append((u, y - d, v) if side == 0 else (y + side * d, u, v))
    for j in range(k):
        j1 = (j + 1) % k
        for c in range(4):
            c1 = (c + 1) % 4
            faces.append((j * 4 + c, j * 4 + c1, j1 * 4 + c1, j1 * 4 + c))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, 42)
    return o


def _return_mould(name, prof, xs, yfp, k):
    """Moulding run across the front (plane y = yfp) and mitred back along both cheeks (planes x = +-xs) to the
    wall. prof = [(projection, z), ...] bottom to top along the visible face; the side projection is scaled by k."""
    verts, faces = [], []
    for d, z in prof:
        xx = xs + k * d
        verts += [(-xx, 0.0, z), (-xx, yfp - d, z), (xx, yfp - d, z), (xx, 0.0, z)]
    for j in range(len(prof) - 1):
        for c in range(3):
            faces.append((j * 4 + c, j * 4 + c + 1, (j + 1) * 4 + c + 1, (j + 1) * 4 + c))
    o = mesh_object(name, verts, faces)
    smooth(o, 35)
    return o


def _plate(name, x0, x1, z0, z1, y, th, cell=0.035, ch=0.004):
    """Sheet-metal plate screwed to the plane y: gridded front (for wear masks) with eased edges."""
    def axis(a, b):
        n = max(2, int(round((b - a - 2 * ch) / cell)))
        return [a] + [a + ch + (b - a - 2 * ch) * i / n for i in range(n + 1)] + [b]
    xs, zs = axis(x0, x1), axis(z0, z1)
    nx, nz = len(xs), len(zs)
    verts = [(x, y - (th * 0.4 if (i in (0, nx - 1) or j in (0, nz - 1)) else th), z) for j, z in enumerate(zs) for i, x in enumerate(xs)]
    faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i) for j in range(nz - 1) for i in range(nx - 1)]
    b = len(verts)
    verts += [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    c = [0, nx - 1, nz * nx - 1, (nz - 1) * nx]
    faces += [(b, b + 1, c[1], c[0]), (b + 1, b + 2, c[2], c[1]), (b + 2, b + 3, c[3], c[2]), (b + 3, b, c[0], c[3])]
    o = mesh_object(name, verts, faces)
    smooth(o, 40)
    return o


def _clip(poly, x0, x1, z0, z1):
    """Clip a convex polygon [(x, z), ...] to a rectangle (Sutherland-Hodgman)."""
    for ax, lim, sg in ((0, x0, 1.0), (0, x1, -1.0), (1, z0, 1.0), (1, z1, -1.0)):
        out = []
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            ia, ib = sg * (a[ax] - lim) >= -1e-9, sg * (b[ax] - lim) >= -1e-9
            if ia:
                out.append(a)
            if ia != ib:
                t = (lim - a[ax]) / (b[ax] - a[ax])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        poly = out
        if len(poly) < 3:
            return []
    area = 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
    return poly if abs(area) > 4e-5 else []


def _leaded(x0, x1, z0, z1, r):
    """Leaded light filling a rectangle: a border of small oblong quarries round a field of diamond quarries.
    Returns ([(tint, polygon)], [came segment]) - tints: 'clear', 'amber', 'ruby', 'green'."""
    quarries, segs, seen = [], [], set()
    bw = min(0.034, (x1 - x0) * 0.14)
    fx0, fx1, fz0, fz1 = x0 + bw, x1 - bw, z0 + bw, z1 - bw

    def add(tint, poly, skip_field_edge=False):
        if len(poly) < 3:
            return
        quarries.append((tint, poly))
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            if any(abs(a[ax] - v) < 1e-6 and abs(b[ax] - v) < 1e-6 for ax, v in ((0, x0), (0, x1), (1, z0), (1, z1))):
                continue                       # under the sash rebate
            if skip_field_edge and any(abs(a[ax] - v) < 1e-6 and abs(b[ax] - v) < 1e-6 for ax, v in ((0, fx0), (0, fx1), (1, fz0), (1, fz1))):
                continue                       # the border quarries already carry this came
            key = tuple(sorted(((round(a[0], 4), round(a[1], 4)), (round(b[0], 4), round(b[1], 4)))))
            if key not in seen:
                seen.add(key)
                segs.append((a, b))

    def rect(xa, xb, za, zb):
        return [(xa, za), (xb, za), (xb, zb), (xa, zb)]

    for xa, xb in ((x0, fx0), (fx1, x1)):      # corner squares
        for za, zb in ((z0, fz0), (fz1, z1)):
            add("amber", rect(xa, xb, za, zb))
    nh = max(2, int(round((fx1 - fx0) / 0.085)))
    nv = max(2, int(round((fz1 - fz0) / 0.085)))
    for k in range(nh):                        # top and bottom runs
        xa, xb = fx0 + (fx1 - fx0) * k / nh, fx0 + (fx1 - fx0) * (k + 1) / nh
        add("ruby" if k % 2 == 0 else "green", rect(xa, xb, z0, fz0))
        add("green" if k % 2 == 0 else "ruby", rect(xa, xb, fz1, z1))
    for k in range(nv):                        # side runs
        za, zb = fz0 + (fz1 - fz0) * k / nv, fz0 + (fz1 - fz0) * (k + 1) / nv
        add("green" if k % 2 == 0 else "ruby", rect(x0, fx0, za, zb))
        add("ruby" if k % 2 == 0 else "green", rect(fx1, x1, za, zb))
    ax = (fx1 - fx0) / 4.0                     # two diamonds across the field
    nrow = max(1, int(round((fz1 - fz0) / (2 * ax * 1.45))))
    az = (fz1 - fz0) / (2.0 * nrow)
    for i in range(5):
        for j in range(2 * nrow + 1):
            if (i + j) % 2:
                continue
            cx, cz = fx0 + i * ax, fz0 + j * az
            di, dj = i - 2, j - nrow
            if di == 0 and dj == 0:
                tint = "ruby"
            elif abs(di) + abs(dj) == 2 and di != 0:
                tint = "green"
            else:
                tint = "amber" if ((di - dj) // 2) % 2 else "clear"
                if r.chance(0.12):
                    tint = "clear" if tint == "amber" else "amber"
            add(tint, _clip([(cx - ax, cz), (cx, cz - az), (cx + ax, cz), (cx, cz + az)], fx0, fx1, fz0, fz1), True)
    return quarries, segs


def _step(W, sd, sh, depth, cx0, sig, chips):
    """One stone step: bull-nosed, eased ends, tread dished `depth` m round x = cx0 where the feet land, nosing
    chipped at `chips` = [(x, radius, bite)]. Hidden back and bottom are left open."""
    rn = min(0.03, sh * 0.28)
    c = 0.012
    ny = max(6, int(round((sd - rn) / 0.06)))
    prof = [(-(sd - rn) * i / ny, sh) for i in range(ny + 1)]
    prof += [(-sd + rn - rn * sin(radians(a)), sh - rn + rn * cos(radians(a))) for a in (22.5, 45.0, 67.5, 90.0)]
    prof += [(-sd, (sh - rn) * 0.5), (-sd, 0.0)]
    nx = int(clamp(round((W - 2 * c) / 0.052), 8, 22))
    xs = [-W / 2] + [-W / 2 + c + (W - 2 * c) * i / nx for i in range(nx + 1)] + [W / 2]
    n = len(prof)
    verts = []
    for ci, x in enumerate(xs):
        end = ci in (0, len(xs) - 1)
        for y, z in prof:
            if end:
                y, z = max(y, -sd + c * 0.8), min(z, sh - c * 0.8)
            wz = smoothstep(sh * 0.4, sh * 0.92, z)
            wy = smoothstep(0.07, 0.2, -y)
            g = exp(-((x - cx0) / sig) ** 2)
            z -= depth * g * wy * wz
            y += 0.6 * depth * g * wz * smoothstep(sd - 0.07, sd - 0.01, -y)   # nosing worn back where the feet land
            for xc, rad, bite in chips:
                d = ((x - xc) ** 2 + (y + sd - rn * 0.3) ** 2 + (z - sh + rn * 0.3) ** 2) ** 0.5
                if d < rad:
                    f = bite * (1.0 - d / rad) ** 0.6
                    y, z = y + 0.75 * f, z - 0.65 * f
            verts.append((x, min(y, 0.0), max(z, 0.0)))
    faces = [(ci * n + j, ci * n + j + 1, (ci + 1) * n + j + 1, (ci + 1) * n + j) for ci in range(len(xs) - 1) for j in range(n - 1)]
    b = len(verts)
    verts += [(-W / 2, 0.0, 0.0), (W / 2, 0.0, 0.0)]
    faces.append(tuple([b] + list(range(n - 1, -1, -1))))
    faces.append(tuple(list(range((len(xs) - 1) * n, len(xs) * n)) + [b + 1]))
    o = mesh_object("step", verts, faces)
    smooth(o, 55)
    return o


BOLECTION = [(0.0, 0.0), (0.0, 0.011), (0.004, 0.0165), (0.011, 0.0175), (0.017, 0.013), (0.021, 0.006), (0.027, 0.004), (0.027, 0.0)]
BEAD = [(0.0, 0.0), (0.0, 0.013), (0.005, 0.0165), (0.012, 0.014), (0.016, 0.007), (0.021, 0.004), (0.021, 0.0)]
PLANT = [(0.0, 0.0), (0.001, 0.005), (0.005, 0.0085), (0.011, 0.0085), (0.014, 0.004), (0.018, 0.003), (0.018, 0.0)]


def _screw(name, r=0.0048, h=0.0022, n=6):
    verts = [(r * cos(2 * pi * i / n), r * sin(2 * pi * i / n), 0.0) for i in range(n)]
    verts += [(0.66 * r * cos(2 * pi * i / n), 0.66 * r * sin(2 * pi * i / n), h * 0.8) for i in range(n)] + [(0.0, 0.0, h)]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [(n + i, n + (i + 1) % n, 2 * n) for i in range(n)]
    o = mesh_object(name, verts, faces)          # no underside: it is bedded on the plate
    smooth(o, 70)
    return o


def _prism_x(name, yz, x0, x1):
    """Extrude a (y, z) section along X - cornices, consoles, drip moulds."""
    n = len(yz)
    verts = [(x0, y, z) for y, z in yz] + [(x1, y, z) for y, z in yz]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, 35)
    uv_grain(o, axis="X")
    return o


def _near(pt, r):
    c = np.array(pt, dtype=np.float64)
    return lambda co, n: np.clip(1.0 - np.linalg.norm(co - c, axis=1) / r, 0.0, 1.0)


def build(p, rng):
    W, H, D, sh = p.width, p.height, p.reveal, p.step
    age = p.wear
    hs = 1 if p.hand == "right" else -1
    e = 0.035                                  # pilaster shafts sit this far inside the plinth / cornice ends
    xs_ = W / 2 - e                            # outer cheek of the pilaster shafts
    jw = clamp(0.115 * W + 0.015, 0.13, 0.2)   # pilaster width
    ow = W - 2 * e - 2 * jw                    # clear opening between the pilasters
    head_h = 0.075 * H
    corn_h = 0.085
    zc = H - head_h                            # soffit of the opening
    zk1 = H - corn_h                           # underside of the cornice
    lin, tb = 0.032, 0.07                      # frame lining, transom bar
    avail = zc - sh - lin - tb - 0.018
    hd = clamp(avail - 0.40, 1.9, 2.1)         # leaf height
    z_d0 = sh + 0.014
    z_d1 = z_d0 + hd
    z_t0 = z_d1 + 0.004
    z_t1 = z_t0 + tb
    z_f0, z_f1 = z_t1, zc - lin                # fanlight sash
    yb, yf = -0.014, -0.060                    # leaf back / front
    yl = -0.080                                # front of the frame lining

    trim, door, brass, iron, stone, glow, shade, card, lead, ink = [], [], [], [], [], [], [], [], [], []
    quarry = {"clear": [], "amber": [], "ruby": [], "green": []}   # leaded-light pieces: (polygon, y)
    cames = []

    # ------------------------------------------------------------------ stone step
    sd = D + 0.30
    ow0 = W - 2 * e - 2 * jw
    sr = rng.fork("step")
    chips = [(sx * (W / 2 - sr.u(0.0, 0.03)), sr.u(0.03, 0.05), sr.u(0.006, 0.012)) for sx in (-1, 1)]
    chips += [(sr.u(-W / 2, W / 2), sr.u(0.022, 0.05), sr.u(0.004, 0.011) * (0.5 + age)) for _ in range(3 + int(round(7 * age)))]
    st = _step(W, sd, sh, 0.004 + 0.012 * age, hs * 0.05, 0.3 * ow0, chips)   # tread hollowed by a century of feet
    displace_noise(st, strength=0.0016, scale=0.11, seed=p.seed)
    transform_verts(st, lambda co: Vector((clamp(co.x, -W / 2, W / 2), min(co.y, 0.0), max(co.z, 0.0))))
    stone.append(st)
    brass.append(_bx("threshold", -ow / 2 + lin, ow / 2 - lin, -0.010, -0.084, sh - 0.004, sh + 0.011, 0.004, 2))

    if p.scraper:                              # boot scraper on the hinge side, feet run in with lead
        bx_, by_, bz_ = -hs * (W / 2 - 0.105), -D - 0.17, sh - 0.004
        for sx in (-1, 1):
            iron.append(lathe("scraper_post", [(0.008, 0.0), (0.0065, 0.012), (0.006, 0.105), (0.009, 0.112), (0.011, 0.122), (0.008, 0.132), (0.0, 0.136)],
                              segments=8, loc=(bx_ + sx * 0.075, by_, bz_)))
            lead.append(cylinder("scraper_lead", 0.016, 0.006, loc=(bx_ + sx * 0.075 + rng.u(-0.003, 0.003), by_, sh - 0.001), verts=8, bevel=0.002, segments=1))
        blade = _bx("scraper_blade", bx_ - 0.075, bx_ + 0.075, by_ - 0.002, by_ + 0.002, bz_ + 0.05, bz_ + 0.09, 0.0012, 1)
        iron.append(blade)

    # ------------------------------------------------------------------ doorcase: pilasters, consoles, entablature
    hc = 0.28
    zk0 = zk1 - hc
    zp1 = sh + 0.26                            # top of the plinth blocks
    z_lock0, z_lock1 = z_d0 + 0.85, z_d0 + 1.04
    con = [(0, 0), (0.016, 0.0), (0.03, 0.018), (0.036, 0.045), (0.033, 0.08), (0.026, 0.115), (0.027, 0.15),
           (0.037, 0.185), (0.054, 0.215), (0.066, 0.24), (0.07, 0.262), (0.07, 0.28), (0, 0.28)]
    for sx in (-1, 1):
        xo, xi = sx * (W / 2 - e), sx * (W / 2 - e - jw)
        xm = (xo + xi) / 2
        sfx = "_l" if sx < 0 else "_r"
        trim.append(_bx("pilaster" + sfx, xo, xi, 0, -D, sh, zc, 0.004, 1, "Z"))
        densify(trim[-1], 0.4)
        trim.append(_bx("plinth" + sfx, sx * (W / 2 - 0.012), xi - sx * 0.012, 0, -D - 0.018, sh - 0.004, zp1, 0.006, 1, "Z"))
        densify(trim[-1], 0.1)
        trim.append(_bx("plinth_cap" + sfx, sx * (W / 2 - 0.019), xi - sx * 0.007, 0, -D - 0.011, zp1, zp1 + 0.028, 0.011, 1, "X"))
        trim.append(_bx("necking" + sfx, sx * (xs_ + 0.012), xi - sx * 0.008, 0, -D - 0.009, zk0 - 0.03, zk0, 0.008, 1, "X"))
        # the outer cheek is finished too: sunk-panel mouldings returned round the side
        for k, (a, b) in enumerate(((zp1 + 0.075, z_lock0 - 0.02), (z_lock1 + 0.02, zk0 - 0.075))):
            trim.append(_mould(f"cheek_mould{k}" + sfx, -D + 0.032, -0.03, a, b, xo, PLANT, side=sx))
        # sunk-panel mouldings planted on the pilaster face, split at lock-rail height
        for k, (a, b) in enumerate(((zp1 + 0.075, z_lock0 - 0.02), (z_lock1 + 0.02, zk0 - 0.075))):
            trim.append(_mould(f"pil_mould{k}" + sfx, xm - jw / 2 + 0.026, xm + jw / 2 - 0.026, a, b, -D + 0.001, PLANT))
        # panelled reveal (inner cheek of the pilaster): two raised fields
        for k, (a, b) in enumerate(((zp1 + 0.06, z_lock0 - 0.02), (z_lock1 + 0.02, zc - 0.07))):
            trim.append(_bx(f"reveal_panel{k}" + sfx, xi, xi - sx * 0.009, yl - 0.022, -D + 0.022, a, b, 0.007, 1, "Z"))
        # scrolled console under the cornice
        c = _prism_x("console" + sfx, [(-D - d, zk0 + z) for d, z in con], xm - (jw - 0.034) / 2, xm + (jw - 0.034) / 2)
        trim.append(c)
        trim.append(_bx("console_rib" + sfx, xm - 0.012, xm + 0.012, -D - 0.02, -D - 0.045, zk0 + 0.03, zk0 + 0.2, 0.008, 1, "Z"))
    trim.append(_bx("frieze", -xs_, xs_, 0, -D - 0.004, zc, zk1 + 0.002, 0.003, 1, "X"))
    xci = W / 2 - e - jw + 0.017               # inner face of the consoles
    trim.append(_bx("architrave_bead", -xci, xci, -D, -D - 0.018, zc + 0.004, zc + 0.03, 0.009, 2, "X"))
    trim.append(_bx("soffit_panel", -ow / 2 + 0.05, ow / 2 - 0.05, yl - 0.022, -D + 0.022, zc - 0.009, zc + 0.01, 0.007, 1, "X"))
    cy = -D
    # cornice: bed fillet, ovolo, corona and cyma, mitred back to the wall along both cheeks
    cor = [(-0.02, 0.0), (0.078, 0.0), (0.078, 0.028), (0.083, 0.033), (0.085, 0.044), (0.093, 0.056), (0.105, 0.064),
           (0.111, 0.066), (0.111, 0.073), (-0.02, 0.073)]
    trim.append(_return_mould("cornice", [(d, zk1 + z) for d, z in cor], xs_, cy, (e - 0.003) / 0.111))
    trim.append(_prism_x("cornice_blocking", [(0, zk1 + 0.07), (cy - 0.1, zk1 + 0.07), (cy - 0.111, zk1 + 0.0735), (0, H - 0.004)],
                         -(W / 2 - 0.004), W / 2 - 0.004))
    # lead capping dressed over the cornice with a turned-down drip edge and aprons over both ends
    cap = [(0, H - 0.004), (0, H), (cy - 0.117, zk1 + 0.077), (cy - 0.117, zk1 + 0.056), (cy - 0.1135, zk1 + 0.056), (cy - 0.1135, zk1 + 0.073)]
    lead.append(_prism_x("lead_cap", cap, -W / 2, W / 2))
    for sx in (-1, 1):
        lead.append(_prism_x("lead_apron", [(0, zk1 + 0.058 + rng.u(0, 0.004)), (cy - 0.117, zk1 + 0.056), (cy - 0.117, zk1 + 0.077), (0, H)],
                             min(sx * (W / 2 - 0.0045), sx * W / 2), max(sx * (W / 2 - 0.0045), sx * W / 2)))
    for k in range(1, 3):                      # wood-cored rolls where the sheets join
        xr = -W / 2 + W * k / 3 + rng.u(-0.03, 0.03)
        lead.append(_prism_x("lead_roll", [(-0.004, H - 0.0045), (-0.004, H), (cy - 0.117, zk1 + 0.077), (cy - 0.117, zk1 + 0.0715)], xr - 0.011, xr + 0.011))
    nd = max(3, int(round(2 * xci / 0.062)))
    dent = mesh_object("dentil", [(sx_ * 0.016, y_, z_) for z_ in (-0.034, 0.0) for y_ in (0.0, -0.024) for sx_ in (-1, 1)],
                       [(2, 3, 7, 6), (0, 2, 6, 4), (3, 1, 5, 7), (0, 1, 3, 2)])    # front, cheeks, underside; top and back are bedded
    pitch = 2 * xci / nd
    trim.append(instances_to_mesh("dentils", dent, [trs((-xci + pitch * (k + 0.5), -D - 0.004, zk1)) for k in range(nd)], keep_base=False))

    # ------------------------------------------------------------------ frame lining, transom bar, fanlight sash
    ox = ow / 2
    for sx in (-1, 1):
        trim.append(_bx("lining_jamb", sx * ox, sx * (ox - lin), 0, yl, sh, zc - lin, 0.004, 1, "Z"))
        densify(trim[-1], 0.55)
    trim.append(_bx("lining_head", -ox, ox, 0, yl, zc - lin, zc, 0.004, 1, "X"))
    trim.append(_bx("transom", -(ox - lin), ox - lin, 0, yl - 0.008, z_t0, z_t1, 0.006, 2, "X"))
    trim.append(_prism_x("transom_drip", [(yl - 0.006, z_t1 - 0.03), (yl - 0.02, z_t1 - 0.026), (yl - 0.026, z_t1 - 0.016),
                                          (yl - 0.026, z_t1 - 0.006), (yl - 0.006, z_t1 + 0.004)], -(ox - lin), ox - lin))
    fx0, fx1 = -(ox - lin), ox - lin
    fs = 0.034
    trim.append(_bx("fan_rail_b", fx0, fx1, -0.02, -0.062, z_f0, z_f0 + fs, 0.004, 1, "X"))
    trim.append(_bx("fan_rail_t", fx0, fx1, -0.02, -0.062, z_f1 - fs, z_f1, 0.004, 1, "X"))
    trim.append(_bx("fan_stile_l", fx0, fx0 + fs, -0.02, -0.062, z_f0 + fs, z_f1 - fs, 0.004, 1, "Z"))
    trim.append(_bx("fan_stile_r", fx1 - fs, fx1, -0.02, -0.062, z_f0 + fs, z_f1 - fs, 0.004, 1, "Z"))
    gx0, gx1, gz0, gz1 = fx0 + fs, fx1 - fs, z_f0 + fs, z_f1 - fs
    npane = 3 if (gx1 - gx0) < 1.0 else 5
    pw = (gx1 - gx0) / npane
    for k in range(1, npane):
        trim.append(_bx("fan_bar", gx0 + k * pw - 0.009, gx0 + k * pw + 0.009, -0.026, -0.057, gz0, gz1, 0.005, 1, "Z"))
    fr = rng.fork("fanlight")
    for k in range(npane):
        if p.fanlight == "stained" and k != npane // 2:
            qs, sg = _leaded(gx0 + k * pw, gx0 + (k + 1) * pw, gz0, gz1, fr)
            for tint, poly in qs:
                quarry[tint].append(poly)
            cames.extend(sg)
        else:
            glow.append(_sheet(_bx("fan_glass", gx0 + k * pw, gx0 + (k + 1) * pw, -0.026, -0.031, gz0, gz1, 0.0)))
    yq = -0.031                                # front face of the fanlight glass

    def fit_text(name, body, size, maxw, cx, cz, y, depth, font, bev=0.0, off=(0.0, 0.0)):
        t = text(name, body, size=size, depth=depth, font=font, bevel=bev)
        mn, mx = world_bbox([t])
        s = min(1.0, maxw / max(mx.x - mn.x, 1e-4))
        place(t, loc=(cx + off[0] - s * (mn.x + mx.x) / 2, y - s * mx.y, cz + off[1] - s * (mn.z + mx.z) / 2), scale=(s, s, s))
        apply_transform(t)
        # signwriting is a film of paint / gold leaf on the glass: keep the face, drop the hidden back and the edges
        bm = bmesh.new()
        bm.from_mesh(t.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > -0.5], context="FACES")
        for v in bm.verts:                     # the film lies in one plane, a known distance off the glass
            v.co.y = y - depth
        bm.to_mesh(t.data)
        bm.free()
        return t

    num = str(p.number).strip()
    if num:                                    # gilt numerals with a black drop shadow, as a signwriter does them on glass
        nsz, ncx, ncz = min(0.17, (gz1 - gz0) * 0.56), gx0 + (npane // 2 + 0.5) * pw, (gz0 + gz1) / 2
        # (glass 0.031 | shadow 0.0352 | gilt 0.0394: 4.2 mm steps, see _sheet)
        shade.append(fit_text("number", num, nsz, pw - 0.06, ncx, ncz, yq - 0.0072, 0.0012, "serif"))
        ink.append(fit_text("number_shadow", num, nsz, pw - 0.06, ncx, ncz, yq - 0.003, 0.0012, "serif", 0.0, (0.0045, -0.0045)))

    # ------------------------------------------------------------------ door leaf / leaves
    lx0, lx1 = -(ox - lin) + 0.003, (ox - lin) - 0.003
    double = (lx1 - lx0) > 1.06
    info = {}

    def leaf(x0, x1, latch, tag, letter):
        lw = x1 - x0
        sw = 0.105 if lw > 0.7 else 0.088
        zr0, zr1 = z_d0 + 0.24, z_lock0
        zg0, zg1 = z_lock1, z_d1 - sw
        xa, xb = x0 + sw, x1 - sw
        door.append(_bx("stile_a" + tag, x0, xa, yb, yf, z_d0, z_d1, 0.003, 1, "Z"))
        door.append(_bx("stile_b" + tag, xb, x1, yb, yf, z_d0, z_d1, 0.003, 1, "Z"))
        densify(door[-1], 0.34), densify(door[-2], 0.34)
        for nm, a, b in (("bottom", z_d0, zr0), ("lock", zr1, zg0), ("top", zg1, z_d1)):
            door.append(_bx(f"rail_{nm}" + tag, xa, xb, yb, yf + rng.u(0.0, 0.0012), a, b, 0.003, 1, "X"))
            densify(door[-1], 0.21)
        cols = [(xa, xb)]
        if xb - xa > 0.5:
            m = (xa + xb) / 2
            door.append(_bx("muntin" + tag, m - 0.045, m + 0.045, yb, yf + 0.0008, zr0, zr1, 0.003, 1, "Z"))
            cols = [(xa, m - 0.045), (m + 0.045, xb)]
        for i, (c0, c1) in enumerate(cols):
            door.append(_bx(f"panel_back{i}" + tag, c0, c1, yf + 0.03, yf + 0.02, zr0, zr1, 0.0, 1, "Z"))
            door.append(_bx(f"panel_field{i}" + tag, c0 + 0.048, c1 - 0.048, yf + 0.022, yf + 0.007, zr0 + 0.048, zr1 - 0.048, 0.011, 1, "Z"))
            door.append(_mould(f"panel_mould{i}" + tag, c0, c1, zr0, zr1, yf + 0.0195, BOLECTION))
        # glazing: bead, pane, bars, warm room behind
        door.append(_mould("glaze_bead" + tag, xa, xb, zg0, zg1, yf + 0.0195, BEAD))
        gw, gh = xb - xa, zg1 - zg0
        glow.append(_sheet(_bx("pane" + tag, xa, xb, yb + 0.004, yb, zg0, zg1, 0.0)))
        if p.lit:                              # shadow-play of the shop behind the obscured glass: counter, till, shelves of bottles
            ir = rng.fork("interior" + tag)
            qd = []

            def R(a, b, c, d):
                a, b, d = max(a, xa), min(b, xb), min(d, zg1)
                if b - a > 0.004 and d - c > 0.004:
                    qd.append([(a, c), (b, c), (b, d), (a, d)])

            zt = zg0 + gh * ir.u(0.15, 0.19)
            R(xa, xb, zg0, zt)
            tx = xa + gw * ir.u(0.08, 0.6)
            R(tx, tx + 0.12, zt, zt + 0.065), R(tx + 0.02, tx + 0.085, zt + 0.065, zt + 0.105)
            for f in (0.44, 0.7):
                zs_ = zg0 + gh * f + ir.u(-0.02, 0.02)
                R(xa, xb, zs_ - 0.011, zs_)
                x = xa + ir.u(0.008, 0.05)
                while x < xb - 0.06:
                    w, h = ir.u(0.03, 0.056), min(ir.u(0.07, 0.15), zg1 - zs_ - 0.012)
                    kind = ir.weighted([("bottle", 0.5), ("jar", 0.3), ("gap", 0.2)])
                    if kind == "bottle" and h > 0.05:      # shouldered bottle with a neck
                        hb, nk = h * ir.u(0.5, 0.62), w * ir.u(0.14, 0.2)
                        qd.append([(x, zs_), (x + w, zs_), (x + w, zs_ + hb), (x + w / 2 + nk, zs_ + hb + w * 0.32), (x + w / 2 + nk, zs_ + h),
                                   (x + w / 2 - nk, zs_ + h), (x + w / 2 - nk, zs_ + hb + w * 0.32), (x, zs_ + hb)])
                    elif kind == "jar" and h > 0.05:       # squat jar with a lid
                        R(x, x + w * 1.15, zs_, zs_ + h * 0.6), R(x + w * 0.12, x + w * 1.03, zs_ + h * 0.6, zs_ + h * 0.6 + 0.012)
                    x += w * 1.15 + ir.u(0.006, 0.04)
            vs, fs_ = [], []
            for poly in qd:
                n0 = len(vs)
                vs += [(a, yb - 0.0055, c) for a, c in poly]
                fs_.append(tuple(range(n0, n0 + len(poly))))
            shade.append(mesh_object("interior" + tag, vs, fs_))

        vx, hz = [], []

        def vbar(x):
            vx.append(x)
            door.append(_bx("gbar_v" + tag, x - 0.011, x + 0.011, yf + 0.018, yf + 0.003, zg0, zg1, 0.005, 1, "Z"))

        def hbar(z):
            hz.append(z)
            door.append(_bx("gbar_h" + tag, xa, xb, yf + 0.018, yf + 0.0042, z - 0.011, z + 0.011, 0.005, 1, "X"))

        s = p.glazing_bars
        if s == "cross":
            vbar((xa + xb) / 2)
            hbar(zg0 + gh * 0.5)
        elif s == "grid":
            nv = 2 if gw > 0.62 else (1 if gw > 0.3 else 0)
            for k in range(1, nv + 1):
                vbar(xa + gw * k / (nv + 1))
            for k in (1, 2):
                hbar(zg0 + gh * k / 3)
        elif s == "margin":
            m = min(0.095, gw * 0.2)
            vbar(xa + m), vbar(xb - m), hbar(zg0 + m), hbar(zg1 - m)
        # the pane the sign card hangs in: the one at reading height, central or towards the handle
        cols, rows = [xa] + sorted(vx) + [xb], [zg0] + sorted(hz) + [zg1]
        ri = max(i for i in range(len(rows) - 1) if rows[i] <= zg0 + gh * 0.42)
        nc = len(cols) - 1
        ci = nc // 2 if (nc % 2 or latch > 0) else nc // 2 - 1
        cell = (cols[ci] + 0.012, cols[ci + 1] - 0.012, rows[ri] + 0.012, rows[ri + 1] - 0.012)
        # roller blind behind the glass
        if p.blind > 0.02:
            drop = gh * clamp(p.blind + rng.u(-0.02, 0.02), 0.03, 0.75)
            shade.append(_sheet(_bx("blind" + tag, xa + 0.004, xb - 0.004, -0.026, -0.030, zg1 - drop, zg1, 0.0)))
            door.append(_bx("blind_lath" + tag, xa + 0.004, xb - 0.004, -0.026, -0.033, zg1 - drop - 0.014, zg1 - drop, 0.002, 1, "X"))
            xr = (xa + xb) / 2
            iron.append(sweep("blind_cord" + tag, [(xr, -0.031, zg1 - drop - 0.012), (xr, -0.031, zg1 - drop - 0.06)], 0.0012, sides=4, smooth_path=False))
            brass.append(torus("blind_ring" + tag, 0.011, 0.002, loc=(xr, -0.031, zg1 - drop - 0.071), rot=(90, 0, 0), major_seg=10, minor_seg=4))
        # brass kick plate with screws
        k0, k1, kz0, kz1 = x0 + 0.04, x1 - 0.04, z_d0 + 0.028, z_d0 + 0.218
        kick.append(_plate("kick_plate" + tag, k0, k1, kz0, kz1, yf, 0.0028, cell=0.042))
        ns = max(3, int(round((k1 - k0) / 0.16)) + 1)
        scr = [(k0 + 0.016 + (k1 - k0 - 0.032) * i / (ns - 1), z) for i in range(ns) for z in (kz0 + 0.016, kz1 - 0.016)]
        scr += [(k0 + 0.016, (kz0 + kz1) / 2), (k1 - 0.016, (kz0 + kz1) / 2)]
        screws.extend(trs((x, yf - 0.0028, z), (90, 0, rng.u(0, 180)), 1.45) for x, z in scr)
        # letter plate on the lock rail
        cx, zl = (x0 + x1) / 2, (zr1 + zg0) / 2
        if letter:
            brass.append(shape_extrude("letter_plate", rounded_rect(0.27, 0.086, 0.01, 3), [rounded_rect(0.202, 0.044, 0.004, 1)],
                                       depth=0.0045, loc=(cx, yf, zl), rot=(90, 0, 0), bevel=0.0015))
            brass.append(_bx("letter_flap", cx - 0.099, cx + 0.099, yf - 0.0005, yf - 0.0032, zl - 0.02, zl + 0.02, 0.0012, 1))
            brass.append(cylinder("flap_hinge", 0.0042, 0.196, loc=(cx, yf - 0.0045, zl + 0.0225), rot=(0, 90, 0), verts=8))
            screws.extend(trs((cx + sxx * 0.119, yf - 0.0045, zl), (90, 0, rng.u(0, 180))) for sxx in (-1, 1))
        # handle on the latch stile
        kx = (x1 - sw / 2) if latch > 0 else (x0 + sw / 2)
        kz = z_d0 + 0.985
        if p.handle == "knob":
            kn = [(0.035, 0.0), (0.037, 0.003), (0.034, 0.006), (0.027, 0.007), (0.023, 0.0105), (0.014, 0.0125), (0.0115, 0.03),
                  (0.0145, 0.036), (0.024, 0.04), (0.0315, 0.048), (0.0335, 0.057), (0.0305, 0.066), (0.0215, 0.0735), (0.009, 0.0768), (0.0, 0.0772)]
            brass.append(lathe("knob" + tag, kn, segments=12, loc=(kx, yf, kz), rot=(90, 0, 0)))
            ez = kz - 0.115
        else:
            brass.append(_bx("pull_plate" + tag, kx - 0.026, kx + 0.026, yf, yf - 0.004, kz - 0.1, kz + 0.28, 0.0015, 1))
            for dz in (-0.04, 0.22):
                brass.append(lathe("pull_post" + tag, [(0.013, 0.0), (0.014, 0.003), (0.008, 0.008), (0.007, 0.04), (0.0105, 0.047), (0.0, 0.05)],
                                   segments=8, loc=(kx, yf - 0.004, kz + dz), rot=(90, 0, 0)))
            grip = [(0.0, -0.022), (0.011, -0.019), (0.013, -0.008), (0.0085, 0.004), (0.0095, 0.02), (0.0125, 0.15),
                    (0.0095, 0.28), (0.0085, 0.296), (0.013, 0.308), (0.011, 0.319), (0.0, 0.322)]
            brass.append(lathe("pull_grip" + tag, grip, segments=8, loc=(kx, yf - 0.052, kz - 0.06)))
            screws.extend(trs((kx, yf - 0.004, kz + dz), (90, 0, rng.u(0, 180))) for dz in (-0.082, 0.262))
            ez = kz - 0.16
        brass.append(lathe("escutcheon" + tag, [(0.017, 0.0), (0.0185, 0.002), (0.015, 0.0045), (0.0, 0.005)], segments=14, loc=(kx, yf, ez), rot=(90, 0, 0)))
        iron.append(_bx("keyhole" + tag, kx - 0.0022, kx + 0.0022, yf - 0.004, yf - 0.0056, ez - 0.008, ez + 0.006, 0.0008, 1))
        info[tag] = dict(xa=xa, xb=xb, zg0=zg0, zg1=zg1, kx=kx, kz=kz, x0=x0, x1=x1, cell=cell)

    screws, kick = [], []
    if double:
        leaf(lx0, -0.002, 1, "_L", hs < 0)
        leaf(0.002, lx1, -1, "_R", hs > 0)
        act = info["_R" if hs > 0 else "_L"]
        door.append(_bx("astragal", -0.009, 0.009, yf, yf - 0.012, z_d0 + 0.215, z_d1, 0.004, 1, "Z"))
    else:
        leaf(lx0, lx1, hs, "", True)
        act = info[""]
    sb = _screw("screw")
    brass.append(instances_to_mesh("screws", sb, screws, keep_base=False))
    brass += kick

    # ------------------------------------------------------------------ 'OPEN' card on a cord behind the glass
    body = str(p.sign_text).strip()
    cr = rng.fork("card")
    if body:
        cx0, cx1, cz0, cz1 = act["cell"]       # the card fits inside one pane so no glazing bar crosses the lettering
        ch = min(clamp(cx1 - cx0 - 0.05, 0.12, 0.32) * 0.5, (cz1 - cz0) * 0.55)
        cw = ch * 2.0
        hang = min(0.085, max(0.03, (cz1 - cz0 - ch) * 0.6))
        ccx, ccz = (cx0 + cx1) / 2, (cz0 + cz1) / 2 - hang * 0.35
        yc = -0.032
        hook = (ccx, ccz + ch / 2 + hang)
        tilt = radians(cr.u(-4.0, 4.0))
        # the card is one glowing sheet 4.5 mm in front of the blind; its border and lettering stand 4.2 mm proud of
        # it (the border is a little frame bedded on the card, open at the back), the cord hangs in front (see _sheet)
        ycf = yc - 0.0025
        pieces = [(_sheet(_bx("card", ccx - cw / 2, ccx + cw / 2, yc, ycf, ccz - ch / 2, ccz + ch / 2, 0.0, 1)), card),
                  (_sheet(_mould("card_border", ccx - cw / 2 + 0.007, ccx + cw / 2 - 0.007, ccz - ch / 2 + 0.007, ccz + ch / 2 - 0.007, ycf,
                                 [(0.0, 0.0), (0.0, 0.0042), (0.0045, 0.0042), (0.0045, 0.0)]), lambda n: n.y < 0.5), door),
                  (fit_text("card_text", body, ch * 0.46, cw - 0.042, ccx, ccz, ycf - 0.0034, 0.0008, "sans"), ink)]
        for sx in (-1, 1):
            pieces.append((sweep("card_cord", [(ccx + sx * cw * 0.4, ycf - 0.006, ccz + ch / 2 - 0.006), (hook[0], yc - 0.001, hook[1])], 0.0011,
                                 sides=4, smooth_path=False), iron))
        ca, sa = cos(tilt), sin(tilt)

        def swing(co):
            dx, dz = co.x - hook[0], co.z - hook[1]
            return Vector((hook[0] + dx * ca - dz * sa, co.y, hook[1] + dx * sa + dz * ca))

        for o, lst in pieces:
            apply_transform(o)
            transform_verts(o, swing)
            lst.append(o)
        brass.append(lathe("card_hook", [(0.006, 0.0), (0.0055, 0.002), (0.002, 0.004), (0.0, 0.0045)], segments=8,
                           loc=(hook[0], yc + 0.0005, hook[1]), rot=(90, 0, 0)))

    # ------------------------------------------------------------------ shop bell on a coiled spring bracket
    xbell = hs * (0.15 if double else (ox - lin - 0.16))
    yt = yl - 0.008
    zs = z_t0 + 0.036
    ys = yt - 0.026
    iron.append(_bx("bell_plate", xbell - 0.017, xbell + 0.017, yt, yt - 0.003, z_t0 + 0.006, z_t0 + 0.064, 0.001, 1))
    iron.append(cylinder("bell_arm", 0.0042, 0.034, loc=(xbell, yt - 0.017, zs), rot=(90, 0, 0), verts=8))
    coil = [tuple(c) for c in spiral_path(0.006, 0.027, turns=2.5, n=44, a0_deg=0.0, plane="XZ", center=(xbell, ys, zs))]
    ex, ez_ = coil[-1][0], coil[-1][2]
    bell_top = (ex - 0.004 * (1 if ex < xbell else -1), ys, zs - 0.088)
    path = coil + [((ex + bell_top[0]) / 2 - 0.004 * (1 if ex < xbell else -1), ys, (ez_ + bell_top[2]) / 2), bell_top]
    spring = sweep("bell_spring", path, [(-0.0012, -0.0055), (0.0012, -0.0055), (0.0012, 0.0055), (-0.0012, 0.0055)], up=(0, 1, 0), step=0.016)
    bprof = [(0.0, 0.054), (0.015, 0.051), (0.021, 0.042), (0.0245, 0.028), (0.03, 0.011), (0.0345, 0.001), (0.037, 0.0), (0.0375, 0.003),
             (0.034, 0.012), (0.0285, 0.03), (0.0255, 0.045), (0.019, 0.055), (0.009, 0.0595), (0.0055, 0.062), (0.0065, 0.066), (0.0, 0.068)]
    bell = lathe("bell", bprof, segments=10, loc=(bell_top[0], ys, bell_top[2] - 0.064))
    recalc_normals(bell)
    clap = sphere("bell_clapper", 0.0075, loc=(bell_top[0], ys, bell_top[2] - 0.06), segments=8, rings=5)
    iron += [spring, clap]
    brass.append(bell)
    bell_objs = [spring, bell, clap]

    # ------------------------------------------------------------------ leaded light: quarries and raised cames
    tint_objs = {}
    for tint, polys in quarry.items():
        if polys:
            vs, fs_ = [], []
            for poly in polys:
                n0 = len(vs)
                vs += [(x, yq, z) for x, z in poly]
                fs_.append(tuple(range(n0, n0 + len(poly))))
            tint_objs[tint] = mesh_object("quarries_" + tint, vs, fs_)
    if cames:
        vs, fs_ = [], []
        flat = len(cames) > 240                # very wide leaded fanlights: flat came strips keep the door in budget
        for a, b in cames:
            ln = hypot(b[0] - a[0], b[1] - a[1])
            nx_, nz_ = -(b[1] - a[1]) / ln * 0.0034, (b[0] - a[0]) / ln * 0.0034
            n0 = len(vs)
            for q in (a, b):
                vs += [(q[0] - nx_, yq + 0.0004, q[1] - nz_), (q[0], yq - 0.003, q[1]), (q[0] + nx_, yq + 0.0004, q[1] + nz_)]
            if flat:
                vs[n0:] = [(x_, yq - 0.004, z_) for x_, _, z_ in vs[n0:]]
                fs_.append((n0, n0 + 3, n0 + 5, n0 + 2))
            else:
                fs_ += [(n0, n0 + 3, n0 + 4, n0 + 1), (n0 + 1, n0 + 4, n0 + 5, n0 + 2)]
        cm = mesh_object("cames", vs, fs_)
        smooth(cm, 80)
        lead.append(cm)

    # ------------------------------------------------------------------ materials
    wgain = 0.3 + 0.5 * age + (0.7 if age >= 0.9 else 0.0)      # heavy flaking only on a really neglected door
    m_trim = mat("wood_painted", p.trim, scale=4.0, wear=wgain, grime=0.2, rough=1.5)
    m_door = mat("wood_painted", p.paint, scale=4.5, wear=wgain, rough=1.8)
    m_brass = mat("brass", (1.4, 1.3, 1.12), rough=1.9, wear=0.7, grime=0.12)   # satin, not mirror: a mirror only shows the dark street
    m_iron = mat("iron_wrought")
    m_stone = mat("stone_smooth", "#b3a892")
    if p.lit:
        m_room = mat("glass_frosted", "#ffad45", glow=1.0, opacity=1.0, rough=1.6)     # obscured glass, the brightest thing
        m_shade = mat("glass_frosted", "#b8702a", glow=0.45, opacity=1.0, rough=1.6)   # whatever stands between lamp and glass
    else:
        m_room = mat("window_dark")
        m_shade = mat("canvas", "#a8996f")
    qg = 0.8 if p.lit else 0.06
    tint_m = {"clear": m_room, "amber": m_shade if p.lit else m_room,
              "ruby": mat("glass_frosted", "#8a2a22", glow=qg, opacity=1.0, rough=1.6),
              "green": mat("glass_frosted", "#2f6146", glow=qg, opacity=1.0, rough=1.6)}
    for lst, m in ((trim, m_trim), (door, m_door), (brass, m_brass), (iron, m_iron), (ink, m_iron), (stone, m_stone),
                   (glow, m_room), (shade, m_shade), (card, m_shade), (lead, mat("lead"))):
        for o in lst:
            assign(o, m)
    for tint, o in tint_objs.items():
        assign(o, tint_m[tint])

    # ------------------------------------------------------------------ art-directed weathering
    sd_ = p.seed
    kz_ = z_d0 + 0.985
    chip = noise_mask(0.05, seed=sd_ + 1, lo=0.48, hi=0.7)
    chip2 = noise_mask(0.03, seed=sd_ + 11, lo=0.45, hi=0.7)
    flake = noise_mask(0.09, seed=sd_ + 12, lo=0.5, hi=0.72)
    old = smoothstep(0.85, 1.0, age)
    latch_t = _near((hs * ox, yl, kz_), 0.2) if not double else (lambda co, n: np.zeros(len(co)))

    def trim_wear(co, n):
        # paint stays sound on the shafts; it goes at the plinth blocks (boots, mops, splash-back) and where
        # hands and keys catch the jamb beside the latch. Arris wear is added automatically on export.
        plz = np.clip((zp1 + 0.05 - co[:, 2]) / 0.12, 0.0, 1.0)
        m = np.maximum(plz * chip(co, n) * (0.35 + 0.65 * age), latch_t(co, n) * chip2(co, n) * (0.3 + 0.6 * age))
        return np.maximum(m, old * flake(co, n) * 0.7)

    for o in trim:
        paint(o, grime=mask_mul(height_ramp(sh, sh + 0.5), 0.45), wear=trim_wear,
              special=mask_mul(noise_mask(0.6, seed=sd_ + 2), 0.15 * age))
    for o in trim:
        if o.name.startswith(("cornice", "frieze", "dentils", "console")):
            # soot and damp under the cornice, rain-washed fade on top
            paint(o, grime=mask_mul(facing((0, 0, -1)), 0.7), special=mask_mul(facing((0, 0, 1)), 0.4))
    hands = [_near((v["kx"], yf, v["kz"]), 0.17) for v in info.values()]

    def door_wear(co, n):
        m = np.zeros(len(co))
        for h in hands:
            m = np.maximum(m, h(co, n))
        toe = np.clip((z_d0 + 0.26 - co[:, 2]) / 0.1, 0.0, 1.0) * chip(co, n) * (0.25 + 0.6 * age)
        return np.maximum(np.maximum(m * chip2(co, n) * (0.35 + 0.55 * age), toe), old * flake(co, n) * 0.6)

    for o in door:
        paint(o, grime=mask_mul(height_ramp(z_d0, z_d0 + 0.4), 0.3), wear=door_wear,
              special=mask_mul(noise_mask(0.4, seed=sd_ + 4), height_ramp(z_d0 + 0.9, z_d1), 0.05 + 0.2 * age))
    streak = noise_mask(0.028, seed=sd_ + 5, lo=0.36, hi=0.68)

    def scuff(co, n):                          # boot scuffs: long horizontal smears along the lower third of the kick plate
        band = np.clip((0.52 - (co[:, 2] - z_d0 - 0.028) / 0.19) / 0.3, 0.0, 1.0)
        return band * streak(co * np.array([0.14, 1.0, 1.0]), n)

    for o in brass:
        if o.name.startswith("kick_plate"):
            # an even satin polish from the weekly rub, dulled and smeared along the foot where boots catch it
            paint(o, wear=lambda co, n: 0.28 - 0.2 * scuff(co, n), grime=lambda co, n: scuff(co, n) * (0.25 + 0.35 * age))
        elif o.name.startswith("threshold"):
            paint(o, wear=lambda co, n: 0.25 + 0.4 * np.exp(-((co[:, 0] - hs * 0.05) / (0.3 * ow0)) ** 2), grime=0.1)
        elif o.name.startswith(("knob", "pull_grip", "letter_flap")):
            paint(o, wear=0.4)
        elif o.name.startswith("bell"):
            paint(o, wear=0.3, special=mask_mul(noise_mask(0.03, seed=sd_ + 7), 0.2), grime=0.1)
        else:
            paint(o, wear=0.3)
    for o in iron:
        if o.name.startswith("scraper"):
            paint(o, special=lambda co, n: np.clip(height_ramp(sh, sh + 0.16)(co, n) * 0.5 + 0.1, 0.0, 1.0), wear=mask_mul(facing((0, 0, 1)), 0.8))
        else:
            paint(o, special=mask_mul(noise_mask(0.02, seed=sd_ + 8, lo=0.3, hi=0.7), 0.3 + 0.4 * age))
    ends = lambda co, n: np.clip((np.abs(co[:, 0]) - (W / 2 - 0.2)) / 0.2, 0.0, 1.0)
    back = lambda co, n: np.clip(1.0 - np.abs(co[:, 1] + D + 0.02) / 0.1, 0.0, 1.0)
    tread = lambda co, n: np.exp(-((co[:, 0] - hs * 0.05) / (0.3 * ow0)) ** 2) * np.clip((-co[:, 1] - 0.08) / 0.15, 0.0, 1.0)
    paint(st, wear=mask_mul(facing((0, 0, 1)), tread, 0.25 + 0.4 * age),
          grime=lambda co, n: np.maximum(height_ramp(0.0, sh * 0.8)(co, n) * 0.7, back(co, n) * 0.6),
          special=mask_mul(lambda co, n: np.maximum(ends(co, n), back(co, n) * 0.6), noise_mask(0.1, seed=sd_ + 9, lo=0.35, hi=0.7), 0.4 + 0.5 * age))

    # ------------------------------------------------------------------ moving part, lights, sockets
    part(bell_objs, "bell", pivot=(xbell, ys, zs), motion="swing", amp=2.5, speed=0.8, axis="Y")
    if p.lit:
        # the glazing is the source; the spill is set low and well out so it pools on the step and the paving
        # instead of flooding the pilaster fronts or mirroring in the panes
        xl = (act["xa"] + act["xb"]) * (0.2 if double else 0.5)
        light("door_glass", (xl, -D - 0.85, sh + 0.45), color="#ffc37a", intensity=p.intensity, range=5.0, flicker=0.04)
    socket("sign", "attach", (0.0, 0.0, H + 0.05), role="sign")
    socket("lantern", "attach", (0.0, -D - 0.06, zk1 - 0.02), role="lantern")
