"""Projecting timber oriel bay. A framed window band (corner posts, mullions, sill / transom / head rails
pegged together) stands on a moulded sill over a panelled or half-timbered apron and a moulded bressummer.
The bay is carried either on carved ogee console brackets with wall plates, bolts and turned pendant
drops, or on a solid stepped coved corbel ending in a boss. The lights are leaded (diamond or square
quarries in lead cames) with arched spandrel heads and stained top lights; one iron casement can stand
ajar on its stay. A moulded dentil cornice carries a small swept lead roof with wood-cored rolls or a
hipped plain-tile roof with hip tiles, flashed to the wall in lead.
CONTRACT: origin = bottom centre of the window band on the wall plane (y = 0); the band is `width` wide
and `height` tall (z 0..height); sill, apron and corbelling hang below z = 0, cornice and roof sit above.
"""
from tk import *
import math
import bpy
import numpy as np
from mathutils import Vector, Matrix

META = {"name": "Oriel bay window", "class": "component", "mount": "wall", "mount_height": 3.3,
        "focus": [0.0, -0.3, 0.55], "tri_budget": 16000}
PARAMS = {
    "width": {"default": 1.7, "min": 1.1, "max": 2.8, "desc": "width of the window band on the wall (m)"},
    "height": {"default": 1.5, "min": 1.0, "max": 2.3, "desc": "height of the window band, sill top to cornice (m)"},
    "depth": {"default": 0.5, "min": 0.3, "max": 0.9, "desc": "projection from the wall (m)"},
    "plan": {"default": "canted", "choices": ["canted", "bowed", "box"], "desc": "plan shape of the bay"},
    "roof": {"default": "lead", "choices": ["lead", "tile"], "desc": "swept lead roof with rolls / hipped plain-tile roof"},
    "roof_rise": {"default": 0.7, "min": 0.35, "max": 1.3, "desc": "height of the roof above the cornice (m)"},
    "support": {"default": "brackets", "choices": ["brackets", "corbel"], "desc": "carved console brackets / solid moulded corbel"},
    "apron": {"default": 0.3, "min": 0.0, "max": 0.6, "desc": "height of the panelled apron below the sill (0 = none)"},
    "infill": {"default": "panel", "choices": ["panel", "plaster"], "desc": "apron infill: fielded timber panels / plaster with saltire braces"},
    "light_width": {"default": 0.48, "min": 0.32, "max": 0.7, "desc": "target width of one light between mullions (m)"},
    "leading": {"default": "diamond", "choices": ["diamond", "square"], "desc": "pattern of the lead cames"},
    "quarry": {"default": 0.105, "min": 0.07, "max": 0.16, "desc": "width of one glass quarry (m)"},
    "head": {"default": "arched", "choices": ["arched", "square"], "desc": "arched spandrel boards in the top lights"},
    "stained": {"default": True, "desc": "stained glass in the top lights"},
    "lit": {"default": True, "desc": "room behind is lit (warm glazing + light spill)"},
    "intensity": {"default": 14.0, "min": 4.0, "max": 60.0, "desc": "light spill in front of the glass (cd)"},
    "open_casement": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "how far one iron casement stands open (0 = all fixed lights)"},
    "finish": {"default": "painted", "choices": ["painted", "oak"], "desc": "painted joinery / dark stained oak"},
    "paint": {"default": "#4c7361", "desc": "paint colour of the frame"},
    "trim": {"default": "#a39064", "desc": "colour of panels / plaster / dentils"},
    "accent": {"default": "#ff9a3c", "desc": "stained top-light colour A"},
    "accent2": {"default": "#ff5a48", "desc": "stained top-light colour B (alternates with A)"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "amount of weathering"},
}
VARIANTS = {
    "bowed_tile": {"plan": "bowed", "roof": "tile", "width": 2.0, "depth": 0.6, "height": 1.35, "support": "corbel",
                   "paint": "#6a2320", "roof_rise": 0.8, "leading": "square", "open_casement": 0.0, "seed": 4},
    "box_oak": {"plan": "box", "finish": "oak", "infill": "plaster", "width": 2.1, "height": 1.7, "depth": 0.55, "light_width": 0.52,
                "roof": "tile", "roof_rise": 0.6, "apron": 0.42, "accent": "#7fd08a", "accent2": "#ffb040", "open_casement": 0.6, "seed": 11},
    "tall_narrow": {"width": 1.25, "height": 2.1, "depth": 0.42, "support": "corbel", "roof_rise": 1.1, "apron": 0.2,
                    "paint": "#243a5a", "trim": "#b3a888", "light_width": 0.4, "open_casement": 0.0, "seed": 23},
    "dark_plain": {"plan": "bowed", "lit": False, "stained": False, "head": "square", "width": 1.5, "height": 1.2,
                   "depth": 0.45, "apron": 0.0, "paint": "#8c9a92", "wear": 0.8, "roof_rise": 0.5, "seed": 31},
}


# ----------------------------------------------------------------------------- plan helpers
def _dirs(pts):
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1e-9
        out.append(((dx / L, dy / L), (dy / L, -dx / L), L))
    return out


def _wall_hit(px, py, d):
    if abs(d[1]) < 1e-6:
        return (px, 0.0)
    s = -py / d[1]
    return (px + s * d[0], 0.0)


def _offset(pts, e, s=1.0):
    """Open plan polyline (wall .. front .. wall) scaled about the wall point (0,0) by s and offset
    outwards by e with mitred corners; both ends are returned to the wall plane y = 0."""
    P = [(x * s, y * s) for x, y in pts]
    if abs(e) < 1e-9:
        return P
    D = _dirs(pts)
    n = len(P)
    d, nn, _ = D[0]
    out = [_wall_hit(P[0][0] + nn[0] * e, P[0][1] + nn[1] * e, d)]
    for i in range(1, n - 1):
        n0, n1 = D[i - 1][1], D[i][1]
        k = max(0.3, 1.0 + n0[0] * n1[0] + n0[1] * n1[1])
        out.append((P[i][0] + e * (n0[0] + n1[0]) / k, P[i][1] + e * (n0[1] + n1[1]) / k))
    d, nn, _ = D[-1]
    out.append(_wall_hit(P[-1][0] + nn[0] * e, P[-1][1] + nn[1] * e, d))
    return out


def _dens(poly, step):
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(1, int(math.ceil(L / step)))
        for j in range(1, k + 1):
            f = j / k
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out


def _poly_eval(poly, s):
    """Point and unit direction at arc length s along a 2D polyline."""
    acc = 0.0
    for a, b in zip(poly[:-1], poly[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if s <= acc + L or b is poly[-1]:
            f = (s - acc) / max(L, 1e-9)
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), ((b[0] - a[0]) / max(L, 1e-9), (b[1] - a[1]) / max(L, 1e-9))
        acc += L
    return poly[-1], (1.0, 0.0)


def _plen(poly):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly[:-1], poly[1:]))


def ploft(name, pts, rings):
    """Loft a moulding / shell round the plan: rings = [(scale, offset, z), ...] ordered like a lathe
    profile (bottom to top along the visible surface). scale 0 + offset 0 = a single pole vertex.
    UVs: U = arc length round the plan (grain follows the moulding), V = along the profile."""
    n = len(pts)
    depth = max(-y for x, y in pts)
    verts, uvs, rows = [], [], []
    vacc, prev = 0.0, None
    for (s, e, z) in rings:
        if prev is not None:
            vacc += math.hypot((s - prev[0]) * depth + (e - prev[1]), z - prev[2])
        prev = (s, e, z)
        if s <= 1e-6 and abs(e) < 1e-6:
            rows.append([len(verts)])
            verts.append((0.0, 0.0, z))
            uvs.append((0.0, vacc))
            continue
        R = _offset(pts, e, s)
        L = [0.0]
        for i in range(1, n):
            L.append(L[-1] + math.hypot(R[i][0] - R[i - 1][0], R[i][1] - R[i - 1][1]))
        row = []
        for i in range(n):
            row.append(len(verts))
            verts.append((R[i][0], min(R[i][1], 0.0), z))
            uvs.append((L[i] - L[-1] / 2, vacc))
        rows.append(row)
    faces = []
    for a, b in zip(rows[:-1], rows[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        for j in range(n - 1):
            if len(a) == 1:
                faces.append((a[0], b[j + 1], b[j]))
            elif len(b) == 1:
                faces.append((a[j], a[j + 1], b[0]))
            else:
                faces.append((a[j], a[j + 1], b[j + 1], b[j]))
    o = mesh_object(name, verts, faces)
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    uv_keep(o)
    smooth(o, 38)
    return o


class MB:
    """Tiny quad-soup builder working through a local->world transform."""

    def __init__(self):
        self.v, self.f = [], []

    def quad(self, a, b, c, d):
        i = len(self.v)
        self.v += [a, b, c, d]
        self.f.append((i, i + 1, i + 2, i + 3))

    def box(self, xf, u0, u1, y0, y1, z0, z1):
        c = lambda u, y, z: xf(u, y, z)
        self.quad(c(u0, y0, z0), c(u1, y0, z0), c(u1, y0, z1), c(u0, y0, z1))
        self.quad(c(u1, y1, z0), c(u0, y1, z0), c(u0, y1, z1), c(u1, y1, z1))
        self.quad(c(u0, y1, z0), c(u0, y0, z0), c(u0, y0, z1), c(u0, y1, z1))
        self.quad(c(u1, y0, z0), c(u1, y1, z0), c(u1, y1, z1), c(u1, y0, z1))
        self.quad(c(u0, y0, z1), c(u1, y0, z1), c(u1, y1, z1), c(u0, y1, z1))
        self.quad(c(u0, y1, z0), c(u1, y1, z0), c(u1, y0, z0), c(u0, y0, z0))

    def came(self, xf, A, B, yg, cw=0.009, ch=0.004):
        """Lead came: flat trapezoid strip lying on the glass plane (y = yg) from A to B (u, z)."""
        tu, tz = B[0] - A[0], B[1] - A[1]
        L = math.hypot(tu, tz)
        if L < 1e-4:
            return
        tu, tz = tu / L, tz / L
        qu, qz = -tz, tu
        b, t = cw / 2, cw * 0.32
        yt = yg - ch
        P = lambda p, k, y: xf(p[0] + qu * k, y, p[1] + qz * k)
        self.quad(P(A, -t, yt), P(B, -t, yt), P(B, t, yt), P(A, t, yt))
        self.quad(P(A, -b, yg), P(B, -b, yg), P(B, -t, yt), P(A, -t, yt))
        self.quad(P(B, b, yg), P(A, b, yg), P(A, t, yt), P(B, t, yt))

    def obj(self, name):
        return mesh_object(name, self.v, self.f) if self.f else None


def _streaks(co, nrm):
    u = co[:, 0] * 1.0 + co[:, 1] * 0.73
    v = np.sin(u * 43.0) + np.sin(u * 17.3 + 1.3) + np.sin(u * 71.1 + 0.5)
    return np.clip(v / 3.0 * 0.5 + 0.5, 0.0, 1.0) ** 1.5


def _band(z0, z1, soft=0.04):
    def fn(co, nrm):
        z = co[:, 2]
        return np.clip((z - z0) / soft, 0, 1) * np.clip((z1 - z) / soft, 0, 1)
    return fn


def build(p, rng):
    W, H = p.width, p.height
    D = min(p.depth, 0.48 * W)
    plan = p.plan
    if plan == "bowed":
        D = max(D, 0.2 * W)
    lit = bool(p.lit)
    wr = p.wear

    # ---- materials
    if p.finish == "oak":
        m_frame = mat("wood_oak", "#5d4230")
        m_trim = mat("wood_oak", "#8b6b49")
    else:
        m_frame = mat("wood_painted", p.paint)
        m_trim = mat("wood_painted", p.trim)
    m_infill = mat("plaster_lime", p.trim) if p.infill == "plaster" else m_trim
    m_lead = mat("lead")
    m_iron = mat("iron_wrought")
    m_glass = mat("window_lit", "#ffc37a", glow=3.0, curtain=0.3, rough=2.5) if lit else mat("window_dark")
    if lit:
        m_st = [mat("window_lit", p.accent, glow=2.6, curtain=0.0, rough=2.5), mat("window_lit", p.accent2, glow=2.6, curtain=0.0, rough=2.5)]
    else:
        m_st = [m_glass, m_glass]
    m_leaf = mat("glass_clear", "#d5e4d6", opacity=0.14, rough=2.0)
    G = {}                                   # material group -> objects

    def add(group, o, m):
        assign(o, m)
        G.setdefault(group, []).append(o)
        return o

    # ---- plan: `arc` carries mouldings / roof / corbel, `st` are the mullion stations (flat lights between)
    if plan == "bowed":
        Rb = (W * W / 4 + D * D) / (2 * D)
        th = math.asin(min(1.0, (W / 2) / Rb))
        cyb = Rb - D

        def arc_pt(f):
            a = -th + 2 * th * f
            return (Rb * math.sin(a), min(0.0, cyb - Rb * math.cos(a)))
        NA = 20
        arc = [arc_pt(i / NA) for i in range(NA + 1)]
        arc[0], arc[-1] = (-W / 2, 0.0), (W / 2, 0.0)
        nbay = max(3, int(round(2 * th * Rb / p.light_width)))
        st = [arc_pt(i / nbay) for i in range(nbay + 1)]
        st[0], st[-1] = arc[0], arc[-1]
        kinds = ["wall"] + ["mid"] * (nbay - 1) + ["wall"]
        corners = arc
        arcd = arc

        def plan_depth(x):
            return math.sqrt(max(Rb * Rb - x * x, 0.0)) - cyb
    else:
        inset = min(0.62 * D, 0.3 * W) if plan == "canted" else 0.0
        corners = [(-W / 2, 0.0), (-W / 2 + inset, -D), (W / 2 - inset, -D), (W / 2, 0.0)]
        st, kinds = [corners[0]], ["wall"]
        for i in range(3):
            a, b = corners[i], corners[i + 1]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            k = max(1, int(round(L / p.light_width)))
            for j in range(1, k + 1):
                f = j / k
                st.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
                kinds.append("mid" if j < k else "corner")
        kinds[-1] = "wall"
        nbay = len(st) - 1
        arc = corners
        arcd = _dens(corners, 0.16)

        def plan_depth(x):
            ax = abs(x)
            if ax <= W / 2 - inset or inset < 1e-6:
                return D
            return D * (W / 2 - ax) / inset
    ns = len(st)
    sdirs = _dirs(st)

    # ---- vertical layout
    SILL = 0.085                              # sill moulding below z = 0
    A = p.apron if p.apron >= 0.12 else 0.0
    BH = 0.14                                 # bressummer
    za0 = -SILL                               # apron top
    za1 = za0 - A                             # apron bottom = bressummer top
    zb = za1 - BH                             # bressummer bottom
    transom = H >= 1.25
    zt = H * 0.71
    T_POST, E_POST = 0.085, 0.008
    Y_RAIL, T_RAIL = 0.006, 0.07
    Y_GLASS = 0.036

    # ---- posts: wall half-posts, mitred corner posts, mullions
    margins = [0.0] * ns

    def post_poly(i, a, t, e):
        P = st[i]
        if i == 0 or i == ns - 1:
            d, n, _ = sdirs[0] if i == 0 else sdirs[-1]
            sg = 1.0 if i == 0 else -1.0
            wo = _wall_hit(P[0] + n[0] * e, P[1] + n[1] * e, d)
            wi = _wall_hit(P[0] - n[0] * t, P[1] - n[1] * t, d)
            si = sg * ((wi[0] - P[0]) * d[0] + (wi[1] - P[1]) * d[1])
            aa = max(a, si + 0.035)
            bo = (P[0] + sg * d[0] * aa + n[0] * e, P[1] + sg * d[1] * aa + n[1] * e)
            bi = (P[0] + sg * d[0] * aa - n[0] * t, P[1] + sg * d[1] * aa - n[1] * t)
            return [wo, bo, bi, wi], aa
        (d0, n0, _), (d1, n1, _) = sdirs[i - 1], sdirs[i]
        cs = clamp(n0[0] * n1[0] + n0[1] * n1[1], -1.0, 1.0)
        turn = math.acos(cs)
        aa = max(a, t * math.tan(turn / 2) + 0.016)
        k = max(0.3, 1.0 + cs)
        mo = (P[0] + e * (n0[0] + n1[0]) / k, P[1] + e * (n0[1] + n1[1]) / k)
        mi = (P[0] - t * (n0[0] + n1[0]) / k, P[1] - t * (n0[1] + n1[1]) / k)
        ao = (P[0] - aa * d0[0] + e * n0[0], P[1] - aa * d0[1] + e * n0[1])
        ai = (P[0] - aa * d0[0] - t * n0[0], P[1] - aa * d0[1] - t * n0[1])
        bo = (P[0] + aa * d1[0] + e * n1[0], P[1] + aa * d1[1] + e * n1[1])
        bi = (P[0] + aa * d1[0] - t * n1[0], P[1] + aa * d1[1] - t * n1[1])
        if turn < 0.02:
            return [ao, bo, bi, ai], aa
        return [ao, mo, bo, bi, mi, ai], aa

    for i in range(ns):
        a = {"wall": 0.075, "corner": 0.07, "mid": 0.03}[kinds[i]]
        poly, aa = post_poly(i, a, T_POST, E_POST)
        margins[i] = aa
        poly = [(x, min(y, 0.0)) for x, y in poly]
        po = extrude_profile(f"post{i}", poly, H + 0.004, loc=(0, 0, -0.002), bevel=0.005, segments=1)
        uv_grain(po, axis="Z")
        add("frame", po, m_frame)
        if A > 0:
            poly2, _ = post_poly(i, aa, 0.05, 0.0)
            poly2 = [(x, min(y, 0.0)) for x, y in poly2]
            so = extrude_profile(f"stud{i}", poly2, A + 0.01, loc=(0, 0, za1 - 0.005), bevel=0.004, segments=1)
            uv_grain(so, axis="Z")
            add("frame", so, m_frame)

    # ---- bays: rails, glass, cames, spandrel heads, apron panels
    g_lit, g_st0, g_st1, g_leaf = MB(), MB(), MB(), MB()
    cames, iron = MB(), MB()
    peg_m = []
    r_case = rng.fork("casement")
    open_bay = r_case.pick(list(range(nbay))) if p.open_casement > 0.02 else -1
    theta = radians(p.open_casement * 42.0)
    r_bay = rng.fork("bays")

    def lattice(xf, u0, u1, z0, z1, yg):
        w, h = u1 - u0, z1 - z0
        if w < 0.06 or h < 0.06:
            return
        nx = max(2, int(round(w / p.quarry)))
        px = w / nx
        if p.leading == "diamond":
            nz = max(2, int(round(h / (px * 1.55))))
            pz = h / nz
            for k in range(1, nx + nz):
                xa, xb = max(0, k - nz) * px, min(nx, k) * px
                cames.came(xf, (u0 + xa, z0 + (k - xa / px) * pz), (u0 + xb, z0 + (k - xb / px) * pz), yg)
            for m in range(-(nz - 1), nx):
                xa, xb = max(0, m) * px, min(nx, m + nz) * px
                cames.came(xf, (u0 + xa, z0 + (xa / px - m) * pz), (u0 + xb, z0 + (xb / px - m) * pz), yg, ch=0.0046)
        else:
            nz = max(2, int(round(h / (px * 1.35))))
            pz = h / nz
            for k in range(1, nx):
                cames.came(xf, (u0 + k * px, z0), (u0 + k * px, z1), yg)
            for k in range(1, nz):
                cames.came(xf, (u0, z0 + k * pz), (u1, z0 + k * pz), yg, ch=0.0046)

    for i in range(nbay):
        P = st[i]
        d, n, Lc = sdirs[i]
        adeg = degrees(math.atan2(d[1], d[0]))
        ml, mr = margins[i], margins[i + 1]
        u0, u1 = ml - 0.004, Lc - mr + 0.004
        w = u1 - u0

        def xf(u, y, z, P=P, d=d, n=n):
            return (P[0] + d[0] * u - n[0] * y, P[1] + d[1] * u - n[1] * y, z)

        def rail(nm, z0, z1, yoff=Y_RAIL, thick=T_RAIL):
            uc = (u0 + u1) / 2
            c = xf(uc, yoff + thick / 2, (z0 + z1) / 2)
            o = box(nm, (w + 0.012, thick, z1 - z0), loc=c, rot=(0, 0, adeg), bevel=0.004, segments=1)
            uv_grain(o, axis="X")
            add("frame", o, m_frame)
            for uu in (u0 + 0.028, u1 - 0.028):          # oak pegs through the tenons
                c2 = xf(uu, yoff - 0.0015, (z0 + z1) / 2 + r_bay.jitter(0.0, 0.004))
                peg_m.append(trs(c2, (90, 0, adeg)))
        rail(f"sillrail{i}", 0.0, 0.065)
        rail(f"headrail{i}", H - 0.075, H)
        lights = []
        if transom:
            rail(f"transom{i}", zt - 0.03, zt + 0.03, yoff=0.0, thick=0.078)
            lights = [(0.065, zt - 0.03, False), (zt + 0.03, H - 0.075, True)]
        else:
            lights = [(0.065, H - 0.075, True)]
        for (z0, z1, top) in lights:
            is_open = (i == open_bay and z0 < 0.1)
            if is_open:
                # interior seen through the opening + an iron casement leaf standing ajar on its stay
                g_lit.quad(xf(u0, 0.078, z0), xf(u1, 0.078, z0), xf(u1, 0.078, z1), xf(u0, 0.078, z1))
                c, s = math.cos(theta), math.sin(theta)
                yh = 0.016

                def lf(u, y, z, xf=xf, u0=u0, c=c, s=s, yh=yh):
                    du, dy = u - u0, y - yh
                    return xf(u0 + du * c + dy * s, yh - du * s + dy * c, z)
                fw = 0.02
                a0, a1, b0, b1 = u0 + 0.004, u1 - 0.004, z0 + 0.004, z1 - 0.004
                iron.box(lf, a0, a0 + fw, 0.008, 0.026, b0, b1)
                iron.box(lf, a1 - fw, a1, 0.008, 0.026, b0, b1)
                iron.box(lf, a0 + fw, a1 - fw, 0.008, 0.026, b0, b0 + fw)
                iron.box(lf, a0 + fw, a1 - fw, 0.008, 0.026, b1 - fw, b1)
                iron.box(lf, a1 - 0.016, a1 - 0.004, -0.012, 0.008, (b0 + b1) / 2 - 0.035, (b0 + b1) / 2 + 0.035)   # handle
                yg = 0.02
                g_leaf.quad(lf(a0 + fw, yg, b0 + fw), lf(a1 - fw, yg, b0 + fw), lf(a1 - fw, yg, b1 - fw), lf(a0 + fw, yg, b1 - fw))
                # CLEARANCE RULE: the two sheets of the leaf's glass 4 mm apart (they were 2: a host that lights this glass got
                # two glowing sheets fighting each other)
                g_leaf.quad(lf(a1 - fw, yg + 0.004, b0 + fw), lf(a0 + fw, yg + 0.004, b0 + fw), lf(a0 + fw, yg + 0.004, b1 - fw), lf(a1 - fw, yg + 0.004, b1 - fw))
                lattice(lf, a0 + fw, a1 - fw, b0 + fw, b1 - fw, yg)
                # stay bar and hinge pintles
                sa = Vector(xf(u0 + w * 0.2, 0.03, z0 + 0.008))
                sb = Vector(lf(u0 + w * 0.72, 0.02, b0 + 0.01))
                add("iron", sweep(f"stay{i}", [sa, sb], 0.0045, sides=6, smooth_path=False), m_iron)
                for hz in (z0 + 0.14, z1 - 0.14):
                    add("iron", cylinder(f"pintle{i}", 0.008, 0.055, loc=xf(u0 + 0.002, yh, hz), verts=8), m_iron)
                continue
            gm = g_lit
            if top and transom and p.stained:
                gm = g_st0 if i % 2 == 0 else g_st1
            gm.quad(xf(u0, Y_GLASS, z0), xf(u1, Y_GLASS, z0), xf(u1, Y_GLASS, z1), xf(u0, Y_GLASS, z1))
            lattice(xf, u0, u1, z0, z1, Y_GLASS)
            if top and p.head == "arched":
                rise = min(0.36 * w, (z1 - z0) * 0.62)
                hb = rise + 0.03
                cx = (rise * rise - w * w / 4) / w
                Rr = w / 2 + cx
                fa = math.atan2(rise, -cx)
                pts = [(w / 2, 0.0), (w / 2, hb), (-w / 2, hb), (-w / 2, 0.0)]
                na = 6
                for k in range(1, na + 1):
                    f = math.pi + (fa - math.pi) * k / na
                    pts.append((cx + Rr * math.cos(f), Rr * math.sin(f)))
                for k in range(na - 1, 0, -1):
                    f = math.pi + (fa - math.pi) * k / na
                    pts.append((-(cx + Rr * math.cos(f)), Rr * math.sin(f)))
                c = xf((u0 + u1) / 2, 0.03, z1 - hb + 0.002)
                o = extrude_profile(f"spandrel{i}", pts, 0.02, loc=c, rot=(90, 0, adeg), bevel=0.003, segments=1)
                uv_grain(o, axis="X")
                add("frame", o, m_frame)
        # apron panel
        if A > 0:
            pw, phh = w - 0.012, A - 0.05
            zc = (za0 + za1) / 2
            if p.infill == "panel":
                if pw > 0.1 and phh > 0.06:
                    o = box(f"field{i}", (pw - 0.05, 0.022, phh - 0.02), loc=xf((u0 + u1) / 2, 0.015, zc), rot=(0, 0, adeg),
                            bevel=0.009, segments=1)
                    uv_grain(o, axis="Z")
                    add("trim", o, m_trim)
            else:
                Lb = math.hypot(pw, phh) * 0.9
                phi = degrees(math.atan2(phh, pw))
                for sg in (-1, 1):
                    o = box(f"brace{i}", (Lb, 0.02, 0.05), loc=xf((u0 + u1) / 2, 0.012 + 0.002 * sg, zc), rot=(0, sg * phi, adeg),
                            bevel=0.004, segments=1)
                    uv_grain(o, axis="X")
                    add("frame", o, m_frame)

    for mb, nm, m in ((g_lit, "glass", m_glass), (g_st0, "stained_a", m_st[0]), (g_st1, "stained_b", m_st[1]),
                      (g_leaf, "leaf_glass", m_leaf)):
        o = mb.obj(nm)
        if o:
            add(nm, o, m)
    o = cames.obj("cames")
    if o:
        add("lead_cames", o, m_lead)
    o = iron.obj("casement_iron")
    if o:
        bevel(o, 0.0015, segments=1)
        add("iron", o, m_iron)
    if peg_m:
        pg = cylinder("peg", 0.0065, 0.007, verts=6)
        add("frame", instances_to_mesh("pegs", pg, peg_m, keep_base=False), m_frame)

    # ---- apron wall
    if A > 0:
        ap = ploft("apron_wall", _dens(st, 0.14), [(1, -0.016, za1 - 0.01), (1, -0.016, za1 + A * 0.33), (1, -0.016, za1 + A * 0.66), (1, -0.016, za0 + 0.01)])
        if p.infill == "plaster":
            uv_box(ap)
        add("infill", ap, m_infill)

    # ---- moulded sill (weathered top, drip nose, cove)
    sill = ploft("sill", arcd, [(1, -0.014, -SILL - 0.004), (1, 0.014, -SILL), (1, 0.02, -0.074), (1, 0.027, -0.062), (1, 0.042, -0.052),
                                (1, 0.062, -0.048), (1, 0.066, -0.052), (1, 0.074, -0.052), (1, 0.08, -0.044), (1, 0.08, -0.022),
                                (1, 0.074, -0.012), (1, 0.03, 0.0), (1, 0.004, 0.005), (1, -0.004, 0.005)])
    add("frame", sill, m_frame)

    # ---- bressummer: moulded base beam of the bay
    brs = ploft("bressummer", arcd, [(1, -0.075, zb + 0.03), (1, -0.075, zb), (1, 0.02, zb), (1, 0.03, zb + 0.01), (1, 0.03, zb + 0.04),
                                     (1, 0.036, zb + 0.046), (1, 0.039, zb + 0.055), (1, 0.036, zb + 0.064), (1, 0.03, zb + 0.07),
                                     (1, 0.03, zb + 0.108), (1, 0.004, zb + BH - 0.004), (1, -0.02, zb + BH)])
    add("frame", brs, m_frame)
    ros = lathe("rosette", [(0.0, 0.0), (0.03, 0.0), (0.033, 0.004), (0.024, 0.009), (0.016, 0.008), (0.012, 0.014), (0.0, 0.018)], segments=8)
    rm = []
    for i in range(nbay):
        d, n, Lc = sdirs[i]
        for f in ((0.5,) if Lc < 0.75 else (0.28, 0.72)):
            c = (st[i][0] + d[0] * Lc * f, st[i][1] + d[1] * Lc * f)
            if plan == "bowed":
                rr_ = math.hypot(c[0], c[1] - cyb)
                c = (c[0] * Rb / rr_, cyb + (c[1] - cyb) * Rb / rr_)
            rm.append(trs((c[0] + n[0] * 0.03, c[1] + n[1] * 0.03, zb + 0.089), (90, 0, degrees(math.atan2(d[1], d[0])))))
    add("trim", instances_to_mesh("rosettes", ros, rm, keep_base=False), m_trim)

    # ---- cornice with frieze, dentils, corona and cyma
    zc0 = H
    cor = ploft("cornice", arcd, [(1, -0.004, zc0 - 0.012), (1, 0.016, zc0 - 0.008), (1, 0.016, zc0 + 0.04), (1, 0.03, zc0 + 0.046),
                                  (1, 0.038, zc0 + 0.052), (1, 0.05, zc0 + 0.064), (1, 0.07, zc0 + 0.072), (1, 0.094, zc0 + 0.076),
                                  (1, 0.098, zc0 + 0.082), (1, 0.098, zc0 + 0.104), (1, 0.104, zc0 + 0.11), (1, 0.118, zc0 + 0.124),
                                  (1, 0.122, zc0 + 0.134), (1, 0.122, zc0 + 0.142), (1, 0.06, zc0 + 0.148)])
    add("frame", cor, m_frame)
    den = box("dentil", (0.032, 0.022, 0.03))
    dm = []
    dpath = _offset(arc, 0.016)
    groups = [dpath] if plan == "bowed" else [[dpath[k], dpath[k + 1]] for k in range(3)]
    for g in groups:
        L = _plen(g)
        k = max(1, int(L / 0.064))
        for j in range(k):
            s = L * (j + 0.5) / k
            pt, dd = _poly_eval(g, s)
            nn = (dd[1], -dd[0])
            dm.append(trs((pt[0] + nn[0] * 0.009, min(pt[1] + nn[1] * 0.009, -0.012), zc0 + 0.02), (0, 0, degrees(math.atan2(dd[1], dd[0])))))
    add("trim", instances_to_mesh("dentils", den, dm, keep_base=False), m_trim)

    # ---- roof
    rise = p.roof_rise
    ze = H + 0.154
    EO = 0.15
    za = ze + rise
    tile = p.roof == "tile"
    if tile:
        zr = lambda t: ze + rise * (0.86 * t + 0.14 * t * t)
    else:
        zr = lambda t: ze + rise * (0.38 * t + 0.62 * t * t)
    NR = 9
    rr = [(1, 0.09, ze - 0.012), (1, EO - 0.004, ze - 0.012), (1, EO, ze - 0.007), (1, EO, ze)]
    rr += [(1 - k / NR, EO * (1 - k / NR), zr(k / NR)) for k in range(1, NR)] + [(0, 0, za)]
    roof = ploft("roof_shell", arcd, rr)
    add("lead", roof, m_lead)
    C = _offset(arc, EO)                           # eave line
    xe = C[-1][0]
    rgroups = [C] if plan == "bowed" else [[C[k], C[k + 1]] for k in range(3)]
    r_roof = rng.fork("roof")

    def st_out(i, e):
        P = st[i]
        (d0, n0, _), (d1, n1, _) = sdirs[i - 1], sdirs[i]
        k = max(0.3, 1.0 + n0[0] * n1[0] + n0[1] * n1[1])
        return (P[0] + e * (n0[0] + n1[0]) / k, P[1] + e * (n0[1] + n1[1]) / k)

    if not tile:
        # wood-cored lead rolls over the hips / bay joints, dressed down over the eave
        for i in range(1, ns - 1):
            ex = st_out(i, EO)
            t1 = 0.93 if kinds[i] == "corner" or plan == "bowed" else 0.72
            path = [(ex[0] * 1.004, ex[1] * 1.004, ze - 0.016)]
            path += [(ex[0] * (1 - t), ex[1] * (1 - t), zr(t) + 0.008) for t in np.linspace(0.0, t1, 9)]
            ro = sweep(f"roll{i}", path, 0.019, sides=7, smooth_path=False, scale_fn=lambda t: 1.0 - 0.25 * t)
            add("lead", ro, m_lead)
    else:
        # plain clay tiles laid in courses on every roof facet (broken bond, cut at hips and wall)
        m_tile = mat("terracotta", "#b5694a")
        tw, th_t = 0.168, 0.012
        tv = [(-tw / 2, 0, 0.026), (0, 0, 0.031), (tw / 2, 0, 0.026), (-tw / 2, 1, 0.001), (0, 1, 0.004), (tw / 2, 1, 0.001),
              (-tw / 2, 0, 0.026 - th_t), (0, 0, 0.031 - th_t), (tw / 2, 0, 0.026 - th_t)]
        tf = [(0, 1, 4, 3), (1, 2, 5, 4), (6, 7, 1, 0), (7, 8, 2, 1), (6, 0, 3), (2, 8, 5)]
        tb = mesh_object("tile", tv, tf)
        uv_planar(tb, (1, 0, 0), (0, 0.25, 0))
        assign(tb, m_tile)
        apex = Vector((0, 0, za))
        smax = max(math.hypot(_plen(g) * 0 + max(-min(q[1] for q in g), 0.1), rise) for g in rgroups)
        nc = max(4, int(math.ceil(smax / 0.118)))
        for gi, g in enumerate(rgroups):
            tm = []
            Lg = _plen(g)
            for k in range(nc):
                t0 = k / nc
                sc = 1 - t0
                Lk = Lg * sc
                if t0 > 0.86 or Lk < 0.4 * tw:
                    continue
                t1 = min(1.0, t0 + 1.0 / nc)
                offs = 0.5 if k % 2 else 0.0
                i0 = int(Lk / 2 / tw) + 2
                for j in range(-i0, i0 + 1):
                    s_c = Lk / 2 + (j + offs) * tw
                    lo, hi = max(0.0, s_c - tw / 2), min(Lk, s_c + tw / 2)
                    if hi - lo < 0.03:
                        continue
                    sm = (lo + hi) / 2
                    pt, dd = _poly_eval(g, sm / sc)
                    base = Vector((pt[0] * sc, pt[1] * sc, zr(t0)))
                    nxt = Vector((pt[0] * (1 - t1), pt[1] * (1 - t1), zr(t1)))
                    dv = Vector((dd[0], dd[1], 0.0))
                    nrm = dv.cross(nxt - base)
                    if nrm.length < 1e-6:
                        continue
                    nrm.normalize()
                    up = nrm.cross(dv).normalized()
                    gexp = (nxt - base).dot(up)
                    Lt = gexp * 2.15
                    ang = radians(r_roof.jitter(0.0, 1.6))
                    d2 = dv * math.cos(ang) + up * math.sin(ang)
                    u2 = up * math.cos(ang) - dv * math.sin(ang)
                    sx = (hi - lo) / tw * 0.985
                    pos = base + up * r_roof.jitter(-0.012, 0.005) + nrm * r_roof.u(0.0, 0.003)
                    M = Matrix(((d2.x * sx, u2.x * Lt, nrm.x, pos.x), (d2.y * sx, u2.y * Lt, nrm.y, pos.y),
                                (d2.z * sx, u2.z * Lt, nrm.z, pos.z), (0, 0, 0, 1)))
                    tm.append(M)
            if not tm:
                continue
            tiles = instances_to_mesh(f"tiles{gi}", tb, tm, keep_base=True)
            if plan != "bowed":
                # tiles are cut to the mitre at the hips: slide stray corners back along the course
                (ax_, ay_), (bx_, by_) = g[0], g[1]
                det = ax_ * by_ - ay_ * bx_
                gl = math.hypot(bx_ - ax_, by_ - ay_)
                ex_, ey_ = (bx_ - ax_) / gl, (by_ - ay_) / gl
                da, db = (ex_ * by_ - ey_ * bx_) / det, (ax_ * ey_ - ay_ * ex_) / det

                def cut(co, det=det, ax_=ax_, ay_=ay_, bx_=bx_, by_=by_, ex_=ex_, ey_=ey_, da=da, db=db):
                    a = (co.x * by_ - co.y * bx_) / det
                    b = (ax_ * co.y - ay_ * co.x) / det
                    sft = 0.0
                    if a < 0 and abs(da) > 1e-9:
                        sft = -a / da
                    elif b < 0 and abs(db) > 1e-9:
                        sft = -b / db
                    return Vector((co.x + ex_ * sft, co.y + ey_ * sft, co.z))
                transform_verts(tiles, cut)
            transform_verts(tiles, lambda co: Vector((co.x, min(co.y, -0.004), co.z)))
            add("tile", tiles, m_tile)
        bpy.data.objects.remove(tb, do_unlink=True)
        if plan != "bowed":
            # half-round hip tiles bedded over the two hips
            hr = 0.058
            hv, hf = [], []
            nsg = 5
            for (yy, rs) in ((0.0, 1.06), (1.0, 0.94), (0.0, 0.84)):
                for q in range(nsg + 1):
                    a = math.pi * q / nsg
                    hv.append((-math.cos(a) * hr * rs, yy, math.sin(a) * hr * rs * 0.85))
            for q in range(nsg):
                hf.append((q, q + 1, nsg + 1 + q + 1, nsg + 1 + q))
                hf.append((2 * (nsg + 1) + q, 2 * (nsg + 1) + q + 1, q + 1, q))
            hbase = mesh_object("hiptile", hv, hf)
            uv_planar(hbase, (1, 0, 0), (0, 0.3, 0))
            assign(hbase, m_tile)
            smooth(hbase, 50)
            hm = []
            for ci in (1, 2):
                cpt = C[ci]
                hl = math.hypot(math.hypot(cpt[0], cpt[1]), rise)
                nh = max(2, int(math.ceil(hl * 0.86 / 0.24)))
                for k in range(nh):
                    t0, t1 = 0.86 * k / nh, 0.86 * (k + 1.25) / nh
                    b0 = Vector((cpt[0] * (1 - t0), cpt[1] * (1 - t0), zr(t0)))
                    b1 = Vector((cpt[0] * (1 - t1), cpt[1] * (1 - t1), zr(t1)))
                    yv = b1 - b0
                    Lh = yv.length
                    yv.normalize()
                    out = Vector((cpt[0], cpt[1], 0)).normalized()
                    xv = yv.cross(Vector((0, 0, 1)))
                    if xv.length < 1e-6:
                        continue
                    xv.normalize()
                    zv = xv.cross(yv).normalized()
                    pos = b0 + zv * (0.012 + r_roof.u(0, 0.004)) - yv * 0.01
                    yv2 = (yv * Lh + zv * (-0.007)).normalized()      # tail lifted over the tile below
                    zv2 = xv.cross(yv2).normalized()
                    hm.append(Matrix(((xv.x, yv2.x * Lh, zv2.x, pos.x), (xv.y, yv2.y * Lh, zv2.y, pos.y),
                                      (xv.z, yv2.z * Lh, zv2.z, pos.z), (0, 0, 0, 1))))
            add("tile", instances_to_mesh("hiptiles", hbase, hm, keep_base=False), m_tile)
        # lead saddle over the top courses
        cap = ploft("apex_saddle", arcd, [(0.2, EO * 0.2 + 0.012, zr(0.8) + 0.02), (0.2, EO * 0.2 + 0.016, zr(0.8) + 0.032),
                                          (0.1, EO * 0.1 + 0.012, zr(0.9) + 0.034), (0, 0, za + 0.03)])
        add("lead", cap, m_lead)

    # lead apex cap with a small spike, and cover flashings dressed up the wall
    semi = [(-math.cos(math.pi * k / 8), -math.sin(math.pi * k / 8)) for k in range(9)]
    semi[0], semi[-1] = (-1.0, 0.0), (1.0, 0.0)
    zk = za + (0.03 if tile else 0.0)
    knob = ploft("apex_cap", semi, [(0.1, 0, zk - 0.11), (0.105, 0, zk - 0.03), (0.07, 0, zk + 0.0), (0.04, 0, zk + 0.03),
                                    (0.05, 0, zk + 0.05), (0.05, 0, zk + 0.065), (0.028, 0, zk + 0.085), (0.012, 0, zk + 0.15), (0, 0, zk + 0.17)])
    add("lead", knob, m_lead)
    for sx in (-1, 1):
        path = [(sx * abs(xe) * (1 - t) * 1.0, -0.0045, zr(t) + (0.035 if tile else 0.02)) for t in np.linspace(0.0, 0.97, 8)]
        fl = sweep(f"flashing{sx}", path, rounded_rect(0.006, 0.1, 0.0015, 1), smooth_path=False, up=(0, 0, 1))
        add("lead", fl, m_lead)

    # ---- support
    pend_prof = [(0.0, -1.0), (0.3, -0.965), (0.62, -0.86), (0.85, -0.7), (0.88, -0.58), (0.7, -0.47), (0.42, -0.42), (0.42, -0.385),
                 (0.78, -0.35), (0.78, -0.31), (0.5, -0.27), (0.5, -0.12), (0.9, -0.07), (1.0, -0.035), (1.0, 0.0)]

    def pendant(nm, loc, r=0.04, L=0.2):
        o = lathe(nm, [(a * r, b * L) for a, b in pend_prof], segments=10, loc=loc)
        add("trim" if p.finish == "painted" else "frame", o, m_trim if p.finish == "painted" else m_frame)
        return o

    zsoff = zb + 0.03
    z_low = zb
    if p.support == "brackets":
        sof = ploft("soffit", arcd, [(0, 0, zsoff), (1, -0.07, zsoff)])
        uv_planar(sof, (1, 0, 0), (0, 1, 0))
        add("soffit", sof, m_frame)
        if plan == "canted":
            xb = max(0.2, W / 2 - inset - 0.03)
        elif plan == "box":
            xb = W / 2 - 0.1
        else:
            xb = 0.3 * W
        xs = [-xb, xb] + ([0.0] if W >= 2.2 else [])
        tb_ = 0.11
        bolt_m = []
        for bi, x in enumerate(xs):
            reach = plan_depth(x) + 0.012
            hb = clamp(reach * 1.3, 0.42, 0.95)
            z_low = min(z_low, zb - hb - 0.05)
            p0 = (-reach, zb - 0.085)
            p1 = (-reach + 0.005, zb - 0.085 - 0.5 * hb)
            p3 = (-0.09, zb - hb + 0.08)
            p2 = (-0.09, zb - hb + 0.08 + 0.48 * hb)
            cur = []
            for k in range(13):
                t = k / 12
                b = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3)
                cur.append((b[0] * p0[0] + b[1] * p1[0] + b[2] * p2[0] + b[3] * p3[0],
                            b[0] * p0[1] + b[1] * p1[1] + b[2] * p2[1] + b[3] * p3[1]))
            outline = [(0.0, zb), (-reach, zb)] + cur + [(-0.105, zb - hb + 0.05), (-0.075, zb - hb), (0.0, zb - hb)]
            bo = extrude_profile(f"bracket{bi}", outline, tb_, loc=(x - tb_ / 2, 0, 0), rot=(90, 0, 90), bevel=0.009, segments=1)
            uv_grain(bo, axis=(-1.0, 0.7, 0.0))
            add("frame", bo, m_frame)
            # raised moulded cheek on both faces (follows the ogee)
            mg = 0.04
            inner = [(-0.035, zb - 0.035), (-reach + mg, zb - 0.035)]
            for k in range(1, 12):
                tx, tz = cur[k + 1][0] - cur[k - 1][0], cur[k + 1][1] - cur[k - 1][1]
                tl = math.hypot(tx, tz) or 1e-9
                inner.append((cur[k][0] - tz / tl * mg, cur[k][1] + tx / tl * mg))
            inner.append((-0.035, zb - hb + 0.1))
            ck = extrude_profile(f"cheek{bi}", inner, tb_ + 0.018, loc=(x - tb_ / 2 - 0.009, 0, 0), rot=(90, 0, 90), bevel=0.006, segments=1)
            uv_grain(ck, axis=(-1.0, 0.7, 0.0))
            add("frame", ck, m_frame)
            # reeded bead down the front edge, volute roll under the nose, scroll at the foot
            bead = sweep(f"bead{bi}", [(x, a - 0.002, b - 0.004) for a, b in cur], 0.015, sides=6)
            add("frame", bead, m_frame)
            add("frame", cylinder(f"volute{bi}", 0.046, tb_ + 0.024, loc=(x, -reach + 0.036, zb - 0.088), rot=(0, 90, 0), verts=14,
                                  bevel=0.008, segments=1), m_frame)
            add("frame", cylinder(f"scroll{bi}", 0.03, tb_ + 0.02, loc=(x, -0.1, zb - hb + 0.062), rot=(0, 90, 0), verts=12,
                                  bevel=0.006, segments=1), m_frame)
            wp = box(f"wallplate{bi}", (tb_ + 0.09, 0.034, hb + 0.14), loc=(x, -0.017, zb - hb / 2 - 0.03), bevel=0.005, segments=1)
            uv_grain(wp, axis="Z")
            add("frame", wp, m_frame)
            for sx in (-1, 1):
                for zz in (zb - 0.1, zb - hb + 0.02):
                    bolt_m.append(trs((x + sx * (tb_ / 2 + 0.023), -0.034, zz), (90, 0, 0)))
            pendant(f"pendant{bi}", (x, -reach + 0.036, zb - 0.128), r=0.036, L=0.17)
            z_low = min(z_low, zb - 0.3)
        bt = bolt("coach_bolt", 0.013, 0.009, kind="square")
        add("iron", instances_to_mesh("bolts", bt, bolt_m, keep_base=False), m_iron)
    else:
        hc = clamp(1.5 * D, 0.5, 1.1)
        zc_ = zb - hc
        z_low = zc_
        rings = [(0, 0, zc_), (0.03, 0.022, zc_ + 0.012), (0.05, 0.05, zc_ + 0.05), (0.06, 0.06, zc_ + 0.09), (0.06, 0.045, zc_ + 0.125),
                 (0.07, 0.02, zc_ + 0.145), (0.07, 0.034, zc_ + 0.155), (0.07, 0.034, zc_ + 0.175)]
        tiers = [(0.07, 0.40, 0.19, 0.48), (0.40, 0.70, 0.48, 0.75), (0.70, 0.97, 0.75, 1.0)]
        for ti, (s0, s1, f0, f1) in enumerate(tiers):
            y0, y1 = zc_ + hc * f0, zc_ + hc * f1 - 0.05
            for k in range(0, 7):
                f = radians(90.0 * k / 6)
                rings.append((s0 + (s1 - s0) * (1 - math.cos(f)), 0.012, y0 + (y1 - y0) * math.sin(f)))
            rings += [(s1, 0.024, y1 + 0.004), (s1, 0.024, y1 + 0.018), (s1, 0.03, y1 + 0.024), (s1, 0.033, y1 + 0.032),
                      (s1, 0.03, y1 + 0.04), (s1, 0.018, y1 + 0.046), (s1, 0.012, y1 + 0.05)]
        rings.append((1.0, 0.0, zb + 0.004))
        cb = ploft("corbel", arcd, rings)
        add("frame", cb, m_frame)
        # pendant drops under the front of the bressummer
        if plan == "bowed":
            pxs = [0.0]
        else:
            pxs = [-(W / 2 - inset) * 0.999, (W / 2 - inset) * 0.999]
        for bi, x in enumerate(pxs):
            dd_ = plan_depth(x)
            pendant(f"pendant{bi}", (x * (1 - 0.02 / max(abs(x), 0.02)) if x else 0.0, -dd_ + (0.005 if plan == "bowed" else 0.012), zb + 0.002), r=0.034, L=0.16)

    # ---- sockets and light
    if lit:
        light("spill", (0.0, -D - 1.15, -0.5), color="#ffc37a", intensity=p.intensity, range=7.0, flicker=0.04,
              shadow_near=D + 0.8 + 0.6 * max(W, H) + 1.2)
    socket("planter", "attach", (0.0, -D - 0.085, -0.03), role="planter")
    socket("bunting", "attach", (0.0, -D - 0.125, H + 0.1), role="banner")

    # ---- art-directed weathering per material group
    seed = p.seed
    for gname, objs in list(G.items()):
        o = join(objs, gname) if len(objs) > 1 else realize(objs[0])
        if gname in ("frame", "trim"):
            under = lambda co, nrm: np.clip(-nrm[:, 2], 0, 1)
            sillz = _band(za1 - 0.02, -0.03, 0.12)
            paint(o,
                  grime=lambda co, nrm: np.clip(0.55 * wr * _streaks(co, nrm) * sillz(co, nrm) + 0.5 * wr * under(co, nrm)
                                                + 0.7 * wr * np.clip((z_low + 0.35 - co[:, 2]) / 0.35, 0, 1)
                                                + 0.5 * wr * _streaks(co, nrm) * _band(H - 0.25, H + 0.02, 0.1)(co, nrm), 0, 1),
                  wear=mask_mul(noise_mask(0.07, seed=seed + 3, lo=0.45, hi=0.75), lambda co, nrm: np.clip(wr * 1.6 * (0.5 + 0.5 * _band(-0.07, 0.03, 0.02)(co, nrm)), 0, 1)),
                  special=mask_mul(noise_mask(0.3, seed=seed + 5, lo=0.3, hi=0.8), lambda co, nrm: wr * np.clip(0.4 + 0.6 * nrm[:, 2], 0, 1)))
        elif gname == "infill":
            paint(o, grime=lambda co, nrm: np.clip(wr * (0.3 + 0.7 * _streaks(co, nrm)) * np.clip((za0 - co[:, 2]) / max(A, 0.1), 0, 1) ** 0.6, 0, 1),
                  special=mask_mul(noise_mask(0.18, seed=seed + 8, lo=0.4, hi=0.8), wr * 0.7),
                  wear=mask_mul(noise_mask(0.1, seed=seed + 9, lo=0.6, hi=0.85), wr * 0.6))
        elif gname == "lead":
            paint(o, grime=lambda co, nrm: np.clip(wr * (0.25 + 0.6 * _streaks(co, nrm)) * np.clip((za - co[:, 2]) / max(rise, 0.1), 0, 1), 0, 1))
        elif gname == "tile":
            paint(o, special=mask_mul(noise_mask(0.22, seed=seed + 13, lo=0.4, hi=0.75), lambda co, nrm: (0.25 + wr) * np.clip((za - co[:, 2]) / max(rise, 0.1), 0.2, 1)),
                  grime=mask_mul(noise_mask(0.1, seed=seed + 14, lo=0.3, hi=0.8), 0.3 + 0.5 * wr))
        elif gname == "iron":
            paint(o, special=mask_mul(noise_mask(0.05, seed=seed + 17, lo=0.35, hi=0.75), 0.25 + 0.6 * wr))
        elif gname == "soffit":
            paint(o, grime=0.35 + 0.3 * wr)
