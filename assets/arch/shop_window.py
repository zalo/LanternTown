"""Traditional timber shopfront bay, built as a joiner would and standing only a pilaster's depth proud of the
wall: two box pilasters on dressed-stone plinths with sunk reeded panels, panelled returns, necking and scrolled
console brackets; a framed stall-riser with fielded panels and bolection mouldings on a stone kerb; a weathered
sill; a display window whose puttied, slightly out-of-true panes are divided by slender turned colonnette
mullions; a moulded transom rail under a row of small lit panes; a tilted fascia with raised gilt lettering,
dentil course, mitred cornice and a lead capping dressed over the edges and turned up the wall. Behind the glass
a shallow display bed with a panelled window-back, a pleated curtain, shelves on brass brackets and a rail of
shaded lamps carries seeded goods (luggage / hats / bottles / books / cakes) with price tickets. Paint ageing is
modelled as crisp flakes, splits and run-off streaks; light sockets throw the window's warm spill onto the pavement.
"""
from tk import *
import math
import numpy as np
import bmesh

META = {"name": "Shopfront display window", "class": "component", "mount": "wall_ground",
        "tri_budget": 22000,            # a whole shopfront bay plus a dressed window display and lettering
        "focus": 0.46}
PARAMS = {
    "width": {"default": 2.6, "min": 1.6, "max": 4.6, "desc": "outer width of the shopfront unit incl. cornice (m)"},
    "height": {"default": 3.1, "min": 2.6, "max": 3.9, "desc": "outer height to the top of the lead flashing (m)"},
    "bays": {"default": 3, "min": 1, "max": 6, "desc": "number of display lights between the turned mullions"},
    "paint": {"default": "#5e1f1c", "desc": "paint colour of the joinery (very dark colours are lifted so they read at night)"},
    "goods": {"default": "luggage", "choices": ["luggage", "hats", "bottles", "books", "cakes"], "desc": "what is on display"},
    "lit": {"default": True, "desc": "shop open: lit display and light spill (False = dark, closed shop)"},
    "depth": {"default": 0.3, "min": 0.22, "max": 0.6, "desc": "projection of the shopfront from the wall (m)"},
    "riser": {"default": 0.62, "min": 0.4, "max": 0.9, "desc": "height of the panelled stall-riser (m)"},
    "transom": {"default": 0.36, "min": 0.0, "max": 0.55, "desc": "height of the transom lights (0 = none)"},
    "arch": {"default": "segmental", "choices": ["flat", "segmental", "round", "pointed"], "desc": "head shape of the lights / transom panes"},
    "fascia": {"default": 0.42, "min": 0.28, "max": 0.6, "desc": "height of the fascia board (m)"},
    "sign_text": {"default": "LUGGAGE", "desc": "raised fascia lettering ('' = blank fascia for a separate sign)"},
    "lettering": {"default": "#d6aa4a", "desc": "signwriter's paint colour of the raised letters (#hex), or 'gilt' = gilded metal leaf"},
    "curtain": {"default": "#d8c6a0", "desc": "colour of the window-back curtain, labels and price tickets"},
    "intensity": {"default": 22.0, "min": 5, "max": 90, "desc": "light spill of the window onto the pavement (cd)"},
    "light_color": {"default": "#ffc37a", "desc": "interior light colour"},
    "glow": {"default": 1.1, "min": 0.2, "max": 5.0, "desc": "glow of the lit curtain, transom panes and lamp bulbs"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "paint ageing: flaked paint, splits, run-off streaks, sun-fade"},
}
VARIANTS = {
    "hatter_narrow": {"width": 1.9, "height": 2.9, "bays": 2, "paint": "#2b4b3c", "goods": "hats", "sign_text": "HATTER",
                      "arch": "round", "transom": 0.42, "riser": 0.7, "depth": 0.36, "curtain": "#cdb48a", "wear": 0.35, "seed": 4},
    "wine_wide": {"width": 3.9, "height": 3.35, "bays": 5, "paint": "#2c4874", "goods": "bottles", "sign_text": "WINES & SPIRITS",
                  "lettering": "gilt", "arch": "pointed", "transom": 0.46, "fascia": 0.48, "riser": 0.72, "wear": 0.55, "seed": 11},
    "bookshop_worn": {"width": 2.3, "height": 3.0, "bays": 2, "paint": "#4d3e31", "goods": "books", "sign_text": "BOOKS",
                      "lettering": "gilt", "arch": "flat", "transom": 0.3, "wear": 0.95, "riser": 0.8, "light_color": "#ffb866", "seed": 23},
    "patisserie": {"width": 3.1, "height": 3.2, "bays": 4, "paint": "#b9c4a5", "goods": "cakes", "sign_text": "PATISSERIE",
                   "lettering": "#33221a", "curtain": "#e2cfc0", "intensity": 20.0,
                   "arch": "round", "transom": 0.34, "riser": 0.46, "depth": 0.36, "wear": 0.2, "seed": 31},
}


# ------------------------------------------------------------------ local geometry helpers
def msweep(name, path, prof, up=(0, 0, 1), closed=False, step=None):
    """Moulding with true mitres: profile (a = outwards = tangent x up, b = along up) along a planar polyline.
    step subdivides the straight runs so vertex weathering masks have resolution."""
    P = []
    for q in path:
        q = Vector(q)
        if not P or (q - P[-1]).length > 1e-5:
            P.append(q)
    if closed and (P[0] - P[-1]).length < 1e-5:
        P.pop()
    if step:
        Q = []
        for i in range(len(P) if closed else len(P) - 1):
            a, b = P[i], P[(i + 1) % len(P)]
            k = max(1, int(math.ceil((b - a).length / step)))
            Q += [a.lerp(b, j / k) for j in range(k)]
        if not closed:
            Q.append(P[-1])
        P = Q
    U = Vector(up).normalized()
    n, m = len(P), len(prof)

    def seg_n(i):
        return (P[(i + 1) % n] - P[i]).normalized().cross(U).normalized()

    verts = []
    for i in range(n):
        if closed:
            n0, n1 = seg_n((i - 1) % n), seg_n(i)
        elif i == 0:
            n0 = n1 = seg_n(0)
        elif i == n - 1:
            n0 = n1 = seg_n(n - 2)
        else:
            n0, n1 = seg_n(i - 1), seg_n(i)
        mv = (n0 + n1) / (1.0 + n0.dot(n1))
        for (a, b) in prof:
            verts.append(tuple(P[i] + mv * a + U * b))
    faces = []
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        for k in range(m):
            k1 = (k + 1) % m
            faces.append((i * m + k, i * m + k1, j * m + k1, j * m + k))
    if not closed:
        faces.append(tuple(range(m)))
        faces.append(tuple((n - 1) * m + k for k in reversed(range(m))))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, 35)
    return o


def prism_x(name, yz, x0, x1, bev=0.0):
    """Closed (y, z) polygon extruded from x0 to x1."""
    m = len(yz)
    verts = [(x0, y, z) for (y, z) in yz] + [(x1, y, z) for (y, z) in yz]
    faces = [(k, (k + 1) % m, m + (k + 1) % m, m + k) for k in range(m)]
    faces.append(tuple(range(m)))
    faces.append(tuple(m + k for k in reversed(range(m))))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    if bev > 0:
        bevel(o, bev, segments=1)
    smooth(o, 35)
    return o


def fielded_panel(name, x0, x1, z0, z1, y, rise=0.014, margin=0.045):
    """Raised-and-fielded joinery panel facing -Y: flat field `rise` proud of the sunk margin."""
    m = min(margin, (x1 - x0) * 0.3, (z1 - z0) * 0.3)
    v = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1),
         (x0 + m, y - rise, z0 + m), (x1 - m, y - rise, z0 + m), (x1 - m, y - rise, z1 - m), (x0 + m, y - rise, z1 - m)]
    f = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    return mesh_object(name, v, f)


def side_field(name, sx, x, yc_, w, z0, z1, rise=0.006):
    """Flat raised field inside a return panel frame, facing +-X."""
    m = min(0.03, w * 0.25)
    y0, y1 = yc_ - w / 2, yc_ + w / 2
    v = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1),
         (x + sx * rise, y0 + m, z0 + m), (x + sx * rise, y1 - m, z0 + m), (x + sx * rise, y1 - m, z1 - m), (x + sx * rise, y0 + m, z1 - m)]
    f = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    o = mesh_object(name, v, f)
    recalc_normals(o)
    return o


def ring_frame(name, C, U, V, N, w, h, m, t):
    """Planted panel frame: rectangle w x h (along U, V) centred on C, members m wide, standing t proud along N."""
    C, U, V, N = Vector(C), Vector(U), Vector(V), Vector(N)
    c = 0.35 * t
    rings = [(w / 2, h / 2, 0.0), (w / 2, h / 2, t), (w / 2 - m + c, h / 2 - m + c, t), (w / 2 - m, h / 2 - m, t - c),
             (w / 2 - m, h / 2 - m, 0.0)]
    verts, faces = [], []
    for (a, b, d) in rings:
        for (su, sv) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(tuple(C + U * (su * a) + V * (sv * b) + N * d))
    for r in range(len(rings) - 1):
        for k in range(4):
            k1 = (k + 1) % 4
            faces.append((r * 4 + k, r * 4 + k1, (r + 1) * 4 + k1, (r + 1) * 4 + k))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def ntris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def night_safe(col, vmin=0.3):
    """Very dark paints vanish under amber lantern light: lift their value (keeping the hue) to a readable minimum."""
    h = str(col).lstrip("#")
    try:
        c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        return col
    v = max(c)
    if v >= vmin:
        return col
    c = [vmin] * 3 if v <= 0.0 else [x * vmin / v for x in c]
    return "#%02x%02x%02x" % tuple(int(round(min(1.0, x) * 255)) for x in c)


def shade(col, k):
    h = str(col).lstrip("#")
    try:
        return "#%02x%02x%02x" % tuple(int(round(clamp(int(h[i:i + 2], 16) * k, 0, 255))) for i in (0, 2, 4))
    except ValueError:
        return col


def paint_value(col):
    h = str(col).lstrip("#")
    try:
        return max(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return 0.5


def colonnette(name, x, y, z0, h, seg=10):
    """Slender turned mullion: square plinth, attic base, vase-turned lower shaft, ring, tapering upper shaft,
    astragal, bell capital and abacus. Returns (painted parts, gilt parts)."""
    zb, zc = 0.11, h - 0.105
    L = zc - zb
    prof = [(0.02, 0.058), (0.026, 0.062), (0.0285, 0.07), (0.026, 0.078), (0.019, 0.082), (0.0185, 0.09), (0.023, 0.096),
            (0.0235, 0.103), (0.017, zb), (0.0205, zb + 0.10 * L), (0.022, zb + 0.19 * L), (0.0185, zb + 0.29 * L),
            (0.014, zb + 0.335 * L), (0.0195, zb + 0.345 * L), (0.021, zb + 0.357 * L), (0.0195, zb + 0.369 * L),
            (0.0145, zb + 0.38 * L), (0.0158, zb + 0.55 * L), (0.015, zb + 0.8 * L), (0.0135, zc - 0.014), (0.018, zc - 0.01),
            (0.018, zc - 0.003), (0.0135, zc)]
    k = 1.25
    pl = box(name + "_plinth", (0.066, 0.066, 0.06), loc=(x, y, z0 + 0.03), bevel=0.003, segments=1)
    sh = lathe(name, [(a * k, b) for (a, b) in prof], segments=seg, loc=(x, y, z0))
    cap = lathe(name + "_cap", [(a * k, b) for (a, b) in [(0.0135, zc), (0.0145, zc + 0.028), (0.02, zc + 0.052), (0.028, zc + 0.068),
                                                         (0.03, zc + 0.078), (0.026, zc + 0.082)]], segments=seg, loc=(x, y, z0))
    ab = box(name + "_abacus", (0.082, 0.07, 0.025), loc=(x, y, z0 + h - 0.0125), bevel=0.003, segments=1)
    return [pl, sh, ab], [cap]


def pane(name, x0, x1, z0, z1, y, r, dirt=1.0, fine=True):
    """One sheet of old glass facing -Y: slightly out of true (seeded tilt), with grime gathered along the putty lines
    and in the lower corners (painted into the vertex grime mask on a grid that is dense near the edges)."""
    def stations(a, b):
        f = (0.0, 0.03, 0.08, 0.2, 0.5, 0.8, 0.92, 0.97, 1.0) if (b - a > 0.45 and fine) else (0.0, 0.04, 0.12, 0.5, 0.88, 0.96, 1.0)
        return [a + (b - a) * t for t in f]
    xs, zz = stations(x0, x1), stations(z0, z1)
    xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
    lim = 0.005
    tx = clamp(math.tan(math.radians(r.jitter(0, 0.5))), -lim / (z1 - zc), lim / (z1 - zc))
    tz = clamp(math.tan(math.radians(r.jitter(0, 0.5))), -lim / (x1 - xc), lim / (x1 - xc))
    verts = [(x, y + tx * (z - zc) + tz * (x - xc), z) for z in zz for x in xs]
    nx = len(xs)
    faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i) for j in range(len(zz) - 1) for i in range(nx - 1)]
    o = mesh_object(name, verts, faces)

    def dirt_fn(co, nrm):
        dx = np.minimum(co[:, 0] - x0, x1 - co[:, 0])
        dz = co[:, 2] - z0
        dt = z1 - co[:, 2]
        m = 0.8 * np.exp(-dz / 0.045) + 0.55 * np.exp(-dx / 0.03) + 0.35 * np.exp(-dt / 0.04) + 0.9 * np.exp(-(dx + dz) / 0.07)
        return np.clip(m * dirt, 0.0, 1.0)
    return o, dirt_fn


class Decals:
    """Crisp paint ageing as thin geometry lying just proud of flat joinery faces: irregular flakes (paint lost to bare
    timber), tapering run-off streaks and hairline splits. Texture-frequency detail that vertex masks cannot give."""

    def __init__(self, r):
        self.r = r
        self.g = {}

    def _put(self, key, pts, fan=True):
        vs, fs = self.g.setdefault(key, ([], []))
        b = len(vs)
        vs += [tuple(q) for q in pts]
        if fan:
            n = len(pts) - 1
            fs += [(b, b + 1 + i, b + 1 + (i + 1) % n) for i in range(n)]
        else:
            fs.append(tuple(range(b, b + len(pts))))

    def flakes(self, O, U, V, N, w, h, n, grain="u", bias=1.0, size=(0.008, 0.026)):
        """n flakes scattered over the rectangle O + [0,w]U + [0,h]V facing N, elongated along the grain and
        gathering towards v = 0 (bias > 1)."""
        r = self.r
        O, U, V, N = Vector(O), Vector(U), Vector(V), Vector(N)
        sgn = -1.0 if U.cross(V).dot(N) < 0 else 1.0
        G = U if grain == "u" else V
        ax = "X" if abs(G.x) > 0.5 else ("Z" if abs(G.z) > 0.5 else "Y")
        n = int(n) + (1 if r.chance(n - int(n)) else 0)
        for _ in range(n):
            cu, cv = r.u(0, w), h * r.u(0, 1) ** bias
            a, el = r.u(*size), r.u(1.4, 3.4)
            ru, rv = (a * el, a) if grain == "u" else (a, a * el)
            k = int(r.u(6, 10))
            ph = r.u(0, 6.28)
            pts = [O + U * cu + V * cv + N * 0.0009]
            for i in range(k):
                t = ph + sgn * (2 * math.pi * i / k + r.jitter(0, 0.3))
                rr = r.u(0.35, 1.0)
                pts.append(O + U * clamp(cu + ru * rr * math.cos(t), 0, w) + V * clamp(cv + rv * rr * math.sin(t), 0, h) + N * 0.0009)
            self._put(("flake", ax), pts)

    def streaks(self, O, U, V, N, w, h, n, length=(0.06, 0.3)):
        """Run-off stains starting on the top edge of the rectangle (O = bottom-left corner) and running down it."""
        r = self.r
        O, U, V, N = Vector(O), Vector(U), Vector(V), Vector(N)
        flip = U.cross(V).dot(N) < 0
        n = int(n) + (1 if r.chance(n - int(n)) else 0)
        for _ in range(n):
            cu = r.u(0.004, w - 0.004)
            wd = min(r.u(0.004, 0.011), cu, w - cu)
            L = min(h, r.u(*length))
            top = O + V * h + N * 0.0007
            pts = [top + U * (cu - wd), top + U * (cu + clamp(r.jitter(0, 0.004), -wd, wd)) - V * L, top + U * (cu + wd)]
            self._put(("dark", "Z"), pts[::-1] if flip else pts, fan=False)

    def crack(self, P0, P1, N, width=0.0016, wob=0.0015):
        r = self.r
        P0, P1, N = Vector(P0), Vector(P1), Vector(N)
        T = (P1 - P0)
        S = T.normalized().cross(N).normalized()
        k = max(2, int(T.length / 0.07))
        prev, wp = P0, width * 0.35
        for i in range(1, k + 1):
            q = P0 + T * (i / k) + S * (r.jitter(0, wob) if i < k else 0.0)
            wd = width * (r.u(0.6, 1.0) if i < k else 0.35)
            o = N * 0.0008
            self._put(("dark", "Z"), [prev + S * (wp / 2) + o, q + S * (wd / 2) + o, q - S * (wd / 2) + o, prev - S * (wp / 2) + o], fan=False)
            prev, wp = q, wd

    def build(self, material, tag=""):
        out = []
        for (kind, ax), (vs, fs) in self.g.items():
            if not fs:
                continue
            o = mesh_object(f"wear_{kind}_{ax}{tag}", vs, fs)
            assign(o, material)
            uv_grain(o, axis=ax)
            out.append((kind, o))
        return out


# ------------------------------------------------------------------ goods (built at the origin, front = -Y)
def _pair(r, M):
    return r.pick([(M["A"], M["B"]), (M["A"], M["B"]), (M["B"], M["A"]), (M["cloth"], M["A"])])


def _case(r, M, w, h, d, m, sm, lying=False):
    """Suitcase: rounded body, two straps with brass latches, lid band and handle."""
    body = rounded_box("case", (w, d, h), radius=min(0.028, d * 0.3, h * 0.3), loc=(0, 0, h / 2), seg=2, bevel=0.004)
    assign(body, m)
    o = [body]
    for sx in (-1, 1):
        s = box("strap", (0.03, d + 0.008, h + 0.008), loc=(sx * w * 0.27, 0, h / 2))
        assign(s, sm)
        la = box("latch", (0.036, 0.008, 0.028), loc=(sx * w * 0.27, -d / 2 - 0.006, h * (0.5 if lying else 0.8)))
        assign(la, M["brass"])
        o += [s, la]
    if lying:
        path = [(-0.06, -d / 2, h / 2), (-0.045, -d / 2 - 0.028, h / 2), (0.045, -d / 2 - 0.028, h / 2), (0.06, -d / 2, h / 2)]
    else:
        path = [(-0.06, 0, h), (-0.045, 0, h + 0.03), (0.045, 0, h + 0.03), (0.06, 0, h)]
        lid = box("lid_band", (w + 0.004, d + 0.004, 0.008), loc=(0, 0, h * 0.72))
        assign(lid, sm)
        o.append(lid)
    hd = sweep("handle", path, 0.007, sides=5, smooth_path=False)
    assign(hd, sm)
    return o + [hd]


def _hatbox(r, M, rad, h, mat_):
    b = lathe("hatbox", [(0.0, 0.0), (rad, 0.0), (rad, h * 0.7), (rad + 0.005, h * 0.7), (rad + 0.005, h - 0.004),
                         (rad + 0.001, h), (0.0, h)], segments=14)
    assign(b, mat_)
    s = box("hb_strap", (0.03, 2 * rad + 0.016, h + 0.006), loc=(0, 0, h / 2))
    assign(s, M["B"] if mat_ is not M["B"] else M["brass"])
    return [b, s]


def _trunk(r, M, w, h, d, upright=False):
    m = r.pick([M["cloth"], M["B"], M["A"]])
    band = M["A"] if m is M["B"] else M["B"]
    body = box("trunk", (w, d, h), loc=(0, 0, h / 2), bevel=0.006, segments=1)
    assign(body, m)
    o = [body]
    if upright:                                   # wardrobe trunk standing on end: slatted bands, centre split, brass caps
        for fz in (0.06, 0.36, 0.64, 0.94):
            sl = box("slat", (w + 0.014, d + 0.014, 0.034), loc=(0, 0, h * fz))
            assign(sl, M["oak"])
            o.append(sl)
        sp = box("split", (0.007, d + 0.006, h - 0.004), loc=(0, 0, h / 2))
        assign(sp, band)
        o.append(sp)
        for sx in (-1, 1):
            for sz in (0.035, h - 0.035):
                cc = box("corner", (0.05, 0.012, 0.05), loc=(sx * (w / 2 - 0.02), -d / 2 - 0.004, sz))
                assign(cc, M["brass"])
                o.append(cc)
        lk = box("lock", (0.05, 0.01, 0.08), loc=(0, -d / 2 - 0.006, h * 0.5), bevel=0.002, segments=1)
        assign(lk, M["brass"])
        return o + [lk]
    rise = d * 0.28
    pts = [(-d / 2, h)] + [(-d / 2 * math.cos(math.pi * k / 6), h + rise * math.sin(math.pi * k / 6)) for k in range(1, 6)] + [(d / 2, h)]
    lid = prism_x("trunk_lid", pts, -w / 2, w / 2)
    assign(lid, m)
    o.append(lid)
    for sx in (-0.46, -0.16, 0.16, 0.46):
        sl = box("slat", (0.04, d + 0.012, h + 0.004), loc=(sx * w, 0, h / 2 + 0.002))
        assign(sl, M["oak"])
        o.append(sl)
    for zz in (0.014, h - 0.014):
        bd = box("band", (w + 0.006, d + 0.006, 0.024), loc=(0, 0, zz))
        assign(bd, band)
        o.append(bd)
    lk = box("lock", (0.06, 0.008, 0.07), loc=(0, -d / 2 - 0.004, h - 0.03), bevel=0.002, segments=1)
    assign(lk, M["brass"])
    return o + [lk]


def make_luggage(r, M, tier, d, hmax):
    dd = min(d, r.u(0.13, 0.16))
    if tier > 0:
        k = r.weighted([("small", 3), ("hatbox", 2), ("medium", 2)])
        if k == "hatbox":
            rad = min(d / 2 - 0.004, r.u(0.08, 0.1))
            return _hatbox(r, M, rad, min(hmax, r.u(0.14, 0.2)), r.pick([M["A"], M["cloth"]])), 2 * rad
        w, h = (r.u(0.3, 0.4), r.u(0.2, 0.26)) if k == "small" else (r.u(0.42, 0.52), r.u(0.28, 0.33))
        m, sm = _pair(r, M)
        return _case(r, M, w, min(h, hmax), dd, m, sm), w
    k = r.weighted([("wardrobe", 2), ("stack", 3), ("tall", 2), ("trunk", 2)])
    if k == "wardrobe" and hmax > 0.6:
        w = r.u(0.42, 0.5)
        return _trunk(r, M, w, min(hmax, r.u(0.66, 0.78)), min(d, 0.19), upright=True), w
    if k == "tall":
        w = r.u(0.4, 0.5)
        m, sm = _pair(r, M)
        return _case(r, M, w, min(hmax, r.u(0.55, 0.68)), dd, m, sm), w
    if k == "trunk":
        w, h, td = r.u(0.6, 0.75), r.u(0.34, 0.42), min(d, 0.19)
        o = _trunk(r, M, w, h, td)
        top = h + td * 0.28
        if top + 0.27 < hmax:
            m, sm = _pair(r, M)
            c = join(_case(r, M, r.u(0.32, 0.4), r.u(0.2, 0.25), dd * 0.9, m, sm), "case_on_trunk")
            place(c, loc=(r.jitter(0, 0.1), 0, top - 0.004), rot=(0, 0, r.jitter(0, 4)))
            o.append(c)
        return o, w
    o, z, wmax = [], 0.0, 0.0                       # stack of cases lying flat, biggest at the bottom
    for i in range(3):
        w, t = r.u(0.52, 0.7) * (1.0 - 0.13 * i), r.u(0.15, 0.2)
        if z + t > hmax:
            break
        m, sm = _pair(r, M)
        c = join(_case(r, M, w, t, min(d, 0.17), m, sm, lying=True), "case_stack")
        place(c, loc=(r.jitter(0, 0.03), 0, z), rot=(0, 0, r.jitter(0, 3)))
        o.append(c)
        z += t + 0.002
        wmax = max(wmax, w + 0.06)
    if z + 0.2 < hmax and r.chance(0.7):
        hb = join(_hatbox(r, M, min(d / 2 - 0.004, 0.09), 0.15, r.pick([M["A"], M["cloth"]])), "hatbox_top")
        place(hb, loc=(r.jitter(0, 0.1), 0, z + 0.002))
        o.append(hb)
    return o, wmax


HATS = {
    "top": [(0.0, 0.004), (0.082, 0.002), (0.15, 0.0), (0.152, 0.008), (0.096, 0.013), (0.089, 0.035), (0.093, 0.15), (0.099, 0.165), (0.0, 0.168)],
    "bowler": [(0.0, 0.004), (0.085, 0.002), (0.132, 0.0), (0.137, 0.012), (0.126, 0.015), (0.094, 0.015), (0.096, 0.06), (0.088, 0.1),
               (0.062, 0.126), (0.0, 0.136)],
    "boater": [(0.0, 0.004), (0.085, 0.002), (0.165, 0.0), (0.165, 0.009), (0.094, 0.011), (0.094, 0.085), (0.09, 0.09), (0.0, 0.09)],
    "cloche": [(0.0, 0.004), (0.095, 0.0), (0.125, -0.012), (0.13, -0.004), (0.104, 0.02), (0.102, 0.07), (0.082, 0.112), (0.0, 0.128)],
}


def make_hats(r, M, tier, d, hmax):
    """Hats on turned stands, alternately tall and low so the brims overlap in elevation and the window fills its height."""
    sc = clamp(d / 0.235, 0.7, 1.0)
    M["n"] = M.get("n", 0) + 1
    hi = max(0.3, hmax - 0.2 * sc)
    lvl = (M["n"] + 1) % 3                         # 0 = low, 1 = middle, 2 = high: neighbours never share a level
    if lvl == 0 and r.chance(0.35):                 # a hat box between the stands, with a hat resting on it
        rad = min(d / 2 - 0.004, r.u(0.1, 0.13))
        sh = r.u(0.15, 0.2)
        base = _hatbox(r, M, rad, sh, r.pick([M["cloth"], M["wood"]]))
        wid = 2 * rad + 0.02
    else:
        sh = (r.u(0.1, 0.16), hi * r.u(0.46, 0.54), hi * r.u(0.9, 1.0))[lvl]
        st = lathe("hatstand", [(0.0, 0.0), (0.06, 0.0), (0.062, 0.008), (0.024, 0.018), (0.01, 0.035), (0.009, sh * 0.5), (0.014, sh * 0.52),
                                (0.009, sh * 0.54), (0.008, sh - 0.04), (0.03, sh - 0.016), (0.05, sh - 0.004), (0.045, sh), (0.0, sh)], segments=8)
        assign(st, M["oak"])
        base = [st]
        wid = 0.15 * sc
    kind = r.pick(["top", "bowler", "boater", "cloche", "top", "bowler", "boater"])
    hat = lathe("hat", [(a * sc, b * sc) for (a, b) in HATS[kind]], segments=14, loc=(0, 0, 0.012 * sc if kind == "cloche" else -0.002))
    hm = {"top": M["A"], "bowler": M["A"], "boater": M["B"], "cloche": M["cloth"]}[kind]
    assign(hat, hm)
    bd = ring_band("hatband", radius=0.0965 * sc if kind != "top" else 0.092 * sc, width=0.03 * sc, thick=0.003,
                   loc=(0, 0, (0.03 if kind != "top" else 0.052) * sc), segments=14)
    assign(bd, M["wood"] if hm is not M["A"] else M["cloth"])
    g = join([hat, bd], "hat_on_stand")
    place(g, loc=(0, 0, sh), rot=(r.u(6, 20), r.jitter(0, 6), r.u(-40, 40)), scale=(1.0, 0.84, 1.0))
    return base + [g], wid


BOTTLES = {    # profile, body diameter, (label radius, z0, z1), neck radius
    "wine": ([(0.0, 0.0), (0.036, 0.0), (0.038, 0.006), (0.038, 0.17), (0.03, 0.2), (0.014, 0.235), (0.013, 0.295), (0.0, 0.297)], 0.076,
             (0.038, 0.05, 0.14), 0.013),
    "hock": ([(0.0, 0.0), (0.033, 0.0), (0.035, 0.006), (0.035, 0.14), (0.026, 0.21), (0.013, 0.27), (0.012, 0.35), (0.0, 0.352)], 0.07,
             (0.035, 0.04, 0.12), 0.012),
    "magnum": ([(0.0, 0.0), (0.048, 0.0), (0.05, 0.008), (0.05, 0.21), (0.04, 0.25), (0.018, 0.3), (0.017, 0.378), (0.0, 0.38)], 0.1,
               (0.05, 0.06, 0.17), 0.017),
    "flask": ([(0.0, 0.0), (0.05, 0.0), (0.055, 0.01), (0.055, 0.1), (0.04, 0.135), (0.016, 0.16), (0.015, 0.205), (0.0, 0.207)], 0.11,
              (0.055, 0.03, 0.085), 0.015),
    "jar": ([(0.0, 0.0), (0.06, 0.0), (0.065, 0.01), (0.068, 0.12), (0.06, 0.15), (0.045, 0.16), (0.045, 0.175), (0.052, 0.18),
             (0.052, 0.19), (0.0, 0.192)], 0.136, None, 0.045),
}


def make_bottles(r, M, tier, d, hmax):
    k = r.weighted([("wine", 4), ("hock", 2), ("magnum", 1.5), ("flask", 1.5), ("jar", 1.5)])
    prof, w, lab, nr = BOTTLES[k]
    s = r.u(0.95, 1.12)
    s = min(s, (d - 0.004) / w, (hmax - 0.01) / prof[-1][1])
    seg = M["seg"]
    b = lathe("bottle", [(a * s, z * s) for (a, z) in prof], segments=seg)
    assign(b, M["stone"] if k == "jar" else (M["A"] if r.chance(0.6) else M["B"]))
    o = [b]
    top = prof[-1][1] * s
    if lab:
        lr, z0, z1 = lab
        lb = lathe("label", [((lr + 0.0015) * s, z0 * s), ((lr + 0.0015) * s, z1 * s)], segments=seg)
        assign(lb, M["cloth"])
        fo = lathe("foil", [((nr + 0.0018) * s, top - 0.055 * s), ((nr + 0.0022) * s, top), (0.0, top + 0.002)], segments=seg)
        assign(fo, M["brass"] if r.chance(0.5) else M["wood"])
        o += [lb, fo]
    else:
        ck = cylinder("cork", nr * s * 0.9, 0.02, loc=(0, 0, top + 0.006), verts=6)
        assign(ck, M["oak"])
        o.append(ck)
    return o, w * s


def _vol(o, M, r, cx, cy, cz, t, d, h, lying=False):
    """One bound volume. Standing: spine to the street with a tooled label. Lying: fore-edge (pages) to the street."""
    lm = r.pick([M["A"], M["B"], M["A"], M["B"], M["wood"], M["oak"]])
    if not lying:
        b = box("book", (t, d, h), loc=(cx, cy, cz + h / 2))
        g = box("tooling", (t - 0.006, 0.002, h * 0.13), loc=(cx, cy - d / 2 - 0.0006, cz + h * r.u(0.62, 0.78)))
        assign(g, M["brass"] if r.chance(0.6) else M["cloth"])
    else:                                   # t = thickness (z), d = depth (y), h = length (x)
        b = box("book", (h, d, t), loc=(cx, cy, cz + t / 2))
        g = box("pages", (h - 0.014, 0.004, t - 0.01), loc=(cx, cy - d / 2 - 0.0006, cz + t / 2))
        assign(g, M["cloth"])
    assign(b, lm)
    o += [b, g]


def make_books(r, M, tier, d, hmax):
    o = []
    d0 = min(d - 0.006, 0.15)
    k = r.weighted([("stack", 3), ("globe", 1.5), ("face", 2.5), ("row", 1)] if tier == 0 else [("row", 3), ("face", 3.5), ("stack", 1)])
    if k == "row":
        n = int(r.u(4, 8))
        ts = [r.u(0.028, 0.06) for _ in range(n)]
        w = sum(ts) + 0.002 * n
        x = -w / 2
        for t in ts:
            dk = d0 * r.u(0.82, 1.0)
            _vol(o, M, r, x + t / 2, -(d0 - dk) / 2, 0.0, t, dk, min(hmax, r.u(0.18, 0.28)))
            x += t + 0.002
        return o, w
    if k == "face":                          # a volume shown cover-out, leaning back on the shelf
        w, h, t = r.u(0.15, 0.2), min(hmax - 0.01, r.u(0.22, 0.29)), r.u(0.025, 0.04)
        b = box("book_face", (w, t, h), loc=(0, 0, h / 2))
        assign(b, r.pick([M["A"], M["B"], M["wood"]]))
        ti = box("title", (w * 0.64, 0.002, h * 0.2), loc=(0, -t / 2 - 0.0008, h * 0.68))
        assign(ti, M["brass"])
        lb = box("imprint", (w * 0.4, 0.002, h * 0.06), loc=(0, -t / 2 - 0.0008, h * 0.2))
        assign(lb, M["cloth"])
        g = join([b, ti, lb], "book_faceout")
        place(g, loc=(0, 0.012, 0.002), rot=(-9, 0, r.jitter(0, 7)))
        return [g], w
    if k == "stack":
        z = 0.0
        w = r.u(0.2, 0.27)
        for i in range(int(r.u(3, 7))):
            t = r.u(0.03, 0.055)
            if z + t > hmax:
                break
            _vol(o, M, r, r.jitter(0, 0.012), r.jitter(0, 0.006), z, t, d0 * r.u(0.85, 1.0), w * r.u(0.82, 1.0), lying=True)
            z += t + 0.001
        return o, w
    rad = min(0.1, d / 2 - 0.008, (hmax - 0.1) / 2)
    st = lathe("globe_stand", [(0.0, 0.0), (0.06, 0.0), (0.062, 0.008), (0.03, 0.016), (0.012, 0.03), (0.01, 0.06), (0.016, 0.066),
                               (0.01, 0.072), (0.0, 0.074)], segments=10)
    assign(st, M["brass"])
    g = sphere("globe", rad, loc=(0, 0, 0.085 + rad), segments=14, rings=8)
    assign(g, M["B"])
    mer = torus("meridian", major=rad + 0.008, minor=0.004, loc=(0, 0, 0.085 + rad), rot=(90, 0, 25), major_seg=18, minor_seg=4)
    assign(mer, M["brass"])
    return [st, g, mer], 2 * rad + 0.03


def make_cakes(r, M, tier, d, hmax):
    sc = clamp(d / 0.25, 0.6, 1.0)
    R = 0.125 * sc
    k = r.weighted([("tiered", 3), ("croq", 1.6), ("sponge", 1.4)] if tier == 0 else [("sponge", 3), ("loaves", 2), ("cupcakes", 2.5)])
    sh = r.u(0.14, 0.24) if (tier == 0 and k != "sponge") else r.u(0.05, 0.1)
    st = lathe("cakestand", [(0.0, 0.0), (0.06 * sc, 0.0), (0.062 * sc, 0.006), (0.02, 0.012), (0.011, 0.028), (0.011, sh - 0.022),
                             (0.03, sh - 0.01), (R, sh - 0.005), (R + 0.004, sh), (0.0, sh + 0.001)], segments=8)
    assign(st, M["brass"])
    o = [st]
    if k == "tiered":                         # the showpiece: three iced tiers with piping
        ch = clamp(hmax - sh - 0.04, 0.2, r.u(0.36, 0.45))
        kz = ch / 0.19
        c = lathe("cake", [(a * sc, b * kz) for (a, b) in [(0.0, 0.0), (0.108, 0.0), (0.11, 0.06), (0.104, 0.066), (0.08, 0.066),
                                                         (0.081, 0.122), (0.075, 0.127), (0.052, 0.127), (0.053, 0.172), (0.047, 0.178),
                                                         (0.0, 0.178)]], segments=10, loc=(0, 0, sh))
        ice = r.pick([M["cloth"], M["B"]])
        assign(c, ice)
        top = sphere("topper", 0.018, loc=(0, 0, sh + 0.178 * kz + 0.014), segments=6, rings=4)
        assign(top, M["wood"])
        o += [c, top]
        for (rr, zz) in ((0.109, 0.01), (0.0805, 0.074), (0.0525, 0.134)):
            b = torus("piping", major=rr * sc, minor=0.007, loc=(0, 0, sh + zz * kz), major_seg=10, minor_seg=3)
            assign(b, M["B"] if ice is M["cloth"] else M["cloth"])
            o.append(b)
    elif k == "croq":                         # croquembouche: a cone of choux buns
        ch = clamp(hmax - sh - 0.04, 0.18, r.u(0.34, 0.46))
        n = 7
        prof = [(0.0, 0.0)]
        for i in range(n):
            rr = (0.092 * (1 - i / n) + 0.014) * sc
            prof += [(rr + 0.011, ch * (i + 0.3) / n), (rr - 0.004, ch * (i + 0.95) / n)]
        prof.append((0.0, ch + 0.012))
        c = lathe("croquembouche", prof, segments=8, loc=(0, 0, sh))
        assign(c, M["C"])
        o.append(c)
    elif k == "sponge":
        hh = r.u(0.06, 0.09)
        c = lathe("sponge", [(0.0, 0.0), (0.1 * sc, 0.0), (0.105 * sc, 0.01), (0.105 * sc, hh), (0.0, hh)], segments=10, loc=(0, 0, sh))
        assign(c, M["C"])
        ic = lathe("icing", [(0.1065 * sc, hh * 0.55), (0.108 * sc, hh * 0.7), (0.1085 * sc, hh), (0.1 * sc, hh + 0.011), (0.0, hh + 0.013)],
                   segments=10, loc=(0, 0, sh))
        assign(ic, r.pick([M["B"], M["cloth"], M["wood"]]))
        o += [c, ic]
        for i in range(4):
            a = 2 * math.pi * (i + 0.3) / 4
            chy = sphere("cherry", 0.013, loc=(0.068 * sc * math.cos(a), 0.068 * sc * math.sin(a), sh + hh + 0.018), segments=5, rings=3)
            assign(chy, M["wood"])
            o.append(chy)
    elif k == "loaves":
        for i in range(3):
            lf = sphere("loaf", 1.0, loc=((i - 1) * 0.064 * sc, r.jitter(0, 0.01), sh + 0.032 + (0.046 if i == 1 else 0.0)),
                        segments=6, rings=4, scale=(0.052 * sc, 0.1 * sc, 0.038))
            assign(lf, M["C"])
            o.append(lf)
    else:
        for i in range(5):
            a = 2 * math.pi * i / 5 + 0.4
            cc = lathe("cupcake", [(0.0, 0.0), (0.02, 0.0), (0.029, 0.036), (0.022, 0.056), (0.0, 0.08)],
                       segments=5, loc=(0.075 * sc * math.cos(a), 0.075 * sc * math.sin(a), sh))
            assign(cc, r.pick([M["B"], M["cloth"], M["C"]]))
            o.append(cc)
    return o, 2 * R + 0.01


MAKERS = {"luggage": make_luggage, "hats": make_hats, "bottles": make_bottles, "books": make_books, "cakes": make_cakes}
GAPS = {"luggage": (0.03, 0.1), "hats": (0.02, 0.06), "bottles": (0.05, 0.14), "books": (0.04, 0.13), "cakes": (0.08, 0.18)}
TICKETS = {"luggage": 0.8, "hats": 0.6, "bottles": 0.3, "books": 0.4, "cakes": 0.6}
TIERS = {"luggage": (0.86, 9.0), "hats": (9.0, 9.0), "bottles": (0.41, 0.41), "books": (0.37, 0.36), "cakes": (0.62, 0.36)}   # floor zone, shelf pitch


def goods_mats(kind, brass, oak, wood, cloth, stone, own):
    """Up to two own materials per display; everything else reuses the shopfront's brass / oak / paint / curtain cloth / stone
    so the whole asset stays within the material budget. own = number of free material slots (2, or 1 with painted lettering)."""
    C = cloth
    if kind == "luggage":
        if own > 1:
            A, B = mat("leather", "#a8642f"), mat("leather", "#2c4034")
        else:                                       # one free slot: tan hide, strapped and trimmed in the shop's own colour
            A, B = mat("leather", "#a8642f"), wood
    elif kind == "hats":
        A, B = mat("canvas", "#4b4038"), (mat("wicker", "#e0c27e") if own > 1 else cloth)
    elif kind == "bottles":
        A = mat("glass_clear", "#3f7d4c", opacity=0.82)
        B = mat("glass_clear", "#8a4e1c", opacity=0.85) if own > 1 else A
    elif kind == "books":
        A, B = mat("leather", "#7d2f25"), (mat("canvas", "#2f5661") if own > 1 else wood)
    else:
        A, C = (mat("wax", "#f1c3bd") if own > 1 else cloth), mat("terracotta", "#b06a32")
        B = A
    return {"A": A, "B": B, "C": C, "brass": brass, "oak": oak, "wood": wood, "cloth": cloth, "stone": stone, "seg": 8}


def fill_row(r, kind, M, tier, x0, x1, yb, z, d, hmax, gapk):
    """Fill one tier with seeded goods standing against the back (yb), price tickets propped along the front edge."""
    items, total = [], 0.0
    avail = x1 - x0
    for _ in range(60):
        objs, w = MAKERS[kind](r, M, tier, d - 0.02, hmax)
        g = r.u(*GAPS[kind]) * gapk
        if total + w > avail:
            delete(objs)
            break
        items.append((objs, w, g))
        total += w + g
    if not items:
        return []
    total -= items[-1][2]
    extra = max(0.0, avail - total) / (len(items) + 1)
    x = x0 + extra
    out = []
    for k, (objs, w, g) in enumerate(items):
        o = join(objs, f"goods_{tier}_{k}")
        place(o, loc=(x + w / 2, yb - (d - 0.02) / 2 + r.jitter(0, 0.004), z), rot=(0, 0, r.jitter(0, 4)))
        out.append(o)
        if r.chance(TICKETS[kind]):
            t = box("ticket", (0.088, 0.002, 0.06), loc=(x + w / 2 + r.jitter(0, w * 0.25), yb - d + 0.01, z + 0.029), rot=(-14, 0, r.jitter(0, 9)))
            assign(t, M["cloth"])
            out.append(t)
            M["tickets"] = M.get("tickets", 0) + 1
        x += w + g + extra
    return out


# ------------------------------------------------------------------ the shopfront
def build(p, rng):
    W, H, D = p.width, p.height, p.depth
    nb = max(1, int(p.bays))
    lit = bool(p.lit)
    wr = p.wear
    wood = mat("wood_painted", night_safe(p.paint), rough=0.62)     # satin oil paint: arrises and mouldings catch the lamps
    gilt = mat("brass", rough=0.85)
    stone = mat("stone_smooth", "#a59c8c")
    lead = mat("lead")
    oak = mat("wood_oak", "#9c774e")
    cloth = mat("canvas", p.curtain)
    glass = mat("glass_clear", opacity=0.16, rough=0.3)
    # one clean luminous material for everything the shop's own light shines through: curtain, transom panes, bulbs
    win = mat("glass_frosted", shade(p.curtain, 0.6), glow=p.glow, opacity=1.0) if lit else mat("window_dark")
    painted_letters = str(p.lettering).strip().lower() not in ("gilt", "gold", "")
    letter_m = mat("wood_painted", p.lettering, rough=0.45) if painted_letters else gilt

    # ---- setting out
    cs = 0.8                                    # cornice scale: projects 0.124, 0.12 high
    cproj, ch = 0.155 * cs, 0.15 * cs
    edge = 0.006                                # lead dressed over the cornice edge
    xo = W / 2 - cproj - 0.02 - edge            # pilaster outer face
    pw = clamp(W * 0.085, 0.18, 0.26)           # pilaster width
    xi = xo - pw                                # pilaster inner face = half clear width
    xc = xo - pw / 2
    Wi = 2 * xi
    zk, zpl = 0.10, 0.30                        # kerb top, pilaster plinth top
    hr = p.riser
    zs = hr + 0.06                              # sill top / bottom of the window frame
    lead_rise = 0.11                            # flashing upstand against the wall
    zc0 = H - lead_rise - ch                    # underside of the cornice
    hf = min(p.fascia, H * 0.18)
    zf0 = zc0 - hf                              # bottom of the fascia = head of the glazing
    ht = p.transom if p.transom >= 0.14 else 0.0
    ztr = zf0 - ht                              # centre line of the transom rail
    yg = D - 0.075                              # back of the glazing frames (all y are distances in front of the wall)
    ygl = yg + 0.012                            # glass line, in a rebate of the frame
    ypf = yg + 0.035                            # front of the glazing frames
    yrf = D - 0.05                              # front of the stall-riser framing
    timber, gilded = [], []
    dec = Decals(rng.fork("wear"))
    nfl = 300.0 * wr ** 1.5                     # flake budget
    nst = 26.0 * wr                             # streak budget
    F = (0, -1, 0)

    # ---- pilasters: stone plinth, box pier, base mould, sunk reeded panel, panelled return, necking, console, cap
    rs = rng.fork("stone")
    base_p = [(0, 0), (0.024, 0), (0.024, 0.03), (0.014, 0.046), (0.005, 0.07), (0, 0.076)]
    neck_p = [(0, 0), (0.005, 0.004), (0.016, 0.012), (0.021, 0.024), (0.021, 0.038), (0.012, 0.042), (0, 0.042)]
    cprof = [(0.0, 0.03), (0.025, 0.0), (0.048, 0.02), (0.056, 0.08), (0.048, 0.14), (0.04, 0.2), (0.04, 0.3), (0.05, 0.45), (0.07, 0.6),
             (0.095, 0.72), (0.112, 0.82), (0.118, 0.9), (0.112, 0.97), (0.10, 1.0), (0.0, 1.0)]
    hcn = hf - 0.03
    zp0, zp1 = zpl + 0.13, zf0 - 0.1
    stones = []
    for sx in (-1, 1):
        cx = sx * xc
        pl = stone_block(f"plinth{sx}", (pw + 0.03, D + 0.012, zpl), loc=(cx, -(D + 0.012) / 2 - 0.003, zpl / 2 + 0.003), rng=rs,
                         chamfer=0.016, detail=0.16)
        stones.append(pl)
        pier = box(f"pier{sx}", (pw, D, zc0 - zpl), loc=(cx, -D / 2, (zpl + zc0) / 2), bevel=0.004, segments=1)
        densify(pier, 0.26)
        uv_grain(pier, axis="Z")
        timber.append(pier)
        if sx < 0:
            path = lambda z: [(-xo, 0, z), (-xo, -D, z), (-xi, -D, z), (-xi, -(D - 0.08), z)]
        else:
            path = lambda z: [(xi, -(D - 0.08), z), (xi, -D, z), (xo, -D, z), (xo, 0, z)]
        timber.append(msweep(f"pier_base{sx}", path(zpl), base_p))
        timber.append(msweep(f"pier_neck{sx}", path(zf0 - 0.05), neck_p))
        fw = pw - 0.07
        fr = shape_extrude(f"pier_panel{sx}", rect(cx - fw / 2, cx + fw / 2, zp0, zp1),
                           [rect(cx - fw / 2 + 0.024, cx + fw / 2 - 0.024, zp0 + 0.024, zp1 - 0.024)], depth=0.012,
                           loc=(0, -D, 0), rot=(90, 0, 0), bevel=0.002)
        timber.append(fr)
        nr = 3
        for k in range(nr):
            rx = cx + (k - (nr - 1) / 2) * (fw - 0.048) / nr
            timber.append(cylinder(f"reed{sx}{k}", 0.011, zp1 - zp0 - 0.075, loc=(rx, -D + 0.002, (zp0 + zp1) / 2), verts=8))
        # the return is joinery too: a planted sunk panel between the same base and necking, and a small one at fascia level
        rw = D - 0.085
        if rw > 0.1:
            yrc = -D / 2 - 0.004
            timber.append(ring_frame(f"return_panel{sx}", (sx * xo, yrc, (zp0 + zp1) / 2), (0, 1, 0), (0, 0, 1), (sx, 0, 0),
                                     rw, zp1 - zp0, 0.026, 0.012))
            timber.append(side_field(f"return_field{sx}", sx, xo, yrc, rw - 0.052, zp0 + 0.026, zp1 - 0.026))
            if zc0 - zf0 > 0.2:
                timber.append(ring_frame(f"return_upper{sx}", (sx * xo, yrc, (zf0 + zc0) / 2 - 0.004), (0, 1, 0), (0, 0, 1), (sx, 0, 0),
                                         rw, zc0 - zf0 - 0.07, 0.024, 0.012))
        ros = lathe(f"rosette{sx}", [(0.0, 0.0), (0.03, 0.0), (0.032, 0.004), (0.022, 0.009), (0.014, 0.008), (0.01, 0.014), (0.0, 0.017)],
                    segments=10, loc=(cx, -D - 0.018, zf0 - 0.029), rot=(90, 0, 0))
        gilded.append(ros)
        # scrolled console carrying the cornice
        cw = pw - 0.05
        timber.append(prism_x(f"console{sx}", [(-(D + f), zf0 + t * hcn) for (f, t) in cprof], cx - cw / 2, cx + cw / 2, bev=0.003))
        timber.append(prism_x(f"console_leaf{sx}", [(-(D + f + (0.008 if 0.0 < t < 1.0 and f > 0 else 0.0)), zf0 + t * hcn) for (f, t) in cprof],
                              cx - cw * 0.2, cx + cw * 0.2, bev=0.002))
        for (f, t, rr) in ((0.03, 0.075, 0.021), (0.078, 0.85, 0.033)):
            timber.append(cylinder(f"volute{sx}", rr, cw + 0.014, loc=(cx, -(D + f), zf0 + t * hcn), rot=(0, 90, 0), verts=14, bevel=0.003, segments=1))
            gilded.append(cylinder(f"volute_eye{sx}", rr * 0.36, cw + 0.024, loc=(cx, -(D + f), zf0 + t * hcn), rot=(0, 90, 0), verts=8))
        timber.append(box(f"console_cap{sx}", (pw + 0.016, 0.125, 0.03), loc=(cx, -(D + 0.05), zc0 - 0.015), bevel=0.004, segments=1))
        # paint ageing on the pier: flaked margins beside the reeded panel and at the foot, run-off below the necking
        mg = (pw - fw) / 2 - 0.006
        for ex in (cx - pw / 2 + 0.005, cx + fw / 2 + 0.001):
            dec.flakes((ex, -D, zpl + 0.08), (1, 0, 0), (0, 0, 1), F, mg, 1.15, nfl * 0.045, grain="v", bias=2.4)
            dec.streaks((ex, -D, zp1 - 0.5), (1, 0, 0), (0, 0, 1), F, mg, 0.45, nst * 0.06)
        dec.flakes((cx - pw / 2 + 0.005, -D, zpl + 0.08), (1, 0, 0), (0, 0, 1), F, pw - 0.01, 0.046, nfl * 0.04, grain="u")
        if rw > 0.1:
            dec.flakes((sx * xo, -D + 0.006, zpl + 0.08), (0, 1, 0), (0, 0, 1), (sx, 0, 0), 0.03, 0.9, nfl * 0.02, grain="v", bias=2.4)
            dec.flakes((sx * xo, -D + 0.006, zpl + 0.08), (0, 1, 0), (0, 0, 1), (sx, 0, 0), D - 0.012, 0.046, nfl * 0.03, grain="u")

    # ---- stone kerb under the stall-riser
    x = -xi
    while x < xi - 1e-3:
        L = min(rs.u(0.5, 0.85), xi - x)
        if xi - (x + L) < 0.32:
            L = xi - x
        stones.append(stone_block("kerb", (L - 0.006, 0.16, zk), loc=(x + L / 2, -(yrf + 0.03 - 0.08), zk / 2 + 0.003), rng=rs,
                                  chamfer=0.01, detail=0.18))
        x += L
    for s in stones:
        assign(s, stone)
        paint(s, grime=mask_mul(height_ramp(0.0, 0.3), 0.3 + 0.3 * wr),
              wear=mask_mul(noise_mask(0.06, seed=p.seed + 15, lo=0.45, hi=0.7), 0.3 + 0.5 * wr),
              special=mask_mul(height_ramp(0.0, 0.2), noise_mask(0.2, seed=p.seed + 5), 0.8 * wr))

    # ---- bay setting out
    jamb, mst = 0.05, 0.036
    bw = (Wi - 2 * jamb - (nb - 1) * mst) / nb
    bx = [-xi + jamb + bw / 2 + k * (bw + mst) for k in range(nb)]
    mx = [-xi + jamb + bw + mst / 2 + k * (bw + mst) for k in range(nb - 1)]

    # ---- stall-riser: backing, mortised frame, fielded panels, bolection mouldings
    back = box("riser_back", (Wi, 0.02, hr - zk), loc=(0, -(yrf - 0.04), (zk + hr) / 2))
    timber.append(back)
    holes = []
    split = 2 if bw > 0.8 else 1
    for c in bx:
        sw = (bw + mst - 0.075 * split) / split
        for s in range(split):
            hx = c - (bw + mst) / 2 + 0.0375 + s * (sw + 0.075)
            holes.append((hx, hx + sw, zk + 0.1, hr - 0.07))
    fr = shape_extrude("riser_frame", rect(-xi, xi, zk, hr), [rect(*h) for h in holes], depth=0.03, loc=(0, -(yrf - 0.03), 0), rot=(90, 0, 0))
    bevel(fr, 0.002, segments=1)
    timber.append(fr)
    bol = [(0.012, 0.0), (0.01, 0.006), (0.002, 0.012), (-0.006, 0.012), (-0.012, 0.006), (-0.016, -0.004), (-0.016, -0.02), (0.012, -0.02)]
    rc = rng.fork("cracks")
    for k, (x0, x1, z0, z1) in enumerate(holes):
        timber.append(fielded_panel(f"riser_panel{k}", x0, x1, z0, z1, -(yrf - 0.03)))
        timber.append(msweep(f"bolection{k}", [(x0, -yrf, z0), (x1, -yrf, z0), (x1, -yrf, z1), (x0, -yrf, z1)], bol, up=(0, -1, 0), closed=True))
        m = min(0.045, (x1 - x0) * 0.3, (z1 - z0) * 0.3)
        yf = -(yrf - 0.03) - 0.014
        dec.flakes((x0 + m, yf, z0 + m), (1, 0, 0), (0, 0, 1), F, x1 - x0 - 2 * m, z1 - z0 - 2 * m, nfl * 0.2 / len(holes), grain="u", bias=1.8,
                   size=(0.01, 0.03))
        dec.streaks((x0 + m, yf, z0 + m), (1, 0, 0), (0, 0, 1), F, x1 - x0 - 2 * m, z1 - z0 - 2 * m, nst * 0.3 / len(holes), length=(0.05, 0.2))
        # hairline shoulders where the muntin is tenoned into the rails, opening up with age
        if k > 0 and rc.chance(0.35 + 0.65 * wr):
            xa = holes[k - 1][1]
            for zz in (z0 - 0.016, z1 + 0.016):
                if rc.chance(0.75):
                    dec.crack((xa - 0.012, -yrf, zz), (x0 + 0.012, -yrf, zz), F, wob=0.0008)
    # bottom rail takes the boots, mops and splash-back: the heaviest flaking, and splits along the grain
    dec.flakes((-xi + 0.004, -yrf, zk + 0.004), (1, 0, 0), (0, 0, 1), F, Wi - 0.008, 0.078, nfl * 0.3, grain="u", bias=1.6, size=(0.009, 0.03))
    dec.flakes((-xi + 0.004, -yrf, hr - 0.05), (1, 0, 0), (0, 0, 1), F, Wi - 0.008, 0.044, nfl * 0.04, grain="u")
    for _ in range(int(1 + 3 * wr)):
        cx0 = rc.u(-xi + 0.05, xi - 0.5)
        zz = zk + rc.u(0.02, 0.07)
        dec.crack((cx0, -yrf, zz), (cx0 + rc.u(0.15, 0.45), -yrf, zz + rc.jitter(0, 0.004)), F)

    # ---- sill (weathered top, bullnose, drip) with a brass sill band
    bk = yrf - yg
    sill_p = [(-bk, 0.0), (0.012, 0.0), (0.016, 0.01), (0.03, 0.014), (0.04, 0.02), (0.045, 0.034), (0.042, 0.047), (0.0, 0.056), (-bk, 0.06)]
    sill = msweep("sill", [(-xi, -yrf, hr), (xi, -yrf, hr)], sill_p, step=0.035 if Wi < 2.2 else 0.06)
    uv_grain(sill, axis="X")
    timber.append(sill)
    sb = box("sill_band", (Wi - 0.1, 0.004, 0.018), loc=(0, -(yrf + 0.046), hr + 0.036), bevel=0.0015, segments=1)
    assign(sb, gilt)
    paint(sb, grime=0.4, wear=noise_mask(0.15, seed=p.seed + 3, lo=0.2, hi=0.6))
    # brass ventilator grille let into the middle riser panel (dulled: it sits in the shadow of the sill)
    gx0, gx1, gz0, gz1 = holes[len(holes) // 2]
    if gx1 - gx0 > 0.34 and gz1 - gz0 > 0.24:
        gc, gzc, gw, gh = (gx0 + gx1) / 2, (gz0 + gz1) / 2, 0.24, 0.1
        slots = [rounded_rect(0.012, gh - 0.036, 0.004, 1) for _ in range(8)]
        slots = [[(gc + (k - 3.5) * 0.026 + a, gzc + b) for (a, b) in sl] for k, sl in enumerate(slots)]
        gr = shape_extrude("vent_grille", [(gc + a, gzc + b) for (a, b) in rounded_rect(gw, gh, 0.01, 2)], slots, depth=0.005,
                           loc=(0, -(yrf - 0.03 + 0.014), 0), rot=(90, 0, 0))
        assign(gr, gilt)
        paint(gr, grime=0.8, special=mask_mul(noise_mask(0.05, seed=p.seed + 12), 0.7))
        gb = box("vent_back", (gw - 0.02, 0.002, gh - 0.02), loc=(gc, -(yrf - 0.03 + 0.0145), gzc))
        assign(gb, oak)
        paint(gb, grime=1.0)

    # ---- glazing frames: main lights and transom lights, cut from boards; puttied panes each a little out of true
    kind = p.arch
    top_m = 0.052 if ht > 0 else 0.045
    hh = (ztr - top_m) - (zs + 0.045)
    main_holes = []
    for c in bx:
        if kind == "flat":
            pts = arch_pts(bw, hh, kind="flat")
        else:
            rise = min(0.075, bw * 0.13) if kind == "segmental" else min(0.11, bw * 0.2)
            pts = arch_pts(bw, hh, spring=hh - rise, kind="segmental", n=8)
        main_holes.append([(c + px, zs + 0.045 + pz) for (px, pz) in pts])
    mf = shape_extrude("light_frame", rect(-xi, xi, zs, ztr), main_holes, depth=0.035, loc=(0, -yg, 0), rot=(90, 0, 0))
    bevel(mf, 0.003, segments=1)
    timber.append(mf)
    rgl = rng.fork("glass")
    putty_p = [(0.001, -0.001), (-0.011, -0.001), (0.001, 0.012)]
    putties = []
    for k, c in enumerate(bx):
        gl, dirt = pane(f"glass{k}", c - bw / 2 - 0.006, c + bw / 2 + 0.006, zs + 0.039, ztr - top_m + 0.006, -ygl, rgl, dirt=0.55 + 0.45 * wr, fine=nb < 4)
        assign(gl, glass)
        paint(gl, grime=dirt)
        # (no bed face: it lay 1 mm behind the glass, which a host may re-skin as opaque dark glass)
        putties.append(lod_faces(msweep(f"putty{k}", [(x, -ygl, z) for (x, z) in main_holes[k]], putty_p, up=(0, -1, 0), closed=True),
                                 drop=[(0, 1, 0), (0, -1, 0)], above=-1, cos=0.9))
    # worn bottom rail and jambs of the lights (rain runs down the glass onto them)
    dec.flakes((-xi + 0.004, -ypf, zs + 0.003), (1, 0, 0), (0, 0, 1), F, Wi - 0.008, 0.038, nfl * 0.14, grain="u", size=(0.007, 0.02))
    for sx in (-1, 1):
        dec.flakes((-xi + 0.004 if sx < 0 else xi - 0.044, -ypf, zs + 0.045), (1, 0, 0), (0, 0, 1), F, 0.04, 0.8, nfl * 0.03, grain="v", bias=2.2)
    if ht > 0:
        tr_holes = []
        npn = max(1, int(round(bw / 0.27)))
        bar = 0.022
        pwid = (bw - (npn - 1) * bar) / npn
        ph = ht - 0.04 - 0.032
        for c in bx:
            for k in range(npn):
                pc = c - bw / 2 + pwid / 2 + k * (pwid + bar)
                if kind == "flat":
                    pts = arch_pts(pwid, ph, kind="flat")
                elif kind == "segmental":
                    pts = arch_pts(pwid, ph, spring=ph - pwid * 0.2, kind="segmental", n=6)
                elif kind == "round":
                    pts = arch_pts(pwid, ph, spring=max(ph * 0.3, ph - pwid / 2), kind="round", n=8)
                else:
                    pts = arch_pts(pwid, ph, spring=max(ph * 0.25, ph - pwid * 0.75), kind="pointed", n=8)
                tr_holes.append([(pc + px, ztr + 0.04 + pz) for (px, pz) in pts])
        tf = shape_extrude("transom_frame", rect(-xi, xi, ztr, zf0), tr_holes, depth=0.03, loc=(0, -yg, 0), rot=(90, 0, 0))
        timber.append(tf)
        rail_p = [(0, -0.04), (0.012, -0.04), (0.015, -0.028), (0.027, -0.02), (0.04, -0.006), (0.042, 0.01), (0.034, 0.014), (0.022, 0.03),
                  (0.012, 0.035), (0, 0.035)]
        rail = msweep("transom_rail", [(-xi, -ypf, ztr), (xi, -ypf, ztr)], rail_p)
        uv_grain(rail, axis="X")
        timber.append(rail)
        # transom lights: obscured panes glowing with the shop's ceiling light
        # (a glowing box is two sheets seen through each other: one sheet, the front face of the old box, at every level)
        tl = lod_faces(box("transom_lights", (Wi - 0.02, 0.006, ht - 0.03), loc=(0, -(yg + 0.0085), ztr + ht / 2 + 0.005)),
                       keep=(0, -1, 0), above=-1, cos=0.9)
        assign(tl, win)
    head_p = [(0, -0.03), (0.012, -0.03), (0.02, -0.02), (0.035, -0.012), (0.045, 0.0), (0.045, 0.012), (0, 0.012)]
    timber.append(msweep("head_mould", [(-xi, -ypf, zf0), (xi, -ypf, zf0)], head_p))
    # turned colonnette mullions standing on the sill, carrying the transom rail
    col_top = ztr - 0.04 if ht > 0 else zf0 - 0.03
    for k, x in enumerate(mx):
        a, b = colonnette(f"mullion{k}", x, -(ypf + 0.016), zs - 0.004, col_top - (zs - 0.004), seg=10 if nb < 4 else 8)
        timber += a
        gilded += b
    for sx in (-1, 1):                              # quarter-round jamb beads
        timber.append(cylinder(f"jamb_bead{sx}", 0.014, col_top - zs, loc=(sx * (xi - 0.004), -(ypf + 0.002), (zs + col_top) / 2), verts=8))

    # ---- display enclosure: oak bed, panelled window-back, pleated curtain, lamp rail
    zfl = zs + 0.012
    bed = box("display_bed", (Wi, yg, 0.03), loc=(0, -yg / 2, zfl - 0.015))
    uv_grain(bed, axis="X")
    ceil_ = box("display_ceiling", (Wi, yg, 0.02), loc=(0, -yg / 2, zf0 - 0.01))
    timber.append(ceil_)
    zlr = (ztr - 0.035) if ht > 0 else (zf0 - 0.06)   # lamp rail under the transom
    ztop = zlr - 0.11                               # goods stay below the lamps
    Hm = ztop - zfl
    hw = clamp(0.36 * Hm, 0.26, 0.5)                # panelled window-back
    wb = box("window_back", (Wi, 0.012, hw), loc=(0, -0.006, zfl + hw / 2))
    uv_grain(wb, axis="Z")
    npan = max(2, int(round(Wi / 0.5)))
    pwd = (Wi - 0.06 * (npan + 1)) / npan
    wf = shape_extrude("window_back_frame", rect(-xi, xi, zfl, zfl + hw),
                       [rect(-xi + 0.06 + k * (pwd + 0.06), -xi + 0.06 + k * (pwd + 0.06) + pwd, zfl + 0.07, zfl + hw - 0.05) for k in range(npan)],
                       depth=0.012, loc=(0, -0.012, 0), rot=(90, 0, 0), bevel=0.002)
    uv_grain(wf, axis="X")
    wc = box("window_back_cap", (Wi, 0.034, 0.022), loc=(0, -0.017, zfl + hw + 0.011), bevel=0.004, segments=1)
    uv_grain(wc, axis="X")
    for o in (bed, wb, wf, wc):
        assign(o, oak)
        paint(o, grime=0.08)
    zc_a, zc_b = zfl + hw + 0.022, zf0 - 0.025
    lam = 0.085 + rng.jitter(0, 0.008)

    def curtain_fn(u, v):
        x = -xi + 0.004 + (Wi - 0.008) * u
        ph = 2 * math.pi * x / lam + 0.7 * math.sin(x * 5.3 + p.seed)
        amp = 0.0095 - 0.004 * v                    # gathered at the head, fuller at the hem
        return (x + 0.004 * (1 - v) * math.sin(ph * 0.5 + 1.0), -(0.006 + amp * (1.0 + math.sin(ph))), zc_a + (zc_b - zc_a) * v)
    cur = surface("curtain", curtain_fn, nu=max(24, int(Wi / lam * 4)), nv=2)
    assign(cur, win if lit else cloth)
    brassware = [cylinder("lamp_rail", 0.006, Wi, loc=(0, -(yg - 0.05), zlr), rot=(0, 90, 0), verts=6)]
    for k, c in enumerate(bx):                      # one small shaded lamp per light, hung from the rail
        brassware.append(lathe(f"win_shade{k}", [(0.0, -0.004), (0.008, -0.005), (0.047, -0.047), (0.051, -0.049), (0.018, -0.008),
                                                 (0.008, 0.0), (0.008, 0.012), (0.0, 0.014)], segments=6, loc=(c, -(yg - 0.05), zlr - 0.016)))
        bulb = sphere(f"win_bulb{k}", 0.014, loc=(c, -(yg - 0.05), zlr - 0.05), segments=6, rings=4)
        assign(bulb, win)

    # ---- the goods: floor of the bed plus shelves on brass brackets, each tier seeded and ticketed
    yb = -0.04
    gd = yg - 0.012 + yb                            # usable depth between the curtain and the glass
    f0, pitch = TIERS[p.goods]
    if p.goods == "luggage":
        f0 = min(0.8, Hm - 0.36) if Hm >= 1.0 else 9.0
    tiers = [(zfl, min(f0, Hm) - 0.03)]
    zsh = zfl + f0
    while zsh + 0.26 <= ztop + 1e-6 and len(tiers) < 4:
        tiers.append((zsh, min(pitch - 0.035, ztop - zsh)))
        zsh += pitch
    for k, (zt_, hm_) in enumerate(tiers[1:]):
        sh = box(f"shelf{k}", (Wi - 0.006, gd + 0.01, 0.018), loc=(0, yb - (gd + 0.01) / 2 + 0.006, zt_ - 0.009), bevel=0.002, segments=1)
        uv_grain(sh, axis="X")
        assign(sh, oak)
        nbk = max(2, int(math.ceil(Wi / 0.75))) + 1
        for j in range(nbk):
            bxk = -xi + 0.03 + (Wi - 0.06) * j / (nbk - 1)
            brassware.append(prism_x(f"shelf_bracket{k}_{j}", [(yb + 0.004, zt_ - 0.018), (yb - gd * 0.75, zt_ - 0.018), (yb - gd * 0.75, zt_ - 0.026),
                                                              (yb + 0.004, zt_ - 0.11)], bxk - 0.004, bxk + 0.004))
    for o in brassware:
        assign(o, gilt)
        paint(o, wear=0.5, grime=0.1)
    M = goods_mats(p.goods, gilt, oak, wood, cloth, stone, 1 if painted_letters else 2)
    M["seg"] = 8 if Wi < 2.6 else 6
    rg = rng.fork("goods")
    gapk = clamp(Wi / 2.1, 1.0, 1.7) if p.goods in ("bottles", "books") else 1.0
    goods = []
    for k, (zt_, hm_) in enumerate(tiers):
        goods += fill_row(rg.fork(f"tier{k}"), p.goods, M, k, -xi + 0.05, xi - 0.05, yb, zt_, gd, hm_, gapk)
    for g in goods:
        paint(g, grime=0.05, wear=mask_mul(noise_mask(0.08, seed=p.seed + 9), 0.4))

    # ---- fascia: tilted board with frame moulding and raised lettering (built flat, then leaned forward)
    tilt = 7.0 + rng.jitter(0, 0.6)
    fparts = []
    fb = box("fascia_board", (Wi, 0.03, hf), loc=(0, 0.015, hf / 2))
    densify(fb, 0.3)
    uv_grain(fb, axis="X")
    assign(fb, wood)
    fparts.append(fb)
    fm_p = [(0, 0), (0, 0.014), (-0.01, 0.018), (-0.02, 0.012), (-0.03, 0.008), (-0.036, 0)]
    fm = msweep("fascia_mould", [(-xi + 0.004, 0, 0.012), (xi - 0.004, 0, 0.012), (xi - 0.004, 0, hf - 0.055), (-xi + 0.004, 0, hf - 0.055)],
                fm_p, up=(0, -1, 0), closed=True)
    assign(fm, wood)
    fparts.append(fm)
    body = str(p.sign_text).strip()
    t = None
    if body:
        fh = hf - 0.067
        size = fh * 0.6
        spc = 1.06 + 0.012 * min(len(body), 12)     # open letter-spacing, more for long words (signwriter's tracking)
        t = text("lettering", body, size=size, depth=0.012, font="serif", spacing=spc)
        mn, mx_ = world_bbox([t])
        tw = mx_.x - mn.x
        maxw = Wi - 0.3
        if tw > maxw:
            delete([t])
            size *= maxw / tw
            t = text("lettering", body, size=size, depth=0.012, font="serif", spacing=spc)
            mn, mx_ = world_bbox([t])
        place(t, loc=(-(mn.x + mx_.x) / 2, 0.0, 0.012 + fh / 2 - (mn.z + mx_.z) / 2))
        bm = bmesh.new()                            # the letters are planted on the board: drop their hidden backs
        bm.from_mesh(t.data)
        bmesh.ops.delete(bm, geom=[fc for fc in bm.faces if fc.normal.z < -0.9], context='FACES')
        bm.to_mesh(t.data)
        bm.free()
        assign(t, letter_m)
        fparts.append(t)
    fdec = Decals(rng.fork("fascia_wear"))
    fx0, fz0, fw_, fh_ = -xi + 0.045, 0.05, Wi - 0.09, hf - 0.145
    fdec.flakes((fx0, 0, fz0), (1, 0, 0), (0, 0, 1), F, fw_, fh_, nfl * 0.12, grain="u", bias=1.5, size=(0.008, 0.024))
    fdec.streaks((fx0, 0, fz0), (1, 0, 0), (0, 0, 1), F, fw_, fh_, nst * 0.5, length=(0.05, 0.22))
    if rc.chance(0.4 + 0.6 * wr):                   # the joint between the two fascia boards has opened
        zj = fz0 + fh_ * rc.u(0.3, 0.42)
        xa = rc.u(-xi + 0.05, -xi + 0.4)
        fdec.crack((xa, 0, zj), (xa + (Wi - 0.5) * rc.u(0.5, 1.0), 0, zj), F, wob=0.0006)
    wear_objs = fdec.build(wood, "_fascia")
    for o in fparts + [w for (_, w) in wear_objs]:
        apply_transform(o)
        place(o, loc=(0, -(D - 0.035), zf0), rot=(tilt, 0, 0))
    if t is not None:
        if painted_letters:
            paint(t, wear=mask_mul(noise_mask(0.05, seed=p.seed + 13, lo=0.5, hi=0.8), wr), grime=0.05)
        else:
            paint(t, wear=1.0, grime=0.0)
    for fas in fparts[:2]:
        paint(fas, special=mask_mul(noise_mask(0.6, seed=p.seed + 1, lo=0.3, hi=0.8), 0.25 + 0.5 * wr),
              wear=mask_mul(noise_mask(0.07, seed=p.seed + 2, lo=0.55, hi=0.8), wr))

    # ---- dentil course, mitred cornice with returns
    timber.append(box("dentil_band", (Wi + 0.01, 0.04, 0.05), loc=(0, -(D + 0.008), zc0 - 0.025)))
    dn = max(3, int((Wi - 0.02) / 0.066))
    dpitch = (Wi - 0.02) / dn
    dent = box("dentil", (dpitch * 0.52, 0.024, 0.032))
    timber.append(instances_to_mesh("dentils", dent, [trs((-xi + 0.01 + (k + 0.5) * dpitch, -(D + 0.04), zc0 - 0.018)) for k in range(dn)],
                                    keep_base=False))
    corn_p = [(a * cs, b * cs) for (a, b) in
              [(-0.03, 0), (0.015, 0), (0.015, 0.015), (0.03, 0.02), (0.045, 0.035), (0.05, 0.05), (0.105, 0.05), (0.105, 0.056), (0.11, 0.056),
               (0.11, 0.095), (0.118, 0.1), (0.13, 0.108), (0.145, 0.125), (0.15, 0.14), (0.155, 0.14), (0.155, 0.15), (-0.03, 0.15)]]
    xp, yc = xo + 0.02, D + 0.02
    corn = msweep("cornice", [(-xp, 0, zc0), (-xp, -yc, zc0), (xp, -yc, zc0), (xp, 0, zc0)], corn_p)
    timber.append(corn)

    # ---- lead capping: laid to a fall, wood-cored rolls, dressed down over the cornice on three sides, turned up the wall
    zt = H - lead_rise
    ye = yc + cproj
    xl = W / 2
    leads = [prism_x("lead_top", [(0, H), (-0.005, H), (-0.006, zt + 0.062), (-0.03, zt + 0.05), (-(ye - 0.02), zt + 0.007), (-(ye + edge - 0.003), zt + 0.005),
                                  (-(ye + edge), zt + 0.001), (-(ye + edge), zt - 0.014), (-(ye + 0.002), zt - 0.014), (-(ye + 0.002), zt), (0, zt)], -(xl - 0.005), xl - 0.005)]
    for sx in (-1, 1):
        leads.append(prism_x(f"lead_apron{sx}", [(0, zt + 0.062), (-0.03, zt + 0.05), (-(ye - 0.02), zt + 0.007), (-(ye + edge), zt + 0.003),
                                                (-(ye + edge), zt - 0.014), (0, zt - 0.014)], sx * (xl - 0.005), sx * xl))
    nroll = max(2, int(round(W / 0.62)))
    rl = rng.fork("lead")
    for k in range(1, nroll):
        rx = -W / 2 + W * k / nroll + rl.jitter(0, 0.02)
        leads.append(sweep(f"lead_roll{k}", [(rx, -0.004, H - 0.006), (rx, -0.016, zt + 0.064), (rx, -(ye - 0.02), zt + 0.016),
                                             (rx, -(ye + edge + 0.001), zt - 0.004)], 0.013, sides=6, smooth_path=False))
    for o in leads:
        assign(o, lead)
        paint(o, grime=mask_mul(noise_mask(0.09, seed=p.seed + 6, lo=0.25, hi=0.8), 0.5 + 0.2 * wr), special=noise_mask(0.3, seed=p.seed + 16, lo=0.4, hi=0.9))

    # ---- finish the joinery: paint, gilding, weathering masks, then the crisp ageing decals
    for o in timber + putties:
        assign(o, wood)
    for o in gilded:
        assign(o, gilt)
        paint(o, grime=0.2, wear=0.7)
    for o in putties:                                # old linseed putty: a dark line round every pane
        paint(o, grime=0.85)
    splash = mask_mul(height_ramp(0.05, 0.4), 0.18 + 0.3 * wr)                         # street dirt kicked up the riser
    kick = mask_mul(height_ramp(zk, zk + 0.42), noise_mask(0.04, seed=p.seed + 7, lo=0.42, hi=0.62), 0.25 + 0.75 * wr)   # scuffed kick zone
    knocks = mask_mul(height_ramp(zpl, 1.3), noise_mask(0.09, seed=p.seed + 17, lo=0.52, hi=0.78), wr)   # bags, boots and carts
    fade = mask_mul(noise_mask(0.7, seed=p.seed + 8, lo=0.3, hi=0.8), facing((0, -1, 0)), 0.2 + 0.5 * wr)
    for o in timber:
        paint(o, grime=splash, wear=kick, special=fade)
        paint(o, wear=knocks)
    # the sill nosing loses its paint first: crisp along-the-grain wear on a finely divided moulding
    paint(sill, wear=mask_mul(noise_mask(0.05, seed=p.seed + 4, lo=0.38, hi=0.6), 0.45 + 0.55 * wr),
          grime=mask_mul(facing((0, 0, 1)), 0.25))
    paint(corn, special=mask_mul(facing((0, -1, 0)), 0.3 + 0.5 * wr))
    wear_objs += dec.build(wood)
    for (kd, o) in wear_objs:
        if kd == "dark":
            paint(o, grime=0.9)                      # run-off stains and open joints
        else:
            paint(o, wear=1.0, grime=0.2)

    # ---- brass swan-neck sign lamps springing from the cornice, washing the fascia
    nlamp = 1 if Wi < 1.5 else (2 if Wi < 3.4 else 3)
    wash = clamp((0.4 / max(0.05, paint_value(p.paint))) ** 2, 0.22, 1.0)      # pale paint needs far less light to read
    yb_, zb = yc + 0.11 * cs, zc0 + 0.076 * cs
    lamps = []
    for k in range(nlamp):
        lx = ((k + 0.5) / nlamp - 0.5) * Wi * (0.86 if nlamp > 1 else 1.0)
        lamps.append(sweep(f"lamp_arm{k}", [(lx, -yb_, zb), (lx, -(yb_ + 0.10), zb + 0.05), (lx, -(yb_ + 0.22), zb + 0.062),
                                          (lx, -(yb_ + 0.30), zb + 0.02), (lx, -(yb_ + 0.315), zb - 0.03)], 0.008, sides=6, step=0.04))
        lamps.append(lathe(f"lamp_rose{k}", [(0.0, 0.0), (0.024, 0.0), (0.025, 0.004), (0.014, 0.008), (0.011, 0.016), (0.0, 0.016)], segments=8,
                           loc=(lx, -yb_ + 0.002, zb), rot=(90, 0, 0)))
        th = 38.0
        sy, sz = -(yb_ + 0.315), zb - 0.03
        lamps.append(lathe(f"lamp_shade{k}", [(0.0, -0.003), (0.01, -0.004), (0.036, -0.03), (0.074, -0.066), (0.078, -0.07), (0.041, -0.03),
                                             (0.021, -0.012), (0.012, 0.0), (0.012, 0.012), (0.0, 0.014)], segments=10, loc=(lx, sy, sz), rot=(th, 0, 0)))
        by, bz = sy + 0.034 * math.sin(math.radians(th)), sz - 0.034 * math.cos(math.radians(th))
        bulb = sphere(f"lamp_bulb{k}", 0.017, loc=(lx, by, bz), segments=6, rings=4)
        assign(bulb, win)
        if lit:
            light(f"sign{k}", (lx, by + 0.012, bz - 0.016), color=p.light_color, intensity=p.intensity * 0.075 * wash, range=2.5, flicker=0.05)
    for o in lamps:
        assign(o, gilt)
        paint(o, grime=0.25, special=mask_mul(noise_mask(0.04, seed=p.seed + 14), 0.5 * wr))

    # ---- light: the display's own top light, and the window's spill thrown out onto the pavement
    if lit:
        # top light from the lamp rail: short range, it only models the goods and the curtain from above
        nsp = 1 if Wi < 3.0 else 2
        if nlamp + nsp < 5:                         # never more than five lights on one shopfront
            light("display", (bx[nb // 2], -(yg - 0.04), zlr - 0.09), color=p.light_color, intensity=p.intensity * 0.03, range=1.4, flicker=0.04)
        # the lit window as a source: one light out in front of the glass, level with the transom lights, so the goods
        # are lit from above and in front, its mirror image falls on the transom rail and not in a pane, the riser
        # (far below, at a glancing angle) stays darker than the window and the pavement pool fades out by 4 m
        for k in range(nsp):
            lx = 0.0 if nsp == 1 else (k - 0.5) * Wi * 0.5
            light(f"spill{k}", (lx, -(ygl + 1.5), ztr + 0.85 * p.transom), color=p.light_color,
                  intensity=p.intensity * (1.0 if nsp == 1 else 0.62), range=5.0, flicker=0.04)
    socket("sign", "attach", (0, -(D + 0.02), zf0 + hf / 2), role="sign", width=Wi, height=hf)
    socket("awning", "attach", (0, -(D - 0.02), zf0 - 0.01), role="awning", width=Wi)
    for sx, nm in ((-1, "l"), (1, "r")):
        socket(f"lantern_{nm}", "attach", (sx * xc, -(D + 0.014), min(2.15, zf0 - 0.3)), role="lantern")
