"""Mullioned casement window of the old-town houses: a painted timber frame set back in a projecting
surround, 1-4 lights divided by moulded mullions, small leaded quarries or glazing-bar panes, a
projecting weathered sill, a timber or stone lintel and optional ledged-and-braced plank shutters.

Construction, wall outwards
  surround  'timber': pegged oak posts tenoned into a lintel beam (end grain, proud pegs, beaten lead
            flashing on top) standing on an oak sill beam with a weathered nose on two brackets;
            'stone' : individually dressed jamb stones (in-and-out quoins), a sloped sill on a bed
            mould and either a monolithic lintel under a label mould (flat head) or a ring of
            voussoirs with a proud keystone on skew-back springers (arched head).
  frame     one painted joinery frame with separately planted mullion mouldings, optional weathered
            transom and a sloped sub-sill; every light has its own sash hung on iron butt hinges.
  glazing   every pane is its own slightly tilted polygon (so each one catches the lantern light and
            shows the room a little differently) held in lead cames with iron saddle bars, or in
            timber glazing bars.  `lit` picks window_lit / window_dark; one casement may stand ajar
            on its stay.
  shutters  ledged-and-braced boards on forged strap hinges and pintles; open leaves fold back
            against the wall and are held by S-shaped shutter dogs, closed leaves are barred and
            leak light through the board joints and the cut-out.

Mount 'wall': origin = centre of the sill line on the wall plane y = 0.  The surround projects
0.13 m and the frame sits ~0.08 m back inside it, so the window has a real reveal when it is simply
stuck onto a flat wall (the surround is the reveal box).  `width` / `height` are the frame size
(sill line to head, crown of the arch for arched heads); surround, sill and shutters add to it -
measure `.size` when cutting openings.

Levels: lod 1 keeps every pane, came and board (cames as flat strips, no bolts / nails, nothing modelled where
the wall or the glass hides it); lod 2 is the surround, one glowing polygon per light behind the lattice of
flat cames / glazing bars, and the shutter leaves as painted faces with their strap hinges.
"""
import math
import numpy as np
from tk import *

META = {"name": "Mullioned casement window", "class": "component", "mount": "wall",
        "mount_height": 1.05, "focus": 0.5, "lod": True}

PARAMS = {
    "width": {"default": 1.1, "min": 0.5, "max": 2.4, "desc": "frame width between the jambs (m)"},
    "height": {"default": 1.35, "min": 0.6, "max": 2.4, "desc": "sill line to head / crown of the arch (m)"},
    "lights": {"default": 2, "min": 1, "max": 4, "desc": "lights side by side, divided by mullions"},
    "head": {"default": "flat", "choices": ["flat", "arched"], "desc": "flat or segmental-arched head"},
    "arch_rise": {"default": 0.17, "min": 0.06, "max": 0.32, "desc": "rise of the arch as a fraction of the width"},
    "surround": {"default": "timber", "choices": ["timber", "stone"], "desc": "pegged oak posts and lintel, or dressed stone"},
    "glazing": {"default": "diamond", "choices": ["diamond", "square", "bars"],
                "desc": "leaded diamond quarries, leaded square quarries, or timber glazing bars"},
    "panes_x": {"default": 4, "min": 1, "max": 8, "desc": "panes across one light"},
    "panes_y": {"default": 6, "min": 1, "max": 14, "desc": "pane rows up one light (square / bars; diamonds keep their proportion)"},
    "transom": {"default": False, "desc": "horizontal transom with small top lights"},
    "lit": {"default": True, "desc": "room behind is lit (window_lit) or dark (window_dark)"},
    "glow": {"default": 3.0, "min": 0.5, "max": 6.0, "desc": "emissive strength of the lit glazing"},
    "curtain": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "how far the curtains inside are drawn"},
    "open": {"default": 0.0, "min": 0.0, "max": 1.0, "desc": "one casement standing ajar (0 = shut)"},
    "shutters": {"default": True, "desc": "pair of plank shutters"},
    "shutter_open": {"default": 1.0, "min": 0.0, "max": 1.0, "desc": "0 = closed and barred, 1 = folded back against the wall"},
    "shutter_cutout": {"default": "heart", "choices": ["none", "heart", "diamond"], "desc": "pierced motif in each leaf"},
    "shutter_paint": {"default": "#3b6b52", "desc": "shutter paint colour"},
    "frame_paint": {"default": "#d6cdb6", "desc": "paint colour of frame, mullions and sashes"},
    "timber": {"default": "#5a4130", "desc": "stain of the oak surround"},
    "stone": {"default": "#9c8f7a", "desc": "tint of the stone surround"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: chipped paint, rust, grime, crooked boards"},
    "intensity": {"default": 0.0, "min": 0.0, "max": 20.0,
                  "desc": "optional spill light in front of a lit window (cd, try 3-5); 0 = glowing glass only, no light "
                          "socket - the right choice for the many windows of a house front"},
}

VARIANTS = {
    "stone_arched": {"surround": "stone", "head": "arched", "width": 1.5, "height": 1.55, "lights": 3,
                     "glazing": "square", "panes_x": 3, "panes_y": 7, "shutters": False,
                     "frame_paint": "#5a231c", "curtain": 0.2, "seed": 7},
    "shuttered": {"width": 0.95, "height": 1.2, "shutter_open": 0.0, "shutter_cutout": "heart",
                  "shutter_paint": "#7c2f24", "glow": 5.0, "seed": 12},
    "cottage_dark": {"width": 0.62, "height": 0.82, "lights": 1, "glazing": "bars", "panes_x": 2, "panes_y": 3,
                     "lit": False, "shutters": False, "age": 0.95, "frame_paint": "#8f9c8a",
                     "timber": "#463628", "seed": 21},
    "tall_transom": {"width": 1.0, "height": 1.95, "lights": 2, "transom": True, "glazing": "bars", "panes_x": 2,
                     "panes_y": 4, "open": 0.55, "shutter_paint": "#2d4c6b", "shutter_cutout": "diamond",
                     "frame_paint": "#e4dfd0", "curtain": 0.6, "glow": 2.3, "intensity": 2.4, "seed": 33},
    "ribbon": {"width": 2.15, "height": 1.0, "lights": 4, "glazing": "diamond", "panes_x": 4, "shutters": False,
               "frame_paint": "#3f3329", "timber": "#4a3525", "age": 0.7, "seed": 41},
}

# joinery sizes (m)
D = 0.13            # projection of the surround in front of the wall plane
FW = 0.055          # face width of the frame members
FD = 0.055          # frame depth (its front face is at y = -FD)
MW = 0.05           # mullion face width
SW = 0.03           # sash stile / rail face width
TH = 0.05           # transom height
Y_GLASS = -0.030
BT = 0.022          # shutter board thickness
LT = 0.022          # shutter ledge thickness


# ------------------------------------------------------------------ 2D helpers (x across, z up)

class _Head:
    """Head line of the opening: flat at H or a segmental arch with its crown at H."""

    def __init__(self, W, H, arched, rise_frac):
        self.W, self.H, self.arched = W, H, arched
        self.rise = 0.0
        if arched:
            self.rise = max(0.04, min(rise_frac * W, W * 0.46, H * 0.4))
            self.R = ((W / 2) ** 2 + self.rise ** 2) / (2 * self.rise)
            self.zc = H - self.R
        self.spring = H - self.rise

    def top(self, x, d=0.0):
        """z of the head line at x, offset inwards (towards the glass) by d."""
        if not self.arched:
            return self.H - d
        r = self.R - d
        return self.zc + math.sqrt(max(r * r - x * x, 0.0))


class _Reg:
    """Convex region under the head line: x0..x1, from z0 up to min(z1, head inset by d)."""

    def __init__(self, hd, x0, x1, z0, z1=None, d=0.0):
        self.hd, self.x0, self.x1, self.z0, self.z1, self.d = hd, x0, x1, z0, z1, d

    def ztop(self, x):
        t = self.hd.top(x, self.d)
        return t if self.z1 is None else min(t, self.z1)

    def inset(self, a):
        return _Reg(self.hd, self.x0 + a, self.x1 - a, self.z0 + a, None if self.z1 is None else self.z1 - a,
                    self.d + a)

    def poly(self, n=8):
        pts = [(self.x0, self.z0), (self.x1, self.z0)]
        lo = min(self.hd.top(self.x0, self.d), self.hd.top(self.x1, self.d))
        if not self.hd.arched or (self.z1 is not None and lo >= self.z1):
            pts += [(self.x1, self.ztop(self.x1)), (self.x0, self.ztop(self.x0))]
        else:
            for i in range(n + 1):
                x = self.x1 + (self.x0 - self.x1) * i / n
                pts.append((x, self.ztop(x)))
        return pts


def _area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def _clip(subj, clipp):
    """Sutherland-Hodgman: clip polygon `subj` by the convex CCW polygon `clipp`."""
    out = list(subj)
    n = len(clipp)
    for i in range(n):
        a, b = clipp[i], clipp[(i + 1) % n]
        ex, ez = b[0] - a[0], b[1] - a[1]
        if abs(ex) + abs(ez) < 1e-9:
            continue
        inp, out = out, []
        if not inp:
            break
        sd = [ex * (q[1] - a[1]) - ez * (q[0] - a[0]) for q in inp]
        for j in range(len(inp)):
            q0, q1 = inp[j], inp[(j + 1) % len(inp)]
            s0, s1 = sd[j], sd[(j + 1) % len(inp)]
            if s0 >= 0:
                out.append(q0)
            if (s0 >= 0) != (s1 >= 0):
                t = s0 / (s0 - s1)
                out.append((q0[0] + (q1[0] - q0[0]) * t, q0[1] + (q1[1] - q0[1]) * t))
    res = []
    for q in out:
        if not res or abs(q[0] - res[-1][0]) + abs(q[1] - res[-1][1]) > 2e-4:
            res.append(q)
    if len(res) > 1 and abs(res[0][0] - res[-1][0]) + abs(res[0][1] - res[-1][1]) <= 2e-4:
        res.pop()
    return res


def _clip_line(q, d, poly):
    """Part of the infinite line q + t d inside the convex CCW polygon -> (a, b) or None."""
    t0, t1 = -1e9, 1e9
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ez = b[0] - a[0], b[1] - a[1]
        num = ex * (q[1] - a[1]) - ez * (q[0] - a[0])
        den = ex * d[1] - ez * d[0]
        if abs(den) < 1e-12:
            if num < -1e-9:
                return None
            continue
        t = -num / den
        if den > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t1 - t0 < 1e-5:
        return None
    return (q[0] + d[0] * t0, q[1] + d[1] * t0), (q[0] + d[0] * t1, q[1] + d[1] * t1)


def _heart(cx, cz, s, n=20):
    pts = []
    for i in range(n):
        t = 2 * pi * i / n
        x = 16 * sin(t) ** 3
        y = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t)
        pts.append((cx + x / 32.0 * s, cz + (y + 2.5) / 32.0 * s))
    return pts


# ------------------------------------------------------------------ mesh helpers

def _prism(name, pts, a0, a1, axis="Y", bevel=0.002, seg=1):
    """Closed 2D polygon extruded between a0 and a1 along a world axis.
    axis 'Y': pts are (x, z);  'X': pts are (y, z);  'Z': pts are (x, y)."""
    lo, hi = min(a0, a1), max(a0, a1)
    o = extrude_profile(name, pts, hi - lo, bevel=bevel, segments=seg)
    if axis == "Y":
        M = Matrix(((1, 0, 0, 0), (0, 0, 1, lo), (0, 1, 0, 0), (0, 0, 0, 1)))
    elif axis == "X":
        M = Matrix(((0, 0, 1, lo), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    else:
        M = Matrix.Translation((0, 0, lo))
    o.data.transform(M)
    if M.determinant() < 0:
        o.data.flip_normals()
    o.data.update()
    return o


def _plate(name, outer, holes, y_back, depth, bevel=0.002):
    """Shape with holes in the XZ plane, from y_back towards -Y by depth."""
    o = shape_extrude(name, outer, holes, depth=depth, loc=(0, y_back, 0), rot=(90, 0, 0), bevel=bevel)
    return apply_transform(o)


def _xform(objs, M):
    """Bake UVs / modifiers, then move the meshes by the 4x4 matrix M (world space)."""
    for o in objs:
        realize(o)
        apply_transform(o)
        o.data.transform(M)
        if M.determinant() < 0:
            o.data.flip_normals()
        o.data.update()
    return objs


def _rot_about(px, py, ang_deg):
    return (Matrix.Translation((px, py, 0)) @ Matrix.Rotation(radians(ang_deg), 4, 'Z')
            @ Matrix.Translation((-px, -py, 0)))


def _mx(*fns):
    def fn(co, n):
        return np.max(np.stack([(f(co, n) if callable(f) else np.full(len(co), float(f))) for f in fns]), axis=0)
    return fn


OUT = (0, -1, 0)     # looking out of the wall: what faces the street
BACK = (0, 1, 0)     # ... and what lies against the wall


def _stone(name, size, loc, rng, chamfer=0.008, rough=0.003, detail=0.17):
    o = box(name, size, bevel=min(chamfer, min(size) * 0.2), segments=1)
    _roughen(o, rng, rough, max(size), detail)
    place(o, loc)
    return lod_faces(o, drop=BACK)


def _roughen(o, rng, rough, big, detail=0.17):
    if lod() >= 2:                                      # far: the dressed block itself (the draws stay)
        rng.randrange(1000), rng.randrange(1000)
    else:
        densify(o, detail)
        displace_noise(o, strength=rough, scale=big * 0.6, seed=rng.randrange(1000))
        displace_noise(o, strength=rough * 0.5, scale=max(big * 0.15, 0.03), seed=rng.randrange(1000))
    smooth(o, 45)
    uv_box(o)
    return o


def _strip_base(name, w, t, back=True, card=False):
    """Unit-length (along X) low came / glazing-bar section: front face and two splayed sides (card: one flat strip)."""
    a, b = w * 0.5, w * 0.28
    v = [(-0.5, 0, -a), (0.5, 0, -a), (0.5, -t, -b), (-0.5, -t, -b), (-0.5, -t, b), (0.5, -t, b), (0.5, 0, a), (-0.5, 0, a)]
    f = [(0, 1, 2, 3), (3, 2, 5, 4), (4, 5, 6, 7)]
    if card:
        v, f = [(-0.5, -t, -a), (0.5, -t, -a), (0.5, -t, a), (-0.5, -t, a)], [(0, 1, 2, 3)]
    o = mesh_object(name, v, f)
    flat(o)
    return o


# ------------------------------------------------------------------ glazing of one light

def _glazing(name, reg, kind, nx, ny, rng, age, gmat, gmat2, seedval, uv_off, lead_m, bar_m, iron, hidden=False):
    """Panes (one polygon each) + cames / glazing bars + saddle bars for one light. Returns objects."""
    Lv = lod()
    poly = reg.poly(10)
    x0, x1 = reg.x0, reg.x1
    z0 = reg.z0
    z1 = max(q[1] for q in poly)
    w, h = x1 - x0, z1 - z0
    cells, lines = [], []
    if kind == "diamond":
        nx = max(1, min(nx, int(w / 0.05)))
        dw = w / nx
        rows = max(1, int(round(h / (dw * 1.55))))
        dh = h / rows
        for j in range(rows + 1):
            for i in range(nx + 1):
                for cx, cz in ((x0 + i * dw, z0 + j * dh), (x0 + (i + 0.5) * dw, z0 + (j + 0.5) * dh)):
                    cells.append([(cx + dw / 2, cz), (cx, cz + dh / 2), (cx - dw / 2, cz), (cx, cz - dh / 2)])
        for k in range(-rows - 1, nx + rows + 2):
            lines.append(((x0 + (k + 0.5) * dw, z0), (dw, dh), 0))
            lines.append(((x0 + (k + 0.5) * dw, z0), (-dw, dh), 1))
    else:
        nx = max(1, min(nx, int(w / 0.06)))
        ny = max(1, min(ny, int(h / 0.06)))
        dw, dh = w / nx, h / ny
        for j in range(ny):
            for i in range(nx):
                xa, za = x0 + i * dw, z0 + j * dh
                cells.append([(xa, za), (xa + dw, za), (xa + dw, za + dh), (xa, za + dh)])
        for i in range(1, nx):
            lines.append(((x0 + i * dw, z0), (0.0, 1.0), 0))
        for j in range(1, ny):
            lines.append(((x0, z0 + j * dh), (1.0, 0.0), 1))

    # panes: each its own polygon, slightly out of plane and tilted (hand-made glass set in soft lead)
    verts, faces, grime, mi = [], [], [], []
    tilt = 0.012 + 0.014 * age if kind != "bars" else 0.006
    for c in cells:
        c = _clip(c, poly)
        if len(c) < 3 or _area(c) < 4e-6:
            continue
        cx = sum(q[0] for q in c) / len(c)
        cz = sum(q[1] for q in c) / len(c)
        ax, az = rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt)
        dy = rng.uniform(-0.0006, 0.0006)
        # CLEARANCE RULE: no corner of a pane leaves the glass plane by more than 2 mm (a large pane at full tilt came
        # within 2.5 mm of the face of the cames, which stands 6 mm in front of that plane)
        far = max(abs(ax * (x - cx) + az * (z - cz)) for (x, z) in c)
        if far > 0.002 - abs(dy):
            ax, az = ax * (0.002 - abs(dy)) / far, az * (0.002 - abs(dy)) / far
        g = 0.3 + rng.uniform(0.0, 0.3) * (0.4 + age) + 0.32 * (0.3 + age) * (1.0 - (cz - z0) / max(h, 1e-3)) ** 2
        if rng.chance(0.06 + 0.1 * age):
            g += 0.3
        b0 = len(verts)
        for (x, z) in c:
            e = min(x - x0, x1 - x, z - z0) / 0.05           # dirt gathers against the frame
            verts.append((x, Y_GLASS + dy + ax * (x - cx) + az * (z - cz), z))
            grime.append(clamp(g + 0.25 * (0.3 + age) * clamp(1.0 - e, 0.0, 1.0), 0.0, 1.0))
        faces.append(tuple(range(b0, b0 + len(c))))
        mi.append(1 if rng.chance(0.14) else 0)
    if Lv >= 2 and faces:                               # far: the whole light as one polygon (every draw above stays)
        one = reg.poly(4)
        verts, faces, mi = [(x, Y_GLASS, z) for (x, z) in one], [tuple(range(len(one)))], [0]
        grime = [sum(grime) / len(grime)] * len(one)
    out = []
    if faces:
        o = mesh_object(name + "_glass", verts, faces)
        assign(o, gmat)
        if gmat2 is not None and gmat2 != gmat:
            o.data.materials.append(gmat2)
            if len(o.data.polygons) == len(mi):
                o.data.polygons.foreach_set("material_index", np.array(mi, dtype=np.int32))
        flat(o)
        uv_planar(o, (1, 0, 0), (0, 0, 1), offset=uv_off)
        if len(o.data.vertices) == len(grime):
            arr = np.array(grime, dtype=np.float32)
            paint(o, grime=lambda co, n: arr, special=seedval)
        else:
            paint(o, grime=0.3, special=seedval)
        out.append(o)

    # cames / bars: one long strip per lattice line, clipped to the light
    if kind == "bars":
        sw_, st_, m_ = 0.019, 0.013, bar_m
    else:
        sw_, st_, m_ = 0.0075 * (2 if Lv >= 2 else 1), 0.0042, lead_m
    mats = []
    nth = {}
    for q, d, fam in lines:
        nth[fam] = nth.get(fam, 0) + 1
        if Lv >= 2 and kind != "bars" and nth[fam] % 2 == 0:   # far: every second came, twice as wide (as dark a light)
            continue
        seg = _clip_line(q, d, poly)
        if seg is None:
            continue
        (xa, za), (xb, zb) = seg
        ln = hypot(xb - xa, zb - za)
        if ln < 0.012:
            continue
        ang = degrees(atan2(zb - za, xb - xa))
        # CLEARANCE RULE: the face of a came stands 6 mm in front of the plane of the panes (they are tilted up to
        # 2 mm out of it, and a came face 1-2 mm over lit glass flickers)
        mats.append(trs(((xa + xb) / 2, Y_GLASS - 0.0018 - 0.0004 * fam, (za + zb) / 2), (0, -ang, 0), (ln, 1, 1)))
    if mats and not (hidden and Lv >= 2):               # lead cames: flat strips at the levels (they keep the pane as dark as it is); glazing bars: cards far away; behind closed shutters: none far away
        base = _strip_base(name + "_cb", sw_, st_, card=Lv >= (2 if kind == "bars" else 1))
        assign(base, m_)
        cm = instances_to_mesh(name + "_cames", base, mats, keep_base=False)
        flat(cm)
        out.append(cm)
    # saddle bars tie the leaded panel to the sash
    if kind != "bars" and h > 0.45:
        nb = 2 if h > 0.8 else 1
        for k in range(nb):
            z = z0 + h * (k + 1) / (nb + 1) + rng.uniform(-0.01, 0.01)
            seg = _clip_line((x0, z), (1.0, 0.0), poly)
            if seg is None:
                continue
            ln = seg[1][0] - seg[0][0] + 0.012
            if Lv >= 2:
                continue
            # CLEARANCE RULE: the bar clears the lit panes (tilted up to 2 mm out of their plane) by 4 mm and more
            b = cylinder(name + f"_saddle{k}", 0.0034, ln, loc=((seg[0][0] + seg[1][0]) / 2, Y_GLASS - 0.0112, z),
                         rot=(0, 90, 0), verts=lod_pick(6, 3))
            assign(b, iron)
            out.append(b)
    return out


# ------------------------------------------------------------------ surrounds

def _timber_surround(p, rng, hd, W, H, jw, Dl, oak, endg, lead_m):
    Lv = lod()
    oaks, leads = [], []
    Hj = hd.spring
    for s in (-1, 1):
        o = plank(f"jamb{s}", length=Hj + 0.02, width=jw, thick=D + 0.01, axis='Z', bevel=0.007, warp=0.0025,
                  rng=rng.fork(f"jamb{s}"), loc=(s * (W / 2 + jw / 2), -D / 2 + 0.005, Hj / 2))
        oaks.append(lod_faces(o, drop=[BACK, (0, 0, 1), (0, 0, -1)]))      # the wall, the lintel and the sill hide these
    ov = 0.10 + rng.uniform(0.0, 0.03)
    L = W + 2 * jw + 2 * ov
    if not hd.arched:
        lh = 0.13 + 0.025 * W
        zt = H + lh
        o = plank("lintel", length=L, width=Dl + 0.01, thick=lh, axis='X', bevel=0.011, warp=0.004,
                  rng=rng.fork("lintel"), end_mat=endg, loc=(rng.uniform(-0.015, 0.015), -Dl / 2 + 0.005, H + lh / 2))
        oaks.append(lod_faces(o, drop=BACK))
        zpeg = H + lh * 0.45
    else:
        zt = H + 0.10
        pts = [(-L / 2, Hj), (-W / 2, Hj)]
        n = (14, 8, 4)[Lv]
        for i in range(1, n):
            x = -W / 2 + W * i / n
            pts.append((x, hd.top(x)))
        pts += [(W / 2, Hj), (L / 2, Hj), (L / 2, zt), (-L / 2, zt)]
        o = _prism("lintel", pts, 0.01, -Dl, "Y", bevel=0.008)
        assign(o, oak)
        assign_faces(o, endg, lambda c, n: abs(n.x) > 0.9)
        uv_grain(o, 'X')
        oaks.append(lod_faces(o, drop=BACK))
        zpeg = Hj + min(0.06, (zt - Hj) * 0.4)
    # sill beam with a weathered nose, on two brackets
    sh, Ds = 0.085, D + 0.075
    Ls = W + 2 * jw + 2 * 0.07
    prof = [(0.01, 0.0), (-D - 0.004, 0.0), (-Ds, -0.022), (-Ds, -0.064), (-Ds + 0.02, -sh), (0.01, -sh)]
    o = _prism("sill", prof, -Ls / 2, Ls / 2, "X", bevel=0.006)
    sd_ = rng.randrange(1000)
    if Lv < 2:
        densify(o, 0.3)
        displace_noise(o, strength=0.002, scale=0.5, seed=sd_)
    lod_faces(o, drop=BACK)
    assign(o, oak)
    assign_faces(o, endg, lambda c, n: abs(n.x) > 0.9)
    uv_grain(o, 'X')
    oaks.append(o)
    cprof = [(0.01, -sh), (-(Ds - 0.03), -sh), (-(Ds - 0.03), -sh - 0.028)]
    nq = (6, 3, 1)[Lv]
    for i in range(1, nq + 1):
        a = radians(90.0 * i / nq)
        cprof.append((-(Ds - 0.03) * cos(a) - 0.012 * sin(a), -sh - 0.028 - 0.15 * sin(a)))
    cprof.append((0.01, -sh - 0.178))
    for s in (-1, 1):
        xc = s * (W / 2 + jw / 2)
        o = _prism(f"corbel{s}", cprof, xc - 0.04, xc + 0.04, "X", bevel=0.005)
        assign(o, oak)
        uv_grain(o, 'Z')
        oaks.append(lod_faces(o, drop=[BACK, (0, 0, 1)]))
    # proud oak pegs through the joints
    mats = []
    pr = rng.fork("pegs")
    for s in (-1, 1):
        xc = s * (W / 2 + jw / 2)
        for dx in (-0.024, 0.024):
            mats.append(trs((xc + dx + pr.uniform(-0.004, 0.004), -Dl - pr.uniform(-0.002, 0.008), zpeg + pr.uniform(-0.012, 0.012)),
                            (90 + pr.uniform(-5, 5), 0, pr.uniform(-5, 5))))
        mats.append(trs((xc + pr.uniform(-0.01, 0.01), -D - pr.uniform(-0.002, 0.008), Hj - 0.06), (90 + pr.uniform(-5, 5), 0, 0)))
        mats.append(trs((xc + pr.uniform(-0.01, 0.01), -D - pr.uniform(-0.002, 0.008), 0.055), (90 + pr.uniform(-5, 5), 0, 0)))
    if Lv < 2:                                          # far: no pegs
        base = cylinder("peg_base", 0.0115, 0.03, verts=lod_pick(7, 5), bevel=0.002, segments=1)
        if Lv:
            lod_faces(base, drop=(0, 0, -1))            # the end in the post
        assign(base, endg)
        pegs = instances_to_mesh("pegs", base, mats, keep_base=False)
        oaks.append(pegs)
    # beaten lead flashing dressed over the lintel
    fl = [(0.01, zt - 0.004), (0.01, zt + 0.005), (-(Dl + 0.012), zt + 0.0015), (-(Dl + 0.015), zt - 0.032),
          (-(Dl + 0.010), zt - 0.033), (-(Dl + 0.008), zt - 0.004)]
    o = _prism("flashing", fl, -L / 2 - 0.008, L / 2 + 0.008, "X", bevel=0.0)
    sd_ = rng.randrange(1000)
    if Lv < 2:
        with lod_as(0 if Lv == 0 else 1):
            densify(o, 0.09 if not Lv else 0.3)         # the beaten dents are finer than a mid-level mesh can hold
        displace_noise(o, strength=0.0022 if not Lv else 0.0012, scale=0.09 if not Lv else 0.3, seed=sd_)
    lod_faces(o, drop=[BACK, (0, 0, -1)])
    smooth(o, 60)
    uv_box(o)
    assign(o, lead_m)
    leads.append(o)
    return oaks, leads, Ds, zt, L


def _stone_surround(p, rng, hd, W, H, jw, Dl, stone_m, quoins):
    st = []
    Hj = hd.spring
    vt = 0.2
    A = math.asin(min(1.0, (W / 2) / hd.R)) if hd.arched else 0.0
    nC = max(3, int(round(Hj / 0.27)))
    hs = [rng.uniform(0.85, 1.15) for _ in range(nC)]
    tot = sum(hs)
    z = 0.0
    for s in (-1, 1):
        r = rng.fork(f"jamb{s}")
        z = 0.0
        for i in range(nC):
            hh = hs[i] / tot * Hj
            long_ = quoins and ((i + (0 if s < 0 else 1)) % 2 == 0)
            w = jw + (0.085 if long_ else 0.0)
            d = D + r.uniform(-0.003, 0.004)
            if hd.arched and i == nC - 1:
                # skew-back springer: its top is cut on the radial joint of the first voussoir
                zo = Hj + vt * cos(A)
                xi = W / 2 + vt * sin(A)
                pts = [(W / 2, z + 0.002), (W / 2 + w, z + 0.002), (W / 2 + w, zo), (min(xi, W / 2 + w - 0.01), zo), (W / 2, Hj)]
                if s < 0:
                    pts = [(-x, zz) for x, zz in reversed(pts)]
                o = _prism(f"springer{s}", pts, 0.01, -d, "Y", bevel=0.008)
                _roughen(o, r, 0.003, 0.3)
                lod_faces(o, drop=BACK)
            else:
                o = _stone(f"jamb{s}_{i}", (w, d + 0.01, hh - 0.004), (s * (W / 2 + w / 2), -d / 2 + 0.005, z + hh / 2), r)
            st.append(o)
            z += hh
    L = W + 2 * jw + 0.10
    if not hd.arched:
        lh = 0.19 + 0.02 * W
        zt = H + lh
        o = _stone("lintel", (L, Dl + 0.01, lh - 0.003), (0, -Dl / 2 + 0.005, H + lh / 2), rng.fork("lintel"), chamfer=0.01, detail=0.2)
        st.append(o)
        # label mould: weathered hood across the head, returned down at both ends on square stops
        hood = [(0.01, zt), (-Dl, zt), (-(Dl + 0.042), zt + 0.03), (-(Dl + 0.042), zt + 0.05), (0.01, zt + 0.078)]
        Lh = L / 2 + 0.065
        o = _prism("hood", hood, -Lh, Lh, "X", bevel=0.005)
        _roughen(o, rng.fork("hood"), 0.002, 0.4, 0.25)
        st.append(lod_faces(o, drop=BACK))
        for s in (-1, 1):
            drop = lh * 0.62
            o = _stone(f"label{s}", (0.062, D * 0.72, drop), (s * (L / 2 + 0.034), -D * 0.36 + 0.005, zt - drop / 2 + 0.01), rng.fork(f"lab{s}"),
                       chamfer=0.006)
            st.append(o)
            o = _stone(f"stop{s}", (0.085, D * 0.86, 0.075), (s * (L / 2 + 0.034), -D * 0.43 + 0.005, zt - drop - 0.025), rng.fork(f"stp{s}"),
                       chamfer=0.012)
            st.append(o)
        zt += 0.078
    else:
        R = hd.R
        n = max(5, int(round(2 * A * (R + vt / 2) / 0.17)))
        if n % 2 == 0:
            n += 1
        g = 0.0025 / R
        vr = rng.fork("vous")
        for i in range(n):
            a0 = -A + 2 * A * i / n + g
            a1 = -A + 2 * A * (i + 1) / n - g
            key = (i == n // 2)
            ro = R + vt + (0.055 if key else vr.uniform(-0.006, 0.008))
            ri = R - (0.012 if key else 0.0)
            am = (a0 + a1) / 2
            pts = [(ri * sin(a), hd.zc + ri * cos(a)) for a in ((a0, am, a1) if lod() < 2 else (a0, a1))] + \
                  [(ro * sin(a), hd.zc + ro * cos(a)) for a in ((a1, am, a0) if lod() < 2 else (a1, a0))]
            d = Dl + (0.028 if key else vr.uniform(-0.003, 0.004))
            o = _prism(f"vous{i}", pts, 0.01, -d, "Y", bevel=0.008)
            _roughen(o, vr, 0.003, 0.25)
            lod_faces(o, drop=BACK)
            if not key:                                 # far: the joint faces between the voussoirs too (the proud key keeps its cheeks)
                lod_faces(o, drop=[(-cos(a0), 0, sin(a0)), (cos(a1), 0, -sin(a1))], above=1, cos=0.8)
            st.append(o)
        zt = H + vt + 0.055
    # sloped sill on a bed mould
    sh, Ds = 0.10, D + 0.085
    Ls = W + 2 * jw + 2 * 0.06
    prof = [(0.01, 0.0), (-D - 0.004, 0.0), (-Ds, -0.026), (-Ds, -0.07), (-Ds + 0.022, -sh), (0.01, -sh)]
    o = _prism("sill", prof, -Ls / 2, Ls / 2, "X", bevel=0.007)
    _roughen(o, rng.fork("sill"), 0.003, 0.5, 0.2)
    st.append(lod_faces(o, drop=BACK))
    o = _stone("bed", (Ls - 0.14, Ds - 0.05, 0.05), (0, -(Ds - 0.06) / 2 + 0.005, -sh - 0.024), rng.fork("bed"), chamfer=0.012, detail=0.25)
    st.append(o)
    assign(st, stone_m)
    return st, Ds, zt, L


# ------------------------------------------------------------------ shutters

def _shutter(side, p, r, hd, W, H, jw, paint_m, iron, age, on_front, zdog_objs):
    """One plank leaf built on the LEFT (hinge at x = -hx) in its closed position, then swung open and
    mirrored for the right-hand side.  Returns (wood objects, iron objects)."""
    Lv = lod()
    hx = W / 2 + jw + 0.014
    py = -(D + 0.036)
    u0, u1 = 0.012, hx - 0.004

    def X(u):
        return -hx + u

    def ztop(x):
        return hd.top(min(abs(x), W / 2 - 1e-4)) - 0.008

    zb = 0.008
    wood, metal, fixed = [], [], []
    nb = max(2, int(round((u1 - u0) / 0.135)))
    wr = [r.uniform(0.8, 1.2) for _ in range(nb)]
    tot = sum(wr)
    edges = [u0]
    for wv in wr:
        edges.append(edges[-1] + wv / tot * (u1 - u0))
    cut_i = nb // 2
    crook = 0.003 + 0.006 * age
    for i in range(nb):
        a, b = X(edges[i]) + 0.0015, X(edges[i + 1]) - 0.0015
        dz0, dz1 = r.uniform(0, crook), r.uniform(-crook, 0)
        pts = [(a, zb + dz0), (b, zb + dz0), (b, ztop(b) + dz1)]
        if hd.arched:
            pts.append(((a + b) / 2, ztop((a + b) / 2) + dz1))
        pts.append((a, ztop(a) + dz1))
        yo = r.uniform(-0.0018, 0.0018) * (0.5 + age)
        if i == cut_i and p.shutter_cutout != "none":
            cw = min(0.078, (b - a) - 0.045)
            cx = (a + b) / 2
            cz = zb + 0.76 * (ztop(cx) - zb)
            if p.shutter_cutout == "heart":
                hole = _heart(cx, cz, cw, (20, 12, 6)[Lv])
            else:
                hole = [(cx + cw * 0.42, cz), (cx, cz + cw * 0.72), (cx - cw * 0.42, cz), (cx, cz - cw * 0.72)]
            o = _plate(f"sb{side}_{i}", pts, [hole] if (Lv < 2 or on_front) else [], py + BT / 2 + yo, BT, bevel=0.0014)   # far: pierced only where light leaks out
        else:
            o = _prism(f"sb{side}_{i}", pts, py - BT / 2 + yo, py + BT / 2 + yo, "Y", bevel=0.0025)
        uv_grain(o, 'Z')
        wood.append(o)
    # ledges and brace on the room side (the side you see when the leaf is folded back)
    hmin = ztop(X(u0)) - zb
    zl = [zb + 0.15 * hmin, zb + 0.85 * hmin]
    la, lb = (jw + 0.022 if on_front else 0.034), u1 - 0.03
    lw = 0.085
    for k, z in enumerate(zl):
        o = box(f"ledge{side}_{k}", (lb - la, LT, lw), loc=(X((la + lb) / 2), py + BT / 2 + LT / 2, z), bevel=0.005, segments=1)
        uv_grain(o, 'X')
        wood.append(o)
    bwx = min(0.115, (lb - la) * 0.36)
    zq0, zq1 = zl[0] + lw / 2, zl[1] - lw / 2
    if lb - la > 0.2:
        pts = [(X(la + 0.012), zq0), (X(la + 0.012 + bwx), zq0), (X(lb - 0.012), zq1), (X(lb - 0.012 - bwx), zq1)]
        o = _prism(f"brace{side}", pts, py + BT / 2, py + BT / 2 + LT * 0.9, "Y", bevel=0.004)
        uv_grain(o, (lb - la - 0.024 - bwx, 0.0, zq1 - zq0))
        wood.append(o)
    assign(wood, paint_m)
    # nails: heads on the ledges, clenched points on the weather face
    mats = []
    for i in range(nb):
        cu = (edges[i] + edges[i + 1]) / 2
        if cu < la + 0.02 or cu > lb - 0.02:
            continue
        for z in zl:
            for dxn, dzn in ((-0.017, 0.021), (0.017, -0.021)):
                mats.append(trs((X(cu) + dxn + r.uniform(-0.004, 0.004), py + BT / 2 + LT - 0.0005, z + dzn), (-90, 0, 0)))
                mats.append(trs((X(cu) + dxn + r.uniform(-0.004, 0.004), py - BT / 2 + 0.0005, z + dzn), (90, 0, 0)))
    if mats and not Lv:
        base = bolt("nail_base", radius=0.0062, height=0.0042, kind="pyramid")
        assign(base, iron)
        metal.append(instances_to_mesh(f"nails{side}", base, mats, keep_base=False))
    # forged strap hinges, cranked knuckles on pintles
    yf = (py - BT / 2) if on_front else (py + BT / 2 + LT)
    sgn = -1.0 if on_front else 1.0
    rmats = []
    for k, z in enumerate(zl):
        Ls = (u1 - u0) * r.uniform(0.6, 0.7)
        t0, t1 = Ls * 0.8, Ls * 0.8 + 0.014
        pts = [(X(0.0), z - 0.018), (X(t0), z - 0.012), (X(t1), z - 0.023), (X(Ls), z), (X(t1), z + 0.023),
               (X(t0), z + 0.012), (X(0.0), z + 0.018)]
        o = _prism(f"strap{side}_{k}", pts, yf, yf + sgn * 0.0045, "Y", bevel=0.001)
        metal.append(lod_faces(o, keep=(0, sgn, 0)))        # levels: the strap as its face on the boards
        for f in (0.1, 0.36, 0.62, 0.8 + 0.014 / Ls):
            rmats.append(trs((X(Ls * f), yf + sgn * 0.004, z), ((90 if on_front else -90), 0, 0)))
        if Lv >= 2:                                         # far: no knuckles, cranks and pintles
            continue
        metal.append(cylinder(f"knuckle{side}_{k}", 0.0115, 0.042, loc=(X(0), py, z), verts=lod_pick(10, 6)))
        ya, yb2 = sorted((py, yf + sgn * 0.0045))
        metal.append(box(f"crank{side}_{k}", (0.016, yb2 - ya, 0.034), loc=(X(0.002), (ya + yb2) / 2, z), bevel=0.002, segments=1))
        # pintle: pin, arm back to the jamb and a bolted plate on the jamb face (stays with the wall)
        fixed.append(cylinder(f"pin{side}_{k}", 0.006, 0.084, loc=(X(0), py, z - 0.004), verts=lod_pick(8, 4)))
        fixed.append(box(f"pin_arm{side}_{k}", (0.013, -D - py + 0.002, 0.014), loc=(X(0), (py - D) / 2, z - 0.03), bevel=0.002, segments=1))
        fixed.append(lod_faces(box(f"pin_plate{side}_{k}", (0.076, 0.005, 0.05), loc=(X(0.03), -D - 0.0025, z - 0.03), bevel=0.0015, segments=1), drop=BACK))
        for dxp in ((0.03, 0.055) if not Lv else ()):
            fixed.append(bolt(f"pin_bolt{side}_{k}_{dxp}", radius=0.006, height=0.004, loc=(X(dxp), -D - 0.005, z - 0.03),
                              rot=(90, 0, 0), kind="square"))
    if not Lv:
        rb = bolt("rivet_base", radius=0.0058, height=0.0036, kind="dome")
        assign(rb, iron)
        metal.append(instances_to_mesh(f"rivets{side}", rb, rmats, keep_base=False))
    if on_front and side < 0:
        # swing bar across the meeting boards + ring pull
        zb_ = zb + 0.5 * hmin
        metal.append(box("swing_bar", (0.3, 0.005, 0.024), loc=(X(u1 - 0.03), py - BT / 2 - 0.006, zb_), bevel=0.0015, segments=1))
        if not Lv:
            metal.append(bolt("bar_pivot", radius=0.009, height=0.006, loc=(X(u1 - 0.15), py - BT / 2 - 0.0085, zb_), rot=(90, 0, 0), kind="dome"))
        if Lv < 2:
            fixed.append(sweep("bar_keep", [(-X(u1) + 0.09, py - BT / 2, zb_ - 0.02), (-X(u1) + 0.09, py - BT / 2 - 0.014, zb_ - 0.02),
                                            (-X(u1) + 0.09, py - BT / 2 - 0.014, zb_ + 0.02)], profile=0.003, sides=5, smooth_path=False))
    if on_front and Lv < 2:
        zr = zb + 0.42 * hmin
        metal.append(torus(f"ring{side}", major=0.022, minor=0.0032, loc=(X(u1 - 0.085), py - BT / 2 - 0.006, zr - 0.022),
                           rot=(90, 0, 0), major_seg=14, minor_seg=5))
        if not Lv:
            metal.append(bolt(f"ring_boss{side}", radius=0.011, height=0.007, loc=(X(u1 - 0.085), py - BT / 2, zr), rot=(90, 0, 0), kind="dome"))
    assign(metal, iron)
    assign(fixed, iron)

    # weathering (painted in the closed position: z does not change when the leaf swings)
    sd = r.randrange(1000)
    chips = _mx(mask_mul(height_ramp(zb, zb + 0.32), noise_mask(0.035, sd, 0.3, 0.62), 0.45 + 0.75 * age),
                mask_mul(noise_mask(0.05, sd + 1, 0.6, 0.78), 0.9 * age))
    paint(wood, wear=chips,
          grime=mask_mul(_mx(height_ramp(zb, zb + 0.5), noise_mask(0.3, sd + 2, 0.4, 0.8)), 0.25 + 0.5 * age),
          special=mask_mul(noise_mask(0.45, sd + 3, 0.25, 0.8), 0.3 + 0.6 * age))
    rust = mask_mul(noise_mask(0.035, sd + 4, 0.3, 0.7), 0.25 + 0.75 * age)
    paint(metal, special=rust, grime=0.15 + 0.2 * age)
    paint(fixed, special=rust, grime=0.15 + 0.2 * age)

    # swing: closed -> folded right back until the free edge touches the wall
    splay = degrees(math.asin(min(0.9, (D + 0.036 - BT / 2 - 0.007) / u1)))
    ang = p.shutter_open * (180.0 + splay)
    if p.shutter_open > 0.9:
        ang -= r.uniform(0.0, 3.0) if side < 0 else 0.0
    Mo = _rot_about(-hx, py, -ang)
    Mm = Matrix.Diagonal((-1.0 if side > 0 else 1.0, 1.0, 1.0, 1.0))
    # shutter dog: spike in the wall with an S-shaped keeper turned up in front of the leaf
    dogs = []
    if p.shutter_open > 0.9 and Lv < 2:                  # far: no shutter dogs
        zd = zb - 0.014
        pw = Mo @ Vector((X(0.74 * u1), py + BT / 2 + 0.005, zd))
        yS = pw.y - 0.004
        spike = cylinder(f"dog_spike{side}", 0.0055, abs(yS) + 0.012, loc=(pw.x, (yS - 0.012) / 2 + 0.006, zd), rot=(90, 0, 0), verts=6)
        path = []
        ca, sa = cos(radians(14)), sin(radians(14))
        ns = (14, 8, 8)[Lv]
        for i in range(ns + 1):
            t = -1.0 + 2.0 * i / ns
            dx, dz = 0.02 * sin(pi * t) * (1.0 - 0.25 * t * t), 0.058 * t
            path.append((pw.x + dx * ca - dz * sa, yS - 0.003, zd + dx * sa + dz * ca))
        if Lv:                                              # the S as a flat strap; no boss
            with lod_as(0):
                sdog = sweep(f"dog_s{side}", path, profile=[(-0.006, -0.002), (0.006, -0.002), (0.006, 0.002), (-0.006, 0.002)], up=(0, 1, 0), smooth_path=False)
            dogs = [spike, lod_faces(sdog, drop=BACK)]
        else:
            sdog = sweep(f"dog_s{side}", path, profile=rounded_rect(0.012, 0.0045, 0.0015, 1), up=(0, 1, 0), smooth_path=False)
            boss = bolt(f"dog_boss{side}", radius=0.008, height=0.005, loc=(pw.x, yS - 0.005, zd), rot=(90, 0, 0), kind="dome")
            dogs = [spike, sdog, boss]
        assign(dogs, iron)
        paint(dogs, special=rust, grime=0.3)
    _xform(wood + metal, Mm @ Mo)
    _xform(fixed + dogs, Mm)
    # a leaf folded back lies against the wall: its back goes, far away it is its face only.  A closed leaf stands
    # proud of the wall: it keeps its back at lod 1 (back faces are what casts the shadow) and its board edges far
    # away (the lit glass would show through the joints).  A half-open leaf is seen from both sides.
    if p.shutter_open > 0.9:
        lod_faces(wood, keep=OUT, above=1)
        lod_faces(wood, drop=BACK, cos=0.9)
    elif p.shutter_open < 0.1:
        lod_faces(wood, keep=[OUT, (1, 0, 0), (-1, 0, 0)], above=1)
    return wood, metal + fixed + dogs


# ------------------------------------------------------------------ build

def build(p, rng):
    Lv = lod()
    W, H = p.width, p.height
    age = p.age
    hd = _Head(W, H, p.head == "arched", p.arch_rise)
    stone_sur = p.surround == "stone"
    jw = 0.15 if stone_sur else 0.11
    Dl = D + (0.058 if p.shutters else 0.03)          # lintel projects further so closed shutters nest under it

    oak = mat("wood_oak", tint=p.timber)
    endg = mat("wood_endgrain", tint=p.timber)
    fpaint = mat("wood_painted", tint=p.frame_paint)
    spaint = mat("wood_painted", tint=p.shutter_paint)
    iron = mat("iron_wrought")
    lead_m = mat("lead")
    stone_m = mat("stone_smooth", tint=p.stone)
    if p.lit:
        # NB: window_lit needs an explicit tint - without one the engine renders it without emission
        glass = mat("window_lit", tint="#ffc27a", glow=p.glow, curtain=p.curtain)
        glass2 = mat("window_lit", tint="#f0cf8e", glow=p.glow * 0.7, curtain=p.curtain)
    else:
        glass = mat("window_dark")
        glass2 = None
    seedval = rng.fork("room").uniform(0.03, 0.97)
    uv_off = (rng.fork("room_uv").uniform(0.0, 2.9), 0.45)

    # ---- surround -------------------------------------------------------------------------
    if stone_sur:
        stones, Ds, ztop_all, Lh = _stone_surround(p, rng.fork("stone"), hd, W, H, jw, Dl, stone_m, quoins=not p.shutters)
        sd = rng.randrange(1000)
        paint(stones,
              grime=mask_mul(_mx(height_ramp(-0.25, 0.25), mask_mul(height_ramp(H + 0.4, H - 0.1), 0.5),
                                 noise_mask(0.25, sd, 0.45, 0.8)), 0.3 + 0.6 * age),
              special=mask_mul(_mx(mask_mul(height_ramp(-0.12, 0.04), facing((0, 0, 1), 1.0)), mask_mul(facing((0, 0, 1), 2.0), 0.7)),
                               noise_mask(0.12, sd + 1, 0.35, 0.65), 0.3 + 0.9 * age))
    else:
        oaks, leads, Ds, ztop_all, Lh = _timber_surround(p, rng.fork("timber"), hd, W, H, jw, Dl, oak, endg, lead_m)
        for o in oaks:
            if len(o.data.materials) == 0:
                assign(o, oak)
        sd = rng.randrange(1000)
        paint(oaks,
              grime=mask_mul(_mx(height_ramp(-0.3, 0.2), mask_mul(noise_mask(0.2, sd, 0.4, 0.75), 0.8)), 0.3 + 0.5 * age),
              wear=mask_mul(noise_mask(0.08, sd + 1, 0.55, 0.8), 0.5 * age))
        paint(leads, grime=mask_mul(noise_mask(0.1, sd + 2, 0.3, 0.7), 0.4 + 0.4 * age))

    # ---- frame: one joinery frame, lights divided by mullions (and a transom) ---------------
    n = max(1, min(p.lights, int((W - 2 * FW + MW) / (0.17 + MW))))
    lwid = (W - 2 * FW - (n - 1) * MW) / n
    zfb = 0.06                                           # top of the frame's sill member
    transom = p.transom and H >= 1.0
    zt = 0.0
    if transom:
        zt = 0.69 * H
        if hd.arched:
            zt = min(zt, hd.top(W / 2 - FW, FW) - 0.13 - TH / 2)
    cols = []
    cells = []                                           # (region, column, is_top)
    for i in range(n):
        x0 = -W / 2 + FW + i * (lwid + MW)
        x1 = x0 + lwid
        cols.append((x0, x1))
        if transom:
            cells.append((_Reg(hd, x0, x1, zfb, zt - TH / 2, FW), i, False))
            cells.append((_Reg(hd, x0, x1, zt + TH / 2, None, FW), i, True))
        else:
            cells.append((_Reg(hd, x0, x1, zfb, None, FW), i, False))
    painted = []
    outer = _Reg(hd, -W / 2 + 0.003, W / 2 - 0.003, 0.0, None, 0.003).poly(18)
    frame = _plate("frame", outer, [c[0].inset(-0.003).poly((8, 6, 4)[Lv]) for c in cells], 0.0, FD, bevel=0.003)
    uv_box(frame)
    lod_faces(frame, drop=BACK)                             # its back is in the wall
    painted.append(frame)
    # planted mullion mouldings (splayed), butting under the head
    for i in range(n - 1):
        xm = cols[i][1] + MW / 2
        zt_m = hd.top(xm, FW) + 0.003
        prof = [(xm - MW / 2 + 0.005, -FD + 0.003), (xm + MW / 2 - 0.005, -FD + 0.003), (xm + 0.011, -FD - 0.017), (xm - 0.011, -FD - 0.017)]
        o = _prism(f"mullion{i}", prof, 0.047, zt_m, "Z", bevel=0.0025)
        uv_grain(o, 'Z')
        painted.append(lod_faces(o, drop=[BACK, (0, 0, 1), (0, 0, -1)]))
    if transom:
        prof = [(-FD + 0.003, zt - TH / 2 + 0.004), (-FD - 0.02, zt - TH / 2 + 0.01), (-FD - 0.027, zt + 0.006),
                (-FD - 0.024, zt + 0.012), (-FD + 0.003, zt + TH / 2 - 0.003)]
        o = _prism("transom", prof, -W / 2 + 0.012, W / 2 - 0.012, "X", bevel=0.002)
        uv_grain(o, 'X')
        painted.append(lod_faces(o, drop=[BACK, (1, 0, 0), (-1, 0, 0)]))
    # sloped sub-sill of the frame, throwing water onto the sill beam
    prof = [(-FD + 0.004, 0.0), (-(D - 0.014), 0.0), (-(D - 0.014), 0.017), (-FD + 0.004, 0.05)]
    o = _prism("subsill", prof, -W / 2 + 0.003, W / 2 - 0.003, "X", bevel=0.0035)
    uv_grain(o, 'X')
    painted.append(lod_faces(o, drop=[BACK, (0, 0, -1), (1, 0, 0), (-1, 0, 0)]))

    # ---- sashes, glazing, hinges ------------------------------------------------------------
    irons = []
    open_col = -1
    opening = p.open if not (p.shutters and p.shutter_open < 0.6) else 0.0
    if opening > 0.02:
        open_col = rng.fork("open").randrange(n)
    leaf_objs = []
    hinge_x, hinge_dir = 0.0, 1
    grng = rng.fork("glass")
    for ci, (reg, col, is_top) in enumerate(cells):
        x0, x1 = cols[col]
        moving = (col == open_col and not is_top)
        mine = []
        sash = _plate(f"sash{ci}", reg.inset(0.0035).poly((8, 6, 4)[Lv]), [reg.inset(SW - 0.002).poly((8, 6, 4)[Lv])], -0.016, 0.03, bevel=0.002)
        uv_box(sash)
        if not moving:                                      # a shut casement shows its face and the rebate towards the glass
            lod_faces(sash, keep=OUT, above=1)
            lod_faces(sash, drop=BACK)
        painted.append(sash)
        mine.append(sash)
        g_m, g_m2 = glass, glass2
        if moving:
            g_m, g_m2 = mat("glass_clear", opacity=0.22), None
        gl = _glazing(f"l{ci}", reg.inset(SW - 0.005), p.glazing, p.panes_x,
                      p.panes_y if not is_top else max(1, int(round(p.panes_y * 0.34))),
                      grng, age, g_m, g_m2, seedval, uv_off, lead_m, fpaint, iron,
                      hidden=bool(p.shutters and p.shutter_open < 0.1))
        for o in gl:
            if o.name.endswith("_cames") and p.glazing == "bars":
                painted.append(o)
            elif o.name.endswith("_cames"):
                paint(o, grime=0.45 + 0.3 * age, wear=0.1)
            elif "_saddle" in o.name:
                irons.append(o)
        mine += gl
        # butt hinges on the outer side of every opening casement
        left = col < n / 2.0
        xh = (x0 - 0.002) if left else (x1 + 0.002)
        if not is_top:
            zlo, zhi = reg.z0, reg.ztop(xh)
            for f in ((0.16, 0.84) if (Lv < 2 or moving) else ()):       # far: no butt hinges
                k = cylinder(f"hinge{ci}_{f}", 0.0058, 0.058, loc=(xh, -FD - 0.004, zlo + (zhi - zlo) * f), verts=lod_pick(8, 4))
                kp = box(f"hinge_leaf{ci}_{f}", (0.03, 0.003, 0.05), loc=(xh + (0.012 if left else -0.012), -0.0465, zlo + (zhi - zlo) * f),
                         bevel=0.001, segments=1)
                irons += [k, kp]
                assign([k, kp], iron)
                mine.append(kp)
        if moving:
            leaf_objs = mine
            hinge_x, hinge_dir = xh, (-1 if left else 1)
            # the room seen through the open casement
            bp = reg.inset(-0.001).poly(8)
            back = mesh_object("room", [(x, -0.006, z) for x, z in bp], [tuple(range(len(bp)))])
            assign(back, glass)
            flat(back)
            uv_planar(back, (1, 0, 0), (0, 0, 1), offset=uv_off)
            paint(back, grime=0.85, special=seedval)

    sd = rng.randrange(1000)
    paint(painted,
          wear=_mx(mask_mul(height_ramp(0.0, 0.3), noise_mask(0.03, sd, 0.4, 0.7), 0.3 + 0.6 * age),
                   mask_mul(noise_mask(0.06, sd + 1, 0.7, 0.85), 0.55 * age)),
          grime=mask_mul(_mx(height_ramp(0.0, 0.35), noise_mask(0.2, sd + 2, 0.45, 0.8)), 0.25 + 0.5 * age),
          special=mask_mul(noise_mask(0.3, sd + 3, 0.3, 0.8), 0.2 + 0.4 * age))
    assign(painted, fpaint)
    rust = mask_mul(noise_mask(0.03, sd + 4, 0.3, 0.7), 0.3 + 0.7 * age)
    paint(irons, special=rust, grime=0.2)

    if leaf_objs:
        ang = opening * 78.0
        M = _rot_about(hinge_x, -FD - 0.004, hinge_dir * ang)
        _xform(leaf_objs, M)
        # casement stay from the sub-sill to the bottom rail
        reg = [c for c in cells if c[1] == open_col and not c[2]][0][0]
        xf = reg.x1 - 0.06 if hinge_dir < 0 else reg.x0 + 0.06
        pe = M @ Vector((xf, -0.047, reg.z0 + 0.012))
        ps = Vector((xf + hinge_dir * 0.0, -(D - 0.03), 0.034))
        stay = sweep("stay", [tuple(ps), tuple(ps + Vector((0, -0.004, 0.012))), tuple(pe + Vector((0, 0, 0.004))), tuple(pe)],
                     profile=0.0032, sides=6, smooth_path=False)
        pin = cylinder("stay_pin", 0.004, 0.022, loc=(ps.x, ps.y, ps.z - 0.004), verts=6)
        assign([stay, pin], iron)
        paint([stay, pin], special=rust, grime=0.2)

    # ---- shutters ------------------------------------------------------------------------
    if p.shutters:
        on_front = p.shutter_open < 0.5
        for side in (-1, 1):
            _shutter(side, p, rng.fork(f"shutter{side}"), hd, W, H, jw, spaint, iron, age, on_front, None)

    # ---- light + sockets -------------------------------------------------------------------
    # By default a lit window is only its glowing glazing (no light socket), so a house front can carry
    # many of them for free.  `intensity` > 0 adds the spill of a bright room: one soft point light
    # floated in front of the glass (in front of the open casement if there is one), with shadow_near
    # beyond the whole window so it fills without casting the window's own shadow.  Closed shutters
    # keep the light in: only the joints and the cut-outs glow.
    if p.lit and p.intensity > 0 and not (p.shutters and p.shutter_open < 0.3):
        k = clamp(W * H / 1.5, 0.8, 1.25)
        lx = 0.5 * (cols[open_col][0] + cols[open_col][1]) if open_col >= 0 else 0.0
        light("window", (lx, -(0.42 + 0.14 * W + 0.3 * opening), H * 0.5), color="#ffb768", intensity=p.intensity * k,
              range=5.5, flicker=0.05, shadow_near=2.5)
    socket("planter", "attach", (0.0, -(Ds - 0.055), 0.0), role="planter", width=round(W, 3))
    socket("lintel", "attach", (0.0, -Dl, ztop_all), role="sign")
