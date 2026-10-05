"""Grand civic hall closing the second square: a long pale-limestone range built at true size.
A projecting ashlar undercroft with an arcade of pointed stained-glass arches carries a balustraded
terrace; behind it rises the tall hall storey (slender traceried windows between stepped pilaster
buttresses, string course, corbelled cornice, steep tiled roof with lucarnes and iron cresting, coped
gables).  A slender square clock tower stands through the front (louvred belfry arch, blind arcade,
corbelled gallery, four lit dials under tiled gablets, corner pinnacles, tiled spire with gilt ball),
octagonal end pavilions with conical roofs and rooster vanes finish the range, and a gabled porch
with a pointed double door, rose window, pinnacled buttresses and lanterns stands up broad steps.
Every facade is built in a local 'frame' (x along the wall, -Y outwards) and stamped round the plan;
windows are kits (moulded surround, plate tracery, leaded glass, saddle bars) instanced per opening.
"""
from math import ceil, sqrt, tan
from types import SimpleNamespace

from tk import *

META = {"name": "Grand civic hall with clock tower", "class": "landmark", "mount": "ground",
        "focus": [4.0, -6.0, 6.5]}
PARAMS = {
    "width": {"default": 38.0, "min": 30.0, "max": 44.0, "desc": "length of the range along the square (m)"},
    "depth": {"default": 12.0, "min": 10.0, "max": 16.0, "desc": "depth of the upper hall body, y = 0 .. depth (m)"},
    "wing_storey_height": {"default": 7.6, "min": 6.2, "max": 9.0, "desc": "height of the upper hall storey above the terrace (m)"},
    "arcade_height": {"default": 4.8, "min": 4.2, "max": 5.4, "desc": "height of the lower arcade storey / terrace level (m)"},
    "tower_height": {"default": 25.0, "min": 20.0, "max": 28.0, "desc": "ground to the top of the tower masonry (m)"},
    "spire_height": {"default": 11.0, "min": 8.0, "max": 14.0, "desc": "height of the tiled spire (m)"},
    "tower_width": {"default": 5.0, "min": 4.4, "max": 6.0, "desc": "side of the square clock tower (m)"},
    "tower_offset": {"default": 0.0, "min": -11.0, "max": 11.0, "desc": "x of the tower centre (m)"},
    "porch_offset": {"default": 8.0, "min": -14.0, "max": 14.0, "desc": "x of the entrance porch along the front (m)"},
    "porch_width": {"default": 7.0, "min": 5.6, "max": 8.4, "desc": "width of the gabled entrance porch (m)"},
    "terrace_depth": {"default": 5.0, "min": 4.0, "max": 7.0, "desc": "projection of the arcade storey and its terrace (m)"},
    "end_pavilions": {"default": "both", "choices": ["both", "left", "right", "none"], "desc": "octagonal end pavilions"},
    "bays": {"default": 11, "min": 7, "max": 15, "desc": "window bays of the upper hall front"},
    "lucarnes": {"default": 4, "min": 0, "max": 8, "desc": "lucarnes on the front roof slope"},
    "roof_pitch": {"default": 50.0, "min": 40.0, "max": 58.0, "desc": "main roof pitch (deg)"},
    "time": {"default": "10:10", "desc": "time shown by the clock dials (HH:MM)"},
    "stone": {"default": "#f3ead6", "desc": "tint of the dressed limestone trim"},
    "glass": {"default": "violet", "choices": ["violet", "ruby", "emerald", "amber"], "desc": "stained glass palette"},
    "roof": {"default": "clay", "choices": ["clay", "slate", "copper"], "desc": "roof covering"},
    "window_glow": {"default": 3.4, "min": 0.0, "max": 6.0, "desc": "emissive strength of the stained glass"},
    "dial_glow": {"default": 3.2, "min": 0.0, "max": 12.0, "desc": "emissive strength of the clock dials and lantern panes"},
    "lantern_intensity": {"default": 300.0, "min": 0.0, "max": 600.0, "desc": "terrace flood lantern intensity (cd); porch / arcade lanterns scale with it"},
    "moss": {"default": 0.25, "min": 0.0, "max": 1.0, "desc": "moss / lichen on stone and roofs"},
}
VARIANTS = {
    "west_tower_slate": {"width": 42.0, "tower_offset": -10.0, "porch_offset": 6.0, "end_pavilions": "right", "roof": "slate",
                         "glass": "emerald", "tower_height": 28.0, "spire_height": 14.0, "bays": 13, "time": "7:25",
                         "stone": "#e9e2d2", "lucarnes": 6, "roof_pitch": 56.0, "seed": 4},
    "compact_ruby": {"width": 30.0, "depth": 10.0, "bays": 8, "end_pavilions": "none", "glass": "ruby", "tower_height": 22.0,
                     "spire_height": 10.0, "tower_width": 4.4, "wing_storey_height": 6.4, "arcade_height": 4.3, "porch_offset": -7.0,
                     "porch_width": 6.0, "lucarnes": 0, "roof_pitch": 42.0, "terrace_depth": 4.0, "moss": 0.7, "time": "4:50", "seed": 9},
    "copper_long": {"width": 44.0, "depth": 15.0, "bays": 15, "roof": "copper", "glass": "amber", "tower_offset": 9.0,
                    "porch_offset": -9.0, "end_pavilions": "left", "terrace_depth": 6.5, "tower_width": 6.0, "time": "3:40",
                    "stone": "#f6e6c4", "wing_storey_height": 8.6, "arcade_height": 5.2, "lucarnes": 8, "seed": 15},
}

TH_WALL = 0.5
PALETTES = {"violet": ("#8a55e0", "#3f6fe6", "#ffb04a"), "ruby": ("#d2363c", "#3d62cc", "#ffc45a"),
            "emerald": ("#2fa36c", "#2b86c2", "#ffd070"), "amber": ("#e8922c", "#c9481f", "#f6d68c")}
GLOW_K = {"violet": (1.0, 1.0, 0.5), "ruby": (1.0, 1.0, 0.5), "emerald": (1.0, 1.0, 0.5), "amber": (0.6, 0.9, 0.4)}


class Frame:
    """A facade frame: canonical x runs along the wall, the wall face is y = 0, outwards is -Y."""

    def __init__(self, ox, oy, yaw):
        self.ox, self.oy, self.yaw = ox, oy, yaw
        self.B = trs((ox, oy, 0.0), (0, 0, yaw))

    def M(self, x=0.0, y=0.0, z=0.0, rot=(0, 0, 0), scale=1.0):
        return self.B @ trs((x, y, z), rot, scale)

    def pt(self, x, y, z):
        v = self.B @ Vector((x, y, z))
        return (v.x, v.y, v.z)

    def put(self, *objs):
        for o in objs:
            if o is None:
                continue
            if isinstance(o, (list, tuple)):
                self.put(*o)
                continue
            apply_transform(o)
            place(o, loc=(self.ox, self.oy, 0.0), rot=(0, 0, self.yaw))


def bx(name, x0, x1, y0, y1, z0, z1, bev=0.0):
    o = box(name, (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), bevel=bev, segments=1)
    apply_transform(o)
    return o


def prism(name, pts, a0, a1, axis="x"):
    """Extrude a 2D polygon along a world axis: axis x -> pts are (y, z); y -> (x, z); z -> (x, y)."""
    def P(u, v, a):
        return (a, u, v) if axis == "x" else ((u, a, v) if axis == "y" else (u, v, a))
    n = len(pts)
    verts = [P(u, v, a0) for u, v in pts] + [P(u, v, a1) for u, v in pts]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def u_ring(outer, inner):
    """Closed polygon of an arch band standing on its two feet (no sill)."""
    return list(outer) + list(reversed(inner))


def build(p, rng):
    W, D, T = p.width, p.depth, p.terrace_depth
    hl = p.arcade_height
    ze = hl + p.wing_storey_height
    TH, SH, tw = p.tower_height, p.spire_height, p.tower_width
    pavL = p.end_pavilions in ("both", "left")
    pavR = p.end_pavilions in ("both", "right")
    Rp, pin = 3.0, 1.3
    ap = Rp * cos(radians(22.5))
    pcx, pcy = W / 2 - pin, pin
    tx = clamp(p.tower_offset, -(W / 2 - 9.5), W / 2 - 9.5)
    tyc = -1.6 + tw / 2
    ov = 0.45

    ASH = mat("ashlar_limestone")
    TRIM = mat("stone_smooth", p.stone)
    ROOF = mat({"clay": "roof_tile_clay", "slate": "roof_slate", "copper": "copper_patina"}[p.roof])
    G = [mat("glass_stained", c, glow=p.window_glow * k) for c, k in zip(PALETTES[p.glass], GLOW_K[p.glass])]
    LIT = mat("emissive", "#ffcf8a", glow=p.dial_glow)
    GILT = mat("metal_plain", "#d4a84b", rough=0.4)
    DARK = mat("iron_wrought")
    WOOD = mat("wood_oak", "#5b3e29")

    stone, roofs, metal = [], [], []

    def ash(o):
        uv_box(o, offset=None)
        assign(o, ASH)
        stone.append(o)
        return o

    def trim(o):
        assign(o, TRIM)
        stone.append(o)
        return o

    def dark(o):
        assign(o, DARK)
        metal.append(o)
        return o

    def inst(name, base, Ms, reg=None, jitter=True):
        apply_transform(base)
        if not Ms:
            delete(base)
            return None
        o = instances_to_mesh(name, base, Ms, uv_jitter=jitter, keep_base=False)
        if reg is not None:
            reg.append(o)
        return o

    def wall(name, x0, x1, z0, z1, holes=(), th=TH_WALL, outline=None, dens=1.5):
        outer = outline or [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        o = shape_extrude(name, outer, list(holes), depth=th, loc=(0, th, 0), rot=(90, 0, 0))
        apply_transform(o)
        densify(o, dens)
        return ash(o)

    def mould(name, prof, x0, x1, m="trim"):
        o = prism(name, prof, x0, x1, "x")
        return trim(o) if m == "trim" else ash(o)

    def roof_quad(name, p00, p10, p11, p01, thick=0.0, step=1.6):
        a, b, c, d = Vector(p00), Vector(p10), Vector(p11), Vector(p01)
        nu = max(1, int(round(max((b - a).length, (c - d).length) / step)))
        nv = max(1, int(round((d - a).length / step)))
        verts, faces = [], []
        for j in range(nv + 1):
            l, r = a.lerp(d, j / nv), b.lerp(c, j / nv)
            for i in range(nu + 1):
                verts.append(tuple(l.lerp(r, i / nu)))
        for j in range(nv):
            for i in range(nu):
                k = j * (nu + 1) + i
                faces.append((k, k + 1, k + nu + 2, k + nu + 1))
        o = mesh_object(name, verts, faces)
        u = (b - a).normalized()
        nrm = u.cross(d - a).normalized()
        v = nrm.cross(u)
        uv_planar(o, u_axis=tuple(u), v_axis=tuple(v))
        if thick > 0:
            solidify(o, thickness=thick, offset=-1.0)
        assign(o, ROOF)
        roofs.append(o)
        return o

    def pyramid_roof(name, cx, cy, n, rot_deg, rings):
        """n-sided tiled roof from rings [(circumradius, z), ...] bottom to top (planar UV per face)."""
        for i in range(n):
            a0, a1 = radians(rot_deg + i * 360.0 / n), radians(rot_deg + (i + 1) * 360.0 / n)
            for (r0, z0), (r1, z1) in zip(rings[:-1], rings[1:]):
                roof_quad(f"{name}_{i}", (cx + r0 * cos(a0), cy + r0 * sin(a0), z0), (cx + r0 * cos(a1), cy + r0 * sin(a1), z0),
                          (cx + r1 * cos(a1), cy + r1 * sin(a1), z1), (cx + r1 * cos(a0), cy + r1 * sin(a0), z1), step=1.4)

    # ------------------------------------------------------------------ stained glass + window kits
    def glass_panel(name, w, h, sp, y, gm, var, tracery, z0=0.0, r=None):
        cols = max(4, int(round((w + 0.1) / 0.155)))
        cw = (w + 0.1) / cols
        rows = int(ceil((h + 0.06 - z0) / cw))
        buckets = [([], []), ([], []), ([], [])]
        rr = r or rng.fork(name)
        for j in range(rows):
            for i in range(cols):
                xa_, za_ = -w / 2 - 0.05 + i * cw, z0 + j * cw
                cx_, cz_ = xa_ + cw / 2, za_ + cw / 2
                if cz_ > sp:
                    d = hypot(cx_, cz_ - (sp + 0.42 * w))
                    k = 2 if d < 0.1 * w else ((var % 2) if d < 0.2 * w else (1 if (i + j) % 2 else 0))
                else:
                    lx = abs(cx_) - w / 4 if tracery else cx_
                    half = w / 4 if tracery else w / 2
                    pz = ((cz_ + 0.35 * var) % 1.3) - 0.65
                    if hypot(lx, pz) < 0.17:
                        k = 2
                    elif (i + j + var) % 4 == 0 or (i - j - var) % 4 == 0:
                        k = 1                                   # diamond quarry lattice
                    elif rr.chance(0.05):
                        k = 2
                    else:
                        k = 0
                vs, fs = buckets[k]
                n = len(vs)
                vs += [(xa_, y, za_), (xa_ + cw, y, za_), (xa_ + cw, y, za_ + cw), (xa_, y, za_ + cw)]
                fs.append((n, n + 1, n + 2, n + 3))
        objs = []
        for k, (vs, fs) in enumerate(buckets):
            if fs:
                o = mesh_object(f"{name}_{k}", vs, fs)
                assign(o, G[gm[k]])
                objs.append(o)
        return join(objs, name)

    kits = []

    def make_kit(name, w, h, gm=(0, 1, 2), tracery=True, fill="glass", band=0.16, rise=0.8):
        k = SimpleNamespace(name=name, w=w, h=h, spring=h - rise * w, parts=[], glass=[], Ms=[])
        sp = k.spring
        inner = arch_pts(w, h, sp, "pointed", 12)
        outer = [(x, z - band) for x, z in arch_pts(w + 2 * band, h + 2.3 * band, sp + band, "pointed", 12)]
        ring = shape_extrude(name + "_ring", outer, [inner], depth=0.11, loc=(0, 0, 0), rot=(90, 0, 0), bevel=0.02)
        assign(ring, TRIM)
        sill = prism(name + "_sill", [(0.05, -band - 0.03), (-0.21, -band - 0.03), (-0.21, -band + 0.05), (-0.11, 0.0), (0.05, 0.03)],
                     -w / 2 - band - 0.07, w / 2 + band + 0.07, "x")
        assign(sill, TRIM)
        k.parts += [ring, sill]
        gy = 0.22
        if tracery:
            m = 0.085
            lw = (w - 3 * m) / 2
            lh = sp - m + 0.75 * lw
            holes = [[(x + sx * (lw / 2 + m / 2), z + m) for x, z in arch_pts(lw, lh, lh - 0.8 * lw, "pointed", 8)] for sx in (-1, 1)]
            holes.append(circle_pts(0.18 * w, 12, center=(0, sp + 0.42 * w)))
            tr = shape_extrude(name + "_trac", arch_pts(w + 0.08, h + 0.05, sp, "pointed", 12), holes, depth=0.1, loc=(0, 0.2, 0), rot=(90, 0, 0))
            assign(tr, TRIM)
            k.parts.append(tr)
            gy = 0.15
        if fill == "glass":
            for v in range(2):
                k.glass.append(glass_panel(f"{name}_glass{v}", w, h, sp, gy, gm, v, tracery))
            bars, z = [], 0.55
            while z < sp + 0.05:
                bars.append(bx("bar", -w / 2 - 0.02, w / 2 + 0.02, gy - 0.04, gy - 0.02, z - 0.012, z + 0.012))
                z += 0.6
            if bars:
                b = join(bars, name + "_bars")
                assign(b, DARK)
                k.parts.append(b)
        else:
            sl, z = [bx("back", -w / 2 - 0.05, w / 2 + 0.05, 0.45, 0.47, 0.0, h + 0.03)], 0.16
            while z < h:
                sl.append(box("slat", (w + 0.08, 0.24, 0.03), (0, 0.32, z), rot=(35, 0, 0)))
                z += 0.27
            b = join(sl, name + "_louvre")
            uv_grain(b, axis="X")
            assign(b, WOOD)
            k.parts.append(b)
        k.hole = lambda x, z: [(px_ + x, pz_ + z) for px_, pz_ in arch_pts(w, h, sp, "pointed", 12)]
        kits.append(k)
        return k

    def make_rose(name, r):
        k = SimpleNamespace(name=name, r=r, parts=[], glass=[], Ms=[])
        ring = shape_extrude(name + "_ring", circle_pts(r + 0.17, 28), [circle_pts(r, 28)], depth=0.11, loc=(0, 0, 0), rot=(90, 0, 0), bevel=0.02)
        assign(ring, TRIM)
        holes = [circle_pts(0.27 * r, 10, center=(0.62 * r * cos(radians(30 + a * 60)), 0.62 * r * sin(radians(30 + a * 60)))) for a in range(6)]
        holes.append(circle_pts(0.25 * r, 10))
        tr = shape_extrude(name + "_trac", circle_pts(r + 0.04, 28), holes, depth=0.1, loc=(0, 0.2, 0), rot=(90, 0, 0))
        assign(tr, TRIM)
        k.parts += [ring, tr]
        n = 10
        cw = (2 * r + 0.1) / n
        buckets = [([], []), ([], []), ([], [])]
        for j in range(n):
            for i in range(n):
                xa_, za_ = -r - 0.05 + i * cw, -r - 0.05 + j * cw
                cx_, cz_ = xa_ + cw / 2, za_ + cw / 2
                d = hypot(cx_, cz_)
                kk = 2 if d < 0.27 * r else (1 if d > 0.9 * r else int((degrees(atan2(cz_, cx_)) + 360) / 60) % 2)
                vs, fs = buckets[kk]
                m = len(vs)
                vs += [(xa_, 0.15, za_), (xa_ + cw, 0.15, za_), (xa_ + cw, 0.15, za_ + cw), (xa_, 0.15, za_ + cw)]
                fs.append((m, m + 1, m + 2, m + 3))
        objs = []
        for kk, (vs, fs) in enumerate(buckets):
            if fs:
                o = mesh_object(f"{name}_g{kk}", vs, fs)
                assign(o, G[kk])
                objs.append(o)
        k.glass.append(join(objs, name + "_glass"))
        k.hole = lambda x, z: circle_pts(r, 24, center=(x, z))
        kits.append(k)
        return k

    # ------------------------------------------------------------------ small repeated bases
    def lantern_body(z=0.0, scale=1.0):
        o = [dark(bx("l_plate", -0.11, 0.11, -0.11, 0.11, 0.0, 0.025))]
        g = cylinder("l_glass", radius=0.135, radius2=0.185, depth=0.3, loc=(0, 0, 0.175), rot=(0, 0, 45), verts=4)
        # CLEARANCE RULE: the glowing body has no ends (its foot lay in the plane of the plate's top, its head 3 mm inside the rim)
        lod_faces(g, drop=[(0, 0, 1), (0, 0, -1)], above=-1, cos=0.9)
        assign(g, LIT)
        o.append(g)
        for sx in (-1, 1):
            for sy in (-1, 1):
                o.append(dark(sweep("l_bar", [(sx * 0.098, sy * 0.098, 0.02), (sx * 0.134, sy * 0.134, 0.33)], 0.01, sides=4, smooth_path=False)))
        o.append(dark(bx("l_rim", -0.145, 0.145, -0.145, 0.145, 0.322, 0.345)))
        o.append(dark(lathe("l_roof", [(0.22, 0.34), (0.215, 0.355), (0.14, 0.41), (0.06, 0.5), (0.035, 0.54), (0.052, 0.57), (0.02, 0.6), (0.0, 0.63)],
                            segments=4, rot=(0, 0, 45), smooth_angle=25)))
        for b in o:
            apply_transform(b)
            metal_drop(b)
        return instances_to_mesh("l_body", join(o, "l_body0"), [trs((0, 0, z), (0, 0, 0), scale)], uv_jitter=False, keep_base=False)

    def lantern_stand_base():
        foot = dark(lathe("l_foot", [(0.0, 0.0), (0.1, 0.0), (0.1, 0.025), (0.04, 0.05), (0.03, 0.2), (0.06, 0.235), (0.07, 0.25), (0.0, 0.25)], segments=8))
        metal_drop(foot)
        return join([lantern_body(0.25), foot], "lantern_stand")

    def lantern_tall_base():
        """Flood lantern on a 2 m iron standard (stands on a balustrade pier)."""
        parts = [dark(lathe("lt_post", [(0.0, 0.0), (0.14, 0.0), (0.14, 0.05), (0.075, 0.1), (0.055, 0.3), (0.08, 0.34), (0.042, 0.4), (0.03, 1.72), (0.062, 1.77),
                                        (0.03, 1.84), (0.03, 1.98), (0.1, 2.02), (0.1, 2.06), (0.0, 2.06)], segments=8)),
                 dark(bx("lt_arm", -0.3, 0.3, -0.012, 0.012, 1.6, 1.625)),
                 dark(sphere("lt_knob", 0.03, loc=(-0.3, 0, 1.612), segments=6, rings=4)), dark(sphere("lt_knob", 0.03, loc=(0.3, 0, 1.612), segments=6, rings=4))]
        for sx in (-1, 1):
            parts.append(dark(sweep("lt_scroll", [(sx * 0.03, 0, 1.25), (sx * 0.14, 0, 1.4), (sx * 0.2, 0, 1.58)], 0.009, sides=5)))
        for o in parts:
            metal_drop(o)
        return join(parts + [lantern_body(2.06, 1.55)], "lantern_tall")

    def lantern_wall_base():
        body = lantern_body()
        place(body, loc=(0, -0.5, -0.72))
        parts = [
            dark(bx("lw_plate", -0.06, 0.06, -0.02, 0.0, -0.2, 0.2, 0.004)),
            dark(sweep("lw_arm", [(0, -0.01, 0.12), (0, -0.25, 0.2), (0, -0.44, 0.14), (0, -0.5, 0.02)], 0.014, sides=6)),
            dark(sweep("lw_stay", [(0, -0.01, -0.15), (0, -0.16, -0.06), (0, -0.3, 0.14)], 0.01, sides=5)),
            dark(sweep("lw_scroll", spiral_path(0.015, 0.075, 1.4, n=20, a0_deg=90, plane="YZ", center=(0, -0.2, 0.05)), 0.008, sides=5)),
            dark(sweep("lw_hang", [(0, -0.5, 0.03), (0, -0.5, -0.1)], 0.008, sides=5, smooth_path=False)),
        ]
        for o in parts:
            metal_drop(o)
        return join(parts + [body], "lantern_wall")

    def metal_drop(o):
        if o in metal:
            metal.remove(o)

    def pinnacle_base():
        o = [lathe("pin", [(0.0, 0.0), (0.41, 0.0), (0.41, 0.12), (0.35, 0.2), (0.35, 1.02), (0.43, 1.1), (0.43, 1.22), (0.31, 1.3), (0.07, 2.5),
                           (0.12, 2.57), (0.12, 2.66), (0.05, 2.72), (0.0, 2.8)], segments=4, rot=(0, 0, 45), smooth_angle=25)]
        for k in range(4):
            a = radians(45 + k * 90)
            for t in (0.2, 0.5, 0.8):
                rr_ = lerp(0.31, 0.07, t)
                o.append(sphere("crocket", 0.06 * (1.15 - 0.4 * t), loc=(rr_ * cos(a) * 1.02, rr_ * sin(a) * 1.02, lerp(1.3, 2.5, t)), segments=6, rings=4))
            o.append(prism("pin_gab", [(-0.2, 0.86), (0.2, 0.86), (0.0, 1.2)], 0.24, 0.28, "y"))
            place(o[-1], rot=(0, 0, k * 90))
        b = join(o, "pinnacle")
        assign(b, TRIM)
        return b

    PIN = SimpleNamespace(base=pinnacle_base(), Ms=[])
    URN = SimpleNamespace(base=lathe("urn", [(0.0, 0.0), (0.13, 0.0), (0.13, 0.04), (0.06, 0.08), (0.05, 0.14), (0.14, 0.24), (0.175, 0.36), (0.15, 0.45),
                                             (0.1, 0.48), (0.115, 0.52), (0.04, 0.58), (0.035, 0.64), (0.0, 0.67)], segments=10), Ms=[])
    assign(URN.base, TRIM)
    BAL = SimpleNamespace(base=lathe("baluster", [(0.0, 0.0), (0.075, 0.0), (0.075, 0.045), (0.042, 0.07), (0.04, 0.1), (0.088, 0.22), (0.08, 0.3),
                                                  (0.04, 0.5), (0.062, 0.56), (0.075, 0.58), (0.075, 0.62), (0.0, 0.62)], segments=6, smooth_angle=50), Ms=[])
    assign(BAL.base, TRIM)
    pier_parts = [bx("pier_s", -0.26, 0.26, -0.26, 0.26, 0.0, 1.06, 0.02), bx("pier_c", -0.33, 0.33, -0.33, 0.33, 1.06, 1.17, 0.03)]
    PIER = SimpleNamespace(base=join(pier_parts, "bal_pier"), Ms=[])
    assign(PIER.base, TRIM)
    CORB = SimpleNamespace(base=prism("corbel", [(0.02, 0.0), (-0.06, 0.0), (-0.2, 0.16), (-0.24, 0.28), (0.02, 0.28)], -0.09, 0.09, "x"), Ms=[])
    assign(CORB.base, TRIM)
    LST = SimpleNamespace(base=lantern_stand_base(), Ms=[])
    LWL = SimpleNamespace(base=lantern_wall_base(), Ms=[])
    LTL = SimpleNamespace(base=lantern_tall_base(), Ms=[])
    lights = []          # (name, world position, intensity, range, colour)

    def corbels(F, x0, x1, z, step=0.62, y=0.0):
        n = max(1, int(round((x1 - x0) / step)))
        for i in range(n + 1):
            CORB.Ms.append(F.M(x0 + (x1 - x0) * i / n, y, z))

    def buttress(name, z0, z1, width=0.62, proj=0.5):
        """Stepped pilaster buttress (canonical, centred on x = 0) from z0 up to the cornice bottom z1."""
        h = z1 - z0
        zm = z0 + h * 0.52
        prof = [(0.05, z0), (-proj, z0), (-proj, zm), (-proj * 0.62, zm + 0.3), (-proj * 0.62, z1 - 0.75), (-0.14, z1 - 0.38), (-0.14, z1), (0.05, z1)]
        a = prism(name, prof, -width / 2, width / 2, "x")
        b = bx(name + "_base", -width / 2 - 0.07, width / 2 + 0.07, -proj - 0.07, 0.05, z0, z0 + 0.42, 0.02)
        c = prism(name + "_cap", [(0.05, zm - 0.05), (-proj - 0.05, zm - 0.05), (-proj - 0.05, zm + 0.03), (-proj * 0.62 - 0.02, zm + 0.36), (0.05, zm + 0.36)],
                  -width / 2 - 0.04, width / 2 + 0.04, "x")
        o = join([a, b, c], name)
        uv_box(o, offset=None)
        assign(o, ASH)
        return SimpleNamespace(base=o, Ms=[])

    def cornice_prof(z, proj=0.42, h=0.5):
        return [(0.05, z - h), (-0.12, z - h), (-0.14, z - h * 0.76), (-proj * 0.72, z - h * 0.44), (-proj * 0.76, z - h * 0.2), (-proj, z - h * 0.12),
                (-proj, z), (0.05, z)]

    def string_prof(z, proj=0.14, h=0.24):
        return [(0.03, z - h), (-proj * 0.55, z - h), (-proj, z - h * 0.6), (-proj, z - h * 0.25), (-proj * 0.3, z), (0.03, z)]

    def plinth_prof(z=0.95, proj=0.22):
        return [(0.03, 0.0), (-proj, 0.0), (-proj, z - 0.27), (-proj * 0.62, z - 0.14), (-proj * 0.45, z), (0.03, z)]

    def split_spans(x0, x1, g0, g1, n_total=None, target=2.7, min_bw=2.1):
        """Regular bays in [x0, g0] and [g1, x1] (a gap is left for the tower / porch)."""
        spans = []
        L1, L2 = max(0.0, g0 - x0), max(0.0, x1 - g1)
        for (a, b, L) in ((x0, g0, L1), (g1, x1, L2)):
            if n_total is None:
                n = int(L / target + 0.35)
            else:
                n = int(round(n_total * L / max(L1 + L2, 1e-3)))
            n = min(n, int(L / min_bw))
            if n > 0:
                spans.append((a, b, n))
        return spans

    # ================================================================== upper hall body
    wsh = p.wing_storey_height
    run = D / 2 + ov
    zr = ze + run * tan(radians(p.roof_pitch))
    zr = min(zr, max(TH - 5.9, ze + run * tan(radians(33.0))))
    tanp = (zr - ze) / run
    xa = -W / 2 + (pin + ap + 0.3 if pavL else 0.7)
    xb = W / 2 - (pin + ap + 0.3 if pavR else 0.7)
    spans_up = split_spans(xa, xb, tx - tw / 2 - 0.62, tx + tw / 2 + 0.62, n_total=int(p.bays))
    bw_up = min([(b - a) / n for a, b, n in spans_up] or [2.6])
    zs = hl + 1.3                                           # sill of the hall windows
    KU = make_kit("win_hall", clamp(bw_up * 0.5, 1.0, 1.5), wsh - 2.35)
    KLn = make_kit("win_lancet", 0.56, 1.9, gm=(2, 0, 1), tracery=False, band=0.13)
    KLu = make_kit("win_lucarne", 0.5, 1.15, gm=(2, 1, 0), tracery=False, band=0.1)
    BUP = buttress("butt_up", hl + 0.1, ze - 0.5)
    BBK = buttress("butt_back", 0.0, ze - 0.5, width=0.7, proj=0.62)

    Ff = Frame(0, 0, 0)
    holes, k = [], 0
    for a, b, n in spans_up:
        bw = (b - a) / n
        for i in range(n):
            xc = a + (i + 0.5) * bw
            holes.append(KU.hole(xc, zs))
            KU.Ms.append(Ff.M(xc, 0, zs))
        for i in range(n + 1):
            BUP.Ms.append(Ff.M(a + i * bw, 0, 0))
    Ff.put(wall("up_front", -W / 2, W / 2, hl - 0.2, ze, holes))
    Ff.put(mould("up_cornice", cornice_prof(ze), -W / 2, W / 2), mould("up_string", string_prof(zs - 0.16), -W / 2, W / 2))
    corbels(Ff, -W / 2 + 0.4, W / 2 - 0.4, ze - 0.74)

    # back wall: the same bays, tall buttresses, small lit lancets low down
    Fb = Frame(0, D, 180)
    nb = int(p.bays)
    bwb = (W - 1.4) / nb
    holes = []
    for i in range(nb):
        xc = -W / 2 + 0.7 + (i + 0.5) * bwb
        holes.append(KU.hole(xc, zs))
        KU.Ms.append(Fb.M(xc, 0, zs))
        if i % 2 == 0:
            holes.append(KLn.hole(xc, 1.7))
            KLn.Ms.append(Fb.M(xc, 0, 1.7))
    for i in range(nb + 1):
        BBK.Ms.append(Fb.M(-W / 2 + 0.7 + i * bwb, 0, 0))
    Fb.put(wall("up_back", -W / 2, W / 2, 0.0, ze, holes, dens=2.4))
    Fb.put(mould("bk_cornice", cornice_prof(ze), -W / 2, W / 2), mould("bk_string", string_prof(zs - 0.16), -W / 2, W / 2),
           mould("bk_string2", string_prof(hl * 0.86, 0.12, 0.2), -W / 2, W / 2), mould("bk_plinth", plinth_prof(), -W / 2 - 0.2, W / 2 + 0.2, "ash"))
    corbels(Fb, -W / 2 + 0.4, W / 2 - 0.4, ze - 0.74)

    # coped gable ends with a tall window and an oculus
    gable_out = [(0, 0), (D, 0), (D, ze - 0.55), (D + ov, ze - 0.3), (D + ov, ze + 0.3), (D / 2, zr + 0.5), (-ov, ze + 0.3), (-ov, ze - 0.3), (0, ze - 0.55)]
    r_oc = clamp((zr - ze) * 0.11, 0.45, 0.85)
    KOC = make_rose("win_oculus", r_oc)
    for side, F, pav in ((1, Frame(W / 2, 0, 90), pavR), (-1, Frame(-W / 2, D, 270), pavL)):
        xc = (pin + ap + 0.6 + D) / 2 if pav else D / 2
        if side < 0:
            xc = D - xc
        holes = [KU.hole(xc, zs)]
        KU.Ms.append(F.M(xc, 0, zs))
        if zr - ze > 4.0:
            zo = ze + (zr - ze) * 0.42
            holes.append(KOC.hole(D / 2, zo))
            KOC.Ms.append(F.M(D / 2, 0, zo))
        F.put(wall("gable", 0, D, 0, ze, holes, outline=gable_out))
        cop = [(-ov - 0.07, ze + 0.28), (-ov - 0.07, ze + 0.44), (D / 2, zr + 0.68), (D + ov + 0.07, ze + 0.44), (D + ov + 0.07, ze + 0.28), (D / 2, zr + 0.5)]
        F.put(trim(prism("gable_coping", cop, -0.07, TH_WALL + 0.07, "y")))
        F.put(mould("gable_string", string_prof(zs - 0.16), 0, D), mould("gable_string2", string_prof(ze - 0.3, 0.16, 0.26), -ov, D + ov),
              mould("gable_plinth", plinth_prof(), -0.2, D + 0.2, "ash"))
        PIN.Ms.append(F.M(D / 2, TH_WALL / 2, zr + 0.6, scale=0.6))
        # quoins on the free corners
        for xq, free in ((0.0, not pav if side > 0 else True), (D, True if side > 0 else not pav)):
            if not free:
                continue
            z, i = 1.0, 0
            while z < ze - 0.9:
                wq = 0.62 if i % 2 == 0 else 0.4
                sgn = 1 if xq == 0.0 else -1
                q = bx("quoin", xq - 0.03 * sgn, xq + sgn * wq, -0.035, 0.02, z + 0.01, z + 0.35, 0.012)
                F.put(trim(q))
                z += 0.36
                i += 1

    # main roof, ridge roll + iron cresting, gutters
    x0r, x1r = -W / 2 + 0.25, W / 2 - 0.25
    roof_quad("roof_f", (x0r, -ov, ze + 0.03), (x1r, -ov, ze + 0.03), (x1r, D / 2, zr), (x0r, D / 2, zr))
    roof_quad("roof_b", (x1r, D + ov, ze + 0.03), (x0r, D + ov, ze + 0.03), (x0r, D / 2, zr), (x1r, D / 2, zr))
    ridge = cylinder("ridge_roll", 0.13, x1r - x0r, loc=(0, D / 2, zr + 0.01), rot=(0, 90, 0), verts=8)
    uv_cyl(ridge, axis="Z")
    assign(ridge, ROOF)
    roofs.append(ridge)
    crest = prism("crest", [(-0.16, 0.0), (0.16, 0.0), (0.16, 0.05), (0.05, 0.08), (0.04, 0.2), (0.115, 0.24), (0.115, 0.31), (0.04, 0.31), (0.0, 0.43),
                            (-0.04, 0.31), (-0.115, 0.31), (-0.115, 0.24), (-0.04, 0.2), (-0.05, 0.08), (-0.16, 0.05)], -0.012, 0.012, "y")
    assign(crest, DARK)
    nc = int((x1r - x0r - 0.6) / 0.32)
    inst("cresting", crest, [trs((x0r + 0.3 + 0.32 * i + 0.16, D / 2, zr + 0.12)) for i in range(nc)], metal)
    for yy in (-ov - 0.07, D + ov + 0.07):
        dark(cylinder("gutter", 0.09, W - 0.2, loc=(0, yy, ze - 0.0), rot=(0, 90, 0), verts=8))

    def ys(z):
        return -ov + (z - ze) / tanp

    def lucarne(F, x, zl=None, ys=ys):
        zl = ze + 0.85 if zl is None else zl
        yf = ys(zl)
        lw, wh, gh = 1.25, 1.4, 0.8
        top = zl + wh + gh
        out = [(-lw / 2, zl - 0.35), (lw / 2, zl - 0.35), (lw / 2, zl + wh), (0, top), (-lw / 2, zl + wh)]
        fr = shape_extrude("luc_front", out, [KLu.hole(0, zl + 0.2)], depth=0.3, loc=(x, yf + 0.3, 0), rot=(90, 0, 0))
        F.put(trim(fr))
        body = prism("luc_body", [(x + u, v) for u, v in out], yf + 0.29, ys(top) + 0.1, "y")
        F.put(ash(body))
        KLu.Ms.append(F.M(x, yf, zl + 0.2))
        xe, zev, zt = lw / 2 + 0.16, zl + wh - 0.14, top + 0.06
        y0, y1 = yf - 0.16, ys(top) + 0.35
        roof_quad("luc_r", F.pt(x + xe, y0, zev), F.pt(x + xe, y1, zev), F.pt(x, y1, zt), F.pt(x, y0, zt), thick=0.06)
        roof_quad("luc_l", F.pt(x - xe, y1, zev), F.pt(x - xe, y0, zev), F.pt(x, y0, zt), F.pt(x, y1, zt), thick=0.06)
        F.put(trim(sphere("luc_ball", 0.09, loc=(x, yf + 0.1, top + 0.14), segments=8, rings=6)))

    if zr - ze > 4.2:
        nl = int(p.lucarnes)
        wings = [(xa + 0.8, tx - tw / 2 - 1.5), (tx + tw / 2 + 1.5, xb - 0.8)]
        Lw = [max(0.0, b - a) for a, b in wings]
        n_left = int(round(nl * Lw[0] / max(sum(Lw), 1e-3)))
        for (a, b), n in zip(wings, (n_left, nl - n_left)):
            n = min(n, int((b - a) / 2.4))
            for i in range(n):
                lucarne(Ff, a + (b - a) * (i + 0.5) / n)
        for i in range(max(0, nl // 2)):
            lucarne(Fb, (W - 8.0) * ((i + 0.5) / max(1, nl // 2) - 0.5))

    # ================================================================== lower arcade storey + terrace
    xl0 = -W / 2 + (pin + 1.2 if pavL else 0.0)
    xl1 = W / 2 - (pin + 1.2 if pavR else 0.0)
    pw = p.porch_width
    px = clamp(p.porch_offset, xl0 + pw / 2 + 1.2, xl1 - pw / 2 - 1.2)
    Fl = Frame(0, -T, 0)
    spans_lo = split_spans(xl0 + 0.4, xl1 - 0.4, px - pw / 2 - 0.5, px + pw / 2 + 0.5, target=bw_up)
    KA = make_kit("win_arcade", 1.7, hl - 1.95, gm=(0, 1, 2))
    za = 1.08
    holes, pier_x, lant_x = [], [xl0 + 0.4, xl1 - 0.4, px], []
    for a, b, n in spans_lo:
        bw = (b - a) / n
        for i in range(n):
            xc = a + (i + 0.5) * bw
            holes.append(KA.hole(xc, za))
            KA.Ms.append(Fl.M(xc, 0, za))
        for i in range(n + 1):
            xq = a + i * bw
            pier_x.append(xq)
            Fl.put(ash(bx("arc_pier", xq - 0.31, xq + 0.31, -0.24, 0.05, 0.9, hl - 0.4)),
                   trim(bx("arc_pier_cap", xq - 0.36, xq + 0.36, -0.29, 0.05, hl - 0.95, hl - 0.8, 0.02)),
                   trim(bx("arc_pier_base", xq - 0.36, xq + 0.36, -0.3, 0.05, 0.0, 1.0, 0.02)))
            if 0 < i < n and (i % 2 == 1 or n < 3):
                lant_x.append((xq, min(xq - a, b - xq)))
                LWL.Ms.append(Fl.M(xq, -0.24, 3.55, scale=1.45))
    pier_x = sorted(set(round(x, 3) for x in pier_x))
    Fl.put(wall("low_front", xl0, xl1, 0.0, hl, holes))
    Fl.put(mould("low_plinth", plinth_prof(), xl0 - 0.22, xl1 + 0.22, "ash"), mould("low_cornice", cornice_prof(hl + 0.1, 0.38, 0.55), xl0 - 0.38, xl1 + 0.38))
    ash(bx("terrace_floor", xl0 + 0.02, xl1 - 0.02, -T + 0.02, 0.3, hl - 0.3, hl + 0.1))
    Frr, Frl = Frame(xl1, -T, 90), Frame(xl0, 0, 270)
    for F, pav, flip in ((Frr, pavR, False), (Frl, pavL, True)):
        xc = (T - (1.2 if pav else 0.0)) / 2
        xc = T - xc if flip else xc
        holes = [KA.hole(xc, za)] if not pav or T > 4.6 else []
        if holes:
            KA.Ms.append(F.M(xc, 0, za))
        F.put(wall("low_return", 0, T, 0.0, hl, holes))
        e0, e1 = (0.0, 0.38) if flip else (-0.38, 0.0)
        F.put(mould("ret_plinth", plinth_prof(), e0 * 0.6, T + e1 * 0.6, "ash"), mould("ret_cornice", cornice_prof(hl + 0.1, 0.38, 0.55), e0, T + e1))

    def balustrade(F, x0, x1, y, zb, piers, toppers=1, tall=()):
        F.put(trim(bx("bal_plinth", x0, x1, y - 0.17, y + 0.17, zb, zb + 0.16, 0.015)), trim(bx("bal_rail", x0, x1, y - 0.19, y + 0.19, zb + 0.78, zb + 0.93, 0.02)))
        ps = sorted(piers)
        tops = []
        for i, x in enumerate(ps):
            PIER.Ms.append(F.M(x, y, zb))
            if any(abs(x - t) < 1e-3 for t in tall):
                LTL.Ms.append(F.M(x, y, zb + 1.17))
                tops.append(F.pt(x, y, zb + 1.17 + 2.06 + 0.32))
            elif toppers and i % 2 == toppers % 2:
                LST.Ms.append(F.M(x, y, zb + 1.17, scale=1.3))
            else:
                URN.Ms.append(F.M(x, y, zb + 1.17, rot=(0, 0, 18 * i), scale=1.1))
        for a, b in zip(ps[:-1], ps[1:]):
            g = (b - a) - 0.56
            n = int(g / 0.3)
            for j in range(n):
                BAL.Ms.append(F.M(a + 0.28 + g * (j + 0.5) / n, y, zb + 0.16, rot=(0, 0, 30 * j)))
        return tops

    ztb = hl + 0.1
    xw0, xw1 = xl0 + 0.4, xl1 - 0.4
    targets = [tx, xw0 + (tx - tw / 2 - xw0) * 0.45, xw1 - (xw1 - tx - tw / 2) * 0.45]
    tall = sorted(set(min(pier_x, key=lambda q: abs(q - x)) for x in targets))
    tops = balustrade(Fl, xw0, xw1, 0.3, ztb, pier_x, toppers=2, tall=tall)
    balustrade(Frr, 0.3, T - 0.1, 0.3, ztb, [0.3 + (T - 0.6) * 0.5, T - 0.35] if T > 4.4 else [T - 0.35], toppers=0)
    balustrade(Frl, 0.1, T - 0.3, 0.3, ztb, [0.35, 0.3 + (T - 0.6) * 0.5] if T > 4.4 else [0.35], toppers=0)
    # tall terrace standards flood the hall wall and the tower shaft
    for t in tops:
        near_tower = abs(t[0] - tx) < tw / 2 + 0.5
        d = (T - 0.3) - (1.85 if near_tower else 0.0)       # same illuminance on the wall / tower face whatever the terrace depth
        lights.append(("flood", t, 10.0 * p.lantern_intensity * min(0.118 * d * d, 3.6), 30.0, "#ffd9a6"))
    if lant_x:
        best = max(lant_x, key=lambda t: t[1] + abs(t[0] - px) * 0.15)
        lights.append(("arcade", Fl.pt(best[0], -1.6, 3.0), p.lantern_intensity * 0.3, 16.0, "#ffc27a"))

    # ================================================================== clock tower
    zc = TH - 5.6                                           # gallery level = base of the clock stage
    rd = min(tw * 0.27, 1.38)
    zd = zc + 1.25 + rd                                     # dial centre
    z0t = hl - 0.1
    lz0 = hl + 1.9
    lz1 = min(ze + 0.9, zc - 1.6)
    KB = make_kit("win_belfry", min(1.65, tw * 0.33), lz1 - lz0, fill="louvre", band=0.2)
    KS = make_kit("win_slit", 0.34, 1.0, tracery=False, fill="louvre", band=0.09)
    zs1 = lz1 + 0.95
    hw = tw / 2 - 0.35
    zgb, zga = zd - 0.2 * rd, zd + rd + 1.6
    tower_frames = [Frame(tx + (tw / 2) * sin(radians(a)), tyc - (tw / 2) * cos(radians(a)), a) for a in (0, 90, 180, 270)]
    hh, mm = 10, 10
    try:
        hh, mm = [int(v) for v in str(p.time).split(":")[:2]]
    except Exception:
        pass
    a_min = (mm % 60) * 6.0
    a_hr = ((hh % 12) + (mm % 60) / 60.0) * 30.0
    for fi, F in enumerate(tower_frames):
        xs_ = tw / 2 - 0.86
        holes = [KB.hole(0, lz0), KS.hole(-xs_, TH - 1.8), KS.hole(xs_, TH - 1.8)]
        KB.Ms.append(F.M(0, 0, lz0))
        KS.Ms += [F.M(-xs_, 0, TH - 1.8), F.M(xs_, 0, TH - 1.8)]
        F.put(wall("tower_face", -tw / 2, tw / 2, z0t, TH, holes))
        if fi != 2:
            F.put(mould("tw_plinth", plinth_prof(0.85, 0.16), -tw / 2 + 0.5, tw / 2 - 0.5))
        F.put(mould("tw_string", string_prof(zs1, 0.16, 0.3), -tw / 2 + 0.3, tw / 2 - 0.3))
        # blind arcade between the belfry arch and the gallery
        band0, band1 = zs1 + 0.45, zc - 0.95
        if band1 - band0 > 1.5:
            n_l, wpl = 4, tw - 1.7
            cwl = wpl / n_l
            bh = min(band1 - band0, 3.4)
            bl_holes = [[(x - wpl / 2 + (i + 0.5) * cwl, z + band0 + 0.2) for x, z in arch_pts(cwl - 0.24, bh - 0.4, None, "pointed", 8)] for i in range(n_l)]
            F.put(trim(shape_extrude("tw_blind", [(-wpl / 2, band0), (wpl / 2, band0), (wpl / 2, band0 + bh), (-wpl / 2, band0 + bh)], bl_holes, depth=0.09,
                                     loc=(0, 0, 0), rot=(90, 0, 0))))
        # corbelled gallery with a pierced parapet
        gx = tw / 2 + 0.5
        F.put(mould("tw_gallery", cornice_prof(zc + 0.06, 0.5, 0.62), -gx, gx))
        corbels(F, -tw / 2 + 0.2, tw / 2 - 0.2, zc - 0.86, step=0.55)
        ncell = max(4, int((2 * gx - 0.3) / 0.5))
        cwc = (2 * gx - 0.3) / ncell
        ph = []
        for i in range(ncell):
            xl_, xr_ = -gx + 0.15 + i * cwc + 0.05, -gx + 0.15 + (i + 1) * cwc - 0.05
            xm_ = (xl_ + xr_) / 2
            ph.append([(xl_, zc + 0.2), (xr_, zc + 0.2), (xm_, zc + 0.72)] if i % 2 == 0 else [(xl_, zc + 0.78), (xm_, zc + 0.26), (xr_, zc + 0.78)])
        F.put(trim(shape_extrude("tw_parapet", [(-gx, zc + 0.06), (gx, zc + 0.06), (gx, zc + 0.9), (-gx, zc + 0.9)], ph, depth=0.12, loc=(0, -0.36, 0), rot=(90, 0, 0))),
              trim(bx("tw_parapet_rail", -gx - 0.03, gx + 0.03, -0.53, -0.31, zc + 0.9, zc + 1.02, 0.02)))
        # gablet with the dial
        F.put(trim(shape_extrude("gablet", [(-hw, zgb), (hw, zgb), (0, zga)], [circle_pts(rd, 28, center=(0, zd))], depth=0.3, loc=(0, 0, 0), rot=(90, 0, 0))))
        a_ = hw + 0.28
        zb_, zt_ = zgb - 0.22, zga + 0.22
        gr = prism("gablet_roof", [(-a_, zb_), (0, zt_), (a_, zb_), (a_ - 0.2, zb_), (0, zt_ - 0.2), (-a_ + 0.2, zb_)], -0.52, 0.02, "y")
        uv_box(gr)
        assign(gr, ROOF)
        roofs.append(gr)
        F.put(gr, trim(sphere("gablet_ball", 0.13, loc=(0, -0.25, zt_ + 0.1), segments=10, rings=6)))
        disc = cylinder("dial", rd + 0.02, 0.03, loc=(0, -0.1, zd), rot=(90, 0, 0), verts=32)
        assign(disc, LIT)
        bez = lathe("dial_bezel", [(rd - 0.1, 0.0), (rd - 0.085, 0.05), (rd + 0.01, 0.075), (rd + 0.1, 0.05), (rd + 0.13, 0.0)], segments=32, loc=(0, -0.3, zd), rot=(90, 0, 0))
        assign(bez, GILT)
        marks = [dark(torus("dial_track", major=rd * 0.93, minor=0.014, loc=(0, -0.125, zd), rot=(90, 0, 0), major_seg=32, minor_seg=4))]
        for k_ in range(12):
            big = k_ % 3 == 0
            b_ = bx("baton", -0.05 if big else -0.03, 0.05 if big else 0.03, -0.012, 0.0, rd * (0.62 if big else 0.7), rd * 0.88)
            place(b_, loc=(0, -0.12, zd), rot=(0, k_ * 30.0, 0))
            marks.append(dark(b_))
        hr = prism("hand_h", [(-0.05, -0.2), (0.05, -0.2), (0.075, rd * 0.36), (0.0, rd * 0.56), (-0.075, rd * 0.36)], -0.015, 0.0, "y")
        place(hr, loc=(0, -0.15, zd), rot=(0, a_hr, 0))
        mn = prism("hand_m", [(-0.035, -0.26), (0.035, -0.26), (0.05, rd * 0.6), (0.0, rd * 0.86), (-0.05, rd * 0.6)], -0.015, 0.0, "y")
        place(mn, loc=(0, -0.175, zd), rot=(0, a_min, 0))
        boss = cylinder("hand_boss", 0.08, 0.04, loc=(0, -0.19, zd), rot=(90, 0, 0), verts=12)
        assign(boss, GILT)
        F.put(disc, bez, marks, dark(hr), dark(mn), boss)
        F.put(mould("tw_cornice", cornice_prof(TH, 0.36, 0.48), -tw / 2 - 0.36, tw / 2 + 0.36))
        corbels(F, -tw / 2 + 0.25, tw / 2 - 0.25, TH - 0.72, step=0.5)
    # clasping corner buttresses, stepping in at each stage, and the corner pinnacles
    for sx in (-1, 1):
        for sy in (-1, 1):
            cxx, cyy = tx + sx * tw / 2, tyc + sy * tw / 2
            for (z0_, z1_, half, out_) in ((z0t, zs1 - 0.1, 0.42, 0.24), (zs1 - 0.1, zc - 0.5, 0.36, 0.17), (zc, TH - 0.4, 0.3, 0.1)):
                ash(bx("tw_butt", cxx + sx * out_ - sx * 2 * half, cxx + sx * out_, cyy + sy * out_ - sy * 2 * half, cyy + sy * out_, z0_, z1_, 0.02))
            trim(prism("tw_setoff", [(-0.44, 0.0), (0.44, 0.0), (0.36, 0.3), (-0.36, 0.3)], -0.44, 0.44, "y")).location = (cxx + sx * (0.24 - 0.42), cyy + sy * (0.24 - 0.42), zs1 - 0.12)
            PIN.Ms.append(trs((cxx - sx * 0.12, cyy - sy * 0.12, TH), (0, 0, 0), 1.0))
            trim(bx("tw_gpost", cxx + sx * 0.3, cxx + sx * 0.56, cyy + sy * 0.3, cyy + sy * 0.56, zc + 0.06, zc + 1.2, 0.02))
            trim(sphere("tw_gball", 0.12, loc=(cxx + sx * 0.43, cyy + sy * 0.43, zc + 1.3), segments=8, rings=6))
    rs = (tw / 2 + 0.14) * sqrt(2.0)
    pyramid_roof("spire", tx, tyc, 4, 45.0, [(rs, TH + 0.02), (rs * 0.7, TH + SH * 0.13), (0.09, TH + SH)])
    for k_ in range(4):
        a = radians(45 + k_ * 90)
        dark(sweep("spire_hip", [(tx + rs * cos(a), tyc + rs * sin(a), TH + 0.05), (tx + rs * 0.7 * cos(a), tyc + rs * 0.7 * sin(a), TH + SH * 0.13 + 0.03),
                                 (tx + 0.09 * cos(a), tyc + 0.09 * sin(a), TH + SH + 0.02)], 0.07, sides=6, smooth_path=False))
    z1s, a1s = TH + SH * 0.13, rs * 0.7 * cos(radians(45.0))
    zsl = TH + SH * 0.27
    if a1s * (1 - (zsl + 2.3 - z1s) / (TH + SH - z1s)) > 0.95:
        for F in tower_frames:
            lucarne(F, 0.0, zsl, lambda z: tw / 2 - a1s * (1 - (z - z1s) / (TH + SH - z1s)))
    fin = lathe("spire_finial", [(0.0, -0.5), (0.16, -0.5), (0.13, -0.2), (0.09, 0.0), (0.07, 0.25), (0.16, 0.3), (0.07, 0.36)] +
                [(0.34 * sin(radians(t)), 0.72 - 0.34 * cos(radians(t))) for t in range(15, 170, 22)] +
                [(0.05, 1.08), (0.035, 1.7), (0.09, 1.8), (0.1, 1.88), (0.0, 2.0)], segments=16, loc=(tx, tyc, TH + SH))
    assign(fin, GILT)
    lights.append(("clock", (tx, tyc - tw / 2 - 2.2, zd - 0.6), p.lantern_intensity * 0.5, 17.0, "#ffc98a"))

    # ================================================================== end pavilions
    def in_body(x, y):
        return -W / 2 < x < W / 2 and 0.0 < y < D

    def in_low(x, y):
        return xl0 < x < xl1 and -T < y <= 0.0

    zp = ze + 0.9
    fw = 2 * Rp * sin(radians(22.5))
    KP = make_kit("win_pav", 0.95, 2.7, gm=(2, 0, 1))
    up0, up1 = hl + 0.9, zp - 0.95
    lev_up = [up0, up1 - 2.7] if up1 - up0 >= 6.2 else [(up0 + up1 - 2.7) / 2]
    hc = 6.8
    for sgn, on in ((-1, pavL), (1, pavR)):
        if not on:
            continue
        cx, cy = sgn * pcx, pcy
        e = 0.42 * tan(radians(22.5))
        for k_ in range(8):
            a = k_ * 45.0
            F = Frame(cx + ap * sin(radians(a)), cy - ap * cos(radians(a)), a)
            ox_, oy_, _ = F.pt(0, -0.4, 0)
            if in_body(ox_, oy_):
                continue
            levels = ([] if in_low(ox_, oy_) else [1.4]) + lev_up
            holes = []
            for z in levels:
                holes.append(KP.hole(0, z))
                KP.Ms.append(F.M(0, 0, z))
            F.put(wall("pav_face", -fw / 2, fw / 2, 0.0, zp, holes, dens=1.2))
            F.put(mould("pav_plinth", plinth_prof(), -fw / 2 - 0.09, fw / 2 + 0.09, "ash"), mould("pav_string", string_prof(hl + 0.1, 0.16, 0.28), -fw / 2 - 0.06, fw / 2 + 0.06),
                  mould("pav_cornice", cornice_prof(zp, 0.42, 0.5), -fw / 2 - e, fw / 2 + e))
            if len(lev_up) == 2:
                F.put(mould("pav_string2", string_prof(lev_up[1] - 0.3, 0.14, 0.24), -fw / 2 - 0.06, fw / 2 + 0.06))
            corbels(F, -fw / 2 + 0.25, fw / 2 - 0.25, zp - 0.74, step=0.5)
        for k_ in range(8):
            a = k_ * 45.0 + 22.5
            vx, vy = cx + Rp * sin(radians(a)), cy - Rp * cos(radians(a))
            if in_body(vx + 0.5 * sin(radians(a)), vy - 0.5 * cos(radians(a))):
                continue
            Frame(vx, vy, a).put(ash(bx("pav_strip", -0.21, 0.21, -0.1, 0.3, 0.0, zp - 0.45, 0.02)))
        rr_ = Rp + 0.5
        pyramid_roof("pav_roof", cx, cy, 8, 22.5 - 90.0, [(rr_, zp + 0.02), (rr_ * 0.7, zp + hc * 0.15), (0.08, zp + hc)])
        f2 = lathe("pav_finial", [(0.0, -0.3), (0.14, -0.3), (0.1, 0.0), (0.06, 0.2), (0.13, 0.26), (0.06, 0.32), (0.16, 0.42), (0.2, 0.55), (0.16, 0.68), (0.05, 0.76), (0.0, 0.8)],
                   segments=12, loc=(cx, cy, zp + hc))
        dark(f2)
        # weathervane: iron spindle, cardinal arms, gilt rooster that spins
        zv = zp + hc + 0.75
        dark(cylinder("vane_spindle", 0.022, 2.1, loc=(cx, cy, zv + 1.05), verts=8))
        for ang in (0, 90):
            dark(cylinder("vane_arm", 0.012, 0.9, loc=(cx, cy, zv + 0.75), rot=(0, 90, ang), verts=6))
        g1 = sphere("vane_ball", 0.1, loc=(cx, cy, zv + 0.45), segments=12, rings=8)
        assign(g1, GILT)
        s_ = 1.25
        rooster = [(0.5, 0.52), (0.42, 0.66), (0.36, 0.74), (0.3, 0.66), (0.26, 0.5), (0.05, 0.36), (-0.25, 0.62), (-0.42, 0.72), (-0.52, 0.6), (-0.38, 0.5),
                   (-0.48, 0.36), (-0.3, 0.3), (-0.18, 0.14), (0.0, 0.05), (0.0, -0.08), (0.08, -0.08), (0.1, 0.04), (0.3, 0.12), (0.38, 0.3), (0.36, 0.46), (0.44, 0.48)]
        rb = prism("vane_rooster", [(x * s_, z * s_) for x, z in rooster], -0.008, 0.008, "y")
        ar = bx("vane_arrow", -0.7, 0.7, -0.012, 0.012, -0.13, -0.09)
        tip = prism("vane_tip", [(0.7, -0.22), (0.95, -0.11), (0.7, 0.0)], -0.008, 0.008, "y")
        tail = prism("vane_tail", [(-0.95, -0.24), (-0.66, -0.11), (-0.95, 0.02), (-0.86, -0.11)], -0.008, 0.008, "y")
        vane = join([rb, ar, tip, tail], f"vane_{'l' if sgn < 0 else 'r'}")
        assign(vane, GILT)
        place(vane, loc=(cx, cy, zv + 1.2), rot=(0, 0, rng.fork(f"vane{sgn}").u(-40, 40)))
        part([vane], f"vane_{'l' if sgn < 0 else 'r'}", pivot=(cx, cy, zv + 1.2), motion="spin", axis="Z")

    # ================================================================== entrance porch
    pd, he, zf = 2.6, 3.9, 0.85
    tang = tan(radians(42.0))
    hap = he + (pw / 2) * tang
    Fp = Frame(px, -T - pd, 0)
    dw, dh, dsp = 2.4, 3.7, 2.5
    door_arch = [(x, z + zf) for x, z in arch_pts(dw, dh, dsp, "pointed", 14)]
    holes = [door_arch]
    r_rose = min(0.74, ((hap - (zf + dh + 0.78)) * 0.743 - 0.27) / 1.743)
    if r_rose > 0.28:
        KR = make_rose("win_rose", r_rose)
        zrose = zf + dh + 0.78 + r_rose
        holes.append(KR.hole(0, zrose))
        KR.Ms.append(Fp.M(0, 0, zrose))
    if pw >= 6.6:
        for sx in (-1, 1):
            xl_ = sx * (pw / 2 - 1.45)
            holes.append(KLn.hole(xl_, zf + 0.75))
            KLn.Ms.append(Fp.M(xl_, 0, zf + 0.75))
    Fp.put(wall("porch_front", 0, 0, 0, 0, holes, outline=[(-pw / 2, 0), (pw / 2, 0), (pw / 2, he), (0, hap), (-pw / 2, he)], dens=1.2))
    for i, (band, dep) in enumerate(((0.2, 0.15), (0.4, 0.08))):
        outer = [(x, z + zf) for x, z in arch_pts(dw + 2 * band, dh + 1.6 * band, dsp, "pointed", 14)]
        inner = [(x, z + zf) for x, z in arch_pts(dw - 0.02, dh - 0.01, dsp, "pointed", 14)]
        Fp.put(trim(shape_extrude(f"door_order{i}", u_ring(outer, inner), depth=dep, loc=(0, 0, 0), rot=(90, 0, 0), bevel=0.02)))
    # leaves: boarded oak, iron straps with studs, ring handles; stone transom, glazed tympanum
    rd_ = rng.fork("door")
    leaves = []
    for sx in (-1, 1):
        for i in range(4):
            x0_ = sx * (0.012 + i * 0.297)
            b_ = bx("door_board", x0_, x0_ + sx * 0.29, 0.34 + rd_.u(0, 0.006), 0.4, zf + 0.01, zf + dsp + 0.02, 0.006)
            uv_grain(b_, axis="Z")
            assign(b_, WOOD)
            leaves.append(b_)
        for zz in (0.35, 1.25, 2.15):
            leaves.append(dark(bx("door_strap", sx * 0.1, sx * 1.17, 0.326, 0.342, zf + zz - 0.04, zf + zz + 0.04, 0.004)))
        h_ = torus("door_ring", major=0.085, minor=0.013, loc=(sx * 0.14, 0.315, zf + 1.12), rot=(90, 0, 0), major_seg=16, minor_seg=6)
        assign(h_, GILT)
        hp = cylinder("door_rose", 0.045, 0.02, loc=(sx * 0.14, 0.332, zf + 1.2), rot=(90, 0, 0), verts=10)
        assign(hp, GILT)
        leaves += [h_, hp]
    stud = bolt("door_stud", 0.02, 0.014, kind="pyramid")
    assign(stud, DARK)
    sm = [trs((sx * (0.2 + 0.22 * i), 0.326, zf + zz), (90, 0, 0)) for sx in (-1, 1) for zz in (0.35, 1.25, 2.15) for i in range(5)]
    leaves.append(inst("door_studs", stud, sm))
    leaves.append(trim(bx("door_transom", -dw / 2 - 0.02, dw / 2 + 0.02, 0.18, 0.46, zf + dsp, zf + dsp + 0.2, 0.02)))
    leaves.append(trim(bx("door_mullion", -0.06, 0.06, 0.2, 0.32, zf + dsp + 0.2, zf + dh)))
    tymp = glass_panel("door_tymp", dw, dh, dsp, 0.3, (0, 1, 2), 1, False, z0=dsp + 0.2)
    place(tymp, loc=(0, 0, zf))
    leaves.append(tymp)
    Fp.put(leaves)
    # corner buttresses with pinnacles, raised gable coping, steps with cheek walls and lantern piers
    for sx in (-1, 1):
        xo, xi = sx * (pw / 2 + 0.3), sx * (pw / 2 - 0.5)
        Fp.put(ash(bx("porch_butt", xi, xo, -0.3, 0.5, 0.0, he + 0.3, 0.02)), trim(bx("porch_butt_base", xi - sx * 0.07, xo + sx * 0.07, -0.37, 0.5, 0.0, 1.0, 0.025)),
               trim(bx("porch_butt_cap", xi - sx * 0.05, xo + sx * 0.05, -0.35, 0.55, he + 0.3, he + 0.45, 0.03)))
        PIN.Ms.append(Fp.M(sx * (pw / 2 - 0.1), 0.1, he + 0.45, scale=0.86))
    sw = 3.4
    for sx in (-1, 1):
        Fp.put(mould("porch_plinth", plinth_prof(), sx * (sw / 2 + 0.44), sx * (pw / 2 - 0.4), "ash"))
    cop = [(-pw / 2 - 0.12, he - 0.12), (-pw / 2 - 0.12, he + 0.1), (0, hap + 0.26), (pw / 2 + 0.12, he + 0.1), (pw / 2 + 0.12, he - 0.12), (0, hap + 0.03)]
    Fp.put(trim(prism("porch_coping", cop, -0.1, 0.6, "y")))
    PIN.Ms.append(Fp.M(0, 0.25, hap + 0.14, scale=0.55))
    Fp.put(trim(bx("porch_landing", -sw / 2, sw / 2, -0.8, 0.45, 0.0, zf, 0.02)))
    rs_ = rng.fork("steps")
    for i in range(1, 5):
        y1_ = -0.8 - 0.32 * (i - 1) + 0.03
        xcut = rs_.u(-0.6, 0.6)
        for (xa_, xb_) in ((-sw / 2, xcut - 0.004), (xcut + 0.004, sw / 2)):
            Fp.put(trim(bx("porch_step", xa_, xb_, -0.8 - 0.32 * i - rs_.u(0, 0.008), y1_, 0.0, zf - 0.17 * i, 0.018)))
    for sx in (-1, 1):
        x0_, x1_ = sx * sw / 2, sx * (sw / 2 + 0.45)
        Fp.put(ash(bx("porch_cheek", x0_, x1_, -1.85, 0.0, 0.0, zf + 0.28)), trim(bx("porch_cheek_cap", x0_ - sx * 0.04, x1_ + sx * 0.04, -1.85, -0.3, zf + 0.28, zf + 0.4, 0.025)),
               ash(bx("porch_lpier", x0_ - sx * 0.04, x1_ + sx * 0.04, -2.3, -1.77, 0.0, 1.45, 0.02)),
               trim(bx("porch_lpier_cap", x0_ - sx * 0.09, x1_ + sx * 0.09, -2.35, -1.72, 1.45, 1.57, 0.03)))
        xlp = sx * (sw / 2 + 0.225)
        LST.Ms.append(Fp.M(xlp, -2.035, 1.57, scale=1.5))
    lights.append(("porch", Fp.pt(0, -2.1, 2.7), p.lantern_intensity * 0.6, 15.0, "#ffc27a"))
    # side walls, eaves cornice, back gable, tiled roof
    for F, flip in ((Frame(px + pw / 2, -T - pd, 90), False), (Frame(px - pw / 2, -T, 270), True)):
        xc = pd / 2 + 0.25
        xc = pd - xc if flip else xc
        KLn.Ms.append(F.M(xc, 0, 1.75))
        F.put(wall("porch_side", 0, pd, 0.0, he, [KLn.hole(xc, 1.75)], dens=1.2), mould("porch_eaves", cornice_prof(he, 0.26, 0.36), 0.0, pd),
              mould("porch_splinth", plinth_prof(), 0.3 if not flip else 0.0, pd if not flip else pd - 0.3, "ash"))
    gp = [(-pw / 2 + 0.02, he - 0.4), (pw / 2 - 0.02, he - 0.4), (pw / 2 - 0.02, he), (0, hap), (-pw / 2 + 0.02, he)]
    Fp.put(ash(prism("porch_back", gp, pd - 0.45, pd - 0.03, "y")), trim(prism("porch_back_coping", cop, pd - 0.55, pd + 0.0, "y")))
    xe = pw / 2 + 0.3
    zev, zt = he - 0.3 * tang + 0.03, hap + 0.03
    roof_quad("porch_roof_r", Fp.pt(xe, 0.3, zev), Fp.pt(xe, pd - 0.3, zev), Fp.pt(0, pd - 0.3, zt), Fp.pt(0, 0.3, zt), thick=0.07)
    roof_quad("porch_roof_l", Fp.pt(-xe, pd - 0.3, zev), Fp.pt(-xe, 0.3, zev), Fp.pt(0, 0.3, zt), Fp.pt(0, pd - 0.3, zt), thick=0.07)
    pr = cylinder("porch_ridge", 0.1, pd - 0.9, loc=Fp.pt(0, pd / 2, zt + 0.02), rot=(90, 0, 0), verts=8)
    assign(pr, ROOF)
    roofs.append(pr)

    # ================================================================== flush instanced kits
    for k in kits:
        for b in k.parts:
            inst(b.name, b, k.Ms, stone if b.name.endswith(("_ring", "_sill", "_trac")) else None)
        for v, g in enumerate(k.glass):
            inst(g.name, g, k.Ms[v::len(k.glass)])
    inst("pinnacles", PIN.base, PIN.Ms, stone)
    inst("urns", URN.base, URN.Ms, stone)
    inst("balusters", BAL.base, BAL.Ms, stone)
    inst("bal_piers", PIER.base, PIER.Ms, stone)
    inst("corbels", CORB.base, CORB.Ms, stone)
    inst("butts_up", BUP.base, BUP.Ms, stone, jitter=False)
    inst("butts_back", BBK.base, BBK.Ms, stone, jitter=False)
    inst("lanterns_stand", LST.base, LST.Ms)
    inst("lanterns_wall", LWL.base, LWL.Ms)
    inst("lanterns_tall", LTL.base, LTL.Ms)

    # ================================================================== lights + sockets
    for i, (nm, pos, cd, rg, col) in enumerate(lights[:6]):
        light(f"{nm}_{i}", pos, color=col, intensity=max(cd, 0.01), range=rg, flicker=0.06, shadow_near=0.85 if nm == "flood" else 0.6)
    socket("door", "attach", Fp.pt(0, 0.0, zf), role="door")
    for sx in (-1, 1):
        socket(f"banner_porch_{'l' if sx < 0 else 'r'}", "attach", Fp.pt(sx * (pw / 2 - 0.1), -0.32, 3.2), role="banner")
    socket("banner_tower", "attach", (tx, tyc - tw / 2 - 0.05, zs1 + 0.3), role="banner")
    for i, (a, b, n) in enumerate(spans_lo):
        socket(f"lantern_arcade_{i}", "attach", Fl.pt(a, -0.26, 3.2), role="lantern")
        socket(f"planter_arcade_{i}", "attach", Fl.pt((a + b) / 2, -0.5, 0.0), role="planter")

    # ================================================================== weathering
    m = p.moss
    sd = p.seed
    splash = mask_mul(height_ramp(0.0, 1.5), 0.5)
    damp = mask_mul(height_ramp(0.0, 3.0), noise_mask(0.9, seed=sd + 5, lo=0.35, hi=0.7), min(1.0, m * 2.2))
    patches = mask_mul(noise_mask(2.4, seed=sd + 3, lo=0.5, hi=0.78), m * 0.9)
    ledges = mask_mul(facing((0, 0, 1), 2.0), noise_mask(0.6, seed=sd + 8, lo=0.3, hi=0.7), min(1.0, m * 2.5))
    north = mask_mul(facing((0, 1, 0), 1.0), noise_mask(1.7, seed=sd + 11, lo=0.4, hi=0.7), min(1.0, m * 1.6))
    streaks = mask_mul(noise_mask(1.3, seed=sd + 17, lo=0.5, hi=0.85), 0.22)
    for o in stone:
        paint(o, grime=splash, special=damp)
        paint(o, grime=streaks, special=patches)
        paint(o, special=ledges)
        paint(o, special=north)
    for o in roofs:
        paint(o, special=mask_mul(noise_mask(2.0, seed=sd + 21, lo=0.4, hi=0.75), min(1.0, m * 1.8)), grime=mask_mul(noise_mask(3.0, seed=sd + 23), 0.45))
    for o in metal:
        paint(o, special=mask_mul(noise_mask(0.5, seed=sd + 31, lo=0.45, hi=0.8), 0.5))
