"""Cafe under a timber gallery: a masonry / plastered ground floor with a lit cafe front (two shopfront
windows and a glazed door) set back under the upper floor, which is carried over the pavement on chamfered
oak posts standing on dressed stone plinths (bressummer, girders, joists, arch braces). The upper storey is
half-timbered (pegged-frame look: sill, plate, posts, rails, studs, braces over lime plaster) with a
balustraded loggia balcony behind the posts, and the whole is covered by one steep tiled roof (individually
laid tiles, ridge tiles, bargeboards, rafter feet, dormers, chimney, lead gutters and downpipes).
The terrace under the gallery carries tables, chairs, barrels, a menu board, candle lanterns, hanging
lanterns, a coffee-cup sign with its bracket lantern on the corner post, flower troughs and a neon
fleur-de-lis on the balustrade. Catalogue parts via use(): shop_window, door_shop, hanging_sign, barrel.
"""
from tk import *
from math import sin, cos, tan, asin, acos, atan2, radians, degrees, pi, hypot, sqrt, ceil
from mathutils import Matrix, Vector

META = {"name": "Cafe under a timber gallery", "class": "building", "mount": "ground", "focus": [-1.2, -2.8, 2.45]}
PARAMS = {
    "width": {"default": 11.0, "min": 8.0, "max": 14.0, "desc": "width of the building along X (m)"},
    "depth": {"default": 7.0, "min": 6.0, "max": 10.0, "desc": "depth of the body behind the front wall plane y = 0 (m)"},
    "storeys": {"default": 2, "min": 2, "max": 3, "desc": "storeys incl. the ground floor (3 = a jettied floor closes over the balcony)"},
    "gallery_depth": {"default": 3.0, "min": 2.5, "max": 4.0, "desc": "projection of the upper floor and its posts in front of y = 0 (m)"},
    "posts": {"default": 4, "min": 3, "max": 6, "desc": "number of gallery posts along the front (reduced if they would stand closer than 2.2 m)"},
    "sign_text": {"default": "CAFE", "desc": "lettering of the hanging cup sign and the gilt letters on the bressummer"},
    "fascia_text_l": {"default": "COFFEE", "desc": "fascia lettering of the left shopfront"},
    "fascia_text_r": {"default": "PATISSERIE", "desc": "fascia lettering of the right shopfront"},
    "menu_text": {"default": "MENU", "desc": "chalk heading on the A-frame menu board ('' = no board)"},
    "furniture": {"default": 3, "min": 0, "max": 6, "desc": "table sets (table, two chairs, candle lantern) on the terrace"},
    "barrels": {"default": 2, "min": 0, "max": 2, "desc": "oak barrels by the corner posts (dropped if the triangle budget would be exceeded)"},
    "balcony_depth": {"default": 1.35, "min": 0.9, "max": 1.8, "desc": "depth of the loggia balcony behind the posts (m)"},
    "balustrade": {"default": "splat", "choices": ["splat", "cross"], "desc": "flat-sawn splat balusters or St Andrew's cross panels"},
    "pattern": {"default": "braced", "choices": ["braced", "close", "square", "cross"], "desc": "half-timber panel pattern of the upper storeys"},
    "ground_wall": {"default": "stone", "choices": ["stone", "brick", "plaster"], "desc": "ground-floor masonry: rubble stone, red brick or limewashed"},
    "roof": {"default": "clay", "choices": ["clay", "slate"], "desc": "roof covering: hand-made clay tiles or slates"},
    "pitch": {"default": 45.0, "min": 38.0, "max": 52.0, "desc": "roof pitch (deg)"},
    "sag": {"default": 0.07, "min": 0.0, "max": 0.2, "desc": "sag of the ridge at mid span (m)"},
    "dormers": {"default": 2, "min": 0, "max": 3, "desc": "lit dormers on the front slope"},
    "planters": {"default": 2, "min": 0, "max": 3, "desc": "flower troughs hung on the balustrade"},
    "flowers": {"default": "#c7323a", "desc": "colour of the trough flowers"},
    "neon": {"default": True, "desc": "neon fleur-de-lis on the balustrade"},
    "neon_color": {"default": "#ffab24", "desc": "neon gas colour"},
    "plaster": {"default": "#e3d5b3", "desc": "limewash colour of the infill panels"},
    "timber": {"default": "#4c3727", "desc": "stain of the oak frame"},
    "paint": {"default": "#1f4a3c", "desc": "paint of shopfront, door, window frames and shutters"},
    "tile_color": {"default": "#c58c72", "desc": "tint of the clay roof tiles"},
    "light": {"default": 1.0, "min": 0.3, "max": 1.6, "desc": "multiplier on all light intensities"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: grime, damp, moss, slipped tiles, crooked timbers"},
}
VARIANTS = {
    "narrow_tall": {"width": 8.0, "depth": 6.0, "storeys": 3, "gallery_depth": 2.5, "posts": 3, "pattern": "close",
                    "roof": "slate", "ground_wall": "brick", "paint": "#6a1f1c", "plaster": "#d9c9a4", "furniture": 2,
                    "dormers": 1, "balustrade": "cross", "pitch": 50.0, "neon_color": "#ff4a6e", "sign_text": "KAFFEE",
                    "fascia_text_l": "TEA ROOM", "barrels": 1, "seed": 7},
    "long_inn": {"width": 14.0, "depth": 8.0, "storeys": 2, "gallery_depth": 4.0, "posts": 6, "pattern": "cross",
                 "ground_wall": "plaster", "plaster": "#ece3cc", "timber": "#36281d", "paint": "#1c2f4a", "furniture": 6,
                 "dormers": 3, "pitch": 42.0, "barrels": 1, "neon_color": "#37e0ff", "sign_text": "GRAND CAFE", "balcony_depth": 1.7,
                 "seed": 12},
    "weathered_ochre": {"width": 9.5, "depth": 6.5, "storeys": 2, "gallery_depth": 3.2, "posts": 3, "pattern": "square",
                        "ground_wall": "stone", "plaster": "#cfa565", "timber": "#5a4433", "paint": "#3c2a5a", "furniture": 1,
                        "dormers": 1, "pitch": 51.0, "sag": 0.17, "age": 1.0, "neon": False, "barrels": 2,
                        "sign_text": "MOKKA", "balustrade": "cross", "seed": 23},
    "three_storey_wide": {"width": 12.5, "depth": 9.0, "storeys": 3, "gallery_depth": 3.4, "posts": 5, "pattern": "braced",
                          "roof": "slate", "ground_wall": "brick", "plaster": "#e9dcc0", "paint": "#254a55", "furniture": 4,
                          "dormers": 2, "pitch": 44.0, "barrels": 1, "sign_text": "CAFE ROYAL", "seed": 31},
}


# ------------------------------------------------------------------ helpers
def frame_m(o, ex, ey, sc=1.0):
    """4x4 matrix whose local X / Y map to ex / ey (orthonormal), Z to ex x ey, origin o."""
    ex = Vector(ex).normalized()
    ey = Vector(ey).normalized()
    ez = ex.cross(ey)
    return Matrix(((ex.x * sc, ey.x * sc, ez.x * sc, o[0]), (ex.y * sc, ey.y * sc, ez.y * sc, o[1]),
                   (ex.z * sc, ey.z * sc, ez.z * sc, o[2]), (0.0, 0.0, 0.0, 1.0)))


def beam(out, p0, p1, w, d, heading=None, bev=0.007, name="tm"):
    """Squared timber from p0 to p1: w = horizontal thickness, d = depth in the vertical plane."""
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    if heading is None:
        hz = degrees(atan2(dy, dx))
        run = hypot(dx, dy)
    else:
        hz = heading
        a = radians(hz)
        run = dx * cos(a) + dy * sin(a)
    ln = sqrt(dx * dx + dy * dy + dz * dz)
    o = box(name, (ln, w, d), loc=((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2),
            rot=(0, -degrees(atan2(dz, run)), hz), bevel=bev, segments=1)
    uv_grain(o, axis='X')
    out.append(o)
    return o


def lite_beam(out, p0, p1, w, d, heading, c=0.006, name="tm"):
    """Cheap applied member (10 tris): no back face, the long arrises of the front face chamfered."""
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    a = radians(heading)
    run = dx * cos(a) + dy * sin(a)
    l = sqrt(dx * dx + dy * dy + dz * dz) / 2
    hw_, hd = w / 2, d / 2
    cc = min(c, d * 0.22)
    o = mesh_object(name, [(-l, hw_, -hd), (l, hw_, -hd), (l, hw_, hd), (-l, hw_, hd),
                           (-l, -hw_, -hd + cc), (l, -hw_, -hd + cc), (l, -hw_, hd - cc), (-l, -hw_, hd - cc)],
                    [(4, 5, 6, 7), (7, 6, 2, 3), (0, 1, 5, 4), (5, 1, 2, 6), (0, 4, 7, 3)])
    place(o, loc=((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2), rot=(0, -degrees(atan2(dz, run)), heading))
    uv_grain(o, axis='X')
    out.append(o)
    return o


class Wall:
    """A vertical wall plane: u runs along the wall, `out` is the distance in front of its face."""

    def __init__(self, x0, y0, heading, length):
        a = radians(heading)
        self.h = heading
        self.t = (cos(a), sin(a))
        self.n = (sin(a), -cos(a))
        self.o = (x0, y0)
        self.L = length

    def pt(self, u, z, out=0.0):
        return (self.o[0] + self.t[0] * u + self.n[0] * out, self.o[1] + self.t[1] * u + self.n[1] * out, z)

    def member(self, lst, u0, z0, u1, z1, face, proud=0.03, back=0.07, bev=0.007):
        c = (proud - back) / 2
        if bev < 0.0101:
            return lite_beam(lst, self.pt(u0, z0, c), self.pt(u1, z1, c), proud + back, face, self.h, c=max(bev, 0.003))
        return beam(lst, self.pt(u0, z0, c), self.pt(u1, z1, c), proud + back, face, heading=self.h, bev=bev)


def slab(name, M, size):
    """Box of `size` centred on the local origin of matrix M (baked into the vertices)."""
    o = box(name, size)
    transform_verts(o, lambda co: M @ co)
    return o


def area2(pts):
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))


def ccw(pts):
    return pts if area2(pts) > 0 else list(reversed(pts))


def tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def brace_profile(R, t):
    """Arch brace filling the corner between a post (u = 0) and a beam soffit (v = 0)."""
    Ro = R + t
    pts = [(R + R * cos(radians(a)), -R + R * sin(radians(a))) for a in [180 - 90 * k / 8 for k in range(9)]]
    a0 = 180 - degrees(asin(R / Ro))
    a1 = 180 - degrees(acos(R / Ro))
    pts += [(max(0.0, R + Ro * cos(radians(a))), min(0.0, -R + Ro * sin(radians(a)))) for a in [a0 + (a1 - a0) * k / 4 for k in range(5)]]
    return ccw(pts)


def stamp(name, kit, matrices):
    """kit: list of (material, [objects at the local origin]); stamps every material group at the matrices."""
    out = []
    for i, (m, objs) in enumerate(kit):
        base = join(objs, f"{name}_b{i}")
        assign(base, m)
        if matrices:
            out.append(instances_to_mesh(f"{name}_{i}", base, matrices, keep_base=False))
        else:
            delete([base])
    return out


def build(p, rng):
    W, D, gd = float(p.width), float(p.depth), float(p.gallery_depth)
    hw = W / 2
    ns = int(p.storeys)
    age = float(p.age)
    G, U = 3.4, 2.75                    # first-floor level, upper storey height
    GB0, GB1 = 2.95, 3.25               # soffit / top of bressummer and girders (clear headroom 2.95 m)
    B = min(float(p.balcony_depth), gd - 0.9)
    ys = -gd                            # front plane of the gallery
    yb = ys + B                         # face of the balcony (first-floor front) wall
    yp = ys + 0.14                      # post axis
    E = G + U * (ns - 1)                # eaves / wall-plate level
    th = radians(p.pitch)
    tn, cs, sn = tan(th), cos(th), sin(th)
    half = (D + gd) / 2
    yr = (D - gd) / 2                   # ridge line
    ov, oe, bed = 0.4, 0.5, 0.14        # verge overhang, eaves overhang, tile bed above the plate
    seed = int(p.seed)

    # ---- materials
    oak = mat("wood_oak", p.timber)
    boards = mat("wood_planks", p.timber)
    plas = mat("plaster_lime", p.plaster)
    stone = mat("stone_smooth", "#a99d8a")
    gwm = {"stone": mat("rubble_fieldstone"), "brick": mat("brick_red"), "plaster": plas}[p.ground_wall]
    brick = mat("brick_red")
    clay = p.roof == "clay"
    tilem = mat("terracotta", p.tile_color) if clay else mat("slate", "#8a909c")
    potm = tilem if clay else mat("terracotta", p.tile_color)
    paintm = mat("wood_painted", p.paint)
    iron = mat("iron_wrought")
    lead = mat("lead")
    brass = mat("brass")
    litm = mat("window_lit", "#ffc37a", glow=2.6)
    darkm = mat("window_dark")
    flame = mat("emissive", "#ffb45e", glow=9)
    lglass = mat("glass_lantern")
    flagm = mat("flagstone_york")

    T, BD, PL, GW, ST, FR, GLIT, GDARK, IR, LD, BR, BK, RW = ([] for _ in range(13))

    def zbed(y):
        return E + bed + tn * (half - abs(y - yr))

    def sagz(x, y):
        t = clamp(1.0 - abs(y - yr) / (half + oe))
        return -float(p.sag) * sin(pi * clamp((x + hw) / W)) * t ** 1.5

    # ------------------------------------------------------------------ windows
    wr = rng.fork("windows")

    def window(wall, uc, z, w, h, lit, door=False, shutters=False, proud=0.045):
        f = 0.06
        for sg in (-1, 1):
            wall.member(FR, uc + sg * (w / 2 - f / 2), z, uc + sg * (w / 2 - f / 2), z + h, f, proud, 0.0, bev=0.004)
        wall.member(FR, uc - w / 2 + f, z + h - f / 2, uc + w / 2 - f, z + h - f / 2, f, proud, 0.0, bev=0.004)
        wall.member(FR, uc - w / 2 - 0.04, z + 0.03, uc + w / 2 + 0.04, z + 0.03, 0.06, proud + 0.045, 0.0, bev=0.004)
        nl = 2 if w > 0.75 else 1
        if nl == 2:
            wall.member(FR, uc, z + 0.06, uc, z + h - f, 0.05, proud - 0.006, 0.0, bev=0.003)
        ny = max(2, int(round((h - 0.12) / 0.34)))
        for j in range(1, ny):
            zz = z + 0.06 + (h - 0.12) * j / ny
            wall.member(FR, uc - w / 2 + f, zz, uc + w / 2 - f, zz, 0.022, proud - 0.018, 0.0, bev=0.0)
        lw = (w - 2 * f) / nl
        for k in range(nl):
            uu = uc - w / 2 + f + lw * (k + 0.5)
            wall.member(FR, uu, z + 0.06, uu, z + h - f, 0.022, proud - 0.02, 0.0, bev=0.0)
        if door:                                             # panelled lower third of a french door
            wall.member(FR, uc - w / 2 + f, z + 0.06 + 0.3, uc + w / 2 - f, z + 0.06 + 0.3, 0.6, proud - 0.012, 0.0, bev=0.004)
        # CLEARANCE RULE: the lit glass runs 1 cm into the sill as it does into the head (its underside lay in the plane of
        # the sill's top)
        g = box("glass", (w - 2 * f + 0.01, 0.008, h - f - 0.04), loc=wall.pt(uc, z + (0.06 + h - f) / 2, 0.012), rot=(0, 0, wall.h))
        (GLIT if lit else GDARK).append(g)
        if shutters:
            sw = w / 2 - 0.012
            nb_ = 3
            bw = sw / nb_
            for sg in (-1, 1):
                c = uc + sg * (w / 2 + 0.025 + sw / 2)
                dz = wr.jitter(0, 0.012)
                for k in range(nb_):
                    uu = c - sw / 2 + bw * (k + 0.5)
                    wall.member(FR, uu, z + 0.03 + dz, uu + wr.jitter(0, 0.003), z + h - 0.03 + dz, bw - 0.006, 0.056, -0.034, bev=0.003)
                for fz in (0.18, 0.82):
                    wall.member(FR, c - sw / 2 + 0.02, z + h * fz + dz, c + sw / 2 - 0.02, z + h * fz + dz, 0.07, 0.07, -0.056, bev=0.003)
                for fz in (0.18, 0.82):                      # strap hinges
                    ue = c - sg * sw / 2
                    wall.member(IR, ue, z + h * fz + dz, ue + sg * sw * 0.7, z + h * fz + dz, 0.03, 0.076, -0.07, bev=0.0)

    # ------------------------------------------------------------------ half-timbered wall
    def frame_wall(wall, z0, h, plan, r, bay=1.9, ends=0.2):
        L = wall.L
        pat = p.pattern
        zs, zt = z0 + 0.18, z0 + h - 0.2
        wall.member(T, 0.08, z0 + 0.09, L - 0.08, z0 + 0.09, 0.18)
        wall.member(T, 0.08, z0 + h - 0.1, L - 0.08, z0 + h - 0.1, 0.2)
        a, b = ends, L - ends
        nb = max(1, int(round((b - a) / bay)))
        us = [a + (b - a) * i / nb for i in range(nb + 1)]
        for i in range(1, nb):
            us[i] += r.jitter(0, 0.03)
            ln = r.jitter(0, 0.012 * (0.4 + age))
            wall.member(T, us[i] - ln, zs, us[i] + ln, zt, r.vary(0.17, 0.08))
        for i in range(nb):
            ua = us[i] + (0.0 if i == 0 else 0.085)
            ub = us[i + 1] - (0.0 if i == nb - 1 else 0.085)
            bw = ub - ua
            uc = (ua + ub) / 2
            op = plan(i, nb)
            if op:
                door = op.get("door", False)
                w = min(1.0 if door else (0.84 if op.get("shutters") else 1.15), bw - 0.36)
                if w < 0.5:
                    op = None
            if op:
                zsill = zs if door else z0 + 0.95
                wh = min(2.05 if door else 1.2, zt - zsill - 0.16)
                for sg in (-1, 1):
                    wall.member(T, uc + sg * (w / 2 + 0.06), zs, uc + sg * (w / 2 + 0.06), zt, 0.12)
                wall.member(T, uc - w / 2, zsill + wh + 0.065, uc + w / 2, zsill + wh + 0.065, 0.13)
                if not door:
                    wall.member(T, uc - w / 2, zsill - 0.065, uc + w / 2, zsill - 0.065, 0.13)
                    if pat == "cross":
                        wall.member(T, uc - w / 2 + 0.03, zs + 0.03, uc + w / 2 - 0.03, zsill - 0.16, 0.1, proud=0.026)
                        wall.member(T, uc - w / 2 + 0.03, zsill - 0.16, uc + w / 2 - 0.03, zs + 0.03, 0.1, proud=0.022)
                    else:
                        wall.member(T, uc, zs, uc, zsill - 0.13, 0.11)
                side = (bw - w - 0.24) / 2
                if side > 0.3:
                    zm = z0 + h * 0.5 + r.jitter(0, 0.05)
                    for sg in (-1, 1):
                        wall.member(T, uc + sg * (w / 2 + 0.12), zm, ua if sg < 0 else ub, zm, 0.12)
                window(wall, uc, zsill, w, wh, op.get("lit", False), door=door, shutters=op.get("shutters", False))
                continue
            zm = z0 + h * 0.5 + r.jitter(0, 0.06)
            wall.member(T, ua, zm, ub, zm, 0.14)
            if pat in ("close", "square"):
                k = max(1, int(round(bw / (0.42 if pat == "close" else 0.95))))
                for j in range(1, k):
                    uu = ua + bw * j / k + r.jitter(0, 0.015)
                    fw = r.vary(0.1 if pat == "close" else 0.12, 0.1)
                    wall.member(T, uu, zs, uu + r.jitter(0, 0.01), zm - 0.07, fw)
                    wall.member(T, uu, zm + 0.07, uu + r.jitter(0, 0.01), zt, fw)
            else:
                rise = (i % 2 == 0) if i != nb - 1 else False
                u0, u1 = (ua + 0.02, ub - 0.02) if rise else (ub - 0.02, ua + 0.02)
                wall.member(T, u0, zs + 0.03, u1, zm - 0.1, 0.13, proud=0.026)
                if pat == "cross":
                    wall.member(T, u1, zs + 0.03, u0, zm - 0.1, 0.13, proud=0.022)
                    wall.member(T, u0, zm + 0.1, u1, zt - 0.03, 0.11, proud=0.026)
                    wall.member(T, u1, zm + 0.1, u0, zt - 0.03, 0.11, proud=0.022)
                else:
                    k = 2 if bw > 1.5 else 1
                    for j in range(1, k + 1):
                        uu = ua + bw * j / (k + 1)
                        wall.member(T, uu, zm + 0.07, uu, zt, 0.11)

    def corner_post(x, y, sx, sy, z0, h):
        o = box("cpost", (0.24, 0.24, h), loc=(x - sx * (0.12 - 0.036), y - sy * (0.12 - 0.036), z0 + h / 2), bevel=0.012, segments=1)
        uv_grain(o, axis='Z')
        T.append(o)

    def plan_maker(r, lit_p, shutters, phase=0, door_mid=False, force_lit=None):
        def plan(i, nb):
            if door_mid:
                if i == nb // 2 or (nb >= 5 and i == nb // 2 - 2):
                    return {"door": True, "lit": True}
                return {"lit": True, "shutters": False} if (i - nb // 2) % 2 == 0 else None
            if nb == 1 or (i + phase) % 2 == 0:
                return {"lit": r.chance(lit_p) or (force_lit is not None and i == min(force_lit, nb - 1)), "shutters": shutters and r.chance(0.8)}
            return None
        return plan

    # ------------------------------------------------------------------ ground floor
    gcore = box("gcore", (W, D, G), loc=(0, D / 2, G / 2))
    densify(gcore, 0.9)
    GW.append(gcore)
    wf = Wall(-hw, 0.0, 0, W)
    g_walls = [Wall(hw, 0.0, 90, D), Wall(hw, D, 180, W), Wall(-hw, D, 270, D)]
    gr = rng.fork("ground")

    def plinth_run(wall, u0, u1, hgt=0.5):
        if u1 - u0 < 0.15:
            return
        k = max(1, int(round((u1 - u0) / 0.95)))
        cuts = [u0 + (u1 - u0) * j / k + (gr.jitter(0, 0.12) if 0 < j < k else 0.0) for j in range(k + 1)]
        for j in range(k):
            wall.member(ST, cuts[j] + 0.004, hgt / 2, cuts[j + 1] - 0.004, hgt / 2, hgt + gr.jitter(0, 0.008), 0.05 + gr.jitter(0, 0.006), 0.05, bev=0.014)

    def ground_window(wall, uc, lit):
        w, h, zsill = 1.0, 1.4, 1.02
        wall.member(ST, uc - w / 2 - 0.1, zsill - 0.06, uc + w / 2 + 0.1, zsill - 0.06, 0.12, 0.1, 0.05, bev=0.012)
        wall.member(ST, uc - w / 2 - 0.2, zsill + h + 0.12, uc + w / 2 + 0.2, zsill + h + 0.12, 0.24, 0.04, 0.05, bev=0.012)
        window(wall, uc, zsill, w, h, lit, shutters=True)

    for wi, gwall in enumerate(g_walls):
        L = gwall.L
        if wi == 1:                                           # back wall: service door + windows
            k = max(2, int(round(L / 3.0)))
            di = gr.pick(list(range(k)))
            ud0 = ud1 = None
            lit_i = gr.pick([j for j in range(k) if j != di])
            for j in range(k):
                uc = L * (j + 0.5) / k
                if j != di:
                    ground_window(gwall, uc, j == lit_i or gr.chance(0.25))
                    continue
                dw, dh = 0.95, 2.05
                ud0, ud1 = uc - dw / 2 - 0.2, uc + dw / 2 + 0.2
                gwall.member(ST, uc - 0.75, 0.085, uc + 0.75, 0.085, 0.17, 0.42, 0.05, bev=0.015)
                for sg in (-1, 1):
                    gwall.member(T, uc + sg * (dw / 2 + 0.07), 0.17, uc + sg * (dw / 2 + 0.07), 0.17 + dh, 0.14, 0.06, 0.06)
                gwall.member(T, uc - dw / 2 - 0.22, 0.17 + dh + 0.1, uc + dw / 2 + 0.22, 0.17 + dh + 0.1, 0.2, 0.07, 0.06)
                nbd = 5
                for q in range(nbd):
                    uu = uc - dw / 2 + dw * (q + 0.5) / nbd
                    gwall.member(FR, uu, 0.18, uu, 0.17 + dh, dw / nbd - 0.006, 0.035, 0.0, bev=0.004)
                for zz in (0.55, 1.85):
                    gwall.member(IR, uc - dw / 2 + 0.01, zz, uc + dw / 2 - 0.2, zz, 0.045, 0.043, -0.035, bev=0.002)
                IR.append(torus("ring", 0.045, 0.007, loc=gwall.pt(uc + dw / 2 - 0.13, 1.08, 0.05), rot=(90, 0, gwall.h), major_seg=12, minor_seg=5))
            if ud0 is not None:
                plinth_run(gwall, 0.3, ud0 - 0.55)
                plinth_run(gwall, ud1 + 0.55, L - 0.3)
        else:
            k = max(1, int(round(L / 3.3)))
            lit_i = gr.pick(list(range(k)))
            for j in range(k):
                ground_window(gwall, L * (j + 0.5) / k + gr.jitter(0, 0.15), j == lit_i or gr.chance(0.3))
            plinth_run(gwall, 0.3, L - 0.3)
    # quoins on the four corners
    for (cx, cy, sx, sy) in ((hw, 0, 1, -1), (hw, D, 1, 1), (-hw, D, -1, 1), (-hw, 0, -1, -1)):
        zq = 0.0
        k = 0
        while zq < G - 0.5:
            hq = gr.vary(0.34, 0.08)
            lx, ly = (0.54, 0.3) if k % 2 == 0 else (0.3, 0.54)
            lx += gr.jitter(0, 0.03)
            ly += gr.jitter(0, 0.03)
            pr = 0.055
            q = box("quoin", (lx + pr, ly + pr, hq - 0.008), loc=(cx - sx * (lx - pr) / 2, cy - sy * (ly - pr) / 2, zq + hq / 2), bevel=0.014, segments=1)
            ST.append(q)
            zq += hq
            k += 1

    # ------------------------------------------------------------------ cafe front under the gallery
    n = max(3, min(int(p.posts), int((W - 0.28) / 2.2) + 1), int(ceil((W - 0.28) / 4.7)) + 1)
    sp = (W - 0.28) / (n - 1)
    px = [-hw + 0.14 + i * sp for i in range(n)]
    xd = 0.0 if n % 2 == 0 else sp / 2
    sh = GB0 - 0.05
    door = use("door_shop", width=1.3, height=sh, paint=p.paint, sign_text="OPEN", seed=seed + 2)
    door.place((xd, 0, 0), sockets=False)
    lights_win = []
    occupied = [(xd - 0.65, xd + 0.65)]
    for side, txt, goods in ((-1, p.fascia_text_l, "cakes"), (1, p.fascia_text_r, "bottles")):
        z0, z1 = (-hw + 0.42, xd - 0.93) if side < 0 else (xd + 0.93, hw - 0.42)
        zl = z1 - z0
        if zl < 1.7:
            continue
        ww = min(3.5, zl)
        xc = (z0 + z1) / 2
        swin = use("shop_window", width=ww, height=sh, bays=max(2, int(round(ww / 1.2))), paint=p.paint, goods=goods,
                   depth=0.4, glow=2.6, sign_text=txt, seed=seed + 5 + side)
        swin.place((xc, 0, 0), sockets=False)
        occupied.append((xc - ww / 2, xc + ww / 2))
        lights_win.append((xc, ww))
    occupied.sort()
    edge = 0.22
    for (a, b) in occupied + [(hw - 0.22, hw)]:
        plinth_run(wf, edge + hw, a + hw - 0.01, hgt=0.62)
        edge = b + 0.01
    # wall plate carried on pilaster posts, with stone bases
    beam(T, (-hw + 0.02, -0.11, (GB0 + GB1) / 2), (hw - 0.02, -0.11, (GB0 + GB1) / 2), 0.22, GB1 - GB0, bev=0.012)
    for sx in (-1, 1):
        ST.append(box("pilbase", (0.3, 0.3, 0.5), loc=(sx * (hw - 0.13), -0.13, 0.25), bevel=0.02, segments=1))
        o = box("pilaster", (0.22, 0.22, GB0 - 0.5), loc=(sx * (hw - 0.13), -0.13, (GB0 + 0.5) / 2), bevel=0.02, segments=1)
        uv_grain(o, axis='Z')
        T.append(o)

    # ------------------------------------------------------------------ gallery: posts, beams, joists, braces
    pr_ = rng.fork("posts")
    ptop = G + U - 0.26
    for x in px:
        sb = stone_block("plinth", (0.46, 0.46, 0.62), loc=(x, yp, 0.31), rng=pr_, chamfer=0.022, detail=0.16)
        ST.append(sb)
        ST.append(box("plcap", (0.53, 0.53, 0.07), loc=(x, yp, 0.655), bevel=0.022, segments=1))
        post = box("post", (0.28, 0.28, ptop - 0.69), loc=(x, yp, (ptop + 0.69) / 2), rot=(0, 0, pr_.jitter(0, 0.8)), bevel=0.03, segments=1)
        uv_grain(post, axis='Z')
        T.append(post)
        T.append(box("collar", (0.33, 0.33, 0.07), loc=(x, yp, GB0 - 0.82), bevel=0.012, segments=1))
        T.append(box("collar", (0.33, 0.33, 0.06), loc=(x, yp, G + 1.14), bevel=0.012, segments=1))
        beam(T, (x, yp + 0.12, (GB0 + GB1) / 2), (x, -0.2, (GB0 + GB1) / 2), 0.22, GB1 - GB0, bev=0.012)          # girder
    beam(T, (-hw, yp, (GB0 + GB1) / 2), (hw, yp, (GB0 + GB1) / 2), 0.24, GB1 - GB0, bev=0.012)                     # bressummer
    beam(T, (-hw, yp, G + U - 0.13), (hw, yp, G + U - 0.13), 0.24, 0.26, bev=0.012)                               # top plate on the posts
    for sx in (-1, 1):
        beam(T, (sx * (hw - 0.12), yp + 0.12, G + U - 0.13), (sx * (hw - 0.12), yb + 0.05, G + U - 0.13), 0.22, 0.26, bev=0.012)
    y = ys + 0.55
    while y < -0.3:                                            # joists
        beam(T, (-hw + 0.1, y, 3.28), (hw - 0.1, y, 3.28), 0.09, 0.18, bev=0.004)
        y += 0.52
    deck = box("deck", (W - 0.02, gd - 0.02, 0.03), loc=(0, -gd / 2, G - 0.015))
    uv_grain(deck, axis='Y')
    BD.append(deck)
    beam(BD, (-hw + 0.01, ys + 0.025, 3.335), (hw - 0.01, ys + 0.025, 3.335), 0.04, 0.2, bev=0.006)               # fascia boards
    for sx in (-1, 1):
        beam(BD, (sx * (hw - 0.025), ys + 0.03, 3.335), (sx * (hw - 0.025), -0.01, 3.335), 0.04, 0.2, bev=0.006)
    # balcony ceiling: boards on small joists
    ceil_ = box("bceil", (W - 0.3, B - 0.1, 0.03), loc=(0, (ys + 0.2 + yb) / 2, G + U - 0.055))
    uv_grain(ceil_, axis='X')
    BD.append(ceil_)
    x = -hw + 0.75
    while x < hw - 0.5:
        beam(T, (x, yp + 0.1, G + U - 0.14), (x, yb + 0.03, G + U - 0.14), 0.09, 0.14, bev=0.004)
        x += 0.72
    # arch braces
    bz = 0.12
    bb = extrude_profile("brace", brace_profile(0.74, 0.13), bz, bevel=0.006, segments=1)
    uv_grain(bb, axis='X')
    assign(bb, oak)
    bm = []
    for i, x in enumerate(px):
        if i < n - 1:
            bm.append(frame_m((x + 0.135, yp + bz / 2, GB0), (1, 0, 0), (0, 0, 1)))
            bm.append(frame_m((x + 0.135, yp + bz / 2, ptop), (1, 0, 0), (0, 0, 1), 0.6))
        if i > 0:
            bm.append(frame_m((x - 0.135, yp - bz / 2, GB0), (-1, 0, 0), (0, 0, 1)))
            bm.append(frame_m((x - 0.135, yp - bz / 2, ptop), (-1, 0, 0), (0, 0, 1), 0.6))
        bm.append(frame_m((x - bz / 2, yp + 0.135, GB0), (0, 1, 0), (0, 0, 1), 0.85))
    braces = instances_to_mesh("braces", bb, bm, keep_base=False)

    # balustrade
    zb0, zb1 = G + 0.11, G + 1.0
    runs = [((px[i] + 0.14, yp), (px[i + 1] - 0.14, yp)) for i in range(n - 1)]
    for sx in (-1, 1):
        runs.append(((sx * (hw - 0.14), yp + 0.14), (sx * (hw - 0.14), yb + 0.03)))
    sm = []
    cross = p.balustrade == "cross"
    for (a, b) in runs:
        ln = hypot(b[0] - a[0], b[1] - a[1])
        if ln < 0.25:
            continue
        ex = ((b[0] - a[0]) / ln, (b[1] - a[1]) / ln, 0.0)
        beam(T, (a[0], a[1], zb0), (b[0], b[1], zb0), 0.07, 0.07, bev=0.006)
        beam(T, (a[0], a[1], zb1), (b[0], b[1], zb1), 0.11, 0.075, bev=0.014)
        if cross:
            k = max(1, int(round(ln / 0.95)))
            for j in range(k + 1):
                if 0 < j < k:
                    q = (a[0] + ex[0] * ln * j / k, a[1] + ex[1] * ln * j / k)
                    o = beam(T, (q[0], q[1], zb0 + 0.03), (q[0], q[1], zb1 - 0.03), 0.06, 0.07, heading=degrees(atan2(ex[1], ex[0])), bev=0.005)
            for j in range(k):
                q0 = (a[0] + ex[0] * (ln * j / k + 0.04), a[1] + ex[1] * (ln * j / k + 0.04))
                q1 = (a[0] + ex[0] * (ln * (j + 1) / k - 0.04), a[1] + ex[1] * (ln * (j + 1) / k - 0.04))
                hd = degrees(atan2(ex[1], ex[0]))
                beam(T, (q0[0], q0[1], zb0 + 0.05), (q1[0], q1[1], zb1 - 0.05), 0.04, 0.06, heading=hd, bev=0.004)
                beam(T, (q0[0], q0[1], zb1 - 0.05), (q1[0], q1[1], zb0 + 0.05), 0.03, 0.06, heading=hd, bev=0.004)
        else:
            k = max(1, int(ln / 0.175))
            for j in range(k):
                d = ln * (j + 0.5) / k
                sm.append(frame_m((a[0] + ex[0] * d, a[1] + ex[1] * d, zb0 + 0.03), ex, (0, 0, 1))
                          @ Matrix.Translation((0, 0, -0.011)))
    if sm:
        hs = zb1 - zb0 - 0.06
        hp = [(0.055, 0.0), (0.055, 0.13), (0.026, 0.22), (0.055, 0.36), (0.055, 0.52), (0.026, 0.66), (0.055, 0.76), (0.055, 1.0)]
        prof = [(x_, v * hs) for (x_, v) in hp] + [(-x_, v * hs) for (x_, v) in reversed(hp)]
        sb_ = extrude_profile("splat", ccw(prof), 0.022)
        uv_grain(sb_, axis='Y')
        assign(sb_, oak)
        splats = instances_to_mesh("splats", sb_, sm, keep_base=False)

    # oak pegs at the joints of the gallery frame
    pg = cylinder("peg", 0.012, 0.03, verts=6)
    assign(pg, oak)
    pgm = []
    for x in px:
        for (dxp, zp) in ((-0.06, GB0 - 0.42), (0.06, GB0 - 0.5), (-0.07, GB0 + 0.1), (0.07, GB0 + 0.2), (-0.06, ptop - 0.3), (0.06, ptop - 0.25),
                          (-0.07, ptop + 0.08), (0.07, ptop + 0.17), (0.0, zb0), (0.0, zb1)):
            if abs(x + dxp) < hw - 0.02:
                pgm.append(trs((x + dxp + pr_.jitter(0, 0.008), ys - 0.004, zp + pr_.jitter(0, 0.01)), (90, 0, 0)))
    instances_to_mesh("pegs", pg, pgm, keep_base=False)

    # flower troughs hung outside the handrail
    npl = int(p.planters)
    if npl > 0:
        fr_ = rng.fork("planters")
        hedge = mat("hedge", "#5f7a3c")
        flw = mat("ceramic_glazed", p.flowers)
        bays_ = sorted(range(n - 1), key=lambda i: -abs((px[i] + px[i + 1]) / 2))
        bays_ = [i for i in bays_ if not (p.neon and abs((px[i] + px[i + 1]) / 2) < 0.6)][:npl]
        fol, flo = [], []
        for i in bays_:
            xc = (px[i] + px[i + 1]) / 2 + fr_.jitter(0, 0.15)
            Lp = min(1.7, sp - 1.0)
            y0_, y1_ = ys - 0.125, ys + 0.075
            zt_ = zb1 - 0.05
            beam(T, (xc - Lp / 2, y0_ + 0.011, zt_ - 0.08), (xc + Lp / 2, y0_ + 0.011, zt_ - 0.08), 0.022, 0.16, bev=0.004)
            beam(T, (xc - Lp / 2, y1_ - 0.011, zt_ - 0.08), (xc + Lp / 2, y1_ - 0.011, zt_ - 0.08), 0.022, 0.16, bev=0.004)
            beam(T, (xc - Lp / 2 + 0.01, (y0_ + y1_) / 2, zt_ - 0.15), (xc + Lp / 2 - 0.01, (y0_ + y1_) / 2, zt_ - 0.15), 0.16, 0.02, bev=0.0)
            for sx in (-1, 1):
                T.append(box("tr_end", (0.022, 0.2, 0.16), loc=(xc + sx * (Lp / 2 - 0.011), (y0_ + y1_) / 2, zt_ - 0.08), bevel=0.004, segments=1))
                beam(IR, (xc + sx * (Lp / 2 - 0.2), y0_ - 0.004, zt_ - 0.17), (xc + sx * (Lp / 2 - 0.2), y0_ - 0.004, zt_ + 0.0), 0.004, 0.03, bev=0.0)
                beam(IR, (xc + sx * (Lp / 2 - 0.2), y0_ - 0.004, zt_ - 0.17), (xc + sx * (Lp / 2 - 0.2), y1_ + 0.02, zt_ - 0.17), 0.03, 0.004, bev=0.0)
            k = int(Lp / 0.17)
            for j in range(k):
                xx = xc - Lp / 2 + Lp * (j + 0.5) / k + fr_.jitter(0, 0.03)
                r_ = fr_.u(0.1, 0.15)
                yy = (y0_ + y1_) / 2 + fr_.jitter(0, 0.03)
                fol.append(sphere("fol", r_, loc=(xx, yy, zt_ + r_ * 0.45), segments=7, rings=5, scale=(1.15, 1.0, fr_.u(0.7, 1.0))))
                if fr_.chance(0.7):                          # trailing growth over the front board
                    dr_ = fr_.u(0.14, 0.42)
                    fol.append(sphere("trail", 0.07, loc=(xx + fr_.jitter(0, 0.04), y0_ - 0.02, zt_ - dr_ / 2 + 0.02), segments=6, rings=4, scale=(1.0, 0.6, dr_ / 0.12 + 0.5)))
                for q in range(fr_.pick([1, 2, 2, 3])):
                    a_ = fr_.u(0, 2 * pi)
                    e_ = fr_.u(0.2, 1.2)
                    flo.append(sphere("flw", fr_.u(0.022, 0.034), loc=(xx + r_ * 1.1 * cos(a_) * sin(e_), yy + r_ * sin(a_) * sin(e_) * 0.95,
                                                                     zt_ + r_ * 0.45 + r_ * 0.85 * cos(e_)), segments=6, rings=4, scale=(1, 1, 0.7)))
        if fol:
            fo = join(fol, "foliage")
            assign(fo, hedge)
            displace_noise(fo, 0.02, 0.09, seed=seed)
            fw_o = join(flo, "flowers")
            assign(fw_o, flw)

    # ------------------------------------------------------------------ upper storeys
    ur = rng.fork("upper")
    for lv in range(1, ns):
        z0 = G + U * (lv - 1)
        y0 = yb if lv == 1 else ys
        core = box("ucore", (W, D - y0, U), loc=(0, (D + y0) / 2, z0 + U / 2))
        densify(core, 0.85)
        PL.append(core)
        walls = [Wall(-hw, y0, 0, W), Wall(hw, y0, 90, D - y0), Wall(hw, D, 180, W), Wall(-hw, D, 270, D - y0)]
        plans = [plan_maker(ur, 0.75, lv > 1, door_mid=(lv == 1)), plan_maker(ur, 0.4, True, phase=lv, force_lit=1),
                 plan_maker(ur, 0.3, True, phase=lv + 1, force_lit=2), plan_maker(ur, 0.4, True, phase=lv + 1, force_lit=0)]
        for wl, pl in zip(walls, plans):
            frame_wall(wl, z0, U, pl, ur)
        for (cx, cy, sx, sy) in ((hw, y0, 1, -1), (hw, D, 1, 1), (-hw, D, -1, 1), (-hw, y0, -1, -1)):
            corner_post(cx, cy, sx, sy, z0, U)

    # ------------------------------------------------------------------ gables
    Lg = D + gd
    hg = half * tn
    for gi, gw_ in enumerate((Wall(hw, ys, 90, Lg), Wall(-hw, D, 270, Lg))):
        tri = extrude_profile("tymp", ccw([(0, 0), (Lg, 0), (Lg, 0.1), (Lg / 2, hg + 0.1), (0, 0.1)]), 0.2,
                              loc=gw_.pt(0, E, -0.2), rot=(90, 0, gw_.h))
        densify(tri, 0.7)
        PL.append(tri)
        gw_.member(T, 0.06, E + 0.11, Lg - 0.06, E + 0.11, 0.22)
        for sg in (-1, 1):                                    # principal rafters
            gw_.member(T, Lg / 2 + sg * (Lg / 2 - 0.12), E + 0.2, Lg / 2 + sg * 0.05, E + hg - 0.06, 0.17, proud=0.026)
        zc = E + hg * 0.5
        hwid = (Lg / 2) * 0.5
        qx = min(0.6, hwid - 0.35)
        for sg in (-1, 1):
            gw_.member(T, Lg / 2 + sg * qx, E + 0.22, Lg / 2 + sg * qx, zc - 0.08, 0.15)                           # queen posts
        gw_.member(T, Lg / 2 - hwid + 0.12, zc, Lg / 2 + hwid - 0.12, zc, 0.16)                                    # collar
        gw_.member(T, Lg / 2, zc + 0.08, Lg / 2, E + hg - 0.2, 0.15)                                                # king strut
        u = 0.95
        while u < Lg / 2 - qx - 0.5:                                                                                # studs
            for sg in (-1, 1):
                uu = Lg / 2 + sg * (Lg / 2 - u)
                top = min(E + u * tn - 0.16, zc - 0.08) if (Lg / 2 - u) < hwid - 0.12 else E + u * tn - 0.16
                gw_.member(T, uu, E + 0.22, uu, top, 0.12)
            u += 1.0
        ww_ = min(0.8, 2 * qx - 0.25)
        window(gw_, Lg / 2, E + 0.62, ww_, min(0.9, zc - E - 0.85), lit=(gi == 0) == ur.chance(0.7))
        gw_.member(T, Lg / 2 - qx, E + 0.55, Lg / 2 + qx, E + 0.55, 0.11)
        # purlin / plate ends under the verge
        for yy in (ys + 0.1, ys + half * 0.5, yr, D - half * 0.5, D - 0.1):
            zz = E + tn * (half - abs(yy - yr)) - 0.07
            xa = hw - 0.05 if gi == 0 else -hw + 0.05
            xb_ = hw + ov - 0.05 if gi == 0 else -hw - ov + 0.05
            beam(RW, (xa, yy, zz), (xb_, yy, zz), 0.14, 0.16, bev=0.01)

    # ------------------------------------------------------------------ roof
    rr = rng.fork("roof")
    slope_len = (half + oe) / cs
    Wr = W + 2 * ov
    k_t = 1.0
    n_est = 2 * Wr * slope_len / (0.25 * 0.2)
    if n_est > 4300:
        k_t = sqrt(n_est / 4300)
    tw, tg = 0.25 * k_t, 0.2 * k_t
    Lt, tt = tg + 0.05, 0.018
    a_ = tw / 2 - 0.002
    zl, zu = 2.0 * tt, 0.9 * tt
    tb = mesh_object("tile", [(-a_, 0, zl), (a_, 0, zl), (a_, Lt, zu), (-a_, Lt, zu), (-a_, 0, zl - tt), (a_, 0, zl - tt)],
                     [(0, 1, 2, 3), (4, 5, 1, 0)])
    uv_box(tb)
    assign(tb, tilem)
    tmats = []
    slip = 0.006 + 0.03 * age

    def tile_plane(o, up, nrm, width, length, keep=None, sag=True):
        up = Vector(up).normalized()
        nrm = Vector(nrm).normalized()
        ax = up.cross(nrm)
        o = Vector(o)
        nrows = int(ceil((length - 0.03) / tg))
        ncols = max(1, int(round(width / tw)))
        twf = width / ncols
        for row in range(nrows):
            s = row * tg
            sy = min(1.0, (length - s + 0.02) / Lt)
            if sy < 0.3:
                continue
            odd = row % 2
            for j in (range(ncols + 1) if odd else range(ncols)):
                sx = twf / tw
                if odd:
                    a = -width / 2 + j * twf
                    if j == 0:
                        a += twf / 4
                        sx *= 0.5
                    elif j == ncols:
                        a -= twf / 4
                        sx *= 0.5
                else:
                    a = -width / 2 + (j + 0.5) * twf
                ds = rr.jitter(0, 0.006)
                rz = rr.jitter(0, 1.3)
                if rr.chance(slip):
                    ds -= rr.u(0.02, 0.07)
                    rz = rr.jitter(0, 5.0)
                pos = o + ax * (a + rr.jitter(0, 0.004)) + up * (s + ds) + nrm * (0.004 * (1 + sin(a * 1.7 + row * 0.9)))
                if keep is not None and not keep(pos):
                    continue
                if sag:
                    pos.z += sagz(pos.x, pos.y)
                tmats.append(frame_m(pos, ax, up) @ Matrix.Rotation(radians(rz), 4, 'Z') @ Matrix.Rotation(radians(rr.jitter(0, 0.9)), 4, 'X')
                             @ Matrix.Diagonal((sx * rr.vary(1.0, 0.02), sy, 1.0, 1.0)))

    # dormers (front slope)
    nd = int(p.dormers)
    dw_, hd_ = 1.3, 1.15
    ph = radians(47)
    yf = ys + max(0.95, half * 0.24)
    y_e = yf + hd_ / tn
    y_a = yf + (hd_ + dw_ / 2 * tan(ph)) / tn
    dxs = [-hw + W * (i + 0.5) / nd for i in range(nd)] if nd else []

    def keep_front(pos):
        for xdm in dxs:
            dx_ = abs(pos.x - xdm)
            if dx_ < dw_ / 2 - 0.16 and yf + 0.12 < pos.y < y_e:
                return False
            if y_e <= pos.y < y_a - 0.2 and dx_ < (dw_ / 2 - 0.16) * (y_a - 0.2 - pos.y) / (y_a - 0.2 - y_e):
                return False
        return True

    zeave = E + bed - tn * oe
    tile_plane((0, ys - oe, zeave), (0, cs, sn), (0, -sn, cs), Wr + 0.06, slope_len, keep=keep_front)
    tile_plane((0, D + oe, zeave), (0, -cs, sn), (0, sn, cs), Wr + 0.06, slope_len)
    for sg, up_, nrm_ in ((1, (0, cs, sn), (0, -sn, cs)), (-1, (0, -cs, sn), (0, sn, cs))):
        ym = yr - sg * (half + oe) / 2
        M = frame_m((0, ym, zbed(ym) - 0.022 * cs), (1, 0, 0), up_)
        dk = slab("roofdeck", M, (Wr - 0.06, slope_len, 0.03))
        uv_planar(dk, u_axis=(1, 0, 0), v_axis=up_)
        BK.append(dk)
        ye = ys - oe if sg > 0 else D + oe
        yi = ys + 0.45 if sg > 0 else D - 0.45
        x = -hw - ov + 0.22
        while x < hw + ov - 0.1:                              # rafter feet
            dzr = 0.1 / cs
            beam(RW, (x, ye + sg * 0.02, zbed(ye) - dzr), (x, yi, zbed(yi) - dzr), 0.085, 0.13, bev=0.005)
            x += 0.58
        # wind board closing the eaves between the rafters
        yw = ys + 0.03 if sg > 0 else D - 0.03
        beam(RW, (-hw, yw, E + 0.04), (hw, yw, E + 0.04), 0.05, 0.2, bev=0.0)
        for gx in (-1, 1):                                    # bargeboards
            xb_ = gx * (hw + ov - 0.01)
            dzb = 0.115 / cs
            o = beam(RW, (xb_, ye - sg * 0.02, zbed(ye) - dzb), (xb_, yr + sg * 0.02, zbed(yr) - dzb), 0.045, 0.25, bev=0.006)
    for gx in (-1, 1):                                        # apex finial posts
        xb_ = gx * (hw + ov - 0.01)
        RW.append(box("finial", (0.085, 0.085, 0.85), loc=(xb_, yr, zbed(yr) - 0.12), bevel=0.012, segments=1))
        RW.append(lathe("drop", [(0.0, -0.09), (0.03, -0.06), (0.05, -0.02), (0.05, 0.0), (0.0, 0.0)], segments=8, loc=(xb_, yr, zbed(yr) - 0.545)))
    # ridge tiles
    rt_r = 0.125
    rprof = [(rt_r * cos(radians(a)), rt_r * sin(radians(a)) - 0.035) for a in [-10 + 200 * k / 6 for k in range(7)]]
    rprof += [((rt_r - 0.02) * cos(radians(a)), (rt_r - 0.02) * sin(radians(a)) - 0.035) for a in [190 - 200 * k / 6 for k in range(7)]]
    rl = 0.46
    rtb = extrude_profile("ridge_tile", ccw(rprof), rl + 0.03)
    uv_box(rtb)
    assign(rtb, tilem)
    rm = []
    kr = int(ceil((Wr + 0.1) / rl))
    for j in range(kr):
        x = -(Wr + 0.1) / 2 + j * (Wr + 0.1) / kr
        zr = zbed(yr) + 0.02 + sagz(x + rl / 2, yr)
        rm.append(frame_m((x, yr + rr.jitter(0, 0.006), zr + rr.jitter(0, 0.004)), (0, 1, 0), (0, 0, 1))
                  @ Matrix.Rotation(radians(rr.jitter(0, 1.2)), 4, 'Y') @ Matrix.Rotation(radians(rr.jitter(0, 2.0)), 4, 'Z'))

    # dormer bodies
    dr = rng.fork("dormers")
    for di, xdm in enumerate(dxs):
        zb_ = zbed(yf) + sagz(xdm, yf) - 0.02
        dwall = Wall(xdm - dw_ / 2, yf, 0, dw_)
        PL.append(box("dfront", (dw_, 0.1, hd_ + 0.1), loc=(xdm, yf + 0.05, zb_ + hd_ / 2 - 0.05)))
        gt = extrude_profile("dgable", ccw([(-dw_ / 2, 0), (dw_ / 2, 0), (0, dw_ / 2 * tan(ph))]), 0.1, loc=(xdm, yf + 0.1, zb_ + hd_), rot=(90, 0, 0))
        uv_planar(gt, u_axis=(0, 0, 1), v_axis=(1, 0, 0))
        BD.append(gt)
        for sg in (-1, 1):
            dwall.member(T, dw_ / 2 + sg * (dw_ / 2 - 0.05), zb_, dw_ / 2 + sg * (dw_ / 2 - 0.05), zb_ + hd_, 0.1, 0.025, 0.05, bev=0.005)
        dwall.member(T, 0.0, zb_ + 0.2, dw_, zb_ + 0.2, 0.1, 0.03, 0.05, bev=0.005)
        dwall.member(T, -0.05, zb_ + hd_ - 0.02, dw_ + 0.05, zb_ + hd_ - 0.02, 0.12, 0.035, 0.05, bev=0.005)
        window(dwall, dw_ / 2, zb_ + 0.25, dw_ - 0.3, hd_ - 0.36, lit=dr.chance(0.8) or di == 0)
        for sg in (-1, 1):                                    # cheeks
            ck = extrude_profile("cheek", ccw([(0.02, -0.12 * tn), (y_e - yf + 0.12, hd_), (0.02, hd_)]), 0.06,
                                 loc=(xdm + sg * dw_ / 2 - (0.06 if sg > 0 else 0.0), yf, zb_), rot=(90, 0, 90))
            PL.append(ck)
            upv = (-sg * cos(ph), 0, sin(ph))
            nv = (sg * sin(ph), 0, cos(ph))
            ovd = 0.14
            sl = (dw_ / 2 + ovd) / cos(ph)
            y0_, y1_ = yf - 0.16, y_a + 0.12
            ze = zb_ + hd_ - ovd * tan(ph) + 0.035
            org = (xdm + sg * (dw_ / 2 + ovd), (y0_ + y1_) / 2, ze)
            tile_plane(org, upv, nv, y1_ - y0_, sl, keep=lambda q: q.z > zbed(q.y) - 0.05 + sagz(q.x, q.y), sag=False)
            Md = frame_m((xdm + sg * (dw_ / 2 + ovd) / 2, (y0_ + y1_) / 2, ze + sl * sin(ph) / 2 - 0.022), (0, sg, 0), upv)
            dk = slab("ddeck", Md, (y1_ - y0_ - 0.03, sl, 0.03))
            BK.append(dk)
            apex = (xdm, yf - 0.15, zb_ + hd_ + dw_ / 2 * tan(ph) - 0.07)
            beam(RW, (xdm + sg * (dw_ / 2 + ovd - 0.01), yf - 0.15, zb_ + hd_ - ovd * tan(ph) - 0.07), apex, 0.035, 0.15, bev=0.005)
        zra = zb_ + hd_ + dw_ / 2 * tan(ph) + 0.05
        kd = max(2, int(ceil((y_a - yf + 0.2) / rl)))
        for j in range(kd):
            rm.append(frame_m((xdm, yf - 0.18 + j * (y_a - yf + 0.2) / kd, zra), (-1, 0, 0), (0, 0, 1)) @ Matrix.Rotation(radians(-90), 4, 'Y')
                      @ Matrix.Diagonal((0.8, 0.8, (y_a - yf + 0.2) / kd / rl * 1.02, 1.0)))
    ridge = instances_to_mesh("ridge", rtb, rm, keep_base=False)
    tiles = instances_to_mesh("tiles", tb, tmats, keep_base=False)

    # chimney on the back slope
    cr = rng.fork("chimney")
    xc_ = hw * 0.42 * (1 if cr.chance(0.5) else -1)
    yc_ = yr + 0.8
    cw_, cd_ = 0.95, 0.6
    zc0 = zbed(yc_ + cd_ / 2) - 0.4
    zc1 = zbed(yr) + 1.0
    BRK = [box("stack", (cw_, cd_, zc1 - zc0), loc=(xc_, yc_, (zc0 + zc1) / 2), bevel=0.006, segments=1),
           box("corbel", (cw_ + 0.08, cd_ + 0.08, 0.075), loc=(xc_, yc_, zc1 - 0.2), bevel=0.005, segments=1),
           box("corbel", (cw_ + 0.17, cd_ + 0.17, 0.075), loc=(xc_, yc_, zc1 - 0.125), bevel=0.005, segments=1)]
    ST.append(box("ccap", (cw_ + 0.1, cd_ + 0.1, 0.085), loc=(xc_, yc_, zc1 - 0.045), bevel=0.02, segments=1))
    pots = []
    for sg in (-1, 1):
        hp_ = cr.vary(0.5, 0.15)
        pots.append(lathe("pot", [(0.0, 0.0), (0.125, 0.0), (0.125, 0.04), (0.1, 0.07), (0.085, hp_ - 0.1), (0.105, hp_ - 0.07), (0.105, hp_),
                                  (0.08, hp_), (0.075, 0.3), (0.0, 0.3)], segments=12, loc=(xc_ + sg * 0.22, yc_, zc1 - 0.01)))
    Mf = frame_m((xc_, yc_, zbed(yc_) + 0.035 + sagz(xc_, yc_)), (-1, 0, 0), (0, -cs, sn))
    LD.append(slab("apron", Mf, (cw_ + 0.34, cd_ / cs + 0.34, 0.012)))
    fx("smoke", "smoke", (xc_ - 0.22, yc_, zc1 + 0.6), size=0.35)

    # gutters and downpipes (lead)
    gprof = [(0.07 * cos(radians(a)), 0.07 * sin(radians(a))) for a in [180 + 180 * k / 6 for k in range(7)]]
    gprof += [(0.062 * cos(radians(a)), 0.062 * sin(radians(a))) for a in [360 - 180 * k / 6 for k in range(7)]]
    zg = zeave - 0.075
    for sg, xo in ((1, hw - 0.14), (-1, -hw + 0.55)):
        yg = (ys - oe - 0.05) if sg > 0 else (D + oe + 0.05)
        LD.append(sweep("gutter", [(-hw - ov + 0.03, yg, zg + 0.02), (hw + ov - 0.03, yg, zg - 0.01)], gprof, smooth_path=False))
        yw = (ys - 0.06) if sg > 0 else (D + 0.1)
        yo = yw - sg * 0.1
        path = [(xo, yg, zg - 0.05), (xo, yg, zg - 0.22), (xo, yg + sg * 0.1, zg - 0.36), (xo, yw - sg * 0.08, zg - 0.74), (xo, yw, zg - 0.86)]
        if sg > 0:
            path += [(xo, yw, 0.95), (xo, yo - 0.02, 0.78), (xo, yo - 0.02, 0.2), (xo, yo - 0.16, 0.08)]
        else:
            path += [(xo, yw, 0.24), (xo, yw + 0.14, 0.09)]
        LD.append(sweep("downpipe", path, 0.04, sides=8, smooth_path=False))
        LD.append(lathe("hopper", [(0.045, -0.12), (0.06, -0.1), (0.1, 0.0), (0.105, 0.03), (0.09, 0.03)], segments=4, loc=(xo, yg, zg - 0.1), rot=(0, 0, 45)))
        z = zg - 1.3
        while z > 1.2:
            IR.append(ring_band("clip", radius=0.043, width=0.035, thick=0.004, loc=(xo, yw, z), segments=10))
            z -= 1.5
        x = -hw - ov + 0.4
        while x < hw + ov - 0.2:                              # gutter brackets
            beam(IR, (x, yg + sg * 0.13, zg + 0.04), (x, yg - sg * 0.02, zg - 0.072), 0.02, 0.008, bev=0.0)
            x += 0.95

    # ------------------------------------------------------------------ terrace floor
    fl = box("flags", (W + 0.3, gd + 0.3, 0.03), loc=(0, -(gd + 0.3) / 2, 0.015))
    densify(fl, 0.55)
    assign(fl, flagm)
    kr_ = rng.fork("kerb")
    kerb = Wall(-hw - 0.25, ys - 0.4, 0, W + 0.5)
    kk = max(2, int(round((W + 0.5) / 1.05)))
    cuts = [(W + 0.5) * j / kk + (kr_.jitter(0, 0.1) if 0 < j < kk else 0) for j in range(kk + 1)]
    for j in range(kk):
        ST.append(box("kerb", (cuts[j + 1] - cuts[j] - 0.008, 0.2, 0.05 + kr_.jitter(0, 0.004)), loc=((cuts[j] + cuts[j + 1]) / 2 - hw - 0.25, ys - 0.33, 0.025), bevel=0.012, segments=1))
    for sx in (-1, 1):
        kk2 = max(2, int(round((gd + 0.25) / 1.0)))
        for j in range(kk2):
            l_ = (gd + 0.22) / kk2
            ST.append(box("kerb", (0.2, l_ - 0.008, 0.05 + kr_.jitter(0, 0.004)), loc=(sx * (hw + 0.2), ys - 0.22 + l_ * (j + 0.5), 0.025), bevel=0.012, segments=1))

    # ------------------------------------------------------------------ lanterns (inline hanging lantern)
    def lantern_kit():
        ir = [cylinder("rod", 0.005, 0.2, loc=(0, 0, -0.1), verts=5),
              torus("eye", 0.018, 0.004, loc=(0, 0, -0.205), rot=(90, 0, 0), major_seg=8, minor_seg=4),
              cone("cap", 0.13, 0.02, 0.085, loc=(0, 0, -0.262), rot=(0, 0, 45), verts=4),
              sphere("knob", 0.016, loc=(0, 0, -0.215), segments=6, rings=4),
              box("base", (0.135, 0.135, 0.012), loc=(0, 0, -0.566)),
              cone("fin", 0.03, 0.0, 0.05, loc=(0, 0, -0.597), rot=(180, 0, 0), verts=6),
              cylinder("burner", 0.012, 0.05, loc=(0, 0, -0.535), verts=6)]
        for sx in (-1, 1):
            for sy in (-1, 1):
                beam(ir, (sx * 0.09, sy * 0.09, -0.305), (sx * 0.064, sy * 0.064, -0.56), 0.011, 0.011, bev=0.0)
            ir.append(box("tf", (0.19, 0.012, 0.014), loc=(0, sx * 0.09, -0.305)))
            ir.append(box("tf", (0.012, 0.19, 0.014), loc=(sx * 0.09, 0, -0.305)))
        gl = [cylinder("lglass", 0.064 * 1.414 - 0.008, 0.25, loc=(0, 0, -0.433), rot=(0, 0, 45), verts=4, cap=False, radius2=0.09 * 1.414 - 0.008)]
        flm = [sphere("flame", 0.021, loc=(0, 0, -0.47), segments=8, rings=5, scale=(1, 1, 1.9))]
        return [(iron, ir), (lglass, gl), (flame, flm)]

    lm = []
    lan_pos = []
    yl = -gd * 0.52
    for i in range(n - 1):
        xx = (px[i] + px[i + 1]) / 2
        IR.append(cylinder("lrod", 0.005, 0.2, loc=(xx, yl, G - 0.13), verts=5))
        IR.append(box("lplate", (0.09, 0.09, 0.008), loc=(xx, yl, G - 0.034)))
        lm.append(trs((xx, yl, G - 0.23), (0, 0, rng.u(0, 40))))
        lan_pos.append((xx, yl, G - 0.23 - 0.47))
    bal_l = ((px[-2] + px[-1]) / 2, yp + 0.02, G + U - 0.26)
    lm.append(trs(bal_l, (0, 0, 20), 0.85))
    # bracket lantern on the corner of the bressummer, lighting the cup sign
    xs_, zs_ = px[0] + 0.02, GB1 - 0.04
    sgn_l = (xs_, ys - 0.44, zs_ - 0.03)
    beam(IR, (xs_, ys + 0.005, zs_), (xs_, ys - 0.5, zs_), 0.014, 0.028, bev=0.0)
    IR.append(sweep("sl_scroll", [(xs_, ys - 0.004, zs_ - 0.3), (xs_, ys - 0.1, zs_ - 0.26), (xs_, ys - 0.26, zs_ - 0.13), (xs_, ys - 0.4, zs_ - 0.016)],
                    [(-0.006, -0.011), (0.006, -0.011), (0.006, 0.011), (-0.006, 0.011)], smooth_path=True))
    IR.append(torus("sl_curl", 0.032, 0.007, loc=(xs_, ys - 0.5, zs_ - 0.036), rot=(0, 90, 0), major_seg=10, minor_seg=5))
    IR.append(box("sl_plate", (0.05, 0.008, 0.4), loc=(xs_, ys - 0.004, zs_ - 0.14)))
    lm.append(trs(sgn_l, (0, 0, 0), 0.8))
    stamp("lantern", lantern_kit(), lm)

    # ------------------------------------------------------------------ signs
    hs_ = use("hanging_sign", emblem="cup", shape="shield", text=str(p.sign_text)[:8], board_color=p.paint, reach=0.7, lamp="none", seed=seed + 9)
    hs_.place((-hw, yp, G - 0.12), rot=(0, 0, -90), sockets=False)
    if str(p.sign_text).strip():
        tsize = 0.2
        t = text("fascia_txt", str(p.sign_text), size=tsize, depth=0.014, loc=(0, 0, 0), font="serif")
        mn, mx = world_bbox([t])
        twid = mx.x - mn.x
        maxw = min(W - 2.5, sp * 1.6)
        if twid > maxw:
            sc = maxw / twid
            delete([t])
            t = text("fascia_txt", str(p.sign_text), size=tsize * sc, depth=0.014, loc=(0, 0, 0), font="serif")
            mn, mx = world_bbox([t])
            twid = mx.x - mn.x
        zc_ = (GB0 + GB1) / 2
        place(t, loc=(-(mn.x + mx.x) / 2, ys - 0.036, zc_ - (mn.z + mx.z) / 2))
        BR.append(t)
        FR.append(box("fascia_board", (twid + 0.5, 0.03, 0.27), loc=(0, ys - 0.02, zc_), bevel=0.006, segments=1))
        for sx in (-1, 1):
            BR.append(lathe("stud", [(0.0, 0.0), (0.022, 0.0), (0.018, 0.008), (0.0, 0.012)], segments=8, loc=(sx * (twid / 2 + 0.16), ys - 0.035, zc_), rot=(90, 0, 0)))
    neon_at = None
    if p.neon:
        neonm = mat("neon", p.neon_color, glow=3.2)
        H = 0.98
        xn = 0.0
        yn = ys - 0.075
        zn = G + 0.13
        NE = []

        def tube(pts, closed=False, smooth=True):
            NE.append(sweep("neon", [(xn + a * H, yn, zn + b * H) for (a, b) in pts], 0.011, closed=closed, sides=6, smooth_path=smooth, step=0.035 if smooth else None))

        def mir(pts):
            return [(-a, b) for (a, b) in pts]

        cp = [(0.0, 1.0), (0.045, 0.9), (0.098, 0.77), (0.105, 0.65), (0.072, 0.54), (0.03, 0.47)]
        tube(cp + list(reversed(mir(cp)))[:-1], closed=True)
        sp_ = [(0.058, 0.47), (0.1, 0.6), (0.17, 0.71), (0.26, 0.76), (0.34, 0.72), (0.375, 0.62), (0.34, 0.53), (0.275, 0.51), (0.24, 0.56)]
        tube(sp_)
        tube(mir(sp_))
        tube([(-0.17, 0.44), (0.17, 0.44), (0.17, 0.385), (-0.17, 0.385)], closed=True, smooth=False)
        lp = [(0.0, 0.36), (0.05, 0.3), (0.06, 0.2), (0.035, 0.09), (0.0, 0.0)]
        tube(lp + list(reversed(mir(lp)))[1:-1], closed=True)
        ls_ = [(0.075, 0.36), (0.12, 0.27), (0.19, 0.21), (0.26, 0.22), (0.29, 0.28)]
        tube(ls_)
        tube(mir(ls_))
        for o in NE:
            assign(o, neonm)
        # wrought-iron carrier frame clamped to the balustrade
        fw_, fh_ = 0.84 * H, 1.06 * H
        yfz = ys - 0.045
        for sx in (-1, 1):
            beam(IR, (xn + sx * fw_ / 2, yfz, zn - 0.03), (xn + sx * fw_ / 2, yfz, zn + fh_ - 0.03), 0.008, 0.03, bev=0.0)
            beam(IR, (xn + sx * 0.2 * H, yfz, zn - 0.03), (xn + sx * 0.2 * H, yfz, zn + fh_ - 0.03), 0.008, 0.02, bev=0.0)
        for zz in (zn - 0.03, zn + 0.41 * H, zn + 0.74 * H, zn + fh_ - 0.03):
            beam(IR, (xn - fw_ / 2, yfz, zz), (xn + fw_ / 2, yfz, zz), 0.008, 0.03, bev=0.0)
        beam(IR, (xn, yfz, zn - 0.03), (xn, yfz, zn + fh_ - 0.03), 0.008, 0.02, bev=0.0)
        for sx in (-1, 1):
            for zz in (zb0, zb1):
                beam(IR, (xn + sx * fw_ / 2, yfz, zz), (xn + sx * fw_ / 2, yp - 0.03, zz), 0.012, 0.012, bev=0.0)
        neon_at = (xn, yn - 0.3, zn + 0.55 * H)

    # ------------------------------------------------------------------ terrace furniture
    fr = rng.fork("furniture")

    def chair_kit():
        w_ = [lathe("seat", [(0.0, 0.435), (0.19, 0.435), (0.2, 0.445), (0.2, 0.46), (0.185, 0.468), (0.0, 0.468)], segments=12)]
        for sx in (-1, 1):
            beam(w_, (sx * 0.15, -0.15, 0.44), (sx * 0.18, -0.19, 0.0), 0.03, 0.03, bev=0.0)
            beam(w_, (sx * 0.16, 0.24, 0.9), (sx * 0.15, 0.16, 0.44), 0.03, 0.03, bev=0.0)
            beam(w_, (sx * 0.15, 0.16, 0.44), (sx * 0.17, 0.21, 0.0), 0.03, 0.03, bev=0.0)
            beam(w_, (sx * 0.17, -0.17, 0.2), (sx * 0.165, 0.185, 0.2), 0.02, 0.02, bev=0.0)
        beam(w_, (-0.175, 0.24, 0.86), (0.175, 0.24, 0.86), 0.024, 0.09, bev=0.006)
        beam(w_, (-0.16, 0.215, 0.68), (0.16, 0.215, 0.68), 0.02, 0.05, bev=0.004)
        beam(w_, (-0.17, -0.172, 0.24), (0.17, -0.172, 0.24), 0.02, 0.02, bev=0.0)
        return [(oak, w_)]

    def table_kit():
        w_ = [lathe("top", [(0.0, 0.69), (0.3, 0.69), (0.345, 0.695), (0.35, 0.71), (0.35, 0.725), (0.34, 0.735), (0.0, 0.735)], segments=20)]
        i_ = [lathe("ped", [(0.055, 0.05), (0.055, 0.08), (0.03, 0.11), (0.024, 0.4), (0.036, 0.44), (0.024, 0.48), (0.024, 0.66), (0.08, 0.69)], segments=10)]
        for k in range(3):
            a = radians(90 + 120 * k)
            beam(i_, (0.03 * cos(a), 0.03 * sin(a), 0.085), (0.29 * cos(a), 0.29 * sin(a), 0.02), 0.035, 0.03, bev=0.0)
        # candle lantern
        cx_, cy_, cz_ = 0.07, -0.05, 0.735
        i_ += [cylinder("cbase", 0.045, 0.012, loc=(cx_, cy_, cz_ + 0.006), verts=10),
               cone("ctop", 0.044, 0.012, 0.03, loc=(cx_, cy_, cz_ + 0.137), verts=8),
               torus("cring", 0.016, 0.003, loc=(cx_, cy_, cz_ + 0.162), rot=(90, 0, 0), major_seg=8, minor_seg=4)]
        g_ = [cylinder("cglass", 0.038, 0.11, loc=(cx_, cy_, cz_ + 0.067), verts=8, cap=False)]
        f_ = [sphere("cflame", 0.012, loc=(cx_, cy_, cz_ + 0.058), segments=6, rings=4, scale=(1, 1, 1.8))]
        return [(oak, w_), (iron, i_), (lglass, g_), (flame, f_)]

    yt = ys + 0.84
    cand = {-1: [], 1: []}
    xa = -hw + 1.7
    while xa < xd - 1.75:
        cand[-1].append(xa)
        xa += 2.0
    xa = hw - 1.7
    while xa > xd + 1.75:
        cand[1].append(xa)
        xa -= 2.0
    order = []
    for k in range(max(len(cand[-1]), len(cand[1]))):
        for s_ in (-1, 1):
            if k < len(cand[s_]):
                order.append(cand[s_][k])
    sets = order[:int(p.furniture)]
    tmx, chx = [], []
    for k, xt in enumerate(sets):
        xt += fr.jitter(0, 0.12)
        yy = yt + fr.jitter(0, 0.1)
        rz = fr.u(0, 360)
        tmx.append(trs((xt, yy, 0.03), (0, 0, rz)))
        a = radians(rz)
        if k < 3:
            fx(f"candle{k}", "flame", (xt + 0.07 * cos(a) + 0.05 * sin(a), yy + 0.07 * sin(a) - 0.05 * cos(a), 0.03 + 0.735 + 0.065), size=0.03)
        for sg in (-1, 1):
            d_ = fr.u(0.56, 0.7)
            chx.append(trs((xt + sg * d_, yy + fr.jitter(0, 0.12), 0.03), (0, 0, -sg * 90 + fr.jitter(0, 22))))
    if B >= 1.2:                                              # a table for two up on the balcony
        xb2 = (px[0] + px[1]) / 2 + fr.jitter(0, 0.3)
        yb2 = (ys + 0.24 + yb) / 2
        tmx.append(trs((xb2, yb2, G), (0, 0, fr.u(0, 360))))
        for sg in (-1, 1):
            chx.append(trs((xb2 + sg * fr.u(0.58, 0.68), yb2 + fr.jitter(0, 0.05), G), (0, 0, -sg * 90 + fr.jitter(0, 15))))
    stamp("table", table_kit(), tmx)
    stamp("chair", chair_kit(), chx)
    bar_rot = (fr.u(0, 360), fr.u(0, 360))

    if str(p.menu_text).strip():
        chalk = mat("chalkboard")
        xi = min(range(n), key=lambda i: abs(px[i] - (xd + sp / 2)))
        xm, ym = px[xi] - 0.62, yp + 0.12
        lean = 13.0
        for sg in (-1, 1):
            cy = ym + sg * 0.19
            rx = -sg * lean
            MB = trs((xm, cy, 0.03 + 0.5), (rx + 0, 0, 0))
            for sx in (-1, 1):
                o = slab("mleg", MB @ Matrix.Translation((sx * 0.27, 0, 0.0)), (0.045, 0.028, 1.02))
                uv_grain(o, axis='Z')
                T.append(o)
            for zz in (0.47, -0.2):
                o = slab("mrail", MB @ Matrix.Translation((0, 0, zz)), (0.5, 0.028, 0.05))
                uv_grain(o, axis='X')
                T.append(o)
            if sg < 0:
                cb = slab("mboard", MB @ Matrix.Translation((0, 0.004, 0.135)), (0.5, 0.012, 0.64))
                assign(cb, chalk)
                mt = text("menu_txt", str(p.menu_text)[:10], size=0.075, depth=0.002, loc=(0, 0, 0), font="script")
                mn, mx = world_bbox([mt])
                Mt = MB @ Matrix.Translation((-(mn.x + mx.x) / 2, -0.004, 0.33))
                transform_verts(mt, lambda co: Mt @ Matrix.Rotation(radians(90), 4, 'X') @ Vector((co.x, co.y, co.z)))
                place(mt, loc=(0, 0, 0), rot=(0, 0, 0))
                PL.append(mt)
                for q in range(4):                           # chalk lines of the menu
                    ln_ = slab("mline", MB @ Matrix.Translation((fr.jitter(0, 0.03), -0.003, 0.2 - q * 0.085)), (fr.u(0.26, 0.4), 0.002, 0.012))
                    PL.append(ln_)
        IR.append(cylinder("mhinge", 0.006, 0.5, loc=(xm, ym, 0.03 + 0.5 + 0.5), rot=(0, 90, 0), verts=6))

    # ------------------------------------------------------------------ lights (6 sockets)
    kL = float(p.light)
    wl = lights_win if lights_win else [(xd, 1.3)]
    for i, (xc, ww) in enumerate(wl[:2]):
        light(f"shop{i}", (xc, -1.05, 1.55), color="#ffc37a", intensity=30 * kL, range=7.5, flicker=0.04)
    if len(wl) < 2:
        light("door", (xd, -0.9, 1.9), color="#ffc37a", intensity=14 * kL, range=6.0, flicker=0.04)
    mid = min(range(len(lan_pos)), key=lambda i: abs(lan_pos[i][0] - xd * 0.5))
    light("lantern_mid", lan_pos[mid], color="#ffbf6a", intensity=32 * kL, range=10.0, flicker=0.14, shadow=True, shadow_near=0.3)
    light("balcony", (bal_l[0], bal_l[1], bal_l[2] - 0.47 * 0.85), color="#ffbf6a", intensity=20 * kL, range=7.0, flicker=0.14, shadow_near=0.3)
    light("sign_lantern", (sgn_l[0], sgn_l[1], sgn_l[2] - 0.47 * 0.8), color="#ffbf6a", intensity=10 * kL, range=7.0, flicker=0.14, shadow_near=0.3)
    if neon_at:
        light("neon", neon_at, color=p.neon_color, intensity=8 * kL, range=5.0, flicker=0.03)
    elif len(wl) >= 2 and len(lan_pos) > 1:
        far = max(range(len(lan_pos)), key=lambda i: abs(lan_pos[i][0] - lan_pos[mid][0]))
        light("lantern_side", lan_pos[far], color="#ffbf6a", intensity=24 * kL, range=8.0, flicker=0.14, shadow_near=0.3)

    # ------------------------------------------------------------------ join, materials, weathering
    def fin(lst, m, name):
        if not lst:
            return None
        o = join(lst, name)
        assign(o, m)
        return o

    tim = fin(T, oak, "timbers")
    brd = fin(BD, boards, "boards")
    rw = fin(RW, oak, "roof_carpentry")
    # the deck under the tiles is clay on top (oak below and at its ends): where a far level loses the laid tiles,
    # the roof stays a tiled roof instead of a black board
    dko = fin(BK, tilem, "roof_deck")
    if dko is not None:
        assign_faces(dko, oak, lambda c, n: n.z < 0.2 or abs(n.x) > 0.9)
    pla = fin(PL, plas, "plaster")
    gwo = fin(GW, gwm, "ground_walls")
    sto = fin(ST, stone, "stones")
    frm = fin(FR, paintm, "joinery")
    fin(GLIT, litm, "glass_lit")
    fin(GDARK, darkm, "glass_dark")
    iro = fin(IR, iron, "ironwork")
    ldo = fin(LD, lead, "leadwork")
    fin(BR, brass, "brasswork")
    bko = fin(BRK, brick, "chimney")
    pto = fin(pots, potm, "pots")

    for o_ in (rw, dko):
        if o_ is not None and p.sag > 0.001:
            densify(o_, 0.8)
            transform_verts(o_, lambda co: Vector((co.x, co.y, co.z + sagz(co.x, co.y))))

    nz = lambda s, sd, lo=0.35, hi=0.7: noise_mask(s, seed=seed + sd, lo=lo, hi=hi)
    ag = 0.4 + 0.8 * age
    if gwo:
        paint(gwo, grime=mask_mul(height_ramp(0.0, 1.5), nz(0.7, 1, 0.2, 0.7), ag),
              special=mask_mul(height_ramp(0.0, 1.8), facing((0, 1, 0)), nz(0.5, 2), 0.9 * ag))
    if sto:
        paint(sto, grime=mask_mul(height_ramp(0.0, 0.9), 0.8 * ag), special=mask_mul(facing((0, 0, 1)), nz(0.25, 3), 0.6 * ag),
              wear=mask_mul(height_ramp(0.05, 0.0), 0.5))
    if pla:
        paint(pla, grime=mask_mul(height_ramp(E + 0.2, E - 1.3), nz(0.45, 4, 0.3, 0.75), 0.75 * ag),
              special=mask_mul(facing((0, 1, 0)), nz(0.9, 5, 0.4, 0.75), 0.7 * ag), wear=mask_mul(nz(0.35, 6, 0.62, 0.8), 0.5 * age))
    if tim:
        paint(tim, grime=mask_mul(height_ramp(0.6, 1.6), 0.6 * ag), wear=mask_mul(nz(0.3, 7, 0.55, 0.8), 0.45 * ag),
              special=mask_mul(facing((0, 1, 0)), height_ramp(G, G + 1.2), nz(0.6, 8), 0.5 * age))
    if rw:
        paint(rw, grime=mask_mul(nz(0.5, 9, 0.3, 0.7), 0.5 * ag), special=mask_mul(facing((0, 1, 0)), nz(0.5, 10), 0.4 * age))
    if dko:
        paint(dko, grime=mask_mul(nz(1.1, 14, 0.25, 0.8), 0.45 * ag), special=0.3 * ag)
    if frm:
        paint(frm, wear=mask_mul(nz(0.12, 11, 0.55, 0.8), 0.7 * ag), special=mask_mul(facing((0, 0, 1)), 0.5 * ag), grime=mask_mul(nz(0.3, 12), 0.35 * ag))
    paint(tiles, special=mask_mul(nz(0.32, 13, 0.5 - 0.1 * age, 0.82), nz(2.2, 21, 0.25, 0.7), lambda co, nrm: 0.3 + 0.7 * (nrm[:, 1] > 0), 0.75 * ag),
          grime=mask_mul(nz(1.1, 14, 0.25, 0.8), 0.45 * ag))
    paint(ridge, special=mask_mul(nz(0.5, 15), 0.8 * ag), grime=0.3 * ag)
    if bko:
        paint(bko, grime=mask_mul(height_ramp(zc1, zc1 - 1.1), 0.9), special=mask_mul(facing((0, 1, 0)), 0.5 * ag))
    if pto:
        paint(pto, grime=0.75)
    if ldo:
        paint(ldo, grime=mask_mul(nz(0.3, 16), 0.5 * ag))
    if iro:
        paint(iro, special=mask_mul(nz(0.15, 17, 0.45, 0.75), 0.5 * ag))
    paint(fl, grime=mask_mul(lambda co, nrm: clamp_arr(1.0 - abs(co[:, 1] + gd * 0.5) / (gd * 0.55)) * 0.0 + 1.0, nz(0.8, 18, 0.35, 0.75), 0.5 * ag),
          wear=mask_mul(lambda co, nrm: wear_path(co, xd, gd), 0.7),
          special=mask_mul(lambda co, nrm: (co[:, 1] < -gd + 0.25) * 1.0, nz(0.4, 19), 0.7 * ag))
    paint(braces, wear=mask_mul(nz(0.3, 20, 0.55, 0.8), 0.4 * ag))

    # barrels last: they are the heaviest props, so the second / first one is dropped when the triangle budget is tight
    nbar = int(p.barrels)
    bpos = [((-hw + 0.14 + 0.66, yp + 0.08, 0.03), bar_rot[0]), ((hw - 0.14 - 0.68, yp + 0.1, 0.03), bar_rot[1])]
    bar = None
    for k in range(nbar):
        st = scene_tris()
        if bar is None:
            if st + 11500 > 159800:
                break
            bar = use("barrel", pose="stand", height=0.92, tap="none", seed=seed + 4)
        elif st + 11500 > 159800:
            break
        bar.place(bpos[k][0], rot=(0, 0, bpos[k][1]), sockets=False)
    socket("banner_l", "attach", (-hw + 0.14, ys, G + 1.6), role="banner")
    socket("banner_r", "attach", (hw - 0.14, ys, G + 1.6), role="banner")
    socket("planter_balcony", "attach", (0.0, ys, zb1), role="planter")
    print("DBG posts", n, "bay %.2f" % sp, "door x %.2f" % xd, "tiles", len(tmats), "tile %.2f x %.2f" % (tw, tg),
          "ridge z %.2f" % zbed(yr), "sets", len(sets))


def scene_tris():
    import bpy
    return sum(len(f.vertices) - 2 for o in bpy.context.scene.objects if o.type == 'MESH' for f in o.data.polygons)


def clamp_arr(a):
    import numpy as np
    return np.clip(a, 0.0, 1.0)


def wear_path(co, xd, gd):
    """Foot-worn flags: along the shopfront and through the bay in front of the door."""
    import numpy as np
    along = np.clip(1.0 - np.abs(co[:, 1] + 1.25) / 0.6, 0.0, 1.0)
    door = np.clip(1.0 - np.abs(co[:, 0] - xd) / 0.8, 0.0, 1.0)
    return np.maximum(along, door) * 0.8
