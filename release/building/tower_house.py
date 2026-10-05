"""Corner tower house: a stone ground storey (plinth blocks, quoins, string course) carrying jettied
half-timbered upper storeys (bressummer on joist ends and knee brackets, pegged posts, rails, braces
and lime-plaster panels) under a steep gabled roof with dormers and a chimney. On the front-left
corner stands a round or octagonal stone stair turret: battered plinth, string courses, slit and small
leaded windows climbing with the stair, a corbelled (or timber-framed) lookout stage a storey above
the eaves, and a bell-cast conical roof of individually laid tiles / slates / shingles (or seamed
copper) with a lead finial and a spinning weathervane. Closed and detailed on all four sides.
"""
from tk import *
from math import sin, cos, tan, radians, degrees, atan2, hypot, pi, sqrt, ceil
from mathutils import Matrix, Vector, Euler
import numpy as np

META = {"name": "Corner tower house", "class": "building", "mount": "ground", "focus": 0.5}
PARAMS = {
    "width": {"default": 8.6, "min": 7.0, "max": 11.0, "desc": "plan width along X at the ground-floor walls (m)"},
    "depth": {"default": 8.0, "min": 7.0, "max": 10.0, "desc": "plan depth, front wall y = 0 to back wall (m)"},
    "storeys": {"default": 2, "min": 2, "max": 3, "desc": "storeys below the eaves (stone ground floor + timber-framed floors)"},
    "ground_h": {"default": 3.2, "min": 2.9, "max": 3.6, "desc": "height of the stone ground storey (m)"},
    "storey_h": {"default": 2.9, "min": 2.7, "max": 3.2, "desc": "height of each timber-framed storey (m)"},
    "turret_radius": {"default": 1.65, "min": 1.4, "max": 2.2, "desc": "radius of the stair turret shaft (m; across-flats / 2 for the octagon)"},
    "turret_plan": {"default": "round", "choices": ["round", "octagon"], "desc": "plan of the corner turret"},
    "turret_rise": {"default": 3.5, "min": 2.4, "max": 4.2, "desc": "height of the turret wall head above the main eaves (m)"},
    "turret_top": {"default": "stone", "choices": ["stone", "timber"], "desc": "lookout stage: corbelled stone or jettied half-timbering"},
    "cone_pitch": {"default": 66.0, "min": 55.0, "max": 72.0, "desc": "pitch of the conical turret roof (deg)"},
    "roof": {"default": "tile", "choices": ["tile", "slate", "shingle"], "desc": "roof covering of the main roof and the turret cone"},
    "turret_roof": {"default": "same", "choices": ["same", "copper"], "desc": "turret cone covering: same as the main roof, or seamed copper"},
    "roof_pitch": {"default": 50.0, "min": 42.0, "max": 56.0, "desc": "pitch of the main roof (deg)"},
    "jetty": {"default": 0.3, "min": 0.0, "max": 0.45, "desc": "overhang of the timber storeys over the stone front (m)"},
    "stone": {"default": "ashlar", "choices": ["ashlar", "rubble"], "desc": "masonry of the ground storey and turret"},
    "plaster": {"default": "#e2d4b2", "desc": "limewash colour of the infill panels"},
    "timber": {"default": "#5a4130", "desc": "stain of the oak frame"},
    "trim": {"default": "#3b6b52", "desc": "paint of shutters, window frames, bargeboards and pipes"},
    "dormers": {"default": 1, "min": 0, "max": 3, "desc": "lit dormers on the front roof slope (as many as fit)"},
    "oriel": {"default": False, "desc": "oriel bay window instead of the middle first-floor front window"},
    "vane": {"default": "rooster", "choices": ["rooster", "banner", "none"], "desc": "weathervane on the turret finial"},
    "date": {"default": "1594", "desc": "year carved on the tablet below the lookout windows (empty = no tablet)"},
    "lit": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "fraction of windows with a lit room behind"},
    "lantern": {"default": True, "desc": "wall lantern beside the front door"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: moss, damp, soot, crooked timbers"},
}
VARIANTS = {
    "octagon_slate": {"width": 7.2, "depth": 7.4, "storeys": 3, "turret_plan": "octagon", "turret_radius": 1.6, "roof": "slate",
                      "stone": "rubble", "plaster": "#d9d3c4", "timber": "#3d2e24", "trim": "#2f3f5c", "turret_rise": 3.4,
                      "cone_pitch": 70.0, "vane": "banner", "dormers": 1, "seed": 7},
    "timber_top_shingle": {"width": 10.6, "depth": 8.4, "storeys": 2, "turret_top": "timber", "turret_radius": 2.05, "roof": "shingle",
                           "plaster": "#d6b98a", "timber": "#4a3526", "trim": "#7a2a22", "dormers": 3, "oriel": True,
                           "roof_pitch": 46.0, "cone_pitch": 60.0, "seed": 19},
    "copper_cap_tall": {"width": 8.0, "depth": 9.6, "storeys": 3, "turret_roof": "copper", "turret_radius": 1.9, "turret_rise": 4.0,
                        "plaster": "#e6ddc8", "trim": "#6a2f2a", "jetty": 0.42, "dormers": 2, "cone_pitch": 68.0,
                        "roof_pitch": 54.0, "lit": 0.8, "seed": 31},
    "octagon_timber_wide": {"width": 11.0, "depth": 7.2, "storeys": 2, "turret_plan": "octagon", "turret_top": "timber",
                            "turret_radius": 1.9, "roof": "tile", "stone": "rubble", "plaster": "#cfc2a0", "trim": "#35505c",
                            "jetty": 0.2, "dormers": 2, "vane": "banner", "date": "1611", "age": 0.9, "lit": 0.4, "seed": 44},
}

ROOSTER = [(-0.06, 0.0), (0.06, 0.0), (0.13, 0.05), (0.17, 0.13), (0.17, 0.2), (0.15, 0.27), (0.16, 0.32), (0.19, 0.31),
           (0.2, 0.28), (0.22, 0.31), (0.21, 0.35), (0.27, 0.365), (0.21, 0.39), (0.2, 0.43), (0.17, 0.41), (0.16, 0.45),
           (0.13, 0.42), (0.11, 0.45), (0.09, 0.4), (0.07, 0.32), (0.04, 0.25), (-0.04, 0.2), (-0.1, 0.22), (-0.14, 0.31),
           (-0.15, 0.41), (-0.21, 0.45), (-0.27, 0.42), (-0.22, 0.39), (-0.21, 0.34), (-0.27, 0.36), (-0.32, 0.31),
           (-0.26, 0.29), (-0.24, 0.24), (-0.31, 0.23), (-0.33, 0.17), (-0.26, 0.17), (-0.2, 0.11), (-0.13, 0.05)]
BANNER = [(-0.5, 0.05), (-0.42, 0.13), (-0.5, 0.21), (-0.42, 0.29), (-0.5, 0.37), (-0.05, 0.33), (0.0, 0.37), (0.05, 0.33),
          (0.3, 0.27), (0.3, 0.33), (0.5, 0.21), (0.3, 0.09), (0.3, 0.15), (0.05, 0.09), (0.0, 0.05), (-0.05, 0.09)]


class _Wall:
    """A vertical wall plane: origin, yaw (deg). u runs along the wall, `out` along the outward normal."""

    def __init__(self, ox, oy, phi, u0, u1):
        self.o, self.phi, self.u0, self.u1 = (ox, oy), phi, u0, u1
        a = radians(phi)
        self.t = (cos(a), sin(a))
        self.n = (sin(a), -cos(a))

    def pt(self, u, z, out=0.0):
        return (self.o[0] + self.t[0] * u + self.n[0] * out, self.o[1] + self.t[1] * u + self.n[1] * out, z)


def _pane(name, w, h, m, loc, yaw, u0=0.0, u1=1.0):
    """Glazing quad facing local -Y whose UVs span the whole window (room / curtains of window_lit fit the opening)."""
    o = mesh_object(name, [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2)], [(0, 1, 2, 3)])
    me = o.data
    uvl = me.uv_layers[0] if len(me.uv_layers) else me.uv_layers.new(name="UVMap")
    for li, uv in zip(me.polygons[0].loop_indices, ((u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0))):
        uvl.data[li].uv = uv
    uv_keep(o)
    place(o, loc, rot=(0, 0, yaw))
    assign(o, m)
    return o


def build(p, rng):
    W, D = float(p.width), float(p.depth)
    st = int(round(p.storeys))
    zg, sh = float(p.ground_h), float(p.storey_h)
    He = zg + (st - 1) * sh
    R = float(p.turret_radius)
    octa = p.turret_plan == "octagon"
    kf = 1.0 / cos(radians(22.5)) if octa else 1.0
    seg = 8 if octa else 40
    lrot = (0, 0, 22.5) if octa else None
    age = float(p.age)
    P = radians(p.roof_pitch)
    timber_top = p.turret_top == "timber"

    # turret axis: the shaft bulges at most 1.0 m outside the plan
    Rt = R + (0.18 if timber_top else 0.14)          # lookout stage radius
    eave_t = 0.22                                    # cone eaves beyond the lookout stage
    bulge = min(1.1 - (Rt - R) - eave_t, 0.6 * R)   # everything stays within 1.2 m of the plan
    c_in = R - bulge
    cx, cy = -W / 2 + c_in, c_in
    Ht = He + float(p.turret_rise)                   # turret wall head
    zc0, zc1 = He - 0.46, He - 0.06                  # corbel table / jetty of the lookout stage

    # the main roof's eaves corner must die inside the turret: limit jetty and eaves overhang
    vx = 0.32
    allow = sqrt(max(0.02, (R + 0.02) ** 2 - (c_in + vx) ** 2)) - c_in
    j = max(0.0, min(float(p.jetty), allow - 0.3))
    ov = clamp(allow - j - 0.12, 0.16, 0.42)

    # ---------------------------------------------------------------- materials
    wall_stone = mat("ashlar_limestone") if p.stone == "ashlar" else mat("rubble_fieldstone")
    dressed = mat("stone_smooth", "#b9ad98")
    plaster = mat("plaster_lime", p.plaster)
    oak = mat("wood_oak", p.timber)
    trim = mat("wood_painted", p.trim)
    lead = mat("lead")
    iron = mat("iron_wrought")
    pipe_iron = mat("iron_painted", p.trim)
    lit_a = mat("window_lit", "#ffc37a", glow=2.6, curtain=0.3)   # NB: window_lit without an explicit tint renders dark
    lit_b = mat("window_lit", "#ffb765", glow=2.2, curtain=0.45)
    lit_t = mat("window_lit", "#ffcb85", glow=2.8, curtain=0.0)
    dark = mat("window_dark")
    boards = mat("wood_planks", p.timber)
    lights = []                                       # (priority, name, loc, cd, range)

    def rad_at(a_deg, rad):
        if not octa:
            return rad
        return rad / cos(radians(((a_deg % 45.0) + 22.5) % 45.0 - 22.5))

    def inside_turret(x, y, rad):
        dx, dy = x - cx, y - cy
        return hypot(dx, dy) <= rad_at(degrees(atan2(dy, dx)), rad)

    def edge_x(yline, rad):            # largest x on the line y = yline that is inside the turret
        t = rad * kf + 0.02
        while t > 0 and not inside_turret(cx + t, yline, rad):
            t -= 0.01
        return cx + t

    def edge_y(xline, rad):
        t = rad * kf + 0.02
        while t > 0 and not inside_turret(xline, cy + t, rad):
            t -= 0.01
        return cy + t

    def L(name, prof, m, segments=None, smooth_angle=None):
        o = lathe(name, [(r * kf, z) for r, z in prof], segments=segments or seg, loc=(cx, cy, 0), rot=lrot,
                  smooth_angle=smooth_angle or (30.0 if octa else 50.0))
        assign(o, m)
        return o

    # ================================================================ MAIN BLOCK
    stone_objs, plaster_objs, timber_objs, dressed_objs = [], [], [], []

    zs = zg - 0.2                                      # top of the masonry
    g = box("ground_walls", (W, D, zs), loc=(0, D / 2, zs / 2))
    densify(g, 0.7)
    assign(g, wall_stone)
    stone_objs.append(g)
    up = box("upper_walls", (W, D + j, He - zs), loc=(0, (D - j) / 2, (He + zs) / 2))
    densify(up, 0.6)
    displace_noise(up, strength=0.006, scale=1.3, seed=p.seed + 3)
    assign(up, plaster)
    plaster_objs.append(up)

    def blk(name, size, loc, rot=None, r=None, bev=0.014):
        o = box(name, size, loc=loc, rot=rot, bevel=bev, segments=1)
        assign(o, dressed)
        o["tsq_smooth"] = 40.0
        dressed_objs.append(o)
        return o

    # ground-floor walls (y = 0 front); the turret hides the front-left corner
    gx0 = edge_x(0.0, R + 0.13)                        # where the front wall leaves the turret
    gy0 = edge_y(-W / 2, R + 0.13)
    GF = _Wall(-W / 2, 0.0, 0.0, gx0 + W / 2, W)
    GR = _Wall(W / 2, 0.0, 90.0, 0.0, D)
    GB = _Wall(W / 2, D, 180.0, 0.0, W)
    GL = _Wall(-W / 2, D, 270.0, 0.0, D - gy0)
    # upper walls (front jettied to y = -j)
    ux0 = edge_x(-j, R + 0.02)
    uy0 = edge_y(-W / 2, R + 0.02)
    UF = _Wall(-W / 2, -j, 0.0, ux0 + W / 2, W)
    UR = _Wall(W / 2, -j, 90.0, 0.0, D + j)
    UB = _Wall(W / 2, D, 180.0, 0.0, W)
    UL = _Wall(-W / 2, D, 270.0, 0.0, D - uy0)

    # ---- front door, lantern and ground-floor front windows
    rg = rng.fork("ground")
    fspan0, fspan1 = gx0 + 0.35, W / 2 - 0.55
    door_w = 1.3
    xd = fspan0 + door_w / 2 + 0.1
    door = use("door_plank", width=door_w, height=2.5, head="round", frame="stone", frame_tone="#b9ad98", number=str(1 + p.seed % 23),
               intensity=2.0, seed=p.seed + 5)
    door.place((xd, 0, 0), sockets={"attach", "fx"})
    lights.append((3, "door_grille", (xd, -0.35, 1.6), 2.0, 3.5))
    door_span = (xd - door_w / 2 - 0.25, xd + door_w / 2 + 0.25)
    rem0 = xd + door_w / 2 + 0.95
    gwin = []
    if fspan1 - rem0 >= 1.2:
        n = max(1, int((fspan1 - rem0 + 0.9) / 2.5))
        for i in range(n):
            gwin.append(rem0 + (fspan1 - rem0) * (i + 0.5) / n)
    gw_w = clamp((fspan1 - rem0) / max(1, len(gwin)) - 1.2, 0.9, 1.5)
    gw_h, gw_z = 1.5, 1.05
    gcomp = {}
    for i, x in enumerate(gwin):
        is_lit = rg.chance(p.lit) or i == 0
        if is_lit not in gcomp:
            gcomp[is_lit] = use("window_casement", width=gw_w, height=gw_h, head="arched", surround="stone", glazing="square",
                                lights=2 if gw_w < 1.3 else 3, stone="#b9ad98", panes_x=3, panes_y=5, shutters=False, lit=is_lit, frame_paint=p.trim, age=age,
                                seed=p.seed + 21 + int(is_lit))
        gcomp[is_lit].place((x, 0, gw_z), sockets={"attach"})
    if p.lantern:
        lx = xd + door_w / 2 + 0.42
        lan = use("wall_lantern", reach=0.5, style="scroll", paint=p.trim, intensity=22.0, rust=0.3 + 0.4 * age, seed=p.seed + 9)
        lan.place((lx, 0, 2.55), sockets={"attach"})
        lights.append((0, "door_lantern", (lx, -0.5, 2.72), 22.0, 9.0))
    socket("sign", "attach", (min(fspan1, xd + door_w / 2 + 1.6), 0, zs - 0.75), role="sign")

    # ---- own plain window for side and back walls (cheap): painted frame, mullion, transom, bars
    def plain_window(wall, u, z, w, h, is_lit, surround, r):
        def bx(name, su, sz, du, dz, out, thick, m, grain=None, bev=0.006):
            o = box(name, (su, thick, sz), bevel=bev, segments=1)
            if grain:
                uv_grain(o, grain)
            place(o, wall.pt(u + du, z + dz, out - thick / 2), rot=(0, 0, wall.phi))
            assign(o, m)
            return o
        m = (lit_a if r.chance(0.6) else lit_b) if is_lit else dark
        _pane("pw_pane", w - 0.04, h - 0.04, m, wall.pt(u, z + h / 2, 0.012), wall.phi)
        fw = 0.07
        for sx in (-1, 1):
            bx("pw_jamb", fw, h, sx * (w / 2 - fw / 2), h / 2, 0.06, 0.1, trim, "Z")
        bx("pw_head", w, fw, 0, h - fw / 2, 0.06, 0.1, trim, "X")
        bx("pw_bottom", w, fw, 0, fw / 2, 0.06, 0.1, trim, "X")
        bx("pw_mullion", 0.055, h, 0, h / 2, 0.055, 0.09, trim, "Z")
        bx("pw_transom", w, 0.045, 0, h * 0.68, 0.05, 0.08, trim, "X")
        for k in (0.24, 0.46):
            bx("pw_bar", w - fw, 0.02, 0, h * k, 0.036, 0.03, trim, "X", bev=0.0)
        for sx in (-1, 1):
            bx("pw_vbar", 0.02, h * 0.66, sx * w / 4, h * 0.34, 0.036, 0.03, trim, "Z", bev=0.0)
        if surround == "timber":
            timber_objs.append(bx("pw_lintel", w + 0.34, 0.15, 0, h + 0.075, 0.046, 0.12, oak, "X", bev=0.008))
            timber_objs.append(bx("pw_sill", w + 0.26, 0.085, 0, -0.042, 0.1, 0.16, oak, "X", bev=0.008))
        else:
            dressed_objs.append(bx("pw_lintel", w + 0.44, 0.26, 0, h + 0.13, 0.05, 0.14, dressed, bev=0.014))
            dressed_objs.append(bx("pw_sill", w + 0.34, 0.12, 0, -0.06, 0.11, 0.2, dressed, bev=0.014))
            nb = 3
            for sx in (-1, 1):
                for k in range(nb):
                    bw = 0.17 if (k + (sx > 0)) % 2 else 0.28
                    dressed_objs.append(bx("pw_jambstone", bw, h / nb - 0.012, sx * (w / 2 + bw / 2), h * (k + 0.5) / nb, 0.04,
                                           0.12, dressed, bev=0.012))

    def spill(wall, u, z, cd, name, prio=2, rangem=6.5):
        lights.append((prio, name, wall.pt(u, z, 0.5), cd, rangem))

    # ---- ground-floor side / back openings
    rw = rng.fork("gside")
    back_door_u = W * 0.3
    for wall, name in ((GR, "r"), (GB, "b"), (GL, "l")):
        u0, u1 = wall.u0 + 0.9, wall.u1 - 0.9
        n = max(1, int((u1 - u0) / 3.0))
        any_lit = False
        for i in range(n):
            u = u0 + (u1 - u0) * (i + 0.5) / n
            if wall is GB and abs(u - back_door_u) < 1.45:
                u = back_door_u + (1.6 if u >= back_door_u else -1.6)
                if not (u0 - 0.3 < u < u1 + 0.3):
                    continue
            is_lit = rw.chance(p.lit * 0.8) or (i == n - 1 and not any_lit and wall is not GL)
            any_lit = any_lit or is_lit
            plain_window(wall, u, 1.15, 0.9, 1.3, is_lit, "stone", rw)
            if is_lit and wall is GB and not any(l[1] == "back_window" for l in lights):
                spill(wall, u, 1.8, 7.0, "back_window", prio=2.5)

    # ---- back door: stone jambs and lintel, ledged plank leaf with strap hinges and a ring
    bd_w, bd_h = 0.95, 2.05

    def bbox_on(wall, name, su, sz, uc, zc, out, thick, m, grain=None, bev=0.008):
        o = box(name, (su, thick, sz), bevel=bev, segments=1)
        if grain:
            uv_grain(o, grain)
        place(o, wall.pt(uc, zc, out - thick / 2), rot=(0, 0, wall.phi))
        assign(o, m)
        return o
    nbp = 5
    door_wood = mat("wood_weathered", "#a39680")
    rb = rng.fork("backdoor")
    for i in range(nbp):
        pw = bd_w / nbp
        o = bbox_on(GB, "bd_board", pw - 0.006, bd_h + 0.12, back_door_u - bd_w / 2 + pw * (i + 0.5), bd_h / 2 + 0.1,
                    0.02 + rb.u(0, 0.004), 0.05, door_wood, "Z")
        timber_objs.append(o)
    for zz in (0.55, 1.75):
        bbox_on(GB, "bd_strap", bd_w - 0.1, 0.05, back_door_u + 0.02, zz, 0.03, 0.012, iron, bev=0.002)
    ring = torus("bd_ring", 0.055, 0.008, loc=GB.pt(back_door_u - bd_w / 2 + 0.14, 1.08, 0.04), rot=(90, 0, 0), major_seg=16, minor_seg=6)
    assign(ring, iron)
    for sx in (-1, 1):
        for k in range(4):
            bw = 0.2 if (k + (sx > 0)) % 2 else 0.3
            dressed_objs.append(bbox_on(GB, "bd_jamb", bw, (bd_h + 0.2) / 4 - 0.012, back_door_u + sx * (bd_w / 2 + bw / 2),
                                        (bd_h + 0.2) * (k + 0.5) / 4, 0.05, 0.14, dressed, bev=0.012))
    dressed_objs.append(bbox_on(GB, "bd_lintel", bd_w + 0.7, 0.3, back_door_u, bd_h + 0.35, 0.06, 0.16, dressed, bev=0.016))
    dressed_objs.append(bbox_on(GB, "bd_step", bd_w + 0.5, 0.16, back_door_u, 0.08, 0.34, 0.5, dressed, bev=0.02))

    # ---- plinth blocks, string course, quoins
    rp = rng.fork("plinth")
    skips = {id(GF): [door_span], id(GB): [(back_door_u - bd_w / 2 - 0.3, back_door_u + bd_w / 2 + 0.3)]}
    for wall in (GF, GR, GB, GL):
        u = wall.u0 - (0.0 if wall in (GF,) else 0.07)
        uend = wall.u1 + (0.07 if wall is not GL else 0.0)
        while u < uend - 0.05:
            ln = min(rp.u(0.75, 1.35), uend - u)
            if uend - u - ln < 0.4:
                ln = uend - u
            mid = u + ln / 2
            cut = False
            for (s0, s1) in skips.get(id(wall), []):
                if u < s1 and u + ln > s0:
                    if mid < (s0 + s1) / 2:
                        ln = s0 - u
                    else:
                        u, ln = s1, u + ln - s1
                    cut = True
            if ln > 0.12:
                hh = 0.5 + rp.u(-0.012, 0.012)
                o = blk("plinth", (ln - 0.012, 0.2, hh), wall.pt(u + ln / 2, hh / 2, 0.075 - 0.1 + rp.u(-0.006, 0.006)),
                        rot=(0, 0, wall.phi), bev=0.022)
            u += max(ln, 0.12) if not cut else max(ln, 0.0) + (0.0 if ln > 0 else 0.3)
        # string course under the timber storey (the jettied front has joists instead)
        if wall is not GF or j < 0.08:
            a0, a1 = wall.u0 - (0.05 if wall is not GF else 0), wall.u1 + (0.05 if wall is not GL else 0)
            n = max(1, int((a1 - a0) / 1.6))
            for i in range(n):
                blk("string", ((a1 - a0) / n - 0.008, 0.16, 0.15), wall.pt(a0 + (a1 - a0) * (i + 0.5) / n, zs - 0.075, 0.055 - 0.08),
                    rot=(0, 0, wall.phi), bev=0.02)
    rq = rng.fork("quoins")
    hq = 0.34
    for (X0, Y0, sx, sy) in ((W / 2, 0.0, 1, -1), (W / 2, D, 1, 1), (-W / 2, D, -1, 1)):
        z = 0.5
        k = 0
        top = zs - (0.15 if (sy > 0 or j < 0.08) else 0.0)
        while z < top - 0.1:
            h = min(hq, top - z)
            la, lb = (0.56, 0.3) if k % 2 == 0 else (0.3, 0.56)
            la += rq.u(-0.03, 0.03)
            lb += rq.u(-0.03, 0.03)
            pr = 0.035
            blk("quoin", (la + pr, lb + pr, h - 0.012), (X0 + sx * (pr - (la + pr) / 2), Y0 + sy * (pr - (lb + pr) / 2), z + h / 2), bev=0.016)
            z += h
            k += 1

    # ---- timber frame of the upper storeys
    def timber(wall, u0, z0, u1, z1, w=0.18, proud=0.035, thick=0.12, r=None, name="timber"):
        ln = hypot(u1 - u0, z1 - z0)
        if ln < 0.08:
            return None
        ang = degrees(atan2(z1 - z0, u1 - u0))
        o = box(name, (ln, thick, w), bevel=0.007, segments=1)
        uv_grain(o, "X")
        place(o, wall.pt((u0 + u1) / 2, (z0 + z1) / 2, proud - thick / 2), rot=(0, -ang, wall.phi))
        assign(o, oak)
        timber_objs.append(o)
        return o

    peg = bolt("peg", 0.014, 0.012, kind="dome")
    assign(peg, oak)
    pegs = []

    def peg_at(wall, u, z, out=0.04):
        pegs.append(Matrix.Translation(Vector(wall.pt(u, z, out))) @ Euler((radians(90), 0, radians(wall.phi))).to_matrix().to_4x4())

    def frame(wall, z0, z1, wins, r):
        """wins: [(u centre, half clear width incl. shutters, window top z)]"""
        u0, u1 = wall.u0, wall.u1
        lean = 0.012 * age
        rail_top = z0 + 0.95
        rail = rail_top - 0.08
        forced = [u0 + 0.11, u1 - 0.11]
        spans = []
        for (uc, hw, wt) in wins:
            a, b = max(u0 + 0.4, uc - hw - 0.1), min(u1 - 0.4, uc + hw + 0.1)
            forced += [a, b]
            spans.append((a, b, uc, wt))
        forced.sort()
        posts = []
        for a, b in zip(forced[:-1], forced[1:]):
            posts.append(a)
            if any(abs(a - s[0]) < 1e-6 and abs(b - s[1]) < 1e-6 for s in spans):
                continue
            gap = b - a
            if gap > 1.75:
                n = int(ceil(gap / 1.35))
                posts += [a + gap * i / n for i in range(1, n)]
        posts.append(forced[-1])
        clean = []
        for q in posts:
            if clean and q - clean[-1] < 0.3:
                continue
            clean.append(q)
        posts = clean
        zlo, zhi = z0 + 0.04, z1 - 0.19
        for i, q in enumerate(posts):
            end = i in (0, len(posts) - 1)
            dq = 0.0 if end else r.u(-lean, lean)
            timber(wall, q - dq, zlo, q + dq, zhi, w=(0.23 if end else r.u(0.17, 0.2)), proud=0.04, name="post")
            peg_at(wall, q + r.u(-0.03, 0.03), zlo + 0.07)
            peg_at(wall, q + r.u(-0.03, 0.03), zhi - 0.05)
            if not end:
                peg_at(wall, q, rail)
        for a, b in zip(posts[:-1], posts[1:]):
            bw = b - a
            mid = (a + b) / 2
            win = next((s for s in spans if s[0] - 0.05 < mid < s[1] + 0.05), None)
            ia, ib = a + 0.085, b - 0.085
            timber(wall, ia - 0.02, rail + r.u(-lean, lean), ib + 0.02, rail + r.u(-lean, lean), w=0.16, proud=0.032, name="rail")
            lo0, lo1 = z0 + 0.05, rail - 0.075
            hi0, hi1 = rail + 0.075, z1 - 0.2
            if win:
                if bw > 1.35:
                    timber(wall, ia, lo0, ib, lo1, w=0.13, proud=0.027)
                    timber(wall, ia, lo1, ib, lo0, w=0.13, proud=0.023)
                else:
                    for f in (0.33, 0.67):
                        timber(wall, a + bw * f, lo0, a + bw * f + r.u(-lean, lean), lo1, w=0.12, proud=0.027)
                wt = win[3]
                if hi1 - wt > 0.32:
                    for f in (-0.28, 0.28):
                        timber(wall, win[2] + f, wt + 0.14, win[2] + f, hi1 + 0.02, w=0.12, proud=0.027)
                continue
            if bw > 0.7:
                kind = r.weighted([("diag", 0.45), ("cross", 0.2), ("stud", 0.2), ("none", 0.15)])
                flip = r.chance(0.5) if 0.2 < (mid - u0) / max(u1 - u0, 1e-3) < 0.8 else (mid - u0) > (u1 - u0) / 2
                if kind == "diag":
                    timber(wall, ia, lo1 if flip else lo0, ib, lo0 if flip else lo1, w=0.14, proud=0.027)
                elif kind == "cross":
                    timber(wall, ia, lo0, ib, lo1, w=0.13, proud=0.027)
                    timber(wall, ia, lo1, ib, lo0, w=0.13, proud=0.023)
                elif kind == "stud":
                    timber(wall, mid, lo0, mid + r.u(-lean, lean), lo1, w=0.13, proud=0.027)
                kind = r.weighted([("brace", 0.5), ("stud", 0.2), ("none", 0.3)])
                if kind == "brace":
                    # curved-looking knee brace: two straight pieces from the post up to the plate
                    zk = hi0 + (hi1 - hi0) * 0.45
                    if flip:
                        timber(wall, ia, zk, ib, hi1, w=0.14, proud=0.027)
                    else:
                        timber(wall, ia, hi1, ib, zk, w=0.14, proud=0.027)
                elif kind == "stud":
                    timber(wall, mid, hi0, mid + r.u(-lean, lean), hi1, w=0.13, proud=0.027)

    # horizontal beams: bressummer / sill beams at every floor, wall plate under the eaves
    def beams(wall, front=False):
        a, b = wall.u0 - (0.0 if wall in (UF,) else 0.045), wall.u1 + (0.045 if wall is not UL else 0.0)
        for k in range(st - 1):
            zk = zg + k * sh
            if k == 0:
                timber(wall, a, zk - 0.08, b, zk - 0.08, w=0.26, proud=0.05, thick=0.16, name="bressummer")
            else:
                timber(wall, a, zk - 0.02, b, zk - 0.02, w=0.22, proud=0.046, thick=0.14, name="girt")
        timber(wall, a, He - 0.1, b, He - 0.1, w=0.2, proud=0.046, thick=0.14, name="plate")

    # windows of the timber storeys
    wf_w, wf_h = 1.05, 1.38
    ws_w, ws_h = 1.05, 1.3
    fcomp = {}

    def front_window(shut, is_lit):
        key = (shut, is_lit)
        if key not in fcomp:
            fcomp[key] = use("window_casement", width=wf_w, height=wf_h, lights=2, surround="timber", glazing="diamond",
                             shutters=shut, shutter_paint=p.trim, frame_paint="#d6cdb6", timber=p.timber, lit=is_lit,
                             age=age, open=0.0, seed=p.seed + 31 + 2 * int(shut) + int(is_lit))
        return fcomp[key]

    rwf = rng.fork("front_windows")
    for wall in (UF, UR, UB, UL):
        beams(wall)
    fspan_u0, fspan_u1 = UF.u0 + 0.3, UF.u1 - 0.3
    nfw = max(1, int((fspan_u1 - fspan_u0) / 2.45))
    front_lit_done = False
    for s in range(1, st):
        z0 = zg + (s - 1) * sh
        z1 = z0 + sh
        sill = z0 + 0.95
        wins = []
        shut = s == 1
        hw_clear = wf_w / 2 + (wf_w / 2 + 0.08 if shut else 0.06)
        for i in range(nfw):
            u = fspan_u0 + (fspan_u1 - fspan_u0) * (i + 0.5) / nfw
            is_lit = rwf.chance(p.lit) or (s == 1 and i == nfw - 1 and not front_lit_done)
            if p.oriel and s == 1 and i == nfw // 2 and nfw >= 1 and (fspan_u1 - fspan_u0) > 2.6:
                ori = use("window_oriel", width=1.7, height=1.5, depth=0.45, plan="canted", roof="lead", roof_rise=0.5,
                          lit=True, intensity=12.0, paint=p.trim, seed=p.seed + 41)
                ori.place(UF.pt(u, sill - 0.1), rot=(0, 0, UF.phi), sockets={"attach"})
                lights.append((1, "oriel", UF.pt(u, sill + 0.6, 0.95), 12.0, 7.0))
                wins.append((u, 0.9, sill + 1.9))
                front_lit_done = True
                continue
            shut_i = shut and (nfw < 3 or i % 2 == 0)
            front_window(shut_i, is_lit).place(UF.pt(u, sill), rot=(0, 0, UF.phi), sockets={"attach"})
            if is_lit and not front_lit_done:
                spill(UF, u, sill + 0.7, 9.0, "front_window", prio=1)
                front_lit_done = True
            wins.append((u, wf_w / 2 + (wf_w / 2 + 0.08 if shut_i else 0.06), sill + wf_h))
        frame(UF, z0, z1, wins, rng.fork(f"frame_f{s}"))
        for wall, nm in ((UR, "r"), (UB, "b"), (UL, "l")):
            r = rng.fork(f"win_{nm}{s}")
            a, b = wall.u0 + 0.8, wall.u1 - 0.8
            n = max(1, int((b - a) / 3.1))
            wins = []
            for i in range(n):
                u = a + (b - a) * (i + 0.5) / n + r.u(-0.15, 0.15)
                is_lit = r.chance(p.lit * 0.85) or (s == 1 and i == 0 and wall is not UL)
                plain_window(wall, u, sill, ws_w, ws_h, is_lit, "timber", r)
                wins.append((u, ws_w / 2 + 0.2, sill + ws_h + 0.15))
            frame(wall, z0, z1, wins, rng.fork(f"frame_{nm}{s}"))
    if pegs:
        o = instances_to_mesh("pegs", peg, pegs, keep_base=False)
        timber_objs.append(o)

    # ---- jetty: joist ends, soffit boards and knee brackets on the stone front
    if j >= 0.08:
        a, b = UF.u0 - 0.3, UF.u1
        x0, x1 = -W / 2 + a, -W / 2 + b
        sof = box("jetty_soffit", (x1 - x0, j, 0.024), loc=((x0 + x1) / 2, -j / 2, zs - 0.012))
        uv_grain(sof, "X")
        assign(sof, boards)
        nj = max(2, int((x1 - x0 - 0.3) / 0.52))
        rj = rng.fork("joists")
        for i in range(nj + 1):
            x = x0 + 0.42 + (x1 - x0 - 0.56) * i / nj
            if x < gx0 + 0.12:
                continue
            o = plank("joist", length=j + 0.14, width=0.14 + rj.u(-0.01, 0.01), thick=0.15, loc=(x, -j / 2 + 0.035, zs - 0.024 - 0.075),
                      bevel=0.008, axis="Y", end_mat="wood_endgrain")
            assign(o, oak, slot=0)
            timber_objs.append(o)
        # knee brackets with a concave soffit, under the main posts
        nb = max(2, int((x1 - gx0) / 1.9))
        hb = 0.62
        prof = [(0.0, 0.0), (0.0, -hb), (0.07, -hb), (0.09, -hb + 0.06)]
        for i in range(1, 8):
            t = i / 8.0
            prof.append((0.09 + (j - 0.07) * (1 - cos(t * pi / 2)), -hb + 0.06 + (hb - 0.24) * sin(t * pi / 2)))
        prof += [(j + 0.02, -0.16), (j + 0.02, 0.0)]
        for i in range(nb + 1):
            x = gx0 + 0.3 + (x1 - 0.2 - gx0 - 0.3) * i / nb
            if abs(x - xd) < door_w / 2 + 0.1:
                x = xd + (door_w / 2 + 0.22) * (1 if x >= xd else -1)
            o = extrude_profile("knee_bracket", prof, 0.12, bevel=0.008)
            zt = zs - 0.17
            transform_verts(o, lambda co, x=x, zt=zt: Vector((x + 0.06 - co.z, -co.x, zt + co.y)))
            uv_grain(o, "Z")
            assign(o, oak)
            timber_objs.append(o)

    # ================================================================ MAIN ROOF, DORMERS, CHIMNEY
    cover = {"tile": "roof_tile_clay", "slate": "roof_slate", "shingle": "wood_shingle"}[p.roof]
    roof = use("roof_gable", width=W, depth=D + j, pitch=p.roof_pitch, type="gable", ridge="x", overhang=ov, covering=cover,
               sag=0.04 + 0.1 * age, gable="bargeboard", gable_wall="timber", attic_window="lit", rooflight="none",
               eaves="rafters", ridge_tile="half_round" if p.roof == "tile" else "lead_roll", gutter="zinc",
               wall_color=p.plaster, timber_color=p.timber, trim_color=p.trim, moss=0.2 + 0.5 * age, seed=p.seed + 11)
    ry = (D - j) / 2
    roof.place((0, ry, He), sockets=False)
    tanP = tan(P)
    zoff = 0.26                                   # roof surface above the wall plate line at the wall face

    def roof_z(y):                               # tile surface height on the front (y < ry) / back slope
        return He + zoff + ((y + j) if y < ry else (D - y)) * tanP

    # dormers on the front slope, clear of the turret
    nd = int(p.dormers)
    dx0, dx1 = cx + Rt + 1.25, W / 2 - 1.45
    if nd > 0 and dx1 - dx0 > -0.2:
        nd = min(nd, 1 + int(max(0.0, dx1 - dx0) / 2.3))
        dcover = {"tile": "tile", "slate": "slate", "shingle": "lead"}[p.roof]
        dy = -j + 1.0
        for i in range(nd):
            x = (dx0 + dx1) / 2 if nd == 1 else dx0 + (dx1 - dx0) * i / (nd - 1)
            dor = use("dormer", width=1.25, height=1.75, pitch=p.roof_pitch, roof="gable", cover=dcover, cladding="plaster",
                      lit=(i == 0 or rng.chance(p.lit)), intensity=0.0, flower_box=False, timber=p.timber, paint=p.trim,
                      plaster=p.plaster, age=age, seed=p.seed + 51 + (i % 2))
            dor.place((x, dy, roof_z(dy)), sockets={"attach"})
            if i == 0:
                lights.append((2, "dormer", (x, dy - 0.55, roof_z(dy) + 0.8), 5.0, 6.0))
    if int(p.dormers) > 0:
        bx_ = clamp(-W * 0.12, -W / 2 + 2.0, W / 2 - 3.6)
        by_ = D - 1.0
        bd_lit = rng.fork("back_dormer").chance(p.lit * 0.6)
        bdor = use("dormer", width=1.15, height=1.6, pitch=p.roof_pitch, roof="hip" if p.roof != "shingle" else "shed",
                   cover={"tile": "tile", "slate": "slate", "shingle": "lead"}[p.roof], cladding="plaster", lit=bd_lit, intensity=0.0,
                   flower_box=False, timber=p.timber, paint=p.trim, plaster=p.plaster, age=age, seed=p.seed + 57)
        bdor.place((bx_, by_, roof_z(by_)), rot=(0, 0, 180), sockets={"attach"})
    # chimney on the back slope near the right gable
    ch_dr = 0.75
    ch_y = ry + ch_dr
    ch_sink = 0.5
    ch_h = ch_sink + ch_dr * tanP + 1.0
    chim = use("chimney", width=0.98, depth=0.6, height=ch_h, material="brick_red" if p.stone == "ashlar" else "stone",
               pots=2, sink=ch_sink, pitch=p.roof_pitch, soot=0.4 + 0.5 * age, moss=0.2 + 0.6 * age, seed=p.seed + 61)
    chim.place((W / 2 - 1.5, ch_y, roof_z(ch_y) - ch_sink), rot=(0, 0, 180), sockets={"fx"})

    # ---- downpipes: cast-iron one on the right flank, a plain one at the back
    dp = use("drainpipe", height=He - 0.35, section="round", offset=0.0, hopper="box", inlet="none", paint=p.trim,
             ageing=age, seed=p.seed + 71)
    dp.place(GR.pt(0.55, 0.0), rot=(0, 0, GR.phi), sockets=False)
    bu = W - 0.45
    pz0, pz1 = 0.22, He - 0.25
    pp = cylinder("back_pipe", 0.048, pz1 - pz0, loc=GB.pt(bu, (pz0 + pz1) / 2, 0.1), verts=10)
    hop = box("back_hopper", (0.26, 0.2, 0.24), loc=GB.pt(bu, pz1 + 0.06, 0.11), bevel=0.012, segments=1)
    shoe = cylinder("back_shoe", 0.048, 0.26, loc=GB.pt(bu, pz0 - 0.04, 0.17), rot=(55, 0, 0), verts=10)
    for o in (pp, hop, shoe):
        assign(o, pipe_iron)
        paint(o, special=mask_mul(noise_mask(0.3, seed=p.seed + 5), 0.3 + 0.5 * age), grime=height_ramp(0.0, 0.8))
    z = 0.9
    while z < pz1 - 0.3:
        e = ring_band("back_pipe_ear", 0.052, 0.045, 0.006, loc=GB.pt(bu, z, 0.1), segments=12)
        f = box("back_pipe_lug", (0.16, 0.1, 0.03), loc=GB.pt(bu, z, 0.05), bevel=0.003, segments=1)
        assign(e, iron)
        assign(f, iron)
        z += 1.7

    # ================================================================ TURRET
    rt_ = rng.fork("turret")
    wall_mat = wall_stone
    prof = [(0.0, 0.0), (R + 0.13, 0.0), (R + 0.13, 0.6), (R + 0.045, 0.76), (R + 0.03, 0.8), (R, zg), (R, zc0 + 0.02), (0.0, zc0 + 0.02)]
    shaft = L("turret_shaft", prof, wall_mat)
    stone_objs.append(shaft)
    cut_shaft, cut_stage = [], []
    stage_frames = []

    def t_at(a_deg, rr, zz):
        a = radians(a_deg)
        return (cx + cos(a) * rr, cy + sin(a) * rr, zz)

    def opening(a_deg, z, w, h, kind, state, rad, cutters, fmat, f=0.12, cames=True, sill=True):
        yaw = a_deg + 90.0
        rw_ = rad_at(a_deg, rad)
        cpts = [(x, y - 0.05) for x, y in arch_pts(w + 0.1, h + 0.1, kind=kind, n=10)]
        cutters.append(extrude_profile("cut", cpts, 0.8, loc=t_at(a_deg, rw_ - 0.42, z), rot=(90, 0, yaw)))
        outer = [(x, y - f) for x, y in arch_pts(w + 2 * f, h + 2 * f, kind=kind, n=10)]
        fr = shape_extrude("tw_frame", outer, [arch_pts(w, h, kind=kind, n=10)], depth=0.31, loc=t_at(a_deg, rw_ - 0.25, z),
                           rot=(90, 0, yaw), bevel=0.008)
        assign(fr, fmat)
        if fmat is oak:
            uv_grain(fr, "Y")
            timber_objs.append(fr)
        else:
            dressed_objs.append(fr)
        if kind == "round" and fmat is dressed:
            ks = box("tw_keystone", (0.13, 0.11, f + 0.11), loc=t_at(a_deg, rw_ + 0.03, z + h + f * 0.5 + 0.015), rot=(0, 0, yaw), bevel=0.012, segments=1)
            assign(ks, fmat)
            dressed_objs.append(ks)
        if sill:
            so = box("tw_sill", (w + 2 * f + 0.1, 0.2, 0.085), loc=t_at(a_deg, rw_ + 0.0, z - f - 0.03), rot=(0, 0, yaw), bevel=0.014, segments=1)
            assign(so, fmat)
            (timber_objs if fmat is oak else dressed_objs).append(so)
        _pane("tw_pane", w + 0.08, h + 0.08, {"lit": lit_t, "dim": lit_b, "dark": dark}[state], t_at(a_deg, rw_ - 0.185, z + h / 2), yaw,
              0.3 if w < 0.3 else 0.12, 0.7 if w < 0.3 else 0.88)
        if cames:
            for o in (box("tw_came", (0.016, 0.016, h), loc=t_at(a_deg, rw_ - 0.172, z + h / 2), rot=(0, 0, yaw)),
                      box("tw_came", (w, 0.016, 0.016), loc=t_at(a_deg, rw_ - 0.172, z + h * 0.36), rot=(0, 0, yaw)),
                      box("tw_came", (w, 0.016, 0.016), loc=t_at(a_deg, rw_ - 0.172, z + h * 0.66), rot=(0, 0, yaw))):
                assign(o, lead)
        else:
            o = box("tw_bar", (0.02, 0.02, h), loc=t_at(a_deg, rw_ - 0.1, z + h / 2), rot=(0, 0, yaw + 45))
            assign(o, iron)

    # stair windows: three per turn of the stair, climbing from the front round to the left flank
    turns = st
    z_first = 1.55
    rev_h = (zc0 - 1.55 - z_first) / turns + 0.0
    angs = [270.0, 225.0, 180.0] if octa else [279.0, 229.0, 181.0]
    k_all = 0
    for tn in range(turns):
        big = rt_.pick([0, 1, 1, 2])
        for i, a in enumerate(angs):
            z = z_first + tn * max(rev_h, 2.2) + i * 0.52
            if not octa:
                a += rt_.u(-4, 4)
            if i == big:
                hgt = 0.95
                if z + hgt > zc0 - 0.45:
                    continue
                state = "lit" if (rt_.chance(p.lit) or tn == 0) else "dark"
                opening(a, z, 0.46, hgt, "round", state, R, cut_shaft, dressed)
                if state == "lit" and not any(l[1] == "stair_window" for l in lights):
                    lights.append((2, "stair_window", t_at(a, R + 0.5, z + 0.5), 5.0, 5.0))
            else:
                hgt = 1.05
                if z + hgt > zc0 - 0.45:
                    continue
                state = "dim" if rt_.chance(p.lit * 0.7) else "dark"
                opening(a, z, 0.15, hgt, "flat", state, R, cut_shaft, dressed, f=0.1, cames=False, sill=False)
            k_all += 1

    # string courses at each floor
    def string_ring(z, rad, m, big=False):
        s = 1.5 if big else 1.0
        o = L("turret_string", [(rad - 0.03, z - 0.07 * s), (rad + 0.035 * s, z - 0.07 * s), (rad + 0.06 * s, z - 0.035 * s),
                                (rad + 0.06 * s, z + 0.03 * s), (rad + 0.02, z + 0.075 * s), (rad - 0.03, z + 0.075 * s)], m)
        (timber_objs if m is oak else dressed_objs).append(o)
        return o
    for k in range(st - 1):
        string_ring(zg + k * sh - 0.1, R, dressed)

    # ---- lookout stage
    look_angs = [135.0, 180.0, 225.0, 270.0, 315.0]
    look_h, look_w = 1.15, 0.56
    look_z = max(He + 0.75, Ht - 1.95)
    if not timber_top:
        sp = [(0.0, zc0), (R + 0.015, zc0)]
        for i in range(1, 6):
            t = i / 5.0
            sp.append((R + 0.015 + (Rt - R - 0.015) * (1 - cos(t * pi / 2)), zc0 + (zc1 - zc0) * sin(t * pi / 2)))
        sp += [(Rt, Ht - 0.3), (Rt + 0.03, Ht - 0.26), (Rt + 0.05, Ht - 0.2), (Rt + 0.05, Ht - 0.1), (Rt, Ht - 0.04), (Rt, Ht), (0.0, Ht)]
        stage = L("turret_stage", sp, wall_mat)
        stone_objs.append(stage)
        string_ring(zc0 - 0.03, R, dressed, big=True)
        string_ring(look_z - 0.32, Rt, dressed)
        # corbel stones carrying the oversailing stage
        ncb = 16 if octa else max(12, int(2 * pi * R / 0.46))
        cb = box("corbel", (0.17, 0.26, 0.24), bevel=0.05, segments=2)
        assign(cb, dressed)
        ms = []
        for i in range(ncb):
            a = 360.0 * (i + 0.5) / ncb
            ms.append(trs(t_at(a, rad_at(a, R) + 0.07, zc0 + 0.2), (0, 0, a + 90)))
        dressed_objs.append(instances_to_mesh("corbels", cb, ms, keep_base=False))
        for a in look_angs:
            state = "lit" if (a in (225.0, 270.0, 315.0) or rt_.chance(p.lit)) else "dark"
            opening(a, look_z, look_w, look_h, "round", state, Rt, cut_stage, dressed)
        # carved date tablet on the front of the stage
        tb_h = min(0.46, look_z - 0.42 - zc1 - 0.16)
        if str(p.date).strip() and tb_h > 0.3:
            ta = 270.0
            tz = (zc1 + look_z - 0.4) / 2
            tw = 0.34 + 0.15 * len(str(p.date).strip())
            tr = rad_at(ta, Rt)
            dressed_objs.append(blk("date_tablet", (tw, 0.12, tb_h), t_at(ta, tr - 0.015, tz), rot=(0, 0, ta + 90), bev=0.012))
            rim = shape_extrude("date_rim", rounded_rect(tw + 0.07, tb_h + 0.07, 0.02), [rounded_rect(tw - 0.07, tb_h - 0.07, 0.012)], depth=0.03,
                                loc=t_at(ta, tr + 0.04, tz), rot=(90, 0, ta + 90), bevel=0.006)
            assign(rim, dressed)
            dressed_objs.append(rim)
            tx = text("date_text", str(p.date).strip(), size=tb_h * 0.5, depth=0.012, loc=t_at(ta, tr + 0.045, tz - tb_h * 0.17),
                      rot=(90, 0, ta + 90), font="serif")
            assign(tx, lead)
    else:
        zj = zc0
        jr = L("turret_jetty", [(R - 0.03, zj - 0.03), (R + 0.06, zj - 0.03), (R + 0.1, zj + 0.03), (R + 0.2, zj + 0.13), (Rt + 0.05, zj + 0.17),
                                (Rt + 0.05, zj + 0.4), (R - 0.03, zj + 0.4)], oak)
        timber_objs.append(jr)
        stage = L("turret_stage", [(0.0, zj + 0.3), (Rt, zj + 0.3), (Rt, Ht), (0.0, Ht)], plaster)
        plaster_objs.append(stage)
        timber_objs.append(L("turret_plate", [(Rt - 0.03, Ht - 0.24), (Rt + 0.045, Ht - 0.24), (Rt + 0.045, Ht - 0.02), (Rt - 0.03, Ht - 0.02)], oak))
        rail_z = look_z - 0.2
        timber_objs.append(L("turret_rail", [(Rt - 0.03, rail_z - 0.08), (Rt + 0.035, rail_z - 0.08), (Rt + 0.035, rail_z + 0.08),
                                             (Rt - 0.03, rail_z + 0.08)], oak))
        # brackets under the jetty and posts between the windows
        nbr = 16
        br = extrude_profile("tj_bracket", [(0.0, 0.0), (0.0, -0.42), (0.05, -0.42), (0.1, -0.2), (0.2, -0.04), (0.2, 0.0)], 0.1, bevel=0.006)
        transform_verts(br, lambda co: Vector((co.z - 0.05, -co.x, co.y)))
        assign(br, oak)
        ms = [trs(t_at(22.5 + 360.0 * i / nbr, rad_at(22.5 + 360.0 * i / nbr, R) - 0.01, zj + 0.0), (0, 0, 22.5 + 360.0 * i / nbr + 90)) for i in range(nbr)]
        timber_objs.append(instances_to_mesh("tj_brackets", br, ms, keep_base=False))
        z0p, z1p = zj + 0.38, Ht - 0.22
        for i in range(8):
            a = 22.5 + 45.0 * i
            rr = rad_at(a, Rt)
            o = box("t_post", (0.2, 0.14, z1p - z0p), loc=t_at(a, rr - 0.03, (z0p + z1p) / 2), rot=(0, 0, a + 90), bevel=0.007, segments=1)
            uv_grain(o, "Z")
            assign(o, oak)
            timber_objs.append(o)
            # saltire under each lookout window
            am = 45.0 * i
            rm = rad_at(am, Rt)
            half = rm * tan(radians(22.5)) - 0.12
            for sgn in (-1, 1):
                ln = hypot(2 * half, rail_z - 0.1 - z0p)
                ang = degrees(atan2(rail_z - 0.1 - z0p, 2 * half)) * sgn
                o = box("t_saltire", (ln, 0.1, 0.11), loc=t_at(am, rm * (0.985 if not octa else 1.0) - 0.022 + 0.004 * sgn, (z0p + rail_z - 0.1) / 2),
                        rot=(0, -ang, am + 90), bevel=0.006, segments=1)
                uv_grain(o, "X")
                assign(o, oak)
                timber_objs.append(o)
        for a in look_angs:
            state = "lit" if (a in (225.0, 270.0, 315.0) or rt_.chance(p.lit)) else "dark"
            opening(a, look_z, look_w + 0.1, look_h - 0.1, "flat", state, Rt, cut_stage, oak, f=0.09)
    lights.append((1, "lookout", t_at(250.0, Rt + 0.9, look_z + 0.35), 6.0, 7.0))
    if cut_shaft:
        boolean(shaft, join(cut_shaft, "cut_shaft"))
    if cut_stage:
        boolean(stage, join(cut_stage, "cut_stage"))

    # ---- conical roof: bell-cast eaves, boarded soffit on rafter feet, individually laid covering
    tcover = "copper" if p.turret_roof == "copper" else p.roof
    Pc = radians(p.cone_pitch)
    Pc2 = Pc - radians(17.0)
    re_, ze = Rt + eave_t, Ht - 0.1
    Ls = 0.8
    r1, z1c = re_ - Ls * cos(Pc2), ze + Ls * sin(Pc2)
    za = z1c + r1 * tan(Pc)
    S = Ls + r1 / cos(Pc)

    def cone_at(s):
        if s <= Ls:
            return re_ - s * cos(Pc2), ze + s * sin(Pc2), Pc2
        return max(0.0, r1 - (s - Ls) * cos(Pc)), z1c + (s - Ls) * sin(Pc), Pc
    under = {"tile": mat("terracotta", "#c98f78"), "slate": mat("slate", "#7f8791"), "shingle": mat("wood_weathered", "#a39680"),
             "copper": mat("copper_patina")}[tcover]     # the laid tiles / slates / shingles use the same material
    cone_o = L("turret_cone", [(Rt - 0.12, ze - 0.03), (re_ - 0.03, ze - 0.03), (re_, ze + 0.0), (r1, z1c), (0.0, za)], under,
               segments=8 if octa else 48)
    assign_faces(cone_o, boards, lambda cc, n: n.z < -0.5)
    # rafter feet
    nrf = 16 if octa else int(2 * pi * re_ / 0.42)
    rf = box("rafter_foot", (0.075, 0.34, 0.1), bevel=0.006, segments=1)
    uv_grain(rf, "Y")
    assign(rf, oak)
    ms = []
    for i in range(nrf):
        a = 360.0 * (i + 0.5) / nrf
        ms.append(trs(t_at(a, rad_at(a, Rt) + 0.13, ze - 0.075), (0, 0, a + 90)))
    timber_objs.append(instances_to_mesh("rafter_feet", rf, ms, keep_base=False))

    def slope_matrix(a_deg, rr, zz, pitch, x_off=0.0):
        a = radians(a_deg)
        o_ = Vector((cos(a), sin(a), 0.0))
        t_ = Vector((-sin(a), cos(a), 0.0))
        u_ = -o_ * cos(pitch) + Vector((0, 0, 1)) * sin(pitch)
        n_ = o_ * sin(pitch) + Vector((0, 0, 1)) * cos(pitch)
        pos = Vector((cx, cy, 0)) + o_ * rr + t_ * x_off + Vector((0, 0, zz)) + n_ * 0.004
        return Matrix(((t_.x, u_.x, n_.x, pos.x), (t_.y, u_.y, n_.y, pos.y), (t_.z, u_.z, n_.z, pos.z), (0, 0, 0, 1)))

    tile_objs = []
    if tcover != "copper":
        spec = {"tile": (0.185, 0.30, 0.155, 0.016, 0.02, mat("terracotta", "#c98f78")),
                "slate": (0.25, 0.42, 0.2, 0.009, 0.012, mat("slate", "#7f8791")),
                "shingle": (0.14, 0.4, 0.17, 0.013, 0.016, mat("wood_weathered", "#a39680"))}[tcover]
        w0, tl, ex, th, lift, tmat = spec
        hw_ = w0 / 2
        vs = [(-hw_, 0, lift), (hw_, 0, lift), (hw_, 0, lift + th), (-hw_, 0, lift + th),
              (-hw_, tl, -0.004), (hw_, tl, -0.004), (hw_, tl, th - 0.004), (-hw_, tl, th - 0.004)]
        fs = [(3, 2, 6, 7), (0, 1, 2, 3), (1, 5, 6, 2), (0, 3, 7, 4)]
        tile = mesh_object("cone_tile", vs, fs)
        assign(tile, tmat)
        if tcover == "shingle":
            uv_grain(tile, "Y")
        else:
            uv_box(tile)
        realize(tile)
        rtl = rng.fork("cone_tiles")
        ms = []
        k = 0
        s = -0.04
        while s < S - 0.42:
            rr, zz, pit = cone_at(max(0.0, s))
            if s < 0:
                rr, zz = rr - s * cos(pit), zz + s * sin(pit)
            if octa:
                hwf = rr * tan(radians(22.5))
                n = int((2 * hwf - 0.05) / w0)
                if n >= 1:
                    wdt = (2 * hwf - 0.03) / n
                    odd = k % 2 == 1 and n > 1
                    for fct in range(8):
                        for i in range(n - (1 if odd else 0)):
                            xo = -hwf + 0.015 + (i + (1.0 if odd else 0.5)) * wdt
                            M = slope_matrix(45.0 * fct, rr, zz, pit, xo)
                            ms.append(M @ trs((0, 0, 0), (rtl.u(-1.5, 1.5), 0, rtl.u(-1.8, 1.8)), (wdt / w0 * rtl.u(0.93, 0.99), rtl.u(0.96, 1.04), 1.0)))
            else:
                n = max(5, int(round(2 * pi * rr / w0)))
                wdt = 2 * pi * rr / n
                a0 = rtl.u(0, 360)
                for i in range(n):
                    if tcover == "tile" and age > 0.3 and rtl.chance(0.004 * age):
                        continue
                    M = slope_matrix(a0 + 360.0 * i / n, rr, zz, pit)
                    ms.append(M @ trs((0, 0, 0), (rtl.u(-1.8, 1.8), 0, rtl.u(-2.2, 2.2)),
                                      (wdt / w0 * rtl.u(0.94, 1.0), rtl.u(0.95, 1.05), rtl.u(0.9, 1.25))))
            s += ex
            k += 1
        tiles = instances_to_mesh("cone_tiles", tile, ms, keep_base=False, seed=p.seed)
        tile_objs.append(tiles)
        if octa:
            hipm = lead if tcover != "tile" else mat("terracotta", "#c98f78")
            for i in range(8):
                a = 22.5 + 45.0 * i
                pts = []
                for q in range(0, 13):
                    s_ = (S - 0.3) * q / 12.0
                    rr, zz, pit = cone_at(s_)
                    pts.append(t_at(a, rr * kf + 0.01, zz + 0.035))
                h = sweep("cone_hip", pts, 0.06, sides=8, smooth_path=False)
                assign(h, hipm)
                tile_objs.append(h)
    else:
        nsm = 16
        for i in range(nsm):
            a = 22.5 + 360.0 * i / nsm
            pts = []
            for q in range(0, 11):
                s_ = (S - 0.35) * q / 10.0
                rr, zz, pit = cone_at(s_)
                pts.append(t_at(a, rad_at(a, rr) + 0.004, zz + 0.012))
            sm_ = sweep("cone_seam", pts, [(-0.008, -0.01), (0.008, -0.01), (0.008, 0.03), (-0.008, 0.03)], smooth_path=False,
                        up=(cos(radians(a)), sin(radians(a)), 0.4))
            assign(sm_, under)
            tile_objs.append(sm_)
        for s_ in (Ls, Ls + (S - Ls) * 0.45):
            rr, zz, pit = cone_at(s_)
            tile_objs.append(L("cone_welt", [(rr + 0.004, zz - 0.02), (rr + 0.022, zz - 0.012), (rr + 0.018, zz + 0.012), (rr - 0.004, zz + 0.02)],
                               under, segments=8 if octa else 48))
        tile_objs.append(cone_o)
    # finial: lead (or copper) cap, ball and spike
    fin_m = under if tcover == "copper" else lead
    rr = 0.3
    fin = lathe("finial", [(rr + 0.02, za - rr * tan(Pc) - 0.03), (0.075, za - 0.05), (0.055, za + 0.06), (0.085, za + 0.1), (0.12, za + 0.17),
                           (0.125, za + 0.23), (0.1, za + 0.3), (0.05, za + 0.36), (0.035, za + 0.42), (0.05, za + 0.46),
                           (0.03, za + 0.52), (0.022, za + 0.7), (0.0, za + 0.71)], segments=20, loc=(cx, cy, 0))
    assign(fin, fin_m)
    # weathervane
    if p.vane != "none":
        gilt = mat("brass")
        # a stout gilt staff straight off the finial, the vane close above it: a hairline iron rod drops out of the
        # far LODs and of distant pixels alike, and the vane is then left hanging in the sky over the cap
        zr0, zr1 = za + 0.5, za + 1.72
        rod = cone("vane_rod", r1=0.034, r2=0.02, depth=zr1 - zr0, loc=(cx, cy, (zr0 + zr1) / 2), verts=8)
        assign(rod, gilt)
        zc_ = za + 0.92
        for rz in (0, 90):
            o = cylinder("vane_arm", 0.014, 0.84, loc=(cx, cy, zc_), rot=(0, 90, rz), verts=6)
            assign(o, iron)
            for sgn in (-1, 1):
                a = radians(rz)
                b_ = sphere("vane_ball", 0.032, loc=(cx + cos(a) * 0.42 * sgn, cy + sin(a) * 0.42 * sgn, zc_), segments=10, rings=6)
                assign(b_, gilt)
        col = lathe("vane_collar", [(0.03, zc_ - 0.07), (0.055, zc_ - 0.035), (0.062, zc_), (0.055, zc_ + 0.035), (0.03, zc_ + 0.07)], segments=12, loc=(cx, cy, 0))
        assign(col, iron)
        zv = za + 1.16
        sc = 1.25
        shape = [(x * sc, y * sc) for x, y in (ROOSTER if p.vane == "rooster" else BANNER)]
        vn = shape_extrude("vane", shape, depth=0.008, loc=(cx, cy + 0.004, zv), rot=(90, 0, 0), bevel=0.002)
        assign(vn, gilt)
        slv = cylinder("vane_sleeve", 0.04, 0.2, loc=(cx, cy, zv - 0.05), verts=10)
        assign(slv, gilt)
        top = sphere("vane_top", 0.04, loc=(cx, cy, zr1), segments=10, rings=6)
        assign(top, gilt)
        paint(vn, special=mask_mul(noise_mask(0.15, seed=p.seed), 0.35 + 0.4 * age))
        part([vn, slv], "vane", pivot=(cx, cy, zv), motion="spin", axis="Z", speed=0.08)

    # ---- lead flashing where the main roof dies against the turret
    fl = []
    for q in range(0, 73):
        a = -70.0 + q * 3.0
        rr = rad_at(a, Rt if True else R) + 0.012
        x, y, _ = t_at(a, rr, 0)
        if x < -W / 2 - 0.12 or y < -j - ov + 0.1 or y > ry:
            continue
        zz = roof_z(y) + 0.02
        if zz > Ht - 0.4 or zz < zc1:
            continue
        fl.append((x, y, zz))
    if len(fl) >= 3:
        f_o = sweep("turret_flashing", fl, [(-0.007, -0.06), (0.007, -0.06), (0.007, 0.16), (-0.007, 0.16)], smooth_path=not octa)
        assign(f_o, lead)

    # ================================================================ WEATHERING
    north = facing((0.3, 1.0, 0.25), 2.0)
    for o in stone_objs:
        paint(o, grime=lambda co, n: np.clip(np.maximum(height_ramp(0.0, 1.4)(co, n) * 0.8, noise_mask(1.7, seed=p.seed + 1)(co, n) * 0.35 * (0.5 + age)), 0, 1),
              special=mask_mul(north, noise_mask(0.8, seed=p.seed + 2, lo=0.3, hi=0.7), 0.25 + 0.6 * age),
              wear=mask_mul(noise_mask(0.6, seed=p.seed + 8, lo=0.5, hi=0.8), 0.3))
    for o in dressed_objs:
        paint(o, grime=mask_mul(height_ramp(0.0, 1.0), 0.7), special=mask_mul(facing((0, 0, 1), 2.0), noise_mask(0.4, seed=p.seed + 4), 0.3 + 0.5 * age))
    for o in plaster_objs:
        paint(o, grime=lambda co, n: np.clip(noise_mask(1.1, seed=p.seed + 6, lo=0.4, hi=0.75)(co, n) * (0.25 + 0.45 * age)
                                              + height_ramp(zs, zs + 0.9)(co, n) * 0.35, 0, 1),
              special=mask_mul(north, noise_mask(1.4, seed=p.seed + 7, lo=0.35, hi=0.7), 0.2 + 0.6 * age),
              wear=mask_mul(noise_mask(0.5, seed=p.seed + 12, lo=0.62, hi=0.8), 0.2 + 0.5 * age))
    for o in timber_objs:
        paint(o, grime=mask_mul(noise_mask(0.9, seed=p.seed + 9), 0.2 + 0.3 * age), wear=mask_mul(noise_mask(0.35, seed=p.seed + 10, lo=0.5, hi=0.8), 0.25 + 0.3 * age))
    for o in tile_objs + [cone_o]:
        paint(o, grime=mask_mul(noise_mask(0.45, seed=p.seed + 13, lo=0.3, hi=0.8), 0.3 + 0.3 * age),
              special=lambda co, n: np.clip(noise_mask(0.7, seed=p.seed + 14, lo=0.4, hi=0.75)(co, n) * (0.25 + 0.6 * age)
                                            * (0.35 + 0.65 * north(co, n)) + (0.5 if tcover == "copper" else 0.0)
                                            * noise_mask(1.2, seed=p.seed + 15, lo=0.2, hi=0.7)(co, n), 0, 1))
    paint(fin, grime=0.2, special=0.4)

    # ================================================================ LIGHTS (max 6, most important first)
    lights.sort(key=lambda l: l[0])
    for (prio, name, loc, cd, rangem) in lights[:6]:
        light(name, loc, intensity=cd, range=rangem, flicker=0.14 if name == "door_lantern" else 0.08, shadow_near=0.45)
    socket("lantern_right", "attach", GR.pt(1.4, 2.8), rot=(0, 0, GR.phi), role="lantern")
    socket("planter_front", "attach", (min(fspan1 - 0.3, xd + 2.2), -1.0, 0.0), role="prop_top")
