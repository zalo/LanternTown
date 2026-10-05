"""Tree pit grate and guard. A cast-iron grate in four bolted-down segments lies flush in the paving
(walking surface 10 mm above the cobbles): every segment has a raised cast border, wedge-shaped
radial slots in concentric rings, raised ring ribs, foundry lettering and a thickened hub collar
round the planting hole onto which the guard is bolted. The casting sits in a thin angle-iron lip
inside a low dressed-stone kerb (stones cut to the plan, worn arris, sunken mortar joints, mortar
fillet down to the cobbles) over a pit of soil or gravel that is recessed below the collar. The
wrought-iron guard is 30 x 8 flat bar: forged bolted feet, two flat hoops made in lugged halves so
the guard opens round the trunk, a rivet at every crossing, an optional band of collared rings and
a flared top with rolled, scrolled or spear tips.
"""
from math import asin, atan2, cos, degrees, hypot, pi, radians, sin

import numpy as np

from tk import *

META = {"name": "Tree pit grate and guard", "class": "prop", "mount": "ground", "focus": [0.12, -0.3, 0.22]}
PARAMS = {
    "size": {"default": 1.3, "min": 0.9, "max": 1.8, "desc": "overall side / diameter incl. kerb (m)"},
    "shape": {"default": "square", "choices": ["square", "round"], "desc": "plan shape of grate and kerb"},
    "hole": {"default": 0.42, "min": 0.3, "max": 0.6, "desc": "diameter of the central planting hole (m)"},
    "rings": {"default": 2, "min": 1, "max": 3, "desc": "concentric rings of radial slots"},
    "kerb": {"default": True, "desc": "dressed stone kerb around the frame (False = thin chamfered iron lip only)"},
    "guard": {"default": True, "desc": "wrought-iron tree guard around the hole"},
    "guard_height": {"default": 1.6, "min": 1.0, "max": 2.1, "desc": "height of the guard above the grate (m)"},
    "bars": {"default": 14, "min": 8, "max": 20, "desc": "number of vertical guard bars"},
    "guard_top": {"default": "flare", "choices": ["flare", "scroll", "spear"], "desc": "tip style of the flared top"},
    "ornament": {"default": "rings", "choices": ["rings", "none"], "desc": "band of collared rings between the bars under the top hoop"},
    "flare": {"default": 0.11, "min": 0.04, "max": 0.2, "desc": "how far the top flares outwards (m)"},
    "gilt_tips": {"default": False, "desc": "gilded bar tips"},
    "paint": {"default": "#3d6351", "desc": "paint colour of the guard (and of a painted grate)"},
    "grate_finish": {"default": "bare", "choices": ["bare", "painted", "rusty"], "desc": "grate: graphite-grey bare casting, painted, or rusted"},
    "maker": {"default": "LANTERN 1887", "desc": "foundry lettering cast into the front segment ('' = none)"},
    "fill": {"default": "soil", "choices": ["soil", "gravel"], "desc": "what fills the pit"},
    "leaves": {"default": 9, "min": 0, "max": 30, "desc": "fallen leaves caught in the slots and against the kerb"},
    "weeds": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "grass tufts growing in the pit and through the slots"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: rust, chipped paint, moss, bent bars"},
}
VARIANTS = {
    "round_scroll": {"shape": "round", "size": 1.5, "rings": 3, "bars": 16, "guard_top": "scroll", "ornament": "none", "guard_height": 1.45,
                     "flare": 0.14, "paint": "#3d5f92", "maker": "T.T. IRONWORKS", "seed": 4},
    "grate_only_rusty": {"guard": False, "size": 1.1, "hole": 0.36, "kerb": False, "grate_finish": "rusty", "leaves": 16,
                         "age": 0.9, "weeds": 1.0, "fill": "gravel", "seed": 11},
    "tall_spear_gilt": {"shape": "round", "size": 1.2, "hole": 0.46, "guard_height": 2.0, "bars": 12, "guard_top": "spear",
                        "gilt_tips": True, "paint": "#74302a", "grate_finish": "painted", "flare": 0.08, "age": 0.25, "leaves": 4, "weeds": 0.0, "seed": 23},
    "small_plain": {"size": 0.95, "hole": 0.34, "rings": 1, "guard_height": 1.2, "bars": 10, "flare": 0.07, "ornament": "none",
                    "paint": "#474c52", "maker": "", "age": 0.65, "seed": 31},
}

ZB = 0.002         # underside of the casting above the paving (it beds on the frame lugs)
ZW = 0.005         # top of the plate web above ZB            ->  7 mm above the paving
ZR = 0.008         # raised borders / ribs above ZB           -> 10 mm: the walking surface
ZH = 0.021         # hub collar round the hole above ZB       -> 23 mm, inside the guard
SEAM = 0.0025      # half width of the gap between segments
BORDER = 0.016     # raised cast border of every segment
FIL = 0.012        # mortar fillet between kerb and cobbles
NC = 48            # points round a full circle of the casting


def _tris(objs):
    return sum(len(f.vertices) - 2 for o in objs if o is not None for f in o.data.polygons)


def _drop_under(o):
    """Remove the never-seen underside of an extruded plate."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    dead = [f for f in bm.faces if f.normal.z < -0.3]
    bmesh.ops.delete(bm, geom=dead, context='FACES')
    bm.to_mesh(o.data)
    bm.free()
    return o


def _two_sided(o, gap=0.0004):
    """Give a one-sided sheet (leaf, grass blade) a flipped copy just underneath."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(o.data)
    ret = bmesh.ops.duplicate(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:])
    nf = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMFace)]
    nv = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]
    bm.normal_update()
    for v in nv:
        v.co -= v.normal * gap
    bmesh.ops.reverse_faces(bm, faces=nf)
    bm.to_mesh(o.data)
    bm.free()
    return o


def _loft(name, rings, closed=False, caps=False):
    """Quad strips between successive rings of 3D points. Rings run from the inside outwards over
    the top, points counter-clockwise seen from above -> normals point up / out."""
    n = len(rings[0])
    verts = [tuple(q) for r in rings for q in r]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n if closed else n - 1):
            j2 = (j + 1) % n
            faces.append((i * n + j, (i + 1) * n + j, (i + 1) * n + j2, i * n + j2))
    if caps and not closed:
        faces.append(tuple(i * n + n - 1 for i in range(len(rings))))
        faces.append(tuple(i * n for i in reversed(range(len(rings)))))
    return mesh_object(name, verts, faces)


def _mix(col, other, k):
    """Blend two '#RRGGBB' colours (k = share of `other`)."""
    try:
        a = [int(col.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
        b = [int(other.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return col
    return "#" + "".join("%02x" % int(round(x + (y - x) * k)) for x, y in zip(a, b))


def _lift(col, k):
    """Lighten a colour a little (dark paint must still read under lantern light)."""
    return _mix(col, "#ffffff", k)


def _quad_outline(shape, R, r_in, g, crn=0.02, n_arc=10):
    """Outline (CCW) of the grate segment in the first quadrant."""
    pts = []
    if shape == "round":
        a_o = asin(g / R)
        m = NC // 4
        for i in range(m + 1):
            a = a_o + (pi / 2 - 2 * a_o) * i / m
            pts.append((R * cos(a), R * sin(a)))
    else:
        pts.append((R, g))
        for i in range(4):
            a = (pi / 2) * i / 3
            pts.append((R - crn + crn * cos(a), R - crn + crn * sin(a)))
        pts.append((g, R))
    a_i = asin(min(0.9, g / r_in))
    for i in range(n_arc + 1):
        a = (pi / 2 - a_i) - (pi / 2 - 2 * a_i) * i / n_arc
        pts.append((r_in * cos(a), r_in * sin(a)))
    return pts


def _slot(a, r0, r1, hw0, hw1, ns=3):
    er = (cos(a), sin(a))
    et = (-sin(a), cos(a))
    pts = []
    for (c, hw, b0) in ((r1 - hw1, hw1, -pi / 2), (r0 + hw0, hw0, pi / 2)):
        for i in range(ns + 1):
            b = b0 + pi * i / ns
            rr, tt = c + hw * cos(b), hw * sin(b)
            pts.append((er[0] * rr + et[0] * tt, er[1] * rr + et[1] * tt))
    return pts


def _loop(shape, hh, z, crn=0.02):
    pts = circle_pts(hh, n=NC) if shape == "round" else rounded_rect(2 * hh, 2 * hh, crn, 2)
    return [(q[0], q[1], z) for q in pts]


def build(p, rng):
    shape = p.shape
    rnd = shape == "round"
    half = p.size / 2
    age = p.age
    kw = 0.09 if p.kerb else 0.0
    fw = 0.010 if p.kerb else 0.032
    gh = half - kw - fw - 0.003              # half extent of the casting itself
    rh = min(p.hole / 2, gh - 0.215)         # planting hole radius
    hubf = 0.070                             # flat of the hub collar (the guard feet bolt onto it)
    rhub = rh + hubf + 0.018                 # foot of the collar ramp
    ra = rhub + BORDER + 0.004               # where the slots start
    mrg = clamp(0.07 * p.size / 1.3, 0.05, 0.075)   # solid band between the slot ends and the outer edge
    nr = min(int(p.rings), max(1, int((gh - mrg - ra) / 0.07)))
    stat = {}

    paint_l = _lift(p.paint, 0.14)
    # the casting: graphite-grey web, the raised borders / ribs / lettering polished paler by feet
    if p.grate_finish == "painted":             # old paint gone dull in the web, rubbed thin and pale on the raised parts
        iron_g = mat("iron_painted", _mix(p.paint, "#3b3b3d", 0.45), rough=1.1, grime=0.8)
        iron_hi = mat("iron_painted", _mix(_lift(p.paint, 0.1), "#8a8c8e", 0.4), rough=0.7, grime=0.8)
    elif p.grate_finish == "rusty":
        iron_g = mat("iron_painted", "#5a5049", rough=1.4, grime=0.8, special=0.55)
        iron_hi = mat("iron_painted", "#857e78", rough=0.75, grime=0.8, special=0.55)
    else:
        iron_g = mat("iron_painted", "#50555b", rough=1.25, grime=0.8)
        iron_hi = mat("iron_painted", "#83878b", rough=0.55, grime=0.8)
    iron = mat("iron_painted", paint_l, rough=0.6)
    gilt = mat("metal_plain", "#d4a84b", rough=0.4)
    stone = mat("stone_rough", "#a9a193", grime=0.7)
    dirt = mat("soil")

    def r_edge(a):
        if rnd:
            return gh
        return gh / max(abs(cos(a)), abs(sin(a)))

    # ------------------------------------------------------------------ one cast segment (first quadrant)
    pitch = 0.055 * max(1.0, p.size / 1.3)
    ribw = 0.03
    a_m = asin(min(0.5, (SEAM + BORDER + 0.01) / ra))
    span = pi / 2 - 2 * a_m
    slots, slot_ends = [], []
    for j in range(nr):
        f0, f1 = j / nr, (j + 1) / nr
        rm = ra + (gh - mrg - ra) * (f0 + f1) / 2
        nj = max(3, int(round(span * rm / pitch)))
        da = span / nj
        for i in range(nj):
            a = a_m + (i + 0.5) * da
            ro = r_edge(a) - mrg
            r0 = ra + (ro - ra) * f0 + (ribw / 2 if j > 0 else 0.0)
            r1 = ra + (ro - ra) * f1 - (ribw / 2 if j < nr - 1 else 0.0)
            hw0 = clamp(0.25 * r0 * da, 0.006, 0.012)
            hw1 = clamp(0.25 * r1 * da, 0.006, 0.012)
            if r1 - r0 > 2.4 * (hw0 + hw1):
                slots.append(_slot(a, r0, r1, hw0, hw1))
                slot_ends.append((a, r0 + hw0, r1 - hw1))
    plate = shape_extrude("plate", _quad_outline(shape, gh, rhub - 0.004, SEAM), slots, depth=ZW)
    _drop_under(plate)
    pieces = [plate]
    brd = shape_extrude("border", _quad_outline(shape, gh - 0.0005, rhub - 0.002, SEAM + 0.0005),
                        [_quad_outline(shape, gh - BORDER, rhub + BORDER - 0.004, SEAM + BORDER, crn=0.008)],
                        depth=ZR - ZW + 0.0005, loc=(0, 0, ZW - 0.0005))
    apply_transform(brd)
    _drop_under(brd)
    pieces.append(brd)
    rib_prof = [(-0.0075, 0.0), (0.0075, 0.0), (0.005, ZR - ZW), (-0.005, ZR - ZW)]
    for j in range(1, nr):
        f = j / nr
        pts = []
        m = 10 if rnd else 12
        for i in range(m + 1):
            a = a_m * 0.75 + (pi / 2 - 1.5 * a_m) * i / m
            r = ra + (r_edge(a) - mrg - ra) * f
            pts.append((r * cos(a), r * sin(a), ZW - 0.0003))
        pieces.append(_drop_under(sweep(f"rib{j}", pts, rib_prof, smooth_path=False, cap=True)))

    # hub collar round the hole: thick inner wall, flat for the guard feet, ramp down to the border
    def hub_ring(r, z, n=10):
        a_s = asin(min(0.9, SEAM / r))
        return [(r * cos(a_s + (pi / 2 - 2 * a_s) * i / n), r * sin(a_s + (pi / 2 - 2 * a_s) * i / n), z) for i in range(n + 1)]

    hub = _loft("hub", [hub_ring(rh, 0.0), hub_ring(rh, ZH - 0.0035), hub_ring(rh + 0.0035, ZH), hub_ring(rh + hubf, ZH),
                        hub_ring(rhub, ZR)], caps=True)
    smooth(hub, 30)
    pieces.append(hub)
    # holding-down bolts through the outer band into the frame lugs
    bt = cylinder("hd_bolt", 0.0125, 0.004, verts=4, radius2=0.0105, loc=(0, 0, 0.002))
    apply_transform(bt)
    _drop_under(bt)
    flat(bt)
    bm_, bolts_xy = [], []
    for da_ in (-36.0, 36.0):
        a = radians(45.0 + da_)
        rb = r_edge(a) - 0.038 / max(abs(cos(a)), abs(sin(a))) if not rnd else gh - 0.038
        bm_.append(trs((rb * cos(a), rb * sin(a), ZW), (0, 0, degrees(a) + rng.u(-25, 25))))
        bolts_xy.append((rb * cos(a), rb * sin(a)))
    pieces.append(instances_to_mesh("hd_bolts", bt, bm_, keep_base=False))
    for o in pieces:
        assign(o, iron_g if o in (plate, hub) else iron_hi)
    seg0 = join(pieces, "segment")
    r_seg = rng.fork("segments")
    segs = []
    for k in range(4):
        s = seg0 if k == 0 else duplicate(seg0, f"segment{k}")
        still = (k == 3)                      # the lettered front segment stays true
        jz = 0.0 if still else r_seg.u(-0.0005, 0.0005) * (0.4 + age)
        place(s, loc=(0, 0, ZB + jz),
              rot=(0 if still else r_seg.u(-0.05, 0.05) * (0.4 + age), 0 if still else r_seg.u(-0.05, 0.05) * (0.4 + age), k * 90.0))
        segs.append(s)

    # ------------------------------------------------------------------ foundry lettering on the front-right segment
    letters = []
    if p.maker.strip():
        tz = ZB + ZW - 0.0005
        tsize = max(0.02, (0.8 * mrg - BORDER - 0.012) / 0.55)
        if rnd:
            Rb = gh - BORDER - 0.006
            glyphs, total = [], 0.0
            for ch in p.maker.strip():
                if ch == " ":
                    glyphs.append((None, tsize * 0.55))
                    total += tsize * 0.55
                    continue
                g = text("glyph", ch, size=tsize, depth=0.003, rot=(0, 0, 0), font="sans")
                xs = [v.co.x for v in g.data.vertices]
                w = (max(xs) - min(xs)) if xs else tsize * 0.5
                glyphs.append((g, w + tsize * 0.22))
                total += w + tsize * 0.22
            phi = radians(-45.0) - 0.5 * total / Rb
            for g, w in glyphs:
                phi_c = phi + 0.5 * w / Rb
                if g is not None:
                    _drop_under(g)
                    place(g, loc=(Rb * cos(phi_c), Rb * sin(phi_c), tz), rot=(0, 0, degrees(phi_c) + 90.0))
                    letters.append(g)
                phi += w / Rb
        else:
            g = text("maker", p.maker.strip(), size=tsize, depth=0.003, loc=(gh * 0.52, -gh + BORDER + 0.006, tz),
                     rot=(0, 0, 0), font="sans", spacing=1.15)
            _drop_under(g)
            xs = [v.co.x for v in g.data.vertices]
            w = (max(xs) - min(xs)) if xs else 0.0
            if w > gh * 0.62:
                place(g, scale=(gh * 0.62 / w, 1.0, 1.0))
            letters.append(g)
        for g in letters:
            assign(g, iron_hi)

    # ------------------------------------------------------------------ angle-iron lip the segments sit in
    if p.kerb:
        hin = half - kw
        frame = _loft("frame", [_loop(shape, gh + 0.0025, 0.004, 0.022), _loop(shape, gh + 0.003, 0.0095, 0.022),
                                _loop(shape, hin - 0.0015, 0.0095, 0.008), _loop(shape, hin - 0.0003, 0.0068, 0.007)], closed=True)
    else:                                     # bare frame: a thin lip chamfered down to the cobbles
        frame = _loft("frame", [_loop(shape, gh + 0.0025, 0.003, 0.022), _loop(shape, gh + 0.003, 0.008, 0.022),
                                _loop(shape, gh + 0.011, 0.008, 0.024), _loop(shape, half - 0.004, 0.0018, 0.016),
                                _loop(shape, half, 0.0, 0.014)], closed=True)
    smooth(frame, 25)
    assign(frame, iron_g)
    stat["grate"] = _tris(segs + letters + [frame])

    # ------------------------------------------------------------------ low stone kerb: cut stones on a mortar bed
    stones = []
    if p.kerb:
        r_k = rng.fork("kerb")
        ws = kw - FIL
        kh = 0.013
        sec = [(0.0, 0.007), (0.004, kh), (0.5 * ws, kh + 0.001), (ws - 0.022, kh - 0.0005), (ws - 0.011, kh - 0.0035),
               (ws - 0.004, kh - 0.008), (ws - 0.0005, 0.0035)]
        mortar = _loft("bed", [_loop(shape, hin + 0.0008, 0.006, 0.006), _loop(shape, half - FIL - 0.003, 0.006, 0.004),
                               _loop(shape, half - FIL + 0.003, 0.0045, 0.008), _loop(shape, half, 0.0, 0.016)], closed=True)
        smooth(mortar, 25)
        assign(mortar, mat("mortar", "#8f887a", grime=0.8))
        paint(mortar, grime=0.4, special=0.25 + 0.4 * age)

        def make_stone(name, fr, s0, s1, nint, unit):
            L = (s1 - s0) * unit
            e = min(0.007, 0.2 * L) / unit
            ts = [s0, s0 + e] + [s0 + e + (s1 - s0 - 2 * e) * i / nint for i in range(1, nint)] + [s1 - e, s1]
            dz, td, tl = r_k.u(-0.001, 0.0012), r_k.u(-0.0012, 0.0012), r_k.u(-0.001, 0.001)
            chip = int(r_k.u(2, len(ts) - 2)) if r_k.chance(0.25 + 0.4 * age) else -1
            mossy = r_k.u(0.2, 1.0)
            rings, gm, sm, wm_ = [], [], [], []
            for k, (d, z) in enumerate(sec):
                ring = []
                for j, s in enumerate(ts):
                    t = (s - s0) / (s1 - s0)
                    endp = j in (0, len(ts) - 1)
                    zz = 0.0045 + (z - 0.0045) * 0.55 if (endp and z > 0.0045) else z
                    zz += dz + td * (d / ws - 0.5) + tl * (2 * t - 1)
                    dd = d
                    if 1 <= k <= 4:
                        zz += r_k.u(-0.0005, 0.0005)
                    if j == chip and k in (3, 4, 5):
                        zz -= r_k.u(0.0015, 0.003)
                        dd -= 0.003
                    ring.append(fr(s, dd, max(0.0012, zz)))
                    ep = clamp(1.0 - min(t, 1.0 - t) * L / 0.035)
                    base = 1.0 if k >= 5 else (0.6 if k == 0 else 0.0)
                    gm.append(min(0.42, 0.34 * ep + 0.3 * base))
                    sm.append(clamp((0.15 + 0.6 * age) * mossy * (0.7 * ep + 0.6 * base)))
                    wm_.append(0.3 if (k in (2, 3) and not endp) else 0.0)
                rings.append(ring)
            b = _loft(name, rings, caps=True)
            smooth(b, 35)
            assign(b, stone)
            ga, sa_, wa = np.array(gm), np.array(sm), np.array(wm_)
            paint(b, grime=lambda co, n: ga if len(ga) == len(co) else 0.2 + 0 * co[:, 0],
                  special=lambda co, n: sa_ if len(sa_) == len(co) else 0.2 + 0 * co[:, 0],
                  wear=lambda co, n: wa if len(wa) == len(co) else 0 * co[:, 0])
            stones.append(b)

        if rnd:
            rc = hin + ws / 2
            n = max(8, int(round(2 * pi * rc / 0.34)))
            a_off = r_k.u(0, 2 * pi)
            cuts = [a_off + (i + r_k.u(-0.14, 0.14)) * 2 * pi / n for i in range(n)]
            cuts.append(cuts[0] + 2 * pi)
            gaps = [r_k.u(0.004, 0.011) for _ in range(n)]
            gaps.append(gaps[0])
            for i in range(n):
                make_stone(f"kerb{i}", lambda s, d, z: ((hin + d) * cos(s), (hin + d) * sin(s), z),
                           cuts[i] + 0.5 * gaps[i] / rc, cuts[i + 1] - 0.5 * gaps[i + 1] / rc, 6, rc)
        else:
            sa0, sb0 = -(half - FIL), hin            # each run starts at its outer corner and butts the next run
            m = max(2, int(round((sb0 - sa0) / 0.48)))
            for k in range(4):
                ca, sa = cos(radians(k * 90.0)), sin(radians(k * 90.0))
                cuts = [0.0] + sorted(clamp((i + r_k.u(-0.22, 0.22)) / m, 0.12, 0.88) for i in range(1, m)) + [1.0]
                gaps = [0.0] + [r_k.u(0.004, 0.011) for _ in range(m - 1)] + [2 * r_k.u(0.003, 0.007)]
                for i in range(m):
                    make_stone(f"kerb{k}_{i}",
                               lambda s, d, z, ca=ca, sa=sa: (s * ca + (hin + d) * sa, s * sa - (hin + d) * ca, z),
                               lerp(sa0, sb0, cuts[i]) + 0.5 * gaps[i], lerp(sa0, sb0, cuts[i + 1]) - 0.5 * gaps[i + 1], 2, 1.0)
        stat["kerb"] = _tris(stones + [mortar])

    # ------------------------------------------------------------------ the pit: packed dirt under the casting, fill in the hole
    r_s = rng.fork("soil")
    ul = _loop(shape, gh + 0.004, 0.0026, 0.022)
    under = mesh_object("under", ul, [tuple(range(len(ul)))])
    assign(under, dirt)
    paint(under, grime=0.85)
    packed = []
    for k in range(4):                          # dirt packed into stretches of the seams
        a = radians(k * 90.0)
        s0, s1 = rhub + 0.012, gh - 0.004
        for q in range(2):
            if r_s.chance(0.35 + 0.5 * age):
                L = (s1 - s0) * r_s.u(0.18, 0.42)
                c = r_s.u(s0 + L / 2, s1 - L / 2)
                packed.append(box(f"seamdirt{k}{q}", (L, 2 * SEAM + 0.0012, 0.003), loc=(c * cos(a), c * sin(a), ZB + ZW - 0.0018),
                                  rot=(0, 0, k * 90.0)))
    for o in packed:
        assign(o, dirt)
        paint(o, grime=0.6)

    def fill_z(r):
        return 0.0028 + 0.0095 * (1.0 - smoothstep(0.12 * rh, 0.95 * rh, r))

    ph = r_s.u(0, 6.28)

    def fill_fn(u, v):
        a = -2 * pi * u
        r = v ** 0.85 * (rh + 0.004)
        return (r * cos(a), r * sin(a), fill_z(r) * (0.85 + 0.15 * sin(3 * a + ph)))

    fill = surface("fill", fill_fn, 20, 6, closed_u=True)
    realize(fill)
    displace_noise(fill, strength=0.003, scale=0.035, seed=p.seed, along_normal=False, axis_mask=(0, 0, 1))
    transform_verts(fill, lambda co: (co.x, co.y, max(0.0014, co.z)))
    uv_planar(fill, (1, 0, 0), (0, 1, 0))
    assign(fill, mat(p.fill))
    paint(fill, grime=lambda co, n: np.clip((np.hypot(co[:, 0], co[:, 1]) - rh * 0.72) / (0.28 * rh), 0.0, 1.0) * 0.55)
    # crumbs, bark bits and stones: real relief that catches the lantern light
    ring1 = [(0.5 * cos(2 * pi * i / 5 + 0.3), 0.5 * sin(2 * pi * i / 5 + 0.3) * (0.8 if i == 2 else 1.0), 0.1 + 0.08 * (i % 2)) for i in range(5)]
    ring0 = [(0.42 * cos(2 * pi * i / 5), 0.42 * sin(2 * pi * i / 5), -0.3) for i in range(5)]
    blob = mesh_object("blob", [(0.06, -0.04, 0.5)] + ring1 + ring0,
                       [(0, 1 + i, 1 + (i + 1) % 5) for i in range(5)] + [(1 + i, 6 + i, 6 + (i + 1) % 5, 1 + (i + 1) % 5) for i in range(5)])
    smooth(blob, 70)
    if p.fill == "gravel":
        sets = [(mat(p.fill), 20, 0.010, 0.024, 0.1), (stone, 30, 0.009, 0.024, 0.2)]
    else:
        sets = [(mat(p.fill), 20, 0.012, 0.032, 0.15), (stone, 9, 0.008, 0.02, 0.3)]
    pebbles = []
    for si, (m_, cnt, d0, d1, gr) in enumerate(sets):
        assign(blob, m_)
        paint(blob, grime=gr, mode="set")
        pm = []
        for i in range(cnt):
            a, r = r_s.u(0, 2 * pi), (rh - 0.012) * r_s.u(0.02, 1.0) ** 0.5
            d = r_s.u(d0, d1)
            pm.append(trs((r * cos(a), r * sin(a), fill_z(r) + 0.12 * d), (r_s.u(-18, 18), r_s.u(-18, 18), r_s.u(0, 360)),
                          (d * r_s.u(0.8, 1.3), d * r_s.u(0.7, 1.1), d * r_s.u(0.45, 0.8))))
        pebbles.append(instances_to_mesh(f"pebbles{si}", blob, pm, keep_base=True))
    delete([blob])
    socket("tree", "attach", (0, 0, 0.011), role="tree")
    stat["pit"] = _tris([under, fill] + packed + pebbles)

    # ------------------------------------------------------------------ weathering of the casting
    rusty = p.grate_finish == "rusty"
    n_a = noise_mask(0.17, seed=p.seed + 1, lo=0.2, hi=0.75)
    n_b = noise_mask(0.06, seed=p.seed + 2, lo=0.3, hi=0.8)
    n_c = noise_mask(0.28, seed=p.seed + 3, lo=0.3, hi=0.7)
    rot4 = [(cos(radians(k * 90.0)), sin(radians(k * 90.0))) for k in range(4)]
    bolts_w = np.array([(x * c - y * s, x * s + y * c) for (x, y) in bolts_xy for (c, s) in rot4])
    ends = [(r * cos(a), r * sin(a)) for (a, r0, r1) in slot_ends for r in (r0, r1)]
    ends_w = np.array([(x * c - y * s, x * s + y * c) for (x, y) in ends for (c, s) in rot4]) if ends else np.zeros((1, 2)) + 99.0
    z_mid = ZB + 0.5 * (ZW + ZR)

    def _geo(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        r = np.hypot(x, y)
        raised = ((z > z_mid) & (n[:, 2] > 0.4)).astype(np.float64)
        seam_d = np.minimum(np.abs(x), np.abs(y))
        return x, y, z, r, raised, seam_d

    def _dmin(co, pts):
        d = np.hypot(co[:, 0][:, None] - pts[:, 0][None, :], co[:, 1][:, None] - pts[:, 1][None, :])
        return d.min(axis=1)

    def m_wear(co, n):                         # feet polish the raised borders and ribs to bare metal
        x, y, z, r, raised, seam_d = _geo(co, n)
        na = n_a(co, n)
        on_hub = (r < rh + hubf + 0.004).astype(np.float64)
        w = raised * (1.0 - on_hub) * 0.75 * (1.0 - na) * (0.5 + 0.5 * age)   # pale raised metal: untrodden stretches stay dark
        return np.maximum(w, np.clip(n[:, 2], 0.0, 1.0) * 0.1 * na)

    def m_rust(co, n):                         # rust lives in the seams, round the bolts, the hole and the outer edge
        x, y, z, r, raised, seam_d = _geo(co, n)
        nb, nc = n_b(co, n), n_c(co, n)
        seam = np.clip(1.0 - (seam_d - SEAM) / 0.024, 0.0, 1.0)
        hole = np.clip(1.0 - (r - rh) / 0.04, 0.0, 1.0)
        de = (gh - r) if rnd else (gh - np.maximum(np.abs(x), np.abs(y)))
        edge = np.clip(1.0 - de / 0.022, 0.0, 1.0)
        bolt = np.clip(1.0 - _dmin(co, bolts_w) / 0.032, 0.0, 1.0)
        loc_ = np.maximum.reduce([seam * (0.45 + 0.55 * nb), bolt * 0.9, hole * (0.5 + 0.5 * nb), edge * 0.75 * nb])
        out = loc_ * (0.4 + 0.6 * age)
        gen = (0.12 + 0.6 * nc * (0.35 + 0.65 * nb)) if rusty else 0.3 * age * nc * nb
        return np.maximum(out, gen) * (1.0 - 0.6 * raised)

    def m_grime(co, n):                        # dirt packed in the slot ends, the seams and down the slot walls
        x, y, z, r, raised, seam_d = _geo(co, n)
        low = (z < ZB + 0.002).astype(np.float64)
        e = np.clip(1.0 - _dmin(co, ends_w) / 0.013, 0.0, 1.0) * 0.8
        sg = np.clip(1.0 - (seam_d - SEAM) / 0.007, 0.0, 1.0) * 0.65
        return np.maximum.reduce([low, e * (1.0 - raised), sg * (1.0 - raised), 0.12 * n_c(co, n) * (1.0 - raised)])

    for s in segs + letters:
        paint(s, wear=m_wear, special=m_rust, grime=m_grime)
    paint(frame, special=mask_mul(noise_mask(0.09, seed=p.seed + 3, lo=0.4, hi=0.8), 0.2 + 0.4 * age),
          wear=mask_mul(facing((0, 0, 1), 2.0), n_a, 0.5), grime=0.25)

    # ------------------------------------------------------------------ fallen leaves: dry, curled, caught where the wind leaves them
    nl = int(p.leaves)
    LW, LL = 0.023, 0.094
    if nl > 0:
        r_l = rng.fork("leaves")
        rows = [(0.0, 0.06), (0.12, 0.58), (0.28, 1.0), (0.46, 0.9), (0.66, 0.6), (0.85, 0.27), (1.0, 0.0)]

        def leaf_fn(u, v):
            vv, wf = rows[min(6, int(round(v * 6)))]
            x = (u - 0.5) * 2 * LW * wf
            return (x, (vv - 0.5) * LL, 0.34 * abs(x) + 0.011 * (2 * (vv - 0.5)) ** 2 + 0.3 * x * (vv - 0.5))

        leaf = surface("leaf", leaf_fn, 2, 6, uv_len=(0.25, 0.25))
        realize(leaf)
        _two_sided(leaf)
        assign(leaf, mat("leaf", "#5c4a35" if age > 0.3 else "#6b6038", rough=2.0))
        wind = r_l.u(0, 2 * pi)
        lm = [[], []]
        for i in range(nl):
            kind = r_l.u()
            sc = r_l.u(0.75, 1.2)
            if kind < 0.3 and slot_ends:        # wedged on edge in a slot
                a, r0, r1 = r_l.pick(slot_ends)
                a += radians(90.0 * (int(r_l.u(0, 4)) % 4))
                r = lerp(r0, r1, r_l.u(0.3, 0.7))
                tilt = r_l.u(38, 66) * (1 if r_l.chance(0.5) else -1)
                M = trs((r * cos(a), r * sin(a), ZB + 0.0015 + LW * sc * abs(sin(radians(tilt))) * 0.92),
                        (0, tilt, degrees(a) - 90.0 + r_l.u(-14, 14)), sc)
            elif kind < 0.66:                   # blown against the kerb / frame lip on the lee side
                a = wind + r_l.u(-1, 1) * r_l.u(0.15, 1.25)
                r = r_edge(a) - r_l.u(-0.012, 0.05)
                over = p.kerb and r > r_edge(a) - 0.012
                M = trs((r * cos(a), r * sin(a), ZB + ZR + (0.006 if over else 0.002) + r_l.u(0, 0.005)),
                        (r_l.u(-9, 9), r_l.u(-9, 9) + (12 if over else 0), degrees(a) + r_l.u(-50, 50)), sc)
            elif kind < 0.82:                   # lodged against the hub collar / guard feet
                a = wind + pi + r_l.u(-1.3, 1.3)
                r = rhub + r_l.u(-0.012, 0.02)
                M = trs((r * cos(a), r * sin(a), ZB + ZR + 0.005), (r_l.u(-8, 8), r_l.u(8, 20), degrees(a) + 180 + r_l.u(-40, 40)), sc)
            else:                               # lying on the plate
                a = r_l.u(0, 2 * pi)
                r = lerp(rhub + 0.05, r_edge(a) - 0.06, r_l.u(0, 1))
                M = trs((r * cos(a), r * sin(a), ZB + ZR + 0.0025), (r_l.u(-6, 6), r_l.u(-6, 6), r_l.u(0, 360)), sc)
            lm[0 if r_l.chance(0.55) else 1].append(M)
        lv = []
        for k in range(2):
            if lm[k]:
                paint(leaf, grime=(0.0, 0.55)[k], mode="set")
                lv.append(instances_to_mesh(f"leaves{k}", leaf, lm[k], uv_jitter=False, keep_base=True))
        delete([leaf])
        stat["leaves"] = _tris(lv)

    # ------------------------------------------------------------------ grass tufts in the hole and through slots
    nt = int(round(p.weeds * 9))
    if nt > 0 and slot_ends:
        r_w = rng.fork("weeds")
        blade = surface("blade", lambda u, v: ((u - 0.5) * 0.008 * (1.0 - v ** 1.6), 0.045 * v * v, 0.11 * v), 1, 3, uv_len=(0.25, 0.25))
        realize(blade)
        _two_sided(blade, 0.0003)
        assign(blade, mat("leaf", "#93a35e", rough=1.6))
        wm = []
        for i in range(nt):
            if i % 3 == 0:
                a = r_w.u(0, 2 * pi)
                r = rh - r_w.u(0.012, 0.05)
                cx, cy, cz = r * cos(a), r * sin(a), fill_z(r) - 0.001
                spread, tall = 0.012, r_w.u(0.7, 1.1)
            else:
                a, r0, r1 = r_w.pick(slot_ends)
                a += radians(90.0 * (int(r_w.u(0, 4)) % 4))
                r = lerp(r0, r1, r_w.u(0.1, 0.9))
                cx, cy, cz = r * cos(a), r * sin(a), ZB + 0.001
                spread, tall = 0.004, r_w.u(0.6, 1.1)
            for b in range(int(r_w.u(4, 7))):
                wm.append(trs((cx + r_w.u(-spread, spread), cy + r_w.u(-spread, spread), cz),
                              (r_w.u(-12, 12), r_w.u(-12, 12), r_w.u(0, 360)), tall * r_w.u(0.6, 1.2)))
        stat["weeds"] = _tris([instances_to_mesh("weeds", blade, wm, uv_jitter=False, keep_base=False)])

    if p.guard:
        stat["guard"] = _guard(p, rng.fork("guard"), ZB + ZH, rh, iron, gilt)


def _guard(p, rng, zt, rh, iron, gilt):
    n = int(p.bars)
    H = p.guard_height
    age = p.age
    style = p.guard_top
    bw, bth = 0.030, 0.008                     # flat bar 30 x 8 mm, edges well rounded
    Rg = rh + 0.020
    fh = clamp(0.2 * H, 0.2, 0.36)
    fo = p.flare * (0.7 if style == "spear" else 1.0)
    z0 = zt + bth / 2 + 0.0002
    bend = 0.018

    # bar angles: keep the feet off the segment seams
    best, off = -1.0, 0.5
    for c in range(20):
        o = c / 20.0
        d = min(min(((i + o) * 360.0 / n) % 90.0, 90.0 - ((i + o) * 360.0 / n) % 90.0) for i in range(n))
        if d > best + 1e-6:
            best, off = d, o
    angs = [radians((i + off) * 360.0 / n) for i in range(n)]
    hurt = int(rng.u(0, n)) % n
    bar_prof = rounded_rect(bth, bw, 0.0032, 1)

    bars, tips_z = [], []
    for i, a in enumerate(angs):
        Hi = H - ((0.085 + (0.075 if i % 2 else 0.0)) if style == "spear" else 0.0) + rng.u(-0.004, 0.004)
        zf = zt + Hi - fh
        rz = [(Rg + 0.054, z0), (Rg + bend, z0)]
        for k in range(1, 4):
            b = radians(-90.0 - 90.0 * k / 3)
            rz.append((Rg + bend + bend * cos(b), z0 + bend + bend * sin(b)))
        nz = max(2, int(round((zf - z0) / 0.5)))
        lean = rng.u(-1, 1) * 0.003 * (0.5 + age)
        for k in range(1, nz):
            t = k / nz
            dent = -0.016 * age * sin(pi * t) ** 3 if i == hurt else 0.0
            rz.append((Rg + lean * sin(pi * t) + dent + rng.u(-0.0008, 0.0008), lerp(z0 + bend, zf, t)))
        fl = bezier_path((Rg, 0, zf), (Rg, 0, zf + 0.5 * fh), (Rg + 0.4 * fo, 0, zf + 0.82 * fh), (Rg + fo, 0, zf + fh), n=5)
        rz += [(q[0], q[2]) for q in fl]
        # tip
        tr, tzz = rz[-1][0] - rz[-2][0], rz[-1][1] - rz[-2][1]
        tl = hypot(tr, tzz)
        tr, tzz = tr / tl, tzz / tl
        body_len = sum(hypot(rz[k + 1][0] - rz[k][0], rz[k + 1][1] - rz[k][1]) for k in range(len(rz) - 1))
        if style == "spear":
            for d in (0.03, 0.065, 0.1, 0.13, 0.15):
                rz.append((fl[-1][0] + tr * d, fl[-1][2] + tzz * d))
            tip_len = 0.15
        else:
            rho0 = 0.0135 if style == "flare" else 0.036      # rolled tip ~ 35 mm across / open scroll
            sweep_deg = 285.0 if style == "flare" else 410.0
            steps = 6 if style == "flare" else 9
            nrm = (tzz, -tr)                   # clockwise normal: curls outwards and down
            px, pz = rz[-1]
            tip_len = 0.0
            ang0 = atan2(-nrm[1], -nrm[0])
            cx, cz = px + rho0 * nrm[0], pz + rho0 * nrm[1]
            prev = (px, pz)
            for k in range(1, steps + 1):
                th = radians(sweep_deg) * k / steps
                rho = rho0 * (1.0 - (0.0 if style == "flare" else 0.6) * k / steps)
                q = (cx + rho * cos(ang0 - th), cz + rho * sin(ang0 - th))
                tip_len += hypot(q[0] - prev[0], q[1] - prev[1])
                prev = q
                rz.append(q)
        tot = body_len + tip_len
        t_tip = body_len / tot

        def sc(t, t_tip=t_tip, style=style):
            if t <= t_tip:
                return 1.0
            s = (t - t_tip) / max(1e-6, 1.0 - t_tip)
            if style == "spear":
                return 1.0 + 0.75 * sin(min(1.0, s / 0.35) * pi / 2) if s < 0.35 else max(0.08, 1.75 * (1.0 - (s - 0.35) / 0.65) ** 0.8)
            return 1.0 - (0.3 if style == "flare" else 0.45) * s

        ca, sa = cos(a), sin(a)
        path = [(r * ca, r * sa, z) for (r, z) in rz]
        bar = sweep(f"bar{i}", path, bar_prof, smooth_path=False, up=(-sa, ca, 0), scale_fn=sc)
        smooth(bar, 52)
        assign(bar, iron)
        ztip = fl[-1][2]
        tips_z.append(ztip)
        if p.gilt_tips:
            lim = ztip - (0.012 if style == "spear" else 0.04)
            rlim = Rg + fo * 0.8
            assign_faces(bar, gilt, lambda c, nn, lim=lim, rlim=rlim: c.z > lim and hypot(c.x, c.y) > rlim)
        bars.append(bar)

    # two hoops of 30 x 6 flat, each in two lugged halves bolted together
    Rh = Rg + bth / 2 + 0.0032
    zh = [zt + clamp(0.11 * H, 0.14, 0.22), zt + H - fh - 0.05]
    mids = [radians((i + off + 0.5) * 360.0 / n) for i in range(n)]
    j0, j1 = mids[0], mids[n // 2]
    hoop_prof = rounded_rect(0.006, 0.030, 0.0024, 1)
    lug, fr = 0.032, 0.008
    hoops = []
    for hz in zh:
        for (A0, A1) in ((j0, j1), (j1, j0 + 2 * pi)):
            d = 0.0045 / Rh
            a0, a1 = A0 + d, A1 - d
            pts = [((Rh + lug) * cos(a0), (Rh + lug) * sin(a0), hz), ((Rh + fr) * cos(a0), (Rh + fr) * sin(a0), hz)]
            am = a0 + 0.3 * fr / Rh
            pts.append(((Rh + 0.25 * fr) * cos(am), (Rh + 0.25 * fr) * sin(am), hz))
            m = max(8, int(round((a1 - a0) * Rh / 0.068)))
            for k in range(m + 1):
                aa = lerp(a0 + fr / Rh, a1 - fr / Rh, k / m)
                pts.append((Rh * cos(aa), Rh * sin(aa), hz))
            am = a1 - 0.3 * fr / Rh
            pts.append(((Rh + 0.25 * fr) * cos(am), (Rh + 0.25 * fr) * sin(am), hz))
            pts += [((Rh + fr) * cos(a1), (Rh + fr) * sin(a1), hz), ((Rh + lug) * cos(a1), (Rh + lug) * sin(a1), hz)]
            h = sweep("hoop", pts, hoop_prof, smooth_path=False, up=(0, 0, 1))
            smooth(h, 50)
            assign(h, iron)
            hoops.append(h)

    # ornament: a ring of 9 mm round stock between every pair of bars, held to the bars by forged collars
    orn = []
    rst = 0.0045
    rr = Rg * sin(pi / n) - bw / 2 - rst + 0.0005
    if p.ornament == "rings" and rr > 0.012:
        zr_ = zh[1] - 0.015 - rr - rst - 0.006
        rb = torus("orn_ring", major=rr, minor=rst, major_seg=12, minor_seg=4)
        smooth(rb, 60)
        rc = Rg * cos(pi / n)
        om = [trs((rc * cos(am), rc * sin(am), zr_), (0, 90, degrees(am))) for am in mids]
        orn.append(instances_to_mesh("orn_rings", rb, om, keep_base=False))
        col = box("orn_collar", (0.017, bw + 0.02, 0.013), bevel=0.0)
        cmx = [trs((Rg * cos(a), Rg * sin(a), zr_), (0, 0, degrees(a))) for a in angs]
        orn.append(instances_to_mesh("orn_collars", col, cmx, keep_base=False))
        for o in orn:
            assign(o, iron)

    # fixings: clamp bolts through the lugs, a rivet at every bar / hoop crossing, a bolt through every foot
    shank = cylinder("clamp_shank", 0.004, 0.034, verts=6, cap=False)
    h1 = cylinder("clamp_head", 0.0085, 0.006, verts=6, loc=(0, 0, 0.0105))
    h2 = cylinder("clamp_nut", 0.0085, 0.007, verts=6, loc=(0, 0, -0.011))
    clamp_b = join([shank, h1, h2], "clamp_bolt")
    flat(clamp_b)
    cm = [trs(((Rh + 0.021) * cos(A), (Rh + 0.021) * sin(A), hz), (90, 0, degrees(A))) for hz in zh for A in (j0, j1)]
    fix = [instances_to_mesh("clamp_bolts", clamp_b, cm, keep_base=False)]
    riv = lathe("rivet", [(0.0085, 0.0), (0.0066, 0.0038), (0.0, 0.0058)], segments=5)
    rm = [trs(((Rh + 0.0028) * cos(a), (Rh + 0.0028) * sin(a), hz), (0, 90, degrees(a))) for hz in zh for a in angs]
    fix.append(instances_to_mesh("rivets", riv, rm, keep_base=False))
    fbolt = cylinder("foot_bolt", 0.0095, 0.006, verts=6, loc=(0, 0, 0.003))
    apply_transform(fbolt)
    _drop_under(fbolt)
    flat(fbolt)
    fm = [trs(((Rg + 0.036) * cos(a), (Rg + 0.036) * sin(a), z0 + bth / 2 - 0.0003), (0, 0, degrees(a) + rng.u(0, 60))) for a in angs]
    fix.append(instances_to_mesh("foot_bolts", fbolt, fm, keep_base=False))
    for o in fix:
        assign(o, iron)

    # ageing: rust creeps up from the feet and sits under the hoops; hands and leaned bicycles chip the paint at ~1 m
    zl = zt
    for o in bars + hoops + fix + orn:
        paint(o,
              special=mask_mul(lambda co, nn: np.clip(1.0 - (co[:, 2] - zl) / 0.2, 0.0, 1.0) * (0.15 + 0.4 * age),
                               noise_mask(0.05, seed=p.seed + 7, lo=0.2, hi=0.8)),
              wear=mask_mul(height_ramp(zl + 0.95, zl + 1.3), height_ramp(zl + 0.95, zl + 0.6), noise_mask(0.07, seed=p.seed + 8, lo=0.45, hi=0.75), 0.2 + 0.5 * age),
              grime=mask_mul(height_ramp(zl, zl + 0.12), 0.45))
    for o in hoops:
        paint(o, special=mask_mul(noise_mask(0.06, seed=p.seed + 9, lo=0.5, hi=0.8), 0.15 + 0.5 * age))
    socket("guard_top", "attach", (0, 0, max(tips_z)), role="prop_top")
    return _tris(bars + hoops + fix + orn)
