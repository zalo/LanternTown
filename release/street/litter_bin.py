"""Victorian cast-iron litter bin. A moulded base ring on cast scroll feet carries the drum: either a
tapering basket of riveted cast slats held by two hoops, or a polygonal drum of pierced, fielded cast
panels between corner colonnettes with a strap-hinged, locked front door. A turned lid ring with raised
lettering sits on top (hinged and padlocked on the slatted bin) and either flares into an open mouth or
carries a ribbed dome hood on turned balusters. A galvanised zinc liner with swaged ribs and bail
handles stands inside and shows through the slats / piercings and the mouth; the front wears the town
crest (crowned shield).
"""
import bmesh
import numpy as np
from tk import *

META = {"name": "Cast-iron litter bin", "class": "prop", "mount": "ground", "focus": 0.62}
PARAMS = {
    "style": {"default": "slatted", "choices": ["slatted", "panelled"], "desc": "drum construction: riveted slat basket or pierced cast panels with a door"},
    "top": {"default": "hooded", "choices": ["hooded", "open"], "desc": "dome hood on balusters, or an open flared mouth"},
    "height": {"default": 0.78, "min": 0.6, "max": 1.0, "desc": "height of the lid ring above the paving (m); the hood adds ~0.35"},
    "diameter": {"default": 0.46, "min": 0.36, "max": 0.6, "desc": "drum diameter at the lid ring (m)"},
    "slats": {"default": 24, "min": 14, "max": 32, "desc": "number of slats (slatted style)"},
    "panels": {"default": 8, "min": 6, "max": 10, "desc": "number of cast panels (panelled style)"},
    "feet": {"default": 4, "choices": [3, 4], "desc": "number of cast feet (and hood balusters)"},
    "paint": {"default": "#263d2e", "desc": "paint colour of the ironwork"},
    "gilt": {"default": "rich", "choices": ["none", "crest", "rich"], "desc": "gilt details: none, crest + lettering, or also finial / beads / rosettes"},
    "crest_colour": {"default": "#14264a", "desc": "enamel colour of the crest shield field"},
    "lettering": {"default": "LITTER", "desc": "raised lettering on the lid ring frieze (empty = none)"},
    "fill": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "how full the liner is (0 = empty, 1 = overflowing)"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "age: paint chips, rust at the base and joints"},
}
VARIANTS = {
    "panelled_hooded": {"style": "panelled", "paint": "#22252c", "gilt": "rich", "height": 0.82, "diameter": 0.5,
                        "crest_colour": "#15402a", "fill": 0.3, "wear": 0.2, "seed": 4},
    "open_slatted_red": {"top": "open", "paint": "#6a1a18", "crest_colour": "#141a16", "slats": 18, "height": 0.7,
                         "diameter": 0.5, "feet": 3, "gilt": "none", "fill": 1.0, "wear": 0.8, "seed": 11},
    "hex_open_blue": {"style": "panelled", "top": "open", "panels": 6, "paint": "#1f3859", "gilt": "crest", "crest_colour": "#101216",
                      "height": 0.66, "diameter": 0.56, "fill": 0.7, "wear": 0.5, "lettering": "RUBBISH", "seed": 23},
    "tall_narrow": {"height": 0.96, "diameter": 0.38, "slats": 16, "feet": 3, "paint": "#1f4d4a", "gilt": "crest",
                    "crest_colour": "#141a16", "fill": 0.1, "wear": 0.1, "lettering": "KEEP TIDY", "seed": 31},
}


def _M(loc=(0, 0, 0), rot=(0, 0, 0)):
    e = Euler((radians(rot[0]), radians(rot[1]), radians(rot[2])), 'XYZ')
    return Matrix.Translation(Vector(loc)) @ e.to_matrix().to_4x4()


def _ring(a, r, z=0.0, rot=(0, 0, 0)):
    """Matrix taking something modelled at the origin facing -Y out to radius r at bearing a (deg, 0 = front)."""
    return Matrix.Rotation(radians(a), 4, 'Z') @ _M((0.0, -r, z), rot)


def _put(objs, M):
    for o in objs:
        apply_transform(o)
        transform_verts(o, lambda co, M=M: M @ co)
    return objs


def _noback(o, simplify=0.0):
    """Bake the transform, drop the hidden faces lying against the surface behind (normal +Y) and optionally
    dissolve needless curve vertices (font outlines)."""
    apply_transform(o)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    back = [f for f in bm.faces if f.normal.y > 0.95]
    if back:
        bmesh.ops.delete(bm, geom=back, context='FACES')
    if simplify > 0:
        bmesh.ops.dissolve_limit(bm, angle_limit=radians(simplify), verts=bm.verts[:], edges=bm.edges[:])
        big = [f for f in bm.faces if len(f.verts) > 4]
        if big:
            bmesh.ops.triangulate(bm, faces=big)
    bm.to_mesh(o.data)
    bm.free()
    return o


def _shield(w, h, k=1.0, n=6):
    side = [(w / 2 * sin(i / n * pi / 2), -h / 2 + 0.62 * h * (i / n) ** 1.25) for i in range(1, n + 1)]
    pts = [(0.0, -h / 2)] + side + [(w / 2, h / 2), (w / 4, h * 0.455), (0.0, h / 2), (-w / 4, h * 0.455), (-w / 2, h / 2)]
    pts += [(-x, y) for x, y in reversed(side)]
    return [(x * k, y * k) for x, y in pts]


def _crown(cw, ch, cz):
    raw = [(-.5, -.3), (.5, -.3), (.5, 0), (.62, .8), (.39, .4), (.28, .82), (.14, .4), (0, 1.0), (-.14, .4), (-.28, .82),
           (-.39, .4), (-.62, .8), (-.5, 0)]
    return [(x * cw, cz + y * ch) for x, y in raw]


def _star(r, cz, n=5):
    return [((r if i % 2 == 0 else r * 0.42) * sin(i * pi / n), cz + (r if i % 2 == 0 else r * 0.42) * cos(i * pi / n))
            for i in range(2 * n)][::-1]


def build(p, rng):
    slatted = p.style == "slatted"
    hooded = p.top == "hooded"
    R = p.diameter / 2
    H = p.height
    wear = p.wear
    sd = int(p.seed)
    iron = mat("iron_painted", p.paint, scale=5.0, rough=1.35)
    gold = mat("metal_plain", "#dcb04e", rough=0.5)
    zinc = mat("zinc")
    brass = mat("brass")
    wrought = mat("iron_wrought")
    enamel = mat("iron_painted", p.crest_colour, scale=5.0)
    g_crest = gold if p.gilt in ("crest", "rich") else iron
    g_rich = gold if p.gilt == "rich" else iron

    irons, gilts, riches = [], [], []          # painted iron / crest-level gilt / rich-level gilt

    zf = 0.078                                 # underside of the base ring (foot clearance)
    zb = zf + 0.075                            # top of the base ring = foot of the drum wall
    zt = H - 0.09                              # top of the drum wall = underside of the lid ring
    nfeet = int(p.feet)
    foot_bear = [45 + k * 90 for k in range(4)] if nfeet == 4 else [60, 180, 300]

    N = int(p.panels)
    Rc = R - 0.008                             # panelled: circumradius of the polygon
    apo = Rc * cos(pi / N)
    fw = 2 * Rc * sin(pi / N)
    Rb = R * 0.84 if slatted else R            # wall radius at the base

    def rw(z):                                 # outer radius of the drum wall
        if not slatted:
            return R
        t = clamp((z - zb) / (zt - zb))
        return Rb + (R - Rb) * t ** 1.4

    # ------------------------------------------------------------------ feet + base ring
    fwid = 0.048
    fp = [(Rb - 0.055, zf + 0.004), (Rb - 0.055, zf - 0.010), (Rb - 0.03, zf - 0.016), (Rb - 0.008, zf - 0.03),
          (Rb + 0.002, 0.016), (Rb - 0.006, 0.006), (Rb - 0.004, 0.0), (Rb + 0.044, 0.0), (Rb + 0.053, 0.008),
          (Rb + 0.051, 0.02), (Rb + 0.038, 0.03), (Rb + 0.028, zf - 0.024), (Rb + 0.032, zf - 0.008), (Rb + 0.027, zf + 0.004)]
    foot = extrude_profile("foot", fp, fwid, bevel=0.004, segments=1)
    transform_verts(foot, lambda co: Vector((fwid / 2 - co.z, -co.x, co.y)))
    irons.append(instances_to_mesh("feet", foot, [_ring(a, 0.0) for a in foot_bear], keep_base=False))

    base = lathe("base_ring", [(0.0, zf + 0.008), (Rb - 0.06, zf + 0.008), (Rb - 0.05, zf), (Rb + 0.022, zf), (Rb + 0.027, zf + 0.005),
                               (Rb + 0.027, zf + 0.02), (Rb + 0.022, zf + 0.026), (Rb + 0.013, zf + 0.031), (Rb + 0.011, zf + 0.04),
                               (Rb + 0.0165, zf + 0.046), (Rb + 0.0185, zf + 0.052), (Rb + 0.0165, zf + 0.058), (Rb + 0.008, zf + 0.063),
                               (Rb + 0.007, zf + 0.075), (Rb - 0.04, zf + 0.075)], segments=36)
    irons.append(base)

    rivet = lathe("rivet", [(0.0062, 0.0), (0.0048, 0.0032), (0.0, 0.0048)], segments=6)
    rivet_m = []

    # ------------------------------------------------------------------ drum wall
    if slatted:
        ns = int(p.slats)
        pitch = 2 * pi * R / ns
        sw_, st_ = pitch * 0.6, 0.008
        prof = [(-sw_ / 2, -st_ / 2), (sw_ / 2, -st_ / 2), (sw_ / 2, st_ * 0.1), (sw_ * 0.2, st_ / 2), (-sw_ * 0.2, st_ / 2), (-sw_ / 2, st_ * 0.1)]
        zs = [lerp(zb - 0.004, zt + 0.012, i / 6) for i in range(7)]
        slat = sweep("slat", [(0.0, -(rw(z) - st_ / 2), z) for z in zs], prof, up=(0, -1, 0), cap=False,
                     scale_fn=lambda t: rw(lerp(zb, zt, t)) / R, smooth_path=False)
        r_sl = rng.fork("slats")
        sm = []
        for k in range(ns):
            a = (k + 0.5) * 360.0 / ns + r_sl.jitter(0.0, 0.5)
            sm.append(Matrix.Rotation(radians(a), 4, 'Z') @ _M((0, r_sl.jitter(0.0, 0.0012), 0), (0, 0, r_sl.jitter(0.0, 1.2))))
        irons.append(instances_to_mesh("slats", slat, sm, keep_base=False))

        z_mid = zb + 0.4 * (zt - zb)
        z_up = zt - 0.026
        for hz in (z_mid, z_up):
            hw = 0.02
            irons.append(lathe("hoop", [(rw(hz - hw) + 0.0005, hz - hw), (rw(hz - hw) + 0.006, hz - hw + 0.002), (rw(hz) + 0.006, hz - 0.005),
                                        (rw(hz) + 0.0095, hz), (rw(hz) + 0.006, hz + 0.005), (rw(hz + hw) + 0.006, hz + hw - 0.002),
                                        (rw(hz + hw) + 0.0005, hz + hw)], segments=36))
            for k in range(ns):
                ak = (k + 0.5) * 360.0 / ns
                if hz == z_up and (abs(ak - 90) < 9 or abs(ak - 270) < 6):
                    continue                   # under the hinge / hasp lugs
                rivet_m.append(_ring(ak, rw(hz - 0.012) + 0.0055, hz - 0.012, (90, 0, 0)))
        wall_in_b = Rb - 0.008
        Rin = R - 0.014
    else:
        hp = zt - zb
        pw = fw - 0.018
        rail_b, rail_m, rail_t = 0.04, 0.036, 0.04
        lf0 = zb + rail_b
        lf1 = lf0 + (hp - rail_b - rail_m - rail_t) * 0.56
        sl0, sl1 = lf1 + rail_m, zt - rail_t
        zm = lf1 + rail_m / 2
        n_sl = max(2, int((pw - 0.024) / 0.036))
        sp = (pw - 0.024) / n_sl
        holes = []
        for i in range(n_sl):
            cx = -(pw - 0.024) / 2 + sp * (i + 0.5)
            holes.append([(x + cx, y + (sl0 + sl1) / 2) for x, y in rounded_rect(sp * 0.5, sl1 - sl0, sp * 0.24, 2)])

        def panel(door):
            plate = shape_extrude("plate", [(-pw / 2, zb - 0.002), (pw / 2, zb - 0.002), (pw / 2, zt + 0.002), (-pw / 2, zt + 0.002)],
                                  holes, depth=0.008, loc=(0, 0.008, 0), rot=(90, 0, 0))
            lw, lh = pw - 0.016, lf1 - lf0
            frame = shape_extrude("frame", [(x, y + (lf0 + lf1) / 2) for x, y in rounded_rect(lw, lh, 0.006, 1)],
                                  [[(x, y + (lf0 + lf1) / 2) for x, y in rounded_rect(lw - 0.024, lh - 0.024, 0.004, 1)]],
                                  depth=0.005, bevel=0.0012, rot=(90, 0, 0))
            sframe = shape_extrude("sframe", [(x, y + (sl0 + sl1) / 2) for x, y in rounded_rect(lw, sl1 - sl0 + 0.02, 0.006, 1)],
                                   [[(x, y + (sl0 + sl1) / 2) for x, y in rounded_rect(lw - 0.012, sl1 - sl0 + 0.008, 0.004, 1)]],
                                   depth=0.0035, bevel=0.001, rot=(90, 0, 0))
            parts = [plate, frame, sframe]
            if not door:
                dw, dh = (lw - 0.04) * 0.5, (lh - 0.05) * 0.5
                cz = (lf0 + lf1) / 2
                parts.append(shape_extrude("lozenge", [(0, cz - dh), (dw, cz), (0, cz + dh), (-dw, cz)], depth=0.0045, bevel=0.0018, rot=(90, 0, 0)))
            for o in parts:
                _noback(o)
            return join(parts, "panel")

        pan = panel(False)
        irons.append(instances_to_mesh("panels", pan, [_ring(k * 360.0 / N, apo) for k in range(1, N)], keep_base=False))
        door = panel(True)
        Fd = _ring(0.0, apo + 0.003)           # the door stands 3 mm proud of the fixed panels
        _put([door], Fd)
        irons.append(door)
        post = cylinder("post", 0.013, hp + 0.004, loc=(0, 0, (zb + zt) / 2), verts=10, cap=False)
        apply_transform(post)
        irons.append(instances_to_mesh("posts", post, [_ring((k + 0.5) * 360.0 / N, Rc) for k in range(N)], keep_base=False))
        collar = lathe("collar", [(0.0125, -0.011), (0.0165, -0.007), (0.0165, 0.007), (0.0125, 0.011)], segments=8)
        irons.append(instances_to_mesh("collars", collar, [_ring((k + 0.5) * 360.0 / N, Rc, zm) for k in range(N)], keep_base=False))
        ros = lathe("rosette", [(0.019, 0.0), (0.019, 0.003), (0.013, 0.006), (0.009, 0.0055), (0.0065, 0.0095), (0.0, 0.012)], segments=8)
        riches.append(instances_to_mesh("rosettes", ros, [_ring(k * 360.0 / N, apo + 0.003, (lf0 + lf1) / 2, (90, 0, 0)) for k in range(1, N)], keep_base=False))

        # strap hinges on the door's right stile, hung on the corner post
        for hz in (zt - 0.024, zb + 0.024):
            x0, x1 = pw / 2 - 0.002, pw / 2 - pw * 0.72
            strap = shape_extrude("strap", [(x0, hz - 0.011), (x0, hz + 0.011), (x1 + 0.024, hz + 0.008), (x1 + 0.014, hz + 0.014), (x1, hz),
                                            (x1 + 0.014, hz - 0.014), (x1 + 0.024, hz - 0.008)], depth=0.004, bevel=0.001, rot=(90, 0, 0))
            hs = [_noback(strap)]
            for dz in (-0.0155, 0.0, 0.0155):
                hs.append(cylinder("knuckle", 0.0075, 0.014, loc=(fw / 2 - 0.006, -0.0085, hz + dz), verts=8))
            _put(hs, Fd)
            irons.extend(hs)
            pin = [cylinder("pin", 0.003, 0.054, loc=(fw / 2 - 0.006, -0.0085, hz), verts=6),
                   sphere("pin_head", 0.0052, loc=(fw / 2 - 0.006, -0.0085, hz + 0.0255), segments=8, rings=4)]
            _put(pin, Fd)
            for o in pin:
                assign(o, wrought)
            for bx in (x0 - 0.02, (x0 + x1) / 2 + 0.008):
                rivet_m.append(Fd @ _M((bx, -0.004, hz), (90, 0, 0)))
        # lock: brass escutcheon with a triangular key spindle, tongue shooting into a keeper on the post
        lx = -pw / 2 + 0.02
        lock = [rounded_box("lock_plate", (0.028, 0.006, 0.046), 0.003, loc=(lx, -0.003, zm), seg=2),
                cylinder("key_boss", 0.0085, 0.004, loc=(lx, -0.008, zm - 0.006), rot=(90, 0, 0), verts=10),
                cylinder("key_spindle", 0.0048, 0.006, loc=(lx, -0.012, zm - 0.006), rot=(90, 0, 0), verts=3)]
        _put(lock, Fd)
        for o in lock:
            assign(o, brass)
            paint(o, wear=0.5, grime=0.25)
        keep = [box("tongue", (0.034, 0.006, 0.012), loc=(-fw / 2 + 0.013, -0.009, zm + 0.012), bevel=0.001),
                box("keeper", (0.016, 0.014, 0.03), loc=(-fw / 2 - 0.001, -0.008, zm + 0.012), bevel=0.0015)]
        _put(keep, Fd)
        irons.extend(keep)
        wall_in_b = apo - 0.008
        Rin = apo - 0.014

    # ------------------------------------------------------------------ lid ring with frieze
    Rf = R + 0.008                             # frieze radius
    rp = [(Rin + 0.001, zt), (R + 0.004, zt), (R + 0.012, zt + 0.004), (R + 0.0145, zt + 0.010), (R + 0.011, zt + 0.016),
          (Rf, zt + 0.019), (Rf, zt + 0.061), (R + 0.012, zt + 0.065)]
    if hooded:
        rp += [(R + 0.02, zt + 0.07), (R + 0.027, zt + 0.078), (R + 0.0275, zt + 0.084), (R + 0.025, H - 0.002), (R + 0.02, H),
               (R - 0.006, H), (Rin + 0.004, H - 0.008)]
        Htop = H
    else:                                      # open mouth: a trumpet lip with a rolled edge
        Htop = H + 0.036
        rp += [(R + 0.017, zt + 0.071), (R + 0.024, zt + 0.084), (R + 0.036, H + 0.012), (R + 0.047, H + 0.023), (R + 0.052, H + 0.028),
               (R + 0.0535, H + 0.032), (R + 0.051, H + 0.0355), (R + 0.046, H + 0.036), (R + 0.04, H + 0.033), (R + 0.022, H + 0.016),
               (R + 0.004, H + 0.002), (Rin + 0.004, H - 0.008)]
    rp += [(Rin, H - 0.016), (Rin, zt + 0.001)]
    ring = lathe("lid_ring", rp, segments=40)
    irons.append(ring)
    if p.gilt == "rich":
        riches.append(torus("bead_top", R + 0.0145, 0.0034, loc=(0, 0, zt + 0.010), major_seg=36, minor_seg=5))
        riches.append(torus("bead_base", Rb + 0.0185, 0.0034, loc=(0, 0, zf + 0.052), major_seg=36, minor_seg=5))

    letters = str(p.lettering).strip()
    if letters:
        tx = text("lettering", letters, size=0.031, depth=0.0032, font="serif")
        _noback(tx, simplify=14.0)
        bmin, bmax = world_bbox([tx])
        wtxt = max(bmax.x - bmin.x, 1e-4)
        cxt, czt = (bmax.x + bmin.x) / 2, (bmax.z + bmin.z) / 2
        kx = min(1.0, Rf * radians(150 if slatted else 200) / wtxt)

        def wrap(co):
            ang = (co.x - cxt) * kx / Rf
            rr = Rf - co.y - 0.0004
            return Vector((rr * sin(ang), -rr * cos(ang), (co.z - czt) * min(1.0, kx * 1.25) + zt + 0.040))

        transform_verts(tx, wrap)
        gilts.append(tx)
        half = degrees(wtxt * kx / Rf) / 2
    else:
        half = 0.0
    for k in range(12):                        # cast studs round the rest of the frieze
        a = k * 30.0 + 15.0
        if min(a, 360.0 - a) > half + 8.0:
            rivet_m.append(_ring(a, Rf, zt + 0.040, (90, 0, 0)) @ Matrix.Scale(1.35, 4))

    # ------------------------------------------------------------------ hinge + hasp of the lid ring (slatted)
    if slatted:
        hg_ = [box("lug_up", (0.046, 0.014, 0.036), loc=(0, -0.007, 0.024), bevel=0.002),
               box("lug_dn", (0.046, 0.014, 0.032), loc=(0, -0.007, -0.022), bevel=0.002)]
        for dx in (-0.0195, 0.0, 0.0195):
            hg_.append(cylinder("knuckle", 0.008, 0.018, loc=(dx, -0.016, 0.0), rot=(0, 90, 0), verts=8))
        Mh = _ring(90.0, R + 0.004, zt - 0.003)
        for bx, bz in ((-0.013, 0.031), (0.013, 0.031), (-0.013, -0.029), (0.013, -0.029)):
            rivet_m.append(Mh @ _M((bx, -0.014, bz), (90, 0, 0)))
        _put(hg_, Mh)
        irons.extend(hg_)
        pin = [cylinder("pin", 0.0032, 0.064, loc=(0, -0.016, 0), rot=(0, 90, 0), verts=6)] + \
              [sphere("pin_head", 0.0055, loc=(sx * 0.0315, -0.016, 0), segments=8, rings=4) for sx in (-1, 1)]
        _put(pin, Mh)
        for o in pin:
            assign(o, wrought)

        r_l = rng.fork("latch")
        la = [box("hasp_lug", (0.03, 0.012, 0.03), loc=(0, -0.006, 0.03), bevel=0.002),
              cylinder("hasp_knuckle", 0.0055, 0.028, loc=(0, -0.014, 0.03), rot=(0, 90, 0), verts=8),
              box("hasp", (0.021, 0.004, 0.066), loc=(0, -0.014, -0.004), bevel=0.0012),
              box("staple_lug", (0.032, 0.01, 0.028), loc=(0, -0.005, -0.024), bevel=0.002)]
        Ml = _ring(270.0, R + 0.004, zt - 0.003)
        _put(la, Ml)
        irons.extend(la)
        st = [torus("staple", 0.0085, 0.0024, loc=(0, -0.0145, -0.024), rot=(0, 90, 0), major_seg=12, minor_seg=5),
              torus("shackle", 0.0095, 0.0023, loc=(0, -0.0195, -0.0375), rot=(90, 0, 0), major_seg=12, minor_seg=5)]
        sw_a = r_l.jitter(0.0, 9.0)
        pl = [rounded_box("padlock", (0.03, 0.013, 0.026), 0.004, loc=(0, -0.0195, -0.0375 - 0.0195), seg=3, bevel=0.0015),
              cylinder("padlock_key", 0.004, 0.002, loc=(0, -0.0265, -0.0375 - 0.021), rot=(90, 0, 0), verts=8)]
        Mp = Ml @ _M((0, -0.0195, -0.028)) @ _M(rot=(0, sw_a, 0)) @ _M((0, 0.0195, 0.028))   # padlock hangs a little askew
        _put([st[0]], Ml)
        _put([st[1]] + pl, Mp)
        for o in st:
            assign(o, wrought)
        for o in pl:
            assign(o, brass)
            paint(o, wear=0.35, grime=0.3, special=0.25)

    # ------------------------------------------------------------------ zinc liner
    r_ln = rng.fork("liner")
    z0 = zb + 0.004
    zr = H - 0.02
    rl0 = wall_in_b - 0.014 - (0.0 if slatted else 0.012)
    rlt = Rin - 0.010

    def rl(z):
        return lerp(rl0, rlt, (z - z0) / (zr - z0))

    fill = p.fill
    zfill = lerp(z0 + 0.04, zr - 0.035, min(fill, 1.0))
    zi = zfill - 0.012 if fill > 0.02 else z0 + 0.005
    lp = [(0.0, z0), (rl0 - 0.006, z0), (rl0, z0 + 0.006)]
    for f in (0.3, 0.72):
        zz = lerp(z0, zr, f)
        lp += [(rl(zz - 0.008), zz - 0.008), (rl(zz) + 0.004, zz), (rl(zz + 0.008), zz + 0.008)]
    lp += [(rlt, zr - 0.012), (rlt + 0.005, zr - 0.008), (rlt + 0.0065, zr - 0.002), (rlt + 0.004, zr + 0.003), (rlt - 0.001, zr + 0.003),
           (rlt - 0.003, zr - 0.003), (rl(zi) - 0.003, zi), (0.0, zi)]
    liner = lathe("liner", lp, segments=32)
    ldx, ldy = r_ln.jitter(0.0, 0.0015), r_ln.jitter(0.0, 0.0015)
    transform_verts(liner, lambda co: Vector((co.x + ldx, co.y + ldy, co.z)))
    assign(liner, zinc)
    lin = [liner]
    for sy in (-1, 1):                         # wire bail handles hanging inside the rim
        yb = sy * (rlt - 0.008)
        hb = sweep("bail", [(-0.042, yb, zr - 0.006), (-0.041, yb, zr - 0.04), (-0.032, yb, zr - 0.05), (0.032, yb, zr - 0.05), (0.041, yb, zr - 0.04),
                            (0.042, yb, zr - 0.006)], 0.0028, sides=5, smooth_path=False)
        assign(hb, zinc)
        lin.append(hb)
    for o in lin:
        paint(o, grime=lambda co, n: np.clip(0.25 + 0.5 * (n[:, 0] * co[:, 0] + n[:, 1] * co[:, 1] < 0) + 0.4 * np.clip((zb + 0.2 - co[:, 2]) / 0.2, 0, 1), 0, 1),
              special=mask_mul(height_ramp(zb, zb + 0.25), noise_mask(0.06, seed=sd + 7), 0.3 + 0.5 * wear))

    # ------------------------------------------------------------------ litter
    if fill > 0.02:
        r_t = rng.fork("litter")
        rf = rl(zfill) - 0.006
        deb = lathe("debris", [(rf, zfill - 0.008), (rf * 0.8, zfill + 0.004), (rf * 0.45, zfill + 0.012), (0.0, zfill + 0.016)], segments=14)
        displace_noise(deb, strength=0.012, scale=0.045, seed=sd + 3, axis_mask=(0, 0, 1), along_normal=False)
        assign(deb, mat("paper", "#4d4438"))
        paint(deb, grime=0.8)
        tints = ["#b9b19b", "#8a6a45"]
        wads = {t: [] for t in tints}
        nb = 3 + int(fill * 5)
        for i in range(nb):
            br = r_t.u(0.028, 0.046)
            ang, rad_ = r_t.u(0, 2 * pi), (rf - br * 0.9) * sqrt(r_t.u(0.02, 1.0))
            zz = zfill + br * r_t.u(0.25, 0.6) + (r_t.u(0.0, 0.045) * (1 - rad_ / rf) if fill > 0.85 else 0.0)
            b = sphere("wad", br, segments=9, rings=6, scale=(r_t.u(0.8, 1.3), r_t.u(0.7, 1.1), r_t.u(0.45, 0.8)))
            apply_transform(b)
            displace_noise(b, strength=br * 0.5, scale=0.017, seed=sd + 20 + i)
            flat(b)
            _put([b], _M((rad_ * cos(ang), rad_ * sin(ang), zz), (r_t.u(-25, 25), r_t.u(-25, 25), r_t.u(0, 360))))
            wads[r_t.weighted([(tints[0], 0.6), (tints[1], 0.4)])].append(b)
        if fill > 0.55:                        # a crumpled newspaper sheet on top of the heap
            sa, sb = r_t.u(0.0, 6.28), r_t.u(0.0, 6.28)
            sx_, sy_ = min(0.2, rf * 1.1), min(0.15, rf * 0.85)
            sh = surface("sheet", lambda u, v: ((u - 0.5) * sx_, (v - 0.5) * sy_ + 0.012 * sin(u * 9 + sa),
                                                zfill + 0.035 + 0.016 * sin(u * 8.0 + sa) * cos(v * 6.0 + sb) + 0.02 * (u - 0.5) ** 2 - 0.05 * (v - 0.5) ** 2),
                         7, 6, thickness=0.0012)
            realize(sh)
            flat(sh)
            _put([sh], _M((r_t.jitter(0, 0.015), r_t.jitter(0, 0.015), 0.0), (0, 0, r_t.u(0, 360))))
            wads[tints[0]].append(sh)
        for t, lst in wads.items():
            if lst:
                o = join(lst, "wads")
                assign(o, mat("paper", t))
                paint(o, grime=mask_mul(noise_mask(0.03, seed=sd + 9), 0.6), special=0.3)

    # ------------------------------------------------------------------ hood
    if hooded:
        hg = clamp(0.62 * R + 0.012, 0.125, 0.2)
        Hz = H + hg
        bal = lathe("baluster", [(0.0125, 0.0), (0.0135, 0.006), (0.0095, 0.012), (0.007, 0.02), (0.0085, 0.03), (0.0125, hg * 0.36), (0.013, hg * 0.43),
                                 (0.0075, hg * 0.62), (0.006, hg - 0.036), (0.0095, hg - 0.028), (0.0075, hg - 0.02), (0.0125, hg - 0.008),
                                 (0.0135, hg)], segments=8)
        irons.append(instances_to_mesh("balusters", bal, [_ring(a, R + 0.007, H - 0.001) for a in foot_bear], keep_base=False))
        Re = R + 0.03
        Rd, rn = Re - 0.012, 0.03
        hd = 0.56 * R + 0.01
        zd0 = Hz + 0.03

        def dome(u):
            return Rd + (rn - Rd) * u, zd0 + hd * (1 - (1 - u) ** 2.2) ** (1 / 2.2)

        us = [0.0, 0.03, 0.09, 0.18, 0.3, 0.45, 0.62, 0.8, 1.0]
        hood = lathe("hood", [(0.0, Hz + hd * 0.5), (R - 0.06, Hz + 0.014), (R - 0.034, Hz), (Re, Hz), (Re + 0.005, Hz + 0.004), (Re + 0.005, Hz + 0.013),
                              (Re - 0.001, Hz + 0.018), (Re - 0.003, Hz + 0.026), (Rd, zd0)] + [dome(u) for u in us[1:]] + [(0.0, zd0 + hd + 0.001)],
                     segments=36)
        irons.append(hood)
        rib = sweep("rib", [(0.0, -(dome(u)[0] + 0.0004), dome(u)[1]) for u in (0.0, 0.03, 0.09, 0.18, 0.3, 0.45, 0.62, 0.8, 0.97)],
                    [(-0.0065, -0.001), (0.0065, -0.001), (0.0035, 0.005), (-0.0035, 0.005)], up=(0, -1, 0), smooth_path=False,
                    scale_fn=lambda t: 1.0 - 0.4 * t)
        nr = nfeet * 2
        for j in range(nr * 3):                # the dome is riveted to its eave ring
            rivet_m.append(_ring(foot_bear[0] + (j + 0.5) * 360.0 / (nr * 3), Re + 0.005, Hz + 0.0085, (90, 0, 0)))
        irons.append(instances_to_mesh("ribs", rib, [_ring(foot_bear[0] + k * 360.0 / nr, 0.0) for k in range(nr)], keep_base=False))
        fin = lathe("finial", [(0.034, 0.0), (0.036, 0.006), (0.027, 0.012), (0.014, 0.02), (0.011, 0.03), (0.021, 0.042), (0.027, 0.056),
                               (0.024, 0.07), (0.012, 0.082), (0.006, 0.09), (0.009, 0.098), (0.0, 0.106)], segments=14, loc=(0, 0, zd0 + hd - 0.004))
        riches.append(fin)
        ztop = zd0 + hd + 0.1
    else:
        ztop = Htop

    # ------------------------------------------------------------------ crest badge
    if slatted:
        bw = clamp(0.56 * R, 0.09, 0.14)
        bh = min(bw * 1.22, (z_up - z_mid) - 0.05)
        zc = (z_mid + z_up) / 2
        slope = degrees(atan2(rw(zc + 0.05) - rw(zc - 0.05), 0.1))
        Mb = _ring(0.0, rw(zc) + 0.0045, zc, (slope, 0, 0))
    else:
        lw_in, lh_in = pw - 0.016 - 0.024, (lf1 - lf0) - 0.024
        bw = min(0.56 * R, lw_in * 0.9, 0.14)
        bh = min(bw * 1.22, lh_in * 0.9)
        Mb = _ring(0.0, apo + 0.003, (lf0 + lf1) / 2)
    rim = shape_extrude("crest_rim", _shield(bw, bh), [_shield(bw, bh, 0.82)], depth=0.009, bevel=0.0015, rot=(90, 0, 0))
    field = shape_extrude("crest_field", _shield(bw, bh, 0.84), depth=0.0055, rot=(90, 0, 0))
    crown = shape_extrude("crest_crown", _crown(bw * 0.5, bh * 0.24, bh * 0.07), depth=0.0045, bevel=0.0008, loc=(0, -0.004, 0), rot=(90, 0, 0))
    star = shape_extrude("crest_star", _star(bw * 0.1, -bh * 0.2), depth=0.004, bevel=0.0008, loc=(0, -0.004, 0), rot=(90, 0, 0))
    for o in (field, crown, star) + (() if slatted else (rim,)):
        _noback(o)
    _put([rim, field, crown, star], Mb)
    irons.append(rim)
    assign(field, enamel)
    paint(field, wear=mask_mul(noise_mask(0.02, seed=sd + 5, lo=0.5, hi=0.8), 0.3 + 0.5 * wear))
    gilts += [crown, star]
    if slatted:                                # the badge is bolted through the slats at its shoulders
        for sx in (-1, 1):
            rivet_m.append(Mb @ _M((sx * bw * 0.43, -0.009, bh * 0.4), (90, 0, 0)))

    irons.append(instances_to_mesh("rivets", rivet, rivet_m, keep_base=False))

    # ------------------------------------------------------------------ materials + weathering
    for o in irons:
        assign(o, iron)
    for o in gilts:
        assign(o, g_crest)
    for o in riches:
        assign(o, g_rich)

    def rim_wear(co, n):                       # hands and the liner being dragged out polish the mouth
        return np.clip((co[:, 2] - (H - 0.03)) / 0.03, 0, 1) * (co[:, 2] < Htop + 0.01) * (0.35 + 0.5 * wear)

    def weep(co, n):                           # a band just under the lid ring only (not the ring or the hood above it)
        return np.clip((co[:, 2] - (zt - 0.14)) / 0.15, 0, 1) * (co[:, 2] < zt + 0.012)

    for o in irons + ([] if p.gilt == "rich" else riches) + ([] if p.gilt != "none" else gilts):
        paint(o, grime=mask_mul(height_ramp(0.0, 0.4), 0.75),
              special=mask_mul(height_ramp(0.0, 0.24 + 0.35 * wear), noise_mask(0.06, seed=sd + 2, lo=0.3, hi=0.7), 0.6 + 0.4 * wear))
        paint(o, wear=rim_wear)
        # rust weeping below the lid ring, worse on a tired bin
        paint(o, special=mask_mul(weep, noise_mask(0.035, seed=sd + 6, lo=0.45, hi=0.75), 0.15 + 0.6 * wear))
    for o in (gilts if p.gilt != "none" else []) + (riches if p.gilt == "rich" else []):
        paint(o, grime=mask_mul(noise_mask(0.03, seed=sd + 8), 0.25 + 0.3 * wear))

    socket("top", "attach", (0, 0, ztop), role="prop_top")
