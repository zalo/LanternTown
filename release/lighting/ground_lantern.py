"""Squat ground lantern as flanking the monument steps. Dressed stone throughout the lower half: a
rough foot slab, a moulded base course, a short die with sunk panels and carved lozenges and a
corbelled table. On the table stands the lantern house with glazed, arched openings: either a
cast bronze / painted iron cage (slender turned corner colonnettes, pierced arch plates with
spandrel eyes, glazing bars, a hinged door with latch) or a carved stone house with wrought-iron
grilles. The metal cap is sheet copper / lead dressed over a core: separate roof planes with
standing seams, hip rolls with ball terminals, a vent cowl and a turned finial; the stone cap is
one carved block. A brass oil burner with a glass chimney carries the flame at knee height.
"""
from tk import *
from math import cos, sin, tan, atan, radians, degrees, pi, hypot, ceil
import numpy as np

META = {"name": "Stone ground lantern", "class": "prop", "mount": "ground", "focus": 0.6}
PARAMS = {
    "height": {"default": 0.86, "min": 0.6, "max": 1.1, "desc": "overall height to the finial tip (m)"},
    "width": {"default": 0.46, "min": 0.36, "max": 0.6, "desc": "width of the foot slab across the flats (m)"},
    "plan": {"default": "square", "choices": ["square", "octagonal"], "desc": "plan shape of base, shaft, lantern house and cap"},
    "cap": {"default": "pyramid", "choices": ["pyramid", "ogee", "dome"], "desc": "roof cap style"},
    "material": {"default": "bronze", "choices": ["bronze", "iron", "stone"],
                 "desc": "lantern house: bronze cage under a copper cap, painted iron cage under a lead cap, or carved stone house with iron grilles"},
    "arch": {"default": "pointed", "choices": ["pointed", "round", "flat"], "desc": "head of the glazed openings"},
    "intensity": {"default": 11.0, "min": 2.0, "max": 40.0, "desc": "light intensity (cd)"},
    "light_color": {"default": "#ffbf6a", "desc": "flame / light colour"},
    "moss": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "moss and lichen on the stonework (0 = clean)"},
    "wear": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "knocked corners, chipped paint, rubbed metal"},
    "stone": {"default": "#c9bda6", "desc": "stone tint"},
    "paint": {"default": "#1b2622", "desc": "paint colour of the iron cage (material = iron)"},
}
VARIANTS = {
    "stone_octagon": {"plan": "octagonal", "material": "stone", "cap": "ogee", "height": 1.0, "width": 0.5, "moss": 0.65,
                      "arch": "round", "stone": "#b8b0a2", "seed": 4},
    "iron_ogee": {"material": "iron", "cap": "ogee", "height": 0.72, "width": 0.4, "arch": "round", "moss": 0.1,
                  "paint": "#1c2835", "intensity": 9.0, "seed": 9},
    "bronze_dome_tall": {"plan": "octagonal", "cap": "dome", "height": 1.1, "width": 0.44, "moss": 0.2, "wear": 0.7,
                         "intensity": 14.0, "seed": 15},
    "mossy_stone_squat": {"material": "stone", "cap": "pyramid", "height": 0.64, "width": 0.52, "moss": 0.95, "wear": 0.9,
                          "arch": "flat", "stone": "#a9a79c", "intensity": 8.0, "seed": 23},
}


def _arch_y(pts, x):
    """Height of an arch outline (from arch_pts) at abscissa x."""
    best = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if min(x0, x1) - 1e-9 <= x <= max(x0, x1) + 1e-9 and abs(x1 - x0) > 1e-9:
            best = max(best, y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return best


def _mesh_uv(name, verts, faces, fuv):
    """Mesh with explicit per-face-corner UVs (metres)."""
    o = mesh_object(name, verts, faces)
    uvl = o.data.uv_layers.new(name="UVMap")
    flat_ = np.array([c for uvs in fuv for uv in uvs for c in uv], dtype=np.float32)
    if len(flat_) == len(o.data.loops) * 2:
        uvl.data.foreach_set("uv", flat_)
    uv_keep(o)
    return o


def build(p, rng):
    H = p.height
    n = 4 if p.plan == "square" else 8
    kn = 1.0 / cos(pi / n)                 # apothem -> corner radius
    tn = tan(pi / n)
    step = 360.0 / n
    yaw0 = 180.0 / n                       # puts the flats on the axes
    metal = p.material != "stone"
    bronze = p.material == "bronze"
    kind = p.arch
    seed = int(p.seed)

    st = mat("stone_smooth", p.stone, grime=0.9)
    st_foot = mat("stone_rough", tuple(c * 0.86 for c in hex_rgb(p.stone)), grime=0.9)   # a harder, slightly darker stone
    brass = mat("brass", rough=0.75, wear=0.8 + p.wear)
    if bronze:
        m_cage = mat("bronze_aged", rough=1.3, wear=1.3 + 0.9 * p.wear)
        m_roof = mat("copper_patina", scale=4.5, rough=3.5)
    elif p.material == "iron":
        m_cage = mat("iron_painted", p.paint, rough=1.35, wear=0.3 + 0.6 * p.wear)
        m_roof = mat("lead", scale=1.5)
    else:
        m_cage = mat("iron_wrought")
        m_roof = st
    g0 = 0.2 + 0.012 * p.intensity
    glass = mat("glass_lantern", "#d8823a", glow=g0, opacity=0.11)
    glass_smoke = mat("glass_lantern", "#8a4e1e", glow=0.4 * g0, opacity=0.2)       # the smoked top third of the panes
    lc = hex_rgb(p.light_color)
    flame_core = mat("emissive", tuple(lerp(c, 1.0, 0.35) for c in lc), glow=9.0 + 0.3 * p.intensity)
    flame_out = mat("emissive", "#ff8426", glow=3.5 + 0.12 * p.intensity)

    # ---- stage dimensions (apothems a_*, levels z*)
    a0 = p.width / 2
    h0, h1, h2, h3, h4, h5, h6 = [H * f for f in (0.085, 0.07, 0.175, 0.09, 0.32, 0.17, 0.09)]
    a_b, a_s, a_t = a0 * 0.82, a0 * 0.56, a0 * 0.9
    a_c = a0 * (0.66 if n == 4 else (0.78 if metal else 0.74))
    a_r = a0 * (0.97 if metal else 0.95)
    z1 = h0
    z2 = z1 + h1
    z3 = z2 + h2
    zc = z3 + h3
    zr = zc + h4
    top = zr + h5
    faces_all = [k * step for k in range(n)]
    glazed = faces_all if metal else [0.0, 90.0, 180.0, 270.0]
    solid_faces = [a for a in faces_all if a not in glazed]

    def ngon(a):
        return [(a * kn * cos(radians(yaw0 + step * k)), a * kn * sin(radians(yaw0 + step * k))) for k in range(n)]

    def nlathe(name, prof, yaw=0.0):
        o = lathe(name, [(r * kn, z) for r, z in prof], segments=n, rot=(0, 0, yaw0 + yaw), smooth_angle=25)
        apply_transform(o)
        return o

    def cove_in(r0, zA, r1, zB, m=3):      # scotia: runs inwards first, then up
        return [(r1 + (r0 - r1) * (1 - sin(radians(90 * i / m))), zA + (zB - zA) * (1 - cos(radians(90 * i / m)))) for i in range(m + 1)]

    def cove_out(r0, zA, r1, zB, m=3):     # cavetto: runs up first, then outwards
        return [(r0 + (r1 - r0) * (1 - cos(radians(90 * i / m))), zA + (zB - zA) * sin(radians(90 * i / m))) for i in range(m + 1)]

    def at(a, ang, z):                     # point on the face at angle `ang` (0 = front, -Y), apothem a
        return (a * sin(radians(ang)), -a * cos(radians(ang)), z)

    def at_t(a, ang, t, z):                # same, shifted t along the face (to the right seen from outside)
        s, c = sin(radians(ang)), cos(radians(ang))
        return (a * s + t * c, -a * c + t * s, z)

    def perim(a, ang):                     # distance from the axis to the n-gon outline in direction ang
        d = ((ang + step / 2) % step) - step / 2
        return a / cos(radians(d))

    def fall_t(co):
        """(t along the face, apothem distance, face index) of world points for the n-gon plan."""
        ang = np.arctan2(co[:, 0], -co[:, 1])
        k = np.round(ang / radians(step))
        a = k * radians(step)
        t = co[:, 0] * np.cos(a) + co[:, 1] * np.sin(a)
        d = co[:, 0] * np.sin(a) - co[:, 1] * np.cos(a)
        return t, d, k

    def streak(co, nrm, sc, sd, lo=0.35, hi=0.75, tk=None):
        """Noise that is constant down the fall line of each face (vertical run-off)."""
        t, _, k = tk if tk is not None else fall_t(co)
        q = np.stack([t, k * 3.71 + 0.37 * sd, np.zeros_like(t)], axis=1)
        return noise_mask(sc, seed=sd, lo=lo, hi=hi)(q, nrm)

    rs = rng.fork("stone")
    stones = []

    def chip(o, a, z, count, r):
        """Knock `count` top corners off an n-gon course: a clean irregular facet cut with a plane."""
        for _ in range(count):
            k = int(r.u(0, n)) % n
            ang = yaw0 + step * k
            c = Vector((a * kn * cos(radians(ang)), a * kn * sin(radians(ang)), z))
            tilt, yaw = r.u(38, 62), ang + r.u(-16, 16)
            nn = Vector((sin(radians(tilt)) * cos(radians(yaw)), sin(radians(tilt)) * sin(radians(yaw)), cos(radians(tilt))))
            p0 = c - nn * r.u(0.007, 0.016) * (0.6 + p.wear)
            S = 0.16
            boolean(o, box("chipcut", (S, S, S), loc=tuple(p0 + nn * (S / 2)), rot=(0, tilt, yaw)))

    def dress(o, chamfer=0.004, rough=0.0012, edge=0.06, material=None):
        if edge:
            densify(o, edge if n == 4 else edge * 1.2)
        displace_noise(o, strength=rough, scale=0.06, seed=rs.u(0, 40), along_normal=False)
        transform_verts(o, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
        bevel(o, chamfer, segments=1)
        assign(o, material or st)
        stones.append(o)
        return o

    def prism(name, a, zA, zB, du=0.03, dv=0.04):
        """Closed n-gon prism whose side faces are regular grids (u across, v straight up): clean
        vertical topology for run-off masks, and upright UVs."""
        fwid = 2 * a * tn
        nu, nv = max(2, int(ceil(fwid / du))), max(1, int(ceil((zB - zA) / dv)))
        verts, faces, fuv = [], [], []
        for k, ang in enumerate(faces_all):
            b = len(verts)
            for j in range(nv + 1):
                for i in range(nu + 1):
                    verts.append(at_t(a, ang, -fwid / 2 + fwid * i / nu, zA + (zB - zA) * j / nv))
            for j in range(nv):
                for i in range(nu):
                    v0 = b + j * (nu + 1) + i
                    faces.append((v0, v0 + 1, v0 + nu + 2, v0 + nu + 1))
                    u0, u1 = k * fwid + fwid * i / nu, k * fwid + fwid * (i + 1) / nu
                    w0, w1 = zA + (zB - zA) * j / nv, zA + (zB - zA) * (j + 1) / nv
                    fuv.append([(u0, w0), (u1, w0), (u1, w1), (u0, w1)])
        for zz, rev in ((zB, False), (zA, True)):
            ring = [at_t(a, ang, -fwid / 2 + fwid * i / nu, zz) for ang in faces_all for i in range(nu)]
            if rev:
                ring = ring[::-1]
            b = len(verts)
            verts += ring
            faces.append(tuple(range(b, b + len(ring))))
            fuv.append([(x, y) for x, y, _ in ring])
        o = _mesh_uv(name, verts, faces, fuv)
        weld(o, 1e-4)
        return o

    # ---- foot slab (rough dressed, weathered top) and moulded base course
    foot = nlathe("foot", [(a0, 0.0), (a0, h0 - 0.016), (a0 - 0.02, h0), (a_b - 0.014, h0)], yaw=rs.jitter(0, 0.8))
    chip(foot, a0, h0 - 0.004, int(round(p.wear * 3)), rs.fork("chip"))
    dress(foot, chamfer=0.006, rough=0.0026, edge=0.065, material=st_foot)
    base = nlathe("base", [(a_b, z1 - 0.002), (a_b, z1 + 0.3 * h1), (a_b - 0.005, z1 + 0.38 * h1)]
                  + cove_in(a_b - 0.005, z1 + 0.38 * h1, a_s + 0.007, z1 + 0.9 * h1)[1:]
                  + [(a_s + 0.007, z2), (a_s - 0.014, z2)], yaw=rs.jitter(0, 0.4))
    dress(base, chamfer=0.003, edge=0.07)

    # ---- die (shaft) with a sunk panel and a carved lozenge on every face
    shaft = prism("shaft", a_s, z2 - 0.002, z3 + 0.002, du=0.032, dv=0.045)
    fw_s = 2 * a_s * tn
    mg = max(0.014, fw_s * 0.16)
    pw, ph = fw_s - 2 * mg, h2 - 2 * max(0.022, 0.14 * h2)
    sink = 0.007
    cutters = [box("pcut", (pw, 2 * sink, ph), loc=at(a_s, a, z2 + h2 / 2), rot=(0, 0, a)) for a in faces_all]
    boolean(shaft, join(cutters, "pcuts"))
    smooth(shaft, 30)
    dress(shaft, chamfer=0.0025, edge=None)
    loz = cone("loz", r1=pw * 0.3, r2=0.0, depth=sink * 1.1, verts=4)
    place(loz, scale=(1.0, min(1.9, ph / pw * 0.62), 1.0))
    realize(loz)
    lozs = instances_to_mesh("lozenges", loz, [trs(at(a_s - sink + sink * 0.5, a, z2 + h2 / 2), (90, 0, a)) for a in faces_all], keep_base=False)
    assign(lozs, st)
    stones.append(lozs)

    # ---- table: astragal, cavetto and a weathered slab that carries the lantern house
    # (the top is a steep weathering - a wash - so rain runs off and the flame only grazes it)
    zs = z3 + 0.36 * h3
    a_w = a_c + 0.016                      # the lantern house stands on the flat inside the wash
    ledge_w = max(a_t - a_w, 0.004)
    wash = min(0.38 * h3, 0.75 * ledge_w)
    table = nlathe("table", [(a_s - 0.014, z3), (a_s + 0.008, z3), (a_s + 0.008, z3 + 0.009), (a_s + 0.002, z3 + 0.013)]
                   + cove_out(a_s + 0.002, z3 + 0.013, a_t - 0.006, zs)[1:]
                   + [(a_t, zs + 0.003), (a_t, zc - wash - 0.003), (a_t - 0.004, zc - wash), (a_w, zc), (a_c - 0.035, zc)], yaw=rs.jitter(0, 0.3))
    chip(table, a_t, zc - wash - 0.001, int(round(p.wear * 1.6)), rs.fork("chip2"))
    dress(table, chamfer=0.003, edge=0.07)

    # ================= lantern house =================
    metals, glasses, smokes = [], [], []

    def glazing(name, pts, y):
        """One sheet of glass per glazed face, outline pts = [(x, z)] on the front face. The glass stops at
        the transom / top bar: the arch heads and spandrel eyes above it are the open vents of the lamp
        (sheltered by the wide eave), so no pane ever stands level with the flame to flare white."""
        area = sum(x0 * z1 - x1 * z0 for (x0, z0), (x1, z1) in zip(pts, pts[1:] + pts[:1]))
        if area < 0:
            pts = pts[::-1]
        o = _mesh_uv(name, [(x, -y, z) for x, z in pts], [tuple(range(len(pts)))], [[(x, z) for x, z in pts]])
        return instances_to_mesh(name + "s", o, [trs(rot=(0, 0, a)) for a in glazed], keep_base=False)
    posts = faces_m = None
    Rsh = 0.006 if n == 4 else 0.005       # colonnette shaft radius
    if metal:
        zfl = zc + 0.02                    # floor inside the cage
        zt = zr - 0.026                    # underside of the head rail
        Lp = zt - zfl
        a_p = a_c - 0.004                  # outer plane of the face plates
        fw = 2 * a_p * tn
        sill = nlathe("sill", [(a_c + 0.012, zc), (a_c + 0.012, zc + 0.007), (a_c + 0.004, zc + 0.011), (a_c + 0.004, zc + 0.017),
                               (a_c - 0.003, zfl), (0.0, zfl)])
        head = nlathe("head", [(0.0, zt), (a_c - 0.003, zt), (a_c + 0.004, zt + 0.004), (a_c + 0.004, zt + 0.014),
                               (a_c + 0.012, zt + 0.02), (a_c + 0.012, zr), (a_c - 0.03, zr)])
        for o in (sill, head):
            bevel(o, 0.001, segments=1)
        metals += [sill, head]
        # slender turned corner colonnettes standing proud of the frame corners
        R, mid = Rsh, Lp / 2
        prof = [(R + 0.0045, 0.0), (R + 0.0045, 0.006), (R + 0.001, 0.009), (R + 0.0035, 0.013), (R, 0.017)]
        if n == 4:
            prof += [(R, mid - 0.005), (R + 0.003, mid - 0.002), (R + 0.003, mid + 0.002), (R, mid + 0.005)]
        prof += [(R, Lp - 0.017), (R + 0.0035, Lp - 0.013), (R + 0.001, Lp - 0.009), (R + 0.0045, Lp - 0.006), (R + 0.0045, Lp)]
        post = lathe("post", prof, segments=8 if n == 4 else 6, smooth_angle=62)
        Rv = a_p * kn + 0.003
        corner_xy = np.array([(Rv * cos(radians(yaw0 + step * k)), Rv * sin(radians(yaw0 + step * k))) for k in range(n)])
        posts = instances_to_mesh("posts", post, [trs((x, y, zfl)) for x, y in corner_xy], keep_base=False)
        metals.append(posts)
        # pierced arch plate + glazing bars (one module, stamped on every glazed face)
        stile = 0.02 if n == 4 else 0.014
        ow = fw - 2 * stile
        sb, tb = 0.01, 0.013
        oh = Lp - sb - tb
        w = ow / 2
        rise = {"pointed": min(1.3 * w, 0.62 * oh), "round": min(w, 0.55 * oh), "flat": 0.0}[kind]
        sp = oh - rise
        arch = arch_pts(ow, oh, spring=sp, kind=kind, n=10)
        outer = [(-fw / 2, 0), (fw / 2, 0), (fw / 2, Lp), (-fw / 2, Lp)]
        holes = [[(x, y + sb) for x, y in arch]]
        if kind != "flat":                 # spandrel eyes
            r_e = 0.24 * min(w, rise)
            while r_e > 0.005:
                cx, cy = w - 0.85 * r_e, oh - 0.85 * r_e
                dense = [(x0 + (x1 - x0) * t / 6, y0 + (y1 - y0) * t / 6) for (x0, y0), (x1, y1) in zip(arch, arch[1:]) for t in range(7)]
                if min(hypot(x - cx, y - cy) for x, y in dense) > r_e + 0.006:
                    break
                r_e *= 0.9
            if r_e > 0.005:
                for sx in (-1, 1):
                    holes.append(circle_pts(r_e, 12 if n == 4 else 8, center=(sx * cx, cy + sb)))
        plate = shape_extrude("plate", outer, holes, depth=0.005, loc=(0, -(a_p - 0.005), zfl), rot=(90, 0, 0))
        mod = [plate]
        yb = a_p - 0.0025
        if ow > 0.125:
            mod.append(box("mullion", (0.006, 0.006, oh + 0.004), loc=(0, -yb, zfl + sb + oh / 2)))
        z_bar = sb + (sp if kind != "flat" else 0.64 * oh)
        mod.append(box("transom", (ow + 0.004, 0.006, 0.006), loc=(0, -yb, zfl + z_bar)))
        if n == 4:
            rivet = lathe("rivet", [(0.0038, 0.0), (0.003, 0.0022), (0.0, 0.0032)], segments=6)
            rvs = [trs((x, -a_p, zfl + zz), (90, 0, 0)) for x in (-0.36 * ow, 0.36 * ow) for zz in (sb * 0.5, Lp - tb * 0.5)]
            mod.append(instances_to_mesh("rivets", rivet, rvs, keep_base=False))
        face = join(mod, "face")
        faces_m = instances_to_mesh("faces", face, [trs(rot=(0, 0, a)) for a in glazed], keep_base=False)
        metals.append(faces_m)
        yp = a_p - 0.0068
        xg, zg0, zg1 = ow / 2 + 0.006, zfl + sb - 0.005, zfl + z_bar
        zgs = zg1 - 0.3 * (zg1 - zg0)       # the top third of every pane is smoked by the lamp
        glasses.append(glazing("pane", [(-xg, zg0), (xg, zg0), (xg, zgs), (-xg, zgs)], yp))
        smokes.append(glazing("pane_smoked", [(-xg, zgs), (xg, zgs), (xg, zg1), (-xg, zg1)], yp))
        # door furniture on the front: two knuckle hinges and a turn latch
        xh = -(w + 0.006)
        zmid = zfl + sb
        for f in (0.2, 0.76):
            metals.append(cylinder("hinge", 0.0034, 0.022, loc=(xh, -(a_p + 0.003), zmid + oh * f), verts=6))
            metals.append(box("hinge_leaf", (0.012, 0.002, 0.014), loc=(xh + 0.007, -(a_p + 0.001), zmid + oh * f)))
        latch = [bolt("latch", 0.0062, 0.007, loc=(w + 0.006, -a_p, zmid + oh * 0.46), rot=(90, 0, 0), kind="dome"),
                 box("latch_bar", (0.024, 0.003, 0.0065), loc=(w - 0.003, -(a_p + 0.0035), zmid + oh * 0.46), rot=(0, rng.u(-14, 6), 0), bevel=0.0008,
                     segments=1)]
        for o in latch:
            assign(o, brass)
            paint(o, wear=0.8)
        z_open0, z_open1 = zfl, zt
        z_glass = zg1
        w_in = a_p - 0.02
    else:
        # carved stone house: solid block, two crossing arched tunnels = four openings
        fw = 2 * a_c * tn
        pier = fw * (0.2 if n == 4 else 0.19)
        ow = fw - 2 * pier
        w = ow / 2
        sillh = max(0.022, 0.1 * h4)
        zfl = zc + sillh
        oh = h4 - sillh - max(0.028, 0.13 * h4)
        rise = {"pointed": min(1.3 * w, 0.62 * oh), "round": min(w, 0.55 * oh), "flat": 0.0}[kind]
        sp = oh - rise
        arch = arch_pts(ow, oh, spring=sp, kind=kind, n=12)
        house = prism("house", a_c, zc - 0.002, zr + 0.002, du=0.03, dv=0.04)
        for a in (0.0, 90.0):
            boolean(house, extrude_profile("tunnel", arch, 2 * a_c + 0.1, loc=at(-(a_c + 0.05), a, zfl), rot=(90, 0, a)))
        smooth(house, 30)
        dress(house, chamfer=0.003, edge=None)
        if solid_faces:                    # diamond-point bosses carved on the closed diagonal faces
            bw = fw * 0.3
            boss = cone("boss", r1=bw, r2=0.0, depth=0.009, verts=4)
            place(boss, scale=(1.0, min(2.6, oh * 0.4 / bw), 1.0))
            realize(boss)
            bosses = instances_to_mesh("bosses", boss, [trs(at(a_c + 0.0035, a, zfl + oh * 0.5), (90, 0, a)) for a in solid_faces], keep_base=False)
            assign(bosses, st)
            stones.append(bosses)
        # glass set in a rebate up to the top bar (open arch head above it), wrought-iron grille in front of it
        zsplit = sp if kind != "flat" else 0.68 * oh
        wb = w + 0.005
        yp = a_c - 0.032
        zgs = zfl + 0.7 * zsplit
        glasses.append(glazing("pane", [(-wb, zfl - 0.002), (wb, zfl - 0.002), (wb, zgs), (-wb, zgs)], yp))
        smokes.append(glazing("pane_smoked", [(-wb, zgs), (wb, zgs), (wb, zfl + zsplit), (-wb, zfl + zsplit)], yp))
        yg = a_c - 0.016
        inner = arch_pts(ow - 0.007, oh - 0.004, spring=sp, kind=kind, n=12)
        gr = [sweep("grille_frame", [(x, -yg, zfl + 0.0035 + y) for x, y in inner], rounded_rect(0.007, 0.007, 0.0015, 1), closed=True,
                    smooth_path=False, up=(0, -1, 0))]
        xs = [0.0] if ow < 0.12 else [-ow / 6, ow / 6]
        for x in xs:
            hb_ = _arch_y(inner, x)
            gr.append(box("grille_v", (0.0055, 0.0055, hb_), loc=(x, -yg, zfl + 0.0035 + hb_ / 2), rot=(0, 0, 45)))
        for f in ((0.34, 0.68) if kind == "flat" else (0.5, 1.0)):
            zz = (sp if kind != "flat" else oh) * f
            gr.append(box("grille_h", (ow - 0.004, 0.004, 0.009), loc=(0, -yg, zfl + 0.0035 + zz)))
        grille = join(gr, "grille")
        metals.append(instances_to_mesh("grilles", grille, [trs(rot=(0, 0, a)) for a in glazed], keep_base=False))
        # pintle hinges let into the stone jamb + a hasp on the other side (front grille opens)
        for f in (0.22, 0.74):
            zz = zfl + (sp if kind != "flat" else oh) * f
            metals.append(cylinder("pintle", 0.004, 0.024, loc=(-w - 0.001, -(yg + 0.006), zz), verts=6))
            metals.append(box("pintle_strap", (0.03, 0.003, 0.012), loc=(-w + 0.013, -(yg + 0.0045), zz)))
        metals.append(box("hasp", (0.028, 0.003, 0.012), loc=(w - 0.006, -(yg + 0.0045), zfl + (sp if kind != "flat" else oh) * 0.5)))
        z_open0, z_open1 = zfl, zfl + oh
        z_glass = zfl + zsplit
        w_in = w

    # ---- brass oil burner (font, stem, gallery), glass chimney and flame in the middle of the house
    open_h = z_open1 - z_open0
    k = clamp(min(open_h / 0.2, (w_in - 0.008) / 0.034), 0.6, 1.25)
    # The wick stands just above the top of the glass: seen from above, the flame then shows through the open arch
    # heads and its reflection falls above the panes instead of flaring in them. A tall house gets a stemmed lamp.
    hb = clamp(z_glass - zfl + 0.02, 0.05, open_h - 0.055)
    z0 = zfl
    tall = hb > 0.096 * k
    hbody = 0.066 * k if tall else hb
    zs = z0 + hb - hbody                   # underside of the oil font
    prof = [(0.0, z0 - 0.001)]
    if tall:                               # spreading foot and baluster stem
        prof += [(0.026 * k, z0 - 0.001), (0.027 * k, z0 + 0.004 * k), (0.013 * k, z0 + 0.011 * k), (0.0075 * k, z0 + 0.02 * k)]
        if zs - z0 > 0.06 * k:
            zm = z0 + 0.02 * k + 0.45 * (zs - z0 - 0.02 * k)
            prof += [(0.0075 * k, zm - 0.009 * k), (0.013 * k, zm), (0.0075 * k, zm + 0.009 * k)]
        prof += [(0.0075 * k, zs - 0.004 * k), (0.012 * k, zs)]
    else:
        prof += [(0.027 * k, z0 - 0.001)]
    kb = 0.76 if tall else 1.0             # a stemmed lamp carries a small peg font
    prof += [(r_ * k, zs + f_ * hbody) for r_, f_ in ((0.03 * kb, 0.13), (0.034 * kb, 0.32), (0.03 * kb, 0.52), (0.014, 0.65), (0.011, 0.71),
                                                     (0.02, 0.76), (0.02, 0.87), (0.008, 0.9), (0.008, 1.0), (0.0, 1.0))]
    font = lathe("font", prof, segments=8)
    assign(font, brass)
    # oily, sooted shoulder right under the flame (so the font does not outshine it), rubbed belly and foot
    paint(font, wear=lambda co, nrm: 0.55 * np.clip(1.0 - np.abs(nrm[:, 2]) * 1.6, 0, 1),
          grime=lambda co, nrm: np.clip(0.25 + 0.75 * np.clip(nrm[:, 2] * 1.5, 0, 1) * (co[:, 2] > zs + 0.2 * hbody), 0, 1))
    zA = zs + 0.87 * hbody
    zf = z0 + hb
    hc = min(0.085 * k, z_open1 - zA - 0.012)
    chim = lathe("chimney", [(0.017 * k, zA), (0.023 * k, zA + 0.14 * hc), (0.021 * k, zA + 0.3 * hc), (0.014 * k, zA + 0.58 * hc), (0.013 * k, zA + hc),
                             (0.0118 * k, zA + hc), (0.0128 * k, zA + 0.58 * hc), (0.0198 * k, zA + 0.3 * hc), (0.0218 * k, zA + 0.14 * hc),
                             (0.0158 * k, zA + 0.002)], segments=10)
    assign(chim, glass)
    fh = min(0.048 * k, 0.78 * (zA + hc - zf))
    kf_ = fh / 0.048
    flame = lathe("flame", [(0.0, zf), (0.005 * kf_, zf + 0.003 * kf_), (0.0095 * kf_, zf + 0.012 * kf_), (0.008 * kf_, zf + 0.024 * kf_),
                            (0.004 * kf_, zf + 0.036 * kf_), (0.0, zf + fh)], segments=8)
    transform_verts(flame, lambda co: Vector((co.x, co.y * 0.6, co.z)))            # a flat-wick flame
    place(flame, rot=(0, 0, 28))
    apply_transform(flame)
    assign(flame, flame_out)
    assign_faces(flame, flame_core, lambda c, nn: zf + 0.0035 * kf_ < c.z < zf + 0.027 * kf_)
    z_light = zf + 0.016 * kf_
    # the lamp's own font, right under the flame, must not throw a black cone over the house; walls, posts and table do
    near = 0.046 if not tall else clamp(0.085 * k, 0.046, 0.6 * a_p if metal else 0.97 * w_in)
    light("flame", (0, 0, z_light), color=p.light_color, intensity=p.intensity, range=6.5, flicker=0.25, shadow_near=near)

    # ================= roof cap and finial =================
    hf = h6
    roofs = []
    if metal:
        te = 0.011
        r_n = 0.021 * a0 / 0.23
        under = [(a_c - 0.02, zr), (a_r - 0.008, zr + 0.001), (a_r, zr + 0.004), (a_r, zr + te)]
    else:
        te = max(0.03, 0.3 * h5)
        r_n = 0.04 * a0 / 0.23
        under = ([(a_c - 0.03, zr), (a_c + 0.005, zr), (a_c + 0.005, zr + 0.2 * te)]
                 + cove_out(a_c + 0.005, zr + 0.2 * te, a_r - 0.006, zr + 0.62 * te)[1:] + [(a_r, zr + 0.62 * te + 0.003), (a_r, zr + te)])
    ze = zr + te
    poly = p.cap != "dome"
    if p.cap == "pyramid":
        m = 6
        surf = [(lerp(a_r - 0.003, r_n, i / m), ze + 0.002 + (top - ze - 0.002) * (i / m) ** 1.45) for i in range(m + 1)]
    elif p.cap == "ogee":
        m = 8
        D = a_r * 0.9 - r_n
        surf = [(a_r - 0.003, ze + 0.002)] + [(r_n + D * (1 - i / m) + 0.15 * D * sin(2 * pi * i / m), ze + 0.005 + (top - ze - 0.005) * i / m)
                                                for i in range(m + 1)]
    else:
        rd = a_c + 0.004
        zd = ze + 0.012
        m = 8
        amax = np.degrees(np.arccos(clamp(r_n * kn / rd, 0.05, 0.9)))
        surf = [(rd, ze + 0.004), (rd, zd)] + [(rd * cos(radians(amax * i / m)), zd + (top - zd) * sin(radians(amax * i / m)) / sin(radians(amax)))
                                                for i in range(1, m + 1)]
    surf_r = np.array([r for r, _ in surf][::-1])
    surf_z = np.array([z for _, z in surf][::-1])
    NG = 8                                 # gores of a dome

    def skin(name, rows):
        """Sheet-metal roof covering: one sheet per roof plane (or dome gore), each with its own
        developed UVs (u across the plane, v up the fall line) so the patina never smears. On the
        flat planes the vertex columns run parallel to the fall line (not converging on the apex),
        so run-off streaks painted per column stay straight."""
        ru = rng.fork(name)
        vl = [0.0]
        for (r0, zA_), (r1, zB_) in zip(rows, rows[1:]):
            vl.append(vl[-1] + hypot(r1 - r0, zB_ - zA_))
        verts, uvs, faces = [], [], []
        if poly:
            dt = 0.019
            N = int(rows[0][0] * tn / dt)
            for g in range(n):
                uo, vo = ru.u(0, 1.2), ru.u(0, 1.2)
                idx = []
                for i, (r, z) in enumerate(rows):
                    hw_ = r * tn
                    ks = [c for c in range(-N, N + 1) if abs(c * dt) < hw_ - 0.3 * dt]
                    row = {}
                    for key, t in [("L", -hw_)] + [(c, c * dt) for c in ks] + [("R", hw_)]:
                        row[key] = len(verts)
                        verts.append(at_t(r, g * step, t, z))
                        uvs.append((uo + 4.5 * t, vo + 0.45 * vl[i]))        # stretched: the oxide grain runs down the fall line
                    idx.append((row, ks))
                for (lo_, kl), (up_, ku) in zip(idx, idx[1:]):
                    if not ku:
                        fan = [lo_["L"]] + [lo_[c] for c in kl] + [lo_["R"]]
                        faces += [(up_["L"], q0, q1) for q0, q1 in zip(fan, fan[1:])] + [(up_["L"], lo_["R"], up_["R"])]
                        continue
                    for c0, c1 in zip(ku, ku[1:]):
                        faces.append((lo_[c0], lo_[c1], up_[c1], up_[c0]))
                    fan = [lo_["L"]] + [lo_[c] for c in kl if c <= ku[0]]
                    faces += [(up_["L"], q0, q1) for q0, q1 in zip(fan, fan[1:])] + [(up_["L"], lo_[ku[0]], up_[ku[0]])]
                    fan = [lo_[c] for c in kl if c >= ku[-1]] + [lo_["R"]]
                    faces += [(up_["R"], q0, q1) for q0, q1 in zip(fan, fan[1:])] + [(up_["R"], up_[ku[-1]], lo_[ku[-1]])]
        else:
            cols = 5
            for g in range(NG):
                uo, vo = ru.u(0, 1.2), ru.u(0, 1.2)
                b = len(verts)
                for i, (r, z) in enumerate(rows):
                    for j in range(cols + 1):
                        s = -1.0 + 2.0 * j / cols
                        th = radians(-90.0 + (g + 0.5 * s) * 360.0 / NG)
                        verts.append((r * cos(th), r * sin(th), z))
                        uvs.append((uo + 4.5 * r * 0.5 * s * 2 * pi / NG, vo + 0.45 * vl[i]))
                for i in range(len(rows) - 1):
                    for j in range(cols):
                        q = [i * (cols + 1) + j, i * (cols + 1) + j + 1, (i + 1) * (cols + 1) + j + 1, (i + 1) * (cols + 1) + j]
                        faces.append(tuple(b + v for v in q))
        o = _mesh_uv(name, verts, faces, [[uvs[v] for v in f] for f in faces])
        smooth(o, 40)
        return o

    def seams(name, rows, xs, hgt=0.0085, wd=0.0035):
        """Standing seams: folded upstands running up the fall line until they die into the hip roll."""
        verts, faces, fuv = [], [], []
        for g in range(n):
            for x in xs:
                pts = []
                for i, (r, z) in enumerate(rows):
                    if r * tn > abs(x) + 0.004:
                        pts.append((r, z))
                    else:
                        (rp_, zp_), re_ = rows[i - 1], (abs(x) + 0.004) / tn
                        f = (rp_ - re_) / max(rp_ - r, 1e-6)
                        pts.append((re_, zp_ + (z - zp_) * f))
                        break
                if len(pts) < 2:
                    continue
                b = len(verts)
                arc = [0.0]
                for i, (r, z) in enumerate(pts):
                    if i:
                        arc.append(arc[-1] + hypot(r - pts[i - 1][0], z - pts[i - 1][1]))
                    verts += [at_t(r, g * step, x - wd, z + 0.0003), at_t(r, g * step, x, z + hgt), at_t(r, g * step, x + wd, z + 0.0003)]
                for i in range(len(pts) - 1):
                    q0, q1 = b + 3 * i, b + 3 * i + 3
                    faces += [(q0, q0 + 1, q1 + 1, q1), (q0 + 1, q0 + 2, q1 + 2, q1 + 1)]
                    fuv += [[(arc[i], 0.0), (arc[i], 0.01), (arc[i + 1], 0.01), (arc[i + 1], 0.0)],
                            [(arc[i], 0.01), (arc[i], 0.02), (arc[i + 1], 0.02), (arc[i + 1], 0.01)]]
                faces.append((b, b + 2, b + 1))
                fuv.append([(0.0, 0.0), (0.0, 0.02), (0.0, 0.01)])
        o = _mesh_uv(name, verts, faces, fuv)
        flat(o)
        return o

    if metal:
        if poly:
            roofs.append(nlathe("eave", under + [(a_r - 0.003, ze + 0.002), (a_r - 0.03, ze + 0.0012)]))
            roofs.append(skin("roof_skin", surf))
            hw = a_r * tn
            roofs.append(seams("roof_seams", surf, [-0.5 * hw, 0.0, 0.5 * hw] if n == 4 else [-0.36 * hw, 0.36 * hw]))
            nr, rib_k, a_rib = n, kn, yaw0
        else:
            roofs.append(nlathe("roof_plate", under + [(a_r - 0.004, ze + 0.002), (rd * 0.98, ze + 0.014), (0.0, ze + 0.014)]))
            rows = []
            for i, (r, z) in enumerate(surf):          # two lapped sheet joints round the dome
                rows.append((r, z))
                if i in (4, 7):
                    rows.append((r + 0.003, z + 0.0006))
            roofs.append(skin("roof_skin", rows))
            roofs.append(torus("dome_ring", major=rd + 0.002, minor=0.005, loc=(0, 0, zd), major_seg=24, minor_seg=5))
            nr, rib_k, a_rib = NG, 1.0, 22.5
        bevel(roofs[0], 0.0012, segments=1)
        # hip rolls / dome ribs dressed over the joints
        ribs = []
        for j in range(nr):
            a = radians(a_rib + 360.0 * j / nr)
            path = [(r * rib_k * cos(a), r * rib_k * sin(a), z + 0.001) for r, z in surf]
            ribs.append(sweep(f"rib{j}", path, 0.0048, sides=5, smooth_path=False))
            if poly:                       # ball terminal where the hip roll meets the eaves
                ribs.append(sphere(f"hipball{j}", 0.0085, loc=(path[0][0], path[0][1], path[0][2] + 0.003), segments=6, rings=4))
        roofs.append(join(ribs, "ribs"))
        # vent cowl (drum + overhanging cap with an air gap) and a turned finial
        kf = hf / 0.077
        rv = r_n * kn + 0.01
        z_cap = top + 0.017 * kf
        vent = lathe("vent", [(rv + 0.004, top - 0.016), (rv + 0.004, top - 0.01), (rv, top - 0.008), (rv, top + 0.011 * kf), (rv * 0.55, top + 0.011 * kf),
                              (rv * 0.55, z_cap), (rv + 0.009, z_cap), (rv + 0.009, z_cap + 0.003), (rv * 0.4, z_cap + 0.014 * kf), (0.0, z_cap + 0.014 * kf)],
                     segments=12, smooth_angle=35)
        roofs.append(vent)
        for o in roofs:
            assign(o, m_roof)
        zs_ = z_cap + 0.012 * kf
        rb = 0.2 * hf
        zb = top + hf - rb * 2.3
        fin = [(rv * 0.5, zs_), (rv * 0.5, zs_ + 0.004), (rv * 0.3, zs_ + 0.008), (rb * 0.42, zb - rb * 1.2), (rb * 0.6, zb - rb * 1.02)]
        fin += [(rb * cos(radians(a)), zb + rb * sin(radians(a))) for a in range(-55, 71, 25)]
        fin += [(rb * 0.34, zb + rb * 1.08), (rb * 0.45, zb + rb * 1.25), (rb * 0.2, zb + rb * 1.6), (0.0, top + hf)]
        finial = lathe("finial", fin, segments=10)
        assign(finial, brass)
    else:
        if poly:
            sroof = [nlathe("roof", under + surf + [(0.0, top)])]
        else:
            sroof = [nlathe("roof_plate", under + [(a_r - 0.004, ze + 0.002), (rd * 0.98, ze + 0.014), (0.0, ze + 0.014)]),
                     lathe("roof_dome", surf + [(0.0, top)], segments=20)]
        for o in sroof:
            dress(o, chamfer=0.004, edge=0.07)
        ez = {"pyramid": 1.0, "ogee": 1.45, "dome": 0.85}[p.cap]
        rb = min(0.36 * hf / ez * 1.0, r_n * 1.0)
        zb = top + hf - rb * ez
        fin = [(r_n * kn * 1.02, top - 0.006), (r_n * kn * 1.02, top + 0.004), (r_n * 0.7, top + 0.012), (r_n * 0.5, max(top + 0.016, zb - rb * ez * 0.95))]
        fin += [(rb * cos(radians(a)), zb + rb * ez * sin(radians(a))) for a in range(-50, 90, 28)] + [(0.0, top + hf)]
        finial = lathe("finial", fin, segments=12)
        dress(finial, chamfer=0.002, rough=0.0008, edge=None)

    # ================= materials and weathering =================
    for o in metals:
        assign(o, m_cage)
    for o in glasses:
        assign(o, glass)
    for o in smokes:
        assign(o, glass_smoke)

    GR = 0.46                              # art-directed grime never darkens stone past ~half its albedo

    def moss_fn(co, nrm):                  # soft damp-green bloom round the modelled moss
        up = np.clip(nrm[:, 2], 0, 1)
        north = np.clip(nrm[:, 1], 0, 1)                       # the back (+Y) is the damp side
        low = np.clip(1.0 - co[:, 2] / (0.5 * H), 0, 1)
        nz = noise_mask(0.17, seed=seed + 11, lo=0.2, hi=0.9)(co, nrm)
        return np.clip((0.7 * up + 0.4 * north * (0.4 + 0.6 * low) + 0.15 * low) * nz * p.moss, 0, 1) * 0.55

    def stone_grime(co, nrm):
        z = co[:, 2]
        tk = fall_t(co)
        vert = 1.0 - np.abs(nrm[:, 2])
        up = np.clip(nrm[:, 2], 0, 1)
        splash = np.clip(1.0 - z / 0.075, 0, 1) ** 1.5 * 0.9                      # splash-back fades out over ~7 cm
        ledge = up * np.where(z < z3, 0.45, 0.8 * (z < zc + 0.01))                  # dirt settles on the ledges and the table wash
        # vertical run-off down the die below the table drip edge
        run = (z > z2) * (z < z3 + 0.004) * vert * streak(co, nrm, 0.03, seed + 5, 0.3, 0.8, tk) * (0.35 + 0.65 * np.clip((z - z2) / h2, 0, 1)) * 0.85
        g = np.maximum(np.maximum(splash, ledge), run)
        if not metal:
            # run-off under the cap eave and below the sills, soot licking out of the heads of the openings
            wall = (z > zc) * (z < zr) * vert
            run2 = wall * streak(co, nrm, 0.028, seed + 7, 0.35, 0.8, tk) * np.clip((z - zc) / h4, 0, 1) ** 2 * 0.7
            near = np.clip(1.0 - (np.abs(tk[0]) - w) / 0.03, 0, 1)
            soot = wall * near * np.clip((z - (z_open1 - 0.35 * rise - 0.01)) / (zr - z_open1 + 0.35 * rise + 0.01), 0, 1) * 1.0
            sill_run = (z < zc) * (z > z3) * np.clip(1.0 - np.abs(tk[0]) / (w + 0.02), 0, 1) * 0.5
            g = np.maximum(g, np.maximum(np.maximum(run2, soot), sill_run))
        soffit = (z > zr - 0.002) * np.clip(-nrm[:, 2] * 1.5, 0, 1) * 1.0           # smoked soffit of the cap
        g = np.clip(np.maximum(g, soffit), 0, 1) * GR
        if not metal:                                                              # the inside of the house is black with lamp soot
            inner = (z > zc + 0.004) * (z < zr - 0.004) * np.clip((a_c - 0.004 - tk[1]) / 0.03, 0, 1)
            g = np.maximum(g, 0.85 * inner)
        return g

    for o in stones:
        paint(o, grime=stone_grime, special=moss_fn)

    if metal:
        wk = 1.0 if bronze else 0.3

        def base_wear(co, nrm):
            return (0.34 + 0.3 * noise_mask(0.05, seed=seed + 31, lo=0.3, hi=0.8)(co, nrm) * (0.5 + p.wear)) * wk

        def post_wear(co, nrm):                                 # rubbed bright on the turned rings
            d = np.min(np.hypot(co[:, None, 0] - corner_xy[None, :, 0], co[:, None, 1] - corner_xy[None, :, 1]), axis=1)
            return np.where(d > Rsh + 0.0012, 0.95, 0.45) * wk

        def face_wear(co, nrm):                                 # rubbed arris round the arch, the eyes and the bars
            t, d, _ = fall_t(co)
            return np.where((np.abs(t) < w + 0.0015) & (d > a_p - 0.0035), 0.9, 0.42) * wk

        def metal_special(co, nrm):                             # verdigris / rust creeping from the joints
            z = co[:, 2]
            nz = noise_mask(0.045, seed=seed + 21, lo=0.3, hi=0.8)(co, nrm)
            low = np.clip(1.0 - (z - zc) / (0.2 * h4), 0, 1)
            high = np.clip(1.0 - (zr - z) / (0.12 * h4), 0, 1)
            if bronze:
                return np.clip((0.5 * low + 0.25 * high + 0.14) * nz * (1.0 + 0.4 * p.moss), 0, 1)
            return np.clip((0.85 * low + 0.65 * high + 0.05) * nz * (0.7 + 0.8 * p.wear), 0, 1)

        def inner_soot(co, nrm):                                # the inside of the cage is smoked
            rr = np.maximum(np.hypot(co[:, 0], co[:, 1]), 1e-6)
            inward = -(nrm[:, 0] * co[:, 0] + nrm[:, 1] * co[:, 1]) / rr
            inside = (co[:, 2] > zc + 0.02) * (co[:, 2] < zr - 0.004) * (rr < a_c * kn)
            hi_ = np.clip((co[:, 2] - zfl) / Lp, 0, 1)
            return np.clip(np.maximum(inward, -nrm[:, 2]) * 2.0, 0, 1) * inside * (0.7 + 0.3 * hi_)

        def unsooted(fn):                                       # nobody polishes the inside: no rubbed metal under the soot
            return lambda co, nrm: fn(co, nrm) * (1.0 - np.clip(inner_soot(co, nrm) * 1.6, 0, 1))

        for o in metals:
            wf = post_wear if o is posts else (face_wear if o is faces_m else base_wear)
            paint(o, special=metal_special, grime=inner_soot, wear=unsooted(wf))

        def roof_special(co, nrm):
            """Verdigris: pooled along the eaves and the upturned corners, fine streaks washed down
            from the seams, hip rolls and the vent collar; brown oxide elsewhere."""
            z = co[:, 2]
            hfr = np.clip((z - ze) / (top - ze), 0, 1)
            if poly:
                t, d, kk = fall_t(co)
                edge = np.clip(np.abs(t) / np.maximum(d * tn, 1e-3), 0, 1)
            else:
                g_ = 2 * pi / NG
                th = np.arctan2(co[:, 1], co[:, 0]) + pi / 2
                rel = ((th + g_ / 2) % g_) - g_ / 2
                kk = np.round((th - rel) / g_)
                t = rel * np.hypot(co[:, 0], co[:, 1])
                edge = np.clip(np.abs(rel) / (g_ / 2), 0, 1)
            tk = (t if poly else rel * rd, None, kk)                 # one value per sheet column / dome meridian
            fine = streak(co, nrm, 0.02, seed + 3, 0.5, 0.7, tk)
            broad = streak(co, nrm, 0.075, seed + 9, 0.3, 0.7, tk)
            reach = 0.3 + 0.65 * streak(co, nrm, 0.045, seed + 13, 0.25, 0.8, tk)    # how far up each streak starts
            pool = np.clip(1.0 - hfr / (0.09 + 0.1 * broad + 0.2 * edge ** 3), 0, 1) ** 0.6
            run = fine * (0.4 + 0.6 * broad) * np.clip(1.0 - hfr / reach, 0, 1) ** 0.5
            hip = np.clip((edge - 0.9) / 0.1, 0, 1) * np.clip(1.0 - hfr / 0.8, 0, 1)
            collar = np.clip((hfr - 0.85) / 0.15, 0, 1) * fine
            # (the sheet's own oxide texture already streaks; the mask only adds the pooled green)
            return np.clip(np.maximum.reduce([0.6 * pool, 0.22 * run, 0.22 * hip, 0.3 * collar]), 0.05, 0.7)

        def soot_fn(co, nrm):                                   # soot round the vent and under the eaves
            t = np.clip((co[:, 2] - (top - 0.12 * h5)) / (0.12 * h5 + 0.03), 0, 1)
            return np.clip(0.5 * t + 0.55 * np.clip(-nrm[:, 2], 0, 1) + 0.2, 0, 1)

        for o in roofs:
            paint(o, special=roof_special, grime=soot_fn)
        paint(finial, wear=0.3 + 0.5 * p.wear, grime=mask_mul(height_ramp(top - 0.01, top + 0.4 * hf), 0.5))
    else:
        for o in metals:
            paint(o, special=mask_mul(noise_mask(0.03, seed=seed + 21, lo=0.35, hi=0.7), 0.7), wear=0.25 + 0.3 * p.wear)

    # ---- moss: thin feathered carpets grown over the up-facing stone (cap, table wash, foot slab) and
    #      raised cushions in the joints and at the foot; thickest on the damp back (north, +Y) side
    #      and in the lee of the finial. An ochre-olive tone that still reads under the amber flame.
    if p.moss > 0.05:
        rm_ = rng.fork("moss")
        m_moss = mat("plaster_rough", "#646a38", scale=3.0)
        cov = 0.4 * p.moss ** 1.3                                 # ~a third of the up-facing stone at moss 0.95
        zup = np.array([0.0, 0.0, 1.0])

        def roof_z(a):
            return float(np.interp(a, surf_r, surf_z))

        def carpet(name, a_lo, a_hi, zfn, cover, round_=False, lee=0.0, cell=0.022, thick=0.0075):
            nf = NG if round_ else n
            hw_ = tan(pi / nf)
            nu = max(2, int(round(2 * a_hi * hw_ / cell)))
            nv = max(3, int(round((a_hi - a_lo) / min(cell, 0.014 + 0.1 * (a_hi - a_lo)))))    # narrow ledges: finer rows
            NU = nf * nu
            rj = rm_.fork(name)
            P_ = np.zeros((NU, nv + 1, 3))
            for I in range(NU):
                kf_, i = divmod(I, nu)
                for j in range(nv + 1):
                    u = -1.0 + 2.0 * (i + (rj.u(-0.32, 0.32) if i else 0.0)) / nu
                    a = a_lo + (a_hi - a_lo) * (j + (rj.u(-0.32, 0.32) if 0 < j < nv else 0.0)) / nv
                    if round_:
                        th = radians((kf_ + 0.5 * u) * 360.0 / nf)
                        P_[I, j] = (a * sin(th), -a * cos(th), zfn(a))
                    else:
                        P_[I, j] = at_t(a, kf_ * step, u * a * hw_, zfn(a))
            pp = P_.reshape(-1, 3)
            nn_ = np.tile(zup, (len(pp), 1))
            rr = np.maximum(np.hypot(pp[:, 0], pp[:, 1]), 1e-6)
            f = (noise_mask(0.1, seed=seed + 41, lo=0.0, hi=1.0)(pp, nn_) + 0.15 * noise_mask(0.035, seed=seed + 43, lo=0.0, hi=1.0)(pp, nn_)
                 + (0.02 + 0.3 * (1.0 - p.moss)) * pp[:, 1] / rr + lee * np.clip(1.0 - (rr - a_lo) / (0.45 * (a_hi - a_lo)), 0, 1)).reshape(NU, nv + 1)
            core = f[:, 1:-1]                                      # the rim rows stay bare: a feathered edge
            wgt = np.linalg.norm(P_[:, 1:-1, :2], axis=2)          # cells narrow towards the axis: cover by area
            order = np.argsort(-core, axis=None)
            cum = np.cumsum(wgt.reshape(-1)[order]) / wgt.sum()
            thr = core.reshape(-1)[order][min(int(np.searchsorted(cum, min(cover, 0.9))), core.size - 1)]
            mossy = np.zeros((NU, nv + 1), bool)
            mossy[:, 1:-1] = core > thr

            def nbr(m):
                c = np.roll(m, 1, axis=0).astype(int) + np.roll(m, -1, axis=0)
                c[:, 1:] += m[:, :-1]
                c[:, :-1] += m[:, 1:]
                return c

            need = 2 if nv > 2 else 1
            mossy &= nbr(mossy) >= need                            # no stray scraps: moss grows in cushions
            mossy &= nbr(mossy) >= need
            if mossy.sum() < 5:                                    # a lone scrap reads as a leaf: cushions carry it instead
                return None
            deep = mossy & (nbr(mossy) == 4)                       # the middle of a cushion stands thicker than its edge
            mn = mossy | np.roll(mossy, -1, axis=0)                # a cell is covered if any of its corners is mossy
            on = mn[:, :-1] | mn[:, 1:]
            bump = noise_mask(0.024, seed=seed + 45, lo=0.2, hi=0.8)(pp, nn_).reshape(NU, nv + 1)
            hgt = thick * (0.55 + 0.8 * bump) * np.where(deep, 1.6, 0.8)
            vid, verts, faces, gv = {}, [], [], []

            def vx(I, j):
                if (I, j) not in vid:
                    vid[(I, j)] = len(verts)
                    x, y, z = P_[I, j]
                    verts.append((x, y, z + (hgt[I, j] if mossy[I, j] else -0.0015)))
                    gv.append((0.3 + 0.6 * bump[I, j] if deep[I, j] else 0.22) if mossy[I, j] else 0.0)
                return vid[(I, j)]

            for I in range(NU):
                I1 = (I + 1) % NU
                for j in range(nv):
                    if on[I, j]:
                        faces.append((vx(I, j), vx(I, j + 1), vx(I1, j + 1), vx(I1, j)))
            o = _mesh_uv(name, verts, faces, [[(verts[v][0], verts[v][1]) for v in f_] for f_ in faces])
            smooth(o, 80)
            assign(o, m_moss)
            gv = np.array(gv)                                      # damp dark heart, dry pale fringe
            paint(o, grime=lambda co, nrm: gv if len(co) == len(gv) else np.full(len(co), 0.3))
            return o

        carpet("moss_foot", a_b + 0.006, a0 - 0.03, lambda a: h0 - 0.0005, cov * 1.1)
        if ledge_w > 0.026:
            carpet("moss_table", a_w + 0.004, a_t - 0.006, lambda a: zc - wash * clamp((a - a_w) / ledge_w, 0, 1) - 0.0003, cov, thick=0.005)
        if not metal:
            if poly:
                carpet("moss_roof", surf_r[0] * 1.05 + 0.006, surf_r[-1] - 0.014, lambda a: roof_z(a) - 0.0005, cov, lee=0.2)
            else:
                carpet("moss_roof", surf_r[0] * 1.1 + 0.006, rd * 0.93, lambda a: roof_z(a) - 0.0005, cov, round_=True, lee=0.2)

        # cushions: clusters of small round domes in the joints, round the foot and against the finial
        clump = lathe("clump", [(1.0, -0.12), (0.72, 0.6), (0.0, 1.0)], segments=6, smooth_angle=80)
        displace_noise(clump, strength=0.07, scale=0.9, seed=seed)
        levels = [("ground", 2.2), ("joint1", 2.2), ("joint2", 0.9), ("foot", 0.8)]
        if ledge_w > 0.03:
            levels += [("table", 1.0)]
        if not metal:
            levels += [("lee", 1.3), ("roof", 0.8)]
        mats = []
        for _ in range(int(round(p.moss ** 1.25 * 25))):
            ang = 180.0 + rm_.u(-115, 115) if rm_.chance(0.72) else rm_.u(0, 360)
            lvl = rm_.weighted(levels)
            fk = round(ang / step) * step
            slope, radial = 0.0, False
            if lvl == "ground":
                s = rm_.u(0.012, 0.026)
                a, z, sc = a0 + rm_.u(-0.003, 0.01), 0.0, (s * rm_.u(1.0, 1.6), s, s * rm_.u(0.45, 0.75))
            elif lvl == "foot":
                s = rm_.u(0.01, 0.018)
                a, z, sc = rm_.u(a_b + 0.014, a0 - 0.03), h0 - 0.002, (s * rm_.u(1.0, 1.5), s, s * rm_.u(0.45, 0.7))
            elif lvl == "joint1":
                s = rm_.u(0.01, 0.019)
                a, z, sc = a_b + rm_.u(0.0, 0.007), h0 - 0.002, (s * rm_.u(1.0, 1.8), s, s * rm_.u(0.6, 0.95))
            elif lvl == "table":
                s = rm_.u(0.008, 0.013)
                a = rm_.u(a_w + 0.012, a_t - 0.014)
                z, sc = zc - wash * clamp((a - a_w) / ledge_w, 0, 1) - 0.0015, (s * rm_.u(1.0, 1.6), s, s * rm_.u(0.45, 0.7))
            elif lvl == "joint2":
                s = rm_.u(0.007, 0.012)
                a, z, sc = a_s + rm_.u(0.006, 0.012), z2 - 0.002, (s * rm_.u(1.0, 1.8), s, s * rm_.u(0.6, 0.9))
            else:
                s = rm_.u(0.011, 0.02)
                a = surf_r[0] + rm_.u(0.004, 0.02) if lvl == "lee" else rm_.u(surf_r[0] + 0.03, surf_r[-1] * 0.85)
                sc = (s * rm_.u(1.0, 1.6), s, s * rm_.u(0.45, 0.75))
                z = roof_z(a) - 0.0015
                slope = degrees(atan((roof_z(a - 0.006) - roof_z(a + 0.006)) / 0.012))
                radial = not poly
            if radial:
                d, fk = a, ang
            else:
                d = perim(a, ang)
            site = trs((d * sin(radians(ang)), -d * cos(radians(ang)), z), (slope, 0, fk))
            mats.append(site @ trs((0, 0, 0), (0, 0, rm_.u(0, 360)), sc))
            for sgn in rm_.weighted([((1,), 1.0), ((-1, 1), 1.0), ((1, 1), 0.5)]):       # satellites along the joint
                f_ = rm_.u(0.5, 0.8)
                off = (sgn * sc[0] * rm_.u(0.9, 1.9), sc[1] * rm_.u(-0.5, 0.5), 0.0)
                mats.append(site @ trs(off, (0, 0, rm_.u(0, 360)), (sc[0] * f_, sc[1] * f_ * rm_.u(0.9, 1.2), sc[2] * f_)))
        if mats:
            tufts = instances_to_mesh("moss_tufts", clump, mats, keep_base=False)
            assign(tufts, m_moss)
            paint(tufts, grime=lambda co, nrm: np.clip((nrm[:, 2] - 0.35) / 0.5, 0, 1) * (0.25 + 0.5 * noise_mask(0.04, seed=seed + 49, lo=0.3, hi=0.8)(co, nrm)))
        else:
            delete(clump)
