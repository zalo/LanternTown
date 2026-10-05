"""Cast-iron street bollard, one hollow sand casting: stepped (octagonal / square / round) pedestal with a
cast date on the front face and lozenge bosses on the others, broach-stopped up to a torus-and-scotia shaft
base, a tapering shaft (spoon-ended flutes, plain cannon-barrel with reinforce rings, or octagonal), a belt
carrying forged chain eyes with a loose ring / length of chain, a moulded neck and a cap (cannon muzzle with a
ball in the bore, ball finial, or acorn). Set with a caulked run-lead joint into a two-piece granite setting
stone that lies nearly flush with the paving. Paint failure is real geometry: the casting is cut along a
corrosion field into paint / scaled-rust zones (contours refined as far as the triangle budget allows), with
rust-bleed halos and run-down streaks painted into the masks around them; road dust lies on the pedestal and
the ledges; gilding on the beads is cut through where it has rubbed off.
"""
import bmesh
import bpy
import numpy as np
from tk import *

META = {"name": "Cast-iron bollard", "class": "prop_small", "mount": "ground", "focus": 0.72, "lod": True}
PARAMS = {
    "height": {"default": 0.95, "min": 0.6, "max": 1.3, "desc": "overall height above the paving (m)"},
    "girth": {"default": 1.0, "min": 0.8, "max": 1.3, "desc": "thickness multiplier (1 = 168 mm shaft foot)"},
    "style": {"default": "fluted", "choices": ["fluted", "plain", "octagonal"], "desc": "shaft style"},
    "flutes": {"default": 10, "min": 6, "max": 12, "desc": "number of flutes (fluted style)"},
    "cap": {"default": "cannon", "choices": ["cannon", "ball", "acorn"], "desc": "cap / finial"},
    "base": {"default": "octagonal", "choices": ["octagonal", "square", "round"], "desc": "pedestal plan"},
    "paint": {"default": "#628c79", "desc": "paint colour of the casting"},
    "trim": {"default": "gilt", "choices": ["gilt", "painted", "none"], "desc": "picked-out beads, date and bosses"},
    "trim_paint": {"default": "#d6ceb8", "desc": "trim colour when trim = painted"},
    "rust": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "rust / paint failure amount"},
    "chain_eyes": {"default": 2, "choices": [0, 1, 2], "desc": "number of chain eyes (0 none, 1 right, 2 both sides)"},
    "chain_links": {"default": 1, "min": 0, "max": 6, "desc": "links left hanging from each eye (1 = a loose ring)"},
    "mark": {"default": "1887", "desc": "cast date / foundry mark on the pedestal front (empty = none)"},
    "pad": {"default": 0.008, "min": 0.0, "max": 0.12, "desc": "how far the granite setting stone stands proud of the paving (0 = set straight in the paving)"},
    "lean": {"default": 1.0, "min": 0.0, "max": 6.0, "desc": "lean from cart knocks (deg); direction from the seed"},
}
VARIANTS = {
    "plain_ball": {"style": "plain", "cap": "ball", "base": "round", "height": 0.8, "girth": 1.18, "paint": "#646c7a",
                   "trim": "none", "chain_eyes": 0, "mark": "", "rust": 0.2, "lean": 0.5, "pad": 0.03, "seed": 4},
    "octagon_acorn": {"style": "octagonal", "cap": "acorn", "base": "square", "height": 1.08, "paint": "#7a1f1c",
                      "trim": "painted", "chain_eyes": 1, "chain_links": 5, "mark": "VII", "rust": 0.25, "seed": 9},
    "tall_blue": {"style": "fluted", "flutes": 12, "cap": "ball", "height": 1.25, "girth": 0.86, "paint": "#3f66a0",
                  "mark": "1902", "rust": 0.12, "lean": 0.0, "seed": 15},
    "rusty_leaning": {"style": "plain", "cap": "cannon", "base": "octagonal", "height": 0.88, "paint": "#8a9a86",
                      "trim": "none", "rust": 1.0, "lean": 5.5, "pad": 0.0, "chain_eyes": 2, "chain_links": 2,
                      "mark": "1851", "seed": 23},
}

T_RUST = 0.6            # corrosion field value where the paint has gone and the iron is scaled
TRI_TARGET = 5850       # prop_small budget is 6000 (the cheaper levels: 39 % / 9.5 % of it, see build())


def _arc(rc, zc, rad, a0, a1, n, kz=1.0):
    """Arc in the (radius, z) plane around (rc, zc); angles in degrees from +radius towards +z."""
    return [(rc + rad * cos(radians(a0 + (a1 - a0) * i / n)), zc + kz * rad * sin(radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _mix_hex(a, b, t):
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(pa, pb))


def _polygonize(obj, n, segs, wfn, rot_deg=0.0, crn=0.3):
    """Turn a lathe (built with `segs` segments) into an n-gon with eased corners; profile radius =
    distance to the flats. wfn(z) -> 0 (round) .. 1 (polygon)."""
    m = segs // n
    da = 2 * pi / segs
    h = pi / n
    rot = radians(rot_deg)

    def f(co):
        r = hypot(co.x, co.y)
        if r < 1e-6:
            return co
        th = atan2(co.y, co.x)
        k = int(round(th / da)) % segs
        j = k % m
        if j == 0:
            off = 0.0
            rp = 0.5 * (1.0 / cos(h) + cos(da * crn) / cos(h - da * crn))
        else:
            off = da * crn if j == 1 else (m - crn) * da if j == m - 1 else j * da
            rp = 1.0 / cos(off - h)
        w = wfn(co.z)
        rr = r * (1.0 + (rp - 1.0) * w)
        ang = (k - j) * da + (j * da) * (1 - w) + off * w + rot
        return Vector((rr * cos(ang), rr * sin(ang), co.z))

    transform_verts(obj, f)


def _lighten_text(o, angle=11.0):
    """Drop the hidden back faces of raised lettering and dissolve the over-dense font curves."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > 0.5], context='FACES')
    bmesh.ops.dissolve_limit(bm, angle_limit=radians(angle), verts=bm.verts[:], edges=bm.edges[:])
    bm.to_mesh(o.data)
    bm.free()


def _hanging_chain(name, top, n, ll, wire, rng):
    """n oval links hanging straight down from `top` (x, y, z = inside top of the first link)."""
    hl, hw = ll / 2 - wire, ll * 0.3 - wire
    pts = [(hl * cos(2 * pi * i / 8) * (1.2 if i % 4 else 1.0), hw * sin(2 * pi * i / 8) * (1.0 if i % 2 else 1.08), 0.0) for i in range(8)]
    base = sweep(name + "_link", pts, wire, closed=True, sides=4, smooth_path=False)
    pitch = ll - 4 * wire + 0.001
    ms = []
    for i in range(n):
        ms.append(trs((top[0] + rng.u(-0.0015, 0.0015), top[1] + rng.u(-0.0015, 0.0015), top[2] - hl - i * pitch),
                      (90 + rng.u(-7, 7), 90 + rng.u(-7, 7), (i % 2) * 90 + rng.u(-12, 12))))
    o = instances_to_mesh(name, base, ms, keep_base=False, uv_jitter=False)
    smooth(o, 180)
    return o


def _lathe(name, profile, lid=False, **kw):
    """tk.lathe, but with the normals verified: the automatic recalculation can turn a short open
    piece (a waisted neck, a bead) inside out, so make the faces point away from the axis.
    lid (far level): the piece is closed across its top. Every turned piece gets its own number of sides at a
    cheaper level, so where one stands on the next the two rings no longer meet and one looks down into the lower
    piece between them."""
    if lid:
        profile = list(profile) + [(0.0, profile[-1][1])]
    o = lathe(name, profile, **kw)
    me = o.data
    tot = sum((q.normal.x * q.center.x + q.normal.y * q.center.y) * q.area for q in me.polygons)
    if tot < 0:
        me.flip_normals()
    return o


def _tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def _refine(o, field, lo, hi, min_edge, passes, step=0.02):
    """Adaptive tessellation: faces through which `field` passes the lo..hi band get their edges longer
    than min_edge split, so a contour of the field has vertices to follow (the surface is unchanged)."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    for _ in range(passes):
        faces = bm.faces[:]
        pts, start = [], []
        for f in faces:
            start.append(len(pts))
            vs = [vv.co for vv in f.verts]
            c = f.calc_center_median()
            pts.append(c)
            for i in range(len(vs)):
                a, b = vs[i], vs[(i + 1) % len(vs)]
                k = max(1, min(8, int((b - a).length / step)))
                for q in range(k):
                    pt = a.lerp(b, q / k)
                    pts.append(pt)
                    pts.append(pt.lerp(c, 0.55))
        vals = field(np.array([(q.x, q.y, q.z) for q in pts], dtype=np.float64))
        mx = np.maximum.reduceat(vals, start)
        mn = np.minimum.reduceat(vals, start)
        edges = set()
        for f, a, b in zip(faces, mn, mx):
            if b > lo and a < hi:
                for e in f.edges:
                    if e.calc_length() > min_edge:
                        edges.add(e)
        if not edges:
            break
        bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=1, use_grid_fill=True)
    bm.to_mesh(o.data)
    bm.free()


def _zone(o, field, thr, slot, cut=True):
    """Cut the mesh along the contour field == thr (marching over triangles) and give the faces on the
    high side material slot `slot`. Returns the number of faces in the zone.
    cut=False (the cheaper levels' small parts): no cut, whole faces go by their mean value."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    vs = bm.verts[:]
    if not vs:
        bm.free()
        return 0
    vals = field(np.array([(q.co.x, q.co.y, q.co.z) for q in vs], dtype=np.float64))
    vals = np.where(np.abs(vals - thr) < 1e-5, thr + 1e-5, vals)
    val = {q: float(x) for q, x in zip(vs, vals)}
    fs = [f for f in bm.faces if min(val[q] for q in f.verts) < thr < max(val[q] for q in f.verts)]
    if not cut:
        n = 0
        for f in bm.faces:
            if sum(val[q] for q in f.verts) / len(f.verts) >= thr:
                f.material_index = slot
                n += 1
        bm.to_mesh(o.data)
        bm.free()
        return n
    if fs:
        tri = bmesh.ops.triangulate(bm, faces=fs)["faces"]
        edges = {e for f in tri for e in f.edges if (val[e.verts[0]] - thr) * (val[e.verts[1]] - thr) < 0}
        new = set()
        for e in edges:
            a, b = e.verts
            fac = min(max((thr - val[a]) / (val[b] - val[a]), 0.1), 0.9)
            _, nv = bmesh.utils.edge_split(e, a, fac)
            val[nv] = thr
            new.add(nv)
        for f in bm.faces[:]:
            nv = [q for q in f.verts if q in new]
            if len(nv) == 2:
                bmesh.utils.face_split(f, nv[0], nv[1])
    n = 0
    for f in bm.faces:
        if min(val[q] for q in f.verts) >= thr and max(val[q] for q in f.verts) > thr:
            f.material_index = slot
            n += 1
    bm.to_mesh(o.data)
    bm.free()
    return n


def build(p, rng):
    H = p.height
    g = p.girth
    v = g * clamp(H / 0.95, 0.72, 1.12)             # vertical scale of the mouldings
    zb = p.pad if p.pad > 0.003 else 0.0
    R = p.rust
    seed = int(p.seed)
    TEX = 0.26                                       # many coats over a sand casting: fine 2-4 mm orange peel
    body_m = mat("iron_painted", p.paint, scale=TEX, rough=1.15, grime=0.6, wear=0.3)
    dust_m = mat("iron_painted", _mix_hex(p.paint, "#a2977f", 0.36), scale=TEX, rough=2.3, grime=0.45, wear=0.3)
    rust_m = mat("iron_rusty", scale=0.8)
    if p.trim == "gilt":
        trim_m = mat("metal_plain", "#e8c877", rough=0.5)
        flat_m = mat("iron_painted", "#f4cc66", scale=TEX, rough=0.75, grime=0.5, wear=0.3)   # gold size on the flat work
    elif p.trim == "painted":
        trim_m = mat("iron_painted", p.trim_paint, scale=TEX, rough=1.25, grime=0.6, wear=0.3)
    else:
        trim_m = body_m
    if p.trim != "gilt":
        flat_m = trim_m
    forged_m = mat("iron_wrought")
    body, trim, forged = [], [], []
    zoned = []                                       # axis-centred castings that get the corrosion cut

    nside = {"octagonal": 8, "square": 4, "round": 0}[p.base]
    ap_foot, ap_die = 0.140 * g, 0.120 * g
    if nside == 4:
        ap_foot, ap_die = 0.128 * g, 0.108 * g
    r_corner = ap_foot / (cos(pi / nside) if nside else 1.0)
    fixed_tris = 0                                   # everything that is not the casting

    # ---- granite setting stone (two pieces, joint under the casting), nearly flush, and the lead joint
    if zb > 0:
        stone = mat("stone_rough", "#8b867d", rough=1.25)
        rs = rng.fork("pad")
        half = r_corner + 0.10
        a_r = min(0.007, zb * 0.75)
        for sy in (-1, 1):
            hx, hy = half + rs.u(-0.006, 0.01), half + rs.u(-0.008, 0.008)
            zt = zb + (0.0 if sy < 0 else rs.u(-0.0015, 0.001))
            tilt = rs.u(-0.006, 0.006)
            rings = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-a_r, 0.0, 1.0), (-0.293 * a_r, -0.293 * a_r, 1.0)]
            rings.append((0.0, -a_r, 1.0) if zt - a_r > 0.004 else (0.0, -zt, 1.0))
            if zt - a_r > 0.004:
                rings.append((0.0, -zt, 1.0))
            base_r = [ap_foot * 0.8, r_corner + 0.034]
            cols = [(0.0, 0.002, 0.016), (0.0, 0.002, 0.0035), (0.0, 0.0075, 0.0)]
            cols += [(22.5 * i, 0.0, 0.0) for i in range(1, 8)]
            cols += [(180.0, 0.0075, 0.0), (180.0, 0.002, 0.0035), (180.0, 0.002, 0.016)]
            verts, faces = [], []
            for th, shift, drop in cols:
                c_, s_ = cos(radians(th)), sy * sin(radians(th))
                ro = 1.0 / ((abs(c_) / hx) ** 9 + (abs(s_) / hy) ** 9) ** (1.0 / 9)
                for ri, (dr, dz, outer) in enumerate(rings):
                    r = (ro if outer else base_r[ri]) + dr
                    x, y = r * c_, r * s_ + sy * shift
                    z = max(zt + dz - drop + tilt * x + (rs.u(-0.0006, 0.0006) if ri in (1, 2) else 0.0), 0.0)
                    if dz <= -zt + 1e-9:
                        z = 0.0
                    verts.append((x, y, z))
            nr = len(rings)
            ck, rk = list(range(len(cols))), list(range(nr))
            if lod() == 1:                               # mid: every other column, the arris as one chamfer
                ck, rk = [0, 2, 4, 6, 8, 10, 12], [0, 1, 2, nr - 1]
            elif lod() == 2:                             # far: a slab with four corners, top + skirt
                ck, rk = [2, 4, 6, 8, 10], [0, 2, nr - 1]
            for ca, cb in zip(ck, ck[1:]):
                for ra, rb in zip(rk, rk[1:]):
                    faces.append((ca * nr + ra, cb * nr + ra, cb * nr + rb, ca * nr + rb))
            if lod():                                    # (the vertices are all computed: same draws from `rs`)
                used = sorted({i for f in faces for i in f})
                remap = {i: k for k, i in enumerate(used)}
                verts = [verts[i] for i in used]
                faces = [tuple(remap[i] for i in f) for f in faces]
            st = mesh_object("setting_stone", verts, faces)
            if sum(q.normal.z * q.area for q in st.data.polygons) < 0:
                st.data.flip_normals()
            smooth(st, 48)
            assign(st, stone)
            fixed_tris += _tris(st)
            paint(st, grime=lambda co, n: np.clip(1.5 - np.hypot(co[:, 0], co[:, 1]) / (r_corner * 1.25), 0, 1) * 0.45,
                  special=mask_mul(noise_mask(0.15, seed=seed + 5, lo=0.45, hi=0.75),
                                   lambda co, n: np.clip(0.3 + co[:, 1] * 3.0, 0, 1), 0.6))
        lsegs = (lod_pick(32, nside, nside) if nside else 32)       # _polygonize needs a multiple of nside
        lead = _lathe("lead_joint", [(ap_foot + 0.017, zb - 0.0025), (ap_foot + 0.0115, zb + 0.0022),
                                    (ap_foot + 0.004, zb + 0.0046), (ap_foot - 0.004, zb + 0.003)], segments=lsegs, smooth_angle=60)
        if nside:
            _polygonize(lead, nside, lsegs, lambda z: 1.0, 180.0 / nside)
        rl = rng.fork("lead")
        ph = [rl.u(0, 6.28) for _ in range(3)]

        def caulk(co):                               # run lead dressed with a caulking iron: uneven lip, tool dents
            th = atan2(co.y, co.x)
            r = hypot(co.x, co.y)
            t = clamp((co.z - (zb - 0.0025)) / 0.0065)
            k = 1.0 + (1.0 - t) * (0.0045 * sin(3 * th + ph[0]) + 0.003 * sin(7 * th + ph[1]) + rl.u(-0.0025, 0.0025)) / r
            dz = 0.0 if t < 0.2 else rl.u(-0.0014, 0.0008) + 0.0006 * sin(11 * th + ph[2])
            return Vector((co.x * k, co.y * k, co.z + dz))

        transform_verts(lead, caulk)
        assign(lead, mat("lead", rough=1.2))
        paint(lead, grime=lambda co, n: 0.44 + 0.3 * noise_mask(0.03, seed=seed + 9, lo=0.3, hi=0.7)(co, n))
        lod_drop(lead, above=1)                          # a 2 cm lip of lead: nothing at the far level
        fixed_tris += _tris(lead)

    # ---- pedestal: stepped foot, die, cornice, broach up to the round torus / scotia shaft base
    def Z(z):
        return zb + z * v

    prof = [(ap_foot, Z(0.0)), (ap_foot, Z(0.034)), (ap_foot - 0.006 * g, Z(0.041)), (ap_die, Z(0.045)),
            (ap_die, Z(0.150)), (ap_die + 0.008 * g, Z(0.158)), (ap_die + 0.009 * g, Z(0.168)),
            (ap_die + 0.004 * g, Z(0.175)), (0.104 * g, Z(0.201))]
    psegs = (lod_pick(40, 2 * nside, nside) if nside else 40)       # mid: crisp corners, a round top for the shaft base
    far = lod() == 2
    ped = _lathe("pedestal", prof, lid=far, segments=psegs, smooth_angle=30)
    if nside:
        _polygonize(ped, nside, psegs, lambda z: 1.0 - smoothstep(Z(0.176), Z(0.200), z), 180.0 / nside,
                    crn=lod_pick(0.3, 0.0, 0.0))
    body.append(ped)
    prof = [(0.104 * g, Z(0.200)), (0.104 * g, Z(0.211))] + _arc(0.098 * g, Z(0.226), 0.014 * g, -50, 62, 3, kz=v / g)
    prof += [(0.093 * g, Z(0.243)), (0.0885 * g, Z(0.251)), (0.092 * g, Z(0.260)), (0.092 * g, Z(0.267)), (0.083 * g, Z(0.275))]
    sbase = _lathe("shaft_base", prof, lid=far, segments=24, smooth_angle=42)
    body.append(sbase)
    zoned += [ped, sbase]

    n_wear = noise_mask(0.05, seed=seed + 11, lo=0.35, hi=0.65)
    ka = radians(rng.fork("knock").u(0, 360))            # side the carts catch
    t_rub = 0.74 - 0.3 * R                               # gilding is rubbed through where f_rub passes this

    def f_rub(co):
        x, y = co[:, 0], co[:, 1]
        side = np.clip((x * cos(ka) + y * sin(ka)) / np.maximum(np.hypot(x, y), 1e-4), 0, 1)
        return n_wear(co, co) + 0.3 * side

    # lozenge bosses on the die faces (date on the front face)
    zd = Z(0.099)
    la, lb, ld = 0.026 * g, 0.036 * v, 0.006
    if lod() < 2:
        loz = mesh_object("lozenge", [(la, 0, 0), (0, lb, 0), (-la, 0, 0), (0, -lb, 0),
                                      (la * 0.45, 0, ld), (0, lb * 0.45, ld), (-la * 0.45, 0, ld), (0, -lb * 0.45, ld)],
                          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)])
    else:                                                # far: the gilt diamond as one flat face
        loz = mesh_object("lozenge", [(la, 0, 0.002), (0, lb, 0.002), (-la, 0, 0.002), (0, -lb, 0.002)], [(0, 1, 2, 3)])
    nl = nside if nside else 6
    lm = []
    for k in range(nl):
        if k == 0 and p.mark:
            continue
        a = radians(k * 360.0 / nl)
        lm.append(trs((ap_die * sin(a), -ap_die * cos(a), zd), (90, 0, degrees(a))))
    lozs = None
    if lm:
        lozs = instances_to_mesh("lozenges", loz, lm, keep_base=False, uv_jitter=False)
        trim.append(lozs)
    else:
        delete(loz)
    if p.mark and lod() < 2:                             # (the date is 3 cm tall: gone at the far level)
        tx = text("mark", str(p.mark), size=0.034 * g, depth=0.0045, loc=(0, 0, 0), font="serif")
        apply_transform(tx)
        _lighten_text(tx, 14.0)
        if _tris(tx) > 420:                                  # long marks: coarser letter outlines
            _lighten_text(tx, 24.0)
        lo, hi = world_bbox([tx])
        face_w = (2 * ap_die * tan(pi / nside) if nside else ap_die * 1.1) * 0.8
        s = min(1.0, face_w / max(hi.x - lo.x, 1e-4), 0.085 * v / max(hi.z - lo.z, 1e-4))
        cx, cz = (lo.x + hi.x) / 2, (lo.z + hi.z) / 2
        ymax = hi.y if lod() == 0 else lo.y + 0.0045         # (the cheaper levels letter with a flat face)

        def seat(co):
            x, d, z = (co.x - cx) * s, (ymax - co.y), (co.z - cz) * s        # d = height above the face
            if nside:
                return Vector((x, -(ap_die - 0.0008 + d), zd + z))
            rr, th = ap_die - 0.0008 + d, x / ap_die
            return Vector((rr * sin(th), -rr * cos(th), zd + z))

        transform_verts(tx, seat)
        trim.append(tx)

    # ---- shaft
    zs0 = Z(0.274)
    plain = p.style == "plain"
    fluted = p.style == "fluted"
    cap_h = {"cannon": 0.140 * v + 0.055 * g, "ball": 0.074 * v + 0.147 * g, "acorn": 0.234 * v}[p.cap]
    zs1 = H - cap_h
    r0, r1 = 0.084 * g, 0.064 * g
    Ls = zs1 - zs0
    if plain:
        r0, r1 = 0.088 * g, 0.062 * g

    def rad(z):
        t = clamp((z - zs0) / Ls)
        return r0 + (r1 - r0) * t + 0.004 * g * sin(pi * t)

    zc = zs1 - 0.050 * v                                  # belt centre (chain eyes)
    bh = 0.024 * v                                        # belt half height
    fz0, fz1 = zs0 + 0.030 * v, zc - bh - 0.022 * v
    e = min(0.026 * g, (fz1 - fz0) * 0.2)
    e_lo = min(2.2 * e, (fz1 - fz0) * 0.3) if fluted else e     # the lower flute end runs out long (sheds water)
    zs = [zs0, fz0, fz0 + 0.4 * e_lo, fz0 + e_lo]
    nmid = max(2, int((fz1 - fz0 - e - e_lo) / (0.2 if fluted else 0.075)))
    zs += [fz0 + e_lo + (fz1 - fz0 - e - e_lo) * i / nmid for i in range(1, nmid)]
    zs += [fz1 - e, fz1 - 0.4 * e, fz1, zs1]
    if lod() == 1:                                        # mid: flute ends as straight run-outs
        zs = [zs0, fz0, fz0 + e_lo, fz1 - e, fz1, zs1] if not plain else [zs0, (zs0 + zs1) / 2, zs1]
    elif lod() == 2:                                      # far: a plain tapering drum
        zs = [zs0, (zs0 + zs1) / 2, zs1]
    nf = int(p.flutes)
    D = min(0.0135 * g, 2 * pi * r1 / nf * 0.36)          # flute depth
    if fluted:
        npf = lod_pick(6, 4, 1)                             # points across a flute: 6 / 4 / none (far: an nf-gon)
        segs = nf * npf
        with lod_as(0):                                     # the rings and segments above are the ones wanted
            shaft = _lathe("shaft", [(rad(z), z) for z in zs], lid=far, segments=segs, smooth_angle=lod_pick(22, 22, 50))
        da = 2 * pi / segs
        offs = [0.0, 0.135, 0.255, 0.5, 0.745, 0.865]       # crisp arris, steep wall, round bottom
        deps = [0.0, 0.0, 0.8, 1.0, 0.8, 0.0]
        if npf == 4:                                        # mid: flat land, steep walls, flat bottom
            offs, deps = [0.135, 0.31, 0.69, 0.865], [0.0, 1.0, 1.0, 0.0]
        elif npf == 1:
            offs, deps = [0.0], [0.3]                       # (the mean radius of the fluted section)

        def flute(co):
            r = hypot(co.x, co.y)
            k = int(round(atan2(co.y, co.x) / da)) % segs
            j = k % npf
            sdn = clamp((co.z - fz0) / e_lo)
            sup = clamp((fz1 - co.z) / e)
            en = sqrt(1 - (1 - sdn) ** 2) * sqrt(1 - (1 - sup) ** 2)
            off = offs[j] if j == 0 and npf != 4 else 0.5 + (offs[j] - 0.5) * (0.5 + 0.5 * en)   # flutes narrow to a round end
            ang = ((k - j) + off * npf) * da - pi / 2 - 0.5 * npf * da                 # a flute centred on the front
            rr = r - D * deps[j] * en
            return Vector((rr * cos(ang), rr * sin(ang), co.z))

        transform_verts(shaft, flute)
    elif p.style == "octagonal":
        osegs = lod_pick(40, 24, 8)
        with lod_as(0):
            shaft = _lathe("shaft", [(rad(z) * 0.985, z) for z in zs], lid=far, segments=osegs, smooth_angle=30)
        _polygonize(shaft, 8, osegs, lambda z: smoothstep(fz0, fz0 + e, z) * (1 - smoothstep(fz1 - e, fz1, z)), 22.5,
                    crn=lod_pick(0.3, 0.3, 0.0))
    else:
        shaft = _lathe("shaft", [(rad(z), z) for z in zs], lid=far, segments=32, smooth_angle=40)
    body.append(shaft)
    zoned.append(shaft)

    def bead(name, z, rr, hh, out=None, n=3, segs=24, gilt=False):
        out = hh if out is None else out
        o = _lathe(name, [(rr - 0.001 + out * cos(radians(a)), z + hh * sin(radians(a))) for a in [-90 + 180 * i / n for i in range(n + 1)]],
                   segments=segs, smooth_angle=65)
        if gilt and p.trim == "gilt":                      # gilding rubbed through to the paint on the proud edge
            assign(o, trim_m)
            o.data.materials.append(body_m)
            _zone(o, f_rub, t_rub, 1, cut=lod() == 0)
        return o

    # reinforce rings on the plain (cannon barrel) shaft, a waist band on the octagonal one
    drip_src = []                                         # (z, strength): mouldings that shed rusty water
    if plain:
        for t, hh in ((0.30, 0.009), (0.36, 0.005), (0.62, 0.007)):
            z = fz0 + (fz1 - fz0) * t
            body.append(bead("reinforce", z, rad(z), hh * v, n=3 if hh > 0.006 else 2, segs=24))
            if hh > 0.006:
                drip_src.append((z - hh * v, 0.75))
    elif p.style == "octagonal":
        z = fz0 + (fz1 - fz0) * 0.5
        wsegs = lod_pick(40, 24, 8)
        wb = _lathe("waist", [(rad(z) - 0.002, z - 0.022 * v), (rad(z) + 0.005 * g, z - 0.016 * v), (rad(z) + 0.005 * g, z + 0.016 * v),
                             (rad(z) - 0.002, z + 0.022 * v)], segments=wsegs, smooth_angle=30)
        _polygonize(wb, 8, wsegs, lambda zz: 1.0, 22.5, crn=lod_pick(0.3, 0.3, 0.0))
        body.append(wb)
        drip_src.append((z - 0.02 * v, 0.75))

    # belt under the neck: flat band between two picked-out beads
    rc = rad(zc)
    belt = _lathe("belt", [(rc - 0.002, zc - bh - 0.006 * v), (rc + 0.0045 * g, zc - bh), (rc + 0.0045 * g, zc + bh),
                           (r1 + 0.0015, zs1 - 0.007 * v)], segments=28, smooth_angle=35)
    body.append(belt)
    drip_src.append((zc - bh - 0.004 * v, 1.0))
    for sz in (-1, 1):
        if lod() < 2:                                     # (5 mm beads: the belt alone at the far level)
            trim.append(bead("belt_bead", zc + sz * bh * 0.72, rc + 0.004 * g, 0.0048 * v, n=2, gilt=True))

    # ---- neck and cap
    def ZN(z):
        return zs1 + z * v

    trim.append(bead("astragal", ZN(0.004), r1 + 0.002, 0.0105 * v, n=3, segs=24, gilt=True))
    rn = r1 - 0.013 * g
    neck = [(r1 + 0.001, ZN(0.010)), (r1 - 0.001, ZN(0.017)), (r1 - 0.005 * g, ZN(0.024)), (r1 - 0.011 * g, ZN(0.034)), (rn, ZN(0.046))]
    cs = 24
    caps = []
    if p.cap == "cannon":
        prof = neck + [(rn + 0.002 * g, ZN(0.066)), (rn + 0.009 * g, ZN(0.087)), (rn + 0.020 * g, ZN(0.105)),
                       (rn + 0.031 * g, ZN(0.116)), (rn + 0.036 * g, ZN(0.118)), (rn + 0.038 * g, ZN(0.126)),
                       (rn + 0.035 * g, ZN(0.136)), (rn + 0.027 * g, ZN(0.1425)), (rn + 0.011 * g, ZN(0.144)), (rn + 0.005 * g, ZN(0.137))]
        caps.append(_lathe("muzzle", prof, lid=far, segments=cs, smooth_angle=50))
        if lod() < 2:
            trim.append(bead("muzzle_ring", ZN(0.060), rn + 0.0015 * g, 0.0048 * v, n=2, segs=cs, gilt=True))
        zcb = ZN(0.140)
        rb = 0.055 * g
        caps.append(_lathe("shot", [(rb * cos(radians(a)), zcb + rb * sin(radians(a))) for a in (-6, 13, 31, 48, 64, 78)] + [(0, zcb + rb)],
                           segments=20, smooth_angle=60))
    elif p.cap == "ball":
        rs_ = rn - 0.002 * g
        rb = 0.076 * g
        zcb = ZN(0.074) + sqrt(rb * rb - rs_ * rs_)
        a0 = degrees(atan2(-sqrt(rb * rb - rs_ * rs_), rs_))
        prof = neck + [(rn, ZN(0.052)), (rn + 0.006 * g, ZN(0.056)), (rn + 0.009 * g, ZN(0.063)), (rn + 0.006 * g, ZN(0.070))]
        prof += _arc(0, zcb, rb, a0, 78, 8)
        zt = zcb + rb * sin(radians(78))
        prof += [(0.0175 * g, zt + 0.003 * g), (0.017 * g, zt + 0.008 * g), (0.012 * g, zt + 0.013 * g), (0, zt + 0.0155 * g)]
        caps.append(_lathe("ball", prof, segments=cs, smooth_angle=50))
        if lod() < 2:
            trim.append(bead("ball_collar", ZN(0.063), rn + 0.008 * g, 0.0045 * v, n=2, segs=cs, gilt=True))
    else:
        caps.append(_lathe("acorn_neck", neck + [(rn, ZN(0.052)), (rn + 0.005 * g, ZN(0.056))], lid=far, segments=cs, smooth_angle=50))
        cz = [0.055, 0.061, 0.070, 0.081, 0.093, 0.104, 0.111, 0.113]
        cr = [rn + 0.004 * g, 0.063 * g, 0.074 * g, 0.081 * g, 0.0835 * g, 0.082 * g, 0.078 * g, 0.070 * g]
        nb = 12
        cup = _lathe("acorn_cup", [(r, ZN(z)) for r, z in zip(cr, cz)], lid=far, segments=nb * 3, smooth_angle=20)

        def scales(co):
            zz = (co.z - zs1) / v
            i = min(range(len(cz)), key=lambda q: abs(cz[q] - zz))
            if i < 1 or i > 5:
                return co
            k = int(round(atan2(co.y, co.x) / (2 * pi / (nb * 3)))) % 3
            ph_ = (k + (1.5 if i % 2 else 0)) % 3
            b = 0.0042 * g * (1.0 if ph_ < 0.4 else (-0.15 if ph_ > 2.4 or ph_ < 1.2 else -0.3))
            r = hypot(co.x, co.y)
            return Vector((co.x * (r + b) / r, co.y * (r + b) / r, co.z + (0.003 * v if ph_ < 0.4 else 0.0)))

        transform_verts(cup, scales)
        trim.append(cup)
        nut = [(0.069 * g, 0.110), (0.073 * g, 0.122), (0.072 * g, 0.142), (0.066 * g, 0.162), (0.055 * g, 0.181),
               (0.039 * g, 0.199), (0.021 * g, 0.212), (0.0095 * g, 0.219), (0.007 * g, 0.227), (0, 0.234)]
        caps.append(_lathe("acorn_nut", [(r, ZN(z)) for r, z in nut], segments=cs, smooth_angle=50))
    body += caps
    if R > 0.55:
        zoned += caps + [belt]

    # ---- chain eyes (forged eye bolts through bosses on the belt) with a loose ring or a few links
    xe = rc + 0.0045 * g + 0.022
    sides = [1, -1][:int(p.chain_eyes)]
    rch = rng.fork("chain")
    for sx in sides:
        if lod() < 2:
            body.append(_lathe("eye_boss", [(0.021, -0.004), (0.021, 0.003), (0.015, 0.008), (0.009, 0.016)], segments=10,
                               loc=(sx * (rc + 0.003 * g), 0, zc), rot=(0, sx * 90, 0), smooth_angle=50))
        forged.append(torus("eye", 0.0165, 0.0055, loc=(sx * (xe + 0.004), 0, zc), rot=(90, 0, 0), major_seg=12, minor_seg=4))
        nlk = int(p.chain_links)
        if nlk == 1:
            forged.append(torus("ring", 0.024, 0.0048, loc=(sx * (xe + 0.006), 0, zc - 0.006 - 0.024),
                                rot=(rch.u(-14, 14), 90 + rch.u(-10, 10), 0), major_seg=12, minor_seg=4))
        elif nlk > 1:
            top = (sx * (xe + 0.0045), 0.0, zc - 0.0165 + 0.0055 + 0.0046)
            nlk = min(nlk, int((top[2] - 0.02) / 0.03))
            forged.append(_hanging_chain("links", top, nlk, 0.05, 0.0046, rch))

    # ---- corrosion field (un-leaned asset space): where it passes T_RUST the paint is gone
    Zr = np.zeros((1, 3))
    n_big = noise_mask(0.17, seed=seed + 3, lo=0.25, hi=0.75)
    n_mid = noise_mask(0.085, seed=seed + 7, lo=0.3, hi=0.72)
    n_all = noise_mask(0.21, seed=seed + 12, lo=0.3, hi=0.72)
    n_chip = noise_mask(0.03, seed=seed + 4, lo=0.45, hi=0.7)
    on = float(smoothstep(0.0, 0.1, R))
    hr = (0.035 + 0.2 * R) * max(v, 0.8)                  # how high the ground-line rust climbs
    zk = min(0.37, zs0 + 0.45 * Ls)                       # hub height of the carts that knock it
    fh = [((sin((m + 1) * 12.9898 + seed * 78.233) * 43758.5453) % 1.0) for m in range(nf)]
    fh = np.array(fh)

    def f_rust(co):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        r = np.maximum(np.hypot(x, y), 1e-4)
        B, M = n_big(co, Zr), n_mid(co, Zr)
        hgt = np.clip(1.0 - (z - zb) / hr, 0, 1)
        s = hgt ** 0.7 * (0.3 + 0.7 * B) * (0.8 + 0.4 * M) * (0.62 + 0.75 * R)
        side = 0.3 + 0.7 * np.clip((x * cos(ka) + y * sin(ka)) / r, 0, 1) ** 0.6
        s = np.maximum(s, np.exp(-((z - zk) / 0.12) ** 2) * side * n_chip(co, Zr) ** 1.3 * (0.85 + 0.5 * R))
        if R > 0.45:
            s = np.maximum(s, (R - 0.45) * 1.75 * (0.2 + 0.8 * n_all(co, Zr)) * (0.85 + 0.3 * M))
        if fluted:                                         # water stands in the lower spoon ends
            th = np.arctan2(y, x)
            m = np.round((th + pi / 2) / (2 * pi / nf)).astype(int) % nf
            t = np.clip((z - zs0) / Ls, 0, 1)
            depth = (r0 + (r1 - r0) * t + 0.004 * g * np.sin(pi * t)) - r
            pool = np.clip(depth / (0.45 * D), 0, 1) * np.clip(1.0 - (z - fz0) / (1.3 * e_lo), 0, 1) ** 0.6 * (z > fz0 - 0.004)
            s = np.maximum(s, pool * fh[m] * (0.55 + 0.6 * R))
        return s * on

    for o in body:
        assign(o, body_m)
    for o in trim:
        if len(o.data.materials) == 0:
            assign(o, flat_m if o.name.startswith(("lozenge", "mark")) else trim_m)
    if lozs is not None and p.trim == "gilt":
        hl_ = pi / nl

        def f_rub_loz(co):                                # the proud faces of the bosses go first
            x, y = co[:, 0], co[:, 1]
            d = np.hypot(x, y) * (np.cos(((np.arctan2(x, -y) + hl_) % (2 * hl_)) - hl_) if nside else 1.0)
            return f_rub(co) + 0.4 * np.clip((d - ap_die) / ld, 0, 1) - 0.22

        lozs.data.materials.append(body_m)
        _zone(lozs, f_rub_loz, t_rub, 1, cut=lod() == 0)
    for o in forged:
        assign(o, forged_m)
    for o in zoned:
        o.data.materials.append(dust_m)
        o.data.materials.append(rust_m)
    for o in (ped, sbase):                                # road dust: the die up to its cornice, and the ledges
        for q in o.data.polygons:
            if q.normal.z > 0.42 or q.center.z < Z(0.152):
                q.material_index = 1
    if R > 0.0:                                           # small parts: whole faces go where the field says so
        for o in body + trim:
            if o in zoned:
                continue
            cs_ = np.array([tuple(o.matrix_basis @ q.center) for q in o.data.polygons], dtype=np.float64)
            hit = f_rust(cs_) >= T_RUST
            if hit.any():
                o.data.materials.append(rust_m)
                k = len(o.data.materials) - 1
                for q, h_ in zip(o.data.polygons, hit):
                    if h_:
                        q.material_index = k
    # cut the rust zones as finely as the triangle budget allows
    if R > 0.0:
        keep = [o.data.copy() for o in zoned]
        target = int(lod_budget() * lod_pick(0.97, 0.97, 0.85)) if lod_budget() else int(TRI_TARGET * lod_pick(1.0, 0.39, 0.095))
        for lv in (0.014, 0.017, 0.02, 0.024, 0.028, 0.034, 0.04, 0.05, 0.06, 0.075, 0.09, None, "faces"):
            for o, me in zip(zoned, keep):
                old = o.data
                o.data = me.copy()
                if old not in keep:
                    bpy.data.meshes.remove(old)
                if lv and lv != "faces":
                    _refine(o, f_rust, T_RUST - 0.035, T_RUST + 0.035, lv * (0.8 if o is ped else 0.85 if o is shaft else 1.15), 4)
                _zone(o, f_rust, T_RUST, 2, cut=lv != "faces" or lod() == 0)    # last resort of a cheaper level: whole faces
            tot = fixed_tris + sum(_tris(o) for o in body + trim + forged)
            if tot <= target:
                break
        for me in keep:
            bpy.data.meshes.remove(me)

    def vkey(c):                                          # position hash of (un-leaned) vertices
        q = np.round(c * 4000.0).astype(np.int64)
        return (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)

    rv = [(0.0, 0.0, -1.0)]                               # vertices of the rust zones (their rim carries the halo)
    for o in zoned:
        me = o.data
        for q in me.polygons:
            if q.material_index == 2:
                rv += [tuple(me.vertices[i].co) for i in q.vertices]
    rust_keys = np.unique(vkey(np.array(rv, dtype=np.float64)))

    # ---- lean (a shear, so the foot stays flat in its lead joint), weathering masks
    la_ = radians(rng.fork("lean").u(0, 360))
    tl = tan(radians(p.lean))
    kx, ky = tl * cos(la_), tl * sin(la_)

    def lean(co):
        d = max(co.z - zb, 0.0)
        return Vector((co.x + kx * d, co.y + ky * d, co.z))

    def unlean(co):
        d = np.maximum(co[:, 2] - zb, 0.0)
        c = co.copy()
        c[:, 0] -= kx * d
        c[:, 1] -= ky * d
        return c

    sph = [rng.fork("streak").u(0, 6.28) for _ in range(3)]
    eye_th = [0.0 if sx > 0 else pi for sx in sides]

    def streaks(c):
        """Rust-water runs below the mouldings and chain eyes (tapering down the surface)."""
        z = c[:, 2]
        th = np.arctan2(c[:, 1], c[:, 0])
        col = np.clip((np.sin(11 * th + sph[0]) * np.sin(17 * th + sph[1]) + 0.3 * np.sin(5 * th + sph[2]) - 0.28) / 0.3, 0, 1)
        out = np.zeros(len(z))
        for zsrc, k in drip_src:
            dz = zsrc - z
            out = np.maximum(out, col * k * np.exp(-np.maximum(dz, 0) / (0.035 + 0.02 * R)) * (dz > -0.02))
        for a in eye_th:
            dth = np.abs(((th - a + pi) % (2 * pi)) - pi)
            dz = (zc - 0.012) - z
            out = np.maximum(out, np.exp(-(dth / 0.16) ** 2) * np.exp(-np.maximum(dz, 0) / (0.05 + 0.03 * R)) * _ss(-0.01, 0.006, dz) * 1.3)
        return out * (0.45 + 0.25 * R) * on

    def m_special(co, n):
        c = unlean(co)
        s = f_rust(c)
        inr = np.isin(vkey(c), rust_keys)                 # the cut is where the mesh says it is, not the field
        s = np.where(inr, np.maximum(s, T_RUST), np.minimum(s, T_RUST - 1e-3))
        halo = np.where(inr, 0.36 - 0.26 * _ss(T_RUST, T_RUST + 0.18, s), 0.36 * np.clip((s - 0.42) / (T_RUST - 0.42), 0, 1) ** 1.5)
        ledge = np.clip(n[:, 2], 0, 1) ** 2 * (0.15 + 0.5 * n_big(c, Zr)) * R * 0.9
        run = np.zeros(len(s))                            # rust water running down from the chips above
        for k in (1, 2, 3):
            up = c.copy()
            up[:, 2] += 0.014 * k
            run = np.maximum(run, _ss(T_RUST - 0.1, T_RUST + 0.05, f_rust(up)) * (1.0 - 0.22 * k))
        run = np.where(inr, 0.0, 0.55 * run)
        return np.clip(np.maximum.reduce([halo, run, streaks(c), ledge]), 0, 1)

    def m_grime(co, n):
        c = unlean(co)
        s = f_rust(c)
        core = _ss(0.7, 1.0, s) * 0.5 + 0.14 * _ss(0.45, T_RUST, s)
        under = np.clip(-n[:, 2], 0, 1) * 0.35 * (0.4 + 0.6 * n_big(c, Zr))
        stain = 0.4 * _ss(0.1, 0.3, m_special(co, n))      # rust-stained paint is brown, not red lead
        return np.clip(np.maximum.reduce([core, under, stain]), 0, 1)

    def m_wear(co, n):
        c = unlean(co)
        z = c[:, 2]
        side = np.clip(n[:, 0] * cos(ka) + n[:, 1] * sin(ka), 0, 1)
        return np.clip(np.exp(-((z - zk) / 0.12) ** 2) * side * n_chip(c, Zr) * (0.25 + 0.3 * R), 0, 1)

    for grp, nm in ((body, "casting"), (trim, "trim"), (forged, "chainwork")):
        if not grp:
            continue
        j = join(grp, nm)
        transform_verts(j, lean)
        if nm == "chainwork":
            paint(j, wear=0.3, special=lambda co, n: np.clip(0.2 + R * 0.6, 0, 1) * noise_mask(0.02, seed=seed + 6, lo=0.3, hi=0.7)(co, n))
        elif nm == "trim":
            paint(j, grime=lambda co, n: m_grime(co, n) * 0.6, wear=0.0, special=lambda co, n: m_special(co, n) * (0.0 if p.trim == "gilt" else 0.8))
        else:
            paint(j, grime=m_grime, wear=m_wear, special=m_special)

    for sx, nm in ((1, "chain_r"), (-1, "chain_l"))[:int(p.chain_eyes)]:
        d = zc - zb
        socket(nm, "attach", (sx * (xe + 0.004) + kx * d, ky * d, zc), rot=(0, 0, sx * 90), role="chain")
