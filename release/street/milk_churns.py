"""A cluster of one to three dairy milk churns left on the paving for the morning cart: spun sheet-steel
bodies (cylindrical with a tapered shoulder, or the older conical railway pattern) standing in a heavy
riveted foot hoop, with a swaged bead, a lapped and riveted shoulder band carrying two forged bar handles
(round bar, both ends flattened into riveted palms), a rolled lip and a loose mushroom lid with a bar grip.
Galvanised churns are dull, oxidised grey sheet with a painted neck band and a riveted brass number plate;
painted churns are enamelled with a stencilled number and chip at the hoops, bead, brim and dents.
One churn stands open (milk showing) with its lid leant against its foot; the bodies are knocked out of
round and really dented from the seed - at high `dents` the foot hoop is kinked and the lid brims bent.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, degrees, atan2, hypot
from mathutils import Matrix, Vector
import bmesh

# three complete riveted vessels in one asset (about 3k each): 9k instead of the 6k prop_small default
META = {"name": "Milk churns", "class": "prop_small", "mount": "ground", "tri_budget": 9000, "focus": 0.36}
PARAMS = {
    "count": {"default": 3, "min": 1, "max": 3, "desc": "number of churns in the group"},
    "height": {"default": 0.64, "min": 0.45, "max": 0.8, "desc": "height of the tallest churn incl. lid (m)"},
    "finish": {"default": "zinc", "choices": ["zinc", "painted", "mixed"], "desc": "galvanised, painted, or alternating"},
    "dents": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "how battered: dent count / depth, out-of-round, kinked hoop, bent lid brims"},
    "shape": {"default": "shouldered", "choices": ["shouldered", "conical"], "desc": "straight body with tapered shoulder, or conical railway churn"},
    "paint": {"default": "#7a2b23", "desc": "body colour (painted) / neck band colour (zinc)"},
    "numbers": {"default": "27 14 8", "desc": "numbers stamped on the churns, space separated, one per churn"},
    "size_mix": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "0 = all the same size, 1 = tall, medium and small"},
    "lid_off": {"default": True, "desc": "front churn stands open with its lid leant against its foot"},
}
VARIANTS = {
    "painted_pair": {"count": 2, "finish": "painted", "paint": "#3d6b57", "height": 0.72, "shape": "conical", "dents": 0.3,
                     "numbers": "5 12", "size_mix": 0.35, "seed": 4},
    "battered_mixed": {"count": 3, "finish": "mixed", "paint": "#cdbf96", "dents": 1.0, "size_mix": 1.0, "height": 0.6,
                       "numbers": "41 9 63", "seed": 11},
    "single_tall": {"count": 1, "height": 0.78, "shape": "conical", "dents": 0.15, "lid_off": False, "numbers": "3",
                    "paint": "#1f3d5c", "seed": 7},
    "blue_small": {"count": 3, "finish": "painted", "paint": "#4d7391", "height": 0.5, "size_mix": 0.0, "dents": 0.6,
                   "lid_off": False, "numbers": "18 19 20", "seed": 23},
}


def _xform(objs, M):
    for o in objs:
        apply_transform(o)
        o.data.transform(M)
        o.data.update()


def _coords(objs):
    out = []
    for o in objs:
        a = np.empty(len(o.data.vertices) * 3, dtype=np.float32)
        o.data.vertices.foreach_get("co", a)
        out.append(a.reshape(-1, 3))
    return np.concatenate(out)


def _lean_text(o):
    """Drop the hidden back faces of a text mesh and merge its over-tessellated outline."""
    apply_transform(o)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
    bm.normal_update()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > 0.5], context='FACES')
    bmesh.ops.dissolve_limit(bm, angle_limit=radians(14), verts=bm.verts[:], edges=bm.edges[:])
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    return o


def _plate(name, w, h, t=0.003, c=0.004):
    """Flat plate in the XZ plane facing -Y with a chamfered border (interior vertices keep the face
    clean so tarnish can be painted into the corners only)."""
    xs = [-w / 2, -w / 2 + c, -w / 6, w / 6, w / 2 - c, w / 2]
    zs = [-h / 2, -h / 2 + c, h / 2 - c, h / 2]
    nx, nz = len(xs), len(zs)
    verts, faces = [], []
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            edge = i in (0, nx - 1) or j in (0, nz - 1)
            verts.append((x, -t / 2 + (0.0015 if edge else 0.0), z))
    for j in range(nz - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.append((a, a + 1, a + nx + 1, a + nx))
    ring = [i for i in range(nx)] + [j * nx + nx - 1 for j in range(1, nz)] + [(nz - 1) * nx + i for i in range(nx - 2, -1, -1)] \
        + [j * nx for j in range(nz - 2, 0, -1)]
    base = len(verts)
    for k in ring:
        verts.append((verts[k][0], t / 2, verts[k][2]))
    m = len(ring)
    for k in range(m):
        faces.append((ring[k], base + k, base + (k + 1) % m, ring[(k + 1) % m]))
    return mesh_object(name, verts, faces)


def _wrap(o, rfn, zc, front, back):
    """Bend a flat part standing in the XZ plane (facing -Y) onto the vessel wall at height zc."""
    apply_transform(o)
    c = _coords([o])
    y0, y1 = float(c[:, 1].min()), float(c[:, 1].max())
    zm = 0.5 * float(c[:, 2].min() + c[:, 2].max())
    xm = 0.5 * float(c[:, 0].min() + c[:, 0].max())
    rc = rfn(zc)

    def f(co):
        z = zc + co.z - zm
        fr = (co.y - y0) / max(y1 - y0, 1e-9)
        r = rfn(z) + front + (back - front) * fr
        a = (co.x - xm) / rc
        return Vector((r * sin(a), -r * cos(a), z))

    transform_verts(o, f)
    return o


def _bar(name, pts, secs, sides=6):
    """Forged bar along a path lying in the XZ plane; every path point has its own (half-width across Y,
    half-thickness in the plane) section, so a round bar can be drawn out into flat palms."""
    n = len(pts)
    Y = Vector((0, 1, 0))
    verts, faces = [], []
    for i in range(n):
        p = Vector(pts[i])
        t = (Vector(pts[min(i + 1, n - 1)]) - Vector(pts[max(i - 1, 0)])).normalized()
        nr = Y.cross(t)
        a, b = secs[i]
        for k in range(sides):
            ang = 2 * pi * k / sides
            verts.append(tuple(p + Y * (a * cos(ang)) + nr * (b * sin(ang))))
    for i in range(n - 1):
        for k in range(sides):
            a0, a1 = i * sides + k, i * sides + (k + 1) % sides
            faces.append((a0, a1, a1 + sides, a0 + sides))
    faces.append(tuple(range(sides - 1, -1, -1)))
    faces.append(tuple(range((n - 1) * sides, n * sides)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, 55)
    return o


# ---- dents: (angle, z, half-width, half-height, depth, wall radius, cos, sin of the dent's own tilt, is a crease)
def _dent_e2(d, a, z):
    u = ((a - d[0] + pi) % (2 * pi) - pi) * d[5]
    w = z - d[1]
    up = (u * d[6] + w * d[7]) / d[2]
    wp = (-u * d[7] + w * d[6]) / d[3]
    return up * up + wp * wp


def _dent_fn(dl, oval, a0):
    def f(co):
        r = hypot(co.x, co.y)
        if r < 1e-5:
            return co
        a = atan2(co.y, co.x)
        d = 0.0
        for dn in dl:
            e = _dent_e2(dn, a, co.z)
            if e < 1.0:
                d += dn[4] * (1.0 - e) ** 0.8
        rr = max(r * (1.0 + oval * cos(2 * (a - a0))) - d, 0.3 * r)
        return Vector((co.x * rr / r, co.y * rr / r, co.z))
    return f


def _refine(o, dl, rfn, rmin):
    """Loop-cut the wall only where it is going to be dented (to about 11 x 15 mm faces) and put the new
    vertices back on the round."""
    if not dl:
        return
    bm = bmesh.new()
    bm.from_mesh(o.data)

    def cut(sub, lim, vertical, minlen):
        es = set()
        for f in bm.faces:
            c = f.calc_center_median()
            if hypot(c.x, c.y) < rmin:
                continue
            a = atan2(c.y, c.x)
            if not any(_dent_e2(d, a, c.z) < lim for d in sub):
                continue
            for e in f.edges:
                v = e.verts[1].co - e.verts[0].co
                if (abs(v.z) > hypot(v.x, v.y)) == vertical and v.length > minlen:
                    es.add(e)
        if es:
            bmesh.ops.subdivide_edges(bm, edges=list(es), cuts=1, use_grid_fill=True)

    cut(dl, 1.25, False, 0.016)
    cut(dl, 1.7, True, 0.036)
    cut(dl, 1.5, True, 0.02)
    for v in bm.verts:
        r = hypot(v.co.x, v.co.y)
        if r > rmin:
            rt = rfn(v.co.z)
            if 1e-6 < rt - r < 0.004:
                v.co.x *= rt / r
                v.co.y *= rt / r
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()


def _lap_seam(o, ang, z0, z1, hw, t, rmin):
    """Cut a vertical strip into the lathed wall at angle `ang` and raise it by t: the lapped side seam."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    d = Vector((cos(ang), sin(ang), 0.0))
    tn = Vector((-sin(ang), cos(ang), 0.0))
    for off in (-hw - 0.003, -hw, hw, hw + 0.003):
        fs = []
        for f in bm.faces:
            c = f.calc_center_median()
            if z0 < c.z < z1 and hypot(c.x, c.y) > rmin and c.dot(d) > 0.8 * hypot(c.x, c.y):
                fs.append(f)
        geom = set(fs)
        for f in fs:
            geom.update(f.edges)
            geom.update(f.verts)
        bmesh.ops.bisect_plane(bm, geom=list(geom), dist=2e-4, plane_co=tn * off, plane_no=tn)
    for v in bm.verts:
        r = hypot(v.co.x, v.co.y)
        if z0 - 1e-4 < v.co.z < z1 + 1e-4 and r > rmin and v.co.dot(d) > 0.8 * r and abs(v.co.dot(tn)) < hw + 3e-4:
            v.co.x *= (r + t) / r
            v.co.y *= (r + t) / r
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()


# ---- vertex-mask helpers (numpy, churn-local space: the vessel axis is the Z axis)
def _wob(k1, p1, k2, p2):
    """Smooth 0..1 variation round the churn; low angular frequency so 24 segments carry it cleanly."""
    def f(co, n):
        a = np.arctan2(co[:, 1], co[:, 0])
        return 0.5 + 0.3 * np.sin(k1 * a + p1) + 0.2 * np.sin(k2 * a + p2)
    return f


def _zin(z0, z1, soft=0.0):
    def f(co, n):
        z = co[:, 2]
        if soft <= 0:
            return ((z >= z0) & (z <= z1)).astype(np.float64)
        return np.clip((z - z0) / soft + 1, 0, 1) * np.clip((z1 - z) / soft + 1, 0, 1)
    return f


def _gt(fn, lo, hi):
    return lambda co, n: np.clip((fn(co, n) - lo) / (hi - lo), 0.0, 1.0)


def _spots(cs, gain):
    def f(co, n):
        m = np.zeros(len(co))
        for c, sig in cs:
            m = np.maximum(m, np.exp(-((co - np.array(c)) ** 2).sum(1) / (sig * sig)))
        return m * gain
    return f


def _near_ang(a0, width):
    def f(co, n):
        a = np.arctan2(co[:, 1], co[:, 0])
        d = np.abs((a - a0 + np.pi) % (2 * np.pi) - np.pi)
        return np.clip(1.0 - d / width, 0.0, 1.0)
    return f


def _rad_gt(r0, r1):
    return lambda co, n: np.clip((np.hypot(co[:, 0], co[:, 1]) - r0) / (r1 - r0), 0.0, 1.0)


def _churn(ci, H, shape, painted, number, is_open, dents, r, face):
    """Build one churn about the origin; returns (body objects, lid objects, info).
    face = angle (rad) by which the number / seam are turned round the body from the local front."""
    Rface = Matrix.Rotation(face, 4, 'Z')
    sd = r.seed + 31 * ci
    s = H / 0.62
    R = 0.27 * H
    zl = H - 0.062 * s                    # lip
    zn = zl - 0.062 * s                    # base of the neck
    if shape == "conical":
        zs, Rb, Rs, rn = zn - 0.065 * s, 1.13 * R, 0.80 * R, 0.57 * R
    else:
        zs, Rb, Rs, rn = zn - 0.125 * s, R, R, 0.58 * R
    tf = 0.0042 * s                       # foot hoop thickness (rolled steel strip)
    tb = 0.0034 * s                       # shoulder band thickness
    ch = 0.0028 * s                       # chamfer on the hoop edges
    zf = 0.038 * s                        # foot hoop height
    zb = zs - 0.010 * s                   # shoulder band top
    za = zb - 0.04 * s                    # shoulder band bottom
    zbead = zf + 0.30 * (zs - zf)

    def wall(z):
        return Rb + (Rs - Rb) * clamp(z / zs)

    # ---- body profile: foot hoop, wall with a swaged bead, shoulder band, shoulder, neck, rolled lip
    prof = [(Rb + tf - 0.002, 0.0), (Rb + tf, 0.003), (wall(zf) + tf, zf - ch), (wall(zf) + 0.0004, zf + 0.0004)]
    wz = [(zf + 0.062 * s, 0.0)]
    z = zf + 0.062 * s + 0.062 * s
    while z < za - 0.03 * s:
        wz.append((z, 0.0))
        z += 0.062 * s
    wz = [q for q in wz if abs(q[0] - zbead) > 0.022 * s]
    wz += [(zbead - 0.012 * s, 0.0), (zbead - 0.0045 * s, 0.0042 * s), (zbead + 0.0045 * s, 0.0042 * s), (zbead + 0.012 * s, 0.0)]
    wz.sort()
    prof += [(wall(z) + dr, z) for z, dr in wz]
    prof += [(wall(za) + 0.0004, za - ch), (wall(za) + tb, za), (wall(zb) + tb, zb), (wall(zb + ch) + 0.0004, zb + ch), (Rs, zs)]
    h = zn - zs
    b0, b1, b2, b3 = (Rs, zs), (Rs, zs + 0.2 * h), (rn + 0.1 * (Rs - rn), zn - 0.34 * h), (rn, zn)
    for k in range(1, 5):
        t = k / 5
        u = 1 - t
        prof.append(tuple(u ** 3 * b0[j] + 3 * u * u * t * b1[j] + 3 * u * t * t * b2[j] + t ** 3 * b3[j] for j in (0, 1)))
    zst0, zst1 = zn, zl - 0.02 * s
    prof += [(rn, zn), (rn + 0.0005 * s, zst1), (rn + 0.005 * s, zl - 0.010 * s), (rn + 0.011 * s, zl - 0.004 * s),
             (rn + 0.009 * s, zl)]
    n_outer = len(prof)
    if is_open:
        prof += [(rn + 0.002, zl), (rn - 0.003, zl - 0.006), (rn - 0.003, zn - 0.004)]
    body = lathe(f"body{ci}", prof, segments=24, smooth_angle=42)
    oz = np.array([q[1] for q in prof[1:n_outer]])
    orr = np.array([q[0] for q in prof[1:n_outer]])

    def body_r(z):
        return float(np.interp(z, oz, orr))

    objs = [body]
    iron = mat("iron_wrought")
    sheet = mat("lead", scale=0.25, wear=0.0, rough=0.75)   # dull oxidised galvanising: matt blue-grey, holds form in shadow
    pnt = mat("iron_painted", r.paint, scale=0.12)
    base = pnt if painted else sheet
    assign(body, base)
    assign_faces(body, iron, lambda c, nrm: c.z < zf + 0.0003 or (za - ch - 1e-4 < c.z < zb + ch + 1e-4))
    if not painted:
        assign_faces(body, pnt, lambda c, nrm: zst0 - 0.001 < c.z < zst1 + 0.001 and hypot(c.x, c.y) > rn - 0.0015)
    elif is_open:                         # tinned inside of the neck
        assign_faces(body, sheet, lambda c, nrm: c.z > zn - 0.01 and hypot(c.x, c.y) < rn - 0.0015)

    a_seam = pi / 2 + face                # lapped side seam up the back
    a_lap = a_seam + 0.42                 # where the shoulder band is lapped
    a_flap = a_seam - 0.55                # where the foot hoop is lapped
    zr = 0.5 * (za + zb)

    def adist(a, b):
        return abs((a - b + pi) % (2 * pi) - pi)

    # ---- side handles: round bar bent to a D, both ends drawn out flat into riveted palms
    zt = zr
    zh = zt - 0.135 * s
    bw, pw, rb_ = 0.0022 * s, 0.0155 * s, 0.0066 * s
    rt, rbm = body_r(zt), body_r(zh)

    def hline(z):
        return rbm + (rt - rbm) * (z - zh) / (zt - zh)

    rm = []                               # rivet matrices
    rivr = 0.0048
    for sg in (-1, 1):
        hp = [(zt + 0.017 * s, bw, (0.010 * s, bw)), (zt + 0.010 * s, bw, (pw, bw)), (zt - 0.010 * s, bw, (pw, bw)),
              (zt - 0.021 * s, bw + 0.003 * s, (0.010 * s, 0.0042 * s)), (zt - 0.036 * s, 0.024 * s, (rb_, rb_)),
              (zt - 0.056 * s, 0.044 * s, (rb_, rb_)),
              (zh + 0.056 * s, 0.044 * s, (rb_, rb_)), (zh + 0.036 * s, 0.024 * s, (rb_, rb_)),
              (zh + 0.021 * s, bw + 0.003 * s, (0.010 * s, 0.0042 * s)), (zh + 0.010 * s, bw, (pw, bw)),
              (zh - 0.010 * s, bw, (pw, bw)), (zh - 0.017 * s, bw, (0.010 * s, bw))]
        hd = _bar(f"handle{ci}", [(sg * (hline(z) + off), 0.0, z) for z, off, _ in hp], [q[2] for q in hp])
        objs.append(hd)
        for zz in (zt + 0.0105 * s, zt - 0.0035 * s, zh - 0.0105 * s, zh + 0.0035 * s):
            rm.append(trs((sg * (hline(zz) + 2 * bw - 0.0006), 0.0, zz), (0, sg * 90, 0), 0.8 * s))

    # ---- hoop rivets (about 60 mm pitch on the band), lap plates with their own pair of rivets
    def hoop(zc, hh, t, a_l, pitch, name):
        rr = body_r(zc)
        nr = max(8, int(round(2 * pi * rr / pitch)))
        a0 = a_l + pi / nr
        for k in range(nr):
            a = a0 + 2 * pi * k / nr
            if adist(a, a_l) < 0.035 * s / rr:
                continue
            if name == "band" and min(adist(a, 0.0), adist(a, pi)) < (pw + 0.007 * s) / rr:
                continue
            rm.append(trs(((rr - 0.0004) * cos(a), (rr - 0.0004) * sin(a), zc), (0, 90, degrees(a)), s))
        lp_ = box(f"lap{ci}", (0.052 * s, 0.004, hh))
        densify(lp_, 0.037 * s)
        _wrap(lp_, body_r, zc, t, -0.0006)
        lp_.data.transform(Matrix.Rotation(a_l + pi / 2, 4, 'Z'))
        for da in (-0.013 * s / rr, 0.013 * s / rr):
            rm.append(trs(((rr + t - 0.0004) * cos(a_l + da), (rr + t - 0.0004) * sin(a_l + da), zc), (0, 90, degrees(a_l + da)), s))
        return lp_

    laps = [hoop(zr, zb - za - 0.004 * s, 0.0028 * s, a_lap, 0.060, "band"),
            hoop(0.5 * zf + 0.001, zf - 0.008 * s, 0.003 * s, a_flap, 0.078, "foot")]
    objs += laps

    # ---- number: stencilled on painted churns, riveted brass plate with raised black-filled digits on zinc
    zc = zf + 0.66 * (za - zf)
    a_num = -pi / 2 + face
    plate = None
    if painted:
        lum = sum(c * w for c, w in zip(hex_rgb(r.paint), (0.2126, 0.7152, 0.0722)))
        sten = mat("iron_painted", "#262220" if lum > 0.3 else "#ddd3b6", scale=0.22)
        tx = _lean_text(text(f"num{ci}", number, size=0.07 * s, depth=0.004, font="sans"))
        _wrap(tx, body_r, zc, 0.0013, -0.002)
        tx.data.transform(Rface)
        assign(tx, sten)
    else:
        pw_, ph_ = 0.118 * s, 0.076 * s
        plate = _plate(f"plate{ci}", pw_, ph_)
        assign(plate, mat("metal_plain", "#b8923f", rough=0.6))
        # tarnish creeps in from the corners only
        paint(plate, grime=lambda co, n: 0.9 * np.clip((np.abs(co[:, 0]) / (pw_ / 2)) * (np.abs(co[:, 2]) / (ph_ / 2)) * 1.5 - 0.45, 0, 1),
              wear=0.0)
        _wrap(plate, body_r, zc, 0.003, -0.001)
        tx = _lean_text(text(f"num{ci}", number, size=max(0.036, 0.043 * s), depth=0.003, font="sans"))
        _wrap(tx, body_r, zc, 0.0052, 0.002)
        for o in (plate, tx):
            o.data.transform(Rface)
        assign(tx, mat("iron_painted", "#191715", scale=0.22))
        objs.append(plate)
        for sx in (-1, 1):
            for sz in (-1, 1):
                a = a_num + sx * (pw_ * 0.5 - 0.009 * s) / body_r(zc)
                z = zc + sz * (ph_ * 0.5 - 0.009 * s)
                rr = body_r(z) + 0.0026
                rm.append(trs((rr * cos(a), rr * sin(a), z), (0, 90, degrees(a)), 0.62 * s))
    objs.append(tx)

    # ---- lapped, riveted side seam up the back: a raised strip cut into the wall itself, hoop to band
    zq0, zq1 = zf + 0.004, za - ch - 0.001
    _lap_seam(body, a_seam, zf + 0.001, za - ch - 0.0002, 0.0085 * s, 0.0024, rn + 0.02 * s)
    for k in range(3):
        z = zq0 + (zq1 - zq0) * (k + 0.5) / 3
        if abs(z - zbead) < 0.018 * s:
            z += 0.026 * s
        rm.append(trs(((body_r(z) + 0.002) * cos(a_seam), (body_r(z) + 0.002) * sin(a_seam), z), (0, 90, degrees(a_seam)), 0.66 * s))

    riv = lathe("rivet", [(rivr, 0.0), (0.0, 0.0036)], segments=5, smooth_angle=80)
    rivets = instances_to_mesh(f"rivets{ci}", riv, rm, keep_base=False)
    objs.append(rivets)
    for o in objs[1:]:
        if o not in (plate, tx):
            assign(o, iron)

    # ---- milk in the open churn
    if is_open:
        mk = cylinder(f"milk{ci}", rn - 0.0026, 0.006, loc=(0, 0, zl - 0.03 * s), verts=16)
        assign(mk, mat("ceramic_glazed", "#f3eedf", scale=0.2, rough=0.5))
        objs.append(mk)

    # ---- dents and out-of-round (one field applied to everything fixed to the body)
    dr = r.rng.fork(f"dents{ci}")
    dl = []
    nd = int(round(1.8 + dents * 3.2 + dr.u(-0.5, 0.5))) if dents > 0.02 else 0
    tries = 0
    while len(dl) < nd and tries < 240:
        tries += 1
        a = dr.u(-pi, pi)
        if len(dl) < 2 and tries < 80:    # the first two sit either side of the number, where they are seen
            a = a_num + (1 if len(dl) else -1) * dr.u(0.7, 1.25)
        crease = len(dl) % 3 == 1         # every third one is a narrow crease from a knock against an edge
        if crease:
            sa, sz = dr.u(0.017, 0.023) * s, dr.u(0.05, 0.075) * s
        else:
            sa, sz = dr.u(0.028, 0.046) * s, dr.u(0.03, 0.058) * s
        rmax = max(sa, sz)
        if za - zf - 0.008 - 2 * rmax < 0.01:
            continue
        z = dr.u(zf + 0.004 + rmax, za - 0.004 - rmax)
        rd = body_r(z)
        if adist(a, a_num) * rd < 0.07 * s + rmax and abs(z - zc) < 0.045 * s + rmax:
            continue
        if min(adist(a, 0.0), adist(a, pi)) * rd < 0.022 * s + rmax and z + rmax > zh - 0.022 * s:
            continue
        if adist(a, a_seam) * rd < 0.012 * s + rmax:
            continue
        if any(hypot(adist(a, d[0]) * rd, z - d[1]) < 0.85 * (rmax + max(d[2], d[3])) for d in dl):
            continue
        tw = dr.u(-1.3, 1.3) if crease else dr.u(-0.9, 0.9)
        ce2 = (sa * cos(tw)) ** 2 + (sz * sin(tw)) ** 2        # half-extent round the wall, squared
        # deep enough to go past flat (c^2 / 2R) into a real hollow
        dep = ce2 / (2 * rd) + (dr.u(0.007, 0.011) if crease else dr.u(0.009, 0.015)) * s * (0.7 + 0.8 * dents)
        dl.append((a, z, sa, sz, dep, rd, cos(tw), sin(tw), crease))
    kink = clamp((dents - 0.55) / 0.45)
    if kink > 0:                          # foot hoop knocked in where the churn was dropped on its rim
        a = dr.u(-pi, pi)
        if adist(a, a_flap) < 0.5:
            a += 1.0
        dl.append((a, zf * 0.35, dr.u(0.05, 0.065) * s, zf * 1.9, (0.014 + 0.016 * kink) * s, body_r(zf * 0.5), 1.0, 0.0, False))
    _refine(body, dl, body_r, rn + 0.02 * s)
    fn = _dent_fn(dl, 0.012 * dents * dr.u(0.4, 1.0), dr.u(0, pi))
    for o in objs:
        apply_transform(o)
        transform_verts(o, fn)
    smooth(body, 62)
    smooth(rivets, 80)

    # ---- mushroom lid with a bar grip
    rl = rn + 0.026 * s
    lp = []
    rp = rn - 0.005
    if is_open:
        lp += [(0.0, zl - 0.05 * s), (rp - 0.004, zl - 0.05 * s), (rp, zl - 0.045 * s), (rp, zl + 0.002), (rl - 0.0025, zl + 0.002)]
    else:
        lp += [(rl - 0.0025, zl - 0.004 * s)]
    lp += [(rl - 0.0025, zl - 0.013 * s), (rl + 0.0005, zl - 0.016 * s), (rl + 0.0036, zl - 0.012 * s), (rl + 0.0036, zl - 0.005 * s),
           (rl + 0.001, zl + 0.004 * s), (rl - 0.01 * s, zl + 0.017 * s), (rl * 0.74, zl + 0.029 * s), (rl * 0.4, zl + 0.037 * s),
           (0.0, zl + 0.043 * s)]
    lid = lathe(f"lid{ci}", lp, segments=20 if is_open else 16, smooth_angle=42)
    uv_box(lid)
    xh = rl * 0.27
    z0 = zl + 0.041 * s
    grip = sweep(f"grip{ci}", bezier_path((-xh, 0, z0 - 0.003), (-xh * 1.2, 0, z0 + 0.027 * s), (xh * 1.2, 0, z0 + 0.027 * s), (xh, 0, z0 - 0.003), n=7),
                 0.006 * s, sides=6, smooth_path=False, up=(0, 1, 0))
    assign(lid, base)
    if painted and is_open:               # tinned plug and underside
        assign_faces(lid, sheet, lambda c, nrm: c.z < zl + 0.003 and hypot(c.x, c.y) < rl - 0.004)
    assign(grip, iron)
    lids = [lid, grip]
    for o in lids:
        apply_transform(o)
    lr = r.rng.fork(f"lid{ci}")
    bend = clamp((dents - 0.45) / 0.55)
    ab = lr.u(-pi, pi)
    if bend > 0:                          # brim bent up where the lid was levered off / dropped
        def fb(co):
            rr = hypot(co.x, co.y)
            if rr < 0.62 * rl or co.z > zl + 0.03 * s:
                return co
            w = max(0.0, cos(min(adist(atan2(co.y, co.x), ab) / 0.75, 1.0) * pi / 2)) ** 2 * (rr - 0.62 * rl) / (0.38 * rl)
            k = 1.0 - 0.10 * bend * w
            return Vector((co.x * k, co.y * k, co.z + (0.006 + 0.012 * bend) * s * w))
        transform_verts(lid, fb)

    # ---- art-directed weathering, painted in churn-local space (the vessel axis is Z)
    w1, w2, w3 = _wob(2, sd * 1.3, 3, sd * 0.7), _wob(3, sd * 2.1, 4, sd * 0.37), _wob(1, sd * 0.9, 4, sd * 1.9)
    hoops = lambda co, n: np.maximum(_zin(0.0, zf + 0.0006)(co, n), _zin(za - 0.0005, zb + 0.0005)(co, n))
    dspot = [((d[5] - d[4]) * cos(d[0]), (d[5] - d[4]) * sin(d[0]), d[1]) for d in dl]
    # splash-back from the wet paving: fades up as an irregular soft gradient
    paint(body, grime=lambda co, n: 0.62 * np.clip(1.0 - co[:, 2] / (0.115 * s * (0.55 + 0.9 * w1(co, n))), 0.0, 1.0) ** 1.4)
    # rain runs under the shoulder band and dust on the shoulder
    paint(body, grime=mask_mul(_zin(za - 0.2 * s, za, 0.12 * s), _gt(w2, 0.35, 0.9), 0.3))
    paint(body, grime=mask_mul(_zin(zs, zn, 0.02 * s), facing((0, 0, 1), 1.5), 0.22))
    # hoops: dark steel, bright where it is rubbed, rust in the laps and low on the foot
    paint(body, wear=mask_mul(hoops, lambda co, n: 0.2 + 0.32 * w3(co, n)),
          special=mask_mul(hoops, lambda co, n: 0.45 * np.maximum(0.5 * _gt(w2, 0.45, 0.95)(co, n),
                                                           np.maximum(_near_ang(a_lap, 0.5)(co, n) * (co[:, 2] > zf * 2), _near_ang(a_flap, 0.45)(co, n) * (co[:, 2] < zf * 2)))))
    paint(body, special=mask_mul(_zin(0.0, zf * 0.5), 0.35))
    if dspot:                             # dirt settles in the hollows
        paint(body, grime=_spots([(c, 0.5 * (d[2] * d[3]) ** 0.5) for c, d in zip(dspot, dl)], 0.3))
        up = []                           # ... and under the upper lip of each dent, where the rain never washes
        for d in dl:
            ce, ve = hypot(d[2] * d[6], d[3] * d[7]), hypot(d[2] * d[7], d[3] * d[6])
            rr = d[5] - 0.6 * d[4]
            up.append(((rr * cos(d[0]), rr * sin(d[0]), d[1] + 0.42 * ve), 0.48 * max(ce, 0.6 * ve)))
        paint(body, grime=_spots(up, 0.62))
    if painted:
        # chips where the wear is - top of the foot hoop, bead crest, band edges, lip, dent crowns - rust runs below
        paint(body, wear=mask_mul(_zin(zf - ch - 0.001, zf + 0.012), _gt(w2, 0.3, 0.75), 0.75),
              special=mask_mul(_zin(zf, zf + 0.085 * s, 0.02 * s), lambda co, n: np.clip(1.0 - (co[:, 2] - zf) / (0.08 * s), 0, 1), _gt(w1, 0.2, 0.8), 0.3))
        paint(body, wear=mask_mul(_zin(zbead - 0.006 * s, zbead + 0.006 * s), _gt(w3, 0.4, 0.8), 0.65),
              special=mask_mul(_zin(zbead - 0.014 * s, zbead - 0.006 * s), _gt(w3, 0.35, 0.8), 0.5))
        paint(body, wear=mask_mul(_zin(za - ch - 0.001, za - ch + 0.001), _gt(w1, 0.35, 0.8), 0.6),
              special=mask_mul(_zin(za - ch - 0.001, za - ch + 0.001), _gt(w1, 0.25, 0.8), 0.55))
        paint(body, wear=mask_mul(_zin(zl - 0.012 * s, zl + 0.001), _rad_gt(rn + 0.003 * s, rn + 0.009 * s), _gt(w2, 0.25, 0.7), 0.6))
        if dspot:
            paint(body, wear=_spots([(c, 0.55 * min(d[2], d[3])) for c, d in zip(dspot, dl)], 0.8),
                  special=_spots([((c[0], c[1], c[2] - 0.7 * d[3]), 0.55 * max(d[2], d[3])) for c, d in zip(dspot, dl)], 0.34),
                  grime=_spots([((c[0], c[1], c[2] - 0.7 * d[3]), 0.6 * max(d[2], d[3])) for c, d in zip(dspot, dl)], 0.3))
    else:
        paint(body, wear=mask_mul(_zin(zst0 - 0.001, zst1 + 0.001), _gt(w3, 0.45, 0.9), 0.5))
    for o in objs[1:]:
        nm = o.name
        if nm.startswith("handle"):
            paint(o, wear=lambda co, n: 0.3 + 0.5 * np.clip(1.0 - np.abs(co[:, 2] - 0.5 * (zt + zh)) / (0.05 * s), 0, 1),
                  special=lambda co, n: 0.3 * np.clip((np.abs(co[:, 2] - 0.5 * (zt + zh)) - 0.045 * s) / (0.03 * s), 0, 1))
        elif nm.startswith("lap"):
            paint(o, wear=0.3, special=0.12)
        elif nm.startswith("rivets"):
            paint(o, wear=0.7, special=mask_mul(_gt(w2, 0.5, 0.9), 0.45))
        elif nm.startswith("num") and painted:
            paint(o, wear=mask_mul(noise_mask(0.03, seed=sd + 14, lo=0.5, hi=0.8), 0.6))
    if painted:
        paint(lid, wear=mask_mul(_rad_gt(rl - 0.004, rl + 0.002), _gt(w1, 0.25, 0.7), 0.75),
              special=mask_mul(_rad_gt(rl * 0.8, rl), _gt(w1, 0.3, 0.8), 0.4))
        paint(lid, wear=mask_mul(_rad_gt(rl * 0.45, rl * 0.3), _zin(zl + 0.03 * s, zl + 0.05 * s), 0.5))
    else:
        paint(lid, grime=mask_mul(_rad_gt(rl * 0.9, rl * 0.5), _zin(zl + 0.02 * s, zl + 0.05 * s), 0.25))
    paint(grip, wear=0.75, special=0.1)

    info = {"s": s, "rl": rl, "zl": zl, "foot": Rb + tf, "oz": oz, "orr": orr}
    if not is_open:
        tl = lr.u(0.3, 1.3) * (0.3 + dents)
        Ml = (Matrix.Translation((0, 0, zl)) @ Matrix.Rotation(radians(lr.u(0, 360)), 4, 'Z') @ Matrix.Rotation(radians(tl), 4, 'X')
              @ Matrix.Rotation(radians(lr.u(0, 360)), 4, 'Z') @ Matrix.Translation((0, 0, -zl + 0.0012 * tl)))
        _xform(lids, Ml)
    return objs, lids, info


def build(p, rng):
    n = int(p.count)
    mix = p.size_mix
    H = p.height
    hs = ([H], [H, H * (1 - 0.2 * mix)], [H, H * (1 - 0.13 * mix), H * (1 - 0.27 * mix)])[n - 1]
    numbers = str(p.numbers).split() or ["7"]
    if p.finish == "mixed":
        painted = [(i + int(p.seed)) % 2 == 1 for i in range(n)]
        if n > 1 and len(set(painted)) == 1:
            painted[0] = not painted[0]
    else:
        painted = [p.finish == "painted"] * n
    foot = [0.27 * h * (1.13 if p.shape == "conical" else 1.0) + 0.0042 * h / 0.62 for h in hs]

    # ---- layout: tallest at the back, smallest in front, shoulders nearly touching
    lr = rng.fork("layout")
    cs = [np.array([0.0, 0.0])]
    if n >= 2:
        a = radians(lr.u(-28, -14) if n == 2 else lr.u(0, 16))
        cs.append(cs[0] + (foot[0] + foot[1] + lr.u(0.012, 0.04)) * np.array([cos(a), sin(a)]))
    if n >= 3:
        ra, rb = foot[0] + foot[2] + lr.u(0.012, 0.035), foot[1] + foot[2] + lr.u(0.012, 0.035)
        dv = cs[1] - cs[0]
        d = float(np.linalg.norm(dv))
        ex = dv / d
        x = (ra * ra - rb * rb + d * d) / (2 * d)
        y = max(ra * ra - x * x, 0.0) ** 0.5
        ey = np.array([ex[1], -ex[0]])              # the front (-Y) side of the pair
        cs.append(cs[0] + ex * x + ey * y)
    io = n - 1 if p.lid_off else -1
    lean_off = lr.u(54, 68)
    lean_side = lr.pick([-1, 1])
    lean_tilt = lr.u(60, 65)

    class _R:                                       # what a churn needs from the asset level
        pass
    R_ = _R()
    R_.paint = p.paint
    R_.rng = rng
    R_.seed = int(p.seed)

    def turn(nb):
        """Turn a churn so its two handles point away from the neighbours (and the leaning lid)."""
        best, bs = 0.0, -1e9
        for k in range(19):
            rz = radians(-45 + 5 * k)
            sc = min([abs((rz + off - b + pi) % (2 * pi) - pi) for b in nb for off in (0.0, pi)] + [radians(75)])
            sc -= 0.3 * abs(rz)
            if sc > bs:
                best, bs = rz, sc
        return best, bs

    allobjs, built = [], []
    for i in range(n):
        nb = [atan2(*(cs[j] - cs[i])[::-1]) for j in range(n) if j != i]
        lid_dir = 0.0
        if i == io:
            opts = []
            for sg in (lean_side, -lean_side):
                ld = radians(-90 + sg * lean_off)
                clear = min([abs((ld - b + pi) % (2 * pi) - pi) for b in nb] + [pi])
                best, bs = turn(nb + [ld])
                opts.append((min(bs, 0.6 * clear), best, ld))
            _, best, lid_dir = max(opts, key=lambda q: q[0])
        else:
            best, _ = turn(nb)
        best += radians(lr.u(-4, 4))
        M = Matrix.Translation((cs[i][0], cs[i][1], 0)) @ Matrix.Rotation(best, 4, 'Z')
        objs, lids, info = _churn(i, hs[i], p.shape, painted[i], numbers[i % len(numbers)], i == io, p.dents, R_,
                                  -0.85 * best + radians(lr.u(-8, 8)))
        _xform(objs, M)
        if i == io:
            # lid leant against the churn: brim on the paving, grip / brim against the wall, clean plug side out
            zl = info["zl"]
            Mr = (Matrix.Rotation(lid_dir, 4, 'Z') @ Matrix.Rotation(radians(lean_tilt + 180), 4, 'Y')
                  @ Matrix.Rotation(radians(lr.u(0, 360)), 4, 'Z') @ Matrix.Translation((0, 0, -zl)))
            _xform(lids, Mr)
            c = _coords(lids)
            zmin = float(c[:, 2].min())
            dv = np.array([cos(lid_dir), sin(lid_dir)])
            dist = info["foot"] + 0.4
            for _ in range(8):
                q = c[:, :2] + dv * dist
                gap = float((np.hypot(q[:, 0], q[:, 1]) - np.interp(c[:, 2] - zmin, info["oz"], info["orr"])).min())
                dist -= gap - 0.0012
            _xform(lids, Matrix.Translation((cs[i][0] + dv[0] * dist, cs[i][1] + dv[1] * dist, -zmin)))
        else:
            _xform(lids, M)
        info["M"] = M
        built.append(info)
        allobjs += objs + lids

    # ---- origin at the centre of the footprint
    c = _coords(allobjs)
    ctr = 0.5 * (c.min(0) + c.max(0))
    T = Matrix.Translation((-float(ctr[0]), -float(ctr[1]), 0.0))
    _xform(allobjs, T)

    top = max(built, key=lambda b: b["zl"])
    if built.index(top) != io:
        tp = T @ top["M"] @ Vector((0, 0, top["zl"] + 0.068 * top["s"]))
        socket("top", "attach", (tp.x, tp.y, tp.z), role="prop_top")
