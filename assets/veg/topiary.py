"""Clipped box (Buxus) topiary in a pot: a bark stem carrying a lumpy, shear-clipped foliage volume as a
ball, cone, helical spiral or two-tier standard. The clipped surface is built like the real thing: a dark
displaced hedge shell, a tight under-layer of small (12-19 mm) box leaves lying on it, and short sprigs of
opposite rounded leaves standing proud for the fuzzy outline, with whole shoots of paler new growth in
patches. It grows from mounded soil in a thrown terracotta festoon pot on pot feet (salt bloom under the
collar, algae on the shaded side, damp foot, chipped rim), a cast-lead cistern planter with strapwork
panels and a date, or a painted timber Versailles case with iron straps. Standards are tied to a bamboo
cane with twine; ivy underplanting creeps over the rim and clings to the pot wall.
"""
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector
from tk import *

META = {"name": "Clipped topiary in pot", "class": "vegetation", "mount": "ground", "focus": 0.38}
PARAMS = {
    "shape": {"default": "two_tier", "choices": ["ball", "cone", "spiral", "two_tier"], "desc": "clipped form"},
    "height": {"default": 1.75, "min": 0.7, "max": 2.6, "desc": "overall height incl. pot (m)"},
    "pot": {"default": "terracotta", "choices": ["terracotta", "lead", "versailles"],
            "desc": "terracotta festoon pot / square cast-lead planter / painted timber Versailles case"},
    "pot_scale": {"default": 1.0, "min": 0.8, "max": 1.35, "desc": "pot size relative to the plant"},
    "girth": {"default": 1.0, "min": 0.8, "max": 1.25, "desc": "fullness (width) of the clipped form"},
    "turns": {"default": 3.0, "min": 2.0, "max": 4.5, "desc": "number of turns of the spiral form"},
    "overgrowth": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "0 = freshly clipped, 1 = overdue (shaggy, long new shoots)"},
    "foliage": {"default": "#ffffff", "desc": "foliage tint (multiplies the box green)"},
    "pot_tint": {"default": "#d8b5a0", "desc": "terracotta tint"},
    "pot_age": {"default": 0.55, "min": 0.0, "max": 1.0, "desc": "age of the terracotta: salt bloom, tide line, damp foot, rim chips"},
    "paint": {"default": "#9dbba2", "desc": "paint colour of the Versailles case"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "algae / moss on the pot and soil"},
    "cane": {"default": True, "desc": "bamboo support cane tied to the stem (standards)"},
    "date": {"default": "1768", "desc": "raised date cast on the lead planter ('' = none)"},
    "ivy": {"default": 4, "min": 0, "max": 7, "desc": "strands of ivy underplanting creeping over the rim"},
    "label": {"default": True, "desc": "wooden plant label stuck in the soil"},
    "litter": {"default": 14, "min": 0, "max": 40, "desc": "fallen clippings on the soil and around the pot"},
}
VARIANTS = {
    "ball_lead": {"shape": "ball", "height": 1.0, "pot": "lead", "pot_scale": 1.2, "overgrowth": 0.15, "ivy": 0, "seed": 4},
    "cone_versailles": {"shape": "cone", "height": 1.9, "pot": "versailles", "girth": 1.05, "overgrowth": 0.2, "seed": 9},
    "spiral_terracotta": {"shape": "spiral", "height": 2.1, "pot": "terracotta", "turns": 3.0, "moss": 0.6, "pot_age": 0.8,
                          "overgrowth": 0.15, "seed": 13},
    "shaggy_ball": {"shape": "ball", "height": 1.35, "pot": "terracotta", "overgrowth": 0.95, "girth": 1.1, "pot_scale": 1.15,
                    "foliage": "#e4f0b4", "pot_tint": "#d9c8ba", "moss": 0.9, "pot_age": 1.0, "litter": 34, "ivy": 6,
                    "label": False, "seed": 21},
}

# leaf templates in units of leaf length: verts (x along the blade, y across, z up) and faces
_PENT_V = np.array([(0.0, 0.0, 0.0), (0.42, -0.3, 0.06), (0.93, -0.17, 0.0), (0.93, 0.17, 0.0), (0.42, 0.3, 0.06)])
_PENT_F = [(0, 1, 4), (1, 2, 3, 4)]                        # obovate box leaf: narrow base, blunt rounded tip, bowed
_WEDGE_V = np.array([(0.0, 0.0, 0.0), (0.92, -0.33, 0.0), (0.92, 0.33, 0.0)])
_WEDGE_F = [(0, 1, 2)]                                     # one triangle for the shaded, mostly hidden bottom layer


def _tmpl(V, F):
    uv = np.stack([0.125 + V[:, 1] / 0.33 * 0.085, 0.02 + V[:, 0] * 0.21], axis=1)   # blade fills the 0.25 m leaf tile
    return (V, F, uv)


def _tint(h, k):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 * k[j] for j, i in enumerate((0, 2, 4))]
    return "#%02x%02x%02x" % tuple(int(round(clamp(v) * 255)) for v in c)


def _unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def _stamp(name, mtl, tmpl, pos, d, n, s):
    """Leaves as ONE mesh (numpy): blade k has its base at pos[k], runs along d[k], faces n[k], length s[k]."""
    V, F, UV = tmpl
    k = len(pos)
    if k == 0:
        return None
    pos, d, n, s = (np.asarray(a, dtype=np.float64) for a in (pos, d, n, s))
    w = np.cross(n, d)
    co = pos[:, None, :] + s[:, None, None] * (V[None, :, 0, None] * d[:, None, :] + V[None, :, 1, None] * w[:, None, :]
                                               + V[None, :, 2, None] * n[:, None, :])
    nv = len(V)
    lv = np.array([i for f in F for i in f], dtype=np.int32)
    nl = len(lv)
    ls = np.cumsum([0] + [len(f) for f in F[:-1]]).astype(np.int32)
    nf = len(F)
    me = bpy.data.meshes.new(name)
    me.vertices.add(k * nv)
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.loops.add(k * nl)
    me.loops.foreach_set("vertex_index", (lv[None, :] + (np.arange(k, dtype=np.int32) * nv)[:, None]).ravel())
    me.polygons.add(k * nf)
    me.polygons.foreach_set("loop_start", (ls[None, :] + (np.arange(k, dtype=np.int32) * nl)[:, None]).ravel())
    me.update(calc_edges=True)
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.tile(UV[lv][None], (k, 1, 1)).astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(k * nf, dtype=bool))
    me.materials.append(mtl)
    o = mesh_object(name, [], [])
    old = o.data
    o.data = me
    bpy.data.meshes.remove(old)
    uv_keep(o)
    o["tsq_smooth"] = "keep"
    return o


def _sample(obj, n, rs):
    """n evenly spread (area-stratified, jittered) surface points + smooth outward normals; also the area."""
    realize(obj)
    me = obj.data
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1) * 0.5
    vn = np.zeros_like(co)
    for j in range(3):
        np.add.at(vn, tri[:, j], cr)
    vn = _unit(vn)
    cw = np.cumsum(area)
    x = (np.arange(n) + rs.random(n) * 0.999) / n * cw[-1]
    pick = np.minimum(np.searchsorted(cw, x), len(area) - 1)
    u, v = rs.random(n), rs.random(n)
    fl = u + v > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    t0, t1, t2 = tri[pick, 0], tri[pick, 1], tri[pick, 2]
    P = a[pick] + (b[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]
    N = _unit(vn[t0] * (1 - u - v)[:, None] + vn[t1] * u[:, None] + vn[t2] * v[:, None])
    return P, N, float(area.sum())


def _perp(N, rs):
    return _unit(np.cross(rs.normal(size=N.shape), N))


def _lying(P, N, rs, L, e0, e1, lift, back):
    """Blades lying on the clipped surface in every direction, tips lifted e0..e1 degrees. Their faces stay
    within ~30 deg of the surface normal, so the foliage shades as one volume and hardly any blade shows its back."""
    k = len(P)
    t = _perp(N, rs)
    e = np.radians(rs.uniform(e0, e1, k))[:, None]
    d = np.cos(e) * t + np.sin(e) * N
    n = np.cos(e) * N - np.sin(e) * t
    ro = np.radians(rs.normal(0, 10, k))[:, None]
    n = np.cos(ro) * n + np.sin(ro) * np.cross(d, n)
    s = rs.uniform(0.85, 1.25, k) * L
    pos = P + N * rs.uniform(lift * 0.3, lift, (k, 1)) - d * (s * back)[:, None]
    return pos, d, n, s


def _sprigs(P, N, rs, L, length, npairs, e0, e1, splay):
    """Box shoots: opposite pairs of blades up a short axis standing out of the surface, each pair turned 90 deg,
    the pairs opening out flat like the clipped face of a real bush."""
    k = len(P)
    a = _unit(N + splay * rs.normal(size=(k, 3)) + np.array([0.0, 0.0, 0.08]))
    u = _perp(a, rs)
    v = np.cross(a, u)
    ph0 = rs.uniform(0, 2 * np.pi, k)
    out = [[], [], [], []]
    for j in range(int(npairs.max())):
        sel = npairs > j
        if not sel.any():
            break
        m = int(sel.sum())
        term = (j == npairs - 1)[sel]
        hj = (length * (j + 0.4) / npairs)[sel]
        for side in (0, 1):
            ang = ph0[sel] + j * np.pi / 2 + side * np.pi + rs.normal(0, 0.3, m)
            r = np.cos(ang)[:, None] * u[sel] + np.sin(ang)[:, None] * v[sel]
            e = np.radians(rs.uniform(e0, e1, m) + term * 5.0)[:, None]
            d = np.cos(e) * r + np.sin(e) * a[sel]
            n = np.cos(e) * a[sel] - np.sin(e) * r
            ro = np.radians(rs.normal(0, 9, m))[:, None]
            n = np.cos(ro) * n + np.sin(ro) * np.cross(d, n)
            s = rs.uniform(0.86, 1.22, m) * L * (1.0 - 0.15 * term)
            for lst, val in zip(out, (P[sel] + a[sel] * hj[:, None] + r * (0.001 + 0.06 * s[:, None]), d, n, s)):
                lst.append(val)
    return tuple(np.concatenate(x) for x in out)


def _ball_prof(rxy, rz, zc, n=22):
    pts = []
    for i in range(n + 1):
        a = radians(-90 + 180.0 * i / n)
        r = 0.0 if i in (0, n) else rxy * cos(a)
        pts.append((r, zc + rz * sin(a) * (0.86 if a < 0 else 1.0)))
    return pts


def _on_face(k, loc, rot=(90, 0, 0), scale=1.0):
    """Matrix for something modelled on the front (-Y) face, moved to face k (0 front, 1 right, 2 back, 3 left)."""
    return trs((0, 0, 0), (0, 0, 90 * k)) @ trs(loc, rot, scale)


def _edge(d, a, square):
    """Radial distance to a square / round outline of half-size d in direction a."""
    return d / max(abs(cos(a)), abs(sin(a))) if square else d


# ------------------------------------------------------------------------------------------- pots
def _pot_terracotta(p, rng, R, h):
    tc = mat("terracotta", p.pot_tint)
    age = p.pot_age
    f = 0.022                                   # pot feet lift it clear of the paving
    rb = R * 0.66
    t = max(0.014, R * 0.075)
    zc0 = h * 0.79                              # underside of the rim collar

    def body(z):                                # thrown wall: straight taper with a slight belly
        s = clamp((z - f) / (zc0 - f))
        return lerp(rb, R * 0.9, s) + 0.03 * R * sin(pi * s)

    zb = f + (zc0 - f) * 0.27                   # raised cordon
    zt = f + (zc0 - f) * 0.5                    # old water / salt tide line
    zm = (zc0 + h) * 0.5
    prof = [(0.0, f + 0.004), (rb - 0.014, f), (rb - 0.004, f + 0.004), (body(f + 0.014), f + 0.014)]
    n1 = max(2, int(round((zb - 0.03 - f) / 0.02)))
    for k in range(1, n1 + 1):
        z = lerp(f + 0.014, zb - 0.016, k / n1)
        prof.append((body(z), z))
    rr = body(zb)
    prof += [(rr + 0.005, zb - 0.011), (rr + 0.009, zb - 0.004), (rr + 0.009, zb + 0.004), (rr + 0.005, zb + 0.011),
             (body(zb + 0.016), zb + 0.016)]
    zz = [lerp(zb + 0.016, zc0 - 0.006, k / max(2, int(round((zc0 - zb) / 0.02)))) for k in range(1, max(2, int(round((zc0 - zb) / 0.02))))]
    zz = sorted([z for z in zz if abs(z - zt) > 0.011] + [zt - 0.007, zt, zt + 0.007])
    prof += [(body(z), z) for z in zz]
    zq = lerp(zc0 + 0.014, zm - 0.012, 0.5)
    zq2 = lerp(zm + 0.01, h - 0.03, 0.5)
    prof += [(body(zc0 - 0.006), zc0 - 0.006), (R - 0.01, zc0 + 0.003), (R - 0.002, zc0 + 0.014), (R - 0.0005, zq), (R, zm - 0.012),
             (R - 0.0035, zm - 0.006), (R - 0.0035, zm + 0.004), (R + 0.001, zm + 0.01), (R + 0.0025, zq2), (R + 0.003, h - 0.03),
             (R + 0.001, h - 0.012), (R - 0.007, h - 0.002), (R - 0.016, h), (R - t - 0.006, h), (R - t - 0.012, h - 0.007),
             (R - t - 0.014, h - 0.075), (rb - t - 0.004, f + 0.03), (0.0, f + 0.03)]        # closed: wall thickness + floor
    pot = lathe("pot", prof, segments=56, smooth_angle=42)
    # chips knocked out of the rolled rim (conchoidal bites showing fresh clay)
    r = rng.fork("chips")
    chips = []
    for k in range(int(round(age * 3.6 + r.u(-0.3, 0.3)))):
        a = radians(-90 + r.jitter(0, 55)) if k == 0 else r.u(0, 2 * pi)
        rc = r.u(0.024, 0.036) * (R / 0.2) ** 0.5
        cc = Vector(((R - 0.012) * cos(a), (R - 0.012) * sin(a), h + rc * r.u(0.15, 0.35)))
        cut = sphere("chip", rc, segments=8, rings=5)
        displace_noise(cut, rc * 0.18, rc * 0.9, seed=p.seed + k)
        set_loc_rot(cut, loc=tuple(cc), rot=(r.jitter(0, 20), r.jitter(0, 20), degrees(a)), scale=(1.0, r.u(1.0, 1.5), r.u(0.85, 1.0)))
        keep, nv0, z0 = pot.data.copy(), len(pot.data.vertices), min(v.co.z for v in pot.data.vertices)
        try:
            boolean(pot, cut)
            zs_ = [v.co.z for v in pot.data.vertices]
            if len(zs_) < nv0 or min(zs_) > z0 + 0.005 or max(zs_) < h - 0.01:
                raise RuntimeError("degenerate result")
            chips.append((np.array(cc), rc))
            bpy.data.meshes.remove(keep)
        except Exception:                        # a failed bite must never cost the pot: skip it
            bad = pot.data
            pot.data = keep
            if bad.users == 0:
                bpy.data.meshes.remove(bad)
    displace_noise(pot, 0.0028, 0.17, seed=p.seed + 3)
    uv_cyl(pot, axis="Z")
    objs = [pot]
    # festoon: swags of rolled clay hung between rosettes under the collar
    ns = 6
    sag = 0.17 * h
    ztop = zc0 - 0.02
    ros = lathe("rosette", [(0.0, 0.0), (0.021, 0.0), (0.023, 0.003), (0.017, 0.007), (0.011, 0.006), (0.008, 0.011), (0.0, 0.013)], segments=12)
    rm = []
    for k in range(ns):
        a0, a1 = 2 * pi * k / ns, 2 * pi * (k + 1) / ns
        path = []
        for i in range(15):
            s = i / 14
            a = lerp(a0, a1, s)
            z = ztop - sag * (1 - (2 * s - 1) ** 2) ** 0.8
            rr_ = body(z) + 0.0025
            path.append((rr_ * sin(a), -rr_ * cos(a), z))
        objs.append(sweep(f"swag{k}", path, 0.0065, sides=6, step=0.022, scale_fn=lambda q: 0.7 + 0.5 * sin(pi * q)))
        rm.append(trs(((body(ztop) - 0.002) * sin(a0), -(body(ztop) - 0.002) * cos(a0), ztop), (90, 0, degrees(a0))))
    objs.append(instances_to_mesh("rosettes", ros, rm, keep_base=False))
    r = rng.fork("feet")
    a_f = r.u(0, 120)
    for k in range(3):
        a = radians(a_f + 120 * k + r.jitter(0, 6))
        objs.append(box(f"pot_foot{k}", (0.07, 0.045, f + 0.006), loc=((rb - 0.025) * cos(a), (rb - 0.025) * sin(a), (f + 0.006) / 2),
                        rot=(0, 0, degrees(a) + 90), bevel=0.005))

    # ---- weathering story. The unworn library terracotta is the chalky, lime-bloomed state; `wear` shows clean
    # orange clay, `special` grows green algae, `grime` is dark damp. So: chalk left alone in a band creeping down
    # from under the collar and along a tide line, clay showing through lower down and raw in the chips, algae up
    # the damp lower third and the shaded side, a dark wet foot.
    ash = radians(35 + rng.fork("shade").jitter(0, 25))
    mg = min(1.0, p.moss * 1.7)
    ag = min(1.0, age * 1.7)
    nz_a = noise_mask(0.1, seed=p.seed + 5, lo=0.18, hi=0.82)
    nz_c = noise_mask(0.055, seed=p.seed + 7, lo=0.2, hi=0.8)
    ph = [rng.fork("streak").u(0, 6.28) for _ in range(3)]

    def streaks(th, z):                          # vertical run-off streaks
        return np.clip(0.55 + 0.3 * np.sin(th * 19 + 2.5 * np.sin(th * 5 + ph[0]) + ph[1]) + 0.25 * np.sin(th * 43 + ph[2] + z * 9), 0, 1)

    def algae(co, n):
        z, th = co[:, 2], np.arctan2(co[:, 1], co[:, 0])
        side = np.clip(0.5 + 0.5 * np.cos(th - ash), 0, 1) ** 1.3
        # green line climbing in tongues from the wet foot, higher on the shaded side. Only low frequencies and a
        # slow ramp: the mask is per vertex, and the shader frays the edge itself where the value hovers near 0.5
        st = 0.5 + 0.3 * np.sin(7 * th + 2.0 * np.sin(3 * th + ph[0]) + ph[1]) + 0.2 * np.sin(12 * th + ph[2])
        top = h * (0.07 + 0.2 * st * (0.3 + 0.7 * nz_a(co, n)) + 0.12 * nz_c(co, n) + 0.34 * side ** 2 * (0.4 + 0.6 * nz_a(co, n))) * (0.35 + 0.95 * mg)
        zz = z - f
        m = np.clip(0.56 + (top - zz) / (0.4 * h), 0, 0.82) * (0.85 + 0.15 * nz_c(co, n))
        m = np.maximum(m, np.exp(-((z - (zc0 - 0.012)) / 0.02) ** 2) * (0.2 + 0.5 * side) * (0.5 + 0.6 * nz_c(co, n)))    # ledge under the collar
        return np.clip(m * min(1.0, mg * 3.0), 0, 0.9)

    def clay(co, n):
        z, th = co[:, 2], np.arctan2(co[:, 1], co[:, 0])
        band = np.clip(1.0 - (zc0 - z) / (0.5 * (zc0 - f)), 0, 1) * (z < zc0 + 0.006)
        salt = np.clip(band * 1.5 + 0.5 * nz_c(co, n) * streaks(th + 1.0, z) - 0.3, 0, 1)
        salt = np.maximum(salt, np.exp(-((z - zt) / 0.006) ** 2) * (0.5 + 0.5 * nz_a(co, n)))
        worn = np.where(z > zc0 + 0.006, np.clip(1.9 * nz_a(co, n) - 0.75, 0, 0.7) * (0.55 + 0.45 * nz_c(co, n)), (0.1 + 0.5 * nz_a(co, n)) * (1.0 - salt))
        m = worn * ag + 0.8 * (1.0 - ag)
        for cc, rc in chips:
            m = np.maximum(m, np.clip(1.6 - np.linalg.norm(co - cc, axis=1) / (rc * 1.2), 0, 1))
        return np.clip(m, 0, 1)

    def damp(co, n):
        z = co[:, 2]
        m = np.clip(1.0 - z / (h * (0.07 + 0.08 * nz_a(co, n))), 0, 1) ** 0.7 * (0.4 + 0.35 * age)
        m = np.maximum(m, 0.3 * (z > zc0) * nz_c(co, n) * age)                  # handled, dirty collar
        for cc, rc in chips:                     # old breaks: raw clay gone dull and dirty
            m = np.maximum(m, 0.45 * np.clip(1.5 - np.linalg.norm(co - cc, axis=1) / (rc * 1.1), 0, 1))
        return np.clip(np.maximum(m, 0.15 * (z < h * 0.5) * nz_c(co, n)), 0, 1)

    for o in objs:
        assign(o, tc)
        paint(o, grime=damp, wear=clay, special=algae)

    def wall(a, z):                              # outer surface radius (incl. cordon / swags) for things clinging to it
        if z >= zc0:
            return R + 0.003
        rr_ = body(z)
        if abs(z - zb) < 0.015:
            rr_ += 0.009
        s = ((a + pi / 2) * ns / (2 * pi)) % 1.0
        zsw = ztop - sag * (1 - (2 * s - 1) ** 2) ** 0.8
        if abs(z - zsw) < 0.016:
            rr_ += 0.011
        return lerp(rr_, R + 0.003, smoothstep(zc0 - 0.05, zc0, z))

    return {"zr": h, "zs": h - 0.04, "rin": R - t - 0.013, "square": False, "rout": R, "rim": R + 0.003, "rim_c": R - t * 0.5 - 0.011,
            "wall": wall}


def _pot_lead(p, rng, R, h):
    lead = mat("lead")
    s = R * 0.92
    f = 0.02
    objs = []
    body = shape_extrude("lead_body", rounded_rect(2 * s, 2 * s, 0.008, 2), [rounded_rect(2 * s - 0.03, 2 * s - 0.03, 0.004, 2)],
                         depth=h - f, loc=(0, 0, f))
    densify(body, 0.06)
    displace_noise(body, 0.0022, 0.16, seed=p.seed + 2)
    objs.append(body)
    objs.append(shape_extrude("lead_rim", rounded_rect(2 * s + 0.026, 2 * s + 0.026, 0.01, 2), [rounded_rect(2 * s - 0.034, 2 * s - 0.034, 0.004, 2)],
                              depth=0.03, loc=(0, 0, h - 0.03), bevel=0.004))
    objs.append(shape_extrude("lead_base", rounded_rect(2 * s + 0.02, 2 * s + 0.02, 0.01, 2), [rounded_rect(2 * s - 0.02, 2 * s - 0.02, 0.004, 2)],
                              depth=0.04, loc=(0, 0, f), bevel=0.004))
    foot = lathe("bun", [(0.0, 0.0), (0.03, 0.0), (0.043, 0.007), (0.044, 0.015), (0.036, f + 0.006)], segments=14)
    objs.append(instances_to_mesh("lead_feet", foot, [trs((sx * (s - 0.05), sy * (s - 0.05), 0)) for sx in (-1, 1) for sy in (-1, 1)], keep_base=False))
    # cast strapwork on every face: main panel frame with a lozenge and rosette, frieze frame above
    w = 2 * s - 0.07
    zlo, zhi = f + 0.065, h * 0.66
    zf0, zf1 = h * 0.7, h - 0.045
    zc, hp = (zlo + zhi) / 2, zhi - zlo
    dz = 0.0065
    deco = [shape_extrude("frame", rounded_rect(w, hp, 0.006, 2), [rounded_rect(w - 0.03, hp - 0.03, 0.004, 2)], depth=dz,
                          loc=(0, -s, zc), rot=(90, 0, 0), bevel=0.002)]
    a, b = w / 2 - 0.03, hp / 2 - 0.03
    kk = 1.0 - 0.02 / min(a, b)
    deco.append(shape_extrude("lozenge", [(a, 0), (0, b), (-a, 0), (0, -b)], [[(a * kk, 0), (0, b * kk), (-a * kk, 0), (0, -b * kk)]],
                              depth=dz, loc=(0, -s, zc), rot=(90, 0, 0), bevel=0.002))
    rr = min(a, b) * 0.36
    deco.append(lathe("boss", [(0.0, 0.0), (rr, 0.0), (rr, 0.004), (rr * 0.8, 0.008), (rr * 0.62, 0.006), (rr * 0.5, 0.012), (rr * 0.25, 0.016), (0.0, 0.017)],
                      segments=16, loc=(0, -s, zc), rot=(90, 0, 0)))
    deco.append(shape_extrude("frieze", rounded_rect(w, zf1 - zf0, 0.005, 2), [rounded_rect(w - 0.022, zf1 - zf0 - 0.022, 0.003, 2)],
                              depth=dz, loc=(0, -s, (zf0 + zf1) / 2), rot=(90, 0, 0), bevel=0.002))
    face = join(deco, "lead_face")
    objs.append(instances_to_mesh("lead_deco", face, [trs((0, 0, 0), (0, 0, 90 * k)) for k in range(4)], keep_base=False))
    stud = bolt("stud", 0.011, 0.007, kind="dome")
    sm = []
    zfc = (zf0 + zf1) / 2
    for k in range(4):
        if k == 0 and p.date:
            ts = (zf1 - zf0 - 0.03) * 0.92
            objs.append(text("date", str(p.date)[:6], size=ts, depth=0.005, loc=(0, -s, zfc - ts * 0.38), font="serif"))
            xs = [-w / 2 + 0.035, w / 2 - 0.035]
        else:
            xs = [-w * 0.3, 0.0, w * 0.3]
        sm += [_on_face(k, (x, -s, zfc)) for x in xs]
    objs.append(instances_to_mesh("lead_studs", stud, sm, keep_base=False))
    for o in objs:
        assign(o, lead)
        paint(o, grime=mask_mul(height_ramp(0.0, h * 0.3), 0.4), wear=mask_mul(height_ramp(h + 0.01, h - 0.05), 0.5))

    def wall(a, z):
        return _edge(s + (0.017 if z > h - 0.034 else 0.0068), a, True)

    return {"zr": h, "zs": h - 0.045, "rin": s - 0.016, "square": True, "rout": s * 1.2, "rim": s + 0.017, "rim_c": s - 0.002, "wall": wall}


def _pot_versailles(p, rng, R, h):
    wood = mat("wood_painted", p.paint)
    iron = mat("iron_wrought")
    s = R * 0.95
    post = clamp(R * 0.24, 0.045, 0.07)
    leg = 0.06
    r = rng.fork("case")
    wobjs, iobjs = [], []
    hp = h + 0.012
    fin_prof = [(post * 0.44, 0.0), (post * 0.44, 0.007), (post * 0.26, 0.012), (post * 0.22, 0.02), (post * 0.36, 0.027),
                (post * 0.58, 0.045), (post * 0.62, 0.058), (post * 0.54, 0.074), (post * 0.3, 0.088), (0.0, 0.093)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * (s - post / 2), sy * (s - post / 2)
            po = box("post", (post, post, hp), loc=(x, y, hp / 2), rot=(0, 0, r.jitter(0, 0.8)), bevel=0.004)
            uv_grain(po, axis="Z")
            wobjs.append(po)
            fi = lathe("finial", fin_prof, segments=18, loc=(x, y, hp))
            wobjs.append(fi)
    # one side modelled on the front face, then stamped round: rails, boards, iron straps with bolts
    span = 2 * s - 2 * post + 0.004
    side_w, side_i = [], []
    for zc_, hh in ((h - 0.045, 0.07), (leg + 0.035, 0.07)):
        rl = box("rail", (span, 0.034, hh), loc=(0, -s + 0.019, zc_), bevel=0.004)
        uv_grain(rl, axis="X")
        side_w.append(rl)
    nb = max(3, int(round(span / 0.11)))
    bw = span / nb
    for i in range(nb):
        bz0, bz1 = leg + 0.066, h - 0.076                # tucked 4 mm into the rail rebates
        bd = box("board", (bw - 0.004, 0.018, bz1 - bz0), loc=(-span / 2 + bw * (i + 0.5), -s + 0.023 + r.jitter(0, 0.002), (bz0 + bz1) / 2),
                 rot=(0, r.jitter(0, 0.3), 0), bevel=0.003)
        densify(bd, 0.06)
        uv_grain(bd, axis="Z")
        side_w.append(bd)
    bt = lathe("cbolt", [(0.0085, 0.0), (0.0085, 0.0012), (0.0068, 0.0022), (0.0055, 0.0045), (0.003, 0.006), (0.0, 0.0065)], segments=8)   # clout-head coach bolt
    bm = []
    for zc_ in (h - 0.045, leg + 0.035):
        for sx in (-1, 1):
            side_i.append(box("strap", (0.15, 0.004, 0.032), loc=(sx * (s - 0.077), -s - 0.0005, zc_), bevel=0.001))
            bm += [trs((sx * (s - xx), -s - 0.0025, zc_), (90, 0, 0)) for xx in (0.025, 0.085, 0.135)]
    side_i.append(instances_to_mesh("strap_bolts", bt, bm, keep_base=False))
    for o in side_w:
        assign(o, wood)
    for o in side_i:
        assign(o, iron)
    fw, fi_ = join(side_w, "side_w"), join(side_i, "side_i")
    rots = [trs((0, 0, 0), (0, 0, 90 * k)) for k in range(4)]
    wobjs.append(instances_to_mesh("case_sides", fw, rots, keep_base=False))
    iobjs.append(instances_to_mesh("case_iron", fi_, rots, keep_base=False))
    # drop-ring lifting handles on the two flanks
    for sx in (-1, 1):
        iobjs.append(torus("ring", 0.036, 0.0055, loc=(sx * (s + 0.008), 0, h - 0.045 - 0.03), rot=(0, 90, 0), major_seg=20, minor_seg=6))
        iobjs.append(box("staple", (0.012, 0.03, 0.022), loc=(sx * (s + 0.004), 0, h - 0.04), bevel=0.003))
    for o in wobjs:
        assign(o, wood)
        paint(o, grime=mask_mul(height_ramp(0.0, 0.3), 0.9), wear=mask_mul(height_ramp(0.0, 0.12), noise_mask(0.05, seed=p.seed, lo=0.4, hi=0.7), 0.7),
              special=mask_mul(facing((0, 0, 1)), 0.5))
    for o in iobjs:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.04, seed=p.seed + 1, lo=0.35, hi=0.75), 0.6))

    def wall(a, z):                              # rails stand proud of the recessed boards
        k = max(smoothstep(h - 0.1, h - 0.078, z), 1.0 - smoothstep(leg + 0.068, leg + 0.09, z))
        return _edge(lerp(s - 0.013, s + 0.003, k), a, True)

    return {"zr": h, "zs": h - 0.05, "rin": s - 0.045, "square": True, "rout": s * 1.2, "rim": s, "rim_c": s - 0.019, "wall": wall}


# ------------------------------------------------------------------------------------------- build
def build(p, rng):
    H = p.height
    rs = np.random.default_rng(int(rng.fork("np").u() * 1e9))
    h_pot = clamp((0.26 if p.shape == "ball" else 0.2) * H + 0.1, 0.24, 0.66) * p.pot_scale
    R = h_pot * 0.56
    pot = {"terracotta": _pot_terracotta, "lead": _pot_lead, "versailles": _pot_versailles}[p.pot](p, rng, R, h_pot)
    zr, zs, rin, sq, wall = pot["zr"], pot["zs"], pot["rin"], pot["square"], pot["wall"]
    og = p.overgrowth

    # (box is a dull, dark green: these put the leaf albedo at 0.06-0.13 - twice that reads as lime, self-lit
    # foliage under a lantern at night)
    leaf_f = mat("hedge", _tint(p.foliage, (0.29, 0.36, 0.25)), rough=4.0, wear=0.0)     # shaded under-layer
    hedge = leaf_f                                # the shell under the leaves is the same shaded green: one draw call
    leaf_d = mat("hedge", _tint(p.foliage, (0.36, 0.44, 0.3)), rough=4.0, wear=0.0)     # body
    leaf_l = mat("hedge", _tint(p.foliage, (0.42, 0.5, 0.32)), rough=4.0, wear=0.0)     # new growth, ~15 % lighter
    straw = mat("wood_weathered", "#d9c58c")
    bark = mat("bark")

    # ---- soil, mounded towards the stem
    if sq:
        soil = grid("soil", (2 * rin, 2 * rin), (8, 8), loc=(0, 0, zs))
        transform_verts(soil, lambda co: Vector((co.x, co.y, co.z + 0.022 * (1 - min(1.0, hypot(co.x, co.y) / rin)))))
    else:
        soil = lathe("soil", [(0.0, zs + 0.022), (rin * 0.35, zs + 0.015), (rin * 0.7, zs + 0.006), (rin + 0.004, zs - 0.004)], segments=24)
    displace_noise(soil, 0.007, 0.05, seed=p.seed, along_normal=False, axis_mask=(0, 0, 1))
    uv_box(soil)
    assign(soil, mat("soil"))
    nm = int(round(p.moss * 7))
    r = rng.fork("moss")
    for i in range(nm):
        a, d = r.u(0, 2 * pi), rin * r.u(0.45, 0.88)
        sz = r.u(0.025, 0.05)
        m = sphere("moss_clump", sz, loc=(d * cos(a), d * sin(a), zs + 0.004), segments=10, rings=6, scale=(1, r.u(0.7, 1.2), 0.42))
        displace_noise(m, 0.005, 0.03, seed=i)
        assign(m, mat("moss"))

    # ---- layout of the clipped form
    S = H - zr
    plant = []          # everything that leans with the plant
    shells = []
    stem_top = H - 0.1
    r_stem = clamp(0.012 + 0.011 * H, 0.016, 0.042)
    cane_top = None
    crowns = []         # (z underside, rxy, rz) for branch forks
    if p.shape == "ball":
        D = clamp(S - 0.05, 0.2, 0.92)
        rz = D / 2
        rxy = rz * 1.03 * p.girth
        zc = H - rz
        shells.append(lathe("ball", _ball_prof(rxy, rz, zc, n=26), segments=56))
        crowns.append((zc - rz * 0.86, rxy, rz))
        stem_top = zc
        if zc - rz - zs > 0.28:
            cane_top = zc - rz * 0.3
    elif p.shape == "two_tier":
        D1 = clamp(0.37 * S, 0.22, 0.85)
        D2 = D1 * 0.68
        gap = 0.13 * S
        z2 = H - D2 / 2
        z1 = H - D2 - gap - D1 / 2
        shells.append(lathe("ball_low", _ball_prof(D1 / 2 * 1.04 * p.girth, D1 / 2, z1, n=24), segments=52))
        shells.append(lathe("ball_top", _ball_prof(D2 / 2 * 1.04 * p.girth, D2 / 2, z2, n=18), segments=40))
        crowns += [(z1 - D1 * 0.43, D1 / 2, D1 / 2), (z2 - D2 * 0.43, D2 / 2, D2 / 2)]
        stem_top = z2
        cane_top = z2 - D2 * 0.2
    elif p.shape == "cone":
        zb = zr + 0.06
        hf = H - zb
        Rb = clamp(0.23 * hf * p.girth, 0.12, 0.6)
        prof = [(0.0, zb + 0.035), (Rb * 0.5, zb + 0.018), (Rb * 0.8, zb + 0.004), (Rb * 0.95, zb + 0.022), (Rb, zb + 0.07)]
        nr = 24
        for i in range(1, nr):
            t = i / nr
            prof.append((Rb * (1 - t) ** 0.86 + 0.03 * (1 - t) ** 0.2, zb + 0.07 + (hf - 0.075) * t))
        prof += [(0.034, H - 0.045), (0.026, H - 0.02), (0.014, H - 0.005), (0.0, H)]       # blunt clipped tip, wide enough to leaf over
        shells.append(lathe("cone", prof, segments=52))
        stem_top = zb + hf * 0.7
    else:
        # spiral: ONE continuous wedge-section band (thick at the clipped outer face, thin at the stem) wound up
        # the stem; radius, pitch and section all taper together so the groove between turns stays open, and
        # the band runs out into a pointed finial
        zb = zr + 0.06
        c = 0.55                                        # top of the band is (1 - c) the size of the bottom
        fin_h = clamp(0.15 * (H - zb), 0.1, 0.34)
        th_e = 2 * pi * p.turns
        K = -math.log(1 - c) / th_e
        Zs = (H - 0.8 * fin_h - zb) / (1 + 0.3 * 2 * pi * K * (2 - c) / c)
        P0 = 2 * pi * Zs * K / c                        # pitch of the first turn
        R0 = clamp(0.2 * (H - zb) * p.girth, 0.12, 0.5)
        a0 = rng.u(0, 2 * pi)

        def band(u, v):
            th = th_e * u
            q = math.exp(-K * th)
            e = (0.04 + 0.96 * smoothstep(0.0, 0.06, u)) * (1.0 - 0.5 * smoothstep(0.93, 1.0, u))
            cs, sn = cos(2 * pi * v), sin(2 * pi * v)
            fx = 0.5 + 0.5 * math.copysign(abs(cs) ** 0.7, cs)
            rho = 0.004 + R0 * q * e * fx
            hb = lerp(0.115, 0.3, fx ** 0.8) * P0 * q * e
            z = zb + 0.3 * P0 + Zs * (1 - q) / c + hb * math.copysign(abs(sn) ** 0.75, sn)
            return (rho * cos(th + a0), rho * sin(th + a0), z)

        sp = surface("spiral", band, nu=int(p.turns * 30), nv=14, closed_v=True)
        realize(sp)
        recalc_normals(sp)
        shells.append(sp)
        rf = R0 * (1 - c) * 0.66 * p.girth ** 0.5
        z0f = H - 1.45 * fin_h
        hfn = H - z0f
        shells.append(lathe("spiral_tip", [(0.0, z0f), (rf * 0.6, z0f + 0.012), (rf * 0.96, z0f + 0.14 * hfn), (rf, z0f + 0.26 * hfn),
                                           (rf * 0.86, z0f + 0.42 * hfn), (rf * 0.6, z0f + 0.6 * hfn), (rf * 0.34, z0f + 0.77 * hfn),
                                           (rf * 0.13, z0f + 0.92 * hfn), (0.0, H)], segments=24))
        stem_top = H - fin_h

    # ---- stem (bark), root flare, branch forks under the crowns, cane and twine ties
    r = rng.fork("stem")
    nst = max(6, int((stem_top - zs) / 0.09))
    wob = 0.006 + 0.006 * og
    pts = [(r.jitter(0, wob) * min(1.0, i / 2), r.jitter(0, wob) * min(1.0, i / 2), lerp(zs - 0.01, stem_top, i / nst)) for i in range(nst + 1)]
    Ls = stem_top - zs
    stem = sweep("stem", pts, r_stem, sides=10, scale_fn=lambda t: lerp(1.0, 0.55, t) + 0.9 * math.exp(-t * Ls / 0.03))
    plant.append(stem)
    for (zu, rxy, rz) in crowns:
        nbr = 4
        ab = r.u(0, 360)
        for k in range(nbr):
            a = radians(ab + k * 360 / nbr + r.jitter(0, 20))
            ca, sa = cos(a), sin(a)
            d = rxy * r.u(0.45, 0.6)
            bp = bezier_path((0, 0, zu - 0.1), (d * 0.15 * ca, d * 0.15 * sa, zu - 0.02), (d * 0.6 * ca, d * 0.6 * sa, zu + rz * 0.05),
                             (d * ca, d * sa, zu + rz * 0.5), n=8)
            plant.append(sweep("branch", bp, r_stem * 0.5, sides=6, step=0.035, scale_fn=lambda t: 1.0 - 0.5 * t))
    for o in plant:
        uv_cyl(o, axis="Z")
        assign(o, bark)
        paint(o, special=mask_mul(height_ramp(zs - 0.02, zs + 0.3), min(1.0, p.moss * 1.4)))
    if p.cane and cane_top:
        cx, cy = r_stem * 0.9 + 0.012, 0.004
        cprof = []
        zz = zs - 0.02
        while zz < cane_top:
            cprof += [(0.0062, zz), (0.0062, zz + 0.012), (0.0075, zz + 0.017), (0.0062, zz + 0.022)]
            zz += r.u(0.17, 0.24)
        cprof.append((0.0055, cane_top))
        cane = lathe("cane", cprof, segments=8, loc=(cx, cy, 0))
        assign(cane, straw)
        plant.append(cane)
        nt = max(2, int((cane_top - zs) / 0.34))
        for k in range(nt):
            zt = lerp(zs + 0.12, cane_top - 0.1, k / max(1, nt - 1)) if nt > 1 else zs + 0.2
            for dzz, tl in ((0.0, 8), (0.006, -6)):
                tie = torus("tie", r_stem * 0.72 + 0.012, 0.0028, loc=(cx * 0.5, cy * 0.5, zt + dzz), rot=(tl, r.jitter(0, 6), 0), major_seg=12, minor_seg=4)
                transform_verts(tie, lambda co: Vector((co.x * 1.25, co.y * 0.85, co.z)))
                assign(tie, straw)                      # raffia twine, same straw colour as the cane
                plant.append(tie)

    # ---- clipped shells: lumpy, shear-faceted, never a perfect solid of revolution; dark and rough so that
    # whatever shows between the leaves reads as the shaded interior of the bush
    ks = 0.5 if p.shape == "spiral" else 1.0
    for i, sh in enumerate(shells):
        displace_noise(sh, (0.02 + 0.03 * og) * ks, 0.22, seed=p.seed + 11 * i)
        displace_noise(sh, (0.01 + 0.01 * og) * ks, 0.075, seed=p.seed + 5 + i)
        displace_noise(sh, 0.006 + 0.004 * og, 0.032, seed=p.seed + 9 + i)
        smooth(sh, 75)
        uv_box(sh)
        assign(sh, hedge)
        paint(sh, grime=0.15)
        plant.append(sh)

    # ---- the whole plant leans a touch out of true
    lean = rng.u(0.4, 1.8) * (1 + og)
    la = rng.u(0, 360)
    ML = Matrix.Translation((0, 0, zs)) @ Matrix.Rotation(radians(lean), 4, Vector((cos(radians(la)), sin(radians(la)), 0))) @ Matrix.Translation((0, 0, -zs))
    for o in plant:
        realize(o)
        transform_verts(o, lambda co: ML @ co)

    # ---- foliage in three layers of small box leaves: dark wedges packed on the shell (shaded interior), a
    # tight face of rounded blades lying on the clipped surface, and short sprigs of opposite leaves standing
    # proud for the fuzzy outline; paler new growth comes as whole shoots, in patches, mostly on top
    L0 = 0.0148
    wedge, pent = _tmpl(_WEDGE_V, _WEDGE_F), _tmpl(_PENT_V, _PENT_F)
    areas = [_sample(sh, 8, rs)[2] for sh in shells]
    A = sum(areas)
    n_w, n_f, n_s = A * 9000.0, A * 4500.0, A * 600.0
    kf = min(1.0, 39000.0 / (n_w + n_f * 3 + n_s * (5.2 + 0.4 * og) * 3))        # triangle cap; big bushes get slightly larger leaves
    Lk = L0 * clamp(kf ** -0.5, 1.0, 1.2)
    fq = [_unit(rs.normal(size=3)) * 2 * pi / rs.uniform(0.16, 0.34) for _ in range(3)]
    fp = rs.uniform(0, 6.28, 3)
    acc = {"w_f": [], "f_f": [], "f_d": [], "s_d": [], "s_l": []}
    for i, sh in enumerate(shells):
        fr = areas[i] / A
        P, N, _ = _sample(sh, max(12, int(n_w * kf * fr)), rs)
        acc["w_f"].append(_lying(P, N, rs, Lk * 1.04, 0, 14, 0.003, 0.4))
        P, N, _ = _sample(sh, max(12, int(n_f * kf * fr)), rs)
        fr_ = _lying(P, N, rs, Lk, 3, 17 + 20 * og, 0.007 + 0.005 * og, 0.35)
        dk = rs.random(len(P)) < 0.35
        acc["f_f"].append(tuple(x[dk] for x in fr_))
        acc["f_d"].append(tuple(x[~dk] for x in fr_))
        P, N, _ = _sample(sh, max(8, int(n_s * kf * fr)), rs)
        k = len(P)
        score = sum(np.sin(P @ fq[j] + fp[j]) for j in range(3)) / 1.5 + 0.8 * N[:, 2] + rs.normal(0, 0.16, k)
        new = score > np.quantile(score, 1.0 - (0.1 + 0.2 * og))
        ln = rs.uniform(0.006, 0.014, k) * (1 + 1.2 * og)
        for key, sel, lmul, npr, e0, e1, spl in (("s_d", ~new, 1.0, rs.integers(2, 4, k), 0, 15 + 12 * og, 0.12 + 0.2 * og),
                                                 ("s_l", new, 1.5 + 3.5 * og, rs.integers(3, 7 if og > 0.5 else 4, k), 5 + 12 * og, 21 + 22 * og, 0.12 + 0.3 * og)):
            if sel.any():
                acc[key].append(_sprigs(P[sel], N[sel], rs, Lk, (ln * lmul)[sel], npr[sel], e0, e1, spl))
    for key, mtl, tm in (("w_f", leaf_f, wedge), ("f_f", leaf_f, pent), ("f_d", leaf_d, pent), ("s_d", leaf_d, pent), ("s_l", leaf_l, pent)):
        if acc[key]:
            arrs = [np.concatenate([a[j] for a in acc[key]]) for j in range(4)]
            _stamp("foliage_" + key, mtl, tm, *arrs)

    # ---- ivy underplanting: stems creep over the rim and cling to the pot wall, leaves shrink to the tip
    niv = int(p.ivy)
    if niv > 0:
        r = rng.fork("ivy")
        LI = 0.046
        iv_pts = [(0, 0, 0), (-0.12 * LI, -0.44 * LI, 0.002), (0.3 * LI, -0.5 * LI, -0.002), (0.56 * LI, -0.24 * LI, 0.001), (LI, 0, -0.004),
                  (0.56 * LI, 0.24 * LI, 0.001), (0.3 * LI, 0.5 * LI, -0.002), (-0.12 * LI, 0.44 * LI, 0.002)]
        ivl = mesh_object("ivy_leaf", iv_pts, [(0, k, k + 1) for k in range(1, 7)])
        uvl = ivl.data.uv_layers.new(name="UVMap")
        for li, loop in enumerate(ivl.data.loops):
            v = iv_pts[loop.vertex_index]
            uvl.data[li].uv = (0.125 + v[1] / LI * 0.2, 0.03 + v[0] / LI * 0.19)
        uv_keep(ivl)
        assign(ivl, mat("leaf", _tint(p.foliage, (0.26, 0.33, 0.22)), rough=2.4, wear=0.0))
        im = []
        a_base = r.u(0, 2 * pi)
        for k in range(niv):
            a = radians(r.pick([0, 90, 180, 270]) + r.u(-24, 24)) if sq else a_base + 2 * pi * k / niv + r.u(-0.5, 0.5)
            Lh = min(zr - 0.06, zr * r.u(0.35, 0.9))
            sd = r.pick([-1, 1])
            r_in = _edge(rin, a, sq)
            pl = [(r_in * 0.5, a - 0.3 * sd, zs + 0.013), (r_in * 0.82, a - 0.1 * sd, zs + 0.017), (_edge(pot["rim_c"], a, sq), a, zr + 0.0045),
                  (_edge(pot["rim"], a, sq) - 0.002, a, zr + 0.002), (_edge(pot["rim"], a, sq) + 0.0038, a, zr - 0.012)]
            z, aa, drift = zr - 0.034, a, r.jitter(0, 0.5)
            while z > zr - Lh:
                drift = clamp(drift * 0.75 + r.jitter(0, 0.4), -0.8, 0.8)
                rho = wall(aa, z) + 0.0038
                aa += drift * 0.022 / max(rho, 0.05)
                pl.append((rho, aa, z))
                z -= 0.022
            pts3 = [Vector((rho * cos(q), rho * sin(q), z)) for (rho, q, z) in pl]
            o = sweep(f"ivy_stem{k}", [tuple(q) for q in pts3], 0.0034, sides=5, step=0.02, scale_fn=lambda t: 1.0 - 0.62 * t)
            assign(o, bark)
            npt = len(pts3)
            for j in range(1, npt):
                for sub in (0, 1):
                    q = pts3[j] if sub == 0 else (pts3[j] + pts3[j - 1]) * 0.5
                    tfrac = (j - 0.5 * sub) / (npt - 1)
                    aj = pl[j][1]
                    hang = j >= 4
                    if sq:
                        rad_v = Vector((round(cos(a)), round(sin(a)), 0)) if abs(abs(cos(a)) - abs(sin(a))) > 1e-6 else Vector((cos(a), sin(a), 0))
                    else:
                        rad_v = Vector((cos(aj), sin(aj), 0))
                    tan_v = Vector((-rad_v.y, rad_v.x, 0))
                    side = 1 if (2 * j + sub) % 2 == 0 else -1
                    if hang:        # flat against the wall, pointing down and outwards from the stem
                        n = (rad_v + Vector(r.vec(0.16)) + Vector((0, 0, 0.12))).normalized()
                        dv = Vector((0, 0, -1.0)) + tan_v * side * r.u(0.5, 1.4)
                    else:           # lying on the soil / over the rim
                        n = (Vector((0, 0, 1.0)) + rad_v * (0.5 if j >= 2 else 0.1) + Vector(r.vec(0.2))).normalized()
                        dv = rad_v * r.u(0.2, 1.0) + tan_v * side * r.u(0.6, 1.4)
                    x = (dv - n * dv.dot(n)).normalized()
                    y = n.cross(x)
                    sc_ = r.u(0.8, 1.15) * lerp(1.0, 0.3, tfrac ** 1.3)
                    c0 = q + tan_v * side * (0.004 + 0.005 * sc_) + n * 0.0035
                    im.append(Matrix(((x.x * sc_, y.x * sc_, n.x * sc_, c0.x), (x.y * sc_, y.y * sc_, n.y * sc_, c0.y),
                                      (x.z * sc_, y.z * sc_, n.z * sc_, c0.z), (0, 0, 0, 1))))
        uv_keep(instances_to_mesh("ivy_leaves", ivl, im, uv_jitter=False, keep_base=False))
    if p.label:
        r = rng.fork("label")
        a = r.u(0, 2 * pi)
        d = _edge(rin, a, sq) * 0.7
        lb = box("label", (0.022, 0.003, 0.16), loc=(d * cos(a), d * sin(a), zs + 0.05), rot=(r.jitter(0, 10), r.jitter(0, 10), degrees(a) + 90 + r.jitter(0, 30)),
                 bevel=0.001)
        uv_grain(lb, axis="Z")
        assign(lb, straw)
        paint(lb, grime=mask_mul(height_ramp(zs, zs + 0.1), 0.8))

    # ---- fallen clippings (single leaves and snipped sprig tips) on the soil and round the foot of the pot
    nlit = int(p.litter)
    if nlit > 0:
        r = rng.fork("litter")
        lit = {0: [], 1: []}
        for i in range(nlit):
            a = r.u(0, 2 * pi)
            if i % 3 == 0:
                d = _edge(rin, a, sq) * r.u(0.3, 0.85)
                z = zs + 0.013 + 0.02 * (1 - d / _edge(rin, a, sq))
            else:
                d = pot["rout"] + r.u(0.0, 0.18)
                z = 0.004
            nl_ = 1 if r.chance(0.45) else r.pick([3, 4, 5])
            yaw0, key = r.u(0, 2 * pi), (1 if r.chance(0.35) else 0)
            for j in range(nl_):
                yaw, tl = yaw0 + j * 2 * pi / nl_ + r.jitter(0, 0.4), radians(r.u(-5, 16) + (8 if nl_ > 1 else 0))
                dv = (cos(yaw) * cos(tl), sin(yaw) * cos(tl), sin(tl))
                nv_ = (-cos(yaw) * sin(tl), -sin(yaw) * sin(tl), cos(tl))
                lit[key].append(((d * cos(a), d * sin(a), z + 0.0012 * j), dv, nv_, L0 * r.u(0.9, 1.25)))
        for key, mtl in ((0, leaf_d), (1, leaf_l)):
            if lit[key]:
                arr = [np.array([it[j] for it in lit[key]]) for j in range(4)]
                _stamp(f"litter{key}", mtl, pent, *arr)
