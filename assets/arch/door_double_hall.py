"""Grand civic double door for the hall / hotel entrance. A dressed-limestone doorcase stands on two worn
stone steps: panelled pedestals, fluted pilasters (or engaged columns) with attic bases and Doric capitals carry
either a full entablature with dentil cornice and triangular pediment, an entablature over a round-arched
fanlight, or a flat hood on scrolled consoles. Inside the moulded stone architrave sits a painted timber lining
with a transom and a lit fanlight with radiating glazing bars over two framed-and-panelled leaves (stiles, rails,
raised-and-fielded panels with bolection mouldings, forged studs, bronze kick plates, lion-mask ring handles,
meeting astragal, hinges). Two small wrought-iron bracket lanterns flank the door. Origin: wall plane, ground
centre; the outer frame is exactly width x height.
"""
import bmesh
import numpy as np
from math import sin, cos, tan, atan2, asin, sqrt, radians, degrees, pi, hypot, exp
from tk import *

META = {"name": "Grand double door", "class": "component", "mount": "wall_ground", "tri_budget": 18000,
        "focus": [0.0, -0.4, 2.0]}
PARAMS = {
    "width": {"default": 3.0, "min": 2.4, "max": 4.0, "desc": "overall width of the stone doorcase incl. cornice (m)"},
    "height": {"default": 4.6, "min": 3.8, "max": 5.8, "desc": "overall height from the paving to the top of the doorcase (m)"},
    "doorcase": {"default": "pediment", "choices": ["pediment", "arched", "hood"],
                 "desc": "entablature + pediment / entablature over an arched fanlight / flat hood on consoles"},
    "order": {"default": "pilaster", "choices": ["pilaster", "column"], "desc": "fluted pilasters or engaged columns"},
    "steps": {"default": 2, "min": 0, "max": 3, "desc": "number of stone steps up to the threshold (threshold z = steps x 0.165)"},
    "paint": {"default": "#1e3a2d", "desc": "paint colour of the leaves and lining"},
    "trim": {"default": "#d8d0ba", "desc": "paint colour of the fanlight glazing bars"},
    "stone": {"default": "#bdb098", "desc": "tint of the doorcase stone"},
    "inscription": {"default": "TOWN HALL", "desc": "gilded lettering on the frieze ('' = none)"},
    "bars": {"default": 7, "min": 3, "max": 11, "desc": "radiating glazing bars in the fanlight"},
    "studs": {"default": "frame", "choices": ["frame", "rails", "none"], "desc": "forged bronze studs on stiles + rails / rails only / none"},
    "lit": {"default": True, "desc": "fanlight lit from the hall inside"},
    "intensity": {"default": 18.0, "min": 5.0, "max": 80.0, "desc": "fanlight spill (cd)"},
    "lanterns": {"default": True, "desc": "pair of wrought-iron bracket lanterns on the pilasters"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: paint wear, grime, moss"},
}
VARIANTS = {
    "hotel_arched": {"doorcase": "arched", "order": "column", "width": 3.3, "height": 4.5, "paint": "#5c1717",
                     "inscription": "GRAND HOTEL", "bars": 9, "steps": 3, "stone": "#c6b795", "seed": 4},
    "flat_hood": {"doorcase": "hood", "width": 2.6, "height": 3.8, "paint": "#1b2c48", "inscription": "",
                  "steps": 1, "studs": "rails", "bars": 5, "stone": "#b9b2a4", "seed": 11},
    "grand_columns": {"doorcase": "pediment", "order": "column", "width": 3.9, "height": 5.6, "paint": "#2b1c12",
                      "inscription": "CURIA CIVITATIS", "steps": 3, "bars": 9, "trim": "#b9a468", "seed": 19},
    "old_guild": {"doorcase": "arched", "width": 2.7, "height": 4.0, "paint": "#3d5a5c", "inscription": "",
                  "steps": 2, "lit": False, "age": 1.0, "bars": 5, "studs": "frame", "stone": "#aaa596", "seed": 27},
}

RISE = 0.165
GOING = 0.31
CP = 0.185            # cornice projection beyond the entablature face (front and sides)
DS = 0.22             # thickness of the stone facing slab (= depth of the door reveal)
AM = 0.13             # width of the moulded stone architrave round the opening
CANG = 360.0 * 9 / 14  # engaged columns: modelled arc (9 of 14 flutes), the rest is buried in the wall
CROT = 270.0 - CANG / 2


# ------------------------------------------------------------------ local mesh helpers
def _obj(name, verts, faces, sm=35.0):
    o = mesh_object(name, verts, faces)
    if sm:
        smooth(o, sm)
    return o


def mould_run(name, prof, x0, x1, d0, back=0.0, side_k=1.0, caps=(True, True), sm=35.0):
    """Horizontal moulding with mitred returns to the wall. prof = [(projection, z)] bottom to top; the body is
    x0..x1 at projection d0, anything proud of d0 also oversails the two ends (x side_k)."""
    verts, faces = [], []
    for (d, z) in prof:
        e = (d - d0) * side_k
        verts += [(x0 - e, -back, z), (x0 - e, -d, z), (x1 + e, -d, z), (x1 + e, -back, z)]
    n = len(prof)
    for j in range(n - 1):
        for i in range(3):
            a = j * 4 + i
            faces.append((a, a + 1, a + 5, a + 4))
    if caps[0]:
        faces.append((0, 3, 2, 1))
    if caps[1]:
        t = (n - 1) * 4
        faces.append((t, t + 1, t + 2, t + 3))
    return _obj(name, verts, faces, sm)


def _offset(path, w, closed, end_dirs):
    n = len(path)
    out = []

    def seg_n(a, b):
        tx, tz = b[0] - a[0], b[1] - a[1]
        l = hypot(tx, tz) or 1e-9
        return (-tz / l, tx / l)

    for i in range(n):
        q = path[i]
        if closed or 0 < i < n - 1:
            n1 = seg_n(path[(i - 1) % n], q)
            n2 = seg_n(q, path[(i + 1) % n])
            mx, mz = n1[0] + n2[0], n1[1] + n2[1]
            l = hypot(mx, mz) or 1e-9
            mx, mz = mx / l, mz / l
            k = w / max(0.3, mx * n1[0] + mz * n1[1])
        else:
            nn = seg_n(path[0], path[1]) if i == 0 else seg_n(path[-2], path[-1])
            ed = end_dirs[0 if i == 0 else 1] if end_dirs else None
            if ed:
                mx, mz = ed
                k = w / max(0.3, mx * nn[0] + mz * nn[1])
            else:
                mx, mz = nn
                k = w
        out.append((q[0] + mx * k, q[1] + mz * k))
    return out


def loft_outline(name, path, prof, closed=False, end_dirs=None, cap_ends=False, sm=35.0):
    """Moulding following a 2D path in the wall plane (x, z), travelled clockwise seen from the front.
    prof = [(w, d)]: w = offset to the outside of the path, d = projection from the wall."""
    n, m = len(path), len(prof)
    verts, faces = [], []
    for (w, d) in prof:
        for (x, z) in _offset(path, w, closed, end_dirs):
            verts.append((x, -d, z))
    for j in range(m - 1):
        for i in range(n if closed else n - 1):
            a = j * n + i
            b = j * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap_ends and not closed:
        faces.append(tuple(j * n for j in range(m)))
        faces.append(tuple(j * n + n - 1 for j in reversed(range(m))))
    o = _obj(name, verts, faces, sm)
    if cap_ends:
        recalc_normals(o)
    return o


def drop_back_faces(o):
    """Remove faces looking at the wall (+Y): never seen on applied lettering / plates."""
    import bmesh
    M = world_matrix(o).to_3x3()
    bm = bmesh.new()
    bm.from_mesh(o.data)
    kill = [f for f in bm.faces if (M @ f.normal).y > 0.7]
    if kill:
        bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(o.data)
    bm.free()


def ebox(name, x0, x1, da, db, z0, z1, bevel=0.003):
    """Box from extents; da..db are projections from the wall."""
    return box(name, (x1 - x0, db - da, z1 - z0), loc=((x0 + x1) / 2, -(da + db) / 2, (z0 + z1) / 2), bevel=bevel, segments=1)


def arc_xz(cx, cz, r, a0, a1, n):
    return [(cx + r * cos(a0 + (a1 - a0) * i / n), cz + r * sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


# ------------------------------------------------------------------ sub-assemblies
def lion_handle(name):
    """Bronze lion mask with a ring through its mouth, modelled facing +Z, y up (placed with rot 90,0,0)."""
    ps = [lathe("l_plate", [(0.105, 0.0), (0.105, 0.005), (0.097, 0.011), (0.088, 0.011), (0.084, 0.017), (0.0, 0.017)], segments=16)]
    mane = lathe("l_mane", [(0.060, 0.012), (0.089, 0.015), (0.097, 0.030), (0.087, 0.047), (0.064, 0.058)], segments=22)

    def scallop(co):
        a = atan2(co.y, co.x)
        k = 1.0 + 0.085 * cos(a * 11) - (0.10 if co.y < -0.02 else 0.0) * max(0.0, -sin(a)) ** 2
        return Vector((co.x * k, co.y * k, co.z + 0.006 * cos(a * 11)))

    transform_verts(mane, scallop)
    ps.append(mane)
    ps.append(sphere("l_head", 0.060, loc=(0, 0.006, 0.043), segments=10, rings=6, scale=(1.0, 1.05, 0.78)))
    for sx in (-1, 1):
        ps.append(sphere("l_brow", 0.022, loc=(sx * 0.027, 0.030, 0.078), segments=7, rings=4, scale=(1.15, 0.8, 0.8)))
        ps.append(sphere("l_ear", 0.019, loc=(sx * 0.050, 0.058, 0.052), segments=7, rings=4, scale=(1.0, 1.0, 0.6)))
        ps.append(sphere("l_cheek", 0.026, loc=(sx * 0.020, -0.018, 0.086), segments=7, rings=4, scale=(1.0, 0.85, 0.8)))
    ps.append(sphere("l_nose", 0.014, loc=(0, 0.000, 0.108), segments=7, rings=4, scale=(1.3, 0.9, 0.8)))
    ps.append(sphere("l_jaw", 0.024, loc=(0, -0.052, 0.070), segments=7, rings=4, scale=(1.0, 0.7, 0.9)))
    ps.append(torus("l_ring", 0.072, 0.0085, loc=(0, -0.105, 0.053), rot=(26, 0, 0), major_seg=16, minor_seg=5))
    ps.append(lathe("l_boss", [(0.017, 0.0), (0.017, 0.008), (0.009, 0.013), (0.0, 0.013)], segments=8, loc=(0, -0.172, 0.0)))
    o = join(ps, name)
    smooth(o, 60)
    return o


def lantern(x, dface, zl, round_r=0.0, axis_d=0.0):
    """Small wrought-iron bracket lantern. Returns (iron objs, glass objs, flame objs, light position)."""
    iron, glass, flame = [], [], []
    reach = 0.34
    yl = -(dface + reach)
    if round_r > 0.0:
        iron.append(ring_band("lan_collar", radius=round_r + 0.002, width=0.04, thick=0.006, loc=(x, -axis_d, zl - 0.02), segments=20))
        dface = axis_d + round_r
    iron.append(ebox("lan_plate", x - 0.03, x + 0.03, dface - 0.002, dface + 0.011, zl - 0.14, zl + 0.10, bevel=0.003))
    arm = bezier_path((x, -dface - 0.008, zl + 0.03), (x, -dface - 0.10, zl + 0.27), (x, yl + 0.0, zl + 0.29), (x, yl, zl + 0.150), n=10)
    iron.append(sweep("lan_arm", arm, 0.008, sides=6, smooth_path=False))
    sc = spiral_path(0.008, 0.062, 1.3, n=16, a0_deg=150, plane="YZ", center=(x, -dface - 0.085, zl - 0.02))
    iron.append(sweep("lan_scroll", sc, [(-0.008, -0.003), (0.008, -0.003), (0.008, 0.003), (-0.008, 0.003)], up=(1, 0, 0), smooth_path=False,
                      scale_fn=lambda t: 0.55 + 0.45 * t))
    iron.append(sweep("lan_stay", [(x, -dface - 0.008, zl - 0.10), (x, -dface - 0.06, zl - 0.085), (x, -dface - 0.118, zl + 0.03)], 0.005, sides=5, smooth_path=False))
    zr = zl + 0.04
    iron.append(torus("lan_eye", 0.014, 0.004, loc=(x, yl, zr + 0.105), rot=(0, 90, 0), major_seg=10, minor_seg=5))
    iron.append(lathe("lan_roof", [(0.084, -0.004), (0.088, 0.004), (0.054, 0.028), (0.024, 0.058), (0.014, 0.064), (0.014, 0.078), (0.021, 0.084), (0.0, 0.094)],
                      segments=4, loc=(x, yl, zr), rot=(0, 0, 45), smooth_angle=20))
    iron.append(lathe("lan_base", [(0.0, -0.232), (0.02, -0.232), (0.03, -0.218), (0.056, -0.212), (0.056, -0.2), (0.046, -0.2)],
                      segments=4, loc=(x, yl, zr), rot=(0, 0, 45), smooth_angle=20))
    glass.append(lathe("lan_glass", [(0.049, -0.203), (0.074, -0.002)], segments=4, loc=(x, yl, zr), rot=(0, 0, 45), smooth_angle=20))
    for k in range(4):
        cx, cy = (1, -1, -1, 1)[k], (1, 1, -1, -1)[k]
        rb, rt = 0.052 * 0.7071, 0.077 * 0.7071
        iron.append(sweep("lan_bar", [(x + cx * rb, yl + cy * rb, zr - 0.206), (x + cx * rt, yl + cy * rt, zr)], 0.0045, sides=4, smooth_path=False))
    iron.append(cylinder("lan_burner", 0.008, 0.07, loc=(x, yl, zr - 0.168), verts=6))
    flame.append(sphere("lan_flame", 0.017, loc=(x, yl, zr - 0.112), segments=8, rings=5, scale=(1.0, 1.0, 1.7)))
    return iron, glass, flame, (x, yl, zr - 0.112)


def build(p, rng):
    W, H = p.width, p.height
    age = p.age
    dc = p.doorcase
    col = p.order == "column"
    n_steps = int(p.steps)
    zs = n_steps * RISE                      # threshold / landing level

    stone = mat("stone_smooth", p.stone)
    step_mat = mat("stone_rough", "#a39a8a")
    wood = mat("wood_painted", p.paint)
    trim = mat("wood_painted", p.trim)
    bronze = mat("bronze_aged")
    gilt = mat("metal_plain", "#d9ae52", rough=0.5)
    iron_m = mat("iron_wrought")
    S, STEP, WD, TR, BR, GI, IR = [], [], [], [], [], [], []      # object lists per material

    # ---------------------------------------------------------------- layout
    bx = W / 2 - CP                           # half width of the entablature body = outer edge of the supports
    pw = clamp(W * 0.115, 0.27, 0.40)         # pilaster width / column diameter
    xp = bx - pw / 2
    oxh = bx - pw - AM - 0.035                # half clear width of the stone opening
    if col:
        cr = pw / 2
        dax = DS + 0.30 * cr                  # column axis projection
        df = dax + cr
        d0 = dax + 0.86 * cr - 0.004          # entablature face
    else:
        cr = 0.0
        dax = DS
        df = DS + 0.10
        d0 = df - 0.008

    wt = 0.035                                # weathered top above the cornice
    if dc == "pediment":
        ah, fh, ch, rc = 0.14, 0.26, 0.18, 0.13
        avail = H - zs - (AM + 0.03) - (ah + fh + ch) - 2.1
        pitch = clamp(atan2(avail - rc * 1.06, W / 2), radians(8), radians(21))
        prise = (W / 2) * tan(pitch)
        zc = H - prise - rc / cos(pitch)
    elif dc == "arched":
        ah, fh, ch = 0.13, 0.22, 0.18
        zc = H - wt
    else:
        ah, fh, ch = 0.0, 0.40, 0.19
        zc = H - wt
    zcap = zc - (ah + fh + ch)                # top of the capitals = bottom of the entablature
    zsill = zs + 0.028                        # top of the threshold sill

    arched = dc == "arched"
    if arched:
        zcrown = zcap - AM - 0.04
        R = oxh
        a_rise = R if (zcrown - zs - R) >= 2.15 else clamp(zcrown - zs - 2.15, 0.22, R)
        zsp = zcrown - a_rise
        rho = (R * R + a_rise * a_rise) / (2 * a_rise)
        zca = zcrown - rho
        alpha = asin(clamp((zsp - zca) / rho, -1.0, 1.0))
        arc_cw = arc_xz(0, zca, rho, pi - alpha, alpha, 20)       # left -> right over the top
        outline = [(-oxh, zs)] + arc_cw + [(oxh, zs)]
        zt0 = zsp - 0.07                       # transom zt0..zsp
        zt1 = zsp
        has_light = True
        z_ot = zcrown
    else:
        z_ot = zcap - AM - 0.03
        oh_ = z_ot - zs
        outline = [(-oxh, zs), (-oxh, z_ot), (oxh, z_ot), (oxh, zs)]
        has_light = oh_ >= 2.6
        if has_light:
            ol = clamp(oh_ - 2.45, 0.30, 0.85)
            zt0 = z_ot - 0.045 - ol - 0.07
            zt1 = zt0 + 0.07
        else:
            zt0 = zt1 = z_ot - 0.045
    LN = 0.047                                 # timber lining width
    leaf_z0 = zsill + 0.006
    leaf_h = zt0 - 0.004 - leaf_z0

    # ---------------------------------------------------------------- steps (individual stones, staggered joints)
    rs = rng.fork("steps")
    land = d0 + 0.46
    for i in range(n_steps):
        dfr = land + (n_steps - 1 - i) * GOING
        nst = max(2, int(round(W / 1.25))) + (i % 2)
        cuts = [-W / 2 + W * (k + rs.jitter(0.0, 0.16)) / nst for k in range(1, nst)]
        xs = [-W / 2] + cuts + [W / 2]
        for k in range(nst):
            xa, xb = xs[k], xs[k + 1]
            b = ebox(f"step{i}_{k}", xa + 0.003, xb - 0.003, 0.0, dfr, i * RISE, (i + 1) * RISE, bevel=0.017)
            apply_transform(b)
            densify(b, 0.3)
            displace_noise(b, strength=0.0035, scale=0.35, seed=p.seed + 31 * i + k)
            ztop = (i + 1) * RISE

            def dish(co, ztop=ztop, dfr=dfr):
                if co.z < ztop - 0.02:
                    return co
                g = exp(-(co.x / 0.5) ** 2) * smoothstep(dfr - 0.36, dfr - 0.12, -co.y)
                return Vector((co.x, co.y, min(co.z, ztop + 0.0015) - 0.007 * g))

            transform_verts(b, dish)
            STEP.append(b)
    # threshold sill between the jambs
    sill = ebox("sill", -(oxh - 0.002), oxh - 0.002, 0.0, DS + 0.035, zs, zsill, bevel=0.007)
    densify(sill, 0.3)
    STEP.append(sill)

    # ---------------------------------------------------------------- stone facing slab with the opening
    sx = bx + 0.05
    poly = [(-sx, zs)] + outline + [(sx, zs), (sx, zcap), (-sx, zcap)]
    S.append(extrude_profile("facing", poly, DS, loc=(0, 0, 0), rot=(90, 0, 0)))
    # moulded stone architrave round the opening
    arch_prof = [(0.0, DS), (0.0, DS + 0.02), (0.03, DS + 0.02), (0.037, DS + 0.032), (0.082, DS + 0.032), (0.092, DS + 0.04),
                 (0.105, DS + 0.055), (0.124, DS + 0.055), (AM, DS + 0.045), (AM, DS)]
    S.append(loft_outline("architrave", outline, arch_prof))

    # ---------------------------------------------------------------- supports: pedestal, base, shaft, capital
    hsup = zcap - zs
    ph = clamp(hsup * 0.17, 0.36, 0.62)
    zb0 = zs + ph
    bh = 0.13
    caph = 0.20
    for sgn in (-1, 1):
        x0, x1 = sgn * xp - pw / 2, sgn * xp + pw / 2
        dp = df + 0.035
        bk = DS - 0.01
        ped = [(dp + 0.032, zs), (dp + 0.032, zs + 0.09), (dp + 0.024, zs + 0.102), (dp + 0.006, zs + 0.125), (dp, zs + 0.14),
               (dp, zb0 - 0.09), (dp + 0.008, zb0 - 0.078), (dp + 0.03, zb0 - 0.058), (dp + 0.037, zb0 - 0.045),
               (dp + 0.037, zb0 - 0.014), (dp + 0.03, zb0)]
        S.append(mould_run("pedestal", ped, x0 - 0.035, x1 + 0.035, dp, back=bk))
        # sunk panel on the pedestal die
        qx, qz0, qz1 = pw / 2 - 0.02, zs + 0.175, zb0 - 0.125
        S.append(loft_outline("ped_panel", [(sgn * xp - qx, qz0), (sgn * xp - qx, qz1), (sgn * xp + qx, qz1), (sgn * xp + qx, qz0)],
                              [(-0.036, dp), (-0.03, dp + 0.009), (-0.008, dp + 0.009), (0.0, dp)], closed=True))
        z_sh0, z_sh1 = zb0 + bh, zcap - caph
        if not col:
            base = [(df + 0.03, zb0), (df + 0.03, zb0 + 0.03), (df + 0.035, zb0 + 0.036), (df + 0.041, zb0 + 0.049), (df + 0.035, zb0 + 0.062),
                    (df + 0.02, zb0 + 0.066), (df + 0.014, zb0 + 0.078), (df + 0.018, zb0 + 0.088), (df + 0.023, zb0 + 0.092),
                    (df + 0.025, zb0 + 0.1), (df + 0.018, zb0 + 0.108), (df + 0.004, zb0 + 0.116), (df, zb0 + bh)]
            S.append(mould_run("pil_base", base, x0, x1, df, back=bk, caps=(False, False)))
            # fluted shaft cross-section
            nf = 5 if pw < 0.36 else 6
            pitch_f = pw * 0.78 / nf
            rf = pitch_f * 0.36
            sec = [(-pw / 2, -bk), (-pw / 2, -df)]
            for k in range(nf):
                xc = -pw * 0.39 + (k + 0.5) * pitch_f
                for j in range(5):
                    th = pi - pi * j / 4
                    sec.append((xc + rf * cos(th), -df + rf * 0.85 * sin(th)))
            sec += [(pw / 2, -df), (pw / 2, -bk)]
            rj = rng.fork("joints%d" % sgn)
            nseg = max(3, int(round((z_sh1 - z_sh0) / 0.6)))
            zj = [z_sh0] + [z_sh0 + (z_sh1 - z_sh0) * (k + rj.jitter(0.0, 0.12)) / nseg for k in range(1, nseg)] + [z_sh1]
            for k in range(nseg):
                g0 = 0.0 if k == 0 else 0.0015
                g1 = 0.0 if k == nseg - 1 else 0.0015
                S.append(extrude_profile("pil_shaft", sec, zj[k + 1] - zj[k] - g0 - g1, loc=(sgn * xp, 0, zj[k] + g0), cap=True))
            z1 = zcap
            capp = [(df, z1 - 0.20), (df + 0.012, z1 - 0.195), (df + 0.017, z1 - 0.185), (df + 0.012, z1 - 0.175), (df, z1 - 0.17),
                    (df, z1 - 0.122), (df + 0.008, z1 - 0.116), (df + 0.008, z1 - 0.106), (df + 0.02, z1 - 0.096), (df + 0.036, z1 - 0.076),
                    (df + 0.044, z1 - 0.058), (df + 0.05, z1 - 0.056), (df + 0.05, z1 - 0.014), (df + 0.057, z1 - 0.009), (df + 0.057, z1)]
            S.append(mould_run("pil_cap", capp, x0, x1, df, back=bk, caps=(False, True)))
            S.append(lathe("neck_rose", [(0.0, 0.0), (0.03, 0.0), (0.033, 0.005), (0.022, 0.011), (0.012, 0.009), (0.008, 0.016), (0.0, 0.019)],
                           segments=10, loc=(sgn * xp, -df, z1 - 0.146), rot=(90, 0, 0)))
        else:
            S.append(ebox("col_plinth", sgn * xp - cr - 0.035, sgn * xp + cr + 0.035, bk, dax + cr + 0.035, zb0, zb0 + 0.04, bevel=0.004))
            base = [(cr + 0.034, zb0 + 0.04), (cr + 0.04, zb0 + 0.052), (cr + 0.034, zb0 + 0.066), (cr + 0.02, zb0 + 0.07), (cr + 0.014, zb0 + 0.082),
                    (cr + 0.018, zb0 + 0.092), (cr + 0.024, zb0 + 0.1), (cr + 0.018, zb0 + 0.11), (cr + 0.004, zb0 + 0.118), (cr, zb0 + bh)]
            S.append(lathe("col_base", base, segments=15, loc=(sgn * xp, -dax, 0), rot=(0, 0, CROT), angle=CANG))
            nfl = 14
            nz = 6
            rj = rng.fork("joints%d" % sgn)
            sp = []
            for i in range(nz + 1):
                zz = z_sh0 + (z_sh1 - z_sh0) * i / nz
                rr = cr * (1 - 0.14 * (i / nz) ** 1.7)
                if 0 < i < nz and i % 2 == 0:
                    zz += rj.jitter(0.0, 0.05)
                    sp += [(rr, zz - 0.003), (rr - 0.006, zz), (rr, zz + 0.003)]
                else:
                    sp.append((rr, zz))
            sh = lathe("col_shaft", sp, segments=36, loc=(sgn * xp, -dax, 0), rot=(0, 0, CROT), angle=CANG, smooth_angle=30)

            def flute(co, nfl=nfl):
                a = atan2(co.y, co.x)
                c = 0.5 + 0.5 * cos(a * nfl)
                r = hypot(co.x, co.y)
                k = (r - 0.012 * (1.0 - c ** 2)) / r if r > 1e-6 else 1.0
                return Vector((co.x * k, co.y * k, co.z))

            transform_verts(sh, flute)
            S.append(sh)
            rt = cr * 0.86
            z1 = zcap
            capp = [(rt, z1 - 0.20), (rt + 0.012, z1 - 0.195), (rt + 0.016, z1 - 0.185), (rt + 0.012, z1 - 0.175), (rt, z1 - 0.17), (rt, z1 - 0.122),
                    (rt + 0.008, z1 - 0.116), (rt + 0.008, z1 - 0.106), (rt + 0.022, z1 - 0.094), (rt + 0.04, z1 - 0.074), (rt + 0.048, z1 - 0.056),
                    (rt + 0.03, z1 - 0.056)]
            S.append(lathe("col_cap", capp, segments=15, loc=(sgn * xp, -dax, 0), rot=(0, 0, CROT), angle=CANG))
            S.append(ebox("col_abacus", sgn * xp - rt - 0.052, sgn * xp + rt + 0.052, bk, dax + rt + 0.052, z1 - 0.056, z1, bevel=0.004))

    # ---------------------------------------------------------------- entablature / hood
    z0 = zcap
    zf0 = z0 + ah
    zf1 = zf0 + fh
    prof = []
    if ah > 0:
        prof += [(d0, z0), (d0, z0 + ah * 0.38), (d0 + 0.011, z0 + ah * 0.42), (d0 + 0.011, z0 + ah * 0.76), (d0 + 0.02, z0 + ah * 0.8),
                 (d0 + 0.034, z0 + ah * 0.9), (d0 + 0.034, zf0), (d0, zf0)]
    else:
        prof += [(d0, z0)]
    prof += [(d0, zf1)]
    if dc == "hood":
        HP = 0.36
        prof += [(d0 + 0.02, zf1), (d0 + 0.02, zf1 + 0.03), (d0 + 0.034, zf1 + 0.044), (d0 + 0.06, zf1 + 0.052), (d0 + HP - 0.04, zf1 + 0.062),
                 (d0 + HP - 0.04, zf1 + 0.112), (d0 + HP - 0.03, zf1 + 0.118), (d0 + HP - 0.01, zf1 + 0.145), (d0 + HP, zf1 + 0.165),
                 (d0 + HP, zf1 + ch), (0.03, zc + wt)]
        S.append(mould_run("hood", prof, -bx, bx, d0, side_k=CP / HP))
    else:
        prof += [(d0 + 0.02, zf1), (d0 + 0.02, zf1 + 0.066), (d0 + 0.036, zf1 + 0.072), (d0 + 0.07, zf1 + 0.082), (d0 + 0.15, zf1 + 0.088),
                 (d0 + 0.15, zf1 + 0.128), (d0 + 0.157, zf1 + 0.133), (d0 + 0.165, zf1 + 0.152), (d0 + CP, zf1 + 0.166), (d0 + CP, zf1 + ch)]
        if dc == "arched":
            prof += [(0.03, zc + wt)]
        S.append(mould_run("entablature", prof, -bx, bx, d0))
        # dentil course
        den = box("dentil", (0.05, 0.05, 0.05))
        nd = max(4, int(2 * bx / 0.098))
        pd = 2 * bx / nd
        mats = [trs((-bx + (k + 0.5) * pd, -(d0 + 0.042), zf1 + 0.036)) for k in range(nd)]
        nds = max(1, int((d0 - 0.02) / 0.098))
        for sgn in (-1, 1):
            mats += [trs((sgn * (bx + 0.022), -(d0 - (k + 0.5) * 0.098), zf1 + 0.036)) for k in range(nds)]
        S.append(instances_to_mesh("dentils", den, mats, keep_base=False))
        for sgn in (-1, 1):
            S.append(lathe("frieze_rose", [(0.0, 0.0), (0.062, 0.0), (0.066, 0.006), (0.05, 0.014), (0.034, 0.01), (0.022, 0.02), (0.01, 0.024), (0.0, 0.03)],
                           segments=12, loc=(sgn * xp, -d0, (zf0 + zf1) / 2), rot=(90, 0, 0)))

    if dc == "hood":
        # scrolled consoles carrying the hood, over the pilasters
        cw = pw * 0.76
        zA, zB = zcap + 0.01, zf1 + 0.045
        for sgn in (-1, 1):
            pts = [(d0 - 0.02, zA), (d0 + 0.05, zA)]
            for i in range(11):
                t = i / 10
                pts.append((d0 + 0.05 + 0.2 * smoothstep(0.12, 0.9, t) + 0.022 * sin(pi * t) - 0.03 * sin(pi * t) ** 6, zA + (zB - zA) * t))
            pts += [(d0 - 0.02, zB)]
            c = extrude_profile("console", pts, cw, bevel=0.004, segments=1)
            xr = sgn * xp + cw / 2
            transform_verts(c, lambda co, xr=xr: Vector((xr - co.z, -co.x, co.y)))
            S.append(c)
            S.append(cylinder("console_roll", 0.074, cw + 0.02, loc=(sgn * xp, -(d0 + 0.2), zB - 0.105), rot=(0, 90, 0), verts=16, bevel=0.005, segments=1))
            S.append(cylinder("console_roll", 0.042, cw + 0.02, loc=(sgn * xp, -(d0 + 0.062), zA + 0.05), rot=(0, 90, 0), verts=12, bevel=0.004, segments=1))
            for k in range(3):
                S.append(ebox("console_rib", sgn * xp - cw * 0.34 + k * cw * 0.34 - 0.012, sgn * xp - cw * 0.34 + k * cw * 0.34 + 0.012,
                              d0 + 0.09, d0 + 0.118, zA + 0.12, zA + 0.2, bevel=0.004))
        # sunk frieze panel between the consoles
        S.append(shape_extrude("frieze_panel", rounded_rect(2 * (bx - pw) - 0.1, fh - 0.1, 0.005, 1), [rounded_rect(2 * (bx - pw) - 0.17, fh - 0.17, 0.005, 1)],
                               depth=0.012, loc=(0, -d0, (zf0 + zf1) / 2), rot=(90, 0, 0), bevel=0.003))

    if dc == "pediment":
        tym = [(-W / 2 + 0.03, zc - 0.005), (W / 2 - 0.03, zc - 0.005), (0.0, zc + prise)]
        tymo = extrude_profile("tympanum", tym, d0, rot=(90, 0, 0))
        densify(tymo, 0.3)
        S.append(tymo)
        rprof = [(0.0, 0.0), (0.0, d0 + 0.02), (0.02, d0 + 0.03), (0.034, d0 + 0.085), (0.04, d0 + 0.15), (0.082, d0 + 0.15), (0.092, d0 + 0.162),
                 (0.122, d0 + CP), (rc, d0 + CP), (rc, 0.0)]
        rpath = [(-W / 2, zc), (0.0, zc + prise), (W / 2, zc)]
        S.append(loft_outline("raking_cornice", rpath, rprof, end_dirs=((0, 1), (0, 1)), cap_ends=True))
        # roundel with a bronze star in the tympanum
        ro = clamp(prise * 0.27, 0.07, 0.2)
        zr = zc + prise * 0.36
        S.append(lathe("roundel", [(ro * 0.66, 0.0), (ro * 0.7, 0.018), (ro * 0.8, 0.028), (ro * 0.93, 0.022), (ro, 0.008), (ro, 0.0)], segments=24,
                       loc=(0, -d0, zr), rot=(90, 0, 0)))
        star = []
        for k in range(16):
            rr = ro * (0.6 if k % 2 == 0 else 0.27)
            star.append((rr * sin(k * pi / 8), rr * cos(k * pi / 8)))
        BR.append(shape_extrude("star", star[::-1], depth=0.012, loc=(0, -d0, zr), rot=(90, 0, 0), bevel=0.002))
        BR.append(lathe("star_boss", [(ro * 0.2, 0.0), (ro * 0.2, 0.014), (ro * 0.12, 0.024), (0.0, 0.028)], segments=10, loc=(0, -d0, zr), rot=(90, 0, 0)))

    if arched:
        # keystone and spandrel paterae
        kb, kt = zcrown - 0.012, zcap + 0.0
        S.append(extrude_profile("keystone", [(-0.065, kb), (0.065, kb), (0.1, kt), (-0.1, kt)], 0.125, loc=(0, -(DS - 0.01), 0), rot=(90, 0, 0), bevel=0.008, segments=1))
        S.append(extrude_profile("keystone_face", [(-0.035, kb + 0.03), (0.035, kb + 0.03), (0.06, kt - 0.03), (-0.06, kt - 0.03)], 0.02, loc=(0, -(DS + 0.108), 0),
                                 rot=(90, 0, 0), bevel=0.005, segments=1))
        xq = bx - pw
        for sgn in (-1, 1):
            cxq, czq = xq - 0.02, zcap - 0.02
            dist = hypot(cxq, czq - zca)
            rsq = min(0.12, (dist - rho - AM) * 0.36)
            if rsq > 0.035:
                ux, uz = cxq / dist, (czq - zca) / dist
                px, pz = cxq - ux * rsq * 1.5, czq - uz * rsq * 1.5
                S.append(lathe("patera", [(0.0, 0.0), (rsq, 0.0), (rsq, 0.008), (rsq * 0.84, 0.016), (rsq * 0.7, 0.01), (rsq * 0.5, 0.022), (rsq * 0.2, 0.03), (0.0, 0.03)],
                               segments=16, loc=(sgn * px, -DS, pz), rot=(90, 0, 0)))
        # impost blocks at the springing
        for sgn in (-1, 1):
            xi0, xi1 = sorted((sgn * (oxh - 0.004), sgn * (oxh + AM + 0.012)))
            S.append(mould_run("impost", [(DS + 0.05, zsp - 0.075), (DS + 0.062, zsp - 0.06), (DS + 0.062, zsp - 0.03), (DS + 0.074, zsp - 0.012), (DS + 0.074, zsp + 0.012)],
                               xi0, xi1, DS + 0.062, back=DS - 0.005, side_k=1.0))

    # ---------------------------------------------------------------- frieze inscription
    if p.inscription.strip():
        target = fh * (0.5 if dc != "hood" else 0.34)
        dt = d0 + (0.004 if dc == "hood" else 0.0)
        t = text("inscription", p.inscription.strip().upper(), size=target, depth=0.009, loc=(0, -dt, 0), font="roman", spacing=1.12)
        drop_back_faces(t)
        mn, mx = world_bbox([t])
        k = min(target / max(mx.z - mn.z, 1e-3), (2 * (bx - pw) - 0.2) / max(mx.x - mn.x, 1e-3))
        place(t, loc=(-(mn.x + mx.x) / 2 * k, -dt, (zf0 + zf1) / 2 - (mn.z + mx.z) / 2 * k), scale=(k, k, 1.0))
        GI.append(t)

    # ---------------------------------------------------------------- timber lining, transom, fanlight
    lin_prof = [(-LN, 0.012), (-LN, 0.088), (-LN + 0.008, 0.096), (-0.012, 0.096), (0.0, 0.088), (0.0, 0.012)]
    WD.append(loft_outline("lining", outline, lin_prof))
    if has_light:
        tr0, tr1 = -(oxh - LN + 0.004), (oxh - LN + 0.004)
        WD.append(mould_run("transom", [(0.1, zt0), (0.1, zt0 + 0.018), (0.108, zt0 + 0.024), (0.122, zt0 + 0.03), (0.122, zt1 - 0.012), (0.1, zt1 - 0.004), (0.096, zt1)],
                            tr0, tr1, 0.1, back=0.012, side_k=0.0))
        # glass
        a_in = oxh - LN + 0.006
        glass_m = mat("window_lit", "#ffc37a", glow=2.6, curtain=0.0) if p.lit else mat("window_dark")
        if arched:
            rg = rho - LN + 0.006
            al2 = asin(clamp((zt1 - zca) / rg, -1.0, 1.0))
            gpts = arc_xz(0, zca, rg, al2, pi - al2, 20)
            g = extrude_profile("fan_glass", gpts, 0.006, loc=(0, -0.036, 0), rot=(90, 0, 0))

            def bound(th):
                cz = zca - zt1
                uc = cz * sin(th)
                return uc + sqrt(max(0.0, uc * uc - cz * cz + (rho - LN) ** 2))
            b_in = zcrown - LN - zt1
        else:
            b_in = z_ot - LN - zt1
            g = ebox("fan_glass", -a_in, a_in, 0.036, 0.042, zt1 - 0.004, z_ot - LN + 0.006, bevel=0.0)

            def bound(th):
                c, s = abs(cos(th)), sin(th)
                return min((oxh - LN) / max(c, 1e-6), b_in / max(s, 1e-6))
        # the lit glass is ONE sheet (the front face, 42 mm out): a thin glowing box is two sheets that fight each other.
        # CLEARANCE RULE: nothing lies parallel within 4 mm of it - hub and bars run through it to 4.5 mm behind, the ring stands 7 mm in front
        apply_transform(g)
        bm = bmesh.new()
        bm.from_mesh(g.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > -0.5], context="FACES")
        bm.to_mesh(g.data)
        bm.free()
        assign(g, glass_m)
        # radiating bars from a hub on the transom, an intermediate ring and a rim
        rh = clamp(0.2 * b_in, 0.07, 0.125)
        hub = arc_xz(0, zt1, rh, 0, pi, 12)
        TR.append(extrude_profile("fan_hub", hub, 0.0385, loc=(0, -0.0375, 0), rot=(90, 0, 0), bevel=0.003, segments=1))
        BR.append(lathe("fan_boss", [(0.0, 0.0), (rh * 0.3, 0.0), (rh * 0.32, 0.004), (rh * 0.2, 0.01), (rh * 0.1, 0.008), (0.0, 0.014)], segments=10, angle=360,
                        loc=(0, -0.076, zt1 + rh * 0.45), rot=(90, 0, 0)))
        nb = int(p.bars)
        for k in range(1, nb + 1):
            th = pi * k / (nb + 1)
            L = bound(th) - rh + 0.03
            rm = rh - 0.012 + L / 2
            TR.append(box("fan_bar", (L, 0.0355, 0.018), loc=(rm * cos(th), -0.05525, zt1 + rm * sin(th)), rot=(0, -degrees(th), 0), bevel=0.003, segments=1))
        ring = []
        for i in range(25):
            th = pi * i / 24
            if arched:
                rr = rh + (bound(th) - rh) * 0.52
            else:
                ea, eb = (oxh - LN) * 0.66, b_in * 0.62
                rr = 1.0 / sqrt((cos(th) / ea) ** 2 + (sin(th) / eb) ** 2)
            ring.append((rr * cos(th), -0.058, zt1 + rr * sin(th)))
        TR.append(sweep("fan_ring", ring, [(-0.015, -0.009), (0.015, -0.009), (0.015, 0.009), (-0.015, 0.009)], up=(0, -1, 0), smooth_path=False, cap=False))
        if p.lit:
            # shadow_near spans the whole doorcase: the glow must not project the pilasters,
            # pediment and bracket lanterns onto the wall as giant silhouettes
            zfl = zt1 + b_in * 0.45
            light("fanlight", (0, -0.75, zfl), intensity=p.intensity, range=7.0, flicker=0.04,
                  shadow_near=hypot(W / 2, H - zfl) + 0.35)

    # ---------------------------------------------------------------- the two leaves
    studs_m, rivets_m = [], []
    rl = rng.fork("leaves")
    backing = ebox("backing", -(oxh - 0.01), oxh - 0.01, 0.004, 0.014, zs + 0.01, zt0 + 0.02, bevel=0.0)
    WD.append(backing)
    lions = []
    for sgn in (-1, 1):
        xa, xb = (-(oxh - LN - 0.003), -0.002) if sgn < 0 else (0.002, oxh - LN - 0.003)
        lw = xb - xa
        sag = rl.jitter(0.0, 0.002)
        zl0 = leaf_z0 + abs(sag)
        lh = leaf_h - abs(sag) - 0.001
        sw = clamp(lw * 0.15, 0.085, 0.125)
        fa, fb = 0.02, 0.075
        pt = clamp(lh * 0.15, 0.26, 0.42)
        rails = [(0.0, 0.25), (0.93, 1.15), (lh - 0.115 - pt - 0.105, lh - 0.115 - pt), (lh - 0.115, lh)]
        parts = []
        for (ex0, ex1) in ((xa, xa + sw), (xb - sw, xb)):
            o = ebox("stile", ex0, ex1, fa, fb, zl0, zl0 + lh, bevel=0.003)
            uv_grain(o, "Z")
            parts.append(o)
        for (r0, r1) in rails:
            o = ebox("rail", xa + sw, xb - sw, fa, fb - 0.0008, zl0 + r0, zl0 + r1, bevel=0.003)
            uv_grain(o, "X")
            parts.append(o)
        for o in parts:
            densify(o, 0.42 if o.name.startswith("stile") else 0.6)
        px0, px1 = xa + sw, xb - sw
        for k in range(3):
            pz0, pz1 = zl0 + rails[k][1], zl0 + rails[k + 1][0]
            o = ebox("panel", px0 - 0.004, px1 + 0.004, 0.03, 0.046, pz0 - 0.004, pz1 + 0.004, bevel=0.0)
            uv_grain(o, "Z")
            parts.append(o)
            ins = min(0.06, (px1 - px0) * 0.16)
            o = ebox("field", px0 + ins, px1 - ins, 0.04, 0.066, pz0 + ins, pz1 - ins, bevel=0.015)
            uv_grain(o, "Z")
            parts.append(o)
            mprof = [(-0.042, 0.046), (-0.034, 0.068), (-0.02, 0.084), (-0.006, 0.084), (0.007, 0.0745)]
            parts.append(loft_outline("bolection", [(px0, pz0), (px0, pz1), (px1, pz1), (px1, pz0)], mprof, closed=True))
        if sgn > 0:
            o = ebox("astragal", -0.018, 0.022, fb - 0.002, fb + 0.014, zl0 + 0.0, zl0 + lh, bevel=0.006)
            uv_grain(o, "Z")
            parts.append(o)
        WD += parts
        # bronze kick plate with rivets
        kx0, kx1 = xa + 0.02, xb - 0.02
        if sgn > 0:
            kx0 = xa + 0.03
        kp = ebox("kick_plate", kx0, kx1, fb - 0.001, fb + 0.0035, zl0 + 0.025, zl0 + 0.215, bevel=0.0015)
        densify(kp, 0.3)
        displace_noise(kp, strength=0.0012, scale=0.09, seed=p.seed + int(sgn) + 3, axis_mask=(0, 1, 0))
        BR.append(kp)
        nr = max(3, int((kx1 - kx0) / 0.15))
        for k in range(nr + 1):
            xk = kx0 + 0.018 + (kx1 - kx0 - 0.036) * k / nr
            for zk in (zl0 + 0.043, zl0 + 0.197):
                rivets_m.append(trs((xk, -(fb + 0.003), zk), (90, 0, 0), 0.55))
        # studs
        if p.studs != "none":
            sp_ = max(0.15, lh / 17.0)
            cols = []
            if p.studs == "frame":
                for xc in (xa + sw / 2, xb - sw / 2 - (0.012 if sgn < 0 else 0.0)):
                    if sgn > 0 and xc < xa + sw:
                        xc = xa + sw / 2 + 0.014
                    nn = int((lh - 0.25 - 0.14) / sp_)
                    for k in range(nn + 1):
                        cols.append((xc, zl0 + 0.25 + 0.07 + (lh - 0.25 - 0.14) * k / nn))
            for ri, (r0, r1) in enumerate(rails[1:], start=1):
                zc_ = zl0 + (r0 + r1) / 2
                nn = max(1, int((px1 - px0 - 0.1) / sp_))
                for k in range(nn + 1):
                    xk = px0 + 0.05 + (px1 - px0 - 0.1) * k / nn
                    if ri == 1 and abs(xk - (xa + xb) / 2) < 0.135:
                        continue
                    cols.append((xk, zc_))
            for (xk, zk) in cols:
                if rl.chance(0.02 * age):
                    continue
                studs_m.append(trs((xk + rl.jitter(0, 0.0015), -(fb - 0.001), zk + rl.jitter(0, 0.0015)), (90, 0, rl.u(-6, 6)), rl.u(0.92, 1.08)))
        # lion-mask ring handle on the lock rail
        lions.append(((xa + xb) / 2, -(fb - 0.001), zl0 + 1.045))
        # hinges
        xh = xa - 0.006 if sgn < 0 else xb + 0.006
        for zh in (zl0 + 0.22, zl0 + lh * 0.5, zl0 + lh - 0.22):
            IR.append(lathe("hinge", [(0.0, -0.082), (0.006, -0.078), (0.008, -0.07), (0.011, -0.064), (0.011, 0.064), (0.008, 0.07), (0.006, 0.078), (0.0, 0.082)],
                            segments=6, loc=(xh, -(fb + 0.008), zh)))
        if sgn > 0:
            # escutcheon and keyhole cover on the meeting stile
            xe = xa + sw * 0.62
            BR.append(rounded_box("escutcheon", (0.042, 0.005, 0.15), 0.012, loc=(xe, -(fb + 0.002), zl0 + 1.04), seg=3, bevel=0.0015))
            BR.append(lathe("key_rose", [(0.0, 0.0), (0.016, 0.0), (0.016, 0.006), (0.01, 0.01), (0.0, 0.011)], segments=10, loc=(xe, -(fb + 0.0045), zl0 + 1.07), rot=(90, 0, 0)))
            BR.append(rounded_box("key_drop", (0.014, 0.005, 0.036), 0.006, loc=(xe + 0.002, -(fb + 0.0065), zl0 + 1.005), rot=(0, 8, 0), seg=2))
    lion = lion_handle("lion")
    for i, (lx, ly, lz) in enumerate(lions):
        o = lion if i == 0 else duplicate(lion, "lion_r")
        place(o, loc=(lx, ly, lz), rot=(90, 0, 0))
        BR.append(o)
    stud = lathe("stud", [(0.0135, 0.0), (0.0135, 0.003), (0.0, 0.013)], segments=4, rot=(0, 0, 45), smooth_angle=20)
    realize(stud)
    if studs_m:
        BR.append(instances_to_mesh("studs", stud, studs_m, keep_base=True))
    if rivets_m:
        BR.append(instances_to_mesh("rivets", stud, rivets_m, keep_base=True))
    delete([stud])

    # ---------------------------------------------------------------- bracket lanterns
    zl = min(zs + 2.3, zcap - 0.62)
    rl_ = 0.0
    if col:
        tt = clamp((zl - 0.02 - z_sh0) / max(z_sh1 - z_sh0, 1e-3), 0.0, 1.0)
        rl_ = cr * (1 - 0.14 * tt ** 1.7)
    if p.lanterns:
        gl, fl = [], []
        for sgn in (-1, 1):
            ir, g2, f2, lp = lantern(sgn * xp, df, zl, round_r=rl_, axis_d=dax)
            IR += ir
            gl += g2
            fl += f2
            light("lantern_l" if sgn < 0 else "lantern_r", lp, intensity=(1.8 if (p.lit and has_light) else 4.5), range=5.0, flicker=0.14, shadow_near=0.4)
        for o in gl:
            assign(o, mat("glass_lantern", glow=0.9))
        for o in fl:
            assign(o, mat("emissive", "#ffd9a0", glow=7))
    else:
        for sgn in (-1, 1):
            socket("lantern_l" if sgn < 0 else "lantern_r", "attach", (sgn * xp, -df, zl), role="lantern")
    socket("threshold", "attach", (0, -DS, zsill), role="door")
    socket("sign", "attach", (0, -(d0 + CP), zc), role="sign")

    # ---------------------------------------------------------------- materials and weathering
    sd = p.seed
    for o in S:
        assign(o, stone)
        paint(o, grime=mask_mul(height_ramp(zs, zs + 1.1), noise_mask(0.5, seed=sd, lo=0.1, hi=0.8), 0.35 + 0.5 * age),
              special=mask_mul(facing((0, 0, 1)), noise_mask(0.3, seed=sd + 1, lo=0.3, hi=0.7), 0.25 + 0.75 * age))

    def tread_wear(co, nrm):
        up = np.clip(nrm[:, 2], 0, 1)
        return up * np.exp(-(co[:, 0] / 0.6) ** 2) * (0.15 + 0.3 * age)

    def tread_moss(co, nrm):
        up = np.clip(nrm[:, 2], 0, 1)
        side = np.clip((np.abs(co[:, 0]) - W * 0.28) / (W * 0.2), 0, 1)
        return up * side * noise_mask(0.22, seed=sd + 5, lo=0.3, hi=0.7)(co, nrm) * (0.3 + 0.7 * age)

    for o in STEP:
        assign(o, step_mat)
        paint(o, wear=tread_wear, special=tread_moss, grime=mask_mul(noise_mask(0.8, seed=sd + 6, lo=0.4, hi=0.9), 0.1 + 0.25 * age))

    zh_ = leaf_z0 + 1.1

    def leaf_wear(co, nrm):
        hands = np.exp(-(((np.abs(co[:, 0]) - 0.2) / 0.3) ** 2 + ((co[:, 2] - zh_) / 0.28) ** 2))
        kicks = np.clip(1.0 - (co[:, 2] - leaf_z0) / 0.4, 0, 1)
        nz_ = noise_mask(0.07, seed=sd + 8, lo=0.25, hi=0.75)(co, nrm)
        return np.clip((0.7 * hands + 0.6 * kicks) * nz_ * (0.25 + 1.1 * age), 0, 1)

    for o in WD:
        assign(o, wood)
        paint(o, wear=leaf_wear, grime=mask_mul(height_ramp(zs, zs + 0.7), 0.3 + 0.4 * age),
              special=mask_mul(noise_mask(0.7, seed=sd + 9), 0.2 + 0.6 * age))
    for o in TR:
        assign(o, trim)
        paint(o, grime=0.15 + 0.3 * age, wear=mask_mul(noise_mask(0.05, seed=sd + 10, lo=0.5, hi=0.8), age * 0.6))
    for o in BR:
        assign(o, bronze)
        paint(o, wear=mask_mul(noise_mask(0.03, seed=sd + 11, lo=0.3, hi=0.7), 0.75),
              special=mask_mul(noise_mask(0.08, seed=sd + 12, lo=0.4, hi=0.8), 0.15 + 0.6 * age))
    for o in GI:
        assign(o, gilt)
        paint(o, grime=0.1 + 0.3 * age)
    for o in IR:
        assign(o, iron_m)
        paint(o, special=mask_mul(noise_mask(0.05, seed=sd + 13, lo=0.4, hi=0.8), 0.15 + 0.5 * age))
