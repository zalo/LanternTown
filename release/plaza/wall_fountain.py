"""Wall-mounted drinking fountain built against a house wall: a dressed-stone aedicule (rough plinth, two
battered pilasters of stacked blocks, an arched shell-headed niche with voussoirs and keystone, inscribed
frieze, moulded cornice and a pediment / stepped / segmental crown) carrying a cast-bronze mask (lion,
green man or plain rosette) whose spout pipe throws a water stream into a stone trough (coursed
semicircular, canted or rectangular tank, or a gadrooned bowl on a baluster). Overflow pipe with a
stained streak down to an iron drain grate, moss on the damp stone, wall lanterns on the pilasters.
"""
from tk import *
import math
import numpy as np

META = {"name": "Wall fountain with mask spout", "class": "structure", "mount": "wall_ground", "focus": 0.47}
PARAMS = {
    "width": {"default": 1.6, "min": 1.3, "max": 2.4, "desc": "width of the stone back panel across the pilasters (m)"},
    "height": {"default": 2.8, "min": 2.3, "max": 3.6, "desc": "overall height including the crown (m)"},
    "mask": {"default": "lion", "choices": ["lion", "green_man", "rosette"], "desc": "bronze spout mask style"},
    "basin": {"default": "round", "choices": ["round", "canted", "trough", "bowl"],
              "desc": "trough shape: coursed semicircular / canted / rectangular tank, or a gadrooned bowl on a baluster"},
    "crown": {"default": "pediment", "choices": ["pediment", "stepped", "segmental"], "desc": "top of the back panel"},
    "moss": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "amount of moss, algae and damp staining"},
    "stone": {"default": "#d6c3a0", "desc": "tint of the dressed stone"},
    "inscription": {"default": "AQVA PVRA", "desc": "text cut into the frieze tablet (empty = plain)"},
    "lanterns": {"default": 2, "choices": [0, 1, 2], "desc": "wall lanterns on the pilasters"},
    "light": {"default": 2.6, "min": 0.3, "max": 20.0, "desc": "intensity of each lantern flame (cd)"},
    "running": {"default": True, "desc": "water running from the spout and the overflow"},
}
VARIANTS = {
    "green_man_trough": {"width": 2.15, "height": 2.6, "mask": "green_man", "basin": "trough", "crown": "stepped",
                         "moss": 0.85, "stone": "#b9b3a2", "inscription": "FONS CIVIVM", "lanterns": 2, "seed": 7},
    "canted_segmental": {"width": 1.8, "height": 3.15, "mask": "lion", "basin": "canted", "crown": "segmental",
                         "moss": 0.2, "stone": "#e0cfae", "inscription": "MDCCCXLII", "seed": 13},
    "bowl_small": {"width": 1.3, "height": 2.4, "mask": "rosette", "basin": "bowl", "crown": "stepped", "moss": 0.3,
                   "stone": "#cdaa90", "inscription": "", "lanterns": 1, "seed": 21},
    "mossy_old": {"width": 1.7, "height": 2.9, "mask": "green_man", "basin": "round", "crown": "pediment", "moss": 1.0,
                  "stone": "#a9a48f", "inscription": "SITIENTES VENITE", "lanterns": 2, "seed": 34},
}


# ------------------------------------------------------------------------------------------ helpers
def _g(x, z, cx, cz, sx, sz):
    return math.exp(-(((x - cx) / sx) ** 2 + ((z - cz) / sz) ** 2))


def _stations(pts):
    """Plan polyline -> [(x, y, nx, ny)] with mitre-scaled right-hand (outward) normals."""
    n = len(pts)
    segn = []
    for i in range(n - 1):
        dx, dy = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        l = math.hypot(dx, dy) or 1e-9
        segn.append((dy / l, -dx / l))
    out = []
    for i in range(n):
        if i == 0:
            nx, ny = segn[0]
        elif i == n - 1:
            nx, ny = segn[-1]
        else:
            ax, ay = segn[i - 1]
            bx, by = segn[i]
            mx, my = ax + bx, ay + by
            l = math.hypot(mx, my) or 1e-9
            mx, my = mx / l, my / l
            k = 1.0 / max(0.3, mx * ax + my * ay)
            nx, ny = mx * k, my * k
        out.append((pts[i][0], pts[i][1], nx, ny))
    return out


def _loft(name, st, prof, cap=True):
    """Run a closed cross-section [(out, z)] along plan stations -> one solid stone / moulding."""
    verts, faces = [], []
    m = len(prof)
    for (x, y, nx, ny) in st:
        for (o, z) in prof:
            verts.append((x + nx * o, y + ny * o, z))
    for i in range(len(st) - 1):
        for k in range(m):
            k2 = (k + 1) % m
            faces.append((i * m + k, i * m + k2, (i + 1) * m + k2, (i + 1) * m + k))
    if cap:
        faces.append(tuple(range(m - 1, -1, -1)))
        e = (len(st) - 1) * m
        faces.append(tuple(range(e, e + m)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, 38)
    return o


def _subdiv(corners, step):
    pts = []
    for i in range(len(corners) - 1):
        a, b = corners[i], corners[i + 1]
        n = max(1, int(round(math.hypot(b[0] - a[0], b[1] - a[1]) / step)))
        for k in range(n):
            t = k / n
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    pts.append(corners[-1])
    return pts


def _cumlen(pts):
    L = [0.0]
    for i in range(1, len(pts)):
        L.append(L[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return L


def _course(name, st, cum, prof, n_stones, phase, rng, bev=0.006, rough=0.0025):
    """Split a lofted course into n separate stones with eased arrises (real joints)."""
    total = cum[-1]
    cuts = [0]
    for k in range(1, n_stones):
        target = total * (k + phase) / n_stones
        j = min(range(len(cum)), key=lambda i: abs(cum[i] - target))
        if j - cuts[-1] >= 2 and len(cum) - 1 - j >= 2:
            cuts.append(j)
    cuts.append(len(st) - 1)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        o = _loft(f"{name}_{a}", st[a:b + 1], prof)
        bevel(o, bev, 2)
        if rough > 0:
            displace_noise(o, strength=rough, scale=0.25, seed=int(rng.u(0, 999)))
        out.append(o)
    return out


def _dens_prof(prof, step=0.07):
    """Subdivide the long outward-facing edges of a loft cross-section so vertex masks have resolution."""
    out = []
    m = len(prof)
    for i in range(m):
        a, b = prof[i], prof[(i + 1) % m]
        out.append(a)
        if a[0] >= -0.03 and b[0] >= -0.03:
            n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / step)
            for k in range(1, n + 1):
                t = k / (n + 1)
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _refine(pts, xc, yf, half=0.17):
    """Split the plan segments around the overflow so the streak there is painted on a fine mesh."""
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if abs(mx - xc) < half and my < yf + 0.12:
            for k in (1, 2):
                out.append((a[0] + (b[0] - a[0]) * k / 3, a[1] + (b[1] - a[1]) * k / 3))
        out.append(b)
    return out


def _brow_z(ax):
    return 0.20 + 0.60 * ax - 0.75 * ax * ax


def _lion_h(X, Z):
    """Relief of the lion face (units of the mask radius): crisp primary forms - overhanging scowling brow,
    deep sockets with lens-shaped eyes, broad flat nose, two heavy muzzle pads, open mouth, chin."""
    ax = abs(X)
    rho2 = (X / 0.60) ** 2 + ((Z + 0.06) / 0.72) ** 2
    if rho2 >= 1.0:
        return 0.0
    h = 0.30 * (1.0 - rho2) ** 0.4                                            # skull dome, steep at the rim
    f = 0.0
    f += 0.05 * _g(X, Z, 0, 0.46, 0.3, 0.14)                                  # forehead
    zb = _brow_z(ax)
    d = Z - zb
    brow = math.exp(-(d / 0.036) ** 2) if d < 0 else math.exp(-(d / 0.12) ** 2)
    f += 0.16 * brow * (1.0 - smoothstep(0.42, 0.58, ax))                     # brow ridge: sharp underside
    f -= 0.055 * math.exp(-(X / 0.035) ** 2) * math.exp(-((Z - 0.33) / 0.1) ** 2)   # frown crease
    f -= 0.13 * math.exp(-((ax - 0.27) / 0.13) ** 2 - ((Z - (zb - 0.115)) / 0.065) ** 2)   # eye sockets
    e2 = ((ax - 0.265) / 0.095) ** 2 + ((Z - (_brow_z(0.265) - 0.118)) / 0.052) ** 2
    if e2 < 1.0:
        f += 0.085 * math.sqrt(1.0 - e2) - 0.03 * math.exp(-e2 / 0.05)        # eyeballs with drilled pupils
    t = clamp((0.30 - Z) / 0.44)
    w = 0.075 + 0.08 * t
    win = smoothstep(-0.22, -0.13, Z) * (1.0 - smoothstep(0.24, 0.40, Z))
    f += (0.09 + 0.09 * t) * math.exp(-(X / w) ** 4) * win                    # flat-topped nose bridge
    f += 0.07 * math.exp(-(((X / 0.15) ** 2 + ((Z + 0.14) / 0.065) ** 2)) ** 1.5)   # nose pad
    f -= 0.06 * math.exp(-((ax - 0.085) / 0.032) ** 2 - ((Z + 0.178) / 0.025) ** 2)   # nostrils
    f += 0.27 * math.exp(-(((ax - 0.15) / 0.19) ** 2 + ((Z + 0.36) / 0.14) ** 2) ** 1.4)   # muzzle pads + philtrum
    f += 0.09 * math.exp(-((ax - 0.42) / 0.14) ** 2 - ((Z + 0.02) / 0.16) ** 2)   # cheekbones
    dm = (X / 0.2) ** 2 + ((Z + 0.555) / 0.068) ** 2
    f -= 0.36 * math.exp(-dm * dm)                                            # open mouth
    f += 0.15 * math.exp(-(X / 0.2) ** 2 - ((Z + 0.70) / 0.075) ** 2)         # chin
    return max(h + f * (1.0 - smoothstep(0.86, 1.0, rho2)), 0.02)


def _lock_fn(sign):
    """One flame-shaped mane lock (unit length along +Z, ridge towards -Y) curling to `sign` x."""
    def fn(u, v):
        s = 2.0 * u - 1.0
        hw = 0.31 * (1.0 - v ** 2.2) ** 0.8 * (0.65 + 0.35 * min(1.0, v / 0.2))
        prof = float(np.interp(abs(s), [0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.84, 0.46, 0.6, 0.0]))
        hg = 0.27 * (1.0 - v ** 3) * (0.55 + 0.45 * min(1.0, v / 0.2))
        lift = 0.07 * (max(0.0, v - 0.7) / 0.3) ** 2
        return (sign * (0.44 * v * v - 0.11 * sin(pi * v)) + s * hw, 0.035 - hg * prof - lift, v)
    return fn


def _gman_h(X, Z):
    Xe = X / 0.82
    R = math.hypot(Xe, Z)
    ax = abs(X)
    h = 0.36 * math.sqrt(max(0.0, 1.0 - (R / 1.0) ** 2))
    d = Z - 0.27
    brow = math.exp(-(d / 0.04) ** 2) if d < 0 else math.exp(-(d / 0.09) ** 2)
    h += 0.10 * brow * math.exp(-(X / 0.5) ** 2) * (1.0 - 0.5 * math.exp(-(X / 0.07) ** 2))   # brow
    h -= 0.11 * _g(ax, Z, 0.22, 0.16, 0.11, 0.06)                             # eye sockets
    e2 = ((ax - 0.22) / 0.085) ** 2 + ((Z - 0.155) / 0.045) ** 2
    if e2 < 1.0:
        h += 0.07 * math.sqrt(1.0 - e2)                                       # eyes
    t = clamp((0.27 - Z) / 0.40)
    h += (0.05 + 0.15 * t) * math.exp(-(X / 0.065) ** 2) * smoothstep(-0.16, -0.10, Z) * (1.0 - smoothstep(0.24, 0.34, Z))   # nose
    h += 0.07 * _g(ax, Z, 0.085, -0.13, 0.05, 0.045)                          # nostril wings
    h += 0.12 * _g(ax, Z, 0.34, -0.10, 0.19, 0.19)                            # cheeks
    d = (X / 0.15) ** 2 + ((Z + 0.42) / 0.085) ** 2
    h += 0.09 * math.exp(-((math.sqrt(d) - 1.0) / 0.4) ** 2)                  # lips
    h -= 0.34 * math.exp(-d * d)                                              # mouth
    h += 0.10 * _g(X, Z, 0, -0.68, 0.2, 0.1)                                  # chin
    return h



# -------------------------------------------------------------------------------------------- build
def build(p, rng):
    W, H = p.width, p.height
    moss = p.moss
    sd = int(p.seed)
    lin = hex_rgb(p.stone)
    st_trim = mat("stone_smooth", p.stone)
    st_field = mat("stone_rough", tuple(c * 0.74 for c in lin))
    bronze = mat("bronze_aged", rough=1.15, wear=0.45)     # calm dark statuary bronze; the automatic edge wear is turned down
    iron = mat("iron_wrought")
    m_water = mat("water", "#20382f", rough=0.3, opacity=0.88, ripple=0.4, depth=0.3)
    m_fall = mat("water_fall", "#dcecf2", glow=0.14, opacity=0.85)
    m_moss = mat("moss")

    # ---------------------------------------------------------------- layout
    D = 0.34                    # body depth of the field (front face at y = -D)
    PF = 0.05                   # pilasters / entablature stand this far in front of the field
    B = 0.06                    # batter of the pilasters at the foot
    ring = 0.14                 # arch ring / jamb width
    zpl = 0.20                  # plinth top
    zrim = clamp(0.29 * H, 0.68, 0.84)
    zs = zrim + 0.08            # sill bottom
    zn0 = zs + 0.08             # niche floor
    if p.crown == "pediment":
        ph = clamp(0.24 * W, 0.3, 0.58)
        crown_h = ph + 0.13
    elif p.crown == "segmental":
        ph = clamp(0.19 * W, 0.26, 0.46)
        crown_h = ph + 0.13
    else:
        ph = 0.38
        crown_h = ph + 0.15
    zb = H - crown_h            # top of the cornice
    hc = 0.14
    zf1 = zb - hc               # top of the frieze / bottom of the cornice
    zf0 = zf1 - 0.24            # bottom of the entablature
    avail = zf0 - 0.05 - ring - zn0
    r = min(0.22 * W, avail * 0.62)
    pw = min(W / 2 - r - ring, max(0.2, 0.16 * W))
    left = W / 2 - r - ring - pw
    if left < 0.13:
        pw += left
    xi = W / 2 - pw             # inner edge of the pilasters
    Re = r + ring
    zc = zf0 - 0.05 - Re        # spring line of the arch
    dv = 0.14                   # depth of the arch ring
    yface = -(D + 0.015)        # front of the ring
    yn0 = yface + dv            # where the niche shell starts
    dn = min(0.2, -yn0 - 0.03)  # niche depth behind the ring
    zm = min(zn0 + 0.5 * (zc + 0.8 * r - zn0), zrim + 0.72)
    Rm = min(0.2, r * 0.56, (zm - zn0 - 0.015) / 1.3)
    rback = Rm * {"lion": 1.2, "green_man": 0.9, "rosette": 0.84}[p.mask]      # the mask's back rim touches the apse wall here
    ym = yn0 + dn * math.sqrt(max(0.05, 1.0 - (rback / r) ** 2)) - 0.003
    Rb = clamp(0.36 * W, 0.44, 0.76)
    tw = 0.115                  # basin wall thickness
    zw = zrim - 0.05            # water level
    ypf = -(D + PF)

    stones = []                 # dressed stone (trim)
    fields = []                 # rougher field / plinth stone

    def blk(name, x0, x1, y0, y1, z0, z1, r_, chamfer=0.01, rough=0.003, detail=0.09):
        o = stone_block(name, (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), loc=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                        chamfer=chamfer, rough=rough, rng=r_, detail=detail)
        return o

    # ---------------------------------------------------------------- plinth
    rp = rng.fork("plinth")
    xo, yo = W / 2 + B, D + PF + 0.7 * B
    xs = [-xo - 0.06, rp.jitter(-xo * 0.3, 0.1), rp.jitter(xo * 0.35, 0.1), xo + 0.06]
    for i in range(3):
        fields.append(blk("plinth", xs[i], xs[i + 1], -(yo + 0.06), 0.0, 0.0, zpl, rp, chamfer=0.014, rough=0.006, detail=0.12))
    pst = _stations([(-xo, 0.0), (-xo, -yo), (rp.jitter(0.0, 0.2), -yo), (xo, -yo), (xo, 0.0)])
    pprof = [(-0.04, zpl - 0.005), (0.05, zpl - 0.005), (0.05, zpl + 0.022), (0.036, zpl + 0.034), (0.0, zpl + 0.07), (-0.04, zpl + 0.07)]
    for a, b in ((0, 2), (2, 4)):
        o = _loft("plinth_mould", pst[a:b + 1], pprof)
        bevel(o, 0.005, 2)
        stones.append(o)

    # ---------------------------------------------------------------- pilasters (battered, stacked blocks)
    for side in (-1, 1):
        rpp = rng.fork(f"pil{side}")
        n = max(3, int(round((zf0 - zpl) / 0.4)))
        hs = [rpp.vary(1.0, 0.18) for _ in range(n)]
        tot = sum(hs)
        z = zpl
        for k in range(n):
            h = (zf0 - zpl) * hs[k] / tot
            o = blk("pilaster", side * xi, side * W / 2, ypf, 0.0, z, z + h, rpp, chamfer=0.011, rough=0.0035)
            cz = z + h / 2
            hw, hd = pw / 2, (D + PF) / 2

            def batter(co, side=side, cz=cz, hw=hw, hd=hd):
                bz = B * (1.0 - clamp((co.z + cz - zpl) / (zf0 - zpl)))
                tx = clamp((side * co.x / hw + 1.0) / 2)
                ty = clamp((-co.y / hd + 1.0) / 2)
                return Vector((co.x + side * tx * bz, co.y - ty * bz * 0.7, co.z))

            transform_verts(o, batter)
            stones.append(o)
            z += h

    # pilaster capitals (astragal + cavetto under the architrave) and torus bases
    for side in (-1, 1):
        if side < 0:
            cst = _stations([(-W / 2, 0.0), (-W / 2, ypf), (-xi, ypf), (-xi, -D)])
            bst_ = _stations([(-W / 2 - B, 0.0), (-W / 2 - B, ypf - 0.7 * B), (-xi, ypf - 0.7 * B), (-xi, -D)])
        else:
            cst = _stations([(xi, -D), (xi, ypf), (W / 2, ypf), (W / 2, 0.0)])
            bst_ = _stations([(xi, -D), (xi, ypf - 0.7 * B), (W / 2 + B, ypf - 0.7 * B), (W / 2 + B, 0.0)])
        z0c = zf0 - 0.115
        o = _loft("pil_capital", cst, [(-0.03, z0c), (0.004, z0c), (0.014, z0c + 0.009), (0.014, z0c + 0.019), (0.004, z0c + 0.028),
                                       (0.004, z0c + 0.055), (0.012, z0c + 0.07), (0.03, z0c + 0.086), (0.036, z0c + 0.092),
                                       (0.036, z0c + 0.116), (-0.03, z0c + 0.116)])
        bevel(o, 0.003, 2)
        stones.append(o)
        z0b_ = zpl + 0.065
        o = _loft("pil_base", bst_, [(-0.04, z0b_), (0.02, z0b_), (0.03, z0b_ + 0.012), (0.03, z0b_ + 0.03), (0.02, z0b_ + 0.042),
                                     (0.008, z0b_ + 0.048), (0.008, z0b_ + 0.062), (-0.004, z0b_ + 0.075), (-0.04, z0b_ + 0.075)])
        bevel(o, 0.003, 2)
        stones.append(o)

    # ---------------------------------------------------------------- dado field behind the basin
    rf = rng.fork("field")
    ncd = max(2, int(round((zs - zpl) / 0.3)))
    for k in range(ncd):
        za, zt = zpl + (zs - zpl) * k / ncd, zpl + (zs - zpl) * (k + 1) / ncd
        xj = rf.u(-0.25, 0.25) * xi
        dd = 0.1 if k == ncd - 1 else 0.25          # the top course is the wet back wall of the basin: finer for the algae masks
        fields.append(blk("dado", -xi, xj, -D, 0.0, za, zt, rf, detail=dd))
        fields.append(blk("dado", xj, xi, -D, 0.0, za, zt, rf, detail=dd))

    # ---------------------------------------------------------------- sill, jambs, arch ring, keystone
    ra = rng.fork("arch")
    sx = min(Re + 0.05, xi)
    stones.append(blk("sill", -sx, sx, -(D + 0.075), yn0 + dn + 0.01, zs, zn0, ra, chamfer=0.016, rough=0.002, detail=0.055))
    nj = max(1, int(round((zc - 0.07 - zn0) / 0.3)))
    for side in (-1, 1):
        for k in range(nj):
            za = zn0 + (zc - 0.07 - zn0) * k / nj
            zt = zn0 + (zc - 0.07 - zn0) * (k + 1) / nj
            stones.append(blk("jamb", side * r, side * Re, yface - ra.u(0, 0.004), yn0, za, zt, ra, chamfer=0.009, rough=0.002))
        stones.append(blk("impost", side * (r - 0.018), side * Re, yface - 0.03, yn0, zc - 0.07, zc, ra, chamfer=0.014, rough=0.002))
    nvs = 9 if r > 0.3 else 7
    for i in range(nvs):
        if i == nvs // 2:
            continue
        a0, a1 = pi * i / nvs, pi * (i + 1) / nvs
        pts = [(Re * cos(a0 + (a1 - a0) * t / 3), Re * sin(a0 + (a1 - a0) * t / 3)) for t in range(4)]
        pts += [(r * cos(a1 + (a0 - a1) * t / 3), r * sin(a1 + (a0 - a1) * t / 3)) for t in range(4)]
        proud = ra.u(0.0, 0.005)
        o = extrude_profile("voussoir", pts, dv + proud, loc=(0, yn0, zc), rot=(90, 0, 0), bevel=0.008)
        stones.append(o)
    ka = radians(90 - 0.5 * 180 / nvs)
    kb = radians(90 + 0.5 * 180 / nvs)
    r_in, r_out = r - 0.018, Re + 0.05
    kpts = [(r_out / sin(kb) * cos(ka) * 1.08, r_out), (r_out / sin(kb) * cos(kb) * 1.08, r_out),
            (r_in * cos(kb), r_in * sin(kb)), (0.0, r_in), (r_in * cos(ka), r_in * sin(ka))]
    key = extrude_profile("keystone", kpts, dv + 0.045, loc=(0, yn0, zc), rot=(90, 0, 0), bevel=0.012)
    stones.append(key)
    # a small raised diamond boss on the keystone
    boss = cone("key_boss", 0.042, 0.0, 0.02, loc=(0, yface - 0.045 - 0.008, zc + (r_in + r_out) / 2 + 0.01), rot=(90, 0, 0), verts=4)
    stones.append(boss)

    # ---------------------------------------------------------------- spandrel / field courses around the arch
    nco = max(1, int(round((zf0 - zn0) / 0.29)))

    def x_in(z):
        if z <= zc:
            return Re
        return math.sqrt(max(Re * Re - (z - zc) ** 2, 0.0))

    for k in range(nco):
        za, zt = zn0 + (zf0 - zn0) * k / nco, zn0 + (zf0 - zn0) * (k + 1) / nco
        for side in (-1, 1):
            xa, xt = x_in(za), x_in(zt)
            if xi - min(xa, xt) < 0.012:
                continue
            xj = None
            if xi - Re > 0.34 and (k + (side > 0)) % 2 == 0:
                xj = Re + rf.u(0.14, xi - Re - 0.14)
                fields.append(blk("field", side * xj, side * xi, -D, 0.0, za, zt, rf))
            xe = xj if xj is not None else xi
            pts = [(xe, za), (xe, zt)]
            for q in range(7):
                zq = zt + (za - zt) * q / 6
                pts.append((x_in(zq), zq))
            clean = []
            for pt in pts:
                if not clean or math.hypot(pt[0] - clean[-1][0], pt[1] - clean[-1][1]) > 2e-3:
                    clean.append(pt)
            if math.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) < 2e-3:
                clean.pop()
            if len(clean) < 3:
                continue
            if side < 0:
                clean = [(-x, z) for x, z in reversed(clean)]
            o = extrude_profile("spandrel", clean, D, loc=(0, 0, 0), rot=(90, 0, 0), bevel=0.008)
            densify(o, 0.11)
            displace_noise(o, strength=0.003, scale=0.3, seed=sd + k * 7 + side)
            o["tsq_smooth"] = 45.0
            fields.append(o)

    # ---------------------------------------------------------------- niche: elliptical apse with a scallop conch
    nl = 9
    niche_lo = surface("niche_wall", lambda u, v: (r * cos(pi * u), yn0 + dn * sin(pi * u), zn0 - 0.01 + (zc - zn0 + 0.01) * v),
                       30, max(3, int((zc - zn0) / 0.07)))

    def conch_fn(u, v):
        a = pi * u
        s = sin(v * pi / 2)
        c = math.sqrt(max(0.0, 1.0 - s * s))
        fl = 0.018 * (s ** 1.1) * (1.0 - s ** 8) * (0.5 - 0.5 * cos(2 * nl * a)) ** 0.7
        return (r * s * cos(a), yn0 + dn * c - fl, zc + r * s * sin(a))

    conch = surface("niche_conch", conch_fn, nl * 8, 12)
    weld(conch, 1e-4)
    for o in (niche_lo, conch):
        recalc_normals(o, inside=True)
        smooth(o, 50)
        stones.append(o)
    band = sweep("niche_band", [(r * cos(pi * i / 20), yn0 + dn * sin(pi * i / 20) - 0.004, zc) for i in range(21)], 0.017, sides=8)
    stones.append(band)

    # ---------------------------------------------------------------- entablature: architrave bead, frieze tablet, cornice
    re_ = rng.fork("entab")
    est = _stations([(-W / 2, 0.0), (-W / 2, ypf), (-xi * re_.u(0.5, 0.9), ypf), (xi * re_.u(0.5, 0.9), ypf), (W / 2, ypf), (W / 2, 0.0)])
    aprof = [(-0.02, zf0), (0.018, zf0), (0.03, zf0 + 0.012), (0.03, zf0 + 0.034), (0.014, zf0 + 0.05), (-0.02, zf0 + 0.05)]
    for a, b in ((0, 3), (3, 5)):
        o = _loft("architrave", est[a:b + 1], aprof)
        bevel(o, 0.004, 2)
        stones.append(o)
    for side in (-1, 1):
        stones.append(blk("frieze_end", side * xi, side * W / 2, ypf, 0.0, zf0, zb, re_, rough=0.002))
    # the tablet is a slab with a gridded face: the inscription boolean then only re-meshes the cells it touches,
    # so no long thin triangles smear the baked weathering across the lettering
    tab = box("frieze_tablet", (2 * xi, 0.09, zb - zf0), loc=(0, ypf + 0.045, (zf0 + zb) / 2))
    densify(tab, 0.05)
    body = (p.inscription or "").strip()
    if body:
        zt0, zt1 = zf0 + 0.05, zf1
        size = min(0.125, (zt1 - zt0) * 0.62)
        txt = text("inscr", body, size=size, depth=0.013, loc=(0, ypf + 0.007, zt0), font="roman")
        mn, mx = world_bbox([txt])
        wtxt = mx.x - mn.x
        fit = 2 * xi - 0.14
        if wtxt > fit:
            delete([txt])
            size *= fit / wtxt
            txt = text("inscr", body, size=size, depth=0.013, loc=(0, ypf + 0.007, zt0), font="roman")
            mn, mx = world_bbox([txt])
        dz = (zt0 + zt1) / 2 - (mn.z + mx.z) / 2
        dx = -(mn.x + mx.x) / 2
        set_loc_rot(txt, loc=(dx, ypf + 0.007, zt0 + dz))
        try:
            boolean(tab, txt, "DIFFERENCE")
        except Exception:                # keep going with a plain tablet
            pass
    pc = 0.11
    cprof = [(-0.03, zf1), (0.012, zf1), (0.012, zf1 + 0.02), (0.022, zf1 + 0.03), (0.04, zf1 + 0.04), (0.052, zf1 + 0.062),
             (0.056, zf1 + 0.07), (0.084, zf1 + 0.074), (0.086, zf1 + 0.104), (0.098, zf1 + 0.11), (pc, zf1 + 0.124),
             (pc, zb - 0.006), (0.0, zb + 0.004), (-0.03, zb + 0.004)]
    for a, b in ((0, 2), (2, 3), (3, 5)):
        o = _loft("cornice", est[a:b + 1], cprof)
        bevel(o, 0.004, 2)
        displace_noise(o, strength=0.002, scale=0.3, seed=sd + a)
        stones.append(o)

    # ---------------------------------------------------------------- crown
    rc = rng.fork("crown")
    xw = W / 2 + pc - 0.01
    ztop = zb + ph
    if p.crown in ("pediment", "segmental"):
        if p.crown == "pediment":
            outer = [(-xw, 0.0), (xw, 0.0), (0.0, ph)]

            def inset(d):
                # triangle offset inwards by d
                sl = math.hypot(xw, ph)
                kx = d * sl / ph + d * xw / ph      # shift of the base corners along x
                top = ph - d * sl / xw
                return [(-xw + kx, d), (xw - kx, d), (0.0, top)]
        else:
            R_ = (xw * xw + ph * ph) / (2 * ph)

            def seg(d, n=22):
                Rr = R_ - d
                zc_ = ph - R_
                x0 = math.sqrt(max(Rr * Rr - (d - zc_) ** 2, 1e-6))
                a0 = math.atan2(d - zc_, x0)
                a1 = pi - a0
                return [(Rr * cos(a0 + (a1 - a0) * i / n), zc_ + Rr * sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]

            outer = seg(0.0)

            def inset(d):
                return seg(d)
        tym = extrude_profile("tympanum", inset(0.03), D + PF, loc=(0, 0, zb), rot=(90, 0, 0), bevel=0.004)
        fields.append(tym)
        fr1 = shape_extrude("rake_cornice", outer, [inset(0.085)], depth=D + PF + 0.075, loc=(0, 0, zb), rot=(90, 0, 0), bevel=0.005)
        fr2 = shape_extrude("rake_fillet", outer, [inset(0.034)], depth=D + PF + 0.1, loc=(0, 0, zb), rot=(90, 0, 0), bevel=0.004)
        stones += [fr1, fr2]
        # carved patera in the tympanum
        rr = min(0.11, ph * 0.26)
        pat = lathe("patera", [(0.0, 0.0), (rr, 0.0), (rr * 1.04, 0.008), (rr * 0.86, 0.016), (rr * 0.7, 0.011), (rr * 0.42, 0.024),
                               (rr * 0.2, 0.02), (0.0, 0.032)], segments=32, loc=(0, ypf, zb + 0.085 + (ph - 0.17) * 0.36), rot=(90, 0, 0))

        def petals(co, rr=rr):
            rad = math.hypot(co.x, co.y)
            if rad < rr * 0.25 or rad > rr * 0.9:
                return co
            a = math.atan2(co.y, co.x)
            return Vector((co.x, co.y, co.z * (0.55 + 0.75 * abs(cos(4 * a)))))

        transform_verts(pat, petals)
        stones.append(pat)
        # ball finial on a saddle block at the apex
        stones.append(blk("apex_block", -0.085, 0.085, -(D + PF + 0.1) * 0.5 - 0.085, -(D + PF + 0.1) * 0.5 + 0.085, ztop - 0.07, ztop + 0.03, rc,
                          chamfer=0.01, rough=0.002))
        stones.append(lathe("finial", [(0.055, 0.0), (0.06, 0.012), (0.035, 0.024), (0.03, 0.034), (0.052, 0.05), (0.066, 0.072),
                                       (0.06, 0.098), (0.036, 0.116), (0.0, 0.123)], segments=20, loc=(0, -(D + PF + 0.1) * 0.5, ztop + 0.028)))
    else:
        h1, h2 = 0.17, 0.15
        w1, w2 = W * 0.9, W * 0.5
        xj = rc.jitter(0.0, 0.1) * w1
        stones.append(blk("attic", -w1 / 2, xj, -(D + PF + 0.02), 0.0, zb, zb + h1, rc, chamfer=0.014))
        stones.append(blk("attic", xj, w1 / 2, -(D + PF + 0.02), 0.0, zb, zb + h1, rc, chamfer=0.014))
        stones.append(blk("attic_top", -w2 / 2, w2 / 2, -(D + PF - 0.03), 0.0, zb + h1, zb + h1 + h2, rc, chamfer=0.012))
        cap = _loft("attic_cap", _stations([(-w2 / 2, 0.0), (-w2 / 2, -(D + PF - 0.03)), (w2 / 2, -(D + PF - 0.03)), (w2 / 2, 0.0)]),
                    [(-0.16, zb + h1 + h2), (0.012, zb + h1 + h2), (0.035, zb + h1 + h2 + 0.022), (0.035, zb + h1 + h2 + 0.045),
                     (0.0, zb + h1 + h2 + 0.06), (-0.16, zb + h1 + h2 + 0.06)])
        bevel(cap, 0.004, 2)
        stones.append(cap)
        # carved scallop on the attic face
        rr = min(0.105, w2 * 0.2)
        nsh = 7
        sh = surface("attic_shell", lambda u, v: (rr * v * cos(pi * u) * 1.25, -0.03 * math.sqrt(max(0.0, 1.0 - v * v)) - 0.008 * v * abs(sin(nsh * pi * u)) * (1 - v ** 6),
                                                  rr * v * sin(pi * u)), nsh * 6, 7)
        weld(sh, 1e-4)
        recalc_normals(sh)
        place(sh, loc=(0, -(D + PF - 0.03) + 0.004, zb + h1 + 0.02))
        smooth(sh, 50)
        stones.append(sh)
        # pylon caps over the pilasters, as on the reference monument
        for side in (-1, 1):
            xc_ = side * (W / 2 - pw / 2)
            hp = ph * 0.62
            stones.append(blk("pylon", xc_ - pw / 2 - 0.01, xc_ + pw / 2 + 0.01, -(D + PF + 0.03), 0.0, zb, zb + hp, rc, chamfer=0.014))
            pyr = cone("pylon_cap", 0.7071, 0.0, 0.11, loc=(xc_, -(D + PF + 0.03) / 2, zb + hp + 0.054), rot=(0, 0, 45), verts=4)
            set_loc_rot(pyr, scale=(pw + 0.06, (D + PF + 0.07), 1.0))
            bevel(pyr, 0.006, 2)
            stones.append(pyr)
        stones.append(lathe("finial", [(0.07, 0.0), (0.075, 0.014), (0.04, 0.028), (0.034, 0.04), (0.06, 0.058), (0.078, 0.085),
                                       (0.07, 0.118), (0.04, 0.14), (0.0, 0.148)], segments=20, loc=(0, -(D + PF - 0.03) / 2, zb + h1 + h2 + 0.058)))

    # ---------------------------------------------------------------- basin
    rb_ = rng.fork("basin")
    y0 = -D
    basin_objs = []
    bst = None
    zb1 = 0.17
    if p.basin == "bowl":
        yfront = y0 - Rb
        z0b = zrim - 0.25
        prof_o = [(0.105, z0b), (0.122, z0b + 0.008), (0.126, z0b + 0.03), (0.16, z0b + 0.045), (Rb * 0.5, z0b + 0.075), (Rb * 0.8, z0b + 0.125),
                  (Rb * 0.94, z0b + 0.175), (Rb * 0.965, z0b + 0.195), (Rb * 0.93, z0b + 0.203), (Rb * 0.935, z0b + 0.214), (Rb, z0b + 0.224),
                  (Rb + 0.016, z0b + 0.236), (Rb + 0.012, zrim - 0.006), (Rb, zrim), (Rb - 0.045, zrim), (Rb - 0.065, zrim - 0.02),
                  (Rb - 0.085, zrim - 0.06), (Rb * 0.6, zrim - 0.15), (0.0, zrim - 0.18)]
        ng = 9
        bowl = lathe("bowl", prof_o, segments=ng * 6, angle=180.0, loc=(0, y0, 0), rot=(0, 0, 180), smooth_angle=42)

        def gadroon(co):
            rad = math.hypot(co.x, co.y)
            if rad < 0.15 or co.z > z0b + 0.196 or co.z < z0b + 0.04:
                return co
            a = math.atan2(co.y, co.x)
            wz = smoothstep(z0b + 0.045, z0b + 0.08, co.z) * (1.0 - smoothstep(z0b + 0.16, z0b + 0.196, co.z))
            k = 1.0 + 0.05 * wz * (abs(sin(ng * a)) - 0.5)
            return Vector((co.x * k, co.y * k, co.z))

        transform_verts(bowl, gadroon)
        basin_objs.append(bowl)
        # engaged half-baluster under the bowl: square plinth, torus + scotia base, vase belly, slender neck with an
        # astragal and a spreading cap whose abacus is the visible seat of the bowl's foot ring
        zp = 0.10
        basin_objs.append(blk("ped_plinth", -0.235, 0.235, y0 - 0.25, y0, 0.0, zp, rb_, chamfer=0.012, rough=0.003))
        top = z0b + 0.004
        hbel = top - zp - 0.085 - 0.07
        ped = lathe("bowl_pedestal", [(0.185, zp - 0.002), (0.197, zp + 0.012), (0.197, zp + 0.028), (0.18, zp + 0.04), (0.145, zp + 0.047),
                                      (0.13, zp + 0.06), (0.14, zp + 0.07), (0.155, zp + 0.085), (0.168, zp + 0.085 + 0.14 * hbel),
                                      (0.155, zp + 0.085 + 0.33 * hbel), (0.115, zp + 0.085 + 0.58 * hbel), (0.082, zp + 0.085 + 0.8 * hbel),
                                      (0.07, zp + 0.085 + 0.95 * hbel), (0.07, top - 0.068), (0.088, top - 0.062), (0.09, top - 0.054),
                                      (0.076, top - 0.048), (0.092, top - 0.036), (0.13, top - 0.022), (0.152, top - 0.018),
                                      (0.152, top - 0.004), (0.144, top), (0.0, top)],
                    segments=40, angle=180.0, loc=(0, y0, 0), rot=(0, 0, 180), smooth_angle=42)
        basin_objs.append(ped)
        ri = Rb - 0.07
        inner_w = [(ri * cos(pi + pi * i / 22), y0 + 0.005 + ri * sin(pi + pi * i / 22)) for i in range(23)]
        x_of, y_of = 0.0, yfront - 0.01
        z_of = zrim - 0.075
    else:
        if p.basin == "round":
            cy = y0 - 0.09
            corners = None
            yfront = cy - Rb
            nst = (5, 4, 5, 6)
        elif p.basin == "canted":
            a_, f_, c_ = Rb * 0.5, Rb * 1.08, Rb * 0.5
            corners = [(-Rb, y0), (-Rb, y0 - a_), (-c_, y0 - f_), (c_, y0 - f_), (Rb, y0 - a_), (Rb, y0)]
            yfront = y0 - f_
            nst = (5, 4, 5, 5)
        else:
            hw_, d_ = min(Rb * 1.18, W / 2 + 0.02), Rb * 0.95
            corners = [(-hw_, y0), (-hw_, y0 - d_), (hw_, y0 - d_), (hw_, y0)]
            yfront = y0 - d_
            nst = (4, 5, 4, 5)

        def plan(step):
            if corners is None:
                n = max(10, int(round(pi * Rb / step)))
                return [(-Rb, y0)] + [(Rb * cos(pi + pi * i / n), cy + Rb * sin(pi + pi * i / n)) for i in range(n + 1)] + [(Rb, y0)]
            return _subdiv(corners, step)

        pts = _refine(plan(0.07 if corners is None else 0.1), 0.0, yfront)
        bst = _stations(pts)
        cum = _cumlen(pts)
        zc0 = zrim - 0.085
        zmid = zb1 + 0.5 * (zc0 - zb1)
        # four courses: moulded base, lower drum with an astragal band, set-back upper drum, moulded coping
        base_prof = [(-tw, 0.0), (0.055, 0.0), (0.055, zb1 - 0.065), (0.04, zb1 - 0.05), (0.03, zb1 - 0.02), (0.0, zb1), (-tw, zb1)]
        low_prof = [(-tw, zb1), (0.0, zb1), (0.003, zmid - 0.052), (0.004, zmid - 0.044), (0.015, zmid - 0.038), (0.02, zmid - 0.024),
                    (0.015, zmid - 0.01), (0.004, zmid - 0.004), (-0.005, zmid), (-tw, zmid)]
        up_prof = [(-tw, zmid), (-0.005, zmid), (-0.005, zc0), (-tw, zc0)]
        cope_prof = [(-tw - 0.02, zc0), (-0.005, zc0), (0.008, zc0 + 0.004), (0.012, zc0 + 0.014), (0.03, zc0 + 0.022), (0.045, zc0 + 0.036),
                     (0.047, zrim - 0.022), (0.03, zrim - 0.004), (0.0, zrim), (-tw + 0.01, zrim), (-tw - 0.02, zrim - 0.018)]
        basin_objs += _course("basin_base", bst, cum, _dens_prof(base_prof), nst[0], 0.0, rb_, bev=0.007)
        basin_objs += _course("basin_low", bst, cum, _dens_prof(low_prof), nst[1], rb_.u(-0.15, 0.15), rb_, bev=0.006)
        basin_objs += _course("basin_up", bst, cum, _dens_prof(up_prof), nst[2], rb_.u(-0.12, 0.12), rb_, bev=0.006)
        basin_objs += _course("basin_cope", bst, cum, cope_prof, nst[3], rb_.u(-0.1, 0.1), rb_, bev=0.006)
        inner_w = [(x + nx * (-tw + 0.012), y + ny * (-tw + 0.012)) for (x, y, nx, ny) in _stations(plan(0.13))]
        inner_w[0] = (inner_w[0][0], y0 + 0.005)
        inner_w[-1] = (inner_w[-1][0], y0 + 0.005)
        x_of, y_of = 0.0, yfront
        z_of = zc0 - 0.05
        # iron cramps leaded across the coping joints
        cr = box("cramp", (0.11, 0.022, 0.008), bevel=0.002)
        mats_c = []
        ncr = nst[3]
        for k in range(1, ncr):
            target = cum[-1] * k / ncr
            j = min(range(len(cum)), key=lambda i: abs(cum[i] - target))
            x, y, nx, ny = bst[j]
            ang = degrees(math.atan2(ny, nx)) + 90
            mats_c.append(trs((x - nx * 0.05, y - ny * 0.05, zrim + 0.002), (0, 0, ang + rb_.jitter(0, 6))))
        if mats_c:
            cro = instances_to_mesh("cramps", cr, mats_c, keep_base=False)
            assign(cro, iron)
            paint(cro, special=0.3, grime=0.4)
        else:
            delete([cr])
    cxb = sum(q[0] for q in inner_w) / len(inner_w)
    cyb = sum(q[1] for q in inner_w) / len(inner_w)

    def fan(name, outline, z):
        n = len(outline)
        verts = [(cxb, cyb, z)]
        for f in (0.5, 1.0):
            verts += [(cxb + (x - cxb) * f, cyb + (y - cyb) * f, z) for x, y in outline]
        faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
        faces += [(1 + i, 1 + n + i, 1 + n + (i + 1) % n, 1 + (i + 1) % n) for i in range(n)]
        o = mesh_object(name, verts, faces)
        for poly in o.data.polygons:
            if poly.normal.z < 0:
                poly.flip()
        uv_planar(o, (1, 0, 0), (0, 1, 0))
        return o

    if p.basin != "bowl":
        fields.append(fan("basin_floor", inner_w, zw - 0.3))

    # overflow pipe, drain kerb stone with a flush iron grate
    zdr = 0.016
    ydr = y_of - 0.17
    brz = []
    if p.basin != "bowl":
        ofp = lathe("overflow", [(0.03, 0.0), (0.032, 0.006), (0.019, 0.01), (0.017, 0.07), (0.021, 0.074), (0.021, 0.084), (0.011, 0.084),
                                 (0.011, 0.03), (0.0, 0.03)], segments=14, loc=(x_of, y_of - 0.002, z_of), rot=(97, 0, 0))
        brz.append(ofp)
        ytip = y_of - 0.085
    else:
        ytip = y_of
    dslab = shape_extrude("drain_slab", rounded_rect(0.44, 0.31, 0.02, 3), [[(x, y - 0.01) for x, y in rounded_rect(0.262, 0.152, 0.008, 2)]],
                          depth=zdr, loc=(x_of, ydr, 0.0), bevel=0.004)
    fields.append(dslab)
    slot = rounded_rect(0.018, 0.105, 0.007, 3)
    holes = [[(x + k * 0.033, y) for x, y in slot] for k in range(-3, 4)]
    grate = shape_extrude("grate", rounded_rect(0.256, 0.146, 0.008, 2), holes, depth=0.008, loc=(x_of, ydr - 0.01, zdr - 0.009))
    assign(grate, iron)
    paint(grate, special=0.4, grime=0.6)
    pit = box("grate_pit", (0.25, 0.14, 0.002), loc=(x_of, ydr - 0.01, 0.003))
    assign(pit, st_field)
    paint(pit, grime=1.0)

    # ---------------------------------------------------------------- bronze mask + spout
    rm_ = rng.fork("mask")
    hsc = 0.92

    def face_front(o):
        if sum(pl.normal.y * pl.area for pl in o.data.polygons) > 0:
            for pl in o.data.polygons:
                pl.flip()
            o.data.update()

    if p.mask == "lion":
        zmouth = zm - 0.555 * Rm * 1.05

        def mfn(u, v):
            a = 2 * pi * u
            X, Z = 0.60 * v * cos(a), -0.06 + 0.72 * v * sin(a)
            return (Rm * X, ym - 0.012 - Rm * hsc * _lion_h(X, Z), zm + Rm * Z * 1.05)

        mask = surface("mask", mfn, 88, 36, closed_u=True)
        ydeep = 0.012
        ytipm = 0.62 * Rm * hsc + 0.075
        # round cast plaque behind the head
        Rpl = Rm * 1.2
        zpq = zm - 0.08 * Rm
        nrp = int(Rpl / 0.016)
        brz.append(lathe("mask_plaque", [(Rpl + 0.006, 0.0), (Rpl + 0.012, 0.007), (Rpl + 0.007, 0.017), (Rpl - 0.008, 0.0135)]
                         + [((Rpl - 0.02) * (nrp - i) / nrp, 0.012) for i in range(nrp + 1)], segments=48, loc=(0, ym, zpq), rot=(90, 0, 0)))
        # mane: two overlapping rings of flame-shaped locks, each a ridged tongue 2-3 cm deep, hanging to either side
        lock = {}
        for sg in (1, -1):
            lock[sg] = surface("lock", _lock_fn(sg), 8, 8)
            weld(lock[sg], 1e-5)
            face_front(lock[sg])
        lm = {1: [], -1: []}
        for ringk, (cnt, f_, ln, tilt, yoff, ph0) in enumerate(((13, 0.88, 0.56, 3.0, 0.006, 0.0), (11, 0.76, 0.50, -10.0, 0.11, 16.0))):
            for k in range(cnt):
                ad = (ph0 + 360.0 * (k + 0.5) / cnt + rm_.jitter(0, 4)) % 360
                if ringk == 1 and 240 < ad < 300:
                    continue                      # the chin and the spout stay free
                ar = radians(ad)
                sg = 1 if cos(ar) >= 0 else -1
                lean = -26.0 * cos(ar) + rm_.jitter(0, 6)        # the locks hang: side locks sweep downwards
                L = Rm * ln * rm_.vary(1.0, 0.1)
                lm[sg].append(trs((Rm * 0.60 * f_ * cos(ar), ym - 0.012 - Rm * yoff, zm + Rm * 1.05 * (-0.06 + 0.72 * f_ * sin(ar))),
                                  (tilt + rm_.jitter(0, 3), 90 - (ad + lean), 0), L))
        for sg in (1, -1):
            if lm[sg]:
                mane = instances_to_mesh("mask_mane", lock[sg], lm[sg], keep_base=False, uv_jitter=False)
                smooth(mane, 50)
                brz.append(mane)
            else:
                delete([lock[sg]])
    elif p.mask == "green_man":
        fr = 0.74
        zmouth = zm - 0.42 * Rm * fr * 1.12

        def mfn(u, v):
            a = 2 * pi * u
            X, Z = v * cos(a), v * sin(a)
            h = _gman_h(X * 0.82, Z) * (1.0 - smoothstep(0.9, 1.0, v)) ** 0.5
            return (Rm * fr * X * 0.82, ym - Rm * hsc * h, zm + Rm * fr * Z * 1.12)

        mask = surface("mask", mfn, 64, 28, closed_u=True)
        ydeep = 0.0
        ytipm = 0.6 * Rm * hsc + 0.05
        # foliage ruff: two rings of lobed leaves sprouting from behind the face
        def leaf_fn(u, v):
            wv = 0.3 * (sin(pi * v ** 0.8) ** 0.7) * (1.0 + 0.3 * sin(v * pi * 5.0))
            x = (u - 0.5) * 2 * wv
            crease = -0.10 * (1.0 - abs(2 * u - 1)) * sin(pi * v)
            curl = -0.22 * v * v + 0.05 * sin(v * pi * 2.2)
            return (x, crease + curl, v)

        leaf = surface("leaf", leaf_fn, 5, 7, thickness=0.035)
        realize(leaf)
        lm = []
        for ringk, (cnt, rad, ln, tilt, ph0) in enumerate(((11, 0.50, 0.62, 12, 0.0), (9, 0.36, 0.5, 30, 20.0))):
            for k in range(cnt):
                ad = ph0 + 360.0 * k / cnt + rm_.jitter(0, 5)
                if 248 < ad % 360 < 292:
                    continue                      # leave the chin free for the spout
                ar = radians(ad)
                s_ = Rm * ln * rm_.vary(1.0, 0.1)
                lm.append(trs((Rm * rad * cos(ar), ym - 0.012 - ringk * 0.012, zm + Rm * rad * sin(ar) * 1.1), (tilt, 90 - ad, rm_.jitter(0, 8)), s_))
        leaves = instances_to_mesh("mask_leaves", leaf, lm, keep_base=False)
        smooth(leaves, 60)
        brz.append(leaves)
    else:
        zmouth = zm
        rr = Rm * 0.82
        mask = lathe("mask", [(0.0, 0.0), (rr, 0.0), (rr * 1.03, 0.012), (rr * 0.92, 0.024), (rr * 0.8, 0.02), (rr * 0.6, 0.045), (rr * 0.34, 0.03),
                              (rr * 0.3, 0.05), (rr * 0.2, 0.06), (0.0, 0.06)], segments=48, loc=(0, ym, zm), rot=(90, 0, 0))

        def petal(co, rr=rr):
            rad = math.hypot(co.x, co.y)
            if rad < rr * 0.36 or rad > rr * 0.9:
                return co
            a = math.atan2(co.y, co.x)
            return Vector((co.x, co.y, co.z * (0.45 + 0.9 * abs(cos(4 * a)) ** 0.7)))

        transform_verts(mask, petal)
        ydeep = -0.03
        ytipm = 0.06 + 0.11
    if p.mask != "rosette":
        weld(mask, 1e-5)
        recalc_normals(mask)
        face_front(mask)
        smooth(mask, 55)
    brz.append(mask)
    # spout pipe (turned nozzle), pitched 10 deg down
    Lp = ytipm - ydeep
    pitch = 10.0
    pipe = lathe("spout", [(0.012, 0.0), (0.0185, 0.0), (0.0185, Lp - 0.04), (0.022, Lp - 0.036), (0.022, Lp - 0.026), (0.0185, Lp - 0.022),
                           (0.0185, Lp - 0.004), (0.016, Lp), (0.0125, Lp), (0.0125, Lp - 0.035), (0.0, Lp - 0.035)], segments=16,
                 loc=(0, ym - ydeep, zmouth), rot=(90 + pitch, 0, 0))
    brz.append(pipe)
    ytp = ym - ydeep - Lp * cos(radians(pitch))
    ztp = zmouth - Lp * sin(radians(pitch))
    for o in brz:
        assign(o, bronze)

    # patina: verdigris collects in the recesses, on ledges and in the run-off below the mouth; only the proud
    # forms (nose, brow, muzzle) are rubbed back to warm metal
    # (the verdigris mask keys hard at 0.5, so the pattern itself has to be the patina map)
    nz_br = noise_mask(0.05, seed=sd + 3, lo=0.25, hi=0.75)
    nz_b2 = noise_mask(0.017, seed=sd + 4, lo=0.2, hi=0.8)
    if p.mask == "lion":
        def hface(x, z):
            return _lion_h(x / Rm, (z - zm) / (Rm * 1.05))
    elif p.mask == "green_man":
        def hface(x, z):
            X, Z = x / (Rm * fr * 0.82), (z - zm) / (Rm * fr * 1.12)
            return _gman_h(X * 0.82, Z) if X * X + Z * Z < 0.98 else 0.0
    else:
        hface = None
    ring8 = [(0.085 * Rm * cos(k * pi / 4), 0.085 * Rm * sin(k * pi / 4)) for k in range(8)]

    def face_special(co, n):
        out = np.zeros(len(co))
        for i in range(len(co)):
            x, z = float(co[i, 0]), float(co[i, 2])
            h0 = hface(x, z)
            out[i] = sum(hface(x + dx, z + dz) for dx, dz in ring8) / 8.0 - h0
        cav = np.clip(out / 0.028, 0, 1)                                 # sockets, creases, the mouth, the rim of the face
        cav = cav * (1.0 - np.exp(-(co[:, 0] / (0.13 * Rm)) ** 2) * (co[:, 2] > zm + 0.12 * Rm))    # (not the forehead furrow)
        below = np.clip((zmouth + 0.01 - co[:, 2]) / (Rm * 0.25), 0, 1) * np.exp(-(co[:, 0] / (Rm * 0.3)) ** 2)
        return np.clip(0.2 + 0.5 * cav * (0.65 + 0.35 * nz_b2(co, n)) + 0.3 * below * nz_br(co, n), 0, 0.75)

    def relief_special(co, n):
        dy = ym - co[:, 1]                                               # height above the wall: green sits low, between the forms
        low = np.clip(1.0 - (dy - 0.012) / 0.014, 0, 1)
        run = np.clip((zmouth - co[:, 2]) / (Rm * 0.4), 0, 1) * np.exp(-(co[:, 0] / (Rm * 0.45)) ** 2)
        tipz = np.clip((np.hypot(co[:, 0], co[:, 2] - zm) - 0.55 * Rm) / (0.5 * Rm), 0, 1) * (1.0 - low)     # weathered tips of locks / leaves
        return np.clip(0.16 + low * (0.06 + 0.4 * nz_br(co, n) - 0.07 * np.clip((co[:, 2] - zm) / Rm, 0, 1)) + 0.3 * run * low + 0.5 * tipz * nz_br(co, n) + 0.1 * np.clip(n[:, 2], 0, 1), 0, 0.75)

    def bronze_wear(co, n):
        prot = (ym - co[:, 1]) / (Rm * hsc)
        return 0.42 * np.clip((prot - 0.4) / 0.22, 0, 1) * np.clip(-n[:, 1], 0, 1)

    for o in brz:
        if o is pipe:
            paint(o, wear=0.35, special=0.3)
        elif p.basin != "bowl" and o is brz[0]:
            paint(o, special=mask_mul(nz_b2, 0.62), wear=0.1)          # overflow pipe: always wet
        elif o is mask and hface is not None:
            paint(o, special=face_special, wear=bronze_wear)
        else:
            paint(o, special=relief_special, wear=0.12)

    # ---------------------------------------------------------------- water: basin surface, stream, froth
    # The basin surface is flat: its ripples, the rings running out from the landing and the foam are drawn by the
    # `water` material from two painted masks (wear = agitation, highest at the landing and falling linearly with
    # the distance; special = foam). Falling water follows the `water_fall` contract: U = metres round the strand,
    # V = along the flow, up from the foot and paced by the speed of the water (the pattern scrolls down V at
    # 2.9 units/s); wear = break-up (0 glassy .. 0.75 ropes .. 1 drops), special = foam.
    yland = max(ytp - 0.42, yfront + tw + 0.16)

    def vattr(o, name, vals):
        me = o.data
        at = me.attributes.get("m_" + name) or me.attributes.new("m_" + name, 'FLOAT', 'POINT')
        at.data.foreach_set("value", np.clip(np.asarray(vals, dtype=np.float32), 0.0, 1.0))

    wpts = list(inner_w)
    qa, qb = wpts[-1], wpts[0]
    ncl = max(2, int(math.hypot(qb[0] - qa[0], qb[1] - qa[1]) / 0.11))
    wpts += [(qa[0] + (qb[0] - qa[0]) * k / ncl, qa[1] + (qb[1] - qa[1]) * k / ncl) for k in range(1, ncl)]
    nw, nr = len(wpts), 18
    wv = [(0.0, yland, zw)]
    for k in range(1, nr + 1):
        g = k / nr
        for (x, y) in wpts:
            Rr = math.hypot(x, y - yland)
            d = lerp(min(0.5 * g ** 1.2, 0.85 * Rr), Rr, g ** 4)
            wv.append((x * d / Rr, yland + (y - yland) * d / Rr, zw))
    wf = [(0, 1 + i, 1 + (i + 1) % nw) for i in range(nw)]
    for k in range(nr - 1):
        o0, o1 = 1 + k * nw, 1 + (k + 1) * nw
        wf += [(o0 + i, o1 + i, o1 + (i + 1) % nw, o0 + (i + 1) % nw) for i in range(nw)]
    water = mesh_object("water", wv, wf)
    if sum(pl.normal.z for pl in water.data.polygons) < 0:
        for pl in water.data.polygons:
            pl.flip()
        water.data.update()
    uv_planar(water, (1, 0, 0), (0, 1, 0))
    smooth(water, 60)
    assign(water, m_water)
    if p.running and len(water.data.vertices) == len(wv):
        dl = [math.hypot(q[0], q[1] - yland) for q in wv]
        vattr(water, "wear", [clamp(0.75 - d / 1.2, 0.0, 1.0) for d in dl])
        vattr(water, "special", [clamp(1.0 - (d - 0.03) / 0.09, 0.0, 1.0) ** 1.4 for d in dl])

    if p.running:
        rw = rng.fork("water")
        s0 = math.tan(radians(pitch)) * abs(yland - ytp)
        hf_ = max(ztp - zw - s0, 0.02)
        tfl_ = math.sqrt(2.0 * hf_ / 9.8)

        def jp(t, dx=0.0, dy=0.0):
            return (dx, ytp + 0.004 + (yland - ytp) * t + dy, ztp - s0 * t - (ztp - zw - s0) * t * t)

        def jspeed(t):
            return math.hypot((yland - ytp) / tfl_, (s0 + 2.0 * hf_ * t) / tfl_)

        def strand(name, pts, ts, rad, sides, scale_fn, brk, foam, speed):
            """Round strand of falling water along `pts` (ts = its parameter at each point, for the speed)."""
            n = len(pts)
            P = np.array(pts, dtype=np.float64)
            vv = [0.0]
            for i in range(1, n):
                vv.append(vv[-1] + float(np.linalg.norm(P[i] - P[i - 1])) * 2.9 / max(speed(0.5 * (ts[i] + ts[i - 1])), 1.2))
            uo = rw.u(0, 5)
            V_, F_, WR_, SP_ = [], [], [], []
            for i in range(n):
                tg = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
                tg /= max(float(np.linalg.norm(tg)), 1e-9)
                a1 = np.cross(tg, (1.0, 0.0, 0.0))
                a1 /= max(float(np.linalg.norm(a1)), 1e-9)
                a2 = np.cross(tg, a1)
                f = i / (n - 1)
                rr = rad * scale_fn(f)
                for k in range(sides):
                    a = pi + 2 * pi * k / sides                  # (the UV seam is at the back)
                    q = P[i] + rr * (cos(a) * a1 + sin(a) * a2)
                    V_.append((float(q[0]), float(q[1]), float(q[2])))
                    WR_.append(brk(f))
                    SP_.append(foam(f))
            for i in range(n - 1):
                for k in range(sides):
                    F_.append((i * sides + k, i * sides + (k + 1) % sides, (i + 1) * sides + (k + 1) % sides, (i + 1) * sides + k))
            o = mesh_object(name, V_, F_)
            me = o.data
            lay = me.uv_layers[0] if len(me.uv_layers) else me.uv_layers.new(name="UVMap")
            for pi_, pl in enumerate(me.polygons):
                k0 = pi_ % sides
                for li in pl.loop_indices:
                    i, k = divmod(me.loops[li].vertex_index, sides)
                    kk = k0 + 1 if k == (k0 + 1) % sides else k0
                    lay.data[li].uv = (uo + 2 * pi * rad * kk / sides, vv[-1] - vv[i])
            uv_keep(o)
            smooth(o, 80)
            vattr(o, "wear", WR_)
            vattr(o, "special", SP_)
            return o

        # glassy rod at the lip that necks, then fattens, opens into ropes and is white where it strikes the basin
        jt = [1.03 * i / 17 for i in range(18)]         # (it ends just under the surface)
        falls = [strand("jet", [jp(t) for t in jt], jt, 0.0072, 8,
                        lambda t: 1.0 - 0.14 * sin(pi * clamp(t / 0.5)) + 0.7 * smoothstep(0.4, 1.0, t),
                        lambda t: 0.03 + 0.7 * t ** 1.7, lambda t: 0.45 * smoothstep(0.82, 1.0, t), jspeed)]
        for k in range(3):                         # strands shed by the rod on its way down
            t0 = rw.u(0.4, 0.6)
            ex, ey = rw.u(-1, 1) * 0.017, rw.u(-1, 1) * 0.014
            st_ = [t0 + (1.0 - t0) * i / 8 for i in range(9)]
            sp_ = [jp(t0 + (1.0 - t0) * i / 8, ex * (i / 8) ** 1.5, ey * (i / 8) ** 1.5) for i in range(9)]
            falls.append(strand("jet_strand", sp_, st_, 0.0038, 5, lambda t: 0.25 + 0.75 * smoothstep(0.0, 0.35, t),
                                lambda t: 0.62 + 0.33 * t, lambda t: 0.3 * smoothstep(0.8, 1.0, t), jspeed))
        # froth where the stream lands: a low boiling mound, a ragged band of spray round it and a few ribbons of drops
        # flung clear (V runs DOWN every throw, so the material's pattern travels up and outwards with the spray)
        fv, ff, fuv, fwr, fsp = [], [], [], [], []

        def fadd(q, uv, wr, sp):
            fv.append(q)
            fuv.append(uv)
            fwr.append(wr)
            fsp.append(sp)
            return len(fv) - 1

        nsg, rb_, hm_ = 10, 0.028, 0.013
        fu, fvo = rw.u(0, 5), rw.u(0, 3)
        c0 = fadd((0.0, yland, zw + hm_), (fu, fvo), 0.7, 0.6)
        rings = []
        for (rf_, fz, wr_, sq_) in ((0.45, 0.85, 0.76, 0.6), (0.8, 0.5, 0.84, 0.5), (1.0, -0.3, 0.96, 0.2)):
            rings.append([fadd((rb_ * rf_ * cos(2 * pi * i / nsg), yland + rb_ * rf_ * sin(2 * pi * i / nsg), zw + hm_ * fz),
                               (fu + rb_ * rf_ * cos(2 * pi * i / nsg), fvo - 0.45 * rf_), wr_, sq_) for i in range(nsg)])
        for i in range(nsg):
            i1 = (i + 1) % nsg
            ff.append((c0, rings[0][i], rings[0][i1]))
            for a_, b_ in zip(rings[:-1], rings[1:]):
                ff.append((a_[i], b_[i], b_[i1], a_[i1]))
        hc_ = rw.u(0.04, 0.06)
        q1, q2, q3 = rw.u(0, 6.28), rw.u(0, 6.28), rw.u(0, 6.28)
        rows = ((0.0, 0.0, 0.93, 0.4), (0.1, 0.4, 0.96, 0.2), (0.24, 0.78, 0.98, 0.08), (0.4, 1.0, 1.0, 0.0))
        b0 = len(fv)
        for i in range(nsg + 1):
            th = 2 * pi * i / nsg
            kh = 0.72 + 0.3 * sin(3 * th + q1) * sin(2 * th + q2) + 0.16 * sin(7 * th + q3)
            for (fr, fz, wr_, sq_) in rows:
                rr_ = 0.024 + 0.07 * fr * kh
                fadd((rr_ * cos(th), yland + rr_ * sin(th), zw - 0.004 + hc_ * fz * kh),
                     (fu + 0.4 * i / nsg, fvo + 1.7 - 9.0 * hc_ * kh * (fz + 0.5 * fr)), wr_, sq_)
        for i in range(nsg):
            for q in range(len(rows) - 1):
                i0 = b0 + i * len(rows) + q
                ff.append((i0, i0 + len(rows), i0 + len(rows) + 1, i0 + 1))
        for _ in range(3):
            da = rw.u(0, 2 * pi)
            ca, sa = cos(da), sin(da)
            sp0, el = rw.u(0.7, 1.1), rw.u(0.95, 1.25)
            vr, vz = sp0 * cos(el), sp0 * sin(el)
            tt_ = 2 * vz / 9.8 * rw.u(0.7, 0.96)
            wd, tilt = rw.u(0.007, 0.012), rw.u(-0.7, 0.7)
            u_, v_ = rw.u(0, 3), rw.u(0, 3)
            b0 = len(fv)
            prev = None
            for q in range(7):
                t = tt_ * q / 6
                pr, pz = vr * t, vz * t - 4.9 * t * t
                if prev is not None:
                    v_ -= math.hypot(pr - prev[0], pz - prev[1]) * 2.9 / max(math.hypot(vr, vz - 9.8 * t), 0.8)
                prev = (pr, pz)
                hw = 0.5 * wd * (0.55 + 0.9 * q / 6)
                for e in (-1.0, 1.0):
                    fadd(((0.02 + pr) * ca - sa * hw * e, yland + (0.02 + pr) * sa + ca * hw * e, zw + 0.004 + pz + tilt * hw * e),
                         (u_ + hw * e, v_), 0.93 + 0.07 * q / 6, 0.15 * (1.0 - q / 6) ** 2)
            for q in range(6):
                i0 = b0 + 2 * q
                ff.append((i0, i0 + 1, i0 + 3, i0 + 2))
        fr_o = mesh_object("froth", fv, ff)
        lay = fr_o.data.uv_layers[0] if len(fr_o.data.uv_layers) else fr_o.data.uv_layers.new(name="UVMap")
        for lp in fr_o.data.loops:
            lay.data[lp.index].uv = fuv[lp.vertex_index]
        uv_keep(fr_o)
        smooth(fr_o, 80)
        vattr(fr_o, "wear", fwr)
        vattr(fr_o, "special", fsp)
        falls.append(fr_o)
        if p.basin != "bowl":
            hd_ = max(z_of - 0.012 - zdr, 0.02)
            dt_ = [i / 8 for i in range(9)]
            dpath = [(x_of + 0.003 * sin(t * 7), ytip + 0.002 - 0.037 * math.sqrt(t), z_of - 0.012 - hd_ * t * t) for t in dt_]
            falls.append(strand("dribble", dpath, dt_, 0.0036, 6, lambda t: 1.0 - 0.35 * t, lambda t: 0.25 + 0.6 * t, lambda t: 0.0,
                                lambda t: 2.0 * hd_ * t / math.sqrt(2.0 * hd_ / 9.8)))
        for o in falls:
            assign(o, m_fall)
        fx("overflow_drip", "drip", (x_of, ytip - 0.03, zdr + 0.05), size=0.05)

    # ---------------------------------------------------------------- lantern positions (needed for the soot)
    nlan = int(p.lanterns)
    reach = 0.52
    z_lb = max(2.0, zf0 - 0.28)                 # underside of the lantern: clear of heads, level with the capitals / frieze
    lan_hi = zf0 - 0.2 < z_lb - 0.45            # low fountain: the bracket is fixed to the frieze instead of the pilaster shaft
    za = (zf0 + zf1) / 2 + 0.04 if lan_hi else min(z_lb - 0.3, zf0 - 0.2)
    lan_sides = () if nlan == 0 else ((-1, 1) if nlan == 2 else (rng.fork("lan").pick((-1, 1)),))
    lan_x = [s_ * (W / 2 - pw / 2) for s_ in lan_sides]
    zfl = z_lb + 0.03 + 0.062

    # ---------------------------------------------------------------- materials + weathering of the stonework
    for o in basin_objs:
        stones.append(o)
    for o in stones + [tab]:
        assign(o, st_trim)
    for o in fields:
        assign(o, st_field)

    nz_a = noise_mask(0.3, seed=sd + 11, lo=0.3, hi=0.7)
    nz_b = noise_mask(0.08, seed=sd + 12, lo=0.3, hi=0.75)
    nz_c = noise_mask(0.035, seed=sd + 13, lo=0.35, hi=0.7)
    nz_d = noise_mask(0.7, seed=sd + 14, lo=0.25, hi=0.75)
    wet = 0.3 + 0.7 * moss

    def streaks(co):
        """(core, halo) of the two water runs: below the mouth down the niche into the basin, and below the overflow."""
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        gm = (z < zmouth) * np.clip((zmouth - z) / 0.06, 0, 1) * (z > zw - 0.06) * (y > yface - 0.1)
        go = (z < z_of + 0.01) * (y < yfront + 0.16) * np.clip((z_of + 0.01 - z) / 0.04, 0, 1)
        wob = 0.008 * np.sin(z * 23.0 + sd)
        dm_x, do_x = x + wob, x - x_of + wob
        core = np.exp(-(dm_x / 0.026) ** 2) * gm * 0.7 + np.exp(-(do_x / 0.03) ** 2) * go
        halo = np.exp(-(dm_x / 0.055) ** 2) * gm + np.exp(-(do_x / 0.052) ** 2) * go * 0.45
        return core, halo

    film_poly = [(cxb + (x - cxb) * 1.05, min(cyb + (y - cyb) * 1.05, y0 + 0.02)) for x, y in inner_w]

    def film(co, n):
        """Algae film on the inside of the basin at the waterline and on the wet back wall."""
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        inward = ((cxb - x) * n[:, 0] + (cyb - y) * n[:, 1]) > 0.15
        ins = np.zeros(len(x), dtype=bool)                              # inside the basin outline (plan view)
        for i in range(len(film_poly)):
            (x1, y1), (x2, y2) = film_poly[i], film_poly[(i + 1) % len(film_poly)]
            if abs(y2 - y1) > 1e-9:
                ins ^= ((y1 > y) != (y2 > y)) & (x < x1 + (x2 - x1) * (y - y1) / (y2 - y1))
        return inward * ins * (z > zw - 0.12) * np.clip(1.0 - (z - zw - 0.025) / 0.075, 0, 1) * (n[:, 2] < 0.6) * (y < -D + 0.03)

    km = moss ** 0.5

    def st_special(co, n):
        z = co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        low = np.clip(1.0 - z / (0.2 + 0.3 * moss), 0, 1) ** 0.7
        core, halo = streaks(co)
        a, b = nz_a(co, n), nz_b(co, n)
        m = km * (1.25 * low * (0.45 + 0.55 * a) * (0.6 + 0.4 * b)                         # shaded base, plinth
                  + 0.95 * up * (0.3 + 0.7 * a) * (z < zrim - 0.05) * (z > 0.05)           # ledges low down
                  + 0.75 * up * nz_d(co, n) * (0.4 + 0.6 * b) * (z > zf1))                 # lichen on the cornice and crown
        m = m + (0.7 + 0.3 * moss) * (1.05 * film(co, n) * (0.75 + 0.25 * b)               # algae at the waterline
                                      + 1.25 * np.clip(halo - 0.9 * core, 0, 1) * (0.5 + 0.5 * nz_c(co, n)))   # green edges of the runs
        return np.clip(m, 0, 0.95)

    def st_grime(co, n):
        x, z = co[:, 0], co[:, 2]
        tone = 0.13 + 0.27 * nz_d(co, n) * (0.5 + 0.5 * nz_a(co, n))                       # patchy overall toning
        low = np.clip(1.0 - z / 0.6, 0, 1) * 0.3 * (0.4 + 0.6 * nz_a(co, n))
        under = np.clip(-n[:, 2], 0, 1) * 0.4
        # rain runs below the cornice, the capitals and the sill
        fine = 0.5 + 0.5 * np.sin(x * 47.0 + 3.0 * np.sin(x * 11.0 + sd))
        rain = np.clip(1.0 - (zf1 - z) / 0.6, 0, 1) * (z < zf1) * fine * 0.36
        rain2 = np.clip(1.0 - (zs - z) / 0.3, 0, 1) * (z < zs) * fine * 0.36
        soot = 0.0
        for lx in lan_x:                                                                   # lamp smoke on the stone behind each lantern
            soot = soot + 0.36 * np.exp(-((x - lx) / 0.26) ** 2) * np.clip((z - (zfl - 0.3)) / 0.3, 0, 1) * np.clip(1.0 - (z - zfl) / 0.8, 0, 1)
        core, halo = streaks(co)
        return np.clip(tone + low + under + rain + rain2 + soot + 0.1 * halo + 0.3 * film(co, n) - 0.3 * core, 0, 1)

    def st_wear(co, n):
        core, halo = streaks(co)
        cope = np.clip(n[:, 2], 0, 1) * (np.abs(co[:, 2] - zrim) < 0.03) * 0.7           # rim polished by hands and buckets
        return np.clip(cope + 1.3 * core * (0.75 + 0.25 * nz_c(co, n)), 0, 1.0)            # pale lime scale down the runs

    for o in stones + fields:
        paint(o, grime=st_grime, special=st_special, wear=st_wear)
    # the tablet only takes soft, low-frequency toning so the cut letters stay crisp
    paint(tab, grime=lambda co, n: 0.13 + 0.22 * nz_d(co, n) + 0.2 * np.clip(1.0 - (zf1 - co[:, 2]) / 0.2, 0, 1), special=0.0)

    # ---------------------------------------------------------------- moss: cushions creeping along the damp joints
    nchain = int(round(moss * 11))
    if nchain > 0:
        rmo = rng.fork("moss")
        tracks = []
        yp = -(yo + 0.064)
        bw = 0.27 if p.basin == "bowl" else (Rb if p.basin != "trough" else hw_) + 0.08
        if xo + 0.06 - bw > 0.12:
            tracks.append([(-xo - 0.05, yp, 0.0), (-bw, yp, 0.0)])
            tracks.append([(bw, yp, 0.0), (xo + 0.05, yp, 0.0)])
        for s_ in (-1, 1):
            tracks.append([(s_ * (xo + 0.066), -0.04, 0.0), (s_ * (xo + 0.066), yp + 0.03, 0.0)])            # foot of the plinth ends
            tracks.append([(s_ * (xi + 0.04), -(yo + 0.03), zpl + 0.07), (s_ * (xo - 0.02), -(yo + 0.03), zpl + 0.07)])   # plinth ledge
        if bst:
            foot = [(x + nx * 0.064, y + ny * 0.064, 0.0) for (x, y, nx, ny) in bst if y < yp - 0.03]
            ledge = [(x + nx * 0.036, y + ny * 0.036, zb1 - 0.038) for (x, y, nx, ny) in bst if y < yp - 0.03]
            third = max(3, len(foot) // 3)
            tracks += [foot[:third], foot[-third:], ledge[:third], ledge[-third:]]
        else:
            tracks.append([(-0.25, y0 - 0.262, 0.0), (0.25, y0 - 0.262, 0.0)])
        res = [resample_path(t, step=0.03, smooth_path=False) for t in tracks if len(t) >= 2]
        res = [t for t in res if len(t) >= 4]
        order = sorted(range(len(res)), key=lambda i: rmo.u())
        blob = sphere("moss_blob", 0.5, segments=5, rings=3, scale=(1.0, 1.0, 0.45))
        realize(blob)
        displace_noise(blob, strength=0.08, scale=0.4, seed=sd)
        mm = []
        for c in range(nchain):
            tr_ = res[order[c % len(order)]]
            ln = min(len(tr_), int(rmo.u(4, 8) * (0.7 + 0.8 * moss)))
            i0 = int(rmo.u(0, len(tr_) - ln + 0.999))
            base = rmo.u(0.035, 0.055) * (0.8 + 0.4 * moss)
            for j in range(ln):
                q = tr_[i0 + j]
                env = 0.45 + 0.55 * sin(pi * (j + 0.5) / ln) ** 0.7
                s_ = base * env * rmo.u(0.8, 1.2)
                mm.append(trs((q[0] + rmo.jitter(0, 0.007), q[1] + rmo.jitter(0, 0.007), q[2] + 0.003), (rmo.jitter(0, 10), rmo.jitter(0, 10), rmo.u(0, 360)),
                              (s_ * rmo.u(1.0, 1.5), s_, s_ * rmo.u(0.7, 1.1))))
        mo = instances_to_mesh("moss", blob, mm, keep_base=False)
        assign(mo, m_moss)
        smooth(mo, 80)

    # ---------------------------------------------------------------- forged swan-neck brackets carrying upright lanterns
    if nlan > 0:
        glassm = mat("glass_lantern", glow=0.8)
        flame_m = mat("emissive", "#ffb45e", glow=7)
        bar = rounded_rect(0.006, 0.017, 0.0015, 1)
        sq = rounded_rect(0.012, 0.012, 0.002, 1)
        s2 = math.sqrt(2.0)
        for side in lan_sides:
            irons = []
            if lan_hi:
                bzl, zpc, hpl = 0.0, (zf0 + zf1) / 2, (zf1 - zf0) / 2 - 0.04
            else:
                bzl, zpc, hpl = B * (1.0 - clamp((za - 0.1 - zpl) / (zf0 - zpl))), za - 0.1, 0.18
            zst_ = zpc - hpl + 0.02
            x0 = side * (W / 2 - pw / 2 + bzl * 0.5)
            yf = ypf - bzl * 0.7 + 0.002
            yl = yf - reach
            irons.append(box("lan_plate", (0.052, 0.008, 2 * hpl), loc=(x0, yf - 0.004, zpc), bevel=0.002))
            for sg_ in (1, -1):
                irons.append(cone("lan_plate_tip", 0.026, 0.0, 0.03, loc=(x0, yf - 0.004, zpc + sg_ * (hpl + 0.015)),
                                  rot=(0 if sg_ > 0 else 180, 0, 0), verts=4))
                irons.append(bolt("lan_bolt", 0.007, 0.005, loc=(x0, yf - 0.008, zpc + sg_ * (hpl - 0.03)), rot=(90, 0, 0), kind="pyramid"))
            arm = bezier_path((x0, yf - 0.004, za), (x0, yf - reach * 0.85, za - 0.03), (x0, yl, za + 0.12 * (z_lb - za)), (x0, yl, z_lb - 0.01), n=20)
            irons.append(sweep("lan_arm", arm, sq, up=(1, 0, 0)))
            q = arm[8]
            dzs = za - zst_
            stay = bezier_path((x0, yf - 0.006, zst_), (x0, yf - 0.15, zst_ - 0.06 * dzs), (x0, q[1] + 0.03, q[2] - 0.62 * dzs), (x0, q[1], q[2] - 0.012), n=14)
            irons.append(sweep("lan_stay", stay, bar, up=(1, 0, 0)))
            sp2 = spiral_path(0.005, 0.19 * dzs, 1.5, n=22, a0_deg=20, plane="YZ", center=(x0, yf - 0.29 * dzs, za - 0.33 * dzs))
            irons.append(sweep("lan_scroll", sp2, bar, up=(1, 0, 0), scale_fn=lambda t: 0.5 + 0.5 * t))
            turns = 1.25
            sp3 = spiral_path(0.006, 0.032, turns, n=18, a0_deg=90 - turns * 360, plane="YZ", center=(x0, yl + 0.045, z_lb - 0.075))
            irons.append(sweep("lan_curl", sp3, bar, up=(1, 0, 0), scale_fn=lambda t: 0.5 + 0.5 * t))
            irons.append(cylinder("lan_collar", 0.011, 0.02, loc=(x0, q[1], q[2] - 0.004), rot=(0, 90, 0), verts=8))
            # lantern: tapered square cage standing on the arm, vented roof with a knob
            zb_ = z_lb + 0.03
            wt_, wb_ = 0.07, 0.048
            zr0 = zb_ + 0.2
            irons.append(lathe("lan_seat", [(0.009, z_lb - 0.014), (0.03, z_lb - 0.01), (0.032, z_lb - 0.004), (0.014, z_lb + 0.002)], segments=10, loc=(x0, yl, 0)))
            irons.append(lathe("lan_roof", [(wt_ * s2 + 0.022, zr0 - 0.004), (wt_ * s2 + 0.024, zr0 + 0.004), (wt_ * s2 * 0.7, zr0 + 0.02),
                                            (wt_ * s2 * 0.42, zr0 + 0.05), (0.03, zr0 + 0.07), (0.03, zr0 + 0.078), (0.04, zr0 + 0.082),
                                            (0.036, zr0 + 0.09), (0.012, zr0 + 0.096), (0.0, zr0 + 0.097)], segments=4, loc=(x0, yl, 0), rot=(0, 0, 45),
                               smooth_angle=20))
            irons.append(sphere("lan_knob", 0.011, loc=(x0, yl, zr0 + 0.104), segments=8, rings=5))
            irons.append(lathe("lan_base", [(0.0, zb_ - 0.028), (0.012, zb_ - 0.026), (0.016, zb_ - 0.012), (wb_ * s2 * 0.7, zb_ - 0.008),
                                            (wb_ * s2 + 0.006, zb_ - 0.004), (wb_ * s2 + 0.006, zb_ + 0.006), (wb_ * s2 - 0.008, zb_ + 0.006),
                                            (0.0, zb_ + 0.004)], segments=4, loc=(x0, yl, 0), rot=(0, 0, 45), smooth_angle=20))
            for cxs, cys in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                irons.append(sweep("lan_bar", [(x0 + cxs * wb_, yl + cys * wb_, zb_), (x0 + cxs * wt_, yl + cys * wt_, zr0)], 0.0042, sides=6))
            irons.append(lathe("lan_rail", [(wt_ * s2 - 0.004, zr0 - 0.012), (wt_ * s2 + 0.005, zr0 - 0.012), (wt_ * s2 + 0.005, zr0 - 0.002),
                                            (wt_ * s2 - 0.004, zr0 - 0.002), (wt_ * s2 - 0.004, zr0 - 0.012)], segments=4, loc=(x0, yl, 0), rot=(0, 0, 45), smooth_angle=20))
            irons.append(cylinder("lan_font", 0.016, 0.03, loc=(x0, yl, zb_ + 0.02), verts=10, radius2=0.011))
            for o in irons:
                assign(o, iron)
                paint(o, special=mask_mul(noise_mask(0.05, seed=sd + 21, lo=0.4, hi=0.75), 0.35))
            gl = lathe("lan_glass", [((wb_ - 0.002) * s2, zb_ + 0.004), ((wt_ - 0.002) * s2, zr0 - 0.006)], segments=4, loc=(x0, yl, 0), rot=(0, 0, 45), smooth_angle=20)
            assign(gl, glassm)
            fl = sphere("lan_flame", 0.011, loc=(x0, yl, zfl), segments=10, rings=6, scale=(1.0, 1.0, 2.3))
            assign(fl, flame_m)
            # shadow_near reaches past the lantern and its whole bracket, so the fitting throws no stripes on the stone
            light("lantern_%s" % ("l" if side < 0 else "r"), (x0, yl, zfl), color="#ffbf6a", intensity=p.light, range=9.0, flicker=0.25,
                  shadow_near=0.8)

    # ---------------------------------------------------------------- bronze drinking cup chained to the pilaster
    rcu = rng.fork("cup")
    cs = rcu.pick((-1, 1))
    xcup = cs * (sx - 0.1)
    ycup = -(D + 0.02)
    cup = lathe("cup", [(0.0, 0.0), (0.027, 0.0), (0.03, 0.004), (0.031, 0.02), (0.036, 0.074), (0.039, 0.08), (0.0355, 0.08),
                        (0.0325, 0.074), (0.0275, 0.008), (0.0, 0.008)], segments=16, loc=(xcup, ycup, zn0 - 0.001), rot=(0, 0, rcu.u(0, 360)))
    ear = torus("cup_ear", 0.014, 0.0035, loc=(xcup + cs * 0.044, ycup, zn0 + 0.045), rot=(90, 0, 0), major_seg=12, minor_seg=6)
    for o in (cup, ear):
        assign(o, bronze)
        paint(o, wear=0.7, special=0.15)
    zst = zn0 + 0.42
    bzs = B * (1.0 - clamp((zst - zpl) / (zf0 - zpl)))
    pst_ = (cs * (xi + 0.06), ypf - 0.7 * bzs - 0.006, zst)
    staple = torus("cup_staple", 0.014, 0.004, loc=pst_, rot=(0, 90, 0), major_seg=12, minor_seg=6)
    ch = chain("cup_chain", catenary((pst_[0], pst_[1] - 0.008, pst_[2] - 0.012), (xcup + cs * 0.056, ycup - 0.002, zn0 + 0.045), sag=0.1, n=18),
               link_len=0.03, wire=0.0026, sides=4)
    for o in (staple, ch):
        assign(o, iron)
        paint(o, special=0.5)
