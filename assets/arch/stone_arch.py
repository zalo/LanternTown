"""Dressed-stone archway surround for a passage or gate opening. Every stone is its own block bedded on a
recessed lime-mortar core: rough-punched plinth blocks with a weathered splay, long-and-short punched jamb
quoins with a recessed inner order and chamfered / roll-moulded arrises, moulded impost blocks (springers with
a weathered shoulder on segmental arches), a ring of radially jointed fine-droved voussoirs, a projecting
keystone with an emblem carved in bold relief inside a sunk panel, an optional hood mould that dies into the
keystone / springers, dressed octagonal wheel-guard stones engaged in the jamb corners, a tethering ring and
hook pintles leaded into the back. Stones are individually out of true, chipped, worn and cracked by age and
seed; soot, rising damp, run-off streaks, a moss film and joint-hugging moss cushions are driven by age / moss.
The opening (width x height clear) is empty; wall plane y = 0 at the front face, the arch runs back to +depth.
"""
import numpy as np
from math import asin, acos, atan2, ceil, exp
from mathutils import Euler, Matrix
from tk import *

# tri_budget: up to ~70 individually modelled stones whose faces are gridded (~12 cm) so that the per-vertex soot /
# damp / moss masks, the out-of-true faces and the chipped arrises have something to live on
META = {"name": "Stone archway surround", "class": "component", "mount": "wall_ground", "focus": 0.8, "tri_budget": 22000}
PARAMS = {
    "width": {"default": 2.4, "min": 1.0, "max": 5.0, "desc": "clear opening width (m)"},
    "height": {"default": 3.3, "min": 2.0, "max": 6.0, "desc": "clear opening height to the crown (m)"},
    "depth": {"default": 0.6, "min": 0.3, "max": 1.6, "desc": "wall thickness the arch runs through (m)"},
    "arch": {"default": "round", "choices": ["round", "segmental", "pointed"], "desc": "arch form"},
    "rise": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "segmental: flat .. high; pointed: blunt .. lancet (round ignores it)"},
    "ring": {"default": 0.42, "min": 0.28, "max": 0.7, "desc": "face depth of the voussoir ring (m)"},
    "jamb": {"default": 0.5, "min": 0.32, "max": 0.9, "desc": "mean face width of the jamb quoins (m)"},
    "orders": {"default": 2, "choices": [1, 2], "desc": "2 = recessed inner order on jamb and arch"},
    "arris": {"default": "chamfer", "choices": ["chamfer", "roll", "plain"], "desc": "treatment of the opening arris"},
    "keystone": {"default": 1.0, "min": 0.7, "max": 1.5, "desc": "keystone width factor"},
    "emblem": {"default": "crown", "choices": ["crown", "shield", "star", "rosette", "none"], "desc": "relief carved on the keystone"},
    "inscription": {"default": "1687", "desc": "date / short text cut on the keystone below the emblem ('' = none)"},
    "emblem_finish": {"default": "carved", "choices": ["carved", "bronze", "gilt"], "desc": "emblem in stone, cast bronze or gilded"},
    "hood": {"default": True, "desc": "projecting hood (label) mould over the arch"},
    "guards": {"default": True, "desc": "wheel-guard stones at the feet"},
    "threshold": {"default": True, "desc": "worn threshold slabs between the jambs (wheel ruts on carriage widths)"},
    "ironwork": {"default": True, "desc": "tethering ring on the front and gate pintles on the back"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "moss / damp growth"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "soot, rain streaks and rubbed arrises"},
    "stone": {"default": "#b9ad98", "desc": "stone colour"},
    "mount": {"default": "ground", "choices": ["ground", "wall_ground"],
              "desc": "ground = free-standing gateway: the bonding tails of the quoins are left hammer-broken and get weathered; "
                      "wall_ground = built into a host wall filling y 0..depth: tails stay plain (they are buried)"},
}
VARIANTS = {
    "pointed_gate": {"arch": "pointed", "width": 2.0, "height": 4.0, "depth": 0.9, "rise": 0.45, "arris": "roll", "emblem": "shield",
                     "emblem_finish": "bronze", "stone": "#868a8c", "moss": 0.6, "age": 0.75, "jamb": 0.56, "inscription": "1542", "seed": 4},
    "carriage_segmental": {"arch": "segmental", "width": 3.8, "height": 3.3, "depth": 0.75, "rise": 0.45, "ring": 0.52, "jamb": 0.56,
                           "orders": 1, "emblem": "star", "stone": "#c6ae88", "moss": 0.15, "keystone": 1.25, "inscription": "1790", "seed": 11},
    "postern_plain": {"width": 1.15, "height": 2.25, "depth": 0.42, "ring": 0.3, "jamb": 0.36, "orders": 1, "arris": "chamfer",
                      "emblem": "none", "inscription": "", "hood": False, "guards": False, "ironwork": False, "moss": 0.9, "age": 0.9,
                      "stone": "#b58a6c", "seed": 23},
    "deep_passage": {"width": 3.0, "height": 3.9, "depth": 1.5, "ring": 0.5, "jamb": 0.62, "emblem": "rosette",
                     "emblem_finish": "gilt", "inscription": "MDCXII", "stone": "#d3c8ab", "moss": 0.3, "age": 0.65, "seed": 31},
}

JOINT = 0.005         # half joint width (m)
CORE_E = 0.0035        # mortar bed sits this far behind the stone faces
PANEL = 0.022          # depth of the sunk emblem panel on the keystone
HOOD_H = 0.117         # radial height of the hood mould
HOOD_P = 0.07          # its projection


def _shade(hexc, f, warm=0.0):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    r, g, b = clamp(r * f * (1 + warm)), clamp(g * f), clamp(b * f * (1 - warm))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _corner(prev, cur, nxt, kind, size):
    """Replace a convex 90 deg profile corner by a chamfer or a quirked roll (bead)."""
    if not kind or size < 1e-4:
        return [cur]

    def unit(p, q):
        dx, dy = p[0] - q[0], p[1] - q[1]
        l = hypot(dx, dy)
        return dx / l, dy / l

    e1, e2 = unit(prev, cur), unit(nxt, cur)

    def P(s, r):
        return (cur[0] + e1[0] * s + e2[0] * r, cur[1] + e1[1] * s + e2[1] * r)

    if kind == "chamfer":
        return [P(size, 0), P(0, size)]
    r, q = size, size * 0.3
    rho = r - q
    pts = [P(r + q, 0), P(r + q, q)]
    for k in range(5):
        ang = radians(270 - 22.5 * k)
        pts.append(P(r + rho * cos(ang), r + rho * sin(ang)))
    pts += [P(q, r + q), P(0, r + q)]
    return pts


def _poly(C):
    out = []
    n = len(C)
    for i, (pt, kind, size) in enumerate(C):
        out += _corner(C[i - 1][0], pt, C[(i + 1) % n][0], kind, size)
    return out


def _sub(prof, step):
    """Subdivide the edges of a closed profile: step(p, q) -> longest allowed segment on that edge."""
    out = []
    n = len(prof)
    for i in range(n):
        p, q = prof[i], prof[(i + 1) % n]
        k = max(1, int(ceil(hypot(q[0] - p[0], q[1] - p[1]) / step(p, q) - 1e-6)))
        out += [(p[0] + (q[0] - p[0]) * j / k, p[1] + (q[1] - p[1]) * j / k) for j in range(k)]
    return out


def _block(name, prof, s_list, fn, post=None):
    """Cross-section `prof` [(u, v)] carried along the stations s_list through fn(u, v, s) -> xyz (world space).
    post(V) may move the (N, 3) vertex array before the mesh is made (per-stone irregularity)."""
    n = len(prof)
    V = np.array([fn(u, v, s) for s in s_list for (u, v) in prof], dtype=float)
    if post is not None:
        V = post(V)
    faces = []
    for i in range(len(s_list) - 1):
        for j in range(n):
            j2 = (j + 1) % n
            faces.append((i * n + j, i * n + j2, (i + 1) * n + j2, (i + 1) * n + j))
    faces.append(tuple(reversed(range(n))))
    k = (len(s_list) - 1) * n
    faces.append(tuple(range(k, k + n)))
    o = mesh_object(name, [tuple(float(c) for c in q) for q in V], faces)
    recalc_normals(o)
    uv_box(o)
    return o


def _localize(o, loc, roty):
    """Give a world-space stone its own frame (local Z = its bed normal) so the box-mapped tooling follows the stone."""
    set_loc_rot(o, loc, (0.0, roty, 0.0))
    o.data.transform(world_matrix(o).inverted())
    o.data.update()
    return o


def _slab(name, x0, x1, y0, y1, nx, ny, zf):
    """Paving slab: gridded top following zf(x, y), vertical skirt down to the ground, closed underneath."""
    verts = [(x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    verts = [(x, y, zf(x, y)) for x, y in verts]
    idx = lambda i, j: j * (nx + 1) + i
    faces = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)) for j in range(ny) for i in range(nx)]
    loop = [idx(i, 0) for i in range(nx + 1)] + [idx(nx, j) for j in range(1, ny + 1)] + \
           [idx(i, ny) for i in range(nx - 1, -1, -1)] + [idx(0, j) for j in range(ny - 1, 0, -1)]
    base = len(verts)
    verts += [(verts[k][0], verts[k][1], 0.0) for k in loop]
    m = len(loop)
    faces += [(loop[k], base + k, base + (k + 1) % m, loop[(k + 1) % m]) for k in range(m)]
    faces.append(tuple(base + k for k in range(m)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    uv_box(o)
    return o


def _loft(name, rings):
    """Stack of rectangles [(x0, x1, y0, y1, z)] -> moulded block."""
    verts, faces = [], []
    for (x0, x1, y0, y1, z) in rings:
        verts += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    for i in range(len(rings) - 1):
        for j in range(4):
            j2 = (j + 1) % 4
            faces.append((i * 4 + j, i * 4 + j2, (i + 1) * 4 + j2, (i + 1) * 4 + j))
    k = (len(rings) - 1) * 4
    faces += [(3, 2, 1, 0), (k, k + 1, k + 2, k + 3)]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    uv_box(o)
    return o


def build(p, rng):
    w, H, D = p.width, p.height, p.depth
    a = w / 2
    t = min(p.ring, 0.3 + 0.2 * w)
    jw = p.jamb
    orders = int(p.orders)
    sd = int(p.seed)
    age, moss = p.age, p.moss
    free = p.mount == "ground"                           # free-standing: the bonding tails are exposed
    GRID = clamp(0.118 * (w + 2 * H) / 9.0 * (1.1 if D > 0.85 else 1.0), 0.105, 0.2)   # vertex pitch of the dressed faces

    # ------------------------------------------------------------------ arch geometry (right half)
    if p.arch == "round":
        rise = a
    elif p.arch == "segmental":
        rise = w * lerp(0.12, 0.34, p.rise)
    else:
        rise = w * lerp(0.62, 1.0, p.rise)
    rise = max(min(rise, H - 1.0), 0.08 * w)
    zs = H - rise                                        # springing height
    if rise <= a * 1.02:                                 # one centre
        rise = min(rise, a)
        R = (a * a + rise * rise) / (2 * rise)
        Cx, Cz = 0.0, zs + rise - R
        a_s, a_c = asin(clamp((R - rise) / R)), pi / 2
    else:                                                # two centres (pointed)
        R = (a * a + rise * rise) / (2 * a)
        Cx, Cz = a - R, zs
        a_s, a_c = 0.0, acos(clamp((R - a) / R))
    skew = a_s > 0.05
    pointed = Cx < -1e-4
    kw = clamp(0.7 * t, 0.22, 0.42) * p.keystone * (0.6 if pointed else 1.0)
    dk = min(kw / 2 / R, 0.25 * (a_c - a_s))             # keystone half angle on the intrados
    a_k = a_c - dk
    hood_h = HOOD_H if p.hood else 0.0
    kp = 0.09 if p.hood else 0.055                       # keystone projection
    # keystone side joint: radial on one-centred arches; on a pointed arch it is turned towards the vertical so the
    # apex stone stays compact instead of fanning out into an arrowhead
    th_k = a_k + (pi / 2 - a_k) * (0.62 if pointed else 0.0)
    qx, qz, kdx, kdz = R * cos(a_k), R * sin(a_k), cos(th_k), sin(th_k)
    qd = qx * kdx + qz * kdz

    def key_pt(r):                                       # where the side joint crosses radius r
        lam = -qd + sqrt(max(qd * qd + r * r - R * R, 0.0))
        return (Cx + qx + lam * kdx, Cz + qz + lam * kdz)

    def key_ang(r):
        x, z = key_pt(r)
        return atan2(z - Cz, x - Cx)

    key_end = lambda r: key_ang(r) - JOINT / r           # last station of a ring stone abutting the keystone
    apex_ang = lambda r: acos(clamp(-Cx / r))            # angle at which radius r reaches the centre line
    if pointed:                                          # apex stone: ring depth + 20 %, the hood mitres over it
        rho_top = max(hypot(H + 1.2 * t - Cz, Cx), R + t + 0.03)
    else:
        rho_top = R + t + hood_h + 0.05
    z_apex = Cz + sqrt(max(rho_top ** 2 - Cx ** 2, 0.0))
    ztop = max(z_apex, Cz + sqrt(max((R + t + hood_h) ** 2 - Cx ** 2, 0.0)))

    # ------------------------------------------------------------------ sections
    t_in = clamp(0.4 * t, 0.12, 0.2)                     # inner order width
    s_y = min(0.10, D * 0.25)                            # inner order recess
    arr = None if p.arris == "plain" else p.arris
    arr_size = 0.04 if arr == "chamfer" else 0.03
    arr_eff = 0.04 if arr == "chamfer" else (0.039 if arr == "roll" else 0.0)

    def section(L, v0, v1, front=True, back=True):
        C = [((0.0, v1), "chamfer" if back else None, 0.02)]
        if front and orders == 2:
            C += [((0.0, v0 + s_y), arr, arr_size), ((t_in, v0 + s_y), None, 0),
                  ((t_in, v0), "chamfer" if arr else None, 0.022)]
        elif front:
            C += [((0.0, v0), arr, arr_size)]
        else:
            C += [((0.0, v0), None, 0)]
        C += [((L, v0), None, 0), ((L, v1), None, 0)]
        return _poly(C)

    def section_core(L):
        e = CORE_E
        cs = max(arr_eff - 0.0035, 0.0)
        C = [((e, D - e), "chamfer", 0.0165)]
        if orders == 2:
            C += [((e, s_y + e), "chamfer" if cs > 0 else None, cs), ((t_in + e, s_y + e), None, 0),
                  ((t_in + e, e), "chamfer" if arr else None, 0.0185)]
        else:
            C += [((e, e), "chamfer" if cs > 0 else None, cs)]
        C += [((L, e), None, 0), ((L, D - e), None, 0)]
        return _poly(C)

    def stepper(L, v1, fine=True):
        def f(pp, qq):
            if pp[0] > L - 1e-4 and qq[0] > L - 1e-4:
                return 0.34                              # tail / extrados
            if pp[1] > v1 - 1e-4 and qq[1] > v1 - 1e-4:
                return 0.2 if fine else 0.32             # back face
            if pp[0] < 0.045 and qq[0] < 0.045:
                return max(GRID * 1.3, v1 / 4.01) if fine else 0.3   # reveal / soffit
            return GRID if fine else 0.3                 # dressed front
        return f

    def arc_fn(sx, u_off=0.0):
        def fn(u, v, s):
            r = R + u_off + u
            ang = s(r) if callable(s) else s
            return (sx * (Cx + r * cos(ang)), v, Cz + r * sin(ang))
        return fn

    def on_arc(ang, rad, sx=1):
        return (sx * (Cx + rad * cos(ang)), Cz + rad * sin(ang))

    def jamb_fn(sx):
        return lambda u, v, s: (sx * (a + u), v, s)

    def stations(s0, s1, ns):                            # s1 may be a callable r -> angle (non-radial end joint)
        if callable(s1):
            return [(lambda r, f=j / ns: s0 + (s1(r) - s0) * f) for j in range(ns + 1)]
        return [s0 + (s1 - s0) * j / ns for j in range(ns + 1)]

    # ------------------------------------------------------------------ materials
    ring_m = [mat("stone_smooth", _shade(p.stone, 1.0), rough=1.5, scale=5.0),
              mat("stone_smooth", _shade(p.stone, 0.87, 0.03), rough=1.5, scale=5.0)]
    fine = mat("stone_smooth", _shade(p.stone, 1.12, 0.01), rough=1.4, scale=6.0)       # impost, keystone, hood
    quoin_m = mat("stone_rough", _shade(p.stone, 0.98, 0.015), scale=2.4)               # punched jamb quoins, soffit lining
    hard = mat("stone_rough", _shade(p.stone, 0.9, -0.02), scale=1.3, grime=0.65)                             # plinth, guards, threshold, broken tails
    mortar = mat("mortar", "#e9dfc8", grime=0.7)
    nA = noise_mask(1.0, seed=sd + 1, lo=0.3, hi=0.75)
    nB = noise_mask(0.4, seed=sd + 3, lo=0.3, hi=0.7)
    nC = noise_mask(0.13, seed=sd + 5, lo=0.3, hi=0.65)
    nD = noise_mask(0.3, seed=sd + 9, lo=0.4, hi=0.72)
    nS = noise_mask(1.0, seed=sd + 11, lo=0.36, hi=0.7)
    nE = noise_mask(0.3, seed=sd + 21, lo=0.56, hi=0.78)   # where the pointing has weathered out
    stones = []                                          # (object, kind, per-stone grime bias)
    sites = []                                           # moss sites: (p0, p1, normal, weight)
    runs = []                                            # rust / dirt runs: (x, on_back, z, length, width)

    def site(p0, p1, nrm, wt):
        p0, p1, nrm = np.array(p0, float), np.array(p1, float), np.array(nrm, float)
        sites.append((p0, p1, nrm / np.linalg.norm(nrm), wt * max(float(np.linalg.norm(p1 - p0)), 0.05)))

    def irregular(r, dep, k=1.0):
        """No stone is set dead true: its face stands 1-5 mm proud or shy and is slightly out of plumb / square."""
        yaw, pit, dy = r.jitter(0.0, 0.007 * k), r.jitter(0.0, 0.007 * k), r.u(-0.0055, 0.001) * k

        def post(V):
            c = V.mean(axis=0)
            wgt = np.clip(1.0 - (V[:, 1] - V[:, 1].min()) / max(dep, 0.2), 0.0, 1.0)     # the back bed stays put
            V[:, 1] += wgt * (dy + yaw * (V[:, 0] - c[0]) + pit * (V[:, 2] - c[2]))
            return V
        return post

    def chip(o, pos, size, r):
        """Knock a faceted spall out of an arris."""
        Rm = np.array(Euler((r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28))).to_matrix())
        ext = (size * r.u(1.0, 1.7), size * r.u(0.7, 1.1), size * r.u(0.5, 0.9))
        pts = []
        for ax in range(3):
            for sg in (1.0, -1.0):
                v = np.zeros(3)
                v[ax] = sg * ext[ax] * r.u(0.75, 1.25)
                pts.append(tuple(float(c) for c in np.array(pos, float) + Rm @ v))
        c = mesh_object("chip", pts, [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)])
        recalc_normals(c)
        boolean(o, c)

    def crack(o, xa, xb, z0, z1, yf, r):
        """Settlement crack: a ragged V-groove running through the stone face from bed to bed."""
        n = 6
        zz = [z0 - 0.03 + (z1 - z0 + 0.06) * i / (n - 1) for i in range(n)]
        xx = [lerp(xa, xb, i / (n - 1)) + (r.jitter(0.0, 0.022) if 0 < i < n - 1 else 0.0) for i in range(n)]
        hw = 0.0035
        V, F = [], []
        for x, z in zip(xx, zz):
            V += [(x - hw, yf - 0.02, z), (x + hw, yf - 0.02, z), (x, yf + 0.034, z)]
        for i in range(n - 1):
            k, m = 3 * i, 3 * i + 3
            F += [(k, m, m + 2, k + 2), (m + 1, k + 1, k + 2, m + 2), (m, k, k + 1, m + 1)]
        F += [(0, 2, 1), (3 * n - 3, 3 * n - 2, 3 * n - 1)]
        c = mesh_object("crack", V, F)
        recalc_normals(c)
        boolean(o, c)

    def finish(o, m, r, zmid, frame=None, bev=True, bias=None, kind="stone", crisp=False, rot90=False):
        low = 0.0 if crisp else (1.0 - smoothstep(0.55, 1.35, zmid)) * (0.35 + 0.65 * age)   # worn towards the ground
        displace_noise(o, strength=0.0011 + 0.0014 * low, scale=0.1, seed=sd + 3 * len(stones))
        if frame is not None:
            _localize(o, *frame)
        if rot90:
            uv_box(o, rot90=True)
        if bev:
            bevel(o, 0.003 if crisp else 0.0045 + 0.0075 * low, segments=1, angle=50)
        smooth(o, 50)
        if not o.data.materials:
            assign(o, m)
        stones.append((o, kind, r.jitter(0.0, 0.12) if bias is None else bias))
        return o

    split = D > 0.85                                     # two stones through the wall
    hi = clamp(0.055 * H, 0.14, 0.2)                     # impost height
    hs = 0.14 if skew else 0.0                           # springer bed below the springing
    hp = clamp(0.11 * H, 0.28, 0.42)                     # plinth height
    z_imp = zs - hs - hi
    L_long, L_short = jw * 1.14, jw * 0.86
    um = (t + hood_h + 0.01) * cos(a_s)                  # reach of the first radial joint
    L_skew = max(L_long, um + 0.03)                      # springer stands flush with the jamb line
    L_imp = max(L_long + 0.03, L_skew + 0.03 if skew else t + hood_h + 0.06)
    x1 = a + L_long + 0.04                               # plinth end
    wob = lambda o: displace_noise(o, strength=0.0032, scale=0.55, seed=sd + 2, along_normal=False)   # coherent drift
    r_all = rng.fork("flaws")
    flaws = []                                           # candidate quoins for a crack / a replaced stone

    for sx in (1, -1):
        r = rng.fork("side%d" % sx)
        rc = rng.fork("chips%d" % sx)
        fnj = jamb_fn(sx)
        # ---- plinth block with a splayed (weathered) top
        o_ = 0.035
        pl = _loft("plinth", [(sx * a, sx * (x1 + o_), -o_, D + o_, 0.0), (sx * a, sx * (x1 + o_), -o_, D + o_, hp - 0.055),
                              (sx * a, sx * x1, 0.0, D, hp)])
        densify(pl, max(0.17, D / 4.5))
        for _ in range(int(round(1 + 2.5 * age))):
            if rc.chance(0.5):
                chip(pl, (sx * (a + rc.u(0.0, x1 - a)), -o_, hp - 0.055 - rc.u(0.0, 0.03)), rc.u(0.03, 0.06), rc)
            else:
                chip(pl, (sx * (a + rc.pick([0.0, x1 + o_ - a])), -o_, rc.u(0.04, hp - 0.08)), rc.u(0.035, 0.065), rc)
        wob(pl)
        finish(pl, hard, r, 0.2)
        site((sx * (a + 0.02), -o_ - 0.004, 0.0), (sx * (x1 + o_), -o_ - 0.004, 0.0), (0, -0.75, 0.66), 3.0)
        site((sx * (x1 + o_ + 0.004), -o_, 0.0), (sx * (x1 + o_ + 0.004), D + o_, 0.0), (sx * 0.75, 0, 0.66), 1.2)
        site((sx * (a + 0.02), D + o_ + 0.004, 0.0), (sx * (x1 + o_), D + o_ + 0.004, 0.0), (0, 0.75, 0.66), 1.5)
        site((sx * (a - 0.004), 0.0, 0.04 if p.threshold else 0.0), (sx * (a - 0.004), D, 0.04 if p.threshold else 0.0),
             (-sx * 0.7, 0, 0.7), 2.0)
        site((sx * (a + 0.03), -0.016, hp - 0.027), (sx * x1, -0.016, hp - 0.027), (0, -0.55, 0.83), 3.2)
        # ---- jamb quoins, long and short alternately, punched faces
        n = max(1, int(round((z_imp - hp) / 0.34)))
        hts = [r.vary(1.0, 0.22) for _ in range(n)]
        k = (z_imp - hp) / sum(hts)
        z = hp
        for i in range(n):
            z1 = z + hts[i] * k
            zm = (z + z1) / 2
            L = (L_long if i % 2 == 0 else L_short) * r.vary(1.0, 0.06)
            ss = stations(z + JOINT, z1 - JOINT, max(1, int(ceil((z1 - z) / (GRID * 1.05)))))
            dm = D * (0.42 if i % 2 == 0 else 0.58) if split else D
            q = _block("quoin", _sub(section(L, 0, dm - JOINT if split else D, True, not split), stepper(L, dm - JOINT if split else D)),
                       ss, fnj, irregular(r, min(dm, 0.5)))
            assign(q, quoin_m)
            if free:                                     # bonding tail left hammer-broken where the wall would tooth in
                assign_faces(q, hard, lambda c, nn, L=L: abs(nn.x) > 0.9 and abs(c.x) > a + L - 0.03)
            xo = a + (t_in if orders == 2 else 0.0)      # the front arris of the outer order
            if zm < 1.5 and rc.chance(0.25 + 0.6 * age):                    # knocks from carts and loads low down
                zc_ = rc.pick([z + 0.01, z1 - 0.01, rc.u(z, z1)])
                if orders == 2 and rc.chance(0.5):
                    chip(q, (sx * (a + 0.012), s_y + 0.012, zc_), rc.u(0.028, 0.05), rc)
                else:
                    chip(q, (sx * (xo + 0.01), 0.01, zc_), rc.u(0.028, 0.055), rc)
            if rc.chance(0.12 + 0.3 * age):                                 # a corner off the tail
                chip(q, (sx * (a + L), 0.0, rc.pick([z, z1])), rc.u(0.03, 0.06), rc)
            wob(q)
            rec = {"o": q, "sx": sx, "x0": xo + 0.05, "x1": a + L - 0.06, "z0": z, "z1": z1, "zm": zm, "r": r, "done": False}
            flaws.append(rec)
            if split:
                Lb = L * r.vary(1.0, 0.05)
                qb = _block("quoin_b", _sub(section(Lb, dm + JOINT, D, False, True), stepper(Lb, D, False)),
                            stations(z + JOINT, z1 - JOINT, max(1, int(ceil((z1 - z) / 0.22)))), fnj)
                wob(qb)
                finish(qb, quoin_m, r, zm, rot90=r.chance(0.4))
            if 0 < i and z < 2.1:
                site((sx * (xo + 0.03), -0.003, z), (sx * (a + L_short * 0.92), -0.003, z), (0, -1, 0), 1.7 * (1.0 - z / 2.1) ** 1.5)
            z = z1
        # ---- moulded impost
        f = hi / 0.18
        prof = [(0.0, 0.0), (0.0, 0.028), (0.012, 0.032), (0.012, 0.048), (0.018, 0.068), (0.032, 0.09), (0.05, 0.106),
                (0.056, 0.112), (0.056, 0.168), (0.046, 0.18)]
        imp = _loft("impost", [(sx * a, sx * (a + L_imp + po), -po, D + po, z_imp + pz * f) for po, pz in prof])
        densify(imp, max(0.19, D / 4.0))
        for _ in range(int(round(0.6 + 2.4 * age))):     # the nosing is always the first thing to get knocked
            if rc.chance(0.4):
                chip(imp, (sx * a, -0.056, z_imp + rc.u(0.12, 0.17) * f), rc.u(0.022, 0.04), rc)
            else:
                chip(imp, (sx * (a + rc.u(0.06, L_imp + 0.05)), -0.058, z_imp + rc.pick([0.112, 0.172]) * f), rc.u(0.02, 0.04), rc)
        wob(imp)
        finish(imp, fine, r, z_imp, bev=False)
        site((sx * (a + 0.05), -0.014, z_imp + hi), (sx * (a + L_imp), -0.014, z_imp + hi), (0, -0.5, 0.87), 2.2)
        # ---- springer (segmental arches): bed cut to the first radial joint, weathered shoulder falling to the jamb line
        if skew:
            zt = zs + um * tan(a_s)
            z_out = max(zt - (L_skew - um) * clamp(1.0 / tan(a_s), 0.6, 1.8), zs - hs + 0.1)   # the shoulder carries on the line of the extrados
            sp = [(a, zs - hs + JOINT), (a + L_skew, zs - hs + JOINT), (a + L_skew, z_out), (a + um, zt), (a, zs)]
            sp = [(sx * x, z_) for x, z_ in (sp if sx > 0 else reversed(sp))]
            sk = extrude_profile("springer", sp, D, loc=(0, D, 0), rot=(90, 0, 0))
            densify(sk, 0.16)
            uv_box(sk)
            finish(sk, ring_m[0], r, zs)
            site((sx * (a + um + 0.02), -0.012, zt - 0.02), (sx * (a + L_skew), -0.012, z_out), (0, -0.5, 0.87), 1.5)
        # ---- label stop where the hood lands
        if p.hood:
            if skew:                                     # a stop block worked on the springer; its top bed is the skew joint
                r1, r2 = R + t - 0.02, R + t + HOOD_H + 0.014
                P1, P2 = on_arc(a_s, r1), on_arc(a_s, r2)
                zb_ = P1[1] - 0.13
                lp = [(P1[0], zb_), (P2[0], zb_), P2, P1]
                lp = [(sx * x, z_) for x, z_ in (lp if sx > 0 else reversed(lp))]
                ls = extrude_profile("label_stop", lp, HOOD_P + 0.03, loc=(0, 0.02, 0), rot=(90, 0, 0), bevel=0.008)
                uv_box(ls)
                lx, lz, ly = sx * (P1[0] + P2[0]) / 2, (zb_ + (P1[1] + P2[1]) / 2) / 2, -HOOD_P - 0.01
                hb_ = min(0.04, 0.38 * (P2[0] - P1[0]))
            else:
                lx, lz = on_arc(a_s + 0.085 / (R + t), R + t + HOOD_H * 0.5, sx)
                ls = box("label_stop", (0.15, 0.2, 0.17), loc=(lx, -0.01, lz), bevel=0.012, segments=1)
                ly, hb_ = -0.11, 0.05
            finish(ls, fine, r, lz, bev=False)
            pv = [(lx + dx, ly, lz + dz) for dx, dz in ((hb_, 0), (0, hb_), (-hb_, 0), (0, -hb_))] + [(lx, ly - 0.028, lz)]
            boss = mesh_object("label_boss", pv, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (3, 2, 1, 0)])
            recalc_normals(boss)
            uv_box(boss)
            finish(boss, fine, r, lz, bev=False)
        # ---- voussoirs
        n = max(2, int(round((R + 0.5 * t) * (a_k - a_s) / 0.33)))
        step = (a_k - a_s) / n
        g = JOINT / (R + t / 2)
        fn = arc_fn(sx)

        def ring(bnds, v0, v1, front, back, name):
            for i in range(len(bnds) - 1):
                L = t if p.hood else t * r.vary(1.0, 0.05)
                s0, e1 = bnds[i] + g, bnds[i + 1]
                last = callable(e1)
                if not last:
                    e1 -= g
                s1m = e1(R + t / 2) if last else e1
                ns = max(2 if front else 1, int(ceil((R + t) * (s1m - s0) / (GRID * 1.1 if front else 0.3))))
                o = _block(name, _sub(section(L, v0, v1, front, back), stepper(L, v1, front)), stations(s0, e1, ns), fn,
                           irregular(r, min(v1 - v0, 0.5), 0.8) if front else None)
                sm = (s0 + s1m) / 2
                cx, cz = on_arc(sm, R + t / 2, sx)
                if front and cz < 1.6 and rc.chance(0.2 + 0.4 * age):
                    chip(o, (cx - sx * 0.5 * t * cos(sm) + sx * 0.01, (s_y if orders == 2 else 0.0) + 0.01, cz - 0.5 * t * sin(sm)),
                         rc.u(0.025, 0.045), rc)
                wob(o)
                finish(o, r.pick([ring_m[0], ring_m[0], ring_m[1]]) if front else quoin_m, r, cz,
                       frame=((cx, (v0 + v1) / 2, cz), degrees(atan2(sx * cos(sm), sin(sm)))))

        b0 = [a_s] + [a_s + step * (i + r.jitter(0.0, 0.09)) for i in range(1, n)] + [key_end]
        if split:
            dm = D * r.u(0.46, 0.54)
            ring(b0, 0, dm - JOINT, True, False, "voussoir")
            ring([a_s] + [a_s + step * (i + 0.5) for i in range(n - 1)] + [key_end], dm + JOINT, D, False, True, "voussoir_b")
        else:
            ring(b0, 0, D, True, True, "voussoir")
        # ---- hood mould in short lengths, dying into the keystone (mitred over the apex stone of a pointed arch)
        if p.hood:
            e = min(0.1, D * 0.3)
            hp_prof = [(0.0, e), (0.0, -0.012), (0.016, -0.012), (0.022, -0.03), (0.04, -0.055), (0.056, -HOOD_P),
                       (0.084, -HOOD_P), (HOOD_H - 0.012, -0.022), (HOOD_H, 0.0), (HOOD_H, e)]
            h_end = (lambda rr: apex_ang(rr) - 0.0015 / rr) if pointed else key_end
            a_e = h_end(R + t)
            nh = max(1, int(round((R + t) * (a_e - a_s) / 0.62)))
            gh = JOINT * 0.7 / (R + t)
            hb = [a_s + (a_e - a_s) * (i + (r.jitter(0.0, 0.12) if 0 < i < nh else 0.0)) / nh for i in range(nh + 1)]
            fh = arc_fn(sx, t)
            for i in range(nh):
                s0 = hb[i] + gh
                e1 = h_end if i == nh - 1 else hb[i + 1] - gh
                s1m = e1(R + t) if callable(e1) else e1
                hd = _block("hood", hp_prof, stations(s0, e1, max(2, int(ceil((s1m - s0) / radians(7))))), fh)
                sm = (s0 + s1m) / 2
                cx, cz = on_arc(sm, R + t + HOOD_H / 2, sx)
                if rc.chance(0.25 + 0.35 * age):         # a bite out of the drip edge
                    bx, bz = on_arc(rc.u(s0, s1m), R + t + 0.06, sx)
                    chip(hd, (bx, -HOOD_P, bz), rc.u(0.016, 0.03), rc)
                wob(hd)
                finish(hd, fine, r, cz, frame=((cx, 0.0, cz), degrees(atan2(sx * cos(sm), sin(sm)))), bev=False)
            for j in range(3):
                ang = a_s + (a_e - a_s) * (0.25 + 0.22 * j)
                (xa_, za_), (xb_, zb_) = on_arc(ang, R + t + HOOD_H + 0.002, sx), on_arc(ang + 0.2, R + t + HOOD_H + 0.002, sx)
                site((xa_, -0.02, za_), (xb_, -0.02, zb_), (sx * cos(ang + 0.1), -0.25, sin(ang + 0.1)), 0.8)
        # ---- recessed mortar bed behind the joints; here and there the pointing has weathered out
        ero = lambda q: float(nE(np.array([q], dtype=float), None)[0])

        def core(name, L, ss, f0):
            prof = section_core(L)
            push = []
            for (u, v) in prof:
                if u <= CORE_E + 0.045:
                    push.append((1.0, 0.0))
                elif v >= D - CORE_E - 0.03:
                    push.append((0.0, -1.0))
                elif v <= CORE_E + 0.03 and u >= t_in:
                    push.append((0.0, 1.0))
                else:
                    push.append((0.7, 0.7))
            dirs = dict(zip(prof, push))

            def fe(u, v, s):
                e_ = 0.015 * ero(f0(u, v, s))
                du, dv = dirs[(u, v)]
                return f0(u + du * e_, v + dv * e_, s)
            o = _block(name, prof, ss, fe)
            wob(o)
            smooth(o, 50)
            assign(o, mortar)
            stones.append((o, "core", 0.0))

        Lj = L_short * 0.92 - 0.012
        zc1 = zs + (0.01 if not skew else 0.0)
        core("core_jamb", Lj, stations(0.02, zc1, max(3, int(ceil(zc1 / 0.19)))), fnj)
        core("core_arch", t * 0.95 - 0.012, stations(a_s, a_c, max(4, int(ceil((R + t / 2) * (a_c - a_s) / 0.19)))), fn)

    # ---- one or two stones have cracked, one was replaced with fresh stone (age and seed decide)
    cand = [q for q in flaws if q["x1"] - q["x0"] > 0.12]
    for _ in range((1 if age > 0.3 else 0) + (1 if age > 0.72 else 0)):
        pool = [q for q in cand if not q["done"] and q["zm"] < 0.62 * z_imp + 0.4]
        if pool:
            q = r_all.pick(pool)
            q["done"] = True
            xa = r_all.u(q["x0"], q["x1"])
            crack(q["o"], q["sx"] * xa, q["sx"] * clamp(xa + r_all.jitter(0.0, 0.07), q["x0"], q["x1"]), q["z0"], q["z1"], 0.0, r_all)
    fresh = None
    if age > 0.4:
        pool = [q for q in flaws if not q["done"] and q["zm"] < 1.7]
        if pool:
            fresh = r_all.pick(pool)
            fresh["done"] = True
            assign(fresh["o"], fine)
    for q in flaws:
        if q is fresh:
            finish(q["o"], fine, q["r"], q["zm"], crisp=True, bias=-0.5, kind="fresh")
        else:
            finish(q["o"], quoin_m, q["r"], q["zm"], rot90=q["r"].chance(0.35))

    # ------------------------------------------------------------------ keystone (through stone, front projecting)
    r = rng.fork("key")
    nb = 3
    right = [on_arc(a_k + (a_c - a_k) * j / nb, R) for j in range(nb + 1)]          # joint -> crown along the intrados
    K1 = key_pt(rho_top)
    angK, angA = key_ang(rho_top), apex_ang(rho_top)
    topr = [on_arc(angK + (angA - angK) * j / 2, rho_top) for j in (1, 2)]
    kpts = [right[0], K1, topr[0], (0.0, topr[1][1]), (-topr[0][0], topr[0][1]), (-K1[0], K1[1])] + \
        [(-x, z) for x, z in right[:-1]] + list(reversed(right[1:]))
    if split:
        dmk = D * 0.5
        key = extrude_profile("keystone", kpts, dmk - JOINT + kp, loc=(0, dmk - JOINT, 0), rot=(90, 0, 0))
        keyb = extrude_profile("keystone_b", kpts, D - dmk - JOINT, loc=(0, D, 0), rot=(90, 0, 0))
        uv_box(keyb)
        finish(keyb, fine, r, H)
    else:
        key = extrude_profile("keystone", kpts, D + kp, loc=(0, D, 0), rot=(90, 0, 0))
    densify(key, 0.16)
    uv_box(key)

    # ---- emblem: bold relief standing in a sunk panel that takes ~60 % of the keystone face
    txt = str(p.inscription).strip()
    cot = kdx / max(kdz, 1e-3)
    x_side = lambda z: right[0][0] + (z - right[0][1]) * cot                  # half width of the keystone face at z
    z_b = H + (0.135 if txt else 0.05)
    z_t = min(K1[1], topr[0][1]) - 0.032
    mrg = 0.03
    xb, xt = x_side(z_b) - mrg, x_side(z_t) - mrg
    y0 = -kp
    yf = y0 + PANEL                                      # panel floor
    panel = p.emblem != "none" and min(xb, xt) > 0.045 and z_t - z_b > 0.1
    emb = []
    if panel:
        cut = extrude_profile("panel_cut", [(-xb, z_b), (xb, z_b), (xt, z_t), (-xt, z_t)], PANEL + 0.06, loc=(0, yf, 0), rot=(90, 0, 0))
        boolean(key, cut)
        zc = (z_b + z_t) / 2
        aw, ah = 2 * min(xb, xt) * 0.9, (z_t - z_b) * 0.86
        dep = PANEL + 0.004                              # the relief stands a hair proud of the keystone face

        def ex(name, pts, d=dep, y=yf, bv=0.003):
            return extrude_profile(name, pts, d, loc=(0, y, zc), rot=(90, 0, 0), bevel=bv, segments=1)

        if p.emblem == "crown":
            S = min(aw / 1.16, ah / 1.0)
            pts = [(-0.5, 0), (0.5, 0), (0.5, 0.36), (0.58, 0.9), (0.29, 0.56), (0, 1.0), (-0.29, 0.56), (-0.58, 0.9), (-0.5, 0.36)]
            emb.append(ex("emblem", [(x * S, (y - 0.5) * S) for x, y in pts]))
            emb.append(box("emblem_band", (1.04 * S, 0.012, 0.15 * S), loc=(0, yf - dep - 0.004, zc - 0.36 * S), bevel=0.003, segments=1))
            for x, y in ((0.58, 0.9), (0, 1.0), (-0.58, 0.9)):
                emb.append(sphere("emblem_pearl", 0.075 * S, loc=(x * S * 0.97, yf - dep + 0.004, zc + (y - 0.5) * S * 0.97), segments=8, rings=5))
            for x in (-0.3, 0.0, 0.3):
                emb.append(sphere("emblem_jewel", 0.055 * S, loc=(x * S, yf - dep - 0.008, zc - 0.36 * S), segments=6, rings=4))
        elif p.emblem == "shield":
            S = min(aw / 0.92, ah / 1.1)
            pts = [(-0.46, 1.0), (-0.46, 0.45), (-0.36, 0.2), (-0.19, 0.06), (0, 0), (0.19, 0.06), (0.36, 0.2), (0.46, 0.45), (0.46, 1.0)]
            emb.append(ex("emblem", [(x * S, (y - 0.5) * 1.1 * S) for x, y in pts], d=dep - 0.006))
            chev = [(-0.4, 0.5), (0, 0.24), (0.4, 0.5), (0.4, 0.74), (0, 0.48), (-0.4, 0.74)]
            emb.append(ex("emblem_chevron", [(x * S, (y - 0.5) * 1.1 * S) for x, y in chev], d=0.012, y=yf - dep + 0.007, bv=0.002))
            emb.append(sphere("emblem_boss", 0.085 * S, loc=(0, yf - dep + 0.004, zc + 0.36 * 1.1 * S), segments=8, rings=5))
        elif p.emblem == "star":                         # faceted five-point star: every ray is a ridge that catches light
            S = min(aw, ah)
            ro, ri = 0.5 * S, 0.2 * S
            ring2 = [((ro if k % 2 == 0 else ri) * sin(radians(36 * k)), (ro if k % 2 == 0 else ri) * cos(radians(36 * k))) for k in range(10)]
            V = [(x, yf + 0.002, zc + z) for x, z in ring2] + [(x, yf - 0.009, zc + z) for x, z in ring2] + [(0.0, yf - dep - 0.012, zc)]
            F = [(k, (k + 1) % 10, 10 + (k + 1) % 10, 10 + k) for k in range(10)] + [(10 + k, 10 + (k + 1) % 10, 20) for k in range(10)]
            st_ = mesh_object("emblem", V, F)
            recalc_normals(st_)
            uv_box(st_)
            emb.append(st_)
            emb.append(sphere("emblem_boss", 0.075 * S, loc=(0, yf - dep - 0.008, zc), segments=8, rings=5))
        elif p.emblem == "rosette":
            S = min(aw, ah)
            pts = [(0.5 * S * (0.6 + 0.4 * abs(cos(radians(3 * 9 * k)))) * cos(radians(9 * k)),
                    0.5 * S * (0.6 + 0.4 * abs(cos(radians(3 * 9 * k)))) * sin(radians(9 * k))) for k in range(40)]
            emb.append(ex("emblem", pts, d=dep - 0.006))
            emb.append(cylinder("emblem_ring", 0.24 * S, 0.012, loc=(0, yf - dep + 0.002, zc), rot=(90, 0, 0), verts=14, bevel=0.003, segments=1))
            emb.append(sphere("emblem_boss", 0.14 * S, loc=(0, yf - dep + 0.004, zc), segments=8, rings=5))
    bevel(key, 0.008, segments=1, angle=40)
    smooth(key, 40)
    assign(key, fine)
    stones.append((key, "key", 0.0))
    soot = lambda co, n: np.clip((co[:, 1] - (y0 + 0.003)) / (PANEL - 0.004), 0, 1) ** 0.8 * (0.5 + 0.35 * age)
    if txt:
        zd = H + 0.07
        size = clamp(1.75 * x_side(zd) / (0.62 * len(txt)), 0.03, 0.085)
        tx = text("inscription", txt, size=size, depth=0.005, loc=(0, y0, zd - size * 0.45), font="roman")
        assign(tx, mat("lead"))                          # letters cut in and run with lead
        paint(tx, grime=0.3 + 0.3 * age)
    if emb:
        if p.emblem_finish == "carved":
            m = fine
        else:
            m = mat("bronze_aged") if p.emblem_finish == "bronze" else mat("metal_plain", "#c9a04a", rough=0.62)
        for o in emb:
            assign(o, m)
            smooth(o, 42)
        paint(emb, grime=lambda co, n: 0.75 * soot(co, n) ** 2.5, special=(0.3 + 0.4 * moss) if p.emblem_finish != "carved" else None)

    # ------------------------------------------------------------------ wheel guards: dressed octagonal spur stones
    guards = []
    if p.guards:
        gs = clamp(w / 2.4, 0.86, 1.1)
        ybk = (s_y if orders == 2 else 0.0) + 0.015      # the back runs into the jamb corner and the plinth
        for sx in (1, -1):
            r = rng.fork("guard%d" % sx)
            k = gs * r.vary(1.0, 0.05)
            prof = [(0.0, 0.0), (0.215, 0.0), (0.215, 0.075), (0.2, 0.1), (0.186, 0.108)] + \
                   [(lerp(0.186, 0.128, j / 6), lerp(0.108, 0.47, j / 6)) for j in range(1, 7)] + \
                   [(0.106, 0.52), (0.062, 0.556), (0.0, 0.568)]
            gx, gy = sx * (a + 0.105 * k), -0.06 * k
            gd = lathe("wheel_guard", [(x * k, z * k) for x, z in prof], segments=8, loc=(gx, gy, 0.0), rot=(0, 0, 22.5), smooth_angle=30)
            apply_transform(gd)
            boolean(gd, box("guard_cut", (0.8, 0.8, 1.6), loc=(sx * (a + 0.4), ybk - 0.4, 0.5)), op='INTERSECT')
            tdir = np.array([-sx * 0.55, -0.83])         # where the tyres and hubs come from

            def scrape(co, gx=gx, gy=gy, k=k, tdir=tdir):
                h = co.z / k
                d = np.array([co.x - gx, co.y - gy])
                l = float(np.linalg.norm(d))
                if l < 1e-4 or not (0.16 < h < 0.44):
                    return co
                fc = max(0.0, float(d @ tdir) / l)
                push = 0.014 * k * (1.0 - abs(h - 0.3) / 0.14) * fc * fc
                return Vector((co.x - d[0] / l * push, co.y - d[1] / l * push, co.z))
            transform_verts(gd, scrape)
            displace_noise(gd, strength=0.003 * k, scale=0.09, seed=sd + sx + 7)
            transform_verts(gd, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
            bevel(gd, 0.007, segments=1, angle=30)
            smooth(gd, 30)
            uv_box(gd)
            assign(gd, hard)
            guards.append((gd, gx, gy, k, tdir))
            site((gx - 0.2 * k, gy - 0.215 * k, 0.0), (gx + 0.2 * k, gy - 0.215 * k, 0.0), (0, -0.7, 0.7), 1.6)

    # ------------------------------------------------------------------ threshold slabs, dished by feet and rutted by wheels
    floor_objs = []
    if p.threshold:
        r = rng.fork("sill")
        ns_ = max(1, int(round(w / 0.9)))
        xs = [-a + 0.004] + [-a + w * (i + r.jitter(0.0, 0.12)) / ns_ for i in range(1, ns_)] + [a - 0.004]
        gauge = 0.0 if w < 1.9 else min(1.35, w - 0.75)

        def top(x, y, h):
            d = 0.007 * exp(-(x / (0.28 * w)) ** 2)
            if gauge:
                d += 0.013 * exp(-((abs(x) - gauge / 2) / 0.075) ** 2)
            return h - d

        for i in range(ns_):
            h = 0.04 + r.jitter(0.0, 0.004)
            x0, x1_ = xs[i] + JOINT, xs[i + 1] - JOINT
            sl = _slab("threshold", x0, x1_, -0.05 + r.jitter(0.0, 0.012), D + 0.05 + r.jitter(0.0, 0.012),
                       max(2, int(ceil((x1_ - x0) / 0.13))), max(3, int(ceil(D / 0.2))), lambda x, y, h=h: top(x, y, h))
            displace_noise(sl, strength=0.0015, scale=0.12, seed=sd + i)
            transform_verts(sl, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
            bevel(sl, 0.008, segments=1, angle=40)
            smooth(sl, 40)
            assign(sl, hard)
            floor_objs.append(sl)
            if i > 0:
                site((xs[i], 0.0, h - 0.004), (xs[i], D, h - 0.004), (0, 0, 1), 0.7)

    # ------------------------------------------------------------------ ironwork
    if p.ironwork:
        iron = mat("iron_wrought")
        lead = mat("lead")
        io, pads = [], []
        zr = min(1.15, zs - 0.25)
        xr = a + (t_in if orders == 2 else 0.0) + (jw - (t_in if orders == 2 else 0.0)) * 0.55
        io.append(cylinder("ring_rose", 0.032, 0.006, loc=(xr, -0.003, zr), rot=(90, 0, 0), verts=12))
        io.append(torus("ring_eye", 0.016, 0.006, loc=(xr, -0.012, zr), rot=(0, 90, 0), major_seg=12, minor_seg=6))
        io.append(torus("tether_ring", 0.055, 0.008, loc=(xr, -0.022, zr - 0.05), rot=(84, 0, 0), major_seg=18, minor_seg=6))
        runs.append((xr, False, zr - 0.02, 0.5, 0.045))
        r = rng.fork("iron")
        for sx in (1, -1):                               # hook pintles for a gate leaf, each on a plate run in with lead
            for zp in ([0.5, zs - 0.35] if zs > 1.5 else [zs * 0.5]):
                x = sx * (a + 0.11)
                pd = box("pintle_lead", (0.105, 0.012, 0.15), loc=(x, D + 0.001, zp + 0.01), bevel=0.004, segments=1)
                displace_noise(pd, strength=0.003, scale=0.04, seed=sd + int(zp * 10))
                pads.append(pd)
                io.append(box("pintle_plate", (0.07, 0.008, 0.115), loc=(x, D + 0.01, zp + 0.012), rot=(0, r.jitter(0, 2.5), 0),
                              bevel=0.002, segments=1))
                io.append(sweep("pintle_hook", [(x, D + 0.008, zp), (x, D + 0.07, zp), (x, D + 0.096, zp + 0.02), (x, D + 0.098, zp + 0.115)],
                                profile=0.0135, sides=6, step=0.025))
                io.append(cylinder("pintle_collar", 0.021, 0.012, loc=(x, D + 0.02, zp), rot=(90, 0, 0), verts=8))
                for dz in (-0.03, 0.055):
                    io.append(bolt("pintle_nail", 0.008, 0.005, loc=(x + r.jitter(0, 0.006), D + 0.014, zp + 0.012 + dz), rot=(-90, 0, 0), kind="pyramid"))
                runs.append((x, True, zp - 0.04, r.u(0.35, 0.6), 0.05))
        for o in io:
            assign(o, iron)
            smooth(o, 40)
        for o in pads:
            assign(o, lead)
            smooth(o, 50)
        paint(io, special=lambda co, n: np.clip(0.45 + 0.3 * age + 0.5 * nC(co, n), 0, 1), grime=0.3 * age)
        paint(pads, grime=lambda co, n: 0.2 + 0.5 * age * nC(co, n))

    # ------------------------------------------------------------------ weathering story (painted per stone)
    S3 = np.array([8.0, 2.5, 0.55])                      # stretches nS into vertical run-off streaks
    gA = 0.4 + 1.2 * age

    def grime(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        ax = np.abs(x)
        rho = np.hypot(ax - Cx, z - Cz)
        st = nS(co * S3, n)
        b1, b2, b4 = nA(co, n), nB(co, n), nD(co, n)
        damp = np.clip((0.26 + 0.42 * b1 - z) / 0.24, 0, 1)                      # rising damp / splash zone, ragged top
        di = z_imp - z
        run = (di > -0.03) * np.clip(1.0 - di / (0.1 + 0.9 * st), 0, 1)          # soot wash and tongues below the impost drip
        arch = z > z_imp + hi * 0.5
        shel = 0.0
        if p.hood:                                                               # sooty crust in the lee of the hood mould
            shel = arch * np.clip((rho - (R + t - 0.21)) / 0.18, 0, 1) * (rho < R + t + 0.004) * (y < 0.06)
        inner = np.clip((y - 0.04) / 0.22, 0, 1) * (((ax < a + 0.04) & ~arch) | (arch & (rho < R + 0.04)))
        inner = inner * np.where(arch, 0.8, 0.45)                                # lamp and torch smoke gathers on the soffit
        ledge = np.clip(n[:, 2], 0, 1) * (z > 0.1)
        g = 0.5 * damp + 0.8 * run * (0.5 + 0.5 * b2) + 0.75 * shel * (0.45 + 0.55 * b2) + inner * (0.4 + 0.6 * b2) \
            + 0.4 * ledge + 0.32 * st * b4 + 0.16 * b4
        for rx, bk, rz, rl, rw in runs:                                          # rust runs below the ironwork
            sel = (y > D - 0.05) if bk else (y < 0.05)
            g = g + 0.9 * sel * np.exp(-((x - rx) / rw) ** 2) * np.clip(1.0 - (rz - z) / rl, 0, 1) * (z < rz + 0.03)
        return np.clip(gA * g, 0, 1)

    def wear(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        ax = np.abs(x)
        band = np.exp(-((z - 1.05) / 0.4) ** 2)                                  # shoulders, hips, sacks and barrow loads
        near = np.clip(1.0 - (ax - a) / 0.15, 0, 1) * (ax > a - 0.03) * np.clip(1.0 - y / 0.32, 0, 1)
        return np.clip((0.45 + 0.9 * age) * band * near * (0.4 + 0.6 * nC(co, n)), 0, 1)

    def special(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        b1, b2 = nA(co, n), nB(co, n)
        side = 0.8 + 0.3 * np.tanh(-x / max(a, 0.5)) + 0.25 * (y > D - 0.03)     # the left jamb and the back stay damp longest
        low = np.clip((0.12 + 1.15 * moss + 0.45 * b1 - z) / 0.55, 0, 1) * (0.3 + 0.7 * b2)
        up = np.clip(n[:, 2], 0, 1) ** 1.5 * (0.45 + 0.55 * b1)
        di = z_imp - z
        wet = (di > -0.03) * np.clip(1.0 - di / 0.6, 0, 1) * nS(co * S3 + 7.3, n)  # green tongues where the ledges shed water
        return np.clip(moss * side * (1.3 * low + 1.6 * up + 0.8 * wet + 0.3 * nD(co, n) * b1), 0, 1)

    for o, kind, b in stones:
        if kind == "core":
            paint(o, grime=lambda co, n: np.clip(0.75 * nE(co, n) + 0.3 * grime(co, n), 0, 1),
                  special=lambda co, n: np.clip(1.5 * special(co, n) + 0.25 * moss * nB(co, n), 0, 1))
        elif kind == "fresh":                            # the replaced stone has not caught up yet
            paint(o, grime=lambda co, n: 0.5 * grime(co, n), special=lambda co, n: 0.3 * special(co, n))
        elif kind == "key":
            pan = lambda co, n: soot(co, n) * (co[:, 1] < yf + 0.01) * (co[:, 2] > z_b - 0.01) if panel else 0.0
            paint(o, grime=lambda co, n: np.maximum(grime(co, n), pan(co, n)), wear=wear, special=special)
        else:
            paint(o, grime=lambda co, n, b=b: np.clip(grime(co, n) + b * (0.5 + age), 0, 1), wear=wear, special=special)
    for sl in floor_objs:                                # trodden centre stays clean and polished, edges collect dirt and moss
        cen = lambda co: np.exp(-(co[:, 0] / (0.42 * w)) ** 2)
        paint(sl, grime=lambda co, n: np.clip(gA * (0.3 + 0.22 * nB(co, n) + 0.2 * nC(co, n)) * (1.0 - 0.7 * cen(co)), 0, 1),
              wear=lambda co, n: 0.45 * np.clip(n[:, 2], 0, 1) * cen(co),
              special=lambda co, n: np.clip(1.4 * moss * (1.0 - cen(co)) ** 2 * (0.3 + 0.7 * nB(co, n)), 0, 1))
    for gd, gx, gy, k, tdir in guards:                   # iron tyres leave a dark scuffed band with a polished scrape in it
        def fc(co):
            d = co[:, :2] - np.array([gx, gy])
            return np.clip((d @ tdir) / np.maximum(np.linalg.norm(d, axis=1), 1e-4), 0, 1)
        paint(gd, grime=lambda co, n: np.clip(0.5 * grime(co, n) + (0.4 + 0.35 * age) * np.exp(-((co[:, 2] - 0.3 * k) / (0.15 * k)) ** 2)
                                              * (0.35 + 0.65 * fc(co)), 0, 1),
              wear=lambda co, n: np.clip(0.95 * np.exp(-((co[:, 2] - 0.3 * k) / (0.055 * k)) ** 2) * fc(co) ** 2
                                         + 0.4 * (co[:, 2] > 0.5 * k), 0, 1),
              special=lambda co, n: 0.8 * special(co, n))

    # ------------------------------------------------------------------ moss: low cushions crowding the joints and ledge corners
    nm = int(round(moss ** 1.15 * 170 * clamp(w / 2.4, 0.75, 1.35)))
    if nm > 0 and sites:
        r = rng.fork("moss")
        rb = rng.fork("moss_base")
        V = [(0.0, 0.0, 1.0)]
        for rad, zz in ((0.62, 0.74), (1.0, -0.12)):
            V += [(rad * rb.vary(1.0, 0.22) * cos(radians(45 * k_ + 20)), rad * rb.vary(1.0, 0.22) * sin(radians(45 * k_ + 20)),
                   zz + rb.jitter(0.0, 0.1)) for k_ in range(8)]
        F = [(0, 1 + k_, 1 + (k_ + 1) % 8) for k_ in range(8)] + [(1 + k_, 9 + k_, 9 + (k_ + 1) % 8, 1 + (k_ + 1) % 8) for k_ in range(8)]
        base = mesh_object("moss_base", V, F)
        recalc_normals(base)
        uv_box(base)
        mats = []
        wsum = [s_[3] * (1.25 if s_[0][0] < 0 else 0.8) for s_ in sites]
        while len(mats) < nm:
            p0, p1, nrm, _ = r.weighted(list(zip(sites, wsum)))
            tg = p1 - p0
            ln = float(np.linalg.norm(tg))
            tg = tg / max(ln, 1e-6)
            tg = tg - nrm * float(tg @ nrm)
            tg = tg / max(float(np.linalg.norm(tg)), 1e-6)
            s0 = r.u(0.0, ln)
            for _ in range(r.pick([1, 2, 2, 3, 3, 4, 5])):                       # cushions grow in clusters along a joint
                s_ = clamp(s0 + r.jitter(0.0, 0.09), 0.0, ln)
                bt = np.cross(nrm, tg)
                pos = p0 + tg * s_ + bt * r.jitter(0.0, 0.012)
                yaw = radians(r.jitter(0.0, 28))
                X = tg * cos(yaw) + bt * sin(yaw)
                Y = np.cross(nrm, X)
                lx = r.u(0.018, 0.045) * (0.8 + 0.5 * moss)
                ly = lx * r.u(0.4, 0.8)
                lz = min(lx, ly) * r.u(0.3, 0.55)
                M = Matrix(((X[0] * lx, Y[0] * ly, nrm[0] * lz, pos[0]), (X[1] * lx, Y[1] * ly, nrm[1] * lz, pos[1]),
                            (X[2] * lx, Y[2] * ly, nrm[2] * lz, pos[2]), (0, 0, 0, 1)))
                mats.append(M)
        ms = instances_to_mesh("moss_cushions", base, mats, keep_base=False, seed=sd)
        assign(ms, mat("moss"))
        smooth(ms, 70)
        paint(ms, grime=lambda co, n: 0.15 + 0.35 * nC(co, n))

    # ------------------------------------------------------------------ sockets for the town generator
    zl = clamp(zs - 0.35, 1.7, 2.6)
    socket("lantern_l", "attach", (-(a + jw * 0.6), 0, zl), role="lantern")
    socket("lantern_r", "attach", (a + jw * 0.6, 0, zl), role="lantern")
    socket("sign", "attach", (0, 0, ztop + 0.35), role="sign")
    socket("gate", "attach", (0, D, 0), role="door")
