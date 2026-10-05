"""Yard cluster left beside a doorway or the pump: a coopered oak bucket (separate tapered staves with eased
arrises and darkened joints, two bevelled iron hoops with riveted laps bleeding rust stains down the wood,
ear staves, knotted hemp rope handle, standing water with a meniscus and a wet line), a galvanised pail
(swaged ribs, rolled rim, lapped soldered seam, riveted ears, wire bail with a turned wooden grip - upright or
tipped on its side), a galvanised / painted / copper watering can (soldered half-dome hood with a wired edge,
braced tapering spout with a brass rose, tube carrying handle and strap back handle) and a birch besom
(crooked hazel pole, layered forked twigs under two bindings).  The besom never balances in mid-air: it lies
low with its twig head on the cobbles and the pole across a rim (`rest`), stands head-down in a bucket
(`in_bucket`), lies flat (`ground`) or leans head-up against the wall plane y = 0 (`wall`, use with
mount = wall_ground).  Items, sizes, finishes, poses and the seeded arrangement are parametric.
"""
from tk import *
import numpy as np
from math import sin, cos, radians, degrees, atan2, atan, asin, acos, hypot, pi, log

META = {"name": "Buckets, watering can and broom", "class": "prop_small", "mount": "ground",
        "tri_budget": 8500, "focus": [0.28, -0.05, 0.2]}
PARAMS = {
    "items": {"default": "wood,zinc,can,broom", "desc": "comma list of what is in the cluster: wood (coopered bucket), zinc (pail), can (watering can), broom (besom)"},
    "water": {"default": "wood", "choices": ["wood", "zinc", "both", "none"], "desc": "which bucket holds water"},
    "fill": {"default": 0.94, "min": 0.2, "max": 0.98, "desc": "water level (fraction of the bucket depth)"},
    "bucket_size": {"default": 0.30, "min": 0.24, "max": 0.38, "desc": "height of the wooden bucket (m)"},
    "staves": {"default": 14, "min": 10, "max": 16, "desc": "number of staves of the wooden bucket"},
    "zinc_size": {"default": 0.27, "min": 0.22, "max": 0.32, "desc": "height of the zinc pail (m)"},
    "zinc_tipped": {"default": False, "desc": "zinc pail lies tipped over on its side"},
    "can_size": {"default": 0.26, "min": 0.2, "max": 0.32, "desc": "body height of the watering can (m)"},
    "can_finish": {"default": "zinc", "choices": ["zinc", "painted", "copper"], "desc": "watering can finish"},
    "can_paint": {"default": "#5f8f6c", "desc": "paint colour of a painted watering can"},
    "broom_length": {"default": 1.45, "min": 1.1, "max": 1.8, "desc": "overall length of the besom (m)"},
    "broom_pose": {"default": "rest", "choices": ["rest", "in_bucket", "wall", "ground"],
                   "desc": "rest: twig head on the cobbles, pole across a vessel rim (15-25 deg); in_bucket: head-down in a bucket; "
                           "wall: head-up against the wall plane y = 0 (cluster in y < 0, set mount = wall_ground); ground: lying flat"},
    "wood_tint": {"default": "#b89a78", "desc": "stain of the bucket staves"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: rust, damp, moss, dents"},
    "spread": {"default": 1.0, "min": 0.9, "max": 1.6, "desc": "spacing of the cluster"},
    "turn": {"default": 0.0, "min": -180.0, "max": 180.0, "desc": "rotate the whole cluster (deg; ignored against a wall)"},
    "mount": {"default": "ground", "choices": ["ground", "wall_ground"], "desc": "ground: free-standing, centred; wall_ground: tucked against the wall plane y = 0"},
    "seed": {"default": 1, "min": 0, "max": 99999, "desc": "arrangement seed"},
}
VARIANTS = {
    "washday": {"items": "wood,zinc", "water": "both", "bucket_size": 0.36, "zinc_size": 0.25, "wood_tint": "#8c6a4a", "fill": 0.8, "seed": 4},
    "gardener": {"items": "can,zinc,broom", "can_finish": "painted", "zinc_tipped": True, "water": "none",
                 "can_size": 0.3, "broom_length": 1.25, "seed": 11},
    "sweeper": {"items": "wood,broom", "broom_pose": "in_bucket", "broom_length": 1.6, "age": 0.95, "wood_tint": "#6e5540", "staves": 10,
                "water": "none", "seed": 23},
    "copper_tipped": {"can_finish": "copper", "zinc_tipped": True, "spread": 1.3, "bucket_size": 0.26, "turn": 35.0, "seed": 8},
    "doorway": {"broom_pose": "wall", "mount": "wall_ground", "can_finish": "painted", "can_paint": "#7c3a30", "water": "zinc",
                "wood_tint": "#9a7650", "seed": 5},
}

PR = 0.0145                                              # besom pole radius


# ---------------------------------------------------------------------------------------------- helpers
class _Kit:
    """Parts of one object grouped by material key; finish() joins each group and places it."""

    def __init__(self, name):
        self.name = name
        self.g = {}

    def add(self, key, obj):
        self.g.setdefault(key, []).append(obj)
        return obj

    def finish(self, mats, loc=(0, 0, 0), rot=(0, 0, 0), center_xy=None, seat=True):
        out = {}
        for key, objs in self.g.items():
            for o in objs:
                assign(o, mats[key])
            j = join(objs, f"{self.name}_{key}")
            place(j, loc=loc, rot=rot)
            apply_transform(j)
            out[key] = j
        mn, mx = world_bbox(list(out.values()))
        dx = dy = 0.0
        if center_xy is not None:
            dx = center_xy[0] - 0.5 * (mn.x + mx.x)
            dy = center_xy[1] - 0.5 * (mn.y + mx.y)
        if seat or center_xy is not None:
            for j in out.values():                      # sit exactly on the paving
                place(j, loc=(dx, dy, -mn.z if seat else 0.0))
                apply_transform(j)
        return out


def _bisect(f, a, b, it=30):
    fa = f(a)
    if fa <= 0:
        return a
    for _ in range(it):
        m = 0.5 * (a + b)
        if f(m) > 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _disc(name, r, z, n=24, down=False):
    vs = [(0.0, 0.0, z)] + [(r * cos(2 * pi * i / n), r * sin(2 * pi * i / n), z) for i in range(n)]
    fs = [((0, 1 + (i + 1) % n, 1 + i) if down else (0, 1 + i, 1 + (i + 1) % n)) for i in range(n)]
    return mesh_object(name, vs, fs)


def _rev(name, prof, seg=24, sm=60.0):
    """Open surface of revolution with explicit winding: a profile running bottom-to-top faces outwards,
    outer-edge-to-centre faces up, top-to-bottom faces the axis (no normal guessing on open sheets)."""
    vs, fs, idx = [], [], []
    for (r, z) in prof:
        if r <= 1e-9:
            idx.append([len(vs)] * seg)
            vs.append((0.0, 0.0, z))
        else:
            idx.append(list(range(len(vs), len(vs) + seg)))
            vs += [(r * cos(2 * pi * i / seg), r * sin(2 * pi * i / seg), z) for i in range(seg)]
    for k in range(len(prof) - 1):
        for i in range(seg):
            j = (i + 1) % seg
            f = []
            for v in (idx[k][i], idx[k][j], idx[k + 1][j], idx[k + 1][i]):
                if v not in f:
                    f.append(v)
            if len(f) >= 3:
                fs.append(tuple(f))
    o = mesh_object(name, vs, fs)
    smooth(o, sm)
    return o


def _rivets(name, pts, r=0.0042, h=0.0024):
    """pts: [(x, y, z, yaw_deg, pitch_deg)] - small dome heads pointing radially outwards."""
    base = lathe("rivet", [(r, 0.0), (r * 0.75, h * 0.72), (0.0, h)], segments=6)
    return instances_to_mesh(name, base, [trs((x, y, z), (0, pc, a)) for (x, y, z, a, pc) in pts], keep_base=False)


def _vmap(pts, vals):
    """Mask from per-vertex values of the mesh as it was authored (nearest authored vertex)."""
    P = np.asarray(pts, dtype=np.float64)
    V = np.asarray(vals, dtype=np.float64)

    def fn(co, n):
        if len(co) == len(P) and float(np.abs(co - P).max()) < 1e-4:
            return V
        d = ((co[:, None, :] - P[None, :, :]) ** 2).sum(2)
        return V[d.argmin(1)]
    return fn


def _adist(a, b):
    return np.abs((a - b + pi) % (2 * pi) - pi)


def _stave(name, a0, a1, levels, Ro, th, dr, tg, tw, gin):
    """One bucket stave between angles a0..a1.  levels = [(z, width factor, inset, grime, wear, moss)];
    eased outer arrises with darkened joints, tg / tw = the stave's own tone, gin = inner (bottom, top) damp."""
    am, hw = 0.5 * (a0 + a1), 0.5 * (a1 - a0)
    vs, fs, G, W, M = [], [], [], [], []
    for (z, wf, ins, g, w, m) in levels:
        ro = Ro(z) + dr - ins
        h = hw * wf
        gv = min(0.004 / ro, h * 0.3)
        for j, (a, r) in enumerate(((am - h, ro - 0.0022), (am - h + gv, ro), (am, ro), (am + h - gv, ro), (am + h, ro - 0.0022))):
            vs.append((r * cos(a), r * sin(a), z))
            e = j in (0, 4)
            G.append(min(1.0, g + tg + (0.55 if e else 0.0)))
            W.append(0.0 if e else min(1.0, w + tw))
            M.append(m)
    nl = len(levels)
    for li, gi in ((0, gin[0]), (nl - 1, gin[1])):
        z, wf, ins = levels[li][:3]
        ri = Ro(z) + dr - th + ins
        h = hw * wf
        for a in (am - h, am, am + h):
            vs.append((ri * cos(a), ri * sin(a), z))
            G.append(gi)
            W.append(0.0)
            M.append(0.0)
    ib, it = nl * 5, nl * 5 + 3
    for i in range(nl - 1):
        b, t = i * 5, i * 5 + 5
        for q in range(4):
            fs.append((b + q, b + q + 1, t + q + 1, t + q))
    for q in range(2):
        fs.append((ib + q + 1, ib + q, it + q, it + q + 1))
    fs.append(tuple([ib] + [i * 5 for i in range(nl)] + [it]))
    fs.append(tuple([it + 2] + [i * 5 + 4 for i in range(nl - 1, -1, -1)] + [ib + 2]))
    tb = (nl - 1) * 5
    fs.append((tb, tb + 1, tb + 2, tb + 3, tb + 4, it + 2, it + 1, it))
    fs.append((4, 3, 2, 1, 0, ib, ib + 1, ib + 2))
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    smooth(o, 35)
    uv_grain(o, axis='Z')
    paint(o, grime=_vmap(vs, G), wear=_vmap(vs, W), special=_vmap(vs, M))
    return o


def _sheet_weather(obj, seg, H, s, ztop, ribs, fl, A, sd, mode, seam=None, drips=()):
    """Directional weathering of a sheet-metal vessel in its own frame (axis = Z): dirt runs under the rim roll,
    rust (verdigris on copper) on the bottom inch, along the lapped seam and below fixings, clean rubbed rim and
    rib crests, a damp gradient inside."""
    rs = np.random.RandomState(int(sd) % 100000)
    st = np.where(rs.rand(seg) < 0.45, rs.uniform(0.5, 1.0, seg), 0.0) * (0.7 + 0.6 * A)
    ln = rs.uniform(0.045, 0.15, seg) * s
    nz1 = noise_mask(0.13, seed=sd + 1, lo=0.3, hi=0.8)
    nz2 = noise_mask(0.03, seed=sd + 2, lo=0.2, hi=0.8)
    nz3 = noise_mask(0.016, seed=sd + 3, lo=0.5, hi=0.7)

    def masks(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        rad = np.maximum(np.hypot(x, y), 1e-6)
        nr = (n[:, 0] * x + n[:, 1] * y) / rad
        ang = np.arctan2(y, x)
        col = np.round(ang / (2 * pi / seg)).astype(int) % seg
        inside = (nr < -0.3) | ((np.abs(nr) <= 0.3) & (n[:, 2] > 0.5) & (z < 0.5 * H) & (z > 0.012 * s))
        n1, n2 = nz1(co, n), nz2(co, n)
        g = st[col] * np.clip((z - (ztop - ln[col])) / ln[col], 0.0, 1.0) ** 1.3
        g = np.maximum(g, 0.16 * n1)
        crest = np.zeros(len(z), dtype=bool)
        flank = np.zeros(len(z), dtype=bool)
        for zr in ribs:
            crest |= np.abs(z - zr) < 0.003 * s
            flank |= np.abs(np.abs(z - zr) - fl * s) < 0.003 * s
        g = np.where(flank, np.maximum(g, 0.32), g)
        roll = z > ztop + 0.0015 * s
        g = np.where(crest | roll, 0.0, g)
        grime = np.where(inside, 0.08 + 0.32 * (1.0 - np.clip(z / H, 0.0, 1.0)), g)
        sp = np.clip(1.0 - (z - 0.008 * s) / (0.03 * s), 0.0, 1.0) * (0.5 + 0.5 * n2) * (0.35 + 0.6 * A)
        if seam is not None:
            sp = np.maximum(sp, (_adist(ang, seam) < 0.6 * 2 * pi / seg) * (0.12 + 0.4 * A) * (0.35 + 0.65 * n2) * (z < ztop))
        for (a_, zt, l_) in drips:
            sp = np.maximum(sp, (_adist(ang, a_) < 0.7 * 2 * pi / seg) * (0.25 + 0.5 * A) * np.clip(1.0 - (zt - z) / l_, 0.0, 1.0) * (z <= zt + 0.002))
        if mode == "copper":
            sp = np.maximum(sp, flank * (0.3 + 0.4 * A))
        special = np.where(inside, 0.3 * A * (z < 0.045 * s), sp)
        if mode == "painted":
            wear = nz3(co, n) * ((crest | roll | (z < 0.012 * s)) & ~inside) * (0.4 + 0.6 * A)
        elif mode == "copper":
            wear = np.where(inside, 0.0, np.where(crest | roll, 0.75, 0.22 * n1))
        else:
            wear = np.zeros(len(z))
        return grime, special, wear

    paint(obj, grime=lambda co, n: masks(co, n)[0], special=lambda co, n: masks(co, n)[1], wear=lambda co, n: masks(co, n)[2])


def _runs(name, Rf, z0, lmin, lmax, count, sd, A, avoid=()):
    """Narrow rust / dirt runs hanging below a rim roll or a swage: thin tapering strips lying on the sheet."""
    rs = np.random.RandomState(int(sd) % 100000)
    vs, fs, S, G = [], [], [], []
    for _ in range(count):
        for _try in range(8):
            a = rs.uniform(0.0, 2 * pi)
            if all(abs((a - b + pi) % (2 * pi) - pi) > 0.3 for b in avoid):
                break
        l, w = lmin + (lmax - lmin) * rs.uniform(0.0, 1.0) ** 1.6, rs.uniform(0.0016, 0.004)
        k0, gq, zs_ = len(vs), rs.uniform(0.4, 0.85), z0 - rs.uniform(0.0, 0.004)
        for (q, wf, sv) in ((0.0, 1.0, 0.62), (0.4, 0.8, 0.5), (0.75, 0.5, 0.36), (1.0, 0.12, 0.0)):
            z = zs_ - l * q
            r = Rf(z) + 0.0011
            for sg in (-1.0, 1.0):
                aa = a + sg * w * wf / r
                vs.append((r * cos(aa), r * sin(aa), z))
                S.append(sv * rs.uniform(0.75, 1.0) * (0.55 + 0.45 * A))
                G.append(gq * (1.0 - 0.6 * q))
        for i in range(3):
            b = k0 + 2 * i
            fs.append((b, b + 2, b + 3, b + 1))
    o = mesh_object(name, vs, fs)
    smooth(o, 80)
    uv_cyl(o)
    paint(o, special=_vmap(vs, S), grime=_vmap(vs, G))
    return o


def _fit_weather(obj, mode, A, sd, hot=None, rub=None):
    """Fittings in the vessel's finish: `hot` = solder joints / fixings that rust (or go green), `rub` = handled areas."""
    nz1 = noise_mask(0.06, seed=sd + 4, lo=0.3, hi=0.8)
    nz3 = noise_mask(0.016, seed=sd + 5, lo=0.48, hi=0.7)
    one = lambda co, n: np.ones(len(co))
    hot = hot or (lambda co, n: np.zeros(len(co)))
    rub = rub or one
    sp = lambda co, n: hot(co, n) * (0.25 + 0.5 * A) * (0.4 + 0.6 * nz1(co, n))
    if mode == "copper":
        paint(obj, grime=lambda co, n: 0.1 * nz1(co, n), special=sp, wear=lambda co, n: np.maximum(0.2, 0.75 * rub(co, n)))
    elif mode == "painted":
        paint(obj, grime=lambda co, n: 0.1 + 0.15 * nz1(co, n), special=sp, wear=lambda co, n: nz3(co, n) * rub(co, n) * (0.4 + 0.6 * A))
    else:
        paint(obj, grime=lambda co, n: 0.1 + 0.2 * nz1(co, n), special=sp)


# ---------------------------------------------------------------------------------------------- wooden bucket
def _wood_bucket(p, rng, H, water, A, lap_dir):
    """Local frame: ear staves on +-Y, the rope handle flops over the rim towards +X; lap_dir = local azimuth
    (deg) the riveted hoop laps face."""
    k = _Kit("wbucket")
    s = H / 0.30
    Rt, Rb, th = 0.158 * s, 0.128 * s, 0.017 * s
    Ro = lambda z: Rb + (Rt - Rb) * z / H
    n = max(8, int(round(p.staves / 2.0)) * 2)
    r = rng.fork("staves")
    half = [r.vary(1.0, 0.2) for _ in range(n // 2)]
    half[0] = 1.08
    tot = sum(half) * 2
    widths = [w / tot * 2 * pi for w in half * 2]
    hz = ((0.14 * H, 0.030 * s), (0.68 * H, 0.028 * s))                 # hoops: bottom edge, width
    z0b, z0t = hz[0][0], hz[0][0] + hz[0][1]
    z1b = hz[1][0]
    damp = 0.4 + 0.5 * A
    a = pi / 2 - widths[0] / 2
    for i, w in enumerate(widths):
        top = H + r.jitter(0.0, 0.0025)
        dr = r.jitter(0.0, 0.0008)
        t = r.u(-1.0, 1.0)
        tg, tw = (0.26 * t, 0.0) if t > 0 else (0.0, -0.32 * t)        # each stave its own tone
        lv = [(0.0, damp, 0.0, A * r.u(0.3, 1.0)),                      # damp, mossy foot
              (z0b - 0.001, max(damp, 0.5 + 0.5 * A), 0.0, A * r.u(0.0, 0.9)),
              (z0t + 0.001, 0.25 + 0.3 * A * r.u(0, 1), 0.0, A * r.u(0.0, 0.4)),
              (z0t + 0.028 * s * r.u(0.7, 1.4), 0.0, 0.0, 0.0),
              (z1b - (0.022 + 0.055 * r.u(0, 1)) * s, 0.0, 0.0, 0.0),   # iron stain running down from the upper hoop
              (z1b - 0.001, 0.55 + 0.45 * A * r.u(0.6, 1.0), 0.0, 0.0)]
        if i % (n // 2) == 0:
            eh = H + 0.064 * s
            shape = [(H + 0.012 * s, 1, 0), (eh - 0.024 * s, 0.88, 0), (eh - 0.008 * s, 0.64, 0.001), (eh, 0.36, 0.0035)]
        else:
            shape = [(top - 0.003, 1, 0), (top, 1, 0.003)]
        levels = [(z, 1.0, 0.0, g, w_, m) for (z, g, w_, m) in lv] + [(z, wf, ins, 0.0, 0.4 * r.u(0.3, 1.0), 0.0) for (z, wf, ins) in shape]
        k.add("wood", _stave(f"stave{i}", a, a + w, levels, Ro, th, dr, tg, tw, (0.85 if water else 0.55, 0.15)))
        a += w
    zb = 0.03 * s
    bot = _disc("wb_bottom", Ro(zb) - th + 0.002, zb + 0.008, 20)
    uv_planar(bot)
    paint(bot, grime=0.75 if water else 0.45)
    k.add("wood", bot)
    rin = lambda z: (Ro(z) - th - 0.0008) * cos(max(widths) / 4.0) - 0.0006
    if water:
        zw = lerp(zb + 0.012, H - 0.008 * s, p.fill)
        rw, am = rin(zw), 0.0007
        # still water: a meniscus climbing the staves and a few mm of slow swell so it catches sky and lantern
        wat = _rev("wb_water", [(rw + 0.0004, zw + 0.0034), (rw - 0.0014, zw + 0.0013), (rw - 0.0045, zw + 0.0002), (0.74 * rw, zw - 0.6 * am),
                                (0.46 * rw, zw + am), (0.2 * rw, zw - 0.5 * am), (0.0, zw + 0.6 * am)], 20, 80.0)
        uv_planar(wat)
        k.add("water", wat)
        wet = _rev("wb_wet", [(rin(zw + 0.011), zw + 0.011), (rin(zw + 0.005), zw + 0.005), (rin(zw - 0.003), zw - 0.003)], 20, 80.0)   # dark wet line
        uv_cyl(wet)
        paint(wet, grime=lambda co, n: np.where(co[:, 2] > zw + 0.008, 0.3, 1.0))
        k.add("wood", wet)
    # iron hoops: bevelled conical bands, each closed with a riveted lap
    rv = []
    hr = rng.fork("hoops")
    t = 0.0026
    for j, (z0, wd) in enumerate(hz):
        z1 = z0 + wd
        zm = 0.5 * (z0 + z1)
        hoop = _rev(f"hoop{j}", [(Ro(z0) + 0.0003, z0), (Ro(z0) + t, z0 + 0.0014), (Ro(zm) + t + 0.0003, zm), (Ro(z1) + t, z1 - 0.0014), (Ro(z1) + 0.0003, z1)], 28, 35.0)
        uv_cyl(hoop)
        rust = noise_mask(0.035, seed=p.seed + 7 + j, lo=0.72 - 0.55 * A, hi=1.1 - 0.6 * A)
        paint(hoop, special=lambda co, n, z0=z0, rust=rust: np.maximum(0.8 * rust(co, n), (co[:, 2] < z0 + 0.002) * (0.1 + 0.35 * A) * (0.4 + 0.6 * rust(co, n))),
              wear=lambda co, n, zm=zm, wd=wd: (np.abs(co[:, 2] - zm) > wd / 2 - 0.002) * 0.4)
        k.add("iron", hoop)
        al = radians(lap_dir + (22.0 if j else -16.0) + hr.jitter(0.0, 14.0))
        rm = Ro(zm) + t + 0.0010
        dl = 0.030 * s / rm
        hb = wd / 2 - 0.0028
        lap = sweep(f"hoop_lap{j}", [(rm * cos(al + dl * q), rm * sin(al + dl * q), zm) for q in (-1.0, -0.33, 0.33, 1.0)],
                    [(-0.0012, -hb), (0.0012, -hb), (0.0012, hb), (-0.0012, hb)], smooth_path=False, up=(0, 0, 1))
        paint(lap, special=mask_mul(noise_mask(0.02, seed=p.seed + 17 + j, lo=0.2, hi=0.8), 0.35 + 0.5 * A), wear=0.2)
        k.add("iron", lap)
        for q in (-0.55, 0.55):
            aq = al + dl * q
            rv.append(((rm + 0.001) * cos(aq), (rm + 0.001) * sin(aq), zm, degrees(aq), 90))
    riv = _rivets("hoop_rivets", rv, r=0.0048, h=0.003)
    paint(riv, wear=0.5, special=0.2 + 0.3 * A)
    k.add("iron", riv)
    # rope handle: stopper knots outside the ears, loop flopped over the rim and down the outside (+X)
    rr = 0.0066 * s
    zE = H + 0.032 * s
    t0 = 0.5 + rng.fork("rope").jitter(0.0, 0.07)
    kk = log(0.5) / log(t0)
    drop = 0.115 * s
    path = [(0.0, Ro(zE) + 0.012 * s, zE)]
    for i in range(23):
        tq = i / 22.0
        phi = radians(90.0 - 180.0 * tq)
        sg = sin(pi * tq ** kk)
        z = zE + 0.010 * s * sin(pi * min(sg / 0.5, 1.0)) - drop * smoothstep(0.45, 1.0, sg)
        rho = lerp(Ro(zE) - th - rr - 0.002, Ro(z) + rr + 0.0015, smoothstep(0.12, 0.6, sg))
        path.append((rho * cos(phi), rho * sin(phi), z))
    path.append((0.0, -(Ro(zE) + 0.012 * s), zE))
    grm = mask_mul(noise_mask(0.05, seed=p.seed + 9), 0.3 + 0.4 * A)
    k.add("rope", paint(sweep("rope", resample_path(path, n=25), rr, sides=5, smooth_path=False), grime=grm))
    for sy in (-1, 1):
        y = sy * (Ro(zE) + 0.012 * s)
        k.add("rope", paint(sphere("knot", 0.0125 * s, loc=(0, y, zE), segments=7, rings=4, scale=(1.0, 0.82, 1.0)), grime=grm))
        k.add("rope", sweep("rope_tail", [(0.0, y, zE - 0.004), (0.006 * sy, y + 0.006 * sy, zE - 0.028 * s), (0.004, y + 0.004 * sy, zE - 0.05 * s)],
                            rr * 0.8, sides=5, smooth_path=False, scale_fn=lambda q: 0.9 + 0.5 * q))
    return k, {"rin": lambda z: rin(z) - 0.004, "zfloor": zb + 0.008, "H": H}


# ---------------------------------------------------------------------------------------------- zinc pail
def _zinc_bucket(p, rng, H, water, tipped, A):
    """Local frame: ears on +-X, the bail swings about X towards -Y."""
    k = _Kit("zbucket")
    s = H / 0.27
    SEG = 24
    Rt, Rb = 0.150 * s, 0.106 * s
    R = lambda z: Rb + (Rt - Rb) * z / H
    ribs = (0.36 * H, 0.70 * H)
    ztop = H - 0.012 * s
    prof = [(Rb - 0.010 * s, 0.014 * s), (Rb - 0.006 * s, 0.0), (Rb + 0.0015, 0.0), (Rb + 0.002, 0.008 * s), (R(0.03 * s) + 0.0003, 0.03 * s)]
    for zr in ribs:
        prof += [(R(zr - 0.009 * s), zr - 0.009 * s), (R(zr) + 0.0035 * s, zr), (R(zr + 0.009 * s), zr + 0.009 * s)]
    rw = R(0.03 * s) - 0.0025
    prof += [(R(0.855 * H), 0.855 * H), (R(ztop), ztop), (Rt + 0.0045 * s, H - 0.0085 * s), (Rt + 0.007 * s, H - 0.003 * s), (Rt + 0.0045 * s, H + 0.0025 * s),
             (Rt - 0.0015 * s, H + 0.001 * s), (Rt - 0.003 * s, H - 0.008 * s),
             (rw, 0.036 * s), (rw - 0.005 * s, 0.0245 * s), (rw - 0.0105 * s, 0.0200 * s)]       # inside wall, folded bottom seam
    body = lathe("zb_body", prof, segments=SEG, smooth_angle=65)
    sa_i = int(rng.fork("seam").u(0.3, 0.7) * SEG)
    sa = 2 * pi * sa_i / SEG
    zc = H - 0.014 * s
    _sheet_weather(body, SEG, H, s, ztop, ribs, 0.009, A, p.seed + 11, "galv", seam=sa, drips=((0.0, 0.86 * H, 0.10 * s), (pi, 0.86 * H, 0.10 * s)))
    displace_noise(body, strength=0.0002 + 0.0008 * A, scale=0.13, seed=p.seed + 31)
    k.add("zinc", body)
    k.add("zinc", _runs("zb_runs", R, ztop - 0.001, 0.028 * s, 0.056 * s, 5, p.seed + 71, A, avoid=(0.0, pi, sa)))
    k.add("zinc", _runs("zb_runs2", R, ribs[1] - 0.011 * s, 0.03 * s, 0.07 * s, 4, p.seed + 72, A, avoid=(sa,)))
    # sheet bottom (slightly oil-canned), dirt gathered in the seam
    rf = rw - 0.0100 * s
    flo = _rev("zb_floor", [(rf, 0.0199 * s), (0.62 * rf, 0.0212 * s), (0.0, 0.0220 * s)], SEG, 80.0)
    uv_planar(flo)
    nzf = noise_mask(0.05, seed=p.seed + 33, lo=0.2, hi=0.8)
    paint(flo, grime=lambda co, n: np.clip(0.18 + 0.55 * (np.hypot(co[:, 0], co[:, 1]) / rf) ** 3 + 0.2 * nzf(co, n), 0, 1),
          special=lambda co, n: (np.hypot(co[:, 0], co[:, 1]) > 0.9 * rf) * 0.45 * A)
    k.add("zinc", flo)
    und = _disc("zb_under", Rb - 0.010 * s + 0.002, 0.014 * s, 14, down=True)
    paint(und, grime=0.5, special=0.5 * A)
    k.add("zinc", und)
    # lapped, soldered side seam
    seam = sweep("zb_seam", [((R(z) + 0.0006) * cos(sa), (R(z) + 0.0006) * sin(sa), z) for z in (0.012 * s, H - 0.014 * s)],
                 rounded_rect(0.013 * s, 0.0018, 0.0006, 1), smooth_path=False, up=(cos(sa), sin(sa), 0))
    paint(seam, special=mask_mul(noise_mask(0.03, seed=p.seed + 35, lo=0.2, hi=0.8), 0.3 + 0.5 * A))
    k.add("zinc", seam)
    if water and not tipped:
        zw = lerp(0.03 * s, H - 0.016 * s, p.fill)
        rq, am = R(zw) - 0.0026, 0.0007
        wat = _rev("zb_water", [(rq, zw + 0.0026), (rq - 0.0016, zw + 0.0009), (rq - 0.0045, zw + 0.0002), (0.74 * rq, zw - 0.6 * am),
                                (0.46 * rq, zw + am), (0.2 * rq, zw - 0.5 * am), (0.0, zw + 0.6 * am)], SEG, 80.0)
        uv_planar(wat)
        k.add("water", wat)
    # ears: stamped lugs riveted below the roll, leaning out over it
    hp, w = 0.062 * s, 0.034 * s
    r_bot, r_top = R(zc - hp / 2) + 0.0006, Rt + 0.0072 * s
    lean = atan((r_top - r_bot) / hp)
    rc = 0.5 * (r_bot + r_top)
    outline = [(-w / 2, -hp / 2), (w / 2, -hp / 2)] + [(w / 2 * cos(radians(a)), hp / 2 - w / 2 + w / 2 * sin(radians(a))) for a in range(0, 181, 30)]
    rv = []
    for sx in (-1, 1):
        ear = extrude_profile("zb_ear", outline, 0.0022, loc=(sx * rc, 0, zc), rot=(90 + degrees(lean), 0, 90 * sx))
        paint(ear, special=lambda co, n: (co[:, 2] < zc - 0.008 * s) * (0.3 + 0.5 * A), grime=0.15)
        k.add("zinc", ear)
        for dy in (-0.008 * s, 0.008 * s):
            zr = zc - 0.019 * s
            rv.append((sx * (rc - 0.019 * s * sin(lean) + 0.0021), dy, zr, 0 if sx > 0 else 180, 90))
    riv = _rivets("zb_rivets", rv, r=0.0036, h=0.002)
    paint(riv, special=0.45 + 0.4 * A)
    k.add("zinc", riv)
    # wire bail with a turned wooden grip
    zp = H + 0.007 * s
    a_, xs, gr = Rt + 0.0125 * s, 0.054 * s, 0.013 * s
    b_ = a_ * 1.12
    ph1 = acos(xs / a_)
    q_top = b_ * sin(ph1)
    if tipped:
        ha = atan((Rt + 0.007 * s - Rb - 0.002) / H)
        th_t = pi / 2 - ha
        shift = (Rb + 0.002) * sin(th_t)
        alpha = _bisect(lambda al: -q_top * sin(al) * sin(th_t) + (zp + q_top * cos(al)) * cos(th_t) + shift - gr - 0.0005, 0.0, radians(100))
    else:
        th_t = 0.0
        alpha = _bisect(lambda al: q_top * sin(al) - (R(zp + q_top * cos(al)) + gr + 0.0015), pi / 2, radians(150))
    e1, e2 = -sin(alpha), cos(alpha)
    half = [(Rt + 0.001, 0.008 * s), (Rt + 0.0035, 0.0), (a_, 0.0)] + [(a_ * cos(ph1 * i / 6), b_ * sin(ph1 * i / 6)) for i in range(1, 7)]
    full = half + [(-x, q) for (x, q) in reversed(half)]
    pts = [(x, q * e1, zp + q * e2) for (x, q) in full]
    bail = sweep("zb_bail", resample_path(pts, n=24), 0.0028 * s, sides=5, smooth_path=False)
    paint(bail, special=mask_mul(noise_mask(0.03, seed=p.seed + 15, lo=0.3, hi=0.7), 0.3 + 0.6 * A), wear=0.3)
    k.add("iron", bail)
    grip = lathe("zb_grip", [(0.0045 * s, -0.048 * s), (0.0115 * s, -0.046 * s), (0.013 * s, -0.03 * s), (0.0112 * s, 0.0), (0.013 * s, 0.03 * s),
                             (0.0115 * s, 0.046 * s), (0.0045 * s, 0.048 * s)], segments=7, loc=(0, q_top * e1, zp + q_top * e2), rot=(0, 90, 0))
    paint(grip, wear=0.45, grime=0.25)
    k.add("grip", grip)
    return k, {"rin": lambda z: R(z) - 0.007, "zfloor": 0.022 * s, "H": H, "tilt": degrees(th_t)}


# ---------------------------------------------------------------------------------------------- watering can
def _spout(s, H, R):
    B = np.array((R * 0.78, 0.0, 0.060 * s))
    T = np.array((R + 0.275 * s, 0.0, H + 0.035 * s))
    r0 = 0.0235 * s
    return B, T, r0, (lambda t: (1.0 - 0.5 * t) + 0.35 * max(0.0, 1.0 - t / 0.13) ** 2)


def _spout_top(s, H, R, x):
    """Height of the top of the spout above the paving at local x (the crook the besom pole rests in)."""
    B, T, r0, sc = _spout(s, H, R)
    t = (x - B[0]) / (T[0] - B[0])
    return B[2] + t * (T[2] - B[2]) + r0 * sc(t) / cos(atan2(T[2] - B[2], T[0] - B[0]))


def _watering_can(p, rng, H, A, mode):
    """Local frame: spout towards +X, handles in the XZ plane."""
    k = _Kit("can")
    s = H / 0.26
    SEG = 24
    R = 0.105 * s
    ztop = H - 0.010 * s
    ribs = (0.047 * s, H - 0.049 * s)
    prof = [(0.0, 0.012 * s), (R - 0.009 * s, 0.012 * s), (R - 0.005 * s, 0.0), (R + 0.001, 0.0), (R + 0.002, 0.006 * s), (R, 0.012 * s),
            (R, 0.040 * s), (R + 0.003 * s, 0.047 * s), (R, 0.054 * s), (R, 0.068 * s), (R, 0.5 * H), (R, H - 0.070 * s), (R, H - 0.056 * s), (R + 0.003 * s, H - 0.049 * s), (R, H - 0.042 * s),
            (R, ztop), (R + 0.004 * s, H - 0.007 * s), (R + 0.0055 * s, H - 0.002 * s), (R + 0.003 * s, H + 0.002 * s), (R - 0.001, H + 0.001 * s),
            (R - 0.002, H - 0.008 * s), (R - 0.002, 0.016 * s), (0.0, 0.016 * s)]
    body = lathe("can_body", prof, segments=SEG, smooth_angle=65)
    B, T, r0, sc = _spout(s, H, R)
    _sheet_weather(body, SEG, H, s, ztop, ribs, 0.007, A, p.seed + 17, mode, seam=pi * (1.0 + 2.0 / SEG * 3),
                   drips=((0.0, B[2] + 0.01 * s, 0.06 * s), (pi, 0.078 * s, 0.06 * s), (pi, H - 0.050 * s, 0.07 * s)))
    displace_noise(body, strength=0.0002 + 0.0007 * A, scale=0.12, seed=p.seed + 47)
    k.add("body", body)
    k.add("body", _runs("can_runs", lambda z: R, ribs[1] - 0.008 * s, 0.02 * s, 0.12 * s, 4, p.seed + 73, A, avoid=(0.0, pi)))
    k.add("body", _runs("can_runs2", lambda z: R, ztop - 0.001, 0.012 * s, 0.03 * s, 3, p.seed + 74, A, avoid=(0.0, pi)))
    # half-dome hood soldered to the rim over the spout half, front edge rolled round a wire
    nu, nv = 12, 4
    hh, Rh, zh = 0.040 * s, R + 0.0012, H + 0.0015 * s
    vs, fs = [], []
    for j in range(nv):
        tq = 0.5 * pi * j / nv
        for i in range(nu + 1):
            a = radians(-90.0 + 180.0 * i / nu)
            vs.append((Rh * cos(tq) * cos(a), Rh * cos(tq) * sin(a), zh + hh * sin(tq)))
    apex = len(vs)
    vs.append((0.0, 0.0, zh + hh))
    for j in range(nv - 1):
        for i in range(nu):
            b = j * (nu + 1) + i
            fs.append((b, b + 1, b + nu + 2, b + nu + 1))
    for i in range(nu):
        b = (nv - 1) * (nu + 1) + i
        fs.append((b, b + 1, apex))
    hd = mesh_object("can_hood", vs, fs)
    solidify(hd, 0.002, offset=-1.0)
    smooth(hd, 70)
    uv_planar(hd)
    joint = lambda co, n: (co[:, 2] < zh + 0.007 * s) * 1.0
    _fit_weather(hd, mode, A, p.seed + 61, hot=joint, rub=lambda co, n: (np.abs(co[:, 0]) < 0.004) * 1.0)
    k.add("body", hd)
    wire = sweep("can_hood_wire", [(0.0, -Rh * cos(q), zh + hh * sin(q)) for q in np.linspace(0.0, pi, 9)], 0.0030 * s, sides=5, smooth_path=False)
    _fit_weather(wire, mode, A, p.seed + 62)
    k.add("body", wire)
    # tapering spout from low on the body, braced back to the rim
    sp = [tuple(B + (T - B) * i / 7.0) for i in range(8)]
    spo = sweep("can_spout", sp, r0, sides=10, smooth_path=False, scale_fn=sc)
    root = lambda co, n: np.clip(1.0 - (np.linalg.norm(co - B[None, :], axis=1) - 0.03 * s) / (0.035 * s), 0.0, 1.0)
    _fit_weather(spo, mode, A, p.seed + 63, hot=root, rub=lambda co, n: (co[:, 0] > T[0] - 0.05 * s) * 1.0)
    k.add("body", spo)
    d = (T - B) / np.linalg.norm(T - B)
    el = degrees(atan2(d[2], d[0]))
    S = B + (T - B) * 0.56 + np.array((-0.2, 0, 0.9)) * r0 * 0.7
    brace = sweep("can_brace", [tuple(S), (R + 0.002, 0.0, H - 0.006 * s)], rounded_rect(0.0035 * s, 0.013 * s, 0.001, 1), smooth_path=False, up=(0, 1, 0))
    _fit_weather(brace, mode, A, p.seed + 64, hot=lambda co, n: (np.abs(co[:, 0] - 0.5 * (S[0] + R)) > 0.4 * (S[0] - R)) * 1.0)
    k.add("body", brace)
    # brass rose with a perforated, slightly domed face
    face = [(0.039, 0.070), (0.028, 0.0735), (0.014, 0.0755), (0.0, 0.0762)]
    rp = [(0.0128, 0.0), (0.0150, 0.001), (0.0150, 0.014), (0.0135, 0.016), (0.039, 0.060), (0.042, 0.062), (0.042, 0.068)] + face
    rloc = tuple(T - d * 0.014 * s)
    rrot = (0.0, 90.0 - el, 0.0)
    rose = lathe("can_rose", [(a * s, b * s) for (a, b) in rp], segments=10, loc=rloc, rot=rrot)
    paint(rose, wear=0.55, grime=0.15, special=mask_mul(noise_mask(0.02, seed=p.seed + 65, lo=0.3, hi=0.8), 0.15 + 0.35 * A))
    k.add("brass", rose)
    hexa = mesh_object("rose_hole", [(0.0023 * s * cos(radians(60 * i)), 0.0023 * s * sin(radians(60 * i)), 0.0) for i in range(6)], [tuple(range(6))])
    M = trs(rloc, rrot)
    hm = []
    for (rad_, cnt, slope) in ((0.0, 1, 0.0), (0.0125, 6, 6.0), (0.025, 11, 15.0)):
        zf = float(np.interp(rad_, [0.0, 0.014, 0.028, 0.039], [0.0762, 0.0755, 0.0735, 0.070])) + 0.0005
        for i in range(cnt):
            a = 360.0 * i / cnt + slope * 3
            hm.append(M @ trs((rad_ * s * cos(radians(a)), rad_ * s * sin(radians(a)), zf * s), (0, slope, a)))
    k.add("iron", instances_to_mesh("rose_holes", hexa, hm, uv_jitter=False, keep_base=False))
    # tube carrying handle from the hood over the top, strap handle down the back
    xh = 0.050 * s
    zs_ = zh + hh * sin(acos(xh / Rh)) - 0.004 * s
    top = [(xh, 0, zs_), (0.047 * s, 0, H + 0.076 * s), (0.012 * s, 0, H + 0.108 * s), (-0.050 * s, 0, H + 0.103 * s),
           (-0.094 * s, 0, H + 0.064 * s), (-R - 0.0045 * s, 0, H + 0.014 * s), (-R - 0.0055 * s, 0, H - 0.024 * s)]
    th_ = sweep("can_top_handle", resample_path(top, n=13), 0.0085 * s, sides=6, smooth_path=False)
    ends = lambda co, n: (co[:, 2] < H + 0.05 * s) * 1.0
    _fit_weather(th_, mode, A, p.seed + 66, hot=ends, rub=lambda co, n: np.clip((co[:, 2] - H - 0.07 * s) / (0.02 * s), 0.0, 1.0))
    k.add("body", th_)
    back = [(-R + 0.002, 0, H - 0.050 * s), (-R - 0.036 * s, 0, H - 0.040 * s), (-R - 0.074 * s, 0, H - 0.080 * s), (-R - 0.078 * s, 0, H * 0.5),
            (-R - 0.052 * s, 0, 0.105 * s), (-R + 0.002, 0, 0.078 * s)]
    bh = sweep("can_back_handle", resample_path(back, n=12), rounded_rect(0.005 * s, 0.024 * s, 0.0018, 1), smooth_path=False, up=(0, 1, 0))
    _fit_weather(bh, mode, A, p.seed + 67, hot=lambda co, n: (co[:, 0] > -R - 0.02 * s) * 1.0,
                 rub=lambda co, n: (co[:, 0] < -R - 0.06 * s) * 1.0)
    k.add("body", bh)
    # soldered pads where the handles land
    rv = [(-R - 0.001, 0.0, H - 0.050 * s, 180, 90), (-R - 0.001, 0.0, 0.078 * s, 180, 90)]
    pads = _rivets("can_pads", rv, r=0.016 * s, h=0.004)
    _fit_weather(pads, mode, A, p.seed + 68, hot=lambda co, n: 0.6 * np.ones(len(co)), rub=lambda co, n: np.zeros(len(co)))
    k.add("body", pads)
    return k, {"R": R, "H": H}


# ---------------------------------------------------------------------------------------------- besom
def _besom_dims(L):
    bs = clamp(L / 1.45, 0.9, 1.1)
    return bs, 0.63 * bs, 0.09 * bs                      # scale, head length (butts to tips), head radius


def _besom(p, rng, L, A):
    """Local frame: handle end at the origin, broom runs up +Z with the twig head at the top.
    The head is a bundle of individual birch twigs in three lengths, bound twice round the pole."""
    k = _Kit("besom")
    bs, Hd, _ = _besom_dims(L)
    zb0 = L - Hd
    Lp = zb0 + 0.22 * bs
    r = rng.fork("besom")
    bx, by = r.jitter(0.0, 0.011), r.jitter(0.0, 0.011)

    def crook(z):
        g = sin(pi * clamp((z - 0.48) / max(Lp - 0.48, 0.1))) if z > 0.48 else 0.0
        return bx * g, by * g

    pts = []
    for i in range(8):
        z = Lp * i / 7.0
        cx, cy = crook(z)
        pts.append((cx + (r.jitter(0, 0.003) if 2 < i < 7 else 0.0), cy, z))
    pole = sweep("pole", pts, PR, sides=7, smooth_path=False, scale_fn=lambda t: 1.0 - 0.2 * t)
    paint(pole, wear=mask_mul(noise_mask(0.12, seed=p.seed + 21, lo=0.2, hi=0.8), 0.6), grime=lambda co, n: np.clip(1.0 - co[:, 2] / 0.12, 0.0, 1.0) * 0.6)
    k.add("pole", pole)
    tz = [0.0, 0.085, 0.19, 0.32, 0.5, 0.72, 0.9, 1.0]
    tr = [0.036, 0.031, 0.033, 0.050, 0.073, 0.088, 0.090, 0.086]
    Rbf = lambda zf: float(np.interp(zf, tz, tr)) * bs
    # dark heart of the bundle (only glimpsed between the twigs)
    core = sweep("besom_heart", [(0.0, 0.0, zb0 + Hd * zf) for zf in (0.03, 0.2, 0.45, 0.7, 0.88)], 1.0, sides=7, smooth_path=False,
                 scale_fn=lambda t: Rbf(0.03 + 0.85 * t) * (0.62 - 0.25 * t))
    paint(core, grime=1.0)
    k.add("twig", core)
    for zf in (0.085, 0.19):
        rc = Rbf(zf) + 0.0045
        ring = [(rc * cos(2 * pi * i / 10), rc * sin(2 * pi * i / 10), zb0 + Hd * zf) for i in range(10)]
        bind = sweep("besom_bind", ring, rounded_rect(0.008, 0.024 * bs, 0.003, 1), closed=True, smooth_path=False)
        paint(bind, grime=0.3)
        k.add("rope", bind)
    tips = noise_mask(0.05, seed=p.seed + 23, lo=0.2, hi=0.8)
    tw_paint = dict(grime=lambda co, n: np.clip(0.12 + 0.3 * tips(co, n) + 0.35 * np.clip((co[:, 2] - zb0) / Hd, 0, 1) ** 2, 0, 1),
                    wear=lambda co, n: (co[:, 2] < zb0 + 0.012) * 0.6)
    N = 68
    for j in range(N):
        a = j * 2.39996 + r.jitter(0.0, 0.25)
        lyr = j % 10
        rho = r.u(0.9, 1.0) if lyr < 5 else (r.u(0.66, 0.86) if lyr < 8 else r.u(0.4, 0.62))
        c = r.u(0, 1)
        tipf = r.u(0.93, 1.03) if c < 0.45 else (r.u(0.74, 0.9) if c < 0.8 else r.u(0.54, 0.7))
        zf0 = r.u(-0.035, 0.04)
        bow, drift, splay, kink = r.jitter(0.0, 0.007), r.jitter(0.0, 0.22), r.u(0.0, 0.02) * rho, r.jitter(0.0, 0.14)
        tw = []
        for q in (0.0, 0.26, 0.62, 1.0):
            zf = lerp(zf0, tipf, q)
            rad = max(rho * Rbf(clamp(zf, 0.0, 1.0)), 0.0135 if zf < 0.34 else 0.0) + bow * sin(pi * q) + splay * q ** 3
            aa = a + drift * q + kink * q ** 3
            tw.append((rad * cos(aa), rad * sin(aa), zb0 + Hd * zf))
        k.add("twig", paint(sweep(f"twig{j}", tw, 0.0050 * r.u(0.8, 1.2), sides=3, smooth_path=False, cap=(j % 3 == 0),
                                  scale_fn=lambda t: 1.0 - 0.62 * t), **tw_paint))
        if j % 2 == 0 and tipf > 0.7:                    # side shoot forking off the twig
            f = r.u(0.1, 0.5)
            b0 = np.array(tw[2]) * (1 - f) + np.array(tw[3]) * f
            dirv = np.array(tw[3]) - np.array(tw[2])
            dirv /= max(np.linalg.norm(dirv), 1e-6)
            a2 = a + r.jitter(0.0, 1.3)
            tip = b0 + dirv * r.u(0.07, 0.14) + np.array((cos(a2), sin(a2), 0.0)) * r.u(0.012, 0.035)
            tip[2] = min(tip[2], zb0 + Hd * 1.04)
            k.add("twig", paint(sweep(f"shoot{j}", [tuple(b0), tuple(tip)], 0.0030, sides=3, smooth_path=False, cap=False,
                                      scale_fn=lambda t: 1.0 - 0.65 * t), **tw_paint))
    return k, {"crook": crook}


# ---------------------------------------------------------------------------------------------- build
def build(p, rng):
    toks = {t for t in str(p["items"]).lower().replace(";", ",").replace(" ", ",").split(",") if t}
    if not (toks & {"wood", "zinc", "can", "broom"}):
        toks = {"wood", "zinc", "can", "broom"}
    A = p.age
    wallm = str(p["mount"]) == "wall_ground"
    pose = str(p.broom_pose) if "broom" in toks else None
    wallf = wallm or pose == "wall"                      # laid out against the wall plane y = 0
    mode = {"zinc": "galv", "painted": "painted", "copper": "copper"}[p.can_finish]
    pole_m = mat("wood_oak", "#d6bc94")
    # weathered hot-dip galvanising: matt light grey bloom (the library `zinc` is a dark mirror metal that goes
    # black away from the key light), rust painted where the zinc has gone
    galv = mat("iron_painted", "#a3aaab", rough=1.3, wear=0.0, scale=4.0)
    can_m = {"galv": galv, "painted": mat("iron_painted", p.can_paint, rough=1.1), "copper": mat("brass", "#ffab85", rough=0.6)}[mode]
    mats = {"wood": mat("wood_oak", p.wood_tint), "iron": mat("iron_painted", "#4d4844", rough=0.8), "rope": mat("rope"),
            "water": mat("water", "#86adb8", opacity=0.9, rough=2.0), "zinc": galv, "grip": pole_m, "pole": pole_m,
            "twig": mat("wood_oak", "#a06a45", rough=1.1), "brass": mat("brass", rough=1.3), "body": can_m}

    ws, zs, cs = p.bucket_size / 0.30, p.zinc_size / 0.27, p.can_size / 0.26
    tipped = bool(p.zinc_tipped) and "zinc" in toks
    lay = rng.fork("layout")
    ground = [g for g in ("wood", "zinc", "can") if g in toks]
    n = len(ground)
    foot = {"wood": 0.166 * ws, "can": 0.125 * cs, "zinc": (0.21 if tipped else 0.160) * zs}
    tops = {"wood": p.bucket_size * 1.22, "can": p.can_size * 1.45, "zinc": 0.31 * zs if tipped else p.zinc_size + 0.012}
    # a loose row (reads from every side and tucks along a wall); the seed shuffles order, gaps and stagger
    d1, d2, d3, d4 = lay.u(0, 1), lay.u(0, 1), lay.u(0, 1), lay.u(0, 1)
    order = list(ground)
    if n == 3:
        kx = int(d2 * 2.999)
        order = order[kx:] + order[:kx]
    if d4 < 0.5:
        order.reverse()
    pos, x = {}, 0.0
    for i, g in enumerate(order):
        if i > 0:
            x += foot[order[i - 1]] + foot[g] + lay.u(0.015, 0.05) + (p.spread - 1.0) * 0.25
        pos[g] = np.array((x, lay.jitter(0.0, 0.045)))
    if n == 3:
        pos[order[1]][1] += lay.u(0.03, 0.10) * (-1.0 if wallf else 1.0)
    if n:
        cx = 0.5 * (pos[order[0]][0] + pos[order[-1]][0])
        for g in order:
            pos[g][0] -= cx
    for _ in range(40):                                  # push overlapping footprints apart
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ground[i], ground[j]
                dv = pos[b] - pos[a]
                dd = float(np.linalg.norm(dv))
                need = foot[a] + foot[b] + 0.012
                if dd < need:
                    dv = dv / max(dd, 1e-6) * (need - dd) * 0.5
                    pos[a] -= dv
                    pos[b] += dv
    if wallf and n:                                      # backs of the vessels a finger off the wall
        sh = -0.03 - max(pos[g][1] + foot[g] for g in ground)
        for g in ground:
            pos[g][1] += sh

    def outward(g):
        """Free side of an item: off the end of the row, or out to the front for one in the middle."""
        if n < 2:
            return 180.0 if d1 < 0.5 else 0.0
        if g == order[0]:
            return 180.0
        return 0.0 if g == order[-1] else 270.0

    def frontish(g, lo, hi):
        """Outward direction swung lo..hi degrees towards the front (-Y)."""
        o = outward(g)
        m = lay.u(lo, hi)
        if o == 270.0:
            return o + (m if d3 < 0.5 else -m) * 0.6
        return o + m if o == 180.0 else o - m

    unit = lambda deg: np.array((cos(radians(deg)), sin(radians(deg))))

    # ---- the besom: what carries it and where it lies
    L = p.broom_length
    bs, Hd, hr_ = _besom_dims(L)
    h0 = 0.045 * bs                                      # height of the head's axis where the twig tips lie on the paving
    upright = [g for g in ("wood", "zinc") if g in toks and not (g == "zinc" and tipped)]
    wat = {"wood": "wood" in toks and p.water in ("wood", "both"), "zinc": "zinc" in toks and p.water in ("zinc", "both") and not tipped}
    sup, plan = None, None
    if pose == "in_bucket":
        cands = [g for g in upright if not wat[g]] or upright
        if cands:
            sup = cands[0]
            plan = {"gam": frontish(sup, 20.0, 70.0), "alpha": radians(lay.u(11.0, 15.0))}
        else:
            pose = "rest"
    can_yaw = None
    if pose == "rest":
        for g in ("wood", "zinc", "can"):
            if g in toks and not (g == "zinc" and tipped):
                sup = g
                break
        if sup is None:
            pose = "ground"
    if "can" in toks and not (pose == "rest" and sup == "can"):
        can_yaw = frontish("can", 30.0, 62.0)
    obst = [(pos[g], foot[g], tops[g], g) for g in ground]
    if can_yaw is not None:                              # the spout sticks out of the can's footprint
        for f_, z_ in ((0.2, 0.2), (0.29, 0.27), (0.37, 0.33)):
            obst.append((pos["can"] + unit(can_yaw) * f_ * cs, 0.035, z_ * cs, "spout"))

    def rest_try(beta):
        """Head lies out along azimuth beta from the contact point; the pole climbs back over the support."""
        hd = unit(beta)
        yaw = None
        if sup == "can":
            Rc = 0.105 * cs
            best = None
            for sg in (1.0, -1.0):
                yw = beta - 90.0 * sg
                sc_ = float(np.dot(unit(yw), unit(outward("can")))) - 0.6 * sin(radians(yw))
                tipS = pos["can"] + unit(yw) * 0.40 * cs
                ok = all(np.linalg.norm(tipS - pos[g]) > foot[g] + 0.04 for g in ground if g != "can")
                if ok and (best is None or sc_ > best[0]):
                    best = (sc_, yw)
            if best is None:
                return None
            yaw = best[1]
            C = pos["can"] + unit(yaw) * (Rc + PR + 0.003)
            hc = _spout_top(cs, p.can_size, Rc, Rc + PR + 0.003) + PR * 0.93
        elif sup == "wood":
            C = pos[sup] + hd * 0.152 * ws
            hc = p.bucket_size - 0.001 + PR * 0.97
        else:
            C = pos[sup] + hd * 0.154 * zs
            hc = p.zinc_size + 0.0025 * zs + PR * 0.97
        sc = clamp(0.56 * L, Hd + 0.10, L - 0.22)
        th = asin(clamp((hc - h0) / sc, sin(radians(6.0)), sin(radians(26.0))))
        sc = (hc - h0) / sin(th)
        if sc > L - 0.10 or sc < Hd + 0.03:
            return None
        tip = C + hd * sc * cos(th)
        for s_ in np.linspace(0.0, L, 30):
            P = tip - hd * s_ * cos(th)
            z = h0 + s_ * sin(th)
            w = hr_ if s_ < Hd * 0.9 else PR
            for (c_, f_, t_, g) in obst:
                if g == sup or (g == "spout" and sup == "can"):
                    continue
                if np.linalg.norm(P - c_) < f_ + w + 0.015 and z - w < t_ + 0.015:
                    return None
        return {"tip": tip, "th": th, "beta": beta, "yaw": yaw, "sc": sc, "hc": hc}

    if pose == "rest":
        b0 = frontish(sup, 0.0, 22.0)
        for off in (0.0, -20.0, 20.0, -40.0, 40.0, -65.0, 65.0, -95.0, 95.0, 130.0, -130.0, 180.0):
            plan = rest_try(b0 + off)
            if plan:
                break
        if plan is None:
            pose, sup = "ground", None

    built, allobjs = {}, []
    infos = {}
    if "wood" in toks:
        if sup == "wood" and pose == "rest":
            yaw = plan["beta"] - 180.0 + lay.jitter(0.0, 16.0)        # pole crosses the rim opposite the flopped rope
        elif sup == "wood":
            yaw = plan["gam"] - 180.0
        else:
            yaw = -90.0 + lay.jitter(0.0, 50.0)
        kit, infos["wood"] = _wood_bucket(p, rng.fork("wood"), p.bucket_size, wat["wood"], A, -90.0 - yaw)
        built["wood"] = kit.finish(mats, loc=(pos["wood"][0], pos["wood"][1], 0), rot=(0, 0, yaw))
    if "zinc" in toks:
        kit, infos["zinc"] = _zinc_bucket(p, rng.fork("zinc"), p.zinc_size, wat["zinc"], tipped, A)
        if sup == "zinc" and pose == "rest":
            yaw = plan["beta"] - 90.0 + lay.jitter(0.0, 20.0)         # pole crosses the rim opposite the hanging bail
        elif sup == "zinc":
            yaw = plan["gam"] - 90.0
        elif tipped:
            o_ = outward("zinc")                         # mouth lies open to the front (never turned into its neighbours)
            yaw = 270.0 + (-lay.u(20.0, 60.0) if o_ == 180.0 else lay.u(-25.0, 8.0) if o_ == 0.0 else lay.jitter(0.0, 15.0)) + 90.0
        else:
            yaw = lay.jitter(0.0, 45.0)
        built["zinc"] = kit.finish(mats, loc=(pos["zinc"][0], pos["zinc"][1], 0), rot=(infos["zinc"]["tilt"], 0, yaw),
                                   center_xy=tuple(pos["zinc"]) if tipped else None)
    if "can" in toks:
        kit, infos["can"] = _watering_can(p, rng.fork("can"), p.can_size, A, mode)
        yaw = plan["yaw"] if can_yaw is None else can_yaw
        built["can"] = kit.finish(mats, loc=(pos["can"][0], pos["can"][1], 0), rot=(0, 0, yaw))
    if "broom" in toks:
        kit, info = _besom(p, rng.fork("broom"), L, A)
        if pose == "in_bucket":
            # head-down in the bucket, twigs splayed on its bottom, pole leaning on the rim
            si = infos[sup]
            gam, al = plan["gam"], plan["alpha"]
            c0 = pos[sup] - unit(gam) * 0.03
            zt = si["zfloor"] + 0.004
            org = c0 + unit(gam) * L * sin(al)
            b = kit.finish(mats, loc=(org[0], org[1], zt + L * cos(al)), rot=(0, 180.0 + degrees(al), gam), seat=False)
            cx_, cy_, Hs, rin, zf_ = float(pos[sup][0]), float(pos[sup][1]), si["H"], si["rin"], si["zfloor"]

            def tuck(v):
                if v.z < Hs:
                    dx, dy = v.x - cx_, v.y - cy_
                    rr_ = hypot(dx, dy)
                    rm = rin(max(v.z, zf_))
                    if rr_ > rm:
                        v.x, v.y = cx_ + dx * rm / rr_, cy_ + dy * rm / rr_
                    v.z = max(v.z, zf_ + 0.003)
                return v
            transform_verts(b["twig"], tuck)
        elif pose == "wall":
            # head-up against the wall (the way a besom is stored), foot out in front of the row end
            side = -1.0 if d1 < 0.5 else 1.0
            xb = 0.0
            if n:
                xb = min(pos[g][0] - foot[g] for g in ground) - 0.10 if side < 0 else max(pos[g][0] + foot[g] for g in ground) + 0.10
            thw = radians(lay.u(11.5, 14.5))
            az = 90.0 + side * lay.u(2.0, 9.0)
            fy = -(0.055 * bs + L * sin(thw) * sin(radians(az)))
            b = kit.finish(mats, loc=(xb, fy, PR * sin(thw)), rot=(0, degrees(thw), az))

            def flat(v):
                v.y = min(v.y, -0.004)
                return v
            for o in b.values():
                transform_verts(o, flat)
        else:
            if pose == "rest":
                th, beta, tip = plan["th"], plan["beta"], plan["tip"]
                lift = info["crook"](L - plan["sc"])[0] * cos(th)     # the pole is a little crooked where it bears on the rim
                th = asin(clamp((plan["hc"] + lift - h0) / plan["sc"], 0.05, 0.6))
            else:                                        # nothing to lean on: flat on the paving in front
                th = asin((PR - h0) / L)
                beta = (180.0 if d1 < 0.5 else 0.0) + lay.jitter(0.0, 10.0)
                ext = min([pos[g][1] - foot[g] for g in ground] + [0.0])
                tip = np.array((0.0, ext - 0.16)) + unit(beta) * L * 0.5
            hd = unit(beta)
            org = tip - hd * L * cos(th)
            b = kit.finish(mats, loc=(org[0], org[1], h0 + L * sin(th)), rot=(0, 90.0 + degrees(th), beta), seat=False)

            def lie(v):                                  # twigs bend flat where the head lies on the cobbles
                v.z = max(v.z, 0.004)
                return v
            transform_verts(b["twig"], lie)
        built["broom"] = b

    # ---- turn the cluster, centre the footprint on the origin (or tuck it against the wall plane y = 0)
    for d_ in built.values():
        allobjs += list(d_.values())
    if abs(p.turn) > 0.01 and not wallf:
        for o in allobjs:
            place(o, rot=(0, 0, p.turn))
            apply_transform(o)
    mn, mx = world_bbox(allobjs)
    dy = -0.5 * (mn.y + mx.y)
    if pose == "wall":
        dy = 0.0
    elif wallm:
        dy = -0.02 - mx.y
    for o in allobjs:
        place(o, loc=(-0.5 * (mn.x + mx.x), dy, 0.0))
        apply_transform(o)

    # ---- damp, dirty feet where things stand on the paving (the rest is painted in each object's own frame)
    for g in ("zinc", "can"):
        if g in built:
            for key in ("zinc", "body"):
                if key in built[g]:
                    paint(built[g][key], grime=mask_mul(height_ramp(0.0, 0.05), 0.25 + 0.35 * A))
