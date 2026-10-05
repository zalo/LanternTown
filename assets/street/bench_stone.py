"""Dressed limestone bench for the square: a thick moulded seat slab (two stones with a mortar joint on
long benches) bedded on carved supports - Ionic scroll consoles with relieved volutes, lion-paw monopodia
or plain moulded piers - each standing on a rough footing stone. Optional wedge-section back slab with
sunk fielded panels and a saddle coping, buttressed by scrolled end piers and tied with wrought-iron
cramps that weep rust down the stone. Every stone is its own block (own tint, own texture patch) bedded
in thin lime-mortar joints; arrises are chipped (real conchoidal cuts with a rough interior), the seat
is dished and polished where people sit, rain has streaked the back below the coping, the feet stand in
a dark green splash zone and moss packs the bed joints.
"""
import math

import bmesh
import numpy as np
from mathutils import Vector

from tk import *

META = {"name": "Stone bench", "class": "furniture", "mount": "ground", "focus": [-0.5, -0.12, 0.42]}
PARAMS = {
    "length": {"default": 1.8, "min": 1.2, "max": 2.8, "desc": "overall length (m); > 2.15 m gets a third support and a two-stone seat"},
    "seat_depth": {"default": 0.48, "min": 0.4, "max": 0.6, "desc": "clear seat depth (m)"},
    "seat_height": {"default": 0.46, "min": 0.4, "max": 0.52, "desc": "height of the seat top (m)"},
    "seat_thick": {"default": 0.11, "min": 0.08, "max": 0.16, "desc": "thickness of the seat slab (m)"},
    "back": {"default": True, "desc": "backrest slab with panel moulding and coping"},
    "back_height": {"default": 0.42, "min": 0.3, "max": 0.6, "desc": "height of the back slab above the seat (m)"},
    "arms": {"default": True, "desc": "scrolled end piers buttressing the back (only with back)"},
    "support": {"default": "scroll", "choices": ["scroll", "lion_paw", "block"], "desc": "carved support style"},
    "stone": {"default": "#cfc5b0", "desc": "stone tint"},
    "moss": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "damp: green film on shaded faces (from ~0.1), moss cushions in the joints (from ~0.3)"},
    "wear": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "dishing and polish of the seat"},
    "chips": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of chipped arrises"},
    "inscription": {"default": "", "desc": "lead-letter inscription (centre back panel, or seat edge / seat top without back)"},
    "candles": {"default": 0, "min": 0, "max": 3, "desc": "votive candle stubs left burning on the seat"},
}
VARIANTS = {
    "backless_paw": {"back": False, "support": "lion_paw", "length": 1.5, "stone": "#b9b7ab", "moss": 0.25,
                     "seat_thick": 0.13, "inscription": "PAX VOBISCVM", "seed": 4},
    "long_memorial": {"length": 2.7, "inscription": "IN MEMORIAM", "stone": "#d6bf9a", "back_height": 0.5,
                      "moss": 0.3, "chips": 0.35, "candles": 2, "seed": 9},
    "plain_mossy": {"back": False, "support": "block", "length": 1.25, "stone": "#a3a69a", "moss": 1.0, "chips": 1.0,
                    "wear": 1.0, "seat_depth": 0.42, "seed": 17},
    "paw_backed": {"support": "lion_paw", "arms": False, "length": 2.05, "stone": "#c4ad98", "back_height": 0.36,
                   "moss": 0.6, "seed": 23},
}

REC = 0.007                                    # depth of the sunk channel around the carved volutes
LISTEL = [(-0.0046, -0.0012), (0.0046, -0.0012), (0.0036, REC), (-0.0036, REC)]


def _shell(name, nu, nv, nw, fn, angle=42):
    """Closed box-topology shell: fn(i, j, k) -> (x, y, z) on the surface lattice (shared vertices)."""
    idx, verts, faces = {}, [], []

    def vid(i, j, k):
        key = (i, j, k)
        if key not in idx:
            idx[key] = len(verts)
            verts.append(tuple(fn(i, j, k)))
        return idx[key]

    def quad(a, b, c, d, flip):
        faces.append((a, d, c, b) if flip else (a, b, c, d))

    for i in range(nu):
        for j in range(nv):
            quad(vid(i, j, nw), vid(i + 1, j, nw), vid(i + 1, j + 1, nw), vid(i, j + 1, nw), False)
            quad(vid(i, j, 0), vid(i + 1, j, 0), vid(i + 1, j + 1, 0), vid(i, j + 1, 0), True)
    for j in range(nv):
        for k in range(nw):
            quad(vid(nu, j, k), vid(nu, j + 1, k), vid(nu, j + 1, k + 1), vid(nu, j, k + 1), False)
            quad(vid(0, j, k), vid(0, j + 1, k), vid(0, j + 1, k + 1), vid(0, j, k + 1), True)
    for k in range(nw):
        for i in range(nu):
            quad(vid(i, nv, k), vid(i, nv, k + 1), vid(i + 1, nv, k + 1), vid(i + 1, nv, k), False)
            quad(vid(i, 0, k), vid(i, 0, k + 1), vid(i + 1, 0, k + 1), vid(i + 1, 0, k), True)
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    uv_box(o)
    smooth(o, angle)
    return o


def _strip(name, nx, nr, fn, flip=False):
    """Open lattice strip (stain decals): fn(i, r) -> (x, y, z); zero-area cells are dropped."""
    verts = [tuple(fn(i, r)) for i in range(nx) for r in range(nr)]
    faces = []
    for i in range(nx - 1):
        for r in range(nr - 1):
            q = [i * nr + r, (i + 1) * nr + r, (i + 1) * nr + r + 1, i * nr + r + 1]
            pts = []
            for a in q:
                if all((Vector(verts[a]) - Vector(verts[b])).length > 1e-5 for b in pts):
                    pts.append(a)
            if len(pts) < 3:
                continue
            faces.append(tuple(reversed(pts)) if flip else tuple(pts))
    o = mesh_object(name, verts, faces)
    uv_box(o)
    smooth(o, 60)
    return o


def _hexa(name, lo, hi):
    """Six-faced block from two quads: lo / hi = 4 corners each (same winding)."""
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    o = mesh_object(name, [tuple(v) for v in lo + hi], faces)
    recalc_normals(o)
    uv_box(o)
    return o


def _drop_faces(obj, pred):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if pred(f.normal)], context='FACES')
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def _tris(objs):
    return sum(sum(len(pl.vertices) - 2 for pl in o.data.polygons) for o in objs if o is not None and o.type == 'MESH')


def _chip(target, specs, r, m_a, m_b, out):
    """Knock conchoidal chips out of arrises: specs = [(point on arris, outward dir, edge axis 0/1/2)].
    The cut faces are split off into their own object (appended to `out`) so the fracture gets a rough,
    darker interior with a crisp rim."""
    cut = []
    for pt, outv, ax in specs:
        rad = r.u(0.024, 0.046)
        d = rad * r.u(0.3, 0.46)
        sc = [1.0, 1.0, 1.0]
        sc[ax] = r.u(1.2, 1.9)
        c = Vector(pt) + Vector(outv).normalized() * (rad - d)
        s = sphere("chipc", rad, loc=tuple(c), segments=8, rings=5, scale=tuple(sc))
        place(s, rot=(r.u(-25, 25), r.u(-25, 25), r.u(0, 60)))
        cut.append(s)
    if not cut:
        return target
    cutter = join(cut, "chipcut") if len(cut) > 1 else cut[0]
    for me in (cutter.data, target.data):
        me.materials.clear()
        me.materials.append(m_a)
        me.materials.append(m_b)
    for pl in cutter.data.polygons:
        pl.material_index = 1
    for pl in target.data.polygons:
        pl.material_index = 0
    boolean(target, cutter)
    bm = bmesh.new()
    bm.from_mesh(target.data)
    cf = [f for f in bm.faces if f.material_index == 1]
    if cf:
        vmap, V, F = {}, [], []
        for f in cf:
            ids = []
            for v in f.verts:
                if v.index not in vmap:
                    vmap[v.index] = len(V)
                    V.append(tuple(v.co))
                ids.append(vmap[v.index])
            F.append(tuple(ids))
        bmesh.ops.delete(bm, geom=cf, context='FACES')
        bm.to_mesh(target.data)
        co = mesh_object(target.name + "_chips", V, F)
        uv_box(co)
        smooth(co, 60)
        out.append(co)
    bm.free()
    return target


def _lines(breaks, lo, hi, step):
    """Sorted grid lines lo..hi containing the break positions, gaps filled to <= step."""
    b = sorted([lo, hi] + [x for x in breaks if lo + 0.002 < x < hi - 0.002])
    out = [b[0]]
    for x in b[1:]:
        gap = x - out[-1]
        if gap < 0.0015:
            continue
        n = int(math.ceil(gap / step)) if gap > step * 1.25 else 1
        prev = out[-1]
        for q in range(1, n + 1):
            out.append(prev + gap * q / n)
    return out


def _lines_rand(breaks, lo, hi, r, step):
    """Like _lines but the fill lines are irregularly spaced; returns (lines, is_fill flags)."""
    b = sorted([lo, hi] + [x for x in breaks if lo + 0.002 < x < hi - 0.002])
    out, fill = [b[0]], [False]
    for x in b[1:]:
        gap = x - out[-1]
        if gap < 0.0015:
            continue
        n = int(math.ceil(gap / step)) if gap > step * 1.3 else 1
        prev = out[-1]
        for q in range(1, n):
            out.append(prev + gap * (q + r.u(-0.28, 0.28)) / n)
            fill.append(True)
        out.append(x)
        fill.append(False)
    return out, fill


def _lead_text(body, cap, loc, rot, maxw, spacing=1.08):
    """Cast-lead letters let into the stone: thin raised text scaled to a true cap height `cap` (m),
    back faces dropped, outlines simplified."""
    tx = text("inscription", body, size=cap * 1.4, depth=0.0025, loc=loc, rot=rot, font="roman", spacing=spacing)
    bm = bmesh.new()
    bm.from_mesh(tx.data)
    bm.normal_update()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z < -0.5], context='FACES')
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(11), verts=bm.verts[:], edges=bm.edges[:], delimit={'NORMAL'})
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(tx.data)
    bm.free()
    xs = [v.co.x for v in tx.data.vertices]
    ys = [v.co.y for v in tx.data.vertices]
    k = cap / max(1e-6, max(ys) - min(ys))
    kx = min(k, maxw / max(1e-6, max(xs) - min(xs)))
    k = max(kx, k * 0.9) if kx < k else k      # long texts are condensed a little, never shrunk below 90 % height
    place(tx, scale=(kx, k, 1.0))
    print("DBG letters cap %.3f width %.2f" % (k * (max(ys) - min(ys)), kx * (max(xs) - min(xs))))
    return tx


def _strat(r, a, b, n, margin=0.05):
    """n stratified random positions in [a, b] (never closer than ~margin)."""
    if n <= 0 or b - a < 2 * margin:
        return []
    w = (b - a) / n
    return [a + w * (k + r.u(0.25, 0.75)) for k in range(n)]


def _volute(x, cy, cz, R, sgn, turns=2.0, hold=0.5, r_end=0.2):
    """Volute fillet in the YZ plane: starts at the top of the roll, follows its outline for `hold` turns,
    then winds in to the eye. Points are spaced by arc length so the tight inner turns stay cheap."""
    pts, th, tot = [], 0.0, 2 * math.pi * turns
    while True:
        f = clamp((th / tot - hold / turns) / (1 - hold / turns))
        rr = R * (1 - (1 - r_end) * f ** 0.9)
        a = math.pi / 2 + sgn * th
        pts.append((x, cy + rr * math.cos(a), cz + rr * math.sin(a)))
        if th >= tot:
            break
        th = min(tot, th + clamp(0.016 / rr, math.radians(15), math.radians(36)))
    return pts


def _patera(name, rad, petals=6, hk=1.0):
    o = lathe(name, [(rad, 0.0), (rad * 0.96, 0.007 * hk), (rad * 0.6, 0.0055 * hk), (rad * 0.3, 0.012 * hk), (0.0, 0.015 * hk)],
              segments=petals * 3)

    def lobes(co):
        rr = math.hypot(co.x, co.y)
        if rr < rad * 0.4:
            return co
        a = math.atan2(co.y, co.x)
        f = smoothstep(rad * 0.4, rad * 0.7, rr)
        c = abs(math.cos(a * petals * 0.5))
        k = 1.0 - 0.16 * f * (1 - c)
        return Vector((co.x * k, co.y * k, co.z * (1.0 - 0.45 * f * (1 - c))))

    transform_verts(o, lobes)
    return o


def _hex_mul(h, f):
    h = h.lstrip("#")
    return "#" + "".join("%02x" % int(clamp(int(h[i * 2:i * 2 + 2], 16) * f[i], 0, 255)) for i in range(3))


def _cramp_outline(length, width, vertical=False, flare=1.45):
    l, w, n = length / 2, width / 2, 0.024
    pts = [(-l, -w * flare), (-l + n, -w), (l - n, -w), (l, -w * flare), (l, w * flare), (l - n, w), (-l + n, w), (-l, w * flare)]
    return [(-y, x) for (x, y) in pts] if vertical else pts


def build(p, rng):
    L, D, ts, SH = p.length, p.seat_depth, p.seat_thick, p.seat_height
    has_back = bool(p.back)
    has_arms = has_back and bool(p.arms)
    TB0, TB1 = 0.12, 0.085                     # back slab thickness at foot / top (front face raked)
    Dt = D + (TB0 if has_back else 0.0)        # total slab depth
    yr = Dt / 2
    yfr = yr - TB0                             # foot of the back's front face
    ZF, JT = 0.035, 0.006                      # footing stone height, bed joint
    H = SH - ts - JT - ZF                      # support height
    z_seat0 = SH - ts
    W = min(0.21, Dt / 2 - 0.04)               # support half depth
    tsup = clamp(0.085 * L, 0.14, 0.19)        # support thickness along the bench
    nsup = 2 if L <= 2.15 else 3
    inset = clamp(0.13 * L, 0.2, 0.32) + (0.04 if has_arms else 0.0)
    sup_x = [-(L / 2 - inset), L / 2 - inset] if nsup == 2 else [-(L / 2 - inset), 0.0, L / 2 - inset]
    hb = p.back_height
    HC = 0.06                                  # coping height
    TA = 0.115                                 # arm / end pier thickness
    JV = 0.0035                                # upright mortar joint between back slab and end piers
    arm_x = L / 2 - 0.035 - TA / 2
    seed = int(p.seed)

    # one limestone, three slightly different blocks of it; fine isotropic grain (no tooling streaks)
    st = [mat("mortar", p.stone), mat("mortar", _hex_mul(p.stone, (1.05, 1.035, 1.0))),
          mat("mortar", _hex_mul(p.stone, (0.945, 0.95, 0.97)))]
    rough_m = mat("stone_rough", p.stone, rough=1.1)       # footings, fracture faces, joint mortar
    moss_m = mat("moss", grime=0.0)
    iron = mat("mortar", "#93552f", grime=0.6)    # deeply rusted wrought iron: matt rust-brown scale, no clean metal left
    lead = mat("lead")
    rust_m = mat("mortar", _hex_mul(p.stone, (0.86, 0.63, 0.42)))

    blocks = []                                # (objects, tint index, value shift -1..1) per separate stone
    chips, feet, irons, joints, stains = [], [], [], [], []
    moss_runs = []                             # (p0, p1, A, B, weight, kind, wmax, hmax)
    amt = clamp((p.moss - 0.28) / 0.72) ** 0.75            # moss cushions: none below ~0.3, joints packed at 1
    moss_over = clamp((p.moss - 0.5) * 2)
    rv = rng.fork("values")

    # ------------------------------------------------------------------ supports
    def support_rows():
        if p.support == "scroll":
            Rs = max(0.03, min(0.064, (H - 0.14) / 2, W * 0.36))
            zc, yc = H - 0.03 - Rs, W - Rs
            neck = yc - 0.014
            wp = W - 0.008
            rows = [(0.0, wp, 0.0, 0.0), (0.042, wp, 0.0, 0.0), (0.05, wp - 0.006, 0.0, 0.0)]
            for t in (0.25, 0.5, 0.75, 1.0):
                rows.append((0.05 + 0.04 * t, neck + (wp - 0.006 - neck) * (1 - math.sqrt(1 - (1 - t) ** 2)), 0.0, REC * t))
            rows.append((zc - Rs - 0.005, neck, 0.0, REC))
            for k in range(13):
                th = math.radians(-90 + 15 * k)
                rows.append((zc + Rs * math.sin(th), yc + Rs * math.cos(th), 0.0055 * math.cos(th) ** 0.6 if 0 < k < 12 else 0.0, REC))
            rows += [(zc + Rs + 0.005, W - 0.02, 0.0, REC), (zc + Rs + 0.0085, W - 0.02, 0.0, 0.0), (H, W - 0.02, 0.0, 0.0)]
            return rows, (Rs, zc, yc)
        if p.support == "lion_paw":
            wc = W - 0.135
            rows = [(0.0, wc + 0.02), (0.035, wc + 0.02), (0.045, wc + 0.012), (0.06, wc), (H - 0.075, wc),
                    (H - 0.055, wc + 0.012), (H - 0.04, wc + 0.035), (H - 0.031, wc + 0.05), (H - 0.026, W - 0.045),
                    (H, W - 0.045)]
            return [(z, w, 0.0, 0.0) for z, w in rows], None
        wd = W - 0.04
        rows = [(0.0, W - 0.01), (0.05, W - 0.01), (0.058, W - 0.018), (0.075, wd), (H * 0.5, wd), (H - 0.085, wd)]
        for t in (0.3, 0.6, 0.85, 1.0):
            rows.append((H - 0.085 + 0.04 * t, wd + 0.025 * (1 - math.sqrt(1 - t * t))))
        rows += [(H - 0.04, W - 0.012), (H, W - 0.012)]
        return [(z, w, 0.0, 0.0) for z, w in rows], None

    rows, vol = support_rows()
    w_top = rows[-1][1]
    w_bot = rows[0][1]
    paw_reach = (W - 0.03 + 0.03) if p.support == "lion_paw" else w_bot
    nxs, nys = (6 if p.support == "scroll" else 3), 4
    sup_objs = []
    for si, sx in enumerate(sup_x):
        r = rng.fork(f"sup{si}")

        def fn(i, j, k, sx=sx):
            z, w, amp, rec = rows[k]
            y = (2 * j / nys - 1) * w
            if amp and j in (0, nys):
                y += (1 if j else -1) * amp * abs(math.sin(3 * math.pi * i / nxs))
            return (sx + (tsup / 2 - rec) * (2 * i / nxs - 1), y, ZF + z)

        s = _shell(f"support{si}", nxs, nys, len(rows) - 1, fn)
        bevel(s, 0.004, segments=1, angle=50)
        displace_noise(s, strength=0.001, scale=0.12, seed=seed + si)
        nchip = int(round(p.chips * r.u(1.5, 3.4)))
        specs = []
        for _ in range(nchip):
            ex, ey = r.pick([-1, 1]), r.pick([-1, 1])
            if r.chance(0.6):                  # plinth corners and top arris get kicked
                specs.append(((sx + ex * tsup / 2, ey * w_bot, ZF + r.u(0.012, 0.045)), (ex, ey, 0), 2))
            else:
                specs.append(((sx + ex * tsup / 2, r.u(-0.6, 0.6) * w_top, ZF + H - 0.012), (ex, 0, 0.2), 1))
        _chip(s, specs[:2] if len(specs) > 2 and abs(specs[0][0][2] - specs[1][0][2]) > 0.02 else specs[:1], r, st[0], rough_m, chips)
        sup_objs.append(s)
        # rough footing stone and mortar bed
        fj = (r.jitter(0, 0.008), r.jitter(0, 0.008), r.jitter(0, 1.5))
        fhx, fhy = tsup / 2 + 0.05, paw_reach + 0.05
        f = stone_block(f"footing{si}", (2 * fhx, 2 * fhy, ZF + 0.004), loc=(sx + fj[0], fj[1], ZF / 2 + 0.002),
                        rot=(0, 0, fj[2]), rng=r, chamfer=0.01, rough=0.004, detail=0.14)
        apply_transform(f)
        transform_verts(f, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
        feet.append(f)
        joints.append(box(f"bed{si}", (tsup - 0.005, 2 * w_top - 0.005, JT + 0.004), loc=(sx, 0, ZF + H + JT / 2)))
        # joint lines where moss gathers: under the seat, round the plinth, round the footing on the paving
        ca, sa = math.cos(math.radians(fj[2])), math.sin(math.radians(fj[2]))
        fc = Vector((sx + fj[0], fj[1], 0.0))
        for (ux, uy, hx_t, hy_t, hx_b, hy_b, wgt) in ((1, 0, tsup / 2, w_top, tsup / 2, w_bot, 0.8), (-1, 0, tsup / 2, w_top, tsup / 2, w_bot, 0.8),
                                                    (0, 1, w_top, tsup / 2, w_bot, tsup / 2, 1.25), (0, -1, w_top, tsup / 2, w_bot, tsup / 2, 0.6)):
            out = Vector((ux, uy, 0))
            tan = Vector((-uy, ux, 0))
            pc = Vector((sx, 0, 0))
            is_side = ux != 0
            zt = ZF + H + JT
            moss_runs.append((pc + out * (hx_t - 0.001) - tan * (hy_t - 0.01) + Vector((0, 0, zt)),
                              pc + out * (hx_t - 0.001) + tan * (hy_t - 0.01) + Vector((0, 0, zt)), out, (0, 0, -1), wgt * 0.8, "corner", 0.016, 0.012))
            if is_side or p.support != "lion_paw":
                moss_runs.append((pc + out * (hx_b - 0.002) - tan * (hy_b - 0.006) + Vector((0, 0, ZF + 0.003)),
                                  pc + out * (hx_b - 0.002) + tan * (hy_b - 0.006) + Vector((0, 0, ZF + 0.003)), out, (0, 0, 1), wgt, "corner", 0.02, 0.014))
            fo = Vector((ux * ca - uy * sa, ux * sa + uy * ca, 0))
            ft = Vector((-fo.y, fo.x, 0))
            fa, fb = (fhx, fhy) if is_side else (fhy, fhx)
            moss_runs.append((fc + fo * (fa - 0.007) - ft * (fb - 0.012) + Vector((0, 0, 0.001)),
                              fc + fo * (fa - 0.007) + ft * (fb - 0.012) + Vector((0, 0, 0.001)), fo, (0, 0, 1), wgt * 1.25, "corner", 0.03, 0.02))

    # carved enrichment of the supports
    if p.support == "scroll":
        Rs, zc, yc = vol
        Rv_ = Rs - 0.0065
        x0 = tsup / 2 - REC
        path = list(reversed(_volute(x0, -yc, ZF + zc, Rv_, 1))) + _volute(x0, yc, ZF + zc, Rv_, -1)
        rel = sweep("vol_relief", path, LISTEL, up=(1, 0, 0), smooth_path=False)
        _drop_faces(rel, lambda n: n.x < -0.5)
        parts = [rel]
        for ey in (-1, 1):
            parts.append(lathe("vol_eye", [(0.0125, 0.0), (0.0115, 0.006), (0.007, 0.009), (0.0, 0.0105)], segments=8,
                               loc=(x0, ey * yc, ZF + zc), rot=(0, 90, 0)))
        if 2 * (yc - Rs) > 0.075:
            pt = _patera("patera", min(0.03, yc - Rs - 0.004), hk=0.62)
            place(pt, loc=(x0, 0, ZF + zc - 0.012), rot=(0, 90, 0))
            parts.append(pt)
        base = join(parts, "relief_base")
        mats = []
        for sx in sup_x:
            mats += [trs((sx, 0, 0)), trs((sx, 0, 0), (0, 0, 180))]
        sup_objs.append(instances_to_mesh("support_carving", base, mats, keep_base=False, uv_jitter=False))
    elif p.support == "lion_paw":
        Wp = W - 0.03
        lt = tsup * 0.8
        yk = -(Wp - 0.062)
        legp = [(0, -(Wp - 0.078), 0.05), (0, -(Wp - 0.086), 0.095), (0, -(Wp - 0.07), H * 0.5), (0, yk, H - 0.105),
                (0, -(Wp - 0.07), H - 0.045), (0, -(Wp - 0.075), H - 0.024)]

        def lsc(t):
            return 0.74 + 0.42 * smoothstep(0.05, 0.72, t) - 0.2 * smoothstep(0.78, 1.0, t) + 0.2 * (1 - smoothstep(0.0, 0.12, t))

        leg = sweep("leg", legp, rounded_rect(0.09, lt, 0.034, 3), up=(1, 0, 0), step=0.03, scale_fn=lsc)
        smooth(leg, 60)
        parts = [leg]
        parts.append(sphere("paw", 0.05, loc=(0, -(Wp - 0.058), 0.034), segments=10, rings=6, scale=(lt * 0.56 / 0.05, 1.05, 0.72)))
        for tf in (-1.5, -0.5, 0.5, 1.5):
            x = tf * lt * 0.245
            yt = -(Wp - 0.012) + abs(tf) * 0.007
            parts.append(sphere("toe", 0.02, loc=(x * 1.08, yt, 0.019), segments=8, rings=5, scale=(0.92, 1.65, 0.95)))
            parts.append(sphere("knuckle", 0.02, loc=(x, yt + 0.034, 0.041), segments=8, rings=5, scale=(0.95, 1.3, 0.95)))
            parts.append(cone("claw", 0.0085, 0.002, 0.026, loc=(x * 1.1, yt - 0.03, 0.0125), rot=(112, 0, -tf * 5), verts=6))
        # leaf-carved knee: raised acanthus lappet on the front of the thigh and a collar under the abacus
        knee = surface("lappet", lambda u, v: ((u - 0.5) * lt * 0.74 * (1 - v ** 2.2 * 0.92),
                                               yk - 0.056 + 0.03 * v ** 2 + 0.012 * (2 * abs(u - 0.5)) ** 2 - 0.004 * math.cos((u - 0.5) * 6 * math.pi) * (1 - v),
                                               H - 0.05 - v * 0.12), 8, 6, thickness=0.007)
        parts.append(knee)
        parts.append(box("collar", (lt * 1.04, 0.105, 0.02), loc=(0, -(Wp - 0.072), H - 0.036), bevel=0.006))
        base = join(parts, "leg_base")
        mats = []
        for sx in sup_x:
            mats += [trs((sx, 0, ZF)), trs((sx, 0, ZF), (0, 0, 180))]
        sup_objs.append(instances_to_mesh("lion_legs", base, mats, keep_base=False))
        # sunk roundel on the pier sides
        pt = _patera("patera", 0.036, 8)
        place(pt, loc=(tsup / 2 - 0.001, 0, ZF + H * 0.52), rot=(0, 90, 0))
        mats = []
        for sx in sup_x:
            mats += [trs((sx, 0, 0)), trs((sx, 0, 0), (0, 0, 180))]
        sup_objs.append(instances_to_mesh("pier_paterae", realize(pt), mats, keep_base=False))
    else:
        pt = _patera("patera", min(0.05, tsup * 0.3), 6)
        place(pt, loc=(0, -(W - 0.04) + 0.001, ZF + H * 0.5), rot=(90, 0, 0))
        mats = []
        for sx in sup_x:
            mats += [trs((sx, 0, 0)), trs((sx, 0, 0), (0, 0, 180))]
        sup_objs.append(instances_to_mesh("pier_paterae", realize(pt), mats, keep_base=False))
    blocks.append((sup_objs, 2, rv.u(-0.6, 0.6)))
    print("DBG tris supports", _tris(sup_objs), "footings", _tris(feet))

    # ------------------------------------------------------------------ seat slab(s)
    nspot = max(2, int(round((L - (0.3 if has_arms else 0.1)) / 0.6)))
    span = L - (0.56 if has_arms else 0.36)
    spots = [(-span / 2 + span * k / (nspot - 1)) for k in range(nspot)]
    y_sit = -Dt / 2 + D * 0.48
    DISH = 0.016 * p.wear                     # hollow worn into the seat at each sitting place
    srow = [(0.0, 0.03), (0.016, 0.03), (0.022, 0.0265), (0.033, 0.0165), (0.044, 0.005), (0.05, 0.0), (ts - 0.02, 0.0),
            (ts - 0.011, 0.0025), (ts - 0.004, 0.0085), (ts, 0.018)]
    slabs = [(-L / 2, L / 2, True, True)] if nsup == 2 else [(-L / 2, -0.0025, True, False), (0.0025, L / 2, False, True)]
    ybrk = [y_sit - 0.16, y_sit]
    if has_back:
        ybrk += [yfr - 0.11, yfr - 0.05, yfr - 0.01, yfr]
    seat_objs = []
    for qi, (xa, xb, ma, mb) in enumerate(slabs):
        r = rng.fork(f"seat{qi}")
        xbrk = list(spots)
        if has_arms:
            for sg in (-1, 1):
                xi_ = sg * (arm_x - TA / 2)
                xbrk += [xi_ - sg * 0.01, xi_, sg * (arm_x + TA / 2)]
        xl = _lines(xbrk, xa, xb, 0.1 if nsup == 2 else 0.13)
        yl = _lines(ybrk, -Dt / 2, Dt / 2, 0.1)
        dz = r.jitter(0, 0.0012) if nsup == 3 else 0.0

        def fn(i, j, k, xa=xa, xb=xb, ma=ma, mb=mb, xl=xl, yl=yl, dz=dz):
            z, ins = srow[k]
            x = lerp(xa + (ins if ma else 0.0), xb - (ins if mb else 0.0), (xl[i] - xa) / (xb - xa))
            y = lerp(-Dt / 2 + ins, Dt / 2 - ins * (0.2 if has_back else 1.0), (yl[j] + Dt / 2) / Dt)
            if k == len(srow) - 1:
                for sxp in spots:
                    z -= DISH * math.exp(-((x - sxp) / 0.2) ** 2 - ((y - y_sit) / 0.15) ** 2)
            return (x, y, z_seat0 + z + dz)

        s = _shell(f"seat{qi}", len(xl) - 1, len(yl) - 1, len(srow) - 1, fn)
        displace_noise(s, strength=0.001, scale=0.25, seed=seed + 40 + qi)
        specs = []
        for x in _strat(r, xa + 0.08, xb - 0.08, int(round(p.chips * (xb - xa) * 2.0))):
            specs.append(((x, -Dt / 2, SH - 0.004), (0, -1, 1), 0))
        for x in _strat(r, xa + 0.1, xb - 0.1, int(round(p.chips * (xb - xa) * 1.3))):
            specs.append(((x + 0.03, -Dt / 2, z_seat0 + 0.05), (0, -1, -0.8), 0))
        if not has_back:
            for x in _strat(r, xa + 0.1, xb - 0.1, int(round(p.chips * (xb - xa) * 1.4))):
                specs.append(((x, Dt / 2, SH - 0.004), (0, 1, 1), 0))
        for (xe, m, sg) in ((xa, ma, -1), (xb, mb, 1)):
            if m and r.chance(0.35 + 0.6 * p.chips) and p.chips > 0.05:
                specs.append(((xe, r.u(-0.3, 0.1) * Dt, SH - 0.004 if not has_arms else z_seat0 + 0.05), (sg, 0, 1 if not has_arms else -0.8), 1))
        _chip(s, specs, r, st[0], rough_m, chips)
        seat_objs.append(s)
        blocks.append(([s], qi % 2, rv.u(-1.0, -0.2) if qi == 0 else rv.u(0.0, 0.8)))
    if nsup == 3:
        joints.append(box("seat_joint", (0.009, Dt - 0.04, ts - 0.03), loc=(0, 0, z_seat0 + ts / 2 - 0.003)))
        moss_runs.append(((0, -Dt / 2 + 0.03, SH - 0.004), (0, (yfr if has_back else Dt / 2) - 0.03, SH - 0.004), (1, 0, 0), (0, 0, 1), 1.1, "dome", 0.011, 0.008))
    print("DBG tris seat", _tris(seat_objs))

    # ------------------------------------------------------------------ back slab, coping, end piers
    z_b0 = SH + 0.003
    z_bt = z_b0 + hb                           # top of the back slab = underside of the coping
    rake = TB0 - TB1
    pz0 = pz1 = 0.0
    streak = None
    if has_back:
        r = rng.fork("back")
        Lb = 2 * (arm_x - TA / 2 - JV) if has_arms else (L - 0.09)
        npan = max(1, int(round((Lb - 0.1) / 0.8)))
        if p.inscription and npan % 2 == 0:
            npan += 1 if (Lb - 0.07 * (npan + 2)) / (npan + 1) > 0.34 else -1
        stile = 0.075
        pw = (Lb - stile * (npan + 1)) / npan
        pz0, pz1 = z_b0 + 0.075, z_b0 + hb - 0.055
        pans = [(-Lb / 2 + stile + k * (pw + stile), -Lb / 2 + stile + k * (pw + stile) + pw) for k in range(npan)]
        stile_c = [pans[k][1] + stile / 2 for k in range(npan - 1)]
        rx0, rx1, rz0, rz1 = pans[0][0], pans[-1][1], pz0, pz1      # rear panel shares the front grid lines
        PA, PB, PC = 0.006, 0.03, 0.037

        def brk(a0, a1):
            return [a0, a0 + PA, a0 + PB, a0 + PC, a1 - PC, a1 - PB, a1 - PA, a1]

        bx = []
        for (a0, a1) in pans:
            bx += brk(a0, a1)
        bx += [sg * (Lb / 2 - 0.009) for sg in (-1, 1)]
        xs, xfill = _lines_rand(bx, -Lb / 2, Lb / 2, r, 0.07 if L < 2.2 else 0.095)
        zs = _lines(brk(pz0, pz1) + [z_b0 + 0.01], z_b0, z_bt, 0.12)

        def fdepth(x, z):
            d = max(min(x - a0, a1 - x, z - pz0, pz1 - z) for (a0, a1) in pans)
            if d <= 0:
                return 0.0
            if d < PA:
                return 0.007 * d / PA
            if d <= PB:
                return 0.007
            return 0.007 + 0.009 * min(1.0, (d - PB) / (PC - PB))

        def y_front(z):
            return yfr + rake * (z - z_b0) / hb

        def y_rear(x, z):
            return yr - 0.009 * clamp(min(x - rx0, rx1 - x, z - rz0, rz1 - z) / PA)

        def bfn(i, j, k):
            x, z = xs[i], zs[k]
            return (x, lerp(y_front(z) + fdepth(x, z), y_rear(x, z), j), z)

        slab = _shell("back_slab", len(xs) - 1, 1, len(zs) - 1, bfn)
        bevel(slab, 0.0045, segments=1, angle=70)
        displace_noise(slab, strength=0.0006, scale=0.25, seed=seed + 50)
        if not has_arms:
            specs = [((sg * Lb / 2, yfr + rake * 0.5, z_b0 + hb * r.u(0.2, 0.8)), (sg, -1, 0), 2) for sg in (-1, 1) if r.chance(0.3 + 0.6 * p.chips)]
            if p.chips > 0.05:
                _chip(slab, specs, r, st[0], rough_m, chips)
        blocks.append(([slab], 2, rv.u(-0.3, 0.7)))
        print("DBG tris back", _tris([realize(slab)]))
        # rain run-off: every irregular lattice column carries its own streak strength / length (front, rear)
        rs = rng.fork("streaks")
        streak = (np.array(xs),
                  np.array([rs.u(0.35, 1.0) if (fl and rs.chance(0.55)) else 0.0 for fl in xfill]),
                  np.array([rs.u(0.12, 0.3) for _ in xs]),
                  np.array([rs.u(0.4, 1.0) if (fl and rs.chance(0.65)) else 0.0 for fl in xfill]),
                  np.array([rs.u(0.14, 0.36) for _ in xs]))
        joints.append(box("back_bed", (Lb - 0.004, TB0 - 0.006, 0.007), loc=(0, yr - TB0 / 2, SH + 0.0012)))
        moss_runs.append(((-Lb / 2 + 0.012, yfr + 0.001, SH + 0.0005), (Lb / 2 - 0.012, yfr + 0.001, SH + 0.0005), (0, -1, 0), (0, 0, 1), 1.25, "corner", 0.02, 0.014))
        moss_runs.append(((-Lb / 2 + 0.012, yr - 0.002, SH - 0.004), (Lb / 2 - 0.012, yr - 0.002, SH - 0.004), (0, 0, 1), (0, 1, 0), 0.42, "dome", 0.014, 0.016))

        # lead-letter inscription in the centre panel
        if p.inscription:
            size = min(0.06, (pz1 - pz0) * 0.22)
            phi = math.degrees(math.atan2(rake, hb))
            zt = (pz0 + pz1) / 2
            yt = y_front(zt) + 0.016
            tx = _lead_text(p.inscription, size, (0, yt, zt), (90 - phi, 0, 0), pw - 0.12)
            assign(tx, lead)
            paint(tx, grime=0.25)

        # saddle coping in one to three stones; the joints sit over the stiles and are tied with iron cramps
        Lc = Lb + (0.0 if has_arms else 0.05)
        ncop = 1 if Lc < 1.35 else (2 if Lc < 2.3 else 3)
        cy = yr - TB1 / 2
        cprof = [(-0.05, 0), (0.05, 0), (0.05, 0.012), (0.056, 0.018), (0.065, 0.024), (0.065, 0.036), (0.058, 0.048), (0.04, 0.057),
                 (0.015, HC), (-0.015, HC), (-0.04, 0.057), (-0.058, 0.048), (-0.065, 0.036), (-0.065, 0.024), (-0.056, 0.018), (-0.05, 0.012)]
        cuts, on_stile = [-Lc / 2], []
        for k in range(1, ncop):
            xk = -Lc / 2 + Lc * k / ncop
            near = min(stile_c, key=lambda c: abs(c - xk)) if stile_c else None
            if near is not None and abs(near - xk) < 0.3 * Lc / ncop:
                cuts.append(near + r.jitter(0, 0.008))
                on_stile.append(True)
            else:
                cuts.append(xk + r.jitter(0, 0.05) * Lc / ncop)
                on_stile.append(False)
        cuts.append(Lc / 2)
        for k in range(ncop):
            xa, xb = cuts[k] + (0.002 if k else 0), cuts[k + 1] - (0.002 if k < ncop - 1 else 0)
            n = max(2, int((xb - xa) / 0.16))
            dzk = r.jitter(0, 0.0012)
            c = sweep(f"coping{k}", [(lerp(xa, xb, i / n), cy, z_bt + dzk) for i in range(n + 1)], cprof, smooth_path=False)
            smooth(c, 35)
            displace_noise(c, strength=0.0008, scale=0.2, seed=seed + 60 + k)
            specs = []
            for x in _strat(r, xa + 0.09, xb - 0.09, int(round(p.chips * (xb - xa) * 1.6))):
                sg = r.pick([-1, 1])
                specs.append(((x, cy + sg * 0.067, z_bt + 0.03), (0, sg, 0.25), 0))
            _chip(c, specs, r, st[0], rough_m, chips)
            blocks.append(([c], (k + 1) % 2 if ncop > 1 else 1, rv.u(-0.8, 0.8)))

        def weep(name, xc, hw, z_top, lmax, yfn, rear, rw, ncol=11, zbrk=()):
            """Rust-stained stone below an iron fixing: thin ragged tongues running down the face, strongest
            under the iron and fading (wear mask -> pale) towards the tips and sides."""
            lens, prev = [], rw.u(0.3, 0.7)
            for i in range(ncol):
                u = 2 * i / (ncol - 1) - 1
                prev = clamp(prev + rw.u(-0.3, 0.3), 0.15, 0.8)
                lens.append(lmax * prev * (1 - 0.6 * u * u))
            for _ in range(2):                 # a couple of long runs where the water actually trickles
                i = rw.pick(list(range(2, ncol - 2)))
                lens[i] = lmax * rw.u(0.8, 1.0)
                lens[i - 1] = max(lens[i - 1], lens[i] * 0.55)
            lens[0] = lens[-1] = 0.004
            dzr = 0.02
            zrows = sorted(set([z_top - q * dzr for q in range(int(lmax / dzr) + 2)] + [z for z in zbrk if z_top - lmax < z < z_top]), reverse=True)
            xoff = [rw.jitter(0, 0.2) for _ in range(ncol)]

            def sfn(i, q):
                x = xc + hw * (2 * (i + xoff[i] * (0 < i < ncol - 1)) / (ncol - 1) - 1)
                z = max(zrows[q], z_top - lens[i])
                return (x, yfn(x, z) + (0.0016 if rear else -0.0016), z)

            o = _strip(name, ncol, len(zrows), sfn, flip=not rear)
            xa_ = np.array([xc + hw * (2 * (i + xoff[i] * (0 < i < ncol - 1)) / (ncol - 1) - 1) for i in range(ncol)])
            la_ = np.array(lens)

            def fade(co, n):
                ln = np.maximum(np.interp(co[:, 0], xa_, la_), 0.01)
                t = np.clip((z_top - co[:, 2]) / ln, 0, 1)
                side = np.clip(np.abs(co[:, 0] - xc) / hw, 0, 1)
                return np.clip(0.9 * t ** 1.3 + 0.4 * side ** 3, 0, 1)

            paint(o, wear=fade)
            stains.append(o)
            return o

        def cramp(name, loc, length, width, mode, rw):
            """Forged flat-bar cramp with fishtailed ends and two domed pins. mode: 'rear_h' | 'rear_v' | 'top'."""
            vert = mode == "rear_v"
            rot = (0, 0, 0) if mode == "top" else (-90, 0, 0)
            bar = extrude_profile(name, _cramp_outline(length, width, vert), 0.0065, loc=loc, rot=rot, bevel=0.0016)
            irons.append(bar)
            for sg in (-1, 1):
                off = sg * (length / 2 - 0.015)
                if mode == "top":
                    pl = (loc[0] + off, loc[1], loc[2] + 0.006)
                elif vert:
                    pl = (loc[0], loc[1] + 0.006, loc[2] + off)
                else:
                    pl = (loc[0] + off, loc[1] + 0.006, loc[2])
                irons.append(lathe(name + "_pin", [(0.009, 0.0), (0.0082, 0.003), (0.0052, 0.0052), (0.0, 0.006)], segments=8, loc=pl, rot=rot))

        ri = rng.fork("iron")
        for k in range(1, ncop):
            cxp = cuts[k]
            cramp("coping_cramp", (cxp, cy, z_bt + HC - 0.0025), 0.13, 0.021, "top", ri)
            for sg in (-1, 1):                 # rust runs down both flanks of the coping ...
                prof = [(0.006, HC), (0.016, HC), (0.04, 0.057), (0.058, 0.048), (0.065, 0.036), (0.065, 0.024)]
                wj = [ri.u(0.012, 0.022) for _ in range(len(prof))]

                def cfn(i, q, sg=sg, prof=prof, wj=wj, cxp=cxp):
                    a, b = prof[q]
                    dn = math.hypot(a, b - 0.026) or 1.0
                    a, b = a + 0.0016 * a / dn, b + 0.0016 * (b - 0.026) / dn
                    return (cxp + (i - 1) * wj[q] * (1.0 if q else 0.5), cy + sg * a, z_bt + b)

                o = _strip("coping_stain", 3, len(prof), cfn, flip=sg < 0)
                paint(o, wear=lambda co, n: 0.15 + 0.3 * np.clip((z_bt + HC - co[:, 2]) / 0.04, 0, 1))
                stains.append(o)
            # ... and on down the face of the slab below the joint
            lf = ri.u(0.11, 0.15) if on_stile[k - 1] else 0.045
            weep("stain_front", cxp + ri.jitter(0, 0.004), 0.024, z_bt - 0.001, min(lf, hb * 0.5), lambda x, z: y_front(z), False, ri, ncol=7)
            weep("stain_rear", cxp + ri.jitter(0, 0.004), 0.028, z_bt - 0.001, min(ri.u(0.12, 0.17), hb * 0.5), y_rear, True, ri, ncol=7,
                 zbrk=(rz1 + 0.0004, rz1 - PA - 0.0004))
        if has_arms:
            # cramps across the upright joints between back slab and end piers (rear face)
            for sg in (-1, 1):
                xj = sg * (Lb / 2 + JV / 2)
                zc_ = z_b0 + hb * ri.u(0.6, 0.7)
                cramp("back_cramp", (xj, yr - 0.0025, zc_), 0.135, 0.024, "rear_h", ri)
                weep("stain_cramp", xj, 0.07, zc_ + 0.004, min(ri.u(0.13, 0.17), zc_ - SH - 0.03), lambda x, z: yr, True, ri, ncol=13)
        else:
            # no piers: iron straps tie the foot of the back slab to the rear edge of the seat
            nst = 2 if Lb < 2.0 else 3
            for k in range(nst):
                xs_ = -Lb / 2 + Lb * (k + 0.5) / nst + ri.jitter(0, 0.05) if nst > 2 else (-1, 1)[k] * Lb * 0.3
                cramp("strap", (xs_, yr - 0.0025, SH + 0.012), 0.125, 0.026, "rear_v", ri)
                weep("stain_strap", xs_, 0.034, SH - 0.047, ts - 0.052,
                     lambda x, z: yr - 0.2 * float(np.interp(z - z_seat0, [a for a, _ in srow], [b for _, b in srow])), True, ri, ncol=7,
                     zbrk=[z_seat0 + a + 0.0003 for a, _ in srow])

        # scrolled end piers
        if has_arms:
            Rv = clamp(0.2 * D * 0.72, 0.055, 0.075)
            hv, hp = 2 * Rv, hb + HC + 0.028
            La = Dt - 0.075
            Ls = La - Rv
            d1, d2 = TB0 + 0.035, Ls - Rv * 0.45
            OFF = 0.014

            def hfun(d):
                return hv + (hp - hv) * (1 - smoothstep(d1, d2, d)) ** 1.25

            # depth columns of the pier lattice; two of them hug the raked face of the back slab so the
            # baked contact shadow in that corner stays a thin joint line instead of a wedge
            dl = [0.0, 0.045, TB1 - 0.004, "F", "F2", d1] + [lerp(d1, d2, q / 7) for q in range(1, 8)] + [(d2 + Ls) / 2, Ls]

            def dcol(j, zt):
                v = dl[j]
                if isinstance(v, str):
                    face = TB0 - rake * clamp(zt * hp / hb) + 0.005
                    return face if v == "F" else face + 0.016
                return v

            ns, nz, nxa = len(dl) - 1, 10, 3
            for ai, sg in enumerate((-1, 1)):
                ra = rng.fork(f"arm{ai}")
                xa_ = sg * arm_x

                def fn(i, j, k, xa_=xa_):
                    zt = 0.5 - 0.5 * math.cos(math.pi * k / nz)
                    d = dcol(j, zt)
                    s = d / Ls
                    y = yr - d - (s ** 4) * Rv * math.sqrt(max(0.0, 1 - (2 * zt - 1) ** 2))
                    z = zt * hfun(d)
                    if k == nz:
                        z -= 0.006 * (2 * i / nxa - 1) ** 2
                    return (xa_ + TA * (i / nxa - 0.5), y, z_b0 + z)

                a = _shell(f"arm{ai}", nxa, ns, nz, fn)
                bevel(a, 0.005, segments=1, angle=50)
                displace_noise(a, strength=0.001, scale=0.2, seed=seed + 80 + ai)
                specs = []
                if ra.chance(0.3 + 0.7 * p.chips) and p.chips > 0.05:
                    specs.append(((xa_ + sg * TA / 2, yr - ra.u(0.0, 0.1), z_b0 + hp - 0.003), (sg, 0, 1), 1))
                if ra.chance(0.2 + 0.6 * p.chips) and p.chips > 0.05:
                    specs.append(((xa_ + sg * TA / 2, yr, z_b0 + hp * ra.u(0.25, 0.5)), (sg, 1, 0), 2))
                _chip(a, specs, ra, st[0], rough_m, chips)
                grp = [a]
                # hidden shadow plate inside the pier, close to its outer face: the realtime shadow map then holds a
                # depth well clear of the inner face, so the pier's shadow stays attached to the back slab and
                # seat instead of leaving a lit sliver along the joint
                xpl = xa_ + sg * (TA / 2 - 0.014)
                dsp = [0.012, d1] + [lerp(d1, d2, q / 7) for q in range(1, 8)]
                pv = [(xpl, yr - d, z_b0 + 0.004) for d in dsp] + [(xpl, yr - d, z_b0 + hfun(d) - 0.025) for d in dsp]
                npl = len(dsp)
                pf = [(q, q + 1, npl + q + 1, npl + q) if sg > 0 else (q + 1, q, npl + q, npl + q + 1) for q in range(npl - 1)]
                grp.append(mesh_object(f"arm_shadow{ai}", pv, pf))
                # raised fillet following the swoop and winding into the volute, on both faces
                yv, zv = yr - Ls, z_b0 + Rv
                for fs in (-1, 1):
                    xf = xa_ + fs * (TA / 2 - 0.0008)
                    path = []
                    nd = 11
                    for q in range(nd):
                        d = lerp(0.03, Ls, q / nd)
                        e = 1e-3
                        hd = (hfun(d + e) - hfun(d - e)) / (2 * e)
                        nn = math.hypot(hd, 1.0)
                        path.append((xf, yr - d - OFF * hd / nn, z_b0 + hfun(d) - OFF / nn))
                    path += _volute(xf, yv, zv, Rv - OFF, 1)
                    rel = sweep(f"arm_relief{ai}{fs}", path, LISTEL, up=(fs, 0, 0), smooth_path=False)
                    _drop_faces(rel, lambda n, fs=fs: n.x * fs < -0.5)
                    grp.append(rel)
                    grp.append(lathe("arm_eye", [(0.0125, 0.0), (0.0115, 0.006), (0.007, 0.009), (0.0, 0.0105)], segments=8,
                                     loc=(xf, yv, zv), rot=(0, fs * 90, 0)))
                blocks.append((grp, 1 if ai == 0 else 0, rv.u(-0.7, 0.7)))
                joints.append(box(f"arm_bed{ai}", (TA - 0.005, Ls - 0.008, 0.007), loc=(xa_, yr - Ls / 2 - 0.001, SH + 0.0012)))
                # upright mortar joint between pier and back slab, finished 2 mm behind the faces
                xi = xa_ - sg * TA / 2
                xo_, xi_ = xi - sg * (JV + 0.003), xi + sg * 0.003
                x_lo, x_hi = min(xo_, xi_), max(xo_, xi_)
                lo = [(x_lo, y_front(z_b0) + 0.002, z_b0), (x_hi, y_front(z_b0) + 0.002, z_b0), (x_hi, yr - 0.002, z_b0), (x_lo, yr - 0.002, z_b0)]
                hi = [(x_lo, y_front(z_bt) + 0.002, z_bt - 0.002), (x_hi, y_front(z_bt) + 0.002, z_bt - 0.002), (x_hi, yr - 0.002, z_bt - 0.002),
                      (x_lo, yr - 0.002, z_bt - 0.002)]
                joints.append(_hexa(f"arm_joint{ai}", lo, hi))
                moss_runs.append(((xi, yfr - 0.012, SH + 0.0005), (xi, yr - Ls + 0.03, SH + 0.0005), (-sg, 0, 0), (0, 0, 1), 1.2, "corner", 0.018, 0.013))
                moss_runs.append(((xa_ + sg * TA / 2, yr - 0.03, SH + 0.0005), (xa_ + sg * TA / 2, yr - Ls + 0.03, SH + 0.0005), (sg, 0, 0), (0, 0, 1), 0.8, "corner", 0.013, 0.011))
    elif p.inscription:
        band = ts - 0.07
        if band - 0.012 >= 0.045:
            # backless: lead letters on the front edge of the seat
            tx = _lead_text(p.inscription, min(0.055, band - 0.012), (0, -Dt / 2 - 0.0003, z_seat0 + 0.05 + band / 2), (90, 0, 0), L * 0.72, spacing=1.14)
        else:
            # thin slab: the letters are let into the seat top along the rear edge instead
            tx = _lead_text(p.inscription, 0.05, (0, yr - 0.085, SH - 0.0003), (0, 0, 0), L * 0.72, spacing=1.14)
        assign(tx, lead)
        paint(tx, grime=0.25)

    # ------------------------------------------------------------------ materials and weathering
    for objs, ti, val in blocks:
        for o in objs:
            assign(o, st[ti])
    for o in feet + joints:
        assign(o, rough_m)
    for o in chips:
        assign(o, st[2])
    for o in stains:
        assign(o, rust_m)
    nm_r = noise_mask(0.012, seed=seed + 5, lo=0.25, hi=0.75)
    for o in irons:
        assign(o, iron)
        paint(o, grime=lambda co, n: 0.75 * nm_r(co, n), wear=lambda co, n: 0.5 * nm_r(co * 1.7 + 0.31, n))

    nm_a = noise_mask(0.2, seed=seed + 11, lo=0.3, hi=0.7)
    nm_b = noise_mask(0.07, seed=seed + 12, lo=0.3, hi=0.75)
    nm_w = noise_mask(0.28, seed=seed + 13, lo=0.2, hi=0.8)
    z_cop = z_bt + HC

    def joint_damp(co, n):
        """Soft damp halo hugging the back / seat bed joint (seat top in front of the back, foot of the back, rear)."""
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        on_seat = up * (np.abs(z - SH) < 0.02) * np.exp(-np.clip(yfr - y, 0, 1) / 0.045)
        on_back = (z > SH) * (y > yfr - 0.004) * np.clip(-n[:, 1], 0, 1) * np.exp(-np.clip(z - SH, 0, 1) / 0.05)
        behind = np.clip(n[:, 1], 0, 1) * np.exp(-np.abs(z - SH) / 0.06)
        return (on_seat + on_back + 0.8 * behind) * (0.35 + 0.65 * nm_b(co, n)) * (0.5 + 0.5 * nm_a(co, n))

    def grime_fn(co, n):
        """Intended darkening 0..1 per vertex, mapped into the band where the shader's grime actually shows
        (below ~0.4 nothing happens, above ~0.65 the stone goes sooty black)."""
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        g = np.clip(1 - z / 0.2, 0, 1) ** 1.1 * (0.6 + 0.3 * nm_a(co, n))          # splash zone: darker lowest 15 cm
        g = g + np.clip(-n[:, 2], 0, 1) * 0.25                                    # undersides
        sit = np.clip(sum(np.exp(-((x - sxp) / 0.2) ** 2 - ((y - y_sit) / 0.16) ** 2) for sxp in spots), 0, 1)
        g = g + up * (np.abs(z - SH) < 0.03) * (1 - sit) * (0.25 + 0.35 * nm_w(co, n))   # dust lies where nobody sits
        if has_back:
            g = g + 0.5 * joint_damp(co, n)
            sx_, af, lf, ar, lr = streak
            on_slab = (z > SH + 0.002) * (z < z_bt + 0.004)
            front = np.clip(-n[:, 1] * 1.6 - 0.6, 0, 1) * on_slab * (y > yfr - 0.004)
            rear = np.clip(n[:, 1] * 1.6 - 0.6, 0, 1) * on_slab
            for face, a_, l_ in ((front, af, lf), (rear, ar, lr)):
                amp = np.interp(x, sx_, a_)
                ln = np.interp(x, sx_, l_)
                run = np.clip(1 - (z_bt - z) / ln, 0, 1) ** 0.8                  # streaks from the coping drip edge
                ledge = np.clip(1 - (pz0 - z) / 0.065, 0, 1) * (z < pz0 + 0.002)     # and from the panel ledge
                g = g + face * (amp * (0.62 * run + 0.4 * ledge) + 0.22 * np.clip(1 - (z_bt - z) / 0.05, 0, 1))
            dirt = up * (z > pz0 - 0.004) * (z < pz0 + 0.035) * (y > yfr)             # dirt lying on the panel ledge
            g = g + 0.5 * dirt
        return np.clip(g, 0, 1)

    def wear_fn(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        sit = sum(np.exp(-((x - sxp) / 0.19) ** 2 - ((y - y_sit) / 0.15) ** 2) for sxp in spots)
        col = np.clip(sum(np.exp(-((x - sxp) / 0.2) ** 2) for sxp in spots), 0, 1)
        top = up * (np.abs(z - SH) < 0.02) * np.clip(sit, 0, 1) * (0.8 + 0.2 * nm_w(co, n))
        nose = np.exp(-((z - (SH - 0.012)) / 0.022) ** 2) * np.clip(-n[:, 1], 0, 1) * (y < -Dt / 2 + 0.04) * col * 0.55
        w = 0.95 * top + nose
        if has_back:                           # shoulders have rubbed the top rail and panels behind each place
            w = w + (z > SH + 0.12) * (z < z_bt) * (y > yfr - 0.004) * np.clip(-n[:, 1], 0, 1) * col * np.exp(-((z - (SH + 0.3)) / 0.12) ** 2) * 0.3
        if has_arms:                           # hands on the arm tops
            w = w + up * (z > SH + 0.09) * (y < yfr - 0.02) * 0.5
        return np.clip(w * p.wear, 0, 1)

    def moss_fn(co, n):
        """special mask of the stone = moss. The shader breaks it up between ~0.42 and ~0.6 (speckled algae film)
        and grows solid moss above, so damp faces are held inside that band and only joints, the splash zone
        and neglected tops are pushed beyond it."""
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        low = np.clip(1 - z / 0.2, 0, 1)
        north = np.clip(n[:, 1], 0, 1) * (0.5 + 0.5 * np.clip(1 - z / 0.7, 0, 1))
        under = np.clip(-n[:, 2], 0, 1) * 0.5 + (z < z_seat0) * (z > z_seat0 - 0.07) * 0.5     # under the seat overhang
        film = low + north * 0.8 + under * 0.7
        heavy = low ** 2 * 0.7
        if has_back:
            film = film + 0.6 * joint_damp(co, n)
            heavy = heavy + up * (z > z_cop - 0.03) * 0.6
        # heavy damp creeps over the ends and rear of the seat top, never where people sit
        sit = np.clip(sum(np.exp(-((x - sxp) / 0.26) ** 2) for sxp in spots), 0, 1)
        edge = np.clip((np.abs(x) - (L / 2 - 0.24)) / 0.18, 0, 1) + np.clip((y - (yr - 0.15)) / 0.12, 0, 1)
        heavy = heavy + up * (np.abs(z - SH) < 0.03) * moss_over * (edge * 0.9 + 0.3 * (1 - sit))
        brk = (0.3 + 0.7 * nm_a(co, n)) * (0.5 + 0.5 * nm_b(co, n))
        f = np.clip(film * brk * clamp(p.moss * 2.2, 0, 1.6), 0, 1)
        h = np.clip(heavy * brk * clamp((p.moss - 0.25) * 2, 0, 1.5), 0, 1)
        return 0.2 + 0.28 * f + 0.17 * h if p.moss > 0.03 else 0.0 * f

    ru = rng.fork("uv")
    for objs, ti, val in blocks:
        g0, w0 = 0.14 * max(0.0, val), 0.1 * max(0.0, -val)
        for o in objs:
            realize(o)
            uv_transform(o, scale=(1.2, 1.9), rot_deg=ru.jitter(0, 3), offset=(ru.u(0, 2), ru.u(0, 2)))    # grain drawn out along the bed
            uv_keep(o)
            paint(o, grime=lambda co, n, g0=g0: 0.36 + 0.23 * np.clip(grime_fn(co, n) + g0, 0, 1),
                  wear=lambda co, n, w0=w0: np.clip(wear_fn(co, n) + w0, 0, 1), special=moss_fn)
    for o in chips:                            # fracture faces: rough and a shade darker than the dressed skin
        paint(o, grime=lambda co, n: 0.43 + 0.07 * nm_b(co, n), special=lambda co, n: 0.95 * moss_fn(co, n))
    for o in feet:
        paint(o, grime=lambda co, n: 0.44 + 0.16 * nm_a(co, n), special=lambda co, n: (0.36 + 0.14 * nm_b(co, n) + 0.16 * clamp(p.moss * 1.3) * nm_a(co, n)) * (1.0 if p.moss > 0.03 else 0.0))
    for o in joints:                           # pale lime mortar
        paint(o, wear=0.55, special=lambda co, n: moss_fn(co, n))

    # ------------------------------------------------------------------ votive candle stubs on the seat
    ncand = int(p.candles)
    if ncand > 0:
        r = rng.fork("candles")
        wax = mat("wax", "#efe3c4")
        flame_m = mat("emissive", "#ffb45e", glow=8)
        xc = r.u(-0.12, 0.12) * L if has_back else spots[-1] + 0.12
        yc_ = (yr - TB0 - 0.17) if has_back else y_sit + 0.06
        for k in range(ncand):
            cr_, ch = r.u(0.014, 0.02), r.u(0.045, 0.12)
            cx_, cy_ = xc + (k - (ncand - 1) / 2) * r.u(0.085, 0.12), yc_ + r.jitter(0, 0.025)
            cz = SH - 0.004
            c = lathe(f"candle{k}", [(cr_ + r.u(0.012, 0.022), 0.0), (cr_ + 0.01, 0.003), (cr_ + 0.003, 0.0055), (cr_, 0.011), (cr_, ch - 0.005),
                                     (cr_ - 0.0025, ch), (cr_ * 0.55, ch - 0.004), (0.0, ch - 0.0055)], segments=12, loc=(cx_, cy_, cz))
            displace_noise(c, strength=0.0012, scale=0.02, seed=seed + 90 + k)
            assign(c, wax)
            fl = lathe(f"flame{k}", [(0.0, 0.0), (0.0035, 0.004), (0.0045, 0.009), (0.003, 0.016), (0.0, 0.025)], segments=8,
                       loc=(cx_, cy_, cz + ch - 0.001))
            assign(fl, flame_m)
            light(f"candle{k}", (cx_, cy_, cz + ch + 0.07), intensity=0.13, range=2.2, flicker=0.55, shadow_near=0.09)
            fx(f"flame{k}", "flame", (cx_, cy_, cz + ch + 0.004), size=0.03)

    # ------------------------------------------------------------------ moss cushions packed into the joints
    if amt > 0.02 and moss_runs:
        rm = rng.fork("moss")
        mv, mf = [], []
        size = 0.7 + 0.5 * amt
        STEP = 0.014

        def vnoise(knots, s, per):
            q = s / per
            i0 = int(q)
            t = q - i0
            t = 0.5 - 0.5 * math.cos(math.pi * t)
            return knots[i0 % len(knots)] * (1 - t) + knots[(i0 + 1) % len(knots)] * t

        for (p0, p1, A, B, wgt, kind, wmax, hmax) in moss_runs:
            a, b = Vector(p0), Vector(p1)
            ln = (b - a).length
            cover = clamp(amt * wgt)
            if ln < 0.05 or cover < 0.03:
                continue
            T = (b - a) / ln
            A, B = Vector(A).normalized(), Vector(B).normalized()
            flip = A.cross(B).dot(T) < 0
            k1 = [rm.u() for _ in range(int(ln / 0.075) + 3)]
            k2 = [rm.u() for _ in range(int(ln / 0.024) + 3)]
            run = []
            samples = [(q * STEP, None) for q in range(int(ln / STEP) + 1)] + [(None, None)]
            for s, _ in samples:
                e = 0.0
                if s is not None:
                    e = clamp((cover - vnoise(k1, s, 0.075)) / 0.13) * clamp(min(s, ln - s) / 0.012)
                if e > 0.04:
                    run.append((s, e))
                    continue
                if len(run) >= 3:
                    base = len(mv)
                    for ri_, (s2, e2) in enumerate(run):
                        f = math.sqrt(e2) * size * (0.6 + 0.8 * vnoise(k2, s2, 0.024))
                        if ri_ in (0, len(run) - 1):
                            f *= 0.2
                        w, h = wmax * f, hmax * f
                        c = a + T * s2
                        for kk in range(5):
                            if kind == "corner":
                                ph = math.radians(22.5 * kk)
                                ca_, sa_ = math.cos(ph) ** 0.75, math.sin(ph) ** 0.75
                            else:
                                ph = math.radians(45 * kk)
                                ca_, sa_ = math.cos(ph), math.sin(ph)
                            jit = rm.u(0.82, 1.2) if 0 < kk < 4 else 1.0
                            v = c + A * (w * ca_ * jit) + B * (h * sa_ * jit) + T * rm.jitter(0, 0.002)
                            if kk == 0:
                                v -= B * 0.002
                            elif kk == 4:
                                v -= (B if kind == "dome" else A) * 0.002
                            mv.append((v.x, v.y, max(v.z, 0.0006)))
                    for ri_ in range(len(run) - 1):
                        for kk in range(4):
                            q0 = base + ri_ * 5 + kk
                            quad = (q0, q0 + 1, q0 + 6, q0 + 5)
                            mf.append(tuple(reversed(quad)) if flip else quad)
                run = []
        if mf:
            mo = mesh_object("moss_cushions", mv, mf)
            uv_box(mo)
            smooth(mo, 80)
            assign(mo, moss_m)
            print("DBG tris moss", _tris([mo]))

    print("DBG tris chips", _tris(chips), "iron", _tris([realize(o) for o in irons]), "stains", _tris(stains))
    for k, sxp in enumerate(spots):
        socket(f"seat{k}", "attach", (sxp, y_sit, SH), role="seat")
    socket("top", "attach", (spots[-1] + 0.02, y_sit + 0.05, SH), role="prop_top")
