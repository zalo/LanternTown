"""Wrought-iron railing run along X: 16 mm square bars (plain or barley-twisted) drift-punched through two flat
rails - each crossing swaged into a square boss - with a forged neck and a slim spear / fleur-de-lis / ball
finial. Heavier 60 mm chamfered standards every ~1.8 m carry a stepped foot, rail collars, a turned knop, a
moulded cap and an urn; each is braced by a square back stay riveted to its back and leaded into a stone pad set
in the paving, and by flat-bar C-scrolls (riveted tail, collar band at the bar / stay). The standards are leaded
into a low kerb of separately dressed blocks of varied length with mortar joints (optional; without it they stand
on bolted base plates). Optional ring frieze and dog bars. The run spans exactly x in [-length/2, +length/2];
`ends` chooses which ends carry a terminal standard so runs butt together in one regular rhythm.
Tertiary detail is chosen from the real triangle count, so long runs get leaner bars instead of more triangles.
Levels of detail (docs/LOD.md): lod 1 keeps every bar, twist, scroll and gilt tip but as bare square stock (no
swaged bosses, forged points, rivets, chamfers); lod 2 is half the bars, half as thick again, with a spike tip.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, degrees, atan2, hypot, copysign

META = {"name": "Wrought-iron railing run", "class": "component", "mount": "ground", "focus": 0.74, "lod": True}
PARAMS = {
    "length": {"default": 3.6, "min": 1.0, "max": 12.0, "desc": "run length along X (m); spans exactly +-length/2"},
    "height": {"default": 1.05, "min": 0.7, "max": 1.8, "desc": "height of the top rail above the ground (m); finials rise ~0.16 above"},
    "bar_spacing": {"default": 0.12, "min": 0.09, "max": 0.2, "desc": "centre spacing of the bars (m)"},
    "finial": {"default": "spear", "choices": ["spear", "fleur", "ball"], "desc": "forged finial on every bar"},
    "bar_style": {"default": "alternate", "choices": ["plain", "alternate", "twisted"], "desc": "plain square bars, every other bar barley-twisted, or all twisted"},
    "kerb": {"default": True, "desc": "low dressed-stone kerb under the railing"},
    "kerb_height": {"default": 0.18, "min": 0.08, "max": 0.45, "desc": "kerb height (m)"},
    "standard_spacing": {"default": 1.8, "min": 1.0, "max": 2.6, "desc": "approximate spacing of the heavy standards (m)"},
    "stays": {"default": True, "desc": "back stays (+Y side) bracing every standard"},
    "frieze": {"default": "none", "choices": ["none", "rings"], "desc": "band of forged rings under the top rail"},
    "dog_bars": {"default": False, "desc": "short spiked bars between the main bars in the lower third"},
    "ends": {"default": "both", "choices": ["both", "left", "right", "none"], "desc": "which run ends carry a terminal standard (chain runs with 'left')"},
    "finish": {"default": "painted", "choices": ["painted", "wrought"], "desc": "painted iron or bare blackened forge finish"},
    "paint": {"default": "#40594b", "desc": "paint colour of the ironwork"},
    "gilt": {"default": "tips", "choices": ["none", "tips", "urns", "both"], "desc": "worn gold leaf: the top third of the bar finials + urn knobs, the urns, or both"},
    "rust": {"default": 0.2, "min": 0.0, "max": 1.0, "desc": "amount of rust at joints, feet and in patches (>= 0.75: rusted through)"},
}
VARIANTS = {
    "fleur_tall_gilt": {"length": 3.0, "height": 1.5, "finial": "fleur", "bar_style": "plain", "frieze": "rings", "gilt": "both",
                        "paint": "#3b434e", "kerb_height": 0.32, "bar_spacing": 0.13, "standard_spacing": 1.5, "rust": 0.1, "seed": 4},
    "ball_low_dogbars": {"length": 3.6, "height": 0.8, "finial": "ball", "bar_style": "twisted", "dog_bars": True, "bar_spacing": 0.17,
                         "paint": "#55231d", "gilt": "none", "kerb_height": 0.12, "standard_spacing": 1.7, "rust": 0.35, "seed": 9},
    "rusty_no_kerb": {"length": 2.4, "height": 1.1, "finial": "spear", "bar_style": "plain", "kerb": False, "finish": "wrought", "gilt": "none",
                      "rust": 0.95, "bar_spacing": 0.105, "standard_spacing": 2.3, "seed": 15},
    "long_run_open_end": {"length": 6.0, "height": 0.95, "finial": "spear", "bar_style": "plain", "ends": "left", "paint": "#5b7480", "bar_spacing": 0.14,
                          "gilt": "none", "standard_spacing": 2.0, "rust": 0.2, "seed": 22},
}

BAR = 0.016        # square bar stock
STD = 0.060        # standard stock
SB = 0.020         # stay stock
KW = 0.30          # kerb width
INSET = 0.07       # end standard centre from the end plane (foot and urn fit inside the run)
PAD_H = 0.035      # stay pad stands this proud of the paving


# ----------------------------------------------------------------------------------------------- helpers
def _tris(o):
    return sum(len(pg.vertices) - 2 for pg in o.data.polygons)


def _outward(o):
    """Make the faces point outwards (signed volume test; open tubes / lathes included)."""
    me = o.data
    if len(me.polygons) == 0:
        return o
    c = np.mean(np.array([v.co[:] for v in me.vertices]), axis=0)
    vol = 0.0
    for pg in me.polygons:
        vol += pg.area * float(np.dot(np.array(pg.center[:]) - c, np.array(pg.normal[:])))
    if vol < 0.0:
        me.flip_normals()
    return o


def _prism(name, rings, bottom_pole=None, top_pole=None, cap_bottom=False, cap_top=False):
    """Square bar along +Z from rings [(z, half_side, angle_deg)]; twisted spans are split along the convex diagonal."""
    verts, faces = [], []
    for z, hs, ang in rings:
        r = hs * 1.41421356
        for k in range(4):
            t = radians(ang + 45.0 + 90.0 * k)
            verts.append((r * cos(t), r * sin(t), z))

    def rad(u, v):
        return hypot(verts[u][0] + verts[v][0], verts[u][1] + verts[v][1])

    for i in range(len(rings) - 1):
        tw = abs(rings[i + 1][2] - rings[i][2]) > 1e-6
        for k in range(4):
            a_, b_ = i * 4 + k, i * 4 + (k + 1) % 4
            c_, d_ = (i + 1) * 4 + (k + 1) % 4, (i + 1) * 4 + k
            if not tw:
                faces.append((a_, b_, c_, d_))
            elif rad(a_, c_) >= rad(b_, d_):
                faces += [(a_, b_, c_), (a_, c_, d_)]
            else:
                faces += [(a_, b_, d_), (b_, c_, d_)]
    top = (len(rings) - 1) * 4
    if bottom_pole is not None:
        verts.append((0.0, 0.0, bottom_pole))
        faces += [((k + 1) % 4, k, len(verts) - 1) for k in range(4)]
    elif cap_bottom:
        faces.append((3, 2, 1, 0))
    if top_pole is not None:
        verts.append((0.0, 0.0, top_pole))
        faces += [(top + k, top + (k + 1) % 4, len(verts) - 1) for k in range(4)]
    elif cap_top:
        faces.append((top, top + 1, top + 2, top + 3))
    o = mesh_object(name, verts, faces)
    uv_grain(o, axis='Z')
    smooth(o, 40)
    return o


def _spike(name, z0, z1, hs, tip=None, foot=None):
    """Far-level bar: a three-sided stick z0..z1 (open below); with tip= just the three-sided point z1..tip, with foot= as well
    the same point downwards (a ball as a six-faced gem)."""
    v = [(2.0 * hs * cos(radians(90.0 + 120.0 * k)), 2.0 * hs * sin(radians(90.0 + 120.0 * k)), z) for z in (z0, z1) for k in range(3)]
    if tip is None:
        f = [(k, (k + 1) % 3, 3 + (k + 1) % 3, 3 + k) for k in range(3)]
    else:
        v = v[3:] + [(0.0, 0.0, tip)]
        f = [(k, (k + 1) % 3, 3) for k in range(3)]
        if foot is not None:
            v.append((0.0, 0.0, foot))
            f += [((k + 1) % 3, k, 4) for k in range(3)]
    o = mesh_object(name, v, f)
    uv_grain(o, axis='Z')
    smooth(o, 40)
    return o


def _volute(R, r_end, a0, a1, spiral_deg, da, ds):
    """2D scroll about the origin: arc of radius R from a0 to a1 (deg), then a spiral on to r_end."""
    pts = []
    n1 = max(2, int(round(abs(a1 - a0) / da)))
    for i in range(n1 + 1):
        t = radians(a0 + (a1 - a0) * i / n1)
        pts.append((R * cos(t), R * sin(t)))
    sg = 1.0 if a1 > a0 else -1.0
    n2 = max(3, int(round(spiral_deg / ds)))
    for i in range(1, n2 + 1):
        f = i / n2
        t = radians(a1 + sg * spiral_deg * f)
        r = R + (r_end - R) * f
        pts.append((r * cos(t), r * sin(t)))
    return pts


def _stone_run(name, x0, x1, y0, y1, h, c, dx, rs=None, top_rows=2, rough=0.0, chips=0, knock=(True, True), lean=0, butt=(False, False)):
    """Dressed stone block as a hand-built extrusion along X (no bottom face): chamfered top arrises, flat ends,
    faces a little out of true, punched tooling relief, knocked corners and chipped arrises.
    lean (level of detail): 1 = no end sections / knocked corners, 2 = a plain chamfered section, true faces and no
    end face where the block butts on the next one (butt), 3 = a plain square section."""
    wz = min(0.012, 0.08 * h)                         # the top is weathered (falls) towards both arrises
    ym, f = (y0 + y1) / 2, 0.3 * (y1 - y0)
    prof = [(y0, 0.0), (y0, (h - c) * 0.5), (y0, h - c), (y0 + c, h - wz)]
    prof += [(ym + f * t_, h) for t_ in {1: (0.0,), 2: (-1.0, 1.0), 3: (-1.0, 0.0, 1.0)}[top_rows]]
    prof += [(y1 - c, h - wz), (y1, h - c), (y1, (h - c) * 0.5), (y1, 0.0)]
    if lean >= 2:
        prof = [(y0, 0.0), (y0, h - c), (y0 + c, h - wz), (y1 - c, h - wz), (y1, h - c), (y1, 0.0)]
        if lean >= 3:
            prof = [(y0, 0.0), (y0, h - wz), (y1, h - wz), (y1, 0.0)]
        rs = None
    m = len(prof)
    ln = x1 - x0
    e = 0.014 if ln > 0.12 and not lean else 0.0
    n = max(1, int(round((ln - 2 * e) / dx)))
    xs = [x0 + e + (ln - 2 * e) * i / n for i in range(n + 1)]
    if e > 0:
        xs = [x0] + xs + [x1]
    V = np.array([(x, y, z) for x in xs for (y, z) in prof], dtype=np.float64).reshape(len(xs), m, 3)
    if rs is not None and rough > 0:
        # every face a little out of true + punched tooling
        inner = np.ones(len(xs))
        inner[0] = inner[-1] = 0.0                   # end sections stay true so blocks / runs butt cleanly
        V[:, 1:m - 1, 2] += np.array([[rs.jitter(0, rough * 0.5) for _ in range(m - 2)] for _ in xs]) * (V[:, 1:m - 1, 2] > h - c - wz - 1e-6) * inner[:, None]
        for k in (1, 2):
            V[:, k, 1] += np.array([rs.jitter(0, rough) for _ in xs]) * inner
            V[:, m - 1 - k, 1] += np.array([rs.jitter(0, rough) for _ in xs]) * inner
        # knocked corners at the block ends
        for i in ([i_ for i_, kn_ in zip((0, len(xs) - 1), knock) if kn_] if e > 0 else ()):
            for kk, sg in (((2, 3), 1.0), ((m - 3, m - 4), -1.0)):
                d = rs.u(0.002, 0.009)
                for k in kk:
                    V[i, k, 2] -= d
                    V[i, k, 1] += sg * d * 0.6
            V[i, 4:m - 4, 2] -= rs.u(0.001, 0.004)
        # chipped arrises
        for _ in range(chips):
            xc = rs.u(x0 + 0.05, x1 - 0.05)
            wr = rs.u(0.05, 0.11)
            d = rs.u(0.008, 0.02)
            front = rs.chance(0.6)
            f = np.clip(1.0 - np.abs(np.array(xs) - xc) / wr, 0.0, 1.0)
            for k in ((2, 3) if front else (m - 3, m - 4)):
                V[:, k, 2] -= d * f * (1.0 if k in (3, m - 4) else 0.55)
                V[:, k, 1] += (1.0 if front else -1.0) * d * f * (0.55 if k in (3, m - 4) else 1.0)
    V[:, :, 0] = np.clip(V[:, :, 0], x0, x1)
    V[:, 0, 2] = 0.0
    V[:, m - 1, 2] = 0.0
    faces = []
    for i in range(len(xs) - 1):
        for k in range(m - 1):
            faces.append((i * m + k, (i + 1) * m + k, (i + 1) * m + k + 1, i * m + k + 1))
    if not (lean >= 2 and butt[0]):
        faces.append(tuple(range(m)))
    if not (lean >= 2 and butt[1]):
        faces.append(tuple(reversed(range((len(xs) - 1) * m, len(xs) * m))))
    o = mesh_object(name, [tuple(v) for v in V.reshape(-1, 3)], faces)
    smooth(o, 9)
    return o


# ----------------------------------------------------------------------------------------------- model
def _ribbon(name, pts, w):
    """Flat two-sided strap in the XZ plane along pts [(x, y, z)], half-width w(t) - the cheap stand-in for a forged scroll."""
    v, n = [], len(pts)
    for i, c in enumerate(pts):
        a_, b_ = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        dx, dz = b_[0] - a_[0], b_[2] - a_[2]
        d = hypot(dx, dz) or 1.0
        k = w(i / (n - 1.0))
        v += [(c[0] - dz / d * k, c[1], c[2] + dx / d * k), (c[0] + dz / d * k, c[1], c[2] - dx / d * k)]
    f = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(n - 1)]
    o = mesh_object(name, v, f + [q_[::-1] for q_ in f])
    uv_grain(o, axis='Z')
    return o


def _construct(p, rng, q_in, X=0):
    skimp = q_in < 0                                 # absurdly dense run: barest bars
    q = max(q_in, 0)                                 # 3 = full hero detail ... 0 = leanest

    def lv(*vals):
        return vals[q]

    Lv = lod()                                       # 0 = the full model, 1 = mid, 2 = far (q is 1 or 0 there)
    # X: last rungs of the ladder at the cheaper levels (lod 1: 1 = flat petals, 2 = flat frieze rings too; lod 2: 1 = every other bar left out)

    L = float(p.length)
    rust = float(p.rust)
    seed = int(p.seed)
    xa, xb = -L / 2, L / 2
    H = float(p.height)
    kerb = bool(p.kerb)
    zk = max(0.06, min(float(p.kerb_height), H - 0.5)) if kerb else 0.0
    zs0 = zk if kerb else 0.012                      # underside of the standards' feet
    z_bot = zs0 + (0.13 if kerb else 0.14)           # bottom rail centre
    z_top = H - 0.010                                # top rail centre (its crown is at H)

    heavy = rust >= 0.75
    if heavy:                                        # rusted through: matte orange-brown scale, no finish left
        iron = mat("iron_painted", "#6b4e36", rough=2.0, wear=0.5, special=1.0, scale=2.5)
        thin_iron = mat("iron_painted", "#7a5a3d", rough=2.0, wear=0.5, special=1.0, scale=2.5)
    elif p.finish == "wrought":
        iron = thin_iron = mat("iron_wrought", rough=0.85, special=1.0 + 1.5 * rust)
    else:
        iron = thin_iron = mat("iron_painted", p.paint, rough=0.62, wear=0.6, special=1.0 + 1.0 * rust, scale=2.5)
    gold = mat("metal_plain", "#a39063", rough=0.7)
    lead = mat("lead")
    tip_mat = gold if p.gilt in ("tips", "both") else thin_iron
    old_gold = mat("metal_plain", "#7f6e4a", rough=0.85)       # big gilt masses must not glow like lamps under the amber key
    urn_mat = old_gold if p.gilt in ("urns", "both") else iron
    knob_mat = gold if p.gilt != "none" else iron

    S = {"iron": [], "kerb": [], "pads": [], "mortar": [], "lead": [], "top_rail": None, "q_in": q}
    iron_objs = S["iron"]

    # ---- layout: nodes (standards) and bars
    left_end = p.ends in ("both", "left")
    right_end = p.ends in ("both", "right")
    x0 = xa + INSET if left_end else xa - INSET      # an open end assumes the neighbour's standard just beyond
    x1 = xb - INSET if right_end else xb + INSET
    span = x1 - x0
    nb = max(1, int(round(span / float(p.standard_spacing))))
    bay = span / nb
    nodes = [x0 + bay * k for k in range(nb + 1)]
    std_x = [x for k, x in enumerate(nodes) if 0 < k < nb or (k == 0 and left_end) or (k == nb and right_end)]
    m = max(2, int(round(bay / float(p.bar_spacing))))
    g = bay / m
    bar_x, gap_x, leg_x = [], [], []
    for k in range(nb):
        for j in range(1, m):
            bar_x.append(nodes[k] + g * j)
        for j in range(m):
            cx = nodes[k] + g * (j + 0.5)
            if xa + 0.02 < cx < xb - 0.02:
                gap_x.append((cx, j == 0 or j == m - 1))
        lx = nodes[k] + g * (m // 2 + 0.5)
        if nb * m > 6 and xa + 0.05 < lx < xb - 0.05:
            leg_x.append(lx)

    rr = rng.fork("forge")
    rings = p.frieze == "rings"
    ring_r = min(0.05, (g - BAR) / 2 - 0.0005)
    z_sub = z_top - 0.010 - 2 * ring_r - 0.004 if rings else z_top
    z_under = z_sub - 0.004 if rings else z_top - 0.010      # underside of the lowest top member
    clear = z_sub - z_bot
    dog = bool(p.dog_bars) and clear > 0.3
    z_dog = z_bot + max(0.14, 0.3 * clear)
    z_dr = z_dog - 0.05
    a = STD / 2
    hh = H - zs0

    # ---- bars: one square prism each - forged point, swaged boss at each rail, (twist), bead, neck - plus finial
    h = BAR / 2
    hb = h + 0.0065
    zb0 = z_bot - 0.06
    zc = H + 0.034 - zb0                             # bead / neck (bar-local z)
    Lb = zc + 0.024
    zB = zc + 0.02                                   # finial base
    tw_lo = (z_dog + 0.06 if dog else z_bot + 0.07)
    tw_hi = z_sub - 0.07
    turns = 0.0
    if q > 0:
        for t_ in (2.0, 1.5, 1.0):
            if t_ <= lv(0, 1.0, 1.5, 2.0) and t_ * 0.17 <= tw_hi - tw_lo:
                turns = t_
                break
    spt = 4 if Lv else lv(6, 6, 6, 7)

    def crossing(z_):
        if Lv:                                       # cheaper levels: the bar runs straight through the rails
            return []
        if q >= 2:
            return [(z_ - 0.023, h, 0.0), (z_ - 0.014, hb, 0.0), (z_ + 0.014, hb, 0.0), (z_ + 0.023, h, 0.0)]
        if q == 1:
            return [(z_ - 0.028, h, 0.0), (z_ + 0.028, h, 0.0)]
        return []

    def bar_rings(twisted):
        # (mid: the stick starts where LOD 0's forged point ends, or the foot of every bar is a gap under the bottom rail)
        R = [(-0.018 if Lv == 1 else 0.0, h, 0.0)] + crossing(z_bot - zb0)
        topc = crossing(z_top - zb0)
        lo = R[-1][0]
        hi = topc[0][0] if topc else zc - 0.012
        ang = 0.0
        if twisted and turns > 0:
            tl = turns * 0.17
            ta = tw_lo + (tw_hi - tw_lo - tl) / 2 - zb0
            ns = int(round(turns * spt))
            ang = 360.0 * turns
            R += [(ta + tl * i / ns, h, ang * i / ns) for i in range(ns + 1)]
        else:
            nm_ = 0 if Lv else lv(1, 0, 0, 0)
            R += [(lo + (hi - lo) * i / (nm_ + 1), h, 0.0) for i in range(1, nm_ + 1)]
        R += topc
        if q == 3:
            R += [(zc - 0.009, h, 0.0), (zc - 0.004, h + 0.005, 0.0), (zc + 0.004, h + 0.005, 0.0), (zc + 0.009, 0.0062, 0.0), (Lb, 0.0052, 0.0)]
        elif q == 2:
            R += [(zc - 0.008, h, 0.0), (zc, h + 0.0052, 0.0), (zc + 0.009, 0.0062, 0.0), (Lb, 0.0052, 0.0)]
        elif q == 1 and not Lv:
            R += [(zc - 0.004, h, 0.0), (zc + 0.009, 0.0062, 0.0), (Lb, 0.0052, 0.0)]
        else:
            R += [(zc - 0.004, h, 0.0), (Lb, 0.0056, 0.0)]
        return [(z, hs, an_ if an_ > 0 or z < R[1][0] + 1e-9 else an_) for z, hs, an_ in _fix_angles(R, ang)]

    def _fix_angles(R, ang):
        # rings above the twist keep its end angle (a multiple of 180 deg, i.e. the same square)
        out, seen = [], False
        for z, hs, an_ in R:
            if an_ > 0:
                seen = True
            out.append((z, hs, ang if (seen and an_ == 0.0) else an_))
        return out

    def blade(lowp, upp, flat=0.36):
        """Flattened leaf blade (diamond section, wide along the run): iron lower part + tip that may be gilt."""
        def squash(co):
            return Vector((co.x, co.y * lerp(1.0, flat, smoothstep(0.0, 0.035, co.z)), co.z))
        lo_ = lathe("f_blade", lowp, segments=4, loc=(0, 0, zB), smooth_angle=12)
        up_ = lathe("f_tip", upp, segments=4, loc=(0, 0, zB), smooth_angle=12)
        for o in (lo_, up_):
            transform_verts(o, squash)
            _outward(o)
        if tip_mat is gold:
            g0, g1 = zB + upp[0][1], zB + upp[-1][1]
            paint(up_, grime=lambda co, nrm: 0.1 + 0.75 * np.clip((g1 - co[:, 2]) / (g1 - g0), 0.0, 1.0))
        return lo_, up_

    def far_bar():
        """lod 2: every bar still stands, as a three-sided stick a third thicker with a three-sided point"""
        hf = h * (1.3 if X == 0 else 1.7)
        top = {"spear": (0.075, 0.110), "fleur": (0.106, 0.138), "ball": (0.0, 0.0)}[p.finial]
        zt = zB + top[0] if top[1] else zc + 0.014
        body = [_spike("bar", 0.0, zt, hf)]
        tips = []
        if top[1]:
            tips.append(_spike("f_tip", 0.0, zt, 0.0085, tip=zB + top[1]))
        else:
            body.append(_spike("f_ball", 0.0, zc + 0.014 + 0.037, 0.016, tip=zc + 0.014 + 0.085, foot=zc + 0.014))
        if p.finial == "fleur":                      # the side petals as one flat diamond across the run
            v_ = [(-0.05, 0.0, zB + 0.07), (0.0, 0.0, zB + 0.012), (0.05, 0.0, zB + 0.07), (0.0, 0.0, zB + 0.1)]
            body.append(mesh_object("f_petal", v_, [(0, 1, 2, 3), (3, 2, 1, 0)]))
        return body, tips

    def finial_parts():
        body, tips = [], []
        q = 0 if Lv else S["q_in"]                   # (cheaper levels: the leanest forged shapes)
        if p.finial == "spear":
            lo_, up_ = blade([(0.0055, 0.0), (0.0072, 0.010), (0.014, 0.040), (0.0125, 0.060), (0.0088, 0.075)] if q == 3 else
                             [(0.0055, 0.0), (0.014, 0.040), (0.0125, 0.060), (0.0088, 0.075)] if q >= 1 else
                             [(0.006, 0.0), (0.014, 0.042), (0.0088, 0.075)],
                             [(0.0088, 0.075), (0.0045, 0.094), (0.0, 0.110)] if q == 3 else [(0.0088, 0.075), (0.0, 0.110)])
            body.append(lo_)
            tips.append(up_)
        elif p.finial == "fleur":
            lo_, up_ = blade([(0.0055, 0.0), (0.0065, 0.022), (0.013, 0.07), (0.0115, 0.09), (0.0078, 0.106)] if q == 3 else
                             [(0.0055, 0.0), (0.0065, 0.022), (0.013, 0.07), (0.0078, 0.106)] if q >= 1 else
                             [(0.006, 0.0), (0.013, 0.07), (0.0078, 0.106)],
                             [(0.0078, 0.106), (0.0038, 0.124), (0.0, 0.138)] if q == 3 else [(0.0078, 0.106), (0.0, 0.138)], flat=0.4)
            body.append(lo_)
            tips.append(up_)
            fl = [(-0.003, -0.0065), (0.003, -0.0065), (0.003, 0.0065), (-0.003, 0.0065)]
            for s in (-1, 1):
                pts = bezier_path((s * 0.004, 0, zB + 0.006), (s * 0.012, 0, zB + 0.068), (s * 0.05, 0, zB + 0.105), (s * 0.048, 0, zB + 0.05),
                                  n=lod_exact(3) if Lv else lv(3, 4, 4, 6))
                if Lv and X:
                    body.append(_ribbon("f_petal", pts, lambda t: 0.006 * (0.7 + 0.6 * sin(pi * min(1.0, t * 1.1)) - 0.3 * t)))
                    continue
                pe = sweep("f_petal", pts, fl, smooth_path=False, cap=q == 3, up=(0, 1, 0),
                           scale_fn=lambda t: 0.7 + 0.6 * sin(pi * min(1.0, t * 1.1)) - 0.3 * t)
                body.append(_outward(pe))
                if q == 3:
                    pts = bezier_path((s * 0.006, 0, zc + 0.004), (s * 0.02, 0, zc - 0.016), (s * 0.034, 0, zc - 0.028), (s * 0.026, 0, zc - 0.048), n=4)
                    ta_ = sweep("f_tail", pts, fl, smooth_path=False, up=(0, 1, 0), scale_fn=lambda t: 1.0 - 0.5 * t)
                    body.append(_outward(ta_))
        else:
            k_ = 1.15
            bp = [(0.0062, 0.0), (0.0125, 0.004), (0.0085, 0.011), (0.0165, 0.017), (0.0225, 0.029), (0.0235, 0.037), (0.0205, 0.049),
                  (0.012, 0.058), (0.0055, 0.062)]
            if q < 3:
                bp = [bp[i] for i in ((0, 3, 4, 6, 8) if q < 2 else (0, 1, 3, 4, 6, 8))]
            bl = lathe("f_ball", [(r * k_, z * k_) for r, z in bp], segments=lod_pick(lv(6, 6, 6, 10), 5), loc=(0, 0, zc + 0.014))
            body.append(_outward(bl))
            bt = lathe("f_btn", [(r * k_, z * k_) for r, z in [(0.0055, 0.062), (0.0068, 0.068), (0.0, 0.074)]], segments=lod_pick(lv(6, 6, 6, 10), 5),
                       loc=(0, 0, zc + 0.014))
            tips.append(_outward(bt))
        return body, tips

    def bar_mesh(twisted):
        if Lv >= 2:
            body, tips = far_bar()
            pr = body.pop(0)
        else:
            pr = _prism("bar", bar_rings(twisted), bottom_pole=None if (q == 0 or Lv) else -0.018, cap_bottom=q == 0 and not Lv)
            body, tips = finial_parts()
        for o in [pr] + body:
            assign(o, thin_iron)
        for o in tips:
            assign(o, tip_mat)
        b = join([pr] + body + tips, "bar_base")
        b["tsq_smooth"] = 38.0 if Lv < 2 else 117.0  # (far: the three-sided points shade round - one flat facet facing a lantern flashes -
        #                                              the sticks, 120 degrees between faces, stay flat)
        return b

    lean = 0.12 + 0.25 * rust
    mats_plain, mats_tw = [], []
    for i, x in enumerate(bar_x):
        M = trs((x + rr.jitter(0, 0.0012), rr.jitter(0, 0.001), zb0 + rr.jitter(0, 0.002)),
                (rr.jitter(0, lean), rr.jitter(0, lean), rr.jitter(0, 2.0)))
        tw = turns > 0 and (p.bar_style == "twisted" or (p.bar_style == "alternate" and i % 2 == 1))
        (mats_tw if tw else mats_plain).append(M)
    if Lv >= 2 and X:                                # (left out after every bar has drawn its lean: nothing else moves)
        mats_plain, mats_tw = mats_plain[::2], mats_tw[::2]
    if mats_plain:
        iron_objs.append(instances_to_mesh("bars", bar_mesh(False), mats_plain, keep_base=False, seed=seed))
    if mats_tw:
        iron_objs.append(instances_to_mesh("bars_tw", bar_mesh(True), mats_tw, keep_base=False, seed=seed + 1))

    # ---- rails (hand-straightened: a little sag and wander); vertices only in the free gaps so nothing is buried
    rx0 = nodes[0] if left_end else xa
    rx1 = nodes[-1] if right_end else xb
    rstep = 99 if Lv else lv(99, 5, 3, 2)

    def rail(name, z, prof, sag, wander):
        r = rr.fork(name)
        pts = [(rx0, 0.0, z)]
        for k in range(nb):
            n0, n1 = nodes[k], nodes[k + 1]
            js = list(range(1, m - 1))
            pick = js[rstep - 1::rstep] if len(js) >= rstep else ([m // 2] if m >= 3 else [])
            xs = [n0 + 0.052] + [n0 + g * (j + 0.5) for j in pick] + [n1 - 0.052]
            for x in xs:
                if rx0 + 0.02 < x < rx1 - 0.02 and x > pts[-1][0] + 0.03:
                    t = (x - n0) / (n1 - n0)
                    e = sin(pi * t)
                    pts.append((x, r.jitter(0, wander) * e, z - sag * e * (bay / 1.8) ** 2 + r.jitter(0, wander * 0.6) * e))
        pts.append((rx1, 0.0, z))
        o = sweep(name, pts, prof, smooth_path=False, up=(0, 0, 1))
        _outward(o)
        smooth(o, 35)
        assign(o, thin_iron)
        return o

    if q >= 2:
        top_prof = [(-0.025, -0.010), (0.025, -0.010), (0.025, -0.002), (0.021, 0.005), (0.011, 0.010), (-0.011, 0.010), (-0.021, 0.005), (-0.025, -0.002)]
        flat_prof = rounded_rect(0.046, 0.014, 0.003, 1)
    else:
        top_prof = [(-0.025, -0.010), (0.025, -0.010), (0.025, -0.001), (0.013, 0.010), (-0.013, 0.010), (-0.025, -0.001)]
        flat_prof = [(-0.023, -0.007), (0.023, -0.007), (0.023, 0.007), (-0.023, 0.007)]
    thin_prof = [(-0.017, -0.004), (0.017, -0.004), (0.017, 0.004), (-0.017, 0.004)]
    S["top_rail"] = rail("rail_top", z_top, top_prof, 0.0015, 0.001)
    iron_objs.append(S["top_rail"])
    iron_objs.append(rail("rail_bot", z_bot, flat_prof, 0.003, 0.0012))
    if rings:
        iron_objs.append(rail("rail_sub", z_sub, thin_prof, 0.002, 0.0008))
        if Lv and X >= 2:                            # a flat six-sided washer in the plane of the run
            rg = _ribbon("ring", [((ring_r - 0.0055) * cos(radians(60 * k)), 0.0, (ring_r - 0.0055) * sin(radians(60 * k))) for k in range(7)], lambda t: 0.006)
        else:
            rg = torus("ring", major=ring_r - 0.0055, minor=0.0055, rot=(90, 0, 0), major_seg=lod_exact(6 if Lv else 5 if skimp else lv(6, 12, 12, 12)),
                       minor_seg=lod_exact(lv(3, 3, 3, 4)))
        apply_transform(rg)
        assign(rg, thin_iron)
        rm = []
        for cx, at_std in gap_x:
            s = ((g - BAR / 2 - STD / 2) / 2 - 0.0005) / ring_r if at_std else 1.0
            s = min(s, 1.0)
            ox = (STD - BAR) / 4 * (1 if any(abs(cx - g / 2 - n_) < 1e-4 for n_ in nodes) else -1) if at_std else 0.0
            rm.append(trs((cx + ox, rr.jitter(0, 0.0008), z_top - 0.010 - ring_r * s), (0, rr.u(0, 360), 0), s))
        iron_objs.append(lod_drop(instances_to_mesh("rings", rg, rm, keep_base=False, seed=seed + 2), above=1))
    if dog:
        iron_objs.append(rail("rail_dog", z_dr, thin_prof, 0.002, 0.0008))
        hd = 0.006
        ld = z_dog - zb0
        zr_ = z_bot - zb0
        dr = [(0.0, hd, 0.0)] + ([(zr_ + 0.03, hd, 0.0)] if q > 0 else []) + [(ld - 0.08, hd, 0.0), (ld - 0.012, hd, 0.0)]
        if q == 3:
            dr += [(ld - 0.004, hd + 0.0035, 0.0), (ld + 0.006, hd + 0.002, 0.0)]
        elif q > 0:
            dr[-1] = (ld, hd + 0.003, 0.0)
        if Lv:
            dr = [(0.0, hd * Lv, 0.0), (ld - 0.012, hd * Lv, 0.0)]
        if Lv >= 2:
            db = _spike("dog_base", 0.0, ld + 0.02, hd * 1.6)       # far: a stick to the middle of its point
        else:
            db = _prism("dog_base", dr, bottom_pole=None if (skimp or Lv) else -0.014, cap_bottom=skimp and not Lv, top_pole=ld + 0.05)
        assign(db, thin_iron)
        dm = [trs((cx + rr.jitter(0, 0.001), 0, zb0 + rr.jitter(0, 0.003)), (rr.jitter(0, lean), rr.jitter(0, lean), rr.jitter(0, 3)))
              for cx, at_std in gap_x if not at_std]
        if dm:
            iron_objs.append(instances_to_mesh("dog_bars", db, dm, keep_base=False, seed=seed + 3))
        else:
            delete(db)

    # ---- standards: ONE chamfered-square turned profile (foot, collars, knop, cap) so no vertex is buried
    zs_top = H + 0.105
    feats = []
    for z_, big in [(z_bot, True), (z_top, True)] + ([(z_sub, False)] if rings else []) + ([(z_dr, False)] if dog else []):
        e_, t_ = (0.011, 0.021) if big else (0.008, 0.013)
        feats.append([(a, z_ - t_ - 0.008, 's'), (a + e_, z_ - t_, 's'), (a + e_, z_ + t_, 's'), (a, z_ + t_ + 0.008, 's')])
    zm = (z_bot + z_sub) / 2
    knop = z_sub - z_bot > 0.45 and not (dog and abs(z_dr - zm) < 0.11)
    if knop:
        kn = [(a, -0.062, 's'), (0.036, -0.052, 'o'), (0.036, -0.042, 'o'), (0.045, -0.02, 'o'), (0.0485, 0.0, 'o'), (0.045, 0.02, 'o'),
              (0.036, 0.042, 'o'), (0.036, 0.052, 'o'), (a, 0.062, 's')]
        if q < 3:
            kn = [kn[i] for i in (0, 1, 3, 4, 5, 7, 8)]
        feats.append([(r, zm + dz, kd) for r, dz, kd in kn])
    feats.sort(key=lambda f: f[0][1])
    occupied = [(zs0, zs0 + 0.09)] + [(f[0][1], f[-1][1]) for f in feats]

    def shaft_free(za, zb_):
        return all(zb_ < o0 - 0.004 or za > o1 + 0.004 for o0, o1 in occupied)

    foot = [(0.058, 0.0), (0.058, 0.018), (0.05, 0.026), (0.046, 0.026), (0.046, 0.052), (0.039, 0.066), (a, 0.085)]
    if q < 3:
        foot = [foot[i] for i in ((0, 1, 3, 4, 6) if q == 0 else (0, 1, 3, 4, 5, 6))]
    if Lv >= 2:                                      # far: a plain plinth, the knop as one bulge, no rail collars
        foot = [foot[0], foot[1], foot[-1]]
        feats = [[f[0], f[len(f) // 2], f[-1]] for f in feats if len(f) > 4]
    prof = [(r, zs0 + dz, 's') for r, dz in foot]
    cap_ = [(a, -0.03), (0.035, -0.022), (0.041, -0.016), (0.041, -0.004), (0.036, 0.002), (0.0, 0.002)]
    if q < 3:
        cap_ = [(a, -0.03), (0.041, -0.016), (0.041, 0.002), (0.0, 0.002)] if q == 0 else [(a, -0.03), (0.035, -0.022), (0.041, -0.016), (0.041, 0.002), (0.0, 0.002)]
    for f in feats + [[(r, zs_top + dz, 's') for r, dz in cap_]]:
        gap = f[0][1] - prof[-1][1]
        if q == 3 and gap > 0.3:
            nn = int(gap / 0.27)
            prof += [(a, prof[-1][1] + gap * i / (nn + 1), 's') for i in range(1, nn + 1)]
        prof += f
    seg = 8 if q >= 1 and not Lv else 4               # (cheaper levels: square stock without the chamfers)
    off = radians(22.5 if seg == 8 else 45.0)
    kz = [(z_, kd) for _, z_, kd in prof]

    def squared(co):
        r = hypot(co.x, co.y)
        if r < 1e-7:
            return co
        kd = min(kz, key=lambda e_: abs(e_[0] - co.z))[1]
        th = atan2(co.y, co.x) + off
        ux, uy = cos(th), sin(th)
        if kd == 'o':
            return Vector((r * ux, r * uy, co.z))
        if seg == 4:
            return Vector((copysign(r, ux), copysign(r, uy), co.z))
        c = min(0.0045, r * 0.16)
        if abs(ux) > abs(uy):
            return Vector((copysign(r, ux), copysign(r - c, uy), co.z))
        return Vector((copysign(r - c, ux), copysign(r, uy), co.z))

    shaft = lathe("std_shaft", [(r, z_) for r, z_, _ in prof], segments=seg, smooth_angle=28)
    transform_verts(shaft, squared)
    _outward(shaft)
    assign(shaft, iron)
    sp = [shaft]
    us = 1.27
    uf = [(0.018, 0.0), (0.026, 0.004), (0.026, 0.010), (0.016, 0.016), (0.011, 0.026), (0.011, 0.032), (0.017, 0.037), (0.032, 0.052),
          (0.041, 0.075), (0.043, 0.092), (0.040, 0.104), (0.030, 0.110), (0.026, 0.116), (0.031, 0.121), (0.031, 0.127), (0.022, 0.136),
          (0.010, 0.146), (0.008, 0.152), (0.0125, 0.160), (0.0105, 0.170), (0.0, 0.177)]
    bi = lv((0, 1, 4, 6, 8, 10, 11, 13, 15), (0, 1, 4, 6, 8, 10, 11, 13, 15), (0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 15), (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15))
    ki = lv((15, 17, 18, 20), (15, 17, 18, 20), (15, 17, 18, 20), (15, 16, 17, 18, 19, 20))
    if Lv >= 2:                                      # far: foot, belly, neck / button and point only
        bi, ki = (1, 6, 9, 11, 15), (15, 18, 20)
    useg = lod_pick(lv(8, 10, 10, 12), 8, 4)
    with lod_as(0 if Lv >= 2 else Lv):               # (far: those few points are final)
        urn = lathe("std_urn", [(uf[i][0] * us, zs_top + 0.002 + uf[i][1] * us) for i in bi], segments=lod_exact(useg))
    assign(_outward(urn), urn_mat)
    if urn_mat is old_gold:                              # old leaf: worn thin and dirty towards the foot, brightest on the shoulder
        u0, u1 = zs_top + 0.03 * us, zs_top + 0.11 * us
        un = noise_mask(0.05, seed=p.seed + 31, lo=0.3, hi=0.7)
        paint(urn, grime=lambda co, nrm: np.clip(0.22 + 0.6 * np.clip((u1 - co[:, 2]) / (u1 - u0), 0.0, 1.0) ** 1.5
                                                 + 0.3 * np.asarray(un(co, nrm)), 0.0, 1.0))
    with lod_as(0 if Lv >= 2 else Lv):
        knob = lathe("std_knob", [(uf[i][0] * us, zs_top + 0.002 + uf[i][1] * us) for i in ki], segments=lod_exact(useg))
    assign(_outward(knob), knob_mat if p.gilt != "urns" else urn_mat)
    sp += [urn, knob]
    if kerb:
        ld_ = box("std_lead", (0.132, 0.132, 0.014), loc=(0, 0, zk + 0.003), bevel=lv(0.0, 0.0, 0.0, 0.004), segments=1)
        displace_noise(ld_, strength=0.0015, scale=0.03, seed=seed)
        assign(ld_, lead)
        sp.append(lod_drop(ld_, above=1))
    std_base = join(sp, "std_base")
    std_base["tsq_smooth"] = 40.0                    # 45 deg shaft chamfers stay crisp, the 10-12 sided urn shades round
    Ms = {x: trs((x, 0, 0), (rr.jitter(0, lean * 0.6), rr.jitter(0, lean * 0.6), rr.jitter(0, 1.0))) for x in std_x}
    if std_x:
        iron_objs.append(instances_to_mesh("standards", std_base, [Ms[x] for x in std_x], keep_base=False, uv_jitter=True, seed=seed + 4))
    else:
        delete(std_base)
    if not kerb:                                     # bolted base plates (static, the end ones reach the end plane exactly)
        for x in std_x:
            pl = box("std_plate", (2 * INSET, 0.26, 0.012), loc=(x, 0, 0.006), bevel=lv(0.0, 0.002, 0.002, 0.002), segments=1)
            densify(pl, 0.1)
            iron_objs.append(pl)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    iron_objs.append(lod_drop(box("std_nut", (0.02, 0.02, 0.012), loc=(x + sx * 0.047, sy * 0.1, 0.0175), rot=(0, 0, rr.u(0, 90)))))

    # ---- forged C-scrolls under the rail beside each standard: 14 x 6 flat bar, tail riveted to the standard,
    #      collar band round the first bar
    fl6 = [(-0.004, -0.008), (0.004, -0.008), (0.004, 0.008), (-0.004, 0.008)]

    def taper(t):
        return 0.75 if t < 0.03 else (1.0 if t < 0.86 else 1.0 - 0.4 * (t - 0.86) / 0.14)

    Rs = min(0.05, (g - a - BAR / 2 - 0.0085) / 2)
    scz = z_under - 0.0045 - Rs
    if q >= 1 and std_x and Rs >= 0.026 and shaft_free(scz - 0.085, scz + 0.012):
        scx = a + 0.0045 + Rs
        vol = _volute(Rs, 0.33 * Rs, 180.0, -90.0, 320.0, 90 if Lv else lv(60, 54, 39, 30), 107 if Lv else lv(80, 80, 64, 45))
        for s_ in (-1, 1):
            xs = [x for x in std_x if (s_ > 0 and x + 0.2 < xb) or (s_ < 0 and x - 0.2 > xa)]
            if not xs:
                continue
            path = [(s_ * (scx - Rs), 0, scz - 0.08)] + [(s_ * (scx + u), 0, scz + v) for u, v in vol]
            with lod_as(0):                          # (the few points of the volute are final)
                so = sweep("scroll", path, fl6, smooth_path=False, up=(0, 1, 0), scale_fn=taper, cap=not Lv)
            smooth(_outward(so), 60)
            rv = lod_drop(lathe("scroll_rivet", [(0.0078, 0.0), (0.006, 0.0042), (0.0, 0.0062)], segments=5,
                                loc=(s_ * (a + 0.0085), 0, scz - 0.058), rot=(0, 90 * s_, 0)))
            cxa, cxb = scx + Rs - 0.005, g + BAR / 2 + 0.0035
            cl = box("scroll_collar", (cxb - cxa, 0.024, 0.016), loc=(s_ * (cxa + cxb) / 2, 0, scz), bevel=lv(0.0, 0.0, 0.0, 0.002), segments=1)
            for o in (so, rv, cl):
                assign(o, iron)
            sb_ = join([so, _outward(rv), cl], "scroll_base")
            sb_["tsq_smooth"] = 50.0
            iron_objs.append(instances_to_mesh(f"scrolls{s_}", sb_, [Ms[x] for x in xs], keep_base=False, seed=seed + 6))

    # ---- mid-bay support legs under the bottom rail
    for lx in leg_x:
        zl0, zl1 = zs0 + 0.004, z_bot - 0.0075
        lg = _prism("leg", [(zl0, 0.009, 0.0), (zl0 + 0.03, 0.009, 0.0), ((zl0 + zl1) / 2, 0.009, 0.0), (zl1, 0.009, 0.0)][::3 if Lv else 1])
        place(lg, (lx, 0, 0))
        assign(lg, thin_iron)
        iron_objs.append(lg)
        if Lv >= 2:
            continue
        ft = box("leg_foot", (0.056, 0.056, 0.012), loc=(lx, 0, zs0 + 0.002 if kerb else 0.006), bevel=lv(0.0, 0.003, 0.003, 0.003), segments=1)
        if kerb:
            assign(ft, lead)
            S["lead"].append(ft)
        else:
            iron_objs.append(ft)

    # ---- back stays: square bar riveted to the back of the standard, leaded into a stone pad set in the paving
    reach = min(0.46, max(0.34 * hh, (KW / 2 + 0.16) if kerb else 0.26))
    if p.stays and std_x:
        vl = 0.09
        lo = (zm + 0.078) if knop else (zs0 + 0.5 * hh)
        if dog:
            lo = max(lo, z_dr + 0.04)
        lo = min(lo, z_sub - 0.045 - vl)
        y1 = a + SB / 2 + 0.0005
        A = (y1 + 0.032, lo - 0.075)
        F = (reach - 0.03, PAD_H + 0.11)
        path = [(0, y1, lo + vl), (0, y1, lo + 0.03), (0, y1, lo), (0, A[0], A[1]),
                (0, lerp(A[0], F[0], 0.33), lerp(A[1], F[1], 0.33)), (0, lerp(A[0], F[0], 0.67), lerp(A[1], F[1], 0.67)),
                (0, F[0], F[1]), (0, reach, PAD_H + 0.035), (0, reach, PAD_H - 0.008)]
        sq = [(-SB / 2, -SB / 2), (SB / 2, -SB / 2), (SB / 2, SB / 2), (-SB / 2, SB / 2)]
        if Lv >= 2:                                  # far: the stay as three straight lengths
            with lod_as(0):
                st = sweep("stay", [path[0], path[2], path[3], path[6], path[8]], [(-SB * 0.7, -SB / 2), (SB * 0.7, -SB / 2), (0.0, SB * 0.7)],
                           smooth_path=False, up=(1, 0, 0), cap=False)
        else:
            st = sweep("stay", path, sq, smooth_path=True, step=lv(0.16, 0.12, 0.10, 0.07), up=(1, 0, 0))
        smooth(_outward(st), 40)
        parts = [st]
        for dz in ((0.02, 0.07) if q > 0 and not Lv else ()):
            parts.append(_outward(lathe("stay_rivet", [(0.0085, 0.0), (0.0065, 0.0045), (0.0, 0.0068)], segments=5,
                                        loc=(0, y1 + SB / 2, lo + dz), rot=(-90, 0, 0))))
        th = atan2(F[0] - A[0], A[1] - F[1])
        R2 = 0.038
        cy = a + 0.0045 + R2
        cz = A[1] - (R2 + 0.0045 + SB / 2 + (cy - A[0]) * cos(th)) / max(sin(th), 0.05)
        kz_ = cz + (R2 + 0.0145) * sin(th)
        if q >= 1 and shaft_free(cz - 0.085, cz + 0.012) and F[1] + 0.04 < kz_ < A[1] - 0.02:
            vol = _volute(R2, 0.33 * R2, 180.0, -90.0, 320.0, 90 if Lv else lv(60, 54, 39, 30), 107 if Lv else lv(80, 80, 64, 45))
            with lod_as(0):
                sc = sweep("stay_scroll", [(0, cy - R2, cz - 0.08)] + [(0, cy + u, cz + v) for u, v in vol], [(-0.008, -0.004), (0.008, -0.004), (0.008, 0.004), (-0.008, 0.004)],
                           smooth_path=False, up=(1, 0, 0), scale_fn=taper, cap=not Lv)
            smooth(_outward(sc), 60)
            parts.append(sc)
            if not Lv:
                parts.append(_outward(lathe("stay_srivet", [(0.0078, 0.0), (0.006, 0.0042), (0.0, 0.0062)], segments=5,
                                            loc=(0, a + 0.0085, cz - 0.058), rot=(-90, 0, 0))))
            parts.append(box("stay_collar", (SB + 0.008, 0.038, 0.016), loc=(0, cy + (R2 + 0.0105) * cos(th), cz + (R2 + 0.0105) * sin(th)),
                             rot=(degrees(th), 0, 0), bevel=lv(0.0, 0.0, 0.0, 0.002), segments=1))
        for o in parts:
            assign(o, iron)
        eye = lathe("stay_lead", [(0.036, 0.0), (0.033, 0.006), (0.02, 0.011), (0.013, 0.012)], segments=lv(6, 6, 8, 10), loc=(0, reach, PAD_H - 0.002))
        assign(_outward(eye), lead)
        lod_drop(eye, above=1)
        stb = join(parts + [eye], "stay_base")
        stb["tsq_smooth"] = 50.0
        iron_objs.append(instances_to_mesh("stays", stb, [Ms[x] for x in std_x], keep_base=False, seed=seed + 8))
        for i, x in enumerate(std_x):
            bx0, bx1 = max(xa, x - 0.13), min(xb, x + 0.13)
            pad = _stone_run(f"stay_pad{i}", bx0, bx1, reach - 0.12, reach + 0.125, PAD_H, 0.008, 0.3 if Lv else lv(0.3, 0.14, 0.13, 0.09),
                             rs=rng.fork(f"pad{i}"), top_rows=1 if Lv else lv(1, 2, 2, 2), rough=0.002, chips=0 if (q == 0 or Lv) else 1, lean=(0, 1, 3)[Lv])
            S["pads"].append(pad)

    # ---- kerb: separately dressed blocks of varied length, tight recessed mortar joints
    if kerb:
        rk = rng.fork("kerb")
        n = max(1, int(round(L / 0.95)))
        w = [rk.u(0.62, 1.3) for _ in range(n)]
        cuts, acc = [xa], xa
        avoid = std_x + leg_x
        for k in range(n - 1):
            acc += L * w[k] / sum(w)
            c = acc
            for sx in avoid:
                if abs(c - sx) < 0.11:
                    c = sx + (0.14 if c >= sx else -0.14)
            if cuts[-1] + 0.3 < c < xb - 0.3:
                cuts.append(c)
        cuts.append(xb)
        gj = 0.0035
        ck = min(0.022, zk * 0.16)
        # cheaper levels keep the kerb closed to a viewer on the street: mid has LOD 0's flat top (one ridge row falls
        # away under a sight line that grazes the top from 15 m), and far, with neither joint nor end face, has its
        # blocks meet edge to edge at the cut - else one looks in through the 7 mm gap
        gk = 0.0 if Lv >= 2 else gj
        for k in range(len(cuts) - 1):
            a2 = cuts[k] + (gk if k > 0 else 0.0)
            b2 = cuts[k + 1] - (gk if k < len(cuts) - 2 else 0.0)
            kb = _stone_run(f"kerb{k}", a2, b2, -KW / 2, KW / 2, zk, ck, (0.6, 0.6, 9.0)[Lv] if Lv else lv(0.5, 0.2, 0.14, 0.1), rs=rk.fork(f"b{k}"),
                            top_rows=(2 if Lv == 1 else 1) if Lv else lv(1, 2, 2, 3), rough=0.0035, chips=0 if (skimp or Lv) else lv(1, 2, 3, 4),
                            knock=(k > 0, k < len(cuts) - 2), lean=Lv, butt=(k > 0, k < len(cuts) - 2))
            S["kerb"].append(kb)
            if k > 0 and Lv < 2:                         # (far: no joints)
                mj = _stone_run(f"joint{k}", cuts[k] - gj, cuts[k] + gj, -KW / 2 - 0.0008, KW / 2 + 0.0008, zk + 0.0008, ck + 0.004, 1.0, top_rows=1,
                                lean=2 * Lv)
                S["mortar"].append(mj)

    S.update(dict(zk=zk, zs0=zs0, z_bot=z_bot, z_top=z_top, z_sub=z_sub, rings=rings, std_x=std_x, leg_x=leg_x, reach=reach, heavy=heavy, q=q))
    S["all"] = S["iron"] + S["kerb"] + S["pads"] + S["mortar"] + S["lead"]
    return S


def build(p, rng):
    L = float(p.length)
    rust = float(p.rust)
    seed = int(p.seed)
    H = float(p.height)
    budget = min(11600.0, 6300.0 + 900.0 * L)
    qs = [(3, 0), (2, 0), (1, 0), (0, 0), (-1, 0)]
    if lod():                                        # a cheaper level: lean detail and its own share of the triangles
        qs = [(1, 0), (0, 0), (-1, 0), (-1, 1), (-1, 2)] if lod() == 1 else [(0, 0), (-1, 0), (-1, 1)]
        budget = 0.97 * lod_budget() if lod_budget() else budget * lod_pick(1.0, 0.38, 0.095)
    for q, X in qs:
        S = _construct(p, rng, q, X)
        n = sum(_tris(o) for o in S["all"])
        if n <= budget or (q, X) == qs[-1]:
            break
        delete(S["all"])
    print(f"DBG detail level {q}: {n} tris (target {budget:.0f})")
    zk, zs0, z_bot, z_top, z_sub = S["zk"], S["zs0"], S["z_bot"], S["z_top"], S["z_sub"]

    # ---- materials + art-directed ageing
    for o in S["iron"]:
        if not o.data.materials:
            assign(o, mat("iron_painted", "#6b4e36", rough=2.0, wear=0.5, special=1.0, scale=2.5) if S["heavy"] else
                   (mat("iron_wrought", rough=0.85, special=1.0 + 1.5 * rust) if p.finish == "wrought" else
                    mat("iron_painted", p.paint, rough=0.62, wear=0.6, special=1.0 + 1.0 * rust, scale=2.5)))
    for o in S["lead"]:
        paint(o, grime=0.25)
    patch = noise_mask(0.38, seed=seed + 3, lo=0.62 - 0.2 * rust, hi=0.82 - 0.2 * rust)      # a few large rust patches, not speckle
    brk = noise_mask(0.06, seed=seed + 4, lo=0.25, hi=0.75)
    nb_ = noise_mask(0.25, seed=seed + 5, lo=0.3, hi=0.7)
    low = height_ramp(zs0, zs0 + 0.28)
    joint_z = [z_bot, z_top] + ([z_sub] if S["rings"] else [])
    wet = noise_mask(0.5, seed=seed + 6, lo=0.45, hi=0.7)
    pit = noise_mask(0.17, seed=seed + 9, lo=0.5, hi=0.72)

    def joints(co, drop, wd):
        j = np.zeros(len(co))
        for z in joint_z:
            j = np.maximum(j, np.exp(-((co[:, 2] - z + drop) / wd) ** 2))
        return j

    def rust_fn(co, nrm):
        # the paint must stay clean (mask exactly 0) between the stories: water sits at the feet and in the
        # bottom-rail joints, and a few large patches have lost their paint
        pa = np.asarray(patch(co, nrm)) * (0.45 + 0.55 * np.asarray(brk(co, nrm)))
        b = np.asarray(nb_(co, nrm))
        lo_ = np.asarray(low(co, nrm))
        j = np.exp(-((co[:, 2] - z_bot + 0.02) / 0.022) ** 2) * np.asarray(wet(co, nrm))
        floor = max(0.0, rust - 0.5) * 0.9           # neglected runs are rusty all over, not just in patches
        v = rust * (1.3 * pa + lo_ * (0.3 + 0.9 * b) * (0.45 + 0.55 * np.asarray(brk(co, nrm))) + j * (0.7 + 0.8 * b)) + floor * (0.35 + 0.65 * b)
        if S["heavy"]:                               # scale all over (the material tint); deep red pitting in patches
            return np.clip(0.75 * np.asarray(pit(co, nrm)) * (0.5 + 0.5 * b) + 0.5 * lo_ * b, 0.0, 1.0)
        return np.clip((v - 0.04) * 1.15, 0.0, 1.0)

    scale_nm = noise_mask(0.09, seed=seed + 8, lo=0.4, hi=0.75)

    def grime_fn(co, nrm):
        gr = 0.45 * np.asarray(low(co, nrm)) * (0.4 + 0.6 * np.asarray(nb_(co, nrm)))
        if S["heavy"]:                               # dark flaking scale mottles the rust
            gr = np.maximum(gr, 0.55 * np.asarray(scale_nm(co, nrm)))
        return np.clip(gr, 0.0, 1.0)

    def chip_fn(co, nrm):                            # paint knocked off where things get handled / rub: the joints
        return np.clip(0.4 * joints(co, 0.0, 0.016) * np.asarray(nb_(co, nrm)), 0.0, 1.0)

    for o in S["iron"]:
        paint(o, special=rust_fn, grime=grime_fn, wear=chip_fn)
    # hands polish the crown of the top rail
    wear_nm = noise_mask(0.3, seed=seed + 7, lo=0.2, hi=0.8)
    paint(S["top_rail"], wear=lambda co, nrm: np.clip(nrm[:, 2], 0, 1) ** 2 * (0.3 + 0.5 * np.asarray(wear_nm(co, nrm))))

    if S["kerb"] or S["pads"]:
        stone = mat("stone_rough", "#989288", grime=1.15, special=1.2)
        pad_stone = mat("stone_rough", "#928c82", grime=1.0, special=1.0)
        mortar = mat("mortar", "#c2b9a6")
        sx = np.array(S["std_x"] + S["leg_x"]) if (S["std_x"] or S["leg_x"]) else np.zeros(0)
        moss_nm = noise_mask(0.2, seed=seed + 11, lo=0.35, hi=0.7)
        dirt_nm = noise_mask(0.16, seed=seed + 12, lo=0.2, hi=0.8)
        hk = max(zk, 0.05)

        def kerb_grime(co, nrm):
            dn = np.asarray(dirt_nm(co, nrm))
            base = np.clip(1.25 - co[:, 2] / hk * (2.0 + 1.6 * dn), 0.0, 1.0) * 0.9          # dirt line at the paving
            base = np.maximum(base, np.clip(nrm[:, 1], 0, 1) * (0.12 + 0.25 * dn))           # damp back face
            if len(sx):
                d = np.min(np.abs(co[:, 0:1] - sx[None, :]), axis=1)
                top = np.exp(-(d / 0.09) ** 2) * np.exp(-(co[:, 1] / 0.1) ** 2)
                streak = np.exp(-(d / 0.05) ** 2) * (0.4 + 0.6 * rust) * (0.5 + 0.5 * dn)
                base = np.maximum(base, np.maximum(0.75 * top, 0.75 * streak))
            return np.clip(base, 0.0, 1.0)

        def kerb_moss(co, nrm):
            mn = np.asarray(moss_nm(co, nrm))
            back = np.clip(nrm[:, 1], 0, 1) * 0.6 + np.clip(nrm[:, 2], 0, 1) * np.clip(co[:, 1] / (KW / 2), 0.0, 1.0) * 0.45
            foot_ = np.clip(1.0 - co[:, 2] / (hk * 0.45), 0.0, 1.0) * 0.6
            return np.clip((back + foot_) * mn + np.clip(nrm[:, 2], 0, 1) * 0.25 * mn * mn, 0.0, 1.0)

        for o in S["kerb"]:
            assign(o, stone)
            paint(o, grime=kerb_grime, special=kerb_moss)
        for o in S["mortar"]:
            assign(o, mortar)
            paint(o, grime=lambda co, nrm: np.clip(1.0 - co[:, 2] / hk * 1.6, 0.1, 0.8), special=kerb_moss)
        for o in S["pads"]:
            assign(o, pad_stone)
            paint(o, grime=lambda co, nrm: 0.1 + 0.3 * np.asarray(dirt_nm(co, nrm)),
                  special=lambda co, nrm: 0.5 * np.asarray(moss_nm(co, nrm)))

    socket("end_l", "attach", (-L / 2, 0, H), rot=(0, 0, -90), role="railing_end")
    socket("end_r", "attach", (L / 2, 0, H), rot=(0, 0, 90), role="railing_end")
