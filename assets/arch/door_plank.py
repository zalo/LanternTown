"""Heavy ledged-and-braced plank door hung in a projecting stone or pegged-timber frame.

Built the way a joiner and a mason would: individually cut vertical boards with V-joints clenched to
the rear ledges by rows of forged rose-head studs on diamond roves, wrought strap hinges with split
scrolled ends riding on pintles driven into the jamb, a drop-ring handle with rose and keyhole
escutcheon, a small spy grille (square bars set in an iron-lined opening, amber horn panes and a
masked lamp box behind them that throws a narrow beam out through the bars) and a foot-worn step.  The frame is long-and-short jamb stones
bedded in lime mortar on plinths, with imposts and a voussoir arch (or lintel and drip cap), or
stop-chamfered oak posts on stone pads carrying an oversailing pegged lintel and curved arch braces.
The leaf opens outwards about the pintle axis (`ajar`).  Nothing may go behind the wall plane, so the
hallway seen through the open door is a true-perspective relief: a real vaulted passage (flagged
floor, ribs, inner door, wall sconce) built at full size and projected towards a viewer on the latch
side into the depth of the frame.
Origin: threshold centre on the wall plane; `width` x `height` is the outer frame size.
"""
import math
import bmesh
import numpy as np
from tk import *

META = {"name": "Arched plank door", "class": "component", "mount": "wall_ground", "focus": 0.62,
        "tri_budget": 14000}
PARAMS = {
    "width": {"default": 1.3, "min": 0.95, "max": 1.7, "desc": "outer frame width (m)"},
    "height": {"default": 2.45, "min": 2.0, "max": 3.2, "desc": "outer frame height from the ground (m)"},
    "head": {"default": "round", "choices": ["flat", "round", "pointed"], "desc": "shape of the door head"},
    "frame": {"default": "stone", "choices": ["stone", "timber"], "desc": "frame construction"},
    "frame_tone": {"default": "#b9ad98", "desc": "stone tint (stone frame) or oak stain (timber frame)"},
    "finish": {"default": "oak", "choices": ["oak", "weathered", "painted"], "desc": "leaf finish"},
    "tone": {"default": "#8a6242", "desc": "oak stain / weathered tint / paint colour of the leaf"},
    "studs": {"default": 5, "min": 0, "max": 8, "desc": "rows of forged studs (clenched through the rear ledges)"},
    "ajar": {"default": 0.0, "min": 0.0, "max": 40.0, "desc": "opening angle of the leaf, outwards (deg)"},
    "hinge_side": {"default": "left", "choices": ["left", "right"], "desc": "side the strap hinges are on"},
    "step": {"default": 0.15, "min": 0.0, "max": 0.22, "desc": "height of the threshold step (m, 0 = flush sill)"},
    "grille": {"default": True, "desc": "small barred spy grille at face height"},
    "lit": {"default": True, "desc": "hallway behind is lit (glowing grille, light through the gap when ajar)"},
    "intensity": {"default": 2.5, "min": 0.0, "max": 12.0, "desc": "hallway light reaching the door (cd through the grille)"},
    "number": {"default": "7", "desc": "house number fixed to the keystone / lintel ('' = none)"},
    "hardware": {"default": "iron", "choices": ["iron", "brass"], "desc": "metal of ring handle and escutcheon"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of wear, paint loss, rust and dishing of the step"},
}
VARIANTS = {
    "pointed_green": {"head": "pointed", "finish": "painted", "tone": "#3a5c48", "hardware": "brass", "width": 1.2,
                      "height": 2.65, "studs": 4, "frame_tone": "#a59c8e", "number": "12", "wear": 0.6, "seed": 4},
    "timber_flat": {"frame": "timber", "head": "flat", "frame_tone": "#8a6a4e", "finish": "weathered", "tone": "#b0a595",
                    "width": 1.15, "height": 2.25, "studs": 6, "grille": False, "lit": False, "hinge_side": "right",
                    "step": 0.06, "wear": 0.9, "number": "3", "seed": 11},
    "ajar_lit": {"ajar": 32.0, "hinge_side": "right", "tone": "#6a4a30", "width": 1.42, "height": 2.6, "intensity": 3.5, "studs": 4,
                 "number": "", "seed": 7},
    "timber_arch_red": {"frame": "timber", "head": "round", "frame_tone": "#6b4a2e", "finish": "painted", "tone": "#682d27",
                        "hardware": "brass", "width": 1.5, "height": 2.75, "studs": 5, "step": 0.2, "number": "21", "wear": 0.7,
                        "seed": 23},
}


def _shade(h, k):
    h = h.lstrip("#")
    return "#%02x%02x%02x" % tuple(min(255, int(int(h[i:i + 2], 16) * k)) for i in (0, 2, 4))


def _curl(u0, v0, head_deg, length, k0, k1, sign, n=13):
    """Scroll path: curvature grows from k0 to k1 (1/m) so the bar winds in on itself."""
    pts, th, x, y, ds = [], radians(head_deg), u0, v0, length / n
    for i in range(n + 1):
        pts.append((x, y))
        th += sign * (k0 + (k1 - k0) * ((i + 0.5) / n) ** 1.5) * ds
        x += cos(th) * ds
        y += sin(th) * ds
    return pts


def _loft(name, rings, closed=True, cap=True):
    """Quads between consecutive rings of equally many points (closed loops are capped)."""
    n = len(rings[0])
    verts = [tuple(v) for r in rings for v in r]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n if closed else n - 1):
            k = (j + 1) % n
            faces.append((i * n + j, i * n + k, (i + 1) * n + k, (i + 1) * n + j))
    if closed and cap:
        faces.append(tuple(range(n))[::-1])
        faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
        vol = 0.0                                          # orient the closed solid outwards by its signed volume
        for f in faces:
            p0 = verts[f[0]]
            for q in range(1, len(f) - 1):
                p1, p2 = verts[f[q]], verts[f[q + 1]]
                vol += (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) + p0[1] * (p1[2] * p2[0] - p1[0] * p2[2])
                        + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]))
        if vol < 0:
            faces = [f[::-1] for f in faces]
    return mesh_object(name, verts, faces)


def build(p, rng):
    W, H = p.width, p.height
    s = 1.0 if p.hinge_side == "left" else -1.0          # +1: hinges on the left (-X) jamb
    timber = p.frame == "timber"
    arched = p.head != "flat"
    D = 0.17                                             # frame projection from the wall
    jw = clamp(W * 0.105, 0.11, 0.16) if timber else clamp(W * 0.135, 0.14, 0.21)
    horn = 0.06 if timber else 0.0                       # lintel oversails the posts
    a = W / 2 - jw - horn                                # half clear opening
    st = p.step
    g = 0.008                                            # clearance leaf / frame
    sill_top = st + 0.022
    zb = sill_top + 0.01                                 # leaf bottom
    y_f = -(D - 0.045)                                   # leaf front face
    tb = 0.042                                           # board thickness
    wearK = p.wear
    jt = 0.009                                           # mortar joint
    seed = int(p.seed)

    # ---------------------------------------------------------------- head geometry
    c = 0.4 * a if p.head == "pointed" else 0.0          # arc centres are c beyond the axis
    R0 = a + c
    th_in = math.acos(c / R0)
    rise = math.sqrt(R0 * R0 - c * c) if arched else 0.0
    rt = jw * 0.95                                       # voussoir depth
    rt_k = rt + 0.03                                     # the keystone stands a little taller
    if timber:
        lh = clamp(jw * 1.65, 0.2, 0.27)                 # deep lintel
        drip_h = 0.05                                    # lead-covered drip board on top of the lintel
        z_under = H - lh - drip_h
        zs = z_under - (0.05 + rise if arched else 0.0)
    elif arched:
        zs = H - math.sqrt((R0 + rt_k) ** 2 - c * c)
    else:
        cap_h = 0.055
        lh = jw * 1.3
        zs = H - cap_h - lh

    def apt(d, ang):
        return (-c + (R0 + d) * cos(ang), zs + (R0 + d) * sin(ang))

    def top_at(x, inset=0.0):
        if not arched:
            return zs - inset
        r = R0 - inset
        return zs + math.sqrt(max(r * r - (abs(x) + c) ** 2, 0.0))

    def outline(d, zbot, n=7):
        if not arched:
            return [(a + d, zbot), (a + d, zs + d), (-a - d, zs + d), (-a - d, zbot)]
        th = math.acos(c / (R0 + d))
        right = [apt(d, th * i / n) for i in range(n + 1)]
        return [(a + d, zbot)] + right + [(-x, z) for (x, z) in reversed(right[:-1])] + [(-a - d, zbot)]

    # ---------------------------------------------------------------- materials
    stone_step = mat("stone_rough", "#9c9385")
    iron = mat("iron_wrought")
    hw = mat("brass") if p.hardware == "brass" else iron
    painted = p.finish == "painted"
    if painted:
        w_a = mat("wood_painted", p.tone, wear=1.5, rough=1.6)
        w_b = mat("wood_painted", _shade(p.tone, 0.84), wear=1.5, rough=1.6)
    elif p.finish == "weathered":
        w_a, w_b = mat("wood_weathered", p.tone), mat("wood_weathered", _shade(p.tone, 0.8))
    else:
        w_a, w_b = mat("wood_oak", p.tone), mat("wood_oak", _shade(p.tone, 0.78))
    w_back = w_b
    # lime mortar of the joints and lime render of the hallway (a timber frame leads into a stone passage)
    mortar = stone_step if timber else mat("mortar", "#c2b8a2")
    # amber horn panes set back in the grille box (glowing panes never shadow a light): the hall light sits in a
    # masked lamp box BEHIND them, so only the panes' own dim glow shows - brighter upper light, duller lower one
    pane_hi = mat("glass_frosted", "#d8842e", glow=0.82)
    pane_lo = mat("glass_frosted", "#b86a22", glow=0.42)

    fr = rng.fork("frame")
    stones, timbers, cores = [], [], []

    def clamp_env(o):
        """Keep masonry in front of the wall plane and on the paving after the hand-dressing noise."""
        apply_transform(o)
        transform_verts(o, lambda co: Vector((clamp(co.x, -W / 2, W / 2), min(co.y, 0.0), clamp(co.z, 0.0, H))))
        return o

    def knock(o, n):
        """Chipped arrises: push a few edge points of the front face back into the stone."""
        realize(o)
        vs = o.data.vertices
        lo = Vector((min(v.co.x for v in vs), min(v.co.y for v in vs), min(v.co.z for v in vs)))
        hi = Vector((max(v.co.x for v in vs), max(v.co.y for v in vs), max(v.co.z for v in vs)))
        ctr = (lo + hi) / 2
        cand = [v for v in vs if v.co.y < lo.y + 0.012 and (min(v.co.x - lo.x, hi.x - v.co.x) < 0.012 or min(v.co.z - lo.z, hi.z - v.co.z) < 0.012)]
        for _ in range(n):
            if not cand:
                break
            q = fr.pick(cand).co.copy()
            dep = fr.u(0.006, 0.013)
            for v in vs:
                dd = (v.co - q).length
                if dd < 0.024:
                    v.co = v.co + (ctr - v.co).normalized() * dep * (1.0 - dd / 0.03)

    def stone_poly(name, pts, depth, cham=0.005):
        o = extrude_profile(name, pts, depth, loc=(0, 0, 0), rot=(90, 0, 0), bevel=cham, segments=1)
        apply_transform(o)
        displace_noise(o, strength=0.003, scale=0.25, seed=int(fr.u(0, 999)))
        smooth(o, 45)
        stones.append(clamp_env(o))
        return o

    def block(name, size, loc, cham=0.0055, detail=0.15, lst=None, chips=0):
        o = stone_block(name, size, loc=loc, chamfer=cham, rough=0.0035, rng=fr, detail=detail)
        clamp_env(o)
        if chips:
            knock(o, chips)
        (stones if lst is None else lst).append(o)
        return o

    # ---------------------------------------------------------------- frame
    if not timber:
        tail = clamp(jw * 0.3, 0.04, 0.06)               # short stones stop this far inside the long ones
        for sx in (-1, 1):
            hp = fr.u(0.27, 0.33)
            block("plinth", (jw, D + 0.03, hp), (sx * (a + jw / 2), -(D + 0.03) / 2, hp / 2), cham=0.008, chips=2)
            z = hp + jt
            z_end = zs - (0.11 if arched else 0.0)
            k = int(fr.u(0, 2))
            while z_end - z > 0.02:
                long_ = k % 2 == 0
                h = fr.u(0.2, 0.27) if long_ else fr.u(0.3, 0.43)
                if z_end - (z + h) < 0.2:
                    h = z_end - z
                w = jw if long_ else jw - tail * fr.u(0.75, 1.0)
                dj = fr.u(0.0, 0.006) + (0.004 if long_ else 0.0)
                block("jamb", (w, D + dj, h - jt), (sx * (a + w / 2), -(D + dj) / 2, z + (h - jt) / 2), detail=0.1, chips=int(fr.u(1, 3.99)))
                z += h
                k += 1
            if arched:
                block("impost", (jw + 0.012, D + 0.024, 0.11), (sx * (a + jw / 2 - 0.006), -(D + 0.024) / 2, zs - 0.055), cham=0.008, chips=1)
            cw = jw - tail - 0.016
            cores.append(box("core", (cw, D - 0.004, zs - 0.03), loc=(sx * (a + 0.004 + cw / 2), -(D - 0.004) / 2, 0.02 + (zs - 0.03) / 2)))
        if arched:
            kap = (0.085 if p.head == "round" else 0.11) / R0                   # keystone half angle
            m = max(2, int(round(R0 * (th_in - kap) / 0.2)))
            cuts = [(th_in - kap) * (i / m) for i in range(m + 1)]
            for i in range(1, m):
                cuts[i] += fr.jitter(0.0, 0.12 * (th_in - kap) / m)
            ja = jt / (R0 + rt * 0.5) / 2
            for i in range(m):
                a0, a1 = cuts[i] + (2 * ja if i == 0 else ja), cuts[i + 1] - ja
                for sx in (1, -1):
                    ro = rt - (fr.u(0.02, 0.04) if (i + (sx > 0)) % 2 else 0.0)  # long and short voussoirs
                    inner = [apt(0, a0 + (a1 - a0) * k / 3) for k in range(4)]
                    outer = [apt(ro, a1 + (a0 - a1) * k / 3) for k in range(4)]
                    poly = inner + outer
                    pts = poly if sx > 0 else [(-x, z) for (x, z) in reversed(poly)]
                    stone_poly("voussoir", pts, D + fr.u(0.0, 0.008))
            th_out = math.acos(c / (R0 + rt_k))
            a0 = cuts[-1] + ja
            inner_r = [apt(0, a0 + (th_in - a0) * k / 3) for k in range(4)]
            outer_r = [apt(rt_k, a0 + (th_out - a0) * k / 3) for k in range(4)]
            inner_r[-1] = (0.0, inner_r[-1][1])
            outer_r[-1] = (0.0, outer_r[-1][1])
            poly = inner_r + [(-x, z) for (x, z) in reversed(inner_r[:-1])] + [(-x, z) for (x, z) in outer_r[:-1]] + list(reversed(outer_r))
            stone_poly("keystone", poly, D + 0.028, cham=0.007)
            # mortar bed of the arch ring
            n = 8
            thi, tho = math.acos(c / (R0 + 0.004)), math.acos(c / (R0 + rt - 0.05))
            ci = [apt(0.004, thi * i / n) for i in range(n + 1)]
            co_ = [apt(rt - 0.05, tho * i / n) for i in range(n + 1)]
            ci[-1] = (0.0, ci[-1][1])
            co_[-1] = (0.0, co_[-1][1])
            poly = ci + [(-x, z) for (x, z) in reversed(ci[:-1])] + [(-x, z) for (x, z) in co_[:-1]] + list(reversed(co_))
            cores.append(extrude_profile("core_arch", poly, D - 0.004, loc=(0, 0, 0), rot=(90, 0, 0)))
        else:
            block("lintel", (W, D + 0.012, lh - 0.002), (0, -(D + 0.012) / 2, zs + lh / 2), cham=0.008, detail=0.16, chips=3)
            block("drip_cap", (W, D + 0.05, cap_h - jt * 0.5), (0, -(D + 0.05) / 2, H - cap_h / 2 + jt * 0.25), cham=0.007, detail=0.2, chips=2)
            cores.append(box("core", (W - 0.03, D - 0.004, 0.04), loc=(0, -(D - 0.004) / 2, zs + lh)))
        for o in cores:
            clamp_env(o)
            assign(o, mortar)
            paint(o, grime=0.12)
    else:
        oak = mat("wood_oak", p.frame_tone)
        endgrain = mat("wood_endgrain", "#6b4a2e")
        pad_h = max(0.18, st + 0.06)
        Dl = D + 0.045                                    # the lintel stands proud of the posts
        pegs = []

        def peg(x, y_face, z):
            pegs.append(trs((x, y_face + 0.011, z), (90, fr.u(0, 90), 0), fr.vary(1.0, 0.08)))

        for sx in (-1, 1):
            block("pad", (jw + 0.06, D + 0.03, pad_h), (sx * (a - 0.005 + (jw + 0.06) / 2), -(D + 0.03) / 2, pad_h / 2), cham=0.012, detail=0.1, chips=2)
            # stop-chamfered post, jowled out under the lintel
            z0, z1 = pad_h - 0.003, z_under + 0.004
            zc0, zc1 = z0 + 0.16, (zs - 0.05 if arched else z_under - 0.22)
            zr = [z0, zc0, zc0 + 0.07]
            nmid = max(1, int(round((zc1 - zc0 - 0.14) / 0.33)))
            zr += [zc0 + 0.07 + (zc1 - zc0 - 0.14) * i / nmid for i in range(1, nmid + 1)]
            zr += [zc1, z1 - 0.14, z1 - 0.07, z1] if z1 - zc1 > 0.2 else [zc1, z1]
            rings = []
            for z in zr:
                ch = 0.042 if zc0 + 0.06 < z < zc1 - 0.06 else 0.004
                jo = 0.03 * smoothstep(z1 - 0.42, z1 - 0.1, z)
                xi, xo = sx * a, sx * (a + jw + jo)
                rings.append([(xi, 0.0, z), (xo, 0.0, z), (xo, -D + ch, z), (xo - sx * ch, -D, z), (xi + sx * ch, -D, z), (xi, -D + ch, z)])
            post = _loft("post", rings)
            displace_noise(post, strength=0.003, scale=0.5, seed=int(fr.u(0, 999)), axis_mask=(1, 1, 0))
            transform_verts(post, lambda co: Vector((clamp(co.x, -W / 2, W / 2), min(co.y, 0.0), co.z)))
            uv_grain(post, "Z")
            timbers.append(post)
            for k in range(2):                                                  # draw-bored pegs of the tenon
                peg(sx * (a + jw / 2) + fr.jitter(0.0, 0.02), -Dl, z_under + lh * (0.27 + 0.42 * k))
        # oversailing lintel, lower arris chamfered between the posts
        xr = [-W / 2, -W / 2 + 0.05, -a - 0.05, -a + 0.03]
        nmid = max(2, int(round(2 * a / 0.25)))
        xr += [-a + 0.03 + (2 * a - 0.06) * i / nmid for i in range(1, nmid + 1)] + [a + 0.05, W / 2 - 0.05, W / 2]
        rings = []
        for x in xr:
            ch = 0.036 if abs(x) < a - 0.02 else 0.005
            cut = 0.05 if abs(x) > W / 2 - 0.01 else 0.0    # the horns are cut back underneath
            rings.append([(x, 0.0, z_under + cut), (x, 0.0, z_under + lh), (x, -Dl + 0.012, z_under + lh), (x, -Dl, z_under + lh - 0.012),
                          (x, -Dl, z_under + ch + cut), (x, -Dl + ch, z_under + cut)])
        lint = _loft("lintel", rings)
        displace_noise(lint, strength=0.0035, scale=0.45, seed=int(fr.u(0, 999)), axis_mask=(0, 1, 1))
        transform_verts(lint, lambda co: Vector((co.x, min(co.y, 0.0), co.z)))
        uv_grain(lint, "X")
        timbers.append(lint)
        # drip board: a sloping plank dressed with lead, nailed along the wall edge
        dbp = [(0.0, 0.0), (-(Dl + 0.04), 0.0), (-(Dl + 0.045), 0.006), (-(Dl + 0.043), 0.014), (0.0, drip_h)]
        db = extrude_profile("drip_board", dbp, W, loc=(-W / 2, 0, z_under + lh), rot=(90, 0, 90), bevel=0.003, segments=1)
        apply_transform(db)
        densify(db, 0.2)
        displace_noise(db, strength=0.003, scale=0.12, seed=int(fr.u(0, 999)), axis_mask=(0, 0, 1))
        transform_verts(db, lambda co: Vector((co.x, min(co.y, 0.0), min(co.z, H))))
        assign(db, mat("lead"))
        smooth(db, 50)
        paint(db, grime=mask_mul(noise_mask(0.15, seed=seed + 21), 0.6))
        db_nails = [trs((-W / 2 + 0.06 + (W - 0.12) * k / 5, -0.03, z_under + lh + drip_h * (1 - 0.03 / (Dl + 0.043)) - 0.001), (-14, 0, 0)) for k in range(6)]
        dn = instances_to_mesh("drip_nails", cylinder("clout", 0.007, 0.004, verts=6, radius2=0.005), db_nails, keep_base=False)
        assign(dn, mat("lead"))
        if arched:
            # curved solid arch braces, tenoned and pegged into post and lintel; boarded corner behind
            tbr = clamp(0.25 * a, 0.1, 0.15)
            th1 = math.acos(R0 / (R0 + tbr))
            th2 = math.asin(min(1.0, (z_under - zs) / (R0 + tbr)))
            n = 9
            curve = [apt(0, th_in * i / n) for i in range(n + 1)]
            curve[-1] = (0.001, curve[-1][1])
            split = th2 > th1 + 0.06
            if split:
                back = [apt(tbr, th2 + (th1 - th2) * i / 4) for i in range(5)]
                back[0] = (back[0][0], z_under)
                back[-1] = (a, back[-1][1])
            else:
                back = [(a, z_under)]
            poly = list(reversed(curve + [(0.001, z_under)] + back))             # CCW
            bd = 0.118
            y0 = -(D - 0.016 - bd)
            for sx in (1, -1):
                pts = poly if sx > 0 else [(-x, z) for (x, z) in reversed(poly)]
                sp = extrude_profile("arch_brace", pts, bd, loc=(0, y0, 0), rot=(90, 0, 0), bevel=0.009, segments=1)
                apply_transform(sp)
                uv_grain(sp, (-sx * 0.72, 0, 0.69))
                timbers.append(sp)
                if split:
                    x_in, z_lo = back[0][0] - 0.03, back[-1][1] - 0.03
                    inf = box("infill", (a - x_in + 0.01, 0.02, z_under - z_lo + 0.01), loc=(sx * (a + x_in) / 2, -0.04, (z_under + z_lo) / 2))
                    apply_transform(inf)
                    uv_grain(inf, "Z")
                    timbers.append(inf)
                    for ang in (th1 + 0.13, th2 - 0.14):
                        qx, qz = apt(tbr * 0.5, ang)
                        peg(sx * qx, y0 - bd, qz)
                    peg(sx * apt(tbr * 0.5, th2)[0], -Dl, z_under + 0.05)
                else:
                    peg(sx * a * 0.5, y0 - bd, (top_at(a * 0.5) + z_under) / 2)
        for o in timbers:
            assign(o, oak)
        assign_faces(lint, endgrain, lambda cc, nn: abs(nn.x) > 0.9)
        pg = cylinder("peg", 0.0125, 0.054, verts=7, radius2=0.0105)
        pgm = instances_to_mesh("pegs", pg, pegs, keep_base=False)
        assign(pgm, endgrain)
        paint(pgm, wear=0.4)
        for o in timbers:
            paint(o, grime=mask_mul(height_ramp(0.0, 0.9), 0.8), wear=mask_mul(height_ramp(1.3, 0.7), height_ramp(0.6, 1.2), 0.35 * wearK))

    # ---------------------------------------------------------------- threshold: sill + worn step
    def dish(o, amount, half_w):
        def f(co):
            if co.z > 0.012:
                co = Vector((co.x, co.y, co.z - amount * math.exp(-(co.x / half_w) ** 2)))
            return co
        transform_verts(o, f)

    step_objs = []
    sill = block("sill", (2 * a + 0.004, D + 0.03, sill_top), (0, -(D + 0.03) / 2, sill_top / 2), cham=0.008, detail=0.16, lst=step_objs)
    dish(sill, 0.007 * wearK, a * 0.55)
    if st > 0.03:
        tread_d = 0.34
        tread_w = 2 * a + 0.1 if timber else W            # timber posts stand on their own pads beside the step
        tread = block("step", (tread_w, tread_d, st), (0, -(D + 0.03) - tread_d / 2, st / 2), cham=0.014, detail=0.13, lst=step_objs, chips=3)
        dish(tread, min(0.018 * wearK, st * 0.25), a * 0.6)

    def tread_wear(co, nn):
        return np.clip(nn[:, 2], 0, 1) * np.exp(-(co[:, 0] / (a * 0.7)) ** 2) * (0.2 + 0.35 * wearK)

    for o in step_objs:
        assign(o, stone_step)
        paint(o, wear=tread_wear, grime=mask_mul(facing((0, -1, 0)), 0.7),
              special=mask_mul(height_ramp(0.0, 0.2), noise_mask(0.18, seed=seed + 5, lo=0.4, hi=0.75), 0.6))

    # ---- stone finish: two quarry tones, per-block dirt, rain streaks under the imposts, hand polish
    if timber:
        stone_a = stone_b = stone_step
    else:
        stone_a, stone_b = mat("stone_smooth", p.frame_tone), mat("stone_smooth", _shade(p.frame_tone, 0.82))
    z_imp = zs - (0.11 if arched else 0.0)
    streak = noise_mask(0.08, seed=seed + 31, lo=0.36, hi=0.56)
    blotch = noise_mask(0.28, seed=seed + 32, lo=0.3, hi=0.8)
    z_hand = st + 1.0

    def stone_grime(g0):
        def f(co, nn):
            z = co[:, 2]
            splash = np.clip(1.0 - z / 0.8, 0, 1) * (0.4 if timber else 0.65)
            sq = co * np.array([1.0, 1.0, 0.08])
            under = np.clip(1.0 - (z_imp - z) / 0.65, 0, 1) * (z < z_imp + 0.01)
            run = under * streak(sq, nn) * 1.0
            soot = np.clip((z - zs) / max(H - zs, 0.1), 0, 1) * 0.3 * (nn[:, 2] < 0.3)
            latch = (co[:, 0] * s > a - 0.03) * (co[:, 0] * s < a + 0.09) * np.exp(-((z - z_hand) / 0.38) ** 2) * 0.4
            return np.clip(splash + run + soot + latch + g0 + blotch(co, nn) * 0.14, 0, 1)
        return f

    def stone_wear(co, nn):
        z = co[:, 2]
        hand = (co[:, 0] * s > a - 0.03) * (co[:, 0] * s < a + 0.06) * (co[:, 1] < -(D - 0.07)) * np.exp(-((z - z_hand) / 0.2) ** 2)
        return np.clip(hand * (0.25 + 0.7 * wearK), 0, 1)

    for o in stones:
        assign(o, stone_a if fr.chance(0.55) else stone_b)
        paint(o, grime=stone_grime(fr.u(0.0, 0.3)), wear=stone_wear,
              special=mask_mul(height_ramp(0.0, 0.55), noise_mask(0.16, seed=seed + 3, lo=0.4, hi=0.7), 0.75))

    # ================================================================ the leaf (built closed, then swung)
    lr = rng.fork("leaf")
    lw = 2 * (a - g)
    leaf = []

    def X(u):                                             # u = distance from the hinge edge of the leaf
        return s * (-(a - g) + u)

    def L(o):
        apply_transform(o)
        leaf.append(o)
        return o

    # ---- ledges at the hinge heights (the straps are bolted through them), boards, grille position
    leaf_top = zs if arched else zs - g                   # top of the hinge edge
    z_bot = zb + 0.28
    z_top = zs - 0.08 if arched else leaf_top - 0.22
    zl = [z_bot, (z_bot + z_top) / 2, z_top]
    led_h = 0.15
    z_head = zs + rise * 0.47 if arched and rise > 0.36 else None   # short fourth ledge up in an arched head
    nb = max(5, int(round(lw / 0.15)))
    if nb % 2 == 0:
        nb += 1
    ws = [lr.vary(1.0, 0.2) for _ in range(nb)]
    ic = nb // 2
    ws[ic] = 1.25
    tot = sum(ws)
    ue = [0.0]
    for w in ws:
        ue.append(ue[-1] + lw * w / tot)
    gh = 0.22
    kL = clamp(lw / 0.9, 0.8, 1.15)
    tipL = 0.25 * kL                                      # length of the scrolled, pointed end of a strap
    strap_long = (zl[2] - zl[0]) > 1.75
    gx0, gx1 = sorted((X(ue[ic]), X(ue[ic + 1])))
    gcx, gw = (gx0 + gx1) / 2, gx1 - gx0
    gz = min(sill_top + 1.42, z_top - 0.195) if p.grille else None   # the grille hangs just under the top strap
    has_grille = gz is not None and gz > sill_top + 1.05 and gz - gh / 2 > zl[1] + led_h / 2 + 0.03
    # straps: (height, length, start of the scrolled end)
    Ls = clamp(lw * 0.88, 0.42, 1.2)
    u0 = Ls - tipL
    straps = [(zl[0], Ls, u0)]
    if has_grille:
        u0t = max(u0, ue[ic + 1] + 0.085)
        straps.append((zl[2], min(u0t + tipL, lw - 0.04), u0t))
    else:
        straps.append((zl[2], Ls, u0))
    if strap_long:
        straps.append((zl[1], Ls * 0.62, Ls * 0.62 - tipL * 0.9))
    curl0 = _curl(0.0, 0.006, 30, 0.33 * kL, 3.0, 70.0, 1)
    scr_ext = max(v for (u, v) in curl0) + 0.014           # how far the scrolls reach above / below a strap

    hand_x, hand_z = X(lw - 0.13), zb + 1.0
    joints_x = np.array([X(u) for u in ue])
    flake = noise_mask(0.11, seed=seed + 41, lo=0.25, hi=0.7)
    fade = noise_mask(0.5, seed=seed + 9, lo=0.3, hi=0.8)

    def board_grime(co, nn):
        h = np.clip(1.0 - (co[:, 2] - zb) / 0.6, 0, 1) ** 1.5
        return np.clip(h * 0.9 + 0.12, 0, 1)

    def board_wear(co, nn):
        dj = np.min(np.abs(co[:, 0][:, None] - joints_x[None, :]), axis=1)
        edge = np.clip(1.0 - dj / 0.011, 0, 1)
        fl = flake(co, nn)
        d = np.hypot(co[:, 0] - hand_x, co[:, 2] - (hand_z - 0.06))
        hand = np.clip(1.0 - d / 0.3, 0, 1)
        kick = np.clip(1.0 - (co[:, 2] - zb) / 0.34, 0, 1)
        if painted:                                       # paint flakes off along the V-joints, the kick zone and round the handle
            v = edge * (0.25 + 0.75 * fl) * 1.1 + kick * (0.35 + 0.65 * fl) * 0.9 + hand * (0.5 + 0.5 * fl) * 0.9 + fl * 0.12
            return np.clip(v * (0.25 + 1.0 * wearK), 0, 1)
        v = hand * 0.9 + kick * 0.5 + edge * 0.25 * fl
        return np.clip(v * (0.3 + 0.9 * wearK), 0, 1) * (0.35 if p.finish == "weathered" else 1.0)

    # shared row heights of the board meshes (dense at the foot, tight loops round the grille so the
    # baked shadow of the iron frame stays under the frame)
    must = [zb + 0.1, zb + 0.2, zb + 0.33, hand_z - 0.3, hand_z - 0.12, hand_z + 0.1]
    if has_grille:
        must += [gz - gh / 2 - 0.06, gz - gh / 2 - 0.036, gz + gh / 2 + 0.036, gz + gh / 2 + 0.06]
    z = zb + 0.33
    levels = sorted(must)
    while z < H:
        z += 0.16
        if all(abs(z - q) > 0.07 for q in levels):
            levels.append(z)
    levels = sorted(levels)

    def board(x0, x1, zlo, top_fn, dy, curved):
        cb, eb = 0.0045, min(0.02, (x1 - x0) * 0.25)
        yf, yb = y_f + dy, y_f + dy + tb
        prof = [(x0, yb), (x0, yf + cb), (x0 + cb, yf), (x0 + eb, yf)]
        if curved:
            xm = (x0 + x1) / 2
            prof.append((0.0 if p.head == "pointed" and abs(xm) < (x1 - x0) / 2 - eb - 0.01 else xm, yf))
        prof += [(x1 - eb, yf), (x1 - cb, yf), (x1, yf + cb), (x1, yb)]
        tops = [top_fn(x) for (x, y) in prof]
        zr = [zlo] + [q for q in levels if zlo + 0.03 < q < min(tops) - 0.035]
        rings = [[(x, y, q) for (x, y) in prof] for q in zr] + [[(x, y, t) for (x, y), t in zip(prof, tops)]]
        return _loft("board", rings)

    board_dy = []
    for i in range(nb):
        x0, x1 = sorted((X(ue[i]), X(ue[i + 1])))
        x0 += 0.0004
        x1 -= 0.0004
        dy = lr.jitter(0.0, 0.0015)
        board_dy.append(dy)
        z0 = zb + lr.u(0.0, 0.005)
        full_top = (lambda x: top_at(x, g)) if arched else (lambda x: zs - g)
        if has_grille and i == ic:
            pieces = [board(x0, x1, z0, lambda x: gz - gh / 2, dy, False), board(x0, x1, gz + gh / 2, full_top, dy, arched)]
        else:
            pieces = [board(x0, x1, z0, full_top, dy, arched)]
        m_i = w_a if lr.chance(0.55) else w_b
        for o in pieces:
            L(o)
            assign(o, m_i)
            uv_grain(o, "Z")
            paint(o, grime=board_grime, wear=board_wear, special=mask_mul(fade, 0.75) if painted else mask_mul(fade, height_ramp(zb + 0.4, zb), 0.5))

    # loose tongues in the board edges (no daylight - or lamplight - through the joints)
    for i in range(1, nb):
        xj = X(ue[i])
        zt = top_at(xj, g) - 0.012
        o = L(box("tongue", (0.022, 0.006, zt - zb - 0.01), loc=(xj, y_f + tb * 0.62, (zb + 0.01 + zt) / 2)))
        assign(o, w_back)

    # ---- ledges and braces on the back
    y_led = y_f + tb + 0.015
    for k, z in enumerate(zl):
        o = L(box("ledge", (lw - 0.07, 0.03, led_h), loc=(0, y_led, z), bevel=0.005, segments=1))
        assign(o, w_back)
        uv_grain(o, "X")
    if z_head is not None:
        xh = math.sqrt(max((R0 - g) ** 2 - (z_head + 0.065 - zs) ** 2, 0.0)) - c
        if xh > 0.12:
            o = L(box("ledge", (2 * xh - 0.07, 0.03, 0.11), loc=(0, y_led, z_head), bevel=0.005, segments=1))
            assign(o, w_back)
            uv_grain(o, "X")
        else:
            z_head = None
    for k in range(2):
        zA, zB = zl[k] + led_h / 2 + 0.004, zl[k + 1] - led_h / 2 - 0.004
        if has_grille and zA < gz < zB:
            continue
        xA, xB = X(0.1), X(lw - 0.1)
        ang = atan2(zB - zA - 0.1, xB - xA)
        ln = hypot(xB - xA, zB - zA - 0.1)
        o = L(box("brace", (ln, 0.028, 0.1), loc=(0, y_led, (zA + zB) / 2), rot=(0, -degrees(ang), 0), bevel=0.005, segments=1))
        assign(o, w_back)
        uv_grain(o, (cos(ang), 0, sin(ang)))

    # ---- weather board along the foot
    wbp = [(0.0, 0.0), (-0.03, 0.0), (-0.033, 0.014), (-0.009, 0.078), (0.0, 0.084)]
    o = L(extrude_profile("weatherboard", wbp, lw - 0.03, loc=(-(lw - 0.03) / 2, y_f, zb + 0.014), rot=(90, 0, 90), bevel=0.003, segments=1))
    assign(o, w_b)
    uv_grain(o, "X")
    paint(o, grime=0.75, wear=(0.9 if painted else 0.5) * wearK)

    # ---- strap hinges on pintles
    px, py = s * (-(a - 0.012)), y_f - 0.012               # pintle axis
    sw, thk = 0.052, 0.007
    rivets, irons, static_iron = [], [], []
    bar = [(-0.0095, -0.0032), (0.0095, -0.0032), (0.0095, 0.0032), (-0.0095, 0.0032)]
    for k, (zst, Lk, u0k) in enumerate(straps):
        hr = lr.fork("strap%d" % k)
        w1 = 0.03
        top = [(0.0, sw / 2), (0.33 * u0k, lerp(sw, w1, 0.33) / 2 + hr.jitter(0, 0.0012)), (0.66 * u0k, lerp(sw, w1, 0.66) / 2 + hr.jitter(0, 0.0012)),
               (u0k + 0.02, w1 / 2), (lerp(u0k, Lk, 0.42), 0.024), (lerp(u0k, Lk, 0.68), 0.017), (Lk, 0.0)]
        poly = top + [(u, -v) for (u, v) in reversed(top[:-1])]
        pts = [(X(u), zst + v) for (u, v) in poly]
        o = L(extrude_profile("strap", pts if s > 0 else pts[::-1], thk, loc=(0, y_f, 0), rot=(90, 0, 0), bevel=0.0015, segments=1))
        irons.append(o)
        for sg in (1, -1):                                 # split scrolled ends
            cp = [(u0k + u, sg * v) for (u, v) in curl0]
            path = [(X(u), y_f - thk / 2, zst + v) for (u, v) in cp]
            sc = L(sweep("scroll", path, bar, up=(0, 1, 0), smooth_path=False, scale_fn=lambda t: 1.0 - 0.45 * t))
            irons.append(sc)
            ue_, ve_ = cp[-1]
            irons.append(L(cylinder("scroll_end", 0.0075, thk * 0.9, loc=(X(ue_), y_f - thk / 2, zst + ve_), rot=(90, 0, 0), verts=7)))
        nq = 5 if u0k > 0.6 else 4
        for q in range(nq):
            u = 0.065 + (u0k - 0.02 - 0.065) * q / (nq - 1)
            rivets.append(trs((X(u), y_f - thk, zst + hr.jitter(0, 0.002)), (90, hr.u(0, 60), 0), 1.0 if q else 1.25))
        irons.append(L(cylinder("eye", 0.0128, sw, loc=(px, py, zst), verts=10, bevel=0.0015, segments=1)))
        static_iron.append(cylinder("pintle_pin", 0.0072, sw + 0.04, loc=(px, py, zst - 0.006), verts=8, bevel=0.002, segments=1))
        static_iron.append(box("pintle_shank", (0.052, 0.02, 0.02), loc=(px - s * 0.026, py, zst - sw / 2 - 0.016), bevel=0.003, segments=1))

    # ---- drop-ring handle, rose, striker and keyhole escutcheon
    Rm = 0.05
    rose = L(lathe("rose", [(0.0, 0.0), (0.05, 0.0), (0.053, 0.003), (0.047, 0.0065), (0.036, 0.006), (0.031, 0.011), (0.02, 0.012), (0.016, 0.02),
                            (0.0085, 0.022), (0.0075, 0.034), (0.011, 0.04), (0.0, 0.046)], segments=14,
               loc=(hand_x, y_f, hand_z), rot=(90, 0, 0)))
    ring = L(torus("ring", major=Rm, minor=0.0068, loc=(hand_x, y_f - 0.029, hand_z - Rm + 0.004), rot=(90, lr.jitter(0, 4), 0), major_seg=22, minor_seg=6))
    kh = circle_pts(0.0055, 7, a0=-50, a1=230, center=(0, 0.008)) + [(-0.003, -0.018), (0.003, -0.018)]
    esc = L(shape_extrude("escutcheon", rounded_rect(0.042, 0.092, 0.017, 2), [kh], depth=0.004,
                          loc=(hand_x, y_f, hand_z - 0.235), rot=(90, 0, 0), bevel=0.0008))
    for o in (rose, ring, esc):
        assign(o, hw)
        smooth(o, 40)
    paint(ring, wear=0.75, mode="set")
    paint(rose, wear=0.3)
    rivets.append(trs((hand_x, y_f, hand_z - 2 * Rm + 0.004), (90, 0, 0), 1.5))
    for dz in (0.036, -0.036):
        rivets.append(trs((hand_x, y_f - 0.004, hand_z - 0.235 + dz), (90, 0, 0), 0.55))

    # ---- spy grille: iron-lined opening through the board, square bars set in its thickness, horn pane behind
    if has_grille:
        rect = [(-gw / 2, -gh / 2), (gw / 2, -gh / 2), (gw / 2, gh / 2), (-gw / 2, gh / 2)]
        inner = [(x * (1 - 0.009 / gw), z * (1 - 0.009 / gh)) for (x, z) in rect]
        irons.append(L(shape_extrude("grille_frame", rounded_rect(gw + 0.062, gh + 0.062, 0.007, 2), [inner],
                                     depth=0.005, loc=(gcx, y_f, gz), rot=(90, 0, 0), bevel=0.001)))
        # the iron liner runs on behind the board as a shallow box that carries the pane (and shades the lamp light sideways)
        irons.append(L(shape_extrude("grille_liner", [(x * 1.004, z * 1.004) for (x, z) in rect], [inner], depth=0.072,
                                     loc=(gcx, y_f + 0.072, gz), rot=(90, 0, 0))))
        y_bar = y_f + tb * 0.45
        nbar = max(2, int(round(gw / 0.05)) - 1)
        for k in range(nbar):
            xk = gcx - gw / 2 + gw * (k + 1) / (nbar + 1)
            irons.append(L(box("bar", (0.013, 0.013, gh + 0.004), loc=(xk, y_bar, gz), rot=(0, 0, 45))))
        irons.append(L(box("bar_cross", (gw + 0.004, 0.006, 0.022), loc=(gcx, y_bar, gz + 0.004))))
        for sx in (-1, 1):
            for sz in (-1, 1):
                rivets.append(trs((gcx + sx * (gw / 2 + 0.015), y_f - 0.005, gz + sz * (gh / 2 + 0.015)), (90, 0, 0), 0.8))
        if p.lit:
            zsp = gz - gh * 0.1                             # two horn sheets, the lower one thicker and darker
            for z0_, z1_, m_ in ((gz - gh / 2 + 0.003, zsp, pane_lo), (zsp, gz + gh / 2 - 0.003, pane_hi)):
                # one glowing sheet per pane (a thin glowing box is two sheets that fight each other, and its edges lie
                # in the liner); CLEARANCE RULE: >= 4 mm to whatever lies parallel in front of or behind a glowing sheet
                po = box("pane", (gw - 0.006, 0.003, z1_ - z0_), loc=(gcx, y_f + 0.034, (z0_ + z1_) / 2))
                bm = bmesh.new()
                bm.from_mesh(po.data)
                bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y < 0.5], context="FACES")
                bm.to_mesh(po.data)
                bm.free()
                po.data.flip_normals()
                assign(L(po), m_)
            irons.append(L(box("came", (gw - 0.006, 0.0088, 0.007), loc=(gcx, y_f + 0.0354, zsp))))   # lead came between them, clasping the sheets
            # lamp box behind the panes: a masking plate with a small window lets the light out as a narrow beam
            # down through the bars onto the step (never onto the reveal or sideways along the wall), a back plate closes the box
            lamp = (gcx + (0.0 if nbar % 2 == 0 else gw / (nbar + 1) / 2), y_f + 0.058, gz + gh / 2 - 0.03)
            lx_, lz_ = lamp[0] - gcx, lamp[2] - gz
            hole = [(lx_ - 0.0103, lz_ - 0.045), (lx_ + 0.0103, lz_ - 0.045), (lx_ + 0.0103, lz_ + 0.001), (lx_ - 0.0103, lz_ + 0.001)]
            irons.append(L(shape_extrude("lamp_mask", [(x * 1.004, z * 1.004) for (x, z) in rect], [hole], depth=0.003,
                                         loc=(gcx, y_f + 0.043, gz), rot=(90, 0, 0))))
            irons.append(L(box("lamp_back", (gw * 1.008, 0.004, gh * 1.008), loc=(gcx, y_f + 0.072, gz))))
        else:
            pane = L(box("hatch", (gw + 0.05, 0.016, gh + 0.05), loc=(gcx, y_f + tb + 0.013, gz), bevel=0.003, segments=1))
            assign(pane, w_back)
            uv_grain(pane, "Z")

    # ---- forged rose-head studs on diamond roves, in rows clenched through the ledges
    b_, t_, h_, rv = 0.014, 0.0034, 0.0088, 0.023
    sv = [(-b_, -b_, 0.001), (b_, -b_, 0.001), (b_, b_, 0.001), (-b_, b_, 0.001), (-t_, -t_, h_), (t_, -t_, h_), (t_, t_, h_), (-t_, t_, h_),
          (rv, 0, 0.0008), (0, rv, 0.0008), (-rv, 0, 0.0008), (0, -rv, 0.0008)]
    stud_base = mesh_object("stud", sv, [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7), (8, 9, 10, 11)])
    assign(stud_base, iron)
    flat(stud_base)
    paint(stud_base, wear=lambda co, nn: np.where(co[:, 2] > 0.005, 0.9, 0.15) * (0.5 + 0.5 * wearK),
          special=lambda co, nn: np.where(co[:, 2] < 0.0009, 0.6, np.where(co[:, 2] < 0.005, 0.3, 0.0)), mode="set")
    of = 0.052
    B_, M_, T_ = zl
    if z_head is not None:
        table = {1: [M_], 2: [B_ + of, T_ - of], 3: [B_ + of, M_, T_ - of], 4: [B_ + of, M_, T_ - of, z_head],
                 5: [B_ - of, B_ + of, M_, T_ - of, z_head], 6: [B_ - of, B_ + of, M_ - of, M_ + of, T_ - of, z_head],
                 7: [B_ - of, B_ + of, M_ - of, M_ + of, T_ - of, T_ + of, z_head],
                 8: [B_ - of, B_ + of, M_ - of, M_ + of, T_ - of, T_ + of, z_head - 0.03, z_head + 0.03]}
    else:
        table = {1: [M_], 2: [B_ + of, T_ - of], 3: [B_ + of, M_, T_ - of], 4: [B_ + of, M_ - of, M_ + of, T_ - of],
                 5: [B_ - of, B_ + of, M_, T_ - of, T_ + of], 6: [B_ - of, B_ + of, M_ - of, M_ + of, T_ - of, T_ + of],
                 7: [B_ - of, B_ + of, M_ - of, M_ + of, T_ - of, T_ + of, (B_ + M_) / 2],
                 8: [B_ - of, B_ + of, M_ - of, M_ + of, T_ - of, T_ + of, (B_ + M_) / 2, (M_ + T_) / 2]}
    studs = []
    for zr in table.get(int(p.studs), []):
        if strap_long and abs(zr - M_) < 0.01:
            zr = M_ + of
        for i in range(nb):
            w = ue[i + 1] - ue[i]
            us = [ue[i] + w * 0.25, ue[i + 1] - w * 0.25] if w > 0.125 else [ue[i] + w / 2]
            for u in us:
                x, z = X(u) + lr.jitter(0, 0.004), zr + lr.jitter(0, 0.004)
                if z > top_at(x, g) - 0.05 or abs(x) > a - g - 0.02:
                    continue
                if has_grille and abs(x - gcx) < gw / 2 + 0.055 and abs(z - gz) < gh / 2 + 0.055:
                    continue
                if any(u < Lk + 0.02 and abs(z - q) < (scr_ext + 0.02 if u > u0k - 0.03 else 0.044) for (q, Lk, u0k) in straps):
                    continue
                if abs(x - hand_x) < 0.08 and hand_z - 0.3 < z < hand_z + 0.08:
                    continue
                if z < zb + 0.115:
                    continue
                studs.append(trs((x, y_f + board_dy[i], z), (90, lr.jitter(0, 14), 0), lr.vary(1.0, 0.1)))
    if studs:
        sm = L(instances_to_mesh("studs", stud_base, studs, keep_base=False))
        flat(sm)
        static_studs = sm
    else:
        delete(stud_base)
    irons.append(L(instances_to_mesh("rivets", lathe("rivet", [(0.0085, 0.0), (0.0075, 0.0035), (0.0, 0.0064)], segments=6), rivets, keep_base=False)))

    def rust(co, nn):
        low = np.clip(1.0 - (co[:, 2] - zb) / 0.7, 0, 1)
        return np.clip(0.08 + 0.45 * low + 0.3 * wearK, 0, 1)

    for o in irons + static_iron:
        assign(o, iron)
        paint(o, special=mask_mul(rust, noise_mask(0.07, seed=seed + 13, lo=0.3, hi=0.75)), grime=0.25)

    # ---- swing the leaf about the pintle axis and tag it as a part
    ang = -s * p.ajar if p.ajar > 0.05 else 0.0
    ca, sa = cos(radians(ang)), sin(radians(ang))
    off = (px - (ca * px - sa * py), py - (sa * px + ca * py), 0.0)
    if ang:
        for o in leaf:
            place(o, loc=off, rot=(0, 0, ang))
    # amp=0: the runtime sways every tagged part by default (3 deg, like a hanging sign) - a door leaf must stand still
    part(leaf, "leaf", pivot=(px, py, 0.0), motion=None, axis="Z", amp=0.0, speed=0.0)

    # ================================================================ the hallway seen through the open door
    lit_hall = p.lit and p.intensity > 0.05
    if p.ajar > 0.5:
        hr_ = rng.fork("hall")
        Dr = 0.045 + 0.06 * smoothstep(3.0, 14.0, p.ajar)   # relief depth available behind the swung leaf
        y_open = -(0.004 + Dr)
        E, ex, ez = 3.2, s * 1.4, sill_top + 1.48          # design eye: someone walking up on the latch side
        d_f = 1.75                                         # length of the passage (m)
        alpha = (d_f - Dr) / (d_f * (E + Dr))
        z_fl = sill_top - 0.004

        def squash(co):
            d = max(co.y, 0.0)
            lam = 1.0 / (1.0 + alpha * d)
            x = clamp(ex + lam * (co.x - ex), -(a + 0.018), a + 0.018)
            ztop = top_at(x, -0.018) if arched else zs + 0.018
            z = clamp(ez + lam * (co.z - ez), sill_top - 0.012, ztop)
            return Vector((x, min(y_open + lam * (E + d) - E, -0.003), z))

        def hall_grime(base, k=0.45):
            def f(co, nn):
                d = np.clip(co[:, 1] / d_f, 0, 1)
                low = np.clip(1.0 - (co[:, 2] - z_fl) / 0.5, 0, 1) * 0.3
                return np.clip(base + k * d + low, 0, 1)
            return f

        def into_hall(o, m, base, grain=None):
            """Finish a part built at full size in passage space (y = depth) and project it into the relief."""
            apply_transform(o)
            if m is not None:
                assign(o, m)
            if grain:
                uv_grain(o, grain)
            realize(o)
            uv_keep(o)
            paint(o, grime=hall_grime(base), mode="set")
            transform_verts(o, squash)
            return o

        # vaulted (or flat-ceiled) passage with two stone ribs, flagged floor, darker end wall with an inner door
        prof = outline(0.015, z_fl)
        deps = [0.0, 0.22, 0.44, 0.62, 0.84, 1.08, 1.26, 1.5, d_f]
        tun = _loft("hall_walls", [[(x, d, z) for (x, z) in prof] for d in deps], closed=False)
        assign(tun, mortar)
        assign_faces(tun, stone_b, lambda cc, nn: 0.44 < cc.y < 0.62 or 1.08 < cc.y < 1.26)
        into_hall(tun, None, 0.12)
        end = mesh_object("hall_end", [(x, d_f, z) for (x, z) in prof], [tuple(range(len(prof)))])
        into_hall(end, mortar, 0.5)
        hwid = a + 0.015
        sub = mesh_object("hall_bed", [(-hwid, 0, z_fl - 0.012), (hwid, 0, z_fl - 0.012), (hwid, d_f, z_fl - 0.012), (-hwid, d_f, z_fl - 0.012)], [(0, 1, 2, 3)])
        into_hall(sub, mortar, 0.9)
        rows = [0.0, 0.36, 0.78, 1.24, d_f]
        for r_i in range(len(rows) - 1):
            d0, d1 = rows[r_i], rows[r_i + 1]
            nf = 2 if r_i % 2 else 3
            xsplit = [-hwid] + sorted(hr_.jitter(-hwid + 2 * hwid * (k + 1) / nf, 0.07) for k in range(nf - 1)) + [hwid]
            for k in range(nf):
                fx0, fx1 = xsplit[k], xsplit[k + 1]
                fl = box("flag", (fx1 - fx0 - 0.014, d1 - d0 - 0.014, 0.02), loc=((fx0 + fx1) / 2, (d0 + d1) / 2, z_fl - 0.01 + hr_.jitter(0, 0.002)))
                into_hall(fl, stone_step, 0.3 + hr_.u(0, 0.2))
        dw_, dh_ = min(0.7, 1.1 * a), min(1.86, zs - z_fl - 0.02)
        dx_ = -s * (hwid - dw_ / 2 - 0.12)
        for k in range(5):
            bw = dw_ / 5
            o = box("inner_door", (bw - 0.006, 0.03, dh_), loc=(dx_ - dw_ / 2 + bw * (k + 0.5), d_f - 0.03, z_fl + dh_ / 2))
            into_hall(o, w_back, 0.35, "Z")
        for sx in (-1, 1):
            o = box("inner_post", (0.09, 0.07, dh_ + 0.05), loc=(dx_ + sx * (dw_ / 2 + 0.045), d_f - 0.035, z_fl + (dh_ + 0.05) / 2))
            into_hall(o, w_a, 0.3, "Z")
        o = box("inner_head", (dw_ + 0.26, 0.08, 0.1), loc=(dx_, d_f - 0.04, z_fl + dh_ + 0.1))
        into_hall(o, w_a, 0.3, "X")
        # wall sconce on the hinge-side wall (hidden behind the open leaf from the latch side)
        xw, sd, sz_ = -s * hwid, 0.62, z_fl + 1.62
        o = box("sconce_plate", (0.012, 0.1, 0.24), loc=(xw + s * 0.006, sd, sz_ - 0.02))
        into_hall(o, iron, 0.2)
        o = sweep("sconce_arm", [(xw + s * 0.01, sd, sz_ - 0.09), (xw + s * 0.085, sd, sz_ - 0.1), (xw + s * 0.105, sd, sz_ - 0.085), (xw + s * 0.105, sd, sz_ - 0.06)],
                  0.006, sides=5, smooth_path=False)
        into_hall(o, iron, 0.2)
        o = lathe("sconce_pan", [(0.0, -0.012), (0.03, -0.01), (0.04, 0.0), (0.034, 0.0), (0.012, -0.003), (0.011, 0.045), (0.0, 0.045)], segments=8,
                  loc=(xw + s * 0.105, sd, sz_ - 0.06))
        into_hall(o, iron, 0.1)
        flame_v = Vector((xw + s * 0.105, sd, sz_ + 0.012))
        if lit_hall:
            fl = sphere("sconce_flame", 0.011, loc=tuple(flame_v), segments=6, rings=4, scale=(0.8, 0.8, 2.0))
            apply_transform(fl)
            assign(fl, mat("emissive", "#ffb45e", glow=6.0))
            transform_verts(fl, squash)
    if lit_hall:
        if p.ajar >= 12:
            # the hallway lamp: its light fans out through the gap from behind the relief (which it cannot
            # burn out), a weak fill a hand in front of the opening stands in for the lit plaster's own glow
            fq = squash(flame_v)
            light("door_spill", (fq.x, -0.003, fq.z), intensity=p.intensity * 2.2, range=5.0, flicker=0.2, shadow_near=0.2)
            light("hall_glow", (s * a * 0.35, y_open - 0.3, sill_top + 1.05), intensity=p.intensity * 0.4, range=1.8, flicker=0.2, shadow_near=1.7)
        elif has_grille:
            # in the lamp box behind the panes (swung with the leaf): shines out through the mask window and the bars only
            lq = (ca * lamp[0] - sa * lamp[1] + off[0], sa * lamp[0] + ca * lamp[1] + off[1], lamp[2])
            light("grille_spill", lq, intensity=p.intensity, range=3.0, flicker=0.15, shadow_near=0.008)
    if str(p.number).strip():
        if timber:
            ny, nz, nsz = -(D + 0.045), z_under + lh / 2, min(0.085, lh * 0.5)
        elif arched:
            ny, nz, nsz = -(D + 0.028), zs + (rise + math.sqrt((R0 + rt_k) ** 2 - c * c)) / 2 + 0.006, min(0.1, rt_k * 0.52)
        else:
            ny, nz, nsz = -(D + 0.012), zs + lh / 2, 0.095
        num = text("number", str(p.number).strip()[:4], size=nsz, depth=0.005, loc=(0, ny + 0.001, nz - nsz * 0.5), font="serif")
        assign(num, hw)
    socket("lantern", "attach", (-s * (W / 2 + 0.32), 0.0, min(H - 0.25, 2.15)), role="lantern")
    socket("sign", "attach", (0.0, 0.0, H + 0.35), role="sign")
    socket("doormat", "attach", (0.0, -D - 0.55, 0.0), role="prop_top")
