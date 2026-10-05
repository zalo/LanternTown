"""Cast-iron finger signpost. A stepped octagonal pedestal with panelled faces is bolted to a dressed
stone pad; a tapering fluted (optionally black-and-white banded) column carries a plain spigot on which
the finger castings are stacked: every finger is a sleeve with set screws, a root flange, a board with a
raised rim and raised lettering on both faces and a scrolled brace underneath. Each finger sits at its
own height and bearing. The post is closed by a turned cap with a gilded ball, pineapple or acorn finial,
or carries a small gas lantern (lantern_head) that lights the fingers.
"""
import bmesh
import numpy as np
from math import sin, cos, radians, pi, atan2, hypot
from mathutils import Matrix, Vector
from tk import *

META = {"name": "Finger signpost", "class": "street_furniture", "mount": "ground", "focus": 0.8,
        "tri_budget": 32000}     # raised lettering on both faces of up to four boards
PARAMS = {
    "height": {"default": 3.4, "min": 3.0, "max": 4.4, "desc": "overall height to the finial tip (m); raised if needed so the lowest finger clears head height"},
    "arms": {"default": 3, "choices": [2, 3, 4], "desc": "number of finger boards"},
    "texts": {"default": "Old Quarter|Hotel|Canal|Gate", "desc": "finger legends, top to bottom, separated by |"},
    "caps": {"default": True, "desc": "set the legends in capitals (as cast)"},
    "bearings": {"default": "auto", "desc": "'auto' (seeded) or comma separated bearings in degrees, 0 = pointing +X"},
    "arm_length": {"default": 1.25, "min": 0.7, "max": 1.45, "desc": "longest allowed finger (m); short legends give shorter fingers"},
    "letter_size": {"default": 0.095, "min": 0.07, "max": 0.13, "desc": "letter height (m); the board is twice as tall"},
    "font": {"default": "sans", "choices": ["sans", "sans_narrow", "serif", "roman"], "desc": "lettering style"},
    "tip": {"default": "point", "choices": ["point", "round", "arrow"], "desc": "shape of the finger end"},
    "finial": {"default": "ball", "choices": ["ball", "pineapple", "acorn", "lantern"], "desc": "finial on top of the post (lantern = small gas lamp lighting the fingers)"},
    "lamp_cd": {"default": 20.0, "min": 8, "max": 70, "desc": "intensity of the lantern finial (cd)"},
    "flutes": {"default": 12, "min": 0, "max": 16, "desc": "flutes on the column (0 = plain)"},
    "banded": {"default": False, "desc": "column painted in alternating bands of paint / board colour"},
    "plinth": {"default": 0.12, "min": 0.0, "max": 0.45, "desc": "height of the dressed stone pad (0 = bolted straight to the paving)"},
    "sag": {"default": 1.0, "min": 0.0, "max": 3.0, "desc": "largest droop of a finger (deg), seeded per finger"},
    "paint": {"default": "#22352c", "desc": "paint colour of post and castings"},
    "board": {"default": "#e6dcc0", "desc": "paint colour of the finger board faces"},
    "letter": {"default": "#16130f", "desc": "paint colour of the raised letters and rims"},
    "gilt": {"default": False, "desc": "letters and rims in gold leaf instead of paint"},
    "accents": {"default": True, "desc": "finial, column band and pedestal rosettes picked out in gold; False = painted over"},
}
VARIANTS = {
    "gilt_pineapple": {"arms": 4, "height": 3.9, "finial": "pineapple", "gilt": True, "board": "#22406e", "paint": "#161a22",
                       "font": "serif", "tip": "round", "texts": "Gate|Canal|Old Quarter|Hotel", "flutes": 16, "seed": 7},
    "banded_arrow": {"arms": 2, "height": 3.1, "banded": True, "flutes": 0, "tip": "arrow", "finial": "acorn", "paint": "#15161a",
                     "board": "#ece8dc", "texts": "Station|Market Sq.", "plinth": 0.0, "accents": False, "font": "sans_narrow", "seed": 13},
    "red_lantern": {"arms": 4, "height": 4.3, "paint": "#5a1414", "board": "#e9dfc2", "letter": "#3a0d0d", "letter_size": 0.115,
                    "arm_length": 1.4, "plinth": 0.3, "finial": "lantern", "font": "roman", "texts": "Clock Tower|Hotel|Cafe|Gate", "seed": 21},
    "green_worn": {"arms": 3, "height": 3.2, "paint": "#27402f", "board": "#27402f", "letter": "#e4d9b6", "tip": "round",
                   "finial": "acorn", "sag": 3.0, "flutes": 8, "accents": False, "texts": "Canal|Clockmaker|Alley", "seed": 34},
}

R_S = 0.036      # spigot radius
R_SL = 0.05      # sleeve outer radius
T_B = 0.012      # board plate thickness


def _inset(pts, d):
    """Inset a CCW polygon by d with mitred corners."""
    out = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = Vector(pts[i - 1]), Vector(pts[i]), Vector(pts[(i + 1) % n])
        d1, d2 = (p1 - p0).normalized(), (p2 - p1).normalized()
        n1, n2 = Vector((-d1.y, d1.x)), Vector((-d2.y, d2.x))
        k = max(0.35, 1.0 + n1.dot(n2))
        q = p1 + (n1 + n2) * (d / k)
        out.append((q.x, q.y))
    return out


def _outline(x0, L, h, kind):
    if kind == "round":
        r = h / 2
        pts = [(x0, -r)] + [(L - r + r * cos(radians(a)), r * sin(radians(a))) for a in range(-90, 91, 15)] + [(x0, r)]
    elif kind == "arrow":
        tl = h * 0.8
        pts = [(x0, -h / 2), (L - tl, -h / 2), (L - tl - 0.02, -h / 2 - 0.04), (L, 0.0), (L - tl - 0.02, h / 2 + 0.04), (L - tl, h / 2), (x0, h / 2)]
    else:
        tl = h * 0.6
        pts = [(x0, -h / 2), (L - tl, -h / 2), (L, 0.0), (L - tl, h / 2), (x0, h / 2)]
    return pts


def _frame(pos, n, t, s):
    """Matrix putting local +Z on n, local X/Y on the diagonals of (t, n x t) - diamond studs."""
    b = n.cross(t)
    x = (t + b).normalized()
    y = n.cross(x)
    return Matrix(((x.x * s, y.x * s, n.x * s, pos.x), (x.y * s, y.y * s, n.y * s, pos.y), (x.z * s, y.z * s, n.z * s, pos.z), (0, 0, 0, 1)))


def build(p, rng):
    iron = mat("iron_painted", p.paint)
    iron_b = mat("iron_painted", p.board)
    gold = mat("metal_plain", "#d4a84b", rough=0.42)
    trim = gold if p.gilt else mat("iron_painted", p.letter)
    accent = gold if p.accents else iron
    pl = p.plinth if p.plinth > 0.02 else 0.0
    H = p.height
    n_arms = int(p.arms)
    ls = p.letter_size
    h = ls * 2.0
    post, gilded = [], []

    # ---------------------------------------------------------------- stone pad
    if pl > 0:
        st = mat("stone_smooth", "#b3a892")
        r = rng.fork("plinth")
        blocks = []
        if pl > 0.24:
            blocks.append(stone_block("pad_low", (0.66, 0.66, pl * 0.45), loc=(0, 0, pl * 0.225), rng=r, chamfer=0.014))
            blocks.append(stone_block("pad_top", (0.52, 0.52, pl * 0.55), loc=(0, 0, pl * 0.725), rng=r, chamfer=0.022))
        else:
            blocks.append(stone_block("pad", (0.54, 0.54, pl), loc=(0, 0, pl / 2), rng=r, chamfer=min(0.02, pl * 0.25)))
        for b in blocks:
            assign(b, st)
            paint(b, grime=height_ramp(0.0, 0.3), special=mask_mul(facing((0, 0, 1)), noise_mask(0.2, seed=p.seed), 0.8))

    # ---------------------------------------------------------------- pedestal
    ped_h = min(0.7, H * 0.18)
    post.append(rounded_box("base_plate", (0.36, 0.36, 0.03), radius=0.035, loc=(0, 0, pl + 0.015), bevel=0.004))
    bt = bolt("hd_bolt", 0.012, 0.01)
    br = rng.fork("bolts")
    post.append(instances_to_mesh("hd_bolts", bt, [trs((sx * 0.135, sy * 0.135, pl + 0.03), (0, 0, br.u(0, 60))) for sx in (-1, 1) for sy in (-1, 1)],
                                  keep_base=False))
    rb = 0.15
    rd = rb - 0.022
    r0, r1 = 0.06, 0.044
    prof = [(0.0, 0.028), (rb + 0.012, 0.028), (rb + 0.012, 0.068), (rb + 0.002, 0.082), (rb - 0.004, 0.1), (rb - 0.016, 0.112), (rd, 0.13),
            (rd, ped_h * 0.74), (rd + 0.01, ped_h * 0.765), (rd + 0.018, ped_h * 0.795), (rd + 0.018, ped_h * 0.825), (rd - 0.006, ped_h * 0.855),
            (rd - 0.022, ped_h * 0.875), (rd - 0.036, ped_h * 0.905), (rd - 0.044, ped_h * 0.945), (r0 + 0.05, ped_h * 0.975),
            (r0 + 0.045, ped_h), (0.0, ped_h)]
    post.append(lathe("pedestal", [(r_, pl + z_) for r_, z_ in prof], segments=8, rot=(0, 0, 22.5), smooth_angle=25))
    apo = rd * cos(radians(22.5))
    fw = 2 * rd * sin(radians(22.5))
    ph0, ph1 = 0.155, ped_h * 0.74 - 0.025
    panel = shape_extrude("panel", rounded_rect(fw * 0.76, ph1 - ph0, 0.012, 1),
                          [rounded_rect(fw * 0.76 - 0.026, ph1 - ph0 - 0.026, 0.007, 1)], depth=0.006, bevel=0.0015)
    ros = lathe("rosette", [(0.0, 0.0), (0.022, 0.0), (0.024, 0.003), (0.017, 0.007), (0.011, 0.006), (0.008, 0.011), (0.0, 0.014)], segments=10)
    pm = []
    for k in range(8):
        ar = radians(k * 45.0)
        pm.append(trs((apo * sin(ar), -apo * cos(ar), pl + (ph0 + ph1) / 2), (90, 0, k * 45.0)))
    post.append(instances_to_mesh("panels", panel, pm, keep_base=False))
    gilded.append(instances_to_mesh("rosettes", ros, pm, keep_base=False))

    # ---------------------------------------------------------------- finger stack layout
    head = None
    if p.finial == "lantern":
        head = use("lantern_head", width=0.27, height=0.34, paint=p.paint, intensity=p.lamp_cd, seed=int(p.seed) + 1)
    fin_h = {"ball": 0.27, "pineapple": 0.41, "acorn": 0.335, "lantern": 0.125 + (head.size.z if head else 0)}[p.finial]
    low_min = H - fin_h - 0.047 - h - (n_arms - 1) * (h + 0.045)   # underside of the lowest sleeve, tightest stacking
    if low_min < 1.95:
        H += 1.95 - low_min
    zf = H - fin_h                                   # top of the spigot / underside of the cap
    top_c = zf - 0.03 - h / 2
    spacing = clamp((top_c - 2.25) / max(n_arms - 1, 1), h + 0.045, h + 0.15)
    drop = spacing - h - 0.04                        # room for the brace under each board
    z_low = top_c - (n_arms - 1) * spacing - h / 2 - drop - 0.012
    z1 = z_low - 0.004                               # top of the column
    z0 = pl + ped_h

    # ---------------------------------------------------------------- column
    za, zb_ = z0 + 0.08, z1 - 0.09

    def rad(z):
        return r0 + (r1 - r0) * clamp((z - za) / max(zb_ - za, 1e-3))

    fz0, fz1 = z0 + 0.13, z1 - 0.14
    nf = int(p.flutes)
    bands = [] if p.banded else [fz0 + (fz1 - fz0) * 0.4]
    nb = max(3, int(round((fz1 - fz0) / 0.2)) | 1)   # odd number of paint bands
    bh = (fz1 - fz0) / nb
    zs = [fz0 + (fz1 - fz0) * i / 18 for i in range(19)] + [fz0 + 0.05, fz1 - 0.05]
    for bz in bands:
        zs += [bz - 0.06, bz - 0.03, bz + 0.03, bz + 0.06]
    if p.banded:
        zs += [fz0 + bh * i for i in range(1, nb)]
    zs.sort()
    sp = [(r0 + 0.034, z0 - 0.002), (r0 + 0.04, z0 + 0.012), (r0 + 0.03, z0 + 0.03), (r0 + 0.012, z0 + 0.042), (r0 + 0.017, z0 + 0.055),
          (r0 + 0.004, z0 + 0.068), (r0, za)]
    last = za
    for z in zs:
        if z - last > 0.011 and zb_ - z > 0.011:
            sp.append((rad(z), z))
            last = z
    sp += [(r1, zb_), (r1 + 0.008, z1 - 0.078), (r1 + 0.02, z1 - 0.06), (r1 + 0.024, z1 - 0.045), (r1 + 0.012, z1 - 0.028),
           (R_SL + 0.007, z1 - 0.014), (R_SL + 0.007, z1 - 0.003), (R_S, z1), (R_S, zf + 0.004)]
    shaft = lathe("shaft", sp, segments=max(32, nf * (4 if nf <= 12 else 3)), smooth_angle=60)

    def flute(co):
        z = co.z
        if nf <= 0 or z < fz0 or z > fz1:
            return co
        e = min(smoothstep(fz0, fz0 + 0.05, z), 1.0 - smoothstep(fz1 - 0.05, fz1, z))
        for bz in bands:
            e = min(e, smoothstep(0.03, 0.06, abs(z - bz)))
        c = 0.5 + 0.5 * cos(atan2(co.y, co.x) * nf)
        r = hypot(co.x, co.y)
        k = (r - 0.0055 * e * (1.0 - c ** 3)) / r if r > 1e-6 else 1.0
        return Vector((co.x * k, co.y * k, co.z))

    transform_verts(shaft, flute)
    assign(shaft, iron)
    if p.banded:
        assign_faces(shaft, iron_b, lambda c, nrm: fz0 < c.z < fz1 and int((c.z - fz0) / bh) % 2 == 1)
    for bz in bands:
        rr = rad(bz)
        post.append(lathe("band", [(rr - 0.002, bz - 0.034), (rr + 0.006, bz - 0.028), (rr + 0.006, bz - 0.02), (rr + 0.016, bz - 0.01), (rr + 0.019, bz),
                                   (rr + 0.016, bz + 0.01), (rr + 0.006, bz + 0.02), (rr + 0.006, bz + 0.028), (rr - 0.002, bz + 0.034)], segments=32))
    # water-leaf collar round the foot of the column
    leaf = surface("leaf", lambda u, v: ((u - 0.5) * 0.052 * (1 - v * v * 0.85), -0.011 * sin(v * pi * 0.8) - 0.006 * v ** 4 - 0.007 * cos((u - 0.5) * pi),
                                         v * 0.095), 4, 6, thickness=0.004)
    realize(leaf)
    post.append(instances_to_mesh("leaves", leaf, [trs(((r0 + 0.004) * sin(radians(k * 45)), -(r0 + 0.004) * cos(radians(k * 45)), z0 + 0.055),
                                                       (0, 0, k * 45), 1.0 + 0.2 * (k % 2)) for k in range(8)], keep_base=False))

    # ---------------------------------------------------------------- fingers
    labels = [s.strip() for s in str(p.texts).split("|") if s.strip()] or ["Town"]
    rb_ = rng.fork("bearings")
    bearings = []
    try:
        given = [float(s) for s in str(p.bearings).split(",")]
    except ValueError:
        given = []
    for i in range(n_arms):
        if i < len(given):
            bearings.append(given[i])
            continue
        side = 0.0 if i % 2 == 0 else 180.0
        b = side + rb_.u(-34, 34)
        for _ in range(8):
            if all(abs(((b - o + 180) % 360) - 180) > 20 for o in bearings):
                break
            b = side + rb_.u(-34, 34)
        bearings.append(b)

    arm_objs = []
    for i in range(n_arms):
        ar = rng.fork(f"arm{i}")
        label = labels[i % len(labels)]
        if p.caps:
            label = label.upper()
        zc = top_c - i * spacing
        zb, zt = -h / 2 - drop - 0.012, h / 2 + 0.02
        irons, plates, trims = [], [], []
        irons.append(lathe("sleeve", [(R_S + 0.0005, zb), (R_SL + 0.005, zb), (R_SL + 0.008, zb + 0.006), (R_SL + 0.005, zb + 0.013), (R_SL, zb + 0.019),
                                      (R_SL, zt - 0.019), (R_SL + 0.005, zt - 0.013), (R_SL + 0.008, zt - 0.006), (R_SL + 0.005, zt), (R_S + 0.0005, zt)],
                           segments=28))
        # set screws on the back of the sleeve, root flange where the board is cast on
        for sz in (-h * 0.28, h * 0.28):
            irons.append(bolt("setscrew", 0.009, 0.009, loc=(-R_SL + 0.001, 0, sz), rot=(0, -90, ar.u(0, 45)), kind="square"))
        irons.append(box("root", (0.02, 0.036, h + 0.014), loc=(R_SL + 0.005, 0, 0), bevel=0.004))
        for sy in (-1, 1):
            for sz in (-h * 0.33, h * 0.33):
                irons.append(lathe("rootbolt", [(0.0068, 0.0), (0.0062, 0.003), (0.0035, 0.0052), (0.0, 0.006)], segments=8,
                                   loc=(R_SL + 0.005, sy * 0.0175, sz), rot=(-90 * sy, 0, 0)))

        # lettering: measure, fit to the board, then size the board to the legend
        tx = text("legend", label, size=ls, depth=0.005, font=p.font)
        apply_transform(tx)
        bm = bmesh.new()                             # thin out the letter outlines; the back caps are buried in the board
        bm.from_mesh(tx.data)
        bmesh.ops.dissolve_limit(bm, angle_limit=radians(5.0), use_dissolve_boundaries=False, verts=bm.verts[:], edges=bm.edges[:])
        bmesh.ops.delete(bm, geom=[f_ for f_ in bm.faces if f_.normal.y > 0.5], context="FACES")
        bmesh.ops.triangulate(bm, faces=bm.faces[:])   # concave letter caps: triangulate here, not by fan
        bm.to_mesh(tx.data)
        bm.free()
        mn, mx = world_bbox([tx])
        tw, th = max(mx.x - mn.x, 1e-3), max(mx.z - mn.z, 1e-3)
        cx, cz = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2
        k = min(ls / th, (h - 0.05) / th) if p.caps else (h - 0.07) / th
        tw *= k
        tip_in = {"point": h * 0.42, "round": h * 0.3, "arrow": h * 0.8 + 0.02}[p.tip]
        xa = R_SL + 0.065
        L = clamp(xa + tw + 0.03 + tip_in, 0.55, p.arm_length)
        xb = L - tip_in - 0.015
        sx = min(1.0, (xb - xa) / tw)
        xc = (xa + xb) / 2
        tb = duplicate(tx, "legend_back")
        yo = T_B / 2 - 0.0008
        transform_verts(tx, lambda co: Vector((xc + (co.x - cx) * k * sx, co.y - yo, (co.z - cz) * k)))
        transform_verts(tb, lambda co: Vector((xc - (co.x - cx) * k * sx, -co.y + yo, (co.z - cz) * k)))
        trims += [tx, tb]

        out = _outline(R_SL - 0.004, L, h, p.tip)
        plates.append(shape_extrude("board", _inset(out, 0.004), depth=T_B, loc=(0, T_B / 2, 0), rot=(90, 0, 0)))
        rim_t = T_B + 0.009
        trims.append(shape_extrude("rim", out, [_inset(out, 0.013)], depth=rim_t, loc=(0, rim_t / 2, 0), rot=(90, 0, 0), bevel=0.0012))
        # scrolled brace under the board
        if drop >= 0.05:
            bar = rounded_rect(0.011, 0.013, 0.003, 1)
            pts = bezier_path((R_SL - 0.003, 0, zb + 0.03), (R_SL + drop * 1.0, 0, zb + 0.03), (R_SL + drop * 1.75, 0, -h / 2 - drop * 0.55),
                              (R_SL + drop * 1.9, 0, -h / 2 + 0.003), n=11)
            irons.append(sweep("brace", pts, bar, up=(0, 1, 0)))
            sc = spiral_path(0.007, drop * 0.36, 1.3, n=22, a0_deg=150, plane="XZ", center=(R_SL + drop * 0.5, 0, -h / 2 - drop * 0.42))
            irons.append(sweep("scroll", sc, bar, up=(0, 1, 0), smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t))
        for o in irons:
            assign(o, iron)
        for o in plates:
            assign(o, iron_b)
        for o in trims:
            assign(o, trim)
        droop = ar.u(0.15, 1.0) * p.sag
        yaw = bearings[i] + ar.u(-1.5, 1.5)
        grp = [join(irons, f"arm{i}_iron"), join(plates, f"arm{i}_board"), join(trims, f"arm{i}_trim")]
        for o in grp:
            place(o, loc=(0, 0, zc), rot=(0, droop, yaw))
        arm_objs.append(grp)
        print(f"DBG arm{i} '{label}' L={L:.2f} z={zc:.2f} bearing={bearings[i]:.0f} squeeze={sx:.2f}")

    # ---------------------------------------------------------------- cap and finial
    neck = [(R_S - 0.001, -0.008), (R_SL + 0.006, -0.002), (R_SL + 0.013, 0.008), (R_SL + 0.011, 0.018), (0.036, 0.032), (0.025, 0.046),
            (0.022, 0.06), (0.031, 0.07), (0.031, 0.078), (0.021, 0.088)]
    fin = []
    if p.finial == "ball":
        R = 0.074
        c = 0.088 + R * 0.95
        pr = neck[5:] + [(R * cos(radians(a)), c + R * sin(radians(a))) for a in range(-70, 81, 10)]
        pr += [(0.013, c + R * 0.992), (0.016, c + R + 0.01), (0.011, c + R + 0.02), (0.0, c + R + 0.024)]
        fin.append(lathe("finial", [(r_, zf + z_) for r_, z_ in pr], segments=32))
        post.append(lathe("cap", [(r_, zf + z_) for r_, z_ in neck[:6]], segments=28))
    elif p.finial == "pineapple":
        a_, b_ = 0.074, 0.112
        c = 0.088 + b_ * 0.96
        post.append(lathe("cap", [(r_, zf + z_) for r_, z_ in neck + [(0.0, 0.09)]], segments=28))
        pr = [(0.02, 0.086)] + [(a_ * cos(radians(a)), c + b_ * sin(radians(a))) for a in range(-72, 81, 12)] + [(0.0, c + b_)]
        fin.append(lathe("pine_body", [(r_ * 0.93, zf + z_) for r_, z_ in pr], segments=20))
        stud = bolt("stud", 0.0225, 0.014, kind="pyramid")
        sm = []
        rows, per = 7, 9
        for j in range(rows):
            e = radians(-56 + 112 * j / (rows - 1))
            rr, zz = a_ * cos(e), zf + c + b_ * sin(e)
            for q in range(per):
                az = 2 * pi * (q + 0.5 * (j % 2)) / per
                nrm = Vector((cos(az) * cos(e) / a_, sin(az) * cos(e) / a_, sin(e) / b_)).normalized()
                tg = Vector((-sin(az), cos(az), 0))
                pos = Vector((rr * cos(az), rr * sin(az), zz)) - nrm * 0.006
                sm.append(_frame(pos, nrm, tg, 0.55 + 0.45 * cos(e)))
        fin.append(instances_to_mesh("pine_studs", stud, sm, keep_base=False))
        lf = surface("crown_leaf", lambda u, v: ((u - 0.5) * 0.04 * (1 - v ** 2 * 0.92), -0.034 * v ** 2.2 - 0.004 * cos((u - 0.5) * pi), v * 0.105),
                     5, 7, thickness=0.003)
        realize(lf)
        lm = [trs((0.014 * sin(radians(q * 60)), -0.014 * cos(radians(q * 60)), zf + c + b_ - 0.012), (0, 0, q * 60), 1.0) for q in range(6)]
        lm += [trs((0.005 * sin(radians(q * 90 + 30)), -0.005 * cos(radians(q * 90 + 30)), zf + c + b_ - 0.008), (12, 0, q * 90 + 30), 0.82) for q in range(4)]
        fin.append(instances_to_mesh("crown", lf, lm, keep_base=True))
        cl = [trs((0.024 * sin(radians(q * 45)), -0.024 * cos(radians(q * 45)), zf + 0.083), (-28, 0, q * 45), 0.62) for q in range(8)]
        fin.append(instances_to_mesh("calyx", lf, cl, keep_base=False))
    elif p.finial == "lantern":
        pr = neck + [(0.034, 0.096), (0.045, 0.106), (0.047, 0.118), (0.04, 0.125), (0.0, 0.125)]
        post.append(lathe("cap", [(r_, zf + z_) for r_, z_ in pr], segments=28))
        head.place((0, 0, zf + 0.125))
    else:
        post.append(lathe("cap", [(r_, zf + z_) for r_, z_ in neck[:6]], segments=28))
        pr = neck[5:] + [(0.034, 0.094), (0.058, 0.108), (0.069, 0.132), (0.071, 0.158), (0.066, 0.172), (0.06, 0.176), (0.058, 0.183), (0.061, 0.2),
                     (0.058, 0.235), (0.047, 0.268), (0.028, 0.296), (0.011, 0.31), (0.007, 0.322), (0.009, 0.328), (0.0, 0.335)]
        fin.append(lathe("finial", [(r_, zf + z_) for r_, z_ in pr], segments=32))
        stud = bolt("stud", 0.0105, 0.006, kind="pyramid")
        sm = []
        for j, (rr, zz) in enumerate([(0.064, 0.121), (0.0705, 0.146), (0.069, 0.166)]):
            for q in range(14):
                az = 2 * pi * (q + 0.5 * (j % 2)) / 14
                nrm = Vector((cos(az), sin(az), -0.35 + 0.3 * j)).normalized()
                sm.append(_frame(Vector((rr * cos(az), rr * sin(az), zf + zz)) - nrm * 0.003, nrm, Vector((-sin(az), cos(az), 0)), 1.0))
        fin.append(instances_to_mesh("cup_studs", stud, sm, keep_base=False))

    # ---------------------------------------------------------------- materials and ageing
    for o in post:
        assign(o, iron)
    for o in gilded:
        assign(o, accent)
    for o in fin:
        assign(o, accent)
    for o in post:
        if o.name.startswith("band"):
            assign(o, accent)
    seed = int(p.seed)
    for o in post:
        if o.name.startswith(("pedestal", "panels", "hd_bolts", "base_plate")):
            paint(o, special=mask_mul(height_ramp(pl, pl + 0.4), noise_mask(0.09, seed=seed + 2, lo=0.3, hi=0.7)), grime=height_ramp(pl, pl + 0.55))
    # hands polish the column at about 1.1 m; rust weeps from under the capital
    paint(shaft, wear=mask_mul(height_ramp(1.2, 1.55), height_ramp(1.2, 0.85), noise_mask(0.05, seed=seed + 4), 0.6),
          special=mask_mul(height_ramp(z1 - 0.03, z1 - 0.35), noise_mask(0.06, seed=seed + 5, lo=0.4, hi=0.75), 0.7))

    def near_post(d0, d1):
        return lambda co, nrm: np.clip((d1 - np.hypot(co[:, 0], co[:, 1])) / (d1 - d0), 0.0, 1.0)

    for i, (a_iron, a_board, a_trim) in enumerate(arm_objs):
        paint(a_iron, special=mask_mul(noise_mask(0.05, seed=seed + 10 + i, lo=0.45, hi=0.8), 0.55), grime=0.15)
        # rain streaks down the face, rust creeping out from the root, chipped paint towards the tip
        paint(a_board, grime=mask_mul(noise_mask(0.09, seed=seed + 20 + i, lo=0.45, hi=0.9), 0.14),
              special=mask_mul(near_post(0.06, 0.2), noise_mask(0.04, seed=seed + 30 + i, lo=0.35, hi=0.75), 0.6),
              wear=mask_mul(noise_mask(0.03, seed=seed + 40 + i, lo=0.62, hi=0.85), 0.22))
        if not p.gilt:
            paint(a_trim, wear=mask_mul(noise_mask(0.03, seed=seed + 50 + i, lo=0.55, hi=0.85), 0.35))
    for o in fin:
        paint(o, grime=mask_mul(facing((0, 0, 1)), 0.45))
    socket("notice", "attach", (0, -(r0 + 0.004), pl + 1.55), role="sign")
