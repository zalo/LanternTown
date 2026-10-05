"""Festoon string lights: a galvanised messenger wire strained between two forged eye fixings (chain +
porcelain strain insulator at the feed end, turnbuckle + insulator at the far end) carries a twisted
cloth-braided twin flex (two ~7 mm cores) that is lashed under it with tie-wire. At every lamp a turned
porcelain connector clamps round wire and flex and drops a short twisted pendant, knotted for strain
relief, to a brass-capped porcelain lampholder with a clear squirrel-cage, pearl or lacquer-coloured
bulb, optionally under a small enamel coolie shade. Lamps lean a few degrees with the slope of the
catenary. Origin = first fixing point, the second fixing point is at (span, 0, 0); the string hangs in
the XZ plane below them. Light sockets (1 per ~1.5 m) sit in the air just under the string between two
lit lamps, so no single bulb carries the source.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, degrees, atan2, asin, hypot
from mathutils import Matrix, Vector

META = {"name": "Festoon string lights", "class": "component", "mount": "hang", "mount_height": 3.0,
        "tri_budget": 18000,           # up to 14 m of twisted flex and 40 lamps in one asset (default ~10k)
        "focus": [0.8, 0.0, -0.3]}
PARAMS = {
    "span": {"default": 4.5, "min": 2.0, "max": 14.0, "desc": "distance between the two fixing points (m); second fixing at (span, 0, 0)"},
    "sag": {"default": 0.42, "min": 0.05, "max": 2.0, "desc": "sag of the string at mid-span (m, limited to 0.3 x span)"},
    "count": {"default": 12, "min": 3, "max": 40, "desc": "number of bulbs along the string"},
    "colours": {"default": "warm", "choices": ["warm", "multicolour", "red_white"], "desc": "bulb colours: all warm, lacquered red / amber / green / blue fairground mix, or red and warm alternating"},
    "bulb_size": {"default": 0.07, "min": 0.035, "max": 0.11, "desc": "bulb diameter (m)"},
    "bulb_shape": {"default": "globe", "choices": ["globe", "pear", "edison"], "desc": "glass shape"},
    "glass": {"default": "clear", "choices": ["clear", "pearl"], "desc": "warm bulbs: clear glass with a visible filament cage, or pearl (frosted)"},
    "drop": {"default": 0.07, "min": 0.0, "max": 0.3, "desc": "length of the pendant flex between the string and each lampholder (m)"},
    "stagger": {"default": 0.0, "min": 0.0, "max": 0.8, "desc": "alternate pendants are shortened by this fraction (cascading lamps)"},
    "shade": {"default": False, "desc": "small enamel coolie shade over every lampholder"},
    "enamel": {"default": "#c6bba4", "desc": "colour of the porcelain holders / enamel shades"},
    "cable": {"default": "#4c3a2a", "desc": "colour of the cloth-braided flex"},
    "ends": {"default": "plate", "choices": ["plate", "ring"], "desc": "fixings: screwed wall plates with a cable gland, or plain forged rings (for posts / hooks)"},
    "dead": {"default": 0.08, "min": 0.0, "max": 0.5, "desc": "fraction of blown or missing bulbs"},
    "intensity": {"default": 12.0, "min": 3.0, "max": 60.0, "desc": "intensity of each light socket (cd); one socket per ~1.5 m of string"},
}
VARIANTS = {
    "fairground": {"span": 7.0, "sag": 0.9, "count": 20, "colours": "multicolour", "bulb_size": 0.055, "bulb_shape": "pear",
                   "drop": 0.02, "cable": "#3a332c", "dead": 0.1, "seed": 4},
    "cafe_shades": {"span": 5.5, "sag": 0.45, "count": 8, "bulb_size": 0.08, "bulb_shape": "pear", "glass": "pearl", "shade": True,
                    "enamel": "#2f5a48", "drop": 0.16, "dead": 0.0, "seed": 11},
    "edison_short": {"span": 3.0, "sag": 0.25, "count": 7, "bulb_size": 0.095, "bulb_shape": "edison", "drop": 0.3, "stagger": 0.6, "ends": "ring",
                     "cable": "#7a5630", "dead": 0.0, "seed": 23},
    "long_red_white": {"span": 11.0, "sag": 1.5, "count": 30, "colours": "red_white", "bulb_size": 0.045, "drop": 0.0, "ends": "ring",
                       "dead": 0.15, "seed": 31},
}

WARM = "#ffbf6a"
BULB_COL = {"red": "#c81008", "amber": "#e07206", "green": "#0a9a2a", "blue": "#1030d0"}      # deep lacquers: the hue has to survive bloom
SCREW = [(0.0062, 0.0), (0.0062, 0.0012), (0.0040, 0.0034), (0.0, 0.0040)]      # round-head brass wood screw
EGG = [(0, 0), (0.009, 0), (0.011, 0.004), (0.011, 0.009), (0.019, 0.015), (0.012, 0.021), (0.011, 0.027),
       (0.012, 0.033), (0.019, 0.039), (0.011, 0.045), (0.011, 0.050), (0.009, 0.054), (0, 0.054)]


def _arc(pts):
    """Arc-length parametrisation of a polyline -> (length, at(s) -> Vector, tan(s) -> Vector)."""
    P = np.asarray(pts, dtype=np.float64)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    S = np.concatenate([[0.0], np.cumsum(d)])

    def at(s):
        s = float(min(max(s, 0.0), S[-1]))
        i = int(min(max(np.searchsorted(S, s, side="right") - 1, 0), len(P) - 2))
        t = (s - S[i]) / max(d[i], 1e-9)
        return Vector(P[i] + (P[i + 1] - P[i]) * t)

    def tan(s):
        v = at(s + 0.02) - at(s - 0.02)
        return v.normalized()

    return float(S[-1]), at, tan


def _bk(o):
    apply_transform(o)
    return o


def _lx(name, prof, x0=0.0, seg=10, sm=50.0):
    """Lathe whose axis runs along +X starting at x0 (profile z = distance along x)."""
    return _bk(lathe(name, prof, segments=seg, loc=(x0, 0, 0), rot=(0, 90, 0), smooth_angle=sm))


def _ring(name, x, major, minor, xz=True, seg=6):
    return _bk(torus(name, major, minor, loc=(x, 0, 0), rot=(90, 0, 0) if xz else None, major_seg=seg, minor_seg=3))


def _insulator(x0, flip, ironl, porcl):
    """Porcelain reel insulator with a wire eye each end, along +X from x0. Returns the x of its far end."""
    ironl.append(_ring("ins_eye_a", x0 + 0.005, 0.007, 0.0019, xz=not flip))
    porcl.append(_lx("insulator", EGG, x0 + 0.010, seg=8))
    ironl.append(_ring("ins_eye_b", x0 + 0.069, 0.007, 0.0019, xz=flip))
    x1 = x0 + 0.076
    ironl.append(_ring("wire_eye", x1 + 0.005, 0.007, 0.0022, xz=not flip))
    return x1


def _glass_profile(shape, D, hs, lod=0):
    """Lathe profile of the bulb glass in holder space (holder top = z 0). Returns (profile, cage(zc, r, h), z_bottom)."""
    R = D / 2.0
    rn = min(0.0135 * hs, R * 0.8)
    zn = -0.052 * hs
    pts = [(rn * 0.96, zn + 0.008 * hs), (rn, zn)]
    if shape == "globe":
        th0 = asin(min(0.95, rn * 1.1 / R))
        zc = zn - 0.006 * hs - R * cos(th0)
        n = (6, 6, 5)[lod]
        for k in range(n + 1):
            th = th0 + (pi - th0) * k / n
            pts.append((R * sin(th) if k < n else 0.0, zc + R * cos(th)))
        return pts, (zc - 0.05 * R, 0.30 * R, 0.8 * R), zc - R
    if shape == "pear":
        Lg = 1.5 * D
        zm = zn - (Lg - R)
        na, nb = ((4, 4), (4, 3), (3, 3))[lod]
        for k in range(1, na + 1):
            t = k / float(na)
            pts.append((rn + (R - rn) * t * t * (3 - 2 * t), zn + (zm - zn) * t))
        for k in range(1, nb + 1):
            th = pi / 2 * k / nb
            pts.append((R * cos(th) if k < nb else 0.0, zm - R * sin(th)))
        return pts, (zm + 0.05 * R, 0.30 * R, 0.95 * R), zm - R
    Lg = 1.95 * D                                  # edison / ST teardrop: long neck, widest low down, pip at the tip
    z1, z3 = zn - 0.60 * Lg, zn - Lg
    na, nb = ((6, 4), (5, 3), (4, 3))[lod]
    for k in range(1, na + 1):
        t = k / float(na)
        e = (t * t * (3 - 2 * t)) ** 1.25
        pts.append((rn + (R - rn) * e, zn + (z1 - zn) * t))
    for k in range(1, nb + 1):
        th = radians(84.0 * k / nb)
        pts.append((R * cos(th), z1 + (z3 - z1) * sin(th)))
    pts += [(0.0030, z3 - 0.0008), (0.0013, z3 - 0.0048), (0.0, z3 - 0.0058)] if lod < 2 else [(0.0018, z3 - 0.0040), (0.0, z3 - 0.0058)]
    return pts, (zn - 0.56 * Lg, 0.27 * R, 1.35 * R), z3 - 0.0058


def build(p, rng):
    span = float(p.span)
    sag = min(float(p.sag), span * 0.3)
    D = float(p.bulb_size)
    iron = mat("iron_wrought", rough=1.35, wear=0.3)       # also the messenger + tie-wire: dull, no bright worn edges to glint
    flexm = mat("rope", p.cable, rough=1.8, wear=0.0)      # braided cloth covering: matt, no pale worn edges
    brass = mat("brass")
    porc = mat("ceramic_glazed", p.enamel)
    insul = mat("ceramic_glazed", "#5c3d28", rough=1.6)    # brown salt-glazed strain insulators / cable gland

    # ------------------------------------------------------------------ fixings (wall frame, not tilted)
    plate = p.ends == "plate"
    ex = 0.034 if plate else 0.014
    fix_iron, fix_porc, fix_brass, plates = [], [], [], []
    for end in (0, 1):
        o_i, o_p, o_b = [], [], []
        if plate:
            if end == 0:        # feed end: back plate with the eye bolt above and a porcelain cable gland below
                pl = _bk(rounded_box("plate", (0.172, 0.064, 0.006), 0.014, loc=(0.003, 0, -0.046), rot=(0, -90, 0), seg=3, bevel=0.0012))
                o_i.append(pl)
                plates.append(pl)
                for sy in (-1, 1):
                    for z in (0.024, -0.116):
                        o_b.append(_bk(place(_lx("plate_screw", SCREW, 0.006, seg=6), loc=(0, sy * 0.019, z))))
                g = _lx("gland", [(0.0165, 0), (0.0165, 0.005), (0.012, 0.008), (0.0105, 0.019), (0.0135, 0.021), (0.0135, 0.028),
                                  (0.0080, 0.031), (0.0, 0.031)], 0.006, seg=10)
                place(g, loc=(0, 0, -0.092))
                o_p.append(_bk(g))
                nut = _bk(cylinder("gland_nut", 0.0195, 0.006, loc=(0.009, 0, -0.092), rot=(0, 90, 0), verts=6))
                o_b.append(nut)
            else:               # far end: round rose
                pl = _lx("rose", [(0, 0), (0.036, 0), (0.036, 0.0036), (0.032, 0.0058), (0.017, 0.0066), (0.0, 0.0066)], 0.0, seg=14)
                o_i.append(pl)
                plates.append(pl)
                for k in range(3):
                    a = radians(90 + k * 120)
                    o_b.append(_bk(place(_lx("rose_screw", SCREW, 0.0058, seg=6), loc=(0, 0.026 * cos(a), 0.026 * sin(a)))))
            # forged eye bolt on a stand-off boss
            o_i.append(_lx("eye_boss", [(0.0135, 0), (0.0135, 0.003), (0.0085, 0.007), (0.0052, 0.011), (0.0052, 0.0245)], 0.005, seg=8))
            o_i.append(_ring("eye", ex, 0.0095, 0.0033, xz=True, seg=10))
        else:                   # plain forged ring that goes over a hook / through a staple
            o_i.append(_ring("fix_ring", 0.014, 0.014, 0.0036, xz=True, seg=14))
        if end == 1:
            for o in o_i + o_p + o_b:
                place(o, loc=(span, 0, 0), rot=(0, 0, 180))
                _bk(o)
        fix_iron += o_i
        fix_porc += o_p
        fix_brass += o_b
    exr = ex if plate else 0.028
    EA, EB = Vector((exr, 0, 0)), Vector((span - exr, 0, 0))

    # ------------------------------------------------------------------ the line the string follows
    L, at, tan = _arc(catenary(EA, EB, sag=sag, n=200))

    # ---- strain hardware, built along +X from the eye and swung down into the line of the wire
    a_i, a_p, a_b = [], [], []
    if plate:
        a_i.append(_bk(chain("chain", [(0.006, 0, 0), (0.032, 0, 0), (0.058, 0, 0)], link_len=0.024, wire=0.0028, sides=3)))
        xa = _insulator(0.064, False, a_i, a_p)
    else:
        a_i.append(_bk(chain("chain", [(0.006, 0, 0), (0.024, 0, 0), (0.041, 0, 0)], link_len=0.024, wire=0.0028, sides=3)))
        xa = _insulator(0.047, True, a_i, a_p)
    a_b.append(_bk(cylinder("ferrule", 0.0046, 0.02, loc=(xa + 0.022, 0, 0), rot=(0, 90, 0), verts=6)))
    hA = xa + 0.024
    b_i, b_p, b_b = [], [], []
    b_i.append(_ring("tb_eye_a", 0.009, 0.007, 0.0023, xz=False))
    b_i.append(_bk(cylinder("tb_rod_a", 0.0027, 0.026, loc=(0.029, 0, 0), rot=(0, 90, 0), verts=5, cap=False)))
    b_b.append(_lx("tb_barrel", [(0, 0), (0.0036, 0), (0.0064, 0.002), (0.0082, 0.008), (0.0068, 0.012), (0.0068, 0.048), (0.0082, 0.052),
                                 (0.0064, 0.058), (0.0036, 0.060), (0, 0.060)], 0.036, seg=6, sm=25))
    b_i.append(_bk(cylinder("tb_rod_b", 0.0027, 0.024, loc=(0.102, 0, 0), rot=(0, 90, 0), verts=5, cap=False)))
    b_i.append(_ring("tb_eye_b", 0.121, 0.007, 0.0023, xz=True))
    xb = _insulator(0.124, True, b_i, b_p)
    b_b.append(_bk(cylinder("ferrule", 0.0046, 0.02, loc=(xb + 0.022, 0, 0), rot=(0, 90, 0), verts=6)))
    hB = xb + 0.024
    da = at(hA) - EA
    db = at(L - hB) - EB
    phi_a = degrees(atan2(-da.z, da.x))
    phi_b = degrees(atan2(-db.z, -db.x))
    for o in a_i + a_p + a_b:
        place(o, loc=EA, rot=(0, phi_a, 0))
        _bk(o)
    for o in b_i + b_p + b_b:
        place(o, loc=EB, rot=(0, phi_b, 180))
        _bk(o)
    hw_iron, hw_porc, hw_brass = a_i + b_i, a_p + b_p, a_b + b_b

    # ---- messenger wire
    r_m = 0.0023
    nm = int(min(110, max(20, L / 0.12)))
    wire = sweep("messenger", [tuple(at(hA - 0.008 + (L - hA - hB + 0.016) * k / nm)) for k in range(nm + 1)], r_m, sides=4, smooth_path=False)

    # ------------------------------------------------------------------ lamp stations and ties
    sf0, sf1 = hA + 0.04, L - hB - 0.04
    hs = 0.72 + 0.28 * smoothstep(0.035, 0.06, D) + 0.15 * smoothstep(0.07, 0.11, D)       # lampholder scale (E14 .. E27 .. E40-ish)
    shade_r = max(0.075, 1.1 * D + 0.02) if p.shade else 0.0
    pitch_min = max(D, 2 * shade_r, 0.05) * 1.25
    count = int(max(1, min(int(p.count), (sf1 - sf0 - 0.08) // pitch_min)))
    sp = (sf1 - sf0 - 0.08) / count
    rs = rng.fork("stations")
    stations = [sf0 + 0.04 + (i + 0.5 + rs.jitter(0.0, min(0.12, max(0.0, 0.5 - pitch_min / sp / 2)))) * sp for i in range(count)]
    ties = [sf0 + 0.02] + [0.5 * (stations[i] + stations[i + 1]) for i in range(count - 1)] + [sf1 - 0.035]
    knots = sorted(ties + stations)
    rk = rng.fork("slack")
    slack = []
    for i in range(len(knots) - 1):
        d = knots[i + 1] - knots[i]
        slack.append(rk.u(0.001, 0.003 + 0.035 * d) * (2.4 if rk.chance(0.12) else 1.0))
    knots_np = np.asarray(knots)

    # ---- twisted twin flex lashed under the messenger, with a drip loop from the gland at the feed end
    lod = 0 if count <= 14 else (1 if count <= 24 else 2)      # lamp detail steps down on long, busy strings
    r_f, a_f = 0.0039, 0.0033                                  # two ~7.5 mm cloth-braided cores
    env = a_f + r_f                                            # envelope radius of the twisted pair
    off0 = r_m + 0.88 * env
    zc_b = 0.5 * (off0 + env - r_m)                            # centre of the wire + flex bundle below the wire axis
    nst = int(min(230, max(36, (sf1 - sf0) / 0.040)))
    fstep = (sf1 - sf0) / nst

    def flex_pt(s):
        i = int(min(max(np.searchsorted(knots_np, s, side="right") - 1, 0), len(knots) - 2))
        f = min(max((s - knots[i]) / (knots[i + 1] - knots[i]), 0.0), 1.0)
        q = at(s)
        dz = off0 + slack[i] * sin(pi * f) ** 2
        return Vector((q.x, q.y + 0.35 * slack[i] * sin(2 * pi * f), q.z - dz))

    main = [flex_pt(sf0 + (sf1 - sf0) * k / nst) for k in range(nst + 1)]
    S0, T0 = main[0], tan(sf0)
    if plate:
        G0 = Vector((0.030, 0, -0.092))
        c1, c2 = G0 + Vector((0.085, 0, -0.016)), S0 - T0 * 0.09 + Vector((0, 0, -0.085))
    else:
        G0 = Vector((0.004, 0, -0.03))
        c1, c2 = G0 + Vector((0.06, 0, -0.05)), S0 - T0 * 0.07 + Vector((0, 0, -0.04))
    rough = bezier_path(G0, c1, c2, S0, n=24)
    nlp = max(5, int(path_length(rough) / fstep))
    loop = [Vector(q) for q in resample_path(rough, n=nlp + 1, smooth_path=False)]
    # two round cores laid up as a pair: each is its own tube on a helix, so the twist reads as two cords, not a ribbon
    cpath = loop[:-1] + main
    ncp = len(cpath)
    dphi = radians(84.0)
    strands = ([], [])
    for k, c in enumerate(cpath):
        T = (cpath[min(k + 1, ncp - 1)] - cpath[max(k - 1, 0)]).normalized()
        U = T.cross(Vector((0, 1, 0))).normalized()
        S = U.cross(T)
        o = (S * cos(k * dphi) + U * sin(k * dphi)) * (a_f * min(1.0, 0.35 + 0.65 * min(k, ncp - 1 - k) / 2.0))
        strands[0].append(tuple(c + o))
        strands[1].append(tuple(c - o))
    flex = join([sweep("flex", st, r_f, sides=4, smooth_path=False, cap=True) for st in strands], "flex")
    # taped end of the flex at the far end
    Te = tan(sf1)
    endcap = cylinder("flex_end", env + 0.0012, 0.026, verts=6)
    place(endcap, loc=main[-1] + Te * 0.008, rot=(0, 90 + degrees(atan2(-Te.z, Te.x)), 0))

    # ---- tie-wire lashings (a few turns round wire + flex) between the lamps
    hh, hw = 0.5 * (off0 + env + r_m) * 0.94, env * 0.93
    tie = _lx("tie", [(0.84, -0.0058), (1.07, -0.0036), (1.07, 0.0036), (0.84, 0.0058)], 0.0, seg=6 if lod < 2 else 5, sm=30)
    transform_verts(tie, lambda co: Vector((co.x, co.y * hw, co.z * hh)))
    rt = rng.fork("ties")
    tm = []
    for s in ties:
        q, t = at(s), tan(s)
        th = atan2(-t.z, t.x)
        tm.append(trs((q.x + zc_b * sin(th) * -1.0, 0, q.z - zc_b * cos(th)), (0, degrees(th) + rt.jitter(0, 7), rt.jitter(0, 9)), rt.u(0.97, 1.06)))
    ties_o = instances_to_mesh("ties", tie, tm, keep_base=False)

    # ------------------------------------------------------------------ lamps
    # turned porcelain connector clamped round wire + flex, with the pendant nipple underneath
    rc = max(0.0165, 0.5 * (off0 + env + r_m) + 0.0045)
    if lod == 0:
        con_prof = [(0.0, -0.023), (0.0095, -0.023), (rc, -0.015), (rc - 0.0022, 0.0), (rc, 0.015), (0.0095, 0.023), (0.0, 0.023)]
    else:
        con_prof = [(0.0, -0.023), (0.0095, -0.023), (rc, -0.014), (rc, 0.014), (0.0095, 0.023), (0.0, 0.023)]
    con = _lx("connector", con_prof, 0.0, seg=6 if lod < 2 else 5)
    place(con, loc=(0, 0, -zc_b))
    _bk(con)
    zj = zc_b + rc + 0.013                            # bottom of the pendant nipple below the wire axis
    nip = lathe("nipple", [(0.0058, -zj), (0.0084, -zj + 0.017)], segments=6 if lod < 2 else 5)
    con = join([con, nip], "connector")
    assign(con, porc)

    # lampholder: brass cord-grip cap over a porcelain skirt
    seg_g = (10, 8, 8)[lod]
    seg_h = (7, 7, 6)[lod]
    cap_prof = [(0.0046, 0.0), (0.0072, -0.006), (0.0098, -0.009), (0.0172, -0.012), (0.0187, -0.0195)]
    if lod == 2:
        cap_prof = [(0.0046, 0.0), (0.0090, -0.008), (0.0178, -0.0125), (0.0187, -0.0195)]
    cap = lathe("holder_cap", cap_prof, segments=seg_h)
    disc = lathe("holder_contact", [(0.0188, -0.047), (0.0, -0.047)], segments=seg_h)
    capb = join([cap, disc], "holder_brass")
    body_prof = [(0.0186, -0.0195), (0.0196, -0.032), (0.0216, -0.036), (0.0198, -0.041), (0.0228, -0.057), (0.0190, -0.047)]
    body = lathe("holder_body", body_prof, segments=seg_h)
    for o in (capb, body):
        transform_verts(o, lambda co: co * hs)
    assign(capb, brass)
    assign(body, porc)
    # strain-relief knot of the pendant flex sitting on the cord grip
    knot = sphere("knot", 0.0076 * hs, loc=(0, 0, 0.0036 * hs), segments=5, rings=3, scale=(1.0, 0.82, 0.74))
    _bk(knot)
    displace_noise(knot, strength=0.0016, scale=0.006, seed=p.seed + 9)
    assign(knot, flexm)
    smooth(knot, 180)

    gprof, (cz, cr, ch), zbot = _glass_profile(p.bulb_shape, D, hs, lod)
    glass = lathe("bulb", gprof, segments=seg_g, smooth_angle=70)
    # filament: squirrel cage hung on a glass-and-wire stem
    nzz = (4, 4, 3)[lod]
    cage_pts = []
    for k in range(2 * nzz + 1):
        a = 2 * pi * k / (2 * nzz)
        cage_pts.append((cr * cos(a), cr * sin(a), cz + (0.5 if k % 2 == 0 else -0.5) * ch))
    r_fil = max(0.0009, 0.015 * D)
    fil = sweep("filament", cage_pts, r_fil, sides=3, smooth_path=False, cap=False)
    assign(fil, mat("emissive", "#ffc27a", glow=40))
    z_st = cz + 0.5 * ch

    def _stem(name):
        o = cylinder(name, 0.0015 * hs, abs(z_st + 0.047 * hs), loc=(0, 0, 0.5 * (z_st - 0.047 * hs)), verts=4, cap=False, radius2=0.0034 * hs)
        assign(o, brass)
        return o
    filament = join([fil, _stem("stem")], "filament")
    # blown lamp: the cage has burnt through - a few dark runs are left hanging off the stem
    fil_d = sweep("filament_dead", cage_pts[:4] + [(cage_pts[4][0] * 0.4, cage_pts[4][1] * 0.4 - 0.25 * cr, cz + 0.1 * ch)], r_fil * 0.9, sides=3,
                  smooth_path=False, cap=False)
    assign(fil_d, iron)
    filament_dead = join([fil_d, _stem("stem_d")], "filament_dead")

    shade_o = shade_lit = None
    if p.shade:
        hsd = (0.05 + 0.28 * D)
        z0 = -0.006 * hs
        r0s = 0.0092 * hs
        # CLEARANCE RULE (depth fighting): the under-surface glows (below) and every material is double sided, so the inner
        # skin of the shade lies a clear 4.6 mm inside the outer one (it was 1-2 mm: the glow fought the enamel through it)
        pa, pb, pc = (r0s, z0), (r0s + (shade_r - r0s) * 0.45, z0 - hsd * 0.52), (shade_r - 0.002, z0 - hsd * 0.97)

        def _inner(q, e0, e1, t=0.0046):          # q moved t along the inward / downward normal of the outer wall e0 -> e1
            dr_, dz_ = e1[0] - e0[0], e1[1] - e0[1]
            l_ = (dr_ * dr_ + dz_ * dz_) ** 0.5
            return (q[0] + t * dz_ / l_, q[1] - t * dr_ / l_)
        ia, ic = _inner(pa, pa, pb), _inner(pc, pb, pc, 0.0062)     # (deeper at the rim, where the outer skin flares down)
        ib = tuple(0.5 * (u_ + v_) for u_, v_ in zip(_inner(pb, pa, pb), _inner(pb, pb, pc)))
        ia = (max(ia[0], 0.003), ia[1])
        ic = (min(max(ic[0], shade_r - 0.0065), shade_r - 0.0035), ic[1])
        if lod == 2:
            shade_o = lathe("shade", [(r0s, z0 + 0.003), pb, (shade_r, z0 - hsd),
                                      (shade_r + 0.0018, z0 - hsd - 0.003), ic, ib, ia],
                             segments=9, smooth_angle=50)
        else:
            shade_o = lathe("shade", [(r0s, z0 + 0.003), pa, pb,
                                      pc, (shade_r, z0 - hsd), (shade_r + 0.0022, z0 - hsd - 0.0016),
                                      (shade_r + 0.0012, z0 - hsd - 0.0042), ic, ib, ia],
                             segments=(14, 12)[lod], smooth_angle=50)
        realize(shade_o)
        assign(shade_o, porc)
        if p.colours == "warm":
            shade_lit = duplicate(shade_o, "shade_lit")
            # the white-enamelled inside of the shade catches the lamp: a faint warm glow on the under-surface
            assign_faces(shade_lit, mat("emissive", "#ffb866", glow=1.4), lambda c, n: n.z < -0.2 and (c.x * c.x + c.y * c.y) < (shade_r - 0.003) ** 2)

    # which colour / state each lamp has, and how it hangs
    seq = {"warm": ["warm"], "multicolour": ["red", "amber", "green", "blue"], "red_white": ["red", "warm"]}[p.colours]
    rb = rng.fork("bulbs")
    off = int(rb.u(0, len(seq) - 0.001))
    kinds, mats_b, mats_con, mats_knot, pend, bulb_c, bulb_b = [], [], [], [], [], [], []
    r_p = 0.0041
    pprof = [(r_p, 0.0), (0.0, r_p * 0.62), (-r_p, 0.0), (0.0, -r_p * 0.62)]      # thin twin pendant flex: twisted flat pair
    for i, s in enumerate(stations):
        q, t = at(s), tan(s)
        th = atan2(-t.z, t.x)
        Mc = trs((q.x, 0, q.z), (0, degrees(th), 0))
        mats_con.append(Mc)
        J = Mc @ Vector((0, 0, -zj + 0.001))
        d0 = Vector((-sin(th), 0.0, -cos(th)))                # the nipple points square off the line, not plumb
        d = max(0.014, float(p.drop) * rb.vary(1.0, 0.25) * (1.0 - float(p.stagger) * (0.0 if i % 2 else 1.0)))
        # the stiff flex leaves the connector square to the wire and gravity only partly pulls the lamp plumb:
        # lamps lean 3-8 degrees, mostly with the slope of the catenary
        k = 0.5 - 0.25 * min(1.0, d / 0.2)
        lx = -sin(th) * k + radians(rb.jitter(0, 2.5))
        ly = radians(rb.jitter(0, 3.0 + 2.5 * min(1.0, d / 0.2)))
        m = hypot(lx, ly)
        lo_, hi_ = radians(3.0), radians(8.5)
        if m < 1e-4:
            lx, m = lo_, lo_
        sc = lo_ / m if m < lo_ else (hi_ / m if m > hi_ else 1.0)
        a_h = Vector((lx * sc, ly * sc, -1.0)).normalized()
        H = J + (d0 + a_h * 1.6).normalized() * d
        if d < 0.03:
            path = [tuple(J), tuple(H)]
        else:
            kk = Vector((rb.jitter(0, 0.09), rb.jitter(0, 0.11), 0.0)) * d
            path = bezier_path(J, J + d0 * d * 0.36 + kk, H - a_h * d * 0.36 - kk * rb.u(0.2, 0.9), H, n=(3 if lod == 2 else 4) + int(d / 0.09))
        po = sweep("pendant", [tuple(v) for v in path], pprof, smooth_path=False, cap=False,
                   twist_deg=(75.0 * (len(path) - 1)) * (1 if i % 2 else -1) if lod < 2 else 0.0)
        smooth(po, 80)
        pend.append(po)
        zax = -a_h
        xax = Vector((0, 1, 0)).cross(zax).normalized()
        yax = zax.cross(xax)
        R3 = Matrix((xax, yax, zax)).transposed()
        Bh = Matrix.Translation(H) @ R3.to_4x4() @ Matrix.Rotation(rb.u(0, 2 * pi), 4, 'Z')
        mats_b.append(Bh)
        mats_knot.append(Bh @ Matrix.Rotation(radians(rb.jitter(0, 14)), 4, 'X'))
        col = seq[(i + off) % len(seq)]
        state = "lit"
        if rb.chance(float(p.dead)):
            state = "missing" if rb.chance(0.3) else "dead"
        kinds.append((col, state))
        bulb_c.append(Bh @ Vector((0, 0, 0.5 * (cz + zbot))))
        bulb_b.append(Bh @ Vector((0, 0, zbot)))

    cons = instances_to_mesh("connectors", con, mats_con, keep_base=False)
    pendants = join(pend, "pendants")
    assign(pendants, flexm)
    if lod < 2 or count <= 32:
        instances_to_mesh("knots", knot, mats_knot, keep_base=False)
    else:
        delete([knot])
    holders_b = instances_to_mesh("holder_caps", capb, mats_b, keep_base=False)
    holders_p = instances_to_mesh("holder_bodies", body, mats_b, keep_base=False)
    if shade_o is not None:
        on = [mats_b[i] for i, (c, st) in enumerate(kinds) if st == "lit" and shade_lit is not None]
        offm = [mats_b[i] for i, (c, st) in enumerate(kinds) if st != "lit" or shade_lit is None]
        for nm_, base_, mm_ in (("shades", shade_lit, on), ("shades_dark", shade_o, offm)):
            if mm_:
                shades = instances_to_mesh(nm_, base_, mm_)
                paint(shades, grime=mask_mul(facing((0, 0, 1)), noise_mask(0.05, seed=p.seed + 5, lo=0.25, hi=0.75), 0.8),
                      wear=mask_mul(noise_mask(0.02, seed=p.seed + 6, lo=0.55, hi=0.8), 0.6))
        delete([o for o in (shade_o, shade_lit) if o is not None])
    paint(holders_p, grime=0.5)
    paint(holders_p, grime=mask_mul(facing((0, 0, 1)), 0.7))
    paint(cons, grime=0.55)
    paint(cons, grime=mask_mul(facing((0, 0, -1)), 0.95))         # soot and scorched dust above every lamp
    paint(holders_b, special=mask_mul(noise_mask(0.012, seed=p.seed + 7, lo=0.45, hi=0.75), 0.6), grime=0.25)

    pearl = p.glass == "pearl"
    clear = mat("glass_clear", "#ffe6c2", opacity=0.15)           # thin, genuinely transparent envelope, faint amber
    groups = {}
    for i, (col, state) in enumerate(kinds):
        if state == "missing":
            continue
        key = "dead" if state == "dead" else col
        groups.setdefault(key, []).append(i)
    lit_fil = []
    for key, idx in groups.items():
        if key == "dead":
            m = clear
        elif key == "warm":
            # lit clear lamp: the same thin envelope with a faint warm fill from the filament inside
            m = mat("glass_frosted", "#ffc98a", glow=3.6) if pearl else mat("glass_lantern", "#ffc98a", glow=0.9, opacity=0.15)
            if not pearl:
                lit_fil += idx
        else:
            m = mat("glass_frosted", BULB_COL[key], glow=2.5)
        assign(glass, m)
        g = instances_to_mesh("bulbs_" + key, glass, [mats_b[i] for i in idx])
        if key == "dead":
            # smoky blackening inside the glass, densest at the cap and heavier on one side
            C = np.array([list(mats_b[i].translation) for i in idx])
            A = np.array([list(mats_b[i].to_3x3() @ Vector((0, 0, -1))) for i in idx])
            X = np.array([list(mats_b[i].to_3x3() @ Vector((1, 0, 0))) for i in idx])
            zn_, zb_ = 0.052 * hs, -zbot

            def soot(co, nrm, C=C, A=A, X=X):
                j = np.linalg.norm(co[:, None, :] - (C + A * 0.5 * (zn_ + zb_))[None], axis=2).argmin(1)
                rel = co - C[j]
                zl = (rel * A[j]).sum(1)
                sx = (rel * X[j]).sum(1) / (0.5 * D)
                f = np.clip((zl - zn_) / (0.62 * (zb_ - zn_)), 0.0, 1.0)
                f = 1.0 - f * f * (3 - 2 * f)
                return np.clip(f * (0.72 + 0.28 * sx) * 0.95 + 0.10, 0.0, 1.0)
            paint(g, grime=soot)
            instances_to_mesh("filaments_dead", filament_dead, [mats_b[i] for i in idx])
    delete([glass, filament_dead])
    if lit_fil:
        instances_to_mesh("filaments", filament, [mats_b[i] for i in lit_fil], keep_base=False)
    else:
        delete([filament])

    # ------------------------------------------------------------------ materials and ageing of the line
    rust = mask_mul(noise_mask(0.02, seed=p.seed + 1, lo=0.35, hi=0.7), 0.75)
    for o in fix_iron + hw_iron:
        assign(o, iron)
        paint(o, special=rust, wear=mask_mul(noise_mask(0.01, seed=p.seed + 2, lo=0.5, hi=0.8), 0.5))
    for o in plates:                                  # rust creeping in from the edges and weeping down from the screws
        paint(o, special=mask_mul(noise_mask(0.035, seed=p.seed + 12, lo=0.4, hi=0.75), 0.6))
    for o in (wire, ties_o):
        assign(o, iron)
    paint(wire, special=mask_mul(noise_mask(0.15, seed=p.seed + 3, lo=0.4, hi=0.75), 0.55), grime=0.6)
    paint(ties_o, special=0.4, grime=0.55)
    for o in fix_porc + hw_porc:                      # weathered, sooty glaze: must not shine like a lamp
        assign(o, insul)
        paint(o, grime=0.45)
        paint(o, grime=mask_mul(facing((0, 0, 1)), 0.9))
    for o in hw_brass + fix_brass:
        assign(o, brass)
        paint(o, special=0.45, grime=0.3)
    assign(flex, flexm)
    assign(endcap, flexm)
    paint(flex, grime=0.4)
    paint(flex, grime=mask_mul(noise_mask(0.3, seed=p.seed + 8, lo=0.35, hi=0.7), 0.75))

    # ------------------------------------------------------------------ light sockets: one per ~1.5 m
    # Each sits in the air between two lit neighbours, 7 cm under their glass and >= 0.24 m from either holder and
    # from the line, so the pool on the ground comes from the string as a whole and no single bulb, holder or stretch
    # of flex is burnt out by a source sitting on it (the engine's soft halo reads as the shared glow of the pair).
    lit = [i for i, (c, st) in enumerate(kinds) if st == "lit"]
    nl = int(min(6, max(1, round(span / 1.5))))
    used = set()
    warm_l = np.array(hex_rgb(WARM))
    pairs = [(i, i + 1) for i in range(count - 1) if kinds[i][1] == "lit" and kinds[i + 1][1] == "lit"] or [(i, i) for i in lit]
    for j in range(nl):
        if not pairs:
            break
        s_t = sf0 + (j + 0.5) / nl * (sf1 - sf0)
        pr = min(pairs, key=lambda ab: abs(0.5 * (stations[ab[0]] + stations[ab[1]]) - s_t) + (10.0 if ab in used else 0.0))
        if pr in used:
            continue
        used.add(pr)
        col = WARM
        if p.colours != "warm":                      # tint the pool towards the lamps of this stretch
            near = [k for k in lit if abs(stations[k] - s_t) <= (sf1 - sf0) / nl * 0.5 + 1e-6] or list(pr)
            acc = np.zeros(3)
            for k in near:
                c = kinds[k][0]
                cc = np.array(hex_rgb(BULB_COL[c])) if c != "warm" else warm_l
                acc += cc / max(cc.max(), 1e-3)
            acc = 0.6 * warm_l + 0.4 * acc / len(near)
            col = "#%02x%02x%02x" % tuple(int(round(255 * (max(0.0, min(1.0, v)) ** (1 / 2.2)))) for v in acc)
        c = 0.5 * (bulb_b[pr[0]] + bulb_b[pr[1]])
        hz = 0.5 * (mats_b[pr[0]].translation.z + mats_b[pr[1]].translation.z)
        half = 0.5 * abs(mats_b[pr[0]].translation.x - mats_b[pr[1]].translation.x)
        zl = min(0.5 * (bulb_b[pr[0]].z + bulb_b[pr[1]].z) - 0.07, hz - max(0.0, 0.24 ** 2 - half ** 2) ** 0.5)
        light(f"festoon_{j}", (c.x, c.y, zl), color=col, intensity=float(p.intensity), range=7.0, flicker=0.04, shadow_near=0.35)
