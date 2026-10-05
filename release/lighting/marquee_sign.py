"""Theatre / shop marquee: a joiner-made painted fascia sign fixed flat to the wall. Horizontal boards form the
lettering field (raised cut-out letters, pinstripe line), framed by a mitred moulding whose trough carries a
border of screw-in bulbs in brass holders. A dentilled cornice with lead capping sits on top and carries the
pediment (swan-neck scrolls with urn / bulb-lit sunburst arch / shaped gable with monogram / plain blocking
course); scrolled consoles, a sill moulding, iron wall cleats and the supply conduit finish it off.
Origin: centre of the lettering board on the wall plane (y = 0); `width` x `height` is the framed board.
"""
from tk import *
import numpy as np
from math import asin, acos, sqrt

META = {"name": "Bulb-bordered marquee sign", "class": "component", "mount": "wall", "mount_height": 3.3,
        "focus": 0.45, "tri_budget": 16000}
PARAMS = {
    "text": {"default": "THEATRE", "desc": "sign lettering (raised letters)"},
    "width": {"default": 2.4, "min": 1.2, "max": 4.5, "desc": "width of the framed board (m)"},
    "height": {"default": 0.62, "min": 0.42, "max": 1.1, "desc": "height of the framed board (m)"},
    "board_color": {"default": "#8a1b25", "desc": "paint colour of the lettering field and tympanum"},
    "letter_color": {"default": "#e3b552", "desc": "letter / gilding colour"},
    "trim_color": {"default": "#d8c9a2", "desc": "paint colour of mouldings, bulb trough and cornice"},
    "gilded": {"default": True, "desc": "letters and ornaments in gold leaf (False = brush-painted)"},
    "font": {"default": "serif", "choices": ["serif", "roman", "sans", "sans_narrow", "script", "noto_serif"],
             "desc": "letter face"},
    "bulb_spacing": {"default": 0.14, "min": 0.09, "max": 0.32, "desc": "bulb pitch along the border (m)"},
    "bulb_color": {"default": "#ffd9a0", "desc": "bulb glow colour"},
    "dead_bulbs": {"default": 0.06, "min": 0.0, "max": 0.4, "desc": "fraction of blown bulbs"},
    "pediment": {"default": "scroll", "choices": ["scroll", "arch", "gable", "flat"],
                 "desc": "top pediment: swan-neck scrolls / sunburst arch / shaped gable / blocking course"},
    "pediment_height": {"default": 0.42, "min": 0.22, "max": 0.8, "desc": "pediment rise above the cornice (m)"},
    "monogram": {"default": "T", "desc": "letter(s) in the gable medallion"},
    "wear": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "paint wear / weathering amount"},
    "intensity": {"default": 2.5, "min": 0.5, "max": 20.0, "desc": "intensity of each light socket (cd)"},
}
VARIANTS = {
    "luggage_arch": {"text": "LUGGAGE", "width": 2.0, "height": 0.56, "pediment": "arch", "pediment_height": 0.5,
                     "board_color": "#17403a", "trim_color": "#cfc6a8", "bulb_spacing": 0.12, "seed": 4},
    "grand_gable": {"text": "GRAND HOTEL", "width": 3.6, "height": 0.8, "pediment": "gable", "pediment_height": 0.7,
                    "monogram": "G", "board_color": "#1b2a4a", "trim_color": "#e4dcc6", "letter_color": "#ecc76a",
                    "bulb_spacing": 0.17, "font": "roman", "seed": 9},
    "cafe_flat": {"text": "Café", "width": 1.4, "height": 0.6, "pediment": "flat", "pediment_height": 0.3,
                  "board_color": "#e2d6b4", "trim_color": "#27433a", "letter_color": "#7a1212", "gilded": False,
                  "font": "script", "bulb_spacing": 0.2, "bulb_color": "#ffc57a", "dead_bulbs": 0.0, "wear": 0.15,
                  "seed": 15},
    "worn_bazaar": {"text": "BAZAAR", "width": 2.8, "height": 0.7, "pediment": "scroll", "pediment_height": 0.6,
                    "board_color": "#2d2a26", "trim_color": "#b9a26c", "letter_color": "#d9d2bd", "gilded": False,
                    "font": "sans_narrow", "bulb_spacing": 0.1, "dead_bulbs": 0.28, "wear": 0.95, "seed": 23},
}


# ------------------------------------------------------------------------------------------ local mesh helpers
def mitre_frame(name, w, h, prof, y0=0.0):
    """Mitred rectangular frame (outer size w x h, centred, facing -Y) from an open profile [(inset, depth)]."""
    cs = [(-w / 2, -h / 2, 1, 1), (w / 2, -h / 2, -1, 1), (w / 2, h / 2, -1, -1), (-w / 2, h / 2, 1, -1)]
    out = []
    for k in range(4):
        a, b = cs[k], cs[(k + 1) % 4]
        verts, faces = [], []
        for (s, d) in prof:
            verts.append((a[0] + a[2] * s, y0 - d, a[1] + a[3] * s))
            verts.append((b[0] + b[2] * s, y0 - d, b[1] + b[3] * s))
        for i in range(len(prof) - 1):
            faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
        o = mesh_object(f"{name}{k}", verts, faces)
        uv_grain(o, axis='X' if k % 2 == 0 else 'Z')
        smooth(o, 35)
        out.append(o)
    return out


def cornice_u(name, w, d0, prof, x0=0.0, z0=0.0, flip=False):
    """Moulding returned round three sides of a block standing against the wall. prof = [(projection, z)]
    bottom-to-top (top-to-bottom with flip=True). Capped top and bottom."""
    verts, faces = [], []
    for (pp, z) in prof:
        xl, xr, yf = x0 - w / 2 - pp, x0 + w / 2 + pp, -(d0 + pp)
        verts += [(xl, 0.0, z0 + z), (xl, yf, z0 + z), (xr, yf, z0 + z), (xr, 0.0, z0 + z)]
    n = len(prof)
    for i in range(n - 1):
        for k in range(3):
            faces.append((4 * i + k, 4 * i + k + 1, 4 * (i + 1) + k + 1, 4 * (i + 1) + k))
    e = 4 * (n - 1)
    faces += [(e, e + 1, e + 2, e + 3), (3, 2, 1, 0)]
    if flip:
        faces = [tuple(reversed(f)) for f in faces]
    o = mesh_object(name, verts, faces)
    uv_grain(o, axis='X')
    smooth(o, 32)
    return o


def prism_x(name, sect, x0, x1):
    """Solid of constant section [(depth, z)] between x0 and x1."""
    n = len(sect)
    verts = [(x0, -d, z) for d, z in sect] + [(x1, -d, z) for d, z in sect]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, 40)
    return o


def edge_mould(mw, tt, proud=0.03):
    """Symmetric capping moulding section (a across, b towards the viewer) that also covers the board edge."""
    h = mw / 2
    return [(h, -tt), (h, proud * 0.6), (h * 0.62, proud), (h * 0.3, proud * 0.7),
            (-h * 0.3, proud * 0.7), (-h * 0.62, proud), (-h, proud * 0.6), (-h, -tt)]


def finial(name, r, loc, kind="ball"):
    if kind == "ball":
        prof = [(r * 0.6, 0.0), (r * 0.6, r * 0.2), (r * 0.3, r * 0.5), (r * 0.82, r * 1.0), (r, r * 1.7),
                (r * 0.8, r * 2.35), (r * 0.2, r * 2.72), (r * 0.16, r * 3.0), (0.0, r * 3.2)]
    else:   # urn with flame
        prof = [(r * 0.62, 0.0), (r * 0.66, r * 0.25), (r * 0.3, r * 0.5), (r * 0.26, r * 0.9), (r * 0.5, r * 1.1), (r * 0.9, r * 1.6),
                (r, r * 2.3), (r * 0.92, r * 2.9), (r * 0.62, r * 3.3), (r * 0.7, r * 3.5), (r * 0.74, r * 3.7), (r * 0.5, r * 3.9),
                (r * 0.34, r * 4.3), (r * 0.42, r * 4.7), (r * 0.3, r * 5.2), (r * 0.1, r * 5.7), (0.0, r * 6.0)]
    return lathe(name, prof, segments=10 if kind == "ball" else 12, loc=loc)


def build(p, rng):
    W, H = p.width, p.height
    seed = int(p.seed)
    c_board = mat("wood_painted", p.board_color, wear=1.0 + p.wear)
    c_trim = mat("wood_painted", p.trim_color, wear=1.0 + p.wear)
    c_gold = mat("metal_plain", p.letter_color, rough=0.6) if p.gilded else mat("wood_painted", p.letter_color)
    c_brass = mat("brass", rough=0.9)
    c_iron = mat("iron_wrought")
    c_lead = mat("lead")
    c_bulb = mat("emissive", p.bulb_color, glow=5.0, sign=1)     # (sign=1: a sign's own light, drawn brighter by the web bake)
    c_dead = mat("glass_frosted", "#9c8f7a", glow=0.0, opacity=0.8)

    board, trim, gold, iron = [], [], [], []

    # ---------------------------------------------------------------- frame: outer moulding + bulb trough + bead
    bw = clamp(0.19 * H, 0.105, 0.15)             # border width, outer edge to lettering field
    bs = clamp(bw / 0.118, 0.9, 1.25)             # bulb scale
    D, dt, yf = 0.101, 0.070, 0.045               # frame depth, trough floor depth, field depth
    xi = bw - 0.028
    trim += mitre_frame("frame", W, H, [(0.0, 0.0), (0.0, 0.088), (0.004, 0.097), (0.010, D), (0.016, 0.097), (0.020, 0.090),
                                        (0.025, 0.090), (0.029, 0.078), (0.031, dt), (xi, dt)])
    gold += mitre_frame("bead", W, H, [(xi, dt), (xi + 0.002, 0.080), (xi + 0.008, 0.087), (xi + 0.015, 0.087),
                                       (xi + 0.021, 0.080), (xi + 0.024, 0.066), (bw, 0.060), (bw, yf - 0.002)])

    # ---------------------------------------------------------------- lettering field: horizontal boards
    Wi, Hi = W - 2 * bw, H - 2 * bw
    nb = max(2, int(round(Hi / 0.13)))
    bh = Hi / nb
    rb_ = rng.fork("boards")
    for i in range(nb):
        z = -Hi / 2 + bh * (i + 0.5)
        b = plank(f"field{i}", length=Wi + 0.012, width=bh - 0.0018, thick=0.02, loc=(0, -(yf - 0.01) + rb_.jitter(0, 0.0008), z),
                  rot=(90, 0, 0), bevel=0.0022, warp=0.0012, rng=rb_, axis='X')
        densify(b, 0.3)
        board.append(b)
    # signwriter's pinstripe line round the field
    ps = 0.028
    gold += mitre_frame("pinstripe", Wi - 2 * ps, Hi - 2 * ps, [(0.0, 0.0), (0.0, 0.0012), (0.006, 0.0012), (0.006, 0.0)], y0=-yf)

    # ---------------------------------------------------------------- raised letters, fitted to the field
    body = str(p.text).strip() or " "
    fit_w, fit_h = Wi - 2 * ps - 0.09, (Hi - 2 * ps) * 0.74
    s0 = 0.2
    t = text("probe", body, size=s0, depth=0.012, font=p.font)
    co = np.array([v.co[:] for v in t.data.vertices]) if len(t.data.vertices) else np.zeros((1, 3))
    tw, th = max(co[:, 0].max() - co[:, 0].min(), 1e-3), max(co[:, 1].max() - co[:, 1].min(), 1e-3)
    delete([t])
    k = min(fit_w / tw, fit_h / th)
    k = min(k, 0.34 / th)                                        # never taller than ~0.34 m
    cx, cz = (co[:, 0].max() + co[:, 0].min()) / 2 * k, (co[:, 1].max() + co[:, 1].min()) / 2 * k
    letters = text("letters", body, size=s0 * k, depth=0.014, loc=(-cx, -yf, -cz), font=p.font)
    gold.append(letters)
    print(f"DBG letters cap={th * k:.3f} m wide={tw * k:.3f} m")

    # ---------------------------------------------------------------- cornice with dentils and lead capping
    d0 = 0.09
    zc = H / 2
    cprof = [(0.014, 0.0), (0.014, 0.010), (0.006, 0.014), (0.006, 0.040), (0.030, 0.043), (0.032, 0.050), (0.040, 0.056),
             (0.047, 0.064), (0.050, 0.066), (0.050, 0.078), (0.055, 0.081), (0.063, 0.090), (0.069, 0.097), (0.069, 0.106)]
    ch, cp = cprof[-1][1], cprof[-1][0]
    cor = cornice_u("cornice", W, d0, cprof, z0=zc)
    densify(cor, 0.4)
    trim.append(cor)
    dent = box("dentil", (0.024, 0.02, 0.022))
    nd = max(4, int(round((W + 0.02) / 0.052)))
    dm = [trs((-W / 2 - 0.0 + (W) * (i + 0.5) / nd, -(d0 + 0.006 + 0.0095), zc + 0.0275)) for i in range(nd)]
    for sx in (-1, 1):
        dm += [trs((sx * (W / 2 + 0.006 + 0.0095), -(0.025 + j * 0.05), zc + 0.0275), (0, 0, 90)) for j in range(2)]
    trim.append(instances_to_mesh("dentils", dent, dm, keep_base=False))
    cap = box("lead_cap", (W + 2 * cp + 0.012, d0 + cp + 0.008, 0.005), loc=(0, -(d0 + cp + 0.008) / 2, zc + ch + 0.0025))
    densify(cap, 0.2)
    displace_noise(cap, 0.0012, 0.12, seed=seed, axis_mask=(0, 0, 1))
    assign(cap, c_lead)
    zp0 = zc + ch + 0.005                                          # pediment springs from here

    # ---------------------------------------------------------------- sill moulding and scrolled consoles
    sill = cornice_u("sill", W, d0, [(0.012, 0.0), (0.016, -0.006), (0.024, -0.012), (0.024, -0.022), (0.014, -0.030),
                                     (0.005, -0.036), (0.0, -0.046)], z0=-H / 2, flip=True)
    trim.append(sill)
    zs0 = -H / 2 - 0.046
    cs_h = clamp(0.3 * H, 0.15, 0.26)
    u = cs_h / 0.2
    csect = [(0.0, 0.0), (0.100, 0.0), (0.110, -0.012 * u), (0.112, -0.03 * u), (0.104, -0.047 * u), (0.086, -0.06 * u),
             (0.068, -0.074 * u), (0.056, -0.094 * u), (0.052, -0.118 * u), (0.055, -0.142 * u), (0.052, -0.164 * u),
             (0.040, -0.184 * u), (0.022, -0.196 * u), (0.0, -0.2 * u)]
    for sx in (-1, 1):
        xc_ = sx * (W / 2 - 0.085)
        con = prism_x("console", [(d, zs0 + z) for d, z in csect], xc_ - 0.036, xc_ + 0.036)
        bevel(con, 0.003, segments=1, angle=60)
        uv_grain(con, axis='Z')
        trim.append(con)
        # raised gilt drop on the console face
        gold.append(lathe("console_boss", [(0.0, 0.0), (0.017, 0.0), (0.019, 0.003), (0.012, 0.008), (0.0, 0.011)], segments=10,
                          loc=(xc_, -0.0535, zs0 - 0.118 * u), rot=(90, 0, 0)))

    # ---------------------------------------------------------------- iron wall cleats and the supply conduit
    bolt_m = []
    for sx in (-1, 1):
        for zc_ in (H * 0.27, -H * 0.27):
            iron.append(box("cleat", (0.062, 0.006, 0.1), loc=(sx * (W / 2 + 0.025), -0.003, zc_), bevel=0.0015, segments=1))
            iron.append(box("cleat_ret", (0.006, 0.05, 0.1), loc=(sx * (W / 2 + 0.003), -0.031, zc_), bevel=0.0015, segments=1))
            for dz in (-0.032, 0.032):
                bolt_m.append(trs((sx * (W / 2 + 0.038), -0.006, zc_ + dz), (90, 0, rng.u(0, 60))))
    bb = cylinder("cleat_bolt", 0.0085, 0.006, loc=(0, 0, 0.003), verts=6)
    iron.append(instances_to_mesh("cleat_bolts", bb, bolt_m, keep_base=False))
    xq = W / 2 - 0.085 - 0.036 - min(0.22, W * 0.09)
    zq = zs0 - 0.085
    iron.append(sweep("conduit", [(xq, -0.045, zs0 + 0.01), (xq, -0.045, zq + 0.03), (xq, -0.041, zq + 0.014), (xq, -0.031, zq + 0.004),
                                  (xq, -0.018, zq), (xq, 0.0, zq)], 0.009, sides=7, smooth_path=False))
    iron.append(cylinder("conduit_rose", 0.024, 0.006, loc=(xq, -0.003, zq), rot=(90, 0, 0), verts=12))
    iron.append(cylinder("conduit_gland", 0.013, 0.012, loc=(xq, -0.045, zs0 - 0.006), verts=8))

    # ---------------------------------------------------------------- bulbs: border run (+ pediment runs)
    sb = 0.031 + (xi - 0.031) / 2                 # trough centre line inset
    bulbs = []                                    # (x, depth, z)
    Lx, Lz = W - 2 * sb, H - 2 * sb
    perim = 2 * (Lx + Lz)
    sp = max(p.bulb_spacing, perim / 110.0)       # hard cap on bulb count
    nx, nz = max(2, int(round(Lx / sp))), max(1, int(round(Lz / sp)))
    for i in range(nx):
        x = -Lx / 2 + Lx * i / nx
        bulbs.append((x, dt, -Lz / 2))
        bulbs.append((-x, dt, Lz / 2))
    for j in range(nz):
        z = -Lz / 2 + Lz * j / nz
        bulbs.append((Lx / 2, dt, z))
        bulbs.append((-Lx / 2, dt, -z))

    # ---------------------------------------------------------------- pediment
    kind = p.pediment
    ph = min(p.pediment_height, 0.42 * W)
    xe = W / 2 + 0.02
    tt = 0.045                                    # tympanum board thickness
    ztop = zp0
    if kind == "scroll":
        mw = clamp(ph * 0.16, 0.045, 0.085)
        r1 = clamp(ph * 0.27, 0.06, 0.17)
        xv = max(0.14 * W, r1 + 0.12)
        zv = ph - r1 - mw / 2
        z0 = mw / 2 + 0.004
        bez = bezier_path((xe - mw / 2, 0, z0), (xe - (xe - xv) * 0.5, 0, z0), (xv + (xe - xv) * 0.5, 0, zv + r1), (xv, 0, zv + r1), n=14)
        turns, r0 = 1.12, r1 * 0.27
        spi = []
        ns = 24
        for i in range(1, ns + 1):
            f = i / ns
            a = radians(90 + 360 * turns * f)
            r = r1 + (r0 - r1) * f ** 0.9
            spi.append((xv + r * cos(a), 0, zv + r * sin(a)))
        cl = [(q[0], q[2]) for q in bez]                           # centre line (x, z) of the swan neck
        arc = [(xv + r1 * cos(radians(a)), zv + r1 * sin(radians(a))) for a in range(105, 271, 15)]
        Rh = [(xe, 0.0), (xe, z0)] + cl[1:] + arc
        outline = Rh + [(-x, z) for x, z in reversed(Rh)]
        tym = shape_extrude("tympanum", outline, depth=tt, loc=(0, 0, zp0), rot=(90, 0, 0))
        board.append(tym)
        prof = edge_mould(mw, tt)
        Lb = path_length(list(bez))
        Ls = path_length([bez[-1]] + spi)
        fb = Lb / (Lb + Ls)
        for sx in (1, -1):
            path = [(sx * q[0], -tt, zp0 + q[2]) for q in list(bez) + spi]
            sw = sweep(f"swan{sx}", path, prof if sx > 0 else [(-a, b) for a, b in reversed(prof)], up=(0, -1, 0), smooth_path=False,
                       scale_fn=lambda f: 1.0 if f < fb else 1.0 - 0.5 * ((f - fb) / (1 - fb)) ** 0.8)
            trim.append(sw)
            gold.append(lathe("rosette", [(0.0, 0.0), (r0 * 1.45, 0.0), (r0 * 1.5, 0.006), (r0 * 1.2, 0.014), (r0 * 0.9, 0.012),
                                          (r0 * 0.7, 0.02), (r0 * 0.35, 0.03), (0.0, 0.032)], segments=14,
                              loc=(sx * xv, -tt - 0.008, zp0 + zv), rot=(90, 0, 0)))
            # end block where the swan neck lands on the cornice
            trim.append(cornice_u("neck_block", 0.07, tt + 0.03, [(0.0, 0.0), (0.0, mw + 0.012), (0.006, mw + 0.016), (0.006, mw + 0.026)],
                                  x0=sx * (xe - 0.03), z0=zp0))
        # plinth and urn in the broken centre
        hp = max(0.06, zv - r1 + 0.05)
        trim.append(cornice_u("plinth", 0.13, 0.058, [(0.012, 0.0), (0.012, 0.016), (0.004, 0.021), (0.0, 0.024), (0.0, hp - 0.022),
                                                      (0.008, hp - 0.016), (0.014, hp - 0.008), (0.014, hp)], z0=zp0))
        # fluted oval paterae (gilt) on the tympanum under each swan neck
        xp = xv + r1 + (xe - xv - r1) * 0.4
        zcl = float(np.interp(xp, [q[0] for q in reversed(cl)], [q[1] for q in reversed(cl)])) - mw / 2
        rp = min(zcl * 0.3, (xe - xv - r1) * 0.14)
        if rp > 0.025:
            for sx in (1, -1):
                pat = lathe("patera", [(rp, 0.0), (rp, 0.005), (rp * 0.9, 0.014), (rp * 0.62, 0.02), (rp * 0.34, 0.012), (rp * 0.26, 0.02), (0.0, 0.024)],
                            segments=24, loc=(sx * xp, -tt, zp0 + zcl * 0.5), rot=(90, 0, 0))

                def flute(co, rp=rp):
                    r = hypot(co.x, co.y)
                    k = smoothstep(rp * 0.3, rp * 0.5, r) * (1.0 - smoothstep(rp * 0.95, rp, r))
                    return Vector((co.x * 1.9, co.y, co.z * (1.0 - 0.7 * k * (0.5 + 0.5 * cos(12 * atan2(co.y, co.x))))))

                transform_verts(pat, flute)
                gold.append(pat)
        ur = clamp(ph * 0.1, 0.03, 0.046)
        gold.append(finial("urn", ur, (0, -0.048, zp0 + hp), "urn"))
        ztop = zp0 + hp + ur * 6
    elif kind == "arch":
        mw = 0.082 * bs
        R = (xe * xe + ph * ph) / (2 * ph)
        zc0 = ph - R
        a_s = asin(clamp(xe / R, -1, 1))
        outline = [(R * sin(a), max(0.0, zc0 + R * cos(a))) for a in np.linspace(a_s, -a_s, 34)]
        tym = shape_extrude("tympanum", outline, depth=tt, loc=(0, 0, zp0), rot=(90, 0, 0))
        board.append(tym)
        Rm = R - mw / 2
        a_m = acos(clamp((R - ph + 0.012) / Rm, -1, 1))
        path = [(Rm * sin(a), -tt, zp0 + zc0 + Rm * cos(a)) for a in np.linspace(a_m, -a_m, 34)]
        h = mw / 2
        band = [(h, -tt), (h, 0.026), (h - 0.006, 0.034), (h - 0.013, 0.034), (h - 0.016, 0.022), (-h + 0.016, 0.022),
                (-h + 0.013, 0.034), (-h + 0.006, 0.034), (-h, 0.026), (-h, -tt)]
        trim.append(sweep("arch_band", path, band, up=(0, -1, 0), smooth_path=False))
        La = 2 * a_m * Rm
        na = max(3, int(round(La / sp)))
        for i in range(1, na):
            a = a_m - 2 * a_m * i / na
            bulbs.append((Rm * sin(a), tt + 0.022, zp0 + zc0 + Rm * cos(a)))
        for sx in (-1, 1):
            trim.append(cornice_u("arch_block", 0.085, tt + 0.036, [(0.0, 0.0), (0.0, 0.07), (0.007, 0.075), (0.007, 0.088)],
                                  x0=sx * (xe - 0.03), z0=zp0))
            gold.append(finial("arch_ball", 0.024, (sx * (xe - 0.03), -(tt + 0.036) / 2, zp0 + 0.088), "ball"))
        # gilt sunburst
        rs = clamp(ph * 0.27, 0.05, 0.2)
        sun = shape_extrude("sun", [(rs * cos(radians(a)), rs * sin(radians(a))) for a in range(0, 181, 12)], depth=0.012,
                            loc=(0, -tt, zp0), rot=(90, 0, 0), bevel=0.002)
        gold.append(sun)
        nr = 13
        for i in range(nr):
            th_ = radians(10 + 160 * i / (nr - 1))
            s_, c_ = sin(th_), cos(th_)
            reach = zc0 * s_ + sqrt(max((zc0 * s_) ** 2 - zc0 * zc0 + (R - mw - 0.02) ** 2, 0.0))
            rr0, rr1 = rs + 0.014, rs + (reach - rs) * (0.94 if i % 2 == 0 else 0.62)
            if rr1 - rr0 < 0.03:
                continue
            w0 = rs * 0.085
            pts = [(rr0 * c_ + w0 * s_, rr0 * s_ - w0 * c_), (rr1 * c_ + 0.002 * s_, rr1 * s_ - 0.002 * c_),
                   (rr1 * c_ - 0.002 * s_, rr1 * s_ + 0.002 * c_), (rr0 * c_ - w0 * s_, rr0 * s_ + w0 * c_)]
            if min(q[1] for q in pts) < 0.004:
                continue
            gold.append(shape_extrude(f"ray{i}", pts, depth=0.006, loc=(0, -tt, zp0), rot=(90, 0, 0)))
        ztop = zp0 + ph
    elif kind == "gable":
        mw = clamp(ph * 0.1, 0.045, 0.07)
        xc = clamp(0.2 * W, 0.24, 0.62)
        zs = ph * 0.5
        z0 = 0.03
        Rh = [(xe, 0.0)]
        for i in range(0, 11):
            f = radians(90 * i / 10)
            Rh.append((xe - (xe - xc) * sin(f), z0 + (zs - z0) * (1 - cos(f))))
        for i in range(1, 12):
            a = radians(90 * i / 12)
            Rh.append((xc * cos(a), zs + (ph - zs) * sin(a)))
        outline = Rh + [(-x, z) for x, z in reversed(Rh[:-1])]
        tym = shape_extrude("tympanum", outline, depth=tt, loc=(0, 0, zp0), rot=(90, 0, 0))
        board.append(tym)
        path = [(x, -tt, zp0 + z) for x, z in outline[1:-1]]
        trim.append(sweep("gable_mould", path, edge_mould(mw, tt), up=(0, -1, 0), step=0.06))
        for sx in (-1, 1):
            trim.append(cornice_u("gable_block", 0.075, tt + 0.03, [(0.0, 0.0), (0.0, 0.06), (0.006, 0.064), (0.006, 0.076)],
                                  x0=sx * (xe - 0.03), z0=zp0))
            gold.append(finial("gable_ball", 0.022, (sx * (xe - 0.03), -(tt + 0.03) / 2, zp0 + 0.076), "ball"))
            gold.append(finial("shoulder_ball", 0.02, (sx * (xc + 0.012), -tt / 2, zp0 + zs + mw * 0.3), "ball"))
        gold.append(finial("gable_urn", 0.026, (0, -tt / 2 - 0.004, zp0 + ph + mw * 0.3), "urn"))
        # medallion with monogram, ringed with bulbs
        rm = min(xc * 0.62, ph * 0.3)
        zm = ph * 0.47
        trim.append(cylinder("medallion", rm, 0.012, loc=(0, -tt - 0.006, zp0 + zm), rot=(90, 0, 0), verts=28, bevel=0.002, segments=1))
        gold.append(torus("medallion_ring", rm, 0.011, loc=(0, -tt - 0.012, zp0 + zm), rot=(90, 0, 0), major_seg=32, minor_seg=6))
        mono = str(p.monogram).strip()
        if mono:
            tm = text("probe2", mono, size=0.2, depth=0.01, font=p.font)
            c2 = np.array([v.co[:] for v in tm.data.vertices]) if len(tm.data.vertices) else np.zeros((1, 3))
            delete([tm])
            w2, h2 = max(c2[:, 0].max() - c2[:, 0].min(), 1e-3), max(c2[:, 1].max() - c2[:, 1].min(), 1e-3)
            k2 = min(rm * 1.15 / w2, rm * 1.05 / h2)
            gold.append(text("monogram", mono, size=0.2 * k2, depth=0.01, font=p.font,
                             loc=(-(c2[:, 0].max() + c2[:, 0].min()) / 2 * k2, -tt - 0.012, zp0 + zm - (c2[:, 1].max() + c2[:, 1].min()) / 2 * k2)))
        rbm = rm + 0.011 + 0.03 * bs
        nm = max(6, int(round(2 * pi * rbm / max(sp, 0.1))))
        if rbm + 0.03 < min(zm, xc - mw):
            for i in range(nm):
                a = 2 * pi * (i + 0.5) / nm
                bulbs.append((rbm * cos(a), tt, zp0 + zm + rbm * sin(a)))
        ztop = zp0 + ph
    else:   # flat blocking course with a raised centre tablet
        hb = clamp(ph * 0.4, 0.08, 0.16)
        trim.append(cornice_u("blocking", W + 0.02, tt, [(0.006, 0.0), (0.006, 0.012), (0.0, 0.016), (0.0, hb - 0.02), (0.006, hb - 0.016),
                                                         (0.012, hb - 0.008), (0.012, hb)], z0=zp0))
        wt = clamp(0.3 * W, 0.34, 0.9)
        ht = hb + clamp(ph * 0.45, 0.07, 0.2)
        board.append(cornice_u("tablet", wt, tt + 0.016, [(0.0, 0.0), (0.0, ht - 0.03)], z0=zp0))
        trim.append(cornice_u("tablet_cap", wt, tt + 0.016, [(0.0, 0.0), (0.008, 0.004), (0.016, 0.012), (0.016, 0.02), (0.006, 0.026), (0.0, 0.03)],
                              z0=zp0 + ht - 0.03))
        ros = lathe("tablet_rosette", [(0.0, 0.0), (0.03, 0.0), (0.033, 0.004), (0.024, 0.01), (0.016, 0.008), (0.011, 0.016), (0.0, 0.02)],
                    segments=12, loc=(0, -(tt + 0.016), zp0 + (ht - 0.03) * 0.55), rot=(90, 0, 0))
        gold.append(ros)
        for sx in (-1, 1):
            trim.append(cornice_u("end_block", 0.085, tt + 0.03, [(0.0, 0.0), (0.0, hb + 0.02), (0.007, hb + 0.025), (0.007, hb + 0.038)],
                                  x0=sx * (W / 2 - 0.03), z0=zp0))
            gold.append(finial("end_ball", 0.026, (sx * (W / 2 - 0.03), -(tt + 0.03) / 2, zp0 + hb + 0.038), "ball"))
            if wt / 2 + 0.2 < W / 2 - 0.2:
                bulbs.append((sx * (wt / 2 + 0.5 * (W / 2 - 0.08 - wt / 2)), tt + 0.0, zp0 + hb * 0.5))
        ztop = zp0 + ht

    # ---------------------------------------------------------------- stamp the bulbs (holder + glass)
    nbulb = len(bulbs)
    seg = 9 if nbulb <= 45 else 8 if nbulb <= 80 else 6
    holder = lathe("holder", [(0.0185 * bs, 0.0), (0.0175 * bs, 0.005), (0.0135 * bs, 0.007), (0.0135 * bs, 0.021 * bs)], segments=seg)
    glass = lathe("glass", [(0.0115 * bs, 0.019 * bs), (0.0195 * bs, 0.034 * bs), (0.0235 * bs, 0.050 * bs), (0.0185 * bs, 0.066 * bs),
                            (0.0, 0.0745 * bs)], segments=seg, smooth_angle=80)
    rbl = rng.fork("bulbs")
    live, dead, allm = [], [], []
    for (x, d, z) in bulbs:
        m = trs((x, -d, z), (90 + rbl.jitter(0, 2.5), 0, rbl.jitter(0, 2.5)), 1.0)
        allm.append(m)
        (dead if rbl.chance(p.dead_bulbs) else live).append(m)
    hold = instances_to_mesh("holders", holder, allm, keep_base=False)
    assign(hold, c_brass)
    if live:
        assign(instances_to_mesh("bulbs_lit", glass, live, keep_base=True), c_bulb)
    if dead:
        assign(instances_to_mesh("bulbs_dead", glass, dead, keep_base=True), c_dead)
    delete([glass])
    print(f"DBG bulbs={nbulb} dead={len(dead)} pitch={sp:.3f}")

    # ---------------------------------------------------------------- materials and weathering
    wr = p.wear
    for o in board:
        assign(o, c_board)
    for o in trim:
        assign(o, c_trim)
    for o in gold:
        assign(o, c_gold)
    for o in iron:
        assign(o, c_iron)

    def streaks(co, n):                                            # rain streaks running down from the cornice
        s = 0.5 + 0.5 * np.sin(co[:, 0] * 23.0 + seed) * np.sin(co[:, 0] * 7.3 + seed * 1.7)
        return np.clip(s, 0, 1) * np.clip((co[:, 2] + H * 0.5) / H, 0, 1) ** 1.5

    for o in board:
        paint(o, grime=mask_mul(streaks, 0.35 + 0.4 * wr), special=mask_mul(noise_mask(0.35, seed=seed + 1), 0.25 + 0.5 * wr),
              wear=mask_mul(noise_mask(0.12, seed=seed + 2, lo=0.55, hi=0.8), wr * 0.8))
    for o in trim:
        paint(o, grime=mask_mul(facing((0, 0, 1)), 0.3 + 0.4 * wr), wear=mask_mul(noise_mask(0.1, seed=seed + 3, lo=0.5, hi=0.8), wr),
              special=mask_mul(noise_mask(0.3, seed=seed + 5), 0.3 * wr))
    for o in iron:
        paint(o, special=mask_mul(noise_mask(0.05, seed=seed + 4, lo=0.3, hi=0.7), 0.35 + 0.5 * wr))
    paint(hold, special=mask_mul(noise_mask(0.07, seed=seed + 6), 0.2 + 0.6 * wr), grime=0.25)
    paint(cap, grime=mask_mul(noise_mask(0.15, seed=seed + 7), 0.6))

    # ---------------------------------------------------------------- lights and sockets
    nl = 2 if W < 1.9 else 3 if W < 3.3 else 4
    for i in range(nl):
        x = (-0.5 + (i + 0.5) / nl) * W
        light(f"marquee{i}", (x, -0.9, 0.0), color=p.bulb_color, intensity=p.intensity, range=6.0, flicker=0.03)
    if kind in ("arch", "gable"):
        light("pediment", (0, -0.8, zp0 + ph * 0.5), color=p.bulb_color, intensity=p.intensity * 0.6, range=4.5, flicker=0.03)
    socket("below", "attach", (0, 0, zs0 - cs_h), role="awning")
    socket("top", "attach", (0, -tt / 2, ztop), rot=(0, 0, 0), role="prop_top")
