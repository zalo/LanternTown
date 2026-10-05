"""Tall Georgian double-hung sash window. A painted boxed frame (outer linings, parting bead, sloped
timber cill) carries two sliding sashes: through stiles, rails, thin puttied glazing bars, one slightly
out-of-true crown-glass pane per light (each with its own faint sky sheen and dust along the putty), horns
on the upper sash, brass fastener and lifts. The joinery is a sound paint film that has let go in crisp
flakes on the cill nosing and the foot of the bottom rail. Behind the glass, spread through the depth of the
frame: roller blind with hem batten, real curtains, then - when lit - the room: an interior-mapped card on
the wall plane and, 1 cm in front of it, the lamp (the one hot spot) and a seeded choice of furniture as warm
smoky silhouettes (plant on a stand / table with oil lamp / chair back / bookcase / nothing; mirrored,
jittered, wallpaper tint varied). Unlit it is a dim moonlit room. Stays within ~0.135 m of the wall (0.165 m
with a cornice). Around it either a dressed-stone surround (jointed moulded architrave, keystone / cornice /
pediment head, moulded sill on scrolled consoles with an apron), the same in painted joinery with a leaded
hood, or a plain stucco reveal under an oak lintel. Optional wrought-iron sill guard of S-scrolls between
square bars.
"""
from tk import *
from math import sin, cos, pi, atan2, sqrt, tan, radians, degrees, floor
import numpy as np

META = {"name": "Sash window with surround", "class": "component", "mount": "wall", "mount_height": 1.15,
        "focus": 0.42}
PARAMS = {
    "width": {"default": 1.05, "min": 0.6, "max": 1.8, "desc": "outer frame width (m)"},
    "height": {"default": 1.9, "min": 1.0, "max": 2.8, "desc": "outer frame height (m)"},
    "panes": {"default": "6/6", "choices": ["6/6", "8/8", "9/9", "4/4", "2/2", "1/1"], "desc": "lights per sash (upper / lower)"},
    "lit": {"default": True, "desc": "room behind is lit (lamp-lit interior + light spill on sill and wall) or dark"},
    "surround": {"default": "stone", "choices": ["stone", "timber", "plain"], "desc": "stone architrave, painted joinery or plain stucco reveal with oak lintel"},
    "head": {"default": "keystone", "choices": ["keystone", "cornice", "pediment"], "desc": "head of the stone / timber surround"},
    "guard": {"default": False, "desc": "wrought-iron sill guard"},
    "guard_height": {"default": 0.3, "min": 0.2, "max": 0.5, "desc": "height of the sill guard (m)"},
    "curtain": {"default": "#8f3a30", "desc": "curtain colour"},
    "drawn": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "how far the curtains are drawn (0 = none, 1 = closed)"},
    "blind": {"default": 0.18, "min": 0.0, "max": 0.9, "desc": "roller blind pulled down (fraction of the height, 0 = none)"},
    "open": {"default": 0.0, "min": 0.0, "max": 0.7, "desc": "lower sash raised (fraction of its travel)"},
    "paint": {"default": "#e4dcc8", "desc": "paint colour of frame and sashes"},
    "surround_colour": {"default": "#bfbbae", "desc": "stone tint / joinery paint / stucco colour of the surround"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "weathering: flaked paint, grime streaks, moss"},
    "glow": {"default": 2.2, "min": 0.5, "max": 8.0, "desc": "brightness of the lit room"},
    "spill": {"default": 5.0, "min": 0.0, "max": 50.0, "desc": "light spilled onto the street (cd)"},
    "room": {"default": "auto", "choices": ["auto", "bare", "plant", "table_lamp", "chair", "bookcase"],
             "desc": "what stands behind the glass of a lit room (auto = picked, mirrored and jittered by the seed)"},
}
VARIANTS = {
    "cornice_guard": {"width": 1.3, "height": 2.2, "panes": "8/8", "head": "cornice", "guard": True, "curtain": "#2f4a6b",
                      "drawn": 0.4, "blind": 0.0, "room": "table_lamp", "seed": 4},
    "timber_dark": {"width": 0.95, "height": 1.7, "panes": "2/2", "surround": "timber", "head": "cornice", "lit": False,
                    "paint": "#d8d2c0", "surround_colour": "#24382c", "open": 0.45, "drawn": 0.3, "blind": 0.35, "seed": 11},
    "plain_cottage": {"width": 0.8, "height": 1.25, "panes": "4/4", "surround": "plain", "paint": "#5a1f1a",
                      "surround_colour": "#d8cdb4", "curtain": "#c9b98a", "drawn": 0.7, "blind": 0.0, "wear": 0.8, "room": "plant",
                      "seed": 23},
    "pediment_tall": {"width": 1.15, "height": 2.5, "panes": "9/9", "head": "pediment", "guard": True, "guard_height": 0.42,
                      "curtain": "#3d5a3a", "drawn": 0.6, "blind": 0.3, "surround_colour": "#bdb3a0", "room": "bookcase", "seed": 31},
}

GRID = {"1/1": (1, 1), "2/2": (2, 1), "4/4": (2, 2), "6/6": (3, 2), "8/8": (4, 2), "9/9": (3, 3)}


# ---------------------------------------------------------------- local mesh helpers
def _refine(secs, min_prof=0.03, along=True, e_prof=0.007, e_len=0.012, max_len=0.6):
    """Add edge loops just inside every long profile edge and near both ends of every span, so the
    vertex-baked AO / grime / wear stay in the corners instead of smearing across whole faces."""
    ref = secs[len(secs) // 2]
    n = len(ref)
    ts = []
    for j in range(n):
        k = (j + 1) % n
        L = sqrt(sum((ref[j][c] - ref[k][c]) ** 2 for c in range(3)))
        ts.append((j, 0.0))
        onwall = all(abs(sc[j][1]) < 1e-6 and abs(sc[k][1]) < 1e-6 for sc in secs)
        if L > min_prof and not onwall:
            ts.append((j, e_prof / L))
            if L > 0.3:
                ts.append((j, 0.5))
            ts.append((j, 1 - e_prof / L))
    secs = [[tuple(sc[j][c] + (sc[(j + 1) % n][c] - sc[j][c]) * t for c in range(3)) for j, t in ts] for sc in secs]
    out = [secs[0]]
    for i in range(len(secs) - 1):
        A, B = secs[i], secs[i + 1]
        ds = [sqrt(sum((A[q][c] - B[q][c]) ** 2 for c in range(3))) for q in range(len(A))]
        L = max(ds)
        if along and min(ds) > 0.04:
            nm = int(L // max_len)
            for f_ in [e_len / L] + [(q + 1) / (nm + 1) for q in range(nm)] + [1 - e_len / L]:
                out.append([tuple(A[q][c] + (B[q][c] - A[q][c]) * f_ for c in range(3)) for q in range(len(A))])
        out.append(B)
    return out


def _loft(name, secs, cap=True, refine=True, along=True, min_prof=0.03):
    """Skin a list of closed sections (same point count) into one solid."""
    if refine:
        secs = _refine(secs, min_prof, along)
    n, m = len(secs[0]), len(secs)
    verts = [tuple(v) for s in secs for v in s]
    faces = []
    for i in range(m - 1):
        for j in range(n):
            k = (j + 1) % n
            faces.append((i * n + j, i * n + k, (i + 1) * n + k, (i + 1) * n + j))
    if cap:
        faces.append(tuple(range(n - 1, -1, -1)))
        faces.append(tuple(range((m - 1) * n, m * n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _px(name, yz, x0, x1, **kw):
    return _loft(name, [[(x0, y, z) for y, z in yz], [(x1, y, z) for y, z in yz]], **kw)


def _pxn(name, yz, x0, x1, step, min_prof=0.03):
    """Like _px but with a section every `step` m, so per-vertex paint masks (flaking, streaks) have resolution."""
    n = max(1, int(round((x1 - x0) / step)))
    return _loft(name, [[(x0 + (x1 - x0) * i / n, y, z) for y, z in yz] for i in range(n + 1)], along=False, min_prof=min_prof)


def _pz(name, xy, z0, z1, zs=None):
    """Extrude an XY profile up Z; zs = explicit section heights (edge loops where the paint masks change)."""
    if zs:
        return _loft(name, [[(x, y, z) for x, y in xy] for z in sorted(set([z0, z1] + [q for q in zs if z0 < q < z1]))], along=False)
    return _loft(name, [[(x, y, z0) for x, y in xy], [(x, y, z1) for x, y in xy]])


def _py(name, xz, y0, y1, refine=True):
    return _loft(name, [[(x, y0, z) for x, z in xz], [(x, y1, z) for x, z in xz]], refine=refine)


def _returned(name, prof, L, d0, z0):
    """Moulding (p = projection, q = height) along the front of a block L wide, d0 deep, returned to the wall."""
    hl = L / 2
    return _loft(name, [[(-hl - a, 0.0, z0 + q) for a, q in prof], [(-hl - a, -(d0 + a), z0 + q) for a, q in prof],
                        [(hl + a, -(d0 + a), z0 + q) for a, q in prof], [(hl + a, 0.0, z0 + q) for a, q in prof]])


def _bez(p0, p1, p2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append(tuple(u ** 3 * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t ** 3 * p3[k] for k in (0, 1)))
    return out


def _sheet(name, fn, nu, nv):
    """Open grid facing -Y: u runs +X, v runs +Z."""
    verts = [fn(i / nu, j / nv) for j in range(nv + 1) for i in range(nu + 1)]
    faces = [(j * (nu + 1) + i, j * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i)
             for j in range(nv) for i in range(nu)]
    return mesh_object(name, verts, faces)


def _s_scroll(height, turns=1.05, n=22):
    """Clothoid S-scroll (curvature grows linearly towards both ends) standing upright, centred, as [(x, z)]."""
    c = 4 * pi * turns
    N = 600
    half = [(0.0, 0.0)]
    x = z = 0.0
    for i in range(N):
        s = (i + 0.5) / N
        ph = c * s * s / 2
        x += cos(ph) / N
        z += sin(ph) / N
        half.append((x, z))
    m = n // 2
    pts = [(-half[int(round(sqrt(i / m) * N))][0], -half[int(round(sqrt(i / m) * N))][1]) for i in range(m, 0, -1)]
    pts.append((0.0, 0.0))
    pts += [half[int(round(sqrt(i / m) * N))] for i in range(1, m + 1)]
    tail = half[int(N * 0.8):]
    ang = pi / 2 - atan2(sum(q[1] for q in tail), sum(q[0] for q in tail))
    ca, sa = cos(ang), sin(ang)
    pts = [(q[0] * ca - q[1] * sa, q[0] * sa + q[1] * ca) for q in pts]
    ext = max(q[1] for q in pts) - min(q[1] for q in pts)
    k = height / ext
    return [(q[0] * k, q[1] * k) for q in pts]


def _console(name, xc, w, zt, h, proj):
    """Scrolled console / corbel: S-profile in YZ, w wide, hanging h below zt, projecting proj from the wall."""
    prof = [(0.0, zt), (-proj, zt), (-proj, zt - 0.02), (-proj + 0.008, zt - 0.03)]
    prof += _bez((-proj + 0.002, zt - 0.045), (-proj - 0.01, zt - 0.5 * h), (-0.22 * proj, zt - 0.45 * h), (-0.2 * proj, zt - 0.82 * h), 4)
    prof += [(-0.27 * proj, zt - 0.88 * h), (-0.28 * proj, zt - 0.95 * h), (-0.2 * proj, zt - h), (0.0, zt - h)]
    o = _px(name, prof, xc - w / 2, xc + w / 2, min_prof=0.08)
    bevel(o, 0.004, segments=1, angle=60)
    smooth(o, 42)
    return o


# ---------------------------------------------------------------- the asset
def build(p, rng):
    W, H = p.width, p.height
    hw = W / 2
    wr = p.wear
    lit = bool(p.lit)
    kind = p.surround
    plain, stone = kind == "plain", kind == "stone"

    def film(col):                   # sound oil paint: an opaque, faintly glossy film without any figure of its own
        return mat("ceramic_glazed", col, rough=1.6, wear=0.0)

    wood = film(p.paint)
    brass = mat("brass")
    glass = mat("glass_clear", "#dfeaf2", opacity=0.2, rough=0.3) if lit else mat("glass_clear", "#dce8ee", opacity=0.2, rough=0.5)
    sheen = mat("glass_lantern", "#a9bfd6", glow=0.2 if lit else 0.3, opacity=0.1)        # cool sky caught by a pane, dust along the putty
    if lit:                          # back-lit fabric glows in its own colour
        cloth = mat("glass_frosted", p.curtain, glow=0.55, opacity=1.0)
        linen = None
    else:                            # dark room: just enough of the fabrics left to read them behind the glass
        cloth = mat("glass_frosted", p.curtain, glow=0.03, opacity=1.0)
        linen = mat("glass_frosted", "#d6c8a4", glow=0.05, opacity=1.0)
    if stone:
        smat = mat("mortar", p.surround_colour, scale=1.5)       # fine granular limestone: no directional figure
    elif kind == "timber":
        smat = mat("plain", p.surround_colour)
    else:
        smat = mat("plaster_lime", p.surround_colour)
    sill_mat = smat
    # where the paint has let go: grey weathered substrate (a material the window carries anyway)
    flake = smat if stone else (mat("lead") if kind == "timber" else mat("wood_oak", "#5a4332"))
    timber, stones, irons, liners = [], [], [], []
    # ================================================================ boxed frame
    # The window is set in the wall: only the outer linings stand proud. Depths (d = -y): room card 0.006 | furniture
    # veils 0.0105 / 0.0192, lamp shade 0.0148 | curtains 0.0263 +- 0.0029 (tie-back box 0.0192 .. 0.0334) | blind
    # 0.0376, its four shading veils 0.0418 .. 0.046 (one mesh, stored back to front), threads 0.0474, slubs 0.0488,
    # batten 0.0419 .. 0.0473 | lower sash back 0.0515, glass 0.057 +- 0.0016, sheen 0.063, face 0.067 | upper sash
    # glass 0.072, sheen 0.078, face 0.082 | lining 0.0895
    # CLEARANCE RULE (depth fighting under bloom): every glowing layer (room card, lamp shade, lit curtains, lit
    # blind, pane sheen) keeps >= 4 mm from whatever lies parallel behind it and in front of it - the host wall
    # behind the card included - and no two sheets share a plane. Keep it when re-spacing anything here.
    LF, PS = 0.058, 0.042            # lining face width, pulley stile set-back
    D_FR, D_PB = 0.0895, 0.0833
    DL = (0.0515, 0.057, 0.067)      # lower sash: back, glass, face
    DU = (0.068, 0.072, 0.082)       # upper sash
    CZ = 0.055                         # top of the timber cill under the lower sash
    CN = D_FR + 0.020                  # cill nosing
    fk = rng.fork("flake")
    flk = {}
    for sx in (-1, 1):
        xy = [(hw, 0.0), (hw, -D_FR), (hw - LF + 0.005, -D_FR), (hw - LF, -D_FR + 0.004), (hw - LF, -D_PB), (hw - PS, -D_PB), (hw - PS, 0.0)]
        j = _pz("frame_jamb", [(sx * x, y) for x, y in xy], 0.03, H, zs=[CZ + 0.004, CZ + 0.03, 0.2, H - 0.012])
        uv_grain(j, "Z")
        timber.append(j)
    yz = [(0.0, H), (-D_FR, H), (-D_FR, H - LF + 0.005), (-D_FR + 0.004, H - LF), (-D_PB, H - LF), (-D_PB, H - PS), (0.0, H - PS)]
    head = _px("frame_head", yz, -hw + LF, hw - LF)
    uv_grain(head, "X")
    timber.append(head)
    # timber cill let 9 mm into the stone sill (no slot under it for light to leak through), throated nosing
    cill, fl_ = _flaky("frame_cill", [(0.0, -0.009), (0.0, CZ), (-(DL[2] + 0.002), CZ), (-(CN - 0.027), 0.0474), (-(CN - 0.013), 0.0404), (-(CN - 0.004), 0.036),
                                      (-CN, 0.031), (-CN, 0.011), (-(CN - 0.003), 0.006), (-(CN - 0.012), 0.006), (-(CN - 0.014), -0.009)],
                       -hw, hw, int(round((1.2 + 4 * wr) * W)), fk, 4, 0.12, 0.8, 0.6, 0.85)
    flk["cill"] = (cill, fl_)
    uv_grain(cill, "X")
    timber.append(cill)

    # ================================================================ sashes
    cols, rows = GRID[p.panes]
    zc, zh = CZ, H - PS - 0.002
    MR = 0.034                          # meeting rails
    hs = (zh - zc + MR) / 2
    lift = p.open * (hs - 0.16)
    SW = 0.048
    panes_v, panes_f = [], []
    gr = rng.fork("glass")
    NG = 2 if cols * rows <= 6 else 1
    sheens = []

    def sash(tag, z0, z1, dd3, top_h, bot_h, dense_bot=False):
        db, dg, dfront = dd3
        x0, x1 = -(hw - PS - 0.002), hw - PS - 0.002
        for sx in (-1, 1):
            xo, xi = sx * x1, sx * (x1 - SW)
            st = _pz(tag + "_stile", [(xo, -db), (xi, -db), (xi, -dg), (xi + sx * 0.009, -dfront), (xo, -dfront)], z0, z1,
                     zs=[z0 + 0.012, 0.17, z1 - 0.012])
            uv_grain(st, "Z")
            timber.append(st)
        xa, xb = x0 + SW, x1 - SW
        za, zb = z0 + bot_h, z1 - top_h
        top = _px(tag + "_top", [(-db, z1), (-db, zb), (-dg, zb), (-dfront, zb + 0.009), (-dfront, z1)], xa, xb)
        if dense_bot:
            Lb = za - 0.009 - z0
            bot, fl_ = _flaky(tag + "_bot", [(-db, z0), (-db, za), (-dg, za), (-dfront, za - 0.009), (-dfront, z0 + 0.014), (-dfront, z0)],
                              xa, xb, int(round((1.0 + 4 * wr) * W)), fk, 4, 1 - 0.032 / Lb, 1 - 0.006 / Lb, 1 - 0.014 / Lb, 1 - 0.004 / Lb)
            flk["bot"] = (bot, fl_, z0)
        else:
            bot = _px(tag + "_bot", [(-db, z0), (-db, za), (-dg, za), (-dfront, za - 0.009), (-dfront, z0)], xa, xb)
        for o in (top, bot):
            uv_grain(o, "X")
            timber.append(o)
        # glazing bars: square nib standing proud of the two putty fillets that bed the glass
        bw, sh, nib = 0.011, 0.0052, 0.0034
        ps_, pn = dg + 0.0056, dfront - 0.001
        for c in range(1, cols):
            xc = xa + (xb - xa) * c / cols
            b = _pz(tag + "_vbar", [(xc - bw, -dg), (xc - sh, -ps_), (xc - nib, -pn), (xc + nib, -pn), (xc + sh, -ps_), (xc + bw, -dg),
                                    (xc + 0.006, -dg + 0.003), (xc - 0.006, -dg + 0.003)], za - 0.002, zb + 0.002)
            uv_grain(b, "Z")
            timber.append(b)
        for r in range(1, rows):
            zr = za + (zb - za) * r / rows
            b = _px(tag + "_hbar", [(-dg, zr - bw), (-ps_, zr - sh), (-pn, zr - nib), (-pn, zr + nib), (-ps_, zr + sh), (-dg, zr + bw),
                                    (-dg + 0.003, zr + 0.006), (-dg + 0.003, zr - 0.006)], xa - 0.002, xb + 0.002)
            uv_grain(b, "X")
            timber.append(b)
        # one crown-glass pane per light: out of true and faintly bellied, so every pane mirrors the lanterns differently
        for c in range(cols):
            for r in range(rows):
                px0, px1 = xa + (xb - xa) * c / cols, xa + (xb - xa) * (c + 1) / cols
                pz0, pz1 = za + (zb - za) * r / rows, za + (zb - za) * (r + 1) / rows
                tx, tz = gr.jitter(0.0, 0.007), gr.jitter(0.0, 0.007)
                bel, ph, fq = gr.jitter(0.0, 0.0009), gr.u(0, 6.28), gr.u(1.2, 2.4)
                i0 = len(panes_v)
                for jj in range(NG + 1):
                    for ii in range(NG + 1):
                        u, v = ii / NG, jj / NG
                        vx, vz = lerp(px0, px1, u), lerp(pz0, pz1, v)
                        dy = tx * (u - 0.5) * (px1 - px0) + tz * (v - 0.5) * (pz1 - pz0)
                        dy += bel * (1 - (2 * u - 1) ** 2) * (1 - (2 * v - 1) ** 2) + 0.00035 * sin(fq * 6.28 * (u + 0.6 * v) + ph)
                        panes_v.append((vx, -dg + max(-0.0016, min(0.0016, dy)), vz))
                for jj in range(NG):
                    for ii in range(NG):
                        a = i0 + jj * (NG + 1) + ii
                        panes_f.append((a, a + 1, a + NG + 2, a + NG + 1))
                # what makes old glass visible: every light sits at its own angle and mirrors another bit of the street
                # (a faint cool sheen over the whole of some lights, on most only above a slanting line - the roofs
                # opposite), and dust gathers along the bottom putty and in the corners. One plane, no overlaps.
                # (the sheen lies 6 mm proud of the glass: >= 4 mm clear of the bellied pane; towards a glazing bar it
                # stops where it would enter the bar, so it never lies under the flat of a nib)
                ds = dg + 0.006
                h1, h2 = gr.u(0.006, 0.02), gr.u(0.006, 0.02)
                px0, px1 = px0 + (0.005 if c > 0 else 0.0), px1 - (0.005 if c < cols - 1 else 0.0)
                pz0, pz1 = pz0 + (0.005 if r > 0 else 0.0), pz1 - (0.005 if r < rows - 1 else 0.0)
                kc_ = min(1.0, (px1 - px0) / 0.2)
                c1, c2 = gr.u(0.02, 0.05) * kc_, gr.u(0.02, 0.05) * kc_
                dust = [(px0, pz0), (px1, pz0), (px1, pz0 + h2 + 1.5 * c2), (px1 - c2, pz0 + h2), (px0 + c1, pz0 + h1), (px0, pz0 + h1 + 1.5 * c1)]
                sheens.append((ds, dust))
                rq = gr.u()
                if rq < 0.2:
                    sheens.append((ds, dust[:1:-1] + [(px1, pz1), (px0, pz1)]))
                elif rq < 0.72:
                    zq, sl = lerp(pz0, pz1, gr.u(0.15, 0.85)), gr.u(-0.14, 0.14) * (px1 - px0)
                    zq0, zq1 = clamp(zq - sl, dust[5][1] + 0.004, pz1), clamp(zq + sl, dust[2][1] + 0.004, pz1)
                    sheens.append((ds, [(px0, zq0), (px1, zq1), (px1, pz1), (px0, pz1)]))
        return x0, x1

    # lower sash (inner track), deep bottom rail; upper sash (outer track) with horns
    zl0, zl1 = zc + lift, zc + hs + lift
    sash("lower", zl0, zl1, DL, MR, 0.082, dense_bot=True)
    x0, x1 = sash("upper", zh - hs, zh, DU, 0.050, MR)
    for sx in (-1, 1):
        xo, xi = sx * x1, sx * (x1 - SW)
        zb = zh - hs
        horn = [(xo, zb + 0.004), (xi, zb + 0.004), (xi, zb - 0.014)] + \
               _bez((xi, zb - 0.014), (xi + sx * 0.004, zb - 0.04), (xo - sx * 0.02, zb - 0.03), (xo - sx * 0.012, zb - 0.06), 5)[1:] + [(xo, zb - 0.06)]
        h = _py("upper_horn", horn, -DU[0], -DU[2], refine=False)
        uv_grain(h, "Z")
        timber.append(h)
    gl = mesh_object("glass", panes_v, panes_f)
    uv_box(gl)
    smooth(gl, 60)
    assign(gl, glass)
    assign(_cutouts("glass_sheen", sheens), sheen)

    def tim_grime(co, n):             # dirt in the putty line round every pane, damp at the cill
        y = -co[:, 1]
        putty = ((np.abs(y - DL[1]) < 0.002) | (np.abs(y - DU[1]) < 0.002)) * 0.7
        low = np.clip(1.0 - co[:, 2] / 0.2, 0.0, 1.0) * (0.06 + 0.2 * wr)
        return np.maximum(putty, low)

    # A sound paint film everywhere (no wear mask at all). It only lets go where water sits, and then in crisp flakes
    # down to the grey weathered timber: cill nosing, foot of the bottom rail, feet of the stiles and linings.
    for o in timber:
        base = o.name.split(".")[0]
        if "bar" not in base and base not in ("frame_cill", "lower_bot"):
            bevel(o, 0.0016, segments=1, angle=35)
        assign(o, wood)
        paint(o, wear=0.0, grime=tim_grime, special=0.0)
        if base == "lower_stile" and fk.chance(0.3 + 0.5 * wr):
            assign_faces(o, flake, lambda c, n: n.y < -0.5 and c.z < zl0 + 0.012)
        elif base == "frame_jamb" and fk.chance(0.25 + 0.6 * wr):
            assign_faces(o, flake, lambda c, n: n.y < -0.5 and CZ + 0.004 < c.z < CZ + 0.03)
    o, spans = flk["cill"]
    assign_faces(o, flake, lambda c, n: 0.007 < c.z < 0.0418 and c.y < -(CN - 0.031) and abs(n.x) < 0.5 and any(a - 0.003 < c.x < b + 0.003 for a, b in spans))
    o, spans, zq = flk["bot"]
    assign_faces(o, flake, lambda c, n: n.y < -0.5 and c.z < zq + 0.022 and any(a - 0.003 < c.x < b + 0.003 for a, b in spans))

    # sash furniture: fastener on the meeting rail, two hook lifts on the bottom rail
    furn = []
    zf_ = zl1
    yf = -(DL[0] + DL[2]) / 2
    furn.append(box("fast_plate", (0.058, 0.014, 0.003), loc=(0, yf, zf_ + 0.0015), bevel=0.001, segments=1))
    furn.append(cylinder("fast_boss", 0.008, 0.008, loc=(0.004, yf, zf_ + 0.007), verts=10))
    furn.append(box("fast_arm", (0.040, 0.006, 0.004), loc=(-0.012, yf, zf_ + 0.012), rot=(0, 0, 14), bevel=0.001, segments=1))
    furn.append(sphere("fast_knob", 0.0055, loc=(-0.031, yf + 0.005, zf_ + 0.0125), segments=8, rings=5))
    for sx in (-1, 1):
        xl = sx * W * 0.21
        zl = zl0 + 0.036
        furn.append(box("lift_plate", (0.022, 0.0025, 0.034), loc=(xl, -DL[2] - 0.0012, zl), bevel=0.001, segments=1))
        furn.append(sweep("lift_hook", [(xl, -DL[2] - 0.002, zl + 0.004), (xl, -DL[2] - 0.011, zl + 0.002), (xl, -DL[2] - 0.015, zl - 0.006),
                                        (xl, -DL[2] - 0.012, zl - 0.013)], [(-0.008, -0.0017), (0.008, -0.0017), (0.008, 0.0017), (-0.008, 0.0017)], up=(1, 0, 0), smooth_path=True, step=0.007))
    for o in furn:
        assign(o, brass)
        paint(o, grime=0.35, special=0.06 + 0.14 * wr, wear=0.3)

    # ================================================================ the room: card, furniture, lamp, curtains, blind
    # From the wall plane outwards: an interior-mapped room card (shaded walls, floor, ceiling with real parallax); 1 cm in
    # front of it the lamp and the furniture as two staggered layers of smoky veil (dim warm brown, a lighter rim towards
    # the lamp); the curtains; and just behind the lower sash the blind. Which furniture stands there and on which side,
    # where the lamp hangs, the cut of the room and the colour of its walls all come from the seed (or the `room` param).
    wi = hw - PS
    rz0, rz1 = CZ - 0.01, H - PS + 0.005
    hr = rz1 - rz0
    gs = p.glow / 2.2
    rr = rng.fork("room")
    side = rr.pick((-1, 1))                                # the furniture stands on this side, the lamp hangs towards the other
    room = rr.pick(("plant", "table_lamp", "chair", "bookcase", "plant", "table_lamp", "chair", "bare"))
    if p.room != "auto":
        room = p.room
    mirror = rr.pick((-1, 1))
    tint = rr.pick(("#e0a868", "#d9a274", "#d8ac78", "#dcb47c", "#d09c68", "#e4b88c", "#d4a880", "#d6a060"))
    zoom_u, v0 = rr.u(0.86, 1.0), rr.u(0.0, 0.07)
    zoom_v = min(1.0 - v0, rr.u(0.86, 0.98))
    shift_u = rr.u(-0.4, 0.4) * (1 - zoom_u)
    cell = (rr.pick((0, 1, 2, 3, 4, 5)), rr.pick((0, 1, 2, 3)))
    xf = side * rr.u(0.40, 0.62) * wi
    lamp_x, lamp_z = -side * rr.u(0.04, 0.36) * wi, rz0 + rr.u(0.56, 0.72) * hr
    if p.blind > 0.02:                                    # a hanging lamp shows below the blind
        lamp_z = max(rz0 + 0.3 * hr, min(lamp_z, (H - PS) - p.blind * (H - PS - CZ - 0.004) - 0.17))
    FS = min(1.15, max(0.6, wi / 0.45))                   # small windows show smaller things
    D_CARD, D_F0, D_F1, D_SH, D_CU, D_BL = 0.006, 0.0105, 0.0192, 0.0148, 0.0263, 0.0376
    veils, hots = [], []

    card = _cutouts("room", [(D_CARD, [(-wi, rz0), (wi, rz0), (wi, rz1), (-wi, rz1)])])
    ku, kv = mirror * zoom_u / (2 * wi), zoom_v / hr
    uv_planar(card, (ku, 0, 0), (0, 0, kv), offset=(0.5 + shift_u, v0 - rz0 * kv))
    room_mat = mat("window_lit", tint, glow=round(1.5 * gs, 3), curtain=0.0) if lit else mat("window_lit", "#4c5a74", glow=0.035, curtain=0.0)
    assign(card, room_mat)

    def sil(pts):
        """Silhouette: the outline on the back layer, on the front layer a copy shrunk away from the lamp (and from above)."""
        xs_, zs_ = [q[0] for q in pts], [q[1] for q in pts]
        ax, bx, az, bz_ = min(xs_), max(xs_), min(zs_), max(zs_)
        tw_ = 1.0 if lamp_x > (ax + bx) / 2 else -1.0
        fx, fz = (ax + bx) / 2 - tw_ * 0.3 * (bx - ax), (az + bz_) / 2 - 0.3 * (bz_ - az)
        kx = max(0.4, 1 - 0.007 / max(1e-4, 0.8 * (bx - ax)))
        kz = max(0.4, 1 - 0.007 / max(1e-4, 0.8 * (bz_ - az)))
        veils.append((D_F0, pts))
        veils.append((D_F1, [(fx + (x - fx) * kx, fz + (z - fz) * kz) for x, z in pts]))

    def ring(xc, zc_, r, n=10):
        return [(xc + r * cos(2 * pi * i / n), zc_ + r * sin(2 * pi * i / n)) for i in range(n)]

    def sym(xc, zb_, prof):                               # turned outline from a (radius, height) profile
        return [(xc + r, zb_ + q) for r, q in prof] + [(xc - r, zb_ + q) for r, q in reversed(prof)]

    def band(xa_, za_, xb_, zb_, w, bow=0.0, n=4):        # bar from a to b, w thick, bowed by `bow`
        top, low = [], []
        for i in range(n + 1):
            t = i / n
            x, z = lerp(xa_, xb_, t), lerp(za_, zb_, t) + bow * (1 - (2 * t - 1) ** 2)
            top.append((x, z + w / 2))
            low.append((x, z - w / 2))
        return low + top[::-1]

    if lit:
        if room == "table_lamp":
            # round table under a cloth; on it an oil lamp with a glowing globe (the only lamp in the room) and a few books
            tw = min(0.21, 0.5 * wi)
            zt_ = rz0 + rr.u(0.08, 0.18) * hr
            lamp_x = xf - side * rr.u(0.0, 0.4) * tw
            lamp_z = zt_ + (0.098 + 0.05) * FS
            sil([(xf - tw - 0.02, rz0 - 0.004), (xf + tw + 0.02, rz0 - 0.004), (xf + tw, zt_ - 0.006), (xf + tw - 0.004, zt_), (xf - tw + 0.004, zt_), (xf - tw, zt_ - 0.006)])
            sil(sym(lamp_x, zt_, [(r * FS, q * FS) for r, q in ((0.042, 0.0), (0.046, 0.007), (0.020, 0.018), (0.014, 0.040), (0.034, 0.055), (0.040, 0.075), (0.030, 0.092), (0.016, 0.098))]))
            hots.append((D_SH, ring(lamp_x, lamp_z, 0.054 * FS, 12)))
            sil([(lamp_x - 0.016 * FS, lamp_z + 0.05 * FS), (lamp_x + 0.016 * FS, lamp_z + 0.05 * FS), (lamp_x + 0.011 * FS, lamp_z + 0.14 * FS), (lamp_x - 0.011 * FS, lamp_z + 0.14 * FS)])
            xk, zk_ = xf + side * 0.5 * tw, zt_
            for i in range(rr.pick((2, 3, 4))):
                w_, h_ = rr.u(0.05, 0.075) * FS, rr.u(0.014, 0.022)
                o_ = rr.u(-0.012, 0.012)
                sil([(xk - w_ + o_, zk_), (xk + w_ + o_, zk_), (xk + w_ + o_, zk_ + h_), (xk - w_ + o_, zk_ + h_)])
                zk_ += h_ + 0.001
        else:
            # pendant oil lamp: chain and gallery, glowing shade (cone or opal globe), burner ring and font below
            if rr.chance(0.35):
                hots.append((D_SH, ring(lamp_x, lamp_z + 0.052, 0.058, 12)))
                top_, rim_ = lamp_z + 0.106, 0.045
            else:
                sh0, sh1, shh = rr.u(0.075, 0.092), rr.u(0.026, 0.036), rr.u(0.068, 0.085)
                hots.append((D_SH, [(lamp_x - sh0, lamp_z), (lamp_x + sh0, lamp_z), (lamp_x + sh1, lamp_z + shh), (lamp_x - sh1, lamp_z + shh)]))
                top_, rim_ = lamp_z + shh, sh0 + 0.004
            sil([(lamp_x - 0.0025, rz1), (lamp_x - 0.0025, top_ + 0.024), (lamp_x - 0.012, top_ + 0.022), (lamp_x - 0.022, top_ - 0.002), (lamp_x + 0.022, top_ - 0.002),
                 (lamp_x + 0.012, top_ + 0.022), (lamp_x + 0.0025, top_ + 0.024), (lamp_x + 0.0025, rz1)])
            sil([(lamp_x + 0.038 * cos(a_), lamp_z - 0.008 + 0.036 * sin(a_)) for a_ in [pi + pi * i / 8 for i in range(9)]] +
                [(lamp_x + rim_, lamp_z - 0.008), (lamp_x + rim_, lamp_z - 0.001), (lamp_x - rim_, lamp_z - 0.001), (lamp_x - rim_, lamp_z - 0.008)])
        if room == "plant":
            # aspidistra in a pot on a turned stand
            zs_ = rz0 + rr.u(0.05, 0.15) * hr
            pr_, ph_ = 0.062 * FS, 0.105 * FS
            sil([(xf - 0.016, rz0 - 0.004), (xf + 0.016, rz0 - 0.004), (xf + 0.013, zs_ - 0.03), (xf + 0.03, zs_ - 0.012), (xf + pr_ + 0.02, zs_ - 0.012), (xf + pr_ + 0.02, zs_),
                 (xf - pr_ - 0.02, zs_), (xf - pr_ - 0.02, zs_ - 0.012), (xf - 0.03, zs_ - 0.012), (xf - 0.013, zs_ - 0.03)])
            sil(sym(xf, zs_, [(pr_ * 0.72, 0.0), (pr_ * 0.80, 0.008), (pr_ * 0.98, ph_ * 0.8), (pr_ * 1.1, ph_ * 0.84), (pr_ * 1.1, ph_)]))
            nl = rr.pick((7, 8, 9))
            for i in range(nl):
                a_ = radians(lerp(25, 155, (i + rr.u(0.15, 0.85)) / nl))
                L_ = rr.u(0.22, 0.38) * FS * (0.7 + 0.3 * sin(a_))
                droop = rr.u(0.15, 0.55) * L_ * abs(cos(a_))
                wl = rr.u(0.03, 0.045) * FS
                le, ri = [], []
                for q in range(7):
                    t = q / 6
                    cx_ = xf + (0.5 * pr_ + L_ * t) * cos(a_)
                    cz_ = zs_ + ph_ - 0.012 + L_ * t * sin(a_) - droop * t * t
                    w_ = 0.5 * wl * sin(pi * min(1.0, 0.04 + t) ** 0.75) + 0.003 * (1 - t)
                    dx_, dz_ = L_ * cos(a_), L_ * sin(a_) - 2 * droop * t
                    dl = sqrt(dx_ * dx_ + dz_ * dz_)
                    le.append((cx_ - dz_ / dl * w_, cz_ + dx_ / dl * w_))
                    ri.append((cx_ + dz_ / dl * w_, cz_ - dx_ / dl * w_))
                sil(ri + le[::-1])
        elif room == "chair":
            # high chair back seen from behind: turned posts with finials, bowed crest rail, slats or spindles
            cw_ = min(0.19, 0.4 * wi) * rr.u(0.9, 1.1)
            zt_ = rz0 + rr.u(0.24, 0.38) * hr
            ln = -side * rr.u(0.0, 0.03)
            for sg in (-1, 1):
                xb_, xt_ = xf + sg * cw_ * 0.9, xf + sg * cw_ + ln
                sil([(xb_ - 0.012, rz0 - 0.004), (xb_ + 0.012, rz0 - 0.004), (xt_ + 0.009, zt_), (xt_ - 0.009, zt_)])
                sil(ring(xt_, zt_ + 0.013, 0.016, 8))
            sil(band(xf - cw_ + ln, zt_ - 0.05, xf + cw_ + ln, zt_ - 0.05, 0.05, 0.028))
            if rr.chance(0.5):
                for i in range(1, 4):
                    zs_ = zt_ - 0.05 - i * rr.u(0.075, 0.095)
                    if zs_ > rz0 + 0.02:
                        sil(band(xf - cw_ * 0.95 + ln * 0.7, zs_, xf + cw_ * 0.95 + ln * 0.7, zs_, 0.032, 0.012))
            else:
                zr_ = max(rz0 + 0.012, zt_ - rr.u(0.26, 0.34))
                sil(band(xf - cw_ * 0.93 + ln * 0.3, zr_, xf + cw_ * 0.93 + ln * 0.3, zr_, 0.026))
                ns_ = rr.pick((3, 4, 5))
                for i in range(1, ns_ + 1):
                    t = i / (ns_ + 1)
                    xs0, xs1 = xf + (2 * t - 1) * cw_ * 0.9 + ln * 0.3, xf + (2 * t - 1) * cw_ + ln
                    sil([(xs0 - 0.006, zr_ + 0.013), (xs0 + 0.006, zr_ + 0.013), (xs1 + 0.005, zt_ - 0.06), (xs1 - 0.005, zt_ - 0.06)])
        elif room == "bookcase":
            # low bookcase against the side wall with a moulded top; on it a vase and a row of books
            lo, hi = sorted((side * rr.u(0.15, 0.4) * wi, side * (wi + 0.004)))
            ztop = rz0 + rr.u(0.18, 0.34) * hr
            sil([(lo, rz0 - 0.004), (hi, rz0 - 0.004), (hi, ztop), (hi + 0.012, ztop + 0.005), (hi + 0.012, ztop + 0.02), (lo - 0.012, ztop + 0.02),
                 (lo - 0.012, ztop + 0.005), (lo, ztop)])
            zs_ = ztop + 0.02
            xv_ = (lo if side > 0 else hi) + side * 0.055 * FS
            sil(sym(xv_, zs_, [(r * FS, q * FS) for r, q in ((0.022, 0.0), (0.026, 0.006), (0.040, 0.03), (0.046, 0.055), (0.040, 0.082), (0.022, 0.104), (0.017, 0.122), (0.026, 0.14))]))
            xk = xv_ + side * 0.07 * FS
            while abs(xk) < wi - 0.035:
                w_, h_ = rr.u(0.016, 0.032), rr.u(0.09, 0.135) * FS
                sil([(xk, zs_), (xk + side * w_, zs_), (xk + side * w_, zs_ + h_), (xk, zs_ + h_)])
                xk += side * (w_ + 0.002)

    zt_c, zb_c = H - PS, CZ + 0.004
    cr = rng.fork("curtain")
    if p.drawn > 0.03:
        shut = smoothstep(0.75, 1.0, p.drawn)
        wt = max(0.07, p.drawn * wi * 1.03)
        wtie = wt * lerp(0.36, 1.0, shut)
        wb = wt * lerp(0.6, 1.0, shut)
        vt = 0.4
        nf = max(2, int(round(wt / 0.045)))
        for sx in (-1, 1):
            ph = cr.u(0, 6.28)
            k2 = cr.u(0.7, 1.3)

            def cfn(u, v, sx=sx, ph=ph, k2=k2):
                if v > vt:
                    w = lerp(wtie, wt, smoothstep(0.0, 1.0, (v - vt) / (1 - vt)) ** 0.8)
                else:
                    w = lerp(wtie, wb, smoothstep(0.0, 1.0, (vt - v) / vt))
                uo = u if sx < 0 else 1 - u                      # 0 at the frame edge
                x = sx * (wi - uo * w)
                f = sin(2 * pi * nf * uo + ph + 1.2 * k2 * sin(2.6 * v + ph)) + 0.35 * sin(2 * pi * nf * 2.3 * uo + 2 * ph)
                return (x, -(D_CU + 0.0021 * f), zb_c + v * (zt_c - zb_c))

            cu = _sheet("curtain", cfn, nf * 3, 5)
            uv_planar(cu, (1, 0, 0), (0, 0, 1))
            smooth(cu, 80)
            assign(cu, cloth)
            if shut < 0.5:                                       # brass tie-back band (its faces 4 mm clear of the folds)
                tie = box("tieback", (wtie + 0.004, 0.0143, 0.018), loc=(sx * (wi - wtie / 2), -D_CU, zb_c + vt * (zt_c - zb_c)), bevel=0.003, segments=1)
                assign(tie, brass)
    br = rng.fork("blind")
    if p.blind > 0.02:
        # Holland roller blind on a hem batten that hangs a touch out of level. Lit, it is the lamp-shade material seen
        # through four smoky veils (a sixteenth of the shade's brightness); a layer of threads tapering down from the
        # roller makes it darker towards the top and gives the weave, with a few sagging weft slubs.
        zb_b = zt_c - p.blind * (zt_c - zb_c)
        bw_ = wi - 0.006
        skew = br.u(-0.004, 0.004)
        outline = [(-bw_, zb_b - skew), (bw_, zb_b + skew), (bw_, zt_c), (-bw_, zt_c)]
        if lit:                                              # lamp-shade stuff behind four smoky veils: 1/16 of the shade
            hots.append((D_BL, outline))
            for i in range(4):
                veils.append((D_BL + 0.0042 + 0.0014 * i, outline))
        else:
            assign(_cutouts("blind", [(D_BL, outline)]), linen)
        nt_ = max(8, int(round(2 * bw_ / 0.0065)))
        for i in range(nt_):
            xa_ = -bw_ + 2 * bw_ * i / nt_
            xb_ = xa_ + 2 * bw_ / nt_
            veils.append((D_BL + 0.0098, [(xa_, zt_c), (xb_, zt_c), ((xa_ + xb_) / 2, lerp(zt_c, zb_b + 0.02, br.u(0.55, 1.0)))]))
        for i in range(3 + int(8 * p.blind)):
            z_ = lerp(zb_b + 0.03, zt_c - 0.01, br.u())
            veils.append((D_BL + 0.0112, band(-bw_, z_, bw_, z_, 0.002, -br.u(0.002, 0.007), 6)))
        bat = box("blind_batten", (2 * bw_, 0.0054, 0.02), loc=(0, -(D_BL + 0.007), zb_b + 0.01), rot=(0, -degrees(atan2(skew, bw_)), 0), bevel=0.001)
        assign(bat, wood)
        uv_grain(bat, "X")
        ring_ = torus("blind_pull", 0.012, 0.0018, loc=(0, -(D_BL + 0.007), zb_b - 0.028), rot=(90, 0, 0), major_seg=9, minor_seg=4)
        cord = box("blind_cord", (0.002, 0.002, 0.03), loc=(0, -(D_BL + 0.007), zb_b - 0.008))
        assign(ring_, brass)
        assign(cord, brass)
    if veils:
        assign(_cutouts("room_veil", veils), mat("glass_clear", "#1a0f08", opacity=0.5, rough=1.0))
    if hots:                                                 # the lamp shade: the one hot spot of the window
        assign(_cutouts("room_hot", hots), mat("emissive", "#ffe0a8", glow=round(4.5 * gs, 3)))
    if lit and p.spill > 0.1:
        # the lamp shines from the room behind the glass (no glint on the panes): it washes the cill, the sill top and the
        # reveals; a weak second source gives the soft pool on the wall and the cobbles below (casts no shadow of the sill)
        light("spill", (lamp_x * 0.5, 0.3, lerp(rz0, lamp_z, 0.7)), color="#ffc37a", intensity=p.spill * 1.6, range=4.0, flicker=0.03)
        light("pool", (0, -0.95, -0.65), color="#ffc37a", intensity=p.spill * 0.8, range=4.0, flicker=0.03, shadow_near=3.0)

    # ================================================================ surround
    sr = rng.fork("surround")
    z_hood, joints = None, []
    EDGE = 0.028                       # the surround dies back to the wall: its outer arris is only this proud
    if plain:
        aw, D0 = 0.11, D_FR + 0.003
        pr0 = [(0.0, 0.0), (0.0, D0 - 0.004), (0.005, D0 + 0.001), (0.024, D0 + 0.001), (0.050, D0 - 0.022), (0.085, 0.034), (0.104, 0.016), (aw, 0.012), (aw, 0.0)]
        lpr = [(0.004, D0 - 0.012), (0.024, D0 - 0.008), (0.085, 0.024), (0.104, 0.008)]
    else:
        aw = (0.115 + 0.022 * W) * (1.0 if stone else 0.86)
        D0 = D_FR + 0.004
        pr0 = [(0.0, 0.0), (0.0, D0 - 0.005), (0.005, D0), (0.22 * aw, D0), (0.22 * aw + 0.002, D0 + 0.006), (0.38 * aw, D0 + 0.006)]
        pr_out = [(0.40 * aw, D0 + 0.008), (0.45 * aw, D0 + 0.012), (0.51 * aw, D0 + 0.018), (0.56 * aw, D0 + 0.020), (0.56 * aw, D0 + 0.023),
                  (0.62 * aw, D0 + 0.023), (0.64 * aw, D0 + 0.019), (0.70 * aw, D0 + 0.002), (0.76 * aw, D0 - 0.014), (0.80 * aw, D0 - 0.020),
                  (0.80 * aw, D0 - 0.026), (0.86 * aw, D0 - 0.027), (0.93 * aw, 0.046), (aw, EDGE)]
        pr0 = pr0 + pr_out + [(aw, 0.0)]
        lpr = [(0.03 * aw, D0 - 0.011), (0.36 * aw, D0 - 0.004), (0.59 * aw, D0 + 0.014), (0.80 * aw, D0 - 0.034), (0.98 * aw, 0.014)]

    def prof_arch(dd=0.0):
        return [(a, b + dd if b > 0 else b) for a, b in pr0]

    def apt(k, s, a, b):
        if k == "L":
            return (-hw - a, -b, s)
        if k == "R":
            return (hw + a, -b, s)
        if k == "TL":
            return (-hw - a, -b, H + a)
        if k == "TR":
            return (hw + a, -b, H + a)
        return (s, -b, H + a)

    def piece(name, path, dd=0.0, vj=(False, False)):
        """Loft the architrave profile along [(kind, s), ...]; vj = V-joint (arris knocked back) at start / end."""
        pr = prof_arch(dd)
        secs = [[apt(k, s, a, b) for a, b in pr] for k, s in path]
        lo = prof_arch(dd - 0.005)
        if vj[0]:
            k, s = path[0]
            secs = [[apt(k, s, a, b) for a, b in lo], [apt(k, s + 0.006, a, b) for a, b in pr]] + secs[1:]
        if vj[1]:
            k, s = path[-1]
            s2 = s + 0.006 if (k == "R" and len(path) > 2) else s - 0.006
            secs = secs[:-1] + [[apt(k, s2, a, b) for a, b in pr], [apt(k, s, a, b) for a, b in lo]]
        o = _loft(name, secs, along=not (vj[0] or vj[1]), min_prof=0.06 if (vj[0] or vj[1]) else 0.035)
        smooth(o, 38)
        return o

    def liner(path):
        liners.append(_liner("cast", [[apt(k, s, a, b) for a, b in lpr] for k, s in path]))

    def qliner(pts):
        liners.append(_liner("cast", [[pts[0], pts[1]], [pts[3], pts[2]]]))

    mortars = []
    if plain:
        for k in ("L", "R"):
            b = piece("reveal", [(k, 0.0), (k, H * 0.5), (k, H)])
            stones.append(b)
            liner([(k, 0.0), (k, H)])
        for b in stones:
            assign(b, smat)
            uv_box(b)
            paint(b, grime=mask_mul(height_ramp(0.0, 0.5), 0.3 + 0.3 * wr), wear=mask_mul(noise_mask(0.05, seed=p.seed + 6, lo=0.6, hi=0.8), 0.6 * wr),
                  special=mask_mul(height_ramp(0.0, 0.6), noise_mask(0.08, seed=p.seed + 8), 0.6 * wr))
        # oak lintel bearing on the wall either side, pegged; its face leans back so it has no boxy flank
        LL, LH = W + 2 * aw + 0.24, 0.14 + 0.03 * W
        DLn = D0 + 0.008
        oak = mat("wood_oak", "#5a4332")
        lin = box("lintel", (LL, DLn, LH), loc=(0, -DLn / 2, H + LH / 2), bevel=0.007)
        apply_transform(lin)
        densify(lin, 0.045)
        displace_noise(lin, 0.003, 0.25, seed=p.seed + 9, axis_mask=(0, 1, 1))

        def lean(co):
            t = clamp((co.z - H) / LH)
            e = clamp((abs(co.x) - (LL / 2 - 0.05)) / 0.05)
            k = (1.0 - 0.42 * t) * (1.0 - 0.45 * e * e)
            return Vector((co.x, min(co.y * k, 0.0), max(co.z, H)))

        transform_verts(lin, lean)
        assign(lin, oak)
        uv_grain(lin, "X")
        # rain off the wall above runs down the face in narrow streaks: never blobs
        lst = noise_mask(0.045, seed=p.seed + 3, lo=0.4, hi=0.8)
        paint(lin, grime=lambda co, n: np.maximum(np.clip(-n[:, 2], 0, 1) * 0.5,
                                                  lst(co * np.array([1.0, 0.0, 0.1]), n) * (0.2 + 0.3 * wr) * (0.4 + 0.6 * smoothstep_np(H, H + LH, co[:, 2]))),
              wear=lambda co, n: lst(co * np.array([1.0, 0.0, 0.1]) + 7.3, n) * (0.15 + 0.25 * wr), special=0.0)
        qliner([(-LL / 2 + 0.01, -0.02, H + 0.008), (LL / 2 - 0.01, -0.02, H + 0.008), (LL / 2 - 0.01, -0.02, H + LH - 0.008), (-LL / 2 + 0.01, -0.02, H + LH - 0.008)])
        assign(liners[-1], oak)
        peg = cylinder("peg", 0.011, 0.012, verts=8, bevel=0.002, segments=1)
        pm = [trs((sx * (hw + aw * 0.5 + dx), -DLn * (1.0 - 0.21) + 0.002, H + LH * 0.5 + sr.jitter(0, 0.015)), (90, 0, sr.u(0, 90)))
              for sx in (-1, 1) for dx in (-0.03, 0.045)]
        pegs = instances_to_mesh("pegs", peg, pm, keep_base=False)
        assign(pegs, oak)
        top_z = H + LH
        Lc = W + 2 * aw
    else:
        e = 0.10 + 0.04 * sr.u()         # the head stone turns down the jamb by this much
        kb = 0.05 + 0.022 * W
        g = 0.007 if stone else 0.0      # mortar bed between the stones
        arch = []
        if stone:
            for k in ("L", "R"):
                n = max(2, int(round((H - e) / (0.25 + 0.13 * H))))
                cuts = [0.0] + [(H - e) * (i + sr.jitter(0.0, 0.16)) / n for i in range(1, n)] + [H - e]
                joints += [(-1 if k == "L" else 1, c) for c in cuts[1:]]
                for i in range(n):
                    a0 = cuts[i] + (g / 2 if i else 0.0)
                    arch.append(piece("jamb_stone", [(k, a0), (k, cuts[i + 1] - g / 2)], sr.jitter(0.0, 0.002), vj=(i > 0, True)))
                    mortars.append(_loft("joint", [[apt(k, s, a, b) for a, b in prof_arch(-0.0035)] for s in (cuts[i + 1] - g / 2 - 0.001, cuts[i + 1] + g / 2 + 0.001)],
                                         refine=False, cap=False))
            if p.head == "keystone":
                arch.append(piece("head_stone", [("L", H - e + g / 2), ("TL", 0), ("T", -kb + 0.012)], sr.jitter(0.0, 0.002), vj=(True, False)))
                arch.append(piece("head_stone", [("T", kb - 0.012), ("TR", 0), ("R", H - e + g / 2)], sr.jitter(0.0, 0.002), vj=(False, True)))
            else:
                arch.append(piece("head_stone", [("L", H - e + g / 2), ("TL", 0), ("TR", 0), ("R", H - e + g / 2)], 0.001, vj=(True, True)))
        else:
            if p.head == "keystone":
                arch.append(piece("arch_l", [("L", 0.0), ("TL", 0), ("T", -kb + 0.012)]))
                arch.append(piece("arch_r", [("T", kb - 0.012), ("TR", 0), ("R", 0.0)]))
            else:
                arch.append(piece("arch", [("L", 0.0), ("TL", 0), ("TR", 0), ("R", 0.0)]))
            for o in arch:
                bevel(o, 0.0016, segments=1, angle=40)
        liner([("L", 0.0), ("TL", 0), ("TR", 0), ("R", 0.0)])
        stones += arch
        top_z = H + aw
        Lc = W + 2 * aw
        if p.head == "keystone":
            # keystone = the architrave mouldings broken forward round a tapering, prow-faced block
            BF, RG = 0.012, 0.007
            kt = kb * 1.42
            body = D0 + 0.006 + BF
            kp = [(-0.026, body - 0.011), (-0.017, body), (0.38 * aw, body)] + [(a, b + BF) for a, b in pr_out] + [(aw + 0.014, 0.016)]
            secs = []
            for a, b in kp:
                w = lerp(kb, kt, clamp((a + 0.026) / (aw + 0.04)))
                z = H + a
                secs.append([(-w, 0.0, z), (-w, -(b - 0.005), z), (-w + 0.006, -b, z), (0.0, -(b + RG), z), (w - 0.006, -b, z), (w, -(b - 0.005), z), (w, 0.0, z)])
            key = _loft("keystone", secs, min_prof=0.05)
            smooth(key, 28)
            stones.append(key)
            qliner([(-kb - 0.002, -(body - 0.012), H - 0.02), (kb + 0.002, -(body - 0.012), H - 0.02), (kt - 0.006, -0.03, H + aw), (-kt + 0.006, -0.03, H + aw)])
            top_z = H + aw + 0.014
        else:
            # entablature: pulvinated frieze on the architrave, consoles flanking it, cornice returned to the wall
            K, KA = 0.74, 0.52                 # cornice scale: height, projection
            fh = 0.10 + 0.02 * W
            zf = H + 0.86 * aw
            zk = zf + fh
            Dp = D0 - 0.004
            Db = D0 + 0.012
            z_hood = zk
            cw = 0.062 + 0.02 * W
            fz = [(0.0, zf), (-Dp, zf), (-Dp, zf + 0.010), (-(Dp + 0.0065), zf + 0.2 * fh), (-(Dp + 0.0115), zf + 0.38 * fh), (-(Dp + 0.013), zf + 0.5 * fh),
                  (-(Dp + 0.0115), zf + 0.62 * fh), (-(Dp + 0.0065), zf + 0.8 * fh), (-Dp, zf + fh - 0.010), (-Dp, zf + fh), (0.0, zf + fh)]
            if stone:
                xj = sr.jitter(0.0, 0.12 * W)
                for xa_, xb_ in ((-Lc / 2, xj - g / 2), (xj + g / 2, Lc / 2)):
                    stones.append(_px("frieze", fz, xa_, xb_, min_prof=0.05))
                mortars.append(_px("joint", [(y + 0.003 if y < 0 else y, z) for y, z in fz], xj - g / 2 - 0.001, xj + g / 2 + 0.001, refine=False, cap=False))
            else:
                stones.append(_px("frieze", fz, -Lc / 2, Lc / 2, min_prof=0.05))
            for o in stones[-2:]:
                if o.name.startswith("frieze"):
                    smooth(o, 38)
            qliner([(-Lc / 2, -(Dp - 0.01), zf), (Lc / 2, -(Dp - 0.01), zf), (Lc / 2, -(Dp - 0.01), zk), (-Lc / 2, -(Dp - 0.01), zk)])
            Lcor = Lc + 2 * cw
            cpr = [(0.0, 0.0), (0.008, 0.0), (0.010, 0.010), (0.016, 0.018), (0.028, 0.027), (0.040, 0.031), (0.046, 0.031), (0.046, 0.036),
                   (0.070, 0.038), (0.074, 0.034), (0.084, 0.034), (0.086, 0.040), (0.086, 0.062), (0.091, 0.066), (0.098, 0.076), (0.107, 0.083),
                   (0.112, 0.085), (0.112, 0.094), (0.0, 0.104)]
            cpr = [(a * KA, q * K) for a, q in cpr]
            hcor, pc = 0.104 * K, 0.112 * KA
            cor = _returned("cornice", cpr, Lcor, Db, zk)
            smooth(cor, 38)
            corc = box("cornice_core", (Lcor, Db, hcor), loc=(0, -Db / 2, zk + hcor / 2))
            stones += [cor, corc]
            hc = fh + 0.10 + 0.03 * H
            for sx in (-1, 1):
                stones.append(_console("hood_console", sx * (Lc / 2 + cw / 2), cw, zk + 0.001, hc, Db + 0.012))
                xc_ = sx * (Lc / 2 + cw / 2)
                qliner([(xc_ - cw / 2 + 0.004, -0.014, zk - hc + 0.01), (xc_ + cw / 2 - 0.004, -0.014, zk - hc + 0.01),
                        (xc_ + cw / 2 - 0.004, -0.05, zk - 0.01), (xc_ - cw / 2 + 0.004, -0.05, zk - 0.01)])
            top_z = zk + hcor
            if p.head == "pediment":
                Le = Lcor / 2 + pc
                pitch = radians(23.0)
                rise = Le * tan(pitch)
                zp = zk + hcor
                # tympanum as a grid (not one big triangle): interior vertices keep the baked AO in the corners only
                Lt, dt_ = Le - 0.06, Db - 0.012
                nt = 2 * max(3, int(round(Lt / 0.13)))
                tsec = []
                for i in range(nt + 1):
                    x = -Lt + 2 * Lt * i / nt
                    ht = max(0.004, (Lt - abs(x)) * tan(pitch))
                    tsec.append([(x, 0.0, zp)] + [(x, -dt_, zp + ht * f_) for f_ in (0.0, 0.22, 0.5, 0.78, 1.0)] + [(x, 0.0, zp + ht)])
                tym = _loft("tympanum", tsec, refine=False)
                stones.append(tym)
                qliner([(-Le + 0.09, -(Db - 0.03), zp), (Le - 0.09, -(Db - 0.03), zp), (0.01, -(Db - 0.03), zp + (Le - 0.09) * tan(pitch)),
                        (-0.01, -(Db - 0.03), zp + (Le - 0.09) * tan(pitch))])
                rpr = [(-Db, 0.0)] + [(a * KA, q * K) for a, q in [(0.0, 0.0), (0.028, 0.004), (0.046, 0.010), (0.050, 0.016), (0.086, 0.018), (0.086, 0.038),
                                                                 (0.094, 0.046), (0.107, 0.056), (0.112, 0.058), (0.112, 0.066)]] + [(-Db, 0.074 * K)]
                kc = 1.0 / cos(pitch)
                z_off = (Le - 0.06) * tan(pitch) - rise           # rake soffit sits on the tympanum edge
                rk = _loft("rake", [[(x, -(Db + b) if b > -Db else 0.0, zp + z_off + rise * (1 - abs(x) / Le) + q * kc) for b, q in rpr]
                                    for x in (-Le, 0.0, Le)])
                smooth(rk, 38)
                stones.append(rk)
                top_z = zp + rise + 0.074 * K * kc
            elif not stone:
                lead_top = box("lead_cap", (Lcor + 2 * pc + 0.008, Db + pc + 0.004, 0.004), loc=(0, -(Db + pc + 0.004) / 2, zk + hcor + 0.001), bevel=0.0012)
                apply_transform(lead_top)
                transform_verts(lead_top, lambda co: Vector((co.x, co.y, co.z + 0.012 * (1 + co.y / (Db + pc)) - 0.012)))
                assign(lead_top, mat("lead"))
                paint(lead_top, grime=mask_mul(noise_mask(0.15, seed=p.seed + 13), 0.6))

    # ---- stone sill: weathered top between stooled ends, rounded nosing with a throat (drip groove), bed mould
    dn = D0 + (0.040 if not plain else 0.036)                    # nosing projection
    SB = CN - 0.030                                                # flat bed the timber cill is let into
    th = 0.082 if not plain else 0.064
    Ls = Lc + (0.07 if not plain else 0.06)
    x_st = hw - 0.005

    def ssec(x, s, jn=0.0, chip=0.0):
        zt = -0.0005 - 0.0195 * s
        pts = [(0.0, 0.0), (SB, 0.0), (dn - 0.011 - chip + jn, zt), (dn - 0.003 - 0.6 * chip + jn, zt - 0.005 - 0.5 * chip), (dn + jn, zt - 0.013 - 0.4 * chip),
               (dn + jn, -th * 0.64), (dn - 0.004 + jn, -th * 0.70), (dn - 0.014, -th * 0.70), (dn - 0.017, -th * 0.60), (dn - 0.025, -th * 0.60),
               (dn - 0.028, -th * 0.70), (dn - 0.040, -th * 0.70), (dn - 0.044, -th * 0.78), (dn - 0.052, -th * 0.91), (dn - 0.058, -th * 0.97),
               (dn - 0.060, -th), (0.0, -th)]
        return [(x, -d, z) for d, z in pts]

    nmid = max(3, int(round(2 * x_st / 0.09)))
    chips = set(sr.pick(list(range(1, nmid))) for _ in range(1 + int(3 * wr)))
    ssecs = [ssec(-Ls / 2, 0.0), ssec(-Ls / 2 + 0.03, 0.0, sr.jitter(0, 0.0015)), ssec(-x_st - 0.004, 0.0), ssec(-x_st + 0.006, 1.0)]
    for i in range(1, nmid):
        xm = -x_st + 2 * x_st * (i + sr.jitter(0, 0.2)) / nmid
        ch = sr.u(0.006, 0.013) if i in chips else 0.0
        ssecs.append(ssec(xm, 1.0, sr.jitter(0.0, 0.002) - 0.3 * ch, ch))
    ssecs += [ssec(x_st - 0.006, 1.0), ssec(x_st + 0.004, 0.0), ssec(Ls / 2 - 0.03, 0.0, sr.jitter(0, 0.0015)), ssec(Ls / 2, 0.0)]
    sill = _loft("sill", ssecs, refine=False)
    smooth(sill, 36)
    sills = [sill]
    bot_z = -th
    hb = 0.0
    if not plain:
        hb = 0.2 + 0.03 * H
        cws = min(0.125, aw * 0.82)
        for sx in (-1, 1):
            sills.append(_console("sill_console", sx * (hw + aw * 0.5), cws, -th + 0.002, hb, dn - 0.058))
            sills.append(box("console_rib", (cws * 0.3, 0.006, hb * 0.34), loc=(sx * (hw + aw * 0.5), -(dn - 0.058) * 0.2 - 0.002, -th - hb * 0.74), bevel=0.002, segments=1))
        za0, za1 = -th - hb + 0.03, -th + 0.004
        # water runs off the sill ends and down the apron: the apron gets an edge loop either side of every run
        runs = [(sx * (hw + aw * 0.5 + o_), sr.u(0.012, 0.02), sr.u(0.7, 1.0)) for sx in (-1, 1) for o_ in (-0.09, sr.u(0.075, 0.085))]
        runs += [(sr.u(-hw * 0.8, hw * 0.8), sr.u(0.012, 0.02), sr.u(0.35, 0.6)) for _ in range(2)]
        xe = Ls / 2 - 0.015
        nb = max(2, int(round(2 * xe / 0.16)))
        xs_ = [-xe + 2 * xe * i / nb for i in range(nb + 1)]
        for xk, wk, ak in runs:
            xs_ = [q for q in xs_ if abs(q - xk) > 2.2 * wk or abs(q) > xe - 1e-6] + [xk - 1.5 * wk, xk, xk + 1.5 * wk]
        xs_ = sorted(q for q in xs_ if -xe <= q <= xe)
        apr = [(0.0, za1), (-0.022, za1)] + [(-0.022, lerp(za1, za0 + 0.005, t)) for t in (0.25, 0.5, 0.75, 1.0)] + [(-0.017, za0), (0.0, za0)]
        ap = _loft("apron", [[(x, y, z) for y, z in apr] for x in xs_], along=False, min_prof=0.5)
        ap2 = box("apron_panel", (W - 0.10, 0.012, hb - 0.10), loc=(0, -0.027, -th - (hb - 0.03) / 2), bevel=0.005, segments=1)
        sills += [ap, ap2]
        bot_z = -th - hb
    for o in sills:
        assign(o, sill_mat)
        uv_box(o)
        if kind == "timber":
            uv_grain(o, "X")
    for o in stones:
        if not plain:
            assign(o, smat)
            if kind == "timber":
                uv_grain(o, "Z" if o.name.startswith(("arch", "hood")) else "X")
            else:
                uv_box(o)
    for o in mortars:                  # lime-putty joints: the stone's own material, rubbed pale and clean
        assign(o, smat)
        uv_box(o)
        paint(o, grime=0.0, wear=1.0, special=0.0)
    for o in liners:
        if not o.data.materials:
            assign(o, smat)
        uv_box(o)

    if plain:
        runs = []

    def run_mask(co, n):
        m = np.zeros(len(co))
        for xk, wk, ak in runs:
            m = np.maximum(m, ak * np.exp(-((co[:, 0] - xk) / wk) ** 2))
        fade = np.clip(1.25 - (-th - co[:, 2]) / max(0.05, hb * 1.05), 0.0, 1.0)
        return m * fade * (co[:, 2] < -th * 0.6)

    # Soot and damp sit only where the rain does not wash: under the hood, under the sill, at the head of the reveal and
    # in the bed joints - and they hang in fine vertical streaks, not blotches. The open faces stay pale.
    fine = noise_mask(0.045, seed=p.seed + 40, lo=0.35, hi=0.8)

    def streaks(co, n):
        return fine(co * np.array([1.0, 0.3, 0.14]), n)

    def recess(co, n):
        z = co[:, 2]
        m = 0.8 * smoothstep_np(-th - 0.13, -th * 0.72, z) * (z < -th * 0.6)
        if z_hood is not None:
            m = np.maximum(m, 0.8 * smoothstep_np(z_hood - 0.15, z_hood - 0.005, z) * (z < z_hood + 0.004))
        m = np.maximum(m, 0.4 * smoothstep_np(H - 0.10, H + 0.005, z) * (z < H + 0.01) * (np.abs(co[:, 0]) < hw + 0.012))
        for sg, zj in joints:
            m = np.maximum(m, 0.55 * np.exp(-((z - zj) / 0.011) ** 2) * (co[:, 0] * sg > hw - 0.002) * (z < H + 0.02))
        return m

    for o in (stones if not plain else []) + sills:
        if kind == "timber":
            paint(o, wear=0.0, special=0.03,
                  grime=lambda co, n: np.maximum(run_mask(co, n) * (0.35 + 0.4 * wr),
                                                 np.maximum(recess(co, n) * 0.7, np.clip(1.0 - (co[:, 2] - bot_z) / 0.4, 0, 1) * (0.2 + 0.25 * wr)) * (0.55 + 0.45 * streaks(co, n))))
        elif o.name.startswith("tympanum"):
            paint(o, grime=mask_mul(streaks, 0.1))               # sheltered under the raking cornice: stays clean
        else:
            paint(o, grime=lambda co, n: np.maximum(run_mask(co, n) * (0.5 + 0.4 * wr),
                                                    recess(co, n) * (0.45 + 0.4 * wr) * (0.55 + 0.45 * streaks(co, n)) + streaks(co, n) * (0.03 + 0.1 * wr)),
                  special=mask_mul(facing((0, 0, 1), 2.0), noise_mask(0.12, seed=p.seed + 17, lo=0.3, hi=0.7), 0.3 + 0.7 * wr),
                  wear=mask_mul(noise_mask(0.05, seed=p.seed + 19, lo=0.55, hi=0.8), 0.3 * wr))

    # ================================================================ wrought-iron sill guard
    if p.guard:
        gh = p.guard_height
        iron = mat("iron_wrought")
        xg = hw + aw * 0.5
        dg_ = dn - 0.03
        dw = (D0 + 0.004) if not plain else D0 - 0.015
        zb, zt = 0.045, gh
        gz0 = -0.012                                               # leaded into the stooled ends
        gz1 = -0.028                                               # ... and into the weathering
        for z, wd in ((zb, 0.022), (zt, 0.028)):
            irons.append(box("guard_rail", (2 * xg + wd, wd, 0.008), loc=(0, -dg_, z), bevel=0.0015, segments=1))
            for sx in (-1, 1):
                irons.append(box("guard_return", (wd, dg_ - dw, 0.008), loc=(sx * xg, -(dg_ + dw) / 2, z), bevel=0.0015, segments=1))
                irons.append(cylinder("guard_rose", 0.017, 0.006, loc=(sx * xg, -(dw + 0.003), z), rot=(90, 0, 0), verts=8))
        for sx in (-1, 1):
            irons.append(box("guard_post", (0.015, 0.015, zt - gz0 + 0.012), loc=(sx * xg, -dg_, (zt + gz0 + 0.012) / 2), bevel=0.002, segments=1))
            irons.append(lathe("guard_finial", [(0.0045, 0.0), (0.009, 0.004), (0.005, 0.009), (0.011, 0.016), (0.015, 0.026), (0.011, 0.037), (0.0, 0.043)],
                               segments=8, loc=(sx * xg, -dg_, zt + 0.008)))
        span = 2 * xg - 0.015
        hsc = zt - zb - 0.008 - 0.006
        sc = _s_scroll(hsc, 1.05, 14)
        ws = max(q[0] for q in sc) - min(q[0] for q in sc)
        nmod = max(1, int(floor(span / (ws + 0.06))))
        pitch = span / nmod
        zmid = (zb + zt) / 2
        gr2 = rng.fork("guard")
        prof = [(-0.0026, -0.006), (0.0026, -0.006), (0.0026, 0.006), (-0.0026, 0.006)]
        for i in range(nmod):
            xc = -span / 2 + pitch * (i + 0.5)
            flip = 1 if i % 2 == 0 else -1
            path = [(xc + flip * q[0], -dg_, zmid + q[1]) for q in sc]
            s = sweep("guard_scroll", path, prof, up=(0, 1, 0), smooth_path=False, scale_fn=lambda t: 0.62 + 0.38 * min(1.0, 4 * t, 4 * (1 - t)))
            place(s, rot=(0, gr2.jitter(0.0, 1.2), 0))
            irons.append(s)
            ztop = max(q[1] for q in sc)
            xtop = [q[0] for q in sc if q[1] == ztop][0]
            for sg in (-1, 1):
                irons.append(box("guard_collar", (0.016, 0.017, 0.007), loc=(xc + flip * sg * xtop, -dg_, zmid + sg * (ztop + 0.002))))
            if i:
                xv = -span / 2 + pitch * i
                irons.append(box("guard_bar", (0.0095, 0.0095, zt - zb), loc=(xv, -dg_, zmid), rot=(0, 0, 45)))
                irons.append(box("guard_foot", (0.011, 0.011, zb - gz1), loc=(xv, -dg_, (zb + gz1) / 2), rot=(0, 0, 45)))
        for o in irons:
            assign(o, iron)
            paint(o, special=mask_mul(height_ramp(0.0, gh * 1.3), noise_mask(0.05, seed=p.seed + 30, lo=0.3, hi=0.7), 0.5 + 0.5 * wr),
                  wear=mask_mul(height_ramp(gh + 0.03, gh - 0.03), 0.6))

    # ================================================================ sockets
    socket("planter", "attach", (0, -(dn + 0.02), 0.0), role="planter")
    socket("shutter_l", "attach", (-hw - aw, 0, H / 2), role="shutter")
    socket("shutter_r", "attach", (hw + aw, 0, H / 2), role="shutter")


def _flaky(name, yz, x0, x1, nfl, rng, si, t_lo, t_hi, t_rest, t_end):
    """Extrude a YZ profile along X through skewed sections: every profile vertex is jittered along X on its own, and
    vertex `si` (it lies on the line between its two neighbours) slides along that line from section to section. The
    solid is unchanged, but the faces on one side of the sliding vertex get a ragged outline - given another material
    they are crisp-edged paint flakes. Returns (object, [(xa, xb), ...]: the stretches meant to flake)."""
    slot = (x1 - x0 - 0.06) / max(1, nfl)
    xs, fl, ts = [x0], [], [t_rest]
    for i in range(nfl):
        ws = [rng.u(0.012, 0.034) for _ in range(rng.pick((2, 3, 3, 4, 5)))]
        while len(ws) > 2 and sum(ws) > slot - 0.03:
            ws.pop()
        if sum(ws) > slot - 0.03:
            continue
        a = x0 + 0.03 + slot * i + 0.012 + rng.u() * (slot - 0.024 - sum(ws))
        fl.append((a, a + sum(ws)))
        xs.append(a)
        ts.append(t_end)
        for q, w in enumerate(ws):
            a += w
            xs.append(a)
            ts.append(t_end if q == len(ws) - 1 else rng.u(t_lo, t_hi))
    xs.append(x1)
    ts.append(t_rest)
    full, tf = [xs[0]], [ts[0]]
    for a, b, tb in zip(xs[:-1], xs[1:], ts[1:]):
        k = int((b - a) // 0.2)
        full += [a + (b - a) * (q + 1) / (k + 1) for q in range(k)] + [b]
        tf += [t_rest] * k + [tb]
    (ya, za), (yb, zb) = yz[si - 1], yz[si + 1]
    secs = []
    for i, (x, t) in enumerate(zip(full, tf)):
        pr = [(lerp(ya, yb, t), lerp(za, zb, t)) if q == si else pt for q, pt in enumerate(yz)]
        amp = 0.3 * min(x - full[i - 1], full[i + 1] - x, 0.03) if 0 < i < len(full) - 1 else 0.0
        secs.append([(x + rng.u(-amp, amp), y, z) for y, z in pr])
    return _loft(name, secs, along=False, min_prof=0.2), fl


def _cutouts(name, items):
    """Flat polygons [(depth, [(x, z), ...]), ...] facing the street, stored back to front (the blending order of
    stacked veils)."""
    verts, faces = [], []
    for d, pts in sorted(items, key=lambda q: q[0]):
        ar = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
        if ar < 0:
            pts = pts[::-1]
        faces.append(tuple(range(len(verts), len(verts) + len(pts))))
        verts.extend((x, -d, z) for x, z in pts)
    o = mesh_object(name, verts, faces)
    uv_planar(o, (1, 0, 0), (0, 0, 1))
    return o


def _liner(name, secs):
    """Hidden skin just under the face of a solid that sits against the wall, facing the wall: the shadow pass
    draws back faces, and the solid's own back face lies in the wall plane (no shadow), so this is its caster."""
    n, m = len(secs[0]), len(secs)
    verts = [tuple(v) for s in secs for v in s]
    faces = [(i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j) for i in range(m - 1) for j in range(n - 1)]
    o = mesh_object(name, verts, faces)
    o.data.update()
    for poly in o.data.polygons:
        if poly.normal.y < 0:
            poly.flip()
    o.data.update()
    return o


# numpy mask helpers (paint() hands world-space (N, 3) arrays)
def smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 if e1 != e0 else 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)
