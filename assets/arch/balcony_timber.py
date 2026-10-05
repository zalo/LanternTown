"""Jettied timber gallery balcony. Heavy stop-chamfered cantilever beams with carved noses bear on the
wall and on pierced, cusped spandrel brackets tenoned into wall pieces; a cogged fascia bressummer,
ledger and joists carry a deck of loose sawn planks. Mortised posts hold a moulded handrail and a
bottom rail with turned / flat-sawn splat / St Andrew's cross balusters. Optionally the posts run up
to an eaves plate with arched braces carrying a lean-to pentice roof: rafters with shaped tails,
sarking boards, hand-made clay tiles laid half-bond, bargeboards and a dressed lead wall flashing.
Wrought-iron scroll brackets on the end posts hang small glazed lanterns.
Origin: centre of the deck's back edge, deck top = z 0, wall plane y = 0.
"""
from tk import *
import math
import numpy as np
from mathutils import Matrix, Vector

META = {"name": "Timber gallery balcony", "class": "component", "mount": "wall", "mount_height": 3.0,
        "tri_budget": 22000, "focus": [0.0, -1.3, 1.2]}
PARAMS = {
    "width": {"default": 3.2, "min": 1.6, "max": 6.0, "desc": "overall width of the deck along the wall (m)"},
    "projection": {"default": 1.15, "min": 0.7, "max": 1.8, "desc": "depth of the deck from the wall plane (m)"},
    "balusters": {"default": "turned", "choices": ["turned", "splat", "cross"], "desc": "balustrade infill: turned, flat-sawn pierced splats, St Andrew's crosses"},
    "roof": {"default": True, "desc": "posts carry a lean-to tiled pentice roof"},
    "wood": {"default": "#7b5a3c", "desc": "wood tone (stain colour of the oak frame)"},
    "rail_height": {"default": 1.0, "min": 0.9, "max": 1.15, "desc": "handrail height above the deck (m)"},
    "bracket": {"default": "cusped", "choices": ["cusped", "arched"], "desc": "carving of the spandrel brackets and roof braces"},
    "bracket_drop": {"default": 0.7, "min": 0.4, "max": 1.0, "desc": "height of the brackets below the beams (m)"},
    "roof_pitch": {"default": 33.0, "min": 25.0, "max": 45.0, "desc": "pentice roof pitch (deg)"},
    "post_height": {"default": 2.15, "min": 1.95, "max": 2.6, "desc": "deck to eaves plate (m), roofed gallery only"},
    "rail_finish": {"default": "oak", "choices": ["oak", "painted"], "desc": "balustrade left in oiled oak or painted"},
    "rail_paint": {"default": "#34483a", "desc": "paint colour of a painted balustrade"},
    "lanterns": {"default": 1, "min": 0, "max": 2, "desc": "lanterns on scrolled iron brackets on the end posts"},
    "intensity": {"default": 14.0, "min": 5.0, "max": 60.0, "desc": "light per lantern (cd)"},
    "age": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "weathering: plank warp, slipped tiles, moss, damp"},
}
VARIANTS = {
    "open_splat": {"roof": False, "balusters": "splat", "width": 4.2, "projection": 1.0, "rail_finish": "painted",
                   "rail_paint": "#3d5a48", "wood": "#5d432e", "bracket": "arched", "lanterns": 1, "seed": 4},
    "cross_long": {"balusters": "cross", "width": 5.0, "projection": 1.4, "wood": "#4a3626", "bracket_drop": 0.9,
                   "roof_pitch": 38.0, "lanterns": 2, "age": 0.6, "seed": 11},
    "juliet_small": {"roof": False, "width": 1.9, "projection": 0.8, "wood": "#a07a52", "bracket_drop": 0.5,
                     "rail_height": 1.05, "lanterns": 1, "age": 0.15, "seed": 23},
    "painted_tall": {"balusters": "splat", "width": 2.6, "projection": 1.3, "post_height": 2.45, "roof_pitch": 42.0,
                     "rail_finish": "painted", "rail_paint": "#6a2a22", "bracket": "arched", "wood": "#6a4a30",
                     "age": 0.85, "seed": 31},
}

PT = 0.13          # post section
DECK_T = 0.032     # deck plank thickness
JOIST_H = 0.11
BEAM_H = 0.20
BEAM_W = 0.16


def _shade(h, k):
    h = h.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda v: max(0, min(255, int(round(v * k))))
    return "#%02x%02x%02x" % (f(r), f(g), f(b))


def beam(name, p0, p1, w, h, up=(0, 0, 1), cham=0.0, edges=(1, 1, 1, 1), runs=None, ease=0.004, seg=0.6,
         shape=None, shape_ts=(), stop=0.05):
    """Sawn timber member from p0 to p1 (w across, h along `up`). The arrises flagged in `edges`
    (order: -side/low, +side/low, +side/high, -side/high) carry a chamfer of `cham` inside the `runs`
    (start, end in m along the member) that dies out in a run-out stop; the rest is just eased.
    shape(t) -> (dlo, dhi) moves the low / high face (carved ends, tapers)."""
    p0, p1 = Vector(p0), Vector(p1)
    t = p1 - p0
    L = t.length
    t = t / L
    side = Vector(up).cross(t)
    if side.length < 1e-4:
        side = Vector((1, 0, 0)).cross(t)
    side.normalize()
    upv = t.cross(side).normalized()
    ts = {0.0, L}
    if seg:
        n = max(1, int(math.ceil(L / seg)))
        ts.update(L * i / n for i in range(1, n))
    if cham > 0 and runs is None:
        runs = [(-1.0, L + 1.0)]
    for (a, b) in (runs or []):
        for v in (a, a + stop, b - stop, b):
            if 0 < v < L:
                ts.add(v)
    for v in shape_ts:
        if 0 < v < L:
            ts.add(v)
    tl = []
    for v in sorted(ts):
        if not tl or v - tl[-1] > 1e-4:
            tl.append(v)
    verts = []
    for tt in tl:
        c = 0.0
        for (a, b) in (runs or []):
            c = max(c, cham * clamp(min((tt - a) / stop, (b - tt) / stop)))
        lo, hi = -h / 2, h / 2
        if shape:
            dl, dh = shape(tt)
            lo, hi = lo + dl, hi + dh
        hw = w / 2
        lim = 0.45 * min(w, hi - lo)
        cs = [min(lim, max(ease, c if e else 0.0)) for e in edges]
        ring = [(-hw, lo + cs[0]), (-hw + cs[0], lo), (hw - cs[1], lo), (hw, lo + cs[1]),
                (hw, hi - cs[2]), (hw - cs[2], hi), (-hw + cs[3], hi), (-hw, hi - cs[3])]
        for (a, b) in ring:
            verts.append(tuple(p0 + t * tt + side * a + upv * b))
    nr = len(tl)
    faces = []
    for r in range(nr - 1):
        for j in range(8):
            k = (j + 1) % 8
            faces.append((r * 8 + j, r * 8 + k, (r + 1) * 8 + k, (r + 1) * 8 + j))
    faces.append(tuple(range(7, -1, -1)))
    faces.append(tuple((nr - 1) * 8 + j for j in range(8)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, 30)
    uv_grain(o, axis=(t.x, t.y, t.z))
    return o


def prism(name, pts, a0, a1, plane="YZ"):
    """Extrude a 2D polygon lying in the YZ (or XZ) plane between a0 and a1 on the third axis."""
    verts = []
    for (u, v) in pts:
        verts += [(a0, u, v), (a1, u, v)] if plane == "YZ" else [(u, a0, v), (u, a1, v)]
    n = len(pts)
    faces = [(2 * i, 2 * ((i + 1) % n), 2 * ((i + 1) % n) + 1, 2 * i + 1) for i in range(n)]
    faces.append(tuple(2 * i for i in range(n))[::-1])
    faces.append(tuple(2 * i + 1 for i in range(n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def spandrel_curve(L, H, style, nose=0.045, n=9):
    """Free edge of a solid spandrel bracket: from the nose (L, nose) to the foot (0, H); u = out, v = drop."""
    pts = [(L, 0.0)]
    if style == "cusped":
        um, vm = 0.55 * L, 0.5 * H + 0.2 * nose
        for i in range(n):
            a = radians(90.0 * i / (n - 1))
            pts.append((L - (L - um) * sin(a), vm - (vm - nose) * cos(a)))
        pts.append((um + 0.012, vm + 0.018))                    # blunt little cusp tip
        for i in range(n):
            a = radians(90.0 * i / (n - 1))
            pts.append((um - um * sin(a), H - (H - vm - 0.03) * cos(a)))
    else:
        for i in range(n + 4):
            a = radians(90.0 * i / (n + 3))
            pts.append((L - L * sin(a) ** 0.9, H - (H - nose) * cos(a) ** 0.9))
    return pts


def build(p, rng):
    W, D, age = p.width, p.projection, p.age
    RH = p.rail_height
    oak = mat("wood_oak", p.wood)
    endg = mat("wood_endgrain", _shade(p.wood, 1.15))
    railm = mat("wood_painted", p.rail_paint) if p.rail_finish == "painted" else mat("wood_oak", _shade(p.wood, 1.3))
    deckm = mat("wood_weathered", "#b6a28a")
    iron = mat("iron_wrought")
    frame, rails, deck, ironw, pegs_m = [], [], [], [], []

    sp = D - 0.095                    # post / fascia centre line, distance from the wall
    yF = -(D - 0.02)                  # fascia front face
    zD = -DECK_T                      # underside of the deck
    zJ = zD - JOIST_H                 # top of the cantilever beams
    zB = zJ - BEAM_H                  # underside of the cantilever beams
    nb = max(1, int(round(W / 1.65)))
    xs = [-(W / 2 - 0.08) + i * (W - 0.16) / nb for i in range(nb + 1)]
    nl = int(p.lanterns)
    lan_idx = [] if nl == 0 else ([nb // 2] if nb >= 2 else [nb] if p.seed % 2 == 0 else [0]) if nl == 1 else [0, nb]
    peg = cylinder("peg", 0.0085, 0.014, verts=5)

    # ---------------------------------------------------------------- cantilever beams + brackets
    Lb_tot = D + 0.12
    tn = D - 0.02
    nose_tab = [(0.0, 0.0), (0.08, 0.02), (0.17, 0.02), (0.3, 0.032), (0.5, 0.06), (0.7, 0.092), (0.86, 0.108), (1.0, 0.112)]

    def nose(t):
        u = (t - tn) / 0.14
        if u <= 0:
            return (0.0, 0.0)
        for (u0, v0), (u1, v1) in zip(nose_tab, nose_tab[1:]):
            if u <= u1:
                return (lerp(v0, v1, (u - u0) / (u1 - u0)), 0.0)
        return (nose_tab[-1][1], 0.0)

    H = p.bracket_drop
    Lk = max(0.25, min(D - 0.32, H * 1.15))
    rb = rng.fork("frame")
    for i, x in enumerate(xs):
        sag = rb.u(-0.004, 0.0) * (1 + age)
        b = beam(f"jetty_beam{i}", (x, -0.003, zJ - BEAM_H / 2), (x, -Lb_tot, zJ - BEAM_H / 2 + sag), BEAM_W, BEAM_H,
                 cham=0.024, edges=(1, 1, 0, 0), runs=[(0.12, tn - 0.05)], shape=nose,
                 shape_ts=[tn + 0.14 * u for (u, _) in nose_tab], seg=0.4)
        assign(b, oak)
        assign_faces(b, endg, lambda c, n: n.y < -0.95)
        frame.append(b)
        # wall piece with a tapered foot, spandrel tenoned between it and the beam
        Lw = H + 0.17
        wp = beam(f"wall_piece{i}", (x, -0.048, zB), (x, -0.048, zB - Lw), PT, 0.09, up=(0, 1, 0), cham=0.016,
                  edges=(1, 1, 0, 0), runs=[(H + 0.01, Lw - 0.13)], seg=0.4,
                  shape=lambda t, Lw=Lw: (0.066 * clamp((t - (Lw - 0.12)) / 0.12) ** 1.6, 0.0),
                  shape_ts=[Lw - 0.12, Lw - 0.08, Lw - 0.04])
        assign(wp, oak)
        frame.append(wp)
        cv = spandrel_curve(Lk, H, p.bracket)
        poly = [(-0.093, zB)] + [(-(0.093 + u), zB - v) for (u, v) in cv]
        sd = prism(f"spandrel{i}", poly, x - 0.045, x + 0.045)
        hr_ = min(0.055, 0.11 * min(Lk, H))
        hole = cylinder("hole", hr_, 0.3, loc=(x, -(0.093 + 0.17 * Lk + 0.02), zB - 0.17 * H - 0.02), rot=(0, 90, 0), verts=10)
        boolean(sd, hole)
        bevel(sd, 0.007, segments=1)
        smooth(sd, 30)
        uv_grain(sd, axis=(0.0, -Lk, H))
        assign(sd, oak)
        frame.append(sd)
        for sx in (-1, 1):
            for (u, dz) in ((0.2, 0.05), (0.75, 0.05)):
                pegs_m.append(trs((x + sx * (BEAM_W / 2 - 0.004), -(0.093 + u * Lk), zB + dz + rb.u(-0.008, 0.008)), (0, 90 * sx, 0)))
            pegs_m.append(trs((x + sx * (PT / 2 - 0.004), -0.05, zB - 0.25 * H), (0, 90 * sx, 0)))
            pegs_m.append(trs((x + sx * (PT / 2 - 0.004), -0.05, zB - 0.8 * H), (0, 90 * sx, 0)))

    # ---------------------------------------------------------------- fascia bressummer, ledger, joists
    fas = beam("fascia", (-W / 2, -sp, zD - 0.08), (W / 2, -sp, zD - 0.08), 0.15, 0.16, cham=0.024, edges=(1, 0, 0, 0),
               runs=[(xs[i] + 0.13 + W / 2, xs[i + 1] - 0.13 + W / 2) for i in range(nb)], seg=0.4)
    assign(fas, oak)
    assign_faces(fas, endg, lambda c, n: abs(n.x) > 0.95)
    frame.append(fas)
    yM = yF
    mould = prism("fascia_mould", [(yM, zD - 0.002), (yM - 0.019, zD - 0.002), (yM - 0.019, zD - 0.012), (yM - 0.011, zD - 0.02),
                                   (yM - 0.007, zD - 0.034), (yM, zD - 0.038)], -W / 2, W / 2)
    smooth(mould, 50)
    uv_grain(mould, axis='X')
    assign(mould, oak)
    frame.append(mould)
    dent = box("dentil", (0.042, 0.014, 0.03))
    nd = max(4, int(W / 0.11))
    dm = [trs((-W / 2 + (k + 0.5) * W / nd, yM - 0.006, zD - 0.055)) for k in range(nd)
          if min(abs(-W / 2 + (k + 0.5) * W / nd - x) for x in xs) > 0.0]
    dn = instances_to_mesh("dentils", dent, dm, keep_base=False)
    uv_grain(dn, axis='X')
    assign(dn, oak)
    frame.append(dn)
    for i, x in enumerate(xs):                      # pegs through the cogged joints
        for dx in (-0.035, 0.035):
            pegs_m.append(trs((x + dx, yF + 0.004, zD - 0.115 + rb.u(-0.006, 0.006)), (90, 0, 0)))
    led = beam("ledger", (-W / 2 + 0.005, -0.053, zD - JOIST_H / 2), (W / 2 - 0.005, -0.053, zD - JOIST_H / 2), 0.10, JOIST_H, seg=0.5)
    assign(led, oak)
    frame.append(led)
    nj = max(1, int(round((sp - 0.05) / 0.45)) - 1)
    for k in range(nj):
        s = 0.053 + (sp - 0.053) * (k + 1) / (nj + 1)
        j = beam(f"joist{k}", (-W / 2 + 0.01, -s, zD - JOIST_H / 2), (W / 2 - 0.01, -s, zD - JOIST_H / 2), 0.075, JOIST_H, seg=0.8)
        assign(j, oak)
        assign_faces(j, endg, lambda c, n: abs(n.x) > 0.95)
        frame.append(j)

    # ---------------------------------------------------------------- deck planks (run out from the wall)
    rd = rng.fork("deck")
    x = -W / 2
    k = 0
    while x < W / 2 - 0.01:
        wk = rd.u(0.15, 0.22)
        if W / 2 - (x + wk) < 0.1:
            wk = W / 2 - x
        lift = rd.u(-1, 1) * 0.004 * age
        pl = beam(f"deck{k}", (x + wk / 2, -0.004, -DECK_T / 2), (x + wk / 2, -(D + rd.u(-0.008, 0.014)), -DECK_T / 2 + lift),
                  wk - rd.u(0.003, 0.007), DECK_T, ease=0.004, seg=0.6)
        assign(pl, deckm)
        assign_faces(pl, endg, lambda c, n: n.y < -0.95)
        deck.append(pl)
        x += wk
        k += 1

    # ---------------------------------------------------------------- posts
    a = radians(p.roof_pitch)
    tanA = tan(a)
    PH = p.post_height
    zt = PH + 0.14                                   # top of the eaves plate

    def z_u(s):                                      # underside of the rafters at distance s from the wall
        return zt + (sp - s) * tanA

    fin_sq = lathe("fin_sq", [(0.0, 0.0), (0.094, 0.0), (0.1, 0.008), (0.1, 0.028), (0.08, 0.042), (0.0, 0.042)], segments=4, smooth_angle=20)
    fin_ball = lathe("fin_ball", [(0.022, 0.04), (0.028, 0.052), (0.021, 0.062), (0.04, 0.085), (0.049, 0.112), (0.042, 0.138),
                                  (0.022, 0.157), (0.008, 0.166), (0.0, 0.167)], segments=10)
    uv_grain(fin_sq, 'Z')
    uv_grain(fin_ball, 'Z')
    fsq_m, fball_m = [], []
    post_top = []
    for i, x in enumerate(xs):
        top = PH if p.roof else (2.1 if i in lan_idx else RH + 0.11)
        post_top.append(top)
        runs = [(0.25 + DECK_T, RH - 0.14 + DECK_T)]
        if top > RH + 0.5:
            runs.append((RH + 0.09 + DECK_T, top - (0.5 if p.roof else 0.12) + DECK_T))
        lean = rb.u(-0.004, 0.004) * (1 + age) if not p.roof else 0.0
        po = beam(f"post{i}", (x, -sp, zD), (x + lean, -sp, top), PT, PT, up=(0, 1, 0), cham=0.022, runs=runs, seg=0.35)
        assign(po, oak)
        frame.append(po)
        if not p.roof:
            fsq_m.append(trs((x + lean, -sp, top), (0, 0, 45)))
            fball_m.append(trs((x + lean, -sp, top), (0, 0, rb.u(0, 90))))
        for zz in (0.12, RH - 0.035):                 # pegs through the rail tenons
            pegs_m.append(trs((x + rb.u(-0.01, 0.01), -(sp + PT / 2 - 0.004), zz), (90, 0, 0)))
    xw = W / 2 - 0.08
    top_w = (z_u(0.05) - 0.11) if p.roof else RH + 0.06
    for sx in (-1, 1):
        wpo = beam(f"wall_post{sx}", (sx * xw, -0.048, zD), (sx * xw, -0.048, top_w), PT, 0.09, up=(0, 1, 0), cham=0.016,
                   edges=(1, 1, 0, 0), runs=[(0.25 + DECK_T, RH - 0.14 + DECK_T)], seg=0.4,
                   shape=None if p.roof else (lambda t, Lw=top_w - zD: (0.05 * clamp((t - (Lw - 0.05)) / 0.05) ** 1.5, 0.0)),
                   shape_ts=() if p.roof else [top_w - zD - 0.05, top_w - zD - 0.025])
        assign(wpo, oak)
        frame.append(wpo)
    for nm, base, mm in (("finial_caps", fin_sq, fsq_m), ("finial_balls", fin_ball, fball_m)):
        if mm:
            f = instances_to_mesh(nm, base, mm, keep_base=False)
            assign(f, oak)
            frame.append(f)
        else:
            delete(base)

    # ---------------------------------------------------------------- balustrade
    zb0, zb1 = 0.15, RH - 0.07
    Hb = zb1 - zb0
    segs = [((xs[i] + PT / 2, -sp), (xs[i + 1] - PT / 2, -sp), 0.0) for i in range(nb)]
    for sx in (-1, 1):
        segs.append(((sx * xw, -0.093), (sx * xw, -(sp - PT / 2)), 90.0))
    hand = [(-0.034, 0.0), (0.034, 0.0), (0.045, 0.011), (0.045, 0.028), (0.039, 0.038), (0.031, 0.056), (0.013, 0.07),
            (-0.013, 0.07), (-0.031, 0.056), (-0.039, 0.038), (-0.045, 0.028), (-0.045, 0.011)]
    bal_m = []
    rr = rng.fork("rail")
    style = p.balusters
    for si, (A, B, yaw) in enumerate(segs):
        A3, B3 = Vector((A[0], A[1], 0.0)), Vector((B[0], B[1], 0.0))
        Ls = (B3 - A3).length
        dr = (B3 - A3) / Ls
        nn = max(2, int(Ls / 0.3))
        sagr = 0.003 + 0.004 * age
        path = [tuple(A3 + dr * (Ls * q / nn) + Vector((0, 0, zb1 - sagr * sin(pi * q / nn) + (rr.u(-0.001, 0.001) if 0 < q < nn else 0))))
                for q in range(nn + 1)]
        hr = sweep(f"handrail{si}", path, hand, smooth_path=False, up=(0, 0, 1))
        rails.append(hr)
        br = beam(f"bottom_rail{si}", A3 + Vector((0, 0, 0.12)), B3 + Vector((0, 0, 0.12)), 0.06, 0.06, cham=0.013,
                  edges=(0, 0, 1, 1), seg=0.5)
        rails.append(br)
        if style == "cross":
            m = max(1, int(round(Ls / (Hb * 1.0))))
            pw = Ls / m
            for q in range(m):
                c0 = A3 + dr * (pw * q)
                c1 = A3 + dr * (pw * (q + 1))
                if q > 0:
                    rails.append(beam(f"muntin{si}_{q}", c0 + Vector((0, 0, zb0 - 0.005)), c0 + Vector((0, 0, zb1 + 0.005)),
                                      0.05, 0.05, cham=0.01, runs=[(0.1, Hb - 0.09)], seg=None))
                ins = 0.025 if q > 0 else 0.0
                ine = 0.025 if q < m - 1 else 0.0
                d0, d1 = c0 + dr * ins, c1 - dr * ine
                rails.append(beam(f"xa{si}_{q}", d0 + Vector((0, 0, zb0 - 0.01)), d1 + Vector((0, 0, zb1 + 0.01)), 0.036, 0.052,
                                  cham=0.008, seg=None))
                rails.append(beam(f"xb{si}_{q}", d0 + Vector((0, 0, zb1 + 0.01)), d1 + Vector((0, 0, zb0 - 0.01)), 0.031, 0.05,
                                  cham=0.008, seg=None))
                cc = (c0 + c1) / 2 + Vector((0, 0, (zb0 + zb1) / 2))
                rg = torus(f"xring{si}_{q}", major=min(pw, Hb) * 0.3, minor=0.015, loc=tuple(cc), rot=(90, 0, yaw), major_seg=14, minor_seg=4)
                rails.append(rg)
                nrm = Vector((dr.y, -dr.x, 0.0))
                for sgn in (-1, 1):
                    bs = lathe(f"xboss{si}_{q}", [(0.0, 0.0), (0.042, 0.0), (0.044, 0.006), (0.03, 0.013), (0.0, 0.018)],
                               segments=7, loc=tuple(cc + nrm * (0.016 * sgn)), rot=(90 * sgn if yaw == 0 else 0, 90 * sgn if yaw else 0, 0))
                    rails.append(bs)
        else:
            pitch = 0.135 if style == "turned" else 0.16
            nbal = max(1, int(round(Ls / pitch)))
            for q in range(nbal):
                c = A3 + dr * (Ls * (q + 0.5) / nbal)
                if style == "turned":
                    bal_m.append(trs((c.x, c.y, zb0), (0, 0, yaw + rr.u(-4, 4))))
                else:
                    bal_m.append(trs((c.x, c.y, zb0), (90 + rr.u(-0.6, 0.6), 0, yaw + rr.u(-1.5, 1.5))) @ Matrix.Translation((0, 0, -0.011)))
    if style == "turned":
        k = Hb / 0.78
        prof = [(0.021, 0.087), (0.031, 0.1), (0.02, 0.125), (0.031, 0.165), (0.04, 0.24), (0.035, 0.32),
                (0.017, 0.555), (0.026, 0.592), (0.017, 0.615), (0.029, 0.668), (0.021, 0.693)]
        tb = lathe("bal_turn", [(r, z * k) for (r, z) in prof], segments=7)
        uv_grain(tb, 'Z')
        b0 = box("bal_b0", (0.058, 0.058, 0.09 * k), loc=(0, 0, 0.045 * k))
        b1 = box("bal_b1", (0.058, 0.058, 0.09 * k), loc=(0, 0, Hb - 0.045 * k))
        uv_grain(b0, 'Z')
        uv_grain(b1, 'Z')
        for o in (tb, b0, b1):
            assign(o, railm)
        base = join([tb, b0, b1], "bal_base")
        rails.append(instances_to_mesh("balusters", base, bal_m, keep_base=False))
    elif style == "splat":
        k = Hb / 0.78
        half = [(0.036, 0.0), (0.036, 0.05), (0.05, 0.085), (0.06, 0.14), (0.062, 0.2), (0.054, 0.27), (0.036, 0.34), (0.024, 0.41),
                (0.021, 0.47), (0.03, 0.52), (0.05, 0.57), (0.056, 0.63), (0.048, 0.69), (0.036, 0.725), (0.036, 0.78)]
        outline = [(u, v * k) for (u, v) in half] + [(-u, v * k) for (u, v) in reversed(half)]
        holes = [circle_pts(0.021, 8, center=(0.0, 0.185 * k)),
                 [(0.0, 0.575 * k), (0.016, 0.615 * k), (0.0, 0.655 * k), (-0.016, 0.615 * k)]]
        sb = shape_extrude("splat", outline, holes, depth=0.022, bevel=0.0)
        uv_grain(sb, 'Y')
        assign(sb, railm)
        rails.append(instances_to_mesh("balusters", sb, bal_m, keep_base=False))
    for o in rails:
        assign(o, railm)

    # ---------------------------------------------------------------- pentice roof
    tiles = None
    roof_objs = []
    if p.roof:
        ovh = 0.26
        hraf, tsark = 0.085, 0.018
        nv = Vector((0.0, -sin(a), cos(a)))          # roof normal
        dv = Vector((0.0, -cos(a), -sin(a)))         # down the slope
        Wr = W + 0.12

        def U(x, s, off=0.0):                        # point on the rafter underside plane (+ off along the normal)
            return Vector((x, -s, z_u(s))) + nv * off

        plate = beam("eaves_plate", (-W / 2 - 0.04, -sp, PH + 0.07), (W / 2 + 0.04, -sp, PH + 0.07), PT, 0.14, cham=0.022,
                     edges=(1, 0, 0, 0), runs=[(xs[i] + 0.5 + W / 2, xs[i + 1] - 0.42 + W / 2) for i in range(nb)], seg=0.5)
        assign(plate, oak)
        assign_faces(plate, endg, lambda c, n: abs(n.x) > 0.95)
        frame.append(plate)
        wplate = beam("wall_plate", (-W / 2, -0.05, z_u(0.05) - 0.055), (W / 2, -0.05, z_u(0.05) - 0.055), 0.09, 0.11, seg=0.6)
        assign(wplate, oak)
        frame.append(wplate)
        for sx in (-1, 1):                           # end ties, wall to plate
            tie = beam(f"tie{sx}", (sx * xw, -0.095, PH + 0.065), (sx * xw, -(sp - PT / 2), PH + 0.065), 0.09, 0.12, cham=0.016,
                       edges=(1, 1, 0, 0), runs=[(0.1, sp - 0.3)], seg=0.5)
            assign(tie, oak)
            frame.append(tie)
        # arched braces post -> plate
        bx, bz = min(0.4, (W - 0.16) / nb * 0.3), 0.42
        cvb = spandrel_curve(bx, bz, p.bracket, nose=0.03, n=7)
        for i, x in enumerate(xs):
            for sg in (-1, 1):
                if (i == 0 and sg < 0) or (i == nb and sg > 0):
                    continue
                poly = [(x + sg * PT / 2, PH)] + [(x + sg * (PT / 2 + u), PH - v) for (u, v) in cvb]
                bo = prism(f"brace{i}_{sg}", poly, -sp - 0.03, -sp + 0.03, plane="XZ")
                bevel(bo, 0.006, segments=1)
                smooth(bo, 30)
                uv_grain(bo, axis=(sg * bx, 0.0, bz))
                assign(bo, oak)
                frame.append(bo)
                pegs_m.append(trs((x + sg * (PT / 2 + 0.5 * bx), -(sp + PT / 2 - 0.004), PH + 0.05), (90, 0, 0)))
            pegs_m.append(trs((x, -(sp + PT / 2 - 0.004), PH - 0.25), (90, 0, 0)))
        # rafters with shaped tails
        nr = max(2, int(round(Wr / 0.55)) + 1)
        s0, s1 = 0.06, D + ovh
        Lr = (s1 - s0) / cos(a)
        rr2 = rng.fork("roof")
        for q in range(nr):
            x = -Wr / 2 + 0.05 + (Wr - 0.1) * q / (nr - 1) + (rr2.u(-0.015, 0.015) if 0 < q < nr - 1 else 0.0)
            rf = beam(f"rafter{q}", U(x, s0, hraf / 2), U(x, s1, hraf / 2), 0.06, hraf, up=nv, seg=None,
                      shape=lambda t, Lr=Lr: (0.05 * clamp((t - (Lr - 0.16)) / 0.16) ** 1.4, 0.0),
                      shape_ts=[Lr - 0.16, Lr - 0.1, Lr - 0.05])
            assign(rf, oak)
            assign_faces(rf, endg, lambda c, n: n.y < -0.75 and abs(n.x) < 0.2 and n.z < -0.3)
            frame.append(rf)
        # sarking boards across the rafters
        s_a, s_b = 0.03, D + ovh + 0.012
        Lsl = (s_b - s_a) / cos(a)
        nbd = max(3, int(math.ceil(Lsl / 0.2)))
        bwid = Lsl / nbd
        for q in range(nbd):
            sm = s_a + (q + 0.5) * bwid * cos(a)
            sb_ = beam(f"sark{q}", U(-Wr / 2 + rr2.u(0, 0.01), sm, hraf + tsark / 2), U(Wr / 2 - rr2.u(0, 0.01), sm, hraf + tsark / 2),
                       bwid - 0.004, tsark, up=nv, ease=0.002, seg=None)
            assign(sb_, deckm)
            roof_objs.append(sb_)
        # hand-made clay tiles, half-bond, laid from the eaves up
        tw_n, Lt, gauge, tt, cam, lift = 0.205, 0.30, 0.15, 0.014, 0.009, 0.027
        Wt = Wr + 0.07
        ncol = max(2, int(round(Wt / tw_n)))
        wx = Wt / ncol
        tw = wx - 0.005
        def make_tile(nm, under):
            tv, tf = [], []
            for iy, yy in enumerate((0.0, Lt)):
                for ix, xx in enumerate((-tw / 2, 0.0, tw / 2)):
                    zc = lift * (1 - yy / Lt) + cam * (1 - (2 * xx / tw) ** 2)
                    tv.append((xx, yy, zc))
                    tv.append((xx, yy, zc + tt))
            vid = lambda iy, ix, top: (iy * 3 + ix) * 2 + top
            for ix in range(2):
                tf.append((vid(0, ix, 1), vid(0, ix + 1, 1), vid(1, ix + 1, 1), vid(1, ix, 1)))         # top
                if under:
                    tf.append((vid(0, ix, 0), vid(1, ix, 0), vid(1, ix + 1, 0), vid(0, ix + 1, 0)))     # underside
                tf.append((vid(0, ix, 0), vid(0, ix + 1, 0), vid(0, ix + 1, 1), vid(0, ix, 1)))         # tail edge
            tf.append((vid(0, 0, 0), vid(0, 0, 1), vid(1, 0, 1), vid(1, 0, 0)))
            tf.append((vid(0, 2, 0), vid(1, 2, 0), vid(1, 2, 1), vid(0, 2, 1)))
            tl_ = mesh_object(nm, tv, tf)
            smooth(tl_, 50)
            uv_planar(tl_, (1, 0, 0), (0, 1, 0), offset=None)
            assign(tl_, mat("terracotta", "#d99672"))
            return tl_


        Rs = Matrix.Rotation(a, 4, 'X')
        q_e = (D + ovh + 0.045) / cos(a)
        tm, tm0 = [], []
        j = 0
        while True:
            qt = q_e - j * gauge
            if qt < 0.12:
                break
            ly = min(1.0, (qt - 0.035) / Lt)
            odd = j % 2 == 1
            cols = range(ncol + 1) if odd else range(ncol)
            for ci in cols:
                sxk = 1.0
                if odd:
                    xc = -Wt / 2 + ci * wx
                    if ci == 0:
                        xc, sxk = -Wt / 2 + wx / 4, 0.5
                    elif ci == ncol:
                        xc, sxk = Wt / 2 - wx / 4, 0.5
                else:
                    xc = -Wt / 2 + (ci + 0.5) * wx
                slip = 0.0
                if rr2.chance(0.05 + 0.12 * age):
                    slip = rr2.u(0.01, 0.02 + 0.04 * age)
                s_t = (qt + slip + rr2.u(-0.004, 0.004)) * cos(a)
                org = U(xc + rr2.u(-0.003, 0.003), s_t, hraf + tsark + rr2.u(0, 0.003))
                M = (Matrix.Translation(org) @ Rs @ Matrix.Rotation(radians(rr2.u(-1.4, 1.4) * (1 + age)), 4, 'Z')
                     @ Matrix.Rotation(radians(rr2.u(-1.5, 1.5)), 4, 'Y')
                     @ Matrix.Diagonal((sxk * rr2.u(0.97, 1.0), ly, rr2.u(0.85, 1.25), 1.0)))
                (tm0 if j == 0 else tm).append(M)
            j += 1
        tiles = join([instances_to_mesh("tiles_eaves", make_tile("tile_e", True), tm0, keep_base=False, seed=p.seed),
                      instances_to_mesh("tiles_field", make_tile("tile_f", False), tm, keep_base=False, seed=p.seed + 1)], "tiles")
        # bargeboards on the verges
        for sx in (-1, 1):
            xb = sx * (Wr / 2 + 0.013)
            bg = beam(f"barge{sx}", U(xb, 0.035, hraf + tsark + 0.012 - 0.08), U(xb, D + ovh + 0.03, hraf + tsark + 0.012 - 0.08),
                      0.026, 0.16, up=nv, ease=0.004, seg=0.5,
                      shape=lambda t, Lr=Lr: (0.07 * clamp((t - (Lr - 0.1)) / 0.14) ** 1.3, 0.0), shape_ts=[Lr - 0.1, Lr - 0.03, Lr + 0.02])
            assign(bg, oak)
            frame.append(bg)
        # dressed lead flashing against the wall
        off = hraf + tsark + lift + tt + cam + 0.006

        def R(s, o_):
            v = U(0.0, s, o_)
            return (v.y, v.z)

        k1, k2 = R(0.012, off + 0.004), R(0.13, off - 0.01 + 0.004)
        k2b, k1b = R(0.13, off - 0.01), R(0.006, off)
        zw = k1[1] + 0.085
        lead = prism("flashing", [(-0.011, zw), k1, k2, k2b, k1b, (-0.006, zw)], -Wt / 2 + 0.01, Wt / 2 - 0.01)
        densify(lead, 0.3)
        displace_noise(lead, strength=0.0025, scale=0.1, seed=p.seed, along_normal=False, axis_mask=(0, 0, 1))
        smooth(lead, 60)
        assign(lead, mat("lead"))
        paint(lead, grime=mask_mul(noise_mask(0.15, seed=p.seed + 9), 0.5))

    # ---------------------------------------------------------------- oak pegs
    pg = instances_to_mesh("pegs", peg, pegs_m, keep_base=False)
    assign(pg, endg)

    # ---------------------------------------------------------------- lanterns on scrolled iron brackets
    glass = mat("glass_lantern", glow=0.8)
    flame_m = mat("emissive", "#ffb45e", glow=4)
    reach = 0.34
    LS = 1.25                                        # lantern scale
    flat_bar = [(-0.0035, -0.01), (0.0035, -0.01), (0.0035, 0.01), (-0.0035, 0.01)]
    for k, i in enumerate(lan_idx):
        x = xs[i]
        yf = -(sp + PT / 2)
        za = post_top[i] - (0.4 if p.roof else 0.2)
        ironw.append(box(f"lb_plate{k}", (0.036, 0.006, 0.36), loc=(x, yf - 0.003, za - 0.09), bevel=0.0015, segments=1))
        bt = bolt("lb_bolt", 0.008, 0.006, kind="pyramid")
        ironw.append(instances_to_mesh(f"lb_bolts{k}", bt, [trs((x, yf - 0.006, za + 0.06), (90, 0, 0)),
                                                            trs((x, yf - 0.006, za - 0.24), (90, 0, 0))], keep_base=False))
        arm = [(x, yf - 0.004, za), (x, yf - reach * 0.5, za), (x, yf - reach, za), (x, yf - reach - 0.036, za + 0.013),
               (x, yf - reach - 0.05, za + 0.042), (x, yf - reach - 0.032, za + 0.064), (x, yf - reach - 0.012, za + 0.05),
               (x, yf - reach - 0.016, za + 0.034)]
        ironw.append(sweep(f"lb_arm{k}", arm, 0.0085, sides=5, step=0.03, scale_fn=lambda t: 1.0 - 0.45 * smoothstep(0.6, 1.0, t)))
        yc, zc = yf - 0.006 - 0.2, za - 0.008 - 0.2
        arc = [(x, yc + 0.2 * cos(radians(q * 90 / 10)), zc + 0.2 * sin(radians(q * 90 / 10))) for q in range(11)]
        ironw.append(sweep(f"lb_brace{k}", arc, flat_bar, up=(1, 0, 0), smooth_path=False))
        sc = spiral_path(0.007, 0.052, 1.6, n=22, a0_deg=45, plane="YZ", center=(x, yf - 0.067, za - 0.068))
        ironw.append(sweep(f"lb_scroll{k}", sc, flat_bar, up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.55 + 0.45 * t))
        hook = (x, yf - reach, za - 0.008)
        lob = []
        hx_, hy_, hz_ = hook
        for q in range(3):
            lob.append(torus(f"lan_link{k}", major=0.011, minor=0.0028, loc=(hx_, hy_, hz_ - 0.008 - q * 0.019), rot=(90, 0, 90 * (q % 2)),
                             major_seg=8, minor_seg=4))
        ztop = hz_ - 0.058
        lob.append(torus(f"lan_ring{k}", major=0.013, minor=0.003, loc=(hx_, hy_, ztop - 0.012), rot=(90, 0, 0), major_seg=10, minor_seg=5))
        zr = ztop - 0.024 - 0.09 * LS
        lob.append(lathe(f"lan_roof{k}", [(r * LS, z * LS) for (r, z) in ((0.0, 0.0), (0.1, 0.0), (0.104, 0.005), (0.062, 0.045), (0.03, 0.06),
                                          (0.03, 0.074), (0.04, 0.079), (0.014, 0.09), (0.0, 0.09))], segments=4, loc=(hx_, hy_, zr), rot=(0, 0, 45), smooth_angle=20))
        zb = zr - 0.19 * LS
        ht, hb = 0.062 * LS, 0.047 * LS
        for (cx, cy) in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            lob.append(beam(f"lan_bar{k}", (hx_ + cx * hb, hy_ + cy * hb, zb), (hx_ + cx * ht, hy_ + cy * ht, zr + 0.002), 0.01, 0.01,
                            up=(0, 1, 0), ease=0.0015, seg=None))
        lob.append(lathe(f"lan_tray{k}", [(r * LS, z * LS) for (r, z) in ((0.0, -0.034), (0.007, -0.03), (0.012, -0.014), (0.03, -0.008), (0.072, 0.0),
                                          (0.076, 0.006), (0.072, 0.013), (0.0, 0.013))], segments=4, loc=(hx_, hy_, zb - 0.012 * LS), rot=(0, 0, 45), smooth_angle=20))
        lob.append(lathe(f"lan_burner{k}", [(0.0, 0.0), (0.024, 0.0), (0.028, 0.014), (0.021, 0.028), (0.008, 0.035), (0.008, 0.06), (0.0, 0.06)],
                         segments=8, loc=(hx_, hy_, zb)))
        for o in lob:
            assign(o, iron)
        gt, gb = ht - 0.004, hb - 0.004
        gv = [(hx_ + cx * gb, hy_ + cy * gb, zb + 0.002) for (cx, cy) in ((1, 1), (-1, 1), (-1, -1), (1, -1))] + \
             [(hx_ + cx * gt, hy_ + cy * gt, zr) for (cx, cy) in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        gl = mesh_object(f"lan_glass{k}", gv, [(q, (q + 1) % 4, 4 + (q + 1) % 4, 4 + q) for q in range(4)])
        recalc_normals(gl)
        assign(gl, glass)
        fl = lathe(f"lan_flame{k}", [(0.0, 0.0), (0.012, 0.011), (0.015, 0.027), (0.009, 0.05), (0.0, 0.074)], segments=8,
                   loc=(hx_, hy_, zb + 0.06))
        assign(fl, flame_m)
        lob += [gl, fl]
        paint(lob[4], grime=facing((0, 0, 1)))            # soot on the lantern roof
        part(lob, f"lantern{k}", pivot=hook, motion="swing", amp=2.5, speed=0.35)
        # the socket sits just outside the front pane so the post behind the lantern is not scorched
        light(f"lantern{k}", (hx_, hy_ - 0.1, zb + 0.09), intensity=p.intensity, range=7.5, shadow_near=0.24)
        socket(f"lantern_hook{k}", "attach", hook, role="lantern")
    for o in ironw:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.06, seed=p.seed + 5, lo=0.35, hi=0.75), 0.4 + 0.5 * age))

    # ---------------------------------------------------------------- weathering
    nz1 = noise_mask(0.28, seed=p.seed + 1)
    nz2 = noise_mask(0.5, seed=p.seed + 2, lo=0.3, hi=0.7)
    nz3 = noise_mask(0.32, seed=p.seed + 3, lo=0.5, hi=0.85)

    def frame_grime(co, n):
        damp = np.clip(1.0 + co[:, 1] / 0.4, 0.0, 1.0)                 # against the wall
        under = np.clip(-co[:, 2] / 0.3, 0.0, 1.0)                     # rain run-off below the deck
        foot = np.clip(1.0 - np.abs(co[:, 2]) / 0.22, 0.0, 1.0)        # where water sits on the deck
        return np.clip((0.45 * damp + 0.45 * under * nz1(co, n) + 0.5 * foot) * (0.5 + age), 0.0, 1.0)

    def frame_moss(co, n):
        up = np.clip(n[:, 2], 0.0, 1.0)
        return np.clip(up * nz2(co, n) * (co[:, 2] < 0.3) * age * 1.2, 0.0, 1.0)

    for o in frame:
        paint(o, grime=frame_grime, special=frame_moss)

    def deck_wear(co, n):
        s = -co[:, 1]
        return np.clip(np.exp(-((s - 0.5 * D) / (0.32 * D)) ** 2) * (0.35 + 0.65 * nz2(co, n)) * 0.8, 0.0, 1.0) * (n[:, 2] > 0.5)

    def deck_grime(co, n):
        s = -co[:, 1]
        edge = np.clip(1.0 - s / 0.3, 0.0, 1.0) + np.clip(1.0 - (D - s) / 0.25, 0.0, 1.0)
        return np.clip((0.5 * edge + 0.4 * nz1(co, n)) * (0.5 + age), 0.0, 1.0)

    for o in deck:
        paint(o, grime=deck_grime, wear=deck_wear, special=mask_mul(nz1, 0.5 * age))
    for o in rails:
        paint(o, wear=lambda co, n: np.clip(n[:, 2], 0, 1) * (co[:, 2] > RH - 0.03) * (0.4 + 0.5 * nz1(co, n)),
              grime=lambda co, n: np.clip(1.0 - (co[:, 2] - 0.09) / 0.2, 0, 1) * (0.4 + 0.6 * age),
              special=mask_mul(nz2, 0.3 + 0.6 * age))
    if tiles is not None:
        def tile_moss(co, n):
            low = np.clip(1.0 - (co[:, 2] - (PH + 0.1)) / 0.9, 0.0, 1.0)
            return np.clip((0.2 + 0.8 * low) * nz3(co, n) * (0.25 + 0.75 * age), 0.0, 0.8)

        def tile_grime(co, n):
            return np.clip(np.clip(1.0 + co[:, 1] / 0.4, 0.0, 1.0) * 0.45 + 0.2 * nz1(co, n), 0.0, 1.0) * (0.3 + 0.6 * age)

        paint(tiles, special=tile_moss, grime=tile_grime)
        for o in roof_objs:
            paint(o, grime=mask_mul(nz1, 0.5 + 0.4 * age))

    # ---------------------------------------------------------------- sockets for the town generator
    socket("door", "attach", (0.0, 0.0, 0.0), role="door")
    for i in range(nb):
        xc = (xs[i] + xs[i + 1]) / 2
        socket(f"planter{i}", "attach", (xc, -(sp + 0.05), RH), role="planter")
    socket("sign", "attach", ((xs[0] + xs[1]) / 2, yF - 0.01, zD - 0.2), role="sign")
