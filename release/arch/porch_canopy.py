"""Door porch canopy: a small timber hood over a doorway. Two plank-cut carved brackets (wall leg with a
turned drop, cantilever arm, pierced brace panel pegged between them) carry a front beam and a roof that
is gabled (open king-post truss, cusped bargeboards, turned finial), near-flat or swept (curved rafters
with exposed tails, boarded cheeks). The covering is modelled piece by piece: hand-made clay tiles with
half-round ridge tiles, split shingles, or lead sheet over wood-cored rolls; lead flashing is dressed
against the wall. A picket / scalloped fretwork valance is nailed to the beams and a small iron candle
lantern hangs on a chain from a screw eye under the roof (or the bare hook is left as a socket).
Wall plane y = 0, origin on the ground below the centre of the hood.
"""
from tk import *
import math
import numpy as np

META = {"name": "Door porch canopy", "class": "component", "mount": "wall", "mount_height": 0.0,
        "tri_budget": 16000,          # tiles, shingles and fretwork boards are modelled one by one
        "focus": [0.0, -0.5, 2.36]}
PARAMS = {
    "width": {"default": 1.7, "min": 1.2, "max": 2.8, "desc": "overall width of the roof (m)"},
    "projection": {"default": 0.95, "min": 0.6, "max": 1.6, "desc": "overall projection of the roof from the wall (m)"},
    "height": {"default": 2.4, "min": 2.1, "max": 3.2, "desc": "underside of the hood beams above the ground (m)"},
    "roof": {"default": "gable", "choices": ["gable", "flat", "swept"],
             "desc": "gable = ridge square to the wall; flat = low pent (lead 5 deg, shingle / tile at their minimum pitch); swept = concave bell-cast pent"},
    "covering": {"default": "tile", "choices": ["tile", "lead", "shingle"], "desc": "roof covering"},
    "pitch": {"default": 44.0, "min": 30.0, "max": 58.0, "desc": "roof pitch of gable / mean pitch of swept roofs (deg)"},
    "paint": {"default": "#47776a", "desc": "paint colour of the joinery"},
    "finish": {"default": "painted", "choices": ["painted", "oak"], "desc": "painted frame, or oiled oak frame with painted trim"},
    "bracket": {"default": "scroll", "choices": ["scroll", "arched", "strut"], "desc": "bracket style: plank-cut scrolled console, cusped arched brace, plain raking strut"},
    "bracket_drop": {"default": 0.62, "min": 0.35, "max": 0.95, "desc": "height of the brackets below the beams (m)"},
    "valance": {"default": "picket", "choices": ["picket", "scallop", "none"], "desc": "fretwork valance boards under the beams (also sets the bargeboard edge)"},
    "valance_drop": {"default": 0.13, "min": 0.07, "max": 0.22, "desc": "how far the valance hangs below the beams (m)"},
    "lantern": {"default": "lantern", "choices": ["lantern", "hook", "none"], "desc": "hanging candle lantern, bare hook (attach socket) or nothing"},
    "intensity": {"default": 4.0, "min": 2.0, "max": 60.0, "desc": "lantern light intensity (cd)"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "moss / lichen on the roof covering"},
    "wear": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "paint wear, slipped tiles and general age"},
}
VARIANTS = {
    "flat_lead": {"roof": "flat", "covering": "lead", "width": 1.5, "projection": 0.8, "bracket": "arched", "valance": "scallop",
                  "paint": "#8a2a24", "bracket_drop": 0.5, "seed": 4},
    "swept_lead": {"roof": "swept", "covering": "lead", "width": 2.0, "projection": 1.15, "pitch": 40, "paint": "#d6ccb2",
                   "bracket": "scroll", "valance": "scallop", "valance_drop": 0.17, "bracket_drop": 0.8, "wear": 0.25, "seed": 11},
    "shingle_oak": {"roof": "gable", "covering": "shingle", "finish": "oak", "width": 1.4, "projection": 0.8, "pitch": 54,
                    "bracket": "strut", "valance": "none", "paint": "#7c2a22", "moss": 0.85, "wear": 0.85, "height": 2.3, "seed": 23},
    "swept_tile_wide": {"roof": "swept", "covering": "tile", "width": 2.5, "projection": 1.35, "pitch": 46, "paint": "#27405e",
                        "bracket": "arched", "bracket_drop": 0.85, "height": 2.55, "moss": 0.6, "seed": 31},
}

ZV = Vector((0, 0, 1))


class Slope:
    """A roof slope: eaves line through O along A (horizontal), rising along the horizontal direction Dh
    following prof = [(d, h), ...] (the top of the boarding). Points are addressed by arc length s from the
    eaves, across position a and normal offset n; nothing passes the plumb plane d = dmax (ridge / wall)."""

    def __init__(self, O, A, Dh, prof, dmax):
        self.O, self.A, self.Dh = Vector(O), Vector(A), Vector(Dh)
        self.prof, self.dmax = prof, dmax
        self.cum = [0.0]
        seg = []
        for (d0, h0), (d1, h1) in zip(prof, prof[1:]):
            self.cum.append(self.cum[-1] + hypot(d1 - d0, h1 - h0))
            seg.append(atan2(h1 - h0, d1 - d0))
        self.S = self.cum[-1]
        self.va = [seg[0]] + [(seg[i] + seg[i + 1]) / 2 for i in range(len(seg) - 1)] + [seg[-1]]
        self.k = 1 if len(prof) == 2 else len(prof) - 1

    def at(self, s):
        pr, cum = self.prof, self.cum
        if s <= 0.0:
            return pr[0][0] + s * cos(self.va[0]), pr[0][1] + s * sin(self.va[0]), self.va[0]
        if s >= self.S:
            e = s - self.S
            return pr[-1][0] + e * cos(self.va[-1]), pr[-1][1] + e * sin(self.va[-1]), self.va[-1]
        i = 0
        while i < len(pr) - 2 and cum[i + 1] < s:
            i += 1
        t = (s - cum[i]) / (cum[i + 1] - cum[i])
        return lerp(pr[i][0], pr[i + 1][0], t), lerp(pr[i][1], pr[i + 1][1], t), lerp(self.va[i], self.va[i + 1], t)

    def s_of_d(self, d):
        pr, cum = self.prof, self.cum
        for i in range(len(pr) - 1):
            if d <= pr[i + 1][0] or i == len(pr) - 2:
                return cum[i] + (d - pr[i][0]) / max(pr[i + 1][0] - pr[i][0], 1e-6) * (cum[i + 1] - cum[i])

    def s_top(self, n):
        """Arc length at which a line n above the boarding reaches the ridge / wall plane."""
        return self.S + max(n, 0.0) * tan(self.va[-1]) + 0.002

    def pt(self, s, a, n=0.0):
        d, h, th = self.at(s)
        dd, hh = d - n * sin(th), h + n * cos(th)
        if dd > self.dmax:
            hh -= (dd - self.dmax) * tan(th)
            dd = self.dmax
        return self.O + self.A * a + self.Dh * dd + ZV * hh

    def mtx(self, s, a, n=0.0):
        d, h, th = self.at(s)
        t = self.Dh * cos(th) + ZV * sin(th)
        nr = -self.Dh * sin(th) + ZV * cos(th)
        p = self.O + self.A * a + self.Dh * d + ZV * h + nr * n
        A = self.A
        return Matrix(((A.x, t.x, nr.x, p.x), (A.y, t.y, nr.y, p.y), (A.z, t.z, nr.z, p.z), (0, 0, 0, 1)))


def loft(name, rings):
    """Closed solid through a list of 4-point rings."""
    verts = [tuple(v) for r in rings for v in r]
    m, faces = 4, []
    for i in range(len(rings) - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            faces.append((a, b, b + m, a + m))
    faces.append(tuple(range(m)))
    faces.append(tuple(range((len(rings) - 1) * m, len(rings) * m)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def slab(name, sl, s0, s1, a0, a1, n0, n1, k=None, bev=0.0):
    """Board / sheet lying on a slope: s0..s1 up the slope, a0..a1 across, n0..n1 normal to it."""
    k = k or sl.k
    rings = []
    for i in range(k + 1):
        s = s0 + (s1 - s0) * i / k
        b = n0(s) if callable(n0) else n0
        t = n1(s) if callable(n1) else n1
        rings.append([sl.pt(s, a0, b), sl.pt(s, a1, b), sl.pt(s, a1, t), sl.pt(s, a0, t)])
    o = loft(name, rings)
    if bev > 0:
        bevel(o, bev, segments=1)
    return o


def ccw(pts):
    ar = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return pts if ar > 0 else pts[::-1]


def brace_shape(style, R, D):
    """Outline (+ pierced holes) of the bracket panel: x out from the wall leg, y down from the arm."""
    e = 0.045
    holes = []
    if style == "arched":
        pts = [(0, 0), (R, 0), (R, -e)]
        n = 20
        for i in range(1, n):
            a = 90.0 * i / n
            f = 1.0 - 0.17 * max(0.0, 1.0 - abs(a - 45.0) / 9.0)         # gothic cusp
            pts.append((R - (R - e) * f * sin(radians(a)), -D + (D - e) * f * cos(radians(a))))
        pts += [(e, -D), (0, -D)]
        ex, ey = R - (R - e) * 0.7071, -D + (D - e) * 0.7071
        holes.append(circle_pts(0.2 * hypot(ex, ey), 12, center=(ex * 0.5, ey * 0.5)))
    elif style == "scroll":
        pts = [(0, 0), (R, 0)]
        pts += [(x, y) for x, y, _ in bezier_path((R, -e, 0), (R * 0.99, -D * 0.64, 0), (R * 0.17, -D * 0.28, 0), (0.078, -D + 0.075, 0), n=20)]
        pts += [(0.092, -D + 0.05), (0.09, -D + 0.022), (0.07, -D + 0.004), (0.045, -D), (0, -D)]            # rolled toe
        m = min(R, D)
        holes.append(circle_pts(0.115 * m, 12, center=(R * 0.36, -D * 0.25)))
        holes.append(circle_pts(0.055 * m, 8, center=(R * 0.71, -D * 0.115 - 0.012)))
        holes.append(circle_pts(0.04 * m, 8, center=(0.06 + R * 0.06, -D * 0.6)))
    else:
        L = hypot(R, D)
        bw = 0.085
        pts = [(R - bw * L / D, 0), (R, 0), (0, -D), (0, -D + bw * L / R)]
    return ccw(pts), holes


def valance_board(style, w, lap, drop, th):
    """One fretwork board: x across (width w), y from +lap (nailed to the beam) down to -drop."""
    holes = []
    if style == "picket":
        y1 = -drop + w * 0.62
        pts = [(-w / 2, lap), (-w / 2, y1), (-w * 0.3, y1 - w * 0.1), (-w * 0.35, y1 - w * 0.24), (0, -drop),
               (w * 0.35, y1 - w * 0.24), (w * 0.3, y1 - w * 0.1), (w / 2, y1), (w / 2, lap)]
        holes.append(circle_pts(w * 0.17, 10, center=(0, y1 + w * 0.1)))
    else:
        y1 = -drop + w / 2
        pts = [(-w / 2, lap), (-w / 2, y1)] + [(w / 2 * cos(radians(a)), y1 + w / 2 * sin(radians(a))) for a in range(210, 331, 30)] + [(w / 2, y1), (w / 2, lap)]
        holes.append(circle_pts(w * 0.16, 10, center=(0, y1 - w * 0.04)))
        holes.append(circle_pts(w * 0.075, 6, center=(0, y1 + w * 0.27)))
    o = shape_extrude("val_board", ccw(pts), holes, depth=th)
    uv_grain(o, "Y")
    realize(o)
    return o


def tile_base(w, L, t, camber):
    v = [(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, L, 0), (-w / 2, L, 0),
         (-w / 2, 0, t), (0, 0, t + camber), (w / 2, 0, t), (w / 2, L, t), (0, L, t + camber), (-w / 2, L, t)]
    f = [(0, 3, 2, 1), (4, 5, 8, 9), (5, 6, 7, 8), (0, 1, 6, 5, 4), (2, 3, 9, 8, 7), (1, 2, 7, 6), (3, 0, 4, 9)]
    o = mesh_object("tile", v, f)
    recalc_normals(o)
    return o


def shingle_base(w, L, t):
    v = [(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, L, 0), (-w / 2, L, 0),
         (-w / 2, 0, t), (w / 2, 0, t), (w / 2, L, t * 0.25), (-w / 2, L, t * 0.25)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    o = mesh_object("shingle", v, f)
    recalc_normals(o)
    uv_grain(o, "Y")
    realize(o)
    return o


def ridge_tile_base(r, t, L, n=8):
    verts = []
    for y in (0.0, L):
        for rr in (r, r - t):
            for i in range(n + 1):
                a = pi * i / n
                verts.append((rr * cos(a), y, rr * sin(a)))

    def ix(yi, ri, i):
        return (yi * 2 + ri) * (n + 1) + i

    faces = []
    for i in range(n):
        faces.append((ix(0, 0, i), ix(0, 0, i + 1), ix(1, 0, i + 1), ix(1, 0, i)))
        faces.append((ix(0, 1, i), ix(1, 1, i), ix(1, 1, i + 1), ix(0, 1, i + 1)))
        faces.append((ix(0, 0, i), ix(0, 1, i), ix(0, 1, i + 1), ix(0, 0, i + 1)))
        faces.append((ix(1, 0, i), ix(1, 0, i + 1), ix(1, 1, i + 1), ix(1, 1, i)))
    faces.append((ix(0, 0, 0), ix(1, 0, 0), ix(1, 1, 0), ix(0, 1, 0)))
    faces.append((ix(0, 0, n), ix(0, 1, n), ix(1, 1, n), ix(1, 0, n)))
    o = mesh_object("ridge_tile", verts, faces)
    recalc_normals(o)
    smooth(o, 50)
    uv_cyl(o, "Y")
    realize(o)
    return o


def unit_matrices(sl, a0, a1, kind, r, wear):
    """Instance matrices of tiles / shingles laid in lapped, half-bonded courses on a slope."""
    tile = kind == "tile"
    gauge, L, w, t = (0.10, 0.265, 0.165, 0.012) if tile else (0.125, 0.36, 0.11, 0.013)
    lift = 2.0 * t if tile else 1.5 * t
    mats, s, ci = [], 0.05, 0
    while s < sl.S - 0.04:
        Leff = min(L, sl.S - 0.012 - s)
        tilt = degrees(math.asin(min(0.25, lift / Leff)))
        if tile:
            n = max(1, round((a1 - a0) / w))
            cw = (a1 - a0) / n
            edges = [a0 + k * cw for k in range(n + 1)] if ci % 2 == 0 else [a0] + [a0 + (k + 0.5) * cw for k in range(n)] + [a1]
        else:
            edges = [a0]
            while edges[-1] < a1 - 0.04:
                edges.append(edges[-1] + r.u(0.075, 0.15))
            sc = (a1 - a0) / (edges[-1] - a0)
            edges = [a0 + (e - a0) * sc for e in edges]
        for e0, e1 in zip(edges, edges[1:]):
            slip = r.u(-0.004, 0.004) * (0.5 + wear)
            if r.chance(0.03 + 0.07 * wear):
                slip -= r.u(0.008, 0.022)
            sx = (e1 - e0 - (0.003 if tile else r.u(0.002, 0.007))) / w
            M = sl.mtx(s + slip, (e0 + e1) / 2) @ trs((0, 0, lift + r.u(0, 0.002)), (-tilt + r.u(-0.7, 0.7), r.u(-1.0, 1.0), r.u(-1.3, 1.3)),
                                                      (sx, Leff / L * (1.0 if tile else r.u(0.97, 1.0)), 1.0 if tile else r.u(0.8, 1.2)))
            mats.append(M)
        s += gauge
        ci += 1
    base = tile_base(w, L, t, 0.004) if tile else shingle_base(w, L, t)
    return base, mats, lift + t + 0.006


def build(p, rng):
    W, PJ, H = p.width, p.projection, p.height
    gable = p.roof == "gable"
    cov = p.covering
    seed = int(p.seed)

    painted = mat("wood_painted", p.paint)
    frame_m = mat("wood_oak", "#6f4e31") if p.finish == "oak" else painted
    deck_m = mat("wood_planks", "#9a8468")
    lead_m = mat("lead")
    iron = mat("iron_wrought")

    bt, ah, leg_d = 0.075, 0.095, 0.07            # arm / leg width, beam depth, leg depth off the wall
    rd, rw, dk = 0.07, 0.055, 0.02                # rafter depth and width, boarding thickness
    ov_e, ov_v = 0.12, 0.07                       # eaves and verge overhang beyond the frame
    xb = W / 2 - (ov_e if gable else ov_v) - bt / 2
    PF = PJ - (ov_v if gable else ov_e)           # front face of the front beam
    xo = xb + bt / 2
    ze = H + ah
    D = min(p.bracket_drop, H - 1.2)
    R = (PF - leg_d) * 0.92
    frame, trim, lead, irons = [], [], [], []

    # ------------------------------------------------------------------ frame: arms, front beam, wall plate, legs
    for sx in (-1, 1):
        arm = box("arm", (bt, PF, ah), loc=(sx * xb, -PF / 2 - 0.0005, H + ah / 2), bevel=0.005, segments=1)
        uv_grain(arm, "Y")
        leg_l = D + 0.11
        leg = box("leg", (bt, leg_d, leg_l), loc=(sx * xb, -leg_d / 2 - 0.0005, H - leg_l / 2), bevel=0.005, segments=1)
        uv_grain(leg, "Z")
        drop = lathe("leg_drop", [(0.0, -0.092), (0.011, -0.087), (0.023, -0.068), (0.027, -0.05), (0.017, -0.034), (0.012, -0.025),
                                  (0.029, -0.015), (0.031, -0.005), (0.0, 0.0)], segments=10, loc=(sx * xb, -leg_d / 2, H - leg_l))
        outer, holes = brace_shape(p.bracket, R, D)
        th = 0.06 if p.bracket == "strut" else 0.046
        brace = shape_extrude("brace", outer, holes, depth=th, loc=(sx * xb + th / 2, -leg_d, H), rot=(90, 0, -90), bevel=0.004)
        uv_grain(brace, (R, D, 0))
        frame += [arm, leg, drop, brace]
    beam = box("front_beam", (2 * xb - bt, 0.075, ah), loc=(0, -PF + 0.0375, H + ah / 2), bevel=0.005, segments=1)
    uv_grain(beam, "X")
    plate = box("wall_plate", (2 * xb - bt, 0.04, ah), loc=(0, -0.0205, H + ah / 2), bevel=0.004, segments=1)
    uv_grain(plate, "X")
    frame += [beam, plate]

    # pegs through the joints (timber) and coach bolts into the wall (iron)
    peg = lathe("peg", [(0.0095, 0.0), (0.0085, 0.003), (0.0045, 0.0052), (0.0, 0.0058)], segments=7)
    pm = []
    for sx in (-1, 1):
        for (y, z) in ((-leg_d - R * 0.22, H + ah * 0.5), (-leg_d - R * 0.78, H + ah * 0.5), (-PF + 0.0375, H + ah * 0.28), (-PF + 0.0375, H + ah * 0.72),
                       (-leg_d * 0.5, H - D * 0.2), (-leg_d * 0.5, H - D * 0.82)):
            pm.append(trs((sx * xo, y, z), (0, sx * 90, 0)))
    frame.append(instances_to_mesh("pegs", peg, pm, keep_base=False))
    cb = join([cylinder("cb_washer", 0.02, 0.003, loc=(0, 0, 0.0015), verts=10), cylinder("cb_head", 0.0135, 0.009, loc=(0, 0, 0.0075), verts=4, radius2=0.012)], "coach_bolt")
    bm = [trs((sx * xb, -leg_d, H - D - 0.06), (90, 0, rng.u(0, 90))) for sx in (-1, 1)]
    bm += [trs((x, -0.0405, H + ah / 2), (90, 0, rng.u(0, 90))) for x in (-xb * 0.62, 0.0, xb * 0.62)]
    irons.append(instances_to_mesh("coach_bolts", cb, bm, keep_base=False))

    # ------------------------------------------------------------------ roof carcass
    edge_amp = {"picket": 0.03, "scallop": 0.022, "none": 0.0}[p.valance]

    def barge_edge(S, lobe):
        nl = max(2, round(S / lobe))
        if p.valance == "picket":
            return nl, (lambda s: -0.135 - edge_amp * abs(sin(pi * nl * clamp(s / S, 0, 1))) ** 0.7)
        return nl, (lambda s: -0.15 - edge_amp * cos(2 * pi * nl * clamp(s / S, 0, 1)))

    decks, roofs = [], []
    y_l = -PF * 0.62                               # lantern position under the hood
    if gable:
        th = radians(p.pitch)
        ex = {"tile": 0.029, "shingle": 0.031, "lead": 0.006}[cov] * sin(th)   # covering build-up beyond the boarding edge
        hw = W / 2 - ex                            # so the overall width over the covering is `width`
        ov_g = ov_e - ex
        prof = [(0.0, 0.0), (hw, hw * tan(th))]
        Oz = ze - (ov_g + bt / 2) * tan(th) + (rd + dk) / cos(th)
        z_ap = Oz + hw * tan(th)                   # apex of the boarding
        slopes = [(1, Slope((-hw, 0, Oz), (0, -1, 0), (1, 0, 0), prof, hw)), (-1, Slope((hw, 0, Oz), (0, 1, 0), (-1, 0, 0), prof, hw))]
        deck_s0 = 0.0 if cov == "lead" else 0.02
        for sg, sl in slopes:
            def ar(y0, y1):
                return tuple(sorted((-sg * y0, -sg * y1)))
            S = sl.S
            yd = -PJ + (0.004 if cov == "lead" else 0.045)
            dkk = slab("deck", sl, deck_s0, S, *ar(yd, -0.001), -dk, 0.0)
            densify(dkk, 0.22)
            uv_grain(dkk, "Y")
            decks.append(dkk)
            ys = [(-PF, -PF + 0.06), (-0.061, -0.001)]
            if PF > 0.8:
                ys.insert(1, (-PF / 2 - 0.028, -PF / 2 + 0.028))
            for (y0, y1) in ys:
                rf = slab("rafter", sl, 0.045, S, *ar(y0, y1), -dk - rd, -dk + 0.001, bev=0.004)
                uv_grain(rf, tuple(sl.Dh * cos(th) + ZV * sin(th)))
                frame.append(rf)
            s_c = (ov_g + bt / 2) / cos(th)
            hb = rd / cos(th) + 0.004
            blk = box("eaves_blocking", (0.032, PF - 0.12, hb), loc=(-sg * xb, -PF / 2, ze + hb / 2 - 0.004))
            uv_grain(blk, "Y")
            frame.append(blk)
            # raking strut of the front truss, from the king-post foot to the rafter
            s_m = S * 0.56
            pm_ = sl.pt(s_m, 0.0, -dk - rd + 0.01)
            p0, p1 = Vector((-sg * 0.02, 0, ze - 0.01)), Vector((pm_.x, 0, pm_.z))
            dv = (p1 - p0).normalized()
            pv = Vector((-dv.z, 0, dv.x)) * 0.024
            st = loft("strut", [[tuple(q + Vector((0, yy, 0))) for q, yy in ((pp - pv, -PF + 0.008), (pp + pv, -PF + 0.008), (pp + pv, -PF + 0.052), (pp - pv, -PF + 0.052))]
                                for pp in (p0, p1)])
            bevel(st, 0.004, segments=1)
            uv_grain(st, tuple(dv))
            frame.append(st)
            nl, fn = barge_edge(S, 0.17)
            bb = slab("bargeboard", sl, 0.0, S, *ar(-PJ + 0.045, -PJ + 0.07), fn, 0.0, k=nl * 6 if edge_amp > 0 else 1)
            uv_grain(bb, tuple(sl.Dh * cos(th) + ZV * sin(th)))
            trim.append(bb)
        # ridge board, king post, finial
        rb_bot = z_ap - dk / cos(th) - 0.125
        rb = box("ridge_board", (0.034, PJ - 0.06, 0.125), loc=(0, -(PJ - 0.06) / 2 - 0.001, rb_bot + 0.0625), bevel=0.003, segments=1)
        uv_grain(rb, "Y")
        kp_top = z_ap - (rd + dk) / cos(th) + 0.03
        kp = box("king_post", (0.062, 0.058, kp_top - ze + 0.01), loc=(0, -PF + 0.03, (kp_top + ze - 0.01) / 2), bevel=0.006, segments=1)
        uv_grain(kp, "Z")
        frame += [rb, kp]
        yf = -PJ + 0.06
        fb = box("finial_block", (0.07, 0.07, 0.3), loc=(0, yf, z_ap - 0.06), bevel=0.008, segments=1)
        uv_grain(fb, "Z")
        fin = lathe("finial", [(0.0, -0.3), (0.012, -0.295), (0.026, -0.27), (0.03, -0.25), (0.02, -0.228), (0.014, -0.216), (0.03, -0.206), (0.034, -0.196),
                               (0.034, -0.19)], segments=14, loc=(0, yf, z_ap - 0.02))
        fin2 = lathe("finial_top", [(0.036, 0.0), (0.04, 0.008), (0.03, 0.018), (0.016, 0.026), (0.013, 0.04), (0.027, 0.056), (0.034, 0.075), (0.031, 0.095),
                                    (0.018, 0.125), (0.009, 0.17), (0.004, 0.22), (0.0, 0.235)], segments=14, loc=(0, yf, z_ap + 0.09))
        trim += [fb, fin, fin2]
        z_hook = rb_bot
        roofs = [(sl, tuple(sorted((sg * PJ, sg * 0.006))), [sg * PJ]) for sg, sl in slopes]
        top_th = th
    else:
        if p.roof == "flat":
            th = radians({"lead": 5.0, "shingle": 21.0, "tile": 27.0}[cov])
            prof = [(0.0, 0.0), (PJ, PJ * tan(th))]
        else:
            Hr = PJ * tan(radians(p.pitch))
            prof = [(PJ * i / 14, Hr * (0.36 * (i / 14) + 0.64 * (i / 14) ** 2.3)) for i in range(15)]
        d_ref = ov_e + 0.0375
        th_e = atan2(prof[1][1] - prof[0][1], prof[1][0] - prof[0][0])
        sl0 = Slope((0, 0, 0), (1, 0, 0), (0, 1, 0), prof, PJ - 0.002)
        _, h_ref, th_ref = sl0.at(sl0.s_of_d(d_ref))
        Oz = ze - h_ref + (rd + dk) / cos(th_ref)
        sl = Slope((0, -PJ, Oz), (1, 0, 0), (0, 1, 0), prof, PJ - 0.002)
        S, top_th = sl.S, sl.va[-1]
        lead_c = cov == "lead"
        xd = W / 2 - (0.004 if lead_c else 0.045)
        dkk = slab("deck", sl, 0.0 if lead_c else 0.02, S, -xd, xd, -dk, 0.0)
        densify(dkk, 0.22)
        uv_grain(dkk, "X")
        decks.append(dkk)
        nr = 3 if 2 * xb < 1.55 else 5
        for i in range(nr):
            a = lerp(-xb, xb, i / (nr - 1))
            rf = slab("rafter", sl, 0.045, S, a - rw / 2, a + rw / 2, -dk - rd, -dk + 0.001, k=max(1, sl.k // 2), bev=0.004)
            uv_grain(rf, (0, cos(th_e), sin(th_e)))
            frame.append(rf)
        s_c = sl.s_of_d(d_ref)
        hb = rd / cos(th_ref) + 0.004
        blk = box("eaves_blocking", (2 * xb, 0.032, hb), loc=(0, -PF + 0.0375, ze + hb / 2 - 0.004))
        uv_grain(blk, "X")
        frame.append(blk)
        # boarded cheeks between the arms and the outer rafters
        for sx in (-1, 1):
            rings = []
            for i in range(25):
                s = S * i / 24
                pu = sl.pt(s, 0.0, -(dk + rd) + 0.004)
                if pu.z > ze + 0.012 and pu.y > -PF + 0.02:
                    x0, x1 = sx * xb - 0.014, sx * xb + 0.014
                    rings.append([(x0, pu.y, ze - 0.002), (x1, pu.y, ze - 0.002), (x1, pu.y, pu.z), (x0, pu.y, pu.z)])
            if len(rings) >= 2 and rings[-1][2][2] - ze > 0.035:
                ck = loft("cheek", rings)
                uv_grain(ck, "Z")
                frame.append(ck)
            nl, fn = barge_edge(S, 0.17)
            a0, a1 = sorted((sx * (W / 2 - 0.07), sx * (W / 2 - 0.045)))
            bb = slab("bargeboard", sl, -0.01, S, a0, a1, fn, 0.0, k=max(sl.k, nl * 6) if edge_amp > 0 else sl.k)
            uv_grain(bb, (0, cos(th_e), sin(th_e)))
            trim.append(bb)
        # ledger under the rafter heads
        lz = sl.pt(S, 0.0, -(dk + rd)).z - 0.046 * tan(top_th)
        if lz - 0.1 > ze + 0.03:
            ld = box("ledger", (2 * xb + bt, 0.045, 0.1), loc=(0, -0.023, lz - 0.05), bevel=0.004, segments=1)
            uv_grain(ld, "X")
            frame.append(ld)
        s_l = sl.s_of_d(PJ + y_l)
        z_hook = sl.pt(s_l, 0.0, -(dk + rd)).z
        roofs = [(sl, (-W / 2, W / 2), [-W / 2, W / 2])]

    # ------------------------------------------------------------------ covering
    r_cov = rng.fork("covering")
    cov_objs = []
    ct = 0.012
    for (sl, (a0, a1), verges) in roofs:
        S = sl.S
        if cov == "lead":
            sheet = slab("lead_sheet", sl, 0.0, sl.s_top(0.006), a0, a1, 0.0, 0.006)
            densify(sheet, 0.2)
            displace_noise(sheet, strength=0.0025, scale=0.13, seed=seed + 3)
            lead.append(sheet)
            lead.append(slab("lead_drip", sl, 0.0, 0.007, a0, a1, -dk - 0.03, 0.006))
            for av in verges:
                lo, hi = sorted((av, av - math.copysign(0.007, av)))
                lead.append(slab("lead_verge", sl, 0.0, S, lo, hi, -dk - 0.022, 0.006))
            nroll = max(2, int(round((a1 - a0) / 0.52)) + 1)
            for i in range(nroll):
                a = lerp(a0 + 0.03, a1 - 0.03, i / (nroll - 1))
                if abs(a) < 0.05 and gable:
                    continue
                path = [tuple(sl.pt(0.024 + (S * 0.995 - 0.024) * j / sl.k, a, 0.012)) for j in range(sl.k + 1)]
                lead.append(sweep("lead_roll", path, 0.021, sides=7, smooth_path=False))
                lead.append(sphere("roll_end", 0.022, loc=tuple(sl.pt(0.024, a, 0.011)), segments=8, rings=5, scale=(1, 1, 0.9)))
            ct = 0.012
        else:
            base, mats, ct = unit_matrices(sl, a0, a1, cov, r_cov, p.wear)
            fl = slab("eaves_fillet", sl, 0.02, 0.08, a0 + 0.004, a1 - 0.004, 0.0, ct - 0.02)
            uv_grain(fl, tuple(sl.A))
            decks.append(fl)
            ma = [m for m in mats if r_cov.chance(0.62)]
            mb = [m for m in mats if not any(m is x for x in ma)]
            for nm, ms in (("cover_a", ma), ("cover_b", mb)):
                if ms:
                    o = instances_to_mesh(nm, base, ms, keep_base=True, seed=seed + len(cov_objs))
                    uv_keep(o)
                    cov_objs.append(o)
            delete([base])

    tile_tints = ("#d08a66", "#a8684c") if cov == "tile" else ("#b9b2a4", "#8f8778")
    cov_ms = [mat("terracotta", t) if cov == "tile" else mat("wood_weathered", t) for t in tile_tints]
    for i, o in enumerate(cov_objs):
        assign(o, cov_ms[i % 2])

    # ridge (gable) / head flashing against the wall
    if gable:
        sl = slopes[0][1]
        if cov == "tile":
            rr = 0.105
            zc = z_ap - rr * tan(th) + (ct - 0.012) / cos(th) + 0.004
            rt = ridge_tile_base(rr, 0.014, 0.31)
            nrt = max(2, round((PJ - 0.01) / 0.3))
            Lr = (PJ - 0.012) / nrt
            rm = [trs((r_cov.u(-0.004, 0.004), -PJ + i * Lr, zc + r_cov.u(-0.003, 0.003)), (r_cov.u(-1.2, 1.2), 0, r_cov.u(-1.5, 1.5)), (1.0, (Lr + 0.012) / 0.31 if i < nrt - 1 else Lr / 0.31, 1.0))
                  for i in range(nrt)]
            ro = instances_to_mesh("ridge_tiles", rt, rm, keep_base=False)
            uv_keep(ro)
            assign(ro, cov_ms[1])
            cov_objs.append(ro)
        for sg, s2 in slopes:
            aw = tuple(sorted((sg * 0.001, sg * 0.115)))
            if cov == "shingle":
                rbd = slab("ridge_board_top", s2, s2.S - 0.13, s2.s_top(ct + 0.018), *sorted((sg * PJ, sg * 0.006)), ct - 0.004, ct + 0.016, bev=0.002)
                uv_grain(rbd, "Y")
                assign(rbd, cov_ms[1])
                cov_objs.append(rbd)
            elif cov == "lead":
                lead.append(slab("ridge_wing", s2, s2.S - 0.12, s2.s_top(0.012), *sorted((sg * PJ, sg * 0.006)), 0.006, 0.011))
            lead.append(slab("flash_cover", s2, 0.03, s2.s_top(ct + 0.03), *aw, ct + 0.012 if cov != "lead" else 0.02, ct + 0.017 if cov != "lead" else 0.025))
        if cov == "lead":
            lead.append(cylinder("ridge_roll", 0.03, PJ - 0.004, loc=(0, -PJ / 2, z_ap + 0.012), rot=(90, 0, 0), verts=10))
        # upstand dressed against the wall, following the roof line
        course = 0.075
        nst = max(2, int((hw - 0.03) * tan(th) / course))
        sw = (hw - 0.03) / nst                     # run of one step
        rings = []
        for sgn in (-1, 1):
            order = range(nst) if sgn < 0 else range(nst - 1, -1, -1)
            for i in order:
                xa, xc = hw - 0.03 - i * sw, hw - 0.03 - (i + 1) * sw          # low end, high end (|x|)
                zt_ = Oz + (hw - xc) * tan(th) + (ct + 0.02) / cos(th) + 0.05
                ends = ((xa, xc + 0.001) if sgn < 0 else (xc + 0.001, xa))
                for xx in ends:
                    zb_ = Oz + (hw - xx) * tan(th) - 0.005
                    rings.append([(sgn * xx, -0.008, zb_), (sgn * xx, -0.001, zb_), (sgn * xx, -0.001, zt_), (sgn * xx, -0.008, zt_)])
        lead.append(loft("flash_upstand", rings))
        z_max = z_ap
    else:
        sl = roofs[0][0]
        fc = slab("flash_cover", sl, sl.S - 0.15, sl.s_top(ct + 0.02), -W / 2 + 0.006, W / 2 - 0.006, ct + 0.004 if cov != "lead" else 0.03, ct + 0.009 if cov != "lead" else 0.035)
        lead.append(fc)
        zt_ = sl.pt(sl.S, 0, 0).z + (ct + 0.01) / max(cos(top_th), 0.3)
        xu = W / 2 - 0.006
        lead.append(loft("flash_upstand", [[(x, -0.011, zt_ - 0.02), (x, -0.002, zt_ - 0.02), (x, -0.002, zt_ + 0.095), (x, -0.011, zt_ + 0.095)] for x in (-xu, 0.0, xu)]))
        z_max = zt_

    # ------------------------------------------------------------------ fretwork valance
    nails = []
    if p.valance != "none":
        r_v = rng.fork("valance")
        lap, vth = 0.04, 0.02
        vbw = 0.128
        vb = valance_board(p.valance, vbw - 0.004, lap, p.valance_drop, vth)
        vm = []

        def run(x0, y0, x1, y1, rot):
            L = hypot(x1 - x0, y1 - y0)
            n = max(1, round(L / vbw))
            for i in range(n):
                t = (i + 0.5) / n
                dz = r_v.u(-0.004, 0.002) - (0.012 * p.wear if r_v.chance(0.12) else 0.0)
                loc = (lerp(x0, x1, t), lerp(y0, y1, t), H + dz)
                vm.append(trs(loc, (90 + r_v.u(-0.5, 0.5), r_v.u(-1.0, 1.0) * (0.4 + p.wear), rot), (L / n / vbw, 1.0, 1.0)))
                nails.append((loc[0], loc[1], H + lap * 0.55 + dz, rot))

        run(-xo - vth, -PF, xo + vth, -PF, 0.0)
        run(xo, -PF, xo, -0.03, 90.0)
        run(-xo, -0.03, -xo, -PF, -90.0)
        vo = instances_to_mesh("valance", vb, vm, keep_base=False, seed=seed + 5)
        uv_keep(vo)
        trim.append(vo)
        nl_b = bolt("nail", 0.0045, 0.003, kind="pyramid")
        nm = []
        for (x, y, z, rot) in nails:
            off = Vector((0, -vth, 0)) if rot == 0.0 else Vector((math.copysign(vth, rot), 0, 0))
            nm.append(trs((x + off.x, y + off.y, z), (90, 0, rot)))
        irons.append(instances_to_mesh("nails", nl_b, nm, keep_base=False))

    # ------------------------------------------------------------------ hook, chain and lantern
    lit = p.lantern == "lantern"
    lz_b = H - 0.27                                # underside of the lantern (door heads are ~2.1 m)
    if p.lantern != "none":
        hk = [cylinder("hook_plate", 0.021, 0.005, loc=(0, y_l, z_hook - 0.0025), verts=10),
              cylinder("hook_shank", 0.0045, 0.02, loc=(0, y_l, z_hook - 0.013), verts=6),
              torus("hook_eye", 0.014, 0.004, loc=(0, y_l, z_hook - 0.034), rot=(0, 90, 0), major_seg=14, minor_seg=6)]
        irons += hk
        if not lit:
            socket("lantern_hook", "attach", (0, y_l, z_hook - 0.05), role="lantern")
    if lit:
        lw, lh = 0.15, 0.22
        hwl = lw / 2
        z0 = lz_b + 0.03
        zt = z0 + 0.008 + lh
        q = sqrt(2.0)
        lan = [box("l_base", (lw + 0.014, lw + 0.014, 0.008), loc=(0, y_l, z0 + 0.004), bevel=0.0015, segments=1),
               lathe("l_drip", [(0.0, -0.03), (0.007, -0.027), (0.011, -0.017), (0.006, -0.008), (0.02, 0.0)], segments=8, loc=(0, y_l, z0)),
               lathe("l_cap", [(0.0, 0.0), (hwl * 1.3 * q, 0.0), (hwl * 1.32 * q, 0.006), (hwl * 0.78 * q, 0.03), (hwl * 0.36 * q, 0.068), (0.022, 0.084),
                               (0.022, 0.098), (0.035, 0.102), (0.035, 0.108), (0.0, 0.12)], segments=4, loc=(0, y_l, zt), rot=(0, 0, 45), smooth_angle=30),
               torus("l_ring", 0.015, 0.0035, loc=(0, y_l, zt + 0.132), rot=(0, 90, 0), major_seg=12, minor_seg=6),
               cylinder("l_cup", 0.016, 0.03, loc=(0, y_l, z0 + 0.023), verts=8, radius2=0.011)]
        for ix_ in (-1, 1):
            for iy in (-1, 1):
                lan.append(box("l_bar", (0.009, 0.009, lh), loc=(ix_ * hwl, y_l + iy * hwl, z0 + 0.008 + lh / 2)))
            lan.append(box("l_rail", (lw, 0.008, 0.01), loc=(0, y_l + ix_ * hwl, zt - 0.005)))
            lan.append(box("l_rail", (0.008, lw, 0.01), loc=(ix_ * hwl, y_l, zt - 0.005)))
        glass = []
        # each pane is ONE glowing sheet (the outer face) fitted between the corner bars and under the top rail: a thin
        # glowing box is two sheets that fight each other. CLEARANCE RULE: nothing parallel within 4 mm of a glowing sheet
        for sgn in (-1, 1):
            glass.append(lod_faces(box("l_glass", (lw - 0.0092, 0.002, lh - 0.012), loc=(0, y_l + sgn * hwl, z0 + 0.004 + lh / 2)),
                                   keep=(0, sgn, 0), above=-1))
            glass.append(lod_faces(box("l_glass", (0.002, lw - 0.0092, lh - 0.012), loc=(sgn * hwl, y_l, z0 + 0.004 + lh / 2)),
                                   keep=(sgn, 0, 0), above=-1))
        for g in glass:
            assign(g, mat("glass_lantern", glow=1.6))
        flame = sphere("l_flame", 0.012, loc=(0, y_l, z0 + 0.065), segments=10, rings=6, scale=(1, 1, 2.1))
        assign(flame, mat("emissive", "#ffc27a", glow=9))
        z_ring = zt + 0.147
        z_eye = z_hook - 0.046
        gap = z_eye - z_ring
        nlk = 3 if gap > 0.16 else (1 if gap > 0.045 else 0)
        z_ch = z_ring + nlk * 0.03                 # top of the short chain
        for i in range(nlk):
            lan.append(torus("l_link", 0.0125, 0.003, loc=(0, y_l, z_ring + 0.006 + i * 0.03), rot=(0, 90, 90 * ((i + 1) % 2)), major_seg=8, minor_seg=4))
            transform_verts(lan[-1], lambda co: Vector((co.x * 1.5, co.y * 0.85, co.z)))
        if gap - nlk * 0.03 > 0.02:                # forged hanging rod with an eye at each end
            lan.append(sweep("l_rod", [(0, y_l, z_eye - 0.012), (0, y_l, z_ch + 0.018)], rounded_rect(0.007, 0.007, 0.001, 1), twist_deg=360.0 * (gap / 0.12),
                             step=0.03, smooth_path=False))
            for zz in (z_eye - 0.002, z_ch + 0.008):
                lan.append(torus("l_rod_eye", 0.011, 0.0032, loc=(0, y_l, zz), rot=(90, 0, 0), major_seg=8, minor_seg=4))
        for o in lan:
            assign(o, iron)
            paint(o, special=mask_mul(noise_mask(0.05, seed=seed + 8, lo=0.4, hi=0.8), 0.5), grime=0.25)
        part(lan + glass + [flame], "lantern", pivot=(0, y_l, z_hook - 0.034), motion="swing", amp=2.0, speed=0.22)
        light("lantern", (0, y_l, z0 + 0.07), intensity=p.intensity, range=6.0, flicker=0.3, shadow=True, shadow_near=0.2)

    # ------------------------------------------------------------------ materials and weathering
    wr = p.wear
    chip = noise_mask(0.07, seed=seed + 1, lo=0.56 - 0.25 * wr, hi=0.82 - 0.2 * wr)
    low = height_ramp(H - D - 0.2, H + 0.05)
    for o in frame:
        assign(o, frame_m)
        paint(o, wear=mask_mul(chip, 0.4 + 0.6 * wr), special=mask_mul(facing((0, -1, 0.4)), 0.3 + 0.5 * wr),
              grime=mask_mul(low, noise_mask(0.2, seed=seed + 2, lo=0.3, hi=0.8), 0.5))
    tips = height_ramp(H - p.valance_drop, H - p.valance_drop + 0.09)
    for o in trim:
        assign(o, painted)
        is_val = o.name.startswith("valance")
        paint(o, wear=mask_mul(chip, 0.45 + 0.55 * wr), special=mask_mul(facing((0, -1, 0.5)), 0.35 + 0.5 * wr),
              grime=mask_mul(tips, 0.3 + 0.5 * wr) if is_val else mask_mul(noise_mask(0.25, seed=seed + 6), 0.35))

    def soot(co, nrm):                             # lamp black on the boarding above the lantern
        d2 = (co[:, 0]) ** 2 + (co[:, 1] - y_l) ** 2
        return (0.8 if lit else 0.3) * np.exp(-d2 / 0.07) + 0.15

    for o in decks:
        assign(o, deck_m)
        paint(o, grime=soot)

    moss_n = noise_mask(0.24, seed=seed + 9, lo=0.42, hi=0.78)

    def moss(co, nrm):                             # damp side against the wall and the lower courses grow moss
        wallf = np.clip(1.0 + co[:, 1] / 0.55, 0.0, 1.0)
        return np.clip(moss_n(co, nrm) * (0.55 + 0.9 * wallf) * p.moss * 1.5, 0.0, 1.0)

    for o in cov_objs:
        paint(o, special=moss, grime=mask_mul(noise_mask(0.4, seed=seed + 12, lo=0.3, hi=0.8), 0.45), wear=mask_mul(chip, 0.3 * wr))
    for o in lead:
        assign(o, lead_m)
        paint(o, grime=mask_mul(noise_mask(0.15, seed=seed + 13, lo=0.3, hi=0.8), 0.55))
    for o in irons:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.04, seed=seed + 14, lo=0.3, hi=0.7), 0.35 + 0.5 * wr))

    socket("door", "attach", (0, 0, 0), role="door")
