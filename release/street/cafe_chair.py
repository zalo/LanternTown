"""Cafe terrace chair in three real constructions. 'bentwood': steam-bent beech in the Thonet manner - one
continuous rod forms both back legs and the back hoop, a second hoop nests inside it, a bent seat ring
carries a cane (or pressed plywood) seat, turned front legs screw into the ring and a stretcher ring ties
all four legs; two little bent knees brace the back. 'iron': forged round rod hoop and legs on a flat-bar
seat band, heart / lyre scrollwork of flat bar bound with collars, round slatted (or cane) seat riveted on
cross bars. 'folding': the slatted park chair - two flat-steel frames crossing on a pivot bolt, a hinged
seat on rails, bowed back slats screwed between the frame bars. The back follows the seat circle in plan
(as the real ones do) and everything is sized from the params.
"""
from math import sin, cos, tan, pi, radians, degrees, sqrt, hypot, atan2, exp
import numpy as np
from mathutils import Vector
from tk import *

META = {"name": "Cafe chair", "class": "furniture", "mount": "ground", "focus": 0.58}
PARAMS = {
    "style": {"default": "bentwood", "choices": ["bentwood", "iron", "folding"], "desc": "construction: bentwood hoop chair / wrought-iron bistro chair / folding slatted park chair"},
    "seat": {"default": "cane", "choices": ["cane", "wood"], "desc": "seat: woven cane, or wood (pressed panel on bentwood, slats on iron / folding)"},
    "seat_height": {"default": 0.46, "min": 0.42, "max": 0.5, "desc": "seat height (m)"},
    "seat_size": {"default": 0.4, "min": 0.36, "max": 0.45, "desc": "seat diameter / width (m)"},
    "back_height": {"default": 0.44, "min": 0.36, "max": 0.52, "desc": "height of the back above the seat (m)"},
    "rake": {"default": 9.0, "min": 4.0, "max": 15.0, "desc": "backward lean of the back (deg)"},
    "splay": {"default": 1.0, "min": 0.5, "max": 1.4, "desc": "how far the legs splay out (1 = normal)"},
    "wood": {"default": "#70482c", "desc": "wood tone (stain) of bentwood frame / unpainted slats"},
    "paint": {"default": "#1f3a2c", "desc": "paint colour of ironwork (and of the wood when 'painted')"},
    "painted": {"default": False, "desc": "paint the wood (bentwood frame, slats) in the paint colour instead of stain"},
    "cushion": {"default": False, "desc": "tie-on striped seat cushion"},
    "cushion_colour": {"default": "#8f2f28", "desc": "cushion stripe colour"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age: rubbed paint / varnish, rust, scuffed feet"},
}
VARIANTS = {
    "iron_heart": {"style": "iron", "seat": "wood", "paint": "#335a46", "wood": "#8a5a32", "seed": 4},
    "folding_park": {"style": "folding", "seat": "wood", "paint": "#8c9c88", "wood": "#96724a", "wear": 0.7, "seed": 8},
    "bentwood_black": {"style": "bentwood", "seat": "wood", "painted": True, "paint": "#17181b", "back_height": 0.5,
                       "cushion": True, "seat_size": 0.42, "wear": 0.75, "seed": 12},
    "iron_cream_cane": {"style": "iron", "seat": "cane", "paint": "#cfc6ae", "back_height": 0.4, "seat_size": 0.37,
                        "splay": 1.3, "wear": 0.9, "seed": 17},
    "folding_red": {"style": "folding", "seat": "cane", "painted": True, "paint": "#7e211d", "seat_size": 0.43,
                    "back_height": 0.4, "cushion": True, "cushion_colour": "#2f4a6a", "seed": 23},
}


# ------------------------------------------------------------------ helpers
def _aim(d):
    """Euler degrees that turn local +Z into direction d (for bolts, stamped parts)."""
    e = Vector((0, 0, 1)).rotation_difference(Vector(d).normalized()).to_euler()
    return (degrees(e.x), degrees(e.y), degrees(e.z))


def _interp(tbl):
    xs = [a for a, _ in tbl]
    ys = [b for _, b in tbl]
    return lambda t: float(np.interp(t, xs, ys))


def _ss(e0, e1, x):
    """Vectorised smoothstep (e0 > e1 gives a falling ramp)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _arch(hw0, hw1, h, top_r, s0=0.0, n_up=6, n_arc=20):
    """Inverted U in back-surface coords (x, s): feet (+-hw0, s0), shoulders +-hw1, crown (0, h)."""
    top_r = min(top_r, h * 0.62)
    hs = h - top_r
    up = [(lerp(hw0, hw1, smoothstep(0.0, 1.0, i / n_up)), lerp(s0, hs, i / n_up)) for i in range(n_up)]
    pts = [(-x, s) for x, s in up]
    for i in range(n_arc + 1):
        a = pi - pi * i / n_arc
        pts.append((hw1 * cos(a), hs + top_r * sin(a)))
    pts += [(x, s) for x, s in reversed(up)]
    return pts, hs


def _back_surface(Rr, z0, rake, h, xA, flatten=0.45):
    """The surface the back lies in: the seat circle (radius Rr) extruded upwards and leaning back by
    `rake`, flattening towards the top. Returns B(x, s, off) -> (x, y, z); s = distance up the back."""
    yA = sqrt(max(Rr * Rr - xA * xA, 1e-6))
    sr, cr = sin(radians(rake)), cos(radians(rake))

    def B(x, s, off=0.0):
        xx = clamp(x, -Rr * 0.985, Rr * 0.985)
        yc = sqrt(Rr * Rr - xx * xx)
        k = 1.0 - flatten * smoothstep(0.0, h, s)
        return (x, yA + k * (yc - yA) + off + s * sr, z0 + s * cr)
    return B


def _ring_through(pf, pb):
    """Circle centred on the Y axis through a front and a back leg point -> (centre y, radius)."""
    c = (pb[0] ** 2 + pb[1] ** 2 - pf[0] ** 2 - pf[1] ** 2) / (2.0 * (pb[1] - pf[1]))
    return c, hypot(pf[0], pf[1] - c)


def _rod(name, a, b, r, sides=8, n=1):
    """Straight round rod; n > 1 adds rings along it so baked AO / masks stay local on long rods."""
    a, b = Vector(a), Vector(b)
    return sweep(name, [tuple(a.lerp(b, i / n)) for i in range(n + 1)], r, sides=sides, smooth_path=False)


def _seg(a, b, step=0.075, end=False):
    n = max(1, int(round((b - a).length / step)))
    return [a.lerp(b, i / n) for i in range(n + (1 if end else 0))]


def _head(name, r, h, washer=0.0, segs=8, low=False):
    """Low-poly domed screw / bolt head sitting on z = 0, pointing +Z, optionally on a washer."""
    b = 0.0
    prof = [(r, 0.0)]
    if washer > r:
        b = 0.0013
        prof = [(washer, 0.0), (washer, b), (r, b)]
    prof += [(r * 0.8, b + h * 0.62), (0.0, b + h)] if low else [(r * 0.87, b + h * 0.55), (r * 0.5, b + h * 0.9), (0.0, b + h)]
    return lathe(name, prof, segments=segs)


def _stamp(name, base, mats, material):
    if not mats:
        delete(base)
        return None
    assign(base, material)
    return instances_to_mesh(name, base, mats, keep_base=False)


def _cane_weave(name, clip, z, rng, sag=None, pitch=0.015, nseg=None, broken=0, underside=True):
    """Open hand-caned mesh (the classic six-way pattern): paired warp and weft strands with two sets
    of diagonals laid over them, each strand a thin flat strip (top + underside, 2-4 spans - cheap).
    clip(c, ang) -> (t0, t1) of the line at perpendicular offset c running in direction ang (radians)
    inside the seat opening, or None. sag(x, y) -> how far the pan has dropped there. `broken`
    diagonals are snapped: one end droops, the other has sprung up."""
    verts, faces = [], []
    th = 0.0009

    def strip(c, ang, ta, tb, zz, sw, la=0.0, lb=0.0):
        L = tb - ta
        if L < 0.014:
            return
        dx, dy = cos(ang), sin(ang)
        nx, ny = -dy, dx
        n = nseg or (2 if L < 0.17 else 3)
        secs = [(i / n, 0.0) for i in range(n + 1)]
        if la:
            secs[0] = (0.0, la)
            secs.insert(1, (min(0.03 / L, 0.3), la * 0.12))
        if lb:
            secs[-1] = (1.0, lb)
            secs.insert(len(secs) - 1, (1.0 - min(0.03 / L, 0.3), lb * 0.12))
        base = len(verts)
        hx, hy = nx * sw * 0.5, ny * sw * 0.5
        for tt, dz in secs:
            t = ta + L * tt
            x, y = c * nx + t * dx, c * ny + t * dy
            zc = zz + dz - (sag(x, y) if sag else 0.0)
            if underside:                                   # two faces back to back: upper first, then lower
                verts.extend([(x - hx, y - hy, zc + th), (x + hx, y + hy, zc + th)])
                verts.extend([(x + hx, y + hy, zc), (x - hx, y - hy, zc)])
            else:
                verts.extend([(x + hx, y + hy, zc + th), (x - hx, y - hy, zc + th)])
        for i in range(len(secs) - 1):
            q = base + 4 * i if underside else base + 2 * i - 2
            if underside:
                faces.append((q + 3, q + 2, q + 6, q + 7))
            faces.append((q + 1, q, q + 4, q + 5) if underside else (q + 2, q + 3, q + 5, q + 4))   # top (normal up)

    N = 40
    for k in range(-N, N):
        for d in (-0.0024, 0.0024):
            for ang, zz in ((0.0, z), (pi * 0.5, z + th * 0.9)):
                tr = clip((k + 0.5) * pitch + d, ang)
                if tr:
                    strip((k + 0.5) * pitch + d, ang, tr[0], tr[1], zz, 0.0029)
    dp = pitch * 0.7071
    diag = []
    for k in range(-2 * N, 2 * N):
        for ang, zz in ((pi * 0.25, z + th * 1.8), (pi * 0.75, z + th * 2.6)):
            tr = clip(k * dp, ang)
            if tr:
                diag.append((k * dp, ang, tr[0], tr[1], zz))
    long_ones = [i for i, s in enumerate(diag) if s[3] - s[2] > 0.14]
    snapped = {}
    for _ in range(min(broken, len(long_ones))):
        snapped[long_ones[int(rng.u(0, len(long_ones) - 0.001))]] = (rng.u(0.3, 0.7), rng.u(0.009, 0.017))
    for i, (c, ang, t0, t1, zz) in enumerate(diag):
        if i in snapped:
            f, up = snapped[i]
            tb = lerp(t0, t1, f)
            strip(c, ang, t0, tb - 0.012, zz, 0.0027, lb=-0.006)
            strip(c, ang, tb + 0.004, t1, zz, 0.0027, la=up)
        else:
            strip(c, ang, t0, t1, zz, 0.0027)
    return mesh_object(name, verts, faces)


def _clip_circle(r):
    return lambda c, ang: (-sqrt(r * r - c * c), sqrt(r * r - c * c)) if abs(c) < r else None


def _clip_rect(a, b):
    def clip(c, ang):
        dx, dy = cos(ang), sin(ang)
        px, py = -c * dy, c * dx
        t0, t1 = -1e9, 1e9
        for p_, d_, h_ in ((px, dx, a), (py, dy, b)):
            if abs(d_) < 1e-9:
                if abs(p_) > h_:
                    return None
            else:
                ta, tb = (-h_ - p_) / d_, (h_ - p_) / d_
                t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        return (t0, t1) if t1 > t0 else None
    return clip


def _cane_soil(cx, cy, rad, w, seed):
    """Sat-on cane: a darker, greyer patch where the sitter lands, broken up by noise."""
    nz = noise_mask(0.05, seed=seed, lo=0.2, hi=0.8)
    return lambda co, nrm: np.clip(np.exp(-(np.hypot(co[:, 0] - cx, co[:, 1] - cy) / rad) ** 2) * (0.5 + 0.5 * nz(co, nrm))
                                   * (0.35 + 0.6 * w), 0, 1)


def _cane_pan(Ri, zt, sag, M, rng, p):
    """Hand-caned round seat pegged into the ring + the spline bead that covers the holes."""
    w = p.wear
    sagf = lambda x, y: sag * max(0.0, 1 - (x * x + y * y) / (Ri * Ri))
    # the pan is only ever seen from above: strands carry a top face only
    pan = _cane_weave("cane_seat", _clip_circle(Ri), zt - 0.0035, rng, sagf,
                      broken=int(max(0.0, w - 0.55) * 11 + rng.u(0, 1)), underside=False)
    assign(pan, M["cane"])
    bead = torus("cane_spline", Ri - 0.0045, 0.0032, loc=(0, 0, zt + 0.0008), major_seg=36, minor_seg=5)
    assign(bead, M["cane_bead"])
    paint(pan, grime=_cane_soil(rng.jitter(0, 0.015), -0.02 + rng.jitter(0, 0.015), Ri * 0.62, w, p.seed + 41))
    paint(bead, grime=0.25 + 0.3 * w)
    return [pan, bead]


def _cushion(p, rng, z, round_r=None, size=None, y=0.0, tie_x=0.1, tie_y=0.2):
    """Tie-on striped canvas cushion: two panels sewn round a piped seam (puckered), sat-in dent,
    tufting button, two pairs of ties at the back; soiled in the middle and along the seam."""
    w = p.wear
    cloth = mat("canvas", p.cushion_colour, stripe=0.05, tint2="#b8ad94")
    pipe = mat("canvas", p.cushion_colour)
    t = rng.u(0.035, 0.04)
    if round_r:
        a = b = round_r
        ne = 2.0
    else:
        a, b = size[0] * 0.5, size[1] * 0.5
        ne = 5.0
    hw = min(a, b)

    def rho(th):                              # outline radius in direction th (circle / rounded square)
        return (abs(cos(th) / a) ** ne + abs(sin(th) / b) ** ne) ** (-1.0 / ne)

    nseg, npk = 40, 10
    ph = rng.u(0, 6.28)
    dent = 0.007 + 0.006 * w
    cx, cy = rng.jitter(0, 0.012), -0.015 + rng.jitter(0, 0.01)
    yaw = radians(rng.jitter(0, 4.0))
    prof = [(0, 0.004), (0.6, 0.001), (0.88, 0.002), (0.965, t * 0.24), (0.992, t * 0.5), (0.965, t * 0.76),
            (0.88, t * 0.98), (0.6, t * 1.07), (0.3, t * 1.03), (0, t * 0.98)]
    pad = lathe("cushion", prof, segments=nseg)

    def top_drop(x, yy):
        return dent * exp(-(hypot(x - cx, yy - cy) / (0.62 * hw)) ** 2) + 0.005 * exp(-(hypot(x, yy) / 0.028) ** 2)

    def pucker(th):
        return sin(npk * th + ph) + 0.4 * sin(2 * npk * th + 1.7 * ph)

    def shape(co):
        f = hypot(co.x, co.y)
        th = atan2(co.y, co.x) + yaw
        e = smoothstep(0.7, 0.97, f)
        r = f * rho(th) * (1.0 + 0.008 * e * pucker(th))
        x, yy = cos(th) * r, sin(th) * r
        topf = smoothstep(0.3 * t, 0.7 * t, co.z)
        zz = co.z - topf * top_drop(x, yy) + (2 * topf - 1) * 0.0024 * e * (1 - smoothstep(0.975, 1.0, f)) * pucker(th)
        return Vector((x, yy + y, zz + z))
    transform_verts(pad, shape)
    displace_noise(pad, strength=0.002, scale=0.06, seed=p.seed + 31)
    smooth(pad, 70)
    uv_planar(pad, (cos(yaw), sin(yaw), 0), (-sin(yaw), cos(yaw), 0))
    assign(pad, cloth)
    rim = []
    for k in range(nseg):
        th = 2 * pi * k / nseg + yaw
        r = rho(th) * (1.004 + 0.008 * pucker(th))
        rim.append((cos(th) * r, sin(th) * r + y, z + t * 0.5 + 0.0012 * sin(3 * th + ph)))
    piping = sweep("cushion_piping", rim, 0.0036, sides=4, closed=True, smooth_path=False)
    assign(piping, pipe)
    btn = sphere("cushion_button", 0.0105, loc=(0, y, z + t * 0.98 - top_drop(0, 0) + 0.0015), segments=8, rings=5, scale=(1, 1, 0.5))
    assign(btn, pipe)
    out = [pad, piping, btn]
    tie = [(-0.0055, -0.001), (0.0055, -0.001), (0.0055, 0.001), (-0.0055, 0.001)]
    for sx in (-1, 1):
        for k, (dx, dz) in enumerate(((0.035, -0.1), (-0.02, -0.075))):
            a0 = (sx * tie_x, tie_y - 0.02, z + t * 0.5)
            m = (sx * (tie_x + dx * 0.6), tie_y + 0.022, z + 0.005)
            e = (sx * (tie_x + dx) + rng.jitter(0, 0.008), tie_y + 0.03 + rng.jitter(0, 0.008), z + dz + rng.jitter(0, 0.01))
            s = sweep(f"cushion_tie{sx}{k}", [a0, m, (m[0], m[1] + 0.004, m[2] - 0.03), e], tie, up=(sx, 0, 0), step=0.02)
            assign(s, pipe)
            out.append(s)
    nz = noise_mask(0.06, seed=p.seed + 5, lo=0.2, hi=0.8)

    def soil(co, nrm):
        d = np.hypot(co[:, 0] - cx, co[:, 1] - y - cy) / hw
        e = (np.abs(co[:, 0] / a) ** ne + np.abs((co[:, 1] - y) / b) ** ne) ** (1.0 / ne)
        g = np.maximum(np.exp(-(d / 0.55) ** 2) * 0.8, _ss(0.72, 1.0, e) * 0.65)
        return np.clip(g * (0.5 + 0.5 * nz(co, nrm)) * (0.5 + 0.6 * w), 0, 1)
    for o in out:
        paint(o, special=mask_mul(facing((0, 0, 1)), 0.3 + 0.35 * w), grime=soil)
    return out


# ------------------------------------------------------------------ bentwood (Thonet) chair
def _bentwood(p, rng, M):
    zs, R, bh, sp, w = p.seat_height, p.seat_size * 0.5, p.back_height, p.splay, p.wear
    wood = M["wood"]
    rh, rw, rr, ri, zf = 0.030, 0.038, 0.0155, 0.0105, 0.008
    frame, scr, blt = [], [], []
    # ---- bent seat ring: flat rounded-rectangle section, 38 wide x 30 deep
    rt = R - rw * 0.5
    nr = 44
    frame.append(sweep("seat_ring", [(rt * cos(2 * pi * k / nr), rt * sin(2 * pi * k / nr), zs - rh * 0.5) for k in range(nr)],
                       rounded_rect(rw, rh, 0.008, 2), closed=True, smooth_path=False))

    # ---- one rod: back leg - hoop - back leg, lying on the curved, raked back surface; the legs are
    #      let 9 mm into a housing in the ring and bolted through from outside
    z0 = zs - rh * 0.5
    Rr = R + rr - 0.009
    xA = Rr * sin(radians(47))
    rake = p.rake + rng.jitter(0, 0.6)
    s_c = (zs + bh - rr - z0) / cos(radians(rake))
    B = _back_surface(Rr, z0, rake, s_c, xA)
    yA = B(xA, 0)[1]

    def leg_pt(sx, u):                       # u = 0 at the seat, 1 at the floor; sabre curve
        e = u ** 1.6
        return (sx * (xA + 0.042 * sp * e), yA + 0.115 * sp * e, lerp(z0, zf, u))

    hw1 = xA + 0.016
    arch, hs = _arch(xA, hw1, s_c, hw1)
    us = (1.0, 0.8, 0.6, 0.4, 0.2)
    pts = [leg_pt(-1, u) for u in us] + [B(x, s) for x, s in arch] + [leg_pt(1, u) for u in reversed(us)]
    ts = 1.0 - path_length([leg_pt(1, u) for u in (0.0,) + tuple(reversed(us))]) / (path_length(pts) * 0.5)

    def hoop_sc(t):
        tt = abs(2 * t - 1)
        return lerp(0.86, 1.0, smoothstep(0.0, ts, tt)) if tt < ts else lerp(1.0, 0.8, (tt - ts) / (1 - ts))
    frame.append(sweep("back_hoop", pts, rr, sides=10, step=0.026, scale_fn=hoop_sc))
    for sx in (-1, 1):
        d = Vector((sx * xA, yA, 0)).normalized()
        blt.append(trs((sx * xA + d.x * (rr - 0.0012), yA + d.y * (rr - 0.0012), z0), _aim(d)))

    # ---- inner hoop housed in the back of the seat ring, bolted
    xi0 = Rr * sin(radians(25))
    off_i = (R + ri - 0.0045) - Rr
    arch2, _ = _arch(xi0, xi0 + 0.018, s_c - 0.085, xi0 + 0.018, s0=-0.012, n_up=5, n_arc=16)
    frame.append(sweep("inner_hoop", [B(x, s, off_i) for x, s in arch2], ri, sides=8, step=0.024,
                       scale_fn=lambda t: 0.88 + 0.12 * abs(2 * t - 1)))
    for sx in (-1, 1):
        q = B(sx * xi0, 0.0, off_i)
        d = Vector((q[0], q[1], 0)).normalized()
        blt.append(trs((q[0] + d.x * (ri - 0.001), q[1] + d.y * (ri - 0.001), z0 - 0.001), _aim(d), 0.85))

    # ---- bent knees bracing the back uprights to the seat ring (bolt through the upright, screw into the ring)
    for sx in (-1, 1):
        s_b = min(0.16, hs * 0.8)
        U = Vector(B(sx * lerp(xA, hw1, smoothstep(0.0, 1.0, s_b / hs)), s_b))
        ag = radians(92)
        G = Vector((sx * rt * sin(ag), rt * cos(ag), zs + 0.002))
        th = Vector((G.x - U.x, G.y - U.y, 0)).normalized()
        P0 = U + th * (rr * 0.55)
        knee = bezier_path(tuple(P0), tuple(P0 + Vector((0, 0, -0.085)) + th * 0.012), tuple(G - th * 0.065 + Vector((0, 0, 0.014))), tuple(G), n=13)
        frame.append(sweep(f"knee{sx}", knee, 0.0078, sides=6, smooth_path=False, scale_fn=lambda t: 1.0 - 0.18 * sin(pi * t)))
        blt.append(trs(tuple(G - th * 0.013 + Vector((0, 0, 0.0092))), (0, 0, 0), 0.85))
        blt.append(trs(tuple(U - th * (rr * 0.93 - 0.001)), _aim(-th)))

    # ---- turned front legs tenoned into the ring from below
    af = radians(40)
    r_fl = 0.015
    fprof = _interp([(0, 1.18), (0.016, 1.18), (0.03, 1.0), (0.045, 1.13), (0.06, 0.98), (0.11, 1.0), (1.0, 0.74)])
    fts = (0, 0.016, 0.03, 0.0375, 0.045, 0.0525, 0.06, 0.11, 0.3, 0.5, 0.7, 0.86, 0.94, 1.0)
    rtf = rt - 0.002
    rf = rtf + 0.052 * sp
    legs_f = []
    for sx in (-1, 1):
        top = Vector((sx * rtf * sin(af), -rtf * cos(af), zs - rh + 0.004))
        foot = Vector((sx * rf * sin(af), -rf * cos(af) - 0.014 * sp, zf))
        frame.append(sweep(f"front_leg{sx}", [tuple(top.lerp(foot, t)) for t in fts], r_fl, sides=10, smooth_path=False, scale_fn=fprof))
        legs_f.append((top, foot))

    # ---- leg ring: clearly smaller than the seat, 140 mm below it, screwed to the inside of all four legs
    zr = zs - 0.14
    top, foot = legs_f[1]
    tf = (top.z - zr) / (top.z - zf)
    pf = top.lerp(foot, tf)
    ub = (z0 - zr) / (z0 - zf)
    pb = leg_pt(1, ub)
    r_st = 0.008
    rin_f = r_fl * fprof(tf) + r_st - 0.003
    rin_b = rr * 0.9 + r_st - 0.003
    df, db = Vector((pf.x, pf.y)).normalized(), Vector((pb[0], pb[1])).normalized()
    cf = (pf.x - df.x * rin_f, pf.y - df.y * rin_f)
    cb = (pb[0] - db.x * rin_b, pb[1] - db.y * rin_b)
    c, major = _ring_through(cf, cb)
    frame.append(torus("stretcher", major, r_st, loc=(0, c, zr), major_seg=40, minor_seg=6))
    for q in (cf, (-cf[0], cf[1]), cb, (-cb[0], cb[1])):
        d = Vector((q[0], q[1] - c, 0)).normalized()
        scr.append(trs((d.x * (major - r_st + 0.0008), c + d.y * (major - r_st + 0.0008), zr), _aim(-d)))

    # ---- small nail-on steel glides under the feet
    glide = lathe("glide", [(0, 0), (0.0095, 0), (0.0118, 0.002), (0.0118, 0.0062), (0, 0.0075)], segments=8)
    feet = [leg_pt(-1, 1.0), leg_pt(1, 1.0)] + [tuple(f) for _, f in legs_f]
    gl = _stamp("glides", glide, [trs((f[0], f[1], 0)) for f in feet], M["glide"])
    paint(gl, special=0.3 + 0.5 * w, grime=0.7)

    # ---- the wear story: varnish rubbed through to pale wood on the hoop crown (hands), the front of the
    #      seat ring (legs / trousers), the top of the leg ring (shoes); feet scuffed, chipped and dirty
    zt_ = zs + bh
    kw = clamp(0.3 + 0.9 * w)
    nzA = noise_mask(0.04, seed=p.seed + 11, lo=0.2, hi=0.8)
    nzC = noise_mask(0.017, seed=p.seed + 12, lo=0.6, hi=0.74)
    nzF = noise_mask(0.012, seed=p.seed + 13, lo=0.35, hi=0.65)
    chip = (0.1 + 0.6 * w) if M["painted"] else (0.03 + 0.22 * w)

    def wearf(co, nrm):
        y, z = co[:, 1], co[:, 2]
        brk = 0.68 + 0.32 * nzA(co, nrm)
        crown = _ss(zt_ - 0.15, zt_ - 0.035, z) * np.clip(0.5 + 0.7 * np.maximum(nrm[:, 2], -nrm[:, 1]), 0, 1)
        seatf = _ss(0.03, 0.006, np.abs(z - (zs - 0.004))) * _ss(-0.05, -0.15, y) * np.clip(0.3 + 0.9 * np.maximum(nrm[:, 2], -nrm[:, 1]), 0, 1)
        stret = _ss(0.016, 0.003, np.abs(z - zr)) * np.clip(0.8 + 1.2 * nrm[:, 2], 0, 1) * (0.6 + 0.4 * _ss(0.1, -0.08, y))
        feet_ = _ss(0.07, 0.012, z) * (0.25 + 0.75 * nzF(co, nrm))
        m = np.maximum.reduce([crown, seatf, stret]) * brk * kw
        return np.clip(np.maximum(m, feet_ * (0.4 + 0.6 * w)) + nzC(co, nrm) * chip, 0, 1)

    def grimef(co, nrm):
        z = co[:, 2]
        return np.clip(np.maximum(_ss(0.042, 0.004, z) * (0.6 + 0.4 * nzF(co, nrm)), _ss(0.3, 0.0, z) * (0.12 + 0.25 * w)), 0, 1)
    for o in frame:
        assign(o, wood)
        paint(o, wear=wearf, grime=grimef)

    # ---- seat
    out = list(frame)
    if p.seat == "cane":
        out += _cane_pan(R - rw + 0.004, zs - 0.003, 0.009 + rng.u(0, 0.006), M, rng.fork("cane"), p)
        z_top = zs
    else:
        Ro = R - 0.007
        zt = zs + 0.005
        dish = lambda r: 0.011 * (1 - (r / Ro) ** 2) - (0.0016 if 0.66 < r / Ro < 0.72 else 0.0)
        rs = [0, 0.25, 0.5, 0.655, 0.665, 0.715, 0.725, 0.87]
        prof = [(0, zs - 0.012)] + [(Ro * 0.9, zs - 0.001)] + [(Ro - 0.0015, zs), (Ro, zs + 0.002), (Ro - 0.001, zt), (Ro - 0.006, zt - dish(Ro - 0.006) + 0.0012)] + \
               [(Ro * k, zt - dish(Ro * k)) for k in reversed(rs)]
        pan = lathe("ply_seat", prof, segments=32)
        uv_grain(pan, axis="Y")
        assign(pan, M["seat_wood"])
        nzS = noise_mask(0.05, seed=p.seed + 21, lo=0.2, hi=0.8)
        paint(pan, wear=lambda co, nrm: np.clip(np.maximum(
                       np.exp(-(np.hypot(co[:, 0], co[:, 1] + 0.03) / (Ro * 0.6)) ** 2) * (0.45 + 0.55 * nzS(co, nrm)) * (0.35 + 0.65 * w),
                       _ss(Ro - 0.02, Ro, np.hypot(co[:, 0], co[:, 1])) * _ss(0.05, -0.1, co[:, 1]) * kw)
                       + nzC(co, nrm) * (0.1 + 0.5 * w), 0, 1))
        nn = 0 if p.cushion else 16                      # the cushion covers the nail ring
        if nn:
            out.append(_stamp("seat_nails", _head("nail", 0.0036, 0.002, segs=6, low=True),
                              [trs(((Ro - 0.012) * cos(2 * pi * (k + 0.5) / nn), (Ro - 0.012) * sin(2 * pi * (k + 0.5) / nn),
                                    zt - dish(Ro - 0.012) - 0.0004)) for k in range(nn)], M["screw"]))
        out.append(pan)
        z_top = zt
    for nm, base, mats in (("screws", _head("screw", 0.0042, 0.0026, segs=6), scr),
                           ("bolts", _head("bolt", 0.0052, 0.0036, washer=0.0082, segs=8), blt)):
        o = _stamp(nm, base, mats, M["screw"])
        paint(o, wear=0.7, grime=0.1)
        out.append(o)
    if p.cushion:
        out += _cushion(p, rng.fork("cushion"), z_top - 0.005, round_r=R - 0.012, tie_x=xA - 0.03, tie_y=yA + 0.035)
    return z_top


# ------------------------------------------------------------------ wrought-iron bistro chair
def _iron(p, rng, M):
    zs, R, bh, sp, w = p.seat_height, p.seat_size * 0.5, p.back_height, p.splay, p.wear
    iron = M["iron"]
    rr, zf, ts_ = 0.008, 0.007, 0.013          # rod radius, foot pad height, seat thickness
    frame, col = [], []
    zb = zs - ts_                               # top of the seat band
    bw = 0.03
    frame.append(ring_band("seat_band", radius=R - 0.0025, width=bw, thick=0.005, loc=(0, 0, zb - bw * 0.5), segments=40))

    z0 = zb - bw * 0.5
    Rr = R + rr - 0.002
    xA = Rr * sin(radians(48))
    rake = p.rake + rng.jitter(0, 0.6)
    s_c = (zs + bh - rr - z0) / cos(radians(rake))
    B = _back_surface(Rr, z0, rake, s_c, xA)
    yA = B(xA, 0)[1]

    def leg_pt(sx, u, x0, y0, dx, dy):
        e = u ** 1.7
        return (sx * (x0 + dx * e), y0 + dy * e, lerp(z0, zf, u))

    hw1 = xA + 0.012
    arch, hs = _arch(xA, hw1, s_c, hw1 * 0.96)
    us = (1.0, 0.86, 0.72, 0.58, 0.44, 0.3, 0.16)
    bl = lambda sx, u: leg_pt(sx, u, xA, yA, 0.03 * sp, 0.105 * sp)
    frame.append(sweep("back_hoop", [bl(-1, u) for u in us] + [B(x, s) for x, s in arch] + [bl(1, u) for u in reversed(us)],
                       rr, sides=7, step=0.024))
    # front legs welded to the outside of the band
    af = radians(40)
    xF, yF = Rr * sin(af), -Rr * cos(af)
    fl = lambda sx, u: leg_pt(sx, u, xF, yF, 0.05 * sp * sin(af), -0.06 * sp * cos(af))
    for sx in (-1, 1):
        frame.append(sweep(f"front_leg{sx}", [(sx * xF, yF, zb - 0.002)] + [fl(sx, u) for u in (0.0, 0.16, 0.3, 0.44, 0.58, 0.72, 0.86, 1.0)],
                           rr, sides=7, step=0.03))
    # stretcher ring
    zr = zs * 0.36
    u_r = (z0 - zr) / (z0 - zf)
    pf, pb = fl(1, u_r), bl(1, u_r)
    c, Rc = _ring_through(pf, pb)
    r_st = 0.006
    major = Rc - rr - r_st + 0.003
    frame.append(torus("stretcher", major, r_st, loc=(0, c, zr), major_seg=36, minor_seg=6))
    # forged pad feet
    pad = lathe("pad", [(0, 0), (0.0145, 0), (0.0165, 0.002), (0.0165, 0.005), (0.0095, 0.009), (0.0085, 0.013), (0, 0.013)], segments=10)
    feet = [bl(-1, 1.0), bl(1, 1.0), fl(-1, 1.0), fl(1, 1.0)]
    frame.append(_stamp("pads", pad, [trs((f[0], f[1], 0), (0, 0, rng.u(0, 90))) for f in feet], iron))

    # ---- heart / lyre scrollwork of flat bar, bound to the hoop with collars
    flatbar = rounded_rect(0.006, 0.014, 0.0016, 1)
    hwi = hw1 - rr - 0.004
    r0 = hwi * 0.45
    sv = min(s_c * 0.74, s_c - 0.07 - 0.0)
    C = (hwi - r0, max(sv - r0, 0.06 + r0 * 0.3))
    s_tip = 0.045
    amax = 2 * pi * 1.3
    vr = lambda a: r0 * (1.0 - 0.8 * (a / amax) ** 0.85)
    bez = bezier_path((0.0, s_tip, 0), (0.085, s_tip + 0.03, 0), (C[0] + r0, max(C[1] - 0.12, s_tip + 0.04), 0), (C[0] + r0, C[1], 0), n=10)
    vol = [(C[0] + vr(a) * cos(a), C[1] + vr(a) * sin(a)) for a in [amax * (k + 1) / 30 for k in range(30)]]
    half = [(q[0], q[1]) for q in bez] + vol
    taper = lambda t: 1.0 if t < 0.72 else lerp(1.0, 0.55, (t - 0.72) / 0.28)
    for sx in (-1, 1):
        frame.append(sweep(f"scroll{sx}", [B(sx * x, s) for x, s in half], flatbar, up=(0, 1, 0.15), smooth_path=False, scale_fn=taper))
    rc = C[0] - vr(pi) - 0.005
    if rc > 0.012:
        frame.append(sweep("lyre_ring", [B(rc * cos(2 * pi * k / 20), C[1] + rc * sin(2 * pi * k / 20)) for k in range(20)],
                           flatbar, up=(0, 1, 0.15), closed=True, smooth_path=False))
    # stem + ball joining the heart tip to the back of the seat band
    tip = B(0, s_tip)
    frame.append(_rod("heart_stem", (0, R + 0.001, zb - 0.008), (tip[0], tip[1], tip[2] + 0.004), 0.006, sides=6, n=2))
    frame.append(sphere("heart_ball", 0.0115, loc=tip, segments=10, rings=6))
    if rc > 0.012 and C[1] - rc - s_tip > 0.06:
        # twisted square bar rising from the heart tip to the ring
        n_tw = 22
        frame.append(sweep("twist_bar", [B(0, lerp(s_tip, C[1] - rc, i / n_tw)) for i in range(n_tw + 1)], rounded_rect(0.0085, 0.0085, 0.0008, 1),
                           up=(0, 1, 0.15), smooth_path=False, twist_deg=360.0 * round((C[1] - rc - s_tip) / 0.07)))
        frame.append(sphere("twist_knop", 0.0095, loc=B(0, (s_tip + C[1] - rc) * 0.5), segments=10, rings=6, scale=(1, 1, 1.5)))

    def collar(x, s, wdt):
        a, b = B(x - 0.004, s), B(x + 0.004, s)
        q = B(x, s)
        col.append(trs(q, (-rake, 0, degrees(atan2(b[1] - a[1], b[0] - a[0]))), (wdt / 0.03, 1, 1)))
    for sx in (-1, 1):
        collar(sx * (hwi + hw1) * 0.5, C[1], hw1 - hwi + rr + 0.014)
        if rc > 0.012:
            collar(sx * (rc + 0.0025), C[1], 0.02)
    cb = rounded_box("collar", (0.03, 0.019, 0.012), radius=0.0075, seg=2, bevel=0.0015)
    frame.append(_stamp("collars", cb, col, iron))

    # ---- seat
    out = [o for o in frame if o]
    if p.seat == "cane":
        out += _cane_pan(R - 0.006, zs - 0.003, 0.008 + rng.u(0, 0.005), M, rng.fork("cane"), p)
        frame.append(torus("seat_rim", R - 0.004, 0.0065, loc=(0, 0, zs - 0.0065), major_seg=40, minor_seg=6))
    else:
        Rs = R + 0.005
        n = max(5, int(round(2 * Rs / 0.052)))
        pitch = 2 * Rs / n
        rv, r_s = [], rng.fork("slats")
        yb_ = R * 0.47
        for k in range(n):
            x0 = max(-Rs + pitch * k + 0.0045, -Rs + 0.012)
            x1 = min(-Rs + pitch * (k + 1) - 0.0045, Rs - 0.012)
            xs = [lerp(x0, x1, i / 4) for i in range(5)]
            poly = [(x, -sqrt(Rs * Rs - x * x)) for x in xs] + [(x, sqrt(Rs * Rs - x * x)) for x in reversed(xs)]
            s = extrude_profile(f"slat{k}", poly, ts_ - 0.001, loc=(0, 0, zb + 0.001 + r_s.jitter(0, 0.0008)), bevel=0.003)
            uv_grain(s, axis="Y")
            assign(s, M["slat"])
            tone = r_s.u(0.0, 0.3)
            paint(s, wear=mask_mul(lambda co, nrm: np.clip(1 - np.hypot(co[:, 0], co[:, 1] + 0.04) / (R * 0.9), 0, 1) ** 0.6,
                                   noise_mask(0.05, seed=p.seed + k, lo=0.25, hi=0.7), 0.35 + 0.65 * w),
                  grime=lambda co, nrm, tone=tone: np.maximum(tone, _ss(0.5, 0.9, np.abs(nrm[:, 0])) * 0.85),   # dirt in the gaps
                  special=mask_mul(facing((0, 0, 1)), (0.3 + 0.5 * w) if M["painted"] else 0.0))
            out.append(s)
            xm = (x0 + x1) * 0.5
            if sqrt(max(Rs * Rs - xm * xm, 0)) > yb_ + 0.02:
                rv += [trs((xm, sy * yb_, zs + r_s.jitter(0, 0.0004))) for sy in (-1, 1)]
        for sy in (-1, 1):
            frame.append(box("seat_bar", (2 * sqrt(R * R - yb_ * yb_) - 0.004, 0.02, 0.005), loc=(0, sy * yb_, zb - 0.0015), bevel=0.001))
        out.append(_stamp("rivets", _head("rivet", 0.0045, 0.0024, segs=6, low=True), rv, M["screw"]))
    nzC = noise_mask(0.017, seed=p.seed + 12, lo=0.6, hi=0.74)
    for o in frame:
        if not o:
            continue
        if o not in out:
            out.append(o)
        assign(o, iron)
        if o.name.startswith(("back_hoop", "scroll", "front_leg", "lyre", "stretcher")):
            displace_noise(o, strength=0.0006, scale=0.04, seed=p.seed + 3)
        paint(o, special=lambda co, nrm: np.clip(np.maximum.reduce([
                  np.clip(1 - co[:, 2] / 0.16, 0, 1),                                             # feet in the wet
                  np.clip(1 - np.abs(co[:, 2] - (zb - 0.02)) / 0.035, 0, 1) * 0.7,                # under the seat band
                  np.clip(1 - np.abs(co[:, 2] - zr) / 0.015, 0, 1) * 0.5]), 0, 1) * (0.15 + 0.85 * w),
              wear=lambda co, nrm: np.clip(np.maximum(np.clip((co[:, 2] - (zs + bh - 0.09)) / 0.08, 0, 1),
                                                      np.clip(1 - np.abs(co[:, 2] - zr) / 0.012, 0, 1) * np.clip(nrm[:, 2], 0, 1)) * (0.2 + 0.7 * w)
                                           + nzC(co, nrm) * (0.15 + 0.6 * w), 0, 1),
              grime=mask_mul(height_ramp(0.0, 0.25), 0.6))
    if p.cushion:
        _cushion(p, rng.fork("cushion"), zs - 0.004, round_r=R - 0.01, tie_x=xA - 0.03, tie_y=yA + 0.03)
    return zs


# ------------------------------------------------------------------ folding slatted park chair
def _folding(p, rng, M):
    zs, W, bh, sp, w = p.seat_height, p.seat_size, p.back_height, p.splay, p.wear
    iron = M["iron"]
    H = zs + bh
    xl = W * 0.5                     # outer frame (back + front legs)
    xi = xl - 0.0115                 # inner frame (rear legs)
    xr = xl - 0.0235                 # seat rails
    bar = rounded_rect(0.005, 0.025, 0.0016, 1)
    zh = zs - 0.03                   # hinge height
    Ff = Vector((0, -0.165 - 0.04 * sp, 0.015))
    Fb = Vector((0, 0.18 + 0.045 * sp, 0.015))
    Pr = Vector((0, 0.125, zh))
    Pf = Vector((0, -0.125, zh))
    brake = radians(p.rake + 6.0 + rng.jitter(0, 0.5))
    T = Vector((0, Pr.y + (H - zh) * tan(brake), H))
    d_up = (T - Pr).normalized()
    n_f = Vector((0, -d_up.z, d_up.y))           # back plane normal, towards the sitter
    Lb = (T - Pr).length
    rc = 0.05
    frame, scr, wood_parts = [], [], []

    def X(v, x):
        return (x, v.y, v.z)

    # ---- outer frame: front leg - kink at the seat - back - over the top - down again (one bent flat bar)
    Pa, Pb = Ff.lerp(Pr, 0.87), Pr + d_up * 0.05
    bend = bezier_path(tuple(Pa), tuple(Pa.lerp(Pr, 0.7)), tuple(Pb.lerp(Pr, 0.7)), tuple(Pb), n=4)
    side = _seg(Ff, Pa) + [Vector(q) for q in bend] + _seg(Pb, Pr + d_up * (Lb - rc))[1:]
    left = [X(v, -xl) for v in side]
    top = []
    for k in range(1, 7):
        a = radians(90 * k / 6)
        v = Pr + d_up * (Lb - rc + rc * sin(a))
        top.append((-xl + rc * (1 - cos(a)), v.y, v.z))
    path = left + top + [(-q[0], q[1], q[2]) for q in reversed(top)] + [(-q[0], q[1], q[2]) for q in reversed(left)]
    frame.append(sweep("frame_outer", path, bar, up=tuple(n_f), smooth_path=False))
    # ---- inner frame: two rear legs up to the seat hinge
    d_in = (Pf - Fb).normalized()
    n_in = Vector((0, -d_in.z, d_in.y))
    for sx in (-1, 1):
        frame.append(sweep(f"frame_inner{sx}", [X(v, sx * xi) for v in _seg(Fb, Pf + d_in * 0.014, end=True)], bar, up=tuple(n_in), smooth_path=False))
    # cross rods (riveted over at the ends)
    def cross(name, v, x, r=0.0052):
        frame.append(_rod(name, (-x - 0.006, v.y, v.z), (x + 0.006, v.y, v.z), r, sides=8, n=5))
        for sx in (-1, 1):
            scr.append(trs((sx * (x + 0.0025), v.y, v.z), (0, sx * 90, 0)))
    cross("rod_front", Ff.lerp(Pr, 0.2), xl)
    cross("rod_rear", Fb.lerp(Pf, 0.17), xi)
    cross("rod_hinge", Pf, xi)
    cross("rod_seat", Pr, xl)
    # pivot where the frames cross
    den = (Pr.y - Ff.y) * (Pf.z - Fb.z) - (Pr.z - Ff.z) * (Pf.y - Fb.y)
    tq = ((Fb.y - Ff.y) * (Pf.z - Fb.z) - (Fb.z - Ff.z) * (Pf.y - Fb.y)) / den
    Q = Ff.lerp(Pr, tq)
    for sx in (-1, 1):
        frame.append(cylinder("pivot_pin", 0.0045, 0.03, loc=(sx * (xl + xi) * 0.5, Q.y, Q.z), rot=(0, 90, 0), verts=10))
        frame.append(cylinder("pivot_washer", 0.011, 0.0065, loc=(sx * (xl + xi) * 0.5, Q.y, Q.z), rot=(0, 90, 0), verts=12, bevel=0.001))
        scr.append(trs((sx * (xl + 0.0025), Q.y, Q.z), (0, sx * 90, 0), 1.5))
        scr.append(trs((sx * (xi - 0.0025), Q.y, Q.z), (0, -sx * 90, 0), 1.5))
    # shoes
    shoe = rounded_box("shoe", (0.013, 0.05, 0.016), radius=0.005, seg=2, bevel=0.002)
    sh = _stamp("shoes", shoe, [trs((sx * x, v.y + dy, 0.008)) for sx in (-1, 1) for x, v, dy in ((xl, Ff, -0.004), (xi, Fb, 0.004))], M["dark"])
    paint(sh, grime=0.6)
    # seat rails (hinged on the front rod, resting on the rear rod)
    y0, y1 = Pf.y - 0.062 if p.seat == "wood" else Pf.y - 0.05, Pr.y + 0.035
    rail_h = 0.022
    for sx in (-1, 1):
        frame.append(box("seat_rail", (0.005, (Pr.y + 0.022) - (Pf.y - 0.02), rail_h), loc=(sx * xr, (Pr.y + Pf.y) * 0.5 + 0.001, zs - 0.012 - rail_h * 0.5 - 0.0005), bevel=0.0015))

    Ls = 2 * (xl - 0.0075)
    slat_m = M["slat"]
    rs = rng.fork("slats")
    if p.seat == "wood":
        depth = y1 - y0
        n = max(5, int(round(depth / 0.052)))
        pitch = depth / n
        for k in range(n):
            yc = y0 + pitch * (k + 0.5)
            s = plank(f"seat_slat{k}", Ls, pitch - 0.010, 0.011, loc=(rs.jitter(0, 0.0012), yc, zs - 0.0055 + rs.jitter(0, 0.0009)),
                      rot=(rs.jitter(0, 0.8), 0, rs.jitter(0, 0.35)), bevel=0.004, warp=0.0012, rng=rs, axis="X")
            wood_parts.append(s)
            for sx in (-1, 1):
                for dy in (-0.23, 0.23):
                    scr.append(trs((sx * xr, yc + dy * pitch, zs - 0.0002)))
        # two back slats, bowed to the sitter's back, screwed between the frame bars
        specs = [(0.5, 0.06), (0.83, 0.07)]
    else:
        # caned seat frame: four rails mitred round a woven panel
        fw = 0.042
        depth = y1 - y0
        yc = (y0 + y1) * 0.5
        for sy in (-1, 1):
            wood_parts.append(plank("seat_frame_x", Ls, fw, 0.016, loc=(0, yc + sy * (depth - fw) * 0.5, zs - 0.008), bevel=0.004, rng=rs, axis="X"))
        for sx in (-1, 1):
            wood_parts.append(plank("seat_frame_y", depth - 2 * fw, fw, 0.016, loc=(sx * (Ls - fw) * 0.5, yc, zs - 0.008), bevel=0.004, rng=rs, axis="Y"))
            for sy in (-1, 1):
                scr.append(trs((sx * xr, yc + sy * (depth * 0.5 - fw * 0.5), zs - 0.0002)))
        pw, pd = Ls - 2 * fw + 0.006, depth - 2 * fw + 0.006
        sag = 0.007 + rs.u(0, 0.004)
        pan = _cane_weave("cane_seat", _clip_rect(pw * 0.5, pd * 0.5), zs - 0.0105, rs, pitch=0.0165,
                          sag=lambda x, y_: sag * cos(pi * x / pw) * cos(pi * y_ / pd), broken=int(max(0.0, w - 0.55) * 11 + rs.u(0, 1)), underside=False)
        transform_verts(pan, lambda co: Vector((co.x, co.y + yc, co.z)))
        assign(pan, M["cane"])
        paint(pan, grime=_cane_soil(0.0, yc - 0.02, 0.11, w, p.seed + 41))
        specs = [(0.42, 0.034), (0.9, 0.034)]

    def back_path(sfrac, bow=0.02):
        o = Pr + d_up * (Lb * sfrac)
        return [(x, o.y + bow * (1 - (x / xl) ** 2) * (-n_f.y), o.z + bow * (1 - (x / xl) ** 2) * (-n_f.z))
                for x in [(-1 + 2 * i / 10) * (xl - 0.003) for i in range(11)]]
    for k, (sf, sw) in enumerate(specs):
        s = sweep(f"back_slat{k}", back_path(sf), rounded_rect(0.012, sw, 0.0035, 2), up=tuple(d_up), smooth_path=False)
        wood_parts.append(s)
        o = Pr + d_up * (Lb * sf)
        for sx in (-1, 1):
            for dv in ((-0.28, 0.28) if sw > 0.05 else (0.0,)):
                q = o + d_up * (dv * sw)
                scr.append(trs((sx * (xl + 0.0025), q.y, q.z), (0, sx * 90, 0)))
    if p.seat == "cane":
        # woven back panel between the two rails
        o0, o1 = specs[0][0], specs[1][0]
        hgt = Lb * (o1 - o0) - specs[0][1] + 0.006
        pan = _cane_weave("cane_back", _clip_rect(xl - 0.003, hgt * 0.5), -0.002, rs, pitch=0.0165, nseg=3)
        om = Pr + d_up * (Lb * (o0 + o1) * 0.5)
        n_b = -n_f

        def to_back(co):
            q = om + d_up * co.y + n_b * (0.02 * (1 - (co.x / xl) ** 2) + co.z)
            return Vector((co.x, q.y, q.z))
        transform_verts(pan, to_back)
        assign(pan, M["cane"])
        paint(pan, grime=mask_mul(noise_mask(0.09, seed=p.seed + 43, lo=0.3, hi=0.8), 0.25 + 0.3 * w))

    nzC = noise_mask(0.017, seed=p.seed + 12, lo=0.6, hi=0.74)
    nzR = noise_mask(0.03, seed=p.seed + 14, lo=0.6, hi=0.8)
    for o in wood_parts:
        assign(o, slat_m)
        tone = rs.u(0.0, 0.32)
        gap = 0.85 if o.name.startswith("seat_slat") else 0.0
        paint(o, wear=lambda co, nrm: np.clip(np.clip(np.maximum(1 - np.hypot(co[:, 0], co[:, 1] + 0.05) / 0.2, (co[:, 2] - (H - 0.12)) / 0.1), 0, 1) ** 0.6
                                              * noise_mask(0.05, seed=p.seed + 7, lo=0.25, hi=0.7)(co, nrm) * (0.35 + 0.65 * w)
                                              + nzC(co, nrm) * (0.15 + 0.6 * w), 0, 1),
              grime=lambda co, nrm, tone=tone, gap=gap: np.maximum(tone, _ss(0.5, 0.9, np.abs(nrm[:, 1])) * gap),   # dirt in the slat gaps
              special=mask_mul(facing((0, 0, 1)), (0.3 + 0.5 * w) if M["painted"] else 0.0))
    for o in frame:
        assign(o, iron)
        paint(o, special=lambda co, nrm: np.clip(np.maximum.reduce([
                  np.clip(1 - co[:, 2] / 0.18, 0, 1),
                  np.clip(1 - np.hypot(co[:, 1] - Q.y, co[:, 2] - Q.z) / 0.05, 0, 1),            # round the pivot
                  np.clip(1 - np.abs(co[:, 2] - zh) / 0.03, 0, 1) * 0.6,
                  nzR(co, nrm) * 0.7]), 0, 1) * (0.12 + 0.6 * w),
              wear=lambda co, nrm: np.clip(np.clip((co[:, 2] - (H - 0.05)) / 0.05, 0, 1) * (0.2 + 0.7 * w) + nzC(co, nrm) * (0.15 + 0.6 * w), 0, 1),
              grime=mask_mul(height_ramp(0.0, 0.25), 0.6))
    _stamp("rivets", _head("rivet", 0.0045, 0.0026, segs=6, low=True), scr, M["screw"])
    if p.cushion:
        _cushion(p, rng.fork("cushion"), zs - 0.003, size=(Ls - 0.03, (y1 - y0) - 0.03), y=(y0 + y1) * 0.5 - 0.004,
                 tie_x=xl - 0.035, tie_y=y1 - 0.01)
    return zs


def build(p, rng):
    painted = bool(p.painted)
    if painted:
        frame_wood = seat_wood = mat("wood_painted", p.paint, rough=0.5)
        slat = mat("wood_painted", p.paint, rough=0.7)
    else:
        # steam-bent beech under shellac: fine tight grain, satin-gloss; sawn slats stay open-grained
        frame_wood = mat("wood_oak", p.wood, rough=0.25, scale=4.0, grime=0.5)
        seat_wood = mat("wood_oak", p.wood, rough=0.4, scale=2.5, grime=0.6)
        slat = mat("wood_oak", p.wood, rough=0.9, scale=1.5)
    M = {
        "painted": painted,
        "wood": frame_wood,
        "seat_wood": seat_wood,
        "slat": slat,
        "iron": mat("iron_painted", p.paint, rough=0.6),
        "cane": mat("paper", "#b79c68", rough=0.6),                  # split rattan strands: dull straw / honey
        "cane_bead": mat("wicker", "#b99c64", scale=2.0),
        "dark": mat("iron_wrought"),
        "glide": mat("zinc"),
        "screw": mat("brass", rough=0.6) if p.style == "bentwood" else mat("zinc"),
    }
    z_top = {"bentwood": _bentwood, "iron": _iron, "folding": _folding}[p.style](p, rng, M)
    socket("seat", "attach", (0, 0, z_top), role="seat")
