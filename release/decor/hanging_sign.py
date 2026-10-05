"""Hanging shop sign on a projecting wrought-iron scroll bracket.

Construction, wall outwards: a forged back plate with spear ends, coach-bolted to the wall; a 22 mm
square arm tenoned through a diamond rosette, with a hand-twisted middle length and a curled tip;
above it a C-scroll stay (riveted to the plate, collared to the arm) with a nested inner scroll and
a ring bound between them, below it a small corner scroll; every scroll eye is upset to a snub end.  The sign hangs under the arm from two
rings (held between forged stops) on three chain links each and is the 'sign' swing part.

The sign itself is either a timber board (separate tongued planks, an applied moulded frame, a
carved-and-painted emblem and raised lettering on BOTH faces, iron hanging straps joggled over the
frame and nailed) cut as a shield, oval or indented rectangle - or a `cutout`: an iron hoop with the
emblem cut from plate, held on stays, with a small riveted name plate beneath.

Optional pair of sign lamps: a two-plate bolted clamp on a square seat left in the twist carries a
tee and two goosenecks with brass lampholders and enamelled conical shades (cream inside), fed by a
flex tied along the stay.  Each shade is aimed so that its rim lights the whole face of the board but
shades the arm and keeps the beam off the wall; the light's range stops short of the plaster.

Mount 'wall': origin = where the arm meets the wall (plane y = 0); the bracket projects along -Y and
the board hangs in the YZ plane, its two faces looking along +X and -X (up and down the street).
"""
import math
import numpy as np
from mathutils.geometry import tessellate_polygon
from tk import *

META = {"name": "Hanging shop sign on bracket", "class": "component", "mount": "wall",
        "mount_height": 3.05, "focus": [0.0, -0.5, -0.41]}
PARAMS = {
    "emblem": {"default": "cup", "choices": ["cup", "boot", "key", "suitcase", "scissors", "bottle", "book", "star"],
               "desc": "trade emblem on the sign"},
    "shape": {"default": "shield", "choices": ["shield", "oval", "rect", "cutout"],
              "desc": "board shape, or 'cutout' = iron silhouette in a hoop"},
    "text": {"default": "CAFE", "desc": "lettering (short word; empty = emblem only)"},
    "font": {"default": "serif", "choices": ["serif", "roman", "sans", "sans_narrow", "script"], "desc": "lettering face"},
    "board_color": {"default": "#30694b", "desc": "painted ground colour of the board / name plate"},
    "accent_color": {"default": "#e6d8b4", "desc": "colour of emblem, lettering and frame"},
    "gilded": {"default": False, "desc": "emblem and lettering in gold leaf instead of paint"},
    "reach": {"default": 0.92, "min": 0.7, "max": 1.3, "desc": "projection of the bracket arm from the wall (m)"},
    "board_w": {"default": 0.5, "min": 0.36, "max": 0.7, "desc": "board width (m)"},
    "board_h": {"default": 0.62, "min": 0.42, "max": 0.8, "desc": "board height (m)"},
    "bracket": {"default": "scroll", "choices": ["scroll", "plain"], "desc": "scrollwork stay or a plain smith's tie bar"},
    "twist": {"default": True, "desc": "hand-twisted length in the arm"},
    "lamp": {"default": "swan", "choices": ["swan", "none"], "desc": "pair of swan-neck sign lamps over the board"},
    "lamp_cd": {"default": 2.5, "min": 0.5, "max": 8.0, "desc": "intensity of each sign lamp (cd)"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: faded and chipped paint, rust, grime"},
}
VARIANTS = {
    "cobbler_oval": {"emblem": "boot", "shape": "oval", "text": "BOOTS", "board_color": "#3b2a1f", "accent_color": "#d9b45a",
                     "gilded": True, "reach": 0.8, "board_w": 0.46, "board_h": 0.6, "bracket": "plain",
                     "age": 0.7, "seed": 5},
    "locksmith_cutout": {"emblem": "key", "shape": "cutout", "text": "KEYS", "board_color": "#15171a",
                         "accent_color": "#c9a13c", "gilded": True, "reach": 0.86, "board_w": 0.48,
                         "font": "roman", "age": 0.55, "seed": 11},
    "bookshop_rect": {"emblem": "book", "shape": "rect", "text": "BOOKS", "board_color": "#27407a",
                      "accent_color": "#e9dfc4", "reach": 1.12, "board_w": 0.62, "board_h": 0.56, "font": "roman",
                      "age": 0.3, "seed": 17},
    "tailor_scissors": {"emblem": "scissors", "shape": "rect", "text": "Tailor", "font": "script", "board_color": "#d3c7a6",
                        "accent_color": "#2c2520", "reach": 0.84, "board_w": 0.44, "board_h": 0.66, "bracket": "plain",
                        "twist": False, "age": 0.85, "seed": 23},
    "wine_shield": {"emblem": "bottle", "shape": "shield", "text": "VINO", "board_color": "#6a1c34",
                    "accent_color": "#d8b24e", "gilded": True, "reach": 1.0, "board_w": 0.54, "board_h": 0.7,
                    "lamp_cd": 2.8, "age": 0.4, "seed": 31},
}

TP = 0.008      # back plate thickness
A = 0.011       # half section of the 22 mm square arm
TB = 0.0115     # scroll flat bar: thickness
BW = 0.028      # scroll flat bar: width
RELIEF = 0.012  # height of the applied emblem relief


# ------------------------------------------------------------------------------------ 2D polygon helpers

def _area(p):
    n = len(p)
    return 0.5 * sum(p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1] for i in range(n))


def _ccw(p):
    p = _dedupe(p)
    return p if _area(p) > 0 else p[::-1]


def _dedupe(p, eps=1e-6):
    out = []
    for q in p:
        if not out or math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > eps:
            out.append((float(q[0]), float(q[1])))
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def _chaikin(p, it=1):
    for _ in range(it):
        q = []
        n = len(p)
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            q.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            q.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        p = q
    return p


def _inset(p, d):
    """Offset a closed loop to the LEFT of its direction of travel by d (inwards for a CCW loop), mitred.
    Keeps the vertex count, so several insets of one outline can be stitched into mouldings."""
    n = len(p)
    out = []
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        e1 = (b[0] - a[0], b[1] - a[1])
        e2 = (c[0] - b[0], c[1] - b[1])
        l1 = math.hypot(*e1) or 1e-9
        l2 = math.hypot(*e2) or 1e-9
        n1 = (-e1[1] / l1, e1[0] / l1)
        n2 = (-e2[1] / l2, e2[0] / l2)
        k = max(1.0 + n1[0] * n2[0] + n1[1] * n2[1], 0.3)
        out.append((b[0] + d * (n1[0] + n2[0]) / k, b[1] + d * (n1[1] + n2[1]) / k))
    return out


def _zspan(poly, x):
    """(zmin, zmax) where the vertical line at x crosses the outline."""
    zs = []
    n = len(poly)
    for i in range(n):
        (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % n]
        if (x1 - x) * (x2 - x) <= 0 and abs(x2 - x1) > 1e-12:
            zs.append(z1 + (x - x1) / (x2 - x1) * (z2 - z1))
    return (min(zs), max(zs)) if zs else None


def _xspan(poly, z):
    r = _zspan([(b, a) for a, b in poly], z)
    return r


def _tf(pts, s=1.0, ox=0.0, oz=0.0, rot=0.0, mirror=False):
    c, sn = cos(radians(rot)), sin(radians(rot))
    out = []
    for x, z in pts:
        if mirror:
            x = -x
        out.append(((x * c - z * sn) * s + ox, (x * sn + z * c) * s + oz))
    return out


def _arc(cx, cz, rx, rz, a0, a1, n):
    return [(cx + rx * cos(radians(a0 + (a1 - a0) * i / n)), cz + rz * sin(radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def _rect(cx, cz, w, h):
    return [(cx - w / 2, cz - h / 2), (cx + w / 2, cz - h / 2), (cx + w / 2, cz + h / 2), (cx - w / 2, cz + h / 2)]


def _ray_hit(polys, ang, rmax):
    """Largest radius < rmax at which a ray from the origin at `ang` (rad) crosses any of the outlines."""
    ux, uz = cos(ang), sin(ang)
    best = None
    for poly in polys:
        n = len(poly)
        for i in range(n):
            (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % n]
            ex, ez = x2 - x1, z2 - z1
            den = ux * ez - uz * ex
            if abs(den) < 1e-12:
                continue
            t = (x1 * ez - z1 * ex) / den
            w = (x1 * uz - z1 * ux) / den
            if 0.0 <= w <= 1.0 and 0.0 < t < rmax and (best is None or t > best):
                best = t
    return best


# ------------------------------------------------------------------------------------ emblem artwork
# Each emblem is a list of (outline, holes, level) in a unit box (about -0.5..0.5); `level` stacks
# overlapping pieces at slightly different relief heights (handle behind bowl, straps over a case...).

def _emblem(kind):
    P = []
    if kind == "cup":
        bowl = [(x, max(z, -0.155)) for x, z in _arc(-0.04, 0.17, 0.26, 0.36, 180, 360, 14)]
        P.append((bowl, [], 0))
        P.append((_arc(-0.04, 0.178, 0.282, 0.03, 0, 360, 14)[:-1], [], 1))                   # mouth: a flat ellipse
        P.append((_arc(0.2, 0.03, 0.17, 0.14, -100, 100, 10) + _arc(0.2, 0.03, 0.105, 0.08, 100, -100, 8), [], -1))
        P.append((_rect(-0.04, -0.168, 0.2, 0.034), [], -1))                                  # foot ring
        # saucer: a thin dish with lens-shaped ends, only a little wider than the cup
        P.append((_arc(-0.04, -0.182, 0.345, 0.015, 0, 180, 6) + _arc(-0.04, -0.182, 0.345, 0.058, 180, 360, 10)[1:-1], [], -0.5))
        # steam: three S-shaped wisps drawn out to a point at both ends
        for x0, z0, ln, ph, amp in ((-0.185, 0.245, 0.23, 0.3, 0.03), (-0.04, 0.255, 0.31, 2.0, 0.036), (0.105, 0.245, 0.22, 3.9, 0.03)):
            left, right = [], []
            for i in range(12):
                s = i / 11
                cx = x0 + amp * sin(2 * pi * 0.95 * s + ph)
                hw = 0.023 * sin(pi * s ** 0.72) ** 0.9 if 0 < i < 11 else 0.0     # fat low down, long thin tail
                left.append((cx - hw, z0 + ln * s))
                right.append((cx + hw, z0 + ln * s))
            P.append((right + left[::-1], [], -1.75))       # steam is the shallowest cut on the board
        P = [(_tf(o, oz=-0.13), [_tf(h, oz=-0.13) for h in hs], lv) for o, hs, lv in P]
    elif kind == "boot":
        body = [(-0.25, 0.36), (-0.245, 0.15), (-0.225, 0.02), (-0.24, -0.12), (-0.275, -0.24), (-0.27, -0.31),
                (0.40, -0.31), (0.47, -0.285), (0.485, -0.24), (0.46, -0.185), (0.36, -0.13), (0.22, -0.075),
                (0.12, -0.01), (0.07, 0.10), (0.055, 0.25), (0.06, 0.36)]
        P.append((_chaikin(body, 1), [], 0))
        P.append((_arc(-0.2, 0.46, 0.055, 0.06, 0, 360, 10)[:-1], [_arc(-0.2, 0.46, 0.026, 0.03, 0, 360, 8)[:-1]], -1))  # pull loop
        P.append(([(-0.28, 0.335), (0.09, 0.335), (0.098, 0.37), (0.095, 0.455), (-0.27, 0.43), (-0.283, 0.39)], [], 1))   # cuff
        P.append(([(-0.075, -0.365), (0.42, -0.365), (0.49, -0.342), (0.512, -0.3), (0.49, -0.285), (0.40, -0.3),
                   (-0.075, -0.3)], [], 1))                                                    # welted sole
        P.append(([(-0.287, -0.47), (-0.1, -0.47), (-0.066, -0.37), (-0.066, -0.298), (-0.298, -0.298)], [], 2))   # stacked heel
        P.append(([(0.3, -0.29), (0.4, -0.29), (0.463, -0.272), (0.476, -0.24), (0.453, -0.194), (0.37, -0.145),
                   (0.318, -0.122), (0.286, -0.2)], [], 1))                                    # toe cap
        P.append(([(-0.045, 0.315), (-0.05, 0.12), (-0.025, 0.0), (0.06, -0.08), (0.14, -0.035), (0.088, 0.09),
                   (0.07, 0.25), (0.072, 0.315)], [], 1))                                      # lace facing
        for ex, ez_ in ((0.014, 0.268), (0.014, 0.198), (0.02, 0.128), (0.034, 0.058), (0.062, -0.008)):
            P.append((_arc(ex, ez_, 0.021, 0.021, 0, 360, 6)[:-1], [], 2))                     # eyelets
    elif kind == "key":
        zj = 0.28 - math.sqrt(0.17 ** 2 - 0.035 ** 2)
        aj = degrees(math.atan2(zj - 0.28, 0.035))
        o = [(-0.035, -0.47), (0.035, -0.47), (0.035, -0.43), (0.2, -0.43), (0.2, -0.35), (0.125, -0.35), (0.125, -0.31),
             (0.2, -0.31), (0.2, -0.23), (0.035, -0.23), (0.035, 0.045), (0.075, 0.045), (0.075, 0.09), (0.035, 0.09)]
        o += _arc(0.0, 0.28, 0.17, 0.17, aj, 180 - aj, 22)
        o += [(-0.035, 0.09), (-0.075, 0.09), (-0.075, 0.045), (-0.035, 0.045)]
        hole = _arc(0.0, 0.28, 0.088, 0.088, 0, 360, 14)[:-1]
        P.append((_tf(o, rot=-38, oz=0.02), [_tf(hole, rot=-38, oz=0.02)], 0))
    elif kind == "suitcase":
        P.append((_tf(rounded_rect(0.86, 0.56, 0.05, 3), oz=-0.09), [], 0))
        P.append((_tf(rounded_rect(0.3, 0.17, 0.05, 3), oz=0.25), [_tf(rounded_rect(0.2, 0.075, 0.03, 2), oz=0.257)], -1))
        for sx in (-0.24, 0.24):
            P.append((_rect(sx, -0.09, 0.075, 0.585), [], 1))                                  # leather straps
            P.append((_rect(sx, 0.07, 0.11, 0.065), [], 2))                                    # buckles
        P.append((_rect(0.0, 0.14, 0.11, 0.07), [], 1))                                        # lock
        for sx in (-1, 1):
            for sz in (-1, 1):
                c = (sx * 0.43, -0.09 + sz * 0.28)
                P.append(([(c[0], c[1]), (c[0] - sx * 0.1, c[1]), (c[0] - sx * 0.1, c[1] - sz * 0.03),
                           (c[0] - sx * 0.03, c[1] - sz * 0.1), (c[0], c[1] - sz * 0.1)], [], 1))  # corner caps
    elif kind == "scissors":
        half = [(-0.03, 0.1), (-0.03, -0.03), (0.03, -0.23), (0.08, -0.21), (0.036, -0.02), (0.04, 0.08), (0.004, 0.5)]
        ring = _arc(0.085, -0.335, 0.105, 0.13, 0, 360, 16)[:-1]
        rhole = _arc(0.085, -0.335, 0.062, 0.087, 0, 360, 12)[:-1]
        for mir, rot, lv in ((False, 14, 0), (True, -14, 1)):
            P.append((_tf(half, rot=rot, mirror=mir), [], lv))
            P.append((_tf(ring, rot=rot, mirror=mir), [_tf(rhole, rot=rot, mirror=mir)], lv))
        P.append((_arc(0, 0, 0.032, 0.032, 0, 360, 10)[:-1], [], 2))                           # pivot screw
        P = [(_tf(o, oz=-0.02), [_tf(h, oz=-0.02) for h in hs], lv) for o, hs, lv in P]
    elif kind == "bottle":
        hp = [(0.0, -0.46), (0.125, -0.46), (0.14, -0.44), (0.14, 0.0), (0.125, 0.08), (0.06, 0.2), (0.045, 0.26),
              (0.045, 0.4), (0.058, 0.41), (0.058, 0.455), (0.045, 0.465), (0.0, 0.465)]
        full = hp[1:-1] + [(-x, z) for x, z in hp[1:-1]][::-1]
        P.append((_tf(full, ox=-0.17), [], 0))
        P.append((_tf(rounded_rect(0.21, 0.2, 0.02, 2), ox=-0.17, oz=-0.2), [], 1))            # label
        P.append((_rect(-0.17, 0.335, 0.105, 0.14), [], 1))                                    # foil capsule
        P.append((_arc(0.23, 0.14, 0.135, 0.25, 180, 360, 12), [], 0))                         # glass bowl
        P.append((_rect(0.23, -0.26, 0.03, 0.34), [], -1))                                     # stem
        P.append(([(0.11, -0.46), (0.35, -0.46), (0.35, -0.44), (0.26, -0.405), (0.2, -0.405), (0.11, -0.44)], [], 0))
    elif kind == "book":
        # an open book seen from a little above: the leaves spring out of a centre gutter, swell and fall
        # away to the fore-edge; both boards of the cover lie at an angle below them and meet in a spine
        # that shows as a 'V' top and bottom; the edges of the leaves fan out in steps under each page
        xk = (0.016, 0.05, 0.11, 0.2, 0.3, 0.385, 0.44)
        fz = lambda x: 0.135 * (1.0 - math.exp(-x / 0.05)) - 0.19 * x          # lift of a leaf at x from the gutter
        top = [(x, 0.25 + fz(x)) for x in xk]
        bot = [(x, -0.2 + fz(x)) for x in xk]
        cover = [(0.0, -0.315), (0.06, -0.262), (0.495, -0.24), (0.495, 0.265), (0.455, 0.275)]
        cover = cover + [(0.0, 0.235)] + [(-x, z) for x, z in cover[::-1]]
        P.append((cover, [], -1))
        for mir in (False, True):
            P.append((_tf(bot + top[::-1], mirror=mir), [], 1))                                  # top leaf
            for k, (dz, dx) in enumerate(((0.026, 0.013), (0.052, 0.026))):                      # fanned leaf edges
                eb = [bot[i] for i in (1, 2, 3, 6)]
                lo_ = [(x + dx * (x / 0.44), z - dz) for x, z in eb]
                hi_ = [(x + dx * (x / 0.44), z - dz + 0.016) for x, z in eb]
                side = [(0.44 + dx, 0.25 + fz(0.44) - dz), (0.44 + dx - 0.012, 0.25 + fz(0.44) - dz)]
                P.append((_tf(lo_ + side + [(hi_[-1][0] - 0.012, hi_[-1][1])] + hi_[-2::-1], mirror=mir), [], 0))
            for k in range(4):                                                                   # lines of print
                xa_, xb_ = 0.075, (0.375 if k < 3 else 0.25)
                xs_ = [xa_ + (xb_ - xa_) * i / 3 for i in range(4)]
                zl = 0.165 - 0.088 * k
                P.append((_tf([(x, zl + fz(x) - 0.012) for x in xs_] + [(x, zl + fz(x) + 0.012) for x in xs_[::-1]], mirror=mir), [], 2))
        P = [(_tf(o, oz=-0.02), [_tf(h, oz=-0.02) for h in hs], lv) for o, hs, lv in P]
    else:  # star
        def star(ro, ri, a0, n=5):
            return [((ro if i % 2 == 0 else ri) * cos(radians(a0 + 180 * i / n)),
                     (ro if i % 2 == 0 else ri) * sin(radians(a0 + 180 * i / n))) for i in range(2 * n)]
        P.append((star(0.37, 0.19, 90 + 36), [], -1))
        P.append((star(0.5, 0.2, 90), [], 0))
        P.append((_arc(0, 0, 0.085, 0.085, 0, 360, 12)[:-1], [], 1))
        P = [(_tf(o, oz=-0.03), [_tf(h, oz=-0.03) for h in hs], lv) for o, hs, lv in P]
    return [(_ccw(o), [_ccw(h)[::-1] for h in hs], lv) for o, hs, lv in P]


# ------------------------------------------------------------------------------------ mesh builders (sign-local)
# Sign-local frame: the board stands in the XZ plane, its front face looks along -Y (toolkit
# convention), origin at the board centre.  The finished sign is turned into the YZ plane at the end.

def _relief(name, outer, holes, y0, depth, chamfer=0.002, draft=0.0):
    """Raised relief of a 2D shape (outer CCW, holes CW) on the plane y = y0, standing `depth` proud
    towards -Y, with a chamfered arris and (optionally) drafted sides like a carved / cast piece, so
    the flanks catch the light. No hidden back face."""
    loops = [outer] + list(holes)
    verts, faces = [], []
    tops = []
    xs, zs = [q[0] for q in outer], [q[1] for q in outer]
    per = sum(math.hypot(outer[i][0] - outer[i - 1][0], outer[i][1] - outer[i - 1][1]) for i in range(len(outer)))
    small = 2.0 * abs(_area(outer)) / per                  # ~ width of a strip, half the radius of a disc
    chamfer = min(chamfer, 0.22 * min(max(xs) - min(xs), max(zs) - min(zs)), 0.62 * min(_thin(lp) for lp in loops))
    for lp in loops:
        n = len(lp)
        b0 = len(verts)
        verts += [(x, y0, z) for x, z in (_inset(lp, -draft) if draft else lp)]
        if small < 0.0115:                                  # a line of print, a pin: one drafted flank is enough
            ins = lp
            verts += [(x, y0 - depth, z) for x, z in lp]
            for i in range(n):
                j = (i + 1) % n
                faces.append((b0 + i, b0 + j, b0 + n + j, b0 + n + i))
            tops.append((b0 + n, ins))
            continue
        ins = _inset(lp, 0.6 * chamfer)                    # 60 deg chamfer: blends with the flank, breaks at the face
        verts += [(x, y0 - (depth - chamfer), z) for x, z in lp]
        verts += [(x, y0 - depth, z) for x, z in ins]
        for i in range(n):
            j = (i + 1) % n
            faces.append((b0 + i, b0 + j, b0 + n + j, b0 + n + i))
            faces.append((b0 + n + i, b0 + n + j, b0 + 2 * n + j, b0 + 2 * n + i))
        tops.append((b0 + 2 * n, ins))
    flat = [i for b, ins in tops for i in range(b, b + len(ins))]
    tri = tessellate_polygon([[Vector((x, z, 0.0)) for x, z in ins] for _, ins in tops])
    for a, b, c in tri:
        ia, ib, ic = flat[a], flat[b], flat[c]
        pa, pb, pc = verts[ia], verts[ib], verts[ic]
        if (pb[0] - pa[0]) * (pc[2] - pa[2]) - (pc[0] - pa[0]) * (pb[2] - pa[2]) < 0:
            ib, ic = ic, ib
        faces.append((ia, ib, ic))
    o = mesh_object(name, verts, faces)
    uv_grain(o, 'Z')
    o["tsq_smooth"] = 40.0
    return o


def _thin(lp):
    """Narrowest neck of a loop: least distance from a vertex to an edge that faces it from the other
    side (so the chamfer of a thin wisp or spike never crosses itself)."""
    n = len(lp)
    nrm = []
    for j in range(n):
        (x1, z1), (x2, z2) = lp[j], lp[(j + 1) % n]
        l = math.hypot(x2 - x1, z2 - z1) or 1e-9
        nrm.append((-(z2 - z1) / l, (x2 - x1) / l))
    best = 1e9
    for i in range(n):
        px, pz = lp[i]
        nx, nz = nrm[i - 1][0] + nrm[i][0], nrm[i - 1][1] + nrm[i][1]
        for j in range(n):
            if j == i or (j + 1) % n == i or nx * nrm[j][0] + nz * nrm[j][1] > -0.5:
                continue
            (x1, z1), (x2, z2) = lp[j], lp[(j + 1) % n]
            ex, ez = x2 - x1, z2 - z1
            t = clamp(((px - x1) * ex + (pz - z1) * ez) / max(ex * ex + ez * ez, 1e-12), 0.0, 1.0)
            best = min(best, math.hypot(px - x1 - t * ex, pz - z1 - t * ez))
    return best


def _fit(safe, pts, e_lo, e_hi, e_max, keep_out=()):
    """Largest emblem that fits the board: scale E, offset (ex, ez) at which every point of the artwork
    lies inside the outline `safe` between the heights e_lo..e_hi and outside the keep-out boxes
    (x0, x1, z0, z1) - the hanger straps."""
    xs, zs = [q[0] for q in pts], [q[1] for q in pts]
    cx, zmin, zmax = 0.5 * (min(xs) + max(xs)), min(zs), max(zs)
    for i in range(60):
        E = e_max * 0.97 ** i
        span = (zmax - zmin) * E
        if span > e_hi - e_lo:
            continue
        for j in (4, 5, 3, 6, 2, 7, 1, 8, 0):
            ez = e_lo + (e_hi - e_lo - span) * j / 8.0 - zmin * E
            q = _tf(pts, s=E, ox=-cx * E, oz=ez)
            if _inside(safe, q) and not any(x0 < x < x1 and z0 < z < z1 for x0, x1, z0, z1 in keep_out for x, z in q):
                return E, -cx * E, ez
    E = e_max * 0.97 ** 60
    return E, -cx * E, 0.5 * (e_lo + e_hi) - 0.5 * (zmin + zmax) * E


def _moulding(name, outline, y0, prof):
    """Frame moulding following a closed CCW outline: prof = [(inset from the edge, height), ...]
    from the outside inwards; stands proud of the plane y = y0 towards -Y."""
    n = len(outline)
    rings = [_inset(outline, e) for e, _ in prof]
    verts = [(x, y0 - h, z) for ring, (_, h) in zip(rings, prof) for x, z in ring]
    faces = []
    for k in range(len(prof) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces.append((k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i))
    o = mesh_object(name, verts, faces)
    uv_grain(o, 'Z')
    o["tsq_smooth"] = 42.0
    return o


def _plank(name, ol, xa, xb, T, fine_x, rows, ease=0.0022):
    """One board of the sign, cut to the outline between x = xa..xb: a structured grid (so paint
    masks have resolution) with eased long arrises, closed all round."""
    cols = {round(xa, 5), round(xb, 5), round(xa + ease, 5), round(xb - ease, 5)}
    n = max(1, int(round((xb - xa) / 0.046)))
    for i in range(1, n):
        cols.add(round(xa + (xb - xa) * i / n, 5))
    for x in fine_x:
        if xa + ease + 0.003 < x < xb - ease - 0.003:
            cols.add(round(x, 5))
    cols = sorted(cols)
    keep = [cols[0], cols[1]]
    for x in cols[2:-2]:
        if x - keep[-1] > 0.0045 and cols[-2] - x > 0.0045:
            keep.append(x)
    keep += [cols[-2], cols[-1]]
    cols = keep
    nc = len(cols)
    verts = []
    for side in (-1, 1):
        for ci, x in enumerate(cols):
            zb, zt = _zspan(ol, x)
            edge = ci in (0, nc - 1)
            y = side * (T / 2 - (ease if edge else 0.0))
            for j in range(rows + 1):
                verts.append((x, y, zb + (zt - zb) * j / rows))
    nv = nc * (rows + 1)
    idx = lambda s, i, j: s * nv + i * (rows + 1) + j
    faces = []
    for i in range(nc - 1):
        for j in range(rows):
            faces.append((idx(0, i, j), idx(0, i + 1, j), idx(0, i + 1, j + 1), idx(0, i, j + 1)))
            faces.append((idx(1, i, j), idx(1, i, j + 1), idx(1, i + 1, j + 1), idx(1, i + 1, j)))
        faces.append((idx(0, i, 0), idx(1, i, 0), idx(1, i + 1, 0), idx(0, i + 1, 0)))                    # bottom edge
        faces.append((idx(0, i, rows), idx(0, i + 1, rows), idx(1, i + 1, rows), idx(1, i, rows)))        # top edge
    for j in range(rows):
        faces.append((idx(0, 0, j), idx(0, 0, j + 1), idx(1, 0, j + 1), idx(1, 0, j)))                    # left edge
        faces.append((idx(0, nc - 1, j), idx(1, nc - 1, j), idx(1, nc - 1, j + 1), idx(0, nc - 1, j + 1)))
    o = mesh_object(name, verts, faces)
    uv_grain(o, 'Z')
    o["tsq_smooth"] = 30.0
    return o


def _lettering(name, body, font, cap, maxw, y0, z, depth=0.007, fat=0.0, draft=0.0012, arc_R=0.0):
    """Raised lettering fitted to cap height `cap` and width `maxw`, front face only (no hidden back).
    fat   : embolden the strokes by this much (m) - thin script faces need it to survive weathering.
    draft : the sides lean in towards the face like carved letters, so they catch the lamp light.
    arc_R : bend the line into a 'smile' of this base-line radius (follows the rim of an oval board)."""
    import bmesh
    o = text(name, body, size=cap / 0.7, depth=depth, loc=(0, y0, z), font=font)
    me = o.data
    if len(me.vertices) == 0:
        delete([o])
        return None
    bm = bmesh_from(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=2e-5)
    for f in [f for f in bm.faces if all(v.co.z < 1e-5 for v in f.verts)]:
        bm.faces.remove(f)
    bmesh.ops.dissolve_limit(bm, angle_limit=radians(16.0), verts=bm.verts[:], edges=bm.edges[:])
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    k = min(maxw / max(w, 1e-6), cap * 1.15 / max(h, 1e-6), 1.6)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    for v in bm.verts:
        v.co.x = (v.co.x - cx) * k
        v.co.y = (v.co.y - cy) * k
    if fat or draft:
        bm.normal_update()
        move = []
        zmid = 0.5 * (min(v.co.z for v in bm.verts) + max(v.co.z for v in bm.verts))
        for v in bm.verts:
            nx = ny = 0.0
            for f in v.link_faces:
                if abs(f.normal.z) < 0.5:
                    nx += f.normal.x
                    ny += f.normal.y
            l = math.hypot(nx, ny)
            if l > 1e-6:
                d = fat + (draft if v.co.z < zmid else 0.0)
                move.append((v, nx / l * d, ny / l * d))
        for v, dx, dy in move:
            v.co.x += dx
            v.co.y += dy
    if arc_R:
        for v in bm.verts:
            th, r = v.co.x / arc_R, arc_R - v.co.y
            v.co.x, v.co.y = r * sin(th), arc_R - r * cos(th)
    bm.to_mesh(me)
    bm.free()
    uv_grain(o, 'Y')
    return o


def _inside(poly, pts):
    """True if every 2D point lies inside the closed outline."""
    n = len(poly)
    for x, z in pts:
        c = False
        for i in range(n):
            (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % n]
            if (z1 > z) != (z2 > z) and x < x1 + (z - z1) / (z2 - z1) * (x2 - x1):
                c = not c
        if not c:
            return False
    return True


def _aim(obj, loc, zdir):
    """Stand a part built along +Z at `loc` with its axis along `zdir`."""
    q = Vector((0, 0, 1)).rotation_difference(Vector(zdir).normalized())
    obj.matrix_basis = Matrix.Translation(Vector(loc)) @ q.to_matrix().to_4x4() @ obj.matrix_basis
    return obj


def _facing(obj, inward):
    """Make a lathed shell face the axis (inward) or away from it: lathe() turns every stand-alone
    shell outwards, which leaves the lining of a lamp shade inside-out."""
    out = sum(pl.area * (pl.normal.x * pl.center.x + pl.normal.y * pl.center.y) for pl in obj.data.polygons)
    if (out > 0) == inward:
        obj.data.flip_normals()
    return obj


def smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bmesh_from(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    return bm


def _outline(shape, W, H):
    """Board outline (CCW, centred) for the timber shapes."""
    if shape == "oval":
        return _ccw(_arc(0, 0, W / 2, H / 2, 0, 360, 40)[:-1])
    if shape == "rect":
        rc, n = min(0.048, W * 0.11), 5
        hw, hh = W / 2, H / 2
        return _ccw(_arc(hw, -hh, rc, rc, 180, 90, n) + _arc(hw, hh, rc, rc, 270, 180, n)
                    + _arc(-hw, hh, rc, rc, 0, -90, n) + _arc(-hw, -hh, rc, rc, 90, 0, n))
    # heater shield: flat top, straight shoulders, sides sweeping to a point
    hw, hh, r = W / 2, H / 2, 0.016
    z0 = hh - 0.34 * H
    right = [(hw * (1 - (i / 12) ** 1.65), z0 - (i / 12) * (z0 + hh)) for i in range(13)]      # shoulder -> point
    top_r = _arc(hw - r, hh - r, r, r, 0, 90, 3)
    top_l = _arc(-hw + r, hh - r, r, r, 90, 180, 3)
    left = [(-x, z) for x, z in right]
    return _ccw(right[::-1] + top_r + top_l + left[1:])


# ------------------------------------------------------------------------------------ bracket helpers (d = out from wall, z)

def _spiral(c, R, turns, th, s, fr=0.3, outward=True, per_turn=13):
    """Scroll end around c. outward=True: from the eye out to angle th (deg); False: from angle th
    winding in. s = +1 ccw / -1 cw travel (in the d-z drawing plane)."""
    n = max(6, int(round(turns * per_turn)))
    pts = []
    for i in range(n + 1):
        t = i / n
        if outward:
            ang = radians(th) - s * 2 * pi * turns * (1 - t)
            r = R * (fr + (1 - fr) * t)
        else:
            ang = radians(th) + s * 2 * pi * turns * t
            r = R * (1 + (fr - 1) * t)
        pts.append((c[0] + r * cos(ang), c[1] + r * sin(ang)))
    return pts


def _r_at(R, turns, q, fr=0.3):
    """Radius of a _spiral a fraction q of a turn inside its outer end."""
    return R * (1 - (1 - fr) * q / turns)


def _unit(a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    l = math.hypot(dx, dz) or 1e-9
    return dx / l, dz / l


def _scroll_path(spA, spB, n_mid=12, k=0.39):
    """C-scroll centre line: spiral A (outward) + Bezier + spiral B (inward). Returns (pts, mid_pts)."""
    ua, ub = _unit(spA[-2], spA[-1]), _unit(spB[0], spB[1])
    pa, pb = spA[-1], spB[0]
    h = k * math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    mid = bezier_path((pa[0], pa[1], 0), (pa[0] + ua[0] * h, pa[1] + ua[1] * h, 0),
                      (pb[0] - ub[0] * h, pb[1] - ub[1] * h, 0), (pb[0], pb[1], 0), n=n_mid)
    mid = [(m[0], m[1]) for m in mid]
    return spA[:-1] + mid + spB[1:], mid


def _p3(pts, x=0.0):
    return [(x, -d, z) for d, z in pts]


def build(p, rng):
    R = p.reach
    age = p.age
    seed = p.seed
    cut = p.shape == "cutout"
    W = min(p.board_w, R - 0.3)
    Hb = p.board_h if not cut else W
    T = 0.032

    # ------------------------------------------------------------------ materials
    # old iron: the WEAR state of this material is sound, dark, handled iron - painted nearly everywhere -
    # and its rusty base only comes through where the weathering below lets it (joints, eyes, plate foot)
    iron = mat("iron_rusty")
    ground = mat("iron_painted" if cut else "wood_painted", p.board_color)
    lum = sum(c * k for c, k in zip(hex_rgb(p.board_color), (0.2126, 0.7152, 0.0722)))     # light or dark ground?
    brass = mat("brass", rough=0.9)
    if p.gilded:
        # gold leaf: the burnished faces and arrises are the material's bright 'wear' state, the recesses
        # stay toned down and warm (painted below) - so every chamfer rolls a highlight under the lamps
        accent = mat("brass", "#fff1cf", rough=1.12)
    else:
        accent = mat("iron_painted" if cut else "wood_painted", p.accent_color)
    core_m = mat("wood_oak", "#5a4636")

    body_iron, fixings = [], []
    joints = []                         # (d, z) of iron-to-iron joints: rust gathers there
    traps = []                          # (d, z) of scroll eyes: they hold rain water
    rj = rng.fork("bracket")

    bar = [(-TB / 2, -BW / 2), (TB / 2, -BW / 2), (TB / 2, BW / 2), (-TB / 2, BW / 2)]
    taper = lambda t: 0.55 + 0.45 * min(1.0, t / 0.16, (1.0 - t) / 0.16)

    def scroll(name, pts, prof=bar, fn=taper):
        o = sweep(name, _p3(pts), prof, smooth_path=False, up=(1, 0, 0), scale_fn=fn)
        displace_noise(o, strength=0.0011, scale=0.09, seed=rj.randrange(999), along_normal=False, axis_mask=(0.4, 1, 1))
        body_iron.append(o)
        for q in (pts[0], pts[-1]):                      # snub ends: the eye of each scroll is upset to a button
            traps.append(q)
            body_iron.append(cylinder("snub", 0.0105, BW * 0.66, loc=(0, -q[0], q[1]), rot=(0, 90, 0), verts=8,
                                      bevel=0.002, segments=1))
        return o

    def collar(d, z, ud, uz, radial, along=0.015, width=BW + 0.008):
        o = box("collar", (width, along, radial), loc=(0, -d, z), rot=(degrees(math.atan2(ud, uz)), 0, 0),
                bevel=0.0025, segments=1)
        body_iron.append(o)
        joints.append((d, z))
        return o

    rivet_at = []                        # (loc, rot) of dome rivets, stamped at the end
    nail_at = []

    # ------------------------------------------------------------------ sign position along the arm
    Rt = 0.046                                             # tip curl radius
    dc = R - 2 * Rt - 0.02 - W / 2                         # board centre, out from the wall
    hs = (0.32 if p.shape in ("rect", "shield") else 0.27) * W      # half spacing of the two hangers
    wall_gap = dc - W / 2

    # ------------------------------------------------------------------ arm: square bar, twisted length, curled tip
    sq = rounded_rect(2 * A, 2 * A, 0.003, 1)
    lamp_on = p.lamp == "swan"
    dl = dc                                                # lamp clamp: mid-way between the two hangers
    seat = 0.03 if lamp_on else 0.0                        # half length of the plain square seat left for it
    tw0, tw1 = dc - hs + 0.036, dc + hs - 0.036
    tw_on = p.twist and (tw1 - tw0) / 2 - seat > 0.05

    def straight(name, d0, d1, twist=0.0):
        n = max(2, int(round((d1 - d0) / (0.0065 if twist else 0.06))))
        o = sweep(name, [(0, -(d0 + (d1 - d0) * i / n), 0) for i in range(n + 1)], sq,
                  smooth_path=False, up=(0, 0, 1), twist_deg=twist, cap=False)
        if twist:
            o["tsq_smooth"] = 62.0                          # eased arrises shade round: no single facet flashes
        body_iron.append(o)
        return o

    if tw_on and lamp_on:
        # twist - square seat - twist: the smith leaves the bar square where the lamp clamp bites
        straight("arm_root", -0.004, tw0)
        turns = max(0.5, round(4 * (dl - seat - tw0) / 0.115) / 4)
        straight("arm_twist", tw0, dl - seat, twist=360.0 * turns)
        straight("arm_seat", dl - seat, dl + seat)
        straight("arm_twist", dl + seat, tw1, twist=-360.0 * turns)
        d_run = tw1
    elif tw_on:
        straight("arm_root", -0.004, tw0)
        straight("arm_twist", tw0, tw1, twist=360.0 * max(1, round((tw1 - tw0) / 0.13)))
        d_run = tw1
    else:
        d_run = -0.004
    dT = R - Rt
    curl = _spiral((dT, Rt), Rt, 1.2, -90, 1, fr=0.34, outward=False, per_turn=14)
    n_run = max(2, int(round((dT - d_run) / 0.06)))
    run = [(d_run + (dT - d_run) * i / n_run, 0.0) for i in range(n_run)]
    frac = (dT - d_run) / ((dT - d_run) + path_length(_p3(curl)))
    tip = sweep("arm_tip", _p3(run + curl), sq, smooth_path=False, up=(0, 0, 1),
                scale_fn=lambda t: 1.0 if t < frac else 1.0 - 0.5 * ((t - frac) / (1 - frac)) ** 0.8)
    body_iron.append(tip)
    body_iron.append(cylinder("snub", 0.0115, 2 * A * 0.72, loc=(0, -curl[-1][0], curl[-1][1]), rot=(0, 90, 0), verts=8,
                              bevel=0.002, segments=1))
    traps.append(curl[-1])

    # ------------------------------------------------------------------ stay above the arm
    H = clamp(0.45 * R, 0.33, 0.5)
    zt_plate = H + 0.06
    if p.bracket == "scroll":
        RA, RB, tA, tB = rj.vary(0.066, 0.06), rj.vary(0.05, 0.06), rj.vary(1.35, 0.05), rj.vary(1.3, 0.05)
        cA = (TP + TB / 2 + _r_at(RA, tA, 0.25), H)
        cB = (R - 2 * Rt - 0.035 - RB, A + TB / 2 + _r_at(RB, tB, 0.25))
        spA = _spiral(cA, RA, tA, 90, -1, outward=True)
        spB = _spiral(cB, RB, tB, 0, -1, outward=False)
        pts, mid = _scroll_path(spA, spB, n_mid=14)
        scroll("stay", pts)
        rivet_at.append(((0, -(TP + TB), cA[1]), (90, 0, 0)))
        joints.append((TP, cA[1]))
        collar(cB[0], TB / 2, 0, 1, 2 * A + TB + 0.007)
        # nested inner scroll
        RA2, RB2 = rj.vary(0.05, 0.06), rj.vary(0.04, 0.06)
        cA2 = (TP + TB / 2 + _r_at(RA2, 1.3, 0.25), 0.56 * H - RA2)
        cB2 = (min(0.5 * R, cB[0] - RB - 0.1) - RB2, A + TB / 2 + _r_at(RB2, 1.25, 0.25))
        spA2 = _spiral(cA2, RA2, 1.3, 90, -1, outward=True)
        spB2 = _spiral(cB2, RB2, 1.25, 0, -1, outward=False)
        pts2, mid2 = _scroll_path(spA2, spB2, n_mid=10)
        scroll("stay_inner", pts2)
        rivet_at.append(((0, -(TP + TB), cA2[1]), (90, 0, 0)))
        joints.append((TP, cA2[1]))
        collar(cB2[0], TB / 2, 0, 1, 2 * A + TB + 0.007)
        # ring bound between the two scrolls where they run closest to parallel
        ps = mid[len(mid) // 2]
        pi_ = min(mid2, key=lambda q: math.hypot(q[0] - ps[0], q[1] - ps[1]))
        gap = math.hypot(pi_[0] - ps[0], pi_[1] - ps[1])
        ud, uz = _unit(pi_, ps)
        rr = gap / 2 - TB
        if rr > 0.018:
            cr = ((ps[0] + pi_[0]) / 2, (ps[1] + pi_[1]) / 2)
            ring = sweep("ring", _p3(_arc(cr[0], cr[1], rr, rr, 0, 360, 20)[:-1]), bar, closed=True, smooth_path=False,
                         up=(1, 0, 0))
            body_iron.append(ring)
            collar(ps[0] - ud * TB / 2, ps[1] - uz * TB / 2, ud, uz, 2 * TB + 0.006, along=0.013)
            collar(pi_[0] + ud * TB / 2, pi_[1] + uz * TB / 2, ud, uz, 2 * TB + 0.006, along=0.013)
        zt_plate = cA[1] + 0.055
    else:
        # plain smith's tie: twisted square bar with flattened, riveted palms and a knop
        d1 = R - 2 * Rt - 0.07
        p0, p1 = (TP + 0.004, H), (d1, A + 0.006)
        ud, uz = _unit(p0, p1)
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        p0_, p1_, ud_, uz_, L_ = p0, p1, ud, uz, L
        sq2 = [(-0.0075, -0.0075), (0.0075, -0.0075), (0.0075, 0.0075), (-0.0075, 0.0075)]
        n = int(L / 0.015)
        tie = sweep("tie", _p3([(p0[0] + ud * L * i / n, p0[1] + uz * L * i / n) for i in range(n + 1)]), sq2,
                    smooth_path=False, up=(1, 0, 0), twist_deg=360.0 * round(L / 0.16))
        tie["tsq_smooth"] = 50.0
        body_iron.append(tie)
        ang = degrees(math.atan2(ud, uz))
        for q, ln in ((p0, 0.07), (p1, 0.075)):
            body_iron.append(box("palm", (0.03, 0.007, ln), loc=(0, -q[0], q[1]), rot=(0 if q is p0 else 90, 0, 0),
                                 bevel=0.002, segments=1))
            joints.append(q)
        rivet_at.append(((0, -(TP + 0.0075), H + 0.02), (90, 0, 0)))
        rivet_at.append(((0, -(TP + 0.0075), H - 0.02), (90, 0, 0)))
        rivet_at.append(((0, -(d1 + 0.02), A + 0.0095), (0, 0, 0)))
        rivet_at.append(((0, -(d1 - 0.02), A + 0.0095), (0, 0, 0)))
        km = (p0[0] + ud * L * 0.5, p0[1] + uz * L * 0.5)
        knop = lathe("knop", [(0.0075, -0.03), (0.0115, -0.024), (0.0115, -0.018), (0.0085, -0.014), (0.017, -0.004),
                              (0.017, 0.004), (0.0085, 0.014), (0.0115, 0.018), (0.0115, 0.024), (0.0075, 0.03)],
                     segments=10, loc=(0, -km[0], km[1]), rot=(ang, 0, 0))
        body_iron.append(knop)
        # corner ring between plate, arm and tie
        rr = 0.055
        cr = (TP + TB / 2 + rr + 0.012, A + TB / 2 + rr + 0.004)
        body_iron.append(sweep("ring", _p3(_arc(cr[0], cr[1], rr, rr, 0, 360, 20)[:-1]), bar, closed=True,
                               smooth_path=False, up=(1, 0, 0)))
        collar(cr[0], TB / 2, 0, 1, 2 * A + TB + 0.007)
        rivet_at.append(((0, -(cr[0] - rr + TB / 2), cr[1]), (90, 0, 0)))
        joints.append((TP, cr[1]))
        zt_plate = H + 0.07

    # ------------------------------------------------------------------ corner scroll under the arm
    zb_plate = -0.2
    dB3 = min(0.2, wall_gap - 0.085)
    if dB3 > 0.11:
        RA3, RB3 = rj.vary(0.044, 0.06), rj.vary(0.036, 0.06)
        cA3 = (TP + TB / 2 + _r_at(RA3, 1.3, 0.25), -(dB3 + 0.035))
        cB3 = (dB3, -(A + TB / 2 + _r_at(RB3, 1.25, 0.25)))
        spA3 = _spiral(cA3, RA3, 1.3, -90, 1, outward=True)
        spB3 = _spiral(cB3, RB3, 1.25, 0, 1, outward=False)
        pts3, _ = _scroll_path(spA3, spB3, n_mid=8)
        scroll("corner_scroll", pts3)
        rivet_at.append(((0, -(TP + TB), cA3[1]), (90, 0, 0)))
        joints.append((TP, cA3[1]))
        collar(cB3[0], -TB / 2, 0, 1, 2 * A + TB + 0.007)
        zb_plate = cA3[1] - 0.06

    # ------------------------------------------------------------------ back plate, rosette, coach bolts
    hw = 0.028
    tipl = 0.075
    ol = [(0, zb_plate - tipl), (hw * 0.7, zb_plate - 0.048), (hw * 1.6, zb_plate - 0.026), (hw, zb_plate),
          (hw, zt_plate), (hw * 1.6, zt_plate + 0.026), (hw * 0.7, zt_plate + 0.048), (0, zt_plate + tipl)]
    ol = ol + [(-x, z) for x, z in ol[-2:0:-1]]
    plate = _relief("back_plate", _chaikin(_ccw(ol), 1), [], 0.0, TP, chamfer=0.002)
    densify(plate, 0.06)
    displace_noise(plate, strength=0.0006, scale=0.05, seed=seed)
    body_iron.append(plate)
    body_iron.append(box("rosette", (0.056, 0.012, 0.056), loc=(0, -(TP + 0.005), 0), rot=(0, 45, 0), bevel=0.003, segments=1))
    joints.append((TP, 0.0))
    bolt_z = [zt_plate + 0.024, 0.085, -0.085, zb_plate - 0.024]
    b0 = join([box("b0", (0.019, 0.019, 0.009), loc=(0, 0, 0.0055), bevel=0.0025, segments=1),
               cylinder("b1", 0.0145, 0.0022, loc=(0, 0, 0.0011), verts=10)], "b0")
    bolts = instances_to_mesh("coach_bolts", b0, [trs((rj.jitter(0, 0.002), -TP, z), (90, rj.u(0, 90), 0)) for z in bolt_z],
                              keep_base=False)
    fixings.append(bolts)

    # ------------------------------------------------------------------ the sign, built sign-local
    S = []                                # everything that swings
    faces = []                            # (obj) face decoration, duplicated to the back face
    rs = rng.fork("sign")
    yf = -T / 2

    z_text = None                                           # (lo, hi) of the lettering band, sign-local
    if cut:
        Rh = W / 2
        z_lo, z_hi = -Rh, Rh
        hoop = sweep("hoop", [(x, 0, z) for x, z in _arc(0, 0, Rh, Rh, 0, 360, 40)[:-1]], _rect(0, 0, 0.006, 0.032),
                     closed=True, smooth_path=False, up=(0, 1, 0))
        assign(hoop, iron)
        S.append(hoop)
        raw = _emblem(p.emblem)
        E = 0.84 * Rh / max(math.hypot(x, z) for o, _, _ in raw for x, z in o)
        pieces = [(_tf(o, s=E), [_tf(h, s=E) for h in hs_], lv) for o, hs_, lv in raw]
        for k, (o, hs_, lv) in enumerate(pieces):
            hd = 0.004 + 0.0015 * (lv + 1)
            for side in (0, 1):
                e = _relief(f"emblem{k}", o, hs_, 0.0, hd, chamfer=0.0015)
                if side:
                    e.matrix_basis = Matrix.Rotation(pi, 4, 'Z') @ e.matrix_basis
                    e.data.transform(Matrix.Diagonal((-1, 1, 1, 1)))
                    e.data.flip_normals()
                assign(e, accent)
                S.append(e)
        # stays from the hoop to the silhouette
        outs = [o for o, _, lv in pieces if lv >= 0]
        n_st = 0
        for a_deg in (90, 210, 330, 30, 150, 270):
            a = radians(a_deg + rs.jitter(0, 4))
            t = _ray_hit(outs, a, Rh - 0.02)
            if t is None or t < 0.03:
                continue
            st = sweep("stay", [((t - 0.012) * cos(a), 0, (t - 0.012) * sin(a)), (Rh * cos(a), 0, Rh * sin(a))], 0.0048,
                       sides=6, smooth_path=False)
            assign(st, iron)
            S.append(st)
            n_st += 1
            if n_st >= 4:
                break
        # name plate under the hoop
        hang = [(-0.5 * Rh, math.sqrt(Rh ** 2 - (0.5 * Rh) ** 2) + 0.011), (0.5 * Rh, math.sqrt(Rh ** 2 - (0.5 * Rh) ** 2) + 0.011)]
        staple_r = 0.014
        if p.text.strip():
            pw, ph = 1.72 * Rh, 0.115
            pz = -Rh - 0.028 - ph / 2
            npl = box("name_plate", (pw, 0.006, ph), loc=(0, 0, pz), bevel=0.0015, segments=1)
            densify(npl, 0.05)
            assign(npl, ground)
            S.append(npl)
            rim = _moulding("plate_rim", _ccw(_rect(0, pz, pw, ph)), -0.003, [(0.0, -0.001), (0.0, 0.003), (0.006, 0.003), (0.008, -0.001)])
            assign(rim, accent)
            faces.append(rim)
            tx = _lettering("lettering", p.text, p.font, ph * 0.56, pw - 0.05, -0.003, pz, depth=0.005)
            if tx:
                assign(tx, accent)
                faces.append(tx)
            for sx in (-1, 1):
                x = sx * 0.33 * pw
                zt = -math.sqrt(Rh ** 2 - x ** 2)
                link = box("plate_strap", (0.022, 0.0045, zt - pz + 0.035), loc=(x, -0.0052, (zt + pz + 0.035) / 2 - 0.004),
                           bevel=0.001, segments=1)
                assign(link, iron)
                faces.append(link)
                nail_at.append(((x, -0.0075, pz + 0.035), (90, 0, 0)))
                nail_at.append(((x, -0.0075, zt + 0.004), (90, 0, 0)))
            z_lo = pz - ph / 2
            z_text = (pz - ph * 0.3, pz + ph * 0.3)
        planks = []
        content = [(0.86 * Rh * cos(i * pi / 6), 0.86 * Rh * sin(i * pi / 6)) for i in range(12)]
        if p.text.strip():
            content += [(sx * pw / 2, pz + sz * ph / 2) for sx in (-1, 1) for sz in (-1, 1)]
    else:
        z_lo, z_hi = -Hb / 2, Hb / 2
        ol = _outline(p.shape, W, Hb)
        x0, x1 = min(x for x, _ in ol), max(x for x, _ in ol)
        # planks
        npl = max(3, int(round(W / 0.15)))
        ws = [rs.vary(1.0, 0.22) for _ in range(npl)]
        edges = [x0 + 0.0012]
        for w in ws:
            edges.append(edges[-1] + w / sum(ws) * (x1 - x0 - 0.0024))
        fine = [x0 + e for e in (0.006, 0.013, 0.022, 0.034, 0.048)] + [x1 - e for e in (0.006, 0.013, 0.022, 0.034, 0.048)]
        if p.shape == "shield":
            fine.append(0.0)
        rows = max(4, int(round(Hb / 0.054)))
        planks = []
        for k in range(npl):
            g0 = 0.0009 if k > 0 else 0.0
            g1 = 0.0009 if k < npl - 1 else 0.0
            pk = _plank(f"plank{k}", ol, edges[k] + g0, edges[k + 1] - g1, T, fine, rows)
            pk.location.y = rs.jitter(0, 0.0009)
            assign(pk, ground)
            planks.append(pk)
        S += planks
        core = shape_extrude("tongues", _inset(ol, 0.012), depth=T * 0.4, loc=(0, T * 0.2, 0), rot=(90, 0, 0))
        assign(core, core_m)
        S.append(core)
        # applied frame moulding (laps the board edge) on the face
        fr = _moulding("frame", ol, yf + 0.001, [(0.001, -0.003), (-0.003, -0.001), (-0.003, 0.0065), (-0.0005, 0.0105),
                                                (0.006, 0.0105), (0.0105, 0.0072), (0.024, 0.0072), (0.03, -0.002)])
        assign(fr, accent)
        faces.append(fr)
        # layout
        inner = _inset(ol, 0.036)
        has_text = bool(p.text.strip())
        script = p.font == "script"
        fat = 0.0024 if script else 0.0                      # a thin script is cut bolder so it carries across a square
        emb = _emblem(p.emblem)
        tx = None
        safe = _inset(ol, 0.036 + 0.011)                    # clear of the frame moulding
        e_lo, e_hi = -Hb / 2 + 0.05, Hb / 2 - 0.05
        e_max = 0.8 * min(Hb, 1.25 * W)
        if p.shape == "oval":
            # lettering bowed along the lower rim, as large as the oval allows; the emblem fills the rest
            keep_out = []
            if has_text:
                cap = clamp(0.125 * Hb, 0.055, 0.095)
                arc = max(0.19, 0.56 * W)
                for shrink in (1.0, 0.92, 0.85, 0.78, 0.7, 0.62, 0.54):
                    tx = _lettering("lettering", p.text, p.font, cap * shrink, 1.25 * arc, yf + 0.001, 0.0, fat=fat, arc_R=arc)
                    if tx is None:
                        break
                    pts2 = [(v.co.x, v.co.y) for v in tx.data.vertices]
                    lo2, hi2 = min(q[1] for q in pts2), max(q[1] for q in pts2)
                    tz = None
                    for i in range(40):
                        zt_ = -Hb / 2 + 0.05 + 0.004 * i - lo2
                        if zt_ + hi2 > -0.07 * Hb:
                            break
                        if _inside(safe, [(x, y + zt_) for x, y in pts2[::3]]):
                            tz = zt_
                            break
                    if tz is not None:
                        tx.location.z = tz
                        z_text = (tz + lo2, tz + hi2)
                        e_lo = tz + hi2 + 0.016
                        break
                    delete([tx])
                    tx = None
        else:
            # the hanger straps come 64 mm down the face: lettering and emblem keep clear of them
            keep_out = [(sx * hs - 0.03, sx * hs + 0.03, Hb / 2 - 0.082, Hb) for sx in (-1, 1)]
            if p.shape == "shield":
                cap = clamp(0.13 * Hb, 0.06, 0.1)
                tz = Hb / 2 - 0.082 - cap / 2                  # name across the shoulders, below the strap ends
            else:
                cap = clamp(0.14 * Hb, 0.06, 0.105) * (1.2 if script else 1.0)
                tz = -Hb / 2 + (0.17 if script else 0.15) * Hb # name along the foot of the board
            if has_text:
                xa = _xspan(inner, tz - cap * 0.55)
                xb = _xspan(inner, tz + cap * 0.55)
                maxw = 2 * min(-xa[0], xa[1], -xb[0], xb[1]) - 0.02
                tx = _lettering("lettering", p.text, p.font, cap, maxw, yf + 0.001, tz, fat=fat)
                if tx:
                    zs_ = [v.co.y for v in tx.data.vertices]
                    z_text = (tz + min(zs_), tz + max(zs_))
                    if p.shape == "shield":
                        e_hi = z_text[0] - 0.02
                    else:
                        e_lo = z_text[1] + 0.022
        # largest emblem that fits what is left of the board: every point of the artwork inside the frame
        E, ex, ez = _fit(safe, [q for o, _, _ in emb for q in o], e_lo, e_hi, e_max, keep_out)
        # what the sign lamps have to light: the whole board bar its top corners, the name and the emblem
        content = [q for q in _inset(ol, 0.03) if q[1] < z_hi - 0.25 * Hb]
        eq = _tf([q for o, _, _ in emb for q in o], s=E, ox=ex, oz=ez)
        content += [(fx(q[0] for q in eq), fz(q[1] for q in eq)) for fx in (min, max) for fz in (min, max)]
        if tx:
            assign(tx, accent)
            faces.append(tx)
            xs_ = [v.co.x for v in tx.data.vertices]
            content += [(fx(xs_), zz) for fx in (min, max) for zz in z_text]
        for k, (o, hs_, lv) in enumerate(emb):
            e = _relief(f"emblem{k}", _tf(o, s=E, ox=ex, oz=ez), [_tf(h, s=E, ox=ex, oz=ez) for h in hs_], yf + 0.001,
                        RELIEF + 0.004 * lv, chamfer=0.0034 if p.gilded else 0.0028, draft=0.0012)
            assign(e, accent)
            faces.append(e)
        # hangers
        if p.shape == "oval":
            staple_r = 0.015
            zt = _zspan(ol, hs)[1]
            hang = [(-hs, zt + 0.004 + staple_r - 0.0045), (hs, zt + 0.004 + staple_r - 0.0045)]
        else:
            staple_r = 0.0
            yo1, yo2 = T / 2 + 0.0105 + 0.0025, T / 2 + 0.0025
            top = Hb / 2
            leg = [(-yo2, top - 0.064), (-yo2, top - 0.047), (-yo1, top - 0.035), (-yo1, top + 0.012)]
            arch = [(yo1 * cos(radians(a)), top + 0.012 + yo1 * sin(radians(a))) for a in (157.5, 135, 112.5, 90, 67.5, 45, 22.5)]
            path = leg + [(y, z) for y, z in arch] + [(-y, z) for y, z in leg[::-1]]
            for sx in (-1, 1):
                stp = sweep("hanger_strap", [(sx * hs, y, z) for y, z in path], _rect(0, 0, 0.004, 0.03),
                            smooth_path=False, up=(1, 0, 0))
                assign(stp, iron)
                S.append(stp)
                nail_at.append(((sx * hs + rs.jitter(0, 0.002), -(yo2 + 0.002), top - 0.0555), (90, 0, 0)))
                nail_at.append(((sx * hs + rs.jitter(0, 0.002), -(yo1 + 0.002), top - 0.016), (90, 0, 0)))
            hang = [(-hs, top + 0.012 + yo1 - 0.002), (hs, top + 0.012 + yo1 - 0.002)]

    if staple_r:
        for hx, hz in hang:
            st = torus("staple", staple_r, 0.0045, loc=(hx, 0, hz - (staple_r - 0.0045)), rot=(0, 90, 0), major_seg=10,
                       minor_seg=4)
            assign(st, iron)
            S.append(st)

    # nails / rivets on the sign face (front), then mirror every face object to the back
    if nail_at:
        n0 = bolt("n0", 0.0062, 0.0042, kind="pyramid")
        nl = instances_to_mesh("strap_nails", n0, [trs(l, (r[0], rs.u(0, 90), r[2])) for l, r in nail_at], keep_base=False)
        assign(nl, iron)
        faces.append(nl)
    for o in faces:
        d = duplicate(o, o.name + "_back")
        d.matrix_basis = Matrix.Rotation(pi, 4, 'Z') @ d.matrix_basis
        S += [o, d]

    # ------------------------------------------------------------------ hang the sign: rings, chains, placement
    wr, Lk, n_links = 0.004, 0.052, 3
    ring_R, ring_r = 0.024, 0.005
    z_ring = A - (ring_R - ring_r)                          # ring rests on top of the arm
    z_contact = z_ring - (ring_R - ring_r)                  # inside bottom of the ring
    z_apex = z_contact - n_links * (Lk - 4 * wr)            # where the strap / staple hangs on the last link
    zc = z_apex - hang[0][1]                                # board centre height
    yaw = rs.jitter(0, 1.4)
    M = Matrix.Translation((0, -dc, zc)) @ Matrix.Rotation(radians(90 + yaw), 4, 'Z')
    for o in S:
        o.matrix_basis = M @ o.matrix_basis

    hw_l, hl = Lk * 0.31 - wr, Lk / 2 - wr
    link0 = sweep("link0", [(hl * cos(2 * pi * i / 10) * (1.0 + 0.12 * abs(sin(2 * pi * i / 10))), hw_l * sin(2 * pi * i / 10), 0)
                            for i in range(10)], wr, closed=True, sides=4, smooth_path=False)
    mats = []
    for hx, hz in hang:
        w = M @ Vector((hx, 0, hz))
        top_pt = Vector((0, w.y, z_contact))
        t = (w - top_pt).normalized()
        e1 = t.cross(Vector((0, 1, 0))).normalized()
        e2 = e1.cross(t).normalized()
        step = (w - top_pt).length / n_links
        for k in range(n_links):
            c = top_pt + t * (step * (k + 0.5))
            wv = e2 if k % 2 == 0 else e1
            roll = Matrix.Rotation(radians(rs.jitter(0, 9)), 4, t)
            m = Matrix((t, wv, t.cross(wv))).transposed().to_4x4()
            mats.append(Matrix.Translation(c) @ roll @ m)
        rg = torus("hang_ring", ring_R, ring_r, loc=(0, w.y, z_ring), rot=(90, rs.jitter(0, 6), 0), major_seg=12, minor_seg=5)
        assign(rg, iron)
        S.append(rg)
        for sy in (-1, 1):                                  # forged stops keep the ring from wandering
            body_iron.append(box("stop", (2 * A + 0.009, 0.007, 2 * A + 0.009), loc=(0, w.y + sy * 0.0125, 0)))
    chains = instances_to_mesh("chains", link0, mats, uv_jitter=False, keep_base=False)
    smooth(chains, 180)
    assign(chains, iron)
    S.append(chains)

    # ------------------------------------------------------------------ swan-neck sign lamps
    # A two-plate forged saddle clamp bolted onto the plain square of the arm just outboard of the
    # scrollwork carries a turned stem with a tee; two tubular goosenecks spring from it, one to each
    # face of the sign, each ending in a brass lampholder with a gallery ring and a deep spun shade
    # with a rolled rim.  The flex is tied along the top of the stay and enters the stem at a gland.
    brass_objs, cable_objs, enamel_objs = [], [], []
    if lamp_on:
        enamel = mat("iron_painted", "#1f2a22")              # stove-enamelled steel shade, green-black outside ...
        enamel_in = mat("iron_painted", "#f1e3c2", rough=1.4)    # ... and cream inside to throw the light
        lx = clamp(0.27 + 0.1 * (Hb - 0.6), 0.26, 0.3)
        z_l = 0.15
        cw, pt_, phw = 0.042, 0.0065, A + 0.0185
        for sz in (1, -1):
            body_iron.append(box("clamp_plate", (2 * phw, cw, pt_), loc=(0, -dl, sz * (A + pt_ / 2)), bevel=0.002, segments=1))
        for sx in (1, -1):
            xb_ = sx * (A + 0.0095)
            body_iron.append(cylinder("clamp_shank", 0.0032, 2 * A, loc=(xb_, -dl, 0), verts=6, cap=False))
            fixings.append(cylinder("clamp_bolt", 0.0066, 0.005, loc=(xb_, -dl, A + pt_ + 0.0025), rot=(0, 0, rj.u(0, 60)), verts=6))
            fixings.append(box("clamp_nut", (0.0115, 0.0115, 0.0058), loc=(xb_, -dl, -(A + pt_ + 0.0029)), rot=(0, 0, rj.u(0, 90))))
        zs0 = A + pt_
        z_tee = zs0 + 0.044
        body_iron.append(lathe("clamp_stem", [(0.0125, 0.0), (0.0125, 0.004), (0.0098, 0.009), (0.0098, 0.026), (0.0128, 0.03),
                                              (0.0128, 0.036)], segments=8, loc=(0, -dl, zs0)))
        body_iron.append(cylinder("clamp_tee", 0.0118, 0.07, loc=(0, -dl, z_tee), rot=(0, 90, 0), verts=8))
        # profiles along the lamp axis: crown of the shade at z = 0, mouth towards -z
        shade_in = [(0.0, -0.011), (0.019, -0.013), (0.0655, -0.104), (0.068, -0.112)]
        shade_out = [(0.068, -0.112), (0.0712, -0.1142), (0.0728, -0.1098), (0.0668, -0.1035),
                     (0.0245, -0.016), (0.0215, -0.006), (0.0215, 0.0)]
        holder_prof = [(0.0215, -0.003), (0.0248, -0.001), (0.0248, 0.004), (0.0165, 0.007), (0.0165, 0.03), (0.0188, 0.031),
                       (0.0188, 0.036), (0.013, 0.041), (0.0098, 0.051)]
        bulb_prof = [(0.0, -0.121), (0.015, -0.1165), (0.0245, -0.1), (0.0235, -0.088), (0.013, -0.062), (0.0115, -0.03)]
        z_src = -0.022                                      # light source, deep in the crown
        half = math.atan2(0.068, 0.112 + z_src)             # beam half angle left by the rim of the shade
        bulb_m = mat("emissive", "#ffd9a8", glow=6.0)
        key = [M @ Vector((x, 0, z)) for x, z in content]
        far_pt = Vector((0, -dc, zc + z_lo))

        def beam(P, tgt, k):
            """Aim the shade at tgt: (axis, source, slack of the worst lit point, nearest wall in the beam)."""
            ax = (P - tgt).normalized()
            lp = P + ax * z_src
            slack = min((q - lp).normalized().dot(-ax) for q in key) - cos(half - radians(3.0))
            # the arm, its rings and the clamp sit a hand's breadth from the bulb: the rim has to shade them,
            # or they burn out as the brightest thing on the sign
            # (the beam looks down, so it is the LOWEST edges of the bar, nuts and stops that it finds first)
            bar = max((Vector((k * bx_, -dl + 0.06 * t, bz_)) - lp).normalized().dot(-ax)
                      for t in range(-6, 7) for bx_, bz_ in ((A + 0.016, -(A + 0.014)), (0.006, -0.034)))
            slack = min(slack, cos(half + radians(2.5)) - bar)
            e1 = ax.cross(Vector((0, 0, 1))).normalized()
            e2 = ax.cross(e1)
            wall = 9.0
            for i in range(36):
                dr = -ax * cos(half) + (e1 * cos(i * pi / 18) + e2 * sin(i * pi / 18)) * sin(half)
                if dr.y > 1e-3:
                    wall = min(wall, -lp.y / dr.y)
            return ax, lp, slack, wall

        for k in (-1, 1):
            P = Vector((k * lx, -dl, z_l))                  # crown of the shade
            # The beam also reaches the wall behind the sign, and pale plaster answers a lamp three times
            # as brightly as a painted board.  So the shade is turned outboard as far as it can go while
            # its rim still clears the name and the emblem, and the range is cut just short of the nearest
            # wall left in the beam: the light dies away softly before it gets there.
            far = (P - far_pt).length
            best = None
            for i in range(18):
                for j in range(20):
                    tgt = Vector((0, -(dc - 0.2 + 0.03 * i), zc + z_lo + (z_hi - z_lo) * (-1.05 + 0.1 * j)))
                    ax, lp, slack, wall = beam(P, tgt, k)
                    score = (1, min(wall, 1.45 * far), slack) if slack >= 0 else (0, slack, 0)
                    if best is None or score > best[0]:
                        best = (score, ax, lp, wall)
            _, ax, lp, wall = best
            top = P + ax * 0.051
            p0 = Vector((k * 0.035, -dl, z_tee))
            c1 = p0 + Vector((k * 0.085, 0, 0.075))
            neck = sweep("lamp_neck", bezier_path(p0, c1, top + ax * 0.12, top, n=9), 0.0088, sides=8, smooth_path=False)
            body_iron.append(neck)
            body_iron.append(cylinder("neck_collar", 0.0138, 0.011, loc=(k * 0.0335, -dl, z_tee), rot=(0, 90, 0), verts=8))
            so = _aim(_facing(lathe("lamp_shade", shade_out, segments=32), False), P, ax)
            si = _aim(_facing(lathe("lamp_shade_in", shade_in, segments=32), True), P, ax)      # lining looks at the bulb
            assign(so, enamel)
            assign(si, enamel_in)
            enamel_objs += [so, si]
            ho = _aim(lathe("lamp_holder", holder_prof, segments=8), P, ax)
            assign(ho, brass)
            brass_objs.append(ho)
            bulb = _aim(lathe("lamp_bulb", bulb_prof, segments=8), P, ax)
            assign(bulb, bulb_m)
            light(f"sign_lamp_{'l' if k < 0 else 'r'}", tuple(lp), color="#ffcf9a", intensity=p.lamp_cd,
                  range=clamp(min(1.03 * wall, 1.5 * far), 0.9, 1.7), flicker=0.03, shadow=True, shadow_near=0.006)
        # flex: wall grommet -> tied along the flank of the inner scroll (or along the top of the tie bar)
        # -> a short slack hop above the arm -> brass gland on the tee.  It never crosses the open scrollwork.
        cr_ = 0.003
        tie_at = []                                         # (x, d, z, normal d, normal z, width, radial)
        if p.bracket == "scroll":
            xc = BW / 2 + cr_ + 0.0004
            z_leave = z_tee + 0.06
            run_ = []
            for i in range(1, len(mid2)):
                (da, za), (db, zb) = mid2[i - 1], mid2[i]
                if i > len(mid2) // 2 and zb < z_leave <= za:        # coming down to the arm: leave the scroll here
                    t = (za - z_leave) / (za - zb)
                    run_.append((da + (db - da) * t, z_leave))
                    break
                run_.append((db, zb))
            ud, uz = _unit(mid2[i - 1], mid2[i])
            zg = mid2[0][1] + 0.014
            gx = xc + 0.03
            path = [(gx, -0.004, zg), (gx - 0.009, -0.03, zg - 0.005)] + [(xc, -d, z) for d, z in run_]
            pa = Vector(path[-1])
            pb = Vector((0.0, -(dl - 0.0118 - 0.014), z_tee))        # mouth of the gland, on the wall side of the tee
            gland_at, gland_dir = (0.0, -(dl - 0.0108), z_tee), (0, 1, 0)
            hop = (pb - pa).length
            sag = Vector((0.0, 0.42 * hop, -0.003 - 0.06 * hop))
            n_ties = (1, len(run_) // 2, len(run_) - 2)
            for i in sorted(set(n_ties)):
                a_, b_ = run_[max(i - 1, 0)], run_[min(i + 1, len(run_) - 1)]
                td, tz_ = _unit(a_, b_)
                tie_at.append((cr_ * 0.5 + 0.001, run_[i][0], run_[i][1], -tz_, td, BW + 2 * cr_ + 0.008, TB + 0.006))
        else:
            t_end = clamp((dl - 0.055 - p0_[0]) / (ud_ * L_), 0.2, 0.95)
            line = []
            for i in range(7):
                t = 0.05 + (t_end - 0.05) * i / 6
                off = 0.0075 + cr_ + 0.0006 + (0.0125 * math.exp(-((t - 0.5) / 0.09) ** 2))
                line.append((p0_[0] + ud_ * L_ * t - uz_ * off, p0_[1] + uz_ * L_ * t + ud_ * off))
                if i in (1, 5):
                    tie_at.append((0.0, line[-1][0] + uz_ * cr_, line[-1][1] - ud_ * cr_, -uz_, ud_, 0.021, 0.026))
            ud, uz = ud_, uz_
            gx = 0.06
            zg = line[0][1] + 0.025
            path = [(gx, -0.004, zg), (gx * 0.8, -0.03, zg - 0.006)] + [(0.0, -d, z) for d, z in line]
            pa = Vector(path[-1])
            pb = Vector((0.0, -dl, z_tee + 0.0118 + 0.014))          # gland on top of the tee
            gland_at, gland_dir = (0.0, -dl, z_tee + 0.0108), (0, 0, 1)
            hop = (pb - pa).length
            sag = Vector((0.0, 0.0, 0.5 * hop))
        head = resample_path(path, n=max(8, int(path_length(path) / 0.024)))
        tail = bezier_path(pa, pa + Vector((0.0, -ud, uz)) * 0.4 * hop, pb + sag, pb, n=max(6, int(hop / 0.014)))
        cable = sweep("cable", [tuple(q) for q in head[:-1]] + [tuple(q) for q in tail], cr_, sides=4, smooth_path=False)
        smooth(cable, 180)
        cable_objs.append(cable)
        assign(cable, mat("rope", "#33291f"))
        gl = _aim(lathe("cable_gland", [(0.0082, 0.0), (0.0082, 0.0045), (0.0062, 0.006), (0.0062, 0.012), (0.0044, 0.0155)],
                        segments=8), gland_at, gland_dir)
        assign(gl, brass)
        brass_objs.append(gl)
        body_iron.append(cylinder("grommet", 0.0115, 0.007, loc=(gx, -0.0035, zg), rot=(90, 0, 0), verts=8))
        for x, d, z, nd, nz, wd, rad in tie_at:
            body_iron.append(box("cable_tie", (wd, 0.0055, rad), loc=(x, -d, z), rot=(degrees(math.atan2(nd, nz)), 0, 0)))

    # ------------------------------------------------------------------ fixings stamped in one mesh each
    if rivet_at:
        r0 = lathe("r0", [(0.0078, 0.0), (0.0068, 0.0034), (0.0038, 0.0056), (0.0, 0.0062)], segments=8)
        fixings.append(instances_to_mesh("rivets", r0, [trs(l, r) for l, r in rivet_at], keep_base=False))

    for o in body_iron + fixings:
        assign(o, iron)

    # ------------------------------------------------------------------ weathering (world space)
    # ONE rust story for every piece of iron: it starts where water is held - collars, rivets, the tenon,
    # scroll eyes, the foot of the back plate, weeping from the bolts - and only blooms faintly on the
    # long runs, which stay graphite dark.  Where it shows it is old, dark and brown, not fresh orange.
    def near(pts, r):
        def f(co):
            m = np.zeros(len(co))
            for d, z in pts:
                m = np.maximum(m, np.exp(-(((co[:, 1] + d) ** 2 + (co[:, 2] - z) ** 2) / r ** 2)))
            return m
        return f

    joint_m, trap_m = near(joints, 0.045), near(traps, 0.048)
    blotch = noise_mask(0.075, seed=seed + 3, lo=0.45, hi=0.75)
    finen = noise_mask(0.02, seed=seed + 4, lo=0.3, hi=0.8)
    patina = noise_mask(0.05, seed=seed + 5, lo=0.25, hi=0.8)

    def lamp_zone(co):                                      # ironwork right under the lamps is kept oiled and dark
        if not lamp_on:
            return np.zeros(len(co))
        return np.exp(-(((co[:, 1] + dl) / 0.2) ** 2 + (co[:, 2] / 0.16) ** 2))

    def rust(co, n, jn=1.0, bl=1.0):
        up = np.clip(n[:, 2], 0.0, 1.0)
        fn = finen(co, n)
        foot = np.clip((zb_plate + 0.05 - co[:, 2]) / 0.12, 0.0, 1.0) * np.clip(1.0 - (-co[:, 1]) / 0.04, 0.0, 1.0)
        m = (0.05 + 0.1 * age) * fn * (0.4 + 0.6 * blotch(co, n))
        m = m + jn * (0.65 + 0.3 * age) * joint_m(co) * (0.55 + 0.45 * fn)
        m = m + (0.4 + 0.4 * age) * trap_m(co) * (0.4 + 0.6 * up * bl) * (0.5 + 0.5 * fn)
        m = m + bl * (0.08 + 0.3 * age) * blotch(co, n) * up ** 2
        m = m + (0.45 + 0.4 * age) * foot * (0.5 + 0.5 * fn)
        # ... and it keeps to those places: the faint bloom between them is pushed back to sound iron
        return 0.92 * smoothstep_np(0.1, 0.62, m * (1.0 - 0.85 * lamp_zone(co)))

    def plate_rust(co, n):                                  # ... and weeps down the plate from every bolt
        m = np.zeros(len(co))
        for z in bolt_z + [0.0]:
            dz = z - co[:, 2]
            m = np.maximum(m, np.exp(-(co[:, 0] / 0.012) ** 2) * np.clip(dz / 0.012, 0.0, 1.0) * np.exp(-np.clip(dz, 0, None) / 0.065))
        return np.clip(np.maximum(m * (0.3 + 0.4 * age), rust(co, n, 0.6)), 0.0, 0.92)

    def ring_rub(co, n):                                    # the hanging rings keep the arm rubbed clean under them
        m = np.zeros(len(co))
        for hx, hz in hang:
            y = (M @ Vector((hx, 0, hz))).y
            m = np.maximum(m, np.exp(-((co[:, 1] - y) / 0.016) ** 2))
        return m

    def iron_paint(o, r, extra_grime=None, dark=0.45):
        """r(co, n) = how rusty.  Sound iron (the material's wear state) everywhere else; the rust itself
        is pulled dark and brown, and dust lies on what faces up."""
        def grime(co, n):
            g = 0.25 * np.clip(n[:, 2], 0.0, 1.0) ** 2 + 0.2 * np.clip(-n[:, 2], 0.0, 1.0) + dark * r(co, n)
            if extra_grime is not None:
                g = np.maximum(g, extra_grime(co, n))
            return np.clip(g, 0.0, 1.0)
        # between the joints the bar keeps its forge scale: sound dark iron with a thin brown oxide skin
        # just showing through it in soft patches (bare metal alone reads as a black hole by day)
        skin = lambda co, n: 0.08 + 0.14 * patina(co, n)
        paint(o, wear=lambda co, n: np.clip(1.0 - skin(co, n) - 1.1 * r(co, n), 0.03, 1.0),
              special=lambda co, n: np.clip(0.46 + 0.14 * r(co, n), 0.0, 1.0), grime=grime)

    oiled = lambda co, n: 0.9 * lamp_zone(co) ** 0.5       # lamp black and oil on the bar right under the lamps
    for o in body_iron:
        if o is plate:
            iron_paint(o, plate_rust, mask_mul(noise_mask(0.08, seed=seed + 6), 0.3 + 0.3 * age), dark=0.6)
        elif o.name.startswith(("arm_twist", "tie")):
            # nothing here may depend on which way a facet looks, or the twist turns into a barber's pole
            iron_paint(o, lambda co, n: rust(co, n, 1.0, 0.0), oiled)
        elif o.name.startswith(("arm", "stop")):
            iron_paint(o, lambda co, n: rust(co, n) * (1.0 - 0.8 * ring_rub(co, n)), oiled)
        elif o.name.startswith(("clamp", "lamp_neck", "neck_collar", "cable_tie", "grommet")):
            iron_paint(o, lambda co, n: 0.4 * rust(co, n, 0.5, 0.0), lambda co, n: 0.3 + 0.5 * lamp_zone(co))   # newer fitting
        else:
            iron_paint(o, rust)
    for o in fixings:                                       # heads of rivets and bolts always go first
        iron_paint(o, lambda co, n: np.clip((0.45 + 0.4 * age) * (0.5 + 0.5 * finen(co, n)) * (1.0 - 0.85 * lamp_zone(co)), 0.0, 0.9),
                   lambda co, n: 0.25 + 0.5 * lamp_zone(co))
    for o in brass_objs:
        paint(o, special=mask_mul(facing((0, 0, 1)), noise_mask(0.05, seed=seed + 8, lo=0.3, hi=0.7), 0.3 + 0.7 * age),
              grime=mask_mul(noise_mask(0.06, seed=seed + 9), 0.25 + 0.3 * age),
              wear=mask_mul(noise_mask(0.03, seed=seed + 10, lo=0.4, hi=0.8), 0.45))
    for o in cable_objs:
        paint(o, grime=0.35)
    for o in enamel_objs:
        inner_ = o.name.startswith("lamp_shade_in")
        paint(o, wear=mask_mul(noise_mask(0.025, seed=seed + 20, lo=0.45, hi=0.8), 0.0 if inner_ else 0.25 + 0.35 * age),
              special=mask_mul(noise_mask(0.04, seed=seed + 21, lo=0.5, hi=0.8), 0.0 if inner_ else 0.5 * age),
              grime=(0.15 if inner_ else mask_mul(facing((0, 0, 1)), 0.3 + 0.4 * age)))
    z_top_w, z_bot_w = zc + z_hi, zc + z_lo
    y_c = -dc
    light_gnd = smoothstep(0.16, 0.5, lum)                  # pale boards show dirt and chips far more than dark ones

    def streak(sd, sx=0.028, sz=0.5, lo=0.45, hi=0.75):
        f = noise_mask(1.0, seed=sd, lo=lo, hi=hi)
        return lambda co, n: f(co / np.array([sx, sx, sz]), n)

    patch = noise_mask(0.2, seed=seed + 12, lo=0.3, hi=0.75)
    chipn = noise_mask(0.05, seed=seed + 13, lo=0.5, hi=0.7)
    speck = noise_mask(0.03, seed=seed + 15, lo=0.38, hi=0.72)
    letgo = noise_mask(0.17, seed=seed + 16, lo=0.3, hi=0.7)
    run = streak(seed + 14)
    if cut:
        sign_ol = _rect(0, z_lo + 0.0575, 1.72 * W / 2, 0.115) if p.text.strip() else None
        joint_x = []
    else:
        sign_ol = ol
        joint_x = edges[1:-1]
    seg_a = np.array(sign_ol) if sign_ol else None
    seg_b = np.roll(seg_a, -1, axis=0) if sign_ol else None

    def hgt(co):                                            # 0 at the bottom of the sign, 1 at the top
        return np.clip((co[:, 2] - z_bot_w) / max(z_top_w - z_bot_w, 1e-3), 0.0, 1.0)

    def rim_dist(xl, zl):                                   # distance (m) to the edge of the board outline
        q = np.stack([xl, zl], axis=1)[:, None, :]
        ab = (seg_b - seg_a)[None, :, :]
        t = np.clip(((q - seg_a[None]) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
        d = q - (seg_a[None] + t[..., None] * ab)
        return np.sqrt((d * d).sum(-1)).min(axis=1)

    def text_band(zl, soft=0.025):                          # 1 over the lettering band, 0 away from it
        if z_text is None:
            return np.zeros(len(zl))
        return np.clip(1.0 - np.maximum(np.maximum(z_text[0] - zl, zl - z_text[1]), 0.0) / soft, 0.0, 1.0)

    def fade(co, n):                                        # sun and rain bleach the paint from the top down
        return np.clip((0.25 + 0.75 * patch(co, n)) * (0.45 + 0.55 * hgt(co)) * (0.35 + 1.25 * age), 0.0, 1.0)

    def chips(co, n):
        """Paint letting go in small flakes: along the plank joints, round the rim and along the drip
        edge at the bottom - never as blotches in the middle of a board, and held back over the
        lettering so the name stays readable however old the sign is."""
        xl, zl = (co[:, 1] - y_c), co[:, 2] - zc
        e = np.exp(-rim_dist(xl, zl) / 0.022)
        for jx in joint_x:
            e = np.maximum(e, 0.8 * np.exp(-np.abs(xl - jx) / 0.009))
        e = np.maximum(e, 0.65 * np.clip(1.0 - hgt(co) / 0.16, 0.0, 1.0))
        m = speck(co, n) * (0.1 + 0.9 * e) * (0.45 + 0.55 * letgo(co, n))
        m *= 1.0 - 0.8 * text_band(zl)
        return np.clip(m * (0.3 + 0.9 * age), 0.0, 0.85 - 0.3 * light_gnd)

    def dirt(co, n):                                        # rain streaks from the top edge / hangers, drip line at the bottom
        h = hgt(co)
        hang_m = np.zeros(len(co))
        for hx, hz in hang:
            y = (M @ Vector((hx, 0, hz))).y
            hang_m = np.maximum(hang_m, np.exp(-((co[:, 1] - y) / 0.03) ** 2) * np.clip((h - 0.35) / 0.4, 0.0, 1.0))
        drip = 0.75 * np.clip(1 - h / 0.1, 0.0, 1.0) * (1.0 - (0.55 + 0.4 * light_gnd) * text_band(co[:, 2] - zc, 0.03))
        m = 0.55 * run(co, n) * h + 0.8 * hang_m * (0.4 + 0.6 * run(co, n)) + drip
        m = m * (0.25 + 1.2 * age) * (1.0 - 0.5 * light_gnd)
        m = m * (1.0 - 0.6 * light_gnd * text_band(co[:, 2] - zc, 0.03))     # a pale board is kept wiped where the name is
        return np.clip(np.minimum(m, 0.9 - 0.4 * light_gnd) + soot(co, n), 0.0, 1.0)

    def soot(co, n):                                        # dust and soot settle on every upward ledge
        ledge = np.clip((co[:, 2] - (z_top_w - 0.016)) / 0.012, 0.0, 1.0) if not cut else 0.0
        return np.maximum(0.85 * np.clip(n[:, 2], 0.0, 1.0) ** 2, 0.8 * ledge) * (1.0 - 0.35 * light_gnd)

    nb = (M.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()          # normal of the board face (world)
    goldn = noise_mask(0.035, seed=seed + 17, lo=0.3, hi=0.75)

    def face_on(n):                                         # 1 on the faces of the relief, 0 on its flanks
        return np.abs(n[:, 0] * nb.x + n[:, 1] * nb.y + n[:, 2] * nb.z)

    def gold_wear(co, n):
        """Satin leaf on the faces, burnished bright in rubbed patches and on every arris (the export
        adds the arris wear); dull down the flanks, where the burnisher never reached."""
        f = face_on(n)
        return np.clip((0.12 + (0.3 + 0.25 * age) * goldn(co, n)) * smoothstep_np(0.25, 0.8, f), 0.0, 1.0)

    def gold_grime(co, n):                                  # toned-down, warm recesses; dust on the ledges
        f = face_on(n)
        return np.clip(np.maximum((0.7 + 0.2 * age) * (1.0 - smoothstep_np(0.08, 0.55, f)),
                                  np.maximum(0.2 * dirt(co, n), 0.5 * soot(co, n))), 0.0, 1.0)

    iron_sign = []
    for o in S:
        ms = o.data.materials[0] if o.data.materials else None
        if o in planks or (cut and o.name.startswith("name_plate")):
            paint(o, special=fade, wear=chips, grime=dirt)
        elif ms is accent and p.gilded:
            paint(o, wear=gold_wear, grime=gold_grime)
        elif ms is accent:
            paint(o, wear=lambda co, n: chipn(co, n) * (0.15 + 0.5 * age) * (1.0 - 0.6 * text_band(co[:, 2] - zc, 0.01)),
                  grime=lambda co, n: np.maximum(0.6 * dirt(co, n), soot(co, n)), special=mask_mul(fade, 0.6))
        elif ms is iron:
            iron_sign.append(o)
    # straps, staples, rings, chains, hoop: the same iron and the same story for both hangers - rust in
    # the bend of the straps and where the links bear on each other, nothing that depends on which side
    def sign_rust(co, n):
        zr = np.clip((co[:, 2] - (z_top_w - 0.03)) / 0.05, 0.0, 1.0) * np.clip((-0.02 - co[:, 2]) / 0.03, 0.0, 1.0)
        m = (0.1 + 0.2 * age) * (0.4 + 0.6 * finen(co, n)) + (0.2 + 0.3 * age) * zr * (0.5 + 0.5 * finen(co, n))
        if cut:
            m = m + rust(co, n, 0.0, 0.6)
        return np.clip(m, 0.0, 0.85)

    for o in iron_sign:
        iron_paint(o, sign_rust, dark=0.7)

    # ------------------------------------------------------------------ parts and sockets
    part(S, "sign", pivot=(0, -dc, 0), motion="swing", amp=2.5, speed=0.32)
    socket("tip", "attach", (0, -R - 0.01, -A), role="bracket_tip")
    socket("under_arm", "attach", (0, -max(0.12, wall_gap * 0.5), -A), role="lantern")

