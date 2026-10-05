"""Solid dressed-stone / brick town house of two or three storeys on a rough-dressed stone plinth.
Construction: four masonry walls built as welded grids with real window openings (channelled,
block-jointed rustication on the ground floor), alternating long-and-short quoins at the corners,
moulded string courses at every floor and a full eaves entablature (frieze, modillion or dentil
course, corona) under the gutter. Sash windows are modelled in the openings (stone architrave, sill
with drip, keystone / cornice hood / pediment heads, box frame, two offset sashes with glazing bars,
lit or dark glazing). The arched front door stands up a flight of stone steps with iron rails, a
wrought-iron balcony with a French window hangs above it, a wall lantern lights the stoop. The roof
(parapet side gables, front gable or hip; slate or clay tile) carries slate- / tile-hung dormers with
painted sashes and tall chimney stacks; a gable stack stands on a chimney breast that runs up the
gable wall. Rainwater runs from the eaves gutter through a swan-neck into a hopper and downpipe.
Back and side walls are plainer: flat-lintel sashes (some blocked up), relieving arches in brickwork,
a boarded back door on a stone stoop, tie plates, a second downpipe, an iron cat ladder on the roof.
"""
from math import sin, cos, tan, atan, asin, radians, degrees, hypot, ceil, copysign, exp, pi

import bmesh

import numpy as np
from mathutils import Matrix, Vector

from tk import *

META = {"name": "Stone gabled house", "class": "building", "mount": "ground", "focus": [0.0, -1.2, 3.2]}
PARAMS = {
    "width": {"default": 7.5, "min": 5.0, "max": 12.0, "desc": "front width, ground-floor walls span x in [-width/2, +width/2] (m)"},
    "depth": {"default": 8.0, "min": 6.0, "max": 11.0, "desc": "body depth from the front wall plane y = 0 to y = +depth (m)"},
    "storeys": {"default": 3, "min": 2, "max": 3, "desc": "number of full storeys below the eaves"},
    "storey_height": {"default": 3.1, "min": 2.8, "max": 3.3, "desc": "ground-floor storey height; upper floors are a little lower (m)"},
    "material": {"default": "ashlar", "choices": ["ashlar", "brick_red", "brick_ochre", "plaster"], "desc": "walling of the upper floors"},
    "tint": {"default": "#d8c59c", "desc": "limewash colour (material = plaster)"},
    "stone": {"default": "#c4b79f", "desc": "tint of the dressed stone: surrounds, quoins, bands, ground floor"},
    "ground": {"default": "rusticated", "choices": ["rusticated", "plain"], "desc": "ground floor: channelled stone blocks or the same walling as above"},
    "roof": {"default": "gable", "choices": ["gable", "gable_front", "hip"], "desc": "side gables with coped parapets, gable to the square, or hipped"},
    "covering": {"default": "slate", "choices": ["slate", "tile"], "desc": "roof covering"},
    "pitch": {"default": 48.0, "min": 40.0, "max": 55.0, "desc": "roof pitch (deg)"},
    "dormers": {"default": 2, "min": 0, "max": 3, "desc": "dormers on the front slope (side slopes for a front gable)"},
    "dormer_style": {"default": "pediment", "choices": ["pediment", "hip"], "desc": "dormer top: small pediment or hipped"},
    "cornice": {"default": "modillion", "choices": ["modillion", "dentil"], "desc": "block course under the eaves cornice"},
    "ladder": {"default": True, "desc": "iron cat ladder hooked over the ridge beside a stack (left out where dormers or stacks leave no room)"},
    "chimneys": {"default": 2, "min": 1, "max": 3, "desc": "number of chimney stacks"},
    "door": {"default": "centre", "choices": ["left", "centre", "right"], "desc": "bay of the front door"},
    "steps": {"default": 3, "min": 2, "max": 4, "desc": "steps up to the front door (sets the plinth / ground-floor level)"},
    "balcony": {"default": True, "desc": "wrought-iron balcony with a French window on the first floor above the door"},
    "lantern": {"default": True, "desc": "wall lantern beside the front door"},
    "lantern_cd": {"default": 10.0, "min": 2.0, "max": 40.0, "desc": "wall lantern intensity (cd)"},
    "lit": {"default": 0.55, "min": 0.0, "max": 1.0, "desc": "fraction of windows with a lit room behind"},
    "joinery": {"default": "#e3dccb", "desc": "paint of sashes and window frames"},
    "door_paint": {"default": "#27433a", "desc": "paint of the front and back doors"},
    "ironwork": {"default": "#1c2622", "desc": "paint of balcony, rails, lantern and rainwater goods"},
    "number": {"default": "7", "desc": "house number on the door head ('' = none)"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: rain streaks, damp, moss, roof sag"},
}
VARIANTS = {
    "brick_hip_wide": {"width": 10.5, "depth": 7.0, "storeys": 2, "material": "brick_red", "ground": "plain", "roof": "hip",
                       "covering": "tile", "pitch": 44.0, "door": "left", "dormers": 2, "dormer_style": "hip", "chimneys": 2, "stone": "#cfc4ae",
                       "door_paint": "#5a1a16", "lit": 0.7, "number": "12", "seed": 11},
    "ochre_front_gable": {"width": 6.2, "depth": 9.5, "storeys": 3, "material": "brick_ochre", "roof": "gable_front",
                          "covering": "slate", "pitch": 52.0, "door": "right", "dormers": 1, "chimneys": 2, "balcony": False,
                          "door_paint": "#1f2f4a", "joinery": "#d9d2bd", "lit": 0.45, "number": "3", "seed": 23},
    "plaster_narrow": {"width": 5.2, "depth": 6.5, "storeys": 3, "material": "plaster", "tint": "#cf9f72", "ground": "rusticated",
                       "roof": "gable", "covering": "tile", "pitch": 50.0, "door": "left", "dormers": 1, "dormer_style": "hip", "cornice": "dentil", "chimneys": 1,
                       "stone": "#b9ad98", "door_paint": "#3a2a1e", "age": 0.8, "lit": 0.6, "number": "21", "seed": 37},
    "ashlar_grand": {"width": 12.0, "depth": 10.0, "storeys": 3, "storey_height": 3.3, "material": "ashlar", "roof": "hip",
                     "covering": "slate", "pitch": 42.0, "door": "centre", "dormers": 3, "chimneys": 2, "steps": 4,
                     "door_paint": "#1b2622", "lit": 0.8, "age": 0.3, "number": "1", "seed": 51},
}

GLOW, CURT = 4.5, 0.3                       # every lit pane: glow and how far the curtains are drawn


# ------------------------------------------------------------------ small helpers
def _shade(hexc, f):
    h = hexc.lstrip("#")
    return "#" + "".join("%02x" % max(0, min(255, int(int(h[i:i + 2], 16) * f))) for i in (0, 2, 4))


def _rect(x0, z0, x1, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def _fill(vals, cell):
    vals = sorted(set(round(v, 4) for v in vals))
    out = [vals[0]]
    for v in vals[1:]:
        gap = v - out[-1]
        if gap < 1e-4:
            continue
        k = int(ceil(gap / cell - 1e-6))
        a = out[-1]
        for i in range(1, k):
            out.append(round(a + gap * i / k, 4))
        out.append(v)
    return out


def _use(asset_id, **kw):
    """use() that survives a sub-asset changing its optional params: unknown keys / choices are dropped."""
    try:
        mod = load_asset(asset_id)
        ok = {}
        for k, v in kw.items():
            sp = mod.PARAMS.get(k)
            if sp is None:
                if k == "seed":
                    ok[k] = v
                continue
            if "choices" in sp and v not in sp["choices"]:
                continue
            ok[k] = v
        return use(asset_id, **ok)
    except Exception as e:  # noqa
        print("DBG use() failed for", asset_id, repr(e)[:200])
        return None


def _remap(comp, table):
    """Share this house's own dressing materials with a composed sub-asset (by library name)."""
    if comp is None or comp.obj is None:
        return comp
    me = comp.obj.data
    for i, m in enumerate(me.materials):
        if m is not None and table.get(m.name.split("~")[0]) is not None:
            me.materials[i] = table[m.name.split("~")[0]]
    return comp


def _wall(name, a, b, z0, z1, openings=(), cell=0.45, gable=None):
    """One wall face as a welded grid with rectangular openings (real holes).
    a, b: plan end points, anticlockwise round the building (outward normal = dir x Z).
    openings: (u_centre, z_bottom, w, h). gable=(h0, h1): above z1 the head rises h0 at the ends and h1 at mid span."""
    ax, ay = a
    bx, by = b
    L = hypot(bx - ax, by - ay)
    dx, dy = (bx - ax) / L, (by - ay) / L
    R = lambda v: round(v, 4)
    umid = L / 2
    zmax = z1 + (gable[1] if gable else 0.0)

    def top(u):
        if not gable:
            return R(z1)
        return R(z1 + gable[0] + (gable[1] - gable[0]) * (1.0 - abs(u - umid) / umid))

    us, zs = {0.0, R(L)}, {R(z0), R(z1), R(zmax)}
    if gable:
        us.add(R(umid))
    for (uc, zb, w, h) in openings:
        us.update((R(uc - w / 2), R(uc + w / 2)))
        zs.update((R(min(zmax, max(z0, zb))), R(max(z0, min(zmax, zb + h)))))
    us, zs = _fill(us, cell), _fill(zs, cell)
    vid, verts, faces = {}, [], []

    def V(u, z):
        key = (R(u), R(z))
        i = vid.get(key)
        if i is None:
            verts.append((ax + dx * u, ay + dy * u, z))
            i = vid[key] = len(verts) - 1
        return i

    for i in range(len(us) - 1):
        ua, ub = us[i], us[i + 1]
        um = (ua + ub) / 2
        ta, tb = top(ua), top(ub)
        lo = min(ta, tb)
        for j in range(len(zs) - 1):
            if zs[j + 1] <= lo + 1e-5:
                zm = (zs[j] + zs[j + 1]) / 2
                if any(abs(um - uc) < w / 2 and zb < zm < zb + h for (uc, zb, w, h) in openings):
                    continue
                faces.append((V(ua, zs[j]), V(ub, zs[j]), V(ub, zs[j + 1]), V(ua, zs[j + 1])))
                continue
            zl = zs[j]                          # cap under the raking head: an n-gon that keeps the edge welded
            if zl < max(ta, tb) - 1e-5:
                right = [z for z in zs if zl + 1e-5 < z < tb - 1e-5]
                left = [z for z in zs if zl + 1e-5 < z < ta - 1e-5]
                loop = [V(ua, zl), V(ub, zl)] + [V(ub, z) for z in right] + [V(ub, tb), V(ua, ta)] + [V(ua, z) for z in reversed(left)]
                poly = []
                for v in loop:
                    if v not in poly:
                        poly.append(v)
                if len(poly) >= 3:
                    faces.append(tuple(poly))
            break
    return mesh_object(name, verts, faces)


def _rust(name, a, b, z0, z1, openings=(), course=0.39, block=0.98, depth=0.03, gap=0.022, rng=None, out=0.0, cell=1.0):
    """Channelled, block-jointed masonry face: one welded strip per course with V joints sunk into it.
    Same conventions as _wall (openings are real holes)."""
    ax, ay = a
    bx, by = b
    L = hypot(bx - ax, by - ay)
    dx, dy = (bx - ax) / L, (by - ay) / L
    nx, ny = dy, -dx
    ax += nx * out - dx * out
    ay += ny * out - dy * out
    L += 2 * out
    R = lambda v: round(v, 4)
    ops = [(uc + out, zb, w, h) for (uc, zb, w, h) in openings]
    g = gap
    nc = max(1, int(round((z1 - z0) / course)))
    ch = (z1 - z0) / nc
    nb = max(1, int(round(L / block)))
    bl = L / nb
    verts, faces = [], []
    for k in range(nc):
        zb_, zt_ = z0 + k * ch, z0 + (k + 1) * ch
        mine = [o for o in ops if o[1] < zt_ - 1e-3 and o[1] + o[3] > zb_ + 1e-3]
        zs, groove = {R(zb_), R(zt_)}, set()
        if k > 0:
            zs.add(R(zb_ + g))
            groove.add(R(zb_))
        if k < nc - 1:
            zs.add(R(zt_ - g))
            groove.add(R(zt_))
        for (uc, zo, w, h) in mine:
            for z in (zo, zo + h):
                if zb_ + g + 0.006 < z < zt_ - g - 0.006:
                    zs.add(R(z))
        us, joints = {0.0, R(L)}, set()
        for (uc, zo, w, h) in mine:
            us.update((R(uc - w / 2), R(uc + w / 2)))
        for j in range(nb + 1):
            uj = R((j + (0.5 if k % 2 == 0 else 0.0)) * bl + (rng.jitter(0.0, bl * 0.05) if rng else 0.0))
            if uj < 0.34 or uj > L - 0.34:
                continue
            if any(uc - w / 2 - 0.07 < uj < uc + w / 2 + 0.07 for (uc, zo, w, h) in mine):
                continue
            us.update((R(uj - g), uj, R(uj + g)))
            joints.add(uj)
        us, zs = _fill(us, cell), sorted(zs)
        vid = {}

        def V(u, z):
            i = vid.get((u, z))
            if i is None:
                d = depth if (z in groove or u in joints) else 0.0
                verts.append((ax + dx * u - nx * d, ay + dy * u - ny * d, z))
                i = vid[(u, z)] = len(verts) - 1
            return i

        for i in range(len(us) - 1):
            um = (us[i] + us[i + 1]) / 2
            for j in range(len(zs) - 1):
                zm = (zs[j] + zs[j + 1]) / 2
                if any(abs(um - uc) < w / 2 and zo < zm < zo + h for (uc, zo, w, h) in mine):
                    continue
                faces.append((V(us[i], zs[j]), V(us[i + 1], zs[j]), V(us[i + 1], zs[j + 1]), V(us[i], zs[j + 1])))
    return mesh_object(name, verts, faces)


def _ring(name, x0, x1, y0, y1, z, prof, seg=0.55, sides=(0, 1, 2, 3), ret=0.0):
    """Moulding running round the plan rectangle with mitred corners. prof: [(out, dz), ...] bottom to top.
    Sides not listed in `sides` only get a capped return of length `ret` at each corner (0 = nothing)."""
    verts, faces = [], []
    m = len(prof)
    nrm = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]

    def corner(c, o):
        return [(x0 - o, y0 - o), (x1 + o, y0 - o), (x1 + o, y1 + o), (x0 - o, y1 + o)][c % 4]

    for s in range(4):
        L = (x1 - x0) if s % 2 == 0 else (y1 - y0)
        (px, py), (qx, qy) = corner(s, 0.0), corner(s + 1, 0.0)
        dx, dy = (qx - px) / L, (qy - py) / L
        if s in sides:
            spans = [(0.0, L)]
        elif ret > 0.0:
            spans = [(0.0, min(ret, L / 2)), (L - min(ret, L / 2), L)]
        else:
            spans = []
        for (ua, ub) in spans:
            k = max(1, int(ceil((ub - ua) / seg)))
            base = len(verts)
            for i in range(k + 1):
                u = ua + (ub - ua) * i / k
                for (o, dz) in prof:
                    if u <= 1e-6:
                        cx, cy = corner(s, o)
                    elif u >= L - 1e-6:
                        cx, cy = corner(s + 1, o)
                    else:
                        cx, cy = px + dx * u + nrm[s][0] * o, py + dy * u + nrm[s][1] * o
                    verts.append((cx, cy, z + dz))
            for i in range(k):
                for j in range(m - 1):
                    v = base + i * m + j
                    faces.append((v, v + m, v + m + 1, v + 1))
            if s not in sides:
                if ua > 1e-6:
                    faces.append(tuple(base + j for j in range(m)))
                else:
                    faces.append(tuple(base + k * m + j for j in reversed(range(m))))
    return mesh_object(name, verts, faces)


def _mould_x(name, sec, length, z=0.0, x=0.0, y=0.0, bevel=0.0):
    """Straight moulding along X. sec: closed CCW section [(out, up), ...], out = distance proud of the wall (-Y)."""
    o = extrude_profile(name, sec, length, bevel=bevel)
    transform_verts(o, lambda co: Vector((x + length / 2 - co.z, y - co.x, co.y + z)))
    return o


def _panes(xa, za, xb, zb, cols, rows, stile=0.045, bar=0.02):
    cw = (xb - xa - 2 * stile - (cols - 1) * bar) / cols
    rh = (zb - za - 2 * stile - (rows - 1) * bar) / rows
    out = []
    for c in range(cols):
        for r in range(rows):
            px, pz = xa + stile + c * (cw + bar), za + stile + r * (rh + bar)
            out.append(_rect(px, pz, px + cw, pz + rh))
    return out


def _surround(tag, w, h, head, a, proud, rev, lite=False):
    """Stone dressings of one opening (wall face y = 0, opening bottom centre at the origin, facing -Y)."""
    x0, x1 = -w / 2, w / 2
    top = h + a
    ps = []
    u = [(x0 - a, 0), (x0, 0), (x0, h), (x1, h), (x1, 0), (x1 + a, 0), (x1 + a, top), (x0 - a, top)]
    ps.append(extrude_profile(tag + "_arch", u, proud + rev, loc=(0, rev, 0), rot=(90, 0, 0), bevel=0.0 if lite else 0.005))
    if not lite:
        f = a * 0.36
        u2 = [(x0 - a, 0), (x0 - a + f, 0), (x0 - a + f, top - f), (x1 + a - f, top - f), (x1 + a - f, 0), (x1 + a, 0), (x1 + a, top), (x0 - a, top)]
        ps.append(extrude_profile(tag + "_fillet", u2, proud + 0.02, loc=(0, 0, 0), rot=(90, 0, 0), bevel=0.004))
    L = w + 2 * a
    if head == "keystone":
        k = [(-0.07, 0), (0.07, 0), (0.105, a + 0.075), (-0.105, a + 0.075)]
        ps.append(extrude_profile(tag + "_key", k, proud + 0.05, loc=(0, 0, h - 0.012), rot=(90, 0, 0), bevel=0.006))
    elif head == "lintel":
        ps.append(box(tag + "_lintel", (L + 0.16, proud + 0.012, a * 0.75), loc=(0, -(proud + 0.012) / 2, top + a * 0.375 - 0.004),
                      bevel=0.0 if lite else 0.006, segments=1))
    elif head in ("cornice", "pediment"):
        ps.append(box(tag + "_frieze", (L - 0.02, proud + 0.012, 0.115), loc=(0, -(proud + 0.012) / 2, top + 0.055), bevel=0.004, segments=1))
        sec = [(0, 0), (proud + 0.02, 0), (proud + 0.03, 0.022), (proud + 0.075, 0.04), (proud + 0.075, 0.062),
               (proud + 0.1, 0.072), (proud + 0.1, 0.098), (0, 0.125)]
        zc = top + 0.11
        ps.append(_mould_x(tag + "_hood", sec, L + 0.14, z=zc))
        for sx in (-1, 1):
            ps.append(box(tag + "_console", (0.085, proud + 0.075, 0.2), loc=(sx * (L / 2 - 0.043), -(proud + 0.075) / 2, zc - 0.1), bevel=0.012, segments=2))
        if head == "pediment":
            hw, ph = L / 2 + 0.07, (L / 2 + 0.07) * 0.36
            zb = zc + 0.11
            ps.append(extrude_profile(tag + "_tymp", [(-hw + 0.06, 0), (hw - 0.06, 0), (0, ph - 0.03)], proud + 0.03, loc=(0, 0, zb), rot=(90, 0, 0)))
            ang = degrees(atan(ph / hw))
            hyp = hypot(hw, ph)
            for sx in (-1, 1):
                ps.append(box(tag + "_rake", (hyp + 0.03, proud + 0.105, 0.06), loc=(sx * hw / 2, -(proud + 0.105) / 2, zb + ph / 2 + 0.012),
                              rot=(0, sx * ang, 0), bevel=0.006, segments=1))
    return ps


def _cull_back(obj, lim=0.98):
    """Drop the faces that look straight into the building (+Y): nobody ever sees the back of a sash or frame."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    kill = [f for f in bm.faces if f.normal.y > lim]
    if kill:
        bmesh.ops.delete(bm, geom=kill, context='FACES')
        bm.to_mesh(obj.data)
    bm.free()


def _sill(tag, w, a, proud, brackets, lite=False):
    L = w + 2 * a + 0.1
    sec = [(-0.1, -0.09), (proud + 0.045, -0.09), (proud + 0.045, -0.074), (proud + 0.07, -0.074), (proud + 0.07, -0.022), (proud - 0.01, 0.0), (-0.1, 0.0)]
    ps = [_mould_x(tag + "_sill", sec, L, bevel=0.0 if lite else 0.003)]
    if brackets:
        for sx in (-1, 1):
            k = [(0, 0), (0.05, 0), (0.105, 0.1), (0.105, 0.13), (0, 0.13)]
            ps.append(_mould_x(tag + "_sillbr", k, 0.085, z=-0.09 - 0.13, x=sx * (w / 2 + a * 0.5), bevel=0.005))
    return ps


def _window(tag, st, jm, w, h, head="keystone", cols=3, rows=(2, 2), a=0.15, proud=0.055, rev=0.14, french=False, brackets=False, lite=False,
            leaf_m=None):
    """Sash (or French) window in a stone surround. Returns (frame proto, glass proto), both facing -Y."""
    stone = _surround(tag, w, h, head, a, proud, rev, lite)
    if not french:
        stone += _sill(tag, w, a, proud, brackets, lite)
    else:
        stone.append(box(tag + "_thresh", (w + 2 * a, rev + proud + 0.03, 0.05), loc=(0, (rev - proud - 0.03) / 2, -0.025), bevel=0.006, segments=1))
    for o in stone:
        assign(o, st)
    x0, x1 = -w / 2, w / 2
    fy = rev - 0.02
    wood = [shape_extrude(tag + "_frame", _rect(x0, 0, x1, h), [_rect(x0 + 0.05, 0.05, x1 - 0.05, h - 0.05)], depth=0.075, loc=(0, fy, 0), rot=(90, 0, 0))]
    X0, X1, Z0, Z1 = x0 + 0.05, x1 - 0.05, 0.05, h - 0.05
    gy = rev - 0.022
    if french:
        pz = Z0 + 0.52
        for (xa, xb) in ((X0, -0.004), (0.004, X1)):
            holes = _panes(xa, pz - 0.045, xb, Z1, 2, 4, stile=0.055, bar=0.02)
            wood.append(shape_extrude(tag + "_leaf", _rect(xa, Z0, xb, Z1), holes, depth=0.04, loc=(0, rev - 0.03, 0), rot=(90, 0, 0)))
            # raised bottom panel
            wood.append(box(tag + "_panel", (xb - xa - 0.2, 0.012, 0.3), loc=((xa + xb) / 2, rev - 0.076, Z0 + 0.26), bevel=0.004, segments=1))
        gv = [(X0, gy, pz - 0.02), (X1, gy, pz - 0.02), (X1, gy, Z1), (X0, gy, Z1)]
        gf = [(0, 1, 2, 3)]
        # espagnolette handle
        wood.append(cylinder(tag + "_knob", 0.014, 0.05, loc=(0.03, rev - 0.085, Z0 + 1.05), rot=(90, 0, 0), verts=8))
    else:
        zm = Z0 + (Z1 - Z0) * rows[1] / (rows[0] + rows[1])
        wood.append(shape_extrude(tag + "_sash_u", _rect(X0, zm - 0.02, X1, Z1), _panes(X0, zm - 0.02, X1, Z1, cols, rows[0]), depth=0.035,
                                  loc=(0, rev - 0.04, 0), rot=(90, 0, 0)))
        wood.append(shape_extrude(tag + "_sash_l", _rect(X0, Z0, X1, zm + 0.02), _panes(X0, Z0, X1, zm + 0.02, cols, rows[1]), depth=0.035,
                                  loc=(0, rev - 0.005, 0), rot=(90, 0, 0)))
        gv = [(X0, gy, Z0), (X1, gy, Z0), (X1, gy, zm), (X0, gy, zm), (X0, gy - 0.03, zm), (X1, gy - 0.03, zm), (X1, gy - 0.03, Z1), (X0, gy - 0.03, Z1)]
        gf = [(0, 1, 2, 3), (4, 5, 6, 7)]
    for o in wood:
        assign(o, leaf_m if (leaf_m is not None and o.name.split(".")[0].endswith(("_leaf", "_panel"))) else jm)
    frame = join(stone + wood, name=tag + "_proto")
    _cull_back(frame)
    glass = mesh_object(tag + "_glass", gv, gf)
    uv_planar(glass, u_axis=(1, 0, 0), v_axis=(0, 0, 1))
    return frame, glass


def _blind(tag, st, wm, w, h, a, proud, rev):
    """Blocked-up window: the stone surround and sill stay, the opening is walled up a hand deep in the reveal."""
    ps = _surround(tag, w, h, "lintel", a, proud, rev, True) + _sill(tag, w, a, proud, False)
    for o in ps:
        assign(o, st)
    pn = box(tag + "_fill", (w + 0.02, 0.04, h + 0.02), loc=(0, 0.115, h / 2))
    assign(pn, wm)
    uv_box(pn)
    return join(ps + [pn], name=tag + "_proto")


def _relief(tag, st, c, zt, rise=0.2, nv=9, depth=0.2):
    """Segmental relieving arch of stone voussoirs over a flat lintel (chord c, springing at height zt)."""
    R = (c * c / 4 + rise * rise) / (2 * rise)
    th = asin(min(1.0, c / 2 / R))
    aw = 2 * th * R / nv
    ps = []
    for i in range(nv):
        phi = -th + (i + 0.5) * 2 * th / nv
        key = 0.035 if i == nv // 2 else 0.0
        rr = R + (depth + key) / 2
        ps.append(box(tag + "_v", (aw * (1 + depth / 2 / R) - 0.012, 0.05 + key * 0.4, depth + key),
                      loc=(rr * sin(phi), -0.007 - key * 0.2, zt + rise - R + rr * cos(phi)), rot=(0, degrees(phi), 0)))
    o = join(ps, name=tag)
    assign(o, st)
    return o


def _strip_litter(me, libs=("slate", "terracotta")):
    """Remove the loose single slates / tiles a roof module scatters over its slopes (they read as litter from the
    square); the eaves course - the lowest ring of such pieces - stays."""
    idx = {i for i, m in enumerate(me.materials) if m is not None and m.name.split("~")[0] in libs}
    if not idx:
        return 0
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    seen, isl = set(), []
    for f in bm.faces:
        if f.index in seen or f.material_index not in idx:
            continue
        stack, faces = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            faces.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        vs = {v for g in faces for v in g.verts}
        if len(vs) > 12:
            continue
        cs = [v.co for v in vs]
        if max(max(c[i] for c in cs) - min(c[i] for c in cs) for i in range(3)) > 0.45:
            continue
        isl.append((sum(c.z for c in cs) / len(cs), faces))
    k = 0
    if isl:
        z0 = min(z for z, _ in isl)
        kill = [g for z, fs in isl if z > z0 + 0.2 for g in fs]
        k = len(kill)
        if kill:
            bmesh.ops.delete(bm, geom=kill, context='FACES')
            bm.to_mesh(me)
    bm.free()
    return k


def _refine(me, libs, pts, max_edge=0.3):
    """Split the long edges of the faces carrying the named materials near the plan points (x, y, radius), so that
    a painted mask has vertices to live on there (a roof slope is otherwise only a handful of big faces)."""
    idx = {i for i, m in enumerate(me.materials) if m is not None and m.name.split("~")[0] in libs}
    if not idx or not pts:
        return
    for _ in range(7):
        bm = bmesh.new()
        bm.from_mesh(me)
        es = []
        for e in bm.edges:
            a, b = e.verts[0].co, e.verts[1].co
            if (a - b).length <= max_edge or not any(f.material_index in idx for f in e.link_faces):
                continue
            dx, dy = b.x - a.x, b.y - a.y
            l2 = dx * dx + dy * dy
            for (px, py, r) in pts:
                t = 0.0 if l2 < 1e-9 else min(1.0, max(0.0, ((px - a.x) * dx + (py - a.y) * dy) / l2))
                if hypot(a.x + dx * t - px, a.y + dy * t - py) < r:
                    es.append(e)
                    break
        if es:
            bmesh.ops.subdivide_edges(bm, edges=es, cuts=1)
            bm.to_mesh(me)
        bm.free()
        if not es:
            break


def _rock(name, L, H, rng, c=0.025, face=0.02, bulge=0.022, back=0.07, nx=6, nz=3):
    """Rock-faced plinth stone: a drafted, chamfered margin round a face left rough from the pick and standing
    proud. The chamfer starts on y = 0, the stone faces -Y and runs `back` into the wall."""
    xs = [-L / 2 + c + (L - 2 * c) * i / nx for i in range(nx + 1)]
    zs = [-H / 2 + c + (H - 2 * c) * j / nz for j in range(nz + 1)]
    verts, faces = [], []
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            edge = i in (0, nx) or j in (0, nz)
            if edge:
                verts.append((x, -face, z))
            else:
                verts.append((x + rng.jitter(0.0, 0.2 * L / nx), -face - bulge * rng.vary(1.0, 0.55), z + rng.jitter(0.0, 0.15 * H / nz)))

    def g(i, j):
        return j * (nx + 1) + i

    for j in range(nz):
        for i in range(nx):
            faces.append((g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)))
    o0 = len(verts)
    for (x, z) in ((-L / 2, -H / 2), (L / 2, -H / 2), (L / 2, H / 2), (-L / 2, H / 2)):
        verts.append((x, 0.0, z))
    for (x, z) in ((-L / 2, -H / 2), (L / 2, -H / 2), (L / 2, H / 2), (-L / 2, H / 2)):
        verts.append((x, back, z))
    faces.append(tuple([o0, o0 + 1] + [g(i, 0) for i in range(nx, -1, -1)]))
    faces.append(tuple([o0 + 1, o0 + 2] + [g(nx, j) for j in range(nz, -1, -1)]))
    faces.append(tuple([o0 + 2, o0 + 3] + [g(i, nz) for i in range(nx + 1)]))
    faces.append(tuple([o0 + 3, o0] + [g(0, j) for j in range(nz + 1)]))
    for k in range(4):
        a, b = o0 + k, o0 + (k + 1) % 4
        faces.append((b, a, a + 4, b + 4))
    return mesh_object(name, verts, faces)


def _prism(name, top, t):
    """Slab under a planar top polygon (points anticlockwise seen from above), thickness t straight down."""
    k = len(top)
    verts = [tuple(p) for p in top] + [(p[0], p[1], p[2] - t) for p in top]
    faces = [tuple(range(k)), tuple(reversed(range(k, 2 * k)))]
    for i in range(k):
        j = (i + 1) % k
        faces.append((i, k + i, k + j, j))
    return mesh_object(name, verts, faces)


def _dormer(tag, w, H, ta, style, cover, jm, lead, glass_m):
    """Roof dormer with slate- / tile-hung cheeks, a painted timber front with a two-light sash and a small
    pediment or hipped top. Origin: middle of the foot of its front on the roof surface; the front faces -Y and
    the main roof rises towards +Y with slope ta."""
    ov, fo = 0.09, 0.13
    hw2 = w / 2 + ov
    tr = tan(radians(30.0 if style == "pediment" else 33.0))
    yb = H / ta
    skin, wood, dark = [], [], []
    skin.append(box(tag + "_cheeks", (w - 0.05, yb + 0.25, H + 0.35), loc=(0, 0.07 + (yb + 0.25) / 2, (H + 0.05 - 0.3) / 2)))
    # painted front: corner posts, apron, sill, box frame with two sashes, frieze board and cornice
    for sx in (-1, 1):
        wood.append(box(tag + "_post", (0.12, 0.09, H + 0.02), loc=(sx * (w / 2 - 0.06), 0.025, H / 2 - 0.01), bevel=0.006, segments=1))
    wood.append(box(tag + "_apron", (w - 0.2, 0.05, 0.13), loc=(0, 0.04, 0.055)))
    wood.append(box(tag + "_sill", (w - 0.16, 0.13, 0.045), loc=(0, -0.025, 0.125), bevel=0.006, segments=1))
    x0, x1, z0, z1 = -(w / 2 - 0.12), w / 2 - 0.12, 0.145, H - 0.14
    wood.append(shape_extrude(tag + "_frame", _rect(x0, z0, x1, z1), [_rect(x0 + 0.045, z0 + 0.045, x1 - 0.045, z1 - 0.045)], depth=0.07,
                              loc=(0, 0.07, 0), rot=(90, 0, 0)))
    X0, X1, Z0, Z1 = x0 + 0.045, x1 - 0.045, z0 + 0.045, z1 - 0.045
    zm = (Z0 + Z1) / 2
    wood.append(shape_extrude(tag + "_sash_u", _rect(X0, zm - 0.02, X1, Z1), _panes(X0, zm - 0.02, X1, Z1, 2, 1, stile=0.04), depth=0.03,
                              loc=(0, 0.05, 0), rot=(90, 0, 0)))
    wood.append(shape_extrude(tag + "_sash_l", _rect(X0, Z0, X1, zm + 0.02), _panes(X0, Z0, X1, zm + 0.02, 2, 1, stile=0.04), depth=0.03,
                              loc=(0, 0.078, 0), rot=(90, 0, 0)))
    wood.append(box(tag + "_frieze", (w + 0.02, 0.105, 0.13), loc=(0, 0.02, H - 0.065), bevel=0.004, segments=1))
    wood.append(_mould_x(tag + "_cornice", [(0, 0), (0.03, 0), (0.05, 0.03), (0.095, 0.045), (0.095, 0.08), (0, 0.08)], 2 * hw2, z=H, y=-0.03))
    for sx in (-1, 1):                          # the cornice returns along the cheeks as a plain eaves board
        wood.append(box(tag + "_eaves", (0.03, yb + 0.1, 0.085), loc=(sx * (hw2 - 0.05), (yb + 0.1) / 2 - 0.03, H + 0.04)))
    glass = mesh_object(tag + "_glass", [(X0, 0.06, Z0), (X1, 0.06, Z0), (X1, 0.06, Z1), (X0, 0.06, Z1)], [(0, 1, 2, 3)])
    uv_planar(glass, u_axis=(1, 0, 0), v_axis=(0, 0, 1))
    assign(glass, glass_m)
    # lead apron dressed down over the slates below the sill
    ap = box(tag + "_lead", (w + 0.16, 0.3, 0.012), loc=(0, -0.1, -0.1 * ta + 0.016), rot=(degrees(atan(ta)), 0, 0))
    dark.append(ap)
    if style == "pediment":
        ze = H + 0.1
        zr = ze + hw2 * tr
        for sx in (-1, 1):
            a_, b_ = (sx * hw2, -fo, ze), (0.0, -fo, zr)
            c_, d_ = (0.0, zr / ta + 0.12, zr), (sx * hw2, ze / ta + 0.12, ze)
            skin.append(_prism(tag + "_slope", [a_, d_, c_, b_] if sx > 0 else [a_, b_, c_, d_], 0.05))
        wood.append(extrude_profile(tag + "_tymp", [(-hw2 + 0.05, 0), (hw2 - 0.05, 0), (0, (hw2 - 0.05) * tr)], 0.08, loc=(0, 0.05, H + 0.08), rot=(90, 0, 0)))
        ang = degrees(atan(tr))
        hyp = hypot(hw2, hw2 * tr)
        for sx in (-1, 1):
            wood.append(box(tag + "_rake", (hyp, 0.095, 0.06), loc=(sx * hw2 / 2, -0.078, H + 0.05 + hw2 * tr / 2 - 0.012), rot=(0, sx * ang, 0),
                            bevel=0.005, segments=1))
        dark.append(cylinder(tag + "_roll", 0.032, zr / ta + fo + 0.06, loc=(0, (zr / ta - fo + 0.1) / 2, zr + 0.012), rot=(90, 0, 0), verts=6))
    else:
        ze = H + 0.08
        zr = ze + hw2 * tr
        FL, FR = (-hw2, -fo, ze), (hw2, -fo, ze)
        BL, BR = (-hw2, ze / ta + 0.12, ze), (hw2, ze / ta + 0.12, ze)
        RF, RB = (0.0, -fo + hw2, zr), (0.0, max(zr / ta + 0.12, -fo + hw2 + 0.05), zr)
        vv = [FL, FR, BR, BL, RF, RB]
        skin.append(mesh_object(tag + "_hip", vv, [(0, 1, 4), (1, 2, 5, 4), (0, 4, 5, 3), (0, 3, 2, 1)]))
        dark.append(cylinder(tag + "_roll", 0.03, RB[1] - RF[1] + 0.04, loc=(0, (RB[1] + RF[1]) / 2, zr + 0.008), rot=(90, 0, 0), verts=6))
        hl = hypot(hw2, hypot(hw2, hw2 * tr))
        for sx in (-1, 1):                      # hip rolls
            dark.append(sweep(tag + "_hiproll", [(sx * hw2, -fo, ze + 0.01), (0.0, -fo + hw2, zr + 0.012)], 0.026, sides=6, smooth_path=False))
    for o in skin:
        assign(o, cover)
        uv_box(o)
    for o in wood:
        assign(o, jm)
    for o in dark:
        assign(o, lead)
    return join(skin + wood + dark + [glass], name=tag)


# ------------------------------------------------------------------ the house
def build(p, rng):
    W, D = float(p.width), float(p.depth)
    n = int(round(p.storeys))
    age = float(p.age)
    seed = int(p.seed)
    slate = p.covering == "slate"
    rusticated = p.ground == "rusticated"

    # ---- materials (one shared set of ten for everything, composed parts included)
    st = mat("stone_smooth", p.stone)
    plm = mat("stone_rough", _shade(p.stone, 0.62))        # plinth: the same stone, rough-dressed, darker and damp
    if p.material == "ashlar":
        wall_m = mat("ashlar_limestone")
    elif p.material == "plaster":
        wall_m = mat("plaster_lime", p.tint)
    else:
        wall_m = mat(p.material)
    wall_hex = p.tint if p.material == "plaster" else {"ashlar": "#cdbfa6", "brick_red": "#9a5a48", "brick_ochre": "#b89a68"}[p.material]
    jm = mat("wood_painted", p.joinery)
    door_m = mat("wood_painted", p.door_paint)
    iron = mat("iron_painted", p.ironwork)
    g_dark = mat("window_dark")
    g_lit = mat("window_lit", "#ffb45a", glow=GLOW, curtain=CURT)   # every lit pane of the house: one warm amber
    cover = mat("roof_slate" if slate else "roof_tile_clay")
    render_m = wall_m if p.material == "plaster" else st
    base = {"stone_smooth": st, "stone_rough": plm, "mortar": st, "rubble_fieldstone": st, "slate": cover, "moss": plm, "terracotta": st,
            "plaster_lime": render_m, "plaster_rough": render_m,
            "iron_wrought": iron, "iron_painted": iron, "iron_rusty": iron, "iron_cast": iron, "brass": iron,
            "window_lit": g_lit, "window_dark": g_dark, "glass_clear": g_dark, "emissive": g_lit,
            "wood_oak": jm, "wood_planks": jm, "wood_painted": jm}

    def norm(comp, **over):
        t = dict(base)
        t.update(over)
        return _remap(comp, t)

    # ---- levels
    rise = 0.17
    nsteps = int(round(p.steps))
    zf0 = nsteps * rise
    sh = float(p.storey_height)
    hs = [sh, sh - 0.12, sh - 0.3] if n == 3 else [sh, sh]
    zf = [zf0]
    for h in hs:
        zf.append(zf[-1] + h)
    z_e = zf[-1]
    ENT = 0.36                                 # height of the eaves entablature under z_e

    # ---- plan, walls and roof geometry helpers
    C = [(-W / 2, 0.0), (W / 2, 0.0), (W / 2, D), (-W / 2, D)]
    walls = [(C[i], C[(i + 1) % 4]) for i in range(4)]
    wlen = [W, D, W, D]

    def wpt(s, u, d=0.0, z=0.0):
        (ax, ay), (bx, by) = walls[s]
        dx, dy = (bx - ax) / wlen[s], (by - ay) / wlen[s]
        return (ax + dx * u - dy * d, ay + dy * u + dx * d, z)

    oh = 0.5                                   # eaves overhang of the roof module
    tanp = tan(radians(p.pitch))
    along_x = p.roof == "gable" or (p.roof == "hip" and W >= D)       # ridge direction
    span, run = (D, W) if along_x else (W, D)
    r_yaw = 0.0 if along_x else 90.0
    kr = max(1.0, span / 10.0)                 # the roof module spans at most 10 m: stretch it across beyond that
    pitch_b = min(61.0, degrees(atan(kr * tanp)))
    tb = tan(radians(pitch_b))
    eave_s = (0, 2) if along_x else (1, 3)     # walls under the main slopes
    ks = [kr if s in eave_s else 1.0 for s in range(4)]
    gables = () if p.roof == "hip" else tuple(s for s in range(4) if s not in eave_s)
    half = span / 2

    def roof_z(s, d):                          # roof surface above wall s, d metres in from its face
        return z_e + (d / ks[s] + oh) * tb

    def roof_pitch(s):
        return degrees(atan(tb / ks[s]))

    z_ridge = roof_z(eave_s[0], half)
    mg = 0.17                                  # the wall head stops this far under the roof surface
    gab = (oh * tb - mg, z_ridge - z_e - mg)

    # the roof module is always built ridge-along-x and turned into place later; it is made first because a
    # parapet gable brings its own masonry face, which the attic windows have to sit on
    if p.roof == "hip":
        rkw = dict(type="hip", finial="spike")
    else:
        rkw = dict(type="gable", gable="parapet", gable_wall="none", finial="ball")
    roof = _use("roof_gable", width=run, depth=span / kr, pitch=pitch_b, overhang=oh, ridge="x",
                covering="roof_slate" if slate else "roof_tile_clay", sag=0.015 + 0.04 * age, kick=0.45, attic_window="none",
                rooflight="none", ladder="none", eaves="cornice", ridge_tile="lead_roll" if slate else "half_round", gutter="iron",
                wall_color=wall_hex, trim_color=p.ironwork, moss=0.1 + 0.3 * age, seed=seed + 5, **rkw)
    gable_off = 0.0                             # how far that face stands in front of the wall plane
    if roof is not None:
        _strip_litter(roof.obj.data)
    if roof is not None and gables:
        me_ = roof.obj.data
        idx_ = {i for i, m_ in enumerate(me_.materials) if m_ is not None and m_.name.split("~")[0].startswith("plaster")}
        xs_ = [abs(me_.vertices[v].co.x) for f in me_.polygons if f.material_index in idx_ for v in f.vertices]
        if xs_:
            gable_off = min(0.12, max(0.0, max(xs_) - run / 2))

    # ---- bays; an end-bay door is drawn in from the corner so its dressings never crowd the quoins
    margin = 0.45
    nb = max(2, min(5, int(round((W - 1.0) / 2.2))))
    sp = (W - 2 * margin) / nb
    bay_c = [margin + (i + 0.5) * sp for i in range(nb)]              # regular bays (back wall)
    bay_u = list(bay_c)                                               # front bays
    door_bay = {"left": 0, "centre": nb // 2, "right": nb - 1}[p.door]
    balcony = bool(p.balcony) and n >= 2
    door_x = bay_u[door_bay] - W / 2
    dmax = W / 2 - (1.5 if balcony else 1.85)
    if abs(door_x) > dmax:
        door_x = copysign(dmax, door_x)
        bay_u[door_bay] = door_x + W / 2
    ww = min(1.0, sp * 0.48)

    # window styles: w, h, sill height above the floor, proto kwargs; heights are then fitted under the string
    # course / entablature above so no head ever runs into a moulding
    S = {
        "ground": dict(w=ww, h=1.75, sill=0.92, kw=dict(head="keystone", cols=3, rows=(2, 2))),
        "first": dict(w=ww, h=1.9, sill=0.62, kw=dict(head="cornice", cols=3, rows=(2, 2), brackets=True)),
        "top": dict(w=ww * 0.94, h=1.38, sill=0.78, kw=dict(head="keystone", cols=3, rows=(2, 2), a=0.13)),
        "french": dict(w=min(1.2, sp * 0.56), h=2.42, sill=0.05, kw=dict(head="pediment", french=True, leaf_m=door_m)),
        "plain_g": dict(w=0.9, h=1.5, sill=0.92, kw=dict(head="lintel", cols=2, rows=(2, 2), a=0.11, proud=0.035, lite=True)),
        "plain": dict(w=0.9, h=1.5, sill=0.92, kw=dict(head="lintel", cols=2, rows=(2, 2), a=0.11, proud=0.035, lite=True)),
        "plain_top": dict(w=0.9, h=1.22, sill=0.86, kw=dict(head="lintel", cols=2, rows=(2, 2), a=0.11, proud=0.035, lite=True)),
    }
    heavy = (W + D) * n * (1.0 + 0.04 * (int(p.dormers) + int(p.chimneys))) > 76.0    # very large houses: trim the upper detail
    if heavy:
        S["top"]["kw"]["lite"] = True
        S["first"]["kw"]["lite"] = True
    if n == 2:
        S["first"]["sill"] = 0.74
        S["first"]["kw"] = dict(head="keystone", cols=3, rows=(2, 2), brackets=True, lite=heavy)

    def head_extra(sd):
        a, hd = sd["kw"].get("a", 0.15), sd["kw"].get("head", "keystone")
        if hd == "keystone":
            return a + 0.07
        if hd == "lintel":
            return 1.75 * a
        if hd == "cornice":
            return a + 0.245
        return a + 0.275 + (sd["w"] / 2 + a + 0.07) * 0.36

    def fit(style, lim, extra=0.0):
        sd = S[style]
        sd["h"] = round(min(sd["h"], lim - sd["sill"] - head_extra(sd) - extra), 3)

    lim_top = hs[n - 1] - ENT - 0.07
    lim_1 = hs[1] - 0.18 if n == 3 else lim_top
    arch_g = 0.0 if rusticated else 0.42        # room for a relieving arch over the plain ground-floor windows
    fit("ground", hs[0] - 0.18)
    fit("first", lim_1)
    fit("top", lim_top)
    if lim_1 - S["french"]["sill"] - head_extra(S["french"]) < 1.95:
        S["french"]["kw"]["head"] = "cornice"
    fit("french", lim_1)
    fit("plain_g", hs[0] - 0.18, arch_g)
    fit("plain", lim_1)
    fit("plain_top", lim_top)
    S["attic"] = dict(w=ww * 0.94, h=1.38, sill=0.5, kw=dict(head="keystone", cols=3, rows=(2, 2), a=0.13, rev=0.045))
    S["attic_plain"] = dict(w=0.9, h=1.22, sill=0.5, kw=dict(head="lintel", cols=2, rows=(2, 2), a=0.11, proud=0.035, rev=0.045, lite=True))
    rl = rng.fork("lit")
    wins = []                                  # dict(s, u, z, style, lit)

    def add_win(s, u, f, style, force=None, sill=None, blind=False):
        if blind:                               # blocked-up opening of the same size
            bs = "blind_" + style
            if bs not in S:
                S[bs] = dict(S[style], blind=True)
            style, lit = bs, False
        else:
            lit = rl.chance(p.lit) if force is None else force
        wins.append(dict(s=s, u=u, z=zf[min(f, n)] + (S[style]["sill"] if sill is None else sill), f=f, style=style, lit=lit))
        return wins[-1]

    def back_style(f):
        return "plain_g" if f == 0 else ("plain_top" if (f == n - 1 and n == 3) else "plain")

    for f in range(n):
        for i in range(nb):
            if f == 0 and i == door_bay:
                continue
            if f == 1 and i == door_bay and balcony:
                add_win(0, bay_u[i], f, "french", force=True if p.lit >= 0.3 else None)
            else:
                add_win(0, bay_u[i], f, "ground" if f == 0 else ("top" if (f == n - 1 and n == 3) else "first"))
    rb = rng.fork("back")
    back_door_bay = (nb - 1 - door_bay) if nb > 2 else int(rb.u() * nb) % nb
    back_door_u = bay_c[back_door_bay]
    for f in range(n):
        for i in range(nb):
            if f == 0 and i == back_door_bay:
                continue
            add_win(2, bay_c[i], f, back_style(f), blind=(f > 0 and nb > 2 and rb.chance(0.22)))
    nch = int(round(p.chimneys))
    stacked = {"gable": (1, 3)[:nch], "gable_front": (2,), "hip": ()}[p.roof]     # gables that carry a flush ridge stack
    for s in (1, 3):
        if s in stacked:                        # a chimney breast runs up the middle of this gable: windows flank it
            us_side = [D * 0.3, D * 0.7] if D >= 7.4 else [D * 0.25, D * 0.75]
        else:
            us_side = [D * 0.3, D * 0.7] if D >= 7.4 else [D * 0.5]
        for f in range(n):
            for u in us_side:
                add_win(s, u, f, back_style(f), blind=rb.chance(0.3 if s == 3 else 0.18))
    # attic: windows in the gables where the raking head leaves room (a pair flanking the flue under a gable stack)
    oculi = []

    def gable_half(zrel):                       # half width of the gable wall face at this height above the eaves
        return half * (1.0 - max(0.0, zrel - gab[0]) / (gab[1] - gab[0]))

    for s in gables:
        style = "attic" if s == 0 else "attic_plain"
        sd, sill = S[style], 0.5
        need = sill + sd["h"] + 0.42
        if s in stacked:
            off = max(1.25, min(1.45, half * 0.3))
            if gable_half(need) >= off + sd["w"] / 2 + 0.3:
                for sg in (-1, 1):
                    add_win(s, wlen[s] / 2 + sg * off, n, style, sill=sill)
        else:
            zo = max(need + 0.75, gab[1] * 0.66) if gable_half(need) >= sd["w"] / 2 + 0.3 else gab[1] * 0.42
            if gable_half(need) >= sd["w"] / 2 + 0.3:
                add_win(s, wlen[s] / 2, n, style, force=True if (s == 0 and p.lit > 0.02) else None, sill=sill)
            if gable_half(zo + 0.5) >= 0.62:
                oculi.append((s, zo))
    glazed = [w_ for w_ in wins if not S[w_["style"]].get("blind")]
    if p.lit > 0.02:                            # make sure every side shows life
        for s in range(4):
            mine = [w_ for w_ in glazed if w_["s"] == s]
            if mine and not any(w_["lit"] for w_ in mine):
                mine[int(rl.u() * len(mine)) % len(mine)]["lit"] = True
        fg = [w_ for w_ in glazed if w_["s"] == 0 and w_["f"] == 0]
        if fg and not any(w_["lit"] for w_ in fg):
            fg[0]["lit"] = True

    # ---- chimney breasts: where a ridge stack stands flush on a gable its flue is told on the wall below as a
    #      projecting breast, wide at the hearths of the ground floor and shouldered in above
    bdw, bdh = 0.95, 2.05
    BR_P = 0.18
    breasts = {}                                # wall s -> (half width low, half width up)
    for s in stacked:
        L = wlen[s]
        obst = [(w_["u"], S[w_["style"]]["w"] / 2 + 0.24) for w_ in wins if w_["s"] == s]
        if s == 2:
            obst.append((back_door_u, bdw / 2 + 0.3))
        clear = min([abs(u - L / 2) - hw_ for (u, hw_) in obst] or [9.0])
        if clear >= 0.725:
            breasts[s] = (0.725, 0.525)
        elif clear >= 0.525:
            breasts[s] = (0.525, 0.525)
    q_br = 0.525 + 0.3
    z_bt = z_e + gab[0] + (gab[1] - gab[0]) * (1.0 - q_br / half) - 0.3 if gables else z_e

    # ---- masonry
    wall_objs, rust_objs = [], []
    rr = rng.fork("rust")
    for s in range(4):
        a, b = walls[s]
        ops = [(w_["u"], w_["z"], S[w_["style"]]["w"], S[w_["style"]]["h"]) for w_ in wins if w_["s"] == s and w_["f"] < n]
        if s == 2:
            ops.append((back_door_u, zf0, bdw, bdh))
        ops_g = [o for o in ops if o[1] < zf[1] - 0.2]
        ops_u = [o for o in ops if o[1] >= zf[1] - 0.2]
        if rusticated:
            o = _rust(f"wall_g{s}", a, b, zf0, zf[1], ops_g, course=0.39, block=0.98, depth=0.03, gap=0.022, rng=rr, cell=0.45)
            assign(o, st)
            rust_objs.append(o)
        else:
            o = _wall(f"wall_g{s}", a, b, zf0, zf[1], ops_g, cell=0.45)
            assign(o, wall_m)
            wall_objs.append(o)
        o = _wall(f"wall_u{s}", a, b, zf[1], z_e, ops_u, cell=0.45, gable=gab if s in gables else None)
        assign(o, wall_m)
        wall_objs.append(o)
    if p.material == "plaster":
        for o in wall_objs:
            displace_noise(o, strength=0.006, scale=0.9, seed=seed)

    # plinth: rough-dressed blocks of the darker stone, one or two courses under a chamfered top course
    rp = rng.fork("plinth")
    PL = 0.09                                   # how far the plinth stands proud of the wall
    top_h = 0.2 if zf0 >= 0.45 else 0.16
    low_h = zf0 - top_h
    ncl = 1 if low_h < 0.42 else 2
    hc = low_h / ncl
    core = box("plinth_core", (W + 0.1, D + 0.1, zf0 - 0.012), loc=(0, D / 2, (zf0 - 0.012) / 2))
    blks = [_rock(f"plinth_blk{i}", 1.0, hc - 0.014, rp) for i in range(4)]
    cap = _mould_x("plinth_capb", [(-0.1, 0.0), (PL, 0.0), (PL, top_h * 0.42), (0.028, top_h - 0.006), (-0.1, top_h - 0.006)], 1.0)
    bm_, cm_ = [[] for _ in blks], []
    for s in range(4):
        u0, u1 = (-PL, wlen[s] + PL) if s % 2 == 0 else (0.045, wlen[s] - 0.045)
        for k in range(ncl + 1):
            nbk = max(1, int(round((u1 - u0) / (0.86 if k < ncl else 1.3))))
            cuts = [u0] + [u0 + (u1 - u0) * (i + (0.5 if k % 2 else 0.0) * (1 if nbk > 1 else 0) * 0.7 + rp.jitter(0.0, 0.2)) / nbk for i in range(1, nbk)] + [u1]
            cuts = sorted(min(u1, max(u0, c)) for c in cuts)
            for i in range(len(cuts) - 1):
                ln = cuts[i + 1] - cuts[i]
                if ln < 0.08:
                    continue
                uc = (cuts[i] + cuts[i + 1]) / 2
                if k < ncl:
                    x, y, _ = wpt(s, uc, -0.05 - rp.jitter(0.006, 0.006))
                    bm_[int(rp.u() * 4) % 4].append(trs((x, y, (k + 0.5) * hc), (0, 0, 90.0 * s), (ln - 0.018, 1.0, 1.0)))
                else:
                    x, y, _ = wpt(s, uc, -rp.jitter(0.0, 0.003))
                    cm_.append(trs((x, y, low_h), (0, 0, 90.0 * s), (ln - 0.008, 1.0, 1.0)))
    rocks = [instances_to_mesh(f"plinth_blocks{i}", b_, m_, keep_base=False) if m_ else None for i, (b_, m_) in enumerate(zip(blks, bm_))]
    delete([b_ for b_, m_ in zip(blks, bm_) if not m_])
    plinth = [core, join([r_ for r_ in rocks if r_ is not None], name="plinth_blocks"), instances_to_mesh("plinth_cap", cap, cm_, keep_base=False)]
    for o in plinth:
        assign(o, plm)
    for o in wall_objs + rust_objs + plinth:
        uv_box(o)

    dress = []
    band = [(0.0, 0.0), (0.028, 0.0), (0.046, 0.02), (0.046, 0.1), (0.072, 0.115), (0.072, 0.155), (0.0, 0.185)]
    for f in range(1, n):
        dress.append(_ring(f"string{f}", -W / 2, W / 2, 0.0, D, zf[f] - 0.12, band))
    # eaves entablature: frieze band right round; on the eaves sides a block course and a deep corona under the
    # gutter, returned a short way round each corner of a gable
    z_en = z_e - ENT
    prof_b = [(0.0, 0.0), (0.03, 0.012), (0.03, 0.15), (0.05, 0.165), (0.05, 0.225), (0.0, 0.24)]
    prof_c = [(0.0, 0.0), (0.03, 0.012), (0.03, 0.15), (0.05, 0.165), (0.05, 0.285), (0.075, 0.3), (0.3, 0.3), (0.3, 0.335), (0.32, 0.345), (0.32, 0.36), (0.0, 0.36)]
    eav = (0, 1, 2, 3) if p.roof == "hip" else eave_s
    if gables:
        dress.append(_ring("frieze", -W / 2, W / 2, 0.0, D, z_en, prof_b, sides=gables))
    dress.append(_ring("cornice", -W / 2, W / 2, 0.0, D, z_en, prof_c, sides=eav, ret=0.42))
    mm_ = []
    if p.cornice == "dentil":
        mod = box("dentil", (0.085, 0.075, 0.09))
        m_sp, m_d, m_z = 0.17, -(0.05 + 0.0375), 0.165 + 0.06
    else:
        mod = box("modillion", (0.115, 0.2, 0.1))
        m_sp, m_d, m_z = 0.42, -(0.05 + 0.1), 0.3 - 0.05
    for s in eav:
        k = max(2, int(round((wlen[s] - 0.3) / m_sp)))
        for i in range(k + 1):
            x, y, _ = wpt(s, 0.15 + (wlen[s] - 0.3) * i / k, m_d)
            mm_.append(trs((x, y, z_en + m_z), (0, 0, 90.0 * s)))
    dress.append(instances_to_mesh("cornice_blocks", mod, mm_, keep_base=False, uv_jitter=False))

    # quoins: long and short blocks alternating up every corner
    nq = max(4, int(round((z_en - zf0) / 0.39)))
    hq = (z_en - zf0) / nq
    ql, qs, qo = 0.56, 0.33, 0.04
    qb = box("quoin", (ql, qs, hq - 0.016), bevel=0.013, segments=1)
    qb2 = box("quoin_b", (ql, qs, hq - 0.016))    # the back corners do without the eased arrises
    rq = rng.fork("quoin")
    qm, qm2 = [], []
    for ci, (cx, cy) in enumerate(C):
        sx, sy = (-1 if cx < 0 else 1), (-1 if cy < D / 2 else 1)
        for k in range(nq):
            long_x = (k + ci) % 2 == 0
            lx, ly = (ql, qs) if long_x else (qs, ql)
            lx *= rq.vary(1.0, 0.05)
            ly *= rq.vary(1.0, 0.05)
            j = rq.jitter(0.0, 0.004)
            (qm if ci < 2 else qm2).append(trs((cx + sx * (qo + j - lx / 2), cy + sy * (qo + j - ly / 2), zf0 + (k + 0.5) * hq), (0, 0, 0 if long_x else 90),
                          (lx / ql, ly / ql, 1.0) if long_x else (ly / ql, lx / qs, 1.0)))
    dress.append(instances_to_mesh("quoins", qb, qm, keep_base=False))
    dress.append(instances_to_mesh("quoins_back", qb2, qm2, keep_base=False))
    for o in dress:
        assign(o, st)
        uv_box(o)

    # chimney breasts (built in wall-local axes: x along the wall, -y out of it, then turned into place)
    breast_objs = []
    for s, (hwl, hwu) in breasts.items():
        L = wlen[s]
        z1 = zf[1] + 0.2
        ps_st, ps_w, ps_pl = [], [], []
        ps_pl.append(box("breast_plinth", (2 * hwl + 2 * PL, BR_P + PL + 0.1, zf0 - 0.004), loc=(0, (0.1 - BR_P - PL) / 2, (zf0 - 0.004) / 2),
                         bevel=0.017, segments=1))
        lo = box("breast_lo", (2 * hwl, BR_P + 0.1 - 0.006, z1 - zf0), loc=(0, (0.1 - BR_P + 0.006) / 2, (z1 + zf0) / 2))
        (ps_st if rusticated else ps_w).append(lo)
        zu0 = z1
        if hwl > hwu + 1e-6:
            ps_st.append(extrude_profile("breast_sh", [(-hwl, 0), (hwl, 0), (hwu, 0.4), (-hwu, 0.4)], BR_P + 0.1, loc=(0, 0.1, z1), rot=(90, 0, 0), bevel=0.006))
            zu0 = z1 + 0.4
        ps_w.append(box("breast_up", (2 * hwu, BR_P + 0.1, z_bt - zu0), loc=(0, (0.1 - BR_P) / 2, (z_bt + zu0) / 2)))
        ps_st.append(_mould_x("breast_cap", [(-0.1, 0.0), (BR_P + 0.025, 0.0), (BR_P + 0.025, 0.05), (-0.1, 0.4)], 2 * hwu + 0.05, z=z_bt))
        for f in range(1, n):
            hw_ = hwl if zf[f] - 0.12 < z1 else hwu
            ps_st.append(_ring("breast_band", -hw_, hw_, -BR_P, 0.1, zf[f] - 0.12, band))
        ps_st.append(_ring("breast_frieze", -hwu, hwu, -BR_P, 0.1, z_en, prof_b))
        x, y, _ = wpt(s, L / 2)
        for grp, m_, tg in ((ps_st, st, "dress"), (ps_w, wall_m, "wall"), (ps_pl, plm, "plinth")):
            if grp:
                o = join(grp, name=f"breast{s}_{tg}")
                assign(o, m_)
                place(o, loc=(x, y, 0.0), rot=(0, 0, 90.0 * s))
                densify(o, 0.3)                 # vertices for the weathering masks to live on
                uv_box(o)
                breast_objs.append(o)
        if rusticated:
            a = wpt(s, L / 2 - hwl, -BR_P)
            b = wpt(s, L / 2 + hwl, -BR_P)
            o = _rust(f"breast_rust{s}", a[:2], b[:2], zf0, zf[1], (), course=0.39, block=0.74, depth=0.03, gap=0.022, rng=rr, cell=0.5)
            assign(o, st)
            uv_box(o)
            breast_objs.append(o)

    # ---- windows: one prototype per style, stamped into every opening
    lights = []                                 # (priority, name, loc, kwargs)
    for style, sd in S.items():
        mine = [w_ for w_ in wins if w_["style"] == style]
        if not mine:
            continue
        fm = []
        mats = {0: [], 1: []}
        for w_ in mine:
            x, y, z = wpt(w_["s"], w_["u"], -(gable_off + 0.04) if w_["f"] == n else 0.0, w_["z"])
            M = trs((x, y, z), (0, 0, 90.0 * w_["s"]))
            fm.append(M)
            mats[1 if w_["lit"] else 0].append(M)
        kw = sd["kw"]
        if sd.get("blind"):
            frame = _blind("w_" + style, st, wall_m, sd["w"], sd["h"], kw.get("a", 0.15), kw.get("proud", 0.055), kw.get("rev", 0.14))
            instances_to_mesh("win_" + style, frame, fm, keep_base=False)
            continue
        frame, glass = _window("w_" + style, st, jm, sd["w"], sd["h"], **kw)
        instances_to_mesh("win_" + style, frame, fm, keep_base=False)
        for gi, ms in mats.items():
            if ms:
                assign(glass, g_dark if gi == 0 else g_lit)
                instances_to_mesh(f"glass_{style}_{gi}", glass, ms, keep_base=True)
        delete([glass])
        if style == "plain_g" and not rusticated:   # brick / plain ground floor: relieving arches over the lintels
            a_ = kw.get("a", 0.11)
            arch = _relief("relief", st, sd["w"] + 2 * a_ + 0.16, sd["h"] + 1.75 * a_)
            uv_box(arch)
            dress.append(instances_to_mesh("relief_arches", arch, fm, keep_base=False))

    def spill(w_, cd, name, pri):               # the lamp of the room: inside, behind the glass - no bare source outside
        sd = S[w_["style"]]
        up = w_["f"] > 0                        # above the head, so the source is never seen through a pane
        x, y, z = wpt(w_["s"], w_["u"], 0.75 if up else 0.42, w_["z"] + sd["h"] + (0.16 if up else 0.2))
        lights.append((pri, name, (x, y, z), dict(color="#ffb45a", intensity=cd, range=5.5, flicker=0.03)))

    lit_fg = [w_ for w_ in wins if w_["s"] == 0 and w_["f"] == 0 and w_["lit"]]
    if lit_fg:
        spill(lit_fg[0], 9.0, "window_front", 1)
        if len(lit_fg) > 1:
            spill(lit_fg[-1], 7.0, "window_front_b", 6)
    lit_b = [w_ for w_ in wins if w_["s"] == 2 and w_["lit"]]
    if lit_b:
        spill(sorted(lit_b, key=lambda w_: w_["f"])[0], 8.0, "window_back", 2)
    for s, pri in ((1, 3), (3, 4)):
        lit_s = [w_ for w_ in wins if w_["s"] == s and w_["lit"]]
        if lit_s:
            spill(sorted(lit_s, key=lambda w_: w_["f"])[0], 7.0, f"window_side{s}", pri)
    lit_fu = [w_ for w_ in wins if w_["s"] == 0 and w_["f"] == 1 and w_["lit"]]
    if lit_fu:
        spill(sorted(lit_fu, key=lambda w_: 0 if w_["style"] == "french" else 1)[0], 5.0, "window_upper", 5)

    # ---- composed parts
    def put(comp, loc, yaw=0.0, scale=(1.0, 1.0, 1.0), light_as=None, effects=False):
        if comp is None or comp.obj is None:
            return None
        o = comp.place(loc, rot=(0, 0, yaw), scale=scale, sockets=False)
        M = Matrix.Translation(Vector(loc)) @ Matrix.Rotation(radians(yaw), 4, 'Z') @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))
        for (sname, kind, sm, props) in comp.sockets:
            wloc = tuple((M @ sm).translation)
            if kind == "light" and light_as is not None:
                pr = {k: v for k, v in props.items() if k in ("color", "intensity", "range", "flicker", "shadow", "shadow_near")}
                pr.update(light_as[2])
                lights.append((light_as[0], light_as[1], wloc, pr))
            elif kind == "fx" and effects:
                pr = dict(props)
                fx(f"{comp.asset_id}_{sname}_{len(lights)}_{int(wloc[0] * 10)}", pr.pop("fx", "smoke"), wloc, **pr)
        return o

    def sock(comp, name):
        if comp is None:
            return None
        for (sname, kind, sm, props) in comp.sockets:
            if sname.startswith(name):
                return sm.translation.copy()
        return None

    # ---- where dormers and stacks stand (needed before the roof is dressed)
    nd = int(round(p.dormers))
    dd = 0.72
    avail = (half - dd) / kr * tb               # height from the foot of a dormer to the ridge
    dH = min(1.25, avail - 0.8)
    dw = 1.2 if dH >= 1.05 else 1.05
    dspots = []                                 # (wall s, u)
    nc = int(round(p.chimneys))
    if nd > 0 and dH >= 0.9:
        if 0 not in eave_s and p.roof != "hip":             # gable to the square: dormers on the side slopes
            cand_d = [(1, D * 0.36), (3, D * 0.64)] + ([(1, D * 0.68)] if nc < 3 else [])
            dspots = cand_d[:nd]
        else:
            if p.roof != "hip":
                lim = W
            elif along_x:
                lim = W / 2 - (dd + 1.6) / kr - dw / 2 - 0.15
            else:
                lim = W / 2 - (dd + 1.6) * kr - dw / 2 - 0.15
            allowed = [u for u in bay_u if abs(u - W / 2) <= lim + 1e-6]
            if not allowed and lim >= 0.0:
                allowed = [W / 2]
            if nd >= len(allowed):
                dspots = [(0, u) for u in allowed]
            else:
                pick = sorted(set(int(round((i + 0.5) * len(allowed) / nd - 0.5)) for i in range(nd)))
                dspots = [(0, allowed[i]) for i in pick]

    # chimneys: gable stacks ride the ridge flush with the gable wall, the others stand in a slope
    cw, cd_ = 0.95, 0.58
    e = run / 2 - cd_ / 2 - 0.02
    ds = max(1.0, half - 1.0)
    cspots = []                                 # (x, y, yaw, on_ridge, wall s, u, d)
    if p.roof == "gable":
        cand = [(None, e, D / 2, 90.0), (None, -e, D / 2, 90.0), (2, W * 0.4, ds)]
    elif p.roof == "gable_front":
        cand = [(None, 0.0, D / 2 + e, 0.0), (3, D * 0.28, ds), (1, D * 0.72, ds)]
    else:
        ends = (1, 3) if along_x else (2, 0)
        de = max(1.0, half / kr - 1.15)         # on a hip end, just below the end of the ridge
        if along_x:
            cand = [(ends[0], wlen[ends[0]] / 2, de), (ends[1], wlen[ends[1]] / 2, de), (eave_s[1], wlen[eave_s[1]] * 0.4, ds)]
        else:                                   # keep the stacks off the hip that faces the square
            cand = [(ends[0], wlen[ends[0]] / 2, de), (3, D * 0.4, ds), (1, D * 0.62, ds)]
    for c_ in cand[:nc]:
        if c_[0] is None:
            cspots.append((c_[1], c_[2], c_[3], True, 0, 0.0, 0.0))
        else:
            s, u, d = c_
            x, y, _ = wpt(s, u, d)
            cspots.append((x, y, 90.0 * s, False, s, u, d))

    # ---- roof: turn the module into place, then grow lichen / moss where water lingers - along the eaves,
    #      against the gable abutments, below the stacks and beside the dormer cheeks
    r_org = (0.0, D / 2, z_e)
    Mroof = Matrix.Translation(Vector(r_org)) @ Matrix.Rotation(radians(r_yaw), 4, 'Z') @ Matrix.Diagonal((1.0, kr, 1.0, 1.0))
    rm = rng.fork("moss")
    patches = []                                # (x, y, rx, ry, strength) in plan

    def patch(s, u, d, ru, rd, k=1.0):
        x, y, _ = wpt(s, u, d)
        patches.append((x, y, ru, rd, k) if s % 2 == 0 else (x, y, rd, ru, k))

    for s in (eave_s if gables else (0, 1, 2, 3)):
        L = wlen[s]
        for i in range(2):
            patch(s, L * (0.18 + 0.64 * rm.u()), -oh + 0.45, rm.vary(1.1, 0.35), rm.vary(0.55, 0.25), 0.9)
        if gables:
            for u in (0.25, L - 0.25):
                if rm.chance(0.75):
                    patch(s, u, half * ks[s] * rm.vary(0.4, 0.3), 0.42, rm.vary(1.5, 0.3), 0.8)
    for (x, y, yaw, on_ridge, s, u, d) in cspots:
        if not on_ridge:
            patch(s, u, d - 1.0, 0.6, 0.95, 1.0)
    for (s, u) in dspots:
        patch(s, u + (dw / 2 + 0.22) * (1 if rm.chance(0.5) else -1), dd + 0.55, 0.26, 0.8, 0.9)

    def roof_moss(co, nrm):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        nz = 0.5 + 0.5 * np.sin(x * 6.1 + 2.0 * np.sin(y * 4.3 + seed)) * np.sin(y * 5.3 + 2.0 * np.sin(x * 3.7 + seed * 0.7))
        g = np.zeros(len(x))
        for (px, py, rx, ry, k) in patches:
            r = np.hypot((x - px) / rx, (y - py) / ry)
            g = np.maximum(g, np.clip(1.25 - r, 0.0, 1.0) * k)
        return np.clip(g * (0.45 + 0.75 * nz) * (0.55 + 0.6 * age), 0.0, 1.0) * (nrm[:, 2] > 0.2) * (z > z_e - 0.05)

    def roof_grime(co, nrm):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        g = np.zeros(len(x))
        for (cx, cy, yaw, on_ridge, s, u, d) in cspots:     # soot washed down the slates from every stack
            r = np.hypot((x - cx) / 0.6, (y - cy) / 0.6)
            g = np.maximum(g, np.clip(1.6 - r, 0.0, 1.0) * 0.7)
        return np.clip(g * (0.5 + age), 0.0, 1.0) * (z > z_e - 0.05)

    roof_o = None
    if roof is not None:
        for i_, m_ in enumerate(roof.obj.data.materials):
            if m_ is not None and m_.name.split("~")[0] in ("roof_slate", "roof_tile_clay"):
                cover = m_
        base["lead"] = base["zinc"] = base["slate"] = cover
        norm(roof, plaster_lime=wall_m, plaster_rough=wall_m, terracotta=cover, moss=cover, slate=cover)
        Mi = Mroof.inverted()
        near = []
        for (px, py, rx, ry, k) in patches:
            q = Mi @ Vector((px, py, z_e))
            near.append((q.x, q.y, 1.15 * max(rx, ry)))
        for (cx, cy, yaw, on_ridge, s, u, d) in cspots:
            q = Mi @ Vector((cx, cy, z_e))
            near.append((q.x, q.y, 0.95))
        _refine(roof.obj.data, ("roof_slate", "roof_tile_clay"), near, 0.6 if heavy else 0.45)
        roof_o = put(roof, r_org, yaw=r_yaw, scale=(1.0, kr, 1.0))
    else:                                       # emergency cover so the house is never open to the sky
        base["lead"] = base["zinc"] = base["slate"] = cover
        sec = [(-span / 2 - oh, 0.0), (span / 2 + oh, 0.0), (0.0, (span / 2 + oh) * tanp)]
        o = extrude_profile("roof_fallback", sec, run + 0.5)
        Mf = Matrix.Translation(Vector(r_org)) @ Matrix.Rotation(radians(r_yaw), 4, 'Z')
        transform_verts(o, lambda co: Mf @ Vector((run / 2 + 0.25 - co.z, co.x, co.y)))
        assign(o, cover)

    # ---- front door, steps, lantern, balcony
    going = 0.29
    landing = max(0.55, min(0.9, 1.5 - (nsteps - 1) * going))
    door_h = min(2.55, sh - 0.55)
    door = norm(_use("door_plank", width=1.3, height=door_h, head="round", frame="stone", frame_tone=p.stone, finish="painted", tone=p.door_paint,
                     studs=3, step=0.04, grille=True, lit=True, intensity=2.5, number=p.number, hardware="brass", wear=0.3 + 0.4 * age,
                     seed=seed + 1), wood_painted=door_m)
    put(door, (door_x, 0.0, zf0))
    steps = norm(_use("door_steps", steps=nsteps, width=1.7, landing=landing, rise=rise, going=going, rail="iron", rail_side="both",
                      balusters=1, plan="straight", tint=p.stone, wear=0.4 + 0.4 * age, moss=0.0, lantern=False,
                      scraper="left" if door_x > 0.3 else "right", seed=seed + 2), stone_rough=st)
    put(steps, (door_x, 0.0, 0.0))
    # the lantern hangs in the middle of the clear wall between the door surround and the next window (or the
    # quoins), fixed at the springing of the door arch so it stays below the balcony consoles
    fr = 0.65                                   # half width of the door with its stone frame
    oth = [bay_u[i] - W / 2 for i in range(nb) if i != door_bay]
    gap = {}
    for sg in (-1, 1):
        nxt = [sg * (xo - door_x) - ww / 2 - 0.19 for xo in oth if sg * (xo - door_x) > 0]
        gap[sg] = min(nxt + [W / 2 - sg * door_x - 0.6]) - fr
    lside = 1 if door_bay <= (nb - 1) / 2 else -1
    if gap[lside] < 0.44 and gap[-lside] > gap[lside]:
        lside = -lside
    lan_x = door_x + lside * (fr + max(0.21, gap[lside] / 2))
    lan_z = zf0 + min(door_h - 1.1, (zf[1] - zf0) - 0.76 - 0.85 - 0.06 if balcony else door_h - 0.9)
    if p.lantern:
        lan = _use("wall_lantern", reach=0.5, style="scroll", carry="stand", roof="ogee", lantern_width=0.26, paint=p.ironwork,
                   rust=0.2 + 0.4 * age, intensity=float(p.lantern_cd), lit=True, drop=0, gas_pipe=False, seed=seed + 3)
        if lan is not None:
            gl_ = [m_ for m_ in lan.obj.data.materials if m_ is not None and m_.name.split("~")[0] == "glass_lantern"]
            norm(lan, glass_lantern=gl_[0] if gl_ else None)
        put(lan, (lan_x, 0.0, lan_z), light_as=(0, "lantern", {}))
    if balcony:
        bw = max(1.7, min(2.3, sp + 0.25, 2 * (min(gap.values()) + fr + 0.19 + 0.0) - 0.1))
        bal = norm(_use("balcony_iron", width=bw, projection=0.8, rail_height=1.0, pattern="scroll", deck="stone", brackets=2, frieze=False,
                        handrail="iron", finial="ball", gilt=True, paint=p.ironwork, stone=p.stone, pots=0, rust=0.15 + 0.4 * age,
                        lantern="none", seed=seed + 4))
        put(bal, (door_x, 0.0, zf[1] + 0.03))
    else:                                       # no balcony: a stone hood on consoles shelters the door instead
        hz = zf0 + door_h + 0.05
        hood = [box("door_hood_frieze", (1.6, 0.07, 0.12), loc=(door_x, -0.035, hz + 0.06), bevel=0.005, segments=1),
                _mould_x("door_hood", [(0, 0), (0.09, 0), (0.12, 0.03), (0.2, 0.05), (0.2, 0.075), (0.27, 0.095), (0.27, 0.14), (0, 0.17)],
                         1.76, z=hz + 0.12, x=door_x)]
        for sx in (-1, 1):
            hood.append(box("door_hood_console", (0.13, 0.19, 0.4), loc=(door_x + sx * 0.74, -0.095, hz - 0.08), bevel=0.02, segments=2))
        for o in hood:
            assign(o, st)
            uv_box(o)
            paint(o, grime=0.25 + 0.3 * age)
    socket("sign", "attach", (door_x - lside * 1.1, 0.0, zf0 + 2.75), role="sign")
    socket("door", "attach", (door_x, 0.0, zf0), role="door")

    # ---- dormers: slate- / tile-hung cheeks, painted sash front, small pediment or hipped top; the room behind
    #      glows through the same glazing as every other lit window of the house
    rd_ = rng.fork("dormer")
    dormer_objs = []
    for k, (s, u) in enumerate(dspots):
        lit = rd_.chance(max(p.lit, 0.35)) or (k == 0 and p.lit > 0.02)
        x, y, _ = wpt(s, u, dd)
        o = _dormer(f"dormer{k}", dw, dH, tan(radians(roof_pitch(s))), p.dormer_style, cover, jm, iron, g_lit if lit else g_dark)
        place(o, loc=(x, y, roof_z(s, dd) - 0.02), rot=(0, 0, 90.0 * s))
        dormer_objs.append(o)

    for k, (x, y, yaw, on_ridge, s, u, d) in enumerate(cspots):
        if on_ridge:
            sink, zb = 0.95, z_ridge - 0.95
            hgt = sink + 1.95 + 0.14 * k
            cp = float(p.pitch)
        else:
            zc = roof_z(s, d)
            cp = roof_pitch(s)
            sink = 0.5 + cd_ / 2 * tan(radians(cp))
            zb = zc - sink
            hgt = min(4.5, sink + (z_ridge - zc) + 1.45)
        rc = rng.fork(f"ch{k}")
        npot = (1 if k else 2) if heavy else 2 + (k + seed) % 2
        ch = norm(_use("chimney", width=cw, depth=cd_, height=hgt, material={"ashlar": "brick_ochre" if heavy else "stone", "plaster": "brick_ochre" if heavy else "stone",
                                 "brick_red": "brick_red", "brick_ochre": "brick_ochre"}[p.material], pots=npot,
                       pot_style=rc.pick(["mixed", "roll", "cannon", "crown", "octagon"]) if k else "mixed",
                       pot_height=rc.vary(0.62, 0.25), corbels=2 + (k + seed) % 2, dentils=False,
                       shoulder=0.0, lean=rc.jitter(0.0, 0.6 + age), soot=0.45 + 0.4 * age,
                       moss=0.0, strap=(k == 0), flashing=not on_ridge, sink=sink, pitch=min(60.0, max(20.0, cp)), roof_stub="none",
                       smoke=(k == 0), seed=seed + 20 + k), rubble_fieldstone=wall_m, brick_red=wall_m, brick_ochre=wall_m)
        put(ch, (x, y, zb), yaw=yaw, effects=True)

    # ---- cat ladder for the sweep: hooked over the ridge beside a stack, on a slope without dormers
    lad = None                                  # (wall s, u)
    d_slopes = {s for (s, u) in dspots}
    if p.ladder:
        if p.roof == "gable":
            lad = (2, 1.3)
        elif p.roof == "gable_front":
            if 3 not in d_slopes:
                lad = (3, (D * 0.28 + 1.25) if nc >= 2 else 1.3)
        elif along_x:
            rh = run / 2 - half / kr            # half length of the ridge
            if rh >= 0.7:
                lad = (2, W / 2 - (rh - 0.35))
        else:
            rh = run / 2 - half / kr
            if rh >= 0.7 and nc < 3:
                lad = (1, D / 2 + (rh - 0.35))
        if lad is not None:
            for (x, y, yaw, on_ridge, s, u, d) in cspots:
                if not on_ridge and s == lad[0] and abs(u - lad[1]) < cw / 2 + 0.6:
                    lad = None
                    break
    if lad is not None:
        s, u = lad
        cpit = cos(radians(roof_pitch(s)))
        d1 = half * ks[s]
        d0 = max(1.3, d1 - 3.6 * cpit)
        lift = 0.11
        parts = []
        for sg in (-1, 1):
            pts = [wpt(s, u + sg * 0.2, d0 + (d1 - 0.12 - d0) * i / 4, roof_z(s, d0 + (d1 - 0.12 - d0) * i / 4) + lift) for i in range(5)]
            pts += [wpt(s, u + sg * 0.2, d1 - 0.02, z_ridge + 0.2), wpt(s, u + sg * 0.2, d1 + 0.14, z_ridge + 0.16),
                    wpt(s, u + sg * 0.2, d1 + 0.32, z_ridge - 0.07)]
            parts.append(sweep("ladder_stile", pts, 0.021, sides=6, smooth_path=False))
            for i in (0, 2, 4):
                dq = d0 + (d1 - 0.12 - d0) * i / 4
                x, y, z = wpt(s, u + sg * 0.2, dq, roof_z(s, dq) + lift / 2 - 0.02)
                parts.append(box("ladder_foot", (0.04, 0.04, lift + 0.05), loc=(x, y, z)))
        nr = max(3, int((d1 - 0.2 - d0) / (0.29 * cpit)))
        for i in range(nr + 1):
            dq = d0 + 0.06 + (d1 - 0.26 - d0) * i / nr
            x, y, z = wpt(s, u, dq, roof_z(s, dq) + lift)
            parts.append(box("ladder_rung", (0.42, 0.032, 0.032), loc=(x, y, z), rot=(0, 0, 90.0 * s)))
        lo_ = join(parts, name="roof_ladder")
        assign(lo_, iron)
        paint(lo_, special=0.7 + 0.3 * age, wear=0.3)   # rusted through its paint: it has to read against the slates

    # ---- rainwater: swan-neck from each gutter outlet into a hopper head and a cast-iron stack on the wall below
    def wall_of(pt):
        dist = [-pt[1], pt[0] - W / 2, pt[1] - D, -W / 2 - pt[0]]
        s = max(range(4), key=lambda i: dist[i])
        (ax, ay), (bx, by) = walls[s]
        u = (pt[0] - ax) * (bx - ax) / wlen[s] + (pt[1] - ay) * (by - ay) / wlen[s]
        return s, min(max(u, 0.24), wlen[s] - 0.24)

    outlets = []
    for nm in ("downpipe_f", "downpipe_b"):
        t = sock(roof, nm)
        if t is not None:
            outlets.append(Mroof @ t)
    if not outlets:
        outlets = [Vector((W / 2 - 0.6, -oh - 0.1, z_e)), Vector((-W / 2 + 0.6, D + oh + 0.1, z_e))]
    hop_z = z_en - 0.3
    pr_, po = 0.047, 0.125                      # pipe radius, stand-off of its axis from the wall
    pipes = []
    for k, pt in enumerate(outlets[:2]):
        s, u = wall_of(pt)
        yaw = 90.0 * s
        px, py, _ = wpt(s, u, -po)
        pipes.append(cylinder("rw_pipe", pr_, hop_z - 0.42, loc=(px, py, 0.2 + (hop_z - 0.42) / 2), verts=10))
        zc = 0.55
        while zc < hop_z - 0.5:
            pipes.append(cylinder("rw_collar", pr_ + 0.011, 0.085, loc=(px, py, zc), verts=10, bevel=0.005, segments=1))
            ex, ey, _ = wpt(s, u, -0.035)
            pipes.append(box("rw_ears", (0.25, 0.07, 0.04), loc=(ex, ey, zc), rot=(0, 0, yaw), bevel=0.005, segments=1))
            for sg in (-1, 1):
                nx_, ny_, _ = wpt(s, u + sg * 0.1, -0.075)
                pipes.append(cylinder("rw_nail", 0.011, 0.012, loc=(nx_, ny_, zc), rot=(90, 0, yaw), verts=6))
            zc += 1.83
        hb = box("rw_hopper", (0.33, 0.25, 0.27), bevel=0.012, segments=1)
        transform_verts(hb, lambda co: Vector((co.x * (0.52 + 0.48 * min(1.0, (co.z + 0.135) / 0.2)),
                                               (co.y + 0.125) * (0.6 + 0.4 * min(1.0, (co.z + 0.135) / 0.2)) - 0.125, co.z)))
        hx, hy, _ = wpt(s, u, -0.135)
        place(hb, loc=(hx, hy, hop_z - 0.135), rot=(0, 0, yaw))
        pipes.append(hb)
        pipes.append(box("rw_hopper_rim", (0.37, 0.285, 0.05), loc=(hx, hy, hop_z + 0.01), rot=(0, 0, yaw), bevel=0.01, segments=1))
        pipes.append(cylinder("rw_spigot", pr_ + 0.006, 0.16, loc=(px, py, hop_z - 0.33), verts=10))
        sx_, sy_, _ = wpt(s, u, -0.24)
        pipes.append(sweep("rw_shoe", [(px, py, 0.34), (px, py, 0.2), (sx_, sy_, 0.09)], pr_, sides=10, smooth_path=False))
        path = [(pt.x, pt.y, pt.z + 0.02), (pt.x, pt.y, pt.z - 0.14), (hx, hy, hop_z + 0.24), (hx, hy, hop_z - 0.04)]
        pipes.append(sweep(f"rw_neck{k}", resample_path(path, n=12), 0.04, sides=8))
    for o in pipes:
        assign(o, iron)
        smooth(o, 40)
        paint(o, special=mask_mul(noise_mask(0.35, seed=seed + 3, lo=0.45, hi=0.8), 0.3 + 0.5 * age), grime=height_ramp(0.0, 0.8))

    # ---- back door: boarded door in a stone surround; the stoop runs along the wall to stay inside the plot
    rbd = rng.fork("backdoor")
    bd = _surround("bd", bdw, bdh, "lintel", 0.13, 0.04, 0.14)
    if not rusticated:
        rs_ = min(0.2, (zf[1] - 0.18 - zf0) - (bdh + 1.75 * 0.13) - 0.22)
        if rs_ >= 0.08:
            bd.append(_relief("bd_relief", st, bdw + 2 * 0.13 + 0.16, bdh + 1.75 * 0.13, rise=rs_))
    for o in bd:
        assign(o, st)
    nbd = 6
    bwid = (bdw - 0.004) / nbd
    for i in range(nbd):
        b_ = plank("bd_board", length=bdh - 0.01, width=bwid - 0.004, thick=0.034, loc=(-bdw / 2 + (i + 0.5) * bwid + 0.002, 0.105, bdh / 2),
                   axis='Z', bevel=0.004, warp=0.0, rng=rbd)
        assign(b_, door_m)
        bd.append(b_)
    for zz in (0.32, bdh - 0.34):
        hg = box("bd_strap", (bdw * 0.72, 0.007, 0.045), loc=(-bdw * 0.12, 0.105 - 0.021, zz), bevel=0.002, segments=1)
        assign(hg, iron)
        bd.append(hg)
    hd = torus("bd_ring", major=0.055, minor=0.007, loc=(bdw * 0.33, 0.105 - 0.03, 1.02), rot=(90, 0, 0), major_seg=16, minor_seg=6)
    assign(hd, iron)
    bd.append(hd)
    bxw = W / 2 - back_door_u                   # world x of the back door
    sdir = 1.0 if bxw > 0 else -1.0             # local +x is world -x: the flight descends towards the middle of the wall
    if 2 in breasts:
        sdir = -sdir                            # ... unless a chimney breast stands there
    sdep, sgo = 0.66, 0.27
    lw = bdw + 0.46
    stoop = [((0.0, lw), 0)] + [((sdir * (lw / 2 + (i - 0.5) * sgo), sgo), i) for i in range(1, nsteps)]
    for (cx_, wd_), i in stoop:
        top = zf0 - i * rise
        sb = stone_block("bd_step", (wd_ + (0.0 if i == 0 else 0.012), sdep - 0.004 * i, top), loc=(cx_, -sdep / 2 - PL * 0.5, -zf0 + top / 2), rng=rbd, chamfer=0.014, detail=0.45)
        assign(sb, st)
        paint(sb, wear=0.25, special=0.2 + 0.4 * age, grime=0.3)
        bd.append(sb)
    bdo = join(bd, name="back_door")
    bx, by, _ = wpt(2, back_door_u)
    place(bdo, loc=(bx, by, zf0), rot=(0, 0, 180))

    # ---- oculus high in the gables that carry no stack
    for (s, zo) in oculi:
        lit_o = rl.chance(p.lit)
        ring = lathe("oculus_ring", [(0.3, 0.0), (0.3, 0.035), (0.33, 0.035), (0.345, 0.06), (0.41, 0.07), (0.44, 0.05), (0.44, 0.0)], segments=24, rot=(90, 0, 0))
        assign(ring, st)
        gl = cylinder("oculus_glass", 0.31, 0.012, loc=(0, -0.008, 0), rot=(90, 0, 0), verts=24)
        assign(gl, g_lit if lit_o else g_dark)
        bars = [box("oculus_bar", (0.62, 0.022, 0.026), loc=(0, -0.02, 0), bevel=0.003, segments=1),
                box("oculus_bar", (0.026, 0.022, 0.62), loc=(0, -0.02, 0), bevel=0.003, segments=1)]
        for b_ in bars:
            assign(b_, jm)
        oc = join([ring, gl] + bars, name=f"oculus{s}")
        x, y, _ = wpt(s, wlen[s] / 2, -(gable_off + 0.005))
        place(oc, loc=(x, y, z_e + zo), rot=(0, 0, 90.0 * s))

    # ---- lights (at most six, most important first)
    lights.sort(key=lambda t: t[0])
    for (_, name, loc, kw) in lights[:6]:
        light(name, loc, **kw)

    # ---- wrought-iron tie plates where the floor beams are anchored through the side and back walls
    tie = join([box("tie_bar", (0.46, 0.012, 0.05), loc=(0, -0.008, 0), rot=(0, 45, 0), bevel=0.003, segments=1),
                box("tie_bar", (0.46, 0.012, 0.05), loc=(0, -0.008, 0), rot=(0, -45, 0), bevel=0.003, segments=1),
                cylinder("tie_bolt", 0.024, 0.03, loc=(0, -0.02, 0), rot=(90, 0, 0), verts=6)], name="tie_proto")
    assign(tie, iron)
    tm = []
    for f in range(1, n):
        for s in (1, 2, 3):
            if s == 2:
                us_t = [margin + i * sp for i in range(1, nb)]
            else:
                us_t = [D * 0.5] if D >= 7.4 else [D * 0.24, D * 0.76]
            for u in us_t:
                if s in breasts and abs(u - wlen[s] / 2) < breasts[s][0] + 0.35:
                    continue
                x, y, _ = wpt(s, u)
                tm.append(trs((x, y, zf[f] + 0.36), (0, 0, 90.0 * s)))
    if tm:
        ties = instances_to_mesh("ties", tie, tm, keep_base=False)
        paint(ties, special=0.35 + 0.4 * age)
    else:
        delete([tie])

    for i, w_ in enumerate([w_ for w_ in wins if w_["s"] == 0 and w_["f"] == 0]):
        socket(f"window_box_{i}", "attach", (w_["u"] - W / 2, -0.13, w_["z"]), role="window_box")

    # ---- weathering: rising damp and splash-back at the foot, rain streaks under every ledge and sill end, run-off
    #      under the raking gable copings, soot down the wall below each gable stack and above the lantern, green
    #      at the pavement line and on the north side
    ledges = [(zf[f] - 0.12, 1.0) for f in range(1, n)] + [(z_en, 1.35)]
    stains = [(w_["s"], w_["u"], S[w_["style"]]["w"] / 2 + 0.14, w_["z"] - 0.09) for w_ in wins]
    soot_walls = [s for s in stacked if s in gables]

    def blot(x, y, z, k):
        return 0.5 + 0.5 * np.sin((x + y) * 1.7 * k + 1.3 * np.sin(z * 1.1 * k + seed)) * np.cos(z * 1.3 * k + 1.7 * np.sin((x - y) * 0.8 * k + seed * 0.37))

    def streaks(co, nrm):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        t = x * 1.0 + y * 1.3
        s = 0.5 + 0.5 * np.sin(t * 5.1 + 2.0 * np.sin(t * 1.7 + seed))
        g = np.clip(1.0 - (z - 0.1) / (0.8 + 1.1 * blot(x, y, z * 0.3, 1.3)), 0.0, 1.0) ** 1.4 * 0.75     # damp rising from the pavement
        g = np.maximum(g, 0.25 * blot(x, y, z, 0.5) ** 2)                                                # old patchy weather-staining
        for (zl, reach) in ledges:
            d = zl - z
            g = np.maximum(g, np.where((d > -0.05) & (d < reach), (1.0 - np.clip(d / reach, 0, 1.0)) * (0.2 + 0.8 * s) * 0.62, 0.0))
        wu = [(np.abs(y) < 0.3, x + W / 2), (np.abs(x - W / 2) < 0.3, y), (np.abs(y - D) < 0.3, W / 2 - x), (np.abs(x + W / 2) < 0.3, D - y)]
        for (ws, uc, hw, zs_) in stains:
            on, u = wu[ws]
            d = zs_ - z
            du = np.abs(u - uc) / hw
            m = on & (du < 1.08) & (d > -0.02) & (d < 1.1)
            g = np.maximum(g, np.where(m, (1.0 - np.clip(d / 1.1, 0.0, 1.0)) ** 1.5 * (0.25 + 0.55 * du * du), 0.0))
        for sg in gables:                       # water shed by the coping runs down the gable under its rake
            on, u = wu[sg]
            L = wlen[sg]
            d = z_e + gab[0] + (gab[1] - gab[0]) * (1.0 - np.abs(u - L / 2) / (L / 2)) - z
            g = np.maximum(g, np.where(on & (z > z_e - 0.4) & (d < 1.5), (1.0 - np.clip(d / 1.5, 0.0, 1.0)) * (0.3 + 0.7 * s) * 0.8, 0.0))
        for sg in soot_walls:                   # soot washed down from the stack
            on, u = wu[sg]
            du = (u - wlen[sg] / 2) / 0.7
            dz = (z_ridge - z) / 4.2
            g = np.maximum(g, np.where(on & (dz < 1.0), np.exp(-du * du) * (1.0 - np.clip(dz, 0.0, 1.0)) ** 1.2 * (0.2 + 0.8 * s) * 0.62, 0.0))
        if p.lantern:
            r = np.hypot((x - lan_x) / 0.5, (z - (lan_z + 1.15)) / 0.9)
            g = np.maximum(g, np.where(wu[0][0], np.clip(1.0 - r, 0.0, 1.0) * 0.8, 0.0))
        g = g + np.where(wu[2][0], 0.08, 0.0)
        return np.clip(g * (0.5 + age), 0.0, 1.0)

    north = facing((0, 1, 0))
    damp = mask_mul(height_ramp(0.0, 2.2), noise_mask(0.7, seed=seed + 8, lo=0.3, hi=0.75), 0.25 + 0.6 * age)
    foot = mask_mul(height_ramp(0.03, 0.4), noise_mask(0.45, seed=seed + 9, lo=0.25, hi=0.7), 0.5 + 0.5 * age)

    def moss(co, nrm):
        return np.clip(damp(co, nrm) * (0.35 + 0.65 * north(co, nrm)), 0.0, 1.0)

    def moss_foot(co, nrm):
        return np.clip(np.maximum(moss(co, nrm), foot(co, nrm) * (0.6 + 0.4 * north(co, nrm))), 0.0, 1.0)

    for o in wall_objs + rust_objs + dress + breast_objs:
        paint(o, grime=streaks, special=moss)
    for o in plinth + [b_ for b_ in breast_objs if "plinth" in b_.name]:
        # the plinth is the dirtiest, dampest stone of the house: splash-back from the pavement, green at its foot
        paint(o, grime=lambda co, nrm: np.clip(0.08 + 0.3 * streaks(co, nrm) + 0.3 * np.clip(1.0 - co[:, 2] / 0.25, 0.0, 1.0), 0.0, 1.0), special=moss_foot,
              wear=0.15)
    if roof_o is not None:                      # lichen and soot on the slopes; the gable faces of the roof weather like the walls
        paint(roof_o, special=roof_moss,
              grime=lambda co, nrm: np.maximum(np.maximum(roof_grime(co, nrm), 0.45 * roof_moss(co, nrm)),
                                               streaks(co, nrm) * (np.abs(nrm[:, 2]) < 0.3) * (co[:, 2] < z_ridge - 0.25)))
    for o in dormer_objs:
        paint(o, grime=lambda co, nrm: np.clip((0.25 + 0.5 * age) * (1.0 - (co[:, 2] - z_e) / 6.0), 0.0, 1.0), special=0.1 + 0.3 * age)
