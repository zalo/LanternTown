"""Wrought-iron wall bracket carrying a glazed lantern (composes `lantern_head`).

Construction, wall outwards: a forged back plate with shaped ends, coach-bolted to the masonry; a
shouldered arm tenoned into it; scrollwork or a stay bar bound to arm and plate with wrapped collars
and rivets; and the lantern, either standing on a turned pedestal above the arm or hanging from a
staple under it on a few chain links (it sways).  Three bracket styles share that anatomy:
`scroll` (square arm with a curled tip, S-scroll stay and a corner volute), `swan` (tapering round
swan-neck with a tie bar that ends in a volute inside the bend) and `gallows` (plain smith's
bracket: twisted square arm, riveted diagonal stay, ring in the corner).  The scrollwork sits on
the side away from the lantern - below a standing lantern (strut), above a hanging one (tie).

The lantern is the catalogue `lantern_head` frame (tray, bars, roof, vent) which this fitting
re-glazes and re-burners as a house-front lantern: clear panes with a smoke film that deepens towards
the roof (thin stacked veils - glass takes no vertex masks), a pierced cresting strip on the eaves,
optionally a blind sheet-iron pane with a brass reflector dish on the wall side, and either a piped
gas burner (copper supply from a flat wall rose through an elbow, along the arm and up into the tray;
brass riser, regulator, gallery and a squat gauze mantle) or an oil font with a flat-wick flame.

Weather is modelled, not only masked: where paint has let go there are rusty-iron runs below the
coach bolts and joints, flakes on the plate edges and the top of the arm, and rusty cuffs at the
collars.  The fitting owns its light: the wall is an arm's length behind the flame, so the source is
only a few candela (~8 x reach^2) and its shadows start beyond the bracket - an amber pool that fades
within a metre, and ironwork that never stripes its own wall.

Mount 'wall': origin = centre of the back plate on the wall plane y = 0, everything projects to -Y.
"""
import math
import bmesh
import numpy as np
from tk import *

META = {"name": "Wall lantern on scroll bracket", "class": "component", "mount": "wall",
        "mount_height": 2.9, "focus": [0.06, -0.42, 0.2], "lod": "far"}
PARAMS = {
    "reach": {"default": 0.58, "min": 0.38, "max": 0.9, "desc": "distance from the wall to the lantern axis (m)"},
    "style": {"default": "scroll", "choices": ["scroll", "swan", "gallows"], "desc": "bracket style"},
    "carry": {"default": "stand", "choices": ["stand", "hang"], "desc": "lantern stands on the arm or hangs from it"},
    "roof": {"default": "ogee", "choices": ["ogee", "pyramid", "dome"], "desc": "lantern roof style"},
    "lantern_width": {"default": 0.23, "min": 0.22, "max": 0.36, "desc": "width of the lantern body (m); the ironwork sections follow it"},
    "crown": {"default": True, "desc": "pierced cresting strip on the lantern eaves"},
    "reflector": {"default": False, "desc": "blind sheet-iron pane with a brass reflector dish on the wall side of the lantern"},
    "drop": {"default": 2, "min": 0, "max": 6, "desc": "chain links between the arm and a hanging lantern"},
    "finish": {"default": "painted", "choices": ["painted", "forged"], "desc": "painted ironwork or bare blacked forge finish"},
    "paint": {"default": "#1f3128", "desc": "paint colour of the ironwork (finish = painted)"},
    "gilt": {"default": True, "desc": "brass rosette and finials instead of plain iron ones"},
    "gas_pipe": {"default": True, "desc": "exposed copper gas supply with stopcock (standing lanterns); otherwise an oil burner"},
    "rust": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "amount of rust at joints, fixings, flaked paint and drips"},
    "intensity": {"default": 2.6, "min": 1, "max": 40,
                  "desc": "light intensity (cd). The wall is only an arm's length behind the flame: ~8 x reach^2 cd keeps the "
                          "wall pool amber instead of white"},
    "lit": {"default": True, "desc": "burner alight"},
}
VARIANTS = {
    "swan_hanging": {"style": "swan", "carry": "hang", "reach": 0.62, "roof": "dome", "lantern_width": 0.24,
                     "paint": "#20407c", "drop": 1, "intensity": 3.0, "seed": 4},
    "gallows_forged": {"style": "gallows", "carry": "hang", "reach": 0.52, "roof": "pyramid", "finish": "forged",
                       "gilt": False, "crown": False, "drop": 3, "rust": 0.8, "intensity": 2.1, "seed": 9},
    "grand_scroll": {"style": "scroll", "carry": "hang", "reach": 0.8, "lantern_width": 0.32, "paint": "#5c1711",
                     "drop": 2, "intensity": 4.8, "rust": 0.3, "seed": 13},
    "swan_standing": {"style": "swan", "carry": "stand", "reach": 0.5, "lantern_width": 0.22, "roof": "pyramid",
                      "paint": "#1b1c1f", "rust": 0.55, "intensity": 2.0, "seed": 21},
    "gallows_standing": {"style": "gallows", "carry": "stand", "reach": 0.66, "lantern_width": 0.28, "roof": "dome",
                         "paint": "#43565e", "gas_pipe": False, "rust": 0.65, "intensity": 3.3, "seed": 30},
}

TP = 0.008          # back plate thickness (m)


# ------------------------------------------------------------------ 2D path helpers (d = out from wall, h)

def _volute(c, r_out, r_in, a_out, turns, rot, per_turn=13, ease=1.3):
    """Scroll end: points from the OUTER end (at angle a_out, degrees) winding inwards; rot = +1 ccw / -1 cw.
    The radius shrinks slowly at first (ease) so the outer end leaves tangentially."""
    n = max(8, int(round(turns * per_turn)))
    out = []
    for i in range(n + 1):
        t = i / n
        a = radians(a_out) + rot * 2.0 * pi * turns * t
        r = r_out * (r_in / r_out) ** (t ** ease)
        out.append((c[0] + r * cos(a), c[1] + r * sin(a)))
    return out


def _bez(p0, p1, p2, p3, n=10):
    out = []
    for i in range(n + 1):
        t = i / n
        w = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3)
        out.append((w[0] * p0[0] + w[1] * p1[0] + w[2] * p2[0] + w[3] * p3[0],
                    w[0] * p0[1] + w[1] * p1[1] + w[2] * p2[1] + w[3] * p3[1]))
    return out


def _line(p0, p1, step=0.035):
    n = max(1, int(math.ceil(hypot(p1[0] - p0[0], p1[1] - p0[1]) / step)))
    return [(p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n) for i in range(n + 1)]


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln = hypot(dx, dy) or 1.0
    return dx / ln, dy / ln


def _fillet(pts, r, n=4):
    """Round the corners of a polyline with radius-ish r (quadratic blend)."""
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        la, lb = hypot(p0[0] - p1[0], p0[1] - p1[1]), hypot(p2[0] - p1[0], p2[1] - p1[1])
        rr = min(r, la * 0.45, lb * 0.45)
        ua, ub = _unit(p1, p0), _unit(p1, p2)
        s = (p1[0] + ua[0] * rr, p1[1] + ua[1] * rr)
        e = (p1[0] + ub[0] * rr, p1[1] + ub[1] * rr)
        for k in range(n + 1):
            t = k / n
            out.append(((1 - t) ** 2 * s[0] + 2 * (1 - t) * t * p1[0] + t * t * e[0],
                        (1 - t) ** 2 * s[1] + 2 * (1 - t) * t * p1[1] + t * t * e[1]))
    out.append(pts[-1])
    return out


def _densify(pts, step=0.03):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        out += _line(a, b, step)[1:]
    return out


def _double_scroll(cA, rA, cB, rB, s, kind="S", turnsA=1.6, turnsB=1.5, rin=0.17):
    """Two volutes joined by a stem: returns (A inner->outer, stem, B outer->inner).
    s = rotation sense of volute A when traversed outwards; 'S' reverses the sense for B, 'C' keeps it."""
    ang = degrees(atan2(cB[1] - cA[1], cB[0] - cA[0]))
    A = _volute(cA, rA, rA * rin, ang - s * 90.0, turnsA, -s)[::-1]
    q = -s if kind == "S" else s
    B = _volute(cB, rB, rB * rin, ang - q * 90.0, turnsB, q)
    tA, tB = _unit(A[-2], A[-1]), _unit(B[0], B[1])
    L = hypot(B[0][0] - A[-1][0], B[0][1] - A[-1][1])
    stem = _bez(A[-1], (A[-1][0] + tA[0] * L * 0.38, A[-1][1] + tA[1] * L * 0.38),
                (B[0][0] - tB[0] * L * 0.38, B[0][1] - tB[1] * L * 0.38), B[0], n=max(6, int(L / 0.022)))
    return A, stem, B


def _near(pt, r):
    """Mask helper: 1 at world point pt fading to 0 at distance r."""
    q = np.array(pt, dtype=np.float64)
    return lambda co, n: np.clip(1.0 - np.linalg.norm(co - q, axis=1) / r, 0.0, 1.0)


def _mask_max(*fns):
    return lambda co, n: np.max(np.stack([f(co, n) for f in fns]), axis=0)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fillet3(pts, r, n=5):
    """Round the corners of a 3D polyline (pipe bends)."""
    out = [tuple(pts[0])]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = Vector(pts[i - 1]), Vector(pts[i]), Vector(pts[i + 1])
        rr = min(r, (p0 - p1).length * 0.45, (p2 - p1).length * 0.45)
        s = p1 + (p0 - p1).normalized() * rr
        e = p1 + (p2 - p1).normalized() * rr
        for j in range(n + 1):
            t = j / n
            out.append(tuple(s * (1 - t) ** 2 + p1 * (2 * (1 - t) * t) + e * (t * t)))
    out.append(tuple(pts[-1]))
    return out


# ------------------------------------------------------------------ back plate

def _wide(far, prof):
    """lod 2: a turning of six sides drawn out to the girth of the round one (the shade it throws stays whole)."""
    return [(r * 1.15, z) for r, z in prof] if far else prof


def _plate_rows(half_h, hw, end, far=False):
    """(z, half width) rows of the plate outline from the bottom tip to the top tip (far: lod 2, the outline of each
    end in four rows and the straight run in three steps)."""
    if end == "spade":
        ln = hw * 2.0
        tab = [(0.0, 0.05), (0.06, 0.22), (0.14, 0.42), (0.24, 0.61), (0.36, 0.76), (0.48, 0.82), (0.58, 0.78), (0.67, 0.64),
               (0.74, 0.50), (0.80, 0.44), (0.86, 0.50), (0.92, 0.72), (0.97, 0.92)]
        cap = [(-ln * (1.0 - s), w * hw) for s, w in tab]
    elif end == "round":
        rl = hw * 1.32
        zc = -rl * 0.86
        cap = []
        for a in (172, 158, 142, 124, 106, 90, 74, 58, 44):
            cap.append((zc + rl * cos(radians(a)), rl * sin(radians(a))))
        cap.append((zc + rl * cos(radians(36)), hw * 0.72))
        cap.append((-0.004, hw * 0.86))
    else:  # clipped corners
        c = hw * 0.26
        cap = [(0.0, hw - c), (c * 0.5, hw - c * 0.5)]
    if far:
        cap = [cap[i] for i in sorted({0, len(cap) // 3, (2 * len(cap)) // 3, len(cap) - 1})]
    rows = [(-half_h + z, w) for z, w in cap]
    z0 = -half_h + (hw * 0.26 if end == "clipped" else 0.006)
    n = 3 if far else max(4, int(round((half_h * 2) / 0.0125)))
    for i in range(n + 1):
        z = z0 + (2 * half_h - 2 * (z0 + half_h)) * i / n
        rows.append((z, hw))
    rows += [(half_h - z, w) for z, w in reversed(cap)]
    return rows


def _plate(name, half_h, hw, end, bev=0.0025, far=False):
    """Forged plate against the wall (back face omitted): bevelled rim, dense enough for the rust / flake masks.
    far: lod 2 - the same outline in a dozen rows, square edges, one face across."""
    rows = _plate_rows(half_h, hw, end, far)
    verts, faces = [], []
    fr = (-0.5, 0.0, 0.5)
    for z, w in rows:
        b = min(bev, w * 0.4)
        if far:
            xs = [(-w, 0.0), (-w, -TP), (w, -TP), (w, 0.0)]
        else:
            xs = [(-w, 0.0), (-w, -(TP - b)), (-(w - b), -TP)] + [(f * (w - b), -TP) for f in fr] + [(w - b, -TP), (w, -(TP - b)), (w, 0.0)]
        verts += [(x, y, z) for x, y in xs]
    m = 4 if far else 6 + len(fr)
    for j in range(len(rows) - 1):
        for i in range(m - 1):
            a = j * m + i
            faces.append((a, a + 1, a + m + 1, a + m))
    faces.append(tuple(range(0, m)))
    faces.append(tuple(reversed(range((len(rows) - 1) * m, len(rows) * m))))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    uv_box(o)
    smooth(o, 38)
    return o


# ------------------------------------------------------------------ lantern helpers

def _strip_component(obj, drop, swap):
    """Remove faces of a placed component (drop(material name, world centre) -> bool), swap materials
    (swap(material) -> material) and prune the emptied slots."""
    me = obj.data
    M = world_matrix(obj)
    mats_ = list(me.materials)
    bm = bmesh.new()
    bm.from_mesh(me)
    dead = [f for f in bm.faces if drop(mats_[f.material_index].name.lower(), M @ f.calc_center_median())]
    bmesh.ops.delete(bm, geom=dead, context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.to_mesh(me)
    bm.free()
    used = sorted({pl.material_index for pl in me.polygons})
    remap = {old: new for new, old in enumerate(used)}
    idx = [remap[pl.material_index] for pl in me.polygons]
    me.materials.clear()
    for i in used:
        me.materials.append(swap(mats_[i]))
    me.polygons.foreach_set("material_index", idx)
    me.update()


def _subdivide_tall(obj, min_dz, cuts):
    """Cut the long upright edges of a mesh (the lantern's corner bars are single spans) so vertex masks can vary
    along them instead of smearing from end to end."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    ed = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > min_dz]
    if ed:
        bmesh.ops.subdivide_edges(bm, edges=ed, cuts=cuts, use_grid_fill=True)
    bm.to_mesh(me)
    bm.free()
    me.update()


def _flake(c, eu, ev, a, b, rng, lim=None, n=10):
    """Irregular patch of lost paint lying in the plane (eu, ev) through c (eu x ev = outward normal): fan of n
    triangles. lim(u, v) -> (u, v) keeps the outline on the face it lies on (flakes stop dead at an arris)."""
    c, eu, ev = Vector(c), Vector(eu), Vector(ev)
    uv = [(0.0, 0.0)]
    ph = rng.u(0.0, 6.28)
    for i in range(n):
        an = 2 * math.pi * (i + rng.jitter(0.0, 0.28)) / n
        r = rng.u(0.55, 1.0) * (0.72 + 0.28 * math.cos(2 * an + ph))
        uv.append((a * r * math.cos(an), b * r * math.sin(an)))
    if lim:
        uv = [lim(u, v) for u, v in uv]
    verts = [tuple(c + eu * u + ev * v) for u, v in uv]
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return verts, faces


def _crest_strip(name, n, pitch, H, th, tall_ends, walls=False, far=0):
    """Pierced cresting: a sheet strip (x along, z up, y = 0 outer face .. th) of pointed lobes, each pierced with a
    lozenge; the end lobes are taller so two strips meeting at a corner make an acroterion. 2 mm sheet: front and
    back faces (the edge walls are optional - they are sub-pixel beyond arm's length).
    far: lod 2 - a plain band of that many pieces, as high as the pierced lobes have iron (front and back face)."""
    verts, faces = [], []
    if far:
        verts = [(n * pitch * i / far, y, z) for i in range(far + 1) for y in (0.0, th) for z in (0.0, 0.7 * H)]
        for i in range(far):
            a = 4 * i
            faces += [(a, a + 4, a + 5, a + 1), (a + 2, a + 3, a + 7, a + 6)]
        n = 0
    for i in range(n):
        end = tall_ends and i in (0, n - 1)
        Hp = H * (1.5 if end else (1.0 if i % 2 == 0 else 0.84))
        hv = 0.46 * H
        hs = hv + 0.45 * (Hp - hv)
        ht = 0.66 * H + (0.3 * H if end else 0.0)
        fr = [(0.0, 0.0), (1.0, 0.0), (1.0, hv), (0.74, hs), (0.5, Hp), (0.26, hs), (0.0, hv),
              (0.5, 0.1 * H), (0.82, 0.34 * H), (0.5, ht), (0.18, 0.34 * H)]
        b = len(verts)
        verts += [((i + u) * pitch, 0.0, z) for u, z in fr]
        verts += [((i + u) * pitch, th, z) for u, z in fr]
        for f in [(0, 1, 7), (1, 2, 8, 7), (2, 3, 8), (3, 4, 9, 8), (4, 5, 10, 9), (5, 6, 10), (6, 0, 7, 10)]:
            faces.append(tuple(b + q for q in f))
            faces.append(tuple(b + 11 + q for q in reversed(f)))
        if walls:
            for a, c in [(2, 3), (3, 4), (4, 5), (5, 6)]:
                faces.append((b + a, b + 11 + a, b + 11 + c, b + c))
    o = mesh_object(name, verts, faces)
    uv_box(o)
    return o


# ------------------------------------------------------------------ build

def build(p, rng):
    R = p.reach
    far = lod() == 2                       # from 44 m on: lantern_head brings its own far model, the rest is cut down here
    hang = p.carry == "hang"
    sg = 1.0 if hang else -1.0            # scrollwork side: +z for a hanging lantern, -z for a standing one
    k = clamp(R / 0.55, 0.8, 1.35)         # ornament scale follows the reach
    forged = p.finish == "forged"
    lw = p.lantern_width
    rust = p.rust
    r_ir = rng.fork("iron")
    gas = bool(p.gas_pipe) and not hang    # piped gas burner with a mantle; otherwise an oil font with a flat-wick flame

    iron = mat("iron_wrought") if forged else mat("iron_painted", p.paint, rough=0.62, special=1.25, wear=0.2)
    fix_m = mat("iron_rusty") if rust > 0.12 else mat("iron_wrought")
    brass = mat("brass")
    copper = mat("brass", "#f0b890", rough=1.15)

    irons, brasses, accents, fixings, rusts = [], [], [], [], []      # mesh objects by material
    joints, tips = [], []                       # world points where rust gathers / where scroll ends get knocked
    riv_m, bolt_m, nail_m = [], [], []
    st = {"zc": 0.0}

    # bar sections grow with the lantern they have to carry (and a little with the reach)
    s = clamp(0.0235 + 0.04 * (lw - 0.22) + 0.012 * (R - 0.38), 0.024, 0.034)

    def W(d, h, x=0.0):
        return (x, -d, st["zc"] + sg * h)

    def bar(name, pts, prof, taper=None, x=0.0, wobble=0.0012, ends=False, **kw):
        o = sweep(name, [W(d, h, x) for d, h in pts], prof, smooth_path=False, up=(1, 0, 0), scale_fn=taper, **kw)
        if wobble:
            displace_noise(o, strength=wobble, scale=0.06, seed=p.seed + len(irons))
        irons.append(o)
        if ends:
            tips.extend([W(*pts[0], x), W(*pts[-1], x)])
        return o

    def collar(d, h0, h1, wx, ln=0.02, tilt=0.0):
        """Wrapped band binding two bars together (h0..h1 across both)."""
        c = W(d, (h0 + h1) / 2)
        rot = (tilt + r_ir.jitter(0.0, 2.5), 0, r_ir.jitter(0.0, 2.0))
        o = box("collar", (wx, ln, abs(h1 - h0)), loc=c, rot=rot, bevel=0.0028)
        irons.append(o)
        if rust > 0.12:
            # rust creeping out from under the band, where the bars are clamped metal to metal and never get painted
            rusts.append(box("collar_rust", (wx - 0.0045, ln + 0.012 + 0.02 * rust, abs(h1 - h0) - 0.0045), loc=c, rot=rot))
        joints.append(c)
        return o

    def m_dir(loc, direction, spin=0.0):
        q = Vector(direction).normalized().to_track_quat('Z', 'Y')
        return Matrix.Translation(Vector(loc)) @ q.to_matrix().to_4x4() @ Matrix.Rotation(radians(spin), 4, 'Z')

    def rivet(d, h, nd, nh, x=0.0):
        """Dome rivet at (d, h) with its head pointing along (nd, nh) in the bracket plane."""
        loc = W(d, h, x)
        riv_m.append(m_dir(loc, (0.0, -nd, sg * nh), r_ir.u(0, 360)))
        joints.append(loc)

    sq = lambda s_, r=0.003: rounded_rect(s_, s_, r, 1)
    band = lambda t_, w_: rounded_rect(t_, w_, 0.0025, 1)
    ends2 = lambda u: 0.46 + 0.54 * min(smoothstep(0.0, 0.16, u), smoothstep(0.0, 0.16, 1.0 - u))
    runs = []                                  # (x, z, strength): where rust water starts running down the plate

    # =============================================================== bracket styles
    if p.style == "scroll":
        t, bw = 0.38 * s, 0.95 * s                # scroll flat bar: thickness (in plane) x width, 10 x 25 mm
        rA, rB = 0.066 * k * r_ir.vary(1.0, 0.06), 0.045 * k * r_ir.vary(1.0, 0.06)
        hA = 0.215 * k
        rA2, rB2 = 0.043 * k * r_ir.vary(1.0, 0.06), 0.031 * k * r_ir.vary(1.0, 0.06)
        hA2 = 0.128 * k
        h_lo, h_hi = -(hA2 + rA2 * 0.75 + 0.036), hA + rA * 0.75 + 0.034
        hw, end = 1.75 * s, "spade"
        st["zc"] = -sg * (h_lo + h_hi) / 2
        arm_s, arm_end = s, R + 0.068 * k

        # arm with a tapering curled tip (curls to the scrollwork side)
        rT = 0.038 * k
        straight = _line((TP - 0.002, 0.0), (arm_end, 0.0), 0.016)
        curl = _volute((arm_end, rT), rT, rT * 0.24, -90.0, 1.3, +1)
        Ls = arm_end
        Lc = sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(curl[:-1], curl[1:]))
        f0 = Ls / (Ls + Lc)
        bar("arm", straight + curl[1:], sq(s), taper=lambda u: 1.0 - 0.52 * smoothstep(f0 - 0.04, 1.0, u))
        tips.append(W(*curl[-1]))

        def fitted(cA, rA_, cB, rB_, tag):
            """S-scroll whose big volute bears on the plate and whose small one bears on the arm."""
            ta_, tb_ = r_ir.fork(tag + "a").vary(1.5, 0.05), r_ir.fork(tag + "b").vary(1.35, 0.05)
            for _ in range(5):
                A_, stem_, B_ = _double_scroll(cA, rA_, cB, rB_, -1, "S", ta_, tb_, rin=0.24)
                ia_ = min(range(len(A_)), key=lambda i: A_[i][0])
                ib_ = min(range(len(B_)), key=lambda i: B_[i][1])
                cA[0] += (TP + t / 2) - A_[ia_][0]
                cB[1] += (s / 2 + t / 2) - B_[ib_][1]
            A_, stem_, B_ = _double_scroll(cA, rA_, cB, rB_, -1, "S", ta_, tb_, rin=0.24)
            return A_ + stem_[1:] + B_[1:], A_[ia_], B_[ib_]

        # S-scroll stay: big volute riveted to the plate, small one collared under the arm
        path, pa, pb = fitted([TP + t / 2 + rA * 0.7, hA], rA, [min(0.66 * R, R - 0.11), s / 2 + t / 2 + rB], rB, "s")
        bar("s_scroll", path, band(t, bw), taper=ends2, ends=True)
        rivet(pa[0] + t / 2, pa[1], 1.0, 0.0)
        runs.append((r_ir.jitter(0.0, 0.006), W(*pa)[2], 0.7))
        collar(pb[0], -s / 2 - 0.003, s / 2 + t + 0.003, max(s, bw) + 0.01, 0.026)

        # counter scroll on the lantern side of the arm (mirrored), same construction at a smaller scale
        path2, pa2, pb2 = fitted([TP + t / 2 + rA2 * 0.7, hA2], rA2, [max(0.27 * R, 0.13), s / 2 + t / 2 + rB2], rB2, "c")
        bar("counter_scroll", [(d, -h) for d, h in path2], band(t * 0.88, bw * 0.9), taper=ends2, ends=True)
        rivet(pa2[0] + t / 2, -pa2[1], 1.0, 0.0)

        # corner volute with a tail lying along the arm; one collar binds it, the arm and the counter scroll
        r1 = 0.031 * k
        c1 = [TP + t / 2 + r1 * 0.85, s / 2 + t / 2 + r1]
        v1 = _volute(c1, r1, r1 * 0.26, -90.0, 1.2, -1)
        dx = (TP + t / 2) - min(q[0] for q in v1)
        v1 = [(q[0] + dx, q[1]) for q in v1]
        tail = _line((pb2[0] + 0.02, v1[0][1]), v1[0], 0.03)
        bar("corner_scroll", tail + v1[1:], band(t * 0.88, bw * 0.9), taper=ends2, ends=True)
        collar(pb2[0], -s / 2 - t - 0.003, s / 2 + t + 0.003, max(s, bw) + 0.01, 0.024)
        ros_h = (v1[0][1] + r1 * 2 + pa[1] - rA * 0.75) / 2
        tip_pt = None

    elif p.style == "swan":
        rn = 0.52 * s                              # neck radius at the root
        t, bw = 0.3 * s, 0.85 * s
        rho = 0.30 * R
        hc = 0.13 * k
        he = hc - 0.085 * k
        rV = 0.5 * rho
        hT = hc + 0.05 * rho + rV
        h_lo, h_hi = -(0.05 + 0.3 * lw), hT + 0.06
        hw, end = 1.5 * s, "round"
        st["zc"] = -sg * (h_lo + h_hi) / 2
        arm_s = 2 * rn * 0.72

        d0 = R - 2 * rho
        neck = _bez((TP - 0.002, 0.0), (d0 * 0.95, 0.0), (d0, hc * 0.25), (d0, hc), n=12)
        arc = [(R - rho + rho * cos(radians(a)), hc + rho * sin(radians(a))) for a in np.linspace(180, 0, 17)]
        neck = neck + arc[1:] + _line((R, hc), (R, he), 0.03)[1:]
        bar("neck", neck, rn, taper=lambda u: 1.0 - 0.3 * u, sides=10, wobble=0.0008)
        # root husk on the plate, bead where the neck straightens
        irons.append(lathe("husk", [(rn + 0.002, 0.0), (rn + 0.013, 0.003), (rn + 0.014, 0.009), (rn + 0.006, 0.014), (rn + 0.009, 0.02),
                                    (rn + 0.008, 0.025), (rn - 0.001, 0.034)], segments=16, loc=W(TP, 0.0), rot=(90, 0, 0)))
        rb = rn * 0.78
        irons.append(lathe("bead", [(rb - 0.001, -0.016), (rb + 0.005, -0.01), (rb + 0.003, -0.004), (rb + 0.008, 0.0), (rb + 0.003, 0.004),
                                    (rb + 0.005, 0.01), (rb - 0.001, 0.016)], segments=14, loc=W(R, hc)))
        # water leaves wrapping the root
        lr = rn + 0.003
        leaf = surface("leaf", lambda u, v: ((u - 0.5) * 0.046 * k * (1.0 - 0.92 * v ** 2.4) * (0.55 + 0.45 * min(1.0, v * 5.0)),
                                             -lr - 0.005 * sin(v * pi) - 0.013 * k * v ** 3 + 0.007 * (1 - cos((u - 0.5) * pi)) * (1 - v),
                                             v * 0.092 * k), 4, 7, thickness=0.003)
        realize(leaf)
        base_m = Matrix.Translation(Vector(W(TP + 0.018, 0.0))) @ Euler((radians(90), 0, 0)).to_matrix().to_4x4()
        lm = [base_m @ Matrix.Rotation(radians(a + r_ir.jitter(0, 6)), 4, 'Z') @ Matrix.Diagonal((1, 1, r_ir.vary(1.0, 0.12), 1))
              for a in (0, 120, 240)]
        irons.append(instances_to_mesh("leaves", leaf, lm, keep_base=False))

        # tie bar: foot riveted to the plate, runs out through the neck and ends in a volute inside the bend
        cV = (R - rho, hc + 0.05 * rho)
        vol = _volute(cV, rV, rV * 0.2, 90.0, 1.5, -1, per_turn=18)
        foot = _fillet([(TP + t / 2, hT + 0.05), (TP + t / 2, hT), (0.06, hT)], 0.012)
        tie = foot + _line((0.06, hT), vol[0], 0.035)[1:] + vol[1:]
        bar("tie", tie, band(t, bw), taper=lambda u: 0.5 + 0.5 * smoothstep(0.0, 0.2, 1.0 - u))
        tips.append(W(*vol[-1]))
        rivet(TP + t, hT + 0.03, 1.0, 0.0)
        runs.append((r_ir.jitter(0.0, 0.006), W(TP, hT + 0.03)[2], 0.8))
        dxn = R - rho - sqrt(max(rho * rho - (hT - hc) ** 2, 0.0))
        collar(dxn, hT - 0.017, hT + 0.017, bw + 0.011, 0.03)
        ros_h = hT * 0.5
        tip_pt = None

    else:  # gallows
        t, bw = 0.32 * s, 1.0 * s
        hS = 0.2 + 0.26 * R
        dS = 0.74 * R
        h_lo, h_hi = -0.06, hS + 0.09
        hw, end = 2.0 * s, "clipped"
        st["zc"] = -sg * (h_lo + h_hi) / 2
        arm_s, arm_end = s, R + 0.07

        # arm: plain - twisted - plain square bar, forged knop at the tip
        ta, tb = 0.2 * R, 0.62 * R
        if hang:
            tb = min(tb, R - 0.12)
        bar("arm_a", _line((TP - 0.002, 0.0), (ta, 0.0), 0.014), sq(s))
        tw = sweep("arm_twist", [W(ta, 0.0), W(tb, 0.0)], sq(s * 0.94, 0.0015), smooth_path=False, step=0.007, up=(1, 0, 0),
                   twist_deg=360.0 * max(1, round((tb - ta) / 0.11)), cap=False)
        irons.append(tw)
        bar("arm_b", _line((tb, 0.0), (arm_end, 0.0), 0.014), sq(s))
        for d in (ta, tb):
            irons.append(box("arm_knuckle", (s + 0.006, 0.012, s + 0.006), loc=W(d, 0.0), bevel=0.003))
        tip = lathe("tip", [(s * 0.42, -0.004), (s * 0.7, 0.003), (s * 0.78, 0.011), (s * 0.5, 0.018), (s * 0.36, 0.024), (s * 0.6, 0.033),
                            (s * 0.74, 0.045), (s * 0.56, 0.059), (s * 0.2, 0.071), (0.0, 0.075)], segments=14, loc=W(arm_end, 0.0), rot=(90, 0, 0))
        accents.append(tip)

        # diagonal stay: flat bar with feet turned flat against plate and arm, two rivets each end
        f = 0.066
        st_pts = _fillet([(TP + t / 2, hS + f), (TP + t / 2, hS), (dS, s / 2 + t / 2), (dS + f, s / 2 + t / 2)], 0.016)
        bar("stay", _densify(st_pts, 0.03), band(t, bw))
        for q in (0.3, 0.78):
            rivet(TP + t, hS + f * q, 1.0, 0.0)
            rivet(dS + f * q, s / 2 + t, 0.0, 1.0)
        runs.append((r_ir.jitter(0.0, 0.008), W(TP, hS + f * 0.3)[2], 0.8))
        # ring wedged in the corner, bound to arm, plate and stay
        a_, b_ = dS - TP, hS - s / 2
        rr = (a_ + b_ - hypot(a_, b_)) / 2 * r_ir.vary(0.97, 0.02)
        wr = 0.26 * s
        cR = (TP + rr, s / 2 + rr)
        ring = torus("ring", rr - wr, wr, loc=W(*cR), rot=(0, 90, 0), major_seg=30, minor_seg=8)
        displace_noise(ring, strength=0.001, scale=0.05, seed=p.seed)
        irons.append(ring)
        collar(cR[0], -s / 2 - 0.003, s / 2 + 2 * wr + 0.002, s + 0.009, 0.018)
        un = _unit((TP, hS), (dS, s / 2))
        nrm = (-un[1], un[0]) if (-un[1] * 1 + un[0] * 1) > 0 else (un[1], -un[0])
        pc = (cR[0] + nrm[0] * (rr - wr), cR[1] + nrm[1] * (rr - wr))
        ang = degrees(atan2(nrm[1], nrm[0]))
        irons.append(box("ring_tie", (bw + 0.008, 2 * wr + t + 0.01, 0.018), loc=W(*pc), rot=(-sg * (ang - 90.0) + 90.0, 0, 0), bevel=0.0028))
        joints.append(W(*pc))
        ros_h = None
        tip_pt = W(arm_end, 0.0)

    zA = st["zc"]                               # world height of the arm axis at the wall
    half_h = (h_hi - h_lo) / 2

    # =============================================================== back plate and fixings
    plate = _plate("plate", half_h, hw, end, far=far)
    displace_noise(plate, strength=0.0009, scale=0.045, seed=p.seed + 3)
    if end == "spade":
        cap_len = hw * 2.0
        zb_ = half_h + cap_len * 0.5
        bolts = [(0.0, zb_), (0.0, -zb_)]
    elif end == "round":
        cap_len = hw * 1.32 * 1.86
        zb_ = half_h + hw * 1.32 * 0.86
        bolts = [(0.0, zb_), (0.0, -zb_)]
    else:
        cap_len = 0.0
        ex, ez = hw - 0.019, half_h - 0.021
        bolts = [(-ex, ez), (ex, ez), (-ex, -ez), (ex, -ez)]
    for (x, z) in bolts:
        if end == "clipped":
            nail_m.append(Matrix.Translation((x, -TP, z)) @ Euler((radians(90), 0, 0)).to_matrix().to_4x4()
                          @ Matrix.Rotation(radians(r_ir.u(0, 90)), 4, 'Z'))
        else:
            bolt_m.append(Matrix.Translation((x, -TP, z)) @ Euler((radians(90 + r_ir.jitter(0, 3)), 0, 0)).to_matrix().to_4x4()
                          @ Matrix.Rotation(radians(r_ir.u(0, 90)), 4, 'Z'))
        runs.append((x + r_ir.jitter(0.0, 0.004), z - 0.006, 1.0))
    # forged shoulder where the arm is tenoned through the plate
    if p.style != "swan":
        irons.append(box("shoulder", (arm_s + 0.024, 0.012, arm_s + 0.024), loc=W(TP + 0.005, 0.0), bevel=0.003))
    joints.append(W(TP, 0.0))
    for sx in (-1, 1):
        runs.append((sx * (arm_s / 2 + 0.008), zA - arm_s / 2, 0.75))
    if ros_h is not None:
        rr_ = 0.9 * s
        ros = lathe("rosette", [(0.0, 0.0), (rr_, 0.0), (rr_ * 1.1, 0.003), (rr_ * 0.8, 0.008), (rr_ * 0.52, 0.007), (rr_ * 0.38, 0.012), (0.0, 0.015)],
                    segments=24, loc=W(TP, ros_h), rot=(90, 0, 0))

        def petals(co):
            r_ = hypot(co.x, co.y)
            if r_ < rr_ * 0.42:
                return co
            f_ = 1.0 + 0.16 * cos(6 * atan2(co.y, co.x))
            return Vector((co.x * f_, co.y * f_, co.z))
        transform_verts(ros, petals)
        accents.append(ros)

    # =============================================================== lantern (composed lantern_head, re-glazed and re-burnered)
    hh = max(0.28, lw * 1.24)                   # glazed body height
    wt, wb = lw, lw * 0.56
    z0 = 0.07                                   # lantern_head: tray level above its collar
    z1 = z0 + hh
    rh_ = lw * (0.36 if p.roof == "pyramid" else 0.42)
    ztop = z1 + 0.02 + rh_ + 0.19
    # a hanging head has its origin at the bearing point of its suspension eye: lantern_head z_org = (z1 + 0.028 + rh + 0.19)
    # + 0.0828 above its collar datum. (This read ztop + 0.028, from before the head grew its strap crown: the panes, burner,
    # reflector and cresting fitted below then sat 63 mm too high in the frame, the glowing panes 3 mm behind the glazing fins.)
    dzh = -(ztop + 0.0908) if hang else 0.0
    head = use("lantern_head", width=lw, height=hh, roof=p.roof, paint=p.paint, crown=False, lit=False,
               mount="hang" if hang else "post", seed=p.seed + 1)
    # lit: seeded panes glowing a rich amber (kept under the bloom knee, thin enough that the burner shows through);
    # out: dark clear glass - never a pale pane
    glass = mat("glass_lantern", "#ffa94e", glow=1.15, opacity=0.27) if p.lit else mat("glass_clear", "#2c312d", opacity=0.13)
    soot_m = mat("glass_lantern", "#1d130b", glow=0.0, opacity=0.085)      # smoke film inside the top of the panes
    lit_c = "#ffcf8a" if gas else "#ffb45e"
    flame_m = mat("emissive", lit_c, glow=7.0 if gas else 6.0) if p.lit else mat("paper", "#8c877b" if gas else "#26211c")
    st["light"] = None

    def lantern(z_place, rz=0.0):
        """Place the head, strip its glazing / bulb burner, and fit soot-graded panes, a blind reflector pane, a real
        burner and a pierced cresting. Returns every object of the lantern."""
        zb = z_place + dzh
        lan = head.place((0, -R, z_place), rot=(0, 0, rz), sockets=False)

        def drop(nm, c):
            if "glass" in nm or "emissive" in nm:
                return True
            return "brass" in nm and hypot(c.x, c.y + R) < 0.03
        _strip_component(lan, drop, lambda m: iron if "iron" in m.name.lower() else (brass if "brass" in m.name.lower() else m))
        if not far:
            _subdivide_tall(lan, hh * 0.6, 2)
        out = [lan]

        def L(x, y, z):
            return (x, -R + y, zb + z)

        def side(a, u, w, z):                    # a = 0 front (-Y), 90 = +X, 180 = wall side, 270 = -X
            ca, sa = cos(radians(a)), sin(radians(a))
            return (u * ca + w * sa, -R + u * sa - w * ca, zb + z)

        # CLEARANCE RULE (depth fighting): a glowing sheet keeps >= 4 mm to anything parallel in front of or behind it.
        # The panes lie 4 mm inside the nominal face of the head on its true taper: the glazing fins / rail lips are then
        # 4.8 mm behind them and the door leaf 7 mm in front. The composed head also leans (lantern_head: seeded "lean",
        # keep the three lines below in step with it), so everything glazed in here is leaned with it - plumb panes in a
        # leaning frame drifted to within a fraction of a millimetre of the astragals.
        gi = 0.0041
        zg0, zs1 = z0 + 0.008, z1 - 0.006
        zg1 = z1 - 0.021 if far else zs1        # lod 2: the head's top rail is closed 20 mm under z1 - pane and soot end below its soffit
        hwz = lambda z: lerp(wb / 2, wt / 2, (z - z0) / hh) - gi
        # lod 2: the mid rail of the far head is a closed bar round the body - the pane stops short of it above and below
        mr_ = float(head.params["mid_rail"])
        zq_ = z0 + hh * clamp(mr_, 0.3, 0.85)
        pspans = [(zg0, zq_ - 0.01), (zq_ + 0.01, zg1)] if far and mr_ >= 0.05 else [(zg0, zg1)]
        blind = bool(p.reflector)
        rl_ = Rng(f"lantern_head/{head.params['seed']}").fork("lean")
        la_ = radians(rl_.u(0.5, 1.5) * (0.35 + 0.65 * float(head.params["age"])) * (0.6 if hang else 1.0))
        lz_ = rl_.u(0, 2 * pi) + radians(rz)
        Lean = (Matrix.Translation(Vector((0, -R, z_place))) @ Matrix.Rotation(la_, 4, Vector((cos(lz_), sin(lz_), 0.0)))
                @ Matrix.Translation(Vector((0, R, -z_place))))

        def leaned(o):
            apply_transform(o)
            return transform_verts(o, lambda co: Lean @ co)

        # --- panes: seeded glass with a smoke film inside - twelve very thin overlapping veils, deepest above the flame,
        # so the soot grades from clear glass at flame height to nearly black under the roof (glass takes no vertex masks)
        nc = 4
        for a in ((0, 90, 270) if blind else (0, 90, 180, 270)):
            g = mesh_object(f"pane{a}", [side(a, sx * hwz(z), hwz(z), z) for zs in pspans for z in zs for sx in (-1, 1)],
                            [(4 * i, 4 * i + 1, 4 * i + 3, 4 * i + 2) for i in range(len(pspans))])
            uv_box(leaned(g))
            assign(g, glass)
            out.append(g)
            for li in range(12):
                cov = 0.64 - 0.048 * li + r_ir.jitter(0.0, 0.008)
                # behind the glazing fins / rail lips (their inner faces are 8.1 mm inside the glowing pane): 4.4 mm and
                # more behind those - a host that merges this film into its lit glass must not get it against the iron
                off = 0.0125 + 0.0003 * li
                vs, fs = [], []
                ncl = 1 if far else nc          # lod 2: each veil one quad at its mean depth
                for i in range(ncl + 1):
                    u = -1.0 + 2.0 * i / ncl
                    zl = zs1 - (zs1 - zg0) * cov * (0.876 if far else 0.66 + 0.34 * cos(u * pi / 2))
                    for z in (zl, zg1):
                        vs.append(side(a, u * (hwz(z) - off - 0.0015), hwz(z) - off, z))
                for i in range(ncl):
                    fs.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
                sv = mesh_object(f"soot{a}_{li}", vs, fs)
                uv_box(leaned(sv))
                assign(sv, soot_m)
                out.append(sv)

        zf = z0 + 0.006                         # tray floor
        if gas:
            # --- gas burner: base boss on the tray, riser with union and regulator, mixing chamber, gallery, gauze mantle
            zg = z0 + hh * 0.3
            mh = clamp(hh * 0.15, 0.04, 0.054)
            zm0 = zg + 0.011
            gt_ = 0.03 if far else 0.017       # lod 2: the cup stands as high as the prongs it carries (those are left out)
            brasses.append(lathe("burner", _wide(far, [(0.027, zf), (0.029, zf + 0.004), (0.022, zf + 0.011), (0.012, zf + 0.016), (0.0085, zf + 0.024),
                                            (0.0085, zg - 0.052), (0.0128, zg - 0.049), (0.0128, zg - 0.037), (0.0085, zg - 0.034),
                                            (0.0085, zg - 0.029), (0.015, zg - 0.023), (0.019, zg - 0.012), (0.016, zg - 0.003), (0.0175, zg),
                                            (0.0255, zg + 0.003), (0.0285, zg + 0.008), (0.0285, zg + gt_), (0.0262, zg + gt_),
                                            (0.0262, zg + 0.009), (0.014, zg + 0.007), (0.0, zg + 0.007)]), segments=lod_pick(12, 12, 6), loc=L(0, 0, 0)))
            prong = box("prong", (0.0034, 0.0024, 0.014), loc=(0, 0, 0))
            brasses.append(lod_drop(instances_to_mesh("gallery_prongs", prong, [trs(L(0.0272 * cos(radians(a_)), 0.0272 * sin(radians(a_)), zg + 0.023),
                                                                           (0, 0, a_ + 90)) for a_ in range(20, 380, 45)], keep_base=False), above=1))
            brasses.append(cylinder("regulator", 0.0048, 0.044, loc=L(0.004, 0, zg - 0.043), rot=(0, 90, 0), verts=8))
            brasses.append(box("reg_lever", (0.0045, 0.03, 0.004), loc=L(0.027, -0.007, zg - 0.043), rot=(0, 0, r_ir.jitter(0, 12))))
            # copper feed: up through the tray floor (where the outside pipe arrives) and across into the burner base
            if p.style != "swan":                # (a swan neck is itself the pipe and comes up through the collar)
                xin = -min(0.058, wb / 2 - 0.017)
                feed = sweep("gas_feed", _fillet3([L(xin, 0, zf - 0.004), L(xin, 0, zf + 0.0125), L(-0.015, 0, zf + 0.0125)], 0.008, 3),
                             0.004, sides=6, smooth_path=False)
                assign(feed, copper)
                out.append(feed)
                brasses.append(cylinder("feed_nut", 0.0068, 0.008, loc=L(xin, 0, zf + 0.003), verts=6))
            # mantle fork: the wire that carries the gauze mantle
            fork = [L(0.0272, 0, zg + 0.016), L(0.0272, 0, zm0 + mh * 0.8), L(0.02, 0, zm0 + mh + 0.006), L(0.0, 0, zm0 + mh + 0.004)]
            brasses.append(sweep("mantle_fork", _fillet3(fork, 0.008, 2), 0.0014, sides=4, smooth_path=False))
            fl = lathe("mantle", [(0.0, zm0), (0.014, zm0 + 0.001), (0.0172, zm0 + mh * 0.3), (0.0166, zm0 + mh * 0.64),
                                  (0.012, zm0 + mh * 0.88), (0.005, zm0 + mh * 0.985), (0.0, zm0 + mh)], segments=12, loc=L(0, 0, 0))
        else:
            # --- oil burner: brass font on the tray, wick collar with a raiser wheel, flat-wick flame
            fs_ = clamp(lw / 0.24, 0.92, 1.25)
            pf = [(0.018, 0.0), (0.03, 0.003), (0.037, 0.012), (0.038, 0.021), (0.033, 0.031), (0.021, 0.038), (0.0135, 0.041)]
            pf = [(r_ * fs_, zf + z_ * fs_) for r_, z_ in pf]
            zc_ = pf[-1][1]
            pf += [(0.0135 * fs_, zc_ + 0.008), (0.02, zc_ + 0.011), (0.022, zc_ + 0.016), (0.022, zc_ + 0.025), (0.0205, zc_ + 0.025),
                   (0.0205, zc_ + 0.018), (0.009, zc_ + 0.016), (0.005, zc_ + 0.027), (0.0, zc_ + 0.027)]
            brasses.append(lathe("font", _wide(far, pf), segments=lod_pick(16, 16, 6), loc=L(0, 0, 0)))
            brasses.append(cylinder("wick_spindle", 0.0022, 0.036, loc=L(0.026, 0, zc_ + 0.013), rot=(0, 90, 0), verts=6))
            brasses.append(cylinder("wick_wheel", 0.008, 0.0035, loc=L(0.045, 0, zc_ + 0.013), rot=(0, 90, 0), verts=10, bevel=0.0008))
            mh = clamp(hh * 0.17, 0.042, 0.06)
            zm0 = zc_ + 0.024
            fl = lathe("flame", [(0.0, zm0), (0.0065, zm0 + mh * 0.1), (0.0095, zm0 + mh * 0.32), (0.0065, zm0 + mh * 0.62),
                                 (0.0022, zm0 + mh * 0.88), (0.0, zm0 + mh)], segments=10, loc=(0, 0, 0))
            if p.lit:
                transform_verts(fl, lambda co: Vector((co.x * 1.35, co.y * 0.6, co.z)))
            else:                               # out: only the charred wick stands above the collar
                transform_verts(fl, lambda co: Vector((co.x * 1.2, co.y * 0.4, zm0 + (co.z - zm0) * 0.14)))
            place(fl, loc=L(0, 0, 0))
        assign(fl, flame_m)
        out.append(fl)
        st["light"] = L(0, 0, zm0 + mh * 0.5)

        if blind:
            # --- blind pane of sheet iron on the wall side, carrying a dished brass reflector behind the flame
            zq = (zg0, zg1)
            vs = [side(180, sx * hwz(z), hwz(z) + dy, z) for dy in (0.0, 0.002) for z in zq for sx in (-1, 1)]
            pnl = mesh_object("blind_pane", vs, [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
            recalc_normals(pnl)
            uv_box(leaned(pnl))
            irons.append(pnl)
            out.append(pnl)
            zr_ = zm0 + mh * 0.55
            rd_ = min(0.06, hwz(zr_) * 0.74)
            tilt = degrees(math.atan((wt - wb) / 2 / (zg1 - zg0)))
            dish = lathe("reflector", [(0.0, 0.0035), (rd_ * 0.45, 0.0045), (rd_ * 0.8, 0.0085), (rd_, 0.0145), (rd_ + 0.004, 0.015),
                                       (rd_ + 0.005, 0.0125), (rd_ + 0.005, 0.0)], segments=18, loc=L(0, hwz(zr_) - 0.0005, zr_), rot=(90 - tilt, 0, 0))
            brasses.append(leaned(dish))

        if p.crown:
            # --- pierced cresting strip standing on the eaves (wrapped round the lip of a dome roof)
            Hc = clamp(0.118 * lw, 0.026, 0.034)
            if p.roof == "dome":
                rc_ = wt / 2 + 0.044
                n_ = max(12, 2 * int(round(pi * rc_ / 0.036)))
                pitch = 2 * pi * rc_ / n_
                cr = _crest_strip("crest", n_, pitch, Hc, 0.002, False, far=8 if far else 0)
                transform_verts(cr, lambda co: Vector(((rc_ - co.y) * sin(co.x / rc_), -(rc_ - co.y) * cos(co.x / rc_), co.z)))
                place(cr, loc=L(0, 0, z1 + 0.024))
            else:
                yc = wt / 2 + 0.0225
                n_ = max(5, int(round(2 * yc / 0.036)) | 1)
                base = _crest_strip("crest_base", n_, 2 * yc / n_, Hc, 0.002, True, far=1 if far else 0)
                cr = instances_to_mesh("crest", base, [Matrix.Translation(Vector(L(0, 0, z1 + 0.024))) @ Matrix.Rotation(radians(a), 4, 'Z')
                                                       @ Matrix.Translation(Vector((-yc, -yc, 0.0))) for a in (0, 90, 180, 270)], keep_base=False)
            flat(cr)
            irons.append(cr)
            out.append(cr)
        return lan, out

    lan_objs = []
    soot = None
    xe = -0.058                                 # where the gas pipe rises into the lantern tray
    if p.style == "swan":
        z_end = W(R, he)[2]
        if hang:
            irons.append(lathe("bell", [(rn * 0.7, -0.006), (0.014, 0.003), (0.021, 0.009), (0.023, 0.015), (0.016, 0.021), (0.0105, 0.026),
                                        (0.013, 0.033), (0.0085, 0.039), (0.0, 0.041)], segments=16, loc=(0, -R, z_end), rot=(180, 0, 0)))
            z_hook = z_end - 0.039
        else:
            cup = lathe("cup", [(rn * 0.7, -0.006), (0.015, 0.004), (0.011, 0.012), (0.014, 0.02), (0.02, 0.026), (0.016, 0.032), (0.03, 0.04),
                                (0.045, 0.046), (0.047, 0.052), (0.041, 0.055), (0.0, 0.055)], segments=20, loc=(0, -R, z_end))
            irons.append(cup)
            z_seat = z_end + 0.055
            if p.gas_pipe:
                # the swan neck is the gas pipe: brass stopcock near the root
                tp_ = W(0.085 * k + 0.03, 0.0)
                brasses += [cylinder("tap_body", rn + 0.0045, 0.022, loc=tp_, rot=(0, 90, 0), verts=12, bevel=0.002),
                            cylinder("tap_plug", 0.006, 2 * rn + 0.026, loc=tp_, rot=(0, 90, 0), verts=8),
                            box("tap_lever", (0.005, 0.012, 0.044), loc=(rn + 0.0145, tp_[1], tp_[2] + 0.013), rot=(r_ir.jitter(0, 14), 0, 0), bevel=0.0015)]
    else:
        if hang:
            z_hook = zA - arm_s / 2
            irons.append(box("saddle", (arm_s + 0.01, 0.032, arm_s + 0.01), loc=(0, -R, zA), bevel=0.003))
            riv_m.append(m_dir((0, -R, zA + arm_s / 2 + 0.005), (0, 0, 1)))
        else:
            zt = zA + arm_s / 2
            irons.append(box("saddle", (arm_s + 0.012, 0.04, arm_s + 0.012), loc=(0, -R, zA), bevel=0.003))
            pp = [(0.018, 0.004), (0.024, 0.008), (0.024, 0.013), (0.015, 0.019), (0.0115, 0.03), (0.0165, 0.041), (0.021, 0.047),
                  (0.015, 0.053), (0.015, 0.058), (0.03, 0.066), (0.045, 0.072), (0.047, 0.078), (0.041, 0.081), (0.0, 0.081)]
            ped = lathe("pedestal", [(r_, z_ * 0.8) for r_, z_ in pp], segments=16, loc=(0, -R, zt))
            irons.append(ped)
            z_seat = zt + 0.081 * 0.8
            drop_f = lathe("pendant", [(0.011, 0.004), (0.016, 0.009), (0.0095, 0.017), (0.013, 0.027), (0.0165, 0.036), (0.012, 0.048),
                                       (0.0045, 0.058), (0.0, 0.061)], segments=14, loc=(0, -R, zA - arm_s / 2), rot=(180, 0, 0))
            accents.append(drop_f)
            if p.gas_pipe:
                # copper supply: out of a brass wall rose beside the plate through an elbow, across to the arm, clipped along its
                # side, then up beside the pedestal and through a union into the lantern tray (the burner riser stands above)
                rp = 0.0045
                xs = -(arm_s / 2 + rp + 0.008)
                xg = -(hw + 0.052)
                yw = -0.05
                z_in = z_seat + 0.058
                d1 = R - 0.105
                pts = [(xg + 0.028, yw, zA), (xs, yw, zA)]
                nseg = max(2, int((d1 - 0.06) / 0.06))
                pts += [(xs + r_ir.jitter(0.0, 0.0008), -(0.07 + (d1 - 0.07) * i / nseg), zA + r_ir.jitter(0.0, 0.0008)) for i in range(1, nseg + 1)]
                pts += [(xe, -R, zA - 0.004), (xe, -R, z_in)]
                pipe = sweep("gas_pipe", _fillet3(pts, 0.024, 4), rp, sides=7, smooth_path=False)
                assign(pipe, copper)
                paint(pipe, special=mask_mul(noise_mask(0.05, seed=p.seed + 6, lo=0.4, hi=0.75), 0.25 + 0.6 * rust), grime=0.3)
                # wall rose with three screws, elbow with two union nuts
                brasses.append(lathe("rose", [(0.0, 0.0), (0.029, 0.0), (0.029, 0.002), (0.0265, 0.0036), (0.016, 0.0038), (0.014, 0.0062),
                                              (0.0, 0.0062)], segments=20, loc=(xg, 0.0, zA), rot=(90, 0, 0)))
                elbow = _fillet3([(xg, -0.004, zA), (xg, yw, zA), (xg + 0.036, yw, zA)], 0.019, 5)
                brasses.append(sweep("elbow", elbow, 0.0078, sides=8, smooth_path=False))
                brasses.append(cylinder("elbow_nut_a", 0.0112, 0.011, loc=(xg, -0.0125, zA), rot=(90, 0, 0), verts=6, bevel=0.001, segments=1))
                brasses.append(cylinder("elbow_nut_b", 0.0112, 0.012, loc=(xg + 0.031, yw, zA), rot=(0, 90, 0), verts=6, bevel=0.001, segments=1))
                scr = cylinder("rose_screw", 0.0034, 0.0022, loc=(0, 0, 0.0011), verts=6)
                brasses.append(instances_to_mesh("rose_screws", scr, [trs((xg + 0.0215 * cos(radians(a_)), -0.0036, zA + 0.0215 * sin(radians(a_))),
                                                                          (90, 0, 0)) for a_ in (80, 200, 320)], keep_base=False))
                brasses.append(cylinder("tray_union", 0.0092, 0.016, loc=(xe, -R, z_in - 0.007), verts=6, bevel=0.001, segments=1))
                tpy = -(0.07 + 0.3 * (d1 - 0.07))
                brasses += [cylinder("tap_body", 0.0088, 0.027, loc=(xs, tpy, zA + 0.001), verts=10, bevel=0.0015),
                            cylinder("tap_union", 0.0074, 0.032, loc=(xs, tpy, zA), rot=(90, 0, 0), verts=8, bevel=0.001),
                            box("tap_lever", (0.006, 0.04, 0.0045), loc=(xs, tpy - 0.009, zA + 0.0165), rot=(0, 0, r_ir.jitter(8, 16)), bevel=0.0015)]
                for dq in (0.07 + 0.08 * (d1 - 0.07), 0.07 + 0.82 * (d1 - 0.07)):
                    irons.append(box("pipe_clip", (abs(xs) + rp - arm_s / 2 + 0.006, 0.013, 2 * rp + 0.007),
                                     loc=(-(arm_s / 2 + abs(xs) + rp) / 2, -dq, zA), bevel=0.002))

    if hang:
        # staple under the arm -> chain links -> lantern eye; the whole drop swings
        Re, re_ = 0.013, 0.0037
        zE = z_hook - Re + 0.004
        eye = torus("staple", Re, re_, loc=(0, -R, zE), rot=(0, 90, 0), major_seg=16, minor_seg=6)
        irons.append(eye)
        a_l, b_l, r_l = 0.0135, 0.0078, 0.003
        lp = []
        for i in range(12):
            an = 2 * pi * i / 12
            lp.append((b_l * cos(an), 0.0, (a_l - b_l) * (1 if sin(an) >= 0 else -1) + b_l * sin(an)))
        link = sweep("link", lp, r_l, closed=True, sides=6, smooth_path=False)
        n_l = int(p.drop)
        z_in = zE - (Re - re_)                    # inner bottom of the staple
        lmats = []
        zc_l = z_in + r_l - a_l
        for i in range(n_l):
            lmats.append(Matrix.Translation((0, -R, zc_l)) @ Matrix.Rotation(radians(0 if i % 2 == 0 else 90), 4, 'Z')
                         @ Matrix.Rotation(radians(r_ir.jitter(0, 5)), 4, 'Y'))
            z_in = zc_l - (a_l - r_l)             # inner bottom of this link
            zc_l = z_in + r_l - a_l
        swing = []
        if lmats:
            ch = instances_to_mesh("links", link, lmats, uv_jitter=False, keep_base=False)
            smooth(ch, 180)
            assign(ch, iron)
            paint(ch, special=0.35 * rust, wear=0.3)
            swing.append(ch)
        else:
            delete(link)
        z_eye = z_in - 0.012                     # the lantern eye bears on the last link (or the staple)
        z_lan = z_eye + 0.016
        nb0 = len(brasses)
        lan, lan_objs = lantern(z_lan, 90 if n_l % 2 else 0)
        swing += lan_objs + brasses[nb0:]
        part(swing, "lantern", pivot=(0, -R, zE - Re), motion="swing", amp=1.6, speed=0.7)
        joints.append((0, -R, z_hook))
        soot = _near((0, -R, z_hook + 0.02), 0.26 + lw * 0.3)
        socket("lantern", "attach", (0, -R, z_lan), role="lantern_mount")
        z_base = z_lan + dzh
    else:
        lan, lan_objs = lantern(z_seat)
        joints.append((0, -R, z_seat - 0.05))
        socket("lantern", "attach", (0, -R, z_seat), role="lantern_mount")
        z_base = z_seat

    # =============================================================== fixings (stamped): proud, bare iron against the paint
    if riv_m:
        rv = bolt("rivet", 0.3 * s, 0.0052, kind="dome")
        fixings.append(instances_to_mesh("rivets", rv, riv_m, keep_base=False))
    if bolt_m:
        washer = cylinder("washer", 0.62 * s, 0.0036, loc=(0, 0, 0.0018), verts=12, bevel=0.001, segments=1)
        hd = bolt("bolt_head", 0.43 * s, 0.0105, loc=(0, 0, 0.0034), kind="square")
        cb = join([washer, hd], "coach_bolt")
        fixings.append(instances_to_mesh("bolts", cb, bolt_m, keep_base=False))
    if nail_m:
        nl = bolt("nail", 0.5 * s, 0.011, kind="pyramid")
        fixings.append(instances_to_mesh("nails", nl, nail_m, keep_base=False))

    # =============================================================== lost paint: rust runs and flakes (bare rusty iron showing)
    if rust > 0.12:
        r_fl = rng.fork("flakes")
        rows_ = _plate_rows(half_h, hw, end)
        pz, pw = [q[0] for q in rows_], [q[1] for q in rows_]
        plate_w = lambda z: float(np.interp(z, pz, pw))
        fv, ff = [], []

        def add(verts, faces):
            b = len(fv)
            fv.extend(verts)
            ff.extend(tuple(b + i for i in f) for f in faces)

        # rust water running down the plate from every bolt, rivet and the arm shoulder: widest under the fixing
        yq = -(TP + 0.0015)
        for (x0, z0_, a_) in runs:
            w0 = (0.5 if a_ > 0.9 else 0.3) * s
            ln = (0.06 + 0.2 * rust) * a_ * r_fl.vary(1.0, 0.2)
            vs, cx, nr = [], x0, 8
            for i in range(nr + 1):
                tq = i / nr
                z = z0_ + 0.004 - ln * tq
                wl = plate_w(z) - 0.004
                if z < -(half_h + cap_len) + 0.004 or wl < 0.004:
                    break
                w = w0 * (1.0 - tq) ** 1.15 * r_fl.vary(1.0, 0.2) + 0.0012
                cx += r_fl.jitter(0.0, 0.002) if i else 0.0
                vs += [(clamp(cx - w, -wl, wl), yq, z), (clamp(cx + w, -wl, wl), yq, z)]
            add(vs, [(2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1) for i in range(len(vs) // 2 - 1)])
        # flakes at the edges of the plate
        for i in range(int(round(1 + 4 * rust))):
            z = r_fl.u(-half_h * 0.85, half_h * 0.85)
            sx = 1 if i % 2 else -1

            def lim_p(u, v, zc=z):
                wl = plate_w(zc + v) - 0.0035
                return clamp(u, -wl - cq[0], wl - cq[0]), v
            cq = (sx * (hw - r_fl.u(0.004, 0.014)), yq, z)
            add(*_flake(cq, (1, 0, 0), (0, 0, 1), r_fl.u(0.008, 0.014), r_fl.u(0.012, 0.026), r_fl, lim_p))
        # flakes on the arm: the top (where water sits), the outer face and the underside seen from the street
        if p.style != "swan":
            hx = arm_s / 2
            e_ = hx - 0.0035
            lo_d, hi_d = (0.08, R - 0.09) if p.style == "scroll" else (0.64 * R + 0.03, R + 0.03)
            spots = [("top", 0.3), ("side", 0.62), ("under", 0.85), ("top", 0.74), ("side", 0.12), ("top", 0.02), ("under", 0.4)]
            for face, q in spots[:int(round(2 + 6 * rust))]:
                d = lerp(lo_d, hi_d, q) + r_fl.jitter(0.0, 0.012)
                al = r_fl.u(0.016, 0.03)
                if p.style == "gallows" and q < 0.2:
                    d = 0.2 * R * r_fl.u(0.35, 0.7)
                    al = 0.014
                lim_a = lambda u, v: (u, clamp(v, -e_, e_))
                if face == "top":
                    add(*_flake((0, -d, zA + hx + 0.0017), (0, -1, 0), (1, 0, 0), al, hx * 1.25, r_fl, lim_a))
                elif face == "under":
                    add(*_flake((0, -d, zA - hx - 0.0017), (0, -1, 0), (-1, 0, 0), al, hx * 1.25, r_fl, lim_a))
                else:
                    add(*_flake((hx + 0.0017, -d, zA), (0, 1, 0), (0, 0, 1), al, hx * 1.25, r_fl, lim_a))
        fl_o = mesh_object("rust_flakes", fv, ff)
        uv_box(fl_o)
        flat(fl_o)
        rusts.append(lod_drop(fl_o, above=1))      # (flakes of a few millimetres: nothing at lod 2)

    # =============================================================== materials and weathering
    irons.append(plate)
    for o in irons:
        assign(o, iron)
    for o in brasses:
        assign(o, brass)
    for o in accents:
        assign(o, brass if p.gilt else iron)
    for o in fixings + rusts:
        assign(o, fix_m)

    jm = _mask_max(*[_near(j, 0.05) for j in joints])
    tm = _mask_max(*[_near(q, 0.03) for q in tips]) if tips else (lambda co, n: np.zeros(len(co)))
    blotch = noise_mask(0.07, seed=p.seed + 2, lo=0.42, hi=0.78)
    fine = noise_mask(0.025, seed=p.seed + 8, lo=0.3, hi=0.8)
    # lantern drip edges: the eaves and the tray rim are where water hangs
    ze, zt_ = z_base + z1 + 0.02, z_base + z0 - 0.004

    def drip(co, n):
        m = np.maximum(np.exp(-((co[:, 2] - ze) / 0.012) ** 2), 0.8 * np.exp(-((co[:, 2] - zt_) / 0.01) ** 2))
        return m * (np.hypot(co[:, 0], co[:, 1] + R) > wb * 0.5)

    def rust_fn(co, n):
        # rust at joints and fixings, in blotches where water sits on up-facing metal, light everywhere when neglected
        up = np.clip(n[:, 2], 0.0, 1.0)
        m = jm(co, n) * (0.55 + 0.45 * fine(co, n)) * 1.3 + blotch(co, n) * 0.6 * _ss(0.3, 0.8, up) + 0.25 * max(rust - 0.5, 0.0)
        m = np.maximum(m, drip(co, n) * (0.35 + 0.65 * fine(co, n)) * 0.9)
        return np.clip(m * (0.2 + rust * 1.5), 0.0, 1.0)

    def streaks(co, n):
        # rust water running down the plate from each bolt, rivet and the arm shoulder, gathering at the bottom tip
        m = np.zeros(len(co))
        for (x, z, a_) in runs:
            dz = z - co[:, 2]
            wd = 0.012 + 0.05 * np.clip(dz, 0.0, 0.2)
            m = np.maximum(m, a_ * np.exp(-((co[:, 0] - x) / wd) ** 2) * np.clip(dz / 0.008, 0.0, 1.0) * np.exp(-np.clip(dz, 0, None) / 0.2))
        m = np.maximum(m, np.clip((-half_h * 0.75 - co[:, 2]) / (half_h * 0.5), 0.0, 1.0) * 0.55)
        return np.clip(m * (0.55 + 0.45 * fine(co, n)) * (0.75 + 1.2 * rust), 0.0, 1.0) * (co[:, 1] > -TP - 0.003)

    tipwear = mask_mul(tm, 0.75)
    for o in irons + [lan]:
        gr = None
        if o is plate:
            gr = mask_mul(noise_mask(0.09, seed=p.seed + 5), 0.35)
        elif o is lan:
            gr = mask_mul(_near((0, -R, z_base + ztop - 0.1), 0.2), 0.8)      # soot round the vent
        elif soot:
            gr = mask_mul(soot, 0.85)
        if forged:
            paint(o, special=_mask_max(rust_fn, streaks) if o is plate else rust_fn, grime=gr, wear=tipwear)
        else:
            sp = [rust_fn, mask_mul(tm, 0.6 * (0.4 + rust))] + ([mask_mul(streaks, 0.6)] if o is plate else [])
            paint(o, special=_mask_max(*sp), grime=gr, wear=tipwear)
    for o in fixings:
        paint(o, special=0.6 + 0.4 * rust, grime=0.15, wear=mask_mul(fine, 0.35))
    for o in rusts:
        paint(o, special=mask_mul(fine, 0.5 + 0.5 * rust))
    for o in brasses + (accents if p.gilt else []):
        paint(o, special=mask_mul(fine, 0.15 + 0.45 * rust), grime=0.2)
    if not p.gilt:
        for o in accents:
            paint(o, special=rust_fn)

    # =============================================================== light: owned by the fitting
    if p.lit and st["light"]:
        lx, ly, lz = st["light"]
        # nothing of the fitting (cage, arm, stay, plate ends) may shadow its own wall: shadows start beyond its farthest point
        far = max(hypot(R, half_h + cap_len - lz), hypot(R, -half_h - cap_len - lz), abs(z_base + ztop - lz), R + 0.1)
        light("lamp", (lx, ly, lz), color="#ffac54", intensity=p.intensity, range=7.5, flicker=0.14 if gas else 0.3,
              shadow_near=round(far + 0.1, 3))
    if tip_pt is not None:
        socket("tip", "attach", tip_pt, role="bracket_tip")
