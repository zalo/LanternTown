"""Octagonal newspaper kiosk in the Parisian manner: dressed-stone kerb, framed-and-panelled painted
timber base between turned corner colonnettes, clear-glazed upper with a serving hatch (cream-lined
reveals, counter with a worn painted nosing on shaped brackets, stacks of papers, magazines pegged on a
rail), display windows with magazine racks seen through the glass, pasted poster panels and a narrow
half-glazed door at the back.  The inside is lined in the cream accent colour and lit by one enamel-shaded
pendant; two small shaded lamps hang from the front corner brackets.  A sheet-metal marquise with a
fretwork valance on scrolled brackets shelters the customers; above it a softly lit frieze carries the
sign lettering, then a dentilled cornice and a lapped zinc / copper roof (onion, tent or bell) with
batten rolls and a finial.
"""
import math
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from tk import *

META = {"name": "Newspaper kiosk", "class": "structure", "mount": "ground", "focus": [0.0, -1.2, 1.62]}
PARAMS = {
    "diameter": {"default": 2.4, "min": 1.9, "max": 3.2, "desc": "width across the flats of the octagon (m)"},
    "wall_height": {"default": 2.05, "min": 2.0, "max": 2.6, "desc": "height of the wall head under the marquise (m)"},
    "roof": {"default": "onion", "choices": ["onion", "tent", "bell"], "desc": "roof style"},
    "roof_height": {"default": 1.0, "min": 0.7, "max": 1.35, "desc": "roof height multiplier"},
    "roof_metal": {"default": "zinc", "choices": ["zinc", "copper"], "desc": "roof and marquise sheet metal"},
    "canopy": {"default": 0.38, "min": 0.2, "max": 0.6, "desc": "projection of the marquise beyond the wall (m)"},
    "valance": {"default": True, "desc": "scalloped fretwork valance under the marquise edge"},
    "paint": {"default": "#2f6b4c", "desc": "paint colour of the exterior joinery"},
    "accent": {"default": "#cdbb8c", "desc": "paint colour of beads, valance, soffit and the interior lining"},
    "sign_text": {"default": "JOURNAUX", "desc": "frieze lettering on the front, back and side faces"},
    "sign_text_alt": {"default": "TABAC", "desc": "frieze lettering on the diagonal faces"},
    "poster_text": {"default": "CIRQUE", "desc": "headline of the large pasted poster"},
    "poster_a": {"default": "#a5382b", "desc": "first print colour of posters and magazine covers"},
    "poster_b": {"default": "#26455a", "desc": "second print colour (also the newspaper ink)"},
    "posters": {"default": 2, "min": 0, "max": 4, "desc": "number of upper faces boarded as poster panels instead of display windows"},
    "stock": {"default": 0.8, "min": 0.0, "max": 1.0, "desc": "how well stocked the racks and counter are"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "paint wear / weathering amount"},
    "corner_lamps": {"default": True, "desc": "shaded lamps hung from the two front corner brackets"},
    "intensity": {"default": 24.0, "min": 5, "max": 120, "desc": "interior pendant intensity (cd)"},
}
VARIANTS = {
    "tent_copper": {"roof": "tent", "roof_metal": "copper", "paint": "#5e1d1a", "accent": "#d9c79a", "diameter": 2.1,
                    "sign_text": "GAZETTE", "sign_text_alt": "NEWS", "posters": 3, "canopy": 0.3, "poster_text": "OPERA", "seed": 7},
    "bell_grand": {"roof": "bell", "roof_metal": "copper", "paint": "#1b2a47", "accent": "#c9a655", "diameter": 3.0,
                   "wall_height": 2.4, "canopy": 0.55, "sign_text": "DAILY NEWS", "sign_text_alt": "BOOKS", "posters": 1,
                   "stock": 1.0, "wear": 0.2, "seed": 13},
    "small_worn": {"roof": "tent", "roof_metal": "zinc", "paint": "#6f7d6a", "accent": "#b9ad8e", "diameter": 1.9,
                   "wall_height": 2.0, "roof_height": 0.8, "canopy": 0.22, "valance": False, "sign_text": "PRESSE",
                   "sign_text_alt": "CARTES", "posters": 4, "stock": 0.35, "wear": 0.95, "intensity": 19, "seed": 21},
    "onion_black": {"roof": "onion", "roof_metal": "copper", "roof_height": 1.2, "paint": "#1c1b1a", "accent": "#b88a3a",
                    "diameter": 2.6, "sign_text": "KIOSK", "sign_text_alt": "CIGARS", "posters": 2, "poster_a": "#b0812a",
                    "poster_b": "#3a2a4a", "poster_text": "VARIETE", "seed": 34},
}

C22 = math.cos(math.radians(22.5))
T22 = math.tan(math.radians(22.5))
Q45 = math.pi / 4.0
ROOFS = {
    "onion": (0.45, [(1.0, 0), (0.93, 0.012), (0.875, 0.05), (0.885, 0.10), (0.925, 0.18), (0.945, 0.27), (0.92, 0.37),
                     (0.84, 0.48), (0.70, 0.59), (0.52, 0.69), (0.35, 0.77), (0.215, 0.84), (0.125, 0.91), (0.08, 1.0)]),
    "tent": (0.58, [(1.0, 0), (0.88, 0.035), (0.73, 0.10), (0.58, 0.19), (0.45, 0.30), (0.335, 0.43), (0.24, 0.57),
                    (0.165, 0.72), (0.11, 0.87), (0.075, 1.0)]),
    "bell": (0.46, [(1.0, 0), (0.985, 0.07), (0.95, 0.17), (0.88, 0.29), (0.76, 0.42), (0.59, 0.54), (0.42, 0.63),
                    (0.29, 0.71), (0.19, 0.80), (0.12, 0.90), (0.08, 1.0)]),
}


def _oct(name, prof, **kw):
    """Octagonal 'lathe': profile given as (apothem, z); flats face +-X / +-Y."""
    return lathe(name, [(r / C22, z) for r, z in prof], segments=8, rot=(0, 0, 22.5), smooth_angle=20.0, **kw)


def _oct_open(name, prof):
    """Same, but leaving out the back face (door)."""
    return lathe(name, [(r / C22, z) for r, z in prof], segments=7, rot=(0, 0, 112.5), angle=315.0, smooth_angle=20.0)


def _rz(k):
    return Matrix.Rotation(math.radians(45.0 * k), 4, 'Z')


def _face(objs, k):
    """Move objects modelled in the front-face frame (wall plane y = -apothem) to face k (k * 45 deg)."""
    lst = objs if isinstance(objs, (list, tuple)) else [objs]
    for o in lst:
        apply_transform(o)
        place(o, rot=(0, 0, 45.0 * k))
    return objs


def _bar(name, size, loc, grain='X', bv=0.004, seg=2):
    o = box(name, size, loc=loc, bevel=bv, segments=seg)
    uv_grain(o, grain)
    if seg > 1:
        smooth(o, 50)
    return o


def _rect(cx, cy, w, h):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def _quad_y(name, cx, cz, w, h, y):
    """Single quad in the XZ plane facing -Y (glass panes, lit sign panels)."""
    return mesh_object(name, [(cx - w / 2, y, cz - h / 2), (cx + w / 2, y, cz - h / 2), (cx + w / 2, y, cz + h / 2),
                              (cx - w / 2, y, cz + h / 2)], [(0, 1, 2, 3)])


class _Decals:
    """Printed areas as single quads 1 mm proud of the sheet, batched into one mesh per ink."""

    def __init__(self):
        self.q = {}

    def z(self, m, cx, cy, w, h, z):                     # lying in XY, facing +Z
        self.q.setdefault(m, []).append([(cx - w / 2, cy - h / 2, z), (cx + w / 2, cy - h / 2, z),
                                         (cx + w / 2, cy + h / 2, z), (cx - w / 2, cy + h / 2, z)])

    def y(self, m, cx, cz, w, h, y):                     # standing in XZ, facing -Y
        self.q.setdefault(m, []).append([(cx - w / 2, y, cz - h / 2), (cx + w / 2, y, cz - h / 2),
                                         (cx + w / 2, y, cz + h / 2), (cx - w / 2, y, cz + h / 2)])

    def objs(self, name, mats):
        out = []
        for key, quads in self.q.items():
            verts = [v for q in quads for v in q]
            faces = [(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3) for i in range(len(quads))]
            o = mesh_object(name + "_" + key, verts, faces)
            assign(o, mats[key])
            out.append(o)
        return out


def _fit_text(name, body, max_w, max_h, depth, font="serif", front_only=False):
    """Lettering centred on the origin in XZ, back on y = 0, facing -Y, scaled to fit max_w x max_h."""
    t = text(name, body, size=max_h, depth=depth, font=font)
    apply_transform(t)
    co = np.array([v.co[:] for v in t.data.vertices])
    if len(co) == 0:
        delete([t])
        return None
    mn, mx = co.min(axis=0), co.max(axis=0)
    w, h = max(mx[0] - mn[0], 1e-4), max(mx[2] - mn[2], 1e-4)
    s = min(max_w / w, max_h / h)
    cx, cz = (mn[0] + mx[0]) / 2, (mn[2] + mx[2]) / 2
    transform_verts(t, lambda c: Vector(((c.x - cx) * s, c.y, (c.z - cz) * s)))
    # the back of the lettering lies on its ground: drop those faces
    bm = bmesh.new()
    bm.from_mesh(t.data)
    back = [f for f in bm.faces if f.normal.y > (-0.5 if front_only else 0.5)]
    bmesh.ops.delete(bm, geom=back, context='FACES')
    bm.to_mesh(t.data)
    bm.free()
    return t


def _oct_co(co):
    """World coords -> (face index, apothem distance, lateral coordinate in the face) for octagonal sheet-metal masks."""
    ang = np.arctan2(co[:, 0], -co[:, 1])
    kk = np.round(ang / Q45)
    da = ang - kk * Q45
    rad = np.hypot(co[:, 0], co[:, 1])
    return kk, rad * np.cos(da), rad * np.sin(da)


def build(p, rng):
    D = p.diameter
    apo = D / 2.0
    Rc = apo / C22
    fw = 2 * apo * T22                       # width of one face
    T = 0.05                                 # joinery thickness
    zp = 0.10                                # top of the stone kerb
    zb = 0.94                                # top of the panelled base
    zs = zb + 0.045                          # top of the sill
    zh = p.wall_height
    zo1 = zh - 0.08                          # head of the openings
    cp = p.canopy
    af = apo - 0.02                          # frieze plane
    zf0 = zh + 0.17
    zf1 = zf0 + 0.30
    CS = 0.85                                # cornice height scale
    ze = zf1 + 0.13 * CS                     # eaves
    ae = af + 0.115
    stock = p.stock
    w_ = p.wear
    sd = p.seed
    zinc = p.roof_metal == "zinc"

    m_main = mat("wood_painted", p.paint, wear=1.0 + 0.8 * w_, grime=1.5 + 0.5 * w_)
    m_acc = mat("wood_painted", p.accent, wear=1.0 + 0.5 * w_)
    # zinc sheet as it was kept on kiosks: coated matte light grey (bare-metal shaders read near-black in shade)
    m_roof = mat("iron_painted", "#b3b9bb", scale=1.5, rough=2.2, wear=0.45) if zinc else mat("copper_patina", scale=2.4)
    m_brass = mat("brass")
    m_glass = mat("glass_clear", "#f4f1e8", opacity=0.12)
    m_glow = mat("glass_frosted", "#f6b25e", glow=0.85)
    m_stone = mat("stone_smooth", "#a89d8a")
    m_paper = mat("paper", "#e7dec6")
    m_ia = mat("paper", p.poster_a)
    m_ib = mat("paper", p.poster_b)
    inks = {"p": m_paper, "a": m_ia, "b": m_ib}

    main, acc, brass, glass, glow, metal = [], [], [], [], [], []
    lined_objs = []                          # joinery already carrying paint outside / lining inside

    def lined(objs, ya):
        """Exterior paint on the object, cream lining on the faces that look into the kiosk."""
        for o in objs:
            apply_transform(o)
            assign(o, m_main)
            assign_faces(o, m_acc, lambda c, n: n.y > 0.6 and c.y > ya + 0.032)
        lined_objs.extend(objs)

    kinds = {0: "hatch", 4: "door", 1: "window", 7: "window"}
    for i, k in enumerate([3, 5, 2, 6]):
        kinds[k] = "poster" if i < int(p.posters) else "window"

    # ------------------------------------------------------------------ stone kerb, step, floor, ceiling
    kerb = _oct("kerb", [(0, 0), (apo + 0.10, 0), (apo + 0.10, 0.052), (apo + 0.088, 0.068), (apo + 0.055, 0.074),
                         (apo + 0.04, zp), (0, zp)])
    densify(kerb, 0.3)
    displace_noise(kerb, 0.003, 0.25, seed=p.seed, axis_mask=(1, 1, 0))
    assign(kerb, m_stone)
    step = stone_block("step", (min(fw, 1.05) * 0.95, 0.36, 0.10), loc=(0, -(apo + 0.10 + 0.17), 0.056), rng=rng.fork("step"),
                       chamfer=0.012, rough=0.003)
    assign(step, m_stone)
    _face(step, 4)
    floor = cylinder("floor", radius=(apo - 0.01) / C22, depth=0.03, loc=(0, 0, zp + 0.015), rot=(0, 0, 22.5), verts=8)
    assign(floor, m_stone)
    for o in (kerb, step, floor):
        paint(o, grime=mask_mul(height_ramp(0.0, 0.16), 0.8),
              special=mask_mul(facing((0, 1, 0)), noise_mask(0.3, seed=p.seed + 1), 0.8),
              wear=mask_mul(facing((0, 0, 1)), 0.35))
    ceil_ = cylinder("ceiling", radius=(apo - T + 0.004) / C22, depth=0.03, loc=(0, 0, zh + 0.05), rot=(0, 0, 22.5), verts=8)
    acc.append(ceil_)

    # ------------------------------------------------------------------ continuous octagonal mouldings
    main.append(_oct_open("skirting", [(apo - 0.005, zp), (apo + 0.03, zp), (apo + 0.03, zp + 0.085), (apo + 0.02, zp + 0.105),
                                       (apo + 0.008, zp + 0.112), (apo - 0.005, zp + 0.112), (apo - 0.005, zp)]))
    main.append(_oct_open("sill", [(apo - 0.01, zb), (apo + 0.03, zb), (apo + 0.04, zb + 0.008), (apo + 0.058, zb + 0.014),
                                   (apo + 0.066, zb + 0.022), (apo + 0.066, zs - 0.006), (apo + 0.06, zs), (apo - 0.01, zs),
                                   (apo - 0.01, zb)]))
    head = _oct("head", [(apo - T, zo1), (apo, zo1), (apo, zh - 0.035), (apo + 0.012, zh - 0.025), (apo + 0.012, zh),
                         (apo + 0.022, zh + 0.012), (apo + 0.04, zh + 0.03), (apo + 0.048, zh + 0.05), (apo + 0.048, zh + 0.088),
                         (apo - T, zh + 0.088), (apo - T, zo1)])
    assign(head, m_main)
    assign_faces(head, m_acc, lambda c, n: n.z < -0.6 or (c.x * n.x + c.y * n.y) < -0.5 * math.hypot(c.x, c.y))
    lined_objs.append(head)
    # marquise: closed sheet-metal section, soffit boarded and painted
    can = _oct("marquise", [(apo - 0.035, zh + 0.215), (apo + cp - 0.03, zh + 0.108), (apo + cp - 0.006, zh + 0.098),
                            (apo + cp, zh + 0.09), (apo + cp, zh + 0.066), (apo + cp - 0.014, zh + 0.06),
                            (apo + 0.03, zh + 0.086), (apo - 0.035, zh + 0.086), (apo - 0.035, zh + 0.215)])
    densify(can, 0.19)
    assign(can, m_roof)
    assign_faces(can, m_acc, lambda c, n: n.z < -0.5)
    # frieze drum backing and upper cornice
    main.append(_oct("drum", [(af - 0.02, zf0 - 0.02), (af - 0.02, zf1)]))
    main.append(_oct("frieze_base", [(af - 0.02, zf0 - 0.03), (af + 0.03, zf0 - 0.03), (af + 0.03, zf0 + 0.012), (af + 0.018, zf0 + 0.022),
                                     (af - 0.02, zf0 + 0.022)]))

    def cz_(dz):
        return zf1 + dz * CS

    main.append(_oct("cornice", [(af - 0.03, zf1 - 0.012), (af + 0.014, zf1 - 0.012), (af + 0.014, cz_(0.012)), (af + 0.022, cz_(0.02)),
                                 (af + 0.022, cz_(0.066)), (af + 0.056, cz_(0.07)), (af + 0.072, cz_(0.08)), (af + 0.084, cz_(0.096)),
                                 (af + 0.104, cz_(0.102)), (af + 0.115, cz_(0.112)), (ae, ze), (ae - 0.03, ze + 0.004), (0, ze + 0.004)]))
    dent = box("dentil", (0.034, 0.032, 0.04 * CS))
    dm = []
    wd = 2 * af * T22 - 0.07
    nd = max(3, int(round(wd / 0.078)))
    rd = rng.fork("dentils")
    for k in range(8):
        for i in range(nd):
            if rd.chance(0.12 * max(0.0, w_ - 0.5)):
                continue                                 # the odd dentil has dropped off an old kiosk
            x = (i - (nd - 1) / 2) * (wd / nd)
            dm.append(_rz(k) @ trs((x, -(af + 0.037), cz_(0.044))))
    main.append(instances_to_mesh("dentils", dent, dm, keep_base=False))

    # ------------------------------------------------------------------ corner colonnettes (turned) and frieze pilasters
    cz1 = zo1
    sh0, sh1 = zp + 0.15, zb - 0.045
    uh0, uh1 = zs + 0.085, cz1 - 0.17
    cprof = [(0.064, zp), (0.064, zp + 0.11), (0.054, zp + 0.125), (0.044, zp + 0.15)]
    cprof += [(0.044, sh0 + (sh1 - sh0) * t) for t in (0.1, 0.25, 0.5, 0.75)]
    cprof += [(0.044, zb - 0.045), (0.05, zb - 0.035), (0.05, zb - 0.02), (0.058, zb), (0.06, zs), (0.05, zs + 0.016), (0.04, zs + 0.03),
              (0.046, zs + 0.045), (0.04, zs + 0.06), (0.034, zs + 0.085)]
    cprof += [(0.034 - 0.003 * t, uh0 + (uh1 - uh0) * t) for t in (0.25, 0.5, 0.75)]
    cprof += [(0.031, cz1 - 0.17), (0.04, cz1 - 0.16), (0.042, cz1 - 0.145), (0.034, cz1 - 0.13), (0.034, cz1 - 0.07),
              (0.044, cz1 - 0.045), (0.056, cz1 - 0.02), (0.058, cz1), (0.05, cz1 + 0.01), (0.05, zh - 0.03)]
    col = lathe("colonnette", cprof, segments=8)
    cm = []
    for k in range(8):
        cm.append(_rz(k + 0.5) @ trs((0, -(Rc - 0.012), 0)))
    colonnettes = instances_to_mesh("colonnettes", col, cm, keep_base=False)
    main.append(colonnettes)
    pil = cylinder("pilaster", radius=0.042, depth=zf1 - zf0 + 0.02, verts=8)
    main.append(instances_to_mesh("pilasters", pil, [_rz(k + 0.5) @ trs((0, -(af / C22 - 0.012), (zf0 + zf1) / 2)) for k in range(8)],
                                  keep_base=False))

    # ------------------------------------------------------------------ printed matter (instanced bases)
    def news_base(i):
        """Folded newspaper lying flat (masthead towards +Y); four front-page layouts, each with its own curled corner."""
        sheet = box("np", (0.25, 0.33, 0.012), loc=(0, 0, 0.006))
        densify(sheet, 0.12)
        assign(sheet, m_paper)
        d = _Decals()
        zt_ = 0.0128

        def L(m, cx, cy, w, h, lift=0.0):
            d.z(m, cx, cy, w, h, zt_ + lift)

        def col(cx, y0, w, n, pitch=0.0185):
            for j in range(n):
                last = j == n - 1
                L("b", cx - (w * 0.16 if last else 0.0), y0 - j * pitch, w * (0.68 if last else 1.0), 0.008)

        def photo(cx, cy, w, h):
            L("b", cx, cy, w, h)
            L("p", cx - w * 0.06, cy + h * 0.2, w * 0.74, h * 0.3, 0.0005)

        if i == 0:                                          # classic: full masthead, photo left, three columns
            L("b", 0, 0.128, 0.2, 0.03)
            L("b", 0, 0.106, 0.216, 0.003)
            photo(-0.056, 0.052, 0.104, 0.086)
            L("b", 0.056, 0.088, 0.1, 0.011)
            L("b", 0.045, 0.072, 0.078, 0.011)
            col(0.056, 0.054, 0.1, 3)
            L("b", 0, -0.006, 0.216, 0.002)
            for cx in (-0.073, 0.0, 0.073):
                col(cx, -0.022, 0.064, 7)
        elif i == 1:                                        # centred red masthead with ears, banner, photo right
            L("a", 0, 0.13, 0.13, 0.028)
            for sx in (-1, 1):
                L("b", sx * 0.09, 0.13, 0.03, 0.028)
            L("b", 0, 0.109, 0.216, 0.002)
            L("b", 0, 0.104, 0.216, 0.002)
            L("b", 0, 0.086, 0.21, 0.014)
            photo(0.052, 0.012, 0.11, 0.11)
            for cx in (-0.085, -0.035):
                col(cx, 0.066, 0.045, 6)
            for cx in (-0.084, -0.028, 0.028, 0.084):
                col(cx, -0.066, 0.048, 5)
        elif i == 2:                                        # masthead left with a date box, wide picture across
            L("b", -0.03, 0.128, 0.15, 0.034)
            L("a", 0.082, 0.128, 0.05, 0.034)
            L("b", 0, 0.105, 0.216, 0.002)
            photo(0, 0.055, 0.214, 0.08)
            L("b", -0.03, 0.006, 0.15, 0.005)
            L("b", -0.036, -0.012, 0.14, 0.012)
            for cx in (-0.073, 0.0):
                col(cx, -0.03, 0.064, 7)
            col(0.073, -0.008, 0.064, 8)
        else:                                               # text heavy: thin masthead, two-line headline, five columns
            L("b", 0, 0.136, 0.214, 0.02)
            L("b", 0, 0.108, 0.2, 0.016)
            L("b", -0.02, 0.086, 0.16, 0.016)
            L("b", 0, 0.07, 0.216, 0.002)
            for j, cx in enumerate((-0.088, -0.044, 0.0, 0.044, 0.088)):
                col(cx, 0.058, 0.037, 6 if j < 2 else (11 if j == 2 else 8))
            photo(-0.066, -0.095, 0.082, 0.07)
            L("a", 0.066, -0.118, 0.082, 0.045)
        o = join([sheet] + d.objs("np%d" % i, inks), "newspaper%d" % i)
        sx, sy, amt = [(1, 1, 0.03), (-1, 1, 0.02), (1, -1, 0.016), (-1, 1, 0.036)][i]

        def curl(c):
            t = clamp((sx * c.x + 0.02) / 0.145) * clamp((sy * c.y + 0.02) / 0.185)
            return Vector((c.x - sx * 0.25 * amt * t * t, c.y - sy * 0.2 * amt * t * t, c.z + amt * t * t))

        transform_verts(o, curl)
        return o

    def mag_base(i):
        """Magazine standing on y = 0 facing -Y: cover with a title band, an image block and a line or two of type."""
        bg = ["p", "a", "b", "p", "a"][i]
        cov = box("mg", (0.2, 0.005, 0.27), loc=(0, -0.0025, 0))
        assign(cov, inks[bg])
        d = _Decals()
        y1, y2 = -0.0058, -0.0066
        if i == 0:
            d.y("a", 0, 0.098, 0.172, 0.046, y1)
            d.y("p", -0.02, 0.098, 0.1, 0.014, y2)
            d.y("b", 0, -0.02, 0.15, 0.135, y1)
            d.y("p", 0.03, 0.005, 0.055, 0.055, y2)
            d.y("a", -0.03, -0.055, 0.07, 0.05, y2)
            d.y("b", -0.015, -0.108, 0.12, 0.009, y1)
        elif i == 1:
            d.y("p", 0, 0.1, 0.16, 0.04, y1)
            d.y("b", 0, 0.1, 0.12, 0.014, y2)
            d.y("p", 0, -0.03, 0.156, 0.15, y1)
            d.y("b", 0, -0.02, 0.13, 0.1, y2)
            d.y("b", -0.02, -0.092, 0.09, 0.008, y2)
        elif i == 2:
            for sx in (-1, 1):
                d.y("p", sx * 0.09, 0, 0.005, 0.25, y1)
                d.y("p", 0, sx * 0.1235, 0.185, 0.005, y1)
            d.y("a", 0, 0.085, 0.15, 0.04, y1)
            d.y("p", 0, 0.085, 0.1, 0.012, y2)
            d.y("p", 0, -0.015, 0.13, 0.11, y1)
            d.y("a", 0.02, -0.03, 0.06, 0.06, y2)
            d.y("p", 0, -0.098, 0.11, 0.008, y1)
        elif i == 3:
            d.y("b", 0, -0.045, 0.2, 0.18, y1)
            d.y("a", 0, 0.105, 0.2, 0.05, y1)
            d.y("p", -0.02, 0.105, 0.12, 0.016, y2)
            d.y("p", -0.02, -0.03, 0.09, 0.1, y2)
            d.y("a", 0.068, -0.108, 0.036, 0.036, y2)
            d.y("p", -0.03, 0.062, 0.12, 0.007, y1)
        else:
            d.y("b", -0.0775, 0, 0.045, 0.27, y1)
            d.y("p", 0.022, 0.098, 0.13, 0.036, y1)
            d.y("p", 0.022, -0.005, 0.13, 0.125, y1)
            d.y("b", 0.022, 0.005, 0.1, 0.075, y2)
            d.y("p", 0.01, -0.092, 0.105, 0.008, y1)
            d.y("p", 0.0, -0.108, 0.085, 0.008, y1)
        return join([cov] + d.objs("mg%d" % i, inks), "magazine%d" % i)

    NN, NM = 4, 5
    news = [news_base(i) for i in range(NN)]
    news_m = [[] for _ in range(NN)]
    mags = [mag_base(i) for i in range(NM)]
    mags_m = [[] for _ in range(NM)]
    bundle = box("bundle", (0.25, 0.33, 0.03), loc=(0, 0, 0.015), bevel=0.005, segments=1)
    assign(bundle, m_paper)
    bundle_m = []

    def add_news(r, M):
        news_m[int(r.u(0, NN - 0.001))].append(M)

    def add_mag(r, M):
        mags_m[int(r.u(0, NM - 0.001))].append(M)

    def add_stack(r, M0, n):
        for i in range(n):
            bundle_m.append(M0 @ trs((r.jitter(0, 0.006), r.jitter(0, 0.006), i * 0.03), (0, 0, r.jitter(0, 3.5))))
        add_news(r, M0 @ trs((r.jitter(0, 0.008), r.jitter(0, 0.008), n * 0.03), (0, 0, r.jitter(0, 7))))

    # posters (built facing -Y on y = 0, centred), stamped with a slight tilt
    def poster_big():
        w, h = 0.5, 0.74
        parts = []
        sh = box("po", (w, 0.0016, h), loc=(0, -0.0008, 0))
        assign(sh, m_paper)
        parts.append(sh)
        bd = box("po_band", (w - 0.05, 0.001, 0.15), loc=(0, -0.002, h / 2 - 0.105))
        assign(bd, m_ia)
        parts.append(bd)
        t = _fit_text("po_title", p.poster_text, w - 0.1, 0.095, 0.0012, font="serif") if p.poster_text.strip() else None
        if t is not None:
            place(t, loc=(0, -0.0025, h / 2 - 0.105))
            assign(t, m_paper)
            parts.append(t)
        im = box("po_img", (w - 0.08, 0.001, 0.3), loc=(0, -0.002, -0.02))
        assign(im, m_ib)
        parts.append(im)
        moon = cylinder("po_moon", radius=0.075, depth=0.001, loc=(0.09, -0.0028, 0.03), rot=(90, 0, 0), verts=20)
        assign(moon, m_paper)
        parts.append(moon)
        tent = extrude_profile("po_tent", [(-0.16, 0), (0.06, 0), (-0.05, 0.17)], 0.001, loc=(-0.02, -0.0025, -0.165), rot=(90, 0, 0))
        assign(tent, m_ia)
        parts.append(tent)
        for j in range(3):
            st = box("po_t", ((w - 0.12) * (1.0 - 0.18 * j), 0.001, 0.022), loc=(0, -0.002, -0.215 - j * 0.045))
            assign(st, m_ib if j != 1 else m_ia)
            parts.append(st)
        return join(parts, "poster_big"), w, h

    def poster_small(i):
        w, h = (0.34, 0.48) if i == 0 else (0.28, 0.4)
        parts = []
        sh = box("ps", (w, 0.0016, h), loc=(0, -0.0008, 0))
        assign(sh, m_ia if i == 0 else m_paper)
        parts.append(sh)
        if i == 0:
            fr = shape_extrude("ps_fr", _rect(0, 0, w - 0.04, h - 0.04), [_rect(0, 0, w - 0.06, h - 0.06)], depth=0.001,
                               loc=(0, -0.0016, 0), rot=(90, 0, 0))
            assign(fr, m_paper)
            parts.append(fr)
            dsk = cylinder("ps_d", radius=0.085, depth=0.001, loc=(0, -0.0021, 0.06), rot=(90, 0, 0), verts=20)
            assign(dsk, m_paper)
            parts.append(dsk)
            for j in range(3):
                st = box("ps_t", (w - 0.12 - 0.04 * j, 0.001, 0.02), loc=(0, -0.0021, -0.09 - j * 0.04))
                assign(st, m_paper)
                parts.append(st)
        else:
            hd = box("ps_h", (w - 0.05, 0.001, 0.07), loc=(0, -0.0021, h / 2 - 0.065))
            assign(hd, m_ib)
            parts.append(hd)
            for j in range(7):
                st = box("ps_t", ((w - 0.07) * (0.6 if j == 6 else 1.0), 0.001, 0.012), loc=(0, -0.0021, h / 2 - 0.14 - j * 0.032))
                assign(st, m_ib)
                parts.append(st)
            sq = box("ps_s", (0.07, 0.001, 0.05), loc=(w / 2 - 0.07, -0.0021, -h / 2 + 0.05))
            assign(sq, m_ia)
            parts.append(sq)
        return join(parts, "poster_small%d" % i), w, h

    pbig, pbw, pbh = poster_big()
    psm = [poster_small(0), poster_small(1)]
    pbig_m, psm_m = [], [[], []]

    # ------------------------------------------------------------------ panelled base (all faces but the door)
    def panel_set(o, x0, x1, z0, z1, yface, r):
        """Recessed field + raised panel + bead moulding filling the opening x0..x1, z0..z1."""
        w, h = x1 - x0, z1 - z0
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        fld = box("field", (w + 0.02, 0.018, h + 0.02), loc=(cx, yface + 0.03, cz))
        uv_grain(fld, 'Z')
        o.append(fld)
        rp = box("raised", (w - 0.085, 0.022, h - 0.085), loc=(cx, yface + 0.019, cz), bevel=0.012, segments=1)
        densify(rp, 0.17)
        uv_grain(rp, 'Z')
        o.append(rp)
        bead = shape_extrude("bead", _rect(0, 0, w, h), [_rect(0, 0, w - 0.034, h - 0.034)], depth=0.013,
                             loc=(cx, yface + 0.021, cz), rot=(90, 0, 0), bevel=0.003)
        return bead

    for k in range(8):
        r = rng.fork("base%d" % k)
        o, a_ = [], []
        if k == 4:
            continue
        ya = -apo
        for sx in (-1, 1):
            st = _bar("stile", (0.10, T, zb - zp), (sx * (fw / 2 - 0.05), ya + T / 2, (zp + zb) / 2), 'Z', seg=1)
            densify(st, 0.15)
            o.append(st)
        o.append(_bar("rail_b", (fw - 0.2, T, 0.16), (0, ya + T / 2, zp + 0.08), 'X', seg=1))
        o.append(_bar("rail_t", (fw - 0.2, T, 0.09), (0, ya + T / 2, zb - 0.045), 'X', seg=1))
        pw = fw - 0.2
        z0, z1 = zp + 0.16, zb - 0.09
        if pw > 0.9:
            o.append(_bar("muntin", (0.08, T, z1 - z0), (0, ya + T / 2, (z0 + z1) / 2), 'Z', seg=1))
            spans = [(-pw / 2, -0.04), (0.04, pw / 2)]
        else:
            spans = [(-pw / 2, pw / 2)]
        for (x0, x1) in spans:
            a_.append(panel_set(o, x0, x1, z0, z1, ya, r))
        # display ledges with papers leaning on the base of the window faces
        if kinds[k] == "window" and k in (1, 7, 2, 6):
            for j, zl in enumerate((zp + 0.13, zp + 0.5)):
                o.append(_bar("ledge", (pw + 0.06, 0.085, 0.022), (0, ya - 0.04, zl), 'X', bv=0.003, seg=1))
                rod = cylinder("ledge_rod", 0.004, pw + 0.04, loc=(0, ya - 0.078, zl + 0.07), rot=(0, 90, 0), verts=6)
                assign(rod, m_brass)
                _face(rod, k)
                for sx in (-1, 1):
                    stp = cylinder("ledge_post", 0.004, 0.07, loc=(sx * (pw / 2 + 0.018), ya - 0.078, zl + 0.04), verts=6)
                    assign(stp, m_brass)
                    _face(stp, k)
                nn = max(1, int((pw + 0.04) / 0.27))
                for i in range(nn):
                    if not r.chance(0.25 + 0.75 * stock):
                        continue
                    x = (i - (nn - 1) / 2) * 0.27 + r.jitter(0, 0.01)
                    lean = 77 + r.jitter(0, 3.0)
                    cz = zl + 0.011 + 0.165 * math.sin(math.radians(lean))
                    cy = ya - 0.066 + 0.165 * math.cos(math.radians(lean))
                    add_news(r, _rz(k) @ trs((x, cy, cz), (lean, 0, 0)) @ trs((0, 0, 0), (0, 0, r.jitter(0, 3.5))))
        elif k != 0 and r.chance(0.55):
            # a pasted bill on the base panel
            i = int(r.u(0, 1.999))
            _, sw, shh = psm[i]
            x = r.jitter(0, max(0.0, (spans[0][1] - spans[0][0] - 0.085 - sw) / 2)) + (spans[0][0] + spans[0][1]) / 2
            psm_m[i].append(_rz(k) @ trs((x, ya + 0.0075, (z0 + z1) / 2 + r.jitter(0, 0.05)), (0, r.jitter(0, 4), 0),
                                         min(1.0, (spans[0][1] - spans[0][0] - 0.1) / sw)))
        lined(o, ya)
        _face(o, k)
        _face(a_, k)
        acc.extend(a_)

    # ------------------------------------------------------------------ upper storey
    oz0, oz1 = zs, zo1
    oh = oz1 - oz0
    ow = fw - 0.16
    ozc = (oz0 + oz1) / 2
    counter = nosing = None
    door_knob = None
    for k in range(8):
        r = rng.fork("upper%d" % k)
        kind = kinds[k]
        ya = -apo
        o, a_, g_, b_ = [], [], [], []
        if kind != "door":
            for sx in (-1, 1):
                st = _bar("ustile", (0.08, T, oh), (sx * (fw / 2 - 0.04), ya + T / 2, ozc), 'Z', seg=1)
                densify(st, 0.4)
                o.append(st)
        if kind == "window":
            m = 0.045
            bw = 0.026
            trh = min(0.26, oh * 0.24)
            ncl = 2 if ow < 0.95 else 3
            holes = []
            lw = (ow - 2 * m - bw * (ncl - 1)) / ncl
            lh = oh - 2 * m - bw - trh
            for i in range(ncl):
                holes.append(_rect(-ow / 2 + m + lw / 2 + i * (lw + bw), -oh / 2 + m + lh / 2, lw, lh))
            nt = ncl + 1
            tw = (ow - 2 * m - bw * (nt - 1)) / nt
            for i in range(nt):
                holes.append(_rect(-ow / 2 + m + tw / 2 + i * (tw + bw), oh / 2 - m - trh / 2, tw, trh))
            sash = shape_extrude("sash", _rect(0, 0, ow, oh), holes, depth=0.034, loc=(0, ya + 0.044, ozc), rot=(90, 0, 0), bevel=0.003)
            uv_grain(sash, 'X')
            o.append(sash)
            g_.append(_quad_y("glass", 0, ozc, ow - 0.05, oh - 0.05, ya + 0.027))
            # magazine racks behind the glass
            zc = oz0 + 0.075 + 0.14
            ncol = max(1, int((ow - 0.05) / 0.215))
            while zc + 0.15 < oz1 - 0.04:
                a_.append(box("rack", (ow - 0.02, 0.035, 0.014), loc=(0, ya + 0.07, zc - 0.143)))
                rr_ = cylinder("rack_rod", 0.0035, ow - 0.02, loc=(0, ya + 0.052, zc - 0.07), rot=(0, 90, 0), verts=5)
                b_.append(rr_)
                for i in range(ncol):
                    if r.chance(0.2 + 0.8 * stock):
                        x = (i - (ncol - 1) / 2) * 0.215 + r.jitter(0, 0.006)
                        add_mag(r, _rz(k) @ trs((x, ya + 0.068, zc), (-r.u(2, 7), r.jitter(0, 3.0), 0)))
                zc += 0.31
        elif kind == "poster":
            brd = box("board", (ow + 0.01, 0.02, oh + 0.01), loc=(0, ya + 0.025, ozc))
            densify(brd, 0.35)
            uv_grain(brd, 'Z')
            o.append(brd)
            fr = shape_extrude("pframe", _rect(0, 0, ow, oh), [_rect(0, 0, ow - 0.06, oh - 0.06)], depth=0.016,
                               loc=(0, ya + 0.016, ozc), rot=(90, 0, 0), bevel=0.004)
            a_.append(fr)
            ypo = ya + 0.0145
            sc = min(1.0, (ow - 0.1) / pbw, (oh - 0.1) / pbh)
            px = r.jitter(0, max(0.0, (ow - 0.08 - pbw * sc) / 2))
            pz = ozc + r.jitter(0.05, 0.04) if oh - 0.08 - pbh * sc > 0.2 else ozc
            pbig_m.append(_rz(k) @ trs((px, ypo, pz), (0, r.jitter(0, 1.5), 0), sc))
            # a smaller bill pasted over a lower corner
            if r.chance(0.7):
                i = int(r.u(0, 1.999))
                _, sw, shh = psm[i]
                s2 = min(0.9, (ow - 0.1) / sw * 0.6)
                sx = r.pick((-1, 1))
                psm_m[i].append(_rz(k) @ trs((sx * (ow / 2 - 0.05 - sw * s2 / 2), ypo - 0.0025, oz0 + 0.06 + shh * s2 / 2 + r.u(0, 0.05)),
                                             (0, r.jitter(0, 5), 0), s2))
            # interior shelves with magazines, seen through the hatch and the windows
            for j in range(3):
                zc = oz0 + 0.2 + j * 0.31
                if zc + 0.15 > oz1:
                    break
                a_.append(box("ishelf", (ow, 0.05, 0.016), loc=(0, ya + 0.07, zc - 0.145)))
                ncol = max(1, int(ow / 0.215))
                for i in range(ncol):
                    if r.chance(0.2 + 0.8 * stock):
                        x = (i - (ncol - 1) / 2) * 0.215 + r.jitter(0, 0.004)
                        add_mag(r, _rz(k) @ trs((x, ya + 0.052, zc), (r.u(2, 6), r.jitter(0, 2.5), 180)))
        elif kind == "hatch":
            # lined reveals of the opening
            for sx in (-1, 1):
                a_.append(box("reveal", (0.007, T + 0.008, oh), loc=(sx * (ow / 2 - 0.0035), ya + T / 2, ozc)))
            a_.append(box("reveal_head", (ow, T + 0.008, 0.007), loc=(0, ya + T / 2, oz1 - 0.0035)))
            # counter board (lining colour, worn) with a painted nosing, on shaped brackets
            cw = ow + 0.14
            cout, cin = 0.34, 0.2
            yf = ya - cout
            counter = rounded_box("counter", (cw, cout + cin + T, 0.04), radius=0.03, loc=(0, ya + (T + cin - cout) / 2, zs + 0.016),
                                  bevel=0.004)
            densify(counter, 0.12)
            uv_grain(counter, 'X')
            a_.append(counter)
            ctop = zs + 0.036
            rn_ = 0.034
            npath = [(-cw / 2 - 0.004, ya - 0.012, zs + 0.016)]
            npath += [(-cw / 2 + 0.03 + rn_ * math.cos(a), yf + 0.03 + rn_ * math.sin(a), zs + 0.016)
                      for a in np.radians(np.linspace(180, 270, 5))]
            npath += [(cw / 2 - 0.03 + rn_ * math.cos(a), yf + 0.03 + rn_ * math.sin(a), zs + 0.016)
                      for a in np.radians(np.linspace(270, 360, 5))]
            npath += [(cw / 2 + 0.004, ya - 0.012, zs + 0.016)]
            nosing = sweep("nosing", npath, rounded_rect(0.016, 0.05, 0.007, 3), smooth_path=False, step=0.05)
            _face([nosing], k)
            main.append(nosing)
            bpts = [(0, 0), (0, -0.27), (0.028, -0.27), (0.04, -0.22), (0.07, -0.15), (0.12, -0.085), (0.19, -0.045), (0.25, -0.036),
                    (0.275, -0.03), (0.275, 0)]
            for sx in (-1, 1):
                bk = extrude_profile("cbracket", bpts, 0.04, bevel=0.004)
                x0 = sx * (cw / 2 - 0.1) + 0.02
                transform_verts(bk, lambda c, x0=x0: Vector((x0 - c.z, ya - 0.03 - c.x, zs - 0.004 + c.y)))
                uv_grain(bk, 'Y')
                o.append(bk)
            # stacks of papers at both ends, till dish in the middle
            nside = 1 if cw < 1.25 else 2
            for sx in (-1, 1):
                for i in range(nside):
                    x = sx * (cw / 2 - 0.16 - i * 0.29)
                    n = int(round(1 + stock * 4 * r.u(0.45, 1.0)))
                    if stock < 0.05:
                        continue
                    add_stack(r, trs((x, ya - cout + 0.185, ctop), (0, 0, r.jitter(0, 3))), n)
            dish = lathe("dish", [(0, 0), (0.05, 0), (0.066, 0.013), (0.061, 0.014), (0.047, 0.005), (0, 0.005)], segments=18,
                         loc=(r.jitter(0, 0.03), ya - 0.12, ctop))
            b_.append(dish)
            # magazines pegged on a rail across the head of the hatch
            rail = cylinder("peg_rail", 0.005, ow, loc=(0, ya + 0.02, oz1 - 0.035), rot=(0, 90, 0), verts=8)
            b_.append(rail)
            ncol = max(1, int((ow - 0.02) / 0.225))
            for i in range(ncol):
                if r.chance(0.3 + 0.7 * stock):
                    x = (i - (ncol - 1) / 2) * 0.225 + r.jitter(0, 0.006)
                    add_mag(r, _rz(k) @ trs((x, ya + 0.02, oz1 - 0.045 - 0.135), (r.jitter(0, 3), r.jitter(0, 4), 0)))
        elif kind == "door":
            dw = min(fw - 0.2, 0.84)
            dz0, dz1 = zp + 0.012, zo1 - 0.006
            for sx in (-1, 1):
                jx0, jx1 = dw / 2, fw / 2
                jb = _bar("jamb", (jx1 - jx0, T + 0.016, zo1 - zp), (sx * (jx0 + jx1) / 2, ya + T / 2 - 0.008, (zp + zo1) / 2), 'Z', seg=1)
                densify(jb, 0.3)
                o.append(jb)
            yd = ya + 0.012                                        # door face, slightly recessed
            dT = 0.042
            lockz = 1.02
            for sx in (-1, 1):
                ds = _bar("dstile", (0.09, dT, dz1 - dz0), (sx * (dw / 2 - 0.05), yd + dT / 2, (dz0 + dz1) / 2), 'Z', seg=1)
                densify(ds, 0.2)
                o.append(ds)
            iw = dw - 0.19
            o.append(_bar("drail_b", (iw, dT, 0.2), (0, yd + dT / 2, dz0 + 0.1), 'X'))
            o.append(_bar("drail_m", (iw, dT, 0.13), (0, yd + dT / 2, lockz), 'X'))
            o.append(_bar("drail_t", (iw, dT, 0.1), (0, yd + dT / 2, dz1 - 0.05), 'X'))
            a_.append(panel_set(o, -iw / 2, iw / 2, dz0 + 0.2, lockz - 0.065, yd, r))
            gz0, gz1 = lockz + 0.065, dz1 - 0.1
            gh = gz1 - gz0
            bw = 0.024
            holes = []
            pw2, ph2 = (iw - bw) / 2, (gh - bw) / 2
            for ix in (-1, 1):
                for iz in (-1, 1):
                    holes.append(_rect(ix * (pw2 + bw) / 2, iz * (ph2 + bw) / 2, pw2, ph2))
            dsash = shape_extrude("dsash", _rect(0, 0, iw + 0.01, gh + 0.01), holes, depth=0.03, loc=(0, yd + 0.036, (gz0 + gz1) / 2),
                                  rot=(90, 0, 0), bevel=0.003)
            o.append(dsash)
            g_.append(_quad_y("dglass", 0, (gz0 + gz1) / 2, iw, gh, yd + 0.022))
            # brass kick plate, knob on a long escutcheon, hinges
            kp = box("kick", (iw - 0.04, 0.003, 0.13), loc=(0, yd - 0.0015, dz0 + 0.1), bevel=0.001, segments=1)
            b_.append(kp)
            scr = bolt("kscrew", 0.005, 0.002, kind='dome')
            sm = []
            for sx in (-1, 1):
                for sz in (-1, 1):
                    sm.append(_rz(k) @ trs((sx * (iw / 2 - 0.04), yd - 0.003, dz0 + 0.1 + sz * 0.048), (90, 0, 0)))
            ks = instances_to_mesh("kscrews", scr, sm, keep_base=False)
            assign(ks, m_brass)
            kx = -(dw / 2 - 0.05)
            b_.append(box("escutcheon", (0.042, 0.004, 0.17), loc=(kx, yd - 0.002, lockz), bevel=0.0015, segments=1))
            b_.append(lathe("knob", [(0.013, 0.0), (0.016, 0.004), (0.009, 0.012), (0.009, 0.03), (0.02, 0.038), (0.026, 0.05), (0.024, 0.06),
                                     (0.014, 0.067), (0.0, 0.069)], segments=14, loc=(kx, yd - 0.004, lockz + 0.03), rot=(90, 0, 0)))
            b_.append(cylinder("keyhole", 0.006, 0.003, loc=(kx, yd - 0.005, lockz - 0.045), rot=(90, 0, 0), verts=8))
            kw_ = _rz(k) @ Vector((kx, yd, lockz + 0.03, 1.0))
            door_knob = (kw_[0], kw_[1], kw_[2])
            for hz in (dz0 + 0.22, (dz0 + dz1) / 2, dz1 - 0.22):
                o.append(cylinder("hinge", 0.0095, 0.1, loc=(dw / 2 - 0.004, yd - 0.006, hz), verts=8))
                o.append(box("hinge_leaf", (0.05, 0.004, 0.085), loc=(dw / 2 - 0.03, yd - 0.001, hz), bevel=0.001, segments=1))
        lined(o, ya)
        _face(o, k)
        _face(a_, k)
        _face(g_, k)
        _face(b_, k)
        acc.extend(a_)
        glass.extend(g_)
        brass.extend(b_)

    # ------------------------------------------------------------------ paint chipped back to the bare wood
    # small ragged patches standing a hair proud of the joinery (same paint material, wear mask = 1):
    # along the arrises of stiles, rails and sill, round the colonnette shafts, most of them low down
    chip_v, chip_f = [], []
    cs_ = 0.75 + 0.8 * w_

    def _rot(P, k):
        a = k * Q45
        c, s = math.cos(a), math.sin(a)
        return (P[0] * c - P[1] * s, P[0] * s + P[1] * c, P[2])

    def chip_edge(k, P0, d, n1, n2, L, w1, w2, r, b=0.004, eps=0.0013):
        P0, d, n1, n2 = (np.array(v, dtype=float) for v in (P0, d, n1, n2))
        base = len(chip_v)
        flip = float(np.dot(np.cross(n2, d), n1)) < 0.0
        for i in range(4):
            env = (0.1, 1.0, 0.75, 0.1)[i]
            a1 = 0.001 + w1 * env * r.u(0.55, 1.0)
            a2 = 0.001 + w2 * env * r.u(0.55, 1.0)
            P = P0 + d * (L * ((0.0, 0.3, 0.72, 1.0)[i] - 0.5))
            for v in (P - n2 * (b + a1) + n1 * eps, P - n2 * b + n1 * eps, P - n1 * b + n2 * eps, P - n1 * (b + a2) + n2 * eps):
                chip_v.append(_rot(v, k))
        for i in range(3):
            for j in range(3):
                q = (base + 4 * i + j, base + 4 * i + j + 1, base + 4 * (i + 1) + j + 1, base + 4 * (i + 1) + j)
                chip_f.append(q[::-1] if flip else q)

    def chip_round(kc, rad, zc, hgt, span, r, th0=0.0):
        cx, cy = _rot((0.0, -(Rc - 0.012), 0.0), kc)[:2]
        tc = kc * Q45 - math.pi / 2 + th0
        rr_ = rad + 0.0025
        base = len(chip_v)
        chip_v.append((cx + rr_ * math.cos(tc), cy + rr_ * math.sin(tc), zc))
        for i in range(7):
            a = 2 * math.pi * i / 7 + r.u(0, 0.5)
            q = r.u(0.5, 1.0)
            th, z = tc + span * 0.5 * q * math.cos(a), zc + hgt * 0.5 * q * math.sin(a)
            chip_v.append((cx + rr_ * math.cos(th), cy + rr_ * math.sin(th), z))
        for i in range(7):
            chip_f.append((base, base + 1 + i, base + 1 + (i + 1) % 7))

    for k in range(8):
        r = rng.fork("chips%d" % k)
        ya = -apo
        # colonnette at the corner to the right of this face: plinth scuffs, lower shaft, a few on the upper shaft
        for i in range(int(round((3.5 + 6 * w_) * r.u(0.7, 1.3)))):
            z = sh0 + 0.03 + (sh1 - sh0 - 0.08) * r.u(0, 1) ** 1.7
            chip_round(k + 0.5, 0.044, z, r.u(0.035, 0.11) * cs_, math.radians(r.u(25, 50)), r, th0=math.radians(r.jitter(0, 75)))
        for i in range(int(round((1.2 + 2.5 * w_) * r.u(0.6, 1.4)))):
            chip_round(k + 0.5, 0.064, zp + r.u(0.03, 0.085), r.u(0.03, 0.05), math.radians(r.u(20, 34)), r, th0=math.radians(r.jitter(0, 70)))
        for i in range(int(round((1.6 + 4 * w_) * r.u(0.6, 1.4)))):
            z = uh0 + 0.04 + (uh1 - uh0 - 0.1) * r.u(0, 1) ** 1.4
            chip_round(k + 0.5, 0.034, z, r.u(0.03, 0.09) * cs_, math.radians(r.u(25, 48)), r, th0=math.radians(r.jitter(0, 75)))
        if k == 4:
            dw_ = min(fw - 0.2, 0.84)
            yd_ = ya + 0.012
            for i in range(int(round(3 + 7 * w_))):               # round the lock, where hands and keys rub
                sx = -1 if i % 3 else 1
                z = 1.02 + r.jitter(0.02, 0.17) if sx < 0 else zp + 0.25 + r.u(0, 1.2)
                chip_edge(k, (sx * (dw_ / 2 - 0.095), yd_, z), (0, 0, 1), (0, -1, 0), (-sx, 0, 0), r.u(0.04, 0.12) * cs_,
                          r.u(0.006, 0.02) * cs_, 0.003, r)
            for i in range(int(round(2 + 4 * w_))):               # bottom rail of the door, kicked open
                chip_edge(k, (r.jitter(0, dw_ / 2 - 0.16), yd_, zp + 0.212), (1, 0, 0), (0, -1, 0), (0, 0, 1), r.u(0.04, 0.12) * cs_,
                          r.u(0.006, 0.02) * cs_, 0.003, r)
            continue
        pw_ = fw - 0.2
        for i in range(int(round((2.5 + 8 * w_) * r.u(0.7, 1.3)))):
            sx = r.pick((-1, 1))
            z = zp + 0.2 + (zb - zp - 0.32) * r.u(0, 1) ** 1.8
            chip_edge(k, (sx * (fw / 2 - 0.10), ya, z), (0, 0, 1), (0, -1, 0), (-sx, 0, 0), r.u(0.04, 0.13) * cs_, r.u(0.006, 0.02) * cs_,
                      0.003, r)
        for i in range(int(round((1.5 + 4 * w_) * r.u(0.6, 1.4)))):
            chip_edge(k, (r.jitter(0, pw_ / 2 - 0.07), ya, zp + 0.16), (1, 0, 0), (0, -1, 0), (0, 0, 1), r.u(0.05, 0.14) * cs_,
                      r.u(0.008, 0.024) * cs_, 0.003, r)
        if k != 0:
            for i in range(int(round((1.5 + 4 * w_) * r.u(0.6, 1.4)))):
                chip_edge(k, (r.jitter(0, fw / 2 - 0.12), -(apo + 0.066), zs), (1, 0, 0), (0, 0, 1), (0, -1, 0), r.u(0.04, 0.12) * cs_,
                          r.u(0.008, 0.02) * cs_, r.u(0.004, 0.008), r, b=0.006)
            for i in range(int(round((1.2 + 5 * w_) * r.u(0.6, 1.4)))):
                sx = r.pick((-1, 1))
                chip_edge(k, (sx * (fw / 2 - 0.08), ya, oz0 + 0.05 + (oh - 0.1) * r.u(0, 1) ** 1.5), (0, 0, 1), (0, -1, 0), (-sx, 0, 0),
                          r.u(0.04, 0.11) * cs_, r.u(0.005, 0.016) * cs_, 0.003, r)
    chips_obj = mesh_object("paint_chips", chip_v, chip_f)
    assign(chips_obj, m_main)
    paint(chips_obj, wear=1.0, grime=0.12)

    # the vendor's stool and a spare bundle on the floor inside
    zfl = zp + 0.03
    sx0, sy0 = 0.12 * apo, 0.3 * apo
    main.append(lathe("stool_seat", [(0, zfl + 0.56), (0.13, zfl + 0.56), (0.15, zfl + 0.572), (0.15, zfl + 0.59), (0.135, zfl + 0.6),
                                     (0, zfl + 0.6)], segments=14, loc=(sx0, sy0, 0)))
    for j in range(3):
        a = math.radians(30 + 120 * j)
        main.append(sweep("stool_leg", [(sx0 + 0.17 * math.cos(a), sy0 + 0.17 * math.sin(a), zfl),
                                        (sx0 + 0.09 * math.cos(a), sy0 + 0.09 * math.sin(a), zfl + 0.565)], 0.015, sides=6, smooth_path=False))
    main.append(torus("stool_ring", major=0.135, minor=0.009, loc=(sx0, sy0, zfl + 0.2), major_seg=14, minor_seg=5))

    # bundles of the morning delivery left on the paving by a front corner (and one inside)
    rb = rng.fork("bundles")
    if stock > 0.25:
        side = rb.pick((1, 7))
        M0 = _rz(side) @ trs((rb.jitter(0.12, 0.08), -(apo + 0.36), 0.0), (0, 0, rb.jitter(0, 14)))
        add_stack(rb, M0, int(round(3 + 4 * stock * rb.u(0.6, 1))))
        if stock > 0.6:
            add_stack(rb, M0 @ trs((0.31, 0.03, 0.0), (0, 0, rb.jitter(0, 20))), int(round(2 + 2 * rb.u(0, 1))))
        add_stack(rb, trs((-0.45 * apo, 0.38 * apo, zfl), (0, 0, rb.jitter(20, 15))), int(round(2 + 5 * stock * rb.u(0.5, 1))))

    printed = []
    for base, ms, nm in ([(news[i], news_m[i], "news%d" % i) for i in range(NN)] + [(mags[i], mags_m[i], "mags%d" % i) for i in range(NM)]
                         + [(bundle, bundle_m, "bundles"), (pbig, pbig_m, "posters_big")]
                         + [(psm[i][0], psm_m[i], "posters_small%d" % i) for i in range(2)]):
        if ms:
            printed.append(instances_to_mesh(nm, base, ms, keep_base=False, seed=p.seed))
        else:
            delete([base])

    # ------------------------------------------------------------------ marquise: batten rolls, brackets, valance
    def rolls(name, pts_fn, nsheet, kk, rad, cut=None):
        """Round batten rolls along a radial profile on face kk: pts_fn(s) -> path for lateral fraction s."""
        out = []
        for j in range(1, nsheet):
            s = 2.0 * j / nsheet - 1.0
            out.append(sweep(name, pts_fn(s, cut if abs(s) > 0.01 else None), rad, sides=5, smooth_path=False))
        return _face(out, kk)

    def can_path(s, cut=None):
        r0, r1 = apo - 0.03, apo + cp - 0.012
        return [(s * r0 * T22, -r0, zh + 0.215 + 0.004), (s * r1 * T22, -r1, zh + 0.103 + 0.004)]

    nsh = max(2, int(round(2 * (apo + cp) * T22 / 0.36)))
    for k in range(8):
        metal.extend(rolls("can_roll", can_path, nsh, k, 0.011))
        hip = sweep("can_hip", [(0, -(apo - 0.03) / C22, zh + 0.222), (0, -(apo + cp - 0.004) / C22, zh + 0.105)], 0.016, sides=8,
                    smooth_path=False)
        metal.extend(_face([hip], k + 0.5))
    # scrolled brackets from the colonnette heads to the marquise edge; the two front ones carry a lamp instead of a scroll
    cpc = cp / C22
    flatbar = [(-0.006, -0.016), (0.006, -0.016), (0.006, 0.016), (-0.006, 0.016)]
    lamp_corners = (0, 7) if p.corner_lamps else ()
    y0 = -(Rc + 0.02)
    for k in range(8):
        pts = bezier_path((0, y0, zo1 - 0.3), (0, y0 - 0.03, zo1 - 0.02), (0, y0 - cpc * 0.45, zh + 0.035), (0, y0 - cpc * 0.86, zh + 0.058), n=9)
        b1 = sweep("bracket", pts, flatbar, up=(1, 0, 0))
        bl = [b1]
        if k not in lamp_corners:
            rs = min(0.075, cpc * 0.22)
            sc = spiral_path(0.008, rs, 1.5, n=18, a0_deg=90, plane="YZ", center=(0, y0 - 0.035 - rs, zh + 0.025 - rs * 1.25))
            bl.append(sweep("bracket_scroll", sc, flatbar, up=(1, 0, 0), scale_fn=lambda t: 0.55 + 0.45 * t))
        main.extend(_face(bl, k + 0.5))
    if p.valance:
        Wv = 2 * (apo + cp) * T22 - 0.012
        nsc = max(3, int(round(Wv / 0.15)))
        wsc = Wv / nsc
        h0 = 0.045
        outer = [(-Wv / 2, 0.0), (-Wv / 2, -h0)]
        holes = []
        for i in range(nsc):
            xl = -Wv / 2 + i * wsc
            for q in range(1, 5):
                t = math.pi * q / 5
                outer.append((xl + wsc / 2 - (wsc / 2) * math.cos(t), -h0 - (wsc * 0.46) * math.sin(t) ** 0.8))
            outer.append((xl + wsc, -h0))
            holes.append(circle_pts(0.015, 6, center=(xl + wsc / 2, -h0 - wsc * 0.14)))
        outer.append((Wv / 2, 0.0))
        for k in range(8):
            v = shape_extrude("valance", outer, holes, depth=0.012, loc=(0, -(apo + cp) + 0.016, zh + 0.07), rot=(90, 0, 0))
            acc.extend(_face([v], k))

    # ------------------------------------------------------------------ softly lit frieze with lettering
    fwf = 2 * (af - 0.02) * T22
    hf = zf1 - zf0 - 0.03
    zfc = (zf0 + 0.02 + zf1 - 0.012) / 2
    letters = []
    for k in range(8):
        fr = shape_extrude("frieze_frame", _rect(0, 0, fwf, hf), [_rect(0, 0, fwf - 0.12, hf - 0.07)], depth=0.03,
                           loc=(0, -(af - 0.02), zfc), rot=(90, 0, 0), bevel=0.004)
        main.extend(_face([fr], k))
        gp = _quad_y("frieze_glass", 0, zfc, fwf - 0.10, hf - 0.05, -(af - 0.0056))
        glow.extend(_face([gp], k))
        # CLEARANCE RULE: the ruled line and the letters stand 5 mm and more in front of the lit glass (it was 0.6 /
        # 1.1 mm); the glass stays where it was, in front of the tucked-in top edge of the marquise sheet
        ln = shape_extrude("frieze_line", _rect(0, 0, fwf - 0.15, hf - 0.1), [_rect(0, 0, fwf - 0.163, hf - 0.113)], depth=0.002,
                           loc=(0, -(af - 0.0006), zfc), rot=(90, 0, 0))
        letters.extend(_face([ln], k))
        body = p.sign_text if k % 2 == 0 else p.sign_text_alt
        if body.strip():
            t = _fit_text("sign", body, fwf - 0.23, hf - 0.15, 0.002, font="serif", front_only=True)
            if t is not None:
                place(t, loc=(0, -(af - 0.0001), zfc))
                letters.extend(_face([t], k))
    for o in letters:
        assign(o, m_main)

    # ------------------------------------------------------------------ roof: lapped sheets, batten rolls, hips, finial
    hk, rprof = ROOFS[p.roof]
    Hr = D * hk * p.roof_height
    a0 = ae - 0.03
    dense = np.array(resample_path([(rr * a0, 0.0, ze + 0.002 + zz * Hr) for rr, zz in rprof], n=64))
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    def at(s):
        return float(np.interp(s, cum, dense[:, 0])), float(np.interp(s, cum, dense[:, 2]))

    nco = max(4, int(round(total / 0.3)))
    mq = 4
    rows, row_lap = [], []
    for j in range(nco):
        for q in range(mq):
            rr, zz = at(total * (j + q / (mq - 1)) / nco)
            rows.append((rr + 0.007 * (1 - q / (mq - 1)) * (1.0 if j > 0 else 0.0), zz))
            # sheltered strip of a sheet: its head, tucked under the drip edge of the sheet above
            row_lap.append((1.0 if q == mq - 1 else (0.2 if q == mq - 2 else 0.0)) if j < nco - 1 else 0.0)
    z_laps = np.array([at(total * j / nco)[1] for j in range(1, nco)])
    zt = rows[-1][1]
    rt = rows[-1][0]
    nrs = max(2, int(round(2 * a0 * T22 / 0.4)))
    nu = nrs * 6
    rv, rf, rlap = [], [], []
    for k in range(8):
        ck, sk = math.cos(k * Q45), math.sin(k * Q45)
        base = len(rv)
        for ri, (rr, zz) in enumerate(rows):
            for i in range(nu + 1):
                x, y = (2.0 * i / nu - 1.0) * rr * T22, -rr
                rv.append((x * ck - y * sk, x * sk + y * ck, zz))
                rlap.append(row_lap[ri])
        for ri in range(len(rows) - 1):
            for i in range(nu):
                a = base + ri * (nu + 1) + i
                rf.append((a, a + 1, a + nu + 2, a + nu + 1))
    roof = mesh_object("roof", rv, rf)
    smooth(roof, 35)
    assign(roof, m_roof)
    rv_np, rlap_np = np.array(rv), np.array(rlap)

    def roof_path(s, cut=None):
        pts = []
        for i in list(range(0, 63, 4)) + [63]:
            rr, zz = float(dense[i, 0]), float(dense[i, 2])
            if cut is not None and rr < cut and i > 6:
                break
            pts.append((s * rr * T22, -(rr + 0.008), zz + 0.004))
        return pts

    for k in range(8):
        metal.extend(rolls("roof_roll", roof_path, nrs, k, 0.011, cut=a0 * 0.42))
        hp = [(0, -(float(dense[i, 0]) + 0.006) / C22, float(dense[i, 2]) + 0.004) for i in list(range(0, 63, 4)) + [63]]
        hip = sweep("roof_hip", hp, 0.02, sides=5, smooth_path=False)
        metal.extend(_face([hip], k + 0.5))
    # little cast acroteria standing on the eaves at the corners
    lf = 0.15 * min(1.0, D / 2.4 + 0.1)
    palm = [(-0.3, 0), (0.3, 0), (0.34, 0.12), (0.5, 0.3), (0.42, 0.52), (0.26, 0.46), (0.3, 0.7), (0.14, 0.72), (0.0, 1.0),
            (-0.14, 0.72), (-0.3, 0.7), (-0.26, 0.46), (-0.42, 0.52), (-0.5, 0.3), (-0.34, 0.12)]
    ra = rng.fork("acroteria")
    for k in range(8):
        if ra.chance(0.45 * max(0.0, w_ - 0.55) / 0.45):
            continue                                     # broken off long ago
        ac = shape_extrude("acroterion", [(x * lf, y * lf) for x, y in palm], [circle_pts(0.012, 6, center=(0, 0.3 * lf))], depth=0.012,
                           loc=(0, -(ae / C22 - 0.045), ze), rot=(90 + ra.jitter(0, 6 * w_), 0, 0))
        metal.extend(_face([ac], k + 0.5))
    fs = min(1.25, max(0.85, D / 2.4)) * 0.82
    rtc = rt / C22
    fin = lathe("finial_base", [(rtc + 0.03, zt - 0.06), (rtc + 0.034, zt - 0.03), (rtc + 0.02, zt - 0.01), (rtc + 0.026, zt + 0.01 * fs),
                                (rtc * 0.7, zt + 0.04 * fs), (rtc * 0.45, zt + 0.09 * fs), (rtc * 0.5, zt + 0.12 * fs), (rtc * 0.95, zt + 0.16 * fs),
                                (rtc * 1.05, zt + 0.2 * fs), (rtc * 0.85, zt + 0.245 * fs), (rtc * 0.4, zt + 0.285 * fs), (0.02, zt + 0.32 * fs),
                                (0.018, zt + 0.36 * fs)], segments=18)
    metal.append(fin)
    zb_ = zt + 0.41 * fs
    brass.append(sphere("finial_ball", 0.06 * fs, loc=(0, 0, zb_), segments=18, rings=10))
    brass.append(lathe("finial_spike", [(0.02, zb_ + 0.05 * fs), (0.028, zb_ + 0.065 * fs), (0.014, zb_ + 0.085 * fs), (0.01, zb_ + 0.2 * fs),
                                        (0.019, zb_ + 0.225 * fs), (0.008, zb_ + 0.25 * fs), (0.0, zb_ + 0.4 * fs)], segments=10))

    # ------------------------------------------------------------------ lamps: enamel shades over a warm bulb
    def shaded_lamp(name, x, y, ztop, zr, rs, rb):
        """Pendant on a brass rod: enamelled cone shade (paint outside, cream inside) with its rim at zr; the bulb
        hangs just below the rim.  The shade casts the light's shadow, so nothing above the rim plane is lit."""
        hs = rs * 0.6
        rn = min(0.03, rs * 0.42)
        brass.append(cylinder(name + "_rod", 0.005, ztop - (zr + hs), loc=(x, y, (ztop + zr + hs) / 2), verts=6))
        brass.append(lathe(name + "_rose", [(0.0, ztop - 0.014), (0.02, ztop - 0.012), (0.026, ztop - 0.003), (0.026, ztop)], segments=10,
                           loc=(x, y, 0)))
        brass.append(lathe(name + "_holder", [(0.0, zr + hs - 0.016), (rn * 0.78, zr + hs - 0.016), (rn * 0.86, zr + hs - 0.008),
                                              (rn * 0.86, zr + hs + 0.02), (rn * 0.5, zr + hs + 0.032), (0.0, zr + hs + 0.034)],
                           segments=10, loc=(x, y, 0)))
        shade = lathe(name + "_shade", [(rs - 0.003, zr - 0.002), (rs, zr), (rs * 0.95, zr + 0.008), (rs * 0.6, zr + hs * 0.52), (rn, zr + hs),
                                        (rn * 0.6, zr + hs + 0.004), (rn * 0.6, zr + hs - 0.003), (rn - 0.002, zr + hs - 0.004),
                                        (rs * 0.6 - 0.004, zr + hs * 0.52 - 0.004), (rs - 0.007, zr + 0.004), (rs - 0.003, zr - 0.002)],
                      segments=16, loc=(x, y, 0))
        assign(shade, m_main)
        assign_faces(shade, m_acc, lambda c, n: n.z < -0.15)
        zbulb = zr - rb * 0.7
        glow.append(sphere(name + "_bulb", rb, loc=(x, y, zbulb), segments=12, rings=8, scale=(1, 1, 1.3)))
        return shade, (x, y, zbulb)

    shades = []
    zr_i = min(zo1 - 0.13, 2.0)
    sh_, lp = shaded_lamp("pendant", 0.0, 0.0, zh + 0.035, zr_i, 0.17, 0.034)
    shades.append(sh_)
    light("interior", lp, color="#ffc27a", intensity=p.intensity, range=8.0, flicker=0.05, shadow=True, shadow_near=0.04)
    lo = cpc * 0.8
    for k in lamp_corners:
        M = _rz(k + 0.5) @ Vector((0.0, y0 - lo, 0.0, 1.0))
        sh_, lp = shaded_lamp("corner_lamp%d" % k, M[0], M[1], zh + 0.035, zh - 0.1, min(0.07, 0.035 + cpc * 0.085), 0.026)
        shades.append(sh_)
        cd = p.intensity * 0.09 * clamp(lo / 0.33, 0.4, 1.0) ** 2
        light("corner%d" % k, lp, color="#ffbf6a", intensity=cd, range=5.0, flicker=0.08, shadow=True, shadow_near=0.025)

    # ------------------------------------------------------------------ materials
    for o in main:
        assign(o, m_main)
    for o in acc:
        assign(o, m_acc)
    for o in brass:
        assign(o, m_brass)
    for o in glass:
        assign(o, m_glass)
    for o in glow:
        assign(o, m_glow)
    for o in metal:
        assign(o, m_roof)

    # ------------------------------------------------------------------ weathering: painted joinery
    def splash(co, n):
        """Dirty splash band in the bottom 0.3 m plus a little general grime."""
        h = co[:, 2] - zp
        nz = noise_mask(0.22, seed=sd + 2, lo=0.1, hi=0.9)(co, n)
        band = np.clip(1.0 - h / 0.32, 0.0, 1.0)
        return np.clip(band ** 0.8 * (0.7 + 0.3 * nz) * (0.85 + 0.15 * w_) + 0.14 * w_ * nz, 0.0, 1.0)

    def chips(co, n):
        """Paint chipped to bare wood: fine chips everywhere, scuffing in the kick zone, hand wear at waist height."""
        h = co[:, 2] - zp
        fine = noise_mask(0.05, seed=sd + 3, lo=0.66 - 0.22 * w_, hi=0.88 - 0.22 * w_)(co, n)
        kick = np.clip(1.0 - h / 0.42, 0.0, 1.0) * noise_mask(0.09, seed=sd + 4, lo=0.25, hi=0.7)(co, n)
        hand = np.exp(-((co[:, 2] - 1.12) / 0.2) ** 2) * noise_mask(0.08, seed=sd + 6, lo=0.3, hi=0.75)(co, n)
        v = fine * (0.2 + 0.7 * w_) + kick * (0.4 + 0.45 * w_) + hand * (0.2 + 0.4 * w_)
        if door_knob is not None:
            dk = np.linalg.norm(co - np.array(door_knob)[None, :], axis=1)
            v = v + np.exp(-(dk / 0.16) ** 2) * (0.35 + 0.4 * w_)
        return np.clip(v, 0.0, 1.0)

    def fade(co, n):
        up = np.clip(n[:, 2], 0.0, 1.0)
        return np.clip((0.25 + 0.75 * up) * noise_mask(0.6, seed=sd + 5, lo=0.2, hi=0.8)(co, n) * (0.3 + 0.9 * w_), 0.0, 1.0)

    for o in main + acc + lined_objs + shades:
        paint(o, grime=splash, wear=chips, special=fade)
    if counter is not None:
        # paint rubbed through where the papers slide and elbows lean, most along the front edge
        def rubbed(co, n):
            kk, ap, xl = _oct_co(co)
            front = np.clip((ap - (apo - 0.05)) / 0.36, 0.0, 1.0)
            nz = noise_mask(0.07, seed=sd + 8, lo=0.15, hi=0.7)(co, n)
            return np.clip((0.3 + 0.45 * w_ + 0.35 * front) * (0.45 + 0.55 * nz) * (0.5 + 0.5 * np.clip(n[:, 2], 0, 1)), 0.0, 1.0)

        paint(counter, wear=rubbed, grime=mask_mul(noise_mask(0.12, seed=sd + 9, lo=0.3, hi=0.9), 0.25 + 0.3 * w_))
        paint(nosing, wear=lambda co, n: np.clip(0.5 + 0.4 * w_ + 0.5 * (noise_mask(0.05, seed=sd + 7, lo=0.2, hi=0.8)(co, n) - 0.5), 0.0, 1.0))

    # ------------------------------------------------------------------ weathering: sheet metal
    def streak(co, n, seed):
        """Run-off streaks: water falls straight down each facet, so the staining is constant along the fall line."""
        kk, ap, xl = _oct_co(co)
        fake = np.stack([xl + kk * 3.1, co[:, 2] * 0.07, np.zeros(len(co))], axis=1)
        return noise_mask(0.05, seed=seed, lo=0.42 if zinc else 0.5, hi=0.8 if zinc else 0.85)(fake, n)

    neck_h = (0.13 if p.roof == "onion" else 0.07) * Hr

    def roof_zones(co, n):
        kk, ap, xl = _oct_co(co)
        z = co[:, 2]
        halfw = np.maximum(ap * T22, 1e-3)
        s = np.clip(xl / halfw, -1.0, 1.0)
        sj = np.round((s + 1.0) * nrs / 2.0) * 2.0 / nrs - 1.0
        roll = np.where(ap > a0 * 0.42, np.exp(-np.abs(s - sj) * halfw / (0.045 if zinc else 0.03)), 0.0)
        hipz = np.exp(-(1.0 - np.abs(s)) * halfw / 0.05)
        if len(co) == len(rv_np) and np.allclose(co, rv_np, atol=1e-3):
            lap = rlap_np
        else:
            dz = z_laps[None, :] - z[:, None]
            lap = np.max(np.where((dz > -0.002) & (dz < 0.06), 1.0 - dz / 0.06, 0.0), axis=1)
        top = np.clip((z - (zt - 0.32)) / 0.26, 0.0, 1.0)
        neck = np.clip(1.0 - (z - ze) / neck_h, 0.0, 1.0)
        under = np.clip((0.12 - n[:, 2]) / 0.3, 0.0, 1.0)                # under the bulge: never rinsed by the rain
        shelter = np.maximum.reduce([(0.75 if zinc else 0.6) * roll, (0.75 if zinc else 0.6) * hipz, lap, top, neck])
        return shelter, neck, lap, under

    def roof_special(co, n):
        if zinc:
            return np.zeros(len(co))
        shelter, neck, lap, under = roof_zones(co, n)
        st = streak(co, n, sd + 11)
        expo = np.clip(n[:, 2], 0.0, 1.0)
        patch = noise_mask(0.9, seed=sd + 16, lo=0.3, hi=0.7)(co, n) - 0.5   # some sheets weather faster than others
        return np.clip(0.12 * w_ + 0.9 * shelter + (0.45 + 0.4 * w_) * st + 0.16 * patch
                       - 0.2 * expo * (1.0 - shelter), 0.0, 1.0)

    def roof_grime(co, n):
        shelter, neck, lap, under = roof_zones(co, n)
        st = streak(co, n, sd + 12)
        if zinc:
            return np.clip((0.02 + 0.3 * np.maximum(shelter, under) + 0.25 * neck + 0.3 * st) * (0.75 + 0.45 * w_), 0.0, 1.0)
        # the underside of the bulge is never rinsed: soot and dust darken it instead of bright verdigris
        return np.clip(0.06 + 0.3 * neck + 0.2 * lap + 0.12 * st + 0.55 * under, 0.0, 1.0)

    for o in [roof] + [m for m in metal if m.name.startswith(("roof", "acro", "finial"))]:
        paint(o, special=roof_special, grime=roof_grime)

    def can_zones(co, n):
        kk, ap, xl = _oct_co(co)
        halfw = np.maximum(ap * T22, 1e-3)
        s = np.clip(xl / halfw, -1.0, 1.0)
        sj = np.round((s + 1.0) * nsh / 2.0) * 2.0 / nsh - 1.0
        roll = np.exp(-np.abs(s - sj) * halfw / (0.035 if zinc else 0.028))
        junction = np.clip(1.0 - (ap - (apo - 0.035)) / 0.13, 0.0, 1.0)
        return np.maximum((0.75 if zinc else 0.6) * roll, junction), junction

    def can_special(co, n):
        if zinc:
            return np.zeros(len(co))
        shelter, junction = can_zones(co, n)
        return np.clip(0.15 * w_ + 0.85 * shelter + (0.3 + 0.4 * w_) * streak(co, n, sd + 13), 0.0, 1.0)

    def can_grime(co, n):
        shelter, junction = can_zones(co, n)
        st = streak(co, n, sd + 14)
        if zinc:
            return np.clip((0.03 + 0.32 * shelter + 0.25 * junction + 0.32 * st) * (0.75 + 0.45 * w_), 0.0, 1.0)
        return np.clip(0.06 + 0.35 * junction + 0.12 * st, 0.0, 1.0)

    for o in [can] + [m for m in metal if m.name.startswith("can")]:
        paint(o, special=can_special, grime=can_grime)
    for o in printed:
        paint(o, grime=mask_mul(noise_mask(0.2, seed=sd + 14, lo=0.4, hi=0.9), 0.2 + 0.3 * w_), special=0.12 + 0.2 * w_)
    for o in brass:
        paint(o, special=mask_mul(noise_mask(0.05, seed=sd + 15), 0.2 + 0.35 * w_))

    socket("customer", "attach", (0, -(apo + 0.75), 0), role="stand")
    socket("finial", "attach", (0, 0, zb_ + 0.4 * fs), role="prop_top")
