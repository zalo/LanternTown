"""Toolkit smoke test: exercises every builder once."""
from tk import *

META = {"name": "tk smoke", "class": "prop", "mount": "ground", "tri_budget": 200000}
PARAMS = {"h": {"default": 1.0, "min": 0.5, "max": 2.0, "desc": "height"}}
VARIANTS = {"tall": {"h": 1.6, "seed": 4}}


def build(p, rng):
    iron = mat("plain", "#333333")
    wood = mat("plain", "#885533", rough=0.8)
    b = box("base", (0.5, 0.5, 0.1), loc=(0, 0, 0.05), bevel=0.01)
    assign(b, iron)
    c = cylinder("col", 0.05, p.h, loc=(0, 0, 0.1 + p.h / 2), bevel=0.004)
    assign(c, iron)
    l = lathe("urn", [(0, 0), (0.1, 0), (0.12, 0.05), (0.08, 0.12), (0.11, 0.2), (0.1, 0.2), (0.07, 0.12), (0, 0.1)], loc=(0.5, 0, 0))
    assign(l, wood)
    s = sweep("scroll", spiral_path(0.02, 0.15, 1.75, plane="XZ", center=(0, 0, 0.1 + p.h)), rounded_rect(0.02, 0.006, 0.002, 2),
              scale_fn=lambda t: 0.6 + 0.4 * t)
    assign(s, iron)
    t = torus("ring", 0.1, 0.01, loc=(0, 0, 0.3))
    assign(t, iron)
    pl = plank("plank", 1.0, 0.12, 0.03, loc=(0, 0.5, 0.015), warp=0.01, rng=rng, end_mat=mat("plain", "#aa8855"))
    assign(pl, wood)
    st = stone_block("stone", (0.4, 0.25, 0.2), loc=(-0.6, 0, 0.1), rng=rng)
    assign(st, "plain")
    tx = text("txt", "CAFE", size=0.1, loc=(0, -0.26, 0.05))
    assign(tx, iron)
    sh = shape_extrude("plate", rounded_rect(0.3, 0.2, 0.03), [circle_pts(0.04, 16)], depth=0.01, loc=(0.5, 0.5, 0.0), bevel=0.002)
    assign(sh, iron)
    ch = chain("chain", catenary((-0.5, 0.5, 0.6), (0.5, 0.5, 0.6), 0.2), link_len=0.04, wire=0.004)
    assign(ch, iron)
    bt = bolt("bolt", 0.01, 0.006)
    assign(bt, iron)
    studs = instances_to_mesh("studs", bt, [trs((0.2 * cos(a), 0.2 * sin(a), 0.1), (0, 0, degrees(a))) for a in [i * pi / 4 for i in range(8)]], keep_base=False)
    sf = surface("cloth", lambda u, v: (u - 0.5, -0.8 + 0.3 * v, 0.5 + 0.05 * sin(u * 12) * v), 24, 8, thickness=0.003)
    assign(sf, mat("plain", "#aa3322"))
    paint(sf, special=height_ramp(0.5, 0.6))
    a = extrude_profile("arch", arch_pts(0.6, 1.0, kind="pointed"), 0.1, loc=(-0.6, 0.6, 0), rot=(90, 0, 0))
    assign(a, "plain")
    cut = box("cut", (0.2, 0.2, 0.2), loc=(-0.6, 0, 0.2))
    boolean(st, cut)
    sp = sphere("globe", 0.08, loc=(0, 0, 0.2 + p.h))
    assign(sp, mat("plain", "#ffcc88", glow=5))
    part([sp], "swing", pivot=(0, 0, 0.3 + p.h), motion="swing", amp=3)
    light("lamp", (0, 0, 0.2 + p.h), intensity=60)
    socket("top", "attach", (0, 0, 0.3 + p.h), role="prop_top")
