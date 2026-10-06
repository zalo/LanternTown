"""<One paragraph: what the object is and how it is constructed, part by part, the way the real thing is made.
Name the materials, the moving parts, what glows, and what the seed varies (lean, chips, missing pieces).>
"""
from tk import *
import numpy as np

# class sets the triangle budget: prop_small 6k, component 12k, prop 16k, furniture 22k, street_furniture 26k,
# vegetation 60k, structure 70k, ground 80k, building 160k, landmark 260k. mount: ground | wall | wall_ground | hang.
# focus: where the night close-up looks (height fraction or [x, y, z]). lod: True (model all levels) or "far".
META = {"name": "Human name", "class": "prop", "mount": "ground", "focus": 0.6, "lod": "far"}

PARAMS = {
    "width": {"default": 1.2, "min": 0.8, "max": 2.0, "desc": "overall width (m)"},
    "style": {"default": "plain", "choices": ["plain", "ornate"], "desc": "trim style"},
    "paint": {"default": "#2b4a3a", "desc": "paint colour of the woodwork"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age: chips, rust runs, splash-back, soot (0 fresh, 1 neglected)"},
    "text": {"default": "CAFE", "desc": "lettering (all text is a parameter)"},
    "intensity": {"default": 30.0, "min": 5.0, "max": 80.0, "desc": "light per lantern (cd)"},
}   # an int "seed" is added automatically

VARIANTS = {    # 3-5, clearly different in size / style / colour / condition; "default" exists implicitly
    "ornate_wide": {"width": 1.8, "style": "ornate", "seed": 7},
    "worn_red": {"paint": "#5a2a1e", "wear": 0.9, "seed": 12},
    "small_green": {"width": 0.9, "paint": "#3d5a3a", "wear": 0.3, "seed": 21},
}


def build(p, rng):
    """Units metres, Z up, the front faces -Y. Everything left in the scene is exported."""
    far = lod() == 2                      # the far level is modelled here: outline, colour zones and glow only
    rc = rng.fork("condition")            # a stable sub-stream per concern: a level may skip draws from a fork
    wood = mat("wood_painted", p.paint, rough=1.1)
    iron = mat("iron_painted", "#1b2622", rough=1.6)
    brass = mat("brass", rough=0.9)
    glass = mat("glass_lantern", "#ffcf8a", glow=4.0)
    body, metal, trim = [], [], []

    # ---- primary form: real dimensions, built the way it is made (separate boards, posts, rails with thickness)
    w, h, d = float(p.width), 0.95, 0.4
    for i, x in enumerate(np.linspace(-w / 2 + 0.06, w / 2 - 0.06, 6)):
        b = plank("board", length=h, width=0.11, thick=0.022, loc=(x + rc.jitter(0, 0.003), 0, h / 2), axis="Z",
                  bevel=lod_pick(0.002, 0.0), warp=rc.u(0.0, 0.004), rng=rc)       # grain UVs run along the board
        body.append(b)
    for sx in (-1, 1):
        post = box("post", (0.07, 0.07, h + 0.1), loc=(sx * (w / 2 - 0.035), 0, (h + 0.1) / 2), bevel=lod_pick(0.004, 0.0), segments=1)
        body.append(post)

    # ---- secondary forms: mouldings, brackets, bands, caps (a moulded profile, never a bare primitive)
    cap = extrude_profile("cap", [(-w / 2 - 0.02, 0), (w / 2 + 0.02, 0), (w / 2 + 0.02, 0.03), (-w / 2 - 0.02, 0.03)],
                          0.46, loc=(0, 0.23, h + 0.1), rot=(90, 0, 0), bevel=lod_pick(0.004, 0.0))
    body.append(cap)

    # ---- tertiary: fixings and relief, dropped at the far level (lod_drop keeps rng and sockets identical)
    tiny = []
    bt = bolt("bolt", 0.008, 0.006)
    tiny.append(instances_to_mesh("bolts", bt, [trs((sx * (w / 2 - 0.035), -0.036, z), (90, 0, 0))
                                                for sx in (-1, 1) for z in (0.15, h - 0.1)], keep_base=False))
    metal.extend(tiny)
    lod_drop(tiny, above=1)

    # ---- light: a visible emissive source AND a light socket in candela (lit ground rich amber, never white)
    lamp = sphere("mantle", 0.03, loc=(0, -0.1, h + 0.3), segments=lod_pick(16, 8, 6), rings=lod_pick(10, 6, 4))
    assign(lamp, mat("emissive", "#ffd9a0", glow=12.0))
    light("lamp", (0, -0.1, h + 0.3), color="#ffbf6a", intensity=p.intensity, range=8.0, flicker=0.1, shadow_near=0.12)

    # ---- text is a parameter: raised, engraved or painted (1 mm proud in a contrasting material)
    if str(p.text).strip():
        tx = text("sign_text", str(p.text), size=0.12, depth=0.004, loc=(0, -0.012, h - 0.25), font="serif")
        assign(tx, brass)

    for o in body:
        assign(o, wood)
    for o in metal:
        assign(o, iron)

    # ---- weathering tells a story (paint the art-directed part; cavity grime, edge wear and AO are automatic)
    wk = 0.3 + 0.9 * float(p.wear)
    paint(body, grime=mask_mul(height_ramp(0.0, 0.2), noise_mask(0.2, seed=p.seed + 1, lo=0.2, hi=0.9), 0.6 * wk),
          wear=mask_mul(noise_mask(0.05, seed=p.seed + 2, lo=0.5, hi=0.7), 0.5 * wk),
          special=0.0)
    paint(metal, grime=0.2, special=mask_mul(height_ramp(0.0, 0.3), 0.8 * wk))    # rust low down on iron

    # ---- sockets: where other assets attach; moving parts with an explicit pivot and axis
    socket("sign_bracket", "attach", (0, -0.05, h + 0.1), rot=(0, 0, 0), role="sign")
