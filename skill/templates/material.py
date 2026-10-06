"""<What the material is and how the real thing is built up, layer by layer: units (stones, boards, tiles), how they
are laid, what the joints are, what wears and where. State the true physical scale (a brick is 215 x 65 mm).>
"""
from tk.tex import *          # numpy as np, Canvas ops (all periodic), Layout, ramp(), mix(), colour(), smoothstep()

META = {"normal_strength": 1.0, "ao_radius": 0.02, "ao_strength": 0.9}   # tile and res come from the catalogue entry


def build(t):
    """t covers ONE tile: t.W x t.H metres, t.w x t.h pixels. Build only from t.* ops and per-pixel maths so the result
    wraps; never absolute pixel ramps, never unseeded randomness. Return linear albedo, height in metres, rough, metal."""
    # 1. layout: the units, snapped so they divide the tile (running bond: 4 lengths x 12 courses of 215 x 65 mm here)
    lay = t.grid(0.225, 0.075, shift=0.5, jitter_w=0.08, seed=1)
    lay = lay.rounded(t, radius=0.004)                        # eased arrises
    # 2. height first, as a sum of physically meant layers (true depths: joints 3-8 mm, grain 0.1-0.6 mm)
    h = lay.dome(radius=0.012, power=1.6) * 0.004              # the unit's crown, 4 mm proud of the mortar
    h += lay.tilt(0.0012, k=1) + (lay.rand(2) - 0.5) * 0.0015  # per-unit level and tilt differences
    h -= (1.0 - lay.mask(gap=0.008, soft=0.002)) * 0.003       # recessed mortar bed
    h += (t.fbm(0.004, octaves=4, gain=0.6, seed=3) - 0.5) * 0.0004     # pores and grit (never white noise: it sparkles)
    h -= smoothstep(0.6, 0.85, t.fbm(0.015, seed=5)) * t.edges(h, radius=0.006, height=0.001) * 0.002   # chipped arrises
    h = t.warp(h, amount=0.0015, scale=0.08, seed=6)          # a millimetre or two of warp breaks the mechanical look
    # 3. masks derived from the height
    cav = t.cavity(h, radius=0.006, depth=0.001)
    edge = t.edges(h, radius=0.004, height=0.0008)
    # 4. colour: 3-6 observed colours through a ramp driven by noise and per-unit variation; low saturation
    key = sat(0.55 * t.fbm(0.3, seed=7) + 0.45 * lay.rand(3))
    alb = ramp(key, [(0.0, "#6e4a3a"), (0.35, "#8e5e46"), (0.7, "#7a4e42"), (1.0, "#5e4238")])   # sRGB hex -> linear
    alb = mix(alb, colour("#3a3128", key.shape), cav * 0.6)               # dirt in the joints
    alb = mix(alb, colour("#a08a74", key.shape), edge * 0.25)             # worn, lighter arrises
    alb = mix(alb, colour("#8a7a6a", key.shape), (1.0 - lay.mask(gap=0.008)) * 0.8)   # mortar
    # 5. roughness from the same masks (polished crowns smoother, joints rougher); metal 0 unless bare metal
    rough = 0.78 - 0.12 * lay.dome(radius=0.02) + 0.18 * cav + 0.08 * (t.fbm(0.05, seed=9) - 0.5)
    return dict(albedo=alb, height=h, rough=sat(rough), metal=0.0)
