"""Brush-painted timber (old oil paint on softwood joinery), grain along +U (X). Tintable white base.

Built in the order the real thing is: (1) the board - a softwood ring field whose hard latewood
ridges telegraph faintly through the film, plus a few shrinkage checks; (2) the paint, laid on in
overlapping brush strokes along the grain: each stroke carries its own bristle ridges, is fatter
where the loaded brush landed and thins out towards the lift-off, with ropey edges where strokes
overlap; a few sagging runs ending in a bead; (3) age: hairline cracks where the wood moved under
the film, small flakes lost beside them down to grey-brown wood, dust in the brush grooves, slight
uneven yellowing and chalking. Large-scale chipping / fading is left to the engine's wear masks.
Runs hang towards -U (boards on doors and shutters stand with U vertical).
"""
from tk.tex import *

META = {"normal_strength": 5.0}

M_X = 8
N_ROWS = 9                         # brush strokes ~ 65 mm wide across the 0.6 m tile
RUNS = [(0.30, 0.17, 0.070), (0.78, 0.46, 0.110), (0.52, 0.81, 0.050), (0.10, 0.63, 0.085), (0.95, 0.08, 0.040), (0.63, 0.30, 0.060)]   # u, v, length m


def _hash(i, k=0.0):
    x = np.sin(i * 12.9898 + k * 78.233) * 43758.5453
    return (x - np.floor(x)).astype(np.float32)


def _wrapd(a, L):
    return (a + L / 2) % L - L / 2


def build(t):
    X, Y, W, H = t.X, t.Y, t.W, t.H

    # ------------------------------------------------------------------ 1. the wood underneath
    P = (t.noise(0.16, seed=1, stretch=(14.0, 1.0), bandwidth=0.45) - 0.5) * 2 * 15.0
    P += (t.noise(0.07, seed=2, stretch=(5.0, 1.0)) - 0.5) * 2 * 1.4
    P += (t.noise(0.014, seed=3, stretch=(8.0, 1.0)) - 0.5) * 2 * 0.2
    phi = P + M_X * X / W
    gx, gy = t.slope(P)
    dens = t.blur(np.sqrt((gx + M_X / W) ** 2 + gy ** 2) + 2.0, 0.004)
    aa = 1 - smoothstep(0.22, 0.45, dens * t.px)
    f = (phi - np.floor(phi)).astype(np.float32)
    late = smoothstep(0.55, 0.9, f) * (1 - smoothstep(0.95, 1.0, f)) * aa
    wood_h = t.blur(late, 0.0012)                                            # film rounds the ridges off

    def fibre_cracks(scale, stretch, gate_scale, gate_lo, gate_hi, wmax, seed):
        c = t.noise(scale, seed=seed, stretch=(stretch, 1.0))
        _, cgy = t.slope(c)
        cgy = np.abs(cgy)
        valid = smoothstep(0.12 / scale, 0.40 / scale, cgy)
        d = np.abs(c - 0.5) / np.maximum(cgy, 0.12 / scale)
        gate = smoothstep(gate_lo, gate_hi, t.noise(gate_scale, seed=seed + 1, stretch=(5.0, 1.0)))
        wd = wmax * gate * valid
        return sat(1 - smoothstep(wd * 0.3, wd + 1e-6, d)) * (gate > 0.01), gate

    # ------------------------------------------------------------------ 2. brush strokes
    rowh = H / N_ROWS
    wob = (t.noise(0.22, seed=10, stretch=(2.0, 1.0)) - 0.5) * 2 * 0.016      # strokes are not ruler-straight
    yr = (Y + wob) / rowh
    row = np.mod(np.floor(yr), N_ROWS)
    vloc = yr - np.floor(yr)                                                  # 0..1 across the stroke
    nseg = 2 + np.floor(_hash(row, 1.0) * 2.99)                               # 2..4 strokes end-to-end per row
    edge_wob = (t.noise(0.03, seed=9) - 0.5) * 0.05
    xr = X / W * nseg + _hash(row, 2.0) + ((_hash(row, 6.0) - 0.5) * 0.16 * (vloc - 0.5 + (_hash(row, 7.0) - 0.5) * 0.6) ** 2 + edge_wob / W) * nseg   # ragged, curved landing edge
    seg = np.mod(np.floor(xr), nseg)
    sloc = xr - np.floor(xr)                                                  # 0 = brush lands, 1 = lifts off
    sid = row * 7.0 + seg
    h1, h2, h3 = _hash(sid, 3.0), _hash(sid, 4.0), _hash(sid, 5.0)

    bristle_src = t.fbm(0.0048, 3, 0.62, seed=11, stretch=(55.0, 1.0))        # ridges left by the bristles
    bristle_src2 = t.fbm(0.016, 2, 0.5, seed=12, stretch=(22.0, 1.0))         # streaks of fat / lean paint
    bx = X + h1 * W
    by = Y + h2 * H
    bristle = t.sample(bristle_src, bx, by)
    lean = t.sample(bristle_src2, bx, by)
    level_out = smoothstep(0.30, 0.75, t.fbm(0.20, 3, 0.5, seed=13))          # zones where the paint flowed out flat
    b_amt = (0.45 + 0.75 * h3) * (0.35 + 0.65 * level_out) * (0.6 + 0.4 * sloc)   # drier brush drags more at the end
    load = (1 - sloc) ** 1.5                                                  # fat where the loaded brush landed
    lap = (np.exp(-(sloc / 0.035) ** 2) + 0.5 * np.exp(-((1 - sloc) / 0.02) ** 2)) * smoothstep(0.25, 0.7, t.noise(0.04, seed=8))  # soft ridge at the stroke ends
    rope = np.exp(-(np.minimum(vloc, 1 - vloc) * rowh / 0.0022) ** 2)         # ropey stroke edges
    sag = t.fbm(0.05, 3, 0.5, seed=14, stretch=(1.0, 2.0))                    # slight curtaining across the grain

    # runs: thin trail ending in a fat bead at the -U end
    run = t.zeros()
    rwob = (t.noise(0.03, seed=15, stretch=(3.0, 1.0)) - 0.5) * 0.004
    for (ru, rv, L) in RUNS:
        dx = _wrapd(X - ru * W, W)
        dy = _wrapd(Y - rv * H, H) + rwob
        s = -dx / L
        ins = smoothstep(-0.02, 0.10, s) * (1 - smoothstep(0.98, 1.02, s))
        wt = 0.0022 + 0.0014 * sat(s)
        trail = sat(1 - (dy / wt) ** 2) * ins * (0.35 + 0.3 * sat(s))
        bead = np.sqrt(sat(1 - ((dx + L) / 0.0075) ** 2 - (dy / 0.0042) ** 2))
        run = np.maximum(run, np.maximum(trail, bead))
    run = t.blur(run, 0.0009)

    # ------------------------------------------------------------------ 3. age
    crack, cgate = fibre_cracks(0.022, 40.0, 0.06, 0.60, 0.88, 0.0013, 20)   # film split along the grain
    crack2, _ = fibre_cracks(0.008, 34.0, 0.03, 0.62, 0.88, 0.0009, 22)
    crack = np.maximum(crack, crack2 * 0.75)
    # cross crazing: short ticks at right angles near the long cracks
    tick = smoothstep(0.80, 0.93, t.noise(0.004, seed=24, stretch=(1.0, 3.0))) * smoothstep(0.02, 0.2, t.blur(crack, 0.004))
    # flakes: angular bits lost beside the cracks
    cl = t.cells(0.007, 0.95, seed=25, aspect=3.0)
    near = t.blur(crack, 0.004)
    chip_sel = (cl.rand(1) < 0.5 * smoothstep(0.05, 0.13, near)) & (cl.rand(2) < 0.16)
    chip = chip_sel.astype(np.float32) * smoothstep(0.0, 0.0012, t.warp(cl.dist, 0.0015, 0.01, seed=26))
    chip_lip = sat(t.blur(chip, 0.0014) * 2.2) * (1 - chip)

    # ------------------------------------------------------------------ height (metres)
    film = (bristle - 0.5) * 0.00026 * b_amt + (lean - 0.5) * 0.00016 + load * 0.00006 + lap * 0.00008 * (0.4 + h3)
    film = film + rope * 0.00006 + (sag - 0.5) * 0.00012 + run * 0.00055
    height = (wood_h - 0.3) * 0.00015 + film
    height = height + (t.fbm(0.3, 3, 0.5, seed=16, stretch=(2.0, 1.0)) - 0.5) * 0.0006      # board not dead flat
    height = height - crack * 0.00022 - tick * 0.00008
    height = height + chip_lip * 0.00008
    wood_tex = t.fibres(0.004, 26.0, seed=27)
    height = mix(height, height - 0.00020 + (late - 0.4) * 0.00010 + (wood_tex - 0.5) * 0.00010, chip)
    dust_spk = t.speckle(density=9000, size=0.0022, seed=17)                  # nibs / dust caught in the wet coat
    height = height + dust_spk * 0.00010

    # ------------------------------------------------------------------ colour
    cav = t.cavity(height, 0.003, 0.00012)
    yellow = t.fbm(0.35, 3, 0.5, seed=18, stretch=(1.5, 1.0))
    thin = sat(0.5 - (lean - 0.5) * 1.6 - (bristle - 0.5) * 0.9 * b_amt)     # thinner film: undercoat greys through
    albedo = ramp(sat(0.5 + 1.2 * (yellow - 0.5) + 0.25 * (h2 - 0.5)),
                  [(0.0, "#c3bdaa"), (0.5, "#d0ccbe"), (1.0, "#d9d6cc")])
    albedo = albedo * (1.0 - 0.10 * smoothstep(0.45, 0.95, thin) - 0.035 * late)[..., None]
    grime = t.fbm(0.12, 4, 0.55, seed=19, stretch=(3.0, 1.0))
    albedo = mix(albedo, albedo * hexlin("#b3a996") * 1.6, sat(cav * (0.45 + 0.5 * grime)) * 0.55)   # dust in the grooves
    fly = t.speckle(density=2600, size=0.0022, seed=28) * smoothstep(0.4, 0.7, t.fbm(0.15, 2, 0.5, seed=29))
    albedo = mix(albedo, hexlin("#6d6558"), fly * 0.45)
    albedo = mix(albedo, hexlin("#5a5145"), sat(crack * 0.8 + tick * 0.35))
    wood_col = ramp(sat(0.5 + (wood_tex - 0.5) * 0.9 - late * 0.3), [(0, "#6b5b48"), (0.5, "#84725c"), (1, "#998770")])
    albedo = mix(albedo, wood_col, chip)
    albedo = mix(albedo, albedo * 0.93, chip_lip * 0.6)

    # ------------------------------------------------------------------ roughness
    chalk = smoothstep(0.35, 0.8, t.fbm(0.18, 4, 0.5, seed=30, stretch=(2.0, 1.0)))       # chalked / rubbed zones
    rough = 0.36 + 0.20 * chalk + 0.10 * (thin - 0.5) + 0.22 * cav + 0.05 * (grime - 0.5)
    rough = rough - 0.08 * run + 0.25 * sat(crack + tick)
    rough = mix(rough, 0.82, chip)
    rough = np.clip(rough, 0.22, 0.95)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
