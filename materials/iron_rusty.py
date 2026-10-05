"""Heavily rusted iron: laminated scale that has delaminated in plates, pitted and blistered.

Rust grows in layers and jacks itself off the metal: old thick plates (dark, compact, purple-brown)
stand proud with cracked, lifted edges; where they have spalled the younger rust below is orange
and powdery; the exposed lows are eaten by pits ringed with fresh orange, and tubercles (blisters)
push up through the scale. All oxide: metal 0 except a few rubbed dark-iron specks.
"""
from tk.tex import *

META = {"normal_strength": 1.5}


def build(t):
    e = 1.5 * t.px / 0.0006                                            # edge softness factor (~1.5 px)

    # ---------------------------------------------------------------- rust-jacked plates (terraces)
    und = (t.fbm(0.15, 3, 0.5, seed=1) - 0.5) * 0.0008
    lam = t.fbm(0.075, 2, 0.5, seed=2)                                 # where the scale has survived
    jag = t.fbm(0.007, 3, 0.6, seed=4)                                 # ragged broken edges
    ths = (0.36, 0.47, 0.57)
    thick = (0.00050, 0.00060, 0.00070)
    lev = t.zeros()
    plates = t.zeros()
    rim = t.zeros()
    for i, (th, tk_) in enumerate(zip(ths, thick)):
        own = t.fbm(0.045, 3, 0.5, seed=10 + i)                        # each lamina breaks along its own line
        f = lam * 0.55 + own * 0.45 + (jag - 0.5) * 0.06
        s = smoothstep(th, th + 0.012 * e, f)
        plates += s * tk_
        lev += s
        lip = smoothstep(th, th + 0.02, f) * (1 - smoothstep(th + 0.03, th + 0.10, f))
        rim = np.maximum(rim, lip)                                     # lifted lip just inside each edge
    lev = lev / len(ths)                                               # 0 = deepest exposed, 1 = oldest plate
    top = smoothstep(0.5, 0.9, lev)
    low = 1 - smoothstep(0.0, 0.45, lev)

    # ---------------------------------------------------------------- cracks in the thick plates
    crk = t.cracks(0.03, 0.0008, seed=20, wander=0.004, density=0.7) * smoothstep(0.4, 0.7, lev)

    # ---------------------------------------------------------------- pits (in the exposed lows) + tubercles
    pz = smoothstep(0.35, 0.7, t.fbm(0.06, 3, 0.5, seed=21))
    p1 = t.speckle(density=4500, size=0.0065, seed=22, soft=0.7)
    p2 = t.speckle(density=16000, size=0.0032, seed=23, soft=0.7)
    p3 = t.speckle(density=60000, size=0.0015, seed=24, soft=0.6)
    pits = sat((p1 * 1.0 + p2 * 0.7) * (0.15 + 0.85 * sat(low + 0.5 * (1 - top))) * (0.25 + 0.75 * pz)
               + p3 * 0.35 * (0.3 + 0.7 * low))
    tb = t.cells(0.012, 1.0, seed=25)
    tsel = (tb.rand(1) < 0.10 + 0.3 * pz * (1 - low)).astype(F)
    tub = tsel * np.sqrt(sat(1 - (tb.f1 / (0.0022 + 0.0026 * tb.rand(2))) ** 2))
    tub = t.blur(tub, 0.0005)

    # ---------------------------------------------------------------- granular rust skin
    grit = t.fbm(0.0028, 3, 0.65, seed=26)
    lump = t.fbm(0.012, 3, 0.55, seed=27)
    skin = (grit - 0.5) * 0.00014 + (lump - 0.5) * 0.00032

    height = und + plates + rim * 0.00022 + skin * (1 - 0.35 * top) + tub * 0.00045
    height = height - pits * 0.00075 - crk * 0.00030

    # ---------------------------------------------------------------- masks
    cav = t.cavity(height, 0.0030, 0.00030)
    edge = t.edges(height, 0.0018, 0.00018)
    blot = t.fbm(0.035, 4, 0.55, seed=28)
    fresh = sat(low * (0.45 + 0.6 * blot) + (1 - top) * (1 - low) * smoothstep(0.55, 0.8, blot) * 0.5)   # young orange rust
    powder = smoothstep(0.55, 0.85, t.fbm(0.02, 4, 0.6, seed=29)) * (1 - top * 0.8)                      # loose ochre bloom
    damp = t.fbm(0.2, 3, 0.5, seed=30)

    # ---------------------------------------------------------------- colour (linear)
    v = sat(0.5 + (grit - 0.5) * 0.5 + (lump - 0.5) * 0.5 + (blot - 0.5) * 0.7)
    c_old = ramp(v, [(0.0, "#241816"), (0.35, "#33221d"), (0.65, "#412a21"), (1.0, "#523326")])      # compact old scale
    c_mid = ramp(v, [(0.0, "#35201a"), (0.35, "#4a2a1c"), (0.65, "#5f361f"), (1.0, "#744424")])      # red-brown
    c_new = ramp(v, [(0.0, "#55301c"), (0.35, "#703e20"), (0.7, "#8a5025"), (1.0, "#a0662f")])       # fresh orange
    albedo = mix(c_mid, c_old, top)
    albedo = mix(albedo, c_new, fresh * 0.9)
    albedo = mix(albedo, hexlin("#a8793f"), powder * 0.35)                                            # ochre dust
    albedo = mix(albedo, hexlin("#a87238"), sat(rim * 0.45 * (1 - low) + edge * 0.2) * (0.3 + 0.7 * jag))   # broken edges
    ring = sat(t.blur(pits, 0.0012) * 2.2 - pits * 1.5)
    albedo = mix(albedo, hexlin("#96531f"), ring * 0.45)                                              # orange halo round pits
    albedo = mix(albedo, hexlin("#1c1310"), sat(pits * 0.85 + crk * 0.8 + cav * 0.3))                 # dark in the holes
    albedo = mix(albedo, hexlin("#6e3d25"), tub * 0.35)
    albedo = albedo * (0.90 + 0.2 * damp)[..., None]
    fleck = t.speckle(density=90000, size=0.0012, seed=32)
    albedo = mix(albedo, hexlin("#b98e4c"), fleck * 0.22 * (1 - top))
    worn = smoothstep(0.75, 1.0, tub) * (tb.rand(4) < 0.3) * 0.7                                      # rubbed tubercle tops
    albedo = mix(albedo, hexlin("#4a4644"), worn)

    rough = 0.86 + 0.10 * (grit - 0.5) - 0.24 * top * (0.6 + 0.8 * (1 - lump)) + 0.08 * fresh + 0.07 * powder
    rough = mix(rough, 0.97, sat(pits + crk))
    rough = mix(rough, 0.5, worn)
    metal = worn * 0.8
    return dict(albedo=albedo, height=height, rough=rough, metal=metal)
