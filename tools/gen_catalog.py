#!/usr/bin/env python3
"""Writes docs/materials.json - the single list of library materials (baked + engine specials).
Edit the table here, then run:  python3 tools/gen_catalog.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# mask responses: what the per-vertex grime / wear / special masks do to this material
G_DIRT = {"color": "#3a3128", "rough": 0.95}
G_SOOT = {"color": "#1d1a17", "rough": 0.9}
G_DAMP = {"color": "#4a4438", "rough": 0.85}


def B(name, family, tile, desc, res=1024, tintable=False, dir=False, grime=G_DIRT, wear=None, special=None, **kw):
    d = {"name": name, "kind": "baked", "family": family, "tile": tile if isinstance(tile, list) else [tile, tile],
         "res": res, "tintable": tintable, "directional": dir, "desc": desc, "grime": grime,
         "wear": wear or {"color": "#b8b0a2", "rough": 0.5, "metal": 0.0},
         "special": special or {"kind": "none"}}
    d.update(kw)
    return d


def S(name, family, desc, tintable=True, **kw):
    d = {"name": name, "kind": "special", "family": family, "tintable": tintable, "desc": desc, "tile": [1, 1]}
    d.update(kw)
    return d


MOSS = {"kind": "moss", "color": "#46531f", "rough": 1.0}
LICHEN = {"kind": "moss", "color": "#7d8250", "rough": 1.0}
RUST = {"kind": "rust", "color": "#7a3f1c", "rough": 0.92}
VERDIGRIS = {"kind": "verdigris", "color": "#4f9a86", "rough": 0.85}
FADE = {"kind": "fade", "color": "#d8d2c4", "rough": 0.9}
WET = {"kind": "wet", "color": "#000000", "rough": 0.08}
STONE_WEAR = {"color": "#bdb6a8", "rough": 0.38, "metal": 0.0}        # polished by feet / hands
WOOD_WEAR = {"color": "#b89468", "rough": 0.55, "metal": 0.0}         # bare pale timber
PAINT_WEAR_WOOD = {"color": "#8d7557", "rough": 0.75, "metal": 0.0}   # paint gone -> grey-brown wood
PAINT_WEAR_IRON = {"color": "#3b2a22", "rough": 0.6, "metal": 0.6}    # paint gone -> dark iron
IRON_WEAR = {"color": "#9a9590", "rough": 0.32, "metal": 1.0}         # burnished bright metal
BRASS_WEAR = {"color": "#e3bb6c", "rough": 0.22, "metal": 1.0}

MATERIALS = [
    # ---------------- ground ----------------
    B("cobble_granite", "ground", 1.6, "Rounded grey-buff granite cobbles in irregular courses, tight sandy joints, foot-polished crowns. THE square paving. special = wet sheen.", res=2048, wear=STONE_WEAR, special=WET, grime=G_DAMP),
    B("sett_basalt", "ground", 1.2, "Dark blue-grey rectangular basalt setts in running bond, crisp but worn. Roadways, gutters, kerb aprons. special = wet.", res=2048, wear=STONE_WEAR, special=WET, grime=G_DAMP),
    B("flagstone_york", "ground", 2.4, "Large warm buff sandstone flags, random rectangular coursing, lipped and cracked. Pavements, terraces, thresholds. special = moss in joints.", res=2048, wear=STONE_WEAR, special=MOSS, grime=G_DAMP),
    B("gravel", "ground", 1.0, "Fine packed gravel and dirt. Yard corners, planting beds, under benches. special = wet.", special=WET),
    B("soil", "ground", 0.6, "Dark crumbly planting soil with bits of bark. Inside pots and planters. special = wet.", special=WET),
    # ---------------- masonry (jointed, for large flat walls) ----------------
    B("ashlar_limestone", "masonry", [2.0, 2.0], "Coursed dressed pale limestone blocks with fine lime joints, tooled faces. Plinths, civic walls, tower bases. special = moss.", res=2048, wear=STONE_WEAR, special=MOSS, grime=G_SOOT),
    B("rubble_fieldstone", "masonry", [2.0, 2.0], "Random rubble wall of mixed grey/ochre field stones in deep recessed mortar. Retaining walls, cellars, back walls. special = moss.", res=2048, wear=STONE_WEAR, special=MOSS, grime=G_SOOT),
    B("brick_red", "masonry", [0.9, 0.9], "Old hand-made red clay brick, running bond (225x75 mm module), lime mortar, colour variation and spalling. special = moss/efflorescence.", res=2048, wear={"color": "#c98d6d", "rough": 0.7, "metal": 0}, special=LICHEN, grime=G_SOOT),
    B("brick_ochre", "masonry", [0.9, 0.9], "Buff/yellow stock brick with dark headers and sooty faces, Flemish-ish bond. special = lichen.", res=2048, wear={"color": "#d4bd8a", "rough": 0.7, "metal": 0}, special=LICHEN, grime=G_SOOT),
    # ---------------- stone (unjointed: for individually modelled blocks) ----------------
    B("stone_smooth", "stone", 1.0, "Fine-grained dressed stone face without joints, subtle tooling and veining. Use on individually modelled blocks: sills, copings, fountain bowls, steps, quoins, statues. Tintable (default warm grey limestone). special = moss.", tintable=True, wear=STONE_WEAR, special=MOSS, grime=G_SOOT),
    B("stone_rough", "stone", 1.0, "Rough-hewn / split stone face without joints, pitted and chiselled. Kerbs, bollards, rubble blocks, boulders, plinth courses. Tintable. special = moss.", tintable=True, wear=STONE_WEAR, special=MOSS, grime=G_SOOT),
    B("slate", "stone", 0.8, "Riven blue-black slate with cleavage ridges. Individually modelled roof slates, hearths, chalk-free sign boards. Tintable. special = lichen.", tintable=True, wear={"color": "#8f959b", "rough": 0.4, "metal": 0}, special=LICHEN),
    B("mortar", "stone", 0.8, "Plain sandy lime mortar / render fill. Behind modelled stones, pointing, cement patches. Tintable. special = moss.", tintable=True, special=MOSS, grime=G_SOOT),
    # ---------------- plaster ----------------
    B("plaster_lime", "plaster", 2.0, "Hand-trowelled lime plaster / limewash: soft undulation, trowel arcs, hairline cracks, faint patch repairs. Tintable (white base - pick any wall colour). wear = plaster chipped to pale substrate; special = damp/algae staining.", res=2048, tintable=True, wear={"color": "#cdbfa6", "rough": 0.9, "metal": 0}, special={"kind": "moss", "color": "#5d6338", "rough": 0.95}, grime=G_SOOT),
    B("plaster_rough", "plaster", 2.0, "Coarse roughcast / stucco with aggregate, flaking patches. Tintable (white base). Back streets, upper storeys, chimneys. special = damp/algae.", res=2048, tintable=True, wear={"color": "#b9aa90", "rough": 0.95, "metal": 0}, special={"kind": "moss", "color": "#5d6338", "rough": 0.95}, grime=G_SOOT),
    # ---------------- roofing (patterned sheets, for big roof planes) ----------------
    B("roof_tile_clay", "roof", [1.2, 1.2], "Overlapping hand-made plain clay roof tiles in courses (eaves at V=0, ridge towards +V). Terracotta reds/oranges, moss in laps. For distant / large roofs; hero roofs should model tiles with 'terracotta'. special = moss.", res=2048, wear={"color": "#c98764", "rough": 0.7, "metal": 0}, special=MOSS, grime=G_SOOT),
    B("roof_slate", "roof", [1.5, 1.2], "Coursed blue-grey roof slates, staggered, some slipped / chipped (eaves at V=0). special = lichen.", res=2048, special=LICHEN, grime=G_SOOT),
    B("wood_shingle", "roof", [1.2, 1.0], "Split cedar shingles in courses, silvered and cupped (eaves at V=0). Turrets, porches, stall roofs. special = moss.", res=2048, wear=WOOD_WEAR, special=MOSS),
    B("copper_patina", "roof", [1.2, 1.2], "Weathered copper sheet: brown oxide to streaky verdigris green (the special mask pushes it fully green). Domes, turret caps, flashing, gutters. Plain sheet without seams - model the standing seams.", wear={"color": "#b87348", "rough": 0.35, "metal": 1.0}, special=VERDIGRIS, grime=G_SOOT, metal=True),
    # ---------------- wood (grain along +U) ----------------
    B("wood_oak", "wood", [1.2, 0.6], "Sawn oak, warm mid brown, visible straight grain with rays and the odd knot. Single continuous board (no seams): model your own planks. Tintable for stains (e.g. #6b4a2e dark, #3a2a1e tarred beam, #c8a070 fresh). wear = pale bare wood.", tintable=True, dir=True, wear=WOOD_WEAR, special=MOSS),
    B("wood_weathered", "wood", [1.2, 0.6], "Silver-grey weathered softwood: raised grain, checks and splits, nail stains. Crates, fences, cart beds, shutters left unpainted. Tintable. special = moss/algae.", tintable=True, dir=True, wear={"color": "#a89a86", "rough": 0.6, "metal": 0}, special={"kind": "moss", "color": "#55602d", "rough": 1.0}),
    B("wood_painted", "wood", [1.2, 0.6], "Brush-painted timber: faint grain telegraphing, brush strokes, drips. Tintable (white base - choose the paint colour). The WEAR mask chips the paint back to grey-brown wood, special = sun-fade. Doors, shutters, window frames, shopfronts, signs, benches.", tintable=True, dir=True, wear=PAINT_WEAR_WOOD, special=FADE),
    B("wood_planks", "wood", [1.6, 1.0], "Tongue-and-groove board cladding: ~125 mm boards running along U with seams, nail heads, slight cupping. For big boarded surfaces you do not model plank-by-plank (stall backs, attic gables, doors' inner faces). Tintable. special = moss.", tintable=True, dir=True, wear=WOOD_WEAR, special=MOSS),
    B("wood_endgrain", "wood", [0.3, 0.3], "End grain: growth rings and radial checks. Beam ends, log ends, barrel heads' edges, chopping blocks. Tintable.", tintable=True, wear=WOOD_WEAR),
    B("bark", "wood", [0.6, 1.2], "Deeply furrowed grey-brown tree bark, ridges running along V (up the trunk). Tree trunks and branches. special = moss/lichen.", special=LICHEN),
    B("wicker", "wood", [0.24, 0.24], "Woven willow basketry: horizontal weavers over vertical stakes (U horizontal). Baskets, hampers, chair seats, cloches. Tintable.", tintable=True, wear={"color": "#c9a878", "rough": 0.5, "metal": 0}),
    # ---------------- metal ----------------
    B("iron_wrought", "metal", 0.5, "Hand-forged black iron: hammer facets, fire scale, blackened/oiled finish. Brackets, scrolls, railings, hinges, lantern frames, chains. wear = burnished bright steel; special = rust.", wear=IRON_WEAR, special=RUST, metal=True, grime=G_DIRT),
    B("iron_painted", "metal", 0.6, "Thickly painted cast iron: sand-cast orange-peel under glossy paint, chips and runs. Tintable (white base - classic colours: #1f2a22 green-black, #20232a, #5a1414 red, #1c2f4a blue). Lamp posts, benches, bollards, pumps, pillar boxes. WEAR chips to dark iron; special = rust bleeding.", tintable=True, wear=PAINT_WEAR_IRON, special=RUST),
    B("iron_rusty", "metal", 0.6, "Heavily rusted, pitted, flaking iron in oranges and dark browns. Old tools, anchors, drain grates, neglected fittings. wear = dark polished iron.", wear={"color": "#5b5450", "rough": 0.4, "metal": 1.0}, special=RUST),
    B("zinc", "metal", 0.6, "Dull grey galvanised / zinc sheet with spangle and white bloom. Buckets, watering cans, gutters, downpipes, roof flashing. special = rust.", wear=IRON_WEAR, special=RUST, metal=True),
    B("brass", "metal", 0.4, "Aged brass: warm yellow metal with brown tarnish in recesses and fine polishing lines. Tintable (#ffffff brass; #f2c9a0 copper-ish; #e6d9c0 nickel). Lantern fittings, door furniture, bells, clock bezels, taps. wear = bright polished; special = verdigris.", tintable=True, wear=BRASS_WEAR, special=VERDIGRIS, metal=True),
    B("bronze_aged", "metal", 0.6, "Dark statuary bronze, chocolate-brown patina with green in recesses and rubbed golden highlights. Statues, fountain spouts, plaques, bells. special = verdigris.", wear={"color": "#c9964f", "rough": 0.25, "metal": 1.0}, special=VERDIGRIS, metal=True),
    B("lead", "metal", 0.5, "Soft dull blue-grey lead sheet with white oxide bloom and dents. Flashing, window cames, cistern linings, roof finials.", wear={"color": "#a7abb0", "rough": 0.35, "metal": 1.0}, metal=True),
    # ---------------- fabric & soft ----------------
    B("canvas", "fabric", [0.25, 0.25], "Heavy cotton canvas, plain weave, slight slub and creasing. Tintable (off-white base). Awnings, market stall covers, sacks, tarps, banners. Optional stripes: mat('canvas', '#a33', stripe=0.12, tint2='#eee4cf') paints stripes across U. special = sun-fade.", tintable=True, special=FADE, wear={"color": "#d9d2c0", "rough": 0.9, "metal": 0}),
    B("burlap", "fabric", [0.15, 0.15], "Coarse open-weave jute sacking, hairy. Sacks, wrapped bundles, root balls. Tintable.", tintable=True, special=FADE),
    B("rope", "fabric", [0.06, 0.03], "Three-strand twisted hemp rope: U runs ALONG the rope (sweep() gives this), strands spiral. Lashings, well ropes, mooring coils. Tintable.", tintable=True, dir=True),
    B("leather", "fabric", 0.4, "Worn vegetable-tanned leather: fine pebbled grain, creases, darkened handling. Tintable (#7a4a2a tan default look, #2a1c14 dark). Straps, luggage, bellows, book spines, seats. wear = pale scuffs.", tintable=True, wear={"color": "#c79a6b", "rough": 0.5, "metal": 0}),
    B("paper", "fabric", 0.5, "Aged laid paper / parchment: fibres, foxing, soft creases. Tintable (cream base). Posters, notices, lantern shades, labels, maps. special = damp stains.", tintable=True, special={"kind": "moss", "color": "#8a7a4f", "rough": 0.95}),
    # ---------------- ceramic / misc ----------------
    B("terracotta", "ceramic", 0.5, "Unglazed fired clay: orange-buff, throwing lines, lime bloom, small chips. Tintable. Flower pots, chimney pots, individually modelled roof tiles, ridge tiles, urns. special = moss/algae.", tintable=True, wear={"color": "#d89b72", "rough": 0.7, "metal": 0}, special=MOSS),
    B("ceramic_glazed", "ceramic", 0.4, "Glossy glazed ceramic / enamel with fine crazing and pooled glaze. Tintable (white base). Tiles, jugs, crockery, enamel signs, glazed pots, insulators. wear = chipped to biscuit clay.", tintable=True, wear={"color": "#c4a07a", "rough": 0.8, "metal": 0}),
    B("chalkboard", "misc", 0.8, "Black slate chalkboard with wiped chalk dust smears and ghosted writing. Cafe menu boards, A-frames. Tintable (dark green boards).", tintable=True, wear={"color": "#9aa0a0", "rough": 0.8, "metal": 0}),
    B("wax", "misc", 0.3, "Candle wax / tallow: soft translucent ivory with drips. Tintable. Candles, seals.", tintable=True, sss=True),
    B("moss", "plant", 0.5, "Dense cushion moss, deep green to yellow-green. Use as a solid material on modelled moss clumps / tufts between stones."),
    B("hedge", "plant", 0.6, "Dense small-leaf foliage surface (box / privet): layered tiny leaves with dark gaps. For clipped hedge volumes, topiary shells, dense ivy masses, far tree canopies. Tintable (autumn / bluish).", tintable=True),
    B("leaf", "plant", [0.25, 0.25], "Single broad leaf blade: midrib, veins, slight gloss; the albedo is a generic green blade filling the tile - map each modelled leaf / petal quad to the whole tile or model leaf outlines in geometry. Tintable (greens, autumn colours, petals: any flower colour).", tintable=True, sss=True),
    # ---------------- engine specials (no textures; defined in web/engine) ----------------
    S("plain", "special", "Matte neutral surface in the tint colour with fine micro-roughness. LAST RESORT for things with no better library material (rubber, bone, plastic-like bakelite). Assets made mostly of 'plain' are rejected."),
    S("metal_plain", "special", "Clean metal in the tint colour (gold leaf #d4a84b, chrome #dddddd, tin, gunmetal). params: rough (default 0.35)."),
    S("glass_clear", "special", "Clear window / bottle glass, thin, slightly wavy, reflective. Tint = glass colour (#ffffff clear, #9fd6c0 bottle green, #7a4a1e amber). params: opacity (default 0.25), rough."),
    S("glass_frosted", "special", "Frosted / etched glass: bright diffusing surface. Lamp globes and shades. Tint = colour; params: glow (emissive strength if lit from inside, 0 = unlit), opacity."),
    S("glass_lantern", "special", "Old seeded lantern glass, faint amber, lit from within (subtle glow so the panes read warm at night). params: glow (default 1.2), opacity (default 0.35)."),
    S("glass_stained", "special", "Stained glass panel glowing from interior light; tint = glass colour per piece (use several mats). params: glow (default 2)."),
    S("window_lit", "special", "Glazing of an OCCUPIED room at night: warm interior seen through slightly wavy glass with a faked room depth (parallax), curtains and soft variation per pane. Tint = interior light colour (default #ffc37a). params: glow (default 3), curtain (0..1 how drawn the curtains are). Put behind modelled glazing bars."),
    S("window_dark", "special", "Glazing of a dark room: near-black, glossy, reflects the lantern light and sky, faint curtain shapes. For unlit windows (mix lit and dark ones!)."),
    S("emissive", "special", "Self-luminous surface: flames, gas mantles, bulb filaments, glowing embers, lit signs. Tint = colour (#ffb45e flame, #ffd9a0 mantle, #fff1d0 bulb). params: glow (2 = dim ember, 8 = lantern flame, 20+ = dazzling bulb). Always pair with a light() socket."),
    S("neon", "special", "Neon / glass tube lettering: saturated emissive core with soft falloff. Tint = gas colour (#ff3b6b, #37e0ff, #ffd24a). params: glow (default 10)."),
    S("water", "special", "Moving water surface: animated ripples, reflective, slightly green-dark. Fountain basins, troughs, puddles, canal. params: flow (ripple speed, default 1), tint for depth colour, opacity."),
    S("water_fall", "special", "Falling / jetting water sheet or stream: streaky, semi-opaque white-blue, scrolls along +U. Fountain jets and spill sheets, gargoyle spouts. params: flow."),
]


def main():
    names = [m["name"] for m in MATERIALS]
    assert len(names) == len(set(names))
    # keep baked previews (mean albedo) from the last bake
    prev = {}
    idx = os.path.join(ROOT, "out", "mat")
    if os.path.isdir(idx):
        for n in os.listdir(idx):
            p = os.path.join(idx, n, "mat.json")
            if os.path.exists(p):
                prev[n] = json.load(open(p)).get("preview")
    for m in MATERIALS:
        m["preview"] = prev.get(m["name"]) or m.get("preview") or [0.5, 0.5, 0.5]
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "materials.json"), "w") as f:
        json.dump({"materials": MATERIALS}, f, indent=1)
    print(len(MATERIALS), "materials;", sum(1 for m in MATERIALS if m["kind"] == "baked"), "baked")


if __name__ == "__main__":
    main()
