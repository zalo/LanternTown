#!/usr/bin/env python3
"""Source of docs/catalog.json - the asset catalogue (run: python3 tools/catalog_src.py)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = []


def a(tier, cat, id, name, cls, mount, brief, deps=(), ref="plaza"):
    C.append({"id": id, "category": cat, "name": name, "class": cls, "mount": mount, "tier": tier,
              "brief": " ".join(brief.split()), "deps": list(deps), "ref": ref})


# ------------------------------------------------------------------ lighting
a(0, "lighting", "lantern_head", "Gas lantern head", "component", "ground", "Four-sided tapered glazed gas lantern head (exemplar).")
a(0, "lighting", "lamp_post_gas", "Gas lamp post", "street_furniture", "ground", "Cast-iron gas lamp post, 1-3 lanterns (exemplar).", ["lantern_head"])
a(1, "lighting", "wall_lantern", "Wall lantern on scroll bracket", "component", "wall",
  """Wrought-iron wall bracket (back plate bolted to the wall, S-scroll or swan-neck arm, stay bar) carrying a glazed lantern, either
  standing on the arm or hanging from it. As on the gate arch and shopfronts of the reference. Params: reach, bracket style
  (scroll / swan / gallows), lantern roof style, paint, intensity. Compose the lantern with use("lantern_head").""", ["lantern_head"], "gate")
a(1, "lighting", "hanging_lantern", "Hanging chain lantern", "component", "hang",
  """Lantern hanging from a ceiling rose / hook on a chain (use chain()), for archways, the canal tunnel and timber galleries.
  Cylindrical or hexagonal cage style distinct from lantern_head: turned cap, pierced gallery, glass cylinder, drip finial. The lantern
  body is a 'swing' part. Params: drop length, style (hex / cylinder / globe), metal (iron / brass / copper), glass tint, intensity.""", (), "canal")
a(1, "lighting", "ground_lantern", "Stone ground lantern", "prop", "ground",
  """Squat ground lantern as flanking the monument steps: dressed stone base and shaft, iron or bronze cage with four glazed openings,
  capped roof with finial; 0.6-1.1 m tall, warm glow at knee height. Params: height, plan (square / octagonal), cap style, material
  (stone / bronze), intensity, moss.""", (), "monument")
a(1, "lighting", "festoon_lights", "Festoon string lights", "component", "hang",
  """A catenary string of festoon bulbs between two fixing points (origin = first point; span and drop are params): twisted cable, bulb
  holders, clear or coloured bulbs with emissive filaments, a few light sockets spread along it (not one per bulb: 1 per ~1.5 m).
  Params: span, sag, bulb count, bulb colours (warm / multicolour), bulb size. 'swing' not needed.""", (), "cafe")
a(1, "lighting", "paper_lantern", "Paper lantern", "prop_small", "hang",
  """Hanging paper / cloth lantern: ribbed bamboo-hoop paper shade (globe, barrel or gourd profile via lathe with rib ridges), wooden
  top and bottom rings, cord and tassel, glowing from inside (glass_frosted or paper with glow), 'swing' part.
  Params: shape, diameter, colour (cream / red / amber), tassel, intensity.""")
a(1, "lighting", "table_lantern", "Table candle lantern", "prop_small", "ground",
  """Small table-top candle lantern / hurricane lamp as on the cafe tables: brass or tin base, glass chimney or four small panes, carry
  ring, wax candle with flame (fx flame + emissive), drips. 0.2-0.35 m tall. Params: style (hurricane / box / jar), metal, height,
  intensity.""", (), "cafe")
a(1, "lighting", "neon_sign_vertical", "Vertical neon hotel sign", "component", "wall",
  """Vertical projecting neon sign as the HOTEL sign with crown on the reference: painted sheet-metal box with rolled edges and rivets,
  two wall brackets with stays, big channel letters stacked vertically with neon tubes (mat 'neon') on both faces, a neon emblem on
  top (crown / star / none), cabling, bulbs optional. Params: text, letter height, tube colour, box colour, emblem, bulbs border.
  Lights in the tube colour.""")
a(1, "lighting", "neon_glyph", "Neon emblem sign", "component", "wall",
  """Wall-mounted neon emblem on a dark backing panel or open frame: fleur-de-lis, star, crescent moon, cup, arrow or key outlined in
  one or two neon tube colours (tubes = swept round rods with standoffs and end caps, transformer box, cable). As the neon
  fleur-de-lis on the reference cafe. Params: glyph, size, colours, backing (panel / frame / none).""", (), "cafe")
a(1, "lighting", "lamp_post_globe", "Electric globe lamp post", "street_furniture", "ground",
  """Ornate electric street lamp: fluted cast-iron column on a stepped base, branching arms carrying 1, 3 or 5 frosted glass globes
  (glass_frosted with glow) with gallery collars and finials. Distinct from the gas lamp post: rounder, more Parisian.
  Params: height, globes (1 / 3 / 5), globe diameter, paint, intensity.""")
a(1, "lighting", "wall_sconce_candle", "Candle wall sconce", "prop_small", "wall",
  """Wrought-iron or brass wall sconce with a reflector back plate and 1-3 candles in drip pans on curled arms, wax drips, flames (fx).
  For passages, the cafe gallery and interiors of arches. Params: arms, metal, reflector shape (oval / shield / none), candle height.""", (), "cafe")
a(1, "lighting", "marquee_sign", "Bulb-bordered marquee sign", "component", "wall",
  """Painted fascia sign board with a border of small emissive bulbs and raised lettering (text param), scrolled top pediment, as a
  theatre / shop marquee. Bulbs: instanced, emissive; a few light sockets. Params: text, width, height, board colour, letter colour,
  bulb spacing, pediment style.""")
a(1, "lighting", "brazier", "Iron fire brazier", "prop", "ground",
  """Wrought-iron fire basket on three legs with glowing coals / logs (emissive embers, fx flame + smoke), ash pan, soot. A night
  watchman's brazier for the gate. Params: diameter, height, style (basket / bowl), fuel (coal / logs), intensity.""", (), "gate")

# ------------------------------------------------------------------ street furniture
a(1, "street", "bench_park", "Cast-iron and timber park bench", "furniture", "ground",
  """Victorian park bench as in the plaza: two or three ornate cast-iron end frames (scrolled arm, curved back, splayed feet) joined by
  timber seat and back slats with dome bolts; slats slightly uneven and worn. Params: length, slats, frame style (scroll / serpent /
  plain), paint, wood tone, wear.""")
a(1, "street", "bench_stone", "Stone bench", "furniture", "ground",
  """Dressed stone bench: thick slab seat on two carved scroll or lion-paw supports, optional backrest slab with panel moulding;
  chipped arrises, moss in the joints, worn seat. Params: length, back (bool), support style, stone tint, moss.""", (), "monument")
a(1, "street", "bollard", "Cast-iron bollard", "prop_small", "ground",
  """Cast-iron street bollard: stepped base, fluted or plain tapering shaft, moulded neck and cap (cannon / ball / acorn), optional
  chain eyes. Params: height, style, cap, paint, rust.""")
a(1, "street", "chain_barrier", "Post and chain barrier", "prop", "ground",
  """A run of 2-6 short posts (stone or iron) linked by sagging iron chain (chain() along catenary()). Origin at the run centre; the
  run is along X. Params: posts, spacing, post material (stone / iron), post height, chain sag.""", (), "monument")
a(1, "street", "litter_bin", "Cast-iron litter bin", "prop", "ground",
  """Victorian litter bin: slatted or cast panelled drum on feet with a hooded or open top, crest badge, hinge and latch, inner zinc
  liner visible. Params: style (slatted / panelled), height, paint, gilt details.""")
a(1, "street", "post_box", "Pillar post box", "prop", "ground",
  """Cast-iron pillar letter box: plinth, cylindrical or hexagonal body, door with hinges, lock and collection plate, letter slot with
  hood, moulded cap with beaded rim and finial. Painted (red / blue / green) with chips and rust at the base.
  Params: plan (round / hex), height, paint, cap style, text on the plate.""")
a(1, "street", "water_pump", "Cast-iron hand pump and trough", "prop", "ground",
  """Village hand water pump: fluted iron body, long curved handle on a pivot (a 'swing'-less static part is fine), spout with bucket
  hook, stone or timber trough below with water (mat 'water'), wet staining. Params: height, trough (stone / wood / none), paint.""")
a(1, "street", "fire_hydrant", "Fire hydrant standpipe", "prop_small", "ground",
  """Old cast-iron hydrant / standpipe: flanged base with bolts, barrel, two capped outlets with chains, pentagon operating nut, dome
  cap. Chipped red or green paint, rust runs. Params: height, outlets, paint.""")
a(1, "street", "signpost_finger", "Finger signpost", "street_furniture", "ground",
  """Cast-iron finger post: fluted column on a base, 2-4 pointing finger boards with raised lettering (text params, e.g. 'First
  District', 'Hotel', 'Canal', 'Gate') at different heights and bearings, ball or pineapple finial.
  Params: height, arms, texts, paint, letter colour.""")
a(1, "street", "street_clock", "Street clock on a post", "street_furniture", "ground",
  """Pedestal street clock: ornate iron column on a panelled base, two-faced drum clock head with glazed, softly lit dials (roman
  numerals as raised / painted marks, hands), scroll brackets and crest. Params: height, dial diameter, time (hour), paint, lit.""")
a(1, "street", "notice_board", "Notice board with posters", "prop", "wall_ground",
  """Timber notice board on two posts with a small pitched cap (lead or shingle), boards covered in overlapping pinned paper posters
  and notices (paper material, tinted, curled corners, some torn). Params: width, posters count, mount (freestanding posts / wall),
  roof (bool), poster palette seed.""", (), "gate")
a(1, "street", "drain_cover", "Manhole and drain covers", "prop_small", "ground",
  """Cast-iron manhole cover / gully grate set flush in a stone or iron frame: raised chequer or radial pattern, lifting keyholes,
  founder's lettering ring, rust and polish. Sits 0-10 mm proud of the paving. Params: shape (round / square / gully grate),
  size, pattern, text.""")
a(1, "street", "tree_grate", "Tree pit grate and guard", "prop", "ground",
  """Cast-iron tree grate in four segments with radial slots around a central hole plus a hooped wrought-iron tree guard (vertical bars,
  two rings, flared top). Params: grate size, shape (square / round), guard (bool), guard height, paint.""")
a(1, "street", "hand_cart", "Vendor hand cart", "furniture", "ground",
  """Two-wheeled wooden costermonger's barrow: spoked wheels with iron tyres, plank bed with side rails, shafts / handles, prop legs,
  loaded with crates of produce or flowers; optional striped canvas canopy on four poles. Params: length, load (fruit / flowers /
  sacks / empty), canopy (bool), paint.""")
a(1, "street", "market_stall", "Market stall with awning", "structure", "ground",
  """Timber-framed market stall: trestle counter with drape, back shelves, sloping striped canvas awning on poles with scalloped
  valance and guy ropes, hanging produce / goods, a hanging lantern. Params: width, depth, canvas colours, goods (fruit / bread /
  flowers / trinkets), valance style.""", (), "cafe")
a(1, "street", "planter_urn", "Stone urn planter", "prop", "ground",
  """Classical carved stone urn on a pedestal: gadrooned bowl, egg-and-dart rim, handles optional, square plinth; planted with a
  clipped shrub or trailing flowers (hedge / leaf materials, flower blobs as small tinted meshes). Params: height, bowl style,
  planting (topiary / flowers / fern / empty), stone tint, moss.""")
a(1, "street", "planter_box", "Timber planter box", "prop", "ground",
  """Panelled wooden planter (Versailles tub): corner posts with ball finials, framed plank panels, iron bands, soil, planted with a
  small tree, topiary cone or flowers. Params: size, paint, planting, finials (bool).""")
a(1, "street", "flower_pots", "Terracotta pot cluster", "prop", "ground",
  """A cluster of 3-7 terracotta pots and a trough of different sizes (lathe with rolled rims, drainage saucers), some glazed, with
  flowering plants, herbs and a trailing ivy; one broken or tipped. Params: count, spread, flower colours, glazed ratio.""", (), "cafe")
a(1, "street", "barrel", "Oak barrel", "prop", "ground",
  """Coopered oak barrel: individual bulged staves with gaps, iron hoops with rivets, head with chime, bung and tap; variants standing,
  on its side on a cradle, half-barrel planter, or with a plank top as a table (as on the reference cafe terrace).
  Params: height, pose (stand / side / half / table), hoops, wood tone, rust.""", (), "cafe")
a(1, "street", "crate_stack", "Wooden crates", "prop", "ground",
  """Stack of 2-6 slatted wooden shipping crates and boxes of varying size, nailed battens, rope handles, stencilled marks, a lid ajar
  with straw; skewed stacking from the seed. Params: count, base size, stencil text, straw (bool), wood tone.""")
a(1, "street", "sack_pile", "Grain sack pile", "prop", "ground",
  """Pile of 3-8 filled burlap sacks (soft sagging shapes: subdivided, slumped, tied necks with rope, stitched seams), one open showing
  grain or coffee beans. Params: count, sack size, pile style (heap / leaning / stacked), contents.""")
a(1, "street", "luggage_stack", "Trunks and suitcases", "prop", "ground",
  """Stack of travel luggage as outside the 'Luggage' shop: steamer trunk with timber slats, brass corners and locks, leather suitcases
  with straps, buckles and handles, a hat box and a carpet bag on top; labels. Params: pieces, palette (tan / oxblood / green),
  stack style, wear.""")
a(1, "street", "cafe_table", "Cafe table", "furniture", "ground",
  """Round bistro table: cast-iron tripod pedestal base with scrolled feet, marble or timber top with brass rim; optional tablecloth
  (draped, folds), with a param-driven table setting (cups, saucers, bottle, small vase). Params: diameter, top (marble / wood /
  cloth), base style, setting (none / coffee / wine).""", (), "cafe")
a(1, "street", "cafe_chair", "Cafe chair", "furniture", "ground",
  """Bentwood / wrought-iron bistro chair: curved back hoop, round cane or slatted seat, splayed legs with a stretcher ring; variant
  folding slatted park chair. Thin elegant members with real thickness. Params: style (bentwood / iron / folding), seat (cane /
  wood), paint or wood tone.""", (), "cafe")
a(1, "street", "parasol", "Cafe parasol", "furniture", "ground",
  """Large canvas market parasol: timber pole in a cast-iron base, hub with spreading ribs, octagonal canopy panels with scalloped
  valance and stitched seams (slight sag between ribs), vent cap, tie cord. Params: diameter, height, canvas colours (plain /
  striped), open amount, valance.""", (), "cafe")
a(1, "street", "menu_board", "A-frame menu board", "prop_small", "ground",
  """Hinged timber A-frame pavement sign with chalkboard panels: chalk lettering (text params as thin raised 'chalk' text), decorative
  chalk border, brass hinges, chain stay, scuffed feet. Params: height, title text, lines of text, frame paint.""", (), "cafe")
a(1, "street", "well_stone", "Stone well with roof", "structure", "ground",
  """Round stone draw-well: coursed curb of individual stone blocks, two timber posts carrying a small shingled or tiled roof, windlass
  with crank handle and coiled rope, hanging bucket, dark water down inside (allow_below_ground). Moss and wet staining.
  Params: diameter, roof (shingle / tile / none), posts (timber / iron arch), bucket.""")
a(1, "street", "bicycle", "Old bicycle", "prop", "ground",
  """Vintage roadster bicycle leaning on its stand: diamond tube frame, spoked wheels (thin spokes, tyres, rims), mudguards, chain guard,
  sprung leather saddle, rod brakes, front wicker basket, small oil lamp. Params: frame paint, basket (bool), lean angle, wheel size.""")
a(1, "street", "wheelbarrow", "Wooden wheelbarrow", "prop", "ground",
  """Timber wheelbarrow with iron-tyred spoked wheel, tapering plank tray, turned handles and legs with iron straps; loaded with soil and
  plants, logs or empty. Params: load, wood tone, paint, wear.""")
a(1, "street", "ladder", "Wooden ladder", "prop_small", "wall_ground",
  """A lamplighter's wooden ladder leaning against a wall: two tapering stiles, round rungs through-tenoned, iron tie rods, worn rung
  centres, rope lashing. Params: length, rungs, lean angle, style (straight / tapering / step-ladder A-frame).""")
a(1, "street", "bucket_set", "Buckets, watering can and broom", "prop_small", "ground",
  """Small cluster: coopered wooden bucket with rope handle, galvanised zinc bucket with wire bail, a zinc watering can with rose, and
  a besom broom leaning (param which items). Water in one bucket. Params: items, arrangement seed.""")
a(1, "street", "milk_churns", "Milk churns", "prop_small", "ground",
  """Two or three steel / zinc milk churns: tapered shoulders, mushroom lids, riveted side handles, dents, stamped numbers.
  Params: count, height, finish (zinc / painted), dents.""")
a(1, "street", "birdbath_sundial", "Sundial / birdbath pedestal", "prop", "ground",
  """Carved stone baluster pedestal carrying either a bronze sundial plate with gnomon and engraved hour lines or a shallow birdbath
  bowl with water. Lichen and weathering. Params: top (sundial / birdbath), height, baluster style, moss.""", (), "monument")
a(1, "street", "pigeons", "Pigeons", "prop_small", "ground",
  """A small group of 2-7 feral pigeons pecking / standing / roosting: smooth organic bodies (subsurf surfaces), folded wings, tail,
  beak, legs; grey-blue with iridescent neck, a white one. Poses and placement from the seed. Params: count, spread, poses.""")
a(1, "street", "cat", "Sitting cat", "prop_small", "ground",
  """A domestic cat, sitting upright or loafing or curled asleep (pose param): convincing organic form with ears, tail wrapped round,
  subtle fur break-up via material choice and tint (black, ginger tabby, grey). Params: pose, coat colour, size.""")
a(1, "street", "rowboat", "Canal rowing boat", "furniture", "ground",
  """Clinker-built wooden rowing skiff moored on the canal: overlapping strakes, ribs, thwarts, rowlocks, oars laid inside, bow ring
  with rope, bilge water, flaking paint and algae line. Origin on the waterline. Params: length, paint, cargo (empty / crates /
  lantern), oars. allow_below_ground.""", (), "canal")
a(1, "street", "mooring_post", "Mooring post and ring", "prop_small", "ground",
  """Canal-side mooring: stout tarred timber pile with iron cap band and rope coil wrapped round it, or a cast-iron mushroom bollard;
  plus an iron ring on a wall staple variant. Params: kind (pile / bollard / ring), height, rope (bool).""", (), "canal")
a(1, "street", "cellar_hatch", "Pavement cellar hatch", "prop", "wall_ground",
  """Sloping double-leaf timber cellar doors set against a wall base on a stone kerb: ledged planks, strap hinges, ring pulls, padlock
  and hasp; one leaf can be propped open showing dark steps. Params: width, open (0..1), paint, wear.""")
a(1, "street", "newspaper_kiosk", "Newspaper kiosk", "structure", "ground",
  """Small octagonal street kiosk: panelled timber base, glazed lit upper with serving hatch and shelf of papers, onion or tent roof in
  zinc / copper with finial and a frieze sign (text param), poster panels. Warm interior light. Params: diameter, roof style,
  paint, sign text.""")

# ------------------------------------------------------------------ signage & decor
a(1, "decor", "hanging_sign", "Hanging shop sign on bracket", "component", "wall",
  """Projecting wrought-iron scroll bracket with a hanging sign ('swing' part on two chain links or hooks): shaped timber board with
  painted / raised emblem and lettering, or a cut-out iron silhouette. As the coffee-cup sign on the reference cafe. Params: emblem
  (cup / boot / key / suitcase / scissors / bottle / book / star), board shape (shield / oval / rect / cutout), text, colours, reach.""", (), "cafe")
a(1, "decor", "fascia_sign", "Painted fascia sign board", "component", "wall",
  """Long painted timber shop fascia with moulded frame, console ends and gilded raised lettering (text param: 'Luggage', 'Items',
  'Accessories', 'Dress Suit'...), optional swan-neck sign lamps above washing it with light. Params: text, width, height, board
  colour, letter colour / gilt, lamps (0-3), frame style.""")
a(1, "decor", "banner_crown", "Hanging banner with crest", "component", "wall",
  """Tall cloth banner hanging from a horizontal pole on wall brackets, as the navy banners with gold crowns beside the gate: hemmed
  fabric with folds and a swallow-tail or pointed bottom, fringe, appliqué emblem (crown / star / fleur / key) in gold, cords with
  tassels; the cloth is a 'flutter' part. Params: width, drop, cloth colour, emblem, bottom shape.""", (), "gate")
a(1, "decor", "bunting", "Bunting string", "component", "hang",
  """A string of triangular or swallow-tail cloth pennants on a rope between two points (origin first point; span, sag params),
  alternating colours from a palette, slight random twist and curl, 'flutter' part. Params: span, sag, flag size, palette, shape.""")
a(1, "decor", "wall_posters", "Pasted wall posters", "prop_small", "wall",
  """A group of 2-6 paper bills pasted and pinned on a wall, overlapping, with curled and torn corners, different sizes and tints,
  simple printed blocks (thin dark rectangles / rules / a big emblem shape as raised 0.3 mm 'ink' geometry) and a headline (text
  param). Params: count, spread, palette, headline, tattered amount.""", (), "gate")
a(1, "decor", "wall_clock", "Bracket wall clock", "component", "wall",
  """Double-sided drum clock projecting from a wall on an ornate iron bracket: glazed softly lit dial with numerals and hands, moulded
  case, crest and drop finial. Params: dial diameter, time, case paint, gilt, lit.""")
a(1, "decor", "weathervane", "Weather vane", "prop_small", "ground",
  """Roof-top weather vane: iron spindle with ball collars, cardinal arms with cut letters N E S W, and a pivoting 'spin' part with a
  cut silhouette (rooster / arrow / ship / star / cat) in gilded copper sheet with real thickness. Origin at the spindle foot.
  Params: motif, height, gilt (bool), verdigris.""")
a(1, "decor", "awning", "Striped shop awning", "component", "wall",
  """Retractable canvas shop awning: roller box on the wall, two folding iron arms, front bar, striped canvas with slight sag and a
  scalloped valance with optional lettering. Params: width, projection, pitch, stripe colours, valance text, scallop style.""")
a(1, "decor", "flag_wall", "Wall flag on angled pole", "component", "wall",
  """Flag on a pole set at 45 degrees in a wall socket bracket: turned pole with gilt ball finial, halyard, draped cloth with deep
  folds and an emblem; 'flutter' part. Params: pole length, flag size, colours, emblem, pattern (plain / bicolour / quartered).""")
a(1, "decor", "plaque", "Inscribed wall plaque", "prop_small", "wall",
  """Stone or bronze wall plaque with moulded frame, engraved multi-line inscription (text params) and a small emblem, four rosette
  fixings; water staining below. As the inscription under the monument star. Params: width, height, material (bronze / stone /
  marble), lines of text, emblem.""", (), "monument")
a(1, "decor", "crest_cartouche", "Heraldic crest cartouche", "prop", "wall",
  """Carved heraldic cartouche for above doors and on the gate: shield with a charge (crown / key / star / lion-ish simple form),
  scrolled strapwork surround, ribbon with motto text, optional gilding and paint. Params: size, charge, material (stone / painted
  wood / bronze), motto.""", (), "gate")
a(1, "decor", "window_box", "Flower window box", "prop_small", "wall",
  """Timber or wrought-iron window box on two brackets, packed with flowers and trailing foliage (clustered small leaf cards with
  thickness or blobs in leaf / hedge materials, flower heads in tinted ceramic_glazed / plain) spilling over the front.
  Params: width, box style (plank / iron basket / terracotta trough), flower colours, fullness.""")
a(1, "decor", "hanging_basket", "Hanging flower basket", "prop_small", "hang",
  """Wire or wicker hanging basket on three chains from a hook, moss-lined, overflowing with flowers and long trailing ivy strands;
  'swing' part. Params: diameter, chain length, flower colours, trailing length.""")
a(1, "decor", "ivy_patch", "Climbing ivy patch", "vegetation", "wall_ground",
  """Ivy climbing a wall from the ground: branching stems (swept, bark) spreading by a seeded growth walk over the wall plane y=0, dense
  overlapping leaf blades (leaf material, small bent quads with thickness feel, varied size and tint) projecting 3-12 cm.
  Params: width, height, density, leaf size, autumn tint amount.""")
a(1, "decor", "shop_mannequin", "Dress mannequin display", "prop", "ground",
  """Tailor's dress form on a turned wooden tripod stand wearing a garment: jester / harlequin diamond-pattern suit with ruff, or a
  tailcoat, or a gown (param), as the jester on the reference 'Dress Suit' shop. Cloth with folds, buttons, stitched seams.
  Params: outfit, colours, height.""")
a(1, "decor", "statue_figure", "Bronze statue on pedestal", "structure", "ground",
  """Civic bronze statue on a stone pedestal with moulded cap and base and an inscription panel: a robed / cloaked standing figure
  holding a lantern or star aloft, built from smooth organic forms (subsurf) with heavy drapery - stylised but dignified; verdigris
  streaks, pigeon-worn head. Params: pose (lantern / star / book), pedestal height, inscription, verdigris.""", (), "monument")

# ------------------------------------------------------------------ architecture modules
a(1, "arch", "window_casement", "Mullioned casement window", "component", "wall",
  """Timber casement window as on every house of the reference: frame and sill, 2-3 lights divided by mullions, small leaded or
  glazing-bar panes (window_lit or window_dark per 'lit' param, each pane slightly different via separate quads), stone or timber
  lintel, projecting sill, optional plank shutters with strap hinges and iron stays, optional curtain. Origin: centre of the sill
  line on the wall plane; the frame sits 0.1 m into the wall (provide a reveal box so it works stuck onto a flat wall).
  Params: width, height, lights, pane grid, lit, shutters, shutter paint, frame paint, head (flat / arched).""")
a(1, "arch", "window_sash", "Sash window with surround", "component", "wall",
  """Tall Georgian sash window: boxed frame, two sashes with glazing bars (six-over-six etc.), meeting rail, stone surround with
  keystone or moulded architrave and bracketed sill, optional small wrought-iron sill guard. window_lit / window_dark.
  Params: width, height, panes, lit, surround (stone / timber / plain), guard (bool), curtain colour.""")
a(1, "arch", "window_oriel", "Oriel bay window", "component", "wall",
  """Projecting timber oriel bay on carved brackets: three-sided or curved glazed bay with leaded lights (lit), moulded sill and cornice,
  small lead or tiled roof, pendant drops. Warm light spilling out (light socket in front). Params: width, height, plan (canted /
  bowed / box), roof (lead / tile), lit, paint.""")
a(1, "arch", "window_stained_arch", "Arched stained-glass window", "component", "wall",
  """Pointed or round-arched stone window with tracery and stained glass (glass_stained in several tints laid out as lead-came panels:
  diamond quarries with a central emblem - star / crown / heart), stone hood mould and sill; glows from within with a coloured light
  socket. As on the clock-tower hall of the reference. Params: width, height, arch (pointed / round), tracery (single / twin / rose),
  palette, emblem.""")
a(1, "arch", "window_round", "Round oculus window", "component", "wall",
  """Circular window: moulded stone or timber ring, radial glazing bars or a quatrefoil, lit glass, optional keystones at the quarters.
  For gables and towers. Params: diameter, bars (radial / quatrefoil / cross), surround, lit.""")
a(1, "arch", "shop_window", "Shopfront display window", "component", "wall_ground",
  """A complete traditional shopfront bay: panelled timber stall-riser, big display window divided by slender turned mullions with a
  transom of small panes above, pilasters with console brackets, fascia space, lit interior display (window_lit glass plus a shelf of
  simple goods silhouettes just behind the glass) and strong warm light spill onto the pavement (light sockets). Origin at the
  pavement centre on the wall plane. Params: width, height, bays, paint, goods (luggage / hats / bottles / books / cakes), lit.""")
a(1, "arch", "door_plank", "Arched plank door", "component", "wall_ground",
  """Heavy ledged oak plank door in a stone or timber frame: vertical boards with V-joints, wrought strap hinges with scrolled ends, ring
  handle, studs in rows, small barred grille, worn threshold step; flat or arched head. Origin at the threshold centre on the wall
  plane. Params: width, height, head (flat / round / pointed), frame (stone / timber), wood tone or paint, studs, ajar angle.""")
a(1, "arch", "door_shop", "Glazed shop door", "component", "wall_ground",
  """Panelled and half-glazed shop door with a rectangular lit fanlight / transom above, brass handle, letter plate and kick plate, bell
  on a spring bracket, 'OPEN' sign card behind the glass (text param), recessed in panelled reveals with one stone step.
  Params: width, height, paint, glazing bars, sign text, lit.""")
a(1, "arch", "door_double_hall", "Grand double door", "component", "wall_ground",
  """Tall double-leaf panelled civic door in a stone doorcase: pilasters or engaged columns, entablature / pediment or arched fanlight
  with radiating bars (lit), bronze lion-ring handles, studded panels, two stone steps. For the hall / hotel entrance.
  Params: width, height, doorcase (pediment / arched / flat hood), paint, steps.""")
a(1, "arch", "dormer", "Roof dormer", "component", "ground",
  """Roof dormer window sitting on a roof slope (origin at the centre of its base on the roof surface; param roof pitch cuts the cheeks
  to fit): timber front with small lit casement, tiled or lead gabled / hipped / shed roof with ridge and bargeboards, lead flashing
  cheeks. Dense lit dormers are a signature of the reference skyline. Params: width, height, roof (gable / hip / shed / eyebrow),
  pitch, lit, cladding (tile / lead / plaster). Built for a roof of `pitch` degrees: geometry extends back to meet the slope.""")
a(1, "arch", "chimney", "Chimney stack with pots", "component", "ground",
  """Brick or stone chimney stack (origin at the centre of its base; it will be sunk into a roof): corbelled oversailing courses, lead
  flashing apron, 1-4 terracotta pots of different designs (plain roll-top, louvred, crown / castellated), soot staining, optional
  smoke fx socket. Individual bricks need not be modelled (brick material) but corners / caps must have real profile.
  Params: width, depth, height, material (brick_red / brick_ochre / stone), pots, lean, soot.""")
a(1, "arch", "balcony_iron", "Wrought-iron balcony", "component", "wall",
  """Small projecting balcony: stone or timber deck slab on two scrolled iron brackets, wrought-iron railing with repeating scroll /
  lyre panels between square posts, moulded handrail, optional flower pots. Origin: centre of the deck's wall edge at floor level.
  Params: width, depth, pattern (scroll / lyre / bars / hearts), paint, pots.""")
a(1, "arch", "balcony_timber", "Timber gallery balcony", "component", "wall",
  """Jettied timber gallery as over the reference cafe: heavy chamfered beams on carved brackets, plank deck, turned or flat-sawn
  baluster rail, posts carrying a lean-to tiled pentice roof (optional). Params: width, depth, balusters (turned / splat / cross),
  roof (bool), wood tone.""", (), "cafe")
a(1, "arch", "drainpipe", "Gutter and downpipe", "component", "wall",
  """Cast-iron rainwater downpipe: ornate hopper head with date / crest, round or square pipe in socketed lengths with eared collars
  bolted to the wall, swan-neck offset at the top, shoe at the bottom discharging over a stone gully; rust streaks and green
  staining. Origin at the top fixing; pipe runs down by 'drop' param. Params: drop, section (round / square), hopper style,
  paint, offset.""")
a(1, "arch", "roof_finial", "Roof finials and ridge cresting", "prop_small", "ground",
  """Roof ornaments: a turned terracotta / lead / iron finial (ball-and-spike, fleur-de-lis, pineapple, star-topped) and a run of
  pierced ridge cresting (param mode finial / cresting / both, cresting length). Origin at the base on the ridge.
  Params: mode, height, material, cresting length, motif.""")
a(1, "arch", "timber_frame_bay", "Half-timbered wall bay", "component", "wall_ground",
  """One bay of half-timbered wall, the signature house wall of the reference: oak sill beam, posts, rails, studs and curved or
  diagonal braces (param pattern: close-studding / square panels / herringbone / St-Andrew's cross / curved braces) pegged at the
  joints with slightly proud pegs, lime-plaster infill panels set 15-25 mm back and slightly pillowed, optional jetty bressummer with
  carved brackets at the top. Origin bottom centre on the wall plane; thickness 0.2 m behind y=0 not needed (front face only +
  returns). Params: width, height, pattern, plaster tint, timber tone, jetty (bool), irregularity.""")
a(1, "arch", "stone_arch", "Stone archway surround", "component", "wall_ground",
  """Dressed stone arch for a passage or gate opening: individually modelled voussoirs with a carved keystone, impost mouldings,
  jamb blocks with chamfered arrises and quoins, optional hood mould and wheel-guard stones at the feet. The opening itself is empty.
  Origin ground centre on the wall plane. Params: opening width, opening height, arch (round / segmental / pointed), depth,
  keystone emblem, moss.""", (), "gate")
a(1, "arch", "buttress", "Stone buttress", "component", "wall_ground",
  """Stepped stone wall buttress with weathered offsets (sloping set-backs with drip mouldings), ashlar quoins, plinth course, mossy
  tops. Params: height, width, projection, stages, moss.""")
a(1, "arch", "column", "Stone column / pilaster", "component", "ground",
  """Classical stone column or square pier with moulded base (plinth, torus, scotia), plain or fluted shaft with entasis and a capital
  (Tuscan / foliate / cushion); pilaster mode = half-depth against a wall. Chips and staining. Params: height, diameter, order,
  fluted, mode (column / pier / pilaster).""")
a(1, "arch", "corbel_bracket", "Carved timber bracket", "prop_small", "wall",
  """Carved oak jetty bracket / console: curved knee brace with chamfer stops, pendant drop, scroll or grotesque-free leaf carving on
  the face, pegged to post and beam. Params: height, projection, style (curved / scroll / plain strut), carving depth.""")
a(1, "arch", "porch_canopy", "Door porch canopy", "component", "wall",
  """Small canopy over a doorway: timber hood on two carved brackets with a lead, tile or shingle roof (gabled, flat or swept),
  bargeboards / fretwork valance, optional hanging lantern hook socket. Params: width, projection, roof (gable / flat / swept),
  covering (tile / lead / shingle), paint.""")
a(1, "arch", "door_steps", "Door stoop steps", "component", "wall_ground",
  """Short flight of 2-5 worn stone steps up to a door with a landing, side cheek walls or simple wrought-iron handrails with scrolled
  ends and a boot scraper. Tread centres dished, joints mossy. Origin ground centre at the wall. Params: steps, width, rail (iron /
  stone cheek / none), stone tint.""")
a(1, "arch", "external_stair", "Timber external staircase", "structure", "wall_ground",
  """Open timber staircase running up along a wall to a first-floor landing / gallery: stringers, plank treads, newel posts with
  caps, balustrade with turned or flat balusters, landing on posts with braces; optional lean-to roof. Params: rise (floor height),
  width, direction (left / right), balusters, roof (bool), wood tone.""", (), "cafe")
a(1, "arch", "roof_gable", "Tiled gable roof module", "component", "ground",
  """A complete pitched roof for a rectangular plan (origin at the centre of the wall-plate rectangle, z = eaves): clay-tile or slate
  slopes with slightly wavy courses (tile material + real edge profile: thick eaves course, verge undercloak, half-round ridge tiles
  as modelled pieces), bargeboards or parapet gables, exposed rafter feet / moulded eaves cornice, gutters. Options: gable / hip /
  half-hip / gambrel, sagging ridge amount. Exposes sockets for dormers and chimneys on the slopes. Params: width, depth, pitch,
  type, overhang, covering (roof_tile_clay / roof_slate / wood_shingle), sag.""")
a(1, "arch", "turret_roof", "Conical turret roof / spire", "component", "ground",
  """Conical, pyramidal or bell-cast spire roof for towers and turrets as the red clock-tower spire: tile / slate / copper covering with
  swept eaves, moulded eaves cornice, optional lucarnes (tiny dormers), ball-and-spike finial with a weathervane socket.
  Origin at the centre of the eaves ring. Params: radius, height, sides (4 / 6 / 8 / round), bellcast, covering, lucarnes.""")
a(1, "arch", "wall_stone_run", "Stone wall segment", "component", "ground",
  """A run of free-standing or retaining stone wall along X (origin at the centre of its base): rubble or ashlar face (material) with
  individually modelled coping stones, plinth course, weep holes, optional end pier; top irregular, moss on the coping, damp at the
  base. Params: length, height, thickness, facing (rubble / ashlar / brick), coping (flat / saddle / half-round), pier ends.""")

# ------------------------------------------------------------------ plaza structures
a(1, "plaza", "balustrade", "Stone balustrade run", "structure", "ground",
  """Classical stone balustrade along X as on the retaining walls and stair sides of the reference plaza: plinth rail, turned bulbous
  balusters (lathe, instanced with tiny random rotations), moulded handrail coping, square piers with panelled faces and ball or
  urn caps at the ends / every N balusters. Chipped, lichen on top. Params: length, height, pier spacing, baluster style
  (vase / double / square), cap (ball / urn / flat), stone tint.""")
a(1, "plaza", "stairs_plaza", "Wide plaza stairs", "structure", "ground",
  """The broad stone staircase connecting two plaza levels (reference centre): 6-14 individually modelled steps with nosings and
  dished worn centres, solid stepped cheek walls with coping and end piers for lamp posts (sockets), optional central landing and
  central handrail in iron. Origin at the centre of the bottom step's front edge; stairs rise toward +Y.
  Params: width, rise (total height), steps, landing (bool), cheek style (stepped / raked), flare (wider at the bottom).""")
a(1, "plaza", "retaining_wall", "Terrace retaining wall", "structure", "ground",
  """Retaining wall of the upper terrace along X: battered ashlar / rubble face with plinth, string course, relieving arches or
  buttress piers at intervals, drain spouts with stains, a coping to carry a balustrade (socket line along the top).
  Origin at the base centre, earth side toward +Y. Params: length, height, facing, pier spacing, arches (bool), moss.""")
a(1, "plaza", "iron_railing", "Wrought-iron railing run", "component", "ground",
  """Railing along X: square bars with forged spear / fleur finials through two flat rails, heavier standards with urn finials and
  back stays every ~1.8 m, on a low stone kerb (optional). As on the canal side of the reference. Params: length, height, bar
  spacing, finial (spear / fleur / ball), kerb (bool), paint, rust.""", (), "canal")
a(1, "plaza", "iron_gate", "Wrought-iron gate with piers", "structure", "ground",
  """Pair of wrought-iron gates hung on stone piers with ball caps: scrollwork panels, dog bars, arched overthrow with a lantern and
  crest, latch; leaves open by an angle param. Params: opening width, height, pier style, overthrow (bool), open angle, paint.""", ["lantern_head"], "gate")
a(1, "plaza", "gate_pier", "Stone gate pier", "structure", "ground",
  """Tall dressed stone pier: plinth, rusticated or panelled shaft of individually cut blocks, cornice cap and a finial (ball, urn,
  pineapple or a lantern socket). Params: height, width, shaft (rusticated / panelled / plain), finial, moss.""", (), "gate")
a(1, "plaza", "fountain_tiered", "Tiered plaza fountain", "landmark", "ground",
  """The plaza's central fountain: octagonal or round stone basin with moulded rim and panelled sides on a stepped kerb, water surface
  (mat 'water'), central baluster stem with two or three scalloped bowls of diminishing size, a finial spout, water sheets falling
  between tiers (mat 'water_fall' strips), lion-mask side spouts, submerged warm lights / small lanterns on the rim (light sockets).
  Wet, mossy, limescale. Params: basin diameter, tiers, plan (round / octagon / quatrefoil), spouts, stone tint, lit.""")
a(1, "plaza", "wall_fountain", "Wall fountain with mask spout", "structure", "wall_ground",
  """Wall-mounted drinking fountain: carved stone back panel with arched niche and a bronze lion / green-man-ish mask spouting a water
  stream (water_fall) into a semicircular stone trough with water, overflow staining, moss. Params: width, height, mask style,
  basin shape, moss.""", (), "monument")
a(1, "plaza", "bridge_stone", "Stone canal bridge", "landmark", "ground",
  """Single-arch humpback stone bridge over the canal as in the reference canal panel: segmental arch ring of voussoirs with keystone,
  rubble spandrels, string course, cobbled deck rising over the arch, parapet walls with coping or iron railings, lamp sockets on
  end piers, wet and mossy near the waterline. Origin at deck centre level with the banks (z=0); the arch and abutments extend
  below (allow_below_ground). Span along X (bridge crosses along X, canal runs along Y). Params: span, width, rise, parapet
  (stone / railing), water depth below.""", (), "canal")
a(1, "plaza", "canal_wall", "Canal quay wall segment", "structure", "ground",
  """Segment of canal edge along X: coping kerb at pavement level (z = 0), ashlar / rubble wall dropping to the water (below ground),
  algae and damp tide line, iron mooring rings, a ladder or narrow stone steps down (param), optional culvert outfall with grate.
  The water plane itself is separate. Origin on the coping edge centre; water side toward -Y. allow_below_ground.
  Params: length, depth, feature (none / ladder / steps / outfall), rings, moss.""", (), "canal")
a(1, "plaza", "tunnel_mouth", "Canal tunnel mouth", "structure", "wall_ground",
  """Arched tunnel portal where the canal disappears under buildings: horseshoe / round stone arch ring with keystone, rusticated
  face wall, a short barrel vault receding into darkness (2-4 m, closed black at the end), a towpath ledge inside and one hanging
  lantern lighting the vault warmly from within (as the lantern-lit tunnel of the reference). allow_below_ground.
  Params: width, height, arch, depth, lantern (bool), moss.""", ["hanging_lantern"], "canal")
a(1, "plaza", "pergola", "Timber pergola / arbour", "structure", "ground",
  """Garden pergola over a path or bench: four to eight chamfered posts on stone pads, beams with shaped ends, cross rafters, diagonal
  braces, climbing vine with leaves over the top and hanging lantern sockets. Params: length, width, height, bays, vine density,
  wood tone / paint.""")
a(1, "plaza", "bandstand", "Octagonal bandstand", "landmark", "ground",
  """Victorian park bandstand: octagonal stone plinth with steps, slender cast-iron columns with ornate spandrel brackets, wrought
  railing, a swept zinc / copper tent roof with cresting and finial, timber boarded soffit with a central hanging lantern and
  festoon bulbs along the eaves. Params: diameter, columns, roof covering, paint, lit.""")
a(1, "plaza", "paving_patch", "Paving feature patch", "ground", "ground",
  """Decorative ground inlay for plaza centres and thresholds, almost flat (0-30 mm relief): radial / fan-pattern cobble rosette or a
  compass star made from individually laid setts and flag segments (real separate stones with joints, slightly uneven), kerb ring.
  Params: diameter, pattern (rosette / compass / concentric / herringbone square), stone tints, unevenness.""")
a(1, "plaza", "kerb_gutter", "Kerb and gutter run", "ground", "ground",
  """A straight (or quarter-circle, param) run of granite kerb stones with a three-row sett gutter channel and an occasional gully
  grate; kerb stones individually cut, slightly misaligned, worn. Along X. Params: length, kerb height, curve radius (0 = straight),
  grate (bool).""")

# ------------------------------------------------------------------ vegetation
a(1, "veg", "tree_plane", "Street plane tree", "vegetation", "ground",
  """Mature deciduous street tree 7-11 m: recursive branching trunk and limbs (swept tapered branches with bark, natural asymmetry,
  flared root buttress), crown built from many small leaf-cluster cards / blobs (leaf / hedge materials, tint variation, enough
  density to read as foliage and catch lantern light from below). Must not look like a lollipop or a cloud of flat planes.
  Params: height, crown width, branchiness, leaf density, season tint (summer / autumn), lean.""")
a(1, "veg", "tree_ornamental", "Small ornamental tree", "vegetation", "ground",
  """Small flowering / clipped tree 3-5 m for plaza corners: slender trunk with stake and tie, rounded or vase crown, blossom option
  (pink / white tinted clusters among the leaves). Params: height, crown shape (round / vase / weeping), blossom colour & amount.""")
a(1, "veg", "tree_cypress", "Columnar cypress", "vegetation", "ground",
  """Tall narrow Italian-cypress-like conifer 4-8 m: dense flame-shaped foliage volume built from many small overlapping tufts (hedge
  material, dark green, tint variation) with an irregular outline and a short visible trunk. Params: height, width, irregularity.""")
a(1, "veg", "topiary", "Clipped topiary in pot", "vegetation", "ground",
  """Box topiary - ball, cone, spiral or two-tier standard (param) - with slightly fuzzy clipped surface (hedge material on a displaced
  volume plus scattered leaf tufts), stem, in a terracotta or lead pot. Params: shape, height, pot style.""")
a(1, "veg", "hedge_run", "Clipped hedge run", "vegetation", "ground",
  """Run of clipped box / privet hedge along X with soft irregular faces (displaced, tufted edge so the silhouette is leafy, not a box),
  bare woody stems visible at the base, optional rounded ends or a low stone kerb. Params: length, height, width, top (flat /
  rounded / castellated), kerb.""")
a(1, "veg", "flower_bed", "Raised flower bed", "vegetation", "ground",
  """Stone-kerbed raised planting bed (rect or round): soil, a mass of low flowering plants in drifts of 2-3 colours, a few taller
  spikes and foliage clumps, built from many small leaf / petal meshes. Params: shape, size, kerb height, colour palette, density.""")
a(1, "veg", "pavement_weeds", "Weeds and moss tufts", "vegetation", "ground",
  """Small ground-level clutter: tufts of grass, dandelion rosettes and moss cushions as grow at wall bases and between cobbles; a
  scatter patch about 0.5-1.5 m across. Params: spread, count, kinds mix, dryness.""")
a(1, "veg", "vine_wisteria", "Wisteria on a wall", "vegetation", "wall_ground",
  """Old wisteria trained up a wall and along a wire: twisted woody trunk from the ground, long horizontal runners, leaves and hanging
  flower racemes (lilac / white) lit beautifully from below. Params: width, height, bloom amount, bloom colour.""")

# ------------------------------------------------------------------ buildings (compose architecture modules)
B = ["window_casement", "window_sash", "window_oriel", "door_plank", "door_shop", "shop_window", "dormer", "chimney", "roof_gable",
     "timber_frame_bay", "wall_lantern", "hanging_sign", "fascia_sign", "window_box", "drainpipe", "balcony_iron", "awning", "door_steps"]
a(2, "building", "house_timber", "Half-timbered town house", "building", "ground",
  """The generic house of the square, fully parametric from width / depth / storeys / seed: stone or brick ground floor with a door
  and windows, one or two jettied half-timbered upper storeys (timber_frame_bay patterns) with lit and dark mullioned windows, steep
  clay-tile gable roof (gable to the street or eaves to the street) with dormers, chimneys, bargeboards; drainpipe, wall lantern,
  window boxes. Back and sides plainer but closed. Origin at the centre of the front wall at pavement level; the building extends to
  +Y. Params: width, depth, storeys, jetty, roof orientation, pattern, plaster tint, lit window ratio, shop (bool - ground floor
  shopfront).""", B)
a(2, "building", "house_stone", "Stone gabled house", "building", "ground",
  """Solid dressed-stone / brick town house, 2-3 storeys: quoins, string courses, sash or casement windows with stone surrounds, arched
  doorway up a few steps, parapet or eaves cornice, slate or tile roof with dormers and tall chimneys, iron balcony on the first
  floor. Params: width, depth, storeys, material (ashlar / brick_red / brick_ochre / plaster tint), roof type, lit ratio, balcony.""", B)
a(2, "building", "shop_luggage", "Luggage shop", "building", "ground",
  """The 'Luggage' shop of the reference: narrow 2-3 storey building whose ground floor is a painted timber shopfront (shop_window +
  door_shop + fascia_sign text param 'Luggage'), awning or hanging suitcase sign, trunks stacked outside (luggage_stack), upper floors
  plaster / timber with lit windows, tiled roof with a dormer. Params: width, storeys, paint, sign text, goods.""", B + ["luggage_stack"])
a(2, "building", "shop_tailor", "Tailor's shop with harlequin facade", "building", "ground",
  """The 'Dress Suit' shop: upper facade in a pastel harlequin diamond pattern (individually modelled / inset coloured plaster lozenges
  or tiles in 3-4 pastel tints), big arched display window with a jester mannequin inside (shop_mannequin), ornate fascia sign,
  curved gable. Params: width, storeys, diamond palette, sign text, gable style.""", B + ["shop_mannequin"])
a(2, "building", "shop_corner", "Corner shop (Items / Accessories)", "building", "ground",
  """Corner building with shopfronts on two street faces and a chamfered corner entrance under a turret or oriel: fascia signs with
  text params ('Items', 'Accessories'), neon emblem, hanging sign, upper floors half-timbered, conical turret roof on the corner.
  Origin at the centre of the main front at pavement level. Params: width, depth, storeys, sign texts, paint, turret.""",
  B + ["neon_glyph", "turret_roof"])
a(2, "building", "cafe_gallery", "Cafe under a timber gallery", "building", "ground",
  """The reference cafe: a building whose first floor is carried over the pavement on timber posts forming a covered terrace /
  gallery; under it a lit cafe front (shop_window, door), hanging coffee-cup sign, hanging lanterns, neon fleur-de-lis; barrels,
  tables, chairs, menu board and candle lanterns on the terrace (via use()). Upper storey half-timbered with a balustraded balcony
  and tiled roof. Params: width, depth, posts, storeys, sign text, terrace furniture count.""",
  B + ["balcony_timber", "hanging_lantern", "neon_glyph", "cafe_table", "cafe_chair", "barrel", "menu_board", "table_lantern"], "cafe")
a(2, "building", "hotel", "Hotel", "building", "ground",
  """The largest plain building of the square, 3-4 storeys: symmetrical stone / stucco front with rows of lit sash windows, iron
  balconies, grand double door under a glazed iron-and-glass canopy with marquee bulbs, the vertical neon HOTEL sign with crown on
  one corner, mansard or steep tiled roof with rows of dormers and chimneys. Params: width, depth, storeys, facade tint, sign text,
  lit ratio.""", B + ["door_double_hall", "neon_sign_vertical", "marquee_sign"])
a(2, "building", "townhouse_narrow", "Narrow pastel townhouse", "building", "ground",
  """Tall narrow rendered townhouse (3.5-5 m wide, 3-4 storeys) in a pastel limewash with a shaped Dutch / stepped / bell gable, sash
  windows with shutters, a hoist beam in the gable, steps to a raised door, cellar hatch. Row variety comes from params and seed.
  Params: width, storeys, gable (stepped / bell / neck / triangular), tint, shutters, lit ratio.""", B + ["cellar_hatch"])
a(2, "building", "warehouse", "Brick canal warehouse", "building", "ground",
  """Brick warehouse backing onto the canal: arched loading doors on each floor with a projecting timber hoist housing (lucam), cast
  iron-framed small-pane windows, stone string courses, slate roof, painted sign lettering on the brick (text param), a wall lantern
  over the door. Params: width, depth, storeys, brick, sign text, lit ratio.""", B, "canal")
a(2, "building", "tower_house", "Corner tower house", "building", "ground",
  """House with a round or octagonal corner stair turret rising a storey above the eaves, conical tile roof with weathervane, slit
  and small lit windows spiralling up the turret, main block half-timbered over stone. Params: width, depth, storeys, turret radius,
  turret plan, roof covering.""", B + ["turret_roof", "weathervane"])
a(2, "building", "gate_arch_town", "Town gate arch", "landmark", "ground",
  """The town gate of the reference: massive dressed-stone gateway with a wide round arch passage (walkable, >= 3 m wide, 4 m high),
  flanking buttressed piers, a parapet / small tiled gatehouse room above with lit windows, an arched sign board over the arch with
  raised gilded lettering (text param, default 'LANTERN SQUARE'), two wall lanterns flanking the arch, navy banners with gold crowns,
  posters on one pier, heavy timber doors folded open inside the passage. Origin at the centre of the passage at ground level;
  passage along Y. Params: width, passage width, height, sign text, banners (bool), gatehouse (bool).""",
  ["stone_arch", "wall_lantern", "banner_crown", "wall_posters", "window_casement", "roof_gable", "crest_cartouche", "buttress"], "gate")
a(2, "building", "clock_tower_hall", "Clock-tower hall", "landmark", "ground",
  """The landmark at the head of the plaza: a civic hall with a tall square clock tower - stone base with a grand door up steps, big
  arched stained-glass windows, hall block with steep roof and dormers; the tower with corner buttresses / quoins, belfry louvres,
  four lit clock dials (hands from a time param), corbelled cornice and a tall red-tiled spire with lucarnes, finial and
  weathervane. 18-26 m to the spire tip. Params: width, depth, tower height, spire height, time, stone tint, roof covering.""",
  ["window_stained_arch", "door_double_hall", "turret_roof", "roof_gable", "dormer", "weathervane", "buttress", "door_steps", "wall_lantern",
   "window_round", "chimney"])
a(2, "building", "castle_far", "Far hilltop castle", "backdrop", "ground",
  """The glowing far landmark of the skyline: a distant hill-top castle / great church silhouette in the spirit of the reference
  board - a crag with woodland, curtain wall, keep, towers and spires, a few lit windows and a softly lit clock or rose - readable
  as a pale silhouette against a navy night sky from 150-300 m. Built to be placed far away at large scale, so it is cheap
  (<= 15k tris at LOD0), hazed, with no light sockets: everything is self-lit in baked moonlight tones plus emissive windows.
  Origin at ground centre, facing -Y. Params: style (castle / cathedral / citadel / ridge hamlet), floodlit, haze, lit_ratio,
  glow, lit_color, moon_az, seed.""", (), "board")
a(2, "building", "civic_hall_grand", "Grand civic hall with clock tower", "landmark", "ground",
  """The great hall that closes the second square (reference docs/ref/t2_hall.png): a LONG, PALE limestone civic range, clearly
  wider than tall, built at true size - never meant to be scaled by the town. From the ground up: (1) a projecting lower storey /
  undercroft of pale ashlar whose front is an arcade of pointed stained-glass arches (violet / blue glass, lit from inside) with a
  moulded plinth; its flat roof is a balustraded terrace (turned stone balusters, piers with small lanterns or urn finials) running
  the whole front; (2) set back behind the terrace, the tall upper hall storey: a regular row of slender arched stained-glass
  windows between pilaster buttresses, string course, corbelled eaves cornice, steep RED clay-tile roof with a few lucarnes and a
  ridge cresting; (3) rising through the front at the centre, a slender square clock tower of the same pale stone - tall arched
  louvred opening, a gilt-framed lit clock dial on each face (hands from a time param), small gablets over the dials, corner
  pinnacles, steep red-tiled spire with ball finial; (4) each end of the range finished by a polygonal or round end pavilion with
  a red conical roof and weathervane (param: none / one / both); (5) in front of the lower storey a gabled entrance porch
  pavilion of pale stone - pointed-arch double door up broad steps, a rose window in its gable, flanking pinnacles and two
  lanterns (param porch_offset moves it along the front; the picture has it right of the tower). Stone courses are 0.3-0.4 m high,
  quoins and mouldings sized for a real building; walls must READ pale and warm at night: wall-lantern / floodlight sockets wash
  the facade, windows glow strongly. 34-42 m to the spire tip. Params: width, depth, wing_storey_height, tower_height,
  spire_height, tower_offset, porch_offset, terrace_depth, end_pavilions, bays, time, stone tint, glass palette, roof covering,
  window_glow, dial_glow, lantern_intensity, moss.""",
  ["window_stained_arch", "door_double_hall", "turret_roof", "weathervane", "door_steps", "wall_lantern", "window_round", "column",
   "roof_finial", "buttress"], "t2_hall")
a(2, "building", "tower_keep_ornate", "Ornate battlemented tower", "landmark", "ground",
  """The second tower of the second square (reference docs/ref/t2_tower.png): a free-standing square civic / guild tower in the
  ornate Gothic-revival manner, built at TRUE size (26-36 m high, 7-10 m square) - never meant to be scaled by the town. Warm
  cream coursed ashlar in 0.3-0.4 m courses with bands of darker red-brown stone or brick (polychrome banding) and dressed quoins;
  a battered plinth with a small pointed door; three or four stages divided by moulded string courses; on the show faces a tall
  two-light stained-glass lancet window in a gilt / pale moulded frame with tracery head (blue / violet / amber glass, strongly
  lit), smaller lit trefoil or slit windows and blind arcading on the other stages, a carved shield or clock-less roundel; at the
  top a corbel table of small arches carrying a machicolated, crenellated parapet (merlons with caps, some pierced), an
  octagonal bartizan / corner turret at each corner rising above the parapet with its own little battlements or stone caps, a
  low lead / tile roof and flagpole behind the parapet. Two or three 0.6 m wall lanterns on brackets and small uplight sockets so
  the upper stages read at night; pigeons optional. All four faces finished (it is seen from every side over the roofs).
  Params: width, height, stages, band colour / band count, bartizans (bool or style), parapet style, window glass palette,
  window_glow, lantern_intensity, stone tint, moss, flag (bool).""",
  ["window_stained_arch", "door_plank", "wall_lantern", "window_round", "corbel_bracket", "roof_finial", "banner_crown"], "t2_tower")
a(2, "building", "monument_star", "Star shrine monument", "landmark", "wall_ground",
  """The monument of the reference: a stone shrine wall with a large glowing stained-glass star set in a round traceried surround
  (glass_stained, strong coloured light), an inscription plaque beneath, two side cascades (water_fall) falling into flanking
  stone basins with water, broad shallow steps in front flanked by two ground lanterns, carved pilasters and a curved pediment.
  Params: width, height, star points, glass palette, inscription, water (bool).""",
  ["ground_lantern", "plaque", "column"], "monument")
a(2, "building", "arcade_passage", "Covered alley passage", "building", "ground",
  """A building bridging an alley: walkable stone-arched passage (>= 2.2 m wide) through the ground floor with a ribbed / beamed
  ceiling and a hanging lantern inside, lit rooms above with an oriel over the arch, tiled roof. Passage along Y through the origin.
  Params: width, depth, passage width, storeys, lit ratio.""", B + ["stone_arch", "hanging_lantern"], "canal")
a(2, "building", "backdrop_row", "Distant roofscape row", "backdrop", "ground",
  """Cheap far-background row of 3-7 simplified houses seen only over the rooftops: correct silhouettes (gables, chimneys, dormers, a
  spire), lit window quads, no fine detail, < 8k triangles total; staggered heights and depths from the seed. Along X.
  Params: length, count, height range, lit ratio.""")


# ------------------------------------------------------------------ contracts
# Param names / placement conventions the town generator (and composing buildings) rely on. id -> (param names, text)
RUN = "The run lies along X and spans exactly x in [-length/2, +length/2] (origin at its centre on the ground); any `length` in the stated range must work, and ends must be clean so runs butt together."
BLD = ("Front wall plane at y = 0 facing -Y, body from y = 0 to y = +depth, ground-floor walls span exactly x in [-width/2, +width/2] "
       "(eaves / jetties / signs may overhang by <= 0.7 m, steps and canopies project <= 1.6 m to -Y). Closed and presentable on ALL four sides "
       "(side and back walls get plain windows, drainpipes, a lit window or two) and roofed - it is seen from behind and above. "
       "Every width / depth / storeys in range must produce a sound building; `seed` re-rolls the detail.")
CONTRACT = {
    "balustrade": (["length", "height"], RUN + " length 1-12 m. Handrail top at z = height."),
    "wall_stone_run": (["length", "height", "thickness"], RUN + " length 1-14 m."),
    "retaining_wall": (["length", "height"], RUN + " length 2-16 m, height 0.6-3.5 m. The visible face is the plane y = 0 facing -Y (a plinth / batter may project <= 0.2 m to -Y); the wall body and coping extend to +Y (the earth side) by 0.45-0.7 m; the flat top of the coping is at exactly z = height (the upper terrace paving meets it and a balustrade stands on it)."),
    "iron_railing": (["length", "height"], RUN + " length 1-12 m."),
    "canal_wall": (["length", "depth"], RUN + " length 2-16 m. The water is on the -Y side: the visible wall face is in the plane y = 0 facing -Y and drops from the coping top (z = 0, quay level) to z = -depth (1-3 m); the coping and wall body extend to +Y (<= 0.6 m); the coping nosing may overhang -Y by <= 0.08 m."),
    "kerb_gutter": (["length"], RUN + " length 1-12 m (straight mode)."),
    "hedge_run": (["length", "height"], RUN + " length 1-10 m."),
    "chain_barrier": (["length"], RUN + " length 1.5-10 m (posts at both ends)."),
    "backdrop_row": (["length"], RUN + " length 12-60 m."),
    "bunting": (["span"], "Origin at the first fixing point, the second fixing point is at (span, 0, 0); span 2-14 m."),
    "festoon_lights": (["span"], "Origin at the first fixing point, the second fixing point is at (span, 0, 0); span 2-14 m."),
    "stairs_plaza": (["width", "rise"], "width 2-10 m, rise 0.5-3.5 m. Origin at the centre of the bottom step's front edge (z = 0); the flight rises towards +Y and the top tread / landing is flush at exactly z = rise at the back of the bounding box (max Y). Nothing projects behind the top edge."),
    "door_steps": (["width", "steps"], "Origin at ground on the wall plane (y = 0), steps descend towards -Y. Top landing at z = steps x ~0.17 m."),
    "external_stair": (["rise"], "Wall plane at y = 0, stair in front of it (-Y); landing floor at z = rise."),
    "stone_arch": (["width", "height", "depth"], "width / height = the CLEAR opening; wall plane y = 0 at the front face, the arch runs back to y = +depth."),
    "tunnel_mouth": (["width", "height"], "width / height = clear opening; portal face at y = 0 facing -Y, vault receding to +Y; z = 0 is the quay level and the water channel is cut down below it (allow_below_ground)."),
    "bridge_stone": (["span", "width", "rise"], "Crosses along X: abutments at x = +-span/2 where the deck meets z = 0; deck `width` along Y; the crown of the deck is `rise` (0.3-1.2 m) above z = 0. span 4-10 m."),
    "fountain_tiered": (["diameter"], "diameter = outer diameter of the basin kerb (2.5-7 m); centred on the origin."),
    "paving_patch": (["diameter"], "diameter 2-12 m, centred on the origin, top surface within 0-30 mm of z = 0."),
    "bandstand": (["diameter"], "diameter 4-8 m, centred."),
    "pergola": (["length", "width", "height"], "Centred on the origin, length along X."),
    "gate_pier": (["height", "width"], "Centred on the origin."),
    "iron_gate": (["width", "height"], "width = clear opening between the piers; centred, gate line along X."),
    "column": (["height", "diameter"], "Centred, base at z = 0, top of the capital at z = height."),
    "buttress": (["height", "width", "depth"], "Wall plane y = 0, projects to -Y."),
    "awning": (["width", "projection"], "Wall plane y = 0, origin at the centre of the top fixing rail."),
    "window_box": (["width"], "Wall plane y = 0; origin at the centre of the back top edge (hangs below a sill)."),
    "shop_window": (["width", "height"], "Outer size of the shopfront window unit; origin at the bottom centre on the wall plane."),
    "window_casement": (["width", "height"], "Outer frame size; origin at the bottom centre of the frame on the wall plane (y = 0); nothing but the frame depth goes behind y = 0. A `lit` bool param."),
    "window_sash": (["width", "height"], "Outer frame size; origin at the bottom centre of the frame on the wall plane (y = 0). A `lit` bool param."),
    "window_oriel": (["width", "height"], "Origin at the bottom centre of the oriel's window band on the wall plane; corbelling extends below. A `lit` bool param."),
    "window_stained_arch": (["width", "height"], "Outer size; origin at the bottom centre on the wall plane."),
    "window_round": (["diameter"], "Origin at the centre of the window on the wall plane."),
    "door_plank": (["width", "height"], "Outer frame size; origin at the bottom centre (threshold) on the wall plane."),
    "door_shop": (["width", "height"], "Outer frame size; origin at the bottom centre (threshold) on the wall plane."),
    "door_double_hall": (["width", "height"], "Outer frame size; origin at the bottom centre (threshold) on the wall plane."),
    "dormer": (["width", "height", "pitch"], "Sits on a roof slope of `pitch` degrees: origin at the centre of the dormer's front base on the roof surface; the cheeks and roof run back (+Y) into the slope."),
    "chimney": (["height", "width"], "Origin at the centre of the stack base; height = from base to the top of the pots' collar."),
    "roof_gable": (["width", "depth", "pitch"], "A complete pitched roof for a rectangular plan: origin at the centre of the plan at eaves level (z = 0); ridge along X by default (`ridge` param 'x' | 'y'); width along X, depth along Y measured at the wall faces (eaves overhang beyond)."),
    "turret_roof": (["diameter", "height"], "Origin at the centre at eaves level."),
    "timber_frame_bay": (["width", "height"], "A wall panel of one storey: spans x in [-width/2, width/2], z in [0, height], front face at y = 0, ~0.15-0.2 m thick towards +Y. Optional `opening_w` / `opening_h` / `opening_sill` params leave a framed hole for a window."),
    "balcony_iron": (["width", "projection"], "Wall plane y = 0; origin at the centre of the floor's back edge."),
    "balcony_timber": (["width", "projection"], "Wall plane y = 0; origin at the centre of the floor's back edge."),
    "porch_canopy": (["width", "projection"], "Wall plane y = 0; origin at ground below the centre."),
    "drainpipe": (["height"], "Wall plane y = 0; origin at ground; hopper head at z = height."),
    "fascia_sign": (["width", "height", "text"], "Wall plane y = 0; origin at the bottom centre."),
    "hanging_sign": (["text"], "Wall plane y = 0; origin where the bracket meets the wall; the bracket projects towards -Y."),
    "house_timber": (["width", "depth", "storeys"], BLD + " width 5-11, depth 6-11, storeys 2-3."),
    "house_stone": (["width", "depth", "storeys"], BLD + " width 5-12, depth 6-11, storeys 2-3."),
    "shop_luggage": (["width", "depth", "storeys"], BLD + " width 5-8, depth 6-10, storeys 2-3."),
    "shop_tailor": (["width", "depth", "storeys"], BLD + " width 6-9, depth 6-10, storeys 2-3."),
    "shop_corner": (["width", "depth", "storeys"], BLD + " width 7-11, depth 7-11, storeys 2-3. The second shop face is the +X side (x = +width/2); the chamfered corner is at the front right (+X, -Y)."),
    "cafe_gallery": (["width", "depth", "storeys", "gallery_depth"], BLD + " width 8-14, depth 6-10. EXCEPTION to the projection limit: the upper floor and its posts project `gallery_depth` (2.5-4 m) in front of y = 0 forming the covered terrace; the terrace floor is the pavement (z = 0, walkable, >= 2.4 m clear headroom) - keep furniture to the sides so a 1.4 m path stays free."),
    "hotel": (["width", "depth", "storeys"], BLD + " width 12-20, depth 8-12, storeys 3-4."),
    "townhouse_narrow": (["width", "depth", "storeys"], BLD + " width 3.5-5.5, depth 7-10, storeys 3-4."),
    "warehouse": (["width", "depth", "storeys"], BLD + " width 8-16, depth 7-11, storeys 3-4."),
    "tower_house": (["width", "depth", "storeys"], BLD + " width 7-11, depth 7-10, storeys 2-3. The turret stands on the front-left corner (-X, -Y) and may bulge <= 1.2 m outside the plan."),
    "gate_arch_town": (["width", "depth", "passage_width"], "Origin at the centre of the passage on the ground; passage axis along Y, the gate body spans x in [-width/2, +width/2] (8-13 m) and y in [-depth/2, +depth/2] (2.5-5 m); the passage is a clear walkable `passage_width` (3-4.5 m) wide and >= 4 m high with NOTHING on its floor; the sign and lanterns are on the -Y face (the approach side) and the +Y face is finished too."),
    "arcade_passage": (["width", "depth", "passage_width", "storeys"], "Origin at the centre of the passage on the ground; passage axis along Y; the building spans x in [-width/2, +width/2] (6-10 m) and y in [-depth/2, +depth/2] (6-11 m); clear walkable passage `passage_width` (2.4-3.5 m) wide, >= 2.8 m high, nothing on its floor. Both the -Y and +Y faces are full facades."),
    "clock_tower_hall": (["width", "depth"], "Front (main door, tower) faces -Y with the front wall plane at y = 0 and the body to +Y; spans x in [-width/2, +width/2] (22-32 m), depth 10-16 m; the tower is centred on x = 0 at the front; entrance steps project <= 3 m to -Y. Closed on all sides."),
    "civic_hall_grand": (["width", "depth", "tower_height", "spire_height", "terrace_depth"], "Front faces -Y. The UPPER hall storey's front wall plane is y = 0 and the body runs to y = +depth (10-16 m); the range spans x in [-width/2, +width/2] (30-44 m; end pavilions may bulge <= 2 m beyond). The lower arcade storey with its terrace projects `terrace_depth` (4-7 m) to -Y, the porch and its steps a further <= 5 m; nothing else projects. Tower centred on x = `tower_offset` (default 0). `tower_height` = ground to the top of the tower's masonry (20-28 m), `spire_height` 8-14 m. Closed on all sides (no walkable interior; the terrace is not walked on). Built at true scale: the town places it at scale 1."),
    "tower_keep_ornate": (["width", "height"], "Origin at the centre of the square plan on the ground; the shaft spans x and y in [-width/2, +width/2] (7-10 m; plinth, corbel table and bartizans may project <= 1.2 m beyond); `height` = ground to the top of the merlons (26-36 m), bartizans / flagpole rise above it. The main show face (door, largest window) is -Y; all four faces are finished. Closed (no walkable interior). Built at true scale: the town places it at scale 1."),
    "monument_star": (["width", "height"], "Back wall plane at y = 0 facing -Y, width 6-10 m, steps and basins project <= 3.5 m to -Y."),
}
for c in C:
    if c["id"] in CONTRACT:
        names, text = CONTRACT[c["id"]]
        c["contract"] = names
        c["brief"] += " CONTRACT (the town generator / composing buildings rely on it - use exactly these PARAM names: " + ", ".join(names) + "): " + " ".join(text.split())
missing = set(CONTRACT) - {c["id"] for c in C}
assert not missing, missing

os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
ids = [c["id"] for c in C]
assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
for c in C:
    for d in c["deps"]:
        assert d in ids, (c["id"], d)
json.dump({"assets": C}, open(os.path.join(ROOT, "docs", "catalog.json"), "w"), indent=1)
from collections import Counter
print(len(C), "assets", dict(Counter(c["category"] for c in C)), dict(Counter(c["tier"] for c in C)))
