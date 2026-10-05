"""tk.asm - sockets (lights / attachment points), animated parts and asset composition."""
import bpy
import json
import importlib.util
import os
import glob
from math import radians
from mathutils import Vector, Matrix, Euler

from .core import (empty, _link, world_matrix, resolve_params, Rng, _STATE, _next_id, world_bbox, P, _auto_id,
                   params_hash)
from .surf import realize
from . import levels as _lod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# Sockets
# ----------------------------------------------------------------------------


def socket(name, kind="attach", loc=(0, 0, 0), rot=None, **props):
    """A named point exported with the asset (GLB node + sidecar json).

    kind 'attach' : a place where the town generator / a parent asset may hang something.
                    Convention: the socket's -Y axis points away from the surface (outwards), Z up.
                    Give it a `role` prop, e.g. role='sign', 'lantern', 'awning', 'planter',
                    'seat', 'prop_top' (things may be put on top), 'door', 'window'.
         'light'  : use light() instead.
         'fx'     : runtime effect anchor, prop fx='flame'|'smoke'|'steam'|'water_jet'|'drip'|'moths'.
    """
    e = empty(f"sock_{kind}_{name}", loc, rot)
    e["tsq_socket"] = kind
    e["tsq_name"] = name
    e["tsq_props"] = json.dumps(props)
    return e


def light(name, loc, color="#ffbf6a", intensity=40.0, range=9.0, flicker=0.12, shadow=False, **props):
    """A warm point light emitted by this asset at `loc` (asset space, meters).

    intensity : candela. Candle ~1, oil lamp 5-15, gas mantle street lantern 40-90,
                small electric bulb 30-60, big arc/electric globe 150-300. Shop window spill 20-60.
    range     : cut-off distance in meters (keep tight: 4-12 m; cost grows with overlap).
    flicker   : 0 = steady electric, 0.1-0.2 gas, 0.4-0.7 candle / open flame.
    color     : '#RRGGBB'. Flame/gas ~ '#ffbf6a', tungsten ~ '#ffc98a', neon: saturated.
    shadow_near=<m> (extra prop): geometry closer to the light than this casts no shadow from it.
                Set it just beyond the light's own housing (lantern cage, shade, bracket) so the
                fitting does not throw a giant shadow of itself: ~0.9 x the housing size.
    Every light needs a visible emissive source (flame / mantle / bulb mesh using an emissive
    material) at the same spot - never a light from nowhere.
    The glow sprite the engine draws on a fitting (a light whose name says lantern / lamp / globe / flame ..., see
    tools/town.py light_is_fitting; helper lights get none) follows the light. Two extra props correct it:
    halo=<candela>   : the output the glow stands for, when the fitting's light is split over helper lights
                       (default: this light's own intensity; 0 = no glow).
    halo_at=(x,y,z)  : where the burner is seen (asset space), when the light itself has to sit off it - or a list
                       of points when one light stands for several burners (the globes of one head): each gets a
                       glow and an equal share of `halo`.
    halo_r=<m>       : the radius of an opaque glowing body around the burner (an opal globe). The glow is drawn
                       that far towards the eye, or the body hides its hot centre and the lamp looks switched off.
    """
    if props.get("halo_at") is not None:
        at = props["halo_at"]
        at = [at] if isinstance(at[0], (int, float)) else at
        props["halo_at"] = [[float(c) for c in a] for a in at]
    return socket(name, "light", loc, None, color=color, intensity=float(intensity), range=float(range),
                  flicker=float(flicker), shadow=bool(shadow), **props)


def fx(name, kind, loc, rot=None, **props):
    """Runtime effect anchor: kind = 'flame' | 'smoke' | 'steam' | 'water_jet' | 'drip' | 'moths'.
    Useful props: size (m), rate, color."""
    return socket(name, "fx", loc, rot, fx=kind, **props)


def part(objs, name, pivot=(0, 0, 0), motion=None, **props):
    """Tag objects as a separately exported, animatable part (everything else merges into 'body').
    pivot  : the hinge / suspension point in asset space.
    motion : runtime animation hint: 'swing' (hanging signs, lanterns on chains - gentle wind sway),
             'spin' (weather vane, windmill), 'flutter' (flags, bunting, laundry), None = static.
    props  : amp (degrees or meters), speed (Hz), axis ('X'|'Y'|'Z')."""
    if isinstance(objs, bpy.types.Object):
        objs = [objs]
    for o in objs:
        if o is None:
            continue
        o["tsq_part"] = name
        o["tsq_pivot"] = [float(c) for c in pivot]
        o["tsq_part_props"] = json.dumps(dict(motion=motion, **props))
    return objs


# ----------------------------------------------------------------------------
# Composition: build another asset inline and stamp it
# ----------------------------------------------------------------------------

_MODULES = {}


def find_asset(asset_id):
    # TSQ_ASSET_DIR (set by tools/town.py): released copies of the sources win over the working files, so a town
    # build never picks up an asset that is half-way through a revision
    for base in filter(None, [os.environ.get("TSQ_ASSET_DIR"), "assets"]):
        hits = glob.glob(os.path.join(ROOT, base, "*", asset_id + ".py"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no asset generator named {asset_id!r} under assets/*/")


def load_asset(asset_id):
    if asset_id in _MODULES:
        return _MODULES[asset_id]
    path = find_asset(asset_id)
    spec = importlib.util.spec_from_file_location("tsq_asset_" + asset_id, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULES[asset_id] = mod
    return mod


def variant_params(mod, variant=None, overrides=None):
    """Resolve params for a named variant (or defaults) plus overrides."""
    base = {}
    variants = getattr(mod, "VARIANTS", {}) or {}
    if variant not in (None, "default"):
        if variant not in variants:
            raise KeyError(f"unknown variant {variant!r}; have {sorted(variants)}")
        base.update(variants[variant])
    elif variant == "default" and "default" in variants:
        base.update(variants["default"])
    base.update(overrides or {})
    return resolve_params(mod.PARAMS, base)


class Component:
    """A sub-asset built inline: one joined mesh (kept out of the scene) + its sockets.
    place() stamps a copy into the scene."""

    def __init__(self, asset_id, obj, sockets, params):
        self.asset_id = asset_id
        self.obj = obj
        self.sockets = sockets  # [(name, kind, Matrix, props)]
        self.params = params
        if obj is not None and len(obj.data.vertices):
            cs = [Vector(c) for c in obj.bound_box]
            self.min = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
            self.max = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
        else:
            self.min = Vector((0, 0, 0))
            self.max = Vector((0, 0, 0))
        self.size = self.max - self.min

    def place(self, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, sockets=True, name=None):
        """Stamp a copy at loc / rot (Euler degrees) / scale. Returns the new mesh object.
        The component's light / attach / fx sockets are copied too (sockets=False to drop them,
        or a set of kinds to keep, e.g. {'light'})."""
        s = (scale, scale, scale) if isinstance(scale, (int, float)) else scale
        M = (Matrix.Translation(Vector(loc)) @ Euler(tuple(radians(a) for a in rot)).to_matrix().to_4x4()
             @ Matrix.Diagonal((s[0], s[1], s[2], 1.0)))
        return self.place_matrix(M, sockets, name)

    def place_matrix(self, M, sockets=True, name=None):
        n = _auto_id("place", self.asset_id)
        o = None
        if self.obj is not None:
            o = self.obj.copy()  # linked mesh data; finalize copies on join
            o.name = name or f"{self.asset_id}_{n}"
            _link(o)
            o.parent = None
            o.matrix_world = M
            o.matrix_basis = M
        if sockets:
            for (sname, kind, sm, props) in self.sockets:
                if isinstance(sockets, (set, list, tuple)) and kind not in sockets:
                    continue
                e = bpy.data.objects.new(f"sock_{kind}_{self.asset_id}{n}_{sname}", None)
                e.empty_display_size = 0.05
                _link(e)
                e.matrix_basis = M @ sm
                e["tsq_socket"] = kind
                e["tsq_name"] = f"{self.asset_id}{n}_{sname}"
                if props.get("halo_at") is not None:       # points in the component's space, like the socket itself
                    props = dict(props, halo_at=[[float(c) for c in M @ Vector(a)] for a in props["halo_at"]])
                if props.get("halo_r"):
                    props = dict(props, halo_r=float(props["halo_r"]) * M.to_scale()[0])
                e["tsq_props"] = json.dumps(props)
        return o


def use(asset_id, variant=None, **overrides):
    """Build another asset generator inline and return it as a Component you can place() many
    times (buildings use this for windows, doors, shutters, chimneys, lanterns, signs...).

        win = use("window_casement", width=0.9, height=1.4, seed=p.seed + 3)
        for x in xs: win.place((x, 0, 3.2))
        w, h = win.size.x, win.size.z          # measure it to cut matching openings

    Every distinct (params) call is a fresh build; reuse the returned Component for repeats."""
    from .geo import join
    mod = load_asset(asset_id)
    params = variant_params(mod, variant, overrides)
    before = set(bpy.data.objects)
    rng = Rng(f"{asset_id}/{params['seed']}")
    tracked = _lod._tracked()       # level-aware build: the composed asset is modelled at the same level (tk.lod())
    level = _lod._LOD["level"]
    plain = tracked and _lod._child_plain(mod)      # no levels of its own: built in full, reduced below
    bounds = None
    if tracked:
        _lod._scope_push(asset_id, params_hash(asset_id, params))
    try:
        if plain:
            _lod._LOD["level"] = 0
        _STATE["child"] = _STATE.get("child", 0) + 1
        try:
            mod.build(P(params), rng)
        finally:
            _STATE["child"] -= 1
            _lod._LOD["level"] = level
        new = [o for o in bpy.data.objects if o not in before]
        # convert any stray curve/text objects
        meshes = [o for o in new if o.type == 'MESH' and (len(o.data.polygons) > 0 or "_lod_empty" in o)]
        socks = []
        for o in new:
            if o.type == 'EMPTY' and "tsq_socket" in o:
                socks.append((o["tsq_name"], o["tsq_socket"], world_matrix(o).copy(), json.loads(o.get("tsq_props", "{}"))))
        joined = join(meshes, name=f"cmp_{asset_id}_{_auto_id('use', asset_id)}") if meshes else None
        if tracked:
            bounds = Component(asset_id, joined, [], params)
            _lod._bbox(bounds)
            if plain and joined is not None:
                joined = _lod._reduce(joined, level, getattr(mod, "META", None))
    finally:
        if tracked:
            _lod._scope_pop(rng)
    for o in new:
        try:
            if o is not joined and o.name in bpy.data.objects and o not in meshes:
                bpy.data.objects.remove(o, do_unlink=True)
        except ReferenceError:
            pass
    if joined is not None:
        for c in list(joined.users_collection):
            c.objects.unlink(joined)
        joined.use_fake_user = True
        for k in ("tsq_part", "tsq_pivot", "tsq_part_props"):
            if k in joined:
                del joined[k]
    comp = Component(asset_id, joined, socks, params)
    if bounds is not None:
        comp.min, comp.max, comp.size = bounds.min, bounds.max, bounds.size
    return comp
