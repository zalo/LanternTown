"""tk - the Lantern Town Square modelling toolkit for Blender's CLI (`blender --background`).

    from tk import *

See docs/AUTHORING.md for the asset contract and docs/API.md for the function reference.
"""
from math import radians, degrees, pi, sin, cos, tan, sqrt, atan2, hypot
from mathutils import Vector, Matrix, Euler

from .core import (P, Rng, resolve_params, params_hash, clear_scene, root, empty, parent, mesh_object, from_bmesh,
                   set_loc_rot, place, world_matrix, apply_transform, apply_modifiers, delete, duplicate, mirror,
                   descendants, world_bbox, hex_rgb, lerp, clamp, smoothstep)
from .surf import (mat, assign, assign_faces, smooth, flat, uv_box, uv_grain, uv_cyl, uv_planar, uv_keep,
                   uv_transform, realize, paint, height_ramp, facing, noise_mask, mask_mul, material_catalog)
from .geo import (bevel, subsurf, solidify, array, weld, simple_deform, boolean, subdivide, densify, displace_noise,
                  transform_verts, recalc_normals, box, cylinder, cone, sphere, torus, lathe, rounded_rect,
                  circle_pts, arch_pts, extrude_profile, shape_extrude, rounded_box, resample_path, path_length,
                  spiral_path, arc_path, catenary, bezier_path, sweep, chain, surface, grid, instances_to_mesh, trs,
                  join, text, plank, stone_block, bolt, ring_band)
from .asm import socket, light, fx, part, use, Component, load_asset, variant_params, find_asset
from .levels import lod, lod_pick, lod_exact, lod_thin, lod_segs, lod_upto, lod_drop, lod_as, lod_budget, lod_faces, lod_memo, lod_path, lod_standin, lod_keep
