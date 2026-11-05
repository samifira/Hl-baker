# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene building: import meshes, auto-group them, bake, render previews. Runs in the worker."""
import os
import math
import bpy
import numpy as np
from mathutils import Vector

from .config import GroupConfig, split_name, FORMATS, enabled_maps
from . import core


class SceneError(Exception):
    pass


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def _enable_addon(name):
    import addon_utils
    try:
        addon_utils.enable(name, default_set=True, persistent=False)
    except Exception:
        pass


def reset_scene():
    """Empty the scene (each worker process starts fresh, so no need to reload factory settings)."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                 bpy.data.lights, bpy.data.images, bpy.data.curves, bpy.data.armatures,
                 bpy.data.actions):
        for idb in list(coll):
            coll.remove(idb)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    return sc


def import_file(path, scale=1.0):
    """Import one mesh file, return the list of new mesh objects."""
    if not os.path.isfile(path):
        raise SceneError(f"File not found: {path}")
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fbx":
        _enable_addon("io_scene_fbx")
        bpy.ops.import_scene.fbx(filepath=path, global_scale=scale, use_custom_normals=True)
        scale = 1.0
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path, global_scale=scale)
        scale = 1.0
    elif ext in (".gltf", ".glb"):
        _enable_addon("io_scene_gltf2")
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path, global_scale=scale)
        scale = 1.0
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path, global_scale=scale)
        scale = 1.0
    elif ext == ".abc":
        bpy.ops.wm.alembic_import(filepath=path)
    elif ext in (".usd", ".usda", ".usdc", ".usdz"):
        bpy.ops.wm.usd_import(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
        for ob in dst.objects:
            if ob is not None:
                bpy.context.scene.collection.objects.link(ob)
    else:
        raise SceneError(f"Unsupported file type: {ext}")
    new = [o for o in bpy.data.objects if o not in before]
    if scale != 1.0:
        roots = [o for o in new if o.parent is None]
        for o in roots:
            o.matrix_world = o.matrix_world.copy()
            o.scale = o.scale * scale
            o.location = o.location * scale
    meshes = [o for o in new if o.type == 'MESH']
    # make sure everything is evaluated & visible in the view layer
    for o in meshes:
        o.hide_viewport = False
        o.hide_render = False
    bpy.context.view_layer.update()
    return meshes


def load_project_meshes(project):
    """Import every file. Returns list of dicts: {obj, role, file}."""
    reset_scene()
    items = []
    for fe in project.files:
        for ob in import_file(fe.path, fe.scale):
            base, kind = split_name(ob.name)
            role = fe.role if fe.role != "auto" else (kind or "unassigned")
            items.append({"obj": ob, "role": role, "file": fe.path, "base": base})
    return items


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def _center(ob):
    pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    return sum(pts, Vector()) / 8.0


def _bbox(ob):
    pts = np.array([tuple(ob.matrix_world @ Vector(c)) for c in ob.bound_box])
    return pts.min(0), pts.max(0)


def _overlap_score(a, b):
    (amin, amax), (bmin, bmax) = _bbox(a), _bbox(b)
    inter = np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0, None)
    vol = np.prod(np.maximum(bmax - bmin, 1e-9))
    return float(np.prod(inter) / vol)


def auto_groups(items, cage_offset=0.05, max_ray=0.0):
    """Toolbag-style quick loader: match by name, then fall back to spatial overlap."""
    lows = [i for i in items if i["role"] == "low"]
    highs = [i for i in items if i["role"] == "high"]
    cages = [i for i in items if i["role"] == "cage"]
    groups, by_base = [], {}
    for it in sorted(lows, key=lambda i: i["obj"].name):
        g = GroupConfig(name=it["base"], low=it["obj"].name, cage_offset=cage_offset,
                        max_ray=max_ray)
        groups.append(g)
        by_base.setdefault(it["base"].lower(), g)
    if not groups:
        return [], [i["obj"].name for i in highs]
    low_obj = {g.name: bpy.data.objects[g.low] for g in groups}
    unmatched = []
    for it in sorted(highs, key=lambda i: i["obj"].name):
        g = by_base.get(it["base"].lower())
        if g is None and len(groups) == 1:
            g = groups[0]
        if g is None:
            # spatial fallback: the low whose bounds best contain this high
            scores = [(_overlap_score(low_obj[gg.name], it["obj"]),
                       -(_center(low_obj[gg.name]) - _center(it["obj"])).length, gg) for gg in groups]
            scores.sort(key=lambda s: (s[0], s[1]), reverse=True)
            g = scores[0][2]
        g.highs.append(it["obj"].name)
    for it in cages:
        g = by_base.get(it["base"].lower())
        if g is None:
            vc = len(it["obj"].data.vertices)
            cands = [gg for gg in groups if len(low_obj[gg.name].data.vertices) == vc and not gg.cage]
            g = cands[0] if len(cands) == 1 else None
        if g is not None:
            g.cage = it["obj"].name
    for g in groups:
        if not g.highs:
            unmatched.append(g.low)
    return groups, unmatched


def describe(items):
    out = []
    for it in items:
        ob = it["obj"]
        out.append({"name": ob.name, "role": it["role"], "file": it["file"],
                    "verts": len(ob.data.vertices), "tris": sum(len(p.vertices) - 2 for p in ob.data.polygons),
                    "uvs": [uv.name for uv in ob.data.uv_layers],
                    "materials": [s.material.name if s.material else "" for s in ob.material_slots]})
    return out


def scan(project):
    items = load_project_meshes(project)
    groups, unmatched = auto_groups(items, project.default_cage_offset, project.default_max_ray)
    return {"objects": describe(items), "groups": [g.__dict__ for g in groups], "unmatched": unmatched}


# ---------------------------------------------------------------------------
# Bake
# ---------------------------------------------------------------------------
def resolve_groups(project, items):
    if project.groups:
        cfgs = project.groups
    else:
        cfgs, _ = auto_groups(items, project.default_cage_offset, project.default_max_ray)
    out, errors = [], []
    for g in cfgs:
        if not g.enabled:
            continue
        low = bpy.data.objects.get(g.low)
        highs = [bpy.data.objects.get(h) for h in g.highs]
        cage = bpy.data.objects.get(g.cage) if g.cage else None
        if low is None:
            errors.append(f"Group '{g.name}': low poly '{g.low}' not found in the loaded files.")
            continue
        missing = [h for h, o in zip(g.highs, highs) if o is None]
        if missing:
            errors.append(f"Group '{g.name}': high poly not found: {', '.join(missing)}")
        out.append(core.RuntimeGroup(g.name, low, [h for h in highs if h], cage,
                                     g.cage_offset, g.max_ray))
    return out, errors


def bake(project, progress, preview_dir=None):
    s = project.settings
    if not s.output_dir:
        raise SceneError("No output folder set.")
    items = load_project_meshes(project)
    groups, errors = resolve_groups(project, items)
    errors += core.validate(bpy.context, s, groups)
    if errors:
        raise SceneError("\n".join(errors))
    progress(event="log", message=f"Baking {len(enabled_maps(s))} map(s) for {len(groups)} group(s) "
                                  f"at {s.width}x{s.height}, AA x{int(s.supersample) ** 2}")
    sess = core.BakeSession(bpy.context, s, groups, progress=progress, preview_dir=preview_dir)
    try:
        sess.setup()
        sess.run()
    finally:
        sess.restore()
    return sess.written


# ---------------------------------------------------------------------------
# Preview render
# ---------------------------------------------------------------------------
def render_preview(project, out_path, yaw=35.0, pitch=25.0, width=960, height=720, samples=32,
                   show="low"):
    s = project.settings
    items = load_project_meshes(project)
    groups, errors = resolve_groups(project, items)
    if not groups:
        raise SceneError("Nothing to preview." + ("\n" + "\n".join(errors) if errors else ""))
    sc = bpy.context.scene
    fmt, depth, ext, _ = FORMATS[s.file_format]
    folder = os.path.abspath(s.output_dir)
    lows = []
    for g in groups:
        setname = core.texture_set_name(s) if s.output_mode == 'SINGLE' else bpy.path.clean_name(g.name)

        def tex(suffix, _set=setname):
            p = os.path.join(folder, f"{_set}_{suffix}.{ext}")
            return p if os.path.exists(p) else None
        if show == "low":
            mat = core._preview_material(setname, tex, s)
            g.low.data.materials.clear()
            g.low.data.materials.append(mat)
            for p in g.low.data.polygons:
                p.material_index = 0
            lows.append(g.low)
        else:
            lows.extend(g.highs)
    keep = set(lows)
    for ob in sc.objects:
        if ob.type == 'MESH' and ob not in keep:
            ob.hide_render = True

    # frame the objects
    pts = np.array([tuple(o.matrix_world @ Vector(c)) for o in lows for c in o.bound_box])
    lo, hi = pts.min(0), pts.max(0)
    center = Vector(((lo + hi) / 2).tolist())
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens)) * min(1.0, height / width)
    dist = radius / math.sin(fov / 2) * 1.05
    y, p = math.radians(yaw), math.radians(pitch)
    d = Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)))
    cam.location = center + d * dist
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    cam_data.clip_start = dist / 1000
    cam_data.clip_end = dist * 10

    # three-point-ish lighting + soft world
    for name, rot, energy in (("key", (0.9, 0.2, 0.6), 3.5), ("rim", (1.2, 0.0, 3.6), 1.5)):
        L = bpy.data.objects.new(name, bpy.data.lights.new(name, 'SUN'))
        L.data.energy = energy
        L.data.angle = math.radians(8)
        L.rotation_euler = (rot[0], rot[1], rot[2] + y)
        sc.collection.objects.link(L)
    w = bpy.data.worlds.new("w")
    sc.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    bg = next(n for n in w.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs['Color'].default_value = (0.18, 0.19, 0.21, 1)
    bg.inputs['Strength'].default_value = 1.0

    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'GPU' if (s.device == 'GPU' and core.enable_gpu()) else 'CPU'
    sc.cycles.samples = samples
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = width, height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX' if 'AgX' in [i.identifier for i in
        sc.view_settings.bl_rna.properties['view_transform'].enum_items] else 'Filmic'
    st = sc.render.image_settings
    st.file_format = 'PNG'
    st.color_mode = 'RGB'
    st.color_depth = '8'
    sc.render.filepath = out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        bpy.ops.render.render(write_still=True)
    except RuntimeError:
        # denoiser unavailable on this machine / build: render without it
        sc.cycles.use_denoising = False
        bpy.ops.render.render(write_still=True)
    return out_path
