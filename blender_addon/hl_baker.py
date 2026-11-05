# SPDX-License-Identifier: GPL-3.0-or-later
"""
HL Baker - a Toolbag-style high-to-low map baker for Blender (Cycles).

Workflow
  1. Name meshes  <name>_low / <name>_high  (also _lp/_hp, _lo/_hi, optional _cage)
  2. 3D View > Sidebar (N) > "HL Baker" > Quick Load   (or build groups by hand)
  3. Tick the maps you want, set resolution / samples / padding / output folder
  4. Bake.  Optionally "Apply Preview" to see the result on the low poly.
"""

bl_info = {
    "name": "HL Baker (Toolbag-style map baker)",
    "author": "Claude",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > HL Baker",
    "description": "Bake groups, cages, and a full map set (normals, AO, curvature, "
                   "thickness, height, position, IDs, PBR channels) from high to low poly",
    "category": "Render",
}

import bpy
import os
import re
import json
import time
import hashlib
import colorsys
import numpy as np
from bpy.props import (StringProperty, IntProperty, FloatProperty, BoolProperty,
                       EnumProperty, PointerProperty, CollectionProperty)

TMP_TAG = "__hlbake_tmp"

# ---------------------------------------------------------------------------
# Map definitions
# ---------------------------------------------------------------------------
# key: (label, file suffix, is_color(sRGB), background RGB)
MAPS = {
    "normal":      ("Normal (Tangent)",  "normal",      False, (0.5, 0.5, 1.0)),
    "normal_obj":  ("Normal (Object)",   "normal_obj",  False, (0.5, 0.5, 0.5)),
    "ao":          ("Ambient Occlusion", "ao",          False, (1.0, 1.0, 1.0)),
    "curvature":   ("Curvature",         "curvature",   False, (0.5, 0.5, 0.5)),
    "thickness":   ("Thickness",         "thickness",   False, (0.0, 0.0, 0.0)),
    "height":      ("Height",            "height",      False, (0.5, 0.5, 0.5)),
    "position":    ("Position",          "position",    False, (0.0, 0.0, 0.0)),
    "material_id": ("Material ID",       "matid",       True,  (0.0, 0.0, 0.0)),
    "object_id":   ("Object ID",         "objid",       True,  (0.0, 0.0, 0.0)),
    "albedo":      ("Albedo",            "albedo",      True,  (0.5, 0.5, 0.5)),
    "roughness":   ("Roughness",         "roughness",   False, (0.5, 0.5, 0.5)),
    "metalness":   ("Metalness",         "metalness",   False, (0.0, 0.0, 0.0)),
    "emissive":    ("Emissive",          "emissive",    True,  (0.0, 0.0, 0.0)),
    "opacity":     ("Opacity",           "opacity",     False, (1.0, 1.0, 1.0)),
}
MAP_ORDER = list(MAPS.keys())

FORMATS = {
    # key: (blender file_format, color_depth, extension, is_float_file)
    "PNG8":   ("PNG", "8", "png", False),
    "PNG16":  ("PNG", "16", "png", False),
    "TGA":    ("TARGA", "8", "tga", False),
    "TIFF16": ("TIFF", "16", "tif", False),
    "EXR32":  ("OPEN_EXR", "32", "exr", True),
}

SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)[ _.\-]+(?P<kind>low|lp|lo|high|hp|hi|cage)(?:[ _.\-]*\d+)?$",
    re.IGNORECASE)
KIND_MAP = {"low": "low", "lp": "low", "lo": "low",
            "high": "high", "hp": "high", "hi": "high", "cage": "cage"}


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
def _mesh_poll(self, obj):
    return obj.type == 'MESH'


class HLB_HighItem(bpy.types.PropertyGroup):
    obj: PointerProperty(type=bpy.types.Object, poll=_mesh_poll, name="High")


class HLB_Group(bpy.types.PropertyGroup):
    name: StringProperty(default="group")
    enabled: BoolProperty(default=True, name="Enabled")
    low: PointerProperty(type=bpy.types.Object, poll=_mesh_poll, name="Low")
    highs: CollectionProperty(type=HLB_HighItem)
    cage: PointerProperty(type=bpy.types.Object, poll=_mesh_poll, name="Cage",
                          description="Optional cage mesh (same topology as the low poly)")
    cage_offset: FloatProperty(name="Cage Offset", default=0.05, min=0.0, soft_max=1.0,
                               update=lambda self, ctx: _update_cage_previews(ctx.scene),
                               unit='LENGTH', precision=4,
                               description="Outer distance: rays start this far outside the low poly")
    max_ray: FloatProperty(name="Max Ray Distance", default=0.0, min=0.0, unit='LENGTH',
                           precision=4,
                           description="Total ray length (0 = unlimited). Use to stop rays hitting "
                                       "far-away surfaces")


class HLB_Settings(bpy.types.PropertyGroup):
    groups: CollectionProperty(type=HLB_Group)
    group_index: IntProperty(default=0)

    # --- output ---
    output_dir: StringProperty(name="Folder", subtype='DIR_PATH', default="//bakes/")
    base_name: StringProperty(name="Name", default="",
                              description="Texture set name. Empty = blend file name")
    output_mode: EnumProperty(name="Output", items=[
        ('SINGLE', "Single Texture Set", "All groups share one UV layout and one set of textures"),
        ('PER_GROUP', "Per Group", "Each group writes its own textures (<group>_<map>)"),
    ], default='SINGLE')
    width: IntProperty(name="Width", default=2048, min=8, max=16384)
    height: IntProperty(name="Height", default=2048, min=8, max=16384)
    supersample: EnumProperty(name="Samples (AA)", items=[
        ('1', "1x", "No supersampling"), ('2', "4x", "Bake at 2x size, downsample"),
        ('4', "16x", "Bake at 4x size, downsample (slow, memory heavy)")],
        default='2', description="Anti-aliasing by supersampling, like Toolbag's sample count")
    ray_samples: IntProperty(name="Ray Samples", default=64, min=1, max=4096,
                             description="Cycles samples for ray-traced maps (AO, thickness, curvature)")
    padding: IntProperty(name="Padding", default=16, min=0, max=1024, subtype='PIXEL',
                         description="Edge padding (dilation) in output pixels")
    padding_full: BoolProperty(name="Fill Entire Texture", default=False,
                               description="Dilate until the whole texture is filled (Toolbag 'Extreme')")
    file_format: EnumProperty(name="Format", items=[
        ('PNG8', "PNG 8-bit", ""), ('PNG16', "PNG 16-bit", ""), ('TGA', "Targa 8-bit", ""),
        ('TIFF16', "TIFF 16-bit", ""), ('EXR32', "OpenEXR 32-bit float", "")], default='PNG16')
    device: EnumProperty(name="Device", items=[('CPU', "CPU", ""), ('GPU', "GPU", "")], default='GPU')
    use_uv: StringProperty(name="UV Map", default="",
                           description="UV map on the low poly to bake into. Empty = active render UV")

    # --- maps ---
    map_normal: BoolProperty(name="Normal (Tangent)", default=True)
    map_normal_obj: BoolProperty(name="Normal (Object)", default=False)
    map_ao: BoolProperty(name="Ambient Occlusion", default=True)
    map_curvature: BoolProperty(name="Curvature", default=True)
    map_thickness: BoolProperty(name="Thickness", default=False)
    map_height: BoolProperty(name="Height", default=False)
    map_position: BoolProperty(name="Position", default=False)
    map_material_id: BoolProperty(name="Material ID", default=False)
    map_object_id: BoolProperty(name="Object ID", default=False)
    map_albedo: BoolProperty(name="Albedo", default=False)
    map_roughness: BoolProperty(name="Roughness", default=False)
    map_metalness: BoolProperty(name="Metalness", default=False)
    map_emissive: BoolProperty(name="Emissive", default=False)
    map_opacity: BoolProperty(name="Opacity", default=False)

    # --- map options ---
    normal_flip_y: EnumProperty(name="Y Channel", items=[
        ('OPENGL', "OpenGL (Y+)", "Blender, Unity, Maya, Toolbag default"),
        ('DIRECTX', "DirectX (Y-)", "Unreal, 3ds Max, Source")], default='OPENGL')
    ao_distance: FloatProperty(name="Distance", default=0.5, min=0.0001, unit='LENGTH')
    ao_ignore_groups: BoolProperty(name="Ignore Groups", default=False,
                                   description="Let high polys from ALL groups occlude each other "
                                               "(off = each group occludes only itself)")
    ao_only_self: BoolProperty(name="Self Only", default=False,
                               description="Each high poly mesh only occludes itself")
    curv_method: EnumProperty(name="Method", items=[
        ('RAY', "Ray Traced", "Resolution independent: probes inside/outside occlusion at a small radius"),
        ('POINTINESS', "Vertex (Pointiness)", "Fast; needs a dense high poly")], default='RAY')
    curv_radius: FloatProperty(name="Radius", default=0.01, min=0.00001, unit='LENGTH', precision=4)
    curv_strength: FloatProperty(name="Strength", default=3.0, min=0.0, soft_max=10.0)
    thick_distance: FloatProperty(name="Distance", default=0.2, min=0.0001, unit='LENGTH')
    height_auto: BoolProperty(name="Auto Range", default=True,
                              description="Normalise so the largest offset fills the 0-1 range")
    height_inner: FloatProperty(name="Inner", default=0.05, min=0.0, unit='LENGTH', precision=4)
    height_outer: FloatProperty(name="Outer", default=0.05, min=0.0, unit='LENGTH', precision=4)
    position_space: EnumProperty(name="Space", items=[
        ('BOUNDS', "Normalized (Bounds)", "0-1 inside the bounding box of all high polys"),
        ('WORLD', "World (raw)", "Raw world coordinates - use EXR")], default='BOUNDS')
    matid_source: EnumProperty(name="Colors", items=[
        ('HASH', "Generated", "Distinct colours generated from material names"),
        ('VIEWPORT', "Viewport Display", "Use each material's Viewport Display colour")], default='HASH')

    show_maps: BoolProperty(default=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def enabled_maps(s):
    return [k for k in MAP_ORDER if getattr(s, "map_" + k)]


def name_color(name, salt=""):
    """Distinct, saturated sRGB colour from a name (golden-ratio hue spacing)."""
    h = int(hashlib.md5((salt + name).encode()).hexdigest()[:8], 16)
    hue = (h * 0.618033988749895) % 1.0
    sat = 0.65 + ((h >> 8) % 30) / 100.0
    val = 0.75 + ((h >> 16) % 25) / 100.0
    return colorsys.hsv_to_rgb(hue, sat, val)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def new_node_material(name):
    mat = bpy.data.materials.new(name)
    try:
        if not mat.use_nodes:
            mat.use_nodes = True
    except Exception:
        pass
    mat[TMP_TAG] = True
    return mat


def clear_nodes(nt):
    for n in list(nt.nodes):
        nt.nodes.remove(n)


def remove_last_slot(ob):
    """Remove the last material slot cleanly (mesh.materials.pop leaves a stale slot)."""
    if len(ob.material_slots) == 0:
        return
    try:
        ob.active_material_index = len(ob.material_slots) - 1
        with bpy.context.temp_override(object=ob, active_object=ob):
            bpy.ops.object.material_slot_remove()
    except Exception:
        ob.data.materials.pop(index=len(ob.data.materials) - 1)


def texture_set_name(s):
    if s.base_name.strip():
        return bpy.path.clean_name(s.base_name.strip())
    if bpy.data.filepath:
        return bpy.path.display_name_from_filepath(bpy.data.filepath)
    return "bake"


def group_objects(g):
    highs = [h.obj for h in g.highs if h.obj is not None]
    return g.low, highs, g.cage


# ---------------------------------------------------------------------------
# Image post-processing (numpy)
# ---------------------------------------------------------------------------
def _downsample2(arr, w):
    """2x2 box downsample of (H, W, C) array, weights w (H, W). Pads odd sizes."""
    H, W = w.shape
    if H % 2 or W % 2:
        arr = np.pad(arr, ((0, H % 2), (0, W % 2), (0, 0)))
        w = np.pad(w, ((0, H % 2), (0, W % 2)))
    H2, W2 = w.shape[0] // 2, w.shape[1] // 2
    aw = (arr * w[..., None]).reshape(H2, 2, W2, 2, -1).sum(axis=(1, 3))
    ws = w.reshape(H2, 2, W2, 2).sum(axis=(1, 3))
    out = np.where(ws[..., None] > 0, aw / np.maximum(ws, 1e-12)[..., None], 0.0)
    return out.astype(np.float32), (ws > 0).astype(np.float32)


def push_pull_fill(rgb, mask):
    """Fill every empty pixel from its nearest valid neighbourhood (pyramid push-pull)."""
    levels = [(rgb, mask.astype(np.float32))]
    a, m = rgb, mask.astype(np.float32)
    while min(m.shape) > 1:
        a, m = _downsample2(a, m)
        levels.append((a, m))
    filled = levels[-1][0]
    if levels[-1][1].max() == 0:
        return rgb
    for a, m in reversed(levels[:-1]):
        H, W = m.shape
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)[:H, :W]
        filled = np.where(m[..., None] > 0, a, up)
    return filled.astype(np.float32)


def dilate_mask(mask, iterations):
    m = mask.copy()
    for _ in range(iterations):
        n = m.copy()
        n[1:, :] |= m[:-1, :]
        n[:-1, :] |= m[1:, :]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        n[1:, 1:] |= m[:-1, :-1]
        n[:-1, :-1] |= m[1:, 1:]
        n[1:, :-1] |= m[:-1, 1:]
        n[:-1, 1:] |= m[1:, :-1]
        if np.array_equal(n, m):
            break
        m = n
    return m


def pad_image(rgb, mask, pad_px, full, background):
    if not mask.any():
        out = np.empty_like(rgb)
        out[...] = background
        return out
    filled = push_pull_fill(rgb, mask)
    if full:
        return filled
    reach = dilate_mask(mask, pad_px) if pad_px > 0 else mask
    out = np.where(reach[..., None], filled, np.asarray(background, np.float32))
    return out.astype(np.float32)


def downsample_box(rgb, factor):
    if factor == 1:
        return rgb
    H, W, C = rgb.shape
    return rgb.reshape(H // factor, factor, W // factor, factor, C).mean(axis=(1, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Material builders for emission-based passes
# ---------------------------------------------------------------------------
PRINCIPLED_INPUTS = {
    "albedo": ["Base Color"],
    "roughness": ["Roughness"],
    "metalness": ["Metallic"],
    "opacity": ["Alpha"],
    "emissive": ["Emission Color", "Emission"],
}


def find_principled(nt):
    out = None
    for n in nt.nodes:
        if n.type == 'OUTPUT_MATERIAL' and n.is_active_output:
            out = n
            break
    if out is None:
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if out is not None and out.inputs['Surface'].is_linked:
        src = out.inputs['Surface'].links[0].from_node
        if src.type == 'BSDF_PRINCIPLED':
            return src
    return next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)


def _input(node, names):
    for nm in names:
        if nm in node.inputs:
            return node.inputs[nm]
    return None


def _feed(nt, sock_src, dst_socket, as_color):
    """Connect a principled input (linked or default value) into dst_socket."""
    if sock_src.is_linked:
        nt.links.new(sock_src.links[0].from_socket, dst_socket)
    else:
        v = sock_src.default_value
        try:
            v = tuple(v)[:3]
        except TypeError:
            v = (v, v, v)
        dst_socket.default_value = (v[0], v[1], v[2], 1.0)


def build_emission_material(src_mat, map_key, s, ctx):
    """Return a temporary material emitting the data for map_key."""
    if src_mat is not None and src_mat.node_tree is not None:
        mat = src_mat.copy()
        mat.name = f"{src_mat.name}{TMP_TAG}_{map_key}"
        mat[TMP_TAG] = True
        nt = mat.node_tree
        principled = find_principled(nt)
    else:
        mat = new_node_material(f"fallback{TMP_TAG}_{map_key}")
        nt = mat.node_tree
        clear_nodes(nt)
        principled = None

    for n in [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL']:
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.target = 'ALL'
    out.is_active_output = True
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.0
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    col = em.inputs['Color']
    L = nt.links.new

    if map_key in ("albedo", "roughness", "metalness", "opacity"):
        sock = _input(principled, PRINCIPLED_INPUTS[map_key]) if principled else None
        if sock is not None:
            _feed(nt, sock, col, map_key == "albedo")
        else:  # no principled: fall back to viewport values
            m = src_mat
            if map_key == "albedo":
                c = tuple(m.diffuse_color)[:3] if m else (0.8, 0.8, 0.8)
            elif map_key == "roughness":
                c = (m.roughness,) * 3 if m else (0.5,) * 3
            elif map_key == "metalness":
                c = (m.metallic,) * 3 if m else (0.0,) * 3
            else:
                c = (m.diffuse_color[3],) * 3 if m else (1.0,) * 3
            col.default_value = (*c, 1.0)

    elif map_key == "emissive":
        ec = _input(principled, PRINCIPLED_INPUTS["emissive"]) if principled else None
        es = _input(principled, ["Emission Strength"]) if principled else None
        if ec is None:
            col.default_value = (0, 0, 0, 1)
        else:
            mul = nt.nodes.new('ShaderNodeVectorMath')
            mul.operation = 'MULTIPLY'
            if ec.is_linked:
                L(ec.links[0].from_socket, mul.inputs[0])
            else:
                mul.inputs[0].default_value = tuple(ec.default_value)[:3]
            if es is None:
                mul.inputs[1].default_value = (1, 1, 1)
            elif es.is_linked:
                L(es.links[0].from_socket, mul.inputs[1])
            else:
                mul.inputs[1].default_value = (es.default_value,) * 3
            L(mul.outputs[0], col)

    elif map_key == "material_id":
        if s.matid_source == 'VIEWPORT' and src_mat is not None:
            c = tuple(src_mat.diffuse_color)[:3]
        else:
            c = tuple(srgb_to_linear(name_color(src_mat.name if src_mat else "None", "mat")))
        col.default_value = (*c, 1.0)

    elif map_key == "object_id":
        oi = nt.nodes.new('ShaderNodeObjectInfo')
        L(oi.outputs['Color'], col)

    elif map_key in ("position", "height_pos"):
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        L(geo.outputs['Position'], col)

    elif map_key == "ao":
        ao = nt.nodes.new('ShaderNodeAmbientOcclusion')
        ao.inputs['Distance'].default_value = s.ao_distance
        ao.only_local = s.ao_only_self
        ao.samples = 1
        L(ao.outputs['AO'], col)

    elif map_key == "thickness":
        ao = nt.nodes.new('ShaderNodeAmbientOcclusion')
        ao.inside = True
        ao.only_local = True
        ao.samples = 1
        ao.inputs['Distance'].default_value = s.thick_distance
        inv = nt.nodes.new('ShaderNodeMath')
        inv.operation = 'SUBTRACT'
        inv.inputs[0].default_value = 1.0
        L(ao.outputs['AO'], inv.inputs[1])
        L(inv.outputs[0], col)

    elif map_key == "curvature":
        if s.curv_method == 'POINTINESS':
            geo = nt.nodes.new('ShaderNodeNewGeometry')
            src = geo.outputs['Pointiness']
            k = s.curv_strength * 4.0
        else:
            ao_out = nt.nodes.new('ShaderNodeAmbientOcclusion')
            ao_out.only_local = True
            ao_out.samples = 1
            ao_out.inputs['Distance'].default_value = s.curv_radius
            ao_in = nt.nodes.new('ShaderNodeAmbientOcclusion')
            ao_in.only_local = True
            ao_in.inside = True
            ao_in.samples = 1
            ao_in.inputs['Distance'].default_value = s.curv_radius
            # convex edges -> inside rays hit quickly; concave -> outside rays hit quickly
            diff = nt.nodes.new('ShaderNodeMath')
            diff.operation = 'SUBTRACT'
            L(ao_out.outputs['AO'], diff.inputs[0])
            L(ao_in.outputs['AO'], diff.inputs[1])
            src = diff.outputs[0]
            k = s.curv_strength
        if s.curv_method == 'POINTINESS':
            sub = nt.nodes.new('ShaderNodeMath')
            sub.operation = 'SUBTRACT'
            L(src, sub.inputs[0])
            sub.inputs[1].default_value = 0.5
            src = sub.outputs[0]
        ma = nt.nodes.new('ShaderNodeMath')
        ma.operation = 'MULTIPLY_ADD'
        ma.use_clamp = True
        L(src, ma.inputs[0])
        ma.inputs[1].default_value = 0.5 * k if s.curv_method == 'RAY' else k
        ma.inputs[2].default_value = 0.5
        L(ma.outputs[0], col)

    elif map_key == "low_pos":
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        L(geo.outputs['Position'], col)

    elif map_key == "low_nrm":
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        L(geo.outputs['Normal'], col)

    return mat


# ---------------------------------------------------------------------------
# The bake engine
# ---------------------------------------------------------------------------
class BakeError(Exception):
    pass


class BakeSession:
    """Owns all temporary state for a bake and restores the scene afterwards."""

    def __init__(self, context, s, groups, report):
        self.ctx = context
        self.scene = context.scene
        self.vl = context.view_layer
        self.s = s
        self.groups = groups
        self.report = report
        self.ss = int(s.supersample)
        self.W = s.width * self.ss
        self.H = s.height * self.ss
        self._saved_scene = {}
        self._saved_objs = {}
        self._saved_slots = {}
        self._added_slots = []
        self._saved_uv = {}
        self._tmp_ids = []
        self.target = None
        self.target_mat = None
        self.written = []

    # ---------------- state save / restore ----------------
    def _remember_obj(self, ob):
        if ob.name in self._saved_objs:
            return
        self._saved_objs[ob.name] = dict(
            hide_render=ob.hide_render, hide_viewport=ob.hide_viewport,
            hide=ob.hide_get(), select=ob.select_get(),
            vis={a: getattr(ob, a) for a in VIS_ATTRS},
            color=tuple(ob.color),
        )

    def _remember_slots(self, ob):
        if ob.name not in self._saved_slots:
            self._saved_slots[ob.name] = [sl.material for sl in ob.material_slots]

    def setup(self):
        sc = self.scene
        cy = sc.cycles
        b = sc.render.bake
        self._saved_scene = dict(
            engine=sc.render.engine, samples=cy.samples, device=cy.device,
            active=self.vl.objects.active,
            bake={k: getattr(b, k) for k in BAKE_ATTRS if hasattr(b, k)},
        )
        for ob in self.vl.objects:
            self._remember_obj(ob)
        sc.render.engine = 'CYCLES'
        cy.device = self.s.device
        if self.s.device == 'GPU':
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                if prefs.compute_device_type == 'NONE':
                    cy.device = 'CPU'
            except Exception:
                cy.device = 'CPU'

        # target image + target material for the low polys
        self.target = bpy.data.images.new(f"hlbake{TMP_TAG}", self.W, self.H,
                                          alpha=True, float_buffer=True)
        self.target.colorspace_settings.name = 'Non-Color'
        self._tmp_ids.append(self.target)
        self.target_mat = self._make_target_material("target", None)

        # every renderable object not part of the bake is hidden from render
        involved = set()
        for g in self.groups:
            low, highs, cage = group_objects(g)
            involved.update(o.name for o in [low, *highs] if o)
        for ob in self.vl.objects:
            if ob.name not in involved and ob.type in RENDERABLE:
                ob.hide_render = True

        # lows: point every slot at the target material, make invisible to rays
        for g in self.groups:
            low = g.low
            self._remember_slots(low)
            if len(low.material_slots) == 0:
                low.data.materials.append(self.target_mat)
                self._added_slots.append(low)
            else:
                for sl in low.material_slots:
                    sl.material = self.target_mat
            for a in VIS_ATTRS:
                setattr(low, a, False)
            if self.s.use_uv and self.s.use_uv in low.data.uv_layers:
                self._saved_uv[low.name] = low.data.uv_layers.active.name
                low.data.uv_layers.active = low.data.uv_layers[self.s.use_uv]
            for ob in group_objects(g)[1]:
                self._remember_slots(ob)

    def _make_target_material(self, tag, emit_key):
        mat = new_node_material(f"{tag}{TMP_TAG}")
        nt = mat.node_tree
        clear_nodes(nt)
        if emit_key is not None:
            mat2 = build_emission_material(None, emit_key, self.s, self)
            bpy.data.materials.remove(mat)
            mat = mat2
            nt = mat.node_tree
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = self.target
        nt.nodes.active = tex
        if emit_key is None:
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
            nt.links.new(dif.outputs[0], out.inputs['Surface'])
        self._tmp_ids.append(mat)
        return mat

    def restore(self):
        sc = self.scene
        # material slots
        for ob in self._added_slots:
            try:
                remove_last_slot(ob)
            except Exception:
                pass
        for name, mats in self._saved_slots.items():
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            for sl, m in zip(ob.material_slots, mats):
                if sl.material != m:
                    sl.material = m
        for name, uv in self._saved_uv.items():
            ob = bpy.data.objects.get(name)
            if ob and uv in ob.data.uv_layers:
                ob.data.uv_layers.active = ob.data.uv_layers[uv]
        # objects
        for name, st in self._saved_objs.items():
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            ob.hide_render = st['hide_render']
            ob.hide_viewport = st['hide_viewport']
            try:
                ob.hide_set(st['hide'])
                ob.select_set(st['select'])
            except RuntimeError:
                pass
            for a, v in st['vis'].items():
                setattr(ob, a, v)
            ob.color = st['color']
        # scene
        ss = self._saved_scene
        if ss:
            b = sc.render.bake
            for k, v in ss['bake'].items():
                try:
                    setattr(b, k, v)
                except Exception:
                    pass
            sc.cycles.samples = ss['samples']
            sc.cycles.device = ss['device']
            sc.render.engine = ss['engine']
            try:
                self.vl.objects.active = ss['active']
            except Exception:
                pass
        # temp datablocks
        for idb in self._tmp_ids:
            try:
                if isinstance(idb, bpy.types.Image):
                    bpy.data.images.remove(idb)
                elif isinstance(idb, bpy.types.Material):
                    bpy.data.materials.remove(idb)
            except ReferenceError:
                pass
        for m in [m for m in bpy.data.materials if m.get(TMP_TAG)]:
            bpy.data.materials.remove(m)

    # ---------------- per-map material swapping ----------------
    def _swap_high_materials(self, map_key, highs):
        cache = {}
        for ob in highs:
            if len(ob.material_slots) == 0:
                m = cache.get(None) or build_emission_material(None, map_key, self.s, self)
                cache[None] = m
                ob.data.materials.append(m)
                self._added_slots.append(ob)
                continue
            saved = self._saved_slots.get(ob.name)
            for i, sl in enumerate(ob.material_slots):
                src = saved[i] if saved and i < len(saved) else sl.material
                key = src.name if src else None
                if key not in cache:
                    cache[key] = build_emission_material(src, map_key, self.s, self)
                sl.material = cache[key]

    def _restore_high_materials(self, highs):
        for ob in highs:
            if ob in self._added_slots:
                remove_last_slot(ob)
                self._added_slots.remove(ob)
            mats = self._saved_slots.get(ob.name, [])
            for sl, m in zip(ob.material_slots, mats):
                sl.material = m
        for m in [m for m in bpy.data.materials if m.get(TMP_TAG) and m not in self._tmp_ids]:
            bpy.data.materials.remove(m)

    # ---------------- core bake call ----------------
    def _visible_for(self, g, wide):
        """Set render visibility for baking group g. wide = all highs visible (AO ignore groups)."""
        for gg in self.groups:
            low, highs, cage = group_objects(gg)
            same = gg == g
            low.hide_render = not same
            for h in highs:
                h.hide_render = not (same or wide)

    def _select(self, objs, active):
        for ob in self.vl.objects:
            if ob.select_get():
                ob.select_set(False)
        for ob in objs:
            ob.hide_viewport = False
            ob.hide_set(False)
            ob.select_set(True)
        self.vl.objects.active = active

    def _clear_target(self):
        self.target.pixels.foreach_set(np.zeros(self.W * self.H * 4, np.float32))

    def _read_target(self):
        buf = np.empty(self.W * self.H * 4, np.float32)
        self.target.pixels.foreach_get(buf)
        return buf.reshape(self.H, self.W, 4)

    def bake_group(self, g, bake_type, selected_to_active=True, normal_space='TANGENT'):
        low, highs, cage = group_objects(g)
        self._clear_target()
        objs = [*highs, low] if selected_to_active else [low]
        self._select(objs, low)
        kw = dict(type=bake_type, target='IMAGE_TEXTURES', use_clear=False, margin=0,
                  use_selected_to_active=selected_to_active)
        if selected_to_active:
            kw.update(cage_extrusion=g.cage_offset, max_ray_distance=g.max_ray)
            if cage is not None:
                kw.update(use_cage=True, cage_object=cage.name)
            else:
                kw.update(use_cage=False)
        if bake_type == 'NORMAL':
            kw.update(normal_space=normal_space, normal_r='POS_X', normal_b='POS_Z',
                      normal_g='NEG_Y' if (normal_space == 'TANGENT'
                                           and self.s.normal_flip_y == 'DIRECTX') else 'POS_Y')
        if not selected_to_active:
            for a in VIS_ATTRS:
                setattr(low, a, True)
        try:
            res = bpy.ops.object.bake(**kw)
        finally:
            if not selected_to_active:
                for a in VIS_ATTRS:
                    setattr(low, a, False)
        if 'FINISHED' not in res:
            raise BakeError(f"Bake failed for group '{g.name}'")
        return self._read_target()

    # ---------------- map drivers ----------------
    def run(self):
        s = self.s
        maps = enabled_maps(s)
        sets = {}  # set name -> {map_key: (rgb, mask)}
        if s.output_mode == 'SINGLE':
            set_of = {g.name: texture_set_name(s) for g in self.groups}
        else:
            set_of = {g.name: bpy.path.clean_name(g.name) for g in self.groups}

        all_highs = [h for g in self.groups for h in group_objects(g)[1]]
        bounds = None
        if "position" in maps:
            pts = np.array([ob.matrix_world @ v.co for ob in all_highs for v in ob.data.vertices],
                           np.float32) if all_highs else np.zeros((1, 3), np.float32)
            bounds = (pts.min(axis=0), pts.max(axis=0))

        wm = self.ctx.window_manager
        total = len(maps) * len(self.groups)
        wm.progress_begin(0, max(total, 1))
        step = 0
        for mk in maps:
            t0 = time.time()
            for setname in set(set_of.values()):
                sets.setdefault(setname, {})[mk] = (
                    np.zeros((self.H, self.W, 3), np.float32), np.zeros((self.H, self.W), bool))

            ray_map = mk in ("ao", "thickness", "curvature") and not (
                mk == "curvature" and s.curv_method == 'POINTINESS')
            self.scene.cycles.samples = s.ray_samples if ray_map else 1

            if mk not in ("normal", "normal_obj"):
                self._swap_high_materials("height_pos" if mk == "height" else mk, all_highs)
            if mk == "object_id":
                for ob in all_highs:
                    ob.color = (*srgb_to_linear(name_color(ob.name, "obj")), 1.0)
            try:
                for g in self.groups:
                    wide = (mk == "ao" and s.ao_ignore_groups)
                    self._visible_for(g, wide)
                    if mk == "normal":
                        px = self.bake_group(g, 'NORMAL', normal_space='TANGENT')
                    elif mk == "normal_obj":
                        px = self.bake_group(g, 'NORMAL', normal_space='OBJECT')
                    elif mk == "height":
                        px = self._bake_height(g)
                    else:
                        px = self.bake_group(g, 'EMIT')
                    rgb, mask = sets[set_of[g.name]][mk]
                    hit = px[..., 3] > 0.5
                    rgb[hit] = px[..., :3][hit]
                    mask |= hit
                    step += 1
                    wm.progress_update(step)
            finally:
                if mk not in ("normal", "normal_obj"):
                    self._restore_high_materials(all_highs)
            print(f"[HL Baker] {MAPS[mk][0]}: {time.time() - t0:.1f}s")

        wm.progress_end()

        # post-process + write
        for setname, mapdata in sets.items():
            for mk, (rgb, mask) in mapdata.items():
                rgb = self._finalize_values(mk, rgb, mask, bounds)
                self._write(setname, mk, rgb, mask)

    def _bake_height(self, g):
        """Height = offset of the high surface along the low normal (signed)."""
        low = g.low
        hp = self.bake_group(g, 'EMIT')               # high positions (world), highs swapped
        mats = []
        for key in ("low_pos", "low_nrm"):
            m = self._make_target_material(key, key)
            for sl in low.material_slots:
                sl.material = m
            mats.append(self.bake_group(g, 'EMIT', selected_to_active=False))
        for sl in low.material_slots:
            sl.material = self.target_mat
        lp, ln = mats
        n = ln[..., :3]
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        h = ((hp[..., :3] - lp[..., :3]) * n).sum(axis=-1)
        out = np.zeros_like(hp)
        out[..., 0] = out[..., 1] = out[..., 2] = h
        out[..., 3] = hp[..., 3]
        return out

    def _finalize_values(self, mk, rgb, mask, bounds):
        s = self.s
        if mk == "height":
            h = rgb[..., 0]
            if s.height_auto:
                # robust: ignore the few rays that travel far (misses / wrong hits)
                m = float(np.percentile(np.abs(h[mask]), 99.5)) if mask.any() else 1.0
                inner = outer = max(m, 1e-8)
            else:
                inner, outer = max(s.height_inner, 1e-8), max(s.height_outer, 1e-8)
            v = np.clip((h + inner) / (inner + outer), 0, 1)
            rgb = np.repeat(v[..., None], 3, axis=-1).astype(np.float32)
        elif mk == "position" and s.position_space == 'BOUNDS' and bounds is not None:
            lo, hi = bounds
            rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-8), 0, 1).astype(np.float32)
        return rgb

    def _write(self, setname, mk, rgb, mask):
        s = self.s
        label, suffix, is_color, bg = MAPS[mk]
        fmt, depth, ext, is_float = FORMATS[s.file_format]
        pad = s.padding * self.ss
        rgb = pad_image(rgb, mask, pad, s.padding_full, bg)
        rgb = downsample_box(rgb, self.ss)
        if mk in ("normal", "normal_obj"):
            v = rgb * 2 - 1
            v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
            rgb = v * 0.5 + 0.5
        if is_color and not is_float:
            rgb = linear_to_srgb(rgb)
        if not is_float:
            rgb = np.clip(rgb, 0, 1)
        h, w = rgb.shape[:2]
        rgba = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], axis=-1)

        folder = bpy.path.abspath(s.output_dir)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{setname}_{suffix}.{ext}")

        img = bpy.data.images.new(f"hlbake_out{TMP_TAG}", w, h, alpha=False, float_buffer=True)
        img.colorspace_settings.name = 'Non-Color'
        img.pixels.foreach_set(rgba.astype(np.float32).ravel())
        st = self.scene.render.image_settings
        vs = self.scene.view_settings
        saved = dict(fmt=st.file_format, depth=st.color_depth, mode=st.color_mode,
                     vt=vs.view_transform, look=vs.look, exp=vs.exposure, gam=vs.gamma)
        try:
            vs.view_transform = 'Standard'
            vs.look = 'None'
            vs.exposure = 0.0
            vs.gamma = 1.0
            st.file_format = fmt
            st.color_mode = 'RGB'
            st.color_depth = depth
            if fmt == 'OPEN_EXR':
                st.exr_codec = 'ZIP'
            img.save_render(path, scene=self.scene)
        finally:
            st.file_format = saved['fmt']
            st.color_mode = saved['mode']
            try:
                st.color_depth = saved['depth']
            except Exception:
                pass
            vs.view_transform = saved['vt']
            vs.look = saved['look']
            vs.exposure = saved['exp']
            vs.gamma = saved['gam']
            bpy.data.images.remove(img)
        self.written.append(path)


VIS_ATTRS = ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
             "visible_volume_scatter", "visible_shadow")
BAKE_ATTRS = ("use_selected_to_active", "use_cage", "cage_extrusion", "max_ray_distance",
              "cage_object", "margin", "use_clear", "normal_space", "normal_r", "normal_g",
              "normal_b", "target")
RENDERABLE = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'CURVES', 'POINTCLOUD', 'VOLUME',
              'GPENCIL', 'GREASEPENCIL'}


def validate(context, s, groups):
    errs = []
    vl = context.view_layer
    if not groups:
        errs.append("No enabled bake groups.")
    if not enabled_maps(s):
        errs.append("No maps enabled.")
    for g in groups:
        low, highs, cage = group_objects(g)
        if low is None:
            errs.append(f"Group '{g.name}': no low poly set.")
            continue
        if not highs:
            errs.append(f"Group '{g.name}': no high poly set.")
        if len(low.data.uv_layers) == 0:
            errs.append(f"Group '{g.name}': low poly '{low.name}' has no UV map.")
        for ob in [low, *highs, *( [cage] if cage else [])]:
            if ob.name not in vl.objects:
                errs.append(f"'{ob.name}' is not in the current view layer (excluded collection?).")
        if low in highs:
            errs.append(f"Group '{g.name}': low poly is also listed as a high poly.")
        if cage is not None and len(cage.data.vertices) != len(low.data.vertices):
            errs.append(f"Group '{g.name}': cage vertex count differs from the low poly.")
    if int(s.supersample) * max(s.width, s.height) > 16384:
        errs.append("Resolution x supersampling exceeds 16384 px - lower one of them.")
    return errs


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class HLB_OT_bake(bpy.types.Operator):
    bl_idname = "hlbake.bake"
    bl_label = "Bake"
    bl_description = "Bake all enabled maps for all enabled groups"
    bl_options = {'REGISTER'}

    only_active: BoolProperty(default=False, options={'SKIP_SAVE'})

    def execute(self, context):
        s = context.scene.hl_baker
        if self.only_active:
            groups = [s.groups[s.group_index]] if 0 <= s.group_index < len(s.groups) else []
        else:
            groups = [g for g in s.groups if g.enabled]
        errs = validate(context, s, groups)
        if errs:
            for e in errs:
                self.report({'ERROR'}, e)
            return {'CANCELLED'}
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        sess = BakeSession(context, s, groups, self.report)
        t0 = time.time()
        try:
            sess.setup()
            sess.run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Bake failed: {e}")
            return {'CANCELLED'}
        finally:
            sess.restore()
        for p in sess.written:
            # refresh any already-loaded copies (e.g. the preview material)
            for im in bpy.data.images:
                if bpy.path.abspath(im.filepath) == p:
                    im.reload()
        self.report({'INFO'}, f"Baked {len(sess.written)} maps in {time.time() - t0:.1f}s "
                              f"-> {bpy.path.abspath(s.output_dir)}")
        return {'FINISHED'}


class HLB_OT_group_add(bpy.types.Operator):
    bl_idname = "hlbake.group_add"
    bl_label = "Add Group"
    bl_description = "New group. Active object = low poly, other selected meshes = high polys"

    def execute(self, context):
        s = context.scene.hl_baker
        g = s.groups.add()
        act = context.active_object
        sel = [o for o in context.selected_objects if o.type == 'MESH']
        if act and act.type == 'MESH' and act in sel:
            g.low = act
            m = SUFFIX_RE.match(act.name)
            g.name = m.group('base') if m else act.name
            for o in sel:
                if o != act:
                    g.highs.add().obj = o
        else:
            g.name = f"group{len(s.groups)}"
        s.group_index = len(s.groups) - 1
        return {'FINISHED'}


class HLB_OT_group_remove(bpy.types.Operator):
    bl_idname = "hlbake.group_remove"
    bl_label = "Remove Group"

    def execute(self, context):
        s = context.scene.hl_baker
        if 0 <= s.group_index < len(s.groups):
            s.groups.remove(s.group_index)
            s.group_index = max(0, min(s.group_index, len(s.groups) - 1))
        return {'FINISHED'}


class HLB_OT_group_clear(bpy.types.Operator):
    bl_idname = "hlbake.group_clear"
    bl_label = "Clear All Groups"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        context.scene.hl_baker.groups.clear()
        return {'FINISHED'}


class HLB_OT_quick_load(bpy.types.Operator):
    bl_idname = "hlbake.quick_load"
    bl_label = "Quick Load"
    bl_description = ("Build bake groups from object names: <name>_low, <name>_high[_N], <name>_cage "
                      "(also _lp/_hp, _lo/_hi). Uses the selection if anything is selected")
    bl_options = {'REGISTER', 'UNDO'}

    replace: BoolProperty(name="Replace Existing Groups", default=True)

    def execute(self, context):
        s = context.scene.hl_baker
        pool = [o for o in context.selected_objects if o.type == 'MESH'] or \
               [o for o in context.view_layer.objects if o.type == 'MESH']
        found = {}
        for ob in pool:
            m = SUFFIX_RE.match(ob.name)
            if not m:
                continue
            base = m.group('base')
            kind = KIND_MAP[m.group('kind').lower()]
            d = found.setdefault(base.lower(), {"name": base, "low": None, "high": [], "cage": None})
            if kind == "high":
                d["high"].append(ob)
            else:
                d[kind] = ob
        if self.replace:
            s.groups.clear()
        made, skipped = 0, []
        for key in sorted(found):
            d = found[key]
            if d["low"] is None or not d["high"]:
                skipped.append(d["name"])
                continue
            g = s.groups.add()
            g.name = d["name"]
            g.low = d["low"]
            for h in sorted(d["high"], key=lambda o: o.name):
                g.highs.add().obj = h
            g.cage = d["cage"]
            made += 1
        s.group_index = max(0, len(s.groups) - 1)
        msg = f"Quick Load: {made} group(s)"
        if skipped:
            msg += f"; incomplete (skipped): {', '.join(skipped)}"
        self.report({'INFO'} if made else {'WARNING'}, msg)
        return {'FINISHED'}


class HLB_OT_high_add(bpy.types.Operator):
    bl_idname = "hlbake.high_add"
    bl_label = "Add Selected as High"
    bl_description = "Add selected meshes (except the group's low) as high polys of the active group"

    def execute(self, context):
        s = context.scene.hl_baker
        if not (0 <= s.group_index < len(s.groups)):
            return {'CANCELLED'}
        g = s.groups[s.group_index]
        have = {h.obj for h in g.highs}
        for o in context.selected_objects:
            if o.type == 'MESH' and o != g.low and o != g.cage and o not in have:
                g.highs.add().obj = o
        return {'FINISHED'}


class HLB_OT_high_remove(bpy.types.Operator):
    bl_idname = "hlbake.high_remove"
    bl_label = "Remove High"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.hl_baker
        g = s.groups[s.group_index]
        if 0 <= self.index < len(g.highs):
            g.highs.remove(self.index)
        return {'FINISHED'}


class HLB_OT_cage_preview(bpy.types.Operator):
    bl_idname = "hlbake.cage_preview"
    bl_label = "Toggle Cage Preview"
    bl_description = "Show/hide a wireframe preview of the ray cage for the active group"

    def execute(self, context):
        s = context.scene.hl_baker
        if not (0 <= s.group_index < len(s.groups)):
            return {'CANCELLED'}
        g = s.groups[s.group_index]
        name = f"{g.name}_cagepreview{TMP_TAG}"
        ob = bpy.data.objects.get(name)
        if ob is not None:
            bpy.data.objects.remove(ob)
            return {'FINISHED'}
        src = g.cage or g.low
        if src is None:
            return {'CANCELLED'}
        ob = bpy.data.objects.new(name, src.data)
        ob.matrix_world = src.matrix_world
        context.scene.collection.objects.link(ob)
        ob.display_type = 'WIRE'
        ob.show_in_front = True
        ob.hide_render = True
        ob.hide_select = True
        ob.color = (1.0, 0.3, 0.1, 1.0)
        if g.cage is None:
            mod = ob.modifiers.new("cage", 'DISPLACE')
            mod.direction = 'NORMAL'
            mod.mid_level = 0.0
            mod.strength = g.cage_offset
            ob["hlb_group"] = g.name
        return {'FINISHED'}


def _update_cage_previews(scene):
    s = scene.hl_baker
    for g in s.groups:
        ob = bpy.data.objects.get(f"{g.name}_cagepreview{TMP_TAG}")
        if ob and "cage" in ob.modifiers:
            if abs(ob.modifiers["cage"].strength - g.cage_offset) > 1e-9:
                ob.modifiers["cage"].strength = g.cage_offset


class HLB_OT_preview(bpy.types.Operator):
    bl_idname = "hlbake.preview"
    bl_label = "Apply Preview Material"
    bl_description = ("Assign a material using the baked maps to every low poly "
                      "(originals are remembered and restored by 'Remove Preview')")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.hl_baker
        fmt, depth, ext, _ = FORMATS[s.file_format]
        folder = bpy.path.abspath(s.output_dir)
        n = 0
        for g in s.groups:
            if not g.enabled or g.low is None:
                continue
            setname = texture_set_name(s) if s.output_mode == 'SINGLE' else bpy.path.clean_name(g.name)

            def tex(suffix):
                p = os.path.join(folder, f"{setname}_{suffix}.{ext}")
                return p if os.path.exists(p) else None
            mat = _preview_material(setname, tex, s)
            low = g.low
            if "hlb_orig_mats" not in low:
                low["hlb_orig_mats"] = json.dumps(
                    [sl.material.name if sl.material else "" for sl in low.material_slots])
            if len(low.material_slots) == 0:
                low.data.materials.append(mat)
            for sl in low.material_slots:
                sl.material = mat
            n += 1
        self.report({'INFO'}, f"Preview applied to {n} low poly object(s)")
        return {'FINISHED'}


def _preview_material(setname, tex, s):
    name = f"HLBake_Preview_{setname}"
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    clear_nodes(nt)
    L = nt.links.new
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (600, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (250, 0)
    L(bsdf.outputs[0], out.inputs['Surface'])

    def img_node(path, noncolor, y):
        n = nt.nodes.new('ShaderNodeTexImage')
        n.location = (-500, y)
        im = bpy.data.images.load(path, check_existing=True)
        im.reload()
        im.colorspace_settings.name = 'Non-Color' if noncolor else 'sRGB'
        n.image = im
        return n

    p_alb, p_ao = tex("albedo"), tex("ao")
    base = None
    if p_alb:
        base = img_node(p_alb, False, 400).outputs['Color']
    if p_ao:
        ao = img_node(p_ao, True, 150)
        mix = nt.nodes.new('ShaderNodeMix'); mix.location = (-150, 300)
        mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
        mix.inputs['Factor'].default_value = 1.0
        if base is not None:
            L(base, mix.inputs['A'])
        else:
            mix.inputs['A'].default_value = (0.6, 0.6, 0.6, 1)
        L(ao.outputs['Color'], mix.inputs['B'])
        base = mix.outputs['Result']
    if base is not None:
        L(base, bsdf.inputs['Base Color'])
    for key, sock in (("roughness", "Roughness"), ("metalness", "Metallic")):
        p = tex(key)
        if p:
            L(img_node(p, True, -150 if key == "roughness" else -350).outputs['Color'], bsdf.inputs[sock])
    p_n = tex("normal")
    if p_n:
        n = img_node(p_n, True, -600)
        col = n.outputs['Color']
        if s.normal_flip_y == 'DIRECTX':
            sep = nt.nodes.new('ShaderNodeSeparateColor'); sep.location = (-250, -600)
            inv = nt.nodes.new('ShaderNodeMath'); inv.operation = 'SUBTRACT'; inv.location = (-100, -650)
            inv.inputs[0].default_value = 1.0
            com = nt.nodes.new('ShaderNodeCombineColor'); com.location = (50, -600)
            L(col, sep.inputs[0]); L(sep.outputs[0], com.inputs[0]); L(sep.outputs[1], inv.inputs[1])
            L(inv.outputs[0], com.inputs[1]); L(sep.outputs[2], com.inputs[2])
            col = com.outputs[0]
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (50, -400)
        if s.use_uv:
            nm.uv_map = s.use_uv
        L(col, nm.inputs['Color'])
        L(nm.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


class HLB_OT_preview_remove(bpy.types.Operator):
    bl_idname = "hlbake.preview_remove"
    bl_label = "Remove Preview"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.hl_baker
        for g in s.groups:
            low = g.low
            if low is None or "hlb_orig_mats" not in low:
                continue
            names = json.loads(low["hlb_orig_mats"])
            if not any(names):
                while len(low.material_slots):
                    remove_last_slot(low)
            else:
                for sl, nm in zip(low.material_slots, names):
                    sl.material = bpy.data.materials.get(nm) if nm else None
            del low["hlb_orig_mats"]
        return {'FINISHED'}


class HLB_OT_open_folder(bpy.types.Operator):
    bl_idname = "hlbake.open_folder"
    bl_label = "Open Output Folder"

    def execute(self, context):
        p = bpy.path.abspath(context.scene.hl_baker.output_dir)
        os.makedirs(p, exist_ok=True)
        bpy.ops.wm.path_open(filepath=p)
        return {'FINISHED'}


class HLB_OT_maps_preset(bpy.types.Operator):
    bl_idname = "hlbake.maps_preset"
    bl_label = "Map Preset"
    preset: EnumProperty(items=[('NONE', "None", ""), ('GAME', "Game Ready", ""),
                                ('ALL', "All", "")])

    def execute(self, context):
        s = context.scene.hl_baker
        game = {"normal", "ao", "curvature", "thickness", "material_id", "position"}
        for k in MAP_ORDER:
            setattr(s, "map_" + k, self.preset == 'ALL' or (self.preset == 'GAME' and k in game))
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class HLB_UL_groups(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "name", text="", emboss=False, icon='MESH_DATA')
        nh = len([h for h in item.highs if h.obj])
        ok = item.low is not None and nh > 0
        row.label(text=f"{nh} hi", icon='CHECKMARK' if ok else 'ERROR')


class HLB_PT_main(bpy.types.Panel):
    bl_label = "HL Baker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "HL Baker"

    def draw(self, context):
        s = context.scene.hl_baker
        L = self.layout
        col = L.column(align=True)
        col.scale_y = 1.6
        col.operator("hlbake.bake", text="Bake", icon='RENDER_STILL')
        row = L.row(align=True)
        row.operator("hlbake.bake", text="Bake Active Group").only_active = True
        row.operator("hlbake.open_folder", text="", icon='FILE_FOLDER')
        row = L.row(align=True)
        row.operator("hlbake.preview", icon='SHADING_TEXTURE')
        row.operator("hlbake.preview_remove", text="", icon='X')


class HLB_PT_groups(bpy.types.Panel):
    bl_label = "Bake Groups"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "HL Baker"
    bl_parent_id = "HLB_PT_main"

    def draw(self, context):
        s = context.scene.hl_baker
        L = self.layout
        L.operator("hlbake.quick_load", icon='AUTO')
        row = L.row()
        row.template_list("HLB_UL_groups", "", s, "groups", s, "group_index", rows=4)
        c = row.column(align=True)
        c.operator("hlbake.group_add", text="", icon='ADD')
        c.operator("hlbake.group_remove", text="", icon='REMOVE')
        c.separator()
        c.operator("hlbake.group_clear", text="", icon='TRASH')
        if not (0 <= s.group_index < len(s.groups)):
            return
        g = s.groups[s.group_index]
        box = L.box()
        box.prop(g, "low", icon='MESH_CUBE')
        box.label(text="High Poly:")
        hc = box.column(align=True)
        for i, h in enumerate(g.highs):
            r = hc.row(align=True)
            r.prop(h, "obj", text="")
            r.operator("hlbake.high_remove", text="", icon='X').index = i
        box.operator("hlbake.high_add", icon='ADD')
        box.separator()
        box.prop(g, "cage_offset")
        box.prop(g, "max_ray")
        box.prop(g, "cage")
        prev = bpy.data.objects.get(f"{g.name}_cagepreview{TMP_TAG}") is not None
        box.operator("hlbake.cage_preview", text="Hide Cage" if prev else "Show Cage",
                     icon='HIDE_OFF' if prev else 'HIDE_ON', depress=prev)


class HLB_PT_output(bpy.types.Panel):
    bl_label = "Output"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "HL Baker"
    bl_parent_id = "HLB_PT_main"

    def draw(self, context):
        s = context.scene.hl_baker
        L = self.layout
        L.use_property_split = True
        L.use_property_decorate = False
        L.prop(s, "output_dir")
        L.prop(s, "base_name")
        L.prop(s, "output_mode")
        L.prop(s, "file_format")
        c = L.column(align=True)
        c.prop(s, "width")
        c.prop(s, "height")
        L.prop(s, "supersample", expand=True)
        L.prop(s, "ray_samples")
        r = L.row(align=True)
        r.prop(s, "padding")
        r.active = not s.padding_full
        L.prop(s, "padding_full")
        L.prop(s, "device", expand=True)
        g = s.groups[s.group_index] if 0 <= s.group_index < len(s.groups) else None
        if g and g.low:
            L.prop_search(s, "use_uv", g.low.data, "uv_layers", text="UV Map")
        else:
            L.prop(s, "use_uv")


class HLB_PT_maps(bpy.types.Panel):
    bl_label = "Maps"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "HL Baker"
    bl_parent_id = "HLB_PT_main"

    def draw(self, context):
        s = context.scene.hl_baker
        L = self.layout
        r = L.row(align=True)
        r.operator("hlbake.maps_preset", text="None").preset = 'NONE'
        r.operator("hlbake.maps_preset", text="Game").preset = 'GAME'
        r.operator("hlbake.maps_preset", text="All").preset = 'ALL'

        def item(key, draw_opts=None):
            b = L.box() if (draw_opts and getattr(s, "map_" + key)) else L
            b.prop(s, "map_" + key)
            if draw_opts and getattr(s, "map_" + key):
                c = b.column()
                c.use_property_split = True
                c.use_property_decorate = False
                draw_opts(c)

        item("normal", lambda c: c.prop(s, "normal_flip_y"))
        item("normal_obj")
        item("ao", lambda c: (c.prop(s, "ao_distance"), c.prop(s, "ao_ignore_groups"),
                              c.prop(s, "ao_only_self")))
        item("curvature", lambda c: (c.prop(s, "curv_method"),
                                     c.prop(s, "curv_radius") if s.curv_method == 'RAY' else None,
                                     c.prop(s, "curv_strength")))
        item("thickness", lambda c: c.prop(s, "thick_distance"))

        def h(c):
            c.prop(s, "height_auto")
            if not s.height_auto:
                c.prop(s, "height_inner")
                c.prop(s, "height_outer")
        item("height", h)
        item("position", lambda c: c.prop(s, "position_space"))
        item("material_id", lambda c: c.prop(s, "matid_source"))
        item("object_id")
        L.separator()
        L.label(text="From High Poly Materials:")
        for k in ("albedo", "roughness", "metalness", "emissive", "opacity"):
            item(k)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
CLASSES = (
    HLB_HighItem, HLB_Group, HLB_Settings,
    HLB_OT_bake, HLB_OT_group_add, HLB_OT_group_remove, HLB_OT_group_clear,
    HLB_OT_quick_load, HLB_OT_high_add, HLB_OT_high_remove, HLB_OT_cage_preview,
    HLB_OT_preview, HLB_OT_preview_remove, HLB_OT_open_folder, HLB_OT_maps_preset,
    HLB_UL_groups, HLB_PT_main, HLB_PT_groups, HLB_PT_output, HLB_PT_maps,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.hl_baker = PointerProperty(type=HLB_Settings)


def unregister():
    del bpy.types.Scene.hl_baker
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
