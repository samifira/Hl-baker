# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake engine shared by the HL Baker app and the Blender add-on (needs bpy)."""
import bpy
import os
import re
import json
import time
import hashlib
import colorsys
import numpy as np

TMP_TAG = "__hlbake_tmp"

from .config import MAPS, MAP_ORDER, FORMATS, SUFFIX_RE, KIND_MAP  # noqa: E402



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
    if getattr(bpy.data, "filepath", ""):
        return bpy.path.display_name_from_filepath(bpy.data.filepath)
    return "bake"


def group_objects(g):
    highs = [getattr(h, "obj", h) for h in g.highs]
    return g.low, [h for h in highs if h is not None], g.cage


class RuntimeGroup:
    """A bake group bound to real Blender objects."""
    def __init__(self, name, low, highs, cage=None, cage_offset=0.05, max_ray=0.0, enabled=True):
        self.name, self.low, self.highs, self.cage = name, low, list(highs), cage
        self.cage_offset, self.max_ray, self.enabled = cage_offset, max_ray, enabled


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

    def __init__(self, context, s, groups, report=None, progress=None, preview_dir=None):
        self.progress = progress or (lambda **kw: None)
        self.preview_dir = preview_dir
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
        cy.device = 'GPU' if (self.s.device == 'GPU' and enable_gpu()) else 'CPU'
        self.progress(event="log", message=f"Cycles device: {cy.device}"
                      + (f" ({_GPU_BACKEND})" if cy.device == 'GPU' else ""))

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

        total = len(maps) * len(self.groups)
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
                    self.progress(event="progress", step=step, total=total,
                                  map=MAPS[mk][0], group=g.name)
            finally:
                if mk not in ("normal", "normal_obj"):
                    self._restore_high_materials(all_highs)
            self.progress(event="log", message=f"{MAPS[mk][0]}: {time.time() - t0:.1f}s")


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
        prev = None
        if self.preview_dir:
            prev = write_preview(rgb, mk, os.path.join(self.preview_dir, f"{setname}_{suffix}.png"),
                                 is_color and is_float, self.scene)
        self.progress(event="map_done", path=path, preview=prev, map=mk, set=setname)


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



# ---------------------------------------------------------------------------
# App helpers
# ---------------------------------------------------------------------------
_GPU_BACKEND = None


def enable_gpu():
    """Enable the best available Cycles GPU backend. Returns True if one was found."""
    global _GPU_BACKEND
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except Exception:
        return False
    for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = backend
        except Exception:
            continue
        try:
            prefs.refresh_devices()
        except Exception:
            prefs.get_devices()
        devs = [d for d in prefs.devices if d.type == backend]
        if devs:
            for d in prefs.devices:
                d.use = d.type in (backend, 'CPU') if backend != 'METAL' else True
            _GPU_BACKEND = backend
            return True
    try:
        prefs.compute_device_type = 'NONE'
    except Exception:
        pass
    return False


def write_preview(rgb, map_key, path, needs_srgb, scene, max_size=1024):
    """8-bit sRGB PNG thumbnail for the GUI."""
    a = np.asarray(rgb, np.float32)
    h, w = a.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_size)))
    a = a[::step, ::step]
    if needs_srgb:
        a = linear_to_srgb(a)
    if map_key == "position" and (a.max() > 1 or a.min() < 0):
        lo, hi = a.reshape(-1, 3).min(0), a.reshape(-1, 3).max(0)
        a = (a - lo) / np.maximum(hi - lo, 1e-8)
    a = np.clip(a, 0, 1)
    h, w = a.shape[:2]
    rgba = np.concatenate([a, np.ones((h, w, 1), np.float32)], axis=-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = bpy.data.images.new("hlb_prev" + TMP_TAG, w, h, alpha=False, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(rgba.ravel())
    st = scene.render.image_settings
    vs = scene.view_settings
    old = (st.file_format, st.color_depth, st.color_mode, vs.view_transform, vs.look)
    try:
        vs.view_transform = 'Standard'
        vs.look = 'None'
        st.file_format = 'PNG'
        st.color_mode = 'RGB'
        st.color_depth = '8'
        img.save_render(path, scene=scene)
    finally:
        st.file_format, st.color_depth, st.color_mode, vs.view_transform, vs.look = old
        bpy.data.images.remove(img)
    return path
