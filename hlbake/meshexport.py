# SPDX-License-Identifier: GPL-3.0-or-later
"""Export display meshes for the live viewer (runs in the worker, needs bpy).

Low polys are triangulated and exported per corner with Blender's MikkTSpace tangents,
the same tangent space Cycles bakes the normal map in. High polys are exported indexed
and simplified by vertex clustering when they are too heavy to display.
"""
import json
import numpy as np
import bpy

HIGH_TRI_LIMIT = 3_000_000


def _world_mats(ob):
    M = np.array(ob.matrix_world, dtype=np.float64)
    M3 = M[:3, :3]
    try:
        Ninv = np.linalg.inv(M3).T
    except np.linalg.LinAlgError:
        Ninv = M3
    return M, M3, Ninv


def _xf_points(P, M):
    return (P @ M[:3, :3].T + M[:3, 3]).astype(np.float32)


def _xf_dirs(D, M3):
    D = D @ M3.T
    return (D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _evaluated_triangulated(ob):
    """Mesh copy of the evaluated object, triangulated, custom normals kept."""
    tmp = ob.copy()
    tmp.data = ob.data
    mod = tmp.modifiers.new("hlb_tri", 'TRIANGULATE')
    mod.quad_method = 'FIXED'
    mod.ngon_method = 'BEAUTY'
    if hasattr(mod, "keep_custom_normals"):
        mod.keep_custom_normals = True
    bpy.context.scene.collection.objects.link(tmp)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = tmp.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    bpy.data.objects.remove(tmp)
    return me


def _corner_normals(me):
    n = len(me.loops)
    out = np.empty(n * 3, np.float32)
    if hasattr(me, "corner_normals"):          # Blender 4.1+
        me.corner_normals.foreach_get("vector", out)
    else:
        me.calc_normals_split()
        me.loops.foreach_get("normal", out)
    return out.reshape(-1, 3)


def _vertex_normals(me):
    out = np.empty(len(me.vertices) * 3, np.float32)
    if hasattr(me, "vertex_normals"):
        me.vertex_normals.foreach_get("vector", out)
    else:
        me.vertices.foreach_get("normal", out)
    return out.reshape(-1, 3)


def export_low(ob, uv_name=""):
    me = _evaluated_triangulated(ob)
    try:
        uvl = me.uv_layers.get(uv_name) if uv_name else None
        if uvl is None:
            uvl = next((u for u in me.uv_layers if u.active_render), None) or me.uv_layers.active
        nl = len(me.loops)
        vidx = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", vidx)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        N = _corner_normals(me)
        VN = _vertex_normals(me)[vidx]
        if uvl is not None:
            uv = np.empty(nl * 2, np.float32)
            uvl.data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)
            try:
                me.calc_tangents(uvmap=uvl.name)
                T = np.empty(nl * 3, np.float32)
                me.loops.foreach_get("tangent", T)
                T = T.reshape(-1, 3)
                S = np.empty(nl, np.float32)
                me.loops.foreach_get("bitangent_sign", S)
            except RuntimeError:
                T, S = _fallback_tangents(N), np.ones(nl, np.float32)
        else:
            uv = np.zeros((nl, 2), np.float32)
            T, S = _fallback_tangents(N), np.ones(nl, np.float32)
        # loops of a triangulated mesh, in polygon order -> 3 corners per triangle
        loop_start = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_start", loop_start)
        order = (loop_start[:, None] + np.arange(3)[None, :]).ravel()
        M, M3, Ninv = _world_mats(ob)
        return {
            "P": _xf_points(co[vidx][order].astype(np.float64), M),
            "N": _xf_dirs(N[order].astype(np.float64), Ninv),
            "T": _xf_dirs(T[order].astype(np.float64), M3),
            "S": (S[order] * np.sign(np.linalg.det(M3) or 1.0)).astype(np.float32),
            "UV": uv[order].astype(np.float32),
            "VN": _xf_dirs(VN[order].astype(np.float64), Ninv),
        }
    finally:
        bpy.data.meshes.remove(me)


def _fallback_tangents(N):
    a = np.where(np.abs(N[:, 2:3]) < 0.9, [[0, 0, 1]], [[1, 0, 0]]).astype(np.float32)
    T = np.cross(a, N)
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)


def export_high(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tri = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        VN = _vertex_normals(me)
        M, M3, Ninv = _world_mats(ob)
        return _xf_points(co.astype(np.float64), M), _xf_dirs(VN.astype(np.float64), Ninv), \
            tri.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()


def cluster_decimate(P, N, F, target):
    """Vertex-clustering simplification to roughly `target` triangles (fast, display only)."""
    if len(F) <= target:
        return P, N, F
    lo, hi = P.min(0), P.max(0)
    ext = np.maximum(hi - lo, 1e-9)
    cells = max(8, int(np.sqrt(target / 2.0) * 1.2))      # surface ~ 2*cells^2 tris
    for _ in range(6):
        size = ext.max() / cells
        key = np.floor((P - lo) / size).astype(np.int64)
        dims = key.max(0) + 1
        k = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
        uniq, inv = np.unique(k, return_inverse=True)
        F2 = inv[F]
        keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
        F2 = F2[keep]
        if len(F2) <= target * 1.1 or cells <= 8:
            break
        cells = int(cells * 0.8)
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n).astype(np.float32)[:, None]
    P2 = np.zeros((n, 3), np.float64)
    N2 = np.zeros((n, 3), np.float64)
    np.add.at(P2, inv, P)
    np.add.at(N2, inv, N)
    P2 /= cnt
    N2 /= np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-12)
    return P2.astype(np.float32), N2.astype(np.float32), F2.astype(np.int32)


def export_display_meshes(groups, out_path, setname_of, uv_name=""):
    """groups: list of core.RuntimeGroup. Writes one .npz for the viewer."""
    arrays, meta = {}, {"groups": [], "high_decimated": False}
    total_high = sum(len(h.data.polygons) * 2 for g in groups for h in g.highs)
    budget_ratio = min(1.0, HIGH_TRI_LIMIT / max(total_high, 1))
    lo_all, hi_all = [], []
    for gi, g in enumerate(groups):
        low = export_low(g.low, uv_name)
        for k, v in low.items():
            arrays[f"g{gi}_low_{k}"] = v
        lo_all.append(low["P"].min(0)); hi_all.append(low["P"].max(0))
        highs = []
        for hi_i, h in enumerate(g.highs):
            P, N, F = export_high(h)
            if budget_ratio < 1.0:
                P, N, F = cluster_decimate(P, N, F, max(2000, int(len(F) * budget_ratio)))
                meta["high_decimated"] = True
            arrays[f"g{gi}_high{hi_i}_P"] = P
            arrays[f"g{gi}_high{hi_i}_N"] = N
            arrays[f"g{gi}_high{hi_i}_F"] = F.astype(np.uint32)
            highs.append(h.name)
            lo_all.append(P.min(0)); hi_all.append(P.max(0))
        meta["groups"].append({"name": g.name, "low": g.low.name, "highs": highs,
                               "set": setname_of(g), "cage_offset": g.cage_offset})
    if lo_all:
        meta["bounds"] = [np.min(lo_all, 0).tolist(), np.max(hi_all, 0).tolist()]
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(out_path, **arrays)
    return out_path
