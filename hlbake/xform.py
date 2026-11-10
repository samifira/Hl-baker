# SPDX-License-Identifier: GPL-3.0-or-later
"""Transforms (location / rotation / scale) shared by the GUI and the worker. numpy only.

A transform is a dict {"loc": [x,y,z], "rot": [x,y,z] degrees, "scale": [x,y,z]}.
Rotation order matches Blender's default XYZ Euler (R = Rz @ Ry @ Rx). Transforms act around the
world origin. An object's own transform is applied first and its file's transform last, like a
parent: moving a file by 1 moves every object in it by exactly 1, corrections included.
"""
import math
import numpy as np


def identity():
    return {"loc": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def normalized(t):
    """Fill in missing keys / coerce types (projects from older versions have no transforms)."""
    out = identity()
    if t:
        for k in out:
            if k in t and len(t[k]) == 3:
                out[k] = [float(v) for v in t[k]]
    return out


def is_identity(t, eps=1e-9):
    t = normalized(t)
    return (max(abs(v) for v in t["loc"]) < eps and max(abs(v) for v in t["rot"]) < eps
            and max(abs(v - 1.0) for v in t["scale"]) < eps)


def rot_matrix(rot_deg):
    x, y, z = (math.radians(a) for a in rot_deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix(t):
    """4x4 float64 matrix  T @ R @ S."""
    t = normalized(t)
    M = np.identity(4)
    M[:3, :3] = rot_matrix(t["rot"]) @ np.diag(t["scale"])
    M[:3, 3] = t["loc"]
    return M


def euler_xyz(R):
    """Rotation matrix -> XYZ Euler angles in degrees (R = Rz @ Ry @ Rx)."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    if abs(sy) < 0.999999:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock: fold everything into z
        x = 0.0
        z = math.atan2(-R[0, 1], R[1, 1])
    return [_clean(math.degrees(a)) for a in (x, y, z)]


def _clean(v, eps=1e-6):
    r = round(v)
    v = r if abs(v - r) < eps else v
    return 0.0 if abs(v) < eps else float(v)


def decompose(M):
    """4x4 -> transform dict. Exact for T @ R @ S with S diagonal (what this app produces)."""
    A = np.asarray(M, np.float64)[:3, :3]
    scale = np.linalg.norm(A, axis=0)
    if np.linalg.det(A) < 0:               # put a mirror into X scale
        scale[0] = -scale[0]
    R = A / np.where(np.abs(scale) > 1e-12, scale, 1.0)
    return {"loc": [_clean(v) for v in np.asarray(M)[:3, 3]], "rot": euler_xyz(R),
            "scale": [_clean(v) for v in scale]}


def combined(file_t, obj_t):
    """World transform applied to an object: its own transform, then its file's (the parent)."""
    return matrix(file_t) @ matrix(obj_t)


def apply_points(M, P):
    P = np.asarray(P, np.float64)
    return P @ M[:3, :3].T + M[:3, 3]


def normal_matrix(M):
    A = np.asarray(M, np.float64)[:3, :3]
    try:
        return np.linalg.inv(A).T
    except np.linalg.LinAlgError:
        return A


# quick-fix presets ------------------------------------------------------------------------
UNIT_FACTORS = [  # (factor, label)
    (100.0, "m → cm (×100)"), (0.01, "cm → m (÷100)"), (10.0, "×10"), (0.1, "÷10"),
    (1000.0, "×1000"), (0.001, "÷1000"), (2.54, "in → cm (×2.54)"), (1 / 2.54, "cm → in (÷2.54)"),
    (0.0254, "in → m (×0.0254)"), (39.3701, "m → in (×39.37)"), (0.3048, "ft → m (×0.3048)"),
    (3.28084, "m → ft (×3.281)"), (30.48, "ft → cm (×30.48)"),
]


def snap_unit_factor(s, tol=0.06):
    """If s is within tol of a common unit factor, return (factor, label), else None."""
    best = None
    for f, label in UNIT_FACTORS + [(1.0, "same")]:
        err = abs(math.log(s / f))
        if err < tol and (best is None or err < best[0]):
            best = (err, f, label)
    return (best[1], best[2]) if best else None


def axis_rotations():
    """The 24 proper rotations that map axes onto axes."""
    mats = []
    import itertools
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                mats.append(R)
    return mats
