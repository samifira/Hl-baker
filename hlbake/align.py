# SPDX-License-Identifier: GPL-3.0-or-later
"""High/low parity checks and auto-alignment (numpy only, runs in the GUI process).

All points are world-space. The result of an alignment is a 4x4 similarity matrix D that,
applied (in world space) to the high poly, puts it onto the low poly.
"""
import math
import numpy as np

from . import xform


# ---------------------------------------------------------------------------
# Sampling / nearest neighbours
# ---------------------------------------------------------------------------
def sample_triangles(P, n, rng):
    """Area-weighted random points on a triangle soup (P: (3k, 3) corners)."""
    T = np.asarray(P, np.float64).reshape(-1, 3, 3)
    if len(T) == 0:
        return np.zeros((0, 3))
    a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    if a.sum() <= 0:
        return T.reshape(-1, 3)[:n]
    idx = rng.choice(len(T), size=n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    t = T[idx]
    return t[:, 0] * w0[:, None] + t[:, 1] * w1[:, None] + t[:, 2] * w2[:, None]


def subsample(P, n, rng):
    P = np.asarray(P, np.float64)
    if len(P) <= n:
        return P
    return P[rng.choice(len(P), size=n, replace=False)]


def nearest(A, B, chunk=512):
    """For each row of A, the index of and squared distance to the nearest row of B."""
    bb = (B * B).sum(1)
    idx = np.empty(len(A), np.int64)
    d2 = np.empty(len(A))
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        D = (a * a).sum(1)[:, None] + bb[None, :] - 2.0 * a @ B.T
        j = D.argmin(1)
        idx[i:i + chunk] = j
        d2[i:i + chunk] = np.maximum(D[np.arange(len(a)), j], 0.0)
    return idx, d2


def trimmed_mean_dist(A, B, keep=0.8):
    _, d2 = nearest(A, B)
    d = np.sqrt(np.sort(d2)[: max(1, int(len(d2) * keep))])
    return float(d.mean())


def fit_error(L, H):
    """Symmetric mean distance (low->high trimmed, high->low untrimmed): parts of the high that
    stick out (bolts, handles) must count, they are what tells similar orientations apart."""
    return 0.5 * (trimmed_mean_dist(L, H, 0.9) + trimmed_mean_dist(H, L, 1.0))


# ---------------------------------------------------------------------------
# Similarity transforms
# ---------------------------------------------------------------------------
def similarity(s, R, t):
    M = np.identity(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def umeyama(src, dst, with_scale=True):
    """Least-squares s, R, t with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    C = Y.T @ X / len(src)
    U, S, Vt = np.linalg.svd(C)
    E = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (S * np.diag(E)).sum() / max((X * X).sum() / len(src), 1e-30) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------------------
# Checks + alignment
# ---------------------------------------------------------------------------
def bbox(P):
    P = np.asarray(P)
    return P.min(0), P.max(0)


def quick_check(low_P, high_P):
    """Cheap bounding-box comparison. Returns None if the pair looks fine, else a reason."""
    (l0, l1), (h0, h1) = bbox(low_P), bbox(high_P)
    el, eh = l1 - l0, h1 - h0
    dl, dh = np.linalg.norm(el), np.linalg.norm(eh)
    if dl <= 0 or dh <= 0:
        return None
    ratio = dh / dl
    if not (0.8 < ratio < 1.25):
        return "scale"
    big = max(el.max(), eh.max())
    if np.any(np.abs(el - eh) > 0.1 * big + 1e-9):
        return "axes"
    if np.linalg.norm((l0 + l1) / 2 - (h0 + h1) / 2) > 0.1 * dl:
        return "offset"
    return None


def auto_align(low_P, high_P, low_is_triangles=True, refine=True, seed=0):
    """Find D (4x4) moving the high (world points) onto the low.

    Returns dict: D, scale, snapped (unit label or None), ambiguous (bool), error_before,
    error_after (relative to the low's size), confident (bool).
    """
    rng = np.random.default_rng(seed)
    low_P = np.asarray(low_P, np.float64)
    high_P = np.asarray(high_P, np.float64)
    (l0, l1), (h0, h1) = bbox(low_P), bbox(high_P)
    diag_l = max(float(np.linalg.norm(l1 - l0)), 1e-12)
    diag_h = max(float(np.linalg.norm(h1 - h0)), 1e-12)
    c_l, c_h = (l0 + l1) / 2, (h0 + h1) / 2

    L = sample_triangles(low_P, 1500, rng) if low_is_triangles else subsample(low_P, 1500, rng)
    H = subsample(high_P, 4000, rng)

    # 1) scale from the bounding-box diagonals (invariant under 90 degree rotations)
    s0 = diag_l / diag_h
    snap = xform.snap_unit_factor(s0, tol=0.12)
    s = snap[0] if snap else s0
    snapped = snap[1] if snap and snap[0] != 1.0 else None
    lock_scale = snap is not None          # a known unit factor (or "already right"): keep it

    # 2) coarse: the 24 axis-aligned orientations around the centres
    err_before = fit_error(L, H)
    scores = []
    for Rc in xform.axis_rotations():
        Y = s * (H - c_h) @ Rc.T + c_l
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rc) - 1) / 2))))
        scores.append((fit_error(L, Y), angle, Rc))
    scores.sort(key=lambda x: x[0])
    best = scores[0][0]
    # orientations that fit (almost) equally well are symmetries of the model: prefer the one
    # closest to how it is now, i.e. the smallest rotation
    tol = best * 1.04 + 5e-4 * diag_l
    ties = [x for x in scores if x[0] <= tol]
    ambiguous = len({round(x[1]) for x in ties}) > 1
    R = min(ties, key=lambda x: x[1])[2]
    D = similarity(s, R, c_l - s * R @ c_h)

    # 3) refine with trimmed ICP (scale stays locked to a snapped unit factor)
    if refine:
        L2 = sample_triangles(low_P, 2000, rng) if low_is_triangles else subsample(low_P, 2000, rng)
        H2 = subsample(high_P, 15000, rng)
        Y = xform.apply_points(D, H2)
        for _ in range(30):
            j, d2 = nearest(L2, Y)
            keep = d2 <= np.quantile(d2, 0.8)
            ds, dR, dt = umeyama(Y[j[keep]], L2[keep], with_scale=not lock_scale)
            step = similarity(ds, dR, dt)
            D = step @ D
            Y = xform.apply_points(step, Y)
            if abs(ds - 1) < 1e-7 and np.linalg.norm(dt) < 1e-7 * diag_l and \
                    np.abs(dR - np.identity(3)).max() < 1e-7:
                break
    # prefer the simplest correction that fits as well: unit/axis problems are usually exact
    # (pure scale/rotation about the origin), and ICP likes to add a hair of tilt or drift
    coarse_origin = similarity(s, R, np.zeros(3))
    coarse_center = similarity(s, R, c_l - s * R @ c_h)
    cands = [(fit_error(L, xform.apply_points(M, H)), M)
             for M in (coarse_origin, coarse_center, D)]
    best_err = min(c[0] for c in cands)
    for err, M in cands:                                  # in order of simplicity
        if err <= best_err * 1.05 + 1e-3 * diag_l:
            D = M
            break
    err_after = fit_error(L, xform.apply_points(D, H))
    s_final = float(np.cbrt(abs(np.linalg.det(D[:3, :3]))))
    return {"D": D, "scale": s_final, "snapped": snapped, "ambiguous": bool(ambiguous),
            "error_before": err_before / diag_l, "error_after": err_after / diag_l,
            "confident": err_after / diag_l < 0.1}


def describe(D, snapped=None):
    """Human description of a correction, e.g. 'scale ÷100 (cm → m), rotate 90° about X'."""
    parts = []
    A = D[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(A))))
    if abs(s - 1) > 1e-3:
        parts.append(f"scale ×{s:.4g}" + (f" ({snapped})" if snapped else ""))
    R = A / s
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    if ang > 0.5:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.linalg.norm(axis) < 1e-6:                    # 180 degrees
            axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        k = int(np.argmax(np.abs(axis)))
        if abs(axis[k]) > 0.999:
            sign = "" if axis[k] > 0 else "-"
            parts.append(f"rotate {ang:.0f}° about {sign}{'XYZ'[k]}")
        else:
            parts.append(f"rotate {ang:.1f}°")
    t = D[:3, 3]
    if np.linalg.norm(t) > 1e-6:
        parts.append("move ({:.3g}, {:.3g}, {:.3g})".format(*t))
    return ", ".join(parts) if parts else "no change"


def check(low_P, high_P, seed=0):
    """Parity check for one group. Returns None when high and low already line up, else the
    auto_align() result with an extra "reason" key ("scale" | "axes" | "offset" | "orientation")."""
    reason = quick_check(low_P, high_P)
    if reason is None:
        # same bounding box can still hide a rotation (cube-like models): test the 24 axis
        # orientations quickly and only flag one that fits clearly better than the current one
        rng = np.random.default_rng(seed)
        low_P = np.asarray(low_P, np.float64)
        L = sample_triangles(low_P, 800, rng)
        H = subsample(np.asarray(high_P, np.float64), 2500, rng)
        l0, l1 = bbox(low_P)
        h0, h1 = bbox(H)
        diag = float(np.linalg.norm(l1 - l0))
        c_l, c_h = (l0 + l1) / 2, (h0 + h1) / 2
        current = fit_error(L, H)
        best = min(fit_error(L, (H - c_h) @ Rc.T + c_l) for Rc in xform.axis_rotations())
        if current <= best * 1.3 + 2e-3 * diag:
            return None
        reason = "orientation"
    r = auto_align(low_P, high_P, seed=seed)
    if r["error_after"] >= r["error_before"] * 0.8:       # alignment didn't really help: no issue
        return None
    r["reason"] = reason
    return r
