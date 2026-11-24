import numpy as np
import cv2
import torch
from utils.color import torch_rgb_to_lab


def to_float01(rgb):
    return rgb.astype(np.float32) / 255.0


def rgb_to_hsv01(rgb01):
    hsv = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(
        np.float32
    )
    return hsv[..., 0] / 180.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def soft_in_orange(h, s):
    # locate the orange/red diffuse reflection from the core box
    hc = 19 / 360.0
    k = 40.0
    dh = np.minimum(np.abs(h - hc), 1.0 - np.abs(h - hc))
    mem_h = np.exp(-((dh * 360.0 / k) ** 2))
    gate = np.clip((s - 0.2) / 0.3, 0.0, 1.0)
    return mem_h * gate


def _dilate1d_bool(mask, r):
    if r <= 0:
        return mask
    kern = np.ones(2 * r + 1, np.int32)
    vals = np.convolve(mask.astype(np.int32), kern, mode="same")
    return vals > 0


def build_masks_and_ref(
    rgb,
    ref_width_frac,
    lum_weight,
    edge_outer_frac=0.3,
    edge_thres_pct=70,
    edge_min_frac=0.12,
    edge_expand_frac=0.05,
):
    # establish source (center) and target (edges) domain masks based on stats
    H, W, _ = rgb.shape
    rgb01 = to_float01(rgb)
    h, s, _ = rgb_to_hsv01(rgb01)
    Y = 0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]

    valid = Y > 0.03
    Y_top = np.zeros(W, np.float32)
    for x in range(W):
        col = Y[:, x][valid[:, x]]
        if col.size < 10:
            col = Y[:, x]
        k = max(1, int(0.2 * col.size))
        Y_top[x] = float(np.partition(col, -k)[-k:].mean())

    xs = np.arange(W)
    dist = np.minimum(xs, W - 1 - xs).astype(np.float32)
    outer = dist <= edge_outer_frac * W
    Yn = (Y_top - Y_top.min()) / (Y_top.max() - Y_top.min() + 1e-6)

    mem = soft_in_orange(h, s)
    mem_col = mem.mean(axis=0)
    score = np.zeros(W, np.float32)

    score[outer] = lum_weight * (1.0 - Yn[outer]) + (1.0 - lum_weight) * mem_col[outer]
    thr = np.percentile(score[outer], edge_thres_pct)
    edge_cols = score >= thr

    min_w = int(edge_min_frac * W)
    if edge_cols[: W // 2].sum() < min_w:
        edge_cols[:min_w] = True
    if edge_cols[W // 2 :].sum() < min_w:
        edge_cols[W - min_w :] = True
    edge_cols = _dilate1d_bool(edge_cols, int(edge_expand_frac * W))

    peak_idx = np.argmax(Y_top)
    half_ref_w = max(1, int(W * ref_width_frac))
    l = max(0, peak_idx - half_ref_w)
    r = min(W, peak_idx + half_ref_w)

    ref_cols = np.zeros(W, np.bool_)
    ref_cols[l:r] = True

    allow = np.zeros(W, np.bool_)
    allow[:l] = True
    allow[r:] = True
    edge_cols = edge_cols & allow

    return valid, Y_top, ref_cols, edge_cols


def build_stage1_features(rgb, valid, Y_top):
    # flatten the 2d core image into 1d column sequences
    H, W, _ = rgb.shape
    xs = np.linspace(0, 1, W, dtype=np.float32)
    rgb01 = to_float01(rgb)
    h, s, v = rgb_to_hsv01(rgb01)

    mem = soft_in_orange(h, s)
    mem_mu = mem.mean(axis=0)
    s_mu = s.mean(axis=0)

    gx = cv2.Sobel((v * 255.0).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel((v * 255.0).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy).mean(axis=0)
    if gm.max() > 0:
        gm = (gm - gm.min()) / (gm.max() - gm.min() + 1e-6)

    Yt = (Y_top - Y_top.min()) / (Y_top.max() - Y_top.min() + 1e-6)
    feats = np.stack([xs, Yt, mem_mu, s_mu, gm], axis=1).astype(np.float32)
    return feats, {"Y_top": Y_top}


def _scan_left_right_spans(edge_cols: np.ndarray, ref_cols: np.ndarray, min_w: int):
    W = edge_cols.size
    c_idx = np.where(ref_cols)[0]
    l_edge = c_idx[0]
    r_edge = c_idx[-1] + 1
    # Left
    i = l_edge - 1
    while i >= 0 and not edge_cols[i]:
        i -= 1
    if i >= 0:
        j = i
        while j >= 0 and edge_cols[j]:
            j -= 1
        L_start, L_end = j + 1, l_edge - 1
    else:
        L_end = l_edge - 1
        L_start = max(0, L_end - min_w + 1)
    if L_end - L_start + 1 < min_w:
        L_start = max(0, L_end - min_w + 1)
    # Right
    i = r_edge
    while i < W and not edge_cols[i]:
        i += 1
    if i < W:
        j = i
        while j < W and edge_cols[j]:
            j += 1
        R_start, R_end = r_edge, j - 1
    else:
        R_start = r_edge
        R_end = min(W - 1, R_start + min_w - 1)
    if R_end - R_start + 1 < min_w:
        R_end = min(W - 1, R_start + min_w - 1)
    return (L_start, L_end), (R_start, R_end)


def _spans_to_mask(left_span, right_span, W):
    mask = np.zeros(W, dtype=bool)
    Ls, Le = left_span
    Rs, Re = right_span
    if Le >= Ls:
        mask[Ls : Le + 1] = True
    if Re >= Rs:
        mask[Rs : Re + 1] = True
    return mask


def smoothstep(value):
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def reference_gate(ref):
    """C1 continuous gate in normalized column coordinates, zero in reference."""
    positions = np.flatnonzero(ref)
    if not len(positions) or np.any(np.diff(positions) != 1):
        raise ValueError("Expected one nonempty contiguous reference interval")
    x = np.arange(len(ref), dtype=np.float32)
    distance = np.maximum(positions[0] - x, x - positions[-1]).clip(0)
    return smoothstep(distance / max(0.08 * (len(ref) - 1), 1.0)).astype(np.float32)


def column_features(rgb, reference_width):
    if len(rgb) < 2 or rgb.shape[1] < 3:
        raise ValueError("Column estimation requires at least two rows and three columns")
    valid, top, ref, edge = build_masks_and_ref(rgb, reference_width / 2, lum_weight=0.7)
    features, _ = build_stage1_features(rgb, valid, top)
    left, right = _scan_left_right_spans(edge, ref, int(0.12 * rgb.shape[1]))
    edge = _spans_to_mask(left, right, rgb.shape[1])
    if not np.any(valid & ref[None, :]) or not np.any(valid & edge[None, :]):
        raise ValueError("Empty reference or target pixel pool")
    return features, top, ref, edge, valid


def chromaticity_features(original, features, device):
    rows = np.linspace(0, len(original) - 1, min(256, len(original))).astype(int)
    sample = torch.tensor(original[rows].reshape(-1, 3).astype(np.float32) / 255, device=device)
    lab = torch_rgb_to_lab(sample).reshape(len(rows), original.shape[1], 3)
    medians = (lab.median(0).values / lab.new_tensor([100, 40, 40])).cpu().numpy()
    return torch.tensor(np.concatenate([features, medians], 1).T[None], device=device)
