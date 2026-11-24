import torch
import numpy as np
from torch.nn import functional as F
from models.network import ColorBranch
from models.brightness import seed_all
from models.mk_mmd_loss import mmd_mk_nd_unbiased
from utils.feature_extraction import chromaticity_features, reference_gate
from utils.color import torch_rgb_to_hsv, torch_rgb_to_lab
from utils.color_response import ColorReference

MMD_KERNEL_BW_HS = [0.02, 0.05, 0.10]
MMD_KERNEL_BW_AB = [2.0, 4.0, 8.0]


def training_bank(original, stage1, ref, seed):
    """Row-stratified, fixed input-only pixels. Group means are robust within each cell."""
    h, w = original.shape[:2]
    rng = np.random.default_rng(seed)
    bands, bins, points = 16, 32, 32
    ys = np.empty((bands, bins, points), np.int64)
    xs = np.empty_like(ys)
    cy = np.empty((bands, 128), np.int64)
    cx = np.empty_like(cy)
    for b in range(bands):
        lo, hi = int(b * h / bands), max(int((b + 1) * h / bands), int(b * h / bands) + 1)
        cy[b] = rng.integers(lo, hi, 128)
        cx[b] = rng.choice(np.flatnonzero(ref), 128)
        for c in range(bins):
            ys[b, c] = rng.integers(lo, hi, points)
            xs[b, c] = rng.integers(
                int(c * w / bins), max(int((c + 1) * w / bins), int(c * w / bins) + 1), points
            )
    rgb = stage1[ys, xs].astype(np.float32) / 255
    source = original[cy, cx].astype(np.float32) / 255
    valid_t = (original[ys, xs].astype(float) @ np.array([0.299, 0.587, 0.114])) > 0.03 * 255
    valid_c = (source @ np.array([0.299, 0.587, 0.114])) > 0.03
    return dict(
        rgb=rgb,
        source=source,
        cols=xs,
        ys=ys,
        valid_t=valid_t,
        valid_c=valid_c,
    )


def robust_mean(values, valid):
    # Last sample axis before the two color coordinates. Trim 25% at either end.
    n = valid.sum(-1)
    masked = torch.where(valid[..., None], values, torch.full_like(values, float("inf")))
    ordered = masked.sort(dim=-2).values
    k = torch.arange(values.shape[-2], device=values.device)
    keep = (k >= (n // 4)[..., None]) & (k < (n - n // 4)[..., None])
    result = (
        torch.where(keep[..., None], ordered, torch.zeros_like(ordered)).sum(-2)
        / keep.sum(-1).clamp_min(1)[..., None]
    )
    return result, n >= 8


def embed_hs(rgb):
    hsv = torch_rgb_to_hsv(rgb)
    ang = 2 * torch.pi * hsv[..., 0]
    sat = hsv[..., 1]
    return torch.stack([sat * ang.cos(), sat * ang.sin(), sat], -1)


def fit_chromaticity(original, stage1, features, ref, edge, device):
    seed, iterations = 2024, 500
    seed_all(seed)
    bank = training_bank(original, stage1, ref, seed)
    target = torch.tensor(bank["rgb"].reshape(-1, 3), device=device)
    response = ColorReference(original, ref, device)
    raw_target = original[bank["ys"], bank["cols"]].reshape(-1, 3)
    context = response.prepare(target, raw_target, bank["ys"].reshape(-1))
    source = torch.tensor(bank["source"].reshape(-1, 3), device=device)
    cols = torch.tensor(bank["cols"].reshape(-1), device=device)
    X = chromaticity_features(original, features, device)
    gate = torch.tensor(reference_gate(ref), device=device)
    target_ids = torch.where(
        torch.tensor(bank["valid_t"].reshape(-1), device=device)
        & torch.tensor(edge, device=device)[cols]
    )[0]
    source_ids = torch.where(torch.tensor(bank["valid_c"].reshape(-1), device=device))[0]
    if min(len(source_ids), len(target_ids)) < 2:
        raise ValueError("Insufficient valid samples for chromaticity MK-MMD")
    ref_ab = torch_rgb_to_lab(source)[:, 1:]
    ref_hs = embed_hs(source)
    local_valid_target = torch.tensor(bank["valid_t"], device=device)
    local_valid_reference = torch.tensor(bank["valid_c"], device=device)
    local_reference, local_good_reference = robust_mean(
        ref_ab.reshape(16, 128, 2), local_valid_reference
    )
    local_active = gate[cols].reshape(16, 32, 32).mean(-1) > 0.1
    local_use = (local_valid_target.sum(-1) >= 8) & local_good_reference[:, None] & local_active
    if not local_use.any():
        raise ValueError("No valid local color cells")
    model = ColorBranch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for step in range(1, iterations + 1):
        optimizer.param_groups[0]["lr"] = 0.002 * (
            0.2 + 0.8 * (1 + np.cos(np.pi * (step - 1) / iterations)) / 2
        )
        fields, bounded = model(X, gate)
        corrected, alpha = response.apply(target, context, fields[cols])
        ab = torch_rgb_to_lab(corrected)[:, 1:]
        hs = embed_hs(corrected)
        ti = target_ids[torch.randperm(len(target_ids), device=device)[:512]]
        si = source_ids[torch.randperm(len(source_ids), device=device)[:512]]
        loss_ab = mmd_mk_nd_unbiased(ref_ab[si], ab[ti], MMD_KERNEL_BW_AB)
        loss_hs = mmd_mk_nd_unbiased(ref_hs[si], hs[ti], MMD_KERNEL_BW_HS)
        local_color, _ = robust_mean(ab.reshape(16, 32, 32, 2), local_valid_target)
        local_anchor = F.smooth_l1_loss(
            local_color[local_use] / 10,
            local_reference[:, None, :].expand_as(local_color)[local_use] / 10,
        )
        normalized = F.interpolate(bounded.T[None], size=129, mode="linear", align_corners=True)[0]
        slope = (normalized[:, 1:] - normalized[:, :-1]) * 128
        curve = (normalized[:, 2:] - 2 * normalized[:, 1:-1] + normalized[:, :-2]) * 128**2
        smooth = 0.00005 * slope.square().mean() + 0.0000002 * curve.square().mean()
        end = 0.0002 * torch.stack([slope[:, 0], slope[:, -1]]).square().mean()
        residual = 0.01 * bounded.square().mean()
        gamut = 0.01 * (1 - alpha).square().mean()
        loss = 1.5 * (loss_ab + loss_hs) + local_anchor + smooth + end + residual + gamut
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        optimizer.zero_grad()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    with torch.no_grad():
        fields, _ = model(X, gate)
    return fields.detach().cpu().numpy()
