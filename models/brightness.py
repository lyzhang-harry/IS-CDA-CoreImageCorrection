import math
import random
import numpy as np
import torch
from models.sampling import TorchPermutationPrefix
from models.network import ColumnBackbone, brightness_head
from models.mk_mmd_loss import mmd_mk_1d_unbiased
from utils.color import torch_smoothstep
from utils.feature_extraction import to_float01

LR_BB, LR_Y = 4e-4, 3e-4
S1_REF_BOOST = 1.02
HL1_Y_LOW, HL1_Y_HIGH, HL1_STRENGTH = 0.5, 0.9, 0.80
W_Y_ALIGN, W_Y_ID, W_Y_SMOOTH, W_Y_FLAT, W_Y_TV, W_MMD_Y = 1.6, 0.4, 0.03, 0.4, 0.1, 1.5
MMD_KERNEL_BW_Y = [0.01, 0.02, 0.05, 0.10]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cosine_warmup_lr(step, base_lr, total_steps, warmup_steps=100, min_ratio=0.2):
    if step <= warmup_steps:
        return base_lr * (0.1 + 0.9 * step / max(1, warmup_steps))
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, t))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def fit_brightness(raw, features, top, ref, edge, valid, device):
    seed_all(2024)
    X = torch.from_numpy(features.T).unsqueeze(0).to(device)
    backbone = ColumnBackbone(features.shape[1], 48).to(device)
    y_head = brightness_head(48).to(device)
    params = [*backbone.parameters(), *y_head.parameters()]
    opt = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": LR_BB},
            {"params": y_head.parameters(), "lr": LR_Y},
        ],
        weight_decay=1e-4,
    )

    def tensor(a):
        return torch.from_numpy(a).to(device)

    top_gpu = tensor(top)
    ref_w, edge_w = (tensor(a.astype(np.float32)) for a in (ref, edge))
    target_w = 1.0 - ref_w
    ref_mean = float(top[ref].mean()) * S1_REF_BOOST
    pools = [torch.from_numpy(np.flatnonzero(valid & mask[None, :])) for mask in (ref, edge)]
    # Upload only the 2 x 2,000 RGB samples used in each iteration.
    rgb = torch.from_numpy(raw.reshape(-1, 3))
    try:
        sample = TorchPermutationPrefix().prefix
        if max(len(pool) for pool in pools) >= 2**32 // 20:
            raise ValueError("Large pixel pool")
    except (RuntimeError, ValueError):
        sample = lambda n, k: torch.randperm(n)[:k]

    def fields():
        g, b = y_head(backbone(X))[0]
        return (g.tanh() + 1) / 2 * 2.7 + 0.8, b.tanh() * 0.2 + 0.05

    def luma(rgb_values):
        return 0.299 * rgb_values[:, 0] + 0.587 * rgb_values[:, 1] + 0.114 * rgb_values[:, 2]

    def curvature(field):
        return (field[:-2] - 2 * field[1:-1] + field[2:]).square().mean()

    for step in range(1, 500 + 1):
        for group, lr in zip(opt.param_groups, (LR_BB, LR_Y)):
            group["lr"] = cosine_warmup_lr(step, lr, total_steps=500)
        gain, bias = fields()
        ref_ids, edge_ids = (pool[sample(len(pool), 2000)] for pool in pools)
        src, dst = (tensor(to_float01(rgb[ids].numpy())) for ids in (ref_ids, edge_ids))
        at = (edge_ids % raw.shape[1]).to(device)
        Y = luma(dst)
        scale = (Y * gain[at] + bias[at]).clamp(0, 1) / (Y + 1e-6)
        protected = torch_smoothstep(HL1_Y_LOW, HL1_Y_HIGH, Y)
        scale = 1.0 + (scale - 1.0) * (1.0 - HL1_STRENGTH * protected)
        corrected = (dst * scale[:, None]).clamp(0, 1)
        mmd = mmd_mk_1d_unbiased(luma(src), luma(corrected), MMD_KERNEL_BW_Y)
        pred = top_gpu * gain + bias
        diff = (pred - ref_mean).abs()
        loss = (
            W_Y_ALIGN * ((target_w * diff).mean() + 0.7 * (edge_w * diff).mean())
            + W_Y_ID * (ref_w * ((gain - 1.0).abs() + 0.5 * bias.abs())).mean()
            + W_Y_SMOOTH * (curvature(gain) + curvature(bias))
            + W_Y_FLAT * pred.var(unbiased=False)
            + W_Y_TV * pred.diff().abs().mean()
            + W_MMD_Y * mmd
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite Y loss at step {step}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        return [a.cpu().numpy() for a in fields()]
