import numpy as np
import torch
from utils.color import torch_rgb_to_hsv, torch_hsv_to_rgb, torch_rgb_to_lab, gamut_map_lab

LIGHTNESS_NODES = (50.0, 65.0, 80.0, 95.0)
HEIGHT_BANDS = 16


def smooth_ramp(value, lower, upper):
    t = ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
    return t.square() * (3.0 - 2.0 * t)


def reference_table(device, original, reference_columns, quantile=0.25):
    """Estimate a low chroma percentile at matching height and lightness."""
    height = original.shape[0]
    columns = np.flatnonzero(reference_columns)
    columns = columns[np.linspace(0, len(columns) - 1, min(len(columns), 32), dtype=int)]
    rows = np.linspace(0, height - 1, HEIGHT_BANDS * 32, dtype=int).reshape(HEIGHT_BANDS, -1)
    samples = original[rows[:, :, None], columns[None, None, :]]
    rgb = torch.as_tensor(samples.astype(np.float32) / 255.0, device=device)
    labs = torch_rgb_to_lab(rgb.reshape(-1, 3)).reshape(HEIGHT_BANDS, -1, 3).cpu().numpy()
    table = np.zeros((HEIGHT_BANDS, len(LIGHTNESS_NODES), 3), dtype=np.float32)

    for i in range(HEIGHT_BANDS):
        first = max(0, i - 1)
        last = min(HEIGHT_BANDS, i + 2)
        nearby = labs[first:last].reshape(-1, 3)
        for j, lightness in enumerate(LIGHTNESS_NODES):
            matched = nearby[np.abs(nearby[:, 0] - lightness) <= 11.0, 1:]
            if len(matched) < 24:
                continue
            color = np.median(matched, axis=0)
            chroma = np.linalg.norm(color)
            neutral = np.clip((28.0 - chroma) / 14.0, 0.0, 1.0)
            support = np.clip((len(matched) - 24) / 72.0, 0.0, 1.0)
            table[i, j, :2] = np.quantile(matched, quantile, axis=0)
            table[i, j, 2] = neutral * support
    return torch.as_tensor(table, device=device)


def reference_at(table, normalized_height, lightness):
    """Interpolate the sampled percentile, weighting by reference support."""
    height_pos = (normalized_height * HEIGHT_BANDS - 0.5).clamp(0.0, HEIGHT_BANDS - 1)
    light_pos = ((lightness - LIGHTNESS_NODES[0]) / 15.0).clamp(0.0, len(LIGHTNESS_NODES) - 1)
    y0, l0 = height_pos.floor().long(), light_pos.floor().long()
    y1 = (y0 + 1).clamp_max(HEIGHT_BANDS - 1)
    l1 = (l0 + 1).clamp_max(len(LIGHTNESS_NODES) - 1)
    dy, dl = height_pos - y0, light_pos - l0
    weights = torch.stack(((1 - dy) * (1 - dl), dy * (1 - dl), (1 - dy) * dl, dy * dl), dim=1)
    nodes = torch.stack((table[y0, l0], table[y1, l0], table[y0, l1], table[y1, l1]), dim=1)
    weights = weights * nodes[..., 2]
    support = weights.sum(dim=1)
    low = (weights[:, :, None] * nodes[..., :2]).sum(dim=1) / support[:, None].clamp_min(1e-7)
    return low, support


def proposed_ab(base, hsv, field):
    hsv_corrected = torch.stack(
        ((hsv[:, 0] + field[:, 1]) % 1.0, (hsv[:, 1] * field[:, 0]).clamp(0.0, 1.0), hsv[:, 2]),
        dim=1,
    )
    hsv_lab = torch_rgb_to_lab(torch_hsv_to_rgb(hsv_corrected))
    proposed = hsv_lab[:, 1:] + field[:, 2:] * torch.tanh(base[:, :1] / 5.0)
    return hsv_lab, proposed


def adapt_chroma(base, proposed, low_reference, support, allowance):
    bright = smooth_ramp(base[:, 0], 56.0, 79.0) * support
    lower = torch.minimum(base[:, 1:], low_reference - allowance)
    beyond = (lower - proposed).clamp_min(0.0)
    # Zero value and slope at the threshold avoid a new chroma step.
    return proposed + bright[:, None] * beyond.square() / (beyond + 0.75)


def distinctive_red(raw_lab):
    """Weight colors whose red signal is large relative to the yellow signal."""
    a, b = raw_lab[:, 1], raw_lab[:, 2]
    return smooth_ramp(a - b, 6.5, 10.0) * smooth_ramp(a, 13.0, 22.0)


def color_displacement(base, proposed, reference, red, allowance, red_strength):
    chroma = proposed
    if reference is not None:
        low, support = reference
        chroma = adapt_chroma(base, chroma, low, support, allowance)
    if red_strength:
        chroma = chroma + red_strength * red[:, None] * (base[:, 1:] - chroma).clamp_min(0.0)
    return chroma


def render(pixels, base, chroma, field):
    rgb, compression = gamut_map_lab(torch.cat((base[:, :1], chroma), dim=1))
    identity = (field[:, 0] == 1.0) & (field[:, 1:].abs().sum(dim=1) == 0.0)
    rgb = rgb + torch.where(identity[:, None], pixels - rgb, torch.zeros_like(rgb)).detach()
    return rgb, compression


class ColorReference:
    def __init__(self, raw, ref, device, quantile=0.25):
        self.height = len(raw)
        self.table = reference_table(device, raw, ref, quantile)

    @torch.no_grad()
    def prepare(self, pixels, raw_pixels, rows):
        base = torch_rgb_to_lab(pixels)
        hsv = torch_rgb_to_hsv(pixels)
        positions = torch.as_tensor(rows, device=pixels.device, dtype=pixels.dtype) / max(
            1, self.height - 1
        )
        reference = reference_at(self.table, positions, base[:, 0])
        original = torch.as_tensor(raw_pixels.astype(np.float32) / 255.0, device=pixels.device)
        red = distinctive_red(torch_rgb_to_lab(original))
        return base, hsv, reference, red

    def apply(self, pixels, context, field):
        base, hsv, reference, red = context
        _, proposed = proposed_ab(base, hsv, field)
        chroma = color_displacement(base, proposed, reference, red, 1.0, 0.75)
        return render(pixels, base, chroma, field)
