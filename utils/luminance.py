import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import logit, expit
from skimage.color import rgb2lab
from utils.color import gamut_map_lab
from utils.feature_extraction import reference_gate

HEIGHT_BANDS, COLUMNS, SAMPLE_ROWS, SAMPLE_COLS = 12, 30, 36, 16
X_SMOOTH = 1.10
LEVELS = (0.20, 0.50, 0.80)
LEVEL_WEIGHTS = np.asarray([0.75, 1.5, 1.0])[:, None, None]


def map_lightness(lightness, params):
    """Monotone tone map, with immovable black and white endpoints."""
    unit = np.clip(lightness / 100.0, 0.0, 1.0)
    scale, offset = np.moveaxis(params[..., :2], -1, 0)
    output = 100.0 * expit(np.exp(scale) * logit(unit.clip(1e-6, 1.0 - 1e-6)) + offset)
    return np.where((unit == 0) | (unit == 1), lightness, output)


def fit_grid(observations, reliability):
    """Regularized smooth fit to the sampled cells, with a shared column prior."""
    weighted = observations * reliability[..., None]
    pooled = gaussian_filter(weighted.sum(axis=0), (X_SMOOTH, 0), mode="nearest")
    weights = gaussian_filter(reliability.sum(axis=0), X_SMOOTH, mode="nearest")
    profile = pooled / (weights[..., None] + 0.12)
    return profile[None, ...]


def sample_lab(rgb, ref):
    height, width = rgb.shape[:2]
    rows = np.linspace(0, height - 1, HEIGHT_BANDS * SAMPLE_ROWS, dtype=int)
    cols = np.linspace(0, width - 1, COLUMNS * SAMPLE_COLS, dtype=int)
    ref_cols = np.flatnonzero(ref)
    source = ref_cols[np.linspace(0, len(ref_cols) - 1, min(len(ref_cols), 64), dtype=int)]
    lab = rgb2lab(rgb[rows[:, None], cols[None, :]].astype(np.float32) / 255.0)
    lab = lab.reshape(HEIGHT_BANDS, SAMPLE_ROWS, COLUMNS, SAMPLE_COLS, 3).transpose(0, 2, 1, 3, 4)
    lab = lab.reshape(HEIGHT_BANDS, COLUMNS, -1, 3)
    reference = rgb2lab(rgb[rows[:, None], source[None, :]].astype(np.float32) / 255.0)
    reference = reference.reshape(HEIGHT_BANDS, -1, 3)
    is_reference = ref[cols].reshape(COLUMNS, SAMPLE_COLS).mean(axis=1)
    return lab, reference, is_reference


def fit_tone(lab, reference, is_reference):
    sample_q = np.quantile(lab[..., 0], (0.20, 0.50, 0.80), axis=2)
    ref_q = np.quantile(reference[..., 0], (0.20, 0.50, 0.80), axis=1)[:, :, None]
    target_z = logit((sample_q / 100.0).clip(0.003, 0.997))
    ref_z = logit((ref_q / 100.0).clip(0.003, 0.997))
    ratio = ((ref_z[2] - ref_z[0]) / (target_z[2] - target_z[0]).clip(0.2)).clip(0.78, 1.28)
    log_scale = 0.65 * np.log(ratio)
    offset = (ref_z[1] - np.exp(log_scale) * target_z[1]).clip(-0.85, 2.6)
    edge_spread = sample_q[2] - sample_q[0]
    ref_spread = ref_q[2] - ref_q[0]
    reliability = ((edge_spread - 1.0) / (0.45 * ref_spread + 2.0)).clip(0.08, 1.0)
    # Match the bulk of the rock distribution while reducing the contribution
    # of an almost signal-free cell. Every pixel still uses the same mapping.
    reliability *= (sample_q[1] / 9.0).clip(0.16, 1.0)
    reliability *= 1.0 - is_reference[None, :]
    expected = np.stack((log_scale, offset), axis=-1)
    field = fit_grid(expected, reliability)
    return field


class Mapping:
    def __init__(self, rgb, ref, gate):
        self.rgb = rgb
        self.ref = ref
        self.gate = gate
        lab, source, reference_fraction = sample_lab(rgb, ref)
        self.field = fit_tone(lab, source, reference_fraction).astype(np.float32)
        self.x_spline = PchipInterpolator(
            np.linspace(0.5 / COLUMNS, 1 - 0.5 / COLUMNS, COLUMNS), self.field, axis=1
        )

    def parameters(self, ys, xs):
        h, w = self.rgb.shape[:2]
        x = (xs / max(w - 1, 1)).clip(0.5 / COLUMNS, 1 - 0.5 / COLUMNS)
        values = self.x_spline(x)
        return values.astype(np.float32) * self.gate[xs][None, :, None]

    @torch.inference_mode()
    def render(self, box, device, frozen_rgb, blend):
        x, y, width, height = box
        out = np.empty((height, width, 3), dtype=np.uint8)
        xs = np.arange(x, x + width)
        for row in range(0, height, 128):
            ys = np.arange(y + row, min(y + row + 128, y + height))
            source = self.rgb[ys[0] : ys[-1] + 1, x : x + width]
            values = self.parameters(ys, xs)
            lab = rgb2lab(source.astype(np.float32) / 255.0)
            lightness = map_lightness(lab[..., 0], values)
            old = frozen_rgb[ys[0] : ys[-1] + 1, x : x + width]
            base = rgb2lab(old.astype(np.float32) / 255.0)
            lightness = base[..., 0] + blend(ys, xs) * (lightness - base[..., 0])
            chroma = base[..., 1:]
            moved = np.concatenate((lightness[..., None], chroma), axis=-1)
            colors = torch.as_tensor(moved.reshape(-1, 3), device=device)
            rgb, _ = gamut_map_lab(colors)
            rendered = (
                rgb.mul(255.0).round().clamp(0, 255).byte().cpu().numpy().reshape(source.shape)
            )
            rendered[:, self.ref[xs]] = source[:, self.ref[xs]]
            out[row : row + len(ys)] = rendered
        return out


def lightness(rgb):
    return rgb2lab(rgb.astype(np.float32) / 255.0)[..., 0]


def sample_cells(values):
    return (
        values.reshape(HEIGHT_BANDS, SAMPLE_ROWS, COLUMNS, SAMPLE_COLS)
        .transpose(0, 2, 1, 3)
        .reshape(HEIGHT_BANDS, COLUMNS, -1)
    )


def reference_interval(raw, rows, ref):
    columns = np.flatnonzero(ref)
    if len(columns) > 96:
        columns = columns[np.linspace(0, len(columns) - 1, 96, dtype=int)]
    values = lightness(raw[rows[:, None], columns[None, :]])
    center = np.quantile(values.reshape(HEIGHT_BANDS, -1), LEVELS, axis=1)
    parts = np.array_split(values, min(4, len(columns)), axis=1)
    spread = np.stack([np.quantile(p.reshape(HEIGHT_BANDS, -1), LEVELS, axis=1) for p in parts])
    span = np.quantile(np.abs(spread - center), 0.75, axis=0)
    return center[:, :, None], (2.25 + 0.9 * span).clip(2.25, 13.0)[:, :, None]


class AdaptiveLuminance:
    def __init__(self, raw, stage1, ref, gate):
        self.mapping = Mapping(raw, ref, gate)
        self.height, self.width = raw.shape[:2]
        rows = np.linspace(0, self.height - 1, HEIGHT_BANDS * SAMPLE_ROWS, dtype=int)
        cols = np.linspace(0, self.width - 1, COLUMNS * SAMPLE_COLS, dtype=int)
        actual = lightness(raw[rows[:, None], cols[None, :]])
        prev = lightness(stage1[rows[:, None], cols[None, :]])
        new = map_lightness(actual, self.mapping.parameters(rows, cols))
        actual_q = np.quantile(sample_cells(actual), LEVELS, axis=2)
        prev_q = np.quantile(sample_cells(prev), LEVELS, axis=2)
        new_q = np.quantile(sample_cells(new), LEVELS, axis=2)

        ref_q, tolerance = reference_interval(raw, rows, ref)
        target_q = np.clip(prev_q, ref_q - tolerance, ref_q + tolerance)
        delta = (new_q - prev_q) / tolerance
        target = (target_q - prev_q) / tolerance
        weight = (
            (np.sum(LEVEL_WEIGHTS * delta * target, axis=0) + 0.35)
            / (np.sum(LEVEL_WEIGHTS * np.square(delta), axis=0) + 1.35)
        ).clip(0.0, 0.85)

        strength = (actual_q[1] / 12.0).clip(0.12, 1.0)
        contrast = ((actual_q[2] - actual_q[0]) / 5.5).clip(0.20, 1.0)
        anchor = gate[cols].reshape(COLUMNS, SAMPLE_COLS).mean(axis=1)
        support = strength * contrast * anchor
        smoothed = gaussian_filter(weight * support, (1.10, 1.02), mode="nearest")
        confidence = gaussian_filter(support, (1.10, 1.02), mode="nearest")
        self.grid = ((smoothed + 0.035) / (confidence + 0.18)).astype(np.float32)
        self.x_nodes = np.linspace(0.5 / COLUMNS, 1 - 0.5 / COLUMNS, COLUMNS)
        self.y_nodes = np.linspace(0.5 / HEIGHT_BANDS, 1 - 0.5 / HEIGHT_BANDS, HEIGHT_BANDS)
        self.profile = PchipInterpolator(self.x_nodes, self.grid, axis=1)

    def weight(self, ys, xs):
        x = (xs / max(1, self.width - 1)).clip(self.x_nodes[0], self.x_nodes[-1])
        y = (ys / max(1, self.height - 1)).clip(self.y_nodes[0], self.y_nodes[-1])
        return (
            PchipInterpolator(self.y_nodes, self.profile(x), axis=0)(y)
            .astype(np.float32)
            .clip(0.0, 0.85)
        )

    def render(self, box, device, color_source):
        return self.mapping.render(box, device, color_source, self.weight)


def gain_field(gain, bias, top, ref):
    matched = np.clip((gain * top + bias) / np.maximum(top, 0.03), 0.5, 4.0)
    log_gain = gaussian_filter1d(np.log(matched), max(1.0, 0.01 * len(gain)), mode="nearest")
    return np.exp(reference_gate(ref) * log_gain).astype(np.float32)


def apply_y(rgb, gain):
    output = np.empty_like(rgb)
    for y in range(0, len(rgb), 128):
        output[y : y + 128] = (
            np.clip(rgb[y : y + 128].astype(np.float32) * gain[None, :, None], 0, 255)
            .round()
            .astype(np.uint8)
        )
    return output
