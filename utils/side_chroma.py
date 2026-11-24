import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
from utils.color import gamut_map_lab, torch_rgb_to_lab, lab
from utils.color_response import (
    ColorReference,
    reference_table,
    reference_at,
    proposed_ab,
    color_displacement,
    render,
)

BANDS, COLUMNS, ROWS_PER_BAND, POINTS_PER_COLUMN = 14, 28, 24, 16
LEVELS = np.asarray([20.0, 35.0, 50.0, 65.0, 80.0, 95.0], dtype=np.float32)


class SideReference(ColorReference):
    def __init__(self, raw, ref, device):
        super().__init__(raw, ref, device, quantile=0.50)
        ix = np.flatnonzero(ref)
        self.ref_limits = (int(ix[0]), int(ix[-1]))
        self.width = raw.shape[1]
        parts = np.array_split(ix, 2) if len(ix) > 1 else (ix, ix)
        self.near = [
            reference_table(device, raw, np.isin(np.arange(self.width), columns), quantile=0.50)
            for columns in parts
        ]

    def prepare_at(self, values, raw, ys, xs):
        base, hsv, reference, red = super().prepare(values, raw, ys)
        lightness = base[:, 0]
        y = torch.as_tensor(ys, device=values.device, dtype=values.dtype) / max(1, self.height - 1)
        x = torch.as_tensor(xs, device=values.device, dtype=values.dtype)
        left, right = self.ref_limits
        is_left = x < left
        side_a, sa = reference_at(self.near[0], y, lightness)
        side_b, sb = reference_at(self.near[1], y, lightness)
        side_mid = torch.where(is_left[:, None], side_a, side_b)
        side_support = torch.where(is_left, sa, sb)
        distance = torch.maximum(left - x, x - right).clamp_min(0)
        proximity = (1 - distance / (0.31 * (self.width - 1))).clamp(0, 1)
        neighbor = 0.8 * proximity.square() * (3.0 - 2.0 * proximity)
        common, support = reference
        w0 = (1 - neighbor) * support
        w1 = neighbor * side_support
        total = w0 + w1
        avg = (w0[:, None] * common + w1[:, None] * side_mid) / total[:, None].clamp_min(1.0e-7)
        return base, hsv, (avg, total), red

    def apply(self, pixels, context, field):
        base, hsv, anchor, red = context
        low, support = anchor
        _, raw_proposal = proposed_ab(base, hsv, field)
        tolerance = [1.05, 0.55]
        lo = torch.minimum(base[:, 1:], low - base.new_tensor(tolerance))
        left = (lo - raw_proposal).clamp_min(0.0)
        t = ((base[:, 0] - 44.0) / 22.0).clamp(0.0, 1.0)
        strength = (t.square() * (3.0 - 2 * t) * support)[:, None]
        chroma = raw_proposal + strength * left.square() / (left + 0.55)
        chroma = color_displacement(base, chroma, None, red, 1.0, 0.75)
        return render(pixels, base, chroma, field)


def ramp(values, lo, hi):
    t = ((values - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth_np(value, lo, hi):
    t = np.clip((value - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def sampled_tiles(rgb, ys, xs):
    values = lab(rgb[ys[:, None], xs[None, :]])
    return (
        values.reshape(BANDS, ROWS_PER_BAND, COLUMNS, POINTS_PER_COLUMN, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(BANDS, COLUMNS, -1, 3)
    )


class Grid:
    def __init__(self, values, height, width):
        a, b = values.shape[:2]
        self.height, self.width = height, width
        self.ys = (np.arange(a) + 0.5) / a
        self.xs = (np.arange(b) + 0.5) / b
        self.profile = PchipInterpolator(self.xs, values, axis=1)

    def patch(self, ys, xs, device):
        xp = (xs / (self.width - 1)).clip(self.xs[0], self.xs[-1])
        yp = (ys / (self.height - 1)).clip(self.ys[0], self.ys[-1])
        values = PchipInterpolator(self.ys, self.profile(xp), axis=0)(yp)
        return torch.as_tensor(
            values.reshape(-1, values.shape[-1]).astype(np.float32), device=device
        )


def side_table(raw, ref, height_rows):
    cols = np.flatnonzero(ref)
    cols = cols[np.linspace(0, len(cols) - 1, min(84, len(cols)), dtype=int)]
    data = lab(raw[height_rows[:, None], cols[None, :]]).reshape(BANDS, -1, 3)
    table = np.zeros((BANDS, len(LEVELS), 3), dtype=np.float32)
    for i, row in enumerate(table):
        region = data[max(0, i - 1) : i + 2].reshape(-1, 3)
        for j, level in enumerate(LEVELS):
            pixels = region[np.abs(region[:, 0] - level) <= 13.0]
            if len(pixels) >= 28:
                row[j, :2] = np.median(pixels[:, 1:], axis=0)
                row[j, 2] = min(1.0, len(pixels) / 160.0)
            else:
                close = np.argsort(np.abs(region[:, 0] - level))[:96]
                row[j, :2] = np.median(region[close, 1:], axis=0)
                row[j, 2] = 0.20
    return table, data


class SideChroma:
    def __init__(self, device, raw, stage, ref, fields):
        self.device, self.raw, self.stage = device, raw, stage
        self.h, self.w = raw.shape[:2]
        self.field = torch.as_tensor(fields, device=device)
        self.base = SideReference(raw, ref, device)
        rows = np.linspace(0, self.h - 1, BANDS * ROWS_PER_BAND, dtype=int)
        cols = np.linspace(0, self.w - 1, COLUMNS * POINTS_PER_COLUMN, dtype=int)
        original, brightened = (sampled_tiles(im, rows, cols) for im in (raw, stage))
        anchor, samples = side_table(raw, ref, rows)
        self.anchor = torch.as_tensor(anchor, device=device)

        red = original[..., 1] - original[..., 2]
        red_upper = gaussian_filter(np.quantile(red, 0.84, axis=2), (0.82, 0.81), mode="nearest")
        local = gaussian_filter(
            np.median(original[..., 1:], axis=2), (0.82, 0.81, 0.0), mode="nearest"
        )
        offset = local - np.median(samples[..., 1:], axis=1)[:, None, :]
        orange = smooth_np(offset[..., 0], 2.0, 13.0) * smooth_np(offset[..., 1], 2.0, 12.0)
        widespread = 1.0 - (1.0 - smooth_np(red_upper, 3.0, 8.5)) * (1.0 - orange)
        self.red_stats = Grid(np.stack([red_upper, widespread], axis=-1), self.h, self.w)

        residual = np.empty((BANDS, COLUMNS, 2), dtype=np.float32)
        for y in range(BANDS):
            lightness = brightened[y, ..., 0]
            baseline = np.stack(
                [np.interp(lightness, LEVELS, anchor[y, :, axis]) for axis in (0, 1)], axis=-1
            )
            residual[y] = np.median(brightened[y, ..., 1:] - baseline, axis=1)
        smooth = gaussian_filter(residual, (0.9, 1.22, 0.0), mode="nearest")
        self.cast = Grid(smooth, self.h, self.w)
        self.ref_center = np.flatnonzero(ref).mean() / (self.w - 1)
        self.side_strength = self.reflection_by_side(stage)

    def reflection_by_side(self, stage):
        """Estimate each side's joint red-yellow cast before making corrections."""
        ys = np.linspace(0, self.h - 1, 100, dtype=int)
        inner = round(0.20 * self.w)
        strengths = []
        for start, stop in ((0, inner), (self.w - inner, self.w)):
            xs = np.linspace(start, stop - 1, 28, dtype=int)
            values = lab(stage[ys[:, None], xs[None, :]]).reshape(-1, 3)
            anchor = self.local_reference(
                torch.as_tensor(values[:, 0], device=self.device), np.repeat(ys, len(xs))
            )
            excess = values[:, 1:] - anchor[:, :2].cpu().numpy()
            orange = np.minimum(excess[:, 0], 1.45 * excess[:, 1]).clip(0.0)
            strengths.append(float(np.quantile(orange, 0.75)))
        return np.asarray(strengths, dtype=np.float32)

    def local_reference(self, brightness, rows):
        y = (
            torch.as_tensor(rows, device=brightness.device, dtype=brightness.dtype)
            / (self.h - 1)
            * BANDS
            - 0.5
        ).clamp(0, BANDS - 1)
        l = ((brightness - float(LEVELS[0])) / 15.0).clamp(0, len(LEVELS) - 1)
        i, j = y.floor().long(), l.floor().long()
        cy, cl = y - i, l - j
        k, n = (i + 1).clamp_max(BANDS - 1), (j + 1).clamp_max(len(LEVELS) - 1)
        at_y = (1 - cl[:, None]) * self.anchor[i, j] + cl[:, None] * self.anchor[i, n]
        at_k = (1 - cl[:, None]) * self.anchor[k, j] + cl[:, None] * self.anchor[k, n]
        return (1 - cy[:, None]) * at_y + cy[:, None] * at_k

    def red_confidence(self, raw, old, local):
        color = torch_rgb_to_lab(raw)
        excess = color[:, 1] - color[:, 2] - local[:, 0]
        distinct = ramp(excess, 1.1, 6.5)
        return old * (1.0 - 0.94 * local[:, 1] * (1.0 - distinct))

    def directional(self, base, local, fitted, red, xs):
        color = base[:, 1:]
        normal = fitted.clamp_min(0.0) + base.new_tensor([0.9, 0.6])
        direction = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        center = (
            torch.as_tensor(xs, device=base.device, dtype=base.dtype) / (self.w - 1)
            - self.ref_center
        )
        natural = 1.3 + 1.3 * torch.exp(-torch.square(center / 0.13))
        chroma = local[:, :2]
        typical = (fitted * direction).sum(1) - natural
        typical = typical.clamp_min(0.0) * ramp(fitted[:, 0], 1.3, 5.5)
        per_pixel = ((color - chroma) * direction).sum(1)
        amount = typical + 0.72 * ramp(typical, 2.5, 12.0) * (per_pixel - typical).clamp_min(0.0)
        amount = torch.minimum(amount, (per_pixel + 0.55).clamp_min(0.0))
        allowed = (color - chroma + base.new_tensor([1.2, 1.25])).clamp_min(0.0)
        stable = ramp(base[:, 0], 14.0, 35.0) * ramp(local[:, 2], 0.18, 0.68) * (1.0 - 0.91 * red)
        return color - stable[:, None] * torch.minimum(amount[:, None] * direction, allowed)

    def guided_response(self, base, prior, directional, local, fitted, original_red, red):
        expected = local[:, :2]
        signal = ramp(local[:, 2], 0.26, 0.68) * ramp(base[:, 0], 29.0, 56.0)
        floor = torch.minimum(base[:, 1:], expected - base.new_tensor([1.25, 0.50]))
        cyan_need = ramp(floor - prior, 0.26, 2.75)
        maximum = torch.minimum(directional, expected + base.new_tensor([0.65, 0.95]))
        lift = (maximum - prior).clamp_min(0.0) * cyan_need * signal[:, None]
        warm_a = ramp(base[:, 1] - expected[:, 0], 11.0, 25.0) * ramp(fitted[:, 0], 7.0, 22.0)
        false_red = (original_red - red).clamp(0.0, 1.0)
        evidence = torch.maximum(warm_a, false_red) * ramp(prior[:, 0] - expected[:, 0], 0.70, 5.10)
        weight = signal * evidence * (1.0 - 0.55 * red)
        reduction = (directional - prior).clamp_max(0.0) * weight[:, None]
        return prior + lift + reduction

    @torch.inference_mode()
    def render(self, box):
        x, y, w, h = box
        out = np.empty((h, w, 3), dtype=np.uint8)
        xs = np.arange(x, x + w)
        columns = self.field[x : x + w]
        for first in range(0, h, 96):
            last = min(h, first + 96)
            ys = np.arange(y + first, y + last)
            source = self.raw[ys[0] : ys[-1] + 1, x : x + w]
            stage = self.stage[ys[0] : ys[-1] + 1, x : x + w]
            device = self.device
            pixels = torch.as_tensor(stage.reshape(-1, 3).astype(np.float32) / 255, device=device)
            original = torch.as_tensor(
                source.reshape(-1, 3).astype(np.float32) / 255, device=device
            )
            locations, coordinates = np.repeat(ys, w), np.tile(xs, len(ys))
            context = self.base.prepare_at(pixels, source.reshape(-1, 3), locations, coordinates)
            luminance, hsv, reference, original_red = context
            red = self.red_confidence(original, original_red, self.red_stats.patch(ys, xs, device))
            local = self.local_reference(luminance[:, 0], locations)
            fit = self.cast.patch(ys, xs, device)
            directional = self.directional(luminance, local, fit, red, coordinates)
            intermediate, _ = self.base.apply(
                pixels, (luminance, hsv, reference, original_red), columns.repeat(len(ys), 1)
            )
            prior = torch_rgb_to_lab(intermediate)[:, 1:]
            guided = self.guided_response(
                luminance, prior, directional, local, fit, original_red, red
            )
            xfrac = pixels.new_tensor(coordinates) / (self.w - 1)
            position = ramp(xfrac, self.ref_center - 0.15, self.ref_center + 0.15)
            side = pixels.new_tensor(self.side_strength)
            weight = 0.94 * (
                (1.0 - position) * ramp(side[0], 17.0, 29.0) + position * ramp(side[1], 17.0, 29.0)
            )
            chroma = guided + weight[:, None] * (directional - guided)
            rgb, _ = gamut_map_lab(torch.cat((luminance[:, :1], chroma), dim=1))
            out[first:last] = (
                rgb.mul(255).round().clamp(0, 255).byte().cpu().numpy().reshape(last - first, w, 3)
            )
        return out
