import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.special import logit
from utils.color import lab, torch_rgb_to_lab, gamut_map_lab

X_SAMPLES, HEIGHTS, ROWS = 161, 14, 32


def smooth_ramp(v):
    t = np.clip(v, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def robust_profiles(original, y_level):
    original = original.reshape(HEIGHTS, ROWS, -1)
    y_level = y_level.reshape(original.shape)
    change = logit(np.clip(y_level / 100, 0.035, 0.965)) - logit(
        np.clip(original / 100, 0.035, 0.965)
    )
    valid = (original > 11) & (original < 95) & (y_level < 97)
    median = np.median(change, axis=1)
    typical = np.median(original, axis=1)
    for iy, ix in np.argwhere(valid.sum(axis=1) >= 10):
        keep = valid[iy, :, ix]
        median[iy, ix] = np.median(change[iy, :, ix][keep])
        typical[iy, ix] = np.median(original[iy, :, ix][keep])
    raw_shape = logit(np.clip(typical / 100, 0.025, 0.975))
    return (gaussian_filter1d(p, 0.86, axis=1, mode="nearest") for p in (median, raw_shape))


def fit_attenuation(distance, deficit, weights):
    """Fit a smooth, nonnegative falloff with zero derivative at the light axis."""
    design = np.stack((distance**2 / (distance + 0.18), distance**3), axis=-1)
    robust = weights.copy()
    coefficients = np.zeros(2)
    for _ in range(5):
        weighted = robust[:, None] * design
        normal = design.T @ weighted + np.diag([0.035, 0.30])
        rhs = weighted.T @ deficit
        coefficients = np.linalg.solve(normal, rhs)
        if (coefficients < 0.0).any():
            candidates = np.diag(np.maximum(rhs / np.diag(normal), 0.0))
            loss = np.sum(robust[:, None] * (deficit[:, None] - design @ candidates) ** 2, axis=0)
            loss += np.sum(np.diag([0.035, 0.30]) @ candidates**2, axis=0)
            coefficients = candidates[:, int(loss.argmin())]
        residual = deficit - design @ coefficients
        robust = weights * np.minimum(1.0, 0.18 / np.maximum(np.abs(residual), 0.001))
    return design @ coefficients


def source_exposure(x, raw, low, high):
    center = 0.5 * (low + high)
    anchor = np.abs(x - center) <= 0.26 * (high - low)
    if not anchor.any():
        anchor[np.argmin(np.abs(x - center))] = True
    deficit = np.median(raw[:, anchor], axis=1)[:, None] - raw
    # Pool evidence across depth before allowing a smaller depth-specific
    # response. A single dark lamina does not set the exposure of a column.
    pooled = np.median(deficit, axis=0)
    physical = np.zeros_like(deficit)
    for side, span in ((x <= center, center), (x > center, 1.0 - center)):
        if span <= 0 or not side.any():
            continue
        xs = x[side]
        distance = np.abs(xs - center) / span
        visible = smooth_ramp((np.minimum(xs, 1.0 - xs) - 0.05) / 0.075)
        weight = visible * smooth_ramp((distance - 0.08) / 0.19)
        consensus = fit_attenuation(distance, pooled[side], weight)
        individual = np.stack([fit_attenuation(distance, row[side], weight) for row in deficit])
        physical[:, side] = 0.72 * consensus + 0.28 * individual
    return physical


def continue_side_exposure(x, raw, previous, low, high):
    physical = source_exposure(x, raw, low, high)
    center = 0.5 * (low + high)
    delta = np.zeros_like(previous)
    for sign, span in ((-1.0, center), (1.0, 1.0 - center)):
        distance = sign * (x - center)
        shoulder = min(0.5 * (high - low) + 0.109, 0.85 * span)
        at_shoulder = np.abs(distance - shoulder) < 0.016
        active = (distance > 0.0) & (distance < shoulder)
        if not active.any():
            continue
        strength = np.median(previous[:, at_shoulder], axis=1).clip(0.0)
        progress = distance[active] / shoulder
        # Continue the observed response toward the light axis, with zero
        # derivative there and no change to the already corrected shoulder.
        tail = strength[:, None] * progress**2
        need = (tail - previous[:, active]).clip(0.0)
        supported = (physical[:, active] - previous[:, active]).clip(0.0)
        ramp = smooth_ramp((shoulder - distance[active]) / 0.043)
        delta[:, active] = 0.43 * np.tanh(np.minimum(need, supported) / 0.43) * ramp
    return gaussian_filter1d(delta, 1.3, axis=1, mode="nearest")


class LightnessField:
    def __init__(self, original, y_level, ref):
        self.height, self.width = original.shape[:2]
        self.x = np.linspace(0, 1, X_SAMPLES)
        self.y = (np.arange(HEIGHTS) + 0.5) / HEIGHTS
        columns = np.rint(self.x * (self.width - 1)).astype(int)
        rows = np.linspace(0, self.height - 1, HEIGHTS * ROWS, dtype=int)
        coords = (rows[:, None], columns[None, :])
        ground, brightness = (lab(image[coords])[..., 0] for image in (original, y_level))
        previous, raw = robust_profiles(ground, brightness)
        low, high = np.flatnonzero(ref)[[0, -1]] / (self.width - 1)
        delta = continue_side_exposure(self.x, raw, previous, low, high)
        delta = gaussian_filter1d(delta, 0.5, axis=0, mode="nearest")
        self.profile = PchipInterpolator(self.x, delta, axis=1)

    def offset(self, ys, xs):
        xp = xs / (self.width - 1)
        yp = (ys / (self.height - 1)).clip(self.y[0], self.y[-1])
        return PchipInterpolator(self.y, self.profile(xp), axis=0)(yp).astype(np.float32)


@torch.inference_mode()
def apply_final_lightness(device, y_level, color, field, bounds):
    x, y, w, h = bounds
    out = np.empty((h, w, 3), dtype=np.uint8)
    xs = np.arange(x, x + w)
    for row in range(0, h, 112):
        end = min(h, row + 112)
        ys = np.arange(y + row, y + end)
        native = y_level[ys[0] : ys[-1] + 1, x : x + w]
        reference = color[ys[0] : ys[-1] + 1, x : x + w]
        previous = torch.as_tensor(native.reshape(-1, 3).astype(np.float32) / 255, device=device)
        kept = torch.as_tensor(reference.reshape(-1, 3).astype(np.float32) / 255, device=device)
        l = torch_rgb_to_lab(previous)[:, 0]
        ab = torch_rgb_to_lab(kept)[:, 1:]
        u = l / 100
        d = torch.as_tensor(field.offset(ys, xs).reshape(-1), device=device)
        # Both the black and the white endpoints of this monotone curve are
        # fixed and the adjustment vanishes naturally for very dark pixels.
        new = 100 * torch.sigmoid(torch.logit(u.clamp(1e-6, 1 - 1e-6)) + d)
        new = torch.where((u <= 0) | (u >= 1), l, new)
        rgb, _ = gamut_map_lab(torch.cat((new[:, None], ab), dim=1))
        rgb = torch.where(d[:, None] == 0, kept, rgb)
        out[row:end] = (
            rgb.mul(255).round().clamp(0, 255).byte().cpu().numpy().reshape(end - row, w, 3)
        )
    return out


class BrightnessProfile:
    """Fit a bounded correction to the sampled Y response, not the image texture."""

    def __init__(self, raw, stage, ref):
        self.h, self.w = raw.shape[:2]
        self.ref = ref
        self.ends = np.flatnonzero(ref)[[0, -1]]
        self.x = np.linspace(0, 1, 181)
        self.y = (np.arange(12) + 0.5) / 12
        xr = np.rint(self.x * (self.w - 1)).astype(int)
        yr = np.linspace(0, self.h - 1, 12 * 24, dtype=int)
        coords = (yr[:, None], xr[None, :])
        change = (lab(stage[coords])[..., 0] - lab(raw[coords])[..., 0]).reshape(
            12, 24, len(self.x)
        )
        center = gaussian_filter1d(np.median(change, axis=1), 0.75, axis=1, mode="nearest")
        if not np.isfinite(center).all():
            raise ValueError("Invalid luminance profile")
        distance = np.maximum(
            self.ends[0] / (self.w - 1) - self.x, self.x - self.ends[1] / (self.w - 1)
        )
        span = smooth_ramp(distance / 0.038) * (1 - smooth_ramp((distance - 0.115) / 0.095))
        step = max(1, round(0.02 * (len(self.x) - 1)))
        center_x = (self.x[:-step] + self.x[step:]) / 2
        dist_x = (distance[:-step] + distance[step:]) / 2
        correction = np.zeros_like(center)
        for columns, side, limit in (
            (self.x < self.ends[0] / (self.w - 1), center_x < self.ends[0] / (self.w - 1), 0.225),
            (self.x > self.ends[1] / (self.w - 1), center_x > self.ends[1] / (self.w - 1), 0.225),
        ):
            available = side & (dist_x >= -0.008) & (dist_x <= limit)
            if not available.any() or not columns.any():
                continue
            best = float("inf")
            for width in (0.0, 2.5, 4.0, 5.5):
                target = (
                    gaussian_filter1d(center, width, axis=1, mode="nearest") if width else center
                )
                offset = ((target - center) * span).clip(-2.4, 2.4)
                proposal = center + offset
                rate = np.abs(proposal[:, step:] - proposal[:, :-step])[:, available]
                objective = np.quantile(rate.max(axis=1), 0.75) + 0.12 * np.sqrt(
                    np.mean(offset[:, columns] ** 2)
                )
                if objective < best:
                    best = objective
                    correction[:, columns] = offset[:, columns]
        self.profile = PchipInterpolator(self.x, correction, axis=1)

    def offset(self, ys, xs):
        vals = self.profile(xs / (self.w - 1))
        out = PchipInterpolator(self.y, vals, axis=0)(
            (ys / (self.h - 1)).clip(self.y[0], self.y[-1])
        )
        distance = np.maximum(self.ends[0] - xs, xs - self.ends[1]) / (self.w - 1)
        out *= smooth_ramp(distance / 0.008)
        out[:, self.ref[xs]] = 0
        return out.astype(np.float32)


@torch.inference_mode()
def y_block(device, old, lightness_offset):
    y = torch.as_tensor(old.reshape(-1, 3).astype(np.float32) / 255, device=device)
    stage = torch_rgb_to_lab(y)
    adjustment = torch.as_tensor(lightness_offset.reshape(-1), device=device)
    dark = ((stage[:, 0] - 8.0) / 18.0).clamp(0.0, 1.0)
    stage[:, 0] = (stage[:, 0] + adjustment * dark.square() * (3.0 - 2 * dark)).clamp(0, 100)
    corrected, _ = gamut_map_lab(stage)
    corrected = torch.where(adjustment[:, None] == 0, y, corrected)
    return corrected.mul(255).round().clamp(0, 255).byte().cpu().numpy().reshape(old.shape)


def stage_at_current_brightness(device, raw, old_y, ref):
    profile = BrightnessProfile(raw, old_y, ref)
    step = np.empty_like(raw)
    xs = np.arange(raw.shape[1])
    for y in range(0, len(raw), 112):
        stop = min(len(raw), y + 112)
        step[y:stop] = y_block(device, old_y[y:stop], profile.offset(np.arange(y, stop), xs))
    return step
