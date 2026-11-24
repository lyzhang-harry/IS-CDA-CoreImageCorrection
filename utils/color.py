import torch
import numpy as np
from skimage.color import rgb2lab


def lab(rgb):
    return rgb2lab(rgb.astype(np.float32) / 255.0)


def torch_rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_val, _ = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val + 1e-7

    v = max_val

    s = diff / (max_val + 1e-7)
    s[max_val < 1e-7] = 0

    h = torch.zeros_like(v)
    mask_r = max_val == r
    mask_g = max_val == g
    mask_b = max_val == b

    h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r]
    h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / diff[mask_g]
    h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / diff[mask_b]

    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=1)


def torch_hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    c = v * s
    x = c * (1 - torch.abs((h * 6.0) % 2 - 1))
    m = v - c

    zeros = torch.zeros_like(h)

    h6 = h * 6.0
    r_prime = torch.zeros_like(h)
    g_prime = torch.zeros_like(h)
    b_prime = torch.zeros_like(h)

    m1 = h6 < 1
    r_prime[m1], g_prime[m1], b_prime[m1] = c[m1], x[m1], zeros[m1]
    m2 = (h6 >= 1) & (h6 < 2)
    r_prime[m2], g_prime[m2], b_prime[m2] = x[m2], c[m2], zeros[m2]
    m3 = (h6 >= 2) & (h6 < 3)
    r_prime[m3], g_prime[m3], b_prime[m3] = zeros[m3], c[m3], x[m3]
    m4 = (h6 >= 3) & (h6 < 4)
    r_prime[m4], g_prime[m4], b_prime[m4] = zeros[m4], x[m4], c[m4]
    m5 = (h6 >= 4) & (h6 < 5)
    r_prime[m5], g_prime[m5], b_prime[m5] = x[m5], zeros[m5], c[m5]
    m6 = h6 >= 5
    r_prime[m6], g_prime[m6], b_prime[m6] = c[m6], zeros[m6], x[m6]

    return torch.stack([r_prime + m, g_prime + m, b_prime + m], dim=1)


def torch_rgb_to_lab(rgb):
    mask_low = rgb <= 0.04045
    rgb_lin = torch.zeros_like(rgb)
    rgb_lin[mask_low] = rgb[mask_low] / 12.92
    rgb_lin[~mask_low] = torch.pow((rgb[~mask_low] + 0.055) / 1.055, 2.4)

    r, g, b = rgb_lin[:, 0], rgb_lin[:, 1], rgb_lin[:, 2]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    thresh = 0.008856
    mask_f = x > thresh
    fx = torch.zeros_like(x)
    fy = torch.zeros_like(y)
    fz = torch.zeros_like(z)

    fx[mask_f] = torch.pow(x[mask_f], 1.0 / 3.0)
    fx[~mask_f] = 7.787 * x[~mask_f] + 16.0 / 116.0

    mask_f = y > thresh
    fy[mask_f] = torch.pow(y[mask_f], 1.0 / 3.0)
    fy[~mask_f] = 7.787 * y[~mask_f] + 16.0 / 116.0

    mask_f = z > thresh
    fz[mask_f] = torch.pow(z[mask_f], 1.0 / 3.0)
    fz[~mask_f] = 7.787 * z[~mask_f] + 16.0 / 116.0

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return torch.stack([L, a, b], dim=1)


def torch_smoothstep(a, b, x):
    t = torch.clamp((x - a) / (b - a + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gamut_map_lab(lab):
    """Compress out-of-gamut chroma in linear RGB while preserving XYZ Y / L*."""
    fy = (lab[:, 0] + 16) / 116
    f = torch.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], 1)
    xyz = torch.where(f > 6 / 29, f**3, (f - 16 / 116) / 7.787) * lab.new_tensor(
        [0.95047, 1.0, 1.08883]
    )
    matrix = lab.new_tensor(
        [
            [3.24048134, -1.53715152, -0.49853633],
            [-0.96925495, 1.87599, 0.04155593],
            [0.05564664, -0.20404134, 1.05731107],
        ]
    )
    linear = xyz @ matrix.T
    gray = xyz[:, 1].clamp(0, 1)
    lo = linear.min(1).values
    hi = linear.max(1).values
    low_scale = gray / (gray - lo).clamp_min(1e-7)
    high_scale = (1 - gray) / (hi - gray).clamp_min(1e-7)
    alpha = torch.minimum(torch.ones_like(gray), torch.minimum(low_scale, high_scale)).clamp(0, 1)
    mapped = (gray[:, None] + alpha[:, None] * (linear - gray[:, None])).clamp(0, 1)
    rgb = torch.where(
        mapped > 0.0031308, 1.055 * mapped.clamp_min(1e-7) ** (1 / 2.4) - 0.055, 12.92 * mapped
    ).clamp(0, 1)
    return rgb, alpha
