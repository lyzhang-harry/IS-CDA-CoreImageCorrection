from importlib import import_module
import cv2
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.color.colorconv import xyz_from_rgb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

LEVELS = np.arange(256, dtype=np.float64) / 255.0
LINEAR = np.where(LEVELS <= 0.04045, LEVELS / 12.92, ((LEVELS + 0.055) / 1.055) ** 2.4)
BLUE, GREEN, RED = xyz_from_rgb[1, ::-1, None] * LINEAR
EDGE = (0, 2, 3, 5, 6, 8)
CENTER = (1, 4, 7)


def banded_niqe(device):
    """Keep full-image NIQE while bounding intermediate tensor sizes."""
    import pyiqa

    arch = import_module("pyiqa.archs.niqe_arch")
    resize = import_module("pyiqa.matlab_utils.resize")
    blockproc = arch.blockproc
    normalize = arch.normalize_img_with_gauss
    resize_axis = resize.resize_1d

    def block_batches(x, kernel, fun):
        bh, bw = kernel
        batch, _, h, w = x.shape
        columns = w // bw
        parts = []
        height = bh * max(1, 64 // columns)
        for section in x.split(height, dim=-2):
            features = blockproc(section, kernel, fun)
            parts.append(features.reshape(batch, columns, section.shape[-2] // bh, -1))
        return torch.cat(parts, dim=2).flatten(1, 2)

    def halo_normalize(x, padding):
        h = x.shape[-2]
        out = torch.empty_like(x)
        for lo in range(0, h, 384):
            hi = min(lo + 384, h)
            l0 = max(0, lo - 3)
            h0 = min(h, hi + 3)
            values = normalize(x[..., l0:h0, :], padding=padding)
            out[..., lo:hi, :] = values[..., lo - l0 : hi - l0, :]
        return out

    def axis_batches(x, dim, **kwargs):
        other = -1 if dim in (-2, 2) else -2
        chunks = x.split(384, dim=other)
        return torch.cat(
            [resize_axis(part.contiguous(), dim, **kwargs) for part in chunks], dim=other
        )

    original = pyiqa.create_metric("niqe", device=device, as_loss=False)

    def evaluate(x):
        # These replacements live only around the NIQE call in this process.
        arch.blockproc = block_batches
        arch.normalize_img_with_gauss = halo_normalize
        resize.resize_1d = axis_batches
        try:
            return original(x)
        finally:
            arch.blockproc = blockproc
            arch.normalize_img_with_gauss = normalize
            resize.resize_1d = resize_axis

    return evaluate


class Quality:
    def __init__(self, device):
        self.device = device
        self.models = {}

    def call(self, metric, *images):
        import pyiqa

        devices = [self.device, "cpu"] if self.device == "cuda" else ["cpu"]
        for device in devices:
            key = (metric, device)
            inputs = None
            try:
                if key not in self.models:
                    self.models[key] = (
                        banded_niqe(device)
                        if metric == "niqe"
                        else pyiqa.create_metric("fsim", device=device, chromatic=True)
                    )
                inputs = [
                    torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].to(
                        device=device, dtype=torch.float32
                    )
                    / 255.0
                    for a in images
                ]
                with torch.inference_mode():
                    value = float(self.models[key](*inputs).item())
                if not np.isfinite(value):
                    raise FloatingPointError(f"{metric} returned {value}")
                return value
            except torch.cuda.OutOfMemoryError:
                if device == "cpu":
                    raise
                inputs = None
                torch.cuda.empty_cache()


def paper_statistics(image):
    """Measure column brightness variation and side-to-center Lab chromaticity differences."""
    h, w = image.shape[:2]
    k = max(1, int(0.2 * h))
    top = []
    for x in range(0, w, 32):
        part = image[:, x : x + 32]
        y = 0.114 * part[..., 2] + 0.587 * part[..., 1] + 0.299 * part[..., 0]
        # Values <= 5 cannot enter the top k if there are k values > 5.
        top.extend(np.partition(y.T, -k, axis=-1)[:, -k:].mean(axis=-1) / 255.0)
    side = max(1, int(0.12 * w))
    mid = max(1, int(0.10 * w))

    def mean_ab(i, j):
        return cv2.cvtColor(image[:, i:j], cv2.COLOR_RGB2LAB)[..., 1:].mean(axis=(0, 1))

    base = mean_ab(w // 2 - mid, w // 2 + mid)
    left = float(np.linalg.norm(mean_ab(0, side) - base))
    right = float(np.linalg.norm(mean_ab(w - side, w) - base))
    return dict(
        sigma_y=float(np.std(top)),
        delta_e_side=(left + right) / 2,
        delta_e_left=left,
        delta_e_right=right,
    )


def paired_statistics(image, target):
    error = 0.0
    for lo in range(0, len(image), 256):
        error += deltaE_ciede2000(
            rgb2lab(image[lo : lo + 256]), rgb2lab(target[lo : lo + 256])
        ).sum()
    psnr = peak_signal_noise_ratio(target, image, data_range=255)
    return dict(
        psnr_db=float(psnr),
        ssim_rgb=float(
            structural_similarity(target, image, win_size=11, channel_axis=2, data_range=255)
        ),
        ciede2000=float(error / (image.shape[0] * image.shape[1])),
    )


def lightness(bgr):
    y = BLUE[bgr[..., 0]] + GREEN[bgr[..., 1]] + RED[bgr[..., 2]]
    return np.where(y > 0.008856, 116 * np.cbrt(y) - 16, (116 * 7.787) * y).astype(np.float32)


def windows(image, mode):
    h, w = image.shape[:2]
    span = max(1, round(0.20 * w))
    height = min(h, 512 if mode == "real" else 256)
    return [
        image[y : y + height, x : x + span]
        for y in (round((h - height) * f) for f in (0.2, 0.5, 0.8))
        for x in (0, (w - span) // 2, w - span)
    ]


def grid_lightness_gap(raw_patches, patches):
    residuals = []
    anchors = [
        np.asarray(
            [
                np.median(row)
                for row in np.array_split(lightness(raw_patches[i][..., ::-1]), 4, axis=0)
            ]
        )
        for i in CENTER
    ]
    for i in EDGE:
        lstar = lightness(patches[i][..., ::-1])
        cells = np.asarray(
            [
                [np.median(cell) for cell in np.array_split(row, 4, axis=1)]
                for row in np.array_split(lstar, 4, axis=0)
            ]
        )
        residuals.append(cells - anchors[i // 3][:, None])
    return float(np.abs(np.stack(residuals)).mean())


def measure(raw, image, mode, quality, target=None):
    if raw.shape != image.shape or (target is not None and raw.shape != target.shape):
        raise ValueError(
            "Input, corrected image and reference must have matching pixel coordinates"
        )
    patches = windows(image, mode)
    source = windows(raw, mode)
    anchors = [rgb2lab(source[i])[..., 1:].mean(axis=(0, 1)) for i in CENTER]
    delta = np.stack(
        [rgb2lab(patches[i])[..., 1:].mean(axis=(0, 1)) - anchors[i // 3] for i in EDGE]
    )
    score = dict(
        Lstar_grid_gap=grid_lightness_gap(source, patches),
        edge_ab_input_center=float(np.linalg.norm(delta, axis=1).mean()),
        edge_abs_delta_a=float(np.abs(delta[:, 0]).mean()),
        edge_abs_delta_b=float(np.abs(delta[:, 1]).mean()),
        center_nearwhite=float(np.mean([(patches[i] >= 250).all(-1).mean() for i in CENTER])),
        niqe=quality.call("niqe", image),
    )
    legacy = paper_statistics(image)
    score.update({k: legacy[k] for k in ("sigma_y", "delta_e_side")})
    if target is not None:
        score.update(paired_statistics(image, target))
        score["fsimc"] = quality.call("fsimc", image, target)
        truth = windows(target, mode)
        records = []
        for i in EDGE:
            a, b = rgb2lab(patches[i]), rgb2lab(truth[i])
            diff = a - b
            records.append(
                (
                    np.abs(diff[..., 0]).mean(),
                    np.linalg.norm(diff[..., 1:], axis=2).mean(),
                    deltaE_ciede2000(b, a).mean(),
                    np.square(patches[i].astype(np.float32) - truth[i]).mean(dtype=np.float64),
                    structural_similarity(
                        truth[i], patches[i], win_size=11, channel_axis=2, data_range=255
                    ),
                    quality.call("fsimc", patches[i], truth[i]),
                )
            )
        lstar, ab, de, mse, ssim, fsim = np.mean(records, axis=0)
        score.update(
            edge_gt_l_mae=lstar,
            edge_gt_ab_mae=ab,
            edge_gt_de00=de,
            edge_gt_psnr_db=float(10 * np.log10(255**2 / mse)) if mse else np.inf,
            edge_gt_ssim=ssim,
            edge_fsimc=fsim,
            center_gt_de00=float(
                np.mean(
                    [
                        deltaE_ciede2000(rgb2lab(truth[i]), rgb2lab(patches[i])).mean()
                        for i in CENTER
                    ]
                )
            ),
        )
    return score
