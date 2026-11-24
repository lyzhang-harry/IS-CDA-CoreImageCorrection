# IS-CDA for optical core scans

IS-CDA (intra-image self-supervised column-wise domain adaptation) corrects peripheral darkening and orange-red casts in RGB scans of cylindrical geological cores. Each scan is processed separately. The brightness network is fitted first, followed by the chromaticity network on the fixed brightness-corrected image. Paired references are used only when scoring synthetic images.

## Installation

Use Python 3.10 and PyTorch 2.7.1. Install PyTorch and torchvision for your hardware, then install the listed dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

The results below were obtained on a single NVIDIA GeForce RTX 5070 Ti Laptop GPU with PyTorch 2.7.1 and CUDA 12.8. The default configuration uses seed 2024, disables TF32 and fits each network for 500 steps. Evaluation uses pyiqa for FSIMc and NIQE. NIQE may need to download its standard parameters on first use.

## Run

```bash
python train.py --mode synthetic
python train.py --mode real
```

The command selects CUDA when available. To use the CPU explicitly, add `--device cpu`.

The default inputs are the 15 paired test images in `data/synthetic/` (targets in `label/`) and the five representative scans in `data/real_cores/`. The synthetic test images come from five parent scans. Their IDs, relative paths and checksums are in [the manifest](data/manifest.json). The full study includes 127 synthetic pairs and 45 real scans.

For another directory, pass `--input_dir` and `--output_dir`. Paired evaluation matches the input filename stem to the reference filename stem in `--gt_dir` and requires equal pixel dimensions. Use a separate output directory for each dataset and parameter setting.

```bash
python train.py --mode real --input_dir ./my_scans --output_dir ./my_results --skip-evaluation
```

After generating the full-resolution PNGs with `train.py`, you can repeat the evaluation with the commands below.

```bash
python evaluate.py --mode synthetic
python evaluate.py --mode real
```

## Model

1. IS-CDA describes each column by its position, robust brightness, orange-red response, saturation and value-channel gradient. The reference band is centered on the peak of the column brightness profile. Its default **total width is 20%** (10% on either side of the peak, clipped at the image boundaries). Use `--reference-width 0.20` for this setting. The earlier `--center_ratio` option specifies the half-width.
2. The brightness network maps the five features at native width to a gain and bias for each column. It fits a brightness profile and uses brightness MK-MMD and regularization. A reference-weighted gain and a lightness mapping produce the brightness-corrected image, which is held fixed while fitting the chromaticity network.
3. The chromaticity network appends the three scaled Lab column medians from the original scan. It predicts saturation scale, hue shift and two Lab offsets on 32 control positions and interpolates them to native width. Its objective includes HS and Lab chromaticity MK-MMD, a same-height reference constraint and regularization. Final rendering accounts for the chromatic response on each side and carries the lightness correction through the reference-band transition.

## Outputs and evaluation

The repository includes side-by-side JPEG previews, per-image and summary CSVs, and an Excel report for each dataset. Running `train.py` writes full-resolution corrected RGB PNGs to `results/synthetic/` or `results/real/`. Evaluation uses these PNGs to compute the corrected-image scores.

Synthetic PSNR and SSIM are computed over the full image. The paired CIEDE2000, lightness and chromaticity errors are also reported at the margins. Real-scan evaluation compares the margins with fixed geometric center windows in the original image at matching heights. The side and center windows have the same coordinates before and after correction, independent of the model's fitted reference band. Each window spans 20% of the image width. The three vertical positions use windows 256 pixels high on synthetic images and 512 pixels high on the real scans. Group scores give each image equal weight.

The result tables also include full-image FSIMc for synthetic images, column brightness variation (`sigma_y`), and side-to-center chromaticity difference (`delta_e_side`). The `delta_e_side` metric uses a full-height center band. The real-scan `edge_ab_input_center` metric uses center windows at matching heights.

## Reproduction on the bundled images

Both networks were fitted from scratch on each bundled image using the default 20% reference band. The table compares the study run and the reproduction run on the same images. The five real scans form a subset of the 45 used in the paper. Both runs selected the same reference and target columns on all 20 images.

| Dataset | Metric | Study run | Reproduction run |
|---|---|---:|---:|
| Synthetic (15) | Whole-image PSNR (dB) | 30.44133 | 30.44340 |
| Synthetic (15) | Peripheral CIEDE2000 | 4.46682 | 4.47305 |
| Synthetic (15) | Whole-image SSIM | 0.97764 | 0.97763 |
| Real (5) | Peripheral lightness discrepancy | 12.72910 | 12.73058 |
| Real (5) | Peripheral chromaticity discrepancy | 1.98038 | 1.97334 |
| Real (5) | Whole-image NIQE | 3.50833 | 3.50857 |

Across the 20 corrected images, the largest absolute difference between runs was 5 on the 0–255 RGB channel scale. In a separate check, rendering from the same fitted parameters produced pixel-identical outputs for HC145_5_1 (synthetic) and HC146_4 (real).

The detailed tables contain the [synthetic metrics](results/synthetic/by_image.csv), [real-scan metrics](results/real/by_image.csv), [comparison between runs](results/reproduction.csv), and [per-image pixel differences and output checksums](results/pixel_comparison.csv).
