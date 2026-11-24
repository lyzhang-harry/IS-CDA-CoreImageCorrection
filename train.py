"""Fit IS-CDA to individual core scans."""

import argparse
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import torch
from models.brightness import fit_brightness
from models.chromaticity import fit_chromaticity
from utils.feature_extraction import column_features, reference_gate
from utils.image_io import image_files, read_rgb, write_png
from utils.luminance import AdaptiveLuminance, apply_y, gain_field
from utils.post_processing import LightnessField, apply_final_lightness, stage_at_current_brightness
from utils.side_chroma import SideChroma

ROOT = Path(__file__).resolve().parent


def correct_image(raw, device, reference_width=0.20):
    if not 0 < reference_width < 1:
        raise ValueError("Reference width must be between zero and one")
    features, top, ref, edge, valid = column_features(raw, reference_width)
    gain, bias = fit_brightness(raw, features, top, ref, edge, valid, device)
    del valid
    height, width = raw.shape[:2]
    bounds = (0, 0, width, height)
    initial = apply_y(raw, gain_field(gain, bias, top, ref))
    mapper = AdaptiveLuminance(raw, initial, ref, reference_gate(ref))
    stage = mapper.render(bounds, device, initial)
    del initial, mapper
    fields = fit_chromaticity(raw, stage, features, ref, edge, device)
    luminance = stage_at_current_brightness(device, raw, stage, ref)
    del stage
    color = SideChroma(device, raw, luminance, ref, fields).render(bounds)
    return apply_final_lightness(
        device, luminance, color, LightnessField(raw, luminance, ref), bounds
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--gt_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    widths = parser.add_mutually_exclusive_group()
    widths.add_argument(
        "--reference-width",
        type=float,
        default=0.20,
        help="Total width as a fraction of the scan width",
    )
    widths.add_argument("--center_ratio", type=float, help="Half-width fraction (legacy option)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Write corrected PNGs only")
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    width = args.reference_width if args.center_ratio is None else 2 * args.center_ratio
    if not 0 < width < 1:
        parser.error("The total reference width must lie between zero and one")
    src = args.input_dir or ROOT / "data" / (
        "synthetic" if args.mode == "synthetic" else "real_cores"
    )
    dst = args.output_dir or ROOT / "results" / args.mode
    if src.resolve() == dst.resolve():
        parser.error("Choose an output directory distinct from the input directory")
    dst.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = image_files(src)
    for index, (ident, path) in enumerate(inputs.items(), 1):
        start = perf_counter()
        print(f"[{index}/{len(inputs)}] {ident}", flush=True)
        result = correct_image(read_rgb(path), torch.device(args.device), width)
        write_png(dst / f"{ident}_corrected.png", result)
        del result
        print(f"  saved in {perf_counter() - start:.1f} s", flush=True)
    if not args.skip_evaluation:
        from evaluate import evaluate_dataset

        evaluate_dataset(args.mode, src, dst, args.gt_dir or src / "label", args.device)


if __name__ == "__main__":
    main()
