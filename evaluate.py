"""Score native corrected PNGs against paired targets or fixed within-scan windows."""

import argparse
from pathlib import Path
import cv2
import pandas as pd
from PIL import Image, ImageDraw
import torch
from utils.image_io import image_files, read_rgb
from utils.metrics import Quality, measure

ROOT = Path(__file__).resolve().parent


def preview(raw, corrected, path):
    h, w = raw.shape[:2]
    scale = min(1.0, 1200 / h, 600 / w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    a, b = (
        Image.fromarray(cv2.resize(im, size, interpolation=cv2.INTER_AREA))
        for im in (raw, corrected)
    )
    canvas = Image.new("RGB", (2 * size[0] + 10, size[1] + 26), "white")
    canvas.paste(a, (0, 26))
    canvas.paste(b, (size[0] + 10, 26))
    text = ImageDraw.Draw(canvas)
    text.text((1, 5), "Input", fill="black")
    text.text((size[0] + 11, 5), "IS-CDA", fill="black")
    canvas.save(path, quality=91)


def evaluate_dataset(mode, input_dir, output_dir, gt_dir, device):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = image_files(input_dir)
    labels = image_files(gt_dir) if mode == "synthetic" else {}
    missing = set(inputs) - set(labels) if mode == "synthetic" else set()
    if missing:
        raise ValueError(f"Missing paired references: {sorted(missing)}")
    quality = Quality(device)
    rows = []
    previews = output_dir / "previews"
    previews.mkdir(exist_ok=True)
    for index, (ident, path) in enumerate(inputs.items(), 1):
        print(f"Scoring [{index}/{len(inputs)}] {ident}", flush=True)
        raw = read_rgb(path)
        out = read_rgb(output_dir / f"{ident}_corrected.png")
        target = read_rgb(labels[ident]) if mode == "synthetic" else None
        for method, image in (("Input", raw), ("IS-CDA", out)):
            rows.append(
                dict(image_id=ident, method=method, **measure(raw, image, mode, quality, target))
            )
        preview(raw, out, previews / f"{ident}.jpg")
    frame = pd.DataFrame(rows)
    summary = frame.drop(columns="image_id").groupby("method", sort=False).mean()
    summary.insert(0, "n", len(inputs))
    frame.to_csv(output_dir / "by_image.csv", index=False)
    summary.to_csv(output_dir / "summary.csv")
    with pd.ExcelWriter(output_dir / f"Evaluation_Report_{mode}.xlsx") as writer:
        frame.to_excel(writer, sheet_name="By image", index=False)
        summary.to_excel(writer, sheet_name="Mean")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real"), required=True)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--gt_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    src = args.input_dir or ROOT / "data" / (
        "synthetic" if args.mode == "synthetic" else "real_cores"
    )
    dst = args.output_dir or ROOT / "results" / args.mode
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    print(evaluate_dataset(args.mode, src, dst, args.gt_dir or src / "label", args.device))
