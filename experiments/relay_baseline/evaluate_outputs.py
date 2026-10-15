"""Full-RGB, no-border-crop metrics for the self-degraded demo probe only."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="label=/absolute/png/dir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods = {x.split("=", 1)[0]: Path(x.split("=", 1)[1]) for x in args.prediction}
    names = sorted(p.name for p in args.ground_truth.glob("*.png"))
    result = {
        "scope": "Single self-degraded official demo prefix; exploratory, not a held-out dataset result.",
        "metric_protocol": "Raw PNG predictions; uint8 full RGB; data_range255; no border crop; SSIM gaussian_weights=True sigma1.5 use_sample_covariance=False; temporal residual L1 unwarped on normalized RGB errors.",
        "frames": len(names), "methods": {},
    }
    for label, directory in methods.items():
        rows, previous_error = [], None
        for index, name in enumerate(names):
            gt, pred = read(args.ground_truth / name), read(directory / name)
            if pred.shape != gt.shape:
                raise ValueError(f"Shape mismatch for {label}/{name}: {pred.shape} vs {gt.shape}")
            error = pred.astype(np.float32) / 255 - gt.astype(np.float32) / 255
            rows.append({
                "frame": index,
                "psnr_rgb_db": float(peak_signal_noise_ratio(gt, pred, data_range=255)),
                "ssim_rgb": float(structural_similarity(gt, pred, data_range=255, channel_axis=2,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
                "unwarped_temporal_residual_l1": None if previous_error is None else float(np.mean(np.abs(error - previous_error))),
            })
            previous_error = error
        result["methods"][label] = {
            "prediction_directory": str(directory), "per_frame": rows,
            "mean_psnr_rgb_db": float(np.mean([r["psnr_rgb_db"] for r in rows])),
            "mean_ssim_rgb": float(np.mean([r["ssim_rgb"] for r in rows])),
            "mean_unwarped_temporal_residual_l1": float(np.mean([r["unwarped_temporal_residual_l1"] for r in rows[1:]])),
            "reference_frame_mean_psnr": float(np.mean([r["psnr_rgb_db"] for r in rows if r["frame"] % 15 == 0])),
            "intermediate_frame_mean_psnr": float(np.mean([r["psnr_rgb_db"] for r in rows if r["frame"] % 15 != 0])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    selected = [i for i in [0, 7, 14, 15, 16, 30] if i < len(names)]
    first = read(args.ground_truth / names[0])
    h, w = first.shape[:2]
    columns = {"GT_source_crop": args.ground_truth, **methods}
    sheet = Image.new("RGB", (w * len(columns), (h + 28) * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(selected):
        for col, (label, directory) in enumerate(columns.items()):
            left, top = col * w, row * (h + 28)
            draw.text((left + 5, top + 6), f"{label}, frame {index}", fill="black")
            sheet.paste(Image.open(directory / names[index]).convert("RGB"), (left, top + 28))
    sheet.save(args.output.with_suffix(".comparison.png"))
    print(json.dumps({label: {k: v for k, v in metrics.items() if k.startswith("mean_")}
                      for label, metrics in result["methods"].items()}), flush=True)


if __name__ == "__main__":
    main()
