#!/usr/bin/env python3
"""Prepared native-baseline adapter; model execution has not been validated.

Consumes only an observation's uint8 RGB THWC frames; never loads a GT image.
The first supported protocol is one complete 2..100-frame native-size clip.
CPU/CUDA only until separate Ascend operator and model parity checks pass.
Inspect-only needs NumPy, but does not import torch or third-party model code.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
LOCKS = {
    "favernet": {"repo": "FaverNet", "commit": "14769f22685f4850c5e203d493c99ad2eba5b68b",
                 "checkpoint": "FaverNet_TUDCommon.pth",
                 "sha256": "c16e5658a639810bb17dc4bc131a631bde6491f9fb5a790c8952dfa16ccdc7b8",
                 "padding_multiple": 4},
    "avernet": {"repo": "AverNet", "commit": "c4eb019f71fe6a2ccb6807dca92451fd8d8b8cd3",
                "checkpoint": "AverNet_final.pth",
                "sha256": "cdba740bab4c47f37a7b0e18eb147a3acab6897b0911744f7c5fbc30054b57ec",
                "padding_multiple": 8},
}


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def read_observation(path, expected_pixel_sha256):
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"])
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB frames with shape [T,H,W,3]")
    if not 2 <= len(frames) <= 100:
        raise ValueError("Prepared protocol requires 2..100 frames; no implicit clip truncation")
    if min(frames.shape[1:3]) < 256:
        raise ValueError("Prepared common protocol requires H,W >=256; no implicit resizing")
    actual = hashlib.sha256(frames.tobytes(order="C")).hexdigest()
    if actual != expected_pixel_sha256:
        raise ValueError("Observation pixels do not match the supplied manifest SHA256")
    return frames, actual


def verify_source(repo, expected_commit):
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != expected_commit:
        raise ValueError(f"Source commit differs: {head}")
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--"], check=True)
    return head


def pad_official(video, multiple):
    """Match the official tail-flip padding, then caller crops original H,W."""
    import torch
    h, w = video.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph:
        video = torch.cat((video, torch.flip(video[..., -ph:, :], (3,))), dim=3)
    if pw:
        video = torch.cat((video, torch.flip(video[..., -pw:], (4,))), dim=4)
    return video


def run(args):
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        raise ValueError("Prepared runner supports cpu or cuda:<index>; Ascend parity is pending")
    lock = LOCKS[args.method]
    repo = (args.repo or ROOT / lock["repo"]).resolve()
    checkpoint = (args.checkpoint or ROOT / "weights" / lock["checkpoint"]).resolve()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    frames, pixel_sha = read_observation(args.input_npz, args.expected_pixel_sha256)
    commit = verify_source(repo, lock["commit"])
    checkpoint_sha = sha_file(checkpoint)
    if lock["sha256"] is None or checkpoint_sha != lock["sha256"]:
        raise ValueError("Checkpoint SHA256 is missing or differs from the acquired official asset")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "openh3.native_baseline.lossless_rgb.v1", "method": args.method,
              "status": "input_preflight_complete_inference_not_run", "task": args.task,
              "source_commit": commit, "checkpoint_sha256": checkpoint_sha,
              "runner_sha256": sha_file(__file__), "input_npz_sha256": sha_file(args.input_npz),
              "input_pixel_sha256": pixel_sha, "input_shape": list(frames.shape),
              "input_color": "RGB", "input_range": "uint8 / 255 -> float32 [0,1]",
              "temporal_access": "offline_bidirectional_entire_supplied_clip",
              "maximum_future_frames": len(frames) - 1,
              "spatial_padding": {"mode": "official_tail_flip", "multiple": lock["padding_multiple"]},
              "resizing": False, "tiling": False, "test_time_augmentation": False,
              "gt_read": False, "device": args.device, "checkpoint_strict_load": None,
              "evaluation_metrics": None, "quality_verdict": "requires_review"}
    metadata_path = args.out / "run.json"
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.inspect_only:
        return report
    try:
        import torch
        if args.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(args.device))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        sys.path.insert(0, str(repo))
        if args.method == "favernet":
            module = importlib.import_module("models.favernet")
            model = module.Network(cpu_cache_length=100)
        else:
            module = importlib.import_module("basicsr.archs.avernet_arch")
            model = module.AverNet(mid_channels=96, prompt_dim=96, num_blocks=7,
                                  spynet_pretrained=None, cpu_cache_length=100)
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise RuntimeError("Imported model does not belong to the pinned checkout")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload if args.method == "favernet" else payload["params"]
        if not any(key.startswith("spynet.") for key in state):
            raise ValueError("Full checkpoint is missing SPyNet; refuse random flow initialization")
        model.load_state_dict(state, strict=True)
        model.eval().to(args.device)
        report["checkpoint_strict_load"] = True
        report["torch_version"] = str(torch.__version__)
        report["mmcv_version"] = importlib.import_module("mmcv").__version__
        video = torch.from_numpy(np.array(frames.transpose(0, 3, 1, 2), copy=True, order="C"))
        video = video.float().div_(255)[None].to(args.device)
        video = pad_official(video, lock["padding_multiple"])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(video)[..., :frames.shape[1], :frames.shape[2]]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        report["one_shot_forward_seconds_not_a_benchmark"] = time.perf_counter() - started
        rgb_raw = output[0].float().cpu().numpy().transpose(0, 2, 3, 1).copy()
        if rgb_raw.shape != frames.shape or not np.isfinite(rgb_raw).all():
            raise ValueError("Invalid output geometry or nonfinite values")
        rgb = np.clip(rgb_raw, 0, 1)
        output_path = args.out / "restored_float32.npz"
        np.savez_compressed(output_path, rgb_raw=rgb_raw, rgb=rgb)
        report.update(status="inference_completed_metrics_not_run", output_npz_sha256=sha_file(output_path),
                      output_rgb_sha256=hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
                      output_raw_range=[float(rgb_raw.min()), float(rgb_raw.max())],
                      output_storage="float32 RGB THWC lossless NPZ; raw plus clamp[0,1]; no uint8 rounding")
    except Exception as error:
        report.update(status="inference_failed", error_type=type(error).__name__, error=str(error))
        metadata_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(LOCKS), required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--expected-pixel-sha256", required=True)
    parser.add_argument("--task", choices=("deblur", "denoise", "compression", "identity", "near_clean"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inspect-only", action="store_true")
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("method", "status", "input_shape", "checkpoint_sha256")}))


if __name__ == "__main__":
    main()
