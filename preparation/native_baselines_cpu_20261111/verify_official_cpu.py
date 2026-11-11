"""Bounded CPU-only verification of unmodified official native baselines.

This is a synthetic execution smoke, not a restoration benchmark or NPU proof.
It requires the real compiled MMCV ops; no operator fallback is implemented.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

sys.dont_write_bytecode = True

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

LOCKS = {
    "faver": ("FaverNet", "FaverNet_TUDCommon.pth", "c16e5658a639810bb17dc4bc131a631bde6491f9fb5a790c8952dfa16ccdc7b8"),
    "aver": ("AverNet", "AverNet_final.pth", "cdba740bab4c47f37a7b0e18eb147a3acab6897b0911744f7c5fbc30054b57ec"),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--baseline", choices=LOCKS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--aver-import-mode", choices=("official_package", "architecture_only"), default="official_package")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    r = {
        "schema": "official_native_cpu_execution_v1",
        "baseline": a.baseline,
        "scope": "synthetic_cpu_smoke_not_quality_benchmark_or_npu_validation",
        "operator_fallback": False,
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "stage": "environment",
    }
    receipt = a.output / "receipt.json"

    def save():
        receipt.write_text(json.dumps(r, indent=2) + "\n")

    save()
    try:
        import numpy as np
        import torch
        import torchvision
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261111)
        r["versions"] = {}
        for name in ("torch", "torchvision", "numpy", "mmcv", "mmengine", "ninja", "setuptools"):
            try:
                r["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                r["versions"][name] = None
        r["torch_npu_imported"] = "torch_npu" in sys.modules
        assert not r["torch_npu_imported"], "CPU verification must not autoload the NPU backend"
        r["stage"] = "real_mmcv_modulated_deform_conv"
        save()
        import mmcv
        from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d
        import mmcv._ext
        r["mmcv_path"] = mmcv.__file__
        r["mmcv_extension_path"] = mmcv._ext.__file__
        r["mmcv_extension_sha256"] = sha(mmcv._ext.__file__)
        # Exercise all 24 deformable groups used by the official FaverNet.
        x = torch.randn(1, 48, 7, 8)
        weight = torch.randn(8, 48, 3, 3) / 30
        bias = torch.randn(8) / 10
        offset = torch.randn(1, 24 * 2 * 3 * 3, 7, 8) * 0.2
        mask = torch.sigmoid(torch.randn(1, 24 * 3 * 3, 7, 8))
        actual = modulated_deform_conv2d(x, offset, mask, weight, bias, 1, 1, 1, 1, 24)
        reference = torchvision.ops.deform_conv2d(x, offset, weight, bias, padding=(1, 1), mask=mask)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
        r["dcn"] = {"device": str(actual.device), "shape": list(actual.shape), "finite": bool(actual.isfinite().all()), "max_abs_vs_torchvision": float((actual - reference).abs().max()), "atol": 2e-5, "rtol": 2e-5, "deform_groups": 24, "nonzero_offsets": True, "nonconstant_masks": True}
        r["stage"] = "official_model_import_and_strict_load"
        save()
        repo_name, weight_name, expected_sha = LOCKS[a.baseline]
        repo = a.assets / repo_name
        source_lock = json.loads(Path(__file__).with_name("source_lock.json").read_text())[repo_name]
        for relative, expected in source_lock["files"].items():
            assert sha(repo / relative) == expected, "Official source mismatch: " + relative
        r["source_lock"] = {"python_files_verified": len(source_lock["files"]), "tree_sha256": source_lock["tree_sha256"]}
        ckpt = a.assets / "weights" / weight_name
        actual_sha = sha(ckpt)
        assert actual_sha == expected_sha, (actual_sha, expected_sha)
        r["checkpoint"] = {"path": str(ckpt), "sha256": actual_sha, "bytes": ckpt.stat().st_size}
        sys.path.insert(0, str(repo))
        if a.baseline == "faver":
            from models.favernet import Network
            model = Network(cpu_cache_length=100)
            source = repo / "models/favernet.py"
        else:
            r["aver_import_mode"] = a.aver_import_mode
            if a.aver_import_mode == "architecture_only":
                # Explicit inference import adapter: omit training/package startup,
                # but execute all architecture, registry, and operator code intact.
                # These are only package namespaces, not substitute model functions.
                for name, directory in (("basicsr", repo / "basicsr"), ("basicsr.utils", repo / "basicsr/utils"), ("basicsr.archs", repo / "basicsr/archs")):
                    assert name not in sys.modules, "Use a fresh interpreter for architecture-only import"
                    package = types.ModuleType(name)
                    package.__path__ = [str(directory)]
                    package.__package__ = name
                    sys.modules[name] = package
            from basicsr.archs.avernet_arch import AverNet
            model = AverNet(mid_channels=96, prompt_dim=96, num_blocks=7, spynet_pretrained=None, cpu_cache_length=100)
            source = repo / "basicsr/archs/avernet_arch.py"
        r["model_source"] = {"path": str(source), "sha256": sha(source)}
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = payload["params"] if a.baseline == "aver" else payload
        assert any(k.startswith("spynet.") for k in state), "Checkpoint must include trained optical flow"
        loaded = model.load_state_dict(state, strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        model.eval()
        r["strict_load"] = {"passed": True, "missing_keys": loaded.missing_keys, "unexpected_keys": loaded.unexpected_keys, "state_keys": len(state), "parameters": sum(t.numel() for t in model.parameters()), "spynet_keys": sum(k.startswith("spynet.") for k in state)}
        r["stage"] = "real_full_model_forward"
        save()
        # Native dimensions satisfy both official models without altering padding.
        rgb = np.random.default_rng(20261111).integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        inp = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float().unsqueeze(0) / 255
        assert inp.device.type == "cpu"
        t0 = time.monotonic()
        with torch.inference_mode():
            out = model(inp)
        elapsed = time.monotonic() - t0
        assert out.shape == inp.shape, (out.shape, inp.shape)
        assert out.device.type == "cpu" and out.isfinite().all()
        output_np = out[0].permute(0, 2, 3, 1).contiguous().numpy()
        np.savez_compressed(a.output / "synthetic_output.npz", output=output_np)
        r["forward"] = {"passed": True, "device": str(out.device), "dtype": str(out.dtype), "input_shape": list(inp.shape), "output_shape": list(out.shape), "seconds": elapsed, "finite": True, "input_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "output_pixel_sha256": hashlib.sha256(output_np.tobytes()).hexdigest(), "output_min": float(out.min()), "output_max": float(out.max()), "mean_abs_change_from_input": float((out - inp).abs().mean()), "threads": 1, "inference_mode": True, "padding": "none_dimensions_native_divisible_by_8", "future_context": "official_bidirectional_full_clip"}
        r["status"] = "passed_cpu_synthetic_only"
        r["stage"] = "complete"
        save()
        print(json.dumps(r, indent=2), flush=True)
        return 0
    except Exception as e:
        r["status"] = "failed"
        r["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        save()
        print(json.dumps(r, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
