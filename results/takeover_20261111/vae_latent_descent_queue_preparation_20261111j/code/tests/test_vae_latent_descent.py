"""Small real Torch CPU decode/checkpoint/backward; no Comfy or NPU imports."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_vae_latent_descent as probe
from compare_vae_gradient_repeatability import atomic_json


class MathAndFilesystemTests(unittest.TestCase):
    def test_fixed_rms_update_and_zero_or_nonfinite_gradient_rejection(self):
        source = np.linspace(-1, 1, 64, dtype=np.float32)
        raw = np.linspace(.25, 1, 64, dtype=np.float32)
        minus, plus, record = probe.normalized_step(source, raw)
        self.assertEqual(minus.dtype, np.float32)
        self.assertAlmostEqual(record["actual_float32_step_rms"], .001, places=8)
        self.assertGreater(record["gradient_dot_direction"], 0)
        self.assertTrue(np.all(minus < source))
        self.assertTrue(np.all(plus > source))
        for bad in (np.zeros_like(raw), np.full_like(raw, np.nan)):
            with self.assertRaises(ValueError): probe.normalized_step(source, bad)

    def test_actual_half_cast_can_erase_a_nonzero_float32_step(self):
        before = np.ones(16, np.float32)
        after = before + np.float32(1e-5)
        self.assertTrue(np.any(before != after))
        result = probe.cast_change(before.astype(np.float16), after.astype(np.float16))
        self.assertEqual(result["changed_coordinates"], 0)
        self.assertEqual(result["actual_cast_step_rms"], 0)

    def test_existing_output_refused_before_device_or_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory), pretrained_manifest=Path("absent"))
            with self.assertRaises(FileExistsError): probe.execute(args)


class NativeBackendCPUTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("run in existing remote Torch CPU environment")
        from vae_checkpoint_context import checkpoint_decoder_blocks
        torch.set_num_threads(1)
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.report = probe.new_report()
        self.source = np.linspace(.2, .8, 24, dtype=np.float32).reshape(6, 4)
        geometry = patch.object(probe, "RGB_SHAPE", self.source.shape)
        geometry.start()
        self.addCleanup(geometry.stop)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.001), requires_grad=False)
            def forward(self, x):
                return x + self.weight * x.square()

        class Stage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Module()
                self.decoder.transformer_blocks = torch.nn.ModuleList(Block() for _ in range(36))
            def forward(self, x):
                pieces = []
                for chunk in x.float().chunk(3):
                    for block in self.decoder.transformer_blocks:
                        chunk = block(chunk)
                    pieces.append(chunk)
                return torch.cat(pieces)

        self.stage = Stage().eval()
        self.weights = {k: v.clone() for k, v in self.stage.state_dict().items()}
        self.backend = probe.NativeBackend.__new__(probe.NativeBackend)
        self.backend.torch = torch
        self.backend.vae = SimpleNamespace(vae_dtype=torch.float16, first_stage_model=self.stage, decode=self.stage)
        self.backend.h3 = SimpleNamespace(dev="cpu", decode=self.stage)
        self.backend.target = torch.zeros_like(torch.from_numpy(self.source))
        self.backend.native_math = nullcontext
        self.backend.decoder_math = lambda *args: nullcontext()
        self.backend.checkpoint_scope = checkpoint_decoder_blocks

    def update(self):
        atomic_json(self.out / "report.json", self.report)

    def assert_restored(self):
        self.assertTrue(all("forward" not in b.__dict__ for b in self.stage.decoder.transformer_blocks))
        self.assertTrue(all(self.torch.equal(self.weights[k], v) for k, v in self.stage.state_dict().items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.stage.parameters()))

    def test_real_three_steps_exact_budget_parity_freeze_and_restore(self):
        probe.run_protocol(self.backend, self.source, self.out, self.report, self.update)
        self.assertTrue(self.report["protocol_complete"])
        self.assertEqual(self.report["forwards_completed"], 8)
        self.assertEqual(self.report["backwards_completed"], 3)
        self.assertTrue(all(row["accepted"] and row["actual_loss_decrease"] > 0 for row in self.report["steps"]))
        first = self.report["steps"][0]["directional_check"]
        self.assertGreater(first["central_difference"], 0)
        self.assertLess(first["relative_magnitude_error_descriptive_only"], .1)
        for row in self.report["steps"]:
            self.assertAlmostEqual(row["actual_float32_step_rms"], .001, places=7)
            self.assertGreater(row["float16_cast_change"]["changed_coordinates"], 0)
        self.assertTrue(all(a["checkpoint_forward_calls"] == a["recompute_calls"] == 108 for a in self.report["checkpoint"].values()))
        self.assert_restored()

    def test_wrong_descent_direction_stops_without_positive_probe_or_later_step(self):
        original = self.backend.gradient
        self.backend.gradient = lambda *args: -original(*args)  # inject a bad update direction
        with self.assertRaisesRegex(ValueError, "stop without further forwards"):
            probe.run_protocol(self.backend, self.source, self.out, self.report, self.update)
        saved = json.loads((self.out / "report.json").read_text())
        self.assertEqual(saved["forwards_completed"], 3)
        self.assertEqual(saved["backwards_completed"], 1)
        self.assertEqual(len(saved["steps"]), 1)
        self.assertFalse(saved["steps"][0]["accepted"])
        self.assertTrue((self.out / "step1_gradient_raw.npy").is_file())
        self.assert_restored()

    def test_parity_failure_stops_before_backward_and_retains_restore_evidence(self):
        self.backend.h3.decode = lambda x: self.stage(x) + .1
        with self.assertRaisesRegex(ValueError, "parity"):
            probe.run_protocol(self.backend, self.source, self.out, self.report, self.update)
        saved = json.loads((self.out / "report.json").read_text())
        self.assertEqual(saved["forwards_completed"], 2)
        self.assertEqual(saved["backward_attempts"], 0)
        self.assertFalse(saved["evaluations"]["step1_gradient"]["parity"]["allclose"])
        self.assertTrue(saved["checkpoint"]["step1_gradient"]["restored"])
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
