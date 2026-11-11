"""CPU evidence/contract tests; no H3 import, assets or accelerator execution."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_h3_trajectory_grad as probe
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch unavailable")
class ProbeTests(unittest.TestCase):
    def toy(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = torch.nn.Module()
                self.attn.qkv_proj = torch.nn.Linear(2, 2)
                self.attn.out_proj = torch.nn.Linear(2, 2)
                self.mlp = torch.nn.Module()
                self.mlp.fc1 = torch.nn.Linear(2, 2)
                self.mlp.fc2 = torch.nn.Linear(2, 2)
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([Block() for _ in range(50)])
        model.register_buffer("frozen_buffer", torch.ones(2))
        model.requires_grad_(False)
        model._openh3_pretrained_record = {"test": True}
        return model

    def test_canonical_base_and_strict_adapter_geometry(self):
        from openh3_runtime import inject_adaptation
        model = self.toy()
        before = probe.base_fingerprint(model, torch)
        versions = probe.snapshot_versions(model)
        names = inject_adaptation(model, 8, 101)
        after = probe.base_fingerprint(model, torch, names)
        self.assertEqual(before, after)
        self.assertTrue(probe.check_versions(model, versions, names)["passed"])
        audit = probe.adapter_audit(model, names, torch)
        self.assertEqual(audit["trainable_tensors"], 400)
        with torch.no_grad():
            model.blocks[0].attn.qkv_proj.lora_b[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "nonzero B"):
            probe.adapter_audit(model, names, torch)

    def test_rank_or_target_subset_rejected(self):
        from openh3_runtime import inject_adaptation
        model = self.toy()
        names = inject_adaptation(model, 4, 101)
        with self.assertRaisesRegex(ValueError, "rank"):
            probe.adapter_audit(model, names, torch)
        with self.assertRaisesRegex(ValueError, "200"):
            probe.adapter_audit(model, names[:-1], torch)

    def test_live_versions_detect_object_and_buffer_mutation(self):
        model = self.toy()
        snapshot = probe.snapshot_versions(model)
        self.assertTrue(probe.check_versions(model, snapshot)["passed"])
        with torch.no_grad():
            model.frozen_buffer.add_(1)
        self.assertFalse(probe.check_versions(model, snapshot)["passed"])
        snapshot = probe.snapshot_versions(model)
        module = model.blocks[0].mlp.fc1
        module.weight = torch.nn.Parameter(module.weight.clone(), requires_grad=False)
        self.assertFalse(probe.check_versions(model, snapshot)["passed"])

    def test_exact_parity_blocks_backward_even_tiny_audio_delta(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = probe.Evidence(Path(temp) / "out", {})
            x = torch.tensor([2.0], requires_grad=True)
            candidate = {"video": x * 1, "audio": x * 1}
            reference = {"video": x.detach().clone(), "audio": x.detach().clone() + 1e-6}
            with self.assertRaisesRegex(RuntimeError, "strict final"):
                probe.parity_then_backward(candidate, reference, evidence, torch)
            saved = json.loads((evidence.path / "partial.json").read_text())
            self.assertFalse(saved["parity"]["audio"]["torch_equal"])
            self.assertTrue((evidence.path / "latent_outputs.npz").is_file())
            self.assertEqual(saved["actual"]["backward_attempts"], 0)
            self.assertIsNone(x.grad)

    def test_parity_allows_exactly_one_same_graph_backward(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = probe.Evidence(Path(temp) / "out", {})
            x = torch.tensor([2.0], requires_grad=True)
            candidate = {"video": x.square(), "audio": x * 3}
            reference = {k: v.detach().clone() for k, v in candidate.items()}
            probe.parity_then_backward(candidate, reference, evidence, torch)
            self.assertEqual(evidence.data["actual"]["completed_backward_calls"], 1)
            self.assertAlmostEqual(x.grad.item(), 32.36, places=4)
            self.assertEqual(evidence.data["parity"]["video"]["max_abs"], 0)

    def test_nonfinite_outputs_fail_before_backward(self):
        a = torch.tensor([float("nan")])
        self.assertFalse(probe.parity_metrics(a, a, torch)["torch_equal"])

    def test_gradient_arrays_saved_before_finite_verdict(self):
        from lora import LoRALinear
        model = torch.nn.Module()
        model.adapter = LoRALinear(torch.nn.Linear(2, 2).requires_grad_(False), 8)
        model.adapter.lora_a.grad = torch.zeros_like(model.adapter.lora_a)
        model.adapter.lora_b.grad = torch.ones_like(model.adapter.lora_b)
        with tempfile.TemporaryDirectory() as temp:
            audit = probe.gradient_audit(model, torch, temp)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["B_nonzero_tensors"], 1)
            self.assertTrue(Path(audit["tensors"]["adapter.lora_a"]["array_file"]).is_file())
            model.adapter.lora_b.grad[0, 0] = float("inf")
            audit = probe.gradient_audit(model, torch, temp)
            self.assertFalse(audit["passed"])
            self.assertEqual(audit["tensors"]["adapter.lora_b"]["inf_count"], 1)

    def test_missing_and_zero_B_gradients_explicit_and_fail(self):
        from lora import LoRALinear
        model = torch.nn.Module()
        model.adapter = LoRALinear(torch.nn.Linear(2, 2).requires_grad_(False), 8)
        audit = probe.gradient_audit(model, torch)
        self.assertEqual(len(audit["missing"]), 2)
        self.assertFalse(audit["passed"])
        model.adapter.lora_b.grad = torch.zeros_like(model.adapter.lora_b)
        audit = probe.gradient_audit(model, torch)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["B_l2"], 0)

    def test_counter_sees_checkpoint_recompute_and_restores_exception(self):
        from openh3_train_trajectory import _checkpoint_modules
        class Block(torch.nn.Module):
            def forward(self, x):
                return x.sin().square()
        block = Block()
        original = block.forward
        phase, counts = ["candidate"], {}
        x = torch.tensor([0.4], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "after backward"):
            with probe.count_block_calls([block], lambda: phase[0], counts):
                with _checkpoint_modules([block], True, lambda: True):
                    y = block(x)
                    phase[0] = "backward"
                    y.sum().backward()
                    raise RuntimeError("after backward")
        self.assertEqual(counts["candidate"]["0"], 1)
        self.assertEqual(counts["backward"]["0"], 1)
        self.assertEqual(block.forward, original)
        self.assertNotIn("forward", block.__dict__)

    def test_noise_observer_is_passthrough_and_restores(self):
        class Nested:
            def __init__(self, values): self.values = values
            def unbind(self): return self.values
        initial = Nested((torch.ones(1, 3), torch.zeros(1, 2)))
        noise = Nested((torch.ones(1, 3) * 2, torch.ones(1, 2) * 3))
        def original(latent, seed):
            self.assertIs(latent, initial)
            self.assertEqual(seed, 101)
            return noise
        def pack(values): return torch.cat(values, -1), [v.shape for v in values]
        module = SimpleNamespace(prepare_noise=original)
        records = []
        with probe.observe_noise(module, pack, lambda: "reference", records, torch):
            self.assertIs(module.prepare_noise(initial, 101), noise)
        self.assertIs(module.prepare_noise, original)
        self.assertEqual(len(records), 1)
        self.assertNotEqual(records[0]["packed_initial_sha256"], records[0]["packed_noise_sha256"])

    def test_forbidden_vae_guard_restores_on_error(self):
        class VAE:
            def decode(self, x): return x
            def encode(self, x): return x
        h3 = SimpleNamespace(vae=VAE())
        old = h3.vae.decode
        calls = {}
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            with probe.forbid_vae(h3, calls):
                h3.vae.decode(1)
        self.assertEqual(h3.vae.decode, old)
        self.assertEqual(calls, {"decode": 1})


class StdlibTests(unittest.TestCase):
    def test_pin_verification_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "code.py"
            path.write_text("reviewed\n")
            pins = {"code.py": probe.sha_file(path)}
            self.assertEqual(len(probe.verify_pins(Path(temp), pins)), 1)
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "pin mismatch"):
                probe.verify_pins(Path(temp), pins)

    def test_new_output_directory_required(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                probe.Evidence(temp, {})


if __name__ == "__main__":
    unittest.main()
