"""Small CPU interface checks; real VAE/NPU is never imported here."""
import importlib.util
import hashlib
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from export_rphi_guides import array_sha
from run_vae_roundtrip_diagnostic import (main, validate_reference,
                                        verify_vae_binding, manifest_vae_binding)


class StaticContractTests(unittest.TestCase):
    def test_manifest_components_bind_when_isolated_defaults_are_missing_and_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {k: root / (k + ".bin") for k in ("vae", "context")}
            for name, path in paths.items():
                path.write_bytes(name.encode())
            records = {k: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for k, p in paths.items()}
            implementation = SimpleNamespace(VAE_PATH=root / "missing_vae", CONTEXT_PATH=root / "missing_context",
                SHA={k: v["sha256"] for k, v in records.items()}, SIZE={"vae": records["vae"]["bytes"]})
            manifest = {**records, "_paths": {k: str(p) for k, p in paths.items()}}
            verified = verify_vae_binding(manifest, implementation)
            self.assertFalse(verified["vae"]["original_default_exists"])
            old = implementation.VAE_PATH, implementation.CONTEXT_PATH
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with manifest_vae_binding(implementation, verified):
                    self.assertEqual(implementation.VAE_PATH, paths["vae"].resolve())
                    self.assertEqual(implementation.CONTEXT_PATH, paths["context"].resolve())
                    raise RuntimeError("simulated loader failure")
            self.assertEqual((implementation.VAE_PATH, implementation.CONTEXT_PATH), old)
            paths["vae"].write_bytes(b"bad")  # same byte count, different actual identity
            with self.assertRaisesRegex(ValueError, "actual SHA"):
                verify_vae_binding(manifest, implementation)

    def test_existing_output_rejected_before_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            args = sum((["--" + name, "missing"] for name in ("ordinary-metrics", "ordinary-guides",
                "controls-metrics", "controls-guides", "dev-manifest", "pretrained-manifest")), [])
            with self.assertRaises(FileExistsError):
                main(args + ["--out", temp])

    def test_incomplete_references_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "completed"):
            validate_reference({"complete": False}, {"complete": True}, "a" * 64, "ordinary")


@unittest.skipUnless(importlib.util.find_spec("torch"), "existing CPU torch required")
class RoundtripTests(unittest.TestCase):
    def fake(self):
        import torch
        class Stage:
            pixel_mean = torch.full((1, 3, 1, 1, 1), .5)
            pixel_std = torch.full((1, 3, 1, 1, 1), .25)
            def _finalize_pixels(self, value):
                return (value * self.pixel_std + self.pixel_mean).clamp_(0, 1)
        class Fake:
            def __init__(self):
                self.vae = type("VAE", (), {"first_stage_model": Stage()})()
                self.encodes, self.decodes = 0, 0
            def encode_guide_pixels(self, rgb, hw):
                self.encodes += 1
                return torch.full((1, 24, 7, hw[0] // 16, hw[1] // 16), float(rgb.mean()))
            def decode(self, latent):
                self.decodes += 1
                # Two already-blended chunks, with trailing frame padding.
                first = torch.full((1, 3, 12, 32, 32), -3.)
                last = torch.full((1, 3, 12, 32, 32), 3.)
                f = self.vae.first_stage_model._finalize_pixels
                return torch.cat([f(first), f(last)], dim=2)[0, :, :22].permute(1, 2, 3, 0).contiguous()
        return Fake()

    def guides_and_reference(self, fake):
        from run_fixed_h3_rphi import tensor_sha
        import torch
        raw = np.full((22, 32, 32, 3), .2, np.float32)
        restored = np.full(raw.shape, .7, np.float32)
        ref = {"sampling": {"guide_pixel_sha256": {}, "calls": {}}}
        for name, pixels in (("raw", raw), ("rphi", restored)):
            ref["sampling"]["guide_pixel_sha256"][name] = array_sha(pixels)
            latent = fake.encode_guide_pixels(torch.from_numpy(pixels.copy()), [32, 32])
            ref["sampling"]["calls"][name] = {"guide_latent_sha256_float32": tensor_sha(latent)}
        fake.encodes = 0
        return raw, restored, ref

    def test_two_guide_paths_capture_unclipped_and_preserve_original_return(self):
        from run_vae_roundtrip_diagnostic import roundtrip_pair
        fake = self.fake()
        raw, restored, ref = self.guides_and_reference(fake)
        original_method = fake.vae.first_stage_model._finalize_pixels.__func__
        outputs, record = roundtrip_pair(fake, raw, restored, ref)
        self.assertEqual((fake.encodes, fake.decodes), (2, 2))
        self.assertIs(fake.vae.first_stage_model._finalize_pixels.__func__, original_method)
        self.assertNotIn("_finalize_pixels", fake.vae.first_stage_model.__dict__)
        for name in ("raw", "rphi"):
            before, after = outputs[name + "_pre_internal_clamp"], outputs[name + "_as_returned_internal_clipped"]
            self.assertEqual(before.shape, (22, 32, 32, 3))
            self.assertEqual(before.dtype, np.float32)
            self.assertLess(before.min(), 0)
            self.assertGreater(before.max(), 1)
            self.assertTrue(np.array_equal(before.clip(0, 1), after))
            self.assertEqual(record[name]["captured_frames_before_final_crop"], 24)
            self.assertEqual(record[name]["clamp_parity_max_abs_error"], 0)

    def test_reference_latent_mismatch_stops_before_decode(self):
        from run_vae_roundtrip_diagnostic import roundtrip_pair
        fake = self.fake()
        raw, restored, ref = self.guides_and_reference(fake)
        ref["sampling"]["calls"]["raw"]["guide_latent_sha256_float32"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "encoded latent SHA"):
            roundtrip_pair(fake, raw, restored, ref)
        self.assertEqual(fake.decodes, 0)

    def test_capture_mismatch_and_exception_restore_hook(self):
        from run_vae_roundtrip_diagnostic import decode_with_preclip_capture
        import torch
        fake = self.fake()
        original = fake.decode
        fake.decode = lambda latent: original(latent) + .001
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            decode_with_preclip_capture(fake, torch.zeros(1), (22, 32, 32, 3))
        self.assertNotIn("_finalize_pixels", fake.vae.first_stage_model.__dict__)
        def fail(latent):
            raise RuntimeError("decode failed")
        fake.decode = fail
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            decode_with_preclip_capture(fake, torch.zeros(1), (22, 32, 32, 3))
        self.assertNotIn("_finalize_pixels", fake.vae.first_stage_model.__dict__)

    def test_metric_target_does_not_enter_roundtrip(self):
        from run_vae_roundtrip_diagnostic import roundtrip_pair, score_outputs
        from eval_rphi import rgb_metrics
        fake = self.fake()
        raw, restored, ref = self.guides_and_reference(fake)
        outputs, _ = roundtrip_pair(fake, raw, restored, ref)
        cache_hashes = {k: array_sha(v) for k, v in outputs.items()}
        losses = []
        for value in (0, 255):
            target = np.full(raw.shape, value, np.uint8)
            gt = target.astype(np.float32) / 255
            prior = {"arms": {"input": {"quality": rgb_metrics(raw, gt)},
                              "rphi_only": {"quality": rgb_metrics(restored, gt)}}}
            for label in ("raw", "rphi"):
                for suffix in ("", "_anchor"):
                    prior["arms"]["h3_" + label + suffix] = {"quality": rgb_metrics(raw, gt)}
            score = score_outputs(outputs, raw, restored, target, prior, require_flow=False)
            losses.append(score["arms"]["raw_pre_internal_clamp"]["quality"]["mae"])
        self.assertNotEqual(losses[0], losses[1])
        self.assertEqual(cache_hashes, {k: array_sha(v) for k, v in outputs.items()})
        self.assertEqual(fake.decodes, 2)


if __name__ == "__main__":
    unittest.main()
