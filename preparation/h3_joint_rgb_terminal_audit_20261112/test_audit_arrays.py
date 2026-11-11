"""Focused real-array integrity/math fixtures, without Torch or a model."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import audit_arrays as audit


class ArrayTests(unittest.TestCase):
    def test_strict_latent_and_declared_rgb_parity_are_distinct(self):
        a = np.ones((2, 3), np.float32); b = a.copy(); b[0,0] += np.float32(.0001)
        self.assertTrue(audit.compare_outputs(b, a, exact=False)["allclose"])
        with self.assertRaisesRegex(ValueError, "strict"):
            audit.compare_outputs(b, a, exact=True)
        b[0,0] += np.float32(.01)
        with self.assertRaisesRegex(ValueError, "allclose"):
            audit.compare_outputs(b, a, exact=False)

    def test_rgb_target_mse_recomputed_and_wrong_target_rejected(self):
        target = np.linspace(.2, .8, 128, dtype=np.float32)[None,None,:,None]
        rgb = np.broadcast_to(target, (5,128,128,3)).copy()
        self.assertEqual(audit.recompute_rgb_mse(rgb, target, audit.array_sha(target)), 0.)
        rgb[:] = np.float32(.5)
        expected = float(np.mean((rgb.astype(np.float64) - target.astype(np.float64))**2))
        self.assertEqual(audit.recompute_rgb_mse(rgb, target, audit.array_sha(target)), expected)
        wrong = target.copy(); wrong[0,0,0,0] += np.float32(.1)
        with self.assertRaisesRegex(ValueError, "target SHA"):
            audit.recompute_rgb_mse(rgb, wrong, audit.array_sha(target))

    def test_saved_file_hash_and_actual_cast(self):
        value = np.arange(24, dtype=np.float32).reshape(2,3,4) / np.float32(7)
        audit.verify_decoder_cast(value, value.astype(np.float16))
        wrong = value.astype(np.float16); wrong.flat[1] += np.float16(.25)
        with self.assertRaisesRegex(ValueError, "cast"):
            audit.verify_decoder_cast(value, wrong)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"array.npy"; np.save(p,value)
            file_sha = audit.file_sha(p)
            self.assertTrue(np.array_equal(value, audit.verified_array(p,file_sha,audit.array_sha(value),value.shape,"float32")))
            with p.open("r+b") as f: f.seek(-1,2); f.write(b"X")
            with self.assertRaisesRegex(ValueError,"file SHA"):
                audit.verified_array(p,file_sha,audit.array_sha(value),value.shape,"float32")

    def test_gradient_unscale_requires_actual_bytes_and_nonzero_support(self):
        raw = np.arange(1,17,dtype=np.float32).reshape(4,4)
        unscaled = raw / np.float32(1024.)
        self.assertEqual(audit.verify_unscale(raw, unscaled, nonzero=True)["elements"],16)
        wrong = unscaled.copy(); wrong.flat[0] += np.float32(1e-8)
        with self.assertRaisesRegex(ValueError,"bytes"):
            audit.verify_unscale(raw,wrong)
        with self.assertRaisesRegex(ValueError,"zero"):
            audit.verify_unscale(np.zeros_like(raw),np.zeros_like(raw),nonzero=True)


if __name__ == "__main__":
    unittest.main()
