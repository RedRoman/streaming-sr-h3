import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from audit_arrays import array_sha, file_sha, pil_same_geometry_hashes, verified_npy, verify_clipped_rphi
from small_archive import package


class ArrayArchiveTests(unittest.TestCase):
    def test_npy_actual_file_and_content_hashes_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latent.npy"
            value = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
            np.save(path, value)
            row = {"file_sha256": file_sha(path), "array_sha256": array_sha(value),
                   "shape": list(value.shape), "dtype": "float32"}
            result, facts = verified_npy(path, row, value.shape)
            self.assertTrue(np.array_equal(result, value)); self.assertTrue(facts["finite"])
            with self.assertRaisesRegex(ValueError, "content SHA"):
                verified_npy(path, {**row, "array_sha256": "0" * 64}, value.shape)
            changed = value.copy(); changed.flat[1] += 1; np.save(path, changed)
            with self.assertRaisesRegex(ValueError, "file SHA"):
                verified_npy(path, row, value.shape)

    def test_pil_quantization_is_floor_not_round_and_rphi_clamp_exact(self):
        raw = np.array([[[[-.2, .5, 1.2], [.499, .999, .1]]]], np.float32)
        clipped = np.clip(raw, 0, 1)
        verify_clipped_rphi(raw, clipped)
        hashes = pil_same_geometry_hashes(clipped)
        u8 = np.array([[[[0, 127, 255], [127, 254, 25]]]], np.uint8)
        self.assertEqual(hashes["pil_uint8_sha256"], array_sha(u8))
        self.assertEqual(hashes["after_float32_sha256"], array_sha(u8.astype(np.float32) / np.float32(255)))
        changed = clipped.copy(); changed.flat[1] += .01
        with self.assertRaisesRegex(ValueError, "exact clipped"):
            verify_clipped_rphi(raw, changed)
        with self.assertRaisesRegex(ValueError, "clipped RGB"):
            pil_same_geometry_hashes(raw)

    def test_nonfinite_or_wrong_shape_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npy"
            array = np.array([np.nan], np.float32); np.save(path, array)
            row = {"file_sha256": file_sha(path), "array_sha256": array_sha(array), "dtype": "float32", "shape": [1]}
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                verified_npy(path, row, [1])
            with self.assertRaisesRegex(ValueError, "shape"):
                verified_npy(path, row, [2])

    def test_archive_keeps_rgb_remote_and_locks_admitted_latents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep, own, run, out = [root / x for x in ("prep", "own", "run", "out")]
            for path in (prep, own, run, out, run / "eval", run / "eval/arrays"):
                path.mkdir()
            (prep / "code").mkdir()
            producer_sources = {"export_train_vae_cache_portable.py", "export_train_vae_cache.py", "openh3_vae_only_assets.py"}
            for name in producer_sources:
                (prep / "code" / name).write_text("# source fixture\n")
            arrays = {}
            for role in ("Z_raw", "Z_R", "VAE_R"):
                path = run / "eval/arrays" / (role + ".npy")
                np.save(path, np.zeros([1, 24, 7, 30, 52], np.float32) if role != "VAE_R" else np.zeros([3], np.float32))
                if role != "VAE_R":
                    arrays[role] = {"path": str(path), "file_sha256": file_sha(path)}
            audit = {"archive_arrays": arrays}
            (out / "terminal_audit.json").write_text(json.dumps(audit))
            result = package(root, prep, run, own, out, audit)
            with tarfile.open(result["path"]) as tar:
                names = tar.getnames()
                self.assertFalse(any("VAE_R" in name for name in names))
                self.assertEqual(sum(name.endswith(".npy") for name in names), 2)
                self.assertTrue(all("prep/code/" + name in names for name in producer_sources))
                manifest = json.load(tar.extractfile("MANIFEST.json"))
                self.assertFalse(manifest["large_RGB_included"])
            with self.assertRaises(FileExistsError):
                package(root, prep, run, own, out, audit)
            np.save(Path(arrays["Z_raw"]["path"]), np.ones([1, 24, 7, 30, 52], np.float32))
            with self.assertRaisesRegex(ValueError, "changed after"):
                package(root, prep, run, own, out, audit)


if __name__ == "__main__":
    unittest.main()
