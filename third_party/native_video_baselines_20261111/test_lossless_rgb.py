"""CPU transport checks only; these do not validate either neural network."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_lossless_rgb import read_observation


class ObservationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "observation.npz"
        self.frames = np.random.default_rng(3).integers(0, 256, (2, 256, 264, 3), dtype=np.uint8)
        self.pixel_sha = hashlib.sha256(self.frames.tobytes(order="C")).hexdigest()

    def test_lossless_rgb_round_trip_preserves_all_channels(self):
        # The object payload must never be accessed or unpickled by the runner.
        np.savez_compressed(self.path, frames=self.frames, target=np.array({"not": "an input"}, dtype=object))
        actual, checksum = read_observation(self.path, self.pixel_sha)
        np.testing.assert_array_equal(actual, self.frames)
        self.assertEqual(checksum, self.pixel_sha)

    def test_pixel_hash_mismatch_rejected(self):
        np.savez_compressed(self.path, frames=self.frames)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            read_observation(self.path, "0" * 64)

    def test_float_or_transposed_input_rejected(self):
        for frames in (self.frames.astype(np.float32), self.frames.transpose(0, 3, 1, 2)):
            with self.subTest(shape=frames.shape, dtype=frames.dtype):
                np.savez_compressed(self.path, frames=frames)
                with self.assertRaisesRegex(ValueError, "uint8 RGB"):
                    read_observation(self.path, self.pixel_sha)

    def test_silent_resize_or_temporal_truncation_rejected(self):
        for frames in (self.frames[:, :128], self.frames[:1], np.repeat(self.frames[:1], 101, axis=0)):
            with self.subTest(shape=frames.shape):
                np.savez_compressed(self.path, frames=frames)
                with self.assertRaises(ValueError):
                    read_observation(self.path, self.pixel_sha)


if __name__ == "__main__":
    unittest.main()
