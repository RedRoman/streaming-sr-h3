"""CPU-only materializer tests; synthetic pixels verify contracts, not quality."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from materialize_omni_pilot import degrade_pixels, parse_plans, pixels_from_raw, verify_group, write_pixels, VERSION
from prepare_omni_pilot import make_plan
from test_takeover_data import fixture


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.random.default_rng(10).integers(0, 256, (3, 32, 48, 3), dtype=np.uint8)
        self.rows, _ = make_plan(fixture(), train_sources=1, dev_sources=1, frames=3, height=32, width=48)

    def test_rejects_wrong_frame_count(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            pixels_from_raw(b"short", (3, 32, 48, 3))

    def test_geometry_determinism_and_target_immutability(self):
        original = self.clean.copy()
        with tempfile.TemporaryDirectory() as directory:
            for task in ("sr", "denoise", "deblur"):
                row = next(r for r in self.rows if r["task"] == task)
                a, _ = degrade_pixels(self.clean, row, "unused", 1, directory)
                b, _ = degrade_pixels(self.clean, row, "unused", 1, directory)
                self.assertTrue(np.array_equal(a, b))
                self.assertEqual(a.shape, (3, 8, 12, 3) if task == "sr" else self.clean.shape)
                self.assertTrue(np.array_equal(original, self.clean))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_compression_roundtrip_exact_frame_count(self):
        row = next(r for r in self.rows if r["task"] == "compression")
        with tempfile.TemporaryDirectory() as directory:
            observed, meta = degrade_pixels(self.clean, row, shutil.which("ffmpeg"), 1, directory)
        self.assertEqual(observed.shape, self.clean.shape)
        self.assertEqual(meta["pixel_format"], "yuv420p")

    def test_plan_rejects_different_target_and_incomplete_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "plan.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in self.rows[:-1]))
            with self.assertRaisesRegex(ValueError, "all four"):
                parse_plans([p])
            rows = copy.deepcopy(self.rows)
            rows[0]["target"]["start_s"] += 1
            p.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "target key"):
                parse_plans([p])

    def test_resume_detects_output_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "target.npz"
            rec = write_pixels(p, self.clean, {"source_sha256": "src", "role": "clean_target"})
            manifest = {"plan_sha256": "plan", "source_sha256": "src", "materializer_version": VERSION,
                        "status": "materialized", "target": rec, "observations": {}}
            verify_group(manifest, "plan", "src")
            p.write_bytes(p.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                verify_group(manifest, "plan", "src")


if __name__ == "__main__":
    unittest.main()
