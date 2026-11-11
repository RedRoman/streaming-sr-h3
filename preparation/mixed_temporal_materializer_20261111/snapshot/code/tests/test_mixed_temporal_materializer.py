"""CPU-only contract tests. No real-source 96-frame smoke is executed here."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import materialize_mixed_temporal_pilot as m

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_CV2 = importlib.util.find_spec("cv2") is not None
HAVE_AV = importlib.util.find_spec("av") is not None
HAVE_PIL = importlib.util.find_spec("PIL") is not None


def fixture():
    plan = json.loads((ROOT / "configs/openh3_20261111/mixed_temporal_data_plan_20261111.json").read_text())
    contract = json.loads((ROOT / "preparation/mixed_temporal_materializer_20261111/execution_contract.json").read_text())
    return plan, contract


class Contracts(unittest.TestCase):
    def test_frozen_profiles_complete_and_ordered(self):
        plan, contract = fixture()
        m.validate_plan(plan, contract)
        altered = copy.deepcopy(plan)
        altered["profiles"][0]["frame_parameters"][11]["frame"] = 12
        with self.assertRaisesRegex(ValueError, "every frame"):
            m.validate_plan(altered, contract)
        altered = copy.deepcopy(plan)
        altered["profiles"][0]["ordered_operators"][:2] = reversed(altered["profiles"][0]["ordered_operators"][:2])
        with self.assertRaisesRegex(ValueError, "operator order"):
            m.validate_plan(altered, contract)

    def test_temporal_crf_and_unpaired_noise_rejected(self):
        plan, contract = fixture()
        plan["profiles"][2]["frame_parameters"][48]["h264_crf"] = 35
        with self.assertRaisesRegex(ValueError, "CRF"):
            m.validate_plan(plan, contract)
        plan, contract = fixture()
        plan["profiles"][3]["noise_stream"] = "wrong"
        with self.assertRaisesRegex(ValueError, "share unit noise"):
            m.validate_plan(plan, contract)

    def test_seed_format_and_window_binding(self):
        source, key = "a" * 64, "b" * 64
        raw = f"omni-mixed-v1|{source}|{key}|paired_native_static|000007".encode()
        self.assertEqual(m.noise_seed(source, key, "paired_native_static", 7), int.from_bytes(hashlib.sha256(raw).digest()[:8], "big"))
        self.assertNotEqual(m.noise_seed(source, key, "paired_native_static", 7), m.noise_seed(source, "c" * 64, "paired_native_static", 7))
        self.assertEqual(m.value_sha({"b": 2, "a": "视"}), m.value_sha({"a": "视", "b": 2}))
        with self.assertRaises(ValueError):
            m.noise_seed(source, key, "x", 7, width=3)

    def registry_files(self, directory, dev_id="dev", dev_sha="b" * 64):
        plan = {"source_roles": {}}
        paths = []
        for split, sid, sha, key in (("train", m.SOURCE, "a" * 64, "existing_train"), ("dev", dev_id, dev_sha, "existing_dev")):
            path = Path(directory) / (split + ".jsonl")
            path.write_text(json.dumps({"source_id": sid, "source_sha256": sha, "split": split}) + "\n")
            paths.append(path)
            plan["source_roles"][key] = {"manifest_sha256": m.file_sha(path), "source_ids": [sid]}
        return plan, paths

    def test_source_id_and_alias_hash_cross_split_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, paths = self.registry_files(directory)
            registry, binding = m.load_registry(*paths, plan)
            self.assertEqual(binding["source_id"], m.SOURCE)
            self.assertEqual(len(registry["sources"]), 2)
            plan, paths = self.registry_files(directory, dev_id=m.SOURCE)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                m.load_registry(*paths, plan)
            plan, paths = self.registry_files(directory, dev_sha="a" * 64)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                m.load_registry(*paths, plan)

    def test_registry_bytes_and_declared_membership_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, paths = self.registry_files(directory)
            paths[0].write_text(paths[0].read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                m.load_registry(*paths, plan)

    def test_exact_frame_byte_length_not_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixels.rgb"
            path.write_bytes(bytes(2 * 3 * 4 * 3))
            m.checked_shape(path, (2, 3, 4, 3))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                m.checked_shape(path, (2, 3, 4, 3))

    def test_timestamps_require_actual_source_and_output_frames(self):
        def text(source_frames=3):
            lines = [f"[showinfo@{tag} @ test] config in time_base: 1/24, frame_rate: 24/1" for tag in ("source", "target")]
            for tag, n in (("source", source_frames), ("target", 3)):
                lines += [f"[showinfo@{tag} @ test] n: {i} pts: {i} pts_time:{i / 24}" for i in range(n)]
            return "\n".join(lines)
        self.assertEqual(len(m.parse_timestamps(text(), frames=3)["target"]), 3)
        with self.assertRaisesRegex(ValueError, "actual source frames"):
            m.parse_timestamps(text(2), frames=3)

    def test_command_rgb_before_integer_crop_without_resize(self):
        spec = {"start_seconds": "2.5", "source_format_start_time": "0", "width": 832, "height": 480, "frames": 96}
        command, graph = m.target_command("input.mp4", spec, "target.rgb", "ffmpeg")
        self.assertLess(graph.index("format=pix_fmts=rgb24"), graph.index("crop="))
        self.assertIn("floor((iw-ow)/2)", graph)
        self.assertIn("exact=1", graph)
        self.assertNotIn("scale=", graph)
        self.assertIn("-copyts", command)
        self.assertLess(command.index("-ss"), command.index("-i"))


@unittest.skipUnless(HAVE_NUMPY, "NumPy unavailable")
class Numerical(unittest.TestCase):
    def test_lossless_npz_pixels_metadata_and_file_receipt(self):
        import numpy as np
        pixels = np.arange(2 * 5 * 7 * 3, dtype=np.uint8).reshape(2, 5, 7, 3)
        meta = {"role": "observation", "target_pixel_sha256": "a" * 64, "task": "mixed_native"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observation.npz"
            receipt = m.save_npz(path, pixels, meta)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["frames"], pixels)
                self.assertEqual(archive["frames"].dtype, np.uint8)
                self.assertEqual(json.loads(str(archive["metadata"])), meta)
            self.assertEqual(receipt["pixel_sha256"], m.pixel_sha(pixels))
            self.assertEqual(receipt["file_sha256"], m.file_sha(path))
            with self.assertRaises(FileExistsError):
                m.save_npz(path, pixels, meta)

    def test_noise_stream_shared_and_global_rng_untouched(self):
        import numpy as np
        np.random.seed(7)
        expected = np.random.random(3)
        np.random.seed(7)
        seed = m.noise_seed("a" * 64, "b" * 64, "native_temporal", 48)
        a = m.unit_noise((8, 8, 3), seed)
        b = m.unit_noise((8, 8, 3), seed)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.random.random(3), expected)
        self.assertEqual(a.dtype, np.float32)

    def test_ties_even_and_zero_blur_bypass(self):
        import numpy as np
        value = np.array([-1.5, -.5, .5, 1.5, 2.5, 255.5], dtype=np.float32)
        np.testing.assert_array_equal(m.quantize(value), [0, 0, 0, 2, 2, 255])
        self.assertIs(m.blur_float(value, 0), value)

    @unittest.skipUnless(HAVE_CV2 and HAVE_PIL, "OpenCV/Pillow unavailable")
    def test_order_controls_share_noise_but_change_result(self):
        import numpy as np
        plan, _ = fixture()
        clean = np.full((40, 48, 3), 128, dtype=np.uint8)
        baseline = clean.copy()
        a, ar = m.precodec_frame(clean, plan["profiles"][0], 0, "a" * 64, "b" * 64)
        b, br = m.precodec_frame(clean, plan["profiles"][1], 0, "a" * 64, "b" * 64)
        self.assertEqual(ar, br)
        self.assertFalse(np.array_equal(a, b))
        self.assertGreater(a.astype(float).var(), b.astype(float).var())
        np.testing.assert_array_equal(clean, baseline)

    @unittest.skipUnless(HAVE_CV2 and HAVE_PIL, "OpenCV/Pillow unavailable")
    def test_sr_fixed_geometry_and_smooth_abrupt_noise_pairing(self):
        import numpy as np
        plan, _ = fixture()
        clean = np.full((40, 48, 3), 128, dtype=np.uint8)
        sr, receipt = m.precodec_frame(clean, plan["profiles"][5], 24, "a" * 64, "b" * 64)
        self.assertEqual(sr.shape, (10, 12, 3))
        self.assertEqual(receipt["unit_noise_shape"], [10, 12, 3])
        _, ar = m.precodec_frame(clean, plan["profiles"][2], 47, "a" * 64, "b" * 64)
        _, br = m.precodec_frame(clean, plan["profiles"][3], 47, "a" * 64, "b" * 64)
        self.assertEqual(ar, br)

    @unittest.skipUnless(HAVE_CV2 and HAVE_PIL and HAVE_AV, "CPU codec dependencies unavailable")
    def test_small_codec_replay_and_no_scene_reset(self):
        import numpy as np
        plan, _ = fixture()
        # Eight 16x24 synthetic frames only. The hard content switch tests that
        # scenecut=0 remains distinct from a mid-clip degradation change.
        clean = np.full((8, 16, 24, 3), 40, dtype=np.uint8)
        clean[4:] = 210
        binding = {"source_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            for profile in (plan["profiles"][0], plan["profiles"][4]):
                receipts = []
                for replay in range(2):
                    obs, raw, receipt = m.observation(clean, profile, binding, "b" * 64, Path(directory), replay, m.Budget(60))
                    self.assertEqual(obs.shape, clean.shape)
                    receipts.append(receipt)
                    del obs
                    raw.unlink()
                self.assertEqual(receipts[0]["observation_pixel_sha256"], receipts[1]["observation_pixel_sha256"])
                self.assertEqual(receipts[0]["noise_frames"], receipts[1]["noise_frames"])
                if profile["profile_id"] != m.PROFILE_IDS[4]:
                    self.assertEqual(receipts[0]["keyframe_indices"], [0])

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_real_ffmpeg_timestamp_taps_and_floor_crop_on_tiny_synthetic(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "synthetic.nut"
            # Odd source dimensions exercise RGB floor-centering, not YUV
            # chroma-aligned cropping. No real project source is decoded.
            pixels = np.arange(12 * 19 * 25 * 3, dtype=np.uint8).reshape(12, 19, 25, 3)
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "25x19", "-r", "24", "-i", "pipe:0", "-c:v", "rawvideo", "-threads", "1", "-f", "nut", str(source)], input=pixels.tobytes(), check=True, timeout=20)
            for first, start in ((0, "0"), (3, "0.125")):
                spec = {"start_seconds": start, "source_format_start_time": "0", "width": 20, "height": 16, "frames": 8}
                raw, log = directory / f"crop{first}.rgb", directory / f"decode{first}.log"
                command, _ = m.target_command(source, spec, raw, "ffmpeg")
                m.run_owned(command, log, m.Budget(30))
                m.checked_shape(raw, (8, 16, 20, 3))
                result = np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(8, 16, 20, 3)
                np.testing.assert_array_equal(result, pixels[first:first + 8, 1:17, 2:22])
                timestamps = m.parse_timestamps(log.read_text(), frames=8)
                self.assertGreaterEqual(len(timestamps["source"]), 8)
                self.assertEqual(len(timestamps["target"]), 8)
                self.assertAlmostEqual(timestamps["source"][0]["pts_time_seconds"], float(start))
                self.assertEqual(timestamps["target"][0]["pts"], 0)


if __name__ == "__main__":
    unittest.main()
