"""CPU tests only. Synthetic NPZ fixtures are not project training artifacts."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import history_utility_data as D

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/openh3_20261111/mixed_temporal_data_plan_20261111.json"
TRAIN = ROOT / "results/takeover_20261111/materialized/train_materialized.jsonl"
DEV = ROOT / "results/takeover_20261111/materialized/dev_materialized.jsonl"
REAL_RECEIPT = ROOT / "results/takeover_20261111/mixed_temporal_single_source_smoke_completed/complete.json"


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


class HistoryUtilityDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.receipt = copy.deepcopy(D._json(REAL_RECEIPT))
        cls.plan = D._json(PLAN)
        cls.sid = cls.receipt["source"]["source_id"]
        # Full required shape, deterministic frame/x/y/channel pattern. No model.
        frame = np.indices((480, 832)).sum(0).astype(np.uint8)
        pixels = (np.arange(96, dtype=np.uint8)[:, None, None, None]
                  + frame[None, :, :, None] + np.arange(3, dtype=np.uint8)[None, None, None, :])
        common = {"schema": D.MATERIALIZER_SCHEMA, **cls.receipt["source"],
                  "window_key": cls.receipt["window_key"], "target_pixel_sha256": D.array_sha(pixels),
                  **{k: cls.receipt[k] for k in ("plan_file_sha256", "registry_sha256", "materializer_sha256", "software_sha256")}}

        def save(name, array, metadata):
            path = cls.directory / (name + ".npz")
            np.savez_compressed(path, frames=array, metadata=D.canonical(metadata))
            return {"path": str(path), "bytes_on_disk": path.stat().st_size, "file_sha256": D.file_sha(path),
                    "pixel_sha256": D.array_sha(array), "shape": list(array.shape), "dtype": "uint8"}
        cls.receipt["target"] = save("target", pixels, {**common, "role": "target", "target_specification": cls.receipt["target_specification"]})
        profiles = {p["profile_id"]: p for p in cls.plan["profiles"]}
        for i, pid in enumerate(D.PROFILES):
            observed = pixels + np.uint8(11 + i)
            profile = profiles[pid]
            artifact = save(pid, observed, {**common, "role": "observation", "profile": profile,
                "parameter_schedule_sha256": D.value_sha(profile["frame_parameters"]), "noise_stream": profile["noise_stream"],
                "blind_model_condition": "not_supplied_new_loader_required"})
            cls.receipt["observations"][pid]["artifact"] = artifact
            for r in cls.receipt["observations"][pid]["replays"]:
                r["observation_pixel_sha256"] = artifact["pixel_sha256"]
        cls.path = cls.directory / "synthetic_fixture_complete.json"
        write_json(cls.path, cls.receipt)
        cls.entry = {"path": str(cls.path), "sha256": D.file_sha(cls.path), "artifact_root": str(cls.directory)}
        cls.data = cls.dataset()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def dataset(cls, entry=None, **kwargs):
        return D.HistoryUtilityDataset([entry or cls.entry], plan=PLAN, train_manifest=TRAIN,
            dev_manifest=DEV, split="train", **kwargs)

    def row(self, frame=2, profile=None, crop=None, external=None, mask=None):
        return self.data._row(self.sid, profile or D.PROFILES[0], frame,
            crop or [4, 8, 96, 96], external or [1, 1], mask, 0)

    def test_actual_receipt_metadata_only_and_research_gate(self):
        entry = {"path": str(REAL_RECEIPT), "sha256": D.file_sha(REAL_RECEIPT)}
        data = self.dataset(entry, verify_pixels=False)
        self.assertEqual(data.source_ids, ["2BdYaGuVAKo"])
        self.assertFalse(data.research_ready)
        self.assertFalse(data.binding["pixels_verified"])
        with self.assertRaisesRegex(ValueError, "metadata-only"):
            data.schedule(steps=1)
        with self.assertRaisesRegex(ValueError, "research needs"):
            self.dataset(entry, scope="research", verify_pixels=False)

    def test_full_npz_causal_availability_shared_crop_and_no_oracle_inputs(self):
        for t, expected in ((0, [0, 0]), (1, [0, 1]), (2, [1, 1]), (95, [1, 1])):
            out = self.data.sample(self.row(t))
            self.assertEqual(set(out["model_inputs"]), {"observed", "availability"})
            x = out["model_inputs"]["observed"]
            self.assertEqual(x.shape, (3, 3, 96, 96)); self.assertEqual(x.dtype, np.float32)
            np.testing.assert_array_equal(out["model_inputs"]["availability"], expected)
            for j, frame in enumerate(D.history_indices(t)):
                self.assertAlmostEqual(float(x[j, 0, 0, 0]), ((frame + 4 + 8 + 11) % 256) / 255, places=7)
            if t == 0: np.testing.assert_array_equal(x[0], x[2])
            if t == 1: np.testing.assert_array_equal(x[0], x[1])
            self.assertAlmostEqual(float(out["target"][0, 0, 0]), ((t + 4 + 8) % 256) / 255, places=7)
        changed = self.row(); changed["history_indices"] = [1, 2, 3]
        with self.assertRaises(ValueError): self.data.sample(changed)
        changed = self.row(); changed["condition"] = [1] * 6
        with self.assertRaises(ValueError): self.data.sample(changed)

    def test_target_is_supervision_not_ordinary_input(self):
        row = self.row(); original = self.data.sample(row); original_pixels = self.data._pixels
        def changed_target(sid, name):
            result = original_pixels(sid, name)
            return result + np.uint8(1) if name == "target" else result
        # Test the sampling call boundary, not a bypass offered by the loader.
        with mock.patch.object(self.data, "_pixels", side_effect=changed_target):
            changed = self.data.sample(row)
        np.testing.assert_array_equal(original["model_inputs"]["observed"], changed["model_inputs"]["observed"])
        self.assertFalse(np.array_equal(original["target"], changed["target"]))

    def test_near_clean_overlap_padding_and_control_identity(self):
        a = self.data.sample(self.row(2, "near_clean"))
        b = self.data.sample(self.row(3, "near_clean"))
        np.testing.assert_array_equal(a["model_inputs"]["observed"][1:], b["model_inputs"]["observed"][:2])
        padded = self.data.sample(self.row(0, "near_clean"))["model_inputs"]["observed"]
        np.testing.assert_array_equal(padded[0], padded[2])
        self.assertEqual(a["metadata"]["observation_pixel_sha256"], b["metadata"]["observation_pixel_sha256"])
        identity = self.data.sample(self.row(3, "identity"))
        np.testing.assert_array_equal(identity["model_inputs"]["observed"][-1], identity["target"])
        shifted = self.data.sample(self.row(2, "near_clean", crop=[4, 12, 96, 96]))
        np.testing.assert_array_equal(a["model_inputs"]["observed"][..., 4:], shifted["model_inputs"]["observed"][..., :-4])

    def test_schedule_exact_replay_composition_masks_and_tamper_rejection(self):
        schedule = self.data.schedule("student", steps=2)
        self.assertEqual(schedule, self.data.schedule("student", steps=2))
        for start in (0, 16):
            profiles = [r["profile_id"] for r in schedule["rows"][start:start+16]]
            for p in D.PROFILES: self.assertEqual(profiles.count(p), 4)
            self.assertEqual(profiles.count("identity"), 2); self.assertEqual(profiles.count("near_clean"), 2)
        self.data.validate_schedule(schedule)
        bad = copy.deepcopy(schedule); bad["rows"][0]["frame_index"] = 95
        with self.assertRaises(ValueError): self.data.validate_schedule(bad)
        teacher = self.data.schedule("teacher", steps=2)
        self.assertNotEqual(teacher["schedule_sha256"], schedule["schedule_sha256"])
        batch = self.data.batch(teacher["rows"][:2])
        self.assertEqual(batch["teacher_weights"].shape, (2, 2, 96, 96))
        self.assertEqual(batch["model_inputs"]["availability"].shape, (2, 2))
        self.assertEqual(batch["control"].dtype, np.int64)

    def test_teacher_local_masks_are_bounded_deterministic_and_boundary_aware(self):
        for name in ("all_available", "oldest_missing", "recent_missing", "both_missing", "local_binary", "local_soft"):
            spec = {"kind": name, "seed": 7}
            a = D.teacher_weights(spec, [0, 1]); b = D.teacher_weights(spec, [0, 1])
            np.testing.assert_array_equal(a, b)
            self.assertTrue(np.all(a[0] == 0)); self.assertTrue(np.all((a >= 0) & (a <= 1)))
            self.assertEqual(a.dtype, np.float32)
        coarse = np.arange(72, dtype=np.float32).reshape(2, 6, 6) / 72
        up = D._bilinear_16(coarse)
        np.testing.assert_array_equal(up[:, 0, 0], coarse[:, 0, 0])
        np.testing.assert_array_equal(up[:, -1, -1], coarse[:, -1, -1])

    def test_strict_source_role_alias_overlap_and_receipt_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = copy.deepcopy(self.plan)
            rows = [json.loads(s) for s in DEV.read_text().splitlines() if s.strip()]
            trainrow = json.loads(TRAIN.read_text().splitlines()[0])
            sid = rows[0]["source_id"]
            for r in rows:
                if r["source_id"] == sid: r["source_sha256"] = trainrow["source_sha256"]
            dev = tmp / "dev.jsonl"; dev.write_text("\n".join(json.dumps(r) for r in rows))
            plan["source_roles"]["existing_dev"]["manifest_sha256"] = D.file_sha(dev)
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                D.load_source_registry(plan, TRAIN, dev)
            for field in ("status", "window_key"):
                receipt = copy.deepcopy(self.receipt); receipt[field] = "corrupt"
                path = tmp / (field + ".json"); write_json(path, receipt)
                with self.assertRaises(ValueError):
                    self.dataset({"path": str(path), "sha256": D.file_sha(path)}, verify_pixels=False)
            receipt = copy.deepcopy(self.receipt)
            receipt["observations"][D.PROFILES[0]]["replays"][0]["keyframe_indices"] = [0, 48]
            path = tmp / "reset.json"; write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, "codec reset"):
                self.dataset({"path": str(path), "sha256": D.file_sha(path)}, verify_pixels=False)

    def test_actual_npz_hash_failure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); path = tmp / "target.npz"
            shutil.copyfile(self.directory / "target.npz", path)
            with path.open("ab") as f: f.write(b"tamper")
            entry = {**self.entry, "artifact_root": str(tmp)}
            with self.assertRaisesRegex(ValueError, "NPZ bytes/SHA"):
                self.dataset(entry)
            out = tmp / "existing.json"; out.write_text("preserve")
            argv = ["data", "--receipts-index", "unused", "--plan", str(PLAN), "--train-manifest", str(TRAIN),
                    "--dev-manifest", str(DEV), "--split", "train", "--out", str(out)]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(FileExistsError): D.main()
            self.assertEqual(out.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
