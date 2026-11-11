"""CPU-only checks for source isolation and honest pilot target provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from prepare_omni_pilot import existing_latent_manifests, make_plan
from takeover_data_audit import audit, read_jsonl, source_id


def fixture():
    candidates = []
    for ident, split in (("train_a", "train"), ("train_b", "train"), ("dev_a", "val"), ("seen", "train")):
        candidates.append({"source_id": ident, "source_video": "/data/" + ident + ".mp4",
                           "available": True, "historical_split": split, "historical_eval": False,
                           "source_kind": "captured_video", "source_group": "source:" + ident,
                           "clean_reference": "original_recording_proxy_not_pristine_truth",
                           "width_height": [1920, 1080], "duration_s": 10.0})
    return {"schema": "openh3.takeover_data_audit.v1", "real_candidates": candidates,
            "evaluations": [{"source_ids": ["seen"], "source_video_paths": {}}],
            "inventory": {"stage7_latents_train": {"samples": [
                {"source_id": x, "latent_path": "/data/" + x + "_w0.safetensors"}
                for x in ("train_a", "train_b", "seen")]},
                "stage7_latents_val": {"samples": [{"source_id": "dev_a", "latent_path": "/data/dev_a_w0.safetensors"}]}}}


class PilotTests(unittest.TestCase):
    def test_targets_shared_and_non_sr_native(self):
        rows, summary = make_plan(fixture(), train_sources=2, dev_sources=1)
        self.assertEqual(len(rows), 12)
        self.assertNotIn("seen", {r["source_id"] for r in rows})
        for ident in {r["source_id"] for r in rows}:
            chosen = [r for r in rows if r["source_id"] == ident]
            self.assertEqual(len({r["target_key"] for r in chosen}), 1)
            self.assertEqual({r["task"] for r in chosen}, {"sr", "denoise", "deblur", "compression"})
            for row in chosen:
                self.assertEqual(row["target"]["pixel_target"], "clean_original_recording")
                if row["task"] != "sr":
                    self.assertEqual(row["degradation"]["observation_resolution"], "target")
        self.assertEqual(summary["frozen_test"]["samples"], 0)
        self.assertFalse(summary["training_started"])

    def test_plan_deterministic_and_order_independent(self):
        data = fixture()
        original = make_plan(data, train_sources=2, dev_sources=1)
        data["real_candidates"].reverse()
        self.assertEqual(original, make_plan(data, train_sources=2, dev_sources=1))

    def test_excludes_source_path_aliases(self):
        data = fixture()
        data["evaluations"][0]["source_video_paths"] = {"another_id": "/data/train_a.mp4"}
        rows, _ = make_plan(data, train_sources=1, dev_sources=1)
        self.assertEqual({r["source_id"] for r in rows}, {"train_b", "dev_a"})

    def test_disjoint_ids_are_not_enough(self):
        data = fixture()
        data["real_candidates"][2]["source_group"] = data["real_candidates"][0]["source_group"]
        with self.assertRaisesRegex(ValueError, "source_group overlap"):
            make_plan(data, train_sources=2, dev_sources=1)

    def test_legacy_manifest_excludes_seen_and_labels_fallback(self):
        data = fixture()
        _, summary = make_plan(data, train_sources=2, dev_sources=1)
        manifests = existing_latent_manifests(data, summary)
        self.assertEqual(len(manifests["train"]["samples"]), 2)
        self.assertNotIn("seen", {s["source_id"] for s in manifests["train"]["samples"]})
        self.assertEqual(manifests["train"]["guide_key"], "guide_raw")
        self.assertFalse(manifests["train"]["pretrained_latent_upsampler_used"])

    def test_does_not_silently_reduce_requested_sources(self):
        with self.assertRaisesRegex(ValueError, "only 2 available"):
            make_plan(fixture(), train_sources=32, dev_sources=1)


class AuditTests(unittest.TestCase):
    def test_window_names(self):
        self.assertEqual(source_id("abc__w10"), "abc")
        self.assertEqual(source_id("abc_w2"), "abc")
        self.assertEqual(source_id("abc_123"), "abc_123")

    def test_manifest_corruption_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "manifest.jsonl"
            p.write_text('{"id":"good"}\ntruncated\n{"no_id":true}\n')
            rows, errors = read_jsonl(p)
            self.assertEqual([r["id"] for r in rows], ["good"])
            self.assertEqual([e["line"] for e in errors], [2, 3])

    def test_audit_verifies_same_resolved_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "data/real_av"
            (real / "_cand").mkdir(parents=True)
            video = real / "_cand/same.mp4"
            video.touch()
            (real / "candidates_big.json").write_text(json.dumps({"selected": [
                {"file": "same.mp4", "wh": [1920, 1080], "duration": 10}]}))
            (root / "data/stage7_train/train").mkdir(parents=True)
            (root / "data/stage7_train/train/same_w0.npz").touch()
            evaluation = root / "data/real_eval/heldout"
            (evaluation / "same").mkdir(parents=True)
            (evaluation / "manifest.json").write_text(json.dumps({"clips": ["same"]}))
            (evaluation / "same/meta.json").write_text(json.dumps({"video": str(video)}))
            result = audit(root, root / "nonexistent")
            self.assertTrue(result["same_source_path_evidence"][0]["same_resolved_source_path"])
            self.assertEqual(result["eligible_real_sources_after_all_historical_eval_exclusions"], {})
            self.assertFalse(result["frozen_test"]["available"])


if __name__ == "__main__":
    unittest.main()
