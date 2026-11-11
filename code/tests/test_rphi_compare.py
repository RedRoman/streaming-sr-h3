"""Matched-comparison contract tests; NumPy only, no model or NPU load.

Run: python -m unittest discover -s code/tests -p 'test_rphi_compare.py' -v
"""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_rphi import assert_matched, paired
from eval_rphi import TASKS


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def evaluation(variant="blind"):
    initial = sha("same initial backbone")
    result = {
        "schema": "rphi.rgb_eval.v1", "complete": True,
        "scope": "full_development_set", "dev_manifest_sha256": sha("dev manifest"),
        "source_ids": ["source-a", "source-b"], "frame_stride": 4,
        "metric_contract": {"rgb": "clipped RGB", "aggregation": "equal sources"},
        "training_contract": {"steps": 1000, "seed": 7, "initial_backbone_sha256": initial},
        "initial_backbone_sha256": initial, "checkpoint_step": 1000,
        "fidelity_state_sha256": sha("same trained fidelity weights"),
        "condition_mode": "known" if variant == "known" else "observations_only",
        "model_config": {"variant": variant, "k": 3, "ch": 8, "blocks": 1, "condition_dim": 6},
        "lpips": {"available": True, "alexnet_sha256": sha("LPIPS weights")},
        "per_sample": [], "calibration_per_task": {}}
    # Three windows for A, one for B: an accidental window-weighted comparison
    # gives a different answer from the required equal source weighting.
    for source, windows in (("source-a", 3), ("source-b", 1)):
        for window in range(windows):
            target = f"{source}-window{window}"
            for task in TASKS:
                metrics = {"psnr": 20., "mse": .04, "mae": .15, "lpips": .2}
                temporal = {"unwarped_rgb_error_change_mae": .05,
                            "flow_aligned_rgb_error_change_mae": .04}
                control = {"model": dict(metrics), "temporal_model": dict(temporal)}
                result["per_sample"].append({
                    "sample_id": f"{target}-{task}", "source_id": source,
                    "source_sha256": sha(source), "target_key": target, "task": task,
                    "target_pixel_sha256": sha(target),
                    "observation_pixel_sha256": sha(target + task),
                    "record_sha256": sha("record-" + target + task),
                    "frame_indices": [0, 4, 8], "model": metrics, "temporal_model": temporal,
                    "controls": {"identity": copy.deepcopy(control),
                                 "near_clean": {**copy.deepcopy(control), "seed": 1234}}})
    return result


class RphiComparisonTests(unittest.TestCase):
    def test_source_input_target_hashes_and_control_seeds_must_match(self):
        reference = evaluation()
        self.assertEqual(len(assert_matched(reference, copy.deepcopy(reference))[0]), 16)
        for field in ("source_id", "source_sha256", "target_key", "task",
                      "target_pixel_sha256", "observation_pixel_sha256", "record_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(reference)
                changed["per_sample"][0][field] = "changed-input"
                with self.assertRaisesRegex(ValueError, "input/target sample mismatch"):
                    assert_matched(changed, reference)
        changed = copy.deepcopy(reference)
        changed["source_ids"] = ["source-a", "different-source"]
        with self.assertRaisesRegex(ValueError, "source_ids"):
            assert_matched(changed, reference)
        changed = copy.deepcopy(reference)
        changed["per_sample"][0]["controls"]["near_clean"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "control input seed"):
            assert_matched(changed, reference)
        changed = copy.deepcopy(reference)
        changed["per_sample"].append(copy.deepcopy(changed["per_sample"][0]))
        with self.assertRaisesRegex(ValueError, "sample sets differ or contain duplicates"):
            assert_matched(changed, reference)

    def test_stride_final_budget_condition_mode_and_initial_hash_are_required(self):
        reference = evaluation()
        changed = copy.deepcopy(reference)
        changed["frame_stride"] = 1
        with self.assertRaisesRegex(ValueError, "frame_stride"):
            assert_matched(changed, reference)
        changed = copy.deepcopy(reference)
        changed["per_sample"][0]["frame_indices"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "input/target sample mismatch"):
            assert_matched(changed, reference)
        changed = copy.deepcopy(reference)
        changed["checkpoint_step"] = 250
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            assert_matched(changed, reference)
        for variant in ("known", "blind", "blind_confidence"):
            with self.subTest(variant=variant):
                good = evaluation(variant)
                for value in (None, "missing"):
                    changed = copy.deepcopy(good)
                    changed["condition_mode"] = value
                    with self.assertRaisesRegex(ValueError, "condition interference"):
                        assert_matched(changed, good)
        for value in (None, ""):
            with self.subTest(initial_hash=value):
                changed = copy.deepcopy(reference)
                changed["initial_backbone_sha256"] = value
                changed["training_contract"]["initial_backbone_sha256"] = value
                # Even two equally incomplete evaluations must fail closed.
                with self.assertRaisesRegex(ValueError, "initial backbone fingerprint"):
                    assert_matched(changed, copy.deepcopy(changed))

    def test_paired_directions_source_weights_and_calibration_separation(self):
        reference = evaluation("known")
        current = evaluation("blind")
        for row in current["per_sample"]:
            if row["source_id"] == "source-a":
                row["model"].update(psnr=21., mse=.03, mae=.1, lpips=.1)
            else:
                row["model"].update(psnr=17., mse=.06, mae=.2, lpips=.5)
        result = paired(current, reference)
        for task in TASKS:
            summary = result["task"][task]
            self.assertEqual(summary["sources"], 2)
            self.assertEqual(summary["samples"], 4)
            psnr = summary["model_minus_baseline"]["psnr"]
            self.assertAlmostEqual(psnr["mean"], -1.)
            self.assertEqual(psnr["direction"], "higher")
            self.assertEqual(psnr["model_better_sources"], 1)
            self.assertEqual(psnr["ties"], 0)
            mse = summary["model_minus_baseline"]["mse"]
            self.assertAlmostEqual(mse["mean"], .005)
            self.assertEqual(mse["direction"], "lower")
            self.assertEqual(mse["model_better_sources"], 1)
            self.assertAlmostEqual(summary["model_minus_baseline"]["lpips"]["mean"], .1)
            self.assertEqual(summary["damage"]["damaged_samples"], 1)
            identity = result["identity"][task]["model_minus_baseline"]["psnr"]
            self.assertEqual(identity["mean"], 0.)
            self.assertEqual(identity["ties"], 2)
        reverse = paired(reference, current)
        self.assertAlmostEqual(reverse["task"]["sr"]["model_minus_baseline"]["psnr"]["mean"], 1.)

        # Improving calibration alone cannot become a claimed RGB improvement.
        confidence = copy.deepcopy(current)
        confidence["model_config"]["variant"] = "blind_confidence"
        confidence["calibration_per_task"] = {task: {"gaussian_nll_without_constant": -20.}
                                                for task in TASKS}
        comparison = paired(confidence, current)
        for task in TASKS:
            for metric in ("psnr", "mse", "mae", "lpips"):
                change = comparison["task"][task]["model_minus_baseline"][metric]
                self.assertEqual(change["mean"], 0.)
                self.assertEqual(change["model_better_sources"], 0)
                self.assertEqual(change["ties"], 2)


if __name__ == "__main__":
    unittest.main()
