"""Post-hoc R_phi evidence audit tests: NumPy only, no model or NPU load."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_rphi_results import ARMS, FLOW_DESCRIPTION, audit_results, main
from eval_rphi import TASKS


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def evaluation(variant):
    initial = sha("same initial backbone")
    result = {"schema": "rphi.rgb_eval.v1", "complete": True,
              "scope": "full_development_set", "dev_manifest_sha256": sha("dev manifest"),
              "source_ids": ["source-a", "source-b"], "frame_stride": 4,
              "metric_contract": {"rgb": "clipped RGB", "temporal": "stride-4 GT proxy"},
              "training_contract": {"steps": 1000, "initial_backbone_sha256": initial},
              "initial_backbone_sha256": initial, "checkpoint_step": 1000,
              "fidelity_state_sha256": sha("same trained fidelity weights"),
              "condition_mode": "known" if variant == "known" else "observations_only",
              "model_config": {"variant": variant, "k": 3, "ch": 8, "blocks": 1, "condition_dim": 6},
              "lpips": {"available": True, "alexnet_sha256": sha("AlexNet weights")},
              "per_sample": [], "calibration_per_task": {}}
    for source, windows in (("source-a", 2), ("source-b", 1)):
        for window in range(windows):
            target = f"{source}-window{window}"
            for task in TASKS:
                metric = {"psnr": 20., "mse": .01, "mae": .05, "lpips": .1}
                temporal = {"pairs": 2, "valid_pairs": 2, "valid_coverage": .9,
                            "unwarped_rgb_error_change_mae": .05, "flow_aligned_rgb_error_change_mae": .04}
                score = {"model": dict(metric), "baseline": dict(metric),
                         "temporal_model": dict(temporal), "temporal_baseline": dict(temporal),
                         "attribution": {"mse_change": 0.}}
                differences = {"rgb": 1e-7}
                if variant == "blind_confidence":
                    differences["log_variance"] = 2e-7
                result["per_sample"].append({
                    "sample_id": f"{target}-{task}", "source_id": source, "source_sha256": sha(source),
                    "target_key": target, "task": task, "target_pixel_sha256": sha(target),
                    "observation_pixel_sha256": sha(target + task), "record_sha256": sha("record" + target + task),
                    "frame_indices": [0, 4, 8], "flow": FLOW_DESCRIPTION, **copy.deepcopy(score),
                    "chunk_equivalence": {"passed": True, "checked_frames": [0, 1, 2, 4, 8],
                                          "tolerance_max_abs": 1e-5, "max_abs_differences": differences},
                    "controls": {"identity": copy.deepcopy(score),
                                 "near_clean": {**copy.deepcopy(score), "seed": 123, "sigma_8bit": 1.0}}})
    return result


def arms():
    return {arm: evaluation(variant) for arm, variant in ARMS.items()}


class RphiAcceptanceTests(unittest.TestCase):
    def test_supported_complete_audit_does_not_declare_quality_passed(self):
        inputs = arms()
        original = copy.deepcopy(inputs)
        report = audit_results(**inputs)
        self.assertTrue(report["complete"])
        self.assertEqual(report["quality_verdict"], "requires_review")
        self.assertEqual(report["comparability"]["status"], "passed")
        self.assertEqual(report["comparability"]["samples_per_arm"], 12)
        self.assertTrue(report["fidelity_weights_identical_blind_and_confidence"])
        self.assertEqual(report["temporal_support_by_sample"][0]["valid_coverage"], .9)
        self.assertEqual(report["quality_by_arm"]["blind"]["task"]["sr"]["sources"], 2)
        self.assertEqual(inputs, original)

    def test_missing_or_disabled_flow_rejected_even_if_all_arms_skip(self):
        for all_arms in (False, True):
            for flow in (None, "disabled", "cv2 unavailable"):
                with self.subTest(all_arms=all_arms, flow=flow):
                    inputs = arms()
                    for arm in inputs if all_arms else ("blind",):
                        inputs[arm]["per_sample"][0]["flow"] = flow
                    with self.assertRaisesRegex(ValueError, "flow description"):
                        audit_results(**inputs)

    def test_support_mismatch_rejected_for_every_control_and_side(self):
        for control in ("task", "identity", "near_clean"):
            for side in ("model", "baseline"):
                for field, value in (("valid_pairs", 1), ("valid_coverage", .8), ("pairs", 3)):
                    with self.subTest(control=control, side=side, field=field):
                        inputs = arms()
                        row = inputs["confidence"]["per_sample"][0]
                        score = row if control == "task" else row["controls"][control]
                        score["temporal_" + side][field] = value
                        with self.assertRaisesRegex(ValueError, "mismatch"):
                            audit_results(**inputs)

    def test_missing_empty_or_nonfinite_temporal_support_rejected(self):
        for field, bad in (("valid_pairs", None), ("valid_pairs", 0), ("pairs", None),
                           ("valid_coverage", None), ("valid_coverage", 0.),
                           ("valid_coverage", float("nan")),
                           ("flow_aligned_rgb_error_change_mae", None)):
            with self.subTest(field=field, bad=bad):
                inputs = arms()
                inputs["blind"]["per_sample"][0]["temporal_model"][field] = bad
                with self.assertRaises(ValueError):
                    audit_results(**inputs)

    def test_missing_lpips_and_actual_sample_values_rejected(self):
        inputs = arms()
        for result in inputs.values():
            result["lpips"]["available"] = False
        with self.assertRaisesRegex(ValueError, "LPIPS must be available"):
            audit_results(**inputs)
        inputs = arms()
        inputs["blind"]["per_sample"][0]["controls"]["identity"]["model"]["lpips"] = None
        with self.assertRaisesRegex(ValueError, "lpips must be finite"):
            audit_results(**inputs)

    def test_chunk_failure_and_inconsistent_pass_flag_rejected(self):
        for value in (False, None):
            inputs = arms()
            inputs["blind"]["per_sample"][0]["chunk_equivalence"]["passed"] = value
            with self.assertRaisesRegex(ValueError, "chunk check did not pass"):
                audit_results(**inputs)
        for key in ("rgb", "log_variance"):
            inputs = arms()
            inputs["confidence"]["per_sample"][0]["chunk_equivalence"]["max_abs_differences"][key] = 1e-3
            with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
                audit_results(**inputs)

    def test_different_fidelity_hash_requires_investigation_not_quality_pass(self):
        for value in (sha("changed fidelity weights"), None, ""):
            with self.subTest(value=value):
                inputs = arms()
                inputs["confidence"]["fidelity_state_sha256"] = value
                result = audit_results(**inputs)
                self.assertTrue(result["complete"])
                self.assertFalse(result["fidelity_weights_identical_blind_and_confidence"])
                self.assertEqual(result["fidelity_state_check"]["status"], "requires_investigation")
                self.assertEqual(result["quality_verdict"], "requires_review")

    def test_quality_summary_preserves_source_weighting_and_control_damage(self):
        inputs = arms()
        for row in inputs["blind"]["per_sample"]:
            row["model"]["psnr"] += 2 if row["source_id"] == "source-a" else -4
            # A stale attribution must not conceal control damage.
            row["controls"]["identity"]["model"]["mse"] = .012
        result = audit_results(**inputs)
        self.assertAlmostEqual(result["blind_minus_known"]["task"]["sr"]["model_minus_baseline"]["psnr"]["mean"], -1.)
        summary = result["quality_by_arm"]["blind"]["identity"]["sr"]
        self.assertEqual(summary["damage"]["damaged_samples"], 3)
        self.assertAlmostEqual(summary["damage"]["max_mse_increase"], .002)

    def test_existing_matched_contract_is_still_enforced(self):
        inputs = arms()
        inputs["blind"]["per_sample"][0]["target_pixel_sha256"] = sha("different target")
        with self.assertRaisesRegex(ValueError, "input/target sample mismatch"):
            audit_results(**inputs)

    def test_cli_records_exact_input_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = []
            paths = {}
            for arm, evaluation_result in arms().items():
                paths[arm] = root / (arm + ".json")
                paths[arm].write_text(json.dumps(evaluation_result))
                argv.extend(["--" + arm, str(paths[arm])])
            out = root / "audit.json"
            argv.extend(["--out", str(out)])
            main(argv)
            first_bytes = out.read_bytes()
            result = json.loads(first_bytes)
            for arm, path in paths.items():
                self.assertEqual(result["inputs"][arm]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                main(argv)
            self.assertEqual(out.read_bytes(), first_bytes)


if __name__ == "__main__":
    unittest.main()
