"""Stage-2 evidence contracts; CPU/NumPy only, no torch or checkpoint loading."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_rphi_stage2 import ARMS, FLOW_DESCRIPTION, TASKS, audit_stage2, main


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixtures():
    result = {}
    for arm, (encoder, guard, variant) in ARMS.items():
        initial = sha("same complete initialization")
        contract = {"steps": 3000, "seed": 20261111, "crop": 96, "k": 3,
                    "batch_size": 16, "identity_fraction": .125, "near_clean_fraction": .125,
                    "observation_encoder": encoder, "initial_backbone_sha256": initial,
                    "train_manifest_sha256": sha("train"), "dev_manifest_sha256": sha("dev"),
                    "train_source_ids": ["train-source"], "train_source_sha256": [sha("train-source")],
                    "dev_source_ids": ["a", "b"], "task_order": list(TASKS),
                    "loss": {"rgb": "Charbonnier", "edge_l1_weight": .1, "identity_guard_weight": guard},
                    "optimizer": {"class": "AdamW", "base_lr": .0001},
                    "lr_schedule": {"warmup_steps": 50, "floor": .2},
                    "determinism": {"requested": "inherit", "runtime": {"torch": "test"}}}
        data = {"schema": "rphi.rgb_eval.v1", "complete": True,
                "scope": "full_development_set", "source_ids": ["a", "b"], "available_sources": 2,
                "dev_manifest_sha256": sha("dev"), "initial_backbone_sha256": initial,
                "initial_fidelity_sha256": initial, "checkpoint_sha256": sha(arm),
                "fidelity_state_sha256": sha("fixed native") if arm in ("native_g05", "frozen_confidence") else sha(arm),
                "checkpoint_step": 3000, "training_contract": contract, "condition_mode": "observations_only",
                "model_config": {"variant": variant, "observation_encoder": encoder,
                                 "k": 3, "ch": 64, "blocks": 8, "condition_dim": 6},
                "frame_stride": 4, "metric_contract": {"rgb": "unit RGB", "temporal": "selected pairs"},
                "lpips": {"available": True, "alexnet_sha256": sha("weights")},
                "device": "npu:0", "dataset_provenance": {"manifest_sha256": sha("dev")},
                "per_sample": [], "calibration_per_task": {}}
        for source, windows in (("a", 2), ("b", 1)):
            for window in range(windows):
                for task in TASKS:
                    target = f"{source}-{window}"
                    metrics = {"psnr": 20., "psnr_from_mean_mse": 20., "mse": .01,
                               "mae": .05, "lpips": .1, "frames": 3, "perfect_frames": 0}
                    temporal = {"pairs": 2, "valid_pairs": 2, "valid_coverage": .9,
                                "unwarped_rgb_error_change_mae": .05, "flow_aligned_rgb_error_change_mae": .04}
                    score = {"model": dict(metrics), "baseline": dict(metrics),
                             "temporal_model": dict(temporal), "temporal_baseline": dict(temporal),
                             "raw_model_mse": .01, "clipped_component_fraction": 0.,
                             "attribution": {"mse_change": 0.},
                             "calibration": {"available": variant == "blind_confidence",
                                             "gaussian_nll_without_constant": -2.,
                                             "variance_calibration_mae": .001,
                                             "coverage_1sigma": .75, "coverage_1_96sigma": .96}}
                    differences = {"rgb": 1e-7}
                    if variant == "blind_confidence":
                        differences["log_variance"] = 1e-7
                    data["per_sample"].append({"sample_id": target + task, "source_id": source,
                        "source_sha256": sha(source), "target_key": target, "task": task,
                        "target_pixel_sha256": sha(target), "observation_pixel_sha256": sha(target + task),
                        "record_sha256": sha("record" + target + task), "frame_indices": [0, 4, 8],
                        "flow": FLOW_DESCRIPTION, **copy.deepcopy(score),
                        "chunk_equivalence": {"passed": True, "checked_frames": [0, 1, 2, 4, 8],
                            "tolerance_max_abs": 1e-5, "max_abs_differences": differences},
                        "controls": {"identity": copy.deepcopy(score),
                                     "near_clean": {**copy.deepcopy(score), "seed": 123, "sigma_8bit": 1.0}}})
        result[arm] = data
    frozen = result["frozen_confidence"]
    parent = result["native_g05"]
    frozen["source_checkpoint_sha256"] = parent["checkpoint_sha256"]
    frozen["calibration_step"] = 1000
    frozen["calibration_contract"] = {
        "schema": "rphi.frozen_confidence_calibration.v1", "steps": 1000,
        "source_checkpoint_sha256": parent["checkpoint_sha256"],
        "source_fidelity_state_sha256": parent["fidelity_state_sha256"], "source_fidelity_step": 3000,
        **{key: copy.deepcopy(parent["training_contract"][key])
           for key in ("train_manifest_sha256", "train_source_ids", "train_source_sha256")}}
    frozen["fidelity_invariance"] = {
        "identical": True, "fidelity_before_sha256": parent["fidelity_state_sha256"],
        "fidelity_after_sha256": parent["fidelity_state_sha256"],
        "raw_output_max_abs_vs_source": {"rgb": 0., "anchor": 0., "condition_code": 0.}}
    return result


class Stage2AuditTests(unittest.TestCase):
    def test_matched_factors_reused_without_mutation_or_quality_claim(self):
        values = fixtures()
        before = copy.deepcopy(values)
        report = audit_stage2(**values)
        self.assertEqual(values, before)
        self.assertEqual(report["quality_verdict"], "requires_review")
        self.assertEqual(report["comparability"]["samples_per_arm"], 12)
        self.assertEqual(report["comparability"]["initial_fidelity_check"], "passed")
        self.assertFalse(report["temporal_interpretation"]["adjacent_frames_scored"])
        self.assertEqual(report["temporal_interpretation"]["phase"], "strided_screening")
        self.assertEqual(report["frozen_fidelity_check"]["status"], "passed")
        self.assertIn("legacy_g20 is absent", report["factors"]["not_estimated"])

    def test_undeclared_training_differences_fail_closed(self):
        for field, value in (("seed", 1), ("crop", 128), ("identity_fraction", .25),
                             ("near_clean_fraction", .25), ("steps", 1000),
                             ("optimizer", {"base_lr": .01}), ("lr_schedule", {}),
                             ("determinism", {"requested": "warn"}), ("extra_recipe", True)):
            with self.subTest(field=field):
                values = fixtures()
                values["native_g20"]["training_contract"][field] = value
                with self.assertRaises(ValueError):
                    audit_stage2(**values)

    def test_declared_encoder_guard_and_variant_are_exact(self):
        for arm in ARMS:
            for where, key, value in (("model_config", "observation_encoder", "wrong"),
                                      ("training_contract", "observation_encoder", "wrong"),
                                      ("model_config", "variant", "known")):
                with self.subTest(arm=arm, where=where, key=key):
                    values = fixtures()
                    values[arm][where][key] = value
                    with self.assertRaises(ValueError):
                        audit_stage2(**values)
        values = fixtures()
        values["native_g20"]["training_contract"]["loss"]["identity_guard_weight"] = .5
        with self.assertRaisesRegex(ValueError, "predeclared"):
            audit_stage2(**values)

    def test_undeclared_model_config_difference_rejected(self):
        for key, value in (("ch", 32), ("extra_feature", True)):
            values = fixtures()
            values["native_g05"]["model_config"][key] = value
            with self.assertRaises(ValueError):
                audit_stage2(**values)

    def test_input_target_control_and_final_budget_contracts_retained(self):
        for field in ("source_sha256", "target_pixel_sha256", "observation_pixel_sha256", "record_sha256"):
            values = fixtures()
            values["native_g20"]["per_sample"][0][field] = sha("changed")
            with self.assertRaisesRegex(ValueError, "input/target"):
                audit_stage2(**values)
        values = fixtures()
        values["native_g20"]["per_sample"][0]["controls"]["near_clean"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "control input seed"):
            audit_stage2(**values)
        values = fixtures()
        values["native_g20"]["checkpoint_step"] = 1000
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            audit_stage2(**values)

    def test_missing_contract_and_initial_fidelity_rejected(self):
        values = fixtures()
        for result in values.values():
            del result["training_contract"]["optimizer"]
        with self.assertRaisesRegex(ValueError, "missing training contract"):
            audit_stage2(**values)
        for value in (None, sha("different init")):
            values = fixtures()
            values["native_g05"]["initial_fidelity_sha256"] = value
            with self.assertRaisesRegex(ValueError, "initial fidelity"):
                audit_stage2(**values)

    def test_frozen_hash_mismatch_is_hard_failure(self):
        for value in (None, sha("different fidelity")):
            values = fixtures()
            values["frozen_confidence"]["fidelity_state_sha256"] = value
            with self.assertRaises(ValueError):
                audit_stage2(**values)

    def test_frozen_parent_training_data_budget_and_probe_chain(self):
        for change in (
            lambda x: x.update(source_checkpoint_sha256=sha("wrong parent")),
            lambda x: x.update(calibration_step=500),
            lambda x: x["calibration_contract"].update(train_manifest_sha256=sha("dev")),
            lambda x: x["calibration_contract"].update(source_fidelity_step=1000),
            lambda x: x["fidelity_invariance"].update(identical=False),
            lambda x: x["fidelity_invariance"]["raw_output_max_abs_vs_source"].update(rgb=.001),
        ):
            values = fixtures()
            change(values["frozen_confidence"])
            with self.assertRaises(ValueError):
                audit_stage2(**values)

    def test_frozen_rgb_metrics_allow_only_numerical_tolerance(self):
        values = fixtures()
        values["frozen_confidence"]["per_sample"][0]["model"]["psnr"] += 1e-5
        report = audit_stage2(**values)
        self.assertGreater(report["frozen_fidelity_check"]["maximum_metric_abs_differences"]["psnr"], 0)
        for control in ("task", "identity", "near_clean"):
            values = fixtures()
            row = values["frozen_confidence"]["per_sample"][0]
            score = row if control == "task" else row["controls"][control]
            score["model"]["mse"] += .001
            with self.assertRaisesRegex(ValueError, "frozen RGB"):
                audit_stage2(**values)
        values = fixtures()
        values["frozen_confidence"]["per_sample"][0]["raw_model_mse"] += .001
        with self.assertRaisesRegex(ValueError, "frozen RGB"):
            audit_stage2(**values)

    def test_calibration_missing_nonfinite_and_invalid_coverage_rejected(self):
        for key, value in (("available", False), ("gaussian_nll_without_constant", float("nan")),
                           ("coverage_1sigma", 1.1), ("variance_calibration_mae", -.1)):
            values = fixtures()
            values["frozen_confidence"]["per_sample"][0]["controls"]["identity"]["calibration"][key] = value
            with self.assertRaises(ValueError):
                audit_stage2(**values)

    def test_lpips_flow_support_chunk_and_baseline_are_checked(self):
        cases = (lambda x: x["lpips"].update(available=False),
                 lambda x: x["per_sample"][0].update(flow="disabled"),
                 lambda x: x["per_sample"][0]["temporal_model"].update(valid_coverage=.8),
                 lambda x: x["per_sample"][0]["chunk_equivalence"].update(passed=False),
                 lambda x: x["per_sample"][0]["controls"]["identity"]["model"].update(lpips=None),
                 lambda x: x["per_sample"][0]["baseline"].update(mse=.02))
        for change in cases:
            values = fixtures()
            change(values["native_g20"])
            with self.assertRaises(ValueError):
                audit_stage2(**values)

    def test_source_coverage_split_and_stride_dishonesty_rejected(self):
        values = fixtures()
        values["native_g20"]["available_sources"] = 3
        with self.assertRaisesRegex(ValueError, "source coverage"):
            audit_stage2(**values)
        values = fixtures()
        for data in values.values():
            data["frame_stride"] = 1
        with self.assertRaisesRegex(ValueError, "frame indices"):
            audit_stage2(**values)
        values = fixtures()
        for data in values.values():
            data["training_contract"]["train_source_sha256"] = [sha("a")]
        with self.assertRaisesRegex(ValueError, "source SHA256 overlap"):
            audit_stage2(**values)

    def test_paired_source_weights_and_stale_damage_attribution(self):
        values = fixtures()
        for row in values["native_g20"]["per_sample"]:
            row["model"]["psnr"] += 2 if row["source_id"] == "a" else -4
            row["controls"]["identity"]["model"]["mse"] += .002
        report = audit_stage2(**values)
        paired = report["guard20_minus_guard05"]["task"]["sr"]
        self.assertAlmostEqual(paired["model_minus_baseline"]["psnr"]["mean"], -1.)
        damage = report["quality_by_arm"]["native_g20"]["identity"]["sr"]["damage"]
        self.assertEqual(damage["damaged_samples"], 3)
        self.assertAlmostEqual(damage["max_mse_increase"], .002)

    def test_stride1_is_still_not_automatic_flicker_acceptance(self):
        values = fixtures()
        for data in values.values():
            data["frame_stride"] = 1
            for row in data["per_sample"]:
                row["frame_indices"] = [0, 1, 2]
                row["chunk_equivalence"]["checked_frames"] = [0, 1, 2]
        report = audit_stage2(**values)
        self.assertTrue(report["temporal_interpretation"]["adjacent_frames_scored"])
        self.assertIn("No flicker quality acceptance", report["temporal_interpretation"]["note"])
        self.assertEqual(report["quality_verdict"], "requires_review")

    def test_cli_hashes_evidence_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, result in fixtures().items():
                path = root / arm / "eval" / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(result))
            out = root / "stage2_audit.json"
            main(["--run-dir", str(root), "--out", str(out)])
            payload = out.read_bytes()
            report = json.loads(payload)
            for arm, entry in report["inputs"].items():
                self.assertEqual(entry["sha256"], hashlib.sha256((root / arm / "eval" / "metrics.json").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                main(["--run-dir", str(root), "--out", str(out)])
            self.assertEqual(out.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
