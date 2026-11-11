"""Audit the declared three-run R_phi stage-2 matrix and frozen calibration.

This CPU/NumPy evidence audit never loads models. It reuses assert_matched after
normalizing ONLY the two declared training factors, and does not turn successful
execution into a quality, adjacent-frame flicker, or generalization claim.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from audit_rphi_results import (
    CONTROLS, FLOW_DESCRIPTION, SUPPORT_FIELDS, TEMPORAL_METRICS,
    _check_chunk, _compact_summary, _finite, _require, _scored, _sha,
)
from compare_rphi import assert_matched, paired
from eval_rphi import TASKS


SCHEMA = "rphi.stage2_audit.v1"
ARMS = {
    "legacy_g05": ("legacy", .5, "blind"),
    "native_g05": ("native", .5, "blind"),
    "native_g20": ("native", 2., "blind"),
    "frozen_confidence": ("native", .5, "blind_confidence"),
}
CONTRACT_FIELDS = (
    "train_manifest_sha256", "dev_manifest_sha256", "train_source_ids",
    "train_source_sha256", "dev_source_ids", "task_order", "k", "crop", "seed",
    "steps", "batch_size", "identity_fraction", "near_clean_fraction", "loss",
    "optimizer", "lr_schedule", "determinism", "observation_encoder",
    "initial_backbone_sha256",
)
RGB_TOLERANCE = {"rtol": 1e-5, "absolute_default": 1e-7,
                 "absolute_psnr_db": 1e-4}


def normalized(result, arm):
    """Validate declared factors, then copy with the two factors neutralized.

    Unknown/additional contract and model-config fields remain in the copy and
    must match. Original evidence is never mutated or relabeled as a known arm.
    """
    encoder, guard, variant = ARMS[arm]
    config, contract = result["model_config"], result["training_contract"]
    _require(config.get("variant") == variant, arm + " variant mismatch")
    for field in CONTRACT_FIELDS:
        _require(field in contract, arm + " missing training contract field: " + field)
    _require(config.get("observation_encoder") == encoder,
             arm + " model observation_encoder mismatch")
    _require(contract["observation_encoder"] == encoder,
             arm + " training observation_encoder mismatch")
    weight = _finite(contract["loss"].get("identity_guard_weight"), arm + " guard", 0)
    _require(weight == guard, arm + " guard is not the predeclared value")
    _require(contract["steps"] == 3000, arm + " requires the matched 3000-step parent budget")
    _require(contract["task_order"] == list(TASKS), arm + " task order mismatch")
    _require(contract["dev_source_ids"] == result["source_ids"], arm + " dev source contract mismatch")
    _require(contract["dev_manifest_sha256"] == result["dev_manifest_sha256"],
             arm + " dev manifest contract mismatch")
    for key in ("initial_backbone_sha256", "fidelity_state_sha256", "checkpoint_sha256"):
        _require(_sha(result.get(key)), arm + " missing SHA256: " + key)
    for key in ("train_manifest_sha256", "dev_manifest_sha256"):
        _require(_sha(contract[key]), arm + " invalid manifest SHA256: " + key)
    _require(not set(contract["train_source_ids"]) & set(contract["dev_source_ids"]),
             arm + " train/dev source ID overlap")
    _require(all(_sha(v) for v in contract["train_source_sha256"])
             and contract["train_source_sha256"], arm + " invalid training source SHA256")
    for row in result["per_sample"]:
        _require(row["source_sha256"] not in contract["train_source_sha256"],
                 arm + " train/dev source SHA256 overlap")
    output = copy.deepcopy(result)
    output["training_contract"]["observation_encoder"] = "declared_factor"
    output["training_contract"]["loss"]["identity_guard_weight"] = "declared_factor"
    output["model_config"]["observation_encoder"] = "declared_factor"
    output["model_config"]["variant"] = "blind"
    return output


def _close_tree(a, b, context, tolerance=None, maximum=None):
    """Compare metric trees; integers/labels/availability must match exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        _require(a.keys() == b.keys(), context + " metric fields mismatch")
        for key in a:
            _close_tree(a[key], b[key], context + "/" + key, tolerance, maximum)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        _finite(a, context)
        _finite(b, context)
        if isinstance(a, int) or isinstance(b, int):
            _require(a == b, context + " integer metric mismatch")
            return
        delta = abs(a - b)
        if maximum is not None:
            key = context.split("/")[-1]
            maximum[key] = max(maximum.get(key, 0.), delta)
        absolute = 0. if tolerance is None else (
            tolerance["absolute_psnr_db"] if context.split("/")[-1].startswith("psnr")
            else tolerance["absolute_default"])
        relative = 0. if tolerance is None else tolerance["rtol"]
        _require(math.isclose(a, b, rel_tol=relative, abs_tol=absolute),
                 context + " metric mismatch exceeds tolerance")
    else:
        _require(a == b, context + " metric value mismatch")


def _support_audit(loaded):
    indexed = {arm: {r["sample_id"]: r for r in result["per_sample"]}
               for arm, result in loaded.items()}
    support_rows = []
    for sample in sorted(indexed["legacy_g05"]):
        reference_support = None
        reference_scores = {}
        for arm, (_, _, variant) in ARMS.items():
            row = indexed[arm][sample]
            context = arm + "/" + sample
            _require(row.get("flow") == FLOW_DESCRIPTION, context + " flow description mismatch")
            frames, stride = row["frame_indices"], loaded[arm]["frame_stride"]
            _require(isinstance(frames, list) and len(frames) >= 2
                     and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in frames)
                     and frames[0] == 0 and all(b - a == stride for a, b in zip(frames, frames[1:])),
                     context + " frame indices do not match declared stride")
            _check_chunk(row, variant, context)
            sigma = _finite(row["controls"]["near_clean"].get("sigma_8bit"), context + " near_clean sigma", 0)
            _require(sigma == 1., context + " near_clean sigma mismatch")
            for control in CONTROLS:
                score, label = _scored(row, control), context + "/" + control
                _require(score.get("flow", row["flow"]) == FLOW_DESCRIPTION, label + " flow description mismatch")
                for kind in ("model", "baseline"):
                    for metric in ("psnr", "mae", "mse", "lpips"):
                        _finite(score.get(kind, {}).get(metric), label + "/" + kind + "/" + metric,
                                0 if metric in ("mae", "mse") else None)
                    temporal = score.get("temporal_" + kind, {})
                    for field in ("pairs", "valid_pairs"):
                        value = temporal.get(field)
                        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                                 label + " missing positive temporal " + field)
                    _require(temporal["pairs"] == len(frames) - 1
                             and temporal["valid_pairs"] <= temporal["pairs"], label + " temporal pair mismatch")
                    coverage = _finite(temporal.get("valid_coverage"), label + " valid_coverage", 0, 1)
                    _require(coverage > 0, label + " empty flow support")
                    for metric in TEMPORAL_METRICS:
                        _finite(temporal.get(metric), label + "/" + metric, 0)
                    support = {key: temporal[key] for key in SUPPORT_FIELDS}
                    if reference_support is None:
                        reference_support = support
                    _require(support == reference_support, label + " temporal support mismatch")
                if control not in reference_scores:
                    reference_scores[control] = score
                for key in ("baseline", "temporal_baseline"):
                    _close_tree(score[key], reference_scores[control][key], label + "/" + key)
        support_rows.append({"sample_id": sample, "flow": FLOW_DESCRIPTION, **reference_support})
    return indexed, support_rows


def _frozen_check(indexed, loaded):
    parent, frozen = loaded["native_g05"], loaded["frozen_confidence"]
    _require(parent["fidelity_state_sha256"] == frozen["fidelity_state_sha256"],
             "frozen_confidence fidelity SHA256 differs from native_g05")
    contract = frozen.get("calibration_contract")
    _require(isinstance(contract, dict) and contract.get("schema") == "rphi.frozen_confidence_calibration.v1",
             "frozen calibration contract missing")
    _require(frozen.get("source_checkpoint_sha256") == parent["checkpoint_sha256"]
             and contract.get("source_checkpoint_sha256") == parent["checkpoint_sha256"],
             "frozen calibration source checkpoint SHA256 mismatch")
    _require(contract.get("source_fidelity_state_sha256") == parent["fidelity_state_sha256"]
             and contract.get("source_fidelity_step") == parent["checkpoint_step"],
             "frozen calibration parent fidelity mismatch")
    steps = contract.get("steps")
    _require(isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
             and frozen.get("calibration_step") == steps, "frozen calibration final budget mismatch")
    for key in ("train_manifest_sha256", "train_source_ids", "train_source_sha256"):
        _require(contract.get(key) == parent["training_contract"][key],
                 "frozen calibration training data mismatch: " + key)
    invariance = frozen.get("fidelity_invariance", {})
    _require(invariance.get("identical") is True
             and invariance.get("fidelity_before_sha256") == parent["fidelity_state_sha256"]
             and invariance.get("fidelity_after_sha256") == parent["fidelity_state_sha256"],
             "frozen calibration fidelity invariance evidence mismatch")
    for key in ("rgb", "anchor", "condition_code"):
        delta = invariance.get("raw_output_max_abs_vs_source", {}).get(key)
        _finite(delta, "frozen probe " + key, 0, 1e-5)
    maximum = {}
    for sample, row in indexed["native_g05"].items():
        other = indexed["frozen_confidence"][sample]
        for control in CONTROLS:
            a, b = _scored(row, control), _scored(other, control)
            for key in ("model", "temporal_model", "raw_model_mse", "clipped_component_fraction"):
                _require(key in a and key in b, "missing frozen RGB evidence: " + key)
                _close_tree(a[key], b[key], "frozen RGB/" + sample + "/" + control + "/" + key,
                            RGB_TOLERANCE, maximum)
            calibration = b.get("calibration", {})
            _require(calibration.get("available") is True, "frozen calibration sample evidence missing")
            for metric in ("gaussian_nll_without_constant", "variance_calibration_mae"):
                _finite(calibration.get(metric), "frozen calibration " + metric,
                        0 if metric == "variance_calibration_mae" else None)
            for metric in ("coverage_1sigma", "coverage_1_96sigma"):
                _finite(calibration.get(metric), "frozen calibration " + metric, 0, 1)
    return {"status": "passed", "fidelity_state_sha256": parent["fidelity_state_sha256"],
            "source_checkpoint_sha256": parent["checkpoint_sha256"], "calibration_steps": steps,
            "recorded_probe_max_abs_limit": 1e-5,
            "rgb_metric_tolerance": dict(RGB_TOLERANCE), "maximum_metric_abs_differences": maximum,
            "meaning": "Identical recorded fidelity state SHA and numerically agreeing RGB metrics; no RGB benefit attributed to calibration. Metric agreement is not a pixel-array identity test."}


def audit_stage2(legacy_g05, native_g05, native_g20, frozen_confidence):
    loaded = dict(legacy_g05=legacy_g05, native_g05=native_g05,
                  native_g20=native_g20, frozen_confidence=frozen_confidence)
    comparable = {arm: normalized(result, arm) for arm, result in loaded.items()}
    reference = comparable["legacy_g05"]
    for arm, result in loaded.items():
        assert_matched(comparable[arm], reference)
        _require(comparable[arm]["model_config"] == reference["model_config"], arm + " undeclared model-config difference")
        _require(result.get("device") == legacy_g05.get("device"), arm + " evaluation device mismatch")
        _require(result.get("dataset_provenance") == legacy_g05.get("dataset_provenance"), arm + " dataset provenance mismatch")
        _require(result["lpips"].get("available") is True, arm + " LPIPS must be available")
        _require(_sha(result["lpips"].get("alexnet_sha256")), arm + " LPIPS weight SHA256 missing")
        sources = result["source_ids"]
        _require(isinstance(sources, list) and sources and len(sources) == len(set(sources)), arm + " invalid source IDs")
        _require(result.get("available_sources") == len(sources), arm + " incomplete source coverage")
        _require(isinstance(result["frame_stride"], int) and not isinstance(result["frame_stride"], bool)
                 and result["frame_stride"] >= 1, arm + " invalid frame stride")
        _require(result["per_sample"], arm + " empty sample evidence")
        for task in TASKS:
            _require({r["source_id"] for r in result["per_sample"] if r["task"] == task} == set(sources),
                     arm + " incomplete task/source coverage: " + task)
        _require(all(r["task"] in TASKS for r in result["per_sample"]), arm + " unknown task")
    initial = {arm: result.get("initial_fidelity_sha256") for arm, result in loaded.items()}
    if any(value is not None for value in initial.values()):
        _require(all(_sha(value) for value in initial.values()) and len(set(initial.values())) == 1,
                 "initial fidelity SHA256 missing or differs across arms")
    indexed, support = _support_audit(loaded)
    frozen_check = _frozen_check(indexed, loaded)
    quality = {arm: {control: {
        task: _compact_summary([{"source_id": row["source_id"], **_scored(row, control)}
                                for row in result["per_sample"] if row["task"] == task])
        for task in TASKS} for control in CONTROLS} for arm, result in loaded.items()}
    stride = legacy_g05["frame_stride"]
    return {"schema": SCHEMA, "complete": True, "quality_verdict": "requires_review",
            "completion_meaning": "Recorded comparability checks passed; no performance threshold or SOTA claim is applied.",
            "scope": "Source-paired development evidence, not a frozen-test or omni generalization result.",
            "factors": {"encoder": "native_g05 minus legacy_g05; same 3000-step budget and guard 0.5",
                        "guard": "native_g20 minus native_g05; same native encoder and budget, guard 2.0 versus 0.5",
                        "calibration": "frozen_confidence versus its native_g05 parent; fidelity must remain fixed",
                        "not_estimated": "Encoder-by-guard interaction: legacy_g20 is absent."},
            "comparability": {"status": "passed", "arms": list(ARMS), "sources": len(legacy_g05["source_ids"]),
                              "source_ids": legacy_g05["source_ids"], "samples_per_arm": len(support),
                              "frame_stride": stride, "lpips_available_all_arms": True,
                              "initial_fidelity_check": "passed" if any(initial.values()) else "not_recorded; initial backbone checked",
                              "flow_support": "Recorded pairs/valid_pairs/valid_coverage exactly match; flow mask pixels have no hash and are not independently verified.",
                              "chunk_scope": "Recorded ordinary task batch-1/batch-2 checks; controls have no separate chunk check."},
            "temporal_interpretation": {"adjacent_frames_scored": stride == 1,
                "phase": "adjacent_frame_development_evaluation" if stride == 1 else "strided_screening",
                "note": "No flicker quality acceptance is implied." if stride == 1 else
                        f"Pairs are {stride} frames apart; adjacent-frame flicker is not evaluated. Candidate needs stride-1 follow-up."},
            "frozen_fidelity_check": frozen_check, "temporal_support_by_sample": support,
            "quality_by_arm": quality,
            "native_minus_legacy": paired(comparable["native_g05"], comparable["legacy_g05"]),
            "guard20_minus_guard05": paired(comparable["native_g20"], comparable["native_g05"]),
            "frozen_minus_parent_numerical_check": paired(comparable["frozen_confidence"], comparable["native_g05"]),
            "confidence_calibration": frozen_confidence.get("calibration_per_task", {}),
            "metric_contract": legacy_g05["metric_contract"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New JSON path; never overwritten")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    loaded, inputs = {}, {}
    for arm in ARMS:
        path = (args.run_dir / arm / "eval" / "metrics.json").resolve()
        payload = path.read_bytes()
        loaded[arm] = json.loads(payload)
        inputs[arm] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    report = audit_stage2(**loaded)
    report["inputs"] = inputs
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(serialized)
    print(json.dumps({"written": str(args.out), "complete": True, "quality_verdict": "requires_review",
                      "frozen_fidelity_check": "passed", "frame_stride": report["comparability"]["frame_stride"]}))


if __name__ == "__main__":
    main()
