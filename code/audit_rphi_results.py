"""Fail-closed comparability audit of three completed R_phi development arms.

This checks recorded evidence, not model quality or the contents of optical-flow
masks. It does not run inference, change runtime files, or establish performance
acceptance thresholds. Outputs are new files; existing evidence is never replaced.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from compare_rphi import assert_matched, paired
from eval_rphi import TASKS, source_summary


SCHEMA = "rphi.results_audit.v1"
ARMS = {"known": "known", "blind": "blind", "confidence": "blind_confidence"}
CONTROLS = ("task", "identity", "near_clean")
FLOW_DESCRIPTION = "GT-proxy Farneback forward flow; in-bounds + forward/backward consistency mask"
SUPPORT_FIELDS = ("pairs", "valid_pairs", "valid_coverage")
TEMPORAL_METRICS = ("unwarped_rgb_error_change_mae", "flow_aligned_rgb_error_change_mae")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, context, minimum=None, maximum=None):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), context + " must be finite and nonmissing")
    _require(minimum is None or value >= minimum, context + " is below its valid range")
    _require(maximum is None or value <= maximum, context + " is above its valid range")
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _scored(row, control):
    return row if control == "task" else row["controls"][control]


def _check_chunk(row, variant, context):
    report = row.get("chunk_equivalence", {})
    _require(report.get("passed") is True, context + " chunk check did not pass")
    frames = report.get("checked_frames")
    _require(isinstance(frames, list) and frames
             and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in frames),
             context + " chunk checked_frames are missing or invalid")
    expected = sorted(set([0, 1, 2, *row["frame_indices"][:3]]))
    _require(frames == expected, context + " chunk frame coverage differs from the evaluator contract")
    tolerance = _finite(report.get("tolerance_max_abs"), context + " chunk tolerance", 0, 1e-5)
    _require(tolerance > 0, context + " chunk tolerance must be positive")
    differences = report.get("max_abs_differences", {})
    required = ("rgb", "log_variance") if variant == "blind_confidence" else ("rgb",)
    for key in required:
        value = _finite(differences.get(key), context + " chunk " + key, 0)
        _require(value <= tolerance, context + " chunk " + key + " exceeds tolerance despite passed flag")


def _compact_summary(rows):
    # Recompute all summaries from sample rows; do not trust cached top-level
    # summaries or attribution when reporting control damage.
    normalized = [{"source_id": row["source_id"], "model": row["model"],
                   "baseline": row["baseline"], "temporal_model": row["temporal_model"],
                   "temporal_baseline": row["temporal_baseline"],
                   "attribution": {"mse_change": row["model"]["mse"] - row["baseline"]["mse"]}}
                  for row in rows]
    summary = source_summary(normalized)
    return {"sources": summary["sources"], "samples": summary["samples"],
            "model": summary["model"], "baseline": summary["baseline"],
            "model_minus_baseline": {key: value["mean"] if value is not None else None
                                     for key, value in summary["model_minus_baseline"].items()},
            "temporal": summary["temporal"], "damage": summary.get("damage")}


def audit_results(known, blind, confidence):
    """Return a quality-review report, or reject missing/incomparable evidence."""
    loaded = dict(known=known, blind=blind, confidence=confidence)
    # Preserve the existing source, pixel, training-budget, and checkpoint checks.
    assert_matched(blind, known)
    assert_matched(confidence, blind)
    for arm, variant in ARMS.items():
        result = loaded[arm]
        _require(result["model_config"]["variant"] == variant, "arm labels do not match checkpoint variants")
        _require(result["lpips"].get("available") is True, arm + " LPIPS must be available")
        _require(_sha(result["lpips"].get("alexnet_sha256")), arm + " LPIPS weight SHA256 is missing")
        _require(result["per_sample"], arm + " sample evidence is empty")
        sources = set(result["source_ids"])
        for task in TASKS:
            actual = {r["source_id"] for r in result["per_sample"] if r["task"] == task}
            _require(actual == sources and sources, arm + " missing source coverage for task " + task)
        _require(all(r["task"] in TASKS for r in result["per_sample"]), arm + " has an unknown task")

    indexed = {arm: {row["sample_id"]: row for row in result["per_sample"]}
               for arm, result in loaded.items()}
    support_evidence = []
    for sample in sorted(indexed["known"]):
        reference_support = None
        reference_sigma = None
        for arm, variant in ARMS.items():
            row = indexed[arm][sample]
            context = arm + "/" + sample
            _require(row.get("flow") == FLOW_DESCRIPTION, context + " flow description missing, disabled, or unsupported")
            frames = row["frame_indices"]
            _require(isinstance(frames, list) and len(frames) >= 2
                     and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in frames)
                     and frames == sorted(set(frames)), context + " temporal frame indices are invalid")
            _check_chunk(row, variant, context)
            sigma = _finite(row["controls"]["near_clean"].get("sigma_8bit"), context + " near_clean sigma_8bit", 0)
            _require(sigma == 1.0, context + " near_clean sigma differs from the recorded evaluator contract")
            if reference_sigma is None:
                reference_sigma = sigma
            _require(sigma == reference_sigma, context + " near_clean sigma mismatch")
            for control in CONTROLS:
                score = _scored(row, control)
                label = context + "/" + control
                # Controls inherit sample.flow under rphi.rgb_eval.v1. Reject an
                # explicit conflicting override rather than silently ignoring it.
                _require(score.get("flow", row["flow"]) == FLOW_DESCRIPTION, label + " flow description mismatch")
                for kind in ("model", "baseline"):
                    rgb = score.get(kind, {})
                    for metric in ("psnr", "mae", "mse", "lpips"):
                        _finite(rgb.get(metric), label + "/" + kind + " " + metric,
                                minimum=0 if metric in ("mae", "mse") else None)
                    temporal = score.get("temporal_" + kind, {})
                    for field in ("pairs", "valid_pairs"):
                        value = temporal.get(field)
                        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                                 label + "/" + kind + " " + field + " must be a nonmissing positive integer")
                    _require(temporal["pairs"] == len(frames) - 1, label + "/" + kind + " temporal pairs mismatch")
                    _require(temporal["valid_pairs"] <= temporal["pairs"], label + "/" + kind + " invalid valid_pairs")
                    coverage = _finite(temporal.get("valid_coverage"), label + "/" + kind + " valid_coverage", 0, 1)
                    _require(coverage > 0, label + "/" + kind + " has no valid flow coverage")
                    for metric in TEMPORAL_METRICS:
                        _finite(temporal.get(metric), label + "/" + kind + " " + metric, 0)
                    support = {key: temporal[key] for key in SUPPORT_FIELDS}
                    if reference_support is None:
                        reference_support = support
                    _require(support == reference_support, label + "/" + kind + " temporal support mismatch")
        support_evidence.append({"sample_id": sample, "flow": FLOW_DESCRIPTION, **reference_support})

    fidelity_hashes = {arm: loaded[arm].get("fidelity_state_sha256") for arm in ("blind", "confidence")}
    identical = (all(_sha(value) for value in fidelity_hashes.values())
                 and fidelity_hashes["blind"] == fidelity_hashes["confidence"])
    quality = {}
    for arm, result in loaded.items():
        quality[arm] = {}
        for control in CONTROLS:
            quality[arm][control] = {
                task: _compact_summary([{"source_id": row["source_id"], **_scored(row, control)}
                                        for row in result["per_sample"] if row["task"] == task])
                for task in TASKS}
    return {"schema": SCHEMA, "complete": True, "quality_verdict": "requires_review",
            "completion_meaning": "Audit execution completed and recorded comparability checks passed; no performance threshold was applied.",
            "scope": "Matched development ablations; not untouched generalization or omni-restoration evidence.",
            "comparability": {"status": "passed", "arms": list(ARMS),
                              "samples_per_arm": len(support_evidence), "frame_stride": known["frame_stride"],
                              "source_ids": known["source_ids"], "lpips_available_all_arms": True,
                              "flow_support": "Exact equality of recorded pairs/valid_pairs/valid_coverage across arms, controls, and model/baseline; pixel masks have no digest and are not independently verified.",
                              "chunk_scope": "Each ordinary task sample's recorded batch-1 versus batch-2 check passed; controls have no separate recorded chunk test."},
            "fidelity_weights_identical_blind_and_confidence": identical,
            "fidelity_state_check": {"identical": identical, "sha256": fidelity_hashes,
                                     "status": "identical" if identical else "requires_investigation",
                                     "meaning": "Calibration-only confidence is expected to preserve fidelity weights; hash equality alone is not quality acceptance."},
            "temporal_support_by_sample": support_evidence,
            "quality_by_arm": quality,
            "blind_minus_known": paired(blind, known),
            "confidence_minus_blind": paired(confidence, blind),
            "confidence_calibration": confidence.get("calibration_per_task", {}),
            "metric_contract": known["metric_contract"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New audit JSON file; never overwritten")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    loaded, evidence = {}, {}
    for arm in ARMS:
        path = getattr(args, arm).resolve()
        payload = path.read_bytes()
        loaded[arm] = json.loads(payload)
        evidence[arm] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    report = audit_results(**loaded)
    report["inputs"] = evidence
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(serialized)
    print(json.dumps({"written": str(args.out), "complete": True, "quality_verdict": "requires_review",
                      "fidelity_state_check": report["fidelity_state_check"]["status"]}))


if __name__ == "__main__":
    main()
