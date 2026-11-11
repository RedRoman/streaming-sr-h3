"""Summarize six already-scored native baselines; never infer or recompute metrics.

Only local archived evidence is read. A complete, successful six-forward/twelve-
stage queue and hash-bound receipts are mandatory before writing final outputs.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TASKS = ("deblur", "denoise", "compression")
METHODS = ("favernet", "avernet")
PRIOR = ROOT / "results/takeover_20261111"
SOURCES = {
    ("favernet", "deblur"): PRIOR / "native_quality_heartbeat_20261111T1240Z/favernet_deblur/metrics.json",
    ("favernet", "denoise"): PRIOR / "native_quality_denoise_increment_20261111T1300Z/metrics.json",
    ("favernet", "compression"): PRIOR / "native_quality_compression_increment_20261111T1312Z/metrics.json",
    ("avernet", "deblur"): PRIOR / "native_quality_aver_deblur_increment_20261111T1330Z/metrics.json",
    ("avernet", "denoise"): PRIOR / "native_quality_aver_denoise_increment_20261111/metrics.json",
    ("avernet", "compression"): OUT / "avernet_compression/metrics.json",
}
CHECKPOINTS = {
    "favernet": "c16e5658a639810bb17dc4bc131a631bde6491f9fb5a790c8952dfa16ccdc7b8",
    "avernet": "cdba740bab4c47f37a7b0e18eb147a3acab6897b0911744f7c5fbc30054b57ec",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived_file(path):
    directory = path.parent
    while not (directory / "archive_manifest.json").is_file():
        if directory == ROOT or directory == directory.parent:
            raise ValueError("No archive binding: " + str(path))
        directory = directory.parent
    verification = read(directory / "local_verification.json")
    if not verification["all_file_hashes_passed"]:
        raise ValueError("Archive has not passed local hash verification")
    manifest = read(directory / "archive_manifest.json")
    key = str(path.relative_to(directory))
    if sha(path) != manifest[key]["sha256"] or path.stat().st_size != manifest[key]["bytes"]:
        raise ValueError("Archived file changed: " + key)
    return directory


def values(quality, temporal):
    return {**{k: quality[k] for k in ("psnr", "psnr_from_mean_mse", "mse", "mae", "lpips", "frames")},
            **{k: temporal[k] for k in ("pairs", "valid_pairs", "valid_coverage",
                                        "flow_aligned_rgb_error_change_mae",
                                        "unwarped_rgb_error_change_mae")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    missing = [str(p.relative_to(ROOT)) for p in SOURCES.values() if not p.is_file()]
    if args.check_inputs:
        print(json.dumps({"ready": not missing, "missing": missing, "scoring_performed": False}))
        return
    if missing:
        raise ValueError("Six completed metric records required: " + repr(missing))
    queue_path = OUT / "avernet_compression/queue_status.json"
    archived_file(queue_path)
    queue = read(queue_path)
    if queue["status"] != "complete" or queue["completed_forwards"] != 6 or queue["failed_models"]:
        raise ValueError("Successful final queue required")
    expected = {(m, t, s) for m in METHODS for t in TASKS for s in ("infer", "metrics")}
    jobs = queue["jobs"]
    if len(jobs) != 12 or {(j["method"], j["task"], j["stage"]) for j in jobs} != expected:
        raise ValueError("Queue stage count/identity mismatch")
    if any(j["returncode"] != 0 or j["guard_reason"] is not None for j in jobs):
        raise ValueError("Failed or guarded stage cannot enter the completed table")

    per_task, evidence, shared = {}, [], None
    for (method, task), path in SOURCES.items():
        directory = archived_file(path)
        m = read(path)
        prediction_path = path.parent / "prediction_run.json"
        archived_file(prediction_path)
        prediction = read(prediction_path)
        admission_path = directory / "admission.json"
        archived_file(admission_path)
        admission = read(admission_path)
        if sha(admission_path) != queue["admission_sha256"]:
            raise ValueError("Admission differs across runs")
        if not m["complete"] or m["method"] != method or m["task"] != task:
            raise ValueError("Metric identity mismatch")
        if m["record"] != admission["records"][task]:
            raise ValueError("Record differs from admitted source/input/target")
        if m["record"]["source_id"] != "6UZVovkR3IU" or m["record"]["split"] != "dev":
            raise ValueError("Unexpected source or data role")
        if prediction["checkpoint_sha256"] != CHECKPOINTS[method] or not prediction["checkpoint_strict_load"]:
            raise ValueError("Official checkpoint binding mismatch")
        if prediction["device"] != "cpu" or prediction["gt_read"] or prediction["resizing"] or prediction["tiling"]:
            raise ValueError("Unexpected inference device/GT/geometry")
        if prediction["input_npz_sha256"] != admission["input_files"][m["record"]["observation_npz"]]:
            raise ValueError("Prediction input file binding mismatch")
        if prediction["input_pixel_sha256"] != m["record"]["observation_pixel_sha256"]:
            raise ValueError("Prediction input pixel binding mismatch")
        if m["frame_indices"] != list(range(22)) or m["frame_stride"] != 1:
            raise ValueError("Full22 stride1 required")
        if not m["metric_gate"]["passed"] or not m["lpips"]["available"] or m["lpips"]["network"] != "alex":
            raise ValueError("LPIPS/metric gate missing")
        if not m["reference_input_gate"]["passed"]:
            raise ValueError("Input baseline equivalence gate failed")
        if not all(c["absolute_difference"] == 0 for c in m["reference_input_gate"]["checks"].values()):
            raise ValueError("Expected observed exact input baseline equality")
        for temporal in (m["temporal_model"], m["temporal_baseline"]):
            if temporal["pairs"] != 21 or temporal["valid_pairs"] != 21:
                raise ValueError("Adjacent temporal support mismatch")

        contract = {k: m["record"][k] for k in ("source_id", "source_sha256", "target_key", "target_pixel_sha256")}
        contract.update({"metric_gate": m["metric_gate"], "lpips": m["lpips"], "flow": m["flow"],
                         "reference_sha256": m["reference"]["file_sha256"], "frame_indices": m["frame_indices"],
                         "frame_stride": m["frame_stride"], "temporal_access": m["temporal_access"]})
        if shared is None:
            shared = contract
        elif shared != contract:
            raise ValueError("Cross-run shared contract mismatch")
        references = {name: values(v["quality"], v["temporal"]) for name, v in m["reference"]["arms"].items()}
        if task not in per_task:
            per_task[task] = {"observation_pixel_sha256": m["record"]["observation_pixel_sha256"],
                              "reference_arms": references, "native_arms": {}}
        elif per_task[task]["reference_arms"] != references:
            raise ValueError("Cross-method reference scores differ")
        model = values(m["model"], m["temporal_model"])
        if any(v["valid_coverage"] != model["valid_coverage"] for v in references.values()):
            raise ValueError("Coverage differs across arms")
        per_task[task]["native_arms"][method] = {
            "metrics": model,
            "clipping": {"component_fraction": m["clipped_component_fraction"],
                         "raw_model_mse": m["raw_model_mse"], "scored_clamped_model_mse": m["model"]["mse"]},
            "checkpoint_sha256": prediction["checkpoint_sha256"],
            "one_shot_model_forward_seconds_not_a_speed_benchmark": prediction["one_shot_forward_seconds_not_a_benchmark"],
            "output_npz_sha256": prediction["output_npz_sha256"],
            "input_baseline_all_numeric_differences_exact_zero": True,
        }
        evidence.append({"method": method, "task": task, "metrics_path": str(path.relative_to(ROOT)),
                         "metrics_sha256": sha(path), "prediction_receipt_sha256": sha(prediction_path)})

    for task, row in per_task.items():
        all_metrics = {**row["reference_arms"], **{k: v["metrics"] for k, v in row["native_arms"].items()}}
        row["paired_native_minus_reference"] = {
            method: {name: {key: all_metrics[method][key] - reference[key]
                            for key in ("psnr", "lpips", "flow_aligned_rgb_error_change_mae")}
                     for name, reference in all_metrics.items() if name != method}
            for method in METHODS}
    summary = {
        "schema": "openh3.native_quality_completed_single_dev_source.v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": True, "source_count": 1, "native_models": list(METHODS), "tasks": list(TASKS),
        "native_completed_clips": 6, "shared_contract": shared, "per_task": per_task,
        "evidence": evidence, "final_queue": queue, "final_queue_file_sha256": sha(queue_path),
        "queue_wall_seconds": queue["finished_unix"] - queue["started_unix"],
        "quality_verdict": "requires_review_single_historical_dev_source_not_SOTA_acceptance",
        "scope_limits": [
            "One previously used development source, not independent frozen test or a multisource ranking.",
            "Aver/Faver use offline22/maxfuture21; parameters and compute are not matched. Rphi-only is causalK3; H3 uses22-frame blocks with different compute.",
            "PSNR is mean frame PSNR. No pooled cross-task ranking: task severity and reference quality differ.",
            "Scores use float32 clamp[0,1] RGB without uint8 quantization. Raw output MSE and clipping fractions are retained; raw LPIPS was not evaluated.",
            "GT-proxy Farneback error-change is not a complete perceptual flicker measure. Prior flow tensor hashes were absent, so exact baseline scores/support are verified without claiming prior flow pixel identity.",
            "Aver uses isolated official architecture-only imports and original CPU operators; this does not verify the full BasicSR training entrypoint.",
            "CPU execution is supported by tensor/model paths and receipts. torch_npu_imported=false is a pre-import/hardcoded inference field and does not prove no subsequent indirect import; warnings or manager handles do not demonstrate NPU inference.",
        ],
        "actions": {"new_model_forwards": 0, "scoring_rerun": False, "old_evidence_redownloaded": False,
                    "runtime_or_queue_modified": False},
    }
    with (OUT / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    fields = ["task", "method", "psnr", "lpips", "flow_aligned_rgb_error_change_mae",
              "mse", "raw_model_mse", "clipped_component_fraction", "frames", "pairs", "valid_pairs", "valid_coverage"]
    with (OUT / "metrics_table.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            row = per_task[task]
            for method, metrics in {**row["reference_arms"], **{k: v["metrics"] for k, v in row["native_arms"].items()}}.items():
                record = {"task": task, "method": method, **{k: metrics[k] for k in fields if k in metrics}}
                if method in row["native_arms"]:
                    clipping = row["native_arms"][method]["clipping"]
                    record.update(raw_model_mse=clipping["raw_model_mse"],
                                  clipped_component_fraction=clipping["component_fraction"])
                writer.writerow(record)
    lines = ["双模型三项 native CPU smoke 已完整结束，6次推理和6次评分均成功。",
             "所有行来自同一历史dev源6UZVovkR3IU、22帧480x832、stride1、21对相邻帧；不能视为独立测试或多源排名。",
             "", "task | input PSNR/LPIPS | Rphi PSNR/LPIPS | Faver PSNR/LPIPS | Aver PSNR/LPIPS | H3-Rphi-anchor PSNR/LPIPS"]
    for task in TASKS:
        row = per_task[task]
        arms = [row["reference_arms"]["input"], row["reference_arms"]["rphi_only"],
                row["native_arms"]["favernet"]["metrics"], row["native_arms"]["avernet"]["metrics"],
                row["reference_arms"]["h3_rphi_anchor"]]
        lines.append(task + " | " + " | ".join(f"{v['psnr']:.6f}/{v['lpips']:.6f}" for v in arms))
    lines += ["", "退步和取舍（native相对input/Rphi）："]
    for task in TASKS:
        for method in METHODS:
            for reference in ("input", "rphi_only"):
                delta = per_task[task]["paired_native_minus_reference"][method][reference]
                worse = [k for k, v in delta.items() if (v < 0 if k == "psnr" else v > 0)]
                if worse:
                    lines.append(f"{task}/{method} vs {reference}: worse={','.join(worse)}, delta={delta}")
    lines += ["", "Clipping（只计RGB分量）："]
    for task in TASKS:
        for method in METHODS:
            v = per_task[task]["native_arms"][method]["clipping"]
            lines.append(f"{task}/{method}: fraction={v['component_fraction']:.9f}, rawMSE={v['raw_model_mse']:.12g}, clampedMSE={v['scored_clamped_model_mse']:.12g}")
    lines += ["", *summary["scope_limits"], "", "本汇总仅读取已归档指标、验证hash/合同；未重跑forward或评分，未修改队列与运行代码。"]
    with (OUT / "review.txt").open("x") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "summary": str(OUT / "summary.json"),
                      "summary_sha256": sha(OUT / "summary.json"), "clips": 6,
                      "queue_wall_seconds": summary["queue_wall_seconds"]}))


if __name__ == "__main__":
    main()
