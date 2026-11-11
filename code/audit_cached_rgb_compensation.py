"""Read-only, standard-library audit of the fixed cached RGB grid; no scoring.

Reads completed metrics and already archived H3/VAE references. Never loads pixels,
models, torch, or a GPU backend. Integrity acceptance is not quality acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

GRID = [0., .05, .1, .25, .5, .75, 1.]
FAMILIES = ["scalar", "vae_compensation"]
TASKS = {"sr", "deblur", "denoise", "compression"}
QUALITY = ("mse", "mae", "psnr", "psnr_from_mean_mse", "lpips")
TEMPORAL = ("unwarped_rgb_error_change_mae", "flow_aligned_rgb_error_change_mae")
LINEAR_SHA = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "missing/invalid SHA256")


def close(a, b, key):
    require(isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b), "nonfinite/missing metric: " + key)
    require(abs(a-b) <= (1e-6 if key == "lpips" else 1e-10), "metric mismatch: " + key)


def validate_score(score, coverage):
    q, t = score["quality"], score["temporal"]
    require(q["frames"] == 22, "must score all 22 frames")
    require(t["pairs"] == t["valid_pairs"] == 21, "all 21 common-flow pairs required")
    close(t["valid_coverage"], coverage, "valid_coverage")
    for name in QUALITY:
        require(isinstance(q.get(name), (int, float)) and math.isfinite(q[name]), "missing/nonfinite " + name)
    for name in TEMPORAL:
        require(isinstance(t.get(name), (int, float)) and math.isfinite(t[name]), "missing/nonfinite " + name)
    require(0 < coverage <= 1, "empty/invalid flow coverage")


def metric_parity(a, b):
    for part, keys in (("quality", QUALITY), ("temporal", TEMPORAL + ("valid_coverage",))):
        for key in keys:
            close(a[part][key], b[part][key], key)
    for part, keys in (("quality", ("frames", "perfect_frames")), ("temporal", ("pairs", "valid_pairs"))):
        for key in keys:
            require(a[part][key] == b[part][key], "count mismatch: " + key)


def flatten(score):
    return {**{key: score["quality"][key] for key in QUALITY},
            **{key: score["temporal"][key] for key in TEMPORAL}}


def subtract(a, b):
    aa, bb = flatten(a), flatten(b)
    return {key: aa[key]-bb[key] for key in aa}


def audit(report, ordinary, controls, vae, reference_hashes):
    require(report["schema"] == "openh3.cached_rgb_compensation.v1" and report.get("complete") is True, "completed supported report required")
    c = report["contract"]
    require(canonical_sha(c) == report["contract_sha256"], "contract receipt mismatch")
    require(c["samples"] == 12 and c["beta_grid"] == GRID and c["families"] == FAMILIES, "wrong sample/fixed-grid contract")
    require(c["geometry"] == [22, 480, 832, 3] and c["frame_stride"] == 1, "wrong frame contract")
    require(c["formula_scalar"] == "clip(R + beta*(G_R-R),0,1)" and c["formula_vae_compensation"] == "clip(R + beta*(G_R-V_R),0,1)", "wrong candidate formula")
    for key in ("H3_calls", "NFE", "train_steps"):
        require(c[key] == report[key] == 0, "new computation outside cache scope")
    for key, value in reference_hashes.items():
        require(c["input_hashes"][key] == value, "reference file hash mismatch: " + key)
    require({k:v for k,v in c["input_hashes"].items() if k != "vae_metrics"} == vae["contract"]["input_hashes"], "VAE/reference input hash chain mismatch")
    for ref in (ordinary, controls, vae):
        require(ref.get("complete") is True, "incomplete reference")
        require(ref["lpips"] == report["lpips"], "LPIPS runtime/weights mismatch")
    lpips = report["lpips"]
    require(lpips["available"] and lpips["device"] == "cpu" and lpips["network"] == "alex" and lpips["downloads"] is False, "local CPU Alex LPIPS required")
    require(c["lpips_weights_sha256"] == lpips["alexnet_sha256"] and c["lpips_linear_weights"]["sha256"] == LINEAR_SHA, "LPIPS weight lock mismatch")
    require(all(not x["NPU_initialized"] and not x["H3_imported"] for x in report["cpu_observations"]) and report["cpu_observations"], "CPU observations failed/missing")
    flow = report["flow"]
    digest(flow["support_sha256"])
    require(flow["pairs"] == 21 and flow["reused_across_all_arms_and_same_source_samples"] is True, "common flow missing")
    refs = {(scope, r["sample_id"]): r for scope, doc in (("ordinary", ordinary), ("controls", controls)) for r in doc["per_sample"]}
    vrefs = {(r["scope"], r["sample_id"]): r for r in vae["per_sample"]}
    rows = report["per_sample"]
    require(len(rows) == len(refs) == len(vrefs) == 12 and set(refs) == set(vrefs), "reference/sample coverage mismatch")
    require({(r["scope"], r["sample_id"]) for r in rows} == set(refs), "sample ID coverage mismatch")
    require(len(report["sample_receipts"]) == 12 and [canonical_sha(r) for r in rows] == report["sample_receipts"], "per-sample receipt mismatch")
    require([r["evidence"] for r in rows] == c["cache_evidence"], "preflight/runtime cache identity mismatch")
    require({r["task"] for r in rows if r["scope"] == "ordinary"} == TASKS and len([r for r in rows if r["scope"] == "ordinary"]) == 4, "ordinary task coverage")
    for control in ("identity", "near_clean"):
        subset = [r for r in rows if r["control"] == control]
        require(len(subset) == 4 and {r["task"] for r in subset} == TASKS, "control task coverage")
    coverage = ordinary["per_sample"][0]["arms"]["input"]["temporal"]["valid_coverage"]
    summary, paired, endpoints = [], [], []
    identity_grid_hashes = {}
    for row in rows:
        key = (row["scope"], row["sample_id"])
        prior, vr, ev = refs[key], vrefs[key], row["evidence"]
        require(ev["sample_id"] == row["sample_id"] and ev["scope"] == row["scope"], "cache sample/scope mismatch")
        require(row["frame_indices"] == prior["frame_indices"] == vr["frame_indices"] == list(range(22)), "frame selection mismatch")
        require(row["task"] == prior["task"] == vr["task"] and row["control"] == prior.get("control") == vr.get("control"), "task/control mismatch")
        require(row["control_recipe"] == prior.get("control_recipe") == vr.get("control_recipe"), "control recipe mismatch")
        require(row["baselines"] == prior["arms"], "reference baseline changed")
        require(row["source_id"] == prior["source_id"] == vr["source_id"], "source ID mismatch")
        require(ev["source_sha256"] == prior["source_sha256"] == vr["source_sha256"], "source SHA mismatch")
        require(ev["record_sha256"] == prior["record_sha256"], "record SHA mismatch")
        require(ev["target_pixel_sha256"] == prior["target_pixel_sha256"] == vr["target_pixel_sha256"] == flow["target_pixel_sha256"], "target mismatch")
        require(prior["flow"] == vr["flow"] == flow["note"], "flow method mismatch")
        require(ev["R_array_sha256"] == prior["sampling"]["guide_pixel_sha256"]["rphi"] == vr["observations"]["rphi"]["guide_pixel_sha256"], "R guide mismatch")
        require(ev["guide_file_sha256"] == vr["guide_file_sha256"], "guide file mismatch")
        require(ev["G_file_sha256"] == prior["decoded_file"]["sha256"] and ev["G_array_sha256"] == prior["decoded_file"]["arrays"]["rphi"], "G cache mismatch")
        require(ev["V_file_sha256"] == vr["cache"]["file_sha256"] and ev["V_array_sha256"] == vr["cache"]["arrays"]["rphi_as_returned_internal_clipped"]["array_sha256"], "V cache mismatch")
        require(ev["V_guide_latent_sha256"] == vr["observations"]["rphi"]["encoded_latent_sha256_float32"] == prior["sampling"]["calls"]["rphi"]["guide_latent_sha256_float32"], "V latent mismatch")
        require(ev["actual_cached_clamp_parity_exact"] and vr["observations"]["rphi"]["clamp_parity_exact"], "V internal clamp parity")
        calls = prior["sampling"]["calls"]
        require(calls["rphi"]["warm_start_latent_sha256_float32"] == calls["raw"]["guide_latent_sha256_float32"], "warm draft changed")
        for name in ("seed", "steps", "sigmas", "cfg"):
            require(calls["raw"]["sampling_record"][name] == calls["rphi"]["sampling_record"][name], "H3 paired setting mismatch")
        for score in row["baselines"].values():
            validate_score(score, coverage)
        validate_score(row["V_baseline"], coverage)
        metric_parity(row["V_baseline"], vr["arms"]["rphi_as_returned_internal_clipped"])
        require([(g["family"], g["beta"]) for g in row["grid"]] == [(f,b) for f in FAMILIES for b in GRID], "grid missing/duplicated/reordered")
        index = {(g["family"],g["beta"]):g for g in row["grid"]}
        for g in row["grid"]:
            score, pre, clip = g["post_final_clip_float32"], g["pre_final_clip_float64"], g["clipping_and_change"]
            validate_score(score, coverage)
            require(pre["frames"] == 22 and pre["lpips"] is None, "preclip scoring contract")
            for metric in QUALITY[:-1]:
                require(math.isfinite(pre[metric]), "nonfinite preclip distortion")
            digest(clip["output_array_sha256"])
            for name in ("clipped_component_fraction", "clipped_pixel_fraction", "changed_component_fraction_vs_R", "changed_pixel_fraction_vs_R", "changed_pixel_fraction_above_1_over_255_vs_R"):
                require(0 <= clip[name] <= 1, "invalid clip/change fraction")
            require(clip["clipped_component_fraction"] <= clip["clipped_pixel_fraction"], "inconsistent pixel clipping fraction")
            require(score["quality"]["mse"] <= pre["mse"] + 1e-9 and score["quality"]["mae"] <= pre["mae"] + 1e-8, "final clipping increased bounded-reference distortion beyond cast tolerance")
            if g["family"] == "scalar":
                require(clip["clipped_component_fraction"] == clip["clipped_pixel_fraction"] == 0, "convex scalar unexpectedly clipped")
            if row["control"] == "identity":
                identity_grid_hashes.setdefault((g["family"],g["beta"]), set()).add(clip["output_array_sha256"])
            summary.append({"scope":row["scope"], "control":row["control"], "task":row["task"], "sample_id":row["sample_id"], "family":g["family"], "beta":g["beta"],
                            "postclip":flatten(score), "preclip":pre, "clipping_and_change":clip,
                            "delta_vs_input":subtract(score, prior["arms"]["input"]), "delta_vs_R":subtract(score, prior["arms"]["rphi_only"]),
                            "post_minus_pre_mse":score["quality"]["mse"]-pre["mse"]})
        for family in FAMILIES:
            g = index[(family,0)]
            require(g["clipping_and_change"]["output_array_sha256"] == ev["R_array_sha256"], "beta0 pixel endpoint mismatch")
            require(g["clipping_and_change"]["change_mse_vs_R"] == 0, "beta0 changed R")
            metric_parity(g["post_final_clip_float32"], prior["arms"]["rphi_only"])
        require(index[("scalar",1)]["clipping_and_change"]["output_array_sha256"] == ev["G_array_sha256"], "scalar beta1 pixel endpoint mismatch")
        metric_parity(index[("scalar",1)]["post_final_clip_float32"], prior["arms"]["h3_rphi"])
        if row["task"] != "sr":
            metric_parity(index[("scalar",.75)]["post_final_clip_float32"], prior["arms"]["h3_rphi_anchor"])
        else:
            require(row["reproduction_checks"]["scalar_beta075_native_alpha025"]["applicable"] is False, "SR cannot claim full-RGB anchor parity")
        for check, result in row["reproduction_checks"].items():
            if result.get("applicable") is not False:
                require(result["passed"] is True, "runtime endpoint failure: " + check)
        endpoints.append({"sample_id":row["sample_id"], "beta0_R_pixel_hash_exact_both_families":True, "scalar_beta1_G_pixel_hash_exact":True,
                          "scalar_beta075_native_anchor_metric_parity":row["task"] != "sr", "compensation_beta1_is_G_endpoint":False})
        for beta in GRID:
            a,b = index[("vae_compensation",beta)],index[("scalar",beta)]
            paired.append({"scope":row["scope"],"control":row["control"],"task":row["task"],"sample_id":row["sample_id"],"beta":beta,
                           "compensation_minus_scalar":subtract(a["post_final_clip_float32"], b["post_final_clip_float32"]),
                           "compensation_change_mse_vs_R":a["clipping_and_change"]["change_mse_vs_R"],
                           "scalar_change_mse_vs_R":b["clipping_and_change"]["change_mse_vs_R"]})
    require(sorted({r["source_id"] for r in rows}) == c["source_ids"] and len(c["source_ids"]) == 1, "one-source scope mismatch")
    require(all(len(values) == 1 for values in identity_grid_hashes.values()), "identity label replicas differ")
    groups = {}
    for group in ("ordinary", "identity", "near_clean"):
        for beta in GRID:
            selected = [r for r in paired if r["beta"] == beta and (r["scope"] == "ordinary" if group == "ordinary" else r["control"] == group)]
            groups[group+"/"+str(beta)] = {"label_records":len(selected), "unique_sources":1,
                "compensation_better_than_scalar_label_count":{k:sum(r["compensation_minus_scalar"][k] < -1e-10 for r in selected) for k in ("mse","mae","lpips","flow_aligned_rgb_error_change_mae")},
                "warning":"counts are task labels/draws within one source, not independent source wins"}
    return {"schema":"openh3.cached_compensation_integrity_audit.v1", "complete":True, "integrity_verdict":"passed", "quality_verdict":"requires_review",
            "selected_beta":None,"selection_performed":False,"evaluation_rerun":False,"source_ids":c["source_ids"],"samples":12,"grid_records":len(summary),
            "flow":flow,"lpips":lpips,"endpoint_checks":endpoints,"identity_label_duplicates_exact":True,
            "metric_direction":"PSNR higher is better; MSE/MAE/LPIPS/error-change lower is better. Deltas are signed candidate minus comparator.",
            "psnr_semantics":"psnr is mean per-frame PSNR; psnr_from_mean_mse is pooled MSE PSNR; they are not interchangeable",
            "summary":summary,"same_beta_comparisons":paired,"descriptive_counts":groups,
            "limits":["All 12 records come from one development source; identity four labels are exact duplicates; near-clean uses four task-keyed draws.",
                "Same beta is not matched repair magnitude; both candidates' change MSE versus R is reported.",
                "G and V are as returned after VAE internal clipping; pre/post clipping here refers only to the final RGB blend.",
                "Compensation beta1 is clip(R+G-V), not G; scalar .75 matches full-RGB native alpha.25 anchor, not SR Q4 anchoring.",
                "Current grid shares a hashed flow support. Historical H3/VAE has matching method/coverage and reproduced endpoint metrics, not a historical flow tensor hash.",
                "No new pixel/model computation in this audit. Cache identity relies on recorded runtime file/array checks and archive hashes.",
                "Raw warm start stays fixed; subtracting VAE(R) is a diagnostic, not independent sampler-error attribution.",
                "No coefficient selected, no deployment or generalization claim, no semantic/factual preservation measurement."]}


def audit_small_bundle(directory, report):
    """Verify archived runtime/launch provenance; no remote or large-cache reads."""
    manifest = json.loads((directory/"files_sha256.json").read_text())
    for name, entry in manifest["files"].items():
        require(Path(name).name == name, "unsafe archive member path")
        path = directory/name
        require(path.stat().st_size == entry["bytes"] and sha_file(path) == entry["sha256"], "archive member mismatch: "+name)
    launch = json.loads((directory/"launch_admission.json").read_text())
    for key,name in (("config_sha256","launch_config.json"), ("launcher_sha256","launch_cpu_grid.py"),
                     ("runtime_manifest_sha256","runtime_manifest.json"), ("inspection_receipt_sha256","inspection_receipt.json")):
        require(sha_file(directory/name) == launch[key], "launch hash mismatch: "+name)
    runtime = json.loads((directory/"runtime_manifest.json").read_text())
    for name,value in report["contract"]["implementation_sha256"].items():
        require(runtime["files"]["code/"+name] == value, "runtime implementation not pinned: "+name)
    source = directory/"diagnose_cached_rgb_compensation.py"
    require(sha_file(source) == report["contract"]["implementation_sha256"][source.name], "archived running source mismatch")
    require("EXIT_CODE:0" in (directory/"queue.log").read_text().splitlines(), "queue did not report success")
    return {"verified_small_members":len(manifest["files"]), "archive_sha256":sha_file(directory/"evidence.tar.gz"),
            "metrics_mtime_utc":manifest["metrics_mtime_utc"], "started_at_utc":launch["started_at_utc"],
            "runtime_code_manifest_bound":True,"launch_pins_verified":True,"queue_exit_code":0,
            "large_caches_downloaded_or_rescored_by_audit":False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("metrics", "ordinary-metrics", "controls-metrics", "vae-metrics", "out"):
        ap.add_argument("--"+name, required=True, type=Path)
    args = ap.parse_args(argv)
    require(not args.out.exists(), "refuse existing audit output")
    paths = {k:getattr(args,k) for k in ("metrics", "ordinary_metrics", "controls_metrics", "vae_metrics")}
    docs = {k:json.loads(p.read_text()) for k,p in paths.items()}
    result = audit(docs["metrics"], docs["ordinary_metrics"], docs["controls_metrics"], docs["vae_metrics"],
                   {k:sha_file(p) for k,p in paths.items() if k != "metrics"})
    result["input_files"] = {k:{"path":str(p.resolve()), "sha256":sha_file(p)} for k,p in paths.items()}
    result["small_bundle"] = audit_small_bundle(args.metrics.parent, docs["metrics"])
    result["audit_implementation_sha256"] = sha_file(__file__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("complete","integrity_verdict","quality_verdict","samples","grid_records","selected_beta")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
