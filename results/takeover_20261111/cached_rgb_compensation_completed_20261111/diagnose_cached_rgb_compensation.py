"""Fixed cached RGB compensation grid. CPU only; no model generation or training.

R and G are the existing clipped float32 Rphi guide and H3-Rphi RGB. V is the
exact guide-matched VAE roundtrip AS RETURNED after its internal clamp.
Targets never enter candidate construction. No coefficient is selected.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from diagnose_rphi_residual_mix import GRID, array_sha, canonical_sha, require_hash, safe_child
from eval_rphi import flow_support, load_local_lpips, rgb_metrics, sha_file, temporal_metrics

SCHEMA = "openh3.cached_rgb_compensation.v1"
FAMILIES = ("scalar", "vae_compensation")
GEOMETRY = [22, 480, 832, 3]
LPIPS_LINEAR_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"


def construct_candidate(restored, generated, vae, beta, family):
    """No target argument. Float64 arithmetic followed by clip and float32 cast."""
    if family not in FAMILIES or beta not in GRID:
        raise ValueError("only predeclared family and beta grid are allowed")
    if any(x.dtype != np.float32 or x.shape != restored.shape or x.ndim != 4 or x.shape[-1] != 3
           or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1 for x in (restored, generated, vae)):
        raise ValueError("candidate inputs must be matching finite clipped float32 THWC")
    r, g = restored.astype(np.float64), generated.astype(np.float64)
    reference = r if family == "scalar" else vae.astype(np.float64)
    pre = r + float(beta) * (g - reference)
    post = np.clip(pre, 0, 1).astype(np.float32)
    outside = (pre < 0) | (pre > 1)
    changed = post != restored
    delta = post.astype(np.float64) - r
    stats = {"pre_final_clip_range": [float(pre.min()), float(pre.max())],
             "clipped_component_fraction": float(outside.mean()),
             "clipped_pixel_fraction": float(outside.any(-1).mean()),
             "changed_component_fraction_vs_R": float(changed.mean()),
             "changed_pixel_fraction_vs_R": float(changed.any(-1).mean()),
             "changed_pixel_fraction_above_1_over_255_vs_R": float((np.abs(delta) > 1/255).any(-1).mean()),
             "max_abs_change_vs_R": float(np.abs(delta).max()),
             "change_mse_vs_R": float(np.square(delta).mean()),
             "clip_and_float32_cast_change_mse": float(np.square(post.astype(np.float64) - pre).mean()),
             "output_array_sha256": array_sha(post)}
    if beta == 0 and not np.array_equal(post, restored):
        raise ValueError("beta0 must exactly reproduce R")
    if family == "scalar" and beta == 1 and not np.array_equal(post, generated):
        raise ValueError("scalar beta1 must exactly reproduce G")
    return pre, post, stats


def verify_cached_npz(path, expected_file_sha, expected_arrays):
    """Verify every declared array and the enclosing file before and after read."""
    require_hash(path, expected_file_sha)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_arrays):
            raise ValueError("cache array coverage mismatch")
        arrays = {}
        for name, expected in expected_arrays.items():
            a = archive[name]
            digest = expected if isinstance(expected, str) else expected["array_sha256"]
            if array_sha(a) != digest or a.dtype != np.float32 or list(a.shape) != GEOMETRY or not np.isfinite(a).all():
                raise ValueError("cache array SHA/dtype/geometry mismatch: " + name)
            if isinstance(expected, dict) and (expected["shape"] != list(a.shape) or expected["dtype"] != str(a.dtype)):
                raise ValueError("cache metadata mismatch: " + name)
            arrays[name] = a
    require_hash(path, expected_file_sha)
    return arrays


def validate_vae_row(row, item):
    prior, exported = item["reference"], item["export"]
    for key in ("sample_id", "source_id", "source_sha256", "task", "target_pixel_sha256"):
        if row[key] != prior[key] or row[key] != exported[key]:
            raise ValueError("VAE/source binding mismatch: " + key)
    if row["scope"] != item["scope"] or row.get("control") != exported.get("control"):
        raise ValueError("VAE scope/control mismatch")
    if (row["guide_file_sha256"] != exported["guide_file_sha256"] or not row["baseline_metric_parity"]
            or row["frame_indices"] != list(range(22)) or row.get("control_recipe") != exported.get("control_recipe")):
        raise ValueError("VAE guide/recipe/frame binding mismatch")
    for label in ("raw", "rphi"):
        o = row["observations"][label]
        sampling = prior["sampling"]
        if (not o["latent_matches_prior_H3_exactly"] or not o["clamp_parity_exact"] or o["clamp_parity_max_abs_error"] != 0
                or o["guide_pixel_sha256"] != sampling["guide_pixel_sha256"][label]
                or o["encoded_latent_sha256_float32"] != sampling["calls"][label]["guide_latent_sha256_float32"]
                or o["encoded_latent_shape"] != [1, 24, 7, 30, 52]):
            raise ValueError("VAE exact guide/latent/clamp parity required")


def load_cached_inputs(item, args, row):
    validate_vae_row(row, item)
    exported, prior = item["export"], item["reference"]
    loaded = item["dataset"].load(item["index"])
    require_hash(item["guide_file"], exported["guide_file_sha256"])
    with np.load(item["guide_file"], allow_pickle=False) as archive:
        restored = archive["rgb"].copy()
    if array_sha(restored) != exported["arrays"]["rgb"]["array_sha256"]:
        raise ValueError("Rphi guide pixel SHA mismatch")
    require_hash(item["guide_file"], exported["guide_file_sha256"])
    gp = safe_child(getattr(args, item["scope"] + "_metrics").parent, prior["decoded_file"]["path"])
    generated_arrays = verify_cached_npz(gp, prior["decoded_file"]["sha256"], prior["decoded_file"]["arrays"])
    vp = safe_child(args.vae_metrics.parent, row["cache"]["path"])
    vae_arrays = verify_cached_npz(vp, row["cache"]["file_sha256"], row["cache"]["arrays"])
    for label in ("raw", "rphi"):
        a, b = vae_arrays[label + "_pre_internal_clamp"], vae_arrays[label + "_as_returned_internal_clipped"]
        if not np.array_equal(a.clip(0, 1), b):
            raise ValueError("actual VAE cached preclamp/returned parity failed")
    evidence = {"sample_id": row["sample_id"], "scope": item["scope"],
        "source_sha256": row["source_sha256"], "target_pixel_sha256": row["target_pixel_sha256"],
        "record_sha256": exported["record_sha256"], "guide_file_sha256": exported["guide_file_sha256"],
        "R_array_sha256": array_sha(restored), "G_file_sha256": prior["decoded_file"]["sha256"],
        "G_array_sha256": prior["decoded_file"]["arrays"]["rphi"],
        "V_file_sha256": row["cache"]["file_sha256"],
        "V_array_sha256": row["cache"]["arrays"]["rphi_as_returned_internal_clipped"]["array_sha256"],
        "V_guide_latent_sha256": row["observations"]["rphi"]["encoded_latent_sha256_float32"],
        "actual_cached_clamp_parity_exact": True}
    return restored, generated_arrays["rphi"], vae_arrays["rphi_as_returned_internal_clipped"], loaded["target"], evidence


def prepare(args):
    # Existing CPU-only inspectors verify source materialization receipts, every
    # guide/sidecar hash, causal K3 history, control recipes and fixed-H3 contract.
    from run_vae_roundtrip_diagnostic import inspect_bundle
    items, hashes, _ = inspect_bundle(args)
    vae = json.loads(args.vae_metrics.read_text())
    if (not vae.get("complete") or vae["contract"]["input_hashes"] != hashes
            or vae["contract"]["samples"] != 12 or vae["contract"]["frame_stride"] != 1
            or vae["contract"]["geometry"] != GEOMETRY or vae["contract"]["NFE"] != 0
            or not vae["freeze_audit"]["VAE_parameter_versions_unchanged"]):
        raise ValueError("completed, frozen, exact-reference VAE diagnostic required")
    hashes["vae_metrics"] = sha_file(args.vae_metrics)
    ordinary_report = json.loads(args.ordinary_metrics.read_text())
    if vae["lpips"] != ordinary_report["lpips"]:
        raise ValueError("VAE/prior H3 LPIPS contract mismatch")
    vae_rows = {(r["scope"], r["sample_id"]): r for r in vae["per_sample"]}
    if len(vae_rows) != 12 or set(vae_rows) != {(i["scope"],i["export"]["sample_id"]) for i in items}:
        raise ValueError("exact ordinary4/control8 VAE coverage required")
    evidence = []
    for i in items:
        *_, ev = load_cached_inputs(i, args, vae_rows[(i["scope"],i["export"]["sample_id"])])
        evidence.append(ev)
        del _
        print(json.dumps({"preflight_sample":len(evidence),"total":12,"cache_hashes_verified":True}),flush=True)
    for name, digest in hashes.items():
        require_hash(getattr(args,name), digest)
    implementation = {str(p.relative_to(Path(__file__).parent)):sha_file(p)
        for p in Path(__file__).parent.rglob("*.py") if "__pycache__" not in p.parts}
    contract = {"input_hashes": hashes, "implementation_sha256":implementation,
        "samples":12,"source_ids":sorted({i["export"]["source_id"] for i in items}),
        "beta_grid":list(GRID),"families":list(FAMILIES),"geometry":GEOMETRY,"frame_stride":1,
        "formula_scalar":"clip(R + beta*(G_R-R),0,1)",
        "formula_vae_compensation":"clip(R + beta*(G_R-V_R),0,1)",
        "V_semantics":"as_returned_internal_clipped VAE roundtrip of exact Rphi guide",
        "G_semantics":"as_returned_internal_clipped Rphi-conditioned H3 with unchanged raw warm start",
        "arithmetic":"float64 mixing; float32 output after final clamp",
        "GT_usage":"integrity/geometry and scoring only; no target argument in construct_candidate",
        "metric":"same eval_rphi; LPIPS on postclip float32 only; preclip RGB distortion reported separately",
        "selection":"all fixed coefficients reported; no oracle or deployment coefficient selected",
        "scope":"one-source development diagnostic; identity repeats and near-clean task-keyed draws are not independent sources",
        "H3_calls":0,"NFE":0,"train_steps":0,"cpu_threads":args.threads,
        "cache_evidence":evidence}
    return items, vae_rows, vae, contract


def parity(actual, expected, label):
    diffs = {}
    for part, keys in (("quality",("mse","mae","psnr","psnr_from_mean_mse","lpips")),
                       ("temporal",("unwarped_rgb_error_change_mae","flow_aligned_rgb_error_change_mae","valid_coverage"))):
        for k in keys:
            a,b = actual[part][k],expected[part][k]
            if a is None or b is None:
                raise ValueError("parity requires available metrics: " + k)
            diff = abs(a-b)
            if diff > (1e-6 if k=="lpips" else 1e-10):
                raise ValueError(label+" reproduction mismatch: "+k+" "+str(diff))
            diffs[part+"/"+k] = diff
    return {"passed":True,"absolute_differences":diffs}


def score_grid(restored, generated, vae, target, lpips_metric, support, prior, vae_reference):
    if target.dtype != np.uint8 or target.shape != restored.shape:
        raise ValueError("matching uint8 metric reference required")
    gt = np.ascontiguousarray(target,dtype=np.float32)/np.float32(255)
    if support is None or len(support) != len(gt)-1:
        raise ValueError("complete common flow support required")
    scored, memo = [], {}
    for family in FAMILIES:
        for beta in GRID:
            pre, post, stats = construct_candidate(restored, generated, vae, beta, family)
            digest = stats["output_array_sha256"]
            if digest not in memo:
                memo[digest] = {"quality":rgb_metrics(post,gt,lpips_metric),
                                "temporal":temporal_metrics(post,gt,support)}
            score = memo[digest]
            if score["temporal"].get("valid_pairs") != len(gt)-1:
                raise ValueError("all 21 flow pairs must have valid support")
            scored.append({"family":family,"beta":beta,"pre_final_clip_float64":rgb_metrics(pre,gt),
                           "post_final_clip_float32":score,"clipping_and_change":stats})
    index = {(r["family"],r["beta"]):r for r in scored}
    checks = {"scalar_beta0":parity(index[("scalar",0)]["post_final_clip_float32"],prior["arms"]["rphi_only"],"beta0"),
              "compensation_beta0":parity(index[("vae_compensation",0)]["post_final_clip_float32"],prior["arms"]["rphi_only"],"compensation beta0"),
              "scalar_beta1":parity(index[("scalar",1)]["post_final_clip_float32"],prior["arms"]["h3_rphi"],"scalar beta1")}
    if prior["task"] != "sr":
        checks["scalar_beta075_native_alpha025"] = parity(index[("scalar",.75)]["post_final_clip_float32"],prior["arms"]["h3_rphi_anchor"],"native anchor")
    else:
        checks["scalar_beta075_native_alpha025"] = {"applicable":False,"reason":"SR uses canonical Q4 anchor, not scalar full-RGB mixing"}
    vscore = {"quality":rgb_metrics(vae,gt,lpips_metric),"temporal":temporal_metrics(vae,gt,support)}
    checks["V_matches_original_diagnostic"] = parity(vscore,vae_reference["arms"]["rphi_as_returned_internal_clipped"],"V")
    return {"grid":scored,"reproduction_checks":checks,"V_baseline":vscore}


def open_result(out, contract, reuse_partial=None, reuse_sha256=None):
    """Always a new output; explicit hash-bound reuse never changes the old run."""
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    digest = canonical_sha(contract)
    previous = None
    if reuse_partial is not None:
        if not reuse_sha256 or reuse_partial.name != "partial_metrics.json":
            raise ValueError("reuse requires explicit partial file SHA")
        require_hash(reuse_partial, reuse_sha256)
        owner=json.loads((reuse_partial.parent/"run_owner.json").read_text())
        previous=json.loads(reuse_partial.read_text())
        if (owner["schema"] != SCHEMA or owner["contract_sha256"] != digest
                or previous["owner_id"] != owner["owner_id"] or previous["contract"] != contract
                or previous["contract_sha256"] != digest):
            raise ValueError("reuse contract mismatch")
        expected=[(e["scope"],e["sample_id"]) for e in contract["cache_evidence"]]
        actual=[(e["scope"],e["sample_id"]) for e in previous["per_sample"]]
        if actual != expected[:len(actual)] or len(actual)>12:
            raise ValueError("reuse completed sample order/coverage mismatch")
        if [canonical_sha(e) for e in previous["per_sample"]] != previous["sample_receipts"]:
            raise ValueError("reuse partial sample digest mismatch")
        for row,evidence in zip(previous["per_sample"],contract["cache_evidence"]):
            if row["evidence"] != evidence or [(g["family"],g["beta"]) for g in row["grid"]] != [(f,b) for f in FAMILIES for b in GRID]:
                raise ValueError("reuse completed sample evidence/grid mismatch")
        require_hash(reuse_partial,reuse_sha256)
    elif reuse_sha256 is not None:
        raise ValueError("reuse SHA requires a source partial file")
    out.mkdir(parents=True,exist_ok=False)
    owner={"schema":SCHEMA,"owner_id":str(uuid.uuid4()),"contract_sha256":digest}
    (out/"run_owner.json").write_text(json.dumps(owner,indent=2)+'\n')
    result={**owner,"complete":False,"contract":contract,"per_sample":[],"sample_receipts":[],
            "limitations":["One development source only; no coefficient selected.",
              "Same raw warm start remains in H3; subtracting VAE(R) is not an isolated causal sampler correction.",
              "G and V already have internal VAE clipping; final blend clipping is separate.",
              "No MSE additivity or independent component attribution is claimed."]}
    if previous is not None:
        for key in ("per_sample","sample_receipts","flow"):
            if key in previous:
                result[key]=previous[key]
        result["explicit_reuse"]={"path":str(reuse_partial.resolve()),"sha256":reuse_sha256,
                                  "samples":len(result["per_sample"]),"original_run_modified":False}
    return result


def cpu_observation(torch, phase):
    initialized=bool(hasattr(torch,"npu") and torch.npu.is_initialized())
    if initialized or "h3_comfy" in sys.modules:
        raise RuntimeError("CPU-only execution violated")
    return {"phase":phase,"torch_npu_in_sys_modules":"torch_npu" in sys.modules,
            "NPU_initialized":initialized,"H3_imported":"h3_comfy" in sys.modules,
            "scope":"point-in-time module/backend observation, not continuous import tracing"}


def accept_common_flow(previous, candidate):
    if previous is not None and any(previous[k] != candidate[k] for k in (
            "support_sha256", "target_pixel_sha256", "pairs")):
        raise ValueError("reused partial common flow support changed")
    return candidate


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("ordinary-metrics","ordinary-guides","controls-metrics","controls-guides","dev-manifest","pretrained-manifest","vae-metrics","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--lpips-weights",type=Path)
    ap.add_argument("--threads",type=int,choices=(1,2),default=2)
    ap.add_argument("--inspect-only",action="store_true")
    ap.add_argument("--reuse-partial",type=Path)
    ap.add_argument("--reuse-partial-sha256")
    args=ap.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise RuntimeError("set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before starting Python")
    import torch
    import cv2
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.threads)
    cpu_checks=[cpu_observation(torch,"before_preflight")]
    begin=time.monotonic()
    items,vae_rows,vae_report,contract=prepare(args)
    cpu_checks.append(cpu_observation(torch,"after_preflight"))
    lpips_path=args.lpips_weights or Path(vae_report["lpips"]["weights_path"])
    require_hash(lpips_path,vae_report["lpips"]["alexnet_sha256"])
    contract["lpips_weights_sha256"]=sha_file(lpips_path)
    lpips_spec=importlib.util.find_spec("lpips")
    if lpips_spec is None or lpips_spec.origin is None:
        raise RuntimeError("local LPIPS package required")
    linear_path=Path(lpips_spec.origin).parent/"weights/v0.1/alex.pth"
    require_hash(linear_path,LPIPS_LINEAR_SHA256)
    contract["lpips_linear_weights"]={"path":str(linear_path),"sha256":LPIPS_LINEAR_SHA256,
                                     "bytes":linear_path.stat().st_size,"version":"0.1"}
    contract["runtime"]={"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"opencv":cv2.__version__}
    if args.inspect_only:
        print(json.dumps({"complete":True,"inspection_only":True,"contract":contract,
                          "contract_sha256":canonical_sha(contract),"H3_imported": "h3_comfy" in sys.modules,
                          "cpu_observations":cpu_checks,
                          "elapsed_seconds":time.monotonic()-begin}),flush=True)
        return 0
    from rphi_h3_controls import write_new,write_partial_atomic
    result=open_result(args.out,contract,args.reuse_partial,args.reuse_partial_sha256)
    result["cpu_observations"]=cpu_checks
    with (args.out/"execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.out/"metrics.json").exists():
            raise FileExistsError("completed run cannot be resumed")
        write_partial_atomic(args.out/"partial_metrics.json",result)
        try:
            metric,info=load_local_lpips(lpips_path)
            if metric is None or any(info[k]!=vae_report["lpips"][k] for k in ("available","network","device","alexnet_sha256","downloads")):
                raise ValueError("same local CPU LPIPS required")
            if getattr(metric,"version",None) != "0.1":
                raise ValueError("LPIPS linear calibration version mismatch")
            require_hash(linear_path,LPIPS_LINEAR_SHA256)
            result["lpips"]=info
            cpu_checks.append(cpu_observation(torch,"after_LPIPS_CPU_load"))
            common_support=None
            for item in items[len(result["per_sample"]):]:
                start=time.monotonic()
                row=vae_rows[(item["scope"],item["export"]["sample_id"])]
                r,g,v,target,evidence=load_cached_inputs(item,args,row)
                if evidence != contract["cache_evidence"][len(result["per_sample"])]:
                    raise ValueError("cache identity changed after preflight")
                if common_support is None:
                    common_support,note=flow_support(target.astype(np.float32)/np.float32(255))
                    if common_support is None:
                        raise RuntimeError("required common flow unavailable: "+note)
                    candidate_flow={"note":note,"support_sha256":canonical_sha([
                        [array_sha(f),array_sha(m)] for f,m in common_support]),"pairs":len(common_support),
                        "target_pixel_sha256":row["target_pixel_sha256"],"reused_across_all_arms_and_same_source_samples":True}
                    result["flow"]=accept_common_flow(result.get("flow"),candidate_flow)
                elif result["flow"]["target_pixel_sha256"] != row["target_pixel_sha256"]:
                    raise ValueError("common flow target mismatch")
                scored=score_grid(r,g,v,target,metric,common_support,item["reference"],row)
                entry={"scope":item["scope"],"sample_id":row["sample_id"],"source_id":row["source_id"],
                    "task":row["task"],"control":row.get("control"),"control_recipe":row.get("control_recipe"),
                    "evidence":evidence,"frame_indices":list(range(22)),"baselines":item["reference"]["arms"],
                    **scored,"seconds":time.monotonic()-start}
                cpu_checks.append(cpu_observation(torch,"after_sample_"+str(len(result["per_sample"])+1)))
                result["per_sample"].append(entry)
                result["sample_receipts"].append(canonical_sha(entry))
                write_partial_atomic(args.out/"partial_metrics.json",result)
                print(json.dumps({"done":len(result["per_sample"]),"total":12,"sample_id":row["sample_id"],"seconds":entry["seconds"]}),flush=True)
            for name,digest in contract["input_hashes"].items():
                require_hash(getattr(args,name),digest)
            require_hash(lpips_path,contract["lpips_weights_sha256"])
            require_hash(linear_path,LPIPS_LINEAR_SHA256)
            for name,digest in contract["implementation_sha256"].items():
                require_hash(Path(__file__).parent/name,digest)
            if len(result["per_sample"]) != 12:
                raise ValueError("incomplete final sample coverage")
            result.update(complete=True,elapsed_seconds_this_invocation=time.monotonic()-begin,
                          H3_calls=0,NFE=0,train_steps=0)
            write_new(args.out/"metrics.json",result)
        except BaseException as error:
            write_new(args.out/("failure-"+str(time.time_ns())+".json"),{
                "complete":False,"completed_samples":len(result["per_sample"]),"error":repr(error)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
