"""Pack/verify small VAE probe evidence; stdlib only, no tensors or model calls.

`pack` reads terminal run artifacts and writes a NEW archive without changing the
run. `audit` validates the externally pinned archive SHA and every manifest byte
SHA, launch bindings, numerical attestations, and optional exact prior-gradient
comparison. A full decoder gate never means full DiT/LoRA training fits.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

MAX_SMALL_BYTES = 16 << 20
CONTRACT_SCHEMA = "openh3.vae_decode_input_gradient_probe.v1"
PAIR_SCHEMA = "openh3.vae_decode_input_gradient_probe.v2"
SCALED_SINGLE_SCHEMA = "openh3.vae_decode_input_gradient_probe.v3"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    require(isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value), "invalid SHA256")


def safe_name(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p)==name and name not in ("", "."), "unsafe evidence member path")
    return name


def read_archive(path, expected_sha):
    valid_sha(expected_sha)
    raw = Path(path).read_bytes()
    require(len(raw)<=MAX_SMALL_BYTES and sha(raw)==expected_sha, "archive byte size/SHA mismatch")
    files, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = safe_name(member.name.rstrip("/") if member.isdir() else member.name)
            if member.isdir():
                continue
            require(member.isfile() and name not in files, "links/special/duplicate archive members forbidden")
            total += member.size
            require(total<=MAX_SMALL_BYTES, "archive uncompressed evidence exceeds 16MiB")
            files[name] = archive.extractfile(member).read()
    candidates = [n for n in files if n.endswith("/evidence_manifest.json")]
    require(len(candidates)==1, "exactly one evidence_manifest.json required")
    manifest_name = candidates[0]
    manifest = json.loads(files[manifest_name])
    require(set(manifest)==set(files)-{manifest_name}, "archive/manifest member coverage mismatch")
    for name, entry in manifest.items():
        safe_name(name)
        require(len(files[name])==entry["bytes"] and sha(files[name])==entry["sha256"], "manifest member mismatch: "+name)
    return files, {"archive_sha256":expected_sha,"manifest_member":manifest_name,
                   "manifest_sha256":sha(files[manifest_name]),"verified_files":len(manifest),"uncompressed_bytes":total}


def exactly_one(files, suffix):
    found = [n for n in files if n.endswith(suffix)]
    require(len(found)==1, "expected exactly one "+suffix)
    return found[0]


def option(argv, flag, default=None):
    if flag not in argv:
        return default
    require(argv.count(flag)==1 and argv.index(flag)+1<len(argv), "invalid duplicated/missing CLI option: "+flag)
    return argv[argv.index(flag)+1]


def gradient_diagnostic_state(stats, shape):
    """Validate counts without conflating missing, nonfinite and all-zero."""
    require(type(stats.get("present")) is bool,"gradient presence must be explicit")
    if not stats["present"]:
        return "missing"
    elements=math.prod(shape)
    require(stats["shape"]==shape and stats["elements"]==elements and stats["dtype"]=="torch.float32","gradient diagnostic geometry/dtype mismatch")
    counts=("zero_count","finite_count","nan_count","inf_count","finite_nonzero_count")
    require(all(type(stats[k]) is int and 0<=stats[k]<=elements for k in counts),"invalid gradient diagnostic counts")
    require(stats["finite_count"]+stats["nan_count"]+stats["inf_count"]==elements,"gradient finite/nonfinite accounting mismatch")
    require(stats["zero_count"]+stats["finite_nonzero_count"]==stats["finite_count"],"gradient zero/nonzero accounting mismatch")
    require(abs(stats["zero_fraction"]-stats["zero_count"]/elements)<=1e-7,"gradient zero fraction mismatch")
    valid_sha(stats["sha256_float32"])
    require(math.isfinite(stats["finite_l2"]) and stats["finite_l2"]>=0,"finite-only gradient norm invalid")
    if stats["finite_count"]:
        require(stats["finite_max_abs"] is not None and math.isfinite(stats["finite_max_abs"]) and stats["finite_max_abs"]>=0,"finite gradient maximum invalid")
    else:
        require(stats["finite_max_abs"] is None,"no finite gradient but finite max recorded")
    if stats["finite_nonzero_count"]:
        require(stats["finite_min_abs_nonzero"] is not None and 0<stats["finite_min_abs_nonzero"]<=stats["finite_max_abs"],"nonzero gradient range invalid")
    else:
        require(stats["finite_min_abs_nonzero"] is None and stats["finite_l2"]==0,"zero-only gradient statistics inconsistent")
    if stats["finite_count"]!=elements:
        return "nonfinite"
    return "finite_nonzero" if stats["finite_nonzero_count"] else "all_zero"


def loss_scale_pair_summary(report, shape, scale):
    diagnostics=report.get("gradient_diagnostics",{})
    require(set(diagnostics)<= {"unscaled_loss_backward","scaled_loss_backward_raw","scaled_loss_backward_unscaled_float32"},"unexpected gradient diagnostic field")
    states={name:gradient_diagnostic_state(stats,shape) for name,stats in diagnostics.items()}
    if report["actual_completed_backwards"]>=1:
        require("unscaled_loss_backward" in states,"first backward gradient diagnostic missing")
    if report["actual_completed_backwards"]>=2:
        require("scaled_loss_backward_raw" in states,"second backward raw diagnostic missing")
    result={"scale":scale,"states":states,"statistics":diagnostics,
            "same_forward_two_backwards":report["actual_completed_forwards"]==2 and report["actual_completed_backwards"]==2,
            "interpretation":"diagnostic incomplete or scaled gradient invalid"}
    if report.get("complete") is True:
        require(states.get("scaled_loss_backward_raw")==states.get("scaled_loss_backward_unscaled_float32")=="finite_nonzero","successful diagnostic requires finite nonzero scaled and FP32-unscaled gradient")
        raw=diagnostics["scaled_loss_backward_raw"];unscaled=diagnostics["scaled_loss_backward_unscaled_float32"]
        require(raw["finite_nonzero_count"]==unscaled["finite_nonzero_count"] and raw["zero_count"]==unscaled["zero_count"],"FP32 unscale changed gradient zero support")
        for key in ("finite_l2","finite_max_abs","finite_min_abs_nonzero"):
            require(math.isclose(raw[key]/scale,unscaled[key],rel_tol=1e-6,abs_tol=1e-20),"scaled/unscaled gradient magnitude mismatch: "+key)
        g=report["latent_gradient"]
        require(g["sha256_float32"]==unscaled["sha256_float32"] and g["nonzero_count"]==unscaled["finite_nonzero_count"],"final gradient differs from FP32-unscaled statistics")
        require(g["l2"]==unscaled["finite_l2"] and g["max_abs"]==unscaled["finite_max_abs"],"final gradient magnitude differs from FP32-unscaled statistics")
        baseline=states["unscaled_loss_backward"]
        result["interpretation"]={"all_zero":"scale_recovers_nonzero_gradient_supports_numerical_scale_sensitivity",
            "finite_nonzero":"both_backward_gradients_nonzero_compare_quantization_and_support",
            "nonfinite":"scaled_gradient_finite_after_nonfinite_baseline_requires_investigation",
            "missing":"scaled_gradient_present_after_missing_baseline_requires_investigation"}[baseline]
        result["unscaled_gradient_matches_first_backward_sha"]=(diagnostics["unscaled_loss_backward"].get("sha256_float32")==unscaled["sha256_float32"])
    result["limits"]=["No proof of exact mathematical gradients from recovery of nonzero values.",
        "Does not uniquely isolate RGB mean reduction from intermediate FP16 gradient arithmetic.",
        "Retained-graph two-backward peak is not a single-backward training memory measurement."]
    return result


def scaled_single_summary(report, shape, scale):
    """Separate v3 single-backward statistics; never fabricate a paired baseline."""
    diagnostics=report.get("gradient_diagnostics",{})
    require(set(diagnostics)<= {"scaled_loss_backward_raw","scaled_loss_backward_unscaled_float32"},"single scaled diagnostic must not contain an unscaled first-backward baseline")
    states={name:gradient_diagnostic_state(stats,shape) for name,stats in diagnostics.items()}
    if report["actual_completed_backwards"]:
        require("scaled_loss_backward_raw" in states,"single scaled raw diagnostic missing")
    if report.get("complete"):
        require(set(states)=={"scaled_loss_backward_raw","scaled_loss_backward_unscaled_float32"} and all(s=="finite_nonzero" for s in states.values()),"single scaled success requires finite nonzero raw and unscaled gradients")
        raw=diagnostics["scaled_loss_backward_raw"];unscaled=diagnostics["scaled_loss_backward_unscaled_float32"]
        require(raw["finite_nonzero_count"]==unscaled["finite_nonzero_count"] and raw["zero_count"]==unscaled["zero_count"],"single scaled FP32 unscale changed zero support")
        for key in ("finite_l2","finite_max_abs","finite_min_abs_nonzero"):
            require(math.isclose(raw[key]/scale,unscaled[key],rel_tol=1e-6,abs_tol=1e-20),"single scaled/unscaled magnitude mismatch: "+key)
        g=report["latent_gradient"]
        require(g["sha256_float32"]==unscaled["sha256_float32"] and g["nonzero_count"]==unscaled["finite_nonzero_count"] and g["l2"]==unscaled["finite_l2"] and g["max_abs"]==unscaled["finite_max_abs"],"single scaled final gradient/statistics mismatch")
    return {"scale":scale,"states":states,"statistics":diagnostics,"first_unscaled_backward_performed":False,
            "gradient_gate":"passed_numerical_probe" if report.get("complete") else "failed_or_incomplete",
            "memory_semantics":"Single backward at fixed loss scale; decoder only; not joint DiT/LoRA memory"}


def validate_report(report, status, queue):
    c = report["contract"]
    require(c["schema"] in (CONTRACT_SCHEMA,PAIR_SCHEMA,SCALED_SINGLE_SCHEMA), "unsupported probe contract")
    paired=c["schema"]==PAIR_SCHEMA
    scaled_single=c["schema"]==SCALED_SINGLE_SCHEMA
    expected_backwards=2 if paired else 1
    require(c["planned_model_forwards"]==2 and c["planned_backward_calls"]==expected_backwards, "unexpected probe budget")
    if paired:
        require(c["diagnostic_kind"]=="loss_scale_pair" and c.get("expected_gradient_sha256") is None,"v2 must be independent paired-loss-scale diagnostic")
        scale=c["paired_loss_scale_diagnostic"]
        require(isinstance(scale,(int,float)) and math.isfinite(scale) and 1<scale<=65536,"invalid diagnostic loss scale")
        require(c.get("fixed_loss_scale",1)==1,"paired mode may not also fix the first loss scale")
    elif scaled_single:
        require(c["diagnostic_kind"]=="scaled_single_backward" and c.get("paired_loss_scale_diagnostic") is None,"v3 must be single scaled backward")
        scale=c["fixed_loss_scale"]
        require(isinstance(scale,(int,float)) and math.isfinite(scale) and 1<scale<=65536,"invalid fixed single loss scale")
        valid_sha(c.get("expected_gradient_sha256"))
    else:
        require(c.get("diagnostic_kind","single_backward")=="single_backward" and c.get("paired_loss_scale_diagnostic") is None,"v1 may not silently use paired scaling")
        require(c.get("fixed_loss_scale",1)==1,"v1 may not silently use fixed loss scaling")
    require(c["H3_denoiser_calls"]==c["optimizer_steps"]==0 and c["VAE_parameters_updated"] is False, "probe exceeded decoder-only scope")
    require(c["forward_comparison_atol"]==.0002 and c["forward_comparison_rtol"]==.0001, "forward tolerance changed")
    shape = c["latent_shape"]
    require(len(shape)==5 and shape[:2]==[1,24] and shape[2] in (2,7) and shape[3:] in ([8,8],[30,52]), "unexpected latent geometry")
    expected = [(shape[2]-2)//5*17+5,shape[3]*16,shape[4]*16,3]
    require(c["expected_rgb_shape"]==expected, "latent/output geometry disagreement")
    argv = queue["steps"][0]["argv"]
    require(len(queue["steps"])==1 and "--execute" in argv, "single executed decoder probe required")
    require(int(option(argv,"--seed","101"))==c["seed"], "seed CLI/contract mismatch")
    require(option(argv,"--rope-mode","eager")==c["grad_rope_mode"], "RoPE CLI/contract mismatch")
    require(int(option(argv,"--latent-t","2"))==shape[2], "latent T CLI/contract mismatch")
    require(option(argv,"--geometry","small")==("full" if shape[3:]==[30,52] else "small"), "geometry CLI/contract mismatch")
    require(("--checkpoint-vae-blocks" in argv)==c.get("checkpoint_vae_blocks",False), "checkpoint CLI/contract mismatch")
    require(option(argv,"--expected-gradient-sha256")==c.get("expected_gradient_sha256"), "expected gradient CLI/contract mismatch")
    if paired:
        require(float(option(argv,"--diagnose-loss-scale","nan"))==scale,"loss scale CLI/contract mismatch")
    else:
        require("--diagnose-loss-scale" not in argv,"v1 queue requests a second backward")
    require(float(option(argv,"--loss-scale","1"))==(scale if scaled_single else 1),"fixed loss scale CLI/contract mismatch")
    forwards, backwards = report["actual_completed_forwards"],report["actual_completed_backwards"]
    require(type(forwards) is int and 0<=forwards<=2 and type(backwards) is int and 0<=backwards<=expected_backwards and (not backwards or forwards==2), "invalid completed-call accounting")
    full = expected==[22,480,832,3]
    summary = {"probe_complete":False,"decoded_geometry":expected,"latent_geometry":shape,
        "actual_completed_forwards":forwards,"actual_completed_backwards":backwards,
        "no_grad_forward_completed":forwards>=1,"grad_forward_completed":forwards>=2,
        "backward_returned":backwards>=1,"all_planned_backwards_returned":backwards==expected_backwards,
        "full_decoder_memory_gate":"not_established_by_paired_backward" if paired else ("failed" if full else "not_tested_small_geometry"),
        "full_H3_DiT_LoRA_memory_gate":"not_tested","restoration_quality_gate":"not_tested",
        "scope":"synthetic latent decoder input gradient; no original RGB objective, no adapter gradient or joint fullH3 memory"}
    if paired:
        summary["loss_scale_pair"]=loss_scale_pair_summary(report,shape,scale)
        summary["paired_scale_gradient_gate"]="failed_or_incomplete"
    elif scaled_single:
        summary["scaled_single_backward"]=scaled_single_summary(report,shape,scale)
        summary["scaled_single_gradient_gate"]="failed_or_incomplete"
    if report.get("complete") is not True:
        require(status["state"]=="failed" and isinstance(report.get("error"),str), "failure report/queue disagreement")
        summary.update(state="failed", error=report["error"], failure_category="out_of_memory" if "out of memory" in report["error"].lower() else "runtime_or_numerical_failure")
        if scaled_single and report.get("gradient_diagnostics",{}).get("scaled_loss_backward_raw"):
            summary["failure_gradient_state"]=gradient_diagnostic_state(report["gradient_diagnostics"]["scaled_loss_backward_raw"],shape)
        elif not paired and report.get("gradient_diagnostics",{}).get("unscaled_loss_backward"):
            summary["failure_gradient_state"]=gradient_diagnostic_state(report["gradient_diagnostics"]["unscaled_loss_backward"],shape)
        elif not paired and "gradient missing, nonfinite or zero" in report["error"]:
            summary["failure_gradient_state"]="unknown_missing_nonfinite_or_zero_not_disambiguated_by_producer"
        return summary
    require(status["state"]=="completed" and status["completed"]==[queue["steps"][0]["name"]], "successful report requires completed queue")
    require(forwards==2 and backwards==expected_backwards, "success requires both forwards and all planned backwards")
    require(report["decoded_shape"]==expected and report["gradient_mode_output_requires_grad"] is True and report["inference_wrapper_output_requires_grad"] is False, "gradient/shape contract failed")
    difference = report["forward_max_abs_difference"]
    require(math.isfinite(difference) and 0<=difference<=.0003+1e-12, "forward maximum exceeds possible bounded-RGB allclose tolerance")
    if report["forward_bitwise_equal"]:
        require(difference==0, "bitwise equality/max error disagreement")
    require(math.isfinite(report["synthetic_loss"]) and report["synthetic_loss"]>=0, "invalid loss")
    grad = report["latent_gradient"]
    require(grad["finite"] is True and grad["elements"]==math.prod(shape) and 0<grad["nonzero_count"]<=grad["elements"], "gradient missing/zero/nonfinite/wrong size")
    require(all(math.isfinite(grad[k]) and grad[k]>0 for k in ("l2","max_abs")), "gradient norm invalid")
    valid_sha(grad["sha256_float32"])
    valid_sha(report["latent_sha256_float32"])
    require(report["VAE_parameter_sha256_before"]==report["VAE_parameter_sha256_after"], "frozen VAE weights changed")
    valid_sha(report["VAE_parameter_sha256_before"])
    require(report["VAE_parameter_count"]>0 and report["VAE_parameter_versions_unchanged"] is True and report["VAE_parameters_have_no_grad"] is True, "frozen parameter attestation failed")
    model=report["model"]
    require(model["precision_used"]=={"matmul_hf32":False,"conv_hf32":False} and model["jit_compile_false"] is True, "precision/JIT contract changed")
    require(model["vae_only"]["dit_loaded"] is False and model["vae_only"]["sampler_calls"]==model["vae_only"]["denoiser_evaluations"]==0, "DiT unexpectedly loaded/called")
    if c.get("checkpoint_vae_blocks"):
        ck = report["decoder_checkpoint"]
        require(ck["schema"]=="openh3.vae_decoder_checkpoint_context.v1" and ck["enabled"] is True and ck["active"] is False and ck["restored"] is True, "checkpoint scope not safely restored")
        require(ck["block_count"]==ck["expected_block_count"]==ck["wrapped"]==36 and ck["recomputed_block_indices"]==list(range(36)), "checkpoint block coverage incomplete")
        require(ck["use_reentrant"] is False and ck["preserve_rng_state"] is True and ck["parameter_or_buffer_mutation_performed"] is False, "checkpoint contract mismatch")
        require(ck["checkpoint_forward_calls"]>=36 and ck["recompute_calls"]>=36*expected_backwards and ck["output_gradient_guards"]>=36 and ck["backward_guard_calls"]>=36*expected_backwards, "checkpoint/backward not exercised")
        summary["checkpoint_audit"]=ck
    pinned=c.get("expected_gradient_sha256")
    if pinned:
        require(report.get("expected_gradient_sha256_matched") is True and grad["sha256_float32"]==pinned, "pinned exact gradient mismatch")
    allocated,reserved=report["npu_peak_allocated_bytes_after_load"],report["npu_peak_reserved_bytes_after_load"]
    require(type(allocated) is int and type(reserved) is int and 0<allocated<=reserved, "missing/invalid NPU peak memory")
    summary.update(probe_complete=True,state="full_decoder_gradient_and_memory_passed" if full else "small_decoder_gradient_passed",
        full_decoder_memory_gate="passed_decoder_only" if full else "not_tested_small_geometry",npu_peak_allocated_bytes=allocated,npu_peak_reserved_bytes=reserved,
        forward_bitwise_equal=report["forward_bitwise_equal"],forward_max_abs_difference=difference,latent_gradient=grad,
        parameters_unchanged=True,forward_allclose="producer asserted under SHA-pinned source; audit checks recorded bound, does not rescore tensors")
    if paired:
        require(report["gradient_loss_scale"]==scale,"report gradient loss scale mismatch")
        summary.update(state="loss_scale_pair_diagnostic_completed",
            paired_scale_gradient_gate="passed_numerical_probe",
            full_decoder_memory_gate="not_established_by_paired_backward",
            paired_retained_graph_peak_observed=True,
            single_backward_training_memory_gate="not_tested_by_paired_diagnostic")
    elif scaled_single:
        require(report["gradient_loss_scale"]==scale,"single backward report loss scale mismatch")
        summary.update(scaled_single_gradient_gate="passed_numerical_probe",gradient_loss_scale=scale,
                       single_backward_training_memory_gate="passed_decoder_only_at_declared_geometry" if full else "not_tested_full_geometry")
    return summary


def inspect_bundle(files, archive_info):
    queue_name=exactly_one(files,"/queue.json")
    admission_name=exactly_one(files,"/admission.json")
    status_name=exactly_one(files,"/queue_status.json")
    probe_name=exactly_one(files,"/code/check_vae_decode_grad.py")
    queues=json.loads(files[queue_name]);admission=json.loads(files[admission_name]);status=json.loads(files[status_name])
    candidates=[n for n in files if n.endswith(("/eval/report.json","/eval/failure.json"))]
    require(len(candidates)==1,"exactly one final report OR failure required")
    name=candidates[0];report=json.loads(files[name]);c=report["contract"]
    require(admission["queue_sha256"]==sha(files[queue_name]),"admission/queue SHA mismatch")
    require(admission["pins_verified"]==len(queues["implementation_sha256"]),"admission pin count incomplete")
    if "passed" in admission:
        require(admission["passed"] is True,"admission explicitly failed")
        require(admission["requirements_verified"]==len(queues["required_json_values"]),"admission requirements incomplete")
        require(admission["lock_available"] is True and admission["queue_rechecks_15seconds"] is True,"admission lock/recheck missing")
        card=admission["card"+str(queues["cards"][0])]
    else:
        # New root producer spells out gates rather than emitting passed:true.
        require(admission["gates_verified"]==len(queues["required_json_values"]),"admission gates incomplete")
        require(admission["lock_observed_available"] is True and admission["queue_stable_recheck_seconds"]==15,"admission lock/recheck missing")
        card=admission["card"]
    require(card["health"]=="OK" and card["pids"]==[] and card["utilization"]==0 and card["total_mb"]-card["used_mb"]>=queues["min_free_mb"],"admission accelerator gate failed")
    require(status["card"] in queues["cards"] and len(queues["cards"])==1,"single-card queue mismatch")
    require(queues["implementation_sha256"][probe_name]==sha(files[probe_name])==c["implementation_sha256"],"probe implementation not bound to queue and report")
    if c.get("checkpoint_vae_blocks"):
        helper=exactly_one(files,"/code/vae_checkpoint_context.py")
        require(queues["implementation_sha256"][helper]==sha(files[helper]),"checkpoint helper not pinned")
    manifest_path=option(queues["steps"][0]["argv"],"--pretrained-manifest")
    require(manifest_path in queues["implementation_sha256"] and queues["implementation_sha256"][manifest_path]==c["manifest_sha256"],"pretrained manifest pin mismatch")
    if manifest_path in files:
        require(sha(files[manifest_path])==c["manifest_sha256"],"archived pretrained manifest mismatch")
    summary=validate_report(report,status,queues)
    summary.update(schema="openh3.vae_gradient_probe_evidence_audit.v1",archive=archive_info,
        report_member=name,report_sha256=sha(files[name]),queue_sha256=sha(files[queue_name]),
        synthetic_probe_only=True,prior_gradient_comparison="not_requested")
    return summary,report,queues


def compare_prior(current, prior, current_queue, prior_queue, prior_report_sha):
    require(current.get("complete") is True and prior.get("complete") is True,"comparison requires two complete numerical probes")
    ca,cb=current["contract"],prior["contract"]
    single_vs_pair=ca["schema"]==SCALED_SINGLE_SCHEMA and cb["schema"]==PAIR_SCHEMA
    if single_vs_pair:
        require(ca["diagnostic_kind"]=="scaled_single_backward" and cb["diagnostic_kind"]=="loss_scale_pair" and ca["planned_backward_calls"]==1 and cb["planned_backward_calls"]==2,"invalid scaled single versus paired comparison")
        require(ca["fixed_loss_scale"]==cb["paired_loss_scale_diagnostic"]==current["gradient_loss_scale"]==prior["gradient_loss_scale"],"prior/current effective loss scale mismatch")
        for name in ("scaled_loss_backward_raw","scaled_loss_backward_unscaled_float32"):
            require(current["gradient_diagnostics"][name]==prior["gradient_diagnostics"][name],"single-versus-pair gradient statistics differ: "+name)
    else:
        require(ca["planned_backward_calls"]==cb["planned_backward_calls"],"prior comparison backward budget changed without declared v3/v2 pairing")
    for key in ("manifest_sha256","pretrained_fingerprint","seed","grad_rope_mode","latent_shape","latent_distribution","expected_rgb_shape","forward_comparison_atol","forward_comparison_rtol","planned_model_forwards","H3_denoiser_calls","optimizer_steps","VAE_parameters_updated"):
        require(ca[key]==cb[key],"prior comparison precondition changed: "+key)
    for key in ("latent_sha256_float32","decode_dtype","VAE_parameter_sha256_before","VAE_parameter_sha256_after","VAE_parameter_count","synthetic_loss"):
        require(current[key]==prior[key],"prior input/weights/loss mismatch: "+key)
    for key in ("precision_used","torch","torch_npu","torch_npu_module_to_cast_weight_dropped","mm_bmm_format_nd","jit_compile_false","comfy_commit","guide_helper_sha256","vae_only"):
        require(current["model"][key]==prior["model"][key],"prior runtime/model mismatch: "+key)
    for key in ("H3_BASE_ENV","COMFYROOT"):
        require(current_queue["environment"][key]==prior_queue["environment"][key],"prior environment changed")
    pa,pb=current_queue["implementation_sha256"],prior_queue["implementation_sha256"]
    require(all(pa[k]==pb[k] for k in pa.keys()&pb.keys()),"shared dependency pin changed")
    expected=ca.get("expected_gradient_sha256")
    require(expected==prior["latent_gradient"]["sha256_float32"]==current["latent_gradient"]["sha256_float32"] and current["expected_gradient_sha256_matched"] is True,"prior exact gradient SHA mismatch")
    require(prior_report_sha in pa.values(),"prior report bytes not pinned by current queue")
    return {"state":"exact_gradient_matched_under_verified_preconditions","gradient_sha256":expected,
            "prior_report_sha256":prior_report_sha,"shared_dependency_hashes_equal":True,
            "runtime_and_input_equal":True,"comparison_kind":"v3_single_scaled_vs_v2_paired_scaled" if single_vs_pair else "same_backward_budget",
            "scope":"same geometry only; this comparison cannot validate a different full-size gradient"}


def pack(root, run, prep, out):
    root=Path(root).resolve();run=safe_name(run);prep=safe_name(prep);out=Path(out)
    require(not out.exists(),"refuse existing archive")
    status=json.loads((root/run/"queue_status.json").read_text())
    require(status["state"] in ("completed","failed"),"only terminal probe may be packed")
    names=[run+"/queue_status.json",prep+"/queue.json",prep+"/admission.json",prep+"/code/check_vae_decode_grad.py"]
    optional=[run+"/eval/report.json",run+"/eval/failure.json",run+"/logs/gradient.log",prep+"/code/vae_checkpoint_context.py",prep+"/cpu_context_tests.log"]
    names += [name for name in optional if (root/name).is_file()]
    require(sum(name.endswith(("/eval/report.json","/eval/failure.json")) for name in names)==1,"exactly one terminal numerical artifact required")
    queue=json.loads((root/prep/"queue.json").read_text())
    manifest=option(queue["steps"][0]["argv"],"--pretrained-manifest")
    if manifest:
        safe_name(manifest)
        names.append(manifest)
    blobs={}
    for name in names:
        path=(root/name).resolve()
        require(path.is_relative_to(root) and path.is_file(),"archive input outside declared root")
        require(path.stat().st_size<=MAX_SMALL_BYTES,"small evidence file exceeds size cap")
        blobs[name]=path.read_bytes()
    require(sum(map(len,blobs.values()))<=MAX_SMALL_BYTES,"small archive total cap exceeded")
    receipt={name:{"bytes":len(data),"sha256":sha(data)} for name,data in blobs.items()}
    blobs[prep+"/evidence_manifest.json"]=(json.dumps(receipt,indent=2)+'\n').encode()
    out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(out,"x:gz") as tar:
        for name,data in blobs.items():
            member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    return {"archive":str(out),"archive_sha256":sha(out.read_bytes()),"bytes":out.stat().st_size,"files":len(receipt),"source_artifacts_modified":False}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest="mode",required=True)
    pp=sub.add_parser("pack")
    for key in ("root","run","prep","out"):pp.add_argument("--"+key,required=True)
    aa=sub.add_parser("audit")
    for key in ("archive","archive-sha256","out"):aa.add_argument("--"+key,required=True)
    aa.add_argument("--reference-archive");aa.add_argument("--reference-sha256")
    args=ap.parse_args(argv)
    if args.mode=="pack":print(json.dumps(pack(args.root,args.run,args.prep,args.out)));return 0
    require(not Path(args.out).exists(),"refuse existing audit output")
    files,info=read_archive(args.archive,args.archive_sha256)
    result,current,queue=inspect_bundle(files,info)
    if args.reference_archive:
        require(args.reference_sha256 is not None,"reference archive requires explicit SHA")
        refs,refinfo=read_archive(args.reference_archive,args.reference_sha256)
        prior_summary,prior,prior_queue=inspect_bundle(refs,refinfo)
        result["prior_gradient_comparison"]=compare_prior(current,prior,queue,prior_queue,prior_summary["report_sha256"])
        result["reference_archive"]=refinfo
    elif current["contract"].get("expected_gradient_sha256") and current.get("complete"):
        result["prior_gradient_comparison"]="producer_matched_pin_but_prior_preconditions_not_independently_verified_without_reference_archive"
    result["audit_implementation_sha256"]=sha(Path(__file__).read_bytes())
    with Path(args.out).open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:result[key] for key in ("state","probe_complete","full_decoder_memory_gate","full_H3_DiT_LoRA_memory_gate")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
