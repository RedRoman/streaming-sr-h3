"""Independent CPU audit of saved full22 joint RGB evidence; no producer replay."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_arrays import (array_sha, compare_outputs, file_sha, recompute_rgb_mse,
    require, verified_array, verify_decoder_cast, verify_reduction, verify_unscale)
from finalize_once import admission, verify


VIDEO = [1,24,7,30,52]
AUDIO = [1,32,2,37]
RGB = [22,480,832,3]
TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
NAMES = {f"blocks.{i}.{target}.lora_{letter}" for i in range(50) for target in TARGETS for letter in ("a","b")}
SMALL = {"guide": (VIDEO,"float32"), "raw_warm_start": (VIDEO,"float32"),
         "reference_video": (VIDEO,"float32"), "candidate_video": (VIDEO,"float32"),
         "reference_audio": (AUDIO,"float32"), "candidate_audio": (AUDIO,"float32"),
         "reference_decoder_input": (VIDEO,"float16"), "candidate_decoder_input": (VIDEO,"float16"),
         "reference_rgb": (RGB,"float32"), "candidate_rgb": (RGB,"float32"),
         "target": (RGB,"float32"),
         "reference_stage_decoder_input": (VIDEO,"float16"), "candidate_stage_decoder_input": (VIDEO,"float16")}


def read_json(path):
    return json.loads(Path(path).read_text())


class Arrays:
    def __init__(self, run, rows):
        self.run, self.rows = Path(run).resolve(), rows
        self.manifest, self.archive = {}, {}

    def load(self, role, shape=None, dtype="float32", record=None):
        row = self.rows[role]
        if record is not None: require(row == record, "nested array reference differs: " + role)
        require(row["role"] == role, "array role differs")
        path = (self.run / row["path"]).resolve()
        require(path.is_relative_to(self.run / "arrays"), "array escaped producer arrays directory")
        value = verified_array(path,row["file_sha256"],row["array_sha256"],shape or row["shape"],dtype)
        self.manifest[role] = {**row,"bytes":path.stat().st_size}
        if role in SMALL and role not in {"reference_rgb", "candidate_rgb", "target"}:
            archive_role = role.replace("_decoder_input","_decoder_cast")
            self.archive[archive_role] = {"path":str(path),"file_sha256":row["file_sha256"]}
        return value


def budget(report, success):
    a = report["actual"]
    limits = {"reference_NFE":3,"candidate_NFE":3,"reference_VAE_decode_calls":1,
              "candidate_VAE_decode_calls":1,"RGB_loss_expressions":1,"backward_attempts":1,"completed_backward_calls":1}
    for name, limit in limits.items():
        require(type(a[name]) is int and 0 <= a[name] <= limit, "invalid operation budget: " + name)
        if success: require(a[name] == limit, "incomplete successful budget: " + name)
    require(a["completed_backward_calls"] <= a["backward_attempts"], "backward completion without attempt")
    require(a["optimizer_steps"] == a["adapter_updates"] == 0, "forbidden optimizer/update")
    require(a["VAE_encode_calls"] == a["Rphi_calls"] == 0, "forbidden encode/Rphi")
    maps = report["block_forward_calls"]
    for group, phases, count, repeats in (("DiT",("reference","candidate","backward"),50,3),
                                          ("VAE",("reference_decode","candidate_decode","backward"),36,3)):
        require(set(maps[group]) <= set(phases), "unexpected block phase")
        for phase, values in maps[group].items():
            require(set(values) <= {str(i) for i in range(count)} and all(type(v) is int and 0 <= v <= repeats for v in values.values()), "block work budget exceeded")
        if success: require(maps[group] == {p:{str(i):repeats for i in range(count)} for p in phases}, "missing/wrong block counts")
    if success:
        require(a["forbidden_calls"] == [] and a["stage_decode_calls"] == {"reference_decode":1,"candidate_decode":1}, "extra stage/forbidden call")
        require(report["budget_passed"] is True and report["block_budget_passed"] is True, "producer budget failed")
    return {"actual":a,"block_counts":maps,"explicit_NFE":a["reference_NFE"]+a["candidate_NFE"]}


def native_tile_geometry(report):
    plan=report["vae_native_plan"]
    expected={"tiling":True,"tile_size":256,"vae_ratio":16,"vae_ratio_t":4,
        "temporal_padding_and_chunks":[0,1],"height_tile_plan":[[0,112,224],[256,256,256],[144,144]],
        "width_tile_plan":[[0,192,384,576],[256,256,256,256],[64,64,64]],
        "output_shape":[1,3,22,480,832],"changed":False}
    require(plan==expected,"native spatial/temporal plan differs")
    rows=report["native_tile_rows"]
    require(len(rows)==6 and report["tile_budget_passed"] is True,"native row budget differs")
    for phase in ("reference_decode","candidate_decode"):
        selected=[r for r in rows if r["phase"]==phase]
        require([r["row_index"] for r in selected]==[0,1,2],"native row order differs")
        for row in selected:
            require(row["input_shape"]==[1,24,7,16,52] and row["x_idx"]==[0,192,384,576]
                and row["x_len"]==[256]*4 and row["pixel_calls"]==1 and row["yielded_tiles"]==4
                and len(row["groups"])==1,"native tiling invocation differs")
            group=row["groups"][0]
            require(group["shape"]==[4,24,7,16,16] and group["dtype"]=="torch.float16","native grouped input differs")
    return {"rows":6,"native_tiles_per_row":4,"temporal_chunks":1}


def native_tiles(report, arrays, contract):
    geometry=native_tile_geometry(report)
    for arm in ("reference","candidate"):
        require(array_sha(arrays[arm+"_stage_decoder_input"])==array_sha(arrays[arm+"_decoder_input"]),"actual stage/outer decoder input differs")
        require(report["stage_decoder_inputs"][arm]==report["arrays"][arm+"_stage_decoder_input"],"stage input receipt differs")
    inspection=report["input_inspection"]
    spec=read_json(Path(contract["root"])/contract["source_contract"]["path"])
    require(file_sha(Path(contract["root"])/contract["source_contract"]["path"])=="0b1b34b1fbedb26cf18b8d9d636e820185fe6ff2e701822b9661a627172d8df3","reviewed source contract changed")
    require(inspection["complete"] is True and inspection["identity"]==spec["inputs"]["identity"]
        and inspection["identity"]["role"]=="train" and inspection["target_model_input"] is False
        and inspection["model_input_keys"]==["guide","raw_warm_start"]
        and inspection["nonzero_adapter_weights_loaded"] is False,"actual train input inspection differs")
    require(inspection["target_float32_sha256"]==array_sha(arrays["target"])==spec["inputs"]["target"]["loss_target_float32_sha256"],"real target pixels differ")
    for key,role in (("guide","guide"),("warm_start","raw_warm_start")):
        require(array_sha(arrays[role])==spec["inputs"][key]["array_sha256"],"actual cache model input differs")
    require(report["decoder_context_budget_passed"] is True,"decoder context budget failed")
    return {**geometry,"saved_stage_inputs_exact":True,
        "real_train_target_SHA_verified":True,"source_identity":inspection["identity"],
        "scope":"Saved input pixels independently checked; native grouping is source-bound producer observation, no model replay."}


def noise(report, arrays, contract):
    observations = report["noise_observations"]
    require(len(observations)==2 and [x["phase"] for x in observations]==["reference","candidate"], "two actual noise observers absent")
    left,right=({k:v for k,v in x.items() if k!="phase"} for x in observations)
    require(left==right and left["seed"]==101 and left["joint_shapes"]==[VIDEO,AUDIO], "noise/input/seed mismatch")
    inputs,candidate=report["inputs"],report["candidate_contract"]
    require(inputs["seed"]==101 and inputs["audio_init"]=="zeros" and inputs["target_is_never_model_input"] is True, "input contract differs")
    require(inputs["context_sha256"]==contract["context_sha256"] and inputs["distinct"] is True
            and array_sha(arrays["guide"])!=array_sha(arrays["raw_warm_start"]), "context or guide/draft mismatch")
    expected={"guide_latent_sha256_float32":array_sha(arrays["guide"]),"raw_warm_start_latent_sha256_float32":array_sha(arrays["raw_warm_start"]),
              "packed_cpu_noise_sha256_float32":left["packed_noise_sha256"],"packed_initial_latent_sha256_float32":left["packed_initial_sha256"]}
    require(all(candidate[k]==v for k,v in expected.items()), "candidate actual input hashes differ")
    require(report["noise_and_initial_match"] is True and candidate["seed"]==101 and candidate["audio_init"]=="zeros"
            and candidate["actual_forward_NFE"]==3 and candidate["checkpoint_blocks"] is True
            and candidate["checkpoint_decoder_blocks"] is True and candidate["target_used_as_model_input"] is False,
            "candidate trajectory mode differs")
    require(np.array_equal(np.asarray(candidate["sigma_schedule_float32"],np.float32),np.asarray([.909375,6/7,2/3,0],np.float32)), "native sigmas differ")
    require(candidate["cfg"]==1 and candidate["mode"]=="base" and candidate["refs"]==[] and candidate["inner_dc"] is False
            and candidate["s_churn"]==0 and candidate["adapter_parameter_tensors"]==400
            and candidate["adapter_B_all_zero_at_session_entry"] is True, "candidate conditioning/adapter contract differs")
    require([x["step"] for x in report["candidate_euler_trace"]]==[0,1,2], "Euler trace budget differs")
    return {"actual_observer_hashes_equal":True,**left,
            "scope":"Packed initial/noise are source-bound observer byte hashes; saved guide/raw arrays independently verified."}


def parity_record(row, reference, candidate, facts):
    require(row["torch_equal"]==facts["exact_equal"] and row["same_shape"] is True and row["finite"] is True, "parity flags disagree with saved arrays")
    require(row["reference_shape"]==list(reference.shape) and row["candidate_shape"]==list(candidate.shape), "parity shape metadata differs")
    require(row["reference_sha256_float32"]==array_sha(reference) and row["candidate_sha256_float32"]==array_sha(candidate), "parity content hashes differ")
    verify_reduction(row["max_abs"],float(np.max(np.abs(reference-candidate))),"parity maxabs")
    delta=(reference-candidate).astype(np.float64)
    relative=float(np.linalg.norm(delta)/max(float(np.linalg.norm(reference.astype(np.float64))),1e-30))
    verify_reduction(row["relative_l2"],relative,"parity relative L2")


def outputs(report, arrays, target_sha):
    facts={}
    for stream in ("video","audio"):
        reference,candidate=arrays["reference_"+stream],arrays["candidate_"+stream]
        facts[stream]=compare_outputs(candidate,reference,exact=True)
        parity_record(report["latent_parity"][stream],reference,candidate,facts[stream])
    for arm in ("reference","candidate"):
        verify_decoder_cast(arrays[arm+"_video"],arrays[arm+"_decoder_input"])
    require(array_sha(arrays["reference_decoder_input"])==array_sha(arrays["candidate_decoder_input"])
            and report["decoder_input_exact"] is True, "actual decoder input bytes differ")
    reference,candidate=arrays["reference_rgb"],arrays["candidate_rgb"]
    facts["RGB"]=compare_outputs(candidate,reference,exact=False)
    parity_record(report["rgb_parity"],reference,candidate,facts["RGB"])
    require(report["rgb_parity"]["allclose"] is True and report["rgb_parity"]["atol"]==.0002
            and report["rgb_parity"]["rtol"]==.0001, "RGB declared allclose differs")
    require(set(report["graph"])=={"video_requires_grad","video_has_grad_fn","rgb_requires_grad","rgb_has_grad_fn"}
            and all(v is True for v in report["graph"].values()), "retained RGB graph attestation missing")
    loss=report["loss"]
    measured=recompute_rgb_mse(candidate,arrays["target"],target_sha)
    verify_reduction(loss["numpy_float64_mse"],measured,"independent RGB target MSE")
    require(type(loss["torch_float32_mse"]) in (int,float) and math.isfinite(loss["torch_float32_mse"])
            and loss["torch_float32_mse"]>=0 and loss["scale"]==1024. and loss["auxiliary_latent_audio_loss"] is False,
            "RGB loss/scale contract differs")
    facts["loss"]={"independent_numpy_float64_mse":measured,"torch_float32_mse_attestation":loss["torch_float32_mse"],
                   "torch32_minus_independent64":loss["torch_float32_mse"]-measured,
                   "note":"Torch32 reduction is the actual backward scalar; NumPy64 is an independent saved-pixel metric, without a new training tolerance."}
    return facts


def grad_stats(array,row):
    finite=np.isfinite(array);valid=array[finite]
    exact={"present":True,"elements":int(array.size),"finite_count":int(finite.sum()),"all_finite":bool(finite.all()),
           "nonzero_finite_count":int(np.count_nonzero(valid)),"nan_count":int(np.isnan(array).sum()),"inf_count":int(np.isinf(array).sum())}
    require(all(row[k]==v for k,v in exact.items()),"gradient finite/count metadata differs")
    norm=float(np.linalg.norm(valid.astype(np.float64)))
    verify_reduction(row["finite_l2"],norm,"gradient float64 norm")
    return norm,exact["nonzero_finite_count"]


def gradients(report, reader, success):
    initial=report["adapter_initial"]
    require(initial["modules"]==200 and initial["trainable_tensors"]==400 and initial["rank"]==8
            and set(initial["tensors"])==NAMES,"declared canonical 400 adapters differ")
    for name,row in initial["tensors"].items():
        require(row["dtype"]=="torch.float32" and len(row["shape"])==2 and row["shape"][0 if name.endswith(".lora_a") else 1]==8,
                "adapter shape/rank/dtype differs")
        if name.endswith(".lora_b"):require(row["all_zero"] is True,"B was not initially zero")
    summary=report["adapter_gradients"];rows=summary["tensors"]
    require(set(rows)==NAMES and summary["expected"]==400,"gradient names/count differ")
    count,b_nonzero,b2=0,0,0.
    for name,row in rows.items():
        if row["present"] is False:
            require(not success,"missing successful adapter gradient");continue
        require(row["present"] is True,"malformed gradient presence")
        raw=reader.load("gradient_raw__"+name,initial["tensors"][name]["shape"],record=row["raw"])
        scaled=reader.load("gradient_unscaled__"+name,raw.shape,record=row["unscaled"])
        verify_unscale(raw,scaled)
        norm,nz=grad_stats(raw,row["statistics_raw"]);grad_stats(scaled,row["statistics_unscaled"])
        count+=1
        if name.endswith(".lora_b"):b2+=norm*norm;b_nonzero+=nz>0
    require(summary["present_tensors"]==count and summary["B_nonzero_tensors"]==b_nonzero,"gradient presence/B summary differs")
    verify_reduction(summary["B_l2_raw"],math.sqrt(b2),"B raw aggregate")
    verify_reduction(summary["B_l2_unscaled"],math.sqrt(b2)/1024.,"B unscaled aggregate")
    if success:require(count==400 and b2>0 and b_nonzero>0 and summary["passed"] is True and summary["all_finite"] is True,"RGB adapter gradient gate failed")
    row=report["retained_video_gradient"]
    raw=reader.load("video_gradient_raw",VIDEO,record=row["raw"])
    scaled=reader.load("video_gradient_unscaled",VIDEO,record=row["unscaled"])
    video=verify_unscale(raw,scaled,nonzero=success)
    grad_stats(raw,row["statistics_raw"]);grad_stats(scaled,row["statistics_unscaled"])
    if success:require(row["passed"] is True,"retained video gradient gate failed")
    return {"adapter_pairs_verified":count,"B_nonzero_tensors":b_nonzero,"B_l2_raw":math.sqrt(b2),"retained_video":video,
            "A_zero_legal":True,"every_B_nonzero_required":False}


def frozen(report, run, success):
    observations={}
    for label in ("frozen_initial","final_frozen"):
        if label not in report:
            require(not success,"missing frozen evidence");continue
        row=report[label];path=(run/row["path"]).resolve()
        require(path.parent==run and file_sha(path)==row["sha256"],"frozen JSON path/SHA differs")
        value=read_json(path);require(set(value)=={"DiT","VAE"},"frozen model identity differs")
        for model,inventory in value.items():
            for kind in ("parameters","registered_buffers"):
                records=inventory[kind]
                expected_counts={"DiT":{"parameters":530,"registered_buffers":2},"VAE":{"parameters":559,"registered_buffers":42}}
                require(len(records)==expected_counts[model][kind],"frozen inventory tensor count differs")
                require(len({x["name"] for x in records})==len(records),"duplicate frozen tensor name")
                content=[{k:r[k] for k in ("name","shape","dtype","sha256_bytes")} for r in records]
                digest=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                require(digest==inventory[kind+"_sha256"],"frozen canonical fingerprint differs")
                if success:
                    require(all(x["requires_grad"] is False and x["grad_is_none"] is True and x["device"]=="npu:0" for x in records),"frozen/residency tensor facts failed")
            require(inventory["parameters"],"empty frozen parameter inventory")
        observations[label]=value
    if success:
        require(report["reference_bytes_restored"] is True and report["final_frozen"]["canonical_bytes_match"] is True,"producer canonical freeze failed")
        for group in ("DiT","VAE"):
            for kind in ("parameters","registered_buffers"):
                require(observations["frozen_initial"][group][kind+"_sha256"]==observations["final_frozen"][group][kind+"_sha256"],"initial/final frozen bytes differ")
        stages={"before_reference_decode","after_reference_decode","after_injection","before_candidate_decode","after_candidate_decode","before_backward","after_backward","finally"}
        require(set(report["state_checks"])==stages,"lifecycle freeze checks missing")
        for stage,checks in report["state_checks"].items():
            expected={"DiT","VAE"} if stage in ("before_reference_decode","after_reference_decode") else {"DiT","VAE","adapters"}
            require(set(checks)==expected,"missing lifecycle model checks")
            for group,row in checks.items():
                count=400 if group=="adapters" else sum(len(observations["final_frozen"][group][kind]) for kind in ("parameters","registered_buffers"))
                require(row["passed"] is True and row["failures"]==[] and row["tensors_checked"]==count,"object/version/frozen lifecycle failed")
    return {"inventory_groups":list(observations),"scope":"Producer-recorded full named parameters/registered buffers and lifecycle checks; canonical JSON hashes independently recomputed. No live model re-read or unregistered Python cache claim."}


def contexts(report):
    require(report["math_before"]==report["math_after"] and report["math_restored"] is True,"global math not restored")
    active=report["DiT_math_active"]
    require(report["DiT_math_after_decode"]==active and active["in_training"] is True,"native decoder leaked into DiT")
    counts=Counter();vae_rope=None
    for row in report["math_observations"]:
        group,phase,state=row["group"],row["phase"],row["state"]
        counts[(group,phase,str(row["block"]))]+=1
        require(state["in_training"] is True and state["jit_compile_false"] is True and state["matmul_allow_hf32"] is False
                and state["conv_allow_hf32"] is False,"block runtime math flags differ")
        require(all(state[k]==report["math_before"][k] for k in ("deterministic_algorithms","deterministic_warn_only")),"determinism changed")
        if group=="DiT" and phase in ("candidate","backward"):require(state==active,"DiT recompute math context differs")
        if group=="VAE" and phase in ("candidate_decode","backward"):
            require(state["rope"]["module"]=="comfy_kitchen.backends.ascend.rope" and state["rope"]["qualname"]=="rms_rope_split_half","VAE native RoPE identity differs")
            if vae_rope is None:vae_rope=state["rope"]
            require(state["rope"]==vae_rope,"VAE RoPE function object changed")
    expected=Counter({(g,p,i):v for g,phases in report["block_forward_calls"].items() for p,blocks in phases.items() for i,v in blocks.items()})
    require(counts==expected,"math observations do not cover actual block entries")
    rows=report["decoder_context_restorations"]
    require(Counter(r["phase"] for r in rows)=={"candidate_decode":109,"backward":108},"decoder context invocation/restoration count differs")
    require(all(r["restored"] is True and r["before"]==r["after"] for r in rows),"decoder context not restored")
    require(report["wrapper_attributes_restored"] is True and report["session_inactive"] is True
            and report["stale_backward_guard_rejected"] is True,"checkpoint wrappers/session not restored")
    require(report["patcher_restoration"]["session_state_initialized"] is True,"successful session never initialized")
    require(all(report["patcher_restoration"][key] is True for key in ("session_shapes_restored", "session_current_patcher_restored",
            "whole_run_current_patcher_restored", "whole_run_shapes_restored")),"session/whole-run patcher or latent shapes not restored")
    return {"math_observation_count":sum(counts.values()),"decoder_context_restorations":len(rows),"VAE_native_rope":vae_rope}


def imports(report, contract):
    expected={str((Path(contract["root"])/k).resolve()):v for k,v in contract["implementation_sha256"].items()}
    require(report["verified_runtime_files"]==expected,"verified runtime map differs")
    for label in ("actual_imports_before_sampling","actual_imports"):
        row=report[label];require(row["passed"] is True and row["unbound"]==[] and row["modules"],"unbound imports")
        require(all(expected.get(v["path"])==v["sha256"] for v in row["modules"].values()),"actual imported source differs")
        require(set(row["synthetic_namespaces"])=={"torch.ops","torch.classes"},"Torch synthetic namespace evidence missing")
        for v in row["synthetic_namespaces"].values():require(expected.get(v["defining_source_path"])==v["defining_source_sha256"],"Torch definition source not pinned")
        require("comfy_kitchen.backends" in row["namespace_packages"],"PEP420 evidence missing")
        for v in row["namespace_packages"].values():
            require(v["loader_type"]=="_frozen_importlib_external.NamespaceLoader" and v["source_files"],"wrong PEP420 loader")
            subset={p:h for p,h in expected.items() if Path(p).suffix==".py" and any(Path(p).is_relative_to(Path(d)) for d in v["directories"])}
            require(v["source_files"]==subset,"PEP420 source tree differs")


def audit(root, config, paths):
    run=paths["run"]/"eval";queue=read_json(paths["run"]/"queue_status.json")
    require(queue["state"] in ("completed","failed","not_launched"),"producer not terminal")
    base={"schema":"openh3.full22_joint_rgb_terminal_audit.v1","complete":False,"producer_terminal_state":queue["state"],
          "errors":[],"checks":{},"archive_arrays":{},"actions":{"NPU_calls":0,"Torch_imports":0,"model_loads":0,"producer_writes":0,"gradient_arrays_copied":0}}
    if queue["state"]=="not_launched":
        require(queue["producer_started"] is False,"not_launched queue already started producer")
        return {**base,"audit_complete":True,"numerical_acceptance":"not_launched_not_accepted","queue_error":queue.get("error"),"scope":"Queue refused launch; no model execution claimed."}
    success=queue["state"]=="completed"
    chosen=run/("report.json" if success else "failure.json")
    if not chosen.is_file():chosen=run/"partial.json"
    report=read_json(chosen)
    require(report["schema"]=="openh3.full22_joint_rgb_probe.v1" and report["complete"] is success,"queue/report state mismatch")
    allowed=set(SMALL)|{"video_gradient_raw","video_gradient_unscaled"}|{prefix+name for prefix in ("gradient_raw__","gradient_unscaled__") for name in NAMES}
    require(set(report["arrays"])<=allowed,"unexpected saved-array role")
    if success:require(set(report["arrays"])==allowed,"missing successful array role")
    contract=read_json(paths["prep"]/"execution_contract.json")
    require(contract["schema"]=="openh3.full22_joint_rgb_probe_execution.v1" and Path(contract["root"]).resolve()==Path(root).resolve(),"execution schema/root differs")
    require(contract["probe_sha256"]==config["producer_sha256"] and config["trajectory_sha256"] in contract["implementation_sha256"].values(),"producer/trajectory pin differs")
    require(report["contract"]=={**contract,"execution_contract_sha256":config["execution_contract_sha256"]},"embedded execution contract differs")
    reader=Arrays(run,report["arrays"]);loaded={}
    def check(label,fn,available=True):
        if not available:base["checks"][label]={"status":"unavailable"};return
        try:base["checks"][label]={"status":"verified","facts":fn()}
        except Exception as error:base["checks"][label]={"status":"invalid","error":repr(error)};base["errors"].append(label+": "+str(error))
    check("budget",lambda:budget(report,success))
    for role,(shape,dtype) in SMALL.items():
        if role in report["arrays"]:
            try:loaded[role]=reader.load(role,shape,dtype)
            except Exception as error:base["errors"].append(role+": "+str(error))
        elif success:base["errors"].append("missing small array: "+role)
    if success and set(loaded)==set(SMALL):
        check("actual_outputs_and_RGB_MSE",lambda:outputs(report,loaded,config["target_sha256"]))
        check("noise_inputs_schedule",lambda:noise(report,loaded,contract))
        check("full22_native_tiles_and_real_train_inputs",lambda:native_tiles(report,loaded,contract))
    check("actual_gradients",lambda:gradients(report,reader,success),success or "adapter_gradients" in report)
    check("frozen_parameters_buffers",lambda:frozen(report,run,success))
    check("context_and_restoration",lambda:contexts(report),success)
    check("actual_imports",lambda:imports(report,contract),success)
    if success:
        require(report["asset_bytes_sha_verified"] is True,"asset verification absent")
        require(report["environment"]["synthetic_no_dataset"] is False and report["environment"]["role"] == "train","wrong data scope")
    base.update(audit_complete=True,complete=success and not base["errors"],producer_phase=report.get("phase"),producer_error=report.get("failure"),
                report_sha256=file_sha(chosen),array_manifest=reader.manifest,archive_arrays=reader.archive)
    base["numerical_acceptance"]="passed_full22_real_train_joint_RGB_engineering_only" if base["complete"] else "producer_failed_not_accepted" if not success else "invalid_evidence"
    base["limitations"]=["Saved RGB/target, cast, latents and gradients independently checked; live model weights were not reloaded.",
        "Frozen registered-buffer/parameter object and version facts are source-bound producer attestations, excluding unregistered caches.",
        "No optimizer/nonzero adapter reload, BF16 merge, restoration-quality, training-throughput, full32 training, or strict gradient-repeatability acceptance."]
    return base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();own=Path(__file__).resolve().parent
    require(not args.out.exists() and not args.out.is_symlink(),"audit output exists")
    config,paths=admission(root,own,args.config,args.config_sha256);verify(config,paths,own)
    require(not args.out.resolve().is_relative_to(paths["run"]),"audit output cannot modify producer run")
    try:result=audit(root,config,paths)
    except Exception as error:result={"schema":"openh3.full22_joint_rgb_terminal_audit.v1","audit_complete":False,"complete":False,
        "numerical_acceptance":"invalid_or_missing_evidence","errors":[repr(error)],"archive_arrays":{},"actions":{"NPU_calls":0,"Torch_imports":0,"producer_writes":0}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"complete":result["complete"],"numerical_acceptance":result["numerical_acceptance"],"errors":result["errors"],"audit_sha256":file_sha(args.out)}))
    return 0 if not result["errors"] else 2


if __name__=="__main__":main()
