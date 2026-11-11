"""Read-only CPU acceptance for the bounded H3 latent probe; no model imports.

Read each gradient array once, retain only small hashes/statistics, and never
copy the 400 arrays. A failed producer remains failed even if its saved partial
evidence is valid. Producer weight/version facts are source-bound attestations,
not a new read of live 40GB model weights. No quality/full-resolution acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROBE_SHA = "4dee97bc24f923329449e98bef20872868d1549b45f3e4342b131fe156b45775"
TRAJECTORY_SHA = "69bc9ca998b02cc9b40704743f63ac14806113732bb250f0c28d7b4bee2fce22"
CONTRACT_SHA = "c23698f469c2f2962adf1006f8a8dcefcac0b689a0e4abc87782ff82620c79cb"
TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
NAMES = {f"blocks.{i}.{target}.lora_{slot}" for i in range(50)
         for target in TARGETS for slot in ("a", "b")}
SHAPES = {"video": [1, 24, 2, 8, 8], "audio": [1, 32, 2, 8]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def near(a, b):
    # Only verifies a descriptive float64 reduction, never gradient acceptance.
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
        float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


class Files:
    def __init__(self, run, workspace):
        self.run = Path(run).resolve()
        self.workspace = Path(workspace).resolve()
        self.relative_run = self.run.relative_to(self.workspace)
        self.manifest = {}

    def checked(self, name, expected=None):
        path = Path(name)
        if not path.is_absolute():
            # The producer CLI receives runs/.../eval, so serialized paths are
            # workspace-relative rather than relative to the eval directory.
            base = self.workspace if path.is_relative_to(self.relative_run) else self.run
            path = base / path
        path = path.resolve()
        require(path.is_relative_to(self.run), "artifact outside the run: " + str(path))
        digest = sha_file(path)
        if expected is not None:
            require(digest == expected, "file SHA mismatch: " + str(path))
        self.manifest[str(path.relative_to(self.run))] = {
            "sha256": digest, "bytes": path.stat().st_size}
        return path


def validate_budget(report, successful):
    actual = report["actual"]
    for key, maximum in (("reference_NFE", 3), ("candidate_NFE", 3),
                         ("backward_attempts", 1), ("completed_backward_calls", 1)):
        value = actual[key]
        require(type(value) is int and 0 <= value <= maximum, "actual budget exceeded: " + key)
        if successful:
            require(value == maximum, "incomplete successful budget: " + key)
    require(actual["completed_backward_calls"] <= actual["backward_attempts"], "invalid backward count")
    require(actual["optimizer_steps"] == actual["RGB_loss_calls"] == 0, "forbidden optimization/RGB work")
    calls = actual["VAE_calls"]
    require(isinstance(calls, dict) and all(type(v) is int and v >= 0 for v in calls.values()), "invalid VAE counter")
    maps = report.get("block_forward_calls", {})
    for phase, values in maps.items():
        require(phase in ("reference", "candidate", "backward"), "unexpected block-call phase")
        require(set(values) <= {str(i) for i in range(50)}, "unknown checkpoint block index")
        require(all(type(v) is int and 0 <= v <= 3 for v in values.values()), "checkpoint block budget exceeded")
    if successful:
        require(not calls, "VAE guard invoked in successful probe")
        require(all(maps.get(p) == {str(i): 3 for i in range(50)}
                    for p in ("reference", "candidate", "backward")), "not all 50 blocks ran/recomputed three times")
        require(report["checkpoint_recomputed_block_calls"] == 150, "wrong recompute total")
    return {"actual": actual, "explicit_NFE": actual["reference_NFE"] + actual["candidate_NFE"],
            "within_declared_compute_budget": True, "VAE_guard_invocations": sum(calls.values()),
            "block_calls": {p: sum(v.values()) for p, v in maps.items()},
            "failure_count_semantics": "NFE pre-hooks count invocations; an unsuccessful invocation is not a completed forward. Backward attempts/completions remain separate."}


def validate_noise(report):
    observations = report["noise_observations"]
    require(len(observations) == 2, "expected two noise observers")
    require([x["phase"] for x in observations] == ["reference", "candidate"], "noise phases differ")
    left, right = ({k: v for k, v in row.items() if k != "phase"} for row in observations)
    require(left == right and left["seed"] == 101, "actual initial/noise/seed mismatch")
    require(left["joint_shapes"] == [SHAPES["video"], SHAPES["audio"]], "joint noise shapes differ")
    inputs, candidate = report["inputs"], report["candidate_contract"]
    require(inputs["seed"] == 101 and inputs["audio_init"] == "zeros", "wrong input RNG/audio")
    require(inputs["distinct"] is True and inputs["guide"]["sha256_bytes"] != inputs["raw_draft"]["sha256_bytes"], "guide/draft identity error")
    for role in ("guide", "raw_draft"):
        require(inputs[role]["shape"] == SHAPES["video"] and inputs[role]["dtype"] == "torch.float32", "wrong input shape/dtype")
    require(candidate["guide_latent_sha256_float32"] == inputs["guide"]["sha256_bytes"], "candidate guide differs")
    require(candidate["raw_warm_start_latent_sha256_float32"] == inputs["raw_draft"]["sha256_bytes"], "candidate draft differs")
    require(candidate["packed_cpu_noise_sha256_float32"] == left["packed_noise_sha256"]
            and candidate["packed_initial_latent_sha256_float32"] == left["packed_initial_sha256"], "candidate noise metadata differs")
    require(report["noise_and_initial_match"] is True and candidate["seed"] == 101
            and candidate["audio_init"] == "zeros" and candidate["actual_forward_NFE"] == 3,
            "candidate noise/count attestation failed")
    sigmas = np.array([.909375, 6 / 7, 2 / 3, 0], np.float32)
    require(np.array_equal(np.asarray(candidate["sigma_schedule_float32"], np.float32), sigmas), "wrong schedule")
    require(candidate["checkpoint_blocks"] is True and candidate["checkpoint_decoder_blocks"] is False
            and candidate["target_used_as_model_input"] is False and candidate["cfg"] == 1,
            "wrong candidate execution mode")
    return {"matched": True, "seed": 101, "audio_init": "zeros", **left,
            "limitation": "Initial/noise bytes were not saved; agreement is from two source-bound passthrough observer hashes."}


def validate_outputs(report, files):
    output = report["latent_outputs"]
    path = files.checked(output["path"], output["sha256"])
    facts = {}
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == {a + "_" + s for a in ("reference", "candidate") for s in SHAPES}, "unexpected output NPZ members")
        for stream, shape in SHAPES.items():
            a, b = arrays["reference_" + stream], arrays["candidate_" + stream]
            require(a.dtype == b.dtype == np.float32 and list(a.shape) == list(b.shape) == shape, "output shape/dtype mismatch")
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            exact = finite and bool(np.array_equal(a, b))
            row = report["parity"][stream]
            require(row["torch_equal"] == exact and row["finite"] == finite and row["same_shape"] is True, "producer equality differs from actual arrays")
            require(row["reference_shape"] == row["candidate_shape"] == shape, "parity shape record differs")
            require(row["reference_sha256_float32"] == array_sha(a) and row["candidate_sha256_float32"] == array_sha(b), "output content SHA mismatch")
            maximum = float(np.max(np.abs(a - b))) if finite else None
            require(row["max_abs"] == maximum, "output max difference record differs")
            facts[stream] = {"exact_equal": exact, "finite": finite, "max_abs": maximum,
                             "reference_sha256": array_sha(a), "candidate_sha256": array_sha(b)}
        reference = report["reference_latents"]
        with np.load(files.checked(reference["path"], reference["sha256"]), allow_pickle=False) as original:
            require(set(original.files) == set(SHAPES), "reference archive keys differ")
            require(all(array_sha(original[s]) == facts[s]["reference_sha256"] for s in SHAPES), "reference output changed")
    return facts


def validate_gradients(report, files):
    initial = report["adapter_initial"]
    require(initial["modules"] == 200 and initial["trainable_tensors"] == 400 and initial["rank"] == 8
            and set(initial["tensors"]) == NAMES, "adapter identity/count mismatch")
    for name, row in initial["tensors"].items():
        require(row["dtype"] == "torch.float32" and len(row["shape"]) == 2
                and row["shape"][0 if name.endswith(".lora_a") else 1] == 8, "invalid adapter dtype/rank")
        if name.endswith(".lora_b"):
            require(row["all_zero"] is True, "B not initially zero")
    audit = report.get("adapter_gradients", report.get("adapter_gradients_on_failure"))
    require(audit is not None and set(audit["tensors"]) == NAMES, "gradient report lacks declared 400 names")
    missing, b_nonzero, b_squared, total_elements, all_finite = [], 0, 0.0, 0, True
    verified = 0
    for name, row in audit["tensors"].items():
        if row["present"] is False:
            missing.append(name)
            continue
        require(row["present"] is True, "bad gradient presence flag")
        path = files.checked(row["array_file"], row["array_file_sha256"])
        array = np.load(path, allow_pickle=False)
        require(array.dtype == np.float32 and list(array.shape) == row["shape"] == initial["tensors"][name]["shape"], "gradient shape/dtype differs: " + name)
        require(row["dtype"] == "torch.float32" and array_sha(array) == row["sha256_bytes"], "gradient content SHA differs: " + name)
        finite = np.isfinite(array)
        values = array[finite].astype(np.float64)
        norm, nonzero = float(np.linalg.norm(values)), int(np.count_nonzero(values))
        counts = {"elements": int(array.size), "finite_count": int(finite.sum()),
                  "nonzero_finite_count": nonzero, "nan_count": int(np.isnan(array).sum()),
                  "inf_count": int(np.isinf(array).sum()), "all_finite": bool(finite.all())}
        require(all(row[k] == v for k, v in counts.items()), "gradient count/finite record differs: " + name)
        require(near(norm, row["finite_l2"]), "gradient norm record differs: " + name)
        if name.endswith(".lora_b"):
            b_squared += norm * norm
            b_nonzero += nonzero > 0
        all_finite = all_finite and bool(finite.all())
        total_elements += array.size
        verified += 1
        del array, finite, values
    passed = not missing and all_finite and b_nonzero > 0 and b_squared > 0
    require(sorted(audit["missing"]) == sorted(missing) and audit["present_tensors"] == verified, "gradient missing/present summary differs")
    require(audit["all_present_finite"] == all_finite and audit["passed"] == passed, "gradient verdict differs")
    require(audit["B_nonzero_tensors"] == b_nonzero and near(audit["B_l2"], math.sqrt(b_squared)), "B aggregate differs")
    return {"verified_files": verified, "missing": missing, "elements": int(total_elements),
            "all_present_finite": all_finite and not missing, "B_nonzero_tensors": b_nonzero,
            "B_l2_float64_recomputed": math.sqrt(b_squared), "passed": passed,
            "norm_reduction_check": "rtol1e-10/atol1e-12 only for float64 norm reductions; file/content hashes exact"}


def validate_frozen(report, files):
    fingerprints = {}
    for filename, key in (("base_before_reference.json", "base_sha256_before_reference"),
                          ("base_after_restore.json", "base_sha256_after_restore"),
                          ("base_after_backward.json", "base_sha256_after_backward"),
                          ("base_on_exit.json", "base_sha256_on_exit")):
        if key not in report:
            continue
        row = read_json(files.checked(filename))
        tensors = row["tensors"]
        require(tensors and len({r["name"] for r in tensors}) == len(tensors), "invalid frozen tensor identities")
        require(tensors == sorted(tensors, key=lambda x: x["name"]), "frozen identities not canonical")
        digest = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(digest == row["sha256"] == report[key], "frozen fingerprint content SHA differs")
        require(row["parameter_tensors"] == sum(x["name"].startswith("parameter:") for x in tensors)
                and row["buffer_tensors"] == sum(x["name"].startswith("buffer:") for x in tensors), "frozen tensor count differs")
        fingerprints[key] = {"sha256": digest, "tensor_count": len(tensors)}
    versions = {k: report[k] for k in ("base_versions_after_injection", "base_versions_after_backward", "base_versions_on_exit") if k in report}
    return {"fingerprints": fingerprints, "all_available_hashes_equal": bool(fingerprints) and len({v["sha256"] for v in fingerprints.values()}) == 1,
            "version_attestations": versions, "adapter_values_unchanged": report.get("adapter_values_unchanged"),
            "limitation": "Saved per-tensor weight hashes and actual-object/version checks are producer attestations, bound to reviewed source; no live model reload."}


def audit_run(run, contract_path, contract_sha=CONTRACT_SHA):
    run, contract_path = Path(run).resolve(), Path(contract_path).resolve()
    require(sha_file(contract_path) == contract_sha, "execution contract SHA differs")
    contract = read_json(contract_path)
    require(contract["probe_sha256"] == PROBE_SHA, "unreviewed producer")
    require(TRAJECTORY_SHA in contract["implementation_sha256"].values(), "corrected checkpoint source missing")
    queue_path = run.parent / "queue_status.json"
    queue = read_json(queue_path)
    require(queue["state"] in ("completed", "failed"), "producer has not reached a terminal queue state")
    successful = queue["state"] == "completed"
    candidates = [run / name for name in ("report.json", "failure.json", "partial.json") if (run / name).is_file()]
    require(candidates, "no terminal/partial evidence")
    chosen = run / ("report.json" if successful else "failure.json")
    if not chosen.is_file():
        chosen = run / "partial.json"
    report = read_json(chosen)
    require(report["schema"] == "openh3.latent_trajectory_probe.v1", "unknown producer schema")
    require(report.get("complete") is successful, "queue/report completion disagreement")
    require(report["contract"] == {**contract, "execution_contract_sha256": contract_sha}, "embedded execution contract differs")
    expected_runtime = {str((Path(contract["root"]) / key).resolve()): value
                        for key, value in contract["implementation_sha256"].items()}
    require(report["verified_runtime_files"] == expected_runtime, "verified runtime map differs from frozen contract")
    files = Files(run, contract["root"])
    files.checked(chosen)
    checks, errors = {}, []

    def check(name, function, available=True):
        if not available:
            checks[name] = {"status": "unavailable"}
            return
        try:
            checks[name] = {"status": "verified", "facts": function()}
        except Exception as error:
            checks[name] = {"status": "invalid", "error": repr(error)}
            errors.append(name + ": " + str(error))

    check("budget", lambda: validate_budget(report, successful))
    check("noise_initial_inputs", lambda: validate_noise(report), successful or len(report.get("noise_observations", [])) == 2)
    check("actual_output_arrays", lambda: validate_outputs(report, files), successful or "latent_outputs" in report)
    check("actual_gradient_arrays", lambda: validate_gradients(report, files), successful or "adapter_gradients" in report or "adapter_gradients_on_failure" in report)
    check("saved_frozen_evidence", lambda: validate_frozen(report, files))

    if successful and not errors:
        try:
            require(all(x["exact_equal"] for x in checks["actual_output_arrays"]["facts"].values()), "actual strict final parity failed")
            require(checks["actual_gradient_arrays"]["facts"]["verified_files"] == 400
                    and checks["actual_gradient_arrays"]["facts"]["passed"], "actual 400-gradient gate failed")
            frozen = checks["saved_frozen_evidence"]["facts"]
            require(set(frozen["fingerprints"]) >= {"base_sha256_before_reference", "base_sha256_after_restore", "base_sha256_after_backward"}
                    and frozen["all_available_hashes_equal"], "successful frozen hash chain incomplete")
            count = frozen["fingerprints"]["base_sha256_after_backward"]["tensor_count"]
            for name in ("base_versions_after_injection", "base_versions_after_backward"):
                row = frozen["version_attestations"][name]
                require(row["passed"] is True and row["failures"] == [] and row["tensors_checked"] == count, "frozen version/object gate failed")
            require(frozen["adapter_values_unchanged"] is True, "adapter weights changed")
            require(report["wrapper_attributes_restored"] is True and report["session_inactive"] is True
                    and report["stale_backward_guard_rejected"] is True, "checkpoint/session not restored")
            require(report["vae_devices_initial"] == report["vae_devices_final"] and report["vae_devices_final"]
                    and all("npu" not in x for x in report["vae_devices_final"]), "VAE residency changed")
            for name in ("actual_imports_before_sampling", "actual_imports"):
                row = report[name]
                require(row["passed"] is True and row["unbound"] == [] and row["modules"], "unbound/empty actual imports")
                require(all(report["verified_runtime_files"].get(v["path"]) == v["sha256"] for v in row["modules"].values()), "actual import pin disagreement")
            precision = report["precision"]
            require(precision["logical_device"] == 0 and precision["jit_compile_false"] is True
                    and precision["matmul_allow_hf32"] is False and precision["conv_allow_hf32"] is False, "precision gate failed")
            require(report["asset_bytes_sha_verified"] is True, "assets not verified")
        except Exception as error:
            errors.append("success_acceptance: " + str(error))
    accepted = successful and not errors
    return {"schema": "openh3.latent_terminal_independent_audit.v1", "audit_complete": True,
            "producer_terminal_state": queue["state"], "producer_phase": report.get("phase"),
            "numerical_acceptance": "passed_small_latent_only" if accepted else "invalid_evidence" if errors else "producer_failed_not_accepted",
            "complete": accepted, "errors": errors, "checks": checks,
            "queue_status_sha256": sha_file(queue_path), "producer_evidence_file": str(chosen),
            "execution_contract_sha256": contract_sha, "producer_error": report.get("failure"),
            "artifact_manifest": files.manifest,
            "verified_artifact_bytes": sum(v["bytes"] for v in files.manifest.values()),
            "actions": {"NPU_calls": 0, "model_loads": 0, "training_steps": 0, "run_writes": 0, "gradient_files_copied": 0},
            "limitations": ["No full22-frame joint DiT/VAE/LoRA memory or RGB-gradient gate", "No nonzero-adapter save/reload or merged-BF16 parity", "No restoration-quality claim", "Failed producer is never promoted from partial finite/nonzero evidence"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="terminal eval directory")
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new audit JSON outside original run")
    args = parser.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(), "audit output already exists")
    require(not args.out.resolve().is_relative_to(args.run.resolve().parent), "audit output must not modify original run directory")
    try:
        result = audit_run(args.run, args.execution_contract)
    except Exception as error:
        result = {"schema": "openh3.latent_terminal_independent_audit.v1", "audit_complete": False,
                  "complete": False, "numerical_acceptance": "invalid_or_missing_evidence",
                  "errors": [repr(error)], "actions": {"NPU_calls": 0, "model_loads": 0, "run_writes": 0}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"numerical_acceptance": result["numerical_acceptance"], "errors": result["errors"], "out": str(args.out), "out_sha256": sha_file(args.out)}))
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
