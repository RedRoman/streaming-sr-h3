"""One-shot CPU audit/archive of frozen VAE latent descent j; no polling/model.

Arrays are independently read and checked. RGB values are not saved: RGB loss,
forward parity and frozen-parameter facts remain source-bound attestations.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile

import numpy as np

PREP = "preparation/vae_latent_descent_20261111j"
RUN = "runs/vae_latent_descent_20261111j"
PROBE_SHA = "b94320cdb2c2ee112ce47e7f377813b6a34935d11428e0e707300dc0e54fdc06"
CONTRACT_SHA = "fe13cc302c6e1995ee037205de2e1d3f495d66633299946e83d52f9f12d27d30"
QUEUE_SHA = "e26ef9881add00e05427b0e92f5be3bb4a9531029a432734a78d4f17e84b09b7"
INPUT_SHA = "f235d9af325c56cc8f42a41587ddb532031cbd232eb0a976cb9ef6fbc2e65ab2"
SCALE, RHO = 1024., .001


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def numeric(actual, expected, label):
    # Metadata reductions only. Actual arrays/updates use exact byte hashes.
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), "numeric metadata differs: " + label)


def statistics(array):
    finite = np.isfinite(array)
    valid = array[finite].astype(np.float64)
    absval = np.abs(valid)
    nz = absval[absval != 0]
    return {"shape": list(array.shape), "dtype": str(array.dtype), "elements": int(array.size),
            "finite_count": int(finite.sum()), "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()), "nonzero_count": int(np.count_nonzero(valid)),
            "zero_count": int(np.count_nonzero(array == 0)), "l2_float64": float(np.linalg.norm(valid)),
            "max_abs": float(absval.max()) if valid.size else None,
            "min_nonzero_abs": float(nz.min()) if nz.size else None, "array_sha256": array_sha(array)}


def verify_statistics(array, recorded):
    actual = statistics(array)
    require(set(actual) == set(recorded), "gradient statistics keys differ")
    for key, value in actual.items():
        if key == "l2_float64":
            numeric(recorded[key], value, key)
        else:
            require(recorded[key] == value, "gradient statistics differ: " + key)
    return actual


def read_array(directory, row, shape):
    directory = Path(directory).resolve()
    path = (directory / row["path"]).resolve()
    require(path.parent == directory and path.suffix == ".npy", "array must be directly within producer eval")
    require(path.stat().st_size <= 2 << 20, "unexpected oversized diagnostic array")
    require(path.stat().st_size == row["bytes"] and sha_file(path) == row["file_sha256"], "array file SHA/size differs")
    a = np.load(path, allow_pickle=False)
    require(a.dtype == np.float32 and str(a.dtype) == row["dtype"] and list(a.shape) == shape == row["shape"], "array dtype/shape differs")
    require(array_sha(a) == row["array_sha256"], "array content SHA differs")
    return a


def checkpoint_passed(a):
    exact = {"enabled": True, "active": False, "restored": True, "block_count": 36,
             "wrapped": 36, "use_reentrant": False, "preserve_rng_state": True,
             "checkpoint_forward_calls": 108, "no_grad_forward_calls": 0, "recompute_calls": 108,
             "output_gradient_guards": 108, "backward_guard_calls": 108,
             "recomputed_block_indices": list(range(36))}
    return all(a.get(k) == v and type(a.get(k)) is type(v) for k, v in exact.items())


def fixed_update(source, raw):
    gradient = np.ascontiguousarray(raw / np.float32(SCALE))
    require(np.isfinite(source).all() and np.isfinite(gradient).all(), "nonfinite latent/gradient")
    rms = float(np.sqrt(np.mean(gradient.astype(np.float64) ** 2)))
    require(math.isfinite(rms) and rms > 0, "zero/nonfinite gradient RMS")
    direction = np.ascontiguousarray(gradient / np.float32(rms))
    delta = np.float32(RHO) * direction
    minus, plus = np.ascontiguousarray(source - delta), np.ascontiguousarray(source + delta)
    actual = source.astype(np.float64) - minus.astype(np.float64)
    return minus, plus, {"gradient_rms_float64": rms, "rho": RHO,
        "actual_float32_step_rms": float(np.sqrt(np.mean(actual ** 2))),
        "gradient_dot_direction": float(np.sum(gradient.astype(np.float64) * direction.astype(np.float64))),
        "predicted_first_order_decrease": float(np.sum(gradient.astype(np.float64) * actual))}


def verify_cast(before, after, recorded):
    before, after = before.astype(np.float16), after.astype(np.float16)
    delta = after.astype(np.float64) - before.astype(np.float64)
    exact = {"before_sha256": array_sha(before), "after_sha256": array_sha(after), "dtype": "float16",
             "changed_coordinates": int(np.count_nonzero(delta)), "elements": int(delta.size)}
    for key, value in exact.items():
        require(recorded[key] == value, "actual cast evidence differs: " + key)
    for key, value in {"changed_fraction": float(np.count_nonzero(delta) / delta.size),
                       "actual_cast_step_rms": float(np.sqrt(np.mean(delta ** 2))),
                       "actual_cast_max_abs_change": float(np.max(np.abs(delta)))}.items():
        numeric(recorded[key], value, key)


def validate_report(report, directory, protocol, successful):
    require(report["schema"] == "openh3.vae_latent_descent_probe.v1", "unknown report schema")
    require(type(report["complete"]) is bool and (not successful or report["complete"]), "completed queue has failed report")
    require(all(report["contract"].get(k) == v for k, v in protocol.items()), "recorded protocol differs")
    attempts, completed = {}, {}
    for stem, limit in (("forward", 8), ("backward", 3)):
        a, c = report[stem + "_attempts"], report[stem + "s_completed"]
        require(type(a) is int and type(c) is int and 0 <= c <= a <= limit, "invalid/exceeded actual budget")
        if successful:
            require(a == c == limit, "successful exact 8F3B budget failed")
        attempts[stem], completed[stem] = a, c
    require(len(report["steps"]) <= 3, "more than three declared steps")
    shapes = protocol["input_shape"]
    expected_names = {"latent_0", "latent_1", "latent_2", "latent_3", "first_positive_latent"} | {
        f"step{i}_gradient_{kind}" for i in range(1, 4) for kind in ("raw", "unscaled")}
    require(set(report["arrays"]) <= expected_names, "unexpected array artifact")
    if successful:
        require(set(report["arrays"]) == expected_names, "missing successful arrays")
    arrays, array_manifest = {}, {}
    for name, row in report["arrays"].items():
        arrays[name] = read_array(directory, row, shapes)
        array_manifest[name] = {**row, "actual_finite_count": int(np.isfinite(arrays[name]).sum()),
                                "actual_elements": int(arrays[name].size)}
    if "latent_0" in arrays:
        require(array_sha(arrays["latent_0"]) == protocol["input_sha256"], "input seed/hash differs")
    checked_updates = []
    for i in range(1, 4):
        label = f"step{i}_gradient"
        raw, unscaled = arrays.get(label + "_raw"), arrays.get(label + "_unscaled")
        if raw is None or unscaled is None:
            if successful:
                raise ValueError("missing gradient pair")
            continue
        require(array_sha(raw / np.float32(SCALE)) == array_sha(unscaled), "FP32 gradient unscale is not exact")
        if label in report["gradients"]:
            verify_statistics(raw, report["gradients"][label]["raw"])
            verify_statistics(unscaled, report["gradients"][label]["unscaled"])
        elif successful:
            raise ValueError("missing gradient statistics")
        if successful:
            require(np.isfinite(raw).all() and np.count_nonzero(raw) > 0, "successful gradient is nonfinite/zero")
        current, following = arrays.get(f"latent_{i-1}"), arrays.get(f"latent_{i}")
        if current is not None and following is not None:
            minus, plus, update = fixed_update(current, raw)
            require(array_sha(minus) == array_sha(following), "fixed normalized latent update is not exact")
            row = report["steps"][i-1]
            require(row["step"] == i, "step order differs")
            for key, value in update.items():
                numeric(row[key], value, key)
            if i == 1 and "first_positive_latent" in arrays:
                require(array_sha(plus) == array_sha(arrays["first_positive_latent"]), "positive diagnostic latent differs")
            if "float16_cast_change" in row:
                verify_cast(current, following, row["float16_cast_change"])
            checked_updates.append(i)
    if successful:
        require(len(report["steps"]) == 3 and checked_updates == [1, 2, 3], "three successful exact updates missing")
        require(all(np.isfinite(a).all() for a in arrays.values()), "nonfinite successful artifact")
    evaluations = report["evaluations"]
    expected_eval = {"initial_reference", "first_positive"} | {f"step{i}_{kind}" for i in range(1, 4) for kind in ("gradient", "next")}
    require(set(evaluations) <= expected_eval and len(evaluations) == completed["forward"], "evaluation/count mismatch")
    require(len(report["gradients"]) <= completed["backward"], "gradient/count mismatch")
    if successful:
        require(set(evaluations) == expected_eval and len(report["gradients"]) == 3, "missing successful evaluations")
    targets = set()
    for label, ev in evaluations.items():
        if label == "initial_reference": key = "latent_0"
        elif label == "first_positive": key = "first_positive_latent"
        else:
            i = int(label[4]); key = f"latent_{i-1 if label.endswith('gradient') else i}"
        if key in arrays:
            require(ev["input_sha256"] == array_sha(arrays[key]), "evaluation/input binding differs")
            require(ev["actual_cast_sha256"] == array_sha(arrays[key].astype(np.float16))
                    and ev["actual_cast_dtype"] == "float16", "evaluation decoder cast binding differs")
        if successful:
            require(ev["finite"] is True and ev["output_shape"] == protocol["RGB_shape"]
                    and ev["output_dtype"] == "torch.float32", "invalid successful RGB observation")
            require(type(ev["loss"]) in (int, float) and math.isfinite(ev["loss"]) and ev["loss"] >= 0, "invalid loss")
            require(len(ev["output_sha256"]) == len(ev["target_sha256"]) == 64, "malformed RGB/target hash")
        targets.add(ev["target_sha256"])
        if label.endswith("gradient") and successful:
            parity = ev["parity"]
            require(parity["allclose"] is True and parity["cast_exact"] is True and parity["requires_grad"] is True,
                    "direct/deployment forward parity gate failed")
            require(type(parity["bitwise_equal"]) is bool and math.isfinite(parity["max_abs_difference"])
                    and parity["max_abs_difference"] >= 0, "invalid parity measurements")
            if parity["bitwise_equal"]:
                require(parity["max_abs_difference"] == 0, "bitwise equality disagrees with difference")
                i = int(label[4])
                reference = evaluations["initial_reference" if i == 1 else f"step{i-1}_next"]
                require(ev["output_sha256"] == reference["output_sha256"], "bitwise parity disagrees with recorded RGB hashes")
    require(len(targets) <= 1, "synthetic target changed within diagnostic")
    losses = []
    for i, row in enumerate(report["steps"], 1):
        before = evaluations.get("initial_reference" if i == 1 else f"step{i-1}_next")
        after = evaluations.get(f"step{i}_next")
        if before and after and row.get("loss_after") is not None:
            numeric(row["loss_before"], before["loss"], "loss_before")
            numeric(row["loss_after"], after["loss"], "loss_after")
            numeric(row["actual_loss_decrease"], before["loss"] - after["loss"], "loss_decrease")
            if successful:
                require(row["accepted"] is True and row["actual_loss_decrease"] > 0, "strict descent failed")
            losses.append({"step": i, "before": before["loss"], "after": after["loss"], "decrease": row["actual_loss_decrease"]})
        if i == 1 and "directional_check" in row:
            dc = row["directional_check"]
            central = (evaluations["first_positive"]["loss"] - after["loss"]) / (2 * RHO)
            dot = row["gradient_dot_direction"]
            for key, value in {"loss_positive": evaluations["first_positive"]["loss"], "central_difference": central,
                    "gradient_dot_direction": dot, "central_minus_gradient_dot": central - dot,
                    "relative_magnitude_error_descriptive_only": abs(central - dot) / abs(dot),
                    "positive_loss_minus_initial": evaluations["first_positive"]["loss"] - before["loss"]}.items():
                numeric(dc[key], value, key)
            require(dc["central_positive_gate"] is bool(central > 0), "central gate does not match measurement")
            require(dc["positive_loss_increased_observation_only"] is bool(evaluations["first_positive"]["loss"] > before["loss"]), "positive observation differs")
            verify_cast(arrays["latent_0"], arrays["first_positive_latent"], dc["positive_float16_cast_change"])
            if successful:
                require(central > 0, "first central direction not positive")
        elif i == 1 and successful:
            raise ValueError("first central check missing")
    checkpoint = {label: checkpoint_passed(a) for label, a in report["checkpoint"].items()}
    if successful:
        require(report["protocol_complete"] is True and report["error"] is None and report["phase"] == "completed_local_descent", "malformed successful state")
        require(set(report["gates"]) == {"protocol", "frozen", "runtime", "budget", "restoration"}
                and all(v is True for v in report["gates"].values()), "producer gates failed")
        require(set(checkpoint) == {f"step{i}_gradient" for i in range(1,4)} and all(checkpoint.values()), "checkpoint/restoration counts differ")
        require(report["frozen_before"] == report["frozen_after"] and report["frozen_versions_unchanged"] is True
                and report["frozen_before"]["parameter_count"] > 0 and len(report["frozen_before"]["sha256"]) == 64, "frozen named_parameters gate failed")
        require(report["runtime_before"] == report["runtime_after"] and report["runtime_before"], "runtime changed or absent")
        require(report["runtime_before"]["jit_compile_false"] is True and report["runtime_before"]["vae_dtype"] == "torch.float16", "runtime dtype/JIT differs")
    return {"attempts": attempts, "completed": completed, "verified_array_manifest": array_manifest,
            "exact_normalized_updates_verified": checked_updates, "losses": losses,
            "checkpoint_gate_observations": checkpoint, "first_central": report["steps"][0].get("directional_check") if report["steps"] else None}


def audit(root):
    root = Path(root).resolve(); prep, run = root / PREP, root / RUN
    require(sha_file(prep / "queue.json") == QUEUE_SHA and sha_file(prep / "execution_contract.json") == CONTRACT_SHA, "queue/execution pins differ")
    execution, queue = read_json(prep / "execution_contract.json"), read_json(prep / "queue.json")
    require(Path(execution["root"]).resolve() == root and execution["frozen_probe_sha256"] == PROBE_SHA, "wrong workspace/producer")
    protocol = execution["frozen_protocol"]
    require(protocol["input_sha256"] == INPUT_SHA and protocol["loss_scale"] == SCALE and protocol["rho_latent_rms"] == RHO, "wrong fixed diagnostic")
    for name, expected in protocol["implementation_sha256"].items():
        require(Path(name).name == name and sha_file(prep / "code" / name) == expected, "producer source pin differs: " + name)
    admission = read_json(prep / "admission.json")
    require(admission["approved_for_single_card_vae_latent_descent"] is True and admission["physical_card"] == 7
            and admission["seed"] == 102 and admission["queue_sha256"] == QUEUE_SHA, "fresh launch admission differs")
    state = read_json(run / "queue_status.json")
    require(state["state"] in ("completed", "failed"), "producer has not reached terminal state")
    report = read_json(run / "eval/report.json")
    if "manifest_sha256" in report["contract"]:
        require(report["contract"]["manifest_sha256"] == execution["manifest_sha256"], "manifest differs from admission")
    elif state["state"] == "completed":
        raise ValueError("successful asset manifest binding absent")
    facts = validate_report(report, run / "eval", protocol, state["state"] == "completed")
    runtime_pins = {str((root / path).resolve()): digest for path, digest in queue["implementation_sha256"].items()}
    for label in ("runtime_before", "runtime_after"):
        if label in report:
            runtime = report[label]
            require(runtime["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "7", "physical card observation differs")
            for row in runtime["imported_source"].values():
                path = Path(row["path"]).resolve()
                pinned = runtime_pins.get(str(path))
                require(pinned == row["sha256"] and sha_file(path) == row["sha256"], "runtime source evidence differs")
    successful = state["state"] == "completed" and report["complete"]
    return {"schema": "openh3.vae_descent_independent_audit.v1", "audit_complete": True,
            "complete": bool(successful), "producer_terminal_state": state["state"],
            "numerical_acceptance": "passed_fixed_synthetic_local_descent" if successful else "producer_failed_not_accepted",
            "producer_error": report["error"], "facts": facts, "errors": [],
            "pins": {"probe_sha256": PROBE_SHA, "contract_sha256": CONTRACT_SHA, "queue_sha256": QUEUE_SHA},
            "report_sha256": sha_file(run / "eval/report.json"), "admission_sha256": sha_file(prep / "admission.json"),
            "queue_status_sha256": sha_file(run / "queue_status.json"),
            "limitations": ["RGB arrays are not saved: scalar losses/forward parity are source-bound producer attestations, not independently rescored pixels",
                            "Only named_parameters bytes and versions are frozen; buffers are outside this fingerprint",
                            "Three local fixed steps do not establish gradient accuracy, gradient repeatability, adapter training, restoration quality, or joint H3 feasibility",
                            "Earlier exact-repeatability failure and skipped paired-history branch remain unchanged"],
            "actions": {"NPU_calls": 0, "Torch_imports": 0, "model_loads": 0, "producer_writes": 0, "arrays_copied": 0}}


def pack(root, out, own):
    root, out, own = map(lambda p: Path(p).resolve(), (root, out, own))
    prep, run = root / PREP, root / RUN
    files = [out / "audit.json", run / "queue_status.json"]
    for directory in (prep, own):
        files += [p for p in directory.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md", ".log")]
    files += list((prep / "code").glob("*.py")) + list((prep / "code/tests").glob("*.py"))
    files += list((run / "eval").glob("*.json")) + list((run / "logs").glob("*.log"))
    files = sorted(set(p.resolve() for p in files)); require(len(files) <= 100, "small archive count exceeded")
    blobs, rows, total = {}, {}, 0
    for p in files:
        require(p.is_relative_to(root) and p.is_file() and p.suffix in (".py", ".json", ".md", ".log"), "archive allowlist violation")
        require(p.stat().st_size <= (16 << 20) - total, "small archive 16MiB bound exceeded")
        data = p.read_bytes(); total += len(data)
        require(total <= 16 << 20, "small archive changed/exceeded byte bound")
        name = str(p.relative_to(root)); blobs[name] = data
        rows[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"schema": "openh3.vae_descent_small_evidence.v1", "files": rows, "arrays_included": False, "weights_included": False}
    blobs["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    require(total + len(blobs["MANIFEST.json"]) <= 16 << 20, "manifest exceeds small archive bound")
    target = out / "evidence.tar.gz"
    with target.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as t:
            for name, data in sorted(blobs.items()):
                member = tarfile.TarInfo(name); member.size = len(data); member.mode = 0o644; member.mtime = 0
                t.addfile(member, io.BytesIO(data))
    return {"path": str(target), "sha256": sha_file(target), "bytes": target.stat().st_size,
            "manifest_sha256": hashlib.sha256(blobs["MANIFEST.json"]).hexdigest(), "members": len(blobs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="fresh independent output directory; no resume")
    args = parser.parse_args(); root, out = args.root.resolve(), args.out.resolve()
    require(out.is_relative_to(root) and not out.is_relative_to(root / RUN), "output must be in workspace, outside producer run")
    state = read_json(root / RUN / "queue_status.json")
    require(state["state"] in ("completed", "failed"), "not terminal; no polling or writes")
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(root)
    except Exception as error:
        result = {"schema": "openh3.vae_descent_independent_audit.v1", "audit_complete": False, "complete": False,
                  "producer_terminal_state": state["state"], "numerical_acceptance": "invalid_or_missing_evidence",
                  "errors": [repr(error)], "actions": {"NPU_calls": 0, "Torch_imports": 0, "producer_writes": 0, "arrays_copied": 0}}
    (out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    status = {"complete": result["complete"], "producer_terminal_state": state["state"], "audit_sha256": sha_file(out / "audit.json")}
    try:
        status["archive"] = pack(root, out, Path(__file__).parent)
    except Exception as error:
        status["complete"] = False; status["archive_error"] = repr(error)
    (out / "finalize_status.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    print(json.dumps(status)); return 0 if status["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
