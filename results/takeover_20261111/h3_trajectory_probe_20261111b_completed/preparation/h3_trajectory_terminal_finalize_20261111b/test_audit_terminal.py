"""Small CPU fixtures; no H3, Torch, model, source-video or accelerator imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_terminal as audit


def save_json(path, value):
    path.write_text(json.dumps(value))


class TerminalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "run/eval"
        self.run.mkdir(parents=True)
        self.contract = {"root": str(self.root), "probe_sha256": audit.PROBE_SHA,
                         "implementation_sha256": {"trajectory.py": audit.TRAJECTORY_SHA, "torch/_ops.py": "c" * 64, "torch/_classes.py": "d" * 64, "comfy_kitchen/backends/cpu.py": "e" * 64}}
        self.contract_path = self.root / "contract.json"
        save_json(self.contract_path, self.contract)
        self.contract_sha = audit.sha_file(self.contract_path)
        runtime_path = str(self.root / "trajectory.py")
        imported = {"passed": True, "unbound": [], "modules": {"trajectory": {"path": runtime_path, "sha256": audit.TRAJECTORY_SHA}}}
        imported["synthetic_namespaces"] = {
            name: {"kind": "exact_Torch_ModuleType_subclass_namespace", "class": class_name,
                   "inherited_pseudo_file": file, "defining_source_path": str(self.root / "torch" / file),
                   "defining_source_sha256": digest}
            for name, class_name, file, digest in (
                ("torch.ops", "torch._ops._Ops", "_ops.py", "c" * 64),
                ("torch.classes", "torch._classes._Classes", "_classes.py", "d" * 64))}
        imported["namespace_packages"] = {"comfy_kitchen.backends": {
            "kind": "PEP420_namespace_package", "loader_type": "_frozen_importlib_external.NamespaceLoader",
            "directories": [str(self.root / "comfy_kitchen/backends")],
            "source_files": {str(self.root / "comfy_kitchen/backends/cpu.py"): "e" * 64}}}
        self.report = {"schema": "openh3.latent_trajectory_probe.v1", "complete": True,
                       "contract": {**self.contract, "execution_contract_sha256": self.contract_sha},
                       "verified_runtime_files": {str(self.root / key): digest for key, digest in self.contract["implementation_sha256"].items()},
                       "actual": {"reference_NFE": 3, "candidate_NFE": 3, "backward_attempts": 1,
                                  "completed_backward_calls": 1, "VAE_calls": {}, "optimizer_steps": 0, "RGB_loss_calls": 0},
                       "block_forward_calls": {p: {str(i): 3 for i in range(50)} for p in ("reference", "candidate", "backward")},
                       "checkpoint_recomputed_block_calls": 150,
                       "wrapper_attributes_restored": True, "session_inactive": True,
                       "stale_backward_guard_rejected": True, "vae_devices_initial": ["cpu"], "vae_devices_final": ["cpu"],
                       "actual_imports_before_sampling": imported, "actual_imports": copy.deepcopy(imported),
                       "precision": {"logical_device": 0, "jit_compile_false": True, "matmul_allow_hf32": False, "conv_allow_hf32": False},
                       "asset_bytes_sha_verified": True, "adapter_values_unchanged": True}
        self.make_noise()
        self.make_outputs()
        self.make_gradients()
        self.make_frozen()
        self.save()

    def save(self, successful=True):
        save_json(self.run / ("report.json" if successful else "failure.json"), self.report)
        save_json(self.run.parent / "queue_status.json", {"state": "completed" if successful else "failed"})

    def artifact(self, path):
        return {"path": str(path), "sha256": audit.sha_file(path)}

    def make_noise(self):
        guide, raw, initial, noise = [audit.array_sha(np.array([x], np.float32)) for x in range(4)]
        record = {"seed": 101, "packed_initial_sha256": initial, "packed_noise_sha256": noise,
                  "joint_shapes": [audit.SHAPES["video"], audit.SHAPES["audio"]]}
        self.report["noise_observations"] = [{"phase": phase, **record} for phase in ("reference", "candidate")]
        self.report["noise_and_initial_match"] = True
        self.report["inputs"] = {"seed": 101, "audio_init": "zeros", "distinct": True,
            **{name: {"shape": audit.SHAPES["video"], "dtype": "torch.float32", "sha256_bytes": sha}
               for name, sha in (("guide", guide), ("raw_draft", raw))}}
        self.report["candidate_contract"] = {"guide_latent_sha256_float32": guide,
            "raw_warm_start_latent_sha256_float32": raw, "packed_cpu_noise_sha256_float32": noise,
            "packed_initial_latent_sha256_float32": initial, "seed": 101, "audio_init": "zeros",
            "actual_forward_NFE": 3, "sigma_schedule_float32": np.array([.909375, 6/7, 2/3, 0], np.float32).tolist(),
            "checkpoint_blocks": True, "checkpoint_decoder_blocks": False, "target_used_as_model_input": False, "cfg": 1}

    def make_outputs(self):
        arrays = {name: np.full(shape, .5, np.float32) for name, shape in audit.SHAPES.items()}
        path = self.run / "outputs.npz"
        np.savez(path, **{a + "_" + s: v for a in ("reference", "candidate") for s, v in arrays.items()})
        self.report["latent_outputs"] = self.artifact(path)
        reference = self.run / "reference.npz"
        np.savez(reference, **arrays)
        self.report["reference_latents"] = self.artifact(reference)
        self.report["parity"] = {s: {"torch_equal": True, "finite": True, "same_shape": True,
            "reference_shape": list(v.shape), "candidate_shape": list(v.shape), "max_abs": 0,
            "reference_sha256_float32": audit.array_sha(v), "candidate_sha256_float32": audit.array_sha(v)} for s, v in arrays.items()}

    def make_gradients(self):
        directory = self.run / "gradients"
        directory.mkdir()
        initial, rows = {}, {}
        b_square = 0
        for name in sorted(audit.NAMES):
            is_b = name.endswith(".lora_b")
            shape = (2, 8) if is_b else (8, 2)
            array = np.full(shape, .125 if is_b else 0, np.float32)
            path = directory / (name + ".npy")
            np.save(path, array)
            norm = float(np.linalg.norm(array.astype(np.float64)))
            b_square += norm * norm if is_b else 0
            initial[name] = {"shape": list(shape), "dtype": "torch.float32", "all_zero": is_b}
            rows[name] = {"present": True, "array_file": str(path), "array_file_sha256": audit.sha_file(path),
                "shape": list(shape), "dtype": "torch.float32", "sha256_bytes": audit.array_sha(array),
                "elements": 16, "finite_count": 16, "nonzero_finite_count": 16 if is_b else 0,
                "nan_count": 0, "inf_count": 0, "all_finite": True, "finite_l2": norm}
        self.report["adapter_initial"] = {"modules": 200, "trainable_tensors": 400, "rank": 8, "tensors": initial}
        self.report["adapter_gradients"] = {"tensors": rows, "missing": [], "present_tensors": 400,
            "all_present_finite": True, "B_nonzero_tensors": 200, "B_l2": b_square ** .5, "passed": True}

    def make_frozen(self):
        rows = [{"name": "parameter:base.weight", "shape": [2], "dtype": "torch.float32", "sha256_bytes": "a" * 64}]
        digest = audit.hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        record = {"sha256": digest, "tensors": rows, "parameter_tensors": 1, "buffer_tensors": 0}
        for suffix in ("before_reference", "after_restore", "after_backward"):
            save_json(self.run / ("base_" + suffix + ".json"), record)
            self.report["base_sha256_" + suffix] = digest
        for suffix in ("after_injection", "after_backward"):
            self.report["base_versions_" + suffix] = {"passed": True, "failures": [], "tensors_checked": 1}

    def run_audit(self):
        return audit.audit_run(self.run, self.contract_path, self.contract_sha)

    def test_explicit_new_producer_pin_cannot_pass_under_old_admission(self):
        new_sha = "b" * 64
        self.contract["probe_sha256"] = new_sha
        save_json(self.contract_path, self.contract)
        self.contract_sha = audit.sha_file(self.contract_path)
        self.report["contract"] = {**self.contract, "execution_contract_sha256": self.contract_sha}
        self.save()
        with self.assertRaisesRegex(ValueError, "unreviewed producer"):
            self.run_audit()
        result = audit.audit_run(self.run, self.contract_path, self.contract_sha, probe_sha=new_sha)
        self.assertTrue(result["complete"])
        self.assertEqual(result["admitted_producer_sha256"], new_sha)

    def test_synthetic_namespace_definition_pin_required_and_not_pseudo_path(self):
        self.report["actual_imports"]["synthetic_namespaces"]["torch.ops"]["defining_source_path"] = "_ops.py"
        self.save()
        self.assertFalse(self.run_audit()["complete"])
        self.report["actual_imports"] = copy.deepcopy(self.report["actual_imports_before_sampling"])
        del self.report["actual_imports"]["synthetic_namespaces"]
        self.save()
        self.assertFalse(self.run_audit()["complete"])

    def test_PEP420_real_source_map_complete_and_pinned(self):
        row = self.report["actual_imports"]["namespace_packages"]["comfy_kitchen.backends"]
        row["source_files"] = {}
        self.save()
        self.assertFalse(self.run_audit()["complete"])
        self.report["actual_imports"] = copy.deepcopy(self.report["actual_imports_before_sampling"])
        self.report["actual_imports"]["namespace_packages"]["comfy_kitchen.backends"]["loader_type"] = "SourceFileLoader"
        self.save()
        self.assertFalse(self.run_audit()["complete"])

    def test_complete_400_real_array_validation(self):
        result = self.run_audit()
        self.assertTrue(result["complete"])
        self.assertEqual(result["checks"]["actual_gradient_arrays"]["facts"]["verified_files"], 400)
        self.assertEqual(result["checks"]["budget"]["facts"]["explicit_NFE"], 6)
        self.assertEqual(result["actions"]["gradient_files_copied"], 0)

    def test_gradient_file_corruption_is_rejected(self):
        path = Path(next(iter(self.report["adapter_gradients"]["tensors"].values()))["array_file"])
        with path.open("r+b") as stream:
            stream.seek(-1, 2); stream.write(b"X")
        result = self.run_audit()
        self.assertFalse(result["complete"])
        self.assertEqual(result["checks"]["actual_gradient_arrays"]["status"], "invalid")

    def test_actual_producer_workspace_relative_paths(self):
        for key in ("latent_outputs", "reference_latents"):
            self.report[key]["path"] = str(Path(self.report[key]["path"]).relative_to(self.root))
        for row in self.report["adapter_gradients"]["tensors"].values():
            row["array_file"] = str(Path(row["array_file"]).relative_to(self.root))
        self.save()
        self.assertTrue(self.run_audit()["complete"])

    def test_forged_gradient_content_sha_is_rejected(self):
        next(iter(self.report["adapter_gradients"]["tensors"].values()))["sha256_bytes"] = "f" * 64
        self.save()
        self.assertFalse(self.run_audit()["complete"])

    def test_inconsistent_output_actual_array_blocks_fake_torch_equal(self):
        path = Path(self.report["latent_outputs"]["path"])
        with np.load(path) as saved: arrays = {k: saved[k].copy() for k in saved.files}
        arrays["candidate_audio"].flat[0] += np.float32(1e-6)
        np.savez(path, **arrays)
        self.report["latent_outputs"] = self.artifact(path)
        self.save()
        self.assertFalse(self.run_audit()["complete"])

    def test_noise_and_block_recompute_mismatches_rejected(self):
        self.report["noise_observations"][1]["seed"] = 102
        self.report["block_forward_calls"]["backward"]["0"] = 2
        self.save()
        result = self.run_audit()
        self.assertEqual(result["checks"]["budget"]["status"], "invalid")
        self.assertEqual(result["checks"]["noise_initial_inputs"]["status"], "invalid")

    def test_missing_gradient_does_not_pass_with_other_nonzero_B(self):
        name = sorted(audit.NAMES)[0]
        self.report["adapter_gradients"]["tensors"][name] = {"present": False}
        self.save()
        self.assertFalse(self.run_audit()["complete"])

    def test_failed_before_first_NFE_retains_failure_without_inventing_evidence(self):
        self.report = {k: self.report[k] for k in ("schema", "contract", "verified_runtime_files", "actual")}
        self.report["complete"] = False
        self.report["phase"] = "load_and_verify_assets"
        for key in ("reference_NFE", "candidate_NFE", "backward_attempts", "completed_backward_calls"):
            self.report["actual"][key] = 0
        self.report["failure"] = {"message": "synthetic load failure"}
        self.save(False)
        result = self.run_audit()
        self.assertEqual(result["numerical_acceptance"], "producer_failed_not_accepted")
        self.assertEqual(result["checks"]["actual_output_arrays"]["status"], "unavailable")
        self.assertFalse(result["complete"])

    def test_failed_producer_with_available_valid_arrays_stays_failed(self):
        self.report["complete"] = False
        self.report["failure"] = {"message": "synthetic final gate failure"}
        self.save(False)
        result = self.run_audit()
        self.assertEqual(result["numerical_acceptance"], "producer_failed_not_accepted")
        self.assertEqual(result["checks"]["actual_gradient_arrays"]["facts"]["verified_files"], 400)

    def test_nonterminal_queue_refused(self):
        save_json(self.run.parent / "queue_status.json", {"state": "running"})
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
