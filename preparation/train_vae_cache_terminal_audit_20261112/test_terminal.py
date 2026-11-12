import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_arrays import array_sha, file_sha, pil_same_geometry_hashes
from audit_inputs import CHECKPOINT_SHA, SOURCE, SOURCE_SHA, RGB_SHAPE, selected_row
from audit_terminal import audit, budget, canonical, frozen, loader_modes, output_relations, preprocessing


def guide_manifest():
    arrays = {k: {"shape": RGB_SHAPE, "dtype": "float32"} for k in ("raw_draft", "rphi_rgb", "rphi_raw_rgb")}
    row = {"source_id": SOURCE, "source_sha256": SOURCE_SHA, "split": "train", "role": "train",
           "complete": True, "checkpoint_sha256": CHECKPOINT_SHA,
           "target_role": "supervision_reference_only_never_model_input", "frame_indices": list(range(22)), "arrays": arrays}
    return {"schema": "rphi.train_rgb_guide_export.v1", "complete": True, "rgb_export_complete": True,
            "role": "train", "split": "train", "source_ids": [SOURCE], "controls_included": [], "samples": 4,
            "checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_step": 3000, "condition_mode": "observations_only",
            "model_config": {"variant": "blind", "k": 3, "ch": 64, "blocks": 8, "condition_dim": 6, "observation_encoder": "legacy"},
            "frozen_state_sha256_before": "same", "frozen_state_sha256_after": "same", "all_parameters_frozen": True,
            "source_isolation": {"source_id_overlap": 0, "source_sha256_overlap": 0, "train_source_ids": [SOURCE],
                "dev_source_ids": ["dev"], "train_source_sha256": [SOURCE_SHA], "dev_source_sha256": ["other"]},
            "per_sample": [{**copy.deepcopy(row), "task": task} for task in ("sr", "deblur", "denoise", "compression")]}


class TerminalTests(unittest.TestCase):
    def test_cpu_checkpoint_deserialization_is_not_Rphi_model_construction(self):
        mode = {"VAE_and_context_actual_SHA": True, "DiT_loaded": False,
                "Rphi_model_constructed": False, "Rphi_checkpoint_deserialized_on_CPU": True,
                "actual_vae_dtype": "torch.float16"}
        loader_modes(mode)
        with self.assertRaisesRegex(ValueError, "model construction"):
            loader_modes({**mode, "Rphi_model_constructed": True})
        with self.assertRaisesRegex(ValueError, "inspection omission"):
            loader_modes({**mode, "Rphi_checkpoint_deserialized_on_CPU": False})

    def test_train_identity_relabel_controls_and_source_overlap_rejected(self):
        parent = guide_manifest(); self.assertEqual(selected_row(parent)["task"], "compression")
        for mutate in (lambda p: p.update(role="dev"),
                       lambda p: p["per_sample"][-1].update(control="identity"),
                       lambda p: p["source_isolation"]["dev_source_sha256"].append(SOURCE_SHA)):
            value = copy.deepcopy(parent); mutate(value)
            with self.assertRaises(ValueError): selected_row(value)

    def test_budget_rejects_extra_or_missing_encode_decode_and_forbidden_model(self):
        report = {"actual": {"encode_calls": 2, "stage_encode_calls": 2, "decode_calls": 1, "stage_decode_calls": 1,
            "PIL_fromarray_calls": {"raw": 22, "rphi": 22}, "forbidden_calls": [],
            "Rphi_calls": 0, "H3_sampler_calls": 0, "NFE": 0, "backward_calls": 0, "optimizer_steps": 0}}
        budget(report, True)
        for key, number in (("encode_calls", 3), ("stage_encode_calls", 1), ("decode_calls", 0), ("NFE", 1)):
            bad = copy.deepcopy(report); bad["actual"][key] = number
            with self.assertRaises(ValueError): budget(bad, True)
        early = copy.deepcopy(report)
        early["actual"].update(encode_calls=0, stage_encode_calls=0, decode_calls=0, stage_decode_calls=0, PIL_fromarray_calls={})
        budget(early, False)

    def test_preprocessing_checks_actual_observer_and_input_hashes(self):
        # Tiny pixel math fixture, with full-size geometry metadata kept separate.
        facts = pil_same_geometry_hashes(np.array([[[[.2, .5, 1.]]]], np.float32))
        independent = {"preprocessing": {label: facts for label in ("raw", "rphi")}}
        declared = {"before_P_float32_sha256": facts["before_float32_sha256"], "PIL_uint8_sha256": facts["pil_uint8_sha256"],
            "after_P_float32_sha256": facts["after_float32_sha256"], "shape": RGB_SHAPE, "dtype": "float32", "PIL_frames": 22}
        observed = {"shape": RGB_SHAPE, "dtype": "float32", "PIL_calls": 22, "matches_CPU_preprocessing": True,
            "actual_PIL_uint8_sha256": facts["pil_uint8_sha256"], "actual_vae_input_sha256": facts["after_float32_sha256"]}
        report = {"inspection": {"preprocessing": {x: copy.deepcopy(declared) for x in ("raw", "rphi")}},
            "observed_preprocessing": {x: copy.deepcopy(observed) for x in ("raw", "rphi")}}
        preprocessing(report, independent)
        report["observed_preprocessing"]["raw"]["actual_vae_input_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "model input"):
            preprocessing(report, independent)

    def test_frozen_registered_buffer_tamper_or_failed_lifecycle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            entry = {"name": "toy", "shape": [1], "dtype": "torch.float16", "sha256_bytes": "0" * 64,
                     "requires_grad": False, "grad_is_none": True, "device": "npu:0", "version": 1}
            initial = {"parameters": [entry], "registered_buffers": [{**entry, "name": "buffer"}]}
            for group in ("parameters", "registered_buffers"):
                initial[group + "_sha256"] = canonical([{k:r[k] for k in ("name", "shape", "dtype", "sha256_bytes")} for r in initial[group]])
            report = {"frozen_check": {"passed": True, "failures": [], "tensors_checked": 2}}
            for name in ("frozen_initial", "frozen_final"):
                path = evidence / (name + ".json"); path.write_text(json.dumps(initial))
                report[name] = {"path": path.name, "sha256": file_sha(path)}
            report["frozen_final"].update(parameter_bytes_unchanged=True, registered_buffer_bytes_unchanged=True)
            frozen(report, evidence, True)
            final = copy.deepcopy(initial); final["registered_buffers"][0]["version"] += 1
            path.write_text(json.dumps(final)); report["frozen_final"]["sha256"] = file_sha(path)
            with self.assertRaisesRegex(ValueError, "version"):
                frozen(report, evidence, True)
            final["registered_buffers"][0]["sha256_bytes"] = "f" * 64
            path.write_text(json.dumps(final)); report["frozen_final"]["sha256"] = file_sha(path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                frozen(report, evidence, True)

    def test_decode_must_use_Z_R_and_its_exact_float16_cast(self):
        shape = [1, 24, 7, 30, 52]
        raw = np.zeros(shape, np.float32); rphi = np.full(shape, .12345, np.float32)
        loaded = {"raw_warm_start_latent": raw, "rphi_guide_latent": rphi}
        stage = {"shape": [1, 3, 22, 480, 832], "dtype": "torch.float16", "array_sha256": "0" * 64}
        report = {"actual_decode_input": {"array_sha256": array_sha(rphi), "shape": shape, "dtype": "torch.float32"},
            "stage_inputs": {"encode_raw": stage, "encode_rphi": stage,
                "decode_rphi": {"shape": shape, "dtype": "torch.float16", "array_sha256": array_sha(rphi.astype(np.float16))}}}
        output_relations(report, loaded)
        wrong = copy.deepcopy(report); wrong["actual_decode_input"]["array_sha256"] = array_sha(raw)
        with self.assertRaisesRegex(ValueError, "saved Z_R"):
            output_relations(wrong, loaded)
        report["stage_inputs"]["decode_rphi"]["array_sha256"] = array_sha(raw.astype(np.float16))
        with self.assertRaisesRegex(ValueError, "FP16 decode"):
            output_relations(report, loaded)

    def test_not_launched_and_missing_failed_report_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / "run"; run.mkdir()
            paths = {"run": run, "evidence": run / "eval"}
            for state, verdict in (("not_launched", "not_launched_not_accepted"), ("failed", "producer_failed_no_report")):
                (run / "queue_status.json").write_text(json.dumps({"state": state}))
                result = audit(root, {}, paths)
                self.assertFalse(result["complete"])
                self.assertEqual(result["engineering_verdict"], verdict)
                self.assertEqual(result["quality_verdict"], "not_assessed")
            (run / "queue_status.json").write_text(json.dumps({"state": "running"}))
            with self.assertRaisesRegex(ValueError, "not terminal"):
                audit(root, {}, paths)


if __name__ == "__main__":
    unittest.main()
