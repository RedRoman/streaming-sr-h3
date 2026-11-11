"""Small CPU tests for frozen-restorer confidence fitting; never load H3/NPU."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from calibrate_rphi import (assert_exact_outputs, assert_fidelity_frozen, build_frozen_models,
                               calibration_step, fidelity_state, output_snapshot,
                               save_calibrated_checkpoint, validate_train_contract)
    from rphi_model import RphiModel, KNOWN_TASKS
    from train_rphi import state_digest
    from eval_rphi import validate_contract


@unittest.skipIf(torch is None, "CPU torch is not installed; run in the existing project venv")
class FrozenCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(19)
        self.source_model = RphiModel(variant="blind", ch=8, blocks=1)
        # Nonzero residual and FiLM weights make incomplete restoration copies visible.
        with torch.no_grad():
            self.source_model.backbone.tail.weight.normal_(std=.01)
            self.source_model.backbone.blocks[0].film.weight.normal_(std=.01)
        self.observed = torch.rand(4, 3, 3, 8, 12)
        self.target = self.observed[:, -1].clone()
        self.contract = {"steps": 1000, "target": "clean_original_recording", "task_order": list(KNOWN_TASKS),
                         "train_manifest_sha256": "a" * 64, "dev_manifest_sha256": "b" * 64,
                         "train_source_ids": ["train-source"], "train_source_sha256": ["c" * 64],
                         "dev_source_ids": ["dev-source"]}
        self.checkpoint = {"schema": "rphi.checkpoint.v1", "model_config": self.source_model.model_config,
                           "state_dict": copy.deepcopy(self.source_model.state_dict()),
                           "fidelity_state_sha256": state_digest(fidelity_state(self.source_model)),
                           "training_contract": self.contract, "step": 1000}

    def test_strict_copy_freezes_all_fidelity_and_reproduces_rgb(self):
        source, model, expected = build_frozen_models(self.checkpoint)
        self.assertEqual(state_digest(fidelity_state(model)), expected)
        self.assertEqual(set(fidelity_state(model)), set(self.checkpoint["state_dict"]))
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("confidence_head."))
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.condition_encoder.training)
        differences = assert_exact_outputs(output_snapshot(model, self.observed), output_snapshot(source, self.observed))
        self.assertEqual(differences["rgb"], 0.)
        self.assertEqual(assert_fidelity_frozen(model, expected), expected)

    def test_head_updates_without_fidelity_or_input_target_gradients(self):
        source, model, expected = build_frozen_models(self.checkpoint)
        before = output_snapshot(source, self.observed)
        initial_head = state_digest(model.confidence_head.state_dict())
        optimizer = torch.optim.AdamW(model.confidence_head.parameters(), lr=.01, weight_decay=0.)
        observed = self.observed.clone().requires_grad_()
        target = self.target.clone().requires_grad_()
        losses = []
        for _ in range(12):
            loss, norm = calibration_step(model, optimizer, observed, target)
            losses.append(loss)
            self.assertTrue(norm >= 0.)
        self.assertLess(losses[-1], losses[0])
        self.assertNotEqual(initial_head, state_digest(model.confidence_head.state_dict()))
        self.assertEqual(assert_fidelity_frozen(model, expected), expected)
        self.assertEqual(assert_exact_outputs(output_snapshot(model, self.observed), before)["rgb"], 0.)
        self.assertIsNone(observed.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.startswith("confidence_head.")))

    def test_rejects_wrong_variant_missing_fidelity_and_corrupt_source(self):
        bad = copy.deepcopy(self.checkpoint)
        bad["model_config"]["variant"] = "known"
        with self.assertRaisesRegex(ValueError, "existing blind"):
            build_frozen_models(bad)
        bad = copy.deepcopy(self.checkpoint)
        bad["state_dict"].pop(next(iter(bad["state_dict"])))
        with self.assertRaises(RuntimeError):
            build_frozen_models(bad)
        bad = copy.deepcopy(self.checkpoint)
        bad["fidelity_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            build_frozen_models(bad)

    def test_future_model_config_is_preserved_and_used_for_both_models(self):
        calls = []
        class ExtendedModel(RphiModel):
            def __init__(self, condition_encoder_version="native-v2", **kwargs):
                calls.append(condition_encoder_version)
                super().__init__(**kwargs)
                self.model_config["condition_encoder_version"] = condition_encoder_version
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["model_config"]["condition_encoder_version"] = "native-v2"
        _, model, _ = build_frozen_models(checkpoint, model_factory=ExtendedModel)
        self.assertEqual(calls, ["native-v2", "native-v2"])
        self.assertEqual(model.model_config["condition_encoder_version"], "native-v2")

    def test_wrong_optimizer_and_mutated_fidelity_are_rejected(self):
        _, model, expected = build_frozen_models(self.checkpoint)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        with self.assertRaisesRegex(ValueError, "exactly the confidence_head"):
            calibration_step(model, optimizer, self.observed, self.target)
        with torch.no_grad():
            model.condition_encoder.net[0].weight.add_(.01)
        with self.assertRaisesRegex(AssertionError, "parameters or buffers changed"):
            assert_fidelity_frozen(model, expected)

    def test_only_exact_training_manifest_and_training_records_accepted(self):
        dataset = SimpleNamespace(manifest_sha256="a" * 64, tasks=list(KNOWN_TASKS),
                                  records=[{"source_id": "train-source", "source_sha256": "c" * 64, "split": "train"}])
        self.assertEqual(validate_train_contract(self.checkpoint, dataset), self.contract)
        dataset.manifest_sha256 = "b" * 64
        with self.assertRaisesRegex(ValueError, "exact train manifest"):
            validate_train_contract(self.checkpoint, dataset)
        dataset.manifest_sha256 = "a" * 64
        dataset.records[0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "only train"):
            validate_train_contract(self.checkpoint, dataset)
        dataset.records[0].update(split="train", source_id="dev-source")
        with self.assertRaisesRegex(ValueError, "IDs/hashes"):
            validate_train_contract(self.checkpoint, dataset)

    def test_checkpoint_reloads_exactly_and_existing_evaluator_accepts_contract(self):
        _, model, expected = build_frozen_models(self.checkpoint)
        optimizer = torch.optim.AdamW(model.confidence_head.parameters(), lr=.01, weight_decay=0.)
        calibration_step(model, optimizer, self.observed, self.target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrated.pt"
            saved = save_calibrated_checkpoint(path, model, optimizer, self.checkpoint, "d" * 64,
                                               {"steps": 1, "statistical_scope": "training only"}, 1, {}, {"identical": True})
            disk = torch.load(path, map_location="cpu", weights_only=True)
            reloaded = RphiModel(**disk["model_config"]).eval()
            reloaded.load_state_dict(disk["state_dict"], strict=True)
            self.assertEqual(state_digest(fidelity_state(reloaded)), expected)
            self.assertEqual(disk["step"], 1000)
            self.assertEqual(disk["calibration_step"], 1)
            self.assertEqual(disk["source_checkpoint_sha256"], "d" * 64)
            self.assertEqual(disk["training_contract"], self.contract)
            self.assertEqual(assert_exact_outputs(output_snapshot(reloaded, self.observed, include_confidence=True),
                                                 output_snapshot(model, self.observed, include_confidence=True))["log_variance"], 0.)
            dev = [{"source_id": "dev-source", "source_sha256": "e" * 64, "split": "dev", "task": task,
                    "target_key": "window0", "target_pixel_sha256": "f" * 64, "observation_pixel_sha256": "1" * 64}
                   for task in KNOWN_TASKS]
            self.assertEqual(validate_contract(disk, "b" * 64, dev), self.contract)
            with self.assertRaises(FileExistsError):
                save_calibrated_checkpoint(path, model, optimizer, self.checkpoint, "d" * 64, {}, 1, {}, {})


if __name__ == "__main__":
    unittest.main()
