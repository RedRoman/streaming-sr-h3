"""New cohort orchestration tests; no real source decode or model imports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import history_utility_data as D
import materialize_history_utility_pilot as M

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RECEIPT = ROOT / "results/takeover_20261111/mixed_temporal_single_source_smoke_completed/complete.json"


def write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True))


class MaterializerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.c = D._json(ROOT / "preparation/history_utility_materializer_20261112a/execution_contract.json")
        plan = D._json(ROOT / "configs/openh3_20261111/mixed_temporal_data_plan_20261111.json")
        self.fake_sources = {}
        for sid in M.SELECTED["train"] + M.SELECTED["dev"]:
            path = self.base / (sid + ".mp4"); path.write_bytes((sid + " synthetic unit fixture").encode())
            self.fake_sources[sid] = {"path": str(path), "bytes": path.stat().st_size, "sha256": D.file_sha(path)}
        self.c["original_sources"] = self.fake_sources
        for split in ("train", "dev"):
            rows = [json.loads(line) for line in (ROOT / f"results/takeover_20261111/materialized/{split}_materialized.jsonl").read_text().splitlines()]
            for row in rows:
                if row["source_id"] in self.fake_sources: row["source_sha256"] = self.fake_sources[row["source_id"]]["sha256"]
            path = self.base / (split + ".jsonl"); path.write_text("\n".join(json.dumps(r) for r in rows))
            plan["source_roles"]["existing_" + split]["manifest_sha256"] = D.file_sha(path)
            self.c["files"][split + "_manifest"] = {"path": str(path), "sha256": D.file_sha(path)}
        p = self.base / "plan.json"; write(p, plan)
        self.c["files"]["plan"] = {"path": str(p), "sha256": D.file_sha(p)}
        prior = D._json(SOURCE_RECEIPT); prior["source"]["source_sha256"] = self.fake_sources["2BdYaGuVAKo"]["sha256"]
        p = self.base / "prior.json"; write(p, prior)
        self.c["files"]["prior_single_source_receipt"] = {"path": str(p), "sha256": D.file_sha(p)}
        for key, file in (("numeric_source", "materialize_mixed_temporal_pilot.py"), ("loader_source", "history_utility_data.py"), ("entry_source", "materialize_history_utility_pilot.py")):
            p = ROOT / "code" / file; self.c["files"][key] = {"path": str(p), "sha256": D.file_sha(p)}

    def tearDown(self): self.temp.cleanup()

    def test_contract_pins_roles_scope_and_old_constants(self):
        _, profiles, registry = M.validate_contract(self.c)
        self.assertEqual([p["profile_id"] for p in profiles], list(D.PROFILES))
        self.assertEqual({s: registry[s]["split"] for s in self.fake_sources}, {s: split for split, ids in M.SELECTED.items() for s in ids})
        old = M.load_numeric(self.c["files"]["numeric_source"]["path"])
        self.assertEqual(old.SOURCE, "2BdYaGuVAKo"); self.assertEqual(len(old.PROFILE_IDS), 6)
        for mutation in (lambda c: c["budget"].update(wall_seconds=1801),
                         lambda c: c.update(scientific_continuity_approved=True),
                         lambda c: c["sources"]["train"].append("unapproved"),
                         lambda c: c["profiles"].append("sr_smooth_blur_noise_h264")):
            bad = copy.deepcopy(self.c); mutation(bad)
            with self.assertRaises(ValueError): M.validate_contract(bad)

    def test_default_inspection_zero_decode_and_changed_source_rejected(self):
        old = M.load_numeric(self.c["files"]["numeric_source"]["path"])
        old.source_probe = mock.Mock(return_value={"streams": [{"width": 832, "height": 480}], "format": {"duration": "8"}})
        with mock.patch.object(M, "load_numeric", return_value=old):
            report = M.inspect(self.c)
        self.assertEqual(old.source_probe.call_count, 4)
        self.assertEqual(report["status"], "inspected_not_materialized")
        self.assertEqual(report["source_file_SHA_verified"], 4)
        Path(self.fake_sources["4EKWTiWjPTQ"]["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "actual original"):
            M.validate_contract(self.c)

    def test_decode_reuses_exact_command_and_inherits_owned_group(self):
        old = M.load_numeric(self.c["files"]["numeric_source"]["path"])
        binding = {"source_id": "fixture", "source_sha256": "a" * 64, "source_group_id": "fixture", "split": "train"}
        spec = old.target_specification(binding, {"format": {"duration": "8", "start_time": "0"}}, self.c)
        source = self.base / "fake.mp4"; budget = mock.Mock()
        actual = []
        class Process:
            pid = 777; returncode = 0
            def __init__(inner, command, **kwargs):
                actual.append((command, kwargs["start_new_session"]))
                raw = Path(command[-1]); raw.touch()
                with raw.open("r+b") as f: f.truncate(np.prod(D.SHAPE).item())
                lines = ["[showinfo@source x] config in time_base: 1/24", "[showinfo@target x] config in time_base: 1/24"]
                for i in range(96):
                    for role in ("source", "target"):
                        lines.append(f"[showinfo@{role} x] n: {i} pts: {i} pts_time:{i/24}")
                kwargs["stderr"].write("\n".join(lines).encode()); kwargs["stderr"].flush()
            def poll(inner): return 0
        with mock.patch.object(M.subprocess, "Popen", Process):
            pixels, record = M.decode_target_owned(old, source, spec, self.base, "ffmpeg", budget)
        expected, _ = old.target_command(source, spec, self.base / "target.rgb", "ffmpeg")
        self.assertEqual(actual, [(expected, False)])
        self.assertEqual(record["decoded_target_count"], 96); self.assertEqual(pixels.shape, D.SHAPE)
        del pixels

    def test_new_source_receipt_replays_and_actual_loader_schema_compatibility(self):
        old = M.load_numeric(self.c["files"]["numeric_source"]["path"])
        plan, profiles, registry = M.validate_contract(self.c)
        sid = "4EKWTiWjPTQ"  # no old-target parity reference for this synthetic fixture
        clean = np.zeros(D.SHAPE, np.uint8)
        old.source_probe = lambda *x: {"format": {"duration": "8", "start_time": "0"}}
        old.software_receipt = lambda *x: {"fixture": "synthetic CPU; no real codec"}
        original = D._json(SOURCE_RECEIPT)
        calls = []
        def decode(*args):
            scratch = args[3]; (scratch / "target_decode.log").write_text("synthetic")
            return clean, {"decoded_target_count": 96, "decoded_source_count": 96}
        def observation(clean, profile, binding, window, scratch, replay, budget):
            calls.append((profile["profile_id"], replay))
            row = copy.deepcopy(original["observations"][profile["profile_id"]]["replays"][0])
            row["observation_pixel_sha256"] = D.array_sha(clean)
            for i, noise in enumerate(row["noise_frames"]):
                noise["seed_uint64"] = old.noise_seed(binding["source_sha256"], window, profile["noise_stream"], i)
                noise["unit_noise_pixel_sha256"] = "a" * 64
            row["noise_schedule_sha256"] = D.value_sha(row["noise_frames"])
            raw = scratch / (str(replay) + ".raw"); raw.touch()
            return clean, raw, row
        old.observation = observation
        with mock.patch.object(M, "decode_target_owned", side_effect=decode):
            entry = M.source_receipt(self.c, old, plan, profiles, registry, registry[sid], self.base / "materialized", mock.Mock())
        self.assertEqual(calls, [(p, r) for p in D.PROFILES for r in (0, 1)])
        receipt = D._json(entry["path"])
        self.assertEqual(receipt["producer_schema"], M.SCHEMA)
        self.assertFalse(receipt["scientific_continuity_approved"])
        data = D.HistoryUtilityDataset([entry], plan=self.c["files"]["plan"]["path"],
            train_manifest=self.c["files"]["train_manifest"]["path"], dev_manifest=self.c["files"]["dev_manifest"]["path"],
            split="train", cache_bytes=128 << 20)
        self.assertFalse(data.research_ready)
        row = data._row(sid, D.PROFILES[0], 1, [192, 368, 96, 96], [1, 1], None, 0)
        np.testing.assert_array_equal(data.batch([row])["model_inputs"]["availability"], [[0, 1]])

    def test_failure_preserved_and_existing_output_never_overwritten(self):
        out = self.base / "failed_run"
        with mock.patch.object(M, "inspect", return_value={}), mock.patch.object(M, "source_receipt", side_effect=MemoryError("fixture768limit")):
            with self.assertRaises(MemoryError): M.execute(self.c, out)
        report = D._json(out / "failure.json")
        self.assertFalse(report["complete"]); self.assertEqual(report["sources_completed"], [])
        self.assertFalse((out / "complete.json").exists())
        old = D.file_sha(out / "failure.json")
        with self.assertRaises(FileExistsError): M.execute(self.c, out)
        self.assertEqual(D.file_sha(out / "failure.json"), old)

    def test_forbidden_import_rejected_without_loading_model(self):
        guard = M.NoModels()
        for name in ("torch", "torch_npu", "h3_comfy", "rphi_model"):
            with self.assertRaises(ImportError): guard.find_spec(name)
        self.assertEqual(len(guard.attempts), 4)
        self.assertEqual(M.check_no_models()["forbidden_loaded_modules"], [])

    def supervisor(self):
        spec = importlib.util.spec_from_file_location("test_history_supervisor", ROOT / "preparation/history_utility_materializer_20261112a/run_once.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_supervisor_terminates_only_owned_group_with_bounded_grace(self):
        supervisor = self.supervisor(); process = mock.Mock(pid=91234)
        with mock.patch.object(supervisor.os, "killpg") as kill, \
             mock.patch.object(supervisor, "group_memory", return_value=[{"pid": 91234, "rss_bytes": 1}]), \
             mock.patch.object(supervisor.time, "monotonic", side_effect=[0, 31]):
            supervisor.stop_owned(process, grace=30)
        self.assertEqual(kill.call_args_list, [mock.call(91234, supervisor.signal.SIGTERM), mock.call(91234, supervisor.signal.SIGKILL)])
        process.wait.assert_called_once_with(timeout=5)

    def test_supervisor_freeze_rejects_changed_limits_and_unpinned_payload(self):
        supervisor = self.supervisor(); run = self.base / "runs" / "worker"
        config = {"schema": "openh3.history_utility_cpu_freeze.v1", "host": "31984", "wall_seconds": 1800,
            "grace_seconds": 30, "memory_mib": 768, "pins": {str(Path(supervisor.__file__).resolve()): supervisor.sha(supervisor.__file__),
                str(self.base / "entry.py"): "0" * 64}, "root": str(self.base), "run": str(run),
            "supervisor": str(self.base / "runs" / "supervisor"), "command": ["python", "entry.py", "--out", str(run)],
            "execution_contract": "missing.json"}
        p = self.base / "freeze.json"; write(p, config)
        with self.assertRaises(FileNotFoundError): supervisor.validate_freeze(p, supervisor.sha(p))
        config["memory_mib"] = 769; write(p, config)
        with self.assertRaisesRegex(ValueError, "CPU limits changed"): supervisor.validate_freeze(p, supervisor.sha(p))
        with self.assertRaisesRegex(ValueError, "freeze SHA differs"): supervisor.validate_freeze(p, "0" * 64)


if __name__ == "__main__": unittest.main()
