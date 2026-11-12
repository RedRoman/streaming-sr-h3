import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recover_history_utility_materialization as R


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.original_entry = self.base / "old.py"; self.original_entry.write_text("# unchanged fixture\n")
        self.artifacts = []
        for i in range(4):
            p = self.base / (str(i) + ".npz"); p.write_bytes(bytes([i]) * 12)
            self.artifacts.append({"path": str(p), "file_sha256": R.sha(p)})
        self.original = {"profiles": ["p0", "p1", "p2"], "files": {"entry_source": self.bound(self.original_entry)}}
        self.original_path = self.base / "original.json"; R.write(self.original_path, self.original)
        receipt = {"source": {"source_id": "2BdYaGuVAKo", "split": "train"}, "target": self.artifacts[0],
            "observations": {pid: {"artifact": a} for pid,a in zip(self.original["profiles"], self.artifacts[1:])}}
        self.receipt_path = self.base / "receipt.json"; R.write(self.receipt_path, receipt)
        failure = {"complete": False, "sources_completed": ["2BdYaGuVAKo"], "receipt_indices": {"train": [self.bound(self.receipt_path)]}}
        self.failure_path = self.base / "failure.json"; R.write(self.failure_path, failure)
        self.contract = {"schema": R.SCHEMA, "remaining_sources": list(R.REMAINING), "reuse_source_id": "2BdYaGuVAKo",
            "scientific_continuity_approved": False, "budget": {"wall_seconds":1800,"grace_seconds":30,"memory_mib":768,
                "simultaneous_compute_workers":1,"CPU_threads":1,"model_calls":0,"NPU_calls":0},
            "files": {"entry": self.bound(Path(R.__file__).resolve()), "original_entry": self.bound(self.original_entry),
                "original_contract": self.bound(self.original_path), "original_failure": self.bound(self.failure_path),
                "reuse_receipt": self.bound(self.receipt_path)},
            "reuse_artifacts": [{"path": a["path"], "sha256": a["file_sha256"]} for a in self.artifacts]}
        self.path = self.base / "recovery.json"; R.write(self.path, self.contract)

    @staticmethod
    def bound(p): return {"path": str(p), "sha256": R.sha(p)}
    def tearDown(self): self.temp.cleanup()

    def test_exact_reuse_inventory_file_hash_and_source_membership(self):
        self.assertEqual(R.validate(self.contract), self.original)
        bad = copy.deepcopy(self.contract); bad["remaining_sources"].insert(0, "2BdYaGuVAKo")
        with self.assertRaisesRegex(ValueError, "cohort"): R.validate(bad)
        Path(self.artifacts[0]["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "NPZ file changed"): R.validate(self.contract)

    def test_first_source_cannot_be_computed(self):
        module = mock.Mock(); module.validate_contract.return_value = ({}, [], {})
        with mock.patch.object(R, "load_original", return_value=module):
            with self.assertRaisesRegex(ValueError, "first source cannot be rerun"):
                R.phase(self.contract, self.base, "source", "2BdYaGuVAKo")
        module.source_receipt.assert_not_called()

    def test_phase_subprocess_inherits_group_and_must_complete(self):
        R.write(self.base / "loader.json", {"complete": True, "phase": "loader"})
        with mock.patch.object(R.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            result = R.run_phase(self.path, self.base, "loader")
        self.assertTrue(result["complete"]); self.assertFalse(run.call_args.kwargs["start_new_session"])
        self.assertIn("--phase", run.call_args.args[0]); self.assertNotIn("--execute", run.call_args.args[0])
        with self.assertRaises(FileExistsError): R.run_phase(self.path, self.base, "loader")

    def results(self):
        reuse = {"entry": self.bound(self.receipt_path), "pid": 100, "complete": True}
        rows = [reuse]
        for i, sid in enumerate(R.REMAINING): rows.append({"entry": {"path": sid, "sha256": "a"*64}, "pid": 101+i, "complete": True})
        rows.append({"loader_validation": {"train": {}, "dev": {}}, "pid": 104, "complete": True})
        return rows

    def test_three_sources_only_separate_loader_counts_and_no_overwrite(self):
        out = self.base / "newrun"; prior_sha = R.sha(self.receipt_path)
        with mock.patch.object(R, "run_phase", side_effect=self.results()) as run:
            report = R.execute(self.path, self.contract, out)
        self.assertEqual([(c.args[2], c.args[3:] or ()) for c in run.call_args_list],
            [("verify_reuse", ()), ("source", (R.REMAINING[0],)), ("source", (R.REMAINING[1],)), ("source", (R.REMAINING[2],)), ("loader", ())])
        self.assertEqual((report["new_target_decodes"],report["new_observation_codec_passes"]),(3,18))
        self.assertEqual([len(report["receipt_indices"][s]) for s in ("train","dev")],[2,2])
        self.assertEqual(prior_sha,R.sha(self.receipt_path)); self.assertFalse((out/'partial.json').exists())
        with self.assertRaises(FileExistsError): R.execute(self.path, self.contract, out)

    def test_failure_stops_remaining_work_and_preserves_reused_source(self):
        out = self.base / "failure_run"
        with mock.patch.object(R, "run_phase", side_effect=[self.results()[0], MemoryError("bounded failure")]) as run:
            with self.assertRaises(MemoryError): R.execute(self.path, self.contract, out)
        self.assertEqual(run.call_count,2); report=R.read(out/'failure.json')
        self.assertFalse(report['complete']);self.assertEqual(report['sources_completed'],['2BdYaGuVAKo'])
        self.assertFalse((out/'complete.json').exists());self.assertTrue((out/'partial.json').is_file())


if __name__ == '__main__': unittest.main()
