"""Stdlib-only launcher admission tests with fake resource/queue operations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("vae_j_launcher", Path(__file__).with_name("run_once.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class AdmissionTests(unittest.TestCase):
    def run_case(self, cards, valid=True):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {"root": str(root), "cards": [7], "min_free_mb": 60000,
                "lock_dir": str(root / "shared_locks"), "status": "run/queue_status.json",
                "fresh_admission": "admission.json", "steps": [{"argv": [
                    "timeout", "--signal=TERM", "--kill-after=30s", "600s", "venv/bin/python", "frozen_probe.py"]}]}
            plan_path = root / "queue.json"
            plan_path.write_text(json.dumps(plan))
            admission = {"schema": "openh3.vae_latent_descent_admission.v1",
                "approved_for_single_card_vae_latent_descent": True, "physical_card": 7, "seed": 102,
                "queue_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest() if valid else "wrong"}
            (root / "admission.json").write_text(json.dumps(admission))
            calls, saves = [], []
            queue = SimpleNamespace(inspect=lambda: cards.pop(0), available=lambda row, minimum: row["idle"],
                save=lambda *args, **kwargs: saves.append(kwargs),
                run_plan=lambda plan, status, card: calls.append(card) or 0)
            with patch.dict(sys.modules, {"takeover_queue": queue}), patch.object(sys, "argv", ["run_once", str(plan_path)]), patch.object(launcher.time, "sleep") as sleep:
                if valid:
                    code = launcher.main()
                else:
                    with self.assertRaisesRegex(ValueError, "fresh root admission"):
                        launcher.main()
                    code = None
            return code, calls, saves, sleep.call_args_list

    def test_busy7_does_not_choose_idle6_or_wait(self):
        code, calls, saved, sleeps = self.run_case([{6: {"idle": True}, 7: {"idle": False}}])
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])
        self.assertEqual(sleeps, [])
        self.assertEqual(saved[0]["state"], "not_launched")

    def test_second_check_failure_does_not_retry(self):
        code, calls, saved, sleeps = self.run_case([{7: {"idle": True}}, {7: {"idle": False}}])
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])
        self.assertEqual(sleeps[0].args, (15,))

    def test_one_success_only_uses7(self):
        code, calls, saved, sleeps = self.run_case([{7: {"idle": True}}, {7: {"idle": True}}])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [7])
        self.assertEqual(len(sleeps), 1)

    def test_wrong_queue_hash_stops_before_resource_read(self):
        code, calls, saved, sleeps = self.run_case([], valid=False)
        self.assertIsNone(code)
        self.assertEqual(calls + saved + sleeps, [])


if __name__ == "__main__":
    unittest.main()
