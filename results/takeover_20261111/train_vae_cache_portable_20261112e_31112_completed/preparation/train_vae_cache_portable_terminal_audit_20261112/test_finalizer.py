import json
from pathlib import Path
import tempfile
import unittest

from finalize_once import final_state, wait_terminal


class FinalizerTests(unittest.TestCase):
    def test_bounded_wait_and_all_three_terminal_states(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            now, sleeps = [0.0], []
            def sleep(seconds):
                sleeps.append(seconds); now[0] += seconds
            observations = []
            result = wait_terminal(path, lambda *args: observations.append(args),
                max_wait=65, poll=30, clock=lambda: now[0], sleep=sleep)
            self.assertFalse(result["terminal"]); self.assertEqual(sleeps, [30, 30, 5])
            self.assertEqual(now[0], 65); self.assertEqual(len(observations), 4)
            for state in ("completed", "failed", "not_launched"):
                path.write_text(json.dumps({"state": state}))
                def no_sleep(seconds):
                    self.fail("terminal state must not wait")
                result = wait_terminal(path, lambda *args: None, sleep=no_sleep)
                self.assertTrue(result["terminal"]); self.assertEqual(result["record"]["state"], state)

    def test_producer_failure_or_audit_failure_never_becomes_complete(self):
        good = {"complete": True, "state": "completed"}
        self.assertTrue(final_state("completed", 0, good)["complete"])
        for state, rc, row in (("failed", 0, good), ("not_launched", 0, good), ("completed", 2, good),
                                ("completed", 0, {"complete": False, "state": "failed"})):
            result = final_state(state, rc, row)
            self.assertFalse(result["complete"]); self.assertEqual(result["state"], "failed")


if __name__ == "__main__":
    unittest.main()
