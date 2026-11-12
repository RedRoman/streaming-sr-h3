"""Linux stdlib fake-process guard tests. No model/NPU imports or invocations."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location("full22_runner", Path(__file__).with_name("run_once.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


@unittest.skipUnless(sys.platform == "linux", "real owned process-group /proc tests require Linux")
class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def worker(self, code, *, wall=3, rss=256 << 20):
        status = {"complete": False, "producer_started": False}
        return R.supervise([sys.executable, "-c", code], self.root, dict(os.environ), self.root/"worker.log",
            status, self.root/"status.json", wall=wall, rss_limit=rss, grace=.15, sample_seconds=.02)
    def assert_clean(self, result):
        self.assertEqual(result["remaining_owned_live_processes"], [])
        self.assertEqual(R.live_rows(result["owned_pgid"]), [])

    def test_normal_completion_measures_actual_group_rss(self):
        r = self.worker("import time; x=bytearray(8<<20);time.sleep(.08)")
        self.assertEqual(r["state"], "worker_completed"); self.assertEqual(r["returncode"], 0)
        self.assertGreater(r["peak_owned_group_RSS_bytes"], 8 << 20)
        self.assertGreaterEqual(r["RSS_samples"], 2); self.assert_clean(r)

    def test_timeout_terminates_descendant_but_not_foreign_process(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True)
        try:
            r = self.worker("import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)", wall=.3)
            self.assertEqual(r["error"]["type"], "TimeoutError"); self.assert_clean(r)
            self.assertIsNone(foreign.poll())
        finally:
            os.killpg(foreign.pid, signal.SIGTERM); foreign.wait(timeout=3)

    def test_owned_child_rss_triggers_cap(self):
        r = self.worker("import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;x=bytearray(64<<20);time.sleep(30)']);time.sleep(30)", rss=40 << 20)
        self.assertEqual(r["error"]["type"], "MemoryError")
        self.assertGreater(r["peak_owned_group_RSS_bytes"], 40 << 20); self.assert_clean(r)

    def test_parent_exit_cannot_orphan_live_child(self):
        r = self.worker("import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(.1)")
        self.assertEqual(r["state"], "failed"); self.assertIn("descendants", r["error"]["message"]); self.assert_clean(r)

    def test_external_term_records_failure_and_cleans_owned_group(self):
        runner = Path(R.__file__).resolve()
        code = "\n".join([
            "import importlib.util,sys,os", "from pathlib import Path",
            "s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)",
            "p=Path(sys.argv[2]);status={'complete':False}",
            "r.supervise([sys.executable,'-c','import subprocess,sys,time;subprocess.Popen([sys.executable,\"-c\",\"import time;time.sleep(30)\"]);time.sleep(30)'],p,dict(os.environ),p/'worker.log',status,p/'status.json',wall=10,rss_limit=256<<20,grace=.15,sample_seconds=.02)"])
        proc = subprocess.Popen([sys.executable, "-c", code, str(runner), str(self.root)], start_new_session=True)
        try:
            until = time.monotonic()+4
            while time.monotonic() < until:
                try:
                    row = json.loads((self.root/"status.json").read_text())
                    if row.get("state") == "running": break
                except (FileNotFoundError, json.JSONDecodeError): pass
                time.sleep(.02)
            else: self.fail("fake supervisor never started")
            time.sleep(.1); proc.send_signal(signal.SIGTERM); proc.wait(timeout=5)
            row = json.loads((self.root/"status.json").read_text())
            self.assertEqual(row["error"]["type"], "Interrupted"); self.assert_clean(row)
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()

    def test_sigterm_ignoring_child_is_killed_within_grace(self):
        r = self.worker("import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)']);time.sleep(30)", wall=.35)
        self.assertEqual(r["error"]["type"], "TimeoutError")
        self.assertTrue(r["owned_cleanup"]["KILL_sent"]); self.assert_clean(r)


class ContractTests(unittest.TestCase):
    def test_changed_resource_budget_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"queue.json"
            R.write(p, {"schema":"openh3.full22_bounded_queue.v1", "root":d,
                        "resource":dict(R.RESOURCE, wall_seconds=3601)})
            with self.assertRaisesRegex(ValueError, "fixed resource budget"): R.validate(p)
    def test_signal_handlers_are_restored_on_exception(self):
        before = {s:signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with R.signal_guard(): raise RuntimeError("fixture")
        self.assertEqual(before, {s:signal.getsignal(s) for s in before})


if __name__ == "__main__": unittest.main()
