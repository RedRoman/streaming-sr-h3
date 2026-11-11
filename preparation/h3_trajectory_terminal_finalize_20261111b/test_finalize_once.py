"""Bounded-wait/allowlist/archive CPU tests; no real queue or accelerator access."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import finalize_once as finalizer


class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.queue = self.root / "queue.json"

    def admission(self):
        own = self.root / "preparation/own"
        own.mkdir(parents=True, exist_ok=True)
        path = own / "admission.json"
        config = {"schema": "openh3.latent_finalizer_admission.v1", "run": "runs/probe_b",
                  "prep": "preparation/probe_b", "out": "runs/probe_b_finalize",
                  **{key: "a" * 64 for key in ("auditor_sha256", "finalizer_sha256", "execution_contract_sha256",
                     "queue_sha256", "probe_sha256", "trajectory_sha256")}}
        path.write_text(json.dumps(config))
        return own, path, config

    def test_explicit_fresh_run_config_and_sha_required(self):
        own, path, config = self.admission()
        result, paths = finalizer.load_admission(self.root, own, path, finalizer.sha_file(path))
        self.assertEqual(result, config)
        self.assertEqual(paths["run"], self.root / "runs/probe_b")
        with self.assertRaisesRegex(ValueError, "SHA"):
            finalizer.load_admission(self.root, own, path, "0" * 64)

    def test_admission_overlap_escape_unknown_field_refused(self):
        own, path, config = self.admission()
        for key, value in (("out", config["run"]), ("out", "runs/../../escape"),
                           ("prep", "preparation/own"), ("unexpected", True),
                           ("probe_sha256", "not a digest")):
            modified = {**config, key: value}
            path.write_text(json.dumps(modified))
            with self.assertRaises(ValueError):
                finalizer.load_admission(self.root, own, path, finalizer.sha_file(path))

    def test_verify_pins_rejects_wrong_contract_and_source(self):
        own, path, config = self.admission()
        prep = self.root / config["prep"]
        prep.mkdir()
        (own / "audit_terminal.py").write_text("pass")
        (prep / "queue.json").write_text("{}")
        contract = {"root": str(self.root), "probe_sha256": config["probe_sha256"],
                    "implementation_sha256": {"trajectory": config["trajectory_sha256"]}}
        (prep / "execution_contract.json").write_text(json.dumps(contract))
        config.update(finalizer_sha256=finalizer.sha_file(finalizer.__file__),
                      auditor_sha256=finalizer.sha_file(own / "audit_terminal.py"),
                      queue_sha256=finalizer.sha_file(prep / "queue.json"),
                      execution_contract_sha256=finalizer.sha_file(prep / "execution_contract.json"))
        paths = {"prep": prep}
        finalizer.verify_runtime_pins(config, paths, own)
        config["probe_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "pin"):
            finalizer.verify_runtime_pins(config, paths, own)
        config["probe_sha256"] = contract["probe_sha256"]
        (prep / "queue.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "SHA"):
            finalizer.verify_runtime_pins(config, paths, own)

    def test_wait_terminal_uses_fixed_bounded_reads_without_audit(self):
        elapsed, notifications = [0.], []
        self.queue.write_text('{"state":"running"}')
        def sleep(seconds):
            elapsed[0] += seconds
            if elapsed[0] == 60:
                self.queue.write_text('{"state":"completed"}')
        result = finalizer.wait_terminal(self.queue, lambda *v: notifications.append(v),
                                        max_wait=90, poll=30, clock=lambda: elapsed[0], sleep=sleep)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["seconds"], 60)
        self.assertEqual(len(notifications), 3)

    def test_wait_timeout_does_not_claim_terminal_or_edit_queue(self):
        elapsed = [0.]
        self.queue.write_text('{"state":"running"}')
        before = self.queue.read_bytes()
        result = finalizer.wait_terminal(self.queue, lambda *v: None, max_wait=65, poll=30,
            clock=lambda: elapsed[0], sleep=lambda s: elapsed.__setitem__(0, elapsed[0] + s))
        self.assertFalse(result["terminal"])
        self.assertEqual(result["seconds"], 65)
        self.assertEqual(self.queue.read_bytes(), before)

    def test_failed_queue_is_a_terminal_observation(self):
        self.queue.write_text('{"state":"failed","reason":"test"}')
        result = finalizer.wait_terminal(self.queue, lambda *v: None, sleep=lambda s: self.fail("must not sleep"))
        self.assertTrue(result["terminal"])
        self.assertEqual(result["record"]["state"], "failed")

    def test_allowlist_excludes_gradient_npy_and_pack_manifest_exact(self):
        run, prep, own, out = [self.root / x for x in ("run", "prep", "own", "out")]
        for p in (run / "eval/adapter_gradients", run / "logs", prep / "code", own, out):
            p.mkdir(parents=True)
        for path in (run / "queue_status.json", run / "eval/report.json", run / "logs/probe.log",
                     prep / "execution_contract.json", prep / "code/probe.py", own / "finalize_once.py",
                     out / "finalize_receipt.json", out / "terminal_audit.json"):
            path.write_text('small evidence\n')
        (run / "eval/adapter_gradients/huge.npy").write_bytes(b"not copied")
        (run / "eval/latent_outputs.npz").write_bytes(b"small latent cache")
        files = finalizer.selected_files(self.root, run, prep, own, out, True)
        self.assertFalse(any(p.suffix == ".npy" for p in files))
        archive = out / "evidence.tar.gz"
        result = finalizer.archive_small(self.root, files, archive)
        with tarfile.open(archive) as packed:
            manifest = json.load(packed.extractfile("MANIFEST.json"))
            for name, row in manifest["files"].items():
                data = packed.extractfile(name).read()
                self.assertEqual(finalizer.sha_bytes(data), row["sha256"])
                self.assertEqual(len(data), row["bytes"])
            self.assertFalse(any(name.endswith(".npy") for name in packed.getnames()))
        self.assertEqual(result["sha256"], finalizer.sha_file(archive))
        with self.assertRaises(FileExistsError):
            finalizer.archive_small(self.root, files, archive)

    def test_archive_bound_and_fresh_output_refusal(self):
        source, target = self.root / "source.json", self.root / "evidence.tar.gz"
        source.write_bytes(b"X" * 1000)
        with self.assertRaisesRegex(ValueError, "limit"):
            finalizer.archive_small(self.root, [source], target, max_bytes=100)
        self.assertFalse(target.exists())
        output = self.root / "already_started"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            output.mkdir(parents=True, exist_ok=False)


if __name__ == "__main__":
    unittest.main()
