"""CPU/std-library tests using the small immutable completed evidence, no scoring."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("audit_cached", ROOT / "code/audit_cached_rgb_compensation.py")
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


class AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r = ROOT / "results/takeover_20261111"
        paths = {
            "metrics": r / "cached_rgb_compensation_completed_20261111/metrics.json",
            "ordinary_metrics": r / "fixed_h3_legacy_smoke_completed/runs/fixed_h3_rphi_legacy_smoke_20261111a/eval/metrics.json",
            "controls_metrics": r / "fixed_h3_controls_completed/runs/fixed_h3_rphi_legacy_controls_smoke_20261111a/eval/metrics.json",
            "vae_metrics": r / "vae_roundtrip_v2_completed/runs/vae_roundtrip_legacy_smoke_20261111b/eval/metrics.json",
        }
        if not all(p.exists() for p in paths.values()):
            raise unittest.SkipTest("immutable completed small evidence not available")
        cls.docs = {k:json.loads(p.read_text()) for k,p in paths.items()}
        cls.hashes = {k:audit_module.sha_file(p) for k,p in paths.items() if k != "metrics"}

    def run_audit(self, report=None, hashes=None):
        d = self.docs
        return audit_module.audit(report or d["metrics"], d["ordinary_metrics"], d["controls_metrics"], d["vae_metrics"], hashes or self.hashes)

    def modified(self, fn):
        report = copy.deepcopy(self.docs["metrics"])
        fn(report)
        # Mutations pass enclosing receipts so the specific semantic gate is tested.
        report["contract_sha256"] = audit_module.canonical_sha(report["contract"])
        report["sample_receipts"] = [audit_module.canonical_sha(r) for r in report["per_sample"]]
        return report

    def test_complete_exact_evidence_passes_without_selecting(self):
        result = self.run_audit()
        self.assertEqual(result["grid_records"], 168)
        self.assertEqual(result["quality_verdict"], "requires_review")
        self.assertIsNone(result["selected_beta"])

    def test_missing_lpips_rejected(self):
        r = self.modified(lambda x:x["per_sample"][0]["grid"][1]["post_final_clip_float32"]["quality"].update(lpips=None))
        with self.assertRaisesRegex(ValueError, "lpips"):
            self.run_audit(r)

    def test_flow_valid_pair_loss_rejected(self):
        r = self.modified(lambda x:x["per_sample"][0]["grid"][1]["post_final_clip_float32"]["temporal"].update(valid_pairs=20))
        with self.assertRaisesRegex(ValueError, "21 common-flow"):
            self.run_audit(r)

    def test_flow_coverage_mismatch_rejected(self):
        r = self.modified(lambda x:x["per_sample"][0]["grid"][1]["post_final_clip_float32"]["temporal"].update(valid_coverage=.5))
        with self.assertRaisesRegex(ValueError, "valid_coverage"):
            self.run_audit(r)

    def test_beta0_pixel_endpoint_rejected(self):
        r = self.modified(lambda x:x["per_sample"][0]["grid"][0]["clipping_and_change"].update(output_array_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "beta0 pixel"):
            self.run_audit(r)

    def test_missing_grid_record_rejected(self):
        r = self.modified(lambda x:x["per_sample"][0]["grid"].pop())
        with self.assertRaisesRegex(ValueError, "grid missing"):
            self.run_audit(r)

    def test_missing_source_record_rejected(self):
        r = self.modified(lambda x:x["per_sample"].pop())
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_audit(r)

    def test_prior_file_hash_rejected(self):
        hashes = dict(self.hashes, ordinary_metrics="0"*64)
        with self.assertRaisesRegex(ValueError, "reference file hash"):
            self.run_audit(hashes=hashes)

    def test_scalar_convex_clip_rejected(self):
        r = self.modified(lambda x:x["per_sample"][0]["grid"][1]["clipping_and_change"].update(clipped_component_fraction=.01,clipped_pixel_fraction=.01))
        with self.assertRaisesRegex(ValueError, "convex scalar"):
            self.run_audit(r)


if __name__ == "__main__":
    unittest.main()
