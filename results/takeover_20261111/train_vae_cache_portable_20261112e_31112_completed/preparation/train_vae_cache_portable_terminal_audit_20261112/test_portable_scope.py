import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_arrays import file_sha
import audit_terminal as audit


def fixture(root):
    root = root.resolve()
    records = {}
    for name in ("base", "vae", "context", "refiner", "turbo"):
        path = root / (name + ".used" if name in ("vae", "context") else "unused_" + name)
        value = (name + "-fixture").encode()
        if name in ("vae", "context"):
            path.write_bytes(value)
        records[name] = {"path": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    parent = {"schema_version": 1, "layout": "comfyui_minimax_h3", "recipe": "refiner-turbo-comfy",
              **{name: records[name] for name in ("base", "vae", "context")},
              "components": [{"role": name, **records[name]} for name in ("refiner", "turbo")]}
    manifest = root / "parent.json"; manifest.write_text(json.dumps(parent))
    digest = file_sha(manifest)
    identity = {"recipe": parent["recipe"], "layout": parent["layout"],
                "assets": {name: row["sha256"] for name, row in records.items()},
                "merge": "sum_fp32_then_cast_once", "sigma": [0.909375, 6 / 7, 2 / 3, 0.0]}
    scope = {"scope": audit.ASSET_SCOPE,
             "checked": {name: {**records[name], "actual_file_checked": True} for name in ("vae", "context")},
             "unchecked_roles": ["base", "refiner", "turbo"],
             "unchecked_policy": "metadata only; not resolved, stat-ed, hashed or loaded",
             "parent_recipe_fingerprint_role": "metadata provenance only, not full recipe asset verification",
             "safe_for_DiT_pretrained_merge": False}
    report = {"portable_component_scope": scope, "legacy_numeric_entry_sha256": audit.LEGACY_PRODUCER_SHA,
              "inspection": {"component_verification": scope, "asset_inspection": audit.ASSET_SCOPE,
                  "all_recipe_asset_files_verified": False, "parent_manifest_sha256": digest,
                  "pretrained_fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                  "legacy_numeric_entry_sha256": audit.LEGACY_PRODUCER_SHA},
              "asset_verification": {"VAE_and_context_actual_SHA": True, "DiT_loaded": False,
                  "Rphi_model_constructed": False, "Rphi_checkpoint_deserialized_on_CPU": True,
                  "actual_vae_dtype": "torch.float16", "verification_scope": audit.ASSET_SCOPE,
                  "full_recipe_assets_verified": False, "checked_roles": ["vae", "context"],
                  "other_recipe_assets": "metadata_only; not resolved, stat-ed, hashed or loaded"}}
    contract = {"root": str(root), "pretrained_manifest": "parent.json", "manifest_sha256": digest,
                "asset_verification_scope": audit.ASSET_SCOPE, "legacy_entry_sha256": audit.LEGACY_PRODUCER_SHA}
    return report, contract, digest


class PortableScopeTests(unittest.TestCase):
    def test_absent_unused_assets_are_never_resolved_statted_or_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            report, contract, digest = fixture(Path(directory))
            original_resolve, original_stat = Path.resolve, Path.stat
            def resolve(path, *args, **kwargs):
                self.assertFalse(path.name.startswith("unused_"), "unused asset was resolved")
                return original_resolve(path, *args, **kwargs)
            def stat(path, *args, **kwargs):
                self.assertFalse(path.name.startswith("unused_"), "unused asset was stat-ed")
                return original_stat(path, *args, **kwargs)
            with patch.object(audit, "PARENT_MANIFEST_SHA", digest), patch.object(Path, "resolve", resolve), patch.object(Path, "stat", stat):
                result = audit.portable_assets(report, contract)
            self.assertEqual(set(result["checked"]), {"vae", "context"})
            self.assertFalse(result["full_recipe_assets_verified"])
            self.assertFalse(result["unused_paths_resolved_by_auditor"])

    def test_used_asset_tamper_or_parent_manifest_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report, contract, digest = fixture(root)
            with patch.object(audit, "PARENT_MANIFEST_SHA", digest):
                (root / "vae.used").write_bytes(b"X" * (root / "vae.used").stat().st_size)
                with self.assertRaisesRegex(ValueError, "actual SHA"):
                    audit.portable_assets(report, contract)
                (root / "parent.json").write_text((root / "parent.json").read_text() + " ")
                with self.assertRaisesRegex(ValueError, "parent manifest"):
                    audit.portable_assets(report, contract)

    def test_full_recipe_claim_changed_fingerprint_or_legacy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report, contract, digest = fixture(Path(directory))
            mutations = [lambda r: r["asset_verification"].update(full_recipe_assets_verified=True),
                         lambda r: r["inspection"].update(pretrained_fingerprint="0" * 64),
                         lambda r: r.update(legacy_numeric_entry_sha256="0" * 64),
                         lambda r: r["portable_component_scope"]["checked"].update(base={})]
            with patch.object(audit, "PARENT_MANIFEST_SHA", digest):
                for mutate in mutations:
                    bad = copy.deepcopy(report); mutate(bad)
                    with self.assertRaises(ValueError):
                        audit.portable_assets(bad, contract)

    def test_admission_requires_all_three_producer_sources_and_rejects_changed_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); own = root / "preparation/audit"; own.mkdir(parents=True)
            prep = root / "preparation/producer"; (prep / "code").mkdir(parents=True)
            for name in audit.SOURCE_FILES:
                (own / name).write_text("# auditor fixture\n")
            hashes = {}
            for name in ("export_train_vae_cache_portable.py", "export_train_vae_cache.py", "openh3_vae_only_assets.py"):
                path = prep / "code" / name; path.write_text("# " + name + "\n"); hashes[name] = file_sha(path)
            for name in ("execution_contract.json", "queue.json"):
                (prep / name).write_text("{}")
            config = {"schema": "openh3.train_vae_cache_portable_cpu_audit_admission.v1",
                "run": "runs/producer", "prep": "preparation/producer", "out": "runs/audit",
                "producer_output_subdir": "cache", "producer_sha256": hashes["export_train_vae_cache_portable.py"],
                "legacy_producer_sha256": hashes["export_train_vae_cache.py"],
                "asset_helper_sha256": hashes["openh3_vae_only_assets.py"],
                "execution_contract_sha256": file_sha(prep / "execution_contract.json"),
                "queue_sha256": file_sha(prep / "queue.json"),
                "source_sha256": {name: file_sha(own / name) for name in audit.SOURCE_FILES}}
            path = own / "admission.json"; path.write_text(json.dumps(config))
            with patch.object(audit, "LEGACY_PRODUCER_SHA", hashes["export_train_vae_cache.py"]):
                audit.admission(root, own, path, file_sha(path))
                (prep / "code/openh3_vae_only_assets.py").write_text("# changed helper\n")
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    audit.admission(root, own, path, file_sha(path))


if __name__ == "__main__":
    unittest.main()
