"""Portable VAE-only admission for the unchanged fixed TRAIN cache exporter.

Only manifest parsing/admission and truthful component-scope reporting differ
from the frozen exporter. All input checks, P preprocessing, VAE loading,
2-encode/1-decode math, precision, padding, guards and output verification reuse
the original implementation. This does not admit a partially verified DiT run.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import export_train_vae_cache as legacy
from openh3_vae_only_assets import read_vae_only_manifest, SCOPE

LEGACY_ENTRY_SHA = "5e2e18f3325ed2473bba088ce4fb1d846b0fa356319a1ff64e0c60958f7714a1"
EXECUTION_SCHEMA = "openh3.train_vae_cache_portable_execution.v1"


def check_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("wrong portable execution schema")
    if contract.get("entry_sha256") != legacy.sha_file(__file__):
        raise ValueError("portable entrypoint not pinned")
    if (contract.get("legacy_entry_sha256") != LEGACY_ENTRY_SHA or
            legacy.sha_file(legacy.__file__) != LEGACY_ENTRY_SHA):
        raise ValueError("frozen numerical exporter differs")
    if contract.get("asset_verification_scope") != SCOPE:
        raise ValueError("explicit used-components-only scope required")
    if contract.get("selected_sample_id") != legacy.SAMPLE_ID:
        raise ValueError("fixed sample identity differs")
    root = Path(contract["root"]).resolve()
    verified = legacy.verify_pins(root,contract["implementation_sha256"])
    for module_path in (Path(__file__).resolve(),Path(legacy.__file__).resolve()):
        if verified.get(str(module_path)) != legacy.sha_file(module_path):
            raise ValueError("portable/legacy entry missing from actual runtime pins")
    import openh3_vae_only_assets
    helper = Path(openh3_vae_only_assets.__file__).resolve()
    if verified.get(str(helper)) != legacy.sha_file(helper):
        raise ValueError("used-components manifest reader missing from runtime pins")
    paths = {}
    for key, expected in legacy.FIXED_SHA.items():
        paths[key] = legacy.checked_path(root,contract[key])
        field = "manifest_sha256" if key == "pretrained_manifest" else key + "_sha256"
        if contract.get(field) != expected or legacy.sha_file(paths[key]) != expected:
            raise ValueError("fixed input SHA mismatch: " + key)
    comfy = legacy.checked_path(root,contract["comfy_root"])
    for relative, expected in legacy.PREPROCESS_SOURCE_SHA.items():
        file = (comfy/relative).resolve()
        if legacy.sha_file(file) != expected or verified.get(str(file)) != expected:
            raise ValueError("official preprocessing source not pinned: " + relative)
    return contract,paths,comfy,verified


class Evidence(legacy.Evidence):
    """Correct the old all-assets-size wording before any partial/final write.

    This only changes provenance text. It never replaces the original math,
    loader, source bindings, tensors, controls or acceptance gates.
    """
    def save(self):
        self.data["portable_component_scope"] = self.data["inspection"]["component_verification"]
        self.data["legacy_numeric_entry_sha256"] = LEGACY_ENTRY_SHA
        asset = self.data.get("asset_verification")
        if asset is not None:
            asset["other_recipe_assets"] = "metadata_only; not resolved, stat-ed, hashed or loaded"
            asset["verification_scope"] = SCOPE
            asset["full_recipe_assets_verified"] = False
            asset["checked_roles"] = ["vae","context"]
        super().save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-contract",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError(args.out)
    contract,paths,comfy,verified = check_contract(args.execution_contract)
    # No call to legacy main or generic read_manifest: those stat unused assets.
    pretrained = read_vae_only_manifest(paths["pretrained_manifest"],
        expected_manifest_sha256=legacy.FIXED_SHA["pretrained_manifest"])
    proof,raw,rphi = legacy.inspect_inputs(paths,comfy)
    proof["pretrained_fingerprint"] = pretrained["_fingerprint"]
    proof["asset_inspection"] = SCOPE
    proof["component_verification"] = pretrained["_verification_scope"]
    proof["parent_manifest_sha256"] = pretrained["_manifest_sha256"]
    proof["all_recipe_asset_files_verified"] = False
    proof["verified_runtime_files"] = len(verified)
    proof["execution_contract_sha256"] = legacy.sha_file(args.execution_contract)
    proof["legacy_numeric_entry_sha256"] = LEGACY_ENTRY_SHA
    if not args.execute:
        args.out.mkdir(parents=True,exist_ok=False)
        legacy.atomic_json(args.out/"inspection.json",proof)
        print(json.dumps({"inspection_complete":True,"sample_id":legacy.SAMPLE_ID,"model_calls":0,
            "asset_verification_scope":SCOPE,"checked_roles":["vae","context"],
            "postP_sha256":{k:v["after_P_float32_sha256"] for k,v in proof["preprocessing"].items()}}),flush=True)
        return
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES","").isdecimal():
        raise ValueError("explicit externally admitted single physical NPU required")
    evidence = Evidence(args.out,{**contract,"execution_contract_sha256":legacy.sha_file(args.execution_contract)},proof)
    try:
        # Frozen loader verifies ONLY VAE/context again, then H3(load_dit=False).
        # Its original FP16 encode/decode path and every existing gate are intact.
        legacy.execute(paths,pretrained,raw,rphi,evidence,verified)
    except BaseException as error:
        evidence.data["complete"] = False
        evidence.data["failure"] = {"type":type(error).__name__,"message":str(error),
            "traceback":traceback.format_exc(),"phase":evidence.data.get("failure_origin_phase",evidence.data["phase"])}
        evidence.save()
        legacy.atomic_json(evidence.path/"failure.json",evidence.data)
        raise


if __name__ == "__main__": main()
