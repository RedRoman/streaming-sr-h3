"""Freeze reviewed train-cache code and exact data bindings; never launches."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/cache/zihaohe/omni_openh3_20261111"
PREP = "preparation/train_vae_cache_20261112b_31112"
RUN = "runs/train_vae_cache_legacy_smoke_20261112b_31112"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


def main():
    prep = ROOT / PREP
    if (prep / "code").exists() or (prep / "execution_contract.json").exists():
        raise FileExistsError("runtime frozen or partial; do not overwrite")
    if sha(prep / "source_contract.json") != "9c2e2bdc64fffec33a2e08c90e36affc87edc4c14ea3bb79467166da277ffdc4":
        raise ValueError("cache source contract changed")
    source = json.loads((prep / "source_contract.json").read_text())
    data = source["available_train_assets"]
    prior = ROOT / "preparation/h3_joint_rgb_probe_20261112a"
    pins = dict(json.loads((prior / "queue.json").read_text())["implementation_sha256"])
    for old_path in list(pins):
        if old_path.startswith("/cache/yunfeng/envs/minimax-h3-npu/lib/python3.12/site-packages/torch/"):
            new_path = old_path.replace("/cache/yunfeng/envs/", "/cache/zihaohe/envs/")
            pins[new_path] = pins.pop(old_path)
    names = {p.name for p in (prior / "code").glob("*.py")}
    names.update({"export_train_vae_cache.py", "export_rphi_train_guides.py"})
    tests = {"test_train_vae_cache.py"}
    for path in [ROOT / "code" / n for n in names] + [ROOT / "code/tests" / n for n in tests]:
        if not path.is_file():
            raise FileNotFoundError(path)
    (prep / "code/tests").mkdir(parents=True)
    for name in sorted(names):
        dest = prep / "code" / name
        shutil.copyfile(ROOT / "code" / name, dest)
        pins[str(dest.relative_to(ROOT))] = sha(dest)
    for name in sorted(tests):
        dest = prep / "code/tests" / name
        shutil.copyfile(ROOT / "code/tests" / name, dest)
        pins[str(dest.relative_to(ROOT))] = sha(dest)
    for name in ("source_contract.json", "run_once.py", "freeze_runtime.py"):
        pins[PREP + "/" + name] = sha(prep / name)
    bindings = {
        "pretrained_manifest": "configs/openh3_20261111/refiner_turbo_30794.json",
        "manifest_sha256": "1023bd6e1df1552240cc7ff4bfac960836d2eeda05e680ecd2eb4b752bf7d975",
        "checkpoint": data["rphi_checkpoint"], "checkpoint_sha256": data["rphi_checkpoint_sha256"],
        "train_manifest": data["train_manifest"], "train_manifest_sha256": data["train_manifest_sha256"],
        "dev_manifest": data["dev_manifest"], "dev_manifest_sha256": data["dev_manifest_sha256"],
        "ordinary_guides_manifest": data["ordinary_manifest"],
        "ordinary_guides_manifest_sha256": data["ordinary_manifest_sha256"],
    }
    for key in ("checkpoint", "train_manifest", "dev_manifest", "ordinary_guides_manifest"):
        pins[bindings[key]] = bindings[key + "_sha256"]
    contract = {"schema": "openh3.train_vae_cache_execution.v1", "root": REMOTE,
        "entry_sha256": sha(prep / "code/export_train_vae_cache.py"),
        "comfy_root": REMOTE + "/code/ComfyUI", **bindings,
        "source_contract": PREP + "/source_contract.json", "source_contract_sha256": sha(prep / "source_contract.json"),
        "selected_sample_id": "rphi-2ac5b3e69cafa70f77d0c6603615ebc56162f4c6bec6e64664ca5bbb7e4b33a9",
        "implementation_sha256": dict(sorted(pins.items())),
        "hard_budget": {"encode": 2, "decode": 1, "Rphi_model_calls": 0, "NFE": 0, "backward": 0, "optimizer_steps": 0},
        "host": "31112", "migration_source_host": "30794", "historical_hold_semantics": "Source30794 no-new-scheme3 restriction carried forward, not a claim of live target-host external hold.",
        "scope": "One source TRAIN ordinary compression engineering cache; no quality, full32 cache, streaming or training claim.",
        "failure_policy": "Preserve failure/partial evidence; no OOM fallback, rescaling, extra forward, retry or overwrite.",
        "other_assets_verification_scope": "Only VAE/context bytes verified live and loaded; remaining full recipe components size/provenance only."}
    write(prep / "execution_contract.json", contract)
    pins[PREP + "/execution_contract.json"] = sha(prep / "execution_contract.json")
    queue = {"schema": "openh3.takeover_queue.v1", "root": REMOTE, "cards": [7], "min_free_mb": 60000,
        "lock_dir": "/cache/zihaohe/.npu-reservations", "fresh_admission": PREP + "/admission.json",
        "environment": {"H3_BASE_ENV": "/cache/yunfeng/envs/minimax-h3-npu", "COMFYROOT": REMOTE + "/code/ComfyUI",
            "PYTHONPATH": REMOTE + "/" + PREP + "/code:" + REMOTE + "/preparation/vae_roundtrip_runtime_20261111_v2/code"},
        "status": RUN + "/queue_status.json", "implementation_sha256": dict(sorted(pins.items())),
        "required_json_values": [
            {"path": "/cache/zihaohe/omni_openh3_20261111/preparation/migration_31112_20261112/SCHEME3_REVIEW_HOLD.source30794.json", "key": "no_new_scheme3_executions", "value": True},
            {"path": data["ordinary_manifest"], "key": "complete", "value": True},
            {"path": data["ordinary_manifest"], "key": "role", "value": "train"},
            {"path": REMOTE + "/" + PREP + "/admission.json", "key": "approved_for_single_train_cache_smoke", "value": True}],
        "steps": [{"name": "single_train_compression_VAE_cache", "argv": [
            "timeout", "--signal=TERM", "--kill-after=30s", "600s", "venv/bin/python",
            PREP + "/code/export_train_vae_cache.py", "--execution-contract", PREP + "/execution_contract.json",
            "--out", RUN + "/cache", "--execute"], "log": RUN + "/logs/export.log"}],
        "automatic_launch": False, "scope": contract["scope"]}
    write(prep / "queue.json", queue)
    write(prep / "freeze.json", {"entry_sha256": contract["entry_sha256"], "implementation_pins": len(pins),
        "execution_contract_sha256": sha(prep / "execution_contract.json"), "queue_sha256": sha(prep / "queue.json"),
        "NPU_launched": False, "fresh_CPU_inspection_and_root_admission_required": True})
    print((prep / "freeze.json").read_text())


if __name__ == "__main__":
    main()
