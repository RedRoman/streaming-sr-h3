"""CPU-only frozen source/asset/seed inspection. Never imports H3 or NPU."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    prep = Path(__file__).resolve().parent
    root = prep.parents[1]
    began = time.monotonic()
    plan = json.loads((prep / "queue.json").read_text())
    contract = json.loads((prep / "execution_contract.json").read_text())
    for relative, expected in plan["implementation_sha256"].items():
        path = root / relative
        if sha(path) != expected:
            raise ValueError("implementation pin mismatch: " + str(path))
    sys.path.insert(0, str(prep / "code"))
    from openh3_assets import read_manifest
    manifest_path = root / "configs/openh3_20261111/refiner_turbo_30794.json"
    if sha(manifest_path) != contract["manifest_sha256"]:
        raise ValueError("wrong asset manifest")
    manifest = read_manifest(manifest_path, verify=False)
    records = {role: manifest[role] for role in ("base", "vae", "context")}
    records.update({row["role"]: row for row in manifest["components"]})
    def check(role):
        path, row = Path(manifest["_paths"][role]), records[role]
        found = sha(path)
        if found != row["sha256"]:
            raise ValueError("asset SHA mismatch: " + role)
        return role, {"path": str(path), "bytes": path.stat().st_size, "sha256": found,
                      "loaded_as_model": False}
    # Two streaming readers, bounded memory; no large tensor/checkpoint loading.
    with ThreadPoolExecutor(max_workers=2) as pool:
        assets = dict(pool.map(check, records))
    import torch
    import numpy as np
    from probe_vae_latent_descent import contract as current_contract, SHAPE, INPUT_SHA
    source = torch.randn(SHAPE, generator=torch.Generator(device="cpu").manual_seed(102), dtype=torch.float32)
    seed_sha = hashlib.sha256(source.numpy().tobytes()).hexdigest()
    if seed_sha != INPUT_SHA or current_contract() != contract["frozen_protocol"]:
        raise ValueError("seed102 bytes or frozen contract differs")
    if "torch_npu" in sys.modules or "h3_comfy" in sys.modules:
        raise RuntimeError("CPU inspection unexpectedly imported a device/model runtime")
    guards = []
    for guard in plan["required_json_values"]:
        if str(guard["path"]).endswith("/admission.json"):
            continue  # exclusively root's fresh admission after this preparation
        value = json.loads(Path(guard["path"]).read_text()).get(guard["key"])
        if value != guard["value"]:
            raise ValueError("historical guard mismatch: " + str(guard))
        guards.append(guard)
    receipt = {"schema": "openh3.vae_descent_queue_CPU_inspection.v1", "complete": True,
        "seconds": time.monotonic() - began, "queue_sha256": sha(prep / "queue.json"),
        "implementation_files_verified": len(plan["implementation_sha256"]),
        "manifest_sha256": sha(manifest_path), "asset_bytes_SHA_verified": assets,
        "source_seed102_sha256": seed_sha, "frozen_protocol_matches": True,
        "historical_guards_verified": guards, "fresh_admission": "not_created; root must independently admit",
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "TORCH_DEVICE_BACKEND_AUTOLOAD": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")},
        "NPU_calls": 0, "model_loads": 0, "torch_npu_imported": False,
        "prior_i_exact_gate": "failed retained", "paired_history_B": "skipped retained"}
    with (prep / "inspection.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"complete": True, "seconds": receipt["seconds"], "files": receipt["implementation_files_verified"],
                      "asset_roles": list(assets), "source_seed102_sha256": seed_sha, "NPU_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
