"""Real-input CPU admission only: no model constructor/forward or NPU import."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("/cache/zihaohe/omni_openh3_20261111")
if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    raise RuntimeError("explicitly disable torch device backend autoload")
sys.path.insert(0, str(HERE / "code"))
from export_rphi_train_controls import inspect_inputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


start = time.monotonic()
pins = json.loads((HERE / "source_pins.json").read_text())
for relative, expected in pins.items():
    if sha(HERE / relative) != expected:
        raise ValueError("source snapshot mismatch: " + relative)
proof, _, _, _ = inspect_inputs(
    ROOT / "runs/rphi_stage2_20261111/legacy_g05/train/checkpoint_step003000.pt",
    ROOT / "datasets/omni_pilot_v1/train_materialized.jsonl",
    ROOT / "datasets/omni_pilot_v1/dev_materialized.jsonl")
assert proof["source_ids"] == ["2BdYaGuVAKo"] and proof["planned_control_samples"] == 8
assert proof["source_isolation"]["train_samples"] == 128
assert proof["source_isolation"]["dev_samples"] == 32
assert proof["inspection_complete"] and not proof["model_constructed"] and proof["model_forward_calls"] == 0
assert "torch_npu" not in sys.modules and not any(k == "comfy" or k.startswith("comfy.") for k in sys.modules)
for relative, expected in pins.items():
    if sha(HERE / relative) != expected:
        raise ValueError("source snapshot changed during inspection: " + relative)
with (HERE / "inspection.json").open("x") as stream:
    json.dump(proof, stream, indent=2, allow_nan=False)
    stream.write("\n")
receipt = {"schema": "rphi.train_controls.real_input_inspection.v1", "passed": True,
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "elapsed_seconds": time.monotonic() - start, "python": sys.executable,
    "inspection_sha256": sha(HERE / "inspection.json"), "inspector_sha256": sha(Path(__file__)),
    "source_pins_sha256": sha(HERE / "source_pins.json"), "train_samples_verified": 128,
    "dev_samples_verified": 32, "selected_training_sources": ["2BdYaGuVAKo"],
    "planned_control_samples": 8, "frames_per_control": 22, "source_provenance": "materialization_attestation_only",
    "model_constructed": False, "model_forward_calls": 0, "torch_npu_imported": False,
    "comfy_imported": False, "npu_calls": 0, "real_control_exported": False, "queue_launched": False}
with (HERE / "inspection_receipt.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(receipt, indent=2), flush=True)
