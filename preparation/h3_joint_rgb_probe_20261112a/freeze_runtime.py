"""Freeze a new joint probe runtime after reviewed CLI/CPU evidence; never launch."""
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/cache/zihaohe/omni_openh3_20261111"
PREP = "preparation/h3_joint_rgb_probe_20261112a"
RUN = "runs/h3_joint_rgb_probe_20261112a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main():
    prep = ROOT / PREP
    if (prep / "execution_contract.json").exists() or (prep / "code").exists():
        raise FileExistsError("runtime already frozen or partial; never overwrite")
    source_contract = json.loads((prep / "source_contract.json").read_text())
    if sha(prep / "source_contract.json") != "1d9acc5451dce4f0454832ff86dffc98eb85f8ed07c9d13c73b776367f560cc7":
        raise ValueError("joint protocol changed")
    for relative, expected in source_contract["source_reading_basis"]["sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("reviewed source changed: " + relative)
    prior_h3 = json.loads((ROOT / "preparation/h3_trajectory_probe_20261111b/execution_contract.json").read_text())
    prior_vae = json.loads((ROOT / "preparation/vae_latent_descent_20261111j/queue.json").read_text())
    pins = dict(prior_h3["implementation_sha256"])
    for name, value in prior_vae["implementation_sha256"].items():
        if name in pins and pins[name] != value:
            raise ValueError("prior runtime lock disagreement: " + name)
        pins[name] = value
    names = {p.name for base in ("h3_trajectory_probe_20261111b", "vae_latent_descent_20261111j")
             for p in (ROOT / "preparation" / base / "code").glob("*.py")}
    names.add("check_h3_joint_rgb_grad.py")
    # Copy only this repository's reviewed implementations, never external packages.
    for name in sorted(names):
        source = ROOT / "code" / name
        if not source.is_file():
            raise FileNotFoundError(source)
    test_names = ("test_h3_joint_rgb_probe.py", "test_openh3_train_trajectory.py")
    for name in test_names:
        if not (ROOT / "code/tests" / name).is_file():
            raise FileNotFoundError(ROOT / "code/tests" / name)
    (prep / "code/tests").mkdir(parents=True)
    for name in sorted(names):
        destination = prep / "code" / name
        shutil.copy2(ROOT / "code" / name, destination)
        pins[str(destination.relative_to(ROOT))] = sha(destination)
    for name in test_names:
        destination = prep / "code/tests" / name
        shutil.copy2(ROOT / "code/tests" / name, destination)
        pins[str(destination.relative_to(ROOT))] = sha(destination)
    for name in ("source_contract.json", "run_once.py", "freeze_runtime.py"):
        pins[PREP + "/" + name] = sha(prep / name)
    contract = {
        "schema": "openh3.joint_rgb_probe_execution.v1", "root": REMOTE,
        "status": "frozen_runtime_pending_CPU_inspection_and_fresh_admission",
        "probe_sha256": sha(prep / "code/check_h3_joint_rgb_grad.py"),
        "manifest_sha256": prior_h3["manifest_sha256"],
        "context_sha256": prior_h3["context_sha256"],
        "source_contract": PREP + "/source_contract.json",
        "source_contract_sha256": sha(prep / "source_contract.json"),
        "implementation_sha256": dict(sorted(pins.items())),
        "fixed_budget": source_contract["hard_budget"],
        "failure_policy": source_contract["execution"]["failure_policy"],
        "native_geometry_clarification": "No external padding or tiling-policy change. Preserve the pinned deployed decoder's own temporal padding/chunking and spatial tiling; record their actual native plan. This is the same reference decoder path, not a fallback.",
        "scope": "small synthetic same-graph RGB backward; no optimizer, quality, full-resolution or generalization acceptance",
    }
    write(prep / "execution_contract.json", contract)
    pins[PREP + "/execution_contract.json"] = sha(prep / "execution_contract.json")
    guards = [
        {"path": "/cache/haoyi/video_sample/cvpr27/res/all_models/checks/qwen_prompt_v3_review_20261103/h3_pulse3_early10_fixed30_20261104/SCHEME3_REVIEW_HOLD.json", "key": "no_new_scheme3_executions", "value": True},
        {"path": REMOTE + "/runs/rphi_stage2_20261111/finalize_status.json", "key": "state", "value": "completed"},
        {"path": REMOTE + "/runs/h3_trajectory_probe_20261111b_finalize/terminal_audit.json", "key": "complete", "value": True},
        {"path": REMOTE + "/runs/vae_latent_descent_20261111j_audit/audit.json", "key": "complete", "value": True},
        {"path": REMOTE + "/runs/vae_gradient_repeatability_20261111i/a1_a2_comparison.json", "key": "a1_a2_exact", "value": False},
        {"path": REMOTE + "/runs/vae_gradient_repeatability_20261111i/a1_a2_comparison.json", "key": "allow_paired_history", "value": False},
        {"path": REMOTE + "/" + PREP + "/admission.json", "key": "approved_for_single_card_joint_rgb_probe", "value": True},
    ]
    queue = {"schema": "openh3.takeover_queue.v1", "root": REMOTE, "cards": [6],
        "min_free_mb": 60000, "lock_dir": "/cache/zihaohe/.npu-reservations",
        "environment": {"H3_BASE_ENV": "/cache/yunfeng/envs/minimax-h3-npu",
            "PYTHONPATH": REMOTE + "/" + PREP + "/code:" + REMOTE + "/preparation/vae_roundtrip_runtime_20261111_v2/code",
            "COMFYROOT": REMOTE + "/code/ComfyUI"},
        "status": RUN + "/queue_status.json", "scope": contract["scope"],
        "implementation_sha256": dict(sorted(pins.items())), "required_json_values": guards,
        "steps": [{"name": "same_graph_small_RGB_backward", "argv": [
            "timeout", "--signal=TERM", "--kill-after=30s", "1800s", "venv/bin/python",
            PREP + "/code/check_h3_joint_rgb_grad.py", "--pretrained-manifest",
            "configs/openh3_20261111/refiner_turbo_30794.json", "--execution-contract",
            PREP + "/execution_contract.json", "--out", RUN + "/eval", "--execute"],
            "log": RUN + "/logs/probe.log"}],
        "automatic_launch": False, "one_shot_launcher": PREP + "/run_once.py"}
    write(prep / "queue.json", queue)
    write(prep / "freeze.json", {"entry_sha256": contract["probe_sha256"],
        "contract_sha256": sha(prep / "execution_contract.json"),
        "queue_sha256": sha(prep / "queue.json"), "implementation_pins": len(pins),
        "copied_sibling_modules": sorted(names), "source_contract_sha256": contract["source_contract_sha256"],
        "NPU_launched": False, "CPU_inspection_required": True})
    print((prep / "freeze.json").read_text())


if __name__ == "__main__":
    main()
