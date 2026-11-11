"""Strict final-checkpoint guide export without changing stage2 training code.

The original exporter remains unchanged. This wrapper verifies its inference
dependencies against the checkpoint, exports into a new directory, and adds
the native encoder and frozen-calibration provenance to the completed manifest.
Only this strengthened manifest is admitted by run_fixed_h3_rphi.py.
"""
import argparse
import json
from pathlib import Path

from eval_rphi import sha_file
from export_rphi_guides import export_guides


def verify_checkpoint_implementation(payload, code_dir):
    if payload.get("step") != payload.get("training_contract", {}).get("steps") or not payload.get("step"):
        raise ValueError("only the final contracted checkpoint may be exported")
    config = payload.get("model_config", {})
    if config.get("variant") not in ("blind", "blind_confidence") or config.get("k") != 3:
        raise ValueError("expected blind K3 checkpoint")
    files = ["rphi_model.py", "rphi_data.py"]
    if config.get("observation_encoder") == "native":
        files.append("rphi_condition_v2.py")
    implementation = {"code/" + name: sha_file(code_dir / name) for name in files}
    recorded = payload.get("implementation_sha256", {})
    for key, digest in implementation.items():
        if recorded.get(key) != digest:
            raise ValueError("checkpoint/inference implementation mismatch: " + key)
    return implementation


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dev-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frame-batch", type=int, default=1)
    ap.add_argument("--max-sources", type=int)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch
    import rphi_model
    checkpoint_sha = sha_file(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    implementation = verify_checkpoint_implementation(payload, Path(rphi_model.__file__).resolve().parent)
    if sha_file(args.checkpoint) != checkpoint_sha:
        raise ValueError("checkpoint changed while validating")
    proof = {"checkpoint_sha256": checkpoint_sha, "inference_implementation_verified": True,
             "implementation_sha256": implementation,
             "wrapper_sha256": sha_file(__file__),
             "fidelity_state_sha256": payload.get("fidelity_state_sha256"),
             "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
             "calibration_contract": payload.get("calibration_contract"),
             "calibration_step": payload.get("calibration_step"),
             "fidelity_invariance": payload.get("fidelity_invariance")}
    del payload
    try:
        manifest = export_guides(**vars(args))
        # Retain the underlying export receipt, but publish the consumer-facing
        # complete manifest only after the strengthened verification succeeds.
        (args.out / "manifest.json").replace(args.out / "manifest.base_export.json")
        if manifest["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("export used a different checkpoint")
        for key, digest in implementation.items():
            if sha_file(Path(rphi_model.__file__).resolve().parent / Path(key).name) != digest:
                raise ValueError("inference code changed during export")
        manifest["stage2_export_proof"] = proof
        temporary = args.out / "manifest.stage2.partial"
        with temporary.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "manifest.json")
    except Exception as error:
        if args.out.is_dir():
            failure = {"schema": "rphi.guide_export.stage2_failure.v1", "complete": False,
                       "error": type(error).__name__ + ": " + str(error)}
            # Handles even a failed manifest rename. A consumer never sees a
            # successful stage2 manifest after a handled post-export failure.
            (args.out / "manifest.json").write_text(json.dumps(failure, indent=2))
            (args.out / "stage2_failure.json").write_text(json.dumps(failure, indent=2))
        raise
    print(json.dumps({"complete": True, "manifest": str(args.out / "manifest.json"),
                      "samples": manifest["samples"], "scope": manifest["scope"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
