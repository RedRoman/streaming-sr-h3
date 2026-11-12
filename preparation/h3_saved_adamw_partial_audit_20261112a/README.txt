Saved H3 AdamW partial-artifact audit — preparation only

Entry: code/audit_h3_saved_adamw_partial.py
No existing frozen source was changed. No SSH, model call, NPU call, remote
inspection or actual-artifact audit was executed while preparing this module.

Install the entry and test into this new preparation's code/ and code/tests/.
Keep existing helpers in preparation/h3_real_adamw_terminal_audit_20261112a/;
all seven helper sources are pinned before import. The old audit() orchestration
is NEVER called. Only its small NumPy/serialization methods are reused.

Complete execution_contract.template.json using actual host observations:
- cpu_runtime: Python sys.version, executable path/full SHA, NumPy version and
  absolute canonical .py/.so file SHA map covering all imported NumPy modules.
  Include NumPy extension binaries and applicable external NumPy library pins.
- additional_source_pins: entry/tests plus root runner/contract-independent
  source dependencies. Values for this module/test are already present.
- root_admission: approved=true, scope=CPU_saved_partial_artifacts_only,
  payload_sha256=SHA256(canonical JSON of the contract excluding root_admission;
  sort_keys=True, separators=(',', ':'), allow_nan=False).
Do not add an approved admission until the parent explicitly admits this audit.
The entry SHA and full execution-contract file SHA must both match exactly.

CLI (distinct fresh --out directories):
python ENTRY --execution-contract CONTRACT --execution-contract-sha256 SHA --out INSPECT_DIR
python ENTRY --execution-contract CONTRACT --execution-contract-sha256 SHA --out AUDIT_DIR --execute

Root owns the one-shot Linux supervisor. Enforce 2 CPUs, 4 GiB owned-process-group
RSS, total 900 seconds with bounded cleanup, no retry. Export OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, TORCH_DEVICE_BACKEND_AUTOLOAD=0. The entry
also checks affinity and threads and arms a fixed remaining-time alarm; it does
not implement the external RSS guard. Default inspection reads small JSON,
source/runtime pins, and no NPY, safetensors or pretrained model weights.

Exact required original saved roles: 3222 = 400*(raw/unscaled/clipped gradient +
before/after parameter) + 1200 post-update AdamW states + 22 input/output/retained
video-gradient records. Each required role is attempted once, without retry;
seven separately pinned full22 accepted input/reference arrays are also checked.
No after-reload fragments, new model evaluations or checkpoint updates are used.
Two checkpoint safetensors files are full-SHA checked, their complete key sets,
FP32 shapes, metadata and contiguous offsets are checked, and every tensor is
compared to the corresponding actual post-update NPY bytes.

Numerical gates: exact raw/1024 unscale, saved FP32 clip coefficient multiply,
independent global norm check; first-step AdamW using unchanged helper tolerances:
m rtol3e-6/atol1e-20, v rtol3e-6/atol1e-30, delta rtol3e-6/atol2e-12.
Per-tensor error quantiles/worst normalized residual/failing coordinate count are
reported. These are audit arithmetic tolerances, never quality thresholds.
Zero-coordinate, serialization and continuation byte gates remain exact.

Output: OUT/audit.json, schema openh3.saved_adamw_partial_audit.v1.
- audit_complete: this independent audit ended.
- complete: saved required subset passed; never means producer completed.
- partial_artifact_acceptance: verified_saved_one_update_and_zero_updated_outputs_only.
- saved_subset_ready_for_continuation: complete AND reference/zero video, audio,
  FP16 decoder input and RGB are each byte exact. Needed by any later independent
  admission; complete alone is insufficient.
- binding.checkpoint and binding.optimizer_checkpoint: {path,sha256,bytes}.
- binding.accepted_full22 exactly preserves the prior contract's three references.
- components.saved_update / outputs: status=verified, facts.passed=true and
  facts.actual_arrays_verified=true only after actual bytes and equations pass.
- original_run_accepted, strict_reload_accepted, merge_accepted and
  functional_deployment_ready are always false. Formal training runs remain zero.
Source-bound saved lifecycle/import statements are labelled attestations, not an
independent read of live 20B weights. No final live-state inventory exists.
The original failed finalizer's generic limitations text is not used as evidence
that previous arithmetic or actual-array auditing succeeded.

Local CPU tests: synthetic NumPy NPYS and hand-written safetensors with all 400
adapter names/1200 state keys; original actual failed-run metadata only in the
scope fixture. They do not execute a model, optimizer or actual persisted run.
See cpu_receipt.json and cpu_tests.log. Actual host execution remains pending.
