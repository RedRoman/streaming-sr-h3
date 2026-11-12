# Saved AdamW reload/merge continuation preparation

This is a new entry and a new execution contract. It does not restart or relabel
the timed-out `h3_real_adamw_update_20261112a` run. No SSH or model execution was
performed while preparing this directory.

The default CLI inspects pinned metadata, actual saved inputs/outputs, original
adapter/optimizer checkpoint bytes and the exact TRAIN cache. It imports no
Torch, Comfy or NPU runtime. Both inspection and execution require the separately
completed CPU saved-subset audit. The template contains deliberately unfilled
audit and admission fields and is not executable as supplied.

```
venv/bin/python preparation/h3_adamw_saved_continuation_preparation_20261112b/code/check_h3_adamw_saved_continuation.py --pretrained-manifest configs/openh3_20261111/refiner_turbo_30794.json --execution-contract NEW_CONTRACT.json --out NEW_INSPECTION
```

Root alone creates the full runtime snapshot, fills its implementation pins and
fresh root admission, and uses the separately frozen bounded runner with
`--execute`. No continuation executes automatically after an audit.

The root admission must have `approved: true`, scope
`saved_TRAIN_AdamW_reload_merge_continuation`, the entry SHA,
`execution_payload_sha256` (canonical JSON excluding `root_admission`) and the
exact `partial_cpu_audit_sha256`. The execution contract schema is
`openh3.saved_adamw_continuation_execution.v1` and includes an absolute `root`.

The fixed work is 6 NFE / 2 native VAE decodes / one 200-matrix merge, with zero
backward, optimizer step, unscale, clipping, encoding, Rphi and new training
checkpoint saves. Fresh loading of the saved adapter and all AdamW state occurs
once. This is not a test of a resumed next optimizer step. Proposed external
limits are 3600 seconds plus 30 seconds owned-process cleanup, 4 CPU cores and
128 GiB owned-process-group RSS; there is no retry or smaller tile fallback.

The first inference uses the unchanged separate FP32 adapter branch. Saved
video, audio, actual FP16 decoder input and RGB must match the old `updated`
arrays byte for byte before any merge. The second uses the declared FP32 addmm
and single BF16 cast merge. Merge output changes and rounding are measured,
with no numerical equivalence threshold or quality claim. The existing
functional gate requires exact original zero/reference RGB and changed merged
RGB coordinates. It can remain false even when execution completes.

Each successful assertion is in memory via `h3_evidence_checks.require`.
Failures, explicit phase transitions, trajectory steps and bounded adapter/state
inventory progress are durable. Each of the actual restored 400 adapter and
1200 optimizer state tensors is copied to CPU once for its saved NPY metadata;
large arrays stay remote. Full base/buffer inventories are initial and final,
with full frozen checks after saved load, before merge and during declared merge
validation. Other decode boundaries check lifecycle. No full base is cloned.

CPU fixture command, with backend auto-loading disabled:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -m unittest discover -s preparation/h3_adamw_saved_continuation_preparation_20261112b/code/tests -p test_h3_adamw_saved_continuation.py -v
```

The local receipt reports 6 passing pure NumPy/standard-library tests and 4
skipped Torch fixtures because local Torch is unavailable. The Torch tests cover
fresh exact optimizer restoration, rejection of invalid states, actual saved
NPY identity/no-overwrite, canonical frozen objects and a real 200-matrix toy
BF16 merge. They still need one actual CPU run in the existing environment.

The result schema is `openh3.saved_adamw_continuation.v1`. `original_run_accepted`
always remains false. A complete result establishes only this continuation's
engineering checks and links the original failure and separately audited saved
update; it is not a monolithic 12-NFE budget pass, formal training, quality,
real-time throughput or generalization acceptance.

This b snapshot corrects only the frozen recipe field mapping `actual_lr` to the
Torch parameter-group field `lr`, preserving its exact float value and all other
optimizer settings. The a snapshot and its failed remote CPU fixture remain
unchanged (failure log SHA ef103b079dd7a541da4c07d8594c178602388b6966eae0dae23d690e1022ed64).
