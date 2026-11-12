# Saved matched/shuffled heads: independent CPU artifact audit

This preparation does not run or import the trainer, head model, optimizer,
restorer, evaluation, or NPU extension. It reads four authenticated `.pt` files
using `torch.load(map_location='cpu', weights_only=True)` and JSON metadata.
No NPZ is opened, and no source image, label array, forward, backward or update
is replayed. It never changes the two research runs.

`config.json` binds the two actual completed research reports and contracts
from the root-downloaded training evidence archive SHA
`35c727be4436fab011eb44fa70abffb2ddb8478f634f4920e80505f661ba9d9f`.
Reading these four archive JSON members did not extract or download the archive
again. The configuration binds this auditor's own bytes and one fresh output.
Root owns remote preparation and the single invocation.

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHON -m unittest discover -s preparation/input_preservation_heads_actual_audit_20261112a -p test_audit_actual.py -v
PYTHON preparation/input_preservation_heads_actual_audit_20261112a/audit_actual.py --config preparation/input_preservation_heads_actual_audit_20261112a/config.json --config-sha256 CONFIG_SHA --out /cache/zihaohe/omni_openh3_20261111/runs/input_preservation_heads_actual_audit_20261112a --execute
```

Replace `PYTHON` with the existing CPU environment's Python. Omitting
`--execute` performs only metadata inspection and claims a fresh output;
do not use the final actual output for a separate inspection.

Execute enforces Linux CPU affinity 2, address-space limit 4GiB, and a 120-second
wall alarm. It starts no child process. TERM/INT/timeout/errors leave a failed
`audit.json` and are not retried; existing output is rejected. A killed process
may lack its terminal report and must not be considered accepted.

The audit independently checks file hashes and unchanged bytes, same exact
initial model/RNG tensors, four FP32 parameter tensors totaling1745 per state,
changed final states, four AdamW parameter states with finite moments and saved
step1000, saved contracts and separate root admissions, both smoke report
bindings, all1000 logged learning rates and their binary float representations,
the original16000 feature row order and feature batch hashes against the bound
cache index, matched label batch hashes, and shuffled label indices against the
authenticated16000-row cross-source-and-group bijection. All1000 finite loss
and gradient diagnostics are inspected.

The saved optimizer is a final state: this audit does not independently
reconstruct1000 updates. Per-step model operations and strict live reload remain
producer attestations. Shuffled numeric label batch hashes are recorded but are
not recomputed without reading label NPZ. Same initial states are checked against
each other, without constructing a seeded model. Final model hashes can differ
without establishing quality. This audit does not run or approve the development
screen, future training, or full-frame validation.

Local tests: five pure metadata tests passed; two genuine Torch saved-tensor
fixtures skipped because local Torch is absent. The receipt does not claim those
two passed. Root can run them in the existing CPU runtime; no dependency install
is requested.

The source bundle has project-relative member names beginning
`preparation/input_preservation_heads_actual_audit_20261112a/`.
Extract once at the project root, never inside the preparation directory.
