# Small joint RGB independent terminal audit / bounded finalizer

Prepared only. No admission configuration, upload, finalizer or model launch is created by this snapshot. Reviewed producer SHA and the actual Torch CPU target SHA are in cpu_receipt.json. Old snapshots are unchanged.

The NumPy-only audit verifies all813 successful actual NPY arrays:11 small input/output/target/cast arrays,400 canonical adapter raw/unscaled pairs,and2 retained-video gradient arrays. Files and content hashes, shapes and dtypes are checked. Latent video/audio strict equality and FP16 casts are independently recomputed. RGB allclose uses candidate relative to reference with the existing atol.0002/rtol.0001. Target is bound to the actual Torch CPU linspace bytes, never generated using a potentially different NumPy rounding recipe.

Actual saved RGB/target independently reproduce numpy_float64_mse. torch_float32_mse remains the actual backward scalar, and its difference from NumPy64 is reported; no invented Torch/NumPy reduction tolerance is a training gate. Raw/1024 FP32 gradient bytes must match exactly. Canonical400 tensor identities, finite/count/norm facts and B aggregate nonzero are checked; zero A and some zero B tensors are legal. Float64 descriptive reductions alone use rtol1e-12/atol1e-15; arrays, updates, counts and parity never use this metadata tolerance.

Full named-parameter and registered-buffer inventory JSON hashes are recomputed and initial/final group hashes compared. Lifecycle object/version/device/frozen checks are source-bound producer attestations; the auditor does not reload40GB weights or claim unregistered Python-cache invariance. Actual6NFE/2decoder/1backward budgets,50x3 DiT and36x1 VAE phases,558 block math observations,73 decoder context restorations (37 forward including outer decode +36 recompute), session and whole-run patcher restoration, wrapper restoration, imports and source pins are checked. Failed and not_launched queues are never promoted; unavailable failure evidence stays unavailable.

Finalizer admission schema:openh3.joint_rgb_cpu_finalizer_admission.v1. Fields:run,prep,out (two-component workspace-relative paths);producer_sha256,trajectory_sha256,execution_contract_sha256,queue_sha256,target_sha256;source_sha256 mapping exactly audit_terminal.py/audit_arrays.py/finalize_once.py/small_archive.py. Configuration must be reviewed, in this preparation, and passed with its exact SHA. Source pins are checked at startup and terminal time. Fresh output refuses repeat/resume.

Only queue JSON is polled every30s for≤2400s. completed/failed/not_launched are terminal; the auditor runs once with≤180s timeout. No Torch/H3/NPU, npu-smi, job controls, producer writes or retries. Immutable finalize_receipt is packed; finalize_status outside carries archive SHA. Archive cap100files and16MiB uncompressed (including MANIFEST); count against the final producer preparation before admission. Small arrays each≤1MiB, so the two5x128x128x3 FP32 RGBs fit. No gradient NPY or weights are packaged. Timeout never packages active producer evidence.

Nine local CPU tests passed covering wrongtarget/MSE, exact vs tolerant parity, cast/filehash/unscale, wrongcounts and restoration, actual2RGB archive/manifest and gradient exclusion, bounded timeout/failed/not_launched. No actual joint result is claimed. After root approval, create one fresh config and run:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python preparation/h3_joint_rgb_terminal_audit_20261112/finalize_once.py --root /cache/zihaohe/omni_openh3_20261111 --config preparation/h3_joint_rgb_terminal_audit_20261112/admission.json --config-sha256 <reviewed-exact-SHA>
```
