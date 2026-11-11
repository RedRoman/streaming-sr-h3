# Isolated official native baseline CPU verification

The official MMCV CPU extension and both models' strict checkpoint loads and
complete synthetic CPU forwards passed. AverNet uses the explicit architecture
import described below; the original package startup still lacks `pyiqa`.
See `evidence/summary.json` for the completed CPU execution evidence.

A subsequent bounded quality queue launched on 31984 at 20:17:12 Singapore time
on 2026-11-11. It processes only source `6UZVovkR3IU`, three native tasks, and two
models, with at most six complete 22-frame 480x832 forwards. Its separate
`evidence/quality_queue/launch_receipt.json` and `partial_snapshot.json` record
the actual launch, environment gates, process guards, and incomplete state.
The immutable prelaunch plan is a historical snapshot; the ETA correction receipt
fixes its arithmetic from 54 to approximately 74.68 inference minutes, plus
5–15 metric minutes. This is still an extrapolation, pending the first real clip.

Monitor `huawei-modelarts-31984` at
`/cache/zihaohe/omni_openh3_20261111/preparation/native_quality_smoke_20261111_v1/queue_status.json`.
Per-task `infer_progress.json` files update every 30 seconds in the two new run
directories. Do not edit any admitted runtime files while this queue is active.

This folder contains a reproducible execution probe and evidence, not an accuracy
benchmark. No NPU is allocated, no training runs, and no shared Python environment
is modified. The official FaverNet Common and AverNet models remain offline,
bidirectional native-resolution baselines; this does not establish streaming,
super-resolution, or Ascend compatibility.

The isolated remote workspace is:

`huawei-modelarts-31984:/cache/zihaohe/omni_openh3_20261111/preparation/native_baselines_cpu_20261111_v1`

The official model assets are read from the existing sibling
`native_baselines_20261111/third_party/native_video_baselines_20261111`.
MMCV is the unmodified official v2.2.0 source at commit
`3a5fb4b96ab1b241e7379adf7c326ef6edef6515`, compiled for CPU in a new
`venv --system-site-packages`. All new dependencies, caches, and build products
stay inside that new workspace. `TORCH_DEVICE_BACKEND_AUTOLOAD=0` prevents NPU
backend loading; `FORCE_NPU=0` and `FORCE_CUDA=0` select CPU compilation.

`verify_official_cpu.py` requires the real compiled MMCV extension. It:

1. Compares official MMCV CPU modulated deformable convolution with torchvision's
   CPU implementation, using 24 deformable groups, nonzero offsets, and varying
   masks (`atol=rtol=2e-5`). Torchvision is a reference, never a model fallback.
2. Checks every locked model Python source file and the exact checkpoint SHA256.
3. Requires `load_state_dict(strict=True)` with zero missing/unexpected keys and
   checkpoint-owned SPyNet weights.
4. Runs the complete official model on deterministic uint8 RGB input with
   shape `T=2,H=W=256`, checking CPU output shape and finiteness. Raw float output
   hashes and elapsed time are recorded. No ground truth is read, no image-quality
   claim is made, and the random synthetic input is not a restoration evaluation.

Each model has its own receipt. A failure or timeout in one probe does not stop
another user's work. An environment or weight-deserialization receipt is not a
successful model-load or model-forward receipt.

AverNet additionally supports an explicit `--aver-import-mode architecture_only`
probe. It creates only Python package namespaces for `basicsr`, `basicsr.archs`,
and `basicsr.utils`, omitting their training/data startup imports. The official
registry, architecture utilities, SpyNet, AverNet, and real MMCV operators execute
unchanged and remain source-hash checked. This is never an automatic fallback:
the default official package-entry result and the architecture-only result have
separate receipts. Passing the latter does not mean the full official training or
evaluation CLI and its optional dependencies are installed.

Example after successful MMCV build:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ./venv/bin/python verify_official_cpu.py \
  --assets /cache/zihaohe/omni_openh3_20261111/preparation/native_baselines_20261111/third_party/native_video_baselines_20261111 \
  --baseline faver --output ./faver
```

Commit only the small probe source, source lock, scripts, and JSON/text evidence.
Downloaded source archives, compiled extensions, Python environments, dependency
caches, and synthetic output arrays are intentionally ignored and remain outside
the project's tracked runtime. Read the final summary receipt for execution
status; merely starting compilation does not count as operator/model validation.
