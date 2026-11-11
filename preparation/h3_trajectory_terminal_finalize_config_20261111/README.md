# Configured one-shot CPU terminal finalizer

This is a new independent implementation. The frozen a finalizer and its failed result remain unchanged. No b admission, upload, or launch is implied by these sources.

Create a fresh `admission.json` in this preparation only after the new producer is reviewed. Its exact keys are `schema` (`openh3.latent_finalizer_admission.v1`), workspace-relative `run`, `prep`, `out`, and six lowercase SHA-256 fields: `auditor_sha256`, `finalizer_sha256`, `execution_contract_sha256`, `queue_sha256`, `probe_sha256`, `trajectory_sha256`. Each run/prep/out has exactly two path components; out must be separate from the producer run. No inferred or automatically updated source pins. Pass its byte SHA explicitly:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python preparation/h3_trajectory_terminal_finalize_config_20261111/finalize_once.py \
  --root /cache/zihaohe/omni_openh3_20261111 \
  --config preparation/h3_trajectory_terminal_finalize_config_20261111/admission.json \
  --config-sha256 <reviewed-exact-config-byte-SHA>
```

Existing output refuses repeat/resume. Startup and terminal-time pin checks bind finalizer/auditor source, producer queue and contract. The contract must bind the admitted producer/trajectory and workspace. The captured config plus SHA are saved in the status; later config edits cannot change this process's admission.

The only poll is producer queue JSON every 30 seconds, at most 2400 seconds. At terminal completed/failed, the pinned NumPy auditor executes once with a separate 180-second timeout. No Torch/H3 import, NPU calls, model starts/stops/restarts, or producer writes. A producer failure never becomes numerical success. New import audits require exact torch.ops/torch.classes class identities and both real defining-source hashes, not their inherited pseudo filenames.

Archive allowlist remains at most 100 files / 16 MiB uncompressed including MANIFEST, with at most 1 MiB each for only latent_outputs.npz/reference_latents.npz. Never packs NPY gradients or weights. Immutable finalize_receipt is inside; final archive SHA is in finalize_status outside to avoid a hash cycle. Timeout leaves producer untouched and does not package its live logs/metrics.

20 local pure NumPy/stdlib tests passed (receipt and log included); zero Torch imports and zero NPU calls. Tests cover explicit new pins, rejection of old or pseudo-source pins, actual 400 tiny gradient arrays, parity/noise/frozen evidence, failed producer preservation, bounded polling, archive manifest, overlap/escape prevention, and duplicate refusal. No actual b execution is claimed.
