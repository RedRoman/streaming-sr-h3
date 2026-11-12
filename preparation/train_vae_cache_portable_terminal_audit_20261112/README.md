# Portable TRAIN VAE cache terminal audit

This is a separate CPU-only snapshot. The prior cache audit and all producer snapshots remain unchanged. The audit does not import Torch, construct a model, allocate an accelerator, replay the producer, or alter its files. Root supplies and admits the final configuration; none is created or dispatched here.

The producer remains the fixed ordinary compression row from TRAIN source `2BdYaGuVAKo`. All actual array, observation/guide, PIL truncation, raw warm-start, two-encode/one-decode, frozen parameter/buffer, runtime import, and restoration checks remain. RGB stays on the remote host; its actual NPY bytes, shape, finite range and content SHA are checked with a memory map. Only the two FP32 latent arrays may enter the evidence archive.

The portable scope is explicitly narrower than a full recipe check. The full parent manifest SHA and original metadata fingerprint remain bound. Only VAE and context files are stat-ed and independently rehashed. Base, Refiner and Turbo records remain metadata only: their paths are not resolved, stat-ed, hashed or loaded by this auditor. Producer non-use is supported by the frozen source and loader boundary records, not by syscall tracing. The report cannot claim full recipe verification or readiness for a DiT merge. Cache engineering acceptance does not imply any restoration quality result.

Admission schema: `openh3.train_vae_cache_portable_cpu_audit_admission.v1`. Exact fields:

```json
{
  "schema": "openh3.train_vae_cache_portable_cpu_audit_admission.v1",
  "run": "runs/PRODUCER",
  "prep": "preparation/PRODUCER_SNAPSHOT",
  "producer_output_subdir": "cache",
  "out": "runs/FRESH_AUDIT",
  "producer_sha256": "PORTABLE_ENTRY_SHA",
  "legacy_producer_sha256": "5e2e18f3325ed2473bba088ce4fb1d846b0fa356319a1ff64e0c60958f7714a1",
  "asset_helper_sha256": "OPENH3_VAE_ONLY_ASSETS_SHA",
  "execution_contract_sha256": "CONTRACT_SHA",
  "queue_sha256": "QUEUE_SHA",
  "source_sha256": {
    "audit_terminal.py": "SHA",
    "audit_arrays.py": "SHA",
    "audit_inputs.py": "SHA",
    "small_archive.py": "SHA"
  }
}
```

The three producer sources must be in the producer snapshot's `code` directory and in its execution contract's implementation pins. The new entry is `export_train_vae_cache_portable.py`; the unchanged numerical entry is `export_train_vae_cache.py`; the component parser is `openh3_vae_only_assets.py`. All three are included by the archive's explicit producer source collection. The finalizer source SHA is a separate launch argument, preventing a configuration hash cycle.

Run only when root has frozen the configuration and launch admission:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python preparation/train_vae_cache_portable_terminal_audit_20261112/finalize_once.py \
  --root /cache/zihaohe/omni_openh3_20261111 \
  --config /cache/zihaohe/omni_openh3_20261111/preparation/train_vae_cache_portable_terminal_audit_20261112/admission.json \
  --config-sha256 ADMITTED_CONFIG_SHA --self-sha256 FROZEN_FINALIZER_SHA \
  --out runs/FRESH_FINALIZER
```

The wrapper checks the queue every 30 seconds for at most 900 seconds, recognizes `completed`, `failed` and `not_launched`, and invokes the auditor at most once with a 180-second timeout. Existing output paths are rejected. Terminal producer failure remains failure; there are no retries or model interventions. The evidence archive permits at most 100 payload files and 16 MiB including its file-SHA manifest; root must check capacity after the producer snapshot is frozen. The large RGB, guide NPZ and weights never enter the archive.

Test this exact new snapshot with:

```sh
python3 -m unittest discover -s preparation/train_vae_cache_portable_terminal_audit_20261112 -p 'test_*.py' -v
```

`cpu_receipt.json` binds fresh local test execution to the source hashes. Older 13-test receipts are not reused as evidence for this snapshot.
