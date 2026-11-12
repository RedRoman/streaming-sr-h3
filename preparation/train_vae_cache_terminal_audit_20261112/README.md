# Single TRAIN-row VAE cache terminal audit

This is a standalone NumPy/standard-library auditor for the frozen
`export_train_vae_cache.py` schema `openh3.train_vae_cache.v1`. It does not import
Torch, build a model, access an accelerator, rerun the producer, wait on a queue,
alter a source cache, or decide restoration quality. Run it once after the
producer queue reaches `completed`, `failed`, or `not_launched`. Existing audit
output and archives are refused. There is no retry or repair mode.

The only admitted source is the previously accepted **TRAIN** source
`2BdYaGuVAKo`, ordinary compression, all 22 frames at 480×832. The accepted guide
manifest SHA and legacy final3000 checkpoint SHA are fixed in `audit_inputs.py`.
Original source-video provenance remains the existing materialization
attestation; no new original-video rehash or pristine-data claim is made.

## Independent checks

- Read the actual selected observed NPZ and all three input guide arrays. Check
  file/content hashes, ordinary TRAIN identity, raw draft = observed uint8 / 255,
  and exact Rphi clipping. Target NPZ bytes are hashed; its pixel array is never
  loaded by this auditor.
- Recompute official same-geometry PIL uint8 truncation and float32 / 255 hashes
  frame by frame, then match both CPU inspection and actual VAE-input observers.
  These checks do not bypass the producer's pinned official preprocessing.
- Read all three output NPYs, including mmap access to the approximately 105 MB
  RGB result. Check file/content SHA, FP32 dtype, geometry and finiteness. Check
  the actual decoder consumed saved Z_R, including its exact FP16 stage cast.
  Z_raw remains the future raw warm-start. Z_raw = Z_R is reported, not treated
  as a quality failure.
- Enforce 2 VAE encodes, 1 Rphi-latent decode, corresponding whole-stage counts,
  22 PIL frames per arm, and no Rphi/H3 sampling/backward/optimizer work.
- Rehash the recorded full named-parameter and registered-buffer inventories,
  compare versions/device/bytes, and check source-bound object/freeze/import
  and wrapper/math restoration evidence. This is not an independent live
  re-read of model weights or unregistered Python caches.

`VAE_R` means H3's as-returned RGB **after the VAE internal clamp**. No VAE
pre-clamp data is claimed. PSNR, LPIPS, a fidelity threshold, training benefit,
generalization and model quality are not acceptance gates.

## Admission and command (not configured or dispatched here)

Root must create a new JSON admission next to these files with exactly:

- `schema`: `openh3.train_vae_cache_cpu_audit_admission.v1`
- `run`: `runs/<producer run>`; `prep`: `preparation/<frozen producer prep>`
- `producer_output_subdir`: the producer's single output child, e.g. `eval`
- `out`: a fresh `runs/<audit output>` distinct from the producer
- `producer_sha256`, `execution_contract_sha256`, `queue_sha256`
- `source_sha256`: exact hashes of `audit_terminal.py`, `audit_arrays.py`,
  `audit_inputs.py`, `small_archive.py`

Producer prep must contain `code/export_train_vae_cache.py`,
`execution_contract.json`, and `queue.json`. Bind all hashes to the final
admitted runtime, not the earlier CPU inspection contract.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python preparation/train_vae_cache_terminal_audit_20261112/audit_terminal.py \
  --root /cache/zihaohe/omni_openh3_20261111 \
  --config preparation/train_vae_cache_terminal_audit_20261112/admission.json \
  --config-sha256 <root-reviewed admission SHA>
```

The output contains `invocation_receipt.json`, `terminal_audit.json`,
`audit_status.json`, and `evidence.tar.gz`. Producer failure is always preserved.
Missing early-failure evidence is labelled unavailable, not reconstructed.
`quality_verdict` is always `not_assessed`.

The archive includes source/queue/admission/JSON/log evidence and at most the
two verified latent NPYs (about 2 MiB total). Large RGB, input guide NPZs and
weights stay remote. Every payload has a SHA/size entry in `MANIFEST.json`.
Hard limits are 100 payload files and 16 MiB uncompressed, including the
manifest; exceeding a limit fails visibly rather than silently dropping proof.

Local tests exercise altered files/content hashes, nonfinite/wrong geometry,
PIL floor versus rounding, exact Rphi clipping, data-role/control relabels,
counts, wrong decoder latent/cast, registered-buffer/version mutations,
failed/not-launched states and archive exclusions. They do not claim a real
VAE export or a successful production terminal audit.

## Optional bounded CPU finalizer

`finalize_once.py` adds only a queue-status wait: every 30 seconds, at most 900
seconds, including immediate handling of `not_launched`. After a terminal state
it invokes `audit_terminal.py` once, with a 180-second subprocess timeout. It
does not launch, cancel or restart the producer. A producer/audit failure remains
a failure. A wait timeout makes no audit invocation.

The existing four-source admission above remains unchanged. Root separately
pins the finalizer source SHA in the launch command's `--self-sha256`; there is
no config/self-hash cycle. `--out` is a fresh status directory separate from both
the producer run and the auditor output in admission. Its final receipt points
to the auditor's small archive and records the exact admission and wrapper SHA.
The archive also includes the wrapper source from this preparation directory.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python preparation/train_vae_cache_terminal_audit_20261112/finalize_once.py \
  --root /cache/zihaohe/omni_openh3_20261111 \
  --config preparation/train_vae_cache_terminal_audit_20261112/admission.json \
  --config-sha256 <root-reviewed admission SHA> \
  --self-sha256 <frozen finalizer source SHA> \
  --out runs/<fresh finalizer status directory>
```

No admission or dispatch is performed by this preparation task. Resource
unavailability is not permission to wait for or start an accelerator job; this
wrapper only observes an independently admitted producer's queue status.
