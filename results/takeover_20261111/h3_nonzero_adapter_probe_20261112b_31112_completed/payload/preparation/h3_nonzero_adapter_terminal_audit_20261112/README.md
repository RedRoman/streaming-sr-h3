# Independent nonzero-adapter terminal audit

Prepared CPU-only tooling. This directory does not admit, launch, restart or modify a model job. It must receive a separately reviewed admission JSON after the producer snapshot is frozen. No admission or dispatch is included here.

The fixed producer is `check_h3_nonzero_adapter.py` SHA `ab710d9a365ac8f502aea177f8993354e2a17d086de97cc71a0e4b265ae99f6b`. The fixed source contract SHA is `d3148504db9fbee2925448ea69e5f2ba83da5eba2127bf3229c9964f445bc08a`, trajectory SHA `69bc9ca998b02cc9b40704743f63ac14806113732bb250f0c28d7b4bee2fce22`.

## Admission and execution

Schema `openh3.nonzero_adapter_cpu_finalizer_admission.v1` has exactly these fields:

- `run`: `runs/<producer-run>`; `prep`: `preparation/<producer-snapshot>`; `out`: a new `runs/<finalizer-run>`.
- `producer_output_subdir`: one safe directory component, normally `eval`.
- `producer_sha256`, `trajectory_sha256`, `execution_contract_sha256`, `queue_sha256`, `source_contract_sha256`, `target_sha256`.
- `source_sha256`: exact SHA mapping for `audit_terminal.py`, `audit_arrays.py`, `audit_checkpoint.py`, `finalize_once.py`, `small_archive.py`.

The admission file must be in this auditor directory. All SHA values are required. The expected target SHA is `10efb5fae35949846d1853d3f2a5812f03b09c15ba0884f62e8e349170c1fef7`.

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python preparation/h3_nonzero_adapter_terminal_audit_20261112/finalize_once.py \
  --root /cache/zihaohe/omni_openh3_20261111 \
  --config /cache/zihaohe/omni_openh3_20261111/preparation/h3_nonzero_adapter_terminal_audit_20261112/ADMISSION.json \
  --config-sha256 REVIEWED_ADMISSION_SHA
```

This command is documentation, not a dispatch. The finalizer creates its new output directory exclusively, locks its own five source files and the producer queue/contract/entry/trajectory hashes, waits up to 4200 seconds (covering the producer's 3600-second limit plus 30-second termination grace) with 30-second observations, invokes exactly one CPU auditor with a 300-second bound, and never retries. Its affinity is limited to two available CPUs. All operations remain outside producer output. The finalizer records completion only for a completed producer plus an independently accepted audit; failed/not-launched producers remain unaccepted.

The direct auditor has the same `--root`, `--config`, `--config-sha256` arguments and requires a new `--out JSON` outside the producer directory. It uses stdlib and NumPy only; no Torch, Comfy, model or NPU imports.

## Acceptance and scope

Actual NPY file/content SHA, role, shape and dtype are checked in place. Complete evidence requires all 400 raw/unscaled adapter gradients, all 400 before/after parameter pairs, retained-video gradients, all four video/audio/RGB outputs, actual decoder FP16 casts, fixed target, strict checkpoint safetensors and the accepted joint reference arrays. The adapter file is independently parsed without Torch: exact 400 keys/shapes/FP32 metadata, bounded non-overlapping storage, finite values, and byte equality with updated arrays.

Hard gates include 12 NFE, 4 VAE decodes, no encodes, exactly one RGB backward/unscale/plain-SGD update/save/destructive clear-reload/merge, all 930 original/recomputed block observations, 76 scoped decoder restorations and four session restorations. Reloaded and updated outputs require strict same-mode equality. Zero-adapter latent parity is strict; its RGB comparison retains candidate-vs-reference `atol=.0002, rtol=.0001`.

Actual saved FP32 raw/1024 gradients must match exactly. The producer's saved same-device expected-after SHA must exactly match the updated bytes. Independently, zero-gradient coordinates must retain their bytes and zero-initialized nonzero-gradient coordinates must match the FP32 product. For nontrivial coordinates, split and float64-fused-then-rounded CPU candidates give descriptive match counts/differences only. Their union is not an acceptance gate and they do not identify or prove an NPU FMA implementation. No new tolerance or learning-rate search is added.

Canonical fingerprint inventories are independently hashed and compared with the accepted joint base/VAE. Exactly 200 named BF16 weight tensors may have one declared write; all biases, untargeted parameters and registered buffers, and the entire VAE must remain unchanged. The actual 20B matrices were not saved: object/version/live-state checks, expected merge hashes and BA/rounding statistics remain source-bound producer attestations. No fresh model load or claimed independent BA recomputation occurs. Merged/unmerged output differences and objective change are descriptive; there is no merge-equivalence or descent threshold.

Each summary reduction alone has `rtol=1e-12, atol=1e-15` for independently reconstructed scalar diagnostics. Array hashes, checkpoint identity, unscale and exact output gates use no such tolerance. Torch FP32 MSE and independently recomputed NumPy FP64 MSE are recorded separately.

The immutable small archive is capped at 128 files / 64 MiB uncompressed. It contains approved small output/input arrays, JSON/log/source evidence, and a manifest. Gradient/parameter arrays and the diagnostic adapter checkpoint are checked in place and excluded. This is a synthetic one-update persistence/numerical check, not restoration quality, optimizer convergence, full-resolution capacity or deployment acceptance.

## CPU tests

```
python3 -m unittest discover -s preparation/h3_nonzero_adapter_terminal_audit_20261112 -p test_audit.py -v
```

Tests use synthetic saved evidence, including all 400 adapter keys and actual small safetensors/NPY files. They also check four-output RGB MSE/casts, exact-vs-descriptive comparisons, budget failures, all-parameter/buffer merge boundaries, context counts, missing/failed evidence, path containment, bounded waiting and immutable archive exclusions. They do not run H3/VAE or validate an actual future producer run.
