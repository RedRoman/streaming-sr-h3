# Snapshot scope

Captured: 2026-10-15T22:19:40+08:00 (Asia/Shanghai). Previous snapshot: 2026-10-15T17:20:33+08:00.

## Included

- Project Python/shell source, protocols, original brief, research notes and the current experiment report.
- JSON/CSV/Markdown numerical evidence, including preserved failed or superseded experiment versions. Read each report's validity labels; a stored result is not automatically a valid conclusion.
- Dataset source/split/hash manifests and license notices, without raw datasets.
- Four existing result panels and ten additional compact summary figures/PDF exports, including H3, B5, Relay and delayed-LR results.

## Excluded

Model weights, learned checkpoints, binary caches, input/output frame collections, downloaded external repositories, virtual environments, terminal logs, live session/PID state and credentials are not included. Paths referenced by reports may point to local-only artifacts. No source dataset or H3 weights are redistributed here. Sealed confirmation images were neither opened nor copied by this snapshot operation.

## Reproducibility

This is a research archive, not a one-command packaged release. Re-running GPU experiments requires separately obtained MiniMax-H3 weights, the matching ComfyUI source, datasets and CUDA dependencies. Paths in experiment scripts may need adjustment. Dataset identities/splits and artifact hashes are retained for verification. Hardware-specific and historical absolute paths describe the original runs.

Included Python source and JSON were parsed, and known credential patterns were scanned before copying. Training was not rerun while publishing this snapshot. Each source file's size and modification time was checked before and after reading; related files may reflect slightly different moments because active research continues. The manifest hashes describe the captured bytes, not a claim that every ongoing experiment is complete.

## Current evidence

- Eight-source native480 H3 sampling consistency: PSNR improves over the matched VAE on 7/8 sources and LPIPS on 8/8; temporal and compute tradeoffs remain.
- B5: all four fixed 672-update arms and validation completed. Physical observed-detail bypass helps; its learned correction does not beat the same zero-training base.
- Delayed LR: the fixed four-arrival rule gets 5/6 source signs correct, with a positive six-source mean but negative two-source validation mean. It is a diagnostic, not a validated new core.
- The natural near/far codec work is a new candidate under development; source/data audits are not a completed learned-method result.

## Updates

Updates are explicit snapshots or reviewed commits from the active project. Do not overwrite ongoing experiments from this archival checkout. This push changes only this independent repository and leaves the active research workspace untouched.
