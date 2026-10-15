# Streaming SR with MiniMax-H3: Research Snapshot

Private research snapshot captured on **2026-10-15T22:19:40+08:00**. This repository contains implementations, experiment protocols, numerical results and positive/negative findings. **A validated novel streaming-SR system has not yet been established.**

- [Experiment report (Chinese)](EXPERIMENT_REPORT.md) — measured outcomes and limitations.
- [Snapshot scope](SNAPSHOT.md) and [source SHA256 manifest](SNAPSHOT_MANIFEST.json).
- [Eight-source H3 sampling-consistency evaluation](results/h3_setup/sampling_dc_davis8/REPORT.md).
- [B5 four-arm structural experiment](results/h3_factorized/factorial_v5/B5_final_report.md).
- [Delayed LR validation](results/real_video_history_inverse/delayed_lr_validation/REPORT.md).
- [RelayVSR baseline](results/relay_baseline/cohort/REPORT.md).

H3 has positive restoration evidence, but the learned historical correction has not shown an independent advantage. New codec/data/protocol files are work in progress unless their own reports explicitly record completed results. This archive is updated through explicit snapshots; active experiment files remain in the original workspace.

---

# Streaming SR: Revisable Detail State

Started 2026-10-15 (Asia/Shanghai). This is an active research project, not a completed paper or verified reproduction.

## Objective
Establish an original, stable core method for streaming video SR: persistent detail identity that can be revised by arriving observations. Explore routes 1–3 from research/source_brief.md, adapt the direction when evidence supports it, and produce an experiment report with progress, positive/negative outcomes and follow-up decisions. The target is a research contribution and experimental scope comparable to a CCF-A submission; that is not yet established.

## User model preference
The user requested MiniMax-H3 as the preferred foundation model. Primary architecture and training must use H3 wherever feasible; other video SR models remain external baselines. Existing controlled studies are diagnostic evidence, not the final H3 method.

## Current method hypothesis
Separate evidence from a persistent generated explanation. Keep observation information G and b separate from the generated detail proposal z. New observations contract the ambiguous subspace; preserve detail identity where the prefix still supplies no evidence, and revise it where genuinely new measurements resolve ambiguity. Plain Kalman filtering, multi-frame inverse SR, and nullspace projection are existing baselines, not claimed inventions. We will test whether a learned, operator-conditioned, bounded-memory implementation adds a stable new capability beyond these baselines.

## Evidence levels
1. Controlled known-operator mechanism experiments: identify the potential and failure modes.
2. Trained small-model experiments on held-out textures and natural patches: test learned update and generalization.
3. Real streaming video with public SR backbones: test perceptual quality, retrieval, occlusion, appearance change, waiting/compute latency and memory.
4. Cross-scale queries and finite-wait temporal interpolation: test extensions after or alongside a viable core.
A result at one level does not prove the following levels. No performance numbers are author-reported as our own.

## Project layout
- research/source_brief.md: original user-provided research brief (copied without modification).
- research/sources_and_method.md: source verification and closest prior art.
- research/resources.md: available compute, dependencies and constraints (no credentials).
- experiments/mechanism/: controlled implementations and commands.
- src/: learned model and reusable interfaces.
- results/: machine-readable outputs, figures, checkpoints, logs.
- EXPERIMENT_REPORT.md: evolving experiment report (written only from verified runs).

## Research decisions
We compare strong history-based inverse reconstruction, not only deliberately weak temporal averaging. Split data by source image/sequence, train seeds independently, and never use evaluation HR at state update time. Preserve failures and give each negative mechanism result a remedy, architecture change, or actionable condition to test. Avoid spending the project on hyperparameter tuning.
