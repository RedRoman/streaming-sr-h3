# Learned core: fixed-budget evidence

Protocol: grayscale 4× known canonical point footprints, BSDS200 train / 100 validation / 200 test, one 48×48 held-out patch per source image. This is a mechanism experiment, not general video SR.

Three training seeds ['17', '29', '43'], six models, 18000 actual optimizer steps; total training time 351.4s on one H100. Each run uses 1000 steps × batch 24; the Gaussian prior sees training images only.

Test final PSNR, mean ± standard deviation over training seeds:

| condition | Gaussian MAP | stateless learned MAP residual | persistent learned MAP residual | raw GRU |
|---|---:|---:|---:|---:|
| test/repeat | 23.369 ± 0.000 | 24.506 ± 0.047 | 24.726 ± 0.021 | 22.738 ± 0.684 |
| test/mixed_repeat | 27.783 ± 0.000 | 28.201 ± 0.071 | 28.303 ± 0.017 | 23.915 ± 0.846 |
| test/new | 31.520 ± 0.000 | 31.160 ± 0.257 | 31.246 ± 0.032 | 24.910 ± 1.094 |
| test/reveal | 40.336 ± 0.000 | 40.336 ± 0.000 | 40.336 ± 0.000 | 24.753 ± 1.199 |
| test/wrong | 28.442 ± 0.000 | 28.345 ± 0.168 | 28.380 ± 0.014 | 24.489 ± 1.004 |

Interpretation:

- Learned non-Gaussian detail improves strongly under one or four observed phases. This reproduces in all three seeds.
- At eight observed phases, the fitted all-history Gaussian MAP has higher PSNR. No superiority claim over this strong baseline is supported there.
- Persistent learned residual is more accurate than stateless residual in some conditions but its independent benefit is not stable across seeds.
- Exact packet duplicates must not increase precision. Independent noisy packets at the same footprint must increase precision; the extended evaluation passes explicit fresh-packet masks.
- All-history ledger and nullspace projection retain a complete reveal and prevent future LR-only frames from rewriting observed detail. They do not solve correspondence errors.
- Main texture_ood labels are corrected to synthetic ID; extended checker_ood uses discontinuous multiplicative patterns absent from training.

Actionable direction for a large H3 adapter: preserve a persistent detail state in weak-evidence directions, project new detail away from constrained directions, and estimate residual risk before injecting it when evidence is sufficient. A validation-only risk control is evaluated separately and is not presented as a new trained architecture.

Additional fixed-checkpoint tests (no extra training):

| condition | Gaussian MAP | learned transport | stateless residual |
|---|---:|---:|---:|
| independent_noise_repeat | 23.453 ± 0.000 | 24.796 ± 0.026 | 24.408 ± 0.079 |
| high_noise_new | 26.957 ± 0.000 | 27.083 ± 0.007 | 27.052 ± 0.070 |
| checker_ood | 13.027 ± 0.000 | 13.309 ± 0.003 | 13.310 ± 0.007 |
| reveal_then_repeat | 40.362 ± 0.000 | 40.362 ± 0.000 | 40.362 ± 0.000 |

Removing the observed-direction projection after a complete reveal reduces PSNR from 40.362 to 34.039 dB. Freezing the prior after the first frame also hurts novel-evidence uptake. These are mechanism ablations of the fixed model, not independently optimized alternative architectures.

The raw ledger CNN fails the unseen precision-count regime (independent same-footprint noise packets). A precision scalar presented out of its training support is insufficient. The analytic posterior plus persistent residual remains usable; this motivates an actual spectral information transition for H3, rather than merely feeding counts to a CNN.

Exploratory risk control:

Use R(a)-R(0)=2a E[e·d]+a²E[d²], with e the MAP error and d the proposed residual, estimated on validation images only. Apply a*=clip(-E[e·d]/E[d²],0,1). No test target enters this coefficient.

For a fresh held-out acquisition schedule at eight phases, raw transport is 31.272 dB, Gaussian MAP 31.558 dB, and the validation-calibrated control 31.658 dB. This follows the initial experiment and uses the same test source images, so it is an exploratory diagnostic rather than a preregistered confirmation. The H3 core must learn an evidence-conditioned transition instead of treating this scalar postprocessing as the final method.
