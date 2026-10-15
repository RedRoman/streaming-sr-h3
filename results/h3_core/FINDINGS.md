# H3 canonical-memory mechanism experiments

The current local-chart learned transport does **not** beat the classical all-history MAP control. Three fixed-budget seeds support that negative conclusion. This is a frozen MiniMax-H3 VAE mechanism study; native H3 DiT restoration effectiveness is a separate experiment.

The actual FP16 H3 VAE checkpoint was used with native 256×256×22 context, normalized latent shape 1×24×7×16×16, and a central 4×4-token spatial chart. Only one already observed LR anchor is repeated to supply native temporal context. Repeating that frame does not write additional observations. Camera and visibility are oracle inputs for these controlled natural-image camera sequences. Thirty-nine frames are a short 1.6-second diagnostic, not a long-revisit experiment.

A fixed rank32 span was fitted from actual E(HR)−E(upLR) residuals on the 200 official BSDS training sources only. It covers 96.39% of those training-bank residuals and 91.57% of the separate test-cache residuals. A conditional rank8 chart is pretrained and then frozen. Four heads see the same chart and, except the current-only control, the same all-history LR ledger: transport 11,016 parameters, stateless 11,016, GRU 10,984, current-only 11,016. All methods use the same unit Gaussian identity draw and classical prior precision I.

Each seed completed 600 chart/prior steps, 400 latent steps per head, and 100 genuine frozen-decoder RGB/LR-witness steps per head: 2,600 optimizer steps per seed, 7,800 for seeds 17/29/43. Training-head sources number 32, validation 8, independent test 16. The span/prior bank may use all 200 train sources; no validation/test source fits it. All 24 validation/test sources are actually decoded at prefixes 0/17/38. Metrics include full frame, a predeclared 64×64 central RGB ROI, a 90%-decoder-response support mask chosen without HR, gradient error, and LR reprojection. Code loss alone is not treated as restoration accuracy.

Three-seed test means at the final prefix:

| Method | Full-frame PSNR | Nominal ROI PSNR | Decoder-support PSNR | LR-witness RMSE |
|---|---:|---:|---:|---:|
| Anchor VAE only | 26.01957 | 25.96836 | 24.60884 | .025438 |
| All-history MAP | 26.03630 | 26.05884 | 24.69431 | .025264 |
| Shared seed | 26.03630 | 26.05879 | 24.69426 | .025264 |
| Bures coupling | 26.03632 | 26.05876 | 24.69426 | .025264 |
| Stateless full ledger | 26.02590 | 25.91830 | 24.56395 | .025389 |
| Learned transport | 26.02527 | 25.91063 | 24.55710 | .025410 |
| GRU full ledger | 26.02313 | 25.89852 | 24.54133 | .025422 |
| Current only | 26.01971 | 25.84176 | 24.50730 | .025462 |

Transport minus MAP is −0.14821 dB on the nominal ROI, image-bootstrap 95% CI [−0.20915, −0.08944], with seed averages taken before resampling the 16 independent sources. Differences against stateless (−0.00766) and GRU (+0.01212) have intervals crossing zero. Smaller state changes are not counted as a quality win. The fixed-PCA seed17 structural control also completed: MAP26.08690, stateless26.06656, transport26.04076, GRU26.02392 on the test ROI. It beats the learned-C variant at the same seed, so conditional-basis superiority is also unsupported.

Two implementation findings matter for interpreting these results. First, default cuDNN TF32 corrupted finite differences of distributed PCA directions despite FP32 decoder weights. Disabling TF32 reduced epsilon .05/.1 degraded-LR derivative disagreement from 46.4% to 0.242% on the audit source. Across all 56 strict caches the LR half-step discrepancy is 0.2206% mean, 0.3389% maximum. Native and scoped official eager strict math agree exactly in that audit. Second, the original Bures control used an inappropriate absolute square-root floor for tiny covariance products; that implementation bug was corrected and the original saved pilot checkpoint reevaluated without retraining. Its large old drift is not an algorithmic failure.

The rank32 local span is already fully observable from the first LR frame under its declared local linear model (test median smallest Gram eigenvalue 9.78; final 772 at effective sigma .005). It is therefore inappropriate to assume a latent nullspace inside this small span. Moreover, transplanting a local HR-encoder difference into an otherwise LR-context latent is not an RGB upper bound: the other tokens remain LR and decoder context is coupled. Its test ROI PSNR is much worse. This is an explicit HR-only transfer diagnostic, never an inference input.

A genuine native memory-guide export also ran: blocks [0,22) and [17,39), first emits [0,22), second [22,39); every guide variant has one clip and 448 native condition tokens. State reads stop at 21/38 and maximum input wait is 21 frames plus compute. Unknown/occluded regions are filled from current LR. These exports used pipeline weights and prove integration only; they are not a successful H3 DiT method.

The next structural audit uses the classical observation quotient x=A0^+ybar+(I−A0^+A0)D(z). It removes initial-footprint LR drift before defining detail evidence, so a repeated identical footprint cannot masquerade as new fine-detail information. Independent repeated packets may separately reduce coarse noise. The projection is a classical control, not claimed novelty. A validation-only audit will measure retained response energy, new-footprint precision, actual decoded RGB capacity, and remaining physical residuals before choosing the next learned architecture.

Artifacts: `nonlinear_fixed_suite/{summary.json,metrics.csv,learned_seed*/results.json,checkpoint.pt,run.log}`, `pca_span/precision_audit.json`, `cache_pca32_strict/audit_summary.json`, `adapter_pipeline_native22_eval_v2`, and `LIVE_STATE.json`. Full caches and active jobs are on H100; completed compact reports/checkpoints are copied locally. Joint RGB chart/transport learning, long revisit, real-motion estimated correspondence, and equal-DC H3 DiT quality comparisons remain incomplete.


The projected nonlinear follow-up completed all three seeds; see projected_nonlinear_v1/FINDINGS.md. Its transport-minus-projected-MAP validation difference is -.00693dB with an interval crossing0. Classical projection/history explains the gain. Incorrect rebased-difference current-only results are excluded and being repaired at the same valid budget. Actual nonlinear LR inversion remains a pending strong control, not a completed benefit.

Projected A2 three-seed current-only repair completed. Final validationROI transport24.81474 vsMAP24.82167/currentfine24.60394; independent learnedtransport benefit remains unsupported. Fixed40-step actualD LR-only inverse improves linearMAP +.27875dB ROI on8val, with40backward/42totalD perimage and finite audited gradients; see nonlinear_lr_inverse_val/results.json. Next actualBF16H3 proposals are native22, samefirstLR seeds101/202/303, not VAEmean priors; learned nativeconditioning updates remain untrained.
