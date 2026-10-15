# Matched raw / sampling-inner DC conditional-response representation pilot

Both versions completed all three 300-step heads from the same initialization/schedule, with true frozen H3 decoder gradients. These are four train / two val development sources. Every reported mean averages the two carrier seeds within each source, then sources equally. The experiment compresses a response family **after A+B have arrived**; it does not demonstrate a state compressed after A that is later revised by B.

| Method | Raw PSNR dB | InnerDC PSNR dB | Raw private GT HF RMSE | DC private GT HF RMSE |
|---|---:|---:|---:|---:|
| mixed | 25.103332 | 25.092372 | 0.04496272 | 0.04492385 |
| additive | 25.170631 | 25.229176 | 0.04407786 | 0.04381118 |
| joint_set_cnp | 25.149107 | 25.212567 | 0.04388683 | 0.04351375 |
| source_carrier | 25.224637 | 25.184179 | 0.04315453 | 0.04391785 |
| dense_conditional_H3_pushforward_not_posterior | 26.164163 | 26.020185 | 0.04330439 | 0.04384398 |
| dense_VAE_family_pushforward | 26.207536 | 26.088606 | 0.04298988 | 0.04382076 |
| joint_robust_inverse | 26.910039 | 26.865900 | 0.04315618 | 0.04391973 |

Mixed minus CNP is negative on both validation sources in both versions: raw −0.008039/−0.083513 dB; innerDC −0.077168/−0.163220 dB. All six-source pairs, support/held query errors, and full likelihood residuals remain in each version’s `sourcewise.csv`, `mixed_minus_CNP.csv`, and `summary.json`. The H3 response family has lower PSNR/SSIM than the corresponding VAE response family in both versions; see the perceptual tradeoff below before interpreting this as an all-metric ranking. This does not contradict the separate real-video eight-source samplingDC baseline: the operators, guides and evaluation protocol differ.

## What the fixed diagnostics do and do not establish

The four-train-source, three-support-coordinate PCA8 retains 91.665% raw / 91.884% DC H3 response energy. Same-state PCA support native MSE is about 0.01285/0.01321 on train; the trained mixed head has 0.06731/0.07382, and CNP 0.07000/0.08337. Thus the chosen optimization/readout is far from the fixed support-reconstruction control even on train; the negative result cannot be attributed solely to insufficient 4/8-channel capacity. RGB and private objectives deliberately differ from a pure native-response reconstruction target, so this is a diagnostic gap, not proof that more optimizer steps would fix the method.

Retaining all24 channels removes support reconstruction error (native MSE about2.3e−16), but three-support affine interpolation still has substantial real nonlinear center error: raw/DC val native MSE0.07090/0.07624 and teacher RGB RMSE0.03439/0.03671. Its actual center PSNR24.78540/24.54443 remains below actual H3 center26.16416/26.02019. The center weights are[0,.5,.5], the midpoint of support1/support2; (+,+) weights[−1,1,1] extrapolate outside the support triangle. No center/held output or GT is fitted. These results locate a nonlinear response interpolation limit as well as a learned fit/objective gap. No rank, gain or loss sweep follows.

The private query consists only of four predefined32×32 corner patches. Mixed has larger displacement from its own carrier and worse private GT HF error than CNP in both versions. This is an error/preservation tradeoff, not useful diversity or identity preservation. Raw CNP also worsens private GT HF relative to its carrier; DC CNP slightly improves this local GT metric while revising the carrier. Preserving a carrier is not synonymous with preserving correct information.

## Prior / early / late association audit

A concentrated final posterior is not itself a failed association test. The complete prefix audit is decisive here: the declared uniform2×5 prior has independent carrier/operator variables; after A alone, every source and both carriers already have operator-given-carrier probability1 at the center, conditional entropy0. Joint mutual information is0 up to1.1e−16. AB can change the carrier weights, e.g. raw train103041 [.881,.119]→[1,0], and both raw val sources switch their favored carrier u0→u1. It does not create or resolve carrier/operator dependence. The B camera is g-independent, so q(g|u) cannot change on B by construction. Together with the post-arrival compression timing, this means the pilot has not tested the proposed early association-retention/late geometry-disambiguation mechanism. It is not evidence disproving that mechanism. Full log densities and every prefix probability remain in `association_prefix_audit.json`.

## Budget and interface evidence

Raw48 new official49 BF16 H3 trajectories + independentDC72 + one49-step TRAIN identity guard =5929 full50-block DiT forwards. Raw reuses24 previously generated far/center trajectories; their earlier compute is not hidden as new work. Formal training is1800 optimizer steps (900 per version),3600 actualD backward calls and528 formal evaluation D calls; the two interface smokes add12 optimizer steps/44D. Including preparation, inside-sampler corrections, family assembly and both240D fixed diagnostics totals5135 top-level nativeD and459 nativeE calls for this matched stage. Counts are not multiplied by repeated static22-frame context. Formal train+eval wall time is206.791s raw and204.281s DC; fixed diagnostics are14.088s and14.087s. See exact costs in each summary.

SamplingDC uses the declared reflect-Gaussian.6/AA bicubic camera, not a replacement video degradation. Independent camera maxabs6.66e−16; feasible P25 fixed point0. TRAIN identity-hook official49 video/audio parity to raw is exactly0, all near guide bytes unchanged, all frozen base weights unchanged. The static native22 context is one already observed far packet, not22 independent measurements or future frames.

PCA coefficient state is24×7×16×16 FP32=172032B per carrier plus the common full24 native carrier172032B; its shared24×8 basis is768B per family, with no mean. Full24 affine coefficients use72 channels=516096B per carrier plus the common carrier, and are a larger-memory diagnostic. Both seeds are stored; q(g|u) adds40B per carrier, in addition to shared operators/observations/model weights and runtime scratch. This is not a compressed-state advantage over raw LR archival.

All weights, readouts, CSV and code snapshots are preserved. Neither version is selected from these two val sources. No old test or sealed DIV2K source is accessed by this stage.

Qualitative panels `val_101085_raw_DC_seed101.png` and `val_101087_raw_DC_seed101.png` display the actual seed101 tensors without averaging images across seeds. Both were visually inspected: the robust inverse retains visibly more local texture/shape than the learned readouts; the native H3 family has some local recovery but the mixed/CNP outputs stay close to the blurred carrier. This is an observation about these two panels, not a general perceptual ranking.

## Independent perceptual evaluation and sourcewise tradeoff

The root independently evaluated96readouts with pretrained AlexNet LPIPS (inputs explicitlyFP32; CUDA matmul/cuDNN TF32 bothoff). The exact metadata, input/model hashes and all source rows are in `../operator_perceptual/evaluation.json`. These metrics change the interpretation of the PSNR-only negative result: mixed has a perceptual/distortion tradeoff, rather than being uniformly worse.

| Method | Raw SSIM | Raw LPIPS↓ | DC SSIM | DC LPIPS↓ |
|---|---:|---:|---:|---:|
|mixed|0.63275471|0.27405355|0.63077885|0.27654629|
|additive|0.63633877|0.28958105|0.63890786|0.29491108|
|joint_set_cnp|0.63336097|0.27893974|0.63749915|0.29378749|
|dense_conditional_H3_pushforward_not_posterior|0.70820980|0.23855655|0.69911340|0.23592880|
|dense_VAE_family_pushforward|0.71456957|0.23688470|0.70680902|0.23748434|
|joint_robust_inverse|0.75885370|0.22546288|0.75710198|0.22370692|

Raw mixed−CNP LPIPS per source is−0.01051500 and+0.00074263: the mean perceptual improvement reverses on one source. DC mixed−CNP is−0.01191965 and−0.02256274, improving both sources despite lower PSNR/mean SSIM. Mixed private GT HF error remains worse than CNP; improved LPIPS does not imply better private information preservation. Both dense actualH3 and strong robust inverse still have substantially lower LPIPS than every learned head. DC denseH3 has slightly better LPIPS than DC VAE(.23592880 versus.23748434), despite lower PSNR/SSIM; raw H3 LPIPS is slightly worse than raw VAE(.23855655 versus.23688470). This does not establish a causal memory or joint-association contribution, given the separately audited prefix limitation.

The fixed next2×2 experiment is unchanged; no version, gain or checkpoint was selected from this added perceptual evaluation.
