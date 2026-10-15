# Fixed actual-H3 conditional-response pilot: completed negative result

The loop-trained connection does not establish a correction/private-preservation advantage over an ordinary endpoint head, GRU, or an order-invariant evidence set. All four predeclared heads completed 300 steps, including when the raw conditional headroom looked weak. No gains, sources, guides, or budgets were changed afterward. This is four BSDS training sources and two validation sources, seed17, development evidence only.

Every head has 18,888 trainable parameters and the same actual H3 response cache, initial X0, immutable channel-PCA8 coordinates, latent private complement, reconstruction/observed-LR/private-query losses, and far/box data consistency. Only the proposed head adds the fixed loop loss. The endpoint control is SFM-style; it is not a full reproduction of the cited sequential flow matching training algorithm.

Final validation means give AB and BA equal weight within each source, then give both sources equal weight. Private revision is high-frequency RGB RMS relative to that source's own X0; lower means more preservation, not necessarily better reconstruction. The global query covers 50% of the frame outside both nominal near rectangles and the four private corners. These are predeclared operational queries, not identified semantic factors or exact camera-support nulls.

| Method | Full PSNR | Global-query PSNR | Observed LR RMSE | Private revision RMS | AB/BA RGB MSE |
|---|---:|---:|---:|---:|---:|
| X0 | 25.42817 | 25.36763 | .031429 | 0 | 0 |
| Loop connection | 25.41770 | 25.38085 | .031623 | .006572 | 1.0349e-5 |
| Basic sequential | 25.42145 | 25.38444 | .031589 | .006486 | 1.2054e-5 |
| GRU | 25.41928 | 25.36952 | .031534 | .006370 | 5.0255e-6 |
| DeepSets | 25.44480 | 25.38816 | .031300 | .004225 | **0 exactly** |
| Same-U H3 response mean | 25.48845 | 25.36594 | .030370 | .003822 | 0 by subset readout |
| Same-U VAE-guide response | 25.48385 | 25.36512 | .030426 | .003772 | 0 by subset readout |
| Classical joint LR map | **27.09748** | 25.41857 | **.005201** | **.000412** | 0 by joint solve |
| Full24 H3 response transplant | 26.03330 | 25.39666 | .022736 | .002992 | 0 by subset readout |
| Full24 VAE-guide response | 26.06502 | 25.39801 | .022424 | .002897 | 0 by subset readout |
| Direct actual H3 current output mean | 26.24232 | 25.22570 | .017867 | .008763 | 0 by subset readout |
| Direct VAE current guide | 26.86790 | **25.45473** | .011523 | .005983 | 0 by subset readout |

The proposed loop reduces the basic head's path MSE by 14.15%, but has slightly worse reconstruction and private preservation. GRU has lower path discrepancy, and the ordinary set control has strict floating-point zero on all six sources because it recomputes from the same canonical evidence set. Neither a repeat no-op nor a small loop improvement is an independent new capability.

The fixed PCA8/private split is itself a bottleneck. It retains approximately 41–61% of actual AB response energy. Withdrawing it, without retraining, gain changes, or extra DiT queries, improves full-frame PSNR by .54485dB for the H3 response and also reduces private-query RGB revision. Freezing a latent complement therefore does not establish visual identity protection. The full24 VAE-guide response is slightly stronger than the full24 H3 response. Directly decoding the actual current H3 samples also remains weaker than decoding their VAE guide; the negative result is not explained solely by a poor response transplant.

The classic joint map remains the strongest full-frame/LR control. It has more high-frequency energy (.03944 RMS) than the loop head (.02805), so its gain is not explained by suppressing texture to hide drift. All raw gradient/HF measurements are retained in `metrics.csv`; independent SSIM/LPIPS evaluation can use the saved RGB tensors. The comparison panels were visually inspected; the classical near update visibly restores more local structure than the trained response heads.

Independent full-RGB perceptual evaluation is complete at [connection_perceptual/evaluation.json](../connection_perceptual/evaluation.json), using the same saved float/DC outputs, Alex LPIPS v0.1, explicitly disabled CUDA matmul/cuDNN TF32, and source-equal weighting after AB/BA averaging. LPIPS: loop .33936487, basic .33933323, GRU .34036219, set .34262832, H3 response .33768883, X0 .34949946, classical .24684156. SSIM: loop .647888, basic .648102, set .650434, classical .766094. Loop has a small average LPIPS advantage over set but the first source reverses that ordering; loop/basic differences also have opposite signs across sources. This does not establish a stable perceptual contribution, and the classical control remains substantially stronger. The full per-source values are copied into `summary.json`; no model was selected using these scores.

The exact finite-prior Bayes/MHT/fixed-source/OT control is a separate, narrowly defined calculation on the two genuine far-only H3 initialization atoms for each val source. Given the specified Gaussian likelihood, both final weights are [1,0], and the controls return X0: full25.42817/global25.36763, zero private change and path discrepancy. This two-atom support cannot create missing detail. Current conditional C_S samples already used the cue; their residual rescoring is explicitly a heuristic, not calibrated posterior weighting. The same-U heuristic mean/min-LR/fixed-source/OT values are all preserved rather than selected afterward.

Per-source paired loop comparisons are in `paired_differences.csv`; every prefix, every trained head, and every executed strong control is in `metrics.csv`. Both train and val rows are retained. The two-source development mean is not assigned a confidence interval or treated as broad population evidence. Initial same-LR natural H3 pair RMS is only .00653/.00375 on the two val sources. No manufactured noise/texture is counted as natural H3 diversity.

The declared camera operator is the same for generation and inference: reflect-Gaussian .6, bilinear camera warp, antialiased bicubic downsampling, independent .003 noise and 8-bit quantization; the fixed effective likelihood sigma is .005. All input payloads are LR/geometry/packet IDs. HR appears only in separately saved training supervision and posthoc metrics. Native22 repeats an already observed canonical query as decoder context; it is not 22 independent observations, real motion, or access to future frames. All proposed and control prefix/duplicate checks have maxabs0. The nominal private/global regions are geometric query definitions; interpolation and filtering can extend actual observation support near their boundaries.

The actual H3 cache contains 56 official full-BF16 native22 samples, each 49 Euler calls through all 50 blocks: 2,744 DiT forward calls, zero DiT backward calls. The initial8 and conditional48 sampling batches took 51.3406s and 236.8976s, with their model loads; preparation and readout time are separate. All native sampler records have 49 finite video/audio events. The training head only differentiates the actual frozen H3 VAE decoder. Official conditional sampling does not inherit the earlier five-step raw-native FP32/BF16 IO discrepancy.

| Stage | Top-level actual VAE decode invocations |
|---|---:|
| Formal four heads ×300 steps ×4 endpoints, forward/backward | 4,800 |
| Formal six-source ×4-head ×4-endpoint evaluation | 96 |
| Eight smoke optimizer steps and its six-source evaluation | 128 |
| Initial native-output readout | 8 |
| Conditional response audit | 48 |
| First strong-control readouts | 26 |
| Repeated controls plus full24 structural diagnosis | 74 |
| Independent validation RGB export | 32 |
| Direct-native readout diagnosis | 24 |
| **Total, including all diagnostics** | **5,236** |

There are also 30 native encode calls (six far guides and 24 subset guides), 720 nonempty-subset CG iterations, 1,200 formal optimizer steps and eight excluded smoke steps. The formal train/eval run took 462.8019s, with 26,793,027,072 peak allocated bytes and 29,578,231,808 peak reserved bytes. Counts refer to top-level VAE encode/decode invocations reconstructed from the fully executed loops, not internal decoder-layer hooks. Extra diagnostic costs are disclosed rather than attributed to the fair 300-step head comparison.

Artifacts: `summary.json`, `metrics.csv`, `paired_differences.csv`, `fixed_seed17/{results.json,checkpoint.pt}`, `strong_controls.json`, `direct_readout_controls.json`, `response_audit.json`, and each `val_*/{eval_readouts.pt,comparison_panel.png}`. Code and sampler SHA manifests are pinned under `code_snapshot`. All early BSDS val/test sources had already been evaluated by another project stage; these sources are development data. No sealed DIV2K confirmation data was read or evaluated. There is no reason from these results to extend this loop head to32/8 or sweep its loss/gain.
