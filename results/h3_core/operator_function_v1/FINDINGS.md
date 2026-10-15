# Fixed conditional-function pilot

All three heads completed 300 actual-decoder training steps from the same initialization schedule. The table averages both carrier seeds within each of the two validation sources, then weights sources equally.

| Method | PSNR dB | Private HF revision RMS | Nominal near LR RMSE |
|---|---:|---:|---:|
| mixed | 25.103332 | 0.01184128 | 0.03292924 |
| mixed_VAE_support_intervention | 25.101716 | 0.01185468 | 0.03295153 |
| additive | 25.170631 | 0.00973997 | 0.03265356 |
| additive_VAE_support_intervention | 25.170485 | 0.00972650 | 0.03265646 |
| joint_set_cnp | 25.149107 | 0.01027992 | 0.03277867 |
| joint_set_cnp_VAE_support_intervention | 25.147625 | 0.01028344 | 0.03279739 |
| source_carrier | 25.224637 | 0.00000000 | 0.03237902 |
| joint_robust_inverse | 26.910039 | 0.00038592 | 0.00558527 |
| dense_conditional_H3_pushforward_not_posterior | 26.164163 | 0.00306541 | 0.01852846 |
| dense_VAE_family_pushforward | 26.207536 | 0.00236221 | 0.01794876 |
| exact_finite_prior_joint_mean | 25.231638 | 0.00240835 | 0.03233161 |
| affine_VAE_family_at_LR_query | 24.614192 | 0.00497152 | 0.03186529 |

Every source and held-query error is retained in `sourcewise.csv`; mixed-minus-CNP paired results are in `mixed_minus_CNP.csv`. The larger private revision of a method describes a tradeoff and must be read alongside GT error, not called identity preservation.

Raw and sampling-inner DC teachers are separate experiments. The training only queries the three fixed support coordinates. The unseen (+,+) composition and center are evaluation labels; no held finite-difference supervision is used.

The exact finite-prior control represents its declared two far-only atoms and five operator coordinates. The H3/VAE conditional-family pushforwards are predictive readouts; scoring a cue-conditioned response is not exact posterior inference.

This development pilot cannot establish a new method, generalization, long memory, or useful H3 identity diversity. All computation and family/learning controls are recorded in `summary.json`.
