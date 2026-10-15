# Fixed conditional-function pilot

All three heads completed 300 actual-decoder training steps from the same initialization schedule. The table averages both carrier seeds within each of the two validation sources, then weights sources equally.

| Method | PSNR dB | Private HF revision RMS | Nominal near LR RMSE |
|---|---:|---:|---:|
| mixed | 25.092372 | 0.01157503 | 0.03284477 |
| mixed_VAE_support_intervention | 25.086415 | 0.01162077 | 0.03287671 |
| additive | 25.229176 | 0.00968080 | 0.03217453 |
| additive_VAE_support_intervention | 25.228816 | 0.00953790 | 0.03215873 |
| joint_set_cnp | 25.212567 | 0.01015677 | 0.03225064 |
| joint_set_cnp_VAE_support_intervention | 25.213499 | 0.01008259 | 0.03219543 |
| source_carrier | 25.184179 | 0.00000000 | 0.03249808 |
| joint_robust_inverse | 26.865900 | 0.00035250 | 0.00558288 |
| dense_conditional_H3_pushforward_not_posterior | 26.020185 | 0.00472799 | 0.02001611 |
| dense_VAE_family_pushforward | 26.088606 | 0.00293025 | 0.01923879 |
| exact_finite_prior_joint_mean | 25.189450 | 0.00322079 | 0.03245372 |
| affine_VAE_family_at_LR_query | 24.274396 | 0.00644922 | 0.03356176 |

Every source and held-query error is retained in `sourcewise.csv`; mixed-minus-CNP paired results are in `mixed_minus_CNP.csv`. The larger private revision of a method describes a tradeoff and must be read alongside GT error, not called identity preservation.

Raw and sampling-inner DC teachers are separate experiments. The training only queries the three fixed support coordinates. The unseen (+,+) composition and center are evaluation labels; no held finite-difference supervision is used.

The exact finite-prior control represents its declared two far-only atoms and five operator coordinates. The H3/VAE conditional-family pushforwards are predictive readouts; scoring a cue-conditioned response is not exact posterior inference.

This development pilot cannot establish a new method, generalization, long memory, or useful H3 identity diversity. All computation and family/learning controls are recorded in `summary.json`.
