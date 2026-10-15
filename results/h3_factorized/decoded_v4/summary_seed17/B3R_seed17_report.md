# B3R decoded-RGB v4: fixed last672, seven controls

32train/8val development source photos; one trainedseed17. Seven fixed672-update controls, every update actual frozen H3D, balanced224(source,query)×3, no latent-L1 main loss, no val selection. All initialize stage1last600 (anchor-only copiesfull); no old16test or sealedDIV2K. Source bootstrap averages all queries within each source first.

Current-image neural control never reads historical images/code; its shared physical reliability filter remains a historical-controller dependency. Geometry-only filter sensitivity is reported separately. Algebraic closure is not itself originality.

This report evaluates **last.pt**, from `val_seed17`. Exact checkpoint step/hash/initial path: evaluation `complete.json`.

|Condition/region|Control|full minus control dB|95% source bootstrap CI|Wins/8|
|---|---|---:|---|---:|
|all/union|current_LR_VAE|+0.2717|[+0.1258, +0.4807]|8|
|all/union|closed_action_no_equiv|+0.0420|[+0.0167, +0.0711]|8|
|all/union|factor_cache_mlp|-0.0447|[-0.1358, +0.0090]|3|
|all/union|nonlinear_group|-0.0071|[-0.0538, +0.0258]|5|
|all/union|gru_online|+0.0590|[+0.0388, +0.0827]|8|
|all/union|gru_replay|+0.0319|[+0.0120, +0.0568]|7|
|all/union|LR_photo_revision_atlas|-3.3996|[-3.8000, -3.1161]|0|
|all/union|LR_photo_revision_inverse|-1.6356|[-1.8730, -1.3879]|0|
|all/union|H3_ED_revision_inverse|-2.3985|[-2.6644, -2.2022]|0|
|all/union|anchor_only|+0.0082|[-0.0182, +0.0466]|3|
|neutral_return/union|current_LR_VAE|+0.2939|[+0.1305, +0.5353]|8|
|neutral_return/union|closed_action_no_equiv|+0.0373|[-0.0051, +0.0765]|6|
|neutral_return/union|factor_cache_mlp|-0.0753|[-0.2308, +0.0134]|3|
|neutral_return/union|nonlinear_group|-0.0205|[-0.0873, +0.0207]|5|
|neutral_return/union|gru_online|+0.0617|[+0.0370, +0.0893]|8|
|neutral_return/union|gru_replay|+0.0246|[+0.0062, +0.0416]|7|
|neutral_return/union|LR_photo_revision_atlas|-3.3919|[-3.7851, -3.1140]|0|
|neutral_return/union|LR_photo_revision_inverse|-2.2190|[-2.4765, -1.9221]|0|
|neutral_return/union|H3_ED_revision_inverse|-2.4878|[-2.7848, -2.2761]|0|
|neutral_return/union|anchor_only|+0.0124|[-0.0107, +0.0413]|4|
|appearance_cue/union|current_LR_VAE|+0.2884|[+0.1225, +0.5381]|8|
|appearance_cue/union|closed_action_no_equiv|+0.0490|[+0.0223, +0.0768]|8|
|appearance_cue/union|factor_cache_mlp|-0.0551|[-0.1509, +0.0054]|3|
|appearance_cue/union|nonlinear_group|-0.0252|[-0.0852, +0.0141]|3|
|appearance_cue/union|gru_online|+0.0682|[+0.0343, +0.1131]|8|
|appearance_cue/union|gru_replay|+0.0440|[+0.0118, +0.0912]|7|
|appearance_cue/union|LR_photo_revision_atlas|-3.4271|[-3.8389, -3.1395]|0|
|appearance_cue/union|LR_photo_revision_inverse|-1.9026|[-2.1155, -1.6446]|0|
|appearance_cue/union|H3_ED_revision_inverse|-2.4602|[-2.7585, -2.2393]|0|
|appearance_cue/union|anchor_only|+0.0080|[-0.0238, +0.0574]|2|
|true_content_change/union|current_LR_VAE|+0.2245|[+0.1210, +0.3471]|7|
|true_content_change/union|closed_action_no_equiv|+0.0363|[-0.0031, +0.0790]|5|
|true_content_change/union|factor_cache_mlp|+0.0017|[-0.0255, +0.0263]|4|
|true_content_change/union|nonlinear_group|+0.0335|[+0.0039, +0.0637]|5|
|true_content_change/union|gru_online|+0.0424|[+0.0112, +0.0745]|6|
|true_content_change/union|gru_replay|+0.0212|[+0.0011, +0.0421]|6|
|true_content_change/union|LR_photo_revision_atlas|-3.3659|[-3.7471, -3.0598]|0|
|true_content_change/union|LR_photo_revision_inverse|-0.6518|[-1.0655, -0.1913]|2|
|true_content_change/union|H3_ED_revision_inverse|-2.2166|[-2.4063, -2.0512]|0|
|true_content_change/union|anchor_only|+0.0045|[-0.0220, +0.0377]|4|

Fixed-checkpoint mechanism interventions (no retraining):

|Condition|Intervention|full minus intervention union dB|CI|Wins/8|
|---|---|---:|---|---:|
|all|closed_action_frozen_photo|-0.0000|[-0.0001, -0.0000]|2|
|all|closed_action_frozen_weights|+0.0004|[-0.0001, +0.0010]|5|
|all|closed_action_frozen_both|+0.0004|[-0.0001, +0.0010]|5|
|all|diagnostic_closed_no_measured_detail|+0.0011|[-0.0021, +0.0049]|5|
|all|diagnostic_closed_shuffled_history|+0.0005|[-0.0008, +0.0018]|7|
|appearance_cue|closed_action_frozen_photo|-0.0001|[-0.0001, -0.0000]|1|
|appearance_cue|closed_action_frozen_weights|+0.0010|[-0.0002, +0.0033]|2|
|appearance_cue|closed_action_frozen_both|+0.0010|[-0.0002, +0.0033]|2|
|appearance_cue|diagnostic_closed_no_measured_detail|+0.0014|[-0.0022, +0.0056]|5|
|appearance_cue|diagnostic_closed_shuffled_history|+0.0006|[-0.0008, +0.0022]|5|
|neutral_return|closed_action_frozen_photo|-0.0000|[-0.0001, +0.0000]|1|
|neutral_return|closed_action_frozen_weights|-0.0003|[-0.0009, +0.0001]|2|
|neutral_return|closed_action_frozen_both|-0.0003|[-0.0009, +0.0001]|3|
|neutral_return|diagnostic_closed_no_measured_detail|+0.0020|[-0.0019, +0.0069]|5|
|neutral_return|diagnostic_closed_shuffled_history|+0.0005|[-0.0009, +0.0018]|4|
|true_content_change|closed_action_frozen_photo|+0.0000|[-0.0001, +0.0001]|4|
|true_content_change|closed_action_frozen_weights|+0.0003|[-0.0012, +0.0017]|4|
|true_content_change|closed_action_frozen_both|+0.0003|[-0.0012, +0.0017]|4|
|true_content_change|diagnostic_closed_no_measured_detail|-0.0001|[-0.0023, +0.0020]|5|
|true_content_change|diagnostic_closed_shuffled_history|+0.0005|[-0.0012, +0.0020]|5|

Raw/DC, per-source and regional data are retained in `method_summary.csv`, `source_aggregates.csv` and `paired_differences.csv`. The `changed` region is evaluated only after actual replacement queries13/17; a pre-confirmation branch claim is not made.

The independent streaming implementation stores compressed h rather than native delta/proxy codes. It is audited against the vectorized rollout; all raw factor h/weight/ID/descriptor and group buffers are included. Appearance-action-only cost may be constant in N for fixed group count, while the actual whole-state update still scans packet IDs/weights, and LR reliability/controller work can scale with N. No O(1) whole pipeline claim.

Shared current LR49152B + native anchor172032B + decoded RGB786432B =1,007,616B of current-query payload, reference LR49152B additionally persistent, and shared camera524288B. Frozen H3 weights are shared and separately reported. Compressed factor h is28672B per packet plus weights1024B,8x8RGB descriptor768B and assumed24B ID/group/geometry. The raw-LR inverse archive is49152B per packet plus controller/metadata; this is a stronger storage comparison than contrasting4 channels with a dense RGB atlas. Python allocator overhead, activations and CUDA peak are separate.

The classical inverse models relative spatial gain+bias before the exact observation operator; observed RGB clipping makes this affine model approximate. Native decoded-atlas photometric normalization is an empirical control, not exact latent equivariance. All methods use the same causal confidence with explicitly different resolution resampling and the same64step current-LR DC. Generated outputs never enter the measurement ledger.

Interpretation must be filled from the fixed results, including whether full beats the same-compression factor-cache/replay-GRU and preserves newly acquired detail beyond current-anchor priors. Algebraic closure alone does not establish originality.
