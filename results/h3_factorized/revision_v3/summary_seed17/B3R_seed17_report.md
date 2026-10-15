# B3R fixed seed17 development pilot

32 train / 8 val natural source photos; synthetic known camera, nuisance-group tags, photometric changes, transient foreground and true local replacement. These BSDS sources were used earlier in the project. No old16test or sealed DIV2K evaluation. One seed; source, not query/packet/epoch, is the bootstrap unit.

All six controls were trained for600 latent+200 actual-native-H3-D steps. The original protocol specified val-selected best checkpoints; parent subsequently required a separate fixed-endpoint comparison, reinitializing all six RGB runs from last600 and evaluating last200. Original selection-pilot and extra physical compute are retained separately. Affine actions, sum statistics, factor caching and replay are standard ingredients; this remains a candidate mechanism experiment.

This report evaluates **best.pt**, from `val_seed17`. Exact checkpoint step/hash/initial path: evaluation `complete.json`.

|Condition/region|Control|full minus control dB|95% source bootstrap CI|Wins/8|
|---|---|---:|---|---:|
|all/union|current_LR_VAE|-0.1727|[-0.2262, -0.1234]|0|
|all/union|closed_action_no_equiv|+0.0463|[+0.0179, +0.0751]|6|
|all/union|factor_cache_mlp|-0.0195|[-0.0313, -0.0090]|1|
|all/union|nonlinear_group|-0.0071|[-0.0222, +0.0078]|4|
|all/union|gru_online|+0.0163|[-0.0047, +0.0399]|6|
|all/union|gru_replay|+0.0018|[-0.0174, +0.0220]|4|
|all/union|LR_photo_revision_atlas|-3.8440|[-4.2574, -3.5070]|0|
|all/union|LR_photo_revision_inverse|-2.0800|[-2.2404, -1.9231]|0|
|all/union|H3_ED_revision_inverse|-2.8429|[-3.1457, -2.5855]|0|
|neutral_return/union|current_LR_VAE|-0.1717|[-0.2261, -0.1241]|0|
|neutral_return/union|closed_action_no_equiv|+0.0497|[+0.0167, +0.0801]|6|
|neutral_return/union|factor_cache_mlp|-0.0210|[-0.0398, -0.0068]|1|
|neutral_return/union|nonlinear_group|-0.0110|[-0.0341, +0.0094]|4|
|neutral_return/union|gru_online|+0.0167|[-0.0030, +0.0361]|7|
|neutral_return/union|gru_replay|-0.0094|[-0.0314, +0.0115]|3|
|neutral_return/union|LR_photo_revision_atlas|-3.8575|[-4.2600, -3.5296]|0|
|neutral_return/union|LR_photo_revision_inverse|-2.6846|[-2.8571, -2.5266]|0|
|neutral_return/union|H3_ED_revision_inverse|-2.9534|[-3.2851, -2.6782]|0|
|appearance_cue/union|current_LR_VAE|-0.1731|[-0.2254, -0.1274]|0|
|appearance_cue/union|closed_action_no_equiv|+0.0537|[+0.0231, +0.0865]|7|
|appearance_cue/union|factor_cache_mlp|-0.0186|[-0.0284, -0.0102]|0|
|appearance_cue/union|nonlinear_group|-0.0066|[-0.0207, +0.0070]|3|
|appearance_cue/union|gru_online|+0.0081|[-0.0210, +0.0411]|4|
|appearance_cue/union|gru_replay|-0.0033|[-0.0254, +0.0235]|2|
|appearance_cue/union|LR_photo_revision_atlas|-3.8887|[-4.3203, -3.5320]|0|
|appearance_cue/union|LR_photo_revision_inverse|-2.3641|[-2.4783, -2.2212]|0|
|appearance_cue/union|H3_ED_revision_inverse|-2.9218|[-3.2902, -2.6229]|0|
|true_content_change/union|current_LR_VAE|-0.1731|[-0.2305, -0.1151]|0|
|true_content_change/union|closed_action_no_equiv|+0.0318|[+0.0084, +0.0571]|6|
|true_content_change/union|factor_cache_mlp|-0.0195|[-0.0311, -0.0076]|1|
|true_content_change/union|nonlinear_group|-0.0038|[-0.0191, +0.0111]|4|
|true_content_change/union|gru_online|+0.0282|[+0.0081, +0.0492]|6|
|true_content_change/union|gru_replay|+0.0206|[+0.0030, +0.0376]|6|
|true_content_change/union|LR_photo_revision_atlas|-3.7634|[-4.1760, -3.4070]|0|
|true_content_change/union|LR_photo_revision_inverse|-1.0494|[-1.4005, -0.6418]|1|
|true_content_change/union|H3_ED_revision_inverse|-2.6142|[-2.8386, -2.4029]|0|

Fixed-checkpoint mechanism interventions (no retraining):

|Condition|Intervention|full minus intervention union dB|CI|Wins/8|
|---|---|---:|---|---:|
|all|closed_action_frozen_photo|+0.0000|[-0.0002, +0.0003]|5|
|all|closed_action_frozen_weights|+0.0031|[+0.0005, +0.0065]|6|
|all|closed_action_frozen_both|+0.0031|[+0.0004, +0.0066]|6|
|all|diagnostic_closed_no_measured_detail|-0.0928|[-0.1212, -0.0652]|0|
|all|diagnostic_closed_shuffled_history|+0.0109|[-0.0143, +0.0403]|4|
|appearance_cue|closed_action_frozen_photo|+0.0000|[-0.0002, +0.0003]|4|
|appearance_cue|closed_action_frozen_weights|-0.0001|[-0.0002, +0.0000]|3|
|appearance_cue|closed_action_frozen_both|-0.0001|[-0.0004, +0.0002]|3|
|appearance_cue|diagnostic_closed_no_measured_detail|-0.0933|[-0.1222, -0.0652]|0|
|appearance_cue|diagnostic_closed_shuffled_history|+0.0098|[-0.0154, +0.0401]|4|
|neutral_return|closed_action_frozen_photo|+0.0001|[-0.0001, +0.0003]|6|
|neutral_return|closed_action_frozen_weights|-0.0000|[-0.0001, +0.0001]|5|
|neutral_return|closed_action_frozen_both|-0.0000|[-0.0003, +0.0002]|5|
|neutral_return|diagnostic_closed_no_measured_detail|-0.0936|[-0.1244, -0.0654]|0|
|neutral_return|diagnostic_closed_shuffled_history|+0.0085|[-0.0204, +0.0417]|4|
|true_content_change|closed_action_frozen_photo|-0.0001|[-0.0005, +0.0003]|5|
|true_content_change|closed_action_frozen_weights|+0.0111|[+0.0020, +0.0231]|6|
|true_content_change|closed_action_frozen_both|+0.0110|[+0.0020, +0.0229]|6|
|true_content_change|diagnostic_closed_no_measured_detail|-0.0911|[-0.1196, -0.0628]|0|
|true_content_change|diagnostic_closed_shuffled_history|+0.0149|[-0.0057, +0.0406]|5|

Raw/DC, per-source and regional data are retained in `method_summary.csv`, `source_aggregates.csv` and `paired_differences.csv`. The `changed` region is evaluated only after actual replacement queries13/17; a pre-confirmation branch claim is not made.

The independent streaming implementation stores compressed h rather than native delta/proxy codes. It is audited against the vectorized rollout; all raw factor h/weight/ID/descriptor and group buffers are included. Appearance-action-only cost may be constant in N for fixed group count, while the actual whole-state update still scans packet IDs/weights, and LR reliability/controller work can scale with N. No O(1) whole pipeline claim.

Shared current LR49152B + native anchor172032B + decoded RGB786432B =1,007,616B of current-query payload, reference LR49152B additionally persistent, and shared camera524288B. Frozen H3 weights are shared and separately reported. Compressed factor h is28672B per packet plus weights1024B,8x8RGB descriptor768B and assumed24B ID/group/geometry. The raw-LR inverse archive is49152B per packet plus controller/metadata; this is a stronger storage comparison than contrasting4 channels with a dense RGB atlas. Python allocator overhead, activations and CUDA peak are separate.

The classical inverse models relative spatial gain+bias before the exact observation operator; observed RGB clipping makes this affine model approximate. Native decoded-atlas photometric normalization is an empirical control, not exact latent equivariance. All methods use the same causal confidence with explicitly different resolution resampling and the same64step current-LR DC. Generated outputs never enter the measurement ledger.

Interpretation must be filled from the fixed results, including whether full beats the same-compression factor-cache/replay-GRU and preserves newly acquired detail beyond current-anchor priors. Algebraic closure alone does not establish originality.
