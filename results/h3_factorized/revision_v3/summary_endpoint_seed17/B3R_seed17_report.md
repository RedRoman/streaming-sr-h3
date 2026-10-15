# B3R fixed seed17 development pilot

32 train / 8 val natural source photos; synthetic known camera, nuisance-group tags, photometric changes, transient foreground and true local replacement. These BSDS sources were used earlier in the project. No old16test or sealed DIV2K evaluation. One seed; source, not query/packet/epoch, is the bootstrap unit.

All six controls were trained for600 latent+200 actual-native-H3-D steps. The original protocol specified val-selected best checkpoints; parent subsequently required a separate fixed-endpoint comparison, reinitializing all six RGB runs from last600 and evaluating last200. Original selection-pilot and extra physical compute are retained separately. Affine actions, sum statistics, factor caching and replay are standard ingredients; this remains a candidate mechanism experiment.

This report evaluates **last.pt**, from `val_endpoint_seed17`. Exact checkpoint step/hash/initial path: evaluation `complete.json`.

|Condition/region|Control|full minus control dB|95% source bootstrap CI|Wins/8|
|---|---|---:|---|---:|
|all/union|current_LR_VAE|-0.1873|[-0.2370, -0.1376]|0|
|all/union|closed_action_no_equiv|+0.0423|[+0.0120, +0.0768]|6|
|all/union|factor_cache_mlp|-0.0196|[-0.0328, -0.0062]|2|
|all/union|nonlinear_group|-0.0158|[-0.0317, -0.0004]|3|
|all/union|gru_online|+0.0020|[-0.0222, +0.0308]|3|
|all/union|gru_replay|-0.0004|[-0.0180, +0.0178]|4|
|all/union|LR_photo_revision_atlas|-3.8586|[-4.2732, -3.5245]|0|
|all/union|LR_photo_revision_inverse|-2.0946|[-2.2497, -1.9428]|0|
|all/union|H3_ED_revision_inverse|-2.8575|[-3.1593, -2.6038]|0|
|neutral_return/union|current_LR_VAE|-0.1876|[-0.2382, -0.1420]|0|
|neutral_return/union|closed_action_no_equiv|+0.0429|[+0.0088, +0.0793]|6|
|neutral_return/union|factor_cache_mlp|-0.0255|[-0.0464, -0.0084]|1|
|neutral_return/union|nonlinear_group|-0.0211|[-0.0450, -0.0012]|4|
|neutral_return/union|gru_online|+0.0010|[-0.0195, +0.0232]|2|
|neutral_return/union|gru_replay|-0.0125|[-0.0324, +0.0077]|4|
|neutral_return/union|LR_photo_revision_atlas|-3.8734|[-4.2762, -3.5449]|0|
|neutral_return/union|LR_photo_revision_inverse|-2.7005|[-2.8702, -2.5483]|0|
|neutral_return/union|H3_ED_revision_inverse|-2.9693|[-3.3021, -2.6960]|0|
|appearance_cue/union|current_LR_VAE|-0.1882|[-0.2366, -0.1421]|0|
|appearance_cue/union|closed_action_no_equiv|+0.0493|[+0.0171, +0.0855]|6|
|appearance_cue/union|factor_cache_mlp|-0.0179|[-0.0308, -0.0056]|2|
|appearance_cue/union|nonlinear_group|-0.0154|[-0.0302, -0.0014]|2|
|appearance_cue/union|gru_online|-0.0066|[-0.0383, +0.0306]|2|
|appearance_cue/union|gru_replay|-0.0071|[-0.0250, +0.0150]|2|
|appearance_cue/union|LR_photo_revision_atlas|-3.9037|[-4.3344, -3.5521]|0|
|appearance_cue/union|LR_photo_revision_inverse|-2.3792|[-2.4912, -2.2418]|0|
|appearance_cue/union|H3_ED_revision_inverse|-2.9368|[-3.3048, -2.6418]|0|
|true_content_change/union|current_LR_VAE|-0.1856|[-0.2413, -0.1251]|0|
|true_content_change/union|closed_action_no_equiv|+0.0313|[+0.0052, +0.0630]|6|
|true_content_change/union|factor_cache_mlp|-0.0163|[-0.0316, -0.0001]|2|
|true_content_change/union|nonlinear_group|-0.0113|[-0.0269, +0.0060]|2|
|true_content_change/union|gru_online|+0.0161|[-0.0097, +0.0449]|4|
|true_content_change/union|gru_replay|+0.0216|[-0.0010, +0.0417]|7|
|true_content_change/union|LR_photo_revision_atlas|-3.7760|[-4.1838, -3.4255]|0|
|true_content_change/union|LR_photo_revision_inverse|-1.0620|[-1.4105, -0.6537]|1|
|true_content_change/union|H3_ED_revision_inverse|-2.6268|[-2.8477, -2.4205]|0|

Fixed-checkpoint mechanism interventions (no retraining):

|Condition|Intervention|full minus intervention union dB|CI|Wins/8|
|---|---|---:|---|---:|
|all|closed_action_frozen_photo|+0.0001|[-0.0003, +0.0004]|5|
|all|closed_action_frozen_weights|+0.0043|[+0.0009, +0.0086]|6|
|all|closed_action_frozen_both|+0.0045|[+0.0009, +0.0088]|6|
|all|diagnostic_closed_no_measured_detail|-0.1099|[-0.1421, -0.0772]|0|
|all|diagnostic_closed_shuffled_history|+0.0132|[-0.0145, +0.0467]|5|
|appearance_cue|closed_action_frozen_photo|+0.0001|[-0.0004, +0.0005]|5|
|appearance_cue|closed_action_frozen_weights|-0.0000|[-0.0002, +0.0002]|4|
|appearance_cue|closed_action_frozen_both|+0.0001|[-0.0004, +0.0005]|6|
|appearance_cue|diagnostic_closed_no_measured_detail|-0.1099|[-0.1424, -0.0764]|0|
|appearance_cue|diagnostic_closed_shuffled_history|+0.0110|[-0.0174, +0.0440]|4|
|neutral_return|closed_action_frozen_photo|+0.0002|[-0.0002, +0.0005]|4|
|neutral_return|closed_action_frozen_weights|-0.0000|[-0.0002, +0.0001]|3|
|neutral_return|closed_action_frozen_both|+0.0001|[-0.0002, +0.0005]|4|
|neutral_return|diagnostic_closed_no_measured_detail|-0.1106|[-0.1446, -0.0781]|0|
|neutral_return|diagnostic_closed_shuffled_history|+0.0089|[-0.0235, +0.0460]|5|
|true_content_change|closed_action_frozen_photo|-0.0001|[-0.0005, +0.0003]|4|
|true_content_change|closed_action_frozen_weights|+0.0152|[+0.0031, +0.0303]|6|
|true_content_change|closed_action_frozen_both|+0.0153|[+0.0030, +0.0306]|6|
|true_content_change|diagnostic_closed_no_measured_detail|-0.1092|[-0.1424, -0.0746]|0|
|true_content_change|diagnostic_closed_shuffled_history|+0.0207|[-0.0031, +0.0514]|7|

Raw/DC, per-source and regional data are retained in `method_summary.csv`, `source_aggregates.csv` and `paired_differences.csv`. The `changed` region is evaluated only after actual replacement queries13/17; a pre-confirmation branch claim is not made.

The independent streaming implementation stores compressed h rather than native delta/proxy codes. It is audited against the vectorized rollout; all raw factor h/weight/ID/descriptor and group buffers are included. Appearance-action-only cost may be constant in N for fixed group count, while the actual whole-state update still scans packet IDs/weights, and LR reliability/controller work can scale with N. No O(1) whole pipeline claim.

Shared current LR49152B + native anchor172032B + decoded RGB786432B =1,007,616B of current-query payload, reference LR49152B additionally persistent, and shared camera524288B. Frozen H3 weights are shared and separately reported. Compressed factor h is28672B per packet plus weights1024B,8x8RGB descriptor768B and assumed24B ID/group/geometry. The raw-LR inverse archive is49152B per packet plus controller/metadata; this is a stronger storage comparison than contrasting4 channels with a dense RGB atlas. Python allocator overhead, activations and CUDA peak are separate.

The classical inverse models relative spatial gain+bias before the exact observation operator; observed RGB clipping makes this affine model approximate. Native decoded-atlas photometric normalization is an empirical control, not exact latent equivariance. All methods use the same causal confidence with explicitly different resolution resampling and the same64step current-LR DC. Generated outputs never enter the measurement ledger.

Interpretation must be filled from the fixed results, including whether full beats the same-compression factor-cache/replay-GRU and preserves newly acquired detail beyond current-anchor priors. Algebraic closure alone does not establish originality.
