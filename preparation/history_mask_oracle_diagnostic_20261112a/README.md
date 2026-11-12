# Post-failure cached-candidate GT diagnostic

Prepared only; no actual RGB caches or model are read during tests. Uses exactly the already independently audited 180 NPZ /1080 FP32 arrays /2880 dev rows. The original failed headroom gate remains failed.

The primary candidates are 00=current-only,10=remove-recent,01=remove-oldest,11=all-history. Oracle exact-tie priority is 00,10,01,11. Each row permits only masks that are subsets of actual history availability. Primary simultaneous LOO deletion uses raw utility <0; equality keeps. A fixed secondary variant deletes only below -.05/255. Neither policy is selected retrospectively as a method.

For each16×16tile, score unclamped FP64 MAE from the existing FP32 arrays. Compare all, K1, both LOO sign policies, four-candidate oracle; report gains vsK1, error gap to oracle, mask rates. Interaction I=E00+E11−E01−E10 is evaluated only for2820both-available frames /101520tiles. mean_frame_RMS is the average of per-frame RMS values, not a pooled RMS. Aggregate96frames first, then equal source/profile weights; degraded three profiles and identity/near-clean remain separate.

A separate secondary-control table adds raw current input to the same saved candidates, with exact tie priority input,00,10,01,11. It reports input selection and the additional error reduction over the four-candidate oracle. This does not alter the primary four-mask result or approve students.

These are GT error floors (and improvement upper bounds) inside a fixed saved candidate set only. Whole-crop binary-mask predictions are scored per tile; no composite image is generated. A spatially varying input mask changes convolution neighborhoods, so these scores are not a general soft-mask gate bound or a deployed-mask forward. GT is diagnostic only, never a deployable selector, dev-trained gate or independent test result. No extra model/H3/VAE calls, students, images, LPIPS or flow.

`array_checks.py` is an unchanged copy of the frozen independent headroom auditor. Only its pure NumPy/hash/row-check helpers are used; its full audit/main is never called. Each batch gets an actual file SHA and array SHA/shape/finite check; no source dataset NPZ or checkpoint is reread. Actual report and independent-audit assessment differences use the old1e-12 float tolerance and exact Boolean gates.

CPU-only budget2cores/4GiB AS/180seconds; one batch in memory. Root must separately admit once. Add admission{path,sha256} to execution_contract.json; admission requires approved:true,scope:offline_cached_GT_diagnostic_only,execution_payload_sha256=canonical sorted compact UTF-8 contract excluding admission,entry_sha256,out=absolute fresh resultJSON path.

Default inspection (small JSON/hash only; zero saved RGB NPZ reads):
`python diagnose_cached_masks.py --execution-contract execution_contract.json --out NEW_INSPECTION.json`

Actual run only after root admission:
`python diagnose_cached_masks.py --execution-contract execution_contract.json --out /cache/zihaohe/omni_openh3_20261111/runs/history_mask_oracle_diagnostic_20261112a/report.json --execute`

One-shot output uses exclusive creation and preserves failure. Root may supervise the single CPU invocation with its existing SSH session mechanism; there is no polling finalizer, retry, or automatic follow-up.
