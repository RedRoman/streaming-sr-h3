# Cached RGB compensation: completed one-source diagnostic

The 12-sample × 7-beta × 2-family grid completed at **22:01:29 Singapore time, 11 November 2026**, exit 0. Runtime 3808.06 seconds. No new H3 calls or training. This audit only reads JSON; it does not rescore pixels.

Use `audit_final.json` for integrity and all 168 exact results, signed deltas versus input/R, and all 84 same-beta comparisons. `scientific_review.json` records conclusions and negative cases. `audit.json` is the initial audit retained before adding archive/launch provenance checks; the final audit is authoritative.

Integrity passed: 11 archived member hashes, frozen source/launch pins, original metric file SHA chain, exact same R/G/V/target identity, CPU Alex LPIPS backbone and linear-weight locks, 22 frames and 21 valid common-flow pairs; beta0 and scalar beta1 pixel endpoints; native scalar .75 versus alpha .25 anchor reproduction. All identity task-label outputs are exact duplicates.

Quality remains **requires_review**, with **no selected beta**. There is one development source (`6UZVovkR3IU`), four ordinary tasks, one identical clean control repeated under four labels, and four near-clean task-keyed noise draws. These are not 12 independent sources.

Compensation is not universally superior: it removes useful denoising, while reducing damage on clean controls relative to scalar mixing without preserving the clean input. Compression remains much worse than R at larger coefficients. Near-clean beta .05 improves PSNR versus scalar but worsens LPIPS/flow versus scalar; both remain better than the original noisy input. Same beta is not matched repair magnitude.

G and V are already internally clamped VAE outputs. Pre/post-clamp here describes only the final float64 RGB blend and float32 cast. Compensation beta1 is `clip(R+G-V)`, not G. Native scalar .75 equals the earlier full-RGB anchor; SR Q4 anchoring is different.

Artifacts were downloaded once as an 82,779-byte archive; no NPZ cache or video download. Archive SHA256: `c0cf4d31748129e4269cda3962a5fa340054f97363343ac6b87c6c446e1a809a`. Metrics SHA256: `09f55ce3dfb060749f9365bdb66859c28c1bec55bc414f16900edb90619baa39`.

Audit script: `code/audit_cached_rgb_compensation.py`; tests: `code/tests/test_cached_compensation_audit.py` (9/9 passed). No running snapshot, score file, plan, or TAKEOVER file was modified.
