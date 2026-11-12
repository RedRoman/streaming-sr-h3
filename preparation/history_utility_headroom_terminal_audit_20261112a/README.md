# P1 frozen-teacher headroom actual-array audit

Independent NumPy implementation; imports no producer, model, Torch or NPU. Reads each of the180 actual saved FP32 RGB NPZ once for arrays and separately streams its file SHA; no source dataset NPZ or checkpoint reread. Verifies1080arrays/2880rows, source/data/finalteacher/row bindings, all raw/clipped MAE/MSE/PSNR, zeroMSE null, changed-pixel/clip fractions, signed tile16 LOO, per-source/profile/phase summaries and the unchanged preregistered operational rule. Source-balanced degraded gain and all available-tile sign support are reconstructed from actual RGB, not accepted from the recorded metrics.

Fixed before outcomes: absolute1e-12 for FP64 error/utility/summary;1e-10 PSNR. File/array hashes, shapes, counts, validity, missing-slot cache equality/zero utility and gate booleans are exact. Gate thresholds are not softened by comparison tolerance. Reports maximum absolute differences.

`config.template.json` intentionally contains nulls. Root separately binds producer plan/execution/status/report/fresh finalizerout, then supplies admission `approved:true`, `scope:teacher_headroom_saved_RGB_audit_only`, `execution_payload_sha256` of sorted compact UTF-8 config excluding only admission, and `finalizer_sha256`. Inspect only: `python finalize_once.py --config CONFIG`. Execute only after root admission: `python finalize_once.py --config CONFIG --execute`.

Finalizer waits2100s maximum/poll30 for completed/failed/not_launched; one audit600s/2CPU/4GiB plus10s owned-process cleanup. No retry, training/evaluation dispatch, checkpoint selection or student admission. The small archive16MiB cap includes report/audit/status/config/contract/protocol/sources/log; excludes RGB NPZ, checkpoint, gradients, and records.jsonl. Actual arrays remain remote. Numeric audit can complete while operational headroom gate fails; the failure is retained.

Scope:96x96 crops, six selected development sources. Full observed history and runtime frozen/count attestations are verified through producer/code/data chains, not reconstructed from unrecorded tensors. No motion-compensated flicker/LPIPS/full-frame/final-quality/generalization/omni claim.
