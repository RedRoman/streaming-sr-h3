# Bounded source identity and sparse content audit

This preparation directory contains a CPU-only audit of the existing 46 video-only candidates and the 290 enumerated historical Stage7 AudioSet source files. It does not construct a dataset, approve a source, assign calibration/test roles, run a model, or modify existing data.

Completed evidence is in `results/takeover_20261111/calibration_candidates_source_identity_audit_20261111.json`; the accompanying `calibration_candidates_source_identity_review_20261111.json` contains the five flagged pair records, all nine cross-time distances per pair, and the remaining eligibility requirements. All 336 files and 1,008 sparse frames completed in 407.97 seconds. The five flags involve two candidates, not five: `6X5vLMqdKZo` has strong sparse similarity to historical `QGd3hjZBFEI`; `Jz4wLRDTF54` shares an all-black sampled thumbnail with four historical sources. These pairs are retained for review, with no assets moved or deleted. The other 44 candidates are not automatically approved. No contact sheet is generated because thumbnail pixels were not retained, only their hashes/features; no extra decoding was launched.

“Full-file SHA” means the full local AudioSet segment MP4, not the entire original online upload. Grouping linked excerpts and reuploads requires evidence beyond literal video IDs and file hashes.

`inputs.json` binds the previously completed inventory, historical-use audit and mixed-temporal data plan by SHA-256. The historical scan is reused, not repeated. The 46 historical sharpness-pass candidates include 45 documented audio-gate failures and one rounded-threshold ambiguity (`Jz4wLRDTF54`); this audit does not reinterpret the audio gate or treat sharpness as a pristine-target verdict.

`audit_sources.py` first verifies each explicitly enumerated file by full SHA-256, size and nanosecond mtime before and after reading. Previously hashed candidate files can be reused only if both size and mtime match. Existing current-pilot train/dev source hashes are checked against the newly read files. The new-read budget is 1 GiB; the process has a 600-second internal audit budget, a 630-second outer timeout, and a 768 MiB between-file RSS stop threshold. This RSS threshold is a checked ceiling before the next file, not an OS hard memory limit. Decoder and numerical-library threads are set to one; work is serial.

After identity checks, three frames per source are selected deterministically at 10%, 50% and 90% of video duration. The frame selected is the first decoded presentation timestamp at or after the requested time, following a backward keyframe seek. Each record contains the actual timestamp and 32×32 RGB thumbnail SHA, 63-bit DCT pHash, 64-bit dHash, RGB mean and grayscale standard deviation. Historical first/middle/last sparse checks are retained in parent evidence, but incompatible thumbnails are not silently reused as this protocol's signatures.

Candidate–candidate and candidate–historical comparisons check all nine cross-time frame pairs. The predeclared triage heuristic flags a pair when both frame grayscale standard deviations are at least 8, pHash Hamming distance is at most 6, dHash distance is at most 8, and mean RGB absolute difference is at most 12 on the 8-bit scale. Exact normalized-thumbnail matches are also listed. These are diagnostic thresholds, not a validated duplicate detector. Low-texture frames are unsuitable for the near-match rule; exact black-frame matches would still be reported separately. Closest historical frame matches are retained even when the rule does not flag them.

Sparse content matches require human review; absence of matches does not establish independence. The check can miss time-offset excerpts, crops, overlays, heavy edits, reuploads and histories outside the enumerated source pool. A YouTube ID is a literal source identifier, not an uploader/content grouping. Full 96-frame readability and target-pixel hashes, source-only visual quality review, grouping of linked excerpts/reuploads, and review of unenumerated history remain prerequisites before role assignment. Foundation-model pretraining overlap remains unknown.

Remote invocation (31984; isolated output only):

```sh
timeout --signal=TERM 630s /cache/zihaohe/streaming_sr0/venv/bin/python \
  /cache/zihaohe/omni_openh3_20261111/preparation/calibration_source_audit_20261111/audit_sources.py \
  --inputs /cache/zihaohe/omni_openh3_20261111/preparation/calibration_source_audit_20261111/inputs.json \
  --out /cache/zihaohe/omni_openh3_20261111/preparation/calibration_source_audit_20261111/source_identity_audit.json
```

The destination must not exist. Progress is atomically written to the new output; a timeout or exception must not be interpreted as completed coverage. All prior source files and historical manifests remain read-only.
