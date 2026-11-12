# Exact P1 schedules and whole-map donor metadata

`code/prepare_history_utility_schedules.py --execution-contract CONTRACT --out FRESH_OUT`
inspects small source/evidence hashes only. `--execute` requires a separate exact
root admission. The root-owned supervisor supplies the owned-process-group outer
budget; the entry also limits itself to2 affinity CPUs, one numerical thread,
900 seconds and32GiB address space (which also bounds resident memory). There are
no subprocesses, models, training, codec calls or MP4 reads in this entry.

Execution reconstructs the accepted19train/6dev pair using the unchanged actual
loader with cache0:100 unique NPZ file/pixel/metadata checks, plus88 metadata-only
opens for new artifacts. It then sets only the train instance's cache budget to
12GiB. This causes19 additional train-target NPZ reads, with19x96 near-clean frames
generated and hashed by the unchanged algorithm. The student schedule reuses the
control-SHA cache. These are fresh reads in addition to the prior terminal audit;
they are not claimed as reusing its actual pixel verification.

Both complete1000x16 schedules must exactly match the prior accepted revised and
numeric schedule digests. Files are created exclusively; no overwrite or resume.
The manifest uses `openh3.history_utility_persistent_schedules.v1`, and has
`teacher_schedule` and `student_schedule` records containing path/file_sha256,
schedule_sha256/numeric_schedule_sha256; `dataset_sha256` maps train/dev;
`dataset_audit` is the original path/SHA binding. The legacy research_ready flag
stays false. `complete` means preparation finished, never training admission.

The donor permutation addresses future whole6x6 utility maps, without reading any
label values. Units are valid predecessor slots of the student schedule. Strata
are exactly profile/control family, original severity phase, lag2/1 and the raw
external_mask pattern. True boundary availability only removes invalid slots;
raw[1,1] boundary cases are not merged with raw[0,1]. Each source must map to one
source_group. Every donor differs in both source ID and source group.

Within each stratum, seed20261112 SHA256 ordering puts each group contiguously,
orders its units deterministically, and rotates by maximum group size. A group
larger than half of a stratum makes the constrained bijection impossible. This is
recorded explicitly, with no retries, resampling, pooling or unchanged-label
fallback. All possible strata remain documented, but any impossible stratum sets
`donor_ready=false` and blocks student training. Teacher preparation can still be
complete. This file is metadata for future cached labels, not cached utility maps.

The manifest saves19 near-clean control digests, donor semantic/file digests and
source/execution provenance. Output is bounded to256MiB. Failures preserve partial
files and a failed manifest; no automatic retry. Root fills admission before any
launch. Source-freeze binds immutable source/test/docs and the execution payload
excluding admission; the final admitted contract file is separately pinned by
root.10 passing CPU fixtures exercise metadata math, digest gates, boundary masks,
group constraints, impossible strata, and exclusive writes. They do not substitute
for the real25-source execution, which has not been run by this agent.
