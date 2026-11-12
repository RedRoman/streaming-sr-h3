# Revised P1 terminal dataset audit

This independent CPU audit waits at most 4200 seconds (30-second polling) for the
frozen 31984 materializer supervisor, then invokes the dataset auditor once.
Default invocation checks small pins only and does not poll the producer or read
NPZ. Execution requires a separate SHA-bound root admission. No automatic retry,
producer replay, model import, MP4 read, decoding, or training is allowed.

Exactly 25 selected sources are required: train19/dev6, 22 new inline authorities
and 3 immutable prior receipts with new reuse sidecars. The frozen revised wrapper
and original loader verify 100 unique NPZ files, actual pixels and metadata. With
cache disabled, creating the teacher schedule additionally rereads 19 train target
NPZ and generates/hashes 19×96 near-clean full frames using the original recipe.
The student schedule reuses these control hashes. Metadata-only opens for the new
88 NPZ are additional. This is not a claim of only100 filesystem reads.

Train/dev dataset instances must both pass and remain legacy research_ready=false;
only the separate revised_pair_ready flag may become true. Both train schedules
use the original seeds and exactly1000×16 rows; only summaries/hashes are saved.
No actual input batch is constructed; shapes are checked from frozen geometry and
rows. The 6 dev sources remain selected development data, with no frozen test.
Passing is dataset/schedule readiness, not training admission or a quality result.

After waiting, audit and small packaging have a 900-second alarm. The worker gets
880 seconds, cleanup10 seconds, leaving time for small packaging. It runs in a new
owned process group, 2 CPU affinity cores, 1 numerical thread, monitored4GiB RSS.
TERM/INT clean only this owned group; an expired timer never targets a reaped PID.
Failed producer/audit/archive leaves readiness false and preserves evidence.
The program creates one allowlisted JSON/log/source archive (≤32MiB payload),
without NPZ/images/videos/weights, and does not itself download it.

Root admission format: approved=true, scope=revised_dataset_terminal_audit_only,
host=31984, freeze_sha256=<exact final freeze SHA>. Invoke finalize_once.py with
--freeze, --freeze-sha256; add --execute --admission --admission-sha256 only after
root review. Fresh output is fixed by execution_plan.json; no overwrite permitted.

Tests use synthetic metadata and the actual frozen numeric schedule code with
synthetic precomputed pixel hashes; Linux also exercises real owned-process cleanup.
They do not claim any real25-source dataset has been validated before execution.
