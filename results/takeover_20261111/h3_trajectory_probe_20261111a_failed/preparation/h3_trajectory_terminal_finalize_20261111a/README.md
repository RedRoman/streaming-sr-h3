# One-shot CPU H3 terminal finalizer

This independent process only reads the admitted H3 a queue JSON every30s, for at most2400s. Upon completed/failed, it runs the pinned NumPy auditor once with an additional180s timeout. It never imports Torch/H3, calls npu-smi, changes accelerator state, or starts/stops/restarts any model job.

The fresh owned output `runs/h3_trajectory_probe_20261111a_finalize` must not exist; existence refuses repeat/resume. `finalize_status.json` records the actual PID, polling state, terminal producer state, audit result and final evidence.tar.gz SHA. Producer failures remain failed. A wait timeout leaves the producer untouched and does not invoke the auditor.

The bounded16MiB uncompressed archive includes an immutable finalize_receipt, the audit JSON and hash manifest, frozen producer/preparation sources/contracts/queue/admission, and terminal run JSON/logs. Only the two named final/reference latent NPZs may be included, each≤1MiB. No gradient NPY or model weight file is selected. Manifest hashes bind the exact bytes packed; final archive SHA lives in finalize_status outside the archive to avoid a hash cycle. On wait timeout, active producer logs/metrics are not packaged.

5 small finalizer CPU tests and10 auditor NumPy CPU tests passed. This is orchestration/integrity validation, not a real H3 result.
