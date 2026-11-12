# One-shot CPU benefit-head runner

This new runner binds trainer 2df11a8563940aea729e2b3e4252f38515c0a66889caa2635debef72239e54cd and the unmodified owned-PG supervisor efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388. Old runners are unchanged. Each separately admitted arm/mode gets a fresh run, 900 seconds +30 seconds owned cleanup, 2 CPU affinity and 4GiB sampled concurrent owned-PG RSS. Output32MiB. No NPU/CANN/card inspector, model replay, retry, resume or evaluation dispatch.

Root first finalizes numerical training contract+admission, adds every exact file/dependency/admission binding to plan pins, then signs a separate launch admission against the exact plan SHA. Supplied templates have null pins/unapproved permissions and cannot execute. Both actual cache audits, exact donor map, frozen metric implementation/weights, actual Torch fixtures and compatible per-arm backend smokes must precede full research admission. The worker enforces these prerequisites; default runner inspection validates only pins/permissions/argv, without importing its numerical worker.

CLI: `PYTHON preparation/input_preservation_head_runner_20261112a/code/run_input_preservation_heads_once.py FINAL_PLAN.json [--inspect-only]`.

Only actual matching terminal report/counts/runtime and checkpoint/initial/metrics file SHA can complete. A two-step smoke artifact cannot satisfy research1000. This wrapper checks authenticated producer records and file bytes, not an independent Torch checkpoint tensor inventory, resumed next trajectory or quality. Failed/changed inputs remain failed; run directory refuses reuse. All original history-headroom failures remain intact.

Local seven fixtures use a mocked supervisor and harmless metadata. Linux owned-PG mechanics are reused by exact tested source SHA; they were not rerun here. Root alone deploys and launches.
