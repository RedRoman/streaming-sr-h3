# Teacher1000 checkpoint terminal audit

This is a separately admitted CPU-only finalizer. It cannot start training or evaluation. The prior backend2 auditor and its receipt remain unchanged.

Fill `teacher_watch_config.template.json` into a new config after the root fixes the teacher1000 producer plan and execution contract. All paths are absolute remote paths. Set `producer_status` to plan `run/status.json`, `report` to plan `out/report.json`, and a fresh finalizer `out`. The plan must pin the exact execution contract, fresh teacher1000 only, with no resume.

The separate admission is a JSON with `approved: true`, `scope: "teacher_final_checkpoint_audit_only"`, `execution_payload_sha256` = canonical sorted compact UTF-8 config hash excluding only `admission`, and `finalizer_sha256` equal to the frozen finalizer source. Record its actual path/SHA in config. The template intentionally has null fields and cannot launch.

Inspect: `python finalize_teacher_once.py --config NEW_CONFIG.json` (does not poll). Root alone may dispatch: `python finalize_teacher_once.py --config NEW_CONFIG.json --execute`.

The finalizer waits at most 7500 seconds, polls every 30, recognizes completed/failed/not_launched, and audits only a complete pinned producer once. The audit uses two CPU cores, 4 GiB address-space cap and 120 seconds; timeout cleanup affects only its own CPU process group. `status.json`, `audit.json`, `audit.log`, `archive_receipt.json` and `evidence.tar.gz` are written to the fresh output. Archive allows only small JSON/JSONL/log/Python/Markdown, at most 16 MiB uncompressed; checkpoint tensors and NPZ/NPY stay remote.

Actual checkpoint tensors, optimizer state and saved schedule rows are independently read on CPU. Runtime operation counts and final live freeze/reload flags remain authenticated producer attestations. This does not prove exact future resumed training, restoration quality, or headroom. Failure remains failure and never triggers training, evaluation, a retry, or a replacement run.
