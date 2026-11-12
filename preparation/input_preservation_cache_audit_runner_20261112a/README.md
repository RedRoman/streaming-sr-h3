# One-shot CPU cache audit runner

New runner only; old followup runtime b is unchanged. The exact reused owned-PG supervisor SHA is efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388. Train: 900s +30s owned cleanup, 2 CPU, 32GiB sampled current owned-PG RSS. Dev: same time/CPU, 4GiB RSS. No NPU inspector, card lock, CANN shell, waiting/retry, model or training invocation is provided.

Root must finalize the numerical auditor execution contract and its separate admission first, then fill the plan's null pins, then bind the resulting exact plan SHA into a distinct launch admission. Neither admission admits head training. The supplied templates are not executable. The runner binds the complete producer and exact auditor/dependencies before it can spawn. It accepts only the actual-array terminal status/counts, CPU backend and valid small archive receipt. A metadata-only complete report fails. Final source/plan pins are rechecked. A claimed run directory is never reused, including on failure.

Inspect only (new runner validates metadata/pins; no actual cache reads):
```
venv/bin/python preparation/input_preservation_cache_audit_runner_20261112a/code/run_input_preservation_cache_audit_once.py FINAL_PLAN.json --inspect-only
```
Root-only actual dispatch, once per role:
```
venv/bin/python preparation/input_preservation_cache_audit_runner_20261112a/code/run_input_preservation_cache_audit_once.py FINAL_PLAN.json
```
The exact worker command is `PYTHON audit_input_preservation_cache.py --execution-contract CONTRACT --out RUN/output --execute`. CPU affinity is two available cores; NPU autoload is disabled and visible device variables are empty. RSS is sampled, not a kernel quota; the frozen supervisor kills only its owned process group on timeout/RSS/signal/failure, and refuses completion with live descendants.

The local tests use fake supervision and small JSON fixtures. They do not repeat the already validated Linux process-group tests or scan actual cache arrays. No SSH/deployment/dispatch was performed here.
