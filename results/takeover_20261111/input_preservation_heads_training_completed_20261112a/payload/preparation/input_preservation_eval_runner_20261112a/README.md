# One-shot final-head CPU evaluation runner

New source only; frozen evaluator25cdeec76d1472fe1a590d27ba0d65553a67ce9fada3d329210874d7bab28e39, trainer2df11a8563940aea729e2b3e4252f38515c0a66889caa2635debef72239e54cd. Reuse exact head-runner pure primitivesf8958a817af333ac9a8332a556abc2d0f8a63d0ee17d7c3e0a17a1a965bad0ec and owned-PG supervisor efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388. Neither old source is modified.

Root fills exact numerical eval contract/admission, both completed1000 training reports and execution contracts, final checkpoint SHA, metric-readiness and all implementation/weight pins. Then freeze plan and create a distinct launch admission bound to exact plan SHA. Templates contain deliberately pending paths/nulls and cannot launch. No auto-listener, wait queue, retry or evaluation after training is implemented.

CLI: `PYTHON preparation/input_preservation_eval_runner_20261112a/code/run_input_preservation_eval_once.py FINAL_PLAN.json [--inspect-only]`.

Per run:1800s +30s owned cleanup,2 CPU,4GiB sampled simultaneous owned-PG RSS,256MiB output. The exact worker argv is evaluator --execution-contract CONTRACT --out RUN/output --execute. CPU auto-NPU import disabled/visible devices empty/noCANN. SIGTERM/INT/timeout/RSS terminate only the owned worker PG; live descendants block completion. A fresh run directory is claimed once, never reused.

Prelaunch verifies both research-final1000/16000-row reports with exact operation counts, same common numerical contracts/initial state, train-cache/dev-cache/metric-readiness bindings and actual checkpoint bytes. The evaluator independently deserializes and verifies the actual checkpoint tensors. The stdlib wrapper itself never imports Torch or reads model tensors.

Terminal gate requires360head calls/5760head rows,720LPIPS calls/11520LPIPS rows,zero restorer/backward/update,actual observed calls180/180/720,all frozen flags,CPU runtime and complete2880record/2850pair/180batch JSONL files with exact SHA/size/line counts. Score/mask/RGB array semantics remain the evaluator's evidence and future independent array audit; this wrapper does not claim to recompute them.

Scientific verdict is preserved separately. `failed_dev_screen` or `pending_review` is still completed execution when full evidence is valid. Only producer/integrity/resource failure makes execution failed. No quality-proven or training/fullframe admission is created.

Seven local fixtures are stdlib mocked supervision, with no actual model/cache/device. Frozen Linux process-group logic is reused rather than re-tested here. source_bundle.tar.gz member layout begins preparation/input_preservation_eval_runner_20261112a/; extract ONCE at project root, not into that prep directory.
