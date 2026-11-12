This is a new bounded supervisor, prepared without launching a model. The actual
queue, producer admission and separate launch admission must be frozen by root.
`runner_contract_template.json` documents the exact fields; it is intentionally
not executable. The launch admission is separate to avoid a SHA dependency cycle.

The runner permits one root-selected physical card and one exact producer command.
It verifies pins and both admissions, acquires the existing shared card lock, waits
15 seconds once, revalidates pins and conditions, and performs the second actual
card/host/disk check. A busy card or lock ends this attempt; there is no idle loop.
Unknown device output fails closed through the pinned `takeover_queue` parser.

The worker has a new process group and four-CPU affinity. A 250 ms supervisor
sample enforces the sum of *current* owned-group RSS (128 GiB), with a 3600-second
worker wall limit. This is sampled RSS enforcement, not a cgroup/hard allocation
quota. It excludes unrelated jobs and does not add historical worker peaks.
TERM/INT, timeout, memory overflow, or a parent leaving live descendants causes
TERM to only that owned group, then KILL after 30 seconds if necessary. Zombies
are recorded separately because they cannot execute or retain RSS. The card lock
is cooperative, not a scheduler reservation against other users.

The runner verifies a complete producer report and reports execution completion;
root's independent array auditor still determines numerical acceptance. It never
replays a producer or reuses existing eval/log/status output. Failures remain
failed, and prelaunch resource failures remain not_launched.

Linux stdlib fake-process tests cover normal completion, simultaneous child RSS,
timeout, a foreign process surviving cleanup, orphan prevention, external TERM,
SIGTERM-ignoring child escalation, immutable resource values and signal handler
restoration. They do not load Torch, a model, or NPU runtime. The local macOS run
passed two contract tests and skipped six Linux-only tests; the 31112 CPU run
passed all eight with zero skips. See `runner_cpu_receipt.json` for exact tested
source hashes, timing and log. These tests are not H3 memory/performance evidence.
