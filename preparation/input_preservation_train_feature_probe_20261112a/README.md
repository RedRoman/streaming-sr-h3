# TRAIN-source/group internal linear Huber diagnostic

This is a new diagnostic preparation, not a change to either frozen1000-step
head, the failed development screen, the K1 teacher, or its caches. Root owns the
single future remote execution. No SSH, upload or actual data run was performed
while preparing this bundle.

Fixed budget:2CPU affinity,4GiB address space,180seconds wall time, no subprocess,
no retries,8MiB maximum individual JSON output. The deadline includes metadata,
1000 full NPZ file hashes, selective array reads and all5 fits. On failure retain
the fresh output and any completed split/report; never increase time implicitly.
SIGTERM/INT/alarm produce a failed report when Python can handle the signal.
A hard-killed process lacking a terminal report is not accepted.

The three admitted input files are the authenticated TRAIN cache manifest, its
completed actual CPU audit, and the original student schedule. No development
file is allowed in the execution file set. Each of the19 train sources stays
inside its original source_group. Sort groups by SHA256 of canonical JSON
`[20261112, group_id]`, break ties by group ID, then assign group rank modulo5.
This exact split is saved as `split.json` before any NPZ feature/label member
is read. There is no label-driven balancing, repartitioning or fold selection.

Each NPZ file is hashed once and its index/array SHA checked. Only raw_features
and normalized_benefit members are decompressed; K1 RGB, targets and raw-benefit
members are not loaded. Features are converted to FP64 and averaged over the two
slots, producing11 tile-local inputs; this equals a shared1x1 linear score
followed by slot mean. It does not reproduce the original nonlinear3x3 network.

For each fold, mean/std come from that fold's training tiles only. All11 channels
are standardized; std<=1e-12 is a zero column, and an intercept is retained.
Fit one12-coefficient linear predictor from zero using exactly200 full-batch
gradient steps, SmoothL1 beta1 plus ridge1e-3 on slopes only. Fixed step is1/L,
where L is the largest eigenvalue of training Z.T@Z/n plus the ridge diagonal.
This is an analytic curvature bound, not a line search or hyperparameter search.
Report the final gradient norm and do not claim exact linear optimality.

The comparison is each fold's training-only constant Huber optimum. Solve both
ends of its minimum interval using60 bisections and select the closest point
to0, preserving a flat interval such as [-3,3]. Label normalization/clipping and
hard decision threshold0 remain unchanged. Held-out scores are computed once.

Report all19 source results and source-macro Huber loss as primary, profile/source
macros and tile-micro results separately. Conditional positive/negative rates
have explicit valid-source denominators. Frames/tiles are correlated. K1's
teacher previously trained on all19 sources: only the probe's head fitting is
group-held-out, not the teacher. No significance, deployed quality, raw MAE gain,
new training permission or independent-test claim follows from this diagnostic.
It cannot establish absence of nonlinear/spatial raw-observation information.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHON -m unittest discover -s preparation/input_preservation_train_feature_probe_20261112a -p test_probe.py -v
PYTHON preparation/input_preservation_train_feature_probe_20261112a/probe.py --execution-contract preparation/input_preservation_train_feature_probe_20261112a/execution_contract.json --execution-contract-sha256 09fa7efa488c2cc8760d1edd264befe5ce1031b73774e529a4a57b4f3d46a32e --out /cache/zihaohe/omni_openh3_20261111/runs/input_preservation_train_feature_probe_20261112a --execute
```

Use the existing Python3.12.14/NumPy1.26.4 CPU environment. Omitting `--execute`
only inspects metadata and writes the split, but claims its fresh output; do not
use the actual output directory for a preliminary inspection. Actual invocation
enforces the runtime versions and thread environment. Local7/7 synthetic tests
passed; no real cache throughput or Linux limit execution is claimed.

The bundle's member names start with
`preparation/input_preservation_train_feature_probe_20261112a/`.
Extract once at project root, not inside the preparation directory.
