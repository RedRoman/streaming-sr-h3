# P1 one-step terminal array audit

This isolated snapshot reads a completed or failed producer report and its saved
NPY files. It imports no Torch, model, optimizer, NPU or Comfy module and performs
no new inference, backpropagation or optimizer step. Invoke the guarded entry only
after the producer is terminal, binding its exact report SHA and using a new output
directory. Never overwrite a failed audit or retry it implicitly.

The NumPy cross-check verifies every available array's file/content SHA, shape,
dtype and finiteness, actual teacher parameter changes, initial student/restorer
inheritance, saved gradient statistics, both teacher/student leave-one-out utility
maps, missing-history support, and the initial zero-logit/one-half gate. The RGB
reference computes float64 absolute differences and color/tile means. Before
viewing any actual-data result, tolerances were fixed to raw utility absolute
2e-7 and normalized utility absolute 3e-4, with zero relative tolerance. Max
errors are retained. Missing-history utility must be exactly zero, and validity
must exactly equal the saved availability mask. These are implementation checks,
not evidence of useful teacher utility, training quality or generalization.

Operation counts and final live baseline/teacher/student freezing depend on the
producer's reported counters/fingerprints. The auditor explicitly distinguishes
these attestations from recomputed saved-array facts. It does not reconstruct an
optimizer step or reload unrecorded final live weights. All ordinary and control
input pixels remain linked to the separately accepted materialization contract;
the audit does not regenerate observations or revalidate all source videos.

The Linux supervisor uses two CPU cores/threads, samples only the newly owned
audit process group's RSS with a 1 GiB limit, allows a 118-second worker phase,
and reserves two seconds for owned-group cleanup under a 120-second stage alarm.
The stage alarm also covers small-evidence packaging. Pre-admission SHA checks
happen before this stage. It never waits for or interrupts the producer.

The archive includes only the producer report, audit/supervisor JSON, a worker log,
the frozen auditor sources and a SHA manifest. It excludes all arrays and model
weights. Actual arrays remain at the producer output; archived references do not
claim their payloads were downloaded. Failed producer states stay failed, and
quality verdict is always `requires_review`.

CLI (root supplies exact report/freeze SHA and fresh output paths):

```text
python run_audit_once.py --freeze /absolute/snapshot/freeze.json --freeze-sha256 SHA --report /absolute/terminal/report.json --expected-report-sha256 SHA --out /absolute/new_audit_run
```

`audit_arrays.py` is the worker. Its CLI is not the resource-guarded public entry.
Three local synthetic NumPy tests cover correct hashes, wrong hashes, NaN despite
valid hashes, reversed utility sign and missing-slot nonzero rejection. They do
not establish an actual-data audit result or test a real model.
