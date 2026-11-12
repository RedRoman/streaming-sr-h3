# Target preparation only — no transfer or NPU launch

This directory is a new local installer/verifier candidate. No installer or
verifier has been run against 31984. The existing download and its
`transfer_jobs.json` are owned by the parent task and are unchanged.

The frozen package contains 346 original-root relative paths, represented by
150 unique blobs and one exact manifest (151 regular tar members). The paths
total 1,196,615,743 bytes; the unique blobs total 1,084,653,343 bytes. The archive
is 718,719,172 bytes with SHA-256
`b26949c1b9c8b9fbd98d94249146c4135a19dda16ec824a465c3987124fd3f2a`.
Its manifest SHA is
`b862d92543b1e994d8896d75936fb94c081ffd2cb2be4e8bd3de65e8e8e20d20`.

The four pre-existing model files are not in the package. They total
53,258,114,120 bytes and still require actual target SHA verification. The full
dependency index contains 1,406 unique paths: 1,364 frozen implementation pins
plus 42 other required files. It comes from target inventory
`ba21bafa64fb5ca557cccee0a58017605f51b3520ac04f024fee6498c89bc05c`.

## Installation guarantees and limits

`install_once.py` checks the exact archive SHA, size, member allowlist, manifest
bytes, every blob SHA and destination identity. It never calls tar extraction.
Absolute paths, traversal, directory/link/special/sparse/PAX members,
duplicates, undeclared members and file/directory conflicts fail closed.

All existing targets are checked before any destination creation. Identical
files retain their inode and mode; conflicting files stop the operation. Each
new target gets its own copied inode, is fsynced, and is committed by an
exclusive link. A destination that appears concurrently causes failure even
when its bytes are equal. The temporary per-target link is then removed.
Targets do not share an inode with other snapshots or the staged blob. New
files are mode 0444; original mode metadata was not in the source manifest and
is not claimed preserved. Invoke Python/shell sources via the interpreter.

This is a journaled, rollback-capable multi-file operation, **not globally
atomic visibility**. An ordinary exception rolls back only files whose inode
is still owned by this transaction, and only its empty directories. Existing
or concurrently replaced files are not deleted. A hard kill, disk failure or
crash can leave a partial install; preserve the journal and reconcile it before
authorizing any new run. There is no automatic retry, overwrite, cleanup of
historical files, environment installation or NPU admission.

Execute needs at least 3 GiB free for unique blob staging, independent target
copies and one temporary copy. Successful staging is retained as evidence;
there is no automatic deletion of it. Both commands below limit affinity to
the first two available CPU cores and address space to 4 GiB. Internal alarm
890 seconds reserves 10 seconds for cleanup; the outer 900-second KILL guard
is necessary for a stalled syscall or slow cleanup.

## Commands after the unique transfer has completed

First verify the completed transfer and copy this newly frozen preparation
directory to the target through the parent's explicit transfer step. This
document does not initiate or replace that transfer. The archive's proposed
target path is below; use the actual verified path if the parent chooses a
different new transfer destination. No old continuation snapshot is edited.

```sh
TASK_ROOT=/cache/zihaohe/omni_openh3_20261111
INSTALL_PREP="$TASK_ROOT/preparation/h3_saved_continuation_target_install_20261112a"
MIGRATION_ARCHIVE="$TASK_ROOT/preparation/h3_saved_continuation_migration_31984_20261112a/output/missing_dependencies.tar.gz"
TASK_PYTHON="$TASK_ROOT/venv/bin/python"
```

Run the default inspection once. It reads the archive and destination files,
writes only a new receipt directory, and installs nothing:

```sh
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout --signal=KILL 900s "$TASK_PYTHON" "$INSTALL_PREP/install_once.py" --archive "$MIGRATION_ARCHIVE" --out "$TASK_ROOT/runs/h3_saved_continuation_target_install_20261112a_inspect"
```

After its complete receipt is reviewed, the separately authorized installation
uses a different new output directory:

```sh
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout --signal=KILL 900s "$TASK_PYTHON" "$INSTALL_PREP/install_once.py" --archive "$MIGRATION_ARCHIVE" --out "$TASK_ROOT/runs/h3_saved_continuation_target_install_20261112a" --execute
```

Do not chain execution onto inspection or rerun either with an existing output.
An identical pre-existing target is preserved; a conflicting target requires
investigation, not replacement.

## Independent read-only full target verification

After installation is complete and a new CPU read is approved, verify all
1,406 files including the 53.26 GB of existing model weights. This never loads
weights into a model, imports Torch/Comfy, or mutates dependencies. Every path
is streamed once for SHA, with file identity checked before and after reading.
`sys.version` and installed Torch/torch_npu distribution versions are compared
exactly with the original runtime record. Package metadata verification is not
a claim that imported runtime behavior or numerical parity has been verified.

```sh
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout --signal=KILL 1800s "$TASK_PYTHON" "$INSTALL_PREP/verify_target.py" --out "$TASK_ROOT/runs/h3_saved_continuation_target_full_verify_20261112a" --full-sha
```

This is bounded to 2 CPU cores, 4 GiB address space and 1,800 seconds; the
internal alarm is 1,780 seconds. A missing file, mismatched SHA, changed file
identity or runtime-metadata difference leaves `full_dependencies_ready=false`.
Without `--full-sha`, the same CLI only stats paths and always leaves full
readiness false; it is not a substitute for the full command. Neither result
authorizes NPU work. Actual backend/import checks and fresh resource/lock
admission remain with the unchanged continuation launcher and parent task.

## Local validation scope

The fixtures use real small gzip/tar files and temporary directories. They
cover SHA failure, traversal, link members, preservation of existing files,
independent same-blob destination inodes, a real exclusive-link race and owned
rollback. Verifier tests distinguish stat-only from full SHA, reject missing
and symlink files, and check exact package metadata without framework imports.
The Linux affinity/AS/alarm code is prepared but not executed on the target;
local passing fixtures do not claim target installation or full dependency
verification.
