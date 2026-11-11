# Independent terminal H3 audit

Read-only NumPy/stdlib validation for the frozen tiny H3 latent probe. Invoke only after queue_status is terminal. The default execution-contract SHA and producer/trajectory source SHAs are fixed to the reviewed a snapshot. No Torch, H3, model or accelerator imports.

Run from the remote workspace with one CPU BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 venv/bin/python preparation/h3_trajectory_terminal_audit_20261111/audit_terminal.py --run runs/h3_trajectory_probe_20261111a/eval --execution-contract preparation/h3_trajectory_probe_20261111a/execution_contract.json --out preparation/h3_trajectory_terminal_audit_20261111/terminal_audit.json
```

The script streams file hashing and reads one gradient array at a time. It validates all 400 NPY file/content SHAs, actual shapes/counts/finite values and B nonzero support, plus both saved final video/audio arrays. Only a small audit JSON and its hash manifest are emitted; gradient files are never copied or packaged. Float64 norms use rtol1e-10/atol1e-12 to allow different CPU reduction order, without relaxing any byte, finite, parity or presence gate.

Input/noise agreement is backed by two frozen-producer observer hashes; the original noise arrays were not saved. Weight hashes and object/version checks are producer attestations, not a new live-model weight inspection. These limitations are explicit in the output. Failed producers stay failed even when available arrays validate. Pre-hook NFE counts are attempted invocations if the model fails within one; they are not invented completed forwards. Full joint RGB/VAE memory, nonzero adapter reload and restoration quality are outside this gate.

The local synthetic tests write 400 tiny actual arrays, validate the successful case, tamper file/content/output claims, reject missing gradients or wrong noise/block support, validate actual workspace-relative producer paths, and preserve failures before or after partial artifacts. They do not read the real running probe.
