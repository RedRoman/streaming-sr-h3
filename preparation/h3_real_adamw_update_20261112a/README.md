This is preparation, not an admission or a launch queue.

The frozen candidate performs one real TRAIN compression AdamW engineering update. It reuses the accepted full22 reference: 12 NFE, 4 native VAE decodes, 1 scaled RGB backward, 1 AdamW update; no encode, Rphi, extra reference, optimizer retry or formal training run. The step-1 learning rate is the preregistered candidate `1e-4 * min(1, 1/100)`, not an established optimal training recipe.

The execution contract contains `root_admission: null`. Root must provide a separate producer admission binding its canonical payload SHA, then a separate queue and launch admission binding the exact queue, execution contract, runner and fresh physical card. Use `run_once.py` only after that approval. It enforces a single physical card, shared lock, two checks 15 seconds apart, at least 60,000 MB device free, 160 GiB host available, 10 GiB disk free, 4 CPU threads, and an owned process-group 128 GiB RSS/3,600-second limit plus 30-second cleanup. No wait for hardware, retry, or automatic budget increase.

Default input inspection is CPU-only:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<this prep>/code:<root>/preparation/vae_roundtrip_runtime_20261111_v2/code <root>/venv/bin/python <this prep>/code/check_h3_real_adamw_update.py --pretrained-manifest <root>/configs/openh3_20261111/refiner_turbo_30794.json --execution-contract <this prep>/execution_contract.json --out <new inspection directory>
```

The helper roots and complete previous full22 runtime pin union are preserved. Root must add the final execution contract, admissions, queue and independent audit source pins to the launch queue without changing old snapshots. The executable is intended for the already matched 31112 host; no imported nonzero-b weights.

Arithmetic and strict adapter/optimizer reload acceptance are distinct from functional readiness. The latter requires a byte-exact zero-versus-accepted-reference RGB control and changed deployed merged RGB. Otherwise it records either no observed change or numerical-mode confounding. Even a change does not establish restoration quality or repeatability. The old synthetic SGD negative result remains valid.

The separate terminal auditor reads actual arrays and safetensors in place, including 400 adapter tensors and 1,200 optimizer states before and after reload. It has no model actions. Root must separately admit its finalizer; the wait is at most 4,200 seconds and its single audit plus small archive at most 900 seconds, 2 CPU cores and 4 GiB owned worker RSS. Large RGB, parameters, gradients and checkpoints remain remote.
