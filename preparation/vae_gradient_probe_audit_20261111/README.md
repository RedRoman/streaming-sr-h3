# VAE gradient probe: small evidence auditing

`code/audit_vae_gradient_probe.py` is stdlib-only. It never imports torch, reads model tensors, scores RGB, or starts jobs. `pack` is optional: root can run it against a terminal run to write a new small tarball; it does not modify source evidence or an existing manifest. `audit` reads tar members in memory, checks the external archive SHA and every member byte SHA/size, launch admission and pinned probe/helper, queue/result state, shapes, forward/backward completion, exact requested gradient, restored checkpoint coverage, frozen parameters and memory records.

```sh
python code/audit_vae_gradient_probe.py pack --root /cache/zihaohe/omni_openh3_20261111 --run runs/RUN --prep preparation/PREP --out NEW_evidence.tar.gz
python code/audit_vae_gradient_probe.py audit --archive EVIDENCE.tar.gz --archive-sha256 SHA --reference-archive PRIOR_C.tar.gz --reference-sha256 0eb1209b7944311729018bb17027a6055d0da83850b032496713ae53b4cfd640 --out NEW_audit.json
```

Omit the reference arguments for standalone full-resolution or failed probes. For a claimed exact prior-gradient comparison, provide the SHA-bound prior archive so input bytes, geometry, seed, loss, dtype, precision, weight fingerprint and shared runtime pins can be verified. The pinned prior gradient alone is weaker evidence and is labeled accordingly.

States distinguish `failed`, `small_decoder_gradient_passed`, and `full_decoder_gradient_and_memory_passed`. The full decoder gate requires 22x480x832, two completed forwards, one completed backward, finite nonzero gradient, forward parity, frozen weights and actual NPU memory observations. Joint full-H3 DiT/LoRA memory and restoration quality are always **not_tested**. Forward allclose and parameter attestations are validated against recorded SHA-pinned producer output, not reconstructed from tensors.

Actual e-versus-c audit passed with exactly36 recomputations. Input-gradient SHA is unchanged, allocated peak5.08281GiB and reserved5.37891GiB. e remains only22x128x128. All16 local CPU tests pass, including the actual earlier full-resolution OOM and malformed success/flow-independent archive cases. No new NPU calls, duplicated downloads, probe edits, or run mutations.

## Explicit v2 and v3 branches

The frozen current auditor passes29 stdlib tests; the retained36-block/two-backward Torch CPU protocol passes3 dedicated tests. Earlier auditors are preserved as `auditor_v1_frozen.py` and `auditor_v2_frozen.py` so prior audit SHA receipts remain reproducible.

V2 `loss_scale_pair` uses two backwards and retained graph. Even complete full-resolution results expose only `paired_scale_gradient_gate=passed_numerical_probe`; `full_decoder_memory_gate=not_established_by_paired_backward`. Missing, nonfinite and all-zero gradients are separate states. Raw scaled and FP32-unscaled counts, magnitudes and final SHA must agree. G independently passed this diagnostic, including the observed all-zero→finite-nonzero transition. This is numerical scale sensitivity, not proof of a unique mean-reduction cause or exact mathematical gradient.

V3 `scaled_single_backward` uses one backward with explicit `fixed_loss_scale`, mutually exclusive with paired mode. The final gradient is FP32-unscaled and SHA-pinned. H-versus-G comparison requires identical effective scale, latent bytes, model weights/dtype/runtime/precision/loss and both raw/unscaled gradient diagnostic records. Only a complete full22x480x832 V3 report may pass the single-backward decoder memory gate. Joint DiT/LoRA memory and restoration quality stay untested.
