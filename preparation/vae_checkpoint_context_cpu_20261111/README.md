# Frozen VAE decoder checkpoint helper

`code/vae_checkpoint_context.py` exposes `checkpoint_decoder_blocks(stage, enabled=False)`. It checkpoints exactly 36 distinct `stage.decoder.transformer_blocks` with `use_reentrant=False` and `preserve_rng_state=True`, bypasses no-grad calls, and restores original per-instance forward attributes in `finally`. Enabled nested/shared-block scopes are rejected.

Keep **both forward and backward** inside the scope and the unchanged decoder math/RoPE scope. Output gradient hooks reject backward after exit even when checkpoint recomputation would not be needed. No weights, buffers, requires-grad flags, or training modes are modified. Audit dictionary is JSON-serializable throughout.

30794 isolated CPU execution: Torch2.10.0+cpu, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, 12/12 tests passed in3.380seconds, zero skips, torch_npu not imported and NPU not initialized. Tests establish exact original/checkpoint output and gradient equality, 36 actual recomputes, unchanged frozen weights/versions, RNG preservation, exception restoration, stale-graph rejection, no-grad/disabled bypass, and nesting/count checks. The downloaded receipt and log hashes match local frozen sources.

**Scope limitation:** this is decoder-only validation. Full H3 DiT recomputation uses eager RoPE while the VAE requires Ascend-native RoPE; wrapping the complete H3 backward in one native context is invalid. A full integration still needs distinct per-recompute `context_fn`. This helper does not implement that integration. Small NPU gradient SHA and full-resolution memory/gradient probes remain separate gates owned by the root task.
