# B3R reproduction and actual execution

Remote root `/home/saiadmin/streaming_sr_20261015`, H100 GPU7, foreground SSH only with ServerAliveInterval15/CountMax4. Complete H3 VAE SHA must match the manifest. Do not overwrite archived outputs: choose a fresh output root when reproducing; cached `.pt` data remains remote.

Original run order (all commands were actually executed; long output was redirected to `revision_v3/logs`):

```sh
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python src/h3_factorized/cache_revision.py --manifest results/h3_factorized/cache_pilot/manifest.json --out results/h3_factorized/revision_v3/cache --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python experiments/h3_factorized/repair_revision_baseline.py --cache results/h3_factorized/revision_v3/cache --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python experiments/h3_factorized/check_revision.py --cache results/h3_factorized/revision_v3/cache --out results/h3_factorized/revision_v3/protocol_checks.json --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python src/h3_factorized/train_revision.py --cache results/h3_factorized/revision_v3/cache --out results/h3_factorized/revision_v3/latent_stage1 --stage latent --seed 17 --steps 600 --batch 4 --lr .002 --validate-every 100
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python src/h3_factorized/train_revision.py --cache results/h3_factorized/revision_v3/cache --out results/h3_factorized/revision_v3/rgb_stage2 --stage rgb --initial results/h3_factorized/revision_v3/latent_stage1 --initial-checkpoint best --seed 17 --steps 200 --batch 4 --lr .0005 --validate-every 50 --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python src/h3_factorized/evaluate_revision.py --cache results/h3_factorized/revision_v3/cache --models results/h3_factorized/revision_v3/rgb_stage2 --checkpoint best --out results/h3_factorized/revision_v3/val_seed17 --seed 17 --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
```

Supplement endpoint: identical settings, stage1 **last600**, all six fresh200, evaluate only **last200**. Original best-selection was valid under its original protocol and remains a separate artifact.

```sh
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python src/h3_factorized/train_revision.py --cache results/h3_factorized/revision_v3/cache --out results/h3_factorized/revision_v3/rgb_endpoint_stage2 --stage rgb --initial results/h3_factorized/revision_v3/latent_stage1 --initial-checkpoint last --seed 17 --steps 200 --batch 4 --lr .0005 --validate-every 50 --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python src/h3_factorized/evaluate_revision.py --cache results/h3_factorized/revision_v3/cache --models results/h3_factorized/revision_v3/rgb_endpoint_stage2 --checkpoint last --out results/h3_factorized/revision_v3/val_endpoint_seed17 --seed 17 --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python experiments/h3_factorized/audit_revision_runtime.py --cache results/h3_factorized/revision_v3/cache --models results/h3_factorized/revision_v3/rgb_endpoint_stage2 --checkpoint last --out results/h3_factorized/revision_v3/runtime_endpoint_seed17 --seed 17
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python experiments/h3_factorized/diagnose_revision_information.py --cache results/h3_factorized/revision_v3/cache --models results/h3_factorized/revision_v3/rgb_endpoint_stage2 --checkpoint last --out results/h3_factorized/revision_v3/information_endpoint_seed17 --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
```

Fixed TRAIN-source objective diagnosis (no test, no selection):

```sh
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python experiments/h3_factorized/overfit_revision_objective.py --cache results/h3_factorized/revision_v3/cache --checkpoint results/h3_factorized/revision_v3/rgb_endpoint_stage2/closed_action_seed17/last.pt --out results/h3_factorized/revision_v3/objective_single_train_seed17 --steps 300 --lr .0005 --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
CUDA_VISIBLE_DEVICES=7 external/h3_venv/bin/python experiments/h3_factorized/audit_objective_sensitivity.py --cache results/h3_factorized/revision_v3/cache --models results/h3_factorized/revision_v3/objective_single_train_seed17 --out results/h3_factorized/revision_v3/objective_input_dependence --vae external/h3_assets/minimax_h3_video_vae_fp16.safetensors --comfy external/ComfyUI_H3
```

CPU plots/statistics were run locally with the already available NumPy/Matplotlib. The remote H3 venv lacks Matplotlib, so its first summary command failed after successful GPU evaluation/runtime audit; no GPU or training rerun was required for that dependency.

```sh
python3 experiments/h3_factorized/summarize_revision.py --root results/h3_factorized/revision_v3
python3 experiments/h3_factorized/summarize_revision.py --root results/h3_factorized/revision_v3 --evaluation val_endpoint_seed17 --runtime runtime_endpoint_seed17 --summary summary_endpoint_seed17
```

Actual sessions, exit codes and physical extra compute are in the parent `SESSION_REGISTRY.json`. Pretraining first zero-D check failed on a small E-output versus saved-code baseline route difference; `zero_precision_probe.json`, `baseline_route_alignment.json` and the successful rerun `protocol_checks.json` preserve the evidence. No neural parameters were trained/selected by that alignment. Exact code is archived in `source_snapshot/`, hashes in `SOURCE_HASHES.json`; all model `best.pt`/`last.pt` include the actual step and optimizer state. Fixed diagnostic steps0/50/150/300 are separately saved.
