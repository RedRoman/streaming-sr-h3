#!/usr/bin/env bash
set -euo pipefail
cd /home/saiadmin/streaming_sr_20261015
# Foreground only. GPU7 was inspected before the recorded run; recheck when
# reproducing rather than taking an occupied GPU from another job.
task_gpu=7
task_used=$(nvidia-smi --id="$task_gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( task_used > 2000 )); then echo "GPU $task_gpu occupied: $task_used MiB"; exit 1; fi
export CUDA_VISIBLE_DEVICES="$task_gpu"
task_python=external/h3_venv/bin/python
task_vae=external/h3_assets/minimax_h3_video_vae_fp16.safetensors
task_comfy=external/ComfyUI_H3
task_cache=results/h3_factorized/cache_pilot
"$task_python" experiments/h3_factorized/check_protocol.py --out results/h3_factorized/protocol_checks.json
"$task_python" src/h3_factorized/cache_pairs.py --data data/BSR/BSDS500/data/images --vae "$task_vae" --comfy "$task_comfy" --out "$task_cache" --counts 32 8 16 --size 256 --context 22
"$task_python" experiments/h3_factorized/check_decoder.py --cache "$task_cache" --vae "$task_vae" --comfy "$task_comfy" --out results/h3_factorized/decoder_gradient_check.json
"$task_python" src/h3_factorized/observations.py --cache "$task_cache" --out results/h3_factorized/observations
"$task_python" experiments/h3_factorized/check_projection.py --cache "$task_cache" --observations results/h3_factorized/observations --out results/h3_factorized/projection_checks.json
"$task_python" src/h3_factorized/train.py --cache "$task_cache" --out results/h3_factorized/latent_stage1 --seeds 17 29 43 --steps 600 --batch 4 --lr .002 --validate-every 100
"$task_python" src/h3_factorized/train_rgb.py --cache "$task_cache" --initial results/h3_factorized/latent_stage1 --out results/h3_factorized/rgb_stage2 --vae "$task_vae" --comfy "$task_comfy" --seeds 17 29 43 --steps 200 --batch 4 --lr .0005 --rgb-every 4 --rgb-weight 5 --validate-every 50
"$task_python" src/h3_factorized/evaluate.py --cache "$task_cache" --runs results/h3_factorized/rgb_stage2 --out results/h3_factorized/test_three_seed --vae "$task_vae" --comfy "$task_comfy" --split test --observations results/h3_factorized/observations
# Copy CSV/JSON/PNG outputs to the local project, then use its existing
# matplotlib runtime (no GPU needed):
# python3 experiments/h3_factorized/summarize.py --results results/h3_factorized/test_three_seed
# python3 experiments/h3_factorized/summarize.py --results results/h3_factorized/test_three_seed/projected
