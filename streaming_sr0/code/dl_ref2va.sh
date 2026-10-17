#!/bin/bash
cd /cache/zihaohe/streaming_sr0
python3 tools/fetch_ranges.py \
  --url https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \
  --out assets/comfy_h3/minimax_h3_ref2va_pruned_bf16.safetensors --size 40225724176 \
  --sha256 37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe --workers 24 2>&1 | tee logs/dl_ref2va.log
