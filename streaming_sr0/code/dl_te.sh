#!/bin/bash
cd /cache/zihaohe/streaming_sr0
python3 tools/fetch_ranges.py \
  --url https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --out assets/comfy_h3/qwen3vl_32b_minimax_h3_bf16.safetensors --size 51506295256 \
  --sha256 600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d --workers 24 2>&1 | tee logs/dl_te.log
