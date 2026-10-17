#!/bin/bash
cd /cache/zihaohe/streaming_sr0
mkdir -p assets/turbo
B=https://hf-mirror.com/lightx2v/Minimax-h3-Turbo/resolve/main
python3 tools/fetch_ranges_v2.py --url $B/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors \
  --out assets/turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors --size 1956193000 \
  --sha256 c8168ebc17bbacc4296103dda2fec1ba85b24392fa08cf2bfbcef0cff0dc3cc8 --workers 8 --chunk 134217728 2>&1 | tee logs/dl_turbo4.log
python3 tools/fetch_ranges_v2.py --url $B/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  --out assets/turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors --size 1956193000 \
  --sha256 2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e --workers 8 --chunk 134217728 2>&1 | tee logs/dl_turbo8.log
