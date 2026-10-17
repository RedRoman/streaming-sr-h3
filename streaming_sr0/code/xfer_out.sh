#!/bin/bash
# 30794 -> /temp relay (shared with 30778). Read-only on sources.
set -e
X=/temp/zihaohe_streaming_xfer_20261016
mkdir -p $X
cd /cache/yunfeng/envs && tar cf $X/env_minimax-h3-npu.tar minimax-h3-npu && echo ENV_TAR_DONE
cd /cache/zihaohe/streaming_sr0 && tar cf $X/ssr0_small.tar code tools env.sh PROTOCOL_step*.md step*_REPORT.md data/events data/lasot_mirror_tree.json torch_home assets/comfy_h3/h3_qwen_context_neutral_restore.pt assets/comfy_h3/h3_qwen_context_neutral_restore.json && echo SMALL_TAR_DONE
for f in minimax_h3_video_vae_fp16.safetensors minimax_h3_fl2va_pruned_bf16.safetensors; do cp assets/comfy_h3/$f $X/$f.tmp && mv $X/$f.tmp $X/$f && echo "COPIED $f"; done
cd $X && sha256sum minimax_h3_video_vae_fp16.safetensors > vae.sha && echo XFER_OUT_DONE
