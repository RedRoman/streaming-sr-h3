#!/usr/bin/env bash
# Every latency asset in Kijai's repo that targets OUR exact checkpoint layout (pruned fl2va).
#
# The step-count LoRAs matter most: the DiT is 13.8 s of a 32.1 s block at our 4 steps, so a 2-step
# adapter halves the single largest line item. pdmd_2step is the one to beat. The w4a8 file is a
# quantisation of the pruned fl2va base itself, i.e. the same 20.11B / 532-tensor layout we load, so it
# is a drop-in candidate rather than a conversion project -- subject to Ascend having the kernels, which
# is the open question.
#
# Headers only would not settle usability here, so these are fetched in full. Sizes are modest except
# the w4a8 base.
set -u
D=/cache/zihaohe/streaming_sr0/assets/speed
mkdir -p "$D"
cd "$D" || exit 1
B=https://hf-mirror.com/Kijai/MiniMax-H3-experimental/resolve/main
for F in \
  loras/minimax_h3_pdmd_2step_lora_avg_rank_38_bf16.safetensors \
  loras/minimax_h3_pdmd_4step_lora_avg_rank_57_bf16.safetensors \
  loras/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors \
  loras/minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16.safetensors \
  loras/minimax_h3_taomate_fl2va_3step_avg_rank_19_bf16.safetensors \
  minimax_h3_lightvae_lite_encoder_fp16.safetensors \
  minimax_h3_lightvae_lite_vae_int8_convrot.safetensors \
  minimax_h3_video_vae_int8_convrot.safetensors ; do
  N=$(basename "$F")
  for try in 1 2 3; do
    curl -L --max-time 3600 -C - -o "$N" "$B/$F" 2>&1 | tail -1
    sz=$(stat -c%s "$N" 2>/dev/null || echo 0)
    [ "$sz" -gt 1000000 ] && break
    sleep 5
  done
  printf '%-72s %12s\n' "$N" "$(stat -c%s "$N" 2>/dev/null || echo FAILED)"
done
echo '--- header sanity: tensor counts and per-module ranks against our base'
python3 - <<'PY'
import json, struct, collections
from pathlib import Path
D = Path('/cache/zihaohe/streaming_sr0/assets/speed')
base = Path('/cache/zihaohe/streaming_sr0/assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors')
with base.open('rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; O = json.loads(f.read(n))
O.pop('__metadata__', None)
strip = lambda k: k.split('diffusion_model.')[-1]
bk = {strip(k): tuple(v['shape']) for k, v in O.items()}
for p in sorted(D.glob('*.safetensors')):
    try:
        with p.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
        h.pop('__metadata__', None)
    except Exception as e:
        print('%-70s unreadable: %s' % (p.name, e)); continue
    ranks = collections.Counter()
    hit = miss = 0
    for k, v in h.items():
        kk = strip(k)
        for suf in ('.lora_A.weight', '.lora_B.weight', '.lora_down.weight', '.lora_up.weight'):
            if kk.endswith(suf):
                ranks[min(v['shape'])] += 1
                tgt = kk[:-len(suf)] + '.weight'
                hit += tgt in bk; miss += tgt not in bk
                break
    print('%-70s tensors %4d  lora_targets_in_base %d/%d  ranks %s'
          % (p.name, len(h), hit, hit + miss, dict(ranks) or '-'))
PY
