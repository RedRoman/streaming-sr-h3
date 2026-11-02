#!/usr/bin/env bash
# Fetch the Open-H3 2K refiner LoRA (5.9 GB) through hf-mirror; archive.org/GDrive/Dropbox are blocked
# from this host and huggingface.co itself is not reachable.
#
# Written as a script rather than passed inline: every inline background download today died instantly
# with an empty log while `bash <file>` sessions ran fine. The inline ones all carried a curl -w format
# string full of % and {} through the tmux session wrapper, which is the one thing that differs.
set -u
D=/cache/zihaohe/streaming_sr0/assets/refiner
mkdir -p "$D"
cd "$D" || exit 1
F=open_regen_2k_refiner_rank384_bf16.safetensors
U="https://hf-mirror.com/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors"
for try in 1 2 3; do
  echo "attempt $try at $(date +%H:%M:%S)"
  curl -L --max-time 7200 -C - -o "$F" "$U" 2>&1 | tail -2
  sz=$(stat -c%s "$F" 2>/dev/null || echo 0)
  echo "size now $sz"
  # the header says 416 tensors / 2.934 B bf16 elements, so ~5.87 GB; accept anything above 5.8 GB
  if [ "$sz" -gt 5800000000 ]; then echo "COMPLETE $sz"; break; fi
  sleep 10
done
python3 - <<'PY'
import json, struct
from pathlib import Path
p = Path('/cache/zihaohe/streaming_sr0/assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors')
print('file bytes', p.stat().st_size)
with p.open('rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
h.pop('__metadata__', None)
print('tensors', len(h), '-> expected 416')
need = 8 + n + max(v['data_offsets'][1] for v in h.values())
print('header implies total bytes', need, 'match:', need == p.stat().st_size)
PY
