#!/usr/bin/env bash
# Band energy on CPU. It imports no torch_npu and does nothing a card is needed for -- the previous run
# only sourced env.sh (which demands a device index) and so died with everything else when a colleague's
# eight-card TP benchmark took all 8 NPUs at 61.5 GB each. Measurements that do not need an accelerator
# should not be killable by accelerator contention.
set -u
cd /cache/zihaohe/streaming_sr0/code || exit 1
export PATH=/cache/zihaohe/streaming_sr0/venv/bin:$PATH
export OMP_NUM_THREADS=24
python -u band_energy.py --out results/band_energy.json \
  --arm bicubic=bicubic \
  --arm infvsr=../results/baseline_infvsr \
  --arm flashvsr=../results/baseline_flashvsr \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm dcy=../results/deploy_runs/t4_dcy_ho \
  --arm dck3=../results/deploy_runs/t4_k3_ho \
  --arm plain2=../results/deploy_runs/t4_plain2 \
  > ../logs/band_energy.log 2>&1
echo "exit=$? $(date +%H:%M:%S)"
tail -12 ../logs/band_energy.log
