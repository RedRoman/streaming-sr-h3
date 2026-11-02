#!/usr/bin/env bash
# Band energy over the full realistic held-out set, plus two training variants that target the deficit.
#
# The 6-clip probe says our output's null band carries 0.458 of the truth's energy and the mid band 0.738,
# against FlashVSR+'s 0.409/0.647 and FlashVSR's 0.562/0.801. So on energy we sit between them, and our
# band cosines are the best of the four (null 0.579, mid 0.859). "Not sharp enough" is therefore an
# energy deficit, not a direction error -- the detail we produce points the right way and is too weak.
#
# The two training variants exist in train_range2 and have never been swept on the big data. Both are
# one-sided, which is the point: the current l1/l1 losses penalise too-sharp and too-blurry equally,
# while the measured defect is purely a shortfall.
#   deficit : (sg - sp).clamp_min(0).mean() on the HR local std -- pays only for missing energy
#   energy  : match the local RMS of the high pass rather than the high pass itself
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
python code/preflight.py --cards 4,5 --need-gb 20 --eval-set data/real_eval/heldout || {
  echo 'preflight failed' >&2; exit 1; }

# full-set band energy for every arm, CPU-bound, runs alongside the trainings
(export ASCEND_RT_VISIBLE_DEVICES=6; source env.sh; cd code
 python -u band_energy.py --out results/band_energy.json \
   --arm bicubic=bicubic \
   --arm infvsr=../results/baseline_infvsr \
   --arm flashvsr=../results/baseline_flashvsr \
   --arm flashvsr_plus=../results/baseline_flashvsr_plus \
   --arm dcy=../results/deploy_runs/t4_dcy_ho \
   --arm dck3=../results/deploy_runs/t4_k3_ho \
   --arm plain2=../results/deploy_runs/t4_plain2 \
   > ../logs/band_energy.log 2>&1
 echo "band_energy exit=$? $(date +%H:%M:%S)") &

D=/cache/zihaohe/range_pairs_x4big
COMMON="--data $D --kinds realistic --ch 128 --blocks 16 --iters 24000 --batch 32 --crop 112 --k 3"
(export ASCEND_RT_VISIBLE_DEVICES=4; source env.sh; cd code
 python -u train_range2.py --tag r7_deficit $COMMON --hf-w 1.0 --std-w 2.0 \
   --std-mode deficit --hf-mode l1 --device npu:0 > ../logs/r7_deficit.log 2>&1
 echo "r7_deficit exit=$? $(date +%H:%M:%S)") &
(export ASCEND_RT_VISIBLE_DEVICES=5; source env.sh; cd code
 python -u train_range2.py --tag r7_energy $COMMON --hf-w 2.0 --std-w 2.0 \
   --std-mode deficit --hf-mode energy --device npu:0 > ../logs/r7_energy.log 2>&1
 echo "r7_energy exit=$? $(date +%H:%M:%S)") &
wait
export ASCEND_RT_VISIBLE_DEVICES=4
source env.sh
cd code
python -u eval_range.py ../results/range_branch/r5_k3.pt ../results/range_branch/r7_deficit.pt \
  ../results/range_branch/r7_energy.pt --data $D --device npu:0 2>&1 \
  | grep -aviE 'owner|warnings.warn|x = torch'
