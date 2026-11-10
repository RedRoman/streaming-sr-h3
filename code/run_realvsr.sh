#!/usr/bin/env bash
# Our shipped arm on RealVSR, the second real-degradation set with a reference.
#
# Same configuration as t4_plain2 so the only thing that changes against every other number in this
# project is where the degradation came from: r5_k3 range branch, dc=rphi, 2 turbo steps, Stage-1 LoRA
# merged, no quota gate, no gain.
#
# Reference metrics ARE defined here (unlike videolq) but read prepare_realvsr.py first: the package is
# natively x2 and already resampled to the GT geometry, so taking the LR as LQ/4 puts a synthetic x4 on
# top of the real degradation. The measured MTF gap to a synthetic x4 control is only 0.03-0.11 -- far
# inside the 0.21 that generalised fine -- which means this set stresses real CONTENT much more than it
# stresses real DEGRADATION. mvsr4x remains the harder test of the operator assumption.
set -u
cd /cache/zihaohe/streaming_sr0/code || exit 1
CARDS=${1:-0,1,2,3,4,5}
IFS=, read -ra CA <<< "$CARDS"
NSH=${#CA[@]}
i=0
for c in "${CA[@]}"; do
  (
    export ASCEND_RT_VISIBLE_DEVICES=$c
    export EVAL_SET=data/real_eval/realvsr
    source ../env.sh
    python run_deploy.py --tag rvsr_plain2 --dc rphi --guide y \
      --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora \
      --shard "$i" --nshards "$NSH" > "../logs/rvsr_p${i}.log" 2>&1
    echo "rvsr shard${i} card${c} exit=$? $(date '+%F %T')" >> ../logs/run_realvsr.status
  ) &
  i=$((i+1))
done
wait
echo "rvsr all done $(date '+%F %T')" >> ../logs/run_realvsr.status
