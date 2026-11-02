#!/usr/bin/env bash
# G5: the multi-operator estimator through the actual sampler, on the operator we ship.
#
# Every omni conclusion so far lives on an intermediate quantity -- the PSNR of R_phi(y) against D(x).
# This project has already measured that intermediate and delivered quality are not monotone (a latent
# gain of 4.7 dB came with a pixel loss of 6.1 dB), so "one model matches five specialists" is not a
# claim about the output until the output is measured. Same 96 held-out clips, same judge, same 4-step
# schedule as the single-operator arms, so the only difference from dcr3 is which estimator produced the
# range.
#
# The operator is passed explicitly (--op 4). That is the setting, not a shortcut: blind estimation of
# the degradation from one LR frame tops out at AUC 0.61-0.68 here, and AirNet/PromptIR-style
# non-blind conditioning is the established precedent.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
RPHI=../results/omni/omni5.pt
LORA=../results/stage1_lora/lora_step002000.safetensors

(export ASCEND_RT_VISIBLE_DEVICES=1; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
 python -u run_deploy.py --tag t4_omni_ho --turbo-steps 4 --dc rphi --guide y --op 4 \
   --rphi "$RPHI" --merge-lora --lora "$LORA" --shard 0 --nshards 2 \
   > ../logs/omni_ho_s0.log 2>&1; echo "shard0 exit=$? $(date +%H:%M:%S)") &

# card 7 is the latency bench; take it when that releases rather than queueing behind a full card
(while pgrep -f 'bench_fixed_cost|bench_e2e_wall' > /dev/null; do sleep 60; done
 export ASCEND_RT_VISIBLE_DEVICES=7; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
 python -u run_deploy.py --tag t4_omni_ho --turbo-steps 4 --dc rphi --guide y --op 4 \
   --rphi "$RPHI" --merge-lora --lora "$LORA" --shard 1 --nshards 2 \
   > ../logs/omni_ho_s1.log 2>&1; echo "shard1 exit=$? $(date +%H:%M:%S)") &
wait
n=$(ls results/deploy_runs/t4_omni_ho | wc -l)
echo "omni sampling: $n dirs at $(date +%H:%M:%S)"
test "$n" -ge 96 || { echo "incomplete, not scoring" >&2; exit 1; }

export ASCEND_RT_VISIBLE_DEVICES=1
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_omni \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm dcr3=../results/deploy_runs/t4_r3_ho \
  --arm omni=../results/deploy_runs/t4_omni_ho \
  --arm bicubic=bicubic \
  --pair omni,dcr3 --pair omni,flashvsr_plus \
  > ../logs/judge_omni.log 2>&1
echo "judge exit=$? at $(date +%H:%M:%S)"
