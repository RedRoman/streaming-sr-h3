#!/usr/bin/env bash
# The sampling-seed noise floor at TWO steps, which is now the shipped configuration.
#
# The floor on record (sd 0.0018 LPIPS, spread 0.0035) was measured at four steps. Two steps draws fewer
# stochastic samples, so its own floor is unmeasured -- and the headline that two steps BEATS four by
# 0.0056 is currently quoted against the four-step floor. If the two-step floor is larger, that claim
# weakens; if smaller, it strengthens. Either way every future two-step comparison needs the right
# denominator.
#
# Same arm as t4_plain2 (r5_k3, 2 steps, dc=rphi, merged Stage-1, no adapter), two further seeds, full 96.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
RPHI=../results/range_branch/r5_k3.pt
LORA=../results/stage1_lora/lora_step002000.safetensors
python code/preflight.py --cards 4,5,6,7 --need-gb 45 --eval-set data/real_eval/heldout \
  --ckpt results/range_branch/r5_k3.pt || { echo 'preflight failed' >&2; exit 1; }
i=0
for SEED in 202 303; do
  for SH in 0 1; do
    dev=$((i + 4))
    (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code
     python -u run_deploy.py --tag t4_plain2_s$SEED --turbo-steps 2 --dc rphi --guide y \
       --rphi "$RPHI" --merge-lora --lora "$LORA" --seed $SEED --shard $SH --nshards 2 \
       > ../logs/n2_${SEED}_$SH.log 2>&1
     echo "seed $SEED shard $SH exit=$? $(date +%H:%M:%S)") &
    i=$((i+1)); sleep 20
  done
done
wait
for SEED in 202 303; do echo "t4_plain2_s$SEED: $(ls -d results/deploy_runs/t4_plain2_s$SEED/*/ 2>/dev/null|wc -l)/96"; done
export ASCEND_RT_VISIBLE_DEVICES=4
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_noise2 \
  --arm s101=../results/deploy_runs/t4_plain2 \
  --arm s202=../results/deploy_runs/t4_plain2_s202 \
  --arm s303=../results/deploy_runs/t4_plain2_s303 \
  --arm dck3=../results/deploy_runs/t4_k3_ho \
  --pair s202,s101 --pair s303,s101 --pair s101,dck3 \
  > ../logs/judge_noise2.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
