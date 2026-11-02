#!/usr/bin/env bash
# Sample the clips the manifest race hid: 15 at x2 and 12 at x8, for both arms.
# run_deploy skips any clip that already has stream_result.json, so this only does the missing ones.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
LORA=../results/stage1_lora/lora_step002000.safetensors
i=0
for OP in 2 8; do
  for M in r5_big omni_m4; do
    dev=$(echo "1 3 4 5" | cut -d' ' -f$((i+1)))
    if [ "$M" = omni_m4 ]; then RPHI=../results/omni/omni_m4.pt; OPARG="--op $OP"
    else RPHI=../results/range_branch/r5_big.pt; OPARG=""; fi
    (export ASCEND_RT_VISIBLE_DEVICES=$dev
     export EVAL_SET=data/real_eval/heldout_s$OP
     export SR_SCALE=$OP
     source env.sh; cd code
     python -u run_deploy.py --tag t4_${M}_s$OP --turbo-steps 4 --dc rphi --guide y \
       --rphi "$RPHI" $OPARG --merge-lora --lora "$LORA" \
       > ../logs/omnifill_${M}_s$OP.log 2>&1
     echo "$M s$OP exit=$? $(date +%H:%M:%S)") &
    i=$((i+1)); sleep 20
  done
done
wait
for OP in 2 8; do for M in r5_big omni_m4; do
  echo "t4_${M}_s$OP: $(ls -d results/deploy_runs/t4_${M}_s$OP/*/ 2>/dev/null|wc -l)/96"
done; done
for OP in 2 4 8; do
  SET=$([ "$OP" = 4 ] && echo heldout || echo heldout_s$OP)
  (export ASCEND_RT_VISIBLE_DEVICES=1; export EVAL_SET=data/real_eval/$SET; export SR_SCALE=$OP
   source env.sh; cd code
   python -u deploy_eval.py --procs 8 eval --name judge_omni_s$OP \
     --arm single=../results/deploy_runs/t4_r5_big_s$OP \
     --arm omni=../results/deploy_runs/t4_omni_m4_s$OP \
     --arm bicubic=bicubic --pair omni,single --pair single,bicubic \
     > ../logs/judge_omni_s$OP.log 2>&1
   echo "judge s$OP exit=$? $(date +%H:%M:%S)") &
done
wait
echo OMNI_FILL_DONE
