#!/usr/bin/env bash
# The omni deliverable: one estimator over the scale family, measured at the OUTPUT.
#
# Every pixel-level number in this project so far is a single operator, x4 realistic. omni's claim is
# "one model over the operator family" and until now it was only ever checked on R_phi's intermediate
# estimate -- and this project has already measured that intermediate and delivered quality are not
# monotone (a 4.7 dB latent gain once came with a 6.1 dB pixel loss). The x2 and x8 held-out sets built
# last night close that: same 96 clips, HR frames byte-identical to the x4 set, so the three operators
# are mutually comparable.
#
# Six runs, one per card: {omni_m4, r5_big} x {2, 4, 8}. r5_big is the matched single-operator control --
# same 4.7M width, same 10351 windows, same 24000 iters, same crop 112, same K=1 (OmniRange has no
# causal stack, so a K=3 control would charge omni for a feature it structurally lacks). The only
# difference between the two rows is how many operators the estimator serves.
#
# Note on what counts as a difference: the sampling-seed noise floor measured last night is sd 0.0018
# LPIPS, spread 0.0035. Anything under about 0.004 is seed luck -- that is how the width step
# (4.7M -> 13.3M, -0.0020) turned out not to be real despite a "significant" paired CI.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
LORA=../results/stage1_lora/lora_step002000.safetensors
i=0
for OP in 2 4 8; do
  for M in r5_big omni_m4; do
    dev=$(echo "1 3 4 5 6 7" | cut -d' ' -f$((i+1)))
    if [ "$M" = omni_m4 ]; then
      RPHI=../results/omni/omni_m4.pt; OPARG="--op $OP"
    else
      RPHI=../results/range_branch/r5_big.pt; OPARG=""
    fi
    # the x4 set is the original `heldout`; only x2 and x8 got an _s<scale> suffix
    SET=$([ "$OP" = 4 ] && echo heldout || echo heldout_s$OP)
    (export ASCEND_RT_VISIBLE_DEVICES=$dev
     export EVAL_SET=data/real_eval/$SET
     export SR_SCALE=$OP
     source env.sh; cd code
     python -u run_deploy.py --tag t4_${M}_s$OP --turbo-steps 4 --dc rphi --guide y \
       --rphi "$RPHI" $OPARG --merge-lora --lora "$LORA" \
       > ../logs/omnidel_${M}_s$OP.log 2>&1
     echo "$M s$OP exit=$? $(date +%H:%M:%S)") &
    i=$((i+1)); sleep 25
  done
done
wait
for OP in 2 4 8; do for M in r5_big omni_m4; do
  echo "t4_${M}_s$OP: $(ls -d results/deploy_runs/t4_${M}_s$OP/*/ 2>/dev/null|wc -l)/96"
done; done

for OP in 2 4 8; do
  SET=$([ "$OP" = 4 ] && echo heldout || echo heldout_s$OP)
  (export ASCEND_RT_VISIBLE_DEVICES=1
   export EVAL_SET=data/real_eval/$SET
   export SR_SCALE=$OP
   source env.sh; cd code
   python -u deploy_eval.py --procs 8 eval --name judge_omni_s$OP \
     --arm single=../results/deploy_runs/t4_r5_big_s$OP \
     --arm omni=../results/deploy_runs/t4_omni_m4_s$OP \
     --arm bicubic=bicubic \
     --pair omni,single --pair single,bicubic \
     > ../logs/judge_omni_s$OP.log 2>&1
   echo "judge s$OP exit=$? $(date +%H:%M:%S)") &
done
wait
echo OMNI_DELIVER_DONE
