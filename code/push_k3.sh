#!/usr/bin/env bash
# Everything that can run at once on what the box has free.
#
# State: data scaling is exhausted (8.7x gave +0.457 dB, the next 2.0x gave +0.076 and mid_ratio went
# slightly down), while the two structural axes that were previously written off have both reversed now
# that the data is there -- capacity (4.7M was 0.14 dB worse than 2.0M at 598 windows, 0.16 dB better at
# 5184) and the causal stack (+0.237 dB and the largest mid_ratio jump yet at K=3). So the spend goes to
# K and width, not to more windows.
#
# 1. deploy r5_k3 on the held-out 96          -- does K=3's +0.237 dB on the estimate reach the output
# 2. r6_k5     K=5, else identical to r5_k3   -- is the stack still climbing
# 3. r6_xl_k3  13.3M at K=3, else identical   -- do K and width compound or overlap
# 4. omni_m    matched-scale omni, 4.7M/K=3/big data/24000 iters, five operators
#              -- the only run that separates "multi-operator costs 0.023 LPIPS" from "that checkpoint
#              was 2.0M on 600 windows for 6000 iters". Waits for deblur/denoise pairs.
#
# Cards 3-6 are a colleague's four-card job and 0/2 hold 35 GB jobs; 1 and 7 are ours at 6 GB. The 20.1B
# DiT needs 61-65 GB and cannot share, so the deploy takes 1 and 7 and the trainings go on 0 and 2 as
# guests (R_phi training measured 2.8-3.5 GB against 26 GB free).
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
D=/cache/zihaohe/range_pairs_x4big
COMMON="--data $D --kinds realistic --iters 24000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0"

# ---- 1. deploy r5_k3, two shards on the two full cards
for sh in 0 1; do
  dev=$([ $sh -eq 0 ] && echo 1 || echo 7)
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
   python -u run_deploy.py --tag t4_k3_ho --turbo-steps 4 --dc rphi --guide y \
     --rphi ../results/range_branch/r5_k3.pt --merge-lora \
     --lora ../results/stage1_lora/lora_step002000.safetensors --shard $sh --nshards 2 \
     > ../logs/k3_ho_s$sh.log 2>&1; echo "deploy shard$sh exit=$? $(date +%H:%M:%S)") &
  sleep 30
done

# ---- 2/3. the two structural follow-ups, as guests
(export ASCEND_RT_VISIBLE_DEVICES=0; source env.sh; cd code;
 python -u train_range2.py --tag r6_k5 --k 5 --ch 128 --blocks 16 $COMMON --device npu:0 \
   > ../logs/r6_k5.log 2>&1; echo "r6_k5 exit=$? $(date +%H:%M:%S)") &
(export ASCEND_RT_VISIBLE_DEVICES=2; source env.sh; cd code;
 python -u train_range2.py --tag r6_xl_k3 --k 3 --ch 192 --blocks 20 $COMMON --device npu:0 \
   > ../logs/r6_xl_k3.log 2>&1; echo "r6_xl_k3 exit=$? $(date +%H:%M:%S)") &

# ---- 4. matched-scale omni, once its pairs exist
(while pgrep -f 'make_ops.py' > /dev/null; do sleep 60; done
 for op in deblur denoise; do
   n=$(ls /cache/zihaohe/range_pairs_${op}big/train 2>/dev/null | wc -l)
   test "$n" -gt 5000 || { echo "omni_m skipped: ${op}big has only $n windows" >&2; exit 1; }
 done
 export ASCEND_RT_VISIBLE_DEVICES=0; export OMNI_DATA_SUFFIX=big; source env.sh; cd code;
 python -u omni_stage1.py --tag omni_m --scales 2,4,8,deblur,denoise \
   --ch 128 --blocks 16 --iters 24000 --batch 30 --crop 112 --device npu:0 --val-limit 24 \
   > ../logs/omni_m.log 2>&1; echo "omni_m exit=$? $(date +%H:%M:%S)") &

wait
echo PUSH_K3_DONE

export ASCEND_RT_VISIBLE_DEVICES=1
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
n=$(ls ../results/deploy_runs/t4_k3_ho | wc -l)
echo "k3 sampling dirs: $n"
if [ "$n" -ge 96 ]; then
  python -u deploy_eval.py --procs 14 eval --name judge_k3 \
    --arm flashvsr_plus=../results/baseline_flashvsr_plus \
    --arm infvsr=../results/baseline_infvsr \
    --arm dcr3=../results/deploy_runs/t4_r3_ho \
    --arm dck3=../results/deploy_runs/t4_k3_ho \
    --arm bicubic=bicubic \
    --pair dck3,dcr3 --pair dck3,flashvsr_plus --pair dck3,infvsr \
    > ../logs/judge_k3.log 2>&1
  echo "judge exit=$? $(date +%H:%M:%S)"
fi
python -u eval_range.py ../results/range_branch/r5_k3.pt ../results/range_branch/r6_k5.pt \
  ../results/range_branch/r6_xl_k3.pt --data $D --device npu:0 2>&1 \
  | grep -viE 'owner|warnings.warn|x = torch'
