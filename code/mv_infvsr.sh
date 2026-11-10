#!/usr/bin/env bash
# InfVSR on MVSR4x. The overnight run picked baseline_infvsr.py, which is the faithful CUDA copy and
# aborts on Ascend with "Torch not compiled with CUDA enabled"; run_infvsr_npu.py is the entry point
# that imports the transfer_to_npu shim first and produced the held-out numbers.
# INFVSR_OUT_SUFFIX keeps this out of the heldout tree -- one directory holding two input scales is the
# mistake that made an earlier x6 table meaningless.
set -u
R=/cache/zihaohe/streaming_sr0
for S in 0 1 2 3; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$S
    export EVAL_SET=data/real_eval/mvsr4x
    export INFVSR_OUT_SUFFIX=_mvsr4x
    export SHARD=$S NSHARDS=4
    source $R/env.sh
    cd $R/code
    python run_infvsr_npu.py > "$R/logs/mv_infvsr_p$S.log" 2>&1
    echo "infvsr shard$S exit=$? $(date '+%F %T')" >> $R/logs/mv_infvsr.status ) &
done
wait
n=$(ls -d $R/results/baseline_infvsr_mvsr4x/*/ 2>/dev/null | wc -l)
echo "infvsr_mvsr4x dirs: $n" >> $R/logs/mv_infvsr.status
# rejudge with every arm now present
( export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/mvsr4x
  source $R/env.sh
  cd $R/code
  python -u deploy_eval.py --procs 10 eval --name judge_mvsr4x_all \
    --arm ours=../results/deploy_runs/mv_r5k3 \
    --arm bicubic=bicubic \
    --arm flashvsr_plus=../results/baseline_flashvsr_plus_mvsr4x \
    --arm infvsr=../results/baseline_infvsr_mvsr4x \
    --pair ours,flashvsr_plus > "$R/logs/mv_judge2.log" 2>&1
  echo "rejudge exit=$? $(date '+%F %T')" >> $R/logs/mv_infvsr.status ) 
