#!/usr/bin/env bash
# Validate dcproj on the two real-degradation sets before it is allowed into the shipped config.
# Everything measured so far is on heldout, whose degradation is synthetic (45/77 h264, 23/77 jpeg), and
# this project has already had one conclusion that held on heldout and reversed on a real set.
#
# Waits for HBM: all eight cards currently carry 27-31 GB of colleagues' jobs and our run peaks at
# 47.7 GB, so it cannot start until one card drops below about 16 GB.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
pick_card () {
  for c in 0 1 2 3 4 5 6 7; do
    u=$(npu-smi info -i "$c" 2>/dev/null | grep -oE '[0-9]+/ 65536' | head -1 | cut -d/ -f1 | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 16000 ] && { echo "$c"; return 0; }
  done
  return 1
}
CARD=""
while [ -z "$CARD" ]; do CARD=$(pick_card) || sleep 300; done
echo "got card $CARD $(date '+%F %T')" >> "$R/logs/queue.status"

for SET in realvsr mvsr4x; do
  (
    export ASCEND_RT_VISIBLE_DEVICES=$CARD
    export EVAL_SET="data/real_eval/$SET"
    source "$R/env.sh"
    python run_deploy.py --tag "dcproj_$SET" --dc rphi --guide y \
      --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora --dc-project \
      > "$R/logs/dcproj_$SET.log" 2>&1
    echo "dcproj_$SET card$CARD exit=$? $(date '+%F %T')" >> "$R/logs/dcproj_real.status"
  )
  export ASCEND_RT_VISIBLE_DEVICES=0
  source "$R/env.sh"
  EVAL_SET="data/real_eval/$SET" python -u deploy_eval.py --procs 10 quota \
    --src "../results/deploy_runs/dcproj_$SET" --ckpt ../results/range_branch/r5_k3.pt --keep-range \
    --tag "dcproj_${SET}_g115" --s0 0.0000001 --p 1.0 --gain 1.15 \
    > "$R/logs/dcproj_${SET}_g115.log" 2>&1
  echo "dcproj_${SET}_g115 exit=$? $(date '+%F %T')" >> "$R/logs/dcproj_real.status"
done

export ASCEND_RT_VISIBLE_DEVICES=0
source "$R/env.sh"
EVAL_SET=data/real_eval/realvsr python -u deploy_eval.py --procs 10 eval --name judge_dcproj_realvsr \
  --arm shipped=../results/deploy_runs/rvsr_plain2 \
  --arm dcproj=../results/deploy_runs/dcproj_realvsr \
  --arm dcproj_g115=../results/deploy/dcproj_realvsr_g115 \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus_realvsr \
  --arm bicubic=bicubic \
  --pair dcproj,shipped --pair dcproj_g115,shipped --pair dcproj_g115,flashvsr_plus \
  > "$R/logs/judge_dcproj_realvsr.log" 2>&1
echo "judge_dcproj_realvsr exit=$? $(date '+%F %T')" >> "$R/logs/dcproj_real.status"
