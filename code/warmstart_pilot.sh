#!/usr/bin/env bash
# Can one step from a warm start match two steps from noise?
#
# The waste being removed: the sampler starts from zeros, generates a full frame, and then innerDC
# throws the range component away and substitutes R_phi(y). So DiT capacity goes into rebuilding
# something the observation already determines. warm_start hands it over at the beginning instead.
#
# Why it is a latency change and not a quality change: sampling is 68% of compute and the step count is
# the only lever on it that does not require a new module. If one step from a warm start holds quality,
# the sampler cost halves with nothing added -- R_phi already runs, because it was always the projection
# target.
#
# The honest cost: these sigmas are NOT on the shift-12 family the turbo LoRA was distilled on. The
# shift exists to concentrate steps at high noise, and routing 0.5 through it returns 0.923, which
# leaves the warm start 8% weight and defeats the purpose. So the warm arms run off-distribution for
# the LoRA, and that is part of what is being measured rather than a detail to hide.
set -u
R=/cache/zihaohe/streaming_sr0
: > $R/logs/warmstart.status
run () {
  local card=$1 tag=$2; shift 2
  ( export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/heldout
    source $R/env.sh
    cd $R/code
    python run_deploy.py --tag "$tag" --dc rphi --guide y \
      --rphi ../results/range_branch/r5_k3.pt --merge-lora "$@" \
      > "$R/logs/${tag}.log" 2>&1
    echo "$tag exit=$? $(date '+%F %T')" >> $R/logs/warmstart.status ) &
}
# 8 clips each, enough to see direction without burning the set
CL=$(python3 -c "
import json
D='$R/data/real_eval/heldout'
cl=[c for c in json.load(open(D+'/manifest.json'))['clips']
    if json.load(open(f'{D}/{c}/meta.json'))['degradation'].get('kind')=='realistic'][:8]
print(','.join(cl))")
run 0 ws_cold2  --turbo-steps 2                                        --clips "$CL"
run 1 ws_warm1  --turbo-steps 1 --warm-start --sigma-start 0.5         --clips "$CL"
run 2 ws_warm1b --turbo-steps 1 --warm-start --sigma-start 0.3         --clips "$CL"
run 3 ws_warm2  --turbo-steps 2 --warm-start --sigma-start 0.5         --clips "$CL"
run 4 ws_cold1  --turbo-steps 1                                        --clips "$CL"
wait
( export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/heldout
  source $R/env.sh
  cd $R/code
  python -u deploy_eval.py --procs 8 eval --name judge_warmstart \
    --arm shipped_cold2=../results/deploy_runs/ws_cold2 \
    --arm cold1=../results/deploy_runs/ws_cold1 \
    --arm warm1_s05=../results/deploy_runs/ws_warm1 \
    --arm warm1_s03=../results/deploy_runs/ws_warm1b \
    --arm warm2_s05=../results/deploy_runs/ws_warm2 \
    --arm bicubic=bicubic > "$R/logs/judge_warmstart.log" 2>&1
  echo "judge exit=$? $(date '+%F %T')" >> $R/logs/warmstart.status )
echo "DONE $(date '+%F %T')" >> $R/logs/warmstart.status
