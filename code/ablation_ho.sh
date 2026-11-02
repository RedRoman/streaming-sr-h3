#!/usr/bin/env bash
# Produce the missing cell of the 2x2 and then score all four with one judge pass.
#
# Three cells already exist: the two sampling runs (deploy_runs/t4_dcy_ho, deploy_runs/t4_r2_ho) and the
# gated version of the R_phi one (deploy/t4r2_ho_final). The fourth -- the gate applied on top of dc=y --
# has to be generated, otherwise the gate's effect can only be read at one level of the other factor and
# an interaction would be invisible. Gate parameters are copied from t4r2_ho_final's own stream_result.json
# (s0=0.02, p=1.0, sharp=0.0, gain=1.15, ema=1.0) so the two gated cells differ only in their source.
set -eu
cd /cache/zihaohe/streaming_sr0
export ASCEND_RT_VISIBLE_DEVICES=0
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code

while pgrep -f 'run_deploy.py --tag t4_dcy_ho' > /dev/null; do sleep 60; done
n_src=$(ls ../results/deploy_runs/t4_dcy_ho | wc -l)
echo "dc=y sampling dirs: $n_src"
test "$n_src" -ge 96 || { echo "dc=y sampling incomplete, refusing to evaluate a partial arm" >&2; exit 1; }

python -u deploy_eval.py --procs 12 quota \
  --src ../results/deploy_runs/t4_dcy_ho --tag ho_dcy_q \
  --ckpt ../results/range_branch/r2_hs.pt \
  --s0 0.02 --p 1.0 --sharp 0.0 --gain 1.15 \
  > ../logs/ho_dcy_q.log 2>&1
echo "quota done: $(ls ../results/deploy/ho_dcy_q | wc -l) clips"

python -u deploy_eval.py --procs 12 eval --name judge_abl \
  --arm dcy=../results/deploy_runs/t4_dcy_ho \
  --arm dcy_q=../results/deploy/ho_dcy_q \
  --arm dcr=../results/deploy_runs/t4_r2_ho \
  --arm dcr_q=../results/deploy/t4r2_ho_final \
  --arm bicubic=bicubic \
  --pair dcy,dcr --pair dcy_q,dcr_q --pair dcy,dcy_q --pair dcr,dcr_q \
  > ../logs/judge_abl.log 2>&1
tail -2 ../logs/judge_abl.log
python -u ablation_table.py
