#!/usr/bin/env bash
# Does an MTF conditioned on the ACTUAL blur kernel beat a one-hot on deblur?
#
# This is the one axis where the two signals should separate most cleanly, and it is the axis where the
# published numbers say MTF loses hardest (-0.144 dB, the worst of the five operators). The diagnosis in
# results/omni/deblur_strata_smoke.json is that MTF was never given the kernel: mtf_cond caches on the
# operator string, every deblur sample passes the bare 'deblur', and operator_desc falls back to
# sigma=1.75. So a 17-dim measurement of U(D(.)) collapsed to one constant across sigma 1.0-2.5, i.e. a
# higher-dimensional one-hot plus a prior that is wrong for every kernel but one.
#
# That the checkpoint cannot simply be re-conditioned at inference is already measured: feeding each
# sample its true descriptor to the SAME weights LOSES 0.41 dB overall and 1.10 dB in the bin furthest
# from 1.75. The network bound that constant vector as the identity of 'deblur'. So the question needs a
# retrain, which is what this is.
#
# Single variable: --mtf-persample. Same operators, same K=3, same iters, batch, crop, width, depth,
# val-limit and seed as omni_cond.sh produced omni_mtf_k3 with. The existing omni_mtf_k3 and
# omni_onehot_k3 are the two controls and are NOT recomputed.
#
# PRE-REGISTERED:
#   1. mtf_persample must beat mtf_k3 on deblur. If it does not, the MTF idea genuinely does not help on
#      a smooth rolloff and the deblur column should be reported as a negative result, not an
#      implementation note.
#   2. The separation must be largest in the sigma bins furthest from 1.75, because those are the samples
#      whose true descriptor differs most from the constant the old arm received.
#   3. The four non-deblur operators must stay within seed noise of omni_mtf_k3. They share no kernel
#      parameter, so the flag cannot legitimately move them; if it does, something other than the
#      conditioning changed.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
CARD=$1; TAG=${2:-omni_mtf_k3_ps}; SEED=${3:-0}
export ASCEND_RT_VISIBLE_DEVICES=$CARD
source env.sh
cd code
python -u omni_stage1.py --tag "$TAG" --scales 2,4,8,deblur,denoise --cond mtf --mtf-persample --k 3 \
  --iters 12000 --log-every 500 --batch 30 --crop 48 --ch 96 --blocks 12 \
  --val-limit 24 --seed "$SEED" --device npu:0
echo "EXIT=$? $(date '+%F %T')"
