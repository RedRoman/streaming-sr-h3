#!/usr/bin/env bash
# One-hot vs MTF conditioning, matched in everything else. The novelty claim, made falsifiable.
#
# Both arms see the same five operators, the same K=3 causal stack, the same data, the same seed. The only
# difference is what the FiLM path is told:
#
#   onehot  5 slots + severity   the operator's NAME, as AirNet/PromptIR/MambaOFR and our own earlier omni
#   mtf     16 bands + severity  the operator MEASURED: the radial MTF of U(D(.)) on a fixed noise probe
#
# Neither is an oracle. D is given in the non-blind setting -- that is forced on us, since eight attempts
# to recover it from one frame top out at AUC 0.736 against a 0.61-0.68 SUPERVISED ceiling -- so both
# signals are equally available at deployment and cost the same at inference.
#
# What separates them is expressiveness, and the probe already shows it: x6 lands strictly between x4 and
# x8 in every band (band 0: 1.00 > 0.98 > 0.88, band 1: 0.83 > 0.44 > 0.17, band 2: 0.32 > 0.04 > 0.00),
# while a one-hot over OPS has no slot for x6 at all and now says so by name instead of scoring it.
#
# PRE-REGISTERED, before any number is read:
#   1. On the five TRAINED operators the two arms should be within the seed noise of each other. If mtf
#      wins there, the gain is capacity from a wider conditioning vector, not expressiveness, and the
#      claim is not supported by this experiment.
#   2. On x6, HELD OUT from both, mtf must show a positive gain and onehot must be unable to run. That
#      asymmetry -- not a dB delta -- is the result.
#   3. K=3 must not cost anything relative to the published K=1 omni numbers (+2.27/+2.53/+1.93 at
#      x2/x4/x8); the x4 line measured +0.31 dB from K=3 and the omni branch could not collect it before.
# A failure of 1 invalidates the framing of 2. It is checked first.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
CARD=$1; COND=$2; TAG=$3; SEED=${4:-0}
export ASCEND_RT_VISIBLE_DEVICES=$CARD
source env.sh
cd code
# crop 48: the x8 pairs are 60x104, and a per-operator crop would change the gradient scale per operator
python -u omni_stage1.py --tag "$TAG" --scales 2,4,8,deblur,denoise --cond "$COND" --k 3 \
  --iters 12000 --log-every 500 --batch 30 --crop 48 --ch 96 --blocks 12 \
  --val-limit 24 --seed "$SEED" --device npu:0
echo "EXIT=$? $(date '+%F %T')"
