#!/usr/bin/env bash
# The suppression side of the gate, which has never been measured at a strength that bites.
#
# gain_oracle.py (96 clips, 1520 frames) says the least-squares optimal rescaling of the null band is
# BELOW one, not above: globally 0.613, per site a median of 0.224 with the 10th percentile negative.
# Every sweep so far went the other way -- judge_sharp used gain 1.3/1.6/1.9 and was monotonically worse,
# and the only completed gate arms (q0005_*) used s0 = 0.005, which makes clip((s/s0)^p,0,1) ~ 1
# everywhere: they move null energy 0.6109 -> 0.6047, about 1% of the 40-60% the oracle asks for. Their
# direction was right on nearly every clip (flicker better on 87/88, warp 88/88, seam 86/86) but the
# magnitudes are 0.0002 in LPIPS, so they test nothing.
#
# Two things are swept here, both deployable (the gate reads only U(R_phi(y)), never the truth):
#   global    a constant gain < 1, the oracle's own global optimum and its neighbours
#   gated     a gate strong enough to matter (s0 0.02 / 0.05) with and without a sub-unit gain
# Expect PSNR up and LPIPS possibly down: the oracle minimises MSE and MSE prefers blur, which is why
# bicubic sits at 24.24 dB / 0.4336. The number that decides whether suppression is real is null_cos,
# where the oracle's per-site gain buys 0.484 -> 0.592 -- a direction gain, not just shrinkage.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
export EVAL_SET=data/real_eval/heldout
export ASCEND_RT_VISIBLE_DEVICES=0
source "$R/env.sh"
SRC=../results/deploy_runs/t4_plain2
CK=../results/range_branch/r5_k3.pt
run () {  # tag s0 p gain
  python -u deploy_eval.py --procs 12 quota --src "$SRC" --ckpt "$CK" --keep-range \
    --tag "$1" --s0 "$2" --p "$3" --gain "$4" > "$R/logs/sup_$1.log" 2>&1
  echo "sup_$1 s0=$2 p=$3 gain=$4 exit=$? $(date '+%F %T')" >> "$R/logs/suppress.status"
}
# global scalar: s0 tiny so the gate is the identity and only the gain acts
run sup_g060 0.0000001 1.0 0.60
run sup_g075 0.0000001 1.0 0.75
run sup_g090 0.0000001 1.0 0.90
# gates that actually bite
run sup_s02_g10 0.02 1.0 1.00
run sup_s05_g10 0.05 1.0 1.00
run sup_s02_g075 0.02 1.0 0.75
echo "suppress sweep done $(date '+%F %T')" >> "$R/logs/suppress.status"
