#!/usr/bin/env bash
# The x6 hold-out, on the x6 eval set -- which is the part the earlier passes got wrong.
#
# t2_mtf_s6 now holds two different experiments: 24 clips from 11-01 read data/real_eval/heldout_s6
# (LR 138x80, a genuine x6 observation) and 53 clips added today read data/real_eval/heldout (LR 208x120,
# x4) because fill_heldout.sh set one EVAL_SET for all three tags. Their LPIPS differs by 0.0843, so the
# pooled 0.2198 is a mixture of two input scales and means nothing. The 53 are kept as what they actually
# are, a mis-conditioning control worth +0.0009 LPIPS against the right descriptor, i.e. nothing.
#
# This writes a clean tag on the correct set so the x6 number stands on one input scale.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export EVAL_SET=data/real_eval/heldout_s6
run () {
  local card=$1 part=$2 clips=$3
  (
    export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/heldout_s6
    source ../env.sh
    python run_deploy.py --tag t2_mtf_s6_ho --dc rphi --guide y \
      --rphi ../results/omni/omni_mtf_k3.pt --op 6 --turbo-steps 2 --merge-lora --clips "$clips" \
      > "../logs/s6true_p${part}.log" 2>&1
    echo "s6_ho part${part} card${card} exit=$? $(date '+%F %T')" >> ../logs/fill_s6_true.status
  ) &
}

cd /cache/zihaohe/streaming_sr0/code
run 0 0 "1jyM3quZ0h0,1xRoPmw1yCY,24cmo2fEQo8,259QGi4PNAc,25TiEJB46Qg,266IiZnynyA,277uoZ9cmB0,2AeLh_rUavc,2Cb9U8poEEI,43dQMHGLDHU"
run 1 1 "4BVEb9P1YxQ,4GlH0-KhInI,4N_pgE-uYYA,4PYNzqXudtk,4QDyavpkq1w,4Wu5YyU1j28,4ZTPDmqaXok,4aN_Vvn1lBY,66B46E_hrrk,69tcopJWwrY"
run 2 2 "6Si3eGkmmy8,6WX4cEj7sTc,6kQcdo770RM,6m4k0yEqvsw,6zprs4dvSC8,78R6KgsSPRk,7DIPyJB4osY,8kIMIXYRLRo,8kx5ST65Fog,8nXNlU9Ghbw"
run 3 3 "8sqewUGPr-w,9L3EMXLufVg,9Qle4fgMbzk,9TA7XP_PHm0,9_XnJpdB-1M,9kLbnqoMy3s,9xD9JdbVMn8,A1ak2lKtjTk,E0OUeIVIHAE,E1wV4qdUVSk"
run 4 4 "EptdhC17avY,EvDGBUbCBRs,Gf8F2tBCaPk,Gz8RlHf3Czs,H84ygnxBXJ4,HGtnqoU1kdc,HbijjxLdyUM,HrwpdRe7meM,HsL4sGuLkv0,JOVfl5fhnQ0"
run 5 5 "Jc73va5hsQU,JjC4SzZwjJg,JnL40ryM1cM,JwWoO-dafDc,K9sMmkRbAFs,KL0TWPmezrY,KOC3GuBco5E,KOb-tRHYK68,KZMg0ccZBjM,MElc-0XqoSw"
run 6 6 "MOv0k6GCsB0,MWJZVSUJHVE,MXffROjfCTs,Ms4usE_5R3o,N-IvW6J2e80,N5yG0Zt6Uoo,PG1nKnvHjYI,PjjojjUn0bk,PxjeXDgT6tY,PyNvYFrM9p4"
run 7 7 "Q4rMvvsalWs,QDRL3DxwFGM,QGd3hjZBFEI,QIjShFhrp6w,SZ7KaU9P4v0,VGUn3tueVLE,VMFy2UAATV4"
wait
echo "ALL DONE $(date "+%F %T")" >> ../logs/fill_s6_true.status
