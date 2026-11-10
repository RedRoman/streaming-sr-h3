#!/usr/bin/env bash
# Complete t4_final so the one clean innerDC-target ablation runs on all 77 realistic clips.
# t4_dcy_ho (dc=y) and t4_final (dc=rphi) differ ONLY in the projection target -- same k1.pt estimator,
# same 4 steps -- which makes this the comparison that isolates the mechanism. The pairing used earlier
# (dcy against plain2) moved three variables at once: target, estimator and step count.
# 24 of 77 were present and two subset illusions were already caught today, so this fills the rest.
set -u
R=/cache/zihaohe/streaming_sr0
run () {
  local card=$1 clips=$2
  ( export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/heldout
    source $R/env.sh
    cd $R/code
    python run_deploy.py --tag t4_final --dc rphi --guide y \
      --rphi ../results/range_branch/k1.pt --turbo-steps 4 --merge-lora --clips "$clips" \
      > "$R/logs/fill_final_p${card}.log" 2>&1
    echo "p${card} exit=$? $(date '+%F %T')" >> $R/logs/fill_dctarget.status ) &
}
cd $R/code
run 0 "1jyM3quZ0h0,1xRoPmw1yCY,24cmo2fEQo8,259QGi4PNAc,25TiEJB46Qg,266IiZnynyA,277uoZ9cmB0,2AeLh_rUavc,2Cb9U8poEEI,43dQMHGLDHU"
run 1 "4BVEb9P1YxQ,4GlH0-KhInI,4N_pgE-uYYA,4PYNzqXudtk,4QDyavpkq1w,4Wu5YyU1j28,4ZTPDmqaXok,4aN_Vvn1lBY,66B46E_hrrk,69tcopJWwrY"
run 2 "6Si3eGkmmy8,6WX4cEj7sTc,6kQcdo770RM,6m4k0yEqvsw,6zprs4dvSC8,78R6KgsSPRk,7DIPyJB4osY,8kIMIXYRLRo,8kx5ST65Fog,8nXNlU9Ghbw"
run 3 "8sqewUGPr-w,9L3EMXLufVg,9Qle4fgMbzk,9TA7XP_PHm0,9_XnJpdB-1M,9kLbnqoMy3s,9xD9JdbVMn8,A1ak2lKtjTk,E0OUeIVIHAE,E1wV4qdUVSk"
run 4 "EptdhC17avY,EvDGBUbCBRs,Gf8F2tBCaPk,Gz8RlHf3Czs,H84ygnxBXJ4,HGtnqoU1kdc,HbijjxLdyUM,HrwpdRe7meM,HsL4sGuLkv0,JOVfl5fhnQ0"
run 5 "Jc73va5hsQU,JjC4SzZwjJg,JnL40ryM1cM,JwWoO-dafDc,K9sMmkRbAFs,KL0TWPmezrY,KOC3GuBco5E,KOb-tRHYK68,KZMg0ccZBjM,MElc-0XqoSw"
run 6 "MOv0k6GCsB0,MWJZVSUJHVE,MXffROjfCTs,Ms4usE_5R3o,N-IvW6J2e80,N5yG0Zt6Uoo,PG1nKnvHjYI,PjjojjUn0bk,PxjeXDgT6tY,PyNvYFrM9p4"
run 7 "Q4rMvvsalWs,QDRL3DxwFGM,QGd3hjZBFEI,QIjShFhrp6w,SZ7KaU9P4v0,VGUn3tueVLE,VMFy2UAATV4"
wait
echo "DONE $(date "+%F %T")" >> /cache/zihaohe/streaming_sr0/logs/fill_dctarget.status
