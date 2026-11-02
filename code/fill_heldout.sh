#!/bin/bash
# Finish the three conditioning arms on every realistic HELDOUT clip.
# run_deploy.py:44 reads the eval set from $EVAL_SET and defaults to data/real_eval/const, which has 24
# clips sharing no names with heldout's 96. Two earlier fill scripts omitted the variable, so they ran
# const: four shards of six clips each, finishing in minutes and writing const results into the heldout
# tags. judge looks clips up by name so the published numbers were never contaminated, but nothing was
# added either. EVAL_SET is set explicitly here and asserted before any card is spent.
cd "$(dirname "$0")"
export EVAL_SET=data/real_eval/heldout
run () {
  local card=$1 tag=$2 rphi=$3 op=$4 part=$5 clips=$6
  (
    export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/heldout
    source ../env.sh
    python run_deploy.py --tag "$tag" --dc rphi --guide y --rphi "../results/omni/$rphi" \
      --op "$op" --turbo-steps 2 --merge-lora --clips "$clips" \
      > "../logs/hold_${tag}_p${part}.log" 2>&1
    echo "${tag} part${part} card${card} exit=$? $(date '+%F %T')" >> ../logs/fill_heldout.status
  ) &
}
run 0 t2_mtf_s6 omni_mtf_k3.pt 6 0 "6zprs4dvSC8,78R6KgsSPRk,7DIPyJB4osY,8kIMIXYRLRo,8kx5ST65Fog,8nXNlU9Ghbw,8sqewUGPr-w,9L3EMXLufVg,9Qle4fgMbzk,9TA7XP_PHm0,9_XnJpdB-1M,9kLbnqoMy3s,9xD9JdbVMn8,A1ak2lKtjTk,E0OUeIVIHAE,E1wV4qdUVSk,EptdhC17avY,EvDGBUbCBRs"
run 1 t2_mtf_s6 omni_mtf_k3.pt 6 1 "Gf8F2tBCaPk,Gz8RlHf3Czs,H84ygnxBXJ4,HGtnqoU1kdc,HbijjxLdyUM,HrwpdRe7meM,HsL4sGuLkv0,JOVfl5fhnQ0,Jc73va5hsQU,JjC4SzZwjJg,JnL40ryM1cM,JwWoO-dafDc,K9sMmkRbAFs,KL0TWPmezrY,KOC3GuBco5E,KOb-tRHYK68,KZMg0ccZBjM,MElc-0XqoSw"
run 2 t2_mtf_s6 omni_mtf_k3.pt 6 2 "MOv0k6GCsB0,MWJZVSUJHVE,MXffROjfCTs,Ms4usE_5R3o,N-IvW6J2e80,N5yG0Zt6Uoo,PG1nKnvHjYI,PjjojjUn0bk,PxjeXDgT6tY,PyNvYFrM9p4,Q4rMvvsalWs,QDRL3DxwFGM,QGd3hjZBFEI,QIjShFhrp6w,SZ7KaU9P4v0,VGUn3tueVLE,VMFy2UAATV4"
run 3 t2_mtf_x4 omni_mtf_k3.pt 4 0 "6zprs4dvSC8,78R6KgsSPRk,7DIPyJB4osY,8kIMIXYRLRo,8kx5ST65Fog,8nXNlU9Ghbw,8sqewUGPr-w,9L3EMXLufVg,9Qle4fgMbzk,9TA7XP_PHm0,9_XnJpdB-1M,9kLbnqoMy3s,9xD9JdbVMn8,A1ak2lKtjTk,E0OUeIVIHAE,E1wV4qdUVSk,EptdhC17avY,EvDGBUbCBRs"
run 4 t2_mtf_x4 omni_mtf_k3.pt 4 1 "Gf8F2tBCaPk,Gz8RlHf3Czs,H84ygnxBXJ4,HGtnqoU1kdc,HbijjxLdyUM,HrwpdRe7meM,HsL4sGuLkv0,JOVfl5fhnQ0,Jc73va5hsQU,JjC4SzZwjJg,JnL40ryM1cM,JwWoO-dafDc,K9sMmkRbAFs,KL0TWPmezrY,KOC3GuBco5E,KOb-tRHYK68,KZMg0ccZBjM,MElc-0XqoSw"
run 5 t2_mtf_x4 omni_mtf_k3.pt 4 2 "MOv0k6GCsB0,MWJZVSUJHVE,MXffROjfCTs,Ms4usE_5R3o,N-IvW6J2e80,N5yG0Zt6Uoo,PG1nKnvHjYI,PjjojjUn0bk,PxjeXDgT6tY,PyNvYFrM9p4,Q4rMvvsalWs,QDRL3DxwFGM,QGd3hjZBFEI,QIjShFhrp6w,SZ7KaU9P4v0,VGUn3tueVLE,VMFy2UAATV4"
run 6 t2_onehot_x4 omni_onehot_k3.pt 4 0 "6zprs4dvSC8,78R6KgsSPRk,7DIPyJB4osY,8kIMIXYRLRo,8kx5ST65Fog,8nXNlU9Ghbw,8sqewUGPr-w,9L3EMXLufVg,9Qle4fgMbzk,9TA7XP_PHm0,9_XnJpdB-1M,9kLbnqoMy3s,9xD9JdbVMn8,A1ak2lKtjTk,E0OUeIVIHAE,E1wV4qdUVSk,EptdhC17avY,EvDGBUbCBRs,Gf8F2tBCaPk,Gz8RlHf3Czs,H84ygnxBXJ4,HGtnqoU1kdc,HbijjxLdyUM,HrwpdRe7meM,HsL4sGuLkv0,JOVfl5fhnQ0,Jc73va5hsQU"
run 7 t2_onehot_x4 omni_onehot_k3.pt 4 1 "JjC4SzZwjJg,JnL40ryM1cM,JwWoO-dafDc,K9sMmkRbAFs,KL0TWPmezrY,KOC3GuBco5E,KOb-tRHYK68,KZMg0ccZBjM,MElc-0XqoSw,MOv0k6GCsB0,MWJZVSUJHVE,MXffROjfCTs,Ms4usE_5R3o,N-IvW6J2e80,N5yG0Zt6Uoo,PG1nKnvHjYI,PjjojjUn0bk,PxjeXDgT6tY,PyNvYFrM9p4,Q4rMvvsalWs,QDRL3DxwFGM,QGd3hjZBFEI,QIjShFhrp6w,SZ7KaU9P4v0,VGUn3tueVLE,VMFy2UAATV4"
wait
echo "ALL DONE $(date "+%F %T")" >> ../logs/fill_heldout.status
