#!/bin/bash
# Teacher targets for one-step distillation, on 31984 where the training windows live (9.8 GB local; moving
# them to 30794 would cost more than the compute). Six shards on the eight idle cards.
# val first so a student can be smoke-trained before the long train split finishes.
cd /cache/zihaohe/streaming_sr0
for split in val train; do
  for sh in 0 1 2 3 4 5; do
    (export ASCEND_RT_VISIBLE_DEVICES=$sh; source env.sh; cd code;
     python -u gen_teacher.py --split $split --shard $sh --nshards 6 \
       > ../logs/teacher_${split}_s$sh.log 2>&1) &
    sleep 35
  done
  wait
  echo "$split done $(date +%H:%M)" >> logs/teacher_progress.txt
done
echo ALLDONE > logs/teacher.done
