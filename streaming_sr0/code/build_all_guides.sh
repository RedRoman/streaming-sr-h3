#!/bin/bash
# Build guides for all 32 events with 8 parallel CPU workers.
cd /cache/zihaohe/streaming_sr0
rm -rf data/step2_guides; : > logs/step2_build.log
SEQS=$(python3 -c "import json;print(' '.join(e['sequence'] for e in json.load(open('data/events/selection.json'))['events']))")
i=0; for s in $SEQS; do echo "$s"; done | split -n r/8 - /tmp/ssr2_seqs_
for f in /tmp/ssr2_seqs_*; do
  (cd code && OMP_NUM_THREADS=4 /cache/zihaohe/streaming_sr0/venv/bin/python -u build_memory_guides.py --only $(cat $f) 2>&1 | grep -v -i warn >> ../logs/step2_build.log) &
done
wait
rm -f /tmp/ssr2_seqs_*
echo BUILD_ALL_DONE >> logs/step2_build.log
