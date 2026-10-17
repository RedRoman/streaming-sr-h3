#!/bin/bash
set -e
cd /cache/zihaohe/streaming_sr0
B=/tmp/ssr_bundle; rm -rf $B; mkdir -p $B/{code,results,data,docs}
cp -r code/*.py code/*.sh $B/code/ 2>/dev/null || true
rm -rf $B/code/__pycache__
cp tools/*.py tools/*.sh $B/code/ 2>/dev/null || true
cp *.md $B/docs/
cp results/*.json $B/results/ 2>/dev/null || true
for d in stage1_metrics stage1_panels vidgen_panels; do [ -d results/$d ] && cp -r results/$d $B/results/; done
cp -r data/vidgen_eval/selection.json data/vidgen_eval/manifest.json $B/data/ 2>/dev/null || true
mkdir -p $B/data/stage2_eval && cp data/stage2_eval/manifest.json $B/data/stage2_eval/ 2>/dev/null || true
mkdir -p $B/data/events && cp data/events/selection.json $B/data/events/ 2>/dev/null || true
# training curve + run configs, no weights
mkdir -p $B/results/stage1_lora && cp results/stage1_lora/train_log.jsonl results/stage1_lora/config.json $B/results/stage1_lora/ 2>/dev/null || true
find results -name 'runtime_*.json' -exec cp --parents {} $B/ \; 2>/dev/null || true
sha256sum results/stage1_lora/lora_step002000.safetensors > $B/results/stage1_lora/lora_step002000.sha256
du -sh $B; tar czf /tmp/ssr_bundle_30778.tgz -C /tmp ssr_bundle; ls -la /tmp/ssr_bundle_30778.tgz
