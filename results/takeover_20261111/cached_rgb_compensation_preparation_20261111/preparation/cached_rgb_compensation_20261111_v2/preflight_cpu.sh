#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TORCH_HOME=/cache/zihaohe/omni_openh3_20261111/torch_home TMPDIR=/cache/zihaohe/omni_openh3_20261111/tmp SSR0=/cache/zihaohe/omni_openh3_20261111
cd /cache/zihaohe/omni_openh3_20261111
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
ulimit -v 25165824
PYTHONPATH=preparation/cached_rgb_compensation_20261111_v2/code venv/bin/python -m unittest discover -s preparation/cached_rgb_compensation_20261111_v2/code/tests -p test_cached_rgb_compensation.py -v > preparation/cached_rgb_compensation_20261111_v2/cpu_tests.log 2>&1
venv/bin/python preparation/cached_rgb_compensation_20261111_v2/code/diagnose_cached_rgb_compensation.py --ordinary-metrics runs/fixed_h3_rphi_legacy_smoke_20261111a/eval/metrics.json --ordinary-guides runs/fixed_h3_rphi_legacy_smoke_20261111a/guides/manifest.json --controls-metrics runs/fixed_h3_rphi_legacy_controls_smoke_20261111a/eval/metrics.json --controls-guides runs/fixed_h3_rphi_legacy_controls_smoke_20261111a/guides/manifest.json --dev-manifest datasets/omni_pilot_v1/dev_materialized.jsonl --pretrained-manifest configs/openh3_20261111/refiner_turbo_30794.json --vae-metrics runs/vae_roundtrip_legacy_smoke_20261111b/eval/metrics.json --threads 2 --inspect-only --out runs/cached_rgb_compensation_legacy_20261111a/eval > preparation/cached_rgb_compensation_20261111_v2/inspection.log 2>&1
