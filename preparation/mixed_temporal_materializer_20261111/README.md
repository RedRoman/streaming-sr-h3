# One-source mixed/temporal CPU engineering smoke

The completed attempt used only existing train source `2BdYaGuVAKo`, one centered 96-frame RGB target, and the six profiles declared by the frozen plan. No candidate from the unaudited 46-source queue, model inference, training, NPU job, or split reassignment is involved.

The preparation is complete: 15 CPU tests passed; the real-source smoke exited zero in 119.63 materialization seconds. Actual decoding yielded 122 source-filter frames and exactly 96 target frames. All six profiles passed two exact pixel-SHA replays. Five H264 profiles had only keyframe 0 in both replays; the remaining profile is independent per-frame JPEG. Static order controls and native smooth/abrupt controls shared identical unit-noise arrays. Independent verification reloaded all seven saved NPZ files and checked exact uint8 RGB pixels, shape, file SHA and common source/target metadata. Conservative parent-plus-child peak RSS was 486,379,520 bytes, below 768 MiB.

The actual target timestamps are PTS 1 through 96 at timebase 1/24; the first exported frame is not forced to PTS 0. The accurately sought source's first presentation timestamp is 3.03637 seconds, while its prescribed centered seek is 3.011 seconds. The frozen `fps=24:round=near` grid begins at PTS 1 for this source. All original and output timestamps are in the receipt. Codec observations reindex the same ordered 96-frame sequence to PTS 0 through 95; no global source-frame ordinal is fabricated.

Files:

- `execution_contract.json`: the root-confirmed executable conventions, including six-digit frame indices, canonical UTF-8 target-spec SHA as window key, RGB-before-floor-center-crop, explicit timestamp taps, source isolation, replay and memory rules.
- `snapshot/code/materialize_mixed_temporal_pilot.py`: frozen runtime copy; the editable repository original is `code/materialize_mixed_temporal_pilot.py`.
- `snapshot/code/tests/test_mixed_temporal_materializer.py`: frozen 15-test suite, including small actual codec and nonzero-seek FFmpeg checks.
- `snapshot/configs/openh3_20261111/mixed_temporal_data_plan_20261111.json`: frozen plan. **Do not update this copy when changing the main plan's status.**
- `snapshot/results/takeover_20261111/materialized/{train,dev}_materialized.jsonl`: existing source-role manifests, bound by their complete file SHA.
- `preflight.py`: checks frozen snapshot SHA, fresh source bytes, source geometry/duration, new output and disk space. It records actual codec-library binary SHA without decoding source RGB.
- `run_smoke.py`: exactly one attempt, internal 600-second budget plus outer 630-second timeout; refuses any existing launch receipt or output and never retries.
- `inspection.json`: local no-source-I/O inspection. The actual remote preflight, launch and saved-pixel verification receipts are archived under `results/takeover_20261111/mixed_temporal_single_source_smoke_completed/`.

The one authorized remote invocation was:

```sh
/cache/zihaohe/streaming_sr0/venv/bin/python \
  /cache/zihaohe/omni_openh3_20261111/preparation/mixed_temporal_materializer_20261111/run_smoke.py
```

The wrapper records its exact bounded command and frozen inputs in `launch_receipt.json`. Its output is `/cache/zihaohe/omni_openh3_20261111/runs/mixed_temporal_single_source_smoke_20261111a`. It is already completed and must not be rerun. Target and observation NPZs remain remote; only the 105,899-byte evidence archive was downloaded and all its member hashes were verified.

This is engineering acceptance only. Natural source cuts remain unknown; the separate two-source scene-cut diagnostic and segmented-codec reset experiment were not run. Smooth and abrupt schedules share noise but have different severity durations, so they are not equal-difficulty ablations. `mixed_native` requires a separately versioned loader and training contract; it must not be mapped silently into the existing four-task one-hots. No pristine-target, independent-test, or restoration-quality claim follows from this smoke.
