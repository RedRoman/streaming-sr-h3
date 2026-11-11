# Native restoration baseline preparation, 2026-11-11

The sibling readiness JSON is the acquisition and protocol record:
`../../configs/openh3_20261111/native_baseline_readiness_20261111.json`.
The official repositories and three acquired weights are local, pinned, and
ignored by the main repository. Neither baseline has completed a model forward
pass in this preparation task. No training environment was changed.

## Prepared source and assets

| Method | Official commit | Weight |
|---|---|---|
| FaverNet TUDCommon | `14769f22685f4850c5e203d493c99ad2eba5b68b` | `weights/FaverNet_TUDCommon.pth` |
| AverNet | `c4eb019f71fe6a2ccb6807dca92451fd8d8b8cd3` | `weights/AverNet_final.pth` |

`weights/spynet_sintel_final-3d2a1287.pth` is also acquired from the shared
official Google Drive release. The complete model files contain SPyNet key
strings; the runner additionally requires their strict tensor-state load.

## First execution protocol

Use a separate CPU/CUDA environment with the official requirements and MMCV
compiled ops. Do not install these conflicting requirements into the active
H3/R_phi environment. FaverNet pins torch 2.8.0, torchvision 0.23.0 and MMCV
2.2.0. AverNet pins torch 2.3.1, does not pin MMCV, and its package setup generates
`basicsr/version.py`. Its README's `requirement.txt` command is a typo: the file
is `requirements.txt`. In a fresh baseline environment, use the official
requirements and `python setup.py develop` for AverNet.

The adapter consumes a materialized observation NPZ's `frames` only (RGB uint8
THWC), checks the manifest's observation pixel SHA256, loads the exact released
checkpoint strictly, and writes lossless float32 RGB. It performs no resizing,
task conditioning, GT read, tiling, temporal truncation or metric calculation.
Official tail-flip padding and crop are retained. Both methods receive future
frames and must be labelled offline. This initial protocol accepts 2–100 frames
and H,W >=256; longer sequences require separately validated caching.

Run from the project root, substituting a native task observation and its
manifest SHA; use a new output directory for every attempt:

```sh
python third_party/native_video_baselines_20261111/run_lossless_rgb.py \
  --method favernet --input-npz /absolute/path/to/observation.npz \
  --expected-pixel-sha256 MANIFEST_OBSERVATION_PIXEL_SHA256 \
  --task denoise --device cpu --inspect-only --out /new/path/preflight
```

For the first actual forward, remove `--inspect-only` and choose a different
`--out`. Use `--method avernet` for the second method and `--device cuda:0` in
an isolated CUDA reference environment. The float32 output NPZ has `rgb_raw`
and `rgb` (clamped to [0,1]) keys; only `rgb` is intended for shared RGB metrics.
All frames are restored. Apply the identical evaluation frame stride later.
The one forward time is diagnostic only, with no warmup or speed claim.

The next validation gate is strict checkpoint load plus a 2–3 frame forward,
then comparison with each official full-clip test path on identical RGB input.
Only after this reference passes should the same operators/model be checked on
Ascend. MMCV 2.2.0 **has** an NPU deformable-convolution branch, but ABI/version,
offset-group numerical parity, grid sampling, FaverNet complex FFT, and cache
device handling remain untested. This runner intentionally has no NPU option.

Neither method is treated as native x4 SR. Native denoise/deblur/compression,
identity and near-clean controls are separate records; no paper-score or SOTA
claim follows from this local development protocol.

## Checks performed locally

```sh
python3 -m unittest discover -s third_party/native_video_baselines_20261111 -p 'test_lossless_rgb.py' -v
```

These tests cover lossless transport, observation-only loading and rejection
of incorrect hashes/dtypes/layouts/geometry. They do not test model execution.
