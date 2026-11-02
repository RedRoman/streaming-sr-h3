# Reproducing streaming-sr-h3

Everything here was measured on Ascend 910B3 (8 x 64 GB) under the private CANN 9.0.1 tree at
`$H3_BASE_ENV/Ascend/cann-9.0.1`. The system CANN on these images is missing the 910B operator package
and fails with 561103; do not try to use it. `source env.sh` after exporting
`ASCEND_RT_VISIBLE_DEVICES` sets every path (it refuses to run without that variable, on purpose).

| component | value |
|---|---|
| DiT | `minimax_h3_fl2va_pruned_bf16.safetensors`, 20.1B |
| Stage-1 LoRA | `results/stage1_lora/lora_step002000.safetensors`, rank 32, merged at inference |
| turbo LoRA | `assets/turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |
| ComfyUI | commit `5a665890ded5020c282f63c4ea2dd9e083e025ff` |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post2 |
| sampler | `turbo_sigmas(2, 12.0)` = `[1.0, 0.923077, 0.0]`, euler, cfg 1.0 |
| innerDC points | `POINTS = [28, 45, 48]` on the 49-step grid, remapped by `round(i * n_steps / 49)` |

Eval sets, by manifest hash, because three different ones share clip names:

| set | manifest sha256[:16] | LR | scale | reference |
|---|---|---|---|---|
| `data/real_eval/heldout` | `609c706aceed3614` | 208x120 | x4 | yes |
| `data/real_eval/heldout_s6` | `e5a01203c24972c3` | 138x80 | x6 | yes (HR byte-identical to heldout) |
| `data/real_eval/videolq` | `4f96c6ebc3fc632d` | 208x120 crop | assumed x4 | **no** |

---

## Main results and the command that produces each

Scored on the 77 realistic clips of `heldout` unless stated. LPIPS noise floor is 0.0009 at 2 steps,
measured by re-scoring one checkpoint, and no difference below it is reported as real.

**Baseline comparison** -- `code/judge_all7.sh`, reads `results/deploy/judge_all7.json`

| arm | LPIPS | PSNR | LPIPS\|GTrange | PSNR\|GTrange |
|---|---|---|---|---|
| InfVSR | 0.2344 | 23.28 | 0.1643 | 27.39 |
| FlashVSR | 0.2253 | 23.42 | 0.1697 | 26.35 |
| FlashVSR+ | 0.2072 | 24.23 | 0.1529 | 27.12 |
| ours 2-step | **0.1929** | **25.78** | **0.1340** | **28.34** |

**x6, an operator no arm trained on** -- `code/fill_s6_true.sh` then `code/judge_s6_true.sh`

LPIPS 0.2585 against bicubic 0.5384, filling 52.0% of the gap, where trained x4 fills 53.0% on its own
set. The one-hot arm raises `SystemExit` here rather than scoring badly: `OPS` has no slot for x6.

**Conditioning, MTF vs one-hot** -- `code/omni_cond.sh`, `code/sweep_data.sh`, `code/deblur_strata.py`

See `results/FROZEN_mtf_boundary.json`. Short version: in vocabulary the two are equivalent or MTF is
slightly behind, at both 590 and 3000 windows per operator, and MTF's seed spread is 2-5x larger. Out of
vocabulary MTF runs where one-hot cannot, but only when the descriptor is bracketed by trained operators
of the same behavioural family -- a sigma-0.6 blur, bracketed by `denoise` and `x2`, collapses 2.9 dB.

**Audio, standalone and joint** -- `code/train_audio_range.py`, `code/audio_joint.py`

12 clips: audio mel 29.262 -> 8.344 dB with the video PSNR bit-identical on 12/12 clips. Bounds on those
clips are passthrough 27.769, VAE-only 16.997, R_phi-audio alone 6.998, codec ceiling 3.086.

**No-reference control** -- `code/nr_score.py`

A NIQE-family score whose pristine model is fitted on this set's own GT (odd clips fit, even clips
scored). It cannot separate FlashVSR+ from the ground truth (median 3.01 vs 2.98) while every reference
metric separates all arms cleanly. This is why the project reports reference numbers.

**Latency** -- `results/deploy_latency.json`, and the phase block in each deploy log

0.733 s/frame of compute at 2 steps (DiT sampling 68% of it), 12.99 s per 22-frame block against a
0.708 s budget: 18.3x slower than real time. FlashVSR+ on the same clips is 11.3x. The streaming
structure (causal blocks, fixed KV, bounded latency, no global lookahead, zero underrun) is verified;
the throughput is not there and is a separate problem.

---

## Traps

Each of these produced a run that exited 0 and wrote plausible numbers. They are listed with the symptom
to look for, because that is the only part that transfers.

**1. `EVAL_SET` defaults to `data/real_eval/const`.** `run_deploy.py`, `baseline_*.py` and
`deploy_eval.py` all read the eval set from this variable and fall back to a 24-clip set whose clip names
do not overlap `heldout` at all. Three fill passes ran it by omitting the export. *Symptom:* a shard
"finishes" in a fraction of the expected time, and the clip count rises by much less than the shard size.
*Check:* assert the first clip of the job exists under the intended set before spending a card;
`fill_s6_true.sh` and `run_videolq.sh` both do.

**2. One tag, two eval sets.** `t2_mtf_s6` ended up holding 24 clips run on `heldout_s6` (x6) and 53 run
on `heldout` (x4), and the pooled LPIPS 0.2198 was a mixture of two input scales. *Symptom:* split the
clips by file mtime and the two groups differ far beyond the noise floor (here 0.0843). *Check:* the LR
resolution implies the scale -- 832/208 = 4, 832/138 = 6.

**3. `audio_scale` is 4, and a VAE-domain target is 4x too small.** `MiniMaxH3.process_latent_in`
multiplies the audio slice by `shift/audio_shift = 12/3` to carry it on the video sigma schedule, and
`process_latent_out` divides it back. *Symptom:* the returned audio is exactly 0.25 x the injected target
with cosine exactly 1.0000 -- a pure scalar rules out time-base and content faults. *Fix:* multiply the
target by `self.patcher.model.audio_scale()`.

**4. Block overlap was concatenated, not cut.** The plan is `range=[17k, 17k+22], new=[17k+5, 17k+22]`,
so whole-block concatenation repeats 5 frames in every 17. Audio was stretched 25.3% (259 latent frames
against 207) and `align()` cannot undo a periodic error. *Symptom:* audio metrics that improve far less
than the bounds predict. Fixing it also moved video PSNR +4.71 dB over 12 clips -- and a single-clip
check said it did nothing, which is why it was nearly dismissed.

**5. `hash()` is salted per process.** `make_ops.py` and `prepare_audio_pairs.py` both seeded a generator
with `abs(hash(name))`, so regenerating a pair set drew different degradations. Both now use
`sha256(...)[:4]`. *Symptom:* two "identical" pair sets whose parameters differ.

**6. `mtf_cond` caches on the operator string.** Every deblur sample passed the bare `'deblur'`, so
`operator_desc` fell back to its default sigma=1.75 and one constant vector covered kernels from 1.0 to
2.5 -- a 17-dim measurement collapsed to a higher-dimensional one-hot. The discarded signal was
`||MTF(1.0) - MTF(2.5)|| = 0.8851`. Fixed behind `--mtf-persample`. *Symptom:* a conditioning signal that
should separate from a one-hot performs identically to it, or worse.

**7. Parallel wall-clock is not per-card throughput.** 24 clips finishing in 13.3 minutes across 4 shards
was read as 34.6 s/clip; the real figure is ~120 s/clip and every ETA built on it was 3.5x optimistic.

**8. End-to-end LPIPS cannot judge the conditioning.** Handing the arm a *wrong* operator descriptor
(x6 for x4 data) costs +0.0009 LPIPS, the same as the entire MTF-vs-one-hot difference, because the
innerDC projection onto R_phi(y) absorbs it. Conditioning has to be judged at the R_phi level.
*Symptom:* two interventions that should differ produce differences at the noise floor -- suspect the
measurement before concluding equivalence.

**9. A single clip cannot validate a fix.** Trap 4 looked inert on clip `1jyM3quZ0h0` (17.507 -> 17.501)
and was +4.71 dB over 12 clips.

**10. Cold start is not the cost.** NPU first-call operator compilation can be a hundred times the steady
state. Every latency number here is warm; if two measurements differ by an order of magnitude, suspect
the measurement.

---

## Order of operations for a clean rebuild

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
source env.sh
cd code
python preflight.py --cards 0 --need-gb 45 --eval-set data/real_eval/heldout --tag <tag>
python patches/check_drift.py                      # which patches this host actually has
EVAL_SET=data/real_eval/heldout python run_deploy.py --tag <tag> --dc rphi --guide y \
  --rphi ../results/omni/omni_m5.pt --op 4 --turbo-steps 2 --merge-lora
EVAL_SET=data/real_eval/heldout python deploy_eval.py --procs 12 eval --name <judge> --arm <tag>=../results/deploy_runs/<tag> --arm bicubic=bicubic
```

`preflight.py` gates free HBM, manifest-against-disk clip counts, checkpoint adapter shapes and
degradation-record agreement across sets. It exists because each of its gates corresponds to a failure
that cost cards.
