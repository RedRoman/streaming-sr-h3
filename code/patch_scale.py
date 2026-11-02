"""Make the downsampling factor a parameter instead of a constant, so the framework can be tested under a
change of operator.

null_volume measured the ordering the omni plan assumes, and it holds: as the unobserved share of the
signal grows from 2.3% (x2) to 23.4% (x16), what a perfect non-generative estimator leaves on the table
grows 0.094 -> 0.546 LPIPS and what our generated detail recovers grows 0.019 -> 0.351. But that was a
re-decomposition of x4 outputs at other split scales; it says nothing about whether R_phi and the gate can
be TRAINED under a different operator. For that the pipeline has to actually run at x2 and x8.

Every edit keeps the default at 4, so with SR_SCALE unset the behaviour is bit-identical to the runs that
produced the current numbers. Nothing is refactored beyond threading the factor through.

h3_comfy is the one place where the factor was genuinely implicit rather than configurable: the
consistency operator was built as (hw[0]//4, hw[1]//4) while `low` -- the observation the projection is
constrained to -- already carries the true low-resolution shape. Deriving it from `low` removes the
constant instead of replacing it with another one.
"""
import sys
from pathlib import Path

C = Path('/cache/zihaohe/streaming_sr0/code')
done = []


def edit(fn, old, new, must=True):
    p = C / fn
    s = p.read_text()
    if new.strip() and new.split('\n')[0].strip() in s:
        done.append(f'{fn}: already'); return
    if old not in s:
        if must:
            print(f'MISS {fn}: {old[:60]!r}'); sys.exit(1)
        done.append(f'{fn}: skip'); return
    p.write_text(s.replace(old, new, 1))
    done.append(f'{fn}: ok')


# 1. the consistency operator takes its low-resolution shape from the observation it constrains
edit('h3_comfy.py',
     "op = BicubicConsistency(tuple(hw), (hw[0] // 4, hw[1] // 4), dtype=torch.float32, device=self.dev) if mode == 'innerdc' else None",
     "# the factor is whatever `low` actually is; it used to be hardcoded to 4 here while low carried the truth\n"
     "        lo_hw = tuple(low.shape[-2:]) if low is not None else (hw[0] // 4, hw[1] // 4)\n"
     "        op = BicubicConsistency(tuple(hw), lo_hw, dtype=torch.float32, device=self.dev) if mode == 'innerdc' else None")

# 2. degradation chain
edit('degrade.py',
     "def pillow_bicubic_x4(hr):\n    h, w = hr.shape[1] // 4, hr.shape[2] // 4",
     "def pillow_bicubic_x4(hr, scale=4):\n    h, w = hr.shape[1] // scale, hr.shape[2] // scale")
edit('degrade.py',
     "def degrade(hr, rng):\n    \"\"\"hr uint8 [T,H,W,3] with H,W divisible by 4 -> (lr uint8 [T,H/4,W/4,3], params dict).\"\"\"\n"
     "    t, H, W, _ = hr.shape\n    size = (W // 4, H // 4)\n"
     "    if rng.random() < P_BICUBIC:\n        return pillow_bicubic_x4(hr), {'kind': 'pillow_bicubic'}",
     "def degrade(hr, rng, scale=4):\n    \"\"\"hr uint8 [T,H,W,3], H,W divisible by scale -> (lr uint8 [T,H/s,W/s,3], params dict).\"\"\"\n"
     "    t, H, W, _ = hr.shape\n    size = (W // scale, H // scale)\n"
     "    if rng.random() < P_BICUBIC:\n        return pillow_bicubic_x4(hr, scale), {'kind': 'pillow_bicubic'}")

# 3. evaluation-set builder
edit('prepare_real_eval.py', "T, H, W = 124, 480, 832",
     "T, H, W = 124, 480, 832\nSCALE = int(os.environ.get('SR_SCALE', 4))")
edit('prepare_real_eval.py', "        lr, params = (degrade_tvary(hr, rng) if a.variant == 'tvary' else DG.degrade(hr, rng))\n"
                             "        assert lr.shape == (T, H // 4, W // 4, 3), lr.shape",
     "        lr, params = (degrade_tvary(hr, rng) if a.variant == 'tvary' else DG.degrade(hr, rng, SCALE))\n"
     "        assert lr.shape == (T, H // SCALE, W // SCALE, 3), lr.shape")
edit('prepare_real_eval.py', "                'frames': T, 'hr_wh': [W, H], 'start_s': start, 'variant': a.variant,",
     "                'frames': T, 'hr_wh': [W, H], 'scale': SCALE, 'start_s': start, 'variant': a.variant,")

# 4. range-branch training pairs
edit('prepare_range_data.py', "SCALE = 4", "SCALE = int(os.environ.get('SR_SCALE', 4))")
edit('prepare_range_data.py', "            lr, rec = degrade(hr, rng)", "            lr, rec = degrade(hr, rng, SCALE)")

# 5. the range/null split used by every offline metric
edit('deploy_eval.py', "def split(x, s=4):", "SR_SCALE = int(os.environ.get('SR_SCALE', 4))\n\n\ndef split(x, s=None):\n    s = SR_SCALE if s is None else s")

print('\n'.join(done))
for f in ('h3_comfy.py', 'degrade.py', 'prepare_real_eval.py', 'prepare_range_data.py', 'deploy_eval.py'):
    import ast
    ast.parse((C / f).read_text())
print('syntax ok')
