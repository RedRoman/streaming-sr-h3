"""Cache FlashVSR+'s expanded attention mask instead of rebuilding it inside every attention call.

A py-spy profile of the single-clip run (4499 samples over 90 s, one 910B card) puts 99.96% of self time
in `_expand_block_mask`, line `if empty.any()`. That line is where the device synchronises, so it absorbs
the cost of the three lines above it: two `repeat_interleave` calls that materialise a dense [B,H,Q,K]
bool mask -- the guard in the same file allows up to 2e9 entries -- followed by a reduction over it. The
clip had produced no frames after four hours.

The expanded mask is a pure function of (block_mask, block_token_size, query_tokens, key_tokens). All four
come from the stream window geometry and the KV-cache occupancy; none depends on the video content. So the
same mask is rebuilt once per attention call, i.e. once per layer per chunk, when the distinct values
number in the low single digits. Keying the cache on the compact block mask's exact bytes makes a hit
bit-identical to a rebuild: this removes repeated work, it does not approximate the attention pattern
(unlike BSA_DENSE=1 on the original FlashVSR, which drops 4.08% of the mask).

Verification is built in rather than assumed. With FVP_MASK_VERIFY=n the first n cache hits rebuild the
mask the original way and assert bitwise equality before returning the cached copy; the counter is
reported through `mask_cache_stats()` and printed by the runner at the end of each clip. Without a hit the
patch would be invisible -- the memory note on patches that are "installed" but never "fire" applies here.

Run with no arguments to patch, `--revert` to restore the backup, `--check` to show current state.
"""
import argparse
import re
from pathlib import Path

TARGET = Path('/cache/zihaohe/baselines/FlashVSR_Plus/models/attention_backends/kernels.py')
BACKUP = TARGET.with_suffix('.py.bak_pre_maskcache')
SENTINEL = '_MASK_CACHE'

ORIGINAL_HEAD = '''def _expand_block_mask(
    block_mask: torch.Tensor,
    block_token_size: int,
    query_tokens: int,
    key_tokens: int,
) -> torch.Tensor:
'''

NEW_CODE = '''_MASK_CACHE = {}
_MASK_CACHE_STATS = {"hit": 0, "miss": 0, "verified": 0, "build_s": 0.0, "key_s": 0.0}
_MASK_CACHE_MAX = int(_os.environ.get("FVP_MASK_CACHE", "8"))
_MASK_VERIFY_BUDGET = int(_os.environ.get("FVP_MASK_VERIFY", "3"))


def mask_cache_stats():
    """Hit/miss counts so a caller can confirm the cache fired rather than assuming it did."""
    return dict(_MASK_CACHE_STATS, entries=len(_MASK_CACHE))


def _build_block_mask(
    block_mask: torch.Tensor,
    block_token_size: int,
    query_tokens: int,
    key_tokens: int,
) -> torch.Tensor:
    """The original expansion, unchanged. Kept verbatim so a cache hit can be checked against it."""
    mask = block_mask.repeat_interleave(block_token_size, dim=-2)
    mask = mask.repeat_interleave(block_token_size, dim=-1)
    mask = mask[..., :query_tokens, :key_tokens].bool()
    empty = ~mask.any(dim=-1, keepdim=True)
    if empty.any():
        fallback = torch.zeros_like(mask)
        diagonal = torch.arange(min(query_tokens, key_tokens), device=mask.device)
        fallback[..., diagonal, diagonal] = True
        mask = torch.where(empty, fallback, mask)
    return mask


def _expand_block_mask(
    block_mask: torch.Tensor,
    block_token_size: int,
    query_tokens: int,
    key_tokens: int,
) -> torch.Tensor:
    global _MASK_CACHE
    _t0 = _time.perf_counter()
    key = (
        tuple(block_mask.shape),
        str(block_mask.dtype),
        str(block_mask.device),
        int(block_token_size),
        int(query_tokens),
        int(key_tokens),
        hash(block_mask.detach().to("cpu").contiguous().numpy().tobytes()),
    )
    _MASK_CACHE_STATS["key_s"] += _time.perf_counter() - _t0
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        _MASK_CACHE_STATS["hit"] += 1
        if _MASK_CACHE_STATS["verified"] < _MASK_VERIFY_BUDGET:
            rebuilt = _build_block_mask(
                block_mask, block_token_size, query_tokens, key_tokens
            )
            if not bool(torch.equal(rebuilt, cached)):
                raise AssertionError(
                    "cached attention mask differs from a rebuild; the cache key is not "
                    "capturing everything the expansion depends on"
                )
            _MASK_CACHE_STATS["verified"] += 1
            del rebuilt
        return cached
    _MASK_CACHE_STATS["miss"] += 1
    _t1 = _time.perf_counter()
    built = _build_block_mask(block_mask, block_token_size, query_tokens, key_tokens)
    _MASK_CACHE_STATS["build_s"] += _time.perf_counter() - _t1
    if len(_MASK_CACHE) >= _MASK_CACHE_MAX:
        # Masks are large; drop the whole table rather than grow without bound. Distinct masks appear
        # only while the KV cache fills, so this should not be reached in steady state.
        _MASK_CACHE = {}
    _MASK_CACHE[key] = built
    return built
'''


def patch(path: Path) -> str:
    src = path.read_text()
    if SENTINEL in src:
        return 'already patched'
    if ORIGINAL_HEAD not in src:
        raise SystemExit('_expand_block_mask does not have the expected signature; refusing to patch')
    if not BACKUP.exists():
        BACKUP.write_text(src)
    # imports the new code needs, added next to the existing ones rather than inside the function
    if 'import os as _os' not in src:
        src = re.sub(r'^(import torch\b.*)$', 'import os as _os\nimport time as _time\n\\1', src,
                     count=1, flags=re.M)
        if 'import os as _os' not in src:
            raise SystemExit('could not find an "import torch" line to anchor the new imports')
    # replace the whole original definition: from its `def` line to the next top-level `def`
    start = src.index(ORIGINAL_HEAD)
    end = src.index('\ndef _run_block_sparse(', start) + 1
    expected_body = src[start:end]
    if 'repeat_interleave' not in expected_body or 'empty.any()' not in expected_body:
        raise SystemExit('the span being replaced is not the expansion body; refusing to patch')
    src = src[:start] + NEW_CODE + '\n\n' + src[end:]
    path.write_text(src)
    return 'patched'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    src = TARGET.read_text()
    if a.check:
        print('patched' if SENTINEL in src else 'unpatched',
              '| backup', 'present' if BACKUP.exists() else 'absent')
        return
    if a.revert:
        if not BACKUP.exists():
            raise SystemExit('no backup to revert to')
        TARGET.write_text(BACKUP.read_text())
        print('reverted from', BACKUP)
        return
    print(patch(TARGET))
    import py_compile
    py_compile.compile(str(TARGET), doraise=True)
    print('compiles OK')


if __name__ == '__main__':
    main()
