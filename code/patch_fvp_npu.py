"""Make FlashVSR+'s window partitioning run on Ascend.

partition_heads and reverse_heads both build a 9-dimensional view, [B,H,nf,wf,nh,wh,nw,ww,D], and then
call .contiguous(). On Ascend that lowers to aclnnInplaceCopy, which rejects tensors with more than 8
dimensions (AclNN_Parameter_Error EZ1001), so inference aborts inside the first attention block.

Batch and heads are both leading axes that the permute leaves adjacent and in order, so folding them into
one axis gives an 8-dimensional tensor and an exactly equivalent result -- no approximation, unlike the
block-mask shim used for the FlashVSR arm. The permutation indices are shifted down by one accordingly.

Kept as a patch script rather than an edited vendor file so the change is auditable and reversible.
"""
import ast
import sys
from pathlib import Path

P = Path('/cache/zihaohe/baselines/FlashVSR_Plus/models/attention_backends/sparse.py')
s = P.read_text()
if 'Ascend' in s:
    print('already patched'); sys.exit(0)
bak = P.with_suffix('.py.bak_pre_npu')
if not bak.exists():
    bak.write_text(s)

old_fwd = """    value = value.view(batch, heads, nf, wf, nh, wh, nw, ww, dim)
    value = value.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).contiguous()
    return value.view(batch, heads, nf * nh * nw, wf * wh * ww, dim), (fp, hp, wp), (nf, nh, nw)"""
new_fwd = """    # Ascend's copy kernel rejects more than 8 dimensions, and [B,H,nf,wf,nh,wh,nw,ww,D] is nine.
    # Batch and heads are adjacent leading axes that the permute keeps in place, so folding them into one
    # axis is exactly equivalent; indices shift down by one.
    value = value.reshape(batch * heads, nf, wf, nh, wh, nw, ww, dim)
    value = value.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return value.view(batch, heads, nf * nh * nw, wf * wh * ww, dim), (fp, hp, wp), (nf, nh, nw)"""
assert old_fwd in s, 'partition_heads body not found'
s = s.replace(old_fwd, new_fwd, 1)

old_rev = """    value = windows.view(batch, heads, nf, nh, nw, wf, wh, ww, dim)
    value = value.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).contiguous()
    value = value.view(batch, heads, fp, hp, wp, dim)"""
new_rev = """    # same 8-dimension limit as partition_heads
    value = windows.reshape(batch * heads, nf, nh, nw, wf, wh, ww, dim)
    value = value.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    value = value.view(batch, heads, fp, hp, wp, dim)"""
assert old_rev in s, 'reverse_heads body not found'
s = s.replace(old_rev, new_rev, 1)
P.write_text(s)
ast.parse(s)
print('patched, syntax ok')
