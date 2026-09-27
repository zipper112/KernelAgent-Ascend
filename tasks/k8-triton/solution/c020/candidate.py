import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl


@triton.jit
def _ccu_regwindow(x_ptr, state_ptr, weight_ptr, bias_ptr, out_ptr, index_ptr,
                   D: tl.constexpr, L: tl.constexpr,
                   HAS_BIAS: tl.constexpr, HAS_INDEX: tl.constexpr,
                   BLOCK_D: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_D + tl.arange(0, BLOCK_D)

    row = batch
    valid = True
    if HAS_INDEX:
        row = tl.load(index_ptr + batch).to(tl.int32)
        valid = row >= 0
        row = tl.where(valid, row, 0)

    dim_mask = offs < D
    other = 0.0
    state_mask = dim_mask & valid

    weight_base = weight_ptr + offs * 4
    w0 = tl.load(weight_base, mask=dim_mask, other=other).to(tl.float32)
    w1 = tl.load(weight_base + 1, mask=dim_mask, other=other).to(tl.float32)
    w2 = tl.load(weight_base + 2, mask=dim_mask, other=other).to(tl.float32)
    w3 = tl.load(weight_base + 3, mask=dim_mask, other=other).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=dim_mask, other=other).to(tl.float32)

    state_base = state_ptr + row * D * 3 + offs * 3
    r0 = tl.load(state_base, mask=state_mask, other=0.0).to(tl.float32)
    r1 = tl.load(state_base + 1, mask=state_mask, other=0.0).to(tl.float32)
    r2 = tl.load(state_base + 2, mask=state_mask, other=0.0).to(tl.float32)

    x_base = x_ptr + batch * L * D + offs
    out_base = out_ptr + batch * L * D + offs
    for token in tl.static_range(L):
        current = tl.load(x_base + token * D, mask=dim_mask, other=0.0).to(tl.float32)
        acc = r0 * w0 + r1 * w1 + r2 * w2 + current * w3
        if HAS_BIAS:
            acc += bias
        acc = acc / (1.0 + tl.exp(-acc))
        if HAS_INDEX:
            acc = tl.where(valid, acc, 0.0)
        tl.store(out_base + token * D, acc.to(out_ptr.dtype.element_ty), mask=dim_mask)
        r0 = r1
        r1 = r2
        r2 = current

    tl.store(state_base, r0.to(state_ptr.dtype.element_ty), mask=state_mask)
    tl.store(state_base + 1, r1.to(state_ptr.dtype.element_ty), mask=state_mask)
    tl.store(state_base + 2, r2.to(state_ptr.dtype.element_ty), mask=state_mask)


_RUNNERS = {}


def kernel(inputs):
    x, state, weight, bias, index = inputs
    batch, length, dim = x.shape
    has_bias = bias is not None
    has_index = index is not None
    bias_arg = bias if has_bias else x
    index_arg = index if has_index else x
    out = torch.empty_like(x)
    key = (batch, length, dim, has_bias, has_index)

    runner = _RUNNERS.get(key)
    if runner is None:
        grid = (batch, triton.cdiv(dim, 256))
        _ccu_regwindow[grid](x, state, weight, bias_arg, out, index_arg,
                             dim, length, has_bias, has_index, 256)
        compiled = _ccu_regwindow.warmup(x, state, weight, bias_arg, out, index_arg,
                                         dim, length, has_bias, has_index, 256,
                                         grid=grid)
        runner = compiled[(grid[0], grid[1], 1)]
        _RUNNERS[key] = runner
        return out

    runner(x, state, weight, bias_arg, out, index_arg)
    return out
