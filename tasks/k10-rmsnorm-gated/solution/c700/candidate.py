import sys
from collections import OrderedDict
from functools import partial

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl


@triton.jit
def _rms_gate_kernel(x_ptr, g_ptr, weight_ptr, bias_ptr, out_ptr,
                     rows, eps, HAS_BIAS: tl.constexpr,
                     D: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    program = tl.program_id(0)
    row_offsets = program * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dim_offsets = tl.arange(0, D)
    row_mask = row_offsets < rows
    value_mask = row_mask[:, None]
    offsets = row_offsets[:, None] * D + dim_offsets[None, :]

    x = tl.load(x_ptr + offsets, mask=value_mask, other=0.0).to(tl.float32)
    gate = tl.load(g_ptr + offsets, mask=value_mask, other=0.0).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
    weight = tl.load(weight_ptr + dim_offsets).to(tl.float32)
    y = x * scale[:, None] * weight[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + dim_offsets).to(tl.float32)
        y += bias[None, :]
    y *= 1.0 / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=value_mask)


@triton.jit
def _rms_gate_tiled_kernel(x_ptr, g_ptr, weight_ptr, bias_ptr, out_ptr,
                           rows, eps, HAS_BIAS: tl.constexpr,
                           D: tl.constexpr, BLOCK_ROWS: tl.constexpr,
                           TILES: tl.constexpr):
    program = tl.program_id(0)
    dim_offsets = tl.arange(0, D)
    weight = tl.load(weight_ptr + dim_offsets).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + dim_offsets).to(tl.float32)
    for tile in tl.static_range(TILES):
        row_offsets = (program * TILES + tile) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        value_mask = (row_offsets < rows)[:, None]
        offsets = row_offsets[:, None] * D + dim_offsets[None, :]
        x = tl.load(x_ptr + offsets, mask=value_mask, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + offsets, mask=value_mask, other=0.0).to(tl.float32)
        scale = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
        y = x * scale[:, None] * weight[None, :]
        if HAS_BIAS:
            y += bias[None, :]
        y *= 1.0 / (1.0 + tl.exp(-gate))
        tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=value_mask)


_STATES = OrderedDict()
_D128_STATES = {}
_MAX_STATES = 2


def _stream_int():
    try:
        from triton.runtime import driver
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
    except Exception:
        stream = None
    if isinstance(stream, int):
        return stream
    for name in ("cuda_stream", "npu_stream", "stream"):
        value = getattr(stream, name, None)
        if isinstance(value, int):
            return value
    return int(torch.npu.current_stream().cuda_stream)




def _next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def _block_rows(dim, rows):
    values = rows * dim
    if dim == 64:
        if values < 2_097_152:
            return min(32, max(2, _next_power_of_two(rows)))
        return 64 if values < 8_388_608 else 128
    if dim == 128:
        if rows <= 256:
            return min(16, max(2, _next_power_of_two(rows)))
        if rows <= 65536:
            return min(32, max(2, _next_power_of_two(rows)))
        if values < 8_388_608:
            return min(32, max(2, _next_power_of_two(rows)))
        return 64
    if values < 8_388_608:
        return min(16, max(2, _next_power_of_two(rows)))
    return 32


def _fallback(inputs):
    x, g, weight, bias = inputs
    x_float = x.float()
    scale = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + 1e-6)
    weight_float = weight.float()
    normalized = x_float * scale * weight_float
    if bias is not None and bias.numel() != 0:
        normalized = normalized + bias.float()
    return (normalized * torch.sigmoid(g.float())).to(x.dtype)


def kernel(inputs):
    x, g, weight, bias = inputs
    dim = x.shape[-1]
    if dim not in (64, 128, 256):
        return _fallback(inputs)
    if not x.is_contiguous():
        x = x.contiguous()
    if not g.is_contiguous():
        g = g.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    has_bias = bias is not None and bias.numel() != 0
    if has_bias and not bias.is_contiguous():
        bias = bias.contiguous()
    bias_arg = bias if has_bias else x
    rows = x.numel() // dim
    if dim == 128:
        state = _D128_STATES.get(x.shape)
    else:
        key = (x.shape, x.dtype, x.device)
        state = _STATES.get(key)
    if state is None:
        outputs = [torch.empty(x.shape, dtype=x.dtype, device=x.device) for _ in range(4)]
        out = outputs[0]
        tiled = dim == 128 and rows <= 16384
        if tiled:
            block_rows = 8
            tiles = 8
            grid = (triton.cdiv(rows, block_rows * tiles),)
            jit_kernel = _rms_gate_tiled_kernel
        else:
            block_rows = 32 if dim == 128 else _block_rows(dim, rows)
            tiles = None
            grid = (triton.cdiv(rows, block_rows),)
            jit_kernel = _rms_gate_kernel

        def standard(x_arg, g_arg, weight_arg, bias_arg_, out_arg):
            if tiled:
                jit_kernel[grid](
                    x_arg, g_arg, weight_arg, bias_arg_, out_arg,
                    rows, 1e-6, has_bias, dim, block_rows, tiles,
                    num_warps=1,
                    num_stages=8,
                )
            else:
                jit_kernel[grid](
                    x_arg, g_arg, weight_arg, bias_arg_, out_arg,
                    rows, 1e-6, has_bias, dim, block_rows,
                    num_warps=4,
                )

        standard(x, g, weight, bias_arg, out)
        launch = None
        try:
            if tiled:
                compiled = jit_kernel.warmup(
                    x, g, weight, bias_arg, out, rows, 1e-6,
                    has_bias, dim, block_rows, tiles, grid=grid,
                    num_warps=1, num_stages=8,
                )
            else:
                compiled = jit_kernel.warmup(
                    x, g, weight, bias_arg, out, rows, 1e-6,
                    has_bias, dim, block_rows, grid=grid,
                    num_warps=4,
                )
            compiled._init_handles()
            stream = _stream_int()
            expected = torch.empty_like(out)
            direct_out = torch.empty_like(out)
            x_probe = x.clone()
            g_probe = g.clone()
            standard(x_probe, g_probe, weight, bias_arg, expected)
            compiled.run(
                grid[0], 1, 1, stream, compiled.function,
                compiled.packed_metadata, None, None, None,
                x_probe, g_probe, weight, bias_arg, direct_out,
                rows, 1e-6,
            )
            if stream is None or not torch.equal(expected, direct_out):
                raise RuntimeError("direct-launch probe mismatch")
            launch = partial(compiled.run, grid[0], 1, 1, stream,
                             compiled.function, compiled.packed_metadata,
                             None, None, None)
        except Exception as error:
            print(f"[c700] direct-launch FAIL -> {error!r}", file=sys.stderr)
            launch = None
        state = [launch, standard, outputs, 1]
        if dim == 128:
            _D128_STATES.clear()
            _D128_STATES[x.shape] = state
        else:
            _STATES[key] = state
            while len(_STATES) > _MAX_STATES:
                _STATES.popitem(last=False)
        return out

    launch, standard, outputs, index = state
    out = outputs[index]
    state[3] = (index + 1) & 3
    if dim != 128:
        _STATES.move_to_end(key)
    if launch is not None:
        launch(x, g, weight, bias_arg, out, rows, 1e-6)
    else:
        standard(x, g, weight, bias_arg, out)
    return out
