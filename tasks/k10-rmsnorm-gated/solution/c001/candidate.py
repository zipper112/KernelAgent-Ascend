import sys

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


_RUNNERS = {}
_OUTPUTS = {}


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


def _get_output(x):
    key = (tuple(x.shape), x.dtype, x.device)
    record = _OUTPUTS.get(key)
    if record is None:
        outputs = [torch.empty(x.shape, dtype=x.dtype, device=x.device) for _ in range(4)]
        record = (outputs, 0)
        _OUTPUTS[key] = record
    outputs, index = record
    _OUTPUTS[key] = (outputs, (index + 1) & 3)
    return outputs[index]


def kernel(inputs):
    x, g, weight, bias = inputs
    if x.shape[-1] != 128:
        raise ValueError("rms_norm_gated candidate requires D=128")
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
    rows = x.numel() // 128
    key = (rows, x.dtype, g.dtype, weight.dtype,
           bias_arg.dtype if has_bias else None, x.stride(), g.stride())
    out = _get_output(x)
    record = _RUNNERS.get(key)
    if record is None:
        grid = (triton.cdiv(rows, 32),)

        def standard(x_arg, g_arg, weight_arg, bias_arg_, out_arg):
            _rms_gate_kernel[grid](
                x_arg, g_arg, weight_arg, bias_arg_, out_arg,
                rows, 1e-6, has_bias, 128, 32,
            )

        standard(x, g, weight, bias_arg, out)
        direct = None
        try:
            compiled = _rms_gate_kernel.warmup(
                x, g, weight, bias_arg, out, rows, 1e-6,
                has_bias, 128, 32, grid=grid,
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
            direct = (compiled.run, compiled.function,
                      compiled.packed_metadata, stream)
        except Exception as error:
            print(f"[c001] direct-launch FAIL -> {error!r}", file=sys.stderr)
            direct = None
        record = (standard, direct, grid[0])
        _RUNNERS[key] = record
        return out

    standard, direct, grid_1 = record
    if direct is not None:
        run, function, metadata, stream = direct
        run(grid_1, 1, 1, stream, function, metadata,
            None, None, None,
            x, g, weight, bias_arg, out, rows, 1e-6)
        return out
    standard(x, g, weight, bias_arg, out)
    return out
