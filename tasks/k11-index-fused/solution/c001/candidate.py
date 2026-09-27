import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl


@triton.jit
def _index_fused_kernel(x_ptr, spec_ptr, ns_ptr, out_ptr,
                        k, m, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    out_base = out_ptr + row * D + cols
    valid = cols < D
    if row < k:
        source = tl.load(spec_ptr + row)
        value = tl.load(x_ptr + source * D + cols, mask=valid, other=0)
        tl.store(out_base, value, mask=valid)
    elif row < k + m:
        source = tl.load(ns_ptr + row - k)
        value = tl.load(x_ptr + source * D + cols, mask=valid, other=0)
        tl.store(out_base, value, mask=valid)
    else:
        tl.store(out_base, tl.zeros([BLOCK_D], dtype=out_ptr.dtype.element_ty), mask=valid)


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
    try:
        return int(torch.npu.current_stream().cuda_stream)
    except Exception:
        return None


def _get_output(x):
    key = (tuple(x.shape), x.dtype, x.device)
    record = _OUTPUTS.get(key)
    if record is None:
        outputs = [torch.empty_like(x) for _ in range(8)]
        record = (outputs, 0)
        _OUTPUTS[key] = record
    outputs, index = record
    _OUTPUTS[key] = (outputs, (index + 1) & 7)
    return outputs[index]


def kernel(inputs):
    x, idx_spec, idx_ns = inputs
    rows, dim = x.shape
    if dim != 4096:
        raise ValueError("index-fused candidate requires D=4096")
    if not x.is_contiguous():
        x = x.contiguous()
    if not idx_spec.is_contiguous():
        idx_spec = idx_spec.contiguous()
    if not idx_ns.is_contiguous():
        idx_ns = idx_ns.contiguous()
    k = idx_spec.numel()
    m = idx_ns.numel()
    if k + m > rows:
        raise ValueError("k + m must be <= T")

    out = _get_output(x)
    key = (rows, k, m, x.dtype, idx_spec.dtype, idx_ns.dtype)
    record = _RUNNERS.get(key)
    if record is None:
        grid = (rows,)

        def standard(x_arg, spec_arg, ns_arg, out_arg):
            _index_fused_kernel[grid](
                x_arg, spec_arg, ns_arg, out_arg, k, m, 4096, 4096
            )

        standard(x, idx_spec, idx_ns, out)
        direct = None
        try:
            compiled = _index_fused_kernel.warmup(
                x, idx_spec, idx_ns, out, k, m, 4096, 4096, grid=grid
            )
            compiled._init_handles()
            stream = _stream_int()
            expected = torch.empty_like(out)
            direct_out = torch.empty_like(out)
            standard(x.clone(), idx_spec.clone(), idx_ns.clone(), expected)
            compiled.run(
                rows, 1, 1, stream, compiled.function, compiled.packed_metadata,
                None, None, None,
                x.clone(), idx_spec.clone(), idx_ns.clone(), direct_out, k, m,
            )
            if stream is None or not torch.equal(expected, direct_out):
                raise RuntimeError("direct-launch probe mismatch")
            direct = (compiled.run, compiled.function, compiled.packed_metadata, stream)
        except Exception as error:
            print(f"[c001] direct-launch FAIL -> {error!r}", file=sys.stderr)
            direct = None
        record = (standard, direct, rows)
        _RUNNERS[key] = record
        return out

    standard, direct, grid_1 = record
    if direct is not None:
        run, function, metadata, stream = direct
        run(grid_1, 1, 1, stream, function, metadata,
            None, None, None, x, idx_spec, idx_ns, out, k, m)
    else:
        standard(x, idx_spec, idx_ns, out)
    return out
