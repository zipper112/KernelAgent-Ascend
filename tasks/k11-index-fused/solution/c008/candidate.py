import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl


@triton.jit
def _index_fused_static_kernel(x_ptr, spec_ptr, ns_ptr, out_ptr,
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


@triton.jit
def _index_fused_row_kernel(x_ptr, spec_ptr, ns_ptr, out_ptr,
                            k, m, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = cols < D
    out_base = out_ptr + row * D + cols
    if row < k:
        source = tl.load(spec_ptr + row)
        value = tl.load(x_ptr + source * D + cols, mask=valid, other=0)
        tl.store(out_base, value, mask=valid)
    elif row < k + m:
        source = tl.load(ns_ptr + row - k)
        value = tl.load(x_ptr + source * D + cols, mask=valid, other=0)
        tl.store(out_base, value, mask=valid)
    else:
        tl.store(out_base, tl.zeros([BLOCK_D], dtype=out_ptr.dtype.element_ty),
                 mask=valid)


@triton.jit
def _index_fused_group_kernel(x_ptr, spec_ptr, ns_ptr, out_ptr,
                              T, k, m, D: tl.constexpr,
                              BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    row_base = tl.program_id(0) * BLOCK_T
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = cols < D
    for row_offset in tl.static_range(BLOCK_T):
        row = row_base + row_offset
        if row < T:
            out_base = out_ptr + row * D + cols
            if row < k:
                source = tl.load(spec_ptr + row)
                value = tl.load(x_ptr + source * D + cols, mask=valid, other=0)
                tl.store(out_base, value, mask=valid)
            elif row < k + m:
                source = tl.load(ns_ptr + row - k)
                value = tl.load(x_ptr + source * D + cols, mask=valid, other=0)
                tl.store(out_base, value, mask=valid)
            else:
                tl.store(out_base, tl.zeros([BLOCK_D], dtype=out_ptr.dtype.element_ty),
                         mask=valid)


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
    block_d = min(4096, triton.next_power_of_2(dim))
    block_t = 4 if rows >= 16384 else 2
    paired = rows >= 8192 and dim <= 4096
    static = dim == 4096 and rows < 8192
    key = (rows, k, m, dim, x.dtype, idx_spec.dtype, idx_ns.dtype,
           static, paired, block_t, block_d)
    record = _RUNNERS.get(key)
    if record is None:
        if static:
            grid = (rows,)
            jit_kernel = _index_fused_static_kernel
        else:
            grid_rows = triton.cdiv(rows, block_t) if paired else rows
            grid = (grid_rows, triton.cdiv(dim, block_d))
            jit_kernel = _index_fused_group_kernel if paired else _index_fused_row_kernel

        def standard(x_arg, spec_arg, ns_arg, out_arg):
            jit_kernel[grid](x_arg, spec_arg, ns_arg, out_arg,
                             *((rows, k, m, dim, block_t, block_d) if paired
                               else (k, m, dim, block_d)))

        standard(x, idx_spec, idx_ns, out)
        direct = None
        try:
            if static:
                warmup_args = (k, m, 4096, 4096)
                run_args = (k, m)
            elif paired:
                warmup_args = (rows, k, m, dim, block_t, block_d)
                run_args = (rows, k, m)
            else:
                warmup_args = (k, m, dim, block_d)
                run_args = (k, m)
            compiled = jit_kernel.warmup(
                x, idx_spec, idx_ns, out, *warmup_args, grid=grid
            )
            compiled._init_handles()
            stream = _stream_int()
            expected = torch.empty_like(out)
            direct_out = torch.empty_like(out)
            standard(x.clone(), idx_spec.clone(), idx_ns.clone(), expected)
            compiled.run(
                grid[0], grid[1] if len(grid) > 1 else 1, 1, stream,
                compiled.function,
                compiled.packed_metadata, None, None, None,
                x.clone(), idx_spec.clone(), idx_ns.clone(), direct_out, *run_args
            )
            if stream is None or not torch.equal(expected, direct_out):
                raise RuntimeError("direct-launch probe mismatch")
            direct = (compiled.run, compiled.function, compiled.packed_metadata, stream)
        except Exception as error:
            print(f"[c008] direct-launch FAIL -> {error!r}", file=sys.stderr)
            direct = None
        record = (standard, direct, grid, run_args)
        _RUNNERS[key] = record
        return out

    standard, direct, saved_grid, run_args = record
    if direct is not None:
        run, function, metadata, stream = direct
        run(saved_grid[0], saved_grid[1] if len(saved_grid) > 1 else 1,
            1, stream, function, metadata,
            None, None, None, x, idx_spec, idx_ns, out, *run_args)
    else:
        standard(x, idx_spec, idx_ns, out)
    return out
