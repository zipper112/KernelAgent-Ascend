import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_kernel(
    x_ptr, state_ptr, w_ptr, b_ptr, idx_ptr, out_ptr,
    B, S, H, TOTAL,
    EPS,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # persistent: 40 cores, flat tile over batch*hidden
    pid = tl.program_id(0)
    # shared per-channel weights loaded inside loop (cheap, cached)
    for base in tl.range(pid * BLOCK, TOTAL, NUM_BLOCKS * BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < TOTAL
        b_idx = offs // H
        h_idx = offs % H

        # dispatch target batch via indices (decode: which requests update)
        real_b = tl.load(idx_ptr + b_idx, mask=mask, other=0)
        flat = real_b * H + h_idx

        # load 3-length sliding window into registers
        s0 = tl.load(state_ptr + flat * 3 + 0, mask=mask, other=0.0)
        s1 = tl.load(state_ptr + flat * 3 + 1, mask=mask, other=0.0)
        s2 = tl.load(state_ptr + flat * 3 + 2, mask=mask, other=0.0)

        w0 = tl.load(w_ptr + h_idx * 4 + 0, mask=mask, other=0.0)
        w1 = tl.load(w_ptr + h_idx * 4 + 1, mask=mask, other=0.0)
        w2 = tl.load(w_ptr + h_idx * 4 + 2, mask=mask, other=0.0)
        w3 = tl.load(w_ptr + h_idx * 4 + 3, mask=mask, other=0.0)
        bias = tl.load(b_ptr + h_idx, mask=mask, other=0.0)

        x_row_base = (real_b * H + h_idx) * S

        for t in tl.range(0, S):
            xt = tl.load(x_ptr + x_row_base + t, mask=mask, other=0.0).to(tl.float32)
            acc = w0 * s0.to(tl.float32) + w1 * s1.to(tl.float32) \
                + w2 * s2.to(tl.float32) + w3 * xt + bias.to(tl.float32)
            # silu
            y = acc / (1.0 + tl.exp(-acc))
            tl.store(out_ptr + x_row_base + t, y.to(out_ptr.dtype.element_ty), mask=mask)
            # slide window in registers
            s0 = s1
            s1 = s2
            s2 = xt.to(state_ptr.dtype.element_ty)

        # write back new state (in-place sliding update)
        tl.store(state_ptr + flat * 3 + 0, s0, mask=mask)
        tl.store(state_ptr + flat * 3 + 1, s1, mask=mask)
        tl.store(state_ptr + flat * 3 + 2, s2, mask=mask)


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs
    B, S, H = x.shape
    out = torch.empty_like(x)
    TOTAL = B * H
    NUM_BLOCKS = 40
    BLOCK = 512  # triton.power_of_2 sized tile; UB-friendly
    grid = (NUM_BLOCKS,)
    _causal_conv1d_update_kernel[grid](
        x, conv_state, weight, bias, indices, out,
        B, S, H, TOTAL,
        1e-6,
        NUM_BLOCKS=NUM_BLOCKS,
        BLOCK=BLOCK,
    )
    return out