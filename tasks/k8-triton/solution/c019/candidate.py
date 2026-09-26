# K8: causal_conv1d_update Triton-Ascend 单派发版
# 语义（同 c002 / fla）: window=[s0,s1,s2,xt], y=silu(sum(w*window)+b)
# state 原地滑动更新: 处理完 S 个 token 后 state=[.., x_{S-3..S-1}]; indices<0 为 pad 槽跳过
import torch
import triton
import triton.language as tl


@triton.jit
def _k8_cconv_update_kernel(
    x_ptr, st_ptr, w_ptr, b_ptr, idx_ptr, out_ptr,
    sx_b, sx_t, ss_b,
    B, H,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nH = tl.cdiv(H, BLOCK)
    total = B * nH
    # persistent：每核跨 tile 循环，单波覆盖全部 (b, h-tile)
    for tile in range(pid, total, nprog):
        b = tile // nH
        hb = tile - b * nH
        offs = hb * BLOCK + tl.arange(0, BLOCK)
        hm = offs < H
        idx = tl.load(idx_ptr + b)          # 逻辑 batch -> 物理 state 行
        m = hm & (idx >= 0)                 # pad 槽跳过
        st_base = st_ptr + idx * ss_b + offs * 3
        s0 = tl.load(st_base + 0, mask=m, other=0.0).to(tl.float32)
        s1 = tl.load(st_base + 1, mask=m, other=0.0).to(tl.float32)
        s2 = tl.load(st_base + 2, mask=m, other=0.0).to(tl.float32)
        w_base = w_ptr + offs * 4
        w0 = tl.load(w_base + 0, mask=m, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1, mask=m, other=0.0).to(tl.float32)
        w2 = tl.load(w_base + 2, mask=m, other=0.0).to(tl.float32)
        w3 = tl.load(w_base + 3, mask=m, other=0.0).to(tl.float32)
        bias = tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
        xb = x_ptr + b * sx_b
        ob = out_ptr + b * sx_b
        # S 静态展开：窗口寄存器内滑动，无任何中间张量
        for t in tl.static_range(S):
            xt = tl.load(xb + t * sx_t + offs, mask=m, other=0.0).to(tl.float32)
            y = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3 + bias
            y = y / (1.0 + tl.exp(-y))      # silu，fp32 累加同 c002 语义
            tl.store(ob + t * sx_t + offs, y.to(out_ptr.dtype.element_ty), mask=m)
            s0 = s1
            s1 = s2
            s2 = xt
        # 原地滑动写回新 state
        tl.store(st_base + 0, s0.to(st_ptr.dtype.element_ty), mask=m)
        tl.store(st_base + 1, s1.to(st_ptr.dtype.element_ty), mask=m)
        tl.store(st_base + 2, s2.to(st_ptr.dtype.element_ty), mask=m)


_CACHE = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
    B, S, H = x.shape
    if bias is None:
        bias = _cached(("z", H, x.device, x.dtype),
                       lambda: torch.zeros(H, dtype=x.dtype, device=x.device))
    if indices is None:
        indices = _cached(("ar", B, x.device),
                          lambda: torch.arange(B, device=x.device))
    out = torch.empty_like(x)               # 仅输出一次分配，无计算中间物
    BLOCK = 256
    nH = (H + BLOCK - 1) // BLOCK
    grid = (min(40, B * nH),)               # ≈40 核单波（A2 单 block 物理核）
    _k8_cconv_update_kernel[grid](
        x, conv_state, weight, bias, indices, out,
        x.stride(0), x.stride(1), conv_state.stride(0),
        B, H,
        S=S, BLOCK=BLOCK,
    )
    return out