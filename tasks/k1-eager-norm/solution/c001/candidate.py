# K1 eager RMSNorm (pure torch / torch_npu), traffic-minimized composition.
#
# Baseline naive eager chain materializes ~5 full-size fp32 intermediates
# (~40B/elem GM traffic) -> mte2-bound + low L2 hit rate.
# This candidate:
#   1) single reduce pass: linalg.vector_norm(ord=2, dtype=fp32) reads the
#      bf16 input directly (~2B/elem) with fp32 accumulation -> (...,1) l2.
#   2) tiny fully in-place scalar chain on the (...,1) fp32 tensor:
#      ms = l2*l2/n + eps ; rstd = rsqrt(ms).
#   3) two-step affine in the input dtype domain: y = x * rstd ; y.mul_(w).
# Total GM traffic ~10B/elem (<=18B/elem worst case if the runtime
# materializes the fp32 cast inside the norm; fallback path ~14B/elem),
# relieving the mte2 bottleneck and lifting the L2 hit rate.
# Output dtype == input dtype; eps = 1e-6.
# No .item()/.cpu()/synchronize; no torch_npu.npu_rms_norm; pure torch ops.

import torch

_VN_DTYPE_OK = None  # one-time probe cache: vector_norm(dtype=...) support on this backend


def _mean_sq_fp32(x, n):
    # fp32 mean-of-squares over the last dim -> fresh (...,1) fp32 tensor
    global _VN_DTYPE_OK
    if _VN_DTYPE_OK is not False:
        try:
            l2 = torch.linalg.vector_norm(
                x, ord=2, dim=-1, keepdim=True, dtype=torch.float32
            )
            _VN_DTYPE_OK = True
            # sum(x^2) -> mean(x^2): exact mul for pow2 n, <=1 ulp otherwise
            l2.pow_(2).mul_(1.0 / n)
            return l2
        except Exception:
            _VN_DTYPE_OK = False
    # fallback: one bf16/fp16 square materialization + fp32-accumulated mean
    return (x * x).mean(dim=-1, keepdim=True, dtype=torch.float32)


def kernel(inputs):
    x = inputs[0]
    weight = inputs[1]
    eps = inputs[2] if len(inputs) > 2 else 1e-6
    n = x.shape[-1]

    ms = _mean_sq_fp32(x, n)   # (...,1) fp32 mean of squares, single pass
    ms.add_(eps)               # eps = 1e-6 (float or 0-d tensor, no sync)
    ms.rsqrt_()                # rstd in fp32, same math as reference chain
    rstd = ms.to(x.dtype)      # (...,1) cast into input dtype domain

    y = x * rstd               # affine step 1: broadcast (...,1) mul, one kernel
    y.mul_(weight)             # affine step 2: in-place broadcast (H,) mul
    return y
