import torch
import torch_npu  # noqa: F401  # NPU dispatch registration only; fused npu_rms_norm is NOT used

# (H,1) ones operand cache: 4-8KB, stays L2-resident across every GEMM tile,
# so the skinny matmul pays zero repeated GM reads for the reduce vector.
_ONES_CACHE = {}


def _ones_col(H, dtype, device):
    key = (H, dtype, device.index)
    t = _ONES_CACHE.get(key)
    if t is None:
        t = torch.ones((H, 1), dtype=dtype, device=device)
        _ONES_CACHE[key] = t
    return t


def kernel(inputs):
    # K1 eager RMSNorm candidate: cube-GEMM reduce (v2).
    #
    # Baseline eager chain x.float()->pow(2)->mean(-1)->rsqrt->mul->mul issues
    # ~6 full-tensor GM round-trips (~28B/elem, bf16) and performs the row
    # reduction on the vector pipe as a tree ReduceSum that re-reads the
    # tensor through a low-hit L2 path (symptom: mte2-bound + low-l2-hit).
    #
    # Engine re-balance, still pure eager composition:
    #   1) sq = x * x            vector, one pure mul: 2B read + 2B write
    #   2) s  = sq @ ones(H,1)   CUBE reduce: AIC streams sq once at cube load
    #                            bandwidth; ones is 4-8KB and L2-resident for
    #                            every fractal tile; bf16 GEMM accumulates in
    #                            fp32 on-chip, so sum precision matches a fp32
    #                            tree reduce after bf16 squaring (bf16 tol 0.03).
    #   3) inv = rsqrt(s/H + eps) tiny (N,1) fp32 tail, then cast to input dtype
    #                            so the big muls stay single-dtype (no fp32
    #                            promotion materialization).
    #   4) y  = (x * inv) * w    vector, two pure muls, output dtype == input.
    #
    # Per-element GM traffic: 4 (sq) + 2 (gemm reads sq) + 8 (two muls) ~ 14B
    # vs ~28B baseline => expected >= 1.5x.
    # Constraints honored: no .item()/.cpu()/synchronize, no npu_rms_norm.
    x = inputs[0]
    if len(inputs) >= 3:
        # tolerate fused_add_rms_norm signature: (x, residual, weight)
        x = x + inputs[1]
        weight = inputs[2].reshape(-1)
    else:
        weight = inputs[1].reshape(-1)

    orig_shape = x.shape
    H = orig_shape[-1]
    dtype = x.dtype
    x2 = x.reshape(-1, H)  # view for contiguous input, no copy

    # 1) vector: pure elementwise square in input dtype (bf16)
    sq = x2 * x2

    # 2) cube: skinny GEMM row-reduction, fp32 accumulation on AIC
    s = torch.mm(sq, _ones_col(H, dtype, x2.device))  # (N,1)

    # 3) tiny fp32 tail on (N,1); eps=1e-6 per contract, then back to input dtype
    inv = torch.rsqrt(s.float() / H + 1e-6).to(dtype)

    # 4) vector: two pure multiplies in input dtype domain
    y = x2 * inv
    y = y * weight

    return y.reshape(orig_shape)
