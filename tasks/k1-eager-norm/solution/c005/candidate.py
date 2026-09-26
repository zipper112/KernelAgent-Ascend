import torch

# K1 eager RMSNorm candidate: k1-eager-rmsnorm-cubegemm-reduce-v3
#
# 结构（相对 baseline 的 x.float().pow(2).mean(...) 链，~28B/elem GM 流量）：
#   1) sq = x * x                —— vector 侧纯乘，bf16 域，无 cast 往返
#   2) ss = sq @ ones(H,1)       —— Cube GEMM 归约：ones 仅 H*2B（常驻 L2），
#                                   A 面（N,H）一次性满带宽搬入，片上 fp32 累加，
#                                   替换 mte2-bound 的 vector 树形 ReduceSum
#   3) inv = rsqrt(ss/H + eps)   —— 只在 (N,1) 小张量上做 fp32 统计，开销可忽略
#   4) y = (x * inv) * w         —— vector 侧两个纯乘，输入 dtype 域，广播 (N,1)/(H,)
# 每元素 GM 流量 ~14B（x 读 2 + sq 写 2 + sq 读 2 + x 读 2 + w 读 2 + y 写 2 + 小量统计）
#
# 精度策略（对应 bf16 0.03 / fp16 0.004 分级容差）：
#   - 平方在输入 dtype 域逐元素量化（相对误差 ~2^-9 for bf16），全为正项求和无抵消；
#   - 求和在 Cube 上 fp32 累加，输出仅一次 dtype 舍入；
#   - mean+eps+rsqrt 全在 fp32 小张量上完成；
#   - 最终 y 与 reference 的 x.float() 链差异 ~0.3% 量级，远小于 0.03。
# 行为约束：eps=1e-6 固定；输出 shape 与 dtype 与输入一致。

_ONES_CACHE = {}


def _ones_col(height, device, dtype):
    # (H,1) 全 1 列向量：尺寸 O(H)（8KB 级），首次创建后常驻，跨调用复用，L2 命中
    key = (height, str(device), dtype)
    t = _ONES_CACHE.get(key)
    if t is None:
        t = torch.ones((height, 1), device=device, dtype=dtype)
        _ONES_CACHE[key] = t
    return t


def kernel(inputs):
    x = inputs[0]
    w = inputs[1]
    bias = None
    if len(inputs) > 2 and torch.is_tensor(inputs[2]):
        bias = inputs[2]

    eps = 1e-6
    orig_shape = x.shape
    H = orig_shape[-1]

    # 输出 dtype 与输入一致：权重/偏置若 dtype 不齐则对齐到 x 的域
    if w.dtype != x.dtype:
        w = w.to(x.dtype)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)

    # (B,S,H) -> (N,H)；连续输入下 reshape 是零拷贝 view
    x2 = x.reshape(-1, H)

    # 1) vector：逐元素平方，停留在输入 dtype 域（省去 fp32 cast 的双向流量）
    sq = x2 * x2

    # 2) cube：行平方和走 GEMM。ones(H,1) 极小且 L2 常驻；A 面 (N,H) 单次流式读入，
    #    片上 fp32 累加后舍入回输入 dtype 输出 (N,1)
    ss = torch.matmul(sq, _ones_col(H, x.device, x.dtype))  # (N,1)

    # 3) 统计量在 fp32 小张量上计算：mean(-1) + eps 再 rsqrt（N 个元素，开销可忽略）
    inv = torch.rsqrt(ss.to(torch.float32) / H + eps)  # (N,1) fp32
    inv = inv.to(x.dtype)

    # 4) vector：两个纯乘 kernel，(N,H)*(N,1) 广播后再乘 (H,) 权重，全程输入 dtype
    y = (x2 * inv) * w
    if bias is not None:
        y = y + bias

    return y.reshape(orig_shape)
