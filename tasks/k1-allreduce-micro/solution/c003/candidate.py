# candidate.py -- k1-allreduce-micro / direction: cube-gemv-offload
# 纯 torch_npu 实现（无 triton，符合契约 dsl: pure-torch；禁 .item()/.cpu() 已遵守）
#
# 调研结论 -> 实现映射：
# 1) w01 (8,2048,4096) fp16 = 128MiB，bandwidth-bound：单 pass 决定延迟。
#    把 last-dim sum 改写为 (rows, K) @ (K, 1) 的 GEMV，走 CANN Cube matmul 管线
#    （fp32 累加 + L2->L0 double-buffer 搬运），对齐 skill 中'带宽受限优先喂满搬运管线'原则。
# 2) w02 (1,1,1048576) 仅 2MiB，latency-bound：任何多 kernel 方案（分块两段式）
#    都会引入额外 launch 开销而变慢，故保留原生单 kernel torch.sum（tail-effect 规避）。
# 3) ones 列向量按 (K, dtype, device) 缓存，消除每次调用的 fill kernel 启动。
# 4) 精度：Cube fp32 累加后舍入 fp16，相对误差 <= 2^-11，满足 fp16 tolerance 0.004；
#    0 行精确为 0，极值行累加次序确定（结果稳定）。

import torch

_ONES = {}  # (K, dtype, device) -> ones (K,1) 列向量缓存


def _ones_col(k, dtype, device):
    key = (int(k), dtype, str(device))
    v = _ONES.get(key)
    if v is None:
        v = torch.ones((k, 1), dtype=dtype, device=device)
        _ONES[key] = v
    return v


def kernel(inputs):
    x = inputs[0]

    if x.numel() == 0 or x.shape[-1] == 0:
        return x.sum(dim=-1)

    n_last = int(x.shape[-1])
    rows = x.numel() // n_last
    dtype = x.dtype

    # 多行、大 K、连续 fp16/bf16：走 Cube GEMV 单 pass 路径
    use_gemv = (
        dtype in (torch.float16, torch.bfloat16)
        and rows >= 8
        and n_last >= 256
        and x.is_contiguous()
    )

    if use_gemv:
        x2 = x.reshape(rows, n_last)          # 连续张量：零拷贝 view
        w = _ones_col(n_last, dtype, x.device)
        y = torch.matmul(x2, w)               # (rows,1) fp32 累加，输出同 dtype
        return y.reshape(x.shape[:-1])        # 零拷贝 view，shape/dtype 对齐 baseline

    # 退化分支：单行/小张量/非连续/非半精度 -> 原生单 kernel reduce
    if not x.is_contiguous():
        x = x.contiguous()
    return x.sum(dim=-1)
