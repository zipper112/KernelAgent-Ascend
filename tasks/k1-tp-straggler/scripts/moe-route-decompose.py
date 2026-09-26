#!/usr/bin/env python3
"""K1 p1 巨块候选深挖：MoE 路由 topk 26ms 异常的分解验证。
问题：26ms 是真算子时长还是张量构造（torch.randn(HIDDEN, E) 在 NPU 上）混进了计时？
分解：router_matmul / topk / randn 构造 分别计时 + 真实 scale 修正。"""
import json
import time

import torch
import torch_npu

torch.manual_seed(42)
torch.npu.set_device(0)
BF16 = torch.bfloat16
HIDDEN, E, TOPK, B = 4096, 160, 8, 4


def timed(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / n * 1000.0


xv = torch.randn(B, HIDDEN, dtype=BF16).npu()
w_route = (torch.randn(HIDDEN, E, dtype=torch.float32) * 0.02).to(BF16).npu()   # 构造在计时外

t_matmul = timed(lambda: xv @ w_route)
logits = xv @ w_route
t_topk = timed(lambda: logits.topk(TOPK, dim=-1))
t_fused = timed(lambda: (xv @ w_route).topk(TOPK, dim=-1))
t_softmax_topk = timed(lambda: torch.softmax(logits.float(), dim=-1).to(BF16).topk(TOPK, dim=-1))

# 对照：正常量级感——B=4 行的 GEMV 应在百 us 级；26ms 意味着什么
t_matmul_b1 = timed(lambda: xv[:1] @ w_route)

print(json.dumps({
    "router_matmul_B4_us": round(t_matmul, 1),
    "topk_alone_us": round(t_topk, 1),
    "matmul+topk_us": round(t_fused, 1),
    "softmax_topk_us": round(t_softmax_topk, 1),
    "router_matmul_B1_us": round(t_matmul_b1, 1),
}, ensure_ascii=False))
