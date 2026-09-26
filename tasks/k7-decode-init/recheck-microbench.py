#!/usr/bin/env python3
"""K7 数据调研：档案的 405-644ms 单发初始化事件在纯 device 计时下是什么量级？
若是真实 device 时间 → 病理（值得查算子替换）；若是首次分配/host 开销 → 调度问题（上报）。
形状按 K7 档推断：draft KV 池（segs×heads×dim×dtype 级）与 expert 状态。"""
import json

import torch
import torch_npu

torch.manual_seed(42)
torch.npu.set_device(0)
BF16 = torch.bfloat16


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


out = {}
# ZerosLike：draft KV 池量级（max_seqs 32 × 45 层不进单次；单次=一层的池段）
for name, shape in (("kv_32x32x128x512", (32, 32, 128, 512)),      # ~64M elem bf16 = 128MB
                    ("kv_8x32x128x512", (8, 32, 128, 512)),
                    ("expert_state_160x4096", (160, 4096))):
    x = torch.randn(*shape, dtype=BF16).npu()
    t = timed(lambda: torch.zeros_like(x))
    out[f"zeros_like_{name}"] = round(t, 1)
    print(f"zeros_like {name} ({x.numel()*2//1024//1024}MB): {t:.1f}us")

# ForeachAddListV2：优化器状态列表级
params = [torch.randn(4096, 4096, dtype=BF16).npu() for _ in range(8)]
deltas = [torch.randn(4096, 4096, dtype=BF16).npu() for _ in range(8)]
t = timed(lambda: torch._foreach_add_(params, deltas, alpha=0.1))
out["foreach_add_8x4096x4096"] = round(t, 1)
print(f"foreach_add_ 8×4096² bf16 ({8*4096*4096*2//1024//1024}MB×2): {t:.1f}us")

# IndexSelect Slice 大池 gather
pool = torch.randn(1024, 4096, dtype=BF16).npu()
idx = torch.arange(64, device="npu")
t = timed(lambda: pool.index_select(0, idx))
out["index_select_1024x4096_take64"] = round(t, 1)
print(f"index_select pool 1024×4096 take64: {t:.1f}us")

print(json.dumps(out))
