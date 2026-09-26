#!/usr/bin/env python3
"""K3 数据重调研：Transpose/Cast 群在 e15 上的实测（档案判"薄"但从未在我们的环境测过）。
档案口径：Transpose 1384 次 3.8s（2.8ms avg）；Cast 57974 次 1.22s（0.021ms avg）。
问题：单发 2.8ms 的 Transpose 是什么形状？是否有更快的实现路径（contiguous vs transpose+copy）？
"""
import json

import torch
import torch_npu

torch.manual_seed(42)
torch.npu.set_device(0)
BF16 = torch.bfloat16


def timed(fn, n=30, warmup=8):
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
# W8A8 权重 NZ 布局适配的典型形状：[H, H] 与 [H, 4H] 的转置拷贝（bf16）
for name, shape in (("4096x4096", (4096, 4096)), ("4096x16384", (4096, 16384)),
                    ("12288x4096", (12288, 4096))):
    x = torch.randn(*shape, dtype=BF16).npu()
    t_contig = timed(lambda: x.transpose(0, 1).contiguous())          # transpose + copy（档案路径）
    t_tofmt = timed(lambda: x.transpose(0, 1).reshape(x.shape[1], x.shape[0]))  # 纯视图（零拷贝，可行性探针）
    out[name] = {"transpose_copy_us": round(t_contig, 1)}
    print(f"{name}: transpose+contiguous={t_contig:.1f}us  "
          f"(纯视图合法={torch.equal(x.transpose(0,1).reshape(shape[1],shape[0]), x.transpose(0,1))})")

# Cast 群：bf16→fp32 逐元素（57974 次的典型小张量）
for shape in ((4096,), (8, 4096), (8192, 4096)):
    x = torch.randn(*shape, dtype=BF16).npu()
    t = timed(lambda: x.float())
    out[f"cast_{shape}"] = round(t, 1)
    print(f"cast bf16→fp32 {shape}: {t:.1f}us")

print(json.dumps(out))
