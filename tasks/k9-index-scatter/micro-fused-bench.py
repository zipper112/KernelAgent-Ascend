#!/usr/bin/env python3
"""K9 融合微基准：8 连发（6×index_select + 2×index_copy_） vs 融合版（3×单次 select + 1×合并写回）。
形状按 K9 档：[B, dim, state_len] state 装配/写回，B=chunk 相关（64/128/256 tok 档）。"""
import json

import torch
import torch_npu

torch.manual_seed(42)
torch.npu.set_device(0)
BF16 = torch.bfloat16

D = 3 * 32 * 128          # qkv 展平宽（32 头 × 128 维 × 3）
H = 32


def timed(fn, n=50, warmup=10):
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


results = {}
for n_tok in (64, 128, 256):
    qkv = torch.randn(n_tok, D, dtype=BF16).npu()
    g1 = torch.randn(1, n_tok, H, 128, dtype=BF16).npu()
    beta = torch.randn(1, n_tok, H, 1, dtype=BF16).npu()
    n_spec = n_tok // 2
    idx_s = torch.randperm(n_tok, device="npu")[:n_spec].long()
    idx_n = torch.tensor([i for i in range(n_tok) if i not in set(idx_s.tolist())],
                         device="npu").long()   # 互补划分（kda.py 语义）
    assert idx_s.numel() + idx_n.numel() == n_tok

    # 原版 8 连发
    buf = torch.zeros_like(qkv)
    def orig():
        qs, qn = qkv.index_select(0, idx_s), qkv.index_select(0, idx_n)
        gs, gn = g1.index_select(1, idx_s), g1.index_select(1, idx_n)
        bs, bn = beta.index_select(1, idx_s), beta.index_select(1, idx_n)
        buf.index_copy_(0, idx_s, qs)
        buf.index_copy_(0, idx_n, qn)
        return qs, qn, gs, gn, bs, bn

    # 融合版：互补 → 单次 select + 切段；单次合并写回（scatter via 索引赋值一次）
    order = torch.cat([idx_s, idx_n])
    def fused():
        qa = qkv.index_select(0, order)
        qs, qn = qa[:n_spec], qa[n_spec:]
        ga = g1.index_select(1, order)
        gs, gn = ga[:, :n_spec], ga[:, n_spec:]
        ba = beta.index_select(1, order)
        bs, bn = ba[:, :n_spec], ba[:, n_spec:]
        buf2 = torch.empty_like(qkv)
        buf2[order] = qa                       # 单次合并写回
        return qs, qn, gs, gn, bs, bn

    t_o = timed(orig)
    t_f = timed(fused)
    # 正确性对照
    o = orig(); f = fused()
    ok = all(torch.equal(a, b) for a, b in zip(o[:2], f[:2])) and \
         torch.equal(o[2], f[2]) and torch.equal(o[4], f[4])
    results[n_tok] = {"orig_us": round(t_o, 1), "fused_us": round(t_f, 1),
                      "speedup": round(t_o / t_f, 2), "correct": ok}
    print(f"n={n_tok}: orig={t_o:.1f}us fused={t_f:.1f}us speedup={t_o/t_f:.2f}x correct={ok}")

print(json.dumps(results))
