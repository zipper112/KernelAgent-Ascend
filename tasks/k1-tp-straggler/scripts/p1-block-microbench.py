#!/usr/bin/env python3
"""K1 p1 巨块候选分解微基准（harness 侧取证，替代会崩引擎的 profiler 端点）。

K1 档待分辨假设：p1 decode 段 0.1-1.8s 巨块 AivKernel = ①MoE/draft 长 vector 串（算子链，
可融合）还是 ②宿主侧卡顿的 device 显形（非算子）。
方法：在 e15 容器内单卡复现 p1 decode 一步的计算构成（MTP k=3 → 每步 draft 3+verify 4 行），
逐算子计时 draft/verify 两相位，看是否存在"单步内长 vector 串"。
用法：docker run ... ascend-triton:8.5.0-910b-arm64 python3 /p1-block-microbench.py
"""
import json
import time

import torch
import torch_npu

torch.manual_seed(42)
torch.npu.set_device(0)
DEV = "npu"
BF16 = torch.bfloat16

# GLM-5.3-Flash KDA 结构参数（K1/K8 档 + vllm-ascend glm5next config 读数）
HIDDEN = 4096
KDA_HEADS = 32
HEAD_DIM = 128
MOE_EXPERTS = 160          # 8 卡 EP 后每卡 ~20，但 eager 串在卡内
MOE_TOPK = 8
CONV_DIM = 4 * KDA_HEADS * HEAD_DIM // 4   # KDA conv 展平宽（K8 档 hidden=4×kda_dim 量级）
MTP_K = 3
B = 1 + MTP_K              # verify 行（draft 3 + 1）


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
    return s.elapsed_time(e) / n * 1000.0   # μs


def bench(name, parts):
    total = 0.0
    out = {}
    for pname, fn in parts:
        us = timed(fn)
        out[pname] = round(us, 1)
        total += us
    print(f"{name:28} total={total:8.1f}us  " + " ".join(f"{k}={v}" for k, v in out.items()))
    return total


print("== p1 decode 单步构成（MTP k=3：draft 3 tok ×3 步 + verify 4 tok ×1）==")

# --- draft 相位（B=1 行的小张量算子串；45 层中 KDA 相关部分的每层代表）---
q = torch.randn(1, KDA_HEADS, HEAD_DIM, dtype=BF16).npu()
k_ = torch.randn(1, KDA_HEADS, HEAD_DIM, dtype=BF16).npu()
v = torch.randn(1, KDA_HEADS, HEAD_DIM, dtype=BF16).npu()
state = torch.randn(1, KDA_HEADS, HEAD_DIM, HEAD_DIM, dtype=BF16).npu()
g = torch.randn(1, KDA_HEADS, 1, 1, dtype=BF16).npu()
beta = torch.randn(1, KDA_HEADS, 1, 1, dtype=BF16).npu()
x1 = torch.randn(1, HIDDEN, dtype=BF16).npu()
w_proj = torch.randn(HIDDEN, HIDDEN, dtype=BF16).npu() * 0.02

draft_parts = [
    ("l2norm(qk)", lambda: torch.nn.functional.normalize(q.float(), dim=-1).to(BF16)),
    ("gating_sigmoid", lambda: torch.sigmoid(g)),
    ("recurrent_update", lambda: torch.einsum("bhdf,bhd->bhf", state, k_)),   # 状态递推本体量级
    ("out_gather", lambda: torch.einsum("bhdf,bhf->bhd", state, v.squeeze(2) if v.dim() == 4 else v)),
    ("proj_matmul", lambda: x1 @ w_proj),
]
t_draft = bench("draft 每层代表串", draft_parts)

# --- verify 相位（B=4 行：+ logits/lm_head 全词表 154880）---
VOCAB = 154880
xv = torch.randn(B, HIDDEN, dtype=BF16).npu()
w_lm = torch.randn(HIDDEN, VOCAB, dtype=BF16).npu() * 0.01
verify_parts = [
    ("lm_head_logits(B=4)", lambda: xv @ w_lm),                     # K2 档中型 allreduce 的上游本体
    ("silu_gate", lambda: torch.nn.functional.silu(xv)),
    ("topk_moe_route", lambda: (xv @ torch.randn(HIDDEN, MOE_EXPERTS, dtype=BF16).npu()*0.02).topk(MOE_TOPK, dim=-1)),
]
t_verify = bench("verify 每层代表串", verify_parts)

# --- K9 的 8 连发（对照：每 chunk 固定开销本体量级）---
qkv_states = torch.randn(64, 3 * KDA_HEADS * HEAD_DIM, dtype=BF16).npu()
g1 = torch.randn(1, 64, KDA_HEADS, HEAD_DIM, dtype=BF16).npu()
idx_s = torch.arange(0, 64, 2, dtype=torch.int64).npu()
idx_n = torch.arange(1, 64, 2, dtype=torch.int64).npu()
buf = torch.zeros_like(qkv_states)
k9_parts = [
    ("6x_index_select", lambda: (qkv_states.index_select(0, idx_s), g1.index_select(1, idx_s), g1.index_select(1, idx_s),
                                  qkv_states.index_select(0, idx_n), g1.index_select(1, idx_n), g1.index_select(1, idx_n))),
    ("2x_index_copy", lambda: (buf.index_copy_(0, idx_s, qkv_states[idx_s]),
                                buf.index_copy_(0, idx_n, qkv_states[idx_n]))),
]
t_k9 = bench("K9 8连发代表(64tok)", k9_parts)

print(json.dumps({"draft_layer_us": t_draft, "verify_layer_us": t_verify, "k9_chunk_us": t_k9}))
