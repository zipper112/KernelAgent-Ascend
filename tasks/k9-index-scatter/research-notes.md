# K9 融合 patch 设计（kda.py spec/non-spec 分组 8→4 派发）

## 病灶（已核实源码 vllm_ascend/models/glm5next/kda.py@bea70ab8f）

spec-decode 激活时每 chunk 固定 8 次 depthwise gather/scatter：
- :404-406 三连 `index_select`（qkv/g1/beta 按 spec_token_indx）
- :408-410 三连 `index_select`（同张量按 non_spec_token_indx）
- :485 `index_copy_(0, spec_token_indx, ...)` 写回
- :528 `index_copy_(0, non_spec_token_indx, ...)` 写回

E068 证据：每 chunk 5 连发（含 conv/recurrent state 装配）× chunk 数线性累积，16k/64k 输入进 prefill top10（~1.4% busy，结构性）。

## 融合方案（torch 级，无上游依赖）

**关键观察**：spec_token_indx 与 non_spec_token_indx 是**互补划分**（一个 token 要么 spec 要么 non-spec）。
因此 6 次 index_select 可合成 **2 次派发**：
1. 构造 `order = cat([spec_idx, non_spec_idx])`（一次小 cat，或 metadata builder 直接给合并索引+分界点——上游 PR 可省）
2. 每张量单次 `index_select(dim, order)` → 切两段即 spec/ns 视图（零拷贝切片）
   - qkv: `qkv_all = qkv_proj_states.index_select(0, order)`；`qkv_spec, qkv_ns = split at len(spec)`
   - g1/beta: 同法 `index_select(1, order)` 一次 + 切段

写回 2 次 index_copy_ 同理合成：`out_full = empty; out_full[order] = cat([spec_out, ns_out])`（或单次 index_copy_ 用 order）。

**8 → 4 次派发**（3 张量 × 2 组 select 合为 3 次单 select + 1 次合并写回），每次派发都是 [n,*] 全张量带宽活——减半即接近线性收益。

## 边界（须保留的语义）

- 纯 spec 步（non_spec 为空）已有 identity 快路径（:395-402）——不动；
- index_select 后的 `.contiguous()` 等存储假设按原样保持（index_select 本身产新连续张量）；
- 互补性若被 metadata builder 打破（两索引有交集），方案不成立——**实施前先加断言验证互补划分**（torch 级一次 bincount 检查，仅 debug 模式跑）。

## 工作量与预期

- patch ~40 行（kda.py 单文件），worktree 打上即测；
- 预期收益：scatter/gather 群时长 ~减半 ≈ prefill busy -0.7%（薄肉，档案排位正确——低于 K1/K8 高于 K3）；
- 依赖：上游 #17299/#17301 若合入则按档案"pick 上游"（上报跟进，非我们做）。
