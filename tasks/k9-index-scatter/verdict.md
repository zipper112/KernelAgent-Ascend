# K9 最终判定：算子侧无净收益 —— 2026-09-26

## 微基准证据（e15 device 7，互补索引忠实复刻 kda.py 语义）

| n_tok | 原版 8 连发 | 融合版（单 select×3+合并写回） | speedup |
|---|---|---|---|
| 64 | 236.4 μs | 283.6 μs | 0.83x |
| 128 | 251.1 μs | 286.0 μs | 0.88x |
| 256 | 248.8 μs | 286.4 μs | 0.87x |

正确性全过（语义融合成立），性能证伪。

## 两条判据

1. **时长不随 n 变化**（64→256 恒 ~240-286μs）→ launch 固定成本主导，非带宽；
   device 侧削减派发次数的收益被 advanced-indexing 写回的等价开销吃掉；
2. msprof 侧单发 0.9-1.1ms vs 我们纯 device 计时 ~0.24ms → **~70% 耗时在 host 发射间隔**
   （eager Python 组织），属图化/调度问题。

## 结构性观察（第二融合形态评估）

select 之后两组立即分流进**两次独立 causal_conv1d 调用**（spec run_mode=1 vs ns
prefill/decode），输出端 zero_+两次 index_copy_ 拼回。把"select×6+conv×2+copy×2"
整体单流化才是大派发缩减——但那改变 conv 调用组织，属上游 #17299/#17301
（KDA prefill 融合三连）的领地，**上报跟进，非算子重组**。

## 处置

- k9-fused-v1 已按证据纪律 reject 入链（solutions.jsonl，stage=microbench）；
- K9 上报 research agent：算子侧已尽（微基准证伪 + host 间隔归属图化）；上游 #17299 合入后按档案"pick"策略处理；
- E068 timeline 取证（kda.py 调用栈归属）随 profiler 端点坑（落盘崩溃）搁置，msprof 原始数据在 /data02/kda/prof 可离线导出。
