# K1 算子面取证结论（p1 段 27 巨块分解）—— 2026-09-26

## 问题（K1 档遗留）

37 个 >20ms 巨块中 10 个与巨型 allReduce 对齐（=TP 等齐，非算子）；**其余 27 个出现在
p1 单流 decode 段（0.1-1.8s 渐减），档案待分辨**：MoE/draft 长 vector 串（算子链，可融合）
还是宿主侧卡顿的 device 显形（非算子）。

## 方法

e15 复刻环境（同镜像 f505184be15b/同形态）两路取证：
1. torch profiler 端点：trace 落盘但引擎死于收尾（v0.30+Ascend profiler 落盘崩溃，
   FRAMEWORK 层导出为空）——该路径不可用（记为工具坑）；
2. **harness 微基准分解**（单卡直接复现 p1 单步计算构成，逐算子计时）：稳定可复现。

## 数据（device 7，torch 2.7.1+torch_npu 2.7.1，MTP k=3 形态）

| p1 单步构成 | 每层代表串实测 | 折算（45 层×每步相位次数） |
|---|---|---|
| draft 相位（B=1） | 399.6μs（l2norm 153/recurrent 107/out_gather 92/proj 28/sigmoid 20） | ~54ms/步（3 次 draft ×45 层量级） |
| verify 相位（B=4） | lm_head 3.3ms + MoE 路由 62μs + silu 23μs | ~3.4ms 主导 |
| K9 8 连发（对照） | 323.8μs/chunk | 线性随 chunk |

（首版 26ms"topk 异常"为测量 bug——randn 构造混入计时，已在分解脚本修正。）

## 结论

**p1 巨块不是算子链**：单步内最大算子量级 lm_head 3.3ms，距 0.1-1.8s 差两个数量级；
全部算子串折算 <100ms/步。27 巨块的构成指向**宿主侧卡顿的 device 显形**（档案假设 B 后半），
与 K1 档"渐减趋势"（宿主 jitter 随负载稳定而减弱）一致。

## 处置

1. **K1 的 p1 巨块上报为非算子问题**（宿主/调度侧，移交 research agent 裁定——与
   torch.compile 评估同属"消灭 eager 发射"主题）；
2. 算子侧在此形态下无 K1 可做项；**K8 已收割的 -34% p1 TPOT 即本形态算子层的天花板收获**；
3. K9（scatter/gather 8→4 融合）为剩余唯一算子项，薄肉（~0.7% prefill busy），方案已备
   （tasks/k9-index-scatter/research-notes.md）。

## 工具沉淀

- 微基准分解法比 profiler 端点稳（不崩引擎、可重复、逐算子归因）——纳入 harness 取证常备；
- profiler 端点坑记录：start/stop 成功但落盘阶段 TBE 子进程全灭、EngineDead；原始
  msprof 数据（/data02/kda/prof，8 rank）已保全可日后用 msprof 离线导出。
