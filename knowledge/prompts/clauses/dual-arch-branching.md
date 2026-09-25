# 条款：双代际分支声明（跨 2201/3510 的任务必选）

## 背景

知识按 DAV_2201（A2/A3/910B 系，UB 192KB）与 DAV_3510（950 系，UB 248KB）双代际组织；部分 API/路线仅限单代际（blaze 仅 950、部分 API 限 A2/A3）。阈值表（Bound 判定、bank conflict、occupancy 相关）两代不同。

## 规则

1. 任务启动时：`config.yaml` 填 `arch`（以 get_npu_arch.py 检测为准，npu-smi Chip Name 不可信）；
2. 知识检索强制带 `--arch` 过滤；索引条目 `arch: both` 之外的不跨代引用；
3. UB 容量等硬约束在 prompt 里给**本代际数值**（2201: 192KB / 3510: 248KB），不得混用；
4. 跨代际移植方向（如 910B 结果迁 950）：视为新方向走完整 Phase 1-2，不许"直接套用结论"；
5. 性能数字引用时六字段中的"代际"字段必填（`arch: dav_2201`）。

## 附：当前两代际的知识分布差异（v0.0）

- 2201：ascendc-performance-best-practices 主体、triton-ascend 全套（akg）、ops-profiling 阈值表现成；
- 3510：ascendc-blaze（MatMul 路线）、ascendc-regbase（RegTensor/asc_vf_call）、ascendc-simt-* 系列。
