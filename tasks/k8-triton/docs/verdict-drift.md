# stop-drift 终局（round 2，自动生成）

连续 3 轮机器漂移判定非 ADVANCED（数据源 benchmark.csv，不采信 LLM 自评）。

## bench 证据
（尚无 bench 证据）

## 已试方向
- c001 [reject] single-launch-fused-rotate-window（Triton-Ascend 单 kernel，零中间张量）
- c001 [reject] single-launch-fused-rotate-window（Triton-Ascend 单 kernel，零中间张量）
- c001 [revise] single-launch-fused-rotate-window（Triton-Ascend 单 kernel，零中间张量）
- c001 [reject] single-launch-fused-rotate-window（Triton-Ascend 单 kernel，零中间张量）
- c002 [reject] fused-single-launch-tensorized（同方向修复：r1 单 kernel 寄存器携带窗口保留，全面去标量化）
- c002 [reject] fused-single-launch-tensorized（同方向修复：r1 单 kernel 寄存器携带窗口保留，全面去标量化）
