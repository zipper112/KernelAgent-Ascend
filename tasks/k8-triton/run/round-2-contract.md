# round-2 契约（harness 自动生成）

direction: fused-single-launch-tensorized（同方向修复：r1 单 kernel 寄存器携带窗口保留，全面去标量化）
hypothesis: r1 空 err_ratio = 执行层挂而非数值超差，首嫌 triton-ascend 标量路径（scalar tl.load / scalar tl.where / 标量-张量混合 mask）；全部改为 (1,)/(BLOCK_D,) 张量运算后可过 verify，单 launch 零中间张量天然压制 95% 派发税 → p50 < 600us
candidate: c002
parent: （根）
success_criteria: 首个可测基线（verify 过 + bench 出数）
blocking: 上轮评审要点见 run/memory/round-1.json
