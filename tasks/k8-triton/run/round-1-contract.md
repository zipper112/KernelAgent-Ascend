# round-1 契约（harness 自动生成）

direction: fused-single-launch 寄存器携带窗口（gather→4-tap conv→silu→原地滑写 state，零中间张量）
hypothesis: None
candidate: c001
parent: （根）
success_criteria: 首个可测基线（verify 过 + bench 出数）
blocking: 上轮评审要点见 run/memory/round-0.json
