# 前战役判定（冷启动 feedback——battle1 实测档案）

## 已证实的性能事实（可信——微基准实测）
- c002 vec 版基线：B=16 p50 1102us（其中纯计算 <50us——95% 是派发与中间流量税）
- battle1 最优 Triton 版实测 606us（-44.7% vs vec），已验证正确——但代码因 cid 复用 bug 丢失
- 目标：B=16 p50 < 600us（>1.8x）

## 606us 版的技术路线（从评审记录还原——重写时按此路线优先）
- fused-single-launch：单 kernel 单 launch，全部融合（读 state→4 元素窗卷积→silu→写 state+输出）
- 寄存器滑窗（regwindow）：(r0,r1,r2) 三寄存器持状态，逐 token 右滑，无中间张量
- static SEQ 展开（MTP k=3 → L=1 或 4，constexpr 静态展开去依赖）
- pad 槽语义：只触碰 indices 命中槽（gather/scatter），未命中槽零副作用

## 已判死方向（battle1/2 实测否决——勿回锅）
- depthwise-conv1d-fused-window、groups-d-conv1d-dispatch-fold（派发税没省掉）
- persistent40 系列（40 block 常驻——grid 太小 NPU 上无收益）
- flat-tile + idxgather 组合（gather 税吃掉滑窗收益）

## 失败教训（BitLesson 提炼）
- 输出必须 250 行内紧凑（长输出被思维链吃光预算→空 content）
- kernel() 签名严格 kernel(inputs)->Tensor（inputs 与 reference.py 一致），别自创接口

## 种子候选（battle2 遗产，已恢复）
- solution/c019/candidate.py：单派发版，verify 全过（err_ratio 0.0/0.0/0.0，83 行）
- 冷启动第一轮应直接 bench c019 建立真实基线行，writer 从它起步精修（勿从零重写）
