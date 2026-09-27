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

## 战役 3 实测档案（2026-09-27，17 轮）
- 参照系：vec 基线 1102μs；战役 3 最优 c008=978.7μs（-11.2%，dev 档；full 档退化 1506=过拟合）
- verify 过的候选谱系：c006 window-carry 1000.4 / c007 constexpr-elide 1133.8 / c008 978.7 /
  c010 neverraise-ring 1274.6——全部在 980-1270μs 波动，无一接近 600
- 教训：launch 开销优化路线（constexpr-elide/stream-pinned/direct-launch）已系统性探索，
  天花板 ~980μs。剩余差距必须靠**减少每步实际计算量/访存量**（算法层）而非派发层

## 首推方向（battle1 实证可达 606μs——必须先试）
fused-single-launch + 寄存器滑窗 + 静态 SEQ 展开（battle1 实测 606μs 的真实结构）：
1. 单 kernel 单 launch：读 3 槽状态→4-tap 窗卷积→silu→写新状态+输出一次完成
2. (r0,r1,r2) 三寄存器持滑窗，逐 token 右滑——全程零中间张量、零 gather/scatter
3. SEQ 作 tl.constexpr 静态展开（L=1 或 4），去运行期循环依赖
4. grid=(B, ceil(D/BLOCK))，BLOCK 按 triton-ascend-reduction-case 的核数粒度结论取
   （40 核满载，64 程序≈40 核+尾波可接受）
5. 严禁 fallback 分支（c010 的 neverraise 结构实测反而慢 30%）

## 战役 5 勘误（2026-09-27 监督审计）
- c020 的 257.2μs 是 w01（B=1）口径——**w02（B=16）实测 1103.7μs，未赢 vec 基线 1102，目标未达**。
  终局报告的"<600 达标"系 dev 集选错（w01 而非声明的 w02），非 agent 作弊；口径已修（bench.py 尊重 task.yaml dev 声明）。
- 真实状态：B=1 已优（257），B=16 无改善（~1104 持平基线）——**瓶颈在批量维**（每 batch 派发/索引/状态访问未摊薄）。
  下一战役主攻：B=16 时 grid 布局把 B 维并行吃满（grid=(B*ceil(D/BLOCK)) 或按 B 分核），参考 ../../knowledge 里 conv/reduction 案例。
- router 查询路径已修（AGENTS.md 锚定仓库根）。
