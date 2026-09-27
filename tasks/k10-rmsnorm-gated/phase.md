# Phase 任务书：K10 rms_norm_gated（K1 池第一靶）

先读同目录 AGENTS.md（工作流+接口契约+已验证打法线索）。

## 目标

**w02（T=4096 prefill）mean_us 显著优于上游基线**；w01/w03 不退化。
基线数字：首轮 bench 上游实现（见下）自建——把上游 rms_norm_gated 包装为
solution/c000/candidate.py 跑 bench --record 建立参照（c000 即基线，勿优化它）。
（体积感参考：K8 同病根治理拿到 -63%；此处病根=包装税+小核本体，非算法差）

## 上游基线代码（c000 用这个）

从容器内 /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/kda/kda.py 的
rms_norm_gated + layer_norm_gated_fwd 原样拷贝（接口适配成 kernel(inputs) 形态：
x,g,weight,bias = inputs; return rms_norm_gated(x,g,weight,bias,"sigmoid")）。

## 已试方向账本

首战，无历史。K8 姊妹任务的教训见 AGENTS.md"已知打法线索"（派发层天花板教训：
包装税省完后若仍不达标，差距在访存模式——考虑 D=128 整行驻寄存器的 tile 策略）。

## 工作节奏

研究（主动查 router/skills）→ c000 基线 bench → 内环 verify --fast 几十次 →
全量 verify → bench --record → 记账 → 有改进才 git commit → 换向。

## 完成判据

w02 mean_us < c000 基线 ×0.7（-30% 起步，达标后可继续冲）且全形状不退化 →
keep + commit + verdict.md 终局报告（附当轮 canonical 输出原文）。
