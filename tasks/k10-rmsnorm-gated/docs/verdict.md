# K10 verdict：c001 static-D128 direct-launch

## 结论

- **达标 keep：`solution/c001/candidate.py`。**
- 正确基线：c000（原样导入容器内上游 `rms_norm_gated`），w02 `417.2us`。
- c001 全量 verify：`3/3 PASS`（w01/w02/w03，err_ratio 均为 0）。
- 性能（canonical bench，NPU 7 无他人进程）：w01 `104.2us`（-74.6%）、w02 `125.8us`（-69.8%）、w03 `128.7us`（-69.1%）；w02 达到基线×`0.3016`，优于目标×0.7，且全形状不退化。
- 旧 CSV 首行 c000 `603.9us` 来自任务目录初始误用 eager 数学包装，不符合“原样上游”契约，不作为本轮比较基线；`2026-09-27T13:58:43` 行才是修正后的 canonical c000。

## 实现

- 单 Triton kernel 一次读 `x/g`，fp32 完成 RMS 归约、weight/bias、sigmoid gate 与 bf16 写回。
- `D=128`、`BLOCK_ROWS=32` 静态化；仅保留行尾 mask，消除上游 D mask、rstd 临时输出与 reshape/contiguous 包装。
- 首调标准发射后捕获 `CompiledKernel`，探针逐位比对后走底层直发；4 深输出环避免热路径输出分配。探针失败自动回退标准 JIT 发射。

## 当轮 canonical 输出原文

```text
solution: c000  set: full  n: 3
  w01: mean=410.3us p50=421.1 p99=455.1 ref=478.1239986419678
  w02: mean=417.2us p50=417.2 p99=426.6 ref=488.05999755869375
  w03: mean=416.3us p50=423.0 p99=432.5 ref=525.1919984817505

mean: 414.6us  p50: 421.1  p99: 455.1  speedup_vs_ref: 1.20x
recorded -> /data01/mahaolong/KAgent/tasks/k10-rmsnorm-gated/docs/benchmark.csv
```

```text
solution: c001
mode:       single-step (dev=False, n=3)
  w01: PASS  steps=None state_inputs=None err_ratio=0.0
  w02: PASS  steps=None state_inputs=None err_ratio=0.0
  w03: PASS  steps=None state_inputs=None err_ratio=0.0

passed: 3/3
```

```text
solution: c001  set: full  n: 3
  w01: mean=104.2us p50=109.3 p99=127.0 ref=525.9680032730103
  w02: mean=125.8us p50=115.8 p99=166.2 ref=528.2240033149719
  w03: mean=128.7us p50=129.5 p99=136.4 ref=527.9759883880615

mean: 119.6us  p50: 115.8  p99: 166.2  speedup_vs_ref: 4.45x
recorded -> /data01/mahaolong/KAgent/tasks/k10-rmsnorm-gated/docs/benchmark.csv
```
