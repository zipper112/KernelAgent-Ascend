# K11 verdict：c001 static-D4096 fused index kernel

## 结论

- **达标 keep：`solution/c001/candidate.py`。**
- canonical 基线：c000（上游等价串原样包装），w02 `318.4us`。
- c001 全量 verify：`3/3 PASS`（w01/w02/w03，err_ratio 均为 0）。task.yaml 声明的 verify mode 为 single-step；本入口未实际启用 chained 门。
- 性能（NPU 7 无他人进程）：w01 `108.6us`（-70.9%）、w02 `116.8us`（-63.3%）、w03 `350.1us`（-49.8%）。w02 为基线×`0.3668`，优于目标×0.7，且全形状不退化。
- 知识库研究零命中，已按盲区条款记录于 `docs/draft.md`；本候选未假设索引连续，运行时逐行读取索引。

## 实现

- 单 Triton kernel 融合两段 gather、concat 布局与 scatter 写回：目标行 `<k` 读 `idx_spec[row]`，`[k,k+m)` 读 `idx_ns[row-k]`，尾部行写零。
- `D=4096/BLOCK_D=4096` 静态特化，按行启动，完整保留任意索引语义。
- 首调标准发射后捕获 `CompiledKernel`，用独立探针逐位比对后底层直发；失败自动回退标准 JIT 发射。8 深输出环消除热路径分配。

## 当轮 canonical 输出原文

```text
solution: c000  set: full  n: 3
  w01: mean=373.3us p50=382.1 p99=398.6 ref=390.96400141716003
  w02: mean=318.4us p50=321.2 p99=326.6 ref=323.0240046977997
  w03: mean=697.4us p50=699.2 p99=700.2 ref=695.5680012702942

mean: 463.0us  p50: 382.1  p99: 700.2  speedup_vs_ref: 1.02x
recorded -> /data01/mahaolong/KAgent/tasks/k11-index-fused/docs/benchmark.csv
```

```text
solution:   c001
mode:       single-step (dev=False, n=3)
  w01: PASS  steps=None state_inputs=None err_ratio=0.0
  w02: PASS  steps=None state_inputs=None err_ratio=0.0
  w03: PASS  steps=None state_inputs=None err_ratio=0.0

passed: 3/3
```

```text
solution: c001  set: full  n: 3
  w01: mean=108.6us p50=118.5 p99=140.9 ref=391.65599942207336
  w02: mean=116.8us p50=119.3 p99=121.1 ref=338.4600043296814
  w03: mean=350.1us p50=349.0 p99=374.3 ref=699.9639987945557

mean: 191.8us  p50: 119.3  p99: 374.3  speedup_vs_ref: 2.83x
recorded -> /data01/mahaolong/KAgent/tasks/k11-index-fused/docs/benchmark.csv
```
