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

---

# K11 Campaign 2 verdict：c008 dimension-bucketed fused index kernel

## 结论

- **达标 keep：`solution/c008/candidate.py`（父方向 c001）。**
- c008 全量 verify：**15/15 PASS**，覆盖原三形状与 T={1024,8192}、D={2048,8192}、
  k:m={1:1,1:3,7:1} 全交叉；err_ratio 全部为 0。正确性无适用域限制。
- c001 原样跑扩集在首个 D=2048 workload 失败：`ValueError: index-fused candidate requires D=4096`，
  因此 c001 性能/正确性适用域为 D=4096。
- 尾部 20 样本专项：w03 c001 mean/p50/p99=`349.6/346.3/391.5us`；
  BLOCK_T=2=`347.5/351.5/365.4us`；BLOCK_T=4=`341.7/340.0/361.2us`。采用 T>=16384→BT4、
  T>=8192→BT2。
- 原三形状 5% 门：同锁 c001 originals mean 为 w01/w02/w03=`100.0/121.0/351.4us`；
  c008 canonical 为 `98.9/120.5/349.4us`，分别 **-1.1%/-0.4%/-0.6%**，无退化。
- CompiledKernel 直发只传运行期参数，不重复传 constexpr；c002/c003 曾因该签名误用回退 JIT。

## 分桶

- D=4096 且 T<8192：保留 c001 字面 constexpr 单行整列 kernel。
- 其他 D：`BLOCK_D=min(4096,next_pow2(D))`；D=8192 为 2 个 4096 列程序。
- T>=8192 且 D<=4096：行程序内串行 BLOCK_T 行；T>=16384 用 4，否则用 2。
- k:m 不做 constexpr 分档：运行期 `k/m` 分支已覆盖，实测三档均保持 1.90x 以上加速。
- 盲区：D=8192 单块 BLOCK_D=8192 未得到干净样本（并发容器 ACL Resource_Busy），不采用、不宣称。

## 维度适用域表（c008 canonical full bench）

| workload | T | D | k:m | mean us | p50 us | p99 us | baseline us | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w01 | 64 | 4096 | 3:1 | 98.9 | 105.2 | 114.1 | 332.3 | 3.36x |
| w02 | 4096 | 4096 | 3:1 | 120.5 | 120.2 | 130.5 | 279.4 | 2.32x |
| w03 | 16384 | 4096 | 3:1 | 349.4 | 348.9 | 359.5 | 699.7 | 2.00x |
| w04 | 8192 | 2048 | 1:1 | 126.2 | 123.1 | 161.3 | 292.9 | 2.32x |
| w05 | 8192 | 2048 | 1:3 | 121.8 | 122.0 | 132.5 | 292.0 | 2.40x |
| w06 | 8192 | 2048 | 7:1 | 129.4 | 126.5 | 143.0 | 288.5 | 2.23x |
| w07 | 8192 | 8192 | 1:1 | 328.4 | 332.1 | 344.6 | 675.2 | 2.06x |
| w08 | 8192 | 8192 | 1:3 | 326.8 | 325.9 | 344.8 | 673.5 | 2.06x |
| w09 | 8192 | 8192 | 7:1 | 363.0 | 360.5 | 380.0 | 691.4 | 1.90x |
| w10 | 1024 | 2048 | 1:1 | 108.8 | 109.5 | 121.6 | 300.8 | 2.76x |
| w11 | 1024 | 2048 | 1:3 | 116.1 | 113.0 | 130.3 | 310.9 | 2.68x |
| w12 | 1024 | 2048 | 7:1 | 129.6 | 124.3 | 156.8 | 318.5 | 2.46x |
| w13 | 1024 | 8192 | 1:1 | 112.4 | 107.6 | 127.6 | 308.8 | 2.75x |
| w14 | 1024 | 8192 | 1:3 | 108.4 | 107.9 | 119.9 | 303.8 | 2.80x |
| w15 | 1024 | 8192 | 7:1 | 111.4 | 107.1 | 131.4 | 306.0 | 2.75x |

## 当轮 canonical 输出原文

```text
solution: c008
mode:     single-step (dev=False, n=15)
  w01: PASS  steps=None state_inputs=None err_ratio=0.0
  w02: PASS  steps=None state_inputs=None err_ratio=0.0
  w03: PASS  steps=None state_inputs=None err_ratio=0.0
  w04: PASS  steps=None state_inputs=None err_ratio=0.0
  w05: PASS  steps=None state_inputs=None err_ratio=0.0
  w06: PASS  steps=None state_inputs=None err_ratio=0.0
  w07: PASS  steps=None state_inputs=None err_ratio=0.0
  w08: PASS  steps=None state_inputs=None err_ratio=0.0
  w09: PASS  steps=None state_inputs=None err_ratio=0.0
  w10: PASS  steps=None state_inputs=None err_ratio=0.0
  w11: PASS  steps=None state_inputs=None err_ratio=0.0
  w12: PASS  steps=None state_inputs=None err_ratio=0.0
  w13: PASS  steps=None state_inputs=None err_ratio=0.0
  w14: PASS  steps=None state_inputs=None err_ratio=0.0
  w15: PASS  steps=None state_inputs=None err_ratio=0.0

passed: 15/15
```

```text
solution: c008  set: full  n: 15
  w01: mean=98.9us p50=105.2 p99=114.1 ref=332.27999210357666
  w02: mean=120.5us p50=120.2 p99=130.5 ref=279.35200333595276
  w03: mean=349.4us p50=348.9 p99=359.5 ref=699.727988243103
  w04: mean=126.2us p50=123.1 p99=161.3 ref=292.9159998894738
  w05: mean=121.8us p50=122.0 p99=132.5 ref=292.0239984980166
  w06: mean=129.4us p50=126.5 p99=143.0 ref=288.45600485801697
  w07: mean=328.4us p50=332.1 p99=344.6 ref=675.1999979250793
  w08: mean=326.8us p50=325.9 p99=344.8 ref=673.4880089759827
  w09: mean=363.0us p50=360.5 p99=380.0 ref=691.4400100708008
  w10: mean=108.8us p50=109.5 p99=121.6 ref=300.8239984512329
  w11: mean=116.1us p50=113.0 p99=130.3 ref=310.94000555808057
  w12: mean=129.6us p50=124.3 p99=156.8 ref=318.46399903297424
  w13: mean=112.4us p50=107.6 p99=127.6 ref=308.82800221443176
  w14: mean=108.4us p50=107.9 p99=119.9 ref=303.81200313598115
  w15: mean=111.4us p50=107.1 p99=131.4 ref=306.01200461387634

mean: 176.7us  p50: 122.0  p99: 380.0  speedup_vs_ref: 2.46x
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
