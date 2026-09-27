# K10 dimension robustness matrix

Canonical run: `python bench.py --solution solution/c600/candidate.py --full --record`, NPU 7 idle, 96 Cartesian workloads, 3 warmup + 5 samples, L2 cleared between samples. Correctness in the same run contract: `96/96 PASS, err_ratio=0`. Cell format is mean time (`speedup vs same-run reference`).

## Final Gate

|Workload|Shape|c001 keep|c600 final|Delta|
|---|---|---:|---:|---:|
|w01|(64,4,128)|104.2us|96.2us|-7.7%|
|w02|(4096,4,128)|125.8us|112.4us|-10.7%|
|w03|(16384,4,128)|128.7us|116.3us|-9.6%|

Aggregate c600 matrix: mean `153.4us`, p50 `113.5us`, p99 `924.2us`, mean speedup `5.18x`.

## Dispatch

|D|Kernel/rows|Boundary|Fallback outside sweep|
|---:|---|---|---|
|64|single tile; `min(32,next_pow2(rows))` below 2,097,152 elements, 64 rows to 8,388,607, 128 rows above|bandwidth region|fp32 torch expression|
|128|rows<=16384: row8 x N tiles, nw1; N=2 for rows<=256 and N=4 above. rows>16384: single tile; 32 rows through 65536, then 64 rows|small/medium vs bandwidth region|fp32 torch expression|
|256|single tile; `min(16,next_pow2(rows))` below 8,388,608 elements, 32 rows above|bandwidth region|fp32 torch expression|

Launch state and the four-buffer output ring are stored in one LRU entry per shape; at most two shapes remain resident, preventing a 96-shape sweep from retaining every large output. The first call probes direct launch against the standard JIT launch and falls back automatically on mismatch.

## D=64

|T\H|2|4|8|16|
|---:|---:|---:|---:|---:|
|1|113.2 (4.77x)|109.2 (5.09x)|101.0 (5.24x)|109.9 (4.82x)|
|64|101.7 (5.29x)|112.9 (4.81x)|111.0 (4.76x)|108.0 (4.88x)|
|256|111.8 (4.77x)|114.3 (4.77x)|112.1 (4.79x)|111.1 (4.77x)|
|1024|111.3 (4.80x)|113.4 (4.67x)|117.6 (4.55x)|114.0 (4.83x)|
|4096|111.2 (4.78x)|114.6 (4.68x)|114.3 (4.81x)|111.2 (4.85x)|
|8192|115.8 (4.73x)|111.5 (4.92x)|115.3 (4.87x)|119.9 (4.49x)|
|16384|118.4 (4.44x)|108.8 (5.02x)|120.5 (4.48x)|189.6 (5.27x)|
|32768|107.3 (11.57x)|121.2 (4.36x)|188.7 (5.38x)|320.6 (6.44x)|

## D=128

|T\H|2|4|8|16|
|---:|---:|---:|---:|---:|
|1|109.3 (4.75x)|110.0 (4.85x)|111.1 (4.53x)|118.8 (4.64x)|
|64|108.8 (4.88x)|96.2 (5.56x)|108.0 (4.69x)|112.8 (4.85x)|
|256|106.8 (4.95x)|113.6 (4.73x)|108.1 (4.76x)|111.2 (4.98x)|
|1024|107.3 (4.90x)|118.0 (4.53x)|111.6 (4.70x)|116.6 (4.70x)|
|4096|111.7 (4.92x)|112.4 (4.91x)|107.5 (5.00x)|126.0 (5.49x)|
|8192|111.8 (4.87x)|115.3 (4.69x)|119.6 (4.47x)|192.4 (5.03x)|
|16384|112.9 (4.79x)|116.3 (4.83x)|183.0 (5.31x)|325.2 (6.40x)|
|32768|126.2 (4.33x)|184.6 (5.49x)|320.9 (6.48x)|522.1 (7.98x)|

## D=256

|T\H|2|4|8|16|
|---:|---:|---:|---:|---:|
|1|115.6 (4.70x)|102.2 (5.41x)|98.4 (5.24x)|110.6 (4.93x)|
|64|106.7 (5.07x)|108.7 (4.99x)|106.8 (4.83x)|123.5 (4.42x)|
|256|113.8 (4.74x)|111.2 (4.98x)|108.3 (4.81x)|109.3 (5.01x)|
|1024|110.8 (4.81x)|136.0 (4.07x)|110.3 (4.82x)|118.8 (4.61x)|
|4096|122.0 (4.58x)|111.4 (5.13x)|117.7 (4.51x)|184.8 (5.29x)|
|8192|108.9 (5.09x)|120.0 (4.73x)|181.1 (5.28x)|316.3 (6.55x)|
|16384|112.2 (4.88x)|188.2 (5.19x)|313.7 (6.89x)|513.8 (8.08x)|
|32768|187.1 (5.17x)|320.1 (6.78x)|516.6 (8.04x)|907.0 (8.93x)|

## Applicability Boundary

- Below roughly 8 million elements per tensor, time is launch/cache-floor bound (about 96-136us); T/H perturbations are mostly noise.
- At and above 8 million elements, time scales with `x/g/output` traffic. D256/H16/T32768 moves about 805MB in 907.0us, or 0.89TB/s effective, marking the memory-bandwidth boundary.
- D256 with a 64-row monolithic tile fails Triton Ascend `PlanMemory`; the bandwidth branch remains 32 rows. A row8-tiled nw1 variant also regressed this extreme to 1298.0us and is rejected.
