# K10 verdict: c600 dimension-robust dual-kernel dispatch

## Conclusion

- **Final keep: `solution/c600/candidate.py`**
- Correctness has no applicability gate in the requested dimension sweep: **96/96 PASS, err_ratio=0**.
- Original three shapes improve versus c001: w01 `96.2us` (-7.7%), w02 `112.4us` (-10.7%), w03 `116.3us` (-9.6%).
- Full 96-combination matrix: mean `153.4us`, p50 `113.5us`, p99 `924.2us`, mean speedup `5.18x`.
- NPU 7 had no foreign process at the post-run check; every reference row in the result JSON is stable.

## Implementation

- One fused Triton kernel reads `x/g`, performs fp32 RMS reduction, affine, sigmoid gate, and bf16 writeback; no rstd temporary or upstream heuristics.
- D128 small/medium shapes use row8 x N static tiles with `num_warps=1`; w03 and bandwidth shapes use the single-tile row32/64 variant. D64/D256 use element-count thresholds.
- First call probes CompiledKernel direct launch against JIT output. Hot paths use a partially pre-bound launch and a four-output ring.
- Launch state plus outputs are one LRU record per shape, capped at two shapes. This preserves hot-shape reuse without retaining all 96 outputs.
- D outside {64,128,256}, including unseen model dims, falls back to the fp32 torch expression; non-contiguous x/g/weight/bias are made contiguous.

## Dimension applicability

- Full matrix and dispatch table: `docs/dimension-matrix.md`.
- Launch/cache floor: below about 8M elements per tensor, roughly 96-136us.
- Bandwidth boundary: at/above 8M elements. The largest D256/H16/T32768 point moves about 805MB in 907us (0.89TB/s effective).
- D256 monolithic 64 rows fails Triton PlanMemory; D256 stays at 32 rows. Applying row8/nw1 tiles there regresses to 1298us and is rejected.

## Current Canonical Output

### verify (96 dimensions)

```text
solution:   c600
mode:       single-step (dev=False, n=96)
  w01: PASS  steps=None state_inputs=None err_ratio=0.0
  w02: PASS  steps=None state_inputs=None err_ratio=0.0
  w03: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h02_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h04_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h08_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d064_h16_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h02_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h04_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h04_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h04_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h04_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h04_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h08_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d128_h16_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h02_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h04_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h08_t32768: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t00001: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t00064: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t00256: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t01024: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t04096: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t08192: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t16384: PASS  steps=None state_inputs=None err_ratio=0.0
  d256_h16_t32768: PASS  steps=None state_inputs=None err_ratio=0.0

passed: 96/96
```

### bench (96 dimensions, recorded)

```text
solution: c600  set: full  n: 96
  w01: mean=96.2us p50=106.3 p99=116.7 ref=534.8639965057373
  w02: mean=112.4us p50=112.7 p99=141.6 ref=552.0879864692688
  w03: mean=116.3us p50=120.0 p99=129.0 ref=562.16801404953
  d064_h02_t00001: mean=113.2us p50=112.8 p99=131.7 ref=539.8200035095215
  d064_h02_t00064: mean=101.7us p50=104.6 p99=138.7 ref=538.53600025177
  d064_h02_t00256: mean=111.8us p50=108.9 p99=118.1 ref=532.972002029419
  d064_h02_t01024: mean=111.3us p50=109.5 p99=118.7 ref=534.5879912376404
  d064_h02_t04096: mean=111.2us p50=105.0 p99=126.7 ref=531.2359929084778
  d064_h02_t08192: mean=115.8us p50=115.6 p99=134.9 ref=548.0639934539795
  d064_h02_t16384: mean=118.4us p50=118.6 p99=154.1 ref=525.4679918289185
  d064_h02_t32768: mean=107.3us p50=103.6 p99=122.7 ref=1242.3120021820068
  d064_h04_t00001: mean=109.2us p50=118.6 p99=123.0 ref=556.0879945755005
  d064_h04_t00064: mean=112.9us p50=111.8 p99=129.0 ref=543.3359980583191
  d064_h04_t00256: mean=114.3us p50=113.8 p99=127.5 ref=544.6120023727417
  d064_h04_t01024: mean=113.4us p50=108.3 p99=133.7 ref=529.4560074806213
  d064_h04_t04096: mean=114.6us p50=112.8 p99=127.3 ref=536.1559987068176
  d064_h04_t08192: mean=111.5us p50=110.6 p99=120.4 ref=548.1839895248413
  d064_h04_t16384: mean=108.8us p50=110.0 p99=115.3 ref=545.8680033683777
  d064_h04_t32768: mean=121.2us p50=125.7 p99=146.1 ref=528.9080023765564
  d064_h08_t00001: mean=101.0us p50=105.0 p99=114.9 ref=529.8199892044067
  d064_h08_t00064: mean=111.0us p50=110.2 p99=118.9 ref=528.8679957389832
  d064_h08_t00256: mean=112.1us p50=108.9 p99=125.3 ref=536.8920087814331
  d064_h08_t01024: mean=117.6us p50=115.6 p99=130.1 ref=534.4600081443787
  d064_h08_t04096: mean=114.3us p50=113.4 p99=125.6 ref=550.2879977226257
  d064_h08_t08192: mean=115.3us p50=114.2 p99=129.9 ref=561.1119985580444
  d064_h08_t16384: mean=120.5us p50=121.7 p99=123.2 ref=540.3119921684265
  d064_h08_t32768: mean=188.7us p50=191.2 p99=196.6 ref=1015.5159950256348
  d064_h16_t00001: mean=109.9us p50=115.6 p99=124.0 ref=529.3320178985596
  d064_h16_t00064: mean=108.0us p50=107.8 p99=112.3 ref=526.5119910240173
  d064_h16_t00256: mean=111.1us p50=109.3 p99=120.0 ref=529.9279928207397
  d064_h16_t01024: mean=114.0us p50=115.1 p99=118.2 ref=550.928008556366
  d064_h16_t04096: mean=111.2us p50=108.1 p99=123.4 ref=539.028000831604
  d064_h16_t08192: mean=119.9us p50=119.7 p99=133.0 ref=538.8520002365112
  d064_h16_t16384: mean=189.6us p50=192.0 p99=207.1 ref=998.8719940185547
  d064_h16_t32768: mean=320.6us p50=322.3 p99=326.3 ref=2064.6880626678467
  d128_h02_t00001: mean=109.3us p50=116.6 p99=137.1 ref=519.4319903850555
  d128_h02_t00064: mean=108.8us p50=108.3 p99=119.1 ref=531.3439965248108
  d128_h02_t00256: mean=106.8us p50=104.8 p99=116.0 ref=528.1719923019409
  d128_h02_t01024: mean=107.3us p50=105.2 p99=116.0 ref=526.0600090026855
  d128_h02_t04096: mean=111.7us p50=113.5 p99=116.6 ref=550.1439929008484
  d128_h02_t08192: mean=111.8us p50=108.2 p99=127.0 ref=543.9839959144592
  d128_h02_t16384: mean=112.9us p50=108.2 p99=124.8 ref=540.7959938049316
  d128_h02_t32768: mean=126.2us p50=128.3 p99=138.1 ref=545.8480000495911
  d128_h04_t00001: mean=110.0us p50=110.5 p99=113.1 ref=533.296000957489
  d128_h04_t00256: mean=113.6us p50=112.4 p99=128.0 ref=537.3600006103516
  d128_h04_t01024: mean=118.0us p50=116.3 p99=134.2 ref=535.0079894065857
  d128_h04_t08192: mean=115.3us p50=118.7 p99=122.8 ref=541.156005859375
  d128_h04_t32768: mean=184.6us p50=191.0 p99=208.6 ref=1013.4239912033081
  d128_h08_t00001: mean=111.1us p50=105.1 p99=143.7 ref=503.53601574897766
  d128_h08_t00064: mean=108.0us p50=106.7 p99=113.2 ref=506.08800053596497
  d128_h08_t00256: mean=108.1us p50=106.4 p99=119.1 ref=514.599996805191
  d128_h08_t01024: mean=111.6us p50=107.8 p99=119.0 ref=525.1760125160217
  d128_h08_t04096: mean=107.5us p50=106.8 p99=117.2 ref=537.0879888534546
  d128_h08_t08192: mean=119.6us p50=121.2 p99=126.3 ref=535.3119969367981
  d128_h08_t16384: mean=183.0us p50=188.4 p99=189.9 ref=972.324001789093
  d128_h08_t32768: mean=320.9us p50=322.2 p99=328.3 ref=2079.859972000122
  d128_h16_t00001: mean=118.8us p50=112.6 p99=148.9 ref=550.8919954299927
  d128_h16_t00064: mean=112.8us p50=114.3 p99=118.0 ref=547.1920013427734
  d128_h16_t00256: mean=111.2us p50=110.1 p99=121.8 ref=553.6359906196594
  d128_h16_t01024: mean=116.6us p50=112.2 p99=137.8 ref=548.2840061187744
  d128_h16_t04096: mean=126.0us p50=129.5 p99=133.4 ref=692.0759916305542
  d128_h16_t08192: mean=192.4us p50=195.4 p99=212.1 ref=967.8319931030273
  d128_h16_t16384: mean=325.2us p50=327.6 p99=330.8 ref=2081.160020828247
  d128_h16_t32768: mean=522.1us p50=528.2 p99=538.7 ref=4166.775989532471
  d256_h02_t00001: mean=115.6us p50=115.3 p99=158.5 ref=543.2119965553284
  d256_h02_t00064: mean=106.7us p50=116.3 p99=123.1 ref=540.7159924507141
  d256_h02_t00256: mean=113.8us p50=111.7 p99=126.0 ref=539.683985710144
  d256_h02_t01024: mean=110.8us p50=111.9 p99=113.7 ref=533.5280060768127
  d256_h02_t04096: mean=122.0us p50=121.1 p99=136.5 ref=558.0839991569519
  d256_h02_t08192: mean=108.9us p50=111.6 p99=117.4 ref=553.8159966468811
  d256_h02_t16384: mean=112.2us p50=120.7 p99=130.2 ref=547.1920013427734
  d256_h02_t32768: mean=187.1us p50=189.1 p99=203.2 ref=967.6279902458191
  d256_h04_t00001: mean=102.2us p50=109.3 p99=113.7 ref=553.1399965286255
  d256_h04_t00064: mean=108.7us p50=106.2 p99=116.1 ref=541.7199969291687
  d256_h04_t00256: mean=111.2us p50=110.6 p99=119.1 ref=554.1439890861511
  d256_h04_t01024: mean=136.0us p50=119.7 p99=204.7 ref=553.9600133895874
  d256_h04_t04096: mean=111.4us p50=108.0 p99=120.1 ref=571.5039968490601
  d256_h04_t08192: mean=120.0us p50=121.9 p99=128.8 ref=567.524003982544
  d256_h04_t16384: mean=188.2us p50=188.2 p99=190.5 ref=976.312005519867
  d256_h04_t32768: mean=320.1us p50=320.3 p99=331.3 ref=2168.7439918518066
  d256_h08_t00001: mean=98.4us p50=105.1 p99=110.2 ref=515.776002407074
  d256_h08_t00064: mean=106.8us p50=107.2 p99=112.2 ref=516.4880037307739
  d256_h08_t00256: mean=108.3us p50=106.3 p99=118.8 ref=521.2719917297363
  d256_h08_t01024: mean=110.3us p50=112.2 p99=123.1 ref=531.9239974021912
  d256_h08_t04096: mean=117.7us p50=122.0 p99=125.9 ref=530.783998966217
  d256_h08_t08192: mean=181.1us p50=185.8 p99=197.4 ref=956.7000031471252
  d256_h08_t16384: mean=313.7us p50=313.6 p99=325.0 ref=2161.8079662323
  d256_h08_t32768: mean=516.6us p50=521.4 p99=526.9 ref=4154.172039031982
  d256_h16_t00001: mean=110.6us p50=109.1 p99=117.4 ref=544.7200059890747
  d256_h16_t00064: mean=123.5us p50=117.6 p99=137.9 ref=546.0679888725281
  d256_h16_t00256: mean=109.3us p50=107.9 p99=114.1 ref=547.8880047798157
  d256_h16_t01024: mean=118.8us p50=122.2 p99=132.3 ref=547.3559975624084
  d256_h16_t04096: mean=184.8us p50=187.2 p99=188.0 ref=977.4680137634277
  d256_h16_t08192: mean=316.3us p50=319.3 p99=323.2 ref=2069.9399948120117
  d256_h16_t16384: mean=513.8us p50=519.1 p99=529.9 ref=4150.060081481934
  d256_h16_t32768: mean=907.0us p50=912.3 p99=924.2 ref=8102.876091003418

mean: 153.4us  p50: 113.5  p99: 924.2  speedup_vs_ref: 5.18x
recorded -> /data01/mahaolong/KAgent/tasks/k10-rmsnorm-gated/docs/benchmark.csv
```
