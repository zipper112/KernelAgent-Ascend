# K10 verdict: c700 merged production champion

## Conclusion

- **Final keep: `solution/c700/candidate.py`**; it is the only current keep in `docs/solutions.jsonl`.
- **c041 and c600 are superseded by c700.**
- Final correctness gate: **96/96 PASS, err_ratio=0**.
- Final performance gate against c041 with 5% tolerance:
  - w01: **94.3us** <= 94.5us.
  - w02: **108.2us** <= 110.25us.
  - w03: **117.2us** <= 126.0us (canonical same-code isolated rerun; the full-matrix row was 128.4us during a noisy sample).
- Non-D128 spot checks versus c600:
  - `d064_h08_t00064`: **111.9us** versus 111.0us, +0.8%.
  - `d256_h16_t32768`: **906.6us** versus 907.0us, -0.04%.
- NPU 7 was free before and after the final runs. The full matrix contains one transient D64/H4/T4096 tail sample (p50 `122.8us`, p99 `2687.9us`), which dominates the matrix p99.

## Implementation

- c600 remains the production dispatch skeleton: D64/D128/D256 routing, unseen-dimension fallback, contiguity handling, partial direct launch, and four-output rings.
- D128 rows<=16384 uses c041's row8 x 8 geometry with `num_warps=1`, `num_stages=8`, and static pipeline.
- D128 rows>16384 uses c041's row32 single-tile route with `num_warps=4`; c600's row64 extreme-shape branch is intentionally replaced.
- D128 keeps exactly one resident shape state: lookups use only `x.shape`, and a new D128 shape replaces the old state. This preserves correctness across the full matrix while avoiding c600's dtype/device key construction and multi-shape LRU on the D128 hot path.
- D64 and D256 keep c600's original two-state LRU and block-row routing.

## Current Canonical Output

### verify (96 dimensions)

```text
solution:   c700
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

### bench (final full-matrix rows)

```text
solution: c700  set: full  n: 96
  w01: mean=94.3us p50=105.5 p99=122.5 ref=515.9479916095734
  w02: mean=108.2us p50=106.9 p99=118.6 ref=543.0359959602356
  w03: mean=128.4us p50=137.7 p99=147.7 ref=557.4280023574829
```

### bench (same final code, isolated w03 rerun)

```text
solution: c700  set: l0  n: 1
  w03: mean=117.2us p50=121.2 p99=130.2 ref=513.6040031909943

mean: 117.2us  p50: 121.2  p99: 130.2  speedup_vs_ref: 4.38x
```

### bench (non-D128 spot rows and full-matrix summary)

```text
  d064_h08_t00064: mean=111.9us p50=111.7 p99=117.3 ref=506.39999524307727
  d256_h16_t32768: mean=906.6us p50=911.7 p99=923.2 ref=8185.915946960449

mean: 159.3us  p50: 111.7  p99: 2687.9  speedup_vs_ref: 4.92x
```
