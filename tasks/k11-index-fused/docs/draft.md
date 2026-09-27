# K11 index-fused draft

## Research

- Knowledge router query for the KDA token-index gather/scatter chain returned no hits.
  Blind spot: no task-local skill covers this exact fused gather+concat+scatter pattern.
- Reused the K10-verified NPU Triton shape cache and CompiledKernel direct-launch pattern.

## Baseline

- `c000` full bench: w01 373.3us, w02 318.4us, w03 697.4us.
- Keep gate: w02 < 222.9us, w01/w03 <= baseline.

## c001

- One Triton kernel: destination row selects `idx_spec[row]`, `idx_ns[row-k]`, or zero.
- Preserve arbitrary index semantics; do not assume scheduler indices are monotonic.
- Specialize D=4096 and direct-launch the compiled kernel after a probe-validated first call.

## Campaign 2 research

- Router query `NPU Triton fused gather scatter tail wave occupancy BLOCK dimension specialization`
  returned no hits (`python` fell back to Python 2 syntax; retried unchanged query with `python3`).
- Blind spot remains: no task-local evidence for optimal BLOCK_D or multi-row BLOCK_T on dav_2201;
  must measure the expanded axis sweep.

## Campaign 2 measurements

- Expanded workload axes to 15 total: originals plus T={1024,8192},
  D={2048,8192}, and k:m={1:1,1:3,7:1} in full cross-product.
- c001 original on the expanded full bench fails at the first D=2048 row with
  `ValueError: index-fused candidate requires D=4096`; its correctness/perf applicability is D=4096 only.
- c002 (2D row-vector tile) and c003 (static row loop) passed correctness but benched w02 at
  241.4/221.1us. Root cause was not the tile alone: compiled direct-launch repeatedly passed
  constexpr arguments, raising `function takes exactly ... arguments`, so both fell back to JIT launch.
- c004 fixed direct-launch runtime args and restored w02 to 120.5us. D=2048 rows were 129.0-129.5us;
  D=8192 rows were 322.5-363.9us in the dev sweep.
- Tail sweep on w03 (20 samples): c001 mean/p50/p99=349.6/346.3/391.5us;
  BLOCK_T=2 =347.5/351.5/365.4us; BLOCK_T=4 =341.7/340.0/361.2us. BT8 did not obtain a clean
  sample because concurrent NPU containers repeatedly hit ACL Resource_Busy; no claim either way.
- c005 full 15-shape bench passed verify 15/15 and recorded mean/p50/p99 by workload, but w01
  regressed versus a same-lock c001 originals run (128.4us vs 100.0us mean).
- c008 therefore restores the literal c001 one-row/one-column kernel for D=4096,T<8192 and uses:
  generic row kernel + BLOCK_D=min(4096,next_pow2(D)) for new dimensions; BLOCK_T=2 at T>=8192;
  BLOCK_T=4 at T>=16384 when D<=4096. D=8192 uses two 4096-wide column programs.
- D=8192 single BLOCK_D=8192 remains a declared blind spot: no clean NPU sample due concurrent ACL
  conflicts. It is not used by c008.
