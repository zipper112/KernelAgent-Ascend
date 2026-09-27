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
