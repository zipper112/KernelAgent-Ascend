# K10 draft

- [2026-09-27 blindspot] router symptom=`rms_norm_gated contiguous launch overhead small kernel` 0 hit；按 AGENTS 记录。后续用 K8 实测谱系与本机 Triton introspection 补知识。
- [2026-09-27 c001-direct-row] 假设：D=128 每行 128 lanes 单程序、静态 D 无 mask，热路径跳过 `.contiguous()`/reshape/rstd 临时与上游 heuristics，并缓存底层 launch；预期 w02 显著下降，w01/w03 不退化。
