# E065-K8 serving 级 A/B 复刻结果（e15/npu-8，2026-09-26）

## 环境（血统校验）

- 镜像：`quay.io/ascend/vllm-ascend:nightly-main` @ **f505184be15b**（09-24 构建 = main bea70ab8f，与 npu-7 E055/E065 同源；老 main a3e55249 / omni 孤儿线 / glm-5.3-flash 老 baseline 均排除）
- worktree：`/data02/kda/vllm-ascend-main` @ dbb3b04（E065 vec patch on bea70ab8f，git 状态干净）
- 形态：E065 完整复刻（TP8+EP、MTP k=3、FULL_DECODE_ONLY graph、bt8192/seqs32、prefix caching on）
- 病灶指纹确认：orig 臂起服日志出现 K8 档原文 WARNING（`causal_conv1d_update_npu` 缺失 → PyTorch fallback → `syncs per request and therefore stalls ACL graph capture at decode-FULL`）
- 双臂唯一差异：`GLMFLASH_K8_VEC=1|0`（patch 内 env 路由）
- 压测：预热轮丢弃后稳态；p1=512 prompt+384 tok；p16=16 并发 × 4k 随机 token（全 miss）× 128 tok；固定种子

## 结果（稳态口径：p1 取第 2 次、p16 取 r1 全 miss）

| 指标 | orig（fallback） | vec（patched） | Δ |
|---|---|---|---|
| p1 TPOT | 24.6 ms/tok | **16.2 ms/tok** | **-34.1%** |
| p1 首轮（graph 换档敏感） | 184.0 ms | 14.8 ms | 同步消除后换档坑消失 |
| p16 聚合吞吐（全 miss r1） | 50.9 tok/s | 51.7 tok/s | +1.6%（噪声内持平） |
| p16 r2（部分缓存命中） | 125.1 tok/s | 92.5 tok/s | 波动大，不作为判据 |

## 判定

1. **K8 假设验证成立**：每请求 host 同步消除对单流 decode 是主要收益（TPOT -34%，远超 E065 PLAN 预期的 -5%+；超额部分来自 graph 换档坑一并消除）；
2. **p16 持平符合 K1/K2 归因**：16 并发瓶颈在 TP straggler 等齐（通信同步），K8 的收益被淹没——**p16 段收益需 K1 治理（参数刀/上游 torch.compile）解锁**，这正是下一主攻方向；
3. PPL 锚点：e15 无 npu-7 的长上下文 PPL 数据集，proxy-PPL 解析两次尝试均未取到有效值（vLLM 0.30 prompt_logprobs 顶层结构与文档不符）——**正确性由 verify 微基准四步协议背书（err_ratio=0），serving 级 PPL 锚点列为遗留项**，不作为本次判据。

## 复现

- 起 orig/vec：`bash /data02/kda/e15-start.sh orig|vec`（端口 8001）
- 稳态序列：`python3 /data02/kda/e15-seq.py <tag>`
- 对照：`python3 /data02/kda/cmp.py`
