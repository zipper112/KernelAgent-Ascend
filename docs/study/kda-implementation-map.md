# GLM-5.3-Flash KDA 实现总纲（阅读地图）

> 源码锚点：vllm-ascend @ bea70ab8f（= 镜像 f505184be15b 血统；本地稀疏检出
> `D:\PyProject\Jev\.k8-ref\va\`）。KDA = Kimi Delta Attention（月之暗面 Kimi Linear
> 的门控 DeltaRule 线性注意力），GLM-5.3-Flash 把它作为混合架构的线性注意力层。

## 一、KDA 在模型里的位置（先建立全局感）

GLM-5.3-Flash 是**混合架构**（`model.py` IsHybrid）：每个 decoder layer 二选一——

- `layer_types[i] == "linear_attention"` → **KDA 层**（`Glm5NextLinearAttention`，档案口径 35 层）
- 其余 → **MLA 稀疏注意力层**（`Glm5NextMLAAttention`，attention.py）

判定入口：`config.py:230 is_kda_layer(layer_idx)`（读 checkpoint 的 per-layer `layer_types`）。
装配处：`model.py:267 Glm5NextDecoderLayer.__init__`（296 行选 KDA / 306 行选 MLA）。
MLP 侧：前 `first_k_dense_replace` 层 dense，其余 MoE（`mlp_layer_types`，`model.py:141/334`）。
另有一个 **MTP draft 层**（`mtp.py`，挂在 base 层之后，spc 解码用）。

所以"KDA 的实现"= 模型层里那一半层的注意力替换件 + 它专属的状态池（mamba 式）。

## 二、代码地图（四层，自上而下）

```
L1 模型层  vllm_ascend/models/glm5next/          （13 文件，~5.3k 行，全部读完约 1 天）
L2 算子编排层
   ├ vllm_ascend/models/glm5next/ops/            GLM 侧门控契约 + 状态装配（Python/torch）
   └ vllm_ascend/ops/kda.py                      共享 AscendC 调用薄封装（89 行）
L3 Triton-Ascend 层  vllm_ascend/ops/triton/     output_writeback / conv_state / fused_gdn_gating 等
L4 闭源算子层  torch.ops._C_ascend.*             chunk_kda_fwd / recurrent_kda / causal_conv1d
                 （镜像内 AscendC .so，msprof 显形为 aclnnChunkKdaFwd 等，无源码）
参照系：vllm.model_executor.layers.mamba.gdn.base.GatedDeltaNetAttention（L1 直接继承它）
        vllm.third_party.flash_linear_attention（fla——KDA 算法的上游参考实现）
```

### L1 逐文件一句话（glm5next/）

| 文件 | 行数 | 职责 |
|---|---|---|
| `model.py` | 1071 | DecoderLayer 装配（KDA/MLA 二选一 + MoE）、Glm5NextModel 主干、ForCausalLM |
| `kda.py` | 530 | **KDA 层本体**（Glm5NextLinearAttention）：投影→分组→卷积→核心算子→写回 |
| `attention.py` | 327 | MLA 稀疏注意力（对照层，非本篇重点） |
| `mtp.py` | 410 | MTP draft 层（speculative 解码） |
| `cache_config.py` | 455 | 状态池布局与配额（conv+recurrent 两池怎么分内存） |
| `cache_views.py` | 142 | 运行期状态池视图 |
| `kv_cache.py` | 278 | MLA 侧 KV cache |
| `config.py` | 390 | 层型/超参（is_kda_layer、mlp_layer_types） |
| `processor.py` / `multimodal.py` | 889/707 | 多模态前处理（与 KDA 无关可后看） |
| `sparse_attn_indexer_kpool.py` | 147 | MLA 侧索引池 |
| `ops/` | 见 L2 | 模型本地算子编排 |

### L2 关键三件（glm5next/ops/）

- **`ops/kda.py`** — "GLM bounded-gate contracts"：`recurrent_kda()`（decode 路径，含
  MTP 拒收回滚 `num_accepted_tokens`；`KDA_MAX_RECURRENT_TOKENS = 8`——**≤8 tok 走递推，
  ≥9 tok 走分块**的分界）与 chunk 调用编排；
- **`ops/state_ops.py`** — `gather_initial_states / scatter_states`：状态池按 slot 索引
  装配/写回（**纯 torch index_select**——CUDA 版走 Triton gdc_wait 在昇腾不可用，这正是
  K9 scatter/gather 群的一部分）；fresh 请求用 `torch.where(keep, out, 0)` 屏蔽脏值
  （避免 NaN×0）且**全程设备侧无 host 同步**（ACL-graph 安全）；
- **`ops/causal_conv1d.py`** — 短卷积状态适配（aclnnCausalConv1d 的非连续态回写处理）；
  PyTorch 回退版在 `vllm_ascend/ops/causal_conv1d.py`（**K8 病灶原位**）。

### L4 闭源算子的调用签名（`vllm_ascend/ops/kda.py`，89 行，值得整读）

- `run_chunk_kda(...)` → `torch.ops._C_ascend.chunk_kda_fwd`：prefill 分块算法
  （`KDA_CHUNK_SIZE=64`，BSND 布局，`state_v_first=True`，输出 final_state）；
- `run_recurrent_kda(...)` → `torch.ops._C_ascend.recurrent_kda`：递推算法
  （token 级 stride 直入、qk l2norm/门控/beta-sigmoid 都有 kernel 内开关）。

## 三、总体流程（一次 KDA 层 forward 的五步）

入口：`kda.py:269 forward → :317 _forward`（拿到 attn_metadata 后的完整路径）：

```
hidden_states
  ① 投影：in_proj_qkvbfg_a（qkv+beta+gate+a 一次融合投影）、g_b_proj（门控 g 两段）
  ② spec/non-spec 分组（MTP 激活时）：6×index_select 按 spec_token_indx /
     non_spec_token_indx 把 token 流拆两路 —— 【K9 病灶：8 连发 scatter/gather】
  ③ 因果短卷积 causal_conv1d（width=4）：两路各自跑，conv_state 滑窗更新
     —— 【K8 病灶：PyTorch 回退逐序列 .item() 同步，我们已修（vec patch -34% TPOT）】
  ④ 核心注意力学（殊途同归到 L4）：
     prefill / 长段 → run_chunk_kda（64-token 分块：块内 UT 形变换 + 跨块状态递传）
     decode / MTP   → run_recurrent_kda（逐 token 递推 + 拒收回滚）
  ⑤ 写回与归一：index_copy_ 拼回输出流（K9 另一半）+ 输出 norm（fused_eh_norm）
```

### 数学核心（一段话，读懂递推就读懂 KDA）

每头维护状态 `S ∈ [Dk, Dv]`。每 token：门控 `g = σ(a_log + dt_bias · w)`（有下界
`lower_bound` 的 bounded gate）决定**遗忘率**；`β = σ(beta_proj)`（有界 0~1）决定**写入
强度**；delta rule 更新 `S ← S·(1−βk kᵀ衰减) + β v kᵀ`（"先删旧再写新"的增量规则，
l2norm 后的 q/k 保证数值稳定）；输出 `o = q·S`。分块算法 = 把这段递推在 64-token 块内
用矩阵形式一次算完（chunk 内二次型 + 块间状态传递），是同一数学的两种调度。

### 状态池（KDA 特有的缓存世界观）

与 MLA 的 KV cache 不同，KDA 层存的是**每序列每层的两个小状态**：
`conv_state [cache, state_len, dim]`（卷积滑窗，width−1=3）+
`recurrent_state [cache, H, Dk, Dv]`（递推状态）。`cache_config.py:33 _Glm5NextCacheLayout`
负责池布局/配额，MTP 的 conv_state 宽度要含 num_spec（`kda.py:115 get_state_shape`）。

## 四、K 档病灶 ↔ 代码位置速查

| K 档 | 病灶 | 代码锚点 |
|---|---|---|
| K8 | conv1d_update PyTorch 回退逐序列同步 | `vllm_ascend/ops/causal_conv1d.py:159`（已修 vec 版） |
| K9 | spec 分组 8 连发 scatter/gather | `models/glm5next/kda.py:404-410, :485, :528` |
| K1 | eager 小核串（无 @support_torch_compile） | `model.py` 全模型层 + `kda.py:7 eager_break_during_capture` |
| K4 | 核心算子高效非瓶颈（防误判档） | L4 闭源算子本体（aclnnChunkKdaFwd 1.2% busy） |

## 五、建议阅读序（半天到一天）

1. `vllm_ascend/ops/kda.py`（89 行，全读）——两张闭源算子的调用卡；
2. `models/glm5next/kda.py`（530 行，主读 `_forward`）——五步流程的现场；
3. `models/glm5next/ops/state_ops.py` + `ops/causal_conv1d.py`——状态池与卷积适配；
4. `model.py` 的 DecoderLayer 段（267-420）——KDA/MLA/MoE 怎么拼起来；
5. 对照上游：fla 库的 gated delta rule 实现（算法原型）。

实验路径：e15 复刻环境（/data02/kda，镜像 f505184be15b + worktree dbb3b04）可单卡
微基准任何一个环节（我们的 p1-block/k9/k3/k7 微基准脚本都在 tasks/*/scripts/）。
