# Round 0 Contract — K1 参数刀三臂

- 主线目标：验证 K8vec 之上，参数刀能否解锁 p16 被淹没的吞吐与 TTFT 尾部
- 目标 AC：p16 r1 tok_s 增益 > +5%（噪声门）；TTFT p99 降幅 > 10%；p1 TPOT 回归 < 5%
- 车道：mainline
- 本轮方向：
  direction: param-knives
  hypothesis: p16 瓶颈=TP8 eager 步调漂移的集合等齐（K1 机理链）；enable_reduce_sample
    削采样段小包、multistream_overlap_shared_expert 填 vector 空泡 → 缩小 rank 步差
  knowledge_refs: [K1-aivkernel-tp-sync, K2-hcom-allreduce-straggler, E060, E061]
  budget: 3 臂 ×（重启 ~15min + 测 ~10min）
  exit_criteria: 三臂稳态数据齐 + 按上述 AC 判定 keep/reject 入证据链
- 阻塞/排队事项：PPL proxy 解析（遗留）
- skill-acknowledgment: none: 本轮无新注入 skill
- 成功标准: 三臂数据可比（同镜像/同形态/同种子/同预热流程），判定结论进 benchmark.csv + solutions.jsonl
- 实验臂定义：
  c001 = baseline（vec，GLM_EXTRA_CFG 空）
  c002 = +enable_reduce_sample
  c003 = +enable_reduce_sample +multistream_overlap_shared_expert
