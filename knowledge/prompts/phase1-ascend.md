# Phase 1 任务书模板：研究并做对（昇腾版）

> 结构对应比赛仓库 phase1 prompt；填充时保留所有章节标题，只填内容。
> 组合条款（反作弊/盲区/双代际等）从 `clauses/` 按需附加到文末。

## Kernel Information

- 算子：`<名称>`，算子族 `<op_family>`
- 硬件代际：`<DAV_2201 | DAV_3510>`（以 get_npu_arch.py 检测结果为准）
- 参考语义：`<数学定义或官方文档引用>`
- workload 集：全量 `<N>` 个（bench/ 定义），开发代表集 `<n>` 个（标 `repr: true` 的行）
- 变量轴与常量轴：`<如 batch×seq 可变，hidden=4096 恒定>`

## 参考实现说明

基线 = `<baseline 路径/调用方式>`，其验证输出以 `kda verify` 为准。
容差 = 契约③引用的官方 dtype 分级表——**禁止自定更宽松或更严格**。

## Phase 1 目标

研究现有实现并产出**第一个正确的**昇腾 kernel。性能重要，但正确性与干净的基线设计优先（竞赛 prompt 原则）。

工作流（顺序强制）：
1. 经知识路由检索同类算子参考（`python knowledge/router/query.py --op-family <X> --arch <Y>`）；
2. 对每个被注入的 skill 切片提交承认记录（valuable_aspects + kernel_application）；
3. 写 `docs/draft.md`（六要素结构，见 contract-template.md 附）；
4. 生成 `docs/plan.md`（AC 验收对：每条 AC 含 Positive + Negative 测试；至少一条 Negative 针对本算子的已知数值陷阱，如全 0 输入）；
5. 计划锁定（`kda contract --lock`）后方可编辑代码；
6. 实现首候选 → `kda verify --workload-set l0` → `kda bench --mode l0` → 代表集 l1；
7. 首候选必须让全量集正确性通过才算 Phase 1 完成。

## 逐轮产出义务（每个工作轮次，硬校验⑤-⑧的来源）

1. **开工**：从 `knowledge/prompts/templates/round-contract.md` 复制填写 `run/round-<N>-contract.md`（主线目标/目标 AC/direction 与 budget/成功标准）；
2. **收工**：从 `templates/round-summary.md` 复制填写 `run/round-<N>-summary.md`——必须含证据指针、Todo 显式清点、BitLesson Delta 段（add/update/none）；
3. **送审**：收工 = 写完 summary 后**主动调用 `kda gate --round <N>`**。Stop-hook 自动触发是糖；宿主不支持 hook 或 hook 失效时，这条命令是唯一兜底——不调 gate 的收工不算收工，产出不被计入证据链；
4. 评审打回（REVISE/REJECT）→ 按反馈修改后进入下一轮（round+1）；`STOP`/熔断 → 停止并等人工。

## 留痕要求（竞赛五段式之留痕段）

- 每个性能相关候选：benchmark.csv 一行（harness 自动）+ solutions.jsonl DAG 条目；
- 每个被否决方向：同样留痕（direction + 否决证据）；
- 遇知识盲区：按盲区条款显式声明，不许编造参考。

## 附加条款

`<从 clauses/ 组合：anti-reward-hacking.md 必选；blindspot-declaration.md 必选；dual-arch-branching.md 双代际任务必选>`
