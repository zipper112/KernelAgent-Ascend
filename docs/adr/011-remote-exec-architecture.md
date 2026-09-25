# ADR-011: 远程执行架构——本地为家、远端为镜 + 无网 NPU 机供给

- 状态：已接受
- 日期：2026-09-25

## 背景

用户指出核心隐患：远程模式下知识库在本地、NPU 计算在远端，实验目录放哪？网络现实（实测 2026-09-25）：e15 无外网且**反向到 jump 不通**（REVERSE-FAIL，策略 A/B 排除）；jump 有网但仅有 python3.7（pip 处理不了新 wheel 元数据）；aarch64 + Python 3.10；8×910B4 全空闲；共用机器（不可污染系统 python）。

## 决策

### 1. 执行架构：本地为家、远端为镜

- **单一事实源在本地** `tasks/<task>/`：证据链（csv/jsonl/audit）、git（keep 即 commit）、plan lock、solutions DAG——**只此一份**；远端镜像随时可拆可重建；
- **远端镜像** `~/kda-ascend/<task>/{payload/, results/}`：payload = 候选代码 + workload 定义 + runner 脚本 + job.json；results = 产出 json。**远端无 git、无知识库、无密钥**——杜绝双仓分裂脑与敏感面扩大；
- **知识库/GLM 密钥永不离开本地**：agent 在本地读知识、写代码、调模型；远端只做确定性计算（verify/bench/profile）。

### 2. Job 生命周期协议（interaction-protocol §8b）

`组装 job.json → tar-over-ssh push → 远端 runner → pull result → 本地 evidence 写链 → git commit`。
- 同步走 **tar-over-ssh 两跳**（Windows 无 rsync；job spec 显式文件清单=天然增量）；
- runner v0 薄实现（verify 容差四步协议 / bench warmup+L2 清除+交错采样）；Phase 1 verify 后端换 vendored KernelVerifier（third_party/akg 一次性推远端）；
- 长任务 Phase 1 换 nohup+status 轮询（防两跳断连丢作业）。

### 3. e15 供给（策略 C：jump 中转下载）

`tools/provision_npu.py`：本地 PyPI JSON API 解析依赖闭包（torch 2.13.0 + torch_npu 2.13.0rc1，cp310 aarch64 实测在架）→ 生成 curl 清单 → jump 下载 wheels/ → 两跳传输 → e15 `~/kda-ascend/venv` 离线安装（**隔离 venv，共用机器零污染**）。CANN toolkit：torch_npu pip 版自带运行时组件；msprof 等 CANN 工具链 Phase 2 需要时走 .run 离线包同通道。

### 3a. 实测结论与转向：native 模式死路，docker 模式为 e15 唯一已验证路径

2026-09-25 全链路实测后修正（保留原始记录，防止后续重蹈）：

- **native 自装树不可用**：pip torch 2.13.0+cpu + torch_npu 2.13.0rc1 + oepkgs CANN 8.5.0.alpha002（nnae 子包 --extract 提取拼树）→ 设备识别 `npu_available: True`，但**全部 AICORE 算子注册失败**（`ParseDynamicKernels` / `ADD_TO_LAUNCHER_LIST_AICORE`，Add/ReduceSum/Cast/Copy 全灭）。补 host_cpu 软链、补装 device-sw-plugin 子包均无效；判为 alpha 版 opp 与驱动 25.5.2 / torch_npu 2.13 组合不兼容，不再投入。
- **机器已验证组合 = `xllm:glm-clone` 容器**（CANN 8.2.RC1 + torch 2.1.0 + torch_npu 2.1.0.post13，MindIE 2.1rc1 服务镜像）+ **设备直通**（/dev/davinci{N} + davinci_manager + devmm_svm + hisi_hdc）+ **宿主驱动只读挂载**（容器内 /driver 为空壳，libascend_hal.so 必须来自宿主）。RMSNorm fp16 实算 `max_diff 0.00195 ok True`。
- **RemoteTarget.exec_mode = docker**：sync `_build_exec_cmd` 构造 `sudo docker run` 命令（payload/results/驱动三个挂载 + 设备直通），入口 `infra/remote/container_entry.sh`（source 镜像内 set_env.sh + 驱动库注入 + cd /work/payload）。native 模式保留（ cann_env 字段），供未来自装可用的目标机。
- **torch_npu 2.1 兼容点**（runner v0.1 已内置）：设备用 `.npu()` 方法（整数 device 走 CUDA 分支报"Torch not compiled with CUDA"）；计时事件 `torch.npu.Event`（CPU-only 枝干无 `torch.Event`，getattr 默认值会先求值崩溃，须 hasattr 三元）。

### 3b. e15 遗留清理（共用机器，留给 Phase 1 前处理）

自装实验树的处置：`~/kda-ascend/{nnae-rt, nnae, nnae-extract, cann, wheels, nnal}` 可整体删除（docker 模式不再依赖；删除前 `du -sh` 确认无他人文件混入）；`~/kda-ascend/env.sh` 已无消费者，随清理删除。镜像目录/执行属主（root 写 results）不影响 ubuntu 读写（0755），pull 正常。

## 后果

- 实验目录问题的最终答案：**远端只是算力工位，一切事实回本地**——断线/换机器/清理远端均不丢证据；
- 两跳传输延迟（单 job 秒级~分钟级）可接受；大量小文件场景 Phase 1 再评估增量优化；
- e15 供给一次性成本 ~数百 MB 下载；重装 = 重跑 provision 脚本。
