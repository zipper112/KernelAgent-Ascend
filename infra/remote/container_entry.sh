#!/bin/bash
# infra/remote/container_entry.sh —— 容器执行入口（ADR-011 §8b docker 模式）。
# 背景（2026-09-25 实测）：e15 自装 CANN 8.5.alpha002 树 AICORE 算子全量注册失败
# （ParseDynamicKernels）；机器已验证可用组合 = xllm:glm-clone 容器（CANN 8.2.RC1
# + torch 2.1.0 + torch_npu 2.1.0）+ 设备直通 + 宿主驱动只读挂载。
#
# 挂载约定（sync.run_job docker 分支构造）：
#   /work/payload                = 任务 payload（ro；runner/job.json/候选代码）
#   /work/results                = 结果输出（rw；<job_id>.json 落这里被 pull）
#   /usr/local/Ascend/driver_host= 宿主机驱动（ro；容器内 /driver 为空壳）
# 参数："$@" = runner 脚本路径 + runner 参数（--job/--results-dir）
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver_host/lib64/common:/usr/local/Ascend/driver_host/lib64/driver:$LD_LIBRARY_PATH
cd /work/payload
exec python3 "$@"
