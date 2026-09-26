"""harness/context.py —— NPU 操作上下文生成器（ADR-013：上下文注入替代远程执行层）。

职责：把"如何访问远程 NPU"编译成一份 markdown 手册，注入 writer prompt（===EXEC=== 块
的写作依据）+ 落盘 tasks/<t>/docs/npu-access.md（人可查）。harness 不再代执行——
LLM 依据手册自己写命令序列，exec_policy 护栏把关，canonical runner 是唯一出数口径。

三条底线（ADR-013）：
1. canonical runner 不可绕——性能/正确性数字只能出自 infra/remote/runner.py；
2. job.json 由 harness 生成（workload 双档/chained 档位是测量纪律，不交给 LLM）；
3. 账本唯一写方 = Evidence（LLM 的 exec 输出进 prompt 与 audit，不直接进账本）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

RUNNER_REL = "infra/remote/runner.py"
ENTRY_REL = "infra/remote/container_entry.sh"


def _sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def build_job_payload(task_root: Path, candidate_id: str, kind: str,
                      workloads: list[dict], extra: dict, device_id: int) -> dict:
    """canonical job.json 内容（原 JobSpec 语义迁移；唯一合法构造方=本函数）。
    kind ∈ verify|bench；docker 模式下 runner 在容器内只见单卡——device_id 恒 0，
    物理卡由 --device 直通表达（physical_device_id 回显）。"""
    import time
    assert kind in ("verify", "bench", "profile"), f"非法 kind: {kind}"
    if kind == "bench":
        assert int(extra.get("warmup", 3)) > 0 and int(extra.get("samples", 5)) > 0
    return {
        "job_id": f"{task_root.name}-{kind}-{candidate_id}-{int(time.time())}",
        "kind": kind, "candidate_id": candidate_id, "task": task_root.name,
        "workloads": workloads,
        "device_id": 0, "physical_device_id": device_id,
        "extra": extra,
    }


def canonical_cmd(job: dict, remote: dict) -> str:
    """canonical 执行命令模板（docker 直通；从原 _build_exec_cmd 迁移为纯模板）。
    remote = config.yaml execution.remote 段；workspace = e15 上任务镜像目录。"""
    ws = remote_workspace(task_root_name=job["task"], remote=remote)
    image = remote.get("docker_image", "")
    assert image, "docker 模式必须给 docker_image"
    dev = job["physical_device_id"]
    return (
        f"sudo docker run --rm"
        f" --device /dev/davinci{dev} --device /dev/davinci_manager"
        f" --device /dev/devmm_svm --device /dev/hisi_hdc"
        f" -v {ws}/payload:/work/payload:ro"
        f" -v {ws}/results:/work/results"
        f" -v /usr/local/Ascend/driver:/usr/local/Ascend/driver_host:ro"
        f" --entrypoint bash {image}"
        f" /work/payload/{ENTRY_REL}"
        f" /work/payload/{RUNNER_REL}"
        f" --job job.json --results-dir /work/results"
    )


def remote_workspace(task_root_name: str, remote: dict) -> str:
    mirror = remote.get("mirror_root", "~/kda-ascend").rstrip("/")
    return f"{mirror}/tasks/{task_root_name}"


def payload_files(task_root: Path, candidate_id: str) -> list[str]:
    """EXEC 同步清单：候选 + 冻结面（reference/workloads）+ runner 双件。
    workloads.yaml 随行只为 sha 溯源（runner 读 job.json 内联定义）。"""
    files = [f"solution/{candidate_id}/candidate.py", RUNNER_REL, ENTRY_REL]
    for extra in ("reference.py", "bench/workloads.yaml"):
        if (task_root / extra).exists():
            files.append(extra)
    return files


def access_manual(task_root: Path, remote: dict) -> str:
    """NPU 访问手册（writer prompt 注入 + 落盘 docs/npu-access.md）。"""
    host = remote.get("host", "yq-e15")
    dev = remote.get("device_id", 0)
    image = remote.get("docker_image", "")
    ws = remote_workspace(task_root.name, remote)
    shas = {}
    for f in ("reference.py", "bench/workloads.yaml"):
        p = task_root / f
        if p.exists():
            shas[f] = _sha8(p)
    sha_txt = "\n".join(f"- {k}: `{v}`" for k, v in shas.items()) or "- （无）"
    verify_job = build_job_payload(task_root, "<cid>", "verify", [], {}, dev)
    return f"""# NPU 访问上下文（harness 生成，每次迭代刷新）

## 环境
- 目标机：`{host}`（从本机直接 `ssh {host}` 可达，BatchMode 免密已配）
- 设备：NPU {dev}（测量前后建议 `npu-smi info` 自查占用——他人进程会污染数据）
- 容器镜像：`{image}`（CANN 8.2.RC1 + torch 2.1.0 + torch_npu 2.1.0；宿主 native CANN 已坏勿用）
- 任务 workspace：`{ws}/`（payload/ 与 results/ 子目录）

## 你的 EXEC 义务（===EXEC=== 块写法）
1. **同步**：把候选与冻结面推到 workspace/payload（tar 管道一条即可）：
   `tar cf - solution/<cid>/candidate.py reference.py bench/workloads.yaml {RUNNER_REL} {ENTRY_REL} | ssh {host} 'mkdir -p {ws}/payload {ws}/results && tar xf - -C {ws}/payload'`
2. **job.json**：harness 已生成在本地 `run/job.json`（workload 档位/chained 参数是测量纪律，
   **你不许改写它**；有异议在 HYPOTHESIS 里说）——同步后放到 `{ws}/payload/job.json`。
3. **canonical 出数**（唯一合法口径，参数照抄）：
   `{canonical_cmd(verify_job, remote)}`
4. **回读结果**：`ssh {host} 'cat {ws}/results/<job_id>.json'`（job_id 见 job.json）。

## 自由度（允许但不替代 canonical）
- 前置探针：compile 冒烟、TRITON_CACHE 检查、单 workload 试跑、npu-smi 状态——加在 canonical 之前
- 诊断复跑：verify 挂了可以小步探（如容器内 python -c 单测 kernel 导入）
- **红线**：出给评审的性能/正确性数字必须出自 canonical runner 的 results json；
  自建计时脚本的结果只可用于诊断叙述，出现在结论里=评审直接 REJECT

## 冻结面（sha256[:8]，篡改=REJECT）
{sha_txt}
- `infra/remote/runner.py`（canonical 口径，勿改）
- `bench/workloads.yaml`（行集冻结；档位选择在 job.json）
"""


__all__ = ["access_manual", "build_job_payload", "canonical_cmd", "payload_files",
           "remote_workspace", "RUNNER_REL", "ENTRY_REL"]
