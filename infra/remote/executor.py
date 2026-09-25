"""infra/remote/ —— 远程执行通道（独立组件）。

通道形态（2026-09-25 实测）：本地 → jump（10.128.81.36:52222）→ yq-e15（8×910B4）。
两跳经 ~/.ssh/config 的既有条目，不引入额外网络配置。

Executor 协议（harness/core 依赖注入的接口）：
    run(cmd, timeout_s)  -> {ok, rc, stdout, stderr}
    push(local_paths)    -> 同步代码/任务定义到远程工作区（Phase 1 连 rsync）
    pull(remote_paths)   -> 拉回证据文件；大文件留远程只拉摘要（Phase 1）
    probe()              -> 远程环境探测（只读）

v0.0 实现 run + probe；push/pull 为桩（Phase 1 落地 rsync/scp 两跳包装）。
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass

_BANNER = re.compile(r"^\*\* WARNING.*|^\*\* This session.*|^\*\* The server.*|Authorized users only\.")


@dataclass
class RemoteTarget:
    jump: str = "jump"      # ~/.ssh/config 跳板条目名
    host: str = "yq-e15"    # 跳板上可达的目标机
    workdir: str = ""       # 可选：命令执行前 cd 的工作区（config.execution.remote.workdir）


class RemoteExecutor:
    """两跳远程执行器。命令包装：ssh <jump> "timeout T ssh <host> 'cd <workdir> && <cmd>'"。

    错误契约（interaction-protocol §8a）：任何失败（SSH 不可达/内层 rc≠0/超时）都返回
    {ok: False, rc, stdout, stderr, error} 的 dict——本层**不抛异常**，调用方按 dict 分支。
    """

    def __init__(self, target: RemoteTarget | None = None):
        self.target = target or RemoteTarget()

    # -- Executor 协议 --
    def run(self, cmd: str, timeout_s: int = 120) -> dict:
        if self.target.workdir:
            cmd = f"cd {shlex.quote(self.target.workdir)} && {cmd}"
        inner = (f"timeout {timeout_s} ssh -o BatchMode=yes "
                 f"-o StrictHostKeyChecking=accept-new {self.target.host} {shlex.quote(cmd)}")
        try:
            proc = subprocess.run(["ssh", "-o", "BatchMode=yes", self.target.jump, inner],
                                  capture_output=True, text=True, timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            return {"ok": False, "rc": 124, "stdout": "", "stderr": "", "error": f"link-timeout>{timeout_s + 30}s"}
        except OSError as e:  # ssh 不存在等
            return {"ok": False, "rc": 127, "stdout": "", "stderr": "", "error": str(e)}
        return {"ok": proc.returncode == 0, "rc": proc.returncode,
                "stdout": self._clean(proc.stdout), "stderr": self._clean(proc.stderr),
                "error": None if proc.returncode == 0 else f"rc={proc.returncode}"}

    def push(self, local_paths: list[str]) -> dict:   # Phase 1: rsync/scp 两跳
        raise NotImplementedError("Phase 1: rsync via jump")

    def pull(self, remote_paths: list[str]) -> dict:  # Phase 1: 只拉摘要
        raise NotImplementedError("Phase 1: rsync via jump")

    def probe(self) -> dict:
        script = (
            "hostname; echo ---NPU---; npu-smi info 2>/dev/null | head -14 || echo no-npu-smi; "
            "echo ---CANN---; ls /usr/local/Ascend/ascend-toolkit 2>/dev/null || echo no-cann-toolkit; "
            "echo ---PY---; python3 --version; "
            "echo ---TORCHNPU---; python3 -c 'import torch_npu; print(torch_npu.__version__)' 2>&1 | tail -1; "
            "echo ---TRITON---; python3 -c 'import triton; print(triton.__version__)' 2>&1 | tail -1; "
            "echo ---DISK---; df -h /home | tail -1"
        )
        r = self.run(script, timeout_s=90)
        out = r["stdout"]
        checks = {
            "npu": "910B4" if "910B4" in out else ("present" if "npu-smi" in out else "none"),
            "cann_toolkit": "missing" if "no-cann-toolkit" in out else "ok",
            "torch_npu": "missing" if ("No module" in out.split("---TORCHNPU---")[0].split("---PY---")[-1] or "Error" in out.split("---TORCHNPU---")[0].split("---PY---")[-1]) else "ok",
            "triton": "missing" if ("No module" in out.split("---TRITON---")[-1] or "Error" in out.split("---TRITON---")[-1]) else "ok",
        }
        return {"reachable": r["ok"], "checks": checks, "raw": out}

    @staticmethod
    def _clean(out: str) -> str:
        return "\n".join(l for l in out.splitlines() if not _BANNER.match(l.strip()))


class LocalExecutor:
    """本地执行（Executor 协议的空实现——core 默认注入，远程为零感知）。"""

    def run(self, cmd: str, timeout_s: int = 120) -> dict:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s)
        return {"ok": proc.returncode == 0, "rc": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr}

    def push(self, local_paths: list[str]) -> dict:
        return {"ok": True, "note": "local: no-op"}

    def pull(self, remote_paths: list[str]) -> dict:
        return {"ok": True, "note": "local: no-op"}

    def probe(self) -> dict:
        return {"reachable": True, "checks": {}, "raw": ""}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="远程执行通道（v0.0：run/probe）")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--jump", default="jump")
    ap.add_argument("--host", default="yq-e15")
    args = ap.parse_args()
    ex = RemoteExecutor(RemoteTarget(args.jump, args.host))
    print(json.dumps(ex.probe() if args.probe else ex.run(args.run), ensure_ascii=False, indent=2))
