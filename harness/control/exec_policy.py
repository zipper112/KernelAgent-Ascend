"""harness/control/exec_policy.py —— LLM EXEC 块安全护栏（ADR-013）。

LLM 自主写远端命令的前提：账本与冻结面不可触碰。exec 前逐行检查，违规整块拒绝
（audit 记 exec-policy-reject，反馈下轮重写）——与 preflight-reject 同款纪律。
"""
from __future__ import annotations

import re

# 账本与冻结面文件名（写入即违规——harness 是唯一写方）
_PROTECTED_FILES = ("benchmark.csv", "solutions.jsonl", "audit.log",
                    "reference.py", "workloads.yaml", "runner.py",
                    "container_entry.sh", "job.json")

# 允许的命令首词（远端操作面 + 同步 + 探针；本地操作只许只读）
_ALLOWED_FIRST = re.compile(
    r"^(ssh|scp|rsync|tar|cat|echo|mkdir|ls|grep|sed|npu-smi|nvidia-smi|python[0-9.]*|"
    r"sudo\s+docker|docker|true|false|which|head|tail|wc|find)\b")

# 危险模式（无论位置）
_FORBIDDEN_PATTERNS = (
    (re.compile(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|-[rf]{2,})\s"), "rm -rf"),
    # 写账本/冻结面：目标=仓内相对路径（docs/xxx、裸文件名）。远端 workspace 路径
    # （~/…、/work/…、…/payload/job.json）是手册规定的合法同步落点，不在保护列。
    (re.compile(r"(>{1,2}|<<)\s*((docs/)?(benchmark\.csv|solutions\.jsonl|audit\.log))\b"), "写账本"),
    (re.compile(r"(>{1,2}|<<)\s*((docs/)?(reference\.py|workloads\.yaml|runner\.py|container_entry\.sh))\b"), "写冻结面"),
    (re.compile(r"(benchmark\.csv|solutions\.jsonl|audit\.log)\s*(<<|>>)"), "追加账本"),
    (re.compile(r"(sed\s+-i|tee\s+-a\b|truncate|dd\s+of=)[^;|&]*(docs/)?(benchmark\.csv|solutions\.jsonl|audit\.log)"), "工具写账本"),
    (re.compile(r"git\s+(push|reset|checkout\s+--|clean)"), "本地 git 破坏性操作"),
    (re.compile(r"mkfs|dd\s+if=/dev/|shutdown|reboot"), "系统破坏"),
)

# 本地（非 ssh 前缀）允许的操作：只读探针与同步源
_LOCAL_ALLOWED = re.compile(r"^(tar|cat|echo|ls|grep|head|tail|wc|which|find)\b")


def check_exec_block(script: str) -> tuple[bool, str | None]:
    """检查 EXEC 块全文（逐逻辑行）。返回 (ok, reason)；ok=False 时 reason 指向首个违规行。"""
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for pat, name in _FORBIDDEN_PATTERNS:
            if pat.search(line):
                return False, f"违规[{name}]: {line[:80]}"
        if not _ALLOWED_FIRST.match(line):
            return False, f"命令不在白名单: {line[:80]}"
        # 本地行（不经 ssh/docker）只许只读
        if not (line.startswith("ssh ") or line.startswith("sudo docker") or line.startswith("docker")):
            if not _LOCAL_ALLOWED.match(line):
                return False, f"本地命令须只读（加 ssh 前缀走远端）: {line[:80]}"
    return True, None


__all__ = ["check_exec_block"]
