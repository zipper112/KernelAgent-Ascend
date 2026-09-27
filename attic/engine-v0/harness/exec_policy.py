"""harness/control/exec_policy.py —— LLM EXEC 块安全护栏（ADR-013；v2 黑名单制）。

v1 前缀白名单三连误伤合法 shell（set -e / export / CID=c003 变量赋值各烧一轮 token）
——教训：LLM 写的是真 shell 脚本，白名单枚举不完 shell 语法。v2 只保黑名单：
账本/冻结面不可写、破坏性命令禁绝；其余（变量/循环/管道/探针）放行。
违规整块拒绝（audit 记 exec-policy-reject，反馈下轮重写）——与 preflight 同款纪律。
"""
from __future__ import annotations

import re

_FORBIDDEN_PATTERNS = (
    # 账本与冻结面：目标=仓内相对路径（docs/xxx、裸文件名）。远端 workspace 落点
    # （~/…、/work/…）是手册规定的合法同步位置，不在保护列。
    (re.compile(r"(>{1,2}|<<)\s*((docs/)?(benchmark\.csv|solutions\.jsonl|audit\.log))\b"), "写账本"),
    (re.compile(r"(>{1,2}|<<)\s*((docs/)?(reference\.py|workloads\.yaml|runner\.py|container_entry\.sh))\b"), "写冻结面"),
    (re.compile(r"(benchmark\.csv|solutions\.jsonl|audit\.log)\s*(<<|>>)"), "追加账本"),
    (re.compile(r"(sed\s+-i|tee\s+-a\b|truncate|dd\s+of=)[^;|&]*(docs/)?(benchmark\.csv|solutions\.jsonl|audit\.log)"), "工具写账本"),
    # 破坏性
    (re.compile(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|-[rf]{2,})\s"), "rm -rf"),
    (re.compile(r"git\s+(push|reset|checkout\s+--|clean)"), "git 破坏性操作"),
    (re.compile(r"mkfs|dd\s+if=/dev/|shutdown|reboot"), "系统破坏"),
    # 远程代码执行注入面
    (re.compile(r"(curl|wget)[^|;]*\|\s*(ba)?sh"), "curl|sh 注入"),
)


def check_exec_block(script: str) -> tuple[bool, str | None]:
    """检查 EXEC 块全文（逐逻辑行，黑名单制）。返回 (ok, reason)。"""
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for pat, name in _FORBIDDEN_PATTERNS:
            if pat.search(line):
                return False, f"违规[{name}]: {line[:80]}"
    return True, None


__all__ = ["check_exec_block"]
