"""infra/secrets/ —— 密钥管理（独立组件）。

规则（ADR-007）：
- 密钥只存在两处：环境变量（优先）或 agent-config/local-secrets.yaml（gitignored）；
- 本模块是**唯一**读密钥的代码；harness/models.py 经此取 key，不自行解析文件；
- 永不打印/日志 key 全文（get() 只返回值，__repr__ 打码）。
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_SECRETS = _REPO_ROOT / "agent-config" / "local-secrets.yaml"


class SecretRef:
    """密钥引用（防止日志误打印）。"""

    def __init__(self, key: str, value: str):
        self._key, self._value = key, value

    def get(self) -> str:
        return self._value

    def __repr__(self) -> str:  # 打码
        v = self._value
        return f"SecretRef({self._key}={v[:4]}***{v[-4:] if len(v) > 8 else ''})"


def get_glm_key() -> SecretRef | None:
    """取 GLM key：环境变量 GLM_API_KEY 优先，其次 local-secrets.yaml。都没有返回 None。"""
    env = os.environ.get("GLM_API_KEY")
    if env:
        return SecretRef("env:GLM_API_KEY", env)
    if _LOCAL_SECRETS.exists():
        text = _LOCAL_SECRETS.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("api_key:"):
                val = line.split("api_key:", 1)[1].strip().strip('"').strip("'")
                if val:
                    return SecretRef("file:local-secrets.yaml", val)
    return None


def get_ssh_targets() -> dict:
    """读 SSH 目标配置（jump/host），无密钥敏感信息，可安全返回。"""
    import yaml  # 容错：无 yaml 时返回默认
    try:
        data = yaml.safe_load(_LOCAL_SECRETS.read_text(encoding="utf-8")) if _LOCAL_SECRETS.exists() else {}
    except Exception:  # noqa: BLE001
        data = {}
    ssh = (data or {}).get("ssh", {}) or {}
    return {"jump": ssh.get("jump_host", "jump"), "host": ssh.get("target", "yq-e15")}


if __name__ == "__main__":
    k = get_glm_key()
    print(f"glm key: {k!r}" if k else "glm key: NOT FOUND (set GLM_API_KEY or local-secrets.yaml)")
    print(f"ssh targets: {get_ssh_targets()}")
