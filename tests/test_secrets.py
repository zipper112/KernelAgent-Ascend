"""tests/test_secrets.py —— infra/secrets/provider.py 单元测试（env monkeypatch + tmp 文件）。

期望先行：每用例 docstring 写「给定 → 当 → 则」。密钥纪律：repr 打码、env 优先、无泄漏。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infra.secrets import provider  # noqa: E402

KEY = "6f56fdd6c6934fbab2d67cad7628e83f.0oiGWgjWKD4iX137"


def test_secretref_repr_masked():
    """给定含完整 key 的 SecretRef → 则 repr 只露前4后4，全文不出现。"""
    r = repr(provider.SecretRef("env:GLM_API_KEY", KEY))
    assert KEY not in r and KEY[:4] in r and KEY[-4:] in r and "***" in r


def test_secretref_get_returns_full_value():
    """给定 SecretRef → 则 get() 返回完整值（业务侧可用）。"""
    assert provider.SecretRef("k", KEY).get() == KEY


def test_get_glm_key_env_priority(monkeypatch, tmp_path):
    """给定 env 有 key 且文件也有 → 则 env 优先（来源标注 env:）。"""
    monkeypatch.setenv("GLM_API_KEY", KEY)
    monkeypatch.setattr(provider, "_LOCAL_SECRETS", tmp_path / "local-secrets.yaml")
    (tmp_path / "local-secrets.yaml").write_text(f'api_key: "filekey123"\n', encoding="utf-8")
    ref = provider.get_glm_key()
    assert ref is not None and ref.get() == KEY


def test_get_glm_key_from_file(monkeypatch, tmp_path):
    """给定 env 无 key、文件有 → 则解析文件值（来源标注 file:）。"""
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    f = tmp_path / "local-secrets.yaml"
    monkeypatch.setattr(provider, "_LOCAL_SECRETS", f)
    f.write_text(f"api_key: '{KEY}'\n", encoding="utf-8")
    ref = provider.get_glm_key()
    assert ref is not None and ref.get() == KEY


def test_get_glm_key_none_when_absent(monkeypatch, tmp_path):
    """给定 env 与文件皆无 → 则 None。"""
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.setattr(provider, "_LOCAL_SECRETS", tmp_path / "nope.yaml")
    assert provider.get_glm_key() is None


def test_get_ssh_targets_defaults_when_missing(monkeypatch, tmp_path):
    """给定无 local-secrets → 则回退默认 jump/yq-e15。"""
    monkeypatch.setattr(provider, "_LOCAL_SECRETS", tmp_path / "nope.yaml")
    assert provider.get_ssh_targets() == {"jump": "jump", "host": "yq-e15"}


def test_get_ssh_targets_parses_file(monkeypatch, tmp_path):
    """给定文件含 ssh 段 → 则解析出目标。"""
    f = tmp_path / "local-secrets.yaml"
    monkeypatch.setattr(provider, "_LOCAL_SECRETS", f)
    f.write_text("ssh:\n  jump_host: jump2\n  target: npu9\n", encoding="utf-8")
    assert provider.get_ssh_targets() == {"jump": "jump2", "host": "npu9"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
