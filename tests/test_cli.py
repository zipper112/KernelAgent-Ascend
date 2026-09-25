"""tests/test_cli.py —— harness/cli.py 桩契约测试。

期望先行：version 出 JSON 且 rc=0；未实现命令出 not-implemented 且 rc=2（防 agent 误判成功）。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "harness" / "cli.py"


def _run(*args):
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, timeout=30)


def test_cli_version_outputs_json():
    """给定 kda version → 则 stdout 是含 version 键的 JSON 且 rc=0。"""
    r = _run("version")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["version"] == "0.0.0-scaffold" and data["protocol"].endswith("interaction-protocol.md")


def test_cli_unimplemented_command_rc2():
    """给定任意未实现命令（verify）→ 则 rc=2 且输出 not-implemented（防误判成功）。"""
    r = _run("verify")
    assert r.returncode == 2
    data = json.loads(r.stdout)
    assert data["error"] == "not-implemented" and data["command"] == "verify"


def test_cli_no_args_usage_error():
    """给定无参数 → 则 argparse usage 报错（rc 非零）。"""
    r = _run()
    assert r.returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
