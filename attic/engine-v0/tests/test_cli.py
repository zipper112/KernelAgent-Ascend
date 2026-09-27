"""tests/test_cli.py —— CLI 顶层契约（v0.2：version/桩命令/无参；详细生命周期在 test_harness_loop）。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(*args):
    return subprocess.run([sys.executable, "-m", "harness.cli", *args],
                          capture_output=True, text=True, timeout=60, cwd=str(ROOT))


def test_cli_version_outputs_json():
    """给定 kda version → 则 stdout 是含 version 键的 JSON 且 rc=0。"""
    r = _run("version")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["version"].startswith("0.2.") and data["protocol"].endswith("interaction-protocol.md")


def test_cli_stub_commands_rc2():
    """给定批 C-E 桩命令（gate/promote/export）→ 则 rc=2 且 not-implemented（防误判成功）。"""
    for cmd in ("gate", "promote", "export"):
        r = _run(cmd, "--task", "rmsnorm-smoke")
        assert r.returncode == 2, cmd
        assert json.loads(r.stdout)["error"] == "not-implemented"


def test_cli_no_args_usage_error():
    """给定无参数 → 则 argparse usage 报错（rc 非零）。"""
    r = _run()
    assert r.returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
