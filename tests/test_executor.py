"""tests/test_executor.py —— infra/remote/executor.py 单元测试（mock subprocess，零 SSH）。

期望先行：每用例 docstring 写「给定 → 当 → 则」。重点：错误契约（不抛异常、rc 语义）、
banner 清洗、probe checks 解析、LocalExecutor 桩行为。
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infra.remote import executor as ex  # noqa: E402


def _cp(rc=0, out="", err=""):
    """executor.run 走 text=True → stdout/stderr 恒 str，mock 对齐。"""
    return subprocess.CompletedProcess(args=[], returncode=rc,
                                       stdout=out.decode() if isinstance(out, bytes) else out,
                                       stderr=err.decode() if isinstance(err, bytes) else err)


# ---------- RemoteExecutor.run 错误契约 ----------

def test_run_success_ok(monkeypatch):
    """给定 ssh rc=0 → 则 ok=True、error=None。"""
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: _cp(0, out="hi"))
    r = ex.RemoteExecutor().run("echo hi")
    assert r["ok"] is True and r["error"] is None and r["stdout"] == "hi"


def test_run_nonzero_rc_returns_error_dict(monkeypatch):
    """给定 ssh rc=3 → 则 ok=False、error='rc=3'、不抛异常。"""
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: _cp(3))
    r = ex.RemoteExecutor().run("bad cmd")
    assert r["ok"] is False and r["error"] == "rc=3" and r["rc"] == 3


def test_run_timeout_returns_124(monkeypatch):
    """给定 SSH 链路超时 → 则 rc=124、error 含 link-timeout，不抛异常。"""
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=99)
    monkeypatch.setattr(ex.subprocess, "run", boom)
    r = ex.RemoteExecutor().run("slow")
    assert r["ok"] is False and r["rc"] == 124 and "link-timeout" in r["error"]


def test_run_oserror_returns_127(monkeypatch):
    """给定 ssh 不存在（OSError）→ 则 rc=127、不抛异常。"""
    def boom(*a, **k):
        raise OSError("ssh not found")
    monkeypatch.setattr(ex.subprocess, "run", boom)
    r = ex.RemoteExecutor().run("x")
    assert r["ok"] is False and r["rc"] == 127 and "ssh not found" in r["error"]


def test_run_workdir_cd_prefix(monkeypatch):
    """给定 workdir=/w → 则内层命令含 'cd /w && <cmd>'。"""
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        return _cp(0)
    monkeypatch.setattr(ex.subprocess, "run", fake)
    ex.RemoteExecutor(ex.RemoteTarget(workdir="/w")).run("ls")
    assert "cd /w && ls" in seen["cmd"][-1]


def test_run_timeout_wrapped(monkeypatch):
    """给定 timeout_s=10 → 则内层 ssh 带 'timeout 10' 且外层 subprocess timeout=40。"""
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return _cp(0)
    monkeypatch.setattr(ex.subprocess, "run", fake)
    ex.RemoteExecutor().run("ls", timeout_s=10)
    assert "timeout 10 " in seen["cmd"][-1] and seen["kw"]["timeout"] == 40


# ---------- _clean banner 滤除 ----------

def test_clean_strips_banner_lines():
    """给定含 SSH WARNING banner 的输出 → 则 banner 行被滤掉、正文保留。"""
    noisy = ("** WARNING: connection is not using a post-quantum key exchange algorithm.\n"
             "** This session may be vulnerable to \"store now, decrypt later\" attacks.\n"
             "Authorized users only. All activities may be monitored and reported.\n"
             "real output line\n")
    out = ex.RemoteExecutor._clean(noisy)
    assert "real output line" in out and "WARNING" not in out and "Authorized" not in out


def test_clean_keeps_normal_output():
    """给定纯净输出 → 则原样保留（无行丢失）。"""
    clean = "a\nb\nc\n"
    assert ex.RemoteExecutor._clean(clean) == "a\nb\nc"


# ---------- probe checks 解析 ----------

def test_probe_parses_healthy_checks(monkeypatch):
    """给定健康探测输出（910B4/toolkit/版本号）→ 则 checks 全 ok 且 reachable=True。"""
    healthy = (
        "yq-e15\n---NPU---\n910B4-1  910B4-1\n---CANN---\n8.2\n---PY---\nPython 3.11\n"
        "---TORCHNPU---\n2.7.1\n---TRITON---\n3.5.0\n---DISK---\n500G"
    )
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: _cp(0, out=healthy))
    p = ex.RemoteExecutor().probe()
    assert p["reachable"] is True
    assert p["checks"]["npu"] == "910B4"
    assert p["checks"]["cann_toolkit"] == "ok"
    assert p["checks"]["torch_npu"] == "ok"
    assert p["checks"]["triton"] == "ok"


def test_probe_parses_missing_stack(monkeypatch):
    """给定裸机输出（npu-smi 在但无 toolkit/torch_npu/triton）→ 则相应 missing。"""
    bare = ("e15\n---NPU---\n910B4\n---CANN---\nno-cann-toolkit\n---PY---\nPython 3.10\n"
            "---TORCHNPU---\nModuleNotFoundError: No module named 'torch_npu'\n"
            "---TRITON---\nNo module named 'triton'\n---DISK---\nx")
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: _cp(0, out=bare))
    p = ex.RemoteExecutor().probe()
    assert p["checks"]["npu"] == "910B4"
    assert p["checks"]["cann_toolkit"] == "missing"
    assert p["checks"]["torch_npu"] == "missing"   # v0 修复点：查 TORCHNPU 段而非 PY 段
    assert p["checks"]["triton"] == "missing"


def test_probe_unreachable(monkeypatch):
    """给定 ssh rc=255 → 则 reachable=False、checks 判 missing/none。"""
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: _cp(255))
    p = ex.RemoteExecutor().probe()
    assert p["reachable"] is False and p["checks"]["npu"] == "none"


# ---------- push/pull 转发 ----------

def test_push_delegates_to_sync(monkeypatch, tmp_path):
    """给定缺文件 push → 则错误 dict 从 sync.push_files 透传（转发不吞错）。"""
    r = ex.RemoteExecutor().push(["no/such.py"], local_root=str(tmp_path), remote_dir="/tmp/x")
    assert r["ok"] is False and "本地缺失" in r["error"]


# ---------- LocalExecutor ----------

def test_local_executor_run_echo():
    """给定本地 echo → 则 ok=True 且 stdout 含输出。"""
    r = ex.LocalExecutor().run("echo kda_local_test")
    assert r["ok"] is True and "kda_local_test" in r["stdout"]


def test_local_executor_run_failing_cmd():
    """给定必败命令 → 则 ok=False 且 rc 非零、不抛异常。"""
    r = ex.LocalExecutor().run("exit 7")
    assert r["ok"] is False and r["rc"] == 7


def test_local_executor_push_pull_noop():
    """给定 push/pull → 则 no-op 返回 ok=True。"""
    le = ex.LocalExecutor()
    assert le.push(["a"])["ok"] is True and le.pull(["a"])["ok"] is True
    assert le.probe()["reachable"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
