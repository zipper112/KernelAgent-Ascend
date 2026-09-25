"""tests/test_harness_loop.py —— 最小闭环单元测试（evidence/state/CLI，协议 v0.2 批 A+B）。

期望先行：每用例 docstring「给定→当→则」。全部离线（verify/bench 的远端路径不在此测，
用桩注入 run_job 结果验证 evidence 落盘契约）。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.core.evidence import Evidence, atomic_write_json  # noqa: E402
from harness.core.state import SessionLock, TaskState  # noqa: E402


# ---------- Evidence ----------

def test_audit_roundtrip_and_actor_guard(tmp_path):
    """给定合法 actor 写 audit → 则 JSON 行可解析；非法 actor → 断言拒绝。"""
    ev = Evidence(tmp_path)
    ev.log_audit("agent", "kda verify", target="c001", round_=3, detail={"passed": True})
    lines = (tmp_path / "docs" / "audit.log").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert rec["actor"] == "agent" and rec["round"] == 3 and rec["detail"]["passed"] is True
    with pytest.raises(AssertionError):
        ev.log_audit("evil", "x")


def test_benchmark_csv_schema_and_verdict_vocab(tmp_path):
    """给定 append_benchmark → 则表头+行符合 §3.1；非法 verdict/workload_set 断言拒绝。"""
    ev = Evidence(tmp_path)
    ev.append_benchmark("c003", "c001", "P2", "l1", 412.5, 409.1, 431.0, 1.38, "keep", "ub")
    head = (tmp_path / "docs" / "benchmark.csv").read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("ts,candidate_id,parent_id,phase,workload_set")
    row = (tmp_path / "docs" / "benchmark.csv").read_text(encoding="utf-8").splitlines()[1]
    assert ",keep," in row and "412.5" in row
    with pytest.raises(AssertionError):
        ev.append_benchmark("c", None, "P1", "l0", 1, 1, 1, 1, "kept")      # 非法词表
    with pytest.raises(AssertionError):
        ev.append_benchmark("c", None, "P1", "huge", 1, 1, 1, 1, "keep")    # 非法集


def test_solution_jsonl_verify_fail_enters_chain(tmp_path):
    """给定 verify 失败的候选 → 则 status=reject + stage=verify 入链（v0.2 防单方向无限烧）。"""
    ev = Evidence(tmp_path)
    ev.append_solution("c002", "c001", "vload", "向量化加载", "reject", round_=2, stage="verify")
    sols = ev.load_solutions()
    assert sols[0]["status"] == "reject" and sols[0]["stage"] == "verify"
    assert sols[0]["diff_ref"] == "solution/c002/"


def test_usage_two_level_ledger(tmp_path):
    """给定两次自报 → 则任务/全局账本都追加且 cumulative 单调累计。"""
    ev = Evidence(tmp_path)
    g = tmp_path / "run-global" / "usage.jsonl"
    ev.append_usage("t", "host-agent", "self-report", 100, 50, global_ledger=g)
    res = ev.append_usage("t", "host-agent", "self-report", 200, 50, global_ledger=g)
    assert res["cumulative_global"] == 400 and res["cumulative_task"] == 400
    assert len(g.read_text(encoding="utf-8").splitlines()) == 2


def test_atomic_write_json(tmp_path):
    """给定 atomic_write_json → 则落盘且无 .tmp 残留。"""
    p = tmp_path / "run" / "state.json"
    p.parent.mkdir()
    atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text(encoding="utf-8"))["a"] == 1
    assert not list(tmp_path.rglob("*.tmp"))


# ---------- SessionLock ----------

def test_lock_create_takeover_and_conflict(tmp_path, monkeypatch):
    """给定锁生命周期 → 则：创建成功；同 mode 刷新 last_seen；异 mode 活跃拒绝；
    陈旧（mtime 30 分钟前）自动接管并记 audit。"""
    ev = Evidence(tmp_path)
    lock = SessionLock(tmp_path, ev, mode="companion")
    c1 = lock.acquire_or_takeover()
    assert c1["mode"] == "companion"
    lock2 = SessionLock(tmp_path, ev, mode="companion")
    c2 = lock2.acquire_or_takeover()          # 同 mode：刷新
    assert c2["last_seen"] >= c1["last_seen"]
    pipeline = SessionLock(tmp_path, ev, mode="pipeline")
    with pytest.raises(PermissionError):      # 异 mode 活跃：拒绝
        pipeline.acquire_or_takeover()
    # 陈旧：mtime 拨回 31 分钟前
    import os, time
    old = time.time() - 31 * 60
    os.utime(tmp_path / "run" / "lock", (old, old))
    c3 = pipeline.acquire_or_takeover()
    assert c3["mode"] == "pipeline"           # 接管成功
    audit = (tmp_path / "docs" / "audit.log").read_text(encoding="utf-8")
    assert "lock-takeover" in audit


# ---------- TaskState ----------

def test_state_init_require_and_direction_fails(tmp_path):
    """给定 new-task 引导 → 则 round=0；require 缺失抛错；bump_direction_fail 累计。"""
    TaskState.init_for_new_task(tmp_path, "t1")
    st = TaskState(tmp_path, Evidence(tmp_path))
    assert st.require()["round"] == 0
    empty = TaskState(tmp_path / "nope", Evidence(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        empty.require()
    assert st.bump_direction_fail("vload") == 1
    assert st.bump_direction_fail("vload") == 2
    assert st.require()["direction_fails"]["vload"] == 2


# ---------- CLI 集成（python -m harness.cli） ----------

def _kda(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "harness.cli", *args],
                          capture_output=True, text=True, timeout=120, cwd=str(cwd))


def test_cli_new_task_creates_seven_pieces_and_state(tmp_path):
    """给定 kda new-task → 则七件套+state.json(round=0) 创建；重复创建 rc=2。"""
    d = tmp_path / "tasks" / "alpha"
    d.parent.mkdir()
    r = _kda("new-task", str(d))
    assert r.returncode == 0
    assert (d / "task.yaml").exists() and (d / "run" / "state.json").exists()
    assert json.loads((d / "run" / "state.json").read_text(encoding="utf-8"))["round"] == 0
    r2 = _kda("new-task", str(d))
    assert r2.returncode == 2


def test_cli_status_and_round_flag(tmp_path):
    """给定新任务 → 则 status 出完整 state 摘要；--round 只出数字。"""
    d = tmp_path / "tasks" / "beta"
    d.parent.mkdir()
    _kda("new-task", str(d))
    r = _kda("status", "--task", str(d))
    assert r.returncode == 0 and json.loads(r.stdout)["round"] == 0
    r2 = _kda("status", "--task", str(d), "--round")
    assert r2.stdout.strip() == "0"


def test_cli_budget_report_writes_ledgers(tmp_path):
    """给定 budget --report → 则任务账本+全局账本各一行、audit 记一条。"""
    d = tmp_path / "tasks" / "gamma"
    d.parent.mkdir()
    _kda("new-task", str(d))
    r = _kda("budget", "--task", str(d), "--report", "--tokens", "12000", "--round", "1")
    assert r.returncode == 0
    assert json.loads(r.stdout)["cumulative_task"] == 12000
    assert (d / "run" / "usage.jsonl").exists()
    assert "budget-report" in (d / "docs" / "audit.log").read_text(encoding="utf-8")


def test_cli_verify_failure_enters_chain_and_counts(tmp_path, monkeypatch):
    """给定 verify 未通过（远端返回 passed=false）→ 则 solutions.jsonl 出 reject/stage=verify 行、
    direction_fails 计数、audit 记条、退出码 1（v0.2 核心防白烧契约）。"""
    d = tmp_path / "tasks" / "delta"
    d.parent.mkdir()
    _kda("new-task", str(d))
    # 造 round-0-contract 带 direction + 候选
    (d / "run" / "round-0-contract.md").write_text(
        "# R0\n- 本轮方向：\n  direction: vload\n", encoding="utf-8")
    (d / "solution" / "c001").mkdir(parents=True)
    (d / "solution" / "c001" / "candidate.py").write_text("def kernel(i): return i[0]\n", encoding="utf-8")
    (d / "reference.py").write_text("def reference(i): return i[0]\n", encoding="utf-8")
    # 远端 enabled 但注入桩（monkeypatch 不了子进程——直接写 config 走 _run_remote_job 的 import 路径）
    (d / "config.yaml").write_text(
        "execution:\n  remote:\n    enabled: true\n    jump: jump\n    host: h\n"
        "    exec_mode: docker\n    docker_image: img\n    device_id: 0\n", encoding="utf-8")
    # 桩：往 sys.modules 注入假 sync（CLI 是子进程，改为环境变量开关不优雅——
    # 本测试直接调 CLI 内部函数路径，等价覆盖）
    sys.path.insert(0, str(ROOT))
    import harness.cli as cli
    fake = {"ok": True, "result": {"passed": False,
                                   "workloads": [{"id": "w01", "passed": False, "err_ratio": 0.9}]}}
    monkeypatch.setattr(cli, "_run_remote_job", lambda *a, **k: fake)
    import argparse as ap
    args = ap.Namespace(task=str(d), candidate="c001", workload_set="l0")
    rc = cli.cmd_verify(args)
    assert rc == 1
    sols = (d / "docs" / "solutions.jsonl").read_text(encoding="utf-8")
    assert '"status": "reject"' in sols and '"stage": "verify"' in sols and '"direction": "vload"' in sols
    state = json.loads((d / "run" / "state.json").read_text(encoding="utf-8"))
    assert state["direction_fails"]["vload"] == 1


def test_cli_stub_commands_return_2(tmp_path):
    """给定桩命令 gate → 则 rc=2 + not-implemented（防误判成功契约保留）。"""
    d = tmp_path / "tasks" / "eps"
    d.parent.mkdir()
    _kda("new-task", str(d))
    r = _kda("gate", "--task", str(d))
    assert r.returncode == 2 and json.loads(r.stdout)["error"] == "not-implemented"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
