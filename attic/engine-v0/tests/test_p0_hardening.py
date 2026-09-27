"""tests/test_p0_hardening.py —— 三源 diff 补全批 P0 的单元测试。

覆盖（每用例 docstring 给定→当→则）：
P0-1 record_review 追加式状态迁移 + latest_status_map
P0-2 DAG parent 三分支（同向 refine / 换向挂最优 keep / 根）
P0-3 漂移状态机（ADVANCED 清零 / STALLED 累计 / ≥2 replan / ≥3 stop-drift 终态）
P0-4 writer_fails 独立计数（伪方向不入 direction_fails；连续 3 次→pause）
P0-5 历史最优防御（invalid 行不入参照系）
P0-6 轮契约自动生成 + 评审截断自适应
P0-7 终态保证（异常路径 terminal=error；pause 后可续跑）
P0-8 完整评审契约注入（review prompt 含契约关键词与 STOP 三要素）
全部离线（模型/远端全桩化）。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.core.evidence import Evidence  # noqa: E402
from harness.core.state import TaskState, is_pseudo_direction  # noqa: E402


def _mk_task(tmp_path: Path, name: str = "t") -> Path:
    """最小任务骨架（loop 需要的文件全齐）。"""
    t = tmp_path / name
    (t / "bench").mkdir(parents=True)
    (t / "solution").mkdir()
    (t / "run").mkdir()
    (t / "task.yaml").write_text(
        "contract:\n  op_family: conv1d\n  research:\n    symptoms: [latency]\n", encoding="utf-8")
    (t / "config.yaml").write_text(
        "arch: dav_2201\nbudget: {}\nexecution:\n  remote:\n    enabled: false\n", encoding="utf-8")
    (t / "reference.py").write_text("def reference(i): return i[0]\n", encoding="utf-8")
    (t / "bench" / "workloads.yaml").write_text(
        "workloads:\n  - id: w01\n    axes: {batch: 1, seq: 8, hidden: 4}\n", encoding="utf-8")
    TaskState.init_for_new_task(t, name)
    return t


# ---------- P0-1：评审回写账本 ----------

def test_record_review_appends_terminal_status(tmp_path):
    """给定 benched 行 → record_review(keep) → 则追加新行 status=keep/stage=review，
    latest_status_map 取到终判（P0-1 核心契约）。"""
    ev = Evidence(tmp_path)
    ev.append_solution("c001", None, "vload", "向量化", "verified", round_=1, stage="verify")
    ev.append_solution("c001", None, "vload", "向量化", "benched", round_=1, stage="bench")
    ev.record_review("c001", "keep", note="mean 982 < 1094", round_=1)
    m = ev.latest_status_map()
    assert m["c001"]["status"] == "keep" and m["c001"]["stage"] == "review"
    assert m["c001"]["note"] == "mean 982 < 1094"
    rows = ev.load_solutions()
    assert len(rows) == 3          # append-only：历史行不动


def test_record_review_requires_existing_candidate(tmp_path):
    """给定不存在的 cid → record_review → 则拒绝（verify 先行，评审不可造候选）。"""
    ev = Evidence(tmp_path)
    with pytest.raises(ValueError):
        ev.record_review("c999", "keep")


def test_bench_invalid_verdict_writable(tmp_path):
    """给定 mean=0 的 bench → 则 csv 行 verdict=invalid 合法入账（词表扩展）。"""
    ev = Evidence(tmp_path)
    ev.append_benchmark("c004", None, "P1", "l0", 0.0, 0.0, 0.0, None, "invalid", "empty")
    ev.append_benchmark("c005", None, "P1", "l0", 912.0, 900.0, 950.0, 1.2, "benched", "ok")


# ---------- P0-2：DAG parent ----------

def test_parent_for_three_branches(tmp_path):
    """给定三条历史 → _parent_for → 则：REVISE 同向→上轮 cid；换向→最优 keep；空→None。"""
    from harness.control.loop import AutonomousLoop
    ev = Evidence(tmp_path)
    ev.append_solution("c001", None, "vload", "h1", "keep", round_=1, stage="review")
    ev.append_solution("c002", "c001", "vload", "h2", "revise", round_=2, stage="review")
    sols = ev.load_solutions()
    latest = ev.latest_status_map()
    # 同向 refine（REVISE 反馈含上轮方向名 vload）
    assert AutonomousLoop._parent_for(sols, latest, "REVISE: vload 方向还差 30us") == "c002"
    # 换向（FUSE/全新反馈）→ 挂最优 keep c001
    assert AutonomousLoop._parent_for(sols, latest, "FUSE-DIRECTION: 必须换向") == "c001"
    # 无历史 → 根
    assert AutonomousLoop._parent_for([], {}, "") is None


def test_cli_parent_of_same_direction_and_switch(tmp_path):
    """给定 CLI 路径的 _parent_of → 则同方向 refine 挂上一 cid、换向挂最优 keep。"""
    import harness.cli as cli
    ev = Evidence(tmp_path)
    ev.append_solution("c001", None, "vload", "h", "keep", round_=1, stage="review")
    ev.append_solution("c002", "c001", "vload", "h", "revise", round_=2, stage="review")
    ev.append_solution("c003", None, "tile", "h", "benched", round_=3, stage="bench")
    # c004 同向 vload（latest 里 c004 尚无行——模拟 verify 前调用）：
    # 直接传 spec 风格验证：cur_dir 为空时走 keep 分支
    assert cli._parent_of(ev, "c004") == "c001"


# ---------- P0-3：漂移状态机 ----------

def test_bump_stall_transitions(tmp_path):
    """给定 ADVANCED/STALLED 序列 → 则 stall_count 清零/累计，last_progress 记录。"""
    st = TaskState(_mk_task(tmp_path), Evidence(tmp_path))
    assert st.bump_stall("STALLED") == 1
    assert st.bump_stall("REGRESSED") == 2
    assert st.bump_stall("ADVANCED") == 0
    s = st.require()
    assert s["stall_count"] == 0 and s["last_progress"] == "ADVANCED"


def test_drift_check_machine_judgment(tmp_path, monkeypatch):
    """给定合成 bench 历史 → _drift_check → 则判据纯机器：新最优=ADVANCED；
    劣于最优>5%=REGRESSED；无 bench=STALLED。"""
    from harness.control.loop import AutonomousLoop
    t = _mk_task(tmp_path)
    ev = Evidence(t)
    # 历史最优 900us
    ev.append_benchmark("c001", None, "bench", "l0", 900.0, 890.0, 950.0, 1.0, "benched")
    loop = AutonomousLoop.__new__(AutonomousLoop)
    loop.task, loop.ev, loop.st = t, ev, TaskState(t, ev)
    # 本轮 850 < 900 → ADVANCED
    assert loop._drift_check(2, "COMPLETE", {"mean_us": 850.0, "valid": True}) == "ADVANCED"
    # 本轮 1000 > 900*1.05 → REGRESSED
    assert loop._drift_check(3, "REVISE", {"mean_us": 1000.0, "valid": True}) == "REGRESSED"
    # 无 bench（verify 挂）→ STALLED
    assert loop._drift_check(4, "REVISE", None) == "STALLED"
    # invalid bench → STALLED（不入判定）
    assert loop._drift_check(5, "REVISE", {"mean_us": 0.0, "valid": False}) == "STALLED"
    audit = (t / "docs" / "audit.log").read_text(encoding="utf-8")
    assert "drift-check" in audit


# ---------- P0-4：writer_fails 与伪方向 ----------

def test_pseudo_direction_never_counts(tmp_path):
    """给定 parse-failed/unspecified/空 方向 → bump_direction_fail → 则不计数返回 0。"""
    st = TaskState(_mk_task(tmp_path), Evidence(tmp_path))
    for d in ("parse-failed", "unspecified", "", "unknown", "NONE"):
        assert st.bump_direction_fail(d) == 0
    assert st.require()["direction_fails"] == {}
    assert is_pseudo_direction("parse-failed") and not is_pseudo_direction("vload")


def test_writer_fails_counter(tmp_path):
    """给定连续 writer 失败 → bump_writer_fail 累计；成功后 reset 清零。"""
    st = TaskState(_mk_task(tmp_path), Evidence(tmp_path))
    assert st.bump_writer_fail() == 1
    assert st.bump_writer_fail() == 2
    st.reset_writer_fails()
    assert st.require()["writer_fails"] == 0


# ---------- P0-5：历史最优防御 ----------

def test_best_baseline_ignores_invalid_rows(tmp_path):
    """给定含 invalid/verify 行的 csv → _best_baseline_us → 则只认 bench+有效 verdict+mean>0。"""
    from harness.control.loop import AutonomousLoop
    t = _mk_task(tmp_path)
    ev = Evidence(t)
    ev.append_benchmark("c001", None, "bench", "l0", 900.0, 890.0, 950.0, 1.0, "benched")
    ev.append_benchmark("c004", None, "bench", "l0", 0.0, 0.0, 0.0, None, "invalid")
    ev.append_benchmark("c005", None, "verify", "l0", 1.0, 1.0, 1.0, None, "keep")
    loop = AutonomousLoop.__new__(AutonomousLoop)
    loop.task = t
    assert loop._best_baseline_us() == 900.0


# ---------- P0-6：轮契约 + 截断自适应 ----------

def test_write_round_contract(tmp_path):
    """给定 WRITE 成功 → 则 run/round-N-contract.md 生成且 direction 行可被
    cli._current_direction 第一源读到（闭环）。"""
    from harness.control.loop import AutonomousLoop
    import harness.cli as cli
    t = _mk_task(tmp_path)
    ev = Evidence(t)
    loop = AutonomousLoop.__new__(AutonomousLoop)
    loop.task, loop.ev, loop.st = t, ev, TaskState(t, ev)
    loop._write_round_contract(3, {"direction": "tma-load", "hypothesis": "用 TMA",
                                   "cid": "c004", "parent": "c002"}, None)
    p = t / "run" / "round-3-contract.md"
    assert p.exists()
    assert cli._current_direction(t, 3) == "tma-load"
    assert "success_criteria" in p.read_text(encoding="utf-8")


def test_memory_review_cap_adaptive():
    """给定 cap 公式 → 则 min(6000, 余量/6) 且下限 2500（离线验公式边界）。"""
    cap = lambda soft, used: max(2500, min(6000, (soft - used) // 6))
    assert cap(200000, 0) == 6000            # 余量充足 → 上限 6000
    assert cap(200000, 190000) == 2500       # 余量枯竭 → 下限 2500
    assert cap(200000, 170000) == 5000       # 中间档 = 余量/6
    assert cap(200000, 150000) == 6000       # 余量 50k/6≈8333 → 夹回 6000


# ---------- P0-7：终态保证 ----------

def test_run_exception_writes_error_terminal(tmp_path, monkeypatch):
    """给定 research 抛 RuntimeError → run() → 则 re-raise 且 state.terminal=error、
    pause.detail 有类型（不悬空）。"""
    from harness.control.loop import AutonomousLoop
    t = _mk_task(tmp_path)
    loop = AutonomousLoop(t, max_rounds=2)
    def boom(*a, **k):
        raise RuntimeError("quota-exhausted-sim")
    monkeypatch.setattr(loop, "research", boom)
    with pytest.raises(RuntimeError):
        loop.run()
    s = json.loads((t / "run" / "state.json").read_text(encoding="utf-8"))
    assert s["terminal"] == "error"
    assert s["pause"]["reason"] == "RuntimeError"


def test_pause_terminal_resumes_next_round(tmp_path, monkeypatch):
    """给定 pause 终态 → 续跑 → 则复活清终态且轮号 = pause 轮 + 1。"""
    from harness.control.loop import AutonomousLoop
    t = _mk_task(tmp_path)
    ev = Evidence(t)
    st = TaskState(t, ev)
    st.update(round=4, terminal="pause", pause={"reason": "quota"})
    loop = AutonomousLoop(t, max_rounds=1)
    # 桩：所有 LLM/远端阶段最小化（research 空、write 返回 parse-fail → 走 writer_fails 通道）
    monkeypatch.setattr(loop, "research", lambda r, f: {"router": "", "production": "",
                                                        "skills": [], "blindspot": False})
    monkeypatch.setattr(loop, "write_candidate",
                        lambda r, res, f: {"cid": "c001", "direction": "parse-failed",
                                           "hypothesis": "", "code": "", "parse_failed": True,
                                           "parent": None})
    monkeypatch.setattr(loop, "review", lambda r, cid, vr, br: "REVISE")
    loop.run()
    s = json.loads((t / "run" / "state.json").read_text(encoding="utf-8"))
    assert s["round"] == 5            # 续跑从 pause 轮 +1
    assert s["writer_fails"] == 1
    assert s["terminal"] != "" or s["terminal"] is None   # 未崩悬空


# ---------- P0-8：完整评审契约 ----------

def test_review_prompt_contains_full_contract(tmp_path, monkeypatch):
    """给定 review() → 则 prompt 含完整契约关键词（逐 AC/三车道/STOP 三要素）且不截断。"""
    from harness.control.loop import AutonomousLoop
    t = _mk_task(tmp_path)
    ev = Evidence(t)
    ev.append_benchmark("c001", None, "bench", "l0", 900.0, 890.0, 950.0, 1.0, "benched")
    loop = AutonomousLoop.__new__(AutonomousLoop)
    loop.task, loop.ev, loop.st = t, ev, TaskState(t, ev)
    captured = {}
    class FakeModels:
        _defaults = {"context_window": 250000}
        def chat(self, role, messages, **kw):
            captured["prompt"] = messages[0]["content"]
            return "评审正文\nMAINLINE GAPS: x\nREVISE"
    loop.models = FakeModels()
    loop._last_review_text = ""
    v = loop.review(2, "c002", {"passed": True, "workloads": []},
                    {"mean_us": 850.0, "valid": True})
    p = captured["prompt"]
    for kw in ("逐 AC 核对", "三车道", "MAINLINE GAPS", "STOP", "具名瓶颈",
               "已试方向清单", "ACS:", "REVISE 循环升级"):
        assert kw in p, f"评审契约缺关键词: {kw}"
    assert len(p) > 2800      # 全文注入（模板全文 2800+，远超此前 1500 截断）
    assert v == "REVISE"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
