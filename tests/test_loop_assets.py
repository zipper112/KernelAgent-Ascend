"""tests/test_loop_assets.py —— 生产闭环纸面资产完整性断言（自审批次 B1）。

背景：审计发现"规格声称但文件不存在"的文档诈骗（phase 模板不知道 gate、评审 prompt 无模板、
round 模板缺失）。本文件把这些教训固化为断言——断链复发即测试红。
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "knowledge" / "prompts"


def test_gate_review_template_exists():
    p = PROMPTS / "gate-review.md"
    assert p.exists(), "gate 评审 prompt 模板缺失（渲染源）"
    text = p.read_text(encoding="utf-8")
    for token in ("MAINLINE_VERDICT", "ACS:", "COMPLETE", "REVISE", "REJECT", "STOP",
                  "FULL_ALIGNMENT", "MAINLINE GAPS"):
        assert token in text, f"gate-review.md 缺结构 token: {token}"


def test_round_templates_exist():
    for name, must in (("round-contract.md", ["direction", "budget", "成功标准"]),
                       ("round-summary.md", ["BitLesson Delta", "Todo 清点", "kda gate"])):
        p = PROMPTS / "templates" / name
        assert p.exists(), f"{name} 缺失"
        text = p.read_text(encoding="utf-8")
        for token in must:
            assert token in text, f"{name} 缺 {token}"


def test_phase_templates_know_gate_and_round_duties():
    """phase 模板必须告知 agent 逐轮产出义务与 kda gate 兜底（A3 教训）。"""
    for name in ("phase1-ascend.md", "phase2-ascend.md", "phase3-ascend.md"):
        text = (PROMPTS / name).read_text(encoding="utf-8")
        assert "kda gate" in text, f"{name} 未告知收工前调 kda gate（兜底链路断裂）"
        assert "round" in text.lower() and "contract" in text.lower() and "summary" in text.lower(), \
            f"{name} 未告知 round 产出义务"


def test_gen_plan_companion_exists():
    p = PROMPTS / "gen-plan-companion.md"
    assert p.exists(), "gen-plan 承载缺失（ADR-010）"
    text = p.read_text(encoding="utf-8")
    for token in ("CORE_RISKS", "CANDIDATE_CRITERIA", "超集", "Negative"):
        assert token in text


def test_lessons_pending_dir_exists():
    assert (ROOT / "knowledge" / "lessons" / "pending").is_dir()


def test_promote_eight_checks_consistent():
    """8 项门口径跨文档一致（D4 教训）。"""
    targets = [
        (ROOT / "harness" / "core" / "README.md", "8 项门"),
        (ROOT / "knowledge" / "prompts" / "contract-template.md", "8 项门"),
        (ROOT / "docs" / "design" / "interaction-protocol.md", "8 项门"),
    ]
    for path, token in targets:
        assert token in path.read_text(encoding="utf-8"), f"{path.name} 未同步 8 项门"
    assert "promote-eight-checks" in (ROOT / "tasks" / "_template" / "task.yaml").read_text(encoding="utf-8")


def test_hardcheck_count_consistent():
    """硬校验 12 项口径一致（v0.2 增预算自报前置；D4 教训）。"""
    proto = (ROOT / "docs" / "design" / "interaction-protocol.md").read_text(encoding="utf-8")
    ctrl = (ROOT / "harness" / "control" / "README.md").read_text(encoding="utf-8")
    assert "11 项" not in proto.split("硬校验链")[1][:200], "协议硬校验段应为 12 项（v0.2）"
    assert proto.count("12. **预算自报在案") == 1, "硬校验第 12 项（预算自报）缺失"
    # control README 允许滞后一个版本（Phase 1 C 批同步），但不得超前宣称
    assert "13 项" not in ctrl


def test_gate_stop_exit_code_defined():
    """STOP 退出码 3 定版（v0.2：终局停机非失败）。"""
    proto = (ROOT / "docs" / "design" / "interaction-protocol.md").read_text(encoding="utf-8")
    assert "`3` = STOP 终局停机" in proto and "3 STOP（终局停机" in proto


def test_gitignore_reinclude_semantics():
    """断点/轮次/摘要入仓 + 大文件排除（A8 教训，git check-ignore 实测）。"""
    import subprocess
    tracked = ["tasks/x/run/state.json", "tasks/x/run/round-3-summary.md",
               "tasks/x/profile/r1/summary.json", "tasks/_template/run/.gitkeep"]
    ignored = ["tasks/x/run/big.dump", "tasks/x/profile/r1/msprof.csv.big", "tasks/x/profile/r1/sub/raw.db"]
    for p in tracked:
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT, capture_output=True)
        assert r.returncode != 0, f"{p} 应入仓却被忽略"
    for p in ignored:
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT, capture_output=True)
        assert r.returncode == 0, f"{p} 应被忽略却会入仓"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
