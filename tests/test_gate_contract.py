"""tests/test_gate_contract.py —— gate 评审契约解析单测（无卡可跑，Phase 1 gate.py 的前置规格测试）。

规格来源：docs/design/interaction-protocol.md §4.3。
先测规格（畸形输出必须判解析失败），Phase 1 的 gate.py 实现必须通过同一组测试。
"""
from __future__ import annotations

import re

VERDICT_RE = re.compile(r"^MAINLINE_VERDICT:\s*(ADVANCED|STALLED|REGRESSED)\s*$", re.M)
ACS_RE = re.compile(r"^ACS:\s*\d+/\d+\s*\|\s*FORGOTTEN:\s*\d+\s*\|\s*UNJUSTIFIED_DEFERRALS:\s*\d+\s*$", re.M)


def parse_review(text: str) -> dict:
    """规格参考实现：gate.py 必须等价或更严。"""
    verdict = VERDICT_RE.findall(text)
    if len(verdict) != 1:
        raise ValueError("MAINLINE_VERDICT 必须恰好一行")
    if not ACS_RE.search(text):
        raise ValueError("ACS 统计行缺失或格式错")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    body, last = lines[:-1], lines[-1]
    if last not in ("COMPLETE", "REVISE", "REJECT"):
        raise ValueError("末行必须是 COMPLETE/REVISE/REJECT")
    if any(l == "COMPLETE" for l in body):
        raise ValueError("COMPLETE 只允许出现在最后一行")
    return {"verdict": verdict[0], "terminal": last}


def test_valid_revise():
    r = parse_review(
        "AC-1: MET …\nAC-2: PARTIAL …\nMAINLINE_VERDICT: ADVANCED\n"
        "ACS: 1/2 | FORGOTTEN: 0 | UNJUSTIFIED_DEFERRALS: 0\nREVISE")
    assert r["verdict"] == "ADVANCED" and r["terminal"] == "REVISE"


def test_valid_complete():
    r = parse_review(
        "AC-1: MET\nMAINLINE_VERDICT: ADVANCED\n"
        "ACS: 2/2 | FORGOTTEN: 0 | UNJUSTIFIED_DEFERRALS: 0\nCOMPLETE")
    assert r["terminal"] == "COMPLETE"


def test_missing_verdict_fails():
    import pytest
    with pytest.raises(ValueError):
        parse_review("AC-1: MET\nACS: 1/1 | FORGOTTEN: 0 | UNJUSTIFIED_DEFERRALS: 0\nREVISE")


def test_missing_acs_fails():
    import pytest
    with pytest.raises(ValueError):
        parse_review("MAINLINE_VERDICT: ADVANCED\nREVISE")


def test_complete_not_last_fails():
    import pytest
    with pytest.raises(ValueError):
        parse_review("COMPLETE\nMAINLINE_VERDICT: ADVANCED\n"
                     "ACS: 2/2 | FORGOTTEN: 0 | UNJUSTIFIED_DEFERRALS: 0\nREVISE")


def test_two_verdict_lines_fail():
    import pytest
    with pytest.raises(ValueError):
        parse_review("MAINLINE_VERDICT: ADVANCED\nMAINLINE_VERDICT: STALLED\n"
                     "ACS: 1/1 | FORGOTTEN: 0 | UNJUSTIFIED_DEFERRALS: 0\nREVISE")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
