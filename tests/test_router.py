"""tests/test_router.py —— 知识路由单测（无卡可跑）。

覆盖三轴检索的契约语义：
- 症状+族+代际命中排序；
- 未知症状必须走"无命中"（退出码 1，触发盲区条款）——反编造防线；
- fundamental 条目不响应症状查询（只走 L0 注入/精确 id）。
"""
import subprocess
import sys
from pathlib import Path

import pytest

QUERY = Path(__file__).resolve().parent.parent / "knowledge" / "router" / "query.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(QUERY), *args, "--compact"],
                          capture_output=True, text=True, timeout=30)


def test_symptom_family_arch_hit():
    r = run("--symptom", "mte2-bound", "--op-family", "norm", "--arch", "dav_2201")
    assert r.returncode == 0
    assert "norm-family-opt" in r.stdout          # 精确命中排首
    assert "bound-quickref" in r.stdout


def test_unknown_symptom_is_blindspot():
    r = run("--symptom", "conv-nonexistent")
    assert r.returncode == 1                       # 盲区路径：不许通配兜底
    assert "NO MATCH" in r.stdout


def test_fundamental_not_matched_by_symptom():
    r = run("--symptom", "mte2-bound", "--op-family", "norm")
    assert r.returncode == 0
    assert "triton-ascend-api-rules" not in r.stdout   # fundamental 不进症状检索


def test_exact_entry_lookup():
    r = run("--entry", "norm-family-opt")
    assert r.returncode == 0
    assert "ascendc-performance-best-practices" in r.stdout


def test_arch_filter_excludes():
    r = run("--symptom", "memory-bound", "--op-family", "reduce", "--arch", "dav_3510")
    # triton-ascend 条目 arch=dav_2201，应被滤掉；norm-family-opt(both) 应在
    assert r.returncode == 0
    assert "triton-ascend-reduce-guide" not in r.stdout
    assert "norm-family-opt" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
