"""tests/test_assets.py —— vendored 资产完整性断言（无卡可跑；防断链复发的 CI 防线）。

覆盖：
- index.yaml 每条 entry 的 path 在仓内存在（ADR-008：迁移自包含）；
- vendor-manifest.yaml 每条 asset 的 path 存在；
- manifest 资产与 knowledge/skills/ 目录双向一致（无孤儿、无漏登）；
- 第三方钉版标记存在；
- akg 主力资产实况（fundamentals/guides/cases/examples 数量）与声明一致。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "knowledge" / "router"))
from query import load_index  # noqa: E402


def _load_yaml(p: Path):
    import yaml
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_index_entries_paths_exist():
    entries = load_index()
    assert len(entries) >= 20, "index v2 应有 ≥20 条（当前含补件）"
    broken = [e["id"] for e in entries if not (ROOT / e["ref"]).exists()]   # ref 均为仓库相对路径
    assert not broken, f"断链条目: {broken}"


def test_manifest_assets_exist():
    data = _load_yaml(ROOT / "deps" / "vendor-manifest.yaml")
    missing = [a["name"] for a in data["assets"] if not (ROOT / a["path"]).exists()]
    assert not missing, f"manifest 断链: {missing}"
    assert len(data["assets"]) >= 70


def test_manifest_dir_bidirectional():
    """manifest 资产 ↔ knowledge/skills 实际目录 双向一致。"""
    data = _load_yaml(ROOT / "deps" / "vendor-manifest.yaml")
    manifest_paths = {a["path"].removeprefix("knowledge/skills/") for a in data["assets"]
                      if a["path"].startswith("knowledge/skills/")}
    actual = set()
    for group_dir in (ROOT / "knowledge" / "skills").iterdir():
        if group_dir.is_dir():
            for d in group_dir.iterdir():
                if d.is_dir():
                    actual.add(f"{group_dir.name}/{d.name}")
    assert manifest_paths == actual, (
        f"manifest-only: {sorted(manifest_paths - actual)[:5]} | dir-only: {sorted(actual - manifest_paths)[:5]}")


def test_akg_pinned_marker():
    assert (ROOT / "third_party" / "akg" / "PINNED_COMMIT").read_text().strip() == "5aa15f3"


def test_akg_triton_ascend_inventory():
    """akg 主力子树实况：fundamentals 7 / guides 5 / cases 22 / examples 6（index 注释与 ADR 声明以此为准）。"""
    base = ROOT / "knowledge" / "skills" / "akg" / "triton-ascend"
    assert len(list((base / "fundamentals").iterdir())) == 7
    assert len(list((base / "guides").iterdir())) == 5
    assert len(list((base / "cases").iterdir())) == 22
    assert len(list((base / "examples").iterdir())) == 6


def test_pilot_critical_paths():
    """试点关键资产的抽查（RMSNorm 闭环四件）。"""
    for p in ("knowledge/skills/core/ascendc-performance-best-practices/references/reduce",
              "knowledge/skills/akg/triton-ascend/examples/triton-ascend-example-layernorm",
              "knowledge/skills/akg/triton-ascend/fundamentals/triton-ascend-api-rules/SKILL.md",
              "knowledge/skills/core/ops-precision-standard/SKILL.md"):
        assert (ROOT / p).exists(), p


CANN_OPS = ROOT / "third_party" / "cann-ops"
needs_cann_ops = pytest.mark.skipif(
    not CANN_OPS.exists(),
    reason="重资产 cann-ops 未 bootstrap（python tools/sync_assets.py --bootstrap-cann-ops）——迁移新机器后跑一次即可")


@needs_cann_ops
def test_production_index_exists_and_valid():
    """生产代码层（ADR-009）：索引存在、条目 ≥1000、抽查路径存在、rms_norm 命中。"""
    import yaml
    pidx = ROOT / "knowledge" / "router" / "production-index.yaml"
    assert pidx.exists(), "production-index.yaml 缺失（跑 tools/build_production_index.py）"
    data = yaml.safe_load(pidx.read_text(encoding="utf-8"))
    entries = data["entries"]
    assert len(entries) >= 1000, f"生产条目应 ≥1000，当前 {len(entries)}"
    # 抽查 20 条路径存在（含首个/末个/随机）
    import random
    sample = entries[:5] + entries[-5:] + random.sample(entries, 10)
    broken = [e["op"] for e in sample if not (ROOT / e["path"]).exists()]
    assert not broken, f"生产索引断链: {broken}"
    # 试点关键命中
    rms = [e for e in entries if e["op"] in ("rms_norm", "add_rms_norm", "layer_norm")]
    assert rms, "rms_norm/layer_norm 生产实现未命中"


@needs_cann_ops
def test_production_joint_query():
    """query.py --production 联合检索：词元精确匹配（norm 族命中，无 random/abs 误入）。"""
    import subprocess, sys
    r = subprocess.run([sys.executable, str(ROOT / "knowledge" / "router" / "query.py"),
                        "--op-family", "norm", "--production", "--compact"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    prod_lines = [l for l in r.stdout.splitlines() if "prod:" in l]
    assert prod_lines, "生产条目未出现在联合检索"
    assert any("add_rms_norm" in l for l in prod_lines), "add_rms_norm 未命中"
    for l in prod_lines:
        assert "random" not in l.split("prod:")[1][:40], f"误命中: {l}"
    # 无 --production 时不出现
    r2 = subprocess.run([sys.executable, str(ROOT / "knowledge" / "router" / "query.py"),
                         "--op-family", "norm", "--compact"],
                        capture_output=True, text=True, timeout=60)
    assert "prod:" not in r2.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
